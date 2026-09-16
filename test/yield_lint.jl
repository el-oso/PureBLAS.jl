# Yield lint — a task-switch point inside a kernel breaks per-thread scratch ownership, silently.
#
# WHY THIS FILE EXISTS. Every buffer a routine writes during a call is one per THREAD
# (`Base.OncePerThread`: `_arena()`, `_l3ws`, the trmm/symm/LU/QR/SVD/eigen/trsv owners). That is correct
# while a call holds its owner without ever yielding: Julia switches tasks only at a yield point, so no
# second task can run on this thread between a scope's enter and its exit. One `@warn` in a kernel, one
# `@spawn`/`wait` join inside a live scope, and two tasks share the thread's arena — arbitrary cross-role
# aliasing, wrong answers, no test failure.
#
# The check is lexical, like every lint here: it reads `src/` for the shapes that can yield. It cannot see
# a yield inside a callee in another package, which is why the design note in `src/arena.jl` names the
# per-task owner (`Base.OncePerTask`, one line) as the answer if a kernel ever has to yield.
#
# WHAT IS BASELINED. `tune.jl` (the calibration driver) and `lbt.jl` (activate/status printing) are not
# kernels and hold no scope; their entries live in `test/yield_lint_baseline.txt` with a reason, exactly
# as `fastpath_lint_baseline.txt` carries reasoned exceptions. A line not in the baseline is a new
# finding; a baseline line that no longer occurs is stale and also fails.

const _YL_SRC = normpath(joinpath(@__DIR__, "..", "src"))
const _YL_BASELINE = joinpath(@__DIR__, "yield_lint_baseline.txt")
# Task-switch points, and the printing macros that reach one through the IO lock.
const _YL_PAT = r"(@spawn|@async|@sync\b|\byield\(|\bwait\(|\bfetch\(|\bsleep\(|\block\(|@lock\b|Channel\{|Channel\(|@threads\b|@info\b|@warn\b|@error\b|@debug\b|\bprintln\(|\bprint\(|\bflush\()"
_yl_srcfiles() = sort!([joinpath(r, f) for (r, _, fs) in walkdir(_YL_SRC) for f in fs if endswith(f, ".jl")])

"""
    yield_scan() -> Vector{String}

Every task-switch point in `src/`, as `relative/path.jl:<code>` keys. Line numbers are deliberately NOT
part of the key — an edit above a site must not invalidate its baseline entry.
"""
function yield_scan()
    hits = String[]
    for f in _yl_srcfiles()
        rel = relpath(f, _YL_SRC)
        for ln in eachline(f)
            code = first(split(ln, '#'; limit = 2))      # a mention in a comment is documentation
            occursin(_YL_PAT, code) || continue
            push!(hits, string(rel, ":", strip(code)))
        end
    end
    return hits
end

_yl_baseline() = isfile(_YL_BASELINE) ?
    Set(filter(l -> !isempty(l) && !startswith(l, "#"), strip.(readlines(_YL_BASELINE)))) : Set{String}()

"""
    yield_violations() -> (new = …, stale = …)

`new`: a task-switch point that is not in the reviewed baseline — remove it from the kernel, or baseline
it with a reason (and check no `@scope` is live across it). `stale`: a baselined line that no longer
occurs — delete the entry.
"""
function yield_violations()
    got = Set(yield_scan())
    base = _yl_baseline()
    return (new = sort!(collect(setdiff(got, base))), stale = sort!(collect(setdiff(base, got))))
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--baseline" in ARGS
        foreach(println, yield_scan())
    else
        r = yield_violations()
        if isempty(r.new) && isempty(r.stale)
            println("yield lint: PASS (no unreviewed task-switch points in src/)")
        else
            println("yield lint: FAIL")
            foreach(x -> println("  NEW   ", x), r.new)
            foreach(x -> println("  STALE ", x), r.stale)
            exit(1)
        end
    end
end
