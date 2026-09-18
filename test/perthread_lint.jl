# Per-thread owner lint — a `OncePerThread` buffer held across a threaded call returns WRONG ANSWERS.
#
# WHY THIS FILE EXISTS. `Base.OncePerThread` is only correct while the code holding the buffer stays on
# one thread. It does not. `gemm!`'s threaded driver yields in its join, and a yielded task usually
# resumes on a DIFFERENT thread — measured on Zen4, **7491 of 9600 yields (78%)**. So a routine that
# fetched thread t1's buffer typically resumes on t3 still using it, while t1's buffer is free for any
# other task to claim. That is not two tasks interleaving on one thread; it is two tasks on two threads
# writing one buffer at once.
#
# It is not theoretical. Before the fix, with 24 concurrent callers: `getrf!` **20/96 wrong (some NaN)**,
# `geqrf!` **17/96**, `symm!` **5/96**, against 0/96 serial for all three. The owners were `_LU_PAD`,
# `_QR_WS` and `_SYMM_SCR` — each held across a `gemm!` that threads.
#
# THE TWO SAFE SHAPES, and every baselined owner must be one of them:
#   (a) the buffer can never be live across a threaded call — the routine holding it never reaches the
#       public `gemm!`, or `gemm!` refuses to thread while it is held (`iszero(_arena().depth)`);
#   (b) the buffer is claimed INSIDE a chunk body, which never yields (`test/yield_lint.jl` is the
#       guard), so it is per-chunk in practice and cannot outlive a migration.
# Anything else must be `Base.OncePerTask`, which ties the buffer to the task — the thing that does not
# move — at a measured ~13 ns extra per lookup.
#
# The check is lexical, like every lint here: a NEW `OncePerThread` owner that is not in the reviewed
# baseline fails the suite, and a baseline entry that no longer exists is stale and also fails.

const _PT_SRC = normpath(joinpath(@__DIR__, "..", "src"))
const _PT_BASELINE = joinpath(@__DIR__, "perthread_lint_baseline.txt")
const _PT_PAT = r"^\s*const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*Base\.OncePerThread"
_pt_srcfiles() = sort!([joinpath(r, f) for (r, _, fs) in walkdir(_PT_SRC) for f in fs if endswith(f, ".jl")])

"""
    perthread_scan() -> Vector{String}

Every `OncePerThread` owner in `src/`, as `relative/path.jl:NAME` keys. The NAME, not the line, is the
key — moving a declaration must not invalidate its reviewed reason.
"""
function perthread_scan()
    hits = String[]
    for f in _pt_srcfiles()
        rel = relpath(f, _PT_SRC)
        for ln in eachline(f)
            m = match(_PT_PAT, first(split(ln, '#'; limit = 2)))
            isnothing(m) || push!(hits, string(rel, ":", m.captures[1]))
        end
    end
    return hits
end

_pt_baseline() = isfile(_PT_BASELINE) ?
    Set(filter(l -> !isempty(l) && !startswith(l, "#"), strip.(readlines(_PT_BASELINE)))) : Set{String}()

"""
    perthread_violations() -> (new = …, stale = …)

`new`: a `OncePerThread` owner nobody has reasoned about — make it `OncePerTask`, or baseline it with
the argument for why it can never be live across a threaded call. `stale`: a baselined owner that no
longer exists — delete the entry.
"""
function perthread_violations()
    got = Set(perthread_scan())
    base = _pt_baseline()
    return (new = sort!(collect(setdiff(got, base))), stale = sort!(collect(setdiff(base, got))))
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--baseline" in ARGS
        foreach(println, perthread_scan())
    else
        r = perthread_violations()
        if isempty(r.new) && isempty(r.stale)
            println("per-thread owner lint: PASS (every OncePerThread owner is reviewed)")
        else
            println("per-thread owner lint: FAIL")
            foreach(x -> println("  NEW   ", x, "  — make it OncePerTask, or baseline it with a reason"), r.new)
            foreach(x -> println("  STALE ", x), r.stale)
            exit(1)
        end
    end
end
