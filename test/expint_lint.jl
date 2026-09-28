# Experiment-table bounds lint — every `_EXPINT[k]` / `_EXPFLAG[_EXPk]` index must be inside the
# array's declared size.  (standalone: `julia test/expint_lint.jl`)
#
# WHY THIS EXISTS. On 2026-08-18 commit `bf44b8b` added a reader at `_EXPINT[7]` while the array was
# still `fill(0, 6)`. EVERY consumer reads these tables `@inbounds`, so that shipped an out-of-bounds
# read in the complex-gemm dispatch — hot path, bounds check compiled away. NOTHING FAILED: `@inbounds`
# deletes the check, and whatever stale heap value sat past the end read as `0`, which is exactly what
# "knob off" looks like. It surfaced only because a probe wrote the same index, and probes are not
# `@inbounds`.
#
# The mechanism was a REVERT: an earlier commit grew the array to 8, a revert undid that growth, and a
# later commit re-added only the CONSUMER. A revert silently removed the precondition of a change that
# had not been written yet. No reviewer sees that, and no runtime test can — hence a lint.
#
# These tables are the project's standard A/B mechanism (cross-run comparison is not adjudicable, so
# knobs must be switchable in ONE process), so new indices get added often and this trap is live every
# time one is.

const _EXPLINT_SRC = joinpath(@__DIR__, "..", "src")

_explint_files() = [joinpath(r, f) for (r, _, fs) in walkdir(_EXPLINT_SRC) for f in fs if endswith(f, ".jl")]

"Declared length from `const <name> = fill(<init>, N)`, or `nothing`."
function explint_declared(name::AbstractString)
    re = Regex("const\\s+$(name)\\s*=\\s*fill\\([^,]+,\\s*(\\d+)\\s*\\)")
    for f in _explint_files(), ln in eachline(f)
        m = match(re, ln)
        isnothing(m) || return parse(Int, m.captures[1])
    end
    return nothing
end

"Literal indices used against `pat` (a capture-1 regex) across src/."
function explint_used(pat::Regex)
    idx = Set{Int}()
    for f in _explint_files(), ln in eachline(f)
        for m in eachmatch(pat, ln)
            push!(idx, parse(Int, m.captures[1]))
        end
    end
    return idx
end

"""
Slots the REGISTRY comments describe as free, as `slot => the line that says so`.

A registry line is `#   _EXPn  <description>` (or `_EXPINT[n]`), and it is read as an offer of a free
slot when the description opens with the word "free". Ranges like `_EXPINT[2..4]  free` count for
every slot they name.
"""
function explint_claimed_free(name::AbstractString)
    free = Dict{Int, String}()
    re = name == "_EXPINT" ?
        r"^#\s*_EXPINT\[(\d+)(?:\.\.(\d+))?\]\s+(\w+)" :
        r"^#\s*_EXP(\d+)\s+(\w+)"
    for f in _explint_files(), ln in eachline(f)
        m = match(re, ln)
        isnothing(m) && continue
        word = lowercase(m.captures[end])
        word == "free" || continue
        lo = parse(Int, m.captures[1])
        hi = (name == "_EXPINT" && !isnothing(m.captures[2])) ? parse(Int, m.captures[2]) : lo
        for k in lo:hi
            free[k] = strip(ln)
        end
    end
    return free
end

"Returns a vector of human-readable violations; empty means clean."
function expint_scan()
    v = String[]
    for (name, pat, initfmt) in (
            ("_EXPINT", r"_EXPINT\[(\d+)\]", "fill(0, N)"),
            ("_EXPFLAG", r"_EXPFLAG\[_EXP(\d+)\]", "fill(false, N)"),
        )
        n = explint_declared(name)
        if isnothing(n)
            push!(v, "$name: no `const $name = $initfmt` declaration found in src/")
            continue
        end
        over = sort([k for k in explint_used(pat) if k > n])
        isempty(over) || push!(
            v, "$name declared with length $n but src/ indexes $(over). Every read is @inbounds, so " *
                "this is an OUT-OF-BOUNDS READ rather than a bounds error. Grow the array IN THE SAME " *
                "COMMIT as the new index."
        )
        # A SLOT THE REGISTRY CALLS FREE WHILE src/ READS IT. The bounds check above cannot see this:
        # the index is in range and the collision is silent, because the new experiment and the
        # shipping arm it lands on share one boolean. Three slots were in that state at once
        # (_EXP9 gating trmm's packed path, _EXP10 the side-R pad, _EXP11 complex rank-k 3M) — each
        # entry said "free again" from an earlier campaign and the later reuse never updated it, so
        # an A/B built on that word silently rerouted an unrelated routine mid-measurement.
        used = explint_used(pat)
        for (k, line) in sort!(collect(explint_claimed_free(name)); by = first)
            k in used && push!(
                v, "$name slot $k: the registry calls it FREE but src/ reads it. A claim on that " *
                    "word collides with a shipping arm instead of conflicting visibly — correct the " *
                    "registry line, or claim a different slot.\n      registry says: $line"
            )
        end
    end
    return v
end

if abspath(PROGRAM_FILE) == @__FILE__
    viol = expint_scan()
    if isempty(viol)
        println("expint lint: clean")
    else
        foreach(x -> println("VIOLATION: ", x), viol)
        exit(1)
    end
end
