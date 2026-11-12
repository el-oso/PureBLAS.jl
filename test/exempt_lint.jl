# Exempt lint — no `PDM: Exempt` marker may appear in src/.
#
# The PDM ladder has three tiers: Derived, Measured, Literal (test/knob_registry.jl `_KR_TIERS`). A
# constant that seems to need a fourth, "not a knob", is one of these instead:
#   * a detected hardware fact (SIMD width, cache size, core count)  -> `PDM: Derived`, naming the detection;
#   * a label or tag                                                 -> an `@enum`, which needs no marker;
#   * a count of other things                                        -> computed from the list it counts;
#   * an on/off switch whose off path nothing runs                   -> deleted, with the off path;
#   * a capability the host may lack                                 -> detected, with a quiet fallback;
#   * a numeric "unset" sentinel                                     -> `nothing` as the preference default;
#   * a routing rule that is not a size                              -> written as the condition itself.
# An `Exempt` marker is how a magic number avoids all of those questions, so the marker is refused.
#
# Run standalone:  julia test/exempt_lint.jl

const _EX_SRC = joinpath(@__DIR__, "..", "src")

# Files that still carry the marker, path relative to src/ with `/` separators. A file listed here
# is skipped — and the lint FAILS once such a file is clean, so an entry cannot outlive its reason
# and the list can only shrink.
const _EX_PENDING = ()

const _EX_MARKER = r"\bPDM\b.*\bExempt\b"

"""
    exempt_scan() -> Vector{String}

One line per `PDM … Exempt` comment in src/ outside `_EX_PENDING`, plus one per `_EX_PENDING` entry
whose file no longer has any.
"""
function exempt_scan()
    hits = String[]
    seen = Set{String}()
    for (root, _, files) in walkdir(_EX_SRC), f in sort(files)
        endswith(f, ".jl") || continue
        rel = replace(relpath(joinpath(root, f), _EX_SRC), '\\' => '/')
        pending = rel in _EX_PENDING
        n = 0
        for (i, ln) in enumerate(eachline(joinpath(root, f)))
            occursin(_EX_MARKER, ln) || continue
            n += 1
            pending || push!(hits, "src/$rel:$i  $(strip(ln))")
        end
        pending && push!(seen, rel)
        pending && n == 0 &&
            push!(hits, "test/exempt_lint.jl  `_EX_PENDING` lists $rel, which is clean — remove the entry")
    end
    for rel in _EX_PENDING
        rel in seen || push!(hits, "test/exempt_lint.jl  `_EX_PENDING` lists $rel, which does not exist — remove the entry")
    end
    return sort!(hits)
end

if abspath(PROGRAM_FILE) == @__FILE__
    hits = exempt_scan()
    if isempty(hits)
        println("exempt lint: PASS (no `PDM: Exempt` marker in src/",
            isempty(_EX_PENDING) ? ")" : "; pending: $(join(_EX_PENDING, ", ")))")
    else
        println("exempt lint: FAIL — the PDM ladder has no Exempt tier (see the header of this file):")
        foreach(h -> println("  ", h), hits)
        exit(1)
    end
end
