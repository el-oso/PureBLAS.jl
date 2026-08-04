# Chairmarks lint — ONLY Chairmarks may time anything under bench/. No exemptions, present or future.
#
# There is no escape hatch in this file BY DESIGN. Earlier versions had two (`# estimator-ok:` per line,
# `# estimator-ok-file:` per file) plus a filename baseline, and every one of them was used by me to keep
# a raw clock alive behind a plausible-sounding reason ("only a cliff detector", "only a GC fraction",
# "only phase decomposition"). That is the same reasoning that let `minimum(@elapsed …)` into the iamax
# probes, where it inverted an unroll ranking and cost a day; and again into a hand-rolled `Measure.ab`,
# where it flipped the sign of the axpy result (0.991 [0.941, 1.076] "falsified" vs Chairmarks
# 1.022 [1.005, 1.030], a real 2.2% win). A rule with an exemption mechanism is a rule I will talk my
# way around. So: no mechanism.
#
# Banned everywhere under bench/: @elapsed, time_ns, @btime, @benchmark, and reducing timings with
# min/mean. Use Chairmarks `@be` and the median (bench/measure.jl `tstat`).
#
# Run standalone: julia test/estimator_lint.jl
const _BENCHDIRS = [joinpath(@__DIR__, "..", "bench"), joinpath(@__DIR__, "..", "bench", "probes")]
const _RAWTIME = r"@elapsed|\btime_ns\s*\(|@btime|@benchmark"
# Unapproved reductions OF CHAIRMARKS SAMPLES. Raw clocks are already banned above, so this only has to
# catch `min`/`mean` applied to a sample vector — deliberately anchored on `.samples` / `.time` so a
# `minimum` over non-timing data (e.g. date strings in plots.jl's provenance line) is not a false hit.
const _BADRED = [
    r"\b(minimum|mean)\s*\([^)]*\.samples"i,
    r"\b(minimum|mean)\s*\([^)]*s\.time"i,
    r"\b(minimum|mean)\s*\(\s*(ts|times|timings)\s*\)"i,
]
const _BENCHPRIM = r"@be\b|Chairmarks"

function estimator_scan()
    viols = String[]
    for dir in _BENCHDIRS, f in (isdir(dir) ? sort(readdir(dir; join = true)) : String[])
        endswith(f, ".jl") || continue
        for (i, ln) in enumerate(readlines(f))
            code = split(ln, '#')[1]
            if occursin(_RAWTIME, code)
                push!(viols, string(basename(f), ":", i, "  raw clock — Chairmarks only: ", strip(code)))
                continue
            end
            for re in _BADRED
                occursin(re, code) || continue
                push!(viols, string(basename(f), ":", i, "  min/mean over samples — use the median: ",
                                    strip(code)))
                break
            end
        end
    end
    return viols
end

# The probe harness must actually use Chairmarks, so it cannot silently revert to a hand loop.
function harness_scan()
    mf = joinpath(@__DIR__, "..", "bench", "measure.jl")
    return (isfile(mf) && !occursin(_BENCHPRIM, read(mf, String))) ?
        ["measure.jl  does NOT use Chairmarks — the probe harness must not hand-roll timing"] : String[]
end

if abspath(PROGRAM_FILE) == @__FILE__
    v = vcat(estimator_scan(), harness_scan())
    if isempty(v)
        println("chairmarks lint: PASS (no raw clocks, no unapproved reductions, no exemptions)")
    else
        println("chairmarks lint: FAIL — $(length(v)) violation(s):")
        foreach(x -> println("  ", x), v)
        exit(1)
    end
end
