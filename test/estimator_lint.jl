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

# ── src/ AUTO-TUNES ────────────────────────────────────────────────────────────────────────────────
# The PDM Measure tier puts a real timing loop INSIDE the package: `OncePerProcess` harnesses that pick
# a shipped kernel variant or block size per host (zaxpy narrow-vs-wide, ger stream count, pstrf batched
# swap, gebrd nb, …). Those cannot use Chairmarks — PureBLAS must not depend on it and must stay
# trim-safe — so `time_ns` is allowed there and ONLY there.
#
# The ESTIMATOR rule still applies, and it was being violated everywhere: on 2026-08-05 all nine of these
# harnesses reduced with `min(t, time_ns() - s)` over 3-5 samples. That is the exact combination that
# ranked an iamax unroll backwards and cost a day — except here it selects the code that SHIPS, on every
# user's machine, and the bench-only lint could never see it.
const _SRCDIRS = [joinpath(@__DIR__, "..", "src"), joinpath(@__DIR__, "..", "src", "blas1"),
                  joinpath(@__DIR__, "..", "src", "blas2"), joinpath(@__DIR__, "..", "src", "blas3"),
                  joinpath(@__DIR__, "..", "src", "lapack"), joinpath(@__DIR__, "..", "src", "cabi")]
# `min`/`minimum` applied to something that came out of a clock. Anchored on the timing idiom so an
# ordinary `min` over sizes or indices is not a false hit.
const _SRCBADRED = [
    r"\bmin\s*\([^)]*time_ns",           # min(t, time_ns() - s)
    r"=\s*min\s*\(\s*t[a-z]*\s*,\s*e\s*\)",   # tn = min(tn, e)  where e is an elapsed
]
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

"Scan src/ auto-tunes: `time_ns` is permitted (no Chairmarks in the package), `min` over it is not."
function src_estimator_scan()
    viols = String[]
    for dir in _SRCDIRS, f in (isdir(dir) ? sort(readdir(dir; join = true)) : String[])
        endswith(f, ".jl") || continue
        for (i, ln) in enumerate(readlines(f))
            code = split(ln, '#')[1]
            for re in _SRCBADRED
                occursin(re, code) || continue
                push!(viols, string("src/", basename(f), ":", i,
                                    "  auto-tune reduces timings with min — use the MEDIAN: ", strip(code)))
                break
            end
        end
    end
    return viols
end

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
    v = vcat(estimator_scan(), harness_scan(), src_estimator_scan())
    if isempty(v)
        println("chairmarks lint: PASS (no raw clocks, no unapproved reductions, no exemptions)")
    else
        println("chairmarks lint: FAIL — $(length(v)) violation(s):")
        foreach(x -> println("  ", x), v)
        exit(1)
    end
end
