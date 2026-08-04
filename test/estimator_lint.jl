# Estimator lint — catches a benchmark reducing TIMINGS with an unapproved statistic.
#
# Modelled on req8_lint.jl, and for the same reason: the trap is invisible in review. CLAUDE.md says
# "**median** times (not min)", plots.jl obeys it, and on 2026-08-03/04 the probe harnesses still used
# `minimum(@elapsed …)` while an early gate_verdict.jl used `mean` of log-ratios. The estimator swap does
# not error — it silently produces confident wrong answers. Here it ranked an iamax unroll the wrong way
# at n=1e6 and cost a day of chasing the resulting contradiction.
#
# Flagged: `minimum(`/`min(`/`mean(` applied to something that looks like timings — an `@elapsed`
# comprehension, or a variable named t/ts/times/elapsed/… — anywhere under bench/.
# Escape hatch: `# estimator-ok: <reason>` on the same line or the line above, naming why this particular
# reduction is not the gate statistic (e.g. a spread diagnostic, which legitimately wants min AND p90).
#
# Run standalone: julia test/estimator_lint.jl
const _BENCHDIRS = [joinpath(@__DIR__, "..", "bench"), joinpath(@__DIR__, "..", "bench", "probes")]
const _EANNOT = r"#\s*estimator-ok:"i
# Raw timing primitives, banned OUTRIGHT inside bench/probes/ — see the check in estimator_scan.
const _RAWTIME = r"@elapsed|\btime_ns\s*\(|@btime|@benchmark"
const _PROBEDIR = "probes"
# min/minimum/mean over an @elapsed comprehension, or over a timing-ish name.
const _BAD = [
    r"\b(minimum|mean)\s*\(\s*@elapsed"i,
    r"\b(minimum|mean)\s*\(\s*\[?\s*@elapsed"i,
    r"\b(minimum|mean)\s*\(\s*(ts|t|times|elapsed|timings|samples)\b"i,
    r"\bmin\s*\(\s*(ts|t|times|elapsed|timings)\s*\["i,
]

function estimator_scan()
    viols = String[]
    # bench/probes/ is scanned too, and that is the WHOLE POINT: the 2026-08-03/04 violations were in
    # throwaway probes under /tmp, invisible to any repo lint. Probes now live in-repo (gitignored
    # contents) so the scan reaches the place the mistake actually happens.
    for dir in _BENCHDIRS, f in (isdir(dir) ? sort(readdir(dir; join = true)) : String[])
        endswith(f, ".jl") || continue
        lines = readlines(f)
        # File-level exemption for scripts that time PHASES INSIDE one routine (where does the time go?)
        # rather than forming a PB-vs-reference ratio. Those legitimately need a raw clock and produce no
        # gate number. Must state why, in the first 20 lines.
        any(l -> occursin(r"#\s*estimator-ok-file:"i, l), first(lines, min(20, length(lines)))) && continue
        for (i, ln) in enumerate(lines)
            (i > 1 && occursin(_EANNOT, lines[i - 1])) && continue
            occursin(_EANNOT, ln) && continue
            code = split(ln, '#')[1]
            # In a PROBE, the raw timing primitive itself is banned — not just the bad reduction. A
            # hand-rolled loop re-opens five decisions at once (estimator, rounds, warm-up, arm order,
            # fresh-vs-warm inputs), and on 2026-08-03/04 every one of them was got wrong in a bespoke
            # probe. `Measure.ab` makes those decisions once. Removing the opportunity beats detecting
            # the mistake.
            # Applies to ALL of bench/, not just probes. `measure.jl` itself was written with a
            # hand-rolled `time_ns()` loop one message after the anti-ad-hoc tooling was built — the
            # approved benchmark driver is Chairmarks `@be`, as plots.jl uses. A hand loop also yields
            # ONE timing per window where @be yields hundreds, which needlessly widened every interval.
            if occursin(_RAWTIME, code)
                push!(viols, string(basename(f), ":", i,
                                    "  raw timing primitive in a probe — use Measure.ab: ", strip(code)))
                continue
            end
            for re in _BAD
                if occursin(re, code)
                    push!(viols, "$(basename(f)):$i  $(strip(code))")
                    break
                end
            end
        end
    end
    return filter(v -> first(split(v, ":")) ∉ _baseline(), viols)
end

# One baseline, shared by both checks, keyed on FILENAME so unrelated line shifts do not defeat it.
# Pre-existing ad-hoc probes are debt; a NEW violation is caught immediately. Note what is in here:
# `zgeqrf_gate.jl` forms a GATE ratio from raw clocks, which is exactly the pattern that produced wrong
# conclusions on 2026-08-03/04 — porting it to Measure.ab should come before trusting its numbers again.
function _baseline()
    base = Set{String}()
    bf = joinpath(@__DIR__, "harness_baseline.txt")
    isfile(bf) && for l in eachline(bf)
        (isempty(strip(l)) || startswith(l, "#")) && continue
        push!(base, strip(l))
    end
    return base
end

# ── Harness monopoly ─────────────────────────────────────────────────────────────────────────────────
# ONLY these two files may drive a benchmark. plots.jl is the gate harness; measure.jl is the probe
# harness. Everything else calls `Measure.ab`.
#
# This is the rule that actually stops "I'll just write a quick timing loop". Banning raw clocks was not
# enough: on 2026-08-04 I hand-rolled `Measure.ab` with `time_ns()` ONE MESSAGE after building the
# anti-ad-hoc tooling, and it took one timing per window where Chairmarks takes hundreds. At axpy n=1e6
# that did not merely widen the interval — it FLIPPED THE SIGN: hand loop 0.991 [0.941, 1.076] ("ivdep is
# slower, falsified"), Chairmarks 1.022 [1.005, 1.030] (ivdep is 2.2% FASTER, decisive). A real win was
# invisible because the harness was mine instead of the proven one. Same failure as substituting `min`
# for `median`, one layer up.
#
# Writing a new harness requires calling `@be` in a new file — which now fails the build.
const _HARNESS_OK = ("plots.jl", "measure.jl")
const _BENCHPRIM = r"@be\b|Chairmarks"

function harness_scan()
    viols = String[]
    for dir in _BENCHDIRS, f in (isdir(dir) ? sort(readdir(dir; join = true)) : String[])
        endswith(f, ".jl") || continue
        b = basename(f)
        lines = readlines(f)
        any(l -> occursin(r"#\s*estimator-ok-file:"i, l), first(lines, min(20, length(lines)))) && continue
        if b ∉ _HARNESS_OK
            for (i, ln) in enumerate(lines)
                occursin(_BENCHPRIM, split(ln, '#')[1]) || continue
                push!(viols, string(b, ":", i, "  drives a benchmark directly — call Measure.ab instead"))
                break
            end
        end
    end
    # Positive check: the probe harness must actually USE Chairmarks, not a hand loop.
    mf = joinpath(@__DIR__, "..", "bench", "measure.jl")
    if isfile(mf) && !occursin(_BENCHPRIM, read(mf, String))
        push!(viols, "measure.jl  does NOT use Chairmarks — the probe harness must not hand-roll timing")
    end
    # Baseline, exactly as req8_lint does it: 21 pre-existing ad-hoc probes drive Chairmarks directly.
    # They are debt, not a reason to fail the build — but a NEW one is caught immediately, which is the
    # whole point. Match on filename so unrelated line shifts don't defeat it. Remove a line from the
    # baseline once that probe is ported to Measure.ab.
    return filter(v -> first(split(v, ":")) ∉ _baseline(), viols)
end

if abspath(PROGRAM_FILE) == @__FILE__
    v = vcat(estimator_scan(), harness_scan())
    if isempty(v)
        println("estimator lint: PASS (approved statistic, and only plots.jl/measure.jl drive benchmarks)")
    else
        println("estimator lint: FAIL — $(length(v)) violation(s):")
        foreach(x -> println("  ", x), v)
        exit(1)
    end
end
