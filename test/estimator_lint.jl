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
        for (i, ln) in enumerate(lines)
            (i > 1 && occursin(_EANNOT, lines[i - 1])) && continue
            occursin(_EANNOT, ln) && continue
            code = split(ln, '#')[1]
            for re in _BAD
                if occursin(re, code)
                    push!(viols, "$(basename(f)):$i  $(strip(code))")
                    break
                end
            end
        end
    end
    return viols
end

if abspath(PROGRAM_FILE) == @__FILE__
    v = estimator_scan()
    if isempty(v)
        println("estimator lint: PASS (no unapproved timing reductions under bench/)")
    else
        println("estimator lint: FAIL — $(length(v)) unapproved timing reduction(s):")
        foreach(x -> println("  ", x), v)
        exit(1)
    end
end
