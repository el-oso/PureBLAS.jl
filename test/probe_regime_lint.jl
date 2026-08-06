# PROBE-REGIME lint — every probe must DECLARE the regime it measures in.
#
# WHY A GATE AND NOT A HABIT. A probe's verdict is valid only in the regime it probed; consulting it
# from another regime is a silent bug — the code is right, the measurement is honest, and the answer
# is wrong. This has now recurred FIVE times (see memory `probe-regime-must-match-live`), and a
# finding written after the first did not stop the next four. Documentation that must be remembered is
# not a control.
#
# THE FIFTH INSTANCE WIDENED WHAT "REGIME" MEANS, which is why this lint checks two axes:
#   * RESIDENCY — L1/L2/L3/DRAM, and whether the operand was freshly written by a preceding kernel.
#   * CALL STRUCTURE — one call per sample vs a rep loop; fresh vs persistent operands. On 2026-08-06 a
#     `scal` ladder probe timing ONE call per `@be` sample reported the public entry frame at +0.1 ns
#     and ranked the kernel 1.0085 vs OpenBLAS, while the gate read that same cell at 0.985. The gate
#     times `@be mk(s) (c -> w(c, reps))` — a FRESH pair per sample and `reps` DEPENDENT passes inside
#     one closure (800 at n=1e4). An ~11 ns per-call frame is 1.2% inside a rep loop and invisible
#     amortised over one call. Rebuilding the probe around the live shape found it immediately.
#
# WHAT THIS ENFORCES (deliberately cheap and mechanical): every `bench/probes/*.jl` carries a
# `# REGIME:` line, or a header that names both axes. It cannot check that the declaration is TRUE —
# only that the author was made to state it, which is the point at which the mismatch becomes visible.
#
# Escape hatch: `# regime-ok: <reason>` for a file that genuinely has no regime (a correctness probe,
# a code-dump helper). `probe_regime_baseline.txt` carries the probes that predate the lint, so a NEW
# undeclared probe fails immediately — the req8_lint_baseline.txt pattern.
#
# Run standalone:  julia test/probe_regime_lint.jl

const _PROBES = joinpath(@__DIR__, "..", "bench", "probes")
const _PR_BASELINE = joinpath(@__DIR__, "probe_regime_baseline.txt")
const _REGIME = r"#\s*REGIME:"i
const _REGIME_OK = r"#\s*regime-ok:"i
# A header that names both axes counts as a declaration even without the literal tag — these are the
# words that actually appear when someone has thought about it.
const _RESIDENCY = r"\b(L1|L2|L3|DRAM|resident|residency|cache-resident|working set)\b"i
const _CALLSHAPE = r"\b(rep loop|rep-loop|reps|per sample|per-sample|one call|single call|call structure|live regime|fresh arrays|persistent array)\b"i

"""
    probe_regime_scan() -> Vector{String}

Probes under `bench/probes/` that declare no regime.
"""
function probe_regime_scan()
    isdir(_PROBES) || return String[]
    base = isfile(_PR_BASELINE) ?
        Set(filter(l -> !isempty(l) && !startswith(l, "#"), strip.(readlines(_PR_BASELINE)))) : Set{String}()
    bad = String[]
    for f in sort!(readdir(_PROBES))
        endswith(f, ".jl") || continue
        f in base && continue
        path = joinpath(_PROBES, f)
        src = read(path, String)
        occursin(_REGIME_OK, src) && continue
        occursin(_REGIME, src) && continue
        # accept a header that names BOTH axes explicitly
        (occursin(_RESIDENCY, src) && occursin(_CALLSHAPE, src)) && continue
        push!(bad, "bench/probes/$f  declares no regime (need a `# REGIME:` line naming residency AND " *
                   "call structure, or `# regime-ok: <reason>`)")
    end
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    v = probe_regime_scan()
    if isempty(v)
        println("probe-regime lint: PASS (every probe declares its regime)")
    else
        println("probe-regime lint: FAIL — $(length(v)) probe(s) without a declared regime:")
        foreach(x -> println("  ", x), v)
        exit(1)
    end
end
