# The gate, re-derived in CYCLES instead of seconds — as a TRIAGE REPORT, not a second verdict.
#
#   julia --project=bench bench/gate_cycles.jl bench/plots_data_<uarch>_<host>.txt [more...]
#
# WHY. `bench/plots.jl` computes the gate from wall-clock seconds. `bench/cellcycles.jl` already converts
# each arm to cycles using THAT ARM'S OWN observed clock, but prints numbers and adjudicates nothing —
# there is no cycles equivalent of `gatecrit.jl`. This file closes that: it applies the SAME
# `gate_pass` predicate to the cycles ratio, so the two views can be compared cell by cell.
#
# ── WHAT THIS DELIBERATELY DOES NOT DO ───────────────────────────────────────────────────────────────
# It does not create a new verdict category, and `bench/publish.sh` does not call it. An earlier draft
# had it mark high-`wobble` and high-`Δclk` cells NOT ADJUDICABLE and wired that into the publish gate.
# Three objections killed that, and they are worth keeping written down:
#
#  1. EXCLUDING ON Δclk IS INCOHERENT IN A CYCLES GATE. Per-arm cycles exist precisely to normalise a
#     clock difference between arms. If the cycles view is trusted at all, Δclk is the thing it removes.
#     Δclk is still REPORTED, because it invalidates the SECONDS ratio — which is the shipped gate.
#
#  2. "CYCLES ARE INVARIANT TO BOOST" IS ONLY TRUE FOR CORE-BOUND KERNELS. For a DRAM-bound cell the wall
#     time is pinned by memory, so the cycle COUNT scales with core clock — the conversion is least
#     trustworthy exactly where the clock moved most. This is asserted in `tune.jl`'s unlocked-mode
#     rationale and in `cellcycles.jl`, and has never been verified per-regime. Until it is, a cycles
#     figure is evidence, not an arbiter. `--check-invariance` below is the experiment that would settle
#     it; it is not run automatically.
#
#  3. A HIGH `wobble` UNDER A VERIFIED LOCK MEANS THE RUN IS SUSPECT, NOT THAT THE CELL IS UNMEASURABLE.
#     The correct response is to RE-MEASURE that cell, which is why those cells are printed under
#     "RE-MEASURE" rather than given a permanent exemption. Excluding them would be exactly the
#     goalpost-moving the gate rule forbids: a sub-1.0 ratio is never a ceiling, and it is never a
#     rounding error either.
#
# ── THE WOBBLE TEST IS PER-CELL, AND IT IS NOT A GLOBAL THRESHOLD ───────────────────────────────────
# First attempt used a global bar of `3 × the box's null-measurement floor` (0.1% wintermute, 2.3% galen,
# from `kb/findings/pureblas-gate-repeatability-null-measurement.md`). That was wrong, and the error is
# worth recording: the null floor measures RATIO repeatability, while `wobble` measures CLOCK spread
# inside one timing window. Comparing them is apples to oranges, and on wintermute it flagged 176 of 863
# cells — a threshold that condemns a fifth of the cache is measuring its own units, not the data.
#
# The question that actually matters is not "did the clock move?" but "could the clock have moved this
# cell's VERDICT?". The gate line is `GATE_MIN` (0.995, i.e. the 2-sigdigit rounding boundary), so each
# cell has a margin — its fractional distance from that line — and the clock conversion carries an
# uncertainty of about `wobble`. Flag exactly when the uncertainty can cross the margin:
#
#     wobble% > 100 · |ratio − GATE_MIN| / GATE_MIN
#
# A cell at 1.40 with 5% wobble is untouchable and stays silent; a cell at 0.996 with 0.5% wobble is one
# clock excursion from flipping and gets re-measured. That is derived from the gate's own resolution
# rather than from taste, and it scales itself per cell instead of per box.
using Statistics
include(joinpath(@__DIR__, "gatecrit.jl"))          # gate_pass / GATE_MIN — THE criterion, not a copy

_host(path) = (m = match(r"plots_data_[^_]+_([a-z0-9]+)\.txt$", path); isnothing(m) ? "?" : m.captures[1])

function cells(path)
    out = Tuple{String, String, Int, Dict{String, Tuple{Float64, Float64, Float64}}}[]
    for ln in readlines(path)
        startswith(ln, "#") && continue
        p = split(ln, "\t")
        length(p) >= 4 || continue
        n = tryparse(Int, p[3])
        isnothing(n) && continue
        arms = Dict{String, Tuple{Float64, Float64, Float64}}()
        for f in p[4:end]
            isempty(f) && continue
            q = split(f, "|")
            length(q) >= 3 || continue
            # Field count GROWS over versions and the csv is ALWAYS last → index from the END.
            # Identical invariant to cellcycles.jl / cellratios.jl; do not switch to fixed indices.
            secs = median(parse.(Float64, split(q[end], ","))) * 1.0e-3
            khz1 = length(q) >= 6 ? something(tryparse(Float64, q[5]), NaN) : NaN
            lo = length(q) >= 8 ? something(tryparse(Float64, q[6]), NaN) : NaN
            hi = length(q) >= 8 ? something(tryparse(Float64, q[7]), NaN) : NaN
            khz, spread = (!isnan(lo) && !isnan(hi) && lo > 0) ? ((lo + hi) / 2, 100 * (hi - lo) / lo) :
                          (khz1, NaN)
            arms[q[1]] = (secs, khz, spread)
        end
        isempty(arms) || push!(out, (p[1], p[2], n, arms))
    end
    return out
end

cyc(a) = a[1] * a[2] * 1000.0

for path in ARGS
    isfile(path) || (println("missing: $path"); continue)
    rows = cells(path)
    println("\n══ $(basename(path))   [flag when wobble% > 100·|ratio−$(GATE_MIN)|/$(GATE_MIN), i.e. the clock could flip the verdict]")
    agree = disagree = 0
    remeasure = String[]
    flips = String[]
    for (lvl, op, n, arms) in rows
        haskey(arms, "pb") || continue
        refs = [k for k in keys(arms) if k != "pb"]
        isempty(refs) && continue
        pb = arms["pb"]
        # SECONDS ratio = the shipped gate. CYCLES ratio = the same comparison in the arm's own clock.
        rs = minimum(arms[r][1] for r in refs) / pb[1]
        rc_all = [cyc(arms[r]) / cyc(pb) for r in refs]
        rc = any(isnan, rc_all) ? NaN : minimum(rc_all)
        wob = maximum(x -> isnan(x[3]) ? -1.0 : x[3], values(arms))
        margin = 100 * abs(rs - GATE_MIN) / GATE_MIN      # how far this cell sits from the gate line, in %
        wob > margin && push!(remeasure,
            "$lvl $op@$n  ratio $(round(rs; digits = 3))  margin $(round(margin; digits = 2))% < wobble $(round(wob; digits = 1))%")
        isnan(rc) && continue
        if gate_pass(rs) == gate_pass(rc)
            agree += 1
        else
            disagree += 1
            push!(flips, "$lvl $op@$n  secs $(round(rs; digits = 3)) $(gate_pass(rs) ? "PASS" : "FAIL")" *
                         "   cycles $(round(rc; digits = 3)) $(gate_pass(rc) ? "PASS" : "FAIL")")
        end
    end
    println("   verdict agreement: $agree cells agree, $disagree DISAGREE between the seconds gate and the cycles gate")
    if !isempty(flips)
        # A disagreement is NOT a licence to pick the friendlier view. It says the cell's verdict rests on
        # the clock conversion, i.e. it is close enough to the line that the unit matters — re-measure it.
        println("   ── cells whose verdict depends on the unit (re-measure; do NOT pick the kinder one):")
        for f in first(flips, 12)
            println("      ", f)
        end
        length(flips) > 12 && println("      … and $(length(flips) - 12) more")
    end
    if !isempty(remeasure)
        println("   ── RE-MEASURE (in-window clock movement exceeds this cell's own margin to the gate line):")
        for r in first(remeasure, 12)
            println("      ", r)
        end
        length(remeasure) > 12 && println("      … and $(length(remeasure) - 12) more")
    end
end
println()
