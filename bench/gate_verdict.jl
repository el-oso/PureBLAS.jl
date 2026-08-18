# Gate verdicts WITH UNCERTAINTY, from the v3 caches. Companion to coverage_ops.jl, which reports a
# point estimate per routine; this reports an interval and a four-way verdict.
#
# WHY. The gate rule ("PB ≥ max(OpenBLAS, AOCL) at every size") was enforced as a threshold on a point
# estimate with no uncertainty attached, which has two consequences:
#   1. A routine truly AT parity flips 🐰/🐢 between runs by construction — half the time noise lands low.
#   2. `min` over ~14 cells × 2 refs × 3 boxes is a MAX-OF-NOISE estimator. The minimum of k near-tied
#      unbiased estimates is biased DOWN by ≈ σ·√(2 ln k) — 1.5–2σ at k=14–28 — and it gets worse every
#      time a size or a box is added. The published "gate" was therefore systematically pessimistic.
# Measured 2026-08-04, both consequences are live: galen `iamax` reads 1.029 against a cached reference
# and 0.999 against a same-round one WITH THE CODE HELD FIXED (on AVX2 the two unroll arms resolve to the
# same value, so the kernel is byte-identical to master). 3% of pure methodology.
#
# WHAT IS PAIRED. The ROUND, not the sample. Arms run in separate `@be` windows, so there is no per-sample
# correspondence to lose — the 48-quantile reduction in plots.jl destroyed nothing that existed. Each
# round contributes one paired log-ratio, and the cache already stores round blocks per arm in matched
# order, so this needs NO cache format change.
#
# Usage: julia --project=bench bench/gate_verdict.jl bench/plots_data_<uarch>_<host>.txt [more…] [group=L1]
using Statistics, Random

const QN = 48
# THE STATISTIC IS THE MEDIAN, EVERYWHERE. That is the approved methodology (CLAUDE.md: "median times
# (not min)"; plots.jl forms median(ref_q ./ pb_q) per round; coverage_ops.jl medians over rounds) and it
# is not negotiable here — the point of the median is exactly to be insensitive to the window tails that
# a `min` ignores and a `mean` over-weights.
#
# An earlier draft of this file used mean±t on log-ratios, and the probe harnesses used `minimum` of
# repeated timings. Both were unapproved substitutions and the second one was actively harmful: `min` is
# optimistic AND tail-blind, so a warm A/B on minima ranked an iamax unroll nb2 > nb4 at n=1e6 while the
# gate's median ranked it 15% WORSE. Sample count was not the problem — with `min`, more samples makes the
# estimator drift further from the median. Do not reintroduce either.
#
# Interval: BOOTSTRAP percentile CI of the median, over the pooled per-round log-ratios. Non-parametric,
# so it needs no normality claim about round medians, and pooling rounds from EVERY available run folds
# the between-run component in automatically — which matters, because a within-run-only interval was
# measured too narrow (2026-08-04: nrm2's run-1 95% CI [2.003, 2.016] did not contain run 2's 1.995).
include(joinpath(@__DIR__, "gatecrit.jl"))   # gate_pass / GATE_MIN — THE gate criterion
const _LGATE = log(GATE_MIN)                 # parity in LOG-ratio space, where this file works
const _NBOOT = 4000
function cellverdict(rs::Vector{Float64}; σdrift = 0.0, seed = 20260804)
    R = length(rs)
    R < 2 && return (ratio = NaN, lo = NaN, hi = NaN, R = R, verdict = :INDET)
    m = median(rs)
    st = Random.MersenneTwister(seed)                  # fixed ⇒ the verdict is reproducible
    bs = Vector{Float64}(undef, _NBOOT)
    idx = Vector{Int}(undef, R)
    for b in 1:_NBOOT
        rand!(st, idx, 1:R)
        bs[b] = median(view(rs, idx))
    end
    sort!(bs)
    lb, ub = bs[max(1, round(Int, 0.025 * _NBOOT))], bs[min(_NBOOT, round(Int, 0.975 * _NBOOT))]
    # σdrift widens for mixed-provenance cells (fresh pb arm vs a reference from an earlier run).
    lb -= 1.96 * σdrift; ub += 1.96 * σdrift
    # ONE null hypothesis: the gate HOLDS. Flag only on evidence that it does not, i.e. the whole
    # interval below parity. No dead band — with the between-run component included the interval IS the
    # dead band, and it self-calibrates per cell instead of being a global constant.
    # Parity is `log(GATE_MIN)`, not 0: the gate is met when the ratio ROUNDED TO TWO SIGNIFICANT
    # DIGITS is >= 1.00 (bench/gatecrit.jl), i.e. ratio >= 0.995. Comparing against 0 here would call a
    # cell failing at a precision the published table does not even show.
    v = lb >= _LGATE ? :PASS :
        ub < _LGATE ? :FAIL :
        (ub - lb) > 0.02 ? :INDET :          # genuinely unresolved (>2% wide) ⇒ more rounds
        :PASS                                # interval contains parity ⇒ no evidence of a regression
    return (ratio = exp(m), lo = exp(lb), hi = exp(ub), R = R, verdict = v)
end

# Per-round log ratios for one (cell, ref); caller concatenates across runs before verdicting.
function roundratios(qref, qpb)
    R = min(length(qref), length(qpb)) ÷ QN
    return [log(median(view(qref, (r - 1) * QN + 1:r * QN) ./ view(qpb, (r - 1) * QN + 1:r * QN)))
            for r in 1:R]
end

# Routine verdict = WORST VERDICT over cells × refs, never `min` of the point estimates. "Must hold at
# every size" is preserved exactly: one FAIL cell fails the routine. Multiple comparisons are asymmetric
# and deliberately uncorrected in the strict direction — "all cells pass" is an intersection test, so a
# routine containing one truly-deficient cell falsely passes only if THAT cell's bound falsely clears
# (≤2.5%), regardless of how many healthy cells surround it. Adding sizes or boxes never inflates
# false-PASS. False FAILs are handled by re-measuring the borderline cell, not by widening every interval.
const _ORD = Dict(:FAIL => 4, :INDET => 3, :PASS => 1)
const _ICON = Dict(:FAIL => "🐢", :INDET => "❓", :PASS => "🐰")

sel = something(findfirst(a -> startswith(a, "group="), ARGS), 0)
want = sel == 0 ? nothing : ARGS[sel][7:end]
const NODRIFT = "nodrift" in ARGS
paths = [a for a in ARGS if !startswith(a, "group=") && a != "nodrift"]

cells = Dict{Tuple{String, String}, Vector{NamedTuple}}()
for path in paths, ln in eachline(path)
    (isempty(strip(ln)) || startswith(ln, "#")) && continue
    p = split(ln, "\t"); length(p) >= 4 || continue
    lvl, op, sz = String(p[1]), String(p[2]), parse(Int, p[3])
    isnothing(want) || lvl == want || continue
    d = Dict{String, Vector{Float64}}(); stamp = Dict{String, String}()
    for f in p[4:end]
        _p = split(f, "|"); a, dt, cm, csv = _p[1], _p[2], _p[3], _p[end]
        d[String(a)] = parse.(Float64, split(csv, ",")); stamp[String(a)] = String(dt)
    end
    haskey(d, "pb") || continue
    for r in ("openblas", "aocl")
        haskey(d, r) || continue
        # Mixed provenance: pb and ref from different runs. σdrift from the 2026-08-04 observation
        # (identical code, cached vs same-round reference differing by up to 3%).
        # `nodrift`: pass when POOLING REPEATS of the same mode (e.g. 3 cached-ref runs). The between-run
        # variance is then already inside the pooled rounds, so adding σdrift on top double-counts it.
        # Measured 2026-08-04: three back-to-back cached-ref runs of identical code gave 0.954/0.963/0.963
        # — 0.9% spread — versus 0.4-0.75% for full-arm. Cached mode is usable for effects above ~2%; the
        # repeats are what turn it into an interval instead of a point.
        mixed = stamp[r] != stamp["pb"]
        v = cellverdict(roundratios(d[r], d["pb"]); σdrift = (mixed && !NODRIFT) ? 0.010 : 0.0)
        push!(get!(cells, (op, lvl), NamedTuple[]), merge(v, (size = sz, ref = r, mixed = mixed)))
    end
end

println(rpad("routine", 10), rpad("verdict", 9), rpad("worst cell", 30), "R   mixed")
for (op, lvl) in sort(collect(keys(cells)))
    cs = cells[(op, lvl)]
    w = cs[argmax([(_ORD[c.verdict], -c.ratio) for c in cs])]
    nmix = count(c -> c.mixed, cs)
    println(rpad(op, 10), rpad(_ICON[w.verdict], 9),
            rpad(string(round(w.ratio; digits=3), " [", round(w.lo; digits=3), ", ",
                        round(w.hi; digits=3), "] n=", w.size, " ", w.ref), 30),
            rpad(w.R, 4), nmix == 0 ? "-" : "$nmix/$(length(cs)) ⚠")
end
println("\nMedian of per-round ratios; 95% bootstrap CI (4000 resamples), rounds POOLED across every ",
        "cache given, so between-run variance is included.")
println("🐰 CI contains or exceeds parity (no evidence of a regression)   ",
        "🐢 CI entirely below parity   ❓ CI wider than 2% (needs more rounds)")
println("⚠ = cell mixes a freshly-measured pb arm with a reference arm from an earlier run; its ",
        "interval is widened by σdrift=1% and it should not decide anything near parity.")
