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
using Statistics

const QN = 48
# t_{R-1, 0.975}. Only a few R occur (plots.jl uses 8 light / 4 heavy); the fallback is a crude
# interpolation — fine, since R outside this set is not a regime we run.
const _TQ = Dict(2 => 12.706, 3 => 4.303, 4 => 3.182, 6 => 2.571, 8 => 2.365, 16 => 2.131, 32 => 2.040)
_t(R) = get(() -> (R < 2 ? Inf : 1.96 + 8.0 / R), _TQ, R)

# δ — the dead band, i.e. the resolution of the whole methodology. A deficit smaller than the machine's
# own day-scale nonstationarity is not a reproducible property of the CODE: it will not replicate
# tomorrow, so chasing it is chasing machine state. PROVISIONAL 0.003, from same-methodology repeats
# minutes apart (galen iamax 0.936/0.934, zdotc 0.997/0.995 ⇒ σ̂_pair ≈ 0.1–0.15%, δ = 2σ̂). This is NOT a
# 1% concession: every currently-open BLAS-1 gap (0.5–3.7%) sits outside it and still reads FAIL/INDET.
# Replace with 2·σ̂_pair per host once a canary history exists.
const DELTA = 0.003

# One cell, one reference. Returns the point ratio, a two-sided 95% interval, and a verdict.
# σdrift > 0 widens the interval for MIXED-PROVENANCE cells (pb and ref measured in different runs) —
# that is what pushes small-gap cached-reference cells to INDET instead of letting them masquerade.
function cellverdict(qref, qpb; δ = DELTA, σdrift = 0.0)
    R = min(length(qref), length(qpb)) ÷ QN
    R < 2 && return (ratio = NaN, lo = NaN, hi = NaN, R = R, verdict = :INDET)
    x = [log(median(view(qref, (r - 1) * QN + 1:r * QN) ./ view(qpb, (r - 1) * QN + 1:r * QN)))
         for r in 1:R]
    m = mean(x)
    hw = sqrt((_t(R) * std(x) / sqrt(R))^2 + (1.96 * σdrift)^2)
    lb, ub = m - hw, m + hw
    v = lb >= 0 ? :PASS :                    # demonstrably at or above the reference
        lb >= log1p(-δ) ? :PASSEQ :          # indistinguishable from parity at our resolution
        ub < log1p(-δ) ? :FAIL :             # demonstrably below even the dead band
        :INDET                               # cannot distinguish — escalate, never force a binary
    return (ratio = exp(m), lo = exp(lb), hi = exp(ub), R = R, verdict = v)
end

# Routine verdict = WORST VERDICT over cells × refs, never `min` of the point estimates. "Must hold at
# every size" is preserved exactly: one FAIL cell fails the routine. Multiple comparisons are asymmetric
# and deliberately uncorrected in the strict direction — "all cells pass" is an intersection test, so a
# routine containing one truly-deficient cell falsely passes only if THAT cell's bound falsely clears
# (≤2.5%), regardless of how many healthy cells surround it. Adding sizes or boxes never inflates
# false-PASS. False FAILs are handled by re-measuring the borderline cell, not by widening every interval.
const _ORD = Dict(:FAIL => 4, :INDET => 3, :PASSEQ => 2, :PASS => 1)
const _ICON = Dict(:FAIL => "🐢", :INDET => "❓", :PASSEQ => "🐰≈", :PASS => "🐰")

sel = something(findfirst(a -> startswith(a, "group="), ARGS), 0)
want = sel == 0 ? nothing : ARGS[sel][7:end]
paths = [a for a in ARGS if !startswith(a, "group=")]

cells = Dict{Tuple{String, String}, Vector{NamedTuple}}()
for path in paths, ln in eachline(path)
    (isempty(strip(ln)) || startswith(ln, "#")) && continue
    p = split(ln, "\t"); length(p) >= 4 || continue
    lvl, op, sz = String(p[1]), String(p[2]), parse(Int, p[3])
    isnothing(want) || lvl == want || continue
    d = Dict{String, Vector{Float64}}(); stamp = Dict{String, String}()
    for f in p[4:end]
        a, dt, cm, csv = split(f, "|"; limit = 4)
        d[String(a)] = parse.(Float64, split(csv, ",")); stamp[String(a)] = String(dt)
    end
    haskey(d, "pb") || continue
    for r in ("openblas", "aocl")
        haskey(d, r) || continue
        # Mixed provenance: pb and ref from different runs. σdrift from the 2026-08-04 observation
        # (identical code, cached vs same-round reference differing by up to 3%).
        mixed = stamp[r] != stamp["pb"]
        v = cellverdict(d[r], d["pb"]; σdrift = mixed ? 0.010 : 0.0)
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
println("\nδ = ", DELTA, " (provisional; = 2·σ̂_pair from same-methodology repeats). ",
        "🐰 lb≥1  🐰≈ lb≥1-δ  ❓ straddles  🐢 ub<1-δ")
println("⚠ = cell mixes a freshly-measured pb arm with a reference arm from an earlier run; its ",
        "interval is widened by σdrift=1% and it should not decide anything near parity.")
