# Per-cell work in CYCLES, read out of a cache — no measurement, no clock touched.
# (Lives in bench/, NOT bench/probes/ — the latter is gitignored, so a tool there vanishes on checkout.)
#
#   julia --project=bench bench/cellcycles.jl <cache.txt> <op> [op...]
#   julia --project=bench bench/cellcycles.jl <cache.txt> --all          # every op in the cache
#
# WHY CYCLES. Seconds conflate two different things: how much WORK a kernel does, and how fast the
# clock happened to be running. A gate ratio hides the problem by dividing it out, but only when both
# arms ran at the same clock — and they often did not, because reference arms are cached from an
# earlier run (the "never re-measure a reference" rule) while the pb arm is fresh. Cycles are the
# invariant: 25.8 kcyc is 25.8 kcyc on a 2.0 GHz laptop and a 3.7 GHz desktop, so the question
# "are we doing more work than AOCL?" gets a direct answer instead of an inference.
#
# This needs NO new measurement. Every arm record already stores its OWN achieved clock (kHz) — not
# the header's, the arm's — so `cycles = seconds * kHz * 1000` is exact per arm and automatically
# accounts for two arms that ran at different clocks. The `Δclk` column surfaces exactly that: a cell
# whose arms disagree by more than a per-cent or two is telling you the seconds-ratio is not
# adjudicable, which is the `per-cell-anchor-not-run-drift` rule made visible.
#
# Two things this makes obvious that seconds do not:
#   * cross-µarch comparison. PB's zgetrf(50) panel took 25.6 kcyc on Zen3 (W=4) and 25.8 kcyc on
#     Zen4 (W=8) — i.e. doubling the vector width bought NOTHING. In seconds that reads as "Zen4 is
#     faster" (it is, it clocks higher) and the real finding is invisible. Cycles named it, and it
#     turned out to be correct behaviour: Zen4 double-pumps 512-bit FMAs over a 256-bit datapath.
#   * "who does less work". A ratio says we are 1.36x slower; cycles say AOCL spends 8.6 kcyc where we
#     spend 11.7 kcyc, which is a budget you can decompose against a roofline.
using Statistics
include(joinpath(@__DIR__, "gatecrit.jl"))   # arm_pair — which arms this cell's tier compares
const CACHE = ARGS[1]
const SEL = ARGS[2:end]
const ALL = "--all" in SEL

rows = Tuple{String, String, Int, Dict{String, Tuple{Float64, Float64, Float64}}}[]
for ln in readlines(CACHE)
    startswith(ln, "#") && continue
    p = split(ln, "\t")
    length(p) >= 4 || continue
    (ALL || p[2] in SEL) || continue
    n = tryparse(Int, p[3])
    isnothing(n) && continue
    arms = Dict{String, Tuple{Float64, Float64, Float64}}()   # arm => (median s, kHz, in-window spread %)
    for f in p[4:end]
        isempty(f) && continue
        q = split(f, "|")
        # arm|time|commit|anchor|freq[|flo|fhi]|samples — the field count GROWS (4, then 5, then 6, now
        # 8) and the csv is ALWAYS last, so index from the END. Same invariant cellratios.jl relies on.
        length(q) >= 3 || continue
        secs = median(parse.(Float64, split(q[end], ","))) * 1.0e-3   # cache stores ms
        # DO NOT assume the clock was pinned. Newer records carry the min/max clock actually observed
        # while this cell was being timed; use the MIDPOINT for the conversion and report the spread, so
        # a variable clock degrades the number honestly instead of silently. Older records have only the
        # single stamp-time sample: use it, and mark the spread UNKNOWN (NaN) rather than claiming 0%.
        khz1 = length(q) >= 6 ? something(tryparse(Float64, q[5]), NaN) : NaN
        lo = length(q) >= 8 ? something(tryparse(Float64, q[6]), NaN) : NaN
        hi = length(q) >= 8 ? something(tryparse(Float64, q[7]), NaN) : NaN
        # A MIDPOINT IS ONLY A CLOCK WHEN THE RANGE IS TIGHT, and on a threaded arm it often is not.
        # `flo` is the MINIMUM over every thread in state R, sampled at the window BOUNDARIES — the two
        # instants at which the gemm pool's workers are most likely to be parked. A core caught coming
        # out of idle reports a low `scaling_cur_freq` even under a hard pin, so the low end of the
        # range can be a sleeping core rather than slow work. Measured, galen `L1/axpy@1000000` arm
        # `pb_mt`: flo = 546664, fhi = 3675043 — a 6.7x range inside ONE cell, on a box pinned
        # `scaling_min_freq == scaling_max_freq == 3701000` and POST-LOCK verified at 3674 MHz under
        # load. The midpoint there is ~2111 MHz against a real ~3675, which put a 43% error straight
        # into the cycle count, silently, in the one tool whose whole purpose is to be clock-invariant.
        #
        # So the range has to earn the midpoint. 3% is the tolerance the audit chain itself uses to
        # decide whether two figures describe the same clock (`bench/check_arm_clocks.sh`, default
        # `tol`), so a range wider than that is one the audit would already refuse to call a single
        # clock, and averaging its ends cannot produce one either.
        #
        # ABSTAIN RATHER THAN SUBSTITUTE. The obvious fallback is `khz1` (field 5), which reads the pin
        # correctly on the cell above — but whether it is sound FLEET-WIDE is unverified, and swapping
        # one unvalidated clock for another is how the 43% got here. `NaN` is honest: the caller
        # already filters it and prints "—", and the spread is still reported so a reader sees WHY the
        # cycles are missing instead of finding a number that averaged a sleeping core.
        khz, spread = if !isnan(lo) && !isnan(hi) && lo > 0
            sp = 100 * (hi - lo) / lo
            sp <= 3.0 ? ((lo + hi) / 2, sp) : (NaN, sp)
        else
            (khz1, NaN)
        end
        arms[q[1]] = (secs, khz, spread)
    end
    isempty(arms) || push!(rows, (p[1], p[2], n, arms))
end

isempty(rows) && (println("no matching cells in $CACHE"); exit(0))

cyc(a) = a[1] * a[2] * 1000.0            # seconds * kHz * 1000 = cycles
fmt(c) = isnan(c) ? "     —" : (c >= 1e6 ? string(round(c / 1e6; digits = 2), "M") :
                                c >= 1e3 ? string(round(c / 1e3; digits = 1), "k") :
                                string(round(Int, c)))

println("cycles per call, from each arm's OWN clock (midpoint of its in-window min/max). ref/pb > 1 ⇒ PB does LESS work.")
println("Δclk  = spread BETWEEN arms — large ⇒ the seconds ratio compares two machine states, not two kernels.")
println("wobble = worst spread WITHIN one arm's own timing windows — large ⇒ the clock moved DURING the measurement,")
println("         so that cell's cycle figures carry that much uncertainty. `?` = pre-range record, cannot verify.")
println(rpad("lvl", 5), rpad("op", 12), lpad("n", 7), lpad("PB", 10), lpad("OB", 10),
        lpad("AOCL", 10), lpad("OB/PB", 8), lpad("AOCL/PB", 9), lpad("Δclk", 7), lpad("wobble", 8))
for (lvl, op, n, arms) in sort(rows, by = r -> (r[2], r[3]))
    pba, _ = arm_pair(arms)                    # serial or threaded, read from the cell — gatecrit.jl
    haskey(arms, pba) || continue
    pbc = cyc(arms[pba])
    pick(rs...) = (i = findfirst(r -> haskey(arms, r), rs); isnothing(i) ? NaN : cyc(arms[rs[i]]))
    obc = pick("openblas", "openblas_mt")
    aoc = pick("aocl", "aocl_mt")
    ks = [a[2] for a in values(arms) if !isnan(a[2]) && a[2] > 0]
    dclk = isempty(ks) ? NaN : 100 * (maximum(ks) - minimum(ks)) / minimum(ks)
    sp = [a[3] for a in values(arms) if !isnan(a[3])]
    wob = isempty(sp) ? NaN : maximum(sp)
    println(rpad(lvl, 5), rpad(op, 12), lpad(n, 7), lpad(fmt(pbc), 10), lpad(fmt(obc), 10),
            lpad(fmt(aoc), 10),
            # A RATIO NEEDS BOTH SIDES. Testing only the reference printed `NaN` the moment PB's own
            # clock became unknown — the reference cycles are fine, the denominator is not, and
            # `NaN` in a column of ratios reads as a defect in the kernel rather than a missing clock.
            lpad(isnan(obc) || isnan(pbc) ? "—" : string(round(obc / pbc; digits = 2)), 8),
            lpad(isnan(aoc) || isnan(pbc) ? "—" : string(round(aoc / pbc; digits = 2)), 9),
            lpad(isnan(dclk) ? "—" : string(round(dclk; digits = 1), "%"), 7),
            lpad(isnan(wob) ? "?" : string(round(wob; digits = 1), "%"), 8))
end
