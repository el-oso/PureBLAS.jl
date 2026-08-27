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
const CACHE = ARGS[1]
const SEL = ARGS[2:end]
const ALL = "--all" in SEL

rows = Tuple{String, String, Int, Dict{String, Tuple{Float64, Float64}}}[]
for ln in readlines(CACHE)
    startswith(ln, "#") && continue
    p = split(ln, "\t")
    length(p) >= 4 || continue
    (ALL || p[2] in SEL) || continue
    n = tryparse(Int, p[3])
    isnothing(n) && continue
    arms = Dict{String, Tuple{Float64, Float64}}()   # arm => (median seconds, kHz)
    for f in p[4:end]
        isempty(f) && continue
        q = split(f, "|")
        # arm|time|commit|anchor|freq|samples — the field count GROWS (it was 4, then 5, then 6) and
        # the csv is ALWAYS last, so index from the END. Same invariant cellratios.jl relies on.
        length(q) >= 3 || continue
        secs = median(parse.(Float64, split(q[end], ","))) * 1.0e-3   # cache stores ms
        khz = length(q) >= 3 ? something(tryparse(Float64, q[end - 1]), NaN) : NaN
        arms[q[1]] = (secs, khz)
    end
    isempty(arms) || push!(rows, (p[1], p[2], n, arms))
end

isempty(rows) && (println("no matching cells in $CACHE"); exit(0))

cyc(a) = a[1] * a[2] * 1000.0            # seconds * kHz * 1000 = cycles
fmt(c) = isnan(c) ? "     —" : (c >= 1e6 ? string(round(c / 1e6; digits = 2), "M") :
                                c >= 1e3 ? string(round(c / 1e3; digits = 1), "k") :
                                string(round(Int, c)))

println("cycles per call (from each arm's OWN recorded clock). ref/pb > 1 ⇒ PB does LESS work.")
println("Δclk = max spread between the arms' clocks; a large Δclk means the SECONDS ratio is not adjudicable.")
@static if true
    println(rpad("lvl", 5), rpad("op", 12), lpad("n", 7), lpad("PB", 10), lpad("OB", 10),
            lpad("AOCL", 10), lpad("OB/PB", 8), lpad("AOCL/PB", 9), lpad("Δclk", 7))
end
for (lvl, op, n, arms) in sort(rows, by = r -> (r[2], r[3]))
    haskey(arms, "pb") || continue
    pbc = cyc(arms["pb"])
    obc = haskey(arms, "openblas") ? cyc(arms["openblas"]) : NaN
    aoc = haskey(arms, "aocl") ? cyc(arms["aocl"]) : NaN
    ks = [a[2] for a in values(arms) if !isnan(a[2]) && a[2] > 0]
    dclk = isempty(ks) ? NaN : 100 * (maximum(ks) - minimum(ks)) / minimum(ks)
    println(rpad(lvl, 5), rpad(op, 12), lpad(n, 7), lpad(fmt(pbc), 10), lpad(fmt(obc), 10),
            lpad(fmt(aoc), 10),
            lpad(isnan(obc) ? "—" : string(round(obc / pbc; digits = 2)), 8),
            lpad(isnan(aoc) ? "—" : string(round(aoc / pbc; digits = 2)), 9),
            lpad(isnan(dclk) ? "—" : string(round(dclk; digits = 1), "%"), 7))
end
