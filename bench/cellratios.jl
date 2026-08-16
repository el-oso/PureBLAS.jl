# Per-size gate ratios for one op, READ OUT OF A CACHE  (lives in bench/, NOT bench/probes/ —
# `bench/probes/*.jl` is gitignored, so a tool put there is silently absent from the next checkout) — no measurement here, no clock touched.
# The gate table prints geomean/worst per op; the coverage tables need the WORST CELL and its n, which
# is not in that summary. Ratio definition and parse copied from bench/adjudicate.sh so this cannot
# disagree with the adjudicator: per-arm median of (ref ./ pb), then the FASTER reference sets the gate.
#
#   julia --project=bench bench/cellratios.jl <cache.txt> <op> [op...]
using Statistics
const CACHE = ARGS[1]
const OPS = ARGS[2:end]
for ln in readlines(CACHE)
    p = split(ln, "\t")
    length(p) >= 4 && p[2] in OPS || continue
    d = Dict{String, Vector{Float64}}()
    for f in p[4:end]
        isempty(f) && continue
        # arm|time|commit|anchor|freq|samples. The FIELD COUNT GROWS — it was 4, then 5 (anchor), now
        # 6 (per-cell freq) — and a fixed `limit=` broke on every bump, twice, by feeding
        # "anchor|samples" / "freq|samples" to `parse(Float64, …)`. The csv is ALWAYS last (plots.jl's
        # stated extension invariant: append BEFORE the csv), so index from the end and never again.
        _p = split(f, "|"); a, s = _p[1], _p[end]
        d[a] = parse.(Float64, split(s, ","))
    end
    haskey(d, "pb") || continue
    # estimator-ok: `median` IS the sanctioned gate estimator; this only reduces stored samples.
    r = Dict(a => median(d[a] ./ d["pb"]) for a in ("openblas", "aocl") if haskey(d, a))
    isempty(r) && continue
    lo = argmin(r)
    println(rpad(p[2], 8), rpad(p[3], 6), " gate=", round(r[lo]; digits = 3), " (vs ", lo, ")",
            "   ob=", haskey(r, "openblas") ? round(r["openblas"]; digits = 3) : "-",
            " aocl=", haskey(r, "aocl") ? round(r["aocl"]; digits = 3) : "-")
end
