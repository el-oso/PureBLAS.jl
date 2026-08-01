# Rank the gate gaps, and separate REAL gaps from measurement noise.
#
# The gate is PB ≥ max(OpenBLAS, AOCL) at every cell. A bare "FAIL" does not say whether a cell misses
# by 35% or by 0.1%, and chasing the latter is optimising against measurement error. This ranks by the
# actual shortfall and prints, per cell, the round-to-round spread of the ratio so a miss can be read
# against the precision of the thing that measured it.
#
# WHY THE SPREAD IS RECOVERABLE AT ALL: v3 stores each arm's quantile vector with one round's 48
# quantiles appended per round, in round order. Splitting the stored vector back into 48-element chunks
# gives the per-round ratio, so `spread` below is a real measured quantity, not an assumption.
#
# CAVEAT, and it matters: this is WITHIN-RUN spread. On 2026-08-01 the same trmv@512 cell read 1.001 in
# an `op=` run and 0.959 in a `group=` sweep — 4%, while within-run round spread at that size was under
# 2%. Process-level variation (address/page mapping, allocator state) is LARGER than round-level and is
# NOT captured here. Treat `spread` as a lower bound on the uncertainty.
#
# Usage:  julia --project=bench bench/gate_gaps.jl bench/plots_data_<uarch>_<host>.txt [more...]

const QN = 48

function cells(path)
    out = []                                   # (lvl, op, size, Dict(arm => Vector{Float64}))
    for ln in eachline(path)
        (isempty(strip(ln)) || startswith(ln, "#")) && continue
        p = split(ln, "\t"); length(p) >= 4 || continue
        d = Dict{String, Vector{Float64}}()
        for f in p[4:end]
            a, _, _, csv = split(f, "|"; limit = 4)
            d[String(a)] = parse.(Float64, split(csv, ","))
        end
        push!(out, (String(p[1]), String(p[2]), parse(Int, p[3]), d))
    end
    return out
end

med(v) = sort(v)[max(1, cld(length(v), 2))]
# per-round ratio medians: chunk both arms' stored vectors into rounds of QN and divide elementwise
function roundratios(qref, qpb)
    n = min(length(qref), length(qpb)) ÷ QN
    return [med([qref[(r - 1) * QN + i] / qpb[(r - 1) * QN + i] for i in 1:QN]) for r in 1:n]
end

rows = []
for path in ARGS, (lvl, op, sz, d) in cells(path)
    haskey(d, "pb") || continue
    refs = [r for r in ("openblas", "aocl") if haskey(d, r)]
    isempty(refs) && continue
    best = ""; worst = Inf; spread = 0.0
    for r in refs                              # the gate is vs the FASTER reference at THIS cell
        rr = roundratios(d[r], d["pb"]); isempty(rr) && continue
        m = med(rr)
        m < worst && (worst = m; best = r; spread = (maximum(rr) - minimum(rr)) / m)
    end
    isfinite(worst) && push!(rows, (worst, spread, lvl, op, sz, best, basename(path)))
end

sort!(rows; by = first)
fails = [r for r in rows if r[1] < 1.0]
println("cells=", length(rows), "  below 1.0=", length(fails))
println("\n  gap    ratio  spread  cell                          vs         cache")
for (ratio, spread, lvl, op, sz, ref, f) in fails
    # a miss smaller than the cell's own round-to-round spread is not distinguishable from noise
    tag = (1.0 - ratio) <= spread ? "  <- within spread" : ""
    println(rpad(string(round(100 * (1 - ratio); digits = 1), "%"), 7),
        rpad(round(ratio; digits = 3), 7), rpad(round(spread; digits = 3), 8),
        rpad("$lvl $op@$sz", 30), rpad(ref, 11), replace(f, "plots_data_" => "", ".txt" => ""), tag)
end
nnoise = count(r -> (1.0 - r[1]) <= r[2], fails)
println("\n$(length(fails) - nnoise) of $(length(fails)) misses exceed their own round spread; ",
    "$nnoise are within it (lower bound on noise — process-level variation is larger).")
