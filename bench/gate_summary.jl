# Parse plots_data_<host>.txt cache → print per-op/size medians, flag < 0.96.
#
# ── RETIRED v1 READER. Two things are stale, not one ────────────────────────────────────────────────
#  1. FORMAT: it wants `plots_data_<host>.txt` with `size=vals;size=vals` in field 3. v3 caches are
#     `plots_data_<uarch>_<host>.txt` with all arms inline as `arm|date|commit|anchor|…` records, so
#     every line falls through and the output is empty-but-plausible.
#  2. CRITERION: `< 0.96` is not the gate. The gate is `>= max(OB, AOCL)` compared at TWO significant
#     digits, and it lives in exactly one place, `bench/gatecrit.jl`. This file predates that rule and
#     re-spells the comparison inline, which the rule exists to forbid.
# `bench/gate_gaps.jl <cache…>` supersedes it on both counts. Refuse rather than mislead.
using Statistics, Printf
host = length(ARGS) >= 1 ? ARGS[1] : "galen"
f = "bench/plots_data_$(host).txt"
isfile(f) || error("gate_summary.jl reads the RETIRED v1 cache `$f`, which no longer exists, and its " *
                   "0.96 threshold is not the gate criterion. Use: julia --project=bench " *
                   "bench/gate_gaps.jl bench/plots_data_*.txt")
below = Tuple{String, String, Int, Float64}[]
for ln in eachline(f)
    parts = split(ln, '\t')
    length(parts) < 3 && continue
    sec, op = parts[1], parts[2]
    for chunk in split(parts[3], ';')
        isempty(chunk) && continue
        kv = split(chunk, '=')
        length(kv) == 2 || continue
        sz = parse(Int, kv[1])
        vals = parse.(Float64, split(kv[2], ','))
        m = median(vals)
        m < 0.96 && push!(below, (sec, op, sz, m))
    end
end
sort!(below, by = x -> x[4])
@printf("%-6s %-10s %-8s %s\n", "SEC", "OP", "SIZE", "MEDIAN")
for (sec, op, sz, m) in below
    @printf("%-6s %-10s %-8d %.3f\n", sec, op, sz, m)
end
println("\n", length(below), " (op,size) pairs below 0.96")
