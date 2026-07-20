# Parse plots_data_<host>.txt cache → print per-op/size medians, flag < 0.96.
using Statistics, Printf
host = length(ARGS) >= 1 ? ARGS[1] : "galen"
f = "bench/plots_data_$(host).txt"
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
