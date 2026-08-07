# THE WORKLIST: every failing (op, size) cell on a box, worst-first, from its cache.
#
# The gate is PB >= max(OpenBLAS, AOCL) at EVERY size, so a row fails on its worst cell and the SIZE is
# the actionable unit, not the op. Ratios are median(ref)/median(pb) per cell against the FASTER
# reference, exactly as plots.jl defines the gate; median at every level, never min or mean.
#
# CAVEAT, read before acting on a number: a cell whose arms carry different timestamps was measured
# ACROSS RUNS, and cross-run drift at cache-boundary sizes is ~3pp — bigger than most residuals here.
# Those cells are flagged `~`; re-measure with `bench/plots.jl bench op=<op>` before spending time on
# one. Cells under ~1% need K=10 replication (bench/cellrep.jl screens, bench/adjudicate.sh decides).
#
#   julia bench/failing.jl bench/plots_data_avx512_wintermute.txt [more caches...]
using Statistics
for path in ARGS
    cells = Tuple{String, String, Int, Float64, String, Bool}[]
    host = "?"
    for ln in eachline(path)
        startswith(ln, "#pbbench") && (m = match(r"host=(\S+)", ln); !isnothing(m) && (host = m[1]); continue)
        p = split(ln, '\t')
        length(p) >= 4 || continue
        d = Dict{String, Vector{Float64}}(); stamps = Dict{String, String}()
        for fld in p[4:end]
            isempty(fld) && continue
            _p = split(fld, '|'); a, ts, s = _p[1], _p[2], _p[end]
            d[a] = parse.(Float64, split(s, ',')); stamps[a] = ts
        end
        haskey(d, "pb") || continue
        tp = median(d["pb"])
        rs = [(r, median(d[r]) / tp) for r in ("openblas", "aocl") if haskey(d, r)]
        isempty(rs) && continue
        r, g = rs[argmin(last.(rs))]
        g < 1.0 || continue
        crossrun = any(stamps[a] != stamps["pb"] for a in keys(stamps))
        push!(cells, (p[1], p[2], parse(Int, p[3]), g, r, crossrun))
    end
    sort!(cells; by = c -> c[4])
    println("\n", "="^88, "\n", host, " — ", length(cells), " failing cells (worst first; `~` = arms measured across runs)\n", "="^88)
    for (lvl, op, n, g, r, xr) in cells
        println(rpad(lvl, 5), rpad(op, 11), rpad("n=$n", 11), rpad(round(g; digits = 3), 8),
                "vs ", rpad(r, 10), xr ? "~" : "")
    end
    byop = Dict{String, Float64}()
    for (_, op, _, g, _, _) in cells
        byop[op] = min(get(byop, op, 1.0), g)
    end
    println("\n  failing OPS (worst cell): ",
            join((string(k, " ", round(v; digits = 3)) for (k, v) in sort(collect(byop); by = last)), "  "))
end
