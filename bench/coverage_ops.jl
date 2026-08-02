# Emit the per-FUNCTION coverage tables of docs/src/coverage.md straight from the v3 caches.
#
# WHY PER FUNCTION. The doc used to carry four lumped rows (BLAS-1 / BLAS-2 dense / BLAS-2 banded-packed
# / BLAS-3) with one verdict each. A single 🐢 over ~40 routines hides which ones actually miss: it made
# `gbmv` (1.9× and gating) and `ztrsm` (0.65) indistinguishable, and it made progress invisible — closing
# three of four cells in a group did not change the row. One line per routine, one verdict per routine.
#
# THE VERDICT IS COMPUTED, NEVER JUDGED. `gate` = min over that routine's cells of the ratio against the
# FASTER reference at each cell — i.e. exactly the project rule PB ≥ max(OpenBLAS, AOCL), evaluated per
# cell and then worst-cased. 🐰 iff gate ≥ 1.0. Both arms come from the same round of the same run (v3),
# so the max is over numbers that saw one machine state.
#
# Usage: julia --project=bench bench/coverage_ops.jl bench/plots_data_<uarch>_<host>.txt [more…] > out.md
const QN = 48
med(v) = (s = sort(v); s[max(1, cld(length(s), 2))])

# op => (level, types) — types are what the bench row actually exercises, not what the routine supports.
const LEVEL = Dict{String, String}()
lvlof(l) = l in ("L1", "CL1") ? "BLAS-1" : l in ("L2", "CL2") ? "BLAS-2" : l in ("L3", "CL3") ? "BLAS-3" : "LAPACK"

cells = Dict{Tuple{String, String}, Vector{Tuple{Int, Float64}}}()   # (level, op) => [(size, gate)]
for path in ARGS, ln in eachline(path)
    (isempty(strip(ln)) || startswith(ln, "#")) && continue
    p = split(ln, "\t"); length(p) >= 4 || continue
    lvl, op, sz = String(p[1]), String(p[2]), parse(Int, p[3])
    d = Dict{String, Vector{Float64}}()
    for f in p[4:end]
        a, _, _, csv = split(f, "|"; limit = 4)
        d[String(a)] = parse.(Float64, split(csv, ","))
    end
    haskey(d, "pb") || continue
    rs = Float64[]
    for r in ("openblas", "aocl")
        haskey(d, r) || continue
        n = min(length(d[r]), length(d["pb"])) ÷ QN
        n == 0 && continue
        push!(rs, med([med([d[r][(i - 1) * QN + q] / d["pb"][(i - 1) * QN + q] for q in 1:QN]) for i in 1:n]))
    end
    isempty(rs) && continue
    push!(get!(cells, (lvlof(lvl), op), Tuple{Int, Float64}[]), (sz, minimum(rs)))  # vs the FASTER reference
end

for section in ("BLAS-1", "BLAS-2", "BLAS-3", "LAPACK")
    ops = sort([k[2] for k in keys(cells) if k[1] == section])
    isempty(ops) && continue
    println("\n#### $section\n")
    println("| routine | gate | worst cell | geomean | sizes |")
    println("|---|---|---|---|---|")
    for op in ops
        cs = sort(cells[(section, op)]; by = first)
        gs = [c[2] for c in cs]
        gate = minimum(gs)
        wi = argmin(gs)
        geo = exp(sum(log, gs) / length(gs))
        icon = gate >= 1.0 ? "🐰" : "🐢"
        println("| `$op` | $icon $(round(gate; digits=2)) | n=$(cs[wi][1]) | $(round(geo; digits=2)) | $(length(cs)) |")
    end
end
