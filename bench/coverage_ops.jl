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

# ONE COLUMN PER MICROARCHITECTURE. A single pooled verdict answers "does this routine gate SOMEWHERE",
# which is the wrong question — the gate is per box. Pooling also makes progress invisible in exactly the
# way the four lumped rows did: on 2026-08-05 `iamax` and `dzasum` went green on Zen4 and the pooled rows
# stayed 🐢, with nothing to say which box still missed or at what size. Per box, the next machine to fix
# is readable straight off the row.
cells = Dict{Tuple{String, String, String}, Vector{Tuple{Int, Float64}}}()  # (level, op, uarch) => [(size, gate)]
const UARCH = String[]                                    # column order = the order caches were given
for path in ARGS
    ua = "?"
    for ln in eachline(path)
        if startswith(ln, "#pbbench")
            mu = match(r"uarch=(\S+)", ln); mi = match(r"isa=(\S+)", ln)
            ua = string(isnothing(mu) ? "?" : mu[1], isnothing(mi) ? "" : " · " * mi[1])
            ua in UARCH || push!(UARCH, ua)
            continue
        end
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
        push!(get!(cells, (lvlof(lvl), op, ua), Tuple{Int, Float64}[]), (sz, minimum(rs)))  # vs the FASTER ref
    end
end

for section in ("BLAS-1", "BLAS-2", "BLAS-3", "LAPACK")
    ops = sort(unique(k[2] for k in keys(cells) if k[1] == section))
    isempty(ops) && continue
    println("\n#### $section\n")
    println("| routine | ", join(UARCH, " | "), " |")
    println("|---", "|---"^length(UARCH), "|")
    for op in ops
        cols = String[]
        for ua in UARCH
            cs = sort(get(cells, (section, op, ua), Tuple{Int, Float64}[]); by = first)
            if isempty(cs)
                push!(cols, "—")                       # not measured on this box
                continue
            end
            gs = [c[2] for c in cs]
            gate = minimum(gs)                          # the gate IS the worst cell, per box
            # A passing row needs no size — it gates everywhere. A failing one names the cell to fix,
            # which is the actionable unit; the ratio alone would say nothing about where to look.
            push!(cols, gate >= 1.0 ? "🐰 $(round(gate; digits = 2))" :
                        "🐢 $(round(gate; digits = 2)) n=$(cs[argmin(gs)][1])")
        end
        println("| `$op` | ", join(cols, " | "), " |")
    end
end
