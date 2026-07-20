# Parse the committed fleet plot caches and itemize every gate MISS (median ratio < 1.0) per
# (op, size, box, baseline). Ratio is PB/baseline speed (>1 = PB faster). Gate = >= max(OB, AOCL).
using Statistics, Printf
const BOXES = ["wintermute" => "Zen4/AVX-512", "galen" => "Zen3/AVX2", "neuromancer" => "Zen5/AVX-512"]
function parse_cache(path)
    d = Dict{Tuple{String, String, Int}, Float64}()   # (op, level, size) -> median ratio
    isfile(path) || return d
    for ln in eachline(path)
        startswith(ln, "#") && continue
        f = split(ln, '\t'); length(f) < 3 && continue
        lvl = String(f[1]); op = String(f[2])
        for blk in split(f[3], ';')
            sp = split(blk, '='); length(sp) == 2 || continue
            sz = tryparse(Int, sp[1]); sz === nothing && continue
            vals = [parse(Float64, x) for x in split(sp[2], ',') if !isempty(x)]
            isempty(vals) && continue
            d[(op, lvl, sz)] = median(vals)
        end
    end
    return d
end
# collect
data = Dict{String, Any}()
for (box, _) in BOXES
    data[box] = (
        ob = parse_cache("bench/plots_data_$(box).txt"),
        aocl = parse_cache("bench/plots_data_$(box)_aocl.txt"),
    )
end
# all (op,lvl,size) keys
allk = Set{Tuple{String, String, Int}}()
for (box, _) in BOXES, k in keys(data[box].ob)
    push!(allk, k)
end
# report misses per box: min(ob, aocl) < THR
THR = 1.0
println("=== GATE MISSES (median PB/baseline < $THR); 'ob'/'ao' = which baseline PB loses to ===")
byop = Dict{String, Vector{String}}()
for (op, lvl, sz) in sort(collect(allk), by = x -> (x[2], x[1], x[3]))
    cells = String[]
    anymiss = false
    for (box, lab) in BOXES
        ob = get(data[box].ob, (op, lvl, sz), NaN)
        ao = get(data[box].aocl, (op, lvl, sz), NaN)
        worst = minimum(filter(!isnan, [ob, ao]); init = Inf)
        if worst < THR
            anymiss = true
            tag = (!isnan(ob) && ob < THR ? @sprintf("ob%.2f", ob) : "") * (!isnan(ao) && ao < THR ? @sprintf("/ao%.2f", ao) : "")
            push!(cells, @sprintf("%s:%s", replace(lab, "/AVX-512" => "·512", "/AVX2" => "·A2"), tag))
        end
    end
    if anymiss
        line = @sprintf("%-4s %-8s n=%-5d  %s", lvl, op, sz, join(cells, "  "))
        push!(get!(byop, op, String[]), line)
    end
end
# print grouped by op, ordered by number of misses (worst first)
for op in sort(collect(keys(byop)), by = o -> -length(byop[o]))
    @printf("\n## %s  (%d miss-cells)\n", op, length(byop[op]))
    for l in byop[op]
        println(l)
    end
end

# ---- SUMMARY: per op, miss-cell count, worst ratio, vs-OB losses (harder) ----
println("\n\n=== SUMMARY (real ops first) ===")
println(rpad("op", 9), rpad("miss", 5), rpad("worst", 7), rpad("vsOB#", 6), "boxes-hit")
rows = Tuple{String, String, Int, Float64, Int, String}[]
for (op, lvl, sz) in allk
end
opstat = Dict{String, Any}()
for (op, lvl, sz) in allk
    worstall = Inf; obloss = 0; boxes = Set{String}()
    for (box, lab) in BOXES
        ob = get(data[box].ob, (op, lvl, sz), NaN); ao = get(data[box].aocl, (op, lvl, sz), NaN)
        w = minimum(filter(!isnan, [ob, ao]); init = Inf)
        if w < 1.0
            worstall = min(worstall, w); push!(boxes, split(lab, '/')[1]); (!isnan(ob)&&ob < 1.0)&&(obloss += 1)
        end
    end
    if worstall < 1.0
        s = get!(opstat, op, Any[0, Inf, 0, Set{String}(), lvl])
        s[1] += 1; s[2] = min(s[2], worstall); s[3] += obloss; union!(s[4], boxes)
    end
end
for op in sort(collect(keys(opstat)), by = o -> (opstat[o][5], -opstat[o][1], opstat[o][2]))
    s = opstat[op]
    @printf("%-9s%-5d%-7.2f%-6d%s [%s]\n", op, s[1], s[2], s[3], join(sort(collect(s[4])), ","), s[5])
end
@printf(
    "\nTOTAL miss-cells across fleet: %d ; real-only: %d\n",
    sum(s[1] for s in values(opstat)),
    sum(s[1] for (o, s) in opstat if !startswith(o, r"z" == ""[1:0] ? "" : "z"))
)
