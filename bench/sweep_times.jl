# Per-group sweep wall-clock, READ OUT OF A CACHE — no measurement, no clock touched.
# Each pb arm stores `arm|YYYY-mm-ddTHH:MM|commit|…`, so first..last timestamp of a group's cells at the
# given commit(s) brackets that group's measurement span. Survives a crash that loses the sweep log.
# Recorded results and the estimating rule: ../kb/findings/sweep-wall-clock-reference.md
#
#   julia --project=bench bench/sweep_times.jl <cache.txt> <commit> [commit...]

using Dates
const ORDER = ["L1","L2","L3","LP","CL1","CL2","CL3","CLP","DL1","DL2","DL3","DLP"]
function times(path, commits)
    t = Dict{String,Vector{DateTime}}()
    for ln in eachline(path)
        p = split(ln, "\t"); length(p) < 4 && continue
        for f in p[4:end]
            q = split(f, "|"); length(q) > 2 && q[1] == "pb" && q[3] in commits || continue
            push!(get!(t, p[1], DateTime[]), DateTime(q[2], dateformat"yyyy-mm-ddTHH:MM"))
        end
    end
    return t
end

t = times(ARGS[1], Tuple(ARGS[2:end]))
for g in ORDER
    haskey(t, g) || (println(rpad(g, 4), " not measured at these commits"); continue)
    a, b = extrema(t[g])
    println(rpad(g, 4), " ", Dates.format(a, "HH:MM"), "–", Dates.format(b, "HH:MM"), "  ",
        Dates.value(b - a) ÷ 60000, " min")
end
