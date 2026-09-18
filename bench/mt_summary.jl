# Summarise a MULTI-THREADED sweep: the pb_mt / pb speedup per cell.
#
# WHY THIS EXISTS RATHER THAN REUSING gate_gaps.jl / gate_cycles.jl. Both of those require a VENDOR
# reference arm and report 0 cells without one — correctly, because they adjudicate the gate. An mt
# cache has no reference arm by design: it lives in its own `mt_data_…` file precisely so that a
# 6-core-affinity run can never overwrite the 1-core-pinned `pb` gate cells (see `CACHE` in plots.jl).
# So there was nothing that could read it.
#
# ⚠ WHAT THIS IS NOT. Not a gate. `PB ≥ max(OpenBLAS, AOCL)` is a SINGLE-THREADED criterion, and a
# threaded PureBLAS against a single-threaded OpenBLAS is flattery, not parity. Everything below is a
# SCALING number: how much did N threads buy over 1 thread, on the same box, in the same process, in
# rotated rounds. Nothing here may be quoted as a gate result.
#
# Estimator: MEDIAN, per the project rule — per-round ratio medians, then the median across rounds, the
# same reduction `gate_gaps.jl` uses. The round spread is printed because it is the precision of the
# thing doing the measuring, and at least one cell needs it (gemm@512 alternates by 40% with arm order).
#
# Usage:  julia --project=bench bench/mt_summary.jl bench/mt_data_<uarch>_<host>.txt [more...]
#         julia --project=bench bench/mt_summary.jl --all bench/mt_data_*.txt    # every cell, not just movers
#
# ALSO A LIBRARY. `bench/gen_threading.jl` includes this file to reuse `cells`, `roundratios`, `med` and
# `mt_rows` rather than keeping a second copy of the cache parser — the pattern `test/perthread_lint.jl`
# uses. The report below runs only when this file is executed directly.

using Printf
include(joinpath(@__DIR__, "freqgate.jl"))     # _freq_ref / _freq_of / _freq_offlock

const QN = 48
# A cell whose speedup is within this of 1.0 is "flat" — not threaded, or too small to pay the join.
const FLAT = 0.05

med(v) = sort(v)[max(1, cld(length(v), 2))]
# Per-round ratio medians: chunk both arms' stored quantile vectors into rounds of QN and divide
# elementwise. pb / pb_mt, so > 1 means threading paid. Same orientation as the gate's reference ratio.
function roundratios(qpb, qmt)
    n = min(length(qpb), length(qmt)) ÷ QN
    return [med([qpb[(r - 1) * QN + i] / qmt[(r - 1) * QN + i] for i in 1:QN]) for r in 1:n]
end

function cells(path)
    out = []
    ref = 0
    offlock = 0
    for ln in eachline(path)
        startswith(ln, "#pbbench") && ((ref, _) = _freq_ref(ln); continue)
        (isempty(strip(ln)) || startswith(ln, "#")) && continue
        p = split(ln, "\t"); length(p) >= 4 || continue
        d = Dict{String, Vector{Float64}}()
        drift = false
        for f in p[4:end]
            _p = split(f, "|"); a, csv = _p[1], _p[end]
            _freq_offlock(_freq_of(_p), ref) && (drift = true)
            d[String(a)] = parse.(Float64, split(csv, ","))
        end
        # An off-lock arm corrupts a ratio exactly as much here as in the gate: a floating clock between
        # the two windows IS the thing being measured otherwise. Count them, never average them in.
        drift && (offlock += 1; continue)
        push!(out, (String(p[1]), String(p[2]), parse(Int, p[3]), d))
    end
    return out, offlock
end

"""
    mt_header(path) -> Dict

The `#pbbench` provenance line as key/value pairs: host, uarch, commit, time, freq, anchor, …
"""
function mt_header(path)
    hdr = first(Iterators.filter(l -> startswith(l, "#pbbench"), eachline(path)))
    return Dict(split(f, "=", limit = 2)[1] => split(f, "=", limit = 2)[end]
                for f in split(hdr, "\t") if occursin("=", f))
end

"""
    mt_rows(path) -> (rows, stats)

`rows` is `(speedup, spread, lvl, op, n)` for every cell carrying BOTH a `pb` and a `pb_mt` arm, sorted
fastest first. `stats` counts what was left out and why — cells, off-lock, no-mt, flat — because a
summary that silently drops cells reads as coverage it does not have.

`no_mt` is expected to be non-zero: the dual groups pass `refs=["generic"]`, which REPLACES the arm
list, so they never carry a `pb_mt` arm. Dual element types cannot reach the threaded path anyway.
"""
function mt_rows(path; flat_band = FLAT)
    rows, offlock = cells(path)
    movers = Tuple{Float64, Float64, String, String, Int}[]
    flat = 0; no_mt = 0
    for (lvl, op, sz, d) in rows
        (haskey(d, "pb") && haskey(d, "pb_mt")) || (no_mt += 1; continue)
        rr = roundratios(d["pb"], d["pb_mt"]); isempty(rr) && continue
        s = med(rr); spread = (maximum(rr) - minimum(rr)) / s
        abs(s - 1) <= flat_band && (flat += 1)
        push!(movers, (s, spread, lvl, op, sz))
    end
    sort!(movers; by = first, rev = true)
    return movers, (cells = length(rows), offlock = offlock, no_mt = no_mt, flat = flat)
end

function main(paths, show_all)
    for path in paths
    isfile(path) || (println("missing: ", path); continue)
    kv = mt_header(path)
    println("\n══ ", basename(path), "   host=", get(kv, "host", "?"), " uarch=", get(kv, "uarch", "?"),
        " commit=", get(kv, "commit", "?"), " time=", get(kv, "time", "?"))
    allrows, st = mt_rows(path)
    iszero(st.cells) && (println("   no adjudicable cells (", st.offlock, " off-lock)"); continue)
    movers = show_all ? allrows : [r for r in allrows if abs(r[1] - 1) > FLAT]
    offlock = st.offlock; missing_mt = st.no_mt; flat = st.flat
    @printf("   %d cells, %d off-lock, %d without a pb_mt arm, %d flat (within %.0f%% of 1.00)\n",
        st.cells, offlock, missing_mt, flat, 100 * FLAT)
    @printf("\n   %-5s %-10s %7s %9s %9s\n", "lvl", "op", "n", "speedup", "spread")
    for (s, sp, lvl, op, sz) in movers
        @printf("   %-5s %-10s %7d %9.2f %8.1f%%\n", lvl, op, sz, s, 100 * sp)
    end
    # The headline is the BEST speedup per op, because a ladder that climbs is the diagnosis: a ratio
    # that falls with n means the routine is not amortising the join.
    println()
    best = Dict{String, Tuple{Float64, Int}}()
    for (s, _, lvl, op, sz) in movers
        k = string(lvl, " ", op)
        (!haskey(best, k) || s > best[k][1]) && (best[k] = (s, sz))
    end
    isempty(best) || println("   best per op: ", join((@sprintf("%s %.2f@%d", k, v[1], v[2]) for (k, v) in sort(collect(best); by = x -> -x[2][1])), "  "))
    end
    return nothing
end

# Report only when RUN, not when included — `bench/gen_threading.jl` includes this file for its parser.
if abspath(PROGRAM_FILE) == @__FILE__
    main([a for a in ARGS if !startswith(a, "--")], "--all" in ARGS)
end
