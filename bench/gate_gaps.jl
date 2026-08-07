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
    out = []                                   # (lvl, op, size, Dict(arm => times), Dict(arm => commit))
    for ln in eachline(path)
        (isempty(strip(ln)) || startswith(ln, "#")) && continue
        p = split(ln, "\t"); length(p) >= 4 || continue
        d = Dict{String, Vector{Float64}}()
        c = Dict{String, String}()
        for f in p[4:end]
            _p = split(f, "|"); a, com, csv = _p[1], _p[3], _p[end]
            d[String(a)] = parse.(Float64, split(csv, ","))
            c[String(a)] = String(com)
        end
        push!(out, (String(p[1]), String(p[2]), parse(Int, p[3]), d, c))
    end
    return out
end

# PROVENANCE. Every arm carries the commit it was measured at, and a cell whose `pb` arm predates HEAD
# is evidence about code that no longer exists. Twice on 2026-08-06 a cell was read as a verdict on a
# change that the cached arm was measured BEFORE — once quoting pre-tiling numbers as post-tiling. The
# cache said so both times, in a field nothing displayed. `stale` marks it in the row itself; the check
# is per-ARM because `arms=pb` deliberately leaves the reference arms old, and that is not staleness.
#
# STALE MEANS "src/ MOVED", NOT "the hash differs". A hash test flags every row in the file the moment
# you commit anything — 133 of 133 on the first run of this — and a flag that is always on is a flag
# nobody reads. `git diff --name-only <c>..HEAD -- src/` asks the question that actually matters: is the
# measured code still the shipping code? A cell measured five doc commits ago is not stale.
#
# AND IT NAMES THE FILES, because a boolean is still too coarse to act on. Measured immediately after
# shipping the boolean version: every BLAS-1 row on both boxes came up stale against `fd56167`, while
# the only src commits since it were `937e906` (iamax), `457b2a7`/`5de2bde` (potrf upper) and a
# net-zero revert — nothing that can touch `scal`/`axpy`/`dot`/`asum`. "Something moved" would have had
# me re-sweep three boxes for ~8 hours to re-derive numbers that were already valid. The file list makes
# that judgement possible without a per-op→per-file mapping nobody would maintain: the reader decides
# relevance, which is the part a tool should not be guessing at.
const HEAD = try
    readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`)
catch
    "unknown"
end
const _SRCDIFF = Dict{String, Vector{String}}()
function srcdiff(c)
    (c in ("?", "unknown", HEAD)) && return String[]
    return get!(_SRCDIFF, c) do
        try
            fs = readchomp(`git -C $(@__DIR__) diff --name-only $c..HEAD -- ../src`)
            isempty(fs) ? String[] : [replace(x, "src/" => "") for x in split(fs, "\n")]
        catch
            String[]                           # commit not in this clone (fleet cache) — can't say, don't cry wolf
        end
    end
end
srcmoved(c) = !isempty(srcdiff(c))

med(v) = sort(v)[max(1, cld(length(v), 2))]
# per-round ratio medians: chunk both arms' stored vectors into rounds of QN and divide elementwise
function roundratios(qref, qpb)
    n = min(length(qref), length(qpb)) ÷ QN
    return [med([qref[(r - 1) * QN + i] / qpb[(r - 1) * QN + i] for i in 1:QN]) for r in 1:n]
end

rows = []
for path in ARGS, (lvl, op, sz, d, com) in cells(path)
    haskey(d, "pb") || continue
    refs = [r for r in ("openblas", "aocl") if haskey(d, r)]
    isempty(refs) && continue
    best = ""; worst = Inf; spread = 0.0
    per = Dict{String, Float64}()              # EVERY reference, not just the binding one — see below
    for r in refs                              # the gate is vs the FASTER reference at THIS cell
        rr = roundratios(d[r], d["pb"]); isempty(rr) && continue
        m = med(rr); per[r] = m
        m < worst && (worst = m; best = r; spread = (maximum(rr) - minimum(rr)) / m)
    end
    pbc = get(com, "pb", "?")
    isfinite(worst) && push!(rows, (worst, spread, lvl, op, sz, best, basename(path), per, pbc))
end

sort!(rows; by = first)
fails = [r for r in rows if r[1] < 1.0]
println("cells=", length(rows), "  below 1.0=", length(fails))
# BOTH REFERENCES ARE PRINTED, not just the binding one. The `ratio`/`vs` columns are the gate (worst
# against the faster reference) — but reporting only that HIDES WHICH LIBRARY BINDS, and that is the
# fact which tells you whether a caller inherits its callee's gap. Measured 2026-08-06, Zen3 n=32:
#     zgemm    OB 0.995  AOCL 0.807   <- AOCL binds
#     ztrmmR   OB 0.779  AOCL 1.377   <- OpenBLAS binds, and we beat AOCL by 38%
#     ztrmm    OB 0.824  AOCL 1.453   <- OpenBLAS binds
# If the triangular ops merely inherited zgemm's deficit they would lose to the SAME reference. They do
# not, so that is OB's better small-n triangular handling — a separate mechanism, not a gemm problem.
# Reading only the worst-of column, I twice assumed inheritance that was not there. A one-sided gap
# (we crush one reference, lose to the other) also usually means the loser implemented something the
# winner did not, which changes what "close this cell" even means.
println("HEAD=$HEAD   (STALE(c) = the pb arm was measured at c, and src/ has changed since. Check the \
legend below:\n             if nothing in that file list can reach this op, the number still stands.)")
println("\n  gap    ratio  spread  cell                          vs        vs_OB   vs_AOCL  cache")
for (ratio, spread, lvl, op, sz, ref, f, per, pbc) in fails
    # a miss smaller than the cell's own round-to-round spread is not distinguishable from noise
    tag = (1.0 - ratio) <= spread ? "  <- within spread" : ""
    srcmoved(pbc) && (tag *= "  <- STALE($pbc)")
    fmt(r) = haskey(per, r) ? rpad(round(per[r]; digits = 3), 8) : rpad("—", 8)
    println(rpad(string(round(100 * (1 - ratio); digits = 1), "%"), 7),
        rpad(round(ratio; digits = 3), 7), rpad(round(spread; digits = 3), 8),
        rpad("$lvl $op@$sz", 30), rpad(ref, 10), fmt("openblas"), fmt("aocl"),
        replace(f, "plots_data_" => "", ".txt" => ""), tag)
end
nnoise = count(r -> (1.0 - r[1]) <= r[2], fails)
println("\n$(length(fails) - nnoise) of $(length(fails)) misses exceed their own round spread; ",
    "$nnoise are within it (lower bound on noise — process-level variation is larger).")

stale = sort!(unique(r[9] for r in fails if srcmoved(r[9])))
if !isempty(stale)
    println("\nSTALE legend — src/ files changed between each measuring commit and HEAD=$HEAD.")
    for c in stale
        fs = srcdiff(c)
        println("  $c → ", length(fs) <= 6 ? join(fs, ", ") : "$(join(first(fs, 6), ", ")) … +$(length(fs) - 6) more")
    end
end
