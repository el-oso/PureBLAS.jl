# Generates the performance plots embedded in docs/src/performance.md: per-op PureBLAS/OpenBLAS ratio
# (single-thread, Float64) for BLAS-1 and BLAS-2. Hand-written SVG — no plotting dependency (keeps the
# bench env light, matches the pure/minimal ethos). Same interleaved-median methodology as the other
# bench scripts. Run pinned:  taskset -c 2 julia --project=bench bench/plots.jl
using PureBLAS, LinearAlgebra, Statistics, Printf
import LinearAlgebra.BLAS as B
BLAS.set_num_threads(1)
const SINK = Ref(0.0); @noinline _run(f) = f()

function paired(ob, pb, rounds)
    _run(ob); _run(pb); o = Float64[]; p = Float64[]
    for _ in 1:rounds
        t0 = time_ns(); v1 = _run(ob); t1 = time_ns(); v2 = _run(pb); t2 = time_ns()
        push!(o, t1 - t0); push!(p, t2 - t1); SINK[] += v1 + v2
    end
    return median(o), median(p)
end
# ratio = OpenBLAS/PureBLAS median over `sizes`; returns (geomean, min) of the per-size ratios.
# repfn(s) sets reps per size — must track the op's work (O(s) for L1, O(s²) for L2) so total work
# per round stays roughly constant instead of exploding at large s.
function sweep(mk, sizes, work_ob, work_pb, repfn)
    rs = Float64[]
    for s in sizes
        ctx = mk(s); reps = repfn(s)
        mo, mp = paired(() -> work_ob(ctx, reps), () -> work_pb(ctx, reps), 15)
        push!(rs, mo / mp)
    end
    return exp(sum(log, rs) / length(rs)), minimum(rs)
end
const _L1REP = s -> clamp(8_000_000 ÷ s, 30, 20000)           # O(s) work
const _L2REP = s -> clamp(400_000_000 ÷ (s * s), 30, 20000)   # O(s²) work

# ── BLAS-1 ──────────────────────────────────────────────────────────────────────────────────────
const L1SZ = (1_000, 10_000, 100_000, 1_000_000)
l1reps(s) = clamp(4_000_000 ÷ s, 2, 20000)
l1 = Pair{String,Tuple{Float64,Float64}}[]
let
    for (nm, ob, pb) in (
        ("axpy", (c, m) -> (for _ in 1:m; B.axpy!(1.7, c[1], c[2]); end; c[2][1]),
                 (c, m) -> (for _ in 1:m; PureBLAS.axpy!(c[2], 1.7, c[1]); end; c[2][1])),
        ("dot", (c, m) -> (s = 0.0; for _ in 1:m; s += B.dot(c[1], c[2]); end; s),
                (c, m) -> (s = 0.0; for _ in 1:m; s += PureBLAS.dot(c[1], c[2]); end; s)),
        ("nrm2", (c, m) -> (s = 0.0; for _ in 1:m; s += B.nrm2(c[1]); end; s),
                 (c, m) -> (s = 0.0; for _ in 1:m; s += PureBLAS.nrm2(c[1]); end; s)),
        ("asum", (c, m) -> (s = 0.0; for _ in 1:m; s += B.asum(c[1]); end; s),
                 (c, m) -> (s = 0.0; for _ in 1:m; s += PureBLAS.asum(c[1]); end; s)),
        ("scal", (c, m) -> (for _ in 1:m; B.scal!(1.0000001, c[1]); end; c[1][1]),
                 (c, m) -> (for _ in 1:m; PureBLAS.scal!(1.0000001, c[1]); end; c[1][1])),
        ("iamax", (c, m) -> (s = 0; for _ in 1:m; s += B.iamax(c[1]); end; s),
                  (c, m) -> (s = 0; for _ in 1:m; s += PureBLAS.iamax(c[1]); end; s)),
    )
        push!(l1, nm => sweep(s -> (randn(s), randn(s)), L1SZ, ob, pb, _L1REP))
    end
end

# ── BLAS-2 ──────────────────────────────────────────────────────────────────────────────────────
const L2SZ = (64, 256, 1024, 4096)
TN = Char(78); TT = Char(84); U = Char(85)
l2 = Pair{String,Tuple{Float64,Float64}}[]
let
    sq(s) = (randn(s, s), randn(s), randn(s))
    pk(s) = (randn((s * (s + 1)) ÷ 2), randn(s), randn(s))
    bd(s) = (k = 16; (randn(2k + 1, s), randn(s), randn(s), k))
    sbd(s) = (k = 16; (randn(k + 1, s), randn(s), randn(s), k))
    add(nm, mk, ob, pb) = push!(l2, nm => sweep(mk, L2SZ, ob, pb, _L2REP))
    add("gemvN", sq, (c, m) -> (for _ in 1:m; B.gemv!(TN, 1.0, c[1], c[2], 0.0, c[3]); end; c[3][1]),
        (c, m) -> (for _ in 1:m; PureBLAS.gemv!(c[3], c[1], c[2]; alpha = 1.0, beta = 0.0); end; c[3][1]))
    add("gemvT", sq, (c, m) -> (for _ in 1:m; B.gemv!(TT, 1.0, c[1], c[2], 0.0, c[3]); end; c[3][1]),
        (c, m) -> (for _ in 1:m; PureBLAS.gemv!(c[3], c[1], c[2]; alpha = 1.0, beta = 0.0, trans = TT); end; c[3][1]))
    add("ger", sq, (c, m) -> (for _ in 1:m; B.ger!(1.0, c[2], c[3], c[1]); end; c[1][1]),
        (c, m) -> (for _ in 1:m; PureBLAS.ger!(1.0, c[2], c[3], c[1]); end; c[1][1]))
    add("symv", sq, (c, m) -> (for _ in 1:m; B.symv!(U, 1.0, c[1], c[2], 0.0, c[3]); end; c[3][1]),
        (c, m) -> (for _ in 1:m; PureBLAS.symv!(c[3], c[1], c[2]; uplo = U, alpha = 1.0, beta = 0.0); end; c[3][1]))
    add("trmv", sq, (c, m) -> (for _ in 1:m; copyto!(c[3], c[2]); B.trmv!(U, TN, TN, c[1], c[3]); end; c[3][1]),
        (c, m) -> (for _ in 1:m; copyto!(c[3], c[2]); PureBLAS.trmv!(c[1], c[3]; uplo = U); end; c[3][1]))
    add("trsv", s -> (A = randn(s, s) ./ (2s); for i in 1:s; A[i, i] = 1 + abs(A[i, i]); end; (A, randn(s), randn(s))),
        (c, m) -> (for _ in 1:m; copyto!(c[3], c[2]); B.trsv!(U, TN, TN, c[1], c[3]); end; c[3][1]),
        (c, m) -> (for _ in 1:m; copyto!(c[3], c[2]); PureBLAS.trsv!(c[1], c[3]; uplo = U); end; c[3][1]))
    add("spmv", pk, (c, m) -> (for _ in 1:m; B.spmv!(U, 1.0, c[1], c[2], 0.0, c[3]); end; c[3][1]),
        (c, m) -> (for _ in 1:m; PureBLAS.spmv!(c[3], c[1], c[2]; uplo = U, alpha = 1.0, beta = 0.0); end; c[3][1]))
    add("gbmvN", bd, (c, m) -> (for _ in 1:m; B.gbmv!(TN, length(c[2]), c[4], c[4], 1.0, c[1], c[2], 0.0, c[3]); end; c[3][1]),
        (c, m) -> (n = length(c[2]); for _ in 1:m; PureBLAS.gbmv!(c[3], c[1], c[2], n, c[4], c[4]; trans = TN, alpha = 1.0, beta = 0.0); end; c[3][1]))
    add("sbmv", sbd, (c, m) -> (for _ in 1:m; B.sbmv!(U, c[4], 1.0, c[1], c[2], 0.0, c[3]); end; c[3][1]),
        (c, m) -> (for _ in 1:m; PureBLAS.sbmv!(c[3], c[1], c[2]; uplo = U, alpha = 1.0, beta = 0.0); end; c[3][1]))
end

# ── SVG bar chart (geomean bar + min tick, gate line at 0.96, parity at 1.0) ─────────────────────
function svg_bars(path, title, data)
    W = 900; H = 420; ml = 60; mr = 20; mt = 50; mb = 70
    n = length(data); pw = W - ml - mr; ph = H - mt - mb
    ymax = max(1.6, maximum(d.second[1] for d in data) * 1.08)
    yof(v) = mt + ph * (1 - v / ymax)
    bw = pw / n * 0.6; gap = pw / n
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="sans-serif">""")
    println(io, """<rect width="$W" height="$H" fill="white"/>""")
    println(io, """<text x="$(W/2)" y="28" text-anchor="middle" font-size="18" font-weight="bold">$title</text>""")
    for v in (0.0, 0.5, 1.0, 1.5)   # y gridlines + labels
        v > ymax && continue
        y = yof(v)
        println(io, """<line x1="$ml" y1="$y" x2="$(W-mr)" y2="$y" stroke="#eee"/>""")
        println(io, """<text x="$(ml-8)" y="$(y+4)" text-anchor="end" font-size="11" fill="#666">$(rstrip(rstrip(string(v),'0'),'.'))×</text>""")
    end
    g96 = yof(0.96); p1 = yof(1.0)
    println(io, """<line x1="$ml" y1="$p1" x2="$(W-mr)" y2="$p1" stroke="#888" stroke-width="1"/>""")
    println(io, """<line x1="$ml" y1="$g96" x2="$(W-mr)" y2="$g96" stroke="#d33" stroke-width="1.5" stroke-dasharray="6 4"/>""")
    println(io, """<text x="$(W-mr)" y="$(g96-5)" text-anchor="end" font-size="11" fill="#d33">0.96× gate</text>""")
    for (i, d) in enumerate(data)
        geo, mn = d.second
        cx = ml + (i - 0.5) * gap; x = cx - bw / 2
        col = mn >= 0.96 ? "#2a8" : "#d33"
        println(io, """<rect x="$x" y="$(yof(geo))" width="$bw" height="$(yof(0.0)-yof(geo))" fill="$col" opacity="0.85"/>""")
        println(io, """<line x1="$(cx-bw/2)" y1="$(yof(mn))" x2="$(cx+bw/2)" y2="$(yof(mn))" stroke="#114" stroke-width="2"/>""")  # min tick
        println(io, """<text x="$cx" y="$(yof(geo)-5)" text-anchor="middle" font-size="11">$(@sprintf("%.2f", geo))</text>""")
        println(io, """<text x="$cx" y="$(H-mb+18)" text-anchor="middle" font-size="12">$(d.first)</text>""")
    end
    println(io, """<text x="$(W/2)" y="$(H-12)" text-anchor="middle" font-size="11" fill="#666">bar = geomean ratio · dark tick = worst size · green = all sizes ≥ gate</text>""")
    println(io, "</svg>")
    write(path, String(take!(io)))
    println("wrote $path")
end

mkpath(joinpath(@__DIR__, "..", "docs", "src", "assets"))
adir = joinpath(@__DIR__, "..", "docs", "src", "assets")
svg_bars(joinpath(adir, "perf_l1.svg"), "BLAS-1: PureBLAS / OpenBLAS (Zen4, 1 thread, Float64)", l1)
svg_bars(joinpath(adir, "perf_l2.svg"), "BLAS-2: PureBLAS / OpenBLAS (Zen4, 1 thread, Float64)", l2)
for (lvl, d) in (("L1", l1), ("L2", l2)), (nm, (geo, mn)) in d
    @printf("%s %-7s geomean=%.2f  min=%.2f  %s\n", lvl, nm, geo, mn, mn >= 0.96 ? "PASS" : "FAIL")
end
