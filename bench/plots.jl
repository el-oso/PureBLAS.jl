# Generates the performance plots embedded in docs/src/performance.md: per-op PureBLAS/OpenBLAS ratio
# (single-thread, Float64) for BLAS-1 and BLAS-2, drawn as VIOLINS over the size sweep. Hand-written SVG —
# no plotting dependency (keeps the bench env light, matches the pure/minimal ethos). Same interleaved-
# paired methodology as the other bench scripts.
#
# The measured ratio samples are CACHED to bench/plots_data_<host>.txt so the (slow) benchmark runs once
# and re-plotting (styling tweaks) is instant. Usage (pinned):
#   taskset -c 2 julia --project=bench bench/plots.jl          # use cache if present, else measure + cache
#   taskset -c 2 julia --project=bench bench/plots.jl bench    # force re-measure (refresh the cache)
#   julia --project=bench bench/plots.jl plot                  # plot from cache only (never measure)
using PureBLAS, LinearAlgebra, Statistics, Printf
import LinearAlgebra.BLAS as B
BLAS.set_num_threads(1)
const SINK = Ref(0.0); @noinline _run(f) = f()

# Sweep an op over `sizes`, pooling every round's interleaved ratio into one sample vector (the violin's
# data). Each round allocates a FRESH context (`mk(s)`) so the sample spans real address/alignment variation
# — essential for `iamax`, where OpenBLAS's `idamax` timing swings ~60% by array address: reusing one buffer
# would freeze the ratio at a single (possibly unlucky) alignment. Timing is interleaved (OpenBLAS then
# PureBLAS back-to-back, drift cancels) and both are warmed on the fresh buffer first (page-fault fairness).
# The gate verdict (geomean, worst) uses the per-SIZE medians — noise-robust, unlike a raw sample min.
# repfn(s) sets reps/size so total work per round stays ~constant (O(s) L1, O(s²) L2).
function sweep(mk, sizes, work_ob, work_pb, repfn; rounds = 20)
    samples = Float64[]; sizemed = Float64[]
    for s in sizes
        reps = repfn(s); rs = Float64[]
        for _ in 1:rounds
            ctx = mk(s)                                              # fresh each round → varied alignment
            _run(() -> work_ob(ctx, 1)); _run(() -> work_pb(ctx, 1))   # warm this buffer
            t0 = time_ns(); v1 = _run(() -> work_ob(ctx, reps)); t1 = time_ns()
            v2 = _run(() -> work_pb(ctx, reps)); t2 = time_ns()
            SINK[] += v1 + v2; push!(rs, (t1 - t0) / (t2 - t1))     # OpenBLAS / PureBLAS
        end
        append!(samples, rs); push!(sizemed, median(rs))
    end
    return (samples, exp(sum(log, sizemed) / length(sizemed)), minimum(sizemed))
end
const _L1REP = s -> clamp(8_000_000 ÷ s, 30, 20000)           # O(s) work
const _L2REP = s -> clamp(400_000_000 ÷ (s * s), 30, 20000)   # O(s²) work

const OpData = Pair{String,Tuple{Vector{Float64},Float64,Float64}}   # name => (ratio samples, geomean, worst)
const L1SZ = (1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000)
const L2SZ = (64, 128, 256, 512, 1024, 2048, 4096)
const TN = Char(78); const TT = Char(84); const U = Char(85)

# Run the full benchmark sweep; returns (l1, l2) as vectors of OpData.
function run_benchmarks()
# ── BLAS-1 ──────────────────────────────────────────────────────────────────────────────────────
l1 = OpData[]
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
l2 = OpData[]
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
    return l1, l2
end

# ── cache: one line per op  «level⟶TAB⟶name⟶TAB⟶geomean⟶TAB⟶worst⟶TAB⟶comma-joined samples» ─────
const CACHE = joinpath(@__DIR__, "plots_data_$(gethostname()).txt")
function save_cache(path, l1, l2)
    open(path, "w") do io
        for (lvl, d) in (("L1", l1), ("L2", l2)), (nm, (xs, geo, mn)) in d
            println(io, lvl, "\t", nm, "\t", geo, "\t", mn, "\t", join(xs, ","))
        end
    end
    println("cached ratio data → $path")
end
function load_cache(path)
    l1 = OpData[]; l2 = OpData[]
    for ln in eachline(path)
        isempty(strip(ln)) && continue
        lvl, nm, geo, mn, xss = split(ln, "\t")
        entry = String(nm) => (parse.(Float64, split(xss, ",")), parse(Float64, geo), parse(Float64, mn))
        push!(lvl == "L1" ? l1 : l2, entry)
    end
    return l1, l2
end

# ── Gaussian KDE (Silverman bandwidth) — the violin's half-width profile ──────────────────────────
function kde(xs, ys)
    n = length(xs); sd = std(xs); h = max(sd > 0 ? 1.06 * sd * n^(-0.2) : 0.02, 1.5e-3)
    invn = 1.0 / (n * h * sqrt(2π))
    return [invn * sum(x -> (u = (y - x) / h; exp(-0.5 * u * u)), xs) for y in ys]
end

# ── SVG violins (KDE body + median line + worst-size tick, gate line at 0.96, parity at 1.0) ──────
function svg_violins(path, title, data)
    W = 900; H = 440; ml = 60; mr = 20; mt = 50; mb = 82
    n = length(data); pw = W - ml - mr; ph = H - mt - mb
    # y-axis top covers the largest sample (nrm2 legitimately ~5× sets the L1 scale), floor 1.6.
    ymax = max(1.6, maximum(maximum(d.second[1]) for d in data) * 1.06)
    yof(v) = mt + ph * (1 - v / ymax)
    gap = pw / n; maxw = gap * 0.42; NY = 72
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
        xs, geo, mn = d.second
        cx = ml + (i - 0.5) * gap
        lo = minimum(xs); hi = maximum(xs); pad = 0.12 * (hi - lo) + 2e-3
        ys = collect(range(max(0.0, lo - pad), min(ymax, hi + pad), length = NY))   # clip top to axis
        dens = kde(xs, ys); dm = maximum(dens); dm = dm == 0 ? 1.0 : dm
        col = mn >= 0.96 ? "#2a8" : "#d33"                    # gate verdict = worst per-size median
        # symmetric body: up the right edge, down the left edge
        pts = String[]
        for k in 1:NY; push!(pts, "$(round(cx + maxw*dens[k]/dm, digits=1)),$(round(yof(ys[k]), digits=1))"); end
        for k in NY:-1:1; push!(pts, "$(round(cx - maxw*dens[k]/dm, digits=1)),$(round(yof(ys[k]), digits=1))"); end
        println(io, """<polygon points="$(join(pts, " "))" fill="$col" opacity="0.5" stroke="$col" stroke-width="1"/>""")
        med = median(xs); wmed = maxw * (kde(xs, [med])[1] / dm)
        ymed = yof(med)
        println(io, """<line x1="$(round(cx-wmed,digits=1))" y1="$ymed" x2="$(round(cx+wmed,digits=1))" y2="$ymed" stroke="#114" stroke-width="2"/>""")  # median
        println(io, """<circle cx="$cx" cy="$(yof(mn))" r="2.4" fill="#114"/>""")                          # worst size
        println(io, """<text x="$cx" y="$(yof(hi)-6)" text-anchor="middle" font-size="11">$(@sprintf("%.2f", med))</text>""")
        println(io, """<text x="$cx" y="$(H-mb+18)" text-anchor="middle" font-size="12">$(d.first)</text>""")
    end
    println(io, """<text x="$(W/2)" y="$(H-12)" text-anchor="middle" font-size="11" fill="#666">violin = ratio distribution over sizes · line = median · dark dot = worst size · green = every size ≥ gate</text>""")
    println(io, "</svg>")
    write(path, String(take!(io)))
    println("wrote $path")
end

# ── measure (and cache) or load from cache, then draw ────────────────────────────────────────────
if "plot" in ARGS                                   # plot-only: cache must exist
    isfile(CACHE) || error("no cache at $CACHE — run without `plot` first to measure")
    l1, l2 = load_cache(CACHE); println("loaded cached data ← $CACHE")
elseif !("bench" in ARGS) && isfile(CACHE)          # default: reuse cache if present
    l1, l2 = load_cache(CACHE); println("loaded cached data ← $CACHE  (pass `bench` to re-measure)")
else                                                # measure fresh and refresh the cache
    l1, l2 = run_benchmarks(); save_cache(CACHE, l1, l2)
end

adir = joinpath(@__DIR__, "..", "docs", "src", "assets"); mkpath(adir)
svg_violins(joinpath(adir, "perf_l1.svg"), "BLAS-1: PureBLAS / OpenBLAS (Zen4, 1 thread, Float64)", l1)
svg_violins(joinpath(adir, "perf_l2.svg"), "BLAS-2: PureBLAS / OpenBLAS (Zen4, 1 thread, Float64)", l2)
for (lvl, d) in (("L1", l1), ("L2", l2)), (nm, (xs, geo, mn)) in d
    @printf("%s %-7s geomean=%.2f  worst=%.2f  %s\n", lvl, nm, geo, mn, mn >= 0.96 ? "PASS" : "FAIL")
end
