# Generates the performance plots embedded in docs/src/performance.md: per-op PureBLAS/OpenBLAS ratio
# (single-thread, Float64). BLAS-1/2 as VIOLINS (ratio distribution over the size sweep); BLAS-3/LAPACK as
# ratio-vs-size TREND lines on a log-y axis (their ratio has a strong size dependence — small n is
# overhead-bound, large n gates). Hand-written SVG — no plotting dependency (keeps the bench env light,
# matches the pure/minimal ethos). Interleaved-paired methodology, same as the other bench scripts.
#
# Measured ratio samples are CACHED per-size to bench/plots_data_<host>.txt so the (slow) benchmark runs
# once and re-plotting (styling tweaks) is instant. Usage (pinned):
#   taskset -c 2 julia --project=bench bench/plots.jl          # use cache if present, else measure + cache
#   taskset -c 2 julia --project=bench bench/plots.jl bench    # force re-measure (refresh the cache)
#   julia --project=bench bench/plots.jl plot                  # plot from cache only (never measure)
#   taskset -c 2 julia --project=bench bench/plots.jl bench mkl # reference = Intel MKL instead of OpenBLAS
#                                                              # (Haswell target: `]add MKL` first; MKL uses
#                                                              #  its native Haswell kernels. On AMD MKL
#                                                              #  throttles to a generic path — Intel only.)
using PureBLAS, LinearAlgebra, Statistics, Printf
using Chairmarks: @be   # robust per-side timing (auto sample-sizing + warmup); replaces hand-rolled time_ns
# Reference BLAS: OpenBLAS (default) or Intel MKL (`mkl` arg). MKL.jl LBT-forwards LinearAlgebra to MKL,
# so `B`/`LAPACK` below transparently measure against whichever is active — one code path, two baselines.
const REFBK = "mkl" in ARGS ? "mkl" : "openblas"
REFBK == "mkl" && @eval using MKL
const REFNAME = REFBK == "mkl" ? "MKL" : "OpenBLAS"
import LinearAlgebra.BLAS as B
BLAS.set_num_threads(1)
const SINK = Ref(0.0); @noinline _run(f) = f()

# name => per-size samples: [(size, [ratio,ratio,…]), …]. One data model feeds both plot types.
const OpData = Pair{String,Vector{Tuple{Int,Vector{Float64}}}}
sizemeds(op) = [(s, median(v)) for (s, v) in op]                    # per-size median ratio
pooled(op) = reduce(vcat, (v for (_, v) in op); init = Float64[])   # all samples (for violins)
geomin(op) = (m = [median(v) for (_, v) in op]; (exp(sum(log, m) / length(m)), minimum(m)))

# Every Chairmarks sample time (seconds) — it reports min but stores all timings; we use the full set.
_times(b) = Float64[smp.time for smp in b.samples]
# Quantile-paired ratio distribution: q-th quantile of OB time ÷ q-th quantile of PB time. q=0.5 IS the
# median ratio (the gate number); the spread across q gives the violin body. Robust to unequal counts.
_qratios(bo, bp) = (to = _times(bo); tp = _times(bp); qs = range(0.03, 0.97; length = 48);
    [quantile(to, q) / quantile(tp, q) for q in qs])

# L1/L2 sweep: one interleaved OB/PB `@be` per size. `evals=1` reruns the setup `mk(s)` per sample so
# address/alignment varies (essential for iamax — OpenBLAS idamax swings ~60% by address) and the mk
# allocation is EXCLUDED from the timed core; `reps` amortizes the timer for tiny ops. All sample timings
# feed the ratio distribution.
function sweep(mk, sizes, work_ob, work_pb, repfn; samples = 400, seconds = 0.5)
    out = Tuple{Int,Vector{Float64}}[]
    for s in sizes
        reps = repfn(s)
        bo = @be mk(s) (c -> work_ob(c, reps)) evals=1 samples=samples seconds=seconds
        bp = @be mk(s) (c -> work_pb(c, reps)) evals=1 samples=samples seconds=seconds
        push!(out, (s, _qratios(bo, bp)))
    end
    return out
end
const _L1REP = s -> clamp(8_000_000 ÷ s, 30, 20000)           # O(s) work
const _L2REP = s -> clamp(400_000_000 ÷ (s * s), 30, 20000)   # O(s²) work

# Heavy O(n³) sweep for L3 / LAPACK. `@be` with `evals=1` runs a FRESH `mk(s)` per sample (the destructive
# op mutates its input → one op per context) and EXCLUDES the mk allocation from the timed core — which
# removes the old hazard where the per-round alloc dropped the core off-clock and biased whichever side was
# timed first. Warmup + sample sizing are Chairmarks'. Small n is measured cleanly (no timer-quantization
# reps hack needed — Chairmarks amortizes internally).
# ⚠ STILL REQUIRES CPU BOOST DISABLED (`echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost`,
# performance governor) so the fixed clock keeps OB vs PB comparable. See memory dev-fleet.
function sweep_heavy(mk, ob1, pb1, sizes; samples = 64, seconds = 4.0)
    out = Tuple{Int,Vector{Float64}}[]
    for s in sizes
        # reps fresh contexts per sample: setup (EXCLUDED from timing) pre-generates them, the core runs the
        # destructive op on each. reps→1 at large n; large at tiny n so a ~50 ns n=8 op isn't measured as a
        # single sub-timer-resolution call (which fabricated n=8 "fails" — evals=1 alone can't amortize it).
        reps = clamp(20_000_000 ÷ (s * s * s), 1, 512)
        bo = @be [mk(s) for _ in 1:reps] (cs -> (v = 0.0; for c in cs; v += ob1(c); end; v)) evals=1 samples=samples seconds=seconds
        bp = @be [mk(s) for _ in 1:reps] (cs -> (v = 0.0; for c in cs; v += pb1(c); end; v)) evals=1 samples=samples seconds=seconds
        push!(out, (s, _qratios(bo, bp)))
    end
    return out
end

const L1SZ = (1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000)
const L2SZ = (64, 128, 256, 512, 1024, 2048, 4096)
const L3SZ = (8, 32, 128, 256, 512, 1024, 2048)   # O(n³); small n included — the 2–2048 gate campaign
const LPSZ = (8, 32, 128, 256, 512, 1024, 2048)   # LAPACK factorizations
const TN = Char(78); const TT = Char(84); const U = Char(85)

# Run the full benchmark sweep; returns (l1, l2, l3, lp) as vectors of OpData.
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

# ── BLAS-3 (O(n³), destructive trmm/trsm; fresh input per round) ──────────────────────────────────
l3 = OpData[]
let
    NN = Char(78); LT = Char(76); UP = Char(85)
    tri(s) = (A = randn(s, s) ./ (2s); for i in 1:s; A[i, i] = 1 + abs(A[i, i]); end; A)
    addh(nm, mk, ob, pb) = push!(l3, nm => sweep_heavy(mk, ob, pb, L3SZ))
    addh("gemm", s -> (randn(s, s), randn(s, s), zeros(s, s)),
        c -> (B.gemm!(NN, NN, 1.0, c[1], c[2], 0.0, c[3]); c[3][1]),
        c -> (PureBLAS.gemm!(c[3], c[1], c[2]); c[3][1]))
    addh("zgemm", s -> (randn(ComplexF64, s, s), randn(ComplexF64, s, s), zeros(ComplexF64, s, s)),
        c -> (B.gemm!(NN, NN, 1.0 + 0im, c[1], c[2], 0.0 + 0im, c[3]); real(c[3][1])),
        c -> (PureBLAS.gemm!(c[3], c[1], c[2]); real(c[3][1])))
    addh("symm", s -> (randn(s, s), randn(s, s), zeros(s, s)),
        c -> (B.symm!(LT, UP, 1.0, c[1], c[2], 0.0, c[3]); c[3][1]),
        c -> (PureBLAS.symm!(c[3], c[1], c[2]; side = LT, uplo = UP); c[3][1]))
    addh("syrk", s -> (randn(s, s), zeros(s, s)),
        c -> (B.syrk!(UP, NN, 1.0, c[1], 0.0, c[2]); c[2][1]),
        c -> (PureBLAS.syrk!(c[2], c[1]; uplo = UP, trans = NN); c[2][1]))
    addh("syr2k", s -> (randn(s, s), randn(s, s), zeros(s, s)),
        c -> (B.syr2k!(UP, NN, 1.0, c[1], c[2], 0.0, c[3]); c[3][1]),
        c -> (PureBLAS.syr2k!(c[3], c[1], c[2]; uplo = UP, trans = NN); c[3][1]))
    addh("trmm", s -> (tri(s), randn(s, s)),
        c -> (B.trmm!(LT, UP, NN, NN, 1.0, c[1], c[2]); c[2][1]),
        c -> (PureBLAS.trmm!(c[2], c[1]; side = LT, uplo = UP); c[2][1]))
    addh("trsm", s -> (tri(s), randn(s, s)),
        c -> (B.trsm!(LT, UP, NN, NN, 1.0, c[1], c[2]); c[2][1]),
        c -> (PureBLAS.trsm!(c[2], c[1]; side = LT, uplo = UP); c[2][1]))
end

# ── LAPACK (O(n³) factorizations; all destructive → fresh input per round) ─────────────────────────
lp = OpData[]
let
    LP = Char(76)
    addh(nm, mk, ob, pb) = push!(lp, nm => sweep_heavy(mk, ob, pb, LPSZ; samples = 40))
    addh("potrf", s -> (A = randn(s, s); A * A' + s * I + zeros(s, s)),
        c -> (LinearAlgebra.LAPACK.potrf!(LP, c); c[1, 1]),
        c -> (PureBLAS.potrf!(c; uplo = LP); c[1, 1]))
    addh("geqrf", s -> randn(s, s),
        c -> (LinearAlgebra.LAPACK.geqrf!(c); c[1, 1]),
        c -> (PureBLAS.geqrf!(c); c[1, 1]))
    addh("getrf", s -> randn(s, s),
        c -> (LinearAlgebra.LAPACK.getrf!(c); c[1, 1]),
        c -> (PureBLAS.getrf!(c); c[1, 1]))
    addh("gesvd", s -> randn(s, s),
        c -> (LinearAlgebra.LAPACK.gesdd!(Char(65), c); c[1, 1]),
        c -> (PureBLAS.gesvd!(c; want_vectors = true); 0.0))
end
    return l1, l2, l3, lp
end

# ── Complex (ComplexF64) surface: the M5 complex-SIMD work. Same methodology; separate plot family so the
# real (Float64) plots stay clean. L1/L2 violins, L3 trend. Oracle = OpenBLAS/MKL complex BLAS. ────────
function run_cmplx_benchmarks()
    T = ComplexF64; TC = Char(67)
    ca = one(T); cb = zero(T)
    cl1 = OpData[]
    let
        for (nm, ob, pb) in (
            ("zaxpy", (c, m) -> (for _ in 1:m; B.axpy!(1.7 + 0.3im, c[1], c[2]); end; real(c[2][1])),
                      (c, m) -> (for _ in 1:m; PureBLAS.axpy!(c[2], 1.7 + 0.3im, c[1]); end; real(c[2][1]))),
            ("zdotc", (c, m) -> (s = zero(T); for _ in 1:m; s += B.dotc(c[1], c[2]); end; real(s)),
                      (c, m) -> (s = zero(T); for _ in 1:m; s += PureBLAS.dot(c[1], c[2]); end; real(s))),
            ("zscal", (c, m) -> (for _ in 1:m; B.scal!(1.0000001 + 0im, c[1]); end; real(c[1][1])),
                      (c, m) -> (for _ in 1:m; PureBLAS.scal!(1.0000001 + 0im, c[1]); end; real(c[1][1]))),
            ("dznrm2", (c, m) -> (s = 0.0; for _ in 1:m; s += B.nrm2(c[1]); end; s),
                       (c, m) -> (s = 0.0; for _ in 1:m; s += PureBLAS.nrm2(c[1]); end; s)),
            ("dzasum", (c, m) -> (s = 0.0; for _ in 1:m; s += B.asum(c[1]); end; s),
                       (c, m) -> (s = 0.0; for _ in 1:m; s += PureBLAS.asum(c[1]); end; s)),
            ("zdotu", (c, m) -> (s = zero(T); for _ in 1:m; s += B.dotu(c[1], c[2]); end; real(s)),
                      (c, m) -> (s = zero(T); for _ in 1:m; s += PureBLAS.dotu(c[1], c[2]); end; real(s))),
            ("izamax", (c, m) -> (s = 0; for _ in 1:m; s += B.iamax(c[1]); end; s),
                       (c, m) -> (s = 0; for _ in 1:m; s += PureBLAS.iamax(c[1]); end; s)),
        )
            push!(cl1, nm => sweep(s -> (randn(T, s), randn(T, s)), L1SZ, ob, pb, _L1REP))
        end
    end
    cl2 = OpData[]
    let
        sq(s) = (randn(T, s, s), randn(T, s), randn(T, s))
        herm(s) = (A = randn(T, s, s); A = A + A'; for i in 1:s; A[i, i] = real(A[i, i]); end; (A, randn(T, s), randn(T, s)))
        tri(s) = (A = randn(T, s, s); for i in 1:s; A[i, i] = 1 + abs(A[i, i]); end; (A, randn(T, s), randn(T, s)))
        cpk(s) = (randn(T, (s * (s + 1)) ÷ 2), randn(T, s), randn(T, s))          # Hermitian packed (hpmv)
        cbd(s) = (k = 16; (randn(T, 2k + 1, s), randn(T, s), randn(T, s), k))      # general banded (gbmv)
        csbd(s) = (k = 16; (randn(T, k + 1, s), randn(T, s), randn(T, s), k))      # Hermitian banded (hbmv)
        add(nm, mk, ob, pb) = push!(cl2, nm => sweep(mk, L2SZ, ob, pb, _L2REP))
        add("zgemvN", sq, (c, m) -> (for _ in 1:m; B.gemv!(TN, ca, c[1], c[2], cb, c[3]); end; real(c[3][1])),
            (c, m) -> (for _ in 1:m; PureBLAS.gemv!(c[3], c[1], c[2]; alpha = ca, beta = cb); end; real(c[3][1])))
        add("zgemvT", sq, (c, m) -> (for _ in 1:m; B.gemv!(TT, ca, c[1], c[2], cb, c[3]); end; real(c[3][1])),
            (c, m) -> (for _ in 1:m; PureBLAS.gemv!(c[3], c[1], c[2]; alpha = ca, beta = cb, trans = TT); end; real(c[3][1])))
        add("zgemvC", sq, (c, m) -> (for _ in 1:m; B.gemv!(TC, ca, c[1], c[2], cb, c[3]); end; real(c[3][1])),
            (c, m) -> (for _ in 1:m; PureBLAS.gemv!(c[3], c[1], c[2]; alpha = ca, beta = cb, trans = TC); end; real(c[3][1])))
        add("zgeru", sq, (c, m) -> (for _ in 1:m; B.geru!(ca, c[2], c[3], c[1]); end; real(c[1][1])),
            (c, m) -> (for _ in 1:m; PureBLAS.ger!(ca, c[2], c[3], c[1]); end; real(c[1][1])))
        add("zhemv", herm, (c, m) -> (for _ in 1:m; B.hemv!(U, ca, c[1], c[2], cb, c[3]); end; real(c[3][1])),
            (c, m) -> (for _ in 1:m; PureBLAS.hemv!(c[3], c[1], c[2]; uplo = U, alpha = ca, beta = cb); end; real(c[3][1])))
        add("ztrmv", tri, (c, m) -> (for _ in 1:m; copyto!(c[3], c[2]); B.trmv!(U, TN, TN, c[1], c[3]); end; real(c[3][1])),
            (c, m) -> (for _ in 1:m; copyto!(c[3], c[2]); PureBLAS.trmv!(c[1], c[3]; uplo = U); end; real(c[3][1])))
        add("ztrsv", tri, (c, m) -> (for _ in 1:m; copyto!(c[3], c[2]); B.trsv!(U, TN, TN, c[1], c[3]); end; real(c[3][1])),
            (c, m) -> (for _ in 1:m; copyto!(c[3], c[2]); PureBLAS.trsv!(c[1], c[3]; uplo = U); end; real(c[3][1])))
        add("zhpmv", cpk, (c, m) -> (for _ in 1:m; B.hpmv!(U, ca, c[1], c[2], cb, c[3]); end; real(c[3][1])),
            (c, m) -> (for _ in 1:m; PureBLAS.hpmv!(c[3], c[1], c[2]; uplo = U, alpha = ca, beta = cb); end; real(c[3][1])))
        add("zgbmvN", cbd, (c, m) -> (for _ in 1:m; B.gbmv!(TN, length(c[2]), c[4], c[4], ca, c[1], c[2], cb, c[3]); end; real(c[3][1])),
            (c, m) -> (n = length(c[2]); for _ in 1:m; PureBLAS.gbmv!(c[3], c[1], c[2], n, c[4], c[4]; trans = TN, alpha = ca, beta = cb); end; real(c[3][1])))
        add("zhbmv", csbd, (c, m) -> (for _ in 1:m; B.hbmv!(U, c[4], ca, c[1], c[2], cb, c[3]); end; real(c[3][1])),
            (c, m) -> (for _ in 1:m; PureBLAS.hbmv!(c[3], c[1], c[2]; uplo = U, alpha = ca, beta = cb); end; real(c[3][1])))
    end
    cl3 = OpData[]
    let
        NN = TN; LT = Char(76); RT = Char(82); UP = U; TC = Char(67)
        ctri(s) = (A = randn(T, s, s) ./ (2s); for i in 1:s; A[i, i] = 1 + abs(A[i, i]); end; A)
        cherm(s) = (A = randn(T, s, s); A = A + A'; for i in 1:s; A[i, i] = real(A[i, i]); end; A)
        addh(nm, mk, ob, pb) = push!(cl3, nm => sweep_heavy(mk, ob, pb, L3SZ))
        addh("zgemm", s -> (randn(T, s, s), randn(T, s, s), zeros(T, s, s)),
            c -> (B.gemm!(NN, NN, ca, c[1], c[2], cb, c[3]); real(c[3][1])),
            c -> (PureBLAS.gemm!(c[3], c[1], c[2]); real(c[3][1])))
        addh("zhemm", s -> (cherm(s), randn(T, s, s), zeros(T, s, s)),
            c -> (B.hemm!(LT, UP, ca, c[1], c[2], cb, c[3]); real(c[3][1])),
            c -> (PureBLAS.hemm!(c[3], c[1], c[2]; side = LT, uplo = UP, alpha = ca, beta = cb); real(c[3][1])))
        addh("zsymm", s -> (randn(T, s, s), randn(T, s, s), zeros(T, s, s)),
            c -> (B.symm!(LT, UP, ca, c[1], c[2], cb, c[3]); real(c[3][1])),
            c -> (PureBLAS.symm!(c[3], c[1], c[2]; side = LT, uplo = UP, alpha = ca, beta = cb); real(c[3][1])))
        addh("zsyrk", s -> (randn(T, s, s), zeros(T, s, s)),
            c -> (B.syrk!(UP, NN, ca, c[1], cb, c[2]); real(c[2][1])),
            c -> (PureBLAS.syrk!(c[2], c[1]; uplo = UP, trans = NN, alpha = ca, beta = cb); real(c[2][1])))
        addh("zherk", s -> (randn(T, s, s), zeros(T, s, s)),
            c -> (B.herk!(UP, NN, 1.0, c[1], 0.0, c[2]); real(c[2][1])),
            c -> (PureBLAS.herk!(c[2], c[1]; uplo = UP, trans = NN, alpha = 1.0, beta = 0.0); real(c[2][1])))
        addh("zher2k", s -> (randn(T, s, s), randn(T, s, s), zeros(T, s, s)),   # were UNPLOTTED (like side-R)
            c -> (B.her2k!(UP, NN, ca, c[1], c[2], 0.0, c[3]); real(c[3][1])),
            c -> (PureBLAS.her2k!(c[3], c[1], c[2]; uplo = UP, trans = NN, alpha = ca, beta = 0.0); real(c[3][1])))
        addh("zsyr2k", s -> (randn(T, s, s), randn(T, s, s), zeros(T, s, s)),
            c -> (B.syr2k!(UP, NN, ca, c[1], c[2], cb, c[3]); real(c[3][1])),
            c -> (PureBLAS.syr2k!(c[3], c[1], c[2]; uplo = UP, trans = NN, alpha = ca, beta = cb); real(c[3][1])))
        addh("ztrmm", s -> (ctri(s), randn(T, s, s)),
            c -> (B.trmm!(LT, UP, NN, NN, ca, c[1], c[2]); real(c[2][1])),
            c -> (PureBLAS.trmm!(c[2], c[1]; side = LT, uplo = UP); real(c[2][1])))
        addh("ztrsm", s -> (ctri(s), randn(T, s, s)),
            c -> (B.trsm!(LT, UP, NN, NN, ca, c[1], c[2]); real(c[2][1])),
            c -> (PureBLAS.trsm!(c[2], c[1]; side = LT, uplo = UP); real(c[2][1])))
        addh("ztrmmR", s -> (ctri(s), randn(T, s, s)),     # side-R: plots measured only side-L → the 0.24
            c -> (B.trmm!(RT, UP, NN, NN, ca, c[1], c[2]); real(c[2][1])),   # side-R routing bug went unseen
            c -> (PureBLAS.trmm!(c[2], c[1]; side = RT, uplo = UP); real(c[2][1])))
        addh("ztrsmR", s -> (ctri(s), randn(T, s, s)),
            c -> (B.trsm!(RT, UP, NN, NN, ca, c[1], c[2]); real(c[2][1])),
            c -> (PureBLAS.trsm!(c[2], c[1]; side = RT, uplo = UP); real(c[2][1])))
    end
    # ── Complex LAPACK (zpotrf/zgetrf/zgeqrf/zgesvd; destructive → fresh input per round). Mirrors the real
    # `lp` group. zgesvd compares VALUES-ONLY (gesdd 'N' vs PB want_vectors=false) — complex singular VECTORS
    # aren't implemented yet, so this is the honest fair fight for what ships. ─────────────────────────────
    clp = OpData[]
    let
        LP = Char(76)  # 'L'
        hpd(s) = (A = randn(T, s, s); A * A' + s * I + zeros(T, s, s))  # Hermitian positive-definite
        addh(nm, mk, ob, pb) = push!(clp, nm => sweep_heavy(mk, ob, pb, LPSZ; samples = 40))
        addh("zpotrf", s -> hpd(s),
            c -> (LinearAlgebra.LAPACK.potrf!(LP, c); real(c[1, 1])),
            c -> (PureBLAS.potrf!(c; uplo = LP); real(c[1, 1])))
        addh("zgeqrf", s -> randn(T, s, s),
            c -> (LinearAlgebra.LAPACK.geqrf!(c); real(c[1, 1])),
            c -> (PureBLAS.geqrf!(c); real(c[1, 1])))
        addh("zgetrf", s -> randn(T, s, s),
            c -> (LinearAlgebra.LAPACK.getrf!(c); real(c[1, 1])),
            c -> (PureBLAS.getrf!(c); real(c[1, 1])))
        addh("zgesvd", s -> randn(T, s, s),
            c -> (LinearAlgebra.LAPACK.gesdd!(Char(78), c); real(c[1, 1])),   # 'N' — singular values only
            c -> (PureBLAS.gesvd!(c; want_vectors = false); 0.0))
    end
    return cl1, cl2, cl3, clp
end

# ── cache: one line per op  «level⟶TAB⟶name⟶TAB⟶ s1=r,r,…;s2=r,r,… » ─────────────────────────────
const CACHE = joinpath(@__DIR__, "plots_data_$(gethostname())$(REFBK == "mkl" ? "_mkl" : "").txt")
function save_cache(path, groups)
    open(path, "w") do io
        for (lvl, d) in groups, (nm, op) in d
            println(io, lvl, "\t", nm, "\t", join(("$(s)=$(join(v, ","))" for (s, v) in op), ";"))
        end
    end
    println("cached ratio data → $path")
end
function load_cache(path)
    g = Dict{String,Vector{OpData}}()
    for ln in eachline(path)
        isempty(strip(ln)) && continue
        lvl, nm, blocks = split(ln, "\t")
        op = Tuple{Int,Vector{Float64}}[]
        for blk in split(blocks, ";")
            sp = split(blk, "="); push!(op, (parse(Int, sp[1]), parse.(Float64, split(sp[2], ","))))
        end
        push!(get!(g, String(lvl), OpData[]), String(nm) => op)
    end
    return g
end

# ── Gaussian KDE (Silverman bandwidth) — the violin's half-width profile ──────────────────────────
function kde(xs, ys)
    n = length(xs); sd = std(xs); h = max(sd > 0 ? 1.06 * sd * n^(-0.2) : 0.02, 1.5e-3)
    invn = 1.0 / (n * h * sqrt(2π))
    return [invn * sum(x -> (u = (y - x) / h; exp(-0.5 * u * u)), xs) for y in ys]
end

# ── SVG violins (BLAS-1/2): KDE body + median line + worst-size tick, gate at 0.96, parity at 1.0 ──
function svg_violins(path, title, data)
    W = 900; H = 440; ml = 60; mr = 20; mt = 50; mb = 82
    n = length(data); pw = W - ml - mr; ph = H - mt - mb
    samps = [pooled(d.second) for d in data]
    ymax = max(1.6, maximum(maximum(s) for s in samps) * 1.06)
    yof(v) = mt + ph * (1 - v / ymax)
    gap = pw / n; maxw = gap * 0.42; NY = 72
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="sans-serif">""")
    println(io, """<rect width="$W" height="$H" fill="white"/>""")
    println(io, """<text x="$(W/2)" y="28" text-anchor="middle" font-size="18" font-weight="bold">$title</text>""")
    for v in (0.0, 0.5, 1.0, 1.5)
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
        xs = samps[i]; _, mn = geomin(d.second)
        cx = ml + (i - 0.5) * gap
        lo = minimum(xs); hi = maximum(xs); pad = 0.12 * (hi - lo) + 2e-3
        ys = collect(range(max(0.0, lo - pad), min(ymax, hi + pad), length = NY))
        dens = kde(xs, ys); dm = maximum(dens); dm = dm == 0 ? 1.0 : dm
        col = mn >= 0.96 ? "#2a8" : "#d33"
        pts = String[]
        for k in 1:NY; push!(pts, "$(round(cx + maxw*dens[k]/dm, digits=1)),$(round(yof(ys[k]), digits=1))"); end
        for k in NY:-1:1; push!(pts, "$(round(cx - maxw*dens[k]/dm, digits=1)),$(round(yof(ys[k]), digits=1))"); end
        println(io, """<polygon points="$(join(pts, " "))" fill="$col" opacity="0.5" stroke="$col" stroke-width="1"/>""")
        med = median(xs); wmed = maxw * (kde(xs, [med])[1] / dm); ymed = yof(med)
        println(io, """<line x1="$(round(cx-wmed,digits=1))" y1="$ymed" x2="$(round(cx+wmed,digits=1))" y2="$ymed" stroke="#114" stroke-width="2"/>""")
        println(io, """<circle cx="$cx" cy="$(yof(mn))" r="2.4" fill="#114"/>""")
        println(io, """<text x="$cx" y="$(yof(hi)-6)" text-anchor="middle" font-size="11">$(@sprintf("%.2f", med))</text>""")
        println(io, """<text x="$cx" y="$(H-mb+18)" text-anchor="middle" font-size="12">$(d.first)</text>""")
    end
    println(io, """<text x="$(W/2)" y="$(H-12)" text-anchor="middle" font-size="11" fill="#666">violin = ratio distribution over sizes · line = median · dark dot = worst size · green = every size ≥ gate</text>""")
    println(io, "</svg>"); write(path, String(take!(io))); println("wrote $path")
end

# ── SVG trend lines (BLAS-3/LAPACK): median ratio vs size, log-x, log-y ──
function svg_trend(path, title, data)
    W = 900; H = 460; ml = 62; mr = 132; mt = 50; mb = 60; pw = W - ml - mr; ph = H - mt - mb
    allsz = sort(unique(reduce(vcat, [[s for (s, _) in d.second] for d in data])))
    xlo = log2(minimum(allsz)); xhi = log2(maximum(allsz))
    xof(s) = ml + pw * (log2(s) - xlo) / (xhi - xlo)
    yhi = max(1.7, 1.15 * maximum(maximum(m for (_, m) in sizemeds(d.second)) for d in data))
    ylo = 0.5; ln = log
    yof(r) = mt + ph * (1 - (ln(clamp(r, ylo, yhi)) - ln(ylo)) / (ln(yhi) - ln(ylo)))
    pal = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="sans-serif">""")
    println(io, """<rect width="$W" height="$H" fill="white"/>""")
    println(io, """<text x="$(W/2)" y="28" text-anchor="middle" font-size="18" font-weight="bold">$title</text>""")
    for r in (0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)  # y gridlines (log)
        r > yhi && continue
        y = yof(r)
        println(io, """<line x1="$ml" y1="$y" x2="$(ml+pw)" y2="$y" stroke="#eee"/>""")
        println(io, """<text x="$(ml-8)" y="$(y+4)" text-anchor="end" font-size="11" fill="#666">$(r)×</text>""")
    end
    for s in allsz                                   # x ticks (log2 sizes)
        x = xof(s)
        println(io, """<line x1="$x" y1="$mt" x2="$x" y2="$(mt+ph)" stroke="#f4f4f4"/>""")
        println(io, """<text x="$x" y="$(mt+ph+18)" text-anchor="middle" font-size="11" fill="#444">$s</text>""")
    end
    println(io, """<line x1="$ml" y1="$(yof(1.0))" x2="$(ml+pw)" y2="$(yof(1.0))" stroke="#888"/>""")
    yg = yof(0.96)
    println(io, """<line x1="$ml" y1="$yg" x2="$(ml+pw)" y2="$yg" stroke="#d33" stroke-width="1.5" stroke-dasharray="6 4"/>""")
    println(io, """<text x="$(ml+4)" y="$(yg-5)" font-size="11" fill="#d33">0.96× gate</text>""")
    for (i, d) in enumerate(data)
        col = pal[mod1(i, length(pal))]
        # ribbon: 25th–75th-percentile band of the per-size round ratios (run-to-run spread around the
        # median). Upper edge left→right, then lower edge right→left → a closed filled polygon.
        rhi = [(xof(s), yof(quantile(v, 0.75))) for (s, v) in d.second]
        rlo = [(xof(s), yof(quantile(v, 0.25))) for (s, v) in reverse(d.second)]
        band = vcat(rhi, rlo)
        println(io, """<polygon points="$(join(("$(round(x,digits=1)),$(round(y,digits=1))" for (x,y) in band), " "))" fill="$col" opacity="0.14" stroke="none"/>""")
        pts = [(xof(s), yof(m)) for (s, m) in sizemeds(d.second)]
        println(io, """<polyline points="$(join(("$(round(x,digits=1)),$(round(y,digits=1))" for (x,y) in pts), " "))" fill="none" stroke="$col" stroke-width="2"/>""")
        for (s, m) in sizemeds(d.second)
            println(io, """<circle cx="$(round(xof(s),digits=1))" cy="$(round(yof(m),digits=1))" r="3" fill="$col"/>""")
        end
        ly = mt + 6 + (i - 1) * 18                   # legend (right margin)
        println(io, """<line x1="$(ml+pw+14)" y1="$ly" x2="$(ml+pw+34)" y2="$ly" stroke="$col" stroke-width="3"/>""")
        println(io, """<text x="$(ml+pw+38)" y="$(ly+4)" font-size="12">$(d.first)</text>""")
    end
    println(io, """<text x="$(ml+pw/2)" y="$(H-10)" text-anchor="middle" font-size="11" fill="#666">matrix size n (log) · y = median ratio (log) · dashed = the 0.96× gate</text>""")
    println(io, "</svg>"); write(path, String(take!(io))); println("wrote $path")
end

# ── measure (and cache) or load from cache, then draw ────────────────────────────────────────────
if "plot" in ARGS
    isfile(CACHE) || error("no cache at $CACHE — run without `plot` first to measure")
    g = load_cache(CACHE); println("loaded cached data ← $CACHE")
elseif !("bench" in ARGS) && isfile(CACHE)
    g = load_cache(CACHE); println("loaded cached data ← $CACHE  (pass `bench` to re-measure)")
else
    l1, l2, l3, lp = run_benchmarks()
    cl1, cl2, cl3, clp = run_cmplx_benchmarks()
    g = Dict("L1" => l1, "L2" => l2, "L3" => l3, "LP" => lp, "CL1" => cl1, "CL2" => cl2, "CL3" => cl3, "CLP" => clp)
    save_cache(CACHE, ["L1" => l1, "L2" => l2, "L3" => l3, "LP" => lp, "CL1" => cl1, "CL2" => cl2, "CL3" => cl3, "CLP" => clp])
end

adir = joinpath(@__DIR__, "..", "docs", "src", "assets"); mkpath(adir)
# ISA label/slug from the detected SIMD width so per-machine plots are self-labelled and don't collide
# (the docs show AVX-512 and AVX2 side by side). AVX-512 W64=8, AVX2 W64=4, NEON W64=2.
const _W64P = PureBLAS._vwidth(Float64)
# ISA label. Same-ISA machines collide on the auto slug (Zen4 & Zen5 are both AVX-512) → allow `slug=NAME`
# and `isa=LABEL` overrides (e.g. `plots.jl bench slug=zen5 isa="AVX-512 native (Zen5)"`) so the fleet's
# per-µarch plots don't overwrite each other.
const _ISAOVR = (i = findfirst(a -> startswith(a, "isa="), ARGS); i === nothing ? nothing : ARGS[i][5:end])
const _SLUGOVR = (i = findfirst(a -> startswith(a, "slug="), ARGS); i === nothing ? nothing : ARGS[i][6:end])
const ISA = _ISAOVR !== nothing ? _ISAOVR :
    (_W64P == 8 ? "AVX-512" : _W64P == 4 ? "AVX2" : _W64P == 2 ? "NEON" : "SIMD")
const _SLUGB = _W64P == 8 ? "avx512" : _W64P == 4 ? "avx2" : _W64P == 2 ? "neon" : "simd"
const SLUG = _SLUGOVR !== nothing ? _SLUGOVR : (REFBK == "mkl" ? "$(_SLUGB)_mkl" : _SLUGB)
const TITLE = "PureBLAS / $REFNAME ($ISA, 1 thread, Float64)"
svg_violins(joinpath(adir, "perf_l1_$SLUG.svg"), "BLAS-1: $TITLE", g["L1"])
svg_violins(joinpath(adir, "perf_l2_$SLUG.svg"), "BLAS-2: $TITLE", g["L2"])
svg_trend(joinpath(adir, "perf_l3_$SLUG.svg"), "BLAS-3: $TITLE", g["L3"])
svg_trend(joinpath(adir, "perf_lapack_$SLUG.svg"), "LAPACK: $TITLE", g["LP"])
# Complex (ComplexF64) surface — the M5 SIMD complex path (only when the cache has it).
const CTITLE = "PureBLAS / $REFNAME ($ISA, 1 thread, ComplexF64)"
haskey(g, "CL1") && svg_violins(joinpath(adir, "perf_cl1_$SLUG.svg"), "Complex BLAS-1: $CTITLE", g["CL1"])
haskey(g, "CL2") && svg_violins(joinpath(adir, "perf_cl2_$SLUG.svg"), "Complex BLAS-2: $CTITLE", g["CL2"])
haskey(g, "CL3") && svg_trend(joinpath(adir, "perf_cl3_$SLUG.svg"), "Complex BLAS-3: $CTITLE", g["CL3"])
haskey(g, "CLP") && svg_trend(joinpath(adir, "perf_clapack_$SLUG.svg"), "Complex LAPACK: $CTITLE", g["CLP"])
for lvl in ("L1", "L2", "L3", "LP", "CL1", "CL2", "CL3", "CLP"), (nm, op) in get(g, lvl, OpData[])
    geo, mn = geomin(op)
    @printf("%-3s %-8s geomean=%.2f  worst=%.2f  %s\n", lvl, nm, geo, mn, mn >= 0.96 ? "PASS" : "FAIL")
end
