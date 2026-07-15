# Generates the performance plots embedded in docs/src/performance.md: per-op PureBLAS/OpenBLAS ratio
# (single-thread, Float64). BLAS-1/2 as VIOLINS (ratio distribution over the size sweep); BLAS-3/LAPACK as
# ratio-vs-size TREND lines on a log-y axis (their ratio has a strong size dependence — small n is
# overhead-bound, large n gates). Hand-written SVG — no plotting dependency (keeps the bench env light,
# matches the pure/minimal ethos). Each (op,size) is measured over repeated rounds; per round the OB and
# PB windows run consecutively (ABBA-alternated) and are reconciled by `_qratios`, then the per-round ratios
# are POOLED (median = gate). Repetition rejects the one-unlucky-window failure single windows are prone to.
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
# Reference BLAS: OpenBLAS (default), Intel MKL (`mkl` arg), or AMD AOCL (`aocl` arg). Each package
# LBT-forwards LinearAlgebra's BLAS+LAPACK to itself on load, so `B`/`LAPACK` below transparently measure
# against whichever is active — one code path, three baselines. AOCL (AMD-tuned BLIS + libFLAME) is a
# SEPARATE baseline from OpenBLAS: its caches/SVGs carry an `_aocl` suffix and never mix with OpenBLAS's.
const REFBK = "aocl" in ARGS ? "aocl" : "mkl" in ARGS ? "mkl" : "openblas"
REFBK == "mkl" && @eval using MKL
# AOCL = AMD's Zen-tuned AOCL-BLIS + AOCL-libFLAME, shipped as the `AOCL_jll` artifact (AMD's own release,
# NOT generic blis_jll/libflame_jll). We LBT-forward its artifact .so paths directly (BLAS→libblis-mt,
# LAPACK→libflame), which is exactly what the AOCL.jl wrapper does — using the JLL keeps the dep to the
# reproducible binary artifact. `libblis-mt` is a multi-thread build; pin to 1 thread for a fair
# single-thread comparison (BLIS reads these at init; BLAS.set_num_threads(1) below re-enforces via LBT).
if REFBK == "aocl"
    ENV["BLIS_NUM_THREADS"] = "1"; ENV["OMP_NUM_THREADS"] = "1"
    @eval using AOCL_jll
    LinearAlgebra.BLAS.lbt_forward(AOCL_jll.aocl_blas_ilp64; clear = true)   # ILP64 BLAS  → libblis-mt.so
    LinearAlgebra.BLAS.lbt_forward(AOCL_jll.aocl_lapack_ilp64)               # ILP64 LAPACK → libflame.so
end
const REFNAME = REFBK == "mkl" ? "MKL" : REFBK == "aocl" ? "AOCL" : "OpenBLAS"
# cache/SVG filename suffix: "" for OpenBLAS (the default baseline — its artefacts are UNTOUCHED), "_mkl"/"_aocl" otherwise
const REFSUF = REFBK == "openblas" ? "" : "_$REFBK"
import LinearAlgebra.BLAS as B
BLAS.set_num_threads(1)

# ── ISA / µarch identity (derived once, up here so `save_cache` can STAMP it into the cache header). A
# later multi-host plot loads several `plots_data_<host>.txt` and must tell Zen4/Zen3/Zen5 apart — the
# filenames are bare hostnames and the SIMD width alone can't (Zen4 & Zen5 are both AVX-512). Same-ISA
# boxes disambiguate via `slug=`/`isa=` CLI overrides (e.g. neuromancer runs `slug=zen5 isa=Zen5`). ─────
const _BENCH_VERSION = 2   # v2 = pooled per-round ratios; v1 = single-window. Bump ⇒ old caches refused.
const _W64P = PureBLAS._vwidth(Float64)
# µarch slug DERIVED from CPU detection (CLAUDE.md req#7 — not a manual flag), so Zen4 vs Zen5 (both
# AVX-512) disambiguate on their own: Zen4 is double-pumped 512, Zen5 is native. Override stays as an
# escape hatch (`slug=`/`isa=`) for an unknown box. This fixes the "run Zen5 without slug=zen5 → mislabel".
const _ISAOVR = (i = findfirst(a -> startswith(a, "isa="), ARGS); isnothing(i) ? nothing : ARGS[i][5:end])
const _SLUGOVR = (i = findfirst(a -> startswith(a, "slug="), ARGS); isnothing(i) ? nothing : ARGS[i][6:end])
const _HWB = PureBLAS._HW
const _AUTOSLUG = _W64P == 8 ? (PureBLAS._double_pumped(_HWB) ? "avx512" : "zen5") :   # Zen4 dp-512 vs Zen5 native
                  _W64P == 4 ? "avx2" : _W64P == 2 ? "neon" : "simd"
const _AUTOISA = _W64P == 8 ? (PureBLAS._double_pumped(_HWB) ? "AVX-512" : "Zen5") :
                 _W64P == 4 ? "AVX2" : _W64P == 2 ? "NEON" : "SIMD"
const ISA = isnothing(_ISAOVR) ? _AUTOISA : _ISAOVR
const _SLUGB = isnothing(_SLUGOVR) ? _AUTOSLUG : _SLUGOVR
const SLUG = "$(_SLUGB)$(REFSUF)"
# Provenance stamped into every cache header (self-describing: which CPU, what code, when measured).
const _CPUNAME = replace(strip(Sys.cpu_info()[1].model), r"[\t\r\n]" => " ")   # e.g. "AMD Ryzen 9 7950X …"
const _COMMIT = try readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`) catch; "unknown" end

# Iteration / robustness modes (for a fast dev loop — full `bench` remains the trustworthy artifact):
#   bench lite       → few rounds + small sizes, ~1–2 min smoke (NOT gate numbers; cache is *_lite.txt)
#   bench op=gemm    → measure ONLY that op, full methodology, MERGE into the (v2) cache
#   bench group=L3   → measure ONLY that level, merge
const _LITE = "lite" in ARGS
const _NODRAW = "nodraw" in ARGS   # fleet boxes: measure + cache only, skip SVG/table render (so their
                                    # working tree stays clean → `git pull` never blocks). Render centrally.
const _SELOP  = (i = findfirst(a -> startswith(a, "op="), ARGS);    isnothing(i) ? nothing : ARGS[i][4:end])
const _SELGRP = (i = findfirst(a -> startswith(a, "group="), ARGS); isnothing(i) ? nothing : ARGS[i][7:end])
_want(lvl, nm) = (isnothing(_SELOP) && isnothing(_SELGRP)) || _SELOP == nm || _SELGRP == lvl
_cap(szs, maxn) = Tuple(s for s in szs if s <= maxn)   # per-op size cap (e.g. skip 4096 for slow ops)
# lite caps sizes at 1024 (drops the expensive 2048/4096 tail) — keeps the meaningful mid-n range while
# skipping the O(n³) large-n sink that dominates wall time. Guarded so a cap never yields an empty tuple.
_sizes(szs) = _LITE ? (t = Tuple(s for s in szs if s <= 1024); isempty(t) ? szs[1:1] : t) : szs

# Repeated rounds reject the one-unlucky-window failure (gemm n=32 read 0.83 in a single window vs 1.01
# true). Keyed on SIZE, deterministic (never on measured duration → identical protocol on every host).
# CRUCIAL: mid-size heavy windows (n=512–1024) are SAMPLES-capped, not seconds-capped, so a single window
# there is exactly the unlucky-window regime — repeat 8×. Only n≥2048 fills a 2 s seconds-bound window;
# still keep 2 rounds there so ABBA order-balance applies (windows are hottest/most order-biased there).
_rounds_light(_sz) = _LITE ? 2 : 8
_rounds_heavy(sz) = _LITE ? 1 : (sz <= 1024 ? 8 : 4)   # n≥2048 was 2 → under-replicated (noise at 4096); 4

# Measure one op ROBUSTLY: skip if filtered out; a per-op try/catch means one op's failure logs and the
# sweep CONTINUES (never all-or-nothing); flush so a run is live-monitorable despite Julia's block-buffered
# file IO. `sweeper` is a thunk returning the per-size ratio vector list.
const _MISSING = String[]   # ops that threw during measurement (surfaced at the end, not just scrolled past)
function _meas!(vec, lvl, nm, sweeper)
    _want(lvl, nm) || return
    print(stderr, "  [$lvl $nm] "); flush(stderr)
    try
        push!(vec, nm => sweeper()); println(stderr, "done"); flush(stderr)
    catch e
        e isa InterruptException && rethrow()   # let Ctrl-C actually stop the run
        push!(_MISSING, "$lvl/$nm"); println(stderr, "FAILED: ", sprint(showerror, e)); flush(stderr)
    end
    return
end

# name => per-size samples: [(size, [ratio,ratio,…]), …]. The single data model the renderer consumes.
const OpData = Pair{String,Vector{Tuple{Int,Vector{Float64}}}}
geomin(op) = (m = [median(v) for (s, v) in op]; (exp(sum(log, m) / length(m)), minimum(m)))  # geomean, worst
# Hermitian-positive-definite operand for (z)potrf, memoized per (T,size): the O(n³) `A*A'` is built ONCE;
# each sample gets a fresh O(n²) copy (potrf is destructive). Avoids an OpenBLAS gemm + 2 big allocs PER
# sample (seconds of wasted setup at n=4096). No `+zeros` — `A*A'+sI` is already dense HPD.
const _HPD = Dict{Tuple{DataType,Int},Any}()
_hpd(T, s) = copy(get!(() -> (A = randn(T, s, s); A * A' + s * I), _HPD, (T, s)))::Matrix{T}

# Every Chairmarks sample time (seconds) — it reports min but stores all timings; we use the full set.
_times(b) = Float64[smp.time for smp in b.samples]
# Quantile-paired ratio distribution: q-th quantile of OB time ÷ q-th quantile of PB time. q=0.5 IS the
# median ratio (the gate number); the spread across q gives the violin body. Robust to unequal counts.
_qratios(bo, bp) = (to = _times(bo); tp = _times(bp); qs = range(0.03, 0.97; length = 48);
    [quantile(to, q) / quantile(tp, q) for q in qs])

# L1/L2 sweep: `_rounds_light(s)` rounds of consecutive ABBA-ordered OB/PB `@be` windows per size, pooling
# the per-round `_qratios`. `evals=1` reruns the setup `mk(s)` per sample so
# address/alignment varies (essential for iamax — OpenBLAS idamax swings ~60% by address) and the mk
# allocation is EXCLUDED from the timed core; `reps` amortizes the timer for tiny ops. All sample timings
# feed the ratio distribution.
function sweep(mk, sizes, work_ob, work_pb, repfn; samples = 400, seconds = 0.15)
    out = Tuple{Int,Vector{Float64}}[]
    for s in sizes
        reps = repfn(s)
        rounds = _rounds_light(s); acc = Float64[]; rmeds = Float64[]
        for r in 1:rounds
            # ABBA: alternate which side runs first each round → cancels the "2nd window runs warmer" bias.
            # `_qratios` per round keeps each OB/PB pairing temporally tight; POOL the ratios (never times).
            if isodd(r)
                bo = @be mk(s) (c -> work_ob(c, reps)) evals=1 samples=samples seconds=seconds
                bp = @be mk(s) (c -> work_pb(c, reps)) evals=1 samples=samples seconds=seconds
            else
                bp = @be mk(s) (c -> work_pb(c, reps)) evals=1 samples=samples seconds=seconds
                bo = @be mk(s) (c -> work_ob(c, reps)) evals=1 samples=samples seconds=seconds
            end
            qr = _qratios(bo, bp); append!(acc, qr); push!(rmeds, median(qr))
        end
        rounds > 1 && (println(stderr, "    n=$s rounds: ", join((@sprintf("%.3f", m) for m in rmeds), " ")); flush(stderr))
        push!(out, (s, acc))
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
        rounds = _rounds_heavy(s)
        secs = s >= 1024 ? 2.0 : seconds   # large-n windows are seconds-bound; 2.0 is plenty and ~halves cost
        acc = Float64[]; rmeds = Float64[]
        for r in 1:rounds
            if isodd(r)   # ABBA order alternation (fires only where rounds>1; n≥512 uses 1 round)
                bo = @be [mk(s) for _ in 1:reps] (cs -> (v = 0.0; for c in cs; v += ob1(c); end; v)) evals=1 samples=samples seconds=secs
                bp = @be [mk(s) for _ in 1:reps] (cs -> (v = 0.0; for c in cs; v += pb1(c); end; v)) evals=1 samples=samples seconds=secs
            else
                bp = @be [mk(s) for _ in 1:reps] (cs -> (v = 0.0; for c in cs; v += pb1(c); end; v)) evals=1 samples=samples seconds=secs
                bo = @be [mk(s) for _ in 1:reps] (cs -> (v = 0.0; for c in cs; v += ob1(c); end; v)) evals=1 samples=samples seconds=secs
            end
            qr = _qratios(bo, bp); append!(acc, qr); push!(rmeds, median(qr))
        end
        rounds > 1 && (println(stderr, "    n=$s rounds: ", join((@sprintf("%.3f", m) for m in rmeds), " ")); flush(stderr))
        push!(out, (s, acc))
    end
    return out
end

const L1SZ = (1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000)
const L2SZ = (64, 128, 256, 512, 1024, 2048, 4096)
const L3SZ = (8, 32, 128, 256, 512, 1024, 2048, 4096)   # O(n³); 4096 shows large-n syrk/trmm behavior
const LPSZ = (8, 32, 128, 256, 512, 1024, 2048, 4096)   # LAPACK factorizations, to 4096
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
        _meas!(l1, "L1", nm, () -> sweep(s -> (randn(s), randn(s)), _sizes(L1SZ), ob, pb, _L1REP))
    end
end

# ── BLAS-2 ──────────────────────────────────────────────────────────────────────────────────────
l2 = OpData[]
let
    sq(s) = (randn(s, s), randn(s), randn(s))
    pk(s) = (randn((s * (s + 1)) ÷ 2), randn(s), randn(s))
    bd(s) = (k = 16; (randn(2k + 1, s), randn(s), randn(s), k))
    sbd(s) = (k = 16; (randn(k + 1, s), randn(s), randn(s), k))
    add(nm, mk, ob, pb) = _meas!(l2, "L2", nm, () -> sweep(mk, _sizes(L2SZ), ob, pb, _L2REP))
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
    addh(nm, mk, ob, pb) = _meas!(l3, "L3", nm, () -> sweep_heavy(mk, ob, pb, _sizes(L3SZ)))
    addh("gemm", s -> (randn(s, s), randn(s, s), zeros(s, s)),
        c -> (B.gemm!(NN, NN, 1.0, c[1], c[2], 0.0, c[3]); c[3][1]),
        c -> (PureBLAS.gemm!(c[3], c[1], c[2]); c[3][1]))
    # (zgemm is measured once, in the complex CL3 group — not duplicated here.)
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
    addh("trsmR", s -> (tri(s), randn(s, s)),   # side-R lower-T (the potrf/getrf panel-solve shape) — the lever
        c -> (B.trsm!('R', 'L', 'T', 'N', 1.0, c[1], c[2]); c[2][1]),
        c -> (PureBLAS.trsm!(c[2], c[1]; side = 'R', uplo = 'L', transA = 'T'); c[2][1]))
end

# ── LAPACK (O(n³) factorizations; all destructive → fresh input per round) ─────────────────────────
lp = OpData[]
let
    LP = Char(76)
    addh(nm, mk, ob, pb; sizes = LPSZ) = _meas!(lp, "LP", nm, () -> sweep_heavy(mk, ob, pb, _sizes(sizes); samples = 40))
    addh("potrf", s -> _hpd(Float64, s),
        c -> (LinearAlgebra.LAPACK.potrf!(LP, c); c[1, 1]),
        c -> (PureBLAS.potrf!(c; uplo = LP); c[1, 1]))
    addh("geqrf", s -> randn(s, s),
        c -> (LinearAlgebra.LAPACK.geqrf!(c); c[1, 1]),
        c -> (PureBLAS.geqrf!(c); c[1, 1]))
    addh("getrf", s -> randn(s, s),
        c -> (LinearAlgebra.LAPACK.getrf!(c); c[1, 1]),
        c -> (PureBLAS.getrf!(c); c[1, 1]))
    # real gesvd capped at 2048: OB gesdd is divide-and-conquer, PB is QR-iteration — at 4096 with vectors
    # that algorithm mismatch dominates (no actionable signal) and a single sample isn't seconds-bounded.
    addh("gesvd", s -> randn(s, s),
        c -> (LinearAlgebra.LAPACK.gesdd!(Char(65), c); c[1, 1]),
        c -> (PureBLAS.gesvd!(c; want_vectors = true); 0.0); sizes = _cap(LPSZ, 2048))
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
            _meas!(cl1, "CL1", nm, () -> sweep(s -> (randn(T, s), randn(T, s)), _sizes(L1SZ), ob, pb, _L1REP))
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
        add(nm, mk, ob, pb) = _meas!(cl2, "CL2", nm, () -> sweep(mk, _sizes(L2SZ), ob, pb, _L2REP))
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
        addh(nm, mk, ob, pb) = _meas!(cl3, "CL3", nm, () -> sweep_heavy(mk, ob, pb, _sizes(_cap(L3SZ, 2048))))  # complex 4096 is a ~10min sink for little signal → cap at 2048 (real L3 keeps 4096)
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
        addh(nm, mk, ob, pb; sizes = LPSZ) = _meas!(clp, "CLP", nm, () -> sweep_heavy(mk, ob, pb, _sizes(_cap(sizes, 2048)); samples = 40))  # cap complex LAPACK at 2048 (zgesvd's 1024 cap survives via nested _cap)
        addh("zpotrf", s -> _hpd(T, s),
            c -> (LinearAlgebra.LAPACK.potrf!(LP, c); real(c[1, 1])),
            c -> (PureBLAS.potrf!(c; uplo = LP); real(c[1, 1])))
        addh("zgeqrf", s -> randn(T, s, s),
            c -> (LinearAlgebra.LAPACK.geqrf!(c); real(c[1, 1])),
            c -> (PureBLAS.geqrf!(c); real(c[1, 1])))
        addh("zgetrf", s -> randn(T, s, s),
            c -> (LinearAlgebra.LAPACK.getrf!(c); real(c[1, 1])),
            c -> (PureBLAS.getrf!(c); real(c[1, 1])))
        # zgesvd now on the BLOCKED complex bidiag (zlabrd panels + gemm trailing) → gates; capped at 2048
        # (the group cap) like the other complex LAPACK ops.
        addh("zgesvd", s -> randn(T, s, s),
            c -> (LinearAlgebra.LAPACK.gesdd!(Char(78), c); real(c[1, 1])),   # 'N' — singular values only
            c -> (PureBLAS.gesvd!(c; want_vectors = false); 0.0))
    end
    return cl1, cl2, cl3, clp
end

# ── cache: one line per op  «level⟶TAB⟶name⟶TAB⟶ s1=r,r,…;s2=r,r,… » ─────────────────────────────
const CACHE = joinpath(@__DIR__, "plots_data_$(gethostname())$(REFSUF)$(_LITE ? "_lite" : "").txt")
function save_cache(path, groups)
    open(path, "w") do io
        # header stamps the methodology version (so old numbers can't silently coexist), the µarch identity
        # (slug/isa) for the multi-host plot, and full provenance: CPU model, code commit, measure time.
        ts = Libc.strftime("%Y-%m-%dT%H:%M", time())
        println(io, "#pbbench\tversion=$(_BENCH_VERSION)\tslug=$SLUG\tisa=$ISA\thost=$(gethostname())",
            "\tcpu=$(_CPUNAME)\tcommit=$(_COMMIT)\ttime=$ts")
        for (lvl, d) in groups, (nm, op) in d
            println(io, lvl, "\t", nm, "\t", join(("$(s)=$(join(v, ","))" for (s, v) in op), ";"))
        end
    end
    println("cached ratio data → $path")
end
# Returns (groups, meta::NamedTuple). meta carries version/slug/isa/host from the header (µarch identity
# for the multi-host overlay). Refuses a cache from an older methodology version (forces re-measure).
function load_cache(path)
    g = Dict{String,Vector{OpData}}()
    meta = (version = 1, slug = "?", isa = "?", host = "?", cpu = "?", commit = "?", time = "?")   # legacy ⇒ v1
    for ln in eachline(path)
        isempty(strip(ln)) && continue
        if startswith(ln, "#pbbench")
            kv = Dict(String(p[1]) => String(p[2]) for p in (split(x, "=") for x in split(ln, "\t")[2:end]) if length(p) == 2)
            meta = (version = parse(Int, get(kv, "version", "1")), slug = get(kv, "slug", "?"),
                    isa = get(kv, "isa", "?"), host = get(kv, "host", "?"), cpu = get(kv, "cpu", "?"),
                    commit = get(kv, "commit", "?"), time = get(kv, "time", "?"))
            continue
        end
        lvl, nm, blocks = split(ln, "\t")
        op = Tuple{Int,Vector{Float64}}[]
        for blk in split(blocks, ";")
            sp = split(blk, "="); push!(op, (parse(Int, sp[1]), parse.(Float64, split(sp[2], ","))))
        end
        push!(get!(g, String(lvl), OpData[]), String(nm) => op)
    end
    meta.version == _BENCH_VERSION || error("cache $path is methodology v$(meta.version); this is " *
        "v$(_BENCH_VERSION) (pooled per-round ratios) — re-measure with `bench` (old numbers aren't comparable)")
    return g, meta
end


# ── Cross-µarch panel grid (the redesign): one SVG per group, one PANEL per op, each overlaying the
# fleet's µarchs as ratio-vs-size lines + q10–q90 bands. Size is always the x-axis (cache transitions show
# as steps); the 3 µarchs share a panel so cross-machine comparison is direct; no panel holds >3 lines so
# the old 11-line/8-colour collision is gone. Fixed colour per µarch, keyed on the cache's stamped slug. ──
const _UARCH = Dict("avx512" => ("#1f77b4", "Zen4 · AVX-512"), "zen5" => ("#2ca02c", "Zen5 · AVX-512"),
                    "avx2" => ("#d62728", "Zen3 · AVX2"))
# AOCL/MKL caches stamp slug=<µarch>_<refbk> (e.g. avx512_aocl); _UARCH is keyed on the BARE µarch slug,
# so strip the refbk suffix before the color/label lookup — else every AOCL series fell to the grey fallback.
_baseslug(slug) = replace(slug, r"_(aocl|mkl)$" => "")
_ucolor(slug) = get(_UARCH, _baseslug(slug), ("#888888", slug))[1]
_ulabel(meta) = get(_UARCH, _baseslug(meta.slug), ("#888888", meta.isa))[2]

# Load every fleet cache (plots_data_<host>.txt) → [(meta, groups), …]. In lite mode loads only *_lite; in
# full mode only full caches. Skips MKL. Refuses stale-version caches via load_cache.
function load_fleet()
    fleet = Tuple{NamedTuple,Dict{String,Vector{OpData}}}[]
    for f in sort(readdir(@__DIR__))
        (startswith(f, "plots_data_") && endswith(f, ".txt")) || continue
        (occursin("_aocl", f) ? "aocl" : occursin("_mkl", f) ? "mkl" : "openblas") == REFBK || continue  # baseline ↔ its own caches; never mix
        occursin("_lite", f) == _LITE || continue
        try                                                    # a stale/foreign/half-written cache must NOT
            g, meta = load_cache(joinpath(@__DIR__, f))        # abort the whole fleet render — skip it loudly
            push!(fleet, (meta, g))
        catch e
            @warn "skipping cache $f (stale version or unreadable)" exception = (e, catch_backtrace())
        end
    end
    slugs = [m.slug for (m, _) in fleet]                        # duplicate µarch ⇒ lines overlap + mislabel
    for s in unique(slugs)
        count(==(s), slugs) > 1 && @warn "duplicate µarch slug '$s' across caches — pass slug=/isa= to disambiguate"
    end
    return fleet
end

_opsin(fleet, gk) = (ops = String[]; for (_, g) in fleet, (nm, _) in get(g, gk, OpData[]); (nm in ops) || push!(ops, nm); end; ops)
_series(g, gk, op) = (i = findfirst(p -> p.first == op, get(g, gk, OpData[])); isnothing(i) ? nothing : get(g, gk, OpData[])[i].second)

function svg_panels(path, title, fleet, gk)
    ops = _opsin(fleet, gk); isempty(ops) && return
    ncol = min(4, length(ops)); nrow = cld(length(ops), ncol)
    pw = 210; ph = 138; ml = 46; mt = 60; gx = 20; gy = 34; pad = 16
    W = ml + ncol * pw + (ncol - 1) * gx + pad
    H = mt + nrow * (ph + gy) + pad
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="sans-serif">""")
    println(io, """<rect width="$W" height="$H" fill="white"/>""")
    println(io, """<text x="$(W/2)" y="26" text-anchor="middle" font-size="17" font-weight="bold">$title</text>""")
    lx = ml
    for (meta, _) in fleet   # legend
        col = _ucolor(meta.slug); lab = _ulabel(meta)
        println(io, """<line x1="$lx" y1="42" x2="$(lx+22)" y2="42" stroke="$col" stroke-width="3"/>""")
        println(io, """<text x="$(lx+27)" y="46" font-size="12">$lab</text>""")
        lx += 27 + 7 * length(lab) + 26
    end
    for (k, op) in enumerate(ops)
        px = ml + ((k - 1) % ncol) * (pw + gx); py = mt + ((k - 1) ÷ ncol) * (ph + gy)
        series = Tuple{String,Vector{Tuple{Int,Vector{Float64}}}}[]; allsz = Int[]
        for (meta, g) in fleet
            ps = _series(g, gk, op); (isnothing(ps) || isempty(ps)) && continue
            push!(series, (meta.slug, ps)); for (s, _) in ps; (s in allsz) || push!(allsz, s); end
        end
        (isempty(series) || isempty(allsz)) && continue
        sort!(allsz)
        xlo = log2(minimum(allsz)); xsp = max(log2(maximum(allsz)) - xlo, 1e-9)
        # y-range from the band extremes (q10/q90), not just medians, so noisy bands don't saturate flat
        ext = Float64[]; for (_, ps) in series, (_, v) in ps; push!(ext, quantile(v, 0.10), median(v), quantile(v, 0.90)); end
        yhi = max(1.6, 1.08 * maximum(ext)); ylo = min(0.5, 0.93 * minimum(ext)); L = log
        xof(s) = px + pw * (log2(s) - xlo) / xsp
        yof(r) = py + ph * (1 - (L(clamp(r, ylo, yhi)) - L(ylo)) / (L(yhi) - L(ylo)))
        println(io, """<rect x="$px" y="$py" width="$pw" height="$ph" fill="none" stroke="#e2e2e2"/>""")
        for (rr, cc, da) in ((1.0, "#d33", """ stroke-dasharray="4 3\""""),)   # gate = parity = 1.0×
            (rr < ylo || rr > yhi) && continue
            println(io, """<line x1="$px" y1="$(yof(rr))" x2="$(px+pw)" y2="$(yof(rr))" stroke="$cc"$da/>""")
        end
        for r in unique(round.([ylo, 1.0, yhi], digits = 2))
            (r < ylo || r > yhi) && continue
            println(io, """<text x="$(px-4)" y="$(yof(r)+3)" text-anchor="end" font-size="9" fill="#999">$(r)×</text>""")
        end
        for (slug, ps) in series
            col = _ucolor(slug)
            bhi = ["$(round(xof(s),digits=1)),$(round(yof(quantile(v,0.90)),digits=1))" for (s, v) in ps]
            blo = ["$(round(xof(s),digits=1)),$(round(yof(quantile(v,0.10)),digits=1))" for (s, v) in reverse(ps)]
            println(io, """<polygon points="$(join(vcat(bhi,blo)," "))" fill="$col" opacity="0.11"/>""")
            ln = ["$(round(xof(s),digits=1)),$(round(yof(median(v)),digits=1))" for (s, v) in ps]
            println(io, """<polyline points="$(join(ln," "))" fill="none" stroke="$col" stroke-width="1.6"/>""")
            for (s, v) in ps; println(io, """<circle cx="$(round(xof(s),digits=1))" cy="$(round(yof(median(v)),digits=1))" r="2" fill="$col"/>"""); end
        end
        println(io, """<text x="$(px+pw/2)" y="$(py-5)" text-anchor="middle" font-size="12" font-weight="bold">$op</text>""")
        # x-axis: a tick + label at every measured size (≥1024 abbreviated as k so they fit the narrow panel)
        for s in allsz
            x = round(xof(s), digits = 1); lbl = s >= 1024 ? "$(s ÷ 1024)k" : "$s"
            println(io, """<line x1="$x" y1="$py" x2="$x" y2="$(py+ph)" stroke="#f4f4f4"/>""")
            println(io, """<text x="$x" y="$(py+ph+11)" text-anchor="middle" font-size="8" fill="#999">$lbl</text>""")
        end
    end
    println(io, "</svg>"); write(path, String(take!(io))); println("wrote $path")
end

# Drift-proof numeric companion to the hand-annotated narrative table: geomean (worst) per op per µarch.
function gen_table(fleet, gkeys)
    io = IOBuffer()
    println(io, "| op | ", join((_ulabel(m) for (m, _) in fleet), " | "), " |")
    println(io, "|---|", repeat("---|", length(fleet)))
    for gk in gkeys, op in _opsin(fleet, gk)
        cells = String[]
        for (_, g) in fleet
            ps = _series(g, gk, op)
            if isnothing(ps) || isempty(ps)
                push!(cells, "–")
            else
                geo, mn = geomin(ps); push!(cells, @sprintf("%.2f (%.2f)", geo, mn))
            end
        end
        println(io, "| `$op` | ", join(cells, " | "), " |")
    end
    String(take!(io))
end

# ── measure (and cache) or load from cache, then draw ────────────────────────────────────────────
if "plot" in ARGS
    isfile(CACHE) || error("no cache at $CACHE — run without `plot` first to measure")
    g, _meta = load_cache(CACHE); println("loaded cached data ← $CACHE")
elseif !("bench" in ARGS) && isfile(CACHE)
    g, _meta = load_cache(CACHE); println("loaded cached data ← $CACHE  (pass `bench` to re-measure)")
else
    l1, l2, l3, lp = run_benchmarks()
    cl1, cl2, cl3, clp = run_cmplx_benchmarks()
    measured = Dict("L1" => l1, "L2" => l2, "L3" => l3, "LP" => lp, "CL1" => cl1, "CL2" => cl2, "CL3" => cl3, "CLP" => clp)
    subset = !isnothing(_SELOP) || !isnothing(_SELGRP)
    if subset
        # subset re-measure: MERGE the measured op(s) into the existing (v2) cache, leaving the rest intact.
        isfile(CACHE) || error("subset re-measure (op=/group=) needs an existing full cache at $CACHE — run a full `bench` first")
        g, meta = load_cache(CACHE)   # load_cache refuses a non-v2 cache
        meta.slug == SLUG || error("subset slug ($SLUG) ≠ cache slug ($(meta.slug)) — merging would relabel the µarch; re-run full `bench`")
        for (lvl, ops) in measured, (nm, r) in ops
            gl = get!(g, lvl, OpData[]); filter!(p -> p.first != nm, gl); push!(gl, nm => r)
        end
        println("merged $(sum(length(ops) for ops in values(measured))) re-measured op(s) into $CACHE")
    else
        g = measured
    end
    save_cache(CACHE, [lvl => get(g, lvl, OpData[]) for lvl in ("L1", "L2", "L3", "LP", "CL1", "CL2", "CL3", "CLP")])
end

adir = joinpath(@__DIR__, "..", "docs", "src", "assets"); mkpath(adir)
# Draw the whole FLEET (every host cache on disk) as cross-µarch panel grids: 8 SVGs, NO per-host suffix
# (a 3-line panel IS the per-host view). One SVG per group. `nodraw` skips this (fleet boxes measure only).
fleet = _NODRAW ? [] : load_fleet()
if isempty(fleet)
    println("no fleet caches on disk to plot")
else
    L = _LITE ? "_lite" : ""; ref = REFNAME
    for (gk, base, ttl) in (("L1", "l1", "BLAS-1"), ("L2", "l2", "BLAS-2"), ("L3", "l3", "BLAS-3"),
                            ("LP", "lapack", "LAPACK"), ("CL1", "cl1", "Complex BLAS-1"),
                            ("CL2", "cl2", "Complex BLAS-2"), ("CL3", "cl3", "Complex BLAS-3"),
                            ("CLP", "clapack", "Complex LAPACK"))
        svg_panels(joinpath(adir, "perf_$(base)$(REFSUF)$L.svg"), "$ttl — PureBLAS / $ref (PB/$ref ratio)", fleet, gk)
    end
    open(joinpath(@__DIR__, "gen_table$(REFSUF)$L.md"), "w") do io   # drift-proof numeric table: geomean (worst) per op/µarch
        println(io, "_Measured (provenance):_\n")
        for (m, _) in fleet   # self-describing: CPU, code commit, measure time per µarch
            println(io, "- **$(_ulabel(m))** (`$(m.host)`) — $(m.cpu), commit `$(m.commit)`, $(m.time)")
        end
        println(io, "\n### Real\n\n", gen_table(fleet, ["L1", "L2", "L3", "LP"]))
        println(io, "\n### Complex\n\n", gen_table(fleet, ["CL1", "CL2", "CL3", "CLP"]))
    end
    println("wrote gen_table$L.md  (fleet: ", join((m.slug for (m, _) in fleet), ", "), ")")
end
# gate summary for THIS host's just-measured/loaded data
for lvl in ("L1", "L2", "L3", "LP", "CL1", "CL2", "CL3", "CLP"), (nm, op) in get(g, lvl, OpData[])
    geo, mn = geomin(op)
    @printf("%-3s %-8s geomean=%.2f  worst=%.2f  %s\n", lvl, nm, geo, mn, mn >= 1.0 ? "PASS" : "FAIL")
end
isempty(_MISSING) || @warn "these ops FAILED during measurement (absent from the cache/plots): $(join(_MISSING, ", "))"
