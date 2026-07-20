# Staged screen→full L3 gate bench. Cuts cycle time by (1) a fast SCREEN pass — one representative
# non-power-of-2 size × all variants — running the FULL size sweep only on routines that need it;
# (2) right-sized reps/rounds via a variance check (keeps interleaved-median, never under-samples);
# (3) file/line-buffered output (run with `stdbuf -oL` or read the results file — no grep-in-a-pipe);
# (4) one julia launch covering correctness + perf for all routines.
#
# Methodology is UNCHANGED from the gate: interleaved paired timing (cancels freq drift), MEDIAN (not
# min), BLAS.set_num_threads(1), pin with `taskset -c N`. Gate = our/OB ≥ 0.96.
#
# Usage:  taskset -c 2 julia --project=bench bench/l3bench.jl [screen|full] [routines...]
#   screen (default): k=1536 × all variants, adaptive rounds, correctness + ratio.
#   full:             k ∈ {768,1024,1536,2048}.
# Routines: gemm symm syrk syr2k trmm trsm (default: all).

using PureBLAS, LinearAlgebra, Statistics
import LinearAlgebra.BLAS as B
BLAS.set_num_threads(1)
const S = Ref(0.0)
@noinline _run(f) = f()
const L = 'L'; const U = 'U'; const Rr = 'R'; const Nn = 'N'; const Tt = 'T'; const Nd = 'N'

# reps so one timed call is ≳ TARGET_MS (L3 is O(k³); large k already ≫ timer noise ⇒ few reps).
const TARGET_MS = 80.0
@inline _reps(k) = clamp(round(Int, TARGET_MS / (2 * k^3 / 50.0e9 * 1.0e3)), 1, 200)

# Interleaved paired timing → per-round ratios (OB time / our time). `reset` runs UNTIMED before each
# timed block (for in-place ops: refill the dst buffer with no allocation — keeps the measured time the
# pure kernel, not the kernel+copy, and removes the GC variance the per-call `copy` caused).
function _ratios(ob, our, reset, rounds)
    reset(); _run(ob); reset(); _run(our)        # warmup
    rs = Float64[]
    for _ in 1:rounds
        reset(); t0 = time_ns(); a = _run(ob); t1 = time_ns()
        reset(); t2 = time_ns(); b = _run(our); t3 = time_ns()
        S[] += a + b; push!(rs, (t1 - t0) / (t3 - t2))
    end
    return rs
end
# Adaptive: grow rounds until the median is stable (IQR/median < tol) or a cap — guards against the
# ±1–2% noise without over-sampling. Returns (median_ratio, relative_IQR, n_rounds).
function _stable(ob, our, reset; rounds = 9, tol = 0.02, cap = 45)
    rs = _ratios(ob, our, reset, rounds)
    while length(rs) < cap
        m = median(rs); riqr = (quantile(rs, 0.75) - quantile(rs, 0.25)) / m
        riqr < tol && break
        append!(rs, _ratios(ob, our, reset, max(4, rounds ÷ 2)))
    end
    m = median(rs)
    return (m, (quantile(rs, 0.75) - quantile(rs, 0.25)) / m, length(rs))
end

# ── per-routine variant tables: (label, our!, ob!, relerr) builders given k ──────────────────────
_tri_relerr(C2, C, up) = (D = up == U ? triu(C2 - C) : tril(C2 - C); maximum(abs, D) / max(maximum(abs, C), eps()))

# Each variant returns (lab, ob, our, reset, ce). For non-in-place ops the op overwrites a shared C
# (reps>1, reset = no-op). For in-place ops (trmm/trsm) ob/our run ONE op on a shared buffer Bw and
# `reset` refills Bw from Bm UNTIMED (reps=1; >1 would overflow/underflow the repeated in-place op).
function _variants(routine, k)
    A = randn(k, k); Bm = randn(k, k); C = randn(k, k); Bw = copy(Bm)
    Ad = A + k * I                                   # diagonally dominant for trsm
    reps = _reps(k); nz = () -> nothing
    vs = NamedTuple[]
    if routine == "gemm"
        for (lab, ta, tb) in (("NN", Nn, Nn), ("NT", Nn, Tt), ("TN", Tt, Nn))
            our = () -> (
                for _ in 1:reps
                    PureBLAS.gemm!(C, A, Bm; transA = ta, transB = tb, alpha = 1.0, beta = 0.0)
                end; C[1]
            )
            ob = () -> (
                for _ in 1:reps
                    B.gemm!(ta, tb, 1.0, A, Bm, 0.0, C)
                end; C[1]
            )
            ce = () -> (
                C1 = copy(C); C2 = copy(C); B.gemm!(ta, tb, 1.0, A, Bm, 0.0, C1);
                PureBLAS.gemm!(C2, A, Bm; transA = ta, transB = tb, alpha = 1.0, beta = 0.0); norm(C2 - C1) / norm(C1)
            )
            push!(vs, (lab = lab, ob = ob, our = our, reset = nz, ce = ce))
        end
    elseif routine == "symm"
        for (lab, sd) in (("L", L), ("R", Rr))
            our = () -> (
                for _ in 1:reps
                    PureBLAS.symm!(C, A, Bm; side = sd, uplo = L, alpha = 1.0, beta = 0.0)
                end; C[1]
            )
            ob = () -> (
                for _ in 1:reps
                    B.symm!(sd, L, 1.0, A, Bm, 0.0, C)
                end; C[1]
            )
            ce = () -> (
                C1 = copy(C); C2 = copy(C); B.symm!(sd, L, 1.0, A, Bm, 0.0, C1);
                PureBLAS.symm!(C2, A, Bm; side = sd, uplo = L, alpha = 1.0, beta = 0.0); norm(C2 - C1) / norm(C1)
            )
            push!(vs, (lab = lab, ob = ob, our = our, reset = nz, ce = ce))
        end
    elseif routine == "syrk"
        for (lab, up, tr) in (("UN", U, Nn), ("LN", L, Nn), ("UT", U, Tt), ("LT", L, Tt))
            our = () -> (
                for _ in 1:reps
                    PureBLAS.syrk!(C, A; uplo = up, trans = tr, alpha = 1.0, beta = 0.0)
                end; C[1]
            )
            ob = () -> (
                for _ in 1:reps
                    B.syrk!(up, tr, 1.0, A, 0.0, C)
                end; C[1]
            )
            ce = () -> (
                C1 = copy(C); C2 = copy(C); B.syrk!(up, tr, 1.0, A, 0.0, C1);
                PureBLAS.syrk!(C2, A; uplo = up, trans = tr, alpha = 1.0, beta = 0.0); _tri_relerr(C2, C1, up)
            )
            push!(vs, (lab = lab, ob = ob, our = our, reset = nz, ce = ce))
        end
    elseif routine == "syr2k"
        for (lab, up, tr) in (("UN", U, Nn), ("LN", L, Nn), ("UT", U, Tt), ("LT", L, Tt))
            our = () -> (
                for _ in 1:reps
                    PureBLAS.syr2k!(C, A, Bm; uplo = up, trans = tr, alpha = 1.0, beta = 0.0)
                end; C[1]
            )
            ob = () -> (
                for _ in 1:reps
                    B.syr2k!(up, tr, 1.0, A, Bm, 0.0, C)
                end; C[1]
            )
            ce = () -> (
                C1 = copy(C); C2 = copy(C); B.syr2k!(up, tr, 1.0, A, Bm, 0.0, C1);
                PureBLAS.syr2k!(C2, A, Bm; uplo = up, trans = tr, alpha = 1.0, beta = 0.0); _tri_relerr(C2, C1, up)
            )
            push!(vs, (lab = lab, ob = ob, our = our, reset = nz, ce = ce))
        end
    elseif routine == "trmm"
        rst = () -> copyto!(Bw, Bm)                   # untimed in-place reset (no allocation)
        for (lab, up, tr) in (("UN", U, Nn), ("LN", L, Nn), ("UT", U, Tt), ("LT", L, Tt))
            our = () -> (PureBLAS.trmm!(Bw, A; side = L, uplo = up, transA = tr, diag = Nd, alpha = 1.0); Bw[1])
            ob = () -> (B.trmm!(L, up, tr, Nd, 1.0, A, Bw); Bw[1])
            ce = () -> (
                B1 = copy(Bm); B2 = copy(Bm); B.trmm!(L, up, tr, Nd, 0.7, A, B1);
                PureBLAS.trmm!(B2, A; side = L, uplo = up, transA = tr, diag = Nd, alpha = 0.7); norm(B2 - B1) / norm(B1)
            )
            push!(vs, (lab = lab, ob = ob, our = our, reset = rst, ce = ce))
        end
    elseif routine == "trsm"
        rst = () -> copyto!(Bw, Bm)
        for (lab, up, tr) in (("UN", U, Nn), ("LN", L, Nn), ("UT", U, Tt), ("LT", L, Tt))
            our = () -> (PureBLAS.trsm!(Bw, Ad; side = L, uplo = up, transA = tr, diag = Nd, alpha = 1.0); Bw[1])
            ob = () -> (B.trsm!(L, up, tr, Nd, 1.0, Ad, Bw); Bw[1])
            ce = () -> (
                B1 = copy(Bm); B2 = copy(Bm); B.trsm!(L, up, tr, Nd, 0.7, Ad, B1);
                PureBLAS.trsm!(B2, Ad; side = L, uplo = up, transA = tr, diag = Nd, alpha = 0.7); norm(B2 - B1) / norm(B1)
            )
            push!(vs, (lab = lab, ob = ob, our = our, reset = rst, ce = ce))
        end
    end
    return vs
end

function bench_routine(routine, k; check = true)
    for v in _variants(routine, k)
        ce = check ? v.ce() : NaN
        m, riqr, nr = _stable(v.ob, v.our, v.reset)
        flag = m >= 0.96 ? "ok " : "LOW"
        println("  $routine k=$k $(v.lab): $(round(m, digits = 3)) [$flag] (relerr=$(round(ce, sigdigits = 2)), iqr=$(round(100riqr, digits = 1))%, rounds=$nr)")
        flush(stdout)
    end
    return
end

# ── driver ───────────────────────────────────────────────────────────────────────────────────────
const ALL = ["gemm", "symm", "syrk", "syr2k", "trmm", "trsm"]
mode = length(ARGS) >= 1 ? ARGS[1] : "screen"
routines = length(ARGS) >= 2 ? ARGS[2:end] : ALL
ks = mode == "full" ? (768, 1024, 1536, 2048) : (1536,)   # screen: one non-po2 size
println("== L3 $mode bench (k=$(collect(ks)), interleaved-median, 1-thread) =="); flush(stdout)
for routine in routines, k in ks
    bench_routine(routine, k)
end
println("== done (S=$(S[])) ==")
