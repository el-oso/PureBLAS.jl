# Zen3 decomposition + pre-test (Fable's plan). Run on galen (Zen3/AVX2).
#  A) packed vs unpacked complex gemm vs OB at trmm-base shapes → the go/no-go for a packed K-TRIM base.
#  A') current full ztrmm 128/256 → driver+unpacked overhead = (packed ceiling) − (ztrmm actual).
#  B) K-TRIM vs dense trmm-L base on the SAME data → if not ~2× apart, overhead dominates flop count.
#  C) zgemvN roofline: PB & OB GB/s vs a pure A-stream → is it kernel/ports or bandwidth?
using PureBLAS, LinearAlgebra, Statistics, Printf
import LinearAlgebra.BLAS as B
BLAS.set_num_threads(1)
const T = ComplexF64; const P = PureBLAS; const SINK = Ref(0.0)
@noinline _run(f) = f()
function med(mk; rounds = 25)
    rs = Float64[]
    for _ in 1:rounds
        c = mk(); _run(c); t = time_ns(); v = c(); SINK[] += v; push!(rs, Float64(time_ns() - t))
    end
    median(rs)
end
sig(C) = real(C[1]) + imag(C[1])

println("== A: complex gemm C += A·B (β=1), m=n=128 — packed vs unpacked vs OpenBLAS ==")
for k in (32, 64, 128)
    m = n = 128; reps = clamp(3_000_000 ÷ (m * n * k ÷ 128), 5, 300)
    A = randn(T, m, k); Bm = randn(T, k, n)
    tob = med(() -> (C = randn(T, m, n); () -> (for _ in 1:reps; B.gemm!('N', 'N', one(T), A, Bm, one(T), C); end; sig(C))))
    tpk = med(() -> (C = randn(T, m, n); () -> (for _ in 1:reps; P.gemm!(C, A, Bm; alpha = one(T), beta = one(T)); end; sig(C))))
    tup = med(() -> (C = randn(T, m, n); () -> (for _ in 1:reps; P._gemm_cmplx_unpacked_go!(false, false, m, n, k, one(T), A, Bm, one(T), C); end; sig(C))))
    @printf("k=%-4d packed/OB=%.3f  unpacked/OB=%.3f  packed-is-%.2f×-faster-than-unpacked\n",
        k, tob / tpk, tob / tup, tup / tpk)
end

println("\n== A': current full ztrmm side-L (K-TRIM), uplo=U trans=N — the actual op ==")
for k in (128, 256)
    reps = clamp(4_000_000 ÷ (k * k), 3, 200)
    A = randn(T, k, k); for i in 1:k; A[i, i] = 1 + abs(A[i, i]); end
    Bm = randn(T, k, k)
    tob = med(() -> (Bc = copy(Bm); () -> (for _ in 1:reps; B.trmm!('L', 'U', 'N', 'N', one(T), A, Bc); end; sig(Bc))))
    tpb = med(() -> (Bc = copy(Bm); () -> (for _ in 1:reps; P.trmm!(Bc, A; side = 'L', uplo = 'U', transA = 'N', diag = 'N'); end; sig(Bc))))
    @printf("n=%-4d ztrmm PB/OB=%.3f\n", k, tob / tpb)
end

println("\n== B: K-TRIM (_trmm_cmplx_small_L!) vs DENSE (_trmm_cmplx_base_L!) — same data ==")
for k in (64, 128, 256)
    reps = clamp(4_000_000 ÷ (k * k), 3, 200)
    A = randn(T, k, k); for i in 1:k; A[i, i] = 1 + abs(A[i, i]); end
    Bm = randn(T, k, k)
    tkt = med(() -> (Bc = copy(Bm); () -> (for _ in 1:reps; P._trmm_cmplx_small_L!(true, false, false, false, k, A, Bc); end; sig(Bc))))
    tdn = med(() -> (Bc = copy(Bm); () -> (for _ in 1:reps; P._trmm_cmplx_base_L!(true, false, false, false, k, A, Bc); end; sig(Bc))))
    @printf("k=%-4d K-TRIM=%.2fµs  dense=%.2fµs  dense/K-TRIM=%.2f×  (flop ratio ≈2×; <2× ⇒ overhead dominates)\n",
        k, tkt / reps / 1e3, tdn / reps / 1e3, tdn / tkt)
end

println("\n== C: zgemvN roofline — GB/s (A read once = 16·m·n bytes) vs pure A-stream ==")
for n in (256, 512, 1024, 2048)
    m = n; reps = clamp(200_000_000 ÷ (m * n), 20, 3000)
    A = randn(T, m, n); x = randn(T, n); y = randn(T, m)
    Ar = reinterpret(Float64, vec(A)); bytes = 16.0 * m * n
    tob = med(() -> (yy = copy(y); () -> (for _ in 1:reps; B.gemv!('N', one(T), A, x, zero(T), yy); end; real(yy[1]))))
    tpb = med(() -> (yy = copy(y); () -> (for _ in 1:reps; P.gemv!(yy, A, x; alpha = one(T), beta = zero(T)); end; real(yy[1]))))
    tst = med(() -> (() -> (s = 0.0; @inbounds @simd for i in eachindex(Ar); s += Ar[i]; end; s)))
    gob = bytes * reps / tob; gpb = bytes * reps / tpb; gst = bytes / tst
    @printf("n=%-5d OB=%.0f  PB=%.0f  stream=%.0f GB/s   PB/OB=%.2f  PB/stream=%.2f  OB/stream=%.2f\n",
        n, gob, gpb, gst, tob / tpb, gpb / gst, gob / gst)
end
println("\nSINK=", SINK[])
