# Same-process controlled A/B: whole-k packed sweep (_TRSM_FULLPACK_ON) vs current recursion+single-leaf,
# both vs ref (AOCL or OpenBLAS). Gate shape side=L uplo=U transA=N, square B. po2 AND non-po2 sizes.
# Usage: taskset -c 8 julia --project=bench bench/trsm_ab3.jl [aocl]
using LinearAlgebra, Chairmarks, Statistics, Printf
const REF = "aocl" in ARGS ? "aocl" : "openblas"
if REF == "aocl"
    @eval using AOCL_jll
    BLAS.lbt_forward(AOCL_jll.aocl_blas_ilp64; clear = true); BLAS.lbt_forward(AOCL_jll.aocl_lapack_ilp64)
end
BLAS.set_num_threads(1)
const ELT = Float64
import PureBLAS
const SIZES = (128, 224, 255, 256, 257, 272, 384, 512, 513, 768, 1024)
tri(::Type{T}, s) where {T} = (
    A = triu(randn(T, s, s)); @inbounds for i in 1:s
        A[i, i] += T(s)
    end; A
)
mk(s) = (tri(ELT, s), randn(ELT, s, s))
@noinline ob1(c) = (BLAS.trsm!('L', 'U', 'N', 'N', one(ELT), c[1], c[2]); c[2][1])
@noinline pb1(c) = (PureBLAS.trsm!(c[2], c[1]; side = 'L', uplo = 'U', transA = 'N'); c[2][1])
med(bo, bp) = (
    to = sort(getproperty.(bo.samples, :time)); tp = sort(getproperty.(bp.samples, :time));
    n = min(length(to), length(tp)); median(to[i] / tp[i] for i in 1:n)
)
bench(s, reps, f, secs) = @be [mk(s) for _ in 1:reps] (
    x -> (
        v = 0.0; for c in x
            v += f(c)
        end; v
    )
) evals = 1 samples = 40 seconds = secs

println("ref=$REF elt=$ELT   (PB_fullpack/ref   PB_recursion/ref   per size)")
for s in SIZES
    reps = clamp(20_000_000 ÷ (s * s * s), 1, 512); secs = s >= 512 ? 2.0 : 2.5
    r_on = Float64[]; r_off = Float64[]
    for round in 1:3
        bo = bench(s, reps, ob1, secs)
        PureBLAS._TRSM_FULLPACK_ON[] = true;  bpn = bench(s, reps, pb1, secs)
        PureBLAS._TRSM_FULLPACK_ON[] = false; bpf = bench(s, reps, pb1, secs)
        push!(r_on, med(bo, bpn)); push!(r_off, med(bo, bpf))
    end
    @printf(
        "n=%-5d  fullpack %.3f   recursion %.3f   (Δ %+.1f%%)\n", s, median(r_on), median(r_off),
        100 * (median(r_on) / median(r_off) - 1)
    )
end
