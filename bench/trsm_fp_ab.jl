# Same-process A/B: whole-k fullpack sweep (_TRSM_FULLPACK_ON) vs shipped recursion+fused-leaf, both vs ref.
# Gate shape side=L uplo=U transA=N, square B. Usage: taskset -c 8 julia --project=bench bench/trsm_fp_ab.jl [aocl]
using LinearAlgebra, Chairmarks, Statistics, Printf
const REF = "aocl" in ARGS ? "aocl" : "openblas"
if REF == "aocl"
    @eval using AOCL_jll
    BLAS.lbt_forward(AOCL_jll.aocl_blas_ilp64; clear = true); BLAS.lbt_forward(AOCL_jll.aocl_lapack_ilp64)
end
BLAS.set_num_threads(1)
import PureBLAS
const SIZES = (384, 512, 513, 640, 768, 896, 1024)
tri(s) = (
    A = triu(randn(Float64, s, s)); @inbounds for i in 1:s
        A[i, i] += Float64(s)
    end; A
)
mk(s) = (tri(s), randn(Float64, s, s))
@noinline ob1(c) = (BLAS.trsm!('L', 'U', 'N', 'N', 1.0, c[1], c[2]); c[2][1])
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
println("ref=$REF   (PB_fullpack/ref   PB_shipped/ref   per size)")
for s in SIZES
    reps = clamp(20_000_000 ÷ (s * s * s), 1, 512); secs = s >= 512 ? 2.0 : 2.5
    r_fp = Float64[]; r_sh = Float64[]
    for round in 1:3
        bo = bench(s, reps, ob1, secs)
        PureBLAS._TRSM_FULLPACK_ON[] = true; PureBLAS._TRSM_FUSEDT_ON[] = true;  bfp = bench(s, reps, pb1, secs)
        PureBLAS._TRSM_FULLPACK_ON[] = false; PureBLAS._TRSM_FUSEDT_ON[] = false; bsh = bench(s, reps, pb1, secs)
        push!(r_fp, med(bo, bfp)); push!(r_sh, med(bo, bsh))
    end
    @printf(
        "n=%-5d  fullpack %.3f   shipped %.3f   (Δ %+.1f%%)\n", s, median(r_fp), median(r_sh),
        100 * (median(r_fp) / median(r_sh) - 1)
    )
end
