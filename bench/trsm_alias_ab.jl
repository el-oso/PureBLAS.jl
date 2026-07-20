# A/B the alias-fallback (route 4K-aliased ldb to the alias-free column-outer pack) on the SHIPPED trsm path.
# Usage: taskset -c 8 julia --project=bench bench/trsm_alias_ab.jl [aocl]
using LinearAlgebra, Chairmarks, Statistics, Printf
const REF = "aocl" in ARGS ? "aocl" : "openblas"
if REF == "aocl"
    @eval using AOCL_jll
    BLAS.lbt_forward(AOCL_jll.aocl_blas_ilp64; clear = true); BLAS.lbt_forward(AOCL_jll.aocl_lapack_ilp64)
end
BLAS.set_num_threads(1); import PureBLAS
const SIZES = (128, 256, 384, 512, 513, 768, 1024)
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
println("ref=$REF   (PB_aliasfix/ref   PB_shipped/ref   per size)")
for s in SIZES
    reps = clamp(20_000_000 ÷ (s * s * s), 1, 512); secs = s >= 512 ? 2.0 : 2.5
    r_fix = Float64[]; r_sh = Float64[]
    for round in 1:3
        bo = bench(s, reps, ob1, secs)
        PureBLAS._GT_ALIAS_FALLBACK[] = true;  bfx = bench(s, reps, pb1, secs)
        PureBLAS._GT_ALIAS_FALLBACK[] = false; bsh = bench(s, reps, pb1, secs)
        push!(r_fix, med(bo, bfx)); push!(r_sh, med(bo, bsh))
    end
    @printf(
        "n=%-5d  aliasfix %.3f   shipped %.3f   (Δ %+.1f%%)\n", s, median(r_fix), median(r_sh),
        100 * (median(r_fix) / median(r_sh) - 1)
    )
end
