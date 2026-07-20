# Does padding B's ldb (break 4K aliasing) lift the SHIPPED trsm PB/AOCL ratio at po2 sizes?
# Both PB and ref get the same padded B (fair). If the RATIO improves, PB has an aliasing weakness ref lacks.
# Usage: taskset -c 8 julia --project=bench bench/trsm_pad_ab.jl [aocl]
using LinearAlgebra, Chairmarks, Statistics, Printf
const REF = "aocl" in ARGS ? "aocl" : "openblas"
if REF == "aocl"
    @eval using AOCL_jll
    BLAS.lbt_forward(AOCL_jll.aocl_blas_ilp64; clear = true); BLAS.lbt_forward(AOCL_jll.aocl_lapack_ilp64)
end
BLAS.set_num_threads(1); import PureBLAS
const SIZES = (256, 384, 512, 768, 1024)
tri(s) = (
    A = triu(randn(Float64, s, s)); @inbounds for i in 1:s
        A[i, i] += Float64(s)
    end; A
)
mkB(s, pad) = pad == 0 ? randn(Float64, s, s) : view(randn(Float64, s + pad, s), 1:s, :)
mk(s, pad) = (tri(s), mkB(s, pad))
@noinline ob1(c) = (BLAS.trsm!('L', 'U', 'N', 'N', 1.0, c[1], c[2]); c[2][1])
@noinline pb1(c) = (PureBLAS.trsm!(c[2], c[1]; side = 'L', uplo = 'U', transA = 'N'); c[2][1])
med(bo, bp) = (
    to = sort(getproperty.(bo.samples, :time)); tp = sort(getproperty.(bp.samples, :time));
    n = min(length(to), length(tp)); median(to[i] / tp[i] for i in 1:n)
)
bench(s, pad, reps, f, secs) = @be [mk(s, pad) for _ in 1:reps] (
    x -> (
        v = 0.0; for c in x
            v += f(c)
        end; v
    )
) evals = 1 samples = 40 seconds = secs
println("ref=$REF   (PB/ref unpadded    PB/ref padded(ldb+8)    Δ)")
for s in SIZES
    reps = clamp(20_000_000 ÷ (s * s * s), 1, 512); secs = s >= 512 ? 2.0 : 2.5
    r0 = Float64[]; r1 = Float64[]
    for round in 1:3
        bo0 = bench(s, 0, reps, ob1, secs); bp0 = bench(s, 0, reps, pb1, secs)
        bo1 = bench(s, 8, reps, ob1, secs); bp1 = bench(s, 8, reps, pb1, secs)
        push!(r0, med(bo0, bp0)); push!(r1, med(bo1, bp1))
    end
    @printf(
        "n=%-5d  unpadded %.3f   padded %.3f   (Δ %+.1f%%)\n", s, median(r0), median(r1),
        100 * (median(r1) / median(r0) - 1)
    )
end
