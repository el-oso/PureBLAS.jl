# F32 potrf small-n: is the gap fixable by a SMALLER base (route n=32/64 through the fast F32 SIMD
# trsm!/syrk! recursion instead of the scalar _potf2 base)? Measure spotrf-L at bases 8/16/24/32.
# Ratio = OB_time / PB_time (>1 = PB gates). Run: julia --project=bench bench/f32base_probe.jl
using PureBLAS, Chairmarks, LinearAlgebra
import LinearAlgebra.LAPACK
import PureBLAS as P
BLAS.set_num_threads(1)
hpd(T, s) = (A = randn(T, s, s); Matrix(A * A' + s * I))
mintime(b) = minimum(x.time for x in b.samples)

for s in (32, 64, 128)
    A0 = hpd(Float32, s)
    ob = @be copy(A0) LAPACK.potrf!('L', _) evals=1 seconds=0.5
    tob = mintime(ob)
    print("spotrf-L n=$s  OB-ratio:")
    for base in (8, 16, 24, 32)
        pb = @be copy(A0) P._potrf_lower!(_, s, base) evals=1 seconds=0.5
        print("  base$base=", round(tob / mintime(pb), digits=2))
    end
    println()
end
