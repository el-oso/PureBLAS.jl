# Final A/B on the REAL shipping path: before = old AVX2 chain (_iamax_chain4!), after = PureBLAS.iamax
# (now routes to _iamax_thresh4! on AVX2). Ratio = ref_time / pb_time (>1 = PB beats the reference).
# Fresh input per sample (evals=1). Run twice: default (OpenBLAS) and `aocl`.
using LinearAlgebra, Statistics, Printf
using Chairmarks: @be
import PureBLAS
const B = LinearAlgebra.BLAS
B.set_num_threads(1)
const REF = "aocl" in ARGS ? "aocl" : "openblas"
if REF == "aocl"
    using AOCL_jll
    B.lbt_forward(AOCL_jll.aocl_blas_ilp64; clear = true)
end
const REFNAME = REF == "aocl" ? "AOCL" : "OpenBLAS"

@noinline before(x) = GC.@preserve x PureBLAS._iamax_chain4!(length(x), pointer(x))  # old AVX2 kernel
@noinline after(x)  = PureBLAS.iamax(x)                                              # new shipping path

med(f, mk; s = 0.20) = median(Float64[smp.time for smp in (@be mk() f evals=1 samples=500 seconds=s).samples])

function run(T)
    println("\n== $REFNAME  T=$T ==   PB/$REFNAME  (>1.0 = PB faster; gate ≥1.0)")
    @printf("%-9s  %10s  %10s\n", "n", "before", "after")
    for n in (1000, 3000, 10000, 100000, 1000000)
        reps = clamp(8_000_000 ÷ n, 30, 20000); mk = () -> randn(T, n)
        tr = med(x -> (s = 0; for _ in 1:reps; s += B.iamax(x); end; s), mk)
        tb = med(x -> (s = 0; for _ in 1:reps; s += before(x); end; s), mk)
        ta = med(x -> (s = 0; for _ in 1:reps; s += after(x); end; s), mk)
        @printf("%-9d  %10.3f  %10.3f\n", n, tr / tb, tr / ta)
    end
end
for T in (Float64, Float32); run(T); end
