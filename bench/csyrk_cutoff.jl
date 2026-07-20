# Small-n crossover: packed-tri path vs recursion path for zherk, to set _CSYRK_PACK_CUT per machine.
using PureBLAS, LinearAlgebra, Statistics, Printf
import LinearAlgebra.BLAS as B
BLAS.set_num_threads(1); const SINK = Ref(ComplexF64(0)); @noinline _run(f) = f()
const T = ComplexF64; const P = PureBLAS
rep3(n) = clamp(20_000_000 ÷ (n^3), 1, 5000)
function med(f; rounds = 15)
    rs = Float64[]
    for _ in 1:rounds
        c = f(); _run(c); t = time_ns(); c(); push!(rs, Float64(time_ns() - t))
    end
    return median(rs)
end
# Both trans, both paths. tr flag: 'N'→tr=false, 'C'→tr=true. up='U'.
for (tc, tr) in (('N', false), ('C', true))
    for n in (8, 12, 16, 24, 32, 48)
        r = rep3(n)
        A = randn(T, n, n)
        scr = () -> zeros(T, n, n)
        ob = med(
            () -> (
                C = scr(); () -> (
                    for _ in 1:r
                        B.herk!('U', tc, 1.0, A, 0.0, C)
                    end
                )
            )
        )
        pk = med(
            () -> (
                C = scr(); () -> (
                    for _ in 1:r
                        P._syrk_scaleC!(C, true, 0.0); P._csyrk_packed!(true, tr, true, 1.0, A, C, n)
                    end
                )
            )
        )
        rc = med(
            () -> (
                C = scr(); () -> (
                    for _ in 1:r
                        P._syrk_scaleC!(C, true, 0.0); P._syrk_rec!(true, tr, true, 1.0, A, C, n, P._l3_tmp(T), 0, n)
                    end
                )
            )
        )
        @printf("herk%s n=%-4d packed=%.3f  recursion=%.3f\n", tc, n, ob / pk, ob / rc)
    end
end
