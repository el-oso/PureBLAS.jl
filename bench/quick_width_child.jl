# One arm of quick_width_ab.jl; run by it, not directly. See that file's header.
using MKL                         # forward LinearAlgebra's BLAS/LAPACK to MKL — BEFORE PureBLAS loads
using PureBLAS, LinearAlgebra, Printf, Random
using Chairmarks: @be

const ARM = ENV["QUICK_WIDTH_ARM"]
const NT = Threads.nthreads()

# Fail before timing anything if the arm's preference did not reach the compiled package.
let want = Dict("detected" => nothing, "fpw32" => (:fpw, 32), "simd32" => (:simd, 32))[ARM]
    isnothing(want) || getfield(PureBLAS._HW, want[1]) == want[2] ||
        error("arm $ARM: PureBLAS detected $(PureBLAS._HW); the preference did not take effect")
end

mkl_set_threads(n) = ccall((:MKL_Set_Num_Threads, MKL.MKL_jll.libmkl_rt), Cvoid, (Cint,), n)
use_mkl(nt) = (PureBLAS.deactivate(); MKL.__init__(); mkl_set_threads(nt); BLAS.set_num_threads(nt))
med(v) = (s = sort(v); s[(length(s) + 1) ÷ 2])

function compare(f, setup, nt)
    PureBLAS.set_num_threads(nt)
    use_mkl(nt); f(setup()); PureBLAS.activate(); f(setup())                 # compile both paths
    mkl = Float64[]; pb = Float64[]
    for _ in 1:2
        use_mkl(nt)
        append!(mkl, [s.time for s in (@be setup() f evals = 1 seconds = 0.5).samples])
        sleep(0.3)                                                          # let MKL's threads go idle
        PureBLAS.activate()
        append!(pb, [s.time for s in (@be setup() f evals = 1 seconds = 0.5).samples])
    end
    use_mkl(nt)
    return med(mkl), med(pb)
end

function main()
    threads = unique((1, PureBLAS.set_num_threads(NT)))
    println("\n=== arm $ARM: PureBLAS detected ", PureBLAS._HW, ", double-pumped = ", PureBLAS._double_pumped(PureBLAS._HW))
    A = randn(MersenneTwister(1), 300, 300); B = randn(MersenneTwister(2), 300, 300)
    use_mkl(1); ref = A * B; PureBLAS.activate(); pb = A * B; use_mkl(1)
    isapprox(pb, ref; rtol = 1e-12) || error("arm $ARM: PureBLAS product differs from MKL's")
    @printf("%-8s %6s", "op", "n")
    for nt in threads; @printf("   %16s", "$nt thread$(nt == 1 ? "" : "s")"); end
    println()
    rng = MersenneTwister(1)
    for n in (1024, 2048)
        A = randn(rng, n, n); B = randn(rng, n, n); S = A' * A + n * I; U = triu(A) + n * I
        for (name, fl, setup, f) in (
                ("gemm", 2.0n^3, () -> zeros(n, n), C -> mul!(C, A, B)),
                ("syrk", 1.0n^3, () -> zeros(n, n), C -> BLAS.syrk!('U', 'N', 1.0, A, 0.0, C)),
                ("trsm", 1.0n^3, () -> copy(B), X -> BLAS.trsm!('L', 'U', 'N', 'N', 1.0, U, X)),
                ("potrf", n^3 / 3, () -> copy(S), X -> LAPACK.potrf!('U', X)),
                ("getrf", 2n^3 / 3, () -> copy(A), X -> LAPACK.getrf!(X)),
            )
            @printf("%-8s %6d", name, n)
            for nt in threads
                tm, tp = compare(f, setup, nt)
                @printf("   %5.2fx %6.1f GF/s", tm / tp, fl / tp / 1e9)
            end
            println(); flush(stdout)
        end
    end
end
main()
