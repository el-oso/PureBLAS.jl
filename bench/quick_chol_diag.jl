# Splits a slow `cholesky` row of quick_vs_mkl.jl into the calls PureBLAS's blocked Cholesky makes.
#
#     julia -t auto bench/quick_chol_diag.jl
#     julia> include("bench/quick_chol_diag.jl")
#
# `cholesky!` factors the upper triangle. PureBLAS's upper factorization halves the matrix and, at the
# top level, spends its time in one trsm (side L, upper, transposed) and one syrk (upper, transposed) on
# n/2 blocks. Timing those beside upper and lower `potrf` says which piece is slow. The output is short
# on purpose, so the numbers can be read off the screen. Ratio as in quick_vs_mkl.jl: >1 = PureBLAS faster.
const QUICK_VS_MKL_NO_MAIN = true
include(joinpath(@__DIR__, "quick_vs_mkl.jl"))

function chol_diag()
    println("CPU ", Sys.CPU_NAME, "; PureBLAS detected ", PureBLAS._HW)
    check_backends()
    threads = unique((1, PureBLAS.set_num_threads(NT)))        # as in quick_vs_mkl.jl
    rng = MersenneTwister(1)
    @printf("%-8s %6s", "call", "n")
    for nt in threads; @printf("  %4dt", nt); end
    println()
    for n in (1024, 2048)
        A = randn(rng, n, n); S = A' * A + n * I
        h = n ÷ 2; P = randn(rng, h, h); U = triu(randn(rng, h, h)) + h * I
        rows = (
            ("potrfU", n, () -> copy(S), X -> LAPACK.potrf!('U', X)),
            ("potrfL", n, () -> copy(S), X -> LAPACK.potrf!('L', X)),
            ("trsmLUT", h, () -> copy(P), X -> BLAS.trsm!('L', 'U', 'T', 'N', 1.0, U, X)),
            ("syrkUT", h, () -> zeros(h, h), C -> BLAS.syrk!('U', 'T', -1.0, P, 1.0, C)),
            ("gemm", h, () -> zeros(h, h), C -> mul!(C, P, U)),
        )
        for (name, sz, setup, f) in rows
            @printf("%-8s %6d", name, sz)
            for nt in threads
                tm, tp = compare(f, setup, nt)
                @printf("  %5.2f", tm / tp)
            end
            println(); flush(stdout)
        end
    end
end
chol_diag()
