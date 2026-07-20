# Effective GF of the whole-k packed fused sweep _trsm_fused_full_L! (the _TRSM_FULLPACK_ON substrate),
# called directly at (k,n), vs a plain dgemm of the same k×n shape (the ~43-GF ceiling PB's own gemm hits
# and the AOCL trsm ~42-GF target). flops(trsm)=n·k² (1× triangular); flops(gemm)=2·k²·n. This is the
# decomposition that FALSIFIED the inv-diag + prefetch levers: small-k is pack-overhead-dominated (k=128
# ~31 GF, amortizes to ~40.6 by k≥384), large-k plateaus at the slab microkernel rate (~40.6 vs dgemm 43).
# Usage: taskset -c 8 julia --project=bench bench/trsm_fullpack_gf.jl
using LinearAlgebra, Chairmarks, Statistics, Printf; import PureBLAS
const P = PureBLAS
BLAS.set_num_threads(1)
tri(s) = (
    A = triu(randn(Float64, s, s)); @inbounds for i in 1:s
        A[i, i] += Float64(s)
    end; A
)

function fp_gf(k, n)
    A = tri(k); Bs = [randn(Float64, k, n) for _ in 1:8]
    f(B) = (P._trsm_fused_full_L!(false, A, B); B[1])
    b = @be Bs (
        x -> (
            v = 0.0; for B in x
                v += f(B)
            end; v
        )
    ) evals = 1 samples = 200 seconds = 2.0
    t = median(getproperty.(b.samples, :time)) / length(Bs)
    return (n * k^2 / t / 1.0e9)
end
function gemm_gf(k, n)
    A = randn(Float64, k, k); Bs = [randn(Float64, k, n) for _ in 1:8]; C = randn(Float64, k, n)
    g(B) = (P.gemm!(C, A, B; alpha = true, beta = false); C[1])
    b = @be Bs (
        x -> (
            v = 0.0; for B in x
                v += g(B)
            end; v
        )
    ) evals = 1 samples = 200 seconds = 2.0
    t = median(getproperty.(b.samples, :time)) / length(Bs)
    return (2 * k^2 * n / t / 1.0e9)
end

@printf("%-6s %-6s | fullpack GF | dgemm GF (2x peak) | ratio\n", "k", "n")
for (k, n) in ((128, 128), (128, 256), (256, 256), (384, 384), (512, 512), (512, 256), (768, 768), (1024, 1024))
    fp = fp_gf(k, n); gg = gemm_gf(k, n)
    @printf("%-6d %-6d | %9.1f   | %14.1f     | %.3f\n", k, n, fp, gg, fp / gg)
end
