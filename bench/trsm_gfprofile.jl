# Profile: fused-leaf effective GF vs a peak dgemm of comparable shape, and vs invL, isolated.
# Usage: taskset -c 8 julia --project=bench bench/trsm_gfprofile.jl
using LinearAlgebra, Chairmarks, Statistics, Printf
import PureBLAS
const P = PureBLAS
BLAS.set_num_threads(1)
tri(s) = (
    A = triu(randn(Float64, s, s)); @inbounds for i in 1:s
        A[i, i] += Float64(s)
    end; A
)

# effective GF of one full fused leaf solve U(KC×KC)·X = B(KC×n): flops ≈ n·KC² (triangular ×2 MAC/2)
function leaf_gf(KC, n)
    A = tri(KC)
    Bs = [randn(Float64, KC, n) for _ in 1:8]
    P._TRSM_FUSED_ON[] = true                                  # _trsm_fused_L! is itself a single leaf of size KC
    f(B) = (P._trsm_fused_L!(false, A, B); B[1])
    b = @be Bs (
        x -> (
            v = 0.0; for B in x
                v += f(B)
            end; v
        )
    ) evals = 1 samples = 200 seconds = 2.0
    t = median(getproperty.(b.samples, :time)) / length(Bs)
    flops = n * KC^2            # ~1× triangular solve flops
    return (flops / t / 1.0e9, t)
end

# reference: plain dgemm C(KC×n) = A(KC×KC)·B(KC×n), flops 2·KC²·n → but compare the SAME n·KC² useful
function gemm_gf(KC, n)
    A = randn(Float64, KC, KC); Bs = [randn(Float64, KC, n) for _ in 1:8]; C = randn(Float64, KC, n)
    g(B) = (P.gemm!(C, A, B; alpha = true, beta = false); C[1])
    b = @be Bs (
        x -> (
            v = 0.0; for B in x
                v += g(B)
            end; v
        )
    ) evals = 1 samples = 200 seconds = 2.0
    t = median(getproperty.(b.samples, :time)) / length(Bs)
    flops = 2 * KC^2 * n       # full gemm 2·MAC
    return (flops / t / 1.0e9, t)
end

function invL_gf(KC, n)
    A = tri(KC); Bs = [randn(Float64, KC, n) for _ in 1:8]
    P._TRSM_FUSED_ON[] = false
    f(B) = (P._trsm_base_invL!(true, false, false, A, B); B[1])
    KC > 32 && return (NaN, NaN)   # invL base only defined ≤ _TRSM_BASE; skip larger
    b = @be Bs (
        x -> (
            v = 0.0; for B in x
                v += f(B)
            end; v
        )
    ) evals = 1 samples = 200 seconds = 2.0
    t = median(getproperty.(b.samples, :time)) / length(Bs)
    flops = n * KC^2
    return (flops / t / 1.0e9, t)
end

@printf("%-6s %-6s | fused GF (1x) | dgemm GF (2x, peak) | invL GF (1x-useful)\n", "KC", "n")
for (KC, n) in ((32, 256), (64, 256), (128, 256), (128, 512), (32, 512), (64, 512))
    fg, _ = leaf_gf(KC, n); gg, _ = gemm_gf(KC, n); ig, _ = invL_gf(KC, n)
    @printf("%-6d %-6d | %11.1f   | %14.1f      | %.1f\n", KC, n, fg, gg, ig)
end
