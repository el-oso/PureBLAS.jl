# REGIME: illustrative, for docs/src/simd.md. How much of gemm's speed comes from cache blocking
# rather than from vectorising the inner loop? The naive kernel below is written so LLVM CAN and does
# vectorise it (@inbounds, @simd on the contiguous k-accumulation) — the difference measured is data
# movement, not instruction selection.
using PureBLAS, LinearAlgebra, Chairmarks, Printf, Statistics

function naive_gemm!(C, A, B)
    m, n = size(C); k = size(A, 2)
    @inbounds for j in 1:n, i in 1:m
        acc = zero(eltype(C))
        @simd for p in 1:k
            acc += A[i, p] * B[p, j]
        end
        C[i, j] = acc
    end
    return C
end

BLAS.set_num_threads(1)
for n in (256, 512, 1000)
    A = randn(n, n); B = randn(n, n); C = zeros(n, n)
    naive_gemm!(C, A, B); PureBLAS.gemm!(C, A, B)          # warm up both
    tn = @be naive_gemm!($C, $A, $B) seconds=2
    tp = @be PureBLAS.gemm!($C, $A, $B) seconds=2
    gf(t) = 2 * n^3 / (t * 1e9)
    mn = median(tn).time; mp = median(tp).time
    @printf("n=%-5d naive %6.2f GFlop/s   PureBLAS %6.2f GFlop/s   ratio %5.1fx\n",
            n, gf(mn), gf(mp), mn / mp)
end
