# Decompose zgetrf phases: panel getf2, laswp, trsm, gemm. Times each phase's total across the factorization
# at given n, plus compares PB trailing gemm/trsm vs OB at representative trailing shapes.
using PureBLAS, LinearAlgebra, Statistics, Printf
import LinearAlgebra.BLAS as B
import LinearAlgebra.LAPACK as LA
using PureBLAS: _getf2!, _laswp!, _lu_nb, _LU_NB
BLAS.set_num_threads(1)
const T = ComplexF64
const P = PureBLAS

# instrumented core: same as _getrf_core! but accumulates per-phase ns
function core_timed(A, ipiv, nb)
    m, n = size(A); k = min(m, n); nb = clamp(nb, 1, k)
    tp = tl = tt = tg = 0.0; pc = 1
    while pc <= k
        pb = min(nb, k - pc + 1); mp = m - pc + 1
        t = time_ns(); _getf2!(view(A, pc:m, pc:(pc + pb - 1)), mp, pb, pc - 1, ipiv, pc - 1); tp += time_ns() - t
        jt0 = pc + pb
        if jt0 <= n
            t = time_ns(); _laswp!(A, ipiv, pc, pc + pb - 1, jt0, n); tl += time_ns() - t
            t = time_ns(); P.trsm!(view(A, pc:(pc + pb - 1), jt0:n), view(A, pc:(pc + pb - 1), pc:(pc + pb - 1)); side = 'L', uplo = 'L', transA = 'N', diag = 'U', alpha = true); tt += time_ns() - t
            if pc + pb <= m
                t = time_ns(); P.gemm!(view(A, (pc + pb):m, jt0:n), view(A, (pc + pb):m, pc:(pc + pb - 1)), view(A, pc:(pc + pb - 1), jt0:n); alpha = -1, beta = true); tg += time_ns() - t
            end
        end
        pc += pb
    end
    return (tp, tl, tt, tg)
end

for n in (128, 256, 512, 1024, 2048)
    nb = _LU_NB
    reps = clamp(20_000_000 ÷ n^2, 3, 100)
    acc = zeros(4)
    for _ in 1:reps
        A = randn(T, n, n); ipiv = Vector{Int}(undef, n)
        acc .+= collect(core_timed(A, ipiv, nb))
    end
    acc ./= reps; tot = sum(acc)
    @printf(
        "n=%-5d nb=%-3d tot=%7.1fus  panel=%5.1f%% laswp=%5.1f%% trsm=%5.1f%% gemm=%5.1f%%\n",
        n, nb, tot / 1000, 100acc[1] / tot, 100acc[2] / tot, 100acc[3] / tot, 100acc[4] / tot
    )
end
