# Correctness of the whole-k packed fused sweep _trsm_fused_full_L! (the _TRSM_FULLPACK_ON substrate),
# called DIRECTLY (bypasses the _TRSM_FULLPACK_MIN gate) so every k — incl. non-multiples of MR (rem
# tail), unit/non-unit diag, strided parent — is exercised. Oracle = OpenBLAS trsm.
# Usage: taskset -c 8 julia --project=bench bench/trsm_fullpack_check.jl
using LinearAlgebra, Printf; import PureBLAS
const P = PureBLAS
BLAS.set_num_threads(1)
tri(s) = (
    A = triu(randn(Float64, s, s)); @inbounds for i in 1:s
        A[i, i] += Float64(s)
    end; A
)

function run()
    worst = 0.0; nfail = 0
    for unit in (false, true)
        diag = unit ? 'U' : 'N'
        for k in (8, 15, 16, 17, 23, 24, 31, 32, 63, 64, 65, 120, 127, 128, 129, 200, 255, 256, 257, 341, 384, 511, 512, 513, 600, 768, 1024)
            for nc in (1, 7, 8, 24, 25, 48, 64, 100, 128, 256)
                A = tri(k)
                B = randn(Float64, k, nc)
                X = copy(B); P._trsm_fused_full_L!(unit, A, X)
                Xr = copy(B); BLAS.trsm!('L', 'U', 'N', diag, 1.0, A, Xr)
                r = norm(X - Xr) / max(norm(Xr), eps())
                worst = max(worst, r)
                if r > 1.0e-11 || isnan(r)
                    nfail += 1
                    nfail <= 15 && @printf("FAIL unit=%s k=%d nc=%d rel=%.2e\n", unit, k, nc, r)
                end
            end
        end
    end
    # strided parent (po2 lda stress)
    for k in (384, 512), nc in (256,)
        Par = randn(Float64, k + 7, nc); B = Matrix(view(Par, 1:k, :))
        A = tri(k)
        X = view(Par, 1:k, :); Xin = copy(B); P._trsm_fused_full_L!(false, A, X)
        Xr = copy(B); BLAS.trsm!('L', 'U', 'N', 'N', 1.0, A, Xr)
        r = norm(Matrix(view(Par, 1:k, :)) - Xr) / norm(Xr); worst = max(worst, r)
        r > 1.0e-11 && (nfail += 1; @printf("FAIL strided k=%d rel=%.2e\n", k, r))
    end
    @printf("worst_rel=%.3e  nfail=%d\n", worst, nfail)
    return nfail
end
@printf("\nTOTAL nfail=%d\n", run())
