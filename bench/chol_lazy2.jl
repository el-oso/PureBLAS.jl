# Fused-internal lazy driver: replicate _chol_panel_f64! but with a trailing COLUMN LIMIT `clim` — the
# fused 128-panel trsm + syrk run as before, but the trailing update is CONFINED to cols ≤ clim (syrk)
# with the below-clim rows done as a gemm; the far [clim:N,clim:N] block is DEFERRED. The lazy driver
# then batches those deferred far updates into ONE high-AI rank-b syrk/gemm per superpanel.
using PureBLAS, LinearAlgebra
using Chairmarks: @be
using Statistics: median
import PureBLAS: _cvptr, _chol_rl_f64!, _trsm_rl_split_f64!, _syrk_lower_split_f64!, _CHOL_BLOCK,
    _CHOL_STH, _CHOL_SB, _CHOL_MC, _L2_BYTES, _CHOL_D, _CHOL_T, syrk!, gemm!
BLAS.set_num_threads(1)
_t(b) = median([s.time for s in b.samples])
spd(n) = (A = randn(n, n); A * A' + n * I)

# factor cols [1:clim] of the N×N (lower) block A, right-looking 128-blocked; sub-panels solved full
# height N; trailing syrk confined to [.:clim], below-clim rows updated by gemm; [clim:N,clim:N] deferred.
function panel_clim!(A::AbstractMatrix{Float64}, N::Int, clim::Int)
    lda = stride(A, 2)
    Tb = _CHOL_T[]
    if size(Tb, 1) < N + 8
        R = (N + 8) % 128 == 0 ? N + 16 : N + 8; Tb = _CHOL_T[] = Matrix{Float64}(undef, R, _CHOL_BLOCK)
    end
    ldT = size(Tb, 1); D = _CHOL_D; ldD = size(D, 1)
    GC.@preserve A Tb D begin
        pa = pointer(A); pT = pointer(Tb); pD = pointer(D)
        j = 0
        while j < clim
            bs = min(_CHOL_BLOCK, clim - j)
            pjj = _cvptr(pa, j + 1, j + 1, lda)
            for c in 0:(bs - 1)
                unsafe_copyto!(pD + (c * ldD + c) * 8, pjj + (c * lda + c) * 8, bs - c)
            end
            _chol_rl_f64!(pD, bs, ldD, _CHOL_SB, _CHOL_STH) || throw(PosDefException(j + 1))
            for c in 0:(bs - 1)
                unsafe_copyto!(pjj + (c * lda + c) * 8, pD + (c * ldD + c) * 8, bs - c)
            end
            m = N - j - bs                                  # full sub-panel height
            if m > 0
                p21 = _cvptr(pa, j + bs + 1, j + 1, lda)
                i0 = 0
                while i0 < m
                    mc = min(_CHOL_MC, m - i0)
                    _trsm_rl_split_f64!(pD, ldD, p21 + i0 * 8, lda, pT + i0 * 8, ldT, bs, mc)
                    i0 += mc
                end
                for c in 0:(bs - 1)
                    unsafe_copyto!(p21 + c * lda * 8, pT + c * ldT * 8, m)
                end
                # trailing CONFINED to cols [j+bs+1 : clim]
                cw = clim - (j + bs)                        # trailing width within superpanel
                if cw > 0
                    # square diagonal part [j+bs+1:clim, j+bs+1:clim] → fused/packed syrk
                    if cw * bs * 8 <= _L2_BYTES ÷ 2
                        _syrk_lower_split_f64!(_cvptr(pa, j + bs + 1, j + bs + 1, lda), lda, pT, ldT, cw, bs)
                    else
                        syrk!(
                            view(A, (j + bs + 1):clim, (j + bs + 1):clim), view(Tb, 1:cw, 1:bs);
                            uplo = 'L', trans = 'N', alpha = -1.0, beta = 1.0
                        )
                    end
                    # below-clim rows [clim+1:N] × [j+bs+1:clim] → gemm (rank bs)
                    below = N - clim
                    if below > 0
                        gemm!(
                            view(A, (clim + 1):N, (j + bs + 1):clim), view(Tb, (cw + 1):(cw + below), 1:bs),
                            view(Tb, 1:cw, 1:bs); transB = 'T', alpha = -1.0, beta = 1.0
                        )
                    end
                end
            end
            j += bs
        end
    end
    return A
end

const _PROF = [0.0, 0.0]   # [cross-update (syrk+gemm from history), internal panel_clim] ns
function chol_lazy2!(A, n::Int; NB2::Int = 1024)
    j = 0
    while j < n
        b = min(NB2, n - j)
        if j > 0
            t = time_ns()
            syrk!(
                view(A, (j + 1):(j + b), (j + 1):(j + b)), view(A, (j + 1):(j + b), 1:j);
                uplo = 'L', trans = 'N', alpha = -1.0, beta = 1.0
            )
            if j + b < n
                gemm!(
                    view(A, (j + b + 1):n, (j + 1):(j + b)), view(A, (j + b + 1):n, 1:j), view(A, (j + 1):(j + b), 1:j);
                    transB = 'T', alpha = -1.0, beta = 1.0
                )
            end
            _PROF[1] += time_ns() - t
        end
        t = time_ns()
        panel_clim!(view(A, (j + 1):n, (j + 1):n), n - j, b)
        _PROF[2] += time_ns() - t
        j += b
    end
    return A
end

for n in (2048, 4096)
    M = spd(n)
    L0 = copy(M); LAPACK.potrf!('L', L0)
    bo = @be spd(n) (M -> (LAPACK.potrf!('L', M); M[1, 1])) evals = 1 samples = 30 seconds = 2.5
    bstd = @be spd(n) (M -> (PureBLAS.potrf!(M; uplo = 'L'); M[1, 1])) evals = 1 samples = 30 seconds = 2.5
    for NB2 in (512, 1024)
        # profile split (cross-update = what packed-history could speed up) over N runs
        _PROF .= 0.0; runs = 0
        b2 = @be spd(n) (M -> (chol_lazy2!(M, n; NB2 = NB2); runs += 1; M[1, 1])) evals = 1 samples = 30 seconds = 2.5
        cross_pct = 100 * _PROF[1] / (_PROF[1] + _PROF[2])
        println("n=$n NB2=$NB2  std/OB=$(round(_t(bo) / _t(bstd), digits = 3))  lazy2/OB=$(round(_t(bo) / _t(b2), digits = 3))  cross-update=$(round(cross_pct))% of lazy (packed-history can only touch this)")
    end
end
