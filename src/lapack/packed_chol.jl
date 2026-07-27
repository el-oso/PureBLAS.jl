# LAPACK packed Cholesky — dpptrf / dpptrs, generic over s/d/c/z (Hermitian for complex; conj folds to
# identity on real). Ports Reference-LAPACK dpptrf.f (left-looking upper via tpsv+dot, right-looking
# lower via scal+spr) and dpptrs.f.
#
# LAPACK packed storage `AP` is a length-n(n+1)/2 vector holding one triangle column-by-column:
#   uplo='U': AP[i + (j-1)·j÷2]        = A[i,j]  for i ≤ j   (columns of the upper triangle)
#   uplo='L': AP[i + (j-1)·(2n-j)÷2]   = A[i,j]  for i ≥ j   (columns of the lower triangle)

@inline _pp_u(i::Int, j::Int) = i + ((j - 1) * j) >> 1                 # upper packed index, i ≤ j
@inline _pp_l(i::Int, j::Int, n::Int) = i + ((j - 1) * (2n - j)) >> 1  # lower packed index, i ≥ j

# Column length below which pptrf!'s upper path forward-substitutes inline instead of calling tpsv!.
# req#8 tier: MEASURE (an algorithm-switch crossover set by call overhead vs vectorised work, not by
# cache residency — the same class as `_GETF2_BASE`), stated rather than dressed up as a derivation.
# Zen4, PB/OpenBLAS, sweeping the cutoff (rows = order n, best per row marked):
#     n=16   cut0 0.913  cut16 1.418  cut32 1.418*        n=96   cut0 0.897  cut32 0.920*  cut64 0.850
#     n=32   cut0 0.883  cut16 0.978  cut32 1.168*        n=128  cut0 0.913  cut32 0.926*  cut64 0.887
#     n=64   cut0 0.884  cut16 0.904  cut32 0.940*        n=512  cut0 0.962  cut32 0.963*  cut96 0.950
# 32 is best-or-within-noise at every order measured; 0 (always tpsv) costs 1.42→0.91 at n=16 and
# 1.17→0.88 at n=32, i.e. it turns two passing sizes into misses.
# ⚠ MEASURED ON ZEN4 ONLY — galen was off the network and Zen5 is not yet re-locked. Needs fleet
# validation before it is trusted to extrapolate (req#8: derive → validate on the fleet → ship).
const _PPTRF_TPSV_MIN = 32

# ── pptrf!: packed Cholesky factorization (dpptrf.f) ──────────────────────────────────────────────
# uplo='U': A = Uᴴ·U (left-looking — solve Uᴴu = col then the diagonal). uplo='L': A = L·Lᴴ
# (right-looking — scale column, rank-1 downdate). Overwrites AP. Throws PosDefException.
function pptrf!(AP::AbstractVector; uplo::AbstractChar = 'L')
    n = _pp_order(length(AP))
    if uplo == 'U'
        # Left-looking, exactly as dpptrf.f does it: per column, ONE packed triangular solve
        # (Uᴴ·u = A[1:j-1, j]) then a dot product for the pivot. The previous form inlined both as a
        # scalar triple loop; the hand-rolled dot carries a loop-dependency LLVM will not vectorise, so
        # it ran the whole O(n³) left-look at scalar speed while the reference used optimised tpsv/dot.
        # Measured on Zen3 vs OpenBLAS before this: 0.55 at n=128 falling to 0.19 at n=1024 (5.3× SLOWER).
        # Both operands are contiguous in upper-packed storage — the leading order-(j-1) triangle is the
        # prefix AP[1:jc], and column j is AP[jc+1 : jc+j-1] — so no packing or copy is needed.
        tr = eltype(AP) <: Complex ? 'C' : 'T'
        @inbounds for j in 1:n
            jc = ((j - 1) * j) >> 1                       # 0-based start of column j
            ajj = real(AP[jc + j])
            if j > 1
                if (j - 1) <= _PPTRF_TPSV_MIN             # short column: inline, no call to amortise
                    for k in 1:(j - 1)
                        s = AP[jc + k]
                        for i in 1:(k - 1)
                            s -= conj(AP[_pp_u(i, k)]) * AP[jc + i]
                        end
                        AP[jc + k] = s / AP[_pp_u(k, k)]
                    end
                else
                    tpsv!(view(AP, 1:jc), view(AP, (jc + 1):(jc + j - 1)); uplo = 'U', trans = tr, diag = 'N')
                end
                for i in 1:(j - 1)                        # U[j,j]² = A[j,j] − Σ|u[i]|²
                    ajj -= abs2(AP[jc + i])
                end
            end
            ajj > 0 || throw(PosDefException(j))
            AP[jc + j] = sqrt(ajj)
        end
    elseif uplo == 'L'
        # Right-looking, as dpptrf.f does it: scale the column, then ONE packed rank-1 downdate of the
        # trailing triangle (dspr/zhpr). In lower-packed storage columns j+1..n follow column j directly,
        # so the trailing triangle of order n-j is exactly the contiguous tail AP[_pp_l(j+1,j+1,n):end]
        # and the multiplier column is the contiguous run AP[_pp_l(j+1,j,n) .. _pp_l(n,j,n)] — spr!/hpr!
        # can take both as views with no packing or copy.
        cplx = eltype(AP) <: Complex
        @inbounds for j in 1:n
            ajj = real(AP[_pp_l(j, j, n)])
            ajj > 0 || throw(PosDefException(j))
            ajj = sqrt(ajj); AP[_pp_l(j, j, n)] = ajj; invd = inv(ajj)
            m = n - j
            m == 0 && continue
            for i in (j + 1):n                              # scale L[j+1:n, j]
                AP[_pp_l(i, j, n)] *= invd
            end
            if m > _PPTRF_TPSV_MIN
                xs = _pp_l(j + 1, j, n)
                x = view(AP, xs:(xs + m - 1))
                A22 = view(AP, _pp_l(j + 1, j + 1, n):length(AP))
                if cplx
                    hpr!(-one(real(eltype(AP))), x, A22; uplo = 'L')
                else
                    spr!(-one(eltype(AP)), x, A22; uplo = 'L')
                end
            else
                for q in (j + 1):n                          # short trailing block: inline downdate
                    lqj = conj(AP[_pp_l(q, j, n)])        # conj(L[q,j])
                    for p in q:n                          # p ≥ q (lower)
                        AP[_pp_l(p, q, n)] -= AP[_pp_l(p, j, n)] * lqj
                    end
                end
            end
        end
    else
        throw(ArgumentError("pptrf!: uplo must be 'L' or 'U'"))
    end
    return AP
end

# ── pptrs!: solve A·X = B with the pptrf! factor (dpptrs.f) ────────────────────────────────────────
function pptrs!(AP::AbstractVector, B::AbstractVecOrMat; uplo::AbstractChar = 'L')
    n = _pp_order(length(AP))
    Bm = _gt_asmat(B); size(Bm, 1) == n || throw(DimensionMismatch("pptrs!: size(B,1) must equal n"))
    nrhs = size(Bm, 2)
    if uplo == 'U'
        @inbounds for j in 1:nrhs
            for k in 1:n                                  # Uᴴ·y = b (forward)
                s = Bm[k, j]
                for i in 1:(k - 1)
                    s -= conj(AP[_pp_u(i, k)]) * Bm[i, j]
                end
                Bm[k, j] = s / AP[_pp_u(k, k)]
            end
            for k in n:-1:1                               # U·x = y (backward)
                s = Bm[k, j]
                for i in (k + 1):n
                    s -= AP[_pp_u(k, i)] * Bm[i, j]
                end
                Bm[k, j] = s / AP[_pp_u(k, k)]
            end
        end
    elseif uplo == 'L'
        @inbounds for j in 1:nrhs
            for k in 1:n                                  # L·y = b (forward)
                s = Bm[k, j]
                for i in 1:(k - 1)
                    s -= AP[_pp_l(k, i, n)] * Bm[i, j]
                end
                Bm[k, j] = s / AP[_pp_l(k, k, n)]
            end
            for k in n:-1:1                               # Lᴴ·x = y (backward)
                s = Bm[k, j]
                for i in (k + 1):n
                    s -= conj(AP[_pp_l(i, k, n)]) * Bm[i, j]
                end
                Bm[k, j] = s / AP[_pp_l(k, k, n)]
            end
        end
    else
        throw(ArgumentError("pptrs!: uplo must be 'L' or 'U'"))
    end
    return B
end

# Recover n from a packed length L = n(n+1)/2 (exact integer inverse; validates the length).
@inline function _pp_order(L::Int)
    n = (isqrt(8L + 1) - 1) >> 1
    (n * (n + 1)) >> 1 == L || throw(DimensionMismatch("packed length $L is not n(n+1)/2"))
    return n
end
