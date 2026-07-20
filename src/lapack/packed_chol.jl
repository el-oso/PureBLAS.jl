# LAPACK packed Cholesky — dpptrf / dpptrs, generic over s/d/c/z (Hermitian for complex; conj folds to
# identity on real). Ports Reference-LAPACK dpptrf.f (left-looking upper via tpsv+dot, right-looking
# lower via scal+spr) and dpptrs.f.
#
# LAPACK packed storage `AP` is a length-n(n+1)/2 vector holding one triangle column-by-column:
#   uplo='U': AP[i + (j-1)·j÷2]        = A[i,j]  for i ≤ j   (columns of the upper triangle)
#   uplo='L': AP[i + (j-1)·(2n-j)÷2]   = A[i,j]  for i ≥ j   (columns of the lower triangle)

@inline _pp_u(i::Int, j::Int) = i + ((j - 1) * j) >> 1                 # upper packed index, i ≤ j
@inline _pp_l(i::Int, j::Int, n::Int) = i + ((j - 1) * (2n - j)) >> 1  # lower packed index, i ≥ j

# ── pptrf!: packed Cholesky factorization (dpptrf.f) ──────────────────────────────────────────────
# uplo='U': A = Uᴴ·U (left-looking — solve Uᴴu = col then the diagonal). uplo='L': A = L·Lᴴ
# (right-looking — scale column, rank-1 downdate). Overwrites AP. Throws PosDefException.
function pptrf!(AP::AbstractVector; uplo::AbstractChar = 'L')
    n = _pp_order(length(AP))
    if uplo == 'U'
        @inbounds for j in 1:n
            for k in 1:(j - 1)                                # solve Uᴴ·u = A[1:j-1, j] (forward subst)
                s = AP[_pp_u(k, j)]
                for i in 1:(k - 1)
                    s -= conj(AP[_pp_u(i, k)]) * AP[_pp_u(i, j)]   # conj(U[i,k])·u[i]
                end
                AP[_pp_u(k, j)] = s / AP[_pp_u(k, k)]     # U[k,k] real
            end
            ajj = real(AP[_pp_u(j, j)])                   # U[j,j]² = A[j,j] − Σ|u[i]|²
            for i in 1:(j - 1)
                ajj -= abs2(AP[_pp_u(i, j)])
            end
            ajj > 0 || throw(PosDefException(j))
            AP[_pp_u(j, j)] = sqrt(ajj)
        end
    elseif uplo == 'L'
        @inbounds for j in 1:n
            ajj = real(AP[_pp_l(j, j, n)])
            ajj > 0 || throw(PosDefException(j))
            ajj = sqrt(ajj); AP[_pp_l(j, j, n)] = ajj; invd = inv(ajj)
            for i in (j + 1):n                              # scale L[j+1:n, j]
                AP[_pp_l(i, j, n)] *= invd
            end
            for q in (j + 1):n                              # rank-1 downdate of the trailing triangle
                lqj = conj(AP[_pp_l(q, j, n)])            # conj(L[q,j])
                for p in q:n                              # p ≥ q (lower)
                    AP[_pp_l(p, q, n)] -= AP[_pp_l(p, j, n)] * lqj
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
