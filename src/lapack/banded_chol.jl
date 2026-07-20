# LAPACK band Cholesky — dpbtrf / dpbtrs, generic over s/d/c/z (Hermitian for complex; conj folds to
# identity on real, so the real/AD path is byte-identical). Ports Reference-LAPACK dpbtf2.f (the
# unblocked right-looking factor) and dpbtrs.f (two band triangular solves).
#
# LAPACK band storage `AB` is a (kd+1)×n matrix (kd = number of super/sub-diagonals):
#   uplo='U': AB[kd+1 + (i-j), j] = A[i,j]  for  max(1,j-kd) ≤ i ≤ j   (diagonal at row kd+1)
#   uplo='L': AB[1 + (i-j),    j] = A[i,j]  for  j ≤ i ≤ min(n,j+kd)   (diagonal at row 1)
# Only the band is referenced/overwritten; factor L (or U) overwrites AB in the same layout.

# ── pbtrf!: band Cholesky factorization (dpbtf2.f, unblocked right-looking) ────────────────────────
# uplo='L': A = L·Lᴴ, uplo='U': A = Uᴴ·U. Overwrites AB with the factor. Throws PosDefException at the
# first non-positive pivot. `kd` defaults to size(AB,1)-1.
function pbtrf!(AB::AbstractMatrix; uplo::AbstractChar = 'L', kd::Integer = size(AB, 1) - 1)
    n = size(AB, 2)
    size(AB, 1) >= kd + 1 || throw(DimensionMismatch("pbtrf!: size(AB,1) must be ≥ kd+1"))
    if uplo == 'L'
        @inbounds for j in 1:n
            ajj = real(AB[1, j])                          # diagonal is real (Hermitian)
            ajj > 0 || throw(PosDefException(j))
            ajj = sqrt(ajj); AB[1, j] = ajj; invd = inv(ajj)
            kn = min(kd, n - j)
            for i in 1:kn                                 # scale L[j+1:j+kn, j] by 1/√ajj
                AB[1 + i, j] *= invd
            end
            for jc in 1:kn                                # trailing rank-1 update within the band
                lcj = conj(AB[1 + jc, j])                   # conj(L[j+jc, j])
                for ir in jc:kn                           # row j+ir ≥ col j+jc (lower)
                    AB[1 + (ir - jc), j + jc] -= AB[1 + ir, j] * lcj
                end
            end
        end
    elseif uplo == 'U'
        @inbounds for j in 1:n
            ajj = real(AB[kd + 1, j])
            ajj > 0 || throw(PosDefException(j))
            ajj = sqrt(ajj); AB[kd + 1, j] = ajj; invd = inv(ajj)
            kn = min(kd, n - j)
            for ir in 1:kn                                # scale row U[j, j+1:j+kn] by 1/√ajj
                AB[kd + 1 - ir, j + ir] *= invd
            end
            for ir in 1:kn                                # trailing update A22 -= U12ᴴ·U12
                ujr = AB[kd + 1 - ir, j + ir]                   # U[j, j+ir]
                for ic in 1:ir                            # row j+ic ≤ col j+ir (upper)
                    AB[kd + 1 - (ir - ic), j + ir] -= conj(AB[kd + 1 - ic, j + ic]) * ujr
                end
            end
        end
    else
        throw(ArgumentError("pbtrf!: uplo must be 'L' or 'U'"))
    end
    return AB
end

# ── pbtrs!: solve A·X = B with the pbtrf! factor (dpbtrs.f) ────────────────────────────────────────
# Two band triangular solves per RHS. uplo='L' (A=L·Lᴴ): L then Lᴴ. uplo='U' (A=Uᴴ·U): Uᴴ then U.
function pbtrs!(
        AB::AbstractMatrix, B::AbstractVecOrMat; uplo::AbstractChar = 'L',
        kd::Integer = size(AB, 1) - 1
    )
    n = size(AB, 2)
    Bm = _gt_asmat(B); size(Bm, 1) == n || throw(DimensionMismatch("pbtrs!: size(B,1) must equal n"))
    nrhs = size(Bm, 2)
    if uplo == 'L'
        @inbounds for j in 1:nrhs
            for k in 1:n                                  # L·y = b (forward)
                yk = Bm[k, j] / AB[1, k]; Bm[k, j] = yk
                for i in (k + 1):min(k + kd, n)
                    Bm[i, j] -= AB[1 + (i - k), k] * yk   # L[i,k]
                end
            end
            for k in n:-1:1                               # Lᴴ·x = y (backward)
                s = Bm[k, j]
                for i in (k + 1):min(k + kd, n)
                    s -= conj(AB[1 + (i - k), k]) * Bm[i, j]   # conj(L[i,k])
                end
                Bm[k, j] = s / AB[1, k]                   # L[k,k] real
            end
        end
    elseif uplo == 'U'
        @inbounds for j in 1:nrhs
            for k in 1:n                                  # Uᴴ·y = b (forward)
                yk = Bm[k, j] / AB[kd + 1, k]; Bm[k, j] = yk
                for i in (k + 1):min(k + kd, n)
                    Bm[i, j] -= conj(AB[kd + 1 - (i - k), i]) * yk   # conj(U[k,i])
                end
            end
            for k in n:-1:1                               # U·x = y (backward)
                s = Bm[k, j]
                for i in (k + 1):min(k + kd, n)
                    s -= AB[kd + 1 - (i - k), i] * Bm[i, j]  # U[k,i]
                end
                Bm[k, j] = s / AB[kd + 1, k]                # U[k,k] real
            end
        end
    else
        throw(ArgumentError("pbtrs!: uplo must be 'L' or 'U'"))
    end
    return B
end
