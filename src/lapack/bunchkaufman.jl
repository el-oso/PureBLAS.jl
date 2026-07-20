# Bunch-Kaufman factorization of a symmetric-indefinite / Hermitian matrix.
#
#   sytrf!  A = L·D·Lᵀ  (uplo='L')  or  U·D·Uᵀ  (uplo='U')   — symmetric (real OR complex-symmetric)
#   hetrf!  A = L·D·Lᴴ  (uplo='L')  or  U·D·Uᴴ  (uplo='U')   — Hermitian (complex; real ⇒ same as sytrf!)
#
# D is block-diagonal with 1×1 and 2×2 pivots. `ipiv` follows the LAPACK convention:
#   ipiv[k] > 0                         : 1×1 pivot, rows/cols k and ipiv[k] were interchanged.
#   ipiv[k] = ipiv[k-1] < 0 (uplo='L')  : 2×2 pivot in (k,k+1); rows/cols k+1 and -ipiv[k] interchanged.
#   ipiv[k] = ipiv[k+1] < 0 (uplo='U')  : 2×2 pivot in (k-1,k); rows/cols k-1 and -ipiv[k] interchanged.
#
# The pivot decision copies LAPACK's `dsytf2`/`zhetf2`/`zsytf2` EXACTLY: the α=(1+√17)/8 threshold and
# the 1×1-vs-2×2 selection (ABSAKK ≥ α·COLMAX → 1×1; else the ROWMAX branch). This is the UNBLOCKED
# Bunch-Kaufman (partial pivoting, LAPACK default `dsytrf`'s base case). Generic scalar loops keep the
# path AD-traceable and trim-safe over any T<:Number; the rank-1/rank-2 trailing updates are written as
# explicit triangular loops (the LAPACK `dsyr`/`dsyr2`/`dger` calls).
#
# `sytrs!`/`hetrs!` solve A·X=B from the factors (mirrors `dsytrs`/`zhetrs`/`zsytrs`).
#
# ponytail: scalar-generic, not SIMD-tuned — correctness-first standalone kernel; the trailing update is
# the syr/ger hot spot to vectorize later when assembled behind the L2 fast paths.

# LAPACK CABS1 / IDAMAX-IZAMAX magnitude: |re|+|im| (== abs on reals). Used for all off-diagonal
# comparisons and the column/row index-max, matching idamax (real) and izamax (complex).
@inline _bk_cabs1(x::Real) = abs(x)
@inline _bk_cabs1(x::Complex) = abs(real(x)) + abs(imag(x))

const _BK_ALPHA = (1 + sqrt(17.0)) / 8   # (1+√17)/8 ≈ 0.6403882, the Bunch-Kaufman pivot threshold

# ---------------------------------------------------------------------------------------------------
# Factorization
# ---------------------------------------------------------------------------------------------------

# Index of the max |·| (CABS1) entry among A[rows, col]; returns the row index. rows must be non-empty.
@inline function _bk_colmax(A, rows::UnitRange{Int}, col::Int)
    imax = first(rows); cmax = _bk_cabs1(A[imax, col])
    @inbounds for i in rows
        v = _bk_cabs1(A[i, col])
        if v > cmax
            cmax = v; imax = i
        end
    end
    return imax, cmax
end
# Index of the max |·| entry among A[row, cols] (a row scan); returns the col index.
@inline function _bk_rowmax(A, row::Int, cols::UnitRange{Int})
    jmax = first(cols); rmax = _bk_cabs1(A[row, jmax])
    @inbounds for j in cols
        v = _bk_cabs1(A[row, j])
        if v > rmax
            rmax = v; jmax = j
        end
    end
    return jmax, rmax
end

# uplo='L': A = L·D·Lᵀ (herm=false) or L·D·Lᴴ (herm=true). Reads/writes only the lower triangle.
function _sytf2_lower!(A::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer}, herm::Bool) where {T}
    n = size(A, 1)
    Tr = real(T)
    alpha = Tr(_BK_ALPHA)
    info = 0
    k = 1
    @inbounds while k <= n
        kstep = 1
        absakk = herm ? abs(real(A[k, k])) : _bk_cabs1(A[k, k])
        if k < n
            imax, colmax = _bk_colmax(A, (k + 1):n, k)
        else
            imax = k; colmax = zero(Tr)
        end
        kp = k
        if max(absakk, colmax) == 0
            info == 0 && (info = k)
            kp = k
            herm && (A[k, k] = real(A[k, k]))
        else
            if absakk >= alpha * colmax
                kp = k                                   # 1×1, no interchange
            else
                # ROWMAX: largest off-diagonal in row imax (cols k..imax-1 and rows imax+1..n of col imax)
                _, rowmax = _bk_rowmax(A, imax, k:(imax - 1))
                if imax < n
                    _, rm2 = _bk_colmax(A, (imax + 1):n, imax)
                    rowmax = max(rowmax, rm2)
                end
                if absakk >= alpha * colmax * (colmax / rowmax)
                    kp = k                               # 1×1 pivot
                elseif (herm ? abs(real(A[imax, imax])) : _bk_cabs1(A[imax, imax])) >= alpha * rowmax
                    kp = imax                            # 1×1 pivot, interchange k <-> imax
                else
                    kp = imax; kstep = 2                 # 2×2 pivot
                end
            end
            kk = k + kstep - 1
            if kp != kk
                # interchange rows/cols kk and kp in the trailing submatrix A(k:n,k:n)
                if kp < n
                    for i in (kp + 1):n
                        t = A[i, kk]; A[i, kk] = A[i, kp]; A[i, kp] = t
                    end
                end
                # the crossing strip A(kk+1:kp-1, kk) <-> A(kp, kk+1:kp-1) (conj for Hermitian)
                for j in (kk + 1):(kp - 1)
                    if herm
                        t = conj(A[j, kk]); A[j, kk] = conj(A[kp, j]); A[kp, j] = t
                    else
                        t = A[j, kk]; A[j, kk] = A[kp, j]; A[kp, j] = t
                    end
                end
                herm && (A[kp, kk] = conj(A[kp, kk]))
                if herm
                    r1 = real(A[kk, kk]); A[kk, kk] = real(A[kp, kp]); A[kp, kp] = r1
                else
                    t = A[kk, kk]; A[kk, kk] = A[kp, kp]; A[kp, kp] = t
                end
                if kstep == 2
                    t = A[k + 1, k]; A[k + 1, k] = A[kp, k]; A[kp, k] = t
                end
            elseif herm
                A[k, k] = real(A[k, k])
                kstep == 2 && (A[k + 1, k + 1] = real(A[k + 1, k + 1]))
            end

            if kstep == 1
                if k < n
                    if herm
                        r1 = one(Tr) / real(A[k, k])
                        for j in (k + 1):n               # rank-1 downdate of lower A(k+1:n,k+1:n)
                            wj = r1 * conj(A[j, k])
                            for i in j:n
                                A[i, j] -= A[i, k] * wj
                            end
                            A[j, k] *= r1                # scale column below diagonal
                        end
                        A[k, k] = real(A[k, k])
                    else
                        d11 = one(T) / A[k, k]
                        for j in (k + 1):n
                            wj = d11 * A[j, k]
                            for i in j:n
                                A[i, j] -= A[i, k] * wj
                            end
                            A[j, k] *= d11
                        end
                    end
                end
            else                                          # 2×2 pivot, columns k and k+1
                if k < n - 1
                    if herm
                        d = hypot(real(A[k + 1, k]), imag(A[k + 1, k]))
                        d11 = real(A[k + 1, k + 1]) / d
                        d22 = real(A[k, k]) / d
                        tt = one(Tr) / (d11 * d22 - one(Tr))
                        d21 = A[k + 1, k] / d
                        dm = tt / d
                        for j in (k + 2):n
                            wk = dm * (d11 * A[j, k] - d21 * A[j, k + 1])
                            wkp1 = dm * (d22 * A[j, k + 1] - conj(d21) * A[j, k])
                            for i in j:n
                                A[i, j] -= A[i, k] * conj(wk) + A[i, k + 1] * conj(wkp1)
                            end
                            A[j, k] = wk; A[j, k + 1] = wkp1
                            A[j, j] = real(A[j, j])
                        end
                    else
                        d21 = A[k + 1, k]
                        d11 = A[k + 1, k + 1] / d21
                        d22 = A[k, k] / d21
                        tt = one(T) / (d11 * d22 - one(T))
                        d21 = tt / d21
                        for j in (k + 2):n
                            wk = d21 * (d11 * A[j, k] - A[j, k + 1])
                            wkp1 = d21 * (d22 * A[j, k + 1] - A[j, k])
                            for i in j:n
                                A[i, j] -= A[i, k] * wk + A[i, k + 1] * wkp1
                            end
                            A[j, k] = wk; A[j, k + 1] = wkp1
                        end
                    end
                end
            end
        end
        if kstep == 1
            ipiv[k] = kp
        else
            ipiv[k] = -kp; ipiv[k + 1] = -kp
        end
        k += kstep
    end
    return info
end

# uplo='U': A = U·D·Uᵀ (herm=false) or U·D·Uᴴ (herm=true). Reads/writes only the upper triangle.
function _sytf2_upper!(A::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer}, herm::Bool) where {T}
    n = size(A, 1)
    Tr = real(T)
    alpha = Tr(_BK_ALPHA)
    info = 0
    k = n
    @inbounds while k >= 1
        kstep = 1
        absakk = herm ? abs(real(A[k, k])) : _bk_cabs1(A[k, k])
        if k > 1
            imax, colmax = _bk_colmax(A, 1:(k - 1), k)
        else
            imax = k; colmax = zero(Tr)
        end
        kp = k
        if max(absakk, colmax) == 0
            info == 0 && (info = k)
            kp = k
            herm && (A[k, k] = real(A[k, k]))
        else
            if absakk >= alpha * colmax
                kp = k
            else
                _, rowmax = _bk_rowmax(A, imax, (imax + 1):k)
                if imax > 1
                    _, rm2 = _bk_colmax(A, 1:(imax - 1), imax)
                    rowmax = max(rowmax, rm2)
                end
                if absakk >= alpha * colmax * (colmax / rowmax)
                    kp = k
                elseif (herm ? abs(real(A[imax, imax])) : _bk_cabs1(A[imax, imax])) >= alpha * rowmax
                    kp = imax
                else
                    kp = imax; kstep = 2
                end
            end
            kk = k - kstep + 1
            if kp != kk
                if kp > 1
                    for i in 1:(kp - 1)
                        t = A[i, kk]; A[i, kk] = A[i, kp]; A[i, kp] = t
                    end
                end
                for j in (kp + 1):(kk - 1)
                    if herm
                        t = conj(A[j, kk]); A[j, kk] = conj(A[kp, j]); A[kp, j] = t
                    else
                        t = A[j, kk]; A[j, kk] = A[kp, j]; A[kp, j] = t
                    end
                end
                herm && (A[kp, kk] = conj(A[kp, kk]))
                if herm
                    r1 = real(A[kk, kk]); A[kk, kk] = real(A[kp, kp]); A[kp, kp] = r1
                else
                    t = A[kk, kk]; A[kk, kk] = A[kp, kp]; A[kp, kp] = t
                end
                if kstep == 2
                    t = A[k - 1, k]; A[k - 1, k] = A[kp, k]; A[kp, k] = t
                end
            elseif herm
                A[k, k] = real(A[k, k])
                kstep == 2 && (A[k - 1, k - 1] = real(A[k - 1, k - 1]))
            end

            if kstep == 1
                if k > 1
                    if herm
                        r1 = one(Tr) / real(A[k, k])
                        for j in 1:(k - 1)                # rank-1 downdate on ORIGINAL column k
                            wj = r1 * conj(A[j, k])
                            for i in 1:j
                                A[i, j] -= A[i, k] * wj
                            end
                        end
                        for j in 1:(k - 1)
                            A[j, k] *= r1
                        end   # then scale column (DSCAL after DSYR)
                        A[k, k] = real(A[k, k])
                    else
                        d11 = one(T) / A[k, k]
                        for j in 1:(k - 1)
                            wj = d11 * A[j, k]
                            for i in 1:j
                                A[i, j] -= A[i, k] * wj
                            end
                        end
                        for j in 1:(k - 1)
                            A[j, k] *= d11
                        end
                    end
                end
            else                                          # 2×2 pivot, columns k and k-1
                if k > 2
                    if herm
                        d = hypot(real(A[k - 1, k]), imag(A[k - 1, k]))
                        d11 = real(A[k, k]) / d
                        d22 = real(A[k - 1, k - 1]) / d
                        tt = one(Tr) / (d11 * d22 - one(Tr))
                        d12 = A[k - 1, k] / d
                        dm = tt / d
                        for j in (k - 2):-1:1
                            wkm1 = dm * (d11 * A[j, k - 1] - conj(d12) * A[j, k])
                            wk = dm * (d22 * A[j, k] - d12 * A[j, k - 1])
                            for i in j:-1:1
                                A[i, j] -= A[i, k] * conj(wk) + A[i, k - 1] * conj(wkm1)
                            end
                            A[j, k] = wk; A[j, k - 1] = wkm1
                            A[j, j] = real(A[j, j])
                        end
                    else
                        d12 = A[k - 1, k]
                        d22 = A[k - 1, k - 1] / d12
                        d11 = A[k, k] / d12
                        tt = one(T) / (d11 * d22 - one(T))
                        d12 = tt / d12
                        for j in (k - 2):-1:1
                            wkm1 = d12 * (d11 * A[j, k - 1] - A[j, k])
                            wk = d12 * (d22 * A[j, k] - A[j, k - 1])
                            for i in j:-1:1
                                A[i, j] -= A[i, k] * wk + A[i, k - 1] * wkm1
                            end
                            A[j, k] = wk; A[j, k - 1] = wkm1
                        end
                    end
                end
            end
        end
        if kstep == 1
            ipiv[k] = kp
        else
            ipiv[k] = -kp; ipiv[k - 1] = -kp
        end
        k -= kstep
    end
    return info
end

"""
    sytrf!(A, ipiv; uplo='L') -> info

Bunch-Kaufman factorization of a symmetric (or complex-symmetric) matrix A = L·D·Lᵀ / U·D·Uᵀ,
in place in the `uplo` triangle. `ipiv` (length n) receives the LAPACK pivot encoding. Returns
`info` (0, or the index of the first zero pivot block). Generic over T<:Number.
"""
function sytrf!(A::AbstractMatrix, ipiv::AbstractVector{<:Integer}; uplo::Char = 'L')
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("sytrf!: A must be square"))
    length(ipiv) == n || throw(DimensionMismatch("sytrf!: length(ipiv) must equal size(A,1)"))
    (uplo == 'L' || uplo == 'U') || throw(ArgumentError("sytrf!: uplo must be 'L' or 'U'"))
    return uplo == 'L' ? _sytf2_lower!(A, ipiv, false) : _sytf2_upper!(A, ipiv, false)
end

"""
    hetrf!(A, ipiv; uplo='L') -> info

Bunch-Kaufman factorization of a Hermitian matrix A = L·D·Lᴴ / U·D·Uᴴ. For real `eltype(A)` this is
identical to `sytrf!` (real symmetric). Otherwise the Hermitian variant (real diagonal, conjugated
off-diagonals) is used, mirroring LAPACK `zhetf2`.
"""
function hetrf!(A::AbstractMatrix, ipiv::AbstractVector{<:Integer}; uplo::Char = 'L')
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("hetrf!: A must be square"))
    length(ipiv) == n || throw(DimensionMismatch("hetrf!: length(ipiv) must equal size(A,1)"))
    (uplo == 'L' || uplo == 'U') || throw(ArgumentError("hetrf!: uplo must be 'L' or 'U'"))
    herm = eltype(A) <: Complex
    return uplo == 'L' ? _sytf2_lower!(A, ipiv, herm) : _sytf2_upper!(A, ipiv, herm)
end

# ---------------------------------------------------------------------------------------------------
# Solve  A·X = B  from the factors (dsytrs / zhetrs / zsytrs)
# ---------------------------------------------------------------------------------------------------

@inline _bk_swap_rows!(B, r1::Int, r2::Int) = @inbounds for j in axes(B, 2)
    t = B[r1, j]; B[r1, j] = B[r2, j]; B[r2, j] = t
end

function _sytrs_lower!(A, ipiv, B, herm::Bool)
    n = size(A, 1); nrhs = size(B, 2)
    @inbounds begin
        # First solve L·D·X = B (forward, k = 1..n)
        k = 1
        while k <= n
            if ipiv[k] > 0                               # 1×1
                kp = ipiv[k]
                kp != k && _bk_swap_rows!(B, k, kp)
                for j in 1:nrhs, i in (k + 1):n
                    B[i, j] -= A[i, k] * B[k, j]
                end
                dk = herm ? real(A[k, k]) : A[k, k]
                for j in 1:nrhs
                    B[k, j] /= dk
                end
                k += 1
            else                                         # 2×2, rows (k,k+1)
                kp = -ipiv[k]
                kp != k + 1 && _bk_swap_rows!(B, k + 1, kp)
                for j in 1:nrhs, i in (k + 2):n
                    B[i, j] -= A[i, k] * B[k, j] + A[i, k + 1] * B[k + 1, j]
                end
                akm1k = A[k + 1, k]
                akm1 = A[k, k] / (herm ? conj(akm1k) : akm1k)
                ak = A[k + 1, k + 1] / akm1k
                denom = akm1 * ak - one(eltype(A))
                for j in 1:nrhs
                    bkm1 = B[k, j] / (herm ? conj(akm1k) : akm1k)
                    bk = B[k + 1, j] / akm1k
                    B[k, j] = (ak * bkm1 - bk) / denom
                    B[k + 1, j] = (akm1 * bk - bkm1) / denom
                end
                k += 2
            end
        end
        # Then solve Lᵀ·X = B (or Lᴴ) (backward, k = n..1)
        k = n
        while k >= 1
            if ipiv[k] > 0                               # 1×1
                for j in 1:nrhs
                    s = zero(eltype(B))
                    for i in (k + 1):n
                        s += (herm ? conj(A[i, k]) : A[i, k]) * B[i, j]
                    end
                    B[k, j] -= s
                end
                kp = ipiv[k]
                kp != k && _bk_swap_rows!(B, k, kp)
                k -= 1
            else                                         # 2×2, rows (k-1,k)
                for j in 1:nrhs
                    s1 = zero(eltype(B)); s2 = zero(eltype(B))
                    for i in (k + 1):n
                        s1 += (herm ? conj(A[i, k]) : A[i, k]) * B[i, j]
                        s2 += (herm ? conj(A[i, k - 1]) : A[i, k - 1]) * B[i, j]
                    end
                    B[k, j] -= s1; B[k - 1, j] -= s2
                end
                kp = -ipiv[k]
                kp != k && _bk_swap_rows!(B, k, kp)
                k -= 2
            end
        end
    end
    return B
end

function _sytrs_upper!(A, ipiv, B, herm::Bool)
    n = size(A, 1); nrhs = size(B, 2)
    @inbounds begin
        # First solve U·D·X = B (backward, k = n..1)
        k = n
        while k >= 1
            if ipiv[k] > 0                               # 1×1
                kp = ipiv[k]
                kp != k && _bk_swap_rows!(B, k, kp)
                for j in 1:nrhs, i in 1:(k - 1)
                    B[i, j] -= A[i, k] * B[k, j]
                end
                dk = herm ? real(A[k, k]) : A[k, k]
                for j in 1:nrhs
                    B[k, j] /= dk
                end
                k -= 1
            else                                         # 2×2, rows (k-1,k)
                kp = -ipiv[k]
                kp != k - 1 && _bk_swap_rows!(B, k - 1, kp)
                for j in 1:nrhs, i in 1:(k - 2)
                    B[i, j] -= A[i, k] * B[k, j] + A[i, k - 1] * B[k - 1, j]
                end
                akm1k = A[k - 1, k]
                akm1 = A[k - 1, k - 1] / akm1k
                ak = A[k, k] / (herm ? conj(akm1k) : akm1k)
                denom = akm1 * ak - one(eltype(A))
                for j in 1:nrhs
                    bkm1 = B[k - 1, j] / akm1k
                    bk = B[k, j] / (herm ? conj(akm1k) : akm1k)
                    B[k - 1, j] = (ak * bkm1 - bk) / denom
                    B[k, j] = (akm1 * bk - bkm1) / denom
                end
                k -= 2
            end
        end
        # Then solve Uᵀ·X = B (or Uᴴ) (forward, k = 1..n)
        k = 1
        while k <= n
            if ipiv[k] > 0                               # 1×1
                for j in 1:nrhs
                    s = zero(eltype(B))
                    for i in 1:(k - 1)
                        s += (herm ? conj(A[i, k]) : A[i, k]) * B[i, j]
                    end
                    B[k, j] -= s
                end
                kp = ipiv[k]
                kp != k && _bk_swap_rows!(B, k, kp)
                k += 1
            else                                         # 2×2, rows (k,k+1)
                for j in 1:nrhs
                    s1 = zero(eltype(B)); s2 = zero(eltype(B))
                    for i in 1:(k - 1)
                        s1 += (herm ? conj(A[i, k]) : A[i, k]) * B[i, j]
                        s2 += (herm ? conj(A[i, k + 1]) : A[i, k + 1]) * B[i, j]
                    end
                    B[k, j] -= s1; B[k + 1, j] -= s2
                end
                kp = -ipiv[k]
                kp != k && _bk_swap_rows!(B, k, kp)
                k += 2
            end
        end
    end
    return B
end

"""
    sytrs!(A, ipiv, B; uplo='L') -> B

Solve A·X = B in place (B overwritten by X) using the symmetric Bunch-Kaufman factors from `sytrf!`.
"""
function sytrs!(A::AbstractMatrix, ipiv::AbstractVector{<:Integer}, B::AbstractVecOrMat; uplo::Char = 'L')
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("sytrs!: A must be square"))
    size(B, 1) == n || throw(DimensionMismatch("sytrs!: size(B,1) must equal size(A,1)"))
    (uplo == 'L' || uplo == 'U') || throw(ArgumentError("sytrs!: uplo must be 'L' or 'U'"))
    Bm = B isa AbstractVector ? reshape(B, n, 1) : B
    uplo == 'L' ? _sytrs_lower!(A, ipiv, Bm, false) : _sytrs_upper!(A, ipiv, Bm, false)
    return B
end

"""
    hetrs!(A, ipiv, B; uplo='L') -> B

Solve A·X = B in place using the Hermitian Bunch-Kaufman factors from `hetrf!` (real ⇒ same as `sytrs!`).
"""
function hetrs!(A::AbstractMatrix, ipiv::AbstractVector{<:Integer}, B::AbstractVecOrMat; uplo::Char = 'L')
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("hetrs!: A must be square"))
    size(B, 1) == n || throw(DimensionMismatch("hetrs!: size(B,1) must equal size(A,1)"))
    (uplo == 'L' || uplo == 'U') || throw(ArgumentError("hetrs!: uplo must be 'L' or 'U'"))
    herm = eltype(A) <: Complex
    Bm = B isa AbstractVector ? reshape(B, n, 1) : B
    uplo == 'L' ? _sytrs_lower!(A, ipiv, Bm, herm) : _sytrs_upper!(A, ipiv, Bm, herm)
    return B
end
