# Symmetric-indefinite / Hermitian LAPACK drivers built on the Bunch-Kaufman factors (bunchkaufman.jl):
#
#   sysv! / hesv!  : one-shot solve  A·X = B         = sytrf!/hetrf! then sytrs!/hetrs!  (trivial compose)
#   sytri! / hetri!: matrix inverse  A⁻¹ from D & L/U = dsytri / zsytri (complex-symmetric) / zhetri (Herm)
#
# `sytri!`/`hetri!` are the real new code: the block back-inversion. From A = L·D·Lᵀ (uplo='L') /
# U·D·Uᵀ (uplo='U'), inv(A) = (Lᵀ)⁻¹·D⁻¹·L⁻¹. LAPACK does it in place, column by column, WITHOUT ever
# forming L⁻¹: invert each D block, then walk k from the trailing end inward computing the k-th column
# of the inverse as  A[·,k] ← -A_trailing·(old A[·,k]),  A[k,k] ← D⁻¹[k,k] - old·new.  The symv against
# the already-inverted trailing block + the two dot products carry the (Lᵀ)⁻¹ and L⁻¹ applications.
# Ported ONE-TO-ONE from reference dsytri/zsytri/zhetri: same loop direction, same interchange (with the
# Hermitian conjugated crossing-strip swap), reusing PureBLAS's own `_symv!`/`_hemv!` + `_dotu`/`_dotc`.
#
# `herm` unifies the three: herm=false covers BOTH real-symmetric (dsytri) and complex-symmetric (zsytri)
# — real diagonal falls out for free, and the 2×2 uses the raw pivot value T (signed real ≡ |·| here; the
# complex square must NOT be an abs). herm=true (zhetri) forces the real diagonal (real(A[k,k]), real of
# the diagonal-correcting dot) and the conjugated (Hermitian) dot/symv/swap.
#
# `syconv!` (dsyconv, the LDLᵀ-packed ⇄ separated-D storage converter) is NOT implemented: neither the
# classic dsytri nor the dsytrs/sysv solve path needs it — they read D straight out of A's stored
# triangle + ipiv. syconv only feeds the Aasen (dsytrf_aa) / rook variants, which PureBLAS does not use.
#
# ponytail: scalar-generic driver reusing the gated L2 symv/hemv; correctness-first, matching
# bunchkaufman.jl. The symv/dot per column are the hot spots (already the gated kernels).

# ── inverse from the factors ──────────────────────────────────────────────────────────────────────

# uplo='L': invert in place from A = L·D·Lᵀ (herm=false) or L·D·Lᴴ (herm=true). `work` is an n-length
# scratch (the LAPACK WORK array). Only the lower triangle is read/written; result is the lower triangle
# of the symmetric/Hermitian inverse.
function _sytri_lower!(
        A::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer}, herm::Bool,
        work::AbstractVector{T}
    ) where {T}
    n = size(A, 1)
    Tr = real(T)
    @inbounds begin
        k = n
        while k >= 1
            if ipiv[k] > 0                                   # 1×1 diagonal block
                A[k, k] = herm ? (one(Tr) / real(A[k, k])) : (one(T) / A[k, k])
                if k < n                                     # compute column k of the inverse
                    m = n - k
                    wv = view(work, 1:m); yv = view(A, (k + 1):n, k)
                    copyto!(wv, yv)
                    Asub = view(A, (k + 1):n, (k + 1):n)
                    herm ? _hemv!(false, m, -one(T), Asub, wv, 1, zero(T), yv, 1) :
                        _symv!(false, m, -one(T), Asub, wv, 1, zero(T), yv, 1)
                    d = herm ? _dotc(m, wv, 1, yv, 1) : _dotu(m, wv, 1, yv, 1)
                    A[k, k] -= herm ? real(d) : d
                end
                kstep = 1
            else                                             # 2×2 diagonal block (columns k-1, k)
                t = herm ? Tr(abs(A[k, k - 1])) : A[k, k - 1]
                ak = herm ? (real(A[k - 1, k - 1]) / t) : (A[k - 1, k - 1] / t)
                akp1 = herm ? (real(A[k, k]) / t) : (A[k, k] / t)
                akkp1 = A[k, k - 1] / t
                d = t * (ak * akp1 - one(Tr))
                A[k - 1, k - 1] = akp1 / d
                A[k, k] = ak / d
                A[k, k - 1] = -akkp1 / d
                if k < n                                     # columns k-1 and k of the inverse
                    m = n - k
                    wv = view(work, 1:m)
                    yk = view(A, (k + 1):n, k); ykm1 = view(A, (k + 1):n, k - 1)
                    Asub = view(A, (k + 1):n, (k + 1):n)
                    copyto!(wv, yk)
                    herm ? _hemv!(false, m, -one(T), Asub, wv, 1, zero(T), yk, 1) :
                        _symv!(false, m, -one(T), Asub, wv, 1, zero(T), yk, 1)
                    dk = herm ? _dotc(m, wv, 1, yk, 1) : _dotu(m, wv, 1, yk, 1)
                    A[k, k] -= herm ? real(dk) : dk
                    A[k, k - 1] -= herm ? _dotc(m, yk, 1, ykm1, 1) : _dotu(m, yk, 1, ykm1, 1)
                    copyto!(wv, ykm1)
                    herm ? _hemv!(false, m, -one(T), Asub, wv, 1, zero(T), ykm1, 1) :
                        _symv!(false, m, -one(T), Asub, wv, 1, zero(T), ykm1, 1)
                    dm = herm ? _dotc(m, wv, 1, ykm1, 1) : _dotu(m, wv, 1, ykm1, 1)
                    A[k - 1, k - 1] -= herm ? real(dm) : dm
                end
                kstep = 2
            end
            # interchange rows/cols k ↔ kp in the trailing block
            kp = abs(ipiv[k])
            if kp != k
                if kp < n
                    for i in (kp + 1):n
                        tmp = A[i, k]; A[i, k] = A[i, kp]; A[i, kp] = tmp
                    end
                end
                for j in (k + 1):(kp - 1)                     # crossing strip A[k+1:kp-1,k] ↔ A[kp,k+1:kp-1]
                    if herm
                        tmp = conj(A[j, k]); A[j, k] = conj(A[kp, j]); A[kp, j] = tmp
                    else
                        tmp = A[j, k]; A[j, k] = A[kp, j]; A[kp, j] = tmp
                    end
                end
                herm && (A[kp, k] = conj(A[kp, k]))
                tmp = A[k, k]; A[k, k] = A[kp, kp]; A[kp, kp] = tmp
                if kstep == 2
                    tmp = A[k, k - 1]; A[k, k - 1] = A[kp, k - 1]; A[kp, k - 1] = tmp
                end
            end
            k -= kstep
        end
    end
    return A
end

# uplo='U': invert in place from A = U·D·Uᵀ / U·D·Uᴴ. Mirror of the lower path (loop k=1..n ascending,
# trailing block is the LEADING submatrix A[1:k-1,1:k-1], 2×2 block is columns k, k+1).
function _sytri_upper!(
        A::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer}, herm::Bool,
        work::AbstractVector{T}
    ) where {T}
    n = size(A, 1)
    Tr = real(T)
    @inbounds begin
        k = 1
        while k <= n
            if ipiv[k] > 0                                   # 1×1 diagonal block
                A[k, k] = herm ? (one(Tr) / real(A[k, k])) : (one(T) / A[k, k])
                if k > 1                                     # compute column k of the inverse
                    m = k - 1
                    wv = view(work, 1:m); yv = view(A, 1:m, k)
                    copyto!(wv, yv)
                    Asub = view(A, 1:m, 1:m)
                    herm ? _hemv!(true, m, -one(T), Asub, wv, 1, zero(T), yv, 1) :
                        _symv!(true, m, -one(T), Asub, wv, 1, zero(T), yv, 1)
                    d = herm ? _dotc(m, wv, 1, yv, 1) : _dotu(m, wv, 1, yv, 1)
                    A[k, k] -= herm ? real(d) : d
                end
                kstep = 1
            else                                             # 2×2 diagonal block (columns k, k+1)
                t = herm ? Tr(abs(A[k, k + 1])) : A[k, k + 1]
                ak = herm ? (real(A[k, k]) / t) : (A[k, k] / t)
                akp1 = herm ? (real(A[k + 1, k + 1]) / t) : (A[k + 1, k + 1] / t)
                akkp1 = A[k, k + 1] / t
                d = t * (ak * akp1 - one(Tr))
                A[k, k] = akp1 / d
                A[k + 1, k + 1] = ak / d
                A[k, k + 1] = -akkp1 / d
                if k > 1                                     # columns k and k+1 of the inverse
                    m = k - 1
                    wv = view(work, 1:m)
                    yk = view(A, 1:m, k); ykp1 = view(A, 1:m, k + 1)
                    Asub = view(A, 1:m, 1:m)
                    copyto!(wv, yk)
                    herm ? _hemv!(true, m, -one(T), Asub, wv, 1, zero(T), yk, 1) :
                        _symv!(true, m, -one(T), Asub, wv, 1, zero(T), yk, 1)
                    dk = herm ? _dotc(m, wv, 1, yk, 1) : _dotu(m, wv, 1, yk, 1)
                    A[k, k] -= herm ? real(dk) : dk
                    A[k, k + 1] -= herm ? _dotc(m, yk, 1, ykp1, 1) : _dotu(m, yk, 1, ykp1, 1)
                    copyto!(wv, ykp1)
                    herm ? _hemv!(true, m, -one(T), Asub, wv, 1, zero(T), ykp1, 1) :
                        _symv!(true, m, -one(T), Asub, wv, 1, zero(T), ykp1, 1)
                    dm = herm ? _dotc(m, wv, 1, ykp1, 1) : _dotu(m, wv, 1, ykp1, 1)
                    A[k + 1, k + 1] -= herm ? real(dm) : dm
                end
                kstep = 2
            end
            # interchange rows/cols k ↔ kp in the leading block
            kp = abs(ipiv[k])
            if kp != k
                for i in 1:(kp - 1)
                    tmp = A[i, k]; A[i, k] = A[i, kp]; A[i, kp] = tmp
                end
                for j in (kp + 1):(k - 1)                     # crossing strip A[kp+1:k-1,k] ↔ A[kp,kp+1:k-1]
                    if herm
                        tmp = conj(A[j, k]); A[j, k] = conj(A[kp, j]); A[kp, j] = tmp
                    else
                        tmp = A[j, k]; A[j, k] = A[kp, j]; A[kp, j] = tmp
                    end
                end
                herm && (A[kp, k] = conj(A[kp, k]))
                tmp = A[k, k]; A[k, k] = A[kp, kp]; A[kp, kp] = tmp
                if kstep == 2
                    tmp = A[k, k + 1]; A[k, k + 1] = A[kp, k + 1]; A[kp, k + 1] = tmp
                end
            end
            k += kstep
        end
    end
    return A
end

"""
    sytri!(A, ipiv; uplo='L') -> A

Overwrite A with the inverse of the symmetric (or complex-symmetric) matrix whose Bunch-Kaufman factors
(`sytrf!` output) are stored in the `uplo` triangle of A with pivots `ipiv`. Only the `uplo` triangle of
A⁻¹ is written (the inverse is symmetric). Generic over T<:Number. Mirrors LAPACK `dsytri`/`zsytri`.
"""
function sytri!(A::AbstractMatrix, ipiv::AbstractVector{<:Integer}; uplo::Char = 'L')
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("sytri!: A must be square"))
    length(ipiv) == n || throw(DimensionMismatch("sytri!: length(ipiv) must equal size(A,1)"))
    (uplo == 'L' || uplo == 'U') || throw(ArgumentError("sytri!: uplo must be 'L' or 'U'"))
    work = Vector{eltype(A)}(undef, n)
    return uplo == 'L' ? _sytri_lower!(A, ipiv, false, work) : _sytri_upper!(A, ipiv, false, work)
end

"""
    hetri!(A, ipiv; uplo='L') -> A

Overwrite A with the inverse of the Hermitian matrix whose Bunch-Kaufman factors (`hetrf!` output) are in
the `uplo` triangle with pivots `ipiv`. For real `eltype(A)` this is identical to `sytri!`. Only the
`uplo` triangle of A⁻¹ is written. Mirrors LAPACK `zhetri` (real diagonal, conjugated off-diagonals).
"""
function hetri!(A::AbstractMatrix, ipiv::AbstractVector{<:Integer}; uplo::Char = 'L')
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("hetri!: A must be square"))
    length(ipiv) == n || throw(DimensionMismatch("hetri!: length(ipiv) must equal size(A,1)"))
    (uplo == 'L' || uplo == 'U') || throw(ArgumentError("hetri!: uplo must be 'L' or 'U'"))
    herm = eltype(A) <: Complex
    work = Vector{eltype(A)}(undef, n)
    return uplo == 'L' ? _sytri_lower!(A, ipiv, herm, work) : _sytri_upper!(A, ipiv, herm, work)
end

# ── one-shot solve  A·X = B  (factor + solve) ───────────────────────────────────────────────────────

"""
    sysv!(uplo, A, B) -> (A, ipiv, B)

Solve the symmetric-indefinite system A·X = B in place: Bunch-Kaufman–factor A (`sytrf!`, in the `uplo`
triangle) then solve (`sytrs!`). On return A holds the factors, `ipiv` the pivots, and B the solution X.
"""
function sysv!(uplo::Char, A::AbstractMatrix, B::AbstractVecOrMat)
    n = size(A, 1)
    ipiv = Vector{Int}(undef, n)
    sytrf!(A, ipiv; uplo = uplo)
    sytrs!(A, ipiv, B; uplo = uplo)
    return A, ipiv, B
end

"""
    hesv!(uplo, A, B) -> (A, ipiv, B)

Solve the Hermitian system A·X = B in place via the Hermitian Bunch-Kaufman factorization (`hetrf!` then
`hetrs!`). For real `eltype(A)` this equals `sysv!`.
"""
function hesv!(uplo::Char, A::AbstractMatrix, B::AbstractVecOrMat)
    n = size(A, 1)
    ipiv = Vector{Int}(undef, n)
    hetrf!(A, ipiv; uplo = uplo)
    hetrs!(A, ipiv, B; uplo = uplo)
    return A, ipiv, B
end
