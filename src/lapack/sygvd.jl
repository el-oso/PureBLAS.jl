# Generalized symmetric/Hermitian-definite eigensolver (LAPACK dsygvd / zhegvd + dsygst / zhegst).
# Solves one of three generalized problems with A symmetric/Hermitian and B symmetric/Hermitian
# POSITIVE-DEFINITE:
#   itype 1:  A·x = λ·B·x
#   itype 2:  A·B·x = λ·x
#   itype 3:  B·A·x = λ·x
#
# This COMPOSES already-validated PureBLAS kernels — it introduces no new math:
#   potrf!(B)             → B = L·Lᴴ  (uplo 'L')  or  B = Uᴴ·U  (uplo 'U')
#   trsm!/trmm! (L3)      → sygst/hegst two-sided reduction to a standard problem C·y = λ·y
#   _syev!/_heev!         → the standard symmetric/Hermitian eigensolver (eigen.jl)
#   trsm!/trmm! (L3)      → back-transform eigenvectors y → x
#
# Reduction (LAPACK dsygst, two-sided triangular solve/mult; C returned in A):
#   itype 1, uplo 'L':  C = L⁻¹·A·L⁻ᴴ        itype 1, uplo 'U':  C = U⁻ᴴ·A·U⁻¹
#   itype 2/3, uplo 'L': C = Lᴴ·A·L          itype 2/3, uplo 'U': C = U·A·Uᴴ
# We run the reduction on the FULL (symmetrized) A rather than dsygs2's one-triangle column
# recurrence — mathematically identical (A symmetric ⇒ C symmetric), two BLAS-3 calls, reuses
# the fastest trsm/trmm paths. _syev!/_heev! then reads one triangle of C as usual.
#
# Back-transform of eigenvectors (LAPACK dsygv/zhegv — NOTE the grouping):
#   itype 1 OR 2:  x = trsm  (uplo 'U' ⇒ solve U·X=Y ; uplo 'L' ⇒ solve Lᴴ·X=Y)
#   itype 3:       x = trmm  (uplo 'U' ⇒ X = Uᴴ·Y     ; uplo 'L' ⇒ X = L·Y)
# (itype 2 back-transforms with trsm like itype 1, NOT with trmm like itype 3 — see report.)
# Eigenvectors then satisfy the generalized B-orthonormality Xᴴ·B·X = I.
#
# Generic over Float64/Float32/ComplexF64/ComplexF32 (trim-safe, SIMD via the composed kernels).

# Fill the non-`uplo` triangle from the `uplo` triangle so the two-sided trsm/trmm see a full
# symmetric/Hermitian A. For complex, the mirrored entries are conjugated and the diagonal is
# realified (LAPACK ignores the imaginary part of a Hermitian diagonal).
function _gvd_symmetrize!(A::AbstractMatrix{T}, uplo::Char) where {T <: Number}
    n = size(A, 1)
    cplx = T <: Complex
    @inbounds if uplo == 'L'
        for j in 1:n
            cplx && (A[j, j] = T(real(A[j, j])))
            for i in (j + 1):n
                A[j, i] = cplx ? conj(A[i, j]) : A[i, j]
            end
        end
    else
        for j in 1:n
            cplx && (A[j, j] = T(real(A[j, j])))
            for i in (j + 1):n
                A[i, j] = cplx ? conj(A[j, i]) : A[j, i]
            end
        end
    end
    return A
end

# Shared worker for both real (sygvd!) and complex (hegvd!). Returns (w, A) for jobz='V'
# (A overwritten with the generalized eigenvectors, columns in ascending-λ order) or (w,) for 'N'.
function _gvd_impl!(
        itype::Integer, jobz::Char, uplo::Char,
        A::AbstractMatrix{T}, B::AbstractMatrix{T}
    ) where {T <: Number}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("gvd!: A must be square"))
    (size(B, 1) == size(B, 2) == n) || throw(DimensionMismatch("gvd!: B must be square, same size as A"))
    (itype == 1 || itype == 2 || itype == 3) || throw(ArgumentError("gvd!: itype must be 1, 2 or 3"))
    (jobz == 'N' || jobz == 'V') || throw(ArgumentError("gvd!: jobz must be 'N' or 'V'"))
    (uplo == 'U' || uplo == 'L') || throw(ArgumentError("gvd!: uplo must be 'U' or 'L'"))
    wantz = jobz == 'V'
    ct = T <: Complex ? 'C' : 'T'         # conjugate-transpose char (complex) vs transpose (real)

    # 1. Cholesky of B (throws PosDefException if B not positive definite).
    potrf!(B; uplo = uplo)

    if n == 0
        return wantz ? (real(T)[], A) : (real(T)[],)
    end

    # 2. Fill the opposite triangle so the reduction operates on the full symmetric/Hermitian A.
    _gvd_symmetrize!(A, uplo)

    # 3. Reduce the generalized problem to a standard one: C := reduction(A) (in place in A).
    if itype == 1
        if uplo == 'L'                                    # C = L⁻¹·A·L⁻ᴴ
            trsm!(A, B; side = 'L', uplo = 'L', transA = 'N', diag = 'N')
            trsm!(A, B; side = 'R', uplo = 'L', transA = ct, diag = 'N')
        else                                              # C = U⁻ᴴ·A·U⁻¹
            trsm!(A, B; side = 'L', uplo = 'U', transA = ct, diag = 'N')
            trsm!(A, B; side = 'R', uplo = 'U', transA = 'N', diag = 'N')
        end
    else                                                  # itype 2 or 3 share the reduction
        if uplo == 'L'                                    # C = Lᴴ·A·L
            trmm!(A, B; side = 'L', uplo = 'L', transA = ct, diag = 'N')
            trmm!(A, B; side = 'R', uplo = 'L', transA = 'N', diag = 'N')
        else                                              # C = U·A·Uᴴ
            trmm!(A, B; side = 'L', uplo = 'U', transA = 'N', diag = 'N')
            trmm!(A, B; side = 'R', uplo = 'U', transA = ct, diag = 'N')
        end
    end

    # 4. Standard symmetric/Hermitian eigensolve of C. λ are the generalized eigenvalues.
    w, Z, _ = T <: Complex ? _heev!(jobz, uplo, A) : _syev!(jobz, uplo, A)

    if !wantz
        return (w,)
    end

    # 5. Back-transform eigenvectors of C (Z) into generalized eigenvectors X (LAPACK dsygv/zhegv).
    if itype == 1 || itype == 2                           # x = inv(L)ᴴ·y  or  inv(U)·y  (trsm)
        trsm!(Z, B; side = 'L', uplo = uplo, transA = (uplo == 'U' ? 'N' : ct), diag = 'N')
    else                                                  # itype 3: x = L·y  or  Uᴴ·y  (trmm)
        trmm!(Z, B; side = 'L', uplo = uplo, transA = (uplo == 'U' ? ct : 'N'), diag = 'N')
    end
    copyto!(A, Z)                                         # A holds the generalized eigenvectors
    return w, A
end

# ── Public API ──────────────────────────────────────────────────────────────────────────────────
# sygvd!(itype, jobz, uplo, A, B) — REAL generalized symmetric-definite eigensolver.
# hegvd!(itype, jobz, uplo, A, B) — COMPLEX generalized Hermitian-definite eigensolver.
# A, B are overwritten (A → eigenvectors for jobz='V'; B → its Cholesky factor). Returns
# (w, A) for jobz='V' or (w,) for jobz='N'; w ascending, length n.
function sygvd!(
        itype::Integer, jobz::Char, uplo::Char,
        A::AbstractMatrix{T}, B::AbstractMatrix{T}
    ) where {T <: Real}
    return _gvd_impl!(itype, jobz, uplo, A, B)
end

function hegvd!(
        itype::Integer, jobz::Char, uplo::Char,
        A::AbstractMatrix{T}, B::AbstractMatrix{T}
    ) where {T <: Complex}
    return _gvd_impl!(itype, jobz, uplo, A, B)
end
