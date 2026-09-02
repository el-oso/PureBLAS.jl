# LAPACK syconv (dsyconv/zsyconv, WAY='C' only — the only mode `LinearAlgebra.LAPACK.syconv!`
# drives): convert a Bunch-Kaufman factorization A = L·D·Lᵀ / U·D·Uᵀ (from PureBLAS's `sytrf!`/
# `hetrf!` in bunchkaufman.jl — LAPACK-standard ipiv encoding, identical for the symmetric and
# Hermitian cases since ipiv only encodes the pivot/2×2-block structure) into "convert" form: the
# off-diagonal entries of the 2×2 blocks of D are extracted into `work` (and zeroed in `A`), and the
# ipiv row/column interchanges are applied to the strictly triangular part of A OUTSIDE the current
# diagonal block (mirrors dsyconv.f/zsyconv.f exactly — pure data movement, no conjugation, so one
# generic port covers real and complex/Hermitian alike). Direct line-for-line port of Reference-
# LAPACK's WAY='C' branches (upper and lower).

"""
    syconv!(uplo, A, ipiv) -> (A, work)
    syconv!(uplo, A, ipiv, work) -> (A, work)

Convert the Bunch-Kaufman factors `(A, ipiv)` (as produced by `sytrf!`/`hetrf!`, `uplo` matching)
into LAPACK's "convert" form in place: `work` (length `n`) receives the off-diagonal entry of each
2×2 block of `D` (`0` where the pivot at that position is 1×1), and those entries are zeroed in `A`.
Mirrors `LinearAlgebra.LAPACK.syconv!`. Generic over `T<:Number` (s/d/c/z).

`work` is an OUTPUT, so it cannot come from the owned workspace; the 4-argument form takes it from the
caller (`length(work) >= n`, `AbstractVector{T}` so a `PtrVector` from a C-ABI shim matches) and
allocates nothing. It never READS `work` and writes every index `1:n` on every call, so an `undef`
buffer is safe and a reused one can never hand back stale data.

`work` must not alias `A`: the extraction reads `A[i-1,i]` into `work[i]` and then zeroes it, and the
row interchanges afterwards permute A wholesale — an aliased buffer would corrupt both silently. The
4-argument form throws an `ArgumentError` when `Base.mightalias` can see the overlap. (It cannot see
through a raw pointer wrapper such as `PtrVector`, which `Base.mightalias` reports as non-aliasing
because the wrapper is `isbits`; a C caller passing an overlapping `work` pointer is outside Julia's
reach either way.)
"""
function syconv!(uplo::AbstractChar, A::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer}) where {T}
    return syconv!(uplo, A, ipiv, Vector{T}(undef, size(A, 1)))
end

function syconv!(
        uplo::AbstractChar, A::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer},
        work::AbstractVector{T}
    ) where {T}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("syconv!: A must be square"))
    length(ipiv) == n || throw(DimensionMismatch("syconv!: length(ipiv) must equal size(A,1)"))
    length(work) >= n || throw(DimensionMismatch("syconv!: length(work) must be at least size(A,1)"))
    (uplo == 'U' || uplo == 'L') || throw(ArgumentError("syconv!: uplo must be 'U' or 'L'"))
    Base.mightalias(work, A) && throw(ArgumentError("syconv!: `work` must not alias `A`"))
    ZERO = zero(T)
    n == 0 && return A, work
    @inbounds if uplo == 'U'
        i = n
        work[1] = ZERO
        while i > 1
            if ipiv[i] < 0
                work[i] = A[i - 1, i]
                work[i - 1] = ZERO
                A[i - 1, i] = ZERO
                i -= 1
            else
                work[i] = ZERO
            end
            i -= 1
        end
        i = n
        while i >= 1
            if ipiv[i] > 0
                ip = ipiv[i]
                if i < n
                    for j in (i + 1):n
                        t = A[ip, j]; A[ip, j] = A[i, j]; A[i, j] = t
                    end
                end
            else
                ip = -ipiv[i]
                if i < n
                    for j in (i + 1):n
                        t = A[ip, j]; A[ip, j] = A[i - 1, j]; A[i - 1, j] = t
                    end
                end
                i -= 1
            end
            i -= 1
        end
    else
        i = 1
        work[n] = ZERO
        while i <= n
            if i < n && ipiv[i] < 0
                work[i] = A[i + 1, i]
                work[i + 1] = ZERO
                A[i + 1, i] = ZERO
                i += 1
            else
                work[i] = ZERO
            end
            i += 1
        end
        i = 1
        while i <= n
            if ipiv[i] > 0
                ip = ipiv[i]
                if i > 1
                    for j in 1:(i - 1)
                        t = A[ip, j]; A[ip, j] = A[i, j]; A[i, j] = t
                    end
                end
            else
                ip = -ipiv[i]
                if i > 1
                    for j in 1:(i - 1)
                        t = A[ip, j]; A[ip, j] = A[i + 1, j]; A[i + 1, j] = t
                    end
                end
                i += 1
            end
            i += 1
        end
    end
    return A, work
end
