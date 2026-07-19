# LAPACK pstrf — Cholesky with COMPLETE (diagonal) pivoting for a positive-SEMIdefinite matrix.
# Pure Julia, generic over T (Float32/Float64/ComplexF32/ComplexF64). At each step the largest
# remaining Schur-complement diagonal is pivoted to the front; the factorization stops at the numerical
# rank (first pivot ≤ tol). Produces Pᵀ·A·P = Lᴴ... with a permutation `piv` and detected `rank`, so it
# backs `cholesky(A, RowMaximum())`.
#
# Ports LAPACK's UNBLOCKED level-2 kernel dpstf2/zpstf2 (the blocked dpstrf gives identical results —
# ponytail: blocked deferred, add when perf-gated). ONE generic path covers real (SPD) and complex
# (Hermitian PSD) by threading `conj`/`real`/`abs2` exactly where zpstf2 differs from dpstf2 (the
# Hermitian symmetric-swap conjugates the off-diagonal triangle; conj is identity on reals).

# Hermitian/symmetric swap of index j ↔ pvt for the LOWER-stored triangle (dpstf2/zpstf2 lower).
@inline function _pstrf_swap_lower!(A::AbstractMatrix{T}, j::Int, pvt::Int, n::Int) where {T}
    @inbounds begin
        A[pvt, pvt] = A[j, j]
        for l in 1:j-1; A[j, l], A[pvt, l] = A[pvt, l], A[j, l]; end     # leading row parts (cols 1:j-1)
        for l in pvt+1:n; A[l, j], A[l, pvt] = A[l, pvt], A[l, j]; end   # col parts below pvt
        for i in j+1:pvt-1                                               # triangle between (conjugated)
            tmp = conj(A[i, j]); A[i, j] = conj(A[pvt, i]); A[pvt, i] = tmp
        end
        A[pvt, j] = conj(A[pvt, j])                                      # (no-op real)
    end
    return A
end
# Hermitian/symmetric swap for the UPPER-stored triangle (dpstf2/zpstf2 upper).
@inline function _pstrf_swap_upper!(A::AbstractMatrix{T}, j::Int, pvt::Int, n::Int) where {T}
    @inbounds begin
        A[pvt, pvt] = A[j, j]
        for l in 1:j-1; A[l, j], A[l, pvt] = A[l, pvt], A[l, j]; end     # leading col parts (rows 1:j-1)
        for c in pvt+1:n; A[j, c], A[pvt, c] = A[pvt, c], A[j, c]; end   # row parts right of pvt
        for i in j+1:pvt-1                                               # triangle between (conjugated)
            tmp = conj(A[j, i]); A[j, i] = conj(A[i, pvt]); A[i, pvt] = tmp
        end
        A[j, pvt] = conj(A[j, pvt])                                      # (no-op real)
    end
    return A
end

"""
    pstrf!(A, piv, tol; uplo='L') -> (A, piv, rank, info)

Cholesky with complete (diagonal) pivoting of a Hermitian positive-semidefinite `A` (LAPACK
{d,z}pstf2). Overwrites the `uplo` triangle of A with the (rank-truncated) Cholesky factor; `piv`
(preallocated length ≥ n) receives the pivot permutation, `rank` the numerical rank, `info` = 0 if
full-rank else 1. `tol < 0` ⇒ default LAPACK stop = n·eps·max_diagonal.

`Pᵀ·A·P` reconstructs from the factor: LOWER ⇒ L(:,1:rank)·L(:,1:rank)ᴴ, UPPER ⇒ U(1:rank,:)ᴴ·U(1:rank,:),
with P defined by `piv`.
"""
function pstrf!(A::AbstractMatrix{T}, piv::AbstractVector{<:Integer}, tol::Real; uplo::Char='L') where {T}
    R = real(T); n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("pstrf!: A must be square"))
    length(piv) >= n || throw(DimensionMismatch("pstrf!: length(piv) < n"))
    lower = uplo == 'L'
    lower || uplo == 'U' || throw(ArgumentError("pstrf!: uplo must be 'L' or 'U'"))
    n == 0 && return A, piv, 0, 0
    @inbounds for i in 1:n; piv[i] = i; end
    work = zeros(R, 2n)                                          # work[1:n] running dot products; [n+1:2n] scratch
    # Initial pivot = largest diagonal.
    pvt = 1; ajj = real(@inbounds A[1, 1])
    @inbounds for i in 2:n
        d = real(A[i, i]); (d > ajj) && (ajj = d; pvt = i)
    end
    if ajj <= zero(R) || isnan(ajj)
        @inbounds A[1, 1] = T(ajj)
        return A, piv, 0, 1
    end
    dstop = tol < 0 ? n * eps(R) * ajj : R(tol)
    rank = n
    @inbounds if lower
        for j in 1:n
            for i in j:n                                        # update Schur-complement diagonals
                (j > 1) && (work[i] += abs2(A[i, j-1]))
                work[n+i] = real(A[i, i]) - work[i]
            end
            if j > 1
                pvt = j; mx = work[n+j]
                for i in j+1:n; (work[n+i] > mx) && (mx = work[n+i]; pvt = i); end
                ajj = work[n+pvt]
                if ajj <= dstop || isnan(ajj)
                    A[j, j] = T(ajj); rank = j - 1; break
                end
            end
            if j != pvt
                _pstrf_swap_lower!(A, j, pvt, n)
                work[pvt] = work[j]
                piv[j], piv[pvt] = piv[pvt], piv[j]
            end
            ajj = sqrt(ajj); A[j, j] = T(ajj)
            if j < n
                for l in 1:j-1                                  # A[j+1:n,j] −= A[j+1:n,1:j-1]·conj(A[j,1:j-1])
                    ajl = conj(A[j, l])
                    for i in j+1:n; A[i, j] -= A[i, l] * ajl; end
                end
                invajj = one(R) / ajj
                for i in j+1:n; A[i, j] *= invajj; end
            end
        end
    else
        for j in 1:n
            for i in j:n
                (j > 1) && (work[i] += abs2(A[j-1, i]))
                work[n+i] = real(A[i, i]) - work[i]
            end
            if j > 1
                pvt = j; mx = work[n+j]
                for i in j+1:n; (work[n+i] > mx) && (mx = work[n+i]; pvt = i); end
                ajj = work[n+pvt]
                if ajj <= dstop || isnan(ajj)
                    A[j, j] = T(ajj); rank = j - 1; break
                end
            end
            if j != pvt
                _pstrf_swap_upper!(A, j, pvt, n)
                work[pvt] = work[j]
                piv[j], piv[pvt] = piv[pvt], piv[j]
            end
            ajj = sqrt(ajj); A[j, j] = T(ajj)
            if j < n
                for c in j+1:n                                  # A[j,j+1:n] −= conj(A[1:j-1,j])·A[1:j-1,j+1:n]
                    s = zero(T)
                    for l in 1:j-1; s += conj(A[l, j]) * A[l, c]; end
                    A[j, c] -= s
                end
                invajj = one(R) / ajj
                for c in j+1:n; A[j, c] *= invajj; end
            end
        end
    end
    return A, piv, rank, (rank == n ? 0 : 1)
end

# Convenience: allocate piv (LinearAlgebra.LAPACK.pstrf!-style return).
function pstrf!(A::AbstractMatrix, tol::Real; uplo::Char='L')
    piv = Vector{Int}(undef, size(A, 1))
    return pstrf!(A, piv, tol; uplo=uplo)
end
