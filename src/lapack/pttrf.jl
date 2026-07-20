# LAPACK SPD tridiagonal LDLᵀ / LDLᴴ factorization and solve — faithful port of Reference-LAPACK
# dpttrf/dpttrs (real) and zpttrf/zpttrs (Hermitian), plus the ptsv driver. Generic over s/d/c/z
# (and any T<:Number). STANDALONE: depends only on Base — not yet wired into the module includes.
#
# The tridiagonal matrix is symmetric-positive-definite (real) or Hermitian-positive-definite
# (complex), stored by two vectors: D (n real diagonal entries) and E (n-1 subdiagonal entries,
# E[i] = A[i+1,i]; the superdiagonal is E[i] real / conj(E[i]) Hermitian). NOTE for complex: D is
# REAL, E is COMPLEX — the diagonal of a Hermitian matrix is real by definition.
#
# Factorization A = L·D·Lᴴ, with L unit lower-bidiagonal. dpttrf/zpttrf overwrite:
#   D ← the diagonal of the (real, positive) middle factor D,
#   E ← the subdiagonal multipliers L[i+1,i] = A[i+1,i]/D[i].
# Recurrence:  D[i+1] -= |E_orig[i]|² / D[i];   E[i] ← E_orig[i] / D[i].
# info = index of the first non-positive pivot (0 if the matrix is SPD), = LAPACK's info>0.

# pttrf!(D, E) → (D, E, info). In-place LDLᴴ factorization. Mirrors dpttrf/zpttrf.
function pttrf!(D::AbstractVector{Tr}, E::AbstractVector{T}) where {Tr <: Real, T <: Number}
    n = length(D)
    length(E) >= max(n - 1, 0) || throw(DimensionMismatch("pttrf!: length(E) < n-1"))
    info = 0
    @inbounds for i in 1:(n - 1)
        if !(D[i] > 0)
            return D, E, i
        end
        ei = E[i]
        f = real(ei) / D[i]                              # real & imag parts of the multiplier
        g = imag(ei) / D[i]
        E[i] = ei / D[i]                                 # L[i+1,i] (complex/real ÷ real pivot)
        D[i + 1] = D[i + 1] - (f * real(ei) + g * imag(ei))   # -= |E_orig[i]|² / D[i]  (stays real)
    end
    if n >= 1 && !(D[n] > 0)
        info = n
    end
    return D, E, info
end

# Shared LDLᴴ solve. upper=false ⇒ E is the SUBdiagonal (uplo='L'): forward with L (no conj),
# backward with Lᴴ (conj). upper=true ⇒ E is the SUPERdiagonal (uplo='U'): conj placement flips.
# For real T, conj is the identity so both branches coincide (matches dpttrs).
function _pttrs_core!(
        D::AbstractVector{<:Real}, E::AbstractVector{T},
        Bm::AbstractMatrix{T}, upper::Bool
    ) where {T <: Number}
    n = length(D)
    nrhs = size(Bm, 2)
    @inbounds for c in 1:nrhs
        for i in 2:n                                     # forward: solve (unit-bidiagonal) L·z = b
            e = upper ? conj(E[i - 1]) : E[i - 1]
            Bm[i, c] -= e * Bm[i - 1, c]
        end
        Bm[n, c] = Bm[n, c] / D[n]                       # backward: D⁻¹ then Lᴴ
        for i in (n - 1):-1:1
            e = upper ? E[i] : conj(E[i])
            Bm[i, c] = Bm[i, c] / D[i] - e * Bm[i + 1, c]
        end
    end
    return Bm
end

# pttrs!(D, E, B; uplo) → B.  Solve A·X = B in place from pttrf!'s factors. Mirrors dpttrs/zpttrs.
# uplo selects whether E is stored as sub- ('L', default) or super- ('U') diagonal (complex only;
# real is symmetric so uplo is a no-op).
function pttrs!(
        D::AbstractVector{<:Real}, E::AbstractVector{T},
        B::AbstractVecOrMat{T}; uplo::AbstractChar = 'L'
    ) where {T <: Number}
    length(D) == size(B isa AbstractVector ? reshape(B, :, 1) : B, 1) ||
        throw(DimensionMismatch("pttrs!: size(B,1) ≠ length(D)"))
    Bm = B isa AbstractVector ? reshape(B, :, 1) : B
    _pttrs_core!(D, E, Bm, uplo == 'U' || uplo == 'u')
    return B
end

# ptsv!(D, E, B) → (D, E, B, info).  Full driver: factor then solve (mirrors dptsv/zptsv).
function ptsv!(
        D::AbstractVector{<:Real}, E::AbstractVector{T},
        B::AbstractVecOrMat{T}; uplo::AbstractChar = 'L'
    ) where {T <: Number}
    _, _, info = pttrf!(D, E)
    info == 0 && pttrs!(D, E, B; uplo = uplo)
    return D, E, B, info
end
