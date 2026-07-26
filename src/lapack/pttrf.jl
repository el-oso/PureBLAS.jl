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

# |E_orig[i]|²/D[i], written as real(conj(e)·t) — two multiplies instead of a full complex product.
# The `ej*t` specialization for real T is NOT cosmetic: `real(ej)*real(t) + imag(ej)*imag(t)` leaves a
# live `fadd x, 0.0` on the real path, and LLVM may only fold that under `nsz` (x + 0.0 ≠ x for x = -0.0),
# so the extra add sits on the strictly serial divide→multiply→subtract recurrence. Measured 3.4 cyc/elem
# — 0.937 → 1.098 vs Reference-LAPACK dpttrf, n = 262144, Zen4. The 3-arg fallback keeps any other
# T <: Number (AD duals reach the ::Real method) on the generic formula.
@inline _pt_absq(ej::Real, t::Real) = ej * t
@inline _pt_absq(ej, t) = real(ej) * real(t) + imag(ej) * imag(t)

# Pivot-check block length. Derived, not tuned: the only cost the block trades against is the worst-case
# re-scan, which re-reads one block of D, so the block is sized so its D+E footprint is what L1 just held
# — the re-scan is then an L1 hit and never a second trip to memory. Measured flat over blk = 256…8192
# (1.129–1.131 vs dpttrf), i.e. the derived value (2048 for d, 1365 for z, 4096 for s on a 32 KiB L1)
# lands inside the plateau rather than on a cliff.
@inline _pt_blk(::Type{Tr}, ::Type{T}) where {Tr, T} = max(_L1_BYTES ÷ (sizeof(Tr) + sizeof(T)), 1)

# pttrf!(D, E) → (D, E, info). In-place LDLᴴ factorization. Mirrors dpttrf/zpttrf.
#
# Reference-LAPACK tests the pivot on EVERY iteration; that branch is correctly predicted but its compare
# still lands in the dependency chain of a recurrence with no independent work to hide it. Here the check
# is hoisted out of the inner loop: a block runs unconditionally while OR-ing a branchless `bad` flag, and
# only a block that actually saw a non-positive (or NaN) pivot is re-scanned to recover the exact index.
# The flag is free — it measures identical to the same loop with the check deleted entirely.
#
# Deviation on the failure path: LAPACK returns the moment it sees a bad pivot, leaving D[info+1:n] and
# E[info:n-1] at their input values, whereas this writes to the end of the failing block before returning.
# `info` itself is identical (including which index is reported when several are bad). LAPACK's contract
# for info > 0 is "the factorization could not be completed" and promises nothing about the contents, and
# ptsv! below skips the solve, so nothing downstream reads them.
function pttrf!(D::AbstractVector{Tr}, E::AbstractVector{T}) where {Tr <: Real, T <: Number}
    n = length(D)
    length(E) >= max(n - 1, 0) || throw(DimensionMismatch("pttrf!: length(E) < n-1"))
    blk = _pt_blk(Tr, T)
    i = 1
    @inbounds while i <= n - 1
        hi = min(i + blk - 1, n - 1)
        (D[i] > 0) || return D, E, i
        bad = false
        for j in i:hi
            ej = E[j]
            t = ej / D[j]                                # L[j+1,j] (complex/real ÷ real pivot)
            E[j] = t
            dn = D[j + 1] - _pt_absq(ej, t)              # stays real
            D[j + 1] = dn
            bad |= !(dn > 0)                             # `!(x > 0)` also catches NaN
        end
        if bad
            for j in i:hi
                (D[j] > 0) || return D, E, j
            end
        end
        i = hi + 1
    end
    return D, E, (n >= 1 && !(D[n] > 0)) ? n : 0
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
