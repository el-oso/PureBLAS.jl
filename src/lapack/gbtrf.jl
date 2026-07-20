# LAPACK general-banded LU with partial pivoting — faithful port of Reference-LAPACK
# dgbtf2 (unblocked banded LU) + dgbtrs (banded LU solve), generic over s/d/c/z (and any
# T<:Number, so Mode 2 / ForwardDiff-traceable). STANDALONE: depends only on Base — not yet
# wired into the module includes.
#
# ── LAPACK band storage (GB) ────────────────────────────────────────────────────────────────
# An n×n matrix with `kl` subdiagonals and `ku` superdiagonals is held in an `ldab × n` array
# AB with ldab ≥ 2·kl+ku+1. Let KV = ku + kl. The band element A(i,j) lives at
#     AB[KV+1 + i-j, j]     for  max(1, j-ku) ≤ i ≤ min(n, j+kl).
# The diagonal A(j,j) sits at row KV+1. Rows 1..kl are RESERVED workspace for the fill-in that
# partial pivoting creates (an LU factor of a (kl,ku) band has up to kl+ku superdiagonals), so
# the caller supplies data in rows kl+1..2kl+ku+1. On exit U occupies rows 1..KV+1 (KV super-
# diagonals) and the unit-L multipliers occupy rows KV+2..2kl+ku+1.
#
# ── Pivoting ────────────────────────────────────────────────────────────────────────────────
# For each column j the pivot is the row of largest |·| among the diagonal and its ≤kl sub-
# diagonal entries (LAPACK metric: |·| for real, cabs1=|Re|+|Im| for complex — matches idamax/
# izamax). ipiv[j] = global 1-based pivot row. `ju` tracks the last column any pivot has reached
# (bounds the trailing rank-1 update to the live band). Pivoting is a correctness boundary.

@inline _gb_cabs1(x::Real) = abs(x)
@inline _gb_cabs1(z::Complex) = abs(real(z)) + abs(imag(z))

# gbtrf!(kl, ku, m, AB) → (AB, ipiv, info).  AB overwritten with L\U in band storage; ipiv the
# min(m,n) pivot rows; info = index of the first exactly-zero pivot (0 if none). Mirrors dgbtf2.
function gbtrf!(kl::Integer, ku::Integer, m::Integer, AB::AbstractMatrix{T}) where {T}
    ldab, n = size(AB)
    kl >= 0 || throw(ArgumentError("gbtrf!: kl < 0"))
    ku >= 0 || throw(ArgumentError("gbtrf!: ku < 0"))
    ldab >= 2 * kl + ku + 1 || throw(DimensionMismatch("gbtrf!: ldab must be ≥ 2kl+ku+1"))
    kv = ku + kl
    mn = min(Int(m), n)
    ipiv = Vector{Int}(undef, mn)
    info = 0
    z = zero(T)
    # Zero the fill-in triangle in the leading columns (ku+2 .. min(kv,n)).
    @inbounds for j in (ku + 2):min(kv, n)
        for i in (kv - j + 2):kl
            AB[i, j] = z
        end
    end
    ju = 1
    @inbounds for j in 1:mn
        if j + kv <= n                                   # zero the fill-in of the incoming column
            for i in 1:kl
                AB[i, j + kv] = z
            end
        end
        km = min(kl, Int(m) - j)                          # # subdiagonal entries in this column
        jp = 1; pmax = _gb_cabs1(AB[kv + 1, j])           # partial-pivot argmax (diag + subdiagonals)
        for i in 2:(km + 1)
            a = _gb_cabs1(AB[kv + i, j]); a > pmax && (pmax = a; jp = i)
        end
        ipiv[j] = jp + j - 1
        if AB[kv + jp, j] != z
            ju = max(ju, min(j + ku + jp - 1, n))         # last column the pivot reaches
            if jp != 1                                    # swap A-row j ↔ A-row (jp+j-1) over cols j..ju
                for jj in j:ju
                    r1 = kv + 1 + j - jj                   # A(j, jj)
                    r2 = kv + jp + j - jj                  # A(jp+j-1, jj)
                    AB[r1, jj], AB[r2, jj] = AB[r2, jj], AB[r1, jj]
                end
            end
            if km > 0
                d = one(T) / AB[kv + 1, j]                 # multipliers L(j+i,j) = A(j+i,j)/pivot
                for i in 1:km
                    AB[kv + 1 + i, j] *= d
                end
                for jj in (j + 1):ju                       # rank-1 trailing update within the band
                    ujj = AB[kv + 1 + j - jj, jj]          # U(j,jj)
                    if ujj != z
                        for i in 1:km
                            AB[kv + 1 + i + j - jj, jj] -= AB[kv + 1 + i, j] * ujj  # A(j+i,jj) -= L(j+i,j)·U(j,jj)
                        end
                    end
                end
            end
        elseif info == 0
            info = j
        end
    end
    return AB, ipiv, info
end

# gbtrs!(trans, kl, ku, m, AB, ipiv, B) → B.  Solve op(A)·X = B in place from gbtrf!'s factors.
# trans ∈ {'N','T','C'}. Mirrors dgbtrs: for 'N', apply L⁻¹·P then a banded upper back-substitution
# (band width kl+ku); for 'T'/'C', a banded upper Uᵀ/Uᴴ forward solve then Lᵀ/Lᴴ with reverse swaps.
function gbtrs!(
        trans::AbstractChar, kl::Integer, ku::Integer, m::Integer,
        AB::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer},
        B::AbstractVecOrMat{T}
    ) where {T}
    ldab, n = size(AB)
    kv = ku + kl
    kd = kv + 1                                           # diagonal row; multipliers at rows kd+1..
    Bm = B isa AbstractVector ? reshape(B, :, 1) : B
    nrhs = size(Bm, 2)
    lnoti = kl > 0
    if trans == 'N' || trans == 'n'
        if lnoti                                          # forward: L·y = P·b
            @inbounds for j in 1:(n - 1)
                lm = min(kl, n - j)
                lp = ipiv[j]
                if lp != j
                    for c in 1:nrhs
                        Bm[lp, c], Bm[j, c] = Bm[j, c], Bm[lp, c]
                    end
                end
                for c in 1:nrhs
                    bj = Bm[j, c]
                    for i in 1:lm
                        Bm[j + i, c] -= AB[kd + i, j] * bj
                    end
                end
            end
        end
        @inbounds for c in 1:nrhs                          # backward: U·x = y (band kv superdiagonals)
            for j in n:-1:1
                xj = Bm[j, c] / AB[kv + 1, j]
                Bm[j, c] = xj
                for i in max(1, j - kv):(j - 1)
                    Bm[i, c] -= AB[kv + 1 + i - j, j] * xj
                end
            end
        end
    else
        cj = (trans == 'C' || trans == 'c')
        @inbounds for c in 1:nrhs                          # forward: Uᵀ/Uᴴ·z = b
            for j in 1:n
                s = Bm[j, c]
                for i in max(1, j - kv):(j - 1)
                    u = AB[kv + 1 + i - j, j]
                    s -= (cj ? conj(u) : u) * Bm[i, c]
                end
                ujj = AB[kv + 1, j]
                Bm[j, c] = s / (cj ? conj(ujj) : ujj)
            end
        end
        if lnoti                                          # backward: Lᵀ/Lᴴ·w = z, with reverse swaps
            @inbounds for j in (n - 1):-1:1
                lm = min(kl, n - j)
                for c in 1:nrhs
                    s = Bm[j, c]
                    for i in 1:lm
                        mij = AB[kd + i, j]
                        s -= (cj ? conj(mij) : mij) * Bm[j + i, c]
                    end
                    Bm[j, c] = s
                end
                lp = ipiv[j]
                if lp != j
                    for c in 1:nrhs
                        Bm[lp, c], Bm[j, c] = Bm[j, c], Bm[lp, c]
                    end
                end
            end
        end
    end
    return B
end
