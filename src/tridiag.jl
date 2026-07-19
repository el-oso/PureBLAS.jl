using LinearAlgebra: SingularException
# LAPACK tridiagonal solvers — dgtsv / dgttrf / dgttrs, generic over s/d/c/z (and any T<:Number,
# so Mode 2 stays AD-traceable). Faithful ports of Reference-LAPACK dgtsv.f, dgttrf.f, dgtts2.f.
#
# Storage (three vectors, LAPACK convention): `d` = main diagonal (length n), `dl` = sub-diagonal
# (length n-1, dl[i] = A[i+1,i]), `du` = super-diagonal (length n-1, du[i] = A[i,i+1]).
#
# The band structure gives O(n) work per RHS — bandwidth-trivial, so these are the plain scalar
# generic loops (no SIMD gain over a 3-term recurrence); one implementation covers all four types.

# Reshape a RHS vector to an n×1 matrix VIEW (shares memory) so the column loops are uniform. No copy.
@inline _gt_asmat(B::AbstractVector) = reshape(B, length(B), 1)
@inline _gt_asmat(B::AbstractMatrix) = B

# ── gtsv!: solve A·X = B in place (combined factorization + solve, partial pivoting) ────────────────
# Gaussian elimination with partial pivoting, storing the LU fill directly in dl/d/du (dgtsv.f). On a
# row interchange the second-superdiagonal fill lands in dl[i] (read back as the b[i+2] coefficient in
# the U back-solve). Overwrites dl, d, du (the factor) and B (← X). Throws SingularException on a zero
# pivot. Returns B.
function gtsv!(dl::AbstractVector, d::AbstractVector, du::AbstractVector, B::AbstractVecOrMat)
    n = length(d)
    (length(dl) == n - 1 && length(du) == n - 1) ||
        throw(DimensionMismatch("gtsv!: expected length(dl)=length(du)=n-1"))
    Bm = _gt_asmat(B); size(Bm, 1) == n || throw(DimensionMismatch("gtsv!: size(B,1) must equal n"))
    nrhs = size(Bm, 2)
    Z = zero(eltype(dl))
    @inbounds for i in 1:n-1
        if abs(d[i]) >= abs(dl[i])                        # no row interchange
            !iszero(d[i]) || throw(SingularException(i))
            fact = dl[i] / d[i]
            d[i+1] -= fact * du[i]
            for j in 1:nrhs
                Bm[i+1, j] -= fact * Bm[i, j]
            end
            dl[i] = Z                                     # no second-superdiagonal fill
        else                                              # interchange rows i and i+1
            fact = d[i] / dl[i]
            d[i] = dl[i]
            temp = d[i+1]
            d[i+1] = du[i] - fact * temp
            if i < n - 1
                dl[i] = du[i+1]                           # fill → dl[i] (coeff of b[i+2])
                du[i+1] = -fact * dl[i]
            end
            du[i] = temp
            for j in 1:nrhs
                t = Bm[i, j]
                Bm[i, j] = Bm[i+1, j]
                Bm[i+1, j] = t - fact * Bm[i+1, j]
            end
        end
    end
    iszero(d[n]) && throw(SingularException(n))
    @inbounds for j in 1:nrhs                             # back-solve U·x = b
        Bm[n, j] /= d[n]
        n > 1 && (Bm[n-1, j] = (Bm[n-1, j] - du[n-1] * Bm[n, j]) / d[n-1])
        for i in n-2:-1:1
            Bm[i, j] = (Bm[i, j] - du[i] * Bm[i+1, j] - dl[i] * Bm[i+2, j]) / d[i]
        end
    end
    return B
end

# ── gttrf!: LU factorization with partial pivoting (dgttrf.f) ─────────────────────────────────────
# Overwrites: d ← U diagonal, du ← U first superdiagonal, dl ← the L multipliers, du2 ← U second
# superdiagonal (fill from interchanges, length n-2), ipiv ← pivots (ipiv[i] ∈ {i, i+1}). Returns the
# 5-tuple like LinearAlgebra.LAPACK.gttrf!. Throws SingularException on a zero U pivot.
function gttrf!(dl::AbstractVector, d::AbstractVector, du::AbstractVector,
                du2::AbstractVector, ipiv::AbstractVector{<:Integer})
    n = length(d)
    (length(dl) == n - 1 && length(du) == n - 1 && length(du2) == max(n - 2, 0) && length(ipiv) == n) ||
        throw(DimensionMismatch("gttrf!: dl,du length n-1; du2 length n-2; ipiv length n"))
    Z = zero(eltype(du2))
    @inbounds for i in 1:n
        ipiv[i] = i
    end
    @inbounds for i in 1:n-2
        du2[i] = Z                                        # default: no second-superdiagonal fill
    end
    @inbounds for i in 1:n-2
        if abs(d[i]) >= abs(dl[i])                        # no interchange, eliminate dl[i]
            if !iszero(d[i])
                fact = dl[i] / d[i]; dl[i] = fact
                d[i+1] -= fact * du[i]
            end
        else                                              # interchange rows i, i+1
            fact = d[i] / dl[i]
            d[i] = dl[i]; dl[i] = fact
            temp = du[i]; du[i] = d[i+1]
            d[i+1] = temp - fact * d[i+1]
            du2[i] = du[i+1]
            du[i+1] = -fact * du[i+1]
            ipiv[i] = i + 1
        end
    end
    if n > 1
        i = n - 1
        @inbounds if abs(d[i]) >= abs(dl[i])
            if !iszero(d[i])
                fact = dl[i] / d[i]; dl[i] = fact
                d[i+1] -= fact * du[i]
            end
        else
            fact = d[i] / dl[i]
            d[i] = dl[i]; dl[i] = fact
            temp = du[i]; du[i] = d[i+1]
            d[i+1] = temp - fact * d[i+1]
            ipiv[i] = i + 1
        end
    end
    @inbounds for i in 1:n
        iszero(d[i]) && throw(SingularException(i))
    end
    return dl, d, du, du2, ipiv
end

# ── gttrs!: solve using the gttrf! factorization (dgtts2.f), trans ∈ {'N','T','C'} ─────────────────
# Overwrites B with the solution. 'N': A·X=B, 'T': Aᵀ·X=B, 'C': Aᴴ·X=B (conjugates the factor).
function gttrs!(trans::AbstractChar, dl::AbstractVector, d::AbstractVector, du::AbstractVector,
                du2::AbstractVector, ipiv::AbstractVector{<:Integer}, B::AbstractVecOrMat)
    n = length(d)
    Bm = _gt_asmat(B); size(Bm, 1) == n || throw(DimensionMismatch("gttrs!: size(B,1) must equal n"))
    nrhs = size(Bm, 2)
    n == 0 && return B
    if trans == 'N'
        @inbounds for j in 1:nrhs
            for i in 1:n-1                                 # L·x = b (forward, applying pivots)
                ip = ipiv[i]
                temp = Bm[i + 1 - ip + i, j] - dl[i] * Bm[ip, j]
                Bm[i, j] = Bm[ip, j]
                Bm[i+1, j] = temp
            end
            Bm[n, j] /= d[n]                               # U·x = b (backward)
            n > 1 && (Bm[n-1, j] = (Bm[n-1, j] - du[n-1] * Bm[n, j]) / d[n-1])
            for i in n-2:-1:1
                Bm[i, j] = (Bm[i, j] - du[i] * Bm[i+1, j] - du2[i] * Bm[i+2, j]) / d[i]
            end
        end
    else
        cj = trans == 'C' ? conj : identity               # Aᵀ (T) or Aᴴ (C)
        @inbounds for j in 1:nrhs
            Bm[1, j] /= cj(d[1])                           # Uᵀ·x = b (forward)
            n > 1 && (Bm[2, j] = (Bm[2, j] - cj(du[1]) * Bm[1, j]) / cj(d[2]))
            for i in 3:n
                Bm[i, j] = (Bm[i, j] - cj(du[i-1]) * Bm[i-1, j] - cj(du2[i-2]) * Bm[i-2, j]) / cj(d[i])
            end
            for i in n-1:-1:1                              # Lᵀ·x = b (backward, applying pivots)
                ip = ipiv[i]
                temp = Bm[i, j] - cj(dl[i]) * Bm[i+1, j]
                Bm[i, j] = Bm[ip, j]
                Bm[ip, j] = temp
            end
        end
    end
    return B
end
