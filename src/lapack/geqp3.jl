# Column-pivoted QR (LAPACK geqp3 / geqpf):  A·P = Q·R, with P chosen so the R diagonal is
# non-increasing in magnitude (|R[1,1]| ≥ |R[2,2]| ≥ …) — rank-revealing. This is the UNBLOCKED
# core (LAPACK dgeqpf/zgeqpf; dgeqp3's blocked trailing update is a perf refinement over the SAME
# numerics), composed from PureBLAS's own Householder reflector kernels `_larfg!`/`_house_left!`
# (svd.jl, standard LAPACK τ convention) plus `_nrm2` (level1.jl, lassq-safe — req#6) for the
# partial-column-norm downdating. Generic over Float32/Float64/ComplexF32/ComplexF64.
#
# tau is standard LAPACK (H_i = I − τ_i·v_i·v_iᴴ, v_i[i]=1 implicit + essential below the diagonal
# of column i, R in the upper triangle) — the SAME convention `LinearAlgebra.LAPACK.geqp3!` returns.
# jpvt is 1-based: jpvt[k] = original index of the column now sitting in position k (so A_in[:,jpvt]
# = Q·R). Pivoting is the classic max-partial-norm rule with the dlaqps √-tolerance recompute test.

# op(H)-apply coefficient for a single reflector: for Qᴴ apply conj(τ) on complex (matches LAPACK
# zlarf w/ CONJG(TAU)); real is symmetric so τ passes through.
@inline _geqp3_tau_Qh(τ::T) where {T <: BlasReal} = τ
@inline _geqp3_tau_Qh(τ::T) where {T <: BlasComplex} = conj(τ)

function geqp3!(
        A::AbstractMatrix{T}, jpvt::AbstractVector{<:Integer},
        tau::AbstractVector{T}
    ) where {T <: BlasFloat}
    m, n = size(A); k = min(m, n); R = real(T)
    length(jpvt) >= n || throw(DimensionMismatch("geqp3!: length(jpvt) < n=$n"))
    length(tau) >= k || throw(DimensionMismatch("geqp3!: length(tau) < min(size(A))=$k"))
    @inbounds for j in 1:n
        jpvt[j] = j
    end
    n == 0 && return A, jpvt, tau
    tol3z = sqrt(eps(R))                                   # dlaqps recompute threshold (√ machine-eps)
    vn1 = Vector{R}(undef, n)                              # current (downdated) partial column 2-norms
    vn2 = Vector{R}(undef, n)                              # reference norm at last exact recompute
    @inbounds for j in 1:n
        nrm = _nrm2(m, view(A, :, j), 1); vn1[j] = nrm; vn2[j] = nrm
    end
    @inbounds for i in 1:k
        # ---- pivot: column of maximal partial norm over i:n → swap to position i ----
        pvt = i; maxn = vn1[i]
        for j in (i + 1):n
            if vn1[j] > maxn
                maxn = vn1[j]; pvt = j
            end
        end
        if pvt != i
            for r in 1:m
                t = A[r, i]; A[r, i] = A[r, pvt]; A[r, pvt] = t
            end
            jpvt[i], jpvt[pvt] = jpvt[pvt], jpvt[i]
            vn1[pvt] = vn1[i]; vn2[pvt] = vn2[i]
        end
        # ---- reflector for column i (rows i:m); β lands on the diagonal ----
        β, τ = _larfg!(view(A, i:m, i)); tau[i] = τ
        A[i, i] = β                                        # _larfg! leaves x[1]=α; place R's diagonal
        # ---- apply H_iᴴ to the trailing columns; then downdate their partial norms ----
        if i < n
            _house_left!(view(A, i:m, (i + 1):n), view(A, i:m, i), _geqp3_tau_Qh(τ))
            for j in (i + 1):n
                if !iszero(vn1[j])
                    temp = one(R) - (abs(A[i, j]) / vn1[j])^2
                    temp = max(temp, zero(R))
                    temp2 = temp * (vn1[j] / vn2[j])^2
                    if temp2 <= tol3z                      # downdated norm degraded → recompute exactly
                        if i < m
                            nrm = _nrm2(m - i, view(A, (i + 1):m, j), 1); vn1[j] = nrm; vn2[j] = nrm
                        else
                            vn1[j] = zero(R); vn2[j] = zero(R)
                        end
                    else
                        vn1[j] = vn1[j] * sqrt(temp)
                    end
                end
            end
        end
    end
    return A, jpvt, tau
end

# Convenience: allocate jpvt + tau, return (A overwritten, jpvt, tau).
function geqp3!(A::AbstractMatrix{T}) where {T <: BlasFloat}
    m, n = size(A)
    jpvt = Vector{Int}(undef, n); tau = Vector{T}(undef, min(m, n))
    geqp3!(A, jpvt, tau)
    return A, jpvt, tau
end
