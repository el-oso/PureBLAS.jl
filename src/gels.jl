# Least-squares / minimum-norm solve (LAPACK gels):  solve  min‖op(A)·x − b‖₂  for the
# overdetermined case (rows ≥ cols, via QR:  x = R⁻¹·Qᴴ·b) and the minimum-norm solution of the
# underdetermined case (rows < cols, via LQ = QR-of-adjoint:  x = Q·[L⁻¹·b; 0]). Composed from
# PureBLAS's existing QR (`geqrf!`, qr.jl — the tuned blocked path for Float64/complex; a tiny
# unblocked `_geqr2!` fills the Float32-real gap qr.jl doesn't cover), the reflector kernel
# `_house_left!` (svd.jl) for the Q/Qᴴ back-transforms, and `trsm!` (level3.jl) for the triangular
# solve. Generic over Float32/Float64/ComplexF32/ComplexF64.  op(A): trans='N' → A, 'T' → Aᵀ,
# 'C' → Aᴴ (trans='T' is rejected for complex, matching LAPACK).
#
# B is sized max(rows,cols of op(A)) × nrhs (LAPACK ldb convention): on input its first `rows` rows
# hold b; on output its first `cols` rows hold the solution x.

# ---- factorization: route through the tuned geqrf! where it exists, else the unblocked core ----
@inline _fast_geqrf(::Type{Float64})    = true
@inline _fast_geqrf(::Type{ComplexF64}) = true
@inline _fast_geqrf(::Type{ComplexF32}) = true
@inline _fast_geqrf(::Type{T}) where {T} = false

# Unblocked Householder QR (LAPACK geqr2), standard τ convention — composes _larfg!/_house_left!.
# Only needed for Float32-real (qr.jl's real geqrf! is Float64-only); kept generic for safety.
function _geqr2!(A::AbstractMatrix{T}, tau::AbstractVector{T}) where {T}
    m, n = size(A); k = min(m, n)
    @inbounds for i in 1:k
        β, τ = _larfg!(view(A, i:m, i)); tau[i] = τ; A[i, i] = β
        if i < n
            τa = T <: Complex ? conj(τ) : τ
            _house_left!(view(A, i:m, i+1:n), view(A, i:m, i), τa)
        end
    end
    return A, tau
end

# Factor M = Q·R in place; leave `tau` in STANDARD LAPACK convention regardless of which path ran
# (geqrf!'s real path stores the inverted "faer" τ = 1/τ_LAPACK, Inf for a trivial reflector — bridge
# it back here so every downstream apply speaks one convention; complex geqrf! already is LAPACK).
function _gels_factor!(M::AbstractMatrix{T}, tau::AbstractVector{T}) where {T}
    m, n = size(M); k = min(m, n)
    if _fast_geqrf(T)
        geqrf!(M, view(tau, 1:k))
        if T <: BlasReal
            @inbounds for i in 1:k
                t = tau[i]; tau[i] = isfinite(t) ? one(T) / t : zero(T)
            end
        end
    else
        _geqr2!(M, view(tau, 1:k))
    end
    return tau
end

# Reflector-apply coefficients (tau in LAPACK convention). Qᴴ applies conj(τ) on complex, Q applies τ.
@inline _tau_Qh(τ::T) where {T<:BlasReal}    = τ
@inline _tau_Qh(τ::T) where {T<:BlasComplex} = conj(τ)
@inline _tau_Q(τ::T)  where {T<:BlasReal}    = τ
@inline _tau_Q(τ::T)  where {T<:BlasComplex} = τ

# B := Qᴴ·B   (Q = H_1⋯H_k stored in the factored M; forward order).  v[1]≡1 implicit.
function _apply_Qh!(M::AbstractMatrix{T}, tau::AbstractVector{T}, B::AbstractMatrix{T}, k::Int) where {T}
    m = size(M, 1)
    @inbounds for i in 1:k
        _house_left!(view(B, i:m, :), view(M, i:m, i), _tau_Qh(tau[i]))
    end
    return B
end
# B := Q·B   (reverse order).
function _apply_Q!(M::AbstractMatrix{T}, tau::AbstractVector{T}, B::AbstractMatrix{T}, k::Int) where {T}
    m = size(M, 1)
    @inbounds for i in k:-1:1
        _house_left!(view(B, i:m, :), view(M, i:m, i), _tau_Q(tau[i]))
    end
    return B
end

function gels!(trans::Char, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<:BlasFloat}
    trans === 'N' || trans === 'T' || trans === 'C' ||
        throw(ArgumentError("gels!: trans must be 'N', 'T' or 'C', got $(repr(trans))"))
    (T <: Complex && trans === 'T') &&
        throw(ArgumentError("gels!: trans='T' invalid for complex element type — use 'C'"))
    # M = op(A), p×q. (A itself is left untouched; LAPACK overwrites A, we don't need to.)
    M = trans === 'N' ? Matrix{T}(A) : (trans === 'T' ? Matrix{T}(transpose(A)) : Matrix{T}(adjoint(A)))
    p, q = size(M); k = min(p, q)
    size(B, 1) >= max(p, q) ||
        throw(DimensionMismatch("gels!: size(B,1)=$(size(B,1)) must be ≥ max(rows,cols of op(A))=$(max(p,q))"))
    nrhs = size(B, 2)
    tau = Vector{T}(undef, k)
    if p >= q
        # overdetermined:  x = R⁻¹·(Qᴴ·b)[1:q]
        _gels_factor!(M, tau)
        _apply_Qh!(M, tau, view(B, 1:p, :), k)
        trsm!(view(B, 1:q, :), view(M, 1:q, 1:q); side = 'L', uplo = 'U', transA = 'N', diag = 'N')
    else
        # underdetermined min-norm:  Mᴴ = Q_r·R_r (tall QR),  M = R_rᴴ·Q_rᴴ,  x = Q_r·[R_r⁻ᴴ·b; 0]
        Mh = Matrix{T}(adjoint(M))                         # q×p, tall
        _gels_factor!(Mh, tau)                             # R_r = Mh[1:p,1:p] upper
        tc = T <: Complex ? 'C' : 'T'
        trsm!(view(B, 1:p, :), view(Mh, 1:p, 1:p); side = 'L', uplo = 'U', transA = tc, diag = 'N')  # w = R_r⁻ᴴ·b
        @inbounds for j in 1:nrhs, r in p+1:q; B[r, j] = zero(T); end
        _apply_Q!(Mh, tau, view(B, 1:q, :), k)             # x = Q_r·[w; 0]
    end
    return A, B
end
