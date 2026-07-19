# LAPACK equality-constrained least squares (gglse):  minimize ‖A·x − c‖₂  subject to  B·x = d,
# for A (m×n), B (p×n), c (length m), d (length p), with  p ≤ n ≤ m+p  and the standard rank
# assumptions (rank(B)=p, rank([A;B])=n).  Port of Reference-LAPACK dgglse/zgglse, composed from
# PureBLAS's own kernels — the generalized RQ factorization of (B, A) plus triangular solves.
#
# Algorithm (dgglse):  RQ-factor B = (0  R)·Q  (R = p×p upper-triangular, Q orthonormal n×n), then
# QR-factor Ã = A·Qᴴ = Z·T.  Substituting y = Q·x  (x = Qᴴ·y):
#   • constraint  B·x = R·y₂ = d          ⟹  y₂ = R⁻¹·d           (y₂ = last p entries of y)
#   • objective   ‖A·x − c‖ = ‖T·y − Zᴴc‖ ⟹  T₁₁·y₁ = (Zᴴc)₁ − T₁₂·y₂   (y₁ = first n−p entries)
#   • solution    x = Qᴴ·(y₁; y₂)
# T₁₁ = Ã[1:n−p,1:n−p] (upper-tri), T₁₂ = Ã[1:n−p, n−p+1:n].  Generic over Float32/Float64/
# ComplexF32/ComplexF64 — self-contained on svd.jl's _larfg! (dlarfg/zlarfg) reflector generator
# plus its own generic reflector-apply kernels (so it needs neither lq.jl nor gels.jl).
#
# ponytail: the orthogonal factor Q of the RQ is accumulated EXPLICITLY (n×n matrix G) rather than
# left implicit in reflectors — RQ reflector bookkeeping (reversed-column frame, orgrq) is the bug
# magnet, an explicit G is O(n³) but bulletproof and gglse is a moderate-size, not perf-gated, driver.

# Apply H = I − τ·v·vᴴ (v[1]≡1 implicit, essential in v[2:]) to C from the RIGHT: C := C·H.
# Generic over T (conj is identity on reals) — covers Float32, unlike svd.jl's Float64-only variant.
@inline function _ggl_house_right!(C::AbstractMatrix{T}, v::AbstractVector{T}, τ::T) where {T}
    iszero(τ) && return C
    nr, len = size(C)
    @inbounds for i in 1:nr
        w = C[i, 1]
        for j in 2:len; w += C[i, j] * v[j]; end          # (C·v)_i, v[1]=1
        w *= τ
        C[i, 1] -= w
        for j in 2:len; C[i, j] -= w * conj(v[j]); end     # C − τ·(Cv)·vᴴ
    end
    return C
end

# Solve R·x = b in place (R nn×nn upper-triangular, non-unit), generic scalar back-substitution.
@inline function _ggl_trsv_upper!(R::AbstractMatrix{T}, b::AbstractVector{T}, nn::Int) where {T}
    @inbounds for i in nn:-1:1
        s = b[i]
        for j in i+1:nn; s -= R[i, j] * b[j]; end
        b[i] = s / R[i, i]
    end
    return b
end

# Unblocked QR of A (m×n) — LAPACK geqr2/zgeqr2, standard τ convention. Reflectors H(i)=I−τ_i·v_i·v_iᴴ
# with essential below the diagonal of column i, τ in tau[i]; upper triangle holds T (=R). Applies
# H(i)ᴴ (conj(τ)) to the trailing columns, mirroring z/dgeqr2. Generic over T.
function _ggl_geqr2!(A::AbstractMatrix{T}, tau::AbstractVector{T}) where {T}
    m, n = size(A); k = min(m, n)
    @inbounds for i in 1:k
        xq = view(A, i:m, i)                              # column i: xq[1]=α, tail below
        β, τ = _larfg!(xq); tau[i] = τ                    # essential now in xq[2:]
        if i < n
            xq[1] = one(T)                                # v[1] ≡ 1 implicit
            _house_left!(view(A, i:m, i+1:n), xq, T <: Complex ? conj(τ) : τ)  # apply H(i)ᴴ
        end
        A[i, i] = T(β)                                    # store R's diagonal
    end
    return A, tau
end

# ĉ := Zᴴ·c  where Z = Q of the QR in `A` (tau).  Zᴴ = H(k)ᴴ···H(1)ᴴ ⟹ apply H(i)ᴴ forward i=1:k.
function _ggl_apply_Zh!(A::AbstractMatrix{T}, tau::AbstractVector{T}, c::AbstractVector{T}, k::Int) where {T}
    m = size(A, 1)
    cm = reshape(c, m, 1)
    @inbounds for i in 1:k
        v = view(A, i:m, i); ai = v[1]; A[i, i] = one(T)  # v[1] ≡ 1
        _house_left!(view(cm, i:m, :), v, T <: Complex ? conj(tau[i]) : tau[i])
        A[i, i] = ai
    end
    return c
end

# Unblocked RQ factorization of B (p×n, p ≤ n), accumulating the orthonormal factor into G (n×n).
# On return B[1:p, n−p+1:n] holds R (p×p upper-triangular) and G = W with  B_orig = R·Gᴴ  (Q = Gᴴ).
# Reflector i (i = p…1) annihilates B[i, 1:n−p+i−1] with pivot at column n−p+i, applied from the right;
# expressed in a reversed-column frame so svd.jl's front-unit _larfg! applies directly. Both B's leading
# rows and the accumulator G undergo the SAME right-multiply by H(i) = I − τ·v·vᴴ (zgerq2's ZLACGV dance
# on the row makes e/R come out right for complex).
function _ggl_gerq2_accumG!(B::AbstractMatrix{T}, G::AbstractMatrix{T}) where {T}
    p, n = size(B)
    p <= n || throw(DimensionMismatch("_ggl_gerq2_accumG!: need rows ≤ cols"))
    @inbounds for j in 1:n, i in 1:n; G[i, j] = (i == j) ? one(T) : zero(T); end   # G := I
    @inbounds for i in p:-1:1
        jj = n - p + i                                   # pivot column (reflector spans cols 1:jj)
        rowrev = view(B, i, jj:-1:1)                     # reversed row: rowrev[1] = B[i,jj] = pivot
        for t in eachindex(rowrev); rowrev[t] = conj(rowrev[t]); end   # ZLACGV (no-op on reals)
        β, τ = _larfg!(rowrev)                           # essential in rowrev[2:], β returned
        rowrev[1] = one(T)                               # v[1] ≡ 1 implicit
        i > 1 && _ggl_house_right!(view(B, 1:i-1, jj:-1:1), rowrev, τ)  # H(i) → B's leading rows
        _ggl_house_right!(view(G, 1:n, jj:-1:1), rowrev, τ)            # H(i) → G  (G := G·H(i))
        rowrev[1] = T(β)                                 # restore R's diagonal (β real)
        for t in eachindex(rowrev); rowrev[t] = conj(rowrev[t]); end   # ZLACGV back
    end
    return B, G
end

"""
    gglse!(A, c, B, d) -> (x, res)

Solve the equality-constrained least-squares problem  minimize ‖A·x − c‖₂  subject to  B·x = d.
`A` is m×n, `B` is p×n, `c` length m, `d` length p, requiring `p ≤ n ≤ m+p`. Returns the solution
`x` (length n) and the residual norm `res = ‖A·x − c‖₂`. `A`, `B`, `c`, `d` are overwritten (LAPACK
dgglse convention). Matches `LinearAlgebra.LAPACK.gglse!`.
"""
function gglse!(A::AbstractMatrix{T}, c::AbstractVector{T}, B::AbstractMatrix{T},
        d::AbstractVector{T}) where {T<:BlasFloat}
    m, n = size(A); p = size(B, 1)
    size(B, 2) == n || throw(DimensionMismatch("gglse!: A and B must have the same number of columns"))
    length(c) == m || throw(DimensionMismatch("gglse!: length(c) ≠ rows(A)"))
    length(d) == p || throw(DimensionMismatch("gglse!: length(d) ≠ rows(B)"))
    (p <= n && n <= m + p) || throw(DimensionMismatch("gglse!: need p ≤ n ≤ m+p (got m=$m,n=$n,p=$p)"))
    np = n - p                                           # y₁ length

    # 1) RQ of B, explicit orthonormal factor G (Q = Gᴴ):  B_orig = R·Gᴴ,  R = B[1:p, np+1:n].
    G = Matrix{T}(undef, n, n)
    _ggl_gerq2_accumG!(B, G)

    # 2) Ã = A·Qᴴ = A·G  (m×n), then QR:  Ã = Z·T  (Z reflectors + T upper-trapezoidal in Ã).
    Atil = Matrix{T}(undef, m, n)
    gemm!(Atil, A, G)                                    # Atil = A·G
    tauz = Vector{T}(undef, min(m, n))
    _ggl_geqr2!(Atil, tauz)

    # 3) ĉ = Zᴴ·c.
    ĉ = Vector{T}(undef, m)
    @inbounds for i in 1:m; ĉ[i] = c[i]; end
    _ggl_apply_Zh!(Atil, tauz, ĉ, min(m, n))

    # 4) y₂ = R⁻¹·d   (R = B[1:p, np+1:n], p×p upper-tri).  y[np+1:n] = y₂.
    y = Vector{T}(undef, n)
    @inbounds for i in 1:p; y[np + i] = d[i]; end
    _ggl_trsv_upper!(view(B, 1:p, np+1:n), view(y, np+1:n), p)

    # 5) ĉ₁ := ĉ₁ − T₁₂·y₂   (T₁₂ = Ã[1:np, np+1:n]).
    @inbounds for i in 1:np
        s = ĉ[i]
        for j in 1:p; s -= Atil[i, np + j] * y[np + j]; end
        ĉ[i] = s
    end

    # 6) y₁ = T₁₁⁻¹·ĉ₁   (T₁₁ = Ã[1:np,1:np] upper-tri).  y[1:np] = y₁.
    if np > 0
        @inbounds for i in 1:np; y[i] = ĉ[i]; end
        _ggl_trsv_upper!(view(Atil, 1:np, 1:np), view(y, 1:np), np)
    end

    # 7) x = Qᴴ·y = G·y.
    x = Vector{T}(undef, n)
    gemv!(x, G, y)

    # residual ‖A·x − c‖₂, computed directly (A and c are untouched by the steps above).
    r = Vector{T}(undef, m)
    gemv!(r, A, x)                                        # r = A·x
    res = zero(real(T))
    @inbounds for i in 1:m; res += abs2(r[i] - c[i]); end
    return x, sqrt(res)
end
