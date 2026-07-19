# LAPACK LQ (gelqf/orglq/unglq/ormlq/unmlq) — pure Julia, generic over T (Float32/Float64/
# ComplexF32/ComplexF64). LQ is the row-wise dual of QR: A (m×n) = L·Q with L lower-trapezoidal
# (m×min) and Q (min×n) orthonormal ROWS. Reflectors are applied from the RIGHT to rows; reflector
# i zeros A[i,i+1:n], its essential v sits in A[i,i+1:n] and τ in tau[i].
#
# Householder = STANDARD LAPACK convention (H = I − τ·v·vᴴ, τ multiplies directly, v[i]=1 implicit),
# reusing svd.jl's `_larfg!` (which carries the underflow/overflow-safe scaled recompute — the known
# reflector-underflow bug class). So `tau` here matches LinearAlgebra.LAPACK.gelqf!/unglq! DIRECTLY —
# NO faer 1/τ inversion (unlike qr.jl's stored tau). One generic path covers real and complex by
# threading `conj` (identity on reals) exactly where LAPACK's z*-routines call ZLACGV — dgelq2≡zgelq2,
# dorgl2≡zungl2, dorml2≡zunml2 collapse to a single implementation each.
#
# Ports LAPACK's UNBLOCKED kernels (dgelq2/dorgl2/dorml2 + complex duals). Correct for all shapes;
# ponytail: blocked (compact-WY, gemm trailing update) deferred — add when this path is perf-gated.

# Apply H = I − τ·v·vᴴ (v[1]≡1 implicit, essential in v[2:]) to C from the RIGHT: C := C·H.
# Generic over T (conj is identity on reals) — covers the Float32 gap svd.jl's _house_right! leaves.
@inline function _lq_apply_right!(C::AbstractMatrix{T}, v::AbstractVector{T}, τ::T) where {T}
    iszero(τ) && return C
    nr, len = size(C)
    @inbounds for i in 1:nr
        w = C[i, 1]
        for j in 2:len; w += C[i, j] * v[j]; end            # (C·v)_i, v[1]=1
        w *= τ
        C[i, 1] -= w
        for j in 2:len; C[i, j] -= w * conj(v[j]); end       # C − τ(Cv)vᴴ
    end
    return C
end

# Apply H = I − τ·v·vᴴ (v[1]≡1 implicit) to C from the LEFT: C := H·C = C − τ·v·(vᴴC).
@inline function _lq_apply_left!(C::AbstractMatrix{T}, v::AbstractVector{T}, τ::T) where {T}
    iszero(τ) && return C
    len, nc = size(C)
    @inbounds for j in 1:nc
        w = C[1, j]
        for i in 2:len; w += conj(v[i]) * C[i, j]; end       # (vᴴ·C)_j, v[1]=1
        w *= τ
        C[1, j] -= w
        for i in 2:len; C[i, j] -= v[i] * w; end
    end
    return C
end

"""
    gelqf!(A, tau) -> A

LQ factorization (LAPACK dgelqf/zgelqf, unblocked dgelq2/zgelq2). Overwrites A (m×n): the lower
trapezoid holds L (m×min), the strict upper part of each row holds the essential reflector v;
`tau[i]` (standard LAPACK convention) the coefficients. `A = L·Q`, Q = H(k)…H(1) (real) /
H(k)ᴴ…H(1)ᴴ (complex), k = min(m,n).
"""
function gelqf!(A::AbstractMatrix{T}, tau::AbstractVector{T}) where {T}
    m, n = size(A); k = min(m, n)
    length(tau) >= k || throw(DimensionMismatch("gelqf!: length(tau) < min(size(A))"))
    @inbounds for i in 1:k
        row = view(A, i, i:n)                                # A[i, i:n], length n−i+1
        for t in eachindex(row); row[t] = conj(row[t]); end  # ZLACGV (no-op on reals)
        β, τ = _larfg!(row)                                  # β real (complex)/scalar (real); v in row[2:]
        tau[i] = τ
        if i < m
            row[1] = one(T)                                  # A[i,i] := 1 (v[1] implicit)
            _lq_apply_right!(view(A, i+1:m, i:n), row, τ)    # H(i) to trailing rows, from the right
        end
        row[1] = T(β)                                        # A[i,i] := β  (β is real)
        for t in eachindex(row); row[t] = conj(row[t]); end  # ZLACGV back
    end
    return A
end

# Convenience: allocate tau, return (A overwritten, tau).
function gelqf!(A::AbstractMatrix{T}) where {T}
    tau = Vector{T}(undef, min(size(A)...))
    gelqf!(A, tau)
    return A, tau
end

"""
    orglq!(A, tau) -> A    (real)
    unglq!(A, tau) -> A    (complex)

Form Q (LAPACK dorglq/zunglq, unblocked dorgl2/zungl2). On entry A (mq×n, mq ≤ n) holds the gelqf
reflectors in its rows; on exit A holds the mq×n matrix Q with orthonormal rows. `k = length(tau)`
reflectors are used (mq ≥ k).
"""
function orglq!(A::AbstractMatrix{T}, tau::AbstractVector{T}) where {T}
    mq, n = size(A); k = length(tau)
    mq <= n || throw(DimensionMismatch("orglq!: rows(A) > cols(A)"))
    k <= mq || throw(DimensionMismatch("orglq!: length(tau) > rows(A)"))
    @inbounds begin
        if k < mq                                            # init rows k+1:mq to unit rows of I
            for j in 1:n, l in k+1:mq
                A[l, j] = (l == j) ? one(T) : zero(T)
            end
        end
        for i in k:-1:1
            τc = conj(tau[i])                                 # forming Q applies H(i)ᴴ = I − conj(τ)·v·vᴴ
            if i < n                                          # (conj is identity on reals → dorgl2)
                for t in i+1:n; A[i, t] = conj(A[i, t]); end  # ZLACGV
                if i < mq
                    A[i, i] = one(T)
                    _lq_apply_right!(view(A, i+1:mq, i:n), view(A, i, i:n), τc)
                end
                for t in i+1:n; A[i, t] = -tau[i] * A[i, t]; end   # ZSCAL(−τ)
                for t in i+1:n; A[i, t] = conj(A[i, t]); end       # ZLACGV back
            end
            A[i, i] = one(T) - τc
            for l in 1:i-1; A[i, l] = zero(T); end
        end
    end
    return A
end
const unglq! = orglq!

"""
    ormlq!(side, trans, A, tau, C) -> C    (real: trans ∈ {'N','T'})
    unmlq!(side, trans, A, tau, C) -> C    (complex: trans ∈ {'N','C'})

Apply Q from the reflectors in `A` (gelqf output, k×nq with reflectors in rows) to `C` (m×n), from
the LEFT (`side='L'`, C := op(Q)·C, Q order nq=m) or RIGHT (`side='R'`, C := C·op(Q), Q order nq=n).
LAPACK dormlq/zunmlq (unblocked dorml2/zunml2). op = Q (`trans='N'`) or Qᴴ (`trans='T'`/`'C'`).
"""
function ormlq!(side::Char, trans::Char, A::AbstractMatrix{T}, tau::AbstractVector{T},
        C::AbstractMatrix{T}) where {T}
    m, n = size(C); k = length(tau)
    left = side == 'L'
    notran = trans == 'N'
    nq = left ? m : n
    # Forward order when applying Q from left or Qᴴ from right; reverse otherwise (dorml2/zunml2).
    if (left && notran) || (!left && !notran)
        i1, i2, i3 = 1, k, 1
    else
        i1, i2, i3 = k, 1, -1
    end
    @inbounds begin
        i = i1
        while i3 > 0 ? i <= i2 : i >= i2
            seg = view(A, i, i:nq)                           # reflector i essential (v[1] at seg[1])
            for t in eachindex(seg); seg[t] = conj(seg[t]); end   # ZLACGV (no-op real)
            aii = A[i, i]; A[i, i] = one(T)
            taui = notran ? conj(tau[i]) : tau[i]            # conj identity on reals
            if left
                _lq_apply_left!(view(C, i:m, 1:n), seg, taui)
            else
                _lq_apply_right!(view(C, 1:m, i:n), seg, taui)
            end
            A[i, i] = aii
            for t in eachindex(seg); seg[t] = conj(seg[t]); end   # ZLACGV back
            i += i3
        end
    end
    return C
end
const unmlq! = ormlq!
