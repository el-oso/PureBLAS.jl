# LAPACK QL (geqlf/orgql/ormql) and RQ (gerqf/orgrq/ormrq) — pure Julia, generic over T
# (Float32/Float64/ComplexF32/ComplexF64). Both are the "opposite-corner" cousins of QR/LQ:
#
#   • QL:  A (m×n) = Q·L,  L lower-triangular in the BOTTOM-RIGHT n×n corner (m≥n) — Householder
#     reflectors are COLUMN reflectors anchored at the BOTTOM (v[last]=1, essential above), generated
#     bottom-up (i = k…1). Ports LAPACK dgeql2/dorg2l/dorm2l + complex duals zgeql2/zung2l/zunm2l.
#   • RQ:  A (m×n) = R·Q,  R upper-triangular in the TOP-RIGHT m×m corner — reflectors are ROW
#     reflectors anchored at the RIGHT (v[last]=1, essential to the left). RQ is the corner-dual of LQ
#     exactly as QL is of QR. Ports LAPACK dgerq2/dorgr2/dormr2 + complex duals zgerq2/zungr2/zunmr2.
#
# Householder = STANDARD LAPACK convention (H = I − τ·v·vᴴ, τ direct, v[last]=1 implicit), so `tau`
# matches LinearAlgebra.LAPACK.geqlf!/orgql!/gerqf!/orgrq! DIRECTLY (no faer 1/τ inversion — unlike
# qr.jl). ONE generic path per routine covers real and complex by threading `conj` (identity on reals)
# exactly where LAPACK's z*-routines call ZLACGV. The reflector GENERATOR is reused from svd.jl's
# `_larfg!` via a REVERSED view (`x[end:-1:1]`): _larfg! wants α at x[1] with the tail below, whereas
# the corner reflectors carry α at the LAST position — a reversed view puts α first, and since _larfg!
# scales elementwise the reversed essential lands in the same storage LAPACK expects.
#
# Ports the UNBLOCKED kernels only. Correct for all shapes; ponytail: blocked (compact-WY) deferred —
# add when perf-gated (mirror qr.jl/geqrf!).

# ── Corner reflector applies (shared by QL and RQ) ──────────────────────────────────────────────────
# H = I − τ·v·vᴴ with v[len]=1 implicit (essential in v[1:len-1]), applied from the LEFT: C := H·C.
# C is len×nc; the "1" of v sits at the BOTTOM row. (conj is identity on reals.)
@inline function _corner_left!(C::AbstractMatrix{T}, v::AbstractVector{T}, τ::T) where {T}
    iszero(τ) && return C
    len, nc = size(C)
    @inbounds for j in 1:nc
        w = C[len, j]
        for i in 1:(len - 1)
            w += conj(v[i]) * C[i, j]
        end   # w = vᴴ·C[:,j]  (v[len]=1)
        w *= τ
        C[len, j] -= w
        for i in 1:(len - 1)
            C[i, j] -= v[i] * w
        end          # C[:,j] −= τ·(vᴴC)·v
    end
    return C
end
# H = I − τ·v·vᴴ with v[len]=1 implicit, applied from the RIGHT: C := C·H. C is nr×len; the "1" of v
# sits at the last COLUMN. Used by ormql/ormrq side='R', gerqf, and orgrq.
@inline function _corner_right!(C::AbstractMatrix{T}, v::AbstractVector{T}, τ::T) where {T}
    iszero(τ) && return C
    nr, len = size(C)
    @inbounds for i in 1:nr
        w = C[i, len]
        for j in 1:(len - 1)
            w += C[i, j] * v[j]
        end          # w = (C·v)_i  (v[len]=1)
        w *= τ
        C[i, len] -= w
        for j in 1:(len - 1)
            C[i, j] -= w * conj(v[j])
        end     # C[i,:] −= τ·w·vᴴ
    end
    return C
end

# ════════════════════════════════════ QL factorization ════════════════════════════════════════════
"""
    geqlf!(A, tau) -> A

QL factorization (LAPACK dgeqlf/zgeqlf, unblocked dgeql2/zgeql2). Overwrites A (m×n): if m≥n the
lower triangle of the bottom-right n×n block holds L; the essential column reflectors sit above the
subdiagonal of that corner, `tau[i]` (standard convention) the coefficients. `A = Q·L`,
Q = H(k)…H(1), k = min(m,n). Reflectors are generated bottom-up.
"""
function geqlf!(A::AbstractMatrix{T}, tau::AbstractVector{T}) where {T}
    m, n = size(A); k = min(m, n)
    length(tau) >= k || throw(DimensionMismatch("geqlf!: length(tau) < min(size(A))"))
    @inbounds for i in k:-1:1
        s = m - k + i; c = n - k + i                         # corner element A[s,c]; α at the bottom
        xr = view(A, s:-1:1, c)                              # reversed → α=xr[1], essential below
        β, τ = _larfg!(xr)                                   # scales A[1:s-1,c] into v; β real (complex)
        tau[i] = τ
        A[s, c] = T(β)                                       # place L's corner diagonal
        # zgeql2 zeros the column with Hᴴ (=I−conj(τ)vvᴴ), so the trailing apply takes conj(τ) (id on reals).
        c > 1 && _corner_left!(view(A, 1:s, 1:(c - 1)), view(A, 1:s, c), T <: Complex ? conj(τ) : τ)
    end
    return A
end
function geqlf!(A::AbstractMatrix{T}) where {T}
    tau = Vector{T}(undef, min(size(A)...)); geqlf!(A, tau); return A, tau
end

"""
    orgql!(A, tau) -> A    (real)
    ungql!(A, tau) -> A    (complex)

Form Q (LAPACK dorgql/zungql, unblocked dorg2l/zung2l). On entry A (m×n, m≥n) holds the geqlf
reflectors; on exit A holds the m×n matrix Q with orthonormal columns. `k = length(tau)` reflectors.
"""
function orgql!(A::AbstractMatrix{T}, tau::AbstractVector{T}) where {T}
    m, n = size(A); k = length(tau)
    n <= m || throw(DimensionMismatch("orgql!: cols(A) > rows(A)"))
    k <= n || throw(DimensionMismatch("orgql!: length(tau) > cols(A)"))
    @inbounds begin
        for j in 1:(n - k)                                       # init leading n−k columns to unit columns of I
            for l in 1:m
                A[l, j] = zero(T)
            end
            A[m - n + j, j] = one(T)
        end
        for i in 1:k
            c = n - k + i; s = m - k + i                     # column ii=c, anchor row s
            τ = tau[i]
            c > 1 && _corner_left!(view(A, 1:s, 1:(c - 1)), view(A, 1:s, c), τ)  # apply H(i) to partial Q
            for l in 1:(s - 1)
                A[l, c] *= -τ
            end               # DSCAL(−τ) the essential
            A[s, c] = one(T) - τ
            for l in (s + 1):m
                A[l, c] = zero(T)
            end
        end
    end
    return A
end
const ungql! = orgql!

"""
    ormql!(side, trans, A, tau, C) -> C    (real: trans ∈ {'N','T'})
    unmql!(side, trans, A, tau, C) -> C    (complex: trans ∈ {'N','C'})

Apply Q (QL column reflectors in `A`, geqlf output) to C (m×n): side='L' → C := op(Q)·C (Q order
nq=m), side='R' → C := C·op(Q) (nq=n). op = Q (trans='N') or Qᴴ (trans='T'/'C'). LAPACK dorm2l/zunm2l.
"""
function ormql!(
        side::Char, trans::Char, A::AbstractMatrix{T}, tau::AbstractVector{T},
        C::AbstractMatrix{T}
    ) where {T}
    m, n = size(C); k = length(tau)
    left = side == 'L'; notran = trans == 'N'
    nq = left ? m : n
    if (left && notran) || (!left && !notran)                # Q=H(k)…H(1): apply H(1) first for Q·left/Qᴴ·right
        i1, i2, i3 = 1, k, 1
    else
        i1, i2, i3 = k, 1, -1
    end
    @inbounds begin
        i = i1
        while i3 > 0 ? i <= i2 : i >= i2
            arow = nq - k + i                                # anchor row of reflector i (v[arow]=1)
            τ = notran ? tau[i] : conj(tau[i])               # Qᴴ applies H(i)ᴴ = I−conj(τ)vvᴴ (conj id on reals)
            if left
                _corner_left!(view(C, 1:arow, 1:n), view(A, 1:arow, i), τ)      # mi = m−k+i = arow
            else
                _corner_right!(view(C, 1:m, 1:arow), view(A, 1:arow, i), τ)     # ni = n−k+i = arow
            end
            i += i3
        end
    end
    return C
end
const unmql! = ormql!

# ════════════════════════════════════ RQ factorization ════════════════════════════════════════════
"""
    gerqf!(A, tau) -> A

RQ factorization (LAPACK dgerqf/zgerqf, unblocked dgerq2/zgerq2). Overwrites A (m×n): the upper
triangle of the top-right m×m block holds R; the essential ROW reflectors sit to the left, `tau[i]`
the coefficients. `A = R·Q`, Q = H(1)…H(k) (real) / H(1)ᴴ…H(k)ᴴ (complex), k = min(m,n).
"""
function gerqf!(A::AbstractMatrix{T}, tau::AbstractVector{T}) where {T}
    m, n = size(A); k = min(m, n)
    length(tau) >= k || throw(DimensionMismatch("gerqf!: length(tau) < min(size(A))"))
    @inbounds for i in k:-1:1
        r = m - k + i; t = n - k + i                         # corner element A[r,t]; α at the right end
        for jj in 1:t
            A[r, jj] = conj(A[r, jj])
        end        # ZLACGV the row (no-op real)
        xr = view(A, r, t:-1:1)                              # reversed → α=xr[1], essential to its left
        β, τ = _larfg!(xr); tau[i] = τ                       # scales A[r,1:t-1] into v; β real (complex)
        A[r, t] = T(β)                                       # place R's corner diagonal
        r > 1 && _corner_right!(view(A, 1:(r - 1), 1:t), view(A, r, 1:t), τ)  # H(i) → A[1:r-1,1:t] from right
        for jj in 1:t
            A[r, jj] = conj(A[r, jj])
        end        # ZLACGV back
    end
    return A
end
function gerqf!(A::AbstractMatrix{T}) where {T}
    tau = Vector{T}(undef, min(size(A)...)); gerqf!(A, tau); return A, tau
end

"""
    orgrq!(A, tau) -> A    (real)
    ungrq!(A, tau) -> A    (complex)

Form Q (LAPACK dorgrq/zungrq, unblocked dorgr2/zungr2). On entry A (m×n, m≤n) holds the gerqf
reflectors in its rows; on exit A holds the m×n matrix Q with orthonormal rows. `k = length(tau)`.
"""
function orgrq!(A::AbstractMatrix{T}, tau::AbstractVector{T}) where {T}
    m, n = size(A); k = length(tau)
    m <= n || throw(DimensionMismatch("orgrq!: rows(A) > cols(A)"))
    k <= m || throw(DimensionMismatch("orgrq!: length(tau) > rows(A)"))
    @inbounds begin
        if k < m                                             # init leading m−k rows to unit rows of I
            for j in 1:n, l in 1:(m - k)
                A[l, j] = zero(T)
            end
            for j in 1:n
                (j > n - m && j <= n - k) && (A[m - n + j, j] = one(T))
            end
        end
        for i in 1:k
            r = m - k + i; t = n - k + i                     # row ii=r, anchor col t
            for jj in 1:(t - 1)
                A[r, jj] = conj(A[r, jj])
            end  # ZLACGV essential
            A[r, t] = one(T)                                 # anchor
            τc = T <: Complex ? conj(tau[i]) : tau[i]        # forming Q applies H(i)ᴴ
            r > 1 && _corner_right!(view(A, 1:(r - 1), 1:t), view(A, r, 1:t), τc)
            for jj in 1:(t - 1)
                A[r, jj] *= -tau[i]
            end         # DSCAL(−τ)
            for jj in 1:(t - 1)
                A[r, jj] = conj(A[r, jj])
            end  # ZLACGV back
            A[r, t] = one(T) - τc
            for jj in (t + 1):n
                A[r, jj] = zero(T)
            end
        end
    end
    return A
end
const ungrq! = orgrq!

"""
    ormrq!(side, trans, A, tau, C) -> C    (real: trans ∈ {'N','T'})
    unmrq!(side, trans, A, tau, C) -> C    (complex: trans ∈ {'N','C'})

Apply Q (RQ row reflectors in `A`, gerqf output) to C (m×n): side='L' → C := op(Q)·C (nq=m),
side='R' → C := C·op(Q) (nq=n). op = Q (trans='N') or Qᴴ (trans='T'/'C'). LAPACK dormr2/zunmr2.
"""
function ormrq!(
        side::Char, trans::Char, A::AbstractMatrix{T}, tau::AbstractVector{T},
        C::AbstractMatrix{T}
    ) where {T}
    m, n = size(C); k = length(tau)
    left = side == 'L'; notran = trans == 'N'
    nq = left ? m : n
    if (left && !notran) || (!left && notran)
        i1, i2, i3 = 1, k, 1
    else
        i1, i2, i3 = k, 1, -1
    end
    @inbounds begin
        i = i1
        while i3 > 0 ? i <= i2 : i >= i2
            c = nq - k + i                                   # anchor column of row-reflector i (v[c]=1)
            for jj in 1:(c - 1)
                A[i, jj] = conj(A[i, jj])
            end  # ZLACGV (no-op real)
            τ = notran ? (T <: Complex ? conj(tau[i]) : tau[i]) : tau[i]  # row-reflector conj rule (cf. ormlq)
            if left
                _corner_left!(view(C, 1:c, 1:n), view(A, i, 1:c), τ)    # mi = m−k+i = c
            else
                _corner_right!(view(C, 1:m, 1:c), view(A, i, 1:c), τ)   # ni = n−k+i = c
            end
            for jj in 1:(c - 1)
                A[i, jj] = conj(A[i, jj])
            end  # ZLACGV back
            i += i3
        end
    end
    return C
end
const unmrq! = ormrq!
