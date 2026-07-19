# LAPACK nonsymmetric-eigen REDUCTION kernels (first half of the general eigensolver `eigen(A)`):
#   gebal  — balance (permute to isolate eigenvalues + diagonal-similarity norm reduction)  [dgebal/zgebal]
#   gehrd  — reduce to upper Hessenberg H = Qᴴ·A·Q via Householder reflectors                [dgehd2/zgehd2]
#   orghr  — form the orthogonal/unitary Q from gehrd's reflectors (eigenvector back-transform) [dorghr/zunghr]
# Correctness-first UNBLOCKED gehrd (dgehd2); the blocked dlahr2+WY path is a perf follow-up (see note on
# gehrd!). Reuses the module's proven Householder machinery: `_larfg!` (dlarfg reflector generator, real &
# complex, svd.jl) and `_house_left!` (dlarf 'Left', generic over Real & Complex, svd.jl). Only the generic
# right-apply is new here (`_larf_right!`) — the existing `_house_right!` is Float64-only. Generic over
# T<:Number (s/d/c/z + AD), SIMD-free scalar loops (dgehd2 is BLAS-2 per column; the SIMD lever is the
# blocked follow-up), trim-safe. The Francis QR (hseqr) consumes these — reduction is numerically faithful.

# Apply H = I − τ·v·vᴴ (v[1]≡1 implicit, essential v[2:] supplied) to C (nr×len) from the RIGHT: C := C·H.
# Generic mirror of svd.jl's Float64-only `_house_right!`; conj is a no-op for T<:Real (reduces to the real
# formula) and the zlarf 'Right' conjugation for T<:Complex.
@inline function _larf_right!(C::AbstractMatrix{T}, v::AbstractVector{T}, τ::T) where {T<:Number}
    iszero(τ) && return C
    nr, len = size(C)
    @inbounds for i in 1:nr
        w = C[i, 1]
        for j in 2:len
            w += C[i, j] * v[j]
        end
        w *= τ
        C[i, 1] -= w
        for j in 2:len
            C[i, j] -= w * conj(v[j])
        end
    end
    return C
end

# ── gebal (LAPACK dgebal/zgebal) ──────────────────────────────────────────────────────────────────────
# DLAMCH('S') = safmin, adjusted so 1/safmin does not overflow (matches reference dlamch).
@inline function _dlamch_safmin(::Type{R}) where {R<:Real}
    sfmin = floatmin(R)
    small = one(R) / floatmax(R)
    small >= sfmin && (sfmin = small * (one(R) + eps(R)))
    return sfmin
end

# lassq-style Euclidean norm (req#6 overflow/underflow-safe) of A[lo:hi, i] (column) — DNRM2/DZNRM2 role.
@inline function _bal_colnrm2(A::AbstractMatrix{T}, i::Int, lo::Int, hi::Int) where {T<:Number}
    R = real(T); scl = zero(R); ssq = one(R); nz = false
    @inbounds for r in lo:hi
        x = A[r, i]
        if !iszero(x)
            nz = true; ax = abs(x)
            if scl < ax
                ssq = one(R) + ssq * (scl / ax)^2; scl = ax
            else
                ssq += (ax / scl)^2
            end
        end
    end
    return nz ? scl * sqrt(ssq) : zero(R)
end
# … of A[i, lo:hi] (row).
@inline function _bal_rownrm2(A::AbstractMatrix{T}, i::Int, lo::Int, hi::Int) where {T<:Number}
    R = real(T); scl = zero(R); ssq = one(R); nz = false
    @inbounds for c in lo:hi
        x = A[i, c]
        if !iszero(x)
            nz = true; ax = abs(x)
            if scl < ax
                ssq = one(R) + ssq * (scl / ax)^2; scl = ax
            else
                ssq += (ax / scl)^2
            end
        end
    end
    return nz ? scl * sqrt(ssq) : zero(R)
end
# |A[ICA,i]| where ICA = IDAMAX/IZAMAX over rows lo:hi (index by cabs1 = |re|+|im|, value by modulus).
@inline function _bal_colamax(A::AbstractMatrix{T}, i::Int, lo::Int, hi::Int) where {T<:Number}
    R = real(T); best = zero(R); ca = zero(R)
    @inbounds for r in lo:hi
        x = A[r, i]; c1 = abs(real(x)) + abs(imag(x))
        c1 > best && (best = c1; ca = abs(x))
    end
    return ca
end
@inline function _bal_rowamax(A::AbstractMatrix{T}, i::Int, lo::Int, hi::Int) where {T<:Number}
    R = real(T); best = zero(R); ra = zero(R)
    @inbounds for c in lo:hi
        x = A[i, c]; c1 = abs(real(x)) + abs(imag(x))
        c1 > best && (best = c1; ra = abs(x))
    end
    return ra
end
# Row/col exchange (dgebal label 20): SCALE(m)=j; swap columns j,m over rows 1:l and rows j,m over cols k:n.
@inline function _bal_exch!(A::AbstractMatrix{T}, scale::AbstractVector{R}, j::Int, m::Int,
        k::Int, l::Int, n::Int) where {T<:Number, R<:Real}
    scale[m] = R(j)
    j == m && return
    @inbounds for r in 1:l
        A[r, j], A[r, m] = A[r, m], A[r, j]
    end
    @inbounds for c in k:n
        A[j, c], A[m, c] = A[m, c], A[j, c]
    end
    return
end

"""
    gebal!(A; job='B') -> (ilo, ihi, scale)

Balance a general square matrix `A` in place (LAPACK dgebal/zgebal). `job`:
`'N'` none, `'P'` permute-only, `'S'` scale-only, `'B'` both (default). Returns the isolated-eigenvalue
range `ilo:ihi` and the `scale` vector (permutation indices outside `ilo:ihi`, diagonal scaling factors
within). The scaling loop copies dgebal exactly: radix `SCLFAC=2`, convergence `FACTOR=0.95`. Generic over
`T<:Number`; `scale` is `real(T)`.
"""
function gebal!(A::AbstractMatrix{T}; job::Char = 'B') where {T<:Number}
    R = real(T)
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("gebal!: A must be square"))
    (job === 'N' || job === 'P' || job === 'S' || job === 'B') ||
        throw(ArgumentError("gebal!: job must be one of N/P/S/B"))
    scale = ones(R, n)                              # sane default for the degenerate l==1 exit
    k = 1; l = n
    n == 0 && return k, l, scale
    if job === 'N'
        return k, l, scale                          # scale already all-ones; ilo=1, ihi=n
    end

    if job !== 'S'
        # ===== permutation: isolate eigenvalues into the leading/trailing borders =====
        while true                                  # push isolated ROWS down (shrinks l)
            found = false
            for j in l:-1:1
                iso = true
                @inbounds for i in 1:l
                    (i != j && !iszero(A[j, i])) && (iso = false; break)
                end
                if iso
                    _bal_exch!(A, scale, j, l, k, l, n)
                    l == 1 && @goto finish          # fully triangularized by permutation
                    l -= 1; found = true; break
                end
            end
            found || break
        end
        while true                                  # push isolated COLUMNS left (grows k)
            found = false
            for j in k:l
                iso = true
                @inbounds for i in k:l
                    (i != j && !iszero(A[i, j])) && (iso = false; break)
                end
                if iso
                    _bal_exch!(A, scale, j, k, k, l, n)
                    k += 1; found = true; break
                end
            end
            found || break
        end
    end

    @inbounds for i in k:l
        scale[i] = one(R)
    end
    job === 'P' && @goto finish

    # ===== scaling: diagonal similarity to reduce the 1-norm of rows/cols k:l (dgebal, verbatim) =====
    sclfac = R(2)
    factor = R(0.95)
    sfmin1 = _dlamch_safmin(R) / eps(R)             # DLAMCH('S')/DLAMCH('P'); DLAMCH('P') = eps*base = eps(R)
    sfmax1 = one(R) / sfmin1
    sfmin2 = sfmin1 * sclfac
    sfmax2 = one(R) / sfmin2
    noconv = true
    while noconv
        noconv = false
        for i in k:l
            c = _bal_colnrm2(A, i, k, l)
            r = _bal_rownrm2(A, i, k, l)
            ca = _bal_colamax(A, i, 1, l)
            ra = _bal_rowamax(A, i, k, n)
            (iszero(c) || iszero(r)) && continue    # guard against underflow-zero C or R
            g = r / sclfac
            f = one(R)
            s = c + r
            while !(c >= g || max(f, c, ca) >= sfmax2 || min(r, g, ra) <= sfmin2)
                f *= sclfac; c *= sclfac; ca *= sclfac
                r /= sclfac; g /= sclfac; ra /= sclfac
            end
            g = c / sclfac
            while !(g < r || max(r, ra) >= sfmax2 || min(f, c, g, ca) <= sfmin2)
                f /= sclfac; c /= sclfac; g /= sclfac; ca /= sclfac
                r *= sclfac; ra *= sclfac
            end
            (c + r >= factor * s) && continue       # not enough reduction → skip
            if f < one(R) && scale[i] < one(R)
                f * scale[i] <= sfmin1 && continue
            end
            if f > one(R) && scale[i] > one(R)
                scale[i] >= sfmax1 / f && continue
            end
            g = one(R) / f
            scale[i] *= f
            noconv = true
            @inbounds for c2 in k:n
                A[i, c2] *= g
            end
            @inbounds for r2 in 1:l
                A[r2, i] *= f
            end
        end
    end

    @label finish
    return k, l, scale
end

# ── gehrd (LAPACK dgehd2/zgehd2, UNBLOCKED) ─────────────────────────────────────────────────────────────
"""
    gehrd!(A, ilo, ihi, tau) -> A

Reduce `A[ilo:ihi, ilo:ihi]` to upper Hessenberg `H = Qᴴ·A·Q` in place (LAPACK dgehd2/zgehd2, unblocked).
Reflector `i` (i = ilo…ihi-1) zeros `A[i+2:ihi, i]`; its essential part is stored below the subdiagonal in
column `i`, `tau[i]` the standard-LAPACK coefficient (`H_i = I − τ_i·v_i·v_iᴴ`). On output the subdiagonal +
upper triangle hold `H`; `tau` outside `[ilo, ihi-1]` is zeroed. Assumes `A` already permuted by `gebal!`.

ponytail: unblocked (BLAS-2/column) for correctness-first; the blocked dlahr2 + compact-WY trailing-gemm
path (`wy.jl` kernels + `gemm!`, mirroring `geqrf!`) is the perf follow-up — flagged, not built.
"""
function gehrd!(A::AbstractMatrix{T}, ilo::Integer, ihi::Integer,
        tau::AbstractVector{T}) where {T<:Number}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("gehrd!: A must be square"))
    length(tau) >= max(n - 1, 0) || throw(DimensionMismatch("gehrd!: length(tau) < n-1"))
    @inbounds for i in 1:(Int(ilo) - 1)
        tau[i] = zero(T)
    end
    @inbounds for i in max(1, Int(ihi)):(n - 1)
        tau[i] = zero(T)
    end
    (ihi - ilo < 1) && return A
    v = Vector{T}(undef, ihi - ilo + 1)
    @inbounds for i in Int(ilo):(Int(ihi) - 1)
        m = ihi - i                                  # reflector length (rows i+1:ihi)
        β, τ = _larfg!(view(A, i+1:ihi, i))          # essential v now in A[i+2:ihi,i]; A[i+1,i] left as α
        tau[i] = τ
        v[1] = one(T)
        for r in 2:m
            v[r] = A[i+r, i]
        end
        vv = view(v, 1:m)
        _larf_right!(view(A, 1:ihi, i+1:ihi), vv, τ)                 # A := A·H(i)   (right, τ)
        i < n && _house_left!(view(A, i+1:ihi, i+1:n), vv, conj(τ))  # A := H(i)ᴴ·A (left, conj τ)
        A[i+1, i] = β                                # subdiagonal element
    end
    return A
end

# Convenience: whole-matrix reduction (ilo=1, ihi=n), allocate tau, return (A, tau) — mirrors LAPACK.gehrd!.
function gehrd!(A::AbstractMatrix{T}) where {T<:Number}
    n = size(A, 1)
    tau = Vector{T}(undef, max(n - 1, 0))
    gehrd!(A, 1, n, tau)
    return A, tau
end

# ── orghr / unghr (LAPACK dorghr/zunghr) ────────────────────────────────────────────────────────────────
"""
    orghr!(A, ilo, ihi, tau) -> Q

Form the orthogonal/unitary `Q = H(ilo)·H(ilo+1)···H(ihi-1)` from the reflectors produced by [`gehrd!`]
(stored below the subdiagonal of `A`, coefficients `tau`). `Q` is `n×n` (identity outside `ilo:ihi`). `A` is
overwritten with `Q` (LAPACK contract) and `Q` is returned. Generic over `T<:Number`; `unghr!` is an alias.

Direct reflector-to-identity accumulation (applies `H(i)` to `I` in decreasing `i`), mirroring `_ormtr!`/
`_unmtr!` trans='N' — correctness-first, no dorgqr shift-trick needed.
"""
function orghr!(A::AbstractMatrix{T}, ilo::Integer, ihi::Integer,
        tau::AbstractVector{T}) where {T<:Number}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("orghr!: A must be square"))
    Q = Matrix{T}(undef, n, n)
    fill!(Q, zero(T))
    @inbounds for i in 1:n
        Q[i, i] = one(T)
    end
    if ihi - ilo >= 1
        v = Vector{T}(undef, ihi - ilo + 1)
        @inbounds for i in (Int(ihi) - 1):-1:Int(ilo)
            m = ihi - i
            v[1] = one(T)
            for r in 2:m
                v[r] = A[i+r, i]
            end
            _house_left!(view(Q, i+1:ihi, 1:n), view(v, 1:m), tau[i])
        end
    end
    copyto!(A, Q)
    return Q
end

unghr!(A::AbstractMatrix{T}, ilo::Integer, ihi::Integer, tau::AbstractVector{T}) where {T<:Number} =
    orghr!(A, ilo, ihi, tau)
