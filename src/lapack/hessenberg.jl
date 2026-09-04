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
@inline function _larf_right!(C::AbstractMatrix{T}, v::AbstractVector{T}, τ::T) where {T <: Number}
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
@inline function _dlamch_safmin(::Type{R}) where {R <: Real}
    sfmin = floatmin(R)
    small = one(R) / floatmax(R)
    small >= sfmin && (sfmin = small * (one(R) + eps(R)))
    return sfmin
end

# lassq-style Euclidean norm (req#6 overflow/underflow-safe) of A[lo:hi, i] (column) — DNRM2/DZNRM2 role.
@inline function _bal_colnrm2(A::AbstractMatrix{T}, i::Int, lo::Int, hi::Int) where {T <: Number}
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
@inline function _bal_rownrm2(A::AbstractMatrix{T}, i::Int, lo::Int, hi::Int) where {T <: Number}
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
@inline function _bal_colamax(A::AbstractMatrix{T}, i::Int, lo::Int, hi::Int) where {T <: Number}
    R = real(T); best = zero(R); ca = zero(R)
    @inbounds for r in lo:hi
        x = A[r, i]; c1 = abs(real(x)) + abs(imag(x))
        c1 > best && (best = c1; ca = abs(x))
    end
    return ca
end
@inline function _bal_rowamax(A::AbstractMatrix{T}, i::Int, lo::Int, hi::Int) where {T <: Number}
    R = real(T); best = zero(R); ra = zero(R)
    @inbounds for c in lo:hi
        x = A[i, c]; c1 = abs(real(x)) + abs(imag(x))
        c1 > best && (best = c1; ra = abs(x))
    end
    return ra
end
# Row/col exchange (dgebal label 20): SCALE(m)=j; swap columns j,m over rows 1:l and rows j,m over cols k:n.
@inline function _bal_exch!(
        A::AbstractMatrix{T}, scale::AbstractVector{R}, j::Int, m::Int,
        k::Int, l::Int, n::Int
    ) where {T <: Number, R <: Real}
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
    gebal!(A, scale; job='B') -> (ilo, ihi)
    gebal!(A; job='B')        -> (ilo, ihi, scale)

Balance a general square matrix `A` in place (LAPACK dgebal/zgebal). `job`:
`'N'` none, `'P'` permute-only, `'S'` scale-only, `'B'` both (default). Returns the isolated-eigenvalue
range `ilo:ihi` and the `scale` vector (permutation indices outside `ilo:ihi`, diagonal scaling factors
within). The scaling loop copies dgebal exactly: radix `SCLFAC=2`, convergence `FACTOR=0.95`. Generic over
`T<:Number`; `scale` is `real(T)`.

The 2-argument form takes the caller's `scale` (length ≥ n, `real(T)`-valued) — the `getrf!(A, ipiv)`
treatment — and allocates nothing; the 1-argument form is the allocating convenience wrapper.
`scale[1:n]` is set to `one` on entry and THAT IS LOAD-BEARING: the `job='N'` and `n == 0` exits write
nothing at all, so an unfilled (or reused) buffer would hand back live garbage where LAPACK returns
all-ones.
"""
function gebal!(
        A::AbstractMatrix{T}, scale::AbstractVector{<:Real}; job::Char = 'B'
    ) where {T <: Number}
    R = real(T)
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("gebal!: A must be square"))
    (job === 'N' || job === 'P' || job === 'S' || job === 'B') ||
        throw(ArgumentError("gebal!: job must be one of N/P/S/B"))
    length(scale) >= n || throw(DimensionMismatch("gebal!: length(scale) < n"))
    # `scale` is written while A is both read and written for the whole routine — an overlapping buffer
    # corrupts both silently (see syconv!'s `work`). PtrVector/PtrMatrix are invisible to mightalias.
    Base.mightalias(scale, A) && throw(ArgumentError("gebal!: `scale` must not alias `A`"))
    @inbounds for i in 1:n                          # sane default for the degenerate l==1 / 'N' exits
        scale[i] = one(R)
    end
    k = 1; l = n
    n == 0 && return k, l
    if job === 'N'
        return k, l                                 # scale already all-ones; ilo=1, ihi=n
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
    return k, l
end

# Allocating convenience form — unchanged behaviour and return value.
function gebal!(A::AbstractMatrix{T}; job::Char = 'B') where {T <: Number}
    scale = Vector{real(T)}(undef, size(A, 1))
    k, l = gebal!(A, scale; job = job)
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
function gehrd!(
        A::AbstractMatrix{T}, ilo::Integer, ihi::Integer,
        tau::AbstractVector{T}
    ) where {T <: Number}
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
    # Arena borrow (arena.jl), not the old owned `gehrdv` field: the reflector staging vector, hoisted ABOVE
    # the reflector loop as @scope requires. The length is loop-invariant (only the live prefix v[1:m]
    # shrinks), and v[1:m] is fully written from A before it is read each iteration, so no fill!. Exact
    # length, no anti-aliasing stride: the field it replaces named none.
    # NOTHING ESCAPES. `vv = view(v, 1:m)` is a PtrVector alias of the borrow and reaches exactly two
    # callees — `_larf_right!` (this file) and `_house_left!` (svd.jl). Both only READ v, elementwise or
    # through a `pointer(v)` inside a `GC.@preserve` that ends with the call, and both return their `C`
    # argument; neither stores it in a field, a global or a closure. `_larfg!` never sees it. The returned
    # `A` and the written `tau` are the caller's own arrays.
    @scope arn begin
        v = borrow!(arn, T, Int(ihi) - Int(ilo) + 1)
        @inbounds for i in Int(ilo):(Int(ihi) - 1)
            m = ihi - i                                  # reflector length (rows i+1:ihi)
            β, τ = _larfg!(view(A, (i + 1):ihi, i))          # essential v now in A[i+2:ihi,i]; A[i+1,i] left as α
            tau[i] = τ
            v[1] = one(T)
            for r in 2:m
                v[r] = A[i + r, i]
            end
            vv = view(v, 1:m)
            _larf_right!(view(A, 1:ihi, (i + 1):ihi), vv, τ)                 # A := A·H(i)   (right, τ)
            i < n && _house_left!(view(A, (i + 1):ihi, (i + 1):n), vv, conj(τ))  # A := H(i)ᴴ·A (left, conj τ)
            A[i + 1, i] = β                                # subdiagonal element
        end
    end
    return A
end

# Convenience: whole-matrix reduction (ilo=1, ihi=n), allocate tau, return (A, tau) — mirrors LAPACK.gehrd!.
function gehrd!(A::AbstractMatrix{T}) where {T <: Number}
    n = size(A, 1)
    tau = Vector{T}(undef, max(n - 1, 0))
    gehrd!(A, 1, n, tau)
    return A, tau
end

# ── orghr / unghr (LAPACK dorghr/zunghr) ────────────────────────────────────────────────────────────────
"""
    orghr!(A, ilo, ihi, tau) -> A   (A overwritten with Q)

Form the orthogonal/unitary `Q = H(ilo)·H(ilo+1)···H(ihi-1)` from the reflectors produced by [`gehrd!`]
(stored below the subdiagonal of `A`, coefficients `tau`). `Q` is `n×n` (identity outside `ilo:ihi`). `A` is
overwritten with `Q` (LAPACK contract) and returned. Generic over `T<:Number`; `unghr!` is an alias.
Use `_orghr_into!(Q, A, ilo, ihi, tau)` when `A` must survive.

Direct reflector-to-identity accumulation (applies `H(i)` to `I` in decreasing `i`), mirroring `_ormtr!`/
`_unmtr!` trans='N' — correctness-first, no dorgqr shift-trick needed.
"""
function orghr!(
        A::AbstractMatrix{T}, ilo::Integer, ihi::Integer,
        tau::AbstractVector{T}
    ) where {T <: Number}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("orghr!: A must be square"))
    # Arena borrow (arena.jl), not the old owned `orghrq` field: the n×n accumulation target. Exact ld (= n)
    # — the field it replaces named no anti-aliasing stride — and no fill! here, because `_orghr_into!`'s
    # first statement is its own fill!(Q, 0) + unit-diagonal write.
    # NOTHING ESCAPES. `Q` reaches exactly two callees: `_orghr_into!` (below — it writes Q and returns it,
    # and that return value is discarded here) and `copyto!`, which reads it. `_orghr_into!` opens its own
    # nested @scope for its `v`, whose borrow is released before this one; it retains nothing across the
    # call. By the time this block exits the values live in the caller's `A`, and the handle dies here.
    @scope arn begin
        Q = borrow!(arn, T, n, n)
        _orghr_into!(Q, A, ilo, ihi, tau)
        copyto!(A, Q)
    end
    return A                                   # A now HOLDS Q; the borrow itself is dead past the @scope.
end

# Non-destructive orghr: build Q into a caller-supplied buffer, leaving `A` (the reflector store) intact.
# This is what lets `ormhr!` drop its `copy(A)`. `Q` MUST NOT alias `A`: the loop reads reflectors out of
# `A[i+r, i]` while writing `Q[i+1:ihi, 1:n]`, exactly the `orgtr!('L', A, tau, A)` failure (eigen.jl:757)
# — netlib overwrites A, this form does not, so an aliased call is silently wrong. Throw instead.
function _orghr_into!(
        Q::AbstractMatrix{T}, A::AbstractMatrix{T}, ilo::Integer, ihi::Integer,
        tau::AbstractVector{T}
    ) where {T <: Number}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("orghr!: A must be square"))
    (size(Q, 1) == n && size(Q, 2) == n) || throw(DimensionMismatch("orghr!: Q must be n×n"))
    Base.mightalias(Q, A) && throw(ArgumentError("orghr!: Q must not alias A (the reflectors are read from A while Q is written)"))
    fill!(Q, zero(T))
    @inbounds for i in 1:n
        Q[i, i] = one(T)
    end
    if ihi - ilo >= 1
        # Arena borrow (arena.jl), not the old owned `orghrv` field: reflector staging, hoisted ABOVE the
        # loop as @scope requires (the length is loop-invariant; v[1:m] is fully written from A before it is
        # read, so no fill!). Still a buffer DISTINCT from gehrd!'s — ormhr! → _orghr_into! is a real nesting
        # chain — but under the arena that is automatic: each scope bumps its own bytes.
        # NOTHING ESCAPES. `view(v, 1:m)` reaches only `_house_left!` (svd.jl), which reads it (elementwise,
        # or via `pointer(v)` inside a `GC.@preserve` bounded by the call) and returns its `C` argument. The
        # `Q` returned below is the CALLER's matrix — never this borrow.
        @scope arn begin
            v = borrow!(arn, T, Int(ihi) - Int(ilo) + 1)
            @inbounds for i in (Int(ihi) - 1):-1:Int(ilo)
                m = ihi - i
                v[1] = one(T)
                for r in 2:m
                    v[r] = A[i + r, i]
                end
                _house_left!(view(Q, (i + 1):ihi, 1:n), view(v, 1:m), tau[i])
            end
        end
    end
    return Q
end

unghr!(A::AbstractMatrix{T}, ilo::Integer, ihi::Integer, tau::AbstractVector{T}) where {T <: Number} =
    orghr!(A, ilo, ihi, tau)

# ── ormhr / unmhr — apply Q from gehrd reflectors to a general matrix C ────────────────────────────────
# LAPACK dormhr/zunmhr applies Q (or Qᴴ) WITHOUT first forming it (a dedicated reflector sweep). Correct-
# ness-first / ponytail: `orghr!` already forms Q correctly and cheaply enough for this batch's scope (a
# direct-LAPACK-caller symbol, not on any PureBLAS-internal hot path) — form Q into an arena borrow with
# `_orghr_into!` (which leaves A unchanged, as C-ABI dormhr/zunmhr requires) then apply via PureBLAS's own
# `gemm!` (trim-safe; avoids routing through `Base.:*`'s generic BLAS dispatch).
"""
    ormhr!(side, trans, ilo, ihi, A, tau, C) -> C    (real: trans ∈ {'N','T'})
    unmhr!(side, trans, ilo, ihi, A, tau, C) -> C    (complex: trans ∈ {'N','C'})

Apply `Q` (LAPACK dormhr/zunmhr — the `gehrd!`/`orghr!` reflectors) to `C`: side='L' → `C := op(Q)·C`,
side='R' → `C := C·op(Q)`. `A`/`tau` are left unchanged (read-only, as in reference LAPACK).
"""
function ormhr!(
        side::Char, trans::Char, ilo::Integer, ihi::Integer, A::AbstractMatrix{T},
        tau::AbstractVector{T}, C::AbstractMatrix{T}
    ) where {T <: Number}
    n = size(A, 1)
    # Arena borrows (arena.jl), replacing the owned `mhrq`/`mhrc` pair: ormhr!'s OWN Q — distinct from
    # orghr!'s because ormhr! NESTS `_orghr_into!`, which under the arena is automatic — and the gemm!
    # staging tile. Both side='L'/'R' arms stage into a buffer of exactly size(C), mutually exclusive
    # branches of one call, so ONE borrow. Exact lds (neither field named an anti-aliasing stride). No
    # fill!: Q is fully written by `_orghr_into!`, and each gemm! runs beta = 0, a pure overwrite of the
    # whole mc×nc tile.
    # NOTHING ESCAPES. `Q` reaches `_orghr_into!` (writes it, returns it — return discarded; it opens its
    # own nested @scope whose borrow is released first) and `gemm!` as a read-only operand. `tmp` reaches
    # `gemm!` as its C argument and then `copyto!`. `gemm!` packs/reads its operands and writes its output
    # tile within the call — it stores no operand in a field, a global or a closure — and the result is
    # copied into the caller's `C` before this block exits. Only `C` (the caller's) is returned.
    @scope arn begin
        Q = borrow!(arn, T, n, n)
        tmp = borrow!(arn, T, size(C, 1), size(C, 2))
        _orghr_into!(Q, A, ilo, ihi, tau)          # non-destructive ⇒ the old `copy(A)` is gone
        ct = T <: Complex ? 'C' : 'T'
        top = trans == 'N' ? 'N' : ct
        if side == 'L'
            gemm!(tmp, Q, C; transA = top, alpha = one(T), beta = zero(T))
        else
            gemm!(tmp, C, Q; transB = top, alpha = one(T), beta = zero(T))
        end
        copyto!(C, tmp)
    end
    return C
end
const unmhr! = ormhr!
