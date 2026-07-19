# LAPACK GENERALIZED nonsymmetric-eigen QZ kernels (A·x = λ·B·x):
#   gghrd  — reduce (A,B) to generalized upper Hessenberg-triangular form  [dgghrd/zgghrd]
#   hgeqz  — QZ iteration → generalized (real/complex) Schur form           [dhgeqz/zhgeqz]
#   tgevc  — generalized eigenvectors by back-substitution (SIDE='R')       [dtgevc/ztgevc]  (in tgevc.jl)
# CORRECTNESS-FIRST unblocked ports of Reference-LAPACK, transcribed line-for-line. This file is
# STANDALONE (not part of the PureBLAS module): every auxiliary (lartg/zlartg, lag2, lasv2, laln2,
# ladiv, lapy2/3) is a local port so the file `include`s cleanly on its own for validation.
#
# REAL path (T<:Real): real generalized Schur — S quasi-upper-triangular (2×2 blocks for
# complex-conjugate eigenvalue pairs), T upper-triangular; eigenvalues λ = (αr+iαi)/β.
# COMPLEX path (T<:Complex): S,T both upper-triangular; λ = α/β.
#
# Generic over T (s/d/c/z); scalar loops (unblocked, correctness-first — mirrors hseqr.jl/trevc.jl).

# ── DLAMCH('S') — safe minimum, adjusted so 1/safmin does not overflow ────────────────────────────────
@inline function _qz_safmin(::Type{R}) where {R<:Real}
    sfmin = floatmin(R)
    small = one(R) / floatmax(R)
    small >= sfmin && (sfmin = small * (one(R) + eps(R)))
    return sfmin
end

# DLAMCH('E')*DLAMCH('B') = eps(R) = ULP (unit in last place).
@inline _qz_ulp(::Type{R}) where {R<:Real} = eps(R)

# ── DLAPY2 / DLAPY3 (overflow-safe hypot) ──────────────────────────────────────────────────────────────
@inline _lapy2(x::R, y::R) where {R<:Real} = hypot(x, y)
@inline function _lapy3(x::R, y::R, z::R) where {R<:Real}
    w = max(abs(x), abs(y), abs(z))
    w == zero(R) && return abs(x) + abs(y) + abs(z)
    return w * sqrt((x / w)^2 + (y / w)^2 + (z / w)^2)
end

# ── DLADIV (robust complex division (a+bi)/(c+di) → (p,q)), Reference-LAPACK ────────────────────────────
@inline function _qz_ladiv2(a::R, b::R, c::R, d::R, r::R, t::R) where {R<:Real}
    if r != zero(R)
        br = b * r
        return br != zero(R) ? (a + br) * t : a * t + (b * t) * r
    else
        return (a + d * (b / c)) * t
    end
end
@inline function _qz_ladiv1(a::R, b::R, c::R, d::R) where {R<:Real}
    r = d / c
    t = one(R) / (c + d * r)
    return _qz_ladiv2(a, b, c, d, r, t), _qz_ladiv2(b, -a, c, d, r, t)
end
@inline function _ladiv(a::R, b::R, c::R, d::R) where {R<:Real}
    BS = R(2); HALF = R(0.5); TWO = R(2)
    aa = a; bb = b; cc = c; dd = d
    ab = max(abs(a), abs(b)); cd = max(abs(c), abs(d)); s = one(R)
    ov = floatmax(R); un = floatmin(R); ϵ = eps(R) / 2; be = BS / (ϵ * ϵ)
    if ab >= HALF * ov; aa *= HALF; bb *= HALF; s *= TWO; end
    if cd >= HALF * ov; cc *= HALF; dd *= HALF; s *= HALF; end
    if ab <= un * BS / ϵ; aa *= be; bb *= be; s /= be; end
    if cd <= un * BS / ϵ; cc *= be; dd *= be; s *= be; end
    if abs(d) <= abs(c)
        p, q = _qz_ladiv1(aa, bb, cc, dd)
    else
        p, q = _qz_ladiv1(bb, aa, dd, cc); q = -q
    end
    return p * s, q * s
end
@inline _zladiv(x::T, y::T) where {T<:Complex} =
    (p = _ladiv(real(x), imag(x), real(y), imag(y)); Complex(p[1], p[2]))

# ── DLARTG (real plane rotation), Reference-LAPACK dlartg.f90 — [c s; -s c][f;g]=[r;0], c≥0 ─────────────
@inline function _lartg(f::R, g::R) where {R<:Real}
    safmin = _qz_safmin(R); safmax = one(R) / safmin
    rtmin = sqrt(safmin); rtmax = sqrt(safmax / 2)
    f1 = abs(f); g1 = abs(g)
    if g == zero(R)
        return one(R), zero(R), f
    elseif f == zero(R)
        return zero(R), copysign(one(R), g), g1
    elseif f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax
        d = sqrt(f * f + g * g)
        c = f1 / d
        r = copysign(d, f)
        return c, g / r, r
    else
        u = min(safmax, max(safmin, f1, g1))
        fs = f / u; gs = g / u
        d = sqrt(fs * fs + gs * gs)
        c = abs(fs) / d
        r = copysign(d, f)
        return c, gs / r, r * u
    end
end

# ── ZLARTG (complex plane rotation), Reference-LAPACK zlartg.f90 ───────────────────────────────────────
# Returns (c::R, s::T, r::T) with [c s; -conj(s) c][f;g]=[r;0], c real ≥ 0.
@inline function _zlartg(f::T, g::T) where {T<:Complex}
    R = real(T)
    safmin = _qz_safmin(R); safmax = one(R) / safmin
    rtmin = sqrt(safmin)
    abssq(z) = real(z)^2 + imag(z)^2
    czero = zero(T)
    if g == czero
        return one(R), czero, f
    elseif f == czero
        if real(g) == zero(R)
            r = abs(imag(g)); return zero(R), conj(g) / r, complex(r)
        elseif imag(g) == zero(R)
            r = abs(real(g)); return zero(R), conj(g) / r, complex(r)
        else
            g1 = max(abs(real(g)), abs(imag(g)))
            rtmax = sqrt(safmax / 2)
            if g1 > rtmin && g1 < rtmax
                g2 = abssq(g); d = sqrt(g2)
                return zero(R), conj(g) / d, complex(d)
            else
                u = min(safmax, max(safmin, g1)); gs = g / u
                g2 = abssq(gs); d = sqrt(g2)
                return zero(R), conj(gs) / d, complex(d * u)
            end
        end
    else
        f1 = max(abs(real(f)), abs(imag(f)))
        g1 = max(abs(real(g)), abs(imag(g)))
        rtmax = sqrt(safmax / 4)
        if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax
            f2 = abssq(f); g2 = abssq(g); h2 = f2 + g2
            if f2 >= h2 * safmin
                c = sqrt(f2 / h2); r = f / c
                rtmx2 = rtmax * 2
                if f2 > rtmin && h2 < rtmx2
                    s = (f / sqrt(f2)) * (conj(g) / sqrt(h2))
                else
                    s = conj(g) * (r / h2)
                end
            else
                d = sqrt(f2 * h2); c = f2 / d
                r = c >= safmin ? f / c : f * (h2 / d)
                s = (f / sqrt(f2)) * (conj(g) / sqrt(h2))
            end
            return c, s, r
        else
            u = min(safmax, max(safmin, f1, g1))
            gs = g / u; g2 = abssq(gs)
            local fs, f2, h2, w
            if f1 / u < rtmin
                v = min(safmax, max(safmin, f1)); w = v / u
                fs = f / v; f2 = abssq(fs); h2 = f2 * w^2 + g2
            else
                w = one(R); fs = f / u; f2 = abssq(fs); h2 = f2 + g2
            end
            if f2 >= h2 * safmin
                c = sqrt(f2 / h2); r = fs / c
                rtmx2 = rtmax * 2
                if f2 > rtmin && h2 < rtmx2
                    s = (fs / sqrt(f2)) * (conj(gs) / sqrt(h2))
                else
                    s = conj(gs) * (r / h2)
                end
            else
                d = sqrt(f2 * h2); c = f2 / d
                r = c >= safmin ? fs / c : fs * (h2 / d)
                s = (fs / sqrt(f2)) * (conj(gs) / sqrt(h2))
            end
            return c * w, s, r * u
        end
    end
end

# ── Plane-rotation appliers (DROT/ZROT). First index (i1/j1) is the "cx" lane. ─────────────────────────
# real: new_cx = c·cx + s·cy ; new_cy = c·cy − s·cx
@inline function _grot_rows!(M::AbstractMatrix{R}, i1::Int, i2::Int, jlo::Int, jhi::Int, c::R, s::R) where {R<:Real}
    @inbounds for j in jlo:jhi
        t = M[i1, j]; u = M[i2, j]
        M[i1, j] = c * t + s * u
        M[i2, j] = c * u - s * t
    end
end
@inline function _grot_cols!(M::AbstractMatrix{R}, j1::Int, j2::Int, ilo::Int, ihi::Int, c::R, s::R) where {R<:Real}
    @inbounds for i in ilo:ihi
        t = M[i, j1]; u = M[i, j2]
        M[i, j1] = c * t + s * u
        M[i, j2] = c * u - s * t
    end
end
# complex: new_cx = c·cx + s·cy ; new_cy = c·cy − conj(s)·cx
@inline function _zrot_rows!(M::AbstractMatrix{T}, i1::Int, i2::Int, jlo::Int, jhi::Int, c::R, s::T) where {T<:Complex,R<:Real}
    @inbounds for j in jlo:jhi
        t = M[i1, j]; u = M[i2, j]
        M[i1, j] = c * t + s * u
        M[i2, j] = c * u - conj(s) * t
    end
end
@inline function _zrot_cols!(M::AbstractMatrix{T}, j1::Int, j2::Int, ilo::Int, ihi::Int, c::R, s::T) where {T<:Complex,R<:Real}
    @inbounds for i in ilo:ihi
        t = M[i, j1]; u = M[i, j2]
        M[i, j1] = c * t + s * u
        M[i, j2] = c * u - conj(s) * t
    end
end

@inline function _qz_init_qz!(comp::AbstractChar, Q::AbstractMatrix{T}, n::Int) where {T}
    if comp === 'I'
        fill!(Q, zero(T))
        @inbounds for i in 1:n; Q[i, i] = one(T); end
    end
    return Q
end

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
#  gghrd — generalized Hessenberg-triangular reduction (dgghrd / zgghrd)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
"""
    gghrd!(compq, compz, A, B, Q, Z; ilo=1, ihi=size(A,1)) -> (A, B)

Reduce a pair `(A,B)` (`B` already upper triangular) to generalized upper Hessenberg form:
`A ← Qᴴ·A·Z` upper Hessenberg, `B ← Qᴴ·B·Z` upper triangular, via Givens rotations. `compq`/`compz`
∈ `'N'`/`'I'`/`'V'` control accumulation of `Q`/`Z` (`'I'` initialises to `I`, `'V'` post-multiplies the
given matrix). Reference-LAPACK dgghrd/zgghrd. Generic over `T<:Number`.
"""
function gghrd!(compq::AbstractChar, compz::AbstractChar, A::AbstractMatrix{T}, B::AbstractMatrix{T},
        Q::AbstractMatrix{T}, Z::AbstractMatrix{T}; ilo::Integer = 1, ihi::Integer = size(A, 1)) where {T<:Real}
    n = size(A, 1)
    ilq = compq !== 'N'; ilz = compz !== 'N'
    ilo = Int(ilo); ihi = Int(ihi)
    _qz_init_qz!(compq, Q, n); _qz_init_qz!(compz, Z, n)
    n <= 1 && return A, B
    @inbounds for jc in 1:n-1, jr in jc+1:n
        B[jr, jc] = zero(T)
    end
    @inbounds for jcol in ilo:ihi-2
        for jrow in ihi:-1:jcol+2
            # Step 1: rotate rows jrow-1,jrow to kill A(jrow,jcol)
            c, s, r = _lartg(A[jrow-1, jcol], A[jrow, jcol])
            A[jrow-1, jcol] = r; A[jrow, jcol] = zero(T)
            _grot_rows!(A, jrow-1, jrow, jcol+1, n, c, s)
            _grot_rows!(B, jrow-1, jrow, jrow-1, n, c, s)
            ilq && _grot_cols!(Q, jrow-1, jrow, 1, n, c, s)
            # Step 2: rotate cols jrow,jrow-1 to kill B(jrow,jrow-1)
            c, s, r = _lartg(B[jrow, jrow], B[jrow, jrow-1])
            B[jrow, jrow] = r; B[jrow, jrow-1] = zero(T)
            _grot_cols!(A, jrow, jrow-1, 1, ihi, c, s)
            _grot_cols!(B, jrow, jrow-1, 1, jrow-1, c, s)
            ilz && _grot_cols!(Z, jrow, jrow-1, 1, n, c, s)
        end
    end
    return A, B
end

function gghrd!(compq::AbstractChar, compz::AbstractChar, A::AbstractMatrix{T}, B::AbstractMatrix{T},
        Q::AbstractMatrix{T}, Z::AbstractMatrix{T}; ilo::Integer = 1, ihi::Integer = size(A, 1)) where {T<:Complex}
    n = size(A, 1)
    ilq = compq !== 'N'; ilz = compz !== 'N'
    ilo = Int(ilo); ihi = Int(ihi)
    _qz_init_qz!(compq, Q, n); _qz_init_qz!(compz, Z, n)
    n <= 1 && return A, B
    @inbounds for jc in 1:n-1, jr in jc+1:n
        B[jr, jc] = zero(T)
    end
    @inbounds for jcol in ilo:ihi-2
        for jrow in ihi:-1:jcol+2
            c, s, r = _zlartg(A[jrow-1, jcol], A[jrow, jcol])
            A[jrow-1, jcol] = r; A[jrow, jcol] = zero(T)
            _zrot_rows!(A, jrow-1, jrow, jcol+1, n, c, s)
            _zrot_rows!(B, jrow-1, jrow, jrow-1, n, c, s)
            ilq && _zrot_cols!(Q, jrow-1, jrow, 1, n, c, conj(s))
            c, s, r = _zlartg(B[jrow, jrow], B[jrow, jrow-1])
            B[jrow, jrow] = r; B[jrow, jrow-1] = zero(T)
            _zrot_cols!(A, jrow, jrow-1, 1, ihi, c, s)
            _zrot_cols!(B, jrow, jrow-1, 1, jrow-1, c, s)
            ilz && _zrot_cols!(Z, jrow, jrow-1, 1, n, c, s)
        end
    end
    return A, B
end

# ── DLAG2 (2×2 generalized eigenvalue), Reference-LAPACK dlag2.f ───────────────────────────────────────
# Inputs: A = [a11 a12; a21 a22], B = [b11 b12; 0 b22] (b21≡0). `safmin` is the perturbation floor.
# Returns (scale1, scale2, wr1, wr2, wi): eigenvalues (wr1±? ) / scale as in dlag2 (WI≠0 ⇒ complex pair).
function _lag2(a11::R, a21::R, a12::R, a22::R, b11i::R, b12i::R, b22i::R, safmin::R) where {R<:Real}
    ONE = one(R); ZERO = zero(R); HALF = R(0.5); FUZZY1 = ONE + R(1) / R(100000)
    rtmin = sqrt(safmin); rtmax = ONE / rtmin; safmax = ONE / safmin
    anorm = max(abs(a11) + abs(a21), abs(a12) + abs(a22), safmin)
    ascale = ONE / anorm
    a11 *= ascale; a21 *= ascale; a12 *= ascale; a22 *= ascale
    b11 = b11i; b12 = b12i; b22 = b22i
    bmin = rtmin * max(abs(b11), abs(b12), abs(b22), rtmin)
    abs(b11) < bmin && (b11 = copysign(bmin, b11))
    abs(b22) < bmin && (b22 = copysign(bmin, b22))
    bnorm = max(abs(b11), abs(b12) + abs(b22), safmin)
    bsize = max(abs(b11), abs(b22))
    bscale = ONE / bsize
    b11 *= bscale; b12 *= bscale; b22 *= bscale
    binv11 = ONE / b11; binv22 = ONE / b22
    s1 = a11 * binv11; s2 = a22 * binv22
    local as12, ss, abi22, pp, shift
    if abs(s1) <= abs(s2)
        as12 = a12 - s1 * b12; as22 = a22 - s1 * b22
        ss = a21 * (binv11 * binv22); abi22 = as22 * binv22 - ss * b12
        pp = HALF * abi22; shift = s1
    else
        as12 = a12 - s2 * b12; as11 = a11 - s2 * b11
        ss = a21 * (binv11 * binv22); abi22 = -ss * b12
        pp = HALF * (as11 * binv11 + abi22); shift = s2
    end
    qq = ss * as12
    local discr, rr
    if abs(pp * rtmin) >= ONE
        discr = (rtmin * pp)^2 + qq * safmin; rr = sqrt(abs(discr)) * rtmax
    else
        if pp^2 + abs(qq) <= safmin
            discr = (rtmax * pp)^2 + qq * safmax; rr = sqrt(abs(discr)) * rtmin
        else
            discr = pp^2 + qq; rr = sqrt(abs(discr))
        end
    end
    local wr1, wr2, wi
    if discr >= ZERO || rr == ZERO
        sum = pp + copysign(rr, pp); diff = pp - copysign(rr, pp)
        wbig = shift + sum; wsmall = shift + diff
        if HALF * abs(wbig) > max(abs(wsmall), safmin)
            wdet = (a11 * a22 - a12 * a21) * (binv11 * binv22); wsmall = wdet / wbig
        end
        if pp > abi22
            wr1 = min(wbig, wsmall); wr2 = max(wbig, wsmall)
        else
            wr1 = max(wbig, wsmall); wr2 = min(wbig, wsmall)
        end
        wi = ZERO
    else
        wr1 = shift + pp; wr2 = wr1; wi = rr
    end
    # Further scaling (SCALE1, SCALE2)
    c1 = bsize * (safmin * max(ONE, ascale))
    c2 = safmin * max(ONE, bnorm)
    c3 = bsize * safmin
    c4 = (ascale <= ONE && bsize <= ONE) ? min(ONE, (ascale / safmin) * bsize) : ONE
    c5 = (ascale <= ONE || bsize <= ONE) ? min(ONE, ascale * bsize) : ONE
    scale1 = ZERO; scale2 = ZERO
    wabs = abs(wr1) + abs(wi)
    wsize = max(safmin, c1, FUZZY1 * (wabs * c2 + c3), min(c4, HALF * max(wabs, c5)))
    if wsize != ONE
        wscale = ONE / wsize
        if wsize > ONE
            scale1 = (max(ascale, bsize) * wscale) * min(ascale, bsize)
        else
            scale1 = (min(ascale, bsize) * wscale) * max(ascale, bsize)
        end
        wr1 *= wscale
        if wi != ZERO
            wi *= wscale; wr2 = wr1; scale2 = scale1
        end
    else
        scale1 = ascale * bsize; scale2 = scale1
    end
    if wi == ZERO
        wsize = max(safmin, c1, FUZZY1 * (abs(wr2) * c2 + c3), min(c4, HALF * max(abs(wr2), c5)))
        if wsize != ONE
            wscale = ONE / wsize
            if wsize > ONE
                scale2 = (max(ascale, bsize) * wscale) * min(ascale, bsize)
            else
                scale2 = (min(ascale, bsize) * wscale) * max(ascale, bsize)
            end
            wr2 *= wscale
        else
            scale2 = ascale * bsize
        end
    end
    return scale1, scale2, wr1, wr2, wi
end

# ── DLASV2 (SVD of a real 2×2 upper-triangular [f g; 0 h]), Reference-LAPACK dlasv2.f ──────────────────
# Returns (ssmin, ssmax, snr, csr, snl, csl).
function _lasv2(f::R, g::R, h::R) where {R<:Real}
    ZERO = zero(R); ONE = one(R); TWO = R(2); FOUR = R(4); HALF = R(0.5)
    ft = f; fa = abs(ft); ht = h; ha = abs(h)
    pmax = 1; swap = ha > fa
    if swap
        pmax = 3
        ft, ht = ht, ft; fa, ha = ha, fa
    end
    gt = g; ga = abs(gt)
    local ssmin, ssmax, clt, crt, slt, srt
    if ga == ZERO
        ssmin = ha; ssmax = fa; clt = ONE; crt = ONE; slt = ZERO; srt = ZERO
    else
        gasmal = true
        if ga > fa
            pmax = 2
            if (fa / ga) < eps(R) / 2
                gasmal = false
                ssmax = ga
                ssmin = ha > ONE ? fa / (ga / ha) : (fa / ga) * ha
                clt = ONE; slt = ht / gt; srt = ONE; crt = ft / gt
            end
        end
        if gasmal
            d = fa - ha
            l = d == fa ? ONE : d / fa
            m = gt / ft
            t = TWO - l
            mm = m * m; tt = t * t
            s = sqrt(tt + mm)
            r = l == ZERO ? abs(m) : sqrt(l * l + mm)
            a = HALF * (s + r)
            ssmin = ha / a; ssmax = fa * a
            if mm == ZERO
                if l == ZERO
                    t = copysign(TWO, ft) * copysign(ONE, gt)
                else
                    t = gt / copysign(d, ft) + m / t
                end
            else
                t = (m / (s + t) + m / (r + l)) * (ONE + a)
            end
            l = sqrt(t * t + FOUR)
            crt = TWO / l; srt = t / l
            clt = (crt + srt * m) / a
            slt = (ht / ft) * srt / a
        end
    end
    local csl, snl, csr, snr
    if swap
        csl = srt; snl = crt; csr = slt; snr = clt
    else
        csl = clt; snl = slt; csr = crt; snr = srt
    end
    local tsign
    if pmax == 1
        tsign = copysign(ONE, csr) * copysign(ONE, csl) * copysign(ONE, f)
    elseif pmax == 2
        tsign = copysign(ONE, snr) * copysign(ONE, csl) * copysign(ONE, g)
    else
        tsign = copysign(ONE, snr) * copysign(ONE, snl) * copysign(ONE, h)
    end
    ssmax = copysign(ssmax, tsign)
    ssmin = copysign(ssmin, tsign * copysign(ONE, f) * copysign(ONE, h))
    return ssmin, ssmax, snr, csr, snl, csl
end

# ── DLARFG on a short vector (real) — mirrors hseqr.jl `_hqr_larfg!` ────────────────────────────────────
@inline function _qz_larfg!(v::AbstractVector{R}, nr::Int) where {R<:Real}
    nr <= 1 && return zero(R)
    @inbounds begin
        α = v[1]; ss = zero(R)
        for i in 2:nr; ss = muladd(v[i], v[i], ss); end
        xnorm = sqrt(ss)
        xnorm == zero(R) && return zero(R)
        β = -copysign(hypot(α, xnorm), α)
        safmn = _qz_safmin(R) / (eps(R) / 2)
        knt = 0
        if abs(β) < safmn
            rsafmn = one(R) / safmn
            while true
                knt += 1
                for i in 2:nr; v[i] *= rsafmn; end
                β *= rsafmn; α *= rsafmn
                (abs(β) < safmn && knt < 20) || break
            end
            ss = zero(R)
            for i in 2:nr; ss = muladd(v[i], v[i], ss); end
            xnorm = sqrt(ss); β = -copysign(hypot(α, xnorm), α)
        end
        τ = (β - α) / β
        s = one(R) / (α - β)
        for i in 2:nr; v[i] *= s; end
        for _ in 1:knt; β *= safmn; end
        v[1] = β
        return τ
    end
end

# Frobenius norm of the ilo:ihi diagonal block (DLANHS 'F' role — subdiagonal included for H).
@inline function _qz_fnorm(A::AbstractMatrix{T}, ilo::Int, ihi::Int) where {T}
    R = real(T); s = zero(R)
    @inbounds for j in ilo:ihi, i in ilo:ihi
        s += abs2(A[i, j])
    end
    return sqrt(s)
end

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
#  hgeqz (REAL) — QZ iteration → real generalized Schur form (dhgeqz)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
function _hgeqz!(wantt::Bool, wantq::Bool, wantz::Bool, H::AbstractMatrix{R}, T::AbstractMatrix{R},
        ilo::Int, ihi::Int, alphar::AbstractVector{R}, alphai::AbstractVector{R}, beta::AbstractVector{R},
        Q::AbstractMatrix{R}, Z::AbstractMatrix{R}) where {R<:Real}
    n = size(H, 1)
    ONE = one(R); ZERO = zero(R); HALF = R(0.5); SAFETY = R(100)
    ilschr = wantt; ilq = wantq; ilz = wantz
    safmin = _qz_safmin(R); safmax = ONE / safmin; ulp = eps(R)
    anorm = _qz_fnorm(H, ilo, ihi); bnorm = _qz_fnorm(T, ilo, ihi)
    atol = max(safmin, ulp * anorm); btol = max(safmin, ulp * bnorm)
    ascale = ONE / max(safmin, anorm); bscale = ONE / max(safmin, bnorm)
    info = 0
    v = Vector{R}(undef, 3)
    # Set eigenvalues ihi+1:n
    @inbounds for j in ihi+1:n
        if T[j, j] < ZERO
            if ilschr
                for jr in 1:j; H[jr, j] = -H[jr, j]; T[jr, j] = -T[jr, j]; end
            else
                H[j, j] = -H[j, j]; T[j, j] = -T[j, j]
            end
            if ilz; for jr in 1:n; Z[jr, j] = -Z[jr, j]; end; end
        end
        alphar[j] = H[j, j]; alphai[j] = ZERO; beta[j] = T[j, j]
    end
    if ihi >= ilo
        ilast = ihi
        if ilschr; ifrstm = 1; ilastm = n; else; ifrstm = ilo; ilastm = ihi; end
        iiter = 0; eshift = ZERO; maxit = 30 * (ihi - ilo + 1)
        converged = false
        jiter = 0
        @inbounds while jiter < maxit
            jiter += 1
            # ═══ Split the matrix if possible ═══
            route = 0            # 70, 80, 110  (0 = still searching)
            ifirst = 0
            if ilast == ilo
                route = 80
            elseif abs(H[ilast, ilast-1]) <= max(safmin, ulp * (abs(H[ilast, ilast]) + abs(H[ilast-1, ilast-1])))
                H[ilast, ilast-1] = ZERO; route = 80
            end
            if route == 0 && abs(T[ilast, ilast]) <= btol
                T[ilast, ilast] = ZERO; route = 70
            end
            if route == 0
                dropped = true
                for j in ilast-1:-1:ilo
                    if j == ilo
                        ilazro = true
                    elseif abs(H[j, j-1]) <= max(safmin, ulp * (abs(H[j, j]) + abs(H[j-1, j-1])))
                        H[j, j-1] = ZERO; ilazro = true
                    else
                        ilazro = false
                    end
                    if abs(T[j, j]) < btol
                        T[j, j] = ZERO
                        ilazr2 = false
                        if !ilazro
                            temp = abs(H[j, j-1]); temp2 = abs(H[j, j]); tempr = max(temp, temp2)
                            if tempr < ONE && tempr != ZERO; temp /= tempr; temp2 /= tempr; end
                            if temp * (ascale * abs(H[j+1, j])) <= temp2 * (ascale * atol); ilazr2 = true; end
                        end
                        if ilazro || ilazr2
                            routed = false
                            for jch in j:ilast-1
                                c, s, r = _lartg(H[jch, jch], H[jch+1, jch])
                                H[jch, jch] = r; H[jch+1, jch] = ZERO
                                _grot_rows!(H, jch, jch+1, jch+1, ilastm, c, s)
                                _grot_rows!(T, jch, jch+1, jch+1, ilastm, c, s)
                                ilq && _grot_cols!(Q, jch, jch+1, 1, n, c, s)
                                ilazr2 && (H[jch, jch-1] = H[jch, jch-1] * c)
                                ilazr2 = false
                                if abs(T[jch+1, jch+1]) >= btol
                                    if jch + 1 >= ilast
                                        route = 80
                                    else
                                        ifirst = jch + 1; route = 110
                                    end
                                    routed = true; break
                                end
                                T[jch+1, jch+1] = ZERO
                            end
                            routed || (route = 70)
                        else
                            for jch in j:ilast-1
                                c, s, r = _lartg(T[jch, jch+1], T[jch+1, jch+1])
                                T[jch, jch+1] = r; T[jch+1, jch+1] = ZERO
                                jch < ilastm - 1 && _grot_rows!(T, jch, jch+1, jch+2, ilastm, c, s)
                                _grot_rows!(H, jch, jch+1, jch-1, ilastm, c, s)
                                ilq && _grot_cols!(Q, jch, jch+1, 1, n, c, s)
                                c, s, r = _lartg(H[jch+1, jch], H[jch+1, jch-1])
                                H[jch+1, jch] = r; H[jch+1, jch-1] = ZERO
                                _grot_cols!(H, jch, jch-1, ifrstm, jch, c, s)
                                _grot_cols!(T, jch, jch-1, ifrstm, jch-1, c, s)
                                ilz && _grot_cols!(Z, jch, jch-1, 1, n, c, s)
                            end
                            route = 70
                        end
                        dropped = false; break
                    elseif ilazro
                        ifirst = j; route = 110; dropped = false; break
                    end
                end
                if dropped
                    info = n + 1; converged = false; break
                end
            end

            # ═══ route 70: T(ilast,ilast)=0 — clear H(ilast,ilast-1) ═══
            if route == 70
                c, s, r = _lartg(H[ilast, ilast], H[ilast, ilast-1])
                H[ilast, ilast] = r; H[ilast, ilast-1] = ZERO
                _grot_cols!(H, ilast, ilast-1, ifrstm, ilast-1, c, s)
                _grot_cols!(T, ilast, ilast-1, ifrstm, ilast-1, c, s)
                ilz && _grot_cols!(Z, ilast, ilast-1, 1, n, c, s)
                route = 80
            end

            # ═══ route 80: standardize trailing 1×1, set α,β ═══
            if route == 80
                if T[ilast, ilast] < ZERO
                    if ilschr
                        for j in ifrstm:ilast; H[j, ilast] = -H[j, ilast]; T[j, ilast] = -T[j, ilast]; end
                    else
                        H[ilast, ilast] = -H[ilast, ilast]; T[ilast, ilast] = -T[ilast, ilast]
                    end
                    if ilz; for j in 1:n; Z[j, ilast] = -Z[j, ilast]; end; end
                end
                alphar[ilast] = H[ilast, ilast]; alphai[ilast] = ZERO; beta[ilast] = T[ilast, ilast]
                ilast -= 1
                if ilast < ilo; converged = true; break; end
                iiter = 0; eshift = ZERO
                if !ilschr
                    ilastm = ilast
                    ifrstm > ilast && (ifrstm = ilo)
                end
                continue     # GO TO 350
            end

            # ═══ route 110: QZ step ═══
            iiter += 1
            !ilschr && (ifrstm = ifirst)
            # ── Compute shifts ──
            usedouble = false
            local s1::R, wr::R
            if (iiter ÷ 10) * 10 == iiter
                # exceptional shift (single)
                if (R(maxit) * safmin) * abs(H[ilast, ilast-1]) < abs(T[ilast-1, ilast-1])
                    eshift = H[ilast, ilast-1] / T[ilast-1, ilast-1]
                else
                    eshift = eshift + ONE / (safmin * R(maxit))
                end
                s1 = ONE; wr = eshift
            else
                s1, s2, wr, wr2, wi = _lag2(H[ilast-1, ilast-1], H[ilast, ilast-1], H[ilast-1, ilast],
                    H[ilast, ilast], T[ilast-1, ilast-1], T[ilast-1, ilast], T[ilast, ilast], safmin * SAFETY)
                if abs((wr / s1) * T[ilast, ilast] - H[ilast, ilast]) >
                   abs((wr2 / s2) * T[ilast, ilast] - H[ilast, ilast])
                    wr, wr2 = wr2, wr; s1, s2 = s2, s1
                end
                wi != ZERO && (usedouble = true)
            end

            if !usedouble
                # ── Fiddle with shift to avoid overflow ──
                temp = min(ascale, ONE) * (HALF * safmax)
                scale = s1 > temp ? temp / s1 : ONE
                temp = min(bscale, ONE) * (HALF * safmax)
                abs(wr) > temp && (scale = min(scale, temp / abs(wr)))
                s1 = scale * s1; wr = scale * wr
                # ── Two consecutive small subdiagonals ──
                istart = ifirst
                for j in ilast-1:-1:ifirst+1
                    istart = j
                    temp = abs(s1 * H[j, j-1])
                    temp2 = abs(s1 * H[j, j] - wr * T[j, j])
                    tempr = max(temp, temp2)
                    if tempr < ONE && tempr != ZERO; temp /= tempr; temp2 /= tempr; end
                    if abs((ascale * H[j+1, j]) * temp) <= (ascale * atol) * temp2
                        break
                    end
                    istart = ifirst
                end
                # ── Single-shift QZ sweep ──
                temp = s1 * H[istart, istart] - wr * T[istart, istart]
                temp2 = s1 * H[istart+1, istart]
                c, s, tempr = _lartg(temp, temp2)
                for j in istart:ilast-1
                    if j > istart
                        c, s, r = _lartg(H[j, j-1], H[j+1, j-1])
                        H[j, j-1] = r; H[j+1, j-1] = ZERO
                    end
                    _grot_rows!(H, j, j+1, j, ilastm, c, s)
                    _grot_rows!(T, j, j+1, j, ilastm, c, s)
                    ilq && _grot_cols!(Q, j, j+1, 1, n, c, s)
                    c, s, r = _lartg(T[j+1, j+1], T[j+1, j])
                    T[j+1, j+1] = r; T[j+1, j] = ZERO
                    _grot_cols!(H, j+1, j, ifrstm, min(j+2, ilast), c, s)
                    _grot_cols!(T, j+1, j, ifrstm, j, c, s)
                    ilz && _grot_cols!(Z, j+1, j, 1, n, c, s)
                end
                continue    # GO TO 350
            end

            # ═══ route 200: Francis double-shift ═══
            if ifirst + 1 == ilast
                # 2×2 block with complex eigenvalues
                b22, b11, sr, cr, sl, cl = _lasv2(T[ilast-1, ilast-1], T[ilast-1, ilast], T[ilast, ilast])
                if b11 < ZERO
                    cr = -cr; sr = -sr; b11 = -b11; b22 = -b22
                end
                _grot_rows!(H, ilast-1, ilast, ilast-1, ilastm, cl, sl)
                _grot_cols!(H, ilast-1, ilast, ifrstm, ilast, cr, sr)
                ilast < ilastm && _grot_rows!(T, ilast-1, ilast, ilast+1, ilastm, cl, sl)
                ifrstm < ilast - 1 && _grot_cols!(T, ilast-1, ilast, ifrstm, ifirst, cr, sr)
                ilq && _grot_cols!(Q, ilast-1, ilast, 1, n, cl, sl)
                ilz && _grot_cols!(Z, ilast-1, ilast, 1, n, cr, sr)
                T[ilast-1, ilast-1] = b11; T[ilast-1, ilast] = ZERO
                T[ilast, ilast-1] = ZERO; T[ilast, ilast] = b22
                if b22 < ZERO
                    for j in ifrstm:ilast; H[j, ilast] = -H[j, ilast]; T[j, ilast] = -T[j, ilast]; end
                    if ilz; for j in 1:n; Z[j, ilast] = -Z[j, ilast]; end; end
                    b22 = -b22
                end
                # recompute shift
                s1, tmpS2, wr, tmp2, wi = _lag2(H[ilast-1, ilast-1], H[ilast, ilast-1], H[ilast-1, ilast],
                    H[ilast, ilast], T[ilast-1, ilast-1], T[ilast-1, ilast], T[ilast, ilast], safmin * SAFETY)
                if wi == ZERO
                    continue      # standardization perturbed shift onto real line → GO TO 350
                end
                s1inv = ONE / s1
                a11 = H[ilast-1, ilast-1]; a21 = H[ilast, ilast-1]
                a12 = H[ilast-1, ilast]; a22 = H[ilast, ilast]
                c11r = s1 * a11 - wr * b11; c11i = -wi * b11
                c12 = s1 * a12; c21 = s1 * a21
                c22r = s1 * a22 - wr * b22; c22i = -wi * b22
                local cz::R, szr::R, szi::R
                if abs(c11r) + abs(c11i) + abs(c12) > abs(c21) + abs(c22r) + abs(c22i)
                    t1 = _lapy3(c12, c11r, c11i)
                    cz = c12 / t1; szr = -c11r / t1; szi = -c11i / t1
                else
                    cz = _lapy2(c22r, c22i)
                    if cz <= safmin
                        cz = ZERO; szr = ONE; szi = ZERO
                    else
                        tempr = c22r / cz; tempi = c22i / cz
                        t1 = _lapy2(cz, c21)
                        cz = cz / t1; szr = -c21 * tempr / t1; szi = c21 * tempi / t1
                    end
                end
                an = abs(a11) + abs(a12) + abs(a21) + abs(a22)
                bn = abs(b11) + abs(b22)
                wabs = abs(wr) + abs(wi)
                local cq::R, sqr::R, sqi::R
                if s1 * an > wabs * bn
                    cq = cz * b11; sqr = szr * b22; sqi = -szi * b22
                else
                    a1r = cz * a11 + szr * a12; a1i = szi * a12
                    a2r = cz * a21 + szr * a22; a2i = szi * a22
                    cq = _lapy2(a1r, a1i)
                    if cq <= safmin
                        cq = ZERO; sqr = ONE; sqi = ZERO
                    else
                        tempr = a1r / cq; tempi = a1i / cq
                        sqr = tempr * a2r + tempi * a2i; sqi = tempi * a2r - tempr * a2i
                    end
                end
                t1 = _lapy3(cq, sqr, sqi)
                cq = cq / t1; sqr = sqr / t1; sqi = sqi / t1
                tempr = sqr * szr - sqi * szi; tempi = sqr * szi + sqi * szr
                b1r = cq * cz * b11 + tempr * b22; b1i = tempi * b22
                b1a = _lapy2(b1r, b1i)
                b2r = cq * cz * b22 + tempr * b11; b2i = -tempi * b11
                b2a = _lapy2(b2r, b2i)
                beta[ilast-1] = b1a; beta[ilast] = b2a
                alphar[ilast-1] = (wr * b1a) * s1inv; alphai[ilast-1] = (wi * b1a) * s1inv
                alphar[ilast] = (wr * b2a) * s1inv; alphai[ilast] = -(wi * b2a) * s1inv
                ilast = ifirst - 1
                if ilast < ilo; converged = true; break; end
                iiter = 0; eshift = ZERO
                if !ilschr
                    ilastm = ilast
                    ifrstm > ilast && (ifrstm = ilo)
                end
                continue    # GO TO 350
            else
                # Usual case: 3×3 or larger — Francis implicit double shift
                ad11 = (ascale * H[ilast-1, ilast-1]) / (bscale * T[ilast-1, ilast-1])
                ad21 = (ascale * H[ilast, ilast-1]) / (bscale * T[ilast-1, ilast-1])
                ad12 = (ascale * H[ilast-1, ilast]) / (bscale * T[ilast, ilast])
                ad22 = (ascale * H[ilast, ilast]) / (bscale * T[ilast, ilast])
                u12 = T[ilast-1, ilast] / T[ilast, ilast]
                ad11l = (ascale * H[ifirst, ifirst]) / (bscale * T[ifirst, ifirst])
                ad21l = (ascale * H[ifirst+1, ifirst]) / (bscale * T[ifirst, ifirst])
                ad12l = (ascale * H[ifirst, ifirst+1]) / (bscale * T[ifirst+1, ifirst+1])
                ad22l = (ascale * H[ifirst+1, ifirst+1]) / (bscale * T[ifirst+1, ifirst+1])
                ad32l = (ascale * H[ifirst+2, ifirst+1]) / (bscale * T[ifirst+1, ifirst+1])
                u12l = T[ifirst, ifirst+1] / T[ifirst+1, ifirst+1]
                v[1] = (ad11 - ad11l) * (ad22 - ad11l) - ad12 * ad21 + ad21 * u12 * ad11l +
                       (ad12l - ad11l * u12l) * ad21l
                v[2] = ((ad22l - ad11l) - ad21l * u12l - (ad11 - ad11l) - (ad22 - ad11l) + ad21 * u12) * ad21l
                v[3] = ad32l * ad21l
                istart = ifirst
                tau = _qz_larfg!(v, 3); v[1] = ONE
                # Sweep
                for j in istart:ilast-2
                    if j > istart
                        v[1] = H[j, j-1]; v[2] = H[j+1, j-1]; v[3] = H[j+2, j-1]
                        tau = _qz_larfg!(v, 3)
                        H[j, j-1] = v[1]; v[1] = ONE
                        H[j+1, j-1] = ZERO; H[j+2, j-1] = ZERO
                    end
                    v2 = v[2]; v3 = v[3]
                    t2 = tau * v2; t3 = tau * v3
                    # Apply 3×3 Householder from the LEFT to H, T
                    for jc in j:ilastm
                        temp = H[j, jc] + v2 * H[j+1, jc] + v3 * H[j+2, jc]
                        H[j, jc] -= temp * tau; H[j+1, jc] -= temp * t2; H[j+2, jc] -= temp * t3
                        temp2 = T[j, jc] + v2 * T[j+1, jc] + v3 * T[j+2, jc]
                        T[j, jc] -= temp2 * tau; T[j+1, jc] -= temp2 * t2; T[j+2, jc] -= temp2 * t3
                    end
                    if ilq
                        for jr in 1:n
                            temp = Q[jr, j] + v2 * Q[jr, j+1] + v3 * Q[jr, j+2]
                            Q[jr, j] -= temp * tau; Q[jr, j+1] -= temp * t2; Q[jr, j+2] -= temp * t3
                        end
                    end
                    # Zero j-th column of T (DLAGBC): swap rows to pivot, LU-factor, solve
                    ilpivt = false
                    temp = max(abs(T[j+1, j+1]), abs(T[j+1, j+2]))
                    temp2 = max(abs(T[j+2, j+1]), abs(T[j+2, j+2]))
                    local u1::R, u2::R, scale::R, w11::R, w12::R, w21::R, w22::R
                    if max(temp, temp2) < safmin
                        scale = ZERO; u1 = ONE; u2 = ZERO
                    else
                        if temp >= temp2
                            w11 = T[j+1, j+1]; w21 = T[j+2, j+1]; w12 = T[j+1, j+2]; w22 = T[j+2, j+2]
                            u1 = T[j+1, j]; u2 = T[j+2, j]
                        else
                            w21 = T[j+1, j+1]; w11 = T[j+2, j+1]; w22 = T[j+1, j+2]; w12 = T[j+2, j+2]
                            u2 = T[j+1, j]; u1 = T[j+2, j]
                        end
                        if abs(w12) > abs(w11)
                            ilpivt = true
                            w12, w11 = w11, w12; w22, w21 = w21, w22
                        end
                        temp = w21 / w11
                        u2 = u2 - temp * u1; w22 = w22 - temp * w12; w21 = ZERO
                        scale = ONE
                        if abs(w22) < safmin
                            scale = ZERO; u2 = ONE; u1 = -w12 / w11
                        else
                            abs(w22) < abs(u2) && (scale = abs(w22 / u2))
                            abs(w11) < abs(u1) && (scale = min(scale, abs(w11 / u1)))
                            u2 = (scale * u2) / w22
                            u1 = (scale * u1 - w12 * u2) / w11
                        end
                    end
                    if ilpivt
                        u1, u2 = u2, u1
                    end
                    t1 = sqrt(scale^2 + u1^2 + u2^2)
                    tau = ONE + scale / t1
                    vs = -ONE / (scale + t1)
                    v[1] = ONE; v[2] = vs * u1; v[3] = vs * u2
                    v2 = v[2]; v3 = v[3]; t2 = tau * v2; t3 = tau * v3
                    # Apply from the RIGHT
                    for jr in ifrstm:min(j+3, ilast)
                        temp = H[jr, j] + v2 * H[jr, j+1] + v3 * H[jr, j+2]
                        H[jr, j] -= temp * tau; H[jr, j+1] -= temp * t2; H[jr, j+2] -= temp * t3
                    end
                    for jr in ifrstm:j+2
                        temp = T[jr, j] + v2 * T[jr, j+1] + v3 * T[jr, j+2]
                        T[jr, j] -= temp * tau; T[jr, j+1] -= temp * t2; T[jr, j+2] -= temp * t3
                    end
                    if ilz
                        for jr in 1:n
                            temp = Z[jr, j] + v2 * Z[jr, j+1] + v3 * Z[jr, j+2]
                            Z[jr, j] -= temp * tau; Z[jr, j+1] -= temp * t2; Z[jr, j+2] -= temp * t3
                        end
                    end
                    T[j+1, j] = ZERO; T[j+2, j] = ZERO
                end
                # Last elements: use Givens rotations
                jl = ilast - 1
                c, s, r = _lartg(H[jl, jl-1], H[jl+1, jl-1])
                H[jl, jl-1] = r; H[jl+1, jl-1] = ZERO
                _grot_rows!(H, jl, jl+1, jl, ilastm, c, s)
                _grot_rows!(T, jl, jl+1, jl, ilastm, c, s)
                ilq && _grot_cols!(Q, jl, jl+1, 1, n, c, s)
                c, s, r = _lartg(T[jl+1, jl+1], T[jl+1, jl])
                T[jl+1, jl+1] = r; T[jl+1, jl] = ZERO
                _grot_cols!(H, jl+1, jl, ifrstm, ilast, c, s)
                _grot_cols!(T, jl+1, jl, ifrstm, ilast-1, c, s)
                ilz && _grot_cols!(Z, jl+1, jl, 1, n, c, s)
                continue    # GO TO 350
            end
        end
        if !converged && info == 0
            info = ilast
        end
    end
    # Set eigenvalues 1:ilo-1
    @inbounds for j in 1:ilo-1
        if T[j, j] < ZERO
            if ilschr
                for jr in 1:j; H[jr, j] = -H[jr, j]; T[jr, j] = -T[jr, j]; end
            else
                H[j, j] = -H[j, j]; T[j, j] = -T[j, j]
            end
            if ilz; for jr in 1:n; Z[jr, j] = -Z[jr, j]; end; end
        end
        alphar[j] = H[j, j]; alphai[j] = ZERO; beta[j] = T[j, j]
    end
    return info
end

"""
    hgeqz!(job, compq, compz, H, T, alpha, beta, Q, Z; ilo=1, ihi=size(H,1)) -> info

QZ iteration on a generalized Hessenberg-triangular pair `(H,T)` (`H` upper Hessenberg, `T` upper
triangular) → generalized Schur form (LAPACK dhgeqz/zhgeqz). `job='E'` eigenvalues only, `'S'` full
Schur (`H→S`, `T→P`). `compq`/`compz` ∈ `'N'`/`'I'`/`'V'` accumulate `Q`/`Z`. Eigenvalues are
`α[i]/β[i]`. Returns `info` (0 = success; `1≤info≤n` = non-convergence).

REAL `T<:Real`: `alpha::Vector{<:Complex}` (real+imag parts, conjugate pairs), `beta::Vector{<:Real}`.
COMPLEX `T<:Complex`: `alpha,beta::Vector{T}`.
"""
function hgeqz!(job::AbstractChar, compq::AbstractChar, compz::AbstractChar, H::AbstractMatrix{R},
        T::AbstractMatrix{R}, alpha::AbstractVector{<:Complex}, beta::AbstractVector{R},
        Q::AbstractMatrix{R}, Z::AbstractMatrix{R}; ilo::Integer = 1, ihi::Integer = size(H, 1)) where {R<:Real}
    n = size(H, 1)
    wantt = job === 'S'; wantq = compq !== 'N'; wantz = compz !== 'N'
    _qz_init_qz!(compq, Q, n); _qz_init_qz!(compz, Z, n)
    alphar = Vector{R}(undef, n); alphai = Vector{R}(undef, n)
    info = _hgeqz!(wantt, wantq, wantz, H, T, Int(ilo), Int(ihi), alphar, alphai, beta, Q, Z)
    @inbounds for i in 1:n; alpha[i] = Complex(alphar[i], alphai[i]); end
    return info
end

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
#  hgeqz (COMPLEX) — QZ iteration → complex generalized Schur (triangular) form (zhgeqz)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
function _zhgeqz!(wantt::Bool, wantq::Bool, wantz::Bool, H::AbstractMatrix{C}, T::AbstractMatrix{C},
        ilo::Int, ihi::Int, alpha::AbstractVector{C}, beta::AbstractVector{C},
        Q::AbstractMatrix{C}, Z::AbstractMatrix{C}) where {C<:Complex}
    R = real(C)
    n = size(H, 1)
    ONE = one(R); ZERO = zero(R); HALF = R(0.5); CZERO = zero(C)
    cabs1(z) = abs(real(z)) + abs(imag(z))
    ilschr = wantt; ilq = wantq; ilz = wantz
    safmin = _qz_safmin(R); ulp = eps(R)
    anorm = _qz_fnorm(H, ilo, ihi); bnorm = _qz_fnorm(T, ilo, ihi)
    atol = max(safmin, ulp * anorm); btol = max(safmin, ulp * bnorm)
    ascale = ONE / max(safmin, anorm); bscale = ONE / max(safmin, bnorm)
    info = 0
    # Set eigenvalues ihi+1:n
    @inbounds for j in ihi+1:n
        absb = abs(T[j, j])
        if absb > safmin
            signbc = conj(T[j, j] / absb); T[j, j] = complex(absb)
            if ilschr
                for i in 1:j-1; T[i, j] *= signbc; end
                for i in 1:j;   H[i, j] *= signbc; end
            else
                H[j, j] *= signbc
            end
            ilz && (for i in 1:n; Z[i, j] *= signbc; end)
        else
            T[j, j] = CZERO
        end
        alpha[j] = H[j, j]; beta[j] = T[j, j]
    end
    if ihi >= ilo
        ilast = ihi
        if ilschr; ifrstm = 1; ilastm = n; else; ifrstm = ilo; ilastm = ihi; end
        iiter = 0; eshift = CZERO; maxit = 30 * (ihi - ilo + 1)
        converged = false
        jiter = 0
        @inbounds while jiter < maxit
            jiter += 1
            # ═══ Split ═══
            route = 0; ifirst = 0
            if ilast == ilo
                route = 60
            elseif cabs1(H[ilast, ilast-1]) <= max(safmin, ulp * (cabs1(H[ilast, ilast]) + cabs1(H[ilast-1, ilast-1])))
                H[ilast, ilast-1] = CZERO; route = 60
            end
            if route == 0 && abs(T[ilast, ilast]) <= btol
                T[ilast, ilast] = CZERO; route = 50
            end
            if route == 0
                dropped = true
                for j in ilast-1:-1:ilo
                    if j == ilo
                        ilazro = true
                    elseif cabs1(H[j, j-1]) <= max(safmin, ulp * (cabs1(H[j, j]) + cabs1(H[j-1, j-1])))
                        H[j, j-1] = CZERO; ilazro = true
                    else
                        ilazro = false
                    end
                    if abs(T[j, j]) < btol
                        T[j, j] = CZERO
                        ilazr2 = false
                        if !ilazro
                            if cabs1(H[j, j-1]) * (ascale * cabs1(H[j+1, j])) <= cabs1(H[j, j]) * (ascale * atol)
                                ilazr2 = true
                            end
                        end
                        if ilazro || ilazr2
                            routed = false
                            for jch in j:ilast-1
                                c, s, r = _zlartg(H[jch, jch], H[jch+1, jch])
                                H[jch, jch] = r; H[jch+1, jch] = CZERO
                                _zrot_rows!(H, jch, jch+1, jch+1, ilastm, c, s)
                                _zrot_rows!(T, jch, jch+1, jch+1, ilastm, c, s)
                                ilq && _zrot_cols!(Q, jch, jch+1, 1, n, c, conj(s))
                                ilazr2 && (H[jch, jch-1] = H[jch, jch-1] * c)
                                ilazr2 = false
                                if cabs1(T[jch+1, jch+1]) >= btol
                                    if jch + 1 >= ilast
                                        route = 60
                                    else
                                        ifirst = jch + 1; route = 70
                                    end
                                    routed = true; break
                                end
                                T[jch+1, jch+1] = CZERO
                            end
                            routed || (route = 50)
                        else
                            for jch in j:ilast-1
                                c, s, r = _zlartg(T[jch, jch+1], T[jch+1, jch+1])
                                T[jch, jch+1] = r; T[jch+1, jch+1] = CZERO
                                jch < ilastm - 1 && _zrot_rows!(T, jch, jch+1, jch+2, ilastm, c, s)
                                _zrot_rows!(H, jch, jch+1, jch-1, ilastm, c, s)
                                ilq && _zrot_cols!(Q, jch, jch+1, 1, n, c, conj(s))
                                c, s, r = _zlartg(H[jch+1, jch], H[jch+1, jch-1])
                                H[jch+1, jch] = r; H[jch+1, jch-1] = CZERO
                                _zrot_cols!(H, jch, jch-1, ifrstm, jch, c, s)
                                _zrot_cols!(T, jch, jch-1, ifrstm, jch-1, c, s)
                                ilz && _zrot_cols!(Z, jch, jch-1, 1, n, c, s)
                            end
                            route = 50
                        end
                        dropped = false; break
                    elseif ilazro
                        ifirst = j; route = 70; dropped = false; break
                    end
                end
                if dropped
                    info = 2 * n + 1; converged = false; break
                end
            end

            # ═══ route 50: clear H(ilast,ilast-1) ═══
            if route == 50
                c, s, r = _zlartg(H[ilast, ilast], H[ilast, ilast-1])
                H[ilast, ilast] = r; H[ilast, ilast-1] = CZERO
                _zrot_cols!(H, ilast, ilast-1, ifrstm, ilast-1, c, s)
                _zrot_cols!(T, ilast, ilast-1, ifrstm, ilast-1, c, s)
                ilz && _zrot_cols!(Z, ilast, ilast-1, 1, n, c, s)
                route = 60
            end

            # ═══ route 60: standardize B, set α,β ═══
            if route == 60
                absb = abs(T[ilast, ilast])
                if absb > safmin
                    signbc = conj(T[ilast, ilast] / absb); T[ilast, ilast] = complex(absb)
                    if ilschr
                        for i in ifrstm:ilast-1; T[i, ilast] *= signbc; end
                        for i in ifrstm:ilast;   H[i, ilast] *= signbc; end
                    else
                        H[ilast, ilast] *= signbc
                    end
                    ilz && (for i in 1:n; Z[i, ilast] *= signbc; end)
                else
                    T[ilast, ilast] = CZERO
                end
                alpha[ilast] = H[ilast, ilast]; beta[ilast] = T[ilast, ilast]
                ilast -= 1
                if ilast < ilo; converged = true; break; end
                iiter = 0; eshift = CZERO
                if !ilschr
                    ilastm = ilast
                    ifrstm > ilast && (ifrstm = ilo)
                end
                continue
            end

            # ═══ route 70: QZ step ═══
            iiter += 1
            !ilschr && (ifrstm = ifirst)
            local shift::C
            if (iiter ÷ 10) * 10 != iiter
                # Wilkinson complex shift
                u12 = (bscale * T[ilast-1, ilast]) / (bscale * T[ilast, ilast])
                ad11 = (ascale * H[ilast-1, ilast-1]) / (bscale * T[ilast-1, ilast-1])
                ad21 = (ascale * H[ilast, ilast-1]) / (bscale * T[ilast-1, ilast-1])
                ad12 = (ascale * H[ilast-1, ilast]) / (bscale * T[ilast, ilast])
                ad22 = (ascale * H[ilast, ilast]) / (bscale * T[ilast, ilast])
                abi22 = ad22 - u12 * ad21
                abi12 = ad12 - u12 * ad11
                shift = abi22
                ctemp = sqrt(abi12) * sqrt(ad21)
                temp = cabs1(ctemp)
                if ctemp != CZERO
                    x = HALF * (ad11 - shift)
                    temp2 = cabs1(x)
                    temp = max(temp, cabs1(x))
                    y = temp * sqrt((x / temp)^2 + (ctemp / temp)^2)
                    if temp2 > ZERO
                        if real(x / temp2) * real(y) + imag(x / temp2) * imag(y) < ZERO
                            y = -y
                        end
                    end
                    shift = shift - ctemp * _zladiv(ctemp, (x + y))
                end
            else
                # Exceptional shift
                if (iiter ÷ 20) * 20 == iiter && bscale * cabs1(T[ilast, ilast]) > safmin
                    eshift = eshift + (ascale * H[ilast, ilast]) / (bscale * T[ilast, ilast])
                else
                    eshift = eshift + (ascale * H[ilast, ilast-1]) / (bscale * T[ilast-1, ilast-1])
                end
                shift = eshift
            end
            # Two consecutive small subdiagonals
            istart = ifirst
            ctemp = ascale * H[ifirst, ifirst] - shift * (bscale * T[ifirst, ifirst])
            found = false
            for j in ilast-1:-1:ifirst+1
                istart = j
                ctemp = ascale * H[j, j] - shift * (bscale * T[j, j])
                temp = cabs1(ctemp)
                temp2 = ascale * cabs1(H[j+1, j])
                tempr = max(temp, temp2)
                if tempr < ONE && tempr != ZERO; temp /= tempr; temp2 /= tempr; end
                if cabs1(H[j, j-1]) * temp2 <= temp * atol
                    found = true; break
                end
            end
            if !found
                istart = ifirst
                ctemp = ascale * H[ifirst, ifirst] - shift * (bscale * T[ifirst, ifirst])
            end
            # Implicit single-shift QZ sweep
            ctemp2 = complex(ascale) * H[istart+1, istart]
            c, s, ctemp3 = _zlartg(ctemp, ctemp2)
            for j in istart:ilast-1
                if j > istart
                    c, s, r = _zlartg(H[j, j-1], H[j+1, j-1])
                    H[j, j-1] = r; H[j+1, j-1] = CZERO
                end
                _zrot_rows!(H, j, j+1, j, ilastm, c, s)
                _zrot_rows!(T, j, j+1, j, ilastm, c, s)
                ilq && _zrot_cols!(Q, j, j+1, 1, n, c, conj(s))
                c, s, r = _zlartg(T[j+1, j+1], T[j+1, j])
                T[j+1, j+1] = r; T[j+1, j] = CZERO
                _zrot_cols!(H, j+1, j, ifrstm, min(j+2, ilast), c, s)
                _zrot_cols!(T, j+1, j, ifrstm, j, c, s)
                ilz && _zrot_cols!(Z, j+1, j, 1, n, c, s)
            end
            continue
        end
        if !converged && info == 0
            info = ilast
        end
    end
    # Set eigenvalues 1:ilo-1
    @inbounds for j in 1:ilo-1
        absb = abs(T[j, j])
        if absb > safmin
            signbc = conj(T[j, j] / absb); T[j, j] = complex(absb)
            if ilschr
                for i in 1:j-1; T[i, j] *= signbc; end
                for i in 1:j;   H[i, j] *= signbc; end
            else
                H[j, j] *= signbc
            end
            ilz && (for i in 1:n; Z[i, j] *= signbc; end)
        else
            T[j, j] = CZERO
        end
        alpha[j] = H[j, j]; beta[j] = T[j, j]
    end
    return info
end

function hgeqz!(job::AbstractChar, compq::AbstractChar, compz::AbstractChar, H::AbstractMatrix{C},
        T::AbstractMatrix{C}, alpha::AbstractVector{C}, beta::AbstractVector{C},
        Q::AbstractMatrix{C}, Z::AbstractMatrix{C}; ilo::Integer = 1, ihi::Integer = size(H, 1)) where {C<:Complex}
    n = size(H, 1)
    wantt = job === 'S'; wantq = compq !== 'N'; wantz = compz !== 'N'
    _qz_init_qz!(compq, Q, n); _qz_init_qz!(compz, Z, n)
    return _zhgeqz!(wantt, wantq, wantz, H, T, Int(ilo), Int(ihi), alpha, beta, Q, Z)
end
