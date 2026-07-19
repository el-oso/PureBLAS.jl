# LAPACK nonsymmetric-eigen SCHUR kernel (second half of the general eigensolver `eigen(A)`):
#   hseqr — Schur decomposition of an upper-Hessenberg matrix via the Francis double-shift QR.
# This is the CORRECTNESS-FIRST *unblocked* path == Reference-LAPACK dlahqr (real) / zlahqr (complex).
# It reduces upper-Hessenberg H to (quasi-)upper-triangular Schur form T and accumulates the Schur
# vectors into Z (H = Z·T·Zᴴ). The blocked multishift + aggressive-early-deflation `dlaqr0` driver is
# a PERF follow-up, explicitly out of scope here.
#
# REAL path (T<:Real): real Schur form — 1×1 diagonal blocks for real eigenvalues, standardized 2×2
# blocks for complex-conjugate pairs (dlanv2). COMPLEX path (T<:Complex): fully upper-triangular T,
# diagonal = eigenvalues. Both copy Reference-LAPACK verbatim: the Ahues–Tisseur small-subdiagonal
# deflation test, the Wilkinson double/single shift from the trailing 2×2, the exceptional shift after
# every KEXSH=10 non-converged iterations (magic 0.75 / −0.4375), and the 30·max(10,nh) iteration cap.
#
# Generic over T (s/d/c/z); scalar loops (dlahqr is BLAS-1.5/column — the SIMD lever lives in the
# blocked follow-up). Self-contained: local dlarfg/dlanv2 ports so the file has no cross-module coupling.

# DLAMCH('S') — safe minimum, adjusted so 1/safmin does not overflow (mirrors eigen.jl `_dlamch_safmin`).
@inline function _hqr_safmin(::Type{R}) where {R<:Real}
    sfmin = floatmin(R)
    small = one(R) / floatmax(R)
    small >= sfmin && (sfmin = small * (one(R) + eps(R)))
    return sfmin
end

# ── DLARFG on a short (nr = 2 or 3) vector, REAL — the bulge-chase reflector generator ────────────────
# Mirrors Reference-LAPACK DLARFG including the KNT rescale for a below-safmin β. On return v[1]=β,
# v[2:nr] hold the essential reflector, and τ is returned (H = I − τ·v·vᵀ, v[1]≡1).
@inline function _hqr_larfg!(v::AbstractVector{R}, nr::Int) where {R<:Real}
    nr <= 1 && return zero(R)
    @inbounds begin
        α = v[1]
        ss = zero(R)
        for i in 2:nr; ss = muladd(v[i], v[i], ss); end
        xnorm = sqrt(ss)
        if xnorm == zero(R)
            return zero(R)                       # β = α, τ = 0, v unchanged
        end
        β = -copysign(hypot(α, xnorm), α)
        safmn = _hqr_safmin(R) / (eps(R) / 2)     # DLAMCH('S')/DLAMCH('E')
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
            xnorm = sqrt(ss)
            β = -copysign(hypot(α, xnorm), α)
        end
        τ = (β - α) / β
        s = one(R) / (α - β)
        for i in 2:nr; v[i] *= s; end
        for _ in 1:knt; β *= safmn; end
        v[1] = β
        return τ
    end
end

# ── DLARFG on a 2-vector, COMPLEX (scalar form) — zlahqr single-shift reflector ───────────────────────
# Returns (β::real, τ::Complex, v2::Complex essential). β is real by the zlarfg phase convention.
@inline function _zlarfg2(α::T, x2::T) where {T<:Complex}
    R = real(T)
    αr = real(α); αi = imag(α)
    xnorm = abs(x2)
    if xnorm == zero(R) && αi == zero(R)
        return αr, zero(T), zero(T)             # τ = 0, β = αr
    end
    β = -copysign(sqrt(αr*αr + αi*αi + xnorm*xnorm), αr)
    safmn = _hqr_safmin(R) / (eps(R) / 2)
    knt = 0
    if abs(β) < safmn
        rsafmn = one(R) / safmn
        while true
            knt += 1
            x2 *= rsafmn
            β *= rsafmn; αr *= rsafmn; αi *= rsafmn
            (abs(β) < safmn && knt < 20) || break
        end
        xnorm = abs(x2)
        β = -copysign(sqrt(αr*αr + αi*αi + xnorm*xnorm), αr)
    end
    τ = Complex((β - αr) / β, -αi / β)
    v2 = x2 / Complex(αr - β, αi)               # x2 * 1/(α−β)  with the rescaled α
    for _ in 1:knt; β *= safmn; end
    return β, τ, v2
end

# ── DLANV2 (Reference-LAPACK verbatim) — standardize a real 2×2 block [a b; c d] to Schur form ────────
# Returns (a,b,c,d, rt1r,rt1i,rt2r,rt2i, cs,sn): the standardized block, its eigenvalue pair, and the
# rotation. If eigenvalues are complex the block is left in the standard [a b; c d] with a=d and b,c of
# opposite sign (b·c < 0). This is the top bug locus — transcribed line-for-line from dlanv2.f.
function _dlanv2(a::R, b::R, c::R, d::R) where {R<:Real}
    Z0 = zero(R); ONE = one(R); HALF = R(0.5); TWO = R(2); MULTPL = R(4)
    eps_p = eps(R)
    safmin = _hqr_safmin(R)
    safmn2 = TWO ^ trunc(Int, log(safmin / eps_p) / log(TWO) / 2)
    safmx2 = ONE / safmn2
    cs = ONE; sn = Z0
    if c == Z0
        cs = ONE; sn = Z0
    elseif b == Z0
        cs = Z0; sn = ONE
        temp = d; d = a; a = temp; b = -c; c = Z0
    elseif (a - d) == Z0 && copysign(ONE, b) != copysign(ONE, c)
        cs = ONE; sn = Z0
    else
        temp = a - d
        p = HALF * temp
        bcmax = max(abs(b), abs(c))
        bcmis = min(abs(b), abs(c)) * copysign(ONE, b) * copysign(ONE, c)
        scale = max(abs(p), bcmax)
        z = (p / scale) * p + (bcmax / scale) * bcmis
        if z >= MULTPL * eps_p
            # Real eigenvalues.
            z = p + copysign(sqrt(scale) * sqrt(z), p)
            a = d + z
            d = d - (bcmax / z) * bcmis
            tau = hypot(c, z)
            cs = z / tau
            sn = c / tau
            b = b - c
            c = Z0
        else
            # Complex eigenvalues, or real (almost) equal eigenvalues.
            count = 0
            sigma = b + c
            while true
                count += 1
                scale = max(abs(temp), abs(sigma))
                if scale >= safmx2
                    sigma *= safmn2; temp *= safmn2
                    count <= 20 && continue
                end
                if scale <= safmn2
                    sigma *= safmx2; temp *= safmx2
                    count <= 20 && continue
                end
                break
            end
            p = HALF * temp
            tau = hypot(sigma, temp)
            cs = sqrt(HALF * (ONE + abs(sigma) / tau))
            sn = -(p / (tau * cs)) * copysign(ONE, sigma)
            aa = a * cs + b * sn
            bb = -a * sn + b * cs
            cc = c * cs + d * sn
            dd = -c * sn + d * cs
            a = aa * cs + cc * sn
            b = (bb * cs) + (dd * sn)
            c = -(aa * sn) + (cc * cs)
            d = -bb * sn + dd * cs
            temp = HALF * (a + d)
            a = temp; d = temp
            if c != Z0
                if b != Z0
                    if copysign(ONE, b) == copysign(ONE, c)
                        # Real eigenvalues: reduce to upper triangular form.
                        sab = sqrt(abs(b))
                        sac = sqrt(abs(c))
                        p = copysign(sab * sac, c)
                        tau = ONE / sqrt(abs(b + c))
                        a = temp + p
                        d = temp - p
                        b = b - c
                        c = Z0
                        cs1 = sab * tau
                        sn1 = sac * tau
                        temp2 = cs * cs1 - sn * sn1
                        sn = cs * sn1 + sn * cs1
                        cs = temp2
                    end
                else
                    b = -c
                    c = Z0
                    temp2 = cs
                    cs = -sn
                    sn = temp2
                end
            end
        end
    end
    rt1r = a; rt2r = d
    if c == Z0
        rt1i = Z0; rt2i = Z0
    else
        rt1i = sqrt(abs(b)) * sqrt(abs(c))
        rt2i = -rt1i
    end
    return a, b, c, d, rt1r, rt1i, rt2r, rt2i, cs, sn
end

# DROT(n): x' = c·x + s·y, y' = c·y − s·x, applied to two strided lanes of a matrix.
@inline function _drot_rows!(H::AbstractMatrix{R}, r1::Int, r2::Int, jlo::Int, jhi::Int, c::R, s::R) where {R<:Real}
    @inbounds for j in jlo:jhi
        t = H[r1, j]; u = H[r2, j]
        H[r1, j] = c * t + s * u
        H[r2, j] = c * u - s * t
    end
end
@inline function _drot_cols!(H::AbstractMatrix{R}, c1::Int, c2::Int, ilo::Int, ihi::Int, c::R, s::R) where {R<:Real}
    @inbounds for i in ilo:ihi
        t = H[i, c1]; u = H[i, c2]
        H[i, c1] = c * t + s * u
        H[i, c2] = c * u - s * t
    end
end

# ── DLAHQR (Reference-LAPACK verbatim), REAL double-shift Francis QR ───────────────────────────────────
# H (n×n, upper Hessenberg on ilo:ihi) → (quasi-)upper-triangular Schur T in place. wr,wi get the
# eigenvalues on ilo:ihi. Z (n×n) accumulates the Schur rotations over rows iloz:ihiz if wantz.
# Returns info: 0 on success; i>0 = failed to converge (eigenvalues i+1:ihi are correct, ilo:i are not).
function _dlahqr!(wantt::Bool, wantz::Bool, H::AbstractMatrix{R}, ilo::Int, ihi::Int,
        wr::AbstractVector{R}, wi::AbstractVector{R}, iloz::Int, ihiz::Int,
        Z::AbstractMatrix{R}) where {R<:Real}
    Z0 = zero(R); ONE = one(R); TWO = R(2)
    DAT1 = R(3) / R(4); DAT2 = R(-0.4375); KEXSH = 10
    info = 0
    n = size(H, 1)
    n == 0 && return info
    if ilo == ihi
        @inbounds wr[ilo] = H[ilo, ilo]; @inbounds wi[ilo] = Z0
        return info
    end
    # ==== clear out the trash ====
    @inbounds for j in ilo:ihi-3
        H[j+2, j] = Z0
        H[j+3, j] = Z0
    end
    ilo <= ihi - 2 && (@inbounds H[ihi, ihi-2] = Z0)

    nh = ihi - ilo + 1
    safmin = _hqr_safmin(R)
    ulp = eps(R)                                   # DLAMCH('P')
    smlnum = safmin * (R(nh) / ulp)

    i1 = 1; i2 = n
    itmax = 30 * max(10, nh)
    kdefl = 0
    v = Vector{R}(undef, 3)

    i = ihi
    @inbounds while i >= ilo
        l = ilo
        converged_split = false
        its = 0
        while its <= itmax
            # ---- look for a single small subdiagonal element ----
            k = i
            while k >= l + 1
                if abs(H[k, k-1]) <= smlnum
                    break
                end
                tst = abs(H[k-1, k-1]) + abs(H[k, k])
                if tst == Z0
                    k - 2 >= ilo && (tst += abs(H[k-1, k-2]))
                    k + 1 <= ihi && (tst += abs(H[k+1, k]))
                end
                if abs(H[k, k-1]) <= ulp * tst
                    ab = max(abs(H[k, k-1]), abs(H[k-1, k]))
                    ba = min(abs(H[k, k-1]), abs(H[k-1, k]))
                    aa = max(abs(H[k, k]), abs(H[k-1, k-1] - H[k, k]))
                    bb = min(abs(H[k, k]), abs(H[k-1, k-1] - H[k, k]))
                    s = aa + ab
                    if ba * (ab / s) <= max(smlnum, ulp * (bb * (aa / s)))
                        break
                    end
                end
                k -= 1
            end
            l = k
            if l > ilo
                H[l, l-1] = Z0
            end
            # exit if a submatrix of order 1 or 2 has split off
            if l >= i - 1
                converged_split = true
                break
            end
            kdefl += 1
            if !wantt
                i1 = l; i2 = i
            end

            local h11, h21, h12, h22
            if kdefl % (2 * KEXSH) == 0
                s = abs(H[i, i-1]) + abs(H[i-1, i-2])
                h11 = DAT1 * s + H[i, i]; h12 = DAT2 * s; h21 = s; h22 = h11
            elseif kdefl % KEXSH == 0
                s = abs(H[l+1, l]) + abs(H[l+2, l+1])
                h11 = DAT1 * s + H[l, l]; h12 = DAT2 * s; h21 = s; h22 = h11
            else
                h11 = H[i-1, i-1]; h21 = H[i, i-1]; h12 = H[i-1, i]; h22 = H[i, i]
            end
            s = abs(h11) + abs(h12) + abs(h21) + abs(h22)
            local rt1r, rt1i, rt2r, rt2i
            if s == Z0
                rt1r = Z0; rt1i = Z0; rt2r = Z0; rt2i = Z0
            else
                h11 /= s; h21 /= s; h12 /= s; h22 /= s
                tr = (h11 + h22) / TWO
                det = (h11 - tr) * (h22 - tr) - h12 * h21
                rtdisc = sqrt(abs(det))
                if det >= Z0
                    rt1r = tr * s; rt2r = rt1r; rt1i = rtdisc * s; rt2i = -rt1i
                else
                    rt1r = tr + rtdisc; rt2r = tr - rtdisc
                    if abs(rt1r - h22) <= abs(rt2r - h22)
                        rt1r *= s; rt2r = rt1r
                    else
                        rt2r *= s; rt1r = rt2r
                    end
                    rt1i = Z0; rt2i = Z0
                end
            end

            # ---- look for two consecutive small subdiagonal elements ----
            mfound = l
            mm = i - 2
            while mm >= l
                h21s = H[mm+1, mm]
                s = abs(H[mm, mm] - rt2r) + abs(rt2i) + abs(h21s)
                h21s = H[mm+1, mm] / s
                v[1] = h21s * H[mm, mm+1] + (H[mm, mm] - rt1r) * ((H[mm, mm] - rt2r) / s) - rt1i * (rt2i / s)
                v[2] = h21s * (H[mm, mm] + H[mm+1, mm+1] - rt1r - rt2r)
                v[3] = h21s * H[mm+2, mm+1]
                s = abs(v[1]) + abs(v[2]) + abs(v[3])
                v[1] /= s; v[2] /= s; v[3] /= s
                if mm == l
                    mfound = mm; break
                end
                if abs(H[mm, mm-1]) * (abs(v[2]) + abs(v[3])) <=
                        ulp * abs(v[1]) * (abs(H[mm-1, mm-1]) + abs(H[mm, mm]) + abs(H[mm+1, mm+1]))
                    mfound = mm; break
                end
                mfound = mm
                mm -= 1
            end
            m = mfound

            # ---- double-shift QR sweep: chase the bulge from row m to i−1 ----
            for k in m:i-1
                nr = min(3, i - k + 1)
                if k > m
                    for r in 1:nr; v[r] = H[k+r-1, k-1]; end
                end
                t1 = _hqr_larfg!(v, nr)
                if k > m
                    H[k, k-1] = v[1]
                    H[k+1, k-1] = Z0
                    k < i - 1 && (H[k+2, k-1] = Z0)
                elseif m > l
                    # avoid a bug when v[2],v[3] underflow
                    H[k, k-1] = H[k, k-1] * (ONE - t1)
                end
                v2 = v[2]
                t2 = t1 * v2
                if nr == 3
                    v3 = v[3]
                    t3 = t1 * v3
                    for j in k:i2
                        sum = H[k, j] + v2 * H[k+1, j] + v3 * H[k+2, j]
                        H[k, j] -= sum * t1
                        H[k+1, j] -= sum * t2
                        H[k+2, j] -= sum * t3
                    end
                    for j in i1:min(k+3, i)
                        sum = H[j, k] + v2 * H[j, k+1] + v3 * H[j, k+2]
                        H[j, k] -= sum * t1
                        H[j, k+1] -= sum * t2
                        H[j, k+2] -= sum * t3
                    end
                    if wantz
                        for j in iloz:ihiz
                            sum = Z[j, k] + v2 * Z[j, k+1] + v3 * Z[j, k+2]
                            Z[j, k] -= sum * t1
                            Z[j, k+1] -= sum * t2
                            Z[j, k+2] -= sum * t3
                        end
                    end
                elseif nr == 2
                    for j in k:i2
                        sum = H[k, j] + v2 * H[k+1, j]
                        H[k, j] -= sum * t1
                        H[k+1, j] -= sum * t2
                    end
                    for j in i1:i
                        sum = H[j, k] + v2 * H[j, k+1]
                        H[j, k] -= sum * t1
                        H[j, k+1] -= sum * t2
                    end
                    if wantz
                        for j in iloz:ihiz
                            sum = Z[j, k] + v2 * Z[j, k+1]
                            Z[j, k] -= sum * t1
                            Z[j, k+1] -= sum * t2
                        end
                    end
                end
            end
            its += 1
        end

        if !converged_split
            info = i          # failure to converge
            return info
        end

        # ---- a submatrix of order 1 or 2 has split off at rows l:i ----
        if l == i
            wr[i] = H[i, i]; wi[i] = Z0
        elseif l == i - 1
            a, b, c, d, rt1r, rt1i, rt2r, rt2i, cs, sn =
                _dlanv2(H[i-1, i-1], H[i-1, i], H[i, i-1], H[i, i])
            H[i-1, i-1] = a; H[i-1, i] = b; H[i, i-1] = c; H[i, i] = d
            wr[i-1] = rt1r; wi[i-1] = rt1i; wr[i] = rt2r; wi[i] = rt2i
            if wantt
                i2 > i && _drot_rows!(H, i-1, i, i+1, i2, cs, sn)
                _drot_cols!(H, i-1, i, i1, i-2, cs, sn)
            end
            if wantz
                _drot_cols!(Z, i-1, i, iloz, ihiz, cs, sn)
            end
        end
        kdefl = 0
        i = l - 1
    end
    return info
end

# ── ZLAHQR (Reference-LAPACK verbatim), COMPLEX single-shift Francis QR ───────────────────────────────
# Fully upper-triangular Schur T; w[ilo:ihi] = diagonal eigenvalues. Keeps subdiagonals real throughout
# (the phase-normalization passes) so the resulting T is a valid complex Schur form for trevc.
function _zlahqr!(wantt::Bool, wantz::Bool, H::AbstractMatrix{T}, ilo::Int, ihi::Int,
        w::AbstractVector{T}, iloz::Int, ihiz::Int, Z::AbstractMatrix{T}) where {T<:Complex}
    R = real(T)
    Z0 = zero(T); ONE = one(T)
    RZERO = zero(R); RHALF = R(0.5); DAT1 = R(3) / R(4); KEXSH = 10
    cabs1(z) = abs(real(z)) + abs(imag(z))
    info = 0
    n = size(H, 1)
    n == 0 && return info
    if ilo == ihi
        @inbounds w[ilo] = H[ilo, ilo]
        return info
    end
    # ==== clear out the trash ====
    @inbounds for j in ilo:ihi-3
        H[j+2, j] = Z0; H[j+3, j] = Z0
    end
    ilo <= ihi - 2 && (@inbounds H[ihi, ihi-2] = Z0)

    # ==== ensure that subdiagonal entries are real ====
    jlo = wantt ? 1 : ilo
    jhi = wantt ? n : ihi
    @inbounds for ii in ilo+1:ihi
        if imag(H[ii, ii-1]) != RZERO
            sc = H[ii, ii-1] / cabs1(H[ii, ii-1])
            sc = conj(sc) / abs(sc)
            H[ii, ii-1] = Complex(abs(H[ii, ii-1]), RZERO)
            for j in ii:jhi;      H[ii, j] *= sc; end              # ZSCAL row ii, cols ii:jhi
            for r in jlo:min(jhi, ii+1); H[r, ii] *= conj(sc); end  # ZSCAL col ii, rows jlo:min(jhi,ii+1)
            if wantz
                for r in iloz:ihiz; Z[r, ii] *= conj(sc); end
            end
        end
    end

    nh = ihi - ilo + 1
    safmin = _hqr_safmin(R)
    ulp = eps(R)
    smlnum = safmin * (R(nh) / ulp)

    i1 = 1; i2 = n
    itmax = 30 * max(10, nh)
    kdefl = 0
    v1 = Z0; v2 = Z0

    i = ihi
    @inbounds while i >= ilo
        l = ilo
        converged = false
        its = 0
        while its <= itmax
            # ---- look for a single small subdiagonal element ----
            k = i
            while k >= l + 1
                if cabs1(H[k, k-1]) <= smlnum
                    break
                end
                tst = cabs1(H[k-1, k-1]) + cabs1(H[k, k])
                if tst == RZERO
                    k - 2 >= ilo && (tst += abs(real(H[k-1, k-2])))
                    k + 1 <= ihi && (tst += abs(real(H[k+1, k])))
                end
                if abs(real(H[k, k-1])) <= ulp * tst
                    ab = max(cabs1(H[k, k-1]), cabs1(H[k-1, k]))
                    ba = min(cabs1(H[k, k-1]), cabs1(H[k-1, k]))
                    aa = max(cabs1(H[k, k]), cabs1(H[k-1, k-1] - H[k, k]))
                    bb = min(cabs1(H[k, k]), cabs1(H[k-1, k-1] - H[k, k]))
                    s = aa + ab
                    if ba * (ab / s) <= max(smlnum, ulp * (bb * (aa / s)))
                        break
                    end
                end
                k -= 1
            end
            l = k
            l > ilo && (H[l, l-1] = Z0)
            if l >= i
                converged = true
                break
            end
            kdefl += 1
            if !wantt
                i1 = l; i2 = i
            end

            local t::T
            if kdefl % (2 * KEXSH) == 0
                s = DAT1 * abs(real(H[i, i-1]))
                t = Complex(s, RZERO) + H[i, i]
            elseif kdefl % KEXSH == 0
                s = DAT1 * abs(real(H[l+1, l]))
                t = Complex(s, RZERO) + H[l, l]
            else
                t = H[i, i]
                u = sqrt(H[i-1, i]) * sqrt(H[i, i-1])
                s = cabs1(u)
                if s != RZERO
                    x = RHALF * (H[i-1, i-1] - t)
                    sx = cabs1(x)
                    s = max(s, cabs1(x))
                    y = s * sqrt((x / s)^2 + (u / s)^2)
                    if sx > RZERO
                        if real(x / sx) * real(y) + imag(x / sx) * imag(y) < RZERO
                            y = -y
                        end
                    end
                    t = t - u * (u / (x + y))
                end
            end

            # ---- look for two consecutive small subdiagonal elements ----
            m = l
            found_m = false
            mm = i - 1
            while mm >= l + 1
                h11 = H[mm, mm]
                h22 = H[mm+1, mm+1]
                h11s = h11 - t
                h21 = real(H[mm+1, mm])
                s = cabs1(h11s) + abs(h21)
                h11s = h11s / s
                h21 = h21 / s
                v1 = h11s; v2 = Complex(h21, RZERO)
                h10 = real(H[mm, mm-1])
                if abs(h10) * abs(h21) <= ulp * (cabs1(h11s) * (cabs1(h11) + cabs1(h22)))
                    m = mm; found_m = true; break
                end
                mm -= 1
            end
            if !found_m
                m = l
                h11 = H[l, l]; h22 = H[l+1, l+1]
                h11s = h11 - t
                h21 = real(H[l+1, l])
                s = cabs1(h11s) + abs(h21)
                h11s = h11s / s; h21 = h21 / s
                v1 = h11s; v2 = Complex(h21, RZERO)
            end

            # ---- single-shift QR sweep ----
            for k in m:i-1
                if k > m
                    v1 = H[k, k-1]; v2 = H[k+1, k-1]
                end
                β, t1, v2n = _zlarfg2(v1, v2)
                if k > m
                    H[k, k-1] = Complex(β, RZERO)
                    H[k+1, k-1] = Z0
                end
                v2 = v2n
                t2 = real(t1 * v2)
                for j in k:i2
                    sum = conj(t1) * H[k, j] + t2 * H[k+1, j]
                    H[k, j] -= sum
                    H[k+1, j] -= sum * v2
                end
                for j in i1:min(k+2, i)
                    sum = t1 * H[j, k] + t2 * H[j, k+1]
                    H[j, k] -= sum
                    H[j, k+1] -= sum * conj(v2)
                end
                if wantz
                    for j in iloz:ihiz
                        sum = t1 * Z[j, k] + t2 * Z[j, k+1]
                        Z[j, k] -= sum
                        Z[j, k+1] -= sum * conj(v2)
                    end
                end
                if k == m && m > l
                    # extra scaling so H(m+1,m) stays real
                    temp = ONE - t1
                    temp = temp / abs(temp)
                    H[m+1, m] = H[m+1, m] * conj(temp)
                    m + 2 <= i && (H[m+2, m+1] = H[m+2, m+1] * temp)
                    for j in m:i
                        if j != m + 1
                            i2 > j && (for c in j+1:i2; H[j, c] *= temp; end)
                            for r in i1:j-1; H[r, j] *= conj(temp); end
                            if wantz
                                for r in iloz:ihiz; Z[r, j] *= conj(temp); end
                            end
                        end
                    end
                end
            end

            # ---- ensure H(i,i-1) is real ----
            temp = H[i, i-1]
            if imag(temp) != RZERO
                rtemp = abs(temp)
                H[i, i-1] = Complex(rtemp, RZERO)
                temp = temp / rtemp
                i2 > i && (for c in i+1:i2; H[i, c] *= conj(temp); end)
                for r in i1:i-1; H[r, i] *= temp; end
                if wantz
                    for r in iloz:ihiz; Z[r, i] *= temp; end
                end
            end
            its += 1
        end

        if !converged
            info = i
            return info
        end
        w[i] = H[i, i]
        kdefl = 0
        i = l - 1
    end
    return info
end

"""
    hseqr!(job, compz, H, ilo, ihi, w, Z) -> info

Schur decomposition of an upper-Hessenberg matrix by the unblocked Francis double-shift QR
(LAPACK `dlahqr`/`zlahqr`). `H` (n×n, upper Hessenberg on `ilo:ihi`) is overwritten with the
(quasi-)upper-triangular Schur form `T`; `w` (length n, complex) receives the eigenvalues; `Z`
accumulates the Schur vectors so that (for `job='S'`, `compz∈{'I','V'}`) `H₀ = Z·T·Zᴴ`.

- `job`  : `'E'` eigenvalues only, `'S'` full Schur form.
- `compz`: `'N'` no vectors, `'I'` `Z := I` then accumulate, `'V'` accumulate into the given `Z`.

Returns `info` (0 = success; `i>0` = failed to converge; `w[i+1:ihi]` are the converged eigenvalues).
Generic over `T<:Real` (real Schur form, 2×2 blocks for complex-conjugate pairs) and `T<:Complex`
(triangular T). For real `T`, `w::AbstractVector{<:Complex}`.
"""
function hseqr!(job::AbstractChar, compz::AbstractChar, H::AbstractMatrix{T}, ilo::Integer,
        ihi::Integer, w::AbstractVector, Z::AbstractMatrix{T}) where {T<:Real}
    n = size(H, 1)
    size(H, 2) == n || throw(DimensionMismatch("hseqr!: H must be square"))
    (job === 'E' || job === 'S') || throw(ArgumentError("hseqr!: job must be 'E' or 'S'"))
    (compz === 'N' || compz === 'I' || compz === 'V') ||
        throw(ArgumentError("hseqr!: compz must be 'N', 'I' or 'V'"))
    wantt = job === 'S'
    wantz = compz !== 'N'
    if compz === 'I'
        fill!(Z, zero(T))
        @inbounds for i in 1:n; Z[i, i] = one(T); end
    end
    wr = Vector{T}(undef, n); wi = Vector{T}(undef, n)
    ilo = Int(ilo); ihi = Int(ihi)
    @inbounds for i in 1:ilo-1; wr[i] = H[i, i]; wi[i] = zero(T); end
    @inbounds for i in ihi+1:n; wr[i] = H[i, i]; wi[i] = zero(T); end
    info = _dlahqr!(wantt, wantz, H, ilo, ihi, wr, wi, 1, n, Z)
    @inbounds for i in 1:n; w[i] = Complex(wr[i], wi[i]); end
    return info
end

function hseqr!(job::AbstractChar, compz::AbstractChar, H::AbstractMatrix{T}, ilo::Integer,
        ihi::Integer, w::AbstractVector{T}, Z::AbstractMatrix{T}) where {T<:Complex}
    n = size(H, 1)
    size(H, 2) == n || throw(DimensionMismatch("hseqr!: H must be square"))
    (job === 'E' || job === 'S') || throw(ArgumentError("hseqr!: job must be 'E' or 'S'"))
    (compz === 'N' || compz === 'I' || compz === 'V') ||
        throw(ArgumentError("hseqr!: compz must be 'N', 'I' or 'V'"))
    wantt = job === 'S'
    wantz = compz !== 'N'
    if compz === 'I'
        fill!(Z, zero(T))
        @inbounds for i in 1:n; Z[i, i] = one(T); end
    end
    ilo = Int(ilo); ihi = Int(ihi)
    @inbounds for i in 1:ilo-1; w[i] = H[i, i]; end
    @inbounds for i in ihi+1:n; w[i] = H[i, i]; end
    return _zlahqr!(wantt, wantz, H, ilo, ihi, w, 1, n, Z)
end
