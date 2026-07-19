# LAPACK nonsymmetric-eigen EIGENVECTOR kernel (companion to hseqr.jl):
#   trevc — right eigenvectors of a (quasi-)upper-triangular Schur form T by back-substitution.
# CORRECTNESS-FIRST port of Reference-LAPACK dtrevc (real) / ztrevc (complex), SIDE='R'. The real
# path uses the dlaln2 1×1/2×2 quasi-triangular solve verbatim; the complex path solves the upper
# triangular system by guarded back-substitution (the direction dtrevc/ztrevc compute, up to scale).
#
# HOWMNY: 'A' overwrite VR with the eigenvectors of T itself; 'B' back-transform — VR holds the Schur
# vectors Z on input and receives the eigenvectors of the original matrix (Z·x). Left eigenvectors
# (SIDE='L'/'B') are a documented follow-up (right is the common case, needed by `eigen`).
#
# Generic over T (s/d/c/z). Self-contained (local dlaln2/dladiv ports).

# ── DLADIV (robust complex division a+bi / c+di), Reference-LAPACK dladiv/dladiv1/dladiv2 ─────────────
@inline function _dladiv2(a::R, b::R, c::R, d::R, r::R, t::R) where {R<:Real}
    if r != zero(R)
        br = b * r
        return br != zero(R) ? (a + br) * t : a * t + (b * t) * r
    else
        return (a + d * (b / c)) * t
    end
end
@inline function _dladiv1(a::R, b::R, c::R, d::R) where {R<:Real}
    r = d / c
    t = one(R) / (c + d * r)
    p = _dladiv2(a, b, c, d, r, t)
    q = _dladiv2(b, -a, c, d, r, t)
    return p, q
end
# Returns (p,q) with p+qi = (a+bi)/(c+di); overflow-safe scaling per dladiv.f.
@inline function _dladiv(a::R, b::R, c::R, d::R) where {R<:Real}
    BS = R(2); HALF = R(0.5); TWO = R(2)
    aa = a; bb = b; cc = c; dd = d
    ab = max(abs(a), abs(b)); cd = max(abs(c), abs(d)); s = one(R)
    ov = floatmax(R); un = floatmin(R); ϵ = eps(R) / 2; be = BS / (ϵ * ϵ)
    if ab >= HALF * ov; aa *= HALF; bb *= HALF; s *= TWO; end
    if cd >= HALF * ov; cc *= HALF; dd *= HALF; s *= HALF; end
    if ab <= un * BS / ϵ; aa *= be; bb *= be; s /= be; end
    if cd <= un * BS / ϵ; cc *= be; dd *= be; s *= be; end
    if abs(d) <= abs(c)
        p, q = _dladiv1(aa, bb, cc, dd)
    else
        p, q = _dladiv1(bb, aa, dd, cc); q = -q
    end
    return p * s, q * s
end

const _LN2_IPIVOT = ((1, 2, 3, 4), (2, 1, 4, 3), (3, 4, 1, 2), (4, 3, 2, 1))
const _LN2_ZSWAP = (false, false, true, true)
const _LN2_RSWAP = (false, true, false, true)

# ── DLALN2 specialized for dtrevc's calls (LTRANS=F, CA=1, D1=D2=1) — Reference-LAPACK verbatim ───────
# Solves (A − w·I)·X = B for the na×na block A (na∈{1,2}), na×nw right-hand side (nw∈{1,2}: real/complex
# eigenvalue w = wr+i·wi), with the same complete-pivoting + SMINI-guarded elimination as dlaln2.f.
# Returns (x11,x21,x12,x22, scale, xnorm). Only the used entries are meaningful.
function _dlaln2(na::Int, nw::Int, smin::R, a11::R, a12::R, a21::R, a22::R,
        b11::R, b21::R, b12::R, b22::R, wr::R, wi::R) where {R<:Real}
    ZERO = zero(R); ONE = one(R)
    smlnum = R(2) * _dtrevc_safmin(R)
    bignum = ONE / smlnum
    smini = max(smin, smlnum)
    scale = ONE
    x11 = ZERO; x21 = ZERO; x12 = ZERO; x22 = ZERO; xnorm = ZERO
    if na == 1
        if nw == 1
            csr = a11 - wr
            cnorm = abs(csr)
            if cnorm < smini
                csr = smini; cnorm = smini
            end
            bnorm = abs(b11)
            if cnorm < ONE && bnorm > ONE
                bnorm > bignum * cnorm && (scale = ONE / bnorm)
            end
            x11 = (b11 * scale) / csr
            xnorm = abs(x11)
        else
            csr = a11 - wr
            csi = -wi
            cnorm = abs(csr) + abs(csi)
            if cnorm < smini
                csr = smini; csi = ZERO; cnorm = smini
            end
            bnorm = abs(b11) + abs(b12)
            if cnorm < ONE && bnorm > ONE
                bnorm > bignum * cnorm && (scale = ONE / bnorm)
            end
            x11, x12 = _dladiv(scale * b11, scale * b12, csr, csi)
            xnorm = abs(x11) + abs(x12)
        end
    else
        # 2×2 system. CR = A − wr·I ; (ltrans=F)
        cr11 = a11 - wr; cr22 = a22 - wr
        cr21 = a21; cr12 = a12
        crv = (cr11, cr21, cr12, cr22)
        if nw == 1
            # real 2×2
            cmax = ZERO; icmax = 0
            for j in 1:4
                if abs(crv[j]) > cmax
                    cmax = abs(crv[j]); icmax = j
                end
            end
            if cmax < smini
                bnorm = max(abs(b11), abs(b21))
                if smini < ONE && bnorm > ONE
                    bnorm > bignum * smini && (scale = ONE / bnorm)
                end
                temp = scale / smini
                x11 = temp * b11; x21 = temp * b21
                xnorm = temp * bnorm
                return x11, x21, x12, x22, scale, xnorm
            end
            piv = _LN2_IPIVOT[icmax]
            ur11 = crv[icmax]; cr21p = crv[piv[2]]; ur12 = crv[piv[3]]; cr22p = crv[piv[4]]
            ur11r = ONE / ur11
            lr21 = ur11r * cr21p
            ur22 = cr22p - ur12 * lr21
            abs(ur22) < smini && (ur22 = smini)
            if _LN2_RSWAP[icmax]
                br1 = b21; br2 = b11
            else
                br1 = b11; br2 = b21
            end
            br2 = br2 - lr21 * br1
            bbnd = max(abs(br1 * (ur22 * ur11r)), abs(br2))
            if bbnd > ONE && abs(ur22) < ONE
                bbnd >= bignum * abs(ur22) && (scale = ONE / bbnd)
            end
            xr2 = (br2 * scale) / ur22
            xr1 = (scale * br1) * ur11r - xr2 * (ur11r * ur12)
            if _LN2_ZSWAP[icmax]
                x11 = xr2; x21 = xr1
            else
                x11 = xr1; x21 = xr2
            end
            xnorm = max(abs(xr1), abs(xr2))
            if xnorm > ONE && cmax > ONE
                if xnorm > bignum / cmax
                    temp = cmax / bignum
                    x11 *= temp; x21 *= temp; xnorm *= temp; scale *= temp
                end
            end
        else
            # complex 2×2 : CI = -wi on the diagonal
            ci11 = -wi; ci22 = -wi
            civ = (ci11, ZERO, ZERO, ci22)
            cmax = ZERO; icmax = 0
            for j in 1:4
                if abs(crv[j]) + abs(civ[j]) > cmax
                    cmax = abs(crv[j]) + abs(civ[j]); icmax = j
                end
            end
            if cmax < smini
                bnorm = max(abs(b11) + abs(b12), abs(b21) + abs(b22))
                if smini < ONE && bnorm > ONE
                    bnorm > bignum * smini && (scale = ONE / bnorm)
                end
                temp = scale / smini
                x11 = temp * b11; x21 = temp * b21; x12 = temp * b12; x22 = temp * b22
                xnorm = temp * bnorm
                return x11, x21, x12, x22, scale, xnorm
            end
            piv = _LN2_IPIVOT[icmax]
            ur11 = crv[icmax]; ui11 = civ[icmax]
            cr21p = crv[piv[2]]; ci21p = civ[piv[2]]
            ur12 = crv[piv[3]]; ui12 = civ[piv[3]]
            cr22p = crv[piv[4]]; ci22p = civ[piv[4]]
            if icmax == 1 || icmax == 4
                # off-diagonals of pivoted C are real
                if abs(ur11) > abs(ui11)
                    temp = ui11 / ur11
                    ur11r = ONE / (ur11 * (ONE + temp^2))
                    ui11r = -temp * ur11r
                else
                    temp = ur11 / ui11
                    ui11r = -ONE / (ui11 * (ONE + temp^2))
                    ur11r = -temp * ui11r
                end
                lr21 = cr21p * ur11r
                li21 = cr21p * ui11r
                ur12s = ur12 * ur11r
                ui12s = ur12 * ui11r
                ur22 = cr22p - ur12 * lr21
                ui22 = ci22p - ur12 * li21
            else
                # diagonals of pivoted C are real
                ur11r = ONE / ur11
                ui11r = ZERO
                lr21 = cr21p * ur11r
                li21 = ci21p * ur11r
                ur12s = ur12 * ur11r
                ui12s = ui12 * ur11r
                ur22 = cr22p - ur12 * lr21 + ui12 * li21
                ui22 = -ur12 * li21 - ui12 * lr21
            end
            u22abs = abs(ur22) + abs(ui22)
            if u22abs < smini
                ur22 = smini; ui22 = ZERO
            end
            if _LN2_RSWAP[icmax]
                br2 = b11; br1 = b21; bi2 = b12; bi1 = b22
            else
                br1 = b11; br2 = b21; bi1 = b12; bi2 = b22
            end
            br2 = br2 - lr21 * br1 + li21 * bi1
            bi2 = bi2 - li21 * br1 - lr21 * bi1
            bbnd = max((abs(br1) + abs(bi1)) * (u22abs * (abs(ur11r) + abs(ui11r))),
                       abs(br2) + abs(bi2))
            if bbnd > ONE && u22abs < ONE
                if bbnd >= bignum * u22abs
                    scale = ONE / bbnd
                    br1 *= scale; bi1 *= scale; br2 *= scale; bi2 *= scale
                end
            end
            xr2, xi2 = _dladiv(br2, bi2, ur22, ui22)
            xr1 = ur11r * br1 - ui11r * bi1 - ur12s * xr2 + ui12s * xi2
            xi1 = ui11r * br1 + ur11r * bi1 - ui12s * xr2 - ur12s * xi2
            if _LN2_ZSWAP[icmax]
                x11 = xr2; x21 = xr1; x12 = xi2; x22 = xi1
            else
                x11 = xr1; x21 = xr2; x12 = xi1; x22 = xi2
            end
            xnorm = max(abs(xr1) + abs(xi1), abs(xr2) + abs(xi2))
            if xnorm > ONE && cmax > ONE
                if xnorm > bignum / cmax
                    temp = cmax / bignum
                    x11 *= temp; x21 *= temp; x12 *= temp; x22 *= temp
                    xnorm *= temp; scale *= temp
                end
            end
        end
    end
    return x11, x21, x12, x22, scale, xnorm
end

@inline function _dtrevc_safmin(::Type{R}) where {R<:Real}
    sfmin = floatmin(R)
    small = one(R) / floatmax(R)
    small >= sfmin && (sfmin = small * (one(R) + eps(R)))
    return sfmin
end

# index of max abs of x[1:m]
@inline function _iamax(x::AbstractVector{R}, m::Int) where {R<:Real}
    ii = 1; best = abs(x[1])
    @inbounds for k in 2:m
        a = abs(x[k])
        a > best && (best = a; ii = k)
    end
    return ii
end

# ── DTREVC right eigenvectors (Reference-LAPACK verbatim), REAL quasi-triangular T ────────────────────
function _dtrevc_right!(over::Bool, T::AbstractMatrix{R}, VR::AbstractMatrix{R}) where {R<:Real}
    ZERO = zero(R); ONE = one(R)
    n = size(T, 1)
    n == 0 && return VR
    unfl = _dtrevc_safmin(R)
    ulp = eps(R)
    smlnum = unfl * (R(n) / ulp)
    bignum = (ONE - ulp) / smlnum
    # column 1-norms of strict upper triangle
    cnorm = zeros(R, n)
    @inbounds for j in 2:n, i in 1:j-1
        cnorm[j] += abs(T[i, j])
    end
    xr = zeros(R, n)          # WORK(1+N .. )  real part
    xi = zeros(R, n)          # WORK(1+N2 .. ) imag part
    ip = 0
    is = n
    ki = n
    @inbounds while ki >= 1
        skip = false
        if ip == 1
            skip = true       # this column handled by the previous (complex pair)
        else
            if ki != 1 && T[ki, ki-1] != ZERO
                ip = -1
            end
        end
        if !skip
            wr = T[ki, ki]; wi = ZERO
            ip != 0 && (wi = sqrt(abs(T[ki, ki-1])) * sqrt(abs(T[ki-1, ki])))
            smin = max(ulp * (abs(wr) + abs(wi)), smlnum)

            if ip == 0
                # ---- real right eigenvector ----
                xr[ki] = ONE
                for k in 1:ki-1; xr[k] = -T[k, ki]; end
                jnxt = ki - 1
                j = ki - 1
                while j >= 1
                    if j > jnxt
                        j -= 1; continue
                    end
                    j1 = j; j2 = j; jnxt = j - 1
                    if j > 1 && T[j, j-1] != ZERO
                        j1 = j - 1; jnxt = j - 2
                    end
                    if j1 == j2
                        x11, _, _, _, scale, xnorm =
                            _dlaln2(1, 1, smin, T[j, j], ZERO, ZERO, ZERO, xr[j], ZERO, ZERO, ZERO, wr, ZERO)
                        if xnorm > ONE && cnorm[j] > bignum / xnorm
                            x11 /= xnorm; scale /= xnorm
                        end
                        if scale != ONE
                            for k in 1:ki; xr[k] *= scale; end
                        end
                        xr[j] = x11
                        for k in 1:j-1; xr[k] -= x11 * T[k, j]; end
                    else
                        x11, x21, _, _, scale, xnorm =
                            _dlaln2(2, 1, smin, T[j-1, j-1], T[j-1, j], T[j, j-1], T[j, j],
                                    xr[j-1], xr[j], ZERO, ZERO, wr, ZERO)
                        if xnorm > ONE
                            beta = max(cnorm[j-1], cnorm[j])
                            if beta > bignum / xnorm
                                x11 /= xnorm; x21 /= xnorm; scale /= xnorm
                            end
                        end
                        if scale != ONE
                            for k in 1:ki; xr[k] *= scale; end
                        end
                        xr[j-1] = x11; xr[j] = x21
                        for k in 1:j-2; xr[k] -= x11 * T[k, j-1]; end
                        for k in 1:j-2; xr[k] -= x21 * T[k, j]; end
                    end
                    j = j1 - 1
                end
                # copy x or Z·x to VR and normalize
                if !over
                    for k in 1:ki; VR[k, is] = xr[k]; end
                    ii = _iamax(view(VR, :, is), ki)
                    remax = ONE / abs(VR[ii, is])
                    for k in 1:ki; VR[k, is] *= remax; end
                    for k in ki+1:n; VR[k, is] = ZERO; end
                else
                    if ki > 1
                        for row in 1:n
                            acc = xr[ki] * VR[row, ki]
                            for k in 1:ki-1; acc += VR[row, k] * xr[k]; end
                            VR[row, ki] = acc
                        end
                    end
                    ii = _iamax(view(VR, :, ki), n)
                    remax = ONE / abs(VR[ii, ki])
                    for k in 1:n; VR[k, ki] *= remax; end
                end
            else
                # ---- complex right eigenvector (pair at columns ki-1, ki) ----
                if abs(T[ki-1, ki]) >= abs(T[ki, ki-1])
                    xr[ki-1] = ONE
                    xi[ki] = wi / T[ki-1, ki]
                else
                    xr[ki-1] = -wi / T[ki, ki-1]
                    xi[ki] = ONE
                end
                xr[ki] = ZERO; xi[ki-1] = ZERO
                for k in 1:ki-2
                    xr[k] = -xr[ki-1] * T[k, ki-1]
                    xi[k] = -xi[ki] * T[k, ki]
                end
                jnxt = ki - 2
                j = ki - 2
                while j >= 1
                    if j > jnxt
                        j -= 1; continue
                    end
                    j1 = j; j2 = j; jnxt = j - 1
                    if j > 1 && T[j, j-1] != ZERO
                        j1 = j - 1; jnxt = j - 2
                    end
                    if j1 == j2
                        x11, _, x12, _, scale, xnorm =
                            _dlaln2(1, 2, smin, T[j, j], ZERO, ZERO, ZERO, xr[j], ZERO, xi[j], ZERO, wr, wi)
                        if xnorm > ONE && cnorm[j] > bignum / xnorm
                            x11 /= xnorm; x12 /= xnorm; scale /= xnorm
                        end
                        if scale != ONE
                            for k in 1:ki; xr[k] *= scale; xi[k] *= scale; end
                        end
                        xr[j] = x11; xi[j] = x12
                        for k in 1:j-1; xr[k] -= x11 * T[k, j]; end
                        for k in 1:j-1; xi[k] -= x12 * T[k, j]; end
                    else
                        x11, x21, x12, x22, scale, xnorm =
                            _dlaln2(2, 2, smin, T[j-1, j-1], T[j-1, j], T[j, j-1], T[j, j],
                                    xr[j-1], xr[j], xi[j-1], xi[j], wr, wi)
                        if xnorm > ONE
                            beta = max(cnorm[j-1], cnorm[j])
                            if beta > bignum / xnorm
                                rec = ONE / xnorm
                                x11 *= rec; x12 *= rec; x21 *= rec; x22 *= rec; scale *= rec
                            end
                        end
                        if scale != ONE
                            for k in 1:ki; xr[k] *= scale; xi[k] *= scale; end
                        end
                        xr[j-1] = x11; xr[j] = x21; xi[j-1] = x12; xi[j] = x22
                        for k in 1:j-2; xr[k] -= x11 * T[k, j-1]; end
                        for k in 1:j-2; xr[k] -= x21 * T[k, j]; end
                        for k in 1:j-2; xi[k] -= x12 * T[k, j-1]; end
                        for k in 1:j-2; xi[k] -= x22 * T[k, j]; end
                    end
                    j = j1 - 1
                end
                # copy and normalize the pair
                if !over
                    for k in 1:ki; VR[k, is-1] = xr[k]; VR[k, is] = xi[k]; end
                    emax = ZERO
                    for k in 1:ki; emax = max(emax, abs(VR[k, is-1]) + abs(VR[k, is])); end
                    remax = ONE / emax
                    for k in 1:ki; VR[k, is-1] *= remax; VR[k, is] *= remax; end
                    for k in ki+1:n; VR[k, is-1] = ZERO; VR[k, is] = ZERO; end
                else
                    if ki > 2
                        for row in 1:n
                            accr = xr[ki-1] * VR[row, ki-1]
                            acci = xi[ki] * VR[row, ki]
                            for k in 1:ki-2
                                accr += VR[row, k] * xr[k]
                                acci += VR[row, k] * xi[k]
                            end
                            VR[row, ki-1] = accr; VR[row, ki] = acci
                        end
                    else
                        for row in 1:n
                            VR[row, ki-1] *= xr[ki-1]
                            VR[row, ki] *= xi[ki]
                        end
                    end
                    emax = ZERO
                    for k in 1:n; emax = max(emax, abs(VR[k, ki-1]) + abs(VR[k, ki])); end
                    remax = ONE / emax
                    for k in 1:n; VR[k, ki-1] *= remax; VR[k, ki] *= remax; end
                end
            end
            is -= 1
            ip != 0 && (is -= 1)
        end
        ip == 1 && (ip = 0)
        ip == -1 && (ip = 1)
        ki -= 1
    end
    return VR
end

# ── ZTREVC right eigenvectors, COMPLEX triangular T ───────────────────────────────────────────────────
# Guarded back-substitution of (T(1:ki-1,1:ki-1) − T(ki,ki))·x = −T(1:ki-1,ki), with the diagonal
# perturbed to SMIN when a difference underflows (the ztrevc correctness guard for clustered spectra).
function _ztrevc_right!(over::Bool, T::AbstractMatrix{C}, VR::AbstractMatrix{C}) where {C<:Complex}
    R = real(C)
    n = size(T, 1)
    n == 0 && return VR
    cabs1(z) = abs(real(z)) + abs(imag(z))
    unfl = _dtrevc_safmin(R)
    ulp = eps(R)
    smlnum = unfl * (R(n) / ulp)
    x = zeros(C, n)
    is = n
    @inbounds for ki in n:-1:1
        smin = max(ulp * cabs1(T[ki, ki]), smlnum)
        for k in 1:ki-1; x[k] = -T[k, ki]; end
        x[ki] = one(C)
        λ = T[ki, ki]
        # back-substitution: (T[1:ki-1,1:ki-1] − λ)·x = rhs
        for j in ki-1:-1:1
            djj = T[j, j] - λ
            cabs1(djj) < smin && (djj = Complex(smin, zero(R)))
            xj = x[j] / djj
            x[j] = xj
            for k in 1:j-1
                x[k] -= xj * T[k, j]
            end
        end
        if !over
            for k in 1:ki; VR[k, is] = x[k]; end
            ii = 1; best = cabs1(VR[1, is])
            for k in 2:ki
                c = cabs1(VR[k, is]); c > best && (best = c; ii = k)
            end
            remax = one(R) / cabs1(VR[ii, is])
            for k in 1:ki; VR[k, is] *= remax; end
            for k in ki+1:n; VR[k, is] = zero(C); end
            is -= 1
        else
            if ki > 1
                for row in 1:n
                    acc = x[ki] * VR[row, ki]
                    for k in 1:ki-1; acc += VR[row, k] * x[k]; end
                    VR[row, ki] = acc
                end
            end
            ii = 1; best = cabs1(VR[1, ki])
            for k in 2:n
                c = cabs1(VR[k, ki]); c > best && (best = c; ii = k)
            end
            remax = one(R) / cabs1(VR[ii, ki])
            for k in 1:n; VR[k, ki] *= remax; end
        end
    end
    return VR
end

"""
    trevc!(side, howmny, T, VL, VR) -> VR

Right eigenvectors of a (quasi-)upper-triangular Schur form `T` by back-substitution
(LAPACK `dtrevc`/`ztrevc`, `SIDE='R'`).

- `howmny='A'`: `VR` (n×n) is overwritten with the right eigenvectors of `T` (`T·vᵣ = λ·vᵣ`).
- `howmny='B'`: back-transform — `VR` holds the Schur vectors `Z` on input and receives the right
  eigenvectors of the original matrix `A = Z·T·Zᴴ`.

`VL` is accepted for signature compatibility and ignored (left eigenvectors, `side∈{'L','B'}`, are a
documented follow-up). For real `T`, a complex-conjugate eigenvector pair occupies two consecutive
columns as (real, imag) parts (LAPACK real convention). Generic over `T<:Real` and `T<:Complex`.
"""
function trevc!(side::AbstractChar, howmny::AbstractChar, T::AbstractMatrix,
        VL, VR::AbstractMatrix)
    side === 'R' || throw(ArgumentError("trevc!: only side='R' (right eigenvectors) is implemented"))
    (howmny === 'A' || howmny === 'B') ||
        throw(ArgumentError("trevc!: howmny must be 'A' (vectors of T) or 'B' (back-transform)"))
    over = howmny === 'B'
    n = size(T, 1)
    size(T, 2) == n || throw(DimensionMismatch("trevc!: T must be square"))
    if !over
        (size(VR, 1) == n && size(VR, 2) >= n) ||
            throw(DimensionMismatch("trevc!: VR must be at least n×n for howmny='A'"))
    end
    return _trevc_dispatch!(over, T, VR)
end

_trevc_dispatch!(over::Bool, T::AbstractMatrix{<:Real}, VR::AbstractMatrix{<:Real}) =
    _dtrevc_right!(over, T, VR)
_trevc_dispatch!(over::Bool, T::AbstractMatrix{<:Complex}, VR::AbstractMatrix{<:Complex}) =
    _ztrevc_right!(over, T, VR)
