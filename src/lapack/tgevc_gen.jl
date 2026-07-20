# LAPACK GENERALIZED eigenvector kernel (companion to qz.jl):
#   tgevc — generalized eigenvectors of a (quasi-)upper-triangular pencil (S,P) by back-substitution.
# CORRECTNESS-FIRST port of Reference-LAPACK dtgevc/ztgevc, SIDE='R' (right eigenvectors — the common
# case, needed by `eigen(A,B)`). Requires qz.jl included first (uses its _lag2/_ladiv/_zladiv/_qz_safmin).
#
# HOWMNY: 'A' overwrite VR with eigenvectors of the pencil (S,P); 'B' back-transform — VR holds the
# right generalized Schur vectors Z on input and receives the eigenvectors of the original pencil (Z·x).
# Real path uses the general DLALN2 1×1/2×2 solve; complex path is guarded triangular back-substitution.
# Generic over T (s/d/c/z). Left eigenvectors (SIDE='L') are a documented follow-up.

# ── DLALN2 (general: solves (ca·A − w·D)·X = s·B, A na×na, D=diag(d1,d2)), Reference-LAPACK dlaln2.f ────
const _QZLN_IPIVOT = ((1, 2, 3, 4), (2, 1, 4, 3), (3, 4, 1, 2), (4, 3, 2, 1))
const _QZLN_ZSWAP = (false, false, true, true)
const _QZLN_RSWAP = (false, true, false, true)

# Returns (x11, x21, x12, x22, scale, xnorm, info). Only the used entries are meaningful.
function _laln2(
        ltrans::Bool, na::Int, nw::Int, smin::R, ca::R, a11::R, a12::R, a21::R, a22::R,
        d1::R, d2::R, b11::R, b21::R, b12::R, b22::R, wr::R, wi::R
    ) where {R <: Real}
    ONE = one(R); ZERO = zero(R)
    smlnum = R(2) * _qz_safmin(R); bignum = ONE / smlnum; smini = max(smin, smlnum)
    scale = ONE; info = 0
    x11 = ZERO; x21 = ZERO; x12 = ZERO; x22 = ZERO; xnorm = ZERO
    if na == 1
        if nw == 1
            csr = ca * a11 - wr * d1; cnorm = abs(csr)
            if cnorm < smini
                csr = smini; cnorm = smini; info = 1
            end
            bnorm = abs(b11)
            if cnorm < ONE && bnorm > ONE
                bnorm > bignum * cnorm && (scale = ONE / bnorm)
            end
            x11 = (b11 * scale) / csr; xnorm = abs(x11)
        else
            csr = ca * a11 - wr * d1; csi = -wi * d1; cnorm = abs(csr) + abs(csi)
            if cnorm < smini
                csr = smini; csi = ZERO; cnorm = smini; info = 1
            end
            bnorm = abs(b11) + abs(b12)
            if cnorm < ONE && bnorm > ONE
                bnorm > bignum * cnorm && (scale = ONE / bnorm)
            end
            x11, x12 = _ladiv(scale * b11, scale * b12, csr, csi)
            xnorm = abs(x11) + abs(x12)
        end
    else
        cr11 = ca * a11 - wr * d1; cr22 = ca * a22 - wr * d2
        if ltrans
            cr12 = ca * a21; cr21 = ca * a12
        else
            cr21 = ca * a21; cr12 = ca * a12
        end
        crv = (cr11, cr21, cr12, cr22)
        if nw == 1
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
                x11 = temp * b11; x21 = temp * b21; xnorm = temp * bnorm; info = 1
                return x11, x21, x12, x22, scale, xnorm, info
            end
            piv = _QZLN_IPIVOT[icmax]
            ur11 = crv[icmax]; cr21v = crv[piv[2]]; ur12 = crv[piv[3]]; cr22v = crv[piv[4]]
            ur11r = ONE / ur11; lr21 = ur11r * cr21v; ur22 = cr22v - ur12 * lr21
            abs(ur22) < smini && (ur22 = smini; info = 1)
            if _QZLN_RSWAP[icmax]
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
            if _QZLN_ZSWAP[icmax]
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
            ci11 = -wi * d1; ci22 = -wi * d2
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
                xnorm = temp * bnorm; info = 1
                return x11, x21, x12, x22, scale, xnorm, info
            end
            piv = _QZLN_IPIVOT[icmax]
            ur11 = crv[icmax]; ui11 = civ[icmax]
            cr21v = crv[piv[2]]; ci21v = civ[piv[2]]
            ur12 = crv[piv[3]]; ui12 = civ[piv[3]]
            cr22v = crv[piv[4]]; ci22v = civ[piv[4]]
            local ur11r, ui11r, lr21, li21, ur12s, ui12s, ur22, ui22
            if icmax == 1 || icmax == 4
                if abs(ur11) > abs(ui11)
                    temp = ui11 / ur11; ur11r = ONE / (ur11 * (ONE + temp^2)); ui11r = -temp * ur11r
                else
                    temp = ur11 / ui11; ui11r = -ONE / (ui11 * (ONE + temp^2)); ur11r = -temp * ui11r
                end
                lr21 = cr21v * ur11r; li21 = cr21v * ui11r
                ur12s = ur12 * ur11r; ui12s = ur12 * ui11r
                ur22 = cr22v - ur12 * lr21; ui22 = ci22v - ur12 * li21
            else
                ur11r = ONE / ur11; ui11r = ZERO
                lr21 = cr21v * ur11r; li21 = ci21v * ur11r
                ur12s = ur12 * ur11r; ui12s = ui12 * ur11r
                ur22 = cr22v - ur12 * lr21 + ui12 * li21
                ui22 = -ur12 * li21 - ui12 * lr21
            end
            u22abs = abs(ur22) + abs(ui22)
            if u22abs < smini
                ur22 = smini; ui22 = ZERO; info = 1
            end
            if _QZLN_RSWAP[icmax]
                br2 = b11; br1 = b21; bi2 = b12; bi1 = b22
            else
                br1 = b11; br2 = b21; bi1 = b12; bi2 = b22
            end
            br2 = br2 - lr21 * br1 + li21 * bi1
            bi2 = bi2 - li21 * br1 - lr21 * bi1
            bbnd = max((abs(br1) + abs(bi1)) * (u22abs * (abs(ur11r) + abs(ui11r))), abs(br2) + abs(bi2))
            if bbnd > ONE && u22abs < ONE
                if bbnd >= bignum * u22abs
                    scale = ONE / bbnd
                    br1 *= scale; bi1 *= scale; br2 *= scale; bi2 *= scale
                end
            end
            xr2, xi2 = _ladiv(br2, bi2, ur22, ui22)
            xr1 = ur11r * br1 - ui11r * bi1 - ur12s * xr2 + ui12s * xi2
            xi1 = ui11r * br1 + ur11r * bi1 - ui12s * xr2 - ur12s * xi2
            if _QZLN_ZSWAP[icmax]
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
    return x11, x21, x12, x22, scale, xnorm, info
end

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
#  tgevc (REAL) — right generalized eigenvectors (dtgevc, SIDE='R')
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
function _tgevc_right!(
        ilback::Bool, S::AbstractMatrix{R}, P::AbstractMatrix{R},
        VR::AbstractMatrix{R}
    ) where {R <: Real}
    n = size(S, 1)
    ONE = one(R); ZERO = zero(R); SAFETY = R(100)
    safmin = _qz_safmin(R); ulp = eps(R)
    small = safmin * n / ulp; big = ONE / small; bignum = ONE / (safmin * n)
    W = zeros(R, n, 6)   # cols: 1=Snorm(WORK j) 2=Pnorm(WORK N+j) 3=x.re 4=x.im 5=bt.re 6=bt.im
    # column 1-norms of strictly-upper (excluding diagonal blocks)
    anorm = abs(S[1, 1]); n > 1 && (anorm += abs(S[2, 1]))
    bnorm = abs(P[1, 1])
    @inbounds for j in 2:n
        temp = ZERO; temp2 = ZERO
        iend = S[j, j - 1] == ZERO ? j - 1 : j - 2
        for i in 1:iend
            temp += abs(S[i, j]); temp2 += abs(P[i, j])
        end
        W[j, 1] = temp; W[j, 2] = temp2
        for i in (iend + 1):min(j + 1, n)
            temp += abs(S[i, j]); temp2 += abs(P[i, j])
        end
        anorm = max(anorm, temp); bnorm = max(bnorm, temp2)
    end
    ascale = ONE / max(anorm, safmin); bscale = ONE / max(bnorm, safmin)
    ieig = n + 1
    ilcplx = false
    je = n
    @inbounds while je >= 1
        if ilcplx
            ilcplx = false; je -= 1; continue
        end
        nw = 1
        if je > 1 && S[je, je - 1] != ZERO
            ilcplx = true; nw = 2
        end
        # singular pencil → unit eigenvector
        if !ilcplx && abs(S[je, je]) <= safmin && abs(P[je, je]) <= safmin
            ieig -= 1
            for jr in 1:n
                VR[jr, ieig] = ZERO
            end
            VR[ieig, ieig] = ONE
            je -= 1; continue
        end
        # clear work x columns
        for jw in 0:(nw - 1), jr in 1:n
            W[jr, 3 + jw] = ZERO
        end
        local acoef, bcoefr, bcoefi, acoefa, bcoefa
        if !ilcplx
            temp = ONE / max(abs(S[je, je]) * ascale, abs(P[je, je]) * bscale, safmin)
            salfar = (temp * S[je, je]) * ascale
            sbeta = (temp * P[je, je]) * bscale
            acoef = sbeta * ascale
            bcoefr = salfar * bscale
            bcoefi = ZERO
            scale = ONE
            lsa = abs(sbeta) >= safmin && abs(acoef) < small
            lsb = abs(salfar) >= safmin && abs(bcoefr) < small
            lsa && (scale = (small / abs(sbeta)) * min(anorm, big))
            lsb && (scale = max(scale, (small / abs(salfar)) * min(bnorm, big)))
            if lsa || lsb
                scale = min(scale, ONE / (safmin * max(ONE, abs(acoef), abs(bcoefr))))
                acoef = lsa ? ascale * (scale * sbeta) : scale * acoef
                bcoefr = lsb ? bscale * (scale * salfar) : scale * bcoefr
            end
            acoefa = abs(acoef); bcoefa = abs(bcoefr)
            W[je, 3] = ONE; xmax = ONE
            for jr in 1:(je - 1)
                W[jr, 3] = bcoefr * P[jr, je] - acoef * S[jr, je]
            end
        else
            s1, tmp1, bcoefr, tmp2, bcoefi = _lag2(
                S[je - 1, je - 1], S[je, je - 1], S[je - 1, je],
                S[je, je], P[je - 1, je - 1], P[je - 1, je], P[je, je], safmin * SAFETY
            )
            acoef = s1
            bcoefi == ZERO && return je - 1   # info
            acoefa = abs(acoef); bcoefa = abs(bcoefr) + abs(bcoefi)
            scale = ONE
            (acoefa * ulp < safmin && acoefa >= safmin) && (scale = (safmin / ulp) / acoefa)
            (bcoefa * ulp < safmin && bcoefa >= safmin) && (scale = max(scale, (safmin / ulp) / bcoefa))
            safmin * acoefa > ascale && (scale = ascale / (safmin * acoefa))
            safmin * bcoefa > bscale && (scale = min(scale, bscale / (safmin * bcoefa)))
            if scale != ONE
                acoef = scale * acoef; acoefa = abs(acoef)
                bcoefr = scale * bcoefr; bcoefi = scale * bcoefi
                bcoefa = abs(bcoefr) + abs(bcoefi)
            end
            temp = acoef * S[je, je - 1]
            temp2r = acoef * S[je, je] - bcoefr * P[je, je]
            temp2i = -bcoefi * P[je, je]
            if abs(temp) >= abs(temp2r) + abs(temp2i)
                W[je, 3] = ONE; W[je, 4] = ZERO
                W[je - 1, 3] = -temp2r / temp; W[je - 1, 4] = -temp2i / temp
            else
                W[je - 1, 3] = ONE; W[je - 1, 4] = ZERO
                temp = acoef * S[je - 1, je]
                W[je, 3] = (bcoefr * P[je - 1, je - 1] - acoef * S[je - 1, je - 1]) / temp
                W[je, 4] = bcoefi * P[je - 1, je - 1] / temp
            end
            xmax = max(abs(W[je, 3]) + abs(W[je, 4]), abs(W[je - 1, 3]) + abs(W[je - 1, 4]))
            creala = acoef * W[je - 1, 3]; cimaga = acoef * W[je - 1, 4]
            crealb = bcoefr * W[je - 1, 3] - bcoefi * W[je - 1, 4]
            cimagb = bcoefi * W[je - 1, 3] + bcoefr * W[je - 1, 4]
            cre2a = acoef * W[je, 3]; cim2a = acoef * W[je, 4]
            cre2b = bcoefr * W[je, 3] - bcoefi * W[je, 4]
            cim2b = bcoefi * W[je, 3] + bcoefr * W[je, 4]
            for jr in 1:(je - 2)
                W[jr, 3] = -creala * S[jr, je - 1] + crealb * P[jr, je - 1] - cre2a * S[jr, je] + cre2b * P[jr, je]
                W[jr, 4] = -cimaga * S[jr, je - 1] + cimagb * P[jr, je - 1] - cim2a * S[jr, je] + cim2b * P[jr, je]
            end
        end
        dmin = max(ulp * acoefa * anorm, ulp * bcoefa * bnorm, safmin)
        # Columnwise triangular solve of (a S - b P) x = 0
        il2by2 = false
        j = je - nw
        while j >= 1
            if !il2by2 && j > 1 && S[j, j - 1] != ZERO
                il2by2 = true; j -= 1; continue    # handle as 2×2 next (when it is j:j+1)
            end
            na = il2by2 ? 2 : 1
            bd1 = P[j, j]; bd2 = il2by2 ? P[j + 1, j + 1] : ZERO
            x11, x21, x12, x22, scale, xnorm, iinfo = _laln2(
                false, na, nw, dmin, acoef,
                S[j, j], S[j, j + 1], S[j + 1, j], S[j + 1, j + 1], bd1, bd2,
                W[j, 3], W[j + 1, 3], W[j, 4], W[j + 1, 4], bcoefr, bcoefi
            )
            if scale < ONE
                for jw in 0:(nw - 1), jr in 1:je
                    W[jr, 3 + jw] *= scale
                end
            end
            xmax = max(scale * xmax, xnorm)
            # store solution
            W[j, 3] = x11; nw == 2 && (W[j, 4] = x12)
            if na == 2
                W[j + 1, 3] = x21; nw == 2 && (W[j + 1, 4] = x22)
            end
            if j > 1
                xscale = ONE / max(ONE, xmax)
                temp = acoefa * W[j, 1] + bcoefa * W[j, 2]
                il2by2 && (temp = max(temp, acoefa * W[j + 1, 1] + bcoefa * W[j + 1, 2]))
                temp = max(temp, acoefa, bcoefa)
                if temp > bignum * xscale
                    for jw in 0:(nw - 1), jr in 1:je
                        W[jr, 3 + jw] *= xscale
                    end
                    xmax *= xscale
                end
                for ja in 1:na
                    if ilcplx
                        creala = acoef * W[j + ja - 1, 3]; cimaga = acoef * W[j + ja - 1, 4]
                        crealb = bcoefr * W[j + ja - 1, 3] - bcoefi * W[j + ja - 1, 4]
                        cimagb = bcoefi * W[j + ja - 1, 3] + bcoefr * W[j + ja - 1, 4]
                        for jr in 1:(j - 1)
                            W[jr, 3] += -creala * S[jr, j + ja - 1] + crealb * P[jr, j + ja - 1]
                            W[jr, 4] += -cimaga * S[jr, j + ja - 1] + cimagb * P[jr, j + ja - 1]
                        end
                    else
                        creala = acoef * W[j + ja - 1, 3]; crealb = bcoefr * W[j + ja - 1, 3]
                        for jr in 1:(j - 1)
                            W[jr, 3] += -creala * S[jr, j + ja - 1] + crealb * P[jr, j + ja - 1]
                        end
                    end
                end
            end
            il2by2 = false
            j -= 1
        end
        # Copy eigenvector to VR (back-transform if ilback)
        ieig -= nw
        if ilback
            for jw in 0:(nw - 1)
                for jr in 1:n
                    W[jr, 5 + jw] = W[1, 3 + jw] * VR[jr, 1]
                end
                for jc in 2:je, jr in 1:n
                    W[jr, 5 + jw] += W[jc, 3 + jw] * VR[jr, jc]
                end
            end
            for jw in 0:(nw - 1), jr in 1:n
                VR[jr, ieig + jw] = W[jr, 5 + jw]
            end
            iend = n
        else
            for jw in 0:(nw - 1), jr in 1:n
                VR[jr, ieig + jw] = W[jr, 3 + jw]
            end
            iend = je
        end
        # scale eigenvector
        xmaxv = ZERO
        if ilcplx
            for jj in 1:iend
                xmaxv = max(xmaxv, abs(VR[jj, ieig]) + abs(VR[jj, ieig + 1]))
            end
        else
            for jj in 1:iend
                xmaxv = max(xmaxv, abs(VR[jj, ieig]))
            end
        end
        if xmaxv > safmin
            xscale = ONE / xmaxv
            for jw in 0:(nw - 1), jr in 1:iend
                VR[jr, ieig + jw] *= xscale
            end
        end
        je -= 1
    end
    return 0
end

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
#  tgevc (COMPLEX) — right generalized eigenvectors (ztgevc, SIDE='R')
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
function _tgevc_right!(
        ilback::Bool, S::AbstractMatrix{C}, P::AbstractMatrix{C},
        VR::AbstractMatrix{C}
    ) where {C <: Complex}
    R = real(C); n = size(S, 1)
    ONE = one(R); ZERO = zero(R); CZERO = zero(C); CONE = one(C)
    cabs1(z) = abs(real(z)) + abs(imag(z))
    safmin = _qz_safmin(R); ulp = eps(R)
    small = safmin * n / ulp; big = ONE / small; bignum = ONE / (safmin * n)
    work = zeros(C, 2n)               # WORK(1:n)=x/sums, WORK(n+1:2n)=back-transform
    rwork = zeros(R, 2n)              # RWORK(j)=Snorm, RWORK(n+j)=Pnorm
    anorm = cabs1(S[1, 1]); bnorm = cabs1(P[1, 1])
    @inbounds for j in 2:n
        s = ZERO; b = ZERO
        for i in 1:(j - 1)
            s += cabs1(S[i, j]); b += cabs1(P[i, j])
        end
        rwork[j] = s; rwork[n + j] = b
        anorm = max(anorm, s + cabs1(S[j, j])); bnorm = max(bnorm, b + cabs1(P[j, j]))
    end
    ascale = ONE / max(anorm, safmin); bscale = ONE / max(bnorm, safmin)
    ieig = n + 1
    @inbounds for je in n:-1:1
        ieig -= 1
        if cabs1(S[je, je]) <= safmin && abs(real(P[je, je])) <= safmin
            for jr in 1:n
                VR[jr, ieig] = CZERO
            end
            VR[ieig, ieig] = CONE
            continue
        end
        temp = ONE / max(cabs1(S[je, je]) * ascale, abs(real(P[je, je])) * bscale, safmin)
        salpha = (temp * S[je, je]) * ascale
        sbeta = (temp * real(P[je, je])) * bscale
        acoeff = sbeta * ascale
        bcoeff = salpha * bscale
        lsa = abs(sbeta) >= safmin && abs(acoeff) < small
        lsb = cabs1(salpha) >= safmin && cabs1(bcoeff) < small
        scale = ONE
        lsa && (scale = (small / abs(sbeta)) * min(anorm, big))
        lsb && (scale = max(scale, (small / cabs1(salpha)) * min(bnorm, big)))
        if lsa || lsb
            scale = min(scale, ONE / (safmin * max(ONE, abs(acoeff), cabs1(bcoeff))))
            acoeff = lsa ? ascale * (scale * sbeta) : scale * acoeff
            bcoeff = lsb ? bscale * (scale * salpha) : scale * bcoeff
        end
        acoefa = abs(acoeff); bcoefa = cabs1(bcoeff)
        xmax = ONE
        for jr in 1:n
            work[jr] = CZERO
        end
        work[je] = CONE
        dmin = max(ulp * acoefa * anorm, ulp * bcoefa * bnorm, safmin)
        for jr in 1:(je - 1)
            work[jr] = acoeff * S[jr, je] - bcoeff * P[jr, je]
        end
        work[je] = CONE
        for j in (je - 1):-1:1
            d = acoeff * S[j, j] - bcoeff * P[j, j]
            cabs1(d) <= dmin && (d = complex(dmin))
            if cabs1(d) < ONE
                if cabs1(work[j]) >= bignum * cabs1(d)
                    temp = ONE / cabs1(work[j])
                    for jr in 1:je
                        work[jr] *= temp
                    end
                end
            end
            work[j] = _zladiv(-work[j], d)
            if j > 1
                if cabs1(work[j]) > ONE
                    temp = ONE / cabs1(work[j])
                    if acoefa * rwork[j] + bcoefa * rwork[n + j] >= bignum * temp
                        for jr in 1:je
                            work[jr] *= temp
                        end
                    end
                end
                ca = acoeff * work[j]; cb = bcoeff * work[j]
                for jr in 1:(j - 1)
                    work[jr] += ca * S[jr, j] - cb * P[jr, j]
                end
            end
        end
        if ilback
            for jr in 1:n
                acc = CZERO
                for jc in 1:je
                    acc += VR[jr, jc] * work[jc]
                end
                work[n + jr] = acc
            end
            isrc = n; iend = n
        else
            isrc = 0; iend = je
        end
        xmaxv = ZERO
        for jr in 1:iend
            xmaxv = max(xmaxv, cabs1(work[isrc + jr]))
        end
        if xmaxv > safmin
            temp = ONE / xmaxv
            for jr in 1:iend
                VR[jr, ieig] = temp * work[isrc + jr]
            end
        else
            iend = 0
        end
        for jr in (iend + 1):n
            VR[jr, ieig] = CZERO
        end
    end
    return 0
end

"""
    tgevc!(side, howmny, S, P, VL, VR) -> info

Generalized eigenvectors of a (quasi-)upper-triangular pencil `(S,P)` (LAPACK dtgevc/ztgevc). Only
`side='R'` (right eigenvectors) is implemented — the common case for `eigen(A,B)`. `howmny='A'` writes
the eigenvectors of `(S,P)` into `VR`; `howmny='B'` back-transforms — `VR` must hold the right
generalized Schur vectors `Z` on entry and receives `Z·x` (eigenvectors of the original pencil).
Generic over `T` (s/d/c/z). Returns `info` (0 = success). `VL` is accepted but unused (`side='R'`).
"""
function tgevc!(
        side::AbstractChar, howmny::AbstractChar, S::AbstractMatrix{T}, P::AbstractMatrix{T},
        VL::AbstractMatrix{T}, VR::AbstractMatrix{T}
    ) where {T <: Number}
    side === 'R' || throw(ArgumentError("tgevc!: only side='R' (right eigenvectors) is implemented"))
    (howmny === 'A' || howmny === 'B') || throw(ArgumentError("tgevc!: howmny must be 'A' or 'B'"))
    ilback = howmny === 'B'
    return _tgevc_right!(ilback, S, P, VR)
end
