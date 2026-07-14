# LAPACK dqds (differential quotient-difference with shifts) for bidiagonal singular VALUES.
# Faithful port of dlasq1/2/3/4/5/6 (netlib reference). This is the algorithm OpenBLAS's dbdsqr
# delegates to when no singular vectors are wanted (verified: dbdsqr → DLASQ1 for ncvt=nru=ncc=0):
# a sqrt-free O(n)-per-step recurrence (add + div + fma), vs the QR sweep's 2 sqrt + 2 div per k.
# It is why values-only SVD is ~1.7× faster than our Golub-Kahan QR bdsqr! — an algorithm gap, not a
# tuning one. Float64-only (the bidiagonal is always real, even for complex A). IEEE arithmetic assumed
# (all target hardware is IEEE-conformant → we keep only the IEEE dlasq5 path). On any non-convergence
# (info≠0) the caller falls back to the QR bdsqr! on the still-intact (abs-d, e) bidiagonal.

# Persistent dqds scalars threaded between DLASQ3 calls (the reference's "saved between calls" args).
mutable struct _DqdsState
    ttype::Int
    dmin1::Float64; dmin2::Float64; dn::Float64; dn1::Float64; dn2::Float64
    g::Float64; tau::Float64
    nfail::Int; iter::Int; ndiv::Int
end
_DqdsState() = _DqdsState(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0)

# --- DLASCL('G'): scale A[lo:hi] from CFROM to CTO with the reference's over/underflow-safe stepping ---
function _dlascl!(A::AbstractVector{Float64}, cfrom::Float64, cto::Float64, lo::Int, hi::Int)
    smlnum = floatmin(Float64); bignum = 1.0 / smlnum
    cfromc = cfrom; ctoc = cto
    @inbounds while true
        cfrom1 = cfromc * smlnum
        local mul::Float64, done::Bool
        if cfrom1 == cfromc
            mul = ctoc / cfromc; done = true                 # cfromc is inf
        else
            cto1 = ctoc / bignum
            if cto1 == ctoc
                mul = ctoc; done = true; cfromc = 1.0        # ctoc is 0 or inf
            elseif abs(cfrom1) > abs(ctoc) && ctoc != 0.0
                mul = smlnum; done = false; cfromc = cfrom1
            elseif abs(cto1) > abs(cfromc)
                mul = bignum; done = false; ctoc = cto1
            else
                mul = ctoc / cfromc; done = true
            end
        end
        for i in lo:hi; A[i] *= mul; end
        done && break
    end
    return A
end

# --- DLAS2: singular values (ssmin, ssmax) of the 2×2 upper triangular [[f,g],[0,h]] ---
@inline function _dlas2(f::Float64, g::Float64, h::Float64)
    fa = abs(f); ga = abs(g); ha = abs(h)
    fhmn = min(fa, ha); fhmx = max(fa, ha)
    if fhmn == 0.0
        ssmin = 0.0
        if fhmx == 0.0
            ssmax = ga
        else
            ssmax = max(fhmx, ga) * sqrt(1.0 + (min(fhmx, ga) / max(fhmx, ga))^2)
        end
    else
        if ga < fhmx
            as = 1.0 + fhmn / fhmx
            at = (fhmx - fhmn) / fhmx
            au = (ga / fhmx)^2
            c = 2.0 / (sqrt(as * as + au) + sqrt(at * at + au))
            ssmin = fhmn * c; ssmax = fhmx / c
        else
            au = fhmx / ga
            if au == 0.0
                ssmin = (fhmn * fhmx) / ga; ssmax = ga
            else
                as = 1.0 + fhmn / fhmx
                at = (fhmx - fhmn) / fhmx
                c = 1.0 / (sqrt(1.0 + (as * au)^2) + sqrt(1.0 + (at * au)^2))
                ssmin = (fhmn * c) * au; ssmin = ssmin + ssmin
                ssmax = ga / (c + c)
            end
        end
    end
    return ssmin, ssmax
end

# --- In-place descending sort of A[lo:hi] (heapsort: trim-safe, no alloc, O(n log n)) ---
@inline function _siftdown!(A::AbstractVector{Float64}, lo::Int, start::Int, len::Int)
    root = start
    @inbounds while 2 * root <= len
        child = 2 * root
        (child < len && A[lo + child - 1] < A[lo + child]) && (child += 1)
        if A[lo + root - 1] < A[lo + child - 1]
            A[lo + root - 1], A[lo + child - 1] = A[lo + child - 1], A[lo + root - 1]
            root = child
        else
            break
        end
    end
    return nothing
end
function _dlasrt_desc!(A::AbstractVector{Float64}, lo::Int, hi::Int)
    n = hi - lo + 1
    n <= 1 && return A
    @inbounds begin
        for start in (n ÷ 2):-1:1; _siftdown!(A, lo, start, n); end
        for endi in n:-1:2
            A[lo + endi - 1], A[lo] = A[lo], A[lo + endi - 1]
            _siftdown!(A, lo, 1, endi - 1)
        end
        i = lo; j = hi                                        # ascending → reverse to descending
        while i < j; A[i], A[j] = A[j], A[i]; i += 1; j -= 1; end
    end
    return A
end

# --- DLASQ6: one dqd step (zero shift, underflow-safe). Returns (dmin,dmin1,dmin2,dn,dnm1,dnm2) ---
function _dlasq6!(Z::Vector{Float64}, i0::Int, n0::Int, pp::Int)
    (n0 - i0 - 1) <= 0 && return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    safmin = floatmin(Float64)
    @inbounds begin
        j4 = 4 * i0 + pp - 3
        emin = Z[j4 + 4]; d = Z[j4]; dmin = d
        if pp == 0
            for j4 in 4*i0:4:4*(n0-3)
                Z[j4-2] = d + Z[j4-1]
                if Z[j4-2] == 0.0
                    Z[j4] = 0.0; d = Z[j4+1]; dmin = d; emin = 0.0
                elseif safmin*Z[j4+1] < Z[j4-2] && safmin*Z[j4-2] < Z[j4+1]
                    temp = Z[j4+1]/Z[j4-2]; Z[j4] = Z[j4-1]*temp; d = d*temp
                else
                    Z[j4] = Z[j4+1]*(Z[j4-1]/Z[j4-2]); d = Z[j4+1]*(d/Z[j4-2])
                end
                dmin = min(dmin, d); emin = min(emin, Z[j4])
            end
        else
            for j4 in 4*i0:4:4*(n0-3)
                Z[j4-3] = d + Z[j4]
                if Z[j4-3] == 0.0
                    Z[j4-1] = 0.0; d = Z[j4+2]; dmin = d; emin = 0.0
                elseif safmin*Z[j4+2] < Z[j4-3] && safmin*Z[j4-3] < Z[j4+2]
                    temp = Z[j4+2]/Z[j4-3]; Z[j4-1] = Z[j4]*temp; d = d*temp
                else
                    Z[j4-1] = Z[j4+2]*(Z[j4]/Z[j4-3]); d = Z[j4+2]*(d/Z[j4-3])
                end
                dmin = min(dmin, d); emin = min(emin, Z[j4-1])
            end
        end
        dnm2 = d; dmin2 = dmin
        j4 = 4*(n0-2) - pp; j4p2 = j4 + 2*pp - 1
        Z[j4-2] = dnm2 + Z[j4p2]
        if Z[j4-2] == 0.0
            Z[j4] = 0.0; dnm1 = Z[j4p2+2]; dmin = dnm1; emin = 0.0
        elseif safmin*Z[j4p2+2] < Z[j4-2] && safmin*Z[j4-2] < Z[j4p2+2]
            temp = Z[j4p2+2]/Z[j4-2]; Z[j4] = Z[j4p2]*temp; dnm1 = dnm2*temp
        else
            Z[j4] = Z[j4p2+2]*(Z[j4p2]/Z[j4-2]); dnm1 = Z[j4p2+2]*(dnm2/Z[j4-2])
        end
        dmin = min(dmin, dnm1); dmin1 = dmin
        j4 = j4 + 4; j4p2 = j4 + 2*pp - 1
        Z[j4-2] = dnm1 + Z[j4p2]
        if Z[j4-2] == 0.0
            Z[j4] = 0.0; dn = Z[j4p2+2]; dmin = dn; emin = 0.0
        elseif safmin*Z[j4p2+2] < Z[j4-2] && safmin*Z[j4-2] < Z[j4p2+2]
            temp = Z[j4p2+2]/Z[j4-2]; Z[j4] = Z[j4p2]*temp; dn = dnm1*temp
        else
            Z[j4] = Z[j4p2+2]*(Z[j4p2]/Z[j4-2]); dn = Z[j4p2+2]*(dnm1/Z[j4-2])
        end
        dmin = min(dmin, dn)
        Z[j4+2] = dn; Z[4*n0-pp] = emin
    end
    return (dmin, dmin1, dmin2, dn, dnm1, dnm2)
end

# --- DLASQ5: one dqds step with shift TAU (IEEE path). Returns (dmin,dmin1,dmin2,dn,dnm1,dnm2,tau) ---
function _dlasq5!(Z::Vector{Float64}, i0::Int, n0::Int, pp::Int, tau::Float64, sigma::Float64, epsm::Float64)
    (n0 - i0 - 1) <= 0 && return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, tau)
    dthresh = epsm * (sigma + tau)
    tau < dthresh * 0.5 && (tau = 0.0)
    zt = (tau == 0.0)                                         # zero-shift variant zeroes tiny d's
    @inbounds begin
        j4 = 4 * i0 + pp - 3
        emin = Z[j4 + 4]; d = Z[j4] - tau; dmin = d; dmin1 = -Z[j4]
        if pp == 0
            for j4 in 4*i0:4:4*(n0-3)
                Z[j4-2] = d + Z[j4-1]
                temp = Z[j4+1] / Z[j4-2]
                d = d*temp - tau
                (zt && d < dthresh) && (d = 0.0)
                dmin = min(dmin, d)
                Z[j4] = Z[j4-1]*temp
                emin = min(Z[j4], emin)
            end
        else
            for j4 in 4*i0:4:4*(n0-3)
                Z[j4-3] = d + Z[j4]
                temp = Z[j4+2] / Z[j4-3]
                d = d*temp - tau
                (zt && d < dthresh) && (d = 0.0)
                dmin = min(dmin, d)
                Z[j4-1] = Z[j4]*temp
                emin = min(Z[j4-1], emin)
            end
        end
        dnm2 = d; dmin2 = dmin
        j4 = 4*(n0-2) - pp; j4p2 = j4 + 2*pp - 1
        Z[j4-2] = dnm2 + Z[j4p2]
        Z[j4] = Z[j4p2+2]*(Z[j4p2]/Z[j4-2])
        dnm1 = Z[j4p2+2]*(dnm2/Z[j4-2]) - tau
        dmin = min(dmin, dnm1); dmin1 = dmin
        j4 = j4 + 4; j4p2 = j4 + 2*pp - 1
        Z[j4-2] = dnm1 + Z[j4p2]
        Z[j4] = Z[j4p2+2]*(Z[j4p2]/Z[j4-2])
        dn = Z[j4p2+2]*(dnm1/Z[j4-2]) - tau
        dmin = min(dmin, dn)
        Z[j4+2] = dn; Z[4*n0-pp] = emin
    end
    return (dmin, dmin1, dmin2, dn, dnm1, dnm2, tau)
end

# --- DLASQ4: choose the dqds shift. Returns (tau, ttype, g). Early exits keep the incoming tau. ---
function _dlasq4(Z::Vector{Float64}, i0::Int, n0::Int, pp::Int, n0in::Int, dmin::Float64,
        dmin1::Float64, dmin2::Float64, dn::Float64, dn1::Float64, dn2::Float64,
        tau::Float64, ttype::Int, g::Float64)
    cnst1 = 0.5630; cnst2 = 1.010; cnst3 = 1.050
    third = 0.3330
    if dmin <= 0.0
        return (-dmin, -1, g)
    end
    nn = 4 * n0 + pp
    s = 0.0
    @inbounds begin
        if n0in == n0
            if dmin == dn || dmin == dn1
                b1 = sqrt(Z[nn-3]) * sqrt(Z[nn-5])
                b2 = sqrt(Z[nn-7]) * sqrt(Z[nn-9])
                a2 = Z[nn-7] + Z[nn-5]
                if dmin == dn && dmin1 == dn1
                    gap2 = dmin2 - a2 - dmin2 * 0.25
                    if gap2 > 0.0 && gap2 > b2
                        gap1 = a2 - dn - (b2/gap2)*b2
                    else
                        gap1 = a2 - dn - (b1 + b2)
                    end
                    if gap1 > 0.0 && gap1 > b1
                        s = max(dn - (b1/gap1)*b1, 0.5*dmin); ttype = -2
                    else
                        s = 0.0
                        (dn > b1) && (s = dn - b1)
                        (a2 > (b1 + b2)) && (s = min(s, a2 - (b1 + b2)))
                        s = max(s, third * dmin); ttype = -3
                    end
                else
                    ttype = -4; s = 0.25 * dmin
                    if dmin == dn
                        gam = dn; a2 = 0.0
                        (Z[nn-5] > Z[nn-7]) && return (tau, ttype, g)
                        b2 = Z[nn-5] / Z[nn-7]; np = nn - 9
                    else
                        np = nn - 2*pp; gam = dn1
                        (Z[np-4] > Z[np-2]) && return (tau, ttype, g)
                        a2 = Z[np-4] / Z[np-2]
                        (Z[nn-9] > Z[nn-11]) && return (tau, ttype, g)
                        b2 = Z[nn-9] / Z[nn-11]; np = nn - 13
                    end
                    a2 = a2 + b2
                    i4 = np
                    while i4 >= 4*i0 - 1 + pp
                        (b2 == 0.0) && break
                        b1 = b2
                        (Z[i4] > Z[i4-2]) && return (tau, ttype, g)
                        b2 = b2*(Z[i4]/Z[i4-2]); a2 = a2 + b2
                        (100.0*max(b2, b1) < a2 || cnst1 < a2) && break
                        i4 -= 4
                    end
                    a2 = cnst3 * a2
                    (a2 < cnst1) && (s = gam*(1.0 - sqrt(a2))/(1.0 + a2))
                end
            elseif dmin == dn2
                ttype = -5; s = 0.25 * dmin
                np = nn - 2*pp
                b1 = Z[np-2]; b2 = Z[np-6]; gam = dn2
                (Z[np-8] > b2 || Z[np-4] > b1) && return (tau, ttype, g)
                a2 = (Z[np-8]/b2)*(1.0 + Z[np-4]/b1)
                if n0 - i0 > 2
                    b2 = Z[nn-13]/Z[nn-15]; a2 = a2 + b2
                    i4 = nn - 17
                    while i4 >= 4*i0 - 1 + pp
                        (b2 == 0.0) && break
                        b1 = b2
                        (Z[i4] > Z[i4-2]) && return (tau, ttype, g)
                        b2 = b2*(Z[i4]/Z[i4-2]); a2 = a2 + b2
                        (100.0*max(b2, b1) < a2 || cnst1 < a2) && break
                        i4 -= 4
                    end
                    a2 = cnst3 * a2
                end
                (a2 < cnst1) && (s = gam*(1.0 - sqrt(a2))/(1.0 + a2))
            else
                if ttype == -6
                    g = g + third*(1.0 - g)
                elseif ttype == -18
                    g = 0.25 * third
                else
                    g = 0.25
                end
                s = g * dmin; ttype = -6
            end
        elseif n0in == n0 + 1
            if dmin1 == dn1 && dmin2 == dn2
                ttype = -7; s = third * dmin1
                (Z[nn-5] > Z[nn-7]) && return (tau, ttype, g)
                b1 = Z[nn-5]/Z[nn-7]; b2 = b1
                if b2 != 0.0
                    i4 = 4*n0 - 9 + pp
                    while i4 >= 4*i0 - 1 + pp
                        a2 = b1
                        (Z[i4] > Z[i4-2]) && return (tau, ttype, g)
                        b1 = b1*(Z[i4]/Z[i4-2]); b2 = b2 + b1
                        (100.0*max(b1, a2) < b2) && break
                        i4 -= 4
                    end
                end
                b2 = sqrt(cnst3 * b2)
                a2 = dmin1 / (1.0 + b2*b2)
                gap2 = 0.5*dmin2 - a2
                if gap2 > 0.0 && gap2 > b2*a2
                    s = max(s, a2*(1.0 - cnst2*a2*(b2/gap2)*b2))
                else
                    s = max(s, a2*(1.0 - cnst2*b2)); ttype = -8
                end
            else
                s = 0.25 * dmin1
                (dmin1 == dn1) && (s = 0.5 * dmin1)
                ttype = -9
            end
        elseif n0in == n0 + 2
            if dmin2 == dn2 && 2.0*Z[nn-5] < Z[nn-7]
                ttype = -10; s = third * dmin2
                (Z[nn-5] > Z[nn-7]) && return (tau, ttype, g)
                b1 = Z[nn-5]/Z[nn-7]; b2 = b1
                if b2 != 0.0
                    i4 = 4*n0 - 9 + pp
                    while i4 >= 4*i0 - 1 + pp
                        (Z[i4] > Z[i4-2]) && return (tau, ttype, g)
                        b1 = b1*(Z[i4]/Z[i4-2]); b2 = b2 + b1
                        (100.0*b1 < b2) && break
                        i4 -= 4
                    end
                end
                b2 = sqrt(cnst3 * b2)
                a2 = dmin2 / (1.0 + b2*b2)
                gap2 = Z[nn-7] + Z[nn-9] - sqrt(Z[nn-11])*sqrt(Z[nn-9]) - a2
                if gap2 > 0.0 && gap2 > b2*a2
                    s = max(s, a2*(1.0 - cnst2*a2*(b2/gap2)*b2))
                else
                    s = max(s, a2*(1.0 - cnst2*b2))
                end
            else
                s = 0.25 * dmin2; ttype = -11
            end
        else
            s = 0.0; ttype = -12
        end
    end
    return (s, ttype, g)
end

# --- DLASQ3 2×2 deflation helper (reference label 40) ---
@inline function _dlasq3_2x2!(Z::Vector{Float64}, nn::Int, n0::Int, sigma::Float64, tol2::Float64)
    @inbounds begin
        if Z[nn-3] > Z[nn-7]
            s = Z[nn-3]; Z[nn-3] = Z[nn-7]; Z[nn-7] = s
        end
        t = 0.5*((Z[nn-7] - Z[nn-3]) + Z[nn-5])
        if Z[nn-5] > Z[nn-3]*tol2 && t != 0.0
            s = Z[nn-3]*(Z[nn-5]/t)
            if s <= t
                s = Z[nn-3]*(Z[nn-5]/(t*(1.0 + sqrt(1.0 + s/t))))
            else
                s = Z[nn-3]*(Z[nn-5]/(t + sqrt(t)*sqrt(t + s)))
            end
            t = Z[nn-7] + (s + Z[nn-5])
            Z[nn-3] = Z[nn-3]*(Z[nn-7]/t)
            Z[nn-7] = t
        end
        Z[4*n0-7] = Z[nn-7] + sigma
        Z[4*n0-3] = Z[nn-3] + sigma
    end
    return nothing
end

# --- DLASQ3: one full dqds sweep (deflation + shift choice + dqds5/6). Mutates st + Z; returns loop vars ---
function _dlasq3!(Z::Vector{Float64}, i0::Int, n0::Int, pp::Int, dmin::Float64, sigma::Float64,
        desig::Float64, qmax::Float64, st::_DqdsState, epsm::Float64)
    tol = epsm * 100.0; tol2 = tol * tol
    n0in = n0
    ttype = st.ttype; dmin1 = st.dmin1; dmin2 = st.dmin2
    dn = st.dn; dn1 = st.dn1; dn2 = st.dn2; g = st.g; tau = st.tau
    @inbounds begin
        # --- deflation (reference labels 10/20/30/40/50) ---
        empty = false
        while true
            if n0 < i0; empty = true; break; end
            if n0 == i0
                Z[4*n0-3] = Z[4*n0+pp-3] + sigma; n0 -= 1; continue
            end
            nn = 4*n0 + pp
            if n0 == i0 + 1
                _dlasq3_2x2!(Z, nn, n0, sigma, tol2); n0 -= 2; continue
            end
            if Z[nn-5] > tol2*(sigma + Z[nn-3]) && Z[nn-2*pp-4] > tol2*Z[nn-7]
                if Z[nn-9] > tol2*sigma && Z[nn-2*pp-8] > tol2*Z[nn-11]
                    break                                    # label 50 reached
                else
                    _dlasq3_2x2!(Z, nn, n0, sigma, tol2); n0 -= 2; continue
                end
            else
                Z[4*n0-3] = Z[4*n0+pp-3] + sigma; n0 -= 1; continue
            end
        end
        if !empty
            (pp == 2) && (pp = 0)
            # reverse the qd-array if warranted
            if dmin <= 0.0 || n0 < n0in
                if 1.5*Z[4*i0+pp-3] < Z[4*n0+pp-3]
                    ipn4 = 4*(i0 + n0)
                    for j4 in 4*i0:4:2*(i0+n0-1)
                        temp = Z[j4-3]; Z[j4-3] = Z[ipn4-j4-3]; Z[ipn4-j4-3] = temp
                        temp = Z[j4-2]; Z[j4-2] = Z[ipn4-j4-2]; Z[ipn4-j4-2] = temp
                        temp = Z[j4-1]; Z[j4-1] = Z[ipn4-j4-5]; Z[ipn4-j4-5] = temp
                        temp = Z[j4];   Z[j4]   = Z[ipn4-j4-4]; Z[ipn4-j4-4] = temp
                    end
                    if n0 - i0 <= 4
                        Z[4*n0+pp-1] = Z[4*i0+pp-1]; Z[4*n0-pp] = Z[4*i0-pp]
                    end
                    dmin2 = min(dmin2, Z[4*n0+pp-1])
                    Z[4*n0+pp-1] = min(Z[4*n0+pp-1], Z[4*i0+pp-1], Z[4*i0+pp+3])
                    Z[4*n0-pp]   = min(Z[4*n0-pp], Z[4*i0-pp], Z[4*i0-pp+4])
                    qmax = max(qmax, Z[4*i0+pp-3], Z[4*i0+pp+1])
                    dmin = -0.0
                end
            end
            tau, ttype, g = _dlasq4(Z, i0, n0, pp, n0in, dmin, dmin1, dmin2, dn, dn1, dn2, tau, ttype, g)
            # dqds step, retrying with smaller shift while DMIN < 0 (reference labels 70/80)
            while true
                dmin, dmin1, dmin2, dn, dn1, dn2, tau = _dlasq5!(Z, i0, n0, pp, tau, sigma, epsm)
                st.ndiv += (n0 - i0 + 2); st.iter += 1
                if dmin >= 0.0 && dmin1 >= 0.0
                    break                                    # success
                elseif dmin < 0.0 && dmin1 > 0.0 && Z[4*(n0-1)-pp] < tol*(sigma + dn1) && abs(dn) < tol*sigma
                    Z[4*(n0-1)-pp+2] = 0.0; dmin = 0.0; break # convergence hidden by negative DN
                elseif dmin < 0.0
                    st.nfail += 1
                    if ttype < -22
                        tau = 0.0
                    elseif dmin1 > 0.0
                        tau = (tau + dmin)*(1.0 - 2.0*epsm); ttype -= 11
                    else
                        tau = 0.25*tau; ttype -= 12
                    end
                    continue
                elseif isnan(dmin)
                    if tau == 0.0
                        dmin, dmin1, dmin2, dn, dn1, dn2 = _dlasq6!(Z, i0, n0, pp)
                        st.ndiv += (n0 - i0 + 2); st.iter += 1; tau = 0.0; break
                    else
                        tau = 0.0; continue
                    end
                else
                    dmin, dmin1, dmin2, dn, dn1, dn2 = _dlasq6!(Z, i0, n0, pp)
                    st.ndiv += (n0 - i0 + 2); st.iter += 1; tau = 0.0; break
                end
            end
            # accumulate the shift into SIGMA (compensated, reference label 90)
            if tau < sigma
                desig += tau; t = sigma + desig; desig -= (t - sigma); sigma = t
            else
                t = sigma + tau; desig = sigma - (t - tau) + desig; sigma = t
            end
        end
        st.ttype = ttype; st.dmin1 = dmin1; st.dmin2 = dmin2
        st.dn = dn; st.dn1 = dn1; st.dn2 = dn2; st.g = g; st.tau = tau
    end
    return (n0, pp, dmin, sigma, desig, qmax)
end

# --- DLASQ2: dqds driver on the qd array Z (length ≥ 4n). Returns info (0 = success). ---
function _dlasq2!(Z::Vector{Float64}, n::Int, st::_DqdsState)
    epsm = eps(Float64); safmin = floatmin(Float64)
    tol = epsm * 100.0; tol2 = tol * tol
    @inbounds begin
        # n == 1,2 are handled in _dlasq1!; here n ≥ 3.
        Z[2*n] = 0.0
        emin = Z[2]; qmax = 0.0; dsum = 0.0; esum = 0.0
        for k in 1:2:2*(n-1)
            (Z[k] < 0.0 || Z[k+1] < 0.0) && return -1
            dsum += Z[k]; esum += Z[k+1]
            qmax = max(qmax, Z[k]); emin = min(emin, Z[k+1])
        end
        (Z[2*n-1] < 0.0) && return -1
        dsum += Z[2*n-1]; qmax = max(qmax, Z[2*n-1])
        # diagonality
        if esum == 0.0
            for k in 2:n; Z[k] = Z[2*k-1]; end
            _dlasrt_desc!(Z, 1, n); Z[2*n-1] = dsum; return 0
        end
        (dsum + esum == 0.0) && (Z[2*n-1] = 0.0; return 0)
        # rearrange for locality: Z = (q1,qq1,e1,ee1,q2,...)
        for k in 2*n:-2:2
            Z[2*k] = 0.0; Z[2*k-1] = Z[k]; Z[2*k-2] = 0.0; Z[2*k-3] = Z[k-1]
        end
        i0 = 1; n0 = n
        # reverse the qd-array if warranted
        if 1.5*Z[4*i0-3] < Z[4*n0-3]
            ipn4 = 4*(i0 + n0)
            for i4 in 4*i0:4:2*(i0+n0-1)
                temp = Z[i4-3]; Z[i4-3] = Z[ipn4-i4-3]; Z[ipn4-i4-3] = temp
                temp = Z[i4-1]; Z[i4-1] = Z[ipn4-i4-5]; Z[ipn4-i4-5] = temp
            end
        end
        # initial split checking via dqd + Li's test (pp = 0, 1)
        pp = 0
        for _kk in 1:2
            d = Z[4*n0+pp-3]
            for i4 in (4*(n0-1)+pp):-4:(4*i0+pp)
                if Z[i4-1] <= tol2*d
                    Z[i4-1] = -0.0; d = Z[i4-3]
                else
                    d = Z[i4-3]*(d/(d + Z[i4-1]))
                end
            end
            emin = Z[4*i0+pp+1]; d = Z[4*i0+pp-3]
            for i4 in (4*i0+pp):4:(4*(n0-1)+pp)
                Z[i4-2*pp-2] = d + Z[i4-1]
                if Z[i4-1] <= tol2*d
                    Z[i4-1] = -0.0; Z[i4-2*pp-2] = d; Z[i4-2*pp] = 0.0; d = Z[i4+1]
                elseif safmin*Z[i4+1] < Z[i4-2*pp-2] && safmin*Z[i4-2*pp-2] < Z[i4+1]
                    temp = Z[i4+1]/Z[i4-2*pp-2]; Z[i4-2*pp] = Z[i4-1]*temp; d = d*temp
                else
                    Z[i4-2*pp] = Z[i4+1]*(Z[i4-1]/Z[i4-2*pp-2]); d = Z[i4+1]*(d/Z[i4-2*pp-2])
                end
                emin = min(emin, Z[i4-2*pp])
            end
            Z[4*n0-pp-2] = d
            qmax = Z[4*i0-pp-2]
            for i4 in (4*i0-pp+2):4:(4*n0-pp-2); qmax = max(qmax, Z[i4]); end
            pp = 1 - pp
        end
        # initialise the dqds state
        st.ttype = 0; st.dmin1 = 0.0; st.dmin2 = 0.0; st.dn = 0.0; st.dn1 = 0.0; st.dn2 = 0.0
        st.g = 0.0; st.tau = 0.0; st.nfail = 0; st.iter = 2; st.ndiv = 2*(n0 - i0)
        alldone = false
        for _iwhila in 1:(n + 1)
            if n0 < 1; alldone = true; break; end
            desig = 0.0
            sigma = (n0 == n) ? 0.0 : -Z[4*n0-1]
            (sigma < 0.0) && return 1
            # find last unreduced submatrix's top index i0, plus qmax/emin and a Gershgorin bound
            emax = 0.0
            emin = (n0 > i0) ? abs(Z[4*n0-5]) : 0.0
            qmin = Z[4*n0-3]; qmax = qmin
            i4found = 4
            for i4 in 4*n0:-4:8
                if Z[i4-5] <= 0.0; i4found = i4; break; end
                if qmin >= 4.0*emax
                    qmin = min(qmin, Z[i4-3]); emax = max(emax, Z[i4-5])
                end
                qmax = max(qmax, Z[i4-7] + Z[i4-5]); emin = min(emin, Z[i4-5])
            end
            i0 = i4found ÷ 4
            pp = 0
            if n0 - i0 > 1
                dee = Z[4*i0-3]; deemin = dee; kmin = i0
                for i4 in (4*i0+1):4:(4*n0-3)
                    dee = Z[i4]*(dee/(dee + Z[i4-2]))
                    if dee <= deemin; deemin = dee; kmin = (i4 + 3) ÷ 4; end
                end
                if (kmin - i0)*2 < n0 - kmin && deemin <= 0.5*Z[4*n0-3]
                    ipn4 = 4*(i0 + n0); pp = 2
                    for i4 in 4*i0:4:2*(i0+n0-1)
                        temp = Z[i4-3]; Z[i4-3] = Z[ipn4-i4-3]; Z[ipn4-i4-3] = temp
                        temp = Z[i4-2]; Z[i4-2] = Z[ipn4-i4-2]; Z[ipn4-i4-2] = temp
                        temp = Z[i4-1]; Z[i4-1] = Z[ipn4-i4-5]; Z[ipn4-i4-5] = temp
                        temp = Z[i4];   Z[i4]   = Z[ipn4-i4-4]; Z[ipn4-i4-4] = temp
                    end
                end
            end
            dmin = -max(0.0, qmin - 2.0*sqrt(qmin)*sqrt(emax))
            nbig = 100*(n0 - i0 + 1)
            blockdone = false
            for _iwhilb in 1:nbig
                if i0 > n0; blockdone = true; break; end
                n0, pp, dmin, sigma, desig, qmax = _dlasq3!(Z, i0, n0, pp, dmin, sigma, desig, qmax, st, epsm)
                pp = 1 - pp
                # when EMIN is very small, check for splits
                if pp == 0 && n0 - i0 >= 3
                    if Z[4*n0] <= tol2*qmax || Z[4*n0-1] <= tol2*sigma
                        splt = i0 - 1; qmax = Z[4*i0-3]; emin = Z[4*i0-1]; oldemn = Z[4*i0]
                        for i4 in 4*i0:4:4*(n0-3)
                            if Z[i4] <= tol2*Z[i4-3] || Z[i4-1] <= tol2*sigma
                                Z[i4-1] = -sigma; splt = i4 ÷ 4; qmax = 0.0
                                emin = Z[i4+3]; oldemn = Z[i4+4]
                            else
                                qmax = max(qmax, Z[i4+1]); emin = min(emin, Z[i4-1]); oldemn = min(oldemn, Z[i4])
                            end
                        end
                        Z[4*n0-1] = emin; Z[4*n0] = oldemn; i0 = splt + 1
                    end
                end
            end
            (!blockdone) && return 2                          # too many iterations on this block
        end
        (!alldone) && return 3
        # move q's to the front and sort descending
        for k in 2:n; Z[k] = Z[4*k-3]; end
        _dlasrt_desc!(Z, 1, n)
    end
    return 0
end

# --- DLASQ1: driver. Overwrites d with the singular values (descending). Z is 4n-scratch. ---
# Returns info; info ≠ 0 leaves d = |diag|, e = superdiag intact (a valid bidiagonal for the QR fallback).
function _dlasq1!(d::AbstractVector{Float64}, e::AbstractVector{Float64}, Z::Vector{Float64}, st::_DqdsState)
    n = length(d)
    n == 0 && return 0
    if n == 1
        d[1] = abs(d[1]); return 0
    elseif n == 2
        ssmn, ssmx = _dlas2(d[1], e[1], d[2]); d[1] = ssmx; d[2] = ssmn; return 0
    end
    sigmx = 0.0
    @inbounds for i in 1:n-1
        d[i] = abs(d[i]); sigmx = max(sigmx, abs(e[i]))
    end
    @inbounds d[n] = abs(d[n])
    if sigmx == 0.0
        _dlasrt_desc!(d, 1, n); return 0
    end
    @inbounds for i in 1:n; sigmx = max(sigmx, d[i]); end
    epsm = eps(Float64); safmin = floatmin(Float64)
    scale = sqrt(epsm / safmin)
    @inbounds for i in 1:n; Z[2*i-1] = d[i]; end
    @inbounds for i in 1:n-1; Z[2*i] = e[i]; end
    _dlascl!(Z, sigmx, scale, 1, 2*n-1)
    @inbounds for i in 1:2*n-1; Z[i] = Z[i]*Z[i]; end
    @inbounds Z[2*n] = 0.0
    info = _dlasq2!(Z, n, st)
    if info == 0
        @inbounds for i in 1:n; d[i] = sqrt(Z[i]); end
        _dlascl!(d, scale, sigmx, 1, n)
    end
    return info
end
