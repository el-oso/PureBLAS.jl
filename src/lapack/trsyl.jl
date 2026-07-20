# LAPACK triangular Sylvester solver (backs `sylvester`/`lyap`):
#   trsyl — solve  op(A)·X ± X·op(B) = scale·C  for the (quasi-)upper-triangular Schur factors A, B.
# CORRECTNESS-FIRST port of Reference-LAPACK dtrsyl (real) / ztrsyl (complex). The real path drives the
# block back-substitution with the dlaln2 (1×1 / 2×1 / 1×2) and dlasy2 (2×2) small-block solves verbatim;
# the complex path is purely 1×1 (A, B are fully triangular) with a guarded scalar divide.
#
# Self-contained (local dlaln2 / dlasy2 / safmin ports, unique `_syl_*` names so it can be `include`d
# alongside hseqr.jl/trevc.jl which carry their own dlanv2/dlaln2 copies).  STANDALONE: not wired into
# the PureBLAS module include list or the C-ABI — driven directly (or from trsen.jl).

@inline function _syl_safmin(::Type{R}) where {R <: Real}
    sfmin = floatmin(R)
    small = one(R) / floatmax(R)
    small >= sfmin && (sfmin = small * (one(R) + eps(R)))
    return sfmin
end

# complete-pivot bookkeeping (shared with trevc's dlaln2)
const _SYL_IPIVOT = ((1, 2, 3, 4), (2, 1, 4, 3), (3, 4, 1, 2), (4, 3, 2, 1))
const _SYL_ZSWAP = (false, false, true, true)
const _SYL_RSWAP = (false, true, false, true)

# ── DLALN2 specialized for dtrsyl: NA=2, NW=1 (real eigenvalue), CA=1, D1=D2=1, general LTRANS ─────────
# Solves (op(A) − wr·I)·x = b for the 2×2 block A=[a11 a12; a21 a22] (op = transpose iff ltrans), 2-vector
# RHS b=(b1,b2). Complete-pivoting + SMINI guard, Reference-LAPACK dlaln2 verbatim. Returns (x1,x2,scale).
@inline function _syl_dlaln2(
        ltrans::Bool, smin::R, a11::R, a12::R, a21::R, a22::R,
        b1::R, b2::R, wr::R
    ) where {R <: Real}
    ZERO = zero(R); ONE = one(R)
    smlnum = R(2) * _syl_safmin(R)
    bignum = ONE / smlnum
    smini = max(smin, smlnum)
    scale = ONE
    cr11 = a11 - wr; cr22 = a22 - wr
    if ltrans
        cr12 = a21; cr21 = a12
    else
        cr21 = a21; cr12 = a12
    end
    crv = (cr11, cr21, cr12, cr22)            # column-major: (1,1),(2,1),(1,2),(2,2)
    cmax = ZERO; icmax = 0
    for j in 1:4
        if abs(crv[j]) > cmax
            cmax = abs(crv[j]); icmax = j
        end
    end
    if cmax < smini
        bnorm = max(abs(b1), abs(b2))
        if smini < ONE && bnorm > ONE
            bnorm > bignum * smini && (scale = ONE / bnorm)
        end
        temp = scale / smini
        return temp * b1, temp * b2, scale
    end
    piv = _SYL_IPIVOT[icmax]
    ur11 = crv[icmax]; cr21p = crv[piv[2]]; ur12 = crv[piv[3]]; cr22p = crv[piv[4]]
    ur11r = ONE / ur11
    lr21 = ur11r * cr21p
    ur22 = cr22p - ur12 * lr21
    abs(ur22) < smini && (ur22 = smini)
    if _SYL_RSWAP[icmax]
        br1 = b2; br2 = b1
    else
        br1 = b1; br2 = b2
    end
    br2 = br2 - lr21 * br1
    bbnd = max(abs(br1 * (ur22 * ur11r)), abs(br2))
    if bbnd > ONE && abs(ur22) < ONE
        bbnd >= bignum * abs(ur22) && (scale = ONE / bbnd)
    end
    xr2 = (br2 * scale) / ur22
    xr1 = (scale * br1) * ur11r - xr2 * (ur11r * ur12)
    if _SYL_ZSWAP[icmax]
        x1 = xr2; x2 = xr1
    else
        x1 = xr1; x2 = xr2
    end
    xnorm = max(abs(xr1), abs(xr2))
    if xnorm > ONE && cmax > ONE && xnorm > bignum / cmax
        temp = cmax / bignum
        x1 *= temp; x2 *= temp; scale *= temp
    end
    return x1, x2, scale
end

# ── DLASY2 (Reference-LAPACK verbatim) — solve the small Sylvester op(TL)·X ± X·op(TR) = scale·B ───────
# n1,n2 ∈ {1,2}. TL (n1×n1), TR (n2×n2), B (n1×n2) are passed as AbstractMatrix (0-based-safe views).
# isgn = ±1 sets the sign; ltranl/ltranr transpose TL/TR. Returns (x11,x21,x12,x22, scale, xnorm, info)
# (X is n1×n2, column-major into the four slots). This is the 2×2 conj-pair block solve — a bug locus.
function _syl_dlasy2(
        ltranl::Bool, ltranr::Bool, isgn::Int, n1::Int, n2::Int,
        TL::AbstractMatrix{R}, TR::AbstractMatrix{R}, B::AbstractMatrix{R}
    ) where {R <: Real}
    ZERO = zero(R); ONE = one(R); TWO = R(2); HALF = R(0.5); EIGHT = R(8)
    eps_p = eps(R)
    smlnum = _syl_safmin(R) / eps_p
    sgn = R(isgn)
    info = 0
    x11 = ZERO; x21 = ZERO; x12 = ZERO; x22 = ZERO; scale = ONE; xnorm = ZERO
    k = n1 + n1 + n2 - 2
    if k == 1
        # 1×1 : (TL + sgn*TR) x = B
        tau1 = TL[1, 1] + sgn * TR[1, 1]
        bet = abs(tau1)
        if bet <= smlnum
            tau1 = smlnum; bet = smlnum; info = 1
        end
        gam = abs(B[1, 1])
        smlnum * gam > bet && (scale = ONE / gam)
        x11 = (B[1, 1] * scale) / tau1
        xnorm = abs(x11)
        return x11, x21, x12, x22, scale, xnorm, info
    elseif k == 2 || k == 3
        # 2-unknown systems (n1=1,n2=2 ⇒ k=2 ; n1=2,n2=1 ⇒ k=3)
        tmp1 = ZERO; tmp2 = ZERO; tmp3 = ZERO; tmp4 = ZERO
        btmp1 = ZERO; btmp2 = ZERO; smin = ZERO
        if k == 2
            smin = max(
                eps_p * max(
                    abs(TL[1, 1]), abs(TR[1, 1]), abs(TR[1, 2]),
                    abs(TR[2, 1]), abs(TR[2, 2])
                ), smlnum
            )
            tmp1 = TL[1, 1] + sgn * TR[1, 1]
            tmp4 = TL[1, 1] + sgn * TR[2, 2]
            if ltranr
                tmp2 = sgn * TR[2, 1]; tmp3 = sgn * TR[1, 2]
            else
                tmp2 = sgn * TR[1, 2]; tmp3 = sgn * TR[2, 1]
            end
            btmp1 = B[1, 1]; btmp2 = B[1, 2]
        else
            smin = max(
                eps_p * max(
                    abs(TR[1, 1]), abs(TL[1, 1]), abs(TL[1, 2]),
                    abs(TL[2, 1]), abs(TL[2, 2])
                ), smlnum
            )
            tmp1 = TL[1, 1] + sgn * TR[1, 1]
            tmp4 = TL[2, 2] + sgn * TR[1, 1]
            if ltranl
                tmp2 = TL[1, 2]; tmp3 = TL[2, 1]
            else
                tmp2 = TL[2, 1]; tmp3 = TL[1, 2]
            end
            btmp1 = B[1, 1]; btmp2 = B[2, 1]
        end
        tmp = (tmp1, tmp2, tmp3, tmp4)
        locu12 = (3, 4, 1, 2); locl21 = (2, 1, 4, 3); locu22 = (4, 3, 2, 1)
        xswpiv = (false, false, true, true); bswpiv = (false, true, false, true)
        ipiv = 1; big = abs(tmp[1])
        for j in 2:4
            abs(tmp[j]) > big && (big = abs(tmp[j]); ipiv = j)
        end
        u11 = tmp[ipiv]
        if abs(u11) <= smin
            info = 1; u11 = smin
        end
        u12 = tmp[locu12[ipiv]]
        l21 = tmp[locl21[ipiv]] / u11
        u22 = tmp[locu22[ipiv]] - u12 * l21
        xswap = xswpiv[ipiv]; bswap = bswpiv[ipiv]
        if abs(u22) <= smin
            info = 1; u22 = smin
        end
        if bswap
            temp = btmp2
            btmp2 = btmp1 - l21 * temp
            btmp1 = temp
        else
            btmp2 = btmp2 - l21 * btmp1
        end
        if (TWO * smlnum) * abs(btmp2) > abs(u22) || (TWO * smlnum) * abs(btmp1) > abs(u11)
            scale = HALF / max(abs(btmp1), abs(btmp2))
            btmp1 *= scale; btmp2 *= scale
        end
        xa2 = btmp2 / u22
        xa1 = btmp1 / u11 - (u12 / u11) * xa2
        if xswap
            temp = xa2; xa2 = xa1; xa1 = temp
        end
        x11 = xa1
        if n1 == 1
            x12 = xa2
            xnorm = abs(x11) + abs(x12)
        else
            x21 = xa2
            xnorm = max(abs(x11), abs(x21))
        end
        return x11, x21, x12, x22, scale, xnorm, info
    end
    # k == 4 : full 2×2×2 block, solve the 4×4 Kronecker system with complete pivoting
    smin = max(abs(TR[1, 1]), abs(TR[1, 2]), abs(TR[2, 1]), abs(TR[2, 2]))
    smin = max(smin, abs(TL[1, 1]), abs(TL[1, 2]), abs(TL[2, 1]), abs(TL[2, 2]))
    smin = max(eps_p * smin, smlnum)
    t16 = zeros(R, 4, 4)
    t16[1, 1] = TL[1, 1] + sgn * TR[1, 1]
    t16[2, 2] = TL[2, 2] + sgn * TR[1, 1]
    t16[3, 3] = TL[1, 1] + sgn * TR[2, 2]
    t16[4, 4] = TL[2, 2] + sgn * TR[2, 2]
    if ltranl
        t16[1, 2] = TL[2, 1]; t16[2, 1] = TL[1, 2]; t16[3, 4] = TL[2, 1]; t16[4, 3] = TL[1, 2]
    else
        t16[1, 2] = TL[1, 2]; t16[2, 1] = TL[2, 1]; t16[3, 4] = TL[1, 2]; t16[4, 3] = TL[2, 1]
    end
    if ltranr
        t16[1, 3] = sgn * TR[1, 2]; t16[2, 4] = sgn * TR[1, 2]
        t16[3, 1] = sgn * TR[2, 1]; t16[4, 2] = sgn * TR[2, 1]
    else
        t16[1, 3] = sgn * TR[2, 1]; t16[2, 4] = sgn * TR[2, 1]
        t16[3, 1] = sgn * TR[1, 2]; t16[4, 2] = sgn * TR[1, 2]
    end
    btmp = R[B[1, 1], B[2, 1], B[1, 2], B[2, 2]]
    jpiv = zeros(Int, 4)
    for i in 1:3
        xmax = ZERO; ipsv = i; jpsv = i
        for ip in i:4, jp in i:4
            if abs(t16[ip, jp]) >= xmax
                xmax = abs(t16[ip, jp]); ipsv = ip; jpsv = jp
            end
        end
        if ipsv != i
            for c in 1:4
                t16[ipsv, c], t16[i, c] = t16[i, c], t16[ipsv, c]
            end
            btmp[i], btmp[ipsv] = btmp[ipsv], btmp[i]
        end
        if jpsv != i
            for r in 1:4
                t16[r, jpsv], t16[r, i] = t16[r, i], t16[r, jpsv]
            end
        end
        jpiv[i] = jpsv
        abs(t16[i, i]) < smin && (info = 1; t16[i, i] = smin)
        for j in (i + 1):4
            t16[j, i] = t16[j, i] / t16[i, i]
            btmp[j] = btmp[j] - t16[j, i] * btmp[i]
            for kk in (i + 1):4
                t16[j, kk] = t16[j, kk] - t16[j, i] * t16[i, kk]
            end
        end
    end
    abs(t16[4, 4]) < smin && (info = 1; t16[4, 4] = smin)
    if (EIGHT * smlnum) * abs(btmp[1]) > abs(t16[1, 1]) ||
            (EIGHT * smlnum) * abs(btmp[2]) > abs(t16[2, 2]) ||
            (EIGHT * smlnum) * abs(btmp[3]) > abs(t16[3, 3]) ||
            (EIGHT * smlnum) * abs(btmp[4]) > abs(t16[4, 4])
        scale = (ONE / EIGHT) / max(abs(btmp[1]), abs(btmp[2]), abs(btmp[3]), abs(btmp[4]))
        for i in 1:4
            btmp[i] *= scale
        end
    end
    tmpv = zeros(R, 4)
    for i in 1:4
        kk = 5 - i
        temp = ONE / t16[kk, kk]
        tmpv[kk] = btmp[kk] * temp
        for j in (kk + 1):4
            tmpv[kk] = tmpv[kk] - (temp * t16[kk, j]) * tmpv[j]
        end
    end
    for i in 1:3
        if jpiv[4 - i] != 4 - i
            tmpv[4 - i], tmpv[jpiv[4 - i]] = tmpv[jpiv[4 - i]], tmpv[4 - i]
        end
    end
    x11 = tmpv[1]; x21 = tmpv[2]; x12 = tmpv[3]; x22 = tmpv[4]
    xnorm = max(abs(tmpv[1]) + abs(tmpv[3]), abs(tmpv[2]) + abs(tmpv[4]))
    return x11, x21, x12, x22, scale, xnorm, info
end

# ── DTRSYL (Reference-LAPACK verbatim), REAL quasi-triangular A (m×m), B (n×n) ─────────────────────────
function _dtrsyl!(
        trana::AbstractChar, tranb::AbstractChar, isgn::Int,
        A::AbstractMatrix{R}, B::AbstractMatrix{R}, C::AbstractMatrix{R}
    ) where {R <: Real}
    ZERO = zero(R); ONE = one(R)
    m = size(A, 1); n = size(B, 1)
    notrna = trana === 'N'; notrnb = tranb === 'N'
    scale = Ref(ONE)   # Ref{R}, concrete: closure-mutated locals must not be plain captured-and-reassigned (Core.Box, trim-unsafe)
    info = 0
    (m == 0 || n == 0) && return C, scale[], info
    eps_p = eps(R)
    smlnum = _syl_safmin(R)
    bignum = ONE / smlnum
    smlnum = smlnum * R(m * n) / eps_p
    bignum = ONE / smlnum
    # SMIN = max(smlnum, eps*max|A|, eps*max|B|)
    amax = ZERO; @inbounds for j in 1:m, i in 1:m
        amax = max(amax, abs(A[i, j]))
    end
    bmax = ZERO; @inbounds for j in 1:n, i in 1:n
        bmax = max(bmax, abs(B[i, j]))
    end
    smin = max(smlnum, eps_p * amax, eps_p * bmax)
    sgn = R(isgn)

    # DDOT helpers (explicit, trim-safe)
    @inline sarow(k, lo, hi, l) = begin
        s = ZERO; @inbounds for i in lo:hi
            s += A[k, i] * C[i, l]
        end; s
    end   # Σ A[k,i]C[i,l]
    @inline sacol(k, lo, hi, l) = begin
        s = ZERO; @inbounds for i in lo:hi
            s += A[i, k] * C[i, l]
        end; s
    end   # Σ A[i,k]C[i,l]
    @inline sbcol(k, lo, hi, l) = begin
        s = ZERO; @inbounds for j in lo:hi
            s += C[k, j] * B[j, l]
        end; s
    end   # Σ C[k,j]B[j,l]
    @inline sbrow(k, l, lo, hi) = begin
        s = ZERO; @inbounds for j in lo:hi
            s += C[k, j] * B[l, j]
        end; s
    end   # Σ C[k,j]B[l,j]
    @inline function scaleC!(sc)
        return if sc != ONE
            @inbounds for jj in 1:n, ii in 1:m
                C[ii, jj] *= sc
            end
            scale[] *= sc
        end
    end
    # small quasi-tri block views
    Ablk(k1, k2) = view(A, k1:k2, k1:k2)
    Bblk(l1, l2) = view(B, l1:l2, l1:l2)

    @inbounds if notrna && notrnb
        lnext = 1
        L = 1
        while L <= n
            if L < lnext
                L += 1; continue
            end
            if L == n
                l1 = L; l2 = L
            elseif B[L + 1, L] != ZERO
                l1 = L; l2 = L + 1; lnext = L + 2
            else
                l1 = L; l2 = L; lnext = L + 1
            end
            knext = m
            K = m
            while K >= 1
                if K > knext
                    K -= 1; continue
                end
                if K == 1
                    k1 = K; k2 = K
                elseif A[K, K - 1] != ZERO
                    k1 = K - 1; k2 = K; knext = K - 2
                else
                    k1 = K; k2 = K; knext = K - 1
                end
                if l1 == l2 && k1 == k2
                    suml = sarow(k1, k1 + 1, m, l1); sumr = sbcol(k1, 1, l1 - 1, l1)
                    vec = C[k1, l1] - (suml + sgn * sumr)
                    scaloc = ONE; a11 = A[k1, k1] + sgn * B[l1, l1]; da11 = abs(a11)
                    if da11 <= smin
                        a11 = smin; da11 = smin; info = 1
                    end
                    db = abs(vec)
                    da11 < ONE && db > ONE && db > bignum * da11 && (scaloc = ONE / db)
                    x = (vec * scaloc) / a11; scaleC!(scaloc); C[k1, l1] = x
                elseif l1 == l2 && k1 != k2
                    suml = sarow(k1, k2 + 1, m, l1); sumr = sbcol(k1, 1, l1 - 1, l1)
                    v1 = C[k1, l1] - (suml + sgn * sumr)
                    suml = sarow(k2, k2 + 1, m, l1); sumr = sbcol(k2, 1, l1 - 1, l1)
                    v2 = C[k2, l1] - (suml + sgn * sumr)
                    x1, x2, scaloc = _syl_dlaln2(false, smin, A[k1, k1], A[k1, k2], A[k2, k1], A[k2, k2], v1, v2, -sgn * B[l1, l1])
                    scaleC!(scaloc); C[k1, l1] = x1; C[k2, l1] = x2
                elseif l1 != l2 && k1 == k2
                    suml = sarow(k1, k1 + 1, m, l1); sumr = sbcol(k1, 1, l1 - 1, l1)
                    v1 = sgn * (C[k1, l1] - (suml + sgn * sumr))
                    suml = sarow(k1, k1 + 1, m, l2); sumr = sbcol(k1, 1, l1 - 1, l2)
                    v2 = sgn * (C[k1, l2] - (suml + sgn * sumr))
                    x1, x2, scaloc = _syl_dlaln2(true, smin, B[l1, l1], B[l1, l2], B[l2, l1], B[l2, l2], v1, v2, -sgn * A[k1, k1])
                    scaleC!(scaloc); C[k1, l1] = x1; C[k1, l2] = x2
                else
                    v11 = C[k1, l1] - (sarow(k1, k2 + 1, m, l1) + sgn * sbcol(k1, 1, l1 - 1, l1))
                    v12 = C[k1, l2] - (sarow(k1, k2 + 1, m, l2) + sgn * sbcol(k1, 1, l1 - 1, l2))
                    v21 = C[k2, l1] - (sarow(k2, k2 + 1, m, l1) + sgn * sbcol(k2, 1, l1 - 1, l1))
                    v22 = C[k2, l2] - (sarow(k2, k2 + 1, m, l2) + sgn * sbcol(k2, 1, l1 - 1, l2))
                    vb = R[v11 v12; v21 v22]
                    x11, x21, x12, x22, scaloc, _, ie = _syl_dlasy2(false, false, isgn, 2, 2, Ablk(k1, k2), Bblk(l1, l2), vb)
                    ie != 0 && (info = 1); scaleC!(scaloc)
                    C[k1, l1] = x11; C[k1, l2] = x12; C[k2, l1] = x21; C[k2, l2] = x22
                end
                K = k1 - 1
            end
            L = l2 + 1
        end
    elseif !notrna && notrnb
        lnext = 1
        L = 1
        while L <= n
            if L < lnext
                L += 1; continue
            end
            if L == n
                l1 = L; l2 = L
            elseif B[L + 1, L] != ZERO
                l1 = L; l2 = L + 1; lnext = L + 2
            else
                l1 = L; l2 = L; lnext = L + 1
            end
            knext = 1
            K = 1
            while K <= m
                if K < knext
                    K += 1; continue
                end
                if K == m
                    k1 = K; k2 = K
                elseif A[K + 1, K] != ZERO
                    k1 = K; k2 = K + 1; knext = K + 2
                else
                    k1 = K; k2 = K; knext = K + 1
                end
                if l1 == l2 && k1 == k2
                    suml = sacol(k1, 1, k1 - 1, l1); sumr = sbcol(k1, 1, l1 - 1, l1)
                    vec = C[k1, l1] - (suml + sgn * sumr)
                    scaloc = ONE; a11 = A[k1, k1] + sgn * B[l1, l1]; da11 = abs(a11)
                    if da11 <= smin
                        a11 = smin; da11 = smin; info = 1
                    end
                    db = abs(vec); da11 < ONE && db > ONE && db > bignum * da11 && (scaloc = ONE / db)
                    x = (vec * scaloc) / a11; scaleC!(scaloc); C[k1, l1] = x
                elseif l1 == l2 && k1 != k2
                    suml = sacol(k1, 1, k1 - 1, l1); sumr = sbcol(k1, 1, l1 - 1, l1)
                    v1 = C[k1, l1] - (suml + sgn * sumr)
                    suml = sacol(k2, 1, k1 - 1, l1); sumr = sbcol(k2, 1, l1 - 1, l1)
                    v2 = C[k2, l1] - (suml + sgn * sumr)
                    x1, x2, scaloc = _syl_dlaln2(true, smin, A[k1, k1], A[k1, k2], A[k2, k1], A[k2, k2], v1, v2, -sgn * B[l1, l1])
                    scaleC!(scaloc); C[k1, l1] = x1; C[k2, l1] = x2
                elseif l1 != l2 && k1 == k2
                    suml = sacol(k1, 1, k1 - 1, l1); sumr = sbcol(k1, 1, l1 - 1, l1)
                    v1 = sgn * (C[k1, l1] - (suml + sgn * sumr))
                    suml = sacol(k1, 1, k1 - 1, l2); sumr = sbcol(k1, 1, l1 - 1, l2)
                    v2 = sgn * (C[k1, l2] - (suml + sgn * sumr))
                    x1, x2, scaloc = _syl_dlaln2(true, smin, B[l1, l1], B[l1, l2], B[l2, l1], B[l2, l2], v1, v2, -sgn * A[k1, k1])
                    scaleC!(scaloc); C[k1, l1] = x1; C[k1, l2] = x2
                else
                    v11 = C[k1, l1] - (sacol(k1, 1, k1 - 1, l1) + sgn * sbcol(k1, 1, l1 - 1, l1))
                    v12 = C[k1, l2] - (sacol(k1, 1, k1 - 1, l2) + sgn * sbcol(k1, 1, l1 - 1, l2))
                    v21 = C[k2, l1] - (sacol(k2, 1, k1 - 1, l1) + sgn * sbcol(k2, 1, l1 - 1, l1))
                    v22 = C[k2, l2] - (sacol(k2, 1, k1 - 1, l2) + sgn * sbcol(k2, 1, l1 - 1, l2))
                    vb = R[v11 v12; v21 v22]
                    x11, x21, x12, x22, scaloc, _, ie = _syl_dlasy2(true, false, isgn, 2, 2, Ablk(k1, k2), Bblk(l1, l2), vb)
                    ie != 0 && (info = 1); scaleC!(scaloc)
                    C[k1, l1] = x11; C[k1, l2] = x12; C[k2, l1] = x21; C[k2, l2] = x22
                end
                K = k2 + 1
            end
            L = l2 + 1
        end
    elseif !notrna && !notrnb
        lnext = n
        L = n
        while L >= 1
            if L > lnext
                L -= 1; continue
            end
            if L == 1
                l1 = L; l2 = L
            elseif B[L, L - 1] != ZERO
                l1 = L - 1; l2 = L; lnext = L - 2
            else
                l1 = L; l2 = L; lnext = L - 1
            end
            knext = 1
            K = 1
            while K <= m
                if K < knext
                    K += 1; continue
                end
                if K == m
                    k1 = K; k2 = K
                elseif A[K + 1, K] != ZERO
                    k1 = K; k2 = K + 1; knext = K + 2
                else
                    k1 = K; k2 = K; knext = K + 1
                end
                if l1 == l2 && k1 == k2
                    suml = sacol(k1, 1, k1 - 1, l1); sumr = sbrow(k1, l1, l1 + 1, n)
                    vec = C[k1, l1] - (suml + sgn * sumr)
                    scaloc = ONE; a11 = A[k1, k1] + sgn * B[l1, l1]; da11 = abs(a11)
                    if da11 <= smin
                        a11 = smin; da11 = smin; info = 1
                    end
                    db = abs(vec); da11 < ONE && db > ONE && db > bignum * da11 && (scaloc = ONE / db)
                    x = (vec * scaloc) / a11; scaleC!(scaloc); C[k1, l1] = x
                elseif l1 == l2 && k1 != k2
                    suml = sacol(k1, 1, k1 - 1, l1); sumr = sbrow(k1, l1, l2 + 1, n)
                    v1 = C[k1, l1] - (suml + sgn * sumr)
                    suml = sacol(k2, 1, k1 - 1, l1); sumr = sbrow(k2, l1, l2 + 1, n)
                    v2 = C[k2, l1] - (suml + sgn * sumr)
                    x1, x2, scaloc = _syl_dlaln2(true, smin, A[k1, k1], A[k1, k2], A[k2, k1], A[k2, k2], v1, v2, -sgn * B[l1, l1])
                    scaleC!(scaloc); C[k1, l1] = x1; C[k2, l1] = x2
                elseif l1 != l2 && k1 == k2
                    suml = sacol(k1, 1, k1 - 1, l1); sumr = sbrow(k1, l1, l2 + 1, n)
                    v1 = sgn * (C[k1, l1] - (suml + sgn * sumr))
                    suml = sacol(k1, 1, k1 - 1, l2); sumr = sbrow(k1, l2, l2 + 1, n)
                    v2 = sgn * (C[k1, l2] - (suml + sgn * sumr))
                    x1, x2, scaloc = _syl_dlaln2(false, smin, B[l1, l1], B[l1, l2], B[l2, l1], B[l2, l2], v1, v2, -sgn * A[k1, k1])
                    scaleC!(scaloc); C[k1, l1] = x1; C[k1, l2] = x2
                else
                    v11 = C[k1, l1] - (sacol(k1, 1, k1 - 1, l1) + sgn * sbrow(k1, l1, l2 + 1, n))
                    v12 = C[k1, l2] - (sacol(k1, 1, k1 - 1, l2) + sgn * sbrow(k1, l2, l2 + 1, n))
                    v21 = C[k2, l1] - (sacol(k2, 1, k1 - 1, l1) + sgn * sbrow(k2, l1, l2 + 1, n))
                    v22 = C[k2, l2] - (sacol(k2, 1, k1 - 1, l2) + sgn * sbrow(k2, l2, l2 + 1, n))
                    vb = R[v11 v12; v21 v22]
                    x11, x21, x12, x22, scaloc, _, ie = _syl_dlasy2(true, true, isgn, 2, 2, Ablk(k1, k2), Bblk(l1, l2), vb)
                    ie != 0 && (info = 1); scaleC!(scaloc)
                    C[k1, l1] = x11; C[k1, l2] = x12; C[k2, l1] = x21; C[k2, l2] = x22
                end
                K = k2 + 1
            end
            L = l1 - 1
        end
    else  # notrna && !notrnb
        lnext = n
        L = n
        while L >= 1
            if L > lnext
                L -= 1; continue
            end
            if L == 1
                l1 = L; l2 = L
            elseif B[L, L - 1] != ZERO
                l1 = L - 1; l2 = L; lnext = L - 2
            else
                l1 = L; l2 = L; lnext = L - 1
            end
            knext = m
            K = m
            while K >= 1
                if K > knext
                    K -= 1; continue
                end
                if K == 1
                    k1 = K; k2 = K
                elseif A[K, K - 1] != ZERO
                    k1 = K - 1; k2 = K; knext = K - 2
                else
                    k1 = K; k2 = K; knext = K - 1
                end
                if l1 == l2 && k1 == k2
                    suml = sarow(k1, k1 + 1, m, l1); sumr = sbrow(k1, l1, l1 + 1, n)
                    vec = C[k1, l1] - (suml + sgn * sumr)
                    scaloc = ONE; a11 = A[k1, k1] + sgn * B[l1, l1]; da11 = abs(a11)
                    if da11 <= smin
                        a11 = smin; da11 = smin; info = 1
                    end
                    db = abs(vec); da11 < ONE && db > ONE && db > bignum * da11 && (scaloc = ONE / db)
                    x = (vec * scaloc) / a11; scaleC!(scaloc); C[k1, l1] = x
                elseif l1 == l2 && k1 != k2
                    suml = sarow(k1, k2 + 1, m, l1); sumr = sbrow(k1, l1, l2 + 1, n)
                    v1 = C[k1, l1] - (suml + sgn * sumr)
                    suml = sarow(k2, k2 + 1, m, l1); sumr = sbrow(k2, l1, l2 + 1, n)
                    v2 = C[k2, l1] - (suml + sgn * sumr)
                    x1, x2, scaloc = _syl_dlaln2(false, smin, A[k1, k1], A[k1, k2], A[k2, k1], A[k2, k2], v1, v2, -sgn * B[l1, l1])
                    scaleC!(scaloc); C[k1, l1] = x1; C[k2, l1] = x2
                elseif l1 != l2 && k1 == k2
                    suml = sarow(k1, k1 + 1, m, l1); sumr = sbrow(k1, l1, l2 + 1, n)
                    v1 = sgn * (C[k1, l1] - (suml + sgn * sumr))
                    suml = sarow(k1, k1 + 1, m, l2); sumr = sbrow(k1, l2, l2 + 1, n)
                    v2 = sgn * (C[k1, l2] - (suml + sgn * sumr))
                    x1, x2, scaloc = _syl_dlaln2(false, smin, B[l1, l1], B[l1, l2], B[l2, l1], B[l2, l2], v1, v2, -sgn * A[k1, k1])
                    scaleC!(scaloc); C[k1, l1] = x1; C[k1, l2] = x2
                else
                    v11 = C[k1, l1] - (sarow(k1, k2 + 1, m, l1) + sgn * sbrow(k1, l1, l2 + 1, n))
                    v12 = C[k1, l2] - (sarow(k1, k2 + 1, m, l2) + sgn * sbrow(k1, l2, l2 + 1, n))
                    v21 = C[k2, l1] - (sarow(k2, k2 + 1, m, l1) + sgn * sbrow(k2, l1, l2 + 1, n))
                    v22 = C[k2, l2] - (sarow(k2, k2 + 1, m, l2) + sgn * sbrow(k2, l2, l2 + 1, n))
                    vb = R[v11 v12; v21 v22]
                    x11, x21, x12, x22, scaloc, _, ie = _syl_dlasy2(false, true, isgn, 2, 2, Ablk(k1, k2), Bblk(l1, l2), vb)
                    ie != 0 && (info = 1); scaleC!(scaloc)
                    C[k1, l1] = x11; C[k1, l2] = x12; C[k2, l1] = x21; C[k2, l2] = x22
                end
                K = k1 - 1
            end
            L = l1 - 1
        end
    end
    return C, scale[], info
end

# ── ZTRSYL (Reference-LAPACK verbatim), COMPLEX triangular A (m×m), B (n×n) ────────────────────────────
function _ztrsyl!(
        trana::AbstractChar, tranb::AbstractChar, isgn::Int,
        A::AbstractMatrix{TC}, B::AbstractMatrix{TC}, C::AbstractMatrix{TC}
    ) where {TC <: Complex}
    R = real(TC)
    ONE = one(R)
    m = size(A, 1); n = size(B, 1)
    notrna = trana === 'N'; notrnb = tranb === 'N'
    cabs1(z) = abs(real(z)) + abs(imag(z))
    scale = Ref(ONE); info = 0   # Ref{R}, concrete: closure-mutated capture must not be a plain reassigned local (Core.Box)
    (m == 0 || n == 0) && return C, scale[], info
    eps_p = eps(R)
    smlnum = _syl_safmin(R)
    bignum = ONE / smlnum
    smlnum = smlnum * R(m * n) / eps_p
    bignum = ONE / smlnum
    amax = zero(R); @inbounds for j in 1:m, i in 1:m
        amax = max(amax, cabs1(A[i, j]))
    end
    bmax = zero(R); @inbounds for j in 1:n, i in 1:n
        bmax = max(bmax, cabs1(B[i, j]))
    end
    smin = max(smlnum, eps_p * amax, eps_p * bmax)
    sgn = TC(isgn)

    # ZDOTU: Σ x_i y_i ; ZDOTC: Σ conj(x_i) y_i
    @inline dotu_arow(k, lo, hi, l) = begin
        s = zero(TC); @inbounds for i in lo:hi
            s += A[k, i] * C[i, l]
        end; s
    end
    @inline dotc_acol(k, lo, hi, l) = begin
        s = zero(TC); @inbounds for i in lo:hi
            s += conj(A[i, k]) * C[i, l]
        end; s
    end
    @inline dotu_bcol(k, lo, hi, l) = begin
        s = zero(TC); @inbounds for j in lo:hi
            s += C[k, j] * B[j, l]
        end; s
    end
    @inline dotc_brow(k, l, lo, hi) = begin
        s = zero(TC); @inbounds for j in lo:hi
            s += conj(C[k, j]) * B[l, j]
        end; s
    end
    @inline function scaleC!(sc)
        return if sc != ONE
            @inbounds for jj in 1:n, ii in 1:m
                C[ii, jj] *= sc
            end
            scale[] *= sc
        end
    end

    @inbounds if notrna && notrnb
        for l in 1:n, k in m:-1:1
            suml = dotu_arow(k, k + 1, m, l)
            sumr = dotu_bcol(k, 1, l - 1, l)
            vec = C[k, l] - (suml + sgn * sumr)
            scaloc = ONE; a11 = A[k, k] + sgn * B[l, l]; da11 = cabs1(a11)
            if da11 <= smin
                a11 = TC(smin); da11 = smin; info = 1
            end
            db = cabs1(vec); da11 < ONE && db > ONE && db > bignum * da11 && (scaloc = ONE / db)
            x = (vec * scaloc) / a11; scaleC!(scaloc); C[k, l] = x
        end
    elseif !notrna && notrnb
        for l in 1:n, k in 1:m
            suml = dotc_acol(k, 1, k - 1, l)
            sumr = dotu_bcol(k, 1, l - 1, l)
            vec = C[k, l] - (suml + sgn * sumr)
            scaloc = ONE; a11 = conj(A[k, k]) + sgn * B[l, l]; da11 = cabs1(a11)
            if da11 <= smin
                a11 = TC(smin); da11 = smin; info = 1
            end
            db = cabs1(vec); da11 < ONE && db > ONE && db > bignum * da11 && (scaloc = ONE / db)
            x = (vec * scaloc) / a11; scaleC!(scaloc); C[k, l] = x
        end
    elseif !notrna && !notrnb
        for l in n:-1:1, k in 1:m
            suml = dotc_acol(k, 1, k - 1, l)
            sumr = dotc_brow(k, l, l + 1, n)
            vec = C[k, l] - (suml + sgn * conj(sumr))
            scaloc = ONE; a11 = conj(A[k, k] + sgn * B[l, l]); da11 = cabs1(a11)
            if da11 <= smin
                a11 = TC(smin); da11 = smin; info = 1
            end
            db = cabs1(vec); da11 < ONE && db > ONE && db > bignum * da11 && (scaloc = ONE / db)
            x = (vec * scaloc) / a11; scaleC!(scaloc); C[k, l] = x
        end
    else  # notrna && !notrnb
        for l in n:-1:1, k in m:-1:1
            suml = dotu_arow(k, k + 1, m, l)
            sumr = dotc_brow(k, l, l + 1, n)
            vec = C[k, l] - (suml + sgn * conj(sumr))
            scaloc = ONE; a11 = A[k, k] + sgn * conj(B[l, l]); da11 = cabs1(a11)
            if da11 <= smin
                a11 = TC(smin); da11 = smin; info = 1
            end
            db = cabs1(vec); da11 < ONE && db > ONE && db > bignum * da11 && (scaloc = ONE / db)
            x = (vec * scaloc) / a11; scaleC!(scaloc); C[k, l] = x
        end
    end
    return C, scale[], info
end

"""
    trsyl!(transa, transb, isgn, A, B, C) -> (C, scale)

Solve the triangular Sylvester equation `op(A)·X + isgn·X·op(B) = scale·C` in place (LAPACK
`dtrsyl`/`ztrsyl`). `A` (m×m) and `B` (n×n) are the (quasi-)upper-triangular Schur factors, `C` (m×n)
holds the right-hand side on entry and the solution `X` on exit. `op(M) = M`, `Mᵀ`, or `Mᴴ` for
`trans = 'N'`, `'T'`, `'C'`. `isgn = ±1`. `scale ∈ (0,1]` is chosen to avoid overflow. Backs
`sylvester`/`lyap`. Generic over `T<:Real` (dlaln2/dlasy2 1×1/2×2 block solves, real Schur form with
conjugate-pair 2×2 blocks) and `T<:Complex` (scalar back-substitution, fully triangular factors).
"""
function trsyl!(
        transa::AbstractChar, transb::AbstractChar, isgn::Integer,
        A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}
    ) where {T}
    (transa === 'N' || transa === 'T' || transa === 'C') ||
        throw(ArgumentError("trsyl!: transa must be 'N', 'T' or 'C'"))
    (transb === 'N' || transb === 'T' || transb === 'C') ||
        throw(ArgumentError("trsyl!: transb must be 'N', 'T' or 'C'"))
    (isgn == 1 || isgn == -1) || throw(ArgumentError("trsyl!: isgn must be ±1"))
    m = size(A, 1); n = size(B, 1)
    (size(A, 2) == m && size(B, 2) == n) || throw(DimensionMismatch("trsyl!: A, B must be square"))
    (size(C, 1) == m && size(C, 2) == n) || throw(DimensionMismatch("trsyl!: C must be m×n"))
    C, scale, _ = _trsyl_dispatch!(transa, transb, Int(isgn), A, B, C)
    return C, scale
end

_trsyl_dispatch!(ta, tb, isgn, A::AbstractMatrix{<:Real}, B, C) = _dtrsyl!(ta, tb, isgn, A, B, C)
_trsyl_dispatch!(ta, tb, isgn, A::AbstractMatrix{<:Complex}, B, C) = _ztrsyl!(ta, tb, isgn, A, B, C)
