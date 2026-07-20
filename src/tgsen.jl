# LAPACK generalized Schur reorder (dtgsen/ztgsen): given a generalized Schur pair (S,T) — S
# (quasi-)upper-triangular, T upper-triangular, with orthonormal/unitary Q,Z such that
# Q·S·Zᴴ = A, Q·T·Zᴴ = B — move the eigenvalues selected by `select` to the leading block via a
# sequence of adjacent-block swaps (dtgexc/ztgexc, built on dtgex2/ztgex2), then read off the
# generalized eigenvalues (alpha,beta) from the reordered diagonal. Port of Reference-LAPACK
# dtgsen/dtgexc/dtgex2/dlag2 and ztgsen/ztgexc/ztgex2, restricted to the IJOB=0 (reorder only, no
# PL/PR/DIF condition numbers) path — the only path Julia's `LinearAlgebra.LAPACK.tgsen!` wrapper
# ever drives (it always calls with ijob=0 and PL/PR/DIF as C_NULL), so DTGSYL is never needed.
#
# STANDALONE: needs `_syl_safmin` (trsyl.jl) and `_lartg`/`_zlartg`/`_grot_rows!`/`_grot_cols!`/
# `_zrot_rows!`/`_zrot_cols!` (qz.jl) in scope — both are already `include`d by PureBLAS.jl before
# this file would be, so `using PureBLAS` is sufficient; only a raw standalone load needs them first.
#
# HONEST SCOPE (real path): the adjacent-block swap dtgex2 has two cases in Reference-LAPACK —
# (1) both blocks 1×1 (a single Givens sweep) and (2) either block 2×2 (needs a small generalized
# Sylvester solve `dtgsy2` + QR/RQ of up to 4×4 + `dlagv2` re-standardization — a large amount of
# further infrastructure). Only case (1) is implemented here; a swap that would require case (2)
# (i.e. reordering touches a 2×2 block — a complex-conjugate generalized eigenvalue pair) is
# REJECTED (mirrors LAPACK's own info=1 "swap rejected" convention) and `tgsen!` throws. This means
# the real path is fully correct for matrix pairs whose generalized eigenvalues are ALL REAL (no 2×2
# blocks anywhere in S); mixed real/complex-eigenvalue reordering on the real path is NOT supported.
# The COMPLEX path has no 2×2 blocks at all (S,T both strictly triangular) so ztgex2 only ever needs
# case (1) — the complex path is therefore COMPLETE, matching Reference-LAPACK exactly.

# Givens pair-update matching LAPACK's DROT/ZROT: (x,y) ↦ (c·x+s·y, c·y−conj(s)·x). Generic (conj
# is the identity on Real), used identically by the real and complex ports below.
@inline _tgs_rot2(c, s, x, y) = (c * x + s * y, c * y - conj(s) * x)
# Frobenius norm of a 2×2 block given its 4 entries (generic real/complex via abs2).
@inline _tgs_f2norm(a, b, c, d) = sqrt(abs2(a) + abs2(b) + abs2(c) + abs2(d))

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# DTGEX2 (real, 1×1-and-1×1 case only) — swap adjacent 1×1 diagonal blocks of (A,B) at (j1,j1+1).
# Returns info (0 = swapped; 1 = swap rejected: would need a 2×2 block, OR failed the LAPACK weak/
# strong stability test).
# ═══════════════════════════════════════════════════════════════════════════════════════════════
function _dtgex2_1x1!(wantq::Bool, wantz::Bool, A::AbstractMatrix{R}, B::AbstractMatrix{R},
        Q::AbstractMatrix{R}, Z::AbstractMatrix{R}, j1::Int) where {R<:Real}
    n = size(A, 1)
    ZERO = zero(R); TWENTY = R(20)
    Ao11 = A[j1, j1]; Ao12 = A[j1, j1+1]; Ao21 = A[j1+1, j1]; Ao22 = A[j1+1, j1+1]
    Bo11 = B[j1, j1]; Bo12 = B[j1, j1+1]; Bo22 = B[j1+1, j1+1]
    S11 = Ao11; S12 = Ao12; S21 = Ao21; S22 = Ao22
    T11 = Bo11; T12 = Bo12; T22 = Bo22
    eps_p = eps(R); smlnum = _syl_safmin(R) / eps_p
    dnorma = _tgs_f2norm(S11, S12, S21, S22)
    dnormb = _tgs_f2norm(T11, T12, ZERO, T22)
    thresha = max(TWENTY * eps_p * dnorma, smlnum)
    threshb = max(TWENTY * eps_p * dnormb, smlnum)
    F = S22 * T11 - T22 * S11
    G = S22 * T12 - T22 * S12
    SA = abs(S22) * abs(T11); SB = abs(S11) * abs(T22)
    cs, sn, _ = _lartg(F, G)                       # DLARTG(F,G,IR(1,2),IR(1,1),·): cs≡IR12, sn≡IR11
    ir11 = sn; ir12 = cs; ir21 = -ir12; ir22 = ir11
    S11, S12 = _tgs_rot2(ir11, ir21, S11, S12)
    S21, S22 = _tgs_rot2(ir11, ir21, S21, S22)
    T11, T12 = _tgs_rot2(ir11, ir21, T11, T12)
    T21 = ZERO
    T21, T22 = _tgs_rot2(ir11, ir21, T21, T22)
    li11, li21, _ = SA >= SB ? _lartg(S11, S21) : _lartg(T11, T21)
    li22 = li11; li12 = -li21
    S11, S21 = _tgs_rot2(li11, li21, S11, S21)
    S12, S22 = _tgs_rot2(li11, li21, S12, S22)
    T11, T21 = _tgs_rot2(li11, li21, T11, T21)
    T12, T22 = _tgs_rot2(li11, li21, T12, T22)
    weak = abs(S21) <= thresha && abs(T21) <= threshb
    weak || return 1
    # Strong stability test: F-norm(Aorig − LI·S·IRᵀ) and F-norm(Borig − LI·T·IRᵀ) (LI,IR the 2×2
    # accumulated rotation matrices; S,T the tentatively-swapped local block).
    M11 = li11 * S11 + li12 * S21; M12 = li11 * S12 + li12 * S22
    M21 = li21 * S11 + li22 * S21; M22 = li21 * S12 + li22 * S22
    P11 = M11 * ir11 + M12 * ir12; P12 = M11 * ir21 + M12 * ir22
    P21 = M21 * ir11 + M22 * ir12; P22 = M21 * ir21 + M22 * ir22
    SAf = _tgs_f2norm(Ao11 - P11, Ao12 - P12, Ao21 - P21, Ao22 - P22)
    N11 = li11 * T11 + li12 * T21; N12 = li11 * T12 + li12 * T22
    N21 = li21 * T11 + li22 * T21; N22 = li21 * T12 + li22 * T22
    QQ11 = N11 * ir11 + N12 * ir12; QQ12 = N11 * ir21 + N12 * ir22
    QQ21 = N21 * ir11 + N22 * ir12; QQ22 = N21 * ir21 + N22 * ir22
    SBf = _tgs_f2norm(Bo11 - QQ11, Bo12 - QQ12, ZERO - QQ21, Bo22 - QQ22)
    (SAf <= thresha && SBf <= threshb) || return 1
    _grot_cols!(A, j1, j1 + 1, 1, j1 + 1, ir11, ir21)
    _grot_cols!(B, j1, j1 + 1, 1, j1 + 1, ir11, ir21)
    _grot_rows!(A, j1, j1 + 1, j1, n, li11, li21)
    _grot_rows!(B, j1, j1 + 1, j1, n, li11, li21)
    A[j1+1, j1] = ZERO; B[j1+1, j1] = ZERO
    wantz && _grot_cols!(Z, j1, j1 + 1, 1, n, ir11, ir21)
    wantq && _grot_cols!(Q, j1, j1 + 1, 1, n, li11, li21)
    return 0
end

# Dispatcher matching DTGEX2's (N1,N2) signature: only the 1×1/1×1 case is implemented (see file
# header); any swap touching a 2×2 block is rejected (info=1), mirroring LAPACK's own rejection code.
@inline function _dtgex2!(wantq::Bool, wantz::Bool, A::AbstractMatrix{R}, B::AbstractMatrix{R},
        Q::AbstractMatrix{R}, Z::AbstractMatrix{R}, j1::Int, n1::Int, n2::Int) where {R<:Real}
    (n1 == 1 && n2 == 1) || return 1
    return _dtgex2_1x1!(wantq, wantz, A, B, Q, Z, j1)
end

# ── DTGEXC (Reference-LAPACK verbatim control flow, mirrors trsen.jl's `_dtrexc!`) ──────────────
# Move the block at IFST to ILST via a walk of adjacent swaps. Returns (info, ilst_final).
function _dtgexc!(wantq::Bool, wantz::Bool, A::AbstractMatrix{R}, B::AbstractMatrix{R},
        Q::AbstractMatrix{R}, Z::AbstractMatrix{R}, ifst0::Int, ilst0::Int) where {R<:Real}
    ZERO = zero(R)
    n = size(A, 1)
    n <= 1 && return 0, ilst0
    ifst = ifst0; ilst = ilst0
    ifst > 1 && A[ifst, ifst-1] != ZERO && (ifst -= 1)
    nbf = 1
    ifst < n && A[ifst+1, ifst] != ZERO && (nbf = 2)
    ilst > 1 && A[ilst, ilst-1] != ZERO && (ilst -= 1)
    nbl = 1
    ilst < n && A[ilst+1, ilst] != ZERO && (nbl = 2)
    ifst == ilst && return 0, ilst
    if ifst < ilst
        nbf == 2 && nbl == 1 && (ilst -= 1)
        nbf == 1 && nbl == 2 && (ilst += 1)
        here = ifst
        while true
            if nbf == 1 || nbf == 2
                nbnext = 1
                here + nbf + 1 <= n && A[here+nbf+1, here+nbf] != ZERO && (nbnext = 2)
                info = _dtgex2!(wantq, wantz, A, B, Q, Z, here, nbf, nbnext)
                info != 0 && return info, here
                here += nbnext
                nbf == 2 && A[here+1, here] == ZERO && (nbf = 3)
            else
                nbnext = 1
                here + 3 <= n && A[here+3, here+2] != ZERO && (nbnext = 2)
                info = _dtgex2!(wantq, wantz, A, B, Q, Z, here + 1, 1, nbnext)
                info != 0 && return info, here
                if nbnext == 1
                    _dtgex2!(wantq, wantz, A, B, Q, Z, here, 1, nbnext)
                    here += 1
                else
                    A[here+2, here+1] == ZERO && (nbnext = 1)
                    if nbnext == 2
                        info = _dtgex2!(wantq, wantz, A, B, Q, Z, here, 1, nbnext)
                        info != 0 && return info, here
                        here += 2
                    else
                        _dtgex2!(wantq, wantz, A, B, Q, Z, here, 1, 1)
                        _dtgex2!(wantq, wantz, A, B, Q, Z, here + 1, 1, 1)
                        here += 2
                    end
                end
            end
            here < ilst || break
        end
    else
        here = ifst
        while true
            if nbf == 1 || nbf == 2
                nbnext = 1
                here >= 3 && A[here-1, here-2] != ZERO && (nbnext = 2)
                info = _dtgex2!(wantq, wantz, A, B, Q, Z, here - nbnext, nbnext, nbf)
                info != 0 && return info, here
                here -= nbnext
                nbf == 2 && A[here+1, here] == ZERO && (nbf = 3)
            else
                nbnext = 1
                here >= 3 && A[here-1, here-2] != ZERO && (nbnext = 2)
                info = _dtgex2!(wantq, wantz, A, B, Q, Z, here - nbnext, nbnext, 1)
                info != 0 && return info, here
                if nbnext == 1
                    _dtgex2!(wantq, wantz, A, B, Q, Z, here, nbnext, 1)
                    here -= 1
                else
                    A[here, here-1] == ZERO && (nbnext = 1)
                    if nbnext == 2
                        info = _dtgex2!(wantq, wantz, A, B, Q, Z, here - 1, 2, 1)
                        info != 0 && return info, here
                        here -= 2
                    else
                        _dtgex2!(wantq, wantz, A, B, Q, Z, here, 1, 1)
                        _dtgex2!(wantq, wantz, A, B, Q, Z, here - 1, 1, 1)
                        here -= 2
                    end
                end
            end
            here > ilst || break
        end
    end
    return 0, here
end

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# DLAG2 — generalized eigenvalues of a real 2×2 pencil (A − λB), B upper-triangular (b21 ≡ 0).
# Returns (scale1, scale2, wr1, wr2, wi): λ₁ = (wr1+i·wi)/scale1, λ₂ = (wr2−i·wi)/scale2.
# ═══════════════════════════════════════════════════════════════════════════════════════════════
function _tgs_dlag2(a11r::R, a21r::R, a12r::R, a22r::R, b11r::R, b12r::R, b22r::R,
        safmin::R) where {R<:Real}
    ZERO = zero(R); ONE = one(R); TWO = R(2); HALF = ONE / TWO; FUZZY1 = ONE + R(1e-5)
    rtmin = sqrt(safmin); rtmax = ONE / rtmin; safmax = ONE / safmin
    anorm = max(abs(a11r) + abs(a21r), abs(a12r) + abs(a22r), safmin)
    ascale = ONE / anorm
    a11 = ascale * a11r; a21 = ascale * a21r; a12 = ascale * a12r; a22 = ascale * a22r
    b11 = b11r; b12 = b12r; b22 = b22r
    bmin = rtmin * max(abs(b11), abs(b12), abs(b22), rtmin)
    abs(b11) < bmin && (b11 = copysign(bmin, b11))
    abs(b22) < bmin && (b22 = copysign(bmin, b22))
    bnorm = max(abs(b11), abs(b12) + abs(b22), safmin)
    bsize = max(abs(b11), abs(b22))
    bscale = ONE / bsize
    b11 *= bscale; b12 *= bscale; b22 *= bscale
    binv11 = ONE / b11; binv22 = ONE / b22
    s1 = a11 * binv11; s2 = a22 * binv22
    local as12, as11, ss, abi22, pp, shift
    if abs(s1) <= abs(s2)
        as12 = a12 - s1 * b12
        as22 = a22 - s1 * b22
        ss = a21 * (binv11 * binv22)
        abi22 = as22 * binv22 - ss * b12
        pp = HALF * abi22
        shift = s1
    else
        as12 = a12 - s2 * b12
        as11 = a11 - s2 * b11
        ss = a21 * (binv11 * binv22)
        abi22 = -ss * b12
        pp = HALF * (as11 * binv11 + abi22)
        shift = s2
    end
    qq = ss * as12
    local discr, r
    if abs(pp * rtmin) >= ONE
        discr = (rtmin * pp)^2 + qq * safmin
        r = sqrt(abs(discr)) * rtmax
    elseif pp^2 + abs(qq) <= safmin
        discr = (rtmax * pp)^2 + qq * safmax
        r = sqrt(abs(discr)) * rtmin
    else
        discr = pp^2 + qq
        r = sqrt(abs(discr))
    end
    local wr1, wr2, wi
    if discr >= ZERO || r == ZERO
        sum_ = pp + copysign(r, pp)
        diff = pp - copysign(r, pp)
        wbig = shift + sum_
        wsmall = shift + diff
        if HALF * abs(wbig) > max(abs(wsmall), safmin)
            wdet = (a11 * a22 - a12 * a21) * (binv11 * binv22)
            wsmall = wdet / wbig
        end
        if pp > abi22
            wr1 = min(wbig, wsmall); wr2 = max(wbig, wsmall)
        else
            wr1 = max(wbig, wsmall); wr2 = min(wbig, wsmall)
        end
        wi = ZERO
    else
        wr1 = shift + pp; wr2 = wr1; wi = r
    end
    c1 = bsize * (safmin * max(ONE, ascale))
    c2 = safmin * max(ONE, bnorm)
    c3 = bsize * safmin
    c4 = (ascale <= ONE && bsize <= ONE) ? min(ONE, (ascale / safmin) * bsize) : ONE
    c5 = (ascale <= ONE || bsize <= ONE) ? min(ONE, ascale * bsize) : ONE
    wabs = abs(wr1) + abs(wi)
    wsize = max(safmin, c1, FUZZY1 * (wabs * c2 + c3), min(c4, HALF * max(wabs, c5)))
    local scale1, scale2
    if wsize != ONE
        wscale = ONE / wsize
        scale1 = wsize > ONE ? (max(ascale, bsize) * wscale) * min(ascale, bsize) :
                                (min(ascale, bsize) * wscale) * max(ascale, bsize)
        wr1 *= wscale
        if wi != ZERO
            wi *= wscale; wr2 = wr1; scale2 = scale1
        end
    else
        scale1 = ascale * bsize; scale2 = scale1
    end
    if wi == ZERO
        wsize2 = max(safmin, c1, FUZZY1 * (abs(wr2) * c2 + c3), min(c4, HALF * max(abs(wr2), c5)))
        if wsize2 != ONE
            wscale2 = ONE / wsize2
            scale2 = wsize2 > ONE ? (max(ascale, bsize) * wscale2) * min(ascale, bsize) :
                                     (min(ascale, bsize) * wscale2) * max(ascale, bsize)
            wr2 *= wscale2
        else
            scale2 = ascale * bsize
        end
    end
    return scale1, scale2, wr1, wr2, wi
end

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# DTGSEN (IJOB=0 path only — reorder + eigenvalues, no PL/PR/DIF; see file header)
# ═══════════════════════════════════════════════════════════════════════════════════════════════
function _dtgsen!(sel::AbstractVector{Bool}, A::AbstractMatrix{R}, B::AbstractMatrix{R},
        Q::AbstractMatrix{R}, Z::AbstractMatrix{R}) where {R<:Real}
    n = size(A, 1)
    ZERO = zero(R)
    alphar = zeros(R, n); alphai = zeros(R, n); beta = zeros(R, n)
    # Collect the selected blocks at the top-left corner of (A,B).
    ks = 0; pair = false
    for k in 1:n
        if pair
            pair = false
        else
            swap = sel[k]
            if k < n && A[k+1, k] != ZERO
                pair = true
                swap = swap || sel[k+1]
            end
            if swap
                ks += 1
                kk = k
                if kk != ks
                    info, _ = _dtgexc!(true, true, A, B, Q, Z, kk, ks)
                    info != 0 && throw(ErrorException(
                        "tgsen!: swap rejected (real path needs a 2×2 block reorder — unsupported; see src/tgsen.jl header)"))
                end
                pair && (ks += 1)
            end
        end
    end
    # Compute generalized eigenvalues of the reordered pair and normalize (sign of B's diagonal).
    k = 1
    @inbounds while k <= n
        if k < n && A[k+1, k] != ZERO
            a11 = A[k, k]; a21 = A[k+1, k]; a12 = A[k, k+1]; a22 = A[k+1, k+1]
            b11 = B[k, k]; b12 = B[k, k+1]; b22 = B[k+1, k+1]
            s1, s2, wr1, wr2, wi = _tgs_dlag2(a11, a21, a12, a22, b11, b12, b22, _syl_safmin(R))
            beta[k] = s1; beta[k+1] = s2
            alphar[k] = wr1; alphar[k+1] = wr2
            alphai[k] = wi; alphai[k+1] = -wi
            k += 2
        else
            if B[k, k] < ZERO
                for i in 1:n
                    A[k, i] = -A[k, i]; B[k, i] = -B[k, i]
                    Q[i, k] = -Q[i, k]
                end
            end
            alphar[k] = A[k, k]; alphai[k] = ZERO; beta[k] = B[k, k]
            k += 1
        end
    end
    return A, B, alphar, alphai, beta, Q, Z
end

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# ZTGEX2 (complex, always 1×1/1×1 — S,T strictly triangular, no 2×2 blocks) + ZTGEXC + ZTGSEN
# ═══════════════════════════════════════════════════════════════════════════════════════════════
function _ztgex2!(wantq::Bool, wantz::Bool, A::AbstractMatrix{C}, B::AbstractMatrix{C},
        Q::AbstractMatrix{C}, Z::AbstractMatrix{C}, j1::Int) where {C<:Complex}
    R = real(C)
    n = size(A, 1)
    Ao11 = A[j1, j1]; Ao12 = A[j1, j1+1]; Ao21 = A[j1+1, j1]; Ao22 = A[j1+1, j1+1]
    Bo11 = B[j1, j1]; Bo12 = B[j1, j1+1]; Bo22 = B[j1+1, j1+1]
    S11 = Ao11; S12 = Ao12; S21 = Ao21; S22 = Ao22
    T11 = Bo11; T12 = Bo12; T22 = Bo22
    eps_p = eps(R); smlnum = _syl_safmin(R) / eps_p
    dnorma = _tgs_f2norm(S11, S12, S21, S22)
    dnormb = _tgs_f2norm(T11, T12, zero(C), T22)
    thresha = max(R(20) * eps_p * dnorma, smlnum)
    threshb = max(R(20) * eps_p * dnormb, smlnum)
    F = S22 * T11 - T22 * S11
    G = S22 * T12 - T22 * S12
    SA = abs(S22) * abs(T11); SB = abs(S11) * abs(T22)
    cz, sz0, _ = _zlartg(G, F)                      # ZLARTG(G,F,CZ,SZ,·) — args swapped vs (F,G)
    sz = -sz0
    S11, S12 = _tgs_rot2(cz, conj(sz), S11, S12)
    S21, S22 = _tgs_rot2(cz, conj(sz), S21, S22)
    T11, T12 = _tgs_rot2(cz, conj(sz), T11, T12)
    T21 = zero(C)
    T21, T22 = _tgs_rot2(cz, conj(sz), T21, T22)
    cq, sq, _ = SA >= SB ? _zlartg(S11, S21) : _zlartg(T11, T21)
    S11, S21 = _tgs_rot2(cq, sq, S11, S21)
    S12, S22 = _tgs_rot2(cq, sq, S12, S22)
    T11, T21 = _tgs_rot2(cq, sq, T11, T21)
    T12, T22 = _tgs_rot2(cq, sq, T12, T22)
    weak = abs(S21) <= thresha && abs(T21) <= threshb
    weak || return 1
    # Strong test: re-apply the (negated-s) rotations to the tentative block and diff vs the original.
    W11 = S11; W12 = S12; W21 = S21; W22 = S22
    WT11 = T11; WT12 = T12; WT21 = T21; WT22 = T22
    W11, W12 = _tgs_rot2(cz, -conj(sz), W11, W12)
    W21, W22 = _tgs_rot2(cz, -conj(sz), W21, W22)
    WT11, WT12 = _tgs_rot2(cz, -conj(sz), WT11, WT12)
    WT21, WT22 = _tgs_rot2(cz, -conj(sz), WT21, WT22)
    W11, W21 = _tgs_rot2(cq, -sq, W11, W21)
    W12, W22 = _tgs_rot2(cq, -sq, W12, W22)
    WT11, WT21 = _tgs_rot2(cq, -sq, WT11, WT21)
    WT12, WT22 = _tgs_rot2(cq, -sq, WT12, WT22)
    SAf = _tgs_f2norm(W11 - Ao11, W12 - Ao12, W21 - Ao21, W22 - Ao22)
    SBf = _tgs_f2norm(WT11 - Bo11, WT12 - Bo12, WT21 - zero(C), WT22 - Bo22)
    (SAf <= thresha && SBf <= threshb) || return 1
    _zrot_cols!(A, j1, j1 + 1, 1, j1 + 1, cz, conj(sz))
    _zrot_cols!(B, j1, j1 + 1, 1, j1 + 1, cz, conj(sz))
    _zrot_rows!(A, j1, j1 + 1, j1, n, cq, sq)
    _zrot_rows!(B, j1, j1 + 1, j1, n, cq, sq)
    A[j1+1, j1] = zero(C); B[j1+1, j1] = zero(C)
    wantz && _zrot_cols!(Z, j1, j1 + 1, 1, n, cz, conj(sz))
    wantq && _zrot_cols!(Q, j1, j1 + 1, 1, n, cq, conj(sq))
    return 0
end

# ── ZTGEXC — sequential adjacent-swap sweep (no block-size bookkeeping needed: always 1×1) ───────
function _ztgexc!(wantq::Bool, wantz::Bool, A::AbstractMatrix{C}, B::AbstractMatrix{C},
        Q::AbstractMatrix{C}, Z::AbstractMatrix{C}, ifst::Int, ilst::Int) where {C<:Complex}
    n = size(A, 1)
    (n <= 1 || ifst == ilst) && return 0
    m1, m2, m3 = ifst < ilst ? (0, -1, 1) : (-1, 0, -1)
    for k in (ifst + m1):m3:(ilst + m2)
        info = _ztgex2!(wantq, wantz, A, B, Q, Z, k)
        info != 0 && return info
    end
    return 0
end

function _ztgsen!(sel::AbstractVector{Bool}, A::AbstractMatrix{C}, B::AbstractMatrix{C},
        Q::AbstractMatrix{C}, Z::AbstractMatrix{C}) where {C<:Complex}
    R = real(C)
    n = size(A, 1)
    ks = 0
    for k in 1:n
        if sel[k]
            ks += 1
            if k != ks
                info = _ztgexc!(true, true, A, B, Q, Z, k, ks)
                info != 0 && throw(ErrorException("tgsen!: swap rejected (ill-conditioned reorder)"))
            end
        end
    end
    alpha = Vector{C}(undef, n); beta = Vector{C}(undef, n)
    safmin = _syl_safmin(R)
    @inbounds for k in 1:n
        dscale = abs(B[k, k])
        if dscale > safmin
            temp2 = B[k, k] / dscale
            temp1 = conj(temp2)
            B[k, k] = C(dscale)
            for j in k+1:n; B[k, j] *= temp1; end
            for j in k:n; A[k, j] *= temp1; end
            for i in 1:n; Q[i, k] *= temp2; end
        else
            B[k, k] = zero(C)
        end
        alpha[k] = A[k, k]; beta[k] = B[k, k]
    end
    return A, B, alpha, beta, Q, Z
end

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# Public API — mirrors `LinearAlgebra.LAPACK.tgsen!(select, S, T, Q, Z)`
# ═══════════════════════════════════════════════════════════════════════════════════════════════
"""
    tgsen!(select, S, T, Q, Z) -> (S, T, alpha, beta, Q, Z)

Reorder the generalized Schur pair `(S,T)` (`S` (quasi-)upper-triangular, `T` upper-triangular,
`Q·S·Zᴴ`/`Q·T·Zᴴ` the original matrix pair) so that the generalized eigenvalues selected by
`select` occupy the leading block, updating `Q`, `Z` in place. `alpha`/`beta` (length `n`, `alpha`
complex, `beta` real for real `S`/`T` — complex for complex `S`/`T`) are the reordered generalized
eigenvalues `alpha[i]/beta[i]`. Mirrors LAPACK `dtgsen`/`ztgsen` at `ijob=0` (reorder only; no
condition-number estimates), matching `LinearAlgebra.LAPACK.tgsen!`'s contract exactly.

REAL path limitation (honest, see file header): only reorders that never touch a 2×2 (complex-
conjugate-pair) diagonal block are supported; such a swap throws rather than silently mis-reorder.
Matrix pairs with exclusively real generalized eigenvalues are unaffected. The COMPLEX path is
complete (no 2×2 blocks exist in a complex Schur form).
"""
function tgsen!(select::AbstractVector, S::AbstractMatrix{R}, T::AbstractMatrix{R},
        Q::AbstractMatrix{R}, Z::AbstractMatrix{R}) where {R<:Real}
    n = size(S, 1)
    (n == size(S, 2) == size(T, 1) == size(T, 2) == size(Q, 1) == size(Q, 2) ==
        size(Z, 1) == size(Z, 2)) ||
        throw(DimensionMismatch("tgsen!: S, T, Q, Z must be square of matching size"))
    length(select) == n || throw(DimensionMismatch("tgsen!: select must have length n"))
    sel = Bool[select[i] != 0 for i in 1:n]
    Ar, Br, alphar, alphai, beta, Qr, Zr = _dtgsen!(sel, S, T, Q, Z)
    alpha = Complex{R}.(alphar, alphai)
    return Ar, Br, alpha, beta, Qr, Zr
end

function tgsen!(select::AbstractVector, S::AbstractMatrix{C}, T::AbstractMatrix{C},
        Q::AbstractMatrix{C}, Z::AbstractMatrix{C}) where {C<:Complex}
    n = size(S, 1)
    (n == size(S, 2) == size(T, 1) == size(T, 2) == size(Q, 1) == size(Q, 2) ==
        size(Z, 1) == size(Z, 2)) ||
        throw(DimensionMismatch("tgsen!: S, T, Q, Z must be square of matching size"))
    length(select) == n || throw(DimensionMismatch("tgsen!: select must have length n"))
    sel = Bool[select[i] != 0 for i in 1:n]
    return _ztgsen!(sel, S, T, Q, Z)
end
