# LAPACK generalized Schur reorder (dtgsen/ztgsen): given a generalized Schur pair (S,T) — S
# (quasi-)upper-triangular, T upper-triangular, with orthonormal/unitary Q,Z such that
# Q·S·Zᴴ = A, Q·T·Zᴴ = B — move the eigenvalues selected by `select` to the leading block via a
# sequence of adjacent-block swaps (dtgexc/ztgexc, built on dtgex2/ztgex2), then read off the
# generalized eigenvalues (alpha,beta) from the reordered diagonal. Port of Reference-LAPACK
# dtgsen/dtgexc/dtgex2 (with its dtgsy2/dgetc2/dgesc2/dlagv2/dgeqr2/dorg2r/dgerq2/dorgr2/dorm2r/
# dormr2 auxiliaries, all ≤4×4 here)/dlag2 and ztgsen/ztgexc/ztgex2, restricted to the IJOB=0 (no
# PL/PR/DIF condition numbers) path — the only path Julia's `LinearAlgebra.LAPACK.tgsen!` wrapper
# ever drives (it always calls with ijob=0 and PL/PR/DIF as C_NULL), so DTGSYL is never needed.
#
# STANDALONE: needs `_syl_safmin` (trsyl.jl) and `_lartg`/`_zlartg`/`_lasv2`/`_qz_larfg!`/
# `_grot_rows!`/`_grot_cols!`/`_zrot_rows!`/`_zrot_cols!` (qz.jl) in scope — both are already
# `include`d by PureBLAS.jl before this file, so `using PureBLAS` is sufficient; only a raw
# standalone load needs them first.
#
# REAL path dtgex2 covers BOTH Reference-LAPACK cases: (1) 1×1↔1×1 (a single Givens sweep,
# `_dtgex2_1x1!`) and (2) any combination touching a 2×2 block — 1×1↔2×2, 2×2↔1×1, 2×2↔2×2
# (`_dtgex2_big!`: generalized Sylvester solve (dtgsy2 single-block, complete-pivot LU with
# dgesc2 scaling), Householder QR/RQ of the ≤4×4 swap factors, LAPACK's weak+strong stability
# tests, and `dlagv2` re-standardization of the new 2×2 blocks). A swap either succeeds or is
# rejected with info=1 exactly when LAPACK would reject it (ill-conditioned swap), and `tgsen!`
# then throws — matching the LAPACKException(1) Julia's wrapper raises. The COMPLEX path has no
# 2×2 blocks at all (S,T both strictly triangular) so ztgex2 only ever needs case (1).

# Givens pair-update matching LAPACK's DROT/ZROT: (x,y) ↦ (c·x+s·y, c·y−conj(s)·x). Generic (conj
# is the identity on Real), used identically by the real and complex ports below.
@inline _tgs_rot2(c, s, x, y) = (c * x + s * y, c * y - conj(s) * x)
# Frobenius norm of a 2×2 block given its 4 entries (generic real/complex via abs2).
@inline _tgs_f2norm(a, b, c, d) = sqrt(abs2(a) + abs2(b) + abs2(c) + abs2(d))

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# DTGEX2 (real, 1×1↔1×1 case) — swap adjacent 1×1 diagonal blocks of (A,B) at (j1,j1+1).
# Returns info (0 = swapped; 1 = swap rejected: failed the LAPACK weak/strong stability test).
# ═══════════════════════════════════════════════════════════════════════════════════════════════
function _dtgex2_1x1!(
        wantq::Bool, wantz::Bool, A::AbstractMatrix{R}, B::AbstractMatrix{R},
        Q::AbstractMatrix{R}, Z::AbstractMatrix{R}, j1::Int
    ) where {R <: Real}
    n = size(A, 1)
    ZERO = zero(R); TWENTY = R(20)
    Ao11 = A[j1, j1]; Ao12 = A[j1, j1 + 1]; Ao21 = A[j1 + 1, j1]; Ao22 = A[j1 + 1, j1 + 1]
    Bo11 = B[j1, j1]; Bo12 = B[j1, j1 + 1]; Bo22 = B[j1 + 1, j1 + 1]
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
    A[j1 + 1, j1] = ZERO; B[j1 + 1, j1] = ZERO
    wantz && _grot_cols!(Z, j1, j1 + 1, 1, n, ir11, ir21)
    wantq && _grot_cols!(Q, j1, j1 + 1, 1, n, li11, li21)
    return 0
end

# ── Small dense helpers for the ≤4×4 dtgex2 workspace. Cold path (reorder driver): plain loops on
# tiny Matrix{R} temporaries; allocation is fine here and everything is trim-safe (no eval/closures).

# Scaled Frobenius norm of A[i1:i2, jl:jh] (DLASSQ-style over/underflow safety).
function _tgs_fnorm(A::AbstractMatrix{R}, i1::Int, i2::Int, jl::Int, jh::Int) where {R <: Real}
    amax = zero(R)
    @inbounds for j in jl:jh, i in i1:i2
        amax = max(amax, abs(A[i, j]))
    end
    (amax == zero(R) || !isfinite(amax)) && return amax
    ss = zero(R)
    @inbounds for j in jl:jh, i in i1:i2
        t = A[i, j] / amax; ss = muladd(t, t, ss)
    end
    return amax * sqrt(ss)
end

# C = op(A)·op(B) (op = transpose iff flag), small dense, into a caller-supplied C. Every element of
# C[1:m, 1:n] is assigned, so C never needs zeroing. C must not alias A or B (all call sites pass a
# workspace ping/pong buffer distinct from both operands).
function _tgs_matmul!(C::AbstractMatrix{R}, ta::Bool, tb::Bool, A::AbstractMatrix{R}, B::AbstractMatrix{R}) where {R <: Real}
    m = ta ? size(A, 2) : size(A, 1)
    kk = ta ? size(A, 1) : size(A, 2)
    n = tb ? size(B, 1) : size(B, 2)
    @inbounds for j in 1:n, i in 1:m
        s = zero(R)
        for k in 1:kk
            a = ta ? A[k, i] : A[i, k]
            b = tb ? B[j, k] : B[k, j]
            s = muladd(a, b, s)
        end
        C[i, j] = s
    end
    return C
end

# ── DGEQR2 on A[1:m, 1:k] (Householder QR, reflectors stored in place, τ[1:k] filled) ────────────
# `v` is caller-supplied staging (length ≥ m); v[1:lv] is written before every `_qz_larfg!`.
function _tgs_geqr2!(A::AbstractMatrix{R}, m::Int, k::Int, τ::AbstractVector{R}, v::AbstractVector{R}) where {R <: Real}
    @inbounds for i in 1:k
        lv = m - i + 1
        for t in 1:lv
            v[t] = A[i + t - 1, i]
        end
        τi = _qz_larfg!(v, lv)
        τ[i] = τi
        for t in 1:lv
            A[i + t - 1, i] = v[t]
        end
        if τi != zero(R)
            for j in (i + 1):k
                s = A[i, j]
                for t in 2:lv
                    s = muladd(v[t], A[i + t - 1, j], s)
                end
                s *= τi
                A[i, j] -= s
                for t in 2:lv
                    A[i + t - 1, j] -= s * v[t]
                end
            end
        end
    end
    return A
end

# ── DORG2R: overwrite A with the full m×m Q of the k reflectors stored by _tgs_geqr2! ────────────
function _tgs_org2r!(A::AbstractMatrix{R}, m::Int, k::Int, τ::AbstractVector{R}) where {R <: Real}
    @inbounds begin
        for j in (k + 1):m
            for l in 1:m
                A[l, j] = zero(R)
            end
            A[j, j] = one(R)
        end
        for i in k:-1:1
            τi = τ[i]
            if i < m
                for j in (i + 1):m
                    s = A[i, j]
                    for t in (i + 1):m
                        s = muladd(A[t, i], A[t, j], s)
                    end
                    s *= τi
                    A[i, j] -= s
                    for t in (i + 1):m
                        A[t, j] -= s * A[t, i]
                    end
                end
                for t in (i + 1):m
                    A[t, i] = -τi * A[t, i]
                end
            end
            A[i, i] = one(R) - τi
            for l in 1:(i - 1)
                A[l, i] = zero(R)
            end
        end
    end
    return A
end

# ── DGERQ2 on the k×n submatrix A[r0+1:r0+k, 1:n] (RQ; reflectors in place, τ[1:k]) ──────────────
# `v` is caller-supplied staging (length ≥ n); v[1:e] is written before every `_qz_larfg!`.
function _tgs_gerq2!(A::AbstractMatrix{R}, r0::Int, k::Int, n::Int, τ::AbstractVector{R}, v::AbstractVector{R}) where {R <: Real}
    @inbounds for i in k:-1:1
        ri = r0 + i
        e = n - k + i
        v[1] = A[ri, e]
        for t in 1:(e - 1)
            v[1 + t] = A[ri, t]
        end
        τi = _qz_larfg!(v, e)
        τ[i] = τi
        A[ri, e] = v[1]
        for t in 1:(e - 1)
            A[ri, t] = v[1 + t]
        end
        if τi != zero(R)
            for r in (r0 + 1):(ri - 1)
                s = A[r, e]
                for t in 1:(e - 1)
                    s = muladd(A[r, t], v[1 + t], s)
                end
                s *= τi
                A[r, e] -= s
                for t in 1:(e - 1)
                    A[r, t] -= s * v[1 + t]
                end
            end
        end
    end
    return A
end

# ── DORGR2: overwrite the m×m A with the full Q of the k RQ reflectors (stored in rows m−k+1..m) ─
function _tgs_orgr2!(A::AbstractMatrix{R}, m::Int, k::Int, τ::AbstractVector{R}) where {R <: Real}
    @inbounds begin
        if k < m
            for j in 1:m
                for l in 1:(m - k)
                    A[l, j] = zero(R)
                end
                j <= m - k && (A[j, j] = one(R))
            end
        end
        for i in 1:k
            ii = m - k + i
            τi = τ[i]
            if τi != zero(R)
                for r in 1:(ii - 1)
                    s = A[r, ii]
                    for t in 1:(ii - 1)
                        s = muladd(A[r, t], A[ii, t], s)
                    end
                    s *= τi
                    A[r, ii] -= s
                    for t in 1:(ii - 1)
                        A[r, t] -= s * A[ii, t]
                    end
                end
            end
            for t in 1:(ii - 1)
                A[ii, t] = -τi * A[ii, t]
            end
            A[ii, ii] = one(R) - τi
            for l in (ii + 1):m
                A[ii, l] = zero(R)
            end
        end
    end
    return A
end

# ── DORM2R (square m×m factor F, k=m): apply Q ('N') or Qᵀ ('T') to C from side 'L'/'R' ──────────
function _tgs_orm2r!(
        left::Bool, trans::Bool, m::Int, F::AbstractMatrix{R}, τ::AbstractVector{R},
        C::AbstractMatrix{R}
    ) where {R <: Real}
    ir = (left != trans) ? (m:-1:1) : (1:1:m)      # (L,N)/(R,T) descending; (L,T)/(R,N) ascending
    @inbounds for i in ir
        τi = τ[i]
        τi == zero(R) && continue
        if left
            for j in 1:m
                s = C[i, j]
                for t in (i + 1):m
                    s = muladd(F[t, i], C[t, j], s)
                end
                s *= τi
                C[i, j] -= s
                for t in (i + 1):m
                    C[t, j] -= s * F[t, i]
                end
            end
        else
            for r in 1:m
                s = C[r, i]
                for t in (i + 1):m
                    s = muladd(C[r, t], F[t, i], s)
                end
                s *= τi
                C[r, i] -= s
                for t in (i + 1):m
                    C[r, t] -= s * F[t, i]
                end
            end
        end
    end
    return C
end

# ── DORMR2 (square m×m factor F, k=m): row-stored RQ reflectors, side 'L'/'R', op 'N'/'T' ────────
function _tgs_ormr2!(
        left::Bool, trans::Bool, m::Int, F::AbstractMatrix{R}, τ::AbstractVector{R},
        C::AbstractMatrix{R}
    ) where {R <: Real}
    ir = (left != trans) ? (m:-1:1) : (1:1:m)
    @inbounds for i in ir
        τi = τ[i]
        τi == zero(R) && continue
        if left
            for j in 1:m
                s = C[i, j]
                for t in 1:(i - 1)
                    s = muladd(F[i, t], C[t, j], s)
                end
                s *= τi
                C[i, j] -= s
                for t in 1:(i - 1)
                    C[t, j] -= s * F[i, t]
                end
            end
        else
            for r in 1:m
                s = C[r, i]
                for t in 1:(i - 1)
                    s = muladd(C[r, t], F[i, t], s)
                end
                s *= τi
                C[r, i] -= s
                for t in 1:(i - 1)
                    C[r, t] -= s * F[i, t]
                end
            end
        end
    end
    return C
end

# ── DGETC2: LU with complete pivoting on the nz×nz Z (tiny pivots perturbed to smin) ─────────────
function _tgs_getc2!(Z::AbstractMatrix{R}, nz::Int, ip::AbstractVector{Int}, jp::AbstractVector{Int}) where {R <: Real}
    ZERO = zero(R)
    info = 0
    eps_p = eps(R); smlnum = _syl_safmin(R) / eps_p
    smin = ZERO
    @inbounds begin
        for i in 1:(nz - 1)
            xmax = ZERO; ipv = i; jpv = i
            for jc in i:nz, ic in i:nz
                if abs(Z[ic, jc]) >= xmax
                    xmax = abs(Z[ic, jc]); ipv = ic; jpv = jc
                end
            end
            i == 1 && (smin = max(eps_p * xmax, smlnum))
            if ipv != i
                for c in 1:nz
                    Z[ipv, c], Z[i, c] = Z[i, c], Z[ipv, c]
                end
            end
            ip[i] = ipv
            if jpv != i
                for r in 1:nz
                    Z[r, jpv], Z[r, i] = Z[r, i], Z[r, jpv]
                end
            end
            jp[i] = jpv
            if abs(Z[i, i]) < smin
                info = i; Z[i, i] = smin
            end
            for r in (i + 1):nz
                Z[r, i] /= Z[i, i]
            end
            for c in (i + 1):nz, r in (i + 1):nz
                Z[r, c] -= Z[r, i] * Z[i, c]
            end
        end
        if abs(Z[nz, nz]) < smin
            info = nz; Z[nz, nz] = smin
        end
        ip[nz] = nz; jp[nz] = nz
    end
    return info
end

# ── DGESC2: solve Z·x = scale·rhs from the _tgs_getc2! factors (scale ≤ 1 guards overflow) ───────
function _tgs_gesc2!(
        Z::AbstractMatrix{R}, nz::Int, rhs::AbstractVector{R}, ip::AbstractVector{Int},
        jp::AbstractVector{Int}
    ) where {R <: Real}
    ONE = one(R); TWO = R(2)
    eps_p = eps(R); smlnum = _syl_safmin(R) / eps_p
    @inbounds begin
        for i in 1:(nz - 1)
            p = ip[i]
            rhs[i], rhs[p] = rhs[p], rhs[i]
        end
        for i in 1:(nz - 1), j in (i + 1):nz
            rhs[j] -= Z[j, i] * rhs[i]
        end
        scale = ONE
        imax = 1
        for i in 2:nz
            abs(rhs[i]) > abs(rhs[imax]) && (imax = i)
        end
        if TWO * smlnum * abs(rhs[imax]) > abs(Z[nz, nz])
            temp = (ONE / TWO) / abs(rhs[imax])
            for i in 1:nz
                rhs[i] *= temp
            end
            scale *= temp
        end
        for i in nz:-1:1
            temp = ONE / Z[i, i]
            rhs[i] *= temp
            for j in (i + 1):nz
                rhs[i] -= rhs[j] * (Z[i, j] * temp)
            end
        end
        for i in (nz - 1):-1:1
            p = jp[i]
            rhs[i], rhs[p] = rhs[p], rhs[i]
        end
    end
    return scale
end

# ── DTGSY2 restricted to ONE diagonal block pair (n1,n2 ≤ 2, exactly dtgex2's use): solve
#      S11·Rm − Lm·S22 = scale·C ,   T11·Rm − Lm·T22 = scale·F
# via the 2·n1·n2 Kronecker system with complete-pivot LU. Overwrites C←Rm, F←Lm. Returns
# (scale, ierr); ierr > 0 ⟺ a pivot was perturbed (caller rejects the swap, as dtgex2 does).
function _tgs_tgsy2!(
        S::AbstractMatrix{R}, Tm::AbstractMatrix{R}, n1::Int, n2::Int,
        C::AbstractMatrix{R}, Fm::AbstractMatrix{R}
    ) where {R <: Real}
    p = n1 * n2; nz = 2 * p
    @scope arn begin
        # Z IS ZEROED HERE and that is load-bearing: it is built entirely by the `+=`/`-=` accumulation
        # below, with no prior full write. rhs/ip/jp are fully written before any read, so they take the
        # borrow as it comes. All four are exact (nz ≤ 8) and die with the scope.
        Z = borrow!(arn, R, nz, nz); fill!(Z, zero(R))
        rhs = borrow!(arn, R, nz)
        ip = borrow!(arn, Int, nz)
        jp = borrow!(arn, Int, nz)
        @inbounds for j in 1:n2, i in 1:n1
            r = (j - 1) * n1 + i
            for k in 1:n1                       # + S11[i,k]·Rm[k,j] / + T11[i,k]·Rm[k,j]
                Z[r, (j - 1) * n1 + k] += S[i, k]
                Z[p + r, (j - 1) * n1 + k] += Tm[i, k]
            end
            for k in 1:n2                       # − Lm[i,k]·S22[k,j] / − Lm[i,k]·T22[k,j]
                Z[r, p + (k - 1) * n1 + i] -= S[n1 + k, n1 + j]
                Z[p + r, p + (k - 1) * n1 + i] -= Tm[n1 + k, n1 + j]
            end
            rhs[r] = C[i, j]
            rhs[p + r] = Fm[i, j]
        end
        ierr = _tgs_getc2!(Z, nz, ip, jp)
        scale = _tgs_gesc2!(Z, nz, rhs, ip, jp)
        @inbounds for j in 1:n2, i in 1:n1
            r = (j - 1) * n1 + i
            C[i, j] = rhs[r]
            Fm[i, j] = rhs[p + r]
        end
        return scale, ierr
    end
end

# ── DLAGV2: standardize the 2×2 diagonal block of (A,B) at (p,p) — B → triangular with the LAPACK
# sign convention, A → standardized 2×2 (complex pair) or triangular (real pair; block splits).
# In-place on the 2×2 block only; returns the applied rotations (csl, snl, csr, snr).
function _tgs_lagv2!(A::AbstractMatrix{R}, B::AbstractMatrix{R}, p::Int) where {R <: Real}
    ZERO = zero(R); ONE = one(R)
    safmin = _syl_safmin(R); ulp = eps(R)
    q = p + 1
    a11 = A[p, p]; a12 = A[p, q]; a21 = A[q, p]; a22 = A[q, q]
    b11 = B[p, p]; b12 = B[p, q]; b22 = B[q, q]; b21 = ZERO
    anorm = max(abs(a11) + abs(a21), abs(a12) + abs(a22), safmin)
    ascale = ONE / anorm
    a11 *= ascale; a12 *= ascale; a21 *= ascale; a22 *= ascale
    bnorm = max(abs(b11), abs(b12) + abs(b22), safmin)
    bscale = ONE / bnorm
    b11 *= bscale; b12 *= bscale; b22 *= bscale
    local csl, snl, csr, snr
    if abs(a21) <= ulp
        csl = ONE; snl = ZERO; csr = ONE; snr = ZERO
        a21 = ZERO; b21 = ZERO
    elseif abs(b11) <= ulp
        csl, snl, _ = _lartg(a11, a21)
        csr = ONE; snr = ZERO
        a11, a21 = _tgs_rot2(csl, snl, a11, a21)
        a12, a22 = _tgs_rot2(csl, snl, a12, a22)
        b11, b21 = _tgs_rot2(csl, snl, b11, b21)
        b12, b22 = _tgs_rot2(csl, snl, b12, b22)
        a21 = ZERO; b11 = ZERO; b21 = ZERO
    elseif abs(b22) <= ulp
        csr, snr0, _ = _lartg(a22, a21)
        snr = -snr0
        a11, a12 = _tgs_rot2(csr, snr, a11, a12)
        a21, a22 = _tgs_rot2(csr, snr, a21, a22)
        b11, b12 = _tgs_rot2(csr, snr, b11, b12)
        b21, b22 = _tgs_rot2(csr, snr, b21, b22)
        csl = ONE; snl = ZERO
        a21 = ZERO; b21 = ZERO; b22 = ZERO
    else
        scale1, _, wr1, _, wi = _tgs_dlag2(a11, a21, a12, a22, b11, b12, b22, safmin)
        if wi == ZERO
            # real eigenvalues: the block splits into two 1×1s
            h1 = scale1 * a11 - wr1 * b11
            h2 = scale1 * a12 - wr1 * b12
            h3 = scale1 * a22 - wr1 * b22
            rr = hypot(h1, h2)
            qq = hypot(scale1 * a21, h3)
            local snr0
            if rr > qq
                csr, snr0, _ = _lartg(h2, h1)
            else
                csr, snr0, _ = _lartg(h3, scale1 * a21)
            end
            snr = -snr0
            a11, a12 = _tgs_rot2(csr, snr, a11, a12)
            a21, a22 = _tgs_rot2(csr, snr, a21, a22)
            b11, b12 = _tgs_rot2(csr, snr, b11, b12)
            b21, b22 = _tgs_rot2(csr, snr, b21, b22)
            h1 = max(abs(a11) + abs(a12), abs(a21) + abs(a22))
            h2 = max(abs(b11) + abs(b12), abs(b21) + abs(b22))
            if scale1 * h1 >= abs(wr1) * h2
                csl, snl, _ = _lartg(b11, b21)
            else
                csl, snl, _ = _lartg(a11, a21)
            end
            a11, a21 = _tgs_rot2(csl, snl, a11, a21)
            a12, a22 = _tgs_rot2(csl, snl, a12, a22)
            b11, b21 = _tgs_rot2(csl, snl, b11, b21)
            b12, b22 = _tgs_rot2(csl, snl, b12, b22)
            a21 = ZERO; b21 = ZERO
        else
            # complex pair: B → diagonal via its 2×2 SVD rotations
            _, _, snr, csr, snl, csl = _lasv2(b11, b12, b22)
            a11, a21 = _tgs_rot2(csl, snl, a11, a21)
            a12, a22 = _tgs_rot2(csl, snl, a12, a22)
            b11, b21 = _tgs_rot2(csl, snl, b11, b21)
            b12, b22 = _tgs_rot2(csl, snl, b12, b22)
            a11, a12 = _tgs_rot2(csr, snr, a11, a12)
            a21, a22 = _tgs_rot2(csr, snr, a21, a22)
            b11, b12 = _tgs_rot2(csr, snr, b11, b12)
            b21, b22 = _tgs_rot2(csr, snr, b21, b22)
            b21 = ZERO; b12 = ZERO
        end
    end
    A[p, p] = anorm * a11; A[q, p] = anorm * a21; A[p, q] = anorm * a12; A[q, q] = anorm * a22
    B[p, p] = bnorm * b11; B[q, p] = bnorm * b21; B[p, q] = bnorm * b12; B[q, q] = bnorm * b22
    return csl, snl, csr, snr
end

# ═══ DTGEX2 general case (m = n1+n2 ∈ {3,4}) — Reference-LAPACK dtgex2.f "CASE 2" verbatim ═══════
# Swap the adjacent n1×n1 / n2×n2 blocks at (j1,j1): generalized Sylvester solve for the swap
# transforms, QR/RQ of the [−L; scale·I] / [scale·I, R] factors, tentative swap, weak + strong
# stability tests (reject with info=1), then dlagv2 re-standardization of any new 2×2 blocks.
function _dtgex2_big!(
        wantq::Bool, wantz::Bool, A::AbstractMatrix{R}, B::AbstractMatrix{R},
        Q::AbstractMatrix{R}, Z::AbstractMatrix{R}, j1::Int, n1::Int, n2::Int
    ) where {R <: Real}
    ZERO = zero(R); ONE = one(R); TWENTY = R(20)
    n = size(A, 1)
    m = n1 + n2
    jm = j1 + m - 1
    jm <= n || return 0                    # LAPACK dtgex2 early no-op guard
    # Every temporary below is a fixed-size arena borrow: n1,n2 ∈ {1,2} at every `_dtgex2!` call site, so
    # m = n1+n2 ≤ 4 and nz = 2·n1·n2 ≤ 8. `_tgs_matmul` used to RETURN a fresh matrix and the caller
    # rebound S/Tm/LI/IR; with borrowed buffers those rebinds are `copyto!`s and the two nested products
    # run through the M1/M2 ping-pong pair. The scope spans the whole routine because LI/IR/w are read by
    # the off-block update at the very end; every borrow sits at the top level of it, none in a loop.
    @scope arn begin
        # Diagonal blocks. All four are fully written by the copy loops below, so no fill!. Borrowed EXACTLY
        # so a shape bug trips @boundscheck rather than reading a stale corner.
        S = borrow!(arn, R, m, m)
        Tm = borrow!(arn, R, m, m)
        Cm = borrow!(arn, R, n1, n2)
        Fm = borrow!(arn, R, n1, n2)
        @inbounds for j in 1:m, i in 1:m
            S[i, j] = A[j1 + i - 1, j1 + j - 1]
            Tm[i, j] = B[j1 + i - 1, j1 + j - 1]
        end
        eps_p = eps(R); smlnum = _syl_safmin(R) / eps_p
        dnorma = _tgs_fnorm(S, 1, m, 1, m)
        dnormb = _tgs_fnorm(Tm, 1, m, 1, m)
        thresha = max(TWENTY * eps_p * dnorma, smlnum)
        threshb = max(TWENTY * eps_p * dnormb, smlnum)
        # generalized Sylvester solve: S11·Rm − Lm·S22 = scale·S12, T11·Rm − Lm·T22 = scale·T12
        @inbounds for j in 1:n2, i in 1:n1
            Cm[i, j] = S[i, n1 + j]
            Fm[i, j] = Tm[i, n1 + j]
        end
        scale, ierr = _tgs_tgsy2!(S, Tm, n1, n2, Cm, Fm)
        ierr > 0 && return 1
        # left transform: QR of [−Lm; scale·I_n2]  →  full m×m Q in LI
        # LI AND IR ARE ZEROED AT THE BORROW, both load-bearing: only LI[1:n1,1:n2] plus a unit sub-diagonal
        # is written here yet `_tgs_geqr2!` reads all of LI[1:m,1:n2]; IR has only two sparse patterns
        # written yet `_tgs_gerq2!` reads rows n2+1:n2+n1 across all of cols 1:m. (Zeroing the whole m×m also
        # covers the columns `_tgs_org2r!`/`_tgs_orgr2!` would have cleared themselves — 16 elements.)
        # The τ and v borrows need none: v[1..lv] is filled before each `_qz_larfg!` and τ[i] assigned before
        # use. `vq`/`vr` stay separate: they belong to `_tgs_geqr2!` and `_tgs_gerq2!`, two distinct helpers.
        LI = borrow!(arn, R, m, m); fill!(LI, ZERO)
        IR = borrow!(arn, R, m, m); fill!(IR, ZERO)
        τl = borrow!(arn, R, m)
        τr = borrow!(arn, R, m)
        τl2 = borrow!(arn, R, m)
        τr2 = borrow!(arn, R, m)
        vq = borrow!(arn, R, 4)
        vr = borrow!(arn, R, 4)
        @inbounds for j in 1:n2
            for i in 1:n1
                LI[i, j] = -Fm[i, j]
            end
            LI[n1 + j, j] = scale
        end
        _tgs_geqr2!(LI, m, n2, τl, vq)
        _tgs_org2r!(LI, m, n2, τl)
        # right transform: RQ of [scale·I_n1, Rm] (rows n2+1..m)  →  full m×m Q in IR
        @inbounds begin
            for j in 1:n2, i in 1:n1
                IR[n2 + i, n1 + j] = Cm[i, j]
            end
            for i in 1:n1
                IR[n2 + i, i] = scale
            end
        end
        _tgs_gerq2!(IR, n2, n1, m, τr, vr)
        _tgs_orgr2!(IR, m, n1, τr)
        # tentative swap: S ← LIᵀ·S·IRᵀ, T likewise (M1 = inner product, M2 = outer; both live at once)
        # M1/M2 are a ping/pong pair — the products nest inner-inside-outer, so BOTH are live at once; PA/PB
        # are the two independent outer products. Every element is assigned by `_tgs_matmul!`'s own loop.
        M1 = borrow!(arn, R, m, m)
        M2 = borrow!(arn, R, m, m)
        PA = borrow!(arn, R, m, m)
        PB = borrow!(arn, R, m, m)
        _tgs_matmul!(M1, true, false, LI, S); _tgs_matmul!(M2, false, true, M1, IR); copyto!(S, M2)
        _tgs_matmul!(M1, true, false, LI, Tm); _tgs_matmul!(M2, false, true, M1, IR); copyto!(Tm, M2)
        # The four route-selection copies. Fully written by the copyto! below, no fill!.
        SCPY = borrow!(arn, R, m, m)
        TCPY = borrow!(arn, R, m, m)
        IRCOP = borrow!(arn, R, m, m)
        LICOP = borrow!(arn, R, m, m)
        copyto!(SCPY, S); copyto!(TCPY, Tm); copyto!(IRCOP, IR); copyto!(LICOP, LI)
        # route 1: triangularize T by RQ (transform from the right)
        _tgs_gerq2!(Tm, 0, m, m, τr2, vr)
        _tgs_ormr2!(false, true, m, Tm, τr2, S)      # S ← S·Qᵀ
        _tgs_ormr2!(true, false, m, Tm, τr2, IR)     # IR ← Q·IR
        brqa21 = _tgs_fnorm(S, n2 + 1, m, 1, n2)
        # route 2: triangularize T by QR (transform from the left)
        _tgs_geqr2!(TCPY, m, m, τl2, vq)
        _tgs_orm2r!(true, true, m, TCPY, τl2, SCPY)  # SCPY ← Qᵀ·SCPY
        _tgs_orm2r!(false, false, m, TCPY, τl2, LICOP) # LICOP ← LICOP·Q
        bqra21 = _tgs_fnorm(SCPY, n2 + 1, m, 1, n2)
        # weak stability test — pick the better route (a rebind of four names before the buffers were owned)
        if bqra21 <= brqa21 && bqra21 <= thresha
            copyto!(S, SCPY); copyto!(Tm, TCPY); copyto!(IR, IRCOP); copyto!(LI, LICOP)
        elseif brqa21 >= thresha
            return 1
        end
        @inbounds for j in 1:(m - 1), i in (j + 1):m
            Tm[i, j] = ZERO
        end
        # strong stability test: ‖A_blk − LI·S·IR‖_F ≤ thresha and ‖B_blk − LI·T·IR‖_F ≤ threshb
        _tgs_matmul!(M1, false, false, LI, S); _tgs_matmul!(PA, false, false, M1, IR)
        _tgs_matmul!(M1, false, false, LI, Tm); _tgs_matmul!(PB, false, false, M1, IR)
        @inbounds for j in 1:m, i in 1:m
            PA[i, j] = A[j1 + i - 1, j1 + j - 1] - PA[i, j]
            PB[i, j] = B[j1 + i - 1, j1 + j - 1] - PB[i, j]
        end
        sa = _tgs_fnorm(PA, 1, m, 1, m)
        sb = _tgs_fnorm(PB, 1, m, 1, m)
        (sa <= thresha && sb <= threshb) || return 1
        # accepted: zero the (2,1) block and copy the swapped pair back
        @inbounds for j in 1:n2, i in (n2 + 1):m
            S[i, j] = ZERO
        end
        @inbounds for j in 1:m, i in 1:m
            A[j1 + i - 1, j1 + j - 1] = S[i, j]
            B[j1 + i - 1, j1 + j - 1] = Tm[i, j]
        end
        # re-standardize the new 2×2 blocks (dlagv2) and fold those rotations into LI / IR
        # QL2/IR2 ARE ZEROED AT THE BORROW, load-bearing: only their (1,1)/(m,m) entries and up to two 2×2
        # corner blocks are set below, yet both are then read IN FULL by `_tgs_matmul!` and the tmpA/tmpB
        # loops. tmpA/tmpB/w are fully written before read.
        QL2 = borrow!(arn, R, m, m); fill!(QL2, ZERO)
        IR2 = borrow!(arn, R, m, m); fill!(IR2, ZERO)
        tmpA = borrow!(arn, R, n2, n1)
        tmpB = borrow!(arn, R, n2, n1)
        w = borrow!(arn, R, m)
        QL2[1, 1] = ONE; IR2[1, 1] = ONE
        QL2[m, m] = ONE; IR2[m, m] = ONE
        if n2 > 1
            csl, snl, csr, snr = _tgs_lagv2!(A, B, j1)
            QL2[1, 1] = csl; QL2[2, 1] = snl; QL2[1, 2] = -snl; QL2[2, 2] = csl
            IR2[1, 1] = csr; IR2[2, 1] = snr; IR2[1, 2] = -snr; IR2[2, 2] = csr
        end
        if n1 > 1
            csl, snl, csr, snr = _tgs_lagv2!(A, B, j1 + n2)
            QL2[n2 + 1, n2 + 1] = csl; QL2[n2 + 2, n2 + 1] = snl
            QL2[n2 + 1, n2 + 2] = -snl; QL2[n2 + 2, n2 + 2] = csl
            IR2[n2 + 1, n2 + 1] = csr; IR2[n2 + 2, n2 + 1] = snr
            IR2[n2 + 1, n2 + 2] = -snr; IR2[n2 + 2, n2 + 2] = csr
        end
        # off-diagonal (1:n2, n2+1:m) block of the standardized pair: QL2ᵀ·(·)·IR2 (block-local)
        @inbounds for j in 1:n1, i in 1:n2
            sA = ZERO; sB = ZERO
            for k in 1:n2
                sA = muladd(QL2[k, i], A[j1 + k - 1, j1 + n2 + j - 1], sA)
                sB = muladd(QL2[k, i], B[j1 + k - 1, j1 + n2 + j - 1], sB)
            end
            tmpA[i, j] = sA; tmpB[i, j] = sB
        end
        @inbounds for j in 1:n1, i in 1:n2
            sA = ZERO; sB = ZERO
            for k in 1:n1
                sA = muladd(tmpA[i, k], IR2[n2 + k, n2 + j], sA)
                sB = muladd(tmpB[i, k], IR2[n2 + k, n2 + j], sB)
            end
            A[j1 + i - 1, j1 + n2 + j - 1] = sA
            B[j1 + i - 1, j1 + n2 + j - 1] = sB
        end
        _tgs_matmul!(M1, false, false, LI, QL2); copyto!(LI, M1)
        _tgs_matmul!(M1, true, false, IR, IR2); copyto!(IR, M1)
        # accumulate into Q, Z — the only part of this routine that scales with n
        bufq, bufz = _tgex2_accum(R, n, m)
        if wantq
            buf = bufq
            @inbounds for j in 1:m, i in 1:n
                s = ZERO
                for k in 1:m
                    s = muladd(Q[i, j1 + k - 1], LI[k, j], s)
                end
                buf[i, j] = s
            end
            @inbounds for j in 1:m, i in 1:n
                Q[i, j1 + j - 1] = buf[i, j]
            end
        end
        if wantz
            buf = bufz
            @inbounds for j in 1:m, i in 1:n
                s = ZERO
                for k in 1:m
                    s = muladd(Z[i, j1 + k - 1], IR[k, j], s)
                end
                buf[i, j] = s
            end
            @inbounds for j in 1:m, i in 1:n
                Z[i, j1 + j - 1] = buf[i, j]
            end
        end
        # update the off-block rows/columns of (A,B)
        if jm < n
            @inbounds for j in (jm + 1):n
                for i in 1:m
                    s = ZERO
                    for k in 1:m
                        s = muladd(LI[k, i], A[j1 + k - 1, j], s)
                    end
                    w[i] = s
                end
                for i in 1:m
                    A[j1 + i - 1, j] = w[i]
                end
                for i in 1:m
                    s = ZERO
                    for k in 1:m
                        s = muladd(LI[k, i], B[j1 + k - 1, j], s)
                    end
                    w[i] = s
                end
                for i in 1:m
                    B[j1 + i - 1, j] = w[i]
                end
            end
        end
        if j1 > 1
            @inbounds for i in 1:(j1 - 1)
                for j in 1:m
                    s = ZERO
                    for k in 1:m
                        s = muladd(A[i, j1 + k - 1], IR[k, j], s)
                    end
                    w[j] = s
                end
                for j in 1:m
                    A[i, j1 + j - 1] = w[j]
                end
                for j in 1:m
                    s = ZERO
                    for k in 1:m
                        s = muladd(B[i, j1 + k - 1], IR[k, j], s)
                    end
                    w[j] = s
                end
                for j in 1:m
                    B[i, j1 + j - 1] = w[j]
                end
            end
        end
        return 0
    end
end

# Dispatcher matching DTGEX2's (N1,N2) signature — full coverage (1↔1, 1↔2, 2↔1, 2↔2).
@inline function _dtgex2!(
        wantq::Bool, wantz::Bool, A::AbstractMatrix{R}, B::AbstractMatrix{R},
        Q::AbstractMatrix{R}, Z::AbstractMatrix{R}, j1::Int, n1::Int, n2::Int
    ) where {R <: Real}
    (n1 == 1 && n2 == 1) && return _dtgex2_1x1!(wantq, wantz, A, B, Q, Z, j1)
    return _dtgex2_big!(wantq, wantz, A, B, Q, Z, j1, n1, n2)
end

# ── DTGEXC (Reference-LAPACK verbatim control flow, mirrors trsen.jl's `_dtrexc!`) ──────────────
# Move the block at IFST to ILST via a walk of adjacent swaps. Returns (info, ilst_final).
function _dtgexc!(
        wantq::Bool, wantz::Bool, A::AbstractMatrix{R}, B::AbstractMatrix{R},
        Q::AbstractMatrix{R}, Z::AbstractMatrix{R}, ifst0::Int, ilst0::Int
    ) where {R <: Real}
    ZERO = zero(R)
    n = size(A, 1)
    n <= 1 && return 0, ilst0
    ifst = ifst0; ilst = ilst0
    ifst > 1 && A[ifst, ifst - 1] != ZERO && (ifst -= 1)
    nbf = 1
    ifst < n && A[ifst + 1, ifst] != ZERO && (nbf = 2)
    ilst > 1 && A[ilst, ilst - 1] != ZERO && (ilst -= 1)
    nbl = 1
    ilst < n && A[ilst + 1, ilst] != ZERO && (nbl = 2)
    ifst == ilst && return 0, ilst
    if ifst < ilst
        nbf == 2 && nbl == 1 && (ilst -= 1)
        nbf == 1 && nbl == 2 && (ilst += 1)
        here = ifst
        while true
            if nbf == 1 || nbf == 2
                nbnext = 1
                here + nbf + 1 <= n && A[here + nbf + 1, here + nbf] != ZERO && (nbnext = 2)
                info = _dtgex2!(wantq, wantz, A, B, Q, Z, here, nbf, nbnext)
                info != 0 && return info, here
                here += nbnext
                nbf == 2 && A[here + 1, here] == ZERO && (nbf = 3)
            else
                nbnext = 1
                here + 3 <= n && A[here + 3, here + 2] != ZERO && (nbnext = 2)
                info = _dtgex2!(wantq, wantz, A, B, Q, Z, here + 1, 1, nbnext)
                info != 0 && return info, here
                if nbnext == 1
                    info = _dtgex2!(wantq, wantz, A, B, Q, Z, here, 1, nbnext)
                    info != 0 && return info, here
                    here += 1
                else
                    A[here + 2, here + 1] == ZERO && (nbnext = 1)
                    if nbnext == 2
                        info = _dtgex2!(wantq, wantz, A, B, Q, Z, here, 1, nbnext)
                        info != 0 && return info, here
                        here += 2
                    else
                        info = _dtgex2!(wantq, wantz, A, B, Q, Z, here, 1, 1)
                        info != 0 && return info, here
                        info = _dtgex2!(wantq, wantz, A, B, Q, Z, here + 1, 1, 1)
                        info != 0 && return info, here
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
                here >= 3 && A[here - 1, here - 2] != ZERO && (nbnext = 2)
                info = _dtgex2!(wantq, wantz, A, B, Q, Z, here - nbnext, nbnext, nbf)
                info != 0 && return info, here
                here -= nbnext
                nbf == 2 && A[here + 1, here] == ZERO && (nbf = 3)
            else
                nbnext = 1
                here >= 3 && A[here - 1, here - 2] != ZERO && (nbnext = 2)
                info = _dtgex2!(wantq, wantz, A, B, Q, Z, here - nbnext, nbnext, 1)
                info != 0 && return info, here
                if nbnext == 1
                    info = _dtgex2!(wantq, wantz, A, B, Q, Z, here, nbnext, 1)
                    info != 0 && return info, here
                    here -= 1
                else
                    A[here, here - 1] == ZERO && (nbnext = 1)
                    if nbnext == 2
                        info = _dtgex2!(wantq, wantz, A, B, Q, Z, here - 1, 2, 1)
                        info != 0 && return info, here
                        here -= 2
                    else
                        info = _dtgex2!(wantq, wantz, A, B, Q, Z, here, 1, 1)
                        info != 0 && return info, here
                        info = _dtgex2!(wantq, wantz, A, B, Q, Z, here - 1, 1, 1)
                        info != 0 && return info, here
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
function _tgs_dlag2(
        a11r::R, a21r::R, a12r::R, a22r::R, b11r::R, b12r::R, b22r::R,
        safmin::R
    ) where {R <: Real}
    ZERO = zero(R); ONE = one(R); TWO = R(2); HALF = ONE / TWO; FUZZY1 = ONE + R(1.0e-5)
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
# `sel` is the CALLER's select vector, tested elementwise here — the entry no longer materialises a
# `Bool[...]` comprehension. `alpha` (complex) and `beta` (real) are caller-provided outputs.
function _dtgsen!(
        sel::AbstractVector, A::AbstractMatrix{R}, B::AbstractMatrix{R},
        Q::AbstractMatrix{R}, Z::AbstractMatrix{R}, alpha::AbstractVector{<:Complex},
        beta::AbstractVector{R}
    ) where {R <: Real}
    n = size(A, 1)
    ZERO = zero(R)
    # Collect the selected blocks at the top-left corner of (A,B).
    ks = 0; pair = false
    for k in 1:n
        if pair
            pair = false
        else
            swap = sel[k] != 0
            if k < n && A[k + 1, k] != ZERO
                pair = true
                swap = swap || sel[k + 1] != 0
            end
            if swap
                ks += 1
                kk = k
                if kk != ks
                    info, _ = _dtgexc!(true, true, A, B, Q, Z, kk, ks)
                    info != 0 && throw(
                        ErrorException(
                            "tgsen!: swap rejected (reordering too ill-conditioned; LAPACK info=1)"
                        )
                    )
                end
                pair && (ks += 1)
            end
        end
    end
    # Compute generalized eigenvalues of the reordered pair and normalize (sign of B's diagonal).
    # No fill! on the scratch: the loop below assigns alphar[k], alphai[k] and beta[k] at every k on
    # both the 2×2 and the 1×1 branch (the old `zeros` was never load-bearing).
    alphar, alphai = _tgsen_alpha_work(R, n)
    k = 1
    @inbounds while k <= n
        if k < n && A[k + 1, k] != ZERO
            a11 = A[k, k]; a21 = A[k + 1, k]; a12 = A[k, k + 1]; a22 = A[k + 1, k + 1]
            b11 = B[k, k]; b12 = B[k, k + 1]; b22 = B[k + 1, k + 1]
            s1, s2, wr1, wr2, wi = _tgs_dlag2(a11, a21, a12, a22, b11, b12, b22, _syl_safmin(R))
            beta[k] = s1; beta[k + 1] = s2
            alphar[k] = wr1; alphar[k + 1] = wr2
            alphai[k] = wi; alphai[k + 1] = -wi
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
    @inbounds for i in 1:n
        alpha[i] = Complex(alphar[i], alphai[i])
    end
    return A, B, alpha, beta, Q, Z
end

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# ZTGEX2 (complex, always 1×1/1×1 — S,T strictly triangular, no 2×2 blocks) + ZTGEXC + ZTGSEN
# ═══════════════════════════════════════════════════════════════════════════════════════════════
function _ztgex2!(
        wantq::Bool, wantz::Bool, A::AbstractMatrix{C}, B::AbstractMatrix{C},
        Q::AbstractMatrix{C}, Z::AbstractMatrix{C}, j1::Int
    ) where {C <: Complex}
    R = real(C)
    n = size(A, 1)
    Ao11 = A[j1, j1]; Ao12 = A[j1, j1 + 1]; Ao21 = A[j1 + 1, j1]; Ao22 = A[j1 + 1, j1 + 1]
    Bo11 = B[j1, j1]; Bo12 = B[j1, j1 + 1]; Bo22 = B[j1 + 1, j1 + 1]
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
    A[j1 + 1, j1] = zero(C); B[j1 + 1, j1] = zero(C)
    wantz && _zrot_cols!(Z, j1, j1 + 1, 1, n, cz, conj(sz))
    wantq && _zrot_cols!(Q, j1, j1 + 1, 1, n, cq, conj(sq))
    return 0
end

# ── ZTGEXC — sequential adjacent-swap sweep (no block-size bookkeeping needed: always 1×1) ───────
function _ztgexc!(
        wantq::Bool, wantz::Bool, A::AbstractMatrix{C}, B::AbstractMatrix{C},
        Q::AbstractMatrix{C}, Z::AbstractMatrix{C}, ifst::Int, ilst::Int
    ) where {C <: Complex}
    n = size(A, 1)
    (n <= 1 || ifst == ilst) && return 0
    m1, m2, m3 = ifst < ilst ? (0, -1, 1) : (-1, 0, -1)
    for k in (ifst + m1):m3:(ilst + m2)
        info = _ztgex2!(wantq, wantz, A, B, Q, Z, k)
        info != 0 && return info
    end
    return 0
end

function _ztgsen!(
        sel::AbstractVector, A::AbstractMatrix{C}, B::AbstractMatrix{C},
        Q::AbstractMatrix{C}, Z::AbstractMatrix{C}, alpha::AbstractVector{C},
        beta::AbstractVector{C}
    ) where {C <: Complex}
    R = real(C)
    n = size(A, 1)
    ks = 0
    for k in 1:n
        if sel[k] != 0
            ks += 1
            if k != ks
                info = _ztgexc!(true, true, A, B, Q, Z, k, ks)
                info != 0 && throw(ErrorException("tgsen!: swap rejected (ill-conditioned reorder)"))
            end
        end
    end
    safmin = _syl_safmin(R)
    @inbounds for k in 1:n
        dscale = abs(B[k, k])
        if dscale > safmin
            temp2 = B[k, k] / dscale
            temp1 = conj(temp2)
            B[k, k] = C(dscale)
            for j in (k + 1):n
                B[k, j] *= temp1
            end
            for j in k:n
                A[k, j] *= temp1
            end
            for i in 1:n
                Q[i, k] *= temp2
            end
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
    tgsen!(select, S, T, Q, Z, alpha, beta) -> (S, T, alpha, beta, Q, Z)   [in-place]

Reorder the generalized Schur pair `(S,T)` (`S` (quasi-)upper-triangular, `T` upper-triangular,
`Q·S·Zᴴ`/`Q·T·Zᴴ` the original matrix pair) so that the generalized eigenvalues selected by
`select` occupy the leading block, updating `Q`, `Z` in place. `alpha`/`beta` (length `n`, `alpha`
complex, `beta` real for real `S`/`T` — complex for complex `S`/`T`) are the reordered generalized
eigenvalues `alpha[i]/beta[i]`. Mirrors LAPACK `dtgsen`/`ztgsen` at `ijob=0` (reorder only; no
condition-number estimates), matching `LinearAlgebra.LAPACK.tgsen!`'s contract exactly.

Both paths are complete. The REAL path handles every adjacent-block swap combination (1×1↔1×1,
1×1↔2×2, 2×2↔1×1, 2×2↔2×2 — complex-conjugate eigenvalue pairs reorder like LAPACK's dtgex2);
a swap LAPACK would reject as too ill-conditioned (info=1) throws, matching the LAPACKException
Julia's wrapper raises. The COMPLEX path has no 2×2 blocks (complex Schur form is triangular).

The trailing-buffer form writes the eigenvalues into the caller's `alpha`/`beta` and allocates
nothing. `select` is read elementwise (`select[i] != 0`), so any integer/Bool vector works.
"""
@inline function _tgsen_dims(select, S, T, Q, Z)
    n = size(S, 1)
    (
        n == size(S, 2) == size(T, 1) == size(T, 2) == size(Q, 1) == size(Q, 2) ==
            size(Z, 1) == size(Z, 2)
    ) ||
        throw(DimensionMismatch("tgsen!: S, T, Q, Z must be square of matching size"))
    length(select) == n || throw(DimensionMismatch("tgsen!: select must have length n"))
    return n
end

function tgsen!(
        select::AbstractVector, S::AbstractMatrix{R}, T::AbstractMatrix{R},
        Q::AbstractMatrix{R}, Z::AbstractMatrix{R}
    ) where {R <: Real}
    n = _tgsen_dims(select, S, T, Q, Z)
    alpha = Vector{Complex{R}}(undef, n); beta = zeros(R, n)
    return _dtgsen!(select, S, T, Q, Z, alpha, beta)
end

function tgsen!(
        select::AbstractVector, S::AbstractMatrix{C}, T::AbstractMatrix{C},
        Q::AbstractMatrix{C}, Z::AbstractMatrix{C}
    ) where {C <: Complex}
    n = _tgsen_dims(select, S, T, Q, Z)
    alpha = Vector{C}(undef, n); beta = Vector{C}(undef, n)
    return _ztgsen!(select, S, T, Q, Z, alpha, beta)
end

# Lesson 9 — alpha and beta are pure outputs, written only in the final normalisation loop after every
# swap, so an unrelated buffer is safe; `alpha === beta` is not (the complex path writes both at the
# same index in one pass, so the second store would clobber the first).
function tgsen!(
        select::AbstractVector, S::AbstractMatrix{R}, T::AbstractMatrix{R},
        Q::AbstractMatrix{R}, Z::AbstractMatrix{R}, alpha::AbstractVector{<:Complex},
        beta::AbstractVector{R}
    ) where {R <: Real}
    n = _tgsen_dims(select, S, T, Q, Z)
    (length(alpha) == n && length(beta) == n) ||
        throw(DimensionMismatch("tgsen!: alpha and beta must have length n"))
    return _dtgsen!(select, S, T, Q, Z, alpha, beta)
end

function tgsen!(
        select::AbstractVector, S::AbstractMatrix{C}, T::AbstractMatrix{C},
        Q::AbstractMatrix{C}, Z::AbstractMatrix{C}, alpha::AbstractVector{C},
        beta::AbstractVector{C}
    ) where {C <: Complex}
    n = _tgsen_dims(select, S, T, Q, Z)
    (length(alpha) == n && length(beta) == n) ||
        throw(DimensionMismatch("tgsen!: alpha and beta must have length n"))
    Base.mightalias(alpha, beta) && throw(ArgumentError("tgsen!: alpha and beta must be distinct"))
    return _ztgsen!(select, S, T, Q, Z, alpha, beta)
end
