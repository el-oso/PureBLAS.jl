# LAPACK Schur reordering (backs `ordschur`):
#   trexc — move one diagonal block of a (quasi-)upper-triangular Schur form to a target position,
#           accumulating the orthogonal/unitary swaps into Q  (dtrexc / ztrexc).
#   trsen — reorder a *selected* set of eigenvalues to the leading block, and (job≠'N') estimate the
#           condition numbers S (eigenvalue-cluster) / SEP (invariant-subspace)  (dtrsen / ztrsen).
# CORRECTNESS-FIRST port of Reference-LAPACK. The real path swaps adjacent 1×1/2×2 blocks with the
# dlaexc reflector-based swap (dlanv2 restandardization — the conj-pair 2×2 swap is the top bug locus);
# the complex path is a Givens sweep. SEP uses the Hager 1-norm estimator (dlacn2) over the Sylvester
# operator, which is why this file DEPENDS ON trsyl.jl (`_dtrsyl!`/`_ztrsyl!`, `_syl_dlasy2`,
# `_syl_safmin`) — `include("trsyl.jl")` first.  STANDALONE: not wired into the module includes / C-ABI.

using LinearAlgebra: givensAlgorithm

# ── DLANV2 (Reference-LAPACK verbatim) — standardize real 2×2 [a b; c d] to Schur form ────────────────
function _exc_dlanv2(a::R, b::R, c::R, d::R) where {R<:Real}
    Z0 = zero(R); ONE = one(R); HALF = R(0.5); TWO = R(2); MULTPL = R(4)
    eps_p = eps(R)
    safmin = _syl_safmin(R)
    safmn2 = TWO ^ trunc(Int, log(safmin / eps_p) / log(TWO) / 2)
    safmx2 = ONE / safmn2
    cs = ONE; sn = Z0
    if c == Z0
    elseif b == Z0
        cs = Z0; sn = ONE
        temp = d; d = a; a = temp; b = -c; c = Z0
    elseif (a - d) == Z0 && copysign(ONE, b) != copysign(ONE, c)
    else
        temp = a - d
        p = HALF * temp
        bcmax = max(abs(b), abs(c))
        bcmis = min(abs(b), abs(c)) * copysign(ONE, b) * copysign(ONE, c)
        scale = max(abs(p), bcmax)
        z = (p / scale) * p + (bcmax / scale) * bcmis
        if z >= MULTPL * eps_p
            z = p + copysign(sqrt(scale) * sqrt(z), p)
            a = d + z
            d = d - (bcmax / z) * bcmis
            tau = hypot(c, z)
            cs = z / tau; sn = c / tau
            b = b - c; c = Z0
        else
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
            aa = a * cs + b * sn; bb = -a * sn + b * cs
            cc = c * cs + d * sn; dd = -c * sn + d * cs
            a = aa * cs + cc * sn; b = bb * cs + dd * sn
            c = -(aa * sn) + cc * cs; d = -bb * sn + dd * cs
            temp = HALF * (a + d); a = temp; d = temp
            if c != Z0
                if b != Z0
                    if copysign(ONE, b) == copysign(ONE, c)
                        sab = sqrt(abs(b)); sac = sqrt(abs(c))
                        p = copysign(sab * sac, c)
                        tau = ONE / sqrt(abs(b + c))
                        a = temp + p; d = temp - p
                        b = b - c; c = Z0
                        cs1 = sab * tau; sn1 = sac * tau
                        temp2 = cs * cs1 - sn * sn1
                        sn = cs * sn1 + sn * cs1; cs = temp2
                    end
                else
                    b = -c; c = Z0
                    temp2 = cs; cs = -sn; sn = temp2
                end
            end
        end
    end
    rt1r = a; rt2r = d
    if c == Z0
        rt1i = Z0; rt2i = Z0
    else
        rt1i = sqrt(abs(b)) * sqrt(abs(c)); rt2i = -rt1i
    end
    return a, b, c, d, rt1r, rt1i, rt2r, rt2i, cs, sn
end

# ── DLARFG on a short real vector: alpha (scalar) over x (essential tail, mutated to v) ────────────────
function _exc_larfg!(alpha::R, x::AbstractVector{R}) where {R<:Real}
    xnorm = zero(R); @inbounds for xi in x; xnorm += xi * xi; end
    xnorm = sqrt(xnorm)
    xnorm == zero(R) && return alpha, zero(R)          # tau = 0
    beta = -copysign(hypot(alpha, xnorm), alpha)
    safmn = _syl_safmin(R) / (eps(R) / 2)
    knt = 0
    if abs(beta) < safmn
        rsafmn = one(R) / safmn
        while true
            knt += 1
            @inbounds for i in eachindex(x); x[i] *= rsafmn; end
            beta *= rsafmn; alpha *= rsafmn
            (abs(beta) < safmn && knt < 20) || break
        end
        xnorm = zero(R); @inbounds for xi in x; xnorm += xi * xi; end
        xnorm = sqrt(xnorm)
        beta = -copysign(hypot(alpha, xnorm), alpha)
    end
    tau = (beta - alpha) / beta
    s = one(R) / (alpha - beta)
    @inbounds for i in eachindex(x); x[i] *= s; end
    for _ in 1:knt; beta *= safmn; end
    return beta, tau
end

# ── DLARFX apply (order ≤ 3 reflector v with explicit unit, generic dense) ─────────────────────────────
@inline function _exc_larfx_l!(v::AbstractVector{R}, tau::R, C::AbstractMatrix{R}) where {R}
    tau == zero(R) && return
    m = size(C, 1); n = size(C, 2)
    @inbounds for j in 1:n
        s = zero(R); for i in 1:m; s += v[i] * C[i, j]; end
        s *= tau
        for i in 1:m; C[i, j] -= v[i] * s; end
    end
end
@inline function _exc_larfx_r!(v::AbstractVector{R}, tau::R, C::AbstractMatrix{R}) where {R}
    tau == zero(R) && return
    m = size(C, 1); n = size(C, 2)
    @inbounds for i in 1:m
        s = zero(R); for j in 1:n; s += C[i, j] * v[j]; end
        s *= tau
        for j in 1:n; C[i, j] -= s * v[j]; end
    end
end

# ── DROT on strided lanes (x' = c·x + s·y, y' = c·y − s·x) ─────────────────────────────────────────────
@inline function _exc_rot_rows!(M::AbstractMatrix, r1::Int, r2::Int, lo::Int, hi::Int, cs, sn)
    @inbounds for j in lo:hi
        t = M[r1, j]; u = M[r2, j]
        M[r1, j] = cs * t + sn * u
        M[r2, j] = cs * u - conj(sn) * t
    end
end
@inline function _exc_rot_cols!(M::AbstractMatrix, c1::Int, c2::Int, lo::Int, hi::Int, cs, sn)
    @inbounds for i in lo:hi
        t = M[i, c1]; u = M[i, c2]
        M[i, c1] = cs * t + sn * u
        M[i, c2] = cs * u - conj(sn) * t
    end
end

# ── DLAEXC (Reference-LAPACK verbatim) — swap adjacent diagonal blocks (n1,n2 ∈ {1,2}) at J1 ───────────
# Returns info (0 ok; 1 = swap would deflate to ill-conditioned / rejected → caller aborts the reorder).
function _dlaexc!(wantq::Bool, T::AbstractMatrix{R}, Q::AbstractMatrix{R},
        j1::Int, n1::Int, n2::Int) where {R<:Real}
    ZERO = zero(R); ONE = one(R); TEN = R(10)
    n = size(T, 1)
    (n == 0 || n1 == 0 || n2 == 0) && return 0
    j1 + n1 > n && return 0
    j2 = j1 + 1; j3 = j1 + 2; j4 = j1 + 3
    if n1 == 1 && n2 == 1
        t11 = T[j1, j1]; t22 = T[j2, j2]
        cs, sn, _ = givensAlgorithm(T[j1, j2], t22 - t11)
        j3 <= n && _exc_rot_rows!(T, j1, j2, j3, n, cs, sn)
        _exc_rot_cols!(T, j1, j2, 1, j1 - 1, cs, sn)
        T[j1, j1] = t22; T[j2, j2] = t11
        wantq && _exc_rot_cols!(Q, j1, j2, 1, n, cs, sn)
        return 0
    end
    nd = n1 + n2
    D = zeros(R, nd, nd)
    @inbounds for jj in 1:nd, ii in 1:nd; D[ii, jj] = T[j1 + ii - 1, j1 + jj - 1]; end
    dnorm = ZERO; @inbounds for x in D; dnorm = max(dnorm, abs(x)); end
    eps_p = eps(R); smlnum = _syl_safmin(R) / eps_p
    thresh = max(TEN * eps_p * dnorm, smlnum)
    TL = view(D, 1:n1, 1:n1); TR = view(D, n1+1:nd, n1+1:nd); BB = view(D, 1:n1, n1+1:nd)
    x11, x21, x12, x22, scale, _, _ = _syl_dlasy2(false, false, -1, n1, n2, TL, TR, BB)
    k = n1 + n1 + n2 - 3
    if k == 1
        # n1=1, n2=2
        u = R[scale, x11, x12]
        _, tau = _exc_larfg!(u[3], view(u, 1:2)); u[3] = ONE
        t11 = T[j1, j1]
        _exc_larfx_l!(u, tau, view(D, 1:3, 1:3)); _exc_larfx_r!(u, tau, view(D, 1:3, 1:3))
        max(abs(D[3, 1]), abs(D[3, 2]), abs(D[3, 3] - t11)) > thresh && return 1
        _exc_larfx_l!(u, tau, view(T, j1:j1+2, j1:n))
        _exc_larfx_r!(u, tau, view(T, 1:j2, j1:j1+2))
        T[j3, j1] = ZERO; T[j3, j2] = ZERO; T[j3, j3] = t11
        wantq && _exc_larfx_r!(u, tau, view(Q, 1:n, j1:j1+2))
    elseif k == 2
        # n1=2, n2=1
        u = R[-x11, -x21, scale]
        _, tau = _exc_larfg!(u[1], view(u, 2:3)); u[1] = ONE
        t33 = T[j3, j3]
        _exc_larfx_l!(u, tau, view(D, 1:3, 1:3)); _exc_larfx_r!(u, tau, view(D, 1:3, 1:3))
        max(abs(D[2, 1]), abs(D[3, 1]), abs(D[1, 1] - t33)) > thresh && return 1
        _exc_larfx_r!(u, tau, view(T, 1:j3, j1:j1+2))
        _exc_larfx_l!(u, tau, view(T, j1:j1+2, j2:n))
        T[j1, j1] = t33; T[j2, j1] = ZERO; T[j3, j1] = ZERO
        wantq && _exc_larfx_r!(u, tau, view(Q, 1:n, j1:j1+2))
    else
        # n1=2, n2=2
        u1 = R[-x11, -x21, scale]
        _, tau1 = _exc_larfg!(u1[1], view(u1, 2:3)); u1[1] = ONE
        temp = -tau1 * (x12 + u1[2] * x22)
        u2 = R[-temp * u1[2] - x22, -temp * u1[3], scale]
        _, tau2 = _exc_larfg!(u2[1], view(u2, 2:3)); u2[1] = ONE
        _exc_larfx_l!(u1, tau1, view(D, 1:3, 1:4)); _exc_larfx_r!(u1, tau1, view(D, 1:4, 1:3))
        _exc_larfx_l!(u2, tau2, view(D, 2:4, 1:4)); _exc_larfx_r!(u2, tau2, view(D, 1:4, 2:4))
        max(abs(D[3, 1]), abs(D[3, 2]), abs(D[4, 1]), abs(D[4, 2])) > thresh && return 1
        _exc_larfx_l!(u1, tau1, view(T, j1:j1+2, j1:n))
        _exc_larfx_r!(u1, tau1, view(T, 1:j4, j1:j1+2))
        _exc_larfx_l!(u2, tau2, view(T, j2:j2+2, j1:n))
        _exc_larfx_r!(u2, tau2, view(T, 1:j4, j2:j2+2))
        T[j3, j1] = ZERO; T[j3, j2] = ZERO; T[j4, j1] = ZERO; T[j4, j2] = ZERO
        if wantq
            _exc_larfx_r!(u1, tau1, view(Q, 1:n, j1:j1+2))
            _exc_larfx_r!(u2, tau2, view(Q, 1:n, j2:j2+2))
        end
    end
    # ---- restandardize the swapped blocks (dlanv2 + rotations) ----
    if n2 == 2
        a, b, c, d, _, _, _, _, cs, sn = _exc_dlanv2(T[j1, j1], T[j1, j2], T[j2, j1], T[j2, j2])
        T[j1, j1] = a; T[j1, j2] = b; T[j2, j1] = c; T[j2, j2] = d
        _exc_rot_rows!(T, j1, j2, j1 + 2, n, cs, sn)
        _exc_rot_cols!(T, j1, j2, 1, j1 - 1, cs, sn)
        wantq && _exc_rot_cols!(Q, j1, j2, 1, n, cs, sn)
    end
    if n1 == 2
        j3b = j1 + n2; j4b = j3b + 1
        a, b, c, d, _, _, _, _, cs, sn = _exc_dlanv2(T[j3b, j3b], T[j3b, j4b], T[j4b, j3b], T[j4b, j4b])
        T[j3b, j3b] = a; T[j3b, j4b] = b; T[j4b, j3b] = c; T[j4b, j4b] = d
        j3b + 2 <= n && _exc_rot_rows!(T, j3b, j4b, j3b + 2, n, cs, sn)
        _exc_rot_cols!(T, j3b, j4b, 1, j3b - 1, cs, sn)
        wantq && _exc_rot_cols!(Q, j3b, j4b, 1, n, cs, sn)
    end
    return 0
end

# ── DTREXC (Reference-LAPACK verbatim), REAL quasi-triangular T ────────────────────────────────────────
# Move the block at IFST to ILST. Returns (info, ilst_final).
function _dtrexc!(wantq::Bool, T::AbstractMatrix{R}, Q::AbstractMatrix{R},
        ifst0::Int, ilst0::Int) where {R<:Real}
    ZERO = zero(R)
    n = size(T, 1)
    n <= 1 && return 0, ilst0
    ifst = ifst0; ilst = ilst0
    ifst > 1 && T[ifst, ifst-1] != ZERO && (ifst -= 1)
    nbf = 1
    ifst < n && T[ifst+1, ifst] != ZERO && (nbf = 2)
    ilst > 1 && T[ilst, ilst-1] != ZERO && (ilst -= 1)
    nbl = 1
    ilst < n && T[ilst+1, ilst] != ZERO && (nbl = 2)
    ifst == ilst && return 0, ilst
    if ifst < ilst
        nbf == 2 && nbl == 1 && (ilst -= 1)
        nbf == 1 && nbl == 2 && (ilst += 1)
        here = ifst
        while true
            if nbf == 1 || nbf == 2
                nbnext = 1
                here + nbf + 1 <= n && T[here+nbf+1, here+nbf] != ZERO && (nbnext = 2)
                info = _dlaexc!(wantq, T, Q, here, nbf, nbnext)
                info != 0 && return info, here
                here += nbnext
                nbf == 2 && T[here+1, here] == ZERO && (nbf = 3)
            else
                nbnext = 1
                here + 3 <= n && T[here+3, here+2] != ZERO && (nbnext = 2)
                info = _dlaexc!(wantq, T, Q, here + 1, 1, nbnext)
                info != 0 && return info, here
                if nbnext == 1
                    _dlaexc!(wantq, T, Q, here, 1, nbnext)
                    here += 1
                else
                    T[here+2, here+1] == ZERO && (nbnext = 1)
                    if nbnext == 2
                        info = _dlaexc!(wantq, T, Q, here, 1, nbnext)
                        info != 0 && return info, here
                        here += 2
                    else
                        _dlaexc!(wantq, T, Q, here, 1, 1)
                        _dlaexc!(wantq, T, Q, here + 1, 1, 1)
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
                here >= 3 && T[here-1, here-2] != ZERO && (nbnext = 2)
                info = _dlaexc!(wantq, T, Q, here - nbnext, nbnext, nbf)
                info != 0 && return info, here
                here -= nbnext
                nbf == 2 && T[here+1, here] == ZERO && (nbf = 3)
            else
                nbnext = 1
                here >= 3 && T[here-1, here-2] != ZERO && (nbnext = 2)
                info = _dlaexc!(wantq, T, Q, here - nbnext, nbnext, 1)
                info != 0 && return info, here
                if nbnext == 1
                    _dlaexc!(wantq, T, Q, here, nbnext, 1)
                    here -= 1
                else
                    T[here, here-1] == ZERO && (nbnext = 1)
                    if nbnext == 2
                        info = _dlaexc!(wantq, T, Q, here - 1, 2, 1)
                        info != 0 && return info, here
                        here -= 2
                    else
                        _dlaexc!(wantq, T, Q, here, 1, 1)
                        _dlaexc!(wantq, T, Q, here - 1, 1, 1)
                        here -= 2
                    end
                end
            end
            here > ilst || break
        end
    end
    return 0, here
end

# ── ZTREXC (Reference-LAPACK verbatim), COMPLEX triangular T — Givens sweep ────────────────────────────
function _ztrexc!(wantq::Bool, T::AbstractMatrix{C}, Q::AbstractMatrix{C},
        ifst::Int, ilst::Int) where {C<:Complex}
    n = size(T, 1)
    (n <= 1 || ifst == ilst) && return 0, ilst
    m1, m2, m3 = ifst < ilst ? (0, -1, 1) : (-1, 0, -1)
    for k in (ifst + m1):m3:(ilst + m2)
        t11 = T[k, k]; t22 = T[k+1, k+1]
        cs, sn, _ = givensAlgorithm(T[k, k+1], t22 - t11)
        # row rotation uses SN; column rotations use conj(SN) (ZROT convention, ztrexc.f)
        k + 2 <= n && _exc_rot_rows!(T, k, k + 1, k + 2, n, cs, sn)
        _exc_rot_cols!(T, k, k + 1, 1, k - 1, cs, conj(sn))
        T[k, k] = t22; T[k+1, k+1] = t11
        wantq && _exc_rot_cols!(Q, k, k + 1, 1, n, cs, conj(sn))
    end
    return 0, ilst
end

"""
    trexc!(compq, T, Q, ifst, ilst) -> (T, Q)

Move the diagonal block of the (quasi-)upper-triangular Schur form `T` at position `ifst` to position
`ilst` by orthogonal (real) / unitary (complex) similarity swaps, accumulating them into `Q` when
`compq='V'` (`compq='N'` leaves `Q` untouched). LAPACK `dtrexc`/`ztrexc`. For real `T`, 1×1 and 2×2
(conjugate-pair) blocks are swapped by `dlaexc`; `ifst`/`ilst` snap to block boundaries as in LAPACK.
"""
function trexc!(compq::AbstractChar, T::AbstractMatrix, Q::AbstractMatrix, ifst::Integer, ilst::Integer)
    (compq === 'V' || compq === 'N') || throw(ArgumentError("trexc!: compq must be 'V' or 'N'"))
    n = size(T, 1)
    size(T, 2) == n || throw(DimensionMismatch("trexc!: T must be square"))
    wantq = compq === 'V'
    (1 <= ifst <= n && 1 <= ilst <= n) || throw(ArgumentError("trexc!: ifst, ilst must be in 1:n"))
    _trexc_dispatch!(wantq, T, Q, Int(ifst), Int(ilst))
    return T, Q
end
_trexc_dispatch!(wantq, T::AbstractMatrix{<:Real}, Q, ifst, ilst) = _dtrexc!(wantq, T, Q, ifst, ilst)
_trexc_dispatch!(wantq, T::AbstractMatrix{<:Complex}, Q, ifst, ilst) = _ztrexc!(wantq, T, Q, ifst, ilst)

# ── DLACN2 / ZLACN2 (Reference-LAPACK) — Hager–Higham 1-norm estimator ─────────────────────────────────
# `apply!(x, kase)` overwrites x with A·x (kase=1) or Aᵀ/Aᴴ·x (kase=2). Returns the estimate of ‖A‖₁.
# Faithful port with the reverse-communication state machine flattened to nested loops. The ONLY real vs
# complex difference: DLACN2 breaks when no sign changed OR est shrinks; ZLACN2 breaks only when est shrinks.
function _lacn2_estimate(n::Int, apply!::F, ::Type{V}) where {F, V}
    R = real(V)
    ITMAX = 5
    x = fill(V(one(R) / R(n)), n); v = zeros(V, n); isgn = zeros(Int, n)
    onenorm(w) = (s = zero(R); @inbounds for wi in w; s += abs(wi); end; s)
    apply!(x, 1)                                           # X ← A·(1/n)
    n == 1 && return abs(x[1])
    est = onenorm(x)
    _lacn2_sign!(x, isgn, V)
    apply!(x, 2)                                           # X ← Aᵀ·ξ
    jmax = _lacn2_imax(x); iter = 2
    while true
        fill!(x, zero(V)); x[jmax] = one(V)
        apply!(x, 1)                                       # X ← A·e_jmax
        copyto!(v, x); estold = est; est = onenorm(x)
        conv = V <: Complex ? (est <= estold) : (!_lacn2_signchanged(x, isgn) || est <= estold)
        conv && break
        _lacn2_sign!(x, isgn, V)
        apply!(x, 2)                                       # X ← Aᵀ·ξ
        jlast = jmax; jmax = _lacn2_imax(x)
        cyc = V <: Complex ? (abs(x[jlast]) != abs(x[jmax])) : (x[jlast] != abs(x[jmax]))
        (cyc && iter < ITMAX) || break
        iter += 1
    end
    # alternating-sign probe vector for one more estimate
    asgn = one(R)
    @inbounds for i in 1:n
        x[i] = V(asgn * (one(R) + R(i - 1) / R(n - 1)))
        asgn = -asgn
    end
    apply!(x, 1)
    temp = R(2) * (onenorm(x) / R(3 * n))
    temp > est && (est = temp)
    return est
end
@inline function _lacn2_sign!(x, isgn, ::Type{V}) where {V}
    R = real(V)
    if V <: Complex
        safmin = _syl_safmin(R)
        @inbounds for i in eachindex(x)
            a = abs(x[i])
            x[i] = a > safmin ? x[i] / a : one(V)
        end
    else
        @inbounds for i in eachindex(x)
            s = x[i] >= zero(R) ? one(R) : -one(R)
            x[i] = s; isgn[i] = round(Int, s)
        end
    end
end
@inline function _lacn2_signchanged(x, isgn)
    @inbounds for i in eachindex(x)
        s = x[i] >= 0 ? 1 : -1
        s != isgn[i] && return true
    end
    return false
end
@inline function _lacn2_imax(x)
    ii = 1; best = abs(x[1])
    @inbounds for i in 2:length(x)
        a = abs(x[i]); a > best && (best = a; ii = i)
    end
    return ii
end

# ── DTRSEN / ZTRSEN reorder driver + condition numbers ────────────────────────────────────────────────
# Returns (T, Q, w, s, sep, info).  job: 'N' reorder only, 'E' + S, 'V' + SEP, 'B' both.
function _dtrsen!(job::AbstractChar, wantq::Bool, select::AbstractVector{Bool},
        T::AbstractMatrix{R}, Q::AbstractMatrix{R}) where {R<:Real}
    ZERO = zero(R); ONE = one(R)
    n = size(T, 1)
    wants = job === 'E' || job === 'B'
    wantsp = job === 'V' || job === 'B'
    sel = collect(Bool, select)
    # count selected (respecting 2×2 conj pairs)
    m = 0; pair = false
    for k in 1:n
        if pair; pair = false; continue; end
        if k < n && T[k+1, k] != ZERO
            pair = true
            (sel[k] || sel[k+1]) && (m += 2)
        else
            sel[k] && (m += 1)
        end
    end
    n1 = m; n2 = n - m
    s = ONE; sep = ZERO; info = 0
    if !(m == n || m == 0)
        # reorder selected eigenvalues to the leading positions via dtrexc swaps
        ks = 0; pair = false
        for k in 1:n
            if pair; pair = false; continue; end
            swap = sel[k]
            if k < n && T[k+1, k] != ZERO
                pair = true
                swap = swap || sel[k+1]
            end
            if swap
                ks += 1
                if k != ks
                    ierr, _ = _dtrexc!(wantq, T, Q, k, ks)
                    (ierr == 1 || ierr == 2) && (info = 1)
                end
                pair && (ks += 1)
            end
        end
    end
    if info == 0 && wants
        if m == n || m == 0
            s = ONE
        else
            # S = scale / ( sqrt(scale²/rnorm + rnorm)·sqrt(rnorm) ),  rnorm = ‖X‖_F of the coupling solve
            Rm = copy(T[1:n1, n1+1:n])                 # off-diagonal coupling block T₁₂
            _, scale, _ = _dtrsyl!('N', 'N', -1, view(T, 1:n1, 1:n1), view(T, n1+1:n, n1+1:n), Rm)
            rnorm = sqrt(sum(abs2, Rm))
            s = rnorm == ZERO ? ONE : scale / (sqrt(scale^2 / rnorm + rnorm) * sqrt(rnorm))
        end
    end
    if info == 0 && wantsp
        if m == n || m == 0
            sep = _one_norm(T)
        else
            nn = n1 * n2
            T11 = T[1:n1, 1:n1]; T22 = T[n1+1:n, n1+1:n]
            scref = Ref(ONE)
            apply! = function (xv, kase)
                Xm = reshape(xv, n1, n2)
                _, sc, _ = kase == 1 ? _dtrsyl!('N', 'N', -1, T11, T22, Xm) :
                                       _dtrsyl!('T', 'T', -1, T11, T22, Xm)
                scref[] = sc
                return nothing
            end
            # estimate ‖L⁻¹‖₁ of the Sylvester operator; SEP = scale/est (cancels trsyl's scaling)
            est = _lacn2_estimate(nn, apply!, R)
            sep = est == ZERO ? ZERO : scref[] / est
        end
    end
    w = _diag_eigs(T)
    return T, Q, w, s, sep, info
end

function _ztrsen!(job::AbstractChar, wantq::Bool, select::AbstractVector{Bool},
        T::AbstractMatrix{C}, Q::AbstractMatrix{C}) where {C<:Complex}
    R = real(C)
    n = size(T, 1)
    wants = job === 'E' || job === 'B'
    wantsp = job === 'V' || job === 'B'
    sel = collect(Bool, select)
    m = count(sel)
    n1 = m; n2 = n - m
    s = one(R); sep = zero(R); info = 0
    if !(m == n || m == 0)
        ks = 0
        for k in 1:n
            if sel[k]
                ks += 1
                k != ks && _ztrexc!(wantq, T, Q, k, ks)
            end
        end
    end
    if wants
        if m == n || m == 0
            s = one(R)
        else
            Rm = C.(T[1:n1, n1+1:n])
            _, scale, _ = _ztrsyl!('N', 'N', -1, view(T, 1:n1, 1:n1), view(T, n1+1:n, n1+1:n), Rm)
            rnorm = sqrt(sum(abs2, Rm))
            s = rnorm == zero(R) ? one(R) : scale / (sqrt(scale^2 / rnorm + rnorm) * sqrt(rnorm))
        end
    end
    if wantsp
        if m == n || m == 0
            sep = _one_norm(T)
        else
            nn = n1 * n2
            T11 = T[1:n1, 1:n1]; T22 = T[n1+1:n, n1+1:n]
            scref = Ref(one(R))
            apply! = function (xv, kase)
                Xm = reshape(xv, n1, n2)
                _, sc, _ = kase == 1 ? _ztrsyl!('N', 'N', -1, T11, T22, Xm) :
                                       _ztrsyl!('C', 'C', -1, T11, T22, Xm)
                scref[] = sc
                return nothing
            end
            est = _lacn2_estimate(nn, apply!, C)
            sep = est == zero(R) ? zero(R) : scref[] / est
        end
    end
    w = _diag_eigs(T)
    return T, Q, w, s, sep, info
end

# eigenvalues from the (quasi-)triangular T diagonal
function _diag_eigs(T::AbstractMatrix{R}) where {R<:Real}
    n = size(T, 1); w = Vector{Complex{R}}(undef, n)
    @inbounds for k in 1:n; w[k] = Complex(T[k, k], zero(R)); end
    @inbounds for k in 1:n-1
        if T[k+1, k] != zero(R)
            wi = sqrt(abs(T[k, k+1])) * sqrt(abs(T[k+1, k]))
            w[k] = Complex(real(w[k]), wi); w[k+1] = Complex(real(w[k+1]), -wi)
        end
    end
    return w
end
_diag_eigs(T::AbstractMatrix{C}) where {C<:Complex} = [T[k, k] for k in 1:size(T, 1)]

_one_norm(T) = maximum(sum(abs, T; dims = 1))

"""
    trsen!(job, compq, select, T, Q) -> (T, Q, w, s, sep)

Reorder the eigenvalues selected by `select` (a `Bool`/`0-1` vector) to the leading diagonal block of
the (quasi-)upper-triangular Schur form `T`, accumulating the swaps into `Q` when `compq='V'`
(LAPACK `dtrsen`/`ztrsen`). `job` selects which condition numbers are returned:
`'N'` none, `'E'` the cluster reciprocal condition `s`, `'V'` the invariant-subspace separation `sep`,
`'B'` both (`s`/`sep` default to `1`/`0` for the other jobs). `w` are the reordered eigenvalues (the
`T` diagonal, with conjugate pairs for real `T`). For real `T`, `select` on either half of a
conjugate pair selects the whole 2×2 block.
"""
function trsen!(job::AbstractChar, compq::AbstractChar, select::AbstractVector,
        T::AbstractMatrix, Q::AbstractMatrix)
    (job === 'N' || job === 'E' || job === 'V' || job === 'B') ||
        throw(ArgumentError("trsen!: job must be 'N', 'E', 'V' or 'B'"))
    (compq === 'V' || compq === 'N') || throw(ArgumentError("trsen!: compq must be 'V' or 'N'"))
    n = size(T, 1)
    size(T, 2) == n || throw(DimensionMismatch("trsen!: T must be square"))
    length(select) == n || throw(DimensionMismatch("trsen!: select must have length n"))
    wantq = compq === 'V'
    selb = Bool[select[i] != 0 for i in 1:n]
    Tr, Qr, w, s, sep, _ = _trsen_dispatch!(job, wantq, selb, T, Q)
    return Tr, Qr, w, s, sep
end
_trsen_dispatch!(job, wantq, sel, T::AbstractMatrix{<:Real}, Q) = _dtrsen!(job, wantq, sel, T, Q)
_trsen_dispatch!(job, wantq, sel, T::AbstractMatrix{<:Complex}, Q) = _ztrsen!(job, wantq, sel, T, Q)
