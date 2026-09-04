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
function _exc_dlanv2(a::R, b::R, c::R, d::R) where {R <: Real}
    Z0 = zero(R); ONE = one(R); HALF = R(0.5); TWO = R(2); MULTPL = R(4)
    eps_p = eps(R)
    safmin = _syl_safmin(R)
    safmn2 = TWO^trunc(Int, log(safmin / eps_p) / log(TWO) / 2)
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
function _exc_larfg!(alpha::R, x::AbstractVector{R}) where {R <: Real}
    xnorm = zero(R); @inbounds for xi in x
        xnorm += xi * xi
    end
    xnorm = sqrt(xnorm)
    xnorm == zero(R) && return alpha, zero(R)          # tau = 0
    beta = -copysign(hypot(alpha, xnorm), alpha)
    safmn = _syl_safmin(R) / (eps(R) / 2)
    knt = 0
    if abs(beta) < safmn
        rsafmn = one(R) / safmn
        while true
            knt += 1
            @inbounds for i in eachindex(x)
                x[i] *= rsafmn
            end
            beta *= rsafmn; alpha *= rsafmn
            (abs(beta) < safmn && knt < 20) || break
        end
        xnorm = zero(R); @inbounds for xi in x
            xnorm += xi * xi
        end
        xnorm = sqrt(xnorm)
        beta = -copysign(hypot(alpha, xnorm), alpha)
    end
    tau = (beta - alpha) / beta
    s = one(R) / (alpha - beta)
    @inbounds for i in eachindex(x)
        x[i] *= s
    end
    for _ in 1:knt
        beta *= safmn
    end
    return beta, tau
end

# ── DLARFX apply (order ≤ 3 reflector v with explicit unit, generic dense) ─────────────────────────────
@inline function _exc_larfx_l!(v::AbstractVector{R}, tau::R, C::AbstractMatrix{R}) where {R}
    tau == zero(R) && return
    m = size(C, 1); n = size(C, 2)
    return @inbounds for j in 1:n
        s = zero(R); for i in 1:m
            s += v[i] * C[i, j]
        end
        s *= tau
        for i in 1:m
            C[i, j] -= v[i] * s
        end
    end
end
@inline function _exc_larfx_r!(v::AbstractVector{R}, tau::R, C::AbstractMatrix{R}) where {R}
    tau == zero(R) && return
    m = size(C, 1); n = size(C, 2)
    return @inbounds for i in 1:m
        s = zero(R); for j in 1:n
            s += C[i, j] * v[j]
        end
        s *= tau
        for j in 1:n
            C[i, j] -= s * v[j]
        end
    end
end

# ── DROT on strided lanes (x' = c·x + s·y, y' = c·y − s·x) ─────────────────────────────────────────────
@inline function _exc_rot_rows!(M::AbstractMatrix, r1::Int, r2::Int, lo::Int, hi::Int, cs, sn)
    return @inbounds for j in lo:hi
        t = M[r1, j]; u = M[r2, j]
        M[r1, j] = cs * t + sn * u
        M[r2, j] = cs * u - conj(sn) * t
    end
end
@inline function _exc_rot_cols!(M::AbstractMatrix, c1::Int, c2::Int, lo::Int, hi::Int, cs, sn)
    return @inbounds for i in lo:hi
        t = M[i, c1]; u = M[i, c2]
        M[i, c1] = cs * t + sn * u
        M[i, c2] = cs * u - conj(sn) * t
    end
end

# ── DLAEXC (Reference-LAPACK verbatim) — swap adjacent diagonal blocks (n1,n2 ∈ {1,2}) at J1 ───────────
# Returns info (0 ok; 1 = swap would deflate to ill-conditioned / rejected → caller aborts the reorder).
function _dlaexc!(
        wantq::Bool, T::AbstractMatrix{R}, Q::AbstractMatrix{R},
        j1::Int, n1::Int, n2::Int
    ) where {R <: Real}
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
    # D is borrowed EXACTLY nd×nd, never a fixed 4×4: the dnorm loop below is `for x in D`, so an
    # oversized buffer at nd == 3 would fold 7 stale elements into `thresh` and silently change which
    # swaps are accepted. Fully written by the copy loop, so no fill! is needed. u1/u2 are separate
    # borrows: in the n1=2,n2=2 branch u1 is still read and applied while u2 is live. The scope opens here
    # rather than at the top because the n1=n2=1 arm above returns without any scratch at all.
    @scope arn begin
        D = borrow!(arn, R, nd, nd)
        u1buf = borrow!(arn, R, 3)
        u2buf = borrow!(arn, R, 3)
        @inbounds for jj in 1:nd, ii in 1:nd
            D[ii, jj] = T[j1 + ii - 1, j1 + jj - 1]
        end
        dnorm = ZERO; @inbounds for x in D
            dnorm = max(dnorm, abs(x))
        end
        eps_p = eps(R); smlnum = _syl_safmin(R) / eps_p
        thresh = max(TEN * eps_p * dnorm, smlnum)
        TL = view(D, 1:n1, 1:n1); TR = view(D, (n1 + 1):nd, (n1 + 1):nd)
        # RHS block D[1:n1, n1+1:nd] as the four column-major scalars `_syl_dlasy2` actually reads.
        bb11 = D[1, n1 + 1]
        bb21 = n1 == 2 ? D[2, n1 + 1] : ZERO
        bb12 = n2 == 2 ? D[1, n1 + 2] : ZERO
        bb22 = (n1 == 2 && n2 == 2) ? D[2, n1 + 2] : ZERO
        x11, x21, x12, x22, scale, _, _ = _syl_dlasy2(false, false, -1, n1, n2, TL, TR, bb11, bb21, bb12, bb22)
        k = n1 + n1 + n2 - 3
        if k == 1
            # n1=1, n2=2
            u = u1buf
            u[1] = scale; u[2] = x11; u[3] = x12
            _, tau = _exc_larfg!(u[3], view(u, 1:2)); u[3] = ONE
            t11 = T[j1, j1]
            _exc_larfx_l!(u, tau, view(D, 1:3, 1:3)); _exc_larfx_r!(u, tau, view(D, 1:3, 1:3))
            max(abs(D[3, 1]), abs(D[3, 2]), abs(D[3, 3] - t11)) > thresh && return 1
            _exc_larfx_l!(u, tau, view(T, j1:(j1 + 2), j1:n))
            _exc_larfx_r!(u, tau, view(T, 1:j2, j1:(j1 + 2)))
            T[j3, j1] = ZERO; T[j3, j2] = ZERO; T[j3, j3] = t11
            wantq && _exc_larfx_r!(u, tau, view(Q, 1:n, j1:(j1 + 2)))
        elseif k == 2
            # n1=2, n2=1
            u = u1buf
            u[1] = -x11; u[2] = -x21; u[3] = scale
            _, tau = _exc_larfg!(u[1], view(u, 2:3)); u[1] = ONE
            t33 = T[j3, j3]
            _exc_larfx_l!(u, tau, view(D, 1:3, 1:3)); _exc_larfx_r!(u, tau, view(D, 1:3, 1:3))
            max(abs(D[2, 1]), abs(D[3, 1]), abs(D[1, 1] - t33)) > thresh && return 1
            _exc_larfx_r!(u, tau, view(T, 1:j3, j1:(j1 + 2)))
            _exc_larfx_l!(u, tau, view(T, j1:(j1 + 2), j2:n))
            T[j1, j1] = t33; T[j2, j1] = ZERO; T[j3, j1] = ZERO
            wantq && _exc_larfx_r!(u, tau, view(Q, 1:n, j1:(j1 + 2)))
        else
            # n1=2, n2=2
            u1 = u1buf
            u1[1] = -x11; u1[2] = -x21; u1[3] = scale
            _, tau1 = _exc_larfg!(u1[1], view(u1, 2:3)); u1[1] = ONE
            temp = -tau1 * (x12 + u1[2] * x22)
            u2 = u2buf
            u2[1] = -temp * u1[2] - x22; u2[2] = -temp * u1[3]; u2[3] = scale
            _, tau2 = _exc_larfg!(u2[1], view(u2, 2:3)); u2[1] = ONE
            _exc_larfx_l!(u1, tau1, view(D, 1:3, 1:4)); _exc_larfx_r!(u1, tau1, view(D, 1:4, 1:3))
            _exc_larfx_l!(u2, tau2, view(D, 2:4, 1:4)); _exc_larfx_r!(u2, tau2, view(D, 1:4, 2:4))
            max(abs(D[3, 1]), abs(D[3, 2]), abs(D[4, 1]), abs(D[4, 2])) > thresh && return 1
            _exc_larfx_l!(u1, tau1, view(T, j1:(j1 + 2), j1:n))
            _exc_larfx_r!(u1, tau1, view(T, 1:j4, j1:(j1 + 2)))
            _exc_larfx_l!(u2, tau2, view(T, j2:(j2 + 2), j1:n))
            _exc_larfx_r!(u2, tau2, view(T, 1:j4, j2:(j2 + 2)))
            T[j3, j1] = ZERO; T[j3, j2] = ZERO; T[j4, j1] = ZERO; T[j4, j2] = ZERO
            if wantq
                _exc_larfx_r!(u1, tau1, view(Q, 1:n, j1:(j1 + 2)))
                _exc_larfx_r!(u2, tau2, view(Q, 1:n, j2:(j2 + 2)))
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
end

# ── DTREXC (Reference-LAPACK verbatim), REAL quasi-triangular T ────────────────────────────────────────
# Move the block at IFST to ILST. Returns (info, ilst_final).
function _dtrexc!(
        wantq::Bool, T::AbstractMatrix{R}, Q::AbstractMatrix{R},
        ifst0::Int, ilst0::Int
    ) where {R <: Real}
    ZERO = zero(R)
    n = size(T, 1)
    n <= 1 && return 0, ilst0
    ifst = ifst0; ilst = ilst0
    ifst > 1 && T[ifst, ifst - 1] != ZERO && (ifst -= 1)
    nbf = 1
    ifst < n && T[ifst + 1, ifst] != ZERO && (nbf = 2)
    ilst > 1 && T[ilst, ilst - 1] != ZERO && (ilst -= 1)
    nbl = 1
    ilst < n && T[ilst + 1, ilst] != ZERO && (nbl = 2)
    ifst == ilst && return 0, ilst
    if ifst < ilst
        nbf == 2 && nbl == 1 && (ilst -= 1)
        nbf == 1 && nbl == 2 && (ilst += 1)
        here = ifst
        while true
            if nbf == 1 || nbf == 2
                nbnext = 1
                here + nbf + 1 <= n && T[here + nbf + 1, here + nbf] != ZERO && (nbnext = 2)
                info = _dlaexc!(wantq, T, Q, here, nbf, nbnext)
                info != 0 && return info, here
                here += nbnext
                nbf == 2 && T[here + 1, here] == ZERO && (nbf = 3)
            else
                nbnext = 1
                here + 3 <= n && T[here + 3, here + 2] != ZERO && (nbnext = 2)
                info = _dlaexc!(wantq, T, Q, here + 1, 1, nbnext)
                info != 0 && return info, here
                if nbnext == 1
                    _dlaexc!(wantq, T, Q, here, 1, nbnext)
                    here += 1
                else
                    T[here + 2, here + 1] == ZERO && (nbnext = 1)
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
                here >= 3 && T[here - 1, here - 2] != ZERO && (nbnext = 2)
                info = _dlaexc!(wantq, T, Q, here - nbnext, nbnext, nbf)
                info != 0 && return info, here
                here -= nbnext
                nbf == 2 && T[here + 1, here] == ZERO && (nbf = 3)
            else
                nbnext = 1
                here >= 3 && T[here - 1, here - 2] != ZERO && (nbnext = 2)
                info = _dlaexc!(wantq, T, Q, here - nbnext, nbnext, 1)
                info != 0 && return info, here
                if nbnext == 1
                    _dlaexc!(wantq, T, Q, here, nbnext, 1)
                    here -= 1
                else
                    T[here, here - 1] == ZERO && (nbnext = 1)
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
function _ztrexc!(
        wantq::Bool, T::AbstractMatrix{C}, Q::AbstractMatrix{C},
        ifst::Int, ilst::Int
    ) where {C <: Complex}
    n = size(T, 1)
    (n <= 1 || ifst == ilst) && return 0, ilst
    m1, m2, m3 = ifst < ilst ? (0, -1, 1) : (-1, 0, -1)
    for k in (ifst + m1):m3:(ilst + m2)
        t11 = T[k, k]; t22 = T[k + 1, k + 1]
        cs, sn, _ = givensAlgorithm(T[k, k + 1], t22 - t11)
        # row rotation uses SN; column rotations use conj(SN) (ZROT convention, ztrexc.f)
        k + 2 <= n && _exc_rot_rows!(T, k, k + 1, k + 2, n, cs, sn)
        _exc_rot_cols!(T, k, k + 1, 1, k - 1, cs, conj(sn))
        T[k, k] = t22; T[k + 1, k + 1] = t11
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
    # Arena borrows so trrfs!/trsen! allocate nothing here — this ran once per right-hand side, so it was
    # 3·nrhs allocations per trrfs! call. The bytes are REUSED and carry whatever the previous borrow at
    # this offset left, so what the old `fill`/`zeros` provided is still explicit below.
    #
    # SIZE HONESTLY: these are O(n) in THIS function's `n`, but `n` is not the caller's matrix order at
    # every call site. trrfs! passes its own n; `_dtrsen!`/`_ztrsen!` pass `nn = n1*n2`, up to n²/4. So
    # the deepest nesting in stage 2 is also the widest: three n²/4 borrows live across the whole
    # estimator, inside the caller's own scope, above `apply!` → `_dtrsyl!`, which opens the stage-1
    # scope at trsyl.jl:209.
    #
    # Only `fill!(x, …)` is load-bearing: x IS read before it is written (the first `apply!` consumes it).
    # `v` is write-only in this function (`copyto!(v, x)` below; the n==1 path returns `abs(x[1])`), and
    # `isgn` is always written by `_lacn2_sign!` before `_lacn2_signchanged` reads it. Their fills are
    # therefore defensive, not required — kept because they are O(n) against an O(n²) estimator and they
    # keep the buffer state independent of call history, which is what makes the poison-invariance test
    # meaningful. Do not "optimise" them away without re-running that test.
    # ESCAPE AUDIT: `x` crosses the `apply!` callback five times. The three closures that reach here are
    # `applyf` (trrfs.jl) and the two `apply!`s (`_dtrsen!`/`_ztrsen!` below); each writes through `x`
    # and returns, none stores it. `v`/`isgn` never leave this function.
    @scope arn begin
        x = borrow!(arn, V, n)
        v = borrow!(arn, V, n)
        isgn = borrow!(arn, Int, n)          # length n for BOTH real and complex (unlike gecon's lacnsgn)
        fill!(x, V(one(R) / R(n))); fill!(v, zero(V)); fill!(isgn, 0)
        onenorm(w) = (
            s = zero(R); @inbounds for wi in w
                s += abs(wi)
            end; s
        )
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
end
@inline function _lacn2_sign!(x, isgn, ::Type{V}) where {V}
    R = real(V)
    return if V <: Complex
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
function _dtrsen!(
        job::AbstractChar, wantq::Bool, select::AbstractVector,
        T::AbstractMatrix{R}, Q::AbstractMatrix{R}, w::AbstractVector{<:Complex}
    ) where {R <: Real}
    ZERO = zero(R); ONE = one(R)
    n = size(T, 1)
    wants = job === 'E' || job === 'B'
    wantsp = job === 'V' || job === 'B'
    sel = select                            # read `sel[k] != 0` directly — no Bool copy of the caller's
    # count selected (respecting 2×2 conj pairs)
    m = 0; pair = false
    for k in 1:n
        if pair
            pair = false; continue
        end
        if k < n && T[k + 1, k] != ZERO
            pair = true
            (sel[k] != 0 || sel[k + 1] != 0) && (m += 2)
        else
            sel[k] != 0 && (m += 1)
        end
    end
    n1 = m; n2 = n - m
    s = ONE; sep = ZERO; info = 0
    if !(m == n || m == 0)
        # reorder selected eigenvalues to the leading positions via dtrexc swaps
        ks = 0; pair = false
        for k in 1:n
            if pair
                pair = false; continue
            end
            swap = sel[k] != 0
            if k < n && T[k + 1, k] != ZERO
                pair = true
                swap = swap || sel[k + 1] != 0
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
            # Exact n1×n2 borrow, default ld: the old `trsenr` field carried no anti-aliasing ld, and the
            # only consumer is `_dtrsyl!`, which strides by `ld` like any other operand.
            # ESCAPE AUDIT: `Rm` is written by `copyto!`, handed DOWN to `_dtrsyl!` — which solves in place
            # and returns it as its first value, discarded here into `_` — and then read by the Frobenius
            # loop, all inside this block. trsyl.jl assigns no field and no global (its own scratch is the
            # stage-1 `@scope` at trsyl.jl:209) and its `_syl_dlasy2` callee takes scalars, so nothing
            # retains the handle. What leaves the scope is the scalar `s`.
            @scope arn begin
                Rm = borrow!(arn, R, n1, n2)                        # off-diagonal coupling block T₁₂
                copyto!(Rm, view(T, 1:n1, (n1 + 1):n))              # fully overwritten ⇒ no fill!
                _, scale, _ = _dtrsyl!('N', 'N', -1, view(T, 1:n1, 1:n1), view(T, (n1 + 1):n, (n1 + 1):n), Rm)
                rnorm = (
                    rn2 = zero(real(eltype(Rm))); @inbounds for x in Rm
                        rn2 += abs2(x)
                    end; sqrt(rn2)
                )
                s = rnorm == ZERO ? ONE : scale / (sqrt(scale^2 / rnorm + rnorm) * sqrt(rnorm))
            end
        end
    end
    if info == 0 && wantsp
        if m == n || m == 0
            sep = _one_norm(T)
        else
            nn = n1 * n2
            # ONE scope for the whole SEP estimate — the four buffers have the SAME lifetime, so they are
            # borrowed together rather than in a second nested scope. THE SCOPE SPANS THE ESTIMATE AND THE
            # READ. `scref` is a 1-slot carrier for the scale that `_dtrsyl!` returns through the closure;
            # it is CAPTURED by `apply!`, written on every kase, and read at `sep = …` AFTER
            # `_lacn2_estimate` has returned. Scoping only the `apply!` block would release it before that
            # read — the one way to get this role wrong. The borrows sit ABOVE the closure, as the macro
            # requires (a `borrow!` lexically inside `apply!` is rejected at expansion time).
            #
            # SIZE: T11 is n1×n1, T22 n2×n2, Xm n1×n2 — these GROW with the caller's order (unlike the
            # stage-1 fixed roles), and `_lacn2_estimate` nests its own three n1·n2 borrows inside this
            # scope, above `_dtrsyl!`'s stage-1 scope at trsyl.jl:209. Exact ld throughout: the fields
            # these replace carried no anti-aliasing ld, and every consumer strides by `ld`.
            #
            # ESCAPE AUDIT. `T11`/`T22`/`Xm`/`scref` are captured by the local closure `apply!`, which is
            # passed to `_lacn2_estimate` and CALLED there; that function only invokes the callback (it
            # never stores it or the handles, and its own x/v/isgn are its own borrows). Through the
            # callback the handles reach `_dtrsyl!`, which solves in place, returns `Xm` as a discarded
            # `_`, and assigns no field or global. `apply!` itself is a local that dies with the block, and
            # only the scalar `sep` leaves. Nothing is returned, stored in a field, or captured by anything
            # outliving this scope.
            @scope sca begin
                T11 = borrow!(sca, R, n1, n1)
                T22 = borrow!(sca, R, n2, n2)
                Xm = borrow!(sca, R, n1, n2)          # all three fully overwritten before any read
                scref = borrow!(sca, R, 1); scref[1] = ONE
                copyto!(T11, view(T, 1:n1, 1:n1)); copyto!(T22, view(T, (n1 + 1):n, (n1 + 1):n))
                # `_lacn2_estimate` hands the closure an arena VECTOR, and `reshape(::SubArray, …)`
                # goes through Base `_reshape` → `_throw_dmrs`, whose 8-piece eagerly-interpolated message
                # despecialises to `print_to_string(::String, ::Vararg{Any})` and fails `--trim=safe`
                # (`reshape(::Vector, …)` took Array's own method, which uses a trim-clean LazyString).
                # `Xm` is a 2-D n1×n2 handle, which dodges that: the staging is `copyto!`, never a reshape.
                # It is now a `PtrMatrix` — isbits and strided, so this also takes _dtrsyl! OFF the SubArray
                # path the workspace view had put it on. `_lacn2_estimate` itself stays 0-alloc, which is
                # what trrfs! needs.
                apply! = function (xv, kase)
                    copyto!(Xm, xv)
                    _, sc, _ = kase == 1 ? _dtrsyl!('N', 'N', -1, T11, T22, Xm) :
                        _dtrsyl!('T', 'T', -1, T11, T22, Xm)
                    copyto!(xv, Xm)
                    scref[1] = sc
                    return nothing
                end
                # estimate ‖L⁻¹‖₁ of the Sylvester operator; SEP = scale/est (cancels trsyl's scaling)
                est = _lacn2_estimate(nn, apply!, R)
                sep = est == ZERO ? ZERO : scref[1] / est
            end
        end
    end
    _diag_eigs!(w, T)
    return T, Q, s, sep, info
end

function _ztrsen!(
        job::AbstractChar, wantq::Bool, select::AbstractVector,
        T::AbstractMatrix{C}, Q::AbstractMatrix{C}, w::AbstractVector{<:Complex}
    ) where {C <: Complex}
    R = real(C)
    n = size(T, 1)
    wants = job === 'E' || job === 'B'
    wantsp = job === 'V' || job === 'B'
    sel = select                            # read `sel[k] != 0` directly — no Bool copy of the caller's
    m = 0; @inbounds for i in 1:n
        sel[i] != 0 && (m += 1)
    end   # not count(): mapreduce MappingRF is --trim-unsafe
    n1 = m; n2 = n - m
    s = one(R); sep = zero(R); info = 0
    if !(m == n || m == 0)
        ks = 0
        for k in 1:n
            if sel[k] != 0
                ks += 1
                k != ks && _ztrexc!(wantq, T, Q, k, ks)
            end
        end
    end
    if wants
        if m == n || m == 0
            s = one(R)
        else
            # ESCAPE AUDIT: identical to `_dtrsen!`'s — `Rm` is filled by `copyto!`, passed DOWN to
            # `_ztrsyl!` (in-place solve; its return of `Rm` is discarded into `_`), then read by the
            # Frobenius loop, all inside the block. trsyl.jl assigns no field and no global, so nothing
            # retains the handle; only the scalar `s` leaves.
            @scope arn begin
                Rm = borrow!(arn, C, n1, n2)
                copyto!(Rm, view(T, 1:n1, (n1 + 1):n))              # fully overwritten ⇒ no fill!
                _, scale, _ = _ztrsyl!('N', 'N', -1, view(T, 1:n1, 1:n1), view(T, (n1 + 1):n, (n1 + 1):n), Rm)
                rnorm = (
                    rn2 = zero(real(eltype(Rm))); @inbounds for x in Rm
                        rn2 += abs2(x)
                    end; sqrt(rn2)
                )
                s = rnorm == zero(R) ? one(R) : scale / (sqrt(scale^2 / rnorm + rnorm) * sqrt(rnorm))
            end
        end
    end
    if wantsp
        if m == n || m == 0
            sep = _one_norm(T)
        else
            nn = n1 * n2
            # One scope for all four, as in `_dtrsen!` — same lifetime, so no second nested scope, and the
            # borrows sit above the closure because the macro rejects a `borrow!` inside one. `scref` is
            # the 1-slot carrier; the scale is real, so it rides as C and reads back `real`. The scope must
            # span the estimate AND the `sep = …` read below, because the closure is what writes the slot.
            # Xm stays a 2-D handle for the same trim reason spelled out in `_dtrsen!` (no reshape on the
            # staging path); as a `PtrMatrix` it is isbits and strided, keeping _ztrsyl! off the SubArray path.
            # ESCAPE AUDIT: `T11`/`T22`/`Xm`/`scref` are captured only by the local `apply!`, which
            # `_lacn2_estimate` calls but never stores; from there they reach `_ztrsyl!`, which solves in
            # place, returns `Xm` into a discarded `_`, and assigns no field or global. Only the scalar
            # `sep` leaves the block.
            @scope sca begin
                T11 = borrow!(sca, C, n1, n1)
                T22 = borrow!(sca, C, n2, n2)
                Xm = borrow!(sca, C, n1, n2)
                scref = borrow!(sca, C, 1); scref[1] = one(C)
                copyto!(T11, view(T, 1:n1, 1:n1)); copyto!(T22, view(T, (n1 + 1):n, (n1 + 1):n))
                apply! = function (xv, kase)
                    copyto!(Xm, xv)
                    _, sc, _ = kase == 1 ? _ztrsyl!('N', 'N', -1, T11, T22, Xm) :
                        _ztrsyl!('C', 'C', -1, T11, T22, Xm)
                    copyto!(xv, Xm)
                    scref[1] = C(sc)
                    return nothing
                end
                est = _lacn2_estimate(nn, apply!, C)
                sep = est == zero(R) ? zero(R) : real(scref[1]) / est
            end
        end
    end
    _diag_eigs!(w, T)
    return T, Q, s, sep, info
end

# eigenvalues from the (quasi-)triangular T diagonal, into a caller-provided w
function _diag_eigs!(w::AbstractVector{<:Complex}, T::AbstractMatrix{R}) where {R <: Real}
    n = size(T, 1)
    @inbounds for k in 1:n
        w[k] = Complex(T[k, k], zero(R))
    end
    @inbounds for k in 1:(n - 1)
        if T[k + 1, k] != zero(R)
            wi = sqrt(abs(T[k, k + 1])) * sqrt(abs(T[k + 1, k]))
            w[k] = Complex(real(w[k]), wi); w[k + 1] = Complex(real(w[k + 1]), -wi)
        end
    end
    return w
end
function _diag_eigs!(w::AbstractVector{<:Complex}, T::AbstractMatrix{C}) where {C <: Complex}
    @inbounds for k in 1:size(T, 1)
        w[k] = T[k, k]
    end
    return w
end

function _one_norm(T)                                    # explicit (not sum(;dims)+maximum: MappingRF --trim-unsafe)
    mx = zero(real(eltype(T)))
    @inbounds for j in 1:size(T, 2)
        c = zero(real(eltype(T))); for i in 1:size(T, 1)
            c += abs(T[i, j])
        end
        c > mx && (mx = c)
    end
    return mx
end

"""
    trsen!(job, compq, select, T, Q, w) -> (T, Q, w, s, sep)
    trsen!(job, compq, select, T, Q)    -> (T, Q, w, s, sep)

Reorder the eigenvalues selected by `select` (a `Bool`/`0-1` vector) to the leading diagonal block of
the (quasi-)upper-triangular Schur form `T`, accumulating the swaps into `Q` when `compq='V'`
(LAPACK `dtrsen`/`ztrsen`). `job` selects which condition numbers are returned:
`'N'` none, `'E'` the cluster reciprocal condition `s`, `'V'` the invariant-subspace separation `sep`,
`'B'` both (`s`/`sep` default to `1`/`0` for the other jobs). `w` are the reordered eigenvalues (the
`T` diagonal, with conjugate pairs for real `T`). For real `T`, `select` on either half of a
conjugate pair selects the whole 2×2 block.

`w` is the only allocated output; the 6-argument form takes the caller's (complex, length ≥ n) buffer
and allocates nothing. It must not overlap `T` — `w` is filled from `T`'s diagonal at the very end.
"""
function trsen!(
        job::AbstractChar, compq::AbstractChar, select::AbstractVector,
        T::AbstractMatrix, Q::AbstractMatrix, w::AbstractVector{<:Complex}
    )
    (job === 'N' || job === 'E' || job === 'V' || job === 'B') ||
        throw(ArgumentError("trsen!: job must be 'N', 'E', 'V' or 'B'"))
    (compq === 'V' || compq === 'N') || throw(ArgumentError("trsen!: compq must be 'V' or 'N'"))
    n = size(T, 1)
    size(T, 2) == n || throw(DimensionMismatch("trsen!: T must be square"))
    length(select) == n || throw(DimensionMismatch("trsen!: select must have length n"))
    length(w) >= n || throw(DimensionMismatch("trsen!: length(w) < n"))
    # `_diag_eigs!` reads T[k,k]/T[k+1,k] while writing w[k]/w[k+1] — an overlapping w corrupts both.
    Base.mightalias(w, T) && throw(ArgumentError("trsen!: `w` must not alias `T`"))
    wantq = compq === 'V'
    Tr, Qr, s, sep, _ = _trsen_dispatch!(job, wantq, select, T, Q, w)
    return Tr, Qr, w, s, sep
end

# Allocating convenience form — unchanged behaviour and return value.
function trsen!(
        job::AbstractChar, compq::AbstractChar, select::AbstractVector,
        T::AbstractMatrix, Q::AbstractMatrix
    )
    w = Vector{Complex{real(eltype(T))}}(undef, size(T, 1))
    return trsen!(job, compq, select, T, Q, w)
end

_trsen_dispatch!(job, wantq, sel, T::AbstractMatrix{<:Real}, Q, w) = _dtrsen!(job, wantq, sel, T, Q, w)
_trsen_dispatch!(job, wantq, sel, T::AbstractMatrix{<:Complex}, Q, w) = _ztrsen!(job, wantq, sel, T, Q, w)

# 5-argument engine entry kept for the C-ABI shims (cabi_lapack.jl:2607/2635), which need `info` and the
# allocated `w`. Same kernels; the buffer is the only difference.
function _trsen_dispatch!(job, wantq, sel, T::AbstractMatrix, Q)
    w = Vector{Complex{real(eltype(T))}}(undef, size(T, 1))
    Tr, Qr, s, sep, info = _trsen_dispatch!(job, wantq, sel, T, Q, w)
    return Tr, Qr, w, s, sep, info
end
