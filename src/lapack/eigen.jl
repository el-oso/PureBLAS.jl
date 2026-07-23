# LAPACK symmetric eigensolver (M-E1) — REAL Float64. Minimal correct chain:
#   sytrd (tridiagonalize) → steqr (tridiagonal QL/QR eigensolver) → ormtr (back-transform vectors).
# Built from three separately-validated kernels; integrated onto PureBLAS BLAS (symv!) and the module's
# SIMD Givens (_givens/_rot_cols!/_swap_cols! from svd.jl). Routes eigen(Symmetric)/eigvals(Symmetric)
# after activate() via the C-ABI syev/syevd/syevr shims (cabi_lapack.jl). Complex/Float32 and the O(n²)
# sterf eigvals path and the divide-and-conquer stedc are LATER milestones (M-E2/M-E3) — out of scope.

# ── Stage 1: unblocked symmetric tridiagonalization (LAPACK dsytd2, LOWER) ─────────────────────────
# Reduces the lower triangle of symmetric A to tridiagonal T = QᵀAQ: d = diagonal, e = subdiagonal.
# One Householder reflector per column; essential vᵢ stored in A[i+2:n, i] (vᵢ[1]≡1 implicit at row i+1),
# tau[i] the standard LAPACK τ. The trailing symmetric matvec is PureBLAS symv! (the memory-bound half of
# the 4/3·n³ flops). Copies dsytd2's τ/2 half-correction verbatim (w += (−τ/2·(wᵀv))·v).
function _sytd2_lower!(
        A::AbstractMatrix{T}, d::AbstractVector{T},
        e::AbstractVector{T}, tau::AbstractVector{T}
    ) where {T <: Real}
    n = size(A, 1)
    n == 0 && return
    v = Vector{T}(undef, n)
    w = Vector{T}(undef, n)
    @inbounds for i in 1:(n - 1)
        m = n - i
        col = view(A, (i + 1):n, i)
        β, τ = _larfg!(col)                       # essential v now in A[i+2:n,i]; col[1]=A[i+1,i] left as-is
        e[i] = β; tau[i] = τ; d[i] = A[i, i]
        if τ != zero(T) && m > 1
            v[1] = one(T)
            for r in 2:m
                v[r] = A[i + r, i]
            end
            Atr = view(A, (i + 1):n, (i + 1):n); vv = view(v, 1:m); ww = view(w, 1:m)
            symv!(ww, Atr, vv; uplo = 'L', alpha = τ, beta = zero(T))   # w = τ·A_trailing·v (lower symmetric)
            s = zero(T)
            for r in 1:m
                s = muladd(ww[r], vv[r], s)
            end
            c = -(τ / 2) * s                       # dsytd2 half-correction
            for r in 1:m
                ww[r] = muladd(c, vv[r], ww[r])
            end
            for jc in 1:m, ir in jc:m              # symmetric rank-2 downdate (lower triangle)
                Atr[ir, jc] -= vv[ir] * ww[jc] + ww[ir] * vv[jc]
            end
        end
    end
    d[n] = A[n, n]
    return
end

# ── Stage 1b: BLOCKED tridiagonalization (LAPACK dlatrd panel + dsyr2k trailing update), LOWER ──────
# The unblocked `_sytd2_lower!` above does ONE memory-bound `symv!` on the shrinking trailing submatrix
# PER column (n Level-2 sweeps, no cache blocking) — it was ~90% of a `jobz='N'` solve and dragged PB to
# 0.63× OB at n=2048. dlatrd reduces `nb` columns of the trailing submatrix at a time, accumulating W so
# HALF the flops move into ONE rank-2·nb `syr2k!` (Level-3, cache-blocked) — the structural OB match.
# Direct-gemv helper `_eg!` skips the kwarg wrapper's dispatch/char-parse (mirrors svd.jl's `_lg!` in the
# analogous blocked bidiagonalization `_labrd!`). Generic over T so complex `_hetrd!` can share the shape.
@inline _eg!(yv, Av, xv, α::T, β::T, tr::Bool) where {T} =
    _gemv!(tr, false, size(Av, 1), size(Av, 2), α, Av, xv, 1, β, yv, 1)
# Conj-aware variant for the complex (zlatrd) panel: op(A) = Aᴴ needs (tr=true, cj=true).
@inline _egc!(yv, Av, xv, α::T, β::T, tr::Bool, cj::Bool) where {T} =
    _gemv!(tr, cj, size(Av, 1), size(Av, 2), α, Av, xv, 1, β, yv, 1)

# Reduce the first `nb` columns of the ns×ns trailing submatrix `As` to tridiagonal form, filling
# d[1:nb], e[1:nb], tau[1:nb] and the panel W[:,1:nb] that drives the caller's syr2k. Faithful dlatrd
# (lower): symv unscaled, the τ scaling + half-correction applied last (so the cross-term gemvs see the
# raw A·v). `tmp` (≥ nb) contiguous-izes the strided W[i,·]/A[i,·] rows for the SIMD gemv fast path.
function _latrd_lower!(
        As::AbstractMatrix{T}, W::AbstractMatrix{T}, d::AbstractVector{T},
        e::AbstractVector{T}, tau::AbstractVector{T}, nb::Int, tmp::AbstractVector{T}
    ) where {T <: Real}
    ns = size(As, 1)
    @inbounds for i in 1:nb
        if i > 1
            # A(i:ns,i) -= A(i:ns,1:i-1)·W(i,1:i-1)ᵀ − W(i:ns,1:i-1)·A(i,1:i-1)ᵀ  (two rank-(i−1) gemv N)
            for t in 1:(i - 1)
                tmp[t] = W[i, t]
            end
            _eg!(view(As, i:ns, i), view(As, i:ns, 1:(i - 1)), view(tmp, 1:(i - 1)), -one(T), one(T), false)
            for t in 1:(i - 1)
                tmp[t] = As[i, t]
            end
            _eg!(view(As, i:ns, i), view(W, i:ns, 1:(i - 1)), view(tmp, 1:(i - 1)), -one(T), one(T), false)
        end
        d[i] = As[i, i]
        if i < ns
            m = ns - i
            col = view(As, (i + 1):ns, i)
            β, τ = _larfg!(col)
            e[i] = β; tau[i] = τ; As[i + 1, i] = one(T)   # explicit unit (symv v[1] AND the trailing syr2k V2)
            v = view(As, (i + 1):ns, i)
            wc = view(W, (i + 1):ns, i)
            symv!(wc, view(As, (i + 1):ns, (i + 1):ns), v; uplo = 'L', alpha = one(T), beta = zero(T))  # wc = A·v
            if i > 1
                wtop = view(W, 1:(i - 1), i)
                _eg!(wtop, view(W, (i + 1):ns, 1:(i - 1)), v, one(T), zero(T), true)     # wtop = W2ᵀ·v
                _eg!(wc, view(As, (i + 1):ns, 1:(i - 1)), wtop, -one(T), one(T), false)  # wc −= A21·wtop
                _eg!(wtop, view(As, (i + 1):ns, 1:(i - 1)), v, one(T), zero(T), true)    # wtop = A21ᵀ·v
                _eg!(wc, view(W, (i + 1):ns, 1:(i - 1)), wtop, -one(T), one(T), false)   # wc −= W2·wtop
            end
            for r in 1:m
                wc[r] *= τ
            end
            s = zero(T)
            for r in 1:m
                s = muladd(wc[r], v[r], s)
            end
            c = -(τ / 2) * s                                # dlatrd half-correction
            for r in 1:m
                wc[r] = muladd(c, v[r], wc[r])
            end
        end
    end
    return
end

# Blocked driver (LAPACK dsytrd, lower): panel-reduce nb columns (dlatrd), rank-2·nb syr2k the trailing
# block, repeat; finish the small tail unblocked. `nb`/`nx` derived (req#8): panel width reuses the QR
# cache-residency width (same rank-nb-trailing-gemm criterion); the unblocked-tail crossover `nx` is a
# multiple of nb (keep blocking while the trailing syr2k spans ≥2 panels — else it is too small to amortize
# the panel's BLAS-2 W-accumulation). Reuses `_sytd2_lower!` as the tail kernel — no separate base case.
_sytrd_nb(n::Int) = _qr_nb(n, n)
function _sytrd_lower!(
        A::AbstractMatrix{T}, d::AbstractVector{T},
        e::AbstractVector{T}, tau::AbstractVector{T}
    ) where {T <: Real}
    n = size(A, 1)
    n == 0 && return
    nb = _sytrd_nb(n)
    nx = 2 * nb                                       # unblocked-tail crossover (≥2 panels ⇒ blocking pays)
    if n <= nx || nb <= 1
        _sytd2_lower!(A, d, e, tau)
        return
    end
    W = Matrix{T}(undef, n, nb)
    tmp = Vector{T}(undef, nb)
    kk = 1
    @inbounds while n - kk + 1 > nx
        pb = min(nb, n - kk)                          # ≤ ns−1 (each panel column needs a trailing row)
        ns = n - kk + 1
        As = view(A, kk:n, kk:n)
        Wp = view(W, 1:ns, 1:pb)
        _latrd_lower!(As, Wp, view(d, kk:(kk + pb - 1)), view(e, kk:(kk + pb - 1)), view(tau, kk:(kk + pb - 1)), pb, tmp)
        rs = kk + pb                                  # trailing block A(rs:n, rs:n) -= V2·W2ᵀ + W2·V2ᵀ
        V2 = view(A, rs:n, kk:(kk + pb - 1))
        W2 = view(W, (pb + 1):ns, 1:pb)
        syr2k!(view(A, rs:n, rs:n), V2, W2; uplo = 'L', trans = 'N', alpha = -one(T), beta = one(T))
        kk += pb
    end
    _sytd2_lower!(view(A, kk:n, kk:n), view(d, kk:n), view(e, kk:(n - 1)), view(tau, kk:(n - 1)))  # tail
    return
end

# ── Stage 2: 2×2 symmetric eigensolvers (LAPACK dlae2 values / dlaev2 values+vector) ───────────────
@inline function _dlae2(a::T, b::T, c::T) where {T <: Real}
    sm = a + c; df = a - c; adf = abs(df); tb = b + b; ab = abs(tb)
    acmx = abs(a) > abs(c) ? a : c; acmn = abs(a) > abs(c) ? c : a   # dlaev2 ACMX/ACMN by magnitude
    if adf > ab
        rt = adf * sqrt(one(T) + (ab / adf)^2)
    elseif adf < ab
        rt = ab * sqrt(one(T) + (adf / ab)^2)
    else
        rt = ab * sqrt(T(2))
    end
    if sm < zero(T)
        rt1 = (sm - rt) / 2
        rt2 = ((acmx / rt1) * acmn) - (b / rt1) * b
    elseif sm > zero(T)
        rt1 = (sm + rt) / 2
        rt2 = ((acmx / rt1) * acmn) - (b / rt1) * b
    else
        rt1 = rt / 2; rt2 = -rt / 2
    end
    return rt1, rt2
end

@inline function _dlaev2(a::T, b::T, c::T) where {T <: Real}
    sm = a + c; df = a - c; adf = abs(df); tb = b + b; ab = abs(tb)
    acmx = abs(a) > abs(c) ? a : c; acmn = abs(a) > abs(c) ? c : a   # dlaev2 ACMX/ACMN by magnitude
    if adf > ab
        rt = adf * sqrt(one(T) + (ab / adf)^2)
    elseif adf < ab
        rt = ab * sqrt(one(T) + (adf / ab)^2)
    else
        rt = ab * sqrt(T(2))
    end
    local rt1, rt2, sgn1
    if sm < zero(T)
        rt1 = (sm - rt) / 2; sgn1 = -1
        rt2 = ((acmx / rt1) * acmn) - (b / rt1) * b
    elseif sm > zero(T)
        rt1 = (sm + rt) / 2; sgn1 = 1
        rt2 = ((acmx / rt1) * acmn) - (b / rt1) * b
    else
        rt1 = rt / 2; rt2 = -rt / 2; sgn1 = 1
    end
    if df >= zero(T)
        cs = df + rt; sgn2 = 1
    else
        cs = df - rt; sgn2 = -1
    end
    acs = abs(cs)
    if acs > ab
        ct = -tb / cs; sn1 = one(T) / sqrt(one(T) + ct * ct); cs1 = ct * sn1
    else
        if ab == zero(T)
            cs1 = one(T); sn1 = zero(T)
        else
            tn = -cs / tb; cs1 = one(T) / sqrt(one(T) + tn * tn); sn1 = tn * cs1
        end
    end
    if sgn1 == sgn2
        tn = cs1; cs1 = -sn1; sn1 = tn
    end
    return rt1, rt2, cs1, sn1
end

# ── Stage 2: implicit-shift QL/QR tridiagonal eigensolver (LAPACK dsteqr) ──────────────────────────
# compz ∈ {'N' no vectors, 'V' rotate given Z, 'I' Z := I then eigenvectors}. d[1:n] diag → eigenvalues
# ascending; e[1:n-1] subdiag → destroyed; Z n×n (unused if 'N'). Wilkinson shift; the module's SIMD
# Givens (_givens/_rot_cols!/_swap_cols!, svd.jl — verified identical sign convention). Generic over T<:Real.
function _steqr!(
        compz::Char, d::AbstractVector{T}, e::AbstractVector{T},
        Z::Union{AbstractMatrix{T}, Nothing}
    ) where {T <: Real}
    n = length(d)
    wantz = compz == 'V' || compz == 'I'
    n == 0 && return d
    if n == 1
        return d
    end
    if compz == 'I' && !isnothing(Z)
        fill!(Z, zero(T))
        @inbounds for i in 1:n
            Z[i, i] = one(T)
        end
    end

    eps_ = eps(T) / 2                    # dlamch('E') = relative machine precision (½ ulp)
    eps2 = eps_ * eps_
    safmin = floatmin(T)
    safmax = one(T) / safmin
    ssfmax = sqrt(safmax) / 3
    ssfmin = sqrt(safmin) / eps2
    MAXIT = 30                            # dsteqr: max QR sweeps per eigenvalue
    nmaxit = n * MAXIT
    jtot = 0

    l1 = 1
    nm1 = n - 1

    @inbounds while true
        # ------ find the next block [l:lend] via the split criterion ------------------------
        l1 > n && break
        l1 > 1 && (e[l1 - 1] = zero(T))
        m = n
        if l1 <= nm1
            found = false
            for mm in l1:nm1
                tst = abs(e[mm])
                if tst == zero(T)
                    m = mm; found = true; break
                end
                if tst <= (sqrt(abs(d[mm])) * sqrt(abs(d[mm + 1]))) * eps_
                    e[mm] = zero(T); m = mm; found = true; break
                end
            end
            found || (m = n)
        end
        l = l1; lsv = l; lend = m; lendsv = lend; l1 = m + 1
        lend == l && continue                              # 1×1 block, already an eigenvalue

        # ------ scale the block to keep |·| within [ssfmin, ssfmax] -------------------------
        anorm = zero(T)
        for i in l:lend
            anorm = max(anorm, abs(d[i]))
        end
        for i in l:(lend - 1)
            anorm = max(anorm, abs(e[i]))
        end
        anorm == zero(T) && continue
        iscale = 0
        if anorm > ssfmax
            iscale = 1; f = ssfmax / anorm
            for i in l:lend
                d[i] *= f
            end
            for i in l:(lend - 1)
                e[i] *= f
            end
        elseif anorm < ssfmin
            iscale = 2; f = ssfmin / anorm
            for i in l:lend
                d[i] *= f
            end
            for i in l:(lend - 1)
                e[i] *= f
            end
        end

        # ------ pick QL (bottom-up) or QR (top-down): iterate toward the smaller-|d| end ----
        if abs(d[lend]) < abs(d[l])
            lend = lsv; l = lendsv
        end

        if lend > l
            # =================== QL iteration ==========================================
            while true
                if l != lend
                    mfound = lend
                    for mm in l:(lend - 1)
                        tst = abs(e[mm])^2
                        if tst <= (eps2 * abs(d[mm])) * abs(d[mm + 1]) + safmin
                            mfound = mm; break
                        end
                    end
                    m = mfound
                else
                    m = lend
                end
                m < lend && (e[m] = zero(T))
                p = d[l]
                if m == l
                    d[l] = p; l += 1                       # eigenvalue converged at top
                    l <= lend ? continue : break
                end
                if m == l + 1                              # 2×2 block
                    if wantz
                        rt1, rt2, c, s = _dlaev2(d[l], e[l], d[l + 1])
                        !isnothing(Z) && _rot_cols!(Z, l, l + 1, c, s)
                    else
                        rt1, rt2 = _dlae2(d[l], e[l], d[l + 1])
                    end
                    d[l] = rt1; d[l + 1] = rt2; e[l] = zero(T)
                    l += 2
                    l <= lend ? continue : break
                end
                jtot == nmaxit && break
                jtot += 1
                g = (d[l + 1] - p) / (2 * e[l])            # Wilkinson shift
                r = hypot(g, one(T))
                g = d[m] - p + e[l] / (g + copysign(r, g))
                s = one(T); c = one(T); p = zero(T)
                for i in (m - 1):-1:l                      # chase bulge upward
                    f = s * e[i]; b = c * e[i]
                    c, s, r = _givens(g, f)
                    i != m - 1 && (e[i + 1] = r)
                    g = d[i + 1] - p
                    r = (d[i] - g) * s + 2 * c * b
                    p = s * r
                    d[i + 1] = g + p
                    g = c * r - b
                    (wantz && !isnothing(Z)) && _rot_cols!(Z, i, i + 1, c, -s)
                end
                d[l] -= p; e[l] = g
            end
        else
            # =================== QR iteration ==========================================
            while true
                if l != lend
                    mfound = lend
                    for mm in l:-1:(lend + 1)
                        tst = abs(e[mm - 1])^2
                        if tst <= (eps2 * abs(d[mm])) * abs(d[mm - 1]) + safmin
                            mfound = mm; break
                        end
                    end
                    m = mfound
                else
                    m = lend
                end
                m > lend && (e[m - 1] = zero(T))
                p = d[l]
                if m == l
                    d[l] = p; l -= 1                       # eigenvalue converged at bottom
                    l >= lend ? continue : break
                end
                if m == l - 1                              # 2×2 block
                    if wantz
                        rt1, rt2, c, s = _dlaev2(d[l - 1], e[l - 1], d[l])
                        !isnothing(Z) && _rot_cols!(Z, l - 1, l, c, s)
                    else
                        rt1, rt2 = _dlae2(d[l - 1], e[l - 1], d[l])
                    end
                    d[l - 1] = rt1; d[l] = rt2; e[l - 1] = zero(T)
                    l -= 2
                    l >= lend ? continue : break
                end
                jtot == nmaxit && break
                jtot += 1
                g = (d[l - 1] - p) / (2 * e[l - 1])        # Wilkinson shift
                r = hypot(g, one(T))
                g = d[m] - p + e[l - 1] / (g + copysign(r, g))
                s = one(T); c = one(T); p = zero(T)
                for i in m:(l - 1)                         # chase bulge downward
                    f = s * e[i]; b = c * e[i]
                    c, s, r = _givens(g, f)
                    i != m && (e[i - 1] = r)
                    g = d[i] - p
                    r = (d[i + 1] - g) * s + 2 * c * b
                    p = s * r
                    d[i] = g + p
                    g = c * r - b
                    (wantz && !isnothing(Z)) && _rot_cols!(Z, i, i + 1, c, s)
                end
                d[l] -= p; e[l - 1] = g
            end
        end

        # ------ undo scaling for this block ------------------------------------------------
        if iscale == 1
            f = anorm / ssfmax
            for i in lsv:lendsv
                d[i] *= f
            end
            for i in lsv:(lendsv - 1)
                e[i] *= f
            end
        elseif iscale == 2
            f = anorm / ssfmin
            for i in lsv:lendsv
                d[i] *= f
            end
            for i in lsv:(lendsv - 1)
                e[i] *= f
            end
        end
        jtot >= nmaxit && break                            # non-convergence guard
    end

    # ------ order eigenvalues (and eigenvector columns) ascending -------------------------
    if !wantz || isnothing(Z)
        sort!(view(d, 1:n))
    else
        @inbounds for ii in 2:n                            # selection sort, swap Z columns
            i = ii - 1; k = i; p = d[i]
            for j in ii:n
                d[j] < p && (k = j; p = d[j])
            end
            if k != i
                d[k] = d[i]; d[i] = p
                _swap_cols!(Z, i, k)
            end
        end
    end
    return d
end

# ── Stage 3: back-transform eigenvectors by Q from the (LOWER) reduction (LAPACK dormtr) ───────────
# Q = H_1·H_2·⋯·H_{n-2}, H_i = I − τ_i·v_i·v_iᵀ acts on rows i+1:n. Essential v_i in A[i+2:n,i]
# (v_i[1]≡1 at row i+1); tau[i] standard LAPACK τ. side='L' real Float64 (steqr back-transform is L/N).
#   trans='N':  C := Q·C  = H_1·(⋯·(H_{n-2}·C))   → apply i = n-2 … 1  (decreasing)
#   trans='T':  C := Qᵀ·C = H_{n-2}·(⋯·(H_1·C))   → apply i = 1 … n-2  (increasing)
#
# BLOCKED (LAPACK dlarfb / dormtr): the per-reflector `_house_left!` above is Level-2 (a rank-1 apply
# per reflector) — it was ~80% of a `jobz='V'` eigensolve at n≥512 (unblocked). Group the reflectors
# into nb-wide panels and apply each as ONE compact-WY block reflector Q_b = I − V·T·Vᵀ (`wy_t!` builds
# T, `wy_apply!` does the triple-gemm) → Level-3, matching OB's dormtr. The nb reflectors for columns
# [pc:pc+pb-1] occupy rows [pc+1:n] as a unit-lower-trapezoid (reflector at col g has its implicit 1 at
# row g+1, essential A[g+2:n,g]) — exactly the `Apanel` shape wy_t! wants. Block order: reverse for
# Q·C (rightmost block H_pc..·C applied first), forward for Qᵀ·C — the block mirror of the scalar loops.
function _ormtr!(
        A::AbstractMatrix{T}, tau::AbstractVector{T},
        C::AbstractMatrix{T}; side::Char = 'L', trans::Char = 'N'
    ) where {T <: Real}
    side === 'L' || throw(ArgumentError("_ormtr!: only side='L' implemented"))
    trans === 'N' || trans === 'T' || throw(ArgumentError("_ormtr!: trans must be 'N' or 'T'"))
    n = size(A, 1); nc = size(C, 2)
    (n <= 2 || nc == 0) && return C          # no reflectors (H_1..H_{n-2})
    k = n - 2                                 # reflector count
    nb = clamp(_qr_nb(n, nc), 1, k)           # derived panel width (cache-residency; shared with QR)
    ws = WYApplyWorkspace{T}(n, nb, nc)
    Tm = Matrix{T}(undef, nb, nb)
    nblk = cld(k, nb)
    order = trans === 'N' ? (nblk:-1:1) : (1:nblk)
    @inbounds for b in order
        pc = (b - 1) * nb + 1
        pb = min(nb, k - pc + 1)
        rs = pc + 1                           # first row H_pc acts on (v_pc[1] at row pc+1)
        m = n - rs + 1                        # panel rows (pc+1 : n)
        Vp = view(ws.V, 1:m, 1:pb)            # build explicit unit-lower-trapezoid panel
        for c in 1:pb
            g = pc + c - 1                    # global reflector column
            for r in 1:(c - 1)
                Vp[r, c] = zero(T)
            end
            Vp[c, c] = one(T)
            for r in (c + 1):m
                Vp[r, c] = A[rs + r - 1, g]   # essential v_g = A[g+2:n, g]
            end
        end
        Tv = view(Tm, 1:pb, 1:pb)
        wy_t!(Tv, Vp, view(tau, pc:(pc + pb - 1)), ws.G)
        wy_apply!(trans, view(C, rs:n, 1:nc), Vp, Tv, ws)
    end
    return C
end

# ── Complex Hermitian tridiagonalization TAIL/base (LAPACK zhetd2, LOWER) — Hermitian analogue of
# _sytd2_lower!; the unblocked base case the blocked `_hetrd!` driver below finishes with (and the whole
# reduction for n below the blocking crossover). Reduces the lower triangle of Hermitian A to real
# tridiagonal T = QᴴAQ: d=diagonal (real), e=subdiag (real — β from complex _larfg! is real by the zlarfg
# phase convention), tau=complex reflectors. Essential v_i in A[i+2:n,i] (v_i[1]≡1 implicit); trailing
# matvec is hemv! (Hermitian). Half-correction uses the CONJUGATE dot wᴴv. Trailing downdate A -= v·wᴴ +
# w·vᴴ (her2), diagonal re-realified. The m==1 (last column) trailing downdate is an EXACT no-op
# (|τ|²=2Re(τ) by unitarity) — skipped; tau[n-1] itself is NOT dropped (it is used by _unmtr!).
function _hetd2_lower!(
        A::AbstractMatrix{T}, d::AbstractVector{R}, e::AbstractVector{R},
        tau::AbstractVector{T}
    ) where {T <: Complex, R <: Real}
    n = size(A, 1)
    n == 0 && return
    v = Vector{T}(undef, n)
    w = Vector{T}(undef, n)
    @inbounds for i in 1:(n - 1)
        m = n - i
        col = view(A, (i + 1):n, i)
        β, τ = _larfg!(col)                       # β real (zlarfg phase convention), τ complex
        e[i] = β; tau[i] = τ; d[i] = real(A[i, i])
        if τ != zero(T) && m > 1
            v[1] = one(T)
            for r in 2:m
                v[r] = A[i + r, i]
            end
            Atr = view(A, (i + 1):n, (i + 1):n); vv = view(v, 1:m); ww = view(w, 1:m)
            hemv!(ww, Atr, vv; uplo = 'L', alpha = τ, beta = zero(T))   # w = τ·Atr·v (Hermitian)
            s = zero(T)
            for r in 1:m
                s += conj(ww[r]) * vv[r]
            end                # wᴴv
            α = -(τ / 2) * s
            for r in 1:m
                ww[r] = muladd(α, vv[r], ww[r])
            end
            for jc in 1:m, ir in jc:m                                  # Hermitian rank-2 downdate (lower)
                upd = vv[ir] * conj(ww[jc]) + ww[ir] * conj(vv[jc])
                if ir == jc
                    Atr[ir, jc] = Complex(real(Atr[ir, jc] - upd), zero(R))  # re-realify diagonal
                else
                    Atr[ir, jc] -= upd
                end
            end
        end
    end
    d[n] = real(A[n, n])
    return
end

# ── Complex BLOCKED tridiagonalization (LAPACK zlatrd panel + zher2k trailing), LOWER ───────────────────
# Hermitian analogue of `_latrd_lower!`: hemv (not symv), Aᴴ cross-terms (gemv 'C'), conjugated column-
# update multipliers, dotc half-correction, diagonal re-realified each step. Fills real d/e, complex tau,
# and W (ns×nb) for the caller's her2k trailing update. τ scaling + half-correction applied last (raw A·v
# feeds the cross-term gemvs), mirroring zlatrd exactly.
function _latrd_lower!(
        As::AbstractMatrix{T}, W::AbstractMatrix{T}, d::AbstractVector{R},
        e::AbstractVector{R}, tau::AbstractVector{T}, nb::Int, tmp::AbstractVector{T}
    ) where {T <: Complex, R <: Real}
    ns = size(As, 1)
    @inbounds for i in 1:nb
        if i > 1
            As[i, i] = Complex(real(As[i, i]), zero(R))                    # realify diagonal (zlatrd)
            for t in 1:(i - 1)
                tmp[t] = conj(W[i, t])
            end
            _egc!(view(As, i:ns, i), view(As, i:ns, 1:(i - 1)), view(tmp, 1:(i - 1)), -one(T), one(T), false, false)
            for t in 1:(i - 1)
                tmp[t] = conj(As[i, t])
            end
            _egc!(view(As, i:ns, i), view(W, i:ns, 1:(i - 1)), view(tmp, 1:(i - 1)), -one(T), one(T), false, false)
            As[i, i] = Complex(real(As[i, i]), zero(R))
        end
        d[i] = real(As[i, i])
        if i < ns
            m = ns - i
            col = view(As, (i + 1):ns, i)
            β, τ = _larfg!(col)
            e[i] = β; tau[i] = τ; As[i + 1, i] = one(T)
            v = view(As, (i + 1):ns, i)
            wc = view(W, (i + 1):ns, i)
            hemv!(wc, view(As, (i + 1):ns, (i + 1):ns), v; uplo = 'L', alpha = one(T), beta = zero(T))  # wc = A·v
            if i > 1
                wtop = view(W, 1:(i - 1), i)
                _egc!(wtop, view(W, (i + 1):ns, 1:(i - 1)), v, one(T), zero(T), true, true)     # wtop = W2ᴴ·v
                _egc!(wc, view(As, (i + 1):ns, 1:(i - 1)), wtop, -one(T), one(T), false, false) # wc −= A21·wtop
                _egc!(wtop, view(As, (i + 1):ns, 1:(i - 1)), v, one(T), zero(T), true, true)    # wtop = A21ᴴ·v
                _egc!(wc, view(W, (i + 1):ns, 1:(i - 1)), wtop, -one(T), one(T), false, false)  # wc −= W2·wtop
            end
            for r in 1:m
                wc[r] *= τ
            end
            s = zero(T)
            for r in 1:m
                s += conj(wc[r]) * v[r]                                    # wᴴ·v (dotc)
            end
            c = -(τ / 2) * s
            for r in 1:m
                wc[r] = muladd(c, v[r], wc[r])
            end
        end
    end
    return
end

# Blocked driver (LAPACK zhetrd, lower): zlatrd panel + rank-2·nb her2k, unblocked `_hetd2_lower!` tail.
# nb/nx derived exactly like the real `_sytrd_lower!`.
function _hetrd!(
        A::AbstractMatrix{T}, d::AbstractVector{R},
        e::AbstractVector{R}, tau::AbstractVector{T}
    ) where {T <: Complex, R <: Real}
    n = size(A, 1)
    n == 0 && return
    nb = _sytrd_nb(n)
    nx = 2 * nb
    if n <= nx || nb <= 1
        _hetd2_lower!(A, d, e, tau)
        return
    end
    W = Matrix{T}(undef, n, nb)
    tmp = Vector{T}(undef, nb)
    kk = 1
    @inbounds while n - kk + 1 > nx
        pb = min(nb, n - kk)
        ns = n - kk + 1
        As = view(A, kk:n, kk:n)
        Wp = view(W, 1:ns, 1:pb)
        _latrd_lower!(As, Wp, view(d, kk:(kk + pb - 1)), view(e, kk:(kk + pb - 1)), view(tau, kk:(kk + pb - 1)), pb, tmp)
        rs = kk + pb
        V2 = view(A, rs:n, kk:(kk + pb - 1))
        W2 = view(W, (pb + 1):ns, 1:pb)
        her2k!(view(A, rs:n, rs:n), V2, W2; uplo = 'L', trans = 'N', alpha = -one(T), beta = one(R))
        kk += pb
    end
    _hetd2_lower!(view(A, kk:n, kk:n), view(d, kk:n), view(e, kk:(n - 1)), view(tau, kk:(n - 1)))
    return
end

# ── Apply Q from _hetrd! (LAPACK zunmtr, side='L', LOWER) to C ────────────────────────────────────────
# Q = H_1·H_2·⋯·H_{n-1}, H_i = I − τ_i·v_i·v_iᴴ acts on rows i+1:n.  *** Uses ALL n-1 reflectors ***
# — unlike REAL _ormtr! (which safely skips i=n-1 because real _larfg! HARDCODES τ=0 for a length-1
# vector). Complex _larfg! has no such hardcode: tau[n-1] is GENERICALLY nonzero for Hermitian data and
# its reflector is NOT a no-op on other columns of C. Dropping it gives O(1) reconstruction error.
#   trans='N':  C := Q·C  = H_1·(⋯·(H_{n-1}·C))    → apply i = n-1 … 1  (decreasing)
#   trans='C':  C := Qᴴ·C = H_{n-1}ᴴ·(⋯·(H_1ᴴ·C))  → apply i = 1 … n-1  (increasing), τ → conj(τ)
# Q = H_1·H_2·⋯·H_{n-1}, H_i = I − τ_i·v_i·v_iᴴ acts on rows i+1:n.  *** Uses ALL n-1 reflectors ***
# — unlike REAL _ormtr! (which safely skips i=n-1 because real _larfg! HARDCODES τ=0 for a length-1
# vector). Complex _larfg! has no such hardcode: tau[n-1] is GENERICALLY nonzero for Hermitian data and
# its reflector is NOT a no-op on other columns of C. Dropping it gives O(1) reconstruction error.
#   trans='N':  C := Q·C  = H_1·(⋯·(H_{n-1}·C))    → apply i = n-1 … 1  (decreasing)
#   trans='C':  C := Qᴴ·C = H_{n-1}ᴴ·(⋯·(H_1ᴴ·C))  → apply i = 1 … n-1  (increasing), τ → conj(τ)
# Complex compact-WY T factor (zlarft forward): Q = I − V·T·Vᴴ. G = Vᴴ·V via herk (upper), then the
# same triangular recurrence as `wy_t!` but over the Hermitian Gram. `Tm` written full (lower zeroed).
function _wy_t_cplx!(
        Tm::AbstractMatrix{T}, Vp::AbstractMatrix{T}, tau::AbstractVector{T}, G::AbstractMatrix{T}
    ) where {T <: Complex}
    bs = length(tau)
    bs == 0 && return Tm
    m = size(Vp, 1)
    Gv = view(G, 1:bs, 1:bs)
    herk!(Gv, view(Vp, 1:m, 1:bs); uplo = 'U', trans = 'C', alpha = true, beta = false)  # G = VᴴV, upper
    @inbounds for c in 1:bs
        tc = tau[c]
        Tm[c, c] = tc
        for r in 1:(c - 1)
            s = zero(T)
            for kk in r:(c - 1)
                s = muladd(Tm[r, kk], Gv[kk, c], s)
            end
            Tm[r, c] = -tc * s
        end
        for r in (c + 1):bs
            Tm[r, c] = zero(T)
        end
    end
    return Tm
end
# Complex block-reflector apply (zlarfb): C := Q·C (trans='N') or Qᴴ·C (trans='C'), Q = I − V·Tm·Vᴴ.
@inline function _wy_apply_cplx!(
        trans::Char, C::AbstractMatrix{T}, Vp::AbstractMatrix{T},
        Tm::AbstractMatrix{T}, ws::WYApplyWorkspace{T}
    ) where {T <: Complex}
    m = size(Vp, 1); bs = size(Vp, 2); nc = size(C, 2)
    (bs == 0 || nc == 0 || m == 0) && return C
    Wv = view(ws.W, 1:bs, 1:nc)
    gemm!(Wv, Vp, C; transA = 'C', alpha = true, beta = false)                # W = Vᴴ·C
    trmm!(Wv, view(Tm, 1:bs, 1:bs); side = 'L', uplo = 'U', transA = trans)   # W := (T or Tᴴ)·W
    gemm!(C, Vp, Wv; alpha = -one(T), beta = one(T))                          # C −= V·W
    return C
end

# BLOCKED zunmtr (side='L', lower): reflectors H_1..H_{n-1} in nb-wide compact-WY blocks (`_wy_t_cplx!`/
# `_wy_apply_cplx!`) → Level-3, mirroring the real `_ormtr!`. Panel for columns [pc:pc+pb-1] occupies rows
# [pc+1:n] as a unit-lower-trapezoid. Reverse block order for Q·C, forward for Qᴴ·C.
function _unmtr!(
        A::AbstractMatrix{T}, tau::AbstractVector{T},
        C::AbstractMatrix{T}; side::Char = 'L', trans::Char = 'N'
    ) where {T <: Complex}
    side === 'L' || throw(ArgumentError("_unmtr!: only side='L' implemented"))
    trans === 'N' || trans === 'C' || throw(ArgumentError("_unmtr!: trans must be 'N' or 'C'"))
    n = size(A, 1); nc = size(C, 2)
    (n <= 1 || nc == 0) && return C           # no reflectors (H_1..H_{n-1} needs n≥2)
    k = n - 1                                  # reflector count (complex: incl. the nontrivial H_{n-1})
    nb = clamp(_qr_nb(n, nc), 1, k)
    ws = WYApplyWorkspace{T}(n, nb, nc)
    Tm = Matrix{T}(undef, nb, nb); G = Matrix{T}(undef, nb, nb)
    nblk = cld(k, nb)
    order = trans === 'N' ? (nblk:-1:1) : (1:nblk)
    @inbounds for b in order
        pc = (b - 1) * nb + 1
        pb = min(nb, k - pc + 1)
        rs = pc + 1
        m = n - rs + 1
        Vp = view(ws.V, 1:m, 1:pb)
        for c in 1:pb
            g = pc + c - 1
            for r in 1:(c - 1)
                Vp[r, c] = zero(T)
            end
            Vp[c, c] = one(T)
            for r in (c + 1):m
                Vp[r, c] = A[rs + r - 1, g]
            end
        end
        Tv = view(Tm, 1:pb, 1:pb)
        _wy_t_cplx!(Tv, Vp, view(tau, pc:(pc + pb - 1)), G)
        _wy_apply_cplx!(trans, view(C, rs:n, 1:nc), Vp, Tv, ws)
    end
    return C
end

# ── orgtr / ungtr — form Q from sytrd/hetrd reflectors (LOWER only) ────────────────────────────────────
# LAPACK dorgtr/zungtr forms the SAME Q that `_ormtr!`/`_unmtr!` (side='L', trans='N') apply as the
# eigenvector back-transform — so Q = _ormtr!(A, tau, I) reuses that already-validated kernel verbatim
# (no new reflector-forming code): starting C from the identity and running the proven trans='N' sweep
# builds exactly Q = H_1·H_2·⋯. Only uplo='L' is implemented (mirrors `_sytd2_lower!`/`_hetrd!`'s own
# restriction) — uplo='U' is a documented follow-up (a genuinely different reflector layout, not a
# transpose of this one); callers get a clean ArgumentError rather than a silently-wrong Q.
"""
    orgtr!(uplo, A, tau) -> Q    (real)
    ungtr!(uplo, A, tau) -> Q    (complex)

Form the orthogonal/unitary `Q` from [`_sytd2_lower!`](@ref)/[`_hetrd!`](@ref)'s reflectors (LAPACK
dorgtr/zungtr). Only `uplo='L'` is implemented. Returns a fresh `n×n` `Q` (does not overwrite `A`).
"""
function orgtr!(uplo::Char, A::AbstractMatrix{T}, tau::AbstractVector{T}) where {T <: Real}
    uplo === 'L' || throw(ArgumentError("orgtr!: only uplo='L' is implemented"))
    n = size(A, 1)
    Q = zeros(T, n, n)
    @inbounds for i in 1:n
        Q[i, i] = one(T)
    end
    _ormtr!(A, tau, Q; side = 'L', trans = 'N')
    return Q
end
function ungtr!(uplo::Char, A::AbstractMatrix{T}, tau::AbstractVector{T}) where {T <: Complex}
    uplo === 'L' || throw(ArgumentError("ungtr!: only uplo='L' is implemented"))
    n = size(A, 1)
    Q = zeros(T, n, n)
    @inbounds for i in 1:n
        Q[i, i] = one(T)
    end
    _unmtr!(A, tau, Q; side = 'L', trans = 'N')
    return Q
end

# ── Values-only tridiagonal eigensolver (LAPACK dsterf) — O(n²) root-free PWK QL/QR, no vectors ───────
# jobz='N' fast path for _syev!/_heev!. d (diag) → eigenvalues ascending in place; e (subdiag) destroyed.
# Returns info (0 = converged; >0 = # off-diagonals that failed to deflate within 30n iterations).
@inline function _sterf_lae2(a::T, b::T, c::T) where {T <: Real}
    sm = a + c; df = a - c; adf = abs(df); tb = b + b; ab = abs(tb)
    acmx = abs(a) > abs(c) ? a : c; acmn = abs(a) > abs(c) ? c : a
    rt = if adf > ab
        adf * sqrt(one(T) + (ab / adf)^2)
    elseif adf < ab
        ab * sqrt(one(T) + (adf / ab)^2)
    else
        ab * sqrt(T(2))
    end
    if sm < zero(T)
        rt1 = (sm - rt) / 2
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b
    elseif sm > zero(T)
        rt1 = (sm + rt) / 2
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b
    else
        rt1 = rt / 2; rt2 = -rt / 2
    end
    return rt1, rt2
end

function _sterf!(d::AbstractVector{T}, e::AbstractVector{T}) where {T <: Real}
    n = length(d)
    n <= 1 && return 0
    @assert length(e) >= n - 1 "_sterf!: e must have length >= n-1"

    eps_ = eps(T) / 2                     # dlamch('E'): unit roundoff (½ ulp)
    eps2 = eps_ * eps_
    safmin = floatmin(T)
    safmax = one(T) / safmin
    ssfmax = sqrt(safmax) / 3
    ssfmin = sqrt(safmin) / eps2
    MAXIT = 30
    nmaxit = n * MAXIT
    jtot = 0
    nm1 = n - 1

    l1 = 1
    @inbounds while true
        l1 > n && break
        l1 > 1 && (e[l1 - 1] = zero(T))

        # ------ find the next unreduced block [l:lend] via the split criterion --------------
        m = n
        if l1 <= nm1
            for mm in l1:nm1
                if abs(e[mm]) <= sqrt(abs(d[mm])) * sqrt(abs(d[mm + 1])) * eps_
                    e[mm] = zero(T); m = mm; break
                end
            end
        end
        l = l1; lsv = l; lend = m; lendsv = lend; l1 = m + 1
        lend == l && continue                          # 1x1 block: already an eigenvalue

        # ------ scale the block into [ssfmin, ssfmax] ---------------------------------------
        anorm = abs(d[lend])
        for i in l:(lend - 1)
            anorm = max(anorm, abs(d[i]), abs(e[i]))
        end
        anorm == zero(T) && continue
        iscale = 0
        if anorm > ssfmax
            iscale = 1; f = ssfmax / anorm
            for i in l:lend
                d[i] *= f
            end
            for i in l:(lend - 1)
                e[i] *= f
            end
        elseif anorm < ssfmin
            iscale = 2; f = ssfmin / anorm
            for i in l:lend
                d[i] *= f
            end
            for i in l:(lend - 1)
                e[i] *= f
            end
        end
        for i in l:(lend - 1)
            e[i] = e[i] * e[i]
        end     # work in squared off-diagonals (root-free)

        # ------ pick QL (bottom-up) or QR (top-down): iterate toward the smaller-|d| end ----
        if abs(d[lend]) < abs(d[l])
            lend = lsv; l = lendsv
        end

        converged = true
        if lend >= l
            # =================== QL iteration (root-free PWK recurrence) ====================
            while true
                m = lend
                if l != lend
                    for mm in l:(lend - 1)
                        if abs(e[mm]) <= eps2 * abs(d[mm] * d[mm + 1])
                            m = mm; break
                        end
                    end
                end
                m < lend && (e[m] = zero(T))
                p = d[l]
                if m == l
                    d[l] = p; l += 1
                    (l <= lend) ? continue : break
                end
                if m == l + 1
                    rte = sqrt(e[l])
                    rt1, rt2 = _sterf_lae2(d[l], rte, d[l + 1])
                    d[l] = rt1; d[l + 1] = rt2; e[l] = zero(T)
                    l += 2
                    (l <= lend) ? continue : break
                end
                if jtot == nmaxit
                    converged = false; break
                end
                jtot += 1
                rte = sqrt(e[l])
                sigma = (d[l + 1] - p) / (2 * rte)
                r = hypot(sigma, one(T))
                sigma = p - rte / (sigma + copysign(r, sigma))
                c = one(T); s = zero(T)
                gamma = d[m] - sigma
                pp = gamma * gamma
                for i in (m - 1):-1:l
                    bb = e[i]
                    rr = pp + bb
                    i != m - 1 && (e[i + 1] = s * rr)
                    oldc = c
                    c = pp / rr; s = bb / rr
                    oldgam = gamma
                    alpha = d[i]
                    gamma = c * (alpha - sigma) - s * oldgam
                    d[i + 1] = oldgam + (alpha - gamma)
                    pp = c != zero(T) ? (gamma * gamma) / c : oldc * bb
                end
                e[l] = s * pp
                d[l] = sigma + gamma
            end
        else
            # =================== QR iteration (root-free PWK recurrence) ====================
            while true
                m = lend
                for mm in l:-1:(lend + 1)
                    if abs(e[mm - 1]) <= eps2 * abs(d[mm] * d[mm - 1])
                        m = mm; break
                    end
                end
                m > lend && (e[m - 1] = zero(T))
                p = d[l]
                if m == l
                    d[l] = p; l -= 1
                    (l >= lend) ? continue : break
                end
                if m == l - 1
                    rte = sqrt(e[l - 1])
                    rt1, rt2 = _sterf_lae2(d[l], rte, d[l - 1])
                    d[l] = rt1; d[l - 1] = rt2; e[l - 1] = zero(T)
                    l -= 2
                    (l >= lend) ? continue : break
                end
                if jtot == nmaxit
                    converged = false; break
                end
                jtot += 1
                rte = sqrt(e[l - 1])
                sigma = (d[l - 1] - p) / (2 * rte)
                r = hypot(sigma, one(T))
                sigma = p - rte / (sigma + copysign(r, sigma))
                c = one(T); s = zero(T)
                gamma = d[m] - sigma
                pp = gamma * gamma
                for i in m:(l - 1)
                    bb = e[i]
                    rr = pp + bb
                    i != m && (e[i - 1] = s * rr)
                    oldc = c
                    c = pp / rr; s = bb / rr
                    oldgam = gamma
                    alpha = d[i + 1]
                    gamma = c * (alpha - sigma) - s * oldgam
                    d[i] = oldgam + (alpha - gamma)
                    pp = c != zero(T) ? (gamma * gamma) / c : oldc * bb
                end
                e[l - 1] = s * pp
                d[l] = sigma + gamma
            end
        end

        # ------ undo scaling for this block --------------------------------------------------
        if iscale == 1
            f = anorm / ssfmax
            for i in lsv:lendsv
                d[i] *= f
            end
        elseif iscale == 2
            f = anorm / ssfmin
            for i in lsv:lendsv
                d[i] *= f
            end
        end

        if !converged
            info = 0
            for i in 1:nm1
                e[i] != zero(T) && (info += 1)
            end
            return info
        end
    end

    sort!(view(d, 1:n))
    return 0
end

# ── Engine: _syev!(jobz, uplo, A) → (w, Z, info). REAL Float64 symmetric eigensolver. ──────────────
# jobz='V' returns eigenvectors in Z (columns, ascending eigenvalue order); 'N' returns empty Z.
# A is destroyed (reduction workspace). uplo='U' is handled by mirroring the upper triangle into the
# lower and running the LOWER path (A symmetric ⇒ A_L[i,j]=A[j,i], i>j). anrm pre-scaling per dsyev.
function _syev!(jobz::Char, uplo::Char, A::AbstractMatrix{T}) where {T <: Real}
    n = size(A, 1)
    if n == 0
        return T[], Matrix{T}(undef, 0, 0), 0
    end
    wantz = jobz == 'V'
    if uplo == 'U'                                    # mirror upper → lower, then run the LOWER path
        @inbounds for j in 1:n, i in (j + 1):n
            A[i, j] = A[j, i]
        end
    end

    # anrm pre-scaling (dsyev): pull |A| into [rmin, rmax] for denormal/huge-norm safety; unscale w after.
    anrm = zero(T)
    @inbounds for j in 1:n, i in j:n
        anrm = max(anrm, abs(A[i, j]))                # max abs of the lower triangle
    end
    rmin = sqrt(floatmin(T)) / eps(T)                 # algorithm constants (req8-ok: LAPACK dsyev scaling band)
    rmax = one(T) / rmin
    sigma = one(T)
    if anrm > zero(T) && anrm < rmin
        sigma = rmin / anrm
    elseif anrm > rmax
        sigma = rmax / anrm
    end
    if sigma != one(T)
        @inbounds for j in 1:n, i in j:n
            A[i, j] *= sigma
        end
    end

    d = Vector{T}(undef, n)
    e = Vector{T}(undef, max(n - 1, 1))
    tau = Vector{T}(undef, max(n - 1, 1))
    _sytrd_lower!(A, d, e, tau)                       # blocked tridiagonalization (dlatrd + syr2k)

    info = 0
    if wantz
        Z = zeros(T, n, n)                            # identity start (stedc base fills it too, but n==1 skips)
        @inbounds for i in 1:n
            Z[i, i] = one(T)
        end
        _stedc!(d, e, Z)                              # divide-and-conquer eigenvectors of T (ascending)
        _ormtr!(A, tau, Z; side = 'L', trans = 'N')  # V = Q·Z_T = eigenvectors of A (already ordered)
    else
        Z = Matrix{T}(undef, 0, 0)
        info = _sterf!(d, e)                          # O(n²) values-only (dsterf); info>0 = non-convergence
    end

    if sigma != one(T)                                # unscale eigenvalues
        invσ = one(T) / sigma
        @inbounds for i in 1:n
            d[i] *= invσ
        end
    end
    return d, Z, info
end

# ── Engine: _heev!(jobz, uplo, A) → (w, Z, info). COMPLEX Hermitian eigensolver (native). ──────────────
# Mirrors _syev! but Hermitian: _hetrd! (real d,e; complex tau) → REAL _stedc!/_sterf! on the tridiagonal
# → back-transform. Eigenvectors: real Z from _stedc! is embedded into a complex matrix, then rotated by
# Q via _unmtr! (ALL n-1 reflectors). anrm pre-scaling on the complex magnitudes (unscale w after).
function _heev!(jobz::Char, uplo::Char, A::AbstractMatrix{T}) where {T <: Complex}
    R = real(T)
    n = size(A, 1)
    if n == 0
        return R[], Matrix{T}(undef, 0, 0), 0
    end
    wantz = jobz == 'V'
    if uplo == 'U'                                    # mirror upper → lower (Hermitian: A_L[i,j]=conj(A[j,i]))
        @inbounds for j in 1:n, i in (j + 1):n
            A[i, j] = conj(A[j, i])
        end
    end

    anrm = zero(R)
    @inbounds for j in 1:n, i in j:n
        anrm = max(anrm, abs(A[i, j]))                # max magnitude of the lower triangle
    end
    rmin = sqrt(floatmin(R)) / eps(R)
    rmax = one(R) / rmin
    sigma = one(R)
    if anrm > zero(R) && anrm < rmin
        sigma = rmin / anrm
    elseif anrm > rmax
        sigma = rmax / anrm
    end
    if sigma != one(R)
        @inbounds for j in 1:n, i in j:n
            A[i, j] *= sigma
        end
    end

    d = Vector{R}(undef, n)
    e = Vector{R}(undef, max(n - 1, 1))
    tau = Vector{T}(undef, max(n - 1, 1))
    _hetrd!(A, d, e, tau)

    info = 0
    if wantz
        Zr = zeros(R, n, n)                           # real eigenvectors of the tridiagonal T
        @inbounds for i in 1:n
            Zr[i, i] = one(R)
        end
        _stedc!(d, e, Zr)
        Z = Matrix{T}(undef, n, n)
        @inbounds for j in 1:n, i in 1:n
            Z[i, j] = T(Zr[i, j])
        end
        _unmtr!(A, tau, Z; side = 'L', trans = 'N')  # V = Q·Z_T = eigenvectors of A (already ordered)
    else
        Z = Matrix{T}(undef, 0, 0)
        info = _sterf!(d, e)                          # info>0 = non-convergence
    end

    if sigma != one(R)                                # unscale eigenvalues
        invσ = one(R) / sigma
        @inbounds for i in 1:n
            d[i] *= invσ
        end
    end
    return d, Z, info
end
