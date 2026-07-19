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
function _sytd2_lower!(A::AbstractMatrix{Float64}, d::AbstractVector{Float64},
        e::AbstractVector{Float64}, tau::AbstractVector{Float64})
    n = size(A, 1)
    n == 0 && return
    v = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)
    @inbounds for i in 1:n-1
        m = n - i
        col = view(A, i+1:n, i)
        β, τ = _larfg!(col)                       # essential v now in A[i+2:n,i]; col[1]=A[i+1,i] left as-is
        e[i] = β; tau[i] = τ; d[i] = A[i, i]
        if τ != 0.0 && m > 1
            v[1] = 1.0
            for r in 2:m; v[r] = A[i+r, i]; end
            Atr = view(A, i+1:n, i+1:n); vv = view(v, 1:m); ww = view(w, 1:m)
            symv!(ww, Atr, vv; uplo = 'L', alpha = τ, beta = 0.0)   # w = τ·A_trailing·v (lower symmetric)
            s = 0.0
            for r in 1:m; s = muladd(ww[r], vv[r], s); end
            c = -(τ / 2) * s                       # dsytd2 half-correction
            for r in 1:m; ww[r] = muladd(c, vv[r], ww[r]); end
            for jc in 1:m, ir in jc:m              # symmetric rank-2 downdate (lower triangle)
                Atr[ir, jc] -= vv[ir] * ww[jc] + ww[ir] * vv[jc]
            end
        end
    end
    d[n] = A[n, n]
    return
end

# ── Stage 2: 2×2 symmetric eigensolvers (LAPACK dlae2 values / dlaev2 values+vector) ───────────────
@inline function _dlae2(a::T, b::T, c::T) where {T<:Real}
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

@inline function _dlaev2(a::T, b::T, c::T) where {T<:Real}
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
function _steqr!(compz::Char, d::AbstractVector{T}, e::AbstractVector{T},
        Z::Union{AbstractMatrix{T},Nothing}) where {T<:Real}
    n = length(d)
    wantz = compz == 'V' || compz == 'I'
    n == 0 && return d
    if n == 1
        return d
    end
    if compz == 'I' && !isnothing(Z)
        fill!(Z, zero(T))
        @inbounds for i in 1:n; Z[i, i] = one(T); end
    end

    eps_  = eps(T) / 2                    # dlamch('E') = relative machine precision (½ ulp)
    eps2  = eps_ * eps_
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
        for i in l:lend; anorm = max(anorm, abs(d[i])); end
        for i in l:lend-1; anorm = max(anorm, abs(e[i])); end
        anorm == zero(T) && continue
        iscale = 0
        if anorm > ssfmax
            iscale = 1; f = ssfmax / anorm
            for i in l:lend; d[i] *= f; end
            for i in l:lend-1; e[i] *= f; end
        elseif anorm < ssfmin
            iscale = 2; f = ssfmin / anorm
            for i in l:lend; d[i] *= f; end
            for i in l:lend-1; e[i] *= f; end
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
                    for mm in l:lend-1
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
                    for mm in l:-1:lend+1
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
            for i in lsv:lendsv; d[i] *= f; end
            for i in lsv:lendsv-1; e[i] *= f; end
        elseif iscale == 2
            f = anorm / ssfmin
            for i in lsv:lendsv; d[i] *= f; end
            for i in lsv:lendsv-1; e[i] *= f; end
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
function _ormtr!(A::AbstractMatrix{Float64}, tau::AbstractVector{Float64},
        C::AbstractMatrix{Float64}; side::Char = 'L', trans::Char = 'N')
    side === 'L' || throw(ArgumentError("_ormtr!: only side='L' implemented"))
    trans === 'N' || trans === 'T' || throw(ArgumentError("_ormtr!: trans must be 'N' or 'T'"))
    n = size(A, 1)
    n <= 2 && return C                       # no reflectors (H_1..H_{n-2})
    v = Vector{Float64}(undef, n)
    order = trans === 'N' ? ((n-2):-1:1) : (1:(n-2))
    @inbounds for i in order
        m = n - i                            # length of v_i (rows i+1:n)
        τ = tau[i]
        v[1] = 1.0
        for r in 2:m
            v[r] = A[i+r, i]                 # essential part = A[i+2:n, i]
        end
        _house_left!(view(C, (i+1):n, :), view(v, 1:m), τ)
    end
    return C
end

# ── Engine: _syev!(jobz, uplo, A) → (w, Z, info). REAL Float64 symmetric eigensolver. ──────────────
# jobz='V' returns eigenvectors in Z (columns, ascending eigenvalue order); 'N' returns empty Z.
# A is destroyed (reduction workspace). uplo='U' is handled by mirroring the upper triangle into the
# lower and running the LOWER path (A symmetric ⇒ A_L[i,j]=A[j,i], i>j). anrm pre-scaling per dsyev.
function _syev!(jobz::Char, uplo::Char, A::AbstractMatrix{Float64})
    n = size(A, 1)
    if n == 0
        return Float64[], Matrix{Float64}(undef, 0, 0), 0
    end
    wantz = jobz == 'V'
    if uplo == 'U'                                    # mirror upper → lower, then run the LOWER path
        @inbounds for j in 1:n, i in j+1:n
            A[i, j] = A[j, i]
        end
    end

    # anrm pre-scaling (dsyev): pull |A| into [rmin, rmax] for denormal/huge-norm safety; unscale w after.
    anrm = 0.0
    @inbounds for j in 1:n, i in j:n
        anrm = max(anrm, abs(A[i, j]))                # max abs of the lower triangle
    end
    rmin = sqrt(floatmin(Float64)) / eps(Float64)     # algorithm constants (req8-ok: LAPACK dsyev scaling band)
    rmax = 1.0 / rmin
    sigma = 1.0
    if anrm > 0.0 && anrm < rmin
        sigma = rmin / anrm
    elseif anrm > rmax
        sigma = rmax / anrm
    end
    if sigma != 1.0
        @inbounds for j in 1:n, i in j:n; A[i, j] *= sigma; end
    end

    d = Vector{Float64}(undef, n)
    e = Vector{Float64}(undef, max(n - 1, 1))
    tau = Vector{Float64}(undef, max(n - 1, 1))
    _sytd2_lower!(A, d, e, tau)

    if wantz
        Z = zeros(Float64, n, n)                      # identity start (steqr compz='I' also fills it, but n==1 skips)
        @inbounds for i in 1:n; Z[i, i] = 1.0; end
        _steqr!('I', d, e, Z)                         # eigenvectors of T (ascending), Z columns permuted
        _ormtr!(A, tau, Z; side = 'L', trans = 'N')   # V = Q·Z_T = eigenvectors of A (already ordered)
    else
        Z = Matrix{Float64}(undef, 0, 0)
        _steqr!('N', d, e, nothing)
    end

    if sigma != 1.0                                   # unscale eigenvalues
        invσ = 1.0 / sigma
        @inbounds for i in 1:n; d[i] *= invσ; end
    end
    return d, Z, 0
end
