# LAPACK SVD — divide-and-conquer bidiagonal solver (bdsdc), Float64. Faithful port of faer 0.24.x
# `faer/src/linalg/svd/bidiag_svd.rs` (Gu-Eisenstat D&C) onto plain Julia + PureBLAS gemm! for the
# vector-combine matmuls. Operates on a LOWER bidiagonal L (diag + subdiag) with an augmented
# (n+1)×(n+1) U; returns L = U[1:n,1:n]·diag(s)·Vᵀ. Drives the singular-VECTORS path of gesvd!
# (the values-only path keeps the cheaper bdsqr!). faer's 0-based index arithmetic is kept verbatim;
# arrays are accessed with a +1 shift. ponytail: Full-U mode only (we always want both U and V).

# Preallocated D&C workspace — allocated ONCE in bdsdc! and threaded through every recursion node.
# The serial post-order traversal means only one node's merge is live at a time, so a single set of
# top-sized scratch serves all nodes ⇒ the recursion is allocation-free (cf. gemm's _gemm_scratch,
# getrf's _LU_PAD). bufA/bufB are (N+1)², bufC/bufD are N²; the rest are length-N vectors.
struct _DCWork
    jc::Vector{Float64}; js::Vector{Float64}
    jidx::Vector{Int}; transp::Vector{Int}; perm::Vector{Int}
    real_ind::Vector{Int}; real_col::Vector{Int}; permloc::Vector{Int}
    col_perm::Vector{Int}; col_perm_inv::Vector{Int}
    shifts::Vector{Float64}; mus::Vector{Float64}; s::Vector{Float64}; zhat::Vector{Float64}
    col0p::Vector{Float64}; diagp::Vector{Float64}
    zhp::Vector{Float64}; rowidx::Vector{Int}; vbuf::Vector{Float64}   # permuted zhat / target rows / contiguous vec vals
    bufA::Matrix{Float64}; bufB::Matrix{Float64}; bufC::Matrix{Float64}; bufD::Matrix{Float64}
end
function _dcwork(n::Int)
    fv() = Vector{Float64}(undef, n); iv() = Vector{Int}(undef, n)
    _DCWork(fv(), fv(), iv(), iv(), iv(), iv(), iv(), iv(), iv(), iv(),
        fv(), fv(), fv(), fv(), fv(), fv(),
        fv(), iv(), fv(),
        Matrix{Float64}(undef, n+1, n+1), Matrix{Float64}(undef, n+1, n+1),
        Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n, n))
end
# Persistent workspace cached across calls (cf. getrf's _LU_PAD) — a workspace sized for the largest n
# seen serves all smaller n (views), so repeated SVDs re-allocate nothing. Single-thread (no MT yet).
const _DC_WS = Ref(_dcwork(1))
const _DC_WS_N = Ref(1)
@inline function _get_dcwork(n::Int)
    if _DC_WS_N[] < n
        _DC_WS[] = _dcwork(n); _DC_WS_N[] = n
    end
    return _DC_WS[]
end

# --- Jacobi rotation (faer JacobiRotation, real arithmetic) -------------------------------------
# make_givens(p,q) → (c,s): the rotation that maps (p,q) with  c·p − s·q = r,  s·p + c·q = 0.
@inline function _mkgivens(p::Float64, q::Float64)
    if q == 0.0
        return (p < 0.0 ? -1.0 : 1.0), 0.0
    elseif p == 0.0
        return 0.0, (q < 0.0 ? 1.0 : -1.0)
    elseif abs(p) > abs(q)
        t = q / p; u = sqrt(muladd(t, t, 1.0)); p < 0.0 && (u = -u)   # |t|<1 ⇒ no overflow, skip Base.hypot
        c = 1.0 / u; return c, -t * c
    else
        t = p / q; u = sqrt(muladd(t, t, 1.0)); q < 0.0 && (u = -u)   # |t|≤1 ⇒ safe
        s = -1.0 / u; return -t * s, s
    end
end

# apply_on_the_left to rows (ra,rb) [0-based]:  x' = c·x − s·y,  y' = s·x + c·y.
@inline function _jac_left!(M::AbstractMatrix{Float64}, ra::Int, rb::Int, c::Float64, s::Float64)
    (c == 1.0 && s == 0.0) && return
    a0 = ra + 1; b0 = rb + 1
    @inbounds for j in 1:size(M, 2)
        a = M[a0, j]; b = M[b0, j]
        M[a0, j] = c * a - s * b
        M[b0, j] = s * a + c * b
    end
end

# apply_on_the_right to columns (ca,cb) [0-based]:  x' = c·x + s·y,  y' = −s·x + c·y. Contiguous
# columns → SIMD over rows (the hot loop in the D&C base-case QR + merge rotations).
@inline function _jac_right!(M::AbstractMatrix{Float64}, ca::Int, cb::Int, c::Float64, s::Float64)
    (c == 1.0 && s == 0.0) && return
    a0 = ca + 1; b0 = cb + 1; nr = size(M, 1)
    if M isa StridedMatrix && stride(M, 1) == 1
        ld = stride(M, 2)
        GC.@preserve M begin
            p = pointer(M); vc = _CVF(c); vs = _CVF(s); i = 1
            @inbounds while i + _CHOLW - 1 <= nr
                pa = _cvptr(p, i, a0, ld); pb = _cvptr(p, i, b0, ld)
                a = vload(_CVF, pa); b = vload(_CVF, pb)
                vstore(vc * a + vs * b, pa); vstore(vc * b - vs * a, pb); i += _CHOLW
            end
            @inbounds while i <= nr
                a = unsafe_load(p, _clidx(i, a0, ld)); b = unsafe_load(p, _clidx(i, b0, ld))
                unsafe_store!(p, c * a + s * b, _clidx(i, a0, ld)); unsafe_store!(p, c * b - s * a, _clidx(i, b0, ld)); i += 1
            end
        end
    else
        @inbounds for i in 1:nr
            a = M[i, a0]; b = M[i, b0]
            M[i, a0] = c * a + s * b
            M[i, b0] = -s * a + c * b
        end
    end
end

# --- Base case: lower-bidiagonal QR-iteration SVD (faer qr_algorithm) ----------------------------
# diag,subdiag length n (subdiag[n-1] ignored). U,V are matrices (cols rotated) or nothing.
# On return diag holds the singular values (descending, ≥0); L = U[:,1:n]·diag·Vᵀ.
function _dc_qr!(diag::AbstractVector{Float64}, subdiag::AbstractVector{Float64}, U, V)
    n = length(diag)
    n == 0 && return true
    epsd = eps(Float64); sml = floatmin(Float64)
    maxiters = max(30, 32) * n * n
    last = n - 1
    mx = 0.0
    @inbounds for i in 1:n
        mx = max(mx, abs(diag[i]), abs(subdiag[i]))
    end
    mx == 0.0 && return true
    maxinv = 1.0 / mx
    @inbounds for i in 1:n
        diag[i] *= maxinv; subdiag[i] *= maxinv
    end
    eps2 = epsd * epsd
    converged = false
    @inbounds for iter in 0:maxiters-1
        for i0 in 0:last-1
            i1 = i0 + 1
            if subdiag[i0+1]^2 <= eps2 * abs(diag[i0+1] * diag[i1+1]) + sml
                subdiag[i0+1] = 0.0
            end
        end
        en = n
        while en >= 2 && subdiag[en-1]^2 <= sml      # faer subdiag[end-2] (0-based) = [en-1] (1-based)
            en -= 1
        end
        en == 1 && (converged = true; break)
        start = en - 1
        while start >= 1 && subdiag[start-1+1] != 0.0
            start -= 1
        end
        found_zero_diag = false
        for i in start:en-2
            if abs(diag[i+1]) <= epsd
                found_zero_diag = true
                val = subdiag[i+1]; subdiag[i+1] = 0.0
                for j in i+1:en-1
                    c, s = _mkgivens(diag[j+1], val)
                    diag[j+1] = c * diag[j+1] - s * val
                    if j + 1 < en
                        val = s * subdiag[j+1]; subdiag[j+1] = c * subdiag[j+1]
                    end
                    !isnothing(V) && _jac_right!(V, j, i, c, -s)  # rot.adjoint().apply_on_the_right((j,i))
                end
            end
        end
        found_zero_diag && continue
        en2 = en - 2; en1 = en - 1
        t00 = (en - start == 2) ? diag[en2+1]^2 : diag[en2+1]^2 + subdiag[en-3+1]^2
        t11 = diag[en1+1]^2 + subdiag[en2+1]^2
        t01 = diag[en2+1] * subdiag[en2+1]
        t01_2 = t01^2
        local mu::Float64
        if t01_2 > sml
            d = 0.5 * (t00 - t11)
            delta = sqrt(d^2 + t01_2); d < 0.0 && (delta = -delta)
            mu = t11 - t01_2 / (d + delta)
        else
            mu = t11
        end
        y = diag[start+1]^2 - mu
        z = diag[start+1] * subdiag[start+1]
        for k in start:en1-1
            c, s = _mkgivens(y, z)
            if k > start
                subdiag[k-1+1] = abs(c * y - s * z)
            end
            diagk = diag[k+1]
            t0 = c * diagk - s * subdiag[k+1]
            t1 = s * diagk + c * subdiag[k+1]
            diagk = t0; subdiag[k+1] = t1
            k1 = k + 1
            y = diagk
            z = -s * diag[k1+1]
            diag[k1+1] = c * diag[k1+1]
            !isnothing(U) && _jac_right!(U, k1, k, c, s)     # apply_on_the_right((k1,k))
            c, s = _mkgivens(y, z)
            diagk = c * y - s * z
            diag[k+1] = diagk
            t0 = c * subdiag[k+1] - s * diag[k1+1]
            t1 = s * subdiag[k+1] + c * diag[k1+1]
            subdiag[k+1] = t0; diag[k1+1] = t1
            if k < en - 2
                y = subdiag[k+1]
                z = -s * subdiag[k1+1]
                subdiag[k1+1] = c * subdiag[k1+1]
            end
            !isnothing(V) && _jac_right!(V, k1, k, c, s)     # apply_on_the_right((k1,k))
        end
    end
    # singular values nonnegative
    @inbounds for j in 0:n-1
        if diag[j+1] < 0.0
            diag[j+1] = -diag[j+1]
            if !isnothing(V)
                for i in 1:n
                    V[i, j+1] = -V[i, j+1]
                end
            end
        end
    end
    # sort descending, permuting U,V columns
    @inbounds for k in 0:n-1
        mxv = 0.0; idx = k
        for kk in k:n-1
            if diag[kk+1] > mxv
                mxv = diag[kk+1]; idx = kk
            end
        end
        if k != idx
            diag[k+1], diag[idx+1] = diag[idx+1], diag[k+1]
            !isnothing(U) && _swap_cols!(U, k+1, idx+1)
            !isnothing(V) && _swap_cols!(V, k+1, idx+1)
        end
    end
    @inbounds for i in 1:n
        diag[i] *= mx; subdiag[i] *= mx
    end
    return converged || true
end

# --- secular equation (faer secular_eq / batch_secular_eq) --------------------------------------
@inline function _secular_eq(shift::Float64, mu::Float64, col0p::AbstractVector{Float64},
        diagp::AbstractVector{Float64}, o::Int)
    res = 1.0
    # @simd reduction: the divisions vectorize (AVX-512 vdivpd) and reassociation is fine — the root is
    # solved to tolerance. This is the hottest bdsdc scalar loop (per profiling).
    @inbounds @simd for i in 1:o
        c = col0p[i]; d = diagp[i]
        res += (c / ((d - shift) - mu)) * (c / ((d + shift) + mu))
    end
    return res
end

# --- secular equation root finder (faer secular_eq_root_finder) ---------------------------------
# Pre-secant bounded-bisection: loop runs `_SEC_BISECT_CAP + 1` iters then breaks. faer uses 4 (→5 iters) to
# pre-tighten the bracket, but the secant + the full-bisection fallback (use_bisection) already guarantee
# convergence — the pre-bisection is only a warm start. Measured: 0 (→1 iter) is fastest AND correctness is
# unchanged (stress: clustered/graded/repeated/tiny-gap spectra all ~1.7e-14). Saves ~4 sec() evals/root, the
# largest single cut to the small-n bdsdc merge cost (root-finding is ~45% of bdsdc). See kb pureblas-svd.
const _SEC_BISECT_CAP = 0
# Secant step of the root finder, HOISTED out of _secular_root to a top-level function: as an inner
# closure, its local `use_bisection` (a Bool reassigned across the loop/nested blocks) got boxed → Any →
# not trim-safe. `sec` (the secular-eq evaluator) is passed as a type parameter so it stays specialized.
# Returns (use_bisection, mu_cur, left_cand, right_cand).
function _secular_secant(sec::F, sh::Float64, left::Float64, right::Float64,
        mu_cur::Float64, mu_prev::Float64, f_cur::Float64, f_prev::Float64) where {F}
    two = 2.0; eight = 8.0; epsilon = eps(Float64)
    if abs(f_prev) < abs(f_cur)
        f_prev, f_cur = f_cur, f_prev
        mu_prev, mu_cur = mu_cur, mu_prev
    end
    left_cand = NaN; right_cand = NaN; use_bisection = false
    sme_sign = (f_prev > 0.0) == (f_cur > 0.0)
    if !sme_sign
        mn, mx = mu_cur < mu_prev ? (mu_cur, mu_prev) : (mu_prev, mu_cur)
        left_cand = mn; right_cand = mx
    end
    while f_cur != 0.0 &&
            abs(mu_cur - mu_prev) > eight * epsilon * max(abs(mu_cur), abs(mu_prev)) &&
            abs(f_cur - f_prev) > epsilon && !use_bisection
        a = (f_cur - f_prev) * (mu_prev * mu_cur) / (mu_prev - mu_cur)
        b = f_cur - a / mu_cur
        mu_zero = -a / b
        f_zero = sec(sh, mu_zero)
        if f_zero < 0.0
            left_cand = mu_zero
        else
            right_cand = mu_zero
        end
        mu_prev = mu_cur; f_prev = f_cur
        mu_cur = mu_zero; f_cur = f_zero
        if sh == left && (mu_cur < 0.0 || mu_cur > right - left)
            use_bisection = true
        end
        if sh == right && (mu_cur > 0.0 || mu_cur < left - right)
            use_bisection = true
        end
        if abs(f_cur) > abs(f_prev)
            k = 1.0
            for _ in 0:3
                mu_opp = -a / (k * f_zero + b)
                f_opp = sec(sh, mu_opp)
                if f_zero < 0.0 && f_opp >= 0.0
                    right_cand = mu_opp; break
                end
                if f_zero > 0.0 && f_opp <= 0.0
                    left_cand = mu_opp; break
                end
                k *= two
            end
            use_bisection = true
        end
    end
    return use_bisection, mu_cur, left_cand, right_cand
end

function _secular_root(col0p::AbstractVector{Float64}, diagp::AbstractVector{Float64}, o::Int,
        left::Float64, right::Float64, last::Bool)
    two = 2.0; eight = 8.0; half = 0.5; epsilon = eps(Float64)
    sec(shift, mu) = _secular_eq(shift, mu, col0p, diagp, o)
    mid = left + (right - left) * half
    f_mid = sec(0.0, mid)
    f_mid_left_shift = sec(left, (right - left) * half)
    f_mid_right_shift = sec(right, (left - right) * half)
    # f_max only feeds the `last` branch (f_right at line below); for !last it equals f_mid_left_shift and
    # is dead — compute it only when needed, saving one sec() eval on every non-last root.
    f_max = last ? sec(left, right - left) : f_mid_left_shift
    shift, mu = (last || f_mid > 0.0) ? (left, (right - left) * half) : (right, (left - right) * half)
    if (f_mid_left_shift <= 0.0) && (f_mid_right_shift > 0.0)
        return shift, mu
    end
    if !last
        if shift == left
            if f_mid_left_shift < 0.0
                shift = right; f_mid = f_mid_right_shift
            end
        elseif f_mid_right_shift > 0.0
            shift = left; f_mid = f_mid_left_shift
        end
    end
    if shift == left
        left_shifted = 0.0; f_left = -Inf
        right_shifted = last ? (right - left) : (right - left) * half
        f_right = last ? f_max : f_mid
    else
        left_shifted = (left - right) * half; f_left = f_mid
        right_shifted = 0.0; f_right = Inf
    end
    f_prev = f_mid
    half0 = half; half1 = half0 * half0; half2 = half1 * half1; half3 = half2 * half2
    base = shift == left ? right_shifted : left_shifted
    mu_values = (base * half3, base * half2, base * half1, base * half0)
    f_values = (sec(shift, mu_values[1]), sec(shift, mu_values[2]), sec(shift, mu_values[3]), sec(shift, mu_values[4]))
    if shift == left
        i = 0
        for idx in 1:4
            if f_values[idx] < 0.0
                left_shifted = mu_values[idx]; f_left = f_values[idx]; i = idx
            end
        end
        if i < 4
            right_shifted = mu_values[i+1]; f_right = f_values[i+1]
        end
    else
        i = 0
        for idx in 1:4
            if f_values[idx] > 0.0
                right_shifted = mu_values[idx]; f_right = f_values[idx]; i = idx
            end
        end
        if i < 4
            left_shifted = mu_values[i+1]; f_left = f_values[i+1]
        end
    end
    iteration_count = 0
    while right_shifted - left_shifted > two * epsilon * max(abs(left_shifted), abs(right_shifted))
        mid_arith = (left_shifted + right_shifted) * half
        mid_geom = sqrt(abs(left_shifted)) * sqrt(abs(right_shifted))
        left_shifted < 0.0 && (mid_geom = -mid_geom)
        mid_shifted = mid_geom == 0.0 ? mid_arith : mid_geom
        f_m = sec(shift, mid_shifted)
        if f_m == 0.0
            return shift, mid_shifted
        elseif f_m > 0.0
            right_shifted = mid_shifted; f_prev = f_right; f_right = f_m
        else
            left_shifted = mid_shifted; f_prev = f_left; f_left = f_m
        end
        iteration_count == _SEC_BISECT_CAP && break
        iteration_count += 1
    end
    if left_shifted == 0.0
        a0, a1, a2, a3 = right_shifted * two, right_shifted, f_prev, f_right
    elseif right_shifted == 0.0
        a0, a1, a2, a3 = left_shifted * two, left_shifted, f_prev, f_left
    else
        a0, a1, a2, a3 = left_shifted, right_shifted, f_left, f_right
    end
    use_bisection, mu_cur, lc, rc = _secular_secant(sec, shift, left, right, a0, a1, a2, a3)
    if !isnan(lc) && !isnan(rc) && lc < rc
        lc > left_shifted && (left_shifted = lc)
        rc < right_shifted && (right_shifted = rc)
    end
    if use_bisection
        while right_shifted - left_shifted > two * epsilon * max(abs(left_shifted), abs(right_shifted))
            mid_shifted = (left_shifted + right_shifted) * half
            f_m = sec(shift, mid_shifted)
            if f_m == 0.0
                break
            elseif f_m > 0.0
                right_shifted = mid_shifted
            else
                left_shifted = mid_shifted
            end
        end
        mu_cur = (left_shifted + right_shifted) * half
    end
    return shift, mu_cur
end

# --- deflation (faer deflation_43 / deflation_44 / deflate) --------------------------------------
# returns (applied::Bool, c, s); rotation stored by caller if applied.
@inline function _deflation_43(diag, col0, i::Int)   # i 0-based
    p = col0[1]; q = col0[i+1]
    (p == 0.0 && q == 0.0) && return false, 0.0, 0.0
    c, s = _mkgivens(p, q)
    r = c * p - s * q
    col0[1] = r; diag[1] = r; col0[i+1] = 0.0
    return true, c, s
end
@inline function _deflation_44(diag, col0, i::Int, j::Int)   # i,j 0-based
    p = col0[i+1]; q = col0[j+1]
    if p == 0.0 && q == 0.0
        diag[i+1] = diag[j+1]; return false, 0.0, 0.0
    end
    c, s = _mkgivens(p, q)
    r = c * p - s * q
    col0[i+1] = r; col0[j+1] = 0.0; diag[i+1] = diag[j+1]
    return true, c, s
end

# deflate the merged problem. jc/js/jidx accumulate Jacobi rotations; perm/transpositions sized n.
# Returns (jacobi_0i, jacobi_ij).
function _deflate!(diag, col0, jc::Vector{Float64}, js::Vector{Float64}, jidx::Vector{Int},
        transp::Vector{Int}, perm::Vector{Int}, k::Int, n::Int,
        real_ind::Vector{Int}, real_col::Vector{Int})
    epsd = eps(Float64); sml = floatmin(Float64)
    max_diag = 0.0
    @inbounds for i in 2:n
        max_diag = max(max_diag, abs(diag[i]))
    end
    max_col0 = 0.0
    @inbounds for i in 1:n
        max_col0 = max(max_col0, abs(col0[i]))
    end
    mx = max(max_diag, max_col0)
    eps_strict = max(epsd * max_diag, sml)
    eps_coarse = 8.0 * epsd * mx
    jacobi_0i = 0; jacobi_ij = 0
    @inbounds begin
        if diag[1] < eps_coarse
            diag[1] = eps_coarse; col0[1] = eps_coarse
        end
        for i in 1:n-1                                  # 0-based i in 1..n-1 → col0[i] = col0[i+1]
            if abs(col0[i+1]) < eps_strict
                col0[i+1] = 0.0
            end
        end
        for i in 1:n-1
            if diag[i+1] < eps_coarse
                ok, c, s = _deflation_43(diag, col0, i)
                if ok
                    jc[jacobi_0i+1] = c; js[jacobi_0i+1] = s; jidx[jacobi_0i+1] = i
                    jacobi_0i += 1
                end
            end
        end
        total_deflation = true
        for i in 1:n-1
            if !(abs(col0[i+1]) < sml)
                total_deflation = false; break
            end
        end
        perm[1] = 0
        p = 1
        for i in 1:n-1
            if abs(diag[i+1]) < sml
                perm[p+1] = i; p += 1
            end
        end
        ii = 1; jj = k + 1
        pp = p
        while pp < n
            if ii >= k + 1
                perm[pp+1] = jj; jj += 1
            elseif jj >= n
                perm[pp+1] = ii; ii += 1
            elseif diag[ii+1] < diag[jj+1]
                perm[pp+1] = jj; jj += 1
            else
                perm[pp+1] = ii; ii += 1
            end
            pp += 1
        end
        if total_deflation
            for i in 1:n-1
                i1 = i - 1
                pi = perm[i+1]
                if abs(diag[pi+1]) < sml || diag[pi+1] > diag[1]
                    perm[i1+1] = perm[i+1]
                else
                    perm[i1+1] = 0; break
                end
            end
        end
        for i in 0:n-1; real_ind[i+1] = i; real_col[i+1] = i; end
        istart = total_deflation ? 0 : 1
        for i in istart:n-1
            pi = perm[(n - (total_deflation ? i + 1 : i)) + 1]
            j = real_col[pi+1]
            a = diag[i+1]; b = diag[j+1]
            diag[i+1] = b; diag[j+1] = a
            if i != 0 && j != 0
                a2 = col0[i+1]; b2 = col0[j+1]
                col0[i+1] = b2; col0[j+1] = a2
            end
            transp[i+1] = j
            real_i = real_ind[i+1]
            real_col[real_i+1] = j
            real_col[pi+1] = i
            real_ind[j+1] = real_i
            real_ind[i+1] = pi
        end
        col0[1] = diag[1]
        for i in 0:n-1
            perm[i+1] = i
        end
        for i in 0:n-1
            j = transp[i+1]
            perm[i+1], perm[j+1] = perm[j+1], perm[i+1]
        end
        i = n - 1
        while i > 0 && (abs(diag[i+1]) < sml || abs(col0[i+1]) < sml)
            i -= 1
        end
        while i > 1
            i1 = i - 1
            if diag[i+1] - diag[i1+1] < eps_strict
                ok, c, s = _deflation_44(diag, col0, i1, i)
                if ok
                    jc[jacobi_0i+jacobi_ij+1] = c; js[jacobi_0i+jacobi_ij+1] = s
                    jidx[jacobi_0i+jacobi_ij+1] = i; jacobi_ij += 1
                end
            end
            i = i1
        end
    end
    return jacobi_0i, jacobi_ij
end

# --- merge: perturb col0, compute singular values & vectors (faer) ------------------------------
@inline function _norm2(v::AbstractVector{Float64}, n::Int)
    s = 0.0
    @inbounds for i in 1:n
        s = hypot(s, v[i])
    end
    return s
end

function _perturb_col0!(zhat, col0, diag, perm::Vector{Int}, s, shifts, mus, n::Int, m::Int)
    if m == 0
        @inbounds for i in 1:n; zhat[i] = 0.0; end
        return
    end
    last_idx = perm[m]                                   # perm[m-1] (0-based value)
    @inbounds for k in 0:n-1
        if col0[k+1] == 0.0
            zhat[k+1] = 0.0; continue
        end
        dk = diag[k+1]
        prod = (s[last_idx+1] + dk) * (mus[last_idx+1] + (shifts[last_idx+1] - dk))
        for l in 0:m-1
            i = perm[l+1]
            i == k && continue
            if i >= k && l == 0
                prod = 0.0; break
            end
            j = i < k ? i : (l > 0 ? perm[l] : i)        # perm[l-1] (0-based value) = perm[l] (1-based)
            term = ((s[j+1] + dk) / (diag[i+1] + dk)) * ((mus[j+1] + (shifts[j+1] - dk)) / (diag[i+1] - dk))
            prod *= term
        end
        tmp = sqrt(prod)
        zhat[k+1] = col0[k+1] > 0.0 ? tmp : -tmp
    end
end

function _compute_singular_values!(shifts, mus, s, diag, diagp, col0, col0p, n::Int, o::Int)
    actual_n = n
    @inbounds while actual_n > 1 && col0[actual_n] == 0.0
        actual_n -= 1
    end
    @inbounds for k in 0:n-1
        s[k+1] = 0.0; shifts[k+1] = 0.0; mus[k+1] = 0.0
        if col0[k+1] == 0.0 || actual_n == 1
            s[k+1] = (k == 0) ? col0[1] : diag[k+1]
            shifts[k+1] = s[k+1]; mus[k+1] = 0.0
            continue
        end
        last_k = (k == actual_n - 1)
        left = diag[k+1]
        if last_k
            right = diag[actual_n] + _norm2(col0, n)
        else
            l = k + 1
            while col0[l+1] == 0.0
                l += 1
            end
            right = diag[l+1]
        end
        shift, mu = _secular_root(col0p, diagp, o, left, right, last_k)
        s[k+1] = shift + mu; shifts[k+1] = shift; mus[k+1] = mu
    end
end

# zhp/dgp/rowidx are the loop-invariant (over k) permuted views: zhp[li]=zhat[perm[li]], dgp[li]=diag[perm[li]],
# rowidx[li]=outer_perm[perm[li]]+1 (all distinct — outer_perm is a permutation). Computing each column's raw
# values CONTIGUOUSLY into vbuf[1:o] lets the divisions vectorize and drops the norm from O(n) to O(o); the
# column is pre-zeroed, so a single scatter through rowidx places the o nonzeros. vm's -1.0 sits at
# outer_perm[1]+1 which never collides with the li≥2 rows (index 0, if present, is always perm[1]), so its
# contribution to the norm is a constant 1.0.
function _compute_singular_vectors!(um, vm, zhat, outer_perm::Vector{Int},
        col_perm_inv::Vector{Int}, actual_n::Int, shifts, mus, n::Int, o::Int,
        zhp::Vector{Float64}, dgp::Vector{Float64}, rowidx::Vector{Int}, vbuf::Vector{Float64})
    @inbounds for k in 0:n-1
        actual_k = (k >= actual_n) ? k : (actual_n - col_perm_inv[k+1] - 1)
        !isnothing(um) && (um[n+1, actual_k+1] = 0.0)
        if zhat[k+1] == 0.0
            !isnothing(um) && (um[outer_perm[k+1]+1, actual_k+1] = 1.0)
            !isnothing(vm) && (vm[outer_perm[k+1]+1, actual_k+1] = 1.0)
            continue
        end
        mu = mus[k+1]; shift = shifts[k+1]
        if !isnothing(um)
            ss = 0.0
            @simd for li in 1:o
                v = (zhp[li] / ((dgp[li] - shift) - mu)) / (dgp[li] + (shift + mu))
                vbuf[li] = v; ss += v * v
            end
            ninv = ss == 0.0 ? 0.0 : 1.0 / sqrt(ss)
            for li in 1:o; um[rowidx[li], actual_k+1] = vbuf[li] * ninv; end
        end
        if !isnothing(vm)
            ss = 1.0                                       # the -1.0 entry at outer_perm[1]+1 contributes 1.0
            @simd for li in 2:o
                v = ((dgp[li] * zhp[li]) / ((dgp[li] - shift) - mu)) / (dgp[li] + (shift + mu))
                vbuf[li] = v; ss += v * v
            end
            ninv = ss == 0.0 ? 0.0 : 1.0 / sqrt(ss)
            vm[outer_perm[1]+1, actual_k+1] = -ninv
            for li in 2:o; vm[rowidx[li], actual_k+1] = vbuf[li] * ninv; end
        end
    end
    !isnothing(um) && (um[n+1, n+1] = 1.0)
end

function _compute_svd_of_m!(um, vm, diag, col0, outer_perm::Vector{Int}, n::Int, ws::_DCWork)
    diag[1] = 0.0
    actual_n = n
    @inbounds while actual_n > 1 && diag[actual_n] == 0.0
        actual_n -= 1
    end
    permloc = ws.permloc
    o = 0
    @inbounds for i in 0:actual_n-1
        if col0[i+1] != 0.0
            o += 1; permloc[o] = i
        end
    end
    col0p = ws.col0p; diagp = ws.diagp
    @inbounds for kk in 0:o-1
        col0p[kk+1] = col0[permloc[kk+1]+1]; diagp[kk+1] = diag[permloc[kk+1]+1]
    end
    shifts = ws.shifts; mus = ws.mus; s = ws.s; zhat = ws.zhat
    _compute_singular_values!(shifts, mus, s, diag, diagp, col0, col0p, n, o)
    _perturb_col0!(zhat, col0, diag, permloc, s, shifts, mus, n, o)
    zhp = ws.zhp; rowidx = ws.rowidx           # loop-invariant permuted views for the vector-formation loop
    @inbounds for li in 1:o
        zhp[li] = zhat[permloc[li]+1]; rowidx[li] = outer_perm[permloc[li]+1]+1
    end
    col_perm = ws.col_perm
    @inbounds for i in 0:actual_n-1; col_perm[i+1] = i; end
    @inbounds for i0 in 0:actual_n-2
        i1 = i0 + 1
        if s[i0+1] > s[i1+1]
            s[i0+1], s[i1+1] = s[i1+1], s[i0+1]
            col_perm[i0+1], col_perm[i1+1] = col_perm[i1+1], col_perm[i0+1]
        end
    end
    col_perm_inv = ws.col_perm_inv
    @inbounds for i in 0:actual_n-1
        col_perm_inv[col_perm[i+1]+1] = i
    end
    _compute_singular_vectors!(um, vm, zhat, outer_perm, col_perm_inv, actual_n, shifts, mus, n, o,
        zhp, diagp, rowidx, ws.vbuf)
    @inbounds for idx in 0:actual_n-1
        diag[idx+1] = s[actual_n-idx-1+1]
    end
    @inbounds for idx in 0:(n-actual_n-1)
        diag[actual_n+idx+1] = s[actual_n+idx+1]
    end
end

# --- vector combine (faer update_u / update_v) via PureBLAS gemm! into shared scratch ------------
function _combine_v!(V, vm, cv, k::Int, rem::Int, n::Int)
    k > 0 && gemm!(view(cv, 1:k, 1:n), view(V, 1:k, 2:k+1), view(vm, 2:k+1, 1:n))
    rem > 0 && gemm!(view(cv, k+2:n, 1:n), view(V, k+2:n, k+2:n), view(vm, k+2:n, 1:n))
    vk0 = V[k+1, 1]
    @inbounds for jj in 1:n; cv[k+1, jj] = vk0 * vm[1, jj]; end
    @inbounds for jj in 1:n, ii in 1:n; V[ii, jj] = cv[ii, jj]; end
end

function _combine_u!(U, um, cu, k::Int, rem::Int, n::Int)
    gemm!(view(cu, 1:k+1, 1:n+1), view(U, 1:k+1, 1:k+1), view(um, 1:k+1, 1:n+1))
    gemm!(view(cu, 1:k+1, 1:n+1), view(U, 1:k+1, n+1:n+1), view(um, n+1:n+1, 1:n+1); alpha = 1.0, beta = 1.0)
    gemm!(view(cu, k+2:n+1, 1:n+1), view(U, k+2:n+1, k+2:n+1), view(um, k+2:n+1, 1:n+1))
    gemm!(view(cu, k+2:n+1, 1:n+1), view(U, k+2:n+1, 1:1), view(um, 1:1, 1:n+1); alpha = 1.0, beta = 1.0)
    @inbounds for jj in 1:n+1, ii in 1:n+1; U[ii, jj] = cu[ii, jj]; end
end

const _DC_THRESHOLD = 64

# --- recursive divide-and-conquer driver (faer divide_and_conquer, Full-U, serial) --------------
# All scratch comes from the preallocated `ws` (see _DCWork) — the recursion allocates nothing.
function _dc!(diag::AbstractVector{Float64}, subdiag::AbstractVector{Float64}, U, V, threshold::Int,
        ws::_DCWork)
    bufA = ws.bufA; bufB = ws.bufB; bufC = ws.bufC; bufD = ws.bufD
    threshold = max(threshold, 4)
    n = length(diag)
    if n < threshold
        ualloc = view(bufA, 1:n+1, 1:n+1)
        fill!(ualloc, 0.0)
        @inbounds for i in 1:n+1; ualloc[i, i] = 1.0; end
        if !isnothing(V)
            fill!(V, 0.0)
            @inbounds for i in 1:n; V[i, i] = 1.0; end
        end
        val = subdiag[n]; subdiag[n] = 0.0
        j = n; i = n
        @inbounds while i > 0
            i -= 1
            c, s = _mkgivens(diag[i+1], val)
            diag[i+1] = c * diag[i+1] - s * val
            if i > 0
                val = s * subdiag[i]; subdiag[i] = c * subdiag[i]   # subdiag[i-1] (0b) = subdiag[i] (1b)
            end
            _jac_right!(ualloc, i, j, c, -s)                        # rot.adjoint().apply_on_the_right((i,j))
        end
        _dc_qr!(diag, subdiag, ualloc, V)
        @inbounds for jj in 1:n+1, ii in 1:n+1; U[ii, jj] = ualloc[ii, jj]; end
        return
    end
    mx = 0.0
    @inbounds for i in 1:n; mx = max(mx, abs(diag[i]), abs(subdiag[i])); end
    if mx == 0.0
        fill!(U, 0.0); @inbounds for i in 1:n+1; U[i, i] = 1.0; end
        if !isnothing(V); fill!(V, 0.0); @inbounds for i in 1:n; V[i, i] = 1.0; end; end
        return
    end
    maxinv = 1.0 / mx
    @inbounds for i in 1:n; diag[i] *= maxinv; subdiag[i] *= maxinv; end
    k = div(n, 2); rem = n - k - 1
    alpha = diag[k+1]; beta = subdiag[k+1]
    d1 = view(diag, 1:k); subd1 = view(subdiag, 1:k)
    d2 = view(diag, k+2:n); subd2 = view(subdiag, k+2:n)
    U1 = view(U, 1:k+1, 2:k+2); U2 = view(U, k+2:n+1, k+2:n+1)
    V1 = isnothing(V) ? nothing : view(V, 1:k, 2:k+1)
    V2 = isnothing(V) ? nothing : view(V, k+2:n, k+2:n)
    _dc!(d1, subd1, U1, V1, threshold, ws)
    _dc!(d2, subd2, U2, V2, threshold, ws)
    !isnothing(V) && (V[k+1, 1] = 1.0)
    @inbounds for i in k-1:-1:0
        diag[i+2] = diag[i+1]
    end
    lambda = U[k+1, k+2]
    phi = U[k+2, n+1]
    al = alpha * lambda; bp = beta * phi
    r0 = hypot(al, bp)
    c0, s0 = r0 == 0.0 ? (1.0, 0.0) : (al / r0, bp / r0)
    col0 = subdiag
    diag[1] = r0; col0[1] = r0
    @inbounds for jj in 1:k
        col0[jj+1] = alpha * U[k+1, jj+1]
    end
    @inbounds for jj in k+1:n-1
        col0[jj+1] = beta * U[k+2, jj+1]
    end
    @inbounds for r in 0:k
        a = U[r+1, k+2]
        U[r+1, 1] = c0 * a
        U[r+1, n+1] = -s0 * a
    end
    @inbounds for r in k+1:n
        xn = U[r+1, n+1]
        U[r+1, 1] = s0 * xn
        U[r+1, n+1] = c0 * xn
    end
    jc = ws.jc; js = ws.js; jidx = ws.jidx; transp = ws.transp; perm = ws.perm
    @inbounds for i in 0:n-1; transp[i+1] = i; end
    jacobi_0i, jacobi_ij = _deflate!(diag, col0, jc, js, jidx, transp, perm, k, n, ws.real_ind, ws.real_col)
    um = view(bufA, 1:n+1, 1:n+1); fill!(um, 0.0)
    vm = isnothing(V) ? nothing : (vmv = view(bufC, 1:n, 1:n); fill!(vmv, 0.0); vmv)
    _compute_svd_of_m!(um, vm, diag, col0, perm, n, ws)
    @inbounds for i in 1:n; col0[i] = 0.0; end
    for t in (jacobi_0i+jacobi_ij):-1:(jacobi_0i+1)
        c = jc[t]; s = js[t]; i0 = jidx[t]
        ii = i0 - 1; jj = i0
        actual_i = perm[ii+1]; actual_j = perm[jj+1]
        _jac_left!(um, actual_j, actual_i, c, s)
        !isnothing(vm) && _jac_left!(vm, actual_j, actual_i, c, s)
    end
    for t in jacobi_0i:-1:1
        c = jc[t]; s = js[t]; i0 = jidx[t]
        _jac_left!(um, i0, 0, c, s)
    end
    !isnothing(V) && _combine_v!(V, vm, view(bufD, 1:n, 1:n), k, rem, n)
    _combine_u!(U, um, view(bufB, 1:n+1, 1:n+1), k, rem, n)
    @inbounds for i in 1:n; diag[i] *= mx; end
    return
end

# Bidiagonal SVD via D&C. Input: upper-bidiagonal B (d=diagonal, e=superdiagonal). Returns
# (s, Ul, Vl) with the LOWER bidiagonal L=Bᵀ = Ul·diag(s)·Vlᵀ (Ul=U[1:n,1:n], Vl=V), so B's left
# vectors = Vl, right vectors = Ul.
function bdsdc!(d::Vector{Float64}, e::Vector{Float64})
    n = length(d)
    diag = copy(d)
    subdiag = zeros(n)
    @inbounds for i in 1:n-1; subdiag[i] = e[i]; end
    U = zeros(n+1, n+1); V = zeros(n, n)
    _dc!(diag, subdiag, U, V, _DC_THRESHOLD, _get_dcwork(n))
    return diag, U[1:n, 1:n], V
end
