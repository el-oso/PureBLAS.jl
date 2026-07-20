# Symmetric TRIDIAGONAL eigensolver via Cuppen divide-and-conquer (LAPACK dstedc/dlaed0-4), generic
# over T<:Real. Mirrors src/svd_dc.jl (validated bidiagonal-SVD D&C) for the recursion tree /
# bracketed secular-equation root finder, but solves the SIMPLER single-pole eigen secular equation
# (dlaed4) with a plain dlaed1-style rank-1 tear (no augmented row, ONE Q). The n≤_STEDC_NB base case
# recurses to the module's own T-generic _steqr! (src/eigen.jl). Powers M-E2/M-E3 eigenvector paths
# (_syev!/_heev! jobz='V'); the crossover is algorithm-intrinsic (LAPACK SMLSIZ), not a perf tuning knob.
#
# Formulas (deflation tolerance, zhat Löwner product, merge z/rho setup) verified against current
# Reference-LAPACK dlaed1.f/dlaed2.f/dlaed3.f/dlaed4.f/dlamch.f.

# --- generic Givens / column rotation (BLAS DROT convention: x←c·x+s·y, y←−s·x+c·y). Kept LOCAL to
# this D&C (deflation rotations), distinct from svd.jl's _givens/_rot_cols! (different sign convention).
@inline function _mkgivens_g(p::T, q::T) where {T <: Real}
    if q == zero(T)
        return (p < zero(T) ? -one(T) : one(T)), zero(T)
    elseif p == zero(T)
        return zero(T), (q < zero(T) ? one(T) : -one(T))
    elseif abs(p) > abs(q)
        t = q / p; u = sqrt(muladd(t, t, one(T))); p < zero(T) && (u = -u)
        c = one(T) / u; return c, -t * c
    else
        t = p / q; u = sqrt(muladd(t, t, one(T))); q < zero(T) && (u = -u)
        s = -one(T) / u; return -t * s, s
    end
end

@inline function _drot_cols!(M::AbstractMatrix{T}, j1::Int, j2::Int, c::T, s::T) where {T <: Real}
    @inbounds @simd for i in 1:size(M, 1)
        a = M[i, j1]; b = M[i, j2]
        M[i, j1] = c * a + s * b
        M[i, j2] = -s * a + c * b
    end
    return M
end

# --- secular-equation bracketed root finder (ported from svd_dc.jl `_secular_secant`/`_secular_root`,
# generic in T, taking the equation `sec::F` as a parameter instead of the SVD's paired-pole form).
function _secular_secant(
        sec::F, sh::T, left::T, right::T,
        mu_cur::T, mu_prev::T, f_cur::T, f_prev::T
    ) where {T <: Real, F}
    two = T(2); eight = T(8); epsilon = eps(T)
    if abs(f_prev) < abs(f_cur)
        f_prev, f_cur = f_cur, f_prev
        mu_prev, mu_cur = mu_cur, mu_prev
    end
    left_cand = T(NaN); right_cand = T(NaN); use_bisection = false
    sme_sign = (f_prev > 0) == (f_cur > 0)
    if !sme_sign
        mn, mx = mu_cur < mu_prev ? (mu_cur, mu_prev) : (mu_prev, mu_cur)
        left_cand = mn; right_cand = mx
    end
    while f_cur != 0 &&
            abs(mu_cur - mu_prev) > eight * epsilon * max(abs(mu_cur), abs(mu_prev)) &&
            abs(f_cur - f_prev) > epsilon && !use_bisection
        a = (f_cur - f_prev) * (mu_prev * mu_cur) / (mu_prev - mu_cur)
        b = f_cur - a / mu_cur
        mu_zero = -a / b
        f_zero = sec(sh, mu_zero)
        if f_zero < 0
            left_cand = mu_zero
        else
            right_cand = mu_zero
        end
        mu_prev = mu_cur; f_prev = f_cur
        mu_cur = mu_zero; f_cur = f_zero
        if sh == left && (mu_cur < 0 || mu_cur > right - left)
            use_bisection = true
        end
        if sh == right && (mu_cur > 0 || mu_cur < left - right)
            use_bisection = true
        end
        if abs(f_cur) > abs(f_prev)
            k = one(T)
            for _ in 0:3
                mu_opp = -a / (k * f_zero + b)
                f_opp = sec(sh, mu_opp)
                if f_zero < 0 && f_opp >= 0
                    right_cand = mu_opp; break
                end
                if f_zero > 0 && f_opp <= 0
                    left_cand = mu_opp; break
                end
                k *= two
            end
            use_bisection = true
        end
    end
    return use_bisection, mu_cur, left_cand, right_cand
end

function _secular_root(sec::F, left::T, right::T, last::Bool) where {T <: Real, F}
    two = T(2); half = T(0.5); epsilon = eps(T)
    mid = left + (right - left) * half
    f_mid = sec(zero(T), mid)
    f_mid_left_shift = sec(left, (right - left) * half)
    f_mid_right_shift = sec(right, (left - right) * half)
    f_max = last ? sec(left, right - left) : f_mid_left_shift
    shift, mu = (last || f_mid > 0) ? (left, (right - left) * half) : (right, (left - right) * half)
    if (f_mid_left_shift <= 0) && (f_mid_right_shift > 0)
        return shift, mu
    end
    if !last
        if shift == left
            if f_mid_left_shift < 0
                shift = right; f_mid = f_mid_right_shift
            end
        elseif f_mid_right_shift > 0
            shift = left; f_mid = f_mid_left_shift
        end
    end
    if shift == left
        left_shifted = zero(T); f_left = T(-Inf)
        right_shifted = last ? (right - left) : (right - left) * half
        f_right = last ? f_max : f_mid
    else
        left_shifted = (left - right) * half; f_left = f_mid
        right_shifted = zero(T); f_right = T(Inf)
    end
    f_prev = f_mid
    half0 = half; half1 = half0 * half0; half2 = half1 * half1; half3 = half2 * half2
    base = shift == left ? right_shifted : left_shifted
    mu_values = (base * half3, base * half2, base * half1, base * half0)
    f_values = (sec(shift, mu_values[1]), sec(shift, mu_values[2]), sec(shift, mu_values[3]), sec(shift, mu_values[4]))
    if shift == left
        i = 0
        for idx in 1:4
            if f_values[idx] < 0
                left_shifted = mu_values[idx]; f_left = f_values[idx]; i = idx
            end
        end
        if i < 4
            right_shifted = mu_values[i + 1]; f_right = f_values[i + 1]
        end
    else
        i = 0
        for idx in 1:4
            if f_values[idx] > 0
                right_shifted = mu_values[idx]; f_right = f_values[idx]; i = idx
            end
        end
        if i < 4
            left_shifted = mu_values[i + 1]; f_left = f_values[i + 1]
        end
    end
    iteration_count = 0
    while right_shifted - left_shifted > two * epsilon * max(abs(left_shifted), abs(right_shifted))
        mid_arith = (left_shifted + right_shifted) * half
        mid_geom = sqrt(abs(left_shifted)) * sqrt(abs(right_shifted))
        left_shifted < 0 && (mid_geom = -mid_geom)
        mid_shifted = mid_geom == 0 ? mid_arith : mid_geom
        f_m = sec(shift, mid_shifted)
        if f_m == 0
            return shift, mid_shifted
        elseif f_m > 0
            right_shifted = mid_shifted; f_prev = f_right; f_right = f_m
        else
            left_shifted = mid_shifted; f_prev = f_left; f_left = f_m
        end
        iteration_count == 0 && break   # faer-style: pre-bisection cap (see svd_dc.jl _SEC_BISECT_CAP)
        iteration_count += 1
    end
    if left_shifted == 0
        a0, a1, a2, a3 = right_shifted * two, right_shifted, f_prev, f_right
    elseif right_shifted == 0
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
            if f_m == 0
                break
            elseif f_m > 0
                right_shifted = mid_shifted
            else
                left_shifted = mid_shifted
            end
        end
        mu_cur = (left_shifted + right_shifted) * half
    end
    return shift, mu_cur
end

# Single-pole secular equation (dlaed4): f(λ) = 1 + ρ·Σ zᵢ²/(dᵢ−λ), λ = shift+mu represented via
# (dᵢ−shift)−mu to avoid catastrophic cancellation near clusters (never form dᵢ−λ directly).
@inline function _secular_eq_eigen(
        shift::T, mu::T, w::AbstractVector{T}, dlambda::AbstractVector{T},
        rho::T, K::Int
    ) where {T <: Real}
    res = one(T)
    @inbounds @simd for i in 1:K
        d = dlambda[i]; zi = w[i]
        res += rho * (zi * zi) / ((d - shift) - mu)
    end
    return res
end

const _STEDC_NB = 25   # req8-ok: LAPACK dstedc SMLSIZ — algorithm-intrinsic base-case crossover (NOT a
# hardware tuning knob; machine-independent, like _SVD_DC_CROSS). steqr↔D&C switch.

# --- recursive D&C driver (dlaed0 recursion + dlaed1 merge + dlaed2 deflation + dlaed3 secular solve)
# d (diag) → eigenvalues ascending; e (subdiag, length n-1) destroyed; Z (n×n) → eigenvectors (Z=I on
# entry not required — the base case / n==1 seed Z themselves; parent slots are zeroed before recursing).
function _dc_eigen!(
        d::AbstractVector{T}, e::AbstractVector{T}, Z::AbstractMatrix{T},
        nb::Int = _STEDC_NB
    ) where {T <: Real}
    n = length(d)
    if n == 0
        return
    end
    if n == 1
        Z[1, 1] = one(T)
        return
    end
    if n <= nb
        # in-house T-generic steqr base (compz='I' fills Z=I then accumulates eigenvectors; d ← ascending
        # eigenvalues in place). e is copied to scratch since _steqr! destroys its subdiagonal.
        fill!(view(Z, 1:n, 1:n), zero(T))
        @inbounds for i in 1:n
            Z[i, i] = one(T)
        end
        ec = Vector{T}(undef, max(n - 1, 1))
        @inbounds for i in 1:(n - 1)
            ec[i] = e[i]
        end
        _steqr!('I', view(d, 1:n), view(ec, 1:max(n - 1, 0)), view(Z, 1:n, 1:n))
        return
    end

    k = n ÷ 2
    d1 = view(d, 1:k); e1 = view(e, 1:(k - 1))
    d2 = view(d, (k + 1):n); e2 = view(e, (k + 1):(n - 1))
    Z1 = view(Z, 1:k, 1:k); Z2 = view(Z, (k + 1):n, (k + 1):n)
    fill!(Z, zero(T))

    # --- dlaed1 tear: T = blkdiag(T1',T2') + rho·v·vᵀ, v = e_k ± e_{k+1}; boundary diagonal entries
    # of the two subproblems get rho subtracted BEFORE recursing.
    rho_raw = e[k]
    absr = abs(rho_raw)
    d[k] -= absr
    d[k + 1] -= absr
    _dc_eigen!(d1, e1, Z1, nb)
    _dc_eigen!(d2, e2, Z2, nb)
    n2 = n - k

    # z = [last row of Q1 ; first row of Q2], sign-flip 2nd half if rho_raw<0, unit-normalize
    # (raw norm is exactly √2 — two unit rows of orthogonal matrices); rho := 2·|rho_raw|.
    z = Vector{T}(undef, n)
    @inbounds for j in 1:k
        z[j] = Z1[k, j]
    end
    @inbounds for j in 1:n2
        z[k + j] = Z2[1, j]
    end
    if rho_raw < 0
        @inbounds for j in (k + 1):n
            z[j] = -z[j]
        end
    end
    invs2 = one(T) / sqrt(T(2))
    @inbounds for j in 1:n
        z[j] *= invs2
    end
    rho = T(2) * absr

    # --- dlaed2: physically sort (d,z,Q-columns) into global ascending d order.
    ord = sortperm(d)
    dsort = d[ord]; zsort = z[ord]; Zsort = Z[:, ord]

    # deflation tolerance: LAPACK dlaed2 TOL = 64·DLAMCH('E')·max(...), DLAMCH('E')=eps(T)/2 → 32·eps·max.
    dmax = maximum(abs, dsort); zmax = maximum(abs, zsort)
    tol = T(32) * eps(T) * max(dmax, zmax)

    deflated = falses(n)
    @inbounds for i in 1:n
        if rho * abs(zsort[i]) <= tol
            deflated[i] = true    # rule (a): negligible z ⇒ already an exact eigenpair (e_i, dsort[i])
        end
    end
    survivors = findall(!, deflated)
    if length(survivors) >= 2
        pj = survivors[1]
        @inbounds for idx in 2:length(survivors)
            nj = survivors[idx]
            s = zsort[pj]; c = zsort[nj]
            tau = hypot(c, s)
            if tau == 0
                pj = nj; continue
            end
            cc = c / tau; ss = -s / tau
            tdiff = dsort[nj] - dsort[pj]
            if abs(tdiff * cc * ss) <= tol
                # rule (b): near-equal poles ⇒ Givens-zero pj's z, chain nj forward carrying τ
                zsort[nj] = tau; zsort[pj] = zero(T)
                _drot_cols!(Zsort, pj, nj, cc, ss)
                newpj = dsort[pj] * cc * cc + dsort[nj] * ss * ss
                newnj = dsort[pj] * ss * ss + dsort[nj] * cc * cc
                dsort[pj] = newpj; dsort[nj] = newnj
                deflated[pj] = true
                pj = nj
            else
                pj = nj
            end
        end
    end
    survivor_idx = findall(!, deflated)
    K = length(survivor_idx)

    Dout = copy(dsort); Qout = copy(Zsort)   # deflated entries are already final at this point

    if K > 0
        dlambda = dsort[survivor_idx]; w = zsort[survivor_idx]
        lam = Vector{T}(undef, K); oidx = Vector{Int}(undef, K); mu = Vector{T}(undef, K)
        sec_ = (shift, mu_) -> _secular_eq_eigen(shift, mu_, w, dlambda, rho, K)
        wsq = zero(T)
        @inbounds for i in 1:K
            wsq += w[i] * w[i]
        end
        @inbounds for j in 1:K
            left = dlambda[j]
            lastj = (j == K)
            right = lastj ? dlambda[K] + rho * wsq : dlambda[j + 1]
            shift, muj = _secular_root(sec_, left, right, lastj)
            lam[j] = shift + muj
            oidx[j] = (shift == left) ? j : j + 1
            mu[j] = muj
        end
        # dlaed3 Löwner ẑ (numerically stable via the shift/mu δ representation, never dᵢ−λⱼ directly)
        delta(i, j) = (i == oidx[j]) ? -mu[j] : (dlambda[i] - dlambda[oidx[j]]) - mu[j]
        zhat = Vector{T}(undef, K)
        @inbounds for i in 1:K
            Wi = delta(i, i)
            for j in 1:K
                j == i && continue
                Wi *= delta(i, j) / (dlambda[i] - dlambda[j])
            end
            zhat[i] = copysign(sqrt(max(-Wi, zero(T))), w[i])
        end
        # dlaed3 eigenvectors of D+ρzzᵀ restricted to survivors: v_j(i) = ẑᵢ/δᵢⱼ, unit-normalized
        V = Matrix{T}(undef, K, K)
        @inbounds for j in 1:K
            ss = zero(T)
            for i in 1:K
                v = zhat[i] / delta(i, j)
                V[i, j] = v; ss += v * v
            end
            ninv = ss == 0 ? zero(T) : one(T) / sqrt(ss)
            for i in 1:K
                V[i, j] *= ninv
            end
        end
        B = Zsort[:, survivor_idx]           # gather the surviving basis columns (n×K)
        R = Matrix{T}(undef, n, K)
        gemm!(R, B, V)                       # final eigenvectors for the survivor slots = B·V
        @inbounds for j in 1:K
            pos = survivor_idx[j]
            Dout[pos] = lam[j]
            for i in 1:n
                Qout[i, pos] = R[i, j]
            end
        end
    end

    # final ascending re-sort merging deflated + secular eigenvalues, so the invariant ("d ascending
    # on return") holds for the parent's z-extraction (last row of Q1 / first row of Q2).
    ord2 = sortperm(Dout)
    copyto!(d, Dout[ord2])                 # copyto!(view/dest, X), not .= / slice-assign: those carry Base
    copyto!(Z, Qout[:, ord2])              # setindex_shape_check / DimensionMismatch error paths (--trim-unsafe)
    return
end

# Public entry point. d (diag, length n) → eigenvalues ascending; e (subdiag, length n-1) destroyed;
# Z (n×n) → eigenvectors (columns). Generic over T<:Real (Float64 and Float32).
#
# Mirrors LAPACK dstedc: split at negligible off-diagonals into INDEPENDENT submatrices, and DLASCL each
# unreduced block to UNIT max-norm before the divide-and-conquer, unscaling its eigenvalues after. Without
# this, the deflation tolerance `32·eps·max(|d|,|z|)` has an absolute floor (z is unit-normalized ⇒ zmax~1)
# that spuriously deflates genuine couplings when ‖T‖ ≪ 1 — perfectly orthonormal eigenvectors of the WRONG
# matrix (resid/‖A‖ → O(1) at ‖A‖=1e-14). The _syev! anrm-prescale can't cover it (its band admits ‖A‖=1e-10
# unscaled). Blocks are decoupled, so Z is block-diagonal; a final global sort restores ascending order.
function _stedc!(
        d::AbstractVector{T}, e::AbstractVector{T}, Z::AbstractMatrix{T};
        nb::Int = _STEDC_NB
    ) where {T <: Real}
    n = length(d)
    size(Z) == (n, n) || throw(DimensionMismatch("_stedc!: Z must be n×n"))
    (n == 0 || length(e) >= n - 1) || throw(DimensionMismatch("_stedc!: e too short"))
    n == 0 && return d, Z
    fill!(Z, zero(T))
    epsT = eps(T)
    lo = 1
    @inbounds while lo <= n
        # find the end of this unreduced submatrix (split at |e[m]| ≤ eps·√(|d[m]·d[m+1]|), dsteqr's test)
        hi = lo
        while hi < n
            if abs(e[hi]) <= epsT * sqrt(abs(d[hi])) * sqrt(abs(d[hi + 1]))
                e[hi] = zero(T); break
            end
            hi += 1
        end
        m = hi - lo + 1
        if m == 1
            Z[lo, lo] = one(T)                                   # 1×1 block: eigenvalue = d[lo], vector = eₗₒ
        else
            db = view(d, lo:hi); eb = view(e, lo:(hi - 1)); Zb = view(Z, lo:hi, lo:hi)
            orgnrm = zero(T)                                     # DLASCL norm = max(|d|, |e|) over the block
            for i in lo:hi
                orgnrm = max(orgnrm, abs(d[i]))
            end
            for i in lo:(hi - 1)
                orgnrm = max(orgnrm, abs(e[i]))
            end
            if orgnrm != zero(T)
                for i in lo:hi
                    d[i] /= orgnrm
                end
                for i in lo:(hi - 1)
                    e[i] /= orgnrm
                end
            end
            _dc_eigen!(db, eb, Zb, nb)                           # solve the unit-scaled block (fills Zb)
            if orgnrm != zero(T)
                for i in lo:hi
                    d[i] *= orgnrm
                end              # unscale eigenvalues (vectors are scale-free)
            end
        end
        lo = hi + 1
    end

    # global ascending sort of eigenvalues + eigenvector columns (blocks are each ascending but interleaved).
    if !issorted(view(d, 1:n))
        ord = sortperm(view(d, 1:n))
        copyto!(view(d, 1:n), d[ord])       # copyto!(view, X), not slice-assign (setindex_shape_check --trim-unsafe)
        copyto!(view(Z, :, 1:n), Z[:, ord])
    end
    return d, Z
end
