# LAPACK generalized SVD of a matrix pair — the classic {s,d,c,z}ggsvd driver, RANK-DEFICIENT-capable.
# For A (m×n), B (p×n) finds unitary U (m×m), V (p×p), Q (n×n) and integers k, l
# (k+l = effective rank of [A;B], l = effective rank of B) such that
#     Uᴴ·A·Q = D1·[0 R],   Vᴴ·B·Q = D2·[0 R]
# with R (k+l)×(k+l) upper triangular nonsingular and (LAPACK dggsvd doc layout)
#   m-k-l ≥ 0:  D1 = [I 0; 0 C; 0 0],  D2 = [0 S; 0 0],   C=diag(α[k+1:k+l]), S=diag(β[k+1:k+l])
#   m-k-l < 0:  D1 = [I 0 0; 0 C 0],   D2 = [0 S 0; 0 0 I; 0 0 0],  C=diag(α[k+1:m]), S=diag(β[k+1:m])
# α[1:k]=1, β[1:k]=0; α[m+1:k+l]=0, β[m+1:k+l]=1 (deficient-m case); α,β zero beyond k+l. αᵢ²+βᵢ²=1.
#
# Method — a faithful port of the reference-LAPACK pipeline (NOT the previous full-rank-only
# QR+CS-of-the-stack shortcut, whose stacked QR lost the smaller matrix's row space under norm
# imbalance and could not produce k, l for rank-deficient pairs):
#   1. dggsvp/zggsvp preprocessing: QR-with-column-pivoting of B (reveals l), RQ compression,
#      QR-with-column-pivoting of the leading n-l columns of A (reveals k), RQ compression, and a
#      final QR of the trailing block — reducing the pair to the k+l-column triangular form.
#   2. dtgsja/ztgsja: Kogbetliantz cyclic Jacobi on the l×l triangular tails (dlags2/zlags2 2×2
#      rotation kernel over dlasv2), convergence via dlapll row-parallelism, then the α/β/R endgame.
# Tolerances follow dggsvd exactly: tola = max(m,n)·max(‖A‖₁,unfl)·ulp, tolb likewise for B —
# these DEFINE k and l, so rank decisions match LAPACK's on the same input.
# Norm imbalance is handled the way LAPACK handles it: A and B are never stacked, each matrix is
# transformed by its own rotations, so ‖A‖≪‖B‖ (or ≫) does not degrade the smaller matrix.
#
# Self-contained on purpose (own _ggs_* Householder/rotation kernels, no gesvd!/gemm! dependency):
# the reflector conventions (real-β zlarfg, RQ row reflectors) must match LAPACK's bit-for-bit
# semantics, and the validation harness runs this file standalone against the OpenBLAS oracle.
# Trim-safe: no eval, no captured-mutated closures, static throw strings.

# ── scaled 2-norm (lassq-style; overflow/underflow safe) ─────────────────────────────────────────
function _ggs_nrm2(x::AbstractVector{T}) where {T}
    RT = real(T)
    scl = zero(RT); ssq = one(RT)
    @inbounds for xi in x
        axi = abs(xi)
        if axi != 0
            if scl < axi
                ssq = one(RT) + ssq * (scl / axi)^2
                scl = axi
            else
                ssq += (axi / scl)^2
            end
        end
    end
    return scl * sqrt(ssq)
end

# 1-norm (max column abs-sum) — dlange('1') equivalent, drives the rank tolerances.
function _ggs_norm1(A::AbstractMatrix{T}) where {T}
    RT = real(T)
    v = zero(RT)
    @inbounds for j in 1:size(A, 2)
        s = zero(RT)
        for i in 1:size(A, 1)
            s += abs(A[i, j])
        end
        v = max(v, s)
    end
    return v
end

# ── Householder reflector (dlarfg/zlarfg, real-β convention) ─────────────────────────────────────
# On exit x[1] = β (real, stored in T), x[2:end] = v tail (v₁ ≡ 1 implicit); returns τ.
# H = I - τ·v·vᴴ satisfies Hᴴ·x = β·e₁. (LAPACK's subnormal rescale loop is skipped: inputs at the
# driver level are pre-scaled by the caller's data, and β is formed via scaled hypot/nrm2.)
function _ggs_larfg!(x::AbstractVector{T}) where {T}
    RT = real(T)
    n = length(x)
    n == 0 && return zero(T)
    alpha = x[1]
    xnorm = n > 1 ? _ggs_nrm2(view(x, 2:n)) : zero(RT)
    if xnorm == 0 && imag(alpha) == 0
        return zero(T)
    end
    beta = -copysign(hypot(real(alpha), imag(alpha), xnorm), real(alpha))
    tau = (T(beta) - alpha) / beta
    scal = one(T) / (alpha - T(beta))
    @inbounds for i in 2:n
        x[i] *= scal
    end
    x[1] = T(beta)
    return tau
end

# C := (I - τ·u·uᴴ)·C, u explicit (caller places the unit element).
function _ggs_larf_left!(C::AbstractMatrix{T}, u::AbstractVector{T}, τ::T) where {T}
    τ == zero(T) && return C
    mm, nn = size(C)
    @inbounds for j in 1:nn
        w = zero(T)
        for i in 1:mm
            w += conj(u[i]) * C[i, j]
        end
        w *= τ
        for i in 1:mm
            C[i, j] -= u[i] * w
        end
    end
    return C
end

# C := C·(I - τ·u·uᴴ), u explicit.
function _ggs_larf_right!(C::AbstractMatrix{T}, u::AbstractVector{T}, τ::T) where {T}
    τ == zero(T) && return C
    mm, nn = size(C)
    w = Vector{T}(undef, mm)
    fill!(w, zero(T))
    @inbounds for k in 1:nn
        uk = u[k]
        if uk != zero(T)
            for i in 1:mm
                w[i] += C[i, k] * uk
            end
        end
    end
    @inbounds for j in 1:nn
        cj = τ * conj(u[j])
        if cj != zero(T)
            for i in 1:mm
                C[i, j] -= w[i] * cj
            end
        end
    end
    return C
end

# ── QR with column pivoting (dgeqpf semantics; exact column-norm recomputation) ──────────────────
# Factors A (m×nc) in place: R above/on the diagonal (real diagonal), reflector tails below,
# tau[i] the reflector scalars, jpvt the pivot map (A_orig[:, jpvt] = Q·R).
function _ggs_geqpf!(A::AbstractMatrix{T}, tau::AbstractVector{T}, jpvt::Vector{Int}) where {T}
    mm, nn = size(A)
    @inbounds for j in 1:nn
        jpvt[j] = j
    end
    kk = min(mm, nn)
    u = Vector{T}(undef, mm)
    @inbounds for i in 1:kk
        jmax = i
        vmax = _ggs_nrm2(view(A, i:mm, i))
        for j in (i + 1):nn
            vj = _ggs_nrm2(view(A, i:mm, j))
            if vj > vmax
                vmax = vj; jmax = j
            end
        end
        if jmax != i
            for r in 1:mm
                A[r, i], A[r, jmax] = A[r, jmax], A[r, i]
            end
            jpvt[i], jpvt[jmax] = jpvt[jmax], jpvt[i]
        end
        τ = _ggs_larfg!(view(A, i:mm, i))
        tau[i] = τ
        if i < nn && τ != zero(T)
            u[1] = one(T)
            for r in (i + 1):mm
                u[r - i + 1] = A[r, i]
            end
            _ggs_larf_left!(view(A, i:mm, (i + 1):nn), view(u, 1:(mm - i + 1)), conj(τ))
        end
    end
    return A
end

# Apply Qᴴ of a factored block (reflectors in Ablk, kk of them) to C from the left (dorm2r 'L','C').
function _ggs_qr_apply_left!(
        Ablk::AbstractMatrix{T}, tau::AbstractVector{T}, kk::Int,
        C::AbstractMatrix{T}
    ) where {T}
    mm = size(Ablk, 1)
    u = Vector{T}(undef, mm)
    @inbounds for i in 1:kk
        u[1] = one(T)
        for r in (i + 1):mm
            u[r - i + 1] = Ablk[r, i]
        end
        _ggs_larf_left!(view(C, i:mm, :), view(u, 1:(mm - i + 1)), conj(tau[i]))
    end
    return C
end

# Form the full mm×mm Q = H(1)···H(kk) from a factored block (dorg2r semantics).
function _ggs_formQ!(
        Qout::AbstractMatrix{T}, Ablk::AbstractMatrix{T}, tau::AbstractVector{T},
        kk::Int
    ) where {T}
    mm = size(Qout, 1)
    fill!(Qout, zero(T))
    @inbounds for i in 1:mm
        Qout[i, i] = one(T)
    end
    u = Vector{T}(undef, mm)
    @inbounds for i in kk:-1:1
        u[1] = one(T)
        for r in (i + 1):mm
            u[r - i + 1] = Ablk[r, i]
        end
        _ggs_larf_left!(view(Qout, i:mm, :), view(u, 1:(mm - i + 1)), tau[i])
    end
    return Qout
end

# ── RQ elimination with co-application (zgerq2 + zunmr2 fused) ───────────────────────────────────
# S (l×n, l ≤ n) ← S·W = [0 T] with T upper triangular (real diagonal); every matrix in `others`
# gets the SAME right-unitary W applied to its leading 1:c column blocks — so any invariant of the
# form X·S or Q-accumulation is preserved exactly.
# RQ of S (S ← [0 T], T upper-triangular real-diagonal), applied to S in place; the per-row right
# reflectors (c, τ, w[1:c]) are RETURNED so the SAME W can be reused on other matrices via
# _ggs_apply_rq_right! — a monomorphic apply (no Vararg — a Vararg{AbstractMatrix} here is NOT
# specialized by the compiler, so `view(X,…)` would dispatch abstractly → --trim can't resolve it).
function _ggs_gerq2!(S::AbstractMatrix{T}) where {T}
    ll, nn = size(S)
    refl = Tuple{Int, T, Vector{T}}[]
    ll == 0 && return refl
    u = Vector{T}(undef, nn)
    @inbounds for i in ll:-1:1
        c = nn - ll + i
        # pivot-first conjugated row: [conj(S[i,c]); conj(S[i,1:c-1])]
        u[1] = conj(S[i, c])
        for j in 1:(c - 1)
            u[1 + j] = conj(S[i, j])
        end
        τ = _ggs_larfg!(view(u, 1:c))
        β = u[1]                       # real value stored in T
        # right reflector H = I - τ·w·wᴴ with w = [u[2:c]; 1]: row i · H = [0…0 β]
        wv = Vector{T}(undef, c)
        for j in 1:(c - 1)
            wv[j] = u[1 + j]
        end
        wv[c] = one(T)
        i > 1 && _ggs_larf_right!(view(S, 1:(i - 1), 1:c), wv, τ)
        for j in 1:(c - 1)
            S[i, j] = zero(T)
        end
        S[i, c] = β
        push!(refl, (c, τ, wv))
    end
    return refl
end
# Apply the stored RQ right-reflectors (from _ggs_gerq2!) to X's leading columns — monomorphic on X.
function _ggs_apply_rq_right!(refl::Vector{Tuple{Int, T, Vector{T}}}, X::AbstractMatrix{T}) where {T}
    @inbounds for (c, τ, wv) in refl
        _ggs_larf_right!(view(X, :, 1:c), wv, τ)
    end
    return X
end

# Permute columns: X ← X[:, perm].
function _ggs_permcols!(X::AbstractMatrix{T}, perm::Vector{Int}) where {T}
    Xc = X[:, perm]
    copyto!(X, Xc)
    return X
end

# ── Givens rotations (dlartg/zlartg) and zrot application ────────────────────────────────────────
function _ggs_lartg(f::T, g::T) where {T <: Real}
    if g == 0
        return (one(T), zero(T), f)
    elseif f == 0
        return (zero(T), g >= 0 ? one(T) : -one(T), abs(g))
    else
        r = copysign(hypot(f, g), f)
        return (f / r, g / r, r)
    end
end

function _ggs_lartg(f::Complex{RT}, g::Complex{RT}) where {RT <: Real}
    T = Complex{RT}
    if g == 0
        return (one(RT), zero(T), f)
    elseif f == 0
        d = abs(g)
        return (zero(RT), conj(g) / d, T(d))
    else
        f1 = abs(f); g1 = abs(g)
        fg = hypot(f1, g1)
        sgnf = f / f1
        return (f1 / fg, sgnf * conj(g) / fg, sgnf * fg)
    end
end

# zrot: x := c·x + s·y ; y := c·y - conj(s)·x₀  (c real, s possibly complex).
function _ggs_rot!(x::AbstractVector{T}, y::AbstractVector{T}, c::Real, s) where {T}
    # every call site passes same-length views (see call sites below) — plain `eachindex(x)`,
    # not the two-array form, so the compiled DimensionMismatch/join/AnnotatedString error path
    # (trim-incompatible: juliac --trim can't verify Base.join's invoke_in_world) never gets built.
    @inbounds for i in eachindex(x)
        xi = x[i]; yi = y[i]
        x[i] = c * xi + s * yi
        y[i] = c * yi - conj(s) * xi
    end
    return nothing
end

# ── dlas2: singular values of a 2×2 upper-triangular [f g; 0 h] ──────────────────────────────────
function _ggs_las2(f::T, g::T, h::T) where {T <: Real}
    fa = abs(f); ga = abs(g); ha = abs(h)
    fhmn = min(fa, ha); fhmx = max(fa, ha)
    if fhmn == 0
        ssmin = zero(T)
        ssmax = fhmx == 0 ? ga :
            max(fhmx, ga) * sqrt(one(T) + (min(fhmx, ga) / max(fhmx, ga))^2)
        return (ssmin, ssmax)
    end
    if ga < fhmx
        as = one(T) + fhmn / fhmx
        at = (fhmx - fhmn) / fhmx
        au = (ga / fhmx)^2
        c = 2 / (sqrt(as * as + au) + sqrt(at * at + au))
        return (fhmn * c, fhmx / c)
    end
    au = fhmx / ga
    if au == 0
        return ((fhmn * fhmx) / ga, ga)
    end
    as = one(T) + fhmn / fhmx
    at = (fhmx - fhmn) / fhmx
    c = one(T) / (sqrt(one(T) + (as * au)^2) + sqrt(one(T) + (at * au)^2))
    ssmin = (fhmn * c) * au
    ssmin += ssmin
    return (ssmin, ga / (c + c))
end

# ── dlasv2: full SVD of a 2×2 upper-triangular [f g; 0 h] with rotations (literal port) ──────────
function _ggs_lasv2(f::T, g::T, h::T) where {T <: Real}
    ft = f; fa = abs(ft); ht = h; ha = abs(h)
    pmax = 1
    swap = ha > fa
    if swap
        pmax = 3
        ft, ht = ht, ft
        fa, ha = ha, fa
    end
    gt = g; ga = abs(gt)
    clt = one(T); crt = one(T); slt = zero(T); srt = zero(T)
    ssmin = ha; ssmax = fa
    if ga != 0
        gasmal = true
        if ga > fa
            pmax = 2
            if fa / ga < eps(T) / 2          # DLAMCH('EPS') = relative machine epsilon
                gasmal = false
                ssmax = ga
                ssmin = ha > 1 ? fa / (ga / ha) : (fa / ga) * ha
                clt = one(T); slt = ht / gt
                srt = one(T); crt = ft / gt
            end
        end
        if gasmal
            d = fa - ha
            el = d == fa ? one(T) : d / fa   # 0 ≤ el ≤ 1
            em = gt / ft                     # |em| ≤ 1/eps
            t = 2 - el
            mm2 = em * em
            tt = t * t
            s_ = sqrt(tt + mm2)
            r_ = el == 0 ? abs(em) : sqrt(el * el + mm2)
            a_ = (s_ + r_) / 2
            ssmin = ha / a_
            ssmax = fa * a_
            if mm2 == 0
                if el == 0
                    t = copysign(T(2), ft) * copysign(one(T), gt)
                else
                    t = gt / copysign(d, ft) + em / t
                end
            else
                t = (em / (s_ + t) + em / (r_ + el)) * (one(T) + a_)
            end
            el = sqrt(t * t + 4)
            crt = 2 / el
            srt = t / el
            clt = (crt + srt * em) / a_
            slt = (ht / ft) * srt / a_
        end
    end
    local csl, snl, csr, snr
    if swap
        csl = srt; snl = crt; csr = slt; snr = clt
    else
        csl = clt; snl = slt; csr = crt; snr = srt
    end
    tsign = pmax == 1 ? copysign(one(T), csr) * copysign(one(T), csl) * copysign(one(T), f) :
        pmax == 2 ? copysign(one(T), snr) * copysign(one(T), csl) * copysign(one(T), g) :
        copysign(one(T), snr) * copysign(one(T), snl) * copysign(one(T), h)
    ssmax = copysign(ssmax, tsign)
    ssmin = copysign(ssmin, tsign * copysign(one(T), f) * copysign(one(T), h))
    return (ssmin, ssmax, snr, csr, snl, csl)
end

# ── dlags2 (real): rotations for the 2×2 triangular pair (literal port) ──────────────────────────
# Computes U, V, Q (2×2 rotations) such that Uᵀ·[a1 a2; 0 a3]·Q and Vᵀ·[b1 b2; 0 b3]·Q stay
# triangular (upper=true; lower-triangular reading for upper=false).
function _ggs_lags2(upper::Bool, a1::T, a2::T, a3::T, b1::T, b2::T, b3::T) where {T <: Real}
    local csu, snu, csv, snv, csq, snq
    if upper
        a_ = a1 * b3
        d_ = a3 * b1
        b_ = a2 * b1 - a1 * b2
        _, _, snr, csr, snl, csl = _ggs_lasv2(a_, b_, d_)
        if abs(csl) >= abs(snl) || abs(csr) >= abs(snr)
            ua11r = csl * a1
            ua12 = csl * a2 + snl * a3
            vb11r = csr * b1
            vb12 = csr * b2 + snr * b3
            aua12 = abs(csl) * abs(a2) + abs(snl) * abs(a3)
            avb12 = abs(csr) * abs(b2) + abs(snr) * abs(b3)
            if (abs(ua11r) + abs(ua12)) != 0 &&
                    aua12 / (abs(ua11r) + abs(ua12)) <= avb12 / (abs(vb11r) + abs(vb12))
                csq, snq, _ = _ggs_lartg(-ua11r, ua12)
            else
                csq, snq, _ = _ggs_lartg(-vb11r, vb12)
            end
            csu = csl; snu = -snl
            csv = csr; snv = -snr
        else
            ua21 = -snl * a1
            ua22 = -snl * a2 + csl * a3
            vb21 = -snr * b1
            vb22 = -snr * b2 + csr * b3
            aua22 = abs(snl) * abs(a2) + abs(csl) * abs(a3)
            avb22 = abs(snr) * abs(b2) + abs(csr) * abs(b3)
            if (abs(ua21) + abs(ua22)) != 0 &&
                    aua22 / (abs(ua21) + abs(ua22)) <= avb22 / (abs(vb21) + abs(vb22))
                csq, snq, _ = _ggs_lartg(-ua21, ua22)
            else
                csq, snq, _ = _ggs_lartg(-vb21, vb22)
            end
            csu = snl; snu = csl
            csv = snr; snv = csr
        end
    else
        a_ = a1 * b3
        d_ = a3 * b1
        c_ = a2 * b3 - a3 * b2
        _, _, snr, csr, snl, csl = _ggs_lasv2(a_, c_, d_)
        if abs(csr) >= abs(snr) || abs(csl) >= abs(snl)
            ua21 = -snr * a1 + csr * a2
            ua22r = csr * a3
            vb21 = -snl * b1 + csl * b2
            vb22r = csl * b3
            aua21 = abs(snr) * abs(a1) + abs(csr) * abs(a2)
            avb21 = abs(snl) * abs(b1) + abs(csl) * abs(b2)
            if (abs(ua21) + abs(ua22r)) != 0 &&
                    aua21 / (abs(ua21) + abs(ua22r)) <= avb21 / (abs(vb21) + abs(vb22r))
                csq, snq, _ = _ggs_lartg(ua22r, ua21)
            else
                csq, snq, _ = _ggs_lartg(vb22r, vb21)
            end
            csu = csr; snu = -snr
            csv = csl; snv = -snl
        else
            ua11 = csr * a1 + snr * a2
            ua12 = snr * a3
            vb11 = csl * b1 + snl * b2
            vb12 = snl * b3
            aua11 = abs(csr) * abs(a1) + abs(snr) * abs(a2)
            avb11 = abs(csl) * abs(b1) + abs(snl) * abs(b2)
            if (abs(ua11) + abs(ua12)) != 0 &&
                    aua11 / (abs(ua11) + abs(ua12)) <= avb11 / (abs(vb11) + abs(vb12))
                csq, snq, _ = _ggs_lartg(ua12, ua11)
            else
                csq, snq, _ = _ggs_lartg(vb12, vb11)
            end
            csu = snr; snu = csr
            csv = snl; snv = csl
        end
    end
    return (csu, snu, csv, snv, csq, snq)
end

# ── zlags2 (complex): rotations for the 2×2 triangular pair, real diagonals (literal port) ───────
_ggs_cabs1(x::Complex) = abs(real(x)) + abs(imag(x))
function _ggs_lags2(
        upper::Bool, a1::RT, a2::Complex{RT}, a3::RT,
        b1::RT, b2::Complex{RT}, b3::RT
    ) where {RT <: Real}
    T = Complex{RT}
    local csu::RT, csv::RT, csq::RT
    local snu::T, snv::T, snq::T
    if upper
        a_ = a1 * b3
        d_ = a3 * b1
        b_ = a2 * b1 - a1 * b2
        fb = abs(b_)
        d1 = fb != 0 ? b_ / fb : one(T)
        _, _, snr, csr, snl, csl = _ggs_lasv2(a_, fb, d_)
        if abs(csl) >= abs(snl) || abs(csr) >= abs(snr)
            ua11r = csl * a1
            ua12 = csl * a2 + d1 * snl * a3
            vb11r = csr * b1
            vb12 = csr * b2 + d1 * snr * b3
            aua12 = abs(csl) * _ggs_cabs1(a2) + abs(snl) * abs(a3)
            avb12 = abs(csr) * _ggs_cabs1(b2) + abs(snr) * abs(b3)
            if (abs(ua11r) + _ggs_cabs1(ua12)) == 0
                csq, snq, _ = _ggs_lartg(-T(vb11r), conj(vb12))
            elseif (abs(vb11r) + _ggs_cabs1(vb12)) == 0
                csq, snq, _ = _ggs_lartg(-T(ua11r), conj(ua12))
            elseif aua12 / (abs(ua11r) + _ggs_cabs1(ua12)) <=
                    avb12 / (abs(vb11r) + _ggs_cabs1(vb12))
                csq, snq, _ = _ggs_lartg(-T(ua11r), conj(ua12))
            else
                csq, snq, _ = _ggs_lartg(-T(vb11r), conj(vb12))
            end
            csu = csl; snu = -d1 * snl
            csv = csr; snv = -d1 * snr
        else
            ua21 = -conj(d1) * snl * a1
            ua22 = -conj(d1) * snl * a2 + csl * a3
            vb21 = -conj(d1) * snr * b1
            vb22 = -conj(d1) * snr * b2 + csr * b3
            aua22 = abs(snl) * _ggs_cabs1(a2) + abs(csl) * abs(a3)
            avb22 = abs(snr) * _ggs_cabs1(b2) + abs(csr) * abs(b3)
            if (_ggs_cabs1(ua21) + _ggs_cabs1(ua22)) == 0
                csq, snq, _ = _ggs_lartg(-conj(vb21), conj(vb22))
            elseif (_ggs_cabs1(vb21) + abs(vb22)) == 0
                csq, snq, _ = _ggs_lartg(-conj(ua21), conj(ua22))
            elseif aua22 / (_ggs_cabs1(ua21) + _ggs_cabs1(ua22)) <=
                    avb22 / (_ggs_cabs1(vb21) + _ggs_cabs1(vb22))
                csq, snq, _ = _ggs_lartg(-conj(ua21), conj(ua22))
            else
                csq, snq, _ = _ggs_lartg(-conj(vb21), conj(vb22))
            end
            csu = snl; snu = d1 * csl
            csv = snr; snv = d1 * csr
        end
    else
        a_ = a1 * b3
        d_ = a3 * b1
        c_ = a2 * b3 - a3 * b2
        fc = abs(c_)
        d1 = fc != 0 ? c_ / fc : one(T)
        _, _, snr, csr, snl, csl = _ggs_lasv2(a_, fc, d_)
        if abs(csr) >= abs(snr) || abs(csl) >= abs(snl)
            ua21 = -d1 * snr * a1 + csr * a2
            ua22r = csr * a3
            vb21 = -d1 * snl * b1 + csl * b2
            vb22r = csl * b3
            aua21 = abs(snr) * abs(a1) + abs(csr) * _ggs_cabs1(a2)
            avb21 = abs(snl) * abs(b1) + abs(csl) * _ggs_cabs1(b2)
            if (_ggs_cabs1(ua21) + abs(ua22r)) == 0
                csq, snq, _ = _ggs_lartg(T(vb22r), vb21)
            elseif (_ggs_cabs1(vb21) + abs(vb22r)) == 0
                csq, snq, _ = _ggs_lartg(T(ua22r), ua21)
            elseif aua21 / (_ggs_cabs1(ua21) + abs(ua22r)) <=
                    avb21 / (_ggs_cabs1(vb21) + abs(vb22r))
                csq, snq, _ = _ggs_lartg(T(ua22r), ua21)
            else
                csq, snq, _ = _ggs_lartg(T(vb22r), vb21)
            end
            csu = csr; snu = -conj(d1) * snr
            csv = csl; snv = -conj(d1) * snl
        else
            ua11 = csr * a1 + conj(d1) * snr * a2
            ua12 = conj(d1) * snr * a3
            vb11 = csl * b1 + conj(d1) * snl * b2
            vb12 = conj(d1) * snl * b3
            aua11 = abs(csr) * abs(a1) + abs(snr) * _ggs_cabs1(a2)
            avb11 = abs(csl) * abs(b1) + abs(snl) * _ggs_cabs1(b2)
            if (_ggs_cabs1(ua11) + _ggs_cabs1(ua12)) == 0
                csq, snq, _ = _ggs_lartg(vb12, vb11)
            elseif (_ggs_cabs1(vb11) + _ggs_cabs1(vb12)) == 0
                csq, snq, _ = _ggs_lartg(ua12, ua11)
            elseif aua11 / (_ggs_cabs1(ua11) + _ggs_cabs1(ua12)) <=
                    avb11 / (_ggs_cabs1(vb11) + _ggs_cabs1(vb12))
                csq, snq, _ = _ggs_lartg(ua12, ua11)
            else
                csq, snq, _ = _ggs_lartg(vb12, vb11)
            end
            csu = snr; snu = conj(d1) * csr
            csv = snl; snv = conj(d1) * csl
        end
    end
    return (csu, snu, csv, snv, csq, snq)
end

# ── dlapll/zlapll: smallest singular value of the 2-column matrix [x y] (destroys x, y) ──────────
function _ggs_lapll!(x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    RT = real(T)
    nn = length(x)
    nn <= 1 && return zero(RT)
    τ = _ggs_larfg!(x)
    a11 = x[1]
    w = y[1]
    @inbounds for i in 2:nn
        w += conj(x[i]) * y[i]
    end
    c = -conj(τ) * w
    y[1] += c
    @inbounds for i in 2:nn
        y[i] += c * x[i]
    end
    _ggs_larfg!(view(y, 2:nn))
    a12 = y[1]
    a22 = y[2]
    ssmin, _ = _ggs_las2(abs(a11), abs(a12), abs(a22))
    return ssmin
end

# ── dggsvp/zggsvp: rank-revealing preprocessing ──────────────────────────────────────────────────
# Reduces (A, B) in place to the LAPACK ggsvp block-triangular form, accumulating U, V, Q when
# wanted; returns (k, l). Tolerances tola/tolb define the effective ranks exactly as LAPACK does.
function _ggs_ggsvp!(
        wantu::Bool, wantv::Bool, wantq::Bool, A::AbstractMatrix{T},
        B::AbstractMatrix{T}, tola::Real, tolb::Real, U::AbstractMatrix{T},
        V::AbstractMatrix{T}, Q::AbstractMatrix{T}
    ) where {T}
    m, n = size(A)
    p = size(B, 1)

    # 1) QR with column pivoting of B: B·P = V·[S11 S12; 0 0]; update A := A·P.
    kb = min(p, n)
    taub = Vector{T}(undef, kb)
    jpvt = Vector{Int}(undef, n)
    _ggs_geqpf!(B, taub, jpvt)
    _ggs_permcols!(A, jpvt)
    l = 0
    @inbounds for i in 1:kb
        abs(B[i, i]) > tolb && (l += 1)
    end
    wantv && _ggs_formQ!(V, B, taub, kb)
    # clean up B: strictly-lower of rows 1:l, and rows l+1:p entirely.
    @inbounds for j in 1:(l - 1), i in (j + 1):l
        B[i, j] = zero(T)
    end
    @inbounds for j in 1:n, i in (l + 1):p
        B[i, j] = zero(T)
    end
    if wantq
        fill!(Q, zero(T))
        @inbounds for j in 1:n
            Q[jpvt[j], j] = one(T)         # Q = P
        end
    end

    # 2) RQ of (S11 S12) (l×n): → [0 T]·Z; update A := A·Zᴴ, Q := Q·Zᴴ.
    if l > 0 && n != l
        Brows = view(B, 1:l, 1:n)
        refl = _ggs_gerq2!(Brows)
        _ggs_apply_rq_right!(refl, A)
        wantq && _ggs_apply_rq_right!(refl, Q)
        # clean up B: leading zero block and strictly-lower of the trailing l×l T.
        @inbounds for j in 1:(n - l), i in 1:l
            B[i, j] = zero(T)
        end
        @inbounds for j in (n - l + 1):n, i in (j - n + l + 1):l
            B[i, j] = zero(T)
        end
    end

    # 3) QR with column pivoting of A11 = A(:, 1:n-l): reveals k; A12 := U1ᴴ·A12; U := U1.
    nl = n - l
    k = 0
    if nl > 0
        A11 = view(A, 1:m, 1:nl)
        ka = min(m, nl)
        taua = Vector{T}(undef, ka)
        jp2 = Vector{Int}(undef, nl)
        _ggs_geqpf!(A11, taua, jp2)
        @inbounds for i in 1:ka
            abs(A[i, i]) > tola && (k += 1)
        end
        l > 0 && _ggs_qr_apply_left!(A11, taua, ka, view(A, 1:m, (nl + 1):n))
        wantu && _ggs_formQ!(U, A11, taua, ka)
        wantq && _ggs_permcols!(view(Q, :, 1:nl), jp2)
        # clean up A: strictly-lower of A(1:k,1:k); A(k+1:m, 1:nl) = 0.
        @inbounds for j in 1:(k - 1), i in (j + 1):k
            A[i, j] = zero(T)
        end
        @inbounds for j in 1:nl, i in (k + 1):m
            A[i, j] = zero(T)
        end
    elseif wantu
        fill!(U, zero(T))
        @inbounds for i in 1:m
            U[i, i] = one(T)
        end
    end

    # 4) RQ of (T11 T12) = A(1:k, 1:nl): → [0 T12]·Z1; Q(:,1:nl) := Q(:,1:nl)·Z1ᴴ.
    if nl > k && k > 0
        Ak = view(A, 1:k, 1:nl)
        refl = _ggs_gerq2!(Ak)
        wantq && _ggs_apply_rq_right!(refl, view(Q, :, 1:nl))
        @inbounds for j in 1:(nl - k), i in 1:k
            A[i, j] = zero(T)
        end
        @inbounds for j in (nl - k + 1):nl, i in (j - nl + k + 1):k
            A[i, j] = zero(T)
        end
    end

    # 5) QR of A(k+1:m, nl+1:n); U(:,k+1:m) := U(:,k+1:m)·U2.
    if m > k && l > 0
        kk2 = min(m - k, l)
        u = Vector{T}(undef, m)
        @inbounds for i in 1:kk2
            τ = _ggs_larfg!(view(A, (k + i):m, nl + i))
            u[1] = one(T)
            for r in (k + i + 1):m
                u[r - k - i + 1] = A[r, nl + i]
            end
            ulen = m - k - i + 1
            i < l && _ggs_larf_left!(view(A, (k + i):m, (nl + i + 1):n), view(u, 1:ulen), conj(τ))
            wantu && _ggs_larf_right!(view(U, :, (k + i):m), view(u, 1:ulen), τ)
        end
        @inbounds for j in (nl + 1):n, i in (j - n + k + l + 1):m
            A[i, j] = zero(T)
        end
    end

    return k, l
end

# ── dtgsja/ztgsja: Kogbetliantz cyclic Jacobi on the preprocessed pair + α/β/R endgame ────────────
# Returns 0 on convergence, 1 if MAXIT cycles did not converge (matching LAPACK's INFO).
function _ggs_tgsja!(
        wantu::Bool, wantv::Bool, wantq::Bool, m::Int, p::Int, n::Int, k::Int, l::Int,
        A::AbstractMatrix{T}, B::AbstractMatrix{T}, tola::Real, tolb::Real,
        alpha::AbstractVector, beta::AbstractVector, U::AbstractMatrix{T},
        V::AbstractMatrix{T}, Q::AbstractMatrix{T}
    ) where {T}
    RT = real(T)
    maxit = 40
    wx = Vector{T}(undef, max(l, 1))
    wy = Vector{T}(undef, max(l, 1))
    upper = false
    converged = false
    @inbounds for _ in 1:maxit
        upper = !upper
        for i in 1:(l - 1), j in (i + 1):l
            a1 = zero(RT); a3 = zero(RT)
            a2 = zero(T); b2 = zero(T)
            k + i <= m && (a1 = real(A[k + i, n - l + i]))
            k + j <= m && (a3 = real(A[k + j, n - l + j]))
            b1 = real(B[i, n - l + i])
            b3 = real(B[j, n - l + j])
            if upper
                k + i <= m && (a2 = A[k + i, n - l + j])
                b2 = B[i, n - l + j]
            else
                k + j <= m && (a2 = A[k + j, n - l + i])
                b2 = B[j, n - l + i]
            end
            csu, snu, csv, snv, csq, snq = _ggs_lags2(upper, a1, a2, a3, b1, b2, b3)
            # rows of A (Uᴴ·A) and B (Vᴴ·B)
            k + j <= m && _ggs_rot!(
                view(A, k + j, (n - l + 1):n), view(A, k + i, (n - l + 1):n),
                csu, conj(snu)
            )
            _ggs_rot!(view(B, j, (n - l + 1):n), view(B, i, (n - l + 1):n), csv, conj(snv))
            # columns of A and B (·Q)
            mkl = min(k + l, m)
            mkl >= 1 && _ggs_rot!(view(A, 1:mkl, n - l + j), view(A, 1:mkl, n - l + i), csq, snq)
            _ggs_rot!(view(B, 1:l, n - l + j), view(B, 1:l, n - l + i), csq, snq)
            if upper
                k + i <= m && (A[k + i, n - l + j] = zero(T))
                B[i, n - l + j] = zero(T)
            else
                k + j <= m && (A[k + j, n - l + i] = zero(T))
                B[j, n - l + i] = zero(T)
            end
            if T <: Complex                    # keep the working diagonals real (ztgsja)
                k + i <= m && (A[k + i, n - l + i] = real(A[k + i, n - l + i]))
                k + j <= m && (A[k + j, n - l + j] = real(A[k + j, n - l + j]))
                B[i, n - l + i] = real(B[i, n - l + i])
                B[j, n - l + j] = real(B[j, n - l + j])
            end
            wantu && k + j <= m && _ggs_rot!(view(U, :, k + j), view(U, :, k + i), csu, snu)
            wantv && _ggs_rot!(view(V, :, j), view(V, :, i), csv, snv)
            wantq && _ggs_rot!(view(Q, :, n - l + j), view(Q, :, n - l + i), csq, snq)
        end
        if !upper
            # convergence: parallelism of corresponding rows of A and B
            err = zero(RT)
            for i in 1:min(l, m - k)
                len = l - i + 1
                for t in 1:len
                    wx[t] = A[k + i, n - l + i + t - 1]
                    wy[t] = B[i, n - l + i + t - 1]
                end
                err = max(err, _ggs_lapll!(view(wx, 1:len), view(wy, 1:len)))
            end
            if abs(err) <= min(tola, tolb)
                converged = true
                break
            end
        end
    end
    converged || return 1

    # endgame: α, β and the triangular R assembled into A (and B for the deficient-m rows)
    @inbounds for i in 1:k
        alpha[i] = one(RT)
        beta[i] = zero(RT)
    end
    @inbounds for i in 1:min(l, m - k)
        a1 = real(A[k + i, n - l + i])
        b1 = real(B[i, n - l + i])
        gamma = b1 / a1
        if isfinite(gamma)
            if gamma < 0
                for t in (n - l + i):n
                    B[i, t] = -B[i, t]
                end
                if wantv
                    for r in 1:p
                        V[r, i] = -V[r, i]
                    end
                end
            end
            bk, ak, _ = _ggs_lartg(abs(gamma), one(RT))
            beta[k + i] = bk
            alpha[k + i] = ak
            if alpha[k + i] >= beta[k + i]
                s = one(RT) / alpha[k + i]
                for t in (n - l + i):n
                    A[k + i, t] *= s
                end
            else
                s = one(RT) / beta[k + i]
                for t in (n - l + i):n
                    B[i, t] *= s
                    A[k + i, t] = B[i, t]
                end
            end
        else
            alpha[k + i] = zero(RT)
            beta[k + i] = one(RT)
            for t in (n - l + i):n
                A[k + i, t] = B[i, t]
            end
        end
    end
    @inbounds for i in (m + 1):(k + l)
        alpha[i] = zero(RT)
        beta[i] = one(RT)
    end
    @inbounds for i in (k + l + 1):n
        alpha[i] = zero(RT)
        beta[i] = zero(RT)
    end
    return 0
end

"""
    ggsvd!(jobu, jobv, jobq, A, B) -> (U, V, Q, alpha, beta, k, l, R)

Generalized singular value decomposition of the pair `(A, B)` (`A` m×n, `B` p×n), matching the
classic LAPACK `{s,d,c,z}ggsvd` driver: finds unitary `U` (m×m), `V` (p×p), `Q` (n×n), integers
`k`, `l` (`k+l` = effective rank of `[A; B]`, `l` = effective rank of `B`) and real vectors
`alpha`, `beta` (length n) such that `Uᴴ·A·Q = D1·[0 R]` and `Vᴴ·B·Q = D2·[0 R]` with `R`
`(k+l)×(k+l)` upper triangular (see the file header for the `D1`/`D2` block layouts). Handles
rank-deficient `A`, `B`, and `[A; B]`. `jobu`/`jobv`/`jobq` are `'U'`/`'V'`/`'Q'` to compute the
corresponding matrix or `'N'` to skip it (skipped outputs are returned as zero matrices).
`A` and `B` are overwritten (LAPACK semantics). Generalized singular values are
`alpha[k+1:k+l] ./ beta[k+1:k+l]`. Element types: `Float32`/`Float64`/`ComplexF32`/`ComplexF64`;
`alpha`/`beta` are always `real(T)`.
"""
function ggsvd!(
        jobu::AbstractChar, jobv::AbstractChar, jobq::AbstractChar,
        A::AbstractMatrix{T}, B::AbstractMatrix{T}
    ) where {T}
    (jobu == 'U' || jobu == 'N') ||
        throw(ArgumentError("ggsvd!: jobu must be 'U' or 'N'"))
    (jobv == 'V' || jobv == 'N') ||
        throw(ArgumentError("ggsvd!: jobv must be 'V' or 'N'"))
    (jobq == 'Q' || jobq == 'N') ||
        throw(ArgumentError("ggsvd!: jobq must be 'Q' or 'N'"))
    m, n = size(A)
    p = size(B, 1)
    size(B, 2) == n || throw(DimensionMismatch("ggsvd!: A and B must have the same number of columns"))
    wantu = jobu == 'U'
    wantv = jobv == 'V'
    wantq = jobq == 'Q'
    RT = real(T)

    # dggsvd tolerances: these define the effective ranks k and l.
    anorm = _ggs_norm1(A)
    bnorm = _ggs_norm1(B)
    tola = max(m, n) * max(anorm, floatmin(RT)) * eps(RT)
    tolb = max(p, n) * max(bnorm, floatmin(RT)) * eps(RT)

    U = zeros(T, m, m)
    V = zeros(T, p, p)
    Q = zeros(T, n, n)
    k, l = _ggs_ggsvp!(wantu, wantv, wantq, A, B, tola, tolb, U, V, Q)

    alpha = Vector{RT}(undef, n)
    beta = Vector{RT}(undef, n)
    info = _ggs_tgsja!(wantu, wantv, wantq, m, p, n, k, l, A, B, tola, tolb, alpha, beta, U, V, Q)
    info == 0 || throw(ErrorException("ggsvd!: the Jacobi-type procedure failed to converge"))

    # R extraction — the layout LinearAlgebra's LAPACK.ggsvd! wrapper returns.
    kl = k + l
    R = zeros(T, kl, kl)
    if m - k - l >= 0
        @inbounds for j in 1:kl, i in 1:j
            R[i, j] = A[i, n - kl + j]
        end
    else
        @inbounds for j in 1:kl, i in 1:min(j, m)
            R[i, j] = A[i, n - kl + j]
        end
        @inbounds for j in 1:kl, i in (m + 1):j
            R[i, j] = B[i - k, n - kl + j]
        end
    end
    return U, V, Q, alpha, beta, k, l, R
end
