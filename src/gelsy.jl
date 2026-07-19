# Rank-deficient least squares via COMPLETE ORTHOGONAL factorization (LAPACK gelsy). Solves
# min‖A·X − B‖₂ for a possibly rank-deficient A, returning the MINIMUM-NORM solution over the
# effective (numerical) rank determined from `rcond`. Composed from PureBLAS's own blocks: pivoted
# QR (`geqp3!`, geqp3.jl), the Householder generator `_larfg!` (svd.jl), the Qᴴ back-transform
# `_apply_Qh!` (gels.jl), and `trsm!` (level3.jl). The RZ (upper-trapezoidal → upper-triangular)
# reduction `tzrzf!` and its apply `ormrz!`/`unmrz!` live here — they are the "complete orthogonal"
# half that gelsy needs. Generic over Float32/Float64/ComplexF32/ComplexF64.
#
# Mirrors Reference-LAPACK exactly: rank via the incremental condition estimator dlaic1/zlaic1
# (`_laic1`), the RZ factorization via dlatrz/dtzrzf (`tzrzf!`) + dlarz (`ormrz!`), and the driver
# sequence of dgelsy/zgelsy (geqp3 → rank → tzrzf → Qᴴ·B → T11-solve → Zᴴ·B → un-pivot).

# ── incremental condition estimator (LAPACK dlaic1 / zlaic1, unified) ───────────────────────────────
# One step of incremental 2-norm condition estimation. x (‖x‖=1) is an approximate singular vector of
# a j×j lower-triangular L with ‖L·x‖ = sest; [wᴴ γ] is the (j+1)-th row of the augmented triangular
# matrix. Returns (sestpr, s, c): the updated singular-value estimate and coefficients giving the new
# approximate vector [s·x; c]. job=1 → largest singular value, job=2 → smallest. The complex (zlaic1)
# formulation subsumes the real one (conj is the identity on ℝ, alpha=Σconj(x)·w = ddot on ℝ / zdotc
# on ℂ), so ONE generic body covers s/d/c/z.
@inline function _laic1(job::Int, x::AbstractVector{T}, sest::R, w::AbstractVector{T},
        gamma::T) where {T<:BlasFloat, R<:Real}
    ε = eps(R) / 2                                       # DLAMCH('Epsilon') — the rounding unit
    j = length(x)
    alpha = zero(T)
    @inbounds for k in 1:j; alpha += conj(x[k]) * w[k]; end
    absalp = abs(alpha); absgam = abs(gamma); absest = abs(sest)
    if job == 1
        # ---- estimating the LARGEST singular value ----
        if sest == zero(R)
            s1 = max(absgam, absalp)
            if s1 == zero(R)
                return zero(R), zero(T), one(T)
            end
            s = alpha / s1; c = gamma / s1; tmp = sqrt(abs2(s) + abs2(c))
            return R(s1 * tmp), s / tmp, c / tmp
        elseif absgam <= ε * absest
            tmp = max(absest, absalp); s1 = absest / tmp; s2 = absalp / tmp
            return R(tmp * sqrt(s1 * s1 + s2 * s2)), one(T), zero(T)
        elseif absalp <= ε * absest
            return absgam <= absest ? (R(absest), one(T), zero(T)) : (R(absgam), zero(T), one(T))
        elseif absest <= ε * absalp || absest <= ε * absgam
            if absgam <= absalp
                tmp = absgam / absalp; scl = sqrt(one(R) + tmp * tmp)
                return R(absalp * scl), (alpha / absalp) / scl, (gamma / absalp) / scl
            else
                tmp = absalp / absgam; scl = sqrt(one(R) + tmp * tmp)
                return R(absgam * scl), (alpha / absgam) / scl, (gamma / absgam) / scl
            end
        else                                             # normal case
            zeta1 = absalp / absest; zeta2 = absgam / absest
            b = (one(R) - zeta1 * zeta1 - zeta2 * zeta2) * R(0.5); cr = zeta1 * zeta1
            t = b > zero(R) ? cr / (b + sqrt(b * b + cr)) : sqrt(b * b + cr) - b
            sine = -(alpha / absest) / t; cosine = -(gamma / absest) / (one(R) + t)
            tmp = sqrt(abs2(sine) + abs2(cosine))
            return R(sqrt(t + one(R)) * absest), sine / tmp, cosine / tmp
        end
    else
        # ---- estimating the SMALLEST singular value (job==2) ----
        if sest == zero(R)
            if max(absgam, absalp) == zero(R)
                sine = one(T); cosine = zero(T)
            else
                sine = -conj(gamma); cosine = conj(alpha)
            end
            s1 = max(abs(sine), abs(cosine)); s = sine / s1; c = cosine / s1
            tmp = sqrt(abs2(s) + abs2(c))
            return zero(R), s / tmp, c / tmp
        elseif absgam <= ε * absest
            return R(absgam), zero(T), one(T)
        elseif absalp <= ε * absest
            return absgam <= absest ? (R(absgam), zero(T), one(T)) : (R(absest), one(T), zero(T))
        elseif absest <= ε * absalp || absest <= ε * absgam
            if absgam <= absalp
                tmp = absgam / absalp; scl = sqrt(one(R) + tmp * tmp)
                return R(absest * (tmp / scl)), -(conj(gamma) / absalp) / scl, (conj(alpha) / absalp) / scl
            else
                tmp = absalp / absgam; scl = sqrt(one(R) + tmp * tmp)
                return R(absest / scl), -(conj(gamma) / absgam) / scl, (conj(alpha) / absgam) / scl
            end
        else                                             # normal case
            zeta1 = absalp / absest; zeta2 = absgam / absest
            norma = max(one(R) + zeta1 * zeta1 + zeta1 * zeta2, zeta1 * zeta2 + zeta2 * zeta2)
            test = one(R) + R(2) * (zeta1 - zeta2) * (zeta1 + zeta2)
            if test >= zero(R)                           # root near zero — compute directly
                b = (zeta1 * zeta1 + zeta2 * zeta2 + one(R)) * R(0.5); cr = zeta2 * zeta2
                t = cr / (b + sqrt(abs(b * b - cr)))
                sine = (alpha / absest) / (one(R) - t); cosine = -(gamma / absest) / t
                sestpr = sqrt(t + R(4) * ε * ε * norma) * absest
            else                                         # root near one — shift
                b = (zeta2 * zeta2 + zeta1 * zeta1 - one(R)) * R(0.5); cr = zeta1 * zeta1
                t = b >= zero(R) ? -cr / (b + sqrt(b * b + cr)) : b - sqrt(b * b + cr)
                sine = -(alpha / absest) / t; cosine = -(gamma / absest) / (one(R) + t)
                sestpr = sqrt(one(R) + t + R(4) * ε * ε * norma) * absest
            end
            tmp = sqrt(abs2(sine) + abs2(cosine))
            return R(sestpr), sine / tmp, cosine / tmp
        end
    end
end

# ── RZ factorization (LAPACK dtzrzf/ztzrzf via the unblocked dlatrz/zlatrz core) ────────────────────
# Reduce an m×n (m ≤ n) UPPER-TRAPEZOIDAL A to UPPER-TRIANGULAR by orthogonal transforms from the
# right:  A = [ R  0 ]·Z,  R m×m upper triangular, Z n×n orthogonal. On exit A[1:m,1:m] is R and
# A[i, m+1:n] holds the essential part of the i-th right reflector Z_i (τ in `tau`, LAPACK convention).
# Numerics identical to dtzrzf (its blocked trailing update is only a perf refinement of this core).
function tzrzf!(A::AbstractMatrix{T}, tau::AbstractVector{T}) where {T<:BlasFloat}
    m, n = size(A)
    m <= n || throw(ArgumentError("tzrzf!: requires m ≤ n (got $m×$n)"))
    length(tau) >= m || throw(DimensionMismatch("tzrzf!: length(tau) < m=$m"))
    L = n - m
    m == 0 && return A, tau
    if L == 0                                            # already upper triangular
        @inbounds for i in 1:m; tau[i] = zero(T); end
        return A, tau
    end
    # zlatrz conjugation dance (identity on ℝ, so the real path is dlatrz unchanged): conjugate the
    # reflector row, generate on conj(pivot), store conj(τ) and conj(β); the right-apply then uses the
    # raw larfg τ (= conj(stored τ)) with dlarz/zlarz's ZGERC-conjugated essential.
    buf = Vector{T}(undef, L + 1)                        # (z)larfg on the (non-contiguous) length-(L+1) row
    @inbounds for i in m:-1:1
        buf[1] = conj(A[i, i])
        for l in 1:L; buf[l+1] = conj(A[i, m+l]); end
        β, τ = _larfg!(buf)                              # H_i annihilates A[i, m+1:n]; β real, τ = τ_larfg
        for l in 1:L; A[i, m+l] = buf[l+1]; end          # essential reflector row (conjugated domain)
        tau[i] = conj(τ); A[i, i] = conj(β)
        if τ != zero(T) && i > 1                         # apply H_i from the RIGHT to rows 1:i-1 (dlarz 'R')
            for r in 1:i-1
                w = A[r, i]
                for l in 1:L; w += A[r, m+l] * A[i, m+l]; end
                w *= τ
                A[r, i] -= w
                for l in 1:L; A[r, m+l] -= w * conj(A[i, m+l]); end
            end
        end
    end
    return A, tau
end

# ── apply Z / Zᴴ from an RZ factorization (LAPACK dormrz/zunmrz via the unblocked dormr3/zunmr3) ─────
# C := op(Z)·C (side='L') or C := C·op(Z) (side='R'), where Z's reflectors are stored in the k×(k+L)
# factor A (reflector i essential = A[i, k+1:k+L]) with coefficients `tau` (k = length(tau), L = the
# trailing "row" count = size(A,2)−k). trans: 'N' → Z, 'T'/'C' → Zᵀ/Zᴴ ('T' rejected for complex).
# `unmrz!` is the complex-facing alias (LAPACK zunmrz), same routine.
function ormrz!(side::Char, trans::Char, A::AbstractMatrix{T}, tau::AbstractVector{T},
        C::AbstractMatrix{T}) where {T<:BlasFloat}
    (side == 'L' || side == 'R') || throw(ArgumentError("ormrz!: side must be 'L' or 'R', got $(repr(side))"))
    (trans == 'N' || trans == 'T' || trans == 'C') ||
        throw(ArgumentError("ormrz!: trans must be 'N', 'T' or 'C', got $(repr(trans))"))
    (T <: Complex && trans == 'T') &&
        throw(ArgumentError("ormrz!: trans='T' invalid for complex — use 'C'"))
    k = length(tau); L = size(A, 2) - k
    k == 0 && return C
    mC, nC = size(C); notran = trans == 'N'; left = side == 'L'
    # loop direction (dormr3): forward when (left & apply-Zᴴ) or (right & apply-Z)
    forward = (left && !notran) || (!left && notran)
    rng = forward ? (1:k) : (k:-1:1)
    if left
        t0 = mC - L                                      # tail rows t0+1 : mC
        @inbounds for i in rng
            τi = notran ? tau[i] : conj(tau[i])
            τi == zero(T) && continue
            for jc in 1:nC
                w = C[i, jc]
                for l in 1:L; w += conj(A[i, k+l]) * C[t0+l, jc]; end
                w *= τi
                C[i, jc] -= w
                for l in 1:L; C[t0+l, jc] -= A[i, k+l] * w; end
            end
        end
    else
        t0 = nC - L                                      # tail cols t0+1 : nC
        @inbounds for i in rng
            τi = notran ? tau[i] : conj(tau[i])
            τi == zero(T) && continue
            for r in 1:mC
                w = C[r, i]
                for l in 1:L; w += C[r, t0+l] * A[i, k+l]; end
                w *= τi
                C[r, i] -= w
                for l in 1:L; C[r, t0+l] -= w * conj(A[i, k+l]); end
            end
        end
    end
    return C
end
const unmrz! = ormrz!

# ── driver (LAPACK dgelsy/zgelsy) ───────────────────────────────────────────────────────────────────
# Solve min‖A·X − B‖₂ (rank-deficient A, m×n) for the minimum-norm solution over the effective rank
# ≤ min(m,n), determined so the leading R-submatrix has 2-norm condition ≤ 1/rcond. B is size
# ≥ max(m,n) × nrhs (LAPACK ldb): input rows 1:m hold b, output rows 1:n hold X. `jpvt` (length ≥ n)
# receives the geqp3 pivots. Overwrites A and B. Returns (B, rank).
function gelsy!(A::AbstractMatrix{T}, B::AbstractMatrix{T}, jpvt::AbstractVector{<:Integer},
        rcond::Real) where {T<:BlasFloat}
    m, n = size(A); mn = min(m, n); R = real(T); nrhs = size(B, 2)
    size(B, 1) >= max(m, n) ||
        throw(DimensionMismatch("gelsy!: size(B,1)=$(size(B,1)) must be ≥ max(m,n)=$(max(m,n))"))
    length(jpvt) >= n || throw(DimensionMismatch("gelsy!: length(jpvt)=$(length(jpvt)) < n=$n"))
    tau = Vector{T}(undef, mn)
    geqp3!(A, jpvt, tau)                                 # A·P = Q·R  (rank-revealing)
    # ---- effective rank via the incremental condition estimator (dgelsy loop) ----
    rank = 0
    xmin = Vector{T}(undef, max(mn, 1)); xmax = Vector{T}(undef, max(mn, 1))
    if mn > 0 && abs(A[1, 1]) != zero(R)
        smax = abs(A[1, 1]); smin = smax
        xmin[1] = one(T); xmax[1] = one(T); rank = 1
        while rank < mn
            i = rank + 1
            sminpr, s1, c1 = _laic1(2, view(xmin, 1:rank), smin, view(A, 1:rank, i), A[i, i])
            smaxpr, s2, c2 = _laic1(1, view(xmax, 1:rank), smax, view(A, 1:rank, i), A[i, i])
            smaxpr * rcond <= sminpr || break
            @inbounds for kk in 1:rank; xmin[kk] *= s1; xmax[kk] *= s2; end
            xmin[rank+1] = c1; xmax[rank+1] = c2
            smin = sminpr; smax = smaxpr; rank += 1
        end
    end
    if rank == 0                                         # A ≈ 0 ⇒ min-norm solution is 0
        @inbounds for jc in 1:nrhs, r in 1:n; B[r, jc] = zero(T); end
        return B, 0
    end
    # ---- complete the orthogonal factorization (RZ) when column-rank-deficient ----
    tauz = Vector{T}(undef, rank)
    rank < n && tzrzf!(view(A, 1:rank, 1:n), tauz)       # [R11 R12] = [T11 0]·Z
    # ---- B := Qᴴ·B  (dormqr/zunmqr 'Left','Transpose'/'Conjugate'; geqp3 reflectors untouched by RZ) ----
    _apply_Qh!(A, tau, view(B, 1:m, :), mn)
    # ---- solve T11·Y = (Qᴴ·B)[1:rank] ----
    trsm!(view(B, 1:rank, :), view(A, 1:rank, 1:rank); side = 'L', uplo = 'U', transA = 'N', diag = 'N')
    @inbounds for jc in 1:nrhs, r in rank+1:n; B[r, jc] = zero(T); end
    # ---- B(1:n) := Zᴴ·[Y; 0]  (min-norm lift back through the RZ rotation) ----
    rank < n && ormrz!('L', T <: Complex ? 'C' : 'T', view(A, 1:rank, 1:n), tauz, view(B, 1:n, :))
    # ---- undo the column pivoting: X[jpvt[i]] = computed[i] ----
    work = Vector{T}(undef, n)
    @inbounds for jc in 1:nrhs
        for i in 1:n; work[jpvt[i]] = B[i, jc]; end
        for i in 1:n; B[i, jc] = work[i]; end
    end
    return B, rank
end
