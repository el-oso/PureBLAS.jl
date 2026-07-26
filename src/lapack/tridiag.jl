using LinearAlgebra: SingularException
# LAPACK tridiagonal solvers — dgtsv / dgttrf / dgttrs, generic over s/d/c/z (and any T<:Number,
# so Mode 2 stays AD-traceable). Faithful ports of Reference-LAPACK dgtsv.f, dgttrf.f, dgtts2.f.
#
# Storage (three vectors, LAPACK convention): `d` = main diagonal (length n), `dl` = sub-diagonal
# (length n-1, dl[i] = A[i+1,i]), `du` = super-diagonal (length n-1, du[i] = A[i,i+1]).
#
# The band structure gives O(n) work per RHS — bandwidth-trivial, so these are the plain scalar
# generic loops (no SIMD gain over a 3-term recurrence); one implementation covers all four types.

# Pivot magnitude. LAPACK compares pivots with CABS1 (|Re|+|Im|), NOT the modulus — `_l1` in core.jl is
# exactly that, and for real T it is plain `abs`. Using `abs` here instead selects a DIFFERENT pivot on
# near-ties for complex input: still a valid factorization (the residual is fine either way) but the
# factors, and ipiv, then disagree with z/cgttrf. It is also cheaper — no hypot on the complex path.
#
# Reshape a RHS vector to an n×1 matrix VIEW (shares memory) so the column loops are uniform. No copy.
@inline _gt_asmat(B::AbstractVector) = reshape(B, length(B), 1)
@inline _gt_asmat(B::AbstractMatrix) = B

# RHS accessors that keep gtsv!'s elimination generic over a vector or a matrix of right-hand sides.
# The point is `_gt_nc`: for a vector it returns the LITERAL 1, so the per-element column loop folds to
# `for j in 1:1` and vanishes. That matters more than it looks — a column loop with a RUNTIME trip count
# sitting inside the elimination costs 10.3 cyc/elem (19.1 → 29.5 measured at n = 65536 and 262144, i.e.
# it more than halves the routine) because it breaks up the scheduling of the divide→multiply→subtract
# recurrence around it. Reshaping a vector to n×1 is NOT what costs — indexing through the reshape
# measures identical to raw indexing (19.13 vs 19.13); it is purely the unknown trip count.
@inline _gt_nc(::AbstractVector) = 1
@inline _gt_nc(X::AbstractMatrix) = size(X, 2)
@inline _gt_nrow(X::AbstractVector) = length(X)
@inline _gt_nrow(X::AbstractMatrix) = size(X, 1)
Base.@propagate_inbounds _gt_at(X::AbstractVector, i, j) = X[i]
Base.@propagate_inbounds _gt_at(X::AbstractMatrix, i, j) = X[i, j]
Base.@propagate_inbounds _gt_set!(X::AbstractVector, i, j, v) = (X[i] = v)
Base.@propagate_inbounds _gt_set!(X::AbstractMatrix, i, j, v) = (X[i, j] = v)

# ── gtsv!: solve A·X = B in place (combined factorization + solve, partial pivoting) ────────────────
# Gaussian elimination with partial pivoting, storing the LU fill directly in dl/d/du (dgtsv.f). On a
# row interchange the second-superdiagonal fill lands in dl[i] (read back as the b[i+2] coefficient in
# the U back-solve). Overwrites dl, d, du (the factor) and B (← X). Throws SingularException on a zero
# pivot. Returns B.
function gtsv!(dl::AbstractVector, d::AbstractVector, du::AbstractVector, B::AbstractVecOrMat)
    n = length(d)
    (length(dl) == n - 1 && length(du) == n - 1) ||
        throw(DimensionMismatch("gtsv!: expected length(dl)=length(du)=n-1"))
    _gt_nrow(B) == n || throw(DimensionMismatch("gtsv!: size(B,1) must equal n"))
    nrhs = _gt_nc(B)
    Z = zero(eltype(dl))
    # The zero-pivot test is DEFERRED out of the loop (see gttrf! below for the full rationale): a
    # branchless `bad` flag is OR-ed per iteration and only a factorization that actually saw a zero
    # pays the rescan. Rescanning the final `d` reports the same index the in-loop test would have:
    # the test only fires on the `_l1(d[i]) >= _l1(dl[i])` branch, which leaves d[i] untouched, while
    # the interchange branch sets d[i] = dl[i] with |dl[i]| > |d[i]| ≥ 0 and so can never leave a zero.
    bad = false
    # This body is a divide→multiply→subtract recurrence whose latency bound is ~19.1 cyc/elem, and a
    # plain Thomas loop hits exactly that. Two things were measured to push it back down to ~20 from the
    # ~30 it sat at, both about how `d` and `dl` move rather than about the arithmetic:
    #
    #  1. `dl[i] = Z` is hoisted into a bulk pass per no-interchange RUN instead of a store per step.
    #     Per-step it cost 10.1 cyc/elem (19.1 → 29.2, flat in n) and storing the same volume to an
    #     UNRELATED array cost the same, so it is not about aliasing with dl's own reads; the bulk pass
    #     prices at 0.3–0.4 cyc/elem. (Bandwidth is not the issue either — a descending 3-read/1-write
    #     stream over these same arrays runs at 1.3–1.7 cyc/elem.)
    #  2. `di` carries the d recurrence in a register rather than re-reading d[i], worth a further
    #     10.4 cyc/elem (30.5 → 20.1), bit-identical. Carrying ONLY d is the win: also carrying the B
    #     recurrence measures WORSE (20.9), and carrying the back-solve is neutral (22.64 vs 22.69), so
    #     this is not a general "hoist everything into registers" rule. With d carried, the pivot test
    #     costs 0.9 cyc/elem — it only looked expensive while d was round-tripping through memory.
    #
    # The fast loop RESUMES after each swap rather than falling through to a general loop for the rest.
    # Swaps are rare even here (instrumented: 1 at n = 65536, 3 at n = 262144 on the benchmark's
    # "diagonally dominant" data) but they are spread through the matrix, so a one-shot fast path hands
    # everything after the FIRST one — most of the matrix — back to the slow loop. Resuming is what keeps
    # the fast path worth having; it was not separately timed against a one-shot version carrying d.
    piv = false
    i = 1
    @inbounds while i <= n - 1
        s = i                                             # start of this no-interchange run
        di = d[i]                                         # d recurrence carried in a register, see below
        while i <= n - 1 && _l1(di) >= _l1(dl[i])
            bad |= iszero(di)                             # `di` IS d[i], the pivot for this step
            fact = dl[i] / di
            di = d[i + 1] - fact * du[i]
            d[i + 1] = di
            for j in 1:nrhs
                _gt_set!(B, i + 1, j, _gt_at(B, i + 1, j) - fact * _gt_at(B, i, j))
            end
            i += 1
        end
        for k in s:(i - 1)
            dl[k] = Z                                     # this run's deferred "no fill" stores, in bulk
        end
        if i <= n - 1                                     # exactly one interchange step, then resume fast
            piv = true
            fact = d[i] / dl[i]
            d[i] = dl[i]
            temp = d[i + 1]
            d[i + 1] = du[i] - fact * temp
            if i < n - 1
                dl[i] = du[i + 1]                           # fill → dl[i] (coeff of b[i+2])
                du[i + 1] = -fact * dl[i]
            end
            du[i] = temp
            for j in 1:nrhs
                t = _gt_at(B, i, j)
                bi1 = _gt_at(B, i + 1, j)
                _gt_set!(B, i, j, bi1)
                _gt_set!(B, i + 1, j, t - fact * bi1)
            end
            bad |= iszero(d[i])
            i += 1
        end
    end
    @inbounds bad |= iszero(d[n])
    if bad
        @inbounds for i in 1:n
            iszero(d[i]) && throw(SingularException(i))
        end
    end
    # Back-solve U·x = b. With no interchange anywhere, every dl[i] is EXACTLY Z, so the b[i+2] term is a
    # guaranteed no-op sitting on the critical path — dropping it is bit-identical, not an approximation,
    # and shortens the recurrence to two terms (22.7 → 19.9 cyc/elem).
    @inbounds for j in 1:nrhs
        _gt_set!(B, n, j, _gt_at(B, n, j) / d[n])
        n > 1 && _gt_set!(B, n - 1, j, (_gt_at(B, n - 1, j) - du[n - 1] * _gt_at(B, n, j)) / d[n - 1])
        if piv
            for i in (n - 2):-1:1
                _gt_set!(
                    B, i, j,
                    (_gt_at(B, i, j) - du[i] * _gt_at(B, i + 1, j) - dl[i] * _gt_at(B, i + 2, j)) / d[i]
                )
            end
        else
            for i in (n - 2):-1:1
                _gt_set!(B, i, j, (_gt_at(B, i, j) - du[i] * _gt_at(B, i + 1, j)) / d[i])
            end
        end
    end
    return B
end

# ── gttrf!: LU factorization with partial pivoting (dgttrf.f) ─────────────────────────────────────
# Overwrites: d ← U diagonal, du ← U first superdiagonal, dl ← the L multipliers, du2 ← U second
# superdiagonal (fill from interchanges, length n-2), ipiv ← pivots (ipiv[i] ∈ {i, i+1}). Returns the
# 5-tuple like LinearAlgebra.LAPACK.gttrf!. Throws SingularException on a zero U pivot.
function gttrf!(
        dl::AbstractVector, d::AbstractVector, du::AbstractVector,
        du2::AbstractVector, ipiv::AbstractVector{<:Integer}
    )
    n = length(d)
    (length(dl) == n - 1 && length(du) == n - 1 && length(du2) == max(n - 2, 0) && length(ipiv) == n) ||
        throw(DimensionMismatch("gttrf!: dl,du length n-1; du2 length n-2; ipiv length n"))
    Z = zero(eltype(du2))
    # Three departures from dgttrf.f, none of them arithmetic. As shipped by the reference this loop runs
    # at ~32 cyc/elem against a ~19.5-cycle divide→multiply→subtract latency bound; the three together
    # take it to ~20.3, i.e. onto that bound (0.95 → 1.55 vs dgttrf at n ≥ 16384):
    #  1. `ipiv[i] = i` and `du2[i] = 0` are written on the no-interchange branch instead of in two
    #     separate prologue passes, which at n = 262144 streamed 2 MB + 2 MB of pure stores.
    #  2. The trailing "any zero pivot" scan — another full pass over d — becomes a branchless flag OR-ed
    #     into the loop, and only a singular factorization pays the rescan that recovers the exact index.
    #  3. `dcur` carries d[i] in a register (see below).
    # (1) and (2) are NOT independent: folding the prologue alone measures NEUTRAL-to-worse (1.014 →
    # 1.000), because with the trailing scan still present d is re-read at the end anyway and the fold
    # only adds store pressure inside the dependent loop. It pays only once the scan is gone. Verified
    # same-process, ABBA-bracketed (reference drift ≤ 0.2%) — measured in separate runs first, and that
    # cross-run pair reported the OPPOSITE sign for the fold.
    bad = false
    # `dcur` is d[i] carried in a register across iterations — the same store→load round trip that cost
    # gtsv! 10.4 cyc/elem (see there); this loop reads d[i] three times per step (pivot test, zero test,
    # divisor) while storing d[i+1], and the neighbouring dl/du2/ipiv stores block LLVM from forwarding it.
    dcur = n >= 1 ? (@inbounds d[1]) : zero(eltype(d))
    @inbounds for i in 1:(n - 2)
        if _l1(dcur) >= _l1(dl[i])                        # no interchange, eliminate dl[i]
            ipiv[i] = i; du2[i] = Z                       # defaults, written here not in a prologue pass
            bad |= iszero(dcur)                           # d[i] keeps this value on this branch
            if !iszero(dcur)
                fact = dl[i] / dcur; dl[i] = fact
                dcur = d[i + 1] - fact * du[i]
                d[i + 1] = dcur
            else
                dcur = d[i + 1]
            end
        else                                              # interchange rows i, i+1
            fact = dcur / dl[i]
            d[i] = dl[i]; bad |= iszero(dl[i])            # d[i] becomes dl[i] on this branch
            dl[i] = fact
            temp = du[i]; dnx = d[i + 1]
            du[i] = dnx
            dcur = temp - fact * dnx
            d[i + 1] = dcur
            du2[i] = du[i + 1]
            du[i + 1] = -fact * du[i + 1]
            ipiv[i] = i + 1
        end
    end
    if n > 1
        i = n - 1
        @inbounds if _l1(dcur) >= _l1(dl[i])
            ipiv[i] = i
            bad |= iszero(dcur)
            if !iszero(dcur)
                fact = dl[i] / dcur; dl[i] = fact
                d[i + 1] -= fact * du[i]
            end
        else
            fact = dcur / dl[i]
            d[i] = dl[i]; bad |= iszero(dl[i])
            dl[i] = fact
            temp = du[i]; dnx = d[i + 1]
            du[i] = dnx
            d[i + 1] = temp - fact * dnx
            ipiv[i] = i + 1
        end
        @inbounds bad |= iszero(d[n])
    end
    @inbounds if n >= 1
        ipiv[n] = n                                       # the last row is never interchanged
        n == 1 && (bad |= iszero(d[1]))
    end
    if bad
        @inbounds for i in 1:n
            iszero(d[i]) && throw(SingularException(i))
        end
    end
    return dl, d, du, du2, ipiv
end

# ── gttrs!: solve using the gttrf! factorization (dgtts2.f), trans ∈ {'N','T','C'} ─────────────────
# Overwrites B with the solution. 'N': A·X=B, 'T': Aᵀ·X=B, 'C': Aᴴ·X=B (conjugates the factor).
function gttrs!(
        trans::AbstractChar, dl::AbstractVector, d::AbstractVector, du::AbstractVector,
        du2::AbstractVector, ipiv::AbstractVector{<:Integer}, B::AbstractVecOrMat
    )
    n = length(d)
    Bm = _gt_asmat(B); size(Bm, 1) == n || throw(DimensionMismatch("gttrs!: size(B,1) must equal n"))
    nrhs = size(Bm, 2)
    n == 0 && return B
    if trans == 'N'
        @inbounds for j in 1:nrhs
            for i in 1:(n - 1)                                 # L·x = b (forward, applying pivots)
                ip = ipiv[i]
                temp = Bm[i + 1 - ip + i, j] - dl[i] * Bm[ip, j]
                Bm[i, j] = Bm[ip, j]
                Bm[i + 1, j] = temp
            end
            Bm[n, j] /= d[n]                               # U·x = b (backward)
            n > 1 && (Bm[n - 1, j] = (Bm[n - 1, j] - du[n - 1] * Bm[n, j]) / d[n - 1])
            for i in (n - 2):-1:1
                Bm[i, j] = (Bm[i, j] - du[i] * Bm[i + 1, j] - du2[i] * Bm[i + 2, j]) / d[i]
            end
        end
    else
        cj = trans == 'C' ? conj : identity               # Aᵀ (T) or Aᴴ (C)
        @inbounds for j in 1:nrhs
            Bm[1, j] /= cj(d[1])                           # Uᵀ·x = b (forward)
            n > 1 && (Bm[2, j] = (Bm[2, j] - cj(du[1]) * Bm[1, j]) / cj(d[2]))
            for i in 3:n
                Bm[i, j] = (Bm[i, j] - cj(du[i - 1]) * Bm[i - 1, j] - cj(du2[i - 2]) * Bm[i - 2, j]) / cj(d[i])
            end
            for i in (n - 1):-1:1                              # Lᵀ·x = b (backward, applying pivots)
                ip = ipiv[i]
                temp = Bm[i, j] - cj(dl[i]) * Bm[i + 1, j]
                Bm[i, j] = Bm[ip, j]
                Bm[ip, j] = temp
            end
        end
    end
    return B
end
