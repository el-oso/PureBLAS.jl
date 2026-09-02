# LAPACK symmetric-tridiagonal partial eigensolver: eigenvalues by bisection (dstebz) and
# eigenvectors by inverse iteration (dstein). STANDALONE — depends on nothing else in PureBLAS
# (no BLAS kernels), so this file can be `include`d on its own. Generic over T<:Real (s/d).
#
# These are the two halves of the LAPACK "expert" tridiagonal path (stev-x / syevx):
#   stebz! : Sturm-sequence bisection — selectable ranges A/V/I, ordered by block or by value.
#   stein! : inverse iteration on (T − λI) with random restart + modified-Gram-Schmidt
#            reorthogonalization against near eigenvectors (dstein). Real-only in LAPACK; complex
#            Hermitian eigenvectors are obtained by back-transforming these through unmtr.
#
# Faithful port of Reference-LAPACK dstebz/dlaebz and dstein/dlagtf/dlagts. The one deliberate
# deviation is documented at its site (the random start vector — see _stein_randvec!).

# ── Sturm negcount: number of eigenvalues of the sub-block T[i1:i2] strictly below x ────────────────
# The dlaneg / dlaebz recurrence tmp = d[j] − e2[j-1]/tmp − x with the pivmin guard. e2 holds the
# SQUARED off-diagonals with zeros at split points, so a single pass over [i1:i2] decouples blocks.
@inline function _sturm_negcount(
        d::AbstractVector{T}, e2::AbstractVector{T},
        i1::Int, i2::Int, x::T, pivmin::T
    ) where {T <: Real}
    cnt = 0
    tmp = d[i1] - x
    (abs(tmp) < pivmin) && (tmp = -pivmin)
    (tmp <= zero(T)) && (cnt += 1)
    @inbounds for j in (i1 + 1):i2
        tmp = d[j] - e2[j - 1] / tmp - x
        (abs(tmp) < pivmin) && (tmp = -pivmin)
        (tmp <= zero(T)) && (cnt += 1)
    end
    return cnt
end

# Bisect [low,high] (with negcount(low)=0, negcount(high)=in) for the kloc-th eigenvalue of T[i1:i2].
# dstebz/dlaebz convergence: stop when |high−low| < max(atoli, pivmin, rtoli·max(|low|,|high|)).
@inline function _sturm_bisect(
        d::AbstractVector{T}, e2::AbstractVector{T}, i1::Int, i2::Int,
        kloc::Int, low::T, high::T, pivmin::T, atoli::T, rtoli::T
    ) where {T <: Real}
    @inbounds while true
        tol = max(atoli, pivmin, rtoli * max(abs(low), abs(high)))
        (high - low <= tol) && break
        mid = (low + high) / 2
        (mid <= low || mid >= high) && break          # bracket collapsed to adjacent floats
        if _sturm_negcount(d, e2, i1, i2, mid, pivmin) >= kloc
            high = mid
        else
            low = mid
        end
    end
    return (low + high) / 2
end

# Gather `out[t] = v[idx[t]]` elementwise. A non-scalar `getindex` with a Vector{Int} index routes
# through Base's `_unsafe_getindex!` -> `throw_checksize_error` (multidimensional.jl), an eagerly
# interpolated DimensionMismatch that `juliac --trim=safe` rejects (req#4). The string is Base's;
# the fancy index is what reaches it. Linear-range `v[1:k]` and `view(...)` are unaffected.
# Annotations are ABSTRACT: `out`, `v` and `idx` are all workspace views under the in-place stebz!,
# and a `::Vector` annotation would not match a SubArray at all (MethodError, not a slowdown).
@inline function _stebz_gather!(
        out::AbstractVector, v::AbstractVector, idx::AbstractVector{<:Integer}
    )
    @inbounds for t in eachindex(idx)
        out[t] = v[Int(idx[t])]
    end
    return out
end

"""
    stebz!(range, order, vl, vu, il, iu, abstol, d, e) -> (w, iblock, isplit, info)
    stebz!(range, order, vl, vu, il, iu, abstol, d, e, w, iblock, isplit) -> (m, nsplit, info)

Eigenvalues of the real symmetric tridiagonal matrix with diagonal `d` (length n) and off-diagonal
`e` (length n−1) by Sturm-sequence bisection (LAPACK `dstebz`). `range`: `'A'` all, `'V'` those in
the half-open interval `(vl, vu]`, `'I'` those with index `il:iu` in ascending order. `order`: `'B'`
groups eigenvalues by split block (ascending within block); `'E'` returns them globally ascending.
`abstol ≤ 0` selects the default tolerance `eps·‖T‖`. Returns eigenvalues `w`, their block index
`iblock`, the block boundaries `isplit`, and `info` (0 on success).

The twelve-argument form is Reference-LAPACK dstebz's own signature: it writes into the caller's
`w`, `iblock` and `isplit` (each length ≥ n) and allocates nothing, returning the COUNTS instead —
`m` valid entries in `w`/`iblock`, `nsplit` in `isplit`. Read only `w[1:m]`, `iblock[1:m]` and
`isplit[1:nsplit]`: everything beyond is stale from a previous call, not zero. The three output
buffers must not alias `d`, `e` or each other — `isplit` is written while `d`/`e` are still being
read, and `iblock` is written while the block loop is still walking `isplit`.
"""
function stebz!(
        range::AbstractChar, order::AbstractChar, vl::T, vu::T, il::Integer, iu::Integer,
        abstol::Real, d::AbstractVector{T}, e::AbstractVector{T},
        w::AbstractVector{T}, iblock::AbstractVector{<:Integer}, isplit::AbstractVector{<:Integer}
    ) where {T <: Real}
    (range in ('A', 'V', 'I')) || throw(ArgumentError("stebz!: range must be 'A','V','I'"))
    (order in ('B', 'E')) || throw(ArgumentError("stebz!: order must be 'B' or 'E'"))
    n = length(d)
    length(e) >= n - 1 || throw(DimensionMismatch("stebz!: e must have length ≥ n-1"))
    (length(w) >= n && length(iblock) >= n && length(isplit) >= n) ||
        throw(DimensionMismatch("stebz!: w, iblock and isplit must each have length ≥ n"))
    (
        Base.mightalias(w, d) || Base.mightalias(w, e) ||
            Base.mightalias(iblock, d) || Base.mightalias(iblock, e) ||
            Base.mightalias(isplit, d) || Base.mightalias(isplit, e) ||
            Base.mightalias(w, iblock) || Base.mightalias(w, isplit) ||
            Base.mightalias(iblock, isplit)
    ) &&
        throw(ArgumentError("stebz!: `w`, `iblock` and `isplit` must not alias `d`, `e` or each other"))
    if n == 0
        return 0, 0, 0
    end
    if range == 'V' && vl >= vu
        return 0, 0, 0
    end
    if range == 'I' && (il < 1 || il > iu || iu > n)
        throw(ArgumentError("stebz!: require 1 ≤ il ≤ iu ≤ n"))
    end

    ulp = eps(T)                       # DLAMCH('P') relative machine precision
    safmin = floatmin(T)
    relfac = T(2); fudge = T(21) / 10     # dstebz RELFAC / FUDGE
    rtoli = ulp * relfac

    # ── split into decoupled blocks; build squared off-diagonals e2 (0 at splits); pivmin ──────────
    # e2's old allocator was `zeros`; the zeroing is NOT load-bearing and the workspace view is undef:
    # the loop below writes every index 1..n−1 unconditionally on both branches before either reader
    # (the Gershgorin loop, `_sturm_negcount`) runs, and the only never-written slot — e2[1] at n==1 —
    # is never read (both readers' loops are empty at n==1).
    e2, perm, perm2, idx, gw, gi = _stebz_work(T, n)
    ns = 0
    pivmin = one(T)
    @inbounds for j in 2:n
        t = e[j - 1]^2
        if abs(d[j] * d[j - 1]) * ulp^2 + safmin > t
            e2[j - 1] = zero(T); ns += 1; isplit[ns] = j - 1   # negligible off-diagonal → split here
        else
            e2[j - 1] = t; pivmin = max(pivmin, t)
        end
    end
    ns += 1; isplit[ns] = n
    pivmin *= safmin

    # ── global Gershgorin bounds [gl,gu], norm, absolute tolerance atoli ────────────────────────────
    gu = d[1]; gl = d[1]; tprev = zero(T)
    @inbounds for j in 1:(n - 1)
        tcur = sqrt(e2[j])
        gu = max(gu, d[j] + tprev + tcur)
        gl = min(gl, d[j] - tprev - tcur)
        tprev = tcur
    end
    gu = max(gu, d[n] + tprev); gl = min(gl, d[n] - tprev)
    tnorm = max(abs(gl), abs(gu))
    gl = gl - fudge * tnorm * ulp * n - fudge * 2 * pivmin
    gu = gu + fudge * tnorm * ulp * n + fudge * pivmin
    atoli = abstol <= 0 ? ulp * tnorm : T(abstol)

    # ── per block, bisect the selected eigenvalues ─────────────────────────────────────────────────
    # 'A': all; 'V': those in (vl,vu] via per-block Sturm counts at the interval ends; 'I': all here,
    # then sliced to the global index band il:iu below (robust under clustering, where value-boundary
    # bisection can miscount). Blocks emerge in natural order, ascending within block.
    mc = 0
    ibegin = 1
    @inbounds for jb in 1:ns
        i1 = ibegin; i2 = Int(isplit[jb]); ibegin = i2 + 1
        if range == 'V'
            nlo = _sturm_negcount(d, e2, i1, i2, vl, pivmin)
            nhi = _sturm_negcount(d, e2, i1, i2, vu, pivmin)
        else                               # 'A' and 'I' start from the full block
            nlo = 0; nhi = i2 - i1 + 1
        end
        for kloc in (nlo + 1):nhi
            λ = _sturm_bisect(d, e2, i1, i2, kloc, gl, gu, pivmin, atoli, rtoli)
            mc += 1; w[mc] = λ; iblock[mc] = jb
        end
    end

    # NOT `sortperm!`, and NOT the default algorithm — both were measured, both are wrong here:
    #   * Base's default (adaptive) sort allocates a scratch buffer (1888 B at n=129); `QuickSort` is
    #     fully in-place, and costs no reproducibility because Base's `Perm` ordering breaks ties by
    #     index itself, so the permutation is identical to the default's whatever the algorithm
    #     (verified over 200 tie-saturated inputs).
    #   * `sortperm!` carries an `axes(ix) != axes(A)` guard whose message interpolates an `Any`, so it
    #     lowers to the despecialized `string(...)` that `juliac --trim=safe` rejects — it failed the
    #     LAPACK trim-compatibility dogfood. Seeding the identity and sorting under `Perm` is the same
    #     computation without that guard.
    if range == 'I'                        # keep only the global index band il:iu (ascending)
        p = view(perm, 1:mc)
        @inbounds for t in 1:mc
            p[t] = t
        end
        sort!(p, QuickSort, Base.Order.Perm(Base.Order.Forward, view(w, 1:mc)))
        # Positions of the wanted eigenvalues, back in block-major (ascending position) order. Invert
        # p into idx, then compact in place — the compaction target k never runs ahead of the read
        # cursor t, so each slot is consumed before it is overwritten. O(m), and no second sort (the
        # obvious `sort!(idx[il:iu])` allocates even under QuickSort).
        @inbounds for r in 1:mc
            idx[p[r]] = r
        end
        k = 0
        @inbounds for t in 1:mc
            r = idx[t]
            if Int(il) <= r <= Int(iu)
                k += 1; idx[k] = t
            end
        end
        iv = view(idx, 1:k)
        _stebz_gather!(gw, w, iv); _stebz_gather!(gi, iblock, iv)
        @inbounds for t in 1:k
            w[t] = gw[t]; iblock[t] = gi[t]
        end
        mc = k
    end

    if order == 'E' && ns > 1
        p2 = view(perm2, 1:mc)             # own field: `perm` above is a different role (lesson 4)
        @inbounds for t in 1:mc
            p2[t] = t
        end
        sort!(p2, QuickSort, Base.Order.Perm(Base.Order.Forward, view(w, 1:mc)))  # ascending by value
        _stebz_gather!(gw, w, p2); _stebz_gather!(gi, iblock, p2)
        @inbounds for t in 1:mc
            w[t] = gw[t]; iblock[t] = gi[t]
        end
    end
    return mc, ns, 0
end

# Allocating convenience form — identical behaviour and return value to the pre-workspace stebz!:
# `w`/`iblock` trimmed to the m eigenvalues found, `isplit` to the nsplit block boundaries.
function stebz!(
        range::AbstractChar, order::AbstractChar, vl::T, vu::T, il::Integer, iu::Integer,
        abstol::Real, d::AbstractVector{T}, e::AbstractVector{T}
    ) where {T <: Real}
    n = length(d)
    w = Vector{T}(undef, n); iblock = Vector{Int}(undef, n); isplit = Vector{Int}(undef, n)
    m, ns, info = stebz!(range, order, vl, vu, il, iu, abstol, d, e, w, iblock, isplit)
    resize!(w, m); resize!(iblock, m); resize!(isplit, ns)
    return w, iblock, isplit, info
end

# ── inverse-iteration tridiagonal LU factor/solve (dlagtf / dlagts JOB=-1) ──────────────────────────
# _dlagtf!: factorize (T − λ) = P·L·U with partial pivoting. a=diag (→U diag), b=super, c=sub (→
# multipliers), d2=2nd super fill, inn=interchange record (inn[n] flags first tiny pivot). Verbatim
# port of Reference-LAPACK dlagtf.
function _dlagtf!(
        a::AbstractVector{T}, λ::T, b::AbstractVector{T}, c::AbstractVector{T},
        tol::T, d2::AbstractVector{T}, inn::AbstractVector{Int}
    ) where {T <: Real}
    n = length(a)
    a[1] -= λ
    inn[n] = 0
    if n == 1
        (a[1] == zero(T)) && (inn[1] = 1)
        return
    end
    epsm = eps(T) / 2                       # DLAMCH('E')
    tl = max(tol, epsm)
    scale1 = abs(a[1]) + abs(b[1])
    @inbounds for k in 1:(n - 1)
        a[k + 1] -= λ
        scale2 = abs(c[k]) + abs(a[k + 1])
        (k < n - 1) && (scale2 += abs(b[k + 1]))
        piv1 = a[k] == zero(T) ? zero(T) : abs(a[k]) / scale1
        if c[k] == zero(T)
            inn[k] = 0; piv2 = zero(T); scale1 = scale2
            (k < n - 1) && (d2[k] = zero(T))
        else
            piv2 = abs(c[k]) / scale2
            if piv2 <= piv1
                inn[k] = 0; scale1 = scale2
                c[k] = c[k] / a[k]
                a[k + 1] -= c[k] * b[k]
                (k < n - 1) && (d2[k] = zero(T))
            else
                inn[k] = 1
                mult = a[k] / c[k]
                a[k] = c[k]
                temp = a[k + 1]
                a[k + 1] = b[k] - mult * temp
                if k < n - 1
                    d2[k] = b[k + 1]
                    b[k + 1] = -mult * d2[k]
                end
                b[k] = temp
                c[k] = mult
            end
        end
        (max(piv1, piv2) <= tl && inn[n] == 0) && (inn[n] = k)
    end
    (abs(a[n]) <= scale1 * tl && inn[n] == 0) && (inn[n] = n)
    return
end

# _dlagts! JOB=-1: solve (T − λ)·x = y in place on y, using dlagtf factors, perturbing tiny pivots
# (inverse-iteration mode). tol is INOUT (recomputed on tol≤0); returns the resolved tol. Verbatim
# port of Reference-LAPACK dlagts JOB=-1.
function _dlagts!(
        a::AbstractVector{T}, b::AbstractVector{T}, c::AbstractVector{T},
        d2::AbstractVector{T}, inn::AbstractVector{Int}, y::AbstractVector{T}, tol::T
    ) where {T <: Real}
    n = length(a)
    epsm = eps(T) / 2                       # DLAMCH('E')
    sfmin = floatmin(T); bignum = one(T) / sfmin
    if tol <= zero(T)
        tol = abs(a[1])
        (n > 1) && (tol = max(tol, abs(a[2]), abs(b[1])))
        @inbounds for k in 3:n
            tol = max(tol, abs(a[k]), abs(b[k - 1]), abs(d2[k - 2]))
        end
        tol *= epsm
        (tol == zero(T)) && (tol = epsm)
    end
    @inbounds for k in 2:n                   # forward substitution with the recorded interchanges
        if inn[k - 1] == 0
            y[k] -= c[k - 1] * y[k - 1]
        else
            temp = y[k - 1]; y[k - 1] = y[k]; y[k] = temp - c[k - 1] * y[k]
        end
    end
    @inbounds for k in n:-1:1                # back substitution with tiny-pivot perturbation
        if k <= n - 2
            temp = y[k] - b[k] * y[k + 1] - d2[k] * y[k + 2]
        elseif k == n - 1
            temp = y[k] - b[k] * y[k + 1]
        else
            temp = y[k]
        end
        ak = a[k]; pert = copysign(tol, ak)
        while true
            absak = abs(ak)
            if absak < one(T)
                if absak < sfmin
                    if absak == zero(T) || abs(temp) * sfmin > absak
                        ak += pert; pert *= 2; continue
                    else
                        temp *= bignum; ak *= bignum
                    end
                elseif abs(temp) > absak * bignum
                    ak += pert; pert *= 2; continue
                end
            end
            break
        end
        y[k] = temp / ak
    end
    return tol
end

# Random start vector, uniform in (−1,1). ponytail: LAPACK uses DLARNV(2,ISEED,…) (the dlaruv 512-word
# multiplicative-congruential table). Its ONLY role is a generic nonzero start for inverse iteration;
# the residual/orthonormality that validate a returned eigenvector cannot distinguish RNGs. A per-call
# xorshift with a persisting seed reproduces DLARNV's substance (fresh random restart, seed advances)
# without transcribing 512 magic constants. Upgrade to a true dlaruv port only if bit-identical LAPACK
# reproduction is ever required.
@inline function _stein_randvec!(x::AbstractVector{T}, seed::Base.RefValue{UInt64}) where {T <: Real}
    s = seed[]
    @inbounds for i in eachindex(x)
        s ⊻= s << 13; s ⊻= s >> 7; s ⊻= s << 17           # xorshift64
        x[i] = 2 * (T(s >> 11) / T(UInt64(1) << 53)) - one(T)   # uniform in (−1,1)
    end
    seed[] = s
    return x
end

"""
    stein!(d, e, w, iblock, isplit) -> Z
    stein!(d, e, w, iblock, isplit, Z) -> Z

Eigenvectors of the real symmetric tridiagonal matrix `(d, e)` for the eigenvalues `w` (with block
labels `iblock` and split boundaries `isplit` from [`stebz!`](@ref)) by inverse iteration (LAPACK
`dstein`). Returns `Z` (n × length(w)); column j is the unit eigenvector for `w[j]`. Uses random
restart and modified-Gram-Schmidt reorthogonalization against near eigenvectors within each block.
Real-only, matching LAPACK (complex Hermitian eigenvectors back-transform these via `unmtr`).

The six-argument form writes into the caller-provided `Z` (size ≥ n × length(w)) and allocates
nothing. It ZEROES `Z[1:n, 1:m]` on entry, and that zeroing is load-bearing, not hygiene: only rows
`b1:bn` of the current split block are ever written, and a tridiagonal eigenvector must be exactly
zero outside its own block. `Z` must not alias `d`, `e`, `w`, `iblock` or `isplit` — all five are
read on every column while `Z` is being filled in.
"""
function stein!(
        d::AbstractVector{T}, e::AbstractVector{T}, w::AbstractVector{T},
        iblock::AbstractVector{<:Integer}, isplit::AbstractVector{<:Integer},
        Z::AbstractMatrix{T}
    ) where {T <: Real}
    n = length(d)
    m = length(w)
    length(e) >= n - 1 || throw(DimensionMismatch("stein!: e must have length ≥ n-1"))
    (size(Z, 1) >= n && size(Z, 2) >= m) ||
        throw(DimensionMismatch("stein!: Z must be at least n × length(w)"))
    (
        Base.mightalias(Z, d) || Base.mightalias(Z, e) || Base.mightalias(Z, w) ||
            Base.mightalias(Z, iblock) || Base.mightalias(Z, isplit)
    ) &&
        throw(ArgumentError("stein!: `Z` must not alias `d`, `e`, `w`, `iblock` or `isplit`"))
    @inbounds for j in 1:m, i in 1:n
        Z[i, j] = zero(T)
    end
    (m == 0 || n == 0) && return Z

    epsm = eps(T)                          # DLAMCH('P')
    odm1 = T(1) / 10; odm3 = T(1) / 1000; ten = T(10)
    maxits = 5; extra = 2

    # per-block scratch (max block size ≤ n) + the deterministic restart seed (dstein's ISEED
    # analogue), which the accessor resets per call — exactly the old `Ref(…)`-inside-stein! semantics.
    av, bv, cv, d2, rhs, inn, seed = _stein_work(T, n)

    j1 = 1
    nblkmax = 0; @inbounds for x in iblock
        nblkmax = max(nblkmax, Int(x))
    end   # not maximum(): abstract-elt MappingRF is --trim-unsafe
    @inbounds for nblk in 1:nblkmax
        b1 = nblk == 1 ? 1 : Int(isplit[nblk - 1]) + 1
        bn = Int(isplit[nblk])
        bz = bn - b1 + 1
        gpind = j1
        # reorthogonalization / stopping criteria for this block
        onenrm = abs(d[b1]) + (bz > 1 ? abs(e[b1]) : zero(T))
        if bz > 1
            onenrm = max(onenrm, abs(d[bn]) + abs(e[bn - 1]))
        end
        for i in (b1 + 1):(bn - 1)
            onenrm = max(onenrm, abs(d[i]) + abs(e[i - 1]) + abs(e[i]))
        end
        ortol = odm3 * onenrm
        dtpcrt = sqrt(odm1 / bz)

        jblk = 0
        xjm = zero(T)
        j = j1
        while j <= m
            if iblock[j] != nblk
                j1 = j; break
            end
            jblk += 1
            xj = w[j]

            if bz == 1
                Z[b1, j] = one(T)
                xjm = xj
                (j == m) && (j1 = j + 1)
                j += 1
                continue
            end

            # nudge apart eigenvalues that are too close (dstein perturbation)
            if jblk > 1
                eps1 = abs(epsm * xj); pertol = ten * eps1
                (xj - xjm < pertol) && (xj = xjm + pertol)
            end

            av1 = view(av, 1:bz); bv1 = view(bv, 1:(bz - 1)); cv1 = view(cv, 1:(bz - 1))
            d21 = view(d2, 1:max(bz - 2, 1)); inn1 = view(inn, 1:bz); r1 = view(rhs, 1:bz)

            _stein_randvec!(r1, seed)
            copyto!(av1, view(d, b1:bn))
            copyto!(bv1, view(e, b1:(bn - 1)))
            copyto!(cv1, view(e, b1:(bn - 1)))
            tol = zero(T)
            _dlagtf!(av1, xj, bv1, cv1, tol, d21, inn1)

            its = 0; nrmchk = 0
            while true
                its += 1
                (its > maxits) && break      # nonconvergence — accept last iterate anyway
                # scale the RHS (dstein SCL) then solve (T−λ)·x = scaled-RHS
                jmax = _iamax(r1)
                scl = bz * onenrm * max(epsm, abs(av1[bz])) / abs(r1[jmax])
                @simd for t in 1:bz
                    r1[t] *= scl
                end
                tol = _dlagts!(av1, bv1, cv1, d21, inn1, r1, tol)

                # modified Gram-Schmidt against near eigenvectors of this block
                if jblk != 1
                    (abs(xj - xjm) > ortol) && (gpind = j)
                    if gpind != j
                        for i in gpind:(j - 1)
                            ztr = -_dot(r1, view(Z, b1:bn, i))
                            _axpy!(ztr, view(Z, b1:bn, i), r1)
                        end
                    end
                end

                jmax = _iamax(r1)
                nrm = abs(r1[jmax])
                (nrm < dtpcrt) && continue   # not yet at stopping criterion
                nrmchk += 1
                (nrmchk < extra + 1) && continue
                break
            end

            # normalize (2-norm 1, sign so the largest component is positive)
            scl = one(T) / _nrm2(r1)
            jmax = _iamax(r1)
            (r1[jmax] < zero(T)) && (scl = -scl)
            @simd for t in 1:bz
                r1[t] *= scl
            end
            for t in 1:bz
                Z[b1 + t - 1, j] = r1[t]
            end

            xjm = xj
            (j == m) && (j1 = j + 1)
            j += 1
        end
    end
    return Z
end

# Allocating convenience form — identical behaviour and return value to the pre-workspace stein!
# (the in-place form does the zeroing the old `zeros(T, n, m)` provided).
function stein!(
        d::AbstractVector{T}, e::AbstractVector{T}, w::AbstractVector{T},
        iblock::AbstractVector{<:Integer}, isplit::AbstractVector{<:Integer}
    ) where {T <: Real}
    return stein!(d, e, w, iblock, isplit, Matrix{T}(undef, length(d), length(w)))
end

# tiny local BLAS-1 helpers (keeps this file standalone; T<:Real, unit stride)
@inline function _iamax(x::AbstractVector{T}) where {T <: Real}
    k = 1; v = abs(x[1])
    @inbounds for i in 2:length(x)
        a = abs(x[i]); (a > v) && (v = a; k = i)
    end
    return k
end
@inline function _dot(x::AbstractVector{T}, y::AbstractVector{T}) where {T <: Real}
    s = zero(T)
    @inbounds @simd for i in eachindex(x)
        s = muladd(x[i], y[i], s)
    end
    return s
end
@inline function _axpy!(a::T, x::AbstractVector{T}, y::AbstractVector{T}) where {T <: Real}
    @inbounds @simd for i in eachindex(x)
        y[i] = muladd(a, x[i], y[i])
    end
    return y
end
@inline function _nrm2(x::AbstractVector{T}) where {T <: Real}
    s = zero(T)
    @inbounds @simd for i in eachindex(x)
        s = muladd(x[i], x[i], s)
    end
    return sqrt(s)
end
