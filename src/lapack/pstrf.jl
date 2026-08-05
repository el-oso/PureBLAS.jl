# LAPACK pstrf — Cholesky with COMPLETE (diagonal) pivoting for a positive-SEMIdefinite matrix.
# Pure Julia, generic over T (Float32/Float64/ComplexF32/ComplexF64). At each step the largest
# remaining Schur-complement diagonal is pivoted to the front; the factorization stops at the numerical
# rank (first pivot ≤ tol). Produces Pᵀ·A·P = Lᴴ... with a permutation `piv` and detected `rank`, so it
# backs `cholesky(A, RowMaximum())`.
#
# Ports LAPACK's UNBLOCKED level-2 kernel dpstf2/zpstf2 (the blocked dpstrf gives identical results —
# ponytail: blocked deferred, add when perf-gated). ONE generic path covers real (SPD) and complex
# (Hermitian PSD) by threading `conj`/`real`/`abs2` exactly where zpstf2 differs from dpstf2 (the
# Hermitian symmetric-swap conjugates the off-diagonal triangle; conj is identity on reals).

# Hermitian/symmetric swap of index j ↔ pvt for the LOWER-stored triangle (dpstf2/zpstf2 lower).
# `lo` bounds the ROW-swap part to columns lo:j-1. The blocked driver passes lo = k (the panel start) and
# DEFERS columns 1:k-1 to one batched pass per panel — that row swap is stride-lda (2 cache lines touched
# per 8-byte element) and measured 47% of the whole factorization at n=2048 when done per column.
@inline function _pstrf_swap_lower!(A::AbstractMatrix{T}, j::Int, pvt::Int, n::Int, lo::Int = 1) where {T}
    @inbounds begin
        A[pvt, pvt] = A[j, j]
        for l in lo:(j - 1)
            A[j, l], A[pvt, l] = A[pvt, l], A[j, l]
        end     # leading row parts (cols lo:j-1)
        for l in (pvt + 1):n
            A[l, j], A[l, pvt] = A[l, pvt], A[l, j]
        end   # col parts below pvt
        for i in (j + 1):(pvt - 1)                                               # triangle between (conjugated)
            tmp = conj(A[i, j]); A[i, j] = conj(A[pvt, i]); A[pvt, i] = tmp
        end
        A[pvt, j] = conj(A[pvt, j])                                      # (no-op real)
    end
    return A
end
# Hermitian/symmetric swap for the UPPER-stored triangle (dpstf2/zpstf2 upper).
@inline function _pstrf_swap_upper!(A::AbstractMatrix{T}, j::Int, pvt::Int, n::Int) where {T}
    @inbounds begin
        A[pvt, pvt] = A[j, j]
        for l in 1:(j - 1)
            A[l, j], A[l, pvt] = A[l, pvt], A[l, j]
        end     # leading col parts (rows 1:j-1)
        for c in (pvt + 1):n
            A[j, c], A[pvt, c] = A[pvt, c], A[j, c]
        end   # row parts right of pvt
        for i in (j + 1):(pvt - 1)                                               # triangle between (conjugated)
            tmp = conj(A[j, i]); A[j, i] = conj(A[i, pvt]); A[i, pvt] = tmp
        end
        A[j, pvt] = conj(A[j, pvt])                                      # (no-op real)
    end
    return A
end

# ── Blocked pivoted Cholesky (LAPACK dpstrf/zpstrf) ───────────────────────────────────────────────
# The unblocked kernel below updates every column against ALL previously-factored columns with scalar
# loops — the whole O(n³) runs at BLAS-1/2 speed. The blocked driver factors an nb-wide PANEL, where
# each column's update touches only the panel's OWN columns (one gemv), and then applies ONE rank-jb
# symmetric/Hermitian update to the trailing block (syrk!/herk!, BLAS-3) — LAPACK dpstrf exactly.
# `work[k:n]` is RESET per panel: the earlier panels' contributions are already folded into A[i,i] by
# their syrk, so `real(A[i,i]) − work[i]` remains the exact Schur-complement diagonal for pivoting.
# Pivot swaps stay FULL-WIDTH (they must permute the already-factored columns too).
# Replay a panel's recorded row swaps on the ALREADY-FACTORED leading columns 1:ncol (LOWER only —
# UPPER's leading swap is `A[l,j] ↔ A[l,pvt]`, contiguous in column-major, so it needs no batching).
# COLUMN-OUTER: each leading column is streamed once and takes all nsw swaps while it is cache-hot,
# instead of being revisited (and evicted) once per panel column. Same element count, ~nb× fewer misses.
# Order is preserved (swaps compose), so replaying in sequence reproduces the per-column result exactly.
@inline function _pstrf_apply_swaps_lower!(
        A::AbstractMatrix{T}, swj::AbstractVector{Int}, swp::AbstractVector{Int}, nsw::Int, ncol::Int
    ) where {T}
    (nsw == 0 || ncol == 0) && return A
    @inbounds for l in 1:ncol, s in 1:nsw
        j = swj[s]; p = swp[s]
        A[j, l], A[p, l] = A[p, l], A[j, l]
    end
    return A
end

@inline function _pstrf_trailing!(C, B, ::Val{LOWER}, ::Type{T}) where {LOWER, T <: Real}
    return syrk!(C, B; uplo = LOWER ? 'L' : 'U', trans = LOWER ? 'N' : 'T', alpha = -one(T), beta = one(T))
end
@inline function _pstrf_trailing!(C, B, ::Val{LOWER}, ::Type{T}) where {LOWER, T <: Complex}
    return herk!(C, B; uplo = LOWER ? 'L' : 'U', trans = LOWER ? 'N' : 'C', alpha = -one(real(T)), beta = one(real(T)))
end

# Panel width. PDM: Derive — same physical criterion as getrf's pivoted panel (this IS getrf's shape:
# a BLAS-2 pivoted panel amortized against a rank-nb BLAS-3 trailing update), so it reuses that
# already-fleet-validated derived width rather than introducing a second, unvalidated formula.
_pstrf_nb(n::Int) = _lu_nb(n)

# UPPER panel row-update variant selector. The output row A[j,j+1:n] is STRIDED (lda), so its L entries
# touch L distinct cache lines — footprint L·_CACHELINE. Two equivalent formulations, and which wins is a
# pure L1-RESIDENCY question (measured crossover between n=128 and n=512, ratio 0.86 → 1.55):
#   • footprint fits ~½L1  ⇒ FUSED: gemv with beta=0 into scratch, then one read-modify-write scatter that
#     folds the subtract and the 1/ajj scaling — saves a whole pass over the row while it stays L1-hot.
#   • footprint spills L1  ⇒ SPLIT: gather the row first, gemv with beta=1, then a PURE-STORE scatter —
#     two sequential-stride streams the prefetcher handles, instead of an RMW that defeats it.
# PDM Derive: `_L1_BYTES`/`_CACHELINE` (detected) + a residency criterion; overridable "pstrf_fuse_max".
const _PSTRF_FUSE_MAXL = @load_preference("pstrf_fuse_max", (_L1_BYTES ÷ 2) ÷ _CACHELINE)::Int

# ── UPPER Schur-diagonal row cache: order above which it pays ─────────────────────────────────────
# The Schur loop reads A[j-1, j:n] — a row, strided by lda, so a scalar gather where LOWER's
# contiguous column vectorises. Caching the row (it is exactly what the previous column computed)
# makes it unit-stride, but costs ONE EXTRA STORE per element. Which way that trades is size
# dependent, measured on Zen4 vs OpenBLAS, uplo='U' (before → after caching unconditionally):
#     n=48  0.943→0.945   n=64  0.921→0.889   n=96  1.007→0.946     ← the store dominates
#     n=128 1.451→1.570   n=256 1.726→2.211   n=512 1.989→2.318
#     n=1024 1.918→2.112  n=2048 1.546→1.915                        ← the gather dominates
# So it is a genuine crossover, and applying it everywhere turns a passing n=96 into a miss.
# req#8 tier: MEASURE. Not Derive — the residency model predicts the wrong answer: the row's
# footprint is (n-j)·_CACHELINE, which stays under L1 until n≈512, yet caching already wins at
# n=128. What actually trades is a store port against a gather, i.e. port balance, which is the
# documented tell for Measure (same class as _pbtrf_cross and _ger_np).
# Candidates are DERIVED as {2,4,8,16}·W, the octave around the measured crossover; the harness
# races the two forms at each candidate order and returns the first where caching wins.
#
# ⚠ TIER HONESTY: this ships as a VALIDATED LITERAL (128), not a Measure knob, and that is a
# deliberate downgrade rather than an oversight. I wrote the OncePerProcess harness twice and it
# disagreed with the authoritative gate BOTH times, in the same direction:
#   • v1 timed the two loop shapes in isolation on a synthetic matrix → answered 32.
#   • v2 raced the real factorization → answered 16–32, unstably, and claimed caching WINS at n=64
#     where bench/plots-grade measurement says it loses (0.921 → 0.889).
# The reason is the harness reuses one buffer (`copyto!` per rep), so A stays L1-warm across
# repetitions — which is exactly the residency question the knob is about. The gate hands every
# sample a fresh context. A harness whose warm-up regime differs from the live one on the very
# property being measured cannot be trusted, and shipping it would have cached at n=48…96 where
# caching regresses. Recorded rather than hidden: this is the probe-vs-live trap in its subtlest
# form, and it is the third time this session a probe outranked the live path.
# So 128 comes from the gate table above, and the Preference override stays. Converting this to a
# genuine Measure knob needs a harness with cold, per-rep contexts — worth doing, not done here.
# ⚠ Zen4 Float64 only; needs fleet validation before it is trusted to extrapolate.
const _PSTRF_ROWCACHE_PREF = @load_preference("pstrf_rowcache_min", 128)
@static if false
    # Races the REAL factorization both ways at each candidate order. An earlier version of this
    # harness timed the two loop shapes in isolation and answered 32 — where the whole-op measurement
    # says caching LOSES (n=48/64/96 all regress). Classic probe-vs-live: a synthetic matrix with a
    # reset accumulator does not reproduce the cache state the live scatter leaves behind. Measure the
    # thing that ships.
    function _measure_pstrf_rowcache(::Type{T})::Int where {T}
        vw = _vwidth(T)
        R = real(T)
        Base.generating_output() && return 16 * vw     # don't burn a measure during precompilation
        try
            cands = (2 * vw, 4 * vw, 8 * vw, 16 * vw)
            wins = falses(length(cands))
            for (ci, c) in pairs(cands)
                A0 = Matrix{T}(undef, c, c)            # diagonally dominant ⇒ HPD, full rank, no early exit
                @inbounds for q in 1:c, p in 1:c
                    A0[p, q] = p == q ? T(4 * c) : T(1) / T(p + q)
                end
                A = similar(A0); piv = Vector{Int}(undef, c); work = zeros(R, 2c)
                nb = min(_pstrf_nb(c), c); scr = Vector{T}(undef, nb + 2c)
                run = ub -> begin
                    copyto!(A, A0); fill!(work, zero(R))
                    @inbounds for i in 1:c
                        piv[i] = i
                    end
                    _pstrf_blocked!(A, piv, work, scr, R(-1), 1, real(A0[1, 1]), nb, Val(false), ub)
                end
                run(true); run(false)                              # untimed warmups (absorb JIT)
                tbs = Vector{UInt64}(undef, 5); tns = Vector{UInt64}(undef, 5)
                for r in 1:5                                       # interleaved (crude ABBA), MEDIAN-of-5
                    s = time_ns(); run(false); tns[r] = time_ns() - s
                    s = time_ns(); run(true); tbs[r] = time_ns() - s
                end
                sort!(tbs); sort!(tns); tb = tbs[3]; tn = tns[3]
                wins[ci] = tb < tn
            end
            # The benefit is NOT monotone in n — measured on Zen4 it wins at tiny orders, LOSES at
            # 48…96 (the extra store dominates while everything is L1-resident) and wins again from
            # 128 up (the gather dominates). So "first candidate that wins" is the wrong search: it
            # answered 16 and would have cached exactly where caching regresses. Take the smallest
            # candidate from which every LARGER candidate also wins — the point past which the
            # benefit is durable.
            best = typemax(Int)
            for ci in length(cands):-1:1
                wins[ci] || break
                best = cands[ci]
            end
            return best
        catch
            return typemax(Int)        # conservative: keep the un-cached form
        end
    end
    const _PSTRF_RC_F32 = Base.OncePerProcess{Int}(() -> _measure_pstrf_rowcache(Float32))
    const _PSTRF_RC_F64 = Base.OncePerProcess{Int}(() -> _measure_pstrf_rowcache(Float64))
    const _PSTRF_RC_C32 = Base.OncePerProcess{Int}(() -> _measure_pstrf_rowcache(ComplexF32))
    const _PSTRF_RC_C64 = Base.OncePerProcess{Int}(() -> _measure_pstrf_rowcache(ComplexF64))
    @inline _pstrf_rowcache_min(::Type{Float32}) = _PSTRF_RC_F32()
    @inline _pstrf_rowcache_min(::Type{Float64}) = _PSTRF_RC_F64()
    @inline _pstrf_rowcache_min(::Type{ComplexF32}) = _PSTRF_RC_C32()
    @inline _pstrf_rowcache_min(::Type{ComplexF64}) = _PSTRF_RC_C64()
    @inline _pstrf_rowcache_min(::Type{T}) where {T} = typemax(Int)   # generic/AD: never cache
else
    @inline _pstrf_rowcache_min(::Type{<:Any}) = _PSTRF_ROWCACHE_PREF::Int   # pinned (trim builds)
end

# Returns `rank` (n if full-rank). `scr` is scratch: scr[1:nb] the contiguous panel-row gather (the
# gemv's x, strided in A), scr[nb+1:nb+n] the UPPER path's contiguous output row (A[j,j+1:n] is
# strided by lda, so gather → gemv → scatter keeps the SIMD gemv on unit stride).
function _pstrf_blocked!(
        A::AbstractMatrix{T}, piv::AbstractVector{<:Integer}, work::AbstractVector{R},
        scr::AbstractVector{T}, dstop::R, pvt_in::Int, ajj_in::R, nb::Int, ::Val{LOWER},
        usebuf::Bool = !LOWER && size(A, 1) >= _pstrf_rowcache_min(T)
    ) where {T, R <: Real, LOWER}
    n = size(A, 1)
    pvt = pvt_in; ajj = ajj_in
    # UPPER only: contiguous copy of the row just written, so the Schur-diagonal loop above reads
    # unit-stride instead of striding A by lda. Carved out of the caller's owned `scr` (GKH) rather
    # than allocated here — scr is sized nb + 2n for exactly this.
    rowb = LOWER ? view(scr, 1:0) : view(scr, (nb + n + 1):(nb + 2n))
    swj = Vector{Int}(undef, nb); swp = Vector{Int}(undef, nb)   # deferred leading row swaps (LOWER)
    @inbounds for k in 1:nb:n
        jb = min(nb, n - k + 1)
        nsw = 0
        for i in k:n
            work[i] = zero(R)                       # per-panel reset (prior panels live in A[i,i])
        end
        for j in k:(k + jb - 1)
            # Schur-complement diagonals. LOWER reads a contiguous column A[j+1:n, j-1]; UPPER would
            # read the ROW A[j-1, j:n], strided by lda — a scalar gather where the lower form
            # vectorises, run O(n²/2) times. Measured on Zen4 (whole-factorization cost of this loop
            # alone): n=48 col 2.26 µs vs row 2.74 µs, n=64 3.18 vs 4.40 — i.e. 5.1% and 7.2% of the
            # op, against gate deficits of 5.7% and 8.1%. It was the whole miss.
            # The row is free, though: it is exactly what the previous column just computed, and that
            # value passes through scratch on its way into A. `rowb` keeps it contiguous, so UPPER
            # reads unit-stride like LOWER does. No staleness: rowb is refreshed at the end of every
            # column, and the pivot swap for column j runs BEFORE column j's update, so the cached
            # row is always post-swap.
            # The `work[n+i] = real(A[i,i]) - work[i]` half MUST stay fused with the accumulate: A[i,i]
            # is a strided diagonal read, and splitting this into two passes over i traverses it twice.
            # Measured cost of getting that wrong: U n=64 0.921 → 0.852, n=96 1.007 → 0.899 — a
            # regression on sizes the row cache never even touches. So the branch is hoisted out of
            # the i-loop and each variant keeps the fusion.
            if j > k
                if LOWER
                    for i in j:n
                        work[i] += abs2(A[i, j - 1]); work[n + i] = real(A[i, i]) - work[i]
                    end
                elseif usebuf
                    for i in j:n
                        work[i] += abs2(rowb[i]); work[n + i] = real(A[i, i]) - work[i]
                    end
                else
                    for i in j:n
                        work[i] += abs2(A[j - 1, i]); work[n + i] = real(A[i, i]) - work[i]
                    end
                end
            else
                for i in j:n
                    work[n + i] = real(A[i, i]) - work[i]
                end
            end
            if j > 1
                pvt = j; mx = work[n + j]
                for i in (j + 1):n
                    (work[n + i] > mx) && (mx = work[n + i]; pvt = i)
                end
                ajj = work[n + pvt]
                if ajj <= dstop || isnan(ajj)
                    A[j, j] = T(ajj)
                    LOWER && _pstrf_apply_swaps_lower!(A, swj, swp, nsw, k - 1)   # flush before bailing
                    return j - 1                            # rank-deficient: stop at numerical rank
                end
            end
            if j != pvt
                if LOWER                                    # leading cols 1:k-1 deferred to the panel flush
                    _pstrf_swap_lower!(A, j, pvt, n, k)
                    nsw += 1; swj[nsw] = j; swp[nsw] = pvt
                else
                    _pstrf_swap_upper!(A, j, pvt, n)
                end
                work[pvt] = work[j]
                piv[j], piv[pvt] = piv[pvt], piv[j]
            end
            ajj = sqrt(ajj); A[j, j] = T(ajj)
            if j < n
                jk = j - k                                  # panel columns already factored (k:j-1)
                invajj = one(R) / ajj
                if LOWER
                    if jk > 0                               # A[j+1:n,j] −= A[j+1:n,k:j-1]·conj(A[j,k:j-1])
                        for t in 1:jk
                            scr[t] = conj(A[j, k + t - 1])
                        end
                        _gemv!(
                            false, false, n - j, jk, -one(T), view(A, (j + 1):n, k:(j - 1)),
                            view(scr, 1:jk), 1, one(T), view(A, (j + 1):n, j), 1
                        )
                    end
                    for i in (j + 1):n
                        A[i, j] *= invajj
                    end
                else
                    L = n - j
                    if jk > 0                               # A[j,j+1:n] −= conj(A[k:j-1,j])ᵀ·A[k:j-1,j+1:n]
                        for t in 1:jk
                            scr[t] = conj(A[k + t - 1, j])
                        end
                        if L <= _PSTRF_FUSE_MAXL          # row stays L1-hot ⇒ fuse (no gather pass)
                            _gemv!(
                                true, false, jk, L, one(T), view(A, k:(j - 1), (j + 1):n),
                                view(scr, 1:jk), 1, zero(T), view(scr, (nb + 1):(nb + L)), 1
                            )
                            if usebuf
                                for c in 1:L
                                    v = (A[j, j + c] - scr[nb + c]) * invajj
                                    A[j, j + c] = v; rowb[j + c] = v
                                end
                            else
                                for c in 1:L
                                    A[j, j + c] = (A[j, j + c] - scr[nb + c]) * invajj
                                end
                            end
                        else                              # row spills L1 ⇒ split: gather, gemv, pure store
                            for c in 1:L
                                scr[nb + c] = A[j, j + c]
                            end
                            _gemv!(
                                true, false, jk, L, -one(T), view(A, k:(j - 1), (j + 1):n),
                                view(scr, 1:jk), 1, one(T), view(scr, (nb + 1):(nb + L)), 1
                            )
                            if usebuf
                                for c in 1:L
                                    v = scr[nb + c] * invajj
                                    A[j, j + c] = v; rowb[j + c] = v
                                end
                            else
                                for c in 1:L
                                    A[j, j + c] = scr[nb + c] * invajj
                                end
                            end
                        end
                    else                              # first column of a panel: no history to apply
                        if usebuf
                            for c in (j + 1):n
                                v = A[j, c] * invajj
                                A[j, c] = v; rowb[c] = v
                            end
                        else
                            for c in (j + 1):n
                                A[j, c] *= invajj
                            end
                        end
                    end
                end
            end
        end
        LOWER && _pstrf_apply_swaps_lower!(A, swj, swp, nsw, k - 1)   # batched leading row swaps
        r0 = k + jb                                          # rank-jb BLAS-3 trailing update
        if r0 <= n
            _pstrf_trailing!(
                view(A, r0:n, r0:n),
                LOWER ? view(A, r0:n, k:(k + jb - 1)) : view(A, k:(k + jb - 1), r0:n),
                Val(LOWER), T
            )
        end
    end
    return n
end

"""
    pstrf!(A, piv, tol; uplo='L') -> (A, piv, rank, info)

Cholesky with complete (diagonal) pivoting of a Hermitian positive-semidefinite `A` (LAPACK
{d,z}pstf2). Overwrites the `uplo` triangle of A with the (rank-truncated) Cholesky factor; `piv`
(preallocated length ≥ n) receives the pivot permutation, `rank` the numerical rank, `info` = 0 if
full-rank else 1. `tol < 0` ⇒ default LAPACK stop = n·eps·max_diagonal.

`Pᵀ·A·P` reconstructs from the factor: LOWER ⇒ L(:,1:rank)·L(:,1:rank)ᴴ, UPPER ⇒ U(1:rank,:)ᴴ·U(1:rank,:),
with P defined by `piv`.
"""
function pstrf!(A::AbstractMatrix{T}, piv::AbstractVector{<:Integer}, tol::Real; uplo::Char = 'L') where {T}
    R = real(T); n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("pstrf!: A must be square"))
    length(piv) >= n || throw(DimensionMismatch("pstrf!: length(piv) < n"))
    lower = uplo == 'L'
    lower || uplo == 'U' || throw(ArgumentError("pstrf!: uplo must be 'L' or 'U'"))
    n == 0 && return A, piv, 0, 0
    @inbounds for i in 1:n
        piv[i] = i
    end
    work = zeros(R, 2n)                                          # work[1:n] running dot products; [n+1:2n] scratch
    # Initial pivot = largest diagonal.
    pvt = 1; ajj = real(@inbounds A[1, 1])
    @inbounds for i in 2:n
        d = real(A[i, i]); (d > ajj) && (ajj = d; pvt = i)
    end
    if ajj <= zero(R) || isnan(ajj)
        @inbounds A[1, 1] = T(ajj)
        return A, piv, 0, 1
    end
    dstop = tol < 0 ? n * eps(R) * ajj : R(tol)
    # Blocked BLAS-3 path (dpstrf) once there is a trailing block worth a syrk; the unblocked kernel
    # below stays for small n and for non-BLAS element types (generic/AD, where syrk!/gemv! have no
    # SIMD path anyway). Identical results — the blocked form only defers the same updates.
    # nb is CAPPED at n so small matrices take the same driver with a single panel and no trailing syrk:
    # that still replaces the unblocked kernel's scalar rank-(j-1) column update with the SIMD `_gemv!`
    # (identical algorithm — with k=1 the panel spans the full history and the swap deferral is a no-op).
    nb = min(_pstrf_nb(n), n)
    if T <: Union{BlasReal, BlasComplex}
        scr = Vector{T}(undef, nb + 2n)     # +n: UPPER contiguous row cache (see rowb in _pstrf_blocked!)
        rank = if lower
            _pstrf_blocked!(A, piv, work, scr, dstop, pvt, ajj, nb, Val(true))
        else
            _pstrf_blocked!(A, piv, work, scr, dstop, pvt, ajj, nb, Val(false))
        end
        return A, piv, rank, (rank == n ? 0 : 1)
    end
    rank = n
    @inbounds if lower
        for j in 1:n
            for i in j:n                                        # update Schur-complement diagonals
                (j > 1) && (work[i] += abs2(A[i, j - 1]))
                work[n + i] = real(A[i, i]) - work[i]
            end
            if j > 1
                pvt = j; mx = work[n + j]
                for i in (j + 1):n
                    (work[n + i] > mx) && (mx = work[n + i]; pvt = i)
                end
                ajj = work[n + pvt]
                if ajj <= dstop || isnan(ajj)
                    A[j, j] = T(ajj); rank = j - 1; break
                end
            end
            if j != pvt
                _pstrf_swap_lower!(A, j, pvt, n)
                work[pvt] = work[j]
                piv[j], piv[pvt] = piv[pvt], piv[j]
            end
            ajj = sqrt(ajj); A[j, j] = T(ajj)
            if j < n
                for l in 1:(j - 1)                                  # A[j+1:n,j] −= A[j+1:n,1:j-1]·conj(A[j,1:j-1])
                    ajl = conj(A[j, l])
                    for i in (j + 1):n
                        A[i, j] -= A[i, l] * ajl
                    end
                end
                invajj = one(R) / ajj
                for i in (j + 1):n
                    A[i, j] *= invajj
                end
            end
        end
    else
        for j in 1:n
            for i in j:n
                (j > 1) && (work[i] += abs2(A[j - 1, i]))
                work[n + i] = real(A[i, i]) - work[i]
            end
            if j > 1
                pvt = j; mx = work[n + j]
                for i in (j + 1):n
                    (work[n + i] > mx) && (mx = work[n + i]; pvt = i)
                end
                ajj = work[n + pvt]
                if ajj <= dstop || isnan(ajj)
                    A[j, j] = T(ajj); rank = j - 1; break
                end
            end
            if j != pvt
                _pstrf_swap_upper!(A, j, pvt, n)
                work[pvt] = work[j]
                piv[j], piv[pvt] = piv[pvt], piv[j]
            end
            ajj = sqrt(ajj); A[j, j] = T(ajj)
            if j < n
                for c in (j + 1):n                                  # A[j,j+1:n] −= conj(A[1:j-1,j])·A[1:j-1,j+1:n]
                    s = zero(T)
                    for l in 1:(j - 1)
                        s += conj(A[l, j]) * A[l, c]
                    end
                    A[j, c] -= s
                end
                invajj = one(R) / ajj
                for c in (j + 1):n
                    A[j, c] *= invajj
                end
            end
        end
    end
    return A, piv, rank, (rank == n ? 0 : 1)
end

# Convenience: allocate piv (LinearAlgebra.LAPACK.pstrf!-style return).
function pstrf!(A::AbstractMatrix, tol::Real; uplo::Char = 'L')
    piv = Vector{Int}(undef, size(A, 1))
    return pstrf!(A, piv, tol; uplo = uplo)
end
