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

# Returns `rank` (n if full-rank). `scr` is scratch: scr[1:nb] the contiguous panel-row gather (the
# gemv's x, strided in A), scr[nb+1:nb+n] the UPPER path's contiguous output row (A[j,j+1:n] is
# strided by lda, so gather → gemv → scatter keeps the SIMD gemv on unit stride).
function _pstrf_blocked!(
        A::AbstractMatrix{T}, piv::AbstractVector{<:Integer}, work::AbstractVector{R},
        scr::AbstractVector{T}, dstop::R, pvt_in::Int, ajj_in::R, nb::Int, ::Val{LOWER}
    ) where {T, R <: Real, LOWER}
    n = size(A, 1)
    pvt = pvt_in; ajj = ajj_in
    swj = Vector{Int}(undef, nb); swp = Vector{Int}(undef, nb)   # deferred leading row swaps (LOWER)
    @inbounds for k in 1:nb:n
        jb = min(nb, n - k + 1)
        nsw = 0
        for i in k:n
            work[i] = zero(R)                       # per-panel reset (prior panels live in A[i,i])
        end
        for j in k:(k + jb - 1)
            for i in j:n                            # Schur-complement diagonals
                (j > k) && (work[i] += abs2(LOWER ? A[i, j - 1] : A[j - 1, i]))
                work[n + i] = real(A[i, i]) - work[i]
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
                            for c in 1:L
                                A[j, j + c] = (A[j, j + c] - scr[nb + c]) * invajj
                            end
                        else                              # row spills L1 ⇒ split: gather, gemv, pure store
                            for c in 1:L
                                scr[nb + c] = A[j, j + c]
                            end
                            _gemv!(
                                true, false, jk, L, -one(T), view(A, k:(j - 1), (j + 1):n),
                                view(scr, 1:jk), 1, one(T), view(scr, (nb + 1):(nb + L)), 1
                            )
                            for c in 1:L
                                A[j, j + c] = scr[nb + c] * invajj
                            end
                        end
                    else
                        for c in (j + 1):n
                            A[j, c] *= invajj
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
        scr = Vector{T}(undef, nb + n)
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
