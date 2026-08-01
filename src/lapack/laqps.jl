# ⚠ WORK IN PROGRESS — NOT INCLUDED BY src/PureBLAS.jl, NOT CALLED BY geqp3!.
#
# This file is a partial port of LAPACK `dlaqps` (blocked pivoted-QR panel). It is committed as
# scaffolding for the next attempt, NOT as working code: when wired into `geqp3!` it produced NaN in
# the factor at every size tried (60×60 … 256×256, both real types), while still returning a valid
# permutation — i.e. the pivoting bookkeeping is plausible and the linear algebra is not.
#
# KNOWN BUG, found by reading LAPACK's dlaqps against this: around the two gemvs that build F, LAPACK
# does
#       AKK = A( RK, K );  A( RK, K ) = ONE      ! implicit unit head of the Householder vector
#       ...DGEMV using A( RK:M, K )...
#       A( RK, K ) = AKK
# so the reflector vector multiplies as [1; v]. This port omits that, multiplying by β (the R diagonal
# `_larfg!` just wrote) instead of 1. That alone corrupts F and hence every deferred update. There may
# be further index errors behind it — the NaN was not chased past this point.
#
# WHEN RESUMING: fix the unit-head handling FIRST, then re-run the reconstruction test (Q·R vs A[:,jpvt]
# plus the non-increasing |diag(R)| pivot invariant) before looking at any timing. The measured prize is
# real — geqp3 is the largest single gate gap in the fleet, 0.561 Zen4 / 0.621 Zen3 vs AOCL at n=2048 —
# and the cause is confirmed structural: this is an O(m·n²) BLAS-2 factorisation racing a blocked one.
#
# Note the earlier blocked attempt (recorded in the kb) was CORRECT but NOT FASTER, killed by per-column
# BLAS-2 entry cost. So correctness is necessary but not sufficient here: once it reconstructs, the
# panel gemvs must be checked for entry overhead before concluding anything from a gate number.
# Blocked pivoted QR panel — LAPACK `dlaqps`, real types.
#
# WHY THIS EXISTS. `geqp3!` was a faithful UNBLOCKED port (`dlaqp2` semantics): per column it applied one
# rank-1 Householder across the ENTIRE trailing matrix, so the whole factorisation is O(m·n²) in BLAS-2.
# AOCL/LAPACK run `dlaqps`, which defers those updates into an auxiliary `F` and settles the trailing
# submatrix with ONE rank-`kb` gemm per panel — the same flops routed through a packed microkernel.
# Measured gap before this: geqp3 0.561 (Zen4) / 0.621 (Zen3) vs AOCL at n=2048, the largest single
# gate gap in the fleet.
#
# THE PREVIOUS BLOCKED ATTEMPT WAS CORRECT BUT NOT FASTER — killed by per-column BLAS-2 entry cost, the
# same failure mode as pptrfL/sytrs (see kb `blas2-entry-overhead-blocks-blocked-lapack`). The lesson
# encoded here: every inner gemv goes POINTER-DIRECT through `_gemv_*_simd!`-backed helpers on raw
# strided views, never through a kwarg `gemv!` on a SubArray, and the panel is sized so those gemvs are
# long enough to amortise their own entry.
#
# Structure per panel column k (rk = offset+k is the pivot row):
#   1. pivot on the largest partial norm, swap A/F/jpvt/vn1/vn2
#   2. bring column k up to date w.r.t. the k-1 deferred reflectors:  A[rk:m,k] -= A[rk:m,1:k-1]·F[k,1:k-1]ᵀ
#   3. generate the reflector for A[rk:m,k]
#   4. extend F:  F[k+1:n,k] = τ·A[rk:m,k+1:n]ᵀ·A[rk:m,k], then subtract the accumulated part
#   5. update ONLY the pivot row A[rk,k+1:n] (the rest of the trailing matrix waits for the gemm)
#   6. downdate the partial norms
# Then once per panel:  A[rk+1:m, kb+1:n] -= A[rk+1:m,1:kb] · F[kb+1:n,1:kb]ᵀ   ← the BLAS-3 payoff.
#
# Returns `kb`, the number of columns actually factored: the norm-recompute branch can stop a panel
# early, and the caller MUST honour the returned value rather than assuming `nb`.
# ── panel helpers ────────────────────────────────────────────────────────────────────────────────
# All four take explicit (row, col, extent) rather than pre-built SubArrays so the views are built once,
# contiguously, at the call — the previous blocked attempt died on per-column SubArray + kwarg-`gemv!`
# entry cost, so these go through the positional `_gemv!`/`_gemm_core!` core entries directly.

# Y[yr:yr+nr-1, yc] -= A[ar:ar+nr-1, ac:ac+nc-1] · x[1:nc]
@inline function _qp_gemv_n!(A, ar::Int, ac::Int, nr::Int, nc::Int, x, Y, yr::Int, yc::Int)
    (nr <= 0 || nc <= 0) && return
    T = eltype(A)
    _gemv!(
        false, false, nr, nc, -one(T), view(A, ar:(ar + nr - 1), ac:(ac + nc - 1)),
        view(x, 1:nc), 1, one(T), view(Y, yr:(yr + nr - 1), yc), 1
    )
    return
end

# Y[yr:yr+nc-1, yc] = α · A[ar:ar+nr-1, ac:ac+nc-1]ᵀ · X[xr:xr+nr-1, xc]      (β = 0: overwrite)
@inline function _qp_gemv_t!(
        A, ar::Int, ac::Int, nr::Int, nc::Int, X, xr::Int, xc::Int, α, Y, yr::Int, yc::Int
    )
    (nr <= 0 || nc <= 0) && return
    T = eltype(A)
    _gemv!(
        true, false, nr, nc, T(α), view(A, ar:(ar + nr - 1), ac:(ac + nc - 1)),
        view(X, xr:(xr + nr - 1), xc), 1, zero(T), view(Y, yr:(yr + nc - 1), yc), 1
    )
    return
end

# Y[yr:yr+nr-1, yc] += F[fr:fr+nr-1, fc:fc+nc-1] · x[1:nc]
@inline function _qp_gemv_acc!(F, fr::Int, fc::Int, nr::Int, nc::Int, x, Y, yr::Int, yc::Int)
    (nr <= 0 || nc <= 0) && return
    T = eltype(F)
    _gemv!(
        false, false, nr, nc, one(T), view(F, fr:(fr + nr - 1), fc:(fc + nc - 1)),
        view(x, 1:nc), 1, one(T), view(Y, yr:(yr + nr - 1), yc), 1
    )
    return
end

# THE BLAS-3 STEP: A[r0:r0+nr-1, c0:c0+nc-1] -= A[r0:r0+nr-1, 1:kb] · F[c0:c0+nc-1, 1:kb]ᵀ
# One rank-kb gemm replacing kb rank-1 updates over the whole trailing block — the entire point of the
# blocked algorithm. `_gemm_core!` (positional) and NOT the kwarg `gemm!`: a SubArray is not a
# StridedMatrix, so `gemm!` would silently fall to the generic kernel and hand back the BLAS-2 cost.
@inline function _qp_gemm_sub!(A, r0::Int, c0::Int, nr::Int, nc::Int, kb::Int, F)
    (nr <= 0 || nc <= 0 || kb <= 0) && return
    T = eltype(A)
    _gemm_core!(
        view(A, r0:(r0 + nr - 1), c0:(c0 + nc - 1)),
        view(A, r0:(r0 + nr - 1), 1:kb),
        view(F, c0:(c0 + nc - 1), 1:kb),
        -one(T), one(T), false, true, false, false
    )
    return
end

function _laqps!(
        m::Int, n::Int, offset::Int, nb::Int, A::AbstractMatrix{T}, lda::Int,
        jpvt::AbstractVector{<:Integer}, tau::AbstractVector{T},
        vn1::AbstractVector{T}, vn2::AbstractVector{T}, auxv::AbstractVector{T},
        F::AbstractMatrix{T}
    ) where {T <: BlasReal}
    tol3z = sqrt(eps(T))
    lastrk = min(m, offset + n)
    lsticc = 0            # column flagged for exact norm recompute; ends the panel (dlaqps `LSTICC`)
    k = 0

    while k < nb && lsticc == 0
        k += 1
        rk = offset + k

        # ── 1. pivot on the largest partial norm among the remaining columns ───────────────────────
        pvt = k; mx = vn1[k]
        @inbounds for j in (k + 1):n
            vn1[j] > mx && (mx = vn1[j]; pvt = j)
        end
        if pvt != k
            @inbounds for r in 1:m                       # swap FULL columns of A (all m rows)
                A[r, pvt], A[r, k] = A[r, k], A[r, pvt]
            end
            @inbounds for r in 1:(k - 1)                 # ...and the matching rows of F built so far
                F[pvt, r], F[k, r] = F[k, r], F[pvt, r]
            end
            jpvt[pvt], jpvt[k] = jpvt[k], jpvt[pvt]
            vn1[pvt] = vn1[k]; vn2[pvt] = vn2[k]
        end

        # ── 2. apply the k-1 DEFERRED reflectors to column k only ─────────────────────────────────
        #      A[rk:m,k] -= A[rk:m,1:k-1] · F[k,1:k-1]ᵀ   (one gemv, not k-1 rank-1 updates)
        if k > 1
            @inbounds for r in 1:(k - 1)
                auxv[r] = F[k, r]
            end
            _qp_gemv_n!(A, rk, 1, m - rk + 1, k - 1, auxv, A, rk, k)
        end

        # ── 3. reflector for the (now current) column k ────────────────────────────────────────────
        if rk < m
            β, τ = _larfg!(view(A, rk:m, k))
            A[rk, k] = β
            tau[k] = τ
        else
            tau[k] = zero(T)
        end

        # ── 4. F[k+1:n,k] = τ·A[rk:m,k+1:n]ᵀ·A[rk:m,k], minus the accumulated contribution ─────────
        if k < n
            τ = tau[k]
            if !iszero(τ)
                _qp_gemv_t!(A, rk, k + 1, m - rk + 1, n - k, A, rk, k, τ, F, k + 1, k)
                if k > 1
                    #   auxv[1:k-1] = -τ · A[rk:m,1:k-1]ᵀ · A[rk:m,k]
                    _qp_gemv_t!(A, rk, 1, m - rk + 1, k - 1, A, rk, k, -τ, F, 1, k)   # into F[1:k-1,k] scratch
                    @inbounds for r in 1:(k - 1)
                        auxv[r] = F[r, k]
                    end
                    #   F[k+1:n,k] += F[k+1:n,1:k-1] · auxv[1:k-1]
                    _qp_gemv_acc!(F, k + 1, 1, n - k, k - 1, auxv, F, k + 1, k)
                end
            else
                @inbounds for r in (k + 1):n
                    F[r, k] = zero(T)
                end
            end
            @inbounds F[k, k] = zero(T)
            @inbounds for r in 1:(k - 1)
                F[r, k] = zero(T)
            end

            # ── 5. update the PIVOT ROW only: A[rk,k+1:n] -= A[rk,1:k]·F[k+1:n,1:k]ᵀ ──────────────
            @inbounds for j in (k + 1):n
                s = zero(T)
                for r in 1:k
                    s += A[rk, r] * F[j, r]
                end
                A[rk, j] -= s
            end

            # ── 6. downdate the partial norms (identical rule to the unblocked path) ──────────────
            @inbounds for j in (k + 1):n
                if !iszero(vn1[j])
                    temp = one(T) - (abs(A[rk, j]) / vn1[j])^2
                    temp = max(temp, zero(T))
                    temp2 = temp * (vn1[j] / vn2[j])^2
                    if temp2 <= tol3z
                        vn2[j] = T(lsticc)      # dlaqps chains flagged columns through vn2
                        lsticc = j
                    else
                        vn1[j] *= sqrt(temp)
                    end
                end
            end
        end
    end

    kb = k
    rk = offset + kb
    # ── the BLAS-3 payoff: settle the whole remaining trailing block in ONE rank-kb gemm ───────────
    if kb < n && rk < m
        _qp_gemm_sub!(A, rk + 1, kb + 1, m - rk, n - kb, kb, F)
    end

    # ── exact recompute for every column the downdate flagged (dlaqps walks the vn2 chain) ────────
    while lsticc > 0
        itemp = Int(round(vn2[lsticc]))
        @inbounds begin
            s = zero(T)
            for r in (rk + 1):m
                s += A[r, lsticc] * A[r, lsticc]
            end
            vn1[lsticc] = sqrt(s); vn2[lsticc] = vn1[lsticc]
        end
        lsticc = itemp
    end
    return kb
end
