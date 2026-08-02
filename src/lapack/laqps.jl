# ⚠ CORRECT BUT ~2-3x SLOWER THAN THE UNBLOCKED PATH. NOT INCLUDED, NOT CALLED. Do not wire this in
# without first fixing the cause below — it was wired, measured, and reverted on 2026-08-02.
#
# This is a VERIFIED-CORRECT port of LAPACK `dlaqps` (blocked pivoted-QR panel). Correctness is not the
# problem and is no longer a confound:
#   Q·R reconstruction vs A0[:,jpvt] — 11 shapes incl. rank-deficient, tall-skinny, short-fat, both real
#   types: relerr 6.5e-16 .. 1.8e-15 (F64), 3.4e-07 .. 1.3e-06 (F32); |diag R| non-increasing;
#   sort(jpvt)==1:n; C-ABI/LBT path jpvt IDENTICAL to OpenBLAS; existing suite 164/164.
#
# MEASURED GATE, Zen4 freq-locked, PB/AOCL (v3 `op=geqp3 arms=pb`, references from the same cache):
#     n=        32      128      256      512     1024     2048
#   unblocked  0.856    0.917    0.756    0.832    0.722    0.561
#   BLOCKED    0.846    0.785    0.456    0.282    0.247    0.417
# Every cell REGRESSED, worst ~3x at n=512-1024. The unblocked path was reinstated.
#
# THIS IS THE SECOND TIME A BLOCKED geqp3 HAS BEEN CORRECT BUT SLOWER HERE. The first attempt is
# recorded in kb `pureblas-blas2-entry-overhead-blocks-blocked-lapack.md`. That the ALGORITHM is right
# (it is LAPACK's own, and it is what AOCL runs to beat us) while this implementation of it is 2-3x
# slower isolates the cause to the inner operations, not the blocking:
#   * the F-build gemv-T is (m-rk)x(n-k) and carries ~half the panel flops — if it is not hitting the
#     SIMD gemv-T kernel, the whole port is BLAS-2 with extra bookkeeping;
#   * a static review (Fable, 2026-08-02) concluded the gemvs were "structurally fine at gate size".
#     THE MEASUREMENT FALSIFIED THAT. Do not re-run that analysis and believe it — profile instead.
#
# NEXT STEP IS A DECOMPOSITION, NOT A REWRITE: time the four inner ops separately (step-2 gemv, F-build
# gemv-T, F acc gemv, trailing `_gemm_core!`) against their own roofline at n=512/1024, and find which
# one is not reaching its kernel. Only then decide whether to fix the call or abandon the approach.
# Suspect #1 is that these views do not satisfy `_l2_simd_ok`, silently taking the generic scalar loop.
#   2. The norm downdate ran unguarded; dlaqps guards it with `RK < LASTRK` (skip on the last row —
#      the trailing vectors below rk are empty and the downdate is meaningless there).
#   3. The LSTICC recompute used a raw sum of squares; req#6 demands lassq-safe `_nrm2`.
# (The old port's restriction of the incremental F update to rows k+1:n — LAPACK updates rows 1:N —
# was checked and is SOUND: every later read of F touches only rows j > column-index, an invariant
# closed under the row swaps, so rows j ≤ k of F(:,k) are never consumed. Kept, it saves work.)
#
# Structure per panel column k (rk = offset+k is the pivot row):
#   1. pivot on the largest partial norm, swap A/F/jpvt/vn1/vn2
#   2. bring column k up to date w.r.t. the k-1 deferred reflectors: A[rk:m,k] -= A[rk:m,1:k-1]·F[k,1:k-1]ᵀ
#   3. generate the reflector for A[rk:m,k]; set the unit head A[rk,k]=1
#   4. extend F: F[k+1:n,k] = τ·A[rk:m,k+1:n]ᵀ·v, then auxv = -τ·A[rk:m,1:k-1]ᵀ·v and
#      F[k+1:n,k] += F[k+1:n,1:k-1]·auxv
#   5. update ONLY the pivot row A[rk,k+1:n] (the rest of the trailing matrix waits for the gemm)
#   6. downdate the partial norms (guarded rk < lastrk); restore A[rk,k] = β
# Then once per panel: A[rk+1:m, kb+1:n] -= A[rk+1:m,1:kb]·F[kb+1:n,1:kb]ᵀ  ← the BLAS-3 payoff.
#
# Returns `kb`, the number of columns actually factored: the norm-recompute branch (LSTICC) can stop
# a panel early, and the caller MUST honour the returned value rather than assuming `nb`.

# ── panel helpers ────────────────────────────────────────────────────────────────────────────────
# All take explicit (row, col, extent) rather than pre-built SubArrays so each view is built exactly
# once at the call; they route to the positional `_gemv!`/`_gemm_core!` core entries directly.

# Column segment A[r1:r2, c] as a vector view. Composed column-then-range ON PURPOSE: for a Matrix
# both steps stay StridedVector (SIMD-eligible); for the C-ABI `PtrMatrix` both steps stay the isbits
# `PtrVector` (ptrmat.jl has no (range, Int) view method, so the direct `view(A, r1:r2, c)` would fall
# to a SubArray-of-PtrMatrix — non-strided, generic-kernel-only).
@inline _qp_colv(A, r1::Int, r2::Int, c::Int) = view(view(A, :, c), r1:r2)

# Y[yr:yr+nr-1, yc] -= A[ar:ar+nr-1, ac:ac+nc-1] · x[1:nc]
@inline function _qp_gemv_n!(A, ar::Int, ac::Int, nr::Int, nc::Int, x, Y, yr::Int, yc::Int)
    (nr <= 0 || nc <= 0) && return
    T = eltype(A)
    _gemv!(
        false, false, nr, nc, -one(T), view(A, ar:(ar + nr - 1), ac:(ac + nc - 1)),
        view(x, 1:nc), 1, one(T), _qp_colv(Y, yr, yr + nr - 1, yc), 1
    )
    return
end

# y[1:nc] = α · A[ar:ar+nr-1, ac:ac+nc-1]ᵀ · X[xr:xr+nr-1, xc]      (β = 0: overwrite)
@inline function _qp_gemv_t!(A, ar::Int, ac::Int, nr::Int, nc::Int, X, xr::Int, xc::Int, α, y)
    (nr <= 0 || nc <= 0) && return
    T = eltype(A)
    _gemv!(
        true, false, nr, nc, T(α), view(A, ar:(ar + nr - 1), ac:(ac + nc - 1)),
        _qp_colv(X, xr, xr + nr - 1, xc), 1, zero(T), view(y, 1:nc), 1
    )
    return
end

# Y[yr:yr+nr-1, yc] += F[fr:fr+nr-1, fc:fc+nc-1] · x[1:nc]
@inline function _qp_gemv_acc!(F, fr::Int, fc::Int, nr::Int, nc::Int, x, Y, yr::Int, yc::Int)
    (nr <= 0 || nc <= 0) && return
    T = eltype(F)
    _gemv!(
        false, false, nr, nc, one(T), view(F, fr:(fr + nr - 1), fc:(fc + nc - 1)),
        view(x, 1:nc), 1, one(T), _qp_colv(Y, yr, yr + nr - 1, yc), 1
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

# A is the m×n sub-block (all m rows, the not-yet-finished columns); offset = rows already factored
# above this panel, so rk = offset+k is the pivot row of panel column k. jpvt/tau/vn1/vn2 are the
# matching sub-views; F is n×nb (ldf = n of the sub-block), auxv length ≥ nb.
function _laqps!(
        m::Int, n::Int, offset::Int, nb::Int, A::AbstractMatrix{T},
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
        #      Row k of F is ldf-strided; copy it into auxv so the gemv x is unit-stride (SIMD path).
        if k > 1
            @inbounds for r in 1:(k - 1)
                auxv[r] = F[k, r]
            end
            _qp_gemv_n!(A, rk, 1, m - rk + 1, k - 1, auxv, A, rk, k)
        end

        # ── 3. reflector for the (now current) column k ────────────────────────────────────────────
        if rk < m
            β, τ = _larfg!(_qp_colv(A, rk, m, k))        # leaves A[rk,k]=α; β is R's diagonal entry
            tau[k] = τ
            akk = β
        else
            tau[k] = zero(T)                             # dlarfg(1): τ=0, β=α (A[rk,k] unchanged)
            akk = @inbounds A[rk, k]
        end
        # dlaqps: AKK = A(RK,K); A(RK,K) = ONE — the reflector vector multiplies as [1; v] in BOTH
        # F-building gemvs and the pivot-row update below. Restored to β at the end of the iteration.
        @inbounds A[rk, k] = one(T)

        if k < n
            # ── 4. F[k+1:n,k] = τ·A[rk:m,k+1:n]ᵀ·v, minus the accumulated contribution ────────────
            τ = tau[k]
            if !iszero(τ)
                _qp_gemv_t!(A, rk, k + 1, m - rk + 1, n - k, A, rk, k, τ, _qp_colv(F, k + 1, n, k))
                if k > 1
                    #   auxv[1:k-1] = -τ · A[rk:m,1:k-1]ᵀ · v   (columns 1:k-1 below row rk hold the
                    #   essential parts of the earlier reflectors — rk is below each of their heads)
                    _qp_gemv_t!(A, rk, 1, m - rk + 1, k - 1, A, rk, k, -τ, auxv)
                    #   F[k+1:n,k] += F[k+1:n,1:k-1] · auxv[1:k-1]
                    #   (dlaqps updates rows 1:n; rows ≤ k are provably never read — see header)
                    _qp_gemv_acc!(F, k + 1, 1, n - k, k - 1, auxv, F, k + 1, k)
                end
            else
                @inbounds for r in (k + 1):n
                    F[r, k] = zero(T)
                end
            end
            @inbounds for r in 1:k
                F[r, k] = zero(T)
            end

            # ── 5. update the PIVOT ROW only: A[rk,k+1:n] -= A[rk,1:k]·F[k+1:n,1:k]ᵀ ──────────────
            #      (r=k term uses the unit head A[rk,k]=1.) r outer so each F column streams
            #      contiguously; the strided A-row writes revisit the same cache lines k≤nb times.
            @inbounds for r in 1:k
                arkr = A[rk, r]
                iszero(arkr) && continue
                @simd for j in (k + 1):n
                    A[rk, j] = muladd(-arkr, F[j, r], A[rk, j])
                end
            end

            # ── 6. downdate the partial norms (dlaqps guards this with RK < LASTRK: on the last
            #      row the trailing vectors below rk are empty and the downdate is meaningless) ────
            if rk < lastrk
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

        @inbounds A[rk, k] = akk                         # restore β (R's diagonal) over the unit head
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
        nrm = _nrm2(m - rk, _qp_colv(A, rk + 1, m, lsticc), 1)   # lassq-safe (req#6), as dlaqps uses dnrm2
        vn1[lsticc] = nrm; vn2[lsticc] = nrm
        lsticc = itemp
    end
    return kb
end
