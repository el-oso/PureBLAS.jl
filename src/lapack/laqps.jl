# Blocked pivoted-QR panel — LAPACK `dlaqps`, real types. Called by `geqp3!` (geqp3.jl) for
# `T <: BlasReal && _strided1(A)` above `_QR_UNBLK_MAX`; the unblocked loop remains the fallback
# (complex, generic/AD, non-strided) and factors the tail after the last full panel.
#
# VERIFIED CORRECT: Q·R reconstruction vs A0[:,jpvt] — 11 shapes incl. rank-deficient, tall-skinny,
# short-fat, both real types: relerr ≤1.8e-15 (F64) / ≤1.3e-06 (F32); |diag R| non-increasing;
# sort(jpvt)==1:n; jpvt IDENTICAL to OpenBLAS on the LBT path; suite 164/164.
#
# PERF HISTORY — the first wiring of this port was CORRECT BUT 2.7-3.1x SLOWER and was reverted
# (2026-08-02). The culprit was found by PHASE DECOMPOSITION (timed copy of this function, Zen4
# freq-locked), NOT by shape/code reading — a prior static review concluded the inner ops were fine
# and was falsified:
#   the OLD step-5 pivot-row update walked the lda-strided row A[rk,k+1:n] once per rank-1 term
#   (k ≤ nb passes/column). At power-of-2 lda (8·lda = 4 KiB at n=512) every row element maps to the
#   SAME L1 set, so all passes thrash to L2/L3: measured 0.05 GFlop/s, 65-77% of panel time — the
#   entire regression. Every gemv/gemm was already on its SIMD kernel (predicates measured true; live
#   shapes within 0.75-1.0x of plain-Matrix probes). NOT the kb entry-overhead failure mode.
# THE FIX (current step 5): gather A[rk,1:k] into auxv, ONE contiguous SIMD gemv for the row delta
# into `wrow`, then a SINGLE strided pass applying the delta with the step-6 downdate folded in.
# Measured after fix (same harness, min-of-samples): blocked/unblocked runtime 0.82/0.70/0.74/0.73/
# 0.74x at n=128/256/512/1024/2048 — blocked wins everywhere it engages. Indicative same-process
# A/B vs AOCL libflame dgeqp3 (LBT-forwarded, 1 thread): 1.16/1.07/1.09/0.98/0.75 at n=128..2048
# (NOT gate numbers — bench/plots.jl is authoritative).
#
# n=2048 GAP — RESOLVED (2026-08-02): the F-build gemv-T was 79.8% of panel time at 6.1 GF/s while
# the same kernel/shape reached 10.8 standalone. Root cause was NOT dirty-DRAM writeback contention
# (in-context replay: dirtying the block before the sweeps changes NOTHING, 7.42 vs 7.54 GF/s —
# writebacks are ~3% of a panel's sweep traffic) and NOT L3 capacity per se. It was gemv-T ROUTING:
# on Zen4 `_gemvt_perscan`'s clean standalone probe picks the per-column dot inside its window, but
# the F-build re-sweeps the SAME trailing block once per panel column, and in that repeated-sweep
# regime the NC=4 blocked kernel wins at EVERY size (38 vs 30 GB/s at the 2048 trailing shape;
# live geqp3 1.03/1.08/1.11/1.34x at n=256/512/1024/2048). Fix: `_qp_gemv_t!` forces the blocked
# kernel (see its header). Zen3 was never affected — its probe already picks blocked, which is why
# the miss was Zen4-only. nb sweep at 2048: 32 (derived) beats 64/128 — deeper panels falsified.
#
# Other fixes vs the first (NaN-producing) port, verified against reference dlaqps:
#   1. UNIT HEAD: dlaqps sets A(RK,K)=ONE after dlarfg and restores AKK only after the F gemvs AND
#      the pivot-row update — the reflector multiplies as [1; v]. The old port left β at the head.
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

# A/B switch for measurement: lets one process time blocked vs unblocked geqp3! back-to-back
# (controlled A/B, not cross-run). true = geqp3! may take the blocked path.

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
# FORCED-BLOCKED gemv-T (measured 2026-08-02, Zen4 freq-locked): the F-build re-sweeps the SAME
# trailing block once per panel column, and in that repeated-sweep regime the NC=4 blocked kernel
# beats the per-column dot at EVERY size — in-context replay 38 vs 30 GB/s at the n=2048 trailing
# shape (dirty-vs-clean A/B: NO difference, the writeback-contention hypothesis is falsified — the
# regime itself is the effect), live geqp3 1.03/1.08/1.11/1.34x at n=256/512/1024/2048, the 2048
# cell 0.75→1.01 vs AOCL. `_gemvt_perscan`'s clean standalone probe ranks per-column ahead inside
# its window on Zen4 and routed the live path there — probe-regime-must-match-live. Zen3 (probe
# false, blocked everywhere) is bit-identical under the force. No tuning constant: this deletes a
# mis-applied Measure knob from a regime it never measured, rather than adding a knob.
@inline function _qp_gemv_t!(A, ar::Int, ac::Int, nr::Int, nc::Int, X, xr::Int, xc::Int, α, y)
    (nr <= 0 || nc <= 0) && return
    T = eltype(A)
    Av = view(A, ar:(ar + nr - 1), ac:(ac + nc - 1))
    xv = _qp_colv(X, xr, xr + nr - 1, xc)
    yv = view(y, 1:nc)
    if T <: BlasReal && _l2_simd_ok(Av, xv, yv, 1, 1)
        _gemv_t_simd!(nr, nc, T(α), Av, xv, zero(T), yv, Val(true), true)
    else
        _gemv!(true, false, nr, nc, T(α), Av, xv, 1, zero(T), yv, 1)
    end
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
        F::AbstractMatrix{T}, wrow::AbstractVector{T}
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

            # ── 5.+6. pivot row update + norm downdate, ONE strided pass ─────────────────────────
            # A[rk,k+1:n] -= A[rk,1:k]·F[k+1:n,1:k]ᵀ (r=k term uses the unit head A[rk,k]=1).
            # MEASURED HAZARD (this was the whole regression): walking the lda-strided pivot row
            # per rank-1 term is catastrophic at power-of-2 lda — every element maps to the same
            # L1 set (lda·8 = 4 KiB at n=512/1024), so k read-modify-write passes all thrash to
            # L2/L3: 0.05 GFlop/s, 65-77% of panel time, the 2.7-3.1x end-to-end loss. Instead:
            # gather A[rk,1:k] (k≤nb elements), ONE contiguous SIMD gemv for the delta into wrow,
            # then a SINGLE strided pass that applies the delta and folds in the downdate (which
            # previously re-read the same pathological row a second time).
            @inbounds for r in 1:k
                auxv[r] = A[rk, r]
            end
            _gemv!(
                false, false, n - k, k, -one(T), view(F, (k + 1):n, 1:k),
                view(auxv, 1:k), 1, zero(T), view(wrow, 1:(n - k)), 1
            )
            if rk < lastrk
                # dlaqps guards the downdate with RK < LASTRK: on the last row the trailing
                # vectors below rk are empty and the downdate is meaningless.
                @inbounds for j in (k + 1):n
                    aj = A[rk, j] + wrow[j - k]
                    A[rk, j] = aj
                    if !iszero(vn1[j])
                        temp = one(T) - (abs(aj) / vn1[j])^2
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
            else
                @inbounds for j in (k + 1):n
                    A[rk, j] += wrow[j - k]
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
