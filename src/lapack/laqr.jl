# LAPACK multishift QR with aggressive early deflation — the PERF path for `hseqr!`, which until now
# ran Reference-LAPACK `dlahqr` alone (classic Francis double shift: ONE 3×3 bulge chased down the
# whole matrix, BLAS-1.5 throughout). `hseqr.jl`'s own header called this driver "a PERF follow-up,
# explicitly out of scope"; this is that follow-up.
#
# WHY IT IS THE WHOLE CELL NOW. After the blocked `gehrd!` (hessenberg.jl, commit 367aec2) the
# eigenvalues-only `geev!` the gate benchmarks decomposes, measured on wintermute/Zen4, as:
#   n=256   gebal 0.8%   gehrd 42.4%   hseqr 54.2%    total ref/pb 0.446
#   n=512   gebal 0.6%   gehrd 16.9%   hseqr 83.0%    total ref/pb 0.379
#   n=1024  gebal 0.3%   gehrd  5.8%   hseqr 91.1%    total ref/pb 0.181
# — 6235 ms of 6848 at n=1024. Tuning `gehrd` further is worth a few percent; this is the campaign.
#
# THE TWO MECHANISMS, and they are independent wins:
#   * MULTISHIFT SWEEP (`_dlaqr5!`). Instead of one bulge, pack `nshfts/2` 3×3 bulges into a diagonal
#     window and chase them together. The reflectors are ACCUMULATED into a small orthogonal `U`, and
#     the far-field rows/columns are then updated with `U` as GEMM — the same level-2→level-3 move the
#     blocked `gehrd!` made, one routine along.
#   * AGGRESSIVE EARLY DEFLATION (`_aed!`). The classic subdiagonal test only deflates at the very
#     bottom. AED Schur-decomposes a trailing nw-window, tests the resulting "spike" for negligible
#     entries, and deflates eigenvalues that the classic test would not find for many more sweeps. At
#     large n this is a large share of dlaqr0's advantage, not a refinement of it.
#
# STAGING, stated because it is a deliberate simplification with a named upgrade path. Reference LAPACK
# has `dlaqr3` (AED) call `dlaqr4` (a second copy of the driver) for large windows, purely to bound
# recursion. `_aed!` here takes `dlaqr2`'s semantics instead: it always Schur-decomposes the window with
# `_dlahqr!`. That is CORRECT for every window size — it is what dlaqr2 does — and only leaves the very
# large windows slower than they could be.
# ponytail: recursive window solve (dlaqr3/dlaqr4) once the cell is green; correctness is unaffected.
#
# REAL ONLY, matching the blocked `gehrd!`: `bench/plots.jl:1003` benchmarks `geev` on
# `randn(Float64, s, s)`. Complex keeps `_zlahqr!`, which is correct and merely slow.

# ── DLAQR1: first column of the shift polynomial ─────────────────────────────────────────────────────
# v := a scalar multiple of the first column of (H − (sr1+i·si1)·I)·(H − (sr2+i·si2)·I), for the 2×2 or
# 3×3 leading block of H. Reference-LAPACK verbatim, including the `s` scaling that keeps the product
# from overflowing and the exact-zero bail-out.
@inline function _dlaqr1!(
        v::AbstractVector{R}, nr::Int, H::AbstractMatrix{R}, r0::Int,
        sr1::R, si1::R, sr2::R, si2::R
    ) where {R <: Real}
    @inbounds if nr == 2
        s = abs(H[r0, r0] - sr2) + abs(si2) + abs(H[r0 + 1, r0])
        if iszero(s)
            v[1] = zero(R); v[2] = zero(R)
        else
            h21s = H[r0 + 1, r0] / s
            v[1] = h21s * H[r0, r0 + 1] + (H[r0, r0] - sr1) * ((H[r0, r0] - sr2) / s) - si1 * (si2 / s)
            v[2] = h21s * (H[r0, r0] + H[r0 + 1, r0 + 1] - sr1 - sr2)
        end
    else
        s = abs(H[r0, r0] - sr2) + abs(si2) + abs(H[r0 + 1, r0]) + abs(H[r0 + 2, r0])
        if iszero(s)
            v[1] = zero(R); v[2] = zero(R); v[3] = zero(R)
        else
            h21s = H[r0 + 1, r0] / s
            h31s = H[r0 + 2, r0] / s
            v[1] = (H[r0, r0] - sr1) * ((H[r0, r0] - sr2) / s) - si1 * (si2 / s) +
                H[r0, r0 + 1] * h21s + H[r0, r0 + 2] * h31s
            v[2] = h21s * (H[r0, r0] + H[r0 + 1, r0 + 1] - sr1 - sr2) + H[r0 + 1, r0 + 2] * h31s
            v[3] = h31s * (H[r0, r0] + H[r0 + 2, r0 + 2] - sr1 - sr2) + h21s * H[r0 + 2, r0 + 1]
        end
    end
    return v
end

# The vigilant deflation test, used at two points in the sweep. Requires BOTH the traditional
# small-compared-to-nearby-diagonals criterion AND Ahues & Tisseur (LAWN 122, 1997); dropping either
# deflates too eagerly and silently loses eigenvalues. Reference-LAPACK verbatim.
@inline function _vigilant_deflate!(
        H::AbstractMatrix{R}, k::Int, ktop::Int, kbot::Int, smlnum::R, ulp::R
    ) where {R <: Real}
    @inbounds begin
        tst1 = abs(H[k, k]) + abs(H[k + 1, k + 1])
        if iszero(tst1)
            k >= ktop + 1 && (tst1 += abs(H[k, k - 1]))
            k >= ktop + 2 && (tst1 += abs(H[k, k - 2]))
            k >= ktop + 3 && (tst1 += abs(H[k, k - 3]))
            k <= kbot - 2 && (tst1 += abs(H[k + 2, k + 1]))
            k <= kbot - 3 && (tst1 += abs(H[k + 3, k + 1]))
            k <= kbot - 4 && (tst1 += abs(H[k + 4, k + 1]))
        end
        if abs(H[k + 1, k]) <= max(smlnum, ulp * tst1)
            h12 = max(abs(H[k + 1, k]), abs(H[k, k + 1]))
            h21 = min(abs(H[k + 1, k]), abs(H[k, k + 1]))
            h11 = max(abs(H[k + 1, k + 1]), abs(H[k, k] - H[k + 1, k + 1]))
            h22 = min(abs(H[k + 1, k + 1]), abs(H[k, k] - H[k + 1, k + 1]))
            scl = h11 + h12
            tst2 = h22 * (h11 / scl)
            if iszero(tst2) || h21 * (h12 / scl) <= max(smlnum, ulp * tst2)
                H[k + 1, k] = zero(R)
            end
        end
    end
    return H
end

# ── DLAQR5: multishift bulge chase with reflector accumulation ────────────────────────────────────────
# Reference-LAPACK verbatim. Chases `nshfts/2` double-shift bulges through `H[ktop:kbot, :]` as one
# chain. Every reflector is also applied to a `kdu×kdu` identity `U`, and the far-field rows/columns are
# then updated by GEMM with `U` — that is where the level-3 arithmetic comes from. The near-diagonal
# slab stays scalar because the bulges overlap there and cannot be batched.
#
# `V` is 3×nbmps reflector storage in LAPACK's layout: `V[1,m]` holds τ (NOT β) and `V[2:3,m]` the
# essential part, which is why `_hqr_larfg!`'s β return is swapped out at each call site below.
# `WH`/`WV` are the GEMM landing buffers and `nh`/`nv` their blocking widths.
function _dlaqr5!(
        wantt::Bool, wantz::Bool, n::Int, ktop::Int, kbot::Int, nshfts::Int,
        sr::AbstractVector{R}, si::AbstractVector{R}, H::AbstractMatrix{R},
        iloz::Int, ihiz::Int, Z, V::AbstractMatrix{R}, U::AbstractMatrix{R},
        nv::Int, WV::AbstractMatrix{R}, nh::Int, WH::AbstractMatrix{R}
    ) where {R <: Real}
    nshfts < 2 && return H
    ktop >= kbot && return H
    # Shuffle shifts into pairs of real shifts and pairs of complex conjugates, assuming conjugates are
    # already adjacent. The 2-shift bulge construction below depends on that ordering.
    @inbounds for i in 1:2:(nshfts - 2)
        if si[i] != -si[i + 1]
            sr[i], sr[i + 1], sr[i + 2] = sr[i + 1], sr[i + 2], sr[i]
            si[i], si[i + 1], si[i + 2] = si[i + 1], si[i + 2], si[i]
        end
    end
    ns = nshfts - (nshfts % 2)
    safmin = _hqr_safmin(R)
    ulp = eps(R)
    smlnum = safmin * (R(n) / ulp)
    @inbounds if ktop + 2 <= kbot
        H[ktop + 2, ktop] = zero(R)            # clear trash left by a previous sweep
    end
    nbmps = ns ÷ 2
    kdu = 4 * nbmps
    vt = zeros(R, 3)
    incol = ktop - 2 * nbmps + 1
    @inbounds while incol <= kbot - 2
        jtop = max(ktop, incol)
        ndcol = incol + kdu
        for c in 1:kdu, r in 1:kdu
            U[r, c] = (r == c) ? one(R) : zero(R)
        end
        krcol = incol
        while krcol <= min(incol + 2 * nbmps - 1, kbot - 2)
            mtop = max(1, (ktop - krcol) ÷ 2 + 1)
            mbot = min(nbmps, (kbot - krcol - 1) ÷ 2)
            m22 = mbot + 1
            bmp22 = (mbot < nbmps) && (krcol + 2 * (m22 - 1) == kbot - 2)
            # ---- special case: the 2×2 reflection at the bottom, handled apart from the chain ----
            if bmp22
                k = krcol + 2 * (m22 - 1)
                if k == ktop - 1
                    _dlaqr1!(view(V, :, m22), 2, H, k + 1, sr[2 * m22 - 1], si[2 * m22 - 1], sr[2 * m22], si[2 * m22])
                    τ = _hqr_larfg!(view(V, :, m22), 2)
                    V[1, m22] = τ
                else
                    V[1, m22] = H[k + 1, k]
                    V[2, m22] = H[k + 2, k]
                    τ = _hqr_larfg!(view(V, :, m22), 2)
                    β = V[1, m22]
                    V[1, m22] = τ
                    H[k + 1, k] = β
                    H[k + 2, k] = zero(R)
                end
                t1 = V[1, m22]; t2 = t1 * V[2, m22]
                for j in jtop:min(kbot, k + 3)
                    refsum = H[j, k + 1] + V[2, m22] * H[j, k + 2]
                    H[j, k + 1] -= refsum * t1
                    H[j, k + 2] -= refsum * t2
                end
                jb2 = min(ndcol, kbot)
                for j in (k + 1):jb2
                    refsum = H[k + 1, j] + V[2, m22] * H[k + 2, j]
                    H[k + 1, j] -= refsum * t1
                    H[k + 2, j] -= refsum * t2
                end
                if k >= ktop && !iszero(H[k + 1, k])
                    _vigilant_deflate!(H, k, ktop, kbot, smlnum, ulp)
                end
                kms = k - incol
                for j in max(1, ktop - incol):kdu
                    refsum = V[1, m22] * (U[j, kms + 1] + V[2, m22] * U[j, kms + 2])
                    U[j, kms + 1] -= refsum
                    U[j, kms + 2] -= refsum * V[2, m22]
                end
            end
            # ---- normal case: the chain of 3×3 reflections ----
            for m in mbot:-1:mtop
                k = krcol + 2 * (m - 1)
                if k == ktop - 1
                    _dlaqr1!(view(V, :, m), 3, H, ktop, sr[2 * m - 1], si[2 * m - 1], sr[2 * m], si[2 * m])
                    τ = _hqr_larfg!(view(V, :, m), 3)
                    V[1, m] = τ
                else
                    # Delayed transformation of the row below the m-th bulge.
                    t1 = V[1, m]; t2 = t1 * V[2, m]; t3 = t1 * V[3, m]
                    refsum = V[3, m] * H[k + 3, k + 2]
                    H[k + 3, k] = -refsum * t1
                    H[k + 3, k + 1] = -refsum * t2
                    H[k + 3, k + 2] -= refsum * t3
                    # Reflection moving the m-th bulge one step.
                    V[1, m] = H[k + 1, k]
                    V[2, m] = H[k + 2, k]
                    V[3, m] = H[k + 3, k]
                    τ = _hqr_larfg!(view(V, :, m), 3)
                    β = V[1, m]
                    V[1, m] = τ
                    # A bulge may collapse through vigilant deflation or destructive underflow.
                    if !iszero(H[k + 3, k]) || !iszero(H[k + 3, k + 1]) || iszero(H[k + 3, k + 2])
                        H[k + 1, k] = β
                        H[k + 2, k] = zero(R)
                        H[k + 3, k] = zero(R)
                    else
                        # Collapsed: try restarting the bulge from the shifts, and keep whichever of the
                        # two leaves only negligible fill.
                        _dlaqr1!(vt, 3, H, k + 1, sr[2 * m - 1], si[2 * m - 1], sr[2 * m], si[2 * m])
                        τv = _hqr_larfg!(vt, 3)
                        vt[1] = τv
                        t1 = vt[1]; t2 = t1 * vt[2]; t3 = t1 * vt[3]
                        refsum = H[k + 1, k] + H[k + 2, k] * vt[2] + H[k + 3, k] * vt[3]
                        if abs(H[k + 2, k] - refsum * t2) + abs(H[k + 3, k] - refsum * t3) >
                                ulp * (abs(H[k, k]) + abs(H[k + 1, k + 1]) + abs(H[k + 2, k + 2]))
                            H[k + 1, k] = β
                            H[k + 2, k] = zero(R)
                            H[k + 3, k] = zero(R)
                        else
                            H[k + 1, k] -= refsum * t1
                            H[k + 2, k] = zero(R)
                            H[k + 3, k] = zero(R)
                            V[1, m] = vt[1]; V[2, m] = vt[2]; V[3, m] = vt[3]
                        end
                    end
                end
                # Right apply, plus the first column of the left apply — both are needed by the vigilant
                # deflation check that follows.
                t1 = V[1, m]; t2 = t1 * V[2, m]; t3 = t1 * V[3, m]
                for j in jtop:min(kbot, k + 3)
                    refsum = H[j, k + 1] + V[2, m] * H[j, k + 2] + V[3, m] * H[j, k + 3]
                    H[j, k + 1] -= refsum * t1
                    H[j, k + 2] -= refsum * t2
                    H[j, k + 3] -= refsum * t3
                end
                refsum = H[k + 1, k + 1] + V[2, m] * H[k + 2, k + 1] + V[3, m] * H[k + 3, k + 1]
                H[k + 1, k + 1] -= refsum * t1
                H[k + 2, k + 1] -= refsum * t2
                H[k + 3, k + 1] -= refsum * t3
                if k >= ktop && !iszero(H[k + 1, k])
                    _vigilant_deflate!(H, k, ktop, kbot, smlnum, ulp)
                end
            end
            # ---- multiply H by the reflections from the left ----
            jbot = min(ndcol, kbot)
            for m in mbot:-1:mtop
                k = krcol + 2 * (m - 1)
                t1 = V[1, m]; t2 = t1 * V[2, m]; t3 = t1 * V[3, m]
                for j in max(ktop, krcol + 2 * m):jbot
                    refsum = H[k + 1, j] + V[2, m] * H[k + 2, j] + V[3, m] * H[k + 3, j]
                    H[k + 1, j] -= refsum * t1
                    H[k + 2, j] -= refsum * t2
                    H[k + 3, j] -= refsum * t3
                end
            end
            # ---- accumulate the orthogonal transformations into U ----
            for m in mbot:-1:mtop
                k = krcol + 2 * (m - 1)
                kms = k - incol
                i2 = max(max(1, ktop - incol), kms - (krcol - incol) + 1)
                i4 = min(kdu, krcol + 2 * (mbot - 1) - incol + 5)
                t1 = V[1, m]; t2 = t1 * V[2, m]; t3 = t1 * V[3, m]
                for j in i2:i4
                    refsum = U[j, kms + 1] + V[2, m] * U[j, kms + 2] + V[3, m] * U[j, kms + 3]
                    U[j, kms + 1] -= refsum * t1
                    U[j, kms + 2] -= refsum * t2
                    U[j, kms + 3] -= refsum * t3
                end
            end
            krcol += 1
        end
        # ---- use U to update the far-from-diagonal entries: THE level-3 step ----
        jt = wantt ? 1 : ktop
        jb = wantt ? n : kbot
        k1 = max(1, ktop - incol)
        nu = (kdu - max(0, ndcol - kbot)) - k1 + 1
        if nu > 0
            Uv = view(U, k1:(k1 + nu - 1), k1:(k1 + nu - 1))
            jcol = min(ndcol, kbot) + 1
            while jcol <= jb                                    # horizontal: H := Uᵀ·H
                jlen = min(nh, jb - jcol + 1)
                Hv = view(H, (incol + k1):(incol + k1 + nu - 1), jcol:(jcol + jlen - 1))
                Wv = view(WH, 1:nu, 1:jlen)
                gemm!(Wv, Uv, Hv; transA = 'T', alpha = one(R), beta = zero(R))
                copyto!(Hv, Wv)
                jcol += nh
            end
            jrow = jt
            while jrow <= max(ktop, incol) - 1                  # vertical: H := H·U
                jlen = min(nv, max(ktop, incol) - jrow)
                Hv = view(H, jrow:(jrow + jlen - 1), (incol + k1):(incol + k1 + nu - 1))
                Wv = view(WV, 1:jlen, 1:nu)
                gemm!(Wv, Hv, Uv; alpha = one(R), beta = zero(R))
                copyto!(Hv, Wv)
                jrow += nv
            end
            if wantz
                jrow = iloz
                while jrow <= ihiz                              # Z := Z·U
                    jlen = min(nv, ihiz - jrow + 1)
                    Zv = view(Z, jrow:(jrow + jlen - 1), (incol + k1):(incol + k1 + nu - 1))
                    Wv = view(WV, 1:jlen, 1:nu)
                    gemm!(Wv, Zv, Uv; alpha = one(R), beta = zero(R))
                    copyto!(Zv, Wv)
                    jrow += nv
                end
            end
        end
        incol += 2 * nbmps
    end
    return H
end

# ── IPARMQ: the shift/window sizing table ────────────────────────────────────────────────────────────
# Reference-LAPACK `iparmq` verbatim. These are the algorithm's OWN tuning table, not PureBLAS tuning:
# they select how many shifts a sweep carries and how wide the deflation window is, and LAPACK's values
# encode published convergence behaviour of the multishift iteration rather than any cache property. A
# derived-from-hardware form would be a different algorithm, not a retuning of this one — so this is a
# literal by nature, and marked accordingly.
# ponytail: a fleet sweep could confirm the crossovers, but the shape must stay LAPACK's. | req8-ok
@inline function _iparmq_ns(nh::Int)
    ns = 2
    nh >= 30 && (ns = 4)
    nh >= 60 && (ns = 10)
    nh >= 150 && (ns = max(10, nh ÷ round(Int, log(nh) / log(2))))
    nh >= 590 && (ns = 64)
    nh >= 3000 && (ns = 128)
    nh >= 6000 && (ns = 256)
    return max(2, ns - (ns % 2))
end
# Below this active-block size `dlahqr` beats the multishift machinery, so `_dlaqr0!` delegates to it.
#
# NOT LAPACK's 75. That value is ILAENV's, and it does not transfer: it is calibrated against reference
# LAPACK's own `dlahqr`, whereas this repo's `_dlahqr!` is a tuned port, so the multishift path has to
# clear a higher bar here. Below n≈200 it does not — `_dlaqr5!`'s far-field GEMM is the whole point of
# the sweep, and at these sizes `_iparmq_ns` gives ns=10, i.e. 5 bulges accumulated into a 20-wide `U`,
# which is too small a GEMM to pay for the AED window solve and the shift bookkeeping around it.
#
# MEASURED, both µarchs, `_dlaqr0!` / `_dlahqr!` on the same Hessenberg input, Chairmarks median, cold
# operand per sample, BOTH BOXES FREQUENCY-LOCKED (>1 means the multishift driver is SLOWER):
#     n        90    100    120    128    160    200    256
#     Zen3   2.094  1.771  1.513  1.307  1.822  1.612  0.721
#     Zen4   2.090  1.818  1.929  1.543  1.930  1.740  0.698
# The two boxes agree to within a few percent and cross between 200 and 256, so this is a property of
# the implementation rather than of one machine. 200 is the largest measured size where `dlahqr` still
# wins; the gate ladder has nothing between 200 and 256, so any value in [200, 255] is equivalent there.
#
# This cost a real gate cell: at `_LAQR_NMIN = 75` the n=100 `geev` cell was 73.5% `hseqr` and failed at
# 0.985 on locked Zen3 — the failure was this crossover, not a kernel gap.
# req8-ok: a falsified derivation. The optimum is a property of the two drivers' relative cost, not of a
# detected cache or ISA constant, and the fleet table above is the evidence. | tune: candidate
const _LAQR_NMIN = 200
const _LAQR_NIBBLE = 14    # skip a sweep when AED deflated ≥ this % of the window (ispec 14)
const _LAQR_KNWSWP = 500   # above this nh, widen the window to 3·ns/2 (ispec 13)
const _LAQR_KEXNW = 5      # exceptional window growth after this many deflation-free iterations
const _LAQR_KEXSH = 6      # exceptional shifts after this many deflation-free iterations

# ── Aggressive early deflation (dlaqr2 semantics) ────────────────────────────────────────────────────
# Schur-decompose the trailing `nw`-window, test the spike for negligible tips, deflate what it can, and
# hand back the undeflated eigenvalues as the next sweep's shifts. Returns `(ns, nd)` — shifts kept and
# eigenvalues deflated.
#
# The window solve uses `_dlahqr!` unconditionally (dlaqr2's choice). Reference LAPACK's dlaqr3 calls
# dlaqr4 for large windows purely to bound recursion; that is a speed refinement, not correctness.
function _aed!(
        wantt::Bool, wantz::Bool, n::Int, ktop::Int, kbot::Int, nw::Int,
        H::AbstractMatrix{R}, iloz::Int, ihiz::Int, Z,
        sr::AbstractVector{R}, si::AbstractVector{R},
        Tw::AbstractMatrix{R}, Vw::AbstractMatrix{R}, wk::AbstractVector{R},
        nv::Int, WV::AbstractMatrix{R}, nh::Int, WH::AbstractMatrix{R}
    ) where {R <: Real}
    (ktop > kbot || nw < 1) && return (0, 0)
    safmin = _hqr_safmin(R)
    ulp = eps(R)
    smlnum = safmin * (R(n) / ulp)
    jw = min(nw, kbot - ktop + 1)
    kwtop = kbot - jw + 1
    @inbounds s = (kwtop == ktop) ? zero(R) : H[kwtop, kwtop - 1]
    if kbot == kwtop
        # 1×1 window: deflate it or keep it as a shift, nothing else to do.
        @inbounds begin
            sr[kwtop] = H[kwtop, kwtop]
            si[kwtop] = zero(R)
            if abs(s) <= max(smlnum, ulp * abs(H[kwtop, kwtop]))
                kwtop > ktop && (H[kwtop, kwtop - 1] = zero(R))
                return (0, 1)
            end
        end
        return (1, 0)
    end
    # ---- convert the window to spike-triangular form ----
    @inbounds begin
        for c in 1:jw, r in 1:jw
            Tw[r, c] = (r <= c) ? H[kwtop + r - 1, kwtop + c - 1] : zero(R)
        end
        for j in 1:(jw - 1)
            Tw[j + 1, j] = H[kwtop + j, kwtop + j - 1]
        end
        for c in 1:jw, r in 1:jw
            Vw[r, c] = (r == c) ? one(R) : zero(R)
        end
    end
    infqr = _dlahqr!(
        true, true, view(Tw, 1:jw, 1:jw), 1, jw,
        view(sr, kwtop:(kwtop + jw - 1)), view(si, kwtop:(kwtop + jw - 1)),
        1, jw, view(Vw, 1:jw, 1:jw)
    )
    # trexc! needs a clean margin near the diagonal.
    @inbounds begin
        for j in 1:(jw - 3)
            Tw[j + 2, j] = zero(R)
            Tw[j + 3, j] = zero(R)
        end
        jw > 2 && (Tw[jw, jw - 2] = zero(R))
    end
    # ---- deflation detection: walk the spike from the bottom ----
    ns = jw
    ilst = infqr + 1
    @inbounds while ilst <= ns
        bulge = ns == 1 ? false : !iszero(Tw[ns, ns - 1])
        if !bulge
            foo = abs(Tw[ns, ns])
            iszero(foo) && (foo = abs(s))
            if abs(s * Vw[1, ns]) <= max(smlnum, ulp * foo)
                ns -= 1
            else
                trexc!('V', view(Tw, 1:jw, 1:jw), view(Vw, 1:jw, 1:jw), ns, ilst)
                ilst += 1
            end
        else
            foo = abs(Tw[ns, ns]) + sqrt(abs(Tw[ns, ns - 1])) * sqrt(abs(Tw[ns - 1, ns]))
            iszero(foo) && (foo = abs(s))
            if max(abs(s * Vw[1, ns]), abs(s * Vw[1, ns - 1])) <= max(smlnum, ulp * foo)
                ns -= 2
            else
                trexc!('V', view(Tw, 1:jw, 1:jw), view(Vw, 1:jw, 1:jw), ns, ilst)
                ilst += 2
            end
        end
    end
    iszero(ns) && (s = zero(R))
    # ---- sort the retained diagonal blocks by magnitude (accuracy on graded matrices) ----
    if ns < jw
        sorted = false
        i = ns + 1
        @inbounds while !sorted
            sorted = true
            kend = i - 1
            i = infqr + 1
            k = (i == ns || iszero(Tw[i + 1, i])) ? i + 1 : i + 2
            while k <= kend
                evi = (k == i + 1) ? abs(Tw[i, i]) :
                    abs(Tw[i, i]) + sqrt(abs(Tw[i + 1, i])) * sqrt(abs(Tw[i, i + 1]))
                evk = if k == kend || iszero(Tw[k + 1, k])
                    abs(Tw[k, k])
                else
                    abs(Tw[k, k]) + sqrt(abs(Tw[k + 1, k])) * sqrt(abs(Tw[k, k + 1]))
                end
                if evi >= evk
                    i = k
                else
                    sorted = false
                    trexc!('V', view(Tw, 1:jw, 1:jw), view(Vw, 1:jw, 1:jw), i, k)
                    i = k
                end
                k = (i == kend || iszero(Tw[i + 1, i])) ? i + 1 : i + 2
            end
        end
    end
    # ---- read the eigenvalues back out of the (quasi-)triangular window ----
    i = jw
    @inbounds while i >= infqr + 1
        if i == infqr + 1 || iszero(Tw[i, i - 1])
            sr[kwtop + i - 1] = Tw[i, i]
            si[kwtop + i - 1] = zero(R)
            i -= 1
        else
            aa, bb = Tw[i - 1, i - 1], Tw[i - 1, i]
            cc, dd = Tw[i, i - 1], Tw[i, i]
            _, _, _, _, r1r, r1i, r2r, r2i, _, _ = _dlanv2(aa, bb, cc, dd)
            sr[kwtop + i - 2] = r1r; si[kwtop + i - 2] = r1i
            sr[kwtop + i - 1] = r2r; si[kwtop + i - 1] = r2i
            i -= 2
        end
    end
    if ns < jw || iszero(s)
        if ns > 1 && !iszero(s)
            # Reflect the spike back into the lower triangle, then re-reduce to Hessenberg form.
            @inbounds for t in 1:ns
                wk[t] = Vw[1, t]
            end
            τ = _hqr_larfg!(view(wk, 1:ns), ns)
            @inbounds wk[1] = one(R)
            @inbounds for c in 1:(jw - 2), r in (c + 2):jw
                Tw[r, c] = zero(R)
            end
            _house_left!(view(Tw, 1:ns, 1:jw), view(wk, 1:ns), τ)
            _larf_right!(view(Tw, 1:ns, 1:ns), view(wk, 1:ns), τ)
            _larf_right!(view(Vw, 1:jw, 1:ns), view(wk, 1:ns), τ)
            gehrd!(view(Tw, 1:jw, 1:jw), 1, ns, view(wk, 1:max(jw - 1, 1)))
        end
        # ---- copy the reduced window back into H ----
        @inbounds begin
            kwtop > 1 && (H[kwtop, kwtop - 1] = s * Vw[1, 1])
            for c in 1:jw, r in 1:min(c, jw)
                H[kwtop + r - 1, kwtop + c - 1] = Tw[r, c]
            end
            for j in 1:(jw - 1)
                H[kwtop + j, kwtop + j - 1] = Tw[j + 1, j]
            end
        end
        if ns > 1 && !iszero(s)
            ormhr!('R', 'N', 1, ns, view(Tw, 1:jw, 1:jw), view(wk, 1:max(ns - 1, 1)), view(Vw, 1:jw, 1:jw))
        end
        # ---- apply the window's orthogonal factor to the slabs outside it, as GEMM ----
        ltop = wantt ? 1 : ktop
        Vv = view(Vw, 1:jw, 1:jw)
        krow = ltop
        @inbounds while krow <= kwtop - 1
            kln = min(nv, kwtop - krow)
            Hv = view(H, krow:(krow + kln - 1), kwtop:(kwtop + jw - 1))
            Wv = view(WV, 1:kln, 1:jw)
            gemm!(Wv, Hv, Vv; alpha = one(R), beta = zero(R))
            copyto!(Hv, Wv)
            krow += nv
        end
        if wantt
            kcol = kbot + 1
            @inbounds while kcol <= n
                kln = min(nh, n - kcol + 1)
                Hv = view(H, kwtop:(kwtop + jw - 1), kcol:(kcol + kln - 1))
                Wv = view(WH, 1:jw, 1:kln)
                gemm!(Wv, Vv, Hv; transA = 'T', alpha = one(R), beta = zero(R))
                copyto!(Hv, Wv)
                kcol += nh
            end
        end
        if wantz
            krow = iloz
            @inbounds while krow <= ihiz
                kln = min(nv, ihiz - krow + 1)
                Zv = view(Z, krow:(krow + kln - 1), kwtop:(kwtop + jw - 1))
                Wv = view(WV, 1:kln, 1:jw)
                gemm!(Wv, Zv, Vv; alpha = one(R), beta = zero(R))
                copyto!(Zv, Wv)
                krow += nv
            end
        end
    end
    return (ns - infqr, jw - ns)
end

# ── DLAQR0: the multishift + AED driver ──────────────────────────────────────────────────────────────
# Reference-LAPACK verbatim in structure. Each iteration: locate the active block, size and run the
# deflation window, then — unless AED deflated enough that another sweep would be wasted — pick shifts
# and run one multishift sweep. Returns LAPACK's `info` (0, or the index at which it gave up).
function _dlaqr0!(
        wantt::Bool, wantz::Bool, H::AbstractMatrix{R}, ilo::Int, ihi::Int,
        wr::AbstractVector{R}, wi::AbstractVector{R}, iloz::Int, ihiz::Int, Z
    ) where {R <: Real}
    n = size(H, 1)
    n == 0 && return 0
    if ihi - ilo + 1 <= _LAQR_NMIN
        return _dlahqr!(wantt, wantz, H, ilo, ihi, wr, wi, iloz, ihiz, Z)
    end
    # Clear the trash below the subdiagonal that a previous sweep may have left.
    @inbounds for i in ilo:(ihi - 3)
        H[i + 2, i] = zero(R)
        H[i + 3, i] = zero(R)
    end
    @inbounds if ilo <= ihi - 2
        H[ihi, ihi - 2] = zero(R)
    end
    nhtot = ihi - ilo + 1
    nwr = clamp(_iparmq_ns(nhtot), 2, min(nhtot, (n - 1) ÷ 3))
    nhtot > _LAQR_KNWSWP && (nwr = min(3 * nwr ÷ 2, min(nhtot, (n - 1) ÷ 3)))
    nsr = _iparmq_ns(nhtot)
    nsr = min(nsr, (n - 3) ÷ 6, ihi - ilo)
    nsr = max(2, nsr - (nsr % 2))
    nwmax = max(2, min((n - 1) ÷ 3, nwr * 2))
    nsmax = max(2, min((n - 3) ÷ 6, nsr))
    nsmax -= nsmax % 2
    kdumax = 4 * (nsmax ÷ 2)
    info = 0
    # ESCAPE AUDIT (@scope arn): every borrow below stays inside. They reach `_aed!` and `_dlaqr5!`,
    # which read/write them elementwise and hand them to `gemm!`/`trexc!`/`gehrd!`/`ormhr!` as operands;
    # none of those retains an argument. `gehrd!` and `ormhr!` open their own nested scopes, bump-
    # allocated after these and released first. The return value is the scalar `info`.
    @scope arn begin
        # Landing-buffer blocking: DERIVED, criterion = the GEMM landing buffer resident in L2 with
        # headroom, which is the same residency rule `_trmv_blk!` and the QR panel use. Bounded below so
        # a tiny window still blocks sensibly, and above by the matrix.
        wide = max(kdumax, nwmax)
        nvh = clamp(_L2_BYTES ÷ (2 * max(wide, 1) * sizeof(R)), 16, n)
        V3 = borrow!(arn, R, 3, max(nsmax ÷ 2, 1))
        Uac = borrow!(arn, R, max(kdumax, 1), max(kdumax, 1))
        WVb = borrow!(arn, R, nvh, wide)
        WHb = borrow!(arn, R, wide, nvh)
        Tw = borrow!(arn, R, nwmax, nwmax)
        Vw = borrow!(arn, R, nwmax, nwmax)
        wk = borrow!(arn, R, max(2 * nwmax, n))
        kbot = ihi
        ndfl = 1
        ndec = -1
        nw = 0
        itmax = 30 * max(10, nhtot)
        it = 0
        while it < itmax
            it += 1
            kbot < ilo && break
            # ---- locate the active block ----
            ktop = ilo
            @inbounds for k in kbot:-1:(ilo + 1)
                if iszero(H[k, k - 1])
                    ktop = k
                    break
                end
                k == ilo + 1 && (ktop = ilo)
            end
            # ---- select the deflation window size ----
            nh = kbot - ktop + 1
            nwupbd = min(nh, (n - 1) ÷ 3)
            nw = ndfl < _LAQR_KEXNW ? min(nwupbd, nwr) : min(nwupbd, 2 * nw)
            if nw < nwmax
                if nw >= nh - 1
                    nw = nh
                else
                    kwtop = kbot - nw + 1
                    @inbounds if abs(H[kwtop, kwtop - 1]) > abs(H[kwtop - 1, kwtop - 2])
                        nw += 1
                    end
                end
            end
            if ndfl < _LAQR_KEXNW
                ndec = -1
            elseif ndec >= 0 || nw >= nwupbd
                ndec += 1
                nw - ndec < 2 && (ndec = 0)
                nw -= ndec
            end
            nw = min(nw, nwmax)
            # ---- aggressive early deflation ----
            ls, ld = _aed!(
                wantt, wantz, n, ktop, kbot, nw, H, iloz, ihiz, Z, wr, wi,
                Tw, Vw, wk, nvh, WVb, nvh, WHb
            )
            kbot -= ld
            ks = kbot - ls + 1
            # ---- skip the sweep when AED deflated enough that another one is likely wasted ----
            if ld == 0 || (100 * ld <= nw * _LAQR_NIBBLE && kbot - ktop + 1 > min(_LAQR_NMIN, nwmax))
                ns = min(nsmax, nsr, max(2, kbot - ktop))
                ns -= ns % 2
                if ndfl % _LAQR_KEXSH == 0
                    # Exceptional shifts, to break rare stagnation.
                    ks = kbot - ns + 1
                    @inbounds for i in kbot:-2:max(ks + 1, ktop + 2)
                        ss = abs(H[i, i - 1]) + abs(H[i - 1, i - 2])
                        aa = R(0.75) * ss + H[i, i]
                        bb = ss
                        cc = R(-0.4375) * ss
                        dd = aa
                        _, _, _, _, r1r, r1i, r2r, r2i, _, _ = _dlanv2(aa, bb, cc, dd)
                        wr[i - 1] = r1r; wi[i - 1] = r1i
                        wr[i] = r2r; wi[i] = r2i
                    end
                    @inbounds if ks == ktop
                        wr[ks + 1] = H[ks + 1, ks + 1]; wi[ks + 1] = zero(R)
                        wr[ks] = wr[ks + 1]; wi[ks] = wi[ks + 1]
                    end
                else
                    # Too few shifts from AED: get more from a trailing principal submatrix.
                    if kbot - ks + 1 <= ns ÷ 2
                        ks = kbot - ns + 1
                        @inbounds for c in 1:ns, r in 1:ns
                            Tw[r, c] = H[ks + r - 1, ks + c - 1]
                        end
                        inf = _dlahqr!(
                            false, false, view(Tw, 1:ns, 1:ns), 1, ns,
                            view(wr, ks:(ks + ns - 1)), view(wi, ks:(ks + ns - 1)), 1, 1,
                            view(Vw, 1:1, 1:1)
                        )
                        ks += inf
                        if ks >= kbot
                            @inbounds begin
                                aa = H[kbot - 1, kbot - 1]; bb = H[kbot - 1, kbot]
                                cc = H[kbot, kbot - 1]; dd = H[kbot, kbot]
                            end
                            _, _, _, _, r1r, r1i, r2r, r2i, _, _ = _dlanv2(aa, bb, cc, dd)
                            @inbounds begin
                                wr[kbot - 1] = r1r; wi[kbot - 1] = r1i
                                wr[kbot] = r2r; wi[kbot] = r2i
                            end
                            ks = kbot - 1
                        end
                    end
                    if kbot - ks + 1 > ns
                        # Bubble sort by magnitude, descending — keeps conjugate pairs adjacent.
                        sorted = false
                        @inbounds for k in kbot:-1:(ks + 1)
                            sorted && break
                            sorted = true
                            for i in ks:(k - 1)
                                if abs(wr[i]) + abs(wi[i]) < abs(wr[i + 1]) + abs(wi[i + 1])
                                    sorted = false
                                    wr[i], wr[i + 1] = wr[i + 1], wr[i]
                                    wi[i], wi[i + 1] = wi[i + 1], wi[i]
                                end
                            end
                        end
                    end
                    @inbounds for i in kbot:-2:(ks + 2)
                        if wi[i] != -wi[i - 1]
                            wr[i], wr[i - 1], wr[i - 2] = wr[i - 1], wr[i - 2], wr[i]
                            wi[i], wi[i - 1], wi[i - 2] = wi[i - 1], wi[i - 2], wi[i]
                        end
                    end
                end
                # Two real shifts left: use only one (LAPACK's note — the pair is redundant).
                @inbounds if kbot - ks + 1 == 2 && iszero(wi[kbot])
                    if abs(wr[kbot] - H[kbot, kbot]) < abs(wr[kbot - 1] - H[kbot, kbot])
                        wr[kbot - 1] = wr[kbot]
                    else
                        wr[kbot] = wr[kbot - 1]
                    end
                end
                ns = min(ns, kbot - ks + 1)
                ns -= ns % 2
                ks = kbot - ns + 1
                if ns >= 2
                    _dlaqr5!(
                        wantt, wantz, n, ktop, kbot, ns,
                        view(wr, ks:(ks + ns - 1)), view(wi, ks:(ks + ns - 1)),
                        H, iloz, ihiz, Z, V3, Uac, nvh, WVb, nvh, WHb
                    )
                end
            end
            ndfl = ld > 0 ? 1 : ndfl + 1
        end
        it >= itmax && kbot >= ilo && (info = kbot)
    end
    return info
end
