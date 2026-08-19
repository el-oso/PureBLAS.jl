# LAPACK band Cholesky — dpbtrf / dpbtrs, generic over s/d/c/z (Hermitian for complex; conj folds to
# identity on real, so the real/AD path is byte-identical). Ports Reference-LAPACK dpbtf2.f (the
# unblocked right-looking factor) and dpbtrs.f (two band triangular solves).
#
# LAPACK band storage `AB` is a (kd+1)×n matrix (kd = number of super/sub-diagonals):
#   uplo='U': AB[kd+1 + (i-j), j] = A[i,j]  for  max(1,j-kd) ≤ i ≤ j   (diagonal at row kd+1)
#   uplo='L': AB[1 + (i-j),    j] = A[i,j]  for  j ≤ i ≤ min(n,j+kd)   (diagonal at row 1)
# Only the band is referenced/overwritten; factor L (or U) overwrites AB in the same layout.


# ── pbtrf!: band Cholesky factorization ───────────────────────────────────────────────────────────
# uplo='L': A = L·Lᴴ, uplo='U': A = Uᴴ·U. Overwrites AB with the factor. Throws PosDefException at
# the first non-positive pivot (global column index, LAPACK info semantics). `kd` defaults to
# size(AB,1)-1.
#
# Two kernels, dispatched on bandwidth:
#   • kd <  _pbtrf_cross(T): the UNBLOCKED right-looking dpbtf2.f port (verbatim the previous
#     pbtrf! loops — bit-identical results; this is the narrow-band gate win: 1.5×OB/3.8×AOCL @kd=4).
#   • kd ≥ _pbtrf_cross(T), T<:BlasFloat, unit-stride storage: the BLOCKED dpbtrf.f port —
#     potrf!/trsm!/syrk!(herk!)/gemm on in-band blocks + the LAPACK work-array corner. uplo='U'
#     has TWO blocked kernels and picks between them at a second measured crossover
#     (_pbtrf_ucross): a conjugate-transpose band repack onto the lower kernel for narrow bands,
#     and a native upper-storage port for wide ones. Generic/AD eltypes (Dual, BigFloat, …) and
#     non-strided storage ALWAYS
#     take the unblocked kernel: the blocked path is deliberately restricted to T<:BlasFloat (it
#     goes through pointer-based PtrMatrix views and the tuned L3 kernels, neither of which is
#     AD-traceable).
#
# The crossover knob is a SELF-TUNING CONSTANT (PDM ladder):
#   P — Preference "pbtrf_cross_kd" pins it (trim/.so builds MUST pin; bench/calibrate.jl may).
#   M — MEASURE tier default: OncePerProcess per eltype, racing the two kernels over a DERIVED,
#       formula-bounded candidate set kd ∈ {2,4,8,16}·W, W = _vwidth(T). Measure (not Derive)
#       because the crossover pits an L1-resident SIMD rank-1 loop against blocked BLAS-3 with
#       per-block dispatch/packing overhead — a port-balance/overhead crossover, not a residency
#       one. The residency model was FALSIFIED on a box we have (Zen4: kd=64 triangle = 16 KB,
#       L1-resident, yet blocked already wins) — the PDM tell for Measure tier, mirroring
#       _ger_np (`_gemvt_nc` was only ever aspirational — never implemented, and MEASURED to be a
#       net wash on 2026-07-31; see the NC table at level2.jl `_gemv_t_simd!`). Candidate bounds ARE Derived: below 2W a trsm/syrk panel column can't
#       fill two SIMD registers, so BLAS-3 can't pay (measured: kd=16=2W unblocked wins on Zen4);
#       by 16W the unblocked working set (16W)²·sizeof ≥ L1 on every fleet µarch, so the scalar
#       rank-1 streams with zero reuse while blocked is compute-bound — if blocked hasn't won by
#       16W the harness returns typemax (never block; keep the proven kernel).
function pbtrf!(AB::AbstractMatrix; uplo::AbstractChar = 'L', kd::Integer = size(AB, 1) - 1)
    n = size(AB, 2)
    size(AB, 1) >= kd + 1 || throw(DimensionMismatch("pbtrf!: size(AB,1) must be ≥ kd+1"))
    (uplo == 'L' || uplo == 'U') || throw(ArgumentError("pbtrf!: uplo must be 'L' or 'U'"))
    T = eltype(AB)
    if T <: BlasFloat && _strided1(AB) && n > 1 && Int(kd) >= _pbtrf_cross(T)
        uplo == 'L' && return _pbtrf_blocked!(AB, n, Int(kd))
        # Two upper kernels; the winner inverts with kd (see _pbtrf_ucross).
        return Int(kd) >= _pbtrf_ucross(T) ? _pbtrf_blocked_U!(AB, n, Int(kd)) :
            _pbtrf_repack_U!(AB, n, Int(kd))
    end
    return uplo == 'L' ? _pbtf2_L!(AB, n, Int(kd)) : _pbtf2_U!(AB, n, Int(kd))
end

# ── unblocked kernels (dpbtf2.f, right-looking) — VERBATIM the previous pbtrf! loop bodies ────────
# Do not touch: these are the narrow-band gate win and must stay bit-identical.
function _pbtf2_L!(AB::AbstractMatrix, n::Int, kd::Int)
    @inbounds for j in 1:n
        ajj = real(AB[1, j])                          # diagonal is real (Hermitian)
        ajj > 0 || throw(PosDefException(j))
        ajj = sqrt(ajj); AB[1, j] = ajj; invd = inv(ajj)
        kn = min(kd, n - j)
        for i in 1:kn                                 # scale L[j+1:j+kn, j] by 1/√ajj
            AB[1 + i, j] *= invd
        end
        for jc in 1:kn                                # trailing rank-1 update within the band
            lcj = conj(AB[1 + jc, j])                   # conj(L[j+jc, j])
            for ir in jc:kn                           # row j+ir ≥ col j+jc (lower)
                AB[1 + (ir - jc), j + jc] -= AB[1 + ir, j] * lcj
            end
        end
    end
    return AB
end
function _pbtf2_U!(AB::AbstractMatrix, n::Int, kd::Int)
    @inbounds for j in 1:n
        ajj = real(AB[kd + 1, j])
        ajj > 0 || throw(PosDefException(j))
        ajj = sqrt(ajj); AB[kd + 1, j] = ajj; invd = inv(ajj)
        kn = min(kd, n - j)
        for ir in 1:kn                                # scale row U[j, j+1:j+kn] by 1/√ajj
            AB[kd + 1 - ir, j + ir] *= invd
        end
        for ir in 1:kn                                # trailing update A22 -= U12ᴴ·U12
            ujr = AB[kd + 1 - ir, j + ir]                   # U[j, j+ir]
            for ic in 1:ir                            # row j+ic ≤ col j+ir (upper)
                AB[kd + 1 - (ir - ic), j + ir] -= conj(AB[kd + 1 - ic, j + ic]) * ujr
            end
        end
    end
    return AB
end

# ── panel width nb — DERIVE tier ──────────────────────────────────────────────────────────────────
# The blocked band factor faces the identical panel-width tradeoff as dense right-looking potrf
# (small nb ⇒ the O(n·nb·kd) panel trsm stays lower-order vs the O(n·kd²) syrk; large nb ⇒ per-block
# dispatch overhead amortizes), so nb RIDES potrf!'s own tuned base width — `_potrf_base(T)` /
# `_CPOTRF_BASE`, themselves self-tuning constants (anchor-fastest-path: same knob potrf! uses to
# pick ITS panels). `min(·, kd)` is the STRUCTURAL bound of the algorithm (reference dpbtrf.f:
# `IF( NB.LE.1 .OR. NB.GT.KD ) THEN → DPBTF2`): a panel wider than the band would make A21 vanish
# and push ALL trailing work through the slow corner path. Reference-LAPACK's own choice is ILAENV
# nb=32 capped at NBMAX=32 — _potrf_base(F64) defaults to the same 32.
#
# ⚠ The "ride _potrf_base" anchoring above was FALSIFIED by measurement and is retained only as history.
# _potrf_base(Float64) = 32, and 32 turns out to be a sharp LOCAL MINIMUM for the BAND factor — worse than
# both 24 and 40 at every bandwidth (Zen4, uplo='L', n=4096, ratio vs AOCL, two independent runs agreeing):
#     kd     nb=24   nb=32   nb=40   nb=48   nb=64
#     96     1.043   0.907   1.417   1.283   1.194
#     128    1.046   0.932   1.291   1.238   1.130
#     160    1.177   0.947   1.250   1.195   1.103
#     256    1.156   1.119   1.177   1.154   1.095
# nb=32 was the ONLY reason pbtrf missed the gate for mid bands (kd = 96/128/160 sat at 0.90-0.95 in BOTH
# triangles). The dense-potrf panel width is simply not the band factor's optimum, so nb gets its own knob.
#
# req#8 tier: MEASURE, same ladder and the same harness shape as `_pbtrf_cross` below — the optimum is a
# panel/port-balance tradeoff, not a residency one (a residency model predicts nb ≤ 32 for kd=128 on a
# 32 KiB L1, i.e. exactly the value measured to be WORST).
# The candidate bracket is DERIVED around the dense potrf panel width `_potrf_base(T)` / `_CPOTRF_BASE`:
# the band panel faces the same trsm-vs-syrk tradeoff, so that IS the physically analogous knob — the part
# this falsifies is only the assumption that the band optimum EQUALS it. So keep the anchor, measure around
# it: {1/2, 3/4, 1, 5/4, 3/2, 2}·nb0, which for nb0 = 32 spans 16…64 and contains the Zen4 optimum (40).
# NOT scaled by _vwidth: an earlier version bracketed on multiples of W and that was wrong — a W=4 box
# (Zen3/AVX2) could then only reach 32, and measured 0.74-1.01 across kd = 96…256 because the optimum was
# outside its own candidate set. The panel width is a blocking/overhead parameter, not a vector-width one.
const _PBTRF_NB_PREF = @load_preference("pbtrf_nb", nothing)
@static if isnothing(_PBTRF_NB_PREF)
    @inline _pbtrf_nb_anchor(::Type{T}) where {T} = T <: Complex ? _CPOTRF_BASE : _potrf_base(T)
    # Derived from vector width — see `_at_pbtrf_nb` (cpuinfo.jl) for the three-box table.
    @inline _pbtrf_nb_tuned(::Type{Float32}) = _at_pbtrf_nb(_HW, Float32)
    # req8-ok: measured-identical literal — 40 in 6/6 processes on BOTH boxes (wintermute Zen4 and
    # galen Zen3, 2026-08-19). Not a derivation: no formula over the detected consts was found that
    # also reproduces the F32/C32/C64 siblings, which differ per box. Reproduces the incumbent exactly.
    @inline _pbtrf_nb_tuned(::Type{Float64}) = 40
    @inline _pbtrf_nb_tuned(::Type{ComplexF32}) = _at_pbtrf_nb(_HW, ComplexF32)
    @inline _pbtrf_nb_tuned(::Type{ComplexF64}) = _at_pbtrf_nb(_HW, ComplexF64)
    @inline _pbtrf_nb_tuned(::Type{T}) where {T} = _pbtrf_nb_anchor(T)   # generic/AD eltypes never block
else
    @inline _pbtrf_nb_tuned(::Type{<:Any}) = _PBTRF_NB_PREF::Int      # pinned (trim builds land here)
end

# ── narrow-band panel width — MEASURE tier, second regime of the same knob ────────────────────────
# `min(_pbtrf_nb_tuned(T), kd)` is wrong when the clamp BINDS. _pbtrf_nb_tuned is measured at a mid
# band (kd = 4·nb0) where it is right, but for kd < nb_tuned the clamp collapses nb onto kd itself —
# which both kills the in-band panel (i2 = kd - ib = 0, so ALL trailing work goes through the corner
# path) and, worse, lands nb on whatever value kd happens to be. On Zen4 that is 32, and this file
# already documents 32 as "a sharp LOCAL MINIMUM for the BAND factor". Measured, uplo='L', ratio vs
# AOCL (n = 1024 / 4096):
#     kd=32:  nb=8 → 1.63/1.64    nb=16 → 1.36/1.33    nb=24 → 1.05/1.04    nb=kd=32 → 0.99/0.96
# i.e. the clamp was turning a 1.6× win into a gate MISS, and it is the ONLY cell in kd = 32…384 that
# missed. Above the clamp the tuned width is right and stays (kd=64 → 2.34, 96 → 1.41, 128 → 1.29,
# 192 → 1.20, all with nb=40, each beating every narrower candidate).
# Tier is MEASURE for the same reason as its wide-band sibling — a panel/port-balance tradeoff, not a
# residency one — and it gets its own probe because the two regimes have different optima (40 vs 8 on
# Zen4: a 5× spread, so one measurement cannot serve both). Candidates ARE Derived, as multiples of
# the vector width {1,2,3,4}·W, probed at kd = 4W: unlike the wide-band bracket (which correctly is
# NOT W-scaled, see above), the narrow-band optimum sits at the bottom of the range where a panel is
# a small multiple of one SIMD register, so W is the right unit here.
const _PBTRF_NBS_PREF = @load_preference("pbtrf_nb_small", nothing)
@static if isnothing(_PBTRF_NBS_PREF)
    # F32 is exactly `_lanes(hw, Float32)` on all three boxes — a formula, not a table.
    @inline _pbtrf_nb_small(::Type{Float32}) = _at_pbtrf_nbs(_HW, Float32)
    @inline _pbtrf_nb_small(::Type{Float64}) = _at_pbtrf_nbs(_HW, Float64)
    # req8-ok: measured-identical literals — 8 in 6/6 processes on BOTH boxes (2026-08-19). The REAL
    # siblings are deliberately left on the duel: they are stable per box but INVERTED between them
    # (F32 wm=16 gl=8, F64 wm=8 gl=16), which no formula over width or cache reproduces.
    @inline _pbtrf_nb_small(::Type{ComplexF32}) = 8
    @inline _pbtrf_nb_small(::Type{ComplexF64}) = 8
    @inline _pbtrf_nb_small(::Type{T}) where {T} = _vwidth(T)   # generic/AD eltypes never block
else
    @inline _pbtrf_nb_small(::Type{<:Any}) = _PBTRF_NBS_PREF::Int    # pinned (trim builds land here)
end
@inline function _pbtrf_nb(::Type{T}, kd::Int) where {T}
    nb = _pbtrf_nb_tuned(T)
    return kd >= nb ? nb : min(_pbtrf_nb_small(T), kd)   # clamp binds ⇒ use the narrow-band width
end

# ── crossover kd — MEASURE tier (see the tier discussion at pbtrf! above) ─────────────────────────
const _PBTRF_CROSS_PREF = @load_preference("pbtrf_cross_kd", nothing)
@static if isnothing(_PBTRF_CROSS_PREF)
    # Base-only + TOTAL (OncePerProcess poisons the process if the initializer throws) → catch →
    # the candidate-set midpoint 4W (the bracket center on the box the gap was measured on:
    # Zen4 F64 W=8 → 32, inside the measured 16 < kd* < 64).
    # Per-eltype OncePerProcess (complex kernels have 4× the flop density — a shared F64 crossover
    # would misplace them); lazy, so only eltypes actually used pay the one-shot ~10–100 ms tune.
    @inline _pbtrf_cross(::Type{Float32}) = _at_pbtrf_cross(_HW, Float32)
    @inline _pbtrf_cross(::Type{Float64}) = _at_pbtrf_cross(_HW, Float64)
    @inline _pbtrf_cross(::Type{ComplexF32}) = _at_pbtrf_cross(_HW, ComplexF32)
    # req8-ok: measured-identical literal — 16 in 6/6 processes on BOTH boxes (2026-08-19). F64 and
    # C32 stay on the duel: both FLIP across processes (wm F64 32/32/32/32/24/32, gl F64 36/36/36/40/
    # 36/40), so there is no incumbent to reproduce and picking one needs gate evidence.
    @inline _pbtrf_cross(::Type{ComplexF64}) = 16
else
    @inline _pbtrf_cross(::Type{<:BlasFloat}) = _PBTRF_CROSS_PREF::Int   # pinned (trim builds land here)
end

# ── uplo='U' kernel crossover: re-pack-onto-L below, native-upper at and above ────────────────────
# Which of the two upper kernels wins inverts with kd (table in the _pbtrf_repack_U! header:
# repack 2.41→1.10 over kd=48…192 then falls off a cliff to 0.845 at kd=256; native climbs
# 0.63→0.92 then takes over at 1.055/1.078). This knob is the switch point.
#
# req#8 tier: MEASURE. Not Derive — the crossover balances the re-pack's diagonal-walk bandwidth
# against the RATIO of PB's upper/transposed L3 rate to its lower/normal one. That ratio is a
# property of our own kernels' port balance, not of any detected const, and it MOVES whenever the
# upper trsm/syrk paths improve — exactly the "our model mispredicts a box we have" tell that
# separates M from D (same reasoning as _pbtrf_cross and _ger_np).
# Candidate bounds ARE Derived, as one octave above _pbtrf_cross's {2,4,8,16}·W ladder, refined to
# 4W granularity because the cliff is sharp (repack is 1.10 at 24W and 0.845 at 32W on Zen4, so a
# pure doubling ladder would strand kd ∈ (24W, 32W) on the losing kernel):
#   lower 8W — below it the walk spans < 8W columns and the copy is measured-trivial (repack wins
#              by 2.3–3.3× at 6W/8W on Zen4); there is nothing for the native kernel to win back.
#   upper 64W — a full band block already exceeds every fleet L2 by then; if native has not won by
#              64W it never will, and the harness returns typemax (never go native).
const _PBTRF_UCROSS_PREF = @load_preference("pbtrf_u_native_kd", nothing)
@static if isnothing(_PBTRF_UCROSS_PREF)
    # Exactly `hw.l2 ÷ 4096` on all three boxes. F64/C64 keep the duel: F64 flips on galen
    # (192x4, 256, 192) and C64 flips on both wintermute (176/192/208) and neuromancer.
    @inline _pbtrf_ucross(::Type{Float32}) = _at_pbtrf_ucross(_HW)
    @inline _pbtrf_ucross(::Type{Float64}) = _at_pbtrf_ucross(_HW, Float64)
    @inline _pbtrf_ucross(::Type{ComplexF32}) = _at_pbtrf_ucross(_HW)
    # C64 flipped 176/192/208 on Zen4 and 192..256 on Zen5 despite an IDENTICAL 1 MiB L2 — which is the
    # signature of a CROSSOVER, not of a broken tuner: at the switch point the two kernels are within
    # ~1% of each other BY DEFINITION, so the argmin over rungs wanders while the loss is ~nothing.
    # Being one rung off therefore costs ~nothing, and the sibling formula is the principled place to
    # sit. Galen measures 128 = exactly l2 ÷ 4096, agreeing with it outright.
    @inline _pbtrf_ucross(::Type{ComplexF64}) = _at_pbtrf_ucross(_HW)
else
    @inline _pbtrf_ucross(::Type{<:BlasFloat}) = _PBTRF_UCROSS_PREF::Int   # pinned (trim builds land here)
end

# ── band-block view: the LAPACK ld-1 trick ────────────────────────────────────────────────────────
# In band storage AB[1+(i-j), j] (uplo='L'), a block of A whose row-col offset r-c is constant
# along each block diagonal becomes a PLAIN column-major matrix when re-read with leading dimension
# LDAB-1: block element (p,q) (0-based) of the block anchored at storage AB[r0, c0] lives at flat
# offset (c0-1+q)·LDAB + (r0-1+p-q) = [(c0-1)·LDAB + (r0-1)] + q·(LDAB-1) + p — i.e. base pointer
# AB[r0,c0], ld = LDAB-1. This is exactly the `AB( r, c ), LDAB-1` idiom every BLAS call in
# reference dpbtrf.f uses. PtrMatrix (isbits, stride(·,1)==1 ⇒ _strided1 ⇒ hits the tuned trsm/syrk
# fast paths, same bridge the Mode-1 C-ABI uses) carries (ptr, m, n, ld).
# In-bounds: every entry the BLAS contract touches has storage row r0+p-q ∈ [1, kd+1] (the
# triangles/trapezoids never leave the band), so all addresses stay inside the AB buffer. The
# OPPOSITE triangles of these views alias OTHER band entries — which is why the diagonal block is
# factored in a dense scratch (see (a) in _pbtrf_blocked!) and why only contract-clean kernels may
# touch the views.
@inline _pb_blk(p0::Ptr{T}, ld2::Int, ldabs::Int, r::Int, c::Int, m::Int, n::Int) where {T} =
    PtrMatrix(p0 + ((c - 1) * ldabs + (r - 1)) * sizeof(T), m, n, ld2)

# Scalar dense lower potf2 (dpotf2.f pivot semantics, right-looking): factors S[1:m,1:m] in place,
# returns 0 or the first failing column. Only runs on the pbtrf failure path (see (b) below).
function _potf2L_col!(S::Matrix{T}, m::Int) where {T}
    @inbounds for j in 1:m
        ajj = real(S[j, j])
        ajj > 0 || return j
        rj = sqrt(ajj); S[j, j] = rj; iv = inv(rj)
        for p in (j + 1):m
            S[p, j] *= iv
        end
        for q in (j + 1):m
            lq = conj(S[q, j])
            for p in q:m
                S[p, q] -= S[p, j] * lq
            end
        end
    end
    return 0
end

# Scalar dense UPPER potf2 (mirror of _potf2L_col!): factors S[1:m,1:m] in place from its upper
# triangle, returns 0 or the first failing column. Failure path of _pbtrf_blocked_U! only.
function _potf2U_col!(S::Matrix{T}, m::Int) where {T}
    @inbounds for j in 1:m
        ajj = real(S[j, j])
        ajj > 0 || return j
        rj = sqrt(ajj); S[j, j] = rj; iv = inv(rj)
        for q in (j + 1):m
            S[j, q] *= iv
        end
        for q in (j + 1):m
            ujq = conj(S[j, q])
            for p in (j + 1):q
                S[p, q] -= ujq * S[j, p]
            end
        end
    end
    return 0
end

# ── blocked kernel, uplo='L' (dpbtrf.f DO 140 / zpbtrf.f) ─────────────────────────────────────────
# Partition at outer column i (Fortran names; ib = min(nb, n-i+1)):
#   A11 (ib×ib)  diagonal block — factored in a DENSE nb×nb scratch, not in the band view, because:
#                (a) potrf!'s real-lower fast path WRITES into the strictly-upper triangle of its
#                    argument (measured: 56 spill writes at m=32) — harmless dense, but through the
#                    ld-1 band view "upper triangle" is other columns' band entries ⇒ corruption
#                    (this was a real observed 4e-4 factor error before the scratch);
#                (b) potrf!'s PosDefException.info is NOT the failing column (measured: always 1),
#                    so on failure the pristine band data is re-factored with the scalar
#                    dpotf2-ordered _potf2L_col! to recover the exact LAPACK info column.
#                The scratch also gives trsm a contiguous ld=nb triangle (locality bonus).
#   A21 (i2×ib), i2 = min(kd-ib, n-i-ib+1) — the sub-diagonal panel fully inside the band, ld-1 view.
#   A31 (i3×ib), i3 = min(ib, n-i-kd+1)    — the CORNER: block rows i+kd… fall outside the band on
#                part of the block. Only its in-band UPPER triangle (ii ≤ jj, since
#                (i+kd+ii-1)-(i+jj-1) ≤ kd ⇔ ii ≤ jj) exists in AB; the rest is structurally ZERO.
#                dpbtrf materializes it in a dense (nb+1)×nb WORK array — in-band triangle copied
#                in, rest zero — runs dense trsm/gemm/syrk on it, and copies the triangle back.
#                The zeros survive the trsm EXACTLY (right-multiplying by a triangular inverse
#                preserves the trapezoid; zero entries are recomputed as sums of 0·x), so WORK is
#                zeroed ONCE per call, as in the reference (dpbtrf.f zeroes it before the I-loop).
#   A22 (i2×i2), A32 (i3×i2), A33 (i3×i3) — trailing blocks receiving syrk/gemm/syrk, ld-1 views.
# LDWORK = nb+1, mirroring reference LDWORK = NBMAX+1 (odd leading dim ⇒ no po2 column-stride cache
# aliasing — same lore as _alias_ld; not a tuning knob).
function _pbtrf_blocked!(AB::AbstractMatrix{T}, n::Int, kd::Int, nb::Int = _pbtrf_nb(T, kd)) where {T}
    ldabs = stride(AB, 2)
    ld2 = ldabs - 1                                    # the LDAB-1 of every BLAS call in dpbtrf.f
    tc = T <: Complex ? 'C' : 'T'                      # dpbtrf 'Transpose' / zpbtrf 'Conjugate transpose'
    rone = one(real(T))
    W, S = _pbtrf_work(T, nb)                          # GKH-owned; W arrives zeroed ONCE (see above),
    #                                                    S is the dense diagonal-block scratch ((a)/(b))
    p0 = pointer(AB); pw = pointer(W); ps = pointer(S)
    GC.@preserve AB W S begin
        for i in 1:nb:n
            ib = min(nb, n - i + 1)
            # dpbtrf: CALL DPOTF2('L', IB, AB(1,I), LDAB-1, II) — via the dense scratch: copy the
            # stored lower triangle of A11 in (zeroing the upper half so the factor kernel sees
            # fully-defined data), run the tuned blocked potrf! there, copy the factor back.
            @inbounds for q in 1:ib
                for p in 1:(q - 1); S[p, q] = zero(T); end
                for p in q:ib; S[p, q] = AB[1 + p - q, i + q - 1]; end
            end
            Sv = PtrMatrix(ps, ib, ib, nb)
            ok = true
            try
                potrf!(Sv; uplo = 'L')
            catch e
                e isa PosDefException || rethrow()
                ok = false
            end
            if !ok
                # dpbtrf: IF(II≠0) INFO = I+II-1. Band data is still pristine — recover the exact
                # failing column with the scalar dpotf2-ordered factor (reference dpbtrf runs
                # potf2 on the diagonal block, so this IS the reference pivot sequence). If the
                # scalar factor succeeds (blocked-vs-scalar roundoff tie on a borderline pivot),
                # keep its factor and continue — matching what the reference would have done.
                @inbounds for q in 1:ib, p in q:ib
                    S[p, q] = AB[1 + p - q, i + q - 1]
                end
                col = _potf2L_col!(S, ib)
                col != 0 && throw(PosDefException(i - 1 + col))
            end
            @inbounds for q in 1:ib, p in q:ib          # factor back into the band
                AB[1 + p - q, i + q - 1] = S[p, q]
            end
            if i + ib <= n
                i2 = min(kd - ib, n - i - ib + 1)      # dpbtrf: I2 — in-band panel rows
                i3 = min(ib, n - i - kd + 1)           # dpbtrf: I3 — corner rows (≤0 ⇒ no corner)
                if i2 > 0
                    # dpbtrf: DTRSM('Right','Lower','T','N', I2, IB, 1, AB(1,I), LDAB-1,
                    #               AB(1+IB,I), LDAB-1)          — A21 := A21·L11⁻ᴴ
                    # (L11 read from the dense scratch Sv — same values, contiguous ld.)
                    A21 = _pb_blk(p0, ld2, ldabs, 1 + ib, i, i2, ib)
                    trsm!(A21, Sv; side = 'R', uplo = 'L', transA = tc, diag = 'N')
                    # dpbtrf: DSYRK('Lower','N', I2, IB, -1, A21, LDAB-1, 1, AB(1,I+IB), LDAB-1)
                    #                                              — A22 -= A21·A21ᴴ
                    A22 = _pb_blk(p0, ld2, ldabs, 1, i + ib, i2, i2)
                    if T <: Complex
                        herk!(A22, A21; uplo = 'L', trans = 'N', alpha = -rone, beta = rone)
                    else
                        syrk!(A22, A21; uplo = 'L', trans = 'N', alpha = -one(T), beta = one(T))
                    end
                end
                if i3 > 0
                    # dpbtrf DO 110/100: copy the (in-band) UPPER triangle of A31 into WORK:
                    #   WORK(II,JJ) = AB(KD+1-JJ+II, JJ+I-1),  JJ=1:IB, II=1:MIN(JJ,I3)
                    # (A31[ii,jj] = A[i+kd+ii-1, i+jj-1] sits at storage row 1+(kd+ii-jj) —
                    #  in-band iff ii ≤ jj; the strictly-lower part of WORK stays 0.)
                    @inbounds for jj in 1:ib, ii in 1:min(jj, i3)
                        W[ii, jj] = AB[kd + 1 - jj + ii, jj + i - 1]
                    end
                    Wv = PtrMatrix(pw, i3, ib, nb + 1)
                    # dpbtrf: DTRSM('Right','Lower','T','N', I3, IB, 1, A11, LDAB-1, WORK, LDWORK)
                    #                                              — A31 := A31·L11⁻ᴴ (dense, in WORK)
                    trsm!(Wv, Sv; side = 'R', uplo = 'L', transA = tc, diag = 'N')
                    if i2 > 0
                        # dpbtrf: DGEMM('N','T', I3, I2, IB, -1, WORK, LDWORK, AB(1+IB,I),
                        #               LDAB-1, 1, AB(1+KD-IB, I+IB), LDAB-1)
                        #                                          — A32 -= A31·A21ᴴ
                        # (_gemm_core! directly: the public gemm! kwarg entry's StridedMatrix
                        #  gate would bounce PtrMatrix to the generic kernel; this is the same
                        #  internal entry the Mode-1 C-ABI uses.)
                        A21 = _pb_blk(p0, ld2, ldabs, 1 + ib, i, i2, ib)
                        A32 = _pb_blk(p0, ld2, ldabs, 1 + kd - ib, i + ib, i3, i2)
                        _gemm_core!(A32, Wv, A21, -one(T), one(T), false, true, false, T <: Complex)
                    end
                    # dpbtrf: DSYRK('Lower','N', I3, IB, -1, WORK, LDWORK, 1, AB(1,I+KD), LDAB-1)
                    #                                          — A33 -= A31·A31ᴴ
                    A33 = _pb_blk(p0, ld2, ldabs, 1, i + kd, i3, i3)
                    if T <: Complex
                        herk!(A33, Wv; uplo = 'L', trans = 'N', alpha = -rone, beta = rone)
                    else
                        syrk!(A33, Wv; uplo = 'L', trans = 'N', alpha = -one(T), beta = one(T))
                    end
                    # dpbtrf DO 130/120: copy the triangle back: AB(KD+1-JJ+II, JJ+I-1) = WORK(II,JJ)
                    @inbounds for jj in 1:ib, ii in 1:min(jj, i3)
                        AB[kd + 1 - jj + ii, jj + i - 1] = W[ii, jj]
                    end
                end
            end
        end
    end
    return AB
end

# ── uplo='U': TWO kernels, split at a measured bandwidth crossover ────────────────────────────────
# Neither upper kernel dominates — the choice inverts with kd, so both ship (see _pbtrf_ucross).
#
#   _pbtrf_repack_U!  — conj-transpose re-pack onto the LOWER kernel. A = Uᴴ·U with U upper-banded ⟺
#     A = L·Lᴴ with L = Uᴴ lower-banded: the same matrix in lower band storage is
#     ABL[1+d, j] = A[j+d, j] = conj(ABU[kd+1-d, j+d]), and the factor maps back as
#     ABU[kd+1-d, j] = conj(ABL[1+d, j-d]). Costs two O(kd·n) band passes but runs the whole
#     factorization on the proven-fastest path (the L kernel gates at 1.16–2.33× AOCL).
#   _pbtrf_blocked_U! — NATIVE port of the reference's own UPLO='U' branch (dpbtrf.f DO 50 /
#     zpbtrf.f), operating directly on upper band storage. No copy, but every L3 call is an
#     upper/transposed one (trsm 'L','U','T'; syrk 'U','T'), which PB does not do as well as the
#     lower/normal ones the repack path reaches.
#
# Measured, Zen4 F64, n ∈ {1024, 4096}, same-process ABBA, ratio vs AOCL (`nat/AOCL` vs `rep/AOCL`):
#     kd      48     64     96    128    160    192  |   256    384
#     repack 2.41   3.26   1.20   1.02   1.11   1.10  |  0.845  1.012
#     native 1.36   1.71   0.63   0.83   0.87   0.92  |  1.055  1.078
# The repack's cost is superlinear in kd — it walks A down a DIAGONAL, so it touches kd distinct
# columns per pass, one cache line (and, past a point, one page) per 8-byte element. At kd=96 that
# copy is ~0.7 ms of a 2.6 ms factorization; by kd=256 it is ~6.2 ms of 14.5 ms and drags the whole
# op to 0.845 — a gate MISS that tuning could not reach (the best tiling recovered 1149 µs of
# 2749 µs → 0.917, and helped kd=128/256 while HURTING kd=160/192/384). Below the crossover the
# copy is cheap and the L kernel's advantage is decisive; above it the copy dominates and the
# native kernel's weaker L3 calls are the lesser evil. Both are needed to gate everywhere.
#
# Failing-column reporting is uplo-independent for the repack path (leading minors of A do not
# depend on which triangle is stored, and _pbtrf_blocked! already reports dpotf2-ordered global
# columns), so it needs no translation.
function _pbtrf_repack_U!(AB::AbstractMatrix{T}, n::Int, kd::Int) where {T}
    ABL = _pbtrf_band(T, kd, n)                        # owned scratch — see _pbtrf_band for why not `undef`
    @inbounds for j in 1:n                             # pack: ABU → conj-transposed lower band
        for d in 0:min(kd, n - j)
            ABL[1 + d, j] = conj(AB[kd + 1 - d, j + d])
        end
    end
    _pbtrf_blocked!(ABL, n, kd)                        # may throw PosDefException(global col) — correct as-is
    @inbounds for j in 1:n                             # unpack: L factor → ABU (U = Lᴴ)
        for d in 0:min(kd, j - 1)
            AB[kd + 1 - d, j] = conj(ABL[1 + d, j - d])
        end
    end
    return AB
end

# Same `_pb_blk` ld-1 trick as the lower kernel: in uplo='U' storage AB[kd+1+(i-j), j] = A[i,j], so a
# block anchored at storage (r0, c0) re-read with ld = LDAB-1 is a plain column-major matrix.
function _pbtrf_blocked_U!(AB::AbstractMatrix{T}, n::Int, kd::Int, nb::Int = _pbtrf_nb(T, kd)) where {T}
    ldabs = stride(AB, 2)
    ld2 = ldabs - 1                                    # the LDAB-1 of every BLAS call in dpbtrf.f
    tc = T <: Complex ? 'C' : 'T'
    rone = one(real(T))
    W, S = _pbtrf_work(T, nb)                          # GKH-owned; W arrives zeroed ONCE per call
    p0 = pointer(AB); pw = pointer(W); ps = pointer(S)
    GC.@preserve AB W S begin
        for i in 1:nb:n
            ib = min(nb, n - i + 1)
            # dpbtrf: CALL DPOTF2('U', IB, AB(KD+1,I), LDAB-1, II) — via the dense scratch (zero the
            # unused lower half so the factor kernel sees fully-defined data).
            @inbounds for q in 1:ib
                for p in 1:q; S[p, q] = AB[kd + 1 + p - q, i + q - 1]; end
                for p in (q + 1):ib; S[p, q] = zero(T); end
            end
            Sv = PtrMatrix(ps, ib, ib, nb)
            ok = true
            try
                potrf!(Sv; uplo = 'U')
            catch e
                e isa PosDefException || rethrow()
                ok = false
            end
            if !ok
                # Band data is still pristine — recover the exact failing column with the scalar
                # potf2-ordered factor (the reference's pivot sequence). If it succeeds (a blocked-vs-
                # scalar roundoff tie on a borderline pivot) keep ITS factor and continue.
                @inbounds for q in 1:ib
                    for p in 1:q; S[p, q] = AB[kd + 1 + p - q, i + q - 1]; end
                    for p in (q + 1):ib; S[p, q] = zero(T); end
                end
                col = _potf2U_col!(S, ib)
                col != 0 && throw(PosDefException(i - 1 + col))
            end
            @inbounds for q in 1:ib, p in 1:q           # factor back into the band
                AB[kd + 1 + p - q, i + q - 1] = S[p, q]
            end
            if i + ib <= n
                i2 = min(kd - ib, n - i - ib + 1)      # dpbtrf: I2 — in-band panel cols
                i3 = min(ib, n - i - kd + 1)           # dpbtrf: I3 — corner cols (≤0 ⇒ no corner)
                if i2 > 0
                    # dpbtrf: DTRSM('Left','Upper','T','N', IB, I2, 1, AB(KD+1,I), LDAB-1,
                    #               AB(KD+1-IB,I+IB), LDAB-1)      — A12 := U11⁻ᴴ·A12
                    A12 = _pb_blk(p0, ld2, ldabs, kd + 1 - ib, i + ib, ib, i2)
                    trsm!(A12, Sv; side = 'L', uplo = 'U', transA = tc, diag = 'N')
                    # dpbtrf: DSYRK('Upper','T', I2, IB, -1, A12, LDAB-1, 1, AB(KD+1,I+IB), LDAB-1)
                    #                                              — A22 -= A12ᴴ·A12
                    A22 = _pb_blk(p0, ld2, ldabs, kd + 1, i + ib, i2, i2)
                    if T <: Complex
                        herk!(A22, A12; uplo = 'U', trans = 'C', alpha = -rone, beta = rone)
                    else
                        syrk!(A22, A12; uplo = 'U', trans = 'T', alpha = -one(T), beta = one(T))
                    end
                end
                if i3 > 0
                    # dpbtrf DO 20/10: WORK(II,JJ) = AB(II-JJ+1, JJ+I+KD-1), JJ=1:I3, II=JJ:IB — the
                    # LOWER triangle of A13. W is IB×I3 here, the mirror of the lower branch's I3×IB,
                    # and its strictly-upper part stays 0 for the life of the call (W is zeroed once,
                    # and copy-in/copy-back both touch only ii ≥ jj). The bound is NOT the lower
                    # branch's `min(jj, i3)` mirrored: `ii in (i3-jj+1):ib` sends the band row index
                    # ii-jj+1 to 2-i3 ≤ 0 at jj=i3, ii=1.
                    @inbounds for jj in 1:i3, ii in jj:ib
                        W[ii, jj] = AB[ii - jj + 1, jj + i + kd - 1]
                    end
                    Wv = PtrMatrix(pw, ib, i3, nb + 1)
                    # dpbtrf: DTRSM('Left','Upper','T','N', IB, I3, 1, A11, LDAB-1, WORK, LDWORK)
                    trsm!(Wv, Sv; side = 'L', uplo = 'U', transA = tc, diag = 'N')
                    if i2 > 0
                        # dpbtrf: DGEMM('T','N', I2, I3, IB, -1, AB(KD+1-IB,I+IB), LDAB-1, WORK,
                        #               LDWORK, 1, AB(1+IB,I+KD), LDAB-1)   — A23 -= A12ᴴ·A13
                        A12 = _pb_blk(p0, ld2, ldabs, kd + 1 - ib, i + ib, ib, i2)
                        A23 = _pb_blk(p0, ld2, ldabs, 1 + ib, i + kd, i2, i3)
                        _gemm_core!(A23, A12, Wv, -one(T), one(T), true, false, T <: Complex, false)
                    end
                    # dpbtrf: DSYRK('Upper','T', I3, IB, -1, WORK, LDWORK, 1, AB(KD+1,I+KD), LDAB-1)
                    A33 = _pb_blk(p0, ld2, ldabs, kd + 1, i + kd, i3, i3)
                    if T <: Complex
                        herk!(A33, Wv; uplo = 'U', trans = 'C', alpha = -rone, beta = rone)
                    else
                        syrk!(A33, Wv; uplo = 'U', trans = 'T', alpha = -one(T), beta = one(T))
                    end
                    # dpbtrf DO 40/30: copy the lower triangle of A13 back into place
                    @inbounds for jj in 1:i3, ii in jj:ib
                        AB[ii - jj + 1, jj + i + kd - 1] = W[ii, jj]
                    end
                end
            end
        end
    end
    return AB
end


# ── pbtrs!: solve A·X = B with the pbtrf! factor (dpbtrs.f) ────────────────────────────────────────
# Two band triangular solves per RHS. uplo='L' (A=L·Lᴴ): L then Lᴴ. uplo='U' (A=Uᴴ·U): Uᴴ then U.
function pbtrs!(
        AB::AbstractMatrix, B::AbstractVecOrMat; uplo::AbstractChar = 'L',
        kd::Integer = size(AB, 1) - 1
    )
    n = size(AB, 2)
    Bm = _gt_asmat(B); size(Bm, 1) == n || throw(DimensionMismatch("pbtrs!: size(B,1) must equal n"))
    nrhs = size(Bm, 2)
    if uplo == 'L'
        @inbounds for j in 1:nrhs
            for k in 1:n                                  # L·y = b (forward)
                yk = Bm[k, j] / AB[1, k]; Bm[k, j] = yk
                for i in (k + 1):min(k + kd, n)
                    Bm[i, j] -= AB[1 + (i - k), k] * yk   # L[i,k]
                end
            end
            for k in n:-1:1                               # Lᴴ·x = y (backward)
                s = Bm[k, j]
                for i in (k + 1):min(k + kd, n)
                    s -= conj(AB[1 + (i - k), k]) * Bm[i, j]   # conj(L[i,k])
                end
                Bm[k, j] = s / AB[1, k]                   # L[k,k] real
            end
        end
    elseif uplo == 'U'
        @inbounds for j in 1:nrhs
            for k in 1:n                                  # Uᴴ·y = b (forward)
                yk = Bm[k, j] / AB[kd + 1, k]; Bm[k, j] = yk
                for i in (k + 1):min(k + kd, n)
                    Bm[i, j] -= conj(AB[kd + 1 - (i - k), i]) * yk   # conj(U[k,i])
                end
            end
            for k in n:-1:1                               # U·x = y (backward)
                s = Bm[k, j]
                for i in (k + 1):min(k + kd, n)
                    s -= AB[kd + 1 - (i - k), i] * Bm[i, j]  # U[k,i]
                end
                Bm[k, j] = s / AB[kd + 1, k]                # U[k,k] real
            end
        end
    else
        throw(ArgumentError("pbtrs!: uplo must be 'L' or 'U'"))
    end
    return B
end
