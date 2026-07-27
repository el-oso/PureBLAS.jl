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
#     rides the SAME lower kernel through a conjugate-transpose band repack (see
#     _pbtrf_blocked_U!). Generic/AD eltypes (Dual, BigFloat, …) and non-strided storage ALWAYS
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
#       _ger_np/_gemvt_nc. Candidate bounds ARE Derived: below 2W a trsm/syrk panel column can't
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
        return uplo == 'U' ? _pbtrf_blocked_U!(AB, n, Int(kd)) : _pbtrf_blocked!(AB, n, Int(kd))
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
@inline _pbtrf_nb(::Type{T}, kd::Int) where {T} =
    min(T <: Complex ? _CPOTRF_BASE : _potrf_base(T), kd)

# ── crossover kd — MEASURE tier (see the tier discussion at pbtrf! above) ─────────────────────────
const _PBTRF_CROSS_PREF = @load_preference("pbtrf_cross_kd", nothing)
@static if isnothing(_PBTRF_CROSS_PREF)
    # Base-only + TOTAL (OncePerProcess poisons the process if the initializer throws) → catch →
    # the candidate-set midpoint 4W (the bracket center on the box the gap was measured on:
    # Zen4 F64 W=8 → 32, inside the measured 16 < kd* < 64).
    function _measure_pbtrf_cross(::Type{T})::Int where {T}
        vw = _vwidth(T)
        Base.generating_output() && return 4 * vw     # don't burn a measure during precompilation
        try
            for c in (2 * vw, 4 * vw, 8 * vw, 16 * vw)   # Derived candidate bounds (see pbtrf! doc)
                nloc = 16 * c       # harness shape, not a tuning knob: enough columns that the
                #                     steady-state trailing updates dominate the ramp-up/down bands
                #                     (16·c ⇒ ≥16 panel rounds at nb ≤ c); run stays ≪ 100 ms.
                ABm = Matrix{T}(undef, c + 1, nloc)
                refill! = () -> begin                  # diagonally-dominant SPD band, uplo='L'
                    fill!(ABm, one(T))
                    @inbounds for j in 1:nloc
                        ABm[1, j] = T(2 * c + 2)       # diag > Σ|offdiag| ≤ 2c ⇒ SPD
                    end
                end
                refill!(); _pbtf2_L!(ABm, nloc, c)                 # untimed warmups (absorb JIT)
                refill!(); _pbtrf_blocked!(ABm, nloc, c)
                tu = typemax(UInt64); tb = typemax(UInt64)
                for _ in 1:3                                       # interleaved (crude ABBA), min-of-3
                    refill!(); s = time_ns(); _pbtf2_L!(ABm, nloc, c);         tu = min(tu, time_ns() - s)
                    refill!(); s = time_ns(); _pbtrf_blocked!(ABm, nloc, c);   tb = min(tb, time_ns() - s)
                end
                tb < tu && return c    # smallest candidate where blocked wins; all kd ≥ c block
            end
            return typemax(Int)        # blocked never won inside the derived bracket ⇒ never block
        catch
            return 4 * vw
        end
    end
    # Per-eltype OncePerProcess (complex kernels have 4× the flop density — a shared F64 crossover
    # would misplace them); lazy, so only eltypes actually used pay the one-shot ~10–100 ms tune.
    const _PBTRF_CROSS_F32 = Base.OncePerProcess{Int}(() -> _measure_pbtrf_cross(Float32))
    const _PBTRF_CROSS_F64 = Base.OncePerProcess{Int}(() -> _measure_pbtrf_cross(Float64))
    const _PBTRF_CROSS_C32 = Base.OncePerProcess{Int}(() -> _measure_pbtrf_cross(ComplexF32))
    const _PBTRF_CROSS_C64 = Base.OncePerProcess{Int}(() -> _measure_pbtrf_cross(ComplexF64))
    @inline _pbtrf_cross(::Type{Float32}) = _PBTRF_CROSS_F32()
    @inline _pbtrf_cross(::Type{Float64}) = _PBTRF_CROSS_F64()
    @inline _pbtrf_cross(::Type{ComplexF32}) = _PBTRF_CROSS_C32()
    @inline _pbtrf_cross(::Type{ComplexF64}) = _PBTRF_CROSS_C64()
else
    @inline _pbtrf_cross(::Type{<:BlasFloat}) = _PBTRF_CROSS_PREF::Int   # pinned (trim builds land here)
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
function _pbtrf_blocked!(AB::AbstractMatrix{T}, n::Int, kd::Int) where {T}
    nb = _pbtrf_nb(T, kd)
    ldabs = stride(AB, 2)
    ld2 = ldabs - 1                                    # the LDAB-1 of every BLAS call in dpbtrf.f
    tc = T <: Complex ? 'C' : 'T'                      # dpbtrf 'Transpose' / zpbtrf 'Conjugate transpose'
    rone = one(real(T))
    W = zeros(T, nb + 1, nb)                           # corner work array, zeroed ONCE (see above)
    S = Matrix{T}(undef, nb, nb)                       # dense diagonal-block scratch (see (a)/(b))
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

# ── blocked kernel, uplo='U': ride the L kernel through a conjugate-transpose band repack ─────────
# A = Uᴴ·U with U upper-banded ⟺ A = L·Lᴴ with L = Uᴴ lower-banded: the SAME matrix in lower band
# storage is ABL[1+d, j] = A[j+d, j] = conj(A[j, j+d]) = conj(ABU[kd+1-d, j+d]), and the factor maps
# back as U[j-d, j] = conj(L[j, j-d]) ⇒ ABU[kd+1-d, j] = conj(ABL[1+d, j-d]).
# Why not a direct U port (dpbtrf.f DO 70 — potrf 'U' + trsm 'L','U','T' + syrk 'U','T')? Built and
# validated it first: correct, but 0.90×OB @kd=64 and 0.40×OB / 0.28×AOCL @kd=256 on Zen4 — PB's
# upper/transposed L3 band paths are far off the lower ones. The repack costs two O(kd·n) band
# passes vs the O(n·kd²) factor — ≤ ~3% at kd ≥ 32 (crossover guarantees kd ≥ 2W) — and puts 'U' on
# the exact proven-fastest path (anchor-fastest-path / wire-the-fastest-path rules). The failing
# column needs no translation: leading minors of A are uplo-independent, and _pbtrf_blocked! already
# reports dpotf2-ordered global columns.
function _pbtrf_blocked_U!(AB::AbstractMatrix{T}, n::Int, kd::Int) where {T}
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
