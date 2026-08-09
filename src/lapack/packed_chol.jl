# LAPACK packed Cholesky — dpptrf / dpptrs, generic over s/d/c/z (Hermitian for complex; conj folds to
# identity on real). Ports Reference-LAPACK dpptrf.f (left-looking upper via tpsv+dot, right-looking
# lower via scal+spr) and dpptrs.f.
#
# LAPACK packed storage `AP` is a length-n(n+1)/2 vector holding one triangle column-by-column:
#   uplo='U': AP[i + (j-1)·j÷2]        = A[i,j]  for i ≤ j   (columns of the upper triangle)
#   uplo='L': AP[i + (j-1)·(2n-j)÷2]   = A[i,j]  for i ≥ j   (columns of the lower triangle)

@inline _pp_u(i::Int, j::Int) = i + ((j - 1) * j) >> 1                 # upper packed index, i ≤ j
@inline _pp_l(i::Int, j::Int, n::Int) = i + ((j - 1) * (2n - j)) >> 1  # lower packed index, i ≥ j

# Column length below which pptrf!'s upper path forward-substitutes inline instead of calling tpsv!.
# req#8 tier: MEASURE (an algorithm-switch crossover set by call overhead vs vectorised work, not by
# cache residency — the same class as `_GETF2_BASE`), stated rather than dressed up as a derivation.
# Zen4, PB/OpenBLAS, sweeping the cutoff (rows = order n, best per row marked):
#     n=16   cut0 0.913  cut16 1.418  cut32 1.418*        n=96   cut0 0.897  cut32 0.920*  cut64 0.850
#     n=32   cut0 0.883  cut16 0.978  cut32 1.168*        n=128  cut0 0.913  cut32 0.926*  cut64 0.887
#     n=64   cut0 0.884  cut16 0.904  cut32 0.940*        n=512  cut0 0.962  cut32 0.963*  cut96 0.950
# 32 is best-or-within-noise at every order measured; 0 (always tpsv) costs 1.42→0.91 at n=16 and
# 1.17→0.88 at n=32, i.e. it turns two passing sizes into misses.
# ⚠ MEASURED ON ZEN4 ONLY — galen was off the network and Zen5 is not yet re-locked. Needs fleet
# validation before it is trusted to extrapolate (req#8: derive → validate on the fleet → ship).
const _PPTRF_TPSV_MIN = 32

# ── lower path: trailing-order below which the rank-1 downdate inlines instead of calling spr! ─────
# SEPARATE KNOB from _PPTRF_TPSV_MIN above. The lower path reused that constant, but its table was
# measured entirely on the UPPER path (tpsv-vs-inline); the lower path's question is spr!-vs-inline,
# a different crossover, and 32 was simply wrong for it. Because `m = n-j < 32` for EVERY column when
# n=32, the whole factorization ran the scalar inline loop and never called spr! at all — measured
# 0.738 vs AOCL, the worst cell in packed Cholesky. Sweeping the cutoff (Zen4, ratio AOCL/PB):
#     n=16  cut8 0.881  cut16 0.976*  cut32 0.976        n=64   cut8 0.921  cut16 0.927*  cut32 0.891
#     n=32  cut8 0.825  cut16 0.846*  cut32 0.733        n=128  cut8 1.012  cut16 1.015*  cut32 1.003
#     n=48  cut8 0.875  cut16 0.886*  cut32 0.822        n=512  cut8 1.261  cut16 1.262   cut32 1.265*
# 16 = 2W is best or tied at every order ≤ 128. (Lowering the cutoff alone did NOT close the gate —
# it took the pointer-direct spr path too; see _spr_simd_lower_ptr!.)
# req#8 tier: MEASURE — a call-overhead-vs-vectorised-work crossover, the same class as _GETF2_BASE
# and _pbtrf_cross, not a residency bound. Candidates are DERIVED as {1,2,3,4}·W: the crossover is
# where one SIMD store's worth of work starts to amortise a call, so the vector width is the unit.
const _PPTRF_SPR_MIN_PREF = @load_preference("pptrf_spr_min", nothing)
@static if isnothing(_PPTRF_SPR_MIN_PREF)
    function _measure_pptrf_spr_min(::Type{T})::Int where {T}
        vw = _vwidth(T)
        Base.generating_output() && return 2 * vw     # don't burn a measure during precompilation
        try
            n = 4 * vw                                # the order where the cutoff bites hardest
            len = (n * (n + 1)) >> 1
            AP = Vector{T}(undef, len)
            refill! = () -> begin                     # diagonally dominant ⇒ HPD in packed lower form
                fill!(AP, one(T))
                @inbounds for j in 1:n
                    AP[_pp_l(j, j, n)] = T(2 * n + 2)
                end
            end
            # ⚠ DUELS AGAINST THE DECLARED DEFAULT. Two defects lived here, and they are the same two
            # found in three other sweeps on 2026-08-06/07:
            #   * `tbest = typemax(UInt64)` — `typemax*95` WRAPS to 2^64-95, still far above any real
            #     time, so the FIRST candidate always displaced and `best = 2*vw` was unreachable. The
            #     effective incumbent was whichever cut happened to be swept first (vw).
            #   * a fixed-margin comparison on two medians, which decides by noise whenever the
            #     candidates sit within the margin. Measured, one knob per fresh process: F64 resolved
            #     8 / 16 — two different packed-Cholesky cuts shipping from one binary.
            # Each candidate now duels the DECLARED DEFAULT over rotated rounds with a per-round median
            # and a supermajority, so the default keeps ties and a lucky window cannot displace it.
            # `refill!` stays OUTSIDE the timed region: the factorization is destructive, so every
            # timed round must start from a fresh HPD packed matrix.
            inc() = (refill!(); _pptrf_lower!(AP, n, 2 * vw))
            # ⚠ WAS `for c in (vw, 3vw, 4vw) … && return c` — first-past-the-post over a hand-written
            # order, not an argmin: it shipped whichever candidate happened to be listed first among
            # those beating the incumbent, and never compared them to each other. Same defect measured
            # on `_ger_np` (Zen5 shipped 8 while 1 was ~12% faster at the gate's n=2048).
            # `_tune_duel_pick` keeps the fixed-incumbent bar and adds the argmin. `_tune_duel` does its
            # own untimed warmup of both arms, so the explicit warmup line is no longer needed.
            cand = ((vw, () -> (refill!(); _pptrf_lower!(AP, n, vw))),
                    (3 * vw, () -> (refill!(); _pptrf_lower!(AP, n, 3 * vw))),
                    (4 * vw, () -> (refill!(); _pptrf_lower!(AP, n, 4 * vw))))
            w = _tune_duel_pick(inc, cand)
            return isnothing(w) ? 2 * vw : w
        catch
            return 2 * vw
        end
    end
    const _PPTRF_SPR_F32 = Base.OncePerProcess{Int}(() -> _measure_pptrf_spr_min(Float32))
    const _PPTRF_SPR_F64 = Base.OncePerProcess{Int}(() -> _measure_pptrf_spr_min(Float64))
    const _PPTRF_SPR_C32 = Base.OncePerProcess{Int}(() -> _measure_pptrf_spr_min(ComplexF32))
    const _PPTRF_SPR_C64 = Base.OncePerProcess{Int}(() -> _measure_pptrf_spr_min(ComplexF64))
    @inline _pptrf_spr_min(::Type{Float32}) = _PPTRF_SPR_F32()
    @inline _pptrf_spr_min(::Type{Float64}) = _PPTRF_SPR_F64()
    @inline _pptrf_spr_min(::Type{ComplexF32}) = _PPTRF_SPR_C32()
    @inline _pptrf_spr_min(::Type{ComplexF64}) = _PPTRF_SPR_C64()
    @inline _pptrf_spr_min(::Type{T}) where {T} = 2 * _vwidth(T)   # generic/AD eltypes
else
    @inline _pptrf_spr_min(::Type{<:Any}) = _PPTRF_SPR_MIN_PREF::Int   # pinned (trim builds land here)
end

# ── pptrf!: packed Cholesky factorization (dpptrf.f) ──────────────────────────────────────────────
# uplo='U': A = Uᴴ·U (left-looking — solve Uᴴu = col then the diagonal). uplo='L': A = L·Lᴴ
# (right-looking — scale column, rank-1 downdate). Overwrites AP. Throws PosDefException.
function pptrf!(AP::AbstractVector; uplo::AbstractChar = 'L')
    n = _pp_order(length(AP))
    if uplo == 'U'
        # Left-looking, exactly as dpptrf.f does it: per column, ONE packed triangular solve
        # (Uᴴ·u = A[1:j-1, j]) then a dot product for the pivot. The previous form inlined both as a
        # scalar triple loop; the hand-rolled dot carries a loop-dependency LLVM will not vectorise, so
        # it ran the whole O(n³) left-look at scalar speed while the reference used optimised tpsv/dot.
        # Measured on Zen3 vs OpenBLAS before this: 0.55 at n=128 falling to 0.19 at n=1024 (5.3× SLOWER).
        # Both operands are contiguous in upper-packed storage — the leading order-(j-1) triangle is the
        # prefix AP[1:jc], and column j is AP[jc+1 : jc+j-1] — so no packing or copy is needed.
        tr = eltype(AP) <: Complex ? 'C' : 'T'
        @inbounds for j in 1:n
            jc = ((j - 1) * j) >> 1                       # 0-based start of column j
            ajj = real(AP[jc + j])
            if j > 1
                if (j - 1) <= _PPTRF_TPSV_MIN             # short column: inline, no call to amortise
                    for k in 1:(j - 1)
                        s = AP[jc + k]
                        for i in 1:(k - 1)
                            s -= conj(AP[_pp_u(i, k)]) * AP[jc + i]
                        end
                        AP[jc + k] = s / AP[_pp_u(k, k)]
                    end
                else
                    tpsv!(view(AP, 1:jc), view(AP, (jc + 1):(jc + j - 1)); uplo = 'U', trans = tr, diag = 'N')
                end
                for i in 1:(j - 1)                        # U[j,j]² = A[j,j] − Σ|u[i]|²
                    ajj -= abs2(AP[jc + i])
                end
            end
            ajj > 0 || throw(PosDefException(j))
            AP[jc + j] = sqrt(ajj)
        end
    elseif uplo == 'L'
        # Right-looking, as dpptrf.f does it: scale the column, then ONE packed rank-1 downdate of the
        # trailing triangle (dspr/zhpr). In lower-packed storage columns j+1..n follow column j directly,
        # so the trailing triangle of order n-j is exactly the contiguous tail AP[_pp_l(j+1,j+1,n):end]
        # and the multiplier column is the contiguous run AP[_pp_l(j+1,j,n) .. _pp_l(n,j,n)] — spr!/hpr!
        # can take both as views with no packing or copy.
        _pptrf_lower!(AP, n, _pptrf_spr_min(eltype(AP)))
    else
        throw(ArgumentError("pptrf!: uplo must be 'L' or 'U'"))
    end
    return AP
end

# Lower packed Cholesky, right-looking (dpptrf.f): scale the column, then ONE packed rank-1 downdate
# of the trailing triangle. In lower-packed storage columns j+1..n follow column j directly, so the
# trailing triangle of order n-j is exactly the contiguous tail AP[_pp_l(j+1,j+1,n):end] and the
# multiplier column is the contiguous run AP[_pp_l(j+1,j,n) .. _pp_l(n,j,n)] — no packing or copy.
# `cut` is a parameter (not read from the knob) so the Measure harness can race candidate values.
function _pptrf_lower!(AP::AbstractVector, n::Int, cut::Int)
    cplx = eltype(AP) <: Complex
    @inbounds for j in 1:n
        ajj = real(AP[_pp_l(j, j, n)])
        ajj > 0 || throw(PosDefException(j))
        ajj = sqrt(ajj); AP[_pp_l(j, j, n)] = ajj; invd = inv(ajj)
        m = n - j
        m == 0 && continue
        for i in (j + 1):n                              # scale L[j+1:n, j]
            AP[_pp_l(i, j, n)] *= invd
        end
        if m > cut
            xs = _pp_l(j + 1, j, n)
            if !cplx && eltype(AP) <: BlasReal && AP isa StridedVector && stride(AP, 1) == 1
                # Straight to the kernel: no SubArrays, no public entry. See _spr_simd_lower_ptr!
                # for the per-call measurement that motivated this.
                T = eltype(AP)
                GC.@preserve AP begin
                    p = pointer(AP)
                    _spr_simd_lower_ptr!(
                        m, -one(T),
                        p + (_pp_l(j + 1, j + 1, n) - 1) * sizeof(T),
                        p + (xs - 1) * sizeof(T)
                    )
                end
            else                                        # complex, AD eltypes, non-unit stride
                x = view(AP, xs:(xs + m - 1))
                A22 = view(AP, _pp_l(j + 1, j + 1, n):length(AP))
                if cplx
                    hpr!(-one(real(eltype(AP))), x, A22; uplo = 'L')
                else
                    spr!(-one(eltype(AP)), x, A22; uplo = 'L')
                end
            end
        else
            for q in (j + 1):n                          # short trailing block: inline downdate
                lqj = conj(AP[_pp_l(q, j, n)])          # conj(L[q,j])
                for p in q:n                            # p ≥ q (lower)
                    AP[_pp_l(p, q, n)] -= AP[_pp_l(p, j, n)] * lqj
                end
            end
        end
    end
    return AP
end

# ── pptrs!: solve A·X = B with the pptrf! factor (dpptrs.f) ────────────────────────────────────────
function pptrs!(AP::AbstractVector, B::AbstractVecOrMat; uplo::AbstractChar = 'L')
    n = _pp_order(length(AP))
    Bm = _gt_asmat(B); size(Bm, 1) == n || throw(DimensionMismatch("pptrs!: size(B,1) must equal n"))
    nrhs = size(Bm, 2)
    if uplo == 'U'
        @inbounds for j in 1:nrhs
            for k in 1:n                                  # Uᴴ·y = b (forward)
                s = Bm[k, j]
                for i in 1:(k - 1)
                    s -= conj(AP[_pp_u(i, k)]) * Bm[i, j]
                end
                Bm[k, j] = s / AP[_pp_u(k, k)]
            end
            for k in n:-1:1                               # U·x = y (backward)
                s = Bm[k, j]
                for i in (k + 1):n
                    s -= AP[_pp_u(k, i)] * Bm[i, j]
                end
                Bm[k, j] = s / AP[_pp_u(k, k)]
            end
        end
    elseif uplo == 'L'
        @inbounds for j in 1:nrhs
            for k in 1:n                                  # L·y = b (forward)
                s = Bm[k, j]
                for i in 1:(k - 1)
                    s -= AP[_pp_l(k, i, n)] * Bm[i, j]
                end
                Bm[k, j] = s / AP[_pp_l(k, k, n)]
            end
            for k in n:-1:1                               # Lᴴ·x = y (backward)
                s = Bm[k, j]
                for i in (k + 1):n
                    s -= conj(AP[_pp_l(i, k, n)]) * Bm[i, j]
                end
                Bm[k, j] = s / AP[_pp_l(k, k, n)]
            end
        end
    else
        throw(ArgumentError("pptrs!: uplo must be 'L' or 'U'"))
    end
    return B
end

# Recover n from a packed length L = n(n+1)/2 (exact integer inverse; validates the length).
@inline function _pp_order(L::Int)
    n = (isqrt(8L + 1) - 1) >> 1
    (n * (n + 1)) >> 1 == L || throw(DimensionMismatch("packed length $L is not n(n+1)/2"))
    return n
end
