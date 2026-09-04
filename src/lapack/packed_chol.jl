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
# ⚠ MEASURED ON ZEN4 ONLY — Zen3 was off the network and Zen5 is not yet re-locked. Needs fleet
# validation before it is trusted to extrapolate (req#8: derive → validate on the fleet → ship).
# PDM: Literal — call-overhead vs vectorised-work crossover, measured Zen4 ONLY; needs fleet validation.
const _PPTRF_TPSV_MIN = 32

# Block width for the blocked lower packed factorisation. PDM tier: PIN-able, default DERIVED from the
# same criterion dense LU/Cholesky use — this faces the identical panel-vs-trailing-BLAS-3 tradeoff, so
# it inherits `_lu_nb`'s validated shape rather than introducing a second unvalidated formula (req#8b).
# NOTE the copy traffic is n³/(6·nb), so a LARGER nb is cheaper on copies but shrinks the BLAS-3
# trailing update; this is the knob to sweep first if the blocked path underperforms.
# PDM: Literal — own panel width, borrows _LU_NB, which is itself a falsified derivation. | tune: candidate
const _PPTRF_BLK_NB = @load_preference("pptrf_blk_nb", _LU_NB)::Int

# Smallest n where unpacking pays. MEASURED: 2.26× at n=32 (bench/probes/pptrf_unpack_strategy.jl), so
# the crossover is below 32; it has NOT been measured between 8 and 32, and this value is provisional
# until it is. Below it the unblocked BLAS-2 kernel is kept — at very small n the O(n²) copy stops
# amortising against O(n³/3) work.
# PDM: Literal — smallest n where unpacking pays; measured 2.26x at n=32, untested between 8 and 32. | tune: candidate
const _PPTRF_BLK_MIN = @load_preference("pptrf_blk_min", 16)::Int

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
# DERIVE (2026-08-19). The Measure duel here is deleted, and the reason is that this session's own
# blocked packed Cholesky made the knob nearly dead. At n >= _PPTRF_BLK_MIN (16) `pptrf!` now routes
# BlasFloat strided input to `_pptrf_lower_blocked!` (see the dispatch below), so `_pptrf_lower!` —
# the only consumer of this cutoff — is reached solely for n < 16 or for generic/AD eltypes.
#
# TWO CONSEQUENCES, both measured:
#  * The gate table above (cut8/16/32 at n=16..512, '16 = 2W is best or tied at every order <= 128')
#    is STALE. It describes the unblocked path at sizes where production no longer runs it.
#  * The duel probed at n = 4*vw, which is 32 for Float64 on AVX-512 — also the blocked regime. It was
#    therefore timing a path production does not take, which is why it resolved 32 on Zen4 (8/8)
#    against a table that says 16, and 16 on Zen3 (6/6). Neither answer governed anything.
# A direct A/B of `_pptrf_lower!` at cut 8/16/32 (Zen4, n=16..256) is likewise off-regime and is
# recorded only to show the disagreement, not as evidence for a value.
#
# So the value reverts to the DERIVED generic fallback that was always beneath the duel: 2*W, the
# vector-width unit the crossover is expressed in (one SIMD store's work amortising a call). Blast
# radius is tiny-n and AD, where correctness matters and a few ns do not. Preference override retained.
# `@load_preference` stays at MODULE level, never in the body: an in-body load is a recorded hazard in
# this tree (it does not const-fold the way the module-level form does). 0 is the unset sentinel, so the
# per-type derivation still applies when no preference is set.
# PDM: Exempt — 0 is the unset sentinel, not a size. | tune: n/a
const _PPTRF_SPR_PREF = @load_preference("pptrf_spr_min", 0)::Int   # req8-ok: unset sentinel, not a tuning value
@inline _pptrf_spr_min(::Type{T}) where {T} = _PPTRF_SPR_PREF > 0 ? _PPTRF_SPR_PREF : 2 * _vwidth(T)

# ── pptrf!: packed Cholesky factorization (dpptrf.f) ──────────────────────────────────────────────
# uplo='U': A = Uᴴ·U (left-looking — solve Uᴴu = col then the diagonal). uplo='L': A = L·Lᴴ
# (right-looking — scale column, rank-1 downdate). Overwrites AP. Throws PosDefException.
# ── Blocked lower packed Cholesky ────────────────────────────────────────────────────────────────
#
# WHY. Measured 2026-08-18 (bench/probes/pptrf32_decomp.jl, same-flops control): our DENSE potrf! is
# 2.9-3.9× faster than this file's packed kernel for identical n³/3 flops, because `_pptrf_lower!` is
# an unblocked BLAS-2 column loop while potrf! is blocked + SIMD on the tuned L3 kernels. The gate gap
# vs AOCL is 18%; the gap vs our own dense path is ~300%. End-to-end unpack→potrf!→repack measured
# 2.26/3.05/3.46/3.68/3.91× at n=32/64/128/256/512 (bench/probes/pptrf_unpack_strategy.jl).
#
# WHY BLOCK-COLUMN AND NOT WHOLE-MATRIX. A whole-matrix unpack needs O(n²) scratch — 32 MB against
# packed's 16 MB at n=2048 — which hands back exactly the memory packed storage exists to save. This
# processes ONE block column at a time: O(n·nb) scratch (~0.8 MB at n=2048, nb=48).
#
# THE COPY TRAFFIC IS ~1%, which is what makes the bounded-memory form viable: each outer step touches
# the trailing triangle once, so total copied elements are Σ(n−k)²/2 over n/nb steps = n³/(6·nb),
# against n³/3 flops — at nb=48 that is 3/(6·48) ≈ 1%.
@inline function _pp_unpack_bcol!(W, AP, n::Int, k::Int, kb::Int)
    Z = zero(eltype(W))
    @inbounds for jj in 1:kb
        j = k + jj - 1
        s = _pp_l(j, j, n); off = j - k        # rows 1..off of this W column sit ABOVE the diagonal
        for i in 1:off
            W[i, jj] = Z                       # not stored in packed form; zero so the gemm sees no garbage
        end
        for t in 0:(n - j)
            W[off + 1 + t, jj] = AP[s + t]     # column j is CONTIGUOUS in lower-packed storage
        end
    end
    return W
end

@inline function _pp_repack_bcol!(AP, W, n::Int, k::Int, kb::Int)
    @inbounds for jj in 1:kb
        j = k + jj - 1
        s = _pp_l(j, j, n); off = j - k
        for t in 0:(n - j)
            AP[s + t] = W[off + 1 + t, jj]     # only the LOWER part is written back
        end
    end
    return AP
end

# Right-looking blocked factorisation. `W` holds the current panel, `V` the trailing block column being
# updated; both are n×nb scratch (only the leading m×kb / mm×jb corner is used).
function _pptrf_lower_blocked!(AP::AbstractVector{T}, n::Int, nb::Int, W, V) where {T}
    k = 1
    while k <= n
        kb = min(nb, n - k + 1); m = n - k + 1
        _pp_unpack_bcol!(W, AP, n, k, kb)
        d = view(W, 1:kb, 1:kb)
        # NOTE: a PosDefException from the diagonal block carries a LOCAL index; it is rethrown with the
        # global column so callers see the same info value the unblocked path reports.
        try
            potrf!(d; uplo = 'L')
        catch e
            e isa PosDefException ? throw(PosDefException(e.info + k - 1)) : rethrow()
        end
        # COMPLEX IS HERMITIAN, NOT SYMMETRIC: A = L·Lᴴ, so the panel solve is against L11ᴴ ('C') and the
        # trailing update is A21·A21ᴴ (conjugated B), not the transposes the real case uses. Gating on
        # `T <: BlasFloat` admits ComplexF32/ComplexF64, so getting this wrong is a WRONG ANSWER, not a
        # slowdown — it broke zpptrf at n=64/129 when this path first shipped with 'T'/no-conjugate.
        m > kb && trsm!(
            view(W, (kb + 1):m, 1:kb), d;
            side = 'R', uplo = 'L', transA = (T <: Complex ? 'C' : 'T'), diag = 'N'
        )
        _pp_repack_bcol!(AP, W, n, k, kb)
        jj = k + kb
        while jj <= n                                  # trailing update, one block column at a time
            jb = min(nb, n - jj + 1); mm = n - jj + 1
            _pp_unpack_bcol!(V, AP, n, jj, jb)
            r0 = jj - k + 1
            _gemm_core!(                                     # V -= A21 · A21ᵀ (real) / A21ᴴ (complex)
                view(V, 1:mm, 1:jb), view(W, r0:m, 1:kb), view(W, r0:(r0 + jb - 1), 1:kb),
                -one(T), one(T), false, true, false, T <: Complex
            )
            _pp_repack_bcol!(AP, V, n, jj, jb)
            jj += nb
        end
        k += nb
    end
    return AP
end

# ── Blocked UPPER packed Cholesky ────────────────────────────────────────────────────────────────
#
# Same motivation as the lower path: the unblocked upper kernel is a left-looking per-column
# tpsv+dot (dpptrf.f's shape) while dense potrf! is blocked + SIMD on the tuned L3 kernels. The
# storage argument that produced the 2.9-3.9x same-flops gap is not uplo-specific.
#
# THE ASYMMETRY THAT SHAPES THIS CODE. For LOWER, the trailing update A22 -= A21·A21ᵀ uses a block
# COLUMN, which is contiguous in lower-packed storage. For UPPER, A22 -= U12ᴴ·U12 needs a block ROW.
# That is still cheap here — upper-packed column c stores rows 1..c contiguously, so rows r1..r2 of
# column c are the contiguous run AP[_pp_u(r1,c) .. _pp_u(r2,c)] — but the PANEL is a row-block
# (kb x ncols) rather than a column-block, and the trailing operand order flips to Aᴴ·B.
@inline function _pp_unpack_brow!(R, AP, k::Int, kb::Int, c1::Int, c2::Int)
    Z = zero(eltype(R))
    @inbounds for cc in 1:(c2 - c1 + 1)
        c = c1 + cc - 1
        len = min(c, k + kb - 1) - k + 1        # rows above the diagonal are NOT stored
        s = _pp_u(k, c)
        for t in 0:(len - 1)
            R[1 + t, cc] = AP[s + t]
        end
        for t in len:(kb - 1)
            R[1 + t, cc] = Z                    # zero, so the gemm never reads scratch
        end
    end
    return R
end

@inline function _pp_repack_brow!(AP, R, k::Int, kb::Int, c1::Int, c2::Int)
    @inbounds for cc in 1:(c2 - c1 + 1)
        c = c1 + cc - 1
        len = min(c, k + kb - 1) - k + 1
        s = _pp_u(k, c)
        for t in 0:(len - 1)
            AP[s + t] = R[1 + t, cc]
        end
    end
    return AP
end

# Trailing block column for the UPPER path: rows r0..c of each column c in c1..c2.
@inline function _pp_unpack_bcolU!(V, AP, r0::Int, c1::Int, c2::Int)
    Z = zero(eltype(V)); m = c2 - r0 + 1
    @inbounds for cc in 1:(c2 - c1 + 1)
        c = c1 + cc - 1
        len = c - r0 + 1
        s = _pp_u(r0, c)
        for t in 0:(len - 1)
            V[1 + t, cc] = AP[s + t]
        end
        for t in len:(m - 1)
            V[1 + t, cc] = Z
        end
    end
    return V
end

@inline function _pp_repack_bcolU!(AP, V, r0::Int, c1::Int, c2::Int)
    @inbounds for cc in 1:(c2 - c1 + 1)
        c = c1 + cc - 1
        s = _pp_u(r0, c)
        for t in 0:(c - r0)
            AP[s + t] = V[1 + t, cc]
        end
    end
    return AP
end

# Right-looking blocked upper factorisation. `R` is the kb x ncols panel (nb x n scratch), `V` the
# trailing block column (n x nb scratch).
function _pptrf_upper_blocked!(AP::AbstractVector{T}, n::Int, nb::Int, R, V) where {T}
    k = 1
    while k <= n
        kb = min(nb, n - k + 1); ncols = n - k + 1
        _pp_unpack_brow!(R, AP, k, kb, k, n)
        d = view(R, 1:kb, 1:kb)
        try
            potrf!(d; uplo = 'U')
        catch e
            e isa PosDefException ? throw(PosDefException(e.info + k - 1)) : rethrow()
        end
        # U11ᴴ · U12 = A12  (complex is HERMITIAN: 'C', not 'T' — the lower path shipped that bug once)
        ncols > kb && trsm!(
            view(R, 1:kb, (kb + 1):ncols), d;
            side = 'L', uplo = 'U', transA = (T <: Complex ? 'C' : 'T'), diag = 'N'
        )
        _pp_repack_brow!(AP, R, k, kb, k, n)
        r0 = k + kb; c1 = r0
        while c1 <= n                                   # A22 -= U12ᴴ · U12, one block column at a time
            jb = min(nb, n - c1 + 1); c2 = c1 + jb - 1; m = c2 - r0 + 1
            _pp_unpack_bcolU!(V, AP, r0, c1, c2)
            _gemm_core!(
                view(V, 1:m, 1:jb),
                view(R, 1:kb, (r0 - k + 1):(c2 - k + 1)),
                view(R, 1:kb, (c1 - k + 1):(c2 - k + 1)),
                -one(T), one(T), true, false, T <: Complex, false
            )
            _pp_repack_bcolU!(AP, V, r0, c1, c2)
            c1 += nb
        end
        k += nb
    end
    return AP
end

function pptrf!(AP::AbstractVector; uplo::AbstractChar = 'L')
    n = _pp_order(length(AP))
    if uplo == 'U'
        Tu = eltype(AP)
        nbu = min(_PPTRF_BLK_NB, n)
        # `_dense1(AP)`, not the inline `AP isa StridedVector && stride(AP,1)==1`: `{s,d,c,z}pptrf_64_`
        # builds a `PtrVector` over the caller's packed buffer (cabi_lapack.jl), and `PtrVector` is not in
        # the closed `StridedVector` union — so every C-ABI `pptrf` skipped this whole BLOCKED arm and took
        # the unblocked per-column `tpsv!` path instead. kb `strided-gates-drop-pointer-operands-to-scalar`.
        if Tu <: BlasFloat && _dense1(AP) && n >= _PPTRF_BLK_MIN
            # Arena borrows, EXACT shapes: the packed-row block R is nb×n (`ld == nbu`) and the
            # trailing-update block V is n×nb (`ld == n`). The two used to SHARE the `pptrfv` field
            # across uplo; separate borrows are strictly safer and cost nothing, since only one uplo
            # runs per call. `ld` is exact rather than padded because the field it replaces carried no
            # padding rule either — its `ld` was just the grown row count.
            @scope arn begin
                R = borrow!(arn, Tu, nbu, n)
                V = borrow!(arn, Tu, n, nbu)
                return _pptrf_upper_blocked!(AP, n, nbu, R, V)
            end
        end
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
        # Blocked path when it can pay: it needs at least one full panel plus a trailing block to
        # amortise the unpack, hence n > 2·nb. Everything else (AD eltypes, strided/offset vectors,
        # small n) keeps the unblocked kernel unchanged.
        T = eltype(AP)
        # nb is CLAMPED to n: for n ≤ nb the loop degenerates to a single panel, i.e. exactly
        # unpack→dense potrf!→repack, which is the 2.26× case measured at n=32. Gating on n > 2·nb
        # would have excluded n=32 and n=48 — the very cells that miss the gate — so the threshold is
        # on where the copy starts to amortise, not on having a trailing block.
        nb = min(_PPTRF_BLK_NB, n)
        if T <: BlasFloat && _dense1(AP) && n >= _PPTRF_BLK_MIN   # `_dense1`: see the uplo='U' arm above
            @scope arn begin                       # two n×nb borrows, exact ld — see the uplo='U' arm
                W = borrow!(arn, T, n, nb)
                V = borrow!(arn, T, n, nb)
                _pptrf_lower_blocked!(AP, n, nb, W, V)
            end
        else
            _pptrf_lower!(AP, n, _pptrf_spr_min(T))
        end
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
            if !cplx && eltype(AP) <: BlasReal && _dense1(AP)   # `_dense1`: see pptrf!'s blocked arms
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
    (n * (n + 1)) >> 1 == L || _throw_packed_len(L)
    return n
end
