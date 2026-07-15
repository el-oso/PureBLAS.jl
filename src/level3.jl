# BLAS-3 beyond gemm: trmm/trsm (triangular ·/solve), and later syrk/herk/syr2k/her2k/symm/hemm.
# Strategy — recursive 2×2 blocking that reuses the gate-passing `gemm!` for the off-diagonal update
# and bottoms out (block ≤ _TRMM_BASE) in the L2 kernels (trmv/trsv per B-column for side L;
# column-axpy/solve for side R). This is the L3 analogue of the trsv/trmv "diagonal block + gemv"
# decomposition: gemm carries the flops, the small triangular base carries the structure. α is applied
# as a final scale (kept out of the recursion). Generic `T<:Number` path via the L2 generic kernels.

const _TRMM_BASE = 128        # ≤ this → _trmm_small! directly (capped by _L3_NB=128 M scratch)
const _TRMM_RPANEL = 512
# side-R packed kc: the triangular B-micropanel (nr=_NR wide, kc deep) is ½·L1 resident — the SAME
# residency criterion as gemm's _KC (identical nr), so derive it from _KC rather than a hand-fit literal
# (was 384 = ¾·L1, a req#8 violation). Preferences "trmm_rkc" pins it if trmm-R measures a different opt.
const _TRMM_RKC = @load_preference("trmm_rkc", _KC)::Int
const _TRMM_RPACK = 448        # > this → packed single-pass side-R (mirrors the L cut at _GEMM_UNPACK_MAX)       # side-R flat-loop panel width (fat off-diag gemms; diagonal recurses)
@inline _trsplit(k::Int) = (k ÷ 2)                 # 2×2 split point
@inline _opchar(tr::Bool, cj::Bool) = tr ? (cj ? 'C' : 'T') : 'N'

# off-diagonal update C += op(A)·B — straight to the dispatch core (skip gemm!'s kwarg/check layer;
# the recursion guarantees the shapes).
@inline _gemm_acc!(C, A, B, tr::Bool, cj::Bool) =
    _gemm_core!(C, A, B, one(eltype(C)), one(eltype(C)), tr, false, cj, false)

# ── trmm side='L':  B := op(A)·B,  A k×k triangular (k=size(B,1)), unscaled ──────────────────────
# NOTE: trmm! routes large real side-L to the single-pass `_trmm_packed!` (the proven-fastest path); a
# cache-oblivious recursion here was measured SLOWER at every size, so this recursion is only the
# fallback for complex / side-R / small. (See memory anchor-fastest-path.)
# Dense small-k trmm base (side L): pivot-outer over contiguous columns of A and B — no per-column
# view/call. N-cases in axpy form (B[i,c]'s contribution scattered to its column band BEFORE it is
# overwritten: upper ascending / lower descending pivots); T-cases in dot form (B[i,c] rebuilt from the
# still-original band: upper descending / lower ascending). Both hit the gated SIMD L1 kernels. Real only.
function _trmm_dense_L!(up::Bool, tr::Bool, unit::Bool, A, B)
    k = size(A, 1); n = size(B, 2); T = eltype(B); sz = sizeof(T)
    lda = stride(A, 2); ldb = stride(B, 2)
    GC.@preserve A B begin
        pA = pointer(A); pB = pointer(B)
        @inbounds if !tr
            for i in (up ? (1:k) : (k:-1:1))
                len = up ? (i - 1) : (k - i); rs = up ? 1 : (i + 1)
                aptr = pA + ((i - 1) * lda + (rs - 1)) * sz
                d = unit ? one(T) : A[i, i]
                for c in 1:n
                    t = B[i, c]
                    len > 0 && _axpy_simd!(len, t, aptr, pB + ((c - 1) * ldb + (rs - 1)) * sz)
                    B[i, c] = d * t
                end
            end
        else
            for i in (up ? (k:-1:1) : (1:k))
                len = up ? (i - 1) : (k - i); rs = up ? 1 : (i + 1)
                aptr = pA + ((i - 1) * lda + (rs - 1)) * sz
                d = unit ? one(T) : A[i, i]
                for c in 1:n
                    s = len > 0 ? _dot_simd(len, aptr, pB + ((c - 1) * ldb + (rs - 1)) * sz, T) : zero(T)
                    B[i, c] = muladd(d, B[i, c], s)
                end
            end
        end
    end
    return B
end
# Materialized-triangle base: copy op(A)'s stored triangle into a dense scratch (other half zero, unit
# diag → 1) and run ONE gemm — 2× the triangle's flops but at gemm throughput, with no per-column calls.
# OB's trmm base is throughput-bound (a multiply, unlike trsm's sequential solve), so this is the base
# that keeps up; the recursion's true gemm off-diagonals bound the waste to ~base/k. Real non-conj.
function _mat_tri!(M, A, k::Int, up::Bool, tr::Bool, unit::Bool)
    T = eltype(M)
    @inbounds if !tr
        for j in 1:k                             # N: per column, copy the stored segment + zero the rest
            lo = up ? 1 : j; hi = up ? j : k     # (contiguous — the compiler vectorizes both loops)
            @simd for i in 1:(lo - 1); M[i, j] = zero(T); end
            @simd for i in lo:hi; M[i, j] = A[i, j]; end
            @simd for i in (hi + 1):k; M[i, j] = zero(T); end
            unit && (M[j, j] = one(T))
        end
    else                                         # T: transpose-on-store (strided source, scalar)
        for j in 1:k
            lo = up ? j : 1; hi = up ? k : j     # M column j = op(A) col j = A row j, stored part
            @simd for i in 1:(lo - 1); M[i, j] = zero(T); end
            for i in lo:hi; M[i, j] = A[j, i]; end
            @simd for i in (hi + 1):k; M[i, j] = zero(T); end
            unit && (M[j, j] = one(T))
        end
    end
    return M
end
# ≤ this → scratch-free dense substitution (per-row SIMD axpy/dot, no materialize/scratch setup).
# Above it → materialize+microkernel (_trmm_small!). MEASURED crossover is k=4 on WIDE SIMD: at k=8 the
# direct path's per-row axpys are length ~k/2=4, only a quarter of an AVX-512 register, so it loses hard
# (Zen4 n=8 dropped 1.20→0.57 when widened to 8) while _trmm_small!'s 8×8 tile is one efficient op.
# Widening only plausibly helps narrow SIMD (Zen3 W=4: len-4 axpy = a full register) — under A/B; keep 4
# as the wide-SIMD-safe default. Preference lets a box override without a code push.
const _TRMM_DDIRECT = @load_preference("trmm_ddirect", 4)
# Small-k trmm at HALF flops and gemm throughput: materialize op(A) into a dense scratch (zeros in the
# unstored half make every read safe), copy B to scratch (in-place source), then run the UNPACKED gemm
# micro-kernels with a per-tile K-TRIM — each C-tile contracts only the p-range where M's triangle is
# nonzero, so the only waste is the mr×mr (or nr×nr) diagonal straddle. No packing, no per-column calls.
# Requires k ≤ _L3_NB (the M scratch); real non-conj.
function _trmm_small!(side_left::Bool, up::Bool, tr::Bool, unit::Bool, A, B)
    T = eltype(B); k = size(A, 1)
    upM = (up != tr)                                     # op(A)'s triangle after the on-store transpose
    M = _l3_tmp(T); _mat_tri!(M, A, k, up, tr, unit)     # full matrix scratch: ldM = _L3_NB, no view
    W = _vwidth(T); mr = _MR * W; nr = _NR; sz = sizeof(T)
    ldM = _L3_NB; ldb = stride(B, 2)
    if side_left                                         # B(k×n) := M·B, IN PLACE, dependency-ordered:
        n = size(B, 2)                                   # upM → top-down row-tiles (each reads rows ≥ its
        GC.@preserve M B begin                           # start, still untouched; registers hold the tile
            Mp = pointer(M); Bp = pointer(B)             # between read and store). lower → bottom-up.
            nt = cld(k, mr)
            for t in (upM ? (0:(nt - 1)) : ((nt - 1):-1:0))
                ir = t * mr; mre = min(mr, k - ir)
                plo = upM ? ir : 0
                phi = upM ? k : min(k, ir + mre)
                Ap = Mp + plo * ldM * sz; kc = phi - plo
                jr = 0
                while jr < n
                    nre = min(nr, n - jr)
                    # The B-operand aliases the store target. Full-strip kernels hold the whole tile in
                    # registers (safe). The EDGE kernel is W-row-block serial: for upper M the zero triangle
                    # exactly masks the stale rows; for lower M it does NOT — copy the strip's source
                    # columns to scratch first.
                    if mre == mr && nre == nr
                        _microkernel_unpacked!(Bp, ldb, Ap, ldM, ir, Bp + plo * sz, ldb, jr, kc,
                            one(T), zero(T), Val(_MR), Val(_NR), Val(false), Val(true))
                    elseif nre == nr
                        _microkernel_unpacked_mrows!(Bp, ldb, Ap, ldM, ir, Bp + plo * sz, ldb, jr, kc,
                            one(T), zero(T), mre, cld(mre, W) == 1 ? Val(1) : Val(_MR),
                            Val(_NR), Val(false), Val(true))
                    elseif upM
                        _microkernel_unpacked_edge!(Bp, ldb, Ap, ldM, ir, Bp + plo * sz, ldb, jr, kc,
                            one(T), zero(T), mre, nre, false, true)
                    else
                        Ec = _trsm_tmp(T, _L3_NB, nr)    # kc×nre source copy (dodges the serial aliasing)
                        lde = size(Ec, 1)
                        GC.@preserve Ec begin
                            Ep = pointer(Ec)
                            @inbounds for j in 0:(nre - 1), p in 0:(kc - 1)
                                unsafe_store!(Ep, unsafe_load(Bp, plo + p + (jr + j) * ldb + 1), p + j * lde + 1)
                            end
                            _microkernel_unpacked_edge!(Bp, ldb, Ap, ldM, ir, Ep - jr * lde * sz, lde, jr, kc,
                                one(T), zero(T), mre, nre, false, true)
                        end
                    end
                    jr += nr
                end
            end
        end
    else                                                 # B(m×k) := B·M, IN PLACE: upM → column-tiles
        m = size(B, 1)                                   # right-to-left (each reads cols ≤ its end, i.e.
        GC.@preserve M B begin                           # untouched to its left); lower → left-to-right.
            Mp = pointer(M); Bp = pointer(B)
            nt = cld(k, nr)
            # Row-blocks OUTER: the in-place hazard is row-local (each tile reads/writes only its own
            # rows), so row-blocks are independent — hoisting them keeps the 16×k A-slab L1-resident
            # across its column tiles instead of re-streaming all m×k per tile (the wide-m killer).
            ir = 0
            while ir < m
                mre = min(mr, m - ir)
                for t in (upM ? ((nt - 1):-1:0) : (0:(nt - 1)))
                    jr = t * nr; nre = min(nr, k - jr)
                    plo = upM ? 0 : jr
                    phi = upM ? min(k, jr + nre) : k
                    Bsp = Mp + plo * sz; kc = phi - plo
                    if mre == mr && nre == nr
                        _microkernel_unpacked!(Bp, ldb, Bp + plo * ldb * sz, ldb, ir, Bsp, ldM, jr, kc,
                            one(T), zero(T), Val(_MR), Val(_NR), Val(false), Val(true))
                    elseif nre == nr
                        _microkernel_unpacked_mrows!(Bp, ldb, Bp + plo * ldb * sz, ldb, ir, Bsp, ldM, jr, kc,
                            one(T), zero(T), mre, cld(mre, W) == 1 ? Val(1) : Val(_MR),
                            Val(_NR), Val(false), Val(true))
                    else
                        # Edge kernel is COLUMN-serial and the A-operand is B itself: column j+1's
                        # contraction re-reads columns already stored (they're inside [plo,phi) on both
                        # uplos). Compute the strip into a dest scratch, copy back after.
                        Ec = _trsm_tmp(T, mr, nr); lde = size(Ec, 1)
                        GC.@preserve Ec begin
                            Ep = pointer(Ec)
                            _microkernel_unpacked_edge!(Ep - (ir + jr * lde) * sz, lde,
                                Bp + plo * ldb * sz, ldb, ir, Bsp, ldM, jr, kc,
                                one(T), zero(T), mre, nre, false, true)
                            @inbounds for j in 0:(nre - 1), r in 0:(mre - 1)
                                unsafe_store!(Bp, unsafe_load(Ep, r + j * lde + 1),
                                    ir + r + (jr + j) * ldb + 1)
                            end
                        end
                    end
                end
                ir += mr
            end
        end
    end
    return B
end
# Complex trmm side-L base: materialize op(A)'s k×k triangle ONCE into scratch, then B := M·B via the
# gating SIMD complex gemm — reads A once, vs trmv-per-column re-reading A's triangle n times (O(k²n)).
# The complex analog of the real `_trmm_small!` (materialize + microkernel). k ≤ _TRMM_BASE = _L3_NB.
function _trmm_cmplx_base_L!(up::Bool, tr::Bool, cj::Bool, unit::Bool, k::Int, A, B)
    T = eltype(B); n = size(B, 2)
    M = _l3_tmp(T); Mv = view(M, 1:k, 1:k)
    _mat_tri!(Mv, A, k, up, tr, unit)               # M = op(A) triangle (generic over T; no conj)
    cj && @inbounds(Mv .= conj.(Mv))                # 'C' variant: conjugate the materialized op
    Bt = _trsm_tmp(T, k, n); Btv = view(Bt, 1:k, 1:n)
    gemm!(Btv, Mv, B)                               # Btv = M·B  (complex SIMD gemm)
    copyto!(B, Btv)                                 # B := M·B
    return B
end

# Complex trmm side-R base: B := B·op(A). Materialize op(A) once, then one SIMD complex gemm (B·M).
function _trmm_cmplx_base_R!(up::Bool, tr::Bool, cj::Bool, unit::Bool, k::Int, A, B)
    T = eltype(B); m = size(B, 1)
    M = _l3_tmp(T); Mv = view(M, 1:k, 1:k)
    _mat_tri!(Mv, A, k, up, tr, unit)
    cj && @inbounds(Mv .= conj.(Mv))
    Bt = _trsm_tmp(T, m, k); Btv = view(Bt, 1:m, 1:k)
    gemm!(Btv, B, Mv)                               # Btv = B·M
    copyto!(B, Btv)
    return B
end

# Complex small-k trmm side-L at HALF the flops of the materialize+dense-gemm base: materialize op(A)
# into the _l3_tmp scratch, then per-row-tile K-TRIMmed _uker_cmplx! calls (contract only op(A)'s
# nonzero p-range per tile — the 2× dense waste is the whole gap; ztrmm n=128 was 0.515 ≈ dense/2).
# IN PLACE: B is operand AND target; each _uker_cmplx! call is atomic (all A/B loads precede all stores),
# and the K-TRIM's p-range is exactly the not-yet-overwritten rows (upM top-down / else bottom-up) → no
# scratch copy needed. B0=overwrite (masked), A1=α==1 (trmm! folds α outside). Requires _CMR ≤ 2 (one
# call per row-tile). Fable-designed 2026-07-05.
function _trmm_cmplx_small_L!(up::Bool, tr::Bool, cj::Bool, unit::Bool, k::Int, A, B)
    Tc = eltype(B); T = real(Tc); n = size(B, 2)
    M = _l3_tmp(Tc); Mv = view(M, 1:k, 1:k)
    _mat_tri!(Mv, A, k, up, tr, unit)                       # M = op(A) triangle (other half zeroed)
    cj && @inbounds(Mv .= conj.(Mv))                        # 'C' variant
    upM = (up != tr)
    W = _vwidth(T); mr = _CMR * W; nr = _CNR_SMALL; sz = sizeof(T)
    ldM = _L3_NB; ldb = stride(B, 2)
    GC.@preserve M B begin
        Mp = Ptr{T}(pointer(M)); Bp = Ptr{T}(pointer(B))
        onr = one(T); zr = zero(T)
        nt = cld(k, mr)
        for t in (upM ? (0:(nt - 1)) : ((nt - 1):-1:0))
            ir = t * mr; mre = min(mr, k - ir)
            plo = upM ? ir : 0                              # K-TRIM: op(A)'s nonzero p-range only
            phi = upM ? k : min(k, ir + mre)
            kc = phi - plo
            Ap = Mp + 2 * plo * ldM * sz                   # M cols [plo,phi); kernel adds ir row offset
            Bs = Bp + 2 * plo * sz                         # B-operand rows [plo,phi); kernel adds jr
            full = cld(mre, W) >= _CMR
            jr = 0
            while jr < n
                nre = min(nr, n - jr)
                if full
                    _uker_cmplx!(Bp, ldb, Ap, ldM, ir, Bs, ldb, jr, kc, onr, zr, mre, nre,
                        Val(_CMR), Val(_CNR_SMALL), Val(false), Val(1), Val(1), Val(true), Val(true), Val(false), Val(false), Val(false), 0, true)
                else
                    _uker_cmplx!(Bp, ldb, Ap, ldM, ir, Bs, ldb, jr, kc, onr, zr, mre, nre,
                        Val(1), Val(_CNR_SMALL), Val(false), Val(1), Val(1), Val(true), Val(true), Val(false), Val(false), Val(false), 0, true)
                end
                jr += nr
            end
        end
    end
    return B
end

# Complex small-k trmm side-R: B(m×k) := B·op(A), half flops via K-TRIM. Row-blocks OUTER (the in-place
# hazard is row-local), column-tiles INNER in dependency order (upM right-to-left / else left-to-right).
# A-operand is B itself (cols [plo,phi)), B-operand is M (rows [plo,phi)); atomic kernel + K-TRIM → the
# read columns are exactly the not-yet-overwritten ones, so no scratch (see _trmm_cmplx_small_L!).
function _trmm_cmplx_small_R!(up::Bool, tr::Bool, cj::Bool, unit::Bool, k::Int, A, B)
    Tc = eltype(B); T = real(Tc); m = size(B, 1)
    M = _l3_tmp(Tc); Mv = view(M, 1:k, 1:k)
    _mat_tri!(Mv, A, k, up, tr, unit)
    cj && @inbounds(Mv .= conj.(Mv))
    upM = (up != tr)
    W = _vwidth(T); mr = _CMR * W; nr = _CNR_SMALL; sz = sizeof(T)
    ldM = _L3_NB; ldb = stride(B, 2)
    GC.@preserve M B begin
        Mp = Ptr{T}(pointer(M)); Bp = Ptr{T}(pointer(B))
        onr = one(T); zr = zero(T); nt = cld(k, nr)
        ir = 0
        while ir < m
            mre = min(mr, m - ir); full = cld(mre, W) >= _CMR
            for t in (upM ? ((nt - 1):-1:0) : (0:(nt - 1)))
                jr = t * nr; nre = min(nr, k - jr)
                plo = upM ? 0 : jr; phi = upM ? min(k, jr + nre) : k; kc = phi - plo
                Aop = Bp + 2 * plo * ldb * sz          # B-operand (A-slot): B cols [plo,phi)
                Bop = Mp + 2 * plo * sz                # M (B-slot): rows [plo,phi)
                if full
                    _uker_cmplx!(Bp, ldb, Aop, ldb, ir, Bop, ldM, jr, kc, onr, zr, mre, nre,
                        Val(_CMR), Val(_CNR_SMALL), Val(false), Val(1), Val(1), Val(true), Val(true), Val(false), Val(false), Val(false), 0, true)
                else
                    _uker_cmplx!(Bp, ldb, Aop, ldb, ir, Bop, ldM, jr, kc, onr, zr, mre, nre,
                        Val(1), Val(_CNR_SMALL), Val(false), Val(1), Val(1), Val(true), Val(true), Val(false), Val(false), Val(false), 0, true)
                end
            end
            ir += mr
        end
    end
    return B
end

# The packed K-TRIM complex trmm base (near-peak PACKED microkernel) vs the weak unpacked _uker_cmplx!.
# Measured (galen/Zen3): the packed complex kernel hits 0.94–0.95×OB at these short-k base shapes, the
# unpacked only 0.68–0.73, and ztrmm is pinned at the unpacked ceiling (0.77 ≈ 0.73). AVX-512 already
# gates via the unpacked path (32 zmm give ample ILP) — restrict packed to AVX2 (W=4) so that gate is
# untouched. Preferences knob "ctrmm_pack". Fable-designed, decomposition-confirmed 2026-07-05.
const _CTRMM_PACK = @load_preference("ctrmm_pack", _vwidth(Float64) == 4)::Bool
# Below this k the packed base's pack overhead loses to the unpacked K-TRIM (measured galen: k=8 0.46 vs
# unpacked ~1.0, k=32 0.75 vs 0.85; crossover ≈48, k=64 packed 0.91 wins). Recursion bases (k>128 split)
# land ≥64 → packed; only tiny single-base trmm stays unpacked. Preferences knob "ctrmm_pack_min".
# pack-vs-unpacked crossover: below this k the unpacked small kernel's lower setup beats the packed path's
# O(k²) M-materialize. Empirically side-DEPENDENT (packed_R amortizes ~4·_CNR, packed_L not until ~8·_CNR
# — the B-vs-M packing asymmetry) and small-n stays sub-gate either way, so a single derived crossover
# doesn't pay; kept at the measured conservative value. Preferences-pinnable. (req#8: acknowledged debt —
# the pack-amortization threshold resists a clean cache/ISA formula; revisit with an OB-style fused pack.)
const _CTRMM_PACK_MIN = @load_preference("ctrmm_pack_min", 48)::Int

# Exact-width remainder column-tile for the packed complex trmm bases. The last column-tile of a
# non-multiple-of-nr panel is partial (width nre∈1:_CNR-1); running it through the nr-wide masked kernel
# computes (nr-nre) PAD columns — and for upper-N that tile sits at MAX K-trim depth (kc=k), so the pad
# is charged at full depth (measured galen ztrmmR spike). Dispatch the runtime nre to a compile-time
# Val{NR}=nre masked kernel so the pad columns are NEVER computed. REQUIRES the slot packed at row-stride
# nre (see packed_R pack loop). B0=A1=true (overwrite; α folded outside). AVX2-only (packed path gated by
# _CTRMM_PACK=W==4); the 5-way branch is compile cost there, never instantiated on AVX-512.
@inline function _trmm_rem_cmplx!(C::Ptr{T}, ldc::Int, AR::Ptr{T}, AI::Ptr{T}, BR::Ptr{T}, BI::Ptr{T},
        kc::Int, alr::T, ali::T, mre::Int, nre::Int, ::Val{MR}, ::Val{SA}, ::Val{SB}) where {T, MR, SA, SB}
    if nre == 1
        _microkernel_cmplx_masked!(C, ldc, AR, AI, BR, BI, kc, alr, ali, mre, nre,
            Val(MR), Val(1), Val(SA), Val(SB), Val(true), Val(true))
    elseif nre == 2
        _microkernel_cmplx_masked!(C, ldc, AR, AI, BR, BI, kc, alr, ali, mre, nre,
            Val(MR), Val(2), Val(SA), Val(SB), Val(true), Val(true))
    elseif nre == 3
        _microkernel_cmplx_masked!(C, ldc, AR, AI, BR, BI, kc, alr, ali, mre, nre,
            Val(MR), Val(3), Val(SA), Val(SB), Val(true), Val(true))
    elseif nre == 4
        _microkernel_cmplx_masked!(C, ldc, AR, AI, BR, BI, kc, alr, ali, mre, nre,
            Val(MR), Val(4), Val(SA), Val(SB), Val(true), Val(true))
    else                                                     # nre == 5
        _microkernel_cmplx_masked!(C, ldc, AR, AI, BR, BI, kc, alr, ali, mre, nre,
            Val(MR), Val(5), Val(SA), Val(SB), Val(true), Val(true))
    end
end

# Packed K-TRIM complex trmm side-L: B := op(A)·B. Materialize op(A)→M (off-triangle zeroed), then PACK B
# ONCE per nc-panel (its data is copied out → the in-place aliasing constraint vanishes, so tiles store
# B0-overwrite in ANY order — no atomic/dependency-order dance) and per output row-tile pack M's K-TRIMmed
# trapezoid M[ir:ir+mre, plo:phi] as the A-operand via _pack_A_cmplx!'s SIMD deinterleave, running the
# near-peak PACKED complex microkernel. (A fused straight-from-A pack was tried 2026-07-09 and REGRESSED
# side-L — it loses _pack_A_cmplx!'s vectorized deinterleave for scalar select-heavy stores; the one-time
# materialize is cheaper than that per-row-tile loss here. Side-R, whose prepack was already scalar, DID
# win from fusing — see _trmm_cmplx_packed_R!.) The diagonal block's below-diagonal zeros are real zeros
# in M (no mask); edge tiles use the masked kernel. α folded outside → A1=true. See kb pureblas-zen3-gate-strategy.
function _trmm_cmplx_packed_L!(up::Bool, tr::Bool, cj::Bool, unit::Bool, k::Int, A, B)
    Tc = eltype(B); T = real(Tc); n = size(B, 2)
    M = _l3_tmp(Tc); Mv = view(M, 1:k, 1:k)
    _mat_tri!(Mv, A, k, up, tr, unit)                       # M = op(A) triangle (other half zeroed)
    cj && @inbounds(Mv .= conj.(Mv))                        # 'C' variant
    upM = (up != tr)
    W = _vwidth(T); mr = _CMR * W; nr = _CNR; sz = sizeof(T)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(n, nr) * nr)
    ApR, ApI, BpR, BpI = _gemm_scratch_cmplx(T, mr * k, cld(nc, nr) * nr * k)
    ldb = stride(B, 2); alr = one(T); ali = zero(T)         # α==1 (folded outside)
    GC.@preserve M B ApR ApI BpR BpI begin
        Bp0 = Ptr{T}(pointer(B)); ARp = pointer(ApR); AIp = pointer(ApI)
        BRp = pointer(BpR); BIp = pointer(BpI)
        jc = 0
        while jc < n
            nce = min(nc, n - jc)
            _pack_B_cmplx!(BpR, BpI, B, 0, jc, k, nce, false, nr)   # B[:, jc-panel], all k rows
            rem = nce - (nce ÷ nr) * nr                    # partial column-tile width (0 if divisible)
            if rem != 0                                    # repack the last slot at row-stride rem (pad-free,
                jip = nce ÷ nr; basep = jip * nr * k       # so the NR=rem kernel computes no pad columns)
                @inbounds for p in 0:(k - 1), c in 0:(rem - 1)
                    v = B[p + 1, jc + jip * nr + c + 1]
                    BpR[basep + p * rem + c + 1] = real(v); BpI[basep + p * rem + c + 1] = imag(v)
                end
            end
            ir = 0
            while ir < k
                mre = min(mr, k - ir)
                plo = upM ? ir : 0                          # K-TRIM: op(A)'s nonzero p-range
                phi = upM ? k : min(k, ir + mre); kc = phi - plo
                _pack_A_cmplx!(ApR, ApI, Mv, ir, plo, mre, kc, false, mr)   # SIMD deinterleave from dense M
                jr = 0
                while jr < nce
                    nre = min(nr, nce - jr); ji = div(jr, nr)
                    boff = (ji * nr * k + plo * nr) * sz
                    Cblk = Bp0 + (2 * ir + 2 * (jc + jr) * ldb) * sz
                    AR = Ptr{T}(ARp); AI = Ptr{T}(AIp)
                    BR = Ptr{T}(BRp + boff); BI = Ptr{T}(BIp + boff)
                    if nre == nr
                        if mre == mr
                            _microkernel_cmplx!(Cblk, ldb, AR, AI, BR, BI, kc, alr, ali,
                                Val(_CMR), Val(_CNR), Val(1), Val(1), Val(true), Val(true))
                        else
                            _microkernel_cmplx_masked!(Cblk, ldb, AR, AI, BR, BI, kc, alr, ali,
                                mre, nre, Val(_CMR), Val(_CNR), Val(1), Val(1), Val(true), Val(true))
                        end
                    else                                     # partial column-tile: stride-rem slot ⇒ plo*rem
                        boffr = (ji * nr * k + plo * rem) * sz
                        _trmm_rem_cmplx!(Cblk, ldb, AR, AI, Ptr{T}(BRp + boffr), Ptr{T}(BIp + boffr),
                            kc, alr, ali, mre, nre, Val(_CMR), Val(1), Val(1))
                    end
                    jr += nr
                end
                ir += mr
            end
            jc += nc
        end
    end
    return B
end

# Packed K-TRIM complex trmm side-R: B := B·op(A). Transposed mirror of _trmm_cmplx_packed_L!: the BIG
# operand is B itself (the gemm A-slot, m×k), packed per mc row-panel (its data is copied out → the
# in-place aliasing dissolves; each output row depends only on its own B row, so B0-overwrite is safe
# in any order). op(A) is the small B-slot operand, FUSED-packed ONCE up front straight from A (uplo/trans/
# conj/unit + off-triangle zeros inline — no dense _mat_tri! materialize, no conj pass, no po2 _l3_tmp
# stride): tile ji at fixed stride ji·nr·k storing only its [plo,phi) rows from slot-row 0 → TOUCHED
# footprint ~k²/2, packed once (no per-ic-panel repack — that regressed the recursion-fed large-k). boff
# needs no plo term (row 0 IS op(A)-row plo); the A-pack keeps the +plo·mr offset (it packs all k B-cols).
# Off-triangle zeros are real zeros in the packed panel (contribute 0, no mask); row/col edges use the
# masked kernel. α folded outside (A1=true); cj baked into the pack ⇒ SA=SB=1. Fable-designed 2026-07-05/09.
function _trmm_cmplx_packed_R!(up::Bool, tr::Bool, cj::Bool, unit::Bool, k::Int, A, B)
    Tc = eltype(B); T = real(Tc); m = size(B, 1)
    upM = (up != tr)
    W = _vwidth(T); mr = _CMR * W; nr = _CNR; sz = sizeof(T)
    # B row-panel is materialized with ALL k cols at once (not a kc-blocked loop), so mc is the
    # CANONICAL 30%·L2 A-block (`_at_gemm_mc`), NOT the per-kc `_at_mc_kc` — keying it on the full k
    # makes mc a moving target that over-blocks small-L2 boxes at small n (galen ztrmmR regression). req#8.
    mc = min(max(mr, (_at_gemm_mc(_HW) ÷ mr) * mr), cld(m, mr) * mr)
    ApR, ApI, BpR, BpI = _gemm_scratch_cmplx(T, mc * k, cld(k, nr) * nr * k)
    ldb = stride(B, 2); alr = one(T); ali = zero(T)         # α==1 (folded outside)
    GC.@preserve B ApR ApI BpR BpI begin
        Bp0 = Ptr{T}(pointer(B)); ARp = pointer(ApR); AIp = pointer(ApI)
        BRp = pointer(BpR); BIp = pointer(BpI)
        # FUSED triangle pack: op(A)'s K-trimmed column-tiles read STRAIGHT from A — uplo/trans/conj/unit
        # and the off-triangle zeros applied inline. No _mat_tri! dense materialize, no conj pass, no po2
        # _l3_tmp stride. Tile ji → base ji·nr·k, rows [plo,phi), row-stride nre (=nr on full tiles; the
        # Route-A remainder slot keeps its pad-free nre). All A reads in-bounds ∀c (gj=jr+c<k, gp<phi≤k) →
        # unconditional load, triangle/diag by select; @simd ivdep on the contiguous BpR/BpI store.
        @inbounds for ji in 0:(cld(k, nr) - 1)
            jr = ji * nr; nre = min(nr, k - jr)
            plo = upM ? 0 : jr; phi = upM ? min(k, jr + nre) : k; kc = phi - plo
            base = ji * nr * k
            for p in 0:(kc - 1)
                gp = plo + p; off = base + p * nre
                @simd ivdep for c in 0:(nre - 1)
                    gj = jr + c
                    a = tr ? A[gj + 1, gp + 1] : A[gp + 1, gj + 1]   # op(A)[gp,gj]
                    intri = upM ? (gp <= gj) : (gp >= gj)
                    dg = unit & (gp == gj)
                    BpR[off + c + 1] = intri ? (dg ? one(T) : real(a)) : zero(T)
                    BpI[off + c + 1] = (intri & !dg) ? (cj ? -imag(a) : imag(a)) : zero(T)
                end
            end
        end
        ic = 0
        while ic < m
            mce = min(mc, m - ic)
            _pack_A_cmplx!(ApR, ApI, B, ic, 0, mce, k, false, mr)   # B row-panel, all k cols, pre-store
            jr = 0
            while jr < k
                nre = min(nr, k - jr); ji = div(jr, nr)
                plo = upM ? 0 : jr                          # K-TRIM: M's nonzero p-range (== small_R)
                phi = upM ? min(k, jr + nre) : k; kc = phi - plo
                ir = 0
                while ir < mce
                    mre = min(mr, mce - ir)
                    aoff = (div(ir, mr) * mr * k + plo * mr) * sz
                    Cblk = Bp0 + (2 * (ic + ir) + 2 * jr * ldb) * sz
                    AR = Ptr{T}(ARp + aoff); AI = Ptr{T}(AIp + aoff)
                    BR = Ptr{T}(BRp + ji * nr * k * sz); BI = Ptr{T}(BIp + ji * nr * k * sz)  # slot ji, row 0 = M-row plo
                    if nre == nr
                        if mre == mr
                            _microkernel_cmplx!(Cblk, ldb, AR, AI, BR, BI, kc, alr, ali,
                                Val(_CMR), Val(_CNR), Val(1), Val(1), Val(true), Val(true))
                        else
                            _microkernel_cmplx_masked!(Cblk, ldb, AR, AI, BR, BI, kc, alr, ali,
                                mre, nre, Val(_CMR), Val(_CNR), Val(1), Val(1), Val(true), Val(true))
                        end
                    else                                     # partial column-tile (last, nre∈1:_CNR-1): NR=nre
                        _trmm_rem_cmplx!(Cblk, ldb, AR, AI, BR, BI, kc, alr, ali, mre, nre,
                            Val(_CMR), Val(1), Val(1))       # kernel → no pad cols computed (upper-N deep-rem)
                    end
                    ir += mr
                end
                jr += nr
            end
            ic += mc
        end
    end
    return B
end

function _trmm_left!(up::Bool, tr::Bool, cj::Bool, unit::Bool, A, B)
    k = size(A, 1)
    if eltype(B) <: BlasReal && !cj && k <= _TRMM_BASE
        return k <= _TRMM_DDIRECT ? _trmm_dense_L!(up, tr, unit, A, B) :
                                    _trmm_small!(true, up, tr, unit, A, B)
    elseif eltype(B) <: BlasComplex && k <= _TRMM_BASE     # complex: K-TRIM small kernel (half flops);
        return !_strided1(B) ? _trmm_cmplx_base_L!(up, tr, cj, unit, k, A, B) :            # strided B → base
               (_CTRMM_PACK && k >= _CTRMM_PACK_MIN) ? _trmm_cmplx_packed_L!(up, tr, cj, unit, k, A, B) :
                             _trmm_cmplx_small_L!(up, tr, cj, unit, k, A, B)     # AVX-512 / tiny-k → unpacked
    elseif k <= _TRMM_BASE                          # AD/generic: trmv on each B column (contiguous)
        @inbounds for c in axes(B, 2)
            _trmv!(up, tr, cj, unit, k, A, view(B, :, c), 1)
        end
        return B
    end
    h = _trsplit(k)
    A11 = view(A, 1:h, 1:h); A22 = view(A, (h + 1):k, (h + 1):k)
    B1 = view(B, 1:h, :); B2 = view(B, (h + 1):k, :)
    # up≠tr → the off-diagonal feeds B1 (process B1's diagonal first, then gemm B1+=off·B2, then B2).
    # up==tr → it feeds B2. The off-diagonal A-block is A12 (above diag) or A21 (below), and gemm's
    # transA carries op. Verified against all four (uplo×trans) cases.
    if up != tr
        off = tr ? view(A, (h + 1):k, 1:h) : view(A, 1:h, (h + 1):k)
        _trmm_left!(up, tr, cj, unit, A11, B1)
        _gemm_acc!(B1, off, B2, tr, cj)
        _trmm_left!(up, tr, cj, unit, A22, B2)
    else
        off = tr ? view(A, 1:h, (h + 1):k) : view(A, (h + 1):k, 1:h)
        _trmm_left!(up, tr, cj, unit, A22, B2)
        _gemm_acc!(B2, off, B1, tr, cj)
        _trmm_left!(up, tr, cj, unit, A11, B1)
    end
    return B
end

# ── trmm side='R':  B := B·op(A),  A k×k triangular (k=size(B,2)), unscaled ───────────────────────
# Base: column-axpy on B's columns. For upper-N, B[:,j] := A[j,j]·B[:,j] + Σ_{i<j} A[i,j]·B[:,i]
# (j descending so B[:,i], i<j, are still original). The four combos mirror trmv's column structure.
function _trmm_right_base!(up::Bool, tr::Bool, cj::Bool, unit::Bool, k::Int, A, B)
    m = size(B, 1)
    # out[:,j] = Σ_i (op A)[i,j]·B[:,i].  (op A)[i,j] = A[i,j] (N) or A[j,i] (T/C). up≠tr ⇒ feeds are
    # the lower-index columns (j descending keeps them original); up==tr ⇒ higher-index (j ascending).
    coef(i, j) = tr ? (cj ? conj(A[j, i]) : A[j, i]) : A[i, j]
    @inbounds if up != tr
        for j in k:-1:1
            unit || _scal_col!(B, j, coef(j, j), m)
            for i in 1:(j - 1); _axpy_col!(B, j, coef(i, j), i, m); end
        end
    else
        for j in 1:k
            unit || _scal_col!(B, j, coef(j, j), m)
            for i in (j + 1):k; _axpy_col!(B, j, coef(i, j), i, m); end
        end
    end
    return B
end
# B[:,j] *= s   and   B[:,j] += a·B[:,i]  on contiguous columns (SIMD where eligible).
@inline function _scal_col!(B, j, s, m)
    if _strided1(B) && eltype(B) <: BlasReal
        GC.@preserve B (_scal_simd_ptr!(pointer(B) + (j - 1) * stride(B, 2) * sizeof(eltype(B)), m, s))
    else
        @inbounds for r in 1:m; B[r, j] *= s; end
    end
end
@inline function _axpy_col!(B, j, a, i, m)
    if _strided1(B) && eltype(B) <: BlasReal
        T = eltype(B); sz = sizeof(T); ldb = stride(B, 2)
        GC.@preserve B _axpy_simd!(m, T(a), pointer(B) + (i - 1) * ldb * sz, pointer(B) + (j - 1) * ldb * sz)
    else
        @inbounds for r in 1:m; B[r, j] += a * B[r, i]; end
    end
end
@inline function _scal_simd_ptr!(p::Ptr{T}, n::Int, s::T) where {T<:BlasReal}
    _scal!(n, s, p, 1)   # _scal! accepts a pointer (level1)
end

# Packed single-pass trmm side-R: B := B·op(A) as ONE K-trimmed blocked-gemm sweep (mirror of
# _trmm_packed!). B is copied once to a contiguous scratch (kills the in-place aliasing outright,
# O(mk) ≪ O(mk²/2)); op(A) packs per pc-block into nr-panels with zeros outside the triangle and only
# the rows each column-tile actually contracts (per-tile K-trim at nr granularity — the trim that kept
# the flat/recursion versions from gemm efficiency lived at panel granularity). Real non-conj.
const _TRMM_BCR = Ref(Matrix{Float64}(undef, 0, 0))
function _pack_B_triR!(Bp::Vector{T}, A, pc::Int, kce::Int, k::Int, upM::Bool, tr::Bool,
        unit::Bool, nr::Int) where {T}
    np = cld(k, nr)
    @inbounds for jp in 0:(np - 1)
        j0 = jp * nr
        plo = upM ? 0 : max(0, j0 - pc)                              # rows this panel's tiles contract
        phi = upM ? clamp(j0 + nr - pc, 0, kce) : kce
        plo >= phi && continue
        base = jp * nr * kce
        for p in plo:(phi - 1)
            gp = pc + p
            for c in 0:(nr - 1)
                gj = j0 + c
                v = if gj < k && (upM ? (gp <= gj) : (gp >= gj))
                    (unit && gp == gj) ? one(T) : (tr ? A[gj + 1, gp + 1] : A[gp + 1, gj + 1])
                else
                    zero(T)
                end
                Bp[base + p * nr + c + 1] = v
            end
        end
    end
    return
end
function _trmm_packedR!(up::Bool, tr::Bool, unit::Bool, A, B, ::Type{T}) where {T<:BlasReal}
    m, k = size(B); W = _vwidth(T); mr = _MR * W; nr = _NR
    upM = (up != tr)
    kc = min(_TRMM_RKC, k); mc = _at_mc_kc(_HW, T, kc, mr, cld(m, mr) * mr)
    _, Bp = _gemm_scratch(T, 0, cld(k, nr) * nr * kc)
    # Pre-pack ALL of B (the gemm A-operand) up front, before any C write — B IS C here, so packing it
    # once both captures the input (no separate copy pass; ~2% of runtime at 1024) and feeds the whole
    # sweep. Slot layout: (pc-block, ic-block) → a fixed-size mr-panel group.
    nic = cld(m, mc); npc = cld(k, kc); slot = cld(mc, mr) * mr * kc
    Apf = _trmm_bpf(T, npc * nic * slot)
    ldb = stride(B, 2); sz = sizeof(T)
    GC.@preserve B Apf Bp begin
        pB = pointer(B)
        pc = 0; pb = 0
        while pc < k                                                 # pre-pack phase (reads only)
            kce = min(kc, k - pc)
            ic = 0; icx = 0
            while ic < m
                mce = min(mc, m - ic)
                off = (pb * nic + icx) * slot
                _pack_A!(view(Apf, (off + 1):(off + cld(mce, mr) * mr * kce)), B, ic, pc, mce, kce,
                    false, one(T), mr)
                ic += mc; icx += 1
            end
            pc += kc; pb += 1
        end
        Apfp = pointer(Apf); Bpp = pointer(Bp)
        pc = 0; pb = 0
        while pc < k
            kce = min(kc, k - pc)
            _pack_B_triR!(Bp, A, pc, kce, k, upM, tr, unit, nr)
            ic = 0; icx = 0
            while ic < m
                mce = min(mc, m - ic)
                App = Apfp + (pb * nic + icx) * slot * sz
                jr = 0
                while jr < k
                    nre = min(nr, k - jr)
                    plo = upM ? 0 : max(0, jr - pc)                  # per-tile K-trim
                    phi = upM ? min(kce, jr + nre - pc) : kce
                    cnt = phi - plo
                    if cnt > 0
                        ow = upM ? (pb == 0) : (pb == div(jr, kc))   # first contributing block → β=0
                        ir = 0
                        while ir < mce
                            mre = min(mr, mce - ir)
                            Apanel = App + (div(ir, mr) * mr * kce + plo * mr) * sz
                            Bpanel = Bpp + (div(jr, nr) * nr * kce + plo * nr) * sz
                            Cblk = pB + ((ic + ir) + jr * ldb) * sz
                            if mre == mr && nre == nr
                                ow ? _microkernel!(Ptr{T}(Cblk), ldb, Ptr{T}(Apanel), Ptr{T}(Bpanel), cnt, Val(_MR), Val(_NR), Val(true)) :
                                     _microkernel!(Ptr{T}(Cblk), ldb, Ptr{T}(Apanel), Ptr{T}(Bpanel), cnt, Val(_MR), Val(_NR), Val(false))
                            else
                                ow ? _microkernel_masked!(Ptr{T}(Cblk), ldb, Ptr{T}(Apanel), Ptr{T}(Bpanel), cnt, mre, nre, Val(_MR), Val(_NR), Val(true)) :
                                     _microkernel_masked!(Ptr{T}(Cblk), ldb, Ptr{T}(Apanel), Ptr{T}(Bpanel), cnt, mre, nre, Val(_MR), Val(_NR), Val(false))
                            end
                            ir += mr
                        end
                    end
                    jr += nr
                end
                ic += mc; icx += 1
            end
            pc += kc; pb += 1
        end
    end
    return B
end
function _trmm_right!(up::Bool, tr::Bool, cj::Bool, unit::Bool, A, B)
    k = size(A, 1)
    if eltype(B) <: BlasReal && !cj && k <= _TRMM_BASE
        return k <= _TRMM_DDIRECT ? _trmm_right_base!(up, tr, cj, unit, k, A, B) :
                                    _trmm_small!(false, up, tr, unit, A, B)
    elseif _strided1(B) && eltype(B) === Float64 && !cj && k > _TRMM_RPACK
        return _trmm_packedR!(up, tr, unit, A, B, Float64)
    elseif eltype(B) <: BlasReal && !cj
        # FLAT panel loop: each _TRMM_RPANEL-column panel of B gets ONE fat off-diagonal gemm on a
        # STORED rectangular A-view (transB carries op; no materialize) + a diagonal solved by the
        # halving recursion (→ _trmm_small! bases). Big panels keep the gemms fat (skinny n=128 gemms
        # measured 0.85 at 2048); the flat level touches B only twice. Diagonal FIRST (consumes the
        # panel's ORIGINAL values), then += off-diagonal (reads other, still-original panels).
        # upM → right-to-left; lower → left-to-right.
        upM = (up != tr); P = _TRMM_RPANEL
        np = cld(k, P)
        for t in (upM ? ((np - 1):-1:0) : (0:(np - 1)))
            jc = t * P; pc = min(P, k - jc)
            Bpan = view(B, :, (jc + 1):(jc + pc))
            Adia = view(A, (jc + 1):(jc + pc), (jc + 1):(jc + pc))
            _trmm_right_recur!(up, tr, cj, unit, Adia, Bpan)
            if upM && jc > 0                     # off-diag: Bpan += B[:,1:jc]·op(A)[1:jc, jc+1:jc+pc]
                Ablk = tr ? view(A, (jc + 1):(jc + pc), 1:jc) : view(A, 1:jc, (jc + 1):(jc + pc))
                _gemm_core!(Bpan, view(B, :, 1:jc), Ablk, one(eltype(B)), one(eltype(B)),
                    false, tr, false, false)
            elseif !upM && jc + pc < k           # off-diag: Bpan += B[:,jc+pc+1:k]·op(A)[jc+pc+1:k, …]
                Ablk = tr ? view(A, (jc + 1):(jc + pc), (jc + pc + 1):k) :
                            view(A, (jc + pc + 1):k, (jc + 1):(jc + pc))
                _gemm_core!(Bpan, view(B, :, (jc + pc + 1):k), Ablk, one(eltype(B)), one(eltype(B)),
                    false, tr, false, false)
            end
        end
        return B
    elseif eltype(B) <: BlasComplex && k <= _TRMM_BASE     # complex: K-TRIM kernels (mirror side-L).
        return !_strided1(B) ? _trmm_cmplx_base_R!(up, tr, cj, unit, k, A, B) :            # strided B → base
               (_CTRMM_PACK && k >= _CTRMM_PACK_MIN) ? _trmm_cmplx_packed_R!(up, tr, cj, unit, k, A, B) :
                             _trmm_cmplx_small_R!(up, tr, cj, unit, k, A, B)    # AVX-512 / tiny-k → unpacked
    elseif k <= _TRMM_BASE                                # AD/generic: scalar column-axpy base
        return _trmm_right_base!(up, tr, cj, unit, k, A, B)
    end
    return _trmm_right_recur!(up, tr, cj, unit, A, B)
end
# Halving recursion (diagonal blocks of the flat loop + the complex/AD path).
function _trmm_right_recur!(up::Bool, tr::Bool, cj::Bool, unit::Bool, A, B)
    k = size(A, 1)
    if k <= _TRMM_BASE
        if eltype(B) <: BlasReal && !cj
            return k <= _TRMM_DDIRECT ? _trmm_right_base!(up, tr, cj, unit, k, A, B) :
                                        _trmm_small!(false, up, tr, unit, A, B)
        elseif eltype(B) <: BlasComplex
            # (the old "side-R packed regresses" note was a routing-bug artifact: the 0.24 was the scalar
            # column-axpy base @_trmm_right!, not a packed kernel — packed_R didn't exist yet.)
            return !_strided1(B) ? _trmm_cmplx_base_R!(up, tr, cj, unit, k, A, B) :
                   (_CTRMM_PACK && k >= _CTRMM_PACK_MIN) ? _trmm_cmplx_packed_R!(up, tr, cj, unit, k, A, B) :
                                 _trmm_cmplx_small_R!(up, tr, cj, unit, k, A, B)
        end
        return _trmm_right_base!(up, tr, cj, unit, k, A, B)
    end
    h = _trsplit(k)
    A11 = view(A, 1:h, 1:h); A22 = view(A, (h + 1):k, (h + 1):k)
    B1 = view(B, :, 1:h); B2 = view(B, :, (h + 1):k)
    # Mirror of left, with B column-blocks and transB carrying op(A). up≠tr → the off-diagonal feeds B2
    # (process B2's diagonal first, gemm B2+=B1·off, then B1); up==tr → feeds B1.
    if up != tr
        off = tr ? view(A, (h + 1):k, 1:h) : view(A, 1:h, (h + 1):k)
        _trmm_right_recur!(up, tr, cj, unit, A22, B2)
        _gemm_accR!(B2, B1, off, tr, cj)
        _trmm_right_recur!(up, tr, cj, unit, A11, B1)
    else
        off = tr ? view(A, 1:h, (h + 1):k) : view(A, (h + 1):k, 1:h)
        _trmm_right_recur!(up, tr, cj, unit, A11, B1)
        _gemm_accR!(B1, B2, off, tr, cj)
        _trmm_right_recur!(up, tr, cj, unit, A22, B2)
    end
    return B
end
# C += B·op(A): straight to the dispatch core (transB carries op; shapes guaranteed by the recursion).
@inline _gemm_accR!(C, Bmat, A, tr::Bool, cj::Bool) =
    _gemm_core!(C, Bmat, A, one(eltype(C)), one(eltype(C)), false, tr, false, cj)

# x := op(A)·x / op(A)⁻¹·x entry: B := α·op(A)·B (side L) or α·B·op(A) (side R), A triangular.
function _trmm!(side_left::Bool, up::Bool, tr::Bool, cj::Bool, unit::Bool, α::Number, A, B)
    if side_left
        _trmm_left!(up, tr, cj, unit, A, B)
    else
        _trmm_right!(up, tr, cj, unit, A, B)
    end
    isone(α) || _scal_all!(B, α)
    return B
end
@inline function _scal_all!(B, α)
    if _strided1(B)
        αT = convert(eltype(B), α); m = size(B, 1); n = size(B, 2); ld = stride(B, 2)
        GC.@preserve B begin
            p = pointer(B)
            if ld == m
                _scal!(m * n, αT, p, 1)                        # fully contiguous — one shot
            else                                               # padded ld: scale each column (skip the gap)
                sz = sizeof(eltype(B))
                for j in 0:(n - 1); _scal!(m, αT, p + j * ld * sz, 1); end
            end
        end
    else
        B .*= α
    end
end

# Pack a triangular op(A) panel: zero the non-stored half, write the diagonal (unit ⇒ 1). packed_upper
# = the packed op(A) is upper-triangular (zero where gi>gp). Used only for diagonal-straddling A-panels
# (off-diagonal panels are fully stored → plain _pack_A!, fully-zero panels are skipped by the driver).
function _pack_A_tri!(Ap::Vector{T}, A, ic::Int, pc::Int, mce::Int, kce::Int, tA::Bool, unit::Bool,
        packed_upper::Bool, alpha::T, mr::Int) where {T}
    if !tA && _strided1(A) && T <: BlasReal
        return _pack_A_tri_simd!(Ap, A, ic, pc, mce, kce, unit, packed_upper, alpha, mr)
    end
    np = cld(mce, mr)
    @inbounds for pi in 0:(np - 1)
        base = pi * mr * kce
        for p in 0:(kce - 1)
            for r in 0:(mr - 1)
                lr = pi * mr + r
                Ap[base + p * mr + r + 1] = if lr < mce
                    gi = ic + lr; gp = pc + p
                    if gi == gp
                        unit ? alpha : alpha * (tA ? A[gp + 1, gi + 1] : A[gi + 1, gp + 1])
                    elseif (packed_upper ? (gi > gp) : (gi < gp))
                        zero(T)
                    else
                        alpha * (tA ? A[gp + 1, gi + 1] : A[gi + 1, gp + 1])
                    end
                else
                    zero(T)
                end
            end
        end
    end
    return
end

# SIMD triangular A-pack (tA='N', dense unit-stride): per (mr-sub-panel, k-column) the stored rows are
# a prefix/suffix vs the diagonal threshold → vector load+scale + masked select (vifelse) to zero the
# rest; diagonal unit-fix is one scalar store. Lifts the straddling-panel packing to ~SIMD speed.
@inline function _pack_A_tri_simd!(Ap::Vector{T}, A, ic::Int, pc::Int, mce::Int, kce::Int, unit::Bool,
        packed_upper::Bool, alpha::T, mr::Int) where {T<:BlasReal}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T); lda = stride(A, 2); MR = mr ÷ W
    np = cld(mce, mr); lanes = Vec(ntuple(i -> i - 1, Val(W)))
    GC.@preserve A Ap begin
        Aptr = pointer(A); App = pointer(Ap); av = V(alpha); zv = zero(V)
        @inbounds for pi in 0:(np - 1)
            base = pi * mr * kce; r0g = ic + pi * mr; full = pi * mr + mr <= mce
            for p in 0:(kce - 1)
                gp = pc + p; rthr = gp - r0g; dst = App + (base + p * mr) * sz
                if full
                    src = Aptr + (r0g + gp * lda) * sz
                    for vi in 0:(MR - 1)
                        rows = lanes + vi * W
                        m = packed_upper ? (rows <= rthr) : (rows >= rthr)
                        vstore(vifelse(m, av * vload(V, src + vi * W * sz), zv), dst + vi * W * sz)
                    end
                else
                    for r in 0:(mr - 1)
                        lr = pi * mr + r; gi = ic + lr
                        Ap[base + p * mr + r + 1] = (lr < mce && (packed_upper ? gi <= gp : gi >= gp)) ?
                            alpha * A[gi + 1, gp + 1] : zero(T)
                    end
                end
                (unit && 0 <= rthr < mr && pi * mr + rthr < mce) && (Ap[base + p * mr + rthr + 1] = alpha)
            end
        end
    end
    return
end

# Single-pass packed trmm, side 'L': B := α·op(A)·B, A triangular m×m. = gemm(op(A_triangle), B) with
# A's non-stored half packed as zero (correct flops with the gemm microkernel). Per A-panel: skip
# fully-zero, plain _pack_A! fully-stored, _pack_A_tri! diagonal-straddling. Real only; α into the pack.
# IN-PLACE (no full B-copy): trmm-L columns are independent, so per jc column-panel we pack ALL of its
# pc-blocks into Bpf (capturing the input) BEFORE zeroing that panel of B — the pack itself is the copy,
# so the separate Bc scratch is gone. (Bpf holds the whole panel: nblk pc-blocks × one packed block.)
# GKH ownership: const-dispatch the gated real types (_trmm_packed! is BlasReal-only, so Float64/Float32
# are the only hot callers) → bare field load, no runtime `get!` (~130 ns) and no box signal. IdDict stays
# as the open-ended fallback only.
const _TRMM_BPF = IdDict{DataType, Vector}()
const _TRMM_BPF_F64 = Float64[]
const _TRMM_BPF_F32 = Float32[]
@inline function _trmm_bpf(::Type{Float64}, len::Int)
    length(_TRMM_BPF_F64) < len && resize!(_TRMM_BPF_F64, len)
    return _TRMM_BPF_F64
end
@inline function _trmm_bpf(::Type{Float32}, len::Int)
    length(_TRMM_BPF_F32) < len && resize!(_TRMM_BPF_F32, len)
    return _TRMM_BPF_F32
end
function _trmm_bpf(::Type{T}, len::Int) where {T}
    v = get!(() -> T[], _TRMM_BPF, T)::Vector{T}
    length(v) < len && resize!(v, len)
    return v
end
function _trmm_packed!(up::Bool, tr::Bool, unit::Bool, α::T, A, B, ::Val{MRV} = Val(_MR)) where {T<:BlasReal, MRV}
    m = size(B, 1); n = size(B, 2); W = _vwidth(T); mr = MRV * W; nr = _NR
    packed_upper = (up != tr)
    kc = min(_KC, m); mc = _at_mc_kc(_HW, T, kc, mr, cld(m, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(n, nr) * nr)
    nblk = cld(m, kc); bpf_blk = cld(nc, nr) * nr * kc          # one packed pc-block slot (padded to kc)
    Ap, _ = _gemm_scratch(T, cld(mc, mr) * mr * kc, 1)
    Bpf = _trmm_bpf(T, nblk * bpf_blk)
    ldc = stride(B, 2); sz = sizeof(T)
    GC.@preserve B Ap Bpf begin
        Cp0 = pointer(B); App = pointer(Ap); Bfp = pointer(Bpf)
        jc = 0
        while jc < n
            nce = min(nc, n - jc)
            pc = 0; pb = 0                                       # Phase 1: pack whole jc-panel of B
            while pc < m
                kce = min(kc, m - pc)
                _pack_B!(Bpf, B, pc, jc, kce, nce, false, nr, pb * bpf_blk)
                pc += kc; pb += 1
            end
            pc = 0; pb = 0                                       # Phase 2: compute from Bpf (no zero pass:
            while pc < m                                          # each tile's FIRST contribution overwrites)
                kce = min(kc, m - pc)
                ic = 0
                while ic < m
                    mce = min(mc, m - ic); a_hi = ic + mce - 1; p_hi = pc + kce - 1
                    zpanel = packed_upper ? (ic > p_hi) : (a_hi < pc)
                    if !zpanel
                        stored = packed_upper ? (a_hi < pc) : (ic > p_hi)
                        stored ? _pack_A!(Ap, A, ic, pc, mce, kce, tr, α, mr) :
                                 _pack_A_tri!(Ap, A, ic, pc, mce, kce, tr, unit, packed_upper, α, mr)
                        jr = 0
                        while jr < nce
                            nre = min(nr, nce - jr); ir = 0
                            while ir < mce
                                mre = min(mr, mce - ir); r0 = ic + ir
                                plo = stored ? 0 : (packed_upper ? max(0, r0 - pc) : 0)
                                cnt = stored ? kce : (packed_upper ? kce - plo : min(kce, r0 + mre - pc))
                                if cnt > 0
                                    # this pc-block is the tile's FIRST contribution → overwrite (β=0),
                                    # no zero pass + no C read. upper: first block = div(r0,kc); lower: pb 0.
                                    ow = packed_upper ? (pb == div(r0, kc)) : (pb == 0)
                                    Apanel = App + (div(ir, mr) * mr * kce + plo * mr) * sz
                                    Bpanel = Bfp + (pb * bpf_blk + div(jr, nr) * nr * kce + plo * nr) * sz
                                    Cblk = Ptr{T}(Cp0 + ((ic + ir) + (jc + jr) * ldc) * sz)
                                    if mre == mr && nre == nr
                                        ow ? _microkernel!(Cblk, ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), cnt, Val(MRV), Val(_NR), Val(true)) :
                                             _microkernel!(Cblk, ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), cnt, Val(MRV), Val(_NR), Val(false))
                                    else
                                        ow ? _microkernel_masked!(Cblk, ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), cnt, mre, nre, Val(MRV), Val(_NR), Val(true)) :
                                             _microkernel_masked!(Cblk, ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), cnt, mre, nre, Val(MRV), Val(_NR), Val(false))
                                    end
                                end
                                ir += mr
                            end
                            jr += nr
                        end
                    end
                    ic += mc
                end
                pc += kc; pb += 1
            end
            jc += nc
        end
    end
    return B
end

# Public: B := α·op(A)·B (side 'L') or α·B·op(A) (side 'R'); A k×k triangular (uplo/transA/diag).
function trmm!(B::AbstractMatrix, A::AbstractMatrix; side::Char = 'L', uplo::Char = 'U',
        transA::Char = 'N', diag::Char = 'N', alpha::Number = true)
    sl = side == 'L'
    k = sl ? size(B, 1) : size(B, 2)
    (size(A, 1) == size(A, 2) == k) || throw(DimensionMismatch("trmm!: A must be $k×$k"))
    # TINY real trmm: go straight to the base kernel, skipping the `_trmm!`→`_trmm_left!/_trmm_right!`
    # wrapper chain (ROADMAP: adds ~16% on a ~50 ns 8×8 op — trmm@8 0.84 via chain vs 0.999 direct). The
    # dispatch below MIRRORS the k≤_TRMM_BASE branches of `_trmm_left!`/`_trmm_right!` exactly.
    if eltype(B) <: BlasReal && transA != 'C' && k <= _TRMM_BASE
        up_ = uplo == 'U'; tr_ = transA != 'N'; unit_ = diag == 'U'
        if sl
            k <= _TRMM_DDIRECT ? _trmm_dense_L!(up_, tr_, unit_, A, B) : _trmm_small!(true, up_, tr_, unit_, A, B)
        else
            k <= _TRMM_DDIRECT ? _trmm_right_base!(up_, tr_, false, unit_, k, A, B) : _trmm_small!(false, up_, tr_, unit_, A, B)
        end
        isone(alpha) || _scal_all!(B, convert(eltype(B), alpha))
    elseif eltype(B) <: BlasComplex && k <= _TRMM_BASE   # tiny complex: same skip (mirrors _trmm_left!/_right! complex base)
        up_ = uplo == 'U'; tr_ = transA != 'N'; cj_ = transA == 'C'; unit_ = diag == 'U'
        if sl
            !_strided1(B) ? _trmm_cmplx_base_L!(up_, tr_, cj_, unit_, k, A, B) :
            (_CTRMM_PACK && k >= _CTRMM_PACK_MIN) ? _trmm_cmplx_packed_L!(up_, tr_, cj_, unit_, k, A, B) :
            _trmm_cmplx_small_L!(up_, tr_, cj_, unit_, k, A, B)
        else
            !_strided1(B) ? _trmm_cmplx_base_R!(up_, tr_, cj_, unit_, k, A, B) :
            (_CTRMM_PACK && k >= _CTRMM_PACK_MIN) ? _trmm_cmplx_packed_R!(up_, tr_, cj_, unit_, k, A, B) :
            _trmm_cmplx_small_R!(up_, tr_, cj_, unit_, k, A, B)
        end
        isone(alpha) || _scal_all!(B, convert(eltype(B), alpha))
    # side-L real large → K-range-trimmed single-pass packed (the straddling tile contracts only its
    # nonzero p-band, not the full kc zero-band — that band was the ~kc/k waste that capped the naive
    # packed trmm). Else (side R, complex/AD, small) → recursion-over-gemm! (no regression).
    elseif sl && eltype(B) <: BlasReal && transA != 'C' && k > _GEMM_UNPACK_MAX
        # 8×8 tile (Val(1), unified W==_NR): finer K-trim staircase + smaller within-tile zero triangle;
        # the proven-fastest, most consistent path across sizes. (A 16×8 bulk helped N-cases at large k
        # but regressed k=768 and the public po2 A-pad path — non-robust, not worth the split.)
        mrv = _unified_ok(eltype(B)) ? Val(1) : Val(_MR)
        # NOTE: no A-pad here (unlike trsm). trmm's po2-ld conflict is mild (~2%); the O(k²) A-copy to
        # pad it costs about the same, so padding is net-negative for trmm — measured. (trsm's conflict
        # was catastrophic 0.78→1.12, there the copy pays.)
        _trmm_packed!(uplo == 'U', transA != 'N', diag == 'U', convert(eltype(B), alpha), A, B, mrv)
    else
        _trmm!(sl, uplo == 'U', transA != 'N', transA == 'C', diag == 'U', alpha, A, B)
    end
    return B
end

# ──────────────────────────────────────────────────────────────────────────────────────────────
# trsm: B := α·op(A)⁻¹·B (side 'L') / α·B·op(A)⁻¹ (side 'R'), A triangular. Same recursive blocking
# as trmm, but: (1) solve the independent block FIRST, (2) the off-diagonal update SUBTRACTS the
# already-solved block (gemm α=-1,β=1), (3) the base is a triangular solve (trsv per column / column
# substitution). α is applied to B up front (B := α·op(A)⁻¹·B = op(A)⁻¹·(αB)).
@inline _gemm_sub!(C, A, B, tr::Bool, cj::Bool) =                                    # C -= op(A)·B
    _gemm_core!(C, A, B, -one(eltype(C)), one(eltype(C)), tr, false, cj, false)
@inline _gemm_subR!(C, Bmat, A, tr::Bool, cj::Bool) =                                # C -= B·op(A)
    _gemm_core!(C, Bmat, A, -one(eltype(C)), one(eltype(C)), false, tr, false, cj)

# trsm base via small triangular INVERSE + gemm (BLIS-style): a block ≤ _TRSM_BASE is solved by
# inverting its NB×NB triangle once (O(NB³/6), tiny) then applying op(inv) as a gemm — so the diagonal
# solve runs at gemm speed instead of scalar back-substitution. The recursion's off-diagonal updates
# are already gemm!. Real only (stability fine for the well-conditioned diagonal blocks trsm assumes);
# complex/conj keep the scalar trsv base.
const _TRSM_BASE = 32
# Small real triangular inverse: V (same uplo as A) = inv(A). Cast as a trsm: V solves A·V = I, so
# V := A⁻¹·I via the vectorized dense-L base (contiguous A-column axpys) instead of a scalar
# strided-row dot — the scalar version was ~20× less efficient/flop and 44% of the invL base.
# I is the identity (also zeroes the non-stored half; forward/back-substitution keeps it zero).
# Always plain inv (tr=false): the invL/invR base applies any transpose at its gemm stage.
# Blocked triangular inverse V = A⁻¹ (V same uplo as A). Split A into 2×2 blocks and combine via the
# (now clipped, fast) gemm instead of the O(nb³) scalar forward-substitution over the identity — which the
# ceiling test showed to be ~20% of the invL leaf at n≈96. Lower (up=false): V21 = -V22·A21·V11; upper:
# V12 = -V11·A12·V22; the opposite off-block is zeroed so V stays triangular (the invL base reads V dense).
# Base blocks (≤ _TRTRI_BASE) use the identity-RHS dense solve. Diagonal blocks recurse with the same
# uplo/unit; the off-diagonal block carries its actual (non-unit) values.
const _TRTRI_BASE = 16
function _trtri!(V, A, nb::Int, up::Bool, unit::Bool)
    T = eltype(V)
    if nb <= _TRTRI_BASE
        fill!(V, zero(T))
        @inbounds for i in 1:nb; V[i, i] = one(T); end
        _trsm_dense_L!(up, false, unit, A, V)
        return V
    end
    h = nb ÷ 2; m = nb - h
    A11 = view(A, 1:h, 1:h); A22 = view(A, (h + 1):nb, (h + 1):nb)
    V11 = view(V, 1:h, 1:h); V22 = view(V, (h + 1):nb, (h + 1):nb)
    _trtri!(V11, A11, h, up, unit)
    _trtri!(V22, A22, m, up, unit)
    if up
        A12 = view(A, 1:h, (h + 1):nb); V12 = view(V, 1:h, (h + 1):nb)
        tmp = _trtri_tmp(T, h, m)
        gemm!(tmp, A12, V22; alpha = true, beta = false)          # tmp = A12·V22   (h×m)
        gemm!(V12, V11, tmp; alpha = -one(T), beta = false)       # V12 = -V11·tmp
        fill!(view(V, (h + 1):nb, 1:h), zero(T))                  # strict-lower stays 0
    else
        A21 = view(A, (h + 1):nb, 1:h); V21 = view(V, (h + 1):nb, 1:h)
        tmp = _trtri_tmp(T, m, h)
        gemm!(tmp, A21, V11; alpha = true, beta = false)          # tmp = A21·V11   (m×h)
        gemm!(V21, V22, tmp; alpha = -one(T), beta = false)       # V21 = -V22·tmp
        fill!(view(V, 1:h, (h + 1):nb), zero(T))                  # strict-upper stays 0
    end
    return V
end
# _trsm_tmp (invL/invR copyback temp) lives in the per-type L3Workspace (see src/workspace.jl).
# side L base: B := op(A)⁻¹·B = op(inv(A))·B (gemm with transA=op into temp, copy back).
function _trsm_base_invL!(up::Bool, tr::Bool, unit::Bool, A, B)
    nb = size(A, 1); n = size(B, 2); T = eltype(B)
    iv = view(_l3_tmp(T), 1:nb, 1:nb); _trtri!(iv, A, nb, up, unit)
    tmp = view(_trsm_tmp(T, nb, n), 1:nb, 1:n)
    # tmp := op(iv)·B. The leaf shape is skewed (nb ≤ _TRSM_BASE tiny, n wide) — the UNPACKED path (no
    # B-pack, no scaleC zero-pass, Val{B0}=overwrite) beats the packed gemm here (measured 0.72× its time
    # at nb=32,n=256; the k=nb pack traffic ≈ the compute). tr='T' needs iv transposed → keep packed gemm.
    if tr
        gemm!(tmp, iv, B; alpha = true, beta = false, transA = 'T')
    else
        _gemm_unpacked!(Val(false), Val(true), nb, n, nb, one(T), iv, B, zero(T), tmp)
    end
    copyto!(B, tmp); return B
end
# side R base: B := B·op(A)⁻¹ = B·op(inv(A)). tmp := B·op(iv) via the unpacked path (transB=op is a free
# Val{TB}; skewed shape m wide, n=k=nb tiny → same unpacked win as invL).
function _trsm_base_invR!(up::Bool, tr::Bool, unit::Bool, A, B)
    nb = size(A, 1); m = size(B, 1); T = eltype(B)
    iv = view(_l3_tmp(T), 1:nb, 1:nb); _trtri!(iv, A, nb, up, unit)
    tmp = view(_trsm_tmp(T, m, nb), 1:m, 1:nb)
    # branch on tr so Val{TB} is a literal (Val(tr) with runtime tr is a runtime dispatch — StrictMode
    # @typestable catches it; the dynamic call also boxes the Val, so this branch is faster too).
    if tr
        _gemm_unpacked!(Val(true), Val(true), m, nb, nb, one(T), B, iv, zero(T), tmp)
    else
        _gemm_unpacked!(Val(false), Val(true), m, nb, nb, one(T), B, iv, zero(T), tmp)
    end
    copyto!(B, tmp); return B
end

# Direct triangular solve base (side L): rank-1 substitution, the eliminate-rows axpy dispatched to the gated
# 4-way-unrolled `_axpy_simd!` (no-trans; trans strided → scalar). n³/2 flops (half of invert+gemm), no gemm
# dispatch. Real non-conj; forward when up==tr. Used as the base ONLY when B is narrow — the per-column axpy
# count grows with n, so for wide B the invL/gemm base wins (routed by _TRSM_NCUT below).
const _TRSM_NCUT = 64          # side-L: B width cut (invL wins from 96 down since the gemm clip; 64 keeps dense only for n≤64)
const _TRSM_NCUT_R = 128       # side-R: B height cut (R's narrow path is stronger than L's — measured, 128 rides it at 1.7×)
# Narrow-B dense-base cutoff. Re-swept at LOCKED CPU freq (2026-07-02): 32 beats 16 (n=32 cold
# 0.565→0.75, worst-size = the gate metric); the old "16, raising hurts n=128" was a boost-noise artifact
# (benchmark with CPU boost OFF). ponytail: could be a Preferences knob if the fleet diverges.
const _TRSM_DBASE = 32
# Column-blocked rank-1 update for the dense trsm base (non-transpose): B[brow0.., c] -= B[irow0, c]·acol
# over all n columns. Holds the A-column vector across a block of 4 B-columns (reuse) and does the short
# rlen-remainder mask ONCE per row-block instead of once per column — the per-column `_axpy_simd!` (though
# inlined) repeated both. `a` points at A[rs,i]; brow0/irow0 are 0-based B rows.
@inline function _trsm_col_r1!(::Type{T}, rlen::Int, a::Ptr{T}, pB::Ptr{T}, irow0::Int, brow0::Int,
        n::Int, ldb::Int) where {T<:BlasReal}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T); lanes = Vec{W, Int}(ntuple(q -> q - 1, Val(W)))
    rfull = (rlen ÷ W) * W; msk = (lanes + rfull) < rlen
    @inbounds begin
        c = 0
        while c + 4 <= n
            s0 = V(-unsafe_load(pB, irow0 + c * ldb + 1)); s1 = V(-unsafe_load(pB, irow0 + (c + 1) * ldb + 1))
            s2 = V(-unsafe_load(pB, irow0 + (c + 2) * ldb + 1)); s3 = V(-unsafe_load(pB, irow0 + (c + 3) * ldb + 1))
            b0 = pB + (brow0 + c * ldb) * sz; b1 = pB + (brow0 + (c + 1) * ldb) * sz
            b2 = pB + (brow0 + (c + 2) * ldb) * sz; b3 = pB + (brow0 + (c + 3) * ldb) * sz
            r = 0
            while r < rfull
                av = vload(V, a + r * sz)
                q = b0 + r * sz; vstore(muladd(av, s0, vload(V, q)), q)
                q = b1 + r * sz; vstore(muladd(av, s1, vload(V, q)), q)
                q = b2 + r * sz; vstore(muladd(av, s2, vload(V, q)), q)
                q = b3 + r * sz; vstore(muladd(av, s3, vload(V, q)), q); r += W
            end
            if rfull < rlen
                av = vload(V, a + rfull * sz, msk)
                q = b0 + rfull * sz; vstore(muladd(av, s0, vload(V, q, msk)), q, msk)
                q = b1 + rfull * sz; vstore(muladd(av, s1, vload(V, q, msk)), q, msk)
                q = b2 + rfull * sz; vstore(muladd(av, s2, vload(V, q, msk)), q, msk)
                q = b3 + rfull * sz; vstore(muladd(av, s3, vload(V, q, msk)), q, msk)
            end
            c += 4
        end
        while c < n
            s0 = V(-unsafe_load(pB, irow0 + c * ldb + 1)); b0 = pB + (brow0 + c * ldb) * sz
            r = 0
            while r < rfull
                q = b0 + r * sz; vstore(muladd(vload(V, a + r * sz), s0, vload(V, q)), q); r += W
            end
            if rfull < rlen
                q = b0 + rfull * sz; vstore(muladd(vload(V, a + rfull * sz, msk), s0, vload(V, q, msk)), q, msk)
            end
            c += 1
        end
    end
    return
end
# Register-tiled trsm-L base for the no-trans f64 case (op(A)=A), GENERAL up/unit: solve A·X=B, A k×k lower
# (fwd) or upper (bwd), unit or non-unit diagonal. W-row blocks — downdate each block against the ALREADY-
# SOLVED rows (vectorized; the 4-B-column unroll reuses the A row-block vector), then a scalar W×W diagonal
# solve. Touches each B element ~once vs the dense base's ~k passes (store-bound BLAS-2). Bit-identical.
# Measured galen: lower-unit (getrf) +119–151%; upper-non-unit (trsm gate) +44–47%. pL/pB = &·[1,1], ld col.
@inline function _trsm_tile_L_f64!(up::Bool, unit::Bool, pL::Ptr{Float64}, ld0::Int, pB::Ptr{Float64}, ldb::Int, k::Int, n::Int)
    W = _CHOLW; nb = k ÷ W
    @inline function doblock(rb)
        solved = up ? ((rb + W):k) : (1:(rb - 1))            # already-solved rows: below (upper) / above (lower)
        c = 1
        @inbounds while c + 3 <= n
            a0 = vload(_CVF, _cvptr(pB, rb, c, ldb));     a1 = vload(_CVF, _cvptr(pB, rb, c + 1, ldb))
            a2 = vload(_CVF, _cvptr(pB, rb, c + 2, ldb)); a3 = vload(_CVF, _cvptr(pB, rb, c + 3, ldb))
            for j in solved
                Lv = vload(_CVF, _cvptr(pL, rb, j, ld0))
                a0 = muladd(_CVF(-unsafe_load(pB, _clidx(j, c, ldb))),     Lv, a0)
                a1 = muladd(_CVF(-unsafe_load(pB, _clidx(j, c + 1, ldb))), Lv, a1)
                a2 = muladd(_CVF(-unsafe_load(pB, _clidx(j, c + 2, ldb))), Lv, a2)
                a3 = muladd(_CVF(-unsafe_load(pB, _clidx(j, c + 3, ldb))), Lv, a3)
            end
            vstore(a0, _cvptr(pB, rb, c, ldb));     vstore(a1, _cvptr(pB, rb, c + 1, ldb))
            vstore(a2, _cvptr(pB, rb, c + 2, ldb)); vstore(a3, _cvptr(pB, rb, c + 3, ldb))
            c += 4
        end
        @inbounds while c <= n
            a = vload(_CVF, _cvptr(pB, rb, c, ldb))
            for j in solved; a = muladd(_CVF(-unsafe_load(pB, _clidx(j, c, ldb))), vload(_CVF, _cvptr(pL, rb, j, ld0)), a); end
            vstore(a, _cvptr(pB, rb, c, ldb)); c += 1
        end
        # W diagonal reciprocals ONCE per block (not per column — the redundant per-cc inv() was a division
        # chain that sank non-unit on wide vectors). Val(W) ⇒ const-length tuple that const-folds.
        recips = unit ? ntuple(_ -> 1.0, Val(W)) :
                 ntuple(q -> @inbounds(inv(unsafe_load(pL, _clidx(rb + q - 1, rb + q - 1, ld0)))), Val(W))
        @inbounds for cc in 1:n                             # scalar W×W in-block diagonal solve
            for ii in (up ? (W-1:-1:0) : (0:W-1))
                s = unsafe_load(pB, _clidx(rb + ii, cc, ldb))
                for jj in (up ? (ii+1:W-1) : (0:ii-1)); s = muladd(-unsafe_load(pL, _clidx(rb + ii, rb + jj, ld0)), unsafe_load(pB, _clidx(rb + jj, cc, ldb)), s); end
                unit || (s *= recips[ii + 1])
                unsafe_store!(pB, s, _clidx(rb + ii, cc, ldb))
            end
        end
    end
    @inline function dorow(i)                               # one tail row (k not a multiple of W)
        rng = up ? (i+1:k) : (1:i-1)
        @inbounds for cc in 1:n
            s = unsafe_load(pB, _clidx(i, cc, ldb))
            for j in rng; s = muladd(-unsafe_load(pL, _clidx(i, j, ld0)), unsafe_load(pB, _clidx(j, cc, ldb)), s); end
            unit || (s *= inv(unsafe_load(pL, _clidx(i, i, ld0))))
            unsafe_store!(pB, s, _clidx(i, cc, ldb))
        end
    end
    # ORDER: solve in the substitution direction. Upper=backward ⇒ tail rows (bottom) FIRST, then blocks
    # bottom-up (they downdate against the tail as "solved"). Lower=forward ⇒ blocks top-down, then tail.
    if up
        for i in (nb*W == k ? (0:-1) : (k:-1:nb*W+1)); dorow(i); end
        for bi in nb:-1:1; doblock((bi - 1) * W + 1); end
    else
        for bi in 1:nb; doblock((bi - 1) * W + 1); end
        for i in (nb*W+1):k; dorow(i); end
    end
    return nothing
end

function _trsm_dense_L!(up::Bool, tr::Bool, unit::Bool, A, B)
    k = size(A, 1); n = size(B, 2); T = eltype(B); sz = sizeof(T)
    lda = stride(A, 2); ldb = stride(B, 2); fwd = (up == tr)
    # Tile crossover (DERIVED, req#8): tile trades dense's ~k store-passes (∝ k²·n/W) for a per-block scalar
    # W×W triangular diagonal solve (∝ k·W·n, depth-W latency chain). Net win ⇒ k·W < k²/W ⇒ k > W². Fleet-
    # validated: galen(W=4,W²=16) wins from k=32, wintermute(W=8,W²=64) from k=96. (Side-R tiles unconditionally
    # — it vectorizes its in-block solve over m, no W² term.) `_CHOLW*_CHOLW` const-folds at compile time.
    if !tr && T === Float64 && A isa StridedMatrix && B isa StridedMatrix &&
            stride(A, 1) == 1 && stride(B, 1) == 1 && k > _CHOLW * _CHOLW   # no-trans strided f64 → tile
        GC.@preserve A B _trsm_tile_L_f64!(up, unit, pointer(A), lda, pointer(B), ldb, k, n)
        return B
    end
    GC.@preserve A B begin
        pA = pointer(A); pB = pointer(B)
        @inbounds for i in (fwd ? (1:k) : (k:-1:1))
            if !unit
                d = inv(A[i, i]); for c in 1:n; B[i, c] *= d; end
            end
            rlen = fwd ? (k - i) : (i - 1); rlen == 0 && continue
            rs = fwd ? (i + 1) : 1
            if tr
                rows = fwd ? ((i + 1):k) : (1:(i - 1))
                for c in 1:n; bic = B[i, c]; @simd for r in rows; B[r, c] -= A[i, r] * bic; end; end
            elseif T <: BlasReal
                aptr = pA + ((i - 1) * lda + (rs - 1)) * sz
                _trsm_col_r1!(T, rlen, Ptr{T}(aptr), pB, i - 1, rs - 1, n, ldb)
            else                                             # complex/generic column rank-1 (trtri base ≤16)
                rows = fwd ? ((i + 1):k) : (1:(i - 1))
                for c in 1:n; bic = B[i, c]; @simd for r in rows; B[r, c] -= A[r, i] * bic; end; end
            end
        end
    end
    return B
end
# Complex trsm side-L base: invert op(A)'s k×k triangle ONCE (generic _trtri! → M⁻¹, reads A once) then
# B := op(M⁻¹)·B via the gating SIMD complex gemm — vs trsv-per-column re-reading A n times. (op(A)⁻¹ =
# op(A⁻¹): the gemm carries the trans/conj on the inverse.) The complex analog of the real invL leaf.
function _trsm_cmplx_base_L!(up::Bool, tr::Bool, cj::Bool, unit::Bool, k::Int, A, B)
    T = eltype(B); n = size(B, 2)
    V = _l3_tmp(T); Vv = view(V, 1:k, 1:k)
    _trtri!(Vv, A, k, up, unit)                                      # Vv = A⁻¹ (as-stored, non-conj)
    Bt = _trsm_tmp(T, k, n); Btv = view(Bt, 1:k, 1:n)
    _gemm_core!(Btv, Vv, B, one(T), zero(T), tr, false, cj, false)   # Btv = op(A⁻¹)·B
    copyto!(B, Btv)
    return B
end

# Complex trsm side-R base: B := B·op(A)⁻¹ = B·op(A⁻¹). Invert once (_trtri!), then one SIMD complex gemm.
function _trsm_cmplx_base_R!(up::Bool, tr::Bool, cj::Bool, unit::Bool, k::Int, A, B)
    T = eltype(B); m = size(B, 1)
    V = _l3_tmp(T); Vv = view(V, 1:k, 1:k)
    _trtri!(Vv, A, k, up, unit)
    Bt = _trsm_tmp(T, m, k); Btv = view(Bt, 1:m, 1:k)
    _gemm_core!(Btv, B, Vv, one(T), zero(T), false, tr, false, cj)   # Btv = B·op(A⁻¹)
    copyto!(B, Btv)
    return B
end

# DIRECT j-outer trsm-L (op=A, no-trans): process each A COLUMN j once (read A once, not per-RHS like
# per-column trsv), scaling B's row j by the precomputed diagonal reciprocal, then a CONTIGUOUS column
# axpy B[·,c] -= x·A[·,j] across every RHS c. Diagonal reciprocals precomputed off the loop (as in trsv).
# This is OB's structure — no trtri, no extra flops. Replaces trtri+trmm for small/mid-n where the
# trtri overhead sank ztrsm-L (n=8–128 was 0.55–0.80). k ≤ _TRMM_BASE (128); reuses _TRSV_RCP.
# One RHS panel (pw columns), j-outer solve, panel kept L1-resident. @inline so A's column pointer and the
# reciprocal table stay register/L1-resident across the panel — fuses the per-column work (the key to
# mid-n: the UNBLOCKED solve streamed the whole B out of L1 k times, ~O(k²·nrhs) L2 traffic; blocking the
# RHS into L1-fitting panels keeps each panel hot so only A is re-read).
@inline function _dLN_panel!(up::Bool, unit::Bool, k::Int, rcp, Ap::Ptr{Tc}, Bp::Ptr{Tc},
        pw::Int, lda::Int, ldb::Int, csz::Int) where {Tc}
    @inbounds for j in (up ? (k:-1:1) : (1:k))
        aj = Ap + (j - 1) * lda * csz                            # &A[1,j] (Julia Ptr+int = BYTES)
        for c in 0:(pw - 1)
            bc = Bp + c * ldb * csz                              # &B[1, panel-col c]
            xj = unit ? unsafe_load(bc, j) : unsafe_load(bc, j) * rcp[j]
            unit || unsafe_store!(bc, xj, j)
            if up
                j > 1 && _axpy_cmplx_simd!(j - 1, -real(xj), -imag(xj), aj, bc)          # B[1:j-1,c] -= xj·A[1:j-1,j]
            else
                j < k && _axpy_cmplx_simd!(k - j, -real(xj), -imag(xj), aj + j * csz, bc + j * csz)  # B[j+1:k,c]
            end
        end
    end
    return
end
function _trsm_cmplx_dLN!(up::Bool, unit::Bool, k::Int, A, B)
    Tc = eltype(B); T = real(Tc); nrhs = size(B, 2); csz = sizeof(Tc)
    lda = stride(A, 2); ldb = stride(B, 2); rcp = _trsv_rcpbuf(T)
    nc = clamp((_L1_BYTES ÷ 2) ÷ (k * csz), 1, nrhs)             # RHS panel fitting ~½ L1 (A col shares it)
    GC.@preserve A B begin
        Ap = Ptr{Tc}(pointer(A)); Bp = Ptr{Tc}(pointer(B))
        unit || @inbounds @simd for j in 1:k; rcp[j] = _crecip(unsafe_load(Ap, (j - 1) * lda + j)); end
        pc = 0
        while pc < nrhs
            pw = min(nc, nrhs - pc)
            _dLN_panel!(up, unit, k, rcp, Ap, Bp + pc * ldb * csz, pw, lda, ldb, csz)
            pc += nc
        end
    end
    return B
end
# n above which trsm-L inverts (trtri) + K-TRIM trmm-on-inverse. At/below it (N case), the direct j-outer
# solve above; the trtri overhead + extra flops sank small/mid-n. Per-box knob.
const _CTRSM_DIRECT_MAX = @load_preference("ctrsm_direct_max", 64)::Int
# Complex trsm-L recursion base for NARROW B (nrhs ≤ _CTRSM_NCUT): blocks > this SPLIT (row-halve + gemm
# off-diagonal update, OB's structure); ≤ this bottom out in a small j-outer base. Monolithic j-outer caps
# ~0.85 at n=128; recursing into small bases + gemm subtracts recovers the blocking (rec=64 → 0.91).
# Wide B keeps the trtri-on-inverse base (_TRMM_BASE) — its invert is amortized by the big gemm. Per-box knob.
const _CTRSM_REC_L = @load_preference("ctrsm_rec_l", 64)::Int
const _CTRSM_NCUT = @load_preference("ctrsm_ncut", 128)::Int   # B-width cut: ≤ → narrow (j-outer recursion)
# Complex trsm K-TRIM: op(A)⁻¹ = op(A⁻¹), A⁻¹ triangular → reuse the trmm K-TRIM kernel on the inverse at
# half the flops (large-n / trans). Small-n N → direct j-outer solve (no trtri; OB's approach).
function _trsm_cmplx_small_L!(up::Bool, tr::Bool, cj::Bool, unit::Bool, k::Int, A, B)
    if !tr && k <= _CTRSM_DIRECT_MAX && _strided1(B)                 # direct back-substitution (no trtri)
        return _trsm_cmplx_dLN!(up, unit, k, A, B)
    end
    T = eltype(B); Vv = view(_trsm_tmp(T, k, k), 1:k, 1:k)
    _trtri!(Vv, A, k, up, unit)                                      # Vv = A⁻¹ (as-stored, non-conj)
    return (_CTRMM_PACK && k >= _CTRMM_PACK_MIN) ? _trmm_cmplx_packed_L!(up, tr, cj, false, k, Vv, B) :
                                                   _trmm_cmplx_small_L!(up, tr, cj, false, k, Vv, B)
end
# Direct complex side-R column-substitution base (no trtri): X·op(A)=B in place, !tr (⟹ !cj), unit or
# non-unit, A k×k upper/lower. Ascending columns when up (up≠tr, tr=false). The side-R mirror of
# _trsm_cmplx_dLN! and the complex sibling of _trsm_dense_R! (same loop, complex SIMD kernels). Beats OB
# for k≤64 (trtri-free) where the invert+trmm base's trtri is 40–66% exposed overhead (measured, galen).
function _trsm_cmplx_dRN!(up::Bool, unit::Bool, k::Int, A, B)
    m = size(B, 1); T = eltype(B); csz = sizeof(T); ldb = stride(B, 2); lda = stride(A, 2)
    GC.@preserve A B begin
        pB = pointer(B); pA = Ptr{T}(pointer(A))
        @inbounds for j in (up ? (1:k) : (k:-1:1))
            pj = pB + (j - 1) * ldb * csz
            for i in (up ? (1:(j - 1)) : ((j + 1):k))
                c = unsafe_load(pA, (j - 1) * lda + i)                                     # A[i,j]
                (real(c) == 0 && imag(c) == 0) ||
                    _axpy_cmplx_simd!(m, -real(c), -imag(c), pB + (i - 1) * ldb * csz, pj)  # X[:,j] -= A[i,j]·X[:,i]
            end
            unit || (r = _crecip(unsafe_load(pA, (j - 1) * lda + j)); _scal_cmplx_simd!(m, real(r), imag(r), pj))
        end
    end
    return B
end
# Direct complex side-R column-substitution base for transA='C' (no trtri): X·Aᴴ = B in place. Conjugate-
# transpose sibling of _trsm_cmplx_dRN!: coef(i,j)=conj(A[j,i]) (row j of A, conjugated), diagonal
# 1/conj(A[j,j]); order is up≠tr with tr=true ⇒ up ? descending : ascending (mirror of _trsm_right_base!).
# Fixes the ztrsmR-C collapse (0.53–0.88 at all n) that the trtri+K-TRIM base caused — the exact path
# zpotrf lower recurses through (side='R', transA='C'), so it was dragging zpotrf n≥128.
function _trsm_cmplx_dRC!(up::Bool, unit::Bool, k::Int, A, B)
    m = size(B, 1); T = eltype(B); csz = sizeof(T); ldb = stride(B, 2); lda = stride(A, 2)
    GC.@preserve A B begin
        pB = pointer(B); pA = Ptr{T}(pointer(A))
        @inbounds for j in (up ? (k:-1:1) : (1:k))
            pj = pB + (j - 1) * ldb * csz
            for i in (up ? ((j + 1):k) : (1:(j - 1)))
                c = conj(unsafe_load(pA, (i - 1) * lda + j))                                # conj(A[j,i])
                (real(c) == 0 && imag(c) == 0) ||
                    _axpy_cmplx_simd!(m, -real(c), -imag(c), pB + (i - 1) * ldb * csz, pj)   # X[:,j] -= conj(A[j,i])·X[:,i]
            end
            unit || (r = _crecip(conj(unsafe_load(pA, (j - 1) * lda + j))); _scal_cmplx_simd!(m, real(r), imag(r), pj))
        end
    end
    return B
end
function _trsm_cmplx_small_R!(up::Bool, tr::Bool, cj::Bool, unit::Bool, k::Int, A, B)
    if !tr && k <= _CTRSM_DIRECT_MAX && _strided1(B)                 # transA='N' direct column-substitution
        return _trsm_cmplx_dRN!(up, unit, k, A, B)
    elseif tr && cj && k <= _CTRSM_DIRECT_MAX && _strided1(B)        # transA='C' direct (no trtri)
        return _trsm_cmplx_dRC!(up, unit, k, A, B)
    end
    T = eltype(B); Vv = view(_trsm_tmp(T, k, k), 1:k, 1:k)
    _trtri!(Vv, A, k, up, unit)
    return (_CTRMM_PACK && k >= _CTRMM_PACK_MIN) ? _trmm_cmplx_packed_R!(up, tr, cj, false, k, Vv, B) :
                                                   _trmm_cmplx_small_R!(up, tr, cj, false, k, Vv, B)
end

# side 'L': B := op(A)⁻¹·B, A k×k (k=size(B,1)), unscaled.
function _trsm_left!(up::Bool, tr::Bool, cj::Bool, unit::Bool, A, B)
    k = size(A, 1)
    if eltype(B) <: BlasReal && !cj
        # narrow B → dense base (few axpy calls); wide B → invL base (gemm-efficient). n is invariant under
        # the side-L row split, so the choice is consistent through the recursion.
        if size(B, 2) <= _TRSM_NCUT
            k <= _TRSM_DBASE && return _trsm_dense_L!(up, tr, unit, A, B)
        elseif k <= _TRSM_BASE
            return _trsm_base_invL!(up, tr, unit, A, B)
        end
    elseif eltype(B) <: BlasComplex                       # complex base (else fall through → gemm-blocked split)
        # nrhs is invariant under the row-split → decide the base once. Wide B: trtri-on-inverse base (its
        # O(k³) invert is amortized by the big off-diagonal gemm). Narrow B (standalone 96/128): trtri
        # overhead is exposed, so recurse into small j-outer bases + gemm subtract (OB's structure).
        recbase = size(B, 2) <= _CTRSM_NCUT ? _CTRSM_REC_L : _TRMM_BASE
        if k <= recbase
            return _strided1(B) ? _trsm_cmplx_small_L!(up, tr, cj, unit, k, A, B) :
                                  _trsm_cmplx_base_L!(up, tr, cj, unit, k, A, B)
        end
    elseif k <= _TRMM_BASE                                 # AD/generic: trsv per column
        @inbounds for c in axes(B, 2)
            _trsv!(up, tr, cj, unit, k, A, view(B, :, c), 1)
        end
        return B
    end
    h = _trsplit(k)
    A11 = view(A, 1:h, 1:h); A22 = view(A, (h + 1):k, (h + 1):k)
    B1 = view(B, 1:h, :); B2 = view(B, (h + 1):k, :)
    if up != tr                                       # solve B2, then B1 -= off·B2, then solve B1
        off = tr ? view(A, (h + 1):k, 1:h) : view(A, 1:h, (h + 1):k)
        _trsm_left!(up, tr, cj, unit, A22, B2)
        _gemm_sub!(B1, off, B2, tr, cj)
        _trsm_left!(up, tr, cj, unit, A11, B1)
    else                                              # solve B1, then B2 -= off·B1, then solve B2
        off = tr ? view(A, 1:h, (h + 1):k) : view(A, (h + 1):k, 1:h)
        _trsm_left!(up, tr, cj, unit, A11, B1)
        _gemm_sub!(B2, off, B1, tr, cj)
        _trsm_left!(up, tr, cj, unit, A22, B2)
    end
    return B
end

# side 'R' base: X·op(A)=B by column substitution. up≠tr ⇒ ascending (feeds lower-index columns);
# up==tr ⇒ descending. Subtract solved columns, then divide by the diagonal (unless unit).
function _trsm_right_base!(up::Bool, tr::Bool, cj::Bool, unit::Bool, k::Int, A, B)
    m = size(B, 1)
    coef(i, j) = tr ? (cj ? conj(A[j, i]) : A[j, i]) : A[i, j]
    @inbounds if up != tr
        for j in 1:k
            for i in 1:(j - 1); _axpy_col!(B, j, -coef(i, j), i, m); end
            unit || _scal_col!(B, j, inv(coef(j, j)), m)
        end
    else
        for j in k:-1:1
            for i in (j + 1):k; _axpy_col!(B, j, -coef(i, j), i, m); end
            unit || _scal_col!(B, j, inv(coef(j, j)), m)
        end
    end
    return B
end

# Register-tiled trsm-R base (f64, GENERAL up/tr/unit): solve X·op(A)=B, X/B m×k, vectorized over m (B rows,
# contiguous). Mirror of the side-L tile with rows↔solve-columns swapped: block NC=4 SOLVE-COLUMNS, downdate
# the block against the ALREADY-SOLVED columns in one W-wide sweep (reuse each solved column's m-vector across
# the 4 block-cols), then a scalar in-block NC×NC coupling solve. coef(j,l)=A[j,l] (tr) or A[l,j]. Each B
# element written ~once vs the dense base's ~k passes. Measured galen (trsmR gate, lower-T): +64–113%. Bit-id.
@inline function _trsm_tile_R_f64!(up::Bool, tr::Bool, unit::Bool, pA::Ptr{Float64}, lda::Int, pB::Ptr{Float64}, ldb::Int, m::Int, k::Int)
    W = _CHOLW; NC = 4; nb = k ÷ NC; asc = (up != tr)
    @inline cf(j, l) = tr ? unsafe_load(pA, _clidx(j, l, lda)) : unsafe_load(pA, _clidx(l, j, lda))
    @inline function doblock(j0)                             # block solve-cols j0..j0+3
        solved = asc ? (1:(j0 - 1)) : ((j0 + NC):k)
        i = 1
        @inbounds while i + W - 1 <= m                       # vectorized downdate of the 4 block-cols over m
            a0 = vload(_CVF, _cvptr(pB, i, j0, ldb));     a1 = vload(_CVF, _cvptr(pB, i, j0 + 1, ldb))
            a2 = vload(_CVF, _cvptr(pB, i, j0 + 2, ldb)); a3 = vload(_CVF, _cvptr(pB, i, j0 + 3, ldb))
            for l in solved
                xv = vload(_CVF, _cvptr(pB, i, l, ldb))
                a0 = muladd(_CVF(-cf(j0, l)),     xv, a0); a1 = muladd(_CVF(-cf(j0 + 1, l)), xv, a1)
                a2 = muladd(_CVF(-cf(j0 + 2, l)), xv, a2); a3 = muladd(_CVF(-cf(j0 + 3, l)), xv, a3)
            end
            vstore(a0, _cvptr(pB, i, j0, ldb));     vstore(a1, _cvptr(pB, i, j0 + 1, ldb))
            vstore(a2, _cvptr(pB, i, j0 + 2, ldb)); vstore(a3, _cvptr(pB, i, j0 + 3, ldb)); i += W
        end
        @inbounds while i <= m                               # m tail
            for t in 0:NC-1
                s = unsafe_load(pB, _clidx(i, j0 + t, ldb))
                for l in solved; s = muladd(-cf(j0 + t, l), unsafe_load(pB, _clidx(i, l, ldb)), s); end
                unsafe_store!(pB, s, _clidx(i, j0 + t, ldb))
            end
            i += 1
        end
        @inbounds for t in (asc ? (0:NC-1) : (NC-1:-1:0))    # in-block 4×4 coupling solve, vectorized over m
            jj = j0 + t; d = unit ? 1.0 : inv(cf(jj, jj)); rng = asc ? (0:t-1) : (t+1:NC-1); i = 1
            while i + W - 1 <= m
                x = vload(_CVF, _cvptr(pB, i, jj, ldb))
                for u in rng; x = muladd(_CVF(-cf(jj, j0 + u)), vload(_CVF, _cvptr(pB, i, j0 + u, ldb)), x); end
                unit || (x = x * _CVF(d)); vstore(x, _cvptr(pB, i, jj, ldb)); i += W
            end
            while i <= m
                s = unsafe_load(pB, _clidx(i, jj, ldb))
                for u in rng; s = muladd(-cf(jj, j0 + u), unsafe_load(pB, _clidx(i, j0 + u, ldb)), s); end
                unit || (s *= d); unsafe_store!(pB, s, _clidx(i, jj, ldb)); i += 1
            end
        end
    end
    @inline function docol(j)                                # one tail solve-col (k not a multiple of NC)
        solved = asc ? (1:j-1) : (j+1:k); d = unit ? 1.0 : inv(cf(j, j)); i = 1
        @inbounds while i + W - 1 <= m
            x = vload(_CVF, _cvptr(pB, i, j, ldb))
            for l in solved; x = muladd(_CVF(-cf(j, l)), vload(_CVF, _cvptr(pB, i, l, ldb)), x); end
            unit || (x = x * _CVF(d)); vstore(x, _cvptr(pB, i, j, ldb)); i += W
        end
        @inbounds while i <= m
            s = unsafe_load(pB, _clidx(i, j, ldb))
            for l in solved; s = muladd(-cf(j, l), unsafe_load(pB, _clidx(i, l, ldb)), s); end
            unit || (s *= d); unsafe_store!(pB, s, _clidx(i, j, ldb)); i += 1
        end
    end
    # ORDER (as side-L): ascending ⇒ blocks low→high then tail cols; descending ⇒ tail cols (high) FIRST,
    # then blocks high→low (blocks downdate against the tail as "solved").
    if asc
        for bi in 1:nb; doblock((bi - 1) * NC + 1); end
        for j in (nb*NC+1):k; docol(j); end
    else
        for j in (nb*NC == k ? (0:-1) : (k:-1:nb*NC+1)); docol(j); end
        for bi in nb:-1:1; doblock((bi - 1) * NC + 1); end
    end
    return nothing
end

function _trsm_dense_R!(up::Bool, tr::Bool, unit::Bool, A, B)
    m = size(B, 1); k = size(A, 2); T = eltype(B); sz = sizeof(T); ldb = stride(B, 2)
    asc = (up != tr)
    if T === Float64 && A isa StridedMatrix && B isa StridedMatrix &&
            stride(A, 1) == 1 && stride(B, 1) == 1 && k >= 4 && m >= _CHOLW    # strided f64 (trsmR gate) → tile
        GC.@preserve A B _trsm_tile_R_f64!(up, tr, unit, pointer(A), stride(A, 2), pointer(B), ldb, m, k)
        return B
    end
    GC.@preserve A B begin
        pB = pointer(B)
        @inbounds for j in (asc ? (1:k) : (k:-1:1))
            pj = pB + (j - 1) * ldb * sz
            for l in (asc ? (1:(j - 1)) : ((j + 1):k))
                coef = tr ? A[j, l] : A[l, j]
                coef == zero(T) || _axpy_simd!(m, -coef, pB + (l - 1) * ldb * sz, pj)
            end
            unit || _scal_simd_ptr!(pj, m, inv(tr ? A[j, j] : A[j, j]))
        end
    end
    return B
end
const _TRSM_R_FUSE = 128       # ponytail: lower-T real-f64 side-R fused-panel base cap (= potrf NB); recurse above
# side 'R': B := B·op(A)⁻¹, A k×k (k=size(B,2)), unscaled.
function _trsm_right!(up::Bool, tr::Bool, cj::Bool, unit::Bool, A, B)
    k = size(A, 1)
    # Lower-transpose real-f64 wide-B fast base: the fused 12-acc substitution (the potrf panel kernel
    # `_trsm_rl_split_f64!`, IN-PLACE, MC row-chunked — verified relerr ~1e-15 across 56 variants) — no
    # trtri, no unpacked-gemm-into-tmp, no copyback, no recurse-to-32. Net win on wide-B (n=256 0.88→1.02
    # gates; geomean up) + the zpotrf/getrf panel shape. Recursion above `_TRSM_R_FUSE` keeps the
    # cache-blocked off-diagonal gemms and bottoms out here. (Square-gate worst n=32 is the narrow base.)
    if !up && tr && !unit && !cj && k <= _TRSM_R_FUSE && eltype(B) === Float64 &&
            size(B, 1) > _TRSM_NCUT_R && B isa StridedMatrix
        m = size(B, 1); ldb = stride(B, 2)
        mc0 = max(_vwidth(Float64), (_L2_BYTES ÷ 2) ÷ (k * 8))   # MC row-chunk: the mc×k slab the k-repasses
        GC.@preserve A B begin                                   # re-read stays L2-resident (req#8; rows independent)
            pA = pointer(A); ldA = stride(A, 2); pB = pointer(B)
            if _alias_ld(ldb)
                # Aliasing ldb (way-stride multiple): the leaf's solved-column re-reads collide in one L1 set.
                # Solve each chunk into an ODD-ld scratch (conflict-free re-reads); psrc=B is read-once
                # (streaming, no conflict amplification), then copy back. Measured galen +41–49% at 512/1024/
                # 1536; the single copy-back pass nets positive. Bit-identical. (Kernel unchanged — pT re-target.)
                S = _trsm_rpack(Float64, mc0, k); lds = stride(S, 2)
                GC.@preserve S begin
                    pS = pointer(S); i0 = 0
                    while i0 < m
                        mc = min(mc0, m - i0)
                        _trsm_rl_split_f64!(pA, ldA, pB + i0 * 8, ldb, pS, lds, k, mc)
                        @inbounds for c in 1:k                     # copy S[1:mc,c] → B[i0+1:i0+mc,c] (SIMD, trim-safe)
                            r = 1
                            while r + _CHOLW - 1 <= mc
                                vstore(vload(_CVF, _cvptr(pS, r, c, lds)), _cvptr(pB, i0 + r, c, ldb)); r += _CHOLW
                            end
                            while r <= mc
                                unsafe_store!(pB, unsafe_load(pS, _clidx(r, c, lds)), _clidx(i0 + r, c, ldb)); r += 1
                            end
                        end
                        i0 += mc
                    end
                end
            else
                i0 = 0
                while i0 < m
                    mc = min(mc0, m - i0)
                    _trsm_rl_split_f64!(pA, ldA, pB + i0 * 8, ldb, pB + i0 * 8, ldb, k, mc)
                    i0 += mc
                end
            end
        end
        return B
    end
    if eltype(B) <: BlasReal && !cj
        # narrow B (few rows) → dense column-substitution base; wide → invR/gemm base. m is invariant
        # under the side-R column split. (Same dense/gemm split as side L, routed by _TRSM_NCUT_R.)
        if size(B, 1) <= _TRSM_NCUT_R
            k <= _TRSM_DBASE && return _trsm_dense_R!(up, tr, unit, A, B)
        elseif k <= _TRSM_BASE
            return _trsm_base_invR!(up, tr, unit, A, B)
        end
    elseif eltype(B) <: BlasComplex
        # Non-trans: k≤64 uses the trtri-free direct base (beats OB; fixes the universal small-n collapse),
        # and 64<k recurses all the way down to it + gated _gemm_subR!. Measured (consistent harness, all
        # three µarchs) uniformly ≥ the invert+K-TRIM base for side-R — the trtri never amortizes here
        # (its O(k³/6) invert is 40–66% exposed even at k=256, where two 128-trtri bases capped 0.95 on AVX2
        # and direct-recurse beats the wide-B trtri path on AVX-512/Zen5 too). Trans keeps the ≤128 base.
        recbase = (!tr || cj) ? _CTRSM_REC_L : _TRMM_BASE   # transA='N'/'C' → direct-base recursion; 'T' → trtri base
        if k <= recbase
            return _strided1(B) ? _trsm_cmplx_small_R!(up, tr, cj, unit, k, A, B) :
                                  _trsm_cmplx_base_R!(up, tr, cj, unit, k, A, B)
        end
    elseif k <= _TRMM_BASE
        return _trsm_right_base!(up, tr, cj, unit, k, A, B)
    end
    h = _trsplit(k)
    A11 = view(A, 1:h, 1:h); A22 = view(A, (h + 1):k, (h + 1):k)
    B1 = view(B, :, 1:h); B2 = view(B, :, (h + 1):k)
    if up != tr                                       # solve B1, then B2 -= B1·off, then solve B2
        off = tr ? view(A, (h + 1):k, 1:h) : view(A, 1:h, (h + 1):k)
        _trsm_right!(up, tr, cj, unit, A11, B1)
        _gemm_subR!(B2, B1, off, tr, cj)
        _trsm_right!(up, tr, cj, unit, A22, B2)
    else                                              # solve B2, then B1 -= B2·off, then solve B1
        off = tr ? view(A, 1:h, (h + 1):k) : view(A, (h + 1):k, 1:h)
        _trsm_right!(up, tr, cj, unit, A22, B2)
        _gemm_subR!(B1, B2, off, tr, cj)
        _trsm_right!(up, tr, cj, unit, A11, B1)
    end
    return B
end

function _trsm!(side_left::Bool, up::Bool, tr::Bool, cj::Bool, unit::Bool, α::Number, A, B)
    isone(α) || _scal_all!(B, α)
    side_left ? _trsm_left!(up, tr, cj, unit, A, B) : _trsm_right!(up, tr, cj, unit, A, B)
    return B
end

# When A's leading dim is a pure power of 2 (≥512), packing its triangular sub-views thrashes one cache
# set (column starts alias) — measured trsm 0.78–0.94 at ld∈{1024,2048} vs 1.0–1.12 at non-po2. Copying
# A once into a padded-ld scratch (ld=k+8) removes the conflict (B-padding doesn't help — it's the A
# sub-view packing). ponytail: only A needs it; B is solved in place. Cost O(k²) ≪ trsm O(k²n).
# A-pad for power-of-2 leading dims: on AVX2 the O(k²) copy costs MORE than the po2 cache-set aliasing it
# avoids — measured on an idle core (galen is shared → use a free core), disabling it lifts trsm n=512
# 0.89→0.94, n=1024 0.95→0.98, n=2048 →1.02, and getrf (built on trsm) 0.88→0.96. The old "conflict is
# catastrophic 0.78→1.12, the copy pays" was a pre-clean (contended / pre-trtri-fix) measurement. Kept
# for AVX-512/other (untested there; trsm already gates), disabled on AVX2.
@inline _badld(ld::Int) = _vwidth(Float64) != 4 && ld >= 512 && (ld & (ld - 1)) == 0
# Aliasing leading dim: a multiple of the L1 WAY STRIDE (L1_BYTES ÷ assoc ÷ 8 doubles; x86 L1 ≈ 8-way) maps
# every matrix column to the same L1 set → conflict misses on repeated column re-reads. Generalizes _badld
# (po2-only): 1536 = 3·512 also aliases on a 32KB/8-way L1. Derived from detected L1 (req#8). Independent of
# vector width (cache geometry, not ISA) — used by the side-R fused leaf's pT-scratch pack.
const _L1_WAY_D = max(64, _L1_BYTES ÷ 64)      # doubles per L1 way (÷8-way ÷8-byte/double)
@inline _alias_ld(ld::Int) = ld >= _L1_WAY_D && ld % _L1_WAY_D == 0
# _l3_apad (po2-ld A-pad, ld=k+8) lives in the per-type L3Workspace (see src/workspace.jl).

# Public: B := α·op(A)⁻¹·B (side 'L') or α·B·op(A)⁻¹ (side 'R'); A k×k triangular (uplo/transA/diag).
function trsm!(B::AbstractMatrix, A::AbstractMatrix; side::Char = 'L', uplo::Char = 'U',
        transA::Char = 'N', diag::Char = 'N', alpha::Number = true)
    sl = side == 'L'
    k = sl ? size(B, 1) : size(B, 2)
    (size(A, 1) == size(A, 2) == k) || throw(DimensionMismatch("trsm!: A must be $k×$k"))
    # tiny-k fast path: skip the _trsm!/_trsm_left!/_trsm_right! dispatch chain (~3 non-inlined calls ≈ 60ns,
    # which dominates when the solve itself is only ~100ns) and go straight to the dense base kernel.
    if k <= _TRSM_DBASE && eltype(B) <: BlasReal && transA != 'C' && isone(alpha)
        up = uplo == 'U'; tr = transA != 'N'; unit = diag == 'U'
        return sl ? _trsm_dense_L!(up, tr, unit, A, B) : _trsm_dense_R!(up, tr, unit, A, B)
    end
    if eltype(B) <: BlasReal && transA != 'C' && k > _GEMM_UNPACK_MAX &&
       _strided1(A) && _badld(stride(A, 2))
        Apad = _l3_apad(eltype(B), k); copyto!(Apad, A)
        _trsm!(sl, uplo == 'U', transA != 'N', false, diag == 'U', alpha, Apad, B)
    else
        _trsm!(sl, uplo == 'U', transA != 'N', transA == 'C', diag == 'U', alpha, A, B)
    end
    return B
end

# ──────────────────────────────────────────────────────────────────────────────────────────────
# syrk/herk: C := α·op(A)·op(A)ᴴ + β·C, C n×n, only the `uplo` triangle referenced/updated.
# trans 'N': op(A)=A (n×k) ⇒ A·Aᴴ. trans 'T'/'C': op(A)=Aᴴ (A k×n) ⇒ Aᴴ·A. syrk: ᵀ (no conj),
# any T<:Number. herk: Hermitian (conj), real α/β, diagonal forced real. Recursive: diagonal blocks
# recurse (scalar base), the off-diagonal block is a full gemm! — breadth-first correctness (gate later).
const _SYRK_BASE = 48

@inline _symstored(up::Bool, i, j) = up ? (i <= j) : (i >= j)
# β-prescale C's stored triangle. Branch-free, contiguous, triangle-only (was: all n² with a per-element
# _symstored branch — measured 22%/13%/8% of syrk! at n=32/128/256, the whole gate gap since the kernel
# already gates). β=1 is a no-op (skip); β=0 zeroes only the stored half.
function _syrk_scaleC!(C, up::Bool, β)
    isone(β) && return C
    T = eltype(C); n = size(C, 2); z = iszero(β)
    @inbounds for j in 1:n
        for i in (up ? (1:j) : (j:n)); C[i, j] = z ? zero(T) : β * C[i, j]; end
    end
    return C
end
# _L3_NB and the NB×NB diagonal-block scratch `_l3_tmp(T)` (the workspace `diag` field) live in
# src/workspace.jl — const-dispatched for Float64/Float32 so it stays a bare field load, no lookup.

# Triangular-store microkernel: same FMA as the gemm masked microkernel, but on store keeps only the
# stored-triangle entries — for a diagonal-straddling C-tile whose top-left global offset is (r0,c0),
# d0=c0-r0; upper keeps local row ≤ d0+j, lower keeps row ≥ d0+j (j = 0-based column). Accumulates into
# C, so K-accumulation across the gemm pc-loop stays correct (no temp needed).
@generated function _microkernel_tri!(C::Ptr{T}, ldc::Int, Ap::Ptr{T}, Bp::Ptr{T}, kc::Int,
        mre::Int, nre::Int, d0::Int, upper::Bool, ::Val{MR}, ::Val{NR}, ::Val{B0} = Val(false)) where {T, MR, NR, B0}
    W = _vwidth(T); sz = sizeof(T); V = Vec{W, T}
    body = quote end
    push!(body.args, :(lanes = Vec{$W, Int}($(Expr(:tuple, (0:(W - 1))...)))))
    for mi in 1:MR, j in 1:NR; push!(body.args, :($(Symbol(:c, mi, :_, j)) = zero($V))); end
    inner = quote end
    for mi in 1:MR
        push!(inner.args, :($(Symbol(:a, mi)) = vload($V, Ap + (p * $MR + $(mi - 1)) * $(W * sz))))
    end
    for j in 1:NR
        push!(inner.args, :($(Symbol(:b, j)) = $V(unsafe_load(Bp + (p * $NR + $(j - 1)) * $sz))))
        for mi in 1:MR
            cs = Symbol(:c, mi, :_, j); push!(inner.args, :($cs = muladd($(Symbol(:a, mi)), $(Symbol(:b, j)), $cs)))
        end
    end
    push!(body.args, :(for p in 0:(kc - 1); $inner; end))
    for j in 1:NR
        stores = quote end
        push!(stores.args, :(colp = C + $(j - 1) * ldc * $sz)); push!(stores.args, :(thr = d0 + $(j - 1)))
        for mi in 1:MR
            cs = Symbol(:c, mi, :_, j)
            st = B0 ? :(vstore($cs, q, mk)) : :(vstore(vload($V, q, mk) + $cs, q, mk))
            push!(stores.args, :(let base = $((mi - 1) * W), q = colp + $((mi - 1) * W * sz)
                rows = lanes + base
                mk = (rows < mre) & (upper ? (rows <= thr) : (rows >= thr))
                $st
            end))
        end
        push!(body.args, :(if $(j - 1) < nre; $stores; end))
    end
    push!(body.args, :(return nothing))
    return body
end

# Two-product fused microkernel for syr2k: C += op(X1)op(Y1) + op(X2)op(Y2) for ONE C-tile, with a
# SINGLE C read-modify-write (not two). Both products' panels share the kc accumulation in the same
# registers; only at the end is C touched. This halves C traffic and (in :tri mode) the masked store
# vs running two separate microkernels per tile. MODE picks the store: :full / :masked / :tri.
@generated function _microkernel2!(C::Ptr{T}, ldc::Int, Ap1::Ptr{T}, Bp1::Ptr{T}, Ap2::Ptr{T},
        Bp2::Ptr{T}, kc::Int, alpha::T, mre::Int, nre::Int, d0::Int, upper::Bool,
        ::Val{MR}, ::Val{NR}, ::Val{MODE}, ::Val{B0} = Val(false)) where {T, MR, NR, MODE, B0}
    W = _vwidth(T); sz = sizeof(T); V = Vec{W, T}
    body = quote end
    push!(body.args, :(av = $V(alpha)))
    if MODE !== :full
        push!(body.args, :(lanes = Vec{$W, Int}($(Expr(:tuple, (0:(W - 1))...)))))
    end
    for j in 1:NR; push!(body.args, :(_prefetch(C + $(j - 1) * ldc * $sz))); end
    for mi in 1:MR, j in 1:NR; push!(body.args, :($(Symbol(:c, mi, :_, j)) = zero($V))); end
    inner = quote end
    for mi in 1:MR
        push!(inner.args, :($(Symbol(:a, mi)) = vload($V, Ap1 + (p * $MR + $(mi - 1)) * $(W * sz))))
        push!(inner.args, :($(Symbol(:e, mi)) = vload($V, Ap2 + (p * $MR + $(mi - 1)) * $(W * sz))))
    end
    for j in 1:NR
        push!(inner.args, :($(Symbol(:b, j)) = $V(unsafe_load(Bp1 + (p * $NR + $(j - 1)) * $sz))))
        push!(inner.args, :($(Symbol(:f, j)) = $V(unsafe_load(Bp2 + (p * $NR + $(j - 1)) * $sz))))
        for mi in 1:MR
            cs = Symbol(:c, mi, :_, j)
            push!(inner.args, :($cs = muladd($(Symbol(:a, mi)), $(Symbol(:b, j)), $cs)))
            push!(inner.args, :($cs = muladd($(Symbol(:e, mi)), $(Symbol(:f, j)), $cs)))
        end
    end
    push!(body.args, :(for p in 0:(kc - 1); $inner; end))
    for j in 1:NR
        stores = quote end
        push!(stores.args, :(colp = C + $(j - 1) * ldc * $sz))
        MODE === :tri && push!(stores.args, :(thr = d0 + $(j - 1)))
        for mi in 1:MR
            cs = Symbol(:c, mi, :_, j)
            if MODE === :full
                st = B0 ? :(vstore(av * $cs, q)) : :(vstore(muladd(av, $cs, vload($V, q)), q))
                push!(stores.args, :(let q = colp + $((mi - 1) * W * sz); $st; end))
            elseif MODE === :masked
                st = B0 ? :(vstore(av * $cs, q, mk)) : :(vstore(muladd(av, $cs, vload($V, q, mk)), q, mk))
                push!(stores.args, :(let q = colp + $((mi - 1) * W * sz)
                    mk = (lanes + $((mi - 1) * W)) < mre; $st; end))
            else # :tri
                st = B0 ? :(vstore(av * $cs, q, mk)) : :(vstore(muladd(av, $cs, vload($V, q, mk)), q, mk))
                push!(stores.args, :(let q = colp + $((mi - 1) * W * sz)
                    rows = lanes + $((mi - 1) * W)
                    mk = (rows < mre) & (upper ? (rows <= thr) : (rows >= thr))
                    $st; end))
            end
        end
        if MODE === :full
            push!(body.args, stores)
        else
            push!(body.args, :(if $(j - 1) < nre; $stores; end))
        end
    end
    push!(body.args, :(return nothing))
    return body
end

# Single-product α-at-store microkernel for the UNIFIED syrk path: C += α·(A·B), with α applied at the
# store (the unified path packs A once and reads it as both operands, so α cannot be folded into the
# pack — both operands would pick it up, giving α²). MODE: :full / :masked / :tri.
@generated function _microkernel_u!(C::Ptr{T}, ldc::Int, Ap::Ptr{T}, Bp::Ptr{T}, kc::Int, alpha::T,
        mre::Int, nre::Int, d0::Int, upper::Bool, ::Val{MR}, ::Val{NR}, ::Val{MODE}, ::Val{B0} = Val(false)) where {T, MR, NR, MODE, B0}
    W = _vwidth(T); sz = sizeof(T); V = Vec{W, T}
    body = quote end
    push!(body.args, :(av = $V(alpha)))
    if MODE !== :full
        push!(body.args, :(lanes = Vec{$W, Int}($(Expr(:tuple, (0:(W - 1))...)))))
    end
    for j in 1:NR; push!(body.args, :(_prefetch(C + $(j - 1) * ldc * $sz))); end
    for mi in 1:MR, j in 1:NR; push!(body.args, :($(Symbol(:c, mi, :_, j)) = zero($V))); end
    inner = quote end
    for mi in 1:MR
        push!(inner.args, :($(Symbol(:a, mi)) = vload($V, Ap + (p * $MR + $(mi - 1)) * $(W * sz))))
    end
    for j in 1:NR
        push!(inner.args, :($(Symbol(:b, j)) = $V(unsafe_load(Bp + (p * $NR + $(j - 1)) * $sz))))
        for mi in 1:MR
            cs = Symbol(:c, mi, :_, j)
            push!(inner.args, :($cs = muladd($(Symbol(:a, mi)), $(Symbol(:b, j)), $cs)))
        end
    end
    push!(body.args, :(for p in 0:(kc - 1); $inner; end))
    for j in 1:NR
        stores = quote end
        push!(stores.args, :(colp = C + $(j - 1) * ldc * $sz))
        MODE === :tri && push!(stores.args, :(thr = d0 + $(j - 1)))
        for mi in 1:MR
            cs = Symbol(:c, mi, :_, j)
            if MODE === :full
                st = B0 ? :(vstore(av * $cs, q)) : :(vstore(muladd(av, $cs, vload($V, q)), q))
                push!(stores.args, :(let q = colp + $((mi - 1) * W * sz); $st; end))
            elseif MODE === :masked
                st = B0 ? :(vstore(av * $cs, q, mk)) : :(vstore(muladd(av, $cs, vload($V, q, mk)), q, mk))
                push!(stores.args, :(let q = colp + $((mi - 1) * W * sz)
                    mk = (lanes + $((mi - 1) * W)) < mre; $st; end))
            else # :tri
                st = B0 ? :(vstore(av * $cs, q, mk)) : :(vstore(muladd(av, $cs, vload($V, q, mk)), q, mk))
                push!(stores.args, :(let q = colp + $((mi - 1) * W * sz)
                    rows = lanes + $((mi - 1) * W)
                    mk = (rows < mre) & (upper ? (rows <= thr) : (rows >= thr))
                    $st; end))
            end
        end
        if MODE === :full
            push!(body.args, stores)
        else
            push!(body.args, :(if $(j - 1) < nre; $stores; end))
        end
    end
    push!(body.args, :(return nothing))
    return body
end

# Single-pass packed syrk (the gate path for large n): syrk = gemm(A, Aᴴ) with a triangular C. Reuses
# gemm's packing + microkernel; classifies each micro-tile vs the diagonal — skip below-diagonal,
# regular/masked microkernel fully-stored, triangular-store microkernel straddling. Packs A once per
# (ic,pc) panel (reads A like a single gemm — no recursion re-reads). Real (BlasReal) only; α folded
# into the packed A by _pack_A!. C's stored triangle must be β-pre-scaled by the caller.
# General triangular-C gemm: C[uplo-triangle] += α·op(X)·op(Y) (X→A-operand, Y→B-operand), n×n result.
# The reusable core behind syrk (Y=X) and syr2k (two passes). Real only; α folded into packed X.
function _trgemm_packed!(::Val{MR}, ::Val{NR}, up::Bool, α::T, X, tXp::Bool, Y, tYp::Bool, C, k::Int,
        ::Val{OV} = Val(false)) where {T<:BlasReal, MR, NR, OV}
    n = size(C, 1); W = _vwidth(T); mr = MR * W; nr = NR
    kc = min(_KC, k); mc = _at_mc_kc(_HW, T, kc, mr, cld(n, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(n, nr) * nr)
    Ap, Bp = _gemm_scratch(T, cld(mc, mr) * mr * kc, cld(nc, nr) * nr * kc)
    ldc = stride(C, 2); sz = sizeof(T)
    GC.@preserve C Ap Bp begin
        Cp0 = pointer(C); App = pointer(Ap); Bpp = pointer(Bp)
        jc = 0
        while jc < n
            nce = min(nc, n - jc); pc = 0
            while pc < k
                kce = min(kc, k - pc)
                b0 = OV && pc == 0             # overwrite C on the FIRST k-block (β=0 fast path), else add
                _pack_B!(Bp, Y, pc, jc, kce, nce, tYp, nr)
                ic = 0
                while ic < n
                    mce = min(mc, n - ic)
                    _pack_A!(Ap, X, ic, pc, mce, kce, tXp, α, mr)
                    jr = 0
                    while jr < nce
                        nre = min(nr, nce - jr); ir = 0
                        while ir < mce
                            mre = min(mr, mce - ir); r0 = ic + ir; c0 = jc + jr
                            skip = up ? (r0 > c0 + nre - 1) : (r0 + mre - 1 < c0)
                            if !skip
                                Apanel = App + (div(ir, mr) * mr * kce) * sz
                                Bpanel = Bpp + (div(jr, nr) * nr * kce) * sz
                                Cblk = Ptr{T}(Cp0 + (r0 + c0 * ldc) * sz)
                                full = up ? (r0 + mre - 1 <= c0) : (r0 >= c0 + nre - 1)
                                if full && mre == mr && nre == nr
                                    b0 ? _microkernel!(Cblk, ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, Val(MR), Val(NR), Val(true)) :
                                         _microkernel!(Cblk, ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, Val(MR), Val(NR), Val(false))
                                elseif full
                                    b0 ? _microkernel_masked!(Cblk, ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, mre, nre, Val(MR), Val(NR), Val(true)) :
                                         _microkernel_masked!(Cblk, ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, mre, nre, Val(MR), Val(NR), Val(false))
                                else
                                    b0 ? _microkernel_tri!(Cblk, ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, mre, nre, c0 - r0, up, Val(MR), Val(NR), Val(true)) :
                                         _microkernel_tri!(Cblk, ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, mre, nre, c0 - r0, up, Val(MR), Val(NR), Val(false))
                                end
                            end
                            ir += mr
                        end
                        jr += nr
                    end
                    ic += mc
                end
                pc += kc
            end
            jc += nc
        end
    end
    return C
end

# Single-pass packed triangular-output COMPLEX syrk/herk (the complex analogue of _trgemm_packed!):
# C[uplo] += α·op(X)·op(Y) (X→A-operand split-pack, Y→B-operand), n×n. Classifies each micro-tile vs
# the diagonal exactly like the real path (skip-below / full / straddle) — the classification is in
# complex row/col units and the interleaving is invisible to it. α is applied at the store (alr/ali/A1),
# NOT folded into the pack (syrk reads the same operand twice → folding would give α²; the complex pack
# has no α slot anyway). Always accumulates (B0=false); the caller β-pre-scales C's stored triangle via
# _syrk_scaleC! (herk!/syrk! already do). SA/SB are the operand conj signs (herk conjugates one side).
function _trgemm_cmplx_packed!(::Val{SA}, ::Val{SB}, ::Val{NR}, ::Val{A1}, up::Bool,
        alr::T, ali::T, X, tXp::Bool, Y, tYp::Bool, C, k::Int) where {SA, SB, NR, A1, T}
    n = size(C, 1); W = _vwidth(T); mr = _CMR * W; nr = NR
    kc = min(_CKC, k)
    mc = _at_mc_kc(_HW, eltype(C), kc, mr, cld(n, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(n, nr) * nr)
    ApR, ApI, BpR, BpI = _gemm_scratch_cmplx(T, cld(mc, mr) * mr * kc, cld(nc, nr) * nr * kc)
    ldc = stride(C, 2); sz = sizeof(T)                 # ldc in COMPLEX elements (kernel does the ×2)
    GC.@preserve C ApR ApI BpR BpI begin
        Cp0 = Ptr{T}(pointer(C)); ARp = pointer(ApR); AIp = pointer(ApI)
        BRp = pointer(BpR); BIp = pointer(BpI)
        jc = 0
        while jc < n
            nce = min(nc, n - jc); pc = 0
            while pc < k
                kce = min(kc, k - pc)
                _pack_B_cmplx!(BpR, BpI, Y, pc, jc, kce, nce, tYp, nr)
                ic = 0
                while ic < n
                    mce = min(mc, n - ic)
                    _pack_A_cmplx!(ApR, ApI, X, ic, pc, mce, kce, tXp, mr)
                    jr = 0
                    while jr < nce
                        nre = min(nr, nce - jr); ir = 0
                        while ir < mce
                            mre = min(mr, mce - ir); r0 = ic + ir; c0 = jc + jr
                            skip = up ? (r0 > c0 + nre - 1) : (r0 + mre - 1 < c0)
                            if !skip
                                aoff = div(ir, mr) * mr * kce * sz
                                boff = div(jr, nr) * nr * kce * sz
                                AR = Ptr{T}(ARp + aoff); AI = Ptr{T}(AIp + aoff)
                                BR = Ptr{T}(BRp + boff); BI = Ptr{T}(BIp + boff)
                                Cblk = Cp0 + (2 * r0 + 2 * c0 * ldc) * sz     # interleaved: ×2 HERE only
                                full = up ? (r0 + mre - 1 <= c0) : (r0 >= c0 + nre - 1)
                                if full && mre == mr && nre == nr
                                    _microkernel_cmplx!(Cblk, ldc, AR, AI, BR, BI, kce, alr, ali,
                                        Val(_CMR), Val(NR), Val(SA), Val(SB), Val(false), Val(A1))
                                elseif full
                                    _microkernel_cmplx_masked!(Cblk, ldc, AR, AI, BR, BI, kce, alr, ali,
                                        mre, nre, Val(_CMR), Val(NR), Val(SA), Val(SB), Val(false), Val(A1))
                                else
                                    _microkernel_cmplx_tri!(Cblk, ldc, AR, AI, BR, BI, kce, alr, ali,
                                        mre, nre, c0 - r0, up, Val(_CMR), Val(NR), Val(SA), Val(SB), Val(false), Val(A1))
                                end
                            end
                            ir += mr
                        end
                        jr += nr
                    end
                    ic += mc
                end
                pc += kc
            end
            jc += nc
        end
    end
    return C
end

# Unified single-pack complex triangular-output kernel (AVX2 mid-n lever). At the AVX2 tile CMR=1 the
# A-panel (mr=W rows) and B-panel (nr=W cols) layouts COINCIDE (mr==nr==W), so for herk/zsyrk (X===Y)
# ONE `_pack_A_cmplx!` of all n rows feeds BOTH operand roles (A read as W-vectors at div(r0,W)·pstr,
# B read as scalar broadcasts at div(c0,W)·pstr of the same buffer). NR=W=4 (not 6): divides every
# benched n (no column-remainder waste), and 8 accumulators + ar/ai/br/bi = 12 ymm leave 4 registers
# of headroom for the tri/masked store epilogue (the NR=6 path is a zero-headroom 16-ymm fit → epilogue
# spills). Kills the ~12% masked/padded-flop + spill waste that capped NR=6 mid-n at 0.80. Reuses the
# NR=4 `_microkernel_cmplx!` family verbatim. Fable-designed 2026-07-06 (OB-source-verified analysis).
# X≠Y (syr2k) packs each operand once into the two buffer pairs (2 packs, not the multi-path's per-role).
function _trgemm_cmplx_packed_u!(::Val{SA}, ::Val{SB}, ::Val{A1}, up::Bool,
        alr::T, ali::T, X, tXp::Bool, Y, tYp::Bool, C, k::Int) where {SA, SB, A1, T}
    n = size(C, 1); W = _vwidth(T); mr = _CMR * W; nr = W          # unified requires nr == mr (CMR=1)
    kc = min(_CKC, k)
    mc = _at_mc_kc(_HW, eltype(C), kc, mr, cld(n, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(n, nr) * nr)
    plen = cld(n, mr) * mr * kc
    onepack = X === Y && tXp == !tYp                               # herk/zsyrk: B-pack == A-pack
    ApR, ApI, BpR, BpI = _gemm_scratch_cmplx(T, plen, plen)
    ldc = stride(C, 2); sz = sizeof(T)
    GC.@preserve C ApR ApI BpR BpI begin
        Cp0 = Ptr{T}(pointer(C)); ARp = pointer(ApR); AIp = pointer(ApI)
        BRp = onepack ? ARp : pointer(BpR); BIp = onepack ? AIp : pointer(BpI)
        jc = 0
        while jc < n
            nce = min(nc, n - jc); pc = 0
            while pc < k
                kce = min(kc, k - pc); pstr = mr * kce
                _pack_A_cmplx!(ApR, ApI, X, 0, pc, n, kce, tXp, mr)         # ONE pack, all n rows
                onepack || _pack_A_cmplx!(BpR, BpI, Y, 0, pc, n, kce, !tYp, mr)
                ic = 0
                while ic < n
                    mce = min(mc, n - ic); jr = 0
                    while jr < nce
                        nre = min(nr, nce - jr); ir = 0
                        while ir < mce
                            mre = min(mr, mce - ir); r0 = ic + ir; c0 = jc + jr
                            skip = up ? (r0 > c0 + nre - 1) : (r0 + mre - 1 < c0)
                            if !skip
                                aoff = div(r0, mr) * pstr * sz
                                boff = div(c0, mr) * pstr * sz             # SAME layout (mr==nr)
                                AR = Ptr{T}(ARp + aoff); AI = Ptr{T}(AIp + aoff)
                                BR = Ptr{T}(BRp + boff); BI = Ptr{T}(BIp + boff)
                                Cblk = Cp0 + (2 * r0 + 2 * c0 * ldc) * sz
                                full = up ? (r0 + mre - 1 <= c0) : (r0 >= c0 + nre - 1)
                                if full && mre == mr && nre == nr
                                    _microkernel_cmplx!(Cblk, ldc, AR, AI, BR, BI, kce, alr, ali,
                                        Val(_CMR), Val(W), Val(SA), Val(SB), Val(false), Val(A1))
                                elseif full
                                    _microkernel_cmplx_masked!(Cblk, ldc, AR, AI, BR, BI, kce, alr, ali,
                                        mre, nre, Val(_CMR), Val(W), Val(SA), Val(SB), Val(false), Val(A1))
                                else
                                    _microkernel_cmplx_tri!(Cblk, ldc, AR, AI, BR, BI, kce, alr, ali,
                                        mre, nre, c0 - r0, up, Val(_CMR), Val(W), Val(SA), Val(SB), Val(false), Val(A1))
                                end
                            end
                            ir += mr
                        end
                        jr += nr
                    end
                    ic += mc
                end
                pc += kc
            end
            jc += nc
        end
    end
    return C
end

# Fused two-product unified complex syr2k/her2k driver. C[tri] += α·op(X)op(Y)ᴴ + α₂·op(Y)op(X)ᴴ, each
# tile visited ONCE with the fused _microkernel2_cmplx! (both products → one register set → one RMW
# store; the two-CALL version regressed on doubled epilogues). Unified NR=W (CMR=1: mr==nr) so the X/Y
# packs share one panel format and serve both operand roles. α-FOLD: packA holds s·op(X), s = (SA==-1 ?
# conj(α) : α); the kernel conj signs then give product-1 coeff σA(s)=α and product-2 σB(s)=ᾱ (her2k)/α
# (syr2k). β·C by the caller. Fable-designed 2026-07-06.
function _trgemm_cmplx_packed2_u!(::Val{SA}, ::Val{SB}, up::Bool, alr::T, ali::T,
        X, tXp::Bool, Y, tYp::Bool, C, k::Int) where {SA, SB, T}
    n = size(C, 1); W = _vwidth(T); mr = _CMR * W; nr = W
    kc = min(_CKC, k)
    mc = _at_mc_kc(_HW, eltype(C), kc, mr, cld(n, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(n, nr) * nr)
    plen = cld(n, mr) * mr * kc
    ApR, ApI, BpR, BpI = _gemm_scratch_cmplx(T, plen, plen)
    sr = alr; si = SA == -1 ? -ali : ali                         # s = SA==-1 ? conj(α) : α (into X-pack)
    noscale = isone(sr) && iszero(si)
    ldc = stride(C, 2); sz = sizeof(T)
    GC.@preserve C ApR ApI BpR BpI begin
        Cp0 = Ptr{T}(pointer(C)); ARp = pointer(ApR); AIp = pointer(ApI)
        BRp = pointer(BpR); BIp = pointer(BpI)
        jc = 0
        while jc < n
            nce = min(nc, n - jc); pc = 0
            while pc < k
                kce = min(kc, k - pc); pstr = mr * kce
                if noscale                                                 # op(X) once (α folded below)
                    _pack_A_cmplx!(ApR, ApI, X, 0, pc, n, kce, tXp, mr)
                elseif !tXp && _strided1(X)                                # contiguous: fold α into the pack write
                    _pack_A_cmplx_simd_scaled!(ApR, ApI, X, 0, pc, n, kce, mr, sr, si)
                else                                                       # transposed/strided: two-pass
                    _pack_A_cmplx!(ApR, ApI, X, 0, pc, n, kce, tXp, mr)
                    _scale_pack_cmplx!(ApR, ApI, cld(n, mr) * pstr, sr, si)
                end
                _pack_A_cmplx!(BpR, BpI, Y, 0, pc, n, kce, !tYp, mr)        # op(Y) once
                ic = 0
                while ic < n
                    mce = min(mc, n - ic); jr = 0
                    while jr < nce
                        nre = min(nr, nce - jr); ir = 0
                        while ir < mce
                            mre = min(mr, mce - ir); r0 = ic + ir; c0 = jc + jr
                            skip = up ? (r0 > c0 + nre - 1) : (r0 + mre - 1 < c0)
                            if !skip
                                aoff = div(r0, mr) * pstr * sz; boff = div(c0, mr) * pstr * sz
                                P1AR = Ptr{T}(ARp + aoff); P1AI = Ptr{T}(AIp + aoff)   # P1: X rows r0
                                P1BR = Ptr{T}(BRp + boff); P1BI = Ptr{T}(BIp + boff)   # P1: Y cols c0
                                P2AR = Ptr{T}(BRp + aoff); P2AI = Ptr{T}(BIp + aoff)   # P2: Y rows r0
                                P2BR = Ptr{T}(ARp + boff); P2BI = Ptr{T}(AIp + boff)   # P2: X cols c0
                                Cblk = Cp0 + (2 * r0 + 2 * c0 * ldc) * sz
                                full = up ? (r0 + mre - 1 <= c0) : (r0 >= c0 + nre - 1)
                                if full && mre == mr && nre == nr
                                    _microkernel2_cmplx!(Cblk, ldc, P1AR, P1AI, P1BR, P1BI,
                                        P2AR, P2AI, P2BR, P2BI, kce, Val(_CMR), Val(W), Val(SA), Val(SB))
                                elseif full
                                    _microkernel2_cmplx_masked!(Cblk, ldc, P1AR, P1AI, P1BR, P1BI,
                                        P2AR, P2AI, P2BR, P2BI, kce, mre, nre, Val(_CMR), Val(W), Val(SA), Val(SB))
                                else
                                    _microkernel2_cmplx_tri!(Cblk, ldc, P1AR, P1AI, P1BR, P1BI,
                                        P2AR, P2AI, P2BR, P2BI, kce, mre, nre, c0 - r0, up,
                                        Val(_CMR), Val(W), Val(SA), Val(SB))
                                end
                            end
                            ir += mr
                        end
                        jr += nr
                    end
                    ic += mc
                end
                pc += kc
            end
            jc += nc
        end
    end
    return C
end
@inline function _csyr2k_fused!(up::Bool, tr::Bool, herm::Bool, alr::T, ali::T, X, Y, C, k::Int) where {T}
    if !herm
        tr ? _trgemm_cmplx_packed2_u!(Val(1), Val(1), up, alr, ali, X, true, Y, false, C, k) :
             _trgemm_cmplx_packed2_u!(Val(1), Val(1), up, alr, ali, X, false, Y, true, C, k)
    elseif tr
        _trgemm_cmplx_packed2_u!(Val(-1), Val(1), up, alr, ali, X, true, Y, false, C, k)
    else
        _trgemm_cmplx_packed2_u!(Val(1), Val(-1), up, alr, ali, X, false, Y, true, C, k)
    end
end

# Add S's `uplo` triangle into C's; herm → force the diagonal real.
# @inline so the (possibly SubArray) C/S args passed from the D&C recursion don't escape to a
# non-inlined callee and heap-box — the recursion drivers rely on this to stay allocation-free.
@inline function _add_tri!(C, S, up::Bool, herm::Bool, b::Int)
    @inbounds for j in 1:b, i in (up ? (1:j) : (j:b)); C[i, j] += S[i, j]; end
    herm && @inbounds for i in 1:b; C[i, i] = real(C[i, i]); end
    return C
end
# Small-n unified single-pack cutoff. On AVX2 the multi-pack double-packs A (both operands) — its pack
# traffic dominates in cold cache at small n (measured: n=32 multi 0.90 vs unified 1.19). The unified
# single-pack halves that traffic and wins for small n, but is latency-starved (W=4 accs) at larger n
# (n=128 unified 0.73 vs multi 1.02) — so cap it low. AVX-512 uses unified everywhere (_unified_ok).
const _SYRK_UNIFIED_MAX = @load_preference("syrk_unified_max", _vwidth(Float64) == 4 ? 48 : 0)::Int
# Single-product triangular multi-pack row-tile MR. On AVX2 (W=4) the 12-acc gemm tile (MR=_MR=3) zero-pads
# the remainder row-panel at n not divisible by 12 → small/mid-n syrk/syr2k below gate (n=64 0.81, 128 0.94,
# 256 0.92 measured galen). MR=2 (mr=2W=8) divides those sizes AND keeps ample ILP (8 accs) for the
# single-product tri kernel → gates the whole AVX2 range (MR2 ≥ MR3 at every n=64..2048, exact correctness).
# Width-conditional: only F64/AVX2 (W=4); F32/AVX2 and all of AVX-512 keep _MR. Knob "syrk_mr".
const _SYRK_MR = @load_preference("syrk_mr", 2)::Int
@inline _tri_mr(::Type{T}) where {T} = _vwidth(T) == 4 ? _SYRK_MR : _MR
# syrk = one triangular-C gemm (Y = X = A). syr2k = two (A·Bᴴ + B·Aᴴ); real ⇒ both use α.
@inline _syrk_packed!(up::Bool, tr::Bool, α::T, A, C, k::Int) where {T<:BlasReal} =
    (_unified_ok(T) || size(C, 1) <= _SYRK_UNIFIED_MAX) ? _trgemm_packed_u!(up, α, A, tr, C, k) :
        _trgemm_packed!(Val(_tri_mr(T)), Val(_NR), up, α, A, tr, A, !tr, C, k)

# Complex packed syrk/herk dispatch. X=Y=A (both operands the same array). tXp=tr, tYp=!tr (identical to
# real _syrk_packed!). Conj signs mirror _syrk_gemm!'s conjA=tr&&cc, conjB=!tr&&cc (cc=herm): herk
# conjugates the operand that is NOT transposed. zsyrk (herm=false) conjugates neither. A1 = (α==1) skips
# the store-time complex α-multiply. Post-pass forces C's diagonal real for herk (Hermitian: reference
# zherk zeroes the diagonal imaginary part on exit; β is real by herk!'s signature so β·C keeps it real).
@inline function _csyrk_packed!(up::Bool, tr::Bool, herm::Bool, α, A, C, k::Int)
    Tc = eltype(C); a = convert(Tc, α); alr = real(a); ali = imag(a); n = size(C, 1)
    nrv = (_CNR_SMALL != _CNR && max(n, k) <= _CGEMM_NRSMALL_MAX) ? Val(_CNR_SMALL) : Val(_CNR)
    isone(a) ? _csyrk_conj(Val(true), nrv, up, tr, herm, alr, ali, A, C, k) :
               _csyrk_conj(Val(false), nrv, up, tr, herm, alr, ali, A, C, k)
    herm && @inbounds for i in 1:n; C[i, i] = real(C[i, i]); end
    return C
end
@inline _csyrk_conj(::Val{A1}, nr::Val, up::Bool, tr::Bool, herm::Bool, alr::T, ali::T, A, C, k::Int) where {A1, T} =
    _ctrgemm_prod!(Val(A1), nr, up, tr, herm, alr, ali, A, A, C, k)   # syrk/herk: X = Y = A
# n at/below which the unified single-pack tri kernel (NR=W) beats the multi-pack NR=6 path. AVX2 only
# (the layouts coincide at CMR=1 ⇒ mr==nr==W; AVX-512 already gates 1.02-1.23, leave it). Knob per box.
# Cap at 512: unified wins n≤512 (128 0.80→0.99), but its full-n pack loses cache reuse vs the mc/nc-
# blocked multi path at n≥1024 (1024 0.937 vs multi 0.948) — hand large-n back to multi. AVX-512 → 0.
const _CSYRK_UNIFIED_MAX = @load_preference("csyrk_unified_max", _vwidth(Float64) == 4 ? 512 : 0)::Int
# n at/above which the complex rank-k product uses Karatsuba-3M (3 REAL tri-output products on split re/im).
# Both complex tri kernels (unified NR=4, multi NR=6) plateau at 0.88-0.96 for n≥256 on AVX2 — the 4-FMA
# complex microkernel's per-flop ceiling, the SAME one zgemm sidesteps with 3M (which gates 1.1-1.2 at
# these sizes). 3M runs the gating real _trgemm_packed! at 25% fewer flops. AVX2-only (`_CGEMM_3M`);
# windowed to [_CSYRK_3M_MIN, _CGEMM_3M_MAX] with k ≥ _CGEMM_3M_KMIN. Knob per box; retune the unified cap
# to meet this after measuring. AVX-512 keeps the multi path (already gates 1.02-1.23).
const _CSYRK_3M_MIN = @load_preference("csyrk_3m_min", 256)::Int

# C[tri] += α·(P1−P2 + i·(P3−P1−P2)) — triangular RMW combine of the 3 real Karatsuba products (caller
# pre-scaled β·C). Mirror of _combine3! (gemm.jl) restricted to the stored triangle (loop bounds only;
# Karatsuba is pointwise in C so the triangle restriction is exact). P's are real n×n top-left blocks.
function _combine3_tri!(C, P1, P2, P3, alpha::Tc, up::Bool, n::Int) where {Tc}
    Tr = real(Tc); ar = real(alpha); ai = imag(alpha)
    ldc = stride(C, 2); ldp = stride(P1, 2)
    GC.@preserve C P1 P2 P3 begin
        pc = Ptr{Tr}(pointer(C)); p1 = pointer(P1); p2 = pointer(P2); p3 = pointer(P3)
        @inbounds for j in 1:n
            cb = (j - 1) * ldc * 2; pb = (j - 1) * ldp
            lo = up ? 1 : j; hi = up ? j : n
            @simd for i in lo:hi
                a = unsafe_load(p1, pb + i); b = unsafe_load(p2, pb + i)
                zr = a - b; zi = unsafe_load(p3, pb + i) - a - b
                or = unsafe_load(pc, cb + 2i - 1); oi = unsafe_load(pc, cb + 2i)
                unsafe_store!(pc, or + ar * zr - ai * zi, cb + 2i - 1)
                unsafe_store!(pc, oi + ar * zi + ai * zr, cb + 2i)
            end
        end
    end
    return
end
# Karatsuba-3M triangular-output rank-k: C[tri] += α·op(X)·op(Y)ᴴ via 3 real tri-output products
# (P1=op(Xr)op(Yr), P2=op(Xi)op(Yi), P3=op(Xs)op(Ys)) through the gating real `_trgemm_packed!` (OV=true
# overwrite → no P pre-zero; off-triangle garbage is never read by the combine). conjX/conjY (herk's ᴴ)
# fold into the split's imag sign. tXp=tr, tYp=!tr ride the sub-products. α applied at the combine (subs
# run α=1). Reuses the 9-buffer 3M scratch + `_split3!`. Buffers are grow-only (n×k splits + 3 n×n P's).
function _ctrgemm_3m!(up::Bool, conjX::Bool, conjY::Bool, tXp::Bool, tYp::Bool, α::Tc, X, Y, C, k::Int) where {Tc}
    Tr = real(Tc); n = size(C, 1)
    rx = size(X, 1); cx = size(X, 2); ry = size(Y, 1); cy = size(Y, 2)
    t = _gemm_3m_scratch(Tr, rx * cx, ry * cy, n * n)
    GC.@preserve t begin
        w(i, r, c) = unsafe_wrap(Array, pointer(t[i]), (r, c))
        Xr = w(1, rx, cx); Xi = w(2, rx, cx); Xs = w(3, rx, cx)
        _split3!(Xr, Xi, Xs, X, conjX, rx, cx)
        if X === Y && conjX == conjY                       # syrk/zsyrk: X,Y split identically — split once
            Yr, Yi, Ys = Xr, Xi, Xs
        else
            Yr = w(4, ry, cy); Yi = w(5, ry, cy); Ys = w(6, ry, cy)
            _split3!(Yr, Yi, Ys, Y, conjY, ry, cy)
        end
        P1 = w(7, n, n); P2 = w(8, n, n); P3 = w(9, n, n); o = one(Tr)
        _trgemm_packed!(Val(_tri_mr(Tr)), Val(_NR), up, o, Xr, tXp, Yr, tYp, P1, k, Val(true))
        _trgemm_packed!(Val(_tri_mr(Tr)), Val(_NR), up, o, Xi, tXp, Yi, tYp, P2, k, Val(true))
        _trgemm_packed!(Val(_tri_mr(Tr)), Val(_NR), up, o, Xs, tXp, Ys, tYp, P3, k, Val(true))
        _combine3_tri!(C, P1, P2, P3, α, up, n)
    end
    return C
end
# ONE triangular-C complex product C[tri] += α·op(X)·op(Y)ᴴ (skip/full/tri tiles). herm conjugates the
# ᴴ operand (tr='N' → Y via SB=-1; tr='C' → X via SA=-1); syrk conjugates neither. syrk/herk pass X=Y=A;
# syr2k/her2k call twice (A,B then B,A). Conj signs mirror _syrk_gemm!'s conjA=tr&&cc, conjB=!tr&&cc.
# X===Y (herk/zsyrk) on AVX2 mid-n → unified single-pack driver (NR=W, half the pack, no NR=6 spill).
@inline function _ctrgemm_prod!(::Val{A1}, ::Val{NR}, up::Bool, tr::Bool, herm::Bool,
        alr::T, ali::T, X, Y, C, k::Int) where {A1, NR, T}
    n = size(C, 1)
    if _CGEMM_3M && _vwidth(T) == 4 && _CSYRK_3M_MIN <= n <= _CGEMM_3M_MAX && k >= _CGEMM_3M_KMIN
        # large-n: Karatsuba-3M (the complex tri kernels plateau ~0.92 here). herk conjugates op(X) at
        # tr='C' (SA=-1) / op(Y) at tr='N' (SB=-1); syrk conjugates neither. tXp=tr, tYp=!tr.
        return _ctrgemm_3m!(up, herm && tr, herm && !tr, tr, !tr, Complex(alr, ali), X, Y, C, k)
    end
    if X === Y && _vwidth(T) == 4 && n <= _CSYRK_UNIFIED_MAX   # herk/zsyrk: single-pack win
        return _ctrgemm_prod_u!(Val(A1), up, tr, herm, alr, ali, X, Y, C, k)
    end
    if !herm
        tr ? _trgemm_cmplx_packed!(Val(1), Val(1), Val(NR), Val(A1), up, alr, ali, X, true, Y, false, C, k) :
             _trgemm_cmplx_packed!(Val(1), Val(1), Val(NR), Val(A1), up, alr, ali, X, false, Y, true, C, k)
    elseif tr
        _trgemm_cmplx_packed!(Val(-1), Val(1), Val(NR), Val(A1), up, alr, ali, X, true, Y, false, C, k)
    else
        _trgemm_cmplx_packed!(Val(1), Val(-1), Val(NR), Val(A1), up, alr, ali, X, false, Y, true, C, k)
    end
end
@inline function _ctrgemm_prod_u!(::Val{A1}, up::Bool, tr::Bool, herm::Bool,
        alr::T, ali::T, X, Y, C, k::Int) where {A1, T}
    if !herm
        tr ? _trgemm_cmplx_packed_u!(Val(1), Val(1), Val(A1), up, alr, ali, X, true, Y, false, C, k) :
             _trgemm_cmplx_packed_u!(Val(1), Val(1), Val(A1), up, alr, ali, X, false, Y, true, C, k)
    elseif tr
        _trgemm_cmplx_packed_u!(Val(-1), Val(1), Val(A1), up, alr, ali, X, true, Y, false, C, k)
    else
        _trgemm_cmplx_packed_u!(Val(1), Val(-1), Val(A1), up, alr, ali, X, false, Y, true, C, k)
    end
end
# Complex syr2k/her2k via the triangular-output kernel: C[tri] += α·op(A)op(B)ᴴ + α2·op(B)op(A)ᴴ
# (α2 = ᾱ for her2k, α for syr2k) as TWO tri-output products — only the stored triangle, no dense n×n
# temp (that was the 2× waste in _syr2k_acc!). β·C applied by the caller (_syrk_scaleC!). her2k forces
# the diagonal real on exit (both products sum to a real diagonal; this clears FP rounding).
# n at/below which complex syr2k/her2k uses the FUSED two-product unified driver (AVX2). Cap 512 mirrors
# _CSYRK_UNIFIED_MAX (large-n full-n pack loses cache reuse vs the blocked multi path). Knob per box.
# Cap 256: fused wins n≤256 (n≤64 beats OB, 128 0.86→0.92); at n≥512 its full-n pack loses cache reuse
# vs the mc/nc-blocked multi tri path (512 0.92 vs 0.944) — hand large-n back to multi. AVX-512 → 0.
# Fused cap lowered 256→192 on AVX2 so n≥256 syr2k/her2k reach the 3M branch in _ctrgemm_prod! (measured:
# n=256 3M = 1.04-1.06 vs fused 0.93-0.94). n≤128 stays fused (3M's two-pass overhead loses there: 0.89).
const _CSYR2K_FUSED_MAX = @load_preference("csyr2k_fused_max", _vwidth(Float64) == 4 ? 192 : 0)::Int
@inline function _csyr2k_packed!(up::Bool, tr::Bool, herm::Bool, α, A, B, C, k::Int)
    Tc = eltype(C); a = convert(Tc, α); n = size(C, 1)
    if _vwidth(real(Tc)) == 4 && n <= _CSYR2K_FUSED_MAX && k > 0
        _csyr2k_fused!(up, tr, herm, real(a), imag(a), A, B, C, k)   # both products, one RMW/tile
    else
        a2 = herm ? conj(a) : a
        nrv = (_CNR_SMALL != _CNR && max(n, k) <= _CGEMM_NRSMALL_MAX) ? Val(_CNR_SMALL) : Val(_CNR)
        _csyr2k_prod!(nrv, up, tr, herm, real(a), imag(a), A, B, C, k)     # α·op(A)op(B)ᴴ
        _csyr2k_prod!(nrv, up, tr, herm, real(a2), imag(a2), B, A, C, k)   # α2·op(B)op(A)ᴴ
    end
    herm && @inbounds for i in 1:n; C[i, i] = real(C[i, i]); end
    return C
end
@inline _csyr2k_prod!(nr::Val, up::Bool, tr::Bool, herm::Bool, alr::T, ali::T, X, Y, C, k::Int) where {T} =
    (isone(alr) && iszero(ali)) ? _ctrgemm_prod!(Val(true), nr, up, tr, herm, alr, ali, X, Y, C, k) :
                                  _ctrgemm_prod!(Val(false), nr, up, tr, herm, alr, ali, X, Y, C, k)

# The fused two-product syr2k driver's four-buffer scratch (two A-packs, two B-packs) is the per-type
# L3Workspace `s2` field — `_syr2k_scratch(T, lenA, lenB)` grows and returns it (see src/workspace.jl).

# Fused two-product triangular-C gemm: C[tri] += α·op(X1)·op(Y1) + α·op(X2)·op(Y2). The core of syr2k.
# Both products are packed (X1,Y1,X2,Y2) and each C-tile is visited ONCE: _microkernel2! accumulates
# both products in registers and does a single C read-modify-write. Running two _trgemm_packed! passes
# instead touches every C-tile twice (the microkernel loads/stores C per call) — measured 2.05× a syrk
# vs OpenBLAS's ~1.93×. This fused tile-pass removes that second C round-trip.
function _trgemm_packed2!(up::Bool, α::T, X1, tX1::Bool, Y1, tY1::Bool,
        X2, tX2::Bool, Y2, tY2::Bool, C, k::Int, ::Val{MRV} = Val(_MR),
        ::Val{NRV} = Val(_NR), ::Val{OV} = Val(false)) where {T<:BlasReal, MRV, NRV, OV}
    n = size(C, 1); W = _vwidth(T); mr = MRV * W; nr = NRV
    kc = min(_KC, k); mc = _at_mc_kc(_HW, T, kc, mr, cld(n, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(n, nr) * nr)
    Ap1, Bp1, Ap2, Bp2 = _syr2k_scratch(T, cld(mc, mr) * mr * kc, cld(nc, nr) * nr * kc)
    ldc = stride(C, 2); sz = sizeof(T)
    GC.@preserve C Ap1 Bp1 Ap2 Bp2 begin
        Cp0 = pointer(C); A1p = pointer(Ap1); B1p = pointer(Bp1); A2p = pointer(Ap2); B2p = pointer(Bp2)
        jc = 0
        while jc < n
            nce = min(nc, n - jc); pc = 0
            while pc < k
                kce = min(kc, k - pc)
                b0 = OV && pc == 0             # overwrite C on the first k-block (β=0), else accumulate
                _pack_B!(Bp1, Y1, pc, jc, kce, nce, tY1, nr)
                _pack_B!(Bp2, Y2, pc, jc, kce, nce, tY2, nr)
                ic = 0
                while ic < n
                    mce = min(mc, n - ic)
                    _pack_A!(Ap1, X1, ic, pc, mce, kce, tX1, one(T), mr)
                    _pack_A!(Ap2, X2, ic, pc, mce, kce, tX2, one(T), mr)
                    jr = 0
                    while jr < nce
                        nre = min(nr, nce - jr); ir = 0
                        while ir < mce
                            mre = min(mr, mce - ir); r0 = ic + ir; c0 = jc + jr
                            skip = up ? (r0 > c0 + nre - 1) : (r0 + mre - 1 < c0)
                            if !skip
                                off = div(ir, mr) * mr * kce; boff = div(jr, nr) * nr * kce
                                a1 = A1p + off * sz; b1 = B1p + boff * sz
                                a2 = A2p + off * sz; b2 = B2p + boff * sz
                                Cblk = Ptr{T}(Cp0 + (r0 + c0 * ldc) * sz)
                                full = up ? (r0 + mre - 1 <= c0) : (r0 >= c0 + nre - 1)
                                if full && mre == mr && nre == nr
                                    b0 ? _microkernel2!(Cblk, ldc, Ptr{T}(a1), Ptr{T}(b1), Ptr{T}(a2), Ptr{T}(b2), kce, α, mre, nre, 0, up, Val(MRV), Val(NRV), Val(:full), Val(true)) :
                                         _microkernel2!(Cblk, ldc, Ptr{T}(a1), Ptr{T}(b1), Ptr{T}(a2), Ptr{T}(b2), kce, α, mre, nre, 0, up, Val(MRV), Val(NRV), Val(:full), Val(false))
                                elseif full
                                    b0 ? _microkernel2!(Cblk, ldc, Ptr{T}(a1), Ptr{T}(b1), Ptr{T}(a2), Ptr{T}(b2), kce, α, mre, nre, 0, up, Val(MRV), Val(NRV), Val(:masked), Val(true)) :
                                         _microkernel2!(Cblk, ldc, Ptr{T}(a1), Ptr{T}(b1), Ptr{T}(a2), Ptr{T}(b2), kce, α, mre, nre, 0, up, Val(MRV), Val(NRV), Val(:masked), Val(false))
                                else
                                    b0 ? _microkernel2!(Cblk, ldc, Ptr{T}(a1), Ptr{T}(b1), Ptr{T}(a2), Ptr{T}(b2), kce, α, mre, nre, c0 - r0, up, Val(MRV), Val(NRV), Val(:tri), Val(true)) :
                                         _microkernel2!(Cblk, ldc, Ptr{T}(a1), Ptr{T}(b1), Ptr{T}(a2), Ptr{T}(b2), kce, α, mre, nre, c0 - r0, up, Val(MRV), Val(NRV), Val(:tri), Val(false))
                                end
                            end
                            ir += mr
                        end
                        jr += nr
                    end
                    ic += mc
                end
                pc += kc
            end
            jc += nc
        end
    end
    return C
end
# Unified single-pack needs the A-pack panel width (mr=W) to equal the B-pack width (nr=_NR) — an
# W×W tile, MR=1 — so it yields exactly W vector accumulators. That's a win only when W is large
# enough to hide FMA latency (W>=8, AVX-512: 8 accs). On AVX2 (W=4) it's just 4 accs = latency-
# STARVED, so there we fall to the multi-pack _trgemm_packed! with the wider _MR×_NR tile (12 accs
# on Zen3) — it double-packs A but that's cheaper than starving. (Zen3-swept 2026-07-02.)
@inline _unified_ok(::Type{T}) where {T} = _vwidth(T) == _NR && _vwidth(T) >= 8

# Unified single-pack syrk: pack A ONCE into W-row panels; the A-operand (vector load, panel ir) and
# the B-operand (scalar broadcast, panel jr) both read that one buffer. 8×8 tile (MR=1) so both packs'
# layouts coincide; α applied at the store (shared buffer ⇒ can't fold α into the pack).
function _trgemm_packed_u!(up::Bool, α::T, A, tAp::Bool, C, k::Int) where {T<:BlasReal}
    n = size(C, 1); W = _vwidth(T); mr = W; nr = _NR
    kc = min(_KC, k); mc = _at_mc_kc(_HW, T, kc, mr, cld(n, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(n, nr) * nr)
    plen = cld(n, mr) * mr * kc
    pk = _syr2k_scratch(T, plen, plen); packA = pk[1]
    ldc = stride(C, 2); sz = sizeof(T)
    GC.@preserve C packA begin
        Cp0 = pointer(C); PA = pointer(packA)
        jc = 0
        while jc < n
            nce = min(nc, n - jc); pc = 0
            while pc < k
                kce = min(kc, k - pc); pstr = mr * kce
                _pack_A!(packA, A, 0, pc, n, kce, tAp, one(T), mr)
                ic = 0
                while ic < n
                    mce = min(mc, n - ic); jr = 0
                    while jr < nce
                        nre = min(nr, nce - jr); ir = 0
                        while ir < mce
                            mre = min(mr, mce - ir); r0 = ic + ir; c0 = jc + jr
                            skip = up ? (r0 > c0 + nre - 1) : (r0 + mre - 1 < c0)
                            if !skip
                                a = PA + div(r0, mr) * pstr * sz; b = PA + div(c0, nr) * pstr * sz
                                Cblk = Ptr{T}(Cp0 + (r0 + c0 * ldc) * sz)
                                full = up ? (r0 + mre - 1 <= c0) : (r0 >= c0 + nre - 1)
                                if full && mre == mr && nre == nr
                                    _microkernel_u!(Cblk, ldc, Ptr{T}(a), Ptr{T}(b), kce, α, mre, nre, 0, up, Val(1), Val(_NR), Val(:full))
                                elseif full
                                    _microkernel_u!(Cblk, ldc, Ptr{T}(a), Ptr{T}(b), kce, α, mre, nre, 0, up, Val(1), Val(_NR), Val(:masked))
                                else
                                    _microkernel_u!(Cblk, ldc, Ptr{T}(a), Ptr{T}(b), kce, α, mre, nre, c0 - r0, up, Val(1), Val(_NR), Val(:tri))
                                end
                            end
                            ir += mr
                        end
                        jr += nr
                    end
                    ic += mc
                end
                pc += kc
            end
            jc += nc
        end
    end
    return C
end

# Unified single-pack syr2k: pack A and B ONCE each (W-row panels); the two products read them in
# swapped roles (A·Bᵀ: packA-rows·packB-cols; B·Aᵀ: packB-rows·packA-cols). 2 packs, not 4.
function _trgemm_packed2_u!(up::Bool, α::T, A, tAp::Bool, Bm, tBp::Bool, C, k::Int) where {T<:BlasReal}
    n = size(C, 1); W = _vwidth(T); mr = W; nr = _NR
    kc = min(_KC, k); mc = _at_mc_kc(_HW, T, kc, mr, cld(n, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(n, nr) * nr)
    plen = cld(n, mr) * mr * kc
    pk = _syr2k_scratch(T, plen, plen); packA = pk[1]; packB = pk[2]
    ldc = stride(C, 2); sz = sizeof(T)
    GC.@preserve C packA packB begin
        Cp0 = pointer(C); PA = pointer(packA); PB = pointer(packB)
        jc = 0
        while jc < n
            nce = min(nc, n - jc); pc = 0
            while pc < k
                kce = min(kc, k - pc); pstr = mr * kce
                _pack_A!(packA, A, 0, pc, n, kce, tAp, one(T), mr)
                _pack_A!(packB, Bm, 0, pc, n, kce, tBp, one(T), mr)
                ic = 0
                while ic < n
                    mce = min(mc, n - ic); jr = 0
                    while jr < nce
                        nre = min(nr, nce - jr); ir = 0
                        while ir < mce
                            mre = min(mr, mce - ir); r0 = ic + ir; c0 = jc + jr
                            skip = up ? (r0 > c0 + nre - 1) : (r0 + mre - 1 < c0)
                            if !skip
                                ip = div(r0, mr) * pstr * sz; jp = div(c0, nr) * pstr * sz
                                a1 = PA + ip; b1 = PB + jp; a2 = PB + ip; b2 = PA + jp
                                Cblk = Ptr{T}(Cp0 + (r0 + c0 * ldc) * sz)
                                full = up ? (r0 + mre - 1 <= c0) : (r0 >= c0 + nre - 1)
                                if full && mre == mr && nre == nr
                                    _microkernel2!(Cblk, ldc, Ptr{T}(a1), Ptr{T}(b1), Ptr{T}(a2), Ptr{T}(b2), kce, α, mre, nre, 0, up, Val(1), Val(_NR), Val(:full))
                                elseif full
                                    _microkernel2!(Cblk, ldc, Ptr{T}(a1), Ptr{T}(b1), Ptr{T}(a2), Ptr{T}(b2), kce, α, mre, nre, 0, up, Val(1), Val(_NR), Val(:masked))
                                else
                                    _microkernel2!(Cblk, ldc, Ptr{T}(a1), Ptr{T}(b1), Ptr{T}(a2), Ptr{T}(b2), kce, α, mre, nre, c0 - r0, up, Val(1), Val(_NR), Val(:tri))
                                end
                            end
                            ir += mr
                        end
                        jr += nr
                    end
                    ic += mc
                end
                pc += kc
            end
            jc += nc
        end
    end
    return C
end

# The two-product _microkernel2! holds 2·MR A-vectors (both products' A packs), so it needs a SMALLER
# tile than gemm: MR·NR + 2·MR + 2 ≤ (vector registers). W=4/AVX2 (16 ymm): MR=2 → 8 accs+4+2=14 fits;
# gemm's MR=3 gives 12+6+2=20 → SPILL (the 0.65 large-n syr2k). W=8 (32 zmm): _MR=2 → 16+4+2=22, fine.
# Overridable "syr2k_mr". (syrk uses the single-product kernel, only MR A-vectors → gemm's tile fits.)
const _SYR2K_MR = @load_preference("syr2k_mr", _vwidth(Float64) == 4 ? 2 : _MR)::Int
# nr for the two-product tile (Preferences knob). Default _NR: widening to NR=5 with MR=2 was measured
# NEUTRAL-to-worse on Zen3 (n=256 unchanged, n=1024 0.985→0.96) — the tile wasn't ILP-starved, so keep _NR.
const _SYR2K_NR = @load_preference("syr2k_nr", _NR)::Int
# n above which syr2k does TWO full-kernel passes (OpenBLAS-style) instead of the fused two-product tile.
# On AVX2 the fused MR=2 tile has only 8 accumulators (ILP-starved on 16 regs); two _trgemm_packed! passes
# (12 accs each) win at n>128 despite 2× C traffic (measured: n=512 0.94→1.00, n=1024 0.95→1.02). AVX-512
# keeps the fused unified path (32 regs, not starved). Overridable "syr2k_2pass".
const _SYR2K_2PASS = @load_preference("syr2k_2pass", _vwidth(Float64) == 4 ? 128 : typemax(Int))::Int
# Handles β internally: the two-pass path can OVERWRITE C on its first pass when β=0 (skipping the
# separate scaleC zero-pass — measured the whole n=256 gate gap, since scaleC + 2 adds is 3 C-touches at
# the L2-resonant size). The fused/unified paths ADD, so they need C β-pre-scaled (zeroed if β=0).
@inline function _syr2k_packed!(up::Bool, tr::Bool, α::T, β::T, A, Bm, C, k::Int) where {T<:BlasReal}
    if _unified_ok(T)
        _syrk_scaleC!(C, up, β)
        return _trgemm_packed2_u!(up, α, A, tr, Bm, tr, C, k)
    elseif size(C, 1) > _SYR2K_2PASS      # C = α·op(A)·op(B)ᵀ + α·op(B)·op(A)ᵀ (+β·C) — two triangular gemms
        β0 = iszero(β)
        β0 || _syrk_scaleC!(C, up, β)      # β≠0: pre-scale; β=0: pass 1 overwrites (Val(true))
        X1, tX1, Y1, tY1, X2, tX2, Y2, tY2 = tr ? (A, true, Bm, false, Bm, true, A, false) :
                                                  (A, false, Bm, true, Bm, false, A, true)
        β0 ? _trgemm_packed!(Val(_tri_mr(T)), Val(_NR), up, α, X1, tX1, Y1, tY1, C, k, Val(true)) :
             _trgemm_packed!(Val(_tri_mr(T)), Val(_NR), up, α, X1, tX1, Y1, tY1, C, k, Val(false))
        _trgemm_packed!(Val(_tri_mr(T)), Val(_NR), up, α, X2, tX2, Y2, tY2, C, k)
        return C
    end
    β0 = iszero(β)
    β0 || _syrk_scaleC!(C, up, β)          # fused kernel writes each C-tile ONCE → overwrite when β=0
    X1, tX1, Y1, tY1, X2, tX2, Y2, tY2 = tr ? (A, true, Bm, false, Bm, true, A, false) :
                                              (A, false, Bm, true, Bm, false, A, true)
    return β0 ?
        _trgemm_packed2!(up, α, X1, tX1, Y1, tY1, X2, tX2, Y2, tY2, C, k, Val(_SYR2K_MR), Val(_SYR2K_NR), Val(true)) :
        _trgemm_packed2!(up, α, X1, tX1, Y1, tY1, X2, tX2, Y2, tY2, C, k, Val(_SYR2K_MR), Val(_SYR2K_NR), Val(false))
end

# Recursive blocked syrk/herk (the gate path): split into 2×2; the two diagonal blocks recurse and the
# off-diagonal block is one large gemm! written straight into C's stored triangle (correct flops, no
# temp). Only the small diagonal BASE (≤ _SYRK_DBASE) goes through a gemm→temp + triangle-add, so the
# unavoidable "compute the full b×b but keep the triangle" waste is confined to tiny base blocks
# (≈ 2·DBASE/n of the flops). Large off-diagonal gemms keep the bulk at peak.
# Recursion base for the diagonal (gemm→temp + triangle-add wastes 2× flops on b×b; smaller base =
# more work in efficient off-diagonal gemms). Preferences-overridable "syrk_dbase" (Zen3 sweep).
const _SYRK_DBASE = @load_preference("syrk_dbase", 32)::Int
# n above which the single-pass packed syrk beats the gemm→temp recursion. On W=8 = _GEMM_UNPACK_MAX
# (unchanged: recursion up to the gemm unpack size, packed above). On AVX2 the recursion base's 2×-flop
# waste bites earlier and the packed path (per-microtile diagonal, no waste) wins from n≈24 — but the
# tiny-n gemm base still wins below that (packing overhead dominates on n≤20). Overridable per machine.
# (OpenBLAS-style dense-scratch + scalar triangular copyback for the diagonal tile was A/B-tested here
# and measured EQUAL to the masked-store _microkernel_tri! on AVX2 — no gain, not adopted.)
const _SYRK_PACK_CUT = @load_preference("syrk_pack_cut", _vwidth(Float64) == 4 ? 23 : _GEMM_UNPACK_MAX)::Int
# n above which complex syrk/herk take the single-pass packed triangular path (no 2×-flop diagonal waste,
# no recursion — vs the wasteful _syrk_rec! below). TRANS-DEPENDENT crossover (measured, Zen4/Zen5):
# trans='N' recursion base packs A's contiguous columns via the fast SIMD deinterleave → it WINS small-n
# (n=8 gates 1.2× on AVX-512), packed wins n≥~24. trans='C'/'T' needs a transposed A-pack → recursion is
# slow at every small n while the packed path amortizes it, so packed wins uniformly (route it from ~n=4).
# Complex micro-tile is _CMR·W complex rows (AVX2 z: 4, AVX-512 z: 16). Per-machine Preferences knobs.
const _CSYRK_PACK_CUT = @load_preference("csyrk_pack_cut", 16)::Int        # trans='N': recursion below this
const _CSYRK_PACK_CUT_T = @load_preference("csyrk_pack_cut_t", 4)::Int     # trans='C'/'T': packed ~always
# trans='N' complex n≤this ⇒ unpacked triangular kernel (`_ctri_unpacked!`): the packed path's operand-pack
# + NR-remainder overhead and the recursion base's 2×-flop waste BOTH miss the gate at small n, while the
# unpacked complex microkernel (what zgemm rides) gates there. The cutoff is where the packed path's NR-tile
# amortization overtakes unpacked — a microkernel-ramp crossover, µarch-specific (NOT a cache formula), so
# it is `_vwidth`-keyed & Preferences-overridable. Measured boost-locked (bench/csyrk_avx2_calib.jl):
#  • AVX2 (W=4, galen): the recursion base (n≤_CSYRK_PACK_CUT=16) 2×-wastes → zherk/zsyrk n=16 DIP to 0.87-
#    0.91 (sub-gate). Unpacked-tri gates all 4 ops at n≤16 (herk 1.49/her2k 1.21) and beats the dip; packed
#    overtakes by n=24, so cutoff=16.  • AVX-512 (W=8): packed's edge overhead is larger → unpacked wins
#    broadly. Measured both boxes boost-locked: Zen4 (wm) unpacked ≥ packed to n=192 (packed reclaims n=256);
#    Zen5 (neuro) unpacked ≥ packed at EVERY n≤256. Cutoff 192 is safe on both (avoids Zen4's n=256 packed
#    preference) and lifts the n=128/192 complex rank-k the factorizations recurse through. (One formula for
#    both µarchs remains req#8 debt.)
const _CSYRK_UNPACK_MAX = @load_preference("csyrk_unpack_max", _vwidth(Float64) == 4 ? 16 : 192)::Int

# ── Unpacked triangular-output complex rank-k/rank-2k (small-n, trans='N'). Routes herk/syrk (and, via two
# products, her2k/syr2k) through the SAME direct-read `_uker_cmplx!` as zgemm (no operand pack) but stores
# only the `up` triangle: off-diagonal tiles store full, the diagonal-straddling tile stores masked (TRI
# mode), below-triangle tiles are skipped (recovering the flops the recursion base wastes). trans='N' only
# (each product is X·Yᴴ: SA=1, TB=true, SB=herm?-1:1); β pre-applied by caller ⇒ accumulate (B0=false).
# Vals resolved to concrete via the sb/a1/ar/nr chain (type-stable + trim-safe).
@inline function _ctri_unpacked!(up::Bool, herm::Bool, α, A, C, k::Int)
    Tc = eltype(C); a = convert(Tc, α); n = size(C, 1)
    k == 0 && return C
    _ctri_sb!(up, herm, real(a), imag(a), A, A, C, k, n)               # rank-k: X = Y = A
    herm && @inbounds for i in 1:n; C[i, i] = real(C[i, i]); end        # Hermitian diagonal is real
    return C
end
# rank-2k: C[tri] += α·A·Bᴴ + α₂·B·Aᴴ (α₂ = conj(α) her2k, α syr2k). Two unpacked-tri products, one each.
@inline function _ctri2_unpacked!(up::Bool, herm::Bool, α, A, B, C, k::Int)
    Tc = eltype(C); a = convert(Tc, α); a2 = herm ? conj(a) : a; n = size(C, 1)
    k == 0 && return C
    _ctri_sb!(up, herm, real(a), imag(a), A, B, C, k, n)              # α·A·Bᴴ
    _ctri_sb!(up, herm, real(a2), imag(a2), B, A, C, k, n)            # α₂·B·Aᴴ
    herm && @inbounds for i in 1:n; C[i, i] = real(C[i, i]); end
    return C
end
@inline _ctri_sb!(up, herm, alr::T, ali::T, X, Y, C, k, n) where {T} = herm ?
    _ctri_a1!(up, Val(-1), alr, ali, X, Y, C, k, n) : _ctri_a1!(up, Val(1), alr, ali, X, Y, C, k, n)
@inline _ctri_a1!(up, sb::Val, alr::T, ali::T, X, Y, C, k, n) where {T} = (isone(alr) && iszero(ali)) ?
    _ctri_ar!(up, sb, Val(true), alr, ali, X, Y, C, k, n) : _ctri_ar!(up, sb, Val(false), alr, ali, X, Y, C, k, n)
@inline _ctri_ar!(up, sb::Val, a1::Val, alr::T, ali::T, X, Y, C, k, n) where {T} = iszero(ali) ?
    _ctri_nr!(up, sb, a1, Val(true), alr, ali, X, Y, C, k, n) : _ctri_nr!(up, sb, a1, Val(false), alr, ali, X, Y, C, k, n)
@inline _ctri_nr!(up, sb::Val, a1::Val, ar::Val, alr::T, ali::T, X, Y, C, k, n) where {T} =
    (_CNR_SMALL != _CNR && max(n, k) <= _CGEMM_NRSMALL_MAX) ?
        _ctri_core!(Val(_CNR_SMALL), up, sb, a1, ar, alr, ali, X, Y, C, k, n) :
        _ctri_core!(Val(_CNR), up, sb, a1, ar, alr, ali, X, Y, C, k, n)
# One product's triangular tile sweep: C[tri] += α·X·Yᴴ. NR-col panels × mr-row tiles; classify each tile
# vs the stored (`up`) diagonal — skip (outside) / full (interior) / tri (straddling). Mirrors `_uker_sweep!`'s
# MR + edge choice + advance. X===Y for rank-k; distinct for each rank-2k product.
function _ctri_core!(::Val{NR}, up::Bool, ::Val{SB}, ::Val{A1}, ::Val{AR},
        alr::T, ali::T, X, Y, C, k::Int, n::Int) where {NR, SB, A1, AR, T}
    W = _vwidth(T); mr = _CMR * W
    ldx = stride(X, 2); ldy = stride(Y, 2); ldc = stride(C, 2)
    parX = parent(X); parY = parent(Y); parC = parent(C)
    GC.@preserve parX parY parC begin
        Xp = Ptr{T}(pointer(X)); Yp = Ptr{T}(pointer(Y)); Cp = Ptr{T}(pointer(C))
        jr = 0
        while jr < n
            nre = min(NR, n - jr)
            ir = 0
            while ir < n
                mre = min(mr, n - ir); nrv = cld(mre, W)
                below = up ? (ir >= jr + nre) : (ir + mre <= jr)         # tile entirely off the stored triangle
                full  = up ? (ir + mre - 1 <= jr) : (ir >= jr + nre - 1) # tile entirely inside it
                if below
                    # skip: nothing stored
                elseif full && mre == mr
                    _uker_cmplx!(Cp, ldc, Xp, ldx, ir, Yp, ldy, jr, k, alr, ali, mre, nre,
                        Val(_CMR), Val(NR), Val(true), Val(1), Val(SB), Val(false), Val(A1), Val(AR), Val(true), Val(false), 0, true)
                elseif full && nrv >= _CMR
                    _uker_cmplx!(Cp, ldc, Xp, ldx, ir, Yp, ldy, jr, k, alr, ali, mre, nre,
                        Val(_CMR), Val(NR), Val(true), Val(1), Val(SB), Val(false), Val(A1), Val(AR), Val(false), Val(false), 0, true)
                elseif full
                    _uker_cmplx!(Cp, ldc, Xp, ldx, ir, Yp, ldy, jr, k, alr, ali, mre, nre,
                        Val(1), Val(NR), Val(true), Val(1), Val(SB), Val(false), Val(A1), Val(AR), Val(false), Val(false), 0, true)
                elseif nrv >= _CMR                                       # diagonal-straddling ⇒ TRI masked store
                    _uker_cmplx!(Cp, ldc, Xp, ldx, ir, Yp, ldy, jr, k, alr, ali, mre, nre,
                        Val(_CMR), Val(NR), Val(true), Val(1), Val(SB), Val(false), Val(A1), Val(AR),
                        Val(false), Val(true), jr - ir, up)
                else
                    _uker_cmplx!(Cp, ldc, Xp, ldx, ir, Yp, ldy, jr, k, alr, ali, mre, nre,
                        Val(1), Val(NR), Val(true), Val(1), Val(SB), Val(false), Val(A1), Val(AR),
                        Val(false), Val(true), jr - ir, up)
                end
                ir += nrv >= _CMR ? mr : W
            end
            jr += NR
        end
    end
    return C
end

# Large real syrk → single-pass packed (gate); complex syrk/herk → unpacked-tri (small trans='N') or
# packed-tri; small → recursion.
function _syrk_blocked!(up::Bool, tr::Bool, herm::Bool, α, A, C, k::Int)
    T = eltype(C)
    if !herm && T <: BlasReal && size(C, 1) > _SYRK_PACK_CUT && k > 0
        return _syrk_packed!(up, tr, convert(T, α), A, C, k)
    elseif T <: Union{ComplexF64, ComplexF32} && k > 0
        n = size(C, 1)
        if !tr && n <= _CSYRK_UNPACK_MAX
            return _ctri_unpacked!(up, herm, α, A, C, k)
        elseif n > (tr ? _CSYRK_PACK_CUT_T : _CSYRK_PACK_CUT)
            return _csyrk_packed!(up, tr, herm, α, A, C, k)
        end
    end
    _syrk_rec!(up, tr, herm, α, A, C, k, _l3_tmp(eltype(C)), 0, size(C, 1))
end
# One gemm sub-block through the @inline `_gemm_core!` (not the non-inlined kwarg gemm!): tr=false ⇒
# C += α·A·Bᵀ (transB), tr=true ⇒ C += α·Aᵀ·B (transA); cc conjugates for the herm (herk) case.
@inline function _syrk_gemm!(C, A, B, α::T, β::T, tr::Bool, cc::Bool) where {T}
    _gemm_core!(C, A, B, α, β, tr, !tr, tr && cc, !tr && cc)
end
# Divide-and-conquer syrk/herk. The recursion carries integer offsets into the ORIGINAL A and C (same
# objects every level — free to pass) instead of fresh sub-block SubArrays, which are non-isbits and
# would heap-box when handed to the non-inlined recursive call. Sub-blocks are materialized as views
# only at the leaf / off-diagonal, feeding the @inline _syrk_gemm!/_add_tri! so they never escape.
# The A block is A's rows for trans='N' (C=A·Aᵀ), columns for trans='T' (C=Aᵀ·A) — built inside an
# `if tr` branch, NOT a `tr ? view(A,:,r) : view(A,r,:)` ternary: the two arms are different SubArray
# types, and merging them makes a non-isbits Union value, which cannot live on the stack and heap-
# boxes every view (this was the residual syrk/herk allocation). One concrete view type per arm stays
# stack-allocated, exactly like the (single-typed) C views.
function _syrk_rec!(up::Bool, tr::Bool, herm::Bool, α, A, C, k::Int, scr, off::Int, n::Int)
    T = eltype(C); a = convert(T, α); cc = herm
    if n <= _SYRK_DBASE
        tmp = view(scr, 1:n, 1:n)
        if tr
            Ab = view(A, :, (off + 1):(off + n))
            _syrk_gemm!(tmp, Ab, Ab, a, zero(T), true, cc)
        else
            Ab = view(A, (off + 1):(off + n), :)
            _syrk_gemm!(tmp, Ab, Ab, a, zero(T), false, cc)
        end
        _add_tri!(view(C, (off + 1):(off + n), (off + 1):(off + n)), tmp, up, herm, n)
        return C
    end
    h = _trsplit(n)
    _syrk_rec!(up, tr, herm, α, A, C, k, scr, off, h)
    _syrk_rec!(up, tr, herm, α, A, C, k, scr, off + h, n - h)
    Co = up ? view(C, (off + 1):(off + h), (off + h + 1):(off + n)) :   # same SubArray type both
              view(C, (off + h + 1):(off + n), (off + 1):(off + h))    # arms — merge is concrete
    if tr
        A1 = view(A, :, (off + 1):(off + h)); A2 = view(A, :, (off + h + 1):(off + n))
        up ? _syrk_gemm!(Co, A1, A2, a, one(T), true, cc) :
             _syrk_gemm!(Co, A2, A1, a, one(T), true, cc)
    else
        A1 = view(A, (off + 1):(off + h), :); A2 = view(A, (off + h + 1):(off + n), :)
        up ? _syrk_gemm!(Co, A1, A2, a, one(T), false, cc) :
             _syrk_gemm!(Co, A2, A1, a, one(T), false, cc)
    end
    return C
end

_syrk_dims(C, A, trans) = (n = size(C, 1); size(C, 2) == n ||
        throw(DimensionMismatch("syrk!: C must be square")); k = trans == 'N' ? size(A, 2) : size(A, 1);
        (trans == 'N' ? size(A, 1) : size(A, 2)) == n || throw(DimensionMismatch("syrk!: op(A) rows ≠ n")); (n, k))

function syrk!(C::AbstractMatrix, A::AbstractMatrix; uplo::Char = 'U', trans::Char = 'N',
        alpha::Number = true, beta::Number = false)
    n, k = _syrk_dims(C, A, trans); up = uplo == 'U'
    _syrk_scaleC!(C, up, beta)
    _syrk_blocked!(up, trans != 'N', false, alpha, A, C, k)
    return C
end
function herk!(C::AbstractMatrix, A::AbstractMatrix; uplo::Char = 'U', trans::Char = 'N',
        alpha::Real = true, beta::Real = false)
    n, k = _syrk_dims(C, A, trans); up = uplo == 'U'
    _syrk_scaleC!(C, up, beta)
    _syrk_blocked!(up, trans != 'N', true, alpha, A, C, k)
    return C
end

# ──────────────────────────────────────────────────────────────────────────────────────────────
# symm/hemm: C := α·A·B + β·C (side 'L', A symmetric/Hermitian n×n) or C := α·B·A + β·C (side 'R',
# A m×m). Only the `uplo` triangle of A is referenced. Diagonal blocks recurse; the off-diagonal A
# block feeds both halves (once as-is, once transposed) via gemm! — the matrix analogue of symv.
@inline function _asym(up::Bool, herm::Bool, A, i, l)
    i == l && return herm ? oftype(A[i, i], real(A[i, i])) : A[i, i]
    _symstored(up, i, l) ? A[i, l] : (herm ? conj(A[l, i]) : A[l, i])
end
# symm's output C is a FULL matrix (no triangle), so symm = gemm with a materialized full symmetric
# A — correct flops (NO 2× waste, unlike syrk). Materialize the symmetric/Hermitian A into a dense
# scratch (O(n²), amortized over the O(n²·m) gemm), then one gemm! carries α and β directly.
const _SYMM_SCR = IdDict{DataType, Matrix}()
function _symm_scr(::Type{T}, n::Int) where {T}
    m = get(_SYMM_SCR, T, nothing)
    if isnothing(m) || size(m, 1) < n; m = Matrix{T}(undef, n, n); _SYMM_SCR[T] = m; end
    return m::Matrix{T}   # the IdDict values are abstract `Matrix` — assert or the view boxes (hemm 160 B)
end
# Const-dispatch the gated types (the IdDict get costs ~130 ns — dominates tiny symm/hemm). Complex too:
# ComplexF64/F32 are the exact types hitting the tiny-n symm/hemm reds, and they were falling through to
# the generic IdDict method above (~130 ns/call). Owned Refs kill that (GKH ownership, no runtime lookup).
const _SYMM_SCR_F64 = Ref(Matrix{Float64}(undef, 0, 0))
const _SYMM_SCR_F32 = Ref(Matrix{Float32}(undef, 0, 0))
const _SYMM_SCR_C64 = Ref(Matrix{ComplexF64}(undef, 0, 0))
const _SYMM_SCR_C32 = Ref(Matrix{ComplexF32}(undef, 0, 0))
@inline function _symm_scr(::Type{Float64}, n::Int)
    m = _SYMM_SCR_F64[]
    size(m, 1) < n && (m = Matrix{Float64}(undef, n, n); _SYMM_SCR_F64[] = m)
    return m
end
@inline function _symm_scr(::Type{Float32}, n::Int)
    m = _SYMM_SCR_F32[]
    size(m, 1) < n && (m = Matrix{Float32}(undef, n, n); _SYMM_SCR_F32[] = m)
    return m
end
@inline function _symm_scr(::Type{ComplexF64}, n::Int)
    m = _SYMM_SCR_C64[]
    size(m, 1) < n && (m = Matrix{ComplexF64}(undef, n, n); _SYMM_SCR_C64[] = m)
    return m
end
@inline function _symm_scr(::Type{ComplexF32}, n::Int)
    m = _SYMM_SCR_C32[]
    size(m, 1) < n && (m = Matrix{ComplexF32}(undef, n, n); _SYMM_SCR_C32[] = m)
    return m
end
# Branch-free symmetric/Hermitian → dense fill: copy the stored triangle (contiguous column segments),
# then mirror it to the other triangle. No per-element uplo/diagonal branch in the hot path.
# Fill each column of Ad CONTIGUOUSLY (stored run + mirror run) so the write stream is sequential
# (was strided mirror writes Ad[j,i]). The stored run is a contiguous column copy (SIMD); the mirror
# reads the crossing row of A (strided load) but writes contiguously — trades strided writes for
# strided reads (prefetchable, no write-combining stalls). herm: mirror conjugates, diagonal → real.
function _symm_materialize!(Ad, up::Bool, herm::Bool, A, n::Int)
    @inbounds if up
        for j in 1:n
            @simd for i in 1:j; Ad[i, j] = A[i, j]; end                     # stored (i≤j): contiguous
            for i in (j + 1):n; Ad[i, j] = herm ? conj(A[j, i]) : A[j, i]; end   # mirror (i>j): col j
        end
    else
        for j in 1:n
            @simd for i in j:n; Ad[i, j] = A[i, j]; end                     # stored (i≥j): contiguous
            for i in 1:(j - 1); Ad[i, j] = herm ? conj(A[j, i]) : A[j, i]; end   # mirror (i<j): col j
        end
    end
    herm && @inbounds for i in 1:n; Ad[i, i] = real(Ad[i, i]); end
    return Ad
end
# SIMD symmetric A-pack (real, dense): tile each FULL row-panel into W×W blocks and classify by the
# diagonal — fully-STORED tiles (all gi≤gp for up) read A directly as vectorized column copies
# (_pack_A_simd! idiom), fully-MIRROR tiles (all gi>gp) read the stored transpose A[gp,i] via the W×W
# register transpose block (_tblk!, the _pack_A_simd_T! idiom), and the lone diagonal tile per W-row-block
# (d=0 when mc/kc are W-aligned, ~1.5% of elements) stays scalar. Block starts (ic,pc mult of W) put every
# crossing tile at d=0; the scalar branch is correct for any d, so misalignment only widens the thin band.
# Kills the 2.7–2.9× scalar-pack tax that sank symm below AOCL (each stored element read ONCE, no materialize).
@inline function _pack_A_sym_simd!(Ap::Vector{T}, A, ic::Int, pc::Int, mce::Int, kce::Int,
        up::Bool, alpha::T, mr::Int, W::Int) where {T<:BlasReal}
    V = Vec{W, T}; sz = sizeof(T); lda = stride(A, 2); sub = mr ÷ W; np = cld(mce, mr)
    GC.@preserve A Ap begin
        Aptr = pointer(A); App = pointer(Ap); av = V(alpha)
        @inbounds for pi in 0:(np - 1)
            base = pi * mr * kce; pbase = pi * mr
            if pbase + mr <= mce                              # full row-panel: wings full-mr, band per-tile
                gtop = ic + pbase; p = 0
                while p + W <= kce
                    gc0 = pc + p; dst0 = App + (base + p * mr) * sz
                    fullstored = up ? (gc0 >= gtop + mr - 1) : (gc0 + W - 1 <= gtop)   # whole mr panel stored
                    fullmirror = up ? (gc0 + W - 1 < gtop)   : (gc0 >= gtop + mr)      # whole mr panel mirror
                    if fullstored                               # A[gi,gp] direct: full-mr column copy
                        for j in 0:(W - 1)
                            src = Aptr + (gtop + (gc0 + j) * lda) * sz; dj = dst0 + j * mr * sz; o = 0
                            while o < mr; vstore(av * vload(V, src + o * sz), dj + o * sz); o += W; end
                        end
                    elseif fullmirror                           # A[gp,gi] transpose: W×W blocks over sub rows
                        for ri in 0:(sub - 1)
                            _tblk!(Ptr{T}(dst0 + ri * W * sz), Ptr{T}(Aptr + ((gtop + ri * W) * lda + gc0) * sz),
                                   lda, mr, av, Val(W))
                        end
                    else                                        # diagonal band (~mr+W wide): per-tile
                        for ri in 0:(sub - 1)
                            rowoff = ri * W; gr0 = gtop + rowoff; dstk = dst0 + rowoff * sz
                            if up ? (gr0 + W - 1 <= gc0) : (gr0 >= gc0 + W - 1)          # stored tile
                                for j in 0:(W - 1)
                                    vstore(av * vload(V, Aptr + (gr0 + (gc0 + j) * lda) * sz), dstk + j * mr * sz)
                                end
                            elseif up ? (gr0 >= gc0 + W) : (gr0 + W <= gc0)              # mirror tile
                                _tblk!(Ptr{T}(dstk), Ptr{T}(Aptr + (gr0 * lda + gc0) * sz), lda, mr, av, Val(W))
                            else                                                         # diagonal tile: scalar
                                for j in 0:(W - 1), l in 0:(W - 1)
                                    gi = gr0 + l; gp = gc0 + j
                                    v = (up ? (gi <= gp) : (gi >= gp)) ? A[gi + 1, gp + 1] : A[gp + 1, gi + 1]
                                    Ap[base + (p + j) * mr + rowoff + l + 1] = alpha * v
                                end
                            end
                        end
                    end
                    p += W
                end
                while p < kce                                    # k tail (kce not mult of W): scalar
                    gp = pc + p
                    for r in 0:(mr - 1)
                        gi = gtop + r
                        v = (up ? (gi <= gp) : (gi >= gp)) ? A[gi + 1, gp + 1] : A[gp + 1, gi + 1]
                        Ap[base + p * mr + r + 1] = alpha * v
                    end
                    p += 1
                end
            else                                                 # partial row-panel → scalar (branchless)
                rhi = min(mr, mce - pbase)
                for p in 0:(kce - 1)
                    gp = pc + p; o = base + p * mr; ls = gp - ic - pbase
                    if up
                        st_end = clamp(ls + 1, 0, rhi)
                        for r in 0:(st_end - 1); Ap[o + r + 1] = alpha * A[ic + pbase + r + 1, gp + 1]; end
                        for r in st_end:(rhi - 1); Ap[o + r + 1] = alpha * A[gp + 1, ic + pbase + r + 1]; end
                    else
                        st_start = clamp(ls, 0, rhi)
                        for r in 0:(st_start - 1); Ap[o + r + 1] = alpha * A[gp + 1, ic + pbase + r + 1]; end
                        for r in st_start:(rhi - 1); Ap[o + r + 1] = alpha * A[ic + pbase + r + 1, gp + 1]; end
                    end
                    for r in rhi:(mr - 1); Ap[o + r + 1] = zero(T); end
                end
            end
        end
    end
    return
end
# Symmetric A-pack for a diagonal-straddling panel (real symm). BRANCHLESS (OpenBLAS-style): per column
# the stored/mirror split is a single crossing, so each column packs a contiguous STORED run (reads A's
# column gp, stride 1) then a MIRROR run (reads A's row gp, stride lda) — no per-element `i≤j` branch in
# the hot loop. Off-diagonal panels use plain _pack_A! (stored: tA=false SIMD; mirror: tA=true).
# Dense real unit-stride with mr a multiple of W → the SIMD tiled pack above (each element read once, SIMD).
function _pack_A_sym!(Ap::Vector{T}, A, ic::Int, pc::Int, mce::Int, kce::Int, up::Bool, alpha::T, mr::Int) where {T}
    W = _vwidth(T)
    if T <: BlasReal && _strided1(A) && mr % W == 0
        return _pack_A_sym_simd!(Ap, A, ic, pc, mce, kce, up, alpha, mr, W)
    end
    np = cld(mce, mr)
    @inbounds for pi in 0:(np - 1)
        base = pi * mr * kce; pbase = pi * mr
        rhi = min(mr, mce - pbase)                 # valid rows r ∈ [0,rhi); r ≥ rhi → pad zero
        for p in 0:(kce - 1)
            gp = pc + p; o = base + p * mr; ls = gp - ic - pbase    # local diagonal crossing (in r)
            if up                                  # stored r ∈ [0,st_end) (gi≤gp), mirror r ∈ [st_end,rhi)
                st_end = clamp(ls + 1, 0, rhi)
                for r in 0:(st_end - 1); Ap[o + r + 1] = alpha * A[ic + pbase + r + 1, gp + 1]; end
                for r in st_end:(rhi - 1); Ap[o + r + 1] = alpha * A[gp + 1, ic + pbase + r + 1]; end
            else                                   # stored r ∈ [st_start,rhi) (gi≥gp), mirror r ∈ [0,st_start)
                st_start = clamp(ls, 0, rhi)
                for r in 0:(st_start - 1); Ap[o + r + 1] = alpha * A[gp + 1, ic + pbase + r + 1]; end
                for r in st_start:(rhi - 1); Ap[o + r + 1] = alpha * A[ic + pbase + r + 1, gp + 1]; end
            end
            for r in rhi:(mr - 1); Ap[o + r + 1] = zero(T); end     # pad rows beyond mce
        end
    end
    return
end
# Complex HERMITIAN A-pack (split re/im) for a diagonal-straddling OR full-mirror panel (hemm side-L):
# per panel-column the stored run reads A[i,gp] direct; the MIRROR run reads A[gp,i] CONJUGATED
# (A_herm[i,gp] = conj(A[gp,i])). No α (applied at the microkernel store). Mirrors _pack_A_sym! + the
# conj that makes it Hermitian. Full-stored panels use the SIMD _pack_A_cmplx! (tA=false) instead.
function _pack_A_sym_cmplx!(ApR::Vector{T}, ApI::Vector{T}, A, ic::Int, pc::Int, mce::Int, kce::Int,
        up::Bool, mr::Int) where {T}
    np = cld(mce, mr)
    @inbounds for pi in 0:(np - 1)
        base = pi * mr * kce; pbase = pi * mr; rhi = min(mr, mce - pbase)
        for p in 0:(kce - 1)
            gp = pc + p; o = base + p * mr; ls = gp - ic - pbase
            if up                                          # stored r∈[0,st_end); mirror(conj) r∈[st_end,rhi)
                st_end = clamp(ls + 1, 0, rhi)
                for r in 0:(st_end - 1)
                    v = A[ic + pbase + r + 1, gp + 1]; ApR[o + r + 1] = real(v); ApI[o + r + 1] = imag(v)
                end
                for r in st_end:(rhi - 1)
                    v = A[gp + 1, ic + pbase + r + 1]; ApR[o + r + 1] = real(v); ApI[o + r + 1] = -imag(v)
                end
            else                                           # mirror(conj) r∈[0,st_start); stored r∈[st_start,rhi)
                st_start = clamp(ls, 0, rhi)
                for r in 0:(st_start - 1)
                    v = A[gp + 1, ic + pbase + r + 1]; ApR[o + r + 1] = real(v); ApI[o + r + 1] = -imag(v)
                end
                for r in st_start:(rhi - 1)
                    v = A[ic + pbase + r + 1, gp + 1]; ApR[o + r + 1] = real(v); ApI[o + r + 1] = imag(v)
                end
            end
            for r in rhi:(mr - 1); ApR[o + r + 1] = zero(T); ApI[o + r + 1] = zero(T); end
        end
    end
    return
end
# Single-pass packed complex hemm (side-L): C := α·A_herm·B + β·C. Standard packed complex gemm (all C
# tiles) but each A-panel is packed from the Hermitian TRIANGLE on the fly (stored → SIMD _pack_A_cmplx!;
# mirror/straddle → _pack_A_sym_cmplx! with conj) — reads the triangle ONCE, no materialize, no 2×
# A-traffic. α at the microkernel store (A1=false); β·C up front. Reuses _microkernel_cmplx!.
function _hemm_packed_L!(up::Bool, α, β, A, B, C)
    Tc = eltype(C); T = real(Tc); n = size(C, 1); m = size(C, 2); W = _vwidth(T); mr = _CMR * W; nr = _CNR
    kc = min(_CKC, n)
    mc = _at_mc_kc(_HW, eltype(C), kc, mr, cld(n, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(m, nr) * nr)
    ApR, ApI, BpR, BpI = _gemm_scratch_cmplx(T, cld(mc, mr) * mr * kc, cld(nc, nr) * nr * kc)
    _scale_C!(C, n, m, convert(Tc, β)); ldc = stride(C, 2); sz = sizeof(T)
    alr = real(convert(Tc, α)); ali = imag(convert(Tc, α))
    GC.@preserve C ApR ApI BpR BpI begin
        Cp0 = Ptr{T}(pointer(C)); ARp = pointer(ApR); AIp = pointer(ApI); BRp = pointer(BpR); BIp = pointer(BpI)
        jc = 0
        while jc < m
            nce = min(nc, m - jc); pc = 0
            while pc < n
                kce = min(kc, n - pc)
                _pack_B_cmplx!(BpR, BpI, B, pc, jc, kce, nce, false, nr)
                ic = 0
                while ic < n
                    mce = min(mc, n - ic); a_hi = ic + mce - 1; p_hi = pc + kce - 1
                    stored = up ? (a_hi <= pc) : (ic >= p_hi)     # read A[i,gp] direct (SIMD)
                    stored ? _pack_A_cmplx!(ApR, ApI, A, ic, pc, mce, kce, false, mr) :
                             _pack_A_sym_cmplx!(ApR, ApI, A, ic, pc, mce, kce, up, mr)  # mirror/straddle (conj)
                    jr = 0
                    while jr < nce
                        nre = min(nr, nce - jr); ir = 0
                        while ir < mce
                            mre = min(mr, mce - ir)
                            AR = Ptr{T}(ARp + div(ir, mr) * mr * kce * sz); AI = Ptr{T}(AIp + div(ir, mr) * mr * kce * sz)
                            BR = Ptr{T}(BRp + div(jr, nr) * nr * kce * sz); BI = Ptr{T}(BIp + div(jr, nr) * nr * kce * sz)
                            Cblk = Cp0 + (2 * (ic + ir) + 2 * (jc + jr) * ldc) * sz
                            if mre == mr && nre == nr
                                _microkernel_cmplx!(Cblk, ldc, AR, AI, BR, BI, kce, alr, ali,
                                    Val(_CMR), Val(_CNR), Val(1), Val(1), Val(false), Val(false))
                            else
                                _microkernel_cmplx_masked!(Cblk, ldc, AR, AI, BR, BI, kce, alr, ali,
                                    mre, nre, Val(_CMR), Val(_CNR), Val(1), Val(1), Val(false), Val(false))
                            end
                            ir += mr
                        end
                        jr += nr
                    end
                    ic += mc
                end
                pc += kc
            end
            jc += nc
        end
    end
    return C
end
# SIMD symmetric B-pack (real, dense, side R): tile each FULL col-panel into W×W blocks. Roles SWAP vs
# side L — the STORED read A[gp,gj] is row-strided (a row of A) → W×W register transpose block (_tblk!,
# stride nr); the MIRROR read A[gj,gp] is a contiguous column of A → per-k column copy. Diagonal tile
# (d=0 when kc/nc are W-aligned) scalar. Each stored element read once (no α — α rides the left operand).
@inline function _pack_B_sym_simd!(Bp::Vector{T}, A, pc::Int, jc::Int, kce::Int, nce::Int,
        up::Bool, nr::Int, W::Int) where {T<:BlasReal}
    V = Vec{W, T}; sz = sizeof(T); lda = stride(A, 2); cn = nr ÷ W; np = cld(nce, nr)
    GC.@preserve A Bp begin
        Aptr = pointer(A); Bpp = pointer(Bp); ov = V(one(T))
        @inbounds for ji in 0:(np - 1)
            base = ji * nr * kce; cbase = ji * nr
            if cbase + nr <= nce                              # full col-panel: wings full-nr, band per-tile
                gjtop = jc + cbase; p = 0
                while p + W <= kce
                    gp0 = pc + p; blk = base + p * nr
                    fullstored = up ? (gp0 + W - 1 <= gjtop) : (gp0 >= gjtop + nr - 1)  # whole nr panel stored
                    fullmirror = up ? (gp0 >= gjtop + nr)    : (gp0 + W - 1 < gjtop)     # whole nr panel mirror
                    if fullstored                               # A[gp,gj] row-strided: transpose blocks
                        for ci in 0:(cn - 1)
                            _tblk!(Ptr{T}(Bpp + (blk + ci * W) * sz),
                                   Ptr{T}(Aptr + (gp0 + (gjtop + ci * W) * lda) * sz), lda, nr, ov, Val(W))
                        end
                    elseif fullmirror                           # A[gj,gp] column: per-k column copy
                        for kk in 0:(W - 1), ci in 0:(cn - 1)
                            vstore(vload(V, Aptr + (gjtop + ci * W + (gp0 + kk) * lda) * sz),
                                   Bpp + (blk + kk * nr + ci * W) * sz)
                        end
                    else                                        # diagonal band: per-tile
                        for ci in 0:(cn - 1)
                            coff = ci * W; gj0 = gjtop + coff
                            if up ? (gp0 + W - 1 <= gj0) : (gp0 >= gj0 + W - 1)           # stored tile
                                _tblk!(Ptr{T}(Bpp + (blk + coff) * sz),
                                       Ptr{T}(Aptr + (gp0 + gj0 * lda) * sz), lda, nr, ov, Val(W))
                            elseif up ? (gp0 >= gj0 + W) : (gp0 + W <= gj0)               # mirror tile
                                for kk in 0:(W - 1)
                                    vstore(vload(V, Aptr + (gj0 + (gp0 + kk) * lda) * sz),
                                           Bpp + (blk + kk * nr + coff) * sz)
                                end
                            else                                                          # diagonal tile: scalar
                                for kk in 0:(W - 1), cc in 0:(W - 1)
                                    gp = gp0 + kk; gj = gj0 + cc
                                    v = (up ? (gp <= gj) : (gp >= gj)) ? A[gp + 1, gj + 1] : A[gj + 1, gp + 1]
                                    Bp[blk + kk * nr + coff + cc + 1] = v
                                end
                            end
                        end
                    end
                    p += W
                end
                while p < kce                                    # k tail (kce not mult of W): scalar
                    gp = pc + p
                    for c in 0:(nr - 1)
                        gj = gjtop + c
                        v = (up ? (gp <= gj) : (gp >= gj)) ? A[gp + 1, gj + 1] : A[gj + 1, gp + 1]
                        Bp[base + p * nr + c + 1] = v
                    end
                    p += 1
                end
            else                                                 # partial col-panel → scalar (branchless)
                chi = min(nr, nce - cbase)
                for p in 0:(kce - 1)
                    gp = pc + p; o = base + p * nr; ls = gp - jc - cbase
                    if up
                        st = clamp(ls, 0, chi)
                        for c in 0:(st - 1); Bp[o + c + 1] = A[jc + cbase + c + 1, gp + 1]; end
                        for c in st:(chi - 1); Bp[o + c + 1] = A[gp + 1, jc + cbase + c + 1]; end
                    else
                        st = clamp(ls + 1, 0, chi)
                        for c in 0:(st - 1); Bp[o + c + 1] = A[gp + 1, jc + cbase + c + 1]; end
                        for c in st:(chi - 1); Bp[o + c + 1] = A[jc + cbase + c + 1, gp + 1]; end
                    end
                    for c in chi:(nr - 1); Bp[o + c + 1] = zero(T); end
                end
            end
        end
    end
    return
end
# Symmetric B-pack for a diagonal-straddling panel (real symm side R): the symmetric matrix is the
# gemm's RIGHT operand. Stored side reads A[gp,gj], mirror side A[gj,gp]. Off-diagonal panels use
# plain _pack_B! (stored: tB=false; mirror: tB=true). No α here — α rides on the left operand's pack.
# Dense real unit-stride with nr a multiple of W → the SIMD tiled pack above (each element read once).
function _pack_B_sym!(Bp::Vector{T}, A, pc::Int, jc::Int, kce::Int, nce::Int, up::Bool, nr::Int) where {T}
    W = _vwidth(T)
    if T <: BlasReal && _strided1(A) && nr % W == 0
        return _pack_B_sym_simd!(Bp, A, pc, jc, kce, nce, up, nr, W)
    end
    np = cld(nce, nr)                              # branchless (OpenBLAS-style): stored/mirror = one crossing
    @inbounds for ji in 0:(np - 1)
        base = ji * nr * kce; cbase = ji * nr
        chi = min(nr, nce - cbase)                 # valid cols c ∈ [0,chi); c ≥ chi → pad zero
        for p in 0:(kce - 1)
            gp = pc + p; o = base + p * nr; ls = gp - jc - cbase
            if up                                  # stored gj≥gp: c ∈ [st,chi) (A row gp, strided); mirror c<st (A col gp)
                st = clamp(ls, 0, chi)
                for c in 0:(st - 1); Bp[o + c + 1] = A[jc + cbase + c + 1, gp + 1]; end
                for c in st:(chi - 1); Bp[o + c + 1] = A[gp + 1, jc + cbase + c + 1]; end
            else                                   # stored gj≤gp: c ∈ [0,st) (A row gp, strided); mirror c≥st (A col gp)
                st = clamp(ls + 1, 0, chi)
                for c in 0:(st - 1); Bp[o + c + 1] = A[gp + 1, jc + cbase + c + 1]; end
                for c in st:(chi - 1); Bp[o + c + 1] = A[jc + cbase + c + 1, gp + 1]; end
            end
            for c in chi:(nr - 1); Bp[o + c + 1] = zero(T); end
        end
    end
    return
end

# Single-pass packed symm (side L, real): C := α·A_sym·B + β·C as one gemm, packing A's symmetric
# panels directly (no n² materialize). M=n, N=m, K=n; classify each A-panel: stored / mirror / straddle.
function _symm_packed_L!(up::Bool, α::T, β::T, A, B, C) where {T<:BlasReal}
    n = size(C, 1); m = size(C, 2); W = _vwidth(T); mr = _MR * W; nr = _NR
    kc = min(_KC, n); mc = _at_mc_kc(_HW, T, kc, mr, cld(n, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(m, nr) * nr)
    db = _strided1(B)                                    # op(B)=B unit row-stride ⇒ read B direct (no B-pack)
    Ap, Bp = _gemm_scratch(T, cld(mc, mr) * mr * kc, db ? 0 : cld(nc, nr) * nr * kc)
    b0 = iszero(β); b0 || _scale_C!(C, n, m, β)   # β=0 ⇒ first kc-block overwrites (skip scale + C RMW)
    ldc = stride(C, 2); ldb = db ? stride(B, 2) : 0; sz = sizeof(T)
    GC.@preserve C Ap Bp B begin
        Cp0 = pointer(C); App = pointer(Ap); Bpp = pointer(Bp); Bp0 = db ? pointer(B) : Bpp
        jc = 0
        while jc < m
            nce = min(nc, m - jc); pc = 0
            while pc < n
                kce = min(kc, n - pc); ow = b0 && pc == 0
                db || _pack_B!(Bp, B, pc, jc, kce, nce, false, nr)
                ic = 0
                while ic < n
                    mce = min(mc, n - ic); a_hi = ic + mce - 1; p_hi = pc + kce - 1
                    stored = up ? (a_hi <= pc) : (ic >= p_hi)
                    mirror = up ? (ic > p_hi) : (a_hi < pc)
                    stored ? _pack_A!(Ap, A, ic, pc, mce, kce, false, α, mr) :
                        mirror ? _pack_A!(Ap, A, ic, pc, mce, kce, true, α, mr) :
                                 _pack_A_sym!(Ap, A, ic, pc, mce, kce, up, α, mr)
                    jr = 0
                    while jr < nce
                        nre = min(nr, nce - jr); ir = 0
                        while ir < mce
                            mre = min(mr, mce - ir)
                            Apanel = App + (div(ir, mr) * mr * kce) * sz
                            Cblk = Cp0 + ((ic + ir) + (jc + jr) * ldc) * sz
                            if db
                                Bc = Bp0 + (pc + (jc + jr) * ldb) * sz
                                _db_tile!(ow, Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bc), ldb, kce, mre, nre, Val(_MR), Val(_NR), Val(W))
                            else
                                Bpanel = Bpp + (div(jr, nr) * nr * kce) * sz
                                if mre == mr && nre == nr
                                    ow ? _microkernel!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, Val(_MR), Val(_NR), Val(true)) :
                                         _microkernel!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, Val(_MR), Val(_NR), Val(false))
                                else
                                    ow ? _microkernel_masked!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, mre, nre, Val(_MR), Val(_NR), Val(true)) :
                                         _microkernel_masked!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, mre, nre, Val(_MR), Val(_NR), Val(false))
                                end
                            end
                            ir += mr
                        end
                        jr += nr
                    end
                    ic += mc
                end
                pc += kc
            end
            jc += nc
        end
    end
    return C
end

# Single-pass packed symm (side R, real): C := α·B·A_sym + β·C. A_sym is the gemm's RIGHT operand.
# M=size(C,1), N=K=n; classify each A_sym panel (pc..K, jc..N): stored / mirror / straddle.
function _symm_packed_R!(up::Bool, α::T, β::T, B, A, C) where {T<:BlasReal}
    M = size(C, 1); n = size(A, 1); W = _vwidth(T); mr = _MR * W; nr = _NR
    kc = min(_KC, n); mc = _at_mc_kc(_HW, T, kc, mr, cld(M, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(n, nr) * nr)
    Ap, Bp = _gemm_scratch(T, cld(mc, mr) * mr * kc, cld(nc, nr) * nr * kc)
    b0 = iszero(β); b0 || _scale_C!(C, M, n, β)   # β=0 ⇒ first kc-block overwrites (skip scale + C RMW)
    ldc = stride(C, 2); sz = sizeof(T)
    GC.@preserve C Ap Bp begin
        Cp0 = pointer(C); App = pointer(Ap); Bpp = pointer(Bp)
        jc = 0
        while jc < n
            nce = min(nc, n - jc); j_hi = jc + nce - 1; pc = 0
            while pc < n
                kce = min(kc, n - pc); p_hi = pc + kce - 1; ow = b0 && pc == 0
                stored = up ? (p_hi <= jc) : (pc >= j_hi)
                mirror = up ? (pc > j_hi) : (p_hi < jc)
                stored ? _pack_B!(Bp, A, pc, jc, kce, nce, false, nr) :
                    mirror ? _pack_B!(Bp, A, pc, jc, kce, nce, true, nr) :
                             _pack_B_sym!(Bp, A, pc, jc, kce, nce, up, nr)
                ic = 0
                while ic < M
                    mce = min(mc, M - ic)
                    _pack_A!(Ap, B, ic, pc, mce, kce, false, α, mr)
                    jr = 0
                    while jr < nce
                        nre = min(nr, nce - jr); ir = 0
                        while ir < mce
                            mre = min(mr, mce - ir)
                            Apanel = App + (div(ir, mr) * mr * kce) * sz
                            Bpanel = Bpp + (div(jr, nr) * nr * kce) * sz
                            Cblk = Cp0 + ((ic + ir) + (jc + jr) * ldc) * sz
                            if mre == mr && nre == nr
                                ow ? _microkernel!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, Val(_MR), Val(_NR), Val(true)) :
                                     _microkernel!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, Val(_MR), Val(_NR), Val(false))
                            else
                                ow ? _microkernel_masked!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, mre, nre, Val(_MR), Val(_NR), Val(true)) :
                                     _microkernel_masked!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, mre, nre, Val(_MR), Val(_NR), Val(false))
                            end
                            ir += mr
                        end
                        jr += nr
                    end
                    ic += mc
                end
                pc += kc
            end
            jc += nc
        end
    end
    return C
end

# n above which symm uses the single-pass packed kernel (branchless symmetric pack) vs materialize+gemm.
# On AVX2 the branchless pack makes packed win from n=128 (0.97 vs materialize 0.95); AVX-512 unchanged
# (materialize ≤ _GEMM_UNPACK_MAX, already gates). Overridable "symm_pack_cut".
const _SYMM_PACK_CUT = @load_preference("symm_pack_cut", _vwidth(Float64) == 4 ? 96 : _GEMM_UNPACK_MAX)::Int
# n above which complex hemm side-L uses the packed Hermitian kernel (reads the triangle once, on-the-fly
# conj-mirror pack). The packed path is the OLD classic-4M kernel (measured 0.85-0.90 at n=64-128 AVX2);
# the materialize path routes to _gemm_core!'s Karatsuba-3M at mid-n — the SAME path complex symm already
# gates on (zsymm n=128 = 1.06). So on AVX2 raise the cut past the whole gate range: hemm rides
# materialize+3M like symm. (AVX-512 keeps 32 — 3M path is AVX2-only; leave the gating classic path.)
const _CHEMM_PACK_CUT = @load_preference("chemm_pack_cut", _vwidth(Float64) == 4 ? 4096 : 32)::Int
function _symm!(side_left::Bool, up::Bool, herm::Bool, α, β, A, B, C)
    n = size(A, 1)
    # Complex side-L in the 3M window → fuse the reflection into the 3M A-split (no materialize, no n²
    # complex scratch). Deletes the materialize tax that dominated mid-n hemm/symm. Concrete-complex only
    # (generic T<:Number / AD path falls through to materialize+_gemm_core!); AVX2-gated via _CGEMM_3M.
    # (n≤40 tried a direct triangle sweep — measured SLOWER than materialize: its many small _uker calls +
    # branchy panel fill cost more than one gemm at tiny n. Reverted; tiny-n stays on materialize.)
    if side_left && eltype(C) <: BlasComplex && _CGEMM_3M && _strided1(A) && _strided1(B) && _strided1(C)
        m2 = size(B, 2)
        if _CGEMM_3M_MIN <= max(n, m2) <= _CGEMM_3M_MAX && min(n, m2) >= _CGEMM_3M_KMIN
            return _hemm_3m_L!(up, herm, α, β, A, B, C)
        end
    end
    # Real, above the pack cut: packed single-pass kernel (each stored A element read once, SIMD pack) —
    # UNLESS the resulting gemm is in the Strassen regime. There materialize+_gemm_core! captures the
    # 7-mult recursion whose flop saving beats the O(n²) copy tax (measured n=2048: packed 0.99× vs
    # materialize+Strassen 1.10× AOCL; n=1024 ~parity). Below Strassen_min, packed wins (no copy tax).
    if !herm && eltype(C) <: BlasReal && n > _SYMM_PACK_CUT &&
            !(_STRASSEN && (side_left ? _strassen_depth(n, size(B, 2), n) :
                                        _strassen_depth(size(B, 1), n, n)) > 0)
        return side_left ?
            _symm_packed_L!(up, convert(eltype(C), α), convert(eltype(C), β), A, B, C) :
            _symm_packed_R!(up, convert(eltype(C), α), convert(eltype(C), β), B, A, C)
    elseif herm && eltype(C) <: BlasComplex && side_left && n > _CHEMM_PACK_CUT &&
           _strided1(B) && _strided1(C)                     # packed Hermitian (no materialize, triangle once)
        return _hemm_packed_L!(up, α, β, A, B, C)
    end
    Ad = view(_symm_scr(eltype(C), n), 1:n, 1:n)
    _symm_materialize!(Ad, up, herm, A, n)
    T = eltype(C); aT = convert(T, α); bT = convert(T, β)  # straight to the dispatch core, both real &
    side_left ? _gemm_core!(C, Ad, B, aT, bT, false, false, false, false) :  # complex — skip the kwarg layer
                _gemm_core!(C, B, Ad, aT, bT, false, false, false, false)
    return C
end
function _symm_check(side_left, A, B, C)
    (size(C) == size(B)) || throw(DimensionMismatch("symm!: C and B must match"))
    k = side_left ? size(B, 1) : size(B, 2)
    (size(A, 1) == size(A, 2) == k) || throw(DimensionMismatch("symm!: A must be $k×$k"))
end
function symm!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix; side::Char = 'L',
        uplo::Char = 'U', alpha::Number = true, beta::Number = false)
    _symm_check(side == 'L', A, B, C); _symm!(side == 'L', uplo == 'U', false, alpha, beta, A, B, C); C
end
function hemm!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix; side::Char = 'L',
        uplo::Char = 'U', alpha::Number = true, beta::Number = false)
    _symm_check(side == 'L', A, B, C); _symm!(side == 'L', uplo == 'U', true, alpha, beta, A, B, C); C
end

# ──────────────────────────────────────────────────────────────────────────────────────────────
# syr2k/her2k: C := α·op(A)·op(B)ᴴ + (α or ᾱ)·op(B)·op(A)ᴴ + β·C, only `uplo` triangle of C.
# trans 'N': A,B are n×k (A·Bᴴ + B·Aᴴ). 'T'/'C': A,B are k×n (Aᴴ·B + Bᴴ·A). her2k: conj + β real +
# diagonal real, second term uses ᾱ. Diagonal blocks recurse; off-diagonal = two gemm!s.
# gemm-temp base (the gate path): both rank-k products into an n×n temp, then triangle-add.
# Same 0-alloc shape as _syrk_rec!: integer `off,n` into the ORIGINAL A/B/C (no fresh sub-block
# SubArrays through the non-inlined recursive call), views materialized only at the leaf /
# off-diagonal inside `if tr` branches (one concrete SubArray type per arm — the merged
# `tr ? view(A,:,r) : view(A,r,:)` ternary is a non-isbits Union that heap-boxes), and all gemms
# through the @inline _syrk_gemm! (not the non-inlined kwarg gemm!).
function _syr2k_acc!(up::Bool, tr::Bool, herm::Bool, α, A, B, C, k::Int, scr, off::Int, n::Int)
    T = eltype(C); a = convert(T, α); a2 = herm ? conj(a) : a; cc = herm
    if n <= _SYRK_DBASE
        r = (off + 1):(off + n); tmp = view(scr, 1:n, 1:n)
        # ONE product M = α·op(A)·op(B)ᴴ (her2k conjugates via cc; syr2k/real = plain transpose), then a
        # symmetrized triangle-add for the 2nd product op(B)op(A)ᴴ = Mᴴ (Mᵀ for syr2k/real): since
        # conj(M[j,i]) = ᾱ·(op(B)op(A)ᴴ)[i,j], the add is C += M[i,j] + conj(M[j,i]) (her2k) / M[j,i]
        # (syr2k). Halves the base gemm work vs the old two-`_syrk_gemm!` path — now the complex base
        # too, not just real (was the tiny-n zsyr2k/zher2k red). her2k diagonal → real (M + conj(M)).
        if tr
            Ab = view(A, :, r); Bb = view(B, :, r); _syrk_gemm!(tmp, Ab, Bb, a, zero(T), true, cc)
        else
            Ab = view(A, r, :); Bb = view(B, r, :); _syrk_gemm!(tmp, Ab, Bb, a, zero(T), false, cc)
        end
        Cd = view(C, r, r)
        if herm
            @inbounds for j in 1:n, i in (up ? (1:j) : (j:n)); Cd[i, j] += tmp[i, j] + conj(tmp[j, i]); end
            @inbounds for i in 1:n; Cd[i, i] = real(Cd[i, i]); end   # clear rounding imag on the diagonal
        else
            @inbounds for j in 1:n, i in (up ? (1:j) : (j:n)); Cd[i, j] += tmp[i, j] + tmp[j, i]; end
        end
        return C           # NOT `return _add_tri!(...)`: that returns the SubArray, making the
    end                    # recursion's return type Union{Matrix,SubArray} — boxes at every level
    h = _trsplit(n)
    _syr2k_acc!(up, tr, herm, α, A, B, C, k, scr, off, h)
    _syr2k_acc!(up, tr, herm, α, A, B, C, k, scr, off + h, n - h)
    Co = up ? view(C, (off + 1):(off + h), (off + h + 1):(off + n)) :   # same SubArray type both
              view(C, (off + h + 1):(off + n), (off + 1):(off + h))    # arms — merge is concrete
    # up: C12 += α·op(A1)op(B2)ᴴ + α2·op(B1)op(A2)ᴴ; low: C21 += α·op(A2)op(B1)ᴴ + α2·op(B2)op(A1)ᴴ
    if tr
        A1 = view(A, :, (off + 1):(off + h)); A2 = view(A, :, (off + h + 1):(off + n))
        B1 = view(B, :, (off + 1):(off + h)); B2 = view(B, :, (off + h + 1):(off + n))
        if up
            _syrk_gemm!(Co, A1, B2, a, one(T), true, cc); _syrk_gemm!(Co, B1, A2, a2, one(T), true, cc)
        else
            _syrk_gemm!(Co, A2, B1, a, one(T), true, cc); _syrk_gemm!(Co, B2, A1, a2, one(T), true, cc)
        end
    else
        A1 = view(A, (off + 1):(off + h), :); A2 = view(A, (off + h + 1):(off + n), :)
        B1 = view(B, (off + 1):(off + h), :); B2 = view(B, (off + h + 1):(off + n), :)
        if up
            _syrk_gemm!(Co, A1, B2, a, one(T), false, cc); _syrk_gemm!(Co, B1, A2, a2, one(T), false, cc)
        else
            _syrk_gemm!(Co, A2, B1, a, one(T), false, cc); _syrk_gemm!(Co, B2, A1, a2, one(T), false, cc)
        end
    end
    return C
end
function _syr2k_dims(C, A, B, trans)
    n = size(C, 1); (size(C, 2) == n && size(A) == size(B)) || throw(DimensionMismatch("syr2k!: shapes"))
    k = trans == 'N' ? size(A, 2) : size(A, 1)
    (trans == 'N' ? size(A, 1) : size(A, 2)) == n || throw(DimensionMismatch("syr2k!: op(A) rows ≠ n"))
    (n, k)
end
# n above which syr2k uses the single-pass fused packed kernel (else the gemm-temp recursion). On AVX2
# _GEMM_UNPACK_MAX (128) was too high — n=128 fell to recursion (gate 0.95) while the packed kernel gates
# (n=128 0.95→1.01). Cut to 96 there (n≥128 packed, ≤64 recursion). AVX-512 unchanged (already gates).
# Overridable "syr2k_pack_cut".
const _SYR2K_PACK_CUT = @load_preference("syr2k_pack_cut", _vwidth(Float64) == 4 ? 96 : _GEMM_UNPACK_MAX)::Int
# Complex syr2k/her2k: n above which the two-product tri-output packed kernel beats the gemm-temp
# recursion (which computes a dense n×n temp per diagonal block — the 2× waste). Tuned per machine.
const _CSYR2K_PACK_CUT = @load_preference("csyr2k_pack_cut", _vwidth(Float64) == 4 ? 8 : 8)::Int
function syr2k!(C::AbstractMatrix, A::AbstractMatrix, Bm::AbstractMatrix; uplo::Char = 'U',
        trans::Char = 'N', alpha::Number = true, beta::Number = false)
    n, k = _syr2k_dims(C, A, Bm, trans); up = uplo == 'U'
    if eltype(C) <: BlasReal && n > _SYR2K_PACK_CUT && k > 0
        _syr2k_packed!(up, trans != 'N', convert(eltype(C), alpha), convert(eltype(C), beta), A, Bm, C, k)
    elseif eltype(C) <: BlasComplex && trans == 'N' && 0 < n <= _CSYRK_UNPACK_MAX && k > 0
        _syrk_scaleC!(C, up, beta)                                     # small-n trans='N': unpacked-tri (2 products)
        _ctri2_unpacked!(up, false, alpha, A, Bm, C, k)
    elseif eltype(C) <: BlasComplex && n > _CSYR2K_PACK_CUT && k > 0
        _syrk_scaleC!(C, up, beta)
        _csyr2k_packed!(up, trans != 'N', false, alpha, A, Bm, C, k)
    else
        _syrk_scaleC!(C, up, beta)
        _syr2k_acc!(up, trans != 'N', false, alpha, A, Bm, C, k, _l3_tmp(eltype(C)), 0, n)
    end
    C
end
function her2k!(C::AbstractMatrix, A::AbstractMatrix, Bm::AbstractMatrix; uplo::Char = 'U',
        trans::Char = 'N', alpha::Number = true, beta::Real = false)
    n, k = _syr2k_dims(C, A, Bm, trans); up = uplo == 'U'
    _syrk_scaleC!(C, up, beta)
    if eltype(C) <: BlasComplex && trans == 'N' && 0 < n <= _CSYRK_UNPACK_MAX && k > 0
        return (_ctri2_unpacked!(up, true, alpha, A, Bm, C, k); C)     # small-n trans='N': unpacked-tri (2 products)
    elseif eltype(C) <: BlasComplex && n > _CSYR2K_PACK_CUT && k > 0
        return (_csyr2k_packed!(up, trans != 'N', true, alpha, A, Bm, C, k); C)
    end
    _syr2k_acc!(up, trans != 'N', true, alpha, A, Bm, C, k, _l3_tmp(eltype(C)), 0, n); C
end
