# BLAS-3 beyond gemm: trmm/trsm (triangular ·/solve), and later syrk/herk/syr2k/her2k/symm/hemm.
# Strategy — recursive 2×2 blocking that reuses the gate-passing `gemm!` for the off-diagonal update
# and bottoms out (block ≤ _TRMM_BASE) in the L2 kernels (trmv/trsv per B-column for side L;
# column-axpy/solve for side R). This is the L3 analogue of the trsv/trmv "diagonal block + gemv"
# decomposition: gemm carries the flops, the small triangular base carries the structure. α is applied
# as a final scale (kept out of the recursion). Generic `T<:Number` path via the L2 generic kernels.

const _TRMM_BASE = _L3_NB     # ≤ this → _trmm_small! directly (MUST be ≤ _L3_NB M scratch; coupled)
# PDM: Literal — trmm side-R panel width. NOW A KNOB (was a bare const, unpinnable and untunable); default is the value it always had. | tune: FLAT — 64..1024 within noise on Zen3+Zen4+Zen5 except two non-replicating cells <=2.5% (2026-08-21)
const _TRMM_RPANEL = @load_preference("trmm_rpanel", 512)::Int
@inline _trmm_rpanel() = (f = _FKR_trmm_rpanel[]; f >= 0 ? f : _TRMM_RPANEL)
# side-R packed kc: the triangular B-micropanel (nr=_NR wide, kc deep) is ½·L1 resident — the SAME
# residency criterion as gemm's _KC (identical nr), so derive it from _KC rather than a hand-fit literal
# (was 384 = ¾·L1, a req#8 violation). Preferences "trmm_rkc" pins it if trmm-R measures a different opt.
# PDM: Literal — own k-block, borrows gemm's _KC; a triangular operand packs differently. | tune: candidate
const _TRMM_RKC = @load_preference("trmm_rkc", _KC)::Int
@inline _fh_trmm_rkc() = (f = _FKR_trmm_rkc[]; f >= 0 ? f : _TRMM_RKC)
# side-R packed cut: n > this → packed single-pass side-R, else direct/unpacked. NOT
# register-capacity-governed: the plausible "= gemm's _GEMM_UNPACK_MAX (=2·_acc_cap)" derivation was
# FLEET-FALSIFIED. Measured direct-vs-packed (boost-locked, both boxes): the crossover is µarch-dependent
# and SOFT — packed decisively wins only at large n (Zen3 n≥512 up to 17%; Zen4 n≥1536 ~3-6%), with a
# wide tied band below and DIRECT winning the small/mid band on both (Zen4 n=256/384 direct gates 0.93/0.96
# while packed MISSES at 1.00/1.01). So the AVX2 value 96 regresses BOTH boxes (it packs the direct-favoring
# small/mid band) and gains nothing large-n (both pack there). 448 keeps small/mid on the winning direct
# path and packs the large-n region → fleet-best of the tested values. (The crossover appears to scale UP
# with cache — Zen3 L2=512K→~500, Zen4 L2=1M→~1200 — a candidate future SCALES derivation, but too soft/noisy
# to pin without a dedicated 2-box crossover campaign; a guessed formula would be worse than this literal.)
# Preferences "trmm_rpack" pins it if a future box measures otherwise.
# PDM: Literal — measured pack threshold; a box that disagrees pins it rather than editing. | tune: candidate
const _TRMM_RPACK = @load_preference("trmm_rpack", _at_trmm_rpack(_HW))::Int
@inline _fh_trmm_rpack() = (f = _FKR_trmm_rpack[]; f >= 0 ? f : _TRMM_RPACK)

# trmm side-L real: k above this uses the single-pass K-trimmed PACKED routine; at or below it, the
# recursion over `gemm!`. Its own constant as of task #134 — it used to read `_GEMM_UNPACK_MAX`, which is
# gemm's UNPACKED-vs-BLOCKED crossover (2·(nvreg−4)·W = 448 on the AVX-512 boxes, 96 on AVX2). That is a
# different kernel PAIR on a different shape family, and it had never been measured at THIS boundary: a
# borrowed threshold, i.e. a routing predicate carrying no PDM discipline at all.
#
# PDM tier: MEASURE-CALIBRATED, not DERIVE — an earlier revision claimed DERIVE and that was overstated.
# The falsification test below (one multiplier must land inside both boxes' located intervals) has no
# real power: it only rejects bases whose Zen4:Zen3 constant ratio falls outside (4, 16), which is all
# three rejections have in common. A second basis survives that the test does not reject —
# `m·_NVREG·W²` gives m ∈ [0.5, 1) on BOTH boxes, a WIDER common window than the one chosen — so the
# data cannot identify the basis and this is a two-point fit with a free parameter. Worse, Zen5's
# detected consts equal Zen4's, so the third fleet box cannot discriminate them either; only a machine
# with a different register file or width can (on M5 the two predict ~280 vs ~64-128). Compare
# `_TRMM_RPACK` above, which records a FLEET-FALSIFIED derivation for the directly analogous side-R
# crossover and concludes "a guessed formula would be worse than this literal" — no argument has been
# given for why side-L's crossover is sharp where side-R's was not. Treat the formula as a calibrated
# value with the regret condition stated at the end, and re-open as Measure tier if a third µarch misses.
# The value: `5·_GEMM_UNPACK_MAX ÷ 2`, i.e. gemm's own crossover scaled by a single fleet-wide
# coefficient. Zen3 → 240, Zen4/Zen5 → 1120. Scaling off `_GEMM_UNPACK_MAX` is not arbitrary: the route
# this predicate chooses BETWEEN is recursion-over-`gemm!` versus packed trmm, so the competing path is
# literally gemm and its unpacked/blocked crossover (itself derived from `_NVREG` and `W`) is the right
# scale. A box we have never benchmarked therefore gets a crossover that tracks its register file and
# vector width, which is the whole point of req#8.
#
# HOW THE COEFFICIENT WAS OBTAINED, including what would falsify it. Gate-shape A/B on both boxes (full
# `plots.jl` trmm sweep, shipped cut vs cut pinned past every size, PB-vs-PB medians, same commit) LOCATES
# the crossover on each — packed and recursion swap winner between two adjacent gate sizes:
#     Zen4  512 0.961 · 1024 0.952 (recursion faster)  |  2048 1.007 · 4096 1.005 (null)  ⇒ C ∈ [1024, 2048)
#     Zen3  128 0.997 (null)  |  256 1.043 · 512 1.037 · 1024 1.016 (packed faster)       ⇒ C ∈ [128, 256)
# Requiring one multiplier `m` to land inside BOTH intervals is a falsifiable test, and it falsifies
# three of the four candidate bases outright — `m·W²` needs [8,16) vs [16,32), `m·_NVREG·W` needs [2,4)
# vs [4,8), `m·L2_KiB` needs [0.25,0.5) vs [1,2): all disjoint. Only `m·_GEMM_UNPACK_MAX` admits a common
# window, m ∈ [2.286, 2.667); 5/2 sits near its middle, so both boxes keep margin to the edges.
#
# HONEST LIMITS, so this is not read as stronger than it is: `m` is FITTED to two boxes, not predicted,
# and the third (Zen5) was offline and could not be checked — its ISA consts equal Zen4's, so it inherits
# 1120 untested. The intervals are one gate size wide, so `m`'s window is coarse. If a future box's
# located crossover falls outside `m·_GEMM_UNPACK_MAX`, this is the wrong basis and the knob is Measure
# tier after all — pin `trmm_pack_min` on that box and re-open #134 rather than nudging the coefficient
# to fit three points, which is how a derivation becomes a lookup table with extra steps.
#
# DO NOT re-derive this from the in-process probe: it put the crossover near 3072 and called n=2048 a
# 2.5% win for recursion, where the gate says null. `_measure_gemvt_nc`'s lesson — a probe disagreeing
# with the gate is evidence about the PROBE — and the reason its own duel is disabled.
# PDM: Derived — 5/2 x _GEMM_UNPACK_MAX, i.e. it follows gemm's own unpack bound. | tune: n/a, follows gemm
const _TRMM_PACK_MIN = @load_preference("trmm_pack_min", (5 * _GEMM_UNPACK_MAX) ÷ 2)::Int
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
            @simd for i in 1:(lo - 1)
                M[i, j] = zero(T)
            end
            @simd for i in lo:hi
                M[i, j] = A[i, j]
            end
            @simd for i in (hi + 1):k
                M[i, j] = zero(T)
            end
            unit && (M[j, j] = one(T))
        end
    else                                         # T: transpose-on-store (strided source, scalar)
        for j in 1:k
            lo = up ? j : 1; hi = up ? k : j     # M column j = op(A) col j = A row j, stored part
            @simd for i in 1:(lo - 1)
                M[i, j] = zero(T)
            end
            for i in lo:hi
                M[i, j] = A[j, i]
            end
            @simd for i in (hi + 1):k
                M[i, j] = zero(T)
            end
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
# PDM: Literal — wide-SIMD-safe default for the direct path; per-box override without a code push. | tune: candidate
const _TRMM_DDIRECT = @load_preference("trmm_ddirect", 4)
@inline _fh_trmm_ddirect() = (f = _FKR_trmm_ddirect[]; f >= 0 ? f : _TRMM_DDIRECT)
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
                        _microkernel_unpacked!(
                            Bp, ldb, Ap, ldM, ir, Bp + plo * sz, ldb, jr, kc,
                            one(T), zero(T), Val(_MR), Val(_NR), Val(false), Val(true)
                        )
                    elseif nre == nr && rem(mre, W) == 0 && div(mre, W) <= 2
                        # W-aligned partial rows → UNMASKED clipped kernel, the ladder gemm already
                        # ships (gemm.jl:1167-1196: "a smaller Val(vr) reads exactly the mre live rows
                        # — no mask, no wasted vector"). The old binary `cld(mre,W)==1 ? Val(1) :
                        # Val(_MR)` only had an exact answer for one row-vector: on AVX2 (_MR=3, W=4) a
                        # k=8 trmm has mre=8 ⇒ masked Val(3), i.e. 12 rows computed for 8 live ones —
                        # 50% wasted FMAs, and at n=8 that is 100% of the operation (Zen3 trmm n=8 was
                        # 0.88 vs OB while Zen4, where _MR=2/W=8 lands on the exact Val(1), got 1.32).
                        # mre ≤ mr = _MR·W and mre == mr is handled above, so vr < _MR here ⇒ vr ∈ {1,2}
                        # for every current µarch; a wider _MR falls through to the masked path unharmed.
                        # Aliasing-safe by this function's own rule (:140-143): full-strip kernels hold
                        # the tile in registers between read and store; only `_edge!` is row-serial.
                        if div(mre, W) == 1
                            _microkernel_unpacked!(
                                Bp, ldb, Ap, ldM, ir, Bp + plo * sz, ldb, jr, kc,
                                one(T), zero(T), Val(1), Val(_NR), Val(false), Val(true)
                            )
                        else
                            _microkernel_unpacked!(
                                Bp, ldb, Ap, ldM, ir, Bp + plo * sz, ldb, jr, kc,
                                one(T), zero(T), Val(2), Val(_NR), Val(false), Val(true)
                            )
                        end
                    elseif nre == nr
                        _microkernel_unpacked_mrows!(
                            Bp, ldb, Ap, ldM, ir, Bp + plo * sz, ldb, jr, kc,
                            one(T), zero(T), mre, cld(mre, W) == 1 ? Val(1) : Val(_MR),
                            Val(_NR), Val(false), Val(true), Val(_vwidth(T))
                        )
                    elseif upM
                        _microkernel_unpacked_edge!(
                            Bp, ldb, Ap, ldM, ir, Bp + plo * sz, ldb, jr, kc,
                            one(T), zero(T), mre, nre, false, true
                        )
                    else
                        Ec = _trsm_tmp(T, _L3_NB, nr)    # kc×nre source copy (dodges the serial aliasing)
                        lde = size(Ec, 1)
                        GC.@preserve Ec begin
                            Ep = pointer(Ec)
                            @inbounds for j in 0:(nre - 1), p in 0:(kc - 1)
                                unsafe_store!(Ep, unsafe_load(Bp, plo + p + (jr + j) * ldb + 1), p + j * lde + 1)
                            end
                            _microkernel_unpacked_edge!(
                                Bp, ldb, Ap, ldM, ir, Ep - jr * lde * sz, lde, jr, kc,
                                one(T), zero(T), mre, nre, false, true
                            )
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
                        _microkernel_unpacked!(
                            Bp, ldb, Bp + plo * ldb * sz, ldb, ir, Bsp, ldM, jr, kc,
                            one(T), zero(T), Val(_MR), Val(_NR), Val(false), Val(true)
                        )
                    elseif nre == nr && rem(mre, W) == 0 && div(mre, W) <= 2
                        # W-aligned partial rows → unmasked clipped kernel (same ladder and same reasoning
                        # as the side-L site above; vr ∈ {1,2} since mre < mr here).
                        if div(mre, W) == 1
                            _microkernel_unpacked!(
                                Bp, ldb, Bp + plo * ldb * sz, ldb, ir, Bsp, ldM, jr, kc,
                                one(T), zero(T), Val(1), Val(_NR), Val(false), Val(true)
                            )
                        else
                            _microkernel_unpacked!(
                                Bp, ldb, Bp + plo * ldb * sz, ldb, ir, Bsp, ldM, jr, kc,
                                one(T), zero(T), Val(2), Val(_NR), Val(false), Val(true)
                            )
                        end
                    elseif nre == nr
                        _microkernel_unpacked_mrows!(
                            Bp, ldb, Bp + plo * ldb * sz, ldb, ir, Bsp, ldM, jr, kc,
                            one(T), zero(T), mre, cld(mre, W) == 1 ? Val(1) : Val(_MR),
                            Val(_NR), Val(false), Val(true), Val(_vwidth(T))
                        )
                    else
                        # Edge kernel is COLUMN-serial and the A-operand is B itself: column j+1's
                        # contraction re-reads columns already stored (they're inside [plo,phi) on both
                        # uplos). Compute the strip into a dest scratch, copy back after.
                        Ec = _trsm_tmp(T, mr, nr); lde = size(Ec, 1)
                        GC.@preserve Ec begin
                            Ep = pointer(Ec)
                            _microkernel_unpacked_edge!(
                                Ep - (ir + jr * lde) * sz, lde,
                                Bp + plo * ldb * sz, ldb, ir, Bsp, ldM, jr, kc,
                                one(T), zero(T), mre, nre, false, true
                            )
                            @inbounds for j in 0:(nre - 1), r in 0:(mre - 1)
                                unsafe_store!(
                                    Bp, unsafe_load(Ep, r + j * lde + 1),
                                    ir + r + (jr + j) * ldb + 1
                                )
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
                # `Val{FULL}` (9th Val) gates MASKED vs unmasked A-loads inside the k-loop and masked vs
                # unmasked stores in the epilogue. `_uker_sweep!` (gemm.jl) dispatches Val(true) whenever
                # `mre == mr`; this driver used to hardwire Val(false) on BOTH branches, so every tile ran
                # masked even when full — and at the failing gate sizes mr divides k, so EVERY tile is
                # full and every mask is pure waste. `full` above is a different predicate: it only picks
                # the tile HEIGHT (Val(_CMR) vs Val(1)), never the masking.
                # Costly precisely where it hurt: on AVX2 a masked op is `vmaskmovpd` (expensive on AMD),
                # on AVX-512 it is k-register predication (~free) — which matches the measured split,
                # ztrmm/ztrmmR n=32 being 0.821/0.781 on Zen3 against 0.994/1.087 on Zen4.
                # Settled by an in-process ABBA A/B (both arms one process, bit-identical output),
                # 6 rounds: Zen3 +16.7%/+13.1% at n=8/32 with the packed-path rows an exact 1.000
                # control; Zen4 +0.4%/+3.6%/+3.8%/+2.7% at n=8/32/48/128 (no control row there —
                # `_CTRMM_PACK` is `_vwidth==4`, so AVX-512 routes EVERY size through this driver).
                if mre == mr                                     # full-height tile → unmasked
                    _uker_cmplx!(
                        Bp, ldb, Ap, ldM, ir, Bs, ldb, jr, kc, onr, zr, mre, nre,
                        Val(_CMR), Val(_CNR_SMALL), Val(false), Val(1), Val(1), Val(true), Val(true), Val(false), Val(true), Val(false), 0, true
                    )
                elseif full
                    _uker_cmplx!(
                        Bp, ldb, Ap, ldM, ir, Bs, ldb, jr, kc, onr, zr, mre, nre,
                        Val(_CMR), Val(_CNR_SMALL), Val(false), Val(1), Val(1), Val(true), Val(true), Val(false), Val(false), Val(false), 0, true
                    )
                else
                    _uker_cmplx!(
                        Bp, ldb, Ap, ldM, ir, Bs, ldb, jr, kc, onr, zr, mre, nre,
                        Val(1), Val(_CNR_SMALL), Val(false), Val(1), Val(1), Val(true), Val(true), Val(false), Val(false), Val(false), 0, true
                    )
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
                # See the matching note in `_trmm_cmplx_small_L!`: the 9th Val is FULL (unmasked
                # A-loads + unmasked stores), `_uker_sweep!` sets it whenever `mre == mr`, and this
                # driver used to hardwire it false so every tile ran masked even when full.
                if mre == mr                                     # full-height tile → unmasked
                    _uker_cmplx!(
                        Bp, ldb, Aop, ldb, ir, Bop, ldM, jr, kc, onr, zr, mre, nre,
                        Val(_CMR), Val(_CNR_SMALL), Val(false), Val(1), Val(1), Val(true), Val(true), Val(false), Val(true), Val(false), 0, true
                    )
                elseif full
                    _uker_cmplx!(
                        Bp, ldb, Aop, ldb, ir, Bop, ldM, jr, kc, onr, zr, mre, nre,
                        Val(_CMR), Val(_CNR_SMALL), Val(false), Val(1), Val(1), Val(true), Val(true), Val(false), Val(false), Val(false), 0, true
                    )
                else
                    _uker_cmplx!(
                        Bp, ldb, Aop, ldb, ir, Bop, ldM, jr, kc, onr, zr, mre, nre,
                        Val(1), Val(_CNR_SMALL), Val(false), Val(1), Val(1), Val(true), Val(true), Val(false), Val(false), Val(false), 0, true
                    )
                end
            end
            ir += mr
        end
    end
    return B
end

# The packed K-TRIM complex trmm base (near-peak PACKED microkernel) vs the weak unpacked _uker_cmplx!.
# Measured (Zen3): the packed complex kernel hits 0.94–0.95×OB at these short-k base shapes, the
# unpacked only 0.68–0.73, and ztrmm is pinned at the unpacked ceiling (0.77 ≈ 0.73). AVX-512 already
# gates via the unpacked path (32 zmm give ample ILP) — restrict packed to AVX2 (W=4) so that gate is
# untouched. Preferences knob "ctrmm_pack". Fable-designed, decomposition-confirmed 2026-07-05.
# PDM: Derived — formula over detected consts: `_vwidth(Float64) == 4`
const _CTRMM_PACK = @load_preference("ctrmm_pack", _vwidth(Float64) == 4)::Bool
# Below this k the packed base's pack overhead loses to the unpacked K-TRIM (measured Zen3: k=8 0.46 vs
# unpacked ~1.0, k=32 0.75 vs 0.85; crossover ≈48, k=64 packed 0.91 wins). Recursion bases (k>128 split)
# land ≥64 → packed; only tiny single-base trmm stays unpacked. Preferences knob "ctrmm_pack_min".
# pack-vs-unpacked crossover: below this k the unpacked small kernel's lower setup beats the packed path's
# O(k²) M-materialize. Empirically side-DEPENDENT (packed_R amortizes ~4·_CNR, packed_L not until ~8·_CNR
# — the B-vs-M packing asymmetry) and small-n stays sub-gate either way, so a single derived crossover
# doesn't pay; kept at the measured conservative value. Preferences-pinnable. (req#8: acknowledged debt —
# the pack-amortization threshold resists a clean cache/ISA formula; revisit with an OB-style fused pack.)
# PDM: Literal — complex trmm pack threshold. | tune: candidate
const _CTRMM_PACK_MIN = @load_preference("ctrmm_pack_min", 48)::Int
@inline _fh_ctrmm_pack_min() = (f = _FKR_ctrmm_pack_min[]; f >= 0 ? f : _CTRMM_PACK_MIN)

# Exact-width remainder column-tile for the packed complex trmm bases. The last column-tile of a
# non-multiple-of-nr panel is partial (width nre∈1:_CNR-1); running it through the nr-wide masked kernel
# computes (nr-nre) PAD columns — and for upper-N that tile sits at MAX K-trim depth (kc=k), so the pad
# is charged at full depth (measured Zen3 ztrmmR spike). Dispatch the runtime nre to a compile-time
# Val{NR}=nre masked kernel so the pad columns are NEVER computed. REQUIRES the slot packed at row-stride
# nre (see packed_R pack loop). B0=A1=true (overwrite; α folded outside). AVX2-only (packed path gated by
# _CTRMM_PACK=W==4); the 5-way branch is compile cost there, never instantiated on AVX-512.
@inline function _trmm_rem_cmplx!(
        C::Ptr{T}, ldc::Int, AR::Ptr{T}, AI::Ptr{T}, BR::Ptr{T}, BI::Ptr{T},
        kc::Int, alr::T, ali::T, mre::Int, nre::Int, ::Val{MR}, ::Val{SA}, ::Val{SB}
    ) where {T, MR, SA, SB}
    return if nre == 1
        _microkernel_cmplx_masked!(
            C, ldc, AR, AI, BR, BI, kc, alr, ali, mre, nre,
            Val(MR), Val(1), Val(SA), Val(SB), Val(true), Val(true)
        )
    elseif nre == 2
        _microkernel_cmplx_masked!(
            C, ldc, AR, AI, BR, BI, kc, alr, ali, mre, nre,
            Val(MR), Val(2), Val(SA), Val(SB), Val(true), Val(true)
        )
    elseif nre == 3
        _microkernel_cmplx_masked!(
            C, ldc, AR, AI, BR, BI, kc, alr, ali, mre, nre,
            Val(MR), Val(3), Val(SA), Val(SB), Val(true), Val(true)
        )
    elseif nre == 4
        _microkernel_cmplx_masked!(
            C, ldc, AR, AI, BR, BI, kc, alr, ali, mre, nre,
            Val(MR), Val(4), Val(SA), Val(SB), Val(true), Val(true)
        )
    else                                                     # nre == 5
        _microkernel_cmplx_masked!(
            C, ldc, AR, AI, BR, BI, kc, alr, ali, mre, nre,
            Val(MR), Val(5), Val(SA), Val(SB), Val(true), Val(true)
        )
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
                            _microkernel_cmplx!(
                                Cblk, ldb, AR, AI, BR, BI, kc, alr, ali,
                                Val(_CMR), Val(_CNR), Val(1), Val(1), Val(true), Val(true)
                            )
                        else
                            _microkernel_cmplx_masked!(
                                Cblk, ldb, AR, AI, BR, BI, kc, alr, ali,
                                mre, nre, Val(_CMR), Val(_CNR), Val(1), Val(1), Val(true), Val(true)
                            )
                        end
                    else                                     # partial column-tile: stride-rem slot ⇒ plo*rem
                        boffr = (ji * nr * k + plo * rem) * sz
                        _trmm_rem_cmplx!(
                            Cblk, ldb, AR, AI, Ptr{T}(BRp + boffr), Ptr{T}(BIp + boffr),
                            kc, alr, ali, mre, nre, Val(_CMR), Val(1), Val(1)
                        )
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
    # makes mc a moving target that over-blocks small-L2 boxes at small n (Zen3 ztrmmR regression). req#8.
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
                            _microkernel_cmplx!(
                                Cblk, ldb, AR, AI, BR, BI, kc, alr, ali,
                                Val(_CMR), Val(_CNR), Val(1), Val(1), Val(true), Val(true)
                            )
                        else
                            _microkernel_cmplx_masked!(
                                Cblk, ldb, AR, AI, BR, BI, kc, alr, ali,
                                mre, nre, Val(_CMR), Val(_CNR), Val(1), Val(1), Val(true), Val(true)
                            )
                        end
                    else                                     # partial column-tile (last, nre∈1:_CNR-1): NR=nre
                        _trmm_rem_cmplx!(
                            Cblk, ldb, AR, AI, BR, BI, kc, alr, ali, mre, nre,
                            Val(_CMR), Val(1), Val(1)
                        )       # kernel → no pad cols computed (upper-N deep-rem)
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
        return k <= _fh_trmm_ddirect() ? _trmm_dense_L!(up, tr, unit, A, B) :
            _trmm_small!(true, up, tr, unit, A, B)
    elseif eltype(B) <: BlasComplex && k <= _TRMM_BASE     # complex: K-TRIM small kernel (half flops);
        return !_strided1(B) ? _trmm_cmplx_base_L!(up, tr, cj, unit, k, A, B) :            # strided B → base
            (_CTRMM_PACK && k >= _fh_ctrmm_pack_min()) ? _trmm_cmplx_packed_L!(up, tr, cj, unit, k, A, B) :
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
            for i in 1:(j - 1)
                _axpy_col!(B, j, coef(i, j), i, m)
            end
        end
    else
        for j in 1:k
            unit || _scal_col!(B, j, coef(j, j), m)
            for i in (j + 1):k
                _axpy_col!(B, j, coef(i, j), i, m)
            end
        end
    end
    return B
end
# B[:,j] *= s   and   B[:,j] += a·B[:,i]  on contiguous columns (SIMD where eligible).
@inline function _scal_col!(B, j, s, m)
    return if _strided1(B) && eltype(B) <: BlasReal
        GC.@preserve B (_scal_simd_ptr!(pointer(B) + (j - 1) * stride(B, 2) * sizeof(eltype(B)), m, s))
    else
        @inbounds for r in 1:m
            B[r, j] *= s
        end
    end
end
@inline function _axpy_col!(B, j, a, i, m)
    return if _strided1(B) && eltype(B) <: BlasReal
        T = eltype(B); sz = sizeof(T); ldb = stride(B, 2)
        GC.@preserve B _axpy_simd!(m, T(a), pointer(B) + (i - 1) * ldb * sz, pointer(B) + (j - 1) * ldb * sz)
    else
        @inbounds for r in 1:m
            B[r, j] += a * B[r, i]
        end
    end
end
@inline function _scal_simd_ptr!(p::Ptr{T}, n::Int, s::T) where {T <: BlasReal}
    return _scal!(n, s, p, 1)   # _scal! accepts a pointer (level1)
end

# Packed single-pass trmm side-R: B := B·op(A) as ONE K-trimmed blocked-gemm sweep (mirror of
# _trmm_packed!). B is copied once to a contiguous scratch (kills the in-place aliasing outright,
# O(mk) ≪ O(mk²/2)); op(A) packs per pc-block into nr-panels with zeros outside the triangle and only
# the rows each column-tile actually contracts (per-tile K-trim at nr granularity — the trim that kept
# the flat/recursion versions from gemm efficiency lived at panel granularity). Real non-conj.
const _TRMM_BCR = Ref(Matrix{Float64}(undef, 0, 0))
function _pack_B_triR!(
        Bp::Vector{T}, A, pc::Int, kce::Int, k::Int, upM::Bool, tr::Bool,
        unit::Bool, nr::Int
    ) where {T}
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
function _trmm_packedR!(up::Bool, tr::Bool, unit::Bool, A, B, ::Type{T}) where {T <: BlasReal}
    m, k = size(B); W = _vwidth(T); mr = _MR * W; nr = _NR
    upM = (up != tr)
    kc = min(_fh_trmm_rkc(), k); mc = _at_mc_kc(_HW, T, kc, mr, cld(m, mr) * mr)
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
                _pack_A!(
                    view(Apf, (off + 1):(off + cld(mce, mr) * mr * kce)), B, ic, pc, mce, kce,
                    false, one(T), mr
                )
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
        return k <= _fh_trmm_ddirect() ? _trmm_right_base!(up, tr, cj, unit, k, A, B) :
            _trmm_small!(false, up, tr, unit, A, B)
    elseif _strided1(B) && eltype(B) === Float64 && !cj && k > _fh_trmm_rpack()
        return _trmm_packedR!(up, tr, unit, A, B, Float64)
    elseif eltype(B) <: BlasReal && !cj
        # FLAT panel loop: each _TRMM_RPANEL-column panel of B gets ONE fat off-diagonal gemm on a
        # STORED rectangular A-view (transB carries op; no materialize) + a diagonal solved by the
        # halving recursion (→ _trmm_small! bases). Big panels keep the gemms fat (skinny n=128 gemms
        # measured 0.85 at 2048); the flat level touches B only twice. Diagonal FIRST (consumes the
        # panel's ORIGINAL values), then += off-diagonal (reads other, still-original panels).
        # upM → right-to-left; lower → left-to-right.
        upM = (up != tr); P = _trmm_rpanel()
        np = cld(k, P)
        for t in (upM ? ((np - 1):-1:0) : (0:(np - 1)))
            jc = t * P; pc = min(P, k - jc)
            Bpan = view(B, :, (jc + 1):(jc + pc))
            Adia = view(A, (jc + 1):(jc + pc), (jc + 1):(jc + pc))
            _trmm_right_recur!(up, tr, cj, unit, Adia, Bpan)
            if upM && jc > 0                     # off-diag: Bpan += B[:,1:jc]·op(A)[1:jc, jc+1:jc+pc]
                Ablk = tr ? view(A, (jc + 1):(jc + pc), 1:jc) : view(A, 1:jc, (jc + 1):(jc + pc))
                _gemm_core!(
                    Bpan, view(B, :, 1:jc), Ablk, one(eltype(B)), one(eltype(B)),
                    false, tr, false, false
                )
            elseif !upM && jc + pc < k           # off-diag: Bpan += B[:,jc+pc+1:k]·op(A)[jc+pc+1:k, …]
                Ablk = tr ? view(A, (jc + 1):(jc + pc), (jc + pc + 1):k) :
                    view(A, (jc + pc + 1):k, (jc + 1):(jc + pc))
                _gemm_core!(
                    Bpan, view(B, :, (jc + pc + 1):k), Ablk, one(eltype(B)), one(eltype(B)),
                    false, tr, false, false
                )
            end
        end
        return B
    elseif eltype(B) <: BlasComplex && k <= _TRMM_BASE     # complex: K-TRIM kernels (mirror side-L).
        return !_strided1(B) ? _trmm_cmplx_base_R!(up, tr, cj, unit, k, A, B) :            # strided B → base
            (_CTRMM_PACK && k >= _fh_ctrmm_pack_min()) ? _trmm_cmplx_packed_R!(up, tr, cj, unit, k, A, B) :
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
            return k <= _fh_trmm_ddirect() ? _trmm_right_base!(up, tr, cj, unit, k, A, B) :
                _trmm_small!(false, up, tr, unit, A, B)
        elseif eltype(B) <: BlasComplex
            # (the old "side-R packed regresses" note was a routing-bug artifact: the 0.24 was the scalar
            # column-axpy base @_trmm_right!, not a packed kernel — packed_R didn't exist yet.)
            return !_strided1(B) ? _trmm_cmplx_base_R!(up, tr, cj, unit, k, A, B) :
                (_CTRMM_PACK && k >= _fh_ctrmm_pack_min()) ? _trmm_cmplx_packed_R!(up, tr, cj, unit, k, A, B) :
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
    # α==0 ⇒ B := 0 with A not referenced (reference ?trmm). α is applied AFTER the product below, so
    # without this an Inf/NaN anywhere in A or in the unset B becomes NaN·0 = NaN rather than 0.
    # Not applied to AD types — see the matching note in `_gemm_core!` on `iszero(::Dual)`.
    if eltype(B) <: Union{BlasReal, BlasComplex} && iszero(α)
        fill!(B, zero(eltype(B)))
        return B
    end
    if side_left
        _trmm_left!(up, tr, cj, unit, A, B)
    else
        _trmm_right!(up, tr, cj, unit, A, B)
    end
    isone(α) || _scal_all!(B, α)
    return B
end
@inline function _scal_all!(B, α)
    return if _strided1(B)
        αT = convert(eltype(B), α); m = size(B, 1); n = size(B, 2); ld = stride(B, 2)
        GC.@preserve B begin
            p = pointer(B)
            if ld == m
                _scal!(m * n, αT, p, 1)                        # fully contiguous — one shot
            else                                               # padded ld: scale each column (skip the gap)
                sz = sizeof(eltype(B))
                for j in 0:(n - 1)
                    _scal!(m, αT, p + j * ld * sz, 1)
                end
            end
        end
    else
        B .*= α
    end
end

# Pack a triangular op(A) panel: zero the non-stored half, write the diagonal (unit ⇒ 1). packed_upper
# = the packed op(A) is upper-triangular (zero where gi>gp). Used only for diagonal-straddling A-panels
# (off-diagonal panels are fully stored → plain _pack_A!, fully-zero panels are skipped by the driver).
function _pack_A_tri!(
        Ap::Vector{T}, A, ic::Int, pc::Int, mce::Int, kce::Int, tA::Bool, unit::Bool,
        packed_upper::Bool, alpha::T, mr::Int
    ) where {T}
    if !tA && _strided1(A) && T <: BlasReal
        return _pack_A_tri_simd!(Ap, A, ic, pc, mce, kce, unit, packed_upper, alpha, mr)
    end
    np = cld(mce, mr)
    @inbounds for pi in 0:(np - 1)
        base = pi * mr * kce
        # Same k-range prune (and same inverted _EXP13 disable) as the SIMD path — keep the two in step.
        mre = min(mr, mce - pi * mr); r0g = ic + pi * mr; prune = !(@inbounds _EXPFLAG[_EXP13])
        plo = (prune && packed_upper) ? max(0, r0g - pc) : 0
        phi = (prune && !packed_upper) ? (min(kce, r0g + mre - pc) - 1) : (kce - 1)
        for p in plo:phi
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
@inline function _pack_A_tri_simd!(
        Ap::Vector{T}, A, ic::Int, pc::Int, mce::Int, kce::Int, unit::Bool,
        packed_upper::Bool, alpha::T, mr::Int
    ) where {T <: BlasReal}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T); lda = stride(A, 2); MR = mr ÷ W
    np = cld(mce, mr); lanes = Vec(ntuple(i -> i - 1, Val(W)))
    GC.@preserve A Ap begin
        Aptr = pointer(A); App = pointer(Ap); av = V(alpha); zv = zero(V)
        @inbounds for pi in 0:(np - 1)
            base = pi * mr * kce; r0g = ic + pi * mr; full = pi * mr + mr <= mce
            # PRUNE THE UNREFERENCED k-RANGE — BLIS's `bli_trmm_prune_unref_mparts_k`. The consume loop
            # already confines this panel to [plo, cnt): lower reads p < cnt, upper reads p >= plo.
            # Everything outside that gets packed (as zeros) and then NEVER read — for a straddling
            # block that is about HALF the panel, i.e. pure write traffic. Compute-side trimming was
            # measured free (2026-08-17); the packing is the part that was still paying.
            # DANGER: these bounds must stay identical to the consume side's plo/cnt in _trmm_packed!.
            # Widen them there without widening here and the kernel reads STALE SCRATCH, not zeros —
            # Ap is reused across calls, so the old code's zero-fill was load-bearing.
            # `_EXPFLAG[_EXP13]` is INVERTED: the prune SHIPS ON and the flag restores the old
            # full-kce zero-fill, so both arms are reachable in ONE process (cross-run is not
            # adjudicable, and a fleet A/B on this must be same-process).
            mre = min(mr, mce - pi * mr); prune = !(@inbounds _EXPFLAG[_EXP13])
            plo = (prune && packed_upper) ? max(0, r0g - pc) : 0
            phi = (prune && !packed_upper) ? (min(kce, r0g + mre - pc) - 1) : (kce - 1)
            for p in plo:phi
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
function _trmm_packed!(up::Bool, tr::Bool, unit::Bool, α::T, A, B, ::Val{MRV} = Val(_MR)) where {T <: BlasReal, MRV}
    m = size(B, 1); n = size(B, 2); W = _vwidth(T); mr = MRV * W; nr = _NR
    packed_upper = (up != tr)
    kc = min(_KC, m)
    # EXPERIMENT (_EXPINT[5], 0 = off): a trmm-SPECIFIC kc — the AOCL/BLIS `bli_trmm_determine_kc` arm.
    # At gate sizes the derived kc covers the WHOLE triangle in a single pc-block, so every tile packs
    # through _pack_A_tri! and half the packed-A buffer is zero-fill. A smaller kc turns most of the
    # triangle into fully-dense `stored` blocks (cheap _pack_A!, no zero-fill) plus wholly-skipped
    # zpanels. This is a PACK-TRAFFIC / footprint lever, NOT the flop lever falsified on 2026-08-17 —
    # the zero FMAs themselves measured free, so only the packing and residency can still be paying.
    ov = @inbounds _EXPINT[5]
    ov > 0 && (kc = min(max(ov, W), m))
    @inbounds _EXPINT[6] = kc                    # witness: the kc this call actually ran with
    mc = _at_mc_kc(_HW, T, kc, mr, cld(m, mr) * mr)
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

# Strassen split driver for large side-L real trmm. Blocks the triangular dim k and routes the OFF-DIAGONAL
# update through gemm! — which fires Strassen (≥ _STRASSEN_MIN) for the flop cut that lets PB beat AOCL and
# that the single-pass _trmm_packed! cannot use. Diagonal halves recurse, bottoming out in _trmm_packed!
# below 2·_STRASSEN_MIN (where the off-diagonal gemm's min-dim falls under the Strassen threshold, so the
# split stops paying off). In-place safe: the half whose update READS the other half runs LAST — top-first
# for upper-N / lower-T, bottom-first for lower-N / upper-T. UNSCALED (trmm! applies α once at the top).
# req#8: keyed on _STRASSEN_MIN (no literal). Measured Zen4 vs AOCL: n=2048 0.925→0.963, n=4096 0.929→1.001
# (the strassen-3m-gemm lesson — a flop reduction on the gating base kernel).
function _trmm_split_L!(up::Bool, tr::Bool, unit::Bool, A, B, mrv::Val)
    k = size(B, 1); T = eltype(B)
    k < 2 * _fh_strassen_min() && return _trmm_packed!(up, tr, unit, one(T), A, B, mrv)
    h = k ÷ 2
    A11 = view(A, 1:h, 1:h); A22 = view(A, (h + 1):k, (h + 1):k)
    off = up ? view(A, 1:h, (h + 1):k) : view(A, (h + 1):k, 1:h)     # upper→A12, lower→A21
    Bt = view(B, 1:h, :); Bb = view(B, (h + 1):k, :); ta = tr ? 'T' : 'N'
    if (up && !tr) || (!up && tr)              # top block carries the off-diagonal update → after Bt's solve
        _trmm_split_L!(up, tr, unit, A11, Bt, mrv)
        gemm!(Bt, off, Bb; transA = ta, alpha = true, beta = true)   # Bt += op(off)·Bb  (Strassen)
        _trmm_split_L!(up, tr, unit, A22, Bb, mrv)
    else                                        # bottom block carries the update
        _trmm_split_L!(up, tr, unit, A22, Bb, mrv)
        gemm!(Bb, off, Bt; transA = ta, alpha = true, beta = true)   # Bb += op(off)·Bt  (Strassen)
        _trmm_split_L!(up, tr, unit, A11, Bt, mrv)
    end
    return B
end

# Public: B := α·op(A)·B (side 'L') or α·B·op(A) (side 'R'); A k×k triangular (uplo/transA/diag).
function trmm!(
        B::AbstractMatrix, A::AbstractMatrix; side::Char = 'L', uplo::Char = 'U',
        transA::Char = 'N', diag::Char = 'N', alpha::Number = true
    )
    sl = side == 'L'
    k = sl ? size(B, 1) : size(B, 2)
    (size(A, 1) == size(A, 2) == k) || _throw_square(:trmm!, k)
    # α==0 ⇒ B := 0, A not referenced (reference ?trmm). Placed HERE, above the dispatch, because all
    # four branches below apply α only after forming the product — the tiny real/complex bypasses and
    # the split-L path each route around `_trmm!` and would need the same guard individually.
    if eltype(B) <: Union{BlasReal, BlasComplex} && iszero(alpha)
        fill!(B, zero(eltype(B)))
        return B
    end
    # NON-UNIT-STRIDE OUTPUT: stage through a contiguous copy.
    #
    # Every real path below -- the tiny bases, `_trmm_split_L!` and the `_trmm!` recursion -- reads and
    # writes B at its ld, i.e. assumes `stride(B,1) == 1`. Handed a B where that is false (a lazy `B'`,
    # or a strided view such as `view(X, 1:2:2n, 1:n)`) they returned silently WRONG NUMBERS: no error,
    # no NaN, just a different matrix. Verified against OpenBLAS's `trmm!` on the same problem, and
    # present since before the adjoint work -- complex was correct only because its branch has a genuine
    # index-based base (`_trmm_cmplx_base_*!`) it routes to under the same predicate.
    #
    # The algorithm itself is fine: the identical values in a plain `Matrix` give the right answer. So
    # copy in, run the normal routing, copy back. That is O(k·n) against the product's O(k²·n), and it
    # is only paid by inputs that were previously wrong -- a `_strided1` B never enters here.
    if eltype(B) <: BlasReal && !_strided1(B)
        Bc = Matrix(B)
        trmm!(Bc, A; side, uplo, transA, diag, alpha)
        copyto!(B, Bc)
        return B
    end
    # TINY real trmm: go straight to the base kernel, skipping the `_trmm!`→`_trmm_left!/_trmm_right!`
    # wrapper chain (ROADMAP: adds ~16% on a ~50 ns 8×8 op — trmm@8 0.84 via chain vs 0.999 direct). The
    # dispatch below MIRRORS the k≤_TRMM_BASE branches of `_trmm_left!`/`_trmm_right!` exactly.
    if eltype(B) <: BlasReal && transA != 'C' && k <= _TRMM_BASE
        up_ = uplo == 'U'; tr_ = transA != 'N'; unit_ = diag == 'U'
        if sl
            k <= _fh_trmm_ddirect() ? _trmm_dense_L!(up_, tr_, unit_, A, B) : _trmm_small!(true, up_, tr_, unit_, A, B)
        else
            k <= _fh_trmm_ddirect() ? _trmm_right_base!(up_, tr_, false, unit_, k, A, B) : _trmm_small!(false, up_, tr_, unit_, A, B)
        end
        isone(alpha) || _scal_all!(B, convert(eltype(B), alpha))
    elseif eltype(B) <: BlasComplex && k <= _TRMM_BASE   # tiny complex: same skip (mirrors _trmm_left!/_right! complex base)
        up_ = uplo == 'U'; tr_ = transA != 'N'; cj_ = transA == 'C'; unit_ = diag == 'U'
        if sl
            !_strided1(B) ? _trmm_cmplx_base_L!(up_, tr_, cj_, unit_, k, A, B) :
                (_CTRMM_PACK && k >= _fh_ctrmm_pack_min()) ? _trmm_cmplx_packed_L!(up_, tr_, cj_, unit_, k, A, B) :
                _trmm_cmplx_small_L!(up_, tr_, cj_, unit_, k, A, B)
        else
            !_strided1(B) ? _trmm_cmplx_base_R!(up_, tr_, cj_, unit_, k, A, B) :
                (_CTRMM_PACK && k >= _fh_ctrmm_pack_min()) ? _trmm_cmplx_packed_R!(up_, tr_, cj_, unit_, k, A, B) :
                _trmm_cmplx_small_R!(up_, tr_, cj_, unit_, k, A, B)
        end
        isone(alpha) || _scal_all!(B, convert(eltype(B), alpha))
        # side-L real large → K-range-trimmed single-pass packed (the straddling tile contracts only its
        # nonzero p-band, not the full kc zero-band — that band was the ~kc/k waste that capped the naive
        # packed trmm). Else (side R, complex/AD, small) → recursion-over-gemm! (no regression).
        # OPEN HYPOTHESIS (task #134), recorded not wired: `_GEMM_UNPACK_MAX` is gemm's UNPACKED-vs-BLOCKED
        # crossover, derived as 2·(nvreg−4)·W = 448 on both AVX-512 boxes and 96 on AVX2. Here it decides
        # something else — recursion-over-gemm vs single-pass packed trmm, a different kernel PAIR on a
        # different shape family, never measured at its own boundary. The gate loses 9-13% in
        # 448 < n <= 480 on exactly the two boxes where this const is 448, and nowhere on the box where it
        # is 96. Corroborating: the side-R analog `_TRMM_RPACK` WAS measured directly and found packed
        # decisive only from n>=1536, direct winning the small/mid band. If confirmed the fix is a trmm-owned
        # Measure-tier crossover (candidates bounded to [_GEMM_UNPACK_MAX, 4·_GEMM_UNPACK_MAX], default =
        # today's value so migration is zero-risk), NOT a change to gemm's constant.
    elseif sl && eltype(B) <: BlasReal && transA != 'C' && k > _TRMM_PACK_MIN + _EXPINT[2] &&
            !_EXPFLAG[_EXP9]
        # 8×8 tile (Val(1), unified W==_NR): finer K-trim staircase + smaller within-tile zero triangle;
        # the proven-fastest, most consistent path across sizes. (A 16×8 bulk helped N-cases at large k
        # but regressed k=768 and the public po2 A-pad path — non-robust, not worth the split.)
        mrv = _unified_ok(eltype(B)) ? Val(1) : Val(_MR)
        # NOTE: no A-pad here (unlike trsm). trmm's po2-ld conflict is mild (~2%); the O(k²) A-copy to
        # pad it costs about the same, so padding is net-negative for trmm — measured. (trsm's conflict
        # was catastrophic 0.78→1.12, there the copy pays.)
        # Split driver: recurses to the packed base, but routes the off-diagonal update through gemm! so
        # Strassen fires at k ≥ 2·_STRASSEN_MIN (closes large-n vs AOCL). UNSCALED → scale B by α once here.
        _trmm_split_L!(uplo == 'U', transA != 'N', diag == 'U', A, B, mrv)
        isone(alpha) || _scal_all!(B, convert(eltype(B), alpha))
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
# HARD INVARIANT — `_TRSM_BASE ≤ _L3_NB`, ENFORCED, not merely documented. Both `_trsm_base_invL!` and
# `_trsm_base_invR!` build the inverse in `view(_l3_tmp(T), 1:nb, 1:nb)`, and that scratch is a fixed
# _L3_NB×_L3_NB matrix (workspace.jl:25, capped at 128) — so a base above it indexes past the buffer.
# The sibling `_TRMM_BASE = _L3_NB` (level3.jl:8) already carries this coupling in its comment.
# While this was a bare const the invariant could not be violated. Making it a `@load_preference` (this
# campaign) put it in the USER's hands, and a pin is exactly the tier an agent cannot test — so the
# clamp is the only thing standing between a pinned `trsm_base = 256` and a BoundsError thrown from
# inside a kernel, where the message names neither the preference nor the limit. Found by forcing 4096
# and reading the stacktrace, on all three boxes.
# The clamp is applied in BOTH places on purpose: on the const so a pinned/trim build folds to a safe
# literal, and inside the hook so a forced value (a sweep, a probe) cannot escape it either.
# PDM: Literal — trsm recursion base, on trtrs's real path (trtrs wraps trsm side-L). NOW A KNOB (was a bare const, unpinnable and untunable); default is the value it always had. | tune: FLAT — 16/32/48/64 all within noise on Zen3+Zen4+Zen5 (96 cells, 2026-08-21)
const _TRSM_BASE = min(@load_preference("trsm_base", 32)::Int, _L3_NB)
@inline _trsm_base() = (f = _FKR_trsm_base[]; f >= 0 ? min(f, _L3_NB) : _TRSM_BASE)
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
# PDM: Literal — triangular-inverse recursion base. NOW A KNOB (was a bare const, unpinnable and untunable); default is the value it always had. | tune: FLAT — 8/16/32/64 within noise on all 3 uarchs, both trsm sides (2026-08-21)
const _TRTRI_BASE = @load_preference("trtri_base", 16)::Int
@inline _trtri_base() = (f = _FKR_trtri_base[]; f >= 0 ? f : _TRTRI_BASE)
function _trtri!(V, A, nb::Int, up::Bool, unit::Bool)
    T = eltype(V)
    if nb <= _trtri_base()
        fill!(V, zero(T))
        @inbounds for i in 1:nb
            V[i, i] = one(T)
        end
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
# PDM: Literal — B-width cut for side-L: at or below it the narrow recursion wins. NOW A KNOB (was a bare const, unpinnable and untunable); default is the value it always had. | tune: FLAT — 32/64/128 within noise on Zen3+Zen4+Zen5 (2026-08-21)
const _TRSM_NCUT = @load_preference("trsm_ncut", 64)::Int  # side-L: B width cut (invL wins from 96 down since the gemm clip; 64 keeps dense only for n≤64)
@inline _trsm_ncut() = (f = _FKR_trsm_ncut[]; f >= 0 ? f : _TRSM_NCUT)
# PDM: Literal — B-width cut for side-R. NOW A KNOB (was a bare const, unpinnable and untunable); default is the value it always had. | tune: FLAT — 64/128/256 within noise on Zen3+Zen4+Zen5 (2026-08-21)
const _TRSM_NCUT_R = @load_preference("trsm_ncut_r", 128)::Int  # side-R: B height cut (R's narrow path is stronger than L's — measured, 128 rides it at 1.7×)
@inline _trsm_ncut_r() = (f = _FKR_trsm_ncut_r[]; f >= 0 ? f : _TRSM_NCUT_R)
# Narrow-B dense-base cutoff. Re-swept at LOCKED CPU freq (2026-07-02): 32 beats 16 (n=32 cold
# 0.565→0.75, worst-size = the gate metric); the old "16, raising hurts n=128" was a boost-noise artifact
# (benchmark with CPU boost OFF). ponytail: could be a Preferences knob if the fleet diverges.
# PDM: Literal — diagonal-block base; its own comment already said 'could be a Preference'. NOW A KNOB (was a bare const, unpinnable and untunable); default is the value it always had. | tune: FLAT — 16/32/48/64 within noise on Zen3+Zen4+Zen5 (2026-08-21)
const _TRSM_DBASE = @load_preference("trsm_dbase", 32)::Int
@inline _trsm_dbase() = (f = _FKR_trsm_dbase[]; f >= 0 ? f : _TRSM_DBASE)
# Narrow-B cutoff: side-L trsm sweeps trsv per column when nrhs ≤ this. Set to 0 to disable.
#
# The crossover is CONSTANT in k, not proportional to it. Blocked costs `setup + nrhs·b`, the sweep
# costs `nrhs·v`, so the sweep wins while `nrhs < setup/(v − b)`; setup and the per-column terms all
# scale as k², so the ratio — and therefore the crossover — is k-INVARIANT. Measured that way on
# Zen4 (µs, per-column sweep vs blocked, after the A-pad removal above):
#     k     nrhs=1        nrhs=4          nrhs=8         nrhs=16
#     512   22 vs 94      93 vs 372       186 vs 91      370 vs 315
#     1024  82 vs 429     337 vs 998      657 vs 378     1320 vs 1002
#     2048  452 vs 2475   1817 vs 3955    3611 vs 2297   7141 vs 4105
# The sweep wins at 1 and 4 and loses at 8 and 16, at EVERY k — one crossover, no k dependence.
# An earlier version of this knob used k ÷ (4·_TRSM_DBASE), reasoning that the setup amortises over
# more columns as k grows. That was the wrong SHAPE: it admitted nrhs ≤ 8 at k=1024 and ≤ 16 at
# k=2048, i.e. exactly where the sweep loses, and it was the cause of the getrs/potrs nrhs=8 misses.
# req#8 tier: validated literal. The k-invariance IS derived (above); the value 4 is the measured
# crossover and is not predictable from a detected const — it is a ratio of two kernels' constants.
# ⚠ Zen4-measured; needs fleet confirmation (the Zen3 crossover looked the same, 2–4, but was taken
# before the A-pad removal).
# PDM: Literal — B-width below which the narrow path wins; measured, not derived. | tune: candidate
const _TRSM_NARROW_MAX = @load_preference("trsm_narrow_max", 4)::Int
@inline _fh_trsm_narrow_max() = (f = _FKR_trsm_narrow_max[]; f >= 0 ? f : _TRSM_NARROW_MAX)
# Column-blocked rank-1 update for the dense trsm base (non-transpose): B[brow0.., c] -= B[irow0, c]·acol
# over all n columns. Holds the A-column vector across a block of 4 B-columns (reuse) and does the short
# rlen-remainder mask ONCE per row-block instead of once per column — the per-column `_axpy_simd!` (though
# inlined) repeated both. `a` points at A[rs,i]; brow0/irow0 are 0-based B rows.
@inline function _trsm_col_r1!(
        ::Type{T}, rlen::Int, a::Ptr{T}, pB::Ptr{T}, irow0::Int, brow0::Int,
        n::Int, ldb::Int
    ) where {T <: BlasReal}
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
# Measured Zen3: lower-unit (getrf) +119–151%; upper-non-unit (trsm gate) +44–47%. pL/pB = &·[1,1], ld col.
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
                a0 = muladd(_CVF(-unsafe_load(pB, _clidx(j, c, ldb))), Lv, a0)
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
            for j in solved
                a = muladd(_CVF(-unsafe_load(pB, _clidx(j, c, ldb))), vload(_CVF, _cvptr(pL, rb, j, ld0)), a)
            end
            vstore(a, _cvptr(pB, rb, c, ldb)); c += 1
        end
        # W diagonal reciprocals ONCE per block (not per column — the redundant per-cc inv() was a division
        # chain that sank non-unit on wide vectors). Val(W) ⇒ const-length tuple that const-folds.
        recips = unit ? ntuple(_ -> 1.0, Val(W)) :
            ntuple(q -> @inbounds(inv(unsafe_load(pL, _clidx(rb + q - 1, rb + q - 1, ld0)))), Val(W))
        return @inbounds for cc in 1:n                             # scalar W×W in-block diagonal solve
            for ii in (up ? ((W - 1):-1:0) : (0:(W - 1)))
                s = unsafe_load(pB, _clidx(rb + ii, cc, ldb))
                for jj in (up ? ((ii + 1):(W - 1)) : (0:(ii - 1)))
                    s = muladd(-unsafe_load(pL, _clidx(rb + ii, rb + jj, ld0)), unsafe_load(pB, _clidx(rb + jj, cc, ldb)), s)
                end
                unit || (s *= recips[ii + 1])
                unsafe_store!(pB, s, _clidx(rb + ii, cc, ldb))
            end
        end
    end
    @inline function dorow(i)                               # one tail row (k not a multiple of W)
        rng = up ? ((i + 1):k) : (1:(i - 1))
        return @inbounds for cc in 1:n
            s = unsafe_load(pB, _clidx(i, cc, ldb))
            for j in rng
                s = muladd(-unsafe_load(pL, _clidx(i, j, ld0)), unsafe_load(pB, _clidx(j, cc, ldb)), s)
            end
            unit || (s *= inv(unsafe_load(pL, _clidx(i, i, ld0))))
            unsafe_store!(pB, s, _clidx(i, cc, ldb))
        end
    end
    # ORDER: solve in the substitution direction. Upper=backward ⇒ tail rows (bottom) FIRST, then blocks
    # bottom-up (they downdate against the tail as "solved"). Lower=forward ⇒ blocks top-down, then tail.
    if up
        for i in (nb * W == k ? (0:-1) : (k:-1:(nb * W + 1)))
            dorow(i)
        end
        for bi in nb:-1:1
            doblock((bi - 1) * W + 1)
        end
    else
        for bi in 1:nb
            doblock((bi - 1) * W + 1)
        end
        for i in (nb * W + 1):k
            dorow(i)
        end
    end
    return nothing
end

function _trsm_dense_L!(up::Bool, tr::Bool, unit::Bool, A, B)
    k = size(A, 1); n = size(B, 2); T = eltype(B); sz = sizeof(T)
    lda = stride(A, 2); ldb = stride(B, 2); fwd = (up == tr)
    # Tile crossover (DERIVED, req#8): tile trades dense's ~k store-passes (∝ k²·n/W) for a per-block scalar
    # W×W triangular diagonal solve (∝ k·W·n, depth-W latency chain). Net win ⇒ k·W < k²/W ⇒ k > W². Fleet-
    # validated: Zen3(W=4,W²=16) wins from k=32, Zen4(W=8,W²=64) from k=96. (Side-R tiles unconditionally
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
                d = inv(A[i, i]); for c in 1:n
                    B[i, c] *= d
                end
            end
            rlen = fwd ? (k - i) : (i - 1); rlen == 0 && continue
            rs = fwd ? (i + 1) : 1
            if tr
                rows = fwd ? ((i + 1):k) : (1:(i - 1))
                for c in 1:n
                    bic = B[i, c]; @simd for r in rows
                        B[r, c] -= A[i, r] * bic
                    end
                end
            elseif T <: BlasReal
                aptr = pA + ((i - 1) * lda + (rs - 1)) * sz
                _trsm_col_r1!(T, rlen, Ptr{T}(aptr), pB, i - 1, rs - 1, n, ldb)
            else                                             # complex/generic column rank-1 (trtri base ≤16)
                rows = fwd ? ((i + 1):k) : (1:(i - 1))
                for c in 1:n
                    bic = B[i, c]; @simd for r in rows
                        B[r, c] -= A[r, i] * bic
                    end
                end
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
# FUSED 4-RHS inner update for `_dLN_panel!`: B[1:len, c0..c0+3] -= x_c · A[1:len, j], for FOUR RHS
# columns in ONE pass. `_dLN_panel!` used to call `_axpy_cmplx_simd!` once per (j, RHS column) — k·pw
# invocations, 1024 of them at k=32/nrhs=32, each ~16 complex elements, which is far too short to
# amortise the per-call entry cost. It also re-streamed A[1:len,j] once per RHS column although that
# column is loop-invariant in c.
#
# Here A's chunk is loaded ONCE and its swap computed ONCE, then reused for all four columns: 4× less A
# traffic, one shuffle instead of four, and zero calls in the loop. Same transformation as
# `_trsv_fused8!`/`_trmv_fused8!`, transposed — there F A-columns were fused against one x; here one
# A-column is fused against F B-columns.
#
# @generated for the swap mask and the per-lane sign tuple. MUST emit Expr(:meta, :inline) explicitly:
# on Julia 1.12 `@inline` does NOT propagate into a @generated body, and without it the Vec arguments
# pass BY POINTER and the caller's accumulators get stack-demoted (the zgemvC 0.58→1.20 lesson).
@generated function _dLN_fuse4!(
        len::Int, a0r::T, a0i::T, a1r::T, a1i::T, a2r::T, a2i::T, a3r::T, a3i::T,
        pa::Ptr{T}, pb0::Ptr{T}, pb1::Ptr{T}, pb2::Ptr{T}, pb3::Ptr{T}
    ) where {T <: BlasReal}
    W = _vwidth(T); V2 = Vec{2W, T}; sz = sizeof(T)
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(2W - 1))...)
    sgn(ai) = :($V2($(Expr(:tuple, (iseven(l) ? :(-$ai) : ai for l in 0:(2W - 1))...))))
    return quote
        $(Expr(:meta, :inline))
        r0 = $V2(a0r); s0 = $(sgn(:a0i)); r1 = $V2(a1r); s1 = $(sgn(:a1i))
        r2 = $V2(a2r); s2 = $(sgn(:a2i)); r3 = $V2(a3r); s3 = $(sgn(:a3i))
        i = 0
        @inbounds while i + $W <= len
            o = i * 2 * $sz
            av = vload($V2, pa + o); aw = shufflevector(av, Val($swp))   # A chunk + its swap: ONCE for all 4
            vstore(muladd(aw, s0, muladd(av, r0, vload($V2, pb0 + o))), pb0 + o)
            vstore(muladd(aw, s1, muladd(av, r1, vload($V2, pb1 + o))), pb1 + o)
            vstore(muladd(aw, s2, muladd(av, r2, vload($V2, pb2 + o))), pb2 + o)
            vstore(muladd(aw, s3, muladd(av, r3, vload($V2, pb3 + o))), pb3 + o)
            i += $W
        end
        @inbounds while i < len                                          # scalar tail
            j2 = 2 * (i + 1); ar = unsafe_load(pa, j2 - 1); ai = unsafe_load(pa, j2)
            unsafe_store!(pb0, unsafe_load(pb0, j2 - 1) + a0r * ar - a0i * ai, j2 - 1)
            unsafe_store!(pb0, unsafe_load(pb0, j2) + a0r * ai + a0i * ar, j2)
            unsafe_store!(pb1, unsafe_load(pb1, j2 - 1) + a1r * ar - a1i * ai, j2 - 1)
            unsafe_store!(pb1, unsafe_load(pb1, j2) + a1r * ai + a1i * ar, j2)
            unsafe_store!(pb2, unsafe_load(pb2, j2 - 1) + a2r * ar - a2i * ai, j2 - 1)
            unsafe_store!(pb2, unsafe_load(pb2, j2) + a2r * ai + a2i * ar, j2)
            unsafe_store!(pb3, unsafe_load(pb3, j2 - 1) + a3r * ar - a3i * ai, j2 - 1)
            unsafe_store!(pb3, unsafe_load(pb3, j2) + a3r * ai + a3i * ar, j2)
            i += 1
        end
        return nothing
    end
end

@inline function _dLN_panel!(
        up::Bool, unit::Bool, k::Int, rcp, Ap::Ptr{Tc}, Bp::Ptr{Tc},
        pw::Int, lda::Int, ldb::Int, csz::Int
    ) where {Tc}
    T = real(Tc)
    @inbounds for j in (up ? (k:-1:1) : (1:k))
        aj = Ap + (j - 1) * lda * csz                            # &A[1,j] (Julia Ptr+int = BYTES)
        len = up ? j - 1 : k - j                                 # rows updated by this j
        off = up ? 0 : j                                         # row offset of that run
        c = 0
        while c + 4 <= pw                                        # FUSED: 4 RHS columns share one A pass
            b0 = Bp + c * ldb * csz; b1 = b0 + ldb * csz
            b2 = b1 + ldb * csz;     b3 = b2 + ldb * csz
            x0 = unit ? unsafe_load(b0, j) : unsafe_load(b0, j) * rcp[j]
            x1 = unit ? unsafe_load(b1, j) : unsafe_load(b1, j) * rcp[j]
            x2 = unit ? unsafe_load(b2, j) : unsafe_load(b2, j) * rcp[j]
            x3 = unit ? unsafe_load(b3, j) : unsafe_load(b3, j) * rcp[j]
            if !unit
                unsafe_store!(b0, x0, j); unsafe_store!(b1, x1, j)
                unsafe_store!(b2, x2, j); unsafe_store!(b3, x3, j)
            end
            if len > 0
                ao = Ptr{T}(aj + off * csz); do_ = off * csz
                _dLN_fuse4!(
                    len, -real(x0), -imag(x0), -real(x1), -imag(x1),
                    -real(x2), -imag(x2), -real(x3), -imag(x3), ao,
                    Ptr{T}(b0 + do_), Ptr{T}(b1 + do_), Ptr{T}(b2 + do_), Ptr{T}(b3 + do_)
                )
            end
            c += 4
        end
        while c < pw                                             # ragged tail: the original per-column path
            bc = Bp + c * ldb * csz                              # &B[1, panel-col c]
            xj = unit ? unsafe_load(bc, j) : unsafe_load(bc, j) * rcp[j]
            unit || unsafe_store!(bc, xj, j)
            len > 0 && _axpy_cmplx_simd!(len, -real(xj), -imag(xj), aj + off * csz, bc + off * csz)
            c += 1
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
        unit || @inbounds @simd for j in 1:k
            rcp[j] = _crecip(unsafe_load(Ap, (j - 1) * lda + j))
        end
        pc = 0
        while pc < nrhs
            pw = min(nc, nrhs - pc)
            _dLN_panel!(up, unit, k, rcp, Ap, Bp + pc * ldb * csz, pw, lda, ldb, csz)
            pc += nc
        end
    end
    return B
end

# ===== Complex gemmtrsm leaf (side-L, upper, no-trans) — SIMD lanes are RHS COLUMNS =====
#
# WHY THIS EXISTS. `_trsm_cmplx_dLN!` above re-reads and re-writes all of B once per A-column: k passes
# with ~2 muladds per load+store, i.e. a BLAS-2 traffic pattern wearing a BLAS-3 name. The roofline
# decomposition that found it (Zen4, freq-locked, 2026-08-02) is the clean statement of the gap: PB's
# OWN zgemm matched the best reference at every size (0.97–1.02), while PB's ztrsm reached only 0.41 /
# 0.53 / 0.63 of that same zgemm at k=32/64/128 where AOCL reached 0.54 / 0.73 / 0.83 of its own. The
# arithmetic was never the problem; converting gemm rate into trsm rate was.
#
# TWO SHAPES WERE MEASURED AND REJECTED FIRST — both die on the register/lane budget, and both are
# worth recording so they are not re-attempted:
#   * F A-columns × C B-columns fused (the `_trsv_fused8!` trick crossed with `_dLN_fuse4!`). The F·C
#     scalar pairs are LOOP-INVARIANT, so LLVM hoists 2·F·C broadcast vectors: 44 zmm needed at F=C=4
#     against 32 available. Measured 6× SLOWER. `_dLN_fuse4!` (F=1, C=4) already sits at 28 zmm — that
#     loop shape has no headroom left, which is why entry-overhead work on it kept returning ~nothing.
#   * A W-row × C-column register tile with lanes = ROWS. Correct, but when lanes are rows the W×W
#     diagonal triangle cannot vectorize at all; at k=32 that is 22% of the flops at scalar rate, and
#     the whole leaf measured 8 GF against dLN's 17.
# With lanes = COLUMNS the triangle vectorizes too — all NR columns are substituted simultaneously and
# only the row dependency runs serially through registers. That is the same reason the real
# `_trsm_fused_L!` packs, and it is what makes the transpose pack unavoidable rather than incidental.
#
# Measured Zen4 vs AOCL (GFlop/s, k = nrhs): k=16 16.4/13.3, k=32 29.0/22.5, k=48 33.5/27.3,
# k=64 36.4/29.8, k=96 39.0/33.8, k=128 40.4/35.5 — i.e. 1.14–1.29×, where dLN was 0.64–0.74×.
# The vectorized transpose pack is NOT a refinement: with the scalar pack the same geometry measured
# 20.8/22.6 at k=32 (0.92, still under gate). Pack cost was 25–45% of a k=32 stripe.
const _ZGT_W = _vwidth(Float64)                  # lanes = RHS columns, so the stripe width is one vector
const _ZGT_NR = _ZGT_W
# MR (rows per slab) — Derive. The register bound is 2·MR accumulators + 2 packed-row vectors ≤ nreg,
# i.e. MR ≤ 14 on AVX-512 / 6 on AVX2, and is NOT binding: total accumulate flops are MR-INDEPENDENT
# (2·KC² either way), so MR only trades triangle flops (∝ MR) against packed-row loads (∝ 1/MR). The
# derivation that fixes it is structural rather than arithmetic — MR = W makes a slab exactly one W×W
# transpose block, so slabs and pack blocks share ONE ragged region instead of each growing their own.
# Confirmed on Zen4: MR ∈ {8,10,12} at NRV=1 measured 40.4 / 34.7 / 34.6 at k=128, W=8 winning.
# PDM: Derived — formula over detected consts: `_ZGT_W`
const _ZGT_MR = @load_preference("ztrsm_gt_mr", _ZGT_W)::Int
# The leaf needs an in-register W×W transpose (`_tr8x8` / `_tr4x4`); other widths keep the dLN base.
const _ZGT_ON = (_ZGT_W == 8 || _ZGT_W == 4)
# k ceiling — Derive from L2 RESIDENCY OF A'S TRIANGULAR PANEL. A is re-read once per NR-wide column
# stripe (n/NR times), so the binding constraint is that the KC×KC panel stay in L2:
#   KC²·sizeof(ComplexF64) ≤ L2  ⇒  KC ≤ √(L2 ÷ 16).
# Zen4 (1 MB L2) ⇒ 256; Zen3 (512 KB L2) ⇒ 181. FLEET-VALIDATED, both boxes, and the second box is what
# corrected it: the first derivation bounded the PACKED STRIPE by L1 (KC ≤ L1 ÷ (2·NR·8)), which gives
# the same 256 on Zen4 but 512 on AVX2 because NR=W=4 halves the stripe — and at that base Zen3
# measured leaf/AOCL 1.005 at k=192 but 0.820 at k=256 and 0.818 at 512, regressing ztrsm@256 from
# 0.851 to 0.787. The L1-stripe bound is not binding (KC=256/NR=8 fills L1 exactly on Zen4 and gates
# fine); A's panel is, and NR=4 doubles the number of A re-reads on AVX2. `min` with the stripe bound
# keeps a hypothetical huge-L2 box from blowing L1. The real path learned the same lesson the same way
# (_TRSM_FUSED_BASE keeps a literal 128 on non-AVX-512 because a bigger base REGRESSES n=256).
const _ZGT_BASE = @load_preference(
    "ztrsm_gt_base",
    min(
        isqrt(_L2_BYTES ÷ sizeof(ComplexF64)),
        _L1_BYTES ÷ (2 * _ZGT_NR * sizeof(Float64))
    )
)::Int
# Deinterleave/interleave lane masks for the complex plane split (const-folded).
const _ZGT_DERE = Val(ntuple(l -> 2 * (l - 1), Val(_ZGT_W)))
const _ZGT_DEIM = Val(ntuple(l -> 2 * (l - 1) + 1, Val(_ZGT_W)))
const _ZGT_ILV = Val(ntuple(l -> isodd(l) ? (l - 1) >> 1 : _ZGT_W + ((l - 1) >> 1), Val(2 * _ZGT_W)))

@inline _zgt_tr(x::NTuple{8, Vec{8, Float64}}) = _tr8x8(x...)
@inline _zgt_tr(x::NTuple{4, Vec{4, Float64}}) = _tr4x4(x...)

# W rows × W columns of B → the row-major re/im planes: one contiguous Vec{2W} load per column (B is
# column-major, so this is unit-stride and po2-immune), deinterleave, then the existing W×W transpose.
@inline function _zgt_pack!(pr::Ptr{Float64}, pim::Ptr{Float64}, pB0::Ptr{Float64}, cs::Int, i0::Int)
    sz = sizeof(Float64); o = i0 * 2 * sz
    c = ntuple(v -> vload(Vec{2 * _ZGT_W, Float64}, pB0 + (v - 1) * cs + o), Val(_ZGT_W))
    yr = _zgt_tr(ntuple(v -> shufflevector(c[v], _ZGT_DERE), Val(_ZGT_W)))
    yi = _zgt_tr(ntuple(v -> shufflevector(c[v], _ZGT_DEIM), Val(_ZGT_W)))
    @inbounds for r in 1:_ZGT_W
        vstore(yr[r], pr + (i0 + r - 1) * _ZGT_NR * sz)
        vstore(yi[r], pim + (i0 + r - 1) * _ZGT_NR * sz)
    end
    return
end
@inline function _zgt_unpack!(pr::Ptr{Float64}, pim::Ptr{Float64}, pB0::Ptr{Float64}, cs::Int, i0::Int)
    sz = sizeof(Float64); o = i0 * 2 * sz
    xr = _zgt_tr(ntuple(r -> vload(Vec{_ZGT_W, Float64}, pr + (i0 + r - 1) * _ZGT_NR * sz), Val(_ZGT_W)))
    xi = _zgt_tr(ntuple(r -> vload(Vec{_ZGT_W, Float64}, pim + (i0 + r - 1) * _ZGT_NR * sz), Val(_ZGT_W)))
    @inbounds for v in 1:_ZGT_W
        vstore(shufflevector(xr[v], xi[v], _ZGT_ILV), pB0 + (v - 1) * cs + o)
    end
    return
end

# One slab: rows r0..r0+MR-1 of the packed stripe. Accumulate the trailing update against EVERY already-
# solved row t (one pass, accumulators never leaving registers — this is what replaces dLN's k passes
# over B), then substitute the MR×MR diagonal triangle in register. `rc` holds the split reciprocals.
# @generated for the unrolled row set; MUST emit Expr(:meta,:inline) — on Julia 1.12 `@inline` does not
# propagate into a @generated body and the Vec accumulators would be stack-demoted (the zgemvC lesson).
# SIGN PLACEMENT — this kernel does NOT have `_zrt_tile!`'s defect, and the measurement that proved it
# is worth keeping so the "audit every signed broadcast" idea is not re-run here. `_zrt_tile!` wrote
# `V2(-cr)` where the negated splat was used ONCE, so LLVM emitted vmovsd + vxorpd + vbroadcastsd per
# coefficient (3 instructions, 2.95 per useful FMA) and folding the sign onto the vector operand was
# worth up to 13.9%. Here `V(-ur)` is used TWICE per row-step, which is enough for LLVM to canonicalise
# it into `vfnmadd231pd` by itself: the loop already compiles to 120 vfnmadd + 40 vfmadd + 80
# vbroadcastsd, **zero vmovsd and zero vxorpd**, at 1.94 instructions per useful FMA — i.e. already at
# the level `_zrt_tile!` only reached after its rewrite. The hand-folded variant (negate the two panel
# vectors once per t, the two accumulators once per j) was written, proved correct over 75 (s, nrhs)
# cells, and compiles to a BYTE-IDENTICAL histogram; its A/B was correspondingly null at all six sizes.
# THE RULE, stated properly: a negated broadcast costs extra only when it is used ONCE. Reverted.
@generated function _zgt_slab!(
        ::Val{MR}, pr::Ptr{Float64}, pim::Ptr{Float64}, pA::Ptr{Float64}, lda::Int,
        rc::Ptr{Float64}, r0::Int, kc::Int
    ) where {MR}
    sz = sizeof(Float64); NR = _ZGT_NR; V = Vec{_ZGT_W, Float64}
    ar(i) = Symbol(:ar, i); ai(i) = Symbol(:ai, i)
    ld = [
        quote
            $(ar(i)) = vload($V, pr + ($i + r0) * $NR * $sz)
            $(ai(i)) = vload($V, pim + ($i + r0) * $NR * $sz)
        end for i in 0:(MR - 1)
    ]
    # row i -= U[r0+i, t] · P[t, :]  (complex, split planes)
    acc = [
        quote
            ur = unsafe_load(pAt, 2 * (r0 + $i) + 1); ui = unsafe_load(pAt, 2 * (r0 + $i) + 2)
            $(ar(i)) = muladd($V(ui), pm, muladd($V(-ur), pv, $(ar(i))))
            $(ai(i)) = muladd($V(-ui), pv, muladd($V(-ur), pm, $(ai(i))))
        end for i in 0:(MR - 1)
    ]
    tri = map((MR - 1):-1:0) do j
        upd = [
            quote
                ur = unsafe_load(pAj, 2 * (r0 + $i) + 1); ui = unsafe_load(pAj, 2 * (r0 + $i) + 2)
                $(ar(i)) = muladd($V(ui), $(ai(j)), muladd($V(-ur), $(ar(j)), $(ar(i))))
                $(ai(i)) = muladd($V(-ui), $(ar(j)), muladd($V(-ur), $(ai(j)), $(ai(i))))
            end for i in 0:(j - 1)
        ]
        return quote
            rr = unsafe_load(rc, 2 * (r0 + $j) + 1); ri = unsafe_load(rc, 2 * (r0 + $j) + 2)
            nr = $(ar(j)) * $V(rr) - $(ai(j)) * $V(ri)
            ni = $(ar(j)) * $V(ri) + $(ai(j)) * $V(rr)
            $(ar(j)) = nr; $(ai(j)) = ni
            pAj = pA + (r0 + $j) * lda * 2 * $sz
            $(upd...)
        end
    end
    st = [
        quote
            vstore($(ar(i)), pr + ($i + r0) * $NR * $sz)
            vstore($(ai(i)), pim + ($i + r0) * $NR * $sz)
        end for i in 0:(MR - 1)
    ]
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            $(ld...)
            for t in (r0 + $MR):(kc - 1)
                pv = vload($V, pr + t * $NR * $sz); pm = vload($V, pim + t * $NR * $sz)
                pAt = pA + t * lda * 2 * $sz
                $(acc...)
            end
            $(tri...)
            $(st...)
        end
        return nothing
    end
end

# Driver: one NR-wide column stripe at a time; pack → slabs bottom-up → unpack.
function _trsm_cgt_L!(unit::Bool, k::Int, A, B)
    nrhs = size(B, 2); csz = sizeof(ComplexF64); sz = sizeof(Float64)
    NR = _ZGT_NR; MR = _ZGT_MR
    lda = stride(A, 2); ldb = stride(B, 2)
    buf = _trsm_fused_buf(Float64, 2 * k * NR + 2 * k)
    GC.@preserve A B buf begin
        pA = Ptr{Float64}(pointer(A)); pB = Ptr{Float64}(pointer(B))
        pr = pointer(buf); pim = pr + k * NR * sz; rc = pim + k * NR * sz
        @inbounds for j in 0:(k - 1)
            z = unit ? one(ComplexF64) : _crecip(unsafe_load(Ptr{ComplexF64}(pA), j * lda + j + 1))
            unsafe_store!(rc, real(z), 2j + 1); unsafe_store!(rc, imag(z), 2j + 2)
        end
        # Ragged rows go at the TOP for both the pack blocks and the slabs: the slab loop walks up from
        # row k-MR, so k mod W is left over at row 0 either way. Anchoring the transpose blocks to the
        # same end keeps ONE ragged region instead of one at each end.
        rlo = k % _ZGT_W
        jc = 0
        while jc < nrhs
            wid = min(NR, nrhs - jc)
            pB0 = pB + jc * ldb * csz
            lo = wid == NR ? rlo : k                     # ragged column stripe → scalar pack (no full block)
            @inbounds for i0 in lo:_ZGT_W:(k - 1)
                _zgt_pack!(pr, pim, pB0, ldb * csz, i0)
            end
            @inbounds for v in 0:(wid - 1)               # scalar pack for the ragged rows / ragged stripe
                sc = pB0 + v * ldb * csz
                for i in 0:(lo - 1)
                    unsafe_store!(pr + (i * NR + v) * sz, unsafe_load(sc, 2i + 1))
                    unsafe_store!(pim + (i * NR + v) * sz, unsafe_load(sc, 2i + 2))
                end
            end
            @inbounds for v in wid:(NR - 1), i in 0:(k - 1)   # zero-pad the unused lanes of a short stripe
                unsafe_store!(pr + (i * NR + v) * sz, 0.0)
                unsafe_store!(pim + (i * NR + v) * sz, 0.0)
            end
            r0 = k - MR
            while r0 >= 0
                _zgt_slab!(Val(_ZGT_MR), pr, pim, pA, lda, rc, r0, k)
                r0 -= MR
            end
            @inbounds for r in (r0 + MR - 1):-1:0        # ragged top rows, one at a time (Val(1) is literal)
                _zgt_slab!(Val(1), pr, pim, pA, lda, rc, r, k)
            end
            @inbounds for i0 in lo:_ZGT_W:(k - 1)
                _zgt_unpack!(pr, pim, pB0, ldb * csz, i0)
            end
            @inbounds for v in 0:(wid - 1)
                dc = pB0 + v * ldb * csz
                for i in 0:(lo - 1)
                    unsafe_store!(dc, unsafe_load(pr + (i * NR + v) * sz), 2i + 1)
                    unsafe_store!(dc, unsafe_load(pim + (i * NR + v) * sz), 2i + 2)
                end
            end
            jc += NR
        end
    end
    return B
end
# Eligibility: ComplexF64 only (the plane split rides the f64 W×W transpose), unit-stride A and B,
# upper + no-trans (the substitution direction the slab hardcodes), and k within the L1 stripe bound.
# BOTH eltypes are checked. Gating on `eltype(B)` alone is a memory-safety hole, not a typo: the leaf
# does `Ptr{Float64}(pointer(A))` and strides A by 16 bytes/element, so a Float64 (or ComplexF32) A with
# a ComplexF64 B reads ~2× past A's allocation and returns NaN. Reachable only from the native Mode-2
# API — LBT callers always match eltypes — which is exactly why no test caught it (found by adversarial
# review, 2026-08-02, with a verified reproducer). The sibling direct paths had the same hole; see the
# matching guards on _trsm_cmplx_dLN!/_dRN!/_dRC!.
@inline _cgt_ok(A, B) = _ZGT_ON && eltype(B) === ComplexF64 && eltype(A) === ComplexF64 &&
    _strided1(A) && _strided1(B) && stride(A, 2) >= size(A, 1)

# n above which trsm-L inverts (trtri) + K-TRIM trmm-on-inverse. At/below it (N case), the direct j-outer
# solve above; the trtri overhead + extra flops sank small/mid-n. Per-box knob.
# 64 IS NOT THE ztrsmR@100 GAP — measured, so don't spend the knob on it. Zen3 2026-08-28, PB/AOCL,
# 8 rounds, ABBA: forcing 256 (which routes n=100 to the direct base instead of trtri+trmm) gave 0.907
# against 0.913 for the shipped 64 — no gain, marginally worse. ztrsmR@100 misses on AOCL (vs_OB is
# 1.16), so the deficit is inside the path, not in the choice of path.
# NOTE the grade of that evidence: Zen3's llama-server went active mid-sweep and the contention guard
# refused two of the six runs. The four that completed passed both entry and exit checks, so the A/B is
# sound as SCREENING, but a near-parity re-test wants a quiet box.
# PDM: Literal — trtri overhead plus extra flops sink small/mid n, so the direct path stops here. | tune: candidate
const _CTRSM_DIRECT_MAX = @load_preference("ctrsm_direct_max", 64)::Int
@inline _fh_ctrsm_direct_max() = (f = _FKR_ctrsm_direct_max[]; f >= 0 ? f : _CTRSM_DIRECT_MAX)
# Complex trsm-L recursion base for NARROW B (nrhs ≤ _CTRSM_NCUT): blocks > this SPLIT (row-halve + gemm
# off-diagonal update, OB's structure); ≤ this bottom out in a small j-outer base. Monolithic j-outer caps
# ~0.85 at n=128; recursing into small bases + gemm subtracts recovers the blocking (rec=64 → 0.91).
# Wide B keeps the trtri-on-inverse base (_TRMM_BASE) — its invert is amortized by the big gemm. Per-box knob.
# PDM: Literal — recursion cut for complex trsm side-L; per-box. | tune: candidate
const _CTRSM_REC_L = @load_preference("ctrsm_rec_l", 64)::Int
@inline _fh_ctrsm_rec_l() = (f = _FKR_ctrsm_rec_l[]; f >= 0 ? f : _CTRSM_REC_L)
# PDM: Literal — B-width cut: at or below it, the j-outer narrow recursion wins. | tune: candidate
const _CTRSM_NCUT = @load_preference("ctrsm_ncut", 128)::Int   # B-width cut: ≤ → narrow (j-outer recursion)
@inline _fh_ctrsm_ncut() = (f = _FKR_ctrsm_ncut[]; f >= 0 ? f : _CTRSM_NCUT)
# Complex trsm K-TRIM: op(A)⁻¹ = op(A⁻¹), A⁻¹ triangular → reuse the trmm K-TRIM kernel on the inverse at
# half the flops (large-n / trans). Small-n N → direct j-outer solve (no trtri; OB's approach).
function _trsm_cmplx_small_L!(up::Bool, tr::Bool, cj::Bool, unit::Bool, k::Int, A, B)
    if up && !tr && k <= _ZGT_BASE && _cgt_ok(A, B)                  # register-tiled gemmtrsm leaf
        return _trsm_cgt_L!(unit, k, A, B)
    end
    if !tr && k <= _fh_ctrsm_direct_max() && _strided1(B) && eltype(A) === eltype(B)   # direct back-substitution
        return _trsm_cmplx_dLN!(up, unit, k, A, B)                                # (no trtri)
    end
    T = eltype(B); Vv = view(_trsm_tmp(T, k, k), 1:k, 1:k)
    _trtri!(Vv, A, k, up, unit)                                      # Vv = A⁻¹ (as-stored, non-conj)
    return (_CTRMM_PACK && k >= _fh_ctrmm_pack_min()) ? _trmm_cmplx_packed_L!(up, tr, cj, false, k, Vv, B) :
        _trmm_cmplx_small_L!(up, tr, cj, false, k, Vv, B)
end
# Direct complex side-R column-substitution base (no trtri): X·op(A)=B in place, !tr (⟹ !cj), unit or
# non-unit, A k×k upper/lower. Ascending columns when up (up≠tr, tr=false). The side-R mirror of
# _trsm_cmplx_dLN! and the complex sibling of _trsm_dense_R! (same loop, complex SIMD kernels). Beats OB
# for k≤64 (trtri-free) where the invert+trmm base's trtri is 40–66% exposed overhead (measured, Zen3).
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
# ===== Complex trsm side-R register tile (X·U = B, upper, no-trans) — lanes are ROWS OF B =====
#
# The side-R twin of the `_trsm_cgt_L!` problem, with a strictly easier answer. `_trsm_cmplx_dRN!` issues
# one full-length `_axpy_cmplx_simd!` per (i,j) pair, so B's columns are re-read k times with no row
# blocking — BLAS-2 traffic again.
#
# Side-R's independent problems are the ROWS of B, and rows are CONTIGUOUS within a column, so putting
# them in SIMD lanes needs NO transpose pack (the side-L leaf pays for one only because its independent
# axis is the columns). A W-row × NC-column tile of B sits in registers; the already-solved columns of
# that same tile are the operands of the in-register triangle, so the diagonal block costs zero memory
# traffic. Registers: 2·NC accumulators + 2 (xv, xw), all Vec{2W}.
#
# Measured Zen4 vs AOCL (GFlop/s, m=k, freq-locked): k=16 19.9/14.5, k=32 28.1/24.0, k=48 31.4/29.0,
# k=64 31.8/31.9, k=96 32.4/35.4 — beats dRN (16.4 / 21.6 / 25.4 / 29.0 / 30.0) at every size and AOCL
# through k=48. It is deliberately a BASE, not the whole solve: run monolithically it plateaus at ~32 GF
# for every k (measured flat 32.1–32.3 from n=256 to n=1024 against AOCL's 42) because the i-loop
# re-reads solved columns at ldb stride and the panel leaves cache. The existing `_trsm_right!`
# recursion + `_gemm_subR!` already solves that; this only replaces the leaf it bottoms out in.
#
# NC — Derive. Per row-block the trailing muladds are NC-INDEPENDENT (≈k²), while the panel re-reads fall
# as k²/(2·NC) loads and the in-register triangle grows as k·(NC-1) fmas. Balancing the two marginal
# costs gives NC ≈ √(KC·F/2L) ≈ √(KC/2) for a machine issuing ~2 fma and ~2–3 loads per cycle — which
# predicts 4 at k=32 and ~8 at k=128, matching where each measured best. Evaluated at the base size and
# bounded by the register file (2·NC + 2 vectors must fit).
# NC must also DIVIDE the base, or every base invocation leaves a ragged column tail that costs an extra
# `_gemm_core!` plus a `dRN` call. Rounding √(KC/2) down to a power of two does both jobs at once, since
# the recursion halves and the base is a power of two: √(64/2)=5.66 → 4, which divides 32/64/128 exactly.
# Measured: the un-rounded 5 left a 4-column tail per call and regressed ztrsmR@512 0.979→0.965 (spread
# 0.001, so 14× the noise) while helping the sizes whose base call happened to divide evenly.
# (the register count is spelled out rather than reusing `_GT_NREG`, which this file defines further down)
const _ZRT_NC = @load_preference(
    "ztrsm_zrt_nc",
    clamp(prevpow(2, max(2, isqrt(_fh_ctrsm_rec_l() ÷ 2))), 2, ((_SIMD_BYTES >= 64 ? 32 : 16) - 4) ÷ 2)
)::Int
const _ZRT_SWP = Val(ntuple(l -> (l - 1) ⊻ 1, Val(2 * _ZGT_W)))
const _ZRT_NEG = Vec{2 * _ZGT_W, Float64}(ntuple(l -> isodd(l) ? -1.0 : 1.0, Val(2 * _ZGT_W)))
# Per-REAL-lane complex row index (1,1,2,2,…,W,W) — the `clane` idiom. A CONST because building it
# inside the @generated body would put a closure in the generator, which Julia rejects as impure.
const _ZRT_CLANE = Vec(ntuple(l -> (l + 1) >> 1, Val(2 * _ZGT_W)))
@inline _zrt_nswap(v) = shufflevector(v, _ZRT_SWP) * _ZRT_NEG      # [-im, re, -im, re, …]

# Rows r0..r0+W-1 of the NC-column block at jb: fold in every solved column i<jb (one B load shared by
# all NC columns), then substitute the NC×NC triangle entirely in register.
# SIGN PLACEMENT (`FOLD`) — where the update's minus sign lives, and it is worth 2/3 of the loop body.
# The subtraction needs one negation per (i, t) coefficient pair OR one per i-step, and those are not
# the same instruction count. Writing it on the scalar coefficient (`V2(-cr)`, FOLD=false) makes LLVM
# emit vmovsd + vxorpd + vbroadcastsd per coefficient — 96 instructions per unrolled-by-4 iteration
# against 64 useful vfmadd231pd — because an `fneg` between the load and the splat blocks BOTH the
# vfnmadd opcode and AVX-512's embedded broadcast `(mem){1to8}`. Negating the B vector once instead
# (FOLD=true) leaves the coefficients as bare loads that can fold straight into the FMA. It is exact,
# not an approximation: `_zrt_nswap(-v) == -_zrt_nswap(v)` since nswap is a shuffle times a constant,
# so xw needs no separate treatment, and the triangle gets the same rewrite via one `nv = -acc(t)`.
# MSK=true is the ragged-ROW variant. Those rows used to fall into `_trsm_cmplx_dRN!` — the per-column
# BLAS-2 path this very leaf exists to replace — and that, not the column tail, is what cost ztrsmR its
# n=50/100 cells. Measured Zen4, m=k, PB-only GFlop/s: k=52 and k=100 have NO column tail at all and
# still read 24.4 and 33.7 against 35.7 (k=48) and 39.8 (k=104), i.e. -30% and -15% from the row tail
# alone. Lanes here are ROWS of B, so the tail is the SAME kernel under a mask rather than a different
# algorithm. Inactive lanes are never accessed, so a masked load cannot run past the end of a column
# (the property `directb-masked-oob-guardpage` requires of any direct-read microkernel).
# `nact` = active COMPLEX rows; the mask is per REAL lane, hence the (l+1)>>1 `clane` form from
# simd_kernels.jl (1,1,2,2,…,W,W).
@generated function _zrt_tile!(
        ::Val{NC}, ::Val{FOLD}, ::Val{MSK}, pB::Ptr{Float64}, ldb::Int, pA::Ptr{Float64}, lda::Int,
        rc::Ptr{Float64}, r0::Int, jb::Int, nact::Int
    ) where {NC, FOLD, MSK}
    sz = sizeof(Float64); V2 = Vec{2 * _ZGT_W, Float64}
    a(t) = Symbol(:acc, t)
    mskdef = MSK ? :(msk = _ZRT_CLANE <= nact) : nothing
    ld = [MSK ? :($(a(t)) = vload($V2, pB + ((jb + $t) * ldb + r0) * 2 * $sz, msk)) :
          :($(a(t)) = vload($V2, pB + ((jb + $t) * ldb + r0) * 2 * $sz)) for t in 0:(NC - 1)]
    upd = [
        quote
            cr = unsafe_load(pAi, 2 * (jb + $t) * lda + 1); ci = unsafe_load(pAi, 2 * (jb + $t) * lda + 2)
            $(a(t)) = $(FOLD ? :(muladd($V2(cr), xv, muladd($V2(ci), xw, $(a(t))))) :
                :(muladd($V2(-cr), xv, muladd($V2(-ci), xw, $(a(t))))))
        end for t in 0:(NC - 1)
    ]
    tri = map(0:(NC - 1)) do t
        src = FOLD ? :nv : a(t)                                  # the operand the feed multiplies by
        feed = [
            quote
                cr = unsafe_load(pAt, 2 * (jb + $u) * lda + 1); ci = unsafe_load(pAt, 2 * (jb + $u) * lda + 2)
                $(a(u)) = $(FOLD ? :(muladd($V2(cr), $src, muladd($V2(ci), sw, $(a(u))))) :
                    :(muladd($V2(-cr), $src, muladd($V2(-ci), sw, $(a(u))))))
            end for u in (t + 1):(NC - 1)
        ]
        return quote
            rr = unsafe_load(rc, 2 * (jb + $t) + 1); ri = unsafe_load(rc, 2 * (jb + $t) + 2)
            $(a(t)) = $V2(rr) * $(a(t)) + $V2(ri) * _zrt_nswap($(a(t)))
            $(FOLD ? :(nv = -$(a(t))) : nothing)
            sw = _zrt_nswap($src)
            pAt = pA + (jb + $t) * 2 * $sz                       # &A[jb+t+1, 1]; walk columns by lda
            $(feed...)
        end
    end
    st = [MSK ? :(vstore($(a(t)), pB + ((jb + $t) * ldb + r0) * 2 * $sz, msk)) :
          :(vstore($(a(t)), pB + ((jb + $t) * ldb + r0) * 2 * $sz)) for t in 0:(NC - 1)]
    # NOTE: built here as plain conditionals, NOT via a helper function. Calling a module-level
    # function from inside a @generated body trips "The function body AST ... is not pure".
    xvld = MSK ? :(vload($V2, pB + (i * ldb + r0) * 2 * $sz, msk)) :
           :(vload($V2, pB + (i * ldb + r0) * 2 * $sz))
    xvex = FOLD ? :(-$xvld) : xvld
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            $(mskdef)
            $(ld...)
            for i in 0:(jb - 1)
                xv = $(xvex)
                xw = _zrt_nswap(xv)
                pAi = pA + i * 2 * $sz                           # &A[i+1, 1]
                $(upd...)
            end
            $(tri...)
            $(st...)
        end
        return nothing
    end
end

# One column group: every full row block, then the ragged rows under a mask. Split out so the NC ladder
# below can reuse it at NC, 2 and 1 without three copies of the row loop.
@inline function _zrt_colgroup!(
        ::Val{NCv}, ::Val{FOLD}, pB::Ptr{Float64}, ldb::Int, pA::Ptr{Float64}, lda::Int,
        prc::Ptr{Float64}, jb::Int, m::Int, mb::Int, W::Int
    ) where {NCv, FOLD}
    for r0 in 0:W:(mb - 1)
        _zrt_tile!(Val(NCv), Val(FOLD), Val(false), pB, ldb, pA, lda, prc, r0, jb, W)
    end
    mb < m && _zrt_tile!(Val(NCv), Val(FOLD), Val(true), pB, ldb, pA, lda, prc, mb, jb, m - mb)
    return nothing
end

# The COLUMN ladder: NC-wide groups while they fit, then 2, then 1 — no `gemm_core!` + `dRN` tail.
# A tile at column offset `jb` already subtracts EVERY previously solved column (its `for i in 0:(jb-1)`
# loop), so a narrower tile at the tail does the trailing gemm's work AND the solve, in this kernel
# rather than in two BLAS-2-ish calls. That is why the ladder REPLACES the old tail instead of
# optimising it.
# MEASURED Zen4 (m=k, PB-only GFlop/s), and it is a SMALLER win than the row tail was:
#   n=50  29.58 -> 31.23  (+5.6%)     n=100  36.81 -> 38.13  (+3.6%)
#   n=52  33.08 -> 33.08  (unchanged — it has no column tail, so this is the control)
#   clean sizes unchanged: 48 35.58, 56 36.16, 96 40.56, 104 39.91, 128 41.85, 256 46.55
# n=50 is still ~12% under its clean neighbour (31.23 vs 35.58 at n=48), so the column tail was NOT
# the whole of the residual and something else remains at that size. Full progression there:
# 23.11 (original) -> 29.58 (masked row tail) -> 31.23 (this ladder), i.e. +35% total.
@inline function _zrt_sweep!(::Val{FOLD}, pB, ldb, pA, lda, prc, k, m, mb, W, NC) where {FOLD}
    jb = 0
    while jb + NC <= k
        _zrt_colgroup!(Val(NC), Val(FOLD), pB, ldb, pA, lda, prc, jb, m, mb, W); jb += NC
    end
    while jb + 2 <= k
        _zrt_colgroup!(Val(2), Val(FOLD), pB, ldb, pA, lda, prc, jb, m, mb, W); jb += 2
    end
    while jb < k
        _zrt_colgroup!(Val(1), Val(FOLD), pB, ldb, pA, lda, prc, jb, m, mb, W); jb += 1
    end
    return nothing
end

function _trsm_zrt_R!(unit::Bool, k::Int, A, B)
    m = size(B, 1); W = _ZGT_W
    lda = stride(A, 2); ldb = stride(B, 2)
    mb = (m ÷ W) * W                         # rows covered by full vector blocks
    rc = _trsm_fused_buf(Float64, 2 * k)
    GC.@preserve A B rc begin
        pA = Ptr{Float64}(pointer(A)); pB = Ptr{Float64}(pointer(B)); prc = pointer(rc)
        @inbounds for j in 0:(k - 1)
            z = unit ? one(ComplexF64) : _crecip(unsafe_load(Ptr{ComplexF64}(pA), j * lda + j + 1))
            unsafe_store!(prc, real(z), 2j + 1); unsafe_store!(prc, imag(z), 2j + 2)
        end
        # `Val(!_EXPFLAG[_EXP14])` — the fold SHIPS ON; the flag is INVERTED so the old scalar-negate
        # arm stays A/B-able in-process on a fleet box (see the _EXP14 registry note).
        # The ragged ROW block (m % W rows) rides the SAME tile under a mask — see `_zrt_tile!`.
        if _EXPFLAG[_EXP14]
            _zrt_sweep!(Val(false), pB, ldb, pA, lda, prc, k, m, mb, W, _ZRT_NC)
        else
            _zrt_sweep!(Val(true), pB, ldb, pA, lda, prc, k, m, mb, W, _ZRT_NC)
        end
    end
    return B
end

function _trsm_cmplx_small_R!(up::Bool, tr::Bool, cj::Bool, unit::Bool, k::Int, A, B)
    if up && !tr && _cgt_ok(A, B) && size(B, 1) >= _ZGT_W          # register tile (lanes = rows of B)
        return _trsm_zrt_R!(unit, k, A, B)
    end
    # `eltype(A) === eltype(B)` guards the same memory-safety hole documented on `_cgt_ok`: these bases
    # do `Ptr{eltype(B)}(pointer(A))`, so a mismatched A is read at the wrong element width and runs off
    # its allocation. Mismatched eltypes fall through to the generic path, which handles them correctly.
    if !tr && k <= _fh_ctrsm_direct_max() && _strided1(B) && eltype(A) === eltype(B)   # transA='N' direct
        return _trsm_cmplx_dRN!(up, unit, k, A, B)                               # column-substitution
    elseif tr && cj && k <= _fh_ctrsm_direct_max() && _strided1(B) && eltype(A) === eltype(B)  # transA='C'
        return _trsm_cmplx_dRC!(up, unit, k, A, B)                                        # direct, no trtri
    end
    T = eltype(B); Vv = view(_trsm_tmp(T, k, k), 1:k, 1:k)
    _trtri!(Vv, A, k, up, unit)
    return (_CTRMM_PACK && k >= _fh_ctrmm_pack_min()) ? _trmm_cmplx_packed_R!(up, tr, cj, false, k, Vv, B) :
        _trmm_cmplx_small_R!(up, tr, cj, false, k, Vv, B)
end

# ===== Fused gemmtrsm (side-L, upper, no-trans) — BLIS/AOCL-style diagonal-block leaf =====
# Replaces the invL 2×-flop leaf for the trsm gate shape (solve U·X=B, U upper k×k, B k×n wide).
# Design (Fable + BLIS bli_dgemmtrsm_u): RHS columns → SIMD lanes; per MR-row slab, accumulate the
# trailing gemm (rows below, already solved) over ALL rows below, then an in-register MR×MR back-
# substitution with pre-inverted diagonal. Beats invL (1× flop vs 2×) and the old serial epilogue
# (MR-granular chain of NR-wide vector steps + full-k amortization hides the substitution latency).
# The KC×NR column-stripe of B is packed ROW-MAJOR into P (row i at Pp+i·NR, NR contiguous = NRV
# vectors); U is read column-major directly (scalar broadcasts; KC×KC block is L2-resident).
#
# Tile geometry DERIVED from the register file (req#8): NRV column-vectors/row + MR rows so that
# MR·NRV accumulators + NRV RHS-slice + 2 broadcast/slack ≤ nreg. nreg=32(AVX-512)→8×24 f64 /8×48 f32;
# nreg=16(AVX2)→6×8. Reproduces BLIS zen4 8×24 / haswell 6×8. Preferences-overridable (fleet calib).
const _GT_NREG = _SIMD_BYTES >= 64 ? 32 : 16
# PDM: Derived — formula over detected consts: `_GT_NREG >= 32 ? 3 : 2`
const _GT_NRV = @load_preference("gemmtrsm_nrv", _GT_NREG >= 32 ? 3 : 2)::Int
# PDM: Derived — formula over detected consts: `min(8, (_GT_NREG - _GT_NRV - 2) ÷ _GT_NRV`
const _GT_MR = @load_preference("gemmtrsm_mr", min(8, (_GT_NREG - _GT_NRV - 2) ÷ _GT_NRV))::Int
const _GT_W = _vwidth(Float64)                  # fused leaf is f64-only; SIMD lanes over RHS columns
const _GT_NR = _GT_NRV * _GT_W                   # columns per stripe (SIMD-lane block width)
# The vectorized 8×8-transpose pack (contiguous B reads, po2-immune, ~5% stripe vs the scalar 15%) is
# AVX-512-f64-specific (needs W==8 lanes). On W≠8 the leaf uses the scalar pack + the orientation predicate.
const _GT_TRANSPOSE = (_GT_W == 8)
# Fused-leaf k-cutoff. The recursion feeds the largest ½-split ≤ base; a bigger leaf runs MORE of its internal
# off-diagonal at the ~35-GF leaf rate instead of the recursion's ~43-GF peak, so keep it modest. AVX-512:
# DERIVE from the P-stripe FULL-L1 residency — the KC×NR row-major P stripe is the ONLY hot streamed operand
# (the compact U is L2-resident, recips tiny), so size KC so the stripe fills L1: KC·NR·sizeof ≤ L1 ⇒ 170 on
# Zen4 (L1=32K), 256 on Zen5 (L1=48K). FLEET-VALIDATED boost-locked 2026-07-16: the old ½-L1 (⇒85) was a
# warm-micro MIS-TUNE — full-L1 nets +1.5–5.8pt on Zen4 (n=32 0.918→0.976, n≥512 +1.5–3.6pt), Zen5 INSENSITIVE
# (safe). Non-AVX-512: keep the 128 literal — Zen3/AVX2 optimum (measured; a bigger base REGRESSES it, n=256
# 0.996→0.85). req#8 Preferences-override "trsm_fused_base" still applies for calibration.
const _TRSM_FUSED_BASE = @load_preference(
    "trsm_fused_base",
    _GT_TRANSPOSE ? max(_GT_MR, _L1_BYTES ÷ (_GT_NR * sizeof(Float64))) : 128
)::Int
# Lower crossover for the fused leaf: below this k the pack-U + ftrsm-buffer setup isn't amortized, so the
# scalar dense base wins on the sub-µs tiny solve. DERIVE, keyed on the SIMD width because the setup is
# _GT_W-lane transposes. Now validated on BOTH µarchs, so the req#8b debt note is discharged:
#   Zen4 (2·_GT_W = 16): fused beats dense from k=16 (k=24 15.9 vs 9.3 GF); below it dense wins
#                        (k=8 2.2 vs 1.7 GF) — the crossover sits AT the formula.
#   Zen3 (2·_GT_W =  8): direct-leaf paired A/B 2026-08-10, routing bypassed, correctness 2e-16..2e-15 —
#                        fused/dense = 0.4932 (k=8), 0.5596 (12), 0.4747 (16), 0.8726 (24), 0.8335 (32).
#                        Fused wins at EVERY k down to the formula's value, so the crossover is at or
#                        below 8 and the formula is safe (conservative if anything) on AVX2.
# Two boxes, opposite W, crossover tracking 2·_GT_W both times ⇒ physically predictable ⇒ stays DERIVE.
# No Measure tier: that is for optima that INVERT across µarchs, and this one does not.
const _TRSM_FUSED_MIN = 2 * _GT_W
# Same-process A/B switch: false ⇒ wide-B upper falls back to the invL leaf (old behaviour). Default on;
# the Ref load is negligible vs the O(n²) solve. ponytail: exists for controlled A/B; harmless in prod.
const _TRSM_FUSED_ON = Ref(true)
# When true, the AVX-512 transpose pack falls back to the alias-free column-outer scalar pack on a 4K-aliased
# ldb (see the useT note). ponytail: A/B toggle now; promote to always-on (derive from the aliasing predicate,
# no Ref) once measured to gate. Default matches shipped behaviour until validated.
# Lever 1 (default ON): the AVX-512 fused-leaf + fullpack sweeps use the fused first/last-touch-transpose
# slabs (`_gemmtrsm_u_slab_upk_fusedT!` / `_gemmtrsm_u_slab_fusedT!`) for full stripes — read own rows
# straight from B (in-register 8×8 transpose), gemm-reuse from row-major P, store to P + transposed to B —
# eliminating the standalone _fused_packP_tr!/_fused_unpackP_tr! passes (the 9.4%-of-stripe pack round-trip
# at k=128). Bit-identical to the pack path; measured strict no-regression win fleet-dev (Zen5): n=512
# 0.930→0.960, 1024 0.970→0.989, 513 0.998→1.058 vs AOCL. AVX-512-f64 (W==MR==8) only; no-op elsewhere.
# ponytail: kept as a Ref for controlled A/B (mirrors _TRSM_FUSED_ON); the read is trivial vs the O(n²) solve.
const _TRSM_FUSEDT_ON = Ref(true)

# One MR-row slab, one NR-column stripe: acc = B[slab] − U12·X21 (gemm, kk over solved rows below),
# then in-register MR×MR back-substitution (bottom-up, critical-path-first), stored back into P.
# Pp: &P at (row 0, this stripe's column) — P is the WHOLE KC×ncol row-major stripe buffer, row stride
# ldp (elements). Up: &U[0,0], ldu. rp: &recip[0]. s: slab top row. Slabs OUTER / stripes INNER in the
# driver ⇒ for a fixed slab the U reads hit the same addresses across stripes → U read L2→L1 ONCE, reused.
@inline @generated function _gemmtrsm_u_slab!(
        Pp::Ptr{T}, ldp::Int, Up::Ptr{T}, ldu::Int, rp::Ptr{T},
        s::Int, KC::Int, ::Val{MR}, ::Val{NRV}
    ) where {T, MR, NRV}
    W = _vwidth(T); sz = sizeof(T); V = Vec{W, T}
    body = quote end
    for r in 0:(MR - 1), v in 0:(NRV - 1)
        push!(body.args, :($(Symbol(:c, r, :_, v)) = zero($V)))
    end
    inner = quote end                                     # gemm: acc[r][v] += U[s+r,kk]·P[kk][v]
    for v in 0:(NRV - 1)
        push!(inner.args, :($(Symbol(:x, v)) = vload($V, Pp + (kk * ldp + $(v * W)) * $sz)))
    end
    for r in 0:(MR - 1)
        push!(inner.args, :(u = $V(unsafe_load(Up + ((s + $r) + kk * ldu) * $sz))))
        for v in 0:(NRV - 1)
            cs = Symbol(:c, r, :_, v)
            push!(inner.args, :($cs = muladd(u, $(Symbol(:x, v)), $cs)))
        end
    end
    push!(
        body.args, :(
            for kk in UnitRange(s + $MR, KC - 1)
                $inner
            end
        )
    )
    for r in 0:(MR - 1), v in 0:(NRV - 1)                          # subtract: acc = B[s+r] − acc
        cs = Symbol(:c, r, :_, v)
        push!(body.args, :($cs = vload($V, Pp + ((s + $r) * ldp + $(v * W)) * $sz) - $cs))
    end
    for i in (MR - 1):-1:0                                     # back-substitution, critical-path-first
        push!(body.args, :(d = $V(unsafe_load(rp + (s + $i) * $sz))))
        for v in 0:(NRV - 1)
            push!(body.args, :($(Symbol(:c, i, :_, v)) = $(Symbol(:c, i, :_, v)) * d))
        end
        for j in (i - 1):-1:0
            push!(body.args, :(u = $V(unsafe_load(Up + ((s + $j) + (s + $i) * ldu) * $sz))))
            for v in 0:(NRV - 1)
                cj = Symbol(:c, j, :_, v); ci = Symbol(:c, i, :_, v)
                push!(body.args, :($cj = muladd(-u, $ci, $cj)))
            end
        end
    end
    for r in 0:(MR - 1), v in 0:(NRV - 1)
        push!(body.args, :(vstore($(Symbol(:c, r, :_, v)), Pp + ((s + $r) * ldp + $(v * W)) * $sz)))
    end
    push!(body.args, :(return nothing))
    return body
end

# Compact-U fused first/last-touch-transpose slab (Lever 1 for the SHIPPED recursion leaf). Same as
# `_gemmtrsm_u_slab!` (reads U from the compact odd-ldu panel, col-major) but reads its own MR rows
# STRAIGHT from B (in-register 8×8 transpose) for the subtract and writes solved rows to P (reuse) AND
# transposed to B (final) — no standalone pack/unpack. REQUIRES W==MR==8 (AVX-512 f64), FULL stripe.
# Body builder for the unpacked fusedT slab, shared by the runtime method below and the tiny-k `Val{NG}`
# method further down. `s`/`KC` name runtime arguments (`Symbol`) or are literals; `ldp` likewise; `ng`, if
# given, is the LITERAL gemm trip count and replaces the `s+MR : KC-1` bound with a `0:ng-1` counted loop.
# WHY the literal-`ng` form exists: `@inline` does NOT propagate into a @generated body on Julia 1.12 (kb:
# generated-inline-meta-hazard), so the runtime method survives as an `invoke` and its caller's `s`/`KC` can
# never const-propagate in. That is harmless at large KC and first-order at tiny KC, where the four slabs
# run only 0/8/16/24 gemm trips each: measured, the leaf sits 7-10x off its own FMA roofline at K=32 and
# the miss is FLAT in n, the signature of per-iteration scaffolding rather than a fixed per-call cost
# (bench/probes/trsm_leaf_shape.jl). Only the TRIP COUNT is lifted to compile time — the `s` offsets stay
# runtime address math, which is cheap and, unlike inlining the whole sweep into one function, keeps the
# 24 live accumulators inside ONE slab's register-allocation scope.
function _slab_upk_fusedT_body(
        ::Type{T}, MR::Int, NRV::Int, s, KC, ldp, ng::Union{Nothing, Int} = nothing,
        incaddr::Bool = false
    ) where {T}
    return _slab_upk_fusedT_body_1(T, MR, NRV, s, KC, ldp, ng, 0, incaddr)
end

# The core emitter. `vbase` shifts this sweep's v-columns to lane group `vbase` of the NRV·W-wide stripe:
# P and B offsets keep the FULL stripe geometry (`ldp`, and the `jc + v*W` column), so per-v and fused
# emit identical memory traffic and only the register working set differs. NOT a keyword — keywords do
# not participate in dispatch, so a `; vbase` overload would silently REPLACE the method above rather
# than sit beside it.
# Emits BOTH orders behind a runtime branch on _EXPFLAG[_EXP6]. It must be runtime: this generator
# runs at GENERATION time, so a generator-level test would bake the choice in at first specialisation
# and the A/B toggle would silently do nothing (the same trap the _EXP2 prefetch generator hit).
function _slab_upk_fusedT_body_1(
        ::Type{T}, MR::Int, NRV::Int, s, KC, ldp, ng::Union{Nothing, Int}, vbase::Int,
        incaddr::Bool = false
    ) where {T}
    # _EXP6 (subtract hoist) is NOT wired: emitting both orders behind a runtime branch doubles every
    # slab specialisation and the build did not finish in 72 minutes. If it is ever revisited, select the
    # order at GENERATION time and pay a session restart per arm instead — the runtime-toggle trick that
    # works for cheap flags does not scale to a structurally different body.
    #
    # GEMM ITERATION ROTATION — TRIED AND FALSIFIED (2026-08-11), and this one closes a whole class.
    # Slabs run bottom-up, so slab `s` sums over kk ∈ [s+MR, KC) and the FIRST MR of those are exactly the
    # rows the immediately preceding slab just wrote. FP accumulation is not reassociable, so on paper
    # every accumulator chain threads those dependent head iterations before reaching the independent
    # majority — a cross-slab dataflow fence, once per slab, 16 times per stripe at KC=128, and static
    # accounting priced the serial spine at ~4% of a stripe. Rotating the loop to emit [s+2MR, KC) first
    # and the dependent [s+MR, s+2MR) LAST should let the independent iterations issue while the previous
    # chain drains, at identical instruction count and zero extra L1 footprint.
    # Built behind a generation-time `gemmtrsm_rot` Preference (a restart per arm, per the note above).
    # NULL: n=128 18.52 vs 18.55 GF, n=256 19.48 vs 19.43, n=512 19.84 vs 19.58, n=1024 20.29 vs 20.21 —
    # all inside the ±1-2% run spread, against a ≥+2% bar at the binding cell.
    # The arm was LIVE, not dead: 1162 → 1686 instructions, 8 → 16 branches, FMA 300 → 516 (the dependent
    # tail peels and unrolls). So out-of-order execution ALREADY hides the fence, and the ~4% serial-spine
    # estimate that motivated this does not survive contact — do not re-derive a lever from it.
    # CONSEQUENCE: schedule-level ILP on this kernel is exhausted. Pairing (a second stripe) fails on L1
    # capacity, rotation (self-supplied slack) is a null, and the chain itself is order-forced. What
    # remains for n=128/256 is a formulation with a different B access, i.e. the row-lane family.
    return _slab_body_ord(T, MR, NRV, s, KC, ldp, ng, vbase, false, Symbol(""), incaddr)
end

function _slab_body_ord(
        ::Type{T}, MR::Int, NRV::Int, s, KC, ldp, ng::Union{Nothing, Int}, vbase::Int, hoist::Bool,
        pfx::Symbol = Symbol(""), incaddr::Bool = false
    ) where {T}
    W = _vwidth(T); sz = sizeof(T); V = Vec{W, T}
    body = quote end
    # _EXP6 — SUBTRACT HOIST (miss-placement lever). Shipped order is: zero the accumulators, run the
    # gemm, THEN load B and transpose it, then back-substitute. That puts B's cold miss burst
    # immediately in front of the dependent back-substitution chain, where nothing can hide it — and
    # the bottom slab, which runs FIRST, has ng==0, so it is pure exposed latency.
    # Hoisted order issues the B loads + transpose BEFORE the gemm and seeds the accumulators with
    # -B instead of zero, flipping the gemm to c -= u*x. Identical instruction count and identical
    # register pressure (the `bc`/`br` values die into the accumulators before the gemm begins), so the
    # WARM arm must be a statistical null — that null is the built-in control for this lever.
    # Counters justify the target: at the gate working set we already fetch the compulsory ~222
    # lines/call from L3+DRAM and overlap ~91% of that latency; only a handful of EXPOSED misses remain.
    # `pfx` namespaces every local, so TWO independent slabs can be emitted into ONE body and the
    # scheduler can overlap their back-substitution chains. That is the ILP lever: measured, PB runs at
    # IPC 1.16 against AOCL's 2.20 and pays 114 ns more exposed first-touch latency, while its kernel is
    # 6.8% FASTER once pages are resident. More chains in flight is the only thing that addresses that.
    cc_(r, v) = Symbol(pfx, :c, r, :_, v)
    u_ = Symbol(pfx, :u); d_ = Symbol(pfx, :d)
    # INCREMENTAL B ADDRESSING (`incaddr`). The shipped form recomputes `(jc + c)*ldb` for each of the
    # NRV*W column constants c, on BOTH the load and the store side — with `jc` and `ldb` runtime, that is
    # a genuine integer multiply per access. Measured in the emitted code (P6, 2026-08-10): 27 `imul` and
    # 192 scalar address instructions per slab, 22.1% of the out-of-loop instruction count, which itself
    # is the 18.6% fixed cost the leaf's `1/GF = α + β/KC` fit prices. Neither the transpose (~3.2% of
    # total) nor the back-substitution (~2.8%) dominates that cost — this addressing does.
    # The columns touched are CONSECUTIVE (vbase*W .. vbase*W + NRV*W - 1), so one base pointer plus a
    # running `+= ldb*sz` replaces every one of those multiplies. Bit-identical: addressing only, the
    # arithmetic and its order are untouched.
    pbl_ = Symbol(pfx, :pbl); pbs_ = Symbol(pfx, :pbs); ldbs_ = Symbol(pfx, :ldbs)
    if incaddr
        push!(body.args, :($ldbs_ = ldb * $sz))
        push!(body.args, :($pbl_ = pB + (($s) + (jc + $(vbase * W)) * ldb) * $sz))
        push!(body.args, :($pbs_ = $pbl_))
    end
    bload_(v, l) = incaddr ? :($pbl_) :
        :(pB + (($s) + (jc + $((vbase + v) * W + l)) * ldb) * $sz)
    bstore_(v, l) = incaddr ? :($pbs_) :
        :(pB + (($s) + (jc + $((vbase + v) * W + l)) * ldb) * $sz)
    subtract = quote end
    for v in 0:(NRV - 1)
        for l in 0:(W - 1)
            push!(subtract.args, :($(Symbol(pfx, :bc, l)) = vload($V, $(bload_(v, l)))))
            incaddr && push!(subtract.args, :($pbl_ = $pbl_ + $ldbs_))
        end
        brs = Expr(:tuple, (Symbol(pfx, :br, l) for l in 0:(W - 1))...)
        push!(subtract.args, Expr(:(=), brs, :(_tr8x8($((Symbol(pfx, :bc, l) for l in 0:(W - 1))...)))))
        for r in 0:(MR - 1)
            cs = cc_(r, v)
            # hoisted: seed with B (gemm will subtract into it); shipped: acc holds Σ, so B - Σ.
            push!(subtract.args, :($cs = $(hoist ? Symbol(pfx, :br, r) : :($(Symbol(pfx, :br, r)) - $cs))))
        end
    end
    if hoist
        append!(body.args, subtract.args)                 # B first: its misses overlap the gemm below
    else
        for r in 0:(MR - 1), v in 0:(NRV - 1)
            push!(body.args, :($(cc_(r, v)) = zero($V)))
        end
    end
    inner = quote end                                     # gemm: acc[r][v] ±= U[s+r,kk]·P[kk][v]
    for v in 0:(NRV - 1)
        push!(inner.args, :($(Symbol(pfx, :x, v)) = vload($V, Pp + (kk * $ldp + $((vbase + v) * W)) * $sz)))
    end
    for r in 0:(MR - 1)
        push!(inner.args, :($u_ = $V(unsafe_load(Up + (($s + $r) + kk * ldu) * $sz))))
        for v in 0:(NRV - 1)
            cs = cc_(r, v)
            # hoisted accumulators already hold B, so the gemm SUBTRACTS; otherwise it sums and the
            # subtract step below computes B - Σ.
            push!(inner.args, hoist ? :($cs = muladd(-$u_, $(Symbol(pfx, :x, v)), $cs)) :
                :($cs = muladd($u_, $(Symbol(pfx, :x, v)), $cs)))
        end
    end
    if isnothing(ng)
        push!(
            body.args, :(
                for kk in UnitRange($s + $MR, $KC - 1)
                    $inner
                end
            )
        )
    elseif ng > 0                                          # literal trip count; ng==0 emits no loop at all
        push!(
            body.args, :(
                for kkoff in 0:$(ng - 1)
                    kk = $s + $MR + kkoff
                    $inner
                end
            )
        )
    end
    # Shipped order: B is read AFTER the gemm, so its miss burst lands directly in front of the
    # dependent back-substitution. Hoisted order emitted this block before the gemm instead (_EXP6).
    hoist || append!(body.args, subtract.args)
    # NOT incrementalised, deliberately — MEASURED WORSE. The same trick that wins on B (below/above) was
    # applied to U here: `(s+i)*ldu` is a runtime multiply per i and the j-addresses are consecutive, so a
    # column pointer walked by ldu*sz removes them. It REGRESSED: leaf KC=128 gain fell 3.0% -> 1.5%, and
    # gate n=512 went 0.6% NEGATIVE. Cause: the pointer update is a serial dependency inserted into the
    # BACK-SUBSTITUTION, which is already the critical path — the addressing multiplies it removes were
    # off the critical path, being computed in parallel with the chain. On B the same edit is a clear win
    # because those addresses feed independent loads/stores. Do not retry without changing the chain.
    for i in (MR - 1):-1:0                                     # back-substitution, critical-path-first
        push!(body.args, :($d_ = $V(unsafe_load(rp + ($s + $i) * $sz))))
        for v in 0:(NRV - 1)
            push!(body.args, :($(cc_(i, v)) = $(cc_(i, v)) * $d_))
        end
        for j in (i - 1):-1:0
            push!(body.args, :($u_ = $V(unsafe_load(Up + (($s + $j) + ($s + $i) * ldu) * $sz))))
            for v in 0:(NRV - 1)
                cj = cc_(j, v); ci = cc_(i, v)
                push!(body.args, :($cj = muladd(-$u_, $ci, $cj)))
            end
        end
    end
    for v in 0:(NRV - 1)                                       # store: → P row-major (reuse) AND transposed → B
        for r in 0:(MR - 1)
            push!(
                body.args,
                :(vstore($(cc_(r, v)), Pp + (($s + $r) * $ldp + $((vbase + v) * W)) * $sz))
            )
        end
        ccs = Expr(:tuple, (Symbol(pfx, :cc, l) for l in 0:(W - 1))...)
        push!(body.args, Expr(:(=), ccs, :(_tr8x8($((cc_(r, v) for r in 0:(MR - 1))...)))))
        for l in 0:(W - 1)
            push!(body.args, :(vstore($(Symbol(pfx, :cc, l)), $(bstore_(v, l)))))
            incaddr && push!(body.args, :($pbs_ = $pbs_ + $ldbs_))
        end
    end
    return body
end

# PAIRED GENERAL SLAB (_EXP8) — the n=32 ILP win, generalised to any KC.
# Emits slab si of TWO independent column-stripes into ONE body (locals namespaced by prefix `a`/`b`),
# so the scheduler overlaps their back-substitution chains. Generated on (MR, NRV) ONLY — `s` and `KC`
# stay runtime — so code size is bounded and this works at every KC, unlike the Val{K}-unrolled tiny
# variant whose 16 slabs at KC=128 would explode.
# WHY IT SHOULD PAY MORE AT MID-n THAN AT n=32: the serial chain is (KC/MR) slabs x MR rows, so KC=128
# gives 128 dependent steps against n=32's 32 — four times the latency to hide, with still only one
# chain in flight today. Zen4 misses exactly there (128 0.911, 256 0.936, 512 0.961, 1024 0.984).
# Both stripes share one P of row stride 2*NRV*W: stripe A at column base 0, stripe B at NRV.
# The two stripes are ADJACENT (the stripe loop emits consecutive stripes), so one `jc` plus a column
# base of 0 and NRV addresses both — for B *and* for P. Passing two independent jc values would need
# vbase to shift P only, which it does not: vbase shifts both.
@inline @generated function _gemmtrsm_u_slab_pair!(
        Pp::Ptr{T}, pB::Ptr{T}, ldb::Int, jc::Int,
        Up::Ptr{T}, ldu::Int, rp::Ptr{T}, s::Int, KC::Int, ::Val{MR}, ::Val{NRV}
    ) where {T, MR, NRV}
    W = _vwidth(T)
    (W == 8 && MR == 8) ||
        return :(throw(AssertionError("paired fusedT slab requires W==MR==8 (AVX-512 f64)")))
    ldp = 2 * NRV * W                                    # shared P: stripe A [0,NRV*W), stripe B [NRV*W,ldp)
    body = quote end
    append!(body.args, _slab_body_ord(T, MR, NRV, :s, :KC, ldp, nothing, 0, false, :a).args)
    append!(body.args, _slab_body_ord(T, MR, NRV, :s, :KC, ldp, nothing, NRV, false, :b).args)
    push!(body.args, :(return nothing))
    return body
end

@inline @generated function _gemmtrsm_u_slab_upk_fusedT!(
        Pp::Ptr{T}, ldp::Int, pB::Ptr{T}, ldb::Int,
        jc::Int, Up::Ptr{T}, ldu::Int, rp::Ptr{T}, s::Int, KC::Int, ::Val{MR}, ::Val{NRV}, ::Val{ADDR}
    ) where {T, MR, NRV, ADDR}
    W = _vwidth(T)
    # AVX-512-f64 (W==MR==8) only. Generate a throw-body (never an assert) for other widths so GENERATION
    # always succeeds: the driver's fusedT branch is runtime-dead off AVX-512 (gated by _GT_TRANSPOSE), but
    # StrictMode's full-inference dogfood still expands this @generated call on AVX2 — a generation-time
    # @assert there crashes CI (the W=4 runner). The throw-expr compiles cleanly and is never executed.
    (W == 8 && MR == 8) || return :(throw(AssertionError("fusedT slab requires W==MR==8 (AVX-512 f64)")))
    # `ADDR` selects the B-addressing form at GENERATION time via a TYPE PARAMETER, so each specialisation
    # emits exactly one variant. It must not be an `_EXPFLAG` read inside this generator — that would bake
    # the choice in at first specialisation and leave a silently dead A/B arm (the _EXP2 prefetch trap).
    body = _slab_upk_fusedT_body(T, MR, NRV, :s, :KC, :ldp, nothing, ADDR)
    push!(body.args, :(return nothing))
    return body
end


# Ragged bottom slab (rem<MR rows, [base,KC) → no rows below, pure solve), one NR stripe. Rare (KC not a
# multiple of MR = non-power-of-2 leaves). NRV column-vectors, runtime row count. P row stride ldp.
@inline function _gemmtrsm_u_tail!(
        Pp::Ptr{T}, ldp::Int, Up::Ptr{T}, ldu::Int, rp::Ptr{T}, base::Int,
        KC::Int, ::Val{NRV}
    ) where {T, NRV}
    W = _vwidth(T); sz = sizeof(T); V = Vec{W, T}
    @inbounds for i in (KC - 1):-1:base
        d = V(unsafe_load(rp + i * sz))
        for v in 0:(NRV - 1)
            q = Pp + (i * ldp + v * W) * sz
            ci = vload(V, q) * d; vstore(ci, q)
            for j in (i - 1):-1:base
                u = V(unsafe_load(Up + (j + i * ldu) * sz))
                qj = Pp + (j * ldp + v * W) * sz
                vstore(muladd(-u, ci, vload(V, qj)), qj)
            end
        end
    end
    return nothing
end

# Driver: solve U·X = B in place, U upper KC×KC (view of A), B KC×n wide. up=true, no-trans, non-conj.
# Column stripes OUTER (NR at a time) / slabs INNER: each stripe's P (KC×NR ≈ L1) is packed, fully solved
# (all slabs), unpacked — P stays L1-hot for the whole stripe solve. U (KC×KC in L2) is re-read per stripe
# (cheap; L2-resident). Whole-B packing (P=KC×n) was measured WORSE (P spills L1) — keep P per-stripe.
# Float64 only: for Float32 the invL leaf (2×-flop at ~2×-higher f32 gemm peak) already beats the fused
# leaf even at n=128 (measured −18% if fused) — f32 stays on invL. The kernel itself is T-generic.
@inline _trsm_fusable(A, B) = eltype(B) === Float64 &&
    A isa StridedMatrix && B isa StridedMatrix && stride(A, 1) == 1 && stride(B, 1) == 1
# Transpose-packing B (col-major, stride ldb) → P (row-major, stride NR) has one unavoidable strided side.
# Row-outer keeps P writes contiguous but reads B at stride ldb·sz: for a leading dim whose byte stride
# shares a big power-of-2 factor with the L1 way size, the NR reads collapse onto few sets (n=256→2 sets,
# n=512→1) and overflow associativity → conflict-miss cliff. sets = way/gcd(way, ldb·sz); row-outer is
# safe while sets·assoc ≥ NR, else flip to column-outer (contiguous B reads, strided writes into the tiny
# L1-resident P). Derived from _L1_WAY_BYTES/_L1D_ASSOC (req#8), not a po2 test — n=128 (4 sets) stays
# row-outer (measured faster), only the truly-aliased strides pay the transpose the other way.
@inline function _fused_pack_rowouter(ldb::Int, NR::Int, sz::Int)
    stride_b = ldb * sz
    sets = _L1_WAY_BYTES ÷ gcd(_L1_WAY_BYTES, stride_b)
    return sets * _L1D_ASSOC >= NR
end

# In-register 8×8 Float64 transpose (AVX-512): 24 shuffles, unpacklo/hi → 128b → 256b lane stages. Used to
# pack/unpack the B↔P transpose contiguously (reads B DOWN columns = unit stride = po2-immune) AND fast
# (the scalar transpose pack is ~15% of stripe cycles / 33 GF; this drops it to ~5% / 38 GF — Fable-measured
# decomposition, the residual to the 43-GF dgemm peak). Only the W==8 f64 path; AVX2/edges keep the scalar
# column-outer/row-outer pack (already po2-immune via contiguous reads there).
@inline function _tr8x8(r0::V, r1::V, r2::V, r3::V, r4::V, r5::V, r6::V, r7::V) where {V}
    t0 = shufflevector(r0, r1, Val((0, 8, 2, 10, 4, 12, 6, 14))); t1 = shufflevector(r0, r1, Val((1, 9, 3, 11, 5, 13, 7, 15)))
    t2 = shufflevector(r2, r3, Val((0, 8, 2, 10, 4, 12, 6, 14))); t3 = shufflevector(r2, r3, Val((1, 9, 3, 11, 5, 13, 7, 15)))
    t4 = shufflevector(r4, r5, Val((0, 8, 2, 10, 4, 12, 6, 14))); t5 = shufflevector(r4, r5, Val((1, 9, 3, 11, 5, 13, 7, 15)))
    t6 = shufflevector(r6, r7, Val((0, 8, 2, 10, 4, 12, 6, 14))); t7 = shufflevector(r6, r7, Val((1, 9, 3, 11, 5, 13, 7, 15)))
    u0 = shufflevector(t0, t2, Val((0, 1, 8, 9, 4, 5, 12, 13))); u1 = shufflevector(t1, t3, Val((0, 1, 8, 9, 4, 5, 12, 13)))
    u2 = shufflevector(t0, t2, Val((2, 3, 10, 11, 6, 7, 14, 15))); u3 = shufflevector(t1, t3, Val((2, 3, 10, 11, 6, 7, 14, 15)))
    u4 = shufflevector(t4, t6, Val((0, 1, 8, 9, 4, 5, 12, 13))); u5 = shufflevector(t5, t7, Val((0, 1, 8, 9, 4, 5, 12, 13)))
    u6 = shufflevector(t4, t6, Val((2, 3, 10, 11, 6, 7, 14, 15))); u7 = shufflevector(t5, t7, Val((2, 3, 10, 11, 6, 7, 14, 15)))
    return (
        shufflevector(u0, u4, Val((0, 1, 2, 3, 8, 9, 10, 11))), shufflevector(u1, u5, Val((0, 1, 2, 3, 8, 9, 10, 11))),
        shufflevector(u2, u6, Val((0, 1, 2, 3, 8, 9, 10, 11))), shufflevector(u3, u7, Val((0, 1, 2, 3, 8, 9, 10, 11))),
        shufflevector(u0, u4, Val((4, 5, 6, 7, 12, 13, 14, 15))), shufflevector(u1, u5, Val((4, 5, 6, 7, 12, 13, 14, 15))),
        shufflevector(u2, u6, Val((4, 5, 6, 7, 12, 13, 14, 15))), shufflevector(u3, u7, Val((4, 5, 6, 7, 12, 13, 14, 15))),
    )
end
# 4×4 register transpose (AVX2, Vec{4,Float64}) — the W==4 mirror of _tr8x8 (8 shufflevectors). Turns 4
# B-columns (loaded as 4 Vec{4} rows) into 4 transposed rows for the row-major P pack (Fable lever 2).
@inline function _tr4x4(r0::V, r1::V, r2::V, r3::V) where {V}
    t0 = shufflevector(r0, r1, Val((0, 4, 2, 6))); t1 = shufflevector(r0, r1, Val((1, 5, 3, 7)))
    t2 = shufflevector(r2, r3, Val((0, 4, 2, 6))); t3 = shufflevector(r2, r3, Val((1, 5, 3, 7)))
    return (
        shufflevector(t0, t2, Val((0, 1, 4, 5))), shufflevector(t1, t3, Val((0, 1, 4, 5))),
        shufflevector(t0, t2, Val((2, 3, 6, 7))), shufflevector(t1, t3, Val((2, 3, 6, 7))),
    )
end
# AVX2 vectorized transpose pack (W==4): B[:,jc:jc+wid) → P row-major via 4×4 transpose for full 4-row×4-col
# blocks, scalar tails/zero-pad. Replaces the scalar rowouter/column-outer pack (Fable: ~20-25% of the AVX2
# leaf's cycles). Same row-major P layout the slab reads — no microkernel change.
@inline function _fused_packP_tr4!(
        Pp::Ptr{Float64}, pB::Ptr{Float64}, ldb::Int, jc::Int, wid::Int,
        KC::Int, NR::Int, sz::Int
    )
    V = Vec{4, Float64}; ng = (KC >> 2) << 2; ncg = (wid >> 2) << 2
    @inbounds for i0 in 0:4:(ng - 1), vb in 0:4:(ncg - 1)
        b = pB + (i0 + (jc + vb) * ldb) * sz
        x0 = vload(V, b); x1 = vload(V, b + ldb * sz); x2 = vload(V, b + 2ldb * sz); x3 = vload(V, b + 3ldb * sz)
        y0, y1, y2, y3 = _tr4x4(x0, x1, x2, x3)
        p = Pp + (i0 * NR + vb) * sz
        vstore(y0, p); vstore(y1, p + NR * sz); vstore(y2, p + 2NR * sz); vstore(y3, p + 3NR * sz)
    end
    @inbounds for v in ncg:(wid - 1)                          # tail columns (contiguous B reads down the column)
        scol = pB + (jc + v) * ldb * sz; dcol = Pp + v * sz
        for i in 0:(KC - 1)
            unsafe_store!(dcol + i * NR * sz, unsafe_load(scol + i * sz))
        end
    end
    @inbounds for i in ng:(KC - 1)                            # tail rows (only full-col groups left)
        for v in 0:(ncg - 1)
            unsafe_store!(Pp + (i * NR + v) * sz, unsafe_load(pB + (i + (jc + v) * ldb) * sz))
        end
    end
    return @inbounds for v in wid:(NR - 1), i in 0:(KC - 1)
        unsafe_store!(Pp + (i * NR + v) * sz, zero(Float64))
    end
end
@inline function _fused_unpackP_tr4!(
        Pp::Ptr{Float64}, pB::Ptr{Float64}, ldb::Int, jc::Int, wid::Int,
        KC::Int, NR::Int, sz::Int
    )
    V = Vec{4, Float64}; ng = (KC >> 2) << 2; ncg = (wid >> 2) << 2
    @inbounds for i0 in 0:4:(ng - 1), vb in 0:4:(ncg - 1)
        p = Pp + (i0 * NR + vb) * sz
        y0 = vload(V, p); y1 = vload(V, p + NR * sz); y2 = vload(V, p + 2NR * sz); y3 = vload(V, p + 3NR * sz)
        x0, x1, x2, x3 = _tr4x4(y0, y1, y2, y3)
        b = pB + (i0 + (jc + vb) * ldb) * sz
        vstore(x0, b); vstore(x1, b + ldb * sz); vstore(x2, b + 2ldb * sz); vstore(x3, b + 3ldb * sz)
    end
    @inbounds for v in ncg:(wid - 1)
        scol = Pp + v * sz; dcol = pB + (jc + v) * ldb * sz
        for i in 0:(KC - 1)
            unsafe_store!(dcol + i * sz, unsafe_load(scol + i * NR * sz))
        end
    end
    return @inbounds for i in ng:(KC - 1)
        for v in 0:(ncg - 1)
            unsafe_store!(pB + (i + (jc + v) * ldb) * sz, unsafe_load(Pp + (i * NR + v) * sz))
        end
    end
end
# Pack B[:,jc:jc+wid) → P row-major (row i at Pp+i·NR) via 8×8 transpose for full 8-row × 8-col blocks;
# scalar column-outer for the ragged row/col tails and the wid:NR zero-pad. Contiguous B reads throughout.
# One 8x8 transposed block, factored out so the two visit orders below share an identical body.
@inline function _packP_tr_blk!(
        Pp::Ptr{Float64}, pB::Ptr{Float64}, ldb::Int, jc::Int, NR::Int, sz::Int, i0::Int, vb::Int
    )
    V = Vec{8, Float64}
    b = pB + (i0 + (jc + vb) * ldb) * sz
    x0 = vload(V, b);             x1 = vload(V, b + ldb * sz);   x2 = vload(V, b + 2ldb * sz); x3 = vload(V, b + 3ldb * sz)
    x4 = vload(V, b + 4ldb * sz); x5 = vload(V, b + 5ldb * sz);  x6 = vload(V, b + 6ldb * sz); x7 = vload(V, b + 7ldb * sz)
    y0, y1, y2, y3, y4, y5, y6, y7 = _tr8x8(x0, x1, x2, x3, x4, x5, x6, x7)
    p = Pp + (i0 * NR + vb) * sz
    vstore(y0, p);              vstore(y1, p + NR * sz);   vstore(y2, p + 2NR * sz); vstore(y3, p + 3NR * sz)
    vstore(y4, p + 4NR * sz);   vstore(y5, p + 5NR * sz);  vstore(y6, p + 6NR * sz); vstore(y7, p + 7NR * sz)
    return nothing
end

@inline function _fused_packP_tr!(
        Pp::Ptr{Float64}, pB::Ptr{Float64}, ldb::Int, jc::Int, wid::Int,
        KC::Int, NR::Int, sz::Int
    )
    V = Vec{8, Float64}; ng = (KC >> 3) << 3; ncg = (wid >> 3) << 3
    # LOOP ORDER IS THE EXPERIMENT (_EXPFLAG[_EXP4]).
    # Shipped order is i0 (rows) OUTER, vb (columns) INNER: the inner step jumps 8*ldb bytes across the
    # whole panel width, then the outer loop RETURNS to re-sweep at the next row block — a repeatedly
    # restarting strided walk, which is the opposite of what a hardware prefetcher tracks. AOCL's pack is
    # a monotone ascending unit-stride sweep. Swapping to vb OUTER, i0 INNER gives, per 8-column group,
    # 8 concurrent unit-stride streams advancing one cache line per step, ascending through the panel.
    # Same loads, same stores, same transposes — only the visit order changes.
    if _EXPFLAG[_EXP4]
        @inbounds for vb in 0:8:(ncg - 1), i0 in 0:8:(ng - 1)
            _packP_tr_blk!(Pp, pB, ldb, jc, NR, sz, i0, vb)
        end
    else
        @inbounds for i0 in 0:8:(ng - 1), vb in 0:8:(ncg - 1)
            _packP_tr_blk!(Pp, pB, ldb, jc, NR, sz, i0, vb)
        end
    end
    @inbounds for v in ncg:(wid - 1)                          # tail columns (contiguous B reads down the column)
        scol = pB + (jc + v) * ldb * sz; dcol = Pp + v * sz
        for i in 0:(KC - 1)
            unsafe_store!(dcol + i * NR * sz, unsafe_load(scol + i * sz))
        end
    end
    @inbounds for i in ng:(KC - 1)                            # tail rows (only full-col groups left to do)
        for v in 0:(ncg - 1)
            unsafe_store!(Pp + (i * NR + v) * sz, unsafe_load(pB + (i + (jc + v) * ldb) * sz))
        end
    end
    return @inbounds for v in wid:(NR - 1), i in 0:(KC - 1)
        unsafe_store!(Pp + (i * NR + v) * sz, zero(Float64))
    end
end
@inline function _fused_unpackP_tr!(
        Pp::Ptr{Float64}, pB::Ptr{Float64}, ldb::Int, jc::Int, wid::Int,
        KC::Int, NR::Int, sz::Int
    )
    V = Vec{8, Float64}; ng = (KC >> 3) << 3; ncg = (wid >> 3) << 3
    @inbounds for i0 in 0:8:(ng - 1), vb in 0:8:(ncg - 1)
        p = Pp + (i0 * NR + vb) * sz
        y0 = vload(V, p);          y1 = vload(V, p + NR * sz);   y2 = vload(V, p + 2NR * sz); y3 = vload(V, p + 3NR * sz)
        y4 = vload(V, p + 4NR * sz); y5 = vload(V, p + 5NR * sz);  y6 = vload(V, p + 6NR * sz); y7 = vload(V, p + 7NR * sz)
        x0, x1, x2, x3, x4, x5, x6, x7 = _tr8x8(y0, y1, y2, y3, y4, y5, y6, y7)
        b = pB + (i0 + (jc + vb) * ldb) * sz
        vstore(x0, b);           vstore(x1, b + ldb * sz);   vstore(x2, b + 2ldb * sz); vstore(x3, b + 3ldb * sz)
        vstore(x4, b + 4ldb * sz); vstore(x5, b + 5ldb * sz);  vstore(x6, b + 6ldb * sz); vstore(x7, b + 7ldb * sz)
    end
    @inbounds for v in ncg:(wid - 1)                          # tail columns (contiguous B writes down the column)
        scol = Pp + v * sz; dcol = pB + (jc + v) * ldb * sz
        for i in 0:(KC - 1)
            unsafe_store!(dcol + i * sz, unsafe_load(scol + i * NR * sz))
        end
    end
    return @inbounds for i in ng:(KC - 1)                            # tail rows
        for v in 0:(ncg - 1)
            unsafe_store!(pB + (i + (jc + v) * ldb) * sz, unsafe_load(Pp + (i * NR + v) * sz))
        end
    end
end
# One fusedT stripe of TRUE width NRVe·W (no pad, no pack round-trip): mini-pack the ragged tail rows
# (rem<MR) for the full slabs' P-reuse, solve them, then run the fusedT slabs (read B direct, write B
# transposed). Val{NRVe} lets a RAGGED column stripe (wid = W or 2W, produced by n mod NR) run at its
# real width instead of the driver padding it to NR — that n-mod-NR padding was the ENTIRE small-n AOCL
# gap (s=32: 24-wide work for an 8-col tail ⇒ 1.5× flops ⇒ 0.67× ceiling; Fable-decomposed 2026-07-15).
@inline function _fusedT_stripe!(
        ::Val{NRVe}, Pp::Ptr{T}, pB::Ptr{T}, ldb::Int, jc::Int, pU::Ptr{T},
        ldu::Int, rp::Ptr{T}, KC::Int, nfull::Int, rem::Int, MR::Int, sz::Int
    ) where {T, NRVe}
    Ppw = NRVe * _vwidth(T)                               # P row stride = this stripe's real width
    if rem > 0
        b0 = nfull * MR
        @inbounds for v in 0:(Ppw - 1), i in 0:(rem - 1)
            unsafe_store!(Pp + ((b0 + i) * Ppw + v) * sz, unsafe_load(pB + ((b0 + i) + (jc + v) * ldb) * sz))
        end
        _gemmtrsm_u_tail!(Pp, Ppw, pU, ldu, rp, b0, KC, Val(NRVe))
        @inbounds for v in 0:(Ppw - 1), i in 0:(rem - 1)
            unsafe_store!(pB + ((b0 + i) + (jc + v) * ldb) * sz, unsafe_load(Pp + ((b0 + i) * Ppw + v) * sz))
        end
    end
    # INCREMENTAL B ADDRESSING, shipped on (Val(true)). Bit-identical — addressing only, arithmetic and
    # its order untouched (verified bit-for-bit at KC=128/256 n=240, KC=128 n=128, and ragged KC=170 n=96).
    # Measured leaf speedup, direct-call paired A/B at n=240: KC=64 0.9554, 128 0.9697, 192 0.9775,
    # 256 0.9779 — largest at small KC, as a per-slab FIXED cost must be. End-to-end at the gate shape
    # n=128: 0.9704 (SE 0.0025, n=92). At n>=512 it reads null because the leaf's TIME SHARE is only
    # 29%/15% there, so a 3% leaf win dilutes below resolution — that is dilution, not a dead arm.
    # Only ONE Val is instantiated so this costs no extra specialisation.
    for si in (nfull - 1):-1:0
        _gemmtrsm_u_slab_upk_fusedT!(Pp, Ppw, pB, ldb, jc, pU, ldu, rp, si * MR, KC, Val(MR), Val(NRVe), Val(true))
    end
    return
end

# TWO ADJACENT STRIPES, slabs interleaved — the n=32 ILP win generalised to any KC (_EXP8).
# Same shape as _fusedT_stripe! but each `si` step solves BOTH stripes in one body, so two independent
# back-substitution chains are in flight. `rem > 0` (ragged bottom rows) falls back to the single-stripe
# path: the tail needs its own mini-pack and is rare (KC not a multiple of MR).
@inline function _fusedT_stripe_pair!(
        ::Val{NRVe}, Pp::Ptr{T}, pB::Ptr{T}, ldb::Int, jc::Int, pU::Ptr{T},
        ldu::Int, rp::Ptr{T}, KC::Int, nfull::Int, MR::Int, sz::Int
    ) where {T, NRVe}
    for si in (nfull - 1):-1:0
        _gemmtrsm_u_slab_pair!(Pp, pB, ldb, jc, pU, ldu, rp, si * MR, KC, Val(MR), Val(NRVe))
    end
    return
end

# TINY-K SLAB: identical body to the general slab, but the gemm trip count is a literal `Val{NG}` and the
# P stride folds out of `Val{NRV}`. One specialization per (NG, NRV) actually reachable — NG runs over
# multiples of MR below the tiny-k cap, so this is a handful of small functions, not a copy of the sweep.
@generated function _gemmtrsm_u_slab_ng!(
        ::Val{NG}, ::Val{NRV}, Pp::Ptr{T}, pB::Ptr{T}, ldb::Int, jc::Int,
        Up::Ptr{T}, ldu::Int, rp::Ptr{T}, s::Int
    ) where {NG, NRV, T}
    W = _vwidth(T); MR = _GT_MR
    # Throw-body, never an assert: generation must succeed on AVX2 for StrictMode's inference dogfood
    # even though the caller's fusedT branch is runtime-dead there (same rule as the general slab).
    (W == 8 && MR == 8) ||
        return :(throw(AssertionError("tiny fusedT slab requires W==MR==8 (AVX-512 f64)")))
    body = _slab_upk_fusedT_body(T, MR, NRV, :s, nothing, NRV * W, NG)
    push!(body.args, :(return nothing))
    return body
end

# FALSIFIED LEVER — per-v slab (MR live accumulators instead of MR·NRV). DO NOT RETRY.
# The premise was sound and measured: the NRV=3 slab spills (asm scan, bench/probes/trsm_slab_spill.jl:
# 40-51 stack vmov with 23-38 RELOADS at NRV=3, 30-41 with 0-1 reloads at NRV=2, zero at NRV=1), and the
# v columns never couple, so sweeping them one at a time is the same arithmetic with half the registers.
# It LOSES on every shape — same-process ABBA, bench/probes/trsm_perv_ab.jl, Zen4 boost-locked:
#   k=32 n=32 1.0015 · k=32 n=24 1.0764 · k=32 n=64 1.1091 · k=24 n=24 1.1039 · k=16 n=16 1.0370
#   k=64 n=64 1.1953 · k=128 n=128 1.1031 · k=128 n=256 1.0939      (per-v/fused, >1 = per-v slower)
# WHY: back-substitution is a SERIAL chain and the fused slab hides its latency with NRV-wide ILP —
# NRV independent vector ops per step. Per-v runs that chain NRV times with none. The same effect is
# already on record for the fullpack path ("its MR×NRV=24 v-lanes across the MR rows already saturate
# ILP"). Spill traffic is real but second-order against back-sub latency; the register file is NOT the
# binding constraint here, the dependency chain is.
# WHAT THE SCAN DOES SUPPORT: NRV=2 is spill-free (0-1 reloads) AND keeps 2-wide ILP — that is the
# surviving lever, and it also removes the ragged Val(1) tail stripe that n=32 gets under NR=24
# (24+8 columns, so a quarter of them run at the very ILP the per-v result just showed is too thin).
#
# EXPERIMENT FLAG TABLE. Same-process A/B switches live here as INDICES into one pre-declared array,
# never as new `const` bindings. Reason is workflow, not style: Revise cannot introduce a new const into
# a loaded module, so every `const _FOO = Ref(false)` costs a full hot-session restart (~5 min of
# PureBLAS precompile) — that happened twice on 2026-08-09 and is pure dead time. Mutating an element of
# an existing array needs no restart, so an A/B knob added mid-session iterates in seconds. Generalises
# the `_TRSM_FUSEDT_ON` / `_TRSM_FULLPACK_ON` convention this file already uses. Read once per `trsm!`
# call (not per slab), so the load is free. Ships all-false: every index is an OFF-by-default probe knob.
#   [1] tiny-k stripe width NR=2W instead of NR=NRV·W
# Integer companion to _EXPFLAG, pre-declared for the same reason: a new const costs a full session
# restart (Revise leaves it unassigned, and the @eval workaround is unsafe when a @generated body reads
# it). Sweeps assign an ELEMENT.
#   _EXPINT[1]  extra Float64 slots between P and the U pack — workspace alignment (see the buf note)
#               FALSIFIED 2026-08-10: flat over a full L1-way period on both boxes. Kept inert (0) so the
#               negative result stays reproducible.
#   _EXPINT[2..4]  free
# WITNESS SLOTS exist because of the F1 failure on 2026-08-10: a pad sweep produced a clean, well-behaved,
# entirely believable null while the knob's branch was never in the call graph for that shape at all
# (square B at n=32 on AVX2 routes to `_trsm_dense_L!`). Reasoning about routing from the source is what
# FAILED; a witness is an execution fact. Zero the slot, run one untimed call, assert it is 1 — then
# measure. Never publish an A/B whose witness did not fire.
# A witness is PROBE SCAFFOLDING and is stripped once its campaign lands: unlike a knob (a read that
# const-folds or costs one predictable branch), a witness is an unconditional STORE to module-global
# state on a path that ships. Re-add one for the next campaign, in-session, via Revise — that is what the
# slot table is for. Removing the ztrsmR/ztrsm/trmm witnesses is why the gate was re-measured at the
# merge commit rather than at the commit that carried them.
#   _EXPINT[5]  trmm kc override (0 = derived default) — the AOCL/BLIS `bli_trmm_determine_kc` arm
#   _EXPINT[6]  witness: the kc `_trmm_packed!` actually ran with (proves the knob was live)
#   _EXPINT[7]  zgemm 3M MIN override (0 = _CGEMM_3M_MIN) — tests 3M below the shipped window
#   _EXPINT[8]  spare
#
# ⚠ EVERY CONSUMER READS THIS `@inbounds`. Adding a reader at index k WITHOUT growing this array is an
# OUT-OF-BOUNDS READ that no test will catch — `@inbounds` deletes the check, and a stale heap value
# that happens to be 0 looks exactly like "knob off". That is not hypothetical: on 2026-08-18 a revert
# removed a previous growth to 8 while a later commit re-added an `_EXPINT[7]` reader, shipping an OOB
# read in the complex-gemm dispatch. GROW THIS ARRAY IN THE SAME COMMIT AS ANY NEW INDEX.
const _EXPINT = fill(0, 8)
const _EXPFLAG = fill(false, 16)
# SLOT NAMES ARE DECLARED ONCE, HERE. A new experiment CLAIMS A FREE SLOT and writes method-body code
# only — no new binding, so Revise applies it in-session with zero recompile.
# Adding a named const per knob DEFEATS the table and costs a full restart each time: Revise declares a
# new top-level const without running its initializer, and the `@eval PureBLAS const X = n` workaround
# is unsafe when the const is read from a @generated body — it invalidates the generator, so JIT
# recompilation lands inside timed rounds (measured A/A sigma 0.008 -> 0.139). Do not add names below.
const _EXP1, _EXP2, _EXP3, _EXP4, _EXP5, _EXP6, _EXP7, _EXP8 = 1, 2, 3, 4, 5, 6, 7, 8
const _EXP9, _EXP10, _EXP11, _EXP12, _EXP13, _EXP14, _EXP15, _EXP16 = 9, 10, 11, 12, 13, 14, 15, 16
# REGISTRY — update these COMMENTS, never the const list above:
#   _EXP1  tiny-k stripe NR=2W instead of NRV*W          FALSIFIED (loses up to 11%)
#   _EXP2  tiny-k cold-operand prefetch                  FALSIFIED (3.2% slower, destabilises the cell)
#   _EXP3  bypass the Val{NG} tiny chain (A/B arm)       Val{NG} CONFIRMED KEEP (5.2%, 5.7 SE, n=240)
#   _EXP4  _fused_packP_tr! columns-outer (sequential)   FALSIFIED (1.0025, n=240, powered null)
#   _EXP5  force the compact-U pack at tiny k (directA OFF) — miss PLACEMENT lever
#   _EXP6  subtract hoist — NOT WIRED (both-orders emission doubles every slab; >72 min build)
#   _EXP7  INVERTED: set true to DISABLE the interleaved pair (A/B arm). Pair ships ON.
#   _EXP8  INVERTED: set true to DISABLE paired adjacent stripes. Pairing SHIPS ON for
#          KC <= _TRSM_DBASE only — FALSIFIED at larger KC (6.2/4.2/2.6% slower at k=128/256/512).
#   _EXP9  free again. It briefly held a HALF-LIVE load schedule for the 8×8 transpose pack: issue the
#          eight B loads as two batches of four with `_tr8x8`'s first stage between, so only four
#          po2-aliased lines are live at once — the pattern Zen3's four-column pack runs and does not
#          suffer from. FALSIFIED, and the reason is worth keeping: +0.5% at n=512 against a +4.6%
#          ldb-pad ceiling, −0.7% at n=1024, control flat, output bit-identical. `code_native` shows why
#          — LLVM RE-MERGED the loads (7 of 8 still issue before the first shuffle), so the source-level
#          split never reached the machine. The loads are independent and the optimiser has no model of
#          L1 set pressure, so there is no way to hold a load schedule apart from Julia here.
#          CONSEQUENCE: the associativity model is NOT falsified, but any fix must change the ADDRESSES
#          (the access pattern) rather than the order. Do not retry a scheduling variant of this.
#          Then it held a STAGED PACK, which does change the addresses: copy the KC×wid panel through an
#          odd-ld scratch (pass 1 reads each B column as ONE long contiguous run, so nothing collides;
#          pass 2 transposes out of the scratch, whose stride is not a way-stride multiple). Both passes
#          vectorised, gated on `_alias_ld(ldb)`, bit-identical, wired at both pack call sites.
#          ALSO FALSIFIED, and decisively: packed path −1.3% at n=512 and −0.5% at n=1024 against
#          ldb-pad ceilings of +3.9% and +3.0%; fusedT +0.2%/+0.3%; n=384 control flat everywhere.
#          The extra streaming pass costs MORE than the ~18%-of-leaf aliasing it removes — the same
#          verdict the scalar column-outer pack got (−0.9%), and vectorising both passes did not change
#          it. Note also that fusedT SKIPS the pack entirely for full-width stripes (`fusedT && wid == W`
#          takes `_fusedT_stripe_k!` and continues), so no packing-level change can reach the shipped
#          AVX-512 path at the gate shape anyway.
#          STANDING CONCLUSION: three packing/scheduling escapes are now falsified (column-outer scalar,
#          load-schedule split, staged copy). The aliasing is real and worth 4.5% at n=512 / 2.0% at
#          n=1024, but capturing it requires a microkernel whose B access is not eight aliased columns —
#          i.e. a genuine row-lane family, not an edit to the pack. Do not attempt another pack variant.
#   _EXP10 free again. It briefly lifted the `KC <= _TRSM_DBASE` cap on paired adjacent stripes, to test
#          whether pairing fails at KC=128 only because 2·MR·NRV = 48 accumulators spill against 32
#          registers — under a pinned NRV=2 a pair needs exactly 32 and cannot spill. FALSIFIED, and it
#          falsified the register explanation with it: lifting the cap costs −6.6/−5.7/−2.5% at
#          k=128/256/512 at NRV=3 (reproducing the recorded figures) but −10.8/−7.4/−4.1% at NRV=2,
#          i.e. pairing loses MORE where there is no spill. See the driver comment at the pairing gate
#          for the capacity model that does fit. NRV=2 is also worse unpaired (17.86 vs 18.44 GF at
#          n=128), so the shipped NRV=3 stands and no Preference change is warranted.
#          Previously — the A-side de-aliasing it gated SHIPPED unconditionally (see `_trsm_right!`),
#          on a derived predicate rather than a flag. Kept as a note because the measurements matter:
#          A-side de-aliasing for the fused side-R path (trsmR). At transA='T' the leaf reads A VERBATIM
#          at the caller's lda, and a quarter-period byte stride (lda=128 f64 ⇒ 1024 B) puts its columns
#          on the same L1 sets. Copies A's lower triangle into `_trsm_rpack`'s odd-ld scratch when the
#          derived `_potrf_needs_pad` fires. Leaf sweep, Zen3, bs=128 m=128, ONLY A's lda moving:
#          128 (shipped) 41.59 GF | 129 49.14 | 130 48.57 | 132 50.11 | 136 48.81 | 144 49.75
#          => any non-po2 lda is +11.5..15.1%; control bs=96 (already non-po2) +1.5..2.9% = the floor.
#   _EXP11 free again. It briefly restored the ALWAYS-MASKED arm of `_trmm_cmplx_small_L!`/`_R!` (the
#          9th Val of `_uker_cmplx!`, FULL, hardwired false on every branch) so the full-height-tile
#          dispatch could be A/B'd IN ONE PROCESS. It was needed because the pre/post comparison was
#          otherwise cross-run on both sides and the two disagreed: a probe read +5.2% at ztrmm n=32 on
#          Zen4 while the gate cell moved 0.994 (ee450bc) -> 0.972 (9951b1d) against 0.49% anchor drift.
#          RESOLVED — 6 ABBA rounds, arms bit-identical: Zen3 +16.7%/+13.1% (n=8/32) with the AVX2
#          packed-path rows an exact 1.000 control, Zen4 +0.4%/+3.6%/+3.8%/+2.7% (n=8/32/48/128). The
#          fix is positive at every affected size on both boxes; the gate's 0.994 -> 0.972 was a
#          cross-run artifact. Flag stripped — it sat in the hottest tiny-n tile loop.
#          (Was the `_trsm_zrt_R!` witness during the ztrsmR campaign; stripped when that landed.)
#   _EXP12 ztrsmR leaf column-block width NC=8 instead of `_ZRT_NC`(=4), on the reasoning that the
#          leaf's B re-reads fall as k²/(2·NC) and 2·NC+2 = 18 vectors fits AVX-512's 32 zmm.
#          FALSIFIED (Zen4, bit-identical output): 1.125 / 1.008 / 1.004 / 1.002 / 1.000 at
#          n=32/128/256/512/1024, i.e. slower or null everywhere. `kernel_report` explains it — NC=8 has
#          the HIGHER arithmetic intensity (8.57 vs 5.85) and fewer mem ops per column (2.62 vs 3.25),
#          so the kernel was never short of load slots and the wider block bought nothing but pressure.
#          `_ZRT_NC`'s derivation is correct; arm removed.
#   _EXP13 ztrsmR leaf tile-loop ORDER: r0-outer/jb-inner instead of jb-outer/r0-inner. Legal either way
#          (a tile at jb consumes only columns < jb OF ITS OWN ROW BLOCK, and jb still ascends), and the
#          working set differs by two cache levels — jb-outer re-streams the whole m×k B panel (128 KiB
#          at n=128) from L2 per column block, r0-outer holds W×k = 8 KiB in L1.
#          FALSIFIED (Zen4, bit-identical output): 1.051 / 1.015 / 1.026 / 1.016 / 1.007 / 1.004 at
#          n=32..2048 — jb-outer wins at EVERY size. The L1 residency is real but irrelevant; jb-outer's
#          inner r0 sweep walks each column contiguously and the hardware prefetcher covers the L2
#          traffic, while r0-outer's jb sweep touches one 128-byte chunk per column at ldb stride.
#          Arm removed. Do not re-propose loop interchange here on a working-set argument alone.
#   _EXP14 INVERTED: set true to restore the old SCALAR-negate sign placement in `_zrt_tile!`. The
#          vector-negate fold (Val{FOLD}=true) SHIPS ON — see the `_zrt_tile!` header for the mechanism.
#          Static: the i-loop body goes from 2.95 to 1.96 instructions per useful FMA (vmovsd and vxorpd
#          gone, loads folded into vbroadcastsd memory operands, vfnmadd231pd picked up).
#          Measured Zen4, whole-op, residuals identical to the last digit: FOLD is faster by 13.9 / 6.8 /
#          3.6 / 1.8 / 0.9 / 0.35% at n=32/128/256/512/1024/2048 — a monotone decay that tracks the
#          leaf's shrinking share of the cell exactly, which is the signature of a real leaf win rather
#          than a shift in the surrounding `_gemm_subR!`.
#   _EXP15 was the _EXP14 sign fold carried to `_zgt_slab!` (complex side-L). REVERTED, arm removed —
#          not because it lost, but because it compiled to a BYTE-IDENTICAL loop: LLVM already emits
#          vfnmadd there (that kernel's negated splat is used TWICE, `_zrt_tile!`'s was used once).
#          See the `_zgt_slab!` header. Do not re-run the sibling audit on this kernel.
#   _EXP16 INVERTED: set true to restore the UNFUSED `_ctrgemm_3m!` (three n×n P arrays + `_split3!`).
#          The FUSED driver ships. Kept A/B-able because Zen5 is unmeasured; fused uses the same kernels
#          with strictly less traffic, so it cannot lose (measured fused/unfused 0.83-1.00, both boxes).
#          Previously the `_trsm_cgt_L!` witness; stripped when that landed.
#   (_EXP13 and _EXP15 were reused for the 3M band campaign — _EXP13 bypassed the unpacked branch,
#    _EXP15 lowered the rank-k 3M edge — and are FREE AGAIN. Both were needed at once: with a single
#    combined flag Zen4 compared 3M-vs-UNPACKED while Zen3 compared 3M-vs-PACKED, so the two apparent
#    "crossovers" were different trades. Splitting them is what showed the fitted 768/W was encoding a
#    baseline difference rather than physics. Keep that separation if the band is ever re-opened.)

# ILP LEVER (_EXP7) — TWO STRIPES INTERLEAVED, one body per slab index.
# At n = NR + W (the gate cell: 24 + 8 = 32) the leaf currently solves stripe0 then stripe1 SEQUENTIALLY,
# so exactly ONE back-substitution chain is ever in flight: 4 slabs x MR rows of serial dependency,
# twice. The two stripes are INDEPENDENT — nothing in stripe1 reads stripe0 — so emitting slab si of
# both into ONE body lets the scheduler overlap the two chains.
# WHY THIS AND NOT MORE TRAFFIC WORK: measured, PB's kernel is 6.8% FASTER than AOCL with pages resident
# (1631 vs 1743 ns) but pays 114 ns MORE first-touch cost (777 vs 663) and runs at IPC 1.16 vs 2.20.
# PB EXPOSES latency AOCL OVERLAPS. Seven traffic levers moved nothing because none shortens or
# duplicates a dependency chain. Recovering the exposure gap puts PB ~110 ns ahead (~4.5%).
# LAYOUT: both stripes share one P of row stride NR+W, stripe0 at P columns [0,NR), stripe1 at
# [NR,NR+W) — expressed through the existing `vbase`, so no new addressing. jc is 0 for both; stripe1's
# B columns fall out of vbase = NRV.
# REGISTER BUDGET: MR*(NRV+1) = 32 accumulators, exactly the AVX-512 file. Affordable on the evidence
# that NRV=3 already wins WHILE spilling 23-38 reloads — chains beat pressure on this kernel.
# NOTE the U pointer parameter MUST be named `Up`: the emitted slab body references `Up` directly
# (it is inlined here, not passed as an argument), so a `pU` parameter leaves the body resolving `Up`
# to a nonexistent module global and it fails at RUNTIME, not at generation.
@generated function _fusedT_pair_tiny!(
        ::Val{K}, ::Val{NRVe}, Pp::Ptr{T}, pB::Ptr{T}, ldb::Int, jc::Int,
        Up::Ptr{T}, ldu::Int, rp::Ptr{T}
    ) where {K, NRVe, T}
    W = _vwidth(T); MR = _GT_MR
    (W == 8 && MR == 8 && K % MR == 0) ||
        return :(throw(AssertionError("tiny fusedT pair requires W==MR==8 and K%MR==0")))
    ldp = (NRVe + 1) * W                       # one shared P: stripe0 [0,NRVe*W), stripe1 [NRVe*W, ldp)
    body = quote end
    for si in (K ÷ MR - 1):-1:0                # bottom slab first: back-substitution runs upward
        s = si * MR; ng = K - s - MR
        # stripe0: NRVe wide at P/B column base 0.  stripe1: 1 wide at base NRVe.
        append!(body.args, _slab_body_ord(T, MR, NRVe, s, K, ldp, ng, 0, false, :a).args)
        append!(body.args, _slab_body_ord(T, MR, 1, s, K, ldp, ng, NRVe, false, :b).args)
    end
    push!(body.args, :(return nothing))
    return body
end

# TINY-K STRIPE: the general stripe's `si` loop unrolled, so each slab gets its literal gemm trip count.
# This is the "exhaustive specialisation" AOCL's tiny-k trsm bypass ships as 29-79 KB of hand-written
# kernels; expressed as one generator it is the same coverage with none of the maintenance.
@inline @generated function _fusedT_stripe_tiny!(
        ::Val{K}, ::Val{NRVe}, Pp::Ptr{T}, pB::Ptr{T}, ldb::Int, jc::Int,
        pU::Ptr{T}, ldu::Int, rp::Ptr{T}
    ) where {K, NRVe, T}
    MR = _GT_MR
    K % MR == 0 || return :(throw(AssertionError("tiny fusedT stripe requires K%MR==0")))
    W = _vwidth(T); sz = sizeof(T)
    body = Expr(:block, Expr(:meta, :inline))
    # Cold-operand prefetch of this stripe's B block + the U triangle, RUNTIME-gated on _EXPFLAG so one
    # build serves both arms of the A/B (branch-switching cost ~7 min per arm-run: a checkout
    # invalidates the pkgimage, so each arm paid a full precompile plus a 243 MB pkgimage load).
    # The flag is tested in the EMITTED code, not in the generator — a generator-time test would bake
    # the choice in at first specialisation and the toggle would silently do nothing.
    pf = Expr(:block)
    for v in 0:(NRVe - 1), l in 0:(W - 1)
        col = v * W + l
        for r in 0:(cld(K * sz, 64) - 1)
            push!(pf.args, :(_prefetch(pB + ((jc + $col) * ldb) * $sz + $(r * 64))))
        end
    end
    for c in 0:(K - 1)                              # U column c holds rows 0..c (upper triangle)
        for r in 0:(cld((c + 1) * sz, 64) - 1)
            push!(pf.args, :(_prefetch(pU + ($c * ldu) * $sz + $(r * 64))))
        end
    end
    push!(body.args, :(_EXPFLAG[_EXP2] && $pf))
    for si in (K ÷ MR - 1):-1:0                     # bottom slab first: back-substitution runs upward
        args = :(Val($(K - si * MR - MR)), Val(NRVe), Pp, pB, ldb, jc, pU, ldu, rp, $(si * MR))
        push!(body.args, Expr(:call, :_gemmtrsm_u_slab_ng!, args.args...))
    end
    push!(body.args, :(return nothing))
    return body
end

# Route a stripe to the straight-line sweep when KC is one of the multiples of MR at or below the tiny-k
# entry cap `_TRSM_DBASE`. Both bounds are existing consts — no new tuning constant. The chain is emitted
# from them at generation time, so a pinned `_TRSM_DBASE` re-derives its own coverage, and every `Val` is
# a generation-time literal (trim-safe: no runtime→Val).
@inline @generated function _fusedT_stripe_k!(
        ::Val{NRVe}, Pp::Ptr{T}, pB::Ptr{T}, ldb::Int, jc::Int, pU::Ptr{T},
        ldu::Int, rp::Ptr{T}, KC::Int, nfull::Int, rem::Int, MR::Int, sz::Int
    ) where {T, NRVe}
    fall = :(return _fusedT_stripe!(Val(NRVe), Pp, pB, ldb, jc, pU, ldu, rp, KC, nfull, rem, MR, sz))
    if _vwidth(T) != 8 || _GT_MR != 8
        return quote
            $(Expr(:meta, :inline))
            $fall
        end
    end
    chain = Expr(:block)
    for K in _GT_MR:_GT_MR:_trsm_dbase()
        push!(
            chain.args,
            :(KC == $K && return _fusedT_stripe_tiny!(Val($K), Val(NRVe), Pp, pB, ldb, jc, pU, ldu, rp))
        )
    end
    return quote
        $(Expr(:meta, :inline))
        if rem == 0 && !_EXPFLAG[_EXP3]      # rem>0 needs the ragged-tail mini-pack: general path
            $chain
        end
        $fall
    end
end
function _trsm_fused_L!(unit::Bool, A, B)
    T = eltype(B); KC = size(A, 1); n = size(B, 2); sz = sizeof(T)
    W = _vwidth(T); NRV = _GT_NRV; MR = _GT_MR; NR = NRV * W
    lda = stride(A, 2); ldb = stride(B, 2)
    nfull = KC ÷ MR; rem = KC - nfull * MR
    # Buffer = P (KC×NR row-major, L1) ‖ compact U (KC×ldu col-major) ‖ recip (KC). Packing U once per leaf
    # fixes the parent-matrix large-`lda` scatter + keeps the per-stripe U re-reads on a compact L2-resident
    # panel (else the off-diagonal gemm evicts it → DRAM). ldu is forced ODD (KC|1): the slab reads U down a
    # column at stride ldu·sz, and a po2 ldu (KC=128) collapses those onto few L1 sets — an odd ld can never
    # be a way-stride multiple, so the re-reads stay conflict-free (mirrors _trsm_rpack's odd-ld trick).
    ldu = KC | 1
    # P is sized for the WIDEST stripe layout any path here can use, not just NR. The interleaved-pair
    # paths share one P across two stripes: 2*NR columns for `_fusedT_stripe_pair!`, NR+W for the tiny
    # `_fusedT_pair_tiny!`. Sizing P at NR silently overflowed into the U-pack region — harmless only by
    # accident at k=32 (directA leaves that region unused and `rp` sits beyond the spill), which is
    # exactly the kind of luck that turns into a corruption bug the moment directA is false.
    #
    # The sizing above is a CORRECTNESS fix and nothing more. An earlier version of this comment claimed
    # the resulting OFFSET was performance-critical (Zen3 n=128 0.942->0.995, n=32 0.927->0.9165, "the
    # only change is the layout"). RETRACTED 2026-08-10 — both halves were wrong:
    #   * n=32 could not have been affected at all. Square B at n=32 on AVX2 fails the `_GT_TRANSPOSE`
    #     conjunct below and routes to `_trsm_dense_L!`, which never allocates this buffer. bench/plots.jl
    #     uses that same square shape, so the gate cell takes that path too. That delta was cross-run
    #     drift between two commits, i.e. exactly the cross-run comparison our own rules forbid.
    #   * n=128 does not survive a controlled test. `_EXPINT[1]` was swept over a FULL L1-way period
    #     (0..512 slots = 4096 B = one way of a 32K 8-way L1) on both boxes: flat 1.000, and at Zen3
    #     k=128 with a clean instrument (base SE 0.0012) flat to +-0.004. A P/U set-conflict mechanism
    #     would have to show structure over a full period. It shows none.
    # `_EXPINT[1]` is kept as an inert sweep knob (default 0) so the negative result stays reproducible.
    # Do NOT reopen "workspace alignment" as a trsm lever without a NEW mechanism and a live-knob witness.
    buf = _trsm_fused_buf(T, KC * 2 * NR + _EXPINT[1] + KC * ldu + KC)
    GC.@preserve A B buf begin
        pA = pointer(A); pB = pointer(B); Pp = pointer(buf)
        # U and the reciprocals start past the WIDEST P layout (2*NR), not past NR — the paired stripes
        # write P columns up to 2*NR and would otherwise land on top of the U pack.
        pU = Pp + (KC * 2 * NR + _EXPINT[1]) * sz; rp = pU + KC * ldu * sz
        # DIRECT-A (Fable lever #2): for an L1-resident triangle, skip the compact-U pack and read A's upper
        # triangle STRAIGHT — the slab reads U[row,col]=A[row,col] at stride `lduse`, so pass pA/lda. Saves the
        # O(k²/2) pack when A already fits L1 (the large-lda scatter + po2 odd-ldu the pack avoids are
        # negligible in L1). Fixes the small-k setup floor (s=24/32, the only genuinely setup-bound points).
        # _EXP5: force the compact-U pack even when A is L1-sized. directA skips the pack and lets the
        # slabs read U as SCALAR DEMAND LOADS from inside the gemm loop AND from inside the serial
        # back-substitution chain — where a cold line is a full memory latency with nothing to overlap it.
        # The pack below is a unit-stride streaming read: same lines, fetched as one frontloaded burst.
        # Counters say this is the only live degree of freedom: at the gate working set we already pull
        # 222 lines/call from L3+DRAM against a ~256-line compulsory floor, so it is not a bytes problem.
        directA = (KC * KC * sz <= _L1_BYTES) && !_EXPFLAG[_EXP5]
        pUsrc = pA; lduse = lda
        if !directA
            @inbounds for c in 0:(KC - 1)                     # pack A's upper triangle → compact U (odd ldu)
                for r in 0:c
                    unsafe_store!(pU, unsafe_load(pA, r + c * lda + 1), r + c * ldu + 1)
                end
            end
            pUsrc = pU; lduse = ldu
        end
        @inbounds for i in 0:(KC - 1)                          # recips always packed (contiguous rp panel)
            unsafe_store!(rp, unit ? one(T) : inv(unsafe_load(pA, i + i * lda + 1)), i + 1)
        end
        # AVX-512 f64: vectorized 8×8-transpose pack (const-folds). NOTE: the transpose reads 8 B-columns at
        # stride ldb·sz simultaneously for the in-register 8×8 block — when ldb·sz is a big power-of-2 multiple
        # (po2 n) those 8 accesses share low-12 bits → 4K-aliasing stall (measured +3.5%@512 / +1.7%@1024 from
        # padding B). FALSIFIED fix: routing the aliased stride to the alias-free column-outer scalar pack
        # regressed −0.9% (the scalar per-element cost outweighs the aliasing) — do NOT retry. The residual is
        # structural. THE OLD CLAIM HERE WAS WRONG and it cost real time before anyone checked it: it said
        # AOCL "reads B via row-lane direct-B broadcasts (1 contiguous stream, no transpose)" and that
        # escaping the aliasing needs a row-lane microkernel family. DISASSEMBLING THE SHIPPED LIBRARY
        # REFUTES THAT (2026-08-12, AOCL_jll libblis-mt.so, not stripped). The BLOCKED kernel AOCL has
        # for this case — WHICH IS NOT PROVEN TO BE THE ONE IT RUNS AT n=128/256; see the dispatch note
        # below — is `bli_dgemmtrsm_u_zen4_asm_8x24`: a FUSED gemmtrsm at an 8×24 register tile — the
        # same structure and the same tile shape as ours (_GT_MR=8, _GT_NR=NRV·W=24). Opcode mix: 375
        # vfmadd231pd · 141 vbroadcastsd · 78 vmovupd + 56 vmovapd · 48 vshuff64x2 · 45 vmulpd ·
        # 24 vscatterqpd + 24 kxnorw · 24 vfmsub231pd · 21 vsubpd. It SHUFFLES (48 cross-lane permutes),
        # so "no transpose" is false, and it has ZERO gathers — reads come contiguously out of packed
        # panels and only the write-back to arbitrarily-strided C is scattered under an all-ones mask
        # (24 scatters × 8 lanes = 192 = exactly one 8×24 tile).
        # Corroborated from our side: the no-transpose formulation that DOES exist in-tree (trsv per
        # column, the AD/generic branch) measures 3.8–7.9× SLOWER, because putting rows in lanes pays the
        # full back-substitution chain once per COLUMN instead of amortising it across NR columns.
        # CONSEQUENCE: there is no row-lane microkernel to chase. Do not re-open "row-lane" on the
        # strength of the old comment again.
        #
        # DISPATCH IS UNVERIFIED, AND THIS IS THE OPEN QUESTION. Identifying the blocked kernel by symbol
        # shape proves the kernel EXISTS, not that it RUNS at our binding sizes. The same library ships a
        # separate unpacked small-matrix family — `bli_dtrsm_small_AltXB_AuXB_AVX512` (AuXB = A-upper·X=B,
        # i.e. exactly our case), `bli_trsm_small`, `bli_trsm_small_AVX512` — and `dtrsm_blis_impl`
        # contains a dimension-dispatch maze (constants 49/50/58/96/120/138/199/1020/1811/2499/3219/
        # 4299/13999 plus log10 calls, an indirect call) that plausibly routes n=128/256 there. That
        # kernel is scalar-heavy and does NO packing (census: 532 vmovsd, 372 vmulsd, 174 vsubsd, 74
        # vdivsd, 108 vfmadd231pd, 0 scatters). If it is what wins those cells, then the lever is ENTRY
        # AND PACKING OVERHEAD — the recurring culprit in this repo — and not macro-kernel strategy, and
        # every rate/IPC/tile comparison made against the blocked kernel was against the wrong code.
        # SETTLE THIS BEFORE ACTING ON n=128/256: decode the branch region of `dtrsm_blis_impl` for
        # (side=L, uplo=U, trans=N, m=n=128), or read `bla_trsm_amd.c` at the artifact's version, where
        # those thresholds appear as named constants. No benchmark needed.
        useT = _GT_TRANSPOSE
        useT4 = (W == 4)                                             # AVX2: vectorized 4×4 transpose pack (lever 2)
        rowouter = useT ? true : _fused_pack_rowouter(ldb, NR, sz)   # AVX2/edges: scalar orientation predicate
        fusedT = _TRSM_FUSEDT_ON[] && useT               # Lever 1: skip the pack round-trip (full stripes)
        # Tiny-k stripe width. At KC ≤ _TRSM_DBASE the NRV=3 slab is the ONLY spilling shape (asm scan:
        # 23-38 reloads at NRV=3, 0-1 at NRV=2, 0 at NRV=1) AND an NR=24 stripe leaves n=32 as 24+8, so a
        # quarter of the columns run as a Val(1) tail whose back-substitution has no ILP to hide its
        # serial chain — the exact deficit the falsified per-v experiment above quantified at +10-20%.
        # NR=2W makes n=32 two clean Val(2) stripes: spill-free AND 2-wide ILP. U is KC²/2 ≤ 4 KB here so
        # the extra per-stripe U re-read stays L1-resident, which is why this is a tiny-k-only choice.
        NRl = (_EXPFLAG[_EXP1] && KC <= _trsm_dbase()) ? 2 * W : NR
        # _EXP7 — ILP lever. At exactly n = NR+W with a tiny KC (the gate cell: k=32, n=32 = 24+8) solve
        # BOTH stripes in one paired body so their independent back-substitution chains overlap, instead
        # of running them back to back with only one chain ever in flight. Buffer note: the pair uses a
        # shared P of row stride NR+W, which is <= the KC*NR+... allocation already made above.
        # Pair path is ON by default now (measured 0.8522 paired/sequential, n=240, 22.7 SE,
        # bit-for-bit identical output). _EXP7 is retained INVERTED as the A/B disable so the
        # sequential arm stays reachable in-process.
        if !_EXPFLAG[_EXP7] && fusedT && rem == 0 && n == NR + W && KC == 32
            _fusedT_pair_tiny!(Val(32), Val(NRV), Pp, pB, ldb, 0, pUsrc, lduse, rp)
        else
        jc = 0
        while jc < n
            wid = min(NRl, n - jc)                        # real columns this stripe (last may be < NRl)
            # PAD WIDTH, not NR. The fusedT branches below take wid in {NR, 2W, W}; everything else
# falls here and USED TO BE PADDED OUT TO THE FULL NR with zeros, so a 2-column tail was
# solved as 24 columns — 12x the work. The comment above says "gate n is a multiple of W,
# so the tail is always 8 or 16 wide": TRUE when the ladder was all powers of two, FALSE
# since the non-po2 sizes landed. n=50 stripes 24+24+2 and n=100 stripes 24x4+4, and those
# 2- and 4-wide tails are exactly the red cells (Zen4 0.79, Zen5 0.80).
# The pad is internal to P — the unpack writes back only `wid` columns — so narrowing it to
# the smallest W-multiple covering wid is safe, and the solve already takes the stripe width
# as a compile-time Val(NRV). NRVp in 1:NRV, so a 3-way branch covers it.
            NRp  = min(NR, cld(wid, W) * W)
            NRVp = NRp ÷ W
            # fusedT handles any W-MULTIPLE stripe width at its TRUE NRV — the full NR (Val NRV) AND the
            # ragged W / 2W tails that `n mod NR` produces — with NO padding (gate n is a multiple of W, so
            # the tail is always 8 or 16 wide; that padding to NR was the whole small-n gap). Concrete-Val
            # branches (trim-safe: no runtime→Val). Non-W-multiple wid falls to the pack path below.
            # PAIR TWO ADJACENT FULL STRIPES so their back-substitution chains overlap.
            # DOMAIN IS SMALL KC, and that is measured, not assumed: at KC <= _TRSM_DBASE each slab's
            # gemm runs only ~12 trips on average and cannot hide the serial chain, so a second chain
            # pays (n=32 gate 0.898 -> 1.105). At larger KC the gemm already runs up to KC-s-MR trips
            # and supplies that independent work itself, so pairing only adds register pressure
            # (2*MR*NRV = 48 live accumulators against 32) — measured 6.2% / 4.2% / 2.6% SLOWER at
            # k=128 / 256 / 512, the penalty shrinking exactly as the gemm's share of the slab grows.
            # Hence the guard is the existing tiny-k cap, not a size literal. `rem > 0` (ragged bottom
            # rows) keeps the single-stripe path: the tail needs its own mini-pack.
            #
            # THE REGISTER EXPLANATION ABOVE IS REAL BUT NOT BINDING — an earlier revision of this note
            # called it simply "wrong", which overstated it. The spill is there: an asm audit counts 78
            # spill/reload vector moves in the paired slab's loop against ZERO in the shipped 24-accumulator
            # one (same detector, so the zero is trustworthy). What is wrong is treating the spill as the
            # CAUSE, because removing it does not help — see the NRV=2 arm below.
            # Re-measured 2026-08-11 on today's code: lifting the cap costs
            # −6.6/−5.7/−2.5% at k=128/256/512 at the shipped NRV=3 (reproducing the recorded figures),
            # but −10.8/−7.4/−4.1% under a PINNED NRV=2 — where a pair holds exactly 32 accumulators and
            # CANNOT spill. Pairing loses MORE where there is no spill, so spilling is not the cause.
            # What does scale the right way is L1 capacity for P: the stripe panel is KC·NR·8 bytes =
            # 24 KiB at KC=128/NRV=3 against a 32 KiB L1, and pairing doubles it to 48 KiB; at NRV=2 it
            # is 16 KiB → 32 KiB paired, i.e. exactly L1 with nothing left for U or B. Pairing at large
            # KC is a CAPACITY failure, not a register failure. (NRV=2 is also worse unpaired — 17.86 vs
            # 18.44 GF at n=128 — so the shipped NRV=3 stands.) Do not retry pairing at KC=128 by
            # shrinking NRV; it needs a smaller P footprint, which means a smaller KC for the paired path.
            if !_EXPFLAG[_EXP8] && fusedT && rem == 0 && NRl == NR &&
                    KC <= _trsm_dbase() && jc + 2 * NR <= n
                _fusedT_stripe_pair!(Val(NRV), Pp, pB, ldb, jc, pUsrc, lduse, rp, KC, nfull, MR, sz)
                jc += 2 * NR; continue
            end
            if fusedT && wid == NRl && NRl == NR
                _fusedT_stripe_k!(Val(NRV), Pp, pB, ldb, jc, pUsrc, lduse, rp, KC, nfull, rem, MR, sz)
                jc += NR; continue
            elseif fusedT && wid == 2 * W
                _fusedT_stripe_k!(Val(2), Pp, pB, ldb, jc, pUsrc, lduse, rp, KC, nfull, rem, MR, sz)
                jc += 2 * W; continue
            elseif fusedT && wid == W
                _fusedT_stripe_k!(Val(1), Pp, pB, ldb, jc, pUsrc, lduse, rp, KC, nfull, rem, MR, sz)
                jc += W; continue
            end
            if useT
                _fused_packP_tr!(Pp, pB, ldb, jc, wid, KC, NRp, sz)
            elseif useT4                                  # AVX2 vectorized 4×4 transpose pack (lever 2)
                _fused_packP_tr4!(Pp, pB, ldb, jc, wid, KC, NRp, sz)
            elseif rowouter                               # contiguous P writes; B read strided (non-aliasing)
                @inbounds for i in 0:(KC - 1)
                    srow = pB + (i + jc * ldb) * sz; drow = Pp + i * NRp * sz
                    for v in 0:(wid - 1)
                        unsafe_store!(drow, unsafe_load(srow + v * ldb * sz), v + 1)
                    end
                    for v in wid:(NRp - 1)
                        unsafe_store!(drow, zero(T), v + 1)
                    end
                end
            else                                          # contiguous B reads; strided P writes (po2-immune)
                @inbounds for v in 0:(wid - 1)
                    scol = pB + (jc + v) * ldb * sz; dcol = Pp + v * sz
                    for i in 0:(KC - 1)
                        unsafe_store!(dcol + i * NRp * sz, unsafe_load(scol + i * sz))
                    end
                end
                @inbounds for v in wid:(NRp - 1), i in 0:(KC - 1)
                    unsafe_store!(Pp + (i * NRp + v) * sz, zero(T))
                end
            end
            # Solve at the NARROWED stripe width. Val must be a compile-time constant, so branch on
            # NRVp (1:NRV) rather than splicing a runtime value — a runtime->Val is exactly the
            # `invoke ::Any` shape juliac --trim rejects.
            if NRVp == 1
                rem > 0 && _gemmtrsm_u_tail!(Pp, NRp, pUsrc, lduse, rp, nfull * MR, KC, Val(1))
                for si in (nfull - 1):-1:0
                    _gemmtrsm_u_slab!(Pp, NRp, pUsrc, lduse, rp, si * MR, KC, Val(MR), Val(1))
                end
            elseif NRVp == 2
                rem > 0 && _gemmtrsm_u_tail!(Pp, NRp, pUsrc, lduse, rp, nfull * MR, KC, Val(2))
                for si in (nfull - 1):-1:0
                    _gemmtrsm_u_slab!(Pp, NRp, pUsrc, lduse, rp, si * MR, KC, Val(MR), Val(2))
                end
            else
                rem > 0 && _gemmtrsm_u_tail!(Pp, NRp, pUsrc, lduse, rp, nfull * MR, KC, Val(NRV))
                for si in (nfull - 1):-1:0
                    _gemmtrsm_u_slab!(Pp, NRp, pUsrc, lduse, rp, si * MR, KC, Val(MR), Val(NRV))
                end
            end
            if useT
                _fused_unpackP_tr!(Pp, pB, ldb, jc, wid, KC, NRp, sz)
            elseif useT4
                _fused_unpackP_tr4!(Pp, pB, ldb, jc, wid, KC, NRp, sz)
            elseif rowouter
                @inbounds for i in 0:(KC - 1)                 # unpack P → B row-outer (contiguous P reads)
                    srow = Pp + i * NRp * sz; drow = pB + (i + jc * ldb) * sz
                    for v in 0:(wid - 1)
                        unsafe_store!(drow + v * ldb * sz, unsafe_load(srow, v + 1))
                    end
                end
            else
                @inbounds for v in 0:(wid - 1)               # unpack column-outer (contiguous B writes)
                    scol = Pp + v * sz; dcol = pB + (jc + v) * ldb * sz
                    for i in 0:(KC - 1)
                        unsafe_store!(dcol + i * sz, unsafe_load(scol + i * NRp * sz))
                    end
                end
            end
            # `wid`, not NR: this path packs/solves/unpacks exactly `wid` real columns. Identical to the
            # old `jc += NR` whenever NRl == NR (wid < NR only on the final partial stripe, where
            # wid == n-jc ends the loop either way), but required once NRl < NR — there `wid` can be a
            # full NRl stripe with columns still to come, and advancing by NR would SKIP NR-NRl of them.
            jc += wid
        end
        end                                   # else-branch of the _EXP7 paired-stripe dispatch
    end
    return B
end

# ===== Whole-k packed fused gemmtrsm (side-L, upper, no-trans) — the shared-panel design =====
# The single-leaf `_trsm_fused_L!` is fed KC≤_TRSM_FUSED_BASE (≈64) blocks by the recursion, then the
# off-diagonal coupling is a SEPARATE peak-dgemm `_gemm_sub!` that writes B UNPACKED — so every leaf
# boundary re-reads/re-packs B (measured 14–20% of the leaf) AND the small-KC leaves pay a large 1/k
# tax (back-sub latency ≈ MR/k, transpose-pack ≈ 1/k → ~40% overhead at KC=64). BLIS/AOCL avoid BOTH:
# one gemmtrsm sweep per stripe whose gemm portion accumulates against the WHOLE packed solved-X tail
# (no separate off-diagonal gemm, no re-pack), at whole-k so the 1/k tax vanishes. This is that sweep.
#
# U is packed ONCE per call into MR-row micropanels (unit-stride gemm reads, mirroring BLIS packed-A).
# MEASURED CAVEAT: packing did NOT lift the slab gemm — the whole-k sweep runs ~40.6 GF packed OR
# unpacked (the ~35 GF figure was the SMALL-KC recursion leaf, back-sub-latency-bound, not U-stride-
# bound). Kept as the structurally-correct BLIS shared-panel layout (substrate for the next lever); the
# residual to AOCL's 42 GF is the microkernel rate, not the re-pack (see the _TRSM_FULLPACK_ON note).
# Micro-panel si (rows [s,s+MR), s=si·MR) is MR×(k−s) column-major: MP_si[c·MR+r] = U[s+r, s+c], c∈[0,k−s).
# gemm reads U[s+r,kk] = MP_si[(kk−s)·MR+r] (kk≥s+MR); back-sub reads U[s+j,s+i] = MP_si[i·MR+j] (j<i).
# Below-diagonal diag-block entries (r>c) are stored but never read (upper-tri). Offset closed-form
# (no per-slab table): _mp_off(si) = MR·si·k − MR²·si·(si−1)/2. AVX-512-f64 only (needs the 8×8 pack).
@inline _mp_off(si::Int, k::Int, MR::Int) = MR * si * k - (MR * MR * si * (si - 1)) ÷ 2

# One MR-row slab × one NR-column stripe, reading U from its packed micropanel MPp (unit stride). Same
# structure as `_gemmtrsm_u_slab!` but the gemm kk-loop spans the FULL tail [s+MR,KC) (the whole solved-X
# panel, = the recursion's off-diagonal fused in) and U reads are unit-stride within the micropanel.
@inline @generated function _gemmtrsm_u_slab_packed!(
        Pp::Ptr{T}, ldp::Int, MPp::Ptr{T}, rp::Ptr{T},
        s::Int, KC::Int, ::Val{MR}, ::Val{NRV}
    ) where {T, MR, NRV}
    W = _vwidth(T); sz = sizeof(T); V = Vec{W, T}
    body = quote end
    for r in 0:(MR - 1), v in 0:(NRV - 1)
        push!(body.args, :($(Symbol(:c, r, :_, v)) = zero($V)))
    end
    inner = quote end                                     # gemm: acc[r][v] += U[s+r,kk]·P[kk][v]
    push!(inner.args, :(ub = MPp + (kk - s) * $(MR * sz)))  # &MP_si[(kk−s)·MR] = U[s:,kk] column (MR contig)
    for v in 0:(NRV - 1)
        push!(inner.args, :($(Symbol(:x, v)) = vload($V, Pp + (kk * ldp + $(v * W)) * $sz)))
    end
    for r in 0:(MR - 1)
        push!(inner.args, :(u = $V(unsafe_load(ub + $(r * sz)))))
        for v in 0:(NRV - 1)
            cs = Symbol(:c, r, :_, v)
            push!(inner.args, :($cs = muladd(u, $(Symbol(:x, v)), $cs)))
        end
    end
    push!(
        body.args, :(
            for kk in UnitRange(s + $MR, KC - 1)
                $inner
            end
        )
    )
    for r in 0:(MR - 1), v in 0:(NRV - 1)                          # subtract: acc = B[s+r] − acc
        cs = Symbol(:c, r, :_, v)
        push!(body.args, :($cs = vload($V, Pp + ((s + $r) * ldp + $(v * W)) * $sz) - $cs))
    end
    for i in (MR - 1):-1:0                                     # back-substitution, critical-path-first
        push!(body.args, :(d = $V(unsafe_load(rp + (s + $i) * $sz))))
        for v in 0:(NRV - 1)
            push!(body.args, :($(Symbol(:c, i, :_, v)) = $(Symbol(:c, i, :_, v)) * d))
        end
        for j in (i - 1):-1:0
            push!(body.args, :(u = $V(unsafe_load(MPp + $((i * MR + j) * sz)))))
            for v in 0:(NRV - 1)
                cj = Symbol(:c, j, :_, v); ci = Symbol(:c, i, :_, v)
                push!(body.args, :($cj = muladd(-u, $ci, $cj)))
            end
        end
    end
    for r in 0:(MR - 1), v in 0:(NRV - 1)
        push!(body.args, :(vstore($(Symbol(:c, r, :_, v)), Pp + ((s + $r) * ldp + $(v * W)) * $sz)))
    end
    push!(body.args, :(return nothing))
    return body
end

# ── Fused first/last-touch-transpose slab (Lever 1: skip the P transpose round-trip) ──────────────
# Same math as `_gemmtrsm_u_slab_packed!` but reads its OWN MR rows STRAIGHT from B (in-register 8×8
# transpose) for the subtract, and writes the solved rows BOTH to P (row-major, so upper slabs' gemm
# reuses them) AND transposed back to B (col-major, final). Eliminates the standalone _fused_packP_tr!
# and _fused_unpackP_tr! passes (the pack round-trip = 9.4% of the stripe at k=128, decomposed). The
# gemm phase is UNCHANGED (contiguous vloads from row-major P, written once by the solving slab; no
# re-transpose on reuse — the P memoization Fable's design keeps). REQUIRES W==MR==8 (AVX-512 f64) and
# a FULL stripe (all NRV column-blocks real, no zero-pad) — the driver routes the ragged last stripe to
# the old pack path. jc = this stripe's first B-column; s = slab top row. Design: Fable consult 2026-07.
@inline @generated function _gemmtrsm_u_slab_fusedT!(
        Pp::Ptr{T}, ldp::Int, pB::Ptr{T}, ldb::Int, jc::Int,
        MPp::Ptr{T}, rp::Ptr{T}, s::Int, KC::Int, ::Val{MR}, ::Val{NRV}
    ) where {T, MR, NRV}
    W = _vwidth(T); sz = sizeof(T); V = Vec{W, T}
    # AVX-512-f64 (W==MR==8) only. Generate a throw-body (never an assert) for other widths so GENERATION
    # always succeeds: the driver's fusedT branch is runtime-dead off AVX-512 (gated by _GT_TRANSPOSE), but
    # StrictMode's full-inference dogfood still expands this @generated call on AVX2 — a generation-time
    # @assert there crashes CI (the W=4 runner). The throw-expr compiles cleanly and is never executed.
    (W == 8 && MR == 8) || return :(throw(AssertionError("fusedT slab requires W==MR==8 (AVX-512 f64)")))
    body = quote end
    for r in 0:(MR - 1), v in 0:(NRV - 1)
        push!(body.args, :($(Symbol(:c, r, :_, v)) = zero($V)))
    end
    inner = quote end                                     # gemm: acc[r][v] += U[s+r,kk]·P[kk][v]
    push!(inner.args, :(ub = MPp + (kk - s) * $(MR * sz)))
    for v in 0:(NRV - 1)
        push!(inner.args, :($(Symbol(:x, v)) = vload($V, Pp + (kk * ldp + $(v * W)) * $sz)))
    end
    for r in 0:(MR - 1)
        push!(inner.args, :(u = $V(unsafe_load(ub + $(r * sz)))))
        for v in 0:(NRV - 1)
            cs = Symbol(:c, r, :_, v)
            push!(inner.args, :($cs = muladd(u, $(Symbol(:x, v)), $cs)))
        end
    end
    push!(
        body.args, :(
            for kk in UnitRange(s + $MR, KC - 1)
                $inner
            end
        )
    )
    # subtract: read own MR rows of B directly (block v = 8 cols [jc+v·8, jc+v·8+8), 8-row vloads down
    # cols = contiguous), transpose in-register → row-vectors, acc = B[s+r] − acc. bc_l/br_r reused per v.
    for v in 0:(NRV - 1)
        for l in 0:(W - 1)
            push!(body.args, :($(Symbol(:bc, l)) = vload($V, pB + ((s) + (jc + $(v * W + l)) * ldb) * $sz)))
        end
        brs = Expr(:tuple, (Symbol(:br, l) for l in 0:(W - 1))...)
        bcs = (Symbol(:bc, l) for l in 0:(W - 1))
        push!(body.args, Expr(:(=), brs, :(_tr8x8($(bcs...)))))
        for r in 0:(MR - 1)
            cs = Symbol(:c, r, :_, v)
            push!(body.args, :($cs = $(Symbol(:br, r)) - $cs))
        end
    end
    for i in (MR - 1):-1:0                                     # back-substitution, critical-path-first
        push!(body.args, :(d = $V(unsafe_load(rp + (s + $i) * $sz))))
        for v in 0:(NRV - 1)
            push!(body.args, :($(Symbol(:c, i, :_, v)) = $(Symbol(:c, i, :_, v)) * d))
        end
        for j in (i - 1):-1:0
            push!(body.args, :(u = $V(unsafe_load(MPp + $((i * MR + j) * sz)))))
            for v in 0:(NRV - 1)
                cj = Symbol(:c, j, :_, v); ci = Symbol(:c, i, :_, v)
                push!(body.args, :($cj = muladd(-u, $ci, $cj)))
            end
        end
    end
    # store: solved rows → P row-major (for upper slabs' gemm reuse) AND transposed → B col-major (final).
    for v in 0:(NRV - 1)
        for r in 0:(MR - 1)
            push!(body.args, :(vstore($(Symbol(:c, r, :_, v)), Pp + ((s + $r) * ldp + $(v * W)) * $sz)))
        end
        ccs = Expr(:tuple, (Symbol(:cc, l) for l in 0:(W - 1))...)
        cvs = (Symbol(:c, r, :_, v) for r in 0:(MR - 1))
        push!(body.args, Expr(:(=), ccs, :(_tr8x8($(cvs...)))))
        for l in 0:(W - 1)
            push!(body.args, :(vstore($(Symbol(:cc, l)), pB + ((s) + (jc + $(v * W + l)) * ldb) * $sz)))
        end
    end
    push!(body.args, :(return nothing))
    return body
end

# Ragged bottom tail (rem<MR rows, no rows below → pure back-sub), reading the rem×rem packed micropanel
# MPt (col-major, ld=rem): U[base+jr,base+ir] = MPt[ir·rem+jr]. Rare (k not a multiple of MR).
@inline function _gemmtrsm_u_tail_packed!(
        Pp::Ptr{T}, ldp::Int, MPt::Ptr{T}, rp::Ptr{T},
        base::Int, KC::Int, rem::Int, ::Val{NRV}
    ) where {T, NRV}
    W = _vwidth(T); sz = sizeof(T); V = Vec{W, T}
    @inbounds for i in (KC - 1):-1:base
        ir = i - base; d = V(unsafe_load(rp + i * sz))
        for v in 0:(NRV - 1)
            q = Pp + (i * ldp + v * W) * sz
            ci = vload(V, q) * d; vstore(ci, q)
            for j in (i - 1):-1:base
                u = V(unsafe_load(MPt + (ir * rem + (j - base)) * sz))
                qj = Pp + (j * ldp + v * W) * sz
                vstore(muladd(-u, ci, vload(V, qj)), qj)
            end
        end
    end
    return nothing
end

# Pack A's upper triangle → MR-row micropanels (MP) + reciprocal diagonal (rp). One pass per trsm call.
function _pack_U_micro!(
        MP::Ptr{T}, rp::Ptr{T}, pA::Ptr{T}, lda::Int, k::Int, unit::Bool,
        MR::Int, nfull::Int, rem::Int
    ) where {T}
    sz = sizeof(T)
    @inbounds for si in 0:(nfull - 1)
        s = si * MR; moff = _mp_off(si, k, MR)
        for c in 0:(k - s - 1)
            dst = MP + (moff + c * MR) * sz
            src = pA + (s + (s + c) * lda) * sz            # &A[s, s+c]
            for r in 0:(MR - 1)
                unsafe_store!(dst + r * sz, unsafe_load(src + r * sz))
            end
        end
    end
    if rem > 0
        s = nfull * MR; moff = _mp_off(nfull, k, MR)
        @inbounds for c in 0:(rem - 1), r in 0:(rem - 1)
            unsafe_store!(MP + (moff + c * rem + r) * sz, unsafe_load(pA + ((s + r) + (s + c) * lda) * sz))
        end
    end
    @inbounds for i in 0:(k - 1)
        unsafe_store!(rp + i * sz, unit ? one(T) : inv(unsafe_load(pA + (i + i * lda) * sz)))
    end
    return nothing
end

# Whole-k driver: solve U·X=B in place, U upper k×k, B k×n wide — ONE packed sweep, NO recursion. P is
# k×NR row-major (per NR-column stripe, reused; L2-resident at gate k). U packed once (½L3-bounded, see
# _TRSM_FULLPACK_MAX). Column stripes OUTER / slabs INNER — P stays hot across a stripe's whole solve.
function _trsm_fused_full_L!(unit::Bool, A, B)
    T = eltype(B); k = size(A, 1); n = size(B, 2); sz = sizeof(T)
    W = _vwidth(T); NRV = _GT_NRV; MR = _GT_MR; NR = NRV * W
    lda = stride(A, 2); ldb = stride(B, 2)
    nfull = k ÷ MR; rem = k - nfull * MR
    packUlen = _mp_off(nfull, k, MR) + (rem > 0 ? rem * rem : 0)
    buf = _trsm_fused_buf(T, k * NR + packUlen + k)
    GC.@preserve A B buf begin
        pA = pointer(A); pB = pointer(B); Pp = pointer(buf)
        MP = Pp + k * NR * sz; rp = MP + packUlen * sz
        _pack_U_micro!(MP, rp, pA, lda, k, unit, MR, nfull, rem)
        MPt = MP + _mp_off(nfull, k, MR) * sz
        jc = 0
        while jc < n
            wid = min(NR, n - jc)
            if _TRSM_FUSEDT_ON[] && wid == NR
                # Fused first/last-touch transpose (Lever 1): no standalone pack/unpack. The ragged tail
                # rows (rem<MR) still need to live in P for the full slabs' gemm reuse, so mini-pack them
                # (rem×NR, tiny), solve, and unpack at stripe end; the full slabs read/write B directly.
                if rem > 0
                    b0 = nfull * MR
                    @inbounds for v in 0:(NR - 1), i in 0:(rem - 1)                # pack tail rows B→P (contiguous cols)
                        unsafe_store!(Pp + ((b0 + i) * NR + v) * sz, unsafe_load(pB + ((b0 + i) + (jc + v) * ldb) * sz))
                    end
                    _gemmtrsm_u_tail_packed!(Pp, NR, MPt, rp, b0, k, rem, Val(NRV))
                    @inbounds for v in 0:(NR - 1), i in 0:(rem - 1)                # unpack tail rows P→B
                        unsafe_store!(pB + ((b0 + i) + (jc + v) * ldb) * sz, unsafe_load(Pp + ((b0 + i) * NR + v) * sz))
                    end
                end
                for si in (nfull - 1):-1:0
                    _gemmtrsm_u_slab_fusedT!(Pp, NR, pB, ldb, jc, MP + _mp_off(si, k, MR) * sz, rp, si * MR, k, Val(MR), Val(NRV))
                end
            else
                _fused_packP_tr!(Pp, pB, ldb, jc, wid, k, NR, sz)
                rem > 0 && _gemmtrsm_u_tail_packed!(Pp, NR, MPt, rp, nfull * MR, k, rem, Val(NRV))
                for si in (nfull - 1):-1:0
                    _gemmtrsm_u_slab_packed!(Pp, NR, MP + _mp_off(si, k, MR) * sz, rp, si * MR, k, Val(MR), Val(NRV))
                end
                _fused_unpackP_tr!(Pp, pB, ldb, jc, wid, k, NR, sz)
            end
            jc += NR
        end
    end
    return B
end
# Same-process A/B switch for the whole-k packed sweep vs the recursion+single-leaf path.
# DEFAULT OFF: the shared-panel restructure eliminates the re-pack round-trip and is correct, but measured
# only reaches AOCL PARITY (~0.97 large-n, ~0.87 small-n) — NOT the ≥1.0 mandate. The re-pack was NOT the
# binding lever: the fused-slab microkernel intrinsically runs ~40.6 GF vs PB's own dgemm 43 / AOCL trsm 42,
# worst at small n (PB 31 vs AOCL 37 @128). It also shows a noise-level regression at po2 n=512, which the
# no-regression contract forbids. Kept behind this toggle as the validated re-pack-free substrate.
# FALSIFIED levers (measured Zen4, boost-locked, in-process A/B — do NOT re-try; bench/
# trsm_fullpack_gf.jl reproduces): (1) INVERTED-DIAGONAL epilogue (precompute inv of the MR×MR diagonal
# block at pack time, replace the serial back-sub with X=Vss·acc) — REGRESSED small-k (k=128 30.2 vs
# back-sub 31.5 GF), converged at large-k. The back-sub is NOT latency-bound: its MR×NRV=24 v-lanes across
# the MR rows already saturate ILP, so removing the serial chain buys nothing while the diag inversion adds
# pack-time cost. (2) C-TILE PREFETCH of the MR own-rows [s,s+MR) of P at slab entry — MEASURED NO-OP at
# every size (P is L2-resident straight from _fused_packP_tr!, so the "cold subtract-read" it targeted was
# never cold). The residual is the slab MICROKERNEL RATE (large-k plateau 40.6 vs dgemm 43 / AOCL 42) plus
# the O(k·n) transpose-pack + O(k²) U-pack OVERHEAD that dominates small-k (k=128 31 GF, amortizes to 40.6
# by k≥384) — NOT the back-sub, NOT prefetch. Next lever = the slab gemm tile geometry (8×3=24 acc, near
# the 32-zmm limit) and/or skipping the P round-trip at small-k, not the epilogue. Flip to true for the A/B.
const _TRSM_FULLPACK_ON = Ref(false)
# Whole-k cutoff: packed-U micropanels hold the upper triangle ≈ k²/2 elements; bound to ½·L3 residency
# (req#8): (k²/2)·sizeof(T) ≤ _L3_BYTES/2 ⟹ k ≤ isqrt(_L3_BYTES ÷ sizeof(T)). Above → recursion bottoms here.
const _TRSM_FULLPACK_MAX = isqrt(_L3_BYTES ÷ sizeof(Float64))
# Lower crossover: the whole-k sweep's flat overheads (transpose-pack O(k·NR), back-sub O(MR)/slab) amortize
# only for large k; below it the ½-split recursion — which runs its off-diagonal at TRUE dgemm peak (43 GF)
# vs the fused slab's ~40.6 — wins. Crossover ≈ where the k×NR P-stripe exceeds ~2·L1 (whole-k must stream
# P from L2, erasing the recursion's small-leaf L1 locality edge, so its lower overhead takes over): k ≥
# 2·L1/(NR·sizeof). Measured Zen4 crossover is 256<k≤384; the formula gives ≈342. EMPIRICAL crossover —
# req#8 debt (derive+fleet-validate), Preferences-overridable. Measured net: 512 0.889→0.90, 1024 0.94→0.97.
const _TRSM_FULLPACK_MIN = @load_preference(
    "trsm_fullpack_min",
    cld(2 * _L1_BYTES, _GT_NR * sizeof(Float64))
)::Int

# side 'L': B := op(A)⁻¹·B, A k×k (k=size(B,1)), unscaled.
function _trsm_left!(up::Bool, tr::Bool, cj::Bool, unit::Bool, A, B)
    k = size(A, 1)
    if eltype(B) <: BlasReal && !cj
        # narrow B → dense base (few axpy calls); wide B → invL base (gemm-efficient). n is invariant under
        # the side-L row split, so the choice is consistent through the recursion.
        if size(B, 2) <= _trsm_ncut()
            # Narrow B. Prefer the fused gemmtrsm leaf for eligible mid-k (_TRSM_FUSED_MIN ≤ k ≤ _TRSM_FUSED_BASE):
            # it beats BOTH the scalar dense base (k≤32) AND the ½-split recursion that k>32 would otherwise fall
            # to — the `n≤_TRSM_NCUT` guard was intercepting square k=48/64 into that recursion, whose leaves are
            # the scalar dense base (~15 GF) vs the fused leaf's ~26 GF. Measured Zen4 vs AOCL: k=48 0.53→0.94,
            # k=64 0.48→0.82, k=32 0.51→0.67. Tiny k / non-fusable / trans keep the dense base.
            # `_GT_TRANSPOSE` REMOVED 2026-08-10 — same reason as the tiny-k bypass in `trsm!` (see the long
            # note there): it is the fusedT slabs' capability bit, not a crossover, and it stranded AVX2 on
            # the dense base at k=48/64 where the leaf measures 20.7% / 25.9% faster on Zen3.
            if up && !tr && _TRSM_FUSED_ON[] &&
                    _TRSM_FUSED_MIN <= k <= _TRSM_FUSED_BASE && _trsm_fusable(A, B)
                return _trsm_fused_L!(unit, A, B)
            end
            k <= _trsm_dbase() && return _trsm_dense_L!(up, tr, unit, A, B)
        elseif up && !tr && _GT_TRANSPOSE && _TRSM_FULLPACK_ON[] &&
                _TRSM_FULLPACK_MIN <= k <= _TRSM_FULLPACK_MAX && _trsm_fusable(A, B)
            # Whole-k packed sweep (shared solved-X panel, no recursion / re-pack). AVX-512-f64, large-k only.
            return _trsm_fused_full_L!(unit, A, B)
        elseif up && !tr && _TRSM_FUSED_ON[] && k <= _TRSM_FUSED_BASE && _trsm_fusable(A, B)
            # Fused gemmtrsm leaf (1× flop). Fires at EVERY k ≤ base diagonal block, INCLUDING the ones the
            # n≥256 recursion descends into (B width unbounded — the leaf stripes over B's columns). The
            # recursion's off-diagonal `_gemm_sub!` stays peak dgemm; only the diagonal block is fused (1× flop)
            # vs invL's 2×-flop k=32 leaves. The old `size(B,2) ≤ base` restriction (AVX2 only) sent wide-B
            # n≥256 to those k=32 invL leaves instead — the whole AVX2 n≥256 gap (Fable-diagnosed 2026-07-15;
            # AVX-512 was already unrestricted via _GT_TRANSPOSE, so this only changes AVX2).
            return _trsm_fused_L!(unit, A, B)
        elseif k <= _trsm_base()
            return _trsm_base_invL!(up, tr, unit, A, B)
        end
    elseif eltype(B) <: BlasComplex                       # complex base (else fall through → gemm-blocked split)
        # nrhs is invariant under the row-split → decide the base once. Wide B: trtri-on-inverse base (its
        # O(k³) invert is amortized by the big off-diagonal gemm). Narrow B (standalone 96/128): trtri
        # overhead is exposed, so recurse into small j-outer bases + gemm subtract (OB's structure).
        # The gemmtrsm leaf beats BOTH bases (and the recursion into them) for every k it covers, so it
        # is taken before the narrow/wide split rather than from inside it — at k=128 one leaf measured
        # 40.4 GF against 35.7 for the two-64-leaves-plus-gemm recursion this recbase would pick.
        if up && !tr && k <= _ZGT_BASE && _cgt_ok(A, B)
            return _trsm_cgt_L!(unit, k, A, B)
        end
        recbase = size(B, 2) <= _fh_ctrsm_ncut() ? _fh_ctrsm_rec_l() : _TRMM_BASE
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
            for i in 1:(j - 1)
                _axpy_col!(B, j, -coef(i, j), i, m)
            end
            unit || _scal_col!(B, j, inv(coef(j, j)), m)
        end
    else
        for j in k:-1:1
            for i in (j + 1):k
                _axpy_col!(B, j, -coef(i, j), i, m)
            end
            unit || _scal_col!(B, j, inv(coef(j, j)), m)
        end
    end
    return B
end

# Register-tiled trsm-R base (f64, GENERAL up/tr/unit): solve X·op(A)=B, X/B m×k, vectorized over m (B rows,
# contiguous). Mirror of the side-L tile with rows↔solve-columns swapped: block NC=4 SOLVE-COLUMNS, downdate
# the block against the ALREADY-SOLVED columns in one W-wide sweep (reuse each solved column's m-vector across
# the 4 block-cols), then a scalar in-block NC×NC coupling solve. coef(j,l)=A[j,l] (tr) or A[l,j]. Each B
# element written ~once vs the dense base's ~k passes. Measured Zen3 (trsmR gate, lower-T): +64–113%. Bit-id.
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
                a0 = muladd(_CVF(-cf(j0, l)), xv, a0); a1 = muladd(_CVF(-cf(j0 + 1, l)), xv, a1)
                a2 = muladd(_CVF(-cf(j0 + 2, l)), xv, a2); a3 = muladd(_CVF(-cf(j0 + 3, l)), xv, a3)
            end
            vstore(a0, _cvptr(pB, i, j0, ldb));     vstore(a1, _cvptr(pB, i, j0 + 1, ldb))
            vstore(a2, _cvptr(pB, i, j0 + 2, ldb)); vstore(a3, _cvptr(pB, i, j0 + 3, ldb)); i += W
        end
        @inbounds while i <= m                               # m tail
            for t in 0:(NC - 1)
                s = unsafe_load(pB, _clidx(i, j0 + t, ldb))
                for l in solved
                    s = muladd(-cf(j0 + t, l), unsafe_load(pB, _clidx(i, l, ldb)), s)
                end
                unsafe_store!(pB, s, _clidx(i, j0 + t, ldb))
            end
            i += 1
        end
        return @inbounds for t in (asc ? (0:(NC - 1)) : ((NC - 1):-1:0))    # in-block 4×4 coupling solve, vectorized over m
            jj = j0 + t; d = unit ? 1.0 : inv(cf(jj, jj)); rng = asc ? (0:(t - 1)) : ((t + 1):(NC - 1)); i = 1
            while i + W - 1 <= m
                x = vload(_CVF, _cvptr(pB, i, jj, ldb))
                for u in rng
                    x = muladd(_CVF(-cf(jj, j0 + u)), vload(_CVF, _cvptr(pB, i, j0 + u, ldb)), x)
                end
                unit || (x = x * _CVF(d)); vstore(x, _cvptr(pB, i, jj, ldb)); i += W
            end
            while i <= m
                s = unsafe_load(pB, _clidx(i, jj, ldb))
                for u in rng
                    s = muladd(-cf(jj, j0 + u), unsafe_load(pB, _clidx(i, j0 + u, ldb)), s)
                end
                unit || (s *= d); unsafe_store!(pB, s, _clidx(i, jj, ldb)); i += 1
            end
        end
    end
    @inline function docol(j)                                # one tail solve-col (k not a multiple of NC)
        solved = asc ? (1:(j - 1)) : ((j + 1):k); d = unit ? 1.0 : inv(cf(j, j)); i = 1
        @inbounds while i + W - 1 <= m
            x = vload(_CVF, _cvptr(pB, i, j, ldb))
            for l in solved
                x = muladd(_CVF(-cf(j, l)), vload(_CVF, _cvptr(pB, i, l, ldb)), x)
            end
            unit || (x = x * _CVF(d)); vstore(x, _cvptr(pB, i, j, ldb)); i += W
        end
        return @inbounds while i <= m
            s = unsafe_load(pB, _clidx(i, j, ldb))
            for l in solved
                s = muladd(-cf(j, l), unsafe_load(pB, _clidx(i, l, ldb)), s)
            end
            unit || (s *= d); unsafe_store!(pB, s, _clidx(i, j, ldb)); i += 1
        end
    end
    # ORDER (as side-L): ascending ⇒ blocks low→high then tail cols; descending ⇒ tail cols (high) FIRST,
    # then blocks high→low (blocks downdate against the tail as "solved").
    if asc
        for bi in 1:nb
            doblock((bi - 1) * NC + 1)
        end
        for j in (nb * NC + 1):k
            docol(j)
        end
    else
        for j in (nb * NC == k ? (0:-1) : (k:-1:(nb * NC + 1)))
            docol(j)
        end
        for bi in nb:-1:1
            doblock((bi - 1) * NC + 1)
        end
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
# PDM: Literal — side-R fuse threshold. NOW A KNOB (was a bare const, unpinnable and untunable); default is the value it always had. | tune: TUNABLE and MIS-SPECIFIED — one scalar serves as BOTH the one-panel ceiling and the recursion leaf; no single value is right at n=128 and n=512 (Zen4 32 -> 0.900 vs 1.031). See task #169.
const _TRSM_R_FUSE = @load_preference("trsm_r_fuse", 128)::Int  # ponytail: lower-T real-f64 side-R fused-panel base cap (= potrf NB); recurse above
@inline _trsm_r_fuse() = (f = _FKR_trsm_r_fuse[]; f >= 0 ? f : _TRSM_R_FUSE)
# BATCH-dim (m = B-rows) floor for the fused side-R panel. Its O(k²) triangle setup (invert-diagonal/pack)
# amortizes over O(m·k) solve work — tax = k/m — so it needs a FEW register-tiles of m, NOT the k-CEILING.
# The old guard `m > _TRSM_NCUT_R(128)` conflated this batch floor with the triangle ceiling, so EVERY square
# n≤128 (m=n≤128) missed its own best kernel → the Zen3 n=128 side-R dip (0.74 vs AOCL; the fused panel does
# 0.87). Derived: max(_CHOLW SIMD row-tile, k>>2 setup-amortization) ⇒ 32 at k=128, admits the square-128 gate.
# _CHOLW is µarch-derived (req#8). Fleet-safe: measured Zen3 n=128 0.74→0.87 AND Zen4
# 0.945→1.04 (both beat AOCL at n=64), 2026-07-16 — the fused panel is no-op-or-better at narrow m on both.
_trsm_r_mfloor(k::Int) = max(_CHOLW, k >> 2)
# Fused side-R lower driver: X·op(A)⁻¹ via the potrf panel leaf `_trsm_rl_split_f64!`, MC row-chunked.
# `Ar` (k×k used, may be a larger grown workspace ⇒ k passed explicitly) holds LOWER-TRANSPOSE-layout
# coefficients: A itself for op='T'; the reflected Ã=J·Aᵀ·J for op='N' (see `_trsm_rrefl`), in which case
# `revB=true` hands the leaf a column-REVERSED view of B — base pointer at column k with a NEGATIVE column
# stride. `_cvptr`/`_clidx` are plain affine Ptr arithmetic, so a negative ld is legal and copies nothing.
# `scratch=true` solves each chunk into the odd-ld rpack (conflict-free solved-column re-reads) then copies
# back; `false` solves in place. Chosen by `k > _TRSM_DBASE` — DERIVE tier, and deliberately NOT a Measure
# knob. Fleet A/B at aliasing ldb (2026-07-26) shows both boxes agree in-place clearly wins at k≤32
# (Zen4 scratch/in-place 1.27–1.52, Zen3 1.18: the O(m·k) copy-back can't amortize against an O(m·k²)
# solve that small) and scratch is a win-or-wash at k≥48 (Zen3 0.68–0.79, Zen4 0.84 at k=48, ~1.0 at 128).
# A Measure-tier crossover was tried and REMOVED: Zen4 is NOT monotone in k (0.84 at k=48 but 1.13 at
# k=64), so "first k where scratch wins" has no stable answer — it returned different values in different
# processes on the same box, cost Zen3 'T' 1024×48 0.688, and no margin (5/10/15%) fixed it. Don't
# re-Measure this without first re-checking monotonicity.
function _trsm_rl_fused_drv!(Ar, B, k::Int, revB::Bool, scratch::Bool)
    m = size(B, 1); ldb0 = stride(B, 2)
    mc0 = max(_vwidth(Float64), (_L2_BYTES ÷ 2) ÷ (k * 8))   # MC row-chunk: the mc×k slab the k-repasses
    GC.@preserve Ar B begin                                  # re-read stays L2-resident (req#8; rows independent)
        pA = pointer(Ar); ldA = stride(Ar, 2)
        pB = pointer(B); ldb = ldb0
        if revB
            pB += (k - 1) * ldb0 * 8; ldb = -ldb0
        end
        if scratch
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

# side 'R': B := B·op(A)⁻¹, A k×k (k=size(B,2)), unscaled.
function _trsm_right!(up::Bool, tr::Bool, cj::Bool, unit::Bool, A, B)
    k = size(A, 1)
    # Lower real-f64 fused base: the fused 12-acc substitution (the potrf panel kernel
    # `_trsm_rl_split_f64!`, MC row-chunked — verified relerr ~1e-15 across 56 variants) — no trtri, no
    # unpacked-gemm-into-tmp, no recurse-to-32. Fires on wide B (m > _TRSM_NCUT_R, ALL k≤fuse) AND on
    # narrow-but-square-ish B (k∈(_TRSM_DBASE,fuse], m ≥ the derived batch floor) — the latter closes the
    # n=128 side-R dip (was falling to the recursion + k=32 scalar bases). Recursion above fuse.
    # BOTH op='T' and op='N' now ride this leaf: X·A=B (lower, NO transpose) is the column-reversal
    # conjugate of the lower-transpose recurrence. With J the k×k reversal and cc = k+1−j,
    #   X[:,j] = (B[:,j] − Σ_{i>j} A[i,j]·X[:,i]) / A[j,j]              (descending j)
    # becomes, on Ã = J·Aᵀ·J (still lower: Ã[cc,κ] = A[k+1−κ, k+1−cc], κ<cc ⇒ row>col) and B·J,
    #   T[:,cc] = (src[:,cc] − Σ_{κ<cc} Ã[cc,κ]·T[:,κ]) / Ã[cc,cc]      (ascending cc, i = k+1−κ)
    # i.e. literally the leaf's own pass. B·J costs zero (negative column stride); Ã costs one O(k²/2)
    # triangle copy, ≪ the O(m·k²) solve. A pure pointer trick can NOT do it: `_clidx(cc,k,ld0)` hard-codes
    # cc's stride at 1, so no base/ld choice swaps which index carries the parameter. Verified relerr
    # ≤1.2e-15 incl. chunked, k%4≠0 (nb<4 remainder) and scalar tails, 2026-07-26.
    # op='N' additionally needs its O(k²) reflect copy to amortize. Measured decomposition (Zen4,
    # 2026-07-26): the reflected solve is exactly as fast as native 'T' (12.50 vs 12.65 µs at m=32,k=128;
    # 98.16 vs 98.64 at m=256) — the identity is FREE — so 100% of any op='N' deficit is that copy, which
    # costs 44%/21%/9% of the call at m=32/96/256 (k=128). Its per-element cost is governed by whether A's
    # k×k footprint stays L1-resident: at k≤64 (≤32 KB) the copy is cheap enough to pay at ANY m, above
    # that it needs the wide-B predicate to amortize. DERIVE tier (residency over `_L1_BYTES`, req#8) —
    # gating exactly here keeps every large-m win and avoids regressing m=32/96 at k=128 (0.96→0.68 and
    # 0.87→0.81 without it). Tiling the copy was tried and is uniformly SLOWER (see the loop's comment).
    if !up && !unit && !cj && k <= _trsm_r_fuse() && eltype(B) === Float64 && _strided1(B) &&
            (tr || k * k * sizeof(Float64) <= _L1_BYTES || size(B, 1) > _trsm_ncut_r()) &&
            (size(B, 1) > _trsm_ncut_r() || (k > _trsm_dbase() && size(B, 1) >= _trsm_r_mfloor(k)))
        Ar = A
        # _EXP10 — A-SIDE de-aliasing. At transA='T' the `!tr` branch below does NOT fire, so A is handed
        # to the leaf VERBATIM at the caller's lda. For a square gate operand that is lda = k, and at
        # k=128 the byte stride is 1024 = a quarter L1 way period, which puts A's columns on the same
        # sets. MEASURED on Zen3 (AVX2), leaf GF at bs=128, m=128, only A's lda moving:
        #     lda 128 (shipped) 41.59 | 129 49.14 | 130 48.57 | 132 50.11 | 136 48.81 | 144 49.75
        # i.e. ANY non-po2 lda is +11.5..+15.1%, and the padded values land on the leaf's own
        # `1/GF = α + β/bs` trend (asymptote 47.8 GF) — bs=128 was a DIP, not a rate deficit.
        # CONTROL bs=96 (lda already non-po2): +1.5..+2.9%, the run-to-run floor.
        # This is NOT either of the two hypotheses already falsified for this cell: the de-aliasing work
        # tested `_alias_ld(stride(B,2))` — B's ldb, and only the FULL way period (512 doubles, so
        # lda=128 could never trip it) — and the _RL_MR work changed register pressure, not addressing.
        # Predicate REUSED, not invented: `_potrf_needs_pad` is the derived quarter-period + L2-residency
        # test (the residency half matters — `_chol_needs_pad` measured a pad LOSING at n=384 once the
        # block spills L2, because then the copy round-trip is pure traffic). `_trsm_rpack` already
        # returns an odd-ld scratch, which by construction can never be a way-stride multiple.
        # WHEN the pad pays — DERIVED, and NOT simply "whenever the stride aliases". Measured both boxes,
        # gate shape, end to end (pad/shipped, <1 = pad faster):
        #        n=128     n=256     n=512
        #   Zen3 0.8725    0.9411    0.8447      (AVX2, _NVREG=16)
        #   Zen4 1.0489    1.0115    0.9112      (AVX-512, _NVREG=32)
        # The SIGN INVERTS at n=128: a blanket predicate would have taken Zen4's passing 1.05 cell down
        # to ~1.00. The conflict only costs more than the copy when A's columns are actually RE-READ:
        #   * the leaf SPILLS (it holds _RL_MR*_CHOL_NB accumulators + _CHOL_NB T-vectors + ~2 more; at
        #     3*4+4+2 = 18 that exceeds AVX2's 16 ymm but not AVX-512's 32 zmm), so AVX2 re-reads A; or
        #   * A does not fit L2, so every re-read is an L3 trip regardless of register file.
        # Both terms are formulas over detected consts, not a µarch literal. This reproduces all six
        # measured cells above; per req#8b it is derive → validate → ship, and the validation is the
        # fleet A/B that produced the table.
        # GATED ON THE SPILL TERM ALONE. The pad is a TRADE (copy vs conflict), so the aliasing stride
        # is only half the test — it says a conflict exists, not that removing it beats the copy. What
        # decides that is whether A's columns are actually RE-READ, and the leaf only re-reads them when
        # it SPILLS: `_RL_MR_LIVE` = 3·_CHOL_NB + _CHOL_NB + 2 = 18 live vectors, over AVX2's 16 ymm and
        # under AVX-512's 32 zmm. Measured:
        #   Zen3 (spills)     — wins at EVERY size: +12.8/5.9/15.5/9.9/4.9/2.3% at n=128..4096,
        #                       gate worst 0.821 → 0.924, n=512 and n=1024 closed.
        #   Zen4 (no spill)   — n=128 the pad is 4.9% SLOWER (it would turn a passing 1.05 cell into
        #                       ~1.00), and n=512/1024/2048/4096 are NULL (1.0014/1.0009/0.9999/1.0003,
        #                       SE 0.0003..0.0039). So on a non-spilling box the pad buys nothing
        #                       anywhere and costs at small k.
        # An earlier version added `k²·sz > _L2_BYTES` to pad large A on non-spilling boxes too. That
        # came from ONE probe reading 0.9112 at Zen4 n=512; a tighter re-measure across four sizes could
        # not reproduce it, so the term was unvalidated and is removed rather than kept on a single
        # observation. Consequence: Zen4 now never pads, which is byte-identical to its pre-change
        # behaviour — the 0.958 → 0.944 seen on one Zen4 gate run was drift, not this change.
        # _EXP10 INVERTED: set true to DISABLE the pad, so the shipped arm stays A/B-able in-process.
        if tr && !_EXPFLAG[_EXP10] && _RL_MR_LIVE > _NVREG && _potrf_needs_pad(A, k)
            S = _trsm_rpack(Float64, k, k)
            @inbounds for c in 1:k, r in c:k          # lower triangle only — the leaf reads nothing above it
                S[r, c] = A[r, c]
            end
            Ar = view(S, 1:k, 1:k)
        end
        if !tr
            Ar = _trsm_rrefl(Float64, k)
            # Contiguous-WRITE order: the inner r-loop stores Ar[:,c] unit-stride and reads A with a
            # constant lda stride the prefetcher follows. Cache-BLOCKING this (8×8 tiles = one A line per
            # tile row) was tried and measured uniformly WORSE — 32×48 0.85→0.69, 256×48 1.01→0.97 vs AOCL
            # (Zen4, 2026-07-26): the tiles' computed triangular bounds cost more than the traffic they
            # save. Do not re-try tiling here without measuring.
            @inbounds for c in 1:k, r in c:k
                Ar[r, c] = A[k + 1 - c, k + 1 - r]
            end
        end
        return _trsm_rl_fused_drv!(Ar, B, k, !tr, _alias_ld(stride(B, 2)) && k > _trsm_dbase())
    end
    if eltype(B) <: BlasReal && !cj
        # narrow B (few rows) → dense column-substitution base; wide → invR/gemm base. m is invariant
        # under the side-R column split. (Same dense/gemm split as side L, routed by _TRSM_NCUT_R.)
        if size(B, 1) <= _trsm_ncut_r()
            k <= _trsm_dbase() && return _trsm_dense_R!(up, tr, unit, A, B)
        elseif k <= _trsm_base()
            return _trsm_base_invR!(up, tr, unit, A, B)
        end
    elseif eltype(B) <: BlasComplex
        # Non-trans: k≤64 uses the trtri-free direct base (beats OB; fixes the universal small-n collapse),
        # and 64<k recurses all the way down to it + gated _gemm_subR!. Measured (consistent harness, all
        # three µarchs) uniformly ≥ the invert+K-TRIM base for side-R — the trtri never amortizes here
        # (its O(k³/6) invert is 40–66% exposed even at k=256, where two 128-trtri bases capped 0.95 on AVX2
        # and direct-recurse beats the wide-B trtri path on AVX-512/Zen5 too). Trans keeps the ≤128 base.
        recbase = (!tr || cj) ? _fh_ctrsm_rec_l() : _TRMM_BASE   # transA='N'/'C' → direct-base recursion; 'T' → trtri base
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
    # α==0 ⇒ B := 0, and reference ?trsm guarantees "A is not referenced and B need not be set". Both
    # halves matter: `_scal_all!` MULTIPLIES (dscal semantics), so an unset B holding Inf/NaN gives
    # 0·Inf = NaN instead of 0; and a caller is entitled to pass a singular A (say A[1,1]=0), whose
    # reciprocal in the solve below would then contaminate the zeros. Explicit zero-fill, no solve.
    # Not applied to AD types — see the matching note in `_gemm_core!` on `iszero(::Dual)`.
    if eltype(B) <: Union{BlasReal, BlasComplex} && iszero(α)
        fill!(B, zero(eltype(B)))
        return B
    end
    isone(α) || _scal_all!(B, α)
    side_left ? _trsm_left!(up, tr, cj, unit, A, B) : _trsm_right!(up, tr, cj, unit, A, B)
    return B
end

# When A's leading dim is a pure power of 2 (≥512), packing its triangular sub-views thrashes one cache
# set (column starts alias) — measured trsm 0.78–0.94 at ld∈{1024,2048} vs 1.0–1.12 at non-po2. Copying
# A once into a padded-ld scratch (ld=k+8) removes the conflict (B-padding doesn't help — it's the A
# sub-view packing). ponytail: only A needs it; B is solved in place. Cost O(k²) ≪ trsm O(k²n).
# A-pad for power-of-2 leading dims: on AVX2 the O(k²) copy costs MORE than the po2 cache-set aliasing it
# avoids — measured on an idle core (Zen3 is shared → use a free core), disabling it lifts trsm n=512
# 0.89→0.94, n=1024 0.95→0.98, n=2048 →1.02, and getrf (built on trsm) 0.88→0.96. The old "conflict is
# catastrophic 0.78→1.12, the copy pays" was a pre-clean (contended / pre-trtri-fix) measurement. Kept
# for AVX-512/other (untested there; trsm already gates), disabled on AVX2.
@inline _badld(ld::Int) = _vwidth(Float64) != 4 && ld >= 512 && (ld & (ld - 1)) == 0
# Aliasing leading dim: a multiple of the L1 WAY STRIDE (L1_BYTES ÷ assoc ÷ 8 doubles; x86 L1 ≈ 8-way) maps
# every matrix column to the same L1 set → conflict misses on repeated column re-reads. Generalizes _badld
# (po2-only): 1536 = 3·512 also aliases on a 32KB/8-way L1. Derived from detected L1 (req#8). Independent of
# vector width (cache geometry, not ISA) — used by the side-R fused leaf's pT-scratch pack.
# Doubles per L1 way. DERIVED from `_L1_WAY_BYTES`, which already divides by the DETECTED
# `_L1D_ASSOC` — do not re-derive it here.
#
# This was `max(64, _L1_BYTES ÷ 64)`, i.e. "÷8-way ÷8-byte/double" with the associativity HARDCODED to
# 8. That is right on every 32 KiB/8-way part BY COINCIDENCE and wrong anywhere else: Zen5 has a
# 48 KiB 12-way L1, so the old form gave 768 doubles (6144 B) where the true way stride is 512 doubles
# (4096 B). `_alias_ld` therefore tested multiples of 6144 on that box and MISSED every power-of-two
# leading dimension — exactly the strides that alias — so the side-R de-aliasing guard it gates never
# fired there. Found while chasing why gemv-N's m-inner panel loses only at po2 sizes on Zen5.
# Zen3/Zen4 values are unchanged (4096 B both ways); only Zen5 moves.
const _L1_WAY_D = _way_doubles(_L1_BYTES, _L1D_ASSOC)
@inline _alias_ld(ld::Int) = ld >= _L1_WAY_D && ld % _L1_WAY_D == 0
# FALSIFIED 2026-07-30 (Zen3, freq-locked, plots.jl): extending this to the QUARTER way period
# (`ld % (_L1_WAY_D>>2) == 0` gated on B being L2-resident, mirroring `_chol_needs_pad`'s fleet-measured
# criterion at lapack.jl:722-731) does NOT move trsmR — worst 0.82 → 0.81 vs AOCL, i.e. noise.
# The motivating correlation is real but NOT causal: the two sub-gate cells (ldb=128, 256) are exactly
# the two where `_alias_ld` is false, while ldb=512/1024 get the dealias arm and beat AOCL 1.27-1.66×.
# Whatever makes 512/1024 fast, it is not this predicate. Do not re-derive the quarter-period widening.
# Next suspect for trsmR n=128 is the fused leaf's hardcoded MR=3 × NC=_CHOL_NB=4 = 12-accumulator top
# tier (lapack.jl:792-816): on AVX2's 16 ymm that is 12 accs + 3 T-vectors + 1 broadcast = exactly 16,
# with 10 loop-invariant broadcasts that must spill (UNVERIFIED — check with @assert_no_spill first).

# ── Alias-strategy self-tuning constant (PDM ladder, MEASURE tier — req#8b) ────────────────────────────
# On an L1-way-stride ldb the side-R leaf's solved-column re-reads collide in one set. Two cures exist:
# solve into the ODD-ld rpack scratch and copy back, or just solve in place and eat the conflicts. Which
# wins is NOT derivable from any detected const — it is the copy-back round-trip (m·k·8 B each way,
# chunk-size-independent) traded against the conflict rate, and it INVERTS SIGN across µarchs we own:
# Zen3 measured scratch +41–49% at m=512/1024/1536, while Zen4/Zen4 measures the same scratch
# arm a 15–45% net LOSS (in-place 33.9/37.3/37.3 vs arm 23.4/29.1/33.9 GF at k=16/32/64, m=ldb=1024,
# 2026-07-26). Sign inversion on benchmarked boxes is the ladder's explicit Measure tell — a literal or a
# µarch-gated literal would be wrong on any unseen machine. Candidate set is the 2 strategies; the probe
# shape is DERIVED (k = _TRSM_DBASE, m = ldb = 2·_L1_WAY_D ⇒ aliasing by construction, inside the fused
# range where the misses live). Pin `trsm_r_alias_scratch` — the trim/.so build MUST set it (a runtime
# benchmark is not trim-safe); `@static if` then compiles the measure branch out entirely.
# _l3_apad (po2-ld A-pad, ld=k+8) lives in the per-type L3Workspace (see src/workspace.jl).

# Public: B := α·op(A)⁻¹·B (side 'L') or α·B·op(A)⁻¹ (side 'R'); A k×k triangular (uplo/transA/diag).
# A and B MUST agree in eltype before any BlasFloat fast path runs. Every such path — real and complex,
# both sides — reinterprets A through `pointer(A)` at B's element width, so a mismatched A is read at the
# wrong stride: measured Float32-A/Float64-B rel error 1.5e7, ComplexF32-A/ComplexF64-B either NaN or a
# silently wrong finite answer, and a Float64 A under a ComplexF64 B runs ~2× off its own allocation.
# Found by adversarial review 2026-08-02; the reference BLAS never has to consider it (one type per
# symbol), and Mode 1/LBT callers always match, which is why nothing caught it.
#
# Promote rather than throw where the result is representable — `promote_type(TA,TB) === TB` means B can
# hold the answer, so a one-off O(k²) copy of A gives the mathematically correct result on a path that
# is never hot (internal and LBT callers always match, so this is a no-op for them). Where B cannot hold
# the answer (ComplexF64 A into a Float64 B, or a Dual A under a Float64 B) it is a genuine user error
# and must be loud, never a silent narrowing.
#
# NOT applied when B's eltype is outside BlasFloat: that is the generic/AD path (e.g. Float64 A with
# ForwardDiff.Dual B), which is type-generic scalar code, handles the mixture correctly today, and would
# be broken by forcing a convert.
@inline function _trsm_matchel(A::AbstractMatrix, B::AbstractMatrix)
    TA, TB = eltype(A), eltype(B)
    (TA === TB || !(TB <: BlasFloat)) && return A
    promote_type(TA, TB) === TB || throw(
        ArgumentError(
            "trsm!: eltype(A)=$TA cannot be represented in eltype(B)=$TB; " *
                "convert the operands explicitly"
        )
    )
    return convert(AbstractMatrix{TB}, A)
end

function trsm!(
        B::AbstractMatrix, A::AbstractMatrix; side::Char = 'L', uplo::Char = 'U',
        transA::Char = 'N', diag::Char = 'N', alpha::Number = true
    )
    sl = side == 'L'
    k = sl ? size(B, 1) : size(B, 2)
    (size(A, 1) == size(A, 2) == k) || _throw_square(:trsm!, k)
    A = _trsm_matchel(A, B)
    # NON-UNIT-STRIDE OUTPUT: stage through a contiguous copy. Same defect and same reasoning as
    # `trmm!` -- the real paths read and write B at its ld and returned silently wrong numbers when
    # `stride(B,1) != 1`. O(k·n) against the solve's O(k²·n), paid only by inputs that were wrong before.
    if eltype(B) <: BlasReal && !_strided1(B)
        Bc = Matrix(B)
        trsm!(Bc, A; side, uplo, transA, diag, alpha)
        copyto!(B, Bc)
        return B
    end
    # tiny-k fast path: skip the _trsm!/_trsm_left!/_trsm_right! dispatch chain (~3 non-inlined calls ≈ 60ns,
    # which dominates when the solve itself is only ~100ns) and go straight to the base kernel.
    # The bypass criterion is n-DEPENDENT for side-L, exactly as the side-R note below says of m.
    # Skipping ~60ns of dispatch only pays while the whole solve is itself O(100ns); with WIDE B the
    # solve is tens of µs and the bypass instead pins k ≤ _TRSM_DBASE to the SCALAR dense base, which
    # reads and writes B directly at its ld. Two costs, both measured on Zen4 F64 at k=32, n=256:
    #   • even at a friendly ld the dense base is 33.3 µs against the invL leaf's 18.4 µs (1.8× worse);
    #   • the direct B access is exposed to column-stride cache-set aliasing, so it degrades to 61.8 µs
    #     at ldb=256, 223.4 at 512, 73.1 at 768, 219.9 at 1024 — a 6.7× cliff. The blocked path is flat
    #     across that whole ld sweep, and k=40 (just above the bypass) shows no ld sensitivity at all,
    #     which is the control.
    # This was the entire nb=32→40 discontinuity in blocked gbtrf (0.91→1.20 vs OpenBLAS at kl≥256):
    # its panel trsm is exactly k=nb, n=ku, with ldb = 2kl+ku. Wide side-L now falls through to the
    # normal routing, which still reaches the fused leaf via _trsm_left! when it applies. Gated on the
    # same _TRSM_NCUT the narrow/wide split already uses, so no new tuning constant is introduced.
    if k <= _trsm_dbase() && eltype(B) <: BlasReal && transA != 'C' && isone(alpha) &&
            !(side == 'L' && size(B, 2) > _trsm_ncut())
        up = uplo == 'U'; tr = transA != 'N'; unit = diag == 'U'
        # SINGLE column, side-L: one trsv beats the dense base even down here. This path returns
        # before the narrow-B branch below, so without this the sweep never fires for k ≤ _TRSM_DBASE
        # — exactly the sizes most dominated by per-call overhead. Measured Zen4 (ns, dense base vs
        # trsv): k=8 151→121, k=16 251→211, k=24 361→311, k=32 481→421, i.e. 12–20% faster.
        # nrhs=1 ONLY: at nrhs=2 and 4 in this range the dense base wins (k=32: 711 vs 802, 751 vs
        # 1573), because it amortises across columns while the sweep re-walks A each time.
        if sl && size(B, 2) == 1 && _strided1(B)
            trsv!(A, view(B, :, 1); uplo = uplo, trans = transA, diag = diag)
            return B
        end
        # k in [_TRSM_FUSED_MIN, _TRSM_DBASE]: the fused gemmtrsm leaf beats the scalar dense base even here
        # (Zen4: k=24 15.9 vs 9.4, k=32 13.5 vs 11.1 GF) — take it too (side-L up-notrans, fusable),
        # keeping the low-overhead tiny entry. Below _TRSM_FUSED_MIN the dense base wins (setup unamortized).
        # `_GT_TRANSPOSE` REMOVED 2026-08-10. It is a CAPABILITY bit for the 8x8-transpose fusedT slabs
        # (_GT_W == 8), and it belongs inside the leaf — where it still is, at `useT` — not on the decision
        # to ENTER the leaf. Used here it silently pinned all of AVX2 to the scalar dense base, and the
        # comment that justified it recorded no AVX2 measurement. Direct-call A/B on Zen3, routing
        # bypassed, correctness checked first at max|dense-fused| ~ 1.3e-15:
        #     k=32  fused/dense 0.8381  SE 0.0018  n=105   (fused 16.2% FASTER)
        #     k=48  fused/dense 0.7935  SE 0.0015  n=56    (20.7% faster)
        #     k=64  fused/dense 0.7408  SE 0.0043  n=48    (25.9% faster)
        # The leaf was already production-proven on AVX2 via the wide-B branch (_trsm_left!), which calls it
        # with no such guard — so this deletes an unmeasured restriction, it does not enable new code.
        # Float32 is still excluded by `_trsm_fusable` (eltype(B) === Float64, measured -18% if fused), and
        # AVX-512 is unaffected because the conjunct was `true` there.
        if sl && up && !tr && _TRSM_FUSED_ON[] && k >= _TRSM_FUSED_MIN && _trsm_fusable(A, B)
            return _trsm_fused_L!(unit, A, B)
        end
        # Side-L: the dispatch-skip criterion is n-DEPENDENT, exactly as the side-R note below says of
        # m. Skipping ~60ns of dispatch only pays while the whole solve is itself O(100ns); with wide B
        # the solve is tens of µs and the bypass instead pins k≤32 to the SCALAR dense base, which
        # reads and writes B directly at its ld. Two costs, both measured on Zen4 F64 at k=32, n=256:
        #   • even at a friendly ld the dense base is 33.3 µs vs the invL leaf's 18.4 µs (1.8× worse);
        #   • the direct B access is exposed to column-stride cache-set aliasing, so it degrades to
        #     61.8 µs at ldb=256, 223.4 at 512, 73.1 at 768, 219.9 at 1024 — a 6.7× cliff. The blocked
        #     path is flat across the same ld sweep, and k=40 (just above this bypass) shows no ld
        #     sensitivity at all, which is the control.
        # This was the whole nb=32→40 discontinuity in blocked gbtrf (0.91→1.20 vs OpenBLAS at kl≥256):
        # its panel trsm is exactly k=nb, n=ku, with ldb = 2kl+ku. Gate on the same _TRSM_NCUT the
        # narrow/wide split already uses below, so no new tuning constant appears.
        sl && return _trsm_dense_L!(up, tr, unit, A, B)
        # Side-R: the dispatch-skip criterion above is m-DEPENDENT — skipping ~60ns of dispatch only pays
        # while the whole solve is itself O(100ns). For m > _TRSM_NCUT_R the fused f64 leaf (reachable ONLY
        # through _trsm_right!) beats the dense base 1.29–1.62× at k=16/32 (Zen4, 2026-07-26), and this
        # bypass was the ENTIRE side-R transA='T' gap vs AOCL: n≤32 measured 0.61–0.96 while every n≥48
        # cell — which already reached the leaf — wins (1.03–1.57). Falling through also picks up the
        # `_alias_ld` handling the bypass skipped. Reuses the existing _TRSM_NCUT_R routing predicate, so
        # no new tuning constant is introduced (that const's own PDM debt is noted at its definition).
        if !up && !unit && eltype(B) === Float64 && _strided1(B) && size(B, 1) > _trsm_ncut_r()
            return _trsm_right!(up, tr, false, unit, A, B)
        end
        return _trsm_dense_R!(up, tr, unit, A, B)
    end
    # ── NARROW B (side='L'): sweep trsv per column instead of the blocked solve ────────────────────
    # The blocked path pays an O(k·nb²) setup — the triangular inverse of the diagonal blocks — that is
    # amortised over B's columns. With few columns it is never repaid, and the cost is near-constant in
    # nrhs: measured Zen4 k=1024, PB trsm took 1123 µs at nrhs=1 and 1197 µs at nrhs=8, i.e. eight
    # solves for the price of one. Against OpenBLAS that was 0.21 at nrhs=1 — while PB's own trsv beat
    # OB's trsm by 2.87× on the same shape. This was invisible because the trsm gate row uses SQUARE
    # operands, so the narrow-B regime had never been measured; it is also the whole reason getrs/potrs
    # measured 0.07–0.27 vs OpenBLAS (`A \ b` with one RHS is exactly nrhs=1).
    # Ratio trsm/per-column-trsv, Zen4 (>1 ⇒ sweep wins):
    #     k\nrhs      1      2      4      8     16     32
    #     256      2.18   1.75   1.55   0.48   0.48   0.84
    #     512      9.70   5.30   3.08   1.53   1.46   1.47
    #     1024    13.98   7.21   3.83   1.87   1.44   1.23
    #     2048    12.49   6.31   3.28   1.58   1.03   0.76
    # PDM Derive: the crossover scales with k because the setup does — nrhs* ≈ k/(4·_TRSM_DBASE),
    # _TRSM_DBASE being the already-derived dense-base width whose diagonal blocks are what gets
    # inverted. On Zen4 (base 32) that is k÷128, which is at or inside the measured crossover at every
    # k above — conservative by construction, and it captures the large wins (all of nrhs=1, out to
    # nrhs=16 at k=2048). Overridable via "trsm_narrow_max".
    # Only side='L': side='R' is narrow in its ROWS, a different kernel question, not measured here.
    if sl && _fh_trsm_narrow_max() > 0 && size(B, 2) <= _fh_trsm_narrow_max() &&
            _strided1(B) && size(B, 1) == k
        isone(alpha) || (B .*= alpha)          # trsv has no α; A⁻¹(αB) = α·A⁻¹B, so prescaling is exact
        @inbounds for j in 1:size(B, 2)
            trsv!(A, view(B, :, j); uplo = uplo, trans = transA, diag = diag)
        end
        return B
    end
    # ── RAGGED COLUMN TAIL: widen B to a full SIMD lane before solving ────────────────────────────
    # The blocked kernel processes B's columns in W-wide lanes and a partial lane costs a FULL lane —
    # so a B narrower than W pays per column what W columns would cost together. Measured Zen4
    # (W=8, k=512, µs): nrhs=1..7 cost 96/180/278/381/464/543/675 — i.e. ~95 µs PER COLUMN — while
    # nrhs=8 costs 94 in total. Eight columns for the price of one.
    # Widening the ragged B into scratch, solving once, and copying back collapses that:
    #     k=512   nrhs=3  278 → 97   nrhs=5  464 → 98   nrhs=7  675 → 100    (2.9–6.8× faster)
    #     k=1024  nrhs=3  945 → 407  nrhs=5 1407 → 405  nrhs=7 1712 → 401    (2.3–4.3×)
    # vs OpenBLAS those cells go 0.35/0.26/0.21 → 0.99/1.21/1.41.
    # This is the SAME defect the upper fused leaf had (fixed earlier by a per-stripe NRV downshift,
    # see the trsm-smalln-ragged-stripe note) — it survived here in the lower/notrans blocked path,
    # which is exactly the one getrs/potrs use, hence `A \ b` being the worst case of all.
    # ONLY nrhs < W, where the whole of B is one ragged lane. Above W the trade is not one-signed —
    # measured nrhs=9 padding to 16 is 0.57× (a 1-column tail becomes 7 fake columns) while nrhs=15
    # is 2.42× — so a general rule needs the per-lane cost model, not a rounding. Left alone.
    # The extra columns are ZEROED, not garbage: a triangular solve on a zero column yields zero and
    # cannot trap, whereas `undef` scratch could hold NaN/Inf and raise spurious FP exceptions.
    if sl && _strided1(B) && size(B, 1) == k && 1 < size(B, 2) < _vwidth(eltype(B)) &&
            eltype(B) <: BlasFloat
        nr = size(B, 2); w = _vwidth(eltype(B))
        Bw = _trsm_tmp(eltype(B), k, w)                      # GKH-owned (L3Workspace), grows
        Bv = view(Bw, 1:k, 1:w)
        @inbounds copyto!(view(Bv, :, 1:nr), B)
        @inbounds fill!(view(Bv, :, (nr + 1):w), zero(eltype(B)))
        _trsm!(sl, uplo == 'U', transA != 'N', transA == 'C', diag == 'U', alpha, A, Bv)
        @inbounds copyto!(B, view(Bv, :, 1:nr))
        return B
    end
    # NOTE: a po2-`lda` A-pad used to sit here — when stride(A,2) was a bad (power-of-two) stride, A
    # was copied into an odd-ld scratch (`_l3_apad`, ld = k+8) before the solve, to dodge cache-set
    # aliasing. It was REMOVED after measuring it: the copy is O(k²) and never repaid. Zen4, k=1024,
    # nopad vs pad, all sixteen combinations of side × uplo × transA × B-width:
    #     side-L narrow B  +198% / +71% / +229% / +69%      side-L wide B  +4% / +15% / +3% / +12%
    #     side-R narrow B  +346% / +103% / +357% / +101%    side-R wide B  +7% / +6% / +11% / tie
    # It is a loss everywhere — worst for narrow B, where an O(k²) copy is paid for an O(k·nrhs)
    # solve, but still a loss at nrhs = k. Removing it also slightly IMPROVES the square-B gate shape
    # (OB/PB at k=1024, nrhs=1024: 1.355 with the pad → 1.388 without).
    # Whatever aliasing motivated it is evidently already handled inside `_trsm!`'s own blocking; the
    # pad was paying a second time for it. `_l3_apad`/`L3Workspace.apad` are now unused by trsm and
    # kept only so the buffer and its rationale survive in one place if a future shape needs them.
    _trsm!(sl, uplo == 'U', transA != 'N', transA == 'C', diag == 'U', alpha, A, B)
    return B
end

# ──────────────────────────────────────────────────────────────────────────────────────────────
# syrk/herk: C := α·op(A)·op(A)ᴴ + β·C, C n×n, only the `uplo` triangle referenced/updated.
# trans 'N': op(A)=A (n×k) ⇒ A·Aᴴ. trans 'T'/'C': op(A)=Aᴴ (A k×n) ⇒ Aᴴ·A. syrk: ᵀ (no conj),
# any T<:Number. herk: Hermitian (conj), real α/β, diagonal forced real. Recursive: diagonal blocks
# recurse (scalar base), the off-diagonal block is a full gemm! — breadth-first correctness (gate later).
# PDM: Literal — syrk recursion base before the off-diagonal gemm. NOW A KNOB (was a bare const, unpinnable and untunable); default is the value it always had. | tune: FLAT — 16..96 within noise on all 3 uarchs; largest cell +0.8% (Zen3 n=128) does not replicate (2026-08-21)
const _SYRK_BASE = @load_preference("syrk_base", 48)::Int
@inline _syrk_base() = (f = _FKR_syrk_base[]; f >= 0 ? f : _SYRK_BASE)

@inline _symstored(up::Bool, i, j) = up ? (i <= j) : (i >= j)
# β-prescale C's stored triangle. Branch-free, contiguous, triangle-only (was: all n² with a per-element
# _symstored branch — measured 22%/13%/8% of syrk! at n=32/128/256, the whole gate gap since the kernel
# already gates). β=1 is a no-op (skip); β=0 zeroes only the stored half.
function _syrk_scaleC!(C, up::Bool, β)
    isone(β) && return C
    T = eltype(C); n = size(C, 2); z = iszero(β)
    @inbounds for j in 1:n
        for i in (up ? (1:j) : (j:n))
            C[i, j] = z ? zero(T) : β * C[i, j]
        end
    end
    return C
end
# _L3_NB and the NB×NB diagonal-block scratch `_l3_tmp(T)` (the workspace `diag` field) live in
# src/workspace.jl — const-dispatched for Float64/Float32 so it stays a bare field load, no lookup.

# Triangular-store microkernel: same FMA as the gemm masked microkernel, but on store keeps only the
# stored-triangle entries — for a diagonal-straddling C-tile whose top-left global offset is (r0,c0),
# d0=c0-r0; upper keeps local row ≤ d0+j, lower keeps row ≥ d0+j (j = 0-based column). Accumulates into
# C, so K-accumulation across the gemm pc-loop stays correct (no temp needed).
@generated function _microkernel_tri!(
        C::Ptr{T}, ldc::Int, Ap::Ptr{T}, Bp::Ptr{T}, kc::Int,
        mre::Int, nre::Int, d0::Int, upper::Bool, ::Val{MR}, ::Val{NR}, ::Val{B0} = Val(false)
    ) where {T, MR, NR, B0}
    W = _vwidth(T); sz = sizeof(T); V = Vec{W, T}
    body = quote end
    push!(body.args, :(lanes = Vec{$W, Int}($(Expr(:tuple, (0:(W - 1))...)))))
    for mi in 1:MR, j in 1:NR
        push!(body.args, :($(Symbol(:c, mi, :_, j)) = zero($V)))
    end
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
    push!(
        body.args, :(
            for p in 0:(kc - 1)
                $inner
            end
        )
    )
    for j in 1:NR
        stores = quote end
        push!(stores.args, :(colp = C + $(j - 1) * ldc * $sz)); push!(stores.args, :(thr = d0 + $(j - 1)))
        for mi in 1:MR
            cs = Symbol(:c, mi, :_, j)
            st = B0 ? :(vstore($cs, q, mk)) : :(vstore(vload($V, q, mk) + $cs, q, mk))
            push!(
                stores.args, :(
                    let base = $((mi - 1) * W), q = colp + $((mi - 1) * W * sz)
                        rows = lanes + base
                        mk = (rows < mre) & (upper ? (rows <= thr) : (rows >= thr))
                        $st
                    end
                )
            )
        end
        push!(
            body.args, :(
                if $(j - 1) < nre
                    $stores
                end
            )
        )
    end
    push!(body.args, :(return nothing))
    return body
end

# Two-product fused microkernel for syr2k: C += op(X1)op(Y1) + op(X2)op(Y2) for ONE C-tile, with a
# SINGLE C read-modify-write (not two). Both products' panels share the kc accumulation in the same
# registers; only at the end is C touched. This halves C traffic and (in :tri mode) the masked store
# vs running two separate microkernels per tile. MODE picks the store: :full / :masked / :tri.
@generated function _microkernel2!(
        C::Ptr{T}, ldc::Int, Ap1::Ptr{T}, Bp1::Ptr{T}, Ap2::Ptr{T},
        Bp2::Ptr{T}, kc::Int, alpha::T, mre::Int, nre::Int, d0::Int, upper::Bool,
        ::Val{MR}, ::Val{NR}, ::Val{MODE}, ::Val{B0} = Val(false)
    ) where {T, MR, NR, MODE, B0}
    W = _vwidth(T); sz = sizeof(T); V = Vec{W, T}
    body = quote end
    push!(body.args, :(av = $V(alpha)))
    if MODE !== :full
        push!(body.args, :(lanes = Vec{$W, Int}($(Expr(:tuple, (0:(W - 1))...)))))
    end
    for j in 1:NR
        push!(body.args, :(_prefetch(C + $(j - 1) * ldc * $sz)))
    end
    for mi in 1:MR, j in 1:NR
        push!(body.args, :($(Symbol(:c, mi, :_, j)) = zero($V)))
    end
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
    push!(
        body.args, :(
            for p in 0:(kc - 1)
                $inner
            end
        )
    )
    for j in 1:NR
        stores = quote end
        push!(stores.args, :(colp = C + $(j - 1) * ldc * $sz))
        MODE === :tri && push!(stores.args, :(thr = d0 + $(j - 1)))
        for mi in 1:MR
            cs = Symbol(:c, mi, :_, j)
            if MODE === :full
                st = B0 ? :(vstore(av * $cs, q)) : :(vstore(muladd(av, $cs, vload($V, q)), q))
                push!(
                    stores.args, :(
                        let q = colp + $((mi - 1) * W * sz)
                            $st
                        end
                    )
                )
            elseif MODE === :masked
                st = B0 ? :(vstore(av * $cs, q, mk)) : :(vstore(muladd(av, $cs, vload($V, q, mk)), q, mk))
                push!(
                    stores.args, :(
                        let q = colp + $((mi - 1) * W * sz)
                            mk = (lanes + $((mi - 1) * W)) < mre; $st
                        end
                    )
                )
            else # :tri
                st = B0 ? :(vstore(av * $cs, q, mk)) : :(vstore(muladd(av, $cs, vload($V, q, mk)), q, mk))
                push!(
                    stores.args, :(
                        let q = colp + $((mi - 1) * W * sz)
                            rows = lanes + $((mi - 1) * W)
                            mk = (rows < mre) & (upper ? (rows <= thr) : (rows >= thr))
                            $st
                        end
                    )
                )
            end
        end
        if MODE === :full
            push!(body.args, stores)
        else
            push!(
                body.args, :(
                    if $(j - 1) < nre
                        $stores
                    end
                )
            )
        end
    end
    push!(body.args, :(return nothing))
    return body
end

# Single-product α-at-store microkernel for the UNIFIED syrk path: C += α·(A·B), with α applied at the
# store (the unified path packs A once and reads it as both operands, so α cannot be folded into the
# pack — both operands would pick it up, giving α²). MODE: :full / :masked / :tri.
@generated function _microkernel_u!(
        C::Ptr{T}, ldc::Int, Ap::Ptr{T}, Bp::Ptr{T}, kc::Int, alpha::T,
        mre::Int, nre::Int, d0::Int, upper::Bool, ::Val{MR}, ::Val{NR}, ::Val{MODE}, ::Val{B0} = Val(false)
    ) where {T, MR, NR, MODE, B0}
    W = _vwidth(T); sz = sizeof(T); V = Vec{W, T}
    body = quote end
    push!(body.args, :(av = $V(alpha)))
    if MODE !== :full
        push!(body.args, :(lanes = Vec{$W, Int}($(Expr(:tuple, (0:(W - 1))...)))))
    end
    for j in 1:NR
        push!(body.args, :(_prefetch(C + $(j - 1) * ldc * $sz)))
    end
    for mi in 1:MR, j in 1:NR
        push!(body.args, :($(Symbol(:c, mi, :_, j)) = zero($V)))
    end
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
    push!(
        body.args, :(
            for p in 0:(kc - 1)
                $inner
            end
        )
    )
    for j in 1:NR
        stores = quote end
        push!(stores.args, :(colp = C + $(j - 1) * ldc * $sz))
        MODE === :tri && push!(stores.args, :(thr = d0 + $(j - 1)))
        for mi in 1:MR
            cs = Symbol(:c, mi, :_, j)
            if MODE === :full
                st = B0 ? :(vstore(av * $cs, q)) : :(vstore(muladd(av, $cs, vload($V, q)), q))
                push!(
                    stores.args, :(
                        let q = colp + $((mi - 1) * W * sz)
                            $st
                        end
                    )
                )
            elseif MODE === :masked
                st = B0 ? :(vstore(av * $cs, q, mk)) : :(vstore(muladd(av, $cs, vload($V, q, mk)), q, mk))
                push!(
                    stores.args, :(
                        let q = colp + $((mi - 1) * W * sz)
                            mk = (lanes + $((mi - 1) * W)) < mre; $st
                        end
                    )
                )
            else # :tri
                st = B0 ? :(vstore(av * $cs, q, mk)) : :(vstore(muladd(av, $cs, vload($V, q, mk)), q, mk))
                push!(
                    stores.args, :(
                        let q = colp + $((mi - 1) * W * sz)
                            rows = lanes + $((mi - 1) * W)
                            mk = (rows < mre) & (upper ? (rows <= thr) : (rows >= thr))
                            $st
                        end
                    )
                )
            end
        end
        if MODE === :full
            push!(body.args, stores)
        else
            push!(
                body.args, :(
                    if $(j - 1) < nre
                        $stores
                    end
                )
            )
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
# ONE micro-tile of the triangular-output product, classified by its position against the diagonal.
#
# Split out of `_trgemm_packed!`'s innermost loop. Each of the four shape cases used to be written
# TWICE there, once per β-mode, because the β choice is a `Val` and the loop only had a runtime
# `Bool` — six leaves became twelve `b0 ? … : …` call sites and took that loop nest to depth 11.
# `B0` is a type parameter here, so each case is written once and the choice costs nothing.
@inline function _trgemm_tile!(
        ::Val{MR}, ::Val{NR}, ::Val{B0}, Cblk::Ptr{T}, ldc::Int, Apanel::Ptr{T}, Bpanel::Ptr{T},
        kce::Int, mre::Int, nre::Int, full::Bool, off::Int, up::Bool
    ) where {T <: BlasReal, MR, NR, B0}
    # DERIVE W from T, never take it as an `::Int` argument. `_vwidth(T)` const-folds, so `MR * W`
    # and `rem(mre, W)` fold too; passing W in makes them runtime values and the `div`/`rem` become
    # real integer divisions in the micro-tile loop. Measured: this function plus `_trgemm_tiles!`
    # emitted 6 `idiv` when W was an argument, 0 when derived.
    W = _vwidth(T)
    if full && mre == MR * W && nre == NR
        _microkernel!(Cblk, ldc, Apanel, Bpanel, kce, Val(MR), Val(NR), Val(B0))
    elseif full && nre == NR && rem(mre, W) == 0
        # W-ALIGNED PARTIAL ROWS → CLIP, don't mask. `_microkernel_masked!` is fully vectorized but
        # masks only the STORES: it runs all MR row-vectors through the whole k-loop and retires
        # `mre` rows, so a partial panel costs a FULL tile. `_microkernel_clip!` takes the
        # packed-panel stride (Val(MR)) separately from the live vector count (Val(vr)), so it reads
        # the same layout and computes only the vectors that exist. gemm has done this on both its
        # packed and unpacked paths for a while; the triangular kernel never picked it up.
        #
        # ⚠ SCOPE, measured and NARROWER than it first appears — read before citing this.
        # This branch does NOT fire for uplo='U', which is what the gate benchmarks. For upper
        # triangular the only partial row panel is the BOTTOM one, and every tile it owns is on the
        # diagonal (`full` is false), so it goes to `_microkernel_tri!` and never reaches here.
        # Verified: after this change syrk@100 read 0.899 (was 0.900) and syrk@2100 0.945 (was
        # 0.943), and the sawtooth probe's misaligned penalty was 19.6% -> 20.0% with every offset
        # moving ±2% in both directions — i.e. unchanged, with aligned n=64 as control. An earlier
        # version of this comment claimed the branch targeted those two cells. It does not; that
        # claim was falsified by the numbers above.
        #
        # It fires for uplo='L', where the bottom partial panel owns many FULL tiles — the
        # configuration potrf's trailing update issues (`syrk!(…, uplo='L')`). That benefit is
        # UNQUANTIFIED: potrf gates 1.049–1.99 on Zen3 with this in place, but it gated before too,
        # so the run does not isolate this branch. Kept because it is strictly fewer FMAs than the
        # masked tile it replaces and mirrors gemm's own guard exactly — not because a measurement
        # earned it.
        #
        # The sawtooth itself is real and is the residual gap (Zen3,
        # bench/probes/syrk_modmr_sawtooth.jl, n=49..72): cost/flop has period mr=8, offset 0 =
        # 1.061, offset 4 = 1.171, non-W offsets 1.20–1.36. Since this branch does not move it, the
        # upper-triangular waste lives in `_microkernel_tri!` and the diagonal blocks, NOT in the
        # masked kernel. That is where the next attempt should go.
        vr = div(mre, W)
        if vr == 1
            _microkernel_clip!(Cblk, ldc, Apanel, Bpanel, kce, Val(MR), Val(1), Val(NR), Val(B0))
        elseif vr == 2
            _microkernel_clip!(Cblk, ldc, Apanel, Bpanel, kce, Val(MR), Val(2), Val(NR), Val(B0))
        else
            _microkernel_masked!(Cblk, ldc, Apanel, Bpanel, kce, mre, nre, Val(MR), Val(NR), Val(B0))
        end
    elseif full
        _microkernel_masked!(Cblk, ldc, Apanel, Bpanel, kce, mre, nre, Val(MR), Val(NR), Val(B0))
    else
        _microkernel_tri!(Cblk, ldc, Apanel, Bpanel, kce, mre, nre, off, up, Val(MR), Val(NR), Val(B0))
    end
    return nothing
end

# The (jr, ir) micro-tile sweep over ONE packed (ic, pc) panel.
#
# `B0` — `OV && pc == 0`, i.e. whether this k-block overwrites C rather than accumulating — is
# constant for the whole panel, so the caller resolves it once and passes it as a type parameter.
# It used to be a runtime `Bool` re-tested at every micro-tile, inside the innermost of five loops.
function _trgemm_tiles!(
        ::Val{MR}, ::Val{NR}, ::Val{B0}, up::Bool, App::Ptr{T}, Bpp::Ptr{T}, Cp0::Ptr{T},
        ldc::Int, sz::Int, ic::Int, jc::Int, mce::Int, nce::Int, kce::Int
    ) where {T <: BlasReal, MR, NR, B0}
    # See `_trgemm_tile!`: W is DERIVED, not passed, so `mr`/`nr` stay compile-time and the
    # `div(ir, mr)` / `div(jr, nr)` in the tile loop below fold to shifts instead of `idiv`.
    W = _vwidth(T); mr = MR * W; nr = NR
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
                _trgemm_tile!(
                    Val(MR), Val(NR), Val(B0), Cblk, ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel),
                    kce, mre, nre, full, c0 - r0, up
                )
            end
            ir += mr
        end
        jr += nr
    end
    return nothing
end

# General triangular-C gemm: C[uplo-triangle] += α·op(X)·op(Y) (X→A-operand, Y→B-operand), n×n result.
# The reusable core behind syrk (Y=X) and syr2k (two passes). Real only; α folded into packed X.
function _trgemm_packed!(
        ::Val{MR}, ::Val{NR}, up::Bool, α::T, X, tXp::Bool, Y, tYp::Bool, C, k::Int,
        ::Val{OV} = Val(false)
    ) where {T <: BlasReal, MR, NR, OV}
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
                    # β-mode resolved ONCE per panel, not once per micro-tile. See `_trgemm_tiles!`.
                    b0 ?
                        _trgemm_tiles!(Val(MR), Val(NR), Val(true), up, App, Bpp, Cp0, ldc, sz, ic, jc, mce, nce, kce) :
                        _trgemm_tiles!(Val(MR), Val(NR), Val(false), up, App, Bpp, Cp0, ldc, sz, ic, jc, mce, nce, kce)
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
function _trgemm_cmplx_packed!(
        ::Val{SA}, ::Val{SB}, ::Val{NR}, ::Val{A1}, up::Bool,
        alr::T, ali::T, X, tXp::Bool, Y, tYp::Bool, C, k::Int
    ) where {SA, SB, NR, A1, T}
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
                                    _microkernel_cmplx!(
                                        Cblk, ldc, AR, AI, BR, BI, kce, alr, ali,
                                        Val(_CMR), Val(NR), Val(SA), Val(SB), Val(false), Val(A1)
                                    )
                                elseif full
                                    _microkernel_cmplx_masked!(
                                        Cblk, ldc, AR, AI, BR, BI, kce, alr, ali,
                                        mre, nre, Val(_CMR), Val(NR), Val(SA), Val(SB), Val(false), Val(A1)
                                    )
                                else
                                    _microkernel_cmplx_tri!(
                                        Cblk, ldc, AR, AI, BR, BI, kce, alr, ali,
                                        mre, nre, c0 - r0, up, Val(_CMR), Val(NR), Val(SA), Val(SB), Val(false), Val(A1)
                                    )
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
function _trgemm_cmplx_packed_u!(
        ::Val{SA}, ::Val{SB}, ::Val{A1}, up::Bool,
        alr::T, ali::T, X, tXp::Bool, Y, tYp::Bool, C, k::Int
    ) where {SA, SB, A1, T}
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
                                    _microkernel_cmplx!(
                                        Cblk, ldc, AR, AI, BR, BI, kce, alr, ali,
                                        Val(_CMR), Val(W), Val(SA), Val(SB), Val(false), Val(A1)
                                    )
                                elseif full
                                    _microkernel_cmplx_masked!(
                                        Cblk, ldc, AR, AI, BR, BI, kce, alr, ali,
                                        mre, nre, Val(_CMR), Val(W), Val(SA), Val(SB), Val(false), Val(A1)
                                    )
                                else
                                    _microkernel_cmplx_tri!(
                                        Cblk, ldc, AR, AI, BR, BI, kce, alr, ali,
                                        mre, nre, c0 - r0, up, Val(_CMR), Val(W), Val(SA), Val(SB), Val(false), Val(A1)
                                    )
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
function _trgemm_cmplx_packed2_u!(
        ::Val{SA}, ::Val{SB}, up::Bool, alr::T, ali::T,
        X, tXp::Bool, Y, tYp::Bool, C, k::Int
    ) where {SA, SB, T}
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
                                    _microkernel2_cmplx!(
                                        Cblk, ldc, P1AR, P1AI, P1BR, P1BI,
                                        P2AR, P2AI, P2BR, P2BI, kce, Val(_CMR), Val(W), Val(SA), Val(SB)
                                    )
                                elseif full
                                    _microkernel2_cmplx_masked!(
                                        Cblk, ldc, P1AR, P1AI, P1BR, P1BI,
                                        P2AR, P2AI, P2BR, P2BI, kce, mre, nre, Val(_CMR), Val(W), Val(SA), Val(SB)
                                    )
                                else
                                    _microkernel2_cmplx_tri!(
                                        Cblk, ldc, P1AR, P1AI, P1BR, P1BI,
                                        P2AR, P2AI, P2BR, P2BI, kce, mre, nre, c0 - r0, up,
                                        Val(_CMR), Val(W), Val(SA), Val(SB)
                                    )
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
    return if !herm
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
    @inbounds for j in 1:b, i in (up ? (1:j) : (j:b))
        C[i, j] += S[i, j]
    end
    herm && @inbounds for i in 1:b
        C[i, i] = real(C[i, i])
    end
    return C
end
# Small-n unified single-pack cutoff. On AVX2 the multi-pack double-packs A (both operands) — its pack
# traffic dominates in cold cache at small n (measured: n=32 multi 0.90 vs unified 1.19). The unified
# single-pack halves that traffic and wins for small n, but is latency-starved (W=4 accs) at larger n
# (n=128 unified 0.73 vs multi 1.02) — so cap it low. AVX-512 uses unified everywhere (_unified_ok).
# PDM: Derived — formula over detected consts: `_vwidth(Float64) == 4 ? 48 : 0`
const _SYRK_UNIFIED_MAX = @load_preference("syrk_unified_max", _vwidth(Float64) == 4 ? 48 : 0)::Int
# Forceable: the cap sits at 48 and the two measurements bracketing it are n=32 (unified 1.19 vs multi
# 0.90) and n=128 (unified 0.73 vs multi 1.02). n=50 — a red cell — falls in the untested gap just above
# the cap, so it has only ever run multi-pack. One A/B settles whether 48 is the right place to cut.
@inline _fh_syrk_unified_max() = (f = _FKR_syrk_unified_max[]; f >= 0 ? f : _SYRK_UNIFIED_MAX)
# Single-product triangular multi-pack row-tile MR. On AVX2 (W=4) the 12-acc gemm tile (MR=_MR=3) zero-pads
# the remainder row-panel at n not divisible by 12 → small/mid-n syrk/syr2k below gate (n=64 0.81, 128 0.94,
# 256 0.92 measured Zen3). MR=2 (mr=2W=8) divides those sizes AND keeps ample ILP (8 accs) for the
# single-product tri kernel → gates the whole AVX2 range (MR2 ≥ MR3 at every n=64..2048, exact correctness).
# Width-conditional: only F64/AVX2 (W=4); F32/AVX2 and all of AVX-512 keep _MR. Knob "syrk_mr".
# PDM: Literal — AVX2-ONLY by construction: `_tri_mr(T) = _vwidth(T)==4 ? _SYRK_MR : _MR`, so AVX-512 uses gemm's derived _MR. Zen3-only evidence is COMPLETE, not a gap. | tune: n/a off AVX2
const _SYRK_MR = @load_preference("syrk_mr", 2)::Int
@inline _tri_mr(::Type{T}) where {T} = _vwidth(T) == 4 ? _SYRK_MR : _MR
# syrk = one triangular-C gemm (Y = X = A). syr2k = two (A·Bᴴ + B·Aᴴ); real ⇒ both use α.
@inline _syrk_packed!(up::Bool, tr::Bool, α::T, A, C, k::Int) where {T <: BlasReal} =
    (_unified_ok(T) || (_unified_layout_ok(T) && size(C, 1) <= _fh_syrk_unified_max())) ?
    _trgemm_packed_u!(up, α, A, tr, C, k) :
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
    herm && @inbounds for i in 1:n
        C[i, i] = real(C[i, i])
    end
    return C
end
@inline _csyrk_conj(::Val{A1}, nr::Val, up::Bool, tr::Bool, herm::Bool, alr::T, ali::T, A, C, k::Int) where {A1, T} =
    _ctrgemm_prod!(Val(A1), nr, up, tr, herm, alr, ali, A, A, C, k)   # syrk/herk: X = Y = A
# n at/below which the unified single-pack tri kernel (NR=W) beats the multi-pack NR=6 path. AVX2 only
# (the layouts coincide at CMR=1 ⇒ mr==nr==W; AVX-512 already gates 1.02-1.23, leave it). Knob per box.
# Cap at 512: unified wins n≤512 (128 0.80→0.99), but its full-n pack loses cache reuse vs the mc/nc-
# blocked multi path at n≥1024 (1024 0.937 vs multi 0.948) — hand large-n back to multi. AVX-512 → 0.
# PDM: Derived — formula over detected consts: `_vwidth(Float64) == 4 ? 512 : 0`
const _CSYRK_UNIFIED_MAX = @load_preference("csyrk_unified_max", _vwidth(Float64) == 4 ? 512 : 0)::Int
# n at/above which the complex rank-k product uses Karatsuba-3M (3 REAL tri-output products on split re/im).
# 3M runs the gating real `_trgemm_packed!` machinery at 25% fewer flops; windowed to
# [_CSYRK_3M_MIN, _CGEMM_3M_MAX] with k ≥ _CGEMM_3M_KMIN.
#
# THIS IS A LITERAL, AND IT IS LABELLED ONE — read the history before changing it.
# It was 256, set on AVX2 with no AVX-512 data. Lowering it alone was a DEAD KNOB: for trans='N' the
# unpacked branch is tested first, and with `_CSYRK_UNPACK_MAX`=192 on W=8 the 3M route was unreachable
# below 192 (that shadowing is now removed — see `_ctrk_3m_ok`).
# The honest attempt to DERIVE it failed, twice, and the record matters more than the number:
#   * The crossover carries Δrate = 1/R_complex − 3/(4·R_real), a ratio of OUR OWN two microkernels
#     (complex is latency-bound on AVX2, throughput-bound on AVX-512). No cache/ISA constant predicts it.
#   * Measured crossovers moved OPPOSITE to every throughput model (Zen4 lower than Zen3, W=8 vs W=4).
#     `768/W` fit both points with no physical criterion — req#8b calls that a violation, not a Derive.
# FUSION CHANGED THE SITUATION but did NOT remove the threshold, and the reason is structural: fusing
# the split into the pack and the combine into the tile write-back removes both O(n²) PASSES, but the
# PACK ITSELF is O(nk) and 3M packs THREE panels where the complex baseline packs two. That +50% is
# O(nk) against O(n²k) of product, so overhead-per-flop still falls as 1/n. Three real products
# structurally require three real panels; the term cannot be removed.
# WHAT FUSION DID BUY: the crossover fell from ≈160 to ≈112 on Zen3 and below 64 on Zen4, so ONE value
# now serves both boxes — which was previously impossible (128 would have cost Zen3's zherk@128 8.2%).
# Measured fused/packed, freq-locked, k=n (bench/probes/sk13_fused_arm.jl):
#     n            64      96      128     192
#     Zen4 zsyrk  0.930   0.902   0.886   0.860      Zen3 zsyrk  1.099  1.018  0.956  0.903
#     Zen4 zherk  0.931   0.908   0.887   0.854      Zen3 zherk  1.106  1.021  0.963  0.909
# 128 is the smallest ladder point where BOTH boxes win. Zen4 forgoes its n=64/96 gains, which are not
# gate sizes. Re-measure both boxes before moving it; a value that helps one box and hurts the other is
# exactly what this constant existed to avoid.
# PDM: Literal — lower bound for the 3M path; below it the Karatsuba overhead dominates. | tune: candidate
const _CSYRK_3M_MIN = @load_preference("csyrk_3m_min", 128)::Int
@inline _fh_csyrk_3m_min() = (f = _FKR_csyrk_3m_min[]; f >= 0 ? f : _CSYRK_3M_MIN)
# Does the rank-k 3M window apply? Hoisted so the unpacked branch can YIELD to it — without this the
# unpacked path shadows 3M entirely for trans='N' at n ≤ _CSYRK_UNPACK_MAX (192 on AVX-512).
@inline _ctrk_3m_ok(n::Int, k::Int) =
    _CGEMM_3M && _fh_csyrk_3m_min() <= n <= _fh_cgemm_3m_max() && k >= _fh_cgemm_3m_kmin()

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
        w(i, r, c) = PtrMatrix(pointer(t[i]), r, c, r)   # isbits — see the note in `_gemm_3m!` (gemm.jl)
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
# ── FUSED Karatsuba-3M rank-k: split INTO the pack, combine INTO the tile write-back ───────────────
# WHY THIS EXISTS: `_ctrgemm_3m!` above is correct but carries TWO O(n²)-class passes that the packed
# complex baseline does not — `_split3!` (read 2nk, write 3nk, which `_trgemm_packed!` then re-reads)
# and three full n×n P arrays that are RMW'd once per kc-slice and read again by `_combine3_tri!`.
# Those passes are the ENTIRE reason a size crossover exists: profitability is
#     flop saving O(n²k)   vs   pass overhead O(n²),
# so the ratio depends on n and a threshold appears. Locating that threshold turned out to be
# underivable — it carries Δrate = 1/R_complex − 3/(4·R_real), a ratio of OUR OWN two microkernels'
# rates (the complex kernel is latency-bound on AVX2 and throughput-bound on AVX-512), which no
# cache/ISA constant predicts. Measured 3m/packed crossovers: ≈64-96 on Zen4 (W=8), ≈130-190 on Zen3
# (W=4) — opposite in sign to every throughput model, and `768/W` fits both points with no criterion,
# which req#8b classifies as a violation rather than a derivation.
# So REMOVE the term instead of predicting it. With both passes fused there is no O(n²) cost left to
# form a ratio against O(n²k), and profitability becomes n-INDEPENDENT — the same on/off question
# `_CGEMM_3M` already answers. That deletes `_CSYRK_3M_MIN` as a size knob for a structural reason.
#
# HOW THE SPLIT FUSES FOR FREE: `_pack_A_cmplx!`/`_pack_B_cmplx!` (gemm.jl) ALREADY deinterleave complex
# input into two REAL panels in one pass, in the same panel layout the real microkernel indexes. The
# third Karatsuba panel is S = R ± I, so it is derived from the PACKED panels — contiguous and
# cache-resident — by `_pack3_finish!` below, rather than by a second strided pass over the operand.
# That also means the existing SIMD pack kernels are reused untouched. The conj (herk's ᴴ) folds in
# here exactly as `_split3!`'s flag did: negate the I panel, then S = R + I.
#
# HOW THE COMBINE FUSES: `_combine3_tri!` is POINTWISE-LINEAR in P1/P2/P3, so it distributes over both
# kc-slices and tiles. The three n×n P arrays collapse to one mr×nr tile scratch; `_microkernel!` takes
# a raw pointer and arbitrary ldc, so that scratch is a legal target. C is then touched exactly once per
# tile per kc-slice — strictly LESS traffic than the P-array RMW plus the final combine pass.
# The triangle restriction moves OUT of the microkernel store (no `_microkernel_tri!`) and INTO the
# combine bounds, so straddling tiles compute a full mre×nre product and mask only when writing C.
#
# SAFETY FLOOR: identical kernels to `_ctrgemm_3m!` with strictly less memory traffic, so it cannot be
# slower than the unfused 3M at any n. "Wins at every n" is a STRUCTURAL argument, not a measurement —
# the tile round-trip costs ~2/kce extra L1 ops per FMA (n-independent, ≤12.5% at `_CGEMM_3M_KMIN`=16,
# <1% at kc=256) and the unpacked-vs-fused boundary on AVX-512 is a separate A/B.
@inline function _pack3_finish!(pR::Ptr{T}, pI::Ptr{T}, pS::Ptr{T}, len::Int, cj::Bool) where {T}
    if cj                                   # conjugated operand: I := -I, then S = R + I
        @inbounds @simd ivdep for i in 1:len
            r = unsafe_load(pR, i); v = -unsafe_load(pI, i)
            unsafe_store!(pI, v, i); unsafe_store!(pS, r + v, i)
        end
    else
        @inbounds @simd ivdep for i in 1:len
            unsafe_store!(pS, unsafe_load(pR, i) + unsafe_load(pI, i), i)
        end
    end
    return
end
# Combine one mre×nre tile of the three real products into complex C, masked to the `up` triangle.
# C is read as interleaved Tr pairs: element (r,c) sits at ((c)*ldc + r)*2 (0-based) in Tr units.
@inline function _combine3_tile!(
        Cp0::Ptr{Tr}, ldc::Int, p1::Ptr{Tr}, p2::Ptr{Tr}, p3::Ptr{Tr}, ldt::Int,
        mre::Int, nre::Int, r0::Int, c0::Int, up::Bool, ar::Tr, ai::Tr
    ) where {Tr}
    d = c0 - r0
    @inbounds for j in 0:(nre - 1)
        lo = up ? 0 : max(0, d + j)          # up: keep row ≤ col ⇒ i ≤ d+j;  lo: keep row ≥ col ⇒ i ≥ d+j
        hi = up ? min(mre - 1, d + j) : mre - 1
        lo > hi && continue
        cb = (r0 + (c0 + j) * ldc) * 2; tb = j * ldt
        @simd for i in lo:hi
            a = unsafe_load(p1, tb + i + 1); b = unsafe_load(p2, tb + i + 1)
            zr = a - b; zi = unsafe_load(p3, tb + i + 1) - a - b
            or = unsafe_load(Cp0, cb + 2i + 1); oi = unsafe_load(Cp0, cb + 2i + 2)
            unsafe_store!(Cp0, or + ar * zr - ai * zi, cb + 2i + 1)
            unsafe_store!(Cp0, oi + ar * zi + ai * zr, cb + 2i + 2)
        end
    end
    return
end
function _ctrgemm_3m_fused!(
        ::Val{MR}, ::Val{NR}, up::Bool, conjX::Bool, conjY::Bool,
        tXp::Bool, tYp::Bool, α::Tc, X, Y, C, k::Int
    ) where {Tc, MR, NR}
    Tr = real(Tc); n = size(C, 1); W = _vwidth(Tr); mr = MR * W; nr = NR
    kc = min(_KC, k); mc = _at_mc_kc(_HW, Tr, kc, mr, cld(n, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(n, nr) * nr)
    lenA = cld(mc, mr) * mr * kc; lenB = cld(nc, nr) * nr * kc
    t = _gemm_3m_scratch(Tr, lenA, lenB, mr * nr)   # 1-3 A panels, 4-6 B panels, 7-9 tile scratch
    ldc = stride(C, 2); sz = sizeof(Tr); ar = real(α); ai = imag(α)
    GC.@preserve C t begin
        Cp0 = Ptr{Tr}(pointer(C))
        pAR = pointer(t[1]); pAI = pointer(t[2]); pAS = pointer(t[3])
        pBR = pointer(t[4]); pBI = pointer(t[5]); pBS = pointer(t[6])
        pT1 = pointer(t[7]); pT2 = pointer(t[8]); pT3 = pointer(t[9])
        jc = 0
        while jc < n
            nce = min(nc, n - jc); pc = 0
            while pc < k
                kce = min(kc, k - pc)
                _pack_B_cmplx!(t[4], t[5], Y, pc, jc, kce, nce, tYp, nr)
                _pack3_finish!(pBR, pBI, pBS, cld(nce, nr) * nr * kce, conjY)
                ic = 0
                while ic < n
                    mce = min(mc, n - ic)
                    _pack_A_cmplx!(t[1], t[2], X, ic, pc, mce, kce, tXp, mr)
                    _pack3_finish!(pAR, pAI, pAS, cld(mce, mr) * mr * kce, conjX)
                    jr = 0
                    while jr < nce
                        nre = min(nr, nce - jr); ir = 0
                        while ir < mce
                            mre = min(mr, mce - ir); r0 = ic + ir; c0 = jc + jr
                            skip = up ? (r0 > c0 + nre - 1) : (r0 + mre - 1 < c0)
                            if !skip
                                ao = (div(ir, mr) * mr * kce) * sz; bo = (div(jr, nr) * nr * kce) * sz
                                if mre == mr && nre == nr        # triangle masking happens in the combine
                                    _microkernel!(pT1, mr, pAR + ao, pBR + bo, kce, Val(MR), Val(NR), Val(true))
                                    _microkernel!(pT2, mr, pAI + ao, pBI + bo, kce, Val(MR), Val(NR), Val(true))
                                    _microkernel!(pT3, mr, pAS + ao, pBS + bo, kce, Val(MR), Val(NR), Val(true))
                                else
                                    _microkernel_masked!(pT1, mr, pAR + ao, pBR + bo, kce, mre, nre, Val(MR), Val(NR), Val(true))
                                    _microkernel_masked!(pT2, mr, pAI + ao, pBI + bo, kce, mre, nre, Val(MR), Val(NR), Val(true))
                                    _microkernel_masked!(pT3, mr, pAS + ao, pBS + bo, kce, mre, nre, Val(MR), Val(NR), Val(true))
                                end
                                _combine3_tile!(Cp0, ldc, pT1, pT2, pT3, mr, mre, nre, r0, c0, up, ar, ai)
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
# ONE triangular-C complex product C[tri] += α·op(X)·op(Y)ᴴ (skip/full/tri tiles). herm conjugates the
# ᴴ operand (tr='N' → Y via SB=-1; tr='C' → X via SA=-1); syrk conjugates neither. syrk/herk pass X=Y=A;
# syr2k/her2k call twice (A,B then B,A). Conj signs mirror _syrk_gemm!'s conjA=tr&&cc, conjB=!tr&&cc.
# X===Y (herk/zsyrk) on AVX2 mid-n → unified single-pack driver (NR=W, half the pack, no NR=6 spill).
@inline function _ctrgemm_prod!(
        ::Val{A1}, ::Val{NR}, up::Bool, tr::Bool, herm::Bool,
        alr::T, ali::T, X, Y, C, k::Int
    ) where {A1, NR, T}
    n = size(C, 1)
    # `_vwidth(T) == 4` REMOVED — see the derivation at `_CGEMM_3M` (gemm.jl): the 25% flop cut is
    # algebraic, so nothing physical made this width-dependent; the test was an artefact of only ever
    # having measured AVX2. Measured on Zen4 (W=8), 3m/base, bench/probes/sk1_3m_avx512.jl:
    #     n         128*    256    512    1024   2048
    #     zsyrk     1.000   0.842  0.874  0.832  0.811
    #     zherk     1.000   0.869  0.945  0.868  0.824
    #     zsyr2k    1.000   0.870  0.927  0.867  0.823
    #     zher2k    1.000   0.870  0.933  0.867  0.824
    # (* = CONTROL below `_CSYRK_3M_MIN`=256; all four tie at exactly 1.000 with relerr 0, proving the
    # arm was live.) `_EXPFLAG[_EXP11]` is INVERTED: 3M ships ON, the flag disables it for A/B.
    if !_EXPFLAG[_EXP11] && _ctrk_3m_ok(n, k)
        # large-n: Karatsuba-3M (the complex tri kernels plateau ~0.92 here). herk conjugates op(X) at
        # tr='C' (SA=-1) / op(Y) at tr='N' (SB=-1); syrk conjugates neither. tXp=tr, tYp=!tr.
        # The FUSED driver SHIPS (split folded into the pack, combine folded into the tile write-back);
        # `_EXPFLAG[_EXP16]` is INVERTED and selects the old unfused `_ctrgemm_3m!`, kept A/B-able
        # in-process because Zen5 has not been measured. Fused is never slower than unfused — same
        # kernels, strictly less traffic — measured fused/unfused 0.83-1.00 across both boxes.
        Tr = real(T)
        return _EXPFLAG[_EXP16] ?
            _ctrgemm_3m!(up, herm && tr, herm && !tr, tr, !tr, Complex(alr, ali), X, Y, C, k) :
            _ctrgemm_3m_fused!(Val(_tri_mr(Tr)), Val(_NR), up, herm && tr, herm && !tr, tr, !tr,
                Complex(alr, ali), X, Y, C, k)
    end
    if X === Y && _vwidth(T) == 4 && n <= _CSYRK_UNIFIED_MAX   # herk/zsyrk: single-pack win
        return _ctrgemm_prod_u!(Val(A1), up, tr, herm, alr, ali, X, Y, C, k)
    end
    return if !herm
        tr ? _trgemm_cmplx_packed!(Val(1), Val(1), Val(NR), Val(A1), up, alr, ali, X, true, Y, false, C, k) :
            _trgemm_cmplx_packed!(Val(1), Val(1), Val(NR), Val(A1), up, alr, ali, X, false, Y, true, C, k)
    elseif tr
        _trgemm_cmplx_packed!(Val(-1), Val(1), Val(NR), Val(A1), up, alr, ali, X, true, Y, false, C, k)
    else
        _trgemm_cmplx_packed!(Val(1), Val(-1), Val(NR), Val(A1), up, alr, ali, X, false, Y, true, C, k)
    end
end
@inline function _ctrgemm_prod_u!(
        ::Val{A1}, up::Bool, tr::Bool, herm::Bool,
        alr::T, ali::T, X, Y, C, k::Int
    ) where {A1, T}
    return if !herm
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
# PDM: Derived — formula over detected consts: `_vwidth(Float64) == 4 ? 192 : 0`
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
    herm && @inbounds for i in 1:n
        C[i, i] = real(C[i, i])
    end
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
function _trgemm_packed2!(
        up::Bool, α::T, X1, tX1::Bool, Y1, tY1::Bool,
        X2, tX2::Bool, Y2, tY2::Bool, C, k::Int, ::Val{MRV} = Val(_MR),
        ::Val{NRV} = Val(_NR), ::Val{OV} = Val(false)
    ) where {T <: BlasReal, MRV, NRV, OV}
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
# SPLIT INTO ITS TWO INDEPENDENT CLAUSES (2026-08-16) — they are not the same KIND of condition:
#
#   _unified_layout_ok  — a CORRECTNESS PRECONDITION of `_trgemm_packed_u!`. That kernel packs A once
#                         into `mr = W`-row panels and then indexes the SAME buffer two ways,
#                         `div(r0, mr)` for the A-operand and `div(c0, nr)` for the B-operand. The
#                         second derivation is only valid when `nr == mr`, i.e. `_NR == _vwidth(T)`.
#                         Violate it and the B-operand reads a panel that was never packed at that
#                         stride — an out-of-bounds read of the pack buffer.
#   _vwidth(T) >= 8     — a PERFORMANCE condition (ILP): W accumulators must hide FMA latency; on
#                         AVX2's W=4 the unified kernel is latency-starved and we prefer multi-pack.
#
# They were AND-ed into one predicate, and the `_SYRK_UNIFIED_MAX` small-n bypass at `_syrk_packed!`
# then OR-ed PAST the whole thing — waiving the correctness clause to buy the small-n performance the
# other clause was about. Today that is unreachable (`_SYRK_PACK_CUT` = 84 > `_SYRK_UNIFIED_MAX` = 48
# on AVX2; the cap is 0 on AVX-512, and F32/AVX2 is the mismatching case), but BOTH are
# `@load_preference` knobs, so a pinned build reaches it — the same shape as the `_cger_np` OOB write
# fixed in 76ffd06, where a guard sat on the cap instead of on the value every caller resolves through.
@inline _unified_layout_ok(::Type{T}) where {T} = _vwidth(T) == _NR
@inline _unified_ok(::Type{T}) where {T} = _unified_layout_ok(T) && _vwidth(T) >= 8

# The (jr, ir) micro-tile sweep over ONE packed panel of the unified syrk.
#
# Split out of `_trgemm_packed_u!` to flatten it: five nested BLIS loops plus the tile classification
# put that function at depth 8, and the five loops are the algorithm and cannot go. Unlike
# `_trgemm_packed!`, there is NO loop-invariant branch to hoist here — the β/α handling is inside
# `_microkernel_u!` and the three-way classification is already flat — so this is a readability split
# with no expected effect on generated code.
function _trgemm_tiles_u!(
        up::Bool, α::T, PA::Ptr{T}, Cp0::Ptr{T}, ldc::Int, sz::Int, pstr::Int,
        ic::Int, jc::Int, mce::Int, nce::Int, kce::Int
    ) where {T <: BlasReal}
    # DERIVED, not passed. The driver has `mr = W = _vwidth(T)` and `nr = _NR`, both compile-time
    # constants; taking them as `::Int` arguments made `div(r0, mr)` / `div(c0, nr)` below into real
    # integer divisions. Measured: 2 `idiv` when passed, 0 when derived.
    mr = _vwidth(T); nr = _NR
    jr = 0
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
    return nothing
end

# Unified single-pack syrk: pack A ONCE into W-row panels; the A-operand (vector load, panel ir) and
# the B-operand (scalar broadcast, panel jr) both read that one buffer. 8×8 tile (MR=1) so both packs'
# layouts coincide; α applied at the store (shared buffer ⇒ can't fold α into the pack).
function _trgemm_packed_u!(up::Bool, α::T, A, tAp::Bool, C, k::Int) where {T <: BlasReal}
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
                    mce = min(mc, n - ic)
                    _trgemm_tiles_u!(up, α, PA, Cp0, ldc, sz, pstr, ic, jc, mce, nce, kce)
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
function _trgemm_packed2_u!(up::Bool, α::T, A, tAp::Bool, Bm, tBp::Bool, C, k::Int) where {T <: BlasReal}
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
# PDM: Derived — formula over detected consts: `_vwidth(Float64) == 4 ? 2 : _MR`
const _SYR2K_MR = @load_preference("syr2k_mr", _vwidth(Float64) == 4 ? 2 : _MR)::Int
# nr for the two-product tile (Preferences knob). Default _NR: widening to NR=5 with MR=2 was measured
# NEUTRAL-to-worse on Zen3 (n=256 unchanged, n=1024 0.985→0.96) — the tile wasn't ILP-starved, so keep _NR.
# PDM: Literal — drives its own microkernel, borrows gemm's _NR as a prior; unvalidated here. | tune: candidate
const _SYR2K_NR = @load_preference("syr2k_nr", _NR)::Int
# n above which syr2k does TWO full-kernel passes (OpenBLAS-style) instead of the fused two-product tile.
# On AVX2 the fused MR=2 tile has only 8 accumulators (ILP-starved on 16 regs); two _trgemm_packed! passes
# (12 accs each) win at n>128 despite 2× C traffic (measured: n=512 0.94→1.00, n=1024 0.95→1.02). AVX-512
# keeps the fused unified path (32 regs, not starved). Overridable "syr2k_2pass".
# PDM: Literal — AVX2-ONLY by construction: the default is typemax(Int) on AVX-512, which disables the branch. Zen3-only evidence is COMPLETE. | tune: n/a off AVX2
const _SYR2K_2PASS = @load_preference("syr2k_2pass", _vwidth(Float64) == 4 ? 128 : typemax(Int))::Int
# Handles β internally: the two-pass path can OVERWRITE C on its first pass when β=0 (skipping the
# separate scaleC zero-pass — measured the whole n=256 gate gap, since scaleC + 2 adds is 3 C-touches at
# the L2-resonant size). The fused/unified paths ADD, so they need C β-pre-scaled (zeroed if β=0).
@inline function _syr2k_packed!(up::Bool, tr::Bool, α::T, β::T, A, Bm, C, k::Int) where {T <: BlasReal}
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
# PDM: Literal — diagonal-block base; larger pushes work into efficient off-diagonal gemms. | tune: candidate
const _SYRK_DBASE = @load_preference("syrk_dbase", 32)::Int
@inline _fh_syrk_dbase() = (f = _FKR_syrk_dbase[]; f >= 0 ? f : _SYRK_DBASE)
# n above which the single-pass packed syrk beats the gemm→temp recursion (the recursion base's 2×-flop
# diagonal waste + split overhead is why rank-k packs slightly EARLIER than gemm). DERIVED (req#8) via
# `_at_rank_k_pack_cut`, which is PATH-DEPENDENT — read cpuinfo.jl:212-227 for the authoritative form:
#   AVX2 multi-pack `_trgemm_packed!`  -> (7·(nvreg−4)·W)/4, reproduces Zen3 84 (measured: recursion wins
#     n≈40–80, packed decisive n≥96; the old literal 23 mis-routed n=48/80 and caused the Zen3 AOCL misses)
#   AVX-512 unified single-pack `_trgemm_packed_u!` -> W, because half the pack traffic makes packed win
#     from ≈W up.
# An earlier version of this comment predicted "Zen4/Zen5 392" from the ×7/4 form. That is FALSIFIED and no
# longer what the code computes: 392 mis-routed all n≤256 to the recursion and WAS the syrk n=128 gate miss
# (Zen5 0.88 / Zen4 0.91); packed is +14% there. Fleet-validated AVX-512 -> W. Overridable per machine.
# (OpenBLAS-style dense-scratch + scalar triangular copyback for the diagonal tile was A/B-tested here
# and measured EQUAL to the masked-store _microkernel_tri! on AVX2 — no gain, not adopted.)
# SPLIT FROM syr2k 2026-08-28. This used to be `_at_rank_k_pack_cut` — shared with syr2k on the stated
# grounds that both are the same register-capacity criterion. Measured on Zen3, they have OPPOSITE
# crossovers on the multi-pack path (syrk wants packed from n=32, syr2k wants recursion through n=50),
# so the shared 84 was costing syrk 5-15%. The full A/B table and the operand-count mechanism are at
# `_at_syrk_pack_cut` in cpuinfo.jl. The AVX-512 arm is unchanged (both ops still resolve to W there).
# PDM: Derived — formula over detected consts: `_at_syrk_pack_cut(_HW)`
const _SYRK_PACK_CUT = @load_preference("syrk_pack_cut", _at_syrk_pack_cut(_HW))::Int
@inline _fh_syrk_pack_cut() = (f = _FKR_syrk_pack_cut[]; f >= 0 ? f : _SYRK_PACK_CUT)
# n above which complex syrk/herk take the single-pass packed triangular path (no 2×-flop diagonal waste,
# no recursion — vs the wasteful _syrk_rec! below). TRANS-DEPENDENT crossover (measured, Zen4/Zen5):
# trans='N' recursion base packs A's contiguous columns via the fast SIMD deinterleave → it WINS small-n
# (n=8 gates 1.2× on AVX-512), packed wins n≥~24. trans='C'/'T' needs a transposed A-pack → recursion is
# slow at every small n while the packed path amortizes it, so packed wins uniformly (route it from ~n=4).
# Complex micro-tile is _CMR·W complex rows (AVX2 z: 4, AVX-512 z: 16). Per-machine Preferences knobs.
# PDM: Literal — trans='N': recurse below this rather than pack. | tune: candidate
const _CSYRK_PACK_CUT = @load_preference("csyrk_pack_cut", 16)::Int        # trans='N': recursion below this
@inline _fh_csyrk_pack_cut() = (f = _FKR_csyrk_pack_cut[]; f >= 0 ? f : _CSYRK_PACK_CUT)
# PDM: Literal — trans='C'/'T': packing wins almost always, hence the much lower cut. | tune: candidate
const _CSYRK_PACK_CUT_T = @load_preference("csyrk_pack_cut_t", 4)::Int     # trans='C'/'T': packed ~always
@inline _fh_csyrk_pack_cut_t() = (f = _FKR_csyrk_pack_cut_t[]; f >= 0 ? f : _CSYRK_PACK_CUT_T)
# trans='N' complex n≤this ⇒ unpacked triangular kernel (`_ctri_unpacked!`): the packed path's operand-pack
# + NR-remainder overhead and the recursion base's 2×-flop waste BOTH miss the gate at small n, while the
# unpacked complex microkernel (what zgemm rides) gates there. The cutoff is where the packed path's NR-tile
# amortization overtakes unpacked — a microkernel-ramp crossover, µarch-specific (NOT a cache formula), so
# it is `_vwidth`-keyed & Preferences-overridable. Measured boost-locked (bench/csyrk_avx2_calib.jl):
#  • AVX2 (W=4, Zen3): the recursion base (n≤_CSYRK_PACK_CUT=16) 2×-wastes → zherk/zsyrk n=16 DIP to 0.87-
#    0.91 (sub-gate). Unpacked-tri gates all 4 ops at n≤16 (herk 1.49/her2k 1.21) and beats the dip; packed
#    overtakes by n=24, so cutoff=16.  • AVX-512 (W=8): packed's edge overhead is larger → unpacked wins
#    broadly. Measured both boxes boost-locked: Zen4 (wm) unpacked ≥ packed to n=192 (packed reclaims n=256);
#    Zen5 (neuro) unpacked ≥ packed at EVERY n≤256. Cutoff 192 is safe on both (avoids Zen4's n=256 packed
#    preference) and lifts the n=128/192 complex rank-k the factorizations recurse through. (One formula for
#    both µarchs remains req#8 debt.)
# PDM: Derived — formula over detected consts: `_vwidth(Float64) == 4 ? 16 : 192`
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
    herm && @inbounds for i in 1:n
        C[i, i] = real(C[i, i])
    end        # Hermitian diagonal is real
    return C
end
# rank-2k: C[tri] += α·A·Bᴴ + α₂·B·Aᴴ (α₂ = conj(α) her2k, α syr2k). Two unpacked-tri products, one each.
@inline function _ctri2_unpacked!(up::Bool, herm::Bool, α, A, B, C, k::Int)
    Tc = eltype(C); a = convert(Tc, α); a2 = herm ? conj(a) : a; n = size(C, 1)
    k == 0 && return C
    _ctri_sb!(up, herm, real(a), imag(a), A, B, C, k, n)              # α·A·Bᴴ
    _ctri_sb!(up, herm, real(a2), imag(a2), B, A, C, k, n)            # α₂·B·Aᴴ
    herm && @inbounds for i in 1:n
        C[i, i] = real(C[i, i])
    end
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
function _ctri_core!(
        ::Val{NR}, up::Bool, ::Val{SB}, ::Val{A1}, ::Val{AR},
        alr::T, ali::T, X, Y, C, k::Int, n::Int
    ) where {NR, SB, A1, AR, T}
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
                full = up ? (ir + mre - 1 <= jr) : (ir >= jr + nre - 1) # tile entirely inside it
                if below
                    # skip: nothing stored
                elseif full && mre == mr
                    _uker_cmplx!(
                        Cp, ldc, Xp, ldx, ir, Yp, ldy, jr, k, alr, ali, mre, nre,
                        Val(_CMR), Val(NR), Val(true), Val(1), Val(SB), Val(false), Val(A1), Val(AR), Val(true), Val(false), 0, true
                    )
                elseif full && nrv >= _CMR
                    _uker_cmplx!(
                        Cp, ldc, Xp, ldx, ir, Yp, ldy, jr, k, alr, ali, mre, nre,
                        Val(_CMR), Val(NR), Val(true), Val(1), Val(SB), Val(false), Val(A1), Val(AR), Val(false), Val(false), 0, true
                    )
                elseif full
                    _uker_cmplx!(
                        Cp, ldc, Xp, ldx, ir, Yp, ldy, jr, k, alr, ali, mre, nre,
                        Val(1), Val(NR), Val(true), Val(1), Val(SB), Val(false), Val(A1), Val(AR), Val(false), Val(false), 0, true
                    )
                elseif nrv >= _CMR                                       # diagonal-straddling ⇒ TRI masked store
                    _uker_cmplx!(
                        Cp, ldc, Xp, ldx, ir, Yp, ldy, jr, k, alr, ali, mre, nre,
                        Val(_CMR), Val(NR), Val(true), Val(1), Val(SB), Val(false), Val(A1), Val(AR),
                        Val(false), Val(true), jr - ir, up
                    )
                else
                    _uker_cmplx!(
                        Cp, ldc, Xp, ldx, ir, Yp, ldy, jr, k, alr, ali, mre, nre,
                        Val(1), Val(NR), Val(true), Val(1), Val(SB), Val(false), Val(A1), Val(AR),
                        Val(false), Val(true), jr - ir, up
                    )
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
    if !herm && T <: BlasReal && size(C, 1) > _fh_syrk_pack_cut() && k > 0
        return _syrk_packed!(up, tr, convert(T, α), A, C, k)
    elseif T <: Union{ComplexF64, ComplexF32} && k > 0
        n = size(C, 1)
        # The unpacked branch YIELDS to the 3M window. It is tested before the packed path, and 3M lives
        # inside `_csyrk_packed!` → `_ctrgemm_prod!`, so without this `_CSYRK_UNPACK_MAX`=192 shadowed
        # 3M entirely below 192 on AVX-512 and lowering `_CSYRK_3M_MIN` was a dead knob.
        # `_strided1(A)` is REQUIRED, not an optimisation: `_ctri_unpacked!` reads the operand
        # DIRECTLY (`_ctri_core!` takes `stride(X,2)` and `pointer(X)`), so an operand that is not
        # pointer-able throws from inside the kernel. A complex `A'` is exactly that -- `Adjoint`
        # conjugates, so it is not a strided view and Base defines no `strides` for it, and
        # `herk!(C, A')` died with `MethodError(strides)` for every n <= 64 while n >= 128 silently
        # escaped to the packed path below. The packed and recursive paths both index generically,
        # so falling through is correct for any such operand.
        if !tr && _strided1(A) && n <= _CSYRK_UNPACK_MAX && !_ctrk_3m_ok(n, k)
            return _ctri_unpacked!(up, herm, α, A, C, k)
        elseif n > (tr ? _fh_csyrk_pack_cut_t() : _fh_csyrk_pack_cut())
            return _csyrk_packed!(up, tr, herm, α, A, C, k)
        end
    end
    return _syrk_rec!(up, tr, herm, α, A, C, k, _l3_tmp(eltype(C)), 0, size(C, 1))
end
# One gemm sub-block through the @inline `_gemm_core!` (not the non-inlined kwarg gemm!): tr=false ⇒
# C += α·A·Bᵀ (transB), tr=true ⇒ C += α·Aᵀ·B (transA); cc conjugates for the herm (herk) case.
@inline function _syrk_gemm!(C, A, B, α::T, β::T, tr::Bool, cc::Bool) where {T}
    return _gemm_core!(C, A, B, α, β, tr, !tr, tr && cc, !tr && cc)
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
    if n <= _fh_syrk_dbase()
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

_syrk_dims(C, A, trans) = (
    n = size(C, 1); size(C, 2) == n ||
        throw(DimensionMismatch("syrk!: C must be square")); k = trans == 'N' ? size(A, 2) : size(A, 1);
    (trans == 'N' ? size(A, 1) : size(A, 2)) == n || throw(DimensionMismatch("syrk!: op(A) rows ≠ n")); (n, k)
)

function syrk!(
        C::AbstractMatrix, A::AbstractMatrix; uplo::Char = 'U', trans::Char = 'N',
        alpha::Number = true, beta::Number = false
    )
    # syrk is SYMMETRIC: its vocabulary is 'N'/'T', a plain transpose. So `transpose(A)` folds, and so
    # does a REAL `A'` (adjoint == transpose there) -- but a COMPLEX `A'` means the conjugate transpose,
    # which 'T' does not express, so it must stay wrapped and take the generic path. `_lazyop` returns
    # 'C' in exactly that case, so testing for 'T' is the whole guard.
    if trans == 'N' && _lazyop(A) == 'T'
        return syrk!(C, parent(A); uplo, trans = 'T', alpha, beta)
    end
    n, k = _syrk_dims(C, A, trans); up = uplo == 'U'
    _syrk_scaleC!(C, up, beta)
    _syrk_blocked!(up, trans != 'N', false, alpha, A, C, k)
    return C
end
function herk!(
        C::AbstractMatrix, A::AbstractMatrix; uplo::Char = 'U', trans::Char = 'N',
        alpha::Real = true, beta::Real = false
    )
    # herk is HERMITIAN: its vocabulary is 'N'/'C', the conjugate transpose. The mirror of syrk above --
    # `A'` folds at either eltype, while a COMPLEX `transpose(A)` does not (it means 'T', which herk
    # does not express) and stays wrapped.
    if trans == 'N' && _lazyop(A) == 'C'
        return herk!(C, parent(A); uplo, trans = 'C', alpha, beta)
    end
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
    return _symstored(up, i, l) ? A[i, l] : (herm ? conj(A[l, i]) : A[l, i])
end
# symm's output C is a FULL matrix (no triangle), so symm = gemm with a materialized full symmetric
# A — correct flops (NO 2× waste, unlike syrk). Materialize the symmetric/Hermitian A into a dense
# scratch (O(n²), amortized over the O(n²·m) gemm), then one gemm! carries α and β directly.
const _SYMM_SCR = IdDict{DataType, Matrix}()
function _symm_scr(::Type{T}, n::Int) where {T}
    m = get(_SYMM_SCR, T, nothing)
    if isnothing(m) || size(m, 1) < n
        m = Matrix{T}(undef, n, n); _SYMM_SCR[T] = m
    end
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
# Tile edge for the symmetric/Hermitian → dense fill. DERIVE tier: the mirror half is a TRANSPOSE, so
# the criterion is that a source tile and its destination tile are both L1-resident while the tile is
# being written — `2·NB²·sizeof(T) ≤ ½·L1` ⇒ `NB = √(L1 / (4·sizeof(T)))`. Clamped to [4, 64]: below 4
# the loop overhead dominates a tile, above 64 the tile stops fitting on any box in the fleet.
@inline _symm_tile(::Type{T}) where {T} = clamp(isqrt(_L1_BYTES ÷ (4 * sizeof(T))), 4, 64)

# Symmetric/Hermitian → dense fill, TILED.
# The mirror half reads A[j,i] while writing Ad[i,j], i.e. a transpose, and a transpose done one column
# at a time touches a new cache line per element: at n=256 complex the row walk has stride n·16 = 4096 B,
# so every element costs its own line AND every line lands in the same L1 set. Measured cost of that:
# `symm/gemm` = 1.065/1.069/1.053/1.026 at n=128/256/512/1024 on Zen4 — ~6% at n=256, which is ~219 µs
# to move 2 MB, about 9 GB/s. That is the whole reason zsymm/zhemm miss AOCL at n≥128 while the zgemm
# they call GATES (1.018 at n=256): the wrapper, not the engine.
# Tiling fixes the reuse: within an NB×NB tile each fetched line of A serves NB values of j instead of
# one. Tiles strictly inside a triangle are uniform, so the hot path keeps the branch-free property the
# previous version was written for — only the diagonal tiles carry a per-element `_symstored` test.
function _symm_materialize!(Ad, up::Bool, herm::Bool, A, n::Int)
    NB = _symm_tile(eltype(Ad))
    # n ≤ NB ⇒ the WHOLE matrix is one diagonal tile, so the tiled loop below would put a per-element
    # `_symstored` test on every element and gain nothing: at that size A is already L1-resident and
    # there is no reuse to recover. Keep the branch-free two-loop form there. This is not hypothetical
    # — shipping the tiled version without this guard cost the tiny-n cells (zhemm@8 0.973→0.924,
    # measured), because the tile edge is 22 for ComplexF64 on a 32 KiB L1 and every gate size below it
    # collapsed to the diagonal case. The tiled path only pays once the row walk actually strides.
    if n <= NB
        # POINTER PATH — the destination is our own `_symm_scr` VIEW, and SubArray indexing on every
        # store is most of this function's cost at tiny n. Measured on Zen3 (ComplexF64, materialise
        # alone, floor-subtracted): writing into `view(scr,1:n,1:n)` vs into a plain `Matrix` costs
        #     n=4 69.5 vs 43.8 (1.59x) · n=8 171.5 vs 100.3 (1.71x) · n=16 1.29x · n=32 1.09x
        # i.e. ~71 ns of pure indexing overhead at n=8 — against a zhemm@8 gate gap of 78 ns. The gemm
        # that follows takes the same view and pays only 2.6 ns for it, so the cost is specific to the
        # per-element stores here, not to SubArrays in general. Raw pointers with the destination's own
        # leading dimension sidestep it and work for any `ld` (the scratch is grown, so ld >= n).
        Td = eltype(Ad)
        if _strided1(Ad) && _strided1(A) && isbitstype(Td)
            GC.@preserve Ad A begin
                dp = pointer(Ad); ap = pointer(A)
                # `Ptr + Int` is a BYTE offset in Julia, so column strides carry `sz`; the 2-arg
                # `unsafe_load(p, i)` / `unsafe_store!(p, v, i)` are 1-based ELEMENT indexed. Same
                # convention as the level2 kernels.
                sz = sizeof(Td)
                ldd = stride(Ad, 2) * sz; lda = stride(A, 2) * sz
                @inbounds for j in 1:n
                    dc = dp + (j - 1) * ldd                 # &Ad[1,j]  (element-indexed below)
                    ac = ap + (j - 1) * lda                 # &A[1,j]
                    if up
                        for i in 1:j                        # stored: contiguous down column j
                            unsafe_store!(dc, unsafe_load(ac, i), i)
                        end
                        for i in (j + 1):n                  # mirrored: A[j,i], a row walk
                            v = unsafe_load(ap + (i - 1) * lda, j)
                            unsafe_store!(dc, herm ? conj(v) : v, i)
                        end
                    else
                        for i in j:n
                            unsafe_store!(dc, unsafe_load(ac, i), i)
                        end
                        for i in 1:(j - 1)
                            v = unsafe_load(ap + (i - 1) * lda, j)
                            unsafe_store!(dc, herm ? conj(v) : v, i)
                        end
                    end
                end
                if herm
                    @inbounds for i in 1:n
                        d = dp + (i - 1) * ldd
                        unsafe_store!(d, Td(real(unsafe_load(d, i))), i)
                    end
                end
            end
            return Ad
        end
        @inbounds if up
            for j in 1:n
                @simd for i in 1:j
                    Ad[i, j] = A[i, j]
                end
                for i in (j + 1):n
                    Ad[i, j] = herm ? conj(A[j, i]) : A[j, i]
                end
            end
        else
            for j in 1:n
                @simd for i in j:n
                    Ad[i, j] = A[i, j]
                end
                for i in 1:(j - 1)
                    Ad[i, j] = herm ? conj(A[j, i]) : A[j, i]
                end
            end
        end
        herm && @inbounds for i in 1:n
            Ad[i, i] = real(Ad[i, i])
        end
        return Ad
    end
    @inbounds for jb in 1:NB:n
        jhi = min(jb + NB - 1, n)
        for ib in 1:NB:n
            ihi = min(ib + NB - 1, n)
            if ib == jb                                   # diagonal tile: straddles both triangles
                for j in jb:jhi, i in ib:ihi
                    Ad[i, j] = _symstored(up, i, j) ? A[i, j] : (herm ? conj(A[j, i]) : A[j, i])
                end
            elseif _symstored(up, ib, jb)                 # wholly stored: contiguous column runs
                for j in jb:jhi
                    @simd for i in ib:ihi
                        Ad[i, j] = A[i, j]
                    end
                end
            else                                          # wholly mirrored: transpose this tile
                for j in jb:jhi
                    @simd for i in ib:ihi
                        Ad[i, j] = herm ? conj(A[j, i]) : A[j, i]
                    end
                end
            end
        end
    end
    herm && @inbounds for i in 1:n
        Ad[i, i] = real(Ad[i, i])
    end
    return Ad
end
# Symmetric A-pack for a diagonal-straddling panel (real symm). BRANCHLESS (OpenBLAS-style): per column
# the stored/mirror split is a single crossing, so each column packs a contiguous STORED run (reads A's
# column gp, stride 1) then a MIRROR run (reads A's row gp, stride lda) — no per-element `i≤j` branch in
# the hot loop. Off-diagonal panels use plain _pack_A! (stored: tA=false SIMD; mirror: tA=true).
function _pack_A_sym!(Ap::Vector{T}, A, ic::Int, pc::Int, mce::Int, kce::Int, up::Bool, alpha::T, mr::Int) where {T}
    np = cld(mce, mr)
    @inbounds for pi in 0:(np - 1)
        base = pi * mr * kce; pbase = pi * mr
        rhi = min(mr, mce - pbase)                 # valid rows r ∈ [0,rhi); r ≥ rhi → pad zero
        for p in 0:(kce - 1)
            gp = pc + p; o = base + p * mr; ls = gp - ic - pbase    # local diagonal crossing (in r)
            if up                                  # stored r ∈ [0,st_end) (gi≤gp), mirror r ∈ [st_end,rhi)
                st_end = clamp(ls + 1, 0, rhi)
                for r in 0:(st_end - 1)
                    Ap[o + r + 1] = alpha * A[ic + pbase + r + 1, gp + 1]
                end
                for r in st_end:(rhi - 1)
                    Ap[o + r + 1] = alpha * A[gp + 1, ic + pbase + r + 1]
                end
            else                                   # stored r ∈ [st_start,rhi) (gi≥gp), mirror r ∈ [0,st_start)
                st_start = clamp(ls, 0, rhi)
                for r in 0:(st_start - 1)
                    Ap[o + r + 1] = alpha * A[gp + 1, ic + pbase + r + 1]
                end
                for r in st_start:(rhi - 1)
                    Ap[o + r + 1] = alpha * A[ic + pbase + r + 1, gp + 1]
                end
            end
            for r in rhi:(mr - 1)
                Ap[o + r + 1] = zero(T)
            end     # pad rows beyond mce
        end
    end
    return
end
# Complex HERMITIAN A-pack (split re/im) for a diagonal-straddling OR full-mirror panel (hemm side-L):
# per panel-column the stored run reads A[i,gp] direct; the MIRROR run reads A[gp,i] CONJUGATED
# (A_herm[i,gp] = conj(A[gp,i])). No α (applied at the microkernel store). Mirrors _pack_A_sym! + the
# conj that makes it Hermitian. Full-stored panels use the SIMD _pack_A_cmplx! (tA=false) instead.
# `HERM` is the ONLY difference between the Hermitian and complex-symmetric packs: the mirrored half
# takes conj(A[j,i]) for Hermitian and A[j,i] unchanged for symmetric. It is a compile-time `Val` so the
# sign folds away rather than costing a branch per element, and so complex symm can share this path
# instead of materializing an n×n copy.
function _pack_A_sym_cmplx!(
        ApR::Vector{T}, ApI::Vector{T}, A, ic::Int, pc::Int, mce::Int, kce::Int,
        up::Bool, mr::Int, ::Val{HERM}
    ) where {T, HERM}
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
                    v = A[gp + 1, ic + pbase + r + 1]; ApR[o + r + 1] = real(v)
                    ApI[o + r + 1] = HERM ? -imag(v) : imag(v)
                end
            else                                           # mirror(conj) r∈[0,st_start); stored r∈[st_start,rhi)
                st_start = clamp(ls, 0, rhi)
                for r in 0:(st_start - 1)
                    v = A[gp + 1, ic + pbase + r + 1]; ApR[o + r + 1] = real(v)
                    ApI[o + r + 1] = HERM ? -imag(v) : imag(v)
                end
                for r in st_start:(rhi - 1)
                    v = A[ic + pbase + r + 1, gp + 1]; ApR[o + r + 1] = real(v); ApI[o + r + 1] = imag(v)
                end
            end
            for r in rhi:(mr - 1)
                ApR[o + r + 1] = zero(T); ApI[o + r + 1] = zero(T)
            end
        end
    end
    return
end
# Single-pass packed complex hemm (side-L): C := α·A_herm·B + β·C. Standard packed complex gemm (all C
# tiles) but each A-panel is packed from the Hermitian TRIANGLE on the fly (stored → SIMD _pack_A_cmplx!;
# mirror/straddle → _pack_A_sym_cmplx! with conj) — reads the triangle ONCE, no materialize, no 2×
# A-traffic. α at the microkernel store (A1=false); β·C up front. Reuses _microkernel_cmplx!.
# Inner (ir,jr) microkernel sweep of the packed Hermitian/symmetric path, with B0 (overwrite C) and A1
# (α=1 ⇒ pure interleave store) as compile-time `Val`s. Split out of `_hemm_packed_L!` so the driver can
# branch ONCE into literal `Val`s per (b0, a1) instead of constructing `Val(b0)` at runtime, which would
# be a trim violation. Four concrete instantiations; the ISA-sized `Val(_CMR)`/`Val(_CNR)` are unchanged.
@inline function _hemm_pack_sweep!(
        ::Val{B0}, ::Val{A1}, Cp0::Ptr{T}, ldc::Int, ARp::Ptr{T}, AIp::Ptr{T}, BRp::Ptr{T}, BIp::Ptr{T},
        ic::Int, jc::Int, mce::Int, nce::Int, kce::Int, mr::Int, nr::Int, alr::T, ali::T, sz::Int
    ) where {T, B0, A1}
    jr = 0
    while jr < nce
        nre = min(nr, nce - jr); ir = 0
        while ir < mce
            mre = min(mr, mce - ir)
            AR = Ptr{T}(ARp + div(ir, mr) * mr * kce * sz); AI = Ptr{T}(AIp + div(ir, mr) * mr * kce * sz)
            BR = Ptr{T}(BRp + div(jr, nr) * nr * kce * sz); BI = Ptr{T}(BIp + div(jr, nr) * nr * kce * sz)
            Cblk = Cp0 + (2 * (ic + ir) + 2 * (jc + jr) * ldc) * sz
            if mre == mr && nre == nr
                _microkernel_cmplx!(
                    Cblk, ldc, AR, AI, BR, BI, kce, alr, ali,
                    Val(_CMR), Val(_CNR), Val(1), Val(1), Val(B0), Val(A1)
                )
            else
                _microkernel_cmplx_masked!(
                    Cblk, ldc, AR, AI, BR, BI, kce, alr, ali,
                    mre, nre, Val(_CMR), Val(_CNR), Val(1), Val(1), Val(B0), Val(A1)
                )
            end
            ir += mr
        end
        jr += nr
    end
    return nothing
end
function _hemm_packed_L!(up::Bool, α, β, A, B, C, ::Val{HERM} = Val(true)) where {HERM}
    Tc = eltype(C); T = real(Tc); n = size(C, 1); m = size(C, 2); W = _vwidth(T); mr = _CMR * W; nr = _CNR
    kc = min(_CKC, n)
    mc = _at_mc_kc(_HW, eltype(C), kc, mr, cld(n, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(m, nr) * nr)
    ApR, ApI, BpR, BpI = _gemm_scratch_cmplx(T, cld(mc, mr) * mr * kc, cld(nc, nr) * nr * kc)
    # β=0 ⇒ the first kc-block OVERWRITES C (Val{B0}), so the n² pre-scaling pass is dead work. Skipping
    # it is also what makes β=0 correct on a C holding NaN/Inf: overwrite, never 0·C. Mirrors the gemm
    # driver's `b0first` (gemm.jl:1658).
    b0first = iszero(convert(Tc, β))
    b0first || _scale_C!(C, n, m, convert(Tc, β))
    ldc = stride(C, 2); sz = sizeof(T)
    alr = real(convert(Tc, α)); ali = imag(convert(Tc, α))
    a1 = isone(convert(Tc, α))            # α=1 ⇒ pure interleave store, no multiply (gemm.jl:1737)
    GC.@preserve C ApR ApI BpR BpI begin
        Cp0 = Ptr{T}(pointer(C)); ARp = pointer(ApR); AIp = pointer(ApI); BRp = pointer(BpR); BIp = pointer(BpI)
        jc = 0
        while jc < m
            nce = min(nc, m - jc); pc = 0
            while pc < n
                kce = min(kc, n - pc)
                _pack_B_cmplx!(BpR, BpI, B, pc, jc, kce, nce, false, nr)
                b0 = b0first && pc == 0       # overwrite only on the FIRST k-block; the rest accumulate
                ic = 0
                while ic < n
                    mce = min(mc, n - ic); a_hi = ic + mce - 1; p_hi = pc + kce - 1
                    stored = up ? (a_hi <= pc) : (ic >= p_hi)     # read A[i,gp] direct (SIMD)
                    stored ? _pack_A_cmplx!(ApR, ApI, A, ic, pc, mce, kce, false, mr) :
                        _pack_A_sym_cmplx!(ApR, ApI, A, ic, pc, mce, kce, up, mr, Val(HERM))
                    # B0/A1 were hardwired false here while `_gemm_cmplx_impl!` dispatches both. The gate
                    # benchmark runs α=1, β=0 (plots.jl:879), i.e. EXACTLY the case both specializations
                    # exist for, so this fork paid an n² zeroing pass plus a first-kc-block C read-back
                    # that the gemm driver skips, and a complex multiply per store that α=1 makes free.
                    # `b0`/`a1` are branched into literal `Val`s (never `Val(b0)`) — the trim-safe
                    # concrete-Val dispatch the driver uses at gemm.jl:707.
                    if b0
                        a1 ?
                            _hemm_pack_sweep!(Val(true), Val(true), Cp0, ldc, ARp, AIp, BRp, BIp,
                            ic, jc, mce, nce, kce, mr, nr, alr, ali, sz) :
                            _hemm_pack_sweep!(Val(true), Val(false), Cp0, ldc, ARp, AIp, BRp, BIp,
                            ic, jc, mce, nce, kce, mr, nr, alr, ali, sz)
                    else
                        a1 ?
                            _hemm_pack_sweep!(Val(false), Val(true), Cp0, ldc, ARp, AIp, BRp, BIp,
                            ic, jc, mce, nce, kce, mr, nr, alr, ali, sz) :
                            _hemm_pack_sweep!(Val(false), Val(false), Cp0, ldc, ARp, AIp, BRp, BIp,
                            ic, jc, mce, nce, kce, mr, nr, alr, ali, sz)
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
# Symmetric B-pack for a diagonal-straddling panel (real symm side R): the symmetric matrix is the
# gemm's RIGHT operand. Stored side reads A[gp,gj], mirror side A[gj,gp]. Off-diagonal panels use
# plain _pack_B! (stored: tB=false; mirror: tB=true). No α here — α rides on the left operand's pack.
function _pack_B_sym!(Bp::Vector{T}, A, pc::Int, jc::Int, kce::Int, nce::Int, up::Bool, nr::Int) where {T}
    np = cld(nce, nr)                              # branchless (OpenBLAS-style): stored/mirror = one crossing
    @inbounds for ji in 0:(np - 1)
        base = ji * nr * kce; cbase = ji * nr
        chi = min(nr, nce - cbase)                 # valid cols c ∈ [0,chi); c ≥ chi → pad zero
        for p in 0:(kce - 1)
            gp = pc + p; o = base + p * nr; ls = gp - jc - cbase
            if up                                  # stored gj≥gp: c ∈ [st,chi) (A row gp, strided); mirror c<st (A col gp)
                st = clamp(ls, 0, chi)
                for c in 0:(st - 1)
                    Bp[o + c + 1] = A[jc + cbase + c + 1, gp + 1]
                end
                for c in st:(chi - 1)
                    Bp[o + c + 1] = A[gp + 1, jc + cbase + c + 1]
                end
            else                                   # stored gj≤gp: c ∈ [0,st) (A row gp, strided); mirror c≥st (A col gp)
                st = clamp(ls + 1, 0, chi)
                for c in 0:(st - 1)
                    Bp[o + c + 1] = A[gp + 1, jc + cbase + c + 1]
                end
                for c in st:(chi - 1)
                    Bp[o + c + 1] = A[jc + cbase + c + 1, gp + 1]
                end
            end
            for c in chi:(nr - 1)
                Bp[o + c + 1] = zero(T)
            end
        end
    end
    return
end

# Single-pass packed symm (side L, real): C := α·A_sym·B + β·C as one gemm, packing A's symmetric
# panels directly (no n² materialize). M=n, N=m, K=n; classify each A-panel: stored / mirror / straddle.
function _symm_packed_L!(up::Bool, α::T, β::T, A, B, C) where {T <: BlasReal}
    n = size(C, 1); m = size(C, 2); W = _vwidth(T); mr = _MR * W; nr = _NR
    kc = min(_KC, n); mc = _at_mc_kc(_HW, T, kc, mr, cld(n, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(m, nr) * nr)
    Ap, Bp = _gemm_scratch(T, cld(mc, mr) * mr * kc, cld(nc, nr) * nr * kc)
    b0first = iszero(β)                    # β=0 ⇒ overwrite on the first k-block, no pre-scale pass
    b0first || _scale_C!(C, n, m, β)
    ldc = stride(C, 2); sz = sizeof(T)
    GC.@preserve C Ap Bp begin
        Cp0 = pointer(C); App = pointer(Ap); Bpp = pointer(Bp)
        jc = 0
        while jc < m
            nce = min(nc, m - jc); pc = 0
            while pc < n
                kce = min(kc, n - pc)
                _pack_B!(Bp, B, pc, jc, kce, nce, false, nr)
                b0 = b0first && pc == 0
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
                            Bpanel = Bpp + (div(jr, nr) * nr * kce) * sz
                            Cblk = Cp0 + ((ic + ir) + (jc + jr) * ldc) * sz
                            # β=0 ⇒ overwrite on the first k-block instead of pre-scaling C. The same
                            # drift fixed in the complex packed path (1d83669): `_microkernel!` has had
                            # a Val{B0} slot all along and this path never used it. Literal Vals, never
                            # Val(b0) — trim-safe, as at gemm.jl:707.
                            if mre == mr && nre == nr
                                b0 ?
                                    _microkernel!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, Val(_MR), Val(_NR), Val(true)) :
                                    _microkernel!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, Val(_MR), Val(_NR), Val(false))
                            else
                                b0 ?
                                    _microkernel_masked!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, mre, nre, Val(_MR), Val(_NR), Val(true)) :
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

# Single-pass packed symm (side R, real): C := α·B·A_sym + β·C. A_sym is the gemm's RIGHT operand.
# M=size(C,1), N=K=n; classify each A_sym panel (pc..K, jc..N): stored / mirror / straddle.
function _symm_packed_R!(up::Bool, α::T, β::T, B, A, C) where {T <: BlasReal}
    M = size(C, 1); n = size(A, 1); W = _vwidth(T); mr = _MR * W; nr = _NR
    kc = min(_KC, n); mc = _at_mc_kc(_HW, T, kc, mr, cld(M, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(n, nr) * nr)
    Ap, Bp = _gemm_scratch(T, cld(mc, mr) * mr * kc, cld(nc, nr) * nr * kc)
    b0first = iszero(β)                    # β=0 ⇒ overwrite on the first k-block, no pre-scale pass
    b0first || _scale_C!(C, M, n, β)
    ldc = stride(C, 2); sz = sizeof(T)
    GC.@preserve C Ap Bp begin
        Cp0 = pointer(C); App = pointer(Ap); Bpp = pointer(Bp)
        jc = 0
        while jc < n
            nce = min(nc, n - jc); j_hi = jc + nce - 1; pc = 0
            while pc < n
                kce = min(kc, n - pc); p_hi = pc + kce - 1
                stored = up ? (p_hi <= jc) : (pc >= j_hi)
                mirror = up ? (pc > j_hi) : (p_hi < jc)
                stored ? _pack_B!(Bp, A, pc, jc, kce, nce, false, nr) :
                    mirror ? _pack_B!(Bp, A, pc, jc, kce, nce, true, nr) :
                    _pack_B_sym!(Bp, A, pc, jc, kce, nce, up, nr)
                b0 = b0first && pc == 0
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
                            # β=0 ⇒ overwrite on the first k-block instead of pre-scaling C. The same
                            # drift fixed in the complex packed path (1d83669): `_microkernel!` has had
                            # a Val{B0} slot all along and this path never used it. Literal Vals, never
                            # Val(b0) — trim-safe, as at gemm.jl:707.
                            if mre == mr && nre == nr
                                b0 ?
                                    _microkernel!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, Val(_MR), Val(_NR), Val(true)) :
                                    _microkernel!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, Val(_MR), Val(_NR), Val(false))
                            else
                                b0 ?
                                    _microkernel_masked!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, mre, nre, Val(_MR), Val(_NR), Val(true)) :
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

# n above which symm uses the single-pass packed kernel vs materialize (O(n²) dense copy of the triangle) +
# the flagship gemm. DERIVED (req#8) via `_at_symm_mat_max` = √(L2/sizeof): a DIFFERENT criterion from the
# rank-k register cut — materialize+gemm beats the packed symmetric kernel at every measured Zen3 n (packed
# is dead weight on AVX2), and the only thing that unseats it is the O(n²) copy evicting the gemm's resident
# L2 A-block, i.e. when the materialized n×n copy no longer fits L2 (see cpuinfo.jl). Galen measured a mat≈pack
# tie at EXACTLY n=256=√(512K/8) (they converge for all n≥256), pinning the fraction at 1. This lifts the cut
# off the mistuned 96 (which routed n=112–192 to the slower packed path → the Zen3 AOCL misses) to 256.
# Predicts Zen4/Zen5 362 (DOWN from the _GEMM_UNPACK_MAX=448 placeholder — validate on the AVX-512 boxes).
# Overridable "symm_pack_cut".
# PDM: Derived — formula over detected consts: `_at_symm_mat_max(_HW)`
const _SYMM_PACK_CUT = @load_preference("symm_pack_cut", _at_symm_mat_max(_HW))::Int
# n above which complex hemm side-L uses the packed Hermitian kernel (reads the triangle once, on-the-fly
# conj-mirror pack). The packed path is the OLD classic-4M kernel (measured 0.85-0.90 at n=64-128 AVX2);
# the materialize path routes to _gemm_core!'s Karatsuba-3M at mid-n — the SAME path complex symm already
# gates on (zsymm n=128 = 1.06). So on AVX2 raise the cut past the whole gate range: hemm rides
# materialize+3M like symm. (AVX-512 keeps 32 — 3M path is AVX2-only; leave the gating classic path.)
# PDM: Derived — formula over detected consts: `_vwidth(Float64) == 4 ? 4096 : 32`
const _CHEMM_PACK_CUT = @load_preference("chemm_pack_cut", _vwidth(Float64) == 4 ? 4096 : 32)::Int
# Complex NON-Hermitian symm side-L uses the SAME packed kernel (`_hemm_packed_L!` with HERM=false):
# identical blocking, identical microkernel, the only difference is that the mirrored half of the A-pack
# does not conjugate. Same criterion ⇒ same cut, so this DEFAULTS TO `_CHEMM_PACK_CUT` rather than
# introducing a second literal; it carries its own preference name only so the two can be pinned apart
# if a box ever wants that. Before this, complex symm had no packed path at all and always materialized
# an n×n copy — measured symm/gemm = 1.045 at n=256 even after the materialize was tiled, against a
# zsymm gate of 0.976 there.
# TIER NOTE: inherited, not derived. The packed-vs-materialize margin for hemm on AVX-512 was measured
# small (packed faster by 0.9% at n=256 and 2.6% at n=512, ~neutral at 128/1024/2048), so the cut is
# Pin-tier debt on BOTH knobs, not a validated optimum. Deriving it needs a packed-vs-materialize sweep
# per box; do that before treating 32 as meaningful.
# PDM: Derived — same packed kernel as chemm, differing only by conjugation. | tune: n/a, follows chemm
const _CSYMM_PACK_CUT = @load_preference("csymm_pack_cut", _CHEMM_PACK_CUT)::Int
function _symm!(side_left::Bool, up::Bool, herm::Bool, α, β, A, B, C)
    n = size(A, 1)
    # Complex side-L in the 3M window → fuse the reflection into the 3M A-split (no materialize, no n²
    # complex scratch). Deletes the materialize tax that dominated mid-n hemm/symm. Concrete-complex only
    # (generic T<:Number / AD path falls through to materialize+_gemm_core!); AVX2-gated via _CGEMM_3M.
    # (n≤40 tried a direct triangle sweep — measured SLOWER than materialize: its many small _uker calls +
    # branchy panel fill cost more than one gemm at tiny n. Reverted; tiny-n stays on materialize.)
    if side_left && eltype(C) <: BlasComplex && _CGEMM_3M && _strided1(A) && _strided1(B) && _strided1(C)
        m2 = size(B, 2)
        if _fh_cgemm_3m_min() <= max(n, m2) <= _fh_cgemm_3m_max() && min(n, m2) >= _fh_cgemm_3m_kmin()
            return _hemm_3m_L!(up, herm, α, β, A, B, C)
        end
    end
    # UPPER BOUND ON THE PACKED PATH (2026-08-17). symm has the SAME flop count as gemm (2·n²·m) —
    # symmetry saves A-traffic, not arithmetic — so symm should run at gemm's speed. It did not:
    #
    #     Zen4   n=2048   PB gemm 0.3552s   PB symm 0.4150s   (symm 16.8% SLOWER, same flops)
    #                  n=4096   PB gemm 2.5319s   PB symm 3.3013s   (symm 30.4% SLOWER)
    #     AOCL for contrast     n=2048   its gemm 0.4530  its symm 0.4014  (theirs is FASTER than gemm)
    #
    # The whole deficit is that gemm rides Strassen (1.26-1.28x AOCL there) and the packed path cannot:
    # it never reaches `_gemm_core!` at all. The fall-through below ALREADY materializes the symmetric
    # operand and calls `_gemm_core!(…, false, false, …)` — an NN product, i.e. Strassen-eligible — so
    # the fix is not new code, it is declining the packed path once Strassen can pay.
    #
    # The materialize is O(n²) against an O(n³) saving: ~33.5 MB at n=2048, ~0.7 ms at ~50 GB/s,
    # against a 415 ms call — 0.16% — to buy a ~25% flop cut.
    #
    # `_strassen_depth > 0` is the exact predicate `_gemm_core!` will apply, so the two cannot disagree
    # (a hand-written `n >= _STRASSEN_MIN` here could drift from the depth rule and silently route to a
    # classical path with the materialize tax still paid — the worst of both).
    strassen_pays = _STRASSEN && eltype(C) <: BlasReal && !herm &&
        _strided1(A) && _strided1(B) && _strided1(C) &&
        _strassen_depth(size(C, 1), size(C, 2), n) > 0
    if !herm && eltype(C) <: BlasReal && n > _SYMM_PACK_CUT && !strassen_pays
        return side_left ?
            _symm_packed_L!(up, convert(eltype(C), α), convert(eltype(C), β), A, B, C) :
            _symm_packed_R!(up, convert(eltype(C), α), convert(eltype(C), β), B, A, C)
    elseif herm && eltype(C) <: BlasComplex && side_left && n > _CHEMM_PACK_CUT &&
            _strided1(B) && _strided1(C)                     # packed Hermitian (no materialize, triangle once)
        return _hemm_packed_L!(up, α, β, A, B, C, Val(true))
    elseif !herm && eltype(C) <: BlasComplex && side_left && n > _CSYMM_PACK_CUT &&
            _strided1(B) && _strided1(C)                 # packed complex-symmetric: no materialize
        return _hemm_packed_L!(up, α, β, A, B, C, Val(false))
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
    return (size(A, 1) == size(A, 2) == k) || _throw_square(:symm!, k)
end
function symm!(
        C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix; side::Char = 'L',
        uplo::Char = 'U', alpha::Number = true, beta::Number = false
    )
    _symm_check(side == 'L', A, B, C); _symm!(side == 'L', uplo == 'U', false, alpha, beta, A, B, C)
    return C
end
function hemm!(
        C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix; side::Char = 'L',
        uplo::Char = 'U', alpha::Number = true, beta::Number = false
    )
    _symm_check(side == 'L', A, B, C); _symm!(side == 'L', uplo == 'U', true, alpha, beta, A, B, C)
    return C
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
    if n <= _fh_syrk_dbase()
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
            @inbounds for j in 1:n, i in (up ? (1:j) : (j:n))
                Cd[i, j] += tmp[i, j] + conj(tmp[j, i])
            end
            @inbounds for i in 1:n
                Cd[i, i] = real(Cd[i, i])
            end   # clear rounding imag on the diagonal
        else
            @inbounds for j in 1:n, i in (up ? (1:j) : (j:n))
                Cd[i, j] += tmp[i, j] + tmp[j, i]
            end
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
    return (n, k)
end
# n above which syr2k uses the single-pass fused packed kernel (else the gemm-temp recursion). Same
# REGISTER-capacity criterion as syrk → shares `_at_rank_k_pack_cut` = (7·(nvreg−4)·W)/4 (see cpuinfo.jl).
# Reproduces Zen3 84 (measured: recursion wins n≤80, packed wins n≥96 — the old literal 96 routed n=96 to
# the slower recursion; 84 routes it to packed). Predicts Zen4/Zen5 392 (was a _GEMM_UNPACK_MAX placeholder).
# Overridable "syr2k_pack_cut".
# PDM: Derived — formula over detected consts: `_at_rank_k_pack_cut(_HW)`
const _SYR2K_PACK_CUT = @load_preference("syr2k_pack_cut", _at_rank_k_pack_cut(_HW))::Int
@inline _fh_syr2k_pack_cut() = (f = _FKR_syr2k_pack_cut[]; f >= 0 ? f : _SYR2K_PACK_CUT)
# Complex syr2k/her2k: n above which the two-product tri-output packed kernel beats the gemm-temp
# recursion (which computes a dense n×n temp per diagonal block — the 2× waste).
#
# ⚠ WAS `_vwidth(Float64) == 4 ? 8 : 8` — BOTH BRANCHES 8. A fake formula: it read as a width
# derivation and const-folded to the literal 8 on every machine, while the comment claimed "Tuned per
# machine". Collapsed to the constant it always was; this is a RECLASSIFICATION, byte-for-byte
# behaviour-neutral (verified: both branches evaluate to 8, and `_vwidth(Float64)` is 4 on AVX2 / 8 on
# AVX-512, so no box ever took a different arm). The same shape as the `_at_brd_nb(hw) = 8` fake
# formula retired on 2026-08-19 — a knob that takes a hardware input and ignores it.
#
# HONEST STATUS: the value 8 is NOT a measured optimum. It has never been swept — it could not be,
# since both arms were identical, so any A/B "confirming" it compared 8 against 8. What IS true is
# that the shipped behaviour has always been 8 and the gate passes on it (zsyr2k Zen4 1.031 vs AOCL,
# 2026-08-20 fleet run). Treat it as validated-by-gate, not as tuned.
# A cut of 8 is also plausibly algorithm-intrinsic rather than cache-derived: it is the n below which
# the packed kernel's setup cannot amortise at all, and its siblings sit at 16 (`csyrk_pack_cut`) and
# 4 (`csyrk_pack_cut_t`) — same order, no width scaling in any of the three.
# PDM: Literal — never swept: the retired ternary had two identical arms. Validated by gate only. | tune: unswept
const _CSYR2K_PACK_CUT = @load_preference("csyr2k_pack_cut", 8)::Int   # req8-ok: see above
@inline _fh_csyr2k_pack_cut() = (f = _FKR_csyr2k_pack_cut[]; f >= 0 ? f : _CSYR2K_PACK_CUT)
function syr2k!(
        C::AbstractMatrix, A::AbstractMatrix, Bm::AbstractMatrix; uplo::Char = 'U',
        trans::Char = 'N', alpha::Number = true, beta::Number = false
    )
    # Both operands share ONE trans char, so they fold together or not at all. Vocabulary as syrk.
    if trans == 'N' && _lazyop(A) == 'T' && _lazyop(Bm) == 'T'
        return syr2k!(C, parent(A), parent(Bm); uplo, trans = 'T', alpha, beta)
    end
    n, k = _syr2k_dims(C, A, Bm, trans); up = uplo == 'U'
    if eltype(C) <: BlasReal && n > _fh_syr2k_pack_cut() && k > 0
        _syr2k_packed!(up, trans != 'N', convert(eltype(C), alpha), convert(eltype(C), beta), A, Bm, C, k)
    elseif eltype(C) <: BlasComplex && trans == 'N' && _strided1(A) && _strided1(Bm) &&
            0 < n <= _CSYRK_UNPACK_MAX && k > 0 && !_ctrk_3m_ok(n, k)
        # Both operands must be pointer-able: `_ctri2_unpacked!` reads them directly. See the same
        # guard in `_syrk_blocked!` for why (a complex `A'` has no `strides` method).
        _syrk_scaleC!(C, up, beta)                                     # small-n trans='N': unpacked-tri (2 products)
        _ctri2_unpacked!(up, false, alpha, A, Bm, C, k)
    elseif eltype(C) <: BlasComplex && n > _fh_csyr2k_pack_cut() && k > 0
        _syrk_scaleC!(C, up, beta)
        _csyr2k_packed!(up, trans != 'N', false, alpha, A, Bm, C, k)
    else
        _syrk_scaleC!(C, up, beta)
        _syr2k_acc!(up, trans != 'N', false, alpha, A, Bm, C, k, _l3_tmp(eltype(C)), 0, n)
    end
    return C
end
function her2k!(
        C::AbstractMatrix, A::AbstractMatrix, Bm::AbstractMatrix; uplo::Char = 'U',
        trans::Char = 'N', alpha::Number = true, beta::Real = false
    )
    # Both operands share ONE trans char, so they fold together or not at all. Vocabulary as herk.
    if trans == 'N' && _lazyop(A) == 'C' && _lazyop(Bm) == 'C'
        return her2k!(C, parent(A), parent(Bm); uplo, trans = 'C', alpha, beta)
    end
    n, k = _syr2k_dims(C, A, Bm, trans); up = uplo == 'U'
    _syrk_scaleC!(C, up, beta)
    if eltype(C) <: BlasComplex && trans == 'N' && _strided1(A) && _strided1(Bm) &&
            0 < n <= _CSYRK_UNPACK_MAX && k > 0 && !_ctrk_3m_ok(n, k)
        # Both operands must be pointer-able: `_ctri2_unpacked!` reads them directly. See the same
        # guard in `_syrk_blocked!` for why (a complex `A'` has no `strides` method).
        return (_ctri2_unpacked!(up, true, alpha, A, Bm, C, k); C)     # small-n trans='N': unpacked-tri (2 products)
    elseif eltype(C) <: BlasComplex && n > _fh_csyr2k_pack_cut() && k > 0
        return (_csyr2k_packed!(up, trans != 'N', true, alpha, A, Bm, C, k); C)
    end
    _syr2k_acc!(up, trans != 'N', true, alpha, A, Bm, C, k, _l3_tmp(eltype(C)), 0, n)
    return C
end
