using LinearAlgebra: PosDefException

# LAPACK-level routines built on the gated Level-3 BLAS. First: Cholesky (potrf).
# Real symmetric positive-definite A = L·Lᵀ (uplo='L') or A = Uᵀ·U (uplo='U'), factored in place into
# the `uplo` triangle. Right-looking BLOCKED algorithm: each NB diagonal block is factored by the
# unblocked `_potf2` base, then the gated trsm (panel solve) + syrk (trailing rank-NB update) carry the
# bulk. Generic over T<:Real (the unblocked base + the generic trsm/syrk path make it ForwardDiff-
# traceable); BlasReal hits the SIMD trsm/syrk. NB = `_CHOL_BLOCK` (req#8: validated NB-insensitive — potrf
# n=1024…4096 within 0.5% over 96/128/192/256; see bench/req8_classification.md); `_CHOL_MC` derived from L2.

# Recursion base for the AD-traceable/fallback potrf (F32, Dual, F64-upper, F64-lower-non-strided). Below
# this, one unblocked potf2 (vectorized inner loop); above, right-looking blocked recurse (fast trsm/syrk).
# NOTE: the GATED F64-lower-strided path uses `_potrf_f64_lower!` (faer) and NEVER reads this — so the base
# is untouched by the potrf gate. req#8: the old 512 was a badly MISTUNED untested assumption (that potf2
# stays competitive to 512): fleet A/B (Zen4+Zen3, boost-locked) shows base=512 runs the unblocked potf2 on
# an L2-sized block and is 2.3–5.2× slower (F32) / up to 43.9× (Zen3 F64-upper) than OB. The unblocked base
# is uncompetitive past n≈64, so the base must be SMALL: base=32 wins hugely on both boxes (Zen4 F32
# 0.80–1.07, Zen3 F32 n=512 43.9→1.75 F64U). Optimum is small + µarch-flat (16–32 both boxes), NOT
# L1-residency (√(L1/8)=64 is WORSE than 32) — it is a recursion-overhead floor, matching `_CHOL_SB`=32.
# INVARIANT literal 32. `potrf_base` pref. (Residual, base-independent: F64-upper power-of-2 n spikes + the
# Zen3 F32 recursion sitting >1.0 even at optimum — separate structural issues, not this base.)
# PDM: Literal — recursion-overhead floor, µarch-flat at 16-32; the residency guess 64 is worse. | tune: no
const _POTRF_BASE = @load_preference("potrf_base", 32)::Int
# (The tiny-UPPER cutoff is `_potrf_udirect(T)` — a real MEASURE-tier auto-tune defined next to the
# lever bodies below, near `_potrf_gen!`. It was briefly a literal 12 here, which req#8b classifies as
# a violation; see the note there for the cost model and why Derive is not available.)
# Lever B (F32): the generic recursion's base crossover is TYPE-dependent. Float32 packs 2× per SIMD lane, so
# the scalar left-looking `_potf2` base is ~2× costlier per element while the SIMD trsm!/syrk! recursion is
# relatively cheaper → the crossover where blocking beats the base HALVES. Measured Zen4 (spotrf-L OB-ratio,
# base32→base16): n=32 0.33→0.62, n=64 0.76→0.96, n=128 0.97→1.14 — closes n≥64. (n=32 stays ~0.62: the
# scalar base still runs the whole 32² factorization; fully closing it needs a fast fused F32 base — the
# F64 faer base ported to Float32 — deferred, lower priority than F64 supernodes.) F64/Dual keep _POTRF_BASE.
# Derived (÷2 off the F64 anchor), µarch-invariant crossover like _CHOL_STH; knob "potrf_base_f32"; fleet-validate.
# MUST be a top-level const (like every other tuning const here): @load_preference inside a function body is a
# per-CALL runtime preferences lookup — it dominated microsecond small-n F32 factorizations (Zen3 spotrf 0.02).
# PDM: Derived — sizeof ratio: F32 is half F64's bytes, so half the n. | tune: n/a, follows potrf_base
const _POTRF_BASE_F32 = @load_preference("potrf_base_f32", _POTRF_BASE >> 1)::Int
@inline _potrf_base(::Type{Float32}) = _POTRF_BASE_F32
@inline _potrf_base(::Type{T}) where {T} = _POTRF_BASE
# Complex has NO fast SIMD base (the scalar potf2 above), so a 512 base = the whole factorization is scalar
# (measured: zpotrf n≤512 = 0.15-0.49× — all base, no recursion). A small base hands the bulk to the fast
# complex ztrsm!/zherk! recursion. Retune per box; Preferences knob "cpotrf_base".
# PDM: Derived — 32 + 4*lanes: a width-independent overhead floor plus a per-lane slope. | tune: n/a
const _CPOTRF_BASE = @load_preference("cpotrf_base", _at_cpotrf_base(_HW))::Int   # req#8: derived 32+4·W (48/64)
# n≤base ⇒ ONE vectorized `_cpotf2_lower!` call; n>base ⇒ right-looking blocked (see _cpotrf_rl_lower!).
# Base sweet spot is where the unblocked SIMD base still gates: AVX-512 (W=8) rides it to 64 (n=64 gates
# 1.09), but on AVX2 (W=4) the narrower datapath makes the n=64 base memory-bound (0.76), so cap it at 32
# (n≤32 gates, n>32 → rl). Keyed on _vwidth like the sibling cuts. Larger bases go memory-bound unblocked
# unblocked (base=128 → n=128 0.72), smaller pay recursion/small-k overhead — like the real f64 path's
# base-case threshold _CHOL_STH. ponytail: flat literal; Zen3(AVX2)/zen5 calibration via the knob.

# Contiguous scratch for the diagonal base block: the recursion's base is a view(A, js, js) whose
# columns are parent_ld apart (poor locality, the memory-bound potf2). Copying it to a contiguous
# buffer, factoring there, and copying back streams contiguous memory (better prefetch/TLB).
# _potf2_buf (potrf diagonal-base contiguous buffer) is the per-type L3Workspace `potf2` field
# (see src/workspace.jl).

# Unblocked right-looking Cholesky of an n×n block, lower triangle. Throws PosDefException at the first
# non-positive pivot (LAPACK's info>0). Reads/writes only the lower triangle.
# Hermitian-aware: `real(A[j,j])` (the diagonal is real for A=LLᴴ) and `conj` on the downdate operand.
# On T<:Real both are identities (compile away) → the real/ForwardDiff path is byte-identical; on
# T<:Complex this is the Hermitian Cholesky (zpotrf), matching LAPACK zpotf2.
function _potf2_lower!(A, n::Int)
    @inbounds for j in 1:n
        d = real(A[j, j])
        d > 0 || return j                       # failing column, LOCAL to this block; 0 = success
        ajj = sqrt(d); A[j, j] = ajj; invd = inv(ajj)
        for i in (j + 1):n
            A[i, j] *= invd
        end                 # scale column j below the diagonal
        for k in (j + 1):n                                        # rank-1 update of the lower trailing
            akj = conj(A[k, j])                                   # Hermitian: L[i,j]·conj(L[k,j])
            for i in k:n
                A[i, k] -= A[i, j] * akj
            end
        end
    end
    return 0
end
# Unblocked, upper triangle: A = Uᴴ·U (Hermitian). conj on the mirror operand A[j,i].
function _potf2_upper!(A, n::Int)
    @inbounds for j in 1:n
        d = real(A[j, j])
        d > 0 || return j                       # failing column, LOCAL to this block; 0 = success
        ajj = sqrt(d); A[j, j] = ajj; invd = inv(ajj)
        for i in (j + 1):n
            A[j, i] *= invd
        end                 # scale row j right of the diagonal
        for k in (j + 1):n                                        # rank-1 update of the upper trailing
            ajk = A[j, k]
            for i in (j + 1):k
                A[i, k] -= conj(A[j, i]) * ajk
            end   # Hermitian: conj(U[j,i])·U[j,k]
        end
    end
    return 0
end

# Base-update row-unroll (MR). MR=2 (two W-blocks/step, 1 L[j,k] load / 2 blocks → more ILP) LIFTS the
# memory-bound base on DOUBLE-PUMPED AVX-512 (Zen4) + AVX2 (Zen3) — n=48 0.97→1.18, n=64 1.11→1.17 — but
# REGRESSES native-512 (Zen5) where MR=1 already saturates. The discriminator is L1D size: 32K on the
# double-pump/AVX2 boxes, 48K on Zen5 (and Intel Tiger/Ice Lake+) native-512 — so key MR on `_L1_BYTES`
# (Zen4 fam 25 / Zen5 fam 26 also differ, but L1D is the causal-adjacent cache signal + already a const).
# PDM: Derived — formula over detected consts: `_at_cpotf2_mr(_HW)`
const _CPOTF2_MR = @load_preference("cpotf2_mr", _at_cpotf2_mr(_HW))::Int   # req#8: derived 64÷datapath_bytes (2 double-pump/AVX2, 1 native-512)

# Vectorized complex Hermitian Cholesky base (lower, A = L·Lᴴ). Complex analogue of `_chol_base_f64!`:
# left-looking, SIMD over i (W complex per step, deinterleaved re/im FMA chains), scalar tail. Column j:
# A[i,j] -= Σ_{k<j} L[i,k]·conj(L[j,k]) (i=j..n), then real diagonal d=A[j,j], scale below-diag by 1/√d.
# Reads only the lower triangle. Throws PosDefException at the first non-positive pivot (LAPACK zpotf2).
# The scalar `_potf2_lower!` above stays for Float32/Dual/other T (this method is BlasComplex-only).
function _cpotf2_lower!(A::AbstractMatrix{Tc}, n::Int) where {Tc <: BlasComplex}
    Tr = real(Tc); W = _vwidth(Tr); V = Vec{W, Tr}; V2 = Vec{2W, Tr}; sz = sizeof(Tr); ld = stride(A, 2)
    GC.@preserve A begin
        p = Ptr{Tr}(pointer(A))
        cx(i, k) = p + ((k - 1) * ld + (i - 1)) * 2 * sz                 # byte Ptr to the re-part of A[i,k]
        @inbounds for j in 1:n
            i = j
            if _CPOTF2_MR >= 2                                           # MR=2 (double-pump/AVX2): 2 W-blocks / step
                while i + 2W - 1 <= n
                    b0 = cx(i, j); b1 = cx(i + W, j)
                    (ar0, ai0) = _deint_cmplx(vload(V2, b0)); (ar1, ai1) = _deint_cmplx(vload(V2, b1))
                    for k in 1:(j - 1)
                        l = cx(j, k); sr = V(unsafe_load(l)); si = V(unsafe_load(l + sz))   # L[j,k], 1 load / 2 blocks
                        (vr0, vi0) = _deint_cmplx(vload(V2, cx(i, k))); (vr1, vi1) = _deint_cmplx(vload(V2, cx(i + W, k)))
                        ar0 = muladd(vr0, -sr, ar0); ar0 = muladd(vi0, -si, ar0); ai0 = muladd(vi0, -sr, ai0); ai0 = muladd(vr0, si, ai0)
                        ar1 = muladd(vr1, -sr, ar1); ar1 = muladd(vi1, -si, ar1); ai1 = muladd(vi1, -sr, ai1); ai1 = muladd(vr1, si, ai1)
                    end
                    vstore(_intlv_cmplx(ar0, ai0), b0); vstore(_intlv_cmplx(ar1, ai1), b1); i += 2W
                end
            end
            while i + W - 1 <= n                                         # SIMD: W complex rows / step
                base = cx(i, j); (ar, ai) = _deint_cmplx(vload(V2, base))
                for k in 1:(j - 1)
                    l = cx(j, k); sr = V(unsafe_load(l)); si = V(unsafe_load(l + sz))   # L[j,k]
                    (vr, vi) = _deint_cmplx(vload(V2, cx(i, k)))         # -v·conj(L[j,k])
                    ar = muladd(vr, -sr, ar); ar = muladd(vi, -si, ar)
                    ai = muladd(vi, -sr, ai); ai = muladd(vr, si, ai)
                end
                vstore(_intlv_cmplx(ar, ai), base); i += W
            end
            if i <= n                                                   # MASKED tail — the W-block body, one pass
                # 2W REAL lanes carry W complex rows, so the mask is per-lane complex INDEX
                # (1,1,2,2,…,W,W) — the `clane` idiom from simd_kernels.jl. Inactive lanes are never
                # accessed, so the interleaved load cannot run off the end of the column.
                msk = Vec(ntuple(l -> (l + 1) >> 1, Val(2W))) <= (n - i + 1)
                base = cx(i, j); (ar, ai) = _deint_cmplx(vload(V2, base, msk))
                for k in 1:(j - 1)
                    l = cx(j, k); sr = V(unsafe_load(l)); si = V(unsafe_load(l + sz))   # L[j,k]
                    (vr, vi) = _deint_cmplx(vload(V2, cx(i, k), msk))    # -v·conj(L[j,k])
                    ar = muladd(vr, -sr, ar); ar = muladd(vi, -si, ar)
                    ai = muladd(vi, -sr, ai); ai = muladd(vr, si, ai)
                end
                vstore(_intlv_cmplx(ar, ai), base, msk)
            end
            d = unsafe_load(cx(j, j))                                    # diagonal is real (Hermitian)
            d > 0 || return j                       # failing column, LOCAL to this block; 0 = success
            ajj = sqrt(d); invd = inv(ajj)
            unsafe_store!(cx(j, j), ajj); unsafe_store!(cx(j, j) + sz, zero(Tr))
            i = j + 1                                                   # scale below-diag by 1/√d (real)
            while i + W - 1 <= n
                base = cx(i, j); vstore(vload(V2, base) * V2(invd), base); i += W
            end
            while i <= n
                unsafe_store!(cx(i, j), unsafe_load(cx(i, j)) * invd)
                unsafe_store!(cx(i, j) + sz, unsafe_load(cx(i, j) + sz) * invd); i += 1
            end
        end
    end
    return 0
end

# Recursive (cache-oblivious) Cholesky. Lower: split 2×2 — factor A11, solve the off-diagonal panel
# A21·L11⁻ᵀ (trsm), downdate A22 -= A21·A21ᵀ (syrk), recurse A22. The top-level trsm/syrk are large-k
# (half-matrix → the gated packed L3 paths); only the ≤_POTRF_BASE diagonal base is scalar potf2.
# Factor a base block, via a contiguous buffer when A is a strided sub-block (better locality).
# These now return the failing column (local, 1-based) or 0 — see `_chol_hyb_f64!`. On failure the buffered
# variants return WITHOUT copying back, so A keeps its input contents exactly as before (the old code threw
# from inside the base, which skipped the copy-back the same way).
# THE BASE BUFFER IS AN ALIASING REMEDY, NOT A LOCALITY ONE — so its predicate has to be an aliasing test.
# `stride(A,2) > n` fires on the EXISTENCE of a stride, which cannot tell ld = n+8 from ld = 4096, and the
# copy round-trip is only worth its 2n² traffic in the second case. Measured on Zen4, ComplexF64, an
# interior diagonal sub-block, buf/nobuf (bench/probes/zpu3_buf_worth_it.jl):
#     n              16      32      48      64
#     lda=512      1.176   0.562   0.469   0.486     ← aliased: 2× WIN
#     lda=1024     1.191   0.563   0.511   0.483     ← aliased: 2× WIN
#     lda=520      1.826   1.626   1.637   1.566     ← NOT aliased: 1.6× LOSS at every n
# The loss row is the one production kept hitting: Lever C factors `view(M,1:n,1:n)` of `_potrf_pad`'s
# scratch, whose ld that helper deliberately keeps OFF the way-stride — so the copy could never pay, and
# it cost zpotrfU@32 1.95 µs of its 6.51 (`view` 5.380 vs `nobuf` 3.426, bench/probes/zpu2_buf_roundtrip.jl)
# while its LOWER sibling on the same kernel at the same size gated at 1.245.
# req#8 DERIVE, and no new knob: reuse `_potrf_needs_pad`'s byte-scaled way-stride test. `n` here is a base
# block (≤ `_CPOTRF_BASE`), hence always L2-resident, so that predicate's residency clause is trivially
# true and only the way-stride term survives.
@inline _potf2_needs_buf(A, n) = A isa SubArray && stride(A, 2) > n &&
    (stride(A, 2) * sizeof(eltype(A))) % (_L1_WAY_BYTES >> 2) == 0
@inline function _potf2b_lower!(A, n::Int)
    if eltype(A) <: BlasComplex                                         # SIMD Hermitian base (via contig buf if aliased)
        if n >= 8 && _potf2_needs_buf(A, n)
            buf = _potf2_buf(eltype(A), n); copyto!(buf, A)
            f = _cpotf2_lower!(buf, n); f == 0 || return f
            copyto!(A, buf); return 0
        end
        return _cpotf2_lower!(A, n)
    end
    if n >= 128 && _potf2_needs_buf(A, n)
        buf = _potf2_buf(eltype(A), n); copyto!(buf, A)
        f = _potf2_lower!(buf, n); f == 0 || return f
        copyto!(A, buf); return 0
    end
    return _potf2_lower!(A, n)
end
@inline function _potf2b_upper!(A, n::Int)
    if n >= 128 && _potf2_needs_buf(A, n)
        buf = _potf2_buf(eltype(A), n); copyto!(buf, A)
        f = _potf2_upper!(buf, n); f == 0 || return f
        copyto!(A, buf); return 0
    end
    return _potf2_upper!(A, n)
end

function _potrf_lower!(A, n::Int, base::Int = _POTRF_BASE)
    n <= base && return _potf2b_lower!(A, n)
    h = n ÷ 2
    f = _potrf_lower!(view(A, 1:h, 1:h), h, base)
    f == 0 || return f
    A21 = view(A, (h + 1):n, 1:h)
    if eltype(A) <: Complex                                    # Hermitian: A21·L11⁻ᴴ + A22 -= A21·A21ᴴ
        trsm!(A21, view(A, 1:h, 1:h); side = 'R', uplo = 'L', transA = 'C', diag = 'N', alpha = true)
        herk!(view(A, (h + 1):n, (h + 1):n), A21; uplo = 'L', trans = 'N', alpha = -1.0, beta = 1.0)
    else
        trsm!(A21, view(A, 1:h, 1:h); side = 'R', uplo = 'L', transA = 'T', diag = 'N', alpha = true)
        syrk!(view(A, (h + 1):n, (h + 1):n), A21; uplo = 'L', trans = 'N', alpha = -1, beta = 1)
    end
    f = _potrf_lower!(view(A, (h + 1):n, (h + 1):n), n - h, base)
    f == 0 || return h + f                                     # trailing block starts at column h+1
    return 0
end
# Upper: A = Uᵀ·U. Off-diagonal panel A12 = U11⁻ᵀ·A12 (trsm side-L), downdate A22 -= A12ᵀ·A12 (syrk).
function _potrf_upper!(A, n::Int, base::Int = _POTRF_BASE)
    n <= base && return _potf2b_upper!(A, n)
    h = n ÷ 2
    f = _potrf_upper!(view(A, 1:h, 1:h), h, base)
    f == 0 || return f
    A12 = view(A, 1:h, (h + 1):n)
    if eltype(A) <: Complex                                    # Hermitian: U11⁻ᴴ·A12 + A22 -= A12ᴴ·A12
        trsm!(A12, view(A, 1:h, 1:h); side = 'L', uplo = 'U', transA = 'C', diag = 'N', alpha = true)
        herk!(view(A, (h + 1):n, (h + 1):n), A12; uplo = 'U', trans = 'C', alpha = -1.0, beta = 1.0)
    else
        trsm!(A12, view(A, 1:h, 1:h); side = 'L', uplo = 'U', transA = 'T', diag = 'N', alpha = true)
        syrk!(view(A, (h + 1):n, (h + 1):n), A12; uplo = 'U', trans = 'T', alpha = -1, beta = 1)
    end
    f = _potrf_upper!(view(A, (h + 1):n, (h + 1):n), n - h, base)
    f == 0 || return h + f                                     # trailing block starts at column h+1
    return 0
end

# Power-of-2 lda cache-set aliasing: the generic potrf recursion's trailing trsm!/syrk! read A sub-views at
# a power-of-2 column stride → columns collapse onto a few L1 sets → conflict misses (measured Zen4 F64-upper
# n=512 1.50× slower vs OB; padding the working ld → 0.74×, PB beats OB). Fix = mirror the gated lower path's
# whole-matrix pad (`_potrf_f64_lower!`): factor in an alias-free (ld=n+8) scratch, copy the `uplo` triangle
# back. Covers the UNGATED generic recursion — F64-upper, F32 (both uplo), complex-upper, F64/complex-lower-
# non-strided; the gated F64/complex-LOWER-STRIDED paths have their own kernels and never reach here.
# req#8: the pad predicate is byte-scaled off _L1_WAY_BYTES (reproduces `_chol_needs_pad`'s F64 128-elt period
# on the fleet, and classifies F32/complex by their own stride bytes). Pad ONLY when aliased (a pad above L2
# loses — measured on the lower path). Dual (non-BlasFloat) skips → byte-identical to the old generic path.
@inline _potrf_needs_pad(A, n) = _strided1(A) && n >= 128 &&
    let sb = stride(A, 2) * sizeof(eltype(A)), q = _L1_WAY_BYTES >> 2
    sb % q == 0 && (sb % (q << 1) == 0 || n * n * sizeof(eltype(A)) <= _L2_BYTES)
end

# PDM: Exempt — boolean switch (path on/off), not a tuned size.
const _POTRF_PAD = @load_preference("potrf_pad", true)::Bool   # disable to A/B the pad's benefit per µarch

# Transposed-triangle copies for the UPPER → gating-LOWER reuse: Lever A (F64, U=Lᵀ) and Lever C (complex
# Hermitian, U=Lᴴ, so `CJ=true` conjugates). Generic over T; the `CJ` flag is a type param so the conj
# const-folds away (real path is byte-identical to the old F64 code). CACHE-BLOCKED: a naive strided transpose
# thrashes L1 at large n (its strided side sweeps the whole matrix, so the cost tracks memory bandwidth and
# varies wildly by µarch); tiling confines each strided sweep to a `_TR_TB`² block that stays L1-resident, so
# the O(n²) transpose cost shrinks uniformly and the lever ≈ the lower kernel on every box. TB=32 is
# µarch-invariant (two 32² F64 tiles = 16 KB ≤ any real L1; complex 32² = 32 KB still fits typical L1).
# PDM: Literal — residency-INVARIANT: two 32^2 F64 tiles = 16 KB, under any real L1. Deriving it would change nothing.
const _TR_TB = 32
@inline function _tri_upper_to_lowerT!(
        pm::Ptr{T}, ldM::Int, pa::Ptr{T}, lda::Int, n::Int,
        ::Val{CJ} = Val(false)
    ) where {T, CJ}
    sz = sizeof(T)
    # Inner loops ordered so the CONTIGUOUS side is the READ of A and the strided side is the write into
    # the scratch M. The obvious other order (contiguous write down M's column, strided read across A's
    # row) is 2.1–3.8× SLOWER whenever A's lda is a power of two — which is exactly the gated sizes.
    # Measured on Zen4, transposing the upper triangle, bit-identical either way:
    #     F64  n=1024 lda=1024   strided-read 2056 µs (4.1 GB/s) → this order 543 µs (15.5)   3.79×
    #     F64  n=2048 lda=2048   strided-read 9459 µs (3.6)      → this order 4560 µs (7.4)   2.07×
    #     zF64 n=1024 lda=1024   strided-read 2516 µs (6.7)      → this order 1205 µs (13.9)  2.09×
    # It is po2 SET ALIASING on the strided side, not page span or bandwidth: at lda·8 % 4096 == 0 every
    # column lands in the same L1 set group. Padding the source lda by 8 recovers the same speed
    # (n=1024: 2095 → 435 µs; n=2048: 9433 → 3438), while lda = 4096 and 8192 are no worse than 2048 —
    # so span is irrelevant and the tile size is not the lever (see _TR_TB). The strided side here is M,
    # whose ldM comes from _potrf_pad and is deliberately NOT a power of two, so it cannot alias.
    # `_tri_lowerT_to_upper!` below keeps the mirrored order for the same reason and measures flat in lda.
    @inbounds for jb in 0:_TR_TB:(n - 1)            # M_lower[i,j] = (CJ ? conj : id)(A_upper[j,i]), i≥j
        je = min(jb + _TR_TB, n)
        for ib in jb:_TR_TB:(n - 1)
            ie = min(ib + _TR_TB, n)
            for i in ib:(ie - 1)
                ao = i * lda                        # A col i contiguous → M row i, tile-confined strided
                for j in jb:min(je - 1, i)          # keep i ≥ j (A's upper triangle)
                    v = unsafe_load(pa + (ao + j) * sz)
                    unsafe_store!(pm + (j * ldM + i) * sz, CJ ? conj(v) : v)
                end
            end
        end
    end
    return nothing
end
@inline function _tri_lowerT_to_upper!(
        pa::Ptr{T}, lda::Int, pm::Ptr{T}, ldM::Int, n::Int,
        ::Val{CJ} = Val(false)
    ) where {T, CJ}
    sz = sizeof(T)
    @inbounds for ib in 0:_TR_TB:(n - 1)            # A_upper[j,i] = (CJ ? conj : id)(M_lower[i,j]), j≤i
        ie = min(ib + _TR_TB, n)
        for jb in 0:_TR_TB:ib
            je = min(jb + _TR_TB, n)
            for i in ib:(ie - 1)
                ao = i * lda
                for j in jb:min(je - 1, i)          # A col i contiguous ← M row i, tile-confined strided
                    v = unsafe_load(pm + (j * ldM + i) * sz)
                    unsafe_store!(pa + (ao + j) * sz, CJ ? conj(v) : v)
                end
            end
        end
    end
    return nothing
end

# ── the Lever A/C bodies, factored out ────────────────────────────────────────────────────────────
# Extracted so the Measure harness below races EXACTLY the code the dispatch runs. Replicating a
# kernel inside its own tuner lets the two drift silently — that is what made a probe report the
# library as unchanged after the library had changed (bench/probes/pbtrf_tile_size.jl, 2026-08-06).
# Lever A (real): U = Lᵀ exactly (UᵀU = LLᵀ = A). `_potrf_f64_lower!` THROWS on a non-positive pivot,
# which is why this returns nothing — the throw happens before the transpose back, so A is unchanged.
# ── NATIVE-UPPER hybrid: factor U in place, no whole-matrix transpose ─────────────────────────────
# The transpose lever below is the right answer at small/mid n — the faer lower kernels gate there and
# the O(n²) round-trip is cheap against O(n³) work. At LARGE n that inverts: the round-trip is four
# extra n² DRAM passes (transpose in, transpose out, each reading and writing), and `_tri_upper_to_
# lowerT!`'s own comment measures itself at 7.4 GB/s at n=2048 — far under stream rate, because a
# transpose has no sequential access on one side.
#
# Measured cost of that round-trip (Zen4, cycles from the gate cache): PB's UPPER-minus-LOWER delta at
# n=2100 is 11% of runtime, where AOCL's is 3% — AOCL factors 'U' natively (VERIFIED by disassembling
# libflame: `dpotrf_` sends n>74 to netlib blocked left-looking on the requested triangle, no
# transpose). That 8-point delta is the whole potrfU large-n miss: Zen4 1000/2100, Zen5 1000/2048/
# 2100/4096.
#
# This is the exact TRANSPOSE-MIRROR of `_chol_hyb_f64!`, which is what makes it safe to add: same
# recursion, same block boundaries, each BLAS-3 call replaced by its transpose-dual, so
#     lower: A21 = trsm(side=R, uplo=L, transA=T);  A22 -= syrk(uplo=L, trans=N, A21)
#     upper: A12 = trsm(side=L, uplo=U, transA=T);  A22 -= syrk(uplo=U, trans=T, A12)
# with A12 = A21ᵀ throughout. Both call the same gating trsm!/syrk! kernels, so no new kernel is
# introduced and no rounding-order argument is needed beyond the one potrf already makes (lapack.jl
# already reassociates vs faer; there is no potrf bit-identity invariant, unlike LU's).
# Crossover: native-upper vs the transpose lever. DERIVE tier — the criterion is cache RESIDENCY, the
# one thing that actually decides it. The lever costs 4 extra n² DRAM passes against O(n³/3) of work,
# so its relative cost is ~12·(n²·sizeof(T)/bandwidth) / (n³/3 / flops) — negligible while A is
# cache-resident, dominant once A streams from DRAM. So switch where A stops fitting L3:
#     n ≥ sqrt(_L3_BYTES / sizeof(T))
# Zen4/Zen5 (16 MiB L3, F64): 1448. Zen3 (32 MiB): 2048. That places the switch between the passing
# n=512-1024 cells (where the lever's small-n advantage is real and measured) and the failing
# n=2048-4096 ones, without a literal. F32 gets a wider threshold automatically via sizeof.
@inline _potrf_unative_min(::Type{T}) where {T} = isqrt(_L3_BYTES ÷ sizeof(T))
# Forceable — and the first thing the hook bought was a FALSIFICATION of the suspicion that prompted it.
#
# The worry: on Zen3 L3=32MiB puts the switch at 2048, so potrfU@1000 takes the LEVER and reads 0.921
# vs AOCL while potrf@1000 (lower, same box, same run) reads 1.131. A 19% spread between an op and its
# own transpose mirror looked like a misplaced crossover.
#
# It is not. Measured Zen3 2026-08-28, boost-locked, PB/AOCL, 8 rounds, ABBA-rotated:
#     potrfU@1000   lever 0.922, 0.920   native 0.891, 0.890   -> lever wins by 3.5%
#     potrfU@512    lever 1.374          native 1.345          -> lever wins by 2.2%
# Forcing native-upper down to n=1000 makes it WORSE. The residency derivation is placing the crossover
# correctly, and potrfU@1000's deficit is the lever's own transpose round-trip (~2 passes over an 8MB A
# against ~6.6ms of factorization, which is the right order for the observed gap), not the choice of
# arm. Closing that cell means a cheaper transpose or a faster native-upper hybrid — NOT moving this
# threshold. Do not re-chase the crossover.
# Cold path — read once per potrf! call, not in any kernel.
@inline _fh_potrf_unative_min(::Type{T}) where {T} =
    (f = _FKR_potrf_unative_min[]; f >= 0 ? f : _potrf_unative_min(T))

function _chol_hyb_upper_f64!(M, n::Int, base::Int)
    if n <= base
        # Base: no native-upper leaf kernel exists, and writing one is a separate piece of work. Fall
        # back to the transpose lever HERE, where it is cheap — the block is `base`-sized, so the
        # round-trip is O(base²) per leaf, i.e. O(n·base) total rather than O(n²) once.
        _potrf_upper_lever_real!(view(M, 1:n, 1:n), n)
        return 0
    end
    h = n ÷ 2
    f = _chol_hyb_upper_f64!(view(M, 1:h, 1:h), h, base)
    f == 0 || return f
    A12 = view(M, 1:h, (h + 1):n)
    trsm!(A12, view(M, 1:h, 1:h); side = 'L', uplo = 'U', transA = 'T', diag = 'N', alpha = true)
    syrk!(view(M, (h + 1):n, (h + 1):n), A12; uplo = 'U', trans = 'T', alpha = -1, beta = 1)
    f = _chol_hyb_upper_f64!(view(M, (h + 1):n, (h + 1):n), n - h, base)
    f == 0 || return h + f
    return 0
end

@inline function _potrf_upper_lever_real!(A::AbstractMatrix{T}, n::Int) where {T}
    M = _potrf_pad(T, n); ldM = size(M, 1); lda = stride(A, 2)
    GC.@preserve A M _tri_upper_to_lowerT!(pointer(M), ldM, pointer(A), lda, n)
    _potrf_f64_lower!(view(M, 1:n, 1:n))            # M lower ← L (throws ⇒ A upper unchanged)
    GC.@preserve A M _tri_lowerT_to_upper!(pointer(A), lda, pointer(M), ldM, n)
    return nothing
end
# Lever C (complex Hermitian): U = Lᴴ, so the transpose CONJUGATES (Val(true)) and column j of U is
# column j of L — the index carries across unchanged. RETURNS the failing column (0 = success); the
# caller throws, so that the transpose-back is skipped and A's upper triangle is left untouched.
@inline function _potrf_upper_lever_cmplx!(A::AbstractMatrix{T}, n::Int) where {T}
    M = _potrf_pad(T, n); ldM = size(M, 1); lda = stride(A, 2)
    GC.@preserve A M _tri_upper_to_lowerT!(pointer(M), ldM, pointer(A), lda, n, Val(true))
    f = _cpotrf_lower!(view(M, 1:n, 1:n), n)
    f == 0 || return f
    GC.@preserve A M _tri_lowerT_to_upper!(pointer(A), lda, pointer(M), ldM, n, Val(true))
    return 0
end

# ── tiny-UPPER cutoff: the largest n still factored in place ──────────────────────────────────────
# PDM **MEASURE** tier, and it has to be. The choice is direct-scalar-base (c_s·n³/3) against
# transpose-plus-vectorised-lower (c_t·2n² + c_v·n³/3, c_v ≪ c_s), so the crossover is
# n* = 6·c_t/(c_s − c_v) — a ratio of our OWN kernels' rates. Nothing in the cache hierarchy or the
# ISA predicts it, it MOVES whenever either kernel improves, and it is TYPE-dependent: measured on
# Zen4 (bench/probes/potrf_upper_cross.jl, direct/transpose, <1 = direct wins)
#     n            8      12      16      20      24      32
#     Float64    0.767   0.815   1.026   0.829   1.415   1.825      ⇒ crossover ≈ 14-16
#     ComplexF64 0.573   0.693   0.867   0.991   1.06    1.342      ⇒ crossover ≈ 22
# A single shipped literal cannot express a per-type crossover, and req#8b names a validated literal
# as a VIOLATION regardless. (I shipped `12` first and labelled it "MEASURE tier, pinned
# conservatively"; "pinned" is the P rung for a SET preference, not for a hardcoded default.)
#
# CANDIDATES ARE DERIVED at both ends, and the STEP matters as much as the ends:
#   start — the transpose moves 2n² elements against n³/3 flops, so it cannot amortise below n ≈ 6;
#           round that cost-model point up to a ladder point. (Starting at the element width instead
#           meant Float32, whose width is 16, never tested below 16 at all.)
#   step  — HALF the element width. The lower kernel is vectorised but the direct base is SCALAR, so
#           the balance shifts at sub-vector granularity; both cost curves are smooth in n, so the
#           step directly bounds the quantisation error in the answer.
#   stop  — `_POTRF_BASE`, above which the direct path is no longer a single base call at all.
# Stepping by the FULL element width was measured wrong on this box: for Float64 on AVX-512 that is 8,
# giving candidates {8,16,24,32}, so the harness returned 8 where direct still wins well past it — a
# quantisation error that sent n=9..15 to the lever needlessly. Half-width gives {8,12,16,20,...}.
#
# THE CROSSOVER IS A SHALLOW, NON-MONOTONIC BAND, not a point — do not read the returned value as
# precise. Probe on Zen4 (direct/transpose, <1 = direct wins):
#     n=8 0.767   12 0.815   16 1.026   20 0.829   24 1.415   32 1.825
# direct loses at 16 and wins again at 20, i.e. ~16-20 is within a few percent either way, and the
# decisive loss only arrives at 24 (+41%). The harness scans ascending and returns the last n before
# the FIRST loss, so on this box it lands on 20; a different box (or a rebuilt kernel) may legitimately
# land anywhere in that band. That is fine — the cost of being wrong inside the band is a few percent
# at sizes the gate does not sample, while being wrong at 24+ is 40%+, and the scan cannot overshoot
# there. It is also exactly why this is Measure tier and not a literal: no reasoning picks a point out
# of a noisy band, only measurement on the host does.
# PDM: Measured — a tiny-n crossover inside a noisy band; only host measurement resolves it. | tune: candidate
const _POTRF_UDIRECT_PREF = @load_preference("potrf_upper_direct_max", nothing)
@static if isnothing(_POTRF_UDIRECT_PREF)
    # elements per SIMD vector for this element type (complex packs two reals per element)
    @inline _potrf_udirect_ew(::Type{T}) where {T} =
        max(1, _vwidth(real(T)) ÷ (T <: Complex ? 2 : 1))
    # DUEL DELETED 2026-08-19. This knob's own comments already confessed it: "20 / 16 / 18 / 12 —
    # four different cutoffs from one binary", and ComplexF64 "16/18/20 across ten fresh processes".
    # Re-measured 6 fresh processes per box: Zen3 resolves 12 for F32/F64/C32 and 10/11/11/8/11/12 for
    # C64; Zen4 24 (F32), 12/20/12/12/12/12 (F64), 20 (C32), 10/12/12/12/12/10 (C64).
    #
    # 12 is the default for three converging reasons, not one:
    #   1. It is the DOCUMENTED SAFE DIRECTION. The comment above states the δ regret bound biases the
    #      walk to exit early because "being early costs only the tiny-n win, being late regresses
    #      n=16+". A fixed 12 is early by construction.
    #   2. It is what juliac/build.jl has PINNED into the .so all along, so this makes the JIT path and
    #      the shipped binary agree instead of diverging per process.
    #   3. It is what Zen3 already resolves for three of four eltypes.
    # The cost is bounded and one-sided: on Zen4 F32/C32 (which resolved 24/20) the tiny-n direct
    # win is given up for n in 12..24. Pin `potrf_upper_direct_max` to recover it on a specific box.
    @inline _potrf_udirect(::Type{<:Any}) = (f = _FKR_potrf_upper_direct_max[]; f >= 0 ? f : _at_potrf_udirect(_HW))
else
    @inline _potrf_udirect(::Type{<:Any}) = _POTRF_UDIRECT_PREF::Int   # pinned (trim builds land here)
end

function _potrf_gen!(A, n::Int, base::Int, up::Bool)
    T = eltype(A)
    # TINY UPPER GOES STRAIGHT TO THE BASE — no transpose, no scratch.
    #
    # Levers A and C below factor an UPPER matrix by conj/transposing it into a scratch, running the
    # gating LOWER kernel, and transposing back. At n=8 that is 2·n² element copies plus scratch
    # handling to support ~n³/3 ≈ 170 flops, and it was the DEEPEST gap on the fleet — same number on
    # both µarchs, with a spread that leaves no doubt:
    #     zpotrfU@8  0.608 (Zen3 spread 0.012, Zen4 0.002)      potrfU@8  0.825 / 0.830
    #
    # THE CUTOFF IS MEASURED, AND `_POTRF_BASE` IS THE WRONG ANSWER — I shipped that first and it
    # regressed exactly what it should have: n=8 fixed, but zpotrfU@32 0.692 -> 0.516 and potrfU@32
    # -> 0.592. The reasoning error is worth keeping: below the base `_potrf_upper!` really does skip
    # the recursion and issue no BLAS-3, but the alternative is not free — `_potf2b_upper!` is SCALAR
    # while the lower kernel it replaces is VECTORIZED. The trade is c_s·n³/3 against c_t·2n² + c_v·n³/3
    # with c_v << c_s, so direct wins only while n < 6·c_t/(c_s − c_v): a ratio of IMPLEMENTATION
    # constants, not a structural boundary and not derivable from cache size or SIMD width.
    #
    # Measured (Zen4/Zen4, freq-locked, bench/probes/potrf_upper_cross.jl, direct/transpose — <1
    # means direct wins):
    #     n            8      12      16      20      24      32      48      64
    #     Float64    0.767   0.815   1.026   0.829   1.415   1.825   2.721   3.24
    #     ComplexF64 0.573   0.693   0.867   0.991   1.06    1.342   1.282   1.207
    # Crossover ≈ 14-16 real, ≈ 22 complex (complex runs later: its conjugating transpose costs more
    # relative to its base). 12 sits below BOTH with margin — 0.815 real, 0.693 complex — so it is
    # robust to a box whose vector kernel is relatively faster, which moves the crossover DOWN. It
    # deliberately leaves the n=16/20 complex wins on the table rather than ride close to a crossover
    # the gate does not sample (LAPACK sizes are 8, 32, 128, …).
    # PDM: MEASURE tier, pinned conservatively. Not derived — see the cost model above for why it
    # cannot be. Re-measure with that probe before tightening it.
    #
    # NOTE `_potrf_upper!` RETURNS the failing column; every caller of `_potrf_gen!` ignores its return
    # value and relies on a thrown PosDefException, so the throw has to happen here or a non-positive-
    # definite matrix would report success.
    if up && n <= _potrf_udirect(T)
        f = _potrf_upper!(A, n, base)
        f == 0 || throw(PosDefException(f))
        return A
    end
    # Lever A: real (F64/F32) UPPER factored via the gating faer LOWER kernels. U = Lᵀ exactly (UᵀU = LLᵀ = A).
    # faer gates at ALL sizes (incl. small-n, where the generic recursion's trsm!/syrk! overhead loses) and self-
    # pads po2 strides — so this closes real-upper's small-n + AVX2 mid-n gaps by reuse, portably. Transpose A's
    # upper into the (alias-free) scratch's lower, factor, transpose L back. F32 rides the same faer path (now
    # generic over T<:BlasReal) — closing F32-upper once F32-lower gates.
    if _POTRF_PAD && up && (T === Float64 || T === Float32) && _strided1(A)
        # Large n: factor the upper triangle NATIVELY (transpose-mirror hybrid) instead of paying the
        # whole-matrix round-trip. The crossover is a RESIDENCY criterion, hence Derive tier: the
        # round-trip's cost is 4 extra n² DRAM passes, which is negligible against O(n³) while the
        # matrix is cache-resident and dominant once it is not. So switch when A stops fitting L3.
        if n >= _fh_potrf_unative_min(T)
            f = _chol_hyb_upper_f64!(A, n, _chol_faer_base(T))
            f == 0 || throw(PosDefException(f))
            return A
        end
        _potrf_upper_lever_real!(A, n)
        return A
    end
    # Lever C: complex Hermitian UPPER via the gating _cpotrf_lower!. U = Lᴴ exactly (UᴴU = LLᴴ = A), so the
    # transpose CONJUGATES (Val(true)). Mirrors Lever A — closes zpotrf-upper's small/mid-n gap (the generic
    # _potrf_upper! recursion's trsm!/herk! overhead loses there; _cpotrf_lower! gates). Conj-transpose A's
    # upper into the scratch lower, factor, conj-transpose Lᴴ back into A's upper.
    if _POTRF_PAD && up && T <: BlasComplex && _strided1(A)
        let f = _potrf_upper_lever_cmplx!(A, n)
            f == 0 || throw(PosDefException(f))     # throw before transposing back ⇒ A upper unchanged
        end
        return A
    end
    if _POTRF_PAD && (T <: BlasReal || T <: BlasComplex) && _potrf_needs_pad(A, n)
        b = _potrf_pad(T, n); lda = stride(A, 2); ldb = size(b, 1); sz = sizeof(T)
        GC.@preserve A b begin
            pa = pointer(A); pb = pointer(b)
            @inbounds for j in 0:(n - 1)          # copy only the `uplo` triangle: upper=col prefix (j+1), lower=col suffix (n-j)
                up ? unsafe_copyto!(pb + (j * ldb) * sz, pa + (j * lda) * sz, j + 1) :
                    unsafe_copyto!(pb + (j * ldb + j) * sz, pa + (j * lda + j) * sz, n - j)
            end
            let f = up ? _potrf_upper!(view(b, 1:n, 1:n), n, base) : _potrf_lower!(view(b, 1:n, 1:n), n, base)
                f == 0 || throw(PosDefException(f))    # throw before copy-back ⇒ A unchanged
            end
            @inbounds for j in 0:(n - 1)          # factored triangle back; the opposite triangle of A is untouched (throws skip copy-back)
                up ? unsafe_copyto!(pa + (j * lda) * sz, pb + (j * ldb) * sz, j + 1) :
                    unsafe_copyto!(pa + (j * lda + j) * sz, pb + (j * ldb + j) * sz, n - j)
            end
        end
        return A
    end
    let f = up ? _potrf_upper!(A, n, base) : _potrf_lower!(A, n, base)
        f == 0 || throw(PosDefException(f))
    end
    return A
end

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# Float64 LOWER fast path — faithful port of faer 0.24.1 `cholesky_recursion_right_looking`
# (github.com/el-oso/BlazingPorts.jl, src/Factorizations.jl). Custom register-blocked SIMD kernels
# (left-looking base, fused trsm, fused syrk) — that fusion beats the gate where the generic recursion
# above paid the general trsm!/syrk! overhead at small Cholesky block sizes. Overwrites the lower
# triangle with L; the upper is scratch (computed into never-read memory). Float64-only; everything
# else (complex/Dual/upper) stays on the generic AD-traceable path. Returns false on a
# non-positive pivot. SIMD via PureBLAS's SIMD.jl layer at the detected width.
# NOTE: the `_f64`-suffixed kernels below are now GENERIC over T<:BlasReal (Float32/Float64); the suffix
# is historical. F64 codegen is byte-identical (W/V/sizeof const-fold for a concrete T).
const _CVF = Vec{_vwidth(Float64), Float64}     # vector type at host width (concrete const; used by lu.jl/svd_dc.jl)
const _CHOLW = _vwidth(Float64)                 # (used by lu.jl/svd_dc.jl)
# req#8 (validated 2026-07-16, Zen4): potrf large-n time is NB-INSENSITIVE — sweeping this 96/128/192/256
# moves n=1024…4096 potrf by <0.5% (noise; _CHOL_MC follows via the derived formula below). The large-n
# residual is the structural panel-major streaming gap ([[pureblas-potrf-campaign]]), not the block size,
# so 128 is a correct µarch-invariant (a formula would add spurious variation for zero gain). Knob-able.
@inline _chol_block(::Type{T}) where {T} = 128   # µarch-invariant NB (byte-relative tuning deferred; knob-able)
# Small-n (≤ _CHOL_FAER_BASE) block params. The left-looking base kernel is only ~24–31% of FMA peak
# (vs BLASFEO's 45–56%: kb pureblas-potrf-campaign) — it's the small-n bottleneck. Blocking SMALL routes
# the bulk through the FMA-efficient rank-k trailing (_trsm_right/_syrk) and confines the slow base to
# ≤16-col diagonal blocks. Measured Zen3: bs32/th16 vs the old bs128/th64 gives +15–37% at n=48–192
# (n=64 14.5→19.9, gate 1.12→1.54×). bs = 2·th keeps the trailing panel L1-resident.
# req#8 NOTE (validated 2026-07-16): th is a µarch-INVARIANT 16, NOT the width-scaled (MR+1)·W the earlier
# comment guessed. Zen4 A/B (same harness, potrf PB/AOCL small-n): th=16 vs th=32(=4·_CHOLW) → th=16 WINS
# n=32 3.46 vs 2.67 (−23%), 64 2.57v2.34, 128 1.75v1.66; ties n≥256. The slow left-looking base wants a SMALL
# fixed diagonal block (confine it, let the fast rank-k trailing do the bulk) — bigger-on-wide-SIMD regresses.
# So 16 is a measured µarch-invariant base cap (a crossover, not a cache/width size), correctly flat. Knob-able.
@inline _chol_sth(::Type{T}) where {T} = 16      # base-case element crossover (µarch-invariant; fleet-tuned later)
@inline _chol_sb(::Type{T}) where {T} = 32       # small-n rl block size (= 2·_chol_sth)
# trsm/syrk column block = the register-TILE WIDTH of the fused Cholesky base kernel (columns unrolled with
# register rank-update accumulators). PDM: proven-invariant Exempt, NOT a cache/width size — 4 columns ×
# their accumulators fit the register file on every ISA, and the small Cholesky base does not benefit from a
# wider tile (same finding as `_chol_sth`: "bigger-on-wide-SIMD regresses"). A register-tile crossover, flat
# across µarch. Pinned (P-tier) for calibration; fleet-confirm the invariance before deriving off _NVREG.
# NOT TUNABLE — STRUCTURAL, AND VALIDATED. The fused base kernel below is HAND-UNROLLED for exactly four
# columns (d0..d3, l10..l32, indices c0..c0+3); `_fh_chol_nb()` appears only as the guard
# `nb == _fh_chol_nb()`. So any other value does not run slower, it runs WRONG:
#     chol_nb = 1, 2, 8, 16, 32  -> PosDefException
#     chol_nb = 3                -> SILENTLY WRONG, rel err 1.0e-01   <-- no exception at all
#     chol_nb = 4                -> rel err 1.5e-15
# (measured 2026-08-22, n=256, oracle-confirmed PD input; same on Zen3 and Zen4.)
# nb=3 passes the guard and the body then reads c0+3, one column past intent; nb=8 passes the guard and
# leaves four columns unfactored, surfacing later as a PosDefException far from the cause.
#
# It was exposed as a `@load_preference` while its own comment already said "proven-invariant Exempt" —
# the marker said Literal/tune-candidate and contradicted the prose. A preference with exactly one legal
# value is not a knob, it is a footgun: pinning chol_nb=3 silently corrupts every Cholesky in the
# process. The preference is KEPT (removing it would break an existing LocalPreferences) but now fails
# LOUDLY at load, naming itself, instead of corrupting results.
# PDM: Exempt — structural: hand-unrolled 4-column register tile, not a tuning size. | tune: n/a — only 4 is correct
const _CHOL_NB = @load_preference("chol_nb", 4)::Int   # trsm panel column block (register-tile width)
_CHOL_NB == 4 || throw(ArgumentError(
    "PureBLAS: preference `chol_nb` = $_CHOL_NB is not supported — the fused Cholesky base kernel is " *
    "hand-unrolled for exactly 4 columns, and any other value silently produces a WRONG factorization " *
    "(or a misleading PosDefException). Remove the `chol_nb` pin from LocalPreferences.toml."))
@inline _fh_chol_nb() = _CHOL_NB   # NO force hook: structural, not tunable — a sweep must not corrupt it
# Live vector count of the fused side-R leaf's top tier (`_trsm_rl_split_f64!` below): MR=3 row-tiers ×
# _CHOL_NB column accumulators, plus the _CHOL_NB T-vectors and the diagonal/broadcast pair. Compared
# against `_NVREG` this says whether that leaf SPILLS on this box — 3*4+4+2 = 18, over AVX2's 16 ymm and
# under AVX-512's 32 zmm. `_trsm_right!` uses it to decide whether an lda conflict on A is worth a
# de-aliasing copy: the conflict only outweighs the copy when A's columns are actually RE-READ, which
# happens when the leaf spills (or when A does not fit L2). DERIVE (req#8) — a formula over detected
# consts, not a µarch literal. Lives here, not in level3.jl, because `_CHOL_NB` is defined in this file
# and level3.jl is included FIRST (PureBLAS.jl:20-21), so a const there cannot see it.
const _RL_MR_LIVE = 3 * _CHOL_NB + _CHOL_NB + 2
# NOT TUNABLE — STRUCTURAL, and WORSE THAN chol_nb: every non-default value returns a silently wrong
# factorization rather than throwing. Same hand-unrolled register tile, same guard-only use.
#     chol_nc = 2, 3, 8, 16, 32, 64, 128 -> WRONG, rel err 9.0e-02 .. 2.4e-01, NO exception
#     chol_nc = 1                        -> PosDefException
#     chol_nc = 4                        -> rel err 1.2e-15
# (measured 2026-08-22, n=256, oracle-confirmed PD input.)
#
# THIS IS ALSO WHY A PERF SWEEP MUST CHECK CORRECTNESS. bench/probes/knob_bulk_sweep.jl timed chol_nc at
# 16/32/64/128 and reported 1.12x / 1.35x / 1.63x "wins" on Zen4. Those arms were not faster kernels —
# they were the WRONG kernel, fast because it was not doing the work. `Measure.ab` takes a `check`
# argument; not passing one is what let a corrupted arm be published as the campaign's biggest speedup.
# PDM: Exempt — structural: hand-unrolled 4-column register tile, not a tuning size. | tune: n/a — only 4 is correct
const _CHOL_NC = @load_preference("chol_nc", 4)::Int   # syrk column block (register-tile width)
_CHOL_NC == 4 || throw(ArgumentError(
    "PureBLAS: preference `chol_nc` = $_CHOL_NC is not supported — the fused Cholesky base kernel is " *
    "hand-unrolled for exactly 4 columns, and any other value SILENTLY produces a wrong factorization. " *
    "Remove the `chol_nc` pin from LocalPreferences.toml."))
@inline _fh_chol_nc() = _CHOL_NC   # NO force hook: structural, not tunable — a sweep must not corrupt it
# Split the base k-reduction into 6 independent FMA chains (vs 3) — pays off only where the reduction is
# latency-bound: Haswell-class Intel AVX2 (narrow OOO). Auto-on there, off on Zen/AVX-512 (their OOO hides
# the chain — measured slight regression), overridable. See [[_INTEL_AVX2]] in cpuinfo.jl.
# PDM: Derived — formula over detected consts: `_INTEL_AVX2`
const _CHOL_BASE_SPLIT = @load_preference("chol_base_split", _INTEL_AVX2)::Bool
@inline _clidx(i, k, ld) = (k - 1) * ld + i                              # 1-based linear index
@inline _cvptr(p::Ptr{T}, i, k, ld) where {T} = p + (((k - 1) * ld + (i - 1)) * sizeof(T))   # byte Ptr to [i,k]

# base case (n ≤ threshold): left-looking SIMD panel Cholesky, ascending-k FMA, scale by 1/√diag.
function _chol_base_f64!(p::Ptr{T}, n::Int, ld::Int) where {T}
    W = _vwidth(T); V = Vec{W, T}
    @inbounds for j in 1:n
        i = j
        while i + 3W - 1 <= n                              # MR=3 row-vectors
            b0 = _cvptr(p, i, j, ld); b1 = _cvptr(p, i + W, j, ld); b2 = _cvptr(p, i + 2W, j, ld)
            a0 = vload(V, b0); a1 = vload(V, b1); a2 = vload(V, b2)
            if _CHOL_BASE_SPLIT
                # Haswell-class: split each row-block's k-reduction into even/odd partials → 6 independent
                # FMA chains so the 2 units aren't starved by the 5-cyc latency (llvm-mca: 10→5 cyc/iter).
                # Reassociates the dot product (not bit-identical to faer, still OpenBLAS-correct).
                d0 = zero(V); d1 = zero(V); d2 = zero(V)
                kk = 1
                while kk + 1 <= j - 1
                    g = V(-unsafe_load(p, _clidx(j, kk, ld))); h = V(-unsafe_load(p, _clidx(j, kk + 1, ld)))
                    a0 = muladd(g, vload(V, _cvptr(p, i, kk, ld)), a0)
                    d0 = muladd(h, vload(V, _cvptr(p, i, kk + 1, ld)), d0)
                    a1 = muladd(g, vload(V, _cvptr(p, i + W, kk, ld)), a1)
                    d1 = muladd(h, vload(V, _cvptr(p, i + W, kk + 1, ld)), d1)
                    a2 = muladd(g, vload(V, _cvptr(p, i + 2W, kk, ld)), a2)
                    d2 = muladd(h, vload(V, _cvptr(p, i + 2W, kk + 1, ld)), d2)
                    kk += 2
                end
                if kk <= j - 1                                  # odd tail k
                    g = V(-unsafe_load(p, _clidx(j, kk, ld)))
                    a0 = muladd(g, vload(V, _cvptr(p, i, kk, ld)), a0)
                    a1 = muladd(g, vload(V, _cvptr(p, i + W, kk, ld)), a1)
                    a2 = muladd(g, vload(V, _cvptr(p, i + 2W, kk, ld)), a2)
                end
                a0 += d0; a1 += d1; a2 += d2
            else                                               # Zen / AVX-512: OOO hides the chain → keep 3
                for k in 1:(j - 1)
                    g = V(-unsafe_load(p, _clidx(j, k, ld)))
                    a0 = muladd(g, vload(V, _cvptr(p, i, k, ld)), a0)
                    a1 = muladd(g, vload(V, _cvptr(p, i + W, k, ld)), a1)
                    a2 = muladd(g, vload(V, _cvptr(p, i + 2W, k, ld)), a2)
                end
            end
            vstore(a0, b0); vstore(a1, b1); vstore(a2, b2); i += 3W
        end
        while i + W - 1 <= n
            base = _cvptr(p, i, j, ld); acc = vload(V, base)
            for k in 1:(j - 1)
                acc = muladd(V(-unsafe_load(p, _clidx(j, k, ld))), vload(V, _cvptr(p, i, k, ld)), acc)
            end
            vstore(acc, base); i += W
        end
        if i <= n                                          # MASKED tail — the MR=1 body, one pass
            msk = Vec(ntuple(l -> l, Val(W))) <= (n - i + 1)   # inactive lanes are never accessed (no OOB)
            base = _cvptr(p, i, j, ld); acc = vload(V, base, msk)
            for k in 1:(j - 1)
                acc = muladd(V(-unsafe_load(p, _clidx(j, k, ld))), vload(V, _cvptr(p, i, k, ld), msk), acc)
            end
            vstore(acc, base, msk)
        end
        d = unsafe_load(p, _clidx(j, j, ld))
        (d > 0) || return j                                # failing column (1-based, local); 0 ⇒ success
        invd = inv(sqrt(d)); vinv = V(invd)
        i = j
        while i + W - 1 <= n
            base = _cvptr(p, i, j, ld); vstore(vload(V, base) * vinv, base); i += W
        end
        while i <= n
            unsafe_store!(p, unsafe_load(p, _clidx(i, j, ld)) * invd, _clidx(i, j, ld)); i += 1
        end
    end
    return 0
end

# panel solve: L10 (m×bs) from L10·L00ᵀ = A10, in place on A10. FUSED — each NB=4 column panel downdates
# against all prior columns AND does the within-panel triangular solve in ONE register pass (no store/re-load
# round-trip between them), MR=3/2/1 tiers like _syrk_lower + a scalar tail. Measured Zen3: 80–92% of FMA
# peak vs the old two-pass ~47–58% (1.6–1.8×) — brings trsm-R to syrk's efficiency, the residual potrf lever.
# The tile stays register-resident across downdate→solve. bs is a multiple of 4 in rl32 (powers of two down
# to 4); the nb<4 remainder is a scalar fallback. Solve: L10[:,c] = (A10[:,c] − Σ_{k<c} L10[:,k]·L00[c,k])/L00[c,c].
function _trsm_right_lower_f64!(p00::Ptr{T}, p10::Ptr{T}, bs::Int, m::Int, ld::Int) where {T}
    W = _vwidth(T); V = Vec{W, T}
    c0 = 1
    @inbounds while c0 <= bs
        nb = min(_fh_chol_nb(), bs - c0 + 1)
        if nb == _fh_chol_nb()
            d0 = inv(unsafe_load(p00, _clidx(c0, c0, ld)));         l10 = -unsafe_load(p00, _clidx(c0 + 1, c0, ld))
            d1 = inv(unsafe_load(p00, _clidx(c0 + 1, c0 + 1, ld))); l20 = -unsafe_load(p00, _clidx(c0 + 2, c0, ld)); l21 = -unsafe_load(p00, _clidx(c0 + 2, c0 + 1, ld))
            d2 = inv(unsafe_load(p00, _clidx(c0 + 2, c0 + 2, ld))); l30 = -unsafe_load(p00, _clidx(c0 + 3, c0, ld)); l31 = -unsafe_load(p00, _clidx(c0 + 3, c0 + 1, ld)); l32 = -unsafe_load(p00, _clidx(c0 + 3, c0 + 2, ld))
            d3 = inv(unsafe_load(p00, _clidx(c0 + 3, c0 + 3, ld)))
            vd0 = V(d0); vd1 = V(d1); vd2 = V(d2); vd3 = V(d3)
            vl10 = V(l10); vl20 = V(l20); vl21 = V(l21); vl30 = V(l30); vl31 = V(l31); vl32 = V(l32)
            i = 1
            while i + 3W - 1 <= m                    # MR=3 fused tier (12 accumulators)
                r1 = i + W; r2 = i + 2W
                a0 = vload(V, _cvptr(p10, i, c0, ld));     b0 = vload(V, _cvptr(p10, r1, c0, ld));     e0 = vload(V, _cvptr(p10, r2, c0, ld))
                a1 = vload(V, _cvptr(p10, i, c0 + 1, ld)); b1 = vload(V, _cvptr(p10, r1, c0 + 1, ld)); e1 = vload(V, _cvptr(p10, r2, c0 + 1, ld))
                a2 = vload(V, _cvptr(p10, i, c0 + 2, ld)); b2 = vload(V, _cvptr(p10, r1, c0 + 2, ld)); e2 = vload(V, _cvptr(p10, r2, c0 + 2, ld))
                a3 = vload(V, _cvptr(p10, i, c0 + 3, ld)); b3 = vload(V, _cvptr(p10, r1, c0 + 3, ld)); e3 = vload(V, _cvptr(p10, r2, c0 + 3, ld))
                for k in 1:(c0 - 1)
                    v0 = vload(V, _cvptr(p10, i, k, ld)); v1 = vload(V, _cvptr(p10, r1, k, ld)); v2 = vload(V, _cvptr(p10, r2, k, ld))
                    g0 = V(-unsafe_load(p00, _clidx(c0, k, ld)));     a0 = muladd(g0, v0, a0); b0 = muladd(g0, v1, b0); e0 = muladd(g0, v2, e0)
                    g1 = V(-unsafe_load(p00, _clidx(c0 + 1, k, ld))); a1 = muladd(g1, v0, a1); b1 = muladd(g1, v1, b1); e1 = muladd(g1, v2, e1)
                    g2 = V(-unsafe_load(p00, _clidx(c0 + 2, k, ld))); a2 = muladd(g2, v0, a2); b2 = muladd(g2, v1, b2); e2 = muladd(g2, v2, e2)
                    g3 = V(-unsafe_load(p00, _clidx(c0 + 3, k, ld))); a3 = muladd(g3, v0, a3); b3 = muladd(g3, v1, b3); e3 = muladd(g3, v2, e3)
                end
                a0 *= vd0; b0 *= vd0; e0 *= vd0
                a1 = muladd(vl10, a0, a1); b1 = muladd(vl10, b0, b1); e1 = muladd(vl10, e0, e1); a1 *= vd1; b1 *= vd1; e1 *= vd1
                a2 = muladd(vl20, a0, a2); b2 = muladd(vl20, b0, b2); e2 = muladd(vl20, e0, e2); a2 = muladd(vl21, a1, a2); b2 = muladd(vl21, b1, b2); e2 = muladd(vl21, e1, e2); a2 *= vd2; b2 *= vd2; e2 *= vd2
                a3 = muladd(vl30, a0, a3); b3 = muladd(vl30, b0, b3); e3 = muladd(vl30, e0, e3); a3 = muladd(vl31, a1, a3); b3 = muladd(vl31, b1, b3); e3 = muladd(vl31, e1, e3); a3 = muladd(vl32, a2, a3); b3 = muladd(vl32, b2, b3); e3 = muladd(vl32, e2, e3); a3 *= vd3; b3 *= vd3; e3 *= vd3
                vstore(a0, _cvptr(p10, i, c0, ld)); vstore(b0, _cvptr(p10, r1, c0, ld)); vstore(e0, _cvptr(p10, r2, c0, ld))
                vstore(a1, _cvptr(p10, i, c0 + 1, ld)); vstore(b1, _cvptr(p10, r1, c0 + 1, ld)); vstore(e1, _cvptr(p10, r2, c0 + 1, ld))
                vstore(a2, _cvptr(p10, i, c0 + 2, ld)); vstore(b2, _cvptr(p10, r1, c0 + 2, ld)); vstore(e2, _cvptr(p10, r2, c0 + 2, ld))
                vstore(a3, _cvptr(p10, i, c0 + 3, ld)); vstore(b3, _cvptr(p10, r1, c0 + 3, ld)); vstore(e3, _cvptr(p10, r2, c0 + 3, ld))
                i += 3W
            end
            while i + 2W - 1 <= m                    # MR=2 fused tier (8 accumulators)
                r1 = i + W
                a0 = vload(V, _cvptr(p10, i, c0, ld));     b0 = vload(V, _cvptr(p10, r1, c0, ld))
                a1 = vload(V, _cvptr(p10, i, c0 + 1, ld)); b1 = vload(V, _cvptr(p10, r1, c0 + 1, ld))
                a2 = vload(V, _cvptr(p10, i, c0 + 2, ld)); b2 = vload(V, _cvptr(p10, r1, c0 + 2, ld))
                a3 = vload(V, _cvptr(p10, i, c0 + 3, ld)); b3 = vload(V, _cvptr(p10, r1, c0 + 3, ld))
                for k in 1:(c0 - 1)
                    v0 = vload(V, _cvptr(p10, i, k, ld)); v1 = vload(V, _cvptr(p10, r1, k, ld))
                    g0 = V(-unsafe_load(p00, _clidx(c0, k, ld)));     a0 = muladd(g0, v0, a0); b0 = muladd(g0, v1, b0)
                    g1 = V(-unsafe_load(p00, _clidx(c0 + 1, k, ld))); a1 = muladd(g1, v0, a1); b1 = muladd(g1, v1, b1)
                    g2 = V(-unsafe_load(p00, _clidx(c0 + 2, k, ld))); a2 = muladd(g2, v0, a2); b2 = muladd(g2, v1, b2)
                    g3 = V(-unsafe_load(p00, _clidx(c0 + 3, k, ld))); a3 = muladd(g3, v0, a3); b3 = muladd(g3, v1, b3)
                end
                a0 *= vd0; b0 *= vd0
                a1 = muladd(vl10, a0, a1); b1 = muladd(vl10, b0, b1); a1 *= vd1; b1 *= vd1
                a2 = muladd(vl20, a0, a2); b2 = muladd(vl20, b0, b2); a2 = muladd(vl21, a1, a2); b2 = muladd(vl21, b1, b2); a2 *= vd2; b2 *= vd2
                a3 = muladd(vl30, a0, a3); b3 = muladd(vl30, b0, b3); a3 = muladd(vl31, a1, a3); b3 = muladd(vl31, b1, b3); a3 = muladd(vl32, a2, a3); b3 = muladd(vl32, b2, b3); a3 *= vd3; b3 *= vd3
                vstore(a0, _cvptr(p10, i, c0, ld)); vstore(b0, _cvptr(p10, r1, c0, ld))
                vstore(a1, _cvptr(p10, i, c0 + 1, ld)); vstore(b1, _cvptr(p10, r1, c0 + 1, ld))
                vstore(a2, _cvptr(p10, i, c0 + 2, ld)); vstore(b2, _cvptr(p10, r1, c0 + 2, ld))
                vstore(a3, _cvptr(p10, i, c0 + 3, ld)); vstore(b3, _cvptr(p10, r1, c0 + 3, ld))
                i += 2W
            end
            while i + W - 1 <= m                     # MR=1 fused tier (4 accumulators)
                a0 = vload(V, _cvptr(p10, i, c0, ld)); a1 = vload(V, _cvptr(p10, i, c0 + 1, ld)); a2 = vload(V, _cvptr(p10, i, c0 + 2, ld)); a3 = vload(V, _cvptr(p10, i, c0 + 3, ld))
                for k in 1:(c0 - 1)
                    v0 = vload(V, _cvptr(p10, i, k, ld))
                    a0 = muladd(V(-unsafe_load(p00, _clidx(c0, k, ld))), v0, a0)
                    a1 = muladd(V(-unsafe_load(p00, _clidx(c0 + 1, k, ld))), v0, a1)
                    a2 = muladd(V(-unsafe_load(p00, _clidx(c0 + 2, k, ld))), v0, a2)
                    a3 = muladd(V(-unsafe_load(p00, _clidx(c0 + 3, k, ld))), v0, a3)
                end
                a0 *= vd0
                a1 = muladd(vl10, a0, a1); a1 *= vd1
                a2 = muladd(vl20, a0, a2); a2 = muladd(vl21, a1, a2); a2 *= vd2
                a3 = muladd(vl30, a0, a3); a3 = muladd(vl31, a1, a3); a3 = muladd(vl32, a2, a3); a3 *= vd3
                vstore(a0, _cvptr(p10, i, c0, ld)); vstore(a1, _cvptr(p10, i, c0 + 1, ld)); vstore(a2, _cvptr(p10, i, c0 + 2, ld)); vstore(a3, _cvptr(p10, i, c0 + 3, ld))
                i += W
            end
            if i <= m                                 # MASKED tail (<W rows) — the MR=1 body, one pass
                msk = Vec(ntuple(l -> l, Val(W))) <= (m - i + 1)   # inactive lanes are never accessed (no OOB)
                a0 = vload(V, _cvptr(p10, i, c0, ld), msk);     a1 = vload(V, _cvptr(p10, i, c0 + 1, ld), msk)
                a2 = vload(V, _cvptr(p10, i, c0 + 2, ld), msk); a3 = vload(V, _cvptr(p10, i, c0 + 3, ld), msk)
                for k in 1:(c0 - 1)
                    v0 = vload(V, _cvptr(p10, i, k, ld), msk)
                    a0 = muladd(V(-unsafe_load(p00, _clidx(c0, k, ld))), v0, a0)
                    a1 = muladd(V(-unsafe_load(p00, _clidx(c0 + 1, k, ld))), v0, a1)
                    a2 = muladd(V(-unsafe_load(p00, _clidx(c0 + 2, k, ld))), v0, a2)
                    a3 = muladd(V(-unsafe_load(p00, _clidx(c0 + 3, k, ld))), v0, a3)
                end
                a0 *= vd0
                a1 = muladd(vl10, a0, a1); a1 *= vd1
                a2 = muladd(vl20, a0, a2); a2 = muladd(vl21, a1, a2); a2 *= vd2
                a3 = muladd(vl30, a0, a3); a3 = muladd(vl31, a1, a3); a3 = muladd(vl32, a2, a3); a3 *= vd3
                vstore(a0, _cvptr(p10, i, c0, ld), msk);     vstore(a1, _cvptr(p10, i, c0 + 1, ld), msk)
                vstore(a2, _cvptr(p10, i, c0 + 2, ld), msk); vstore(a3, _cvptr(p10, i, c0 + 3, ld), msk)
            end
        else                                              # nb<4 remainder, masked over ROWS
            i = 1
            while i <= m
                msk = Vec(ntuple(l -> l, Val(W))) <= (m - i + 1)
                for dj in 0:(nb - 1)
                    cc = c0 + dj; s = vload(V, _cvptr(p10, i, cc, ld), msk)
                    for k in 1:(cc - 1)
                        s = muladd(V(-unsafe_load(p00, _clidx(cc, k, ld))), vload(V, _cvptr(p10, i, k, ld), msk), s)
                    end
                    vstore(s / V(unsafe_load(p00, _clidx(cc, cc, ld))), _cvptr(p10, i, cc, ld), msk)
                end
                i += W
            end
        end
        c0 += _fh_chol_nb()
    end
    return nothing
end

# one trailing column j: A11[i,j] -= Σ_c L10[j,c]·L10[i,c]  (remainder / <NC-column path).
@inline function _syrk_panel_f64!(p11::Ptr{T}, p10::Ptr{T}, j::Int, m::Int, bs::Int, ld::Int) where {T}
    W = _vwidth(T); V = Vec{W, T}
    i = ((j - 1) ÷ W) * W + 1
    @inbounds while i + W - 1 <= m
        b = _cvptr(p11, i, j, ld); a = vload(V, b)
        for c in 1:bs
            a = muladd(V(-unsafe_load(p10, _clidx(j, c, ld))), vload(V, _cvptr(p10, i, c, ld)), a)
        end
        vstore(a, b); i += W
    end
    @inbounds if i <= m                              # MASKED tail — the body above, one pass
        msk = Vec(ntuple(l -> l, Val(W))) <= (m - i + 1)   # inactive lanes are never accessed (no OOB)
        b = _cvptr(p11, i, j, ld); a = vload(V, b, msk)
        for c in 1:bs
            a = muladd(V(-unsafe_load(p10, _clidx(j, c, ld))), vload(V, _cvptr(p10, i, c, ld), msk), a)
        end
        vstore(a, b, msk)
    end
    return nothing
end

# trailing symmetric rank-bs update A11 (m×m) −= L10·L10ᵀ. Register-blocked MR rows × NC cols.
function _syrk_lower_f64!(p11::Ptr{T}, p10::Ptr{T}, m::Int, bs::Int, ld::Int) where {T}
    W = _vwidth(T); V = Vec{W, T}
    j = 1
    @inbounds while j + _fh_chol_nc() - 1 <= m
        i = ((j - 1) ÷ W) * W + 1                # W-aligned triangular start (skip upper blocks)
        while i + 3W - 1 <= m                          # MR=3 × NC=4 = 12 accumulators
            r1 = i + W; r2 = i + 2W
            e00 = _cvptr(p11, i, j, ld);      A00 = vload(V, e00)
            e10 = _cvptr(p11, r1, j, ld);     C00 = vload(V, e10)
            e20 = _cvptr(p11, r2, j, ld);     D00 = vload(V, e20)
            e01 = _cvptr(p11, i, j + 1, ld);  A01 = vload(V, e01)
            e11 = _cvptr(p11, r1, j + 1, ld); C01 = vload(V, e11)
            e21 = _cvptr(p11, r2, j + 1, ld); D01 = vload(V, e21)
            e02 = _cvptr(p11, i, j + 2, ld);  A02 = vload(V, e02)
            e12 = _cvptr(p11, r1, j + 2, ld); C02 = vload(V, e12)
            e22 = _cvptr(p11, r2, j + 2, ld); D02 = vload(V, e22)
            e03 = _cvptr(p11, i, j + 3, ld);  A03 = vload(V, e03)
            e13 = _cvptr(p11, r1, j + 3, ld); C03 = vload(V, e13)
            e23 = _cvptr(p11, r2, j + 3, ld); D03 = vload(V, e23)
            for c in 1:bs
                v0 = vload(V, _cvptr(p10, i, c, ld)); v1 = vload(V, _cvptr(p10, r1, c, ld)); v2 = vload(V, _cvptr(p10, r2, c, ld))
                g0 = V(-unsafe_load(p10, _clidx(j, c, ld)));     A00 = muladd(g0, v0, A00); C00 = muladd(g0, v1, C00); D00 = muladd(g0, v2, D00)
                g1 = V(-unsafe_load(p10, _clidx(j + 1, c, ld))); A01 = muladd(g1, v0, A01); C01 = muladd(g1, v1, C01); D01 = muladd(g1, v2, D01)
                g2 = V(-unsafe_load(p10, _clidx(j + 2, c, ld))); A02 = muladd(g2, v0, A02); C02 = muladd(g2, v1, C02); D02 = muladd(g2, v2, D02)
                g3 = V(-unsafe_load(p10, _clidx(j + 3, c, ld))); A03 = muladd(g3, v0, A03); C03 = muladd(g3, v1, C03); D03 = muladd(g3, v2, D03)
            end
            vstore(A00, e00); vstore(A01, e01); vstore(A02, e02); vstore(A03, e03)
            vstore(C00, e10); vstore(C01, e11); vstore(C02, e12); vstore(C03, e13)
            vstore(D00, e20); vstore(D01, e21); vstore(D02, e22); vstore(D03, e23)
            i += 3W
        end
        while i + 2W - 1 <= m                          # MR=2 × NC=4 = 8 accumulators
            r1 = i + W
            d00 = _cvptr(p11, i, j, ld);     A00 = vload(V, d00)
            d10 = _cvptr(p11, r1, j, ld);    B00 = vload(V, d10)
            d01 = _cvptr(p11, i, j + 1, ld); A01 = vload(V, d01)
            d11 = _cvptr(p11, r1, j + 1, ld); B01 = vload(V, d11)
            d02 = _cvptr(p11, i, j + 2, ld); A02 = vload(V, d02)
            d12 = _cvptr(p11, r1, j + 2, ld); B02 = vload(V, d12)
            d03 = _cvptr(p11, i, j + 3, ld); A03 = vload(V, d03)
            d13 = _cvptr(p11, r1, j + 3, ld); B03 = vload(V, d13)
            for c in 1:bs
                v0 = vload(V, _cvptr(p10, i, c, ld)); v1 = vload(V, _cvptr(p10, r1, c, ld))
                g0 = V(-unsafe_load(p10, _clidx(j, c, ld)));     A00 = muladd(g0, v0, A00); B00 = muladd(g0, v1, B00)
                g1 = V(-unsafe_load(p10, _clidx(j + 1, c, ld))); A01 = muladd(g1, v0, A01); B01 = muladd(g1, v1, B01)
                g2 = V(-unsafe_load(p10, _clidx(j + 2, c, ld))); A02 = muladd(g2, v0, A02); B02 = muladd(g2, v1, B02)
                g3 = V(-unsafe_load(p10, _clidx(j + 3, c, ld))); A03 = muladd(g3, v0, A03); B03 = muladd(g3, v1, B03)
            end
            vstore(A00, d00); vstore(A01, d01); vstore(A02, d02); vstore(A03, d03)
            vstore(B00, d10); vstore(B01, d11); vstore(B02, d12); vstore(B03, d13)
            i += 2W
        end
        while i + W - 1 <= m
            b0 = _cvptr(p11, i, j, ld);     a0 = vload(V, b0)
            b1 = _cvptr(p11, i, j + 1, ld); a1 = vload(V, b1)
            b2 = _cvptr(p11, i, j + 2, ld); a2 = vload(V, b2)
            b3 = _cvptr(p11, i, j + 3, ld); a3 = vload(V, b3)
            for c in 1:bs
                lic = vload(V, _cvptr(p10, i, c, ld))
                a0 = muladd(V(-unsafe_load(p10, _clidx(j, c, ld))), lic, a0)
                a1 = muladd(V(-unsafe_load(p10, _clidx(j + 1, c, ld))), lic, a1)
                a2 = muladd(V(-unsafe_load(p10, _clidx(j + 2, c, ld))), lic, a2)
                a3 = muladd(V(-unsafe_load(p10, _clidx(j + 3, c, ld))), lic, a3)
            end
            vstore(a0, b0); vstore(a1, b1); vstore(a2, b2); vstore(a3, b3); i += W
        end
        if i <= m                                        # MASKED tail — the MR=1 body, one pass
            msk = Vec(ntuple(l -> l, Val(W))) <= (m - i + 1)   # inactive lanes are never accessed (no OOB)
            b0 = _cvptr(p11, i, j, ld);     a0 = vload(V, b0, msk)
            b1 = _cvptr(p11, i, j + 1, ld); a1 = vload(V, b1, msk)
            b2 = _cvptr(p11, i, j + 2, ld); a2 = vload(V, b2, msk)
            b3 = _cvptr(p11, i, j + 3, ld); a3 = vload(V, b3, msk)
            for c in 1:bs
                lic = vload(V, _cvptr(p10, i, c, ld), msk)
                a0 = muladd(V(-unsafe_load(p10, _clidx(j, c, ld))), lic, a0)
                a1 = muladd(V(-unsafe_load(p10, _clidx(j + 1, c, ld))), lic, a1)
                a2 = muladd(V(-unsafe_load(p10, _clidx(j + 2, c, ld))), lic, a2)
                a3 = muladd(V(-unsafe_load(p10, _clidx(j + 3, c, ld))), lic, a3)
            end
            vstore(a0, b0, msk); vstore(a1, b1, msk); vstore(a2, b2, msk); vstore(a3, b3, msk)
        end
        j += _fh_chol_nc()
    end
    while j <= m
        _syrk_panel_f64!(p11, p10, j, m, bs, ld); j += 1
    end
    return nothing
end

# right-looking recursive driver (faer cholesky_recursion_right_looking).
function _chol_rl_f64!(p::Ptr{T}, n::Int, ld::Int, block_size::Int, threshold::Int) where {T}
    W = _vwidth(T); V = Vec{W, T}
    n <= threshold && return _chol_base_f64!(p, n, ld)
    bs_outer = min(nextpow(2, n) ÷ 2, block_size)
    j = 0
    while j < n
        bs = min(bs_outer, n - j)
        f = _chol_rl_f64!(_cvptr(p, j + 1, j + 1, ld), bs, ld, block_size, threshold)
        f == 0 || return j + f                             # lift the sub-block's column into this frame
        m = n - j - bs
        if m > 0
            p10 = _cvptr(p, j + bs + 1, j + 1, ld); p11 = _cvptr(p, j + bs + 1, j + bs + 1, ld)
            _trsm_right_lower_f64!(_cvptr(p, j + 1, j + 1, ld), p10, bs, m, ld)
            _syrk_lower_f64!(p11, p10, m, bs, ld)
        end
        j += bs
    end
    return 0
end

# A power-of-two leading dimension aliases columns into the same cache sets (the LDA=2^k conflict,
# ~1.3–1.5× slower at n≥512). When A's stride is a po2, factor in a padded (ld+8) scratch and copy
# back — bit-identical, ld is pure addressing. Reusable buffer via _chol_pad (single-thread; MT deferred).
# _chol_faer_base: ≤ this → faer rl kernels; above → hybrid halving routing the O(n³) trailing through the
# cache-blocked gating syrk!/trsm!. AVX-512 (32 regs) rides the faer syrk to 1024 (n=1024 0.70→0.87,
# n=2048 0.85→0.91 on the hybrid; W=8 stays off the AVX2 panel driver). AVX2 (16 regs) never halves — its
# large-n path is the fused panel driver (n>_CHOL_RL_MAX), so its base = _CHOL_RL_MAX (all n≤224 → rl32).
# AVX2: block-small rl32 (confined slow base + faer rank-k trailing) beats the cache-blocked panel driver
# until the trailing submatrix outgrows L2 — measured Zen3 crossover 224 (rl 37.8 vs panel 33.6) → 256
# (rl 28.2 vs panel 34.5). Bound: n² · 8 ≲ L2 ⇒ n ≲ √(L2/8) ≈ 256; the working panel needs headroom so
# 7⁄8 of that ≈ 224 → √(_L2_BYTES/8)·7⁄8 (Zen3 512 KB L2 → 224 EXACT). NB: the 7⁄8 is a ONE-POINT FIT to
# Zen3's 224/256 bracket — the √-L2 form is physical but the coefficient is unvalidated off 512K; the only
# extrapolation point in the fleet (Zen4 1 MB) is EXEMPTED below (W=8 branch). Validate on the next AVX2 box
# with ≠512K L2 before trusting the scaling (a bare literal 224 has the same epistemic content today).
# W=8 is a DIFFERENT criterion — the hybrid-halving faer base (32-reg), not the √-L2 crossover (which would
# give ~317). 128 is µarch-invariant across the AVX-512 fleet (Zen4+Zen5 both gate potrf with it) → kept flat.
@inline _chol_rl_max(::Type{T}) where {T} = _NVREG == 32 ? 128 : round(Int, sqrt(_L2_BYTES / sizeof(T)) * (7 / 8))
@inline _chol_faer_base(::Type{T}) where {T} = _NVREG == 32 ? 1024 : _chol_rl_max(T)
# Pad when columns alias L1 sets: Zen L1 = 64 sets × 64 B, so stride·8 a multiple of 64·64=4096 B
# (stride % 512 == 0) maps every column to the same sets. %256 = half-period (2 cols/set), %128 =
# quarter-period (4 cols/set) — both thrash L1. But the pad is an n² copy round-trip, so it only wins
# when it's cheap vs the aliasing it removes: strong (%256) aliasing always, OR quarter-period (%128)
# only when the matrix is L2-resident (copy cheap, L1-aliasing-dominated). Measured on Zen3: n=128
# (128 KB, %128) 0.887→1.018 (pad wins); n=384 (1.1 MB > L2, %128 not %256) 0.918→0.899 (pad LOSES).
@inline function _chol_needs_pad(A, n)
    T = eltype(A); sb = stride(A, 2) * sizeof(T)      # aliasing is on the BYTE stride
    # DERIVED from the detected way stride, not the literals 1024/2048. Those were quarter- and
    # half-period of an ASSUMED 4096 B way stride ("Zen L1 = 64 sets × 64 B" above) — correct on the
    # whole fleet only because 32 KiB/8-way and 48 KiB/12-way both give 4096 B, and wrong on any other
    # geometry (a 4-way 32 KiB L1 aliases at 8192 B, so the literals would fire spuriously and pay an
    # n² copy for aliasing that is not there). The sibling `_potrf_needs_pad` already derives exactly
    # this as `_L1_WAY_BYTES >> 2`; having one spelled as a derivation and one as literals is the same
    # duplication that let `_L1_WAY_D` hardcode assoc=8 and silently disable de-aliasing on Zen5.
    # Behaviour is UNCHANGED on all three boxes (4096 >> 2 == 1024).
    q = _L1_WAY_BYTES >> 2
    return n >= 128 && sb % q == 0 && (sb % (q << 1) == 0 || n * n * sizeof(T) <= _L2_BYTES)
end

# Hybrid driver: the faer kernels are fastest at small/medium n but their syrk isn't cache-blocked, so
# they fade at large n (panel re-streamed). Recurse by halving — the big off-diagonal blocks go through
# PureBLAS's cache-blocked trsm!/syrk! (which gate at large k) — and bottom out in the faer kernels.
# Returns the FIRST non-positive-pivot column (1-based, relative to this call's frame), 0 on success —
# it does not throw, so the offsets can be lifted through the halving recursion and the caller reports the
# true global column. LAPACK dpotrf sets info to exactly this; before, every failure reported column 1
# regardless, which reached LBT callers through `dpotrf_64_` (`cholesky(A; check=true)` named the wrong
# column). Success is the hot path and returns a plain 0, same register traffic as the old Bool.
function _chol_hyb_f64!(M, n::Int, base::Int)
    if n <= base
        return GC.@preserve M _chol_rl_f64!(
            pointer(M), n, stride(M, 2), _chol_sb(eltype(M)), _chol_sth(eltype(M))
        )
    end
    h = n ÷ 2
    f = _chol_hyb_f64!(view(M, 1:h, 1:h), h, base)
    f == 0 || return f
    A21 = view(M, (h + 1):n, 1:h)
    trsm!(A21, view(M, 1:h, 1:h); side = 'R', uplo = 'L', transA = 'T', diag = 'N', alpha = true)
    syrk!(view(M, (h + 1):n, (h + 1):n), A21; uplo = 'L', trans = 'N', alpha = -1, beta = 1)
    f = _chol_hyb_f64!(view(M, (h + 1):n, (h + 1):n), n - h, base)
    f == 0 || return h + f                                 # trailing block starts at column h+1
    return 0
end

# ── Fused panel driver: po2-strided AVX2 potrf without the whole-matrix pad round-trip ──────────────
# Measured (Zen3, kb pureblas-cholesky): the po2-stride tax lives in the trsm B-panel and the
# faer base reading A directly (syrk! packs both operands — stride-immune), and the whole-pad fix pays
# an n² copy round-trip that IS the residual gate gap at n=256–1024. Fix: per NB=128 block, (1) factor
# the diagonal block in a conflict-free scratch D, (2) solve the panel INTO a conflict-free workspace T
# with a split-ld trsm whose FIRST TOUCH reads po2 A21 as its initial operand load (the copy-in is
# fused away — zero extra traffic), (3) update the trailing from T (cache-resident @inline syrk when
# the T slab fits L2, packed syrk! reading T otherwise), (4) stream T back to A21 exactly ONCE (the
# factor's own output write). The @inline split kernels compose in ONE function body so T never
# round-trips through A21 between the trsm and the syrk. AVX-512 and non-po2 stay on the paths above.

# Split-ld faer trsm panel solve: L10·L00ᵀ = A10 with p00 (diag factor) at ld0, the panel SOLVED INTO
# pT at ldt, and the po2 source psrc at lds read exactly once (each column's initial load, before its
# first update). Same math/order as _trsm_right_lower_f64! — the c0==1 register pass degenerates to
# the first-touch copy (empty k-loop), fusing the copy-in.
# THIS KERNEL SPILLS ON AVX2 — AND THAT IS NOT THE BOTTLENECK. DO NOT "FIX" IT BY SHRINKING MR.
# Measured 2026-07-30 (`code_native` + bench/plots.jl op=trsmR, Zen3 freq-locked). The MR=3 tier
# holds MR·NC accumulators (12) plus NC + NC(NC−1)/2 = 10 loop-invariant broadcasts (`vd0..vd3`,
# `vl10..vl32`) = 22 live vectors against 16 ymm, so AVX2 spills where AVX-512 (32 zmm) does not:
#     Zen4  0 spill-stores /  0 reloads   trsmR n=128 vs AOCL 1.05
#     Zen3      Zen3 10 spill-stores / 21 reloads   trsmR n=128 vs AOCL 0.84
# That correlation is seductive and WRONG as a lever. Gating the row-tiers to cut live values (the 10
# invariants scale with `_CHOL_NB`, not MR, so each dropped tier only buys back NC registers) gives:
#     MR=3  10/21 spills → trsmR 1.16/0.81 vs AOCL, 1.13/0.95 vs OB   ← ships
#     MR=2   9/ 9 spills → trsmR 0.87/0.39 vs AOCL, 0.84/0.55 vs OB
#     MR=1   3/ 2 spills → trsmR 0.87/0.40 vs AOCL, 0.88/0.56 vs OB
# Removing the spill almost entirely makes the routine ~2× WORSE: the 12-accumulator ILP dominates the
# spill traffic by a wide margin, and the spilled values are the loop-invariant broadcasts, which stay
# L1-hot and reload cheaply. So MR=3 is correct on both µarchs and the trsmR n=128 gap (0.81 vs AOCL)
# is STILL UNEXPLAINED — both the L1-aliasing predicate (see level3.jl `_alias_ld`) and the register
# spill are now measured dead ends. Next hypotheses should be tested somewhere other than this tier.
@inline function _trsm_rl_split_f64!(
        p00::Ptr{T}, ld0::Int, psrc::Ptr{T}, lds::Int,
        pT::Ptr{T}, ldt::Int, bs::Int, m::Int
    ) where {T}
    W = _vwidth(T); V = Vec{W, T}
    c0 = 1
    @inbounds while c0 <= bs
        nb = min(_fh_chol_nb(), bs - c0 + 1)
        if nb == _fh_chol_nb()
            # FUSED: downdate (vs solved cols in T) + within-panel 4×4 solve in ONE register pass, store to T
            # once (kills the store/re-load round-trip of the old two-pass split kernel). +11–25% at the panel
            # shape (Zen3, bs=128). Coeffs from the diag factor p00; @simd ivdep kept on the downdate k-loop.
            d0 = inv(unsafe_load(p00, _clidx(c0, c0, ld0)));         l10 = -unsafe_load(p00, _clidx(c0 + 1, c0, ld0))
            d1 = inv(unsafe_load(p00, _clidx(c0 + 1, c0 + 1, ld0))); l20 = -unsafe_load(p00, _clidx(c0 + 2, c0, ld0)); l21 = -unsafe_load(p00, _clidx(c0 + 2, c0 + 1, ld0))
            d2 = inv(unsafe_load(p00, _clidx(c0 + 2, c0 + 2, ld0))); l30 = -unsafe_load(p00, _clidx(c0 + 3, c0, ld0)); l31 = -unsafe_load(p00, _clidx(c0 + 3, c0 + 1, ld0)); l32 = -unsafe_load(p00, _clidx(c0 + 3, c0 + 2, ld0))
            d3 = inv(unsafe_load(p00, _clidx(c0 + 3, c0 + 3, ld0)))
            vd0 = V(d0); vd1 = V(d1); vd2 = V(d2); vd3 = V(d3)
            vl10 = V(l10); vl20 = V(l20); vl21 = V(l21); vl30 = V(l30); vl31 = V(l31); vl32 = V(l32)
            i = 1
            while i + 3W - 1 <= m                            # MR=3 fused tier (12 accumulators)
                r1 = i + W; r2 = i + 2W                 # first touch: po2 A21 (psrc), read once
                a00 = vload(V, _cvptr(psrc, i, c0, lds));     a10 = vload(V, _cvptr(psrc, r1, c0, lds));     a20 = vload(V, _cvptr(psrc, r2, c0, lds))
                a01 = vload(V, _cvptr(psrc, i, c0 + 1, lds)); a11 = vload(V, _cvptr(psrc, r1, c0 + 1, lds)); a21 = vload(V, _cvptr(psrc, r2, c0 + 1, lds))
                a02 = vload(V, _cvptr(psrc, i, c0 + 2, lds)); a12 = vload(V, _cvptr(psrc, r1, c0 + 2, lds)); a22 = vload(V, _cvptr(psrc, r2, c0 + 2, lds))
                a03 = vload(V, _cvptr(psrc, i, c0 + 3, lds)); a13 = vload(V, _cvptr(psrc, r1, c0 + 3, lds)); a23 = vload(V, _cvptr(psrc, r2, c0 + 3, lds))
                @simd ivdep for k in 1:(c0 - 1)                       # solved columns: conflict-free T (12 FMA/iter → SW-pipeline; A/B'd)
                    v0 = vload(V, _cvptr(pT, i, k, ldt)); v1 = vload(V, _cvptr(pT, r1, k, ldt)); v2 = vload(V, _cvptr(pT, r2, k, ldt))
                    g = V(-unsafe_load(p00, _clidx(c0, k, ld0)));     a00 = muladd(g, v0, a00); a10 = muladd(g, v1, a10); a20 = muladd(g, v2, a20)
                    g = V(-unsafe_load(p00, _clidx(c0 + 1, k, ld0))); a01 = muladd(g, v0, a01); a11 = muladd(g, v1, a11); a21 = muladd(g, v2, a21)
                    g = V(-unsafe_load(p00, _clidx(c0 + 2, k, ld0))); a02 = muladd(g, v0, a02); a12 = muladd(g, v1, a12); a22 = muladd(g, v2, a22)
                    g = V(-unsafe_load(p00, _clidx(c0 + 3, k, ld0))); a03 = muladd(g, v0, a03); a13 = muladd(g, v1, a13); a23 = muladd(g, v2, a23)
                end
                a00 *= vd0; a10 *= vd0; a20 *= vd0
                a01 = muladd(vl10, a00, a01); a11 = muladd(vl10, a10, a11); a21 = muladd(vl10, a20, a21); a01 *= vd1; a11 *= vd1; a21 *= vd1
                a02 = muladd(vl20, a00, a02); a12 = muladd(vl20, a10, a12); a22 = muladd(vl20, a20, a22); a02 = muladd(vl21, a01, a02); a12 = muladd(vl21, a11, a12); a22 = muladd(vl21, a21, a22); a02 *= vd2; a12 *= vd2; a22 *= vd2
                a03 = muladd(vl30, a00, a03); a13 = muladd(vl30, a10, a13); a23 = muladd(vl30, a20, a23); a03 = muladd(vl31, a01, a03); a13 = muladd(vl31, a11, a13); a23 = muladd(vl31, a21, a23); a03 = muladd(vl32, a02, a03); a13 = muladd(vl32, a12, a13); a23 = muladd(vl32, a22, a23); a03 *= vd3; a13 *= vd3; a23 *= vd3
                vstore(a00, _cvptr(pT, i, c0, ldt));      vstore(a01, _cvptr(pT, i, c0 + 1, ldt))
                vstore(a02, _cvptr(pT, i, c0 + 2, ldt));  vstore(a03, _cvptr(pT, i, c0 + 3, ldt))
                vstore(a10, _cvptr(pT, r1, c0, ldt));     vstore(a11, _cvptr(pT, r1, c0 + 1, ldt))
                vstore(a12, _cvptr(pT, r1, c0 + 2, ldt)); vstore(a13, _cvptr(pT, r1, c0 + 3, ldt))
                vstore(a20, _cvptr(pT, r2, c0, ldt));     vstore(a21, _cvptr(pT, r2, c0 + 1, ldt))
                vstore(a22, _cvptr(pT, r2, c0 + 2, ldt)); vstore(a23, _cvptr(pT, r2, c0 + 3, ldt))
                i += 3W
            end
            while i + 2W - 1 <= m                            # MR=2 fused tier (8 accumulators)
                r1 = i + W
                a00 = vload(V, _cvptr(psrc, i, c0, lds));     a10 = vload(V, _cvptr(psrc, r1, c0, lds))
                a01 = vload(V, _cvptr(psrc, i, c0 + 1, lds)); a11 = vload(V, _cvptr(psrc, r1, c0 + 1, lds))
                a02 = vload(V, _cvptr(psrc, i, c0 + 2, lds)); a12 = vload(V, _cvptr(psrc, r1, c0 + 2, lds))
                a03 = vload(V, _cvptr(psrc, i, c0 + 3, lds)); a13 = vload(V, _cvptr(psrc, r1, c0 + 3, lds))
                @simd ivdep for k in 1:(c0 - 1)
                    v0 = vload(V, _cvptr(pT, i, k, ldt)); v1 = vload(V, _cvptr(pT, r1, k, ldt))
                    g = V(-unsafe_load(p00, _clidx(c0, k, ld0)));     a00 = muladd(g, v0, a00); a10 = muladd(g, v1, a10)
                    g = V(-unsafe_load(p00, _clidx(c0 + 1, k, ld0))); a01 = muladd(g, v0, a01); a11 = muladd(g, v1, a11)
                    g = V(-unsafe_load(p00, _clidx(c0 + 2, k, ld0))); a02 = muladd(g, v0, a02); a12 = muladd(g, v1, a12)
                    g = V(-unsafe_load(p00, _clidx(c0 + 3, k, ld0))); a03 = muladd(g, v0, a03); a13 = muladd(g, v1, a13)
                end
                a00 *= vd0; a10 *= vd0
                a01 = muladd(vl10, a00, a01); a11 = muladd(vl10, a10, a11); a01 *= vd1; a11 *= vd1
                a02 = muladd(vl20, a00, a02); a12 = muladd(vl20, a10, a12); a02 = muladd(vl21, a01, a02); a12 = muladd(vl21, a11, a12); a02 *= vd2; a12 *= vd2
                a03 = muladd(vl30, a00, a03); a13 = muladd(vl30, a10, a13); a03 = muladd(vl31, a01, a03); a13 = muladd(vl31, a11, a13); a03 = muladd(vl32, a02, a03); a13 = muladd(vl32, a12, a13); a03 *= vd3; a13 *= vd3
                vstore(a00, _cvptr(pT, i, c0, ldt));     vstore(a01, _cvptr(pT, i, c0 + 1, ldt))
                vstore(a02, _cvptr(pT, i, c0 + 2, ldt)); vstore(a03, _cvptr(pT, i, c0 + 3, ldt))
                vstore(a10, _cvptr(pT, r1, c0, ldt));     vstore(a11, _cvptr(pT, r1, c0 + 1, ldt))
                vstore(a12, _cvptr(pT, r1, c0 + 2, ldt)); vstore(a13, _cvptr(pT, r1, c0 + 3, ldt))
                i += 2W
            end
            while i + W - 1 <= m                             # MR=1 fused tier (4 accumulators)
                a0 = vload(V, _cvptr(psrc, i, c0, lds)); a1 = vload(V, _cvptr(psrc, i, c0 + 1, lds)); a2 = vload(V, _cvptr(psrc, i, c0 + 2, lds)); a3 = vload(V, _cvptr(psrc, i, c0 + 3, lds))
                for k in 1:(c0 - 1)
                    vk = vload(V, _cvptr(pT, i, k, ldt))
                    a0 = muladd(V(-unsafe_load(p00, _clidx(c0, k, ld0))), vk, a0)
                    a1 = muladd(V(-unsafe_load(p00, _clidx(c0 + 1, k, ld0))), vk, a1)
                    a2 = muladd(V(-unsafe_load(p00, _clidx(c0 + 2, k, ld0))), vk, a2)
                    a3 = muladd(V(-unsafe_load(p00, _clidx(c0 + 3, k, ld0))), vk, a3)
                end
                a0 *= vd0
                a1 = muladd(vl10, a0, a1); a1 *= vd1
                a2 = muladd(vl20, a0, a2); a2 = muladd(vl21, a1, a2); a2 *= vd2
                a3 = muladd(vl30, a0, a3); a3 = muladd(vl31, a1, a3); a3 = muladd(vl32, a2, a3); a3 *= vd3
                vstore(a0, _cvptr(pT, i, c0, ldt));     vstore(a1, _cvptr(pT, i, c0 + 1, ldt))
                vstore(a2, _cvptr(pT, i, c0 + 2, ldt)); vstore(a3, _cvptr(pT, i, c0 + 3, ldt))
                i += W
            end
            if i <= m                                        # MASKED tail (<W rows) — the MR=1 body, one pass
                msk = Vec(ntuple(l -> l, Val(W))) <= (m - i + 1)          # inactive lanes are never accessed (no OOB)
                a0 = vload(V, _cvptr(psrc, i, c0, lds), msk);     a1 = vload(V, _cvptr(psrc, i, c0 + 1, lds), msk)
                a2 = vload(V, _cvptr(psrc, i, c0 + 2, lds), msk); a3 = vload(V, _cvptr(psrc, i, c0 + 3, lds), msk)
                for k in 1:(c0 - 1)
                    vk = vload(V, _cvptr(pT, i, k, ldt), msk)
                    a0 = muladd(V(-unsafe_load(p00, _clidx(c0, k, ld0))), vk, a0)
                    a1 = muladd(V(-unsafe_load(p00, _clidx(c0 + 1, k, ld0))), vk, a1)
                    a2 = muladd(V(-unsafe_load(p00, _clidx(c0 + 2, k, ld0))), vk, a2)
                    a3 = muladd(V(-unsafe_load(p00, _clidx(c0 + 3, k, ld0))), vk, a3)
                end
                a0 *= vd0
                a1 = muladd(vl10, a0, a1); a1 *= vd1
                a2 = muladd(vl20, a0, a2); a2 = muladd(vl21, a1, a2); a2 *= vd2
                a3 = muladd(vl30, a0, a3); a3 = muladd(vl31, a1, a3); a3 = muladd(vl32, a2, a3); a3 *= vd3
                vstore(a0, _cvptr(pT, i, c0, ldt), msk);     vstore(a1, _cvptr(pT, i, c0 + 1, ldt), msk)
                vstore(a2, _cvptr(pT, i, c0 + 2, ldt), msk); vstore(a3, _cvptr(pT, i, c0 + 3, ldt), msk)
            end
        else
            i = 1
            while i <= m                                          # nb<4 remainder, masked over ROWS
                msk = Vec(ntuple(l -> l, Val(W))) <= (m - i + 1)
                for dj in 0:(nb - 1)
                    cc = c0 + dj; s = vload(V, _cvptr(psrc, i, cc, lds), msk)
                    for k in 1:(cc - 1)
                        s = muladd(V(-unsafe_load(p00, _clidx(cc, k, ld0))), vload(V, _cvptr(pT, i, k, ldt), msk), s)
                    end
                    vstore(s / V(unsafe_load(p00, _clidx(cc, cc, ld0))), _cvptr(pT, i, cc, ldt), msk)
                end
                i += W
            end
        end
        c0 += _fh_chol_nb()
    end
    return nothing
end

# Split-ld faer syrk column j: A11[i,j] (at ld1) −= Σ_c T[j,c]·T[i,c] (T at ldt).
@inline function _syrk_panel_split_f64!(p11::Ptr{T}, ld1::Int, pT::Ptr{T}, ldt::Int, j::Int, m::Int, bs::Int) where {T}
    W = _vwidth(T); V = Vec{W, T}
    i = ((j - 1) ÷ W) * W + 1
    @inbounds while i + W - 1 <= m
        b = _cvptr(p11, i, j, ld1); a = vload(V, b)
        for c in 1:bs
            a = muladd(V(-unsafe_load(pT, _clidx(j, c, ldt))), vload(V, _cvptr(pT, i, c, ldt)), a)
        end
        vstore(a, b); i += W
    end
    @inbounds if i <= m                              # MASKED tail — the body above, one pass
        msk = Vec(ntuple(l -> l, Val(W))) <= (m - i + 1)   # inactive lanes are never accessed (no OOB)
        b = _cvptr(p11, i, j, ld1); a = vload(V, b, msk)
        for c in 1:bs
            a = muladd(V(-unsafe_load(pT, _clidx(j, c, ldt))), vload(V, _cvptr(pT, i, c, ldt), msk), a)
        end
        vstore(a, b, msk)
    end
    return nothing
end

# Split-ld faer trailing update: A11 (m×m at ld1, po2 is fine — registers carry the RMW across the
# k-loop) −= T·Tᵀ with the PANEL read from conflict-free T at ldt. Body = _syrk_lower_f64! with the
# two operands' lds split.
@inline function _syrk_lower_split_f64!(
        p11::Ptr{T}, ld1::Int, pT::Ptr{T}, ldt::Int,
        m::Int, bs::Int
    ) where {T}
    W = _vwidth(T); V = Vec{W, T}
    j = 1
    @inbounds while j + _fh_chol_nc() - 1 <= m
        i = ((j - 1) ÷ W) * W + 1
        while i + 3W - 1 <= m                          # MR=3 × NC=4 = 12 accumulators
            r1 = i + W; r2 = i + 2W
            e00 = _cvptr(p11, i, j, ld1);      A00 = vload(V, e00)
            e10 = _cvptr(p11, r1, j, ld1);     C00 = vload(V, e10)
            e20 = _cvptr(p11, r2, j, ld1);     D00 = vload(V, e20)
            e01 = _cvptr(p11, i, j + 1, ld1);  A01 = vload(V, e01)
            e11 = _cvptr(p11, r1, j + 1, ld1); C01 = vload(V, e11)
            e21 = _cvptr(p11, r2, j + 1, ld1); D01 = vload(V, e21)
            e02 = _cvptr(p11, i, j + 2, ld1);  A02 = vload(V, e02)
            e12 = _cvptr(p11, r1, j + 2, ld1); C02 = vload(V, e12)
            e22 = _cvptr(p11, r2, j + 2, ld1); D02 = vload(V, e22)
            e03 = _cvptr(p11, i, j + 3, ld1);  A03 = vload(V, e03)
            e13 = _cvptr(p11, r1, j + 3, ld1); C03 = vload(V, e13)
            e23 = _cvptr(p11, r2, j + 3, ld1); D03 = vload(V, e23)
            for c in 1:bs
                v0 = vload(V, _cvptr(pT, i, c, ldt)); v1 = vload(V, _cvptr(pT, r1, c, ldt)); v2 = vload(V, _cvptr(pT, r2, c, ldt))
                g0 = V(-unsafe_load(pT, _clidx(j, c, ldt)));     A00 = muladd(g0, v0, A00); C00 = muladd(g0, v1, C00); D00 = muladd(g0, v2, D00)
                g1 = V(-unsafe_load(pT, _clidx(j + 1, c, ldt))); A01 = muladd(g1, v0, A01); C01 = muladd(g1, v1, C01); D01 = muladd(g1, v2, D01)
                g2 = V(-unsafe_load(pT, _clidx(j + 2, c, ldt))); A02 = muladd(g2, v0, A02); C02 = muladd(g2, v1, C02); D02 = muladd(g2, v2, D02)
                g3 = V(-unsafe_load(pT, _clidx(j + 3, c, ldt))); A03 = muladd(g3, v0, A03); C03 = muladd(g3, v1, C03); D03 = muladd(g3, v2, D03)
            end
            vstore(A00, e00); vstore(A01, e01); vstore(A02, e02); vstore(A03, e03)
            vstore(C00, e10); vstore(C01, e11); vstore(C02, e12); vstore(C03, e13)
            vstore(D00, e20); vstore(D01, e21); vstore(D02, e22); vstore(D03, e23)
            i += 3W
        end
        while i + 2W - 1 <= m                          # MR=2 × NC=4 = 8 accumulators
            r1 = i + W
            d00 = _cvptr(p11, i, j, ld1);      A00 = vload(V, d00)
            d10 = _cvptr(p11, r1, j, ld1);     B00 = vload(V, d10)
            d01 = _cvptr(p11, i, j + 1, ld1);  A01 = vload(V, d01)
            d11 = _cvptr(p11, r1, j + 1, ld1); B01 = vload(V, d11)
            d02 = _cvptr(p11, i, j + 2, ld1);  A02 = vload(V, d02)
            d12 = _cvptr(p11, r1, j + 2, ld1); B02 = vload(V, d12)
            d03 = _cvptr(p11, i, j + 3, ld1);  A03 = vload(V, d03)
            d13 = _cvptr(p11, r1, j + 3, ld1); B03 = vload(V, d13)
            for c in 1:bs
                v0 = vload(V, _cvptr(pT, i, c, ldt)); v1 = vload(V, _cvptr(pT, r1, c, ldt))
                g0 = V(-unsafe_load(pT, _clidx(j, c, ldt)));     A00 = muladd(g0, v0, A00); B00 = muladd(g0, v1, B00)
                g1 = V(-unsafe_load(pT, _clidx(j + 1, c, ldt))); A01 = muladd(g1, v0, A01); B01 = muladd(g1, v1, B01)
                g2 = V(-unsafe_load(pT, _clidx(j + 2, c, ldt))); A02 = muladd(g2, v0, A02); B02 = muladd(g2, v1, B02)
                g3 = V(-unsafe_load(pT, _clidx(j + 3, c, ldt))); A03 = muladd(g3, v0, A03); B03 = muladd(g3, v1, B03)
            end
            vstore(A00, d00); vstore(A01, d01); vstore(A02, d02); vstore(A03, d03)
            vstore(B00, d10); vstore(B01, d11); vstore(B02, d12); vstore(B03, d13)
            i += 2W
        end
        while i + W - 1 <= m
            b0 = _cvptr(p11, i, j, ld1);     a0 = vload(V, b0)
            b1 = _cvptr(p11, i, j + 1, ld1); a1 = vload(V, b1)
            b2 = _cvptr(p11, i, j + 2, ld1); a2 = vload(V, b2)
            b3 = _cvptr(p11, i, j + 3, ld1); a3 = vload(V, b3)
            for c in 1:bs
                lic = vload(V, _cvptr(pT, i, c, ldt))
                a0 = muladd(V(-unsafe_load(pT, _clidx(j, c, ldt))), lic, a0)
                a1 = muladd(V(-unsafe_load(pT, _clidx(j + 1, c, ldt))), lic, a1)
                a2 = muladd(V(-unsafe_load(pT, _clidx(j + 2, c, ldt))), lic, a2)
                a3 = muladd(V(-unsafe_load(pT, _clidx(j + 3, c, ldt))), lic, a3)
            end
            vstore(a0, b0); vstore(a1, b1); vstore(a2, b2); vstore(a3, b3); i += W
        end
        if i <= m                                        # MASKED tail — the MR=1 body, one pass
            msk = Vec(ntuple(l -> l, Val(W))) <= (m - i + 1)   # inactive lanes are never accessed (no OOB)
            b0 = _cvptr(p11, i, j, ld1);     a0 = vload(V, b0, msk)
            b1 = _cvptr(p11, i, j + 1, ld1); a1 = vload(V, b1, msk)
            b2 = _cvptr(p11, i, j + 2, ld1); a2 = vload(V, b2, msk)
            b3 = _cvptr(p11, i, j + 3, ld1); a3 = vload(V, b3, msk)
            for c in 1:bs
                lic = vload(V, _cvptr(pT, i, c, ldt), msk)
                a0 = muladd(V(-unsafe_load(pT, _clidx(j, c, ldt))), lic, a0)
                a1 = muladd(V(-unsafe_load(pT, _clidx(j + 1, c, ldt))), lic, a1)
                a2 = muladd(V(-unsafe_load(pT, _clidx(j + 2, c, ldt))), lic, a2)
                a3 = muladd(V(-unsafe_load(pT, _clidx(j + 3, c, ldt))), lic, a3)
            end
            vstore(a0, b0, msk); vstore(a1, b1, msk); vstore(a2, b2, msk); vstore(a3, b3, msk)
        end
        j += _fh_chol_nc()
    end
    while j <= m
        _syrk_panel_split_f64!(p11, ld1, pT, ldt, j, m, bs); j += 1
    end
    return nothing
end

# Owned conflict-free scratches (GKH ownership; single-thread — MT deferred project-wide). The 3 buffers
# live in the per-T L3Workspace (see workspace.jl); accessors below grow them on demand.
# _chol_mc: trsm row chunk — the mc×NB T slab the k-repasses re-read stays L2-resident (slab ≤ L2/2).
@inline _chol_mc(::Type{T}) where {T} = max(_vwidth(T), (_L2_BYTES ÷ 2) ÷ (_chol_block(T) * sizeof(T)))

function _chol_d(::Type{T}) where {T}   # diag block scratch, (_chol_block+8)×_chol_block
    ws = _l3ws(T); b = ws.chold; nb = _chol_block(T)
    if size(b, 1) < nb + 8 || size(b, 2) < nb
        b = Matrix{T}(undef, nb + 8, nb); ws.chold = b
    end
    return b
end
function _chol_t(::Type{T}, R::Int) where {T}   # panel workspace, R×_chol_block
    ws = _l3ws(T); b = ws.cholt; nb = _chol_block(T)
    if size(b, 1) < R || size(b, 2) < nb
        b = Matrix{T}(undef, R, nb); ws.cholt = b
    end
    return b
end
function _chol_pad(::Type{T}, R::Int, n::Int) where {T}   # faer whole-matrix pad, ld=R (=n+8)
    ws = _l3ws(T); b = ws.cholpad
    if size(b, 1) < R || size(b, 2) < n
        b = Matrix{T}(undef, R, n); ws.cholpad = b
    end
    return b
end

function _chol_panel_f64!(A, n::Int, blk::Int = _chol_block(eltype(A)))
    T = eltype(A)
    lda = stride(A, 2)
    R = (n + 8) % 128 == 0 ? n + 16 : n + 8                    # keep ldT itself alias-free
    Tb = _chol_t(T, R)
    ldT = size(Tb, 1); D = _chol_d(T); ldD = size(D, 1)
    GC.@preserve A Tb D begin
        pa = pointer(A); pT = pointer(Tb); pD = pointer(D)
        j = 0
        @inbounds while j < n
            bs = min(blk, n - j)
            pjj = _cvptr(pa, j + 1, j + 1, lda)
            for c in 0:(bs - 1)                                   # diag block lower triangle → D (L1/L2)
                unsafe_copyto!(pD + (c * ldD + c) * sizeof(T), pjj + (c * lda + c) * sizeof(T), bs - c)
            end
            let f = _chol_rl_f64!(pD, bs, ldD, _chol_sb(T), _chol_sth(T))
                f == 0 || throw(PosDefException(j + f))    # j = 0-based block offset, f = column within it
            end
            for c in 0:(bs - 1)                                   # factored diag back (tiny)
                unsafe_copyto!(pjj + (c * lda + c) * sizeof(T), pD + (c * ldD + c) * sizeof(T), bs - c)
            end
            m = n - j - bs
            if m > 0
                p21 = _cvptr(pa, j + bs + 1, j + 1, lda)
                i0 = 0                                        # fused panel solve → T, MC row chunks
                while i0 < m
                    mc = min(_chol_mc(T), m - i0)
                    _trsm_rl_split_f64!(pD, ldD, p21 + i0 * sizeof(T), lda, pT + i0 * sizeof(T), ldT, bs, mc)
                    i0 += mc
                end
                p22 = _cvptr(pa, j + bs + 1, j + bs + 1, lda)
                if m * bs * sizeof(T) <= _L2_BYTES ÷ 2       # T slab L2-resident: fused inline syrk
                    _syrk_lower_split_f64!(p22, lda, pT, ldT, m, bs)
                else                                          # big trailing: cache-blocked syrk! reads T
                    syrk!(
                        view(A, (j + bs + 1):n, (j + bs + 1):n), view(Tb, 1:m, 1:bs);
                        uplo = 'L', trans = 'N', alpha = -1, beta = 1
                    )
                end
                for c in 0:(bs - 1)                               # stream the factor back to A21 ONCE
                    unsafe_copyto!(p21 + c * lda * sizeof(T), pT + c * ldT * sizeof(T), m)
                end
            end
            j += bs
        end
    end
    return A
end

function _potrf_f64_lower!(A, base::Int = _chol_faer_base(eltype(A)))
    T = eltype(A)
    n = size(A, 1)
    n == 0 && return A
    if _NVREG == 16 && n > _chol_rl_max(T)
        # AVX2: the fused panel driver beats the hybrid/whole-pad path at EVERY size (measured Zen3,
        # 200–4000: transition dips 384/448/640 0.91-0.94→1.01-1.03, non-po2 large 0.98→1.00-1.03). It was
        # originally gated to po2-aliased strides only (its raison d'être was dodging the po2 pad round-trip),
        # but it's a better-composed blocked driver everywhere — the hybrid's generic trsm!(side=R,transA=T)
        # is the side-R-T laggard the panel driver's fused split-ld trsm avoids. (AVX-512 W=8 stays below.)
        return _chol_panel_f64!(A, n)
    end
    # AVX2 reaches here only for n ≤ _CHOL_RL_MAX (rl32 regime): rl32's small 32-blocks are alias-tolerant
    # (measured Zen3: rl32-direct ≥ pad on every po2 stride/subview in-range, +7–8% at po2-128) so the pad
    # is dead weight — skip it. W=8 still pads (its larger rl blocks' po2-tolerance is unmeasured).
    if _NVREG != 16 && _chol_needs_pad(A, n)      # factor in a non-conflicting (ld = n+8) scratch, copy back
        R = n + 8
        b = _chol_pad(T, R, n)
        Mw = view(b, 1:n, 1:n)
        # explicit contiguous per-column copies — copyto! on SubArrays is elementwise (the LU pad lesson)
        lda = stride(A, 2); ldb = size(b, 1)
        GC.@preserve A b begin
            pa = pointer(A); pb = pointer(b)
            # Lower triangle only: the faer lower path reads/writes exclusively the lower triangle + diagonal
            # (base kernel loads rows ≥ j, cols < j), and the scratch upper is never-read workspace. Copying
            # column j from its diagonal down (n-j elts) halves the copy — the copy is the WHOLE pad overhead
            # (~16 MB at n=1024), so this lifts the po2-input gate directly (2n²→n² moved).
            @inbounds for j in 0:(n - 1)
                unsafe_copyto!(pb + (j * ldb + j) * sizeof(T), pa + (j * lda + j) * sizeof(T), n - j)
            end
            f = _chol_hyb_f64!(Mw, n, base)
            f == 0 || throw(PosDefException(f))            # throw BEFORE copy-back: A stays untouched
            @inbounds for j in 0:(n - 1)
                unsafe_copyto!(pa + (j * lda + j) * sizeof(T), pb + (j * ldb + j) * sizeof(T), n - j)
            end
        end
    else
        f = _chol_hyb_f64!(A, n, base)
        f == 0 || throw(PosDefException(f))
    end
    return A
end

# Recursive right-looking blocked complex Hermitian Cholesky (lower). Panel width nb ∝ n/4 (OpenBLAS's own
# policy — see potrf_L_single.c: nb = min(n/4, GEMM_Q)), capped at `_CPOTRF_NBMAX`. This keeps the panel
# COUNT ~constant (≈4) and block shapes SCALING CONTINUOUSLY with n — the two things that make the curve
# smooth. A fixed nb=32 instead made panel count jump 2→4→8→16 (large-n falloff: AVX2 0.82 at n≥1024) and
# a big base cutoff added a discrete base→blocked step; nb=n/4 removes both. Per panel: factor the diagonal
# jb-block RECURSIVELY (jb>base ⇒ blocks again; jb≤base ⇒ the vectorized unblocked base), trsm side-R 'C'
# panel solve, herk 'N' rank-jb trailing downdate — all gating L3. BlasComplex only (Dual/upper → generic).
# PDM: Derived — formula over detected consts: `_at_cpotrf_nbmax(_HW)`
const _CPOTRF_NBMAX = @load_preference("cpotrf_nbmax", _at_cpotrf_nbmax(_HW))::Int   # req#8: derived 64+16·W (128/192)
@inline _chol_nb(n::Int) = clamp((n >> 2) & ~15, 32, _CPOTRF_NBMAX)     # ~n/4, rounded to a multiple of 16
function _cpotrf_lower!(A, n::Int)
    n <= _CPOTRF_BASE && return _potf2b_lower!(A, n)                    # unblocked vectorized base
    nb = _chol_nb(n)
    j = 1
    @inbounds while j <= n
        jb = min(nb, n - j + 1)
        f = _cpotrf_lower!(view(A, j:(j + jb - 1), j:(j + jb - 1)), jb) # factor diagonal jb-block (recurse)
        f == 0 || return j - 1 + f                                      # lift: block starts at column j
        if j + jb <= n
            db = view(A, j:(j + jb - 1), j:(j + jb - 1)); pan = view(A, (j + jb):n, j:(j + jb - 1))
            trsm!(pan, db; side = 'R', uplo = 'L', transA = 'C', diag = 'N', alpha = true)  # L21 = A21·L11⁻ᴴ
            herk!(view(A, (j + jb):n, (j + jb):n), pan; uplo = 'L', trans = 'N', alpha = -1.0, beta = 1.0)  # A22 -= L21·L21ᴴ
        end
        j += jb
    end
    return 0
end

# Public: Cholesky factor A in place into its `uplo` triangle. Returns A. Throws PosDefException if A is
# not positive definite. Float64 lower → faer fast path; complex lower → right-looking blocked; else the
# generic AD-traceable recursion.
function potrf!(A::AbstractMatrix; uplo::Char = 'L')
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("potrf!: A must be square"))
    if uplo == 'L' && _strided1(A)
        (eltype(A) === Float64 || eltype(A) === Float32) && return _potrf_f64_lower!(A)
        # n≤base: one vectorized base call (fast ≤64, single contiguous factor). n>base: right-looking
        # blocked (small-nb panels → big amortizing trailing herks). Splitting n≤64 into panels regressed it.
        if eltype(A) <: BlasComplex                                    # recursive nb=n/4 (base handled inside)
            f = _cpotrf_lower!(A, n)
            f == 0 || throw(PosDefException(f))
            return A
        end
    end
    base = eltype(A) <: Complex ? _CPOTRF_BASE : _potrf_base(eltype(A))   # complex→_CPOTRF_BASE; F32→halved (Lever B); F64/Dual→_POTRF_BASE
    _potrf_gen!(A, n, base, uplo != 'L')                       # pads po2-aliased strides (see _potrf_needs_pad)
    return A
end
