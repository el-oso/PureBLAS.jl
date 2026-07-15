# BLAS Level-3 GEMM: C := alpha·op(A)·op(B) + beta·C  (column-major, op ∈ {N,T,C}).
#
# Architecture = the GotoBLAS/BLIS 5-loop with packing (see ROADMAP M2): pack A and B into
# microkernel-friendly panels (zero-padded to mr/nr so edges need no special microkernel), then a
# register-blocked SIMD.jl microkernel accumulates an mr×nr C tile. alpha is folded into the A pack;
# beta is applied once up front. Real (Float32/Float64) take the blocked path; complex / Dual / any
# other T<:Number, and non-contiguous C, take the generic triple-loop (correct + AD-traceable).
#
# Block sizes are a first cut for Zen4 (AVX-512, 8 Float64/vec) — tune via the calibration knobs.
# ponytail: blocks hand-set for Zen4; lift into Preferences when tuning the fleet.

# Register-blocked microkernel tile: _MR vector-rows × _NR columns = _MR*_NR vector accumulators.
# The k-loop holds _MR*_NR accumulators + _MR A-vectors + 1 B-broadcast live at once, so the tile
# must satisfy  _MR*_NR + _MR + 1 ≲ (vector registers).  Register count differs by ISA, so the tile
# is WIDTH-ADAPTIVE (keyed on the Float64 lane count as an ISA proxy), Preferences-overridable per
# host (the fleet calibration knob — see ROADMAP M7/Zen3 tuning):
#   W64=8  AVX-512, 32 zmm : 2×8 = 16 accs  (Zen4/Zen5 — the tuned sweet spot, UNCHANGED)
#   W64=4  AVX2,     16 ymm: 3×4 = 12 accs  (Zen3/galen — 2×8 spilled; 3×4 gates, galen-swept 2026-07-02)
#   W64=2  NEON,     32 regs: 2×8            (placeholder; tune on M5 later)
# ponytail: per-width defaults below; override via Preferences "gemm_mr"/"gemm_nr" to sweep.
const _W64 = _vwidth(Float64)
# req#8: derived from detected hardware (cpuinfo.jl `_at_gemm_*`). Reproduce the fleet's tuned literals —
# MR/NR: galen 3/4, Zen4/Zen5 2/8 (behavior-preserving, was `_W64==4 ? …`). KC/MC/NC: Zen4 256/144/2040
# (bit-identical), galen 512/72/2044, Zen5 384/96/1360 — the KC=512 raises galen's B-micropanel from ¼ to
# ½ L1 (the "AVX-512-tile-on-AVX2" residency miss). Real-path (Float64/_NR); complex gemm uses `_CKC`.
const _MR = @load_preference("gemm_mr", _at_gemm_mr(_HW))::Int
const _NR = @load_preference("gemm_nr", _at_gemm_nr(_HW))::Int
# mc is derived per-CALLER from the LOCAL kc + element type via `_at_mc_kc` (joint residency
# mc·kc·sizeof ≤ 30%·L2), not a single const — a standalone `_MC` bakes the canonical kc and
# under-blocks small-kc callers (potrf's trailing gemm) / mis-sizes complex (16 B/elt). req#8.
const _NC = @load_preference("gemm_nc", _at_gemm_nc(_HW))::Int   # B col block ≤ ¼·L3, po2-dodged
const _KC = @load_preference("gemm_kc", _at_gemm_kc(_HW))::Int   # B micropanel kc·_NR·8 ≤ ½·L1 (BLIS)
# Short-k split-reduction tile (cpuinfo.jl `_at_gemm_split_*`): tall _SMR·W×_SNR tile, S-way k-split, for the
# small-n window where the wide tile under-fills. _SPLIT_OK const-folds the whole path off on AVX2 (already ≥1.0
# there via _GEMM_MR1_MAX; would spill). req#8; measured n=32 0.85→1.07× OB (wintermute).
const _SPLIT_OK  = _at_gemm_split_ok(_HW)
const _SMR = _at_gemm_split_mr(_HW)
const _SNR = _at_gemm_split_nr(_HW)
const _GEMM_SPLIT_MAX = @load_preference("gemm_split_max", _at_gemm_split_max(_HW))::Int

# Software prefetch hint (read, high locality, data cache) via the LLVM intrinsic. Used to pull the
# C output tile toward cache at microkernel entry so the read-modify-write at the end (cold C from
# memory for large n, reloaded once per kc-block) overlaps the k-loop instead of stalling.
@inline function _prefetch(p::Ptr{T}) where {T}
    Base.llvmcall(
        ("""
         declare void @llvm.prefetch.p0(ptr, i32, i32, i32)
         define void @entry(ptr %p) #0 {
           call void @llvm.prefetch.p0(ptr %p, i32 0, i32 3, i32 1)
           ret void
         }
         attributes #0 = { alwaysinline }
         """, "entry"),
        Cvoid, Tuple{Ptr{T}}, p)
end
# (software prefetch tried and reverted — it regressed; the HW prefetcher already covers the
#  contiguous packed streams.)

# Register-blocked microkernel for a FULL mr×nr tile. @generated so the mr/nr fan-out is
# straight-line (literal indices — never index a tuple with a runtime var; that boxes/allocates).
# The k loop stays a runtime loop; its body is fully unrolled over (MR vectors)×(NR cols).
@generated function _microkernel!(C::Ptr{T}, ldc::Int, Ap::Ptr{T}, Bp::Ptr{T}, kc::Int,
        ::Val{MR}, ::Val{NR}, ::Val{B0} = Val(false)) where {T, MR, NR, B0}
    W = _vwidth(T); sz = sizeof(T); V = Vec{W, T}
    body = quote end
    # prefetch the C output tile (NR columns) so the cold read-modify-write at the end overlaps the
    # k-loop — matters at large n where C doesn't fit cache.
    for j in 1:NR
        push!(body.args, :(_prefetch(C + $(j - 1) * ldc * $sz)))
    end
    # accumulator init
    for mi in 1:MR, j in 1:NR
        push!(body.args, :($(Symbol(:c, mi, :_, j)) = zero($V)))
    end
    # k loop: load MR A-vectors, broadcast NR B-scalars, MR*NR FMAs
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
    # store: C[:, j] += accumulators  (B0 ⇒ overwrite: C[:, j] = accumulators, skip the C read)
    for j in 1:NR
        push!(body.args, :(colp = C + $(j - 1) * ldc * $sz))
        for mi in 1:MR
            cs = Symbol(:c, mi, :_, j)
            st = B0 ? :(vstore($cs, q)) : :(vstore(vload($V, q) + $cs, q))
            push!(body.args, :(let q = colp + $((mi - 1) * W * sz); $st; end))
        end
    end
    push!(body.args, :(return nothing))
    return body
end

# Clip kernel: a W-aligned partial row-tile (mre = VR·W < mr) reads the SAME mr-strided packed panel
# (PMR = _MR vectors per k-step) but computes/stores only the VR live row-vectors — clean, no mask, no
# wasted trailing-vector FMA. Closes the misaligned-m penalty (measured: aligned m ≈ 1.14× OB, m=32 with
# an 8-row=2·W remainder ≈ 0.97; the masked kernel computed _MR vectors to use VR and paid masked stores).
# Only for full columns (nre==nr); a column remainder still routes to the masked kernel.
@generated function _microkernel_clip!(C::Ptr{T}, ldc::Int, Ap::Ptr{T}, Bp::Ptr{T}, kc::Int,
        ::Val{PMR}, ::Val{VR}, ::Val{NR}, ::Val{B0} = Val(false)) where {T, PMR, VR, NR, B0}
    W = _vwidth(T); sz = sizeof(T); V = Vec{W, T}
    body = quote end
    for j in 1:NR
        push!(body.args, :(_prefetch(C + $(j - 1) * ldc * $sz)))
    end
    for mi in 1:VR, j in 1:NR
        push!(body.args, :($(Symbol(:c, mi, :_, j)) = zero($V)))
    end
    inner = quote end
    for mi in 1:VR
        push!(inner.args, :($(Symbol(:a, mi)) = vload($V, Ap + (p * $PMR + $(mi - 1)) * $(W * sz))))
    end
    for j in 1:NR
        push!(inner.args, :($(Symbol(:b, j)) = $V(unsafe_load(Bp + (p * $NR + $(j - 1)) * $sz))))
        for mi in 1:VR
            cs = Symbol(:c, mi, :_, j)
            push!(inner.args, :($cs = muladd($(Symbol(:a, mi)), $(Symbol(:b, j)), $cs)))
        end
    end
    push!(body.args, :(for p in 0:(kc - 1); $inner; end))
    for j in 1:NR
        push!(body.args, :(colp = C + $(j - 1) * ldc * $sz))
        for mi in 1:VR
            cs = Symbol(:c, mi, :_, j)
            st = B0 ? :(vstore($cs, q)) : :(vstore(vload($V, q) + $cs, q))
            push!(body.args, :(let q = colp + $((mi - 1) * W * sz); $st; end))
        end
    end
    push!(body.args, :(return nothing))
    return body
end

# Vectorized edge kernel for partial blocked tiles. The packed panels are zero-padded to mr×nr, so
# the FULL MR×NR compute is correct (padded rows/cols give zero accumulators); we only mask the
# accumulating store — rows via a SIMD mask (mre), columns via a guard (nre). Same register blocking
# as the full kernel; far faster than the old scalar fallback (which tanked non-multiple sizes,
# e.g. n=100 was 0.40× OpenBLAS).
@generated function _microkernel_masked!(C::Ptr{T}, ldc::Int, Ap::Ptr{T}, Bp::Ptr{T}, kc::Int,
        mre::Int, nre::Int, ::Val{MR}, ::Val{NR}, ::Val{B0} = Val(false)) where {T, MR, NR, B0}
    W = _vwidth(T); sz = sizeof(T); V = Vec{W, T}
    body = quote end
    lanetuple = Expr(:tuple, (0:(W - 1))...)
    push!(body.args, :(lanes = Vec{$W, Int}($lanetuple)))
    for mi in 1:MR
        push!(body.args, :($(Symbol(:msk, mi)) = (lanes + $((mi - 1) * W)) < mre))
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
    push!(body.args, :(for p in 0:(kc - 1); $inner; end))
    for j in 1:NR
        stores = quote end
        push!(stores.args, :(colp = C + $(j - 1) * ldc * $sz))
        for mi in 1:MR
            cs = Symbol(:c, mi, :_, j); mk = Symbol(:msk, mi)
            st = B0 ? :(vstore($cs, q, $mk)) : :(vstore(vload($V, q, $mk) + $cs, q, $mk))
            push!(stores.args, :(let q = colp + $((mi - 1) * W * sz); $st; end))
        end
        push!(body.args, :(if $(j - 1) < nre; $stores; end))
    end
    push!(body.args, :(return nothing))
    return body
end

# SIMD pack for tA='N', dense unit-column-stride A: each full mr-row panel column is a contiguous
# segment of A, so pack = vector load + scale + vector store. The scalar packing was the measured
# large-n bottleneck (pack_A ≈ 8.6 GB/s; this lifts it toward memory bandwidth). Partial last panel
# (zero-padded) stays scalar.
@inline function _pack_A_simd!(Ap::AbstractVector{T}, A, ic::Int, pc::Int, mce::Int,
        kce::Int, alpha::T, mr::Int) where {T}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T); lda = stride(A, 2)
    np = cld(mce, mr)
    GC.@preserve A Ap begin
        Aptr = pointer(A); App = pointer(Ap); av = V(alpha)
        @inbounds for pi in 0:(np - 1)
            base = pi * mr * kce
            r0 = ic + pi * mr
            if pi * mr + mr <= mce            # full panel — vectorized column copy + scale
                for p in 0:(kce - 1)
                    src = Aptr + (r0 + (pc + p) * lda) * sz
                    dst = App + (base + p * mr) * sz
                    o = 0
                    while o < mr
                        vstore(av * vload(V, src + o * sz), dst + o * sz)
                        o += W
                    end
                end
            else                              # partial panel — scalar, zero-padded
                for p in 0:(kce - 1)
                    for r in 0:(mr - 1)
                        lr = pi * mr + r
                        Ap[base + p * mr + r + 1] = lr < mce ? alpha * A[ic + lr + 1, pc + p + 1] : zero(T)
                    end
                end
            end
        end
    end
    return
end

# One W×W transpose-pack block, FULLY unrolled (no runtime tuple indexing): load W contiguous A-columns
# (`src + r·lda`), transpose in registers (shuffle butterfly), store the W transposed rows scaled by α
# at `dst + pp·mrstride`. @generated so the loads/shuffles/stores schedule statically.
@generated function _tblk!(dst::Ptr{T}, src::Ptr{T}, lda::Int, mrstride::Int, av::Vec{W, T}, ::Val{W}) where {T, W}
    sz = sizeof(T); V = Vec{W, T}; q = Int(round(log2(W)))
    body = quote end
    cur = [Symbol(:c_, r) for r in 0:(W - 1)]
    for r in 0:(W - 1)
        push!(body.args, :($(cur[r + 1]) = vload($V, src + $r * lda * $sz)))
    end
    for stage in 0:(q - 1)
        s = 1 << stage; nxt = [Symbol(:s, stage, :_, i) for i in 0:(W - 1)]
        for i in 0:(W - 1)
            if (i & s) == 0
                j = i | s
                lo = ntuple(e0 -> (e = e0 - 1; blk = e ÷ (2s); w = e % (2s); w < s ? blk * 2s + w : W + blk * 2s + (w - s)), W)
                hi = ntuple(e0 -> (e = e0 - 1; blk = e ÷ (2s); w = e % (2s); w < s ? blk * 2s + s + w : W + blk * 2s + s + (w - s)), W)
                push!(body.args, :($(nxt[i + 1]) = shufflevector($(cur[i + 1]), $(cur[j + 1]), Val($lo))))
                push!(body.args, :($(nxt[j + 1]) = shufflevector($(cur[i + 1]), $(cur[j + 1]), Val($hi))))
            end
        end
        cur = nxt
    end
    for pp in 0:(W - 1)
        push!(body.args, :(vstore(av * $(cur[pp + 1]), dst + $pp * mrstride * $sz)))
    end
    push!(body.args, :(return nothing))
    body
end

# SIMD transpose pack for tA='T' (op(A)=Aᵀ), dense unit-stride A, mr a multiple of W. op(A)[gi,gp] =
# A[gp,gi] would need A's ROWS (strided) — the old scalar fallback. Instead read W contiguous A-columns
# (op(A) rows), W contraction-rows at a time, transpose the W×W register block, and store the transposed
# rows into Ap. Partial panels / contraction tail stay scalar (zero-padded).
@inline function _pack_A_simd_T!(Ap::Vector{T}, A, ic::Int, pc::Int, mce::Int,
        kce::Int, alpha::T, mr::Int) where {T}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T); lda = stride(A, 2)
    np = cld(mce, mr); kfull = (kce ÷ W) * W; sub = mr ÷ W
    GC.@preserve A Ap begin
        Aptr = pointer(A); App = pointer(Ap); av = V(alpha)
        @inbounds for pi in 0:(np - 1)
            base = pi * mr * kce
            if pi * mr + mr <= mce                       # full panel
                for ri in 0:(sub - 1)
                    r0 = pi * mr + ri * W                 # op(A) row offset within the block
                    p = 0
                    while p < kfull                       # W×W transpose blocks
                        src = Aptr + ((ic + r0) * lda + (pc + p)) * sz   # A[pc+p, ic+r0]; col r at +r·lda
                        dst = App + (base + p * mr + ri * W) * sz        # row pp at +pp·mr
                        _tblk!(Ptr{T}(dst), Ptr{T}(src), lda, mr, av, Val(W))
                        p += W
                    end
                    while p < kce                          # contraction tail — scalar
                        for r in 0:(W - 1)
                            Ap[base + p * mr + ri * W + r + 1] = alpha * A[pc + p + 1, ic + r0 + r + 1]
                        end
                        p += 1
                    end
                end
            else                                          # partial panel — scalar, zero-padded
                for p in 0:(kce - 1), r in 0:(mr - 1)
                    lr = pi * mr + r
                    Ap[base + p * mr + r + 1] = lr < mce ? alpha * A[pc + p + 1, ic + lr + 1] : zero(T)
                end
            end
        end
    end
    return
end

# Transpose A (k×m, the transA='T' operand) into `At` (m×k, column-major) via the W×W SIMD block
# transpose — so the unpacked microkernel can run it as a plain N·N product (no B-packing). At[i,p]=A[p,i].
@inline function _transpose_dense!(At::Vector{T}, A, m::Int, k::Int) where {T}
    W = _vwidth(T); sz = sizeof(T); lda = stride(A, 2); ov = Vec{W, T}(one(T))
    mfull = (m ÷ W) * W; kfull = (k ÷ W) * W
    GC.@preserve A At begin
        Aptr = pointer(A); Tptr = pointer(At)
        i = 0
        @inbounds while i < mfull
            p = 0
            while p < kfull                              # W×W transpose block: A[p:p+W, i:i+W] → At[i:i+W, p:p+W]
                _tblk!(Ptr{T}(Tptr + (p * m + i) * sz), Ptr{T}(Aptr + (i * lda + p) * sz), lda, m, ov, Val(W))
                p += W
            end
            while p < k                                  # contraction tail for this W-row block — scalar
                for r in 0:(W - 1); At[p * m + i + r + 1] = A[p + 1, i + r + 1]; end
                p += 1
            end
            i += W
        end
        @inbounds while i < m                            # row tail (m not a multiple of W) — scalar
            for p in 0:(k - 1); At[p * m + i + 1] = A[p + 1, i + 1]; end
            i += 1
        end
    end
    return
end

# Pack a mc_eff×kc_eff block of op(A) into mr-row panels (zero-padded), scaling by alpha.
function _pack_A!(Ap::AbstractVector{T}, A, ic::Int, pc::Int, mce::Int, kce::Int, tA::Bool, alpha::T, mr::Int) where {T}
    if !tA && _strided1(A)
        return _pack_A_simd!(Ap, A, ic, pc, mce, kce, alpha, mr)
    end
    if tA && _strided1(A) && T <: BlasReal && mr % _vwidth(T) == 0
        return _pack_A_simd_T!(Ap, A, ic, pc, mce, kce, alpha, mr)
    end
    np = cld(mce, mr)
    @inbounds for pi in 0:(np - 1)
        base = pi * mr * kce
        for p in 0:(kce - 1)
            for r in 0:(mr - 1)
                lr = pi * mr + r
                Ap[base + p * mr + r + 1] = if lr < mce
                    gi = ic + lr; gp = pc + p
                    alpha * (tA ? A[gp + 1, gi + 1] : A[gi + 1, gp + 1])
                else
                    zero(T)
                end
            end
        end
    end
    return
end

# Pack a kc_eff×nc_eff block of op(B) into nr-col panels (zero-padded). `boff` lets the block be written
# into the middle of a larger panel buffer (used by the in-place trmm, which packs a whole B column-panel
# across all pc-blocks before overwriting B). AbstractVector so a `view` into that buffer is accepted.
function _pack_B!(Bp::AbstractVector{T}, B, pc::Int, jc::Int, kce::Int, nce::Int, tB::Bool, nr::Int, boff::Int = 0) where {T}
    np = cld(nce, nr)
    @inbounds for ji in 0:(np - 1)
        base = boff + ji * nr * kce
        if ji * nr + nr <= nce                       # full panel. Reads stay elementwise (a wide vload
            j0 = jc + ji * nr                        # stalls on freshly-written C operands — geqrf's
            if tB                                    # trailing update: measured −25% with _tblk!).
                # op(B)=Bᵀ: B[j0+c, gp] — for fixed gp the c-run is down a B column (contiguous), so the
                # p-outer/c-inner order already streams the reads; keep it (contiguous vector stores too).
                for p in 0:(kce - 1)
                    gp = pc + p
                    @simd ivdep for c in 0:(nr - 1)
                        Bp[base + p * nr + c + 1] = B[j0 + c + 1, gp + 1]
                    end
                end
            else
                # op(B)=B: B[gp, j0+c] — the contiguous B direction is down a column (p), so loop c-outer/
                # p-inner: each read-stream is one sequential B column (1 stream vs nr strided), stores
                # scatter into the hot Bp panel. Measured 1.3–1.35× faster pack than p-outer (galen).
                for c in 0:(nr - 1)
                    j = j0 + c
                    @simd ivdep for p in 0:(kce - 1)
                        Bp[base + p * nr + c + 1] = B[pc + p + 1, j + 1]
                    end
                end
            end
            continue
        end
        for p in 0:(kce - 1)
            for c in 0:(nr - 1)
                lc = ji * nr + c
                Bp[base + p * nr + c + 1] = if lc < nce
                    gj = jc + lc; gp = pc + p
                    tB ? B[gj + 1, gp + 1] : B[gp + 1, gj + 1]
                else
                    zero(T)
                end
            end
        end
    end
    return
end
function _scale_C!(C, m::Int, n::Int, beta::T) where {T}
    if iszero(beta)
        @inbounds for j in 1:n, i in 1:m
            C[i, j] = zero(T)
        end
    elseif !isone(beta)
        @inbounds for j in 1:n, i in 1:m
            C[i, j] *= beta
        end
    end
    return
end

# Reusable GEMM packing buffers now live in the per-type L3Workspace (see src/workspace.jl):
# `_gemm_scratch(T, lenA, lenB)` returns its `gpackA`/`gpackB` fields, grown on demand.

# ── Direct-B microkernels (op(B)=B, no B-pack) ───────────────────────────────────────────────
# A comes from the packed panel (contiguous, PMR vectors/k-step, VR live rows), but B is read STRAIGHT
# from the user array: B[pc+p, jc+jr+j] = Bc + p + j·ldb — contiguous in the k-index p (col-major B), so
# op(B)=B needs NO pack. Eliminates the entire B-pack pass (measured: n=192 0.93→0.98 vs AOCL on galen).
# alpha is folded in the A-pack (as for _microkernel!), so no alpha here. VR<PMR ⇒ clip (no mask).
@generated function _microkernel_db!(C::Ptr{T}, ldc::Int, Ap::Ptr{T}, Bc::Ptr{T}, ldb::Int, kc::Int,
        ::Val{PMR}, ::Val{VR}, ::Val{NR}, ::Val{B0}) where {T, PMR, VR, NR, B0}
    W = _vwidth(T); sz = sizeof(T); V = Vec{W, T}
    body = quote end
    for j in 1:NR; push!(body.args, :(_prefetch(C + $(j - 1) * ldc * $sz))); end
    for mi in 1:VR, j in 1:NR; push!(body.args, :($(Symbol(:c, mi, :_, j)) = zero($V))); end
    inner = quote end
    for mi in 1:VR
        push!(inner.args, :($(Symbol(:a, mi)) = vload($V, Ap + (p * $PMR + $(mi - 1)) * $(W * sz))))
    end
    for j in 1:NR
        push!(inner.args, :($(Symbol(:b, j)) = $V(unsafe_load(Bc + (p + $(j - 1) * ldb) * $sz))))
        for mi in 1:VR
            cs = Symbol(:c, mi, :_, j)
            push!(inner.args, :($cs = muladd($(Symbol(:a, mi)), $(Symbol(:b, j)), $cs)))
        end
    end
    push!(body.args, :(for p in 0:(kc - 1); $inner; end))
    for j in 1:NR
        push!(body.args, :(colp = C + $(j - 1) * ldc * $sz))
        for mi in 1:VR
            cs = Symbol(:c, mi, :_, j)
            st = B0 ? :(vstore($cs, q)) : :(vstore(vload($V, q) + $cs, q))
            push!(body.args, :(let q = colp + $((mi - 1) * W * sz); $st; end))
        end
    end
    push!(body.args, :(return nothing))
    return body
end

# Direct-B masked tile: partial rows (mre) via a SIMD mask, partial columns (nre) via a guard. Same
# packed-A / direct-B split as _microkernel_db!; mirrors _microkernel_masked!'s store masking.
@generated function _microkernel_db_masked!(C::Ptr{T}, ldc::Int, Ap::Ptr{T}, Bc::Ptr{T}, ldb::Int,
        kc::Int, mre::Int, nre::Int, ::Val{MR}, ::Val{NR}, ::Val{B0}) where {T, MR, NR, B0}
    W = _vwidth(T); sz = sizeof(T); V = Vec{W, T}
    body = quote end
    lanetuple = Expr(:tuple, (0:(W - 1))...)
    push!(body.args, :(lanes = Vec{$W, Int}($lanetuple)))
    for mi in 1:MR
        push!(body.args, :($(Symbol(:msk, mi)) = (lanes + $((mi - 1) * W)) < mre))
    end
    for mi in 1:MR, j in 1:NR; push!(body.args, :($(Symbol(:c, mi, :_, j)) = zero($V))); end
    inner = quote end
    for mi in 1:MR
        push!(inner.args, :($(Symbol(:a, mi)) = vload($V, Ap + (p * $MR + $(mi - 1)) * $(W * sz))))
    end
    for j in 1:NR
        push!(inner.args, :($(Symbol(:b, j)) = $V(unsafe_load(Bc + (p + $(j - 1) * ldb) * $sz))))
        for mi in 1:MR
            cs = Symbol(:c, mi, :_, j)
            push!(inner.args, :($cs = muladd($(Symbol(:a, mi)), $(Symbol(:b, j)), $cs)))
        end
    end
    push!(body.args, :(for p in 0:(kc - 1); $inner; end))
    for j in 1:NR
        stores = quote end
        push!(stores.args, :(colp = C + $(j - 1) * ldc * $sz))
        for mi in 1:MR
            cs = Symbol(:c, mi, :_, j); mk = Symbol(:msk, mi)
            st = B0 ? :(vstore($cs, q, $mk)) : :(vstore(vload($V, q, $mk) + $cs, q, $mk))
            push!(stores.args, :(let q = colp + $((mi - 1) * W * sz); $st; end))
        end
        push!(body.args, :(if $(j - 1) < nre; $stores; end))
    end
    push!(body.args, :(return nothing))
    return body
end

# One (jc,pc) sweep over the ic panels: pack A, run the microkernels over the nce×mce C-block.
# ::Val{B0} folds beta==0 on the FIRST kc-block (pc==0) into an OVERWRITE store — skips both the
# up-front _scale_C! zeroing AND the microkernel's C read-modify-write (~2× C traffic at small n,
# where k≤kc ⇒ a single kc-block; the measured small-n AOCL gap). Parity with the unpacked path.
# ::Val{DB} (op(B)=B and B unit row-stride): read B direct (Bp0/ldb), no B-pack — the compile-time
# branch const-folds; the packed side (DB=false) keeps Bpp for tB / strided-B.
function _blocked_pc_sweep!(::Val{B0}, ::Val{DB}, Cp0::Ptr{T}, App::Ptr{T}, Bpp::Ptr{T}, Bp0::Ptr{T},
        Ap, A, jc::Int, pc::Int, m::Int, nce::Int, kce::Int, mc::Int, mr::Int, nr::Int, W::Int,
        ldc::Int, ldb::Int, sz::Int, tA::Bool, alpha::T) where {T, B0, DB}
    ic = 0
    while ic < m
        mce = min(mc, m - ic)
        _pack_A!(Ap, A, ic, pc, mce, kce, tA, alpha, mr)
        jr = 0
        while jr < nce
            nre = min(nr, nce - jr)
            ir = 0
            while ir < mce
                mre = min(mr, mce - ir)
                Apanel = App + (div(ir, mr) * mr * kce) * sz
                Cblk = Cp0 + ((ic + ir) + (jc + jr) * ldc) * sz
                if DB
                    Bc = Bp0 + (pc + (jc + jr) * ldb) * sz   # B[pc, jc+jr]
                    if mre == mr && nre == nr
                        _microkernel_db!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bc), ldb,
                            kce, Val(_MR), Val(_MR), Val(_NR), Val(B0))
                    elseif nre == nr && rem(mre, W) == 0
                        vr = div(mre, W)
                        if vr == 1
                            _microkernel_db!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bc), ldb,
                                kce, Val(_MR), Val(1), Val(_NR), Val(B0))
                        elseif vr == 2
                            _microkernel_db!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bc), ldb,
                                kce, Val(_MR), Val(2), Val(_NR), Val(B0))
                        else
                            _microkernel_db_masked!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bc), ldb,
                                kce, mre, nre, Val(_MR), Val(_NR), Val(B0))
                        end
                    else
                        _microkernel_db_masked!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bc), ldb,
                            kce, mre, nre, Val(_MR), Val(_NR), Val(B0))
                    end
                else
                    Bpanel = Bpp + (div(jr, nr) * nr * kce) * sz
                    if mre == mr && nre == nr
                        _microkernel!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel),
                            kce, Val(_MR), Val(_NR), Val(B0))
                    elseif nre == nr && rem(mre, W) == 0   # W-aligned partial rows → clean clip (no mask)
                        vr = div(mre, W)
                        if vr == 1
                            _microkernel_clip!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel),
                                kce, Val(_MR), Val(1), Val(_NR), Val(B0))
                        elseif vr == 2
                            _microkernel_clip!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel),
                                kce, Val(_MR), Val(2), Val(_NR), Val(B0))
                        else
                            _microkernel_masked!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel),
                                kce, mre, nre, Val(_MR), Val(_NR), Val(B0))
                        end
                    else
                        _microkernel_masked!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel),
                            kce, mre, nre, Val(_MR), Val(_NR), Val(B0))
                    end
                end
                ir += mr
            end
            jr += nr
        end
        ic += mc
    end
    return nothing
end

# Direct-B tile dispatch (op(B)=B, unit row-stride: read B WITHOUT packing) — the _blocked_pc_sweep!
# DB ladder as a reusable @inline so symm side-L gets the same no-B-pack kernels (the dgemm small-n AOCL
# lever). `ow` ⇒ β=0 first-block overwrite. mre/nre are the live tile extents.
@inline function _db_tile!(ow::Bool, Cblk::Ptr{T}, ldc::Int, Apanel::Ptr{T}, Bc::Ptr{T}, ldb::Int,
        kce::Int, mre::Int, nre::Int, ::Val{MR}, ::Val{NR}, ::Val{W}) where {T, MR, NR, W}
    mr = MR * W
    if mre == mr && nre == NR
        ow ? _microkernel_db!(Cblk, ldc, Apanel, Bc, ldb, kce, Val(MR), Val(MR), Val(NR), Val(true)) :
             _microkernel_db!(Cblk, ldc, Apanel, Bc, ldb, kce, Val(MR), Val(MR), Val(NR), Val(false))
    elseif nre == NR && rem(mre, W) == 0
        vr = div(mre, W)
        if vr == 1
            ow ? _microkernel_db!(Cblk, ldc, Apanel, Bc, ldb, kce, Val(MR), Val(1), Val(NR), Val(true)) :
                 _microkernel_db!(Cblk, ldc, Apanel, Bc, ldb, kce, Val(MR), Val(1), Val(NR), Val(false))
        elseif vr == 2
            ow ? _microkernel_db!(Cblk, ldc, Apanel, Bc, ldb, kce, Val(MR), Val(2), Val(NR), Val(true)) :
                 _microkernel_db!(Cblk, ldc, Apanel, Bc, ldb, kce, Val(MR), Val(2), Val(NR), Val(false))
        else
            ow ? _microkernel_db_masked!(Cblk, ldc, Apanel, Bc, ldb, kce, mre, nre, Val(MR), Val(NR), Val(true)) :
                 _microkernel_db_masked!(Cblk, ldc, Apanel, Bc, ldb, kce, mre, nre, Val(MR), Val(NR), Val(false))
        end
    else
        ow ? _microkernel_db_masked!(Cblk, ldc, Apanel, Bc, ldb, kce, mre, nre, Val(MR), Val(NR), Val(true)) :
             _microkernel_db_masked!(Cblk, ldc, Apanel, Bc, ldb, kce, mre, nre, Val(MR), Val(NR), Val(false))
    end
end

# Blocked real GEMM (the optimized path). C must have unit column stride (pointer + vstore).
function _gemm_blocked!(tA::Bool, tB::Bool, m::Int, n::Int, k::Int,
        alpha::T, A, B, beta::T, C) where {T<:BlasReal}
    if iszero(alpha) || k == 0
        _scale_C!(C, m, n, beta)   # nothing to accumulate ⇒ C := βC (or 0)
        return C
    end
    b0first = iszero(beta)             # β=0 ⇒ overwrite on the first kc-block; no zeroing, no RMW read
    b0first || _scale_C!(C, m, n, beta)
    # op(B)=B with unit row-stride ⇒ read B direct in the microkernel (no B-pack, B is contiguous in k).
    db = !tB && _strided1(B)
    W = _vwidth(T); mr = _MR * W; nr = _NR
    # Cap block sizes by the actual problem so small GEMMs don't allocate/pack huge panels.
    kc = min(_KC, k)
    mc = _at_mc_kc(_HW, T, kc, mr, cld(m, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(n, nr) * nr)
    Ap, Bp = _gemm_scratch(T, cld(mc, mr) * mr * kc, db ? 0 : cld(nc, nr) * nr * kc)
    ldc = stride(C, 2); ldb = stride(B, 2); sz = sizeof(T)
    GC.@preserve C Ap Bp B begin
        Cp0 = pointer(C); App = pointer(Ap); Bpp = pointer(Bp); Bp0 = db ? pointer(B) : Bpp
        jc = 0
        while jc < n
            nce = min(nc, n - jc)
            pc = 0
            while pc < k
                kce = min(kc, k - pc)
                db || _pack_B!(Bp, B, pc, jc, kce, nce, tB, nr)
                b0 = b0first && pc == 0     # concrete-Val dispatch (trim-safe: no Union{Val}→Any)
                if db
                    if b0
                        _blocked_pc_sweep!(Val(true), Val(true), Cp0, App, Bpp, Bp0, Ap, A, jc, pc,
                            m, nce, kce, mc, mr, nr, W, ldc, ldb, sz, tA, alpha)
                    else
                        _blocked_pc_sweep!(Val(false), Val(true), Cp0, App, Bpp, Bp0, Ap, A, jc, pc,
                            m, nce, kce, mc, mr, nr, W, ldc, ldb, sz, tA, alpha)
                    end
                else
                    if b0
                        _blocked_pc_sweep!(Val(true), Val(false), Cp0, App, Bpp, Bp0, Ap, A, jc, pc,
                            m, nce, kce, mc, mr, nr, W, ldc, ldb, sz, tA, alpha)
                    else
                        _blocked_pc_sweep!(Val(false), Val(false), Cp0, App, Bpp, Bp0, Ap, A, jc, pc,
                            m, nce, kce, mc, mr, nr, W, ldc, ldb, sz, tA, alpha)
                    end
                end
                pc += kc
            end
            jc += nc
        end
    end
    return C
end

# ── Unpacked path for small matrices (BLASFEO-style size dispatch) ────────────────────────────
# For cache-resident matrices the O(n²) packing rivals the O(n³) compute, so we skip packing and run
# the register microkernel directly on column-major data: A's columns are contiguous (vector loads),
# B is strided scalar broadcasts. Requires tA='N' (transposed A would need strided loads). Reference:
# BLASFEO switches algorithms by matrix size (Frison et al., arXiv:1902.08115). See kb finding
# pureblas-gemm-performance.
# Use the unpacked path when max(m,n,k) ≤ this. Tuned on Zen4: unpacked (no packing, re-streams A
# from L2/L3) beats the blocked path until A no longer fits ~L2 (square n≈448; n=512 flips to
# blocked). ponytail: crude max() heuristic; a rectangular A (m·k fits but n huge) would also prefer
# unpacked — refine to an A-fits-L2 test if skewed shapes matter.
# Width-adaptive + Preferences-overridable (the unpacked path fits A in L2; the L2/register
# tradeoff differs by ISA — Zen4 flips at n≈448, Zen3/AVX2 loses earlier). Override "gemm_unpack_max".
# AVX2 re-swept 128→80 after the direct-B blocked path (no B-pack) became faster: the unpacked path now
# only wins n≤~64 (measured galen: n=64 unpk 1.05 vs blk 1.00; n=96 unpk 1.00 vs blk 1.04; n=128 unpk
# 0.89 vs blk 1.01 vs AOCL) — so n=96–128 now route to blocked-direct-B. AVX-512 unmeasured here (left 448).
const _GEMM_UNPACK_MAX = @load_preference("gemm_unpack_max", _W64 == 8 ? 448 : _W64 == 4 ? 80 : 192)::Int
@inline _use_unpacked(m, n, k) = max(m, n, k) <= _GEMM_UNPACK_MAX

# Unpacked microkernel: full mr×nr tile, reading A (tA='N') and op(B) directly. alpha applied at
# store (C already beta-scaled). TB selects op(B); a compile-time Val so the address folds.
@generated function _microkernel_unpacked!(C::Ptr{T}, ldc::Int, A::Ptr{T}, lda::Int, ir::Int,
        B::Ptr{T}, ldb::Int, jr::Int, k::Int, alpha::T, beta::T,
        ::Val{MR}, ::Val{NR}, ::Val{TB}, ::Val{B0}) where {T, MR, NR, TB, B0}
    W = _vwidth(T); sz = sizeof(T); V = Vec{W, T}
    body = quote end
    for mi in 1:MR, j in 1:NR
        push!(body.args, :($(Symbol(:c, mi, :_, j)) = zero($V)))
    end
    inner = quote end
    for mi in 1:MR
        push!(inner.args, :($(Symbol(:a, mi)) = vload($V, A + ((ir + $((mi - 1) * W)) + p * lda) * $sz)))
    end
    for j in 1:NR
        baddr = TB ? :(B + ((jr + $(j - 1)) + p * ldb) * $sz) : :(B + (p + (jr + $(j - 1)) * ldb) * $sz)
        push!(inner.args, :($(Symbol(:b, j)) = $V(unsafe_load($baddr))))
        for mi in 1:MR
            cs = Symbol(:c, mi, :_, j)
            push!(inner.args, :($cs = muladd($(Symbol(:a, mi)), $(Symbol(:b, j)), $cs)))
        end
    end
    push!(body.args, :(for p in 0:(k - 1); $inner; end))
    # store: C = alpha·acc (+ beta·C if beta≠0). beta folded here, so no separate scale-C pass; the
    # B0 (beta==0) path skips the C read entirely (and so ignores NaN in C, per BLAS).
    push!(body.args, :(av = $V(alpha)))
    B0 || push!(body.args, :(bv = $V(beta)))
    for j in 1:NR
        push!(body.args, :(colp = C + (ir + (jr + $(j - 1)) * ldc) * $sz))
        for mi in 1:MR
            cs = Symbol(:c, mi, :_, j)
            st = B0 ? :(vstore(av * $cs, q)) : :(vstore(muladd(bv, vload($V, q), av * $cs), q))
            push!(body.args, :(let q = colp + $((mi - 1) * W * sz); $st; end))
        end
    end
    push!(body.args, :(return nothing))
    return body
end

# Masked-row kernel: partial rows (mre<mr), exactly NR valid columns (all in bounds). Full MR×NR
# register blocking (A reused across columns, B across rows), masking only the row loads/stores.
# Called with NR=_NR for the full-column partial-row case (concrete, common) and NR=nre for the
# partial-column strip (NR small; see the dispatch barrier below). Folds beta (B0 ⇒ beta=0).
@generated function _microkernel_unpacked_mrows!(C::Ptr{T}, ldc::Int, A::Ptr{T}, lda::Int, ir::Int,
        B::Ptr{T}, ldb::Int, jr::Int, k::Int, alpha::T, beta::T, mre::Int,
        ::Val{MR}, ::Val{NR}, ::Val{TB}, ::Val{B0}) where {T, MR, NR, TB, B0}
    W = _vwidth(T); sz = sizeof(T); V = Vec{W, T}
    body = quote end
    lanetuple = Expr(:tuple, (0:(W - 1))...)   # literal (0,1,…,W-1); @generated body must be pure
    push!(body.args, :(lanes = Vec{$W, Int}($lanetuple)))
    for mi in 1:MR
        push!(body.args, :($(Symbol(:msk, mi)) = (lanes + $((mi - 1) * W)) < mre))
    end
    for mi in 1:MR, j in 1:NR
        push!(body.args, :($(Symbol(:c, mi, :_, j)) = zero($V)))
    end
    inner = quote end
    for mi in 1:MR
        push!(inner.args, :($(Symbol(:a, mi)) = vload($V, A + ((ir + $((mi - 1) * W)) + p * lda) * $sz, $(Symbol(:msk, mi)))))
    end
    for j in 1:NR
        baddr = TB ? :(B + ((jr + $(j - 1)) + p * ldb) * $sz) : :(B + (p + (jr + $(j - 1)) * ldb) * $sz)
        push!(inner.args, :($(Symbol(:b, j)) = $V(unsafe_load($baddr))))
        for mi in 1:MR
            cs = Symbol(:c, mi, :_, j)
            push!(inner.args, :($cs = muladd($(Symbol(:a, mi)), $(Symbol(:b, j)), $cs)))
        end
    end
    push!(body.args, :(for p in 0:(k - 1); $inner; end))
    push!(body.args, :(av = $V(alpha)))
    B0 || push!(body.args, :(bv = $V(beta)))
    for j in 1:NR
        push!(body.args, :(colp = C + (ir + (jr + $(j - 1)) * ldc) * $sz))
        for mi in 1:MR
            cs = Symbol(:c, mi, :_, j); mk = Symbol(:msk, mi)
            st = B0 ? :(vstore(av * $cs, q, $mk)) : :(vstore(muladd(bv, vload($V, q, $mk), av * $cs), q, $mk))
            push!(body.args, :(let q = colp + $((mi - 1) * W * sz); $st; end))
        end
    end
    push!(body.args, :(return nothing))
    return body
end

# Partial-COLUMN strip (nre<nr): masked-row, vectorized, one k-loop per (row-vector, column) — less
# A-reuse than the full kernel, but concrete-typed (tB/b0 plain Bool) so NO dynamic dispatch, and it
# touches exactly nre columns (no wasted compute). Measured the best option for the thin n-remainder
# strip on small matrices: Val(nre) dispatch and clamp/guard both lost to dispatch cost / waste.
@inline function _microkernel_unpacked_edge!(C::Ptr{T}, ldc::Int, A::Ptr{T}, lda::Int, ir::Int,
        B::Ptr{T}, ldb::Int, jr::Int, k::Int, alpha::T, beta::T, mre::Int, nre::Int,
        tB::Bool, b0::Bool) where {T}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T)
    lanes = Vec{W, Int}(ntuple(i -> i - 1, Val(W)))
    av = V(alpha); bv = V(beta)
    @inbounds for j in 0:(nre - 1)
        for row0 in 0:W:(mre - 1)
            mask = (lanes + row0) < mre
            acc = zero(V)
            for p in 0:(k - 1)
                a = vload(V, A + ((ir + row0) + p * lda) * sz, mask)
                bsc = tB ? unsafe_load(B, (jr + j) + p * ldb + 1) : unsafe_load(B, p + (jr + j) * ldb + 1)
                acc = muladd(a, V(bsc), acc)
            end
            q = C + ((ir + row0) + (jr + j) * ldc) * sz
            res = av * acc
            if b0
                vstore(res, q, mask)
            else
                vstore(muladd(bv, vload(V, q, mask), res), q, mask)
            end
        end
    end
    return nothing
end

# One FULL nr-column block at column jrc: the ir loop with full/masked-row dispatch (concrete Vals).
@inline function _unpacked_cols_full!(Cp::Ptr{T}, ldc::Int, Ap::Ptr{T}, lda::Int, Bp::Ptr{T},
        ldb::Int, m::Int, jrc::Int, k::Int, alpha::T, beta::T, mr::Int,
        ::Val{TB}, ::Val{B0}) where {T, TB, B0}
    W = _vwidth(T)
    ir = 0
    while ir < m
        mre = min(mr, m - ir)
        if mre == mr
            _microkernel_unpacked!(Cp, ldc, Ap, lda, ir, Bp, ldb, jrc, k, alpha, beta,
                Val(_MR), Val(_NR), Val(TB), Val(B0))
        elseif cld(mre, W) == 1
            _microkernel_unpacked_mrows!(Cp, ldc, Ap, lda, ir, Bp, ldb, jrc, k, alpha, beta,
                mre, Val(1), Val(_NR), Val(TB), Val(B0))
        else
            _microkernel_unpacked_mrows!(Cp, ldc, Ap, lda, ir, Bp, ldb, jrc, k, alpha, beta,
                mre, Val(_MR), Val(_NR), Val(TB), Val(B0))
        end
        ir += mr
    end
    return nothing
end

# Unpacked driver (tA='N'). TB and B0 (beta==0) are Vals so the microkernel specializes; resolved
# once at the boundary. beta is folded into the microkernel — no separate scale-C pass.
# Small-n unpacked driver with SINGLE-vector (W-row) tiles: below the full tile's row height (_MR·W) the
# matrix can't fill one full tile of rows, so its 16-acc setup doesn't amortize over short k. Same masked/
# edge kernels, Val(1) rows. DERIVED (req#8, was a bare 40): _at_gemm_mr·W → W=8=16 (measured: full tile now
# beats mr1 at n=32, 0.833→0.860). W=4→12: sweep-validate galen (measured 40) or pin "gemm_mr1_max"=40 there.
const _GEMM_MR1_MAX = @load_preference("gemm_mr1_max", _at_gemm_mr1_max(_HW))::Int
function _gemm_unpacked_mr1!(::Val{TB}, ::Val{B0}, m::Int, n::Int, k::Int,
        alpha::T, A, B, beta::T, C) where {T<:BlasReal, TB, B0}
    W = _vwidth(T); nr = _NR
    lda = stride(A, 2); ldb = stride(B, 2); ldc = stride(C, 2)
    # Preserve the parent arrays, not the (possibly SubArray) operands — GC.@preserve on a view forces
    # the otherwise-stack SubArray onto the heap to root it (64 B/call); pointers are computed anyway.
    parA = parent(A); parB = parent(B); parC = parent(C)
    GC.@preserve parA parB parC begin
        Ap = pointer(A); Bp = pointer(B); Cp = pointer(C)
        jr = 0
        while jr < n
            nre = min(nr, n - jr)
            ir = 0
            while ir < m
                mre = min(W, m - ir)
                if mre == W && nre == nr
                    _microkernel_unpacked!(Cp, ldc, Ap, lda, ir, Bp, ldb, jr, k, alpha, beta,
                        Val(1), Val(_NR), Val(TB), Val(B0))
                elseif nre == nr
                    _microkernel_unpacked_mrows!(Cp, ldc, Ap, lda, ir, Bp, ldb, jr, k, alpha, beta,
                        mre, Val(1), Val(_NR), Val(TB), Val(B0))
                else
                    _microkernel_unpacked_edge!(Cp, ldc, Ap, lda, ir, Bp, ldb, jr, k, alpha, beta,
                        mre, nre, TB, B0)
                end
                ir += W
            end
            jr += nr
        end
    end
    return C
end
# Tall split-reduction microkernel: MR row-blocks × NR cols, S independent partial accumulators per cell,
# summed at store — keeps MR·NR·S (=_ILP_TARGET) chains live from k=0 to cover the short-k fill the wide
# tile leaves exposed. Full tile only (mr rows × NR cols, all in bounds); partials delegate to the wide
# masked/edge kernels. Same store/beta folding as _microkernel_unpacked!. req#8; wintermute n=32 0.85→1.07×.
@generated function _microkernel_unpacked_split!(C::Ptr{T}, ldc::Int, A::Ptr{T}, lda::Int, ir::Int,
        B::Ptr{T}, ldb::Int, jr::Int, k::Int, alpha::T, beta::T,
        ::Val{MR}, ::Val{NR}, ::Val{S}, ::Val{TB}, ::Val{B0}) where {T, MR, NR, S, TB, B0}
    W = _vwidth(T); sz = sizeof(T); V = Vec{W, T}
    body = quote end
    for mi in 1:MR, j in 1:NR, s in 0:(S - 1)
        push!(body.args, :($(Symbol(:c, mi, :_, j, :_, s)) = zero($V)))
    end
    step = (pex, s) -> begin
        q = quote end
        for mi in 1:MR
            push!(q.args, :($(Symbol(:a, mi)) = vload($V, A + ((ir + $((mi - 1) * W)) + $pex * lda) * $sz)))
        end
        for j in 1:NR
            baddr = TB ? :(B + ((jr + $(j - 1)) + $pex * ldb) * $sz) : :(B + ($pex + (jr + $(j - 1)) * ldb) * $sz)
            push!(q.args, :($(Symbol(:b, j)) = $V(unsafe_load($baddr))))
            for mi in 1:MR
                cs = Symbol(:c, mi, :_, j, :_, s)
                push!(q.args, :($cs = muladd($(Symbol(:a, mi)), $(Symbol(:b, j)), $cs)))
            end
        end
        q
    end
    steps = [step(:(p + $s), s) for s in 0:(S - 1)]
    push!(body.args, quote
        p = 0
        while p + $S <= k; $(steps...); p += $S; end   # S independent chains per cell
        while p < k; $(step(:p, 0)); p += 1; end       # k-remainder into chain 0
    end)
    push!(body.args, :(av = $V(alpha)))
    B0 || push!(body.args, :(bv = $V(beta)))
    for j in 1:NR
        push!(body.args, :(colp = C + (ir + (jr + $(j - 1)) * ldc) * $sz))
        for mi in 1:MR
            red = Symbol(:c, mi, :_, j, :_, 0)
            for s in 1:(S - 1); red = :($red + $(Symbol(:c, mi, :_, j, :_, s))); end
            st = B0 ? :(vstore(av * $red, q)) : :(vstore(muladd(bv, vload($V, q), av * $red), q))
            push!(body.args, :(let q = colp + $((mi - 1) * W * sz); $st; end))
        end
    end
    push!(body.args, :(return nothing))
    return body
end

# Small-n split driver (wide AVX-512 only — caller gates on _SPLIT_OK). Full _SMR·W×_SNR tiles via the split
# kernel; row remainder → wide masked-row kernel, col remainder → edge strip (reuse; remainders are small).
function _gemm_unpacked_split!(::Val{TB}, ::Val{B0}, m::Int, n::Int, k::Int,
        alpha::T, A, B, beta::T, C) where {T<:BlasReal, TB, B0}
    W = _vwidth(T); mr = _SMR * W; nr = _SNR
    lda = stride(A, 2); ldb = stride(B, 2); ldc = stride(C, 2)
    parA = parent(A); parB = parent(B); parC = parent(C)
    GC.@preserve parA parB parC begin
        Ap = pointer(A); Bp = pointer(B); Cp = pointer(C)
        jr = 0
        while jr < n
            nre = min(nr, n - jr)
            ir = 0
            while ir < m
                mre = min(mr, m - ir)
                if nre == nr && rem(mre, W) == 0
                    # full OR W-aligned partial rows: split kernel at vr = mre/W row-blocks (keeps the
                    # split's B-reuse for the row remainder — mrows(NR=_SNR) would lose it). _SMR=4 → vr∈1:4.
                    vr = div(mre, W)
                    if vr == _SMR
                        _microkernel_unpacked_split!(Cp, ldc, Ap, lda, ir, Bp, ldb, jr, k, alpha, beta,
                            Val(_SMR), Val(_SNR), Val(_GEMM_SPLIT_S), Val(TB), Val(B0))
                    elseif vr == 1
                        _microkernel_unpacked_split!(Cp, ldc, Ap, lda, ir, Bp, ldb, jr, k, alpha, beta,
                            Val(1), Val(_SNR), Val(_GEMM_SPLIT_S), Val(TB), Val(B0))
                    elseif vr == 2
                        _microkernel_unpacked_split!(Cp, ldc, Ap, lda, ir, Bp, ldb, jr, k, alpha, beta,
                            Val(2), Val(_SNR), Val(_GEMM_SPLIT_S), Val(TB), Val(B0))
                    else
                        _microkernel_unpacked_split!(Cp, ldc, Ap, lda, ir, Bp, ldb, jr, k, alpha, beta,
                            Val(3), Val(_SNR), Val(_GEMM_SPLIT_S), Val(TB), Val(B0))
                    end
                elseif nre == nr
                    _microkernel_unpacked_mrows!(Cp, ldc, Ap, lda, ir, Bp, ldb, jr, k, alpha, beta,
                        mre, Val(_SMR), Val(_SNR), Val(TB), Val(B0))
                else
                    _microkernel_unpacked_edge!(Cp, ldc, Ap, lda, ir, Bp, ldb, jr, k, alpha, beta,
                        mre, nre, TB, B0)
                end
                ir += mr
            end
            jr += nr
        end
    end
    return C
end

function _gemm_unpacked!(::Val{TB}, ::Val{B0}, m::Int, n::Int, k::Int,
        alpha::T, A, B, beta::T, C) where {T<:BlasReal, TB, B0}
    if iszero(alpha) || k == 0
        _scale_C!(C, m, n, beta)   # C := beta·C (or 0); no contraction to do
        return C
    end
    if max(m, n, k) <= _GEMM_MR1_MAX
        return _gemm_unpacked_mr1!(Val(TB), Val(B0), m, n, k, alpha, A, B, beta, C)
    end
    # Short-k split path: small-n window where the wide tile under-fills (needs ≥1 full tall row-tile so the
    # split kernel actually fires; else the NR=2 remainder loses B-reuse to the wide NR). Const-folds off on AVX2.
    if _SPLIT_OK && m >= _SMR * _vwidth(T) && max(m, n, k) <= _GEMM_SPLIT_MAX
        return _gemm_unpacked_split!(Val(TB), Val(B0), m, n, k, alpha, A, B, beta, C)
    end
    W = _vwidth(T); mr = _MR * W; nr = _NR
    lda = stride(A, 2); ldb = stride(B, 2); ldc = stride(C, 2)
    parA = parent(A); parB = parent(B); parC = parent(C)   # preserve parents, not view wrappers (no box)
    GC.@preserve parA parB parC begin
        Ap = pointer(A); Bp = pointer(B); Cp = pointer(C)
        if B0 && n >= nr
            # β=0 path: walk full nr-column blocks; handle the column remainder by whichever is
            # cheaper. Large remainder (≥nr/2): OVERLAP — a full nr-block at n-nr overlapping inward
            # (≤nr/2 cols recomputed, overwrite is idempotent for β=0). Small remainder (<nr/2, e.g.
            # n mod nr == 1): the per-column edge (exact nre columns, no overlap waste). Rows use the
            # masked-row kernel throughout (no waste). β=0 only (overlap can't double-apply β).
            jr = 0
            while jr + nr <= n
                _unpacked_cols_full!(Cp, ldc, Ap, lda, Bp, ldb, m, jr, k, alpha, beta, mr, Val(TB), Val(B0))
                jr += nr
            end
            nre = n - jr
            if nre > 0
                if 2 * nre >= nr   # large remainder → overlap a full block at n-nr
                    _unpacked_cols_full!(Cp, ldc, Ap, lda, Bp, ldb, m, n - nr, k, alpha, beta, mr, Val(TB), Val(B0))
                else               # small remainder → exact per-column edge strip
                    ir = 0
                    while ir < m
                        mre = min(mr, m - ir)
                        _microkernel_unpacked_edge!(Cp, ldc, Ap, lda, ir, Bp, ldb, jr, k, alpha, beta,
                            mre, nre, TB, B0)
                        ir += mr
                    end
                end
            end
        else
            # β≠0, or m<mr / n<nr: masked path (full / masked-rows / per-column edge).
            jr = 0
            while jr < n
                nre = min(nr, n - jr)
                ir = 0
                while ir < m
                    mre = min(mr, m - ir)
                    nv1 = cld(mre, W) == 1   # ≤1 live row-vector → concrete Val(1), else Val(_MR)
                    if mre == mr && nre == nr        # full tile — no masks
                        _microkernel_unpacked!(Cp, ldc, Ap, lda, ir, Bp, ldb, jr, k, alpha, beta,
                            Val(_MR), Val(_NR), Val(TB), Val(B0))
                    elseif nre == nr && rem(mre, W) == 0   # W-aligned partial rows → clean clipped kernel
                        # the unpacked kernel reads A directly, so a smaller Val(vr) reads exactly the mre
                        # live rows — no mask, no wasted vector (mirrors the packed clip; closes the same
                        # misaligned-m penalty on the n≤_GEMM_UNPACK_MAX path, e.g. trsm off-diagonals).
                        vr = div(mre, W)
                        if vr == 1
                            _microkernel_unpacked!(Cp, ldc, Ap, lda, ir, Bp, ldb, jr, k, alpha, beta,
                                Val(1), Val(_NR), Val(TB), Val(B0))
                        elseif vr == 2
                            _microkernel_unpacked!(Cp, ldc, Ap, lda, ir, Bp, ldb, jr, k, alpha, beta,
                                Val(2), Val(_NR), Val(TB), Val(B0))
                        else
                            _microkernel_unpacked_mrows!(Cp, ldc, Ap, lda, ir, Bp, ldb, jr, k, alpha, beta,
                                mre, Val(_MR), Val(_NR), Val(TB), Val(B0))
                        end
                    elseif nre == nr                 # truly-partial rows (mre % W ≠ 0) → masked
                        if nv1
                            _microkernel_unpacked_mrows!(Cp, ldc, Ap, lda, ir, Bp, ldb, jr, k, alpha, beta,
                                mre, Val(1), Val(_NR), Val(TB), Val(B0))
                        else
                            _microkernel_unpacked_mrows!(Cp, ldc, Ap, lda, ir, Bp, ldb, jr, k, alpha, beta,
                                mre, Val(_MR), Val(_NR), Val(TB), Val(B0))
                        end
                    else                             # partial-column strip
                        _microkernel_unpacked_edge!(Cp, ldc, Ap, lda, ir, Bp, ldb, jr, k, alpha, beta,
                            mre, nre, TB, B0)
                    end
                    ir += mr
                end
                jr += nr
            end
        end
    end
    return C
end

# Generic GEMM (complex, Dual, non-contiguous C). Correct and AD-traceable; not blocked.
function _gemm_generic!(tA::Bool, tB::Bool, cA::Bool, cB::Bool, m::Int, n::Int, k::Int,
        alpha, A, B, beta, C)
    Tc = eltype(C)
    @inbounds for j in 1:n, i in 1:m
        C[i, j] = iszero(beta) ? zero(Tc) : beta * C[i, j]
    end
    iszero(alpha) && return C
    @inbounds for j in 1:n
        for p in 1:k
            bpj = tB ? (cB ? conj(B[j, p]) : B[j, p]) : B[p, j]
            ab = alpha * bpj
            for i in 1:m
                aip = tA ? (cA ? conj(A[p, i]) : A[p, i]) : A[i, p]
                C[i, j] = muladd(aip, ab, C[i, j])
            end
        end
    end
    return C
end

# ── Complex blocked GEMM (split-pack SIMD) ───────────────────────────────────────────────────
# The complex path was scalar (routed to _gemm_generic!). This is the SIMD version, portable
# Vec{W,T} over the REAL type T (W = real lanes). Design (memory openblas-complex-simd-design):
#   * SPLIT pack: the packer de-interleaves op(A)/op(B) ONCE into a real panel + an imag panel
#     ([Ar…][Ai…]); the k-loop is then straight-line real FMA — no per-iter deinterleave shuffle.
#   * One complex MAC = 4 real FMAs on real accumulators Cr,Ci.
#   * conj(op) is compile-time SIGNS SA,SB ∈ {±1} (transX=='C' ⇒ −1) — ONE kernel, no data negate.
#   * alpha (complex) applied at store; C pre-scaled by beta once (kc-blocked accumulation).
# Complex is ~2× flops/byte of real → compute-bound; gating is FMA-port saturation, not bandwidth.
# 2 accumulators/tile ⇒ HALVE the real tile dims for register pressure (_CMR/_CNR, overridable).
# Register budget per tile: 2·CMR·CNR (Cr,Ci accs) + 2·CMR (ar,ai) + 2 (br,bi) ≤ (vector registers).
# W=8 AVX-512 (32 regs): 2×4 = 16 accs (Zen4/5). W=4 AVX2 (16 ymm): 1×6 = 12 accs + 2 + 2 = 16, an
# EXACT fit — galen-swept 2026-07-02 (CNR=6 beat 4 at large n: more independent chains saturate the
# FMA ports; CMR=2 or CNR=8 spill AVX2 and tank to ~0.5–0.67). Same 12-acc AVX2 lesson as real gemm.
const _CMR = @load_preference("cgemm_mr", _W64 == 4 ? 1 : 2)::Int
const _CNR = @load_preference("cgemm_nr", _W64 == 4 ? 6 : 4)::Int
# Narrower nr for mid-small n: nr=6 doesn't divide most n → the last column-panel wastes compute on
# masked (padded) columns. nr=4 divides 8,16,20,24,28,32,40,48,64 cleanly → no column masking; it trades
# ~2 accumulator chains (worse large-n) for no-waste (better mid-small). W=4/AVX2 only (galen-swept:
# nr=4 lifts n=20 0.71→0.79, n=32 0.78→0.86); on W=8 mid-small is the unpacked path's job → _CNR_SMALL
# == _CNR makes the size branch a no-op. Crossover ≈ n=64.
const _CNR_SMALL = @load_preference("cgemm_nr_small", _W64 == 4 ? 4 : _CNR)::Int
const _CGEMM_NRSMALL_MAX = @load_preference("cgemm_nrsmall_max", _W64 == 4 ? 64 : 0)::Int
# req#8: complex contraction block — the SPLIT of real _KC (BLIS stores kc per datatype). Complex B
# micropanel is kc·nr·sizeof(ComplexF64) ≤ ½·L1; key on max(_CNR,_CNR_SMALL) so the small-nr branch
# under-fills L1 (never overflows). Reproduces Zen4 256 (old _KC, complex path bit-identical); galen 168,
# Zen5 384. Used at the complex-packed sites (_gemm_cmplx_impl!, _trgemm_cmplx_packed*, _hemm_packed_L!).
const _CKC = @load_preference("cgemm_kc", _l1_block(_HW, ComplexF64, max(_CNR, _CNR_SMALL)))::Int
# Small-n cutoff: below this the blocked machinery (pack + interleave-store) loses to the plain scalar
# triple loop. Width-adaptive default, Preferences-overridable (measured per machine).
const _CGEMM_TINY = @load_preference("cgemm_tiny", 6)::Int
# _CGEMM_TINY < max(m,n,k) ≤ this (and tA='N'): the UNPACKED tiny-n path. W=8: unpacked (skip pack,
# free MR) beats blocked broadly on Zen4/32-reg → 192. W=4/AVX2: unpacked's per-panel re-deinterleave
# loses to the vectorized-pack blocked path by n≈16, and the CNR=6 tile + deinterleave temp spill, so
# only the tiniest n win → 12. Above this, blocked (with vectorized packs). galen-swept 2026-07-02.
# AVX2: the unpacked direct-read complex kernel BEATS the blocked path through n≈40 (measured galen:
# n=16 1.20 vs 1.00, n=24 1.06 vs 0.94, n=32 0.99 vs 0.94, n=40 0.99 vs 0.97) — no pack/scratch overhead
# while A·B still fit L1. It collapses at n=48 (0.68, no cache blocking), but 3M (_CGEMM_3M_MIN=48) owns
# n≥48, so 40 is a clean handoff (41-47 → blocked ~0.97, ungated). Was 12 (far too conservative). This
# also lifts the small-n rank/hemm/symm ops that route through _gemm_core!. AVX-512 unchanged.
const _CGEMM_UNPACK_MAX = @load_preference("cgemm_unpack_max", _W64 == 4 ? 40 : 192)::Int

# Karatsuba 3M route for complex GEMM (see _gemm_3m!): 3 real gemms on split re/im, 25% fewer flops on
# the gating real kernel. BEATS OB at mid-n on AVX2 where the 4-FMA complex kernel is latency-bound
# (measured: 64×128×64 1.5×, 128³ 1.06×, 160³ 1.28×). Default ON for W=4 (AVX2 — the gap); OFF elsewhere
# (Zen4/Zen5 complex kernel already near-peak; untested — Preferences-enable to try). Windowed to the
# range where 3M's split/combine overhead is amortized; below _CGEMM_3M_MIN the blocked/unpacked paths win.
const _CGEMM_3M = @load_preference("cgemm_3m", _W64 == 4)::Bool
const _CGEMM_3M_MIN = @load_preference("cgemm_3m_min", 48)::Int    # max(m,n,k) ≥ this
const _CGEMM_3M_MAX = @load_preference("cgemm_3m_max", 2048)::Int  # max(m,n,k) ≤ this
const _CGEMM_3M_KMIN = @load_preference("cgemm_3m_kmin", 16)::Int  # min(m,n,k) ≥ this (thin gemms: overhead dominates)

# Strassen-Winograd for REAL gemm (see _gemm_strassen!): recursive 2×2 blocking, 7 half-size products
# instead of 8 (~14% fewer flops/level, compounding), each running the gating classical kernel as base.
# Beats OB at large n where classical is at the FMA roofline (measured AVX2: 2-level 1.20× at n=2048,
# 1.26× at 4096). Split while min(m,n,k) ≥ _STRASSEN_MIN (base stays ≥ ~min/2), capped at _MAXDEPTH.
# NN + real α/β only (trans/complex fall back). Default ON for W=4/8 (real gemm is throughput-bound on
# both AVX2 and AVX-512 — Strassen's flop cut is ISA-independent); per-box threshold via Preferences.
const _STRASSEN = @load_preference("strassen", _W64 == 4 || _W64 == 8)::Bool
const _STRASSEN_MIN = @load_preference("strassen_min", 1024)::Int      # split while min(m,n,k) ≥ this
const _STRASSEN_MAXDEPTH = @load_preference("strassen_maxdepth", 3)::Int
@inline function _strassen_depth(m::Int, n::Int, k::Int)
    d = 0; s = min(m, n, k)
    while s >= _STRASSEN_MIN && d < _STRASSEN_MAXDEPTH; d += 1; s >>= 1; end
    return d
end

# Split-pack op(A) into mr-row panels: real parts → ApR, imag → ApI (same panel layout as _pack_A!,
# so the microkernel indexes both identically). No alpha (folded at store), no conj (kernel sign).
function _pack_A_cmplx!(ApR::Vector{T}, ApI::Vector{T}, A, ic::Int, pc::Int, mce::Int, kce::Int,
        tA::Bool, mr::Int) where {T}
    if !tA && _strided1(A)     # contiguous columns → vectorized deinterleave
        return _pack_A_cmplx_simd!(ApR, ApI, A, ic, pc, mce, kce, mr)
    elseif tA && _strided1(A) && mr % _vwidth(T) == 0   # transposed → W×W register-transpose pack
        return _pack_A_cmplx_simd_T!(ApR, ApI, A, ic, pc, mce, kce, mr)
    end
    np = cld(mce, mr)
    @inbounds for pi in 0:(np - 1)
        base = pi * mr * kce
        for p in 0:(kce - 1)
            for r in 0:(mr - 1)
                lr = pi * mr + r; idx = base + p * mr + r + 1
                if lr < mce
                    v = tA ? A[pc + p + 1, ic + lr + 1] : A[ic + lr + 1, pc + p + 1]
                    ApR[idx] = real(v); ApI[idx] = imag(v)
                else
                    ApR[idx] = zero(T); ApI[idx] = zero(T)
                end
            end
        end
    end
    return
end
function _pack_B_cmplx!(BpR::Vector{T}, BpI::Vector{T}, B, pc::Int, jc::Int, kce::Int, nce::Int,
        tB::Bool, nr::Int) where {T}
    np = cld(nce, nr)
    @inbounds for ji in 0:(np - 1)
        base = ji * nr * kce
        if ji * nr + nr <= nce                   # full panel — branch-free; @simd ivdep vectorizes the
            j0 = jc + ji * nr                    # contiguous BpR/BpI stores (reads split re/im per elt)
            for p in 0:(kce - 1)
                gp = pc + p
                if tB
                    @simd ivdep for c in 0:(nr - 1)
                        v = B[j0 + c + 1, gp + 1]; BpR[base + p * nr + c + 1] = real(v); BpI[base + p * nr + c + 1] = imag(v)
                    end
                else
                    @simd ivdep for c in 0:(nr - 1)
                        v = B[gp + 1, j0 + c + 1]; BpR[base + p * nr + c + 1] = real(v); BpI[base + p * nr + c + 1] = imag(v)
                    end
                end
            end
            continue
        end
        for p in 0:(kce - 1)
            for c in 0:(nr - 1)
                lc = ji * nr + c; idx = base + p * nr + c + 1
                if lc < nce
                    v = tB ? B[jc + lc + 1, pc + p + 1] : B[pc + p + 1, jc + lc + 1]
                    BpR[idx] = real(v); BpI[idx] = imag(v)
                else
                    BpR[idx] = zero(T); BpI[idx] = zero(T)
                end
            end
        end
    end
    return
end

# The complex microkernel body (shared by full + masked). Emits the accumulator init, the 4-FMA
# k-loop, and the store expressions; `storefn(mi,j,q,cr,ci)` builds the per-cell store statement so
# the full (plain vstore) and masked (guarded, masked vstore) variants share everything else.
function _cmplx_kernel_body(T, W, MR, NR, SA, SB, storefn, A1 = false, AR = false)
    sz = sizeof(T); V = Vec{W, T}
    body = quote end
    # prefetch the C output tile (parity with the real microkernel) so the cold read-modify-write store
    # epilogue overlaps the k-loop — C column j base is at 2·(j-1)·ldc reals (interleaved complex).
    for j in 1:NR
        push!(body.args, :(_prefetch(C + $((j - 1) * 2) * ldc * $sz)))
    end
    for mi in 1:MR, j in 1:NR
        push!(body.args, :($(Symbol(:cr, mi, :_, j)) = zero($V)))
        push!(body.args, :($(Symbol(:ci, mi, :_, j)) = zero($V)))
    end
    inner = quote end
    for mi in 1:MR
        push!(inner.args, :($(Symbol(:ar, mi)) = vload($V, ApR + (p * $MR + $(mi - 1)) * $(W * sz))))
        push!(inner.args, :($(Symbol(:ai, mi)) = vload($V, ApI + (p * $MR + $(mi - 1)) * $(W * sz))))
    end
    for j in 1:NR
        push!(inner.args, :($(Symbol(:br, j)) = $V(unsafe_load(BpR + (p * $NR + $(j - 1)) * $sz))))
        push!(inner.args, :($(Symbol(:bi, j)) = $V(unsafe_load(BpI + (p * $NR + $(j - 1)) * $sz))))
        for mi in 1:MR
            cr = Symbol(:cr, mi, :_, j); ci = Symbol(:ci, mi, :_, j)
            ar = Symbol(:ar, mi); ai = Symbol(:ai, mi); br = Symbol(:br, j); bi = Symbol(:bi, j)
            # Cr += ar·br + s1·(ai·bi),  s1 = −SA·SB ;  Ci += SB·(ar·bi) + SA·(ai·br)
            push!(inner.args, :($cr = muladd($ar, $br, $cr)))
            aibi = SA * SB == 1 ? :(-$ai) : :($ai)   # s1 = −SA·SB
            push!(inner.args, :($cr = muladd($aibi, $bi, $cr)))
            arbi = SB == 1 ? :($ar) : :(-$ar)
            push!(inner.args, :($ci = muladd($arbi, $bi, $ci)))
            aibr = SA == 1 ? :($ai) : :(-$ai)
            push!(inner.args, :($ci = muladd($aibr, $br, $ci)))
        end
    end
    push!(body.args, quote
        @inbounds @simd ivdep for p in 0:(kc - 1)
            $inner
        end
    end)
    # store: res = alpha·(cr + i·ci); C += res. C is interleaved [r0 i0 r1 i1 …]; a complex add is an
    # elementwise real add of the interleaved reps, so we only interleave `res` once (no deinterleave
    # of old C) and add straight into C. beta already applied to C. ilv: (r0,i0,r1,i1,…) from [resr;resi].
    ilv = Expr(:tuple, (iseven(l) ? l ÷ 2 : W + l ÷ 2 for l in 0:(2W - 1))...)
    A1 || push!(body.args, :(avr = $V(alr); avi = $V(ali)))
    for j in 1:NR, mi in 1:MR
        cr = Symbol(:cr, mi, :_, j); ci = Symbol(:ci, mi, :_, j)
        q = :(C + ($(j - 1) * ldc * 2 + $((mi - 1) * 2W)) * $sz)
        # A1 (alpha==1): resv = interleave(cr,ci) directly — skip the complex-multiply-by-alpha entirely.
        # AR (alpha REAL, e.g. −1 for the ztrsm/hemm subtract): imag(α)=0 ⇒ the cross terms vanish, so just
        # scale by avr (2 muls/cell, no 4-mul+2-add). Full complex multiply only when α has an imag part.
        # Matters at short k where the store epilogue isn't amortized (the β=1 mid-n gap traced here).
        st = A1 ? :(resv = shufflevector($cr, $ci, Val($ilv))) :
             AR ? :(resv = shufflevector(avr * $cr, avr * $ci, Val($ilv))) :
                  :(resv = shufflevector(avr * $cr - avi * $ci, avr * $ci + avi * $cr, Val($ilv)))
        push!(body.args, storefn(mi, j, q, st))
    end
    push!(body.args, :(return nothing))
    return body
end

# Full mr×nr complex tile. B0 ⇒ overwrite C (skip the read) — the first/only kc-block when beta=0.
@generated function _microkernel_cmplx!(C::Ptr{T}, ldc::Int, ApR::Ptr{T}, ApI::Ptr{T},
        BpR::Ptr{T}, BpI::Ptr{T}, kc::Int, alr::T, ali::T,
        ::Val{MR}, ::Val{NR}, ::Val{SA}, ::Val{SB}, ::Val{B0} = Val(false),
        ::Val{A1} = Val(false), ::Val{AR} = Val(false)) where {T, MR, NR, SA, SB, B0, A1, AR}
    W = _vwidth(T)
    storefn = (mi, j, q, st) -> B0 ?
        quote let qq = $q; $st; vstore(resv, qq); end end :
        quote let qq = $q; $st; vstore(vload(Vec{$(2W), $T}, qq) + resv, qq); end end
    _cmplx_kernel_body(T, W, MR, NR, SA, SB, storefn, A1, AR)
end

# Masked complex tile: mre valid complex rows (2·mre real lanes), nre valid columns (column guard).
# Packed panels are zero-padded so the full compute is correct; only the store is masked/guarded.
@generated function _microkernel_cmplx_masked!(C::Ptr{T}, ldc::Int, ApR::Ptr{T}, ApI::Ptr{T},
        BpR::Ptr{T}, BpI::Ptr{T}, kc::Int, alr::T, ali::T, mre::Int, nre::Int,
        ::Val{MR}, ::Val{NR}, ::Val{SA}, ::Val{SB}, ::Val{B0} = Val(false),
        ::Val{A1} = Val(false), ::Val{AR} = Val(false)) where {T, MR, NR, SA, SB, B0, A1, AR}
    W = _vwidth(T)
    lanetuple = Expr(:tuple, (0:(2W - 1))...)
    # row mask: real lane l valid iff its complex row l÷2 is < the valid rows in THIS vector
    # (mre − (mi−1)·W). Column guard j<nre. B0 ⇒ masked overwrite (skip the C read).
    storefn = (mi, j, q, st) -> quote
        if $(j - 1) < nre
            let qq = $q, msk = (Vec{$(2W), Int}($lanetuple)) < 2 * (mre - $((mi - 1) * W))
                $st
                $(B0 ? :(vstore(resv, qq, msk)) : :(vstore(vload(Vec{$(2W), $T}, qq, msk) + resv, qq, msk)))
            end
        end
    end
    _cmplx_kernel_body(T, W, MR, NR, SA, SB, storefn, A1, AR)
end

# Triangular-store complex tile for the single-pass packed syrk/herk path (parity with the real
# _microkernel_tri!): the straddling diagonal tile. Combines the interleaved ROW cap (mre complex
# rows) with a per-column diagonal threshold. Everything (mre, thr, d0, crow) is in COMPLEX-row
# units; the ×2 for interleaving lives only in the C address (built by the caller). Real lane
# l∈0:2W-1 of vector-block mi holds complex row r = l÷2 + (mi-1)·W — baked into a compile-time tuple
# (0,0,1,1,…) so lanes 2r,2r+1 share a mask bit (re/im store or skip together). No scalar loop.
@generated function _microkernel_cmplx_tri!(C::Ptr{T}, ldc::Int, ApR::Ptr{T}, ApI::Ptr{T},
        BpR::Ptr{T}, BpI::Ptr{T}, kc::Int, alr::T, ali::T, mre::Int, nre::Int,
        d0::Int, upper::Bool, ::Val{MR}, ::Val{NR}, ::Val{SA}, ::Val{SB},
        ::Val{B0} = Val(false), ::Val{A1} = Val(false), ::Val{AR} = Val(false)) where {T, MR, NR, SA, SB, B0, A1, AR}
    W = _vwidth(T)
    crowtuple = Expr(:tuple, ((l ÷ 2) for l in 0:(2W - 1))...)   # (0,0,1,1,…,W−1,W−1)
    storefn = (mi, j, q, st) -> quote
        if $(j - 1) < nre
            let qq = $q, thr = d0 + $(j - 1),
                crow = Vec{$(2W), Int}($crowtuple) + $((mi - 1) * W)
                mk = (crow < mre) & (upper ? (crow <= thr) : (crow >= thr))
                $st
                $(B0 ? :(vstore(resv, qq, mk)) :
                       :(vstore(vload(Vec{$(2W), $T}, qq, mk) + resv, qq, mk)))
            end
        end
    end
    _cmplx_kernel_body(T, W, MR, NR, SA, SB, storefn, A1, AR)
end

# ── Fused two-product complex microkernel (syr2k/her2k AVX2 mid-n lever) ─────────────────────
# C[tile] += Σₖ σA(a1)·σB(b1) + σA(a2)·σB(b2) — BOTH rank-k products accumulated into ONE register set
# with ONE RMW store (two _microkernel_cmplx! calls/tile = two prologues + two RMW epilogues, measured
# REGRESSION). No α args: the driver folds s = (SA==-1 ? conj(α) : α) into the X-pack, so the conj signs
# deliver product-1 coefficient σA(s)=α and product-2 σB(s)=ᾱ (her2k)/α (syr2k) — the store is then the
# cheap A1-style interleave+add. Budget (W=4,MR=1,NR=4): 8 accs + 4 A-vecs + 2 sequenced broadcasts =
# 14/16 ymm (separate per-product accs would be 16+6 → spill, the NR=6 lesson). Fable-designed 2026-07-06.
function _cmplx_kernel_body2(T, W, MR, NR, SA, SB, storefn)
    sz = sizeof(T); V = Vec{W, T}
    body = quote end
    for j in 1:NR
        push!(body.args, :(_prefetch(C + $((j - 1) * 2) * ldc * $sz)))
    end
    for mi in 1:MR, j in 1:NR
        push!(body.args, :($(Symbol(:cr, mi, :_, j)) = zero($V)))
        push!(body.args, :($(Symbol(:ci, mi, :_, j)) = zero($V)))
    end
    inner = quote end
    for mi in 1:MR
        push!(inner.args, :($(Symbol(:a1r, mi)) = vload($V, A1R + (p * $MR + $(mi - 1)) * $(W * sz))))
        push!(inner.args, :($(Symbol(:a1i, mi)) = vload($V, A1I + (p * $MR + $(mi - 1)) * $(W * sz))))
        push!(inner.args, :($(Symbol(:a2r, mi)) = vload($V, A2R + (p * $MR + $(mi - 1)) * $(W * sz))))
        push!(inner.args, :($(Symbol(:a2i, mi)) = vload($V, A2I + (p * $MR + $(mi - 1)) * $(W * sz))))
    end
    for j in 1:NR
        for (pn, BR, BI) in ((1, :B1R, :B1I), (2, :B2R, :B2I))   # b{pn} pair dies before the next loads
            br = Symbol(:b, pn, :r, j); bi = Symbol(:b, pn, :i, j)
            push!(inner.args, :($br = $V(unsafe_load($BR + (p * $NR + $(j - 1)) * $sz))))
            push!(inner.args, :($bi = $V(unsafe_load($BI + (p * $NR + $(j - 1)) * $sz))))
            for mi in 1:MR
                cr = Symbol(:cr, mi, :_, j); ci = Symbol(:ci, mi, :_, j)
                ar = Symbol(:a, pn, :r, mi); ai = Symbol(:a, pn, :i, mi)
                push!(inner.args, :($cr = muladd($ar, $br, $cr)))
                aibi = SA * SB == 1 ? :(-$ai) : :($ai)
                push!(inner.args, :($cr = muladd($aibi, $bi, $cr)))
                arbi = SB == 1 ? :($ar) : :(-$ar)
                push!(inner.args, :($ci = muladd($arbi, $bi, $ci)))
                aibr = SA == 1 ? :($ai) : :(-$ai)
                push!(inner.args, :($ci = muladd($aibr, $br, $ci)))
            end
        end
    end
    push!(body.args, quote
        @inbounds @simd ivdep for p in 0:(kc - 1)
            $inner
        end
    end)
    ilv = Expr(:tuple, (iseven(l) ? l ÷ 2 : W + l ÷ 2 for l in 0:(2W - 1))...)
    for j in 1:NR, mi in 1:MR
        cr = Symbol(:cr, mi, :_, j); ci = Symbol(:ci, mi, :_, j)
        q = :(C + ($(j - 1) * ldc * 2 + $((mi - 1) * 2W)) * $sz)
        st = :(resv = shufflevector($cr, $ci, Val($ilv)))       # α folded into the pack ⇒ no store-α
        push!(body.args, storefn(mi, j, q, st))
    end
    push!(body.args, :(return nothing))
    return body
end
@generated function _microkernel2_cmplx!(C::Ptr{T}, ldc::Int, A1R::Ptr{T}, A1I::Ptr{T},
        B1R::Ptr{T}, B1I::Ptr{T}, A2R::Ptr{T}, A2I::Ptr{T}, B2R::Ptr{T}, B2I::Ptr{T},
        kc::Int, ::Val{MR}, ::Val{NR}, ::Val{SA}, ::Val{SB},
        ::Val{B0} = Val(false)) where {T, MR, NR, SA, SB, B0}
    W = _vwidth(T)
    storefn = (mi, j, q, st) -> B0 ?
        quote let qq = $q; $st; vstore(resv, qq); end end :
        quote let qq = $q; $st; vstore(vload(Vec{$(2W), $T}, qq) + resv, qq); end end
    _cmplx_kernel_body2(T, W, MR, NR, SA, SB, storefn)
end
@generated function _microkernel2_cmplx_masked!(C::Ptr{T}, ldc::Int, A1R::Ptr{T}, A1I::Ptr{T},
        B1R::Ptr{T}, B1I::Ptr{T}, A2R::Ptr{T}, A2I::Ptr{T}, B2R::Ptr{T}, B2I::Ptr{T},
        kc::Int, mre::Int, nre::Int, ::Val{MR}, ::Val{NR}, ::Val{SA}, ::Val{SB},
        ::Val{B0} = Val(false)) where {T, MR, NR, SA, SB, B0}
    W = _vwidth(T)
    lanetuple = Expr(:tuple, (0:(2W - 1))...)
    storefn = (mi, j, q, st) -> quote
        if $(j - 1) < nre
            let qq = $q, msk = (Vec{$(2W), Int}($lanetuple)) < 2 * (mre - $((mi - 1) * W))
                $st
                $(B0 ? :(vstore(resv, qq, msk)) : :(vstore(vload(Vec{$(2W), $T}, qq, msk) + resv, qq, msk)))
            end
        end
    end
    _cmplx_kernel_body2(T, W, MR, NR, SA, SB, storefn)
end
@generated function _microkernel2_cmplx_tri!(C::Ptr{T}, ldc::Int, A1R::Ptr{T}, A1I::Ptr{T},
        B1R::Ptr{T}, B1I::Ptr{T}, A2R::Ptr{T}, A2I::Ptr{T}, B2R::Ptr{T}, B2I::Ptr{T},
        kc::Int, mre::Int, nre::Int, d0::Int, upper::Bool, ::Val{MR}, ::Val{NR},
        ::Val{SA}, ::Val{SB}, ::Val{B0} = Val(false)) where {T, MR, NR, SA, SB, B0}
    W = _vwidth(T)
    crowtuple = Expr(:tuple, ((l ÷ 2) for l in 0:(2W - 1))...)
    storefn = (mi, j, q, st) -> quote
        if $(j - 1) < nre
            let qq = $q, thr = d0 + $(j - 1),
                crow = Vec{$(2W), Int}($crowtuple) + $((mi - 1) * W)
                mk = (crow < mre) & (upper ? (crow <= thr) : (crow >= thr))
                $st
                $(B0 ? :(vstore(resv, qq, mk)) :
                       :(vstore(vload(Vec{$(2W), $T}, qq, mk) + resv, qq, mk)))
            end
        end
    end
    _cmplx_kernel_body2(T, W, MR, NR, SA, SB, storefn)
end
# Complex-scale a packed split panel in place: (PR,PI) ← s·(PR,PI). Folds α into the fused driver's
# X-pack (padded zeros scale to zero — the whole packed region is safe). Skipped when α==1.
function _scale_pack_cmplx!(PR::Vector{T}, PI::Vector{T}, len::Int, sr::T, si::T) where {T}
    @inbounds @simd ivdep for idx in 1:len
        r = PR[idx]; im = PI[idx]
        PR[idx] = muladd(sr, r, -(si * im))
        PI[idx] = muladd(sr, im, si * r)
    end
    return
end

# Complex split-pack buffers live in the per-type L3Workspace `cg` field (see src/workspace.jl);
# `_gemm_scratch_cmplx(T, lenA, lenB)` grows and returns them.

# Complex blocked driver, specialized on conj signs SA,SB (resolved once at the boundary below).
function _gemm_cmplx_impl!(::Val{SA}, ::Val{SB}, ::Val{NR}, ::Val{A1}, ::Val{AR}, tA::Bool, tB::Bool,
        m::Int, n::Int, k::Int, alpha, A, B, beta, C) where {SA, SB, NR, A1, AR}
    Tc = eltype(C); T = real(Tc)
    b0 = iszero(beta)
    b0 || _scale_C!(C, m, n, convert(Tc, beta))   # beta=0 ⇒ first kc-block overwrites (no scale pass)
    if iszero(alpha) || k == 0
        b0 && _scale_C!(C, m, n, zero(Tc))
        return C
    end
    W = _vwidth(T); mr = _CMR * W; nr = NR
    kc = min(_CKC, k)
    mc = _at_mc_kc(_HW, eltype(C), kc, mr, cld(m, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(n, nr) * nr)
    ApR, ApI, BpR, BpI = _gemm_scratch_cmplx(T, cld(mc, mr) * mr * kc, cld(nc, nr) * nr * kc)
    a = convert(Tc, alpha); alr = real(a); ali = imag(a)
    ldc = stride(C, 2); sz = sizeof(T)
    GC.@preserve C ApR ApI BpR BpI begin
        Cp0 = Ptr{T}(pointer(C)); ARp = pointer(ApR); AIp = pointer(ApI)
        BRp = pointer(BpR); BIp = pointer(BpI)
        jc = 0
        while jc < n
            nce = min(nc, n - jc)
            pc = 0
            while pc < k
                kce = min(kc, k - pc)
                ov = b0 && pc == 0        # overwrite C on the first kc-block (beta=0); else accumulate
                _pack_B_cmplx!(BpR, BpI, B, pc, jc, kce, nce, tB, nr)
                ic = 0
                while ic < m
                    mce = min(mc, m - ic)
                    _pack_A_cmplx!(ApR, ApI, A, ic, pc, mce, kce, tA, mr)
                    jr = 0
                    while jr < nce
                        nre = min(nr, nce - jr)
                        ir = 0
                        while ir < mce
                            mre = min(mr, mce - ir)
                            aoff = div(ir, mr) * mr * kce * sz
                            boff = div(jr, nr) * nr * kce * sz
                            Cblk = Cp0 + (2 * (ic + ir) + (jc + jr) * ldc * 2) * sz
                            aR = Ptr{T}(ARp + aoff); aI = Ptr{T}(AIp + aoff)
                            bR = Ptr{T}(BRp + boff); bI = Ptr{T}(BIp + boff)
                            if mre == mr && nre == nr
                                ov ? _microkernel_cmplx!(Ptr{T}(Cblk), ldc, aR, aI, bR, bI,
                                        kce, alr, ali, Val(_CMR), Val(NR), Val(SA), Val(SB), Val(true), Val(A1), Val(AR)) :
                                     _microkernel_cmplx!(Ptr{T}(Cblk), ldc, aR, aI, bR, bI,
                                        kce, alr, ali, Val(_CMR), Val(NR), Val(SA), Val(SB), Val(false), Val(A1), Val(AR))
                            else
                                ov ? _microkernel_cmplx_masked!(Ptr{T}(Cblk), ldc, aR, aI, bR, bI,
                                        kce, alr, ali, mre, nre, Val(_CMR), Val(NR), Val(SA), Val(SB), Val(true), Val(A1), Val(AR)) :
                                     _microkernel_cmplx_masked!(Ptr{T}(Cblk), ldc, aR, aI, bR, bI,
                                        kce, alr, ali, mre, nre, Val(_CMR), Val(NR), Val(SA), Val(SB), Val(false), Val(A1), Val(AR))
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
function _gemm_cmplx_blocked!(tA::Bool, tB::Bool, cA::Bool, cB::Bool, m::Int, n::Int, k::Int,
        alpha, A, B, beta, C)
    # Size-adaptive tile width: mid-small n use a narrower nr (fewer column-mask waste tiles, since nr=6
    # doesn't divide most n), large n use the wider register-optimal nr. No-op where _CNR_SMALL==_CNR
    # (W=8: small n is handled by the unpacked path, blocked only sees large n). galen-swept 2026-07-02.
    ac = convert(eltype(C), alpha)
    a1 = isone(ac)              # alpha==1 ⇒ pure interleave store (no multiply)
    ar = iszero(imag(ac))       # alpha REAL (incl. −1, the subtract) ⇒ scale-only store (2 muls, no cross)
    nr = (_CNR_SMALL != _CNR && max(m, n, k) <= _CGEMM_NRSMALL_MAX) ? Val(_CNR_SMALL) : Val(_CNR)
    a1 ? _cmplx_blk_conj(nr, Val(true), Val(true), tA, tB, cA, cB, m, n, k, alpha, A, B, beta, C) :
    ar ? _cmplx_blk_conj(nr, Val(false), Val(true), tA, tB, cA, cB, m, n, k, alpha, A, B, beta, C) :
         _cmplx_blk_conj(nr, Val(false), Val(false), tA, tB, cA, cB, m, n, k, alpha, A, B, beta, C)
end
@inline function _cmplx_blk_conj(::Val{NR}, ::Val{A1}, ::Val{AR}, tA::Bool, tB::Bool, cA::Bool, cB::Bool,
        m::Int, n::Int, k::Int, alpha, A, B, beta, C) where {NR, A1, AR}
    if cA
        cB ? _gemm_cmplx_impl!(Val(-1), Val(-1), Val(NR), Val(A1), Val(AR), tA, tB, m, n, k, alpha, A, B, beta, C) :
             _gemm_cmplx_impl!(Val(-1), Val(1), Val(NR), Val(A1), Val(AR), tA, tB, m, n, k, alpha, A, B, beta, C)
    else
        cB ? _gemm_cmplx_impl!(Val(1), Val(-1), Val(NR), Val(A1), Val(AR), tA, tB, m, n, k, alpha, A, B, beta, C) :
             _gemm_cmplx_impl!(Val(1), Val(1), Val(NR), Val(A1), Val(AR), tA, tB, m, n, k, alpha, A, B, beta, C)
    end
end

# ── Vectorized complex-pack helpers ──────────────────────────────────────────────────────────
# The blocked path's small-n loss was pack+scale overhead (MEASURED 40% of runtime at n=32 F64, 84%
# F32; only 3.3% at n=512). The packs are now vectorized: A's de-interleave is a contiguous Vec{2W}
# load + shuffle (`_pack_A_cmplx_simd!`, used by `_pack_A_cmplx!`), B's is an `@simd ivdep` split into
# the contiguous BpR/BpI panels. Structure stays blocked (pack B ONCE, reuse) — streaming B instead
# was tried and LOST (re-reads B per row-tile: ~0.45 on galen). Complex `_scale_C!` is vectorized too.

# Deinterleave a Vec{2W} [r i r i…] into (reals, imags) via one shuffle each (indices fold at compile).
@inline @generated function _deint_cmplx(av::Vec{N, T}) where {N, T}
    W = N ÷ 2
    ev = Expr(:tuple, (2 * (i - 1) for i in 1:W)...); od = Expr(:tuple, (2 * (i - 1) + 1 for i in 1:W)...)
    :((shufflevector(av, Val($ev)), shufflevector(av, Val($od))))
end

# Parity-preserving halving reduce: an interleaved [even,odd,even,odd,…] partial-product Vec{N}
# (N a power of 2 ≥ 2) → Vec{2} = [Σ even-lanes, Σ odd-lanes]. Each level halves-and-adds; every half
# has even length down to width 2, so even/odd lane parity is preserved at every step. Replaces
# `_deint_cmplx` + two full-width horizontal `sum`s in the complex dot / gemv-T epilogue: strictly
# fewer shuffles (the first fold of a multi-register Vec, e.g. Vec{16,F64}=2 zmm, is a bare add), and
# it dominates the reduction cost that swamps the tiny main loop at small m. Fable-designed 2026-07-14.
@inline @generated function _fold2_cmplx(v::Vec{N, T}) where {N, T}
    body = :v; n = N
    while n > 2
        h = n ÷ 2
        lo = Expr(:tuple, (0:(h - 1))...); hi = Expr(:tuple, (h:(n - 1))...)
        body = :(shufflevector($body, Val($lo)) + shufflevector($body, Val($hi)))
        n = h
    end
    body
end
# Inverse of _deint_cmplx: interleave separate re/im W-vectors back to a Vec{2W} (re,im,re,im,…).
@inline @generated function _intlv_cmplx(vr::Vec{W, T}, vi::Vec{W, T}) where {W, T}
    ilv = Expr(:tuple, (iseven(l) ? l ÷ 2 : W + l ÷ 2 for l in 0:(2W - 1))...)
    :(shufflevector(vr, vi, Val($ilv)))
end

# Vectorized A-pack (tA='N', contiguous columns, mr a multiple of W): load Vec{2W} chunks of each
# column, deinterleave → real panel ApR / imag panel ApI. Partial last panel stays scalar (zero-pad).
function _pack_A_cmplx_simd!(ApR::Vector{T}, ApI::Vector{T}, A, ic::Int, pc::Int,
        mce::Int, kce::Int, mr::Int) where {T}
    W = _vwidth(T); sz = sizeof(T); lda = stride(A, 2); np = cld(mce, mr)
    GC.@preserve A ApR ApI begin
        Aptr = Ptr{T}(pointer(A)); PR = pointer(ApR); PI = pointer(ApI)
        @inbounds for pi in 0:(np - 1)
            base = pi * mr * kce; r0 = ic + pi * mr
            if pi * mr + mr <= mce                       # full panel — vectorized deinterleave
                for p in 0:(kce - 1)
                    col = 2 * (r0 + (pc + p) * lda); dst = base + p * mr
                    o = 0
                    while o < mr
                        ar, ai = _deint_cmplx(vload(Vec{2W, T}, Aptr + (col + 2 * o) * sz))
                        vstore(ar, PR + (dst + o) * sz); vstore(ai, PI + (dst + o) * sz)
                        o += W
                    end
                end
            else                                         # partial panel — scalar, zero-padded
                for p in 0:(kce - 1), r in 0:(mr - 1)
                    lr = pi * mr + r; idx = base + p * mr + r + 1
                    if lr < mce
                        v = A[ic + lr + 1, pc + p + 1]; ApR[idx] = real(v); ApI[idx] = imag(v)
                    else
                        ApR[idx] = zero(T); ApI[idx] = zero(T)
                    end
                end
            end
        end
    end
    return
end
# _pack_A_cmplx_simd! with a complex scale s=(sr,si) FOLDED into the store (PR=ar·sr−ai·si, PI=ar·si+ai·sr)
# — one pass replaces pack + a separate _scale_pack_cmplx! pass for the fused two-product driver's α≠1
# (contiguous X only; the fused kernel bakes α into the X-pack since its 2-product microkernel can't apply
# it at the store). Saves a full re-read+re-write of the pack per kc-block. Deinterleave = _pack_A_cmplx_simd!.
function _pack_A_cmplx_simd_scaled!(ApR::Vector{T}, ApI::Vector{T}, A, ic::Int, pc::Int,
        mce::Int, kce::Int, mr::Int, sr::T, si::T) where {T}
    W = _vwidth(T); sz = sizeof(T); lda = stride(A, 2); np = cld(mce, mr)
    svr = Vec{W, T}(sr); svi = Vec{W, T}(si)
    GC.@preserve A ApR ApI begin
        Aptr = Ptr{T}(pointer(A)); PR = pointer(ApR); PI = pointer(ApI)
        @inbounds for pi in 0:(np - 1)
            base = pi * mr * kce; r0 = ic + pi * mr
            if pi * mr + mr <= mce
                for p in 0:(kce - 1)
                    col = 2 * (r0 + (pc + p) * lda); dst = base + p * mr
                    o = 0
                    while o < mr
                        ar, ai = _deint_cmplx(vload(Vec{2W, T}, Aptr + (col + 2 * o) * sz))
                        nar = muladd(svr, ar, -(svi * ai)); nai = muladd(svr, ai, svi * ar)
                        vstore(nar, PR + (dst + o) * sz); vstore(nai, PI + (dst + o) * sz)
                        o += W
                    end
                end
            else
                for p in 0:(kce - 1), r in 0:(mr - 1)
                    lr = pi * mr + r; idx = base + p * mr + r + 1
                    if lr < mce
                        v = A[ic + lr + 1, pc + p + 1]; re = real(v); im = imag(v)
                        ApR[idx] = sr * re - si * im; ApI[idx] = sr * im + si * re
                    else
                        ApR[idx] = zero(T); ApI[idx] = zero(T)
                    end
                end
            end
        end
    end
    return
end

# One W×W COMPLEX transpose-pack block: load W interleaved Vec{2W} column-chunks of A (column r at
# +2r·lda reals), _deint_cmplx each → W real + W imag Vec{W}s, run the _tblk! shuffle butterfly on both
# sets, store transposed row pp at dstR/dstI + pp·mrstride. No α (scaled at store), no conj (kernel SA
# sign). lda in COMPLEX elements. Fable-designed 2026-07-06 (== scalar ref over 288 shape/offset cases).
@generated function _tblk_cmplx!(dstR::Ptr{T}, dstI::Ptr{T}, src::Ptr{T}, lda::Int,
        mrstride::Int, ::Val{W}) where {T, W}
    sz = sizeof(T); q = Int(round(log2(W)))
    body = quote end
    curR = [Symbol(:r_, r) for r in 0:(W - 1)]
    curI = [Symbol(:i_, r) for r in 0:(W - 1)]
    for r in 0:(W - 1)
        push!(body.args, :(($(curR[r + 1]), $(curI[r + 1])) =
            _deint_cmplx(vload(Vec{$(2W), $T}, src + $(2r) * lda * $sz))))
    end
    for stage in 0:(q - 1)
        s = 1 << stage
        nxtR = [Symbol(:sr, stage, :_, i) for i in 0:(W - 1)]
        nxtI = [Symbol(:si, stage, :_, i) for i in 0:(W - 1)]
        for i in 0:(W - 1)
            if (i & s) == 0
                j = i | s
                lo = ntuple(e0 -> (e = e0 - 1; blk = e ÷ (2s); w = e % (2s); w < s ? blk * 2s + w : W + blk * 2s + (w - s)), W)
                hi = ntuple(e0 -> (e = e0 - 1; blk = e ÷ (2s); w = e % (2s); w < s ? blk * 2s + s + w : W + blk * 2s + s + (w - s)), W)
                push!(body.args, :($(nxtR[i + 1]) = shufflevector($(curR[i + 1]), $(curR[j + 1]), Val($lo))))
                push!(body.args, :($(nxtR[j + 1]) = shufflevector($(curR[i + 1]), $(curR[j + 1]), Val($hi))))
                push!(body.args, :($(nxtI[i + 1]) = shufflevector($(curI[i + 1]), $(curI[j + 1]), Val($lo))))
                push!(body.args, :($(nxtI[j + 1]) = shufflevector($(curI[i + 1]), $(curI[j + 1]), Val($hi))))
            end
        end
        curR = nxtR; curI = nxtI
    end
    for pp in 0:(W - 1)
        push!(body.args, :(vstore($(curR[pp + 1]), dstR + $pp * mrstride * $sz)))
        push!(body.args, :(vstore($(curI[pp + 1]), dstI + $pp * mrstride * $sz)))
    end
    push!(body.args, :(return nothing))
    body
end
# Vectorized split A-pack for tA (op(A)=Aᵀ/Aᴴ, contiguous columns, mr a multiple of W): op(A)[gi,gp] =
# A[gp,gi] would read A's rows (strided) — the old scalar fallback. Instead read W contiguous A-columns
# (op(A) rows), W contraction-rows at a time, deint+transpose the W×W block into ApR/ApI. Partial panels
# / contraction tail stay scalar (zero-padded). No α, no conj (both applied downstream).
@inline function _pack_A_cmplx_simd_T!(ApR::Vector{T}, ApI::Vector{T}, A, ic::Int, pc::Int,
        mce::Int, kce::Int, mr::Int) where {T}
    W = _vwidth(T); sz = sizeof(T); lda = stride(A, 2)
    np = cld(mce, mr); kfull = (kce ÷ W) * W; sub = mr ÷ W
    GC.@preserve A ApR ApI begin
        Aptr = Ptr{T}(pointer(A)); PR = pointer(ApR); PI = pointer(ApI)
        @inbounds for pi in 0:(np - 1)
            base = pi * mr * kce
            if pi * mr + mr <= mce                       # full panel — W×W register transposes
                for ri in 0:(sub - 1)
                    r0 = pi * mr + ri * W                # op(A) row offset within the block
                    p = 0
                    while p < kfull
                        src = Aptr + 2 * ((ic + r0) * lda + (pc + p)) * sz   # A[pc+p, ic+r0]
                        doff = base + p * mr + ri * W
                        _tblk_cmplx!(PR + doff * sz, PI + doff * sz, src, lda, mr, Val(W))
                        p += W
                    end
                    while p < kce                        # contraction tail — scalar
                        for r in 0:(W - 1)
                            v = A[pc + p + 1, ic + r0 + r + 1]
                            idx = base + p * mr + ri * W + r + 1
                            ApR[idx] = real(v); ApI[idx] = imag(v)
                        end
                        p += 1
                    end
                end
            else                                         # partial panel — scalar, zero-padded
                for p in 0:(kce - 1), r in 0:(mr - 1)
                    lr = pi * mr + r; idx = base + p * mr + r + 1
                    if lr < mce
                        v = A[pc + p + 1, ic + lr + 1]; ApR[idx] = real(v); ApI[idx] = imag(v)
                    else
                        ApR[idx] = zero(T); ApI[idx] = zero(T)
                    end
                end
            end
        end
    end
    return
end

# ── Complex UNPACKED path for TINY n (tA='N') ────────────────────────────────────────────────
# For very small cache-resident complex GEMM the blocked machinery (pack + mr-tile masking) doesn't
# amortize: read A/B directly and de-interleave A's [r i r i…] columns IN the k-loop (2 shuffles per
# A-vector). Addressing via lda (not a packed panel) makes MR FREE → the row remainder uses a Val(1)
# single-vector tile (masking waste ≤ one W-vector, not a whole padded mr-tile — decisive for tiny F32
# where mr=32). Wins ONLY at tiny n (the per-panel re-deinterleave costs n/nr× — it loses to blocked by
# n≈16 on AVX2), so the cutoff is small (esp. W=4). tA='N' required (contiguous A columns for the load).
@generated function _uker_cmplx!(C::Ptr{T}, ldc::Int, A::Ptr{T}, lda::Int, ir::Int,
        B::Ptr{T}, ldb::Int, jr::Int, k::Int, alr::T, ali::T, mre::Int, nre::Int,
        ::Val{MR}, ::Val{NR}, ::Val{TB}, ::Val{SA}, ::Val{SB},
        ::Val{B0}, ::Val{A1},
        ::Val{AR}, ::Val{FULL},
        ::Val{TRI}, d0::Int, upper::Bool) where {T, MR, NR, TB, SA, SB, B0, A1, AR, FULL, TRI}
    # NOTE: no default args — they generate default-arg TRAMPOLINE methods that juliac's --trim verifier
    # cannot resolve through this @generated function (unresolved invoke ::Any → .so build fails). Every
    # call site MUST pass all 10 Vals + d0 + upper explicitly. (Regression guard: bench/juliac build.)
    W = _vwidth(T); sz = sizeof(T); V = Vec{W, T}; V2W = Vec{2W, T}
    ilv = Expr(:tuple, (iseven(l) ? l ÷ 2 : W + l ÷ 2 for l in 0:(2W - 1))...)
    swp = Expr(:tuple, (l ⊻ 1 for l in 0:(2W - 1))...)                       # (1,0,3,2,…): swap re/im pairs
    signt = Expr(:tuple, (iseven(l) ? :(-one($T)) : :(one($T)) for l in 0:(2W - 1))...)  # (−1,1,−1,1,…)
    lanetuple = Expr(:tuple, (0:(2W - 1))...)
    body = quote end
    # FULL (all mr rows valid): unmasked loads/stores — drops a vmaskmov from the hot loop. Edge tiles mask.
    FULL || push!(body.args, :(lanes2 = Vec{$(2W), Int}($lanetuple)))
    FULL || for mi in 1:MR
        push!(body.args, :($(Symbol(:m2, mi)) = lanes2 < 2 * (mre - $((mi - 1) * W))))
    end
    for mi in 1:MR, j in 1:NR
        push!(body.args, :($(Symbol(:cr, mi, :_, j)) = zero($V)))
        push!(body.args, :($(Symbol(:ci, mi, :_, j)) = zero($V)))
    end
    inner = quote end
    for mi in 1:MR
        aoff = :((2 * (ir + $((mi - 1) * W)) + p * lda * 2) * $sz)
        aload = FULL ? :(vload($V2W, A + $aoff)) : :(vload($V2W, A + $aoff, $(Symbol(:m2, mi))))
        push!(inner.args, :($(Symbol(:av, mi)) = $aload))
        push!(inner.args, :(($(Symbol(:ar, mi)), $(Symbol(:ai, mi))) = _deint_cmplx($(Symbol(:av, mi)))))
    end
    for j in 1:NR
        boff = TB ? :((2 * ((jr + $(j - 1)) + p * ldb)) * $sz) : :((2 * (p + (jr + $(j - 1)) * ldb)) * $sz)
        push!(inner.args, :($(Symbol(:bp, j)) = B + $boff))
        push!(inner.args, :($(Symbol(:br, j)) = $V(unsafe_load($(Symbol(:bp, j))))))
        push!(inner.args, :($(Symbol(:bi, j)) = $V(unsafe_load($(Symbol(:bp, j)) + $sz))))
        for mi in 1:MR
            cr = Symbol(:cr, mi, :_, j); ci = Symbol(:ci, mi, :_, j)
            ar = Symbol(:ar, mi); ai = Symbol(:ai, mi); br = Symbol(:br, j); bi = Symbol(:bi, j)
            push!(inner.args, :($cr = muladd($ar, $br, $cr)))
            push!(inner.args, :($cr = muladd($(SA * SB == 1 ? :(-$ai) : ai), $bi, $cr)))
            push!(inner.args, :($ci = muladd($(SB == 1 ? ar : :(-$ar)), $bi, $ci)))
            push!(inner.args, :($ci = muladd($(SA == 1 ? ai : :(-$ai)), $br, $ci)))
        end
    end
    push!(body.args, quote                                # k-reduction: 4 FMA/cell (complex) → @simd ivdep
        @inbounds @simd ivdep for p in 0:(k - 1)          # lets LLVM software-pipeline (register accs, no
            $inner                                        # in-loop memory dep; stores are in the epilogue).
        end                                               # FMA-density win — helps complex, would regress
    end)                                                  # the 1-FMA/cell real _microkernel_unpacked! (kb).
    # Store epilogue (OB's structure): interleave the split acc → zi=[zr,zi,…], then fold α (and, when
    # accumulating, C) into an FMA chain in the INTERLEAVED domain. REAL α ⇒ one FMA `zi·αr (+C)` (no
    # swap); complex α ⇒ add the cross term via a swapped-lane FMA with alternating-sign αi. The C load is
    # the FMA's addend memory operand (β=1 costs 0 extra instructions — OB parity; not a separate load+add).
    (A1 && B0) || push!(body.args, :(avr = $V2W(alr)))                  # α_re broadcast (all but A1-overwrite)
    AR || push!(body.args, :(aialt = $V2W(ali) * $V2W($signt)))         # (−α_im,α_im,…): complex-α cross term
    for j in 1:NR
        stores = quote end
        TRI && push!(stores.args, :(thr = d0 + $(j - 1)))               # tri-store threshold for this column
        for mi in 1:MR
            cr = Symbol(:cr, mi, :_, j); ci = Symbol(:ci, mi, :_, j)
            mk = TRI ? :mkl : Symbol(:m2, mi)                           # TRI ⇒ per-tile edge∧triangle mask (mkl)
            q = :(C + ((jr + $(j - 1)) * ldc * 2 + 2 * (ir + $((mi - 1) * W))) * $sz)
            vst(v) = FULL ? :(vstore($v, qq)) : :(vstore($v, qq, $mk))   # FULL ⇒ unmasked store/C-load
            cvl = FULL ? :(vload($V2W, qq)) : :(vload($V2W, qq, $mk))
            if B0 && A1                                                  # β=0, α=1: pure interleave store
                st = vst(:(shufflevector($cr, $ci, Val($ilv))))
            elseif B0                                                    # β=0: resv = α·z (real: 1 mul; complex: +cross)
                st = AR ? vst(:(avr * shufflevector($cr, $ci, Val($ilv)))) :
                          :(let ziv = shufflevector($cr, $ci, Val($ilv))
                                $(vst(:(muladd(ziv, avr, shufflevector(ziv, Val($swp)) * aialt)))) end)
            else                                                         # accumulate: resv = C + α·z; C folds into the FMA addend
                st = AR ? vst(:(muladd(shufflevector($cr, $ci, Val($ilv)), avr, $cvl))) :
                          :(let ziv = shufflevector($cr, $ci, Val($ilv))
                                $(vst(:(muladd(ziv, avr, muladd(shufflevector(ziv, Val($swp)), aialt, $cvl))))) end)
            end
            if TRI                                                       # edge mask ∧ triangle: upper keep row≤thr, lower row≥thr
                trik = :($(Symbol(:m2, mi)) & (upper ? (lanes2 < 2 * (thr - $((mi - 1) * W) + 1)) :
                                                       (lanes2 >= 2 * (thr - $((mi - 1) * W)))))
                push!(stores.args, :(let qq = $q, mkl = $trik; $st; end))
            else
                push!(stores.args, :(let qq = $q; $st; end))
            end
        end
        push!(body.args, :(if $(j - 1) < nre; $stores; end))
    end
    push!(body.args, :(return nothing))
    return body
end
# NR for the unpacked complex kernel: the mid-n bases (ztrsm/ztrmm off-diagonal, small zherk) run ~100%
# of their flops here and are LATENCY-bound — NR=4 gives only 8 accumulator chains ≈ Zen3's lat×tput
# (zero slack). NR=6 → 12 chains (fits 16 ymm: 12 accs + ar/ai + br/bi) hides the FMA latency. Tiny-n
# keeps NR=4 (NR=6's column remainder wastes more than the extra chains buy). Cutoff is per-box.
const _CUKER_NR6_MIN = @load_preference("cuker_nr6_min", _W64 == 4 ? 48 : typemax(Int))::Int
# TB/OV/A1/AR are Val TYPE-PARAMS (not value args) so the trimmer union-splits _uker_sweep! into concrete
# methods — a value `tB ? Val(true) : Val(false)` reaching `_uker_cmplx!`'s ::Val{TB} through an untyped
# arg is an "unresolved call" for --trim (the zgemm_64_/cgemm_64_ trim failure). Mirrors the real path.
@inline function _uker_sweep!(::Val{NR}, Cp, ldc, Ap, lda, Bp, ldb, m::Int, n::Int, k::Int,
        alr, ali, W::Int, mr::Int, ::Val{TB}, ::Val{SA}, ::Val{SB},
        ::Val{OV}, ::Val{A1}, ::Val{AR}) where {NR, TB, SA, SB, OV, A1, AR}
    jr = 0
    while jr < n
        nre = min(NR, n - jr)
        ir = 0
        while ir < m
            mre = min(mr, m - ir)
            nrv = cld(mre, W)
            if nrv >= _CMR
                if mre == mr                                         # full-row interior tile → unmasked
                    _uker_cmplx!(Cp, ldc, Ap, lda, ir, Bp, ldb, jr, k, alr, ali, mre, nre,
                        Val(_CMR), Val(NR), Val(TB), Val(SA), Val(SB), Val(OV), Val(A1), Val(AR), Val(true), Val(false), 0, true)
                else
                    _uker_cmplx!(Cp, ldc, Ap, lda, ir, Bp, ldb, jr, k, alr, ali, mre, nre,
                        Val(_CMR), Val(NR), Val(TB), Val(SA), Val(SB), Val(OV), Val(A1), Val(AR), Val(false), Val(false), 0, true)
                end
            else
                _uker_cmplx!(Cp, ldc, Ap, lda, ir, Bp, ldb, jr, k, alr, ali, mre, nre,
                    Val(1), Val(NR), Val(TB), Val(SA), Val(SB), Val(OV), Val(A1), Val(AR), Val(false), Val(false), 0, true)
            end
            ir += nrv >= _CMR ? mr : W
        end
        jr += NR
    end
    return
end
# Trim-safe flag resolution: the juliac/TrimCheck trimmer cannot union-split FOUR simultaneous
# Union{Val{true},Val{false}} args to _uker_sweep! (> the split limit — the real gemm gets away with 2).
# So resolve tB/b0/a1/ar to CONCRETE Vals through a chain of 2-way branches; each leaf call to _uker_sweep!
# has fully-concrete Val args → resolved for --trim. REQUIRED for zgemm_64_/cgemm_64_ (LBT). ­­args passed
# positionally to keep the chain terse.
@inline _res_tb!(nr, Cp, ldc, Ap, lda, Bp, ldb, m, n, k, alr, ali, W, mr, tB::Bool, sa, sb, b0, a1, ar) =
    tB ? _res_ov!(nr, Cp, ldc, Ap, lda, Bp, ldb, m, n, k, alr, ali, W, mr, Val(true), sa, sb, b0, a1, ar) :
         _res_ov!(nr, Cp, ldc, Ap, lda, Bp, ldb, m, n, k, alr, ali, W, mr, Val(false), sa, sb, b0, a1, ar)
@inline _res_ov!(nr, Cp, ldc, Ap, lda, Bp, ldb, m, n, k, alr, ali, W, mr, tb, sa, sb, b0::Bool, a1, ar) =
    b0 ? _res_a1!(nr, Cp, ldc, Ap, lda, Bp, ldb, m, n, k, alr, ali, W, mr, tb, sa, sb, Val(true), a1, ar) :
         _res_a1!(nr, Cp, ldc, Ap, lda, Bp, ldb, m, n, k, alr, ali, W, mr, tb, sa, sb, Val(false), a1, ar)
@inline _res_a1!(nr, Cp, ldc, Ap, lda, Bp, ldb, m, n, k, alr, ali, W, mr, tb, sa, sb, ov, a1::Bool, ar) =
    a1 ? _res_ar!(nr, Cp, ldc, Ap, lda, Bp, ldb, m, n, k, alr, ali, W, mr, tb, sa, sb, ov, Val(true), ar) :
         _res_ar!(nr, Cp, ldc, Ap, lda, Bp, ldb, m, n, k, alr, ali, W, mr, tb, sa, sb, ov, Val(false), ar)
@inline _res_ar!(nr, Cp, ldc, Ap, lda, Bp, ldb, m, n, k, alr, ali, W, mr, tb, sa, sb, ov, a1v, ar::Bool) =
    ar ? _uker_sweep!(nr, Cp, ldc, Ap, lda, Bp, ldb, m, n, k, alr, ali, W, mr, tb, sa, sb, ov, a1v, Val(true)) :
         _uker_sweep!(nr, Cp, ldc, Ap, lda, Bp, ldb, m, n, k, alr, ali, W, mr, tb, sa, sb, ov, a1v, Val(false))
function _gemm_cmplx_unpacked!(::Val{SA}, ::Val{SB}, tB::Bool, m::Int, n::Int, k::Int,
        alpha, A, B, beta, C) where {SA, SB}
    Tc = eltype(C); T = real(Tc)
    b0 = iszero(beta)
    a = convert(Tc, alpha); alr = real(a); ali = imag(a)
    b0 || _scale_C!(C, m, n, convert(Tc, beta))   # β=0 ⇒ kernel OVERWRITES; β≠0 ⇒ pre-scale then accumulate
    if iszero(alpha) || k == 0
        b0 && _scale_C!(C, m, n, zero(Tc))
        return C
    end
    a1 = isone(a); ar = iszero(ali)               # α=1 ⇒ pure interleave store; α real ⇒ single-FMA fold
    W = _vwidth(T); mr = _CMR * W
    lda = stride(A, 2); ldb = stride(B, 2); ldc = stride(C, 2)
    parA = parent(A); parB = parent(B); parC = parent(C)   # preserve parents, not view wrappers (no box)
    GC.@preserve parA parB parC begin
        Ap = Ptr{T}(pointer(A)); Bp = Ptr{T}(pointer(B)); Cp = Ptr{T}(pointer(C))
        if max(m, n, k) >= _CUKER_NR6_MIN         # full-tile mid-n: NR=6 (latency slack). tiny-n: NR=4.
            _res_tb!(Val(_CNR), Cp, ldc, Ap, lda, Bp, ldb, m, n, k, alr, ali, W, mr, tB, Val(SA), Val(SB), b0, a1, ar)
        else
            _res_tb!(Val(_CNR_SMALL), Cp, ldc, Ap, lda, Bp, ldb, m, n, k, alr, ali, W, mr, tB, Val(SA), Val(SB), b0, a1, ar)
        end
    end
    return C
end
# tA='N' ⇒ SA=1 (conj only rides transA='C', which sets tA); only cB matters.
function _gemm_cmplx_unpacked_go!(tB::Bool, cB::Bool, m::Int, n::Int, k::Int, alpha, A, B, beta, C)
    cB ? _gemm_cmplx_unpacked!(Val(1), Val(-1), tB, m, n, k, alpha, A, B, beta, C) :
         _gemm_cmplx_unpacked!(Val(1), Val(1), tB, m, n, k, alpha, A, B, beta, C)
end

# ── Karatsuba 3M complex GEMM ────────────────────────────────────────────────────────────────
# C := α·op(A)·op(B) + β·C via THREE real gemms on split re/im: P1=op(Ar)op(Br), P2=op(Ai)op(Bi),
# P3=op(Ar+Ai)op(Br+Bi); then Cᵣ=P1−P2, Cᵢ=P3−P1−P2. 25% fewer real flops than the 4-FMA complex
# kernel AND each pass runs the gating real microkernel (depth-1 FMA chains — the complex kernel is
# depth-2/latency-bound at mid-n). Measured to BEAT OB at mid-n (64×128×64 1.5×, 128³ 1.06×, 160³ 1.28×).
# Conj folds into the imag sign; the transpose rides the real sub-gemms. Error ~5e-16 vs the 4M oracle.
# Deinterleave M (r×c, any column stride) → Xr, Xi (Xi negated if conj), Xs = Xr+Xi (contiguous scratch).
# Write the top-left (r×c) block of the persistent X buffers (leading dim = stride(Xr,2), ≥ r).
function _split3!(Xr, Xi, Xs, M, conj::Bool, r::Int, c::Int)
    Tr = eltype(Xr); s = conj ? -one(Tr) : one(Tr)
    ldm = stride(M, 2); ldx = stride(Xr, 2)         # M col stride (complex, handles strided views); X col stride
    GC.@preserve M Xr Xi Xs begin
        pm = Ptr{Tr}(pointer(M)); pr = pointer(Xr); pi = pointer(Xi); ps = pointer(Xs)
        @inbounds for j in 1:c
            mb = (j - 1) * ldm * 2; xb = (j - 1) * ldx
            @simd for i in 1:r
                re = unsafe_load(pm, mb + 2i - 1); im = s * unsafe_load(pm, mb + 2i)
                unsafe_store!(pr, re, xb + i); unsafe_store!(pi, im, xb + i); unsafe_store!(ps, re + im, xb + i)
            end
        end
    end
    return
end
# Hermitian/symmetric split for 3M hemm/symm side-L: split A_herm (stored `up` triangle) into re/im/sum
# reading the triangle IN PLACE — no materialize, no n² complex scratch. Stored run reads A[i,j] direct
# (contiguous column, @simd); mirror run reads A[j,i] (row j, strided) with im negated for herm; herm
# diagonal → real (im=0). The 3M A-split IS the reflection — deletes the materialize pre-pass that
# dominated small/mid-n hemm/symm. herm=false ⇒ symm (mirror not conjugated).
function _split3_sym!(Xr, Xi, Xs, A, up::Bool, herm::Bool, n::Int)
    Tr = eltype(Xr); ms = herm ? -one(Tr) : one(Tr)
    lda = stride(A, 2); ldx = stride(Xr, 2)
    GC.@preserve A Xr Xi Xs begin
        pa = Ptr{Tr}(pointer(A)); pr = pointer(Xr); pii = pointer(Xi); ps = pointer(Xs)
        @inbounds for j in 1:n
            xb = (j - 1) * ldx; cj = (j - 1) * lda * 2                 # A column j (real offset)
            slo, shi = up ? (1, j) : (j, n)                            # stored rows: read A[i,j] direct
            @simd for i in slo:shi
                re = unsafe_load(pa, cj + 2i - 1); im = unsafe_load(pa, cj + 2i)
                unsafe_store!(pr, re, xb + i); unsafe_store!(pii, im, xb + i); unsafe_store!(ps, re + im, xb + i)
            end
            if herm                                                   # diagonal is real
                unsafe_store!(pii, zero(Tr), xb + j); unsafe_store!(ps, unsafe_load(pr, xb + j), xb + j)
            end
            mlo, mhi = up ? (j + 1, n) : (1, j - 1)                    # mirror rows: read A[j,i] (row j, strided)
            for i in mlo:mhi
                ci = (i - 1) * lda * 2
                re = unsafe_load(pa, ci + 2j - 1); im = ms * unsafe_load(pa, ci + 2j)
                unsafe_store!(pr, re, xb + i); unsafe_store!(pii, im, xb + i); unsafe_store!(ps, re + im, xb + i)
            end
        end
    end
    return
end
# C := α·(P1−P2 + i(P3−P1−P2)) + β·C.  β=0 ⇒ skip the C read (overwrite).
function _combine3!(C, P1, P2, P3, alpha::Tc, beta::Tc, m::Int, n::Int) where {Tc}
    Tr = real(Tc); ar = real(alpha); ai = imag(alpha); br = real(beta); bi = imag(beta); b0 = iszero(beta)
    ldc = stride(C, 2); ldp = stride(P1, 2)   # C col stride (complex); P col stride (real, ≥ m — top-left block)
    GC.@preserve C P1 P2 P3 begin
        pc = Ptr{Tr}(pointer(C)); p1 = pointer(P1); p2 = pointer(P2); p3 = pointer(P3)
        @inbounds for j in 1:n
            cb = (j - 1) * ldc * 2; pb = (j - 1) * ldp
            if b0                                                 # β=0: overwrite (no C read)
                @simd for i in 1:m
                    a = unsafe_load(p1, pb + i); b = unsafe_load(p2, pb + i)
                    zr = a - b; zi = unsafe_load(p3, pb + i) - a - b
                    unsafe_store!(pc, ar * zr - ai * zi, cb + 2i - 1); unsafe_store!(pc, ar * zi + ai * zr, cb + 2i)
                end
            else                                                  # C := α·z + β·C
                @simd for i in 1:m
                    a = unsafe_load(p1, pb + i); b = unsafe_load(p2, pb + i)
                    zr = a - b; zi = unsafe_load(p3, pb + i) - a - b
                    or = unsafe_load(pc, cb + 2i - 1); oi = unsafe_load(pc, cb + 2i)
                    unsafe_store!(pc, ar * zr - ai * zi + br * or - bi * oi, cb + 2i - 1)
                    unsafe_store!(pc, ar * zi + ai * zr + br * oi + bi * or, cb + 2i)
                end
            end
        end
    end
    return
end
# Real gemm on the top-left (m×n) block of persistent max-sized buffers: explicit logical (m,n,k) + the
# matrix's own leading dim, so no per-call wrapping/allocation. Mirrors _gemm_core!'s real dispatch.
@inline function _gemm_real_dims!(tA::Bool, tB::Bool, m::Int, n::Int, k::Int, alpha::T, beta::T, A, B, C) where {T}
    if !tA && _use_unpacked(m, n, k)
        _gemm_unpacked!(tB ? Val(true) : Val(false), iszero(beta) ? Val(true) : Val(false), m, n, k, alpha, A, B, beta, C)
    else
        _gemm_blocked!(tA, tB, m, n, k, alpha, A, B, beta, C)   # blocked also covers the transpose case
    end
    return C
end
function _gemm_3m!(tA::Bool, tB::Bool, cA::Bool, cB::Bool, m::Int, n::Int, k::Int, alpha, A, B, beta, C)
    Tc = eltype(C); Tr = real(Tc)
    ra = size(A, 1); ca = size(A, 2); rb = size(B, 1); cb = size(B, 2)   # stored dims (trans folded by sub-gemm)
    t = _gemm_3m_scratch(Tr, ra * ca, rb * cb, m * n)   # grow-only flat buffers
    GC.@preserve t begin      # unsafe_wrap the first r·c → CONTIGUOUS r×c matrix (ld=r; no strided top-left)
        w(i, r, c) = unsafe_wrap(Array, pointer(t[i]), (r, c))
        Ar = w(1, ra, ca); Ai = w(2, ra, ca); As = w(3, ra, ca)
        Br = w(4, rb, cb); Bi = w(5, rb, cb); Bs = w(6, rb, cb)
        P1 = w(7, m, n); P2 = w(8, m, n); P3 = w(9, m, n)
        _split3!(Ar, Ai, As, A, cA, ra, ca); _split3!(Br, Bi, Bs, B, cB, rb, cb)
        o = one(Tr); z = zero(Tr)
        _gemm_real_dims!(tA, tB, m, n, k, o, z, Ar, Br, P1)
        _gemm_real_dims!(tA, tB, m, n, k, o, z, Ai, Bi, P2)
        _gemm_real_dims!(tA, tB, m, n, k, o, z, As, Bs, P3)
        _combine3!(C, P1, P2, P3, convert(Tc, alpha), convert(Tc, beta), m, n)
    end
    return C
end
# Karatsuba-3M complex hemm/symm side-L: C := α·A_herm·B + β·C, with A split from its stored triangle by
# _split3_sym! (no materialize, no n² complex scratch). Mirror of _gemm_3m! (tA=tB='N', k=n=size(A,1)).
# herm=false ⇒ symm. `_combine3!` applies α and β (β=0 overwrites), so no separate scaleC. This deletes
# the materialize pre-pass for complex hemm/symm in the 3M window — the 3M split already reads all of A.
function _hemm_3m_L!(up::Bool, herm::Bool, α, β, A, B, C)
    Tc = eltype(C); Tr = real(Tc); n = size(A, 1); m = size(B, 2)
    t = _gemm_3m_scratch(Tr, n * n, size(B, 1) * m, n * m)
    GC.@preserve t begin
        w(i, r, c) = unsafe_wrap(Array, pointer(t[i]), (r, c))
        Ar = w(1, n, n); Ai = w(2, n, n); As = w(3, n, n)
        Br = w(4, n, m); Bi = w(5, n, m); Bs = w(6, n, m)
        P1 = w(7, n, m); P2 = w(8, n, m); P3 = w(9, n, m)
        _split3_sym!(Ar, Ai, As, A, up, herm, n)
        _split3!(Br, Bi, Bs, B, false, n, m)
        o = one(Tr); z = zero(Tr)
        _gemm_real_dims!(false, false, n, m, n, o, z, Ar, Br, P1)
        _gemm_real_dims!(false, false, n, m, n, o, z, Ai, Bi, P2)
        _gemm_real_dims!(false, false, n, m, n, o, z, As, Bs, P3)
        _combine3!(C, P1, P2, P3, convert(Tc, α), convert(Tc, β), n, m)
    end
    return C
end

# ── Strassen-Winograd real GEMM ──────────────────────────────────────────────────────────────
# C := α·A·B + β·C, recursive 2×2 (Douglas-Heroux-Slishman-Smith form, 15 adds). Sub-products run with
# α=1,β=0; the top-level combine applies α,β. Buffers come from the per-level workspace pool; quadrants
# are views (tiny headers, negligible at Strassen's large n). Dims must be divisible by 2^depth (the
# entry pads odd/awkward sizes). Verified symbolically + numerically (~1e-14 vs the OB oracle).
function _strassen_rec!(C, A, Bm, depth::Int, level::Int, alpha::T, beta::T) where {T}
    if depth == 0
        return _gemm_real_dims!(false, false, size(C, 1), size(C, 2), size(A, 2), alpha, beta, A, Bm, C)
    end
    m = size(C, 1); n = size(C, 2); k = size(A, 2); mh = m ÷ 2; nh = n ÷ 2; kh = k ÷ 2
    A11 = @view A[1:mh, 1:kh]; A12 = @view A[1:mh, (kh+1):k]; A21 = @view A[(mh+1):m, 1:kh]; A22 = @view A[(mh+1):m, (kh+1):k]
    B11 = @view Bm[1:kh, 1:nh]; B12 = @view Bm[1:kh, (nh+1):n]; B21 = @view Bm[(kh+1):k, 1:nh]; B22 = @view Bm[(kh+1):k, (nh+1):n]
    C11 = @view C[1:mh, 1:nh]; C12 = @view C[1:mh, (nh+1):n]; C21 = @view C[(mh+1):m, 1:nh]; C22 = @view C[(mh+1):m, (nh+1):n]
    TA, TB, P1, P2, P3, P4, P5, P6, P7, U = _strassen_lvl_scratch(T, level, mh, nh, kh)
    o = one(T); z = zero(T); dm = depth - 1; lv = level + 1
    @. TA = A21 + A22; @. TB = B12 - B11; _strassen_rec!(P5, TA, TB, dm, lv, o, z)   # S1,T1 → P5
    @. TA = TA - A11;  @. TB = B22 - TB;  _strassen_rec!(P6, TA, TB, dm, lv, o, z)   # S2,T2 → P6
    @. TB = TB - B21;                     _strassen_rec!(P4, A22, TB, dm, lv, o, z)  # T4 → P4
    @. TA = A12 - TA;                     _strassen_rec!(P3, TA, B22, dm, lv, o, z)  # S4 → P3
    @. TA = A11 - A21; @. TB = B22 - B12; _strassen_rec!(P7, TA, TB, dm, lv, o, z)   # S3,T3 → P7
    _strassen_rec!(P1, A11, B11, dm, lv, o, z)
    _strassen_rec!(P2, A12, B21, dm, lv, o, z)
    @. U = P1 + P6                                                                   # U1
    if iszero(beta)
        @. C11 = alpha * (P1 + P2); @. C12 = alpha * (U + P5 + P3)
        @. C21 = alpha * (U + P7 - P4); @. C22 = alpha * (U + P7 + P5)
    else
        @. C11 = alpha * (P1 + P2) + beta * C11; @. C12 = alpha * (U + P5 + P3) + beta * C12
        @. C21 = alpha * (U + P7 - P4) + beta * C21; @. C22 = alpha * (U + P7 + P5) + beta * C22
    end
    return C
end
# Entry: pick adaptive depth, pad m,n,k up to a multiple of 2^depth (odd-n) if needed, recurse.
function _gemm_strassen!(m::Int, n::Int, k::Int, alpha, A, B, beta, C)
    T = eltype(C); D = _strassen_depth(m, n, k); p = 1 << D
    mp = cld(m, p) * p; np = cld(n, p) * p; kp = cld(k, p) * p
    a = convert(T, alpha); b = convert(T, beta)
    if mp == m && np == n && kp == k                       # already clean — recurse in place (β applied at top)
        _strassen_rec!(C, A, B, D, 0, a, b)
    else                                                   # odd/awkward: pad to even^D with zeros, copy back
        Ap, Bp, Cp = _strassen_pad_scratch(T, mp, kp, np)
        fill!(Ap, zero(T)); @inbounds @views Ap[1:m, 1:k] .= A
        fill!(Bp, zero(T)); @inbounds @views Bp[1:k, 1:n] .= B
        _strassen_rec!(Cp, Ap, Bp, D, 0, one(T), zero(T))
        Cv = @view Cp[1:m, 1:n]
        iszero(b) ? (@inbounds @. C = a * Cv) : (@inbounds @. C = a * Cv + b * C)
    end
    return C
end

"""
    gemm!(C, A, B; alpha=1, beta=0, transA='N', transB='N')

In-place GEMM: `C := alpha·op(A)·op(B) + beta·C`, with `op` set by `transA`/`transB`
(`'N'`/`'T'`/`'C'`). Real and complex dense (unit column stride) `C` use SIMD paths (real: blocked/
unpacked; complex: split-pack blocked, or an unpacked tiny-n path); AD element types and strided
`C` use the generic scalar path.
"""
# Dispatch core (no kwargs, no dim checks) — callers that already know shapes are valid (e.g. the trsm/trmm
# recursion's off-diagonal updates) call this directly to skip gemm!'s public-entry overhead. `@inline` so
# the branch cascade folds at the call site (the kb "call the inner kernel, skip the kwarg layer" fix).
# Direct tiny GEMM (max dim ≤ _GEMM_TINY): a plain register-accumulated triple loop. The masked 16×8
# unpacked micro-kernel costs ~100–135 ns on a 2×2..6×6 problem (mask setup + dead lanes) — more than
# OpenBLAS's entire ccall; the naive loop wins below W-sized problems.
const _GEMM_TINY = 6
# !tA tiny path: each C column is ONE masked W-vector (m ≤ W). Per column: k masked A-column loads ×
# broadcast B scalars — FMA chains are per-column (ILP across the j loop), no 16-row mask machinery.
function _gemm_tiny_vec!(C, A, B, alpha::T, beta::T, tB::Bool, m::Int, n::Int, k::Int) where {T<:BlasReal}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T)
    lda = stride(A, 2); ldb = stride(B, 2); ldc = stride(C, 2)
    lanes = Vec{W, Int}(ntuple(i -> i - 1, Val(W))); mask = lanes < m
    av = V(alpha); b0 = iszero(beta)
    parA = parent(A); parB = parent(B); parC = parent(C)   # preserve parents, not view wrappers (no box)
    GC.@preserve parA parB parC begin
        pA = pointer(A); pB = pointer(B); pC = pointer(C)
        @inbounds for j in 0:(n - 1)
            acc = zero(V)
            for p in 0:(k - 1)
                bsc = tB ? unsafe_load(pB, j + p * ldb + 1) : unsafe_load(pB, p + j * ldb + 1)
                acc = muladd(vload(V, pA + p * lda * sz, mask), V(bsc), acc)
            end
            q = pC + j * ldc * sz
            res = av * acc
            b0 ? vstore(res, q, mask) : vstore(muladd(V(beta), vload(V, q, mask), res), q, mask)
        end
    end
    return C
end
function _gemm_tiny_v!(C, A, B, alpha::T, beta::T, ::Val{TA}, ::Val{TB}, m::Int, n::Int, k::Int) where {T, TA, TB}
    b0 = iszero(beta)
    @inbounds for j in 1:n, i in 1:m
        s = zero(T)
        for p in 1:k
            a = TA ? A[p, i] : A[i, p]
            b = TB ? B[j, p] : B[p, j]
            s = muladd(a, b, s)
        end
        C[i, j] = b0 ? alpha * s : muladd(beta, C[i, j], alpha * s)
    end
    return C
end
@inline function _gemm_tiny!(C, A, B, alpha::T, beta::T, tA::Bool, tB::Bool, m::Int, n::Int, k::Int) where {T}
    if !tA && T <: BlasReal && _strided1(A) && _strided1(B) && m <= _vwidth(T)
        return _gemm_tiny_vec!(C, A, B, alpha, beta, tB, m, n, k)
    end
    if tA
        tB ? _gemm_tiny_v!(C, A, B, alpha, beta, Val(true), Val(true), m, n, k) :
             _gemm_tiny_v!(C, A, B, alpha, beta, Val(true), Val(false), m, n, k)
    else
        tB ? _gemm_tiny_v!(C, A, B, alpha, beta, Val(false), Val(true), m, n, k) :
             _gemm_tiny_v!(C, A, B, alpha, beta, Val(false), Val(false), m, n, k)
    end
    return C
end
@inline function _gemm_core!(C, A, B, alpha::T, beta::T, tA::Bool, tB::Bool, cA::Bool, cB::Bool) where {T}
    m = size(C, 1); n = size(C, 2); k = tA ? size(A, 1) : size(A, 2)
    if T <: BlasReal && _strided1(C)
        if max(m, n, k) <= _GEMM_TINY && !cA && !cB
            return _gemm_tiny!(C, A, B, alpha, beta, tA, tB, m, n, k)
        end
        if _STRASSEN && !tA && !tB && _strided1(A) && _strided1(B) && _strassen_depth(m, n, k) > 0
            return _gemm_strassen!(m, n, k, alpha, A, B, beta, C)   # large-n real: 7-mult recursion beats OB
        end
        if _strided1(A) && _strided1(B) && _use_unpacked(m, n, k)
            if !tA
                _gemm_unpacked!(tB ? Val(true) : Val(false), iszero(beta) ? Val(true) : Val(false),
                    m, n, k, alpha, A, B, beta, C)
            else
                At, _ = _gemm_scratch(T, m * k, 0)
                _transpose_dense!(At, A, m, k)
                Am = reshape(view(At, 1:(m * k)), m, k)
                _gemm_unpacked!(tB ? Val(true) : Val(false), iszero(beta) ? Val(true) : Val(false),
                    m, n, k, alpha, Am, B, beta, C)
            end
        else
            _gemm_blocked!(tA, tB, m, n, k, alpha, A, B, beta, C)
        end
    elseif T <: BlasComplex && _strided1(C) && max(m, n, k) > _CGEMM_TINY
        if _CGEMM_3M && _CGEMM_3M_MIN <= max(m, n, k) <= _CGEMM_3M_MAX && min(m, n, k) >= _CGEMM_3M_KMIN
            _gemm_3m!(tA, tB, cA, cB, m, n, k, alpha, A, B, beta, C)   # Karatsuba 3M: beats OB at mid-n
        elseif !tA && _strided1(A) && _strided1(B) && max(m, n, k) <= _CGEMM_UNPACK_MAX
            _gemm_cmplx_unpacked_go!(tB, cB, m, n, k, alpha, A, B, beta, C)
        else
            _gemm_cmplx_blocked!(tA, tB, cA, cB, m, n, k, alpha, A, B, beta, C)
        end
    else
        _gemm_generic!(tA, tB, cA, cB, m, n, k, alpha, A, B, beta, C)
    end
    return C
end
function gemm!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix;
        alpha = one(eltype(C)), beta = zero(eltype(C)), transA::Char = 'N', transB::Char = 'N')
    T = eltype(C)
    tA = transA != 'N'; tB = transB != 'N'
    m = size(C, 1); n = size(C, 2)
    k = tA ? size(A, 1) : size(A, 2)
    (tA ? size(A, 2) : size(A, 1)) == m ||
        throw(DimensionMismatch("gemm!: op(A) rows ≠ size(C,1)=$m"))
    (tB ? size(B, 1) : size(B, 2)) == n ||
        throw(DimensionMismatch("gemm!: op(B) cols ≠ size(C,2)=$n"))
    (tB ? size(B, 2) : size(B, 1)) == k ||
        throw(DimensionMismatch("gemm!: inner dimensions disagree (k=$k)"))
    if T <: BlasFloat && C isa StridedMatrix && stride(C, 1) == 1
        _gemm_core!(C, A, B, T(alpha), T(beta), tA, tB, transA == 'C', transB == 'C')
    else
        _gemm_generic!(tA, tB, transA == 'C', transB == 'C', m, n, k, alpha, A, B, beta, C)
    end
    return C
end

"""
    gemm(A, B; transA='N', transB='N') -> C

Allocating GEMM: returns `op(A)·op(B)`.
"""
function gemm(A::AbstractMatrix, B::AbstractMatrix; transA::Char = 'N', transB::Char = 'N')
    T = promote_type(eltype(A), eltype(B))
    m = transA != 'N' ? size(A, 2) : size(A, 1)
    n = transB != 'N' ? size(B, 1) : size(B, 2)
    C = zeros(T, m, n)
    return gemm!(C, A, B; alpha = one(T), beta = zero(T), transA, transB)
end
