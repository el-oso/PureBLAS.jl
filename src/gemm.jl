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

const _MR = 2     # microkernel height in vectors  → mr = _MR * W rows (F64: 16, F32: 32)
const _NR = 8     # microkernel width in columns (MR*NR = 16 accumulators — Zen4 sweet spot;
#                   3×8=24 spilled and regressed, 2×6 left registers idle)
const _MC = 144   # A row block (L2); rounded down to a multiple of mr at runtime
const _NC = 2040  # B col block (L3); rounded down to a multiple of nr
const _KC = 256   # contraction block — sized so the B micropanel (kc·nr·8) ≈ ½ L1, per BLIS
#                   (kc=512 filled all of L1 and starved A streaming → slower large-n)

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
@inline function _pack_A_simd!(Ap::Vector{T}, A::StridedMatrix{T}, ic::Int, pc::Int, mce::Int,
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
@inline function _pack_A_simd_T!(Ap::Vector{T}, A::StridedMatrix{T}, ic::Int, pc::Int, mce::Int,
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
@inline function _transpose_dense!(At::Vector{T}, A::StridedMatrix{T}, m::Int, k::Int) where {T}
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
function _pack_A!(Ap::Vector{T}, A, ic::Int, pc::Int, mce::Int, kce::Int, tA::Bool, alpha::T, mr::Int) where {T}
    if !tA && A isa StridedMatrix{T} && stride(A, 1) == 1
        return _pack_A_simd!(Ap, A, ic, pc, mce, kce, alpha, mr)
    end
    if tA && A isa StridedMatrix{T} && stride(A, 1) == 1 && T <: BlasReal && mr % _vwidth(T) == 0
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
        if ji * nr + nr <= nce                       # full panel: branch-free, `@simd ivdep` vectorizes the
            j0 = jc + ji * nr                        # contiguous stores (reads stay elementwise — a wide
            if tB                                    # vload here stalls on freshly-written C operands, e.g.
                for p in 0:(kce - 1)                 # geqrf's trailing update: measured −25% with _tblk!).
                    gp = pc + p
                    @simd ivdep for c in 0:(nr - 1)
                        Bp[base + p * nr + c + 1] = B[j0 + c + 1, gp + 1]
                    end
                end
            else
                for p in 0:(kce - 1)
                    gp = pc + p
                    @simd ivdep for c in 0:(nr - 1)
                        Bp[base + p * nr + c + 1] = B[gp + 1, j0 + c + 1]
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

# Reusable packing buffers, keyed by element type, grown on demand — avoids a per-call ~MB malloc
# that dominates small/medium GEMM. ponytail: single global scratch; make task-local for M4 threading
# (and revisit if dgemm_64_ ever enters the trim build — a global Dict isn't trim-safe).
const _GEMM_SCRATCH = Dict{DataType, Tuple{Vector, Vector}}()
function _gemm_scratch(::Type{T}, lenA::Int, lenB::Int) where {T}
    Ap, Bp = get!(() -> (T[], T[]), _GEMM_SCRATCH, T)::Tuple{Vector{T}, Vector{T}}
    length(Ap) < lenA && resize!(Ap, lenA)
    length(Bp) < lenB && resize!(Bp, lenB)
    return Ap, Bp
end

# Blocked real GEMM (the optimized path). C must have unit column stride (pointer + vstore).
function _gemm_blocked!(tA::Bool, tB::Bool, m::Int, n::Int, k::Int,
        alpha::T, A, B, beta::T, C) where {T<:BlasReal}
    _scale_C!(C, m, n, beta)
    (iszero(alpha) || k == 0) && return C
    W = _vwidth(T); mr = _MR * W; nr = _NR
    # Cap block sizes by the actual problem so small GEMMs don't allocate/pack huge panels.
    kc = min(_KC, k)
    mc = min(max(mr, (_MC ÷ mr) * mr), cld(m, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(n, nr) * nr)
    Ap, Bp = _gemm_scratch(T, cld(mc, mr) * mr * kc, cld(nc, nr) * nr * kc)
    ldc = stride(C, 2); sz = sizeof(T)
    GC.@preserve C Ap Bp begin
        Cp0 = pointer(C); App = pointer(Ap); Bpp = pointer(Bp)
        jc = 0
        while jc < n
            nce = min(nc, n - jc)
            pc = 0
            while pc < k
                kce = min(kc, k - pc)
                _pack_B!(Bp, B, pc, jc, kce, nce, tB, nr)
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
                            Bpanel = Bpp + (div(jr, nr) * nr * kce) * sz
                            Cblk = Cp0 + ((ic + ir) + (jc + jr) * ldc) * sz
                            if mre == mr && nre == nr
                                _microkernel!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel),
                                    kce, Val(_MR), Val(_NR))
                            else
                                _microkernel_masked!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel),
                                    kce, mre, nre, Val(_MR), Val(_NR))
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
const _GEMM_UNPACK_MAX = 448
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
function _gemm_unpacked!(::Val{TB}, ::Val{B0}, m::Int, n::Int, k::Int,
        alpha::T, A, B, beta::T, C) where {T<:BlasReal, TB, B0}
    if iszero(alpha) || k == 0
        _scale_C!(C, m, n, beta)   # C := beta·C (or 0); no contraction to do
        return C
    end
    W = _vwidth(T); mr = _MR * W; nr = _NR
    lda = stride(A, 2); ldb = stride(B, 2); ldc = stride(C, 2)
    GC.@preserve A B C begin
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
                    elseif nre == nr                 # partial rows, full columns
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

"""
    gemm!(C, A, B; alpha=1, beta=0, transA='N', transB='N')

In-place GEMM: `C := alpha·op(A)·op(B) + beta·C`, with `op` set by `transA`/`transB`
(`'N'`/`'T'`/`'C'`). Real dense (unit column stride) `C` uses the blocked SIMD path; everything
else (complex, AD element types, strided `C`) uses the generic path.
"""
# Dispatch core (no kwargs, no dim checks) — callers that already know shapes are valid (e.g. the trsm/trmm
# recursion's off-diagonal updates) call this directly to skip gemm!'s public-entry overhead. `@inline` so
# the branch cascade folds at the call site (the kb "call the inner kernel, skip the kwarg layer" fix).
@inline function _gemm_core!(C, A, B, alpha::T, beta::T, tA::Bool, tB::Bool, cA::Bool, cB::Bool) where {T}
    m = size(C, 1); n = size(C, 2); k = tA ? size(A, 1) : size(A, 2)
    if T <: BlasReal && C isa StridedMatrix && stride(C, 1) == 1
        if A isa StridedMatrix && B isa StridedMatrix && stride(A, 1) == 1 &&
                stride(B, 1) == 1 && _use_unpacked(m, n, k)
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
    if T <: BlasReal && C isa StridedMatrix && stride(C, 1) == 1
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
