# SME (Scalable Matrix Extension) Float64 GEMM kernel for Apple Silicon.
#
# WHY THIS EXISTS. On Apple Silicon the unit that does fast FP64 matrix work is SME, not NEON.
# `fmopa za.d` is an 8x8 FP64 outer product — 64 FMAs, 128 flops, in one instruction — against
# NEON's 16 flops/cycle. Measured on an M6: NEON FP64 roofline 60.4 GFLOP/s, OpenBLAS 66 (i.e.
# already at that roofline), this kernel 549. The gap is a different execution unit, not tuning.
#
# The kernel is LLVM IR through `Base.llvmcall`, not inline assembly. Two structural rules make
# that work in a live Julia session:
#
#   1. `entry` is a trivial wrapper and the real body is a `noinline` sibling. `emit_llvmcall`
#      marks the llvmcall function `AlwaysInline`, and `AlwaysInlinerPass` does not consult
#      `AArch64TTIImpl::areInlineCompatible` — the check that otherwise refuses to merge a
#      streaming function into a non-streaming caller. Without the sibling the body is inlined
#      into ordinary Julia code, loses `aarch64_pstate_sm_body`, and executes z-register
#      instructions with PSTATE.SM = 0 (SIGILL).
#   2. `aarch64_inout_za` plus explicit `za.enable`/`za.disable`, NOT `aarch64_new_za`. The latter
#      emits prologue calls to the Arm SME ABI routines (`__arm_tpidr2_save` and friends), which
#      live as private-external symbols inside `libclang_rt.osx.a` and are exported by no system
#      dylib, so the JIT cannot resolve them.
#
# Target features are `+sme,+sme2,+sme-f64f64` and deliberately NOT `+sve`: with `+sve` LLVM uses
# SVE for ordinary operations OUTSIDE streaming mode, and this hardware has no non-streaming SVE.
#
# SAFETY. A kernel must be a leaf — no allocation, yield, or throw inside the streaming region.
# The garbage collector is not a hazard: Julia parks threads by guard-page polling and a leaf
# kernel never polls, so a collection waits out the call in flight (measured: a full collection
# took 3.3 ms against a 225 ms kernel loop) rather than landing inside one. Verified over 82,442
# calls under a continuous allocate-and-collect storm with every value exact.

# WHERE THIS CODE BELONGS. This file is the one place hand-written LLVM IR lives in PureBLAS —
# roughly 100 lines of it, emitted by Julia functions that interpolate the detected tile geometry,
# so changing `_SME_LANES` changes the kernel. Everything else about SME is two detection constants
# in `cpuinfo.jl`, one dispatch line in `gemm.jl`, and one `__init__` hook in `lbt.jl`.
#
# The long-term aim is to find every extension and kernel this hardware needs, then upstream the lot
# to SIMD.jl — which already generates LLVM IR itself (49 `llvmcall` sites and a whole
# `LLVM_intrinsics.jl`), so generated IR is not the obstacle. The obstacle is a TYPE:
# `<vscale x 2 x double>` is a SCALABLE vector with no Julia representation, while `Vec{N,T}` is
# fixed-width by construction. Hosting this needs a new type concept, not a new function.
# That work can come LATER: no scalable type crosses the Julia boundary here — every entry point
# takes `Ptr` and `Int` — so these kernels are shippable exactly as they stand.
#
# A package extension was evaluated and REJECTED. It works mechanically: a package may appear in
# both `[deps]` and `[weakdeps]`, and the extension fires when the user loads the trigger — verified,
# including that it does NOT fire when the trigger merely arrives as a transitive dependency. It is
# wrong here because the `--trim` build does not load extensions, so `libpureblas.so` (Mode 1, the
# C-host drop-in) would silently lose SME and fall back to NEON. Mode 1 needs the matrix unit, so
# the kernel stays in `src/`.

# ZA holds the C tile as four 8x8 FP64 tiles:
#     za0 = rows 0-7,  cols 0-7      za1 = rows 0-7,  cols 8-15
#     za2 = rows 8-15, cols 0-7      za3 = rows 8-15, cols 8-15
# so the register tile is _SME_MR x _SME_NR. Four tiles is enough to saturate issue at one
# `fmopa` per cycle (measured 546 of a 549 ceiling in context), so the remaining four ZA tiles
# would buy panel traffic, not compute.
# Geometry falls back to the 8-lane shape on machines that report nothing, so the IR strings
# still build there. They are never executed: every entry point is gated on `_SME_F64`.
const _SME_L = (_SME_F64 && _SME_LANES > 0) ? _SME_LANES : 8
const _SME_MR = 2 * _SME_L
const _SME_NR = 2 * _SME_L

# (ZA tile, slice, C column, C row offset) for each of the 32 vertical slices of the C tile.
# VERTICAL slices, not horizontal: a vertical slice is one column over all 8 rows, which is
# exactly the contiguous run a column-major C holds.
function _sme_c_slots()
    slots = Tuple{Int, Int, Int, Int}[]
    L = _SME_L
    for jj in 0:(L - 1)
        push!(slots, (0, jj, jj, 0))
        push!(slots, (2, jj, jj, L))
        push!(slots, (1, jj, jj + L, 0))
        push!(slots, (3, jj, jj + L, L))
    end
    return slots
end

const _SME_ATTRS = """
attributes #0 = { noinline "aarch64_inout_za" "aarch64_pstate_sm_body"
  "target-features"="+sme,+sme2,+sme-f64f64" }
"""

const _SME_DECLS = """
declare void @llvm.aarch64.sme.zero(i32)
declare void @llvm.aarch64.sme.za.enable()
declare void @llvm.aarch64.sme.za.disable()
declare void @llvm.aarch64.sme.mopa.nxv2f64(i32, <vscale x 2 x i1>, <vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)
declare void @llvm.aarch64.sme.ld1d.vert(<vscale x 2 x i1>, ptr, i32, i32)
declare void @llvm.aarch64.sme.st1d.vert(<vscale x 2 x i1>, ptr, i32, i32)
declare void @llvm.aarch64.sme.ld1d.horiz(<vscale x 2 x i1>, ptr, i32, i32)
declare void @llvm.aarch64.sme.st1d.horiz(<vscale x 2 x i1>, ptr, i32, i32)
"""

# ── Macrokernel ────────────────────────────────────────────────────────────────────────────────
# The panel loops live INSIDE the streaming region, so streaming mode is entered once per
# macrokernel call rather than once per register tile.
#
# C is pre-offset by the caller to the (ic, jc) corner; tile (ip, jp) writes
# C + (ip*MR + jp*NR*ldc) and reads Ap + ip*kce*MR, Bp + jp*kce*NR.
#
# beta == 0 zeroes ZA; otherwise C is PRELOADED into ZA and the `fmopa`s accumulate on top, so
# the epilogue is a plain store with no read-modify-write. Same instruction count either way.
function _sme_macro_ir(overwrite::Bool)
    slots = _sme_c_slots()
    L = _SME_L
    io = IOBuffer()
    print(io, _SME_DECLS)
    print(io, """
define void @entry(ptr %c, i64 %ldc, ptr %ap, ptr %bp, i64 %nip, i64 %njp, i64 %kce) {
  call void @macro(ptr %c, i64 %ldc, ptr %ap, ptr %bp, i64 %nip, i64 %njp, i64 %kce)
  ret void
}

define internal void @macro(ptr %c, i64 %ldc, ptr %ap, ptr %bp, i64 %nip, i64 %njp, i64 %kce) #0 {
entry:
  call void @llvm.aarch64.sme.za.enable()
  %kcmr = mul nsw i64 %kce, $(_SME_MR)
  %kcnr = mul nsw i64 %kce, $(_SME_NR)
  %ldcnr = mul nsw i64 %ldc, $(_SME_NR)
  %anyi = icmp sgt i64 %nip, 0
  %anyj = icmp sgt i64 %njp, 0
  %go = and i1 %anyi, %anyj
  br i1 %go, label %jploop, label %done

jploop:
  %jp = phi i64 [ 0, %entry ], [ %jpn, %jpend ]
  %bo = mul nsw i64 %jp, %kcnr
  %bpj = getelementptr inbounds double, ptr %bp, i64 %bo
  %co = mul nsw i64 %jp, %ldcnr
  %cj = getelementptr inbounds double, ptr %c, i64 %co
  br label %iploop

iploop:
  %ip = phi i64 [ 0, %jploop ], [ %ipn, %ipend ]
  %ao = mul nsw i64 %ip, %kcmr
  %api = getelementptr inbounds double, ptr %ap, i64 %ao
  %cio = mul nsw i64 %ip, $(_SME_MR)
  %ci = getelementptr inbounds double, ptr %cj, i64 %cio
""")
    if overwrite
        println(io, "  call void @llvm.aarch64.sme.zero(i32 255)")
    else
        for (n, (tile, sl, col, row)) in enumerate(slots)
            println(io, "  %lo$n = mul nsw i64 %ldc, $col")
            println(io, "  %lp$n = getelementptr inbounds double, ptr %ci, i64 %lo$n")
            println(io, "  %lq$n = getelementptr inbounds double, ptr %lp$n, i64 $row")
            println(io, "  call void @llvm.aarch64.sme.ld1d.vert(<vscale x 2 x i1> splat (i1 true), ptr %lq$n, i32 $tile, i32 $sl)")
        end
    end
    print(io, """
  %kpos = icmp sgt i64 %kce, 0
  br i1 %kpos, label %kloop, label %tstore

kloop:
  %kk = phi i64 [ 0, %iploop ], [ %kkn, %kloop ]
  %aoff = mul nsw i64 %kk, $(_SME_MR)
  %boff = mul nsw i64 %kk, $(_SME_NR)
  %pa0 = getelementptr inbounds double, ptr %api, i64 %aoff
  %pa1 = getelementptr inbounds double, ptr %pa0, i64 $L
  %pb0 = getelementptr inbounds double, ptr %bpj, i64 %boff
  %pb1 = getelementptr inbounds double, ptr %pb0, i64 $L
  %va0 = load <vscale x 2 x double>, ptr %pa0, align 8
  %va1 = load <vscale x 2 x double>, ptr %pa1, align 8
  %vb0 = load <vscale x 2 x double>, ptr %pb0, align 8
  %vb1 = load <vscale x 2 x double>, ptr %pb1, align 8
  call void @llvm.aarch64.sme.mopa.nxv2f64(i32 0, <vscale x 2 x i1> splat (i1 true), <vscale x 2 x i1> splat (i1 true), <vscale x 2 x double> %va0, <vscale x 2 x double> %vb0)
  call void @llvm.aarch64.sme.mopa.nxv2f64(i32 1, <vscale x 2 x i1> splat (i1 true), <vscale x 2 x i1> splat (i1 true), <vscale x 2 x double> %va0, <vscale x 2 x double> %vb1)
  call void @llvm.aarch64.sme.mopa.nxv2f64(i32 2, <vscale x 2 x i1> splat (i1 true), <vscale x 2 x i1> splat (i1 true), <vscale x 2 x double> %va1, <vscale x 2 x double> %vb0)
  call void @llvm.aarch64.sme.mopa.nxv2f64(i32 3, <vscale x 2 x i1> splat (i1 true), <vscale x 2 x i1> splat (i1 true), <vscale x 2 x double> %va1, <vscale x 2 x double> %vb1)
  %kkn = add nuw nsw i64 %kk, 1
  %kdone = icmp eq i64 %kkn, %kce
  br i1 %kdone, label %tstore, label %kloop

tstore:
""")
    for (n, (tile, sl, col, row)) in enumerate(slots)
        println(io, "  %so$n = mul nsw i64 %ldc, $col")
        println(io, "  %sp$n = getelementptr inbounds double, ptr %ci, i64 %so$n")
        println(io, "  %sq$n = getelementptr inbounds double, ptr %sp$n, i64 $row")
        println(io, "  call void @llvm.aarch64.sme.st1d.vert(<vscale x 2 x i1> splat (i1 true), ptr %sq$n, i32 $tile, i32 $sl)")
    end
    print(io, """
  br label %ipend

ipend:
  %ipn = add nuw nsw i64 %ip, 1
  %ipdone = icmp eq i64 %ipn, %nip
  br i1 %ipdone, label %jpend, label %iploop

jpend:
  %jpn = add nuw nsw i64 %jp, 1
  %jpdone = icmp eq i64 %jpn, %njp
  br i1 %jpdone, label %done, label %jploop

done:
  call void @llvm.aarch64.sme.za.disable()
  ret void
}
""")
    print(io, _SME_ATTRS)
    return String(take!(io))
end

const _SME_MACRO_ACC = _sme_macro_ir(false)
const _SME_MACRO_OVER = _sme_macro_ir(true)

@inline function _sme_macro!(
        C::Ptr{Float64}, ldc::Int, Ap::Ptr{Float64}, Bp::Ptr{Float64},
        nip::Int, njp::Int, kce::Int, overwrite::Bool
    )
    if overwrite
        Base.llvmcall(
            (_SME_MACRO_OVER, "entry"), Cvoid,
            Tuple{Ptr{Float64}, Int64, Ptr{Float64}, Ptr{Float64}, Int64, Int64, Int64},
            C, Int64(ldc), Ap, Bp, Int64(nip), Int64(njp), Int64(kce)
        )
    else
        Base.llvmcall(
            (_SME_MACRO_ACC, "entry"), Cvoid,
            Tuple{Ptr{Float64}, Int64, Ptr{Float64}, Ptr{Float64}, Int64, Int64, Int64},
            C, Int64(ldc), Ap, Bp, Int64(nip), Int64(njp), Int64(kce)
        )
    end
    return nothing
end

# ── B packing by ZA transpose ──────────────────────────────────────────────────────────────────
# The kernel wants _SME_NR contiguous B values per k-step, i.e. a ROW of a column-major B. A ZA
# tile supplies that directly: load L columns as VERTICAL slices, store L rows as HORIZONTAL
# slices, and the tile has transposed an LxL block with no scalar access at all. Measured 80 GB/s
# against 33 for an explicit-vector transpose and 25 for the scalar one.
#
# Pointers are pre-offset by the caller: `b` at B[pc, jc], `bp` at the panel base.
#   src(cb, pb, jj) = b  + (pb*L + (cb*L + jj)*ldb)
#   dst(cb, pb, i)  = bp + ((cb>>1)*kce*NR + (pb*L + i)*NR + (cb&1)*L)
function _sme_packb_ir()
    L = _SME_L
    io = IOBuffer()
    print(io, _SME_DECLS)
    print(io, """
define void @entry(ptr %bp, ptr %b, i64 %ldb, i64 %kce, i64 %ncb) {
  call void @packb(ptr %bp, ptr %b, i64 %ldb, i64 %kce, i64 %ncb)
  ret void
}

define internal void @packb(ptr %bp, ptr %b, i64 %ldb, i64 %kce, i64 %ncb) #0 {
entry:
  call void @llvm.aarch64.sme.za.enable()
  %kb = sdiv i64 %kce, $L
  %anycb = icmp sgt i64 %ncb, 0
  %anykb = icmp sgt i64 %kb, 0
  %go = and i1 %anycb, %anykb
  br i1 %go, label %cbloop, label %done

cbloop:
  %cb = phi i64 [ 0, %entry ], [ %cbnext, %cbend ]
  %jp = lshr i64 %cb, 1
  %jblk = and i64 %cb, 1
  %jpoff = mul nsw i64 %jp, %kce
  %jpoffn = mul nsw i64 %jpoff, $(_SME_NR)
  %jbo = mul nsw i64 %jblk, $L
  %dstbase = add nsw i64 %jpoffn, %jbo
  %colbase = mul nsw i64 %cb, $L
  br label %pbloop

pbloop:
  %pb = phi i64 [ 0, %cbloop ], [ %pbnext, %pbloop ]
  %p0 = mul nsw i64 %pb, $L
""")
    for jj in 0:(L - 1)
        println(io, "  %c$jj = add nsw i64 %colbase, $jj")
        println(io, "  %cs$jj = mul nsw i64 %c$jj, %ldb")
        println(io, "  %so$jj = add nsw i64 %p0, %cs$jj")
        println(io, "  %sp$jj = getelementptr inbounds double, ptr %b, i64 %so$jj")
        println(io, "  call void @llvm.aarch64.sme.ld1d.vert(<vscale x 2 x i1> splat (i1 true), ptr %sp$jj, i32 0, i32 $jj)")
    end
    for i in 0:(L - 1)
        println(io, "  %r$i = add nsw i64 %p0, $i")
        println(io, "  %rn$i = mul nsw i64 %r$i, $(_SME_NR)")
        println(io, "  %do$i = add nsw i64 %dstbase, %rn$i")
        println(io, "  %dp$i = getelementptr inbounds double, ptr %bp, i64 %do$i")
        println(io, "  call void @llvm.aarch64.sme.st1d.horiz(<vscale x 2 x i1> splat (i1 true), ptr %dp$i, i32 0, i32 $i)")
    end
    print(io, """
  %pbnext = add nuw nsw i64 %pb, 1
  %pbdone = icmp eq i64 %pbnext, %kb
  br i1 %pbdone, label %cbend, label %pbloop

cbend:
  %cbnext = add nuw nsw i64 %cb, 1
  %cbdone = icmp eq i64 %cbnext, %ncb
  br i1 %cbdone, label %done, label %cbloop

done:
  call void @llvm.aarch64.sme.za.disable()
  ret void
}
""")
    print(io, _SME_ATTRS)
    return String(take!(io))
end

const _SME_PACKB = _sme_packb_ir()

@inline function _sme_packb!(bp::Ptr{Float64}, b::Ptr{Float64}, ldb::Int, kce::Int, ncb::Int)
    Base.llvmcall(
        (_SME_PACKB, "entry"), Cvoid,
        Tuple{Ptr{Float64}, Ptr{Float64}, Int64, Int64, Int64},
        bp, b, Int64(ldb), Int64(kce), Int64(ncb)
    )
    return nothing
end

# ── A packing ──────────────────────────────────────────────────────────────────────────────────
# A panel: _SME_MR rows per k-step, k-major, so each k-step of a panel is _SME_MR CONTIGUOUS
# elements of one A column and the copy is a straight vector move. alpha is folded in here, as in
# the SIMD path, so no separate scaling pass touches C.
#
# Panel-outer, NOT k-outer: k-outer reads a column contiguously but scatters its writes across
# panels, and that measured 5x SLOWER (24 vs 120 GB/s) — the scattered stores cost far more than
# the strided loads save. Do not re-try it.
#
# Reading A in place instead of packing (giving the kernel an `lda` stride) was also measured and
# is much WORSE — 0.24x at n=4096 — because the panel is re-read once per column panel and the
# packed copy is what keeps it cache- and TLB-resident.
function _sme_pack_A!(
        Ap::Ptr{Float64}, A::Ptr{Float64}, lda::Int,
        ic::Int, pc::Int, mce::Int, kce::Int, kpad::Int, alpha::Float64
    )
    MR = _SME_MR
    np = cld(mce, MR)
    one_alpha = alpha == 1.0
    let pa = Ap
        @inbounds for ip in 0:(np - 1)
            base = ip * kpad * MR
            i0 = ip * MR
            full_rows = (i0 + MR) <= mce
            for p in 0:(kpad - 1)
                d = pa + (base + p * MR) * 8
                if p >= kce
                    for i in 0:(MR - 1)
                        unsafe_store!(d + i * 8, 0.0)
                    end
                elseif full_rows && one_alpha
                    s = A + (ic + i0 + (pc + p) * lda) * 8
                    for i in 0:(MR - 1)
                        unsafe_store!(d + i * 8, unsafe_load(s + i * 8))
                    end
                else
                    s = A + (ic + i0 + (pc + p) * lda) * 8
                    for i in 0:(MR - 1)
                        v = (i0 + i) < mce ? alpha * unsafe_load(s + i * 8) : 0.0
                        unsafe_store!(d + i * 8, v)
                    end
                end
            end
        end
    end
    return nothing
end

# Scalar B pack with zero fill, for blocks whose edges the ZA transpose cannot read: it works in
# whole LxL blocks and would run off the end of B on a ragged column or depth remainder.
function _sme_pack_B_edge!(
        Bp::Ptr{Float64}, B::Ptr{Float64}, ldb::Int,
        pc::Int, jc::Int, kce::Int, nce::Int, kpad::Int
    )
    NR = _SME_NR
    npn = cld(nce, NR)
    let pb = Bp
        @inbounds for jp in 0:(npn - 1)
            base = jp * kpad * NR
            j0 = jp * NR
            for p in 0:(kpad - 1)
                d = pb + (base + p * NR) * 8
                for j in 0:(NR - 1)
                    v = (p < kce && (j0 + j) < nce) ?
                        unsafe_load(B + ((pc + p) + (jc + j0 + j) * ldb) * 8) : 0.0
                    unsafe_store!(d + j * 8, v)
                end
            end
        end
    end
    return nothing
end

# Scales a packed panel in place. Needed only when alpha must ride a panel built by the ZA
# transpose, which has no scaling form; the panel is cache-resident at this point.
function _sme_scale_panel!(P::Ptr{Float64}, len::Int, s::Float64)
    for i in 0:(len - 1)
        q = P + i * 8
        unsafe_store!(q, s * unsafe_load(q))
    end
    return nothing
end

# ── Block sizes ────────────────────────────────────────────────────────────────────────────────
# Derived, per req#8, but from a criterion the SIMD path does not have: C RE-STREAMING.
#
# The BLIS `jc -> pc -> ic` order reads and writes C once per `pc` block, i.e. cld(k, KC) times.
# On a SIMD microkernel that term is noise against compute. SME multiplies compute by ~8x while
# leaving memory untouched, so it inverts and becomes dominant — measured at n=2048, grouping a
# block-size sweep by C passes: 8 passes 399, 4 passes 450, 2 passes 477, 1 pass 509 GFLOP/s.
#
# So KC is made as large as a packed-panel memory budget allows (minimizing C passes) rather than
# sized for L1 residency, and NC spans the whole of n so A is packed exactly once. MC then sets
# the A-panel working set against L2.
# PDM: Measured — a packing-memory ceiling, not a residency criterion: KC is grown to minimize C passes, and where a larger panel stops paying depends on packing throughput against kernel throughput. | tune: sweep
const _SME_PANEL_BUDGET = @load_preference("sme_panel_bytes", 64 * 1024 * 1024)::Int

@inline function _sme_blocks(m::Int, n::Int, k::Int)
    NC = max(_SME_NR, n)
    # Depth block: as deep as the budget allows, so C is re-streamed as few times as possible.
    kb = max(_SME_L, (_SME_PANEL_BUDGET ÷ sizeof(Float64)) ÷ max(NC, 1))
    KC = min(k, kb)
    KC -= KC % _SME_L
    KC = max(KC, _SME_L)
    # Row block: the A panel (MC x KC) is the operand re-read per column panel, so hold it to a
    # share of L2 the way the SIMD path holds its own A block.
    mb = max(_SME_MR, ((_L2_BYTES * 3) ÷ 10) ÷ (KC * sizeof(Float64)))
    MC = min(m, mb)
    MC -= MC % _SME_MR
    MC = max(MC, _SME_MR)
    return MC, NC, KC
end

# ── Driver ─────────────────────────────────────────────────────────────────────────────────────
# C = beta*C + alpha*op(A)*op(B), column-major, Float64, on the SME path. alpha rides in the A
# pack; beta is applied by scaling C once up front, after which every depth block accumulates.
#
# Ragged edges: the packed panels are zero-filled to whole MRxNR tiles, so the macrokernel always
# runs full tiles. A tile that would write outside C is computed into a contiguous scratch tile
# and the live part copied back — the extra cost falls only on the edges.
function _sme_gemm!(
        C::Ptr{Float64}, ldc::Int, A::Ptr{Float64}, lda::Int, B::Ptr{Float64}, ldb::Int,
        m::Int, n::Int, k::Int, alpha::Float64, beta::Float64, tA::Bool, tB::Bool,
        Ap::Ptr{Float64}, Bp::Ptr{Float64}, Cs::Ptr{Float64},
        MC::Int, NC::Int, KC::Int
    )
    MR = _SME_MR
    NR = _SME_NR
    # beta == 0 is handled by ZEROING ZA on the first depth block instead of zeroing C here.
    # The whole point of this path is that C traffic dominates once compute is ~8x faster, so an
    # extra full pass over C costs real throughput -- 134 MB of pointless writes at n = 4096.
    # Any other beta needs one scaling pass, after which every block accumulates.
    if beta != 0.0 && beta != 1.0
        @inbounds for j in 0:(n - 1), i in 0:(m - 1)
            p = C + (i + j * ldc) * 8
            unsafe_store!(p, beta * unsafe_load(p))
        end
    end
    # alpha folds into whichever panel is built by the scaling (contiguous) packer. The ZA
    # transpose cannot scale, so when A is transposed alpha rides B -- and B is packed once per
    # (jc, pc) rather than once per (jc, pc, ic), which is the cheaper side anyway.
    bscale = tA ? alpha : 1.0
    if m == 0 || n == 0 || k == 0 || alpha == 0.0
        # No product to add. C still needs beta applied, including the beta == 0 case that the
        # block loop would otherwise have handled.
        if beta == 0.0
            @inbounds for j in 0:(n - 1), i in 0:(m - 1)
                unsafe_store!(C + (i + j * ldc) * 8, 0.0)
            end
        end
        return nothing
    end
    overwrite_first = beta == 0.0

    jc = 0
    while jc < n
        nce = min(NC, n - jc)
        npad = cld(nce, NR) * NR
        pc = 0
        while pc < k
            kce = min(KC, k - pc)
            kpad = cld(kce, _SME_L) * _SME_L
            # Which side needs transposing follows from the storage, not from preference:
            #   op(A)=A  -- A is m x k, so MR rows of a column are already contiguous
            #   op(A)=A2 -- A is k x m, so that run is strided and needs a transpose
            #   op(B)=B  -- B is k x n, so the run is strided and needs a transpose
            #   op(B)=B2 -- B is n x k, so the run is contiguous
            # So N,T needs no transpose at all and T,N needs two. The ZA transpose works in whole
            # LxL blocks and would read past the operand on a ragged edge, hence the scalar
            # fallback there.
            if tB
                _sme_pack_A!(Bp, B, ldb, jc, pc, nce, kce, kpad, bscale)
            elseif kce == kpad && nce == npad
                _sme_packb!(Bp, B + (pc + jc * ldb) * 8, ldb, kpad, npad ÷ _SME_L)
                bscale == 1.0 || _sme_scale_panel!(Bp, npad * kpad, bscale)
            else
                _sme_pack_B_edge!(Bp, B, ldb, pc, jc, kce, nce, kpad)
                bscale == 1.0 || _sme_scale_panel!(Bp, npad * kpad, bscale)
            end
            ic = 0
            while ic < m
                mce = min(MC, m - ic)
                mpad = cld(mce, MR) * MR
                if !tA
                    _sme_pack_A!(Ap, A, lda, ic, pc, mce, kce, kpad, alpha)
                elseif kce == kpad && mce == mpad
                    _sme_packb!(Ap, A + (pc + ic * lda) * 8, lda, kpad, mpad ÷ _SME_L)
                else
                    _sme_pack_B_edge!(Ap, A, lda, pc, ic, kce, mce, kpad)
                end
                _sme_macro_edges!(
                    C, ldc, Ap, Bp, Cs, ic, jc, mce, nce, mpad, npad, kpad,
                    overwrite_first && pc == 0
                )
                ic += MC
            end
            pc += KC
        end
        jc += NC
    end
    return nothing
end

# Runs the macrokernel over a packed block. Whole tiles that fit inside C go straight in; a tile
# that would overhang goes through a contiguous MRxNR scratch tile and is copied back live-part
# only.
function _sme_macro_edges!(
        C::Ptr{Float64}, ldc::Int, Ap::Ptr{Float64}, Bp::Ptr{Float64},
        Cs::Ptr{Float64}, ic::Int, jc::Int, mce::Int, nce::Int,
        mpad::Int, npad::Int, kpad::Int, over::Bool
    )
    MR = _SME_MR
    NR = _SME_NR
    nip = mpad ÷ MR
    njp = npad ÷ NR
    full_i = (mce % MR == 0)
    full_j = (nce % NR == 0)
    let pa = Ap, pb = Bp, pcs = Cs
        if full_i && full_j
            _sme_macro!(C + (ic + jc * ldc) * 8, ldc, pa, pb, nip, njp, kpad, over)
            return nothing
        end
        # Interior tiles in one call, then the ragged last row/column tile by tile.
        nip_full = mce ÷ MR
        njp_full = nce ÷ NR
        if nip_full > 0 && njp_full > 0
            _sme_macro!(C + (ic + jc * ldc) * 8, ldc, pa, pb, nip_full, njp_full, kpad, over)
        end
        for jp in 0:(njp - 1), ip in 0:(nip - 1)
            (ip < nip_full && jp < njp_full) && continue
            rows = min(MR, mce - ip * MR)
            cols = min(NR, nce - jp * NR)
            (rows <= 0 || cols <= 0) && continue
            api = pa + ip * kpad * MR * 8
            bpj = pb + jp * kpad * NR * 8
            # Scratch tile is contiguous with leading dimension MR; seed it with the live C so
            # the accumulate is exact, and zero the dead part.
            for j in 0:(NR - 1), i in 0:(MR - 1)
                v = (!over && i < rows && j < cols) ?
                    unsafe_load(C + ((ic + ip * MR + i) + (jc + jp * NR + j) * ldc) * 8) : 0.0
                unsafe_store!(pcs + (i + j * MR) * 8, v)
            end
            _sme_macro!(pcs, MR, api, bpj, 1, 1, kpad, false)
            for j in 0:(cols - 1), i in 0:(rows - 1)
                unsafe_store!(
                    C + ((ic + ip * MR + i) + (jc + jp * NR + j) * ldc) * 8,
                    unsafe_load(pcs + (i + j * MR) * 8)
                )
            end
        end
    end
    return nothing
end

# Scratch sizes for a given problem, so callers can size workspace without replaying the loop.
@inline function _sme_scratch_sizes(m::Int, n::Int, k::Int)
    MC, NC, KC = _sme_blocks(m, n, k)
    kpad = cld(min(KC, k), _SME_L) * _SME_L
    apad = cld(min(MC, m), _SME_MR) * _SME_MR
    bpad = cld(min(NC, n), _SME_NR) * _SME_NR
    return (apad * kpad, bpad * kpad, _SME_MR * _SME_NR, MC, NC, KC)
end

# ── Entry point from `_gemm_core!` ─────────────────────────────────────────────────────────────
# Float64, op(A)=A, op(B)=B, unit row stride on all three. Scratch comes from the shared Level-3
# workspace: the MRxNR edge tile is carved off the tail of the A slot so no new workspace field is
# needed (that struct's constructor is a long positional list and adding to it has shipped a bug).
#
# Below `_SME_MIN` the problem is too small for the packed panels to pay for themselves and the
# existing SIMD routes are better. It is a MEASURED crossover, not a residency formula: it depends on
# packing throughput against kernel throughput, neither predictable from a cache size.
#
# THE CROSSOVER IS NOT A THRESHOLD IN n, IT IS GOVERNED BY TILE OCCUPANCY, so there are two cuts
# below and a predicate that uses both. Square Float64, SME against the SIMD path it displaces,
# measured at EVERY n in one process on one set of operands (`bench/probes/sme_min_crossover.jl`):
#
#     rem = n % MR       n=47   48   49   63   64   65   79   80   81   95   96   97
#     ratio             0.62 2.57 0.44 0.96 3.39 0.67 1.40 4.57 0.96 1.83 5.51 1.24
#
# An exact multiple wins from 3*MR up and keeps winning. Everything else packs a remainder row-panel
# and runs it at a fraction of tile occupancy, and the worst remainder — 1 — still LOSES at n=81.
# So a single cut cannot be both safe and small: 4*MR admits n=65 at 0.67 and n=81 at 0.96, and
# 3*MR admits n=49 at 0.44. Only at 6*MR does every remainder finally pay.
#
# Sampling every OTHER n hides this: a 2-step sweep reads 48, 56, 64 and concludes 3*MR is safe,
# which regressed gemm@50 by 38% (1142 -> 1582 us) before the dense sweep found 49, 51, 53.
# PDM: Measured — tile-occupancy crossover, not a residency formula: the general cut is where the worst remainder (1) starts paying, and exact multiples of MR are admitted earlier because they pack no remainder panel at all. | tune: sweep
const _SME_MIN = @load_preference("sme_min", 6 * _SME_MR)::Int
# Exact multiples of the row-panel carry no remainder and win from here up; see the table above.
const _SME_MIN_EXACT = @load_preference("sme_min_exact", 3 * _SME_MR)::Int

# THE KERNEL MUST NOT ENTER THE PACKAGE IMAGE, and `@noinline` alone does not achieve that.
#
# A function holding ZA state needs the streaming vector length to size its stack frame, which
# lowers to `rdsvl`. A package image is generated for a GENERIC CPU, where that instruction cannot
# be selected at all (`LLVM ERROR: Cannot select: AArch64ISD::RDSVL`), so this code can only ever
# be compiled for the host. Precompilation reaches it by INFERENCE, not by execution -- the
# workload multiplies 8x8 matrices and never takes the branch -- and inference walks straight
# through a call boundary, so `@noinline` does not stop it.
#
# The barrier is a function pointer resolved at load time: inference sees an opaque `Ptr`, the
# trampoline (and hence the kernel) is compiled on the host, and the `ccall` through it has
# concrete argument types so nothing is boxed and nothing is allocated.
const _SME_TRAMPOLINE = Ref{Any}(nothing)      # roots the closure against collection
const _SME_ENTRY = Ref{Ptr{Cvoid}}(C_NULL)

function _sme_entry_cabi(
        C::Ptr{Float64}, ldc::Int, A::Ptr{Float64}, lda::Int, B::Ptr{Float64}, ldb::Int,
        m::Int, n::Int, k::Int, alpha::Float64, beta::Float64, tA::Int, tB::Int,
        Ap::Ptr{Float64}, Bp::Ptr{Float64}, Cs::Ptr{Float64}, MC::Int, NC::Int, KC::Int
    )
    _sme_gemm!(C, ldc, A, lda, B, ldb, m, n, k, alpha, beta, tA != 0, tB != 0,
               Ap, Bp, Cs, MC, NC, KC)
    return nothing
end

# Called from `__init__`. The target is fetched by NAME at run time, so inference sees only `Any`
# and never reaches the streaming kernel.
function _sme_init!()
    _SME_F64 || return nothing
    # `__init__` also runs inside the PRECOMPILE process, whose codegen targets the generic image
    # CPU. Building the trampoline there compiles the kernel into the image and fails on `rdsvl`.
    ccall(:jl_generating_output, Cint, ()) == 0 || return nothing
    try
        # `getfield(@__MODULE__, :name)` is NOT opaque -- module and symbol are both constants, so
        # inference folds it back to the concrete function and walks into the kernel anyway.
        # `inferencebarrier` forces the value to `Any`, which is what actually stops it.
        f = Base.inferencebarrier(_sme_entry_cabi)
        cf = @cfunction($f, Cvoid,
            (Ptr{Float64}, Int, Ptr{Float64}, Int, Ptr{Float64}, Int,
             Int, Int, Int, Float64, Float64, Int, Int,
             Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Int, Int, Int))
        _SME_TRAMPOLINE[] = cf
        _SME_ENTRY[] = Base.unsafe_convert(Ptr{Cvoid}, cf)
    catch
        # A machine that advertises the feature but cannot build the kernel keeps the SIMD path.
        _SME_ENTRY[] = C_NULL
    end
    return nothing
end

# BOTH C DIMENSIONS MUST CARRY AT LEAST TWO TILES, and `k` is not part of that test. The kernel
# computes an `_SME_MR` x `_SME_NR` output tile; when C is narrower than a tile, most of ZA is
# written and thrown away, and no amount of `k` recovers it. Measured against the SIMD path it
# displaces (`bench/probes/sme_min_crossover.jl`), same operands, same process:
#
#     (m,n,k)        SME/SIMD        (m,n,k)        SME/SIMD
#     (64,64,8)        2.97          (8,8,64)         0.14
#     (32,32,64)       1.68          (64,8,8)         0.14
#     (64,64,64)       3.41          (2048,8,8)       0.34
#
# A thin `k` is fine — (64,64,8) is the fastest relative win in that table — so the criterion is
# `min(m, n)`, not `min(m, n, k)`, and testing `max(m, n, k)` alone admits exactly the shapes whose
# only large dimension is the one that does not help.
#
# THE SIZE CUT HAS TWO ARMS because the cost is tile occupancy, not size (see `_SME_MIN`): a shape
# that divides the panel exactly pays from `_SME_MIN_EXACT`, everything else has to reach `_SME_MIN`
# before its remainder panel is amortized.
@inline function _sme_tile_ok(m, n)
    min(m, n) >= 2 * _SME_MR || return false
    (m % _SME_MR == 0 && n % _SME_NR == 0) ?
        max(m, n) >= _SME_MIN_EXACT : max(m, n) >= _SME_MIN
end

@inline _sme_eligible(::Type{T}, m, n, k, tA, tB, cA, cB, C, A, B) where {T} =
    T === Float64 && _SME_F64 && !cA && !cB &&
        _strided1(C) && _strided1(A) && _strided1(B) &&
        _sme_tile_ok(m, n) && _SME_ENTRY[] !== C_NULL

# AN SME-ELIGIBLE CALL IS NOT COLUMN-SPLIT, for the same reason a Strassen call is not: splitting it
# does not divide the work between units, it queues the workers behind one of them. The coprocessor
# is SHARED BY THE CLUSTER, so `nw` chunks that each route back into `_gemm_core_body!` all contend
# for the same hardware. Measured on an M6 at n=4096, Float64: 503 GFLOP/s with this guard against
# 301 without it (six workers, one unit) and 177 for the threaded NEON path that the guard declines.
#
# The threaded entry is the ONLY place this can be decided. A guard inside `_gemm_core_body!` is too
# late — `_gemm_threaded!` reaches that body once per chunk through `_gemm_run_chunk`, so by then the
# split has already happened.
@inline _sme_owns(::Type{T}, m, n, k, tA, tB, cA, cB, C, A, B) where {T} =
    _sme_eligible(T, m, n, k, tA, tB, cA, cB, C, A, B)

# Bumped once per SME gemm, and read by the threading liveness gate: a call this guard takes away
# from the pool has to show that the coprocessor ran instead. Once per call against milliseconds of
# kernel, so it costs nothing measurable; it is the SME counterpart of the pool's `gen` word.
const _SME_CALLS = Threads.Atomic{Int}(0)

@noinline function _gemm_sme!(C, A, B, alpha::Float64, beta::Float64, m::Int, n::Int, k::Int,
                             tA::Bool, tB::Bool)
    Threads.atomic_add!(_SME_CALLS, 1)
    asz, bsz, csz, MC, NC, KC = _sme_scratch_sizes(m, n, k)
    # The pack buffers are per-TASK (`_gemm_scratch` -> `_gpackws`), not per-thread: cooperative
    # A-packing meets the other workers at a barrier while the packed B panel is live. Going
    # through `_gemm_scratch` keeps both growths at the one growth point, so a single barrier
    # covers them. The C scratch is carved off the tail of the A buffer.
    Ap, Bp = _gemm_scratch(Float64, asz + csz, bsz)
    ldc = stride(C, 2); lda = stride(A, 2); ldb = stride(B, 2)
    GC.@preserve C A B Ap Bp begin
        pap = pointer(Ap)
        ccall(
            _SME_ENTRY[], Cvoid,
            (Ptr{Float64}, Int, Ptr{Float64}, Int, Ptr{Float64}, Int,
             Int, Int, Int, Float64, Float64, Int, Int,
             Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Int, Int, Int),
            pointer(C), ldc, pointer(A), lda, pointer(B), ldb,
            m, n, k, alpha, beta, tA ? 1 : 0, tB ? 1 : 0,
            pap, pointer(Bp), pap + asz * sizeof(Float64), MC, NC, KC
        )
    end
    return C
end
