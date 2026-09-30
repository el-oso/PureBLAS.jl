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
    # The four operand strides are RUNTIME arguments, which is what lets the kernel read an operand
    # IN PLACE instead of from a packed panel — a packed panel and a column-major operand differ in
    # nothing else (see `_sme_pack_A!`'s layout and `_sme_unpacked_a`). Making them dynamic costs
    # nothing: LLVM strength-reduces both loops to add chains either way, measured 0.994-1.018
    # against the constant-stride form at n=64..1024 with bit-identical output.
    args = "ptr %c, i64 %ldc, ptr %ap, ptr %bp, i64 %nip, i64 %njp, i64 %kce, " *
        "i64 %aip, i64 %aks, i64 %bjp, i64 %bks"
    print(io, """
define void @entry($args) {
  call void @macro($args)
  ret void
}

define internal void @macro($args) #0 {
entry:
  call void @llvm.aarch64.sme.za.enable()
  %ldcnr = mul nsw i64 %ldc, $(_SME_NR)
  %anyi = icmp sgt i64 %nip, 0
  %anyj = icmp sgt i64 %njp, 0
  %go = and i1 %anyi, %anyj
  br i1 %go, label %jploop, label %done

jploop:
  %jp = phi i64 [ 0, %entry ], [ %jpn, %jpend ]
  %bo = mul nsw i64 %jp, %bjp
  %bpj = getelementptr inbounds double, ptr %bp, i64 %bo
  %co = mul nsw i64 %jp, %ldcnr
  %cj = getelementptr inbounds double, ptr %c, i64 %co
  br label %iploop

iploop:
  %ip = phi i64 [ 0, %jploop ], [ %ipn, %ipend ]
  %ao = mul nsw i64 %ip, %aip
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
  %aoff = mul nsw i64 %kk, %aks
  %boff = mul nsw i64 %kk, %bks
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

# `aip`/`bjp` step one panel-row / panel-column block; `aks`/`bks` step one depth position. A PACKED
# panel and an operand read IN PLACE differ only in these four numbers — see `_sme_panel_strides`.
@inline function _sme_macro!(
        C::Ptr{Float64}, ldc::Int, Ap::Ptr{Float64}, Bp::Ptr{Float64},
        nip::Int, njp::Int, kce::Int, aip::Int, aks::Int, bjp::Int, bks::Int, overwrite::Bool
    )
    T = Tuple{
        Ptr{Float64}, Int64, Ptr{Float64}, Ptr{Float64}, Int64, Int64, Int64,
        Int64, Int64, Int64, Int64,
    }
    if overwrite
        Base.llvmcall(
            (_SME_MACRO_OVER, "entry"), Cvoid, T,
            C, Int64(ldc), Ap, Bp, Int64(nip), Int64(njp), Int64(kce),
            Int64(aip), Int64(aks), Int64(bjp), Int64(bks)
        )
    else
        Base.llvmcall(
            (_SME_MACRO_ACC, "entry"), Cvoid, T,
            C, Int64(ldc), Ap, Bp, Int64(nip), Int64(njp), Int64(kce),
            Int64(aip), Int64(aks), Int64(bjp), Int64(bks)
        )
    end
    return nothing
end

# Strides for a PACKED panel of `kpad` depth positions: a row block is `kpad*MR` apart, a depth
# position `MR` apart. The unpacked forms are in `_sme_panel_strides`.
@inline _sme_packed_a_strides(kpad::Int) = (kpad * _SME_MR, _SME_MR)
@inline _sme_packed_b_strides(kpad::Int) = (kpad * _SME_NR, _SME_NR)

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

# ── Reading A in place instead of packing it ────────────────────────────────────────────────────
# A packed A panel holds `Ap[ip*kpad*MR + p*MR + i] = A[ic + ip*MR + i, pc + p]`, so for op(A) = A
# the pack is a PURE STRIDED COPY: the kernel can read A itself by taking `aks = lda` for its depth
# step and `aip = MR` for its row-block step (`_sme_macro!`). A whole pass over A disappears — 17%
# of the gemm at n=128, 28% counting both operands.
#
# The two routes are BITWISE EQUAL, and that is load-bearing, not incidental: the kernel sees the
# same values in the same order either way, so which route a block takes cannot perturb a result.
# Threading splits columns, so `mce` — and hence the choice — is the same for every worker anyway.
#
# It is not free, and the cost grows with the block. A packed panel is one contiguous stream the
# hardware prefetcher covers; read in place, each depth step touches MR doubles — two whole cache
# lines at MR=16, so no bytes are wasted — but the next step is `lda` away, which puts consecutive
# steps on few L1 sets when `lda` is unlucky.
#
# Three preconditions, all structural:
#   - op(A) = A. When A is transposed its MR-run is strided, not contiguous, and a copy is the only
#     way to give the kernel what it loads.
#   - alpha == 1. Scaling happens inside the contiguous packer; with no copy there is nowhere for it
#     to ride. Routing it onto B instead would add a pass over the B panel, which is the cost this
#     is removing.
#   - The block is a whole number of tiles in m. An in-place read has no zero-filled remainder, so a
#     padded ROW would read past the operand. Depth needs no such condition: the kernel's depth
#     COUNT (`kce`) and a panel's depth STRIDE (`kpad`) are separate arguments, so a ragged k runs
#     exactly `kce` steps and the padding is simply never reached.
#
# THE CUT IS SIZED FOR THE WORST `lda`, because the caller's is not ours to choose. Square Float64
# through `gemm!`, A a view of a power-of-two-wide parent so the stride always aliases, packed
# against in place (`bench/probes/sme_inplace_worstcase.jl`, _EXP17 A/B in one process):
#
#     A block / L1  0.25   0.56   1.00   1.56   2.25   3.06   4.00   5.06   6.25   9.00  16.00
#     packed/inplc  1.939  1.599  1.418  1.111  1.102  1.053  1.057  0.968  0.982  0.959  0.949
#
# so in place pays up to four L1-fuls and loses beyond five. With a FRIENDLY stride (the same sizes
# at lda+1) it wins everywhere measured, 1.14-1.91, and `sme_inplace_alias.jl` isolates that to the
# stride alone: at n=512 the ratio is 0.967 at lda=512 and 1.143 at lda=513, nothing else changed.
#
# THE CUT KEEPS MARGIN, AND THAT IS NOT CAUTION — IT IS A MEASURED REQUIREMENT. The table above is
# gemm walking its own operand; a caller that drives more traffic through the same cache tips the
# same block negative. `symm` at n=256 measures 0.936 under a four-L1-ful cut and 1.006 under a
# two-L1-ful one (`bench/probes/sme_inplace_regress.jl`, three independent probes agreeing), which is
# exactly the 1.05-1.06 edge of the table giving way. Two L1-fuls costs gemm@256 5.7% and syrk@512
# 10.2% — neither is a binding cell, and neither gate moves — and buys no regression anywhere
# measured, which is the better resting state for a route every BLAS-3 routine reaches.
#
# WHY THE CUT IS RESIDENCY AND NOT AN ALIASING PREDICATE. `_alias_ld` exists for this class of
# problem, but its period is `_L1_WAY_D` = L1 ÷ associativity, and associativity comes from a CPUID
# leaf that does not exist on aarch64 — `_L1D_ASSOC` is the fallback 8 here, not a detected value. So
# `_alias_ld` is false for every stride that actually loses on this machine (lda = 256, 512, 768,
# 1024), and a set-conflict criterion cannot be founded on a number we are guessing. A residency cut
# holds for any stride; widening it to admit friendly strides is left open, and the 7-20% it gives up
# at blocks of 5-30 L1-fuls is recorded above rather than claimed.
# req8-ok: a coefficient over the DETECTED L1 with the falsifying tables above, not a fitted magic
# number — the criterion is A-block residency, and the coefficient sits one step inside the measured
# flip so a caller with its own cache traffic does not land on the edge.
# PDM: Derived — A-block residency against the detected L1: an in-place walk stays as cheap as a contiguous stream while the block the kernel re-reads is a couple of L1-fuls, and the coefficient is set one step inside where that was measured to flip under a worst-case stride, because a caller with extra cache traffic tips the edge. | tune: n/a, follows _L1_BYTES
const _SME_INPLACE_MAX =
    @load_preference("sme_inplace_max", 2 * (_L1_BYTES ÷ sizeof(Float64)))::Int

@inline function _sme_inplace_cap()
    ov = @inbounds _EXPINT[4]           # sweep override; 0 = the shipped cut
    return ov > 0 ? ov : _SME_INPLACE_MAX
end

@inline function _sme_inplace_a(mce::Int, kce::Int, alpha::Float64, tA::Bool)
    (!tA && alpha == 1.0 && !(@inbounds _EXPFLAG[_EXP17])) || return false
    mce % _SME_MR == 0 || return false
    return mce * kce <= _sme_inplace_cap()
end

# The same trick on the other operand, and the transpose condition is the MIRROR of A's: the kernel
# wants NR contiguous values of B per depth step, which is a row of column-major B — strided when B
# is k x n and contiguous when B is n x k. So in place needs `op(B) = B'` where A needed `op(A) = A`,
# and it is the N,T form that gets both (`_syrk_gemm!` at trans='N', `_gemm_accR!` at transA='T').
# `bscale`, not `alpha`: which panel carries the scaling is decided by `tA` in the driver, and the
# condition is that this panel carries none.
@inline function _sme_inplace_b(nce::Int, kce::Int, bscale::Float64, tB::Bool)
    (tB && bscale == 1.0 && !(@inbounds _EXPFLAG[_EXP18])) || return false
    nce % _SME_NR == 0 || return false
    return nce * kce <= _sme_inplace_cap()
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
    # ROUND ONLY A KC THAT IS ACTUALLY A SPLIT. Rounding down to a whole `_SME_L` keeps a depth
    # block's pack aligned, but applying it when `KC == k` turns any k not divisible by the lane
    # count into TWO blocks — 96 + 4 at k=100 — and the tail block pays a second pack, a second
    # full pass over C, and every ragged edge tile again. The pack already zero-fills `kce` up to
    # `kpad`, so a single block of the true k needs no rounding at all.
    #
    # Measured on an M6, driver KC against KC = k, output bitwise identical:
    #   n,k = 100,100 -> 1.31x   128,100 -> 1.20x   256,250 -> 1.10x
    #         500,500 -> 1.07x  2100,2100 -> 1.04x
    # It costs every SME caller at every `k % _SME_L != 0`, and n=100 is a gate cell.
    if KC < k
        KC -= KC % _SME_L
        KC = max(KC, _SME_L)
    end
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
            local pb::Ptr{Float64}, bjp::Int, bks::Int
            if _sme_inplace_b(nce, kce, bscale, tB)
                # B itself IS the panel — the op(B)=B2 row of the table above is the contiguous case,
                # so this is the same trick `_sme_inplace_a` plays on the other operand.
                pb = B + (jc + pc * ldb) * 8
                bjp = NR
                bks = ldb
            else
                if tB
                    _sme_pack_A!(Bp, B, ldb, jc, pc, nce, kce, kpad, bscale)
                elseif kce == kpad && nce == npad
                    _sme_packb!(Bp, B + (pc + jc * ldb) * 8, ldb, kpad, npad ÷ _SME_L)
                    bscale == 1.0 || _sme_scale_panel!(Bp, npad * kpad, bscale)
                else
                    _sme_pack_B_edge!(Bp, B, ldb, pc, jc, kce, nce, kpad)
                    bscale == 1.0 || _sme_scale_panel!(Bp, npad * kpad, bscale)
                end
                pb = Bp
                bjp = kpad * NR
                bks = NR
            end
            ic = 0
            while ic < m
                mce = min(MC, m - ic)
                mpad = cld(mce, MR) * MR
                local pa::Ptr{Float64}, aip::Int, aks::Int
                if _sme_inplace_a(mce, kce, alpha, tA)
                    # A itself IS the panel: the kernel walks it at `lda` and no copy happens.
                    pa = A + (ic + pc * lda) * 8
                    aip = MR
                    aks = lda
                else
                    if !tA
                        _sme_pack_A!(Ap, A, lda, ic, pc, mce, kce, kpad, alpha)
                    elseif kce == kpad && mce == mpad
                        _sme_packb!(Ap, A + (pc + ic * lda) * 8, lda, kpad, mpad ÷ _SME_L)
                    else
                        _sme_pack_B_edge!(Ap, A, lda, pc, ic, kce, mce, kpad)
                    end
                    pa = Ap
                    aip = kpad * MR
                    aks = MR
                end
                _sme_macro_edges!(
                    C, ldc, pa, pb, Cs, ic, jc, mce, nce, mpad, npad, kce, kpad,
                    aip, aks, bjp, bks, overwrite_first && pc == 0
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
#
# `Ap`/`Bp` may be a packed panel or the operand itself; `aip`/`aks`/`bjp`/`bks` say which. An
# operand read in place has no zero-filled remainder, so the caller only ever hands one to a block
# whose own dimension is a whole number of tiles — every path below that runs PADDED tiles
# (`mpad`>`mce`, `npad`>`nce`) is therefore reached only with packed panels on the padded side.
function _sme_macro_edges!(
        C::Ptr{Float64}, ldc::Int, Ap::Ptr{Float64}, Bp::Ptr{Float64},
        Cs::Ptr{Float64}, ic::Int, jc::Int, mce::Int, nce::Int,
        mpad::Int, npad::Int, kce::Int, kpad::Int, aip::Int, aks::Int, bjp::Int, bks::Int, over::Bool
    )
    MR = _SME_MR
    NR = _SME_NR
    nip = mpad ÷ MR
    njp = npad ÷ NR
    full_i = (mce % MR == 0)
    full_j = (nce % NR == 0)
    let pa = Ap, pb = Bp, pcs = Cs
        if full_i && full_j
            _sme_macro!(C + (ic + jc * ldc) * 8, ldc, pa, pb, nip, njp, kce,
                        aip, aks, bjp, bks, over)
            return nothing
        end
        # PAD THE BLOCK, NOT THE TILE, when it fits. The per-tile path below pays a macrokernel
        # prologue for every edge tile, and that prologue is most of what an edge tile costs:
        # measured at n=100, 0.26 us per tile of which 0.037 is the copy. Rounding the block up to
        # whole tiles makes it ONE call. Measured against the per-tile path, whole-gemm times:
        # n=50 6.64 -> ~2.4 us, n=100 14.7 -> ~7.5, n=132 26.4 -> ~16.8.
        if mpad * npad <= _sme_cpad_cap()
            for j in 0:(npad - 1), i in 0:(mpad - 1)
                v = (!over && i < mce && j < nce) ?
                    unsafe_load(C + ((ic + i) + (jc + j) * ldc) * 8) : 0.0
                unsafe_store!(pcs + (i + j * mpad) * 8, v)
            end
            _sme_macro!(pcs, mpad, pa, pb, nip, njp, kce, aip, aks, bjp, bks, false)
            for j in 0:(nce - 1), i in 0:(mce - 1)
                unsafe_store!(C + ((ic + i) + (jc + j) * ldc) * 8,
                              unsafe_load(pcs + (i + j * mpad) * 8))
            end
            return nothing
        end
        # Interior tiles in one call, then the ragged last row/column tile by tile.
        nip_full = mce ÷ MR
        njp_full = nce ÷ NR
        if nip_full > 0 && njp_full > 0
            _sme_macro!(C + (ic + jc * ldc) * 8, ldc, pa, pb, nip_full, njp_full, kce,
                        aip, aks, bjp, bks, over)
        end
        for jp in 0:(njp - 1), ip in 0:(nip - 1)
            (ip < nip_full && jp < njp_full) && continue
            rows = min(MR, mce - ip * MR)
            cols = min(NR, nce - jp * NR)
            (rows <= 0 || cols <= 0) && continue
            api = pa + ip * aip * 8
            bpj = pb + jp * bjp * 8
            # Scratch tile is contiguous with leading dimension MR; seed it with the live C so
            # the accumulate is exact, and zero the dead part.
            for j in 0:(NR - 1), i in 0:(MR - 1)
                v = (!over && i < rows && j < cols) ?
                    unsafe_load(C + ((ic + ip * MR + i) + (jc + jp * NR + j) * ldc) * 8) : 0.0
                unsafe_store!(pcs + (i + j * MR) * 8, v)
            end
            _sme_macro!(pcs, MR, api, bpj, 1, 1, kce, aip, aks, bjp, bks, false)
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
# How large a PADDED C BLOCK the edge path may materialise, in elements. A ragged block costs one
# macrokernel call per edge tile today, and that call's prologue dominates its work: measured at
# n=100, 13 edge tiles at 0.26 us each of which only 0.037 is the scratch copy. Rounding the whole
# block up to whole tiles turns those 13 calls into ONE, at the price of a copy in and out.
#
# Bounded by L1 because the padded block is written, read by the kernel, and read back immediately:
# beyond L1 the copy stops being cheap and the per-tile path is the better of the two. At 128 KiB
# that covers a square block to n = 128, which is exactly the range where the whole matrix is one
# ragged block and the per-call overhead is the entire cost.
@inline _sme_cpad_cap() = max(_SME_MR * _SME_NR, _L1_BYTES ÷ sizeof(Float64))

@inline function _sme_scratch_sizes(m::Int, n::Int, k::Int)
    MC, NC, KC = _sme_blocks(m, n, k)
    kpad = cld(min(KC, k), _SME_L) * _SME_L
    apad = cld(min(MC, m), _SME_MR) * _SME_MR
    bpad = cld(min(NC, n), _SME_NR) * _SME_NR
    csz = max(_SME_MR * _SME_NR, min(apad * bpad, _sme_cpad_cap()))
    return (apad * kpad, bpad * kpad, csz, MC, NC, KC)
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
# PDM: Measured — the same occupancy crossover for shapes that pack no remainder panel at all; one full row panel already pays, per the table above. | tune: sweep
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
# A KERNEL THAT BUILDS IS NOT A KERNEL THAT IS CORRECT, and nothing else here checks the difference.
# The guards above cover a machine that lacks the feature and a build that raises; neither covers a
# geometry assumption that is wrong for this streaming vector length, which produces WRONG NUMBERS
# and no symptom. (Nor can the `try` above catch an instruction-selection failure: `Cannot select`
# is `report_fatal_error`, a process abort, not a Julia exception.)
#
# So both kernels answer a known question before their pointers are published. The data is
# ASYMMETRIC and row/column distinguishable -- `i + 1000j` -- because a transposed or row-permuted
# geometry is invisible on symmetric input, which is exactly the bug class this is here to catch.
# Shapes are chosen to reach the edge paths: a ragged gemm exercises the edge macrokernel and the
# scalar B pack, and a gemv row count that is a sum of several ladder blocks plus a scrap exercises
# the block halving and the overlapping tail.
#
# A MISMATCH THROWS. It means the hardware model in this file is wrong for this machine, which is a
# bug to report rather than a condition to degrade around; falling back silently would leave a wrong
# model undiscovered. `sme_f64 = false` in LocalPreferences is the escape hatch, and the message
# names it.
function _sme_selftest()
    L = _SME_L
    ref(m, n, k, a, b, A, B, C) = begin
        R = copy(C)
        for j in 1:n, i in 1:m
            s = 0.0
            for p in 1:k
                s = muladd(A[i, p], B[p, j], s)
            end
            R[i, j] = a * s + b * R[i, j]
        end
        R
    end
    gen(m, n) = [i + 1000.0 * j for i in 1:m, j in 1:n]
    worst = 0.0
    for (m, n, k) in ((2 * _SME_MR, 2 * _SME_NR, L), (2 * _SME_MR + 1, 2 * _SME_NR + 3, L + 1))
        A = gen(m, k); B = gen(k, n)
        for a in (1.5,), b in (0.0, 0.5)
            C = gen(m, n)
            want = ref(m, n, k, a, b, A, B, C)
            got = copy(C)
            _gemm_sme!(got, A, B, a, b, m, n, k, false, false)
            worst = max(worst, maximum(abs, got .- want) / maximum(abs, want))
        end
    end
    # gemv: several ladder blocks plus a scrap below one group, and an accumulate form
    for (m, bet) in ((_SME_GEMV_BLK * _SME_GEMV_NGMAX + _SME_GEMV_BLK * 2 + 5, 0.0),
                     (_SME_GEMV_BLK * 3, 0.5))
        n = 7
        A = gen(m, n); x = [1.0 + 0.25 * j for j in 1:n]; y0 = gen(m, 1)[:, 1]
        want = copy(y0)
        for i in 1:m
            s = 0.0
            for j in 1:n
                s = muladd(A[i, j], x[j], s)
            end
            want[i] = 1.25 * s + bet * want[i]
        end
        got = copy(y0)
        _sme_gemv!(m, n, 1.25, A, x, bet, got)
        worst = max(worst, maximum(abs, got .- want) / maximum(abs, want))
    end
    # gemv-T: a row count that is NOT a multiple of the group height and a column count that is not
    # a multiple of NC, so both tails run, plus an accumulate form.
    for (m, n, bet) in ((_SME_GEMVT_MINM + 4 * _SME_L + 5, _SME_GEMVT_NC * 3 + 1, 0.0),
                        (_SME_GEMVT_MINM, _SME_GEMVT_NC * 2, 0.5))
        A = gen(m, n); x = [1.0 + 0.25 * i for i in 1:m]; y0 = gen(n, 1)[:, 1]
        want = copy(y0)
        for j in 1:n
            s = 0.0
            for i in 1:m
                s = muladd(A[i, j], x[i], s)
            end
            want[j] = 1.25 * s + bet * want[j]
        end
        got = copy(y0)
        _sme_gemvt!(m, n, 1.25, A, x, bet, got)
        worst = max(worst, maximum(abs, got .- want) / maximum(abs, want))
    end
    return worst
end


function _sme_init!()
    _SME_F64 || return nothing
    # `__init__` also runs inside the PRECOMPILE process, whose codegen targets the generic image
    # CPU. Building the trampoline there compiles the kernel into the image and fails on `rdsvl`.
    ccall(:jl_generating_output, Cint, ()) == 0 || return nothing
    # Both trampolines come from the builders at the end of this file, which carry the barrier or
    # drop it according to the compiling process's own target — see the block comment there.
    try
        cf = _sme_entry_cf()
        _SME_TRAMPOLINE[] = cf
        _SME_ENTRY[] = Base.unsafe_convert(Ptr{Cvoid}, cf)
    catch
        # A machine that advertises the feature but cannot build the kernel keeps the SIMD path.
        _SME_ENTRY[] = C_NULL
    end
    try
        gf = _sme_gemv_cf()
        _SME_GEMV_TRAMPOLINE[] = gf
        _SME_GEMV_ENTRY[] = Base.unsafe_convert(Ptr{Cvoid}, gf)
    catch
        _SME_GEMV_ENTRY[] = C_NULL
    end
    try
        tf = _sme_gemvt_cf()
        _SME_GEMVT_TRAMPOLINE[] = tf
        _SME_GEMVT_ENTRY[] = Base.unsafe_convert(Ptr{Cvoid}, tf)
    catch
        _SME_GEMVT_ENTRY[] = C_NULL
    end
    try
        df = _sme_dot_cf()
        _SME_DOT_TRAMPOLINE[] = df
        _SME_DOT_ENTRY[] = Base.unsafe_convert(Ptr{Cvoid}, df)
    catch
        _SME_DOT_ENTRY[] = C_NULL
    end
    try
        af = _sme_asum_cf()
        _SME_ASUM_TRAMPOLINE[] = af
        _SME_ASUM_ENTRY[] = Base.unsafe_convert(Ptr{Cvoid}, af)
    catch
        _SME_ASUM_ENTRY[] = C_NULL
    end
    try
        rf = _sme_ger_cf()
        _SME_GER_TRAMPOLINE[] = rf
        _SME_GER_ENTRY[] = Base.unsafe_convert(Ptr{Cvoid}, rf)
    catch
        _SME_GER_ENTRY[] = C_NULL
    end
    try
        xf = _sme_axpy_cf()
        _SME_AXPY_TRAMPOLINE[] = xf
        _SME_AXPY_ENTRY[] = Base.unsafe_convert(Ptr{Cvoid}, xf)
    catch
        _SME_AXPY_ENTRY[] = C_NULL
    end
    try
        sf = _sme_scal_cf()
        _SME_SCAL_TRAMPOLINE[] = sf
        _SME_SCAL_ENTRY[] = Base.unsafe_convert(Ptr{Cvoid}, sf)
    catch
        _SME_SCAL_ENTRY[] = C_NULL
    end
    # Both pointers are live now, so the kernels can be asked a question with a known answer.
    if _SME_ENTRY[] !== C_NULL && _SME_GEMV_ENTRY[] !== C_NULL && _SME_GEMVT_ENTRY[] !== C_NULL
        err = try
            _sme_selftest()
        catch e
            _SME_ENTRY[] = C_NULL; _SME_GEMV_ENTRY[] = C_NULL; _SME_GEMVT_ENTRY[] = C_NULL
            rethrow(e)
        end
        if !(err < 1e-12)
            _SME_ENTRY[] = C_NULL; _SME_GEMV_ENTRY[] = C_NULL; _SME_GEMVT_ENTRY[] = C_NULL
            # The message carries NO interpolated value and prints nothing. `__init__` is in the
            # trim graph, where `error("...$x...")` lowers to `print_to_string` over a
            # `Vararg{Any}` tuple and `--trim=safe` rejects it as an unresolved call; and IO here
            # would be a task-switch point in a kernel file, which `test/yield_lint.jl` guards.
            # The measured error is not lost — `_sme_selftest()` returns it, and the message says
            # so. An error PATH is where a dynamic `string(...)` is easiest to write and hardest to
            # notice: it never runs, so nothing but a linter ever objects.
            error(_SME_SELFTEST_MSG)
        end
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
# ONE FLOOR, NOT TWO. A ragged operand used to need a much higher floor than a tile-exact one,
# because its edge tiles each cost a separate macrokernel call — measured at 0.26 us apiece, of
# which only 0.037 was the scratch copy, the rest a prologue paid for ~0.05 us of work. Padding the
# whole block to whole tiles turns those calls into one (see `_sme_cpad_cap`), so the reason for the
# second, higher floor is gone with it.
#
# Measured on an M6 at the sizes the old ragged floor rejected, forced SME against the path taken
# today: n=40 0.81x (loses), n=50 1.25x, n=72 1.92x, n=88 2.52x. The wins begin at 50 and n=40 is
# still excluded by `_SME_MIN_EXACT` = 48, so the surviving floor is the one that separates them.
# Tile-exact sizes already on SME are unchanged: n=64 0.99, n=80 1.00, n=96 1.04.
# A TILE-EXACT SHAPE HAS ITS OWN, LOWER FLOOR — it pays no remainder panel at all, so the only cost
# it carries above two whole tiles is the kernel's own prologue. `_SME_MIN_EXACT` is the floor for a
# RAGGED shape, which must also amortise the padded block; that separation is what the surviving
# floor above was measuring, and re-running the crossover dense (every n, not every other, for the
# reason the `_SME_MIN` note records) puts the exact floor two tiles down:
#
#     n         16    24    32    33    40    47    48    49    64    80    96
#     SME/NEON 1.42  0.29  4.92  0.66  0.93  1.43  6.84  1.36  7.15  9.00  9.42
#
# Exact multiples of MR win from 32 — n=32 measures 4.92x the path it was declining to — while
# ragged shapes do not turn until ~47, which `_SME_MIN_EXACT` = 3*MR already separates. Lowering the
# single floor to 32 instead would admit n=33 at 0.66 and n=40 at 0.93.
#
# ⚠ Those ragged figures were 0.44 / 0.67 / 0.96 at n=49 / 65 / 81 when the floor was last set. They
# moved because gemm now reads its operands in place; see the stale-calibration note in
# `_SYR2K_SME_MIN`. This predicate is the same kind of frozen comparison and wants re-running
# whenever the kernel under it changes.
@inline function _sme_tile_ok(m, n)
    min(m, n) >= 2 * _SME_MR || return false
    (m % _SME_MR == 0 && n % _SME_MR == 0) && return true
    return max(m, n) >= _SME_MIN_EXACT
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

# ══ GEMV ═══════════════════════════════════════════════════════════════════════════════════════
# y = alpha*A*x + beta*y, A column-major Float64.
#
# A DIFFERENT MECHANISM FROM THE GEMM ABOVE, found by elimination. The coprocessor is fast only
# when it accumulates into ZA and slow at ordinary vector arithmetic. Measured on an M6, one
# buffer, four independent chains throughout:
#
#     fadd into z registers ........   66.6 GB/s   (UNCHANGED by 4x wider loads)
#     NEON, 8 accumulators .........  136
#     fmla into ZA, ONE slice group .  256.7
#     fmla into ZA, FOUR groups ..... 1013         (Accelerate reaches 977)
#
# The decisive row is the second: quadrupling bytes per instruction moved the result by 0.1 GB/s,
# which proves the 66.6 wall is the ADD, not the load port. `fmopa` cannot serve here either — it
# reads at most 64 B per instruction, so at one per cycle its ceiling is ~250 GB/s.
#
# So `y` LIVES in ZA while the columns of A stream past: one round trip through the unit instead
# of one per column. Four independent slice groups rather than one is worth 4x, because every
# `fmla` into the same slice is a serial dependency chain.
#
# ⚠ THE vg1x4 SLICE ARGUMENT IS A GROUP SELECTOR, NOT A VECTOR INDEX. Passing `4g` happens to work
# at four groups — those values are distinct and in range — and silently produces WRONG RESULTS at
# eight and sixteen. Consecutive indices 0..NG-1 are correct.
function _gemv_ir(ng::Int, acc::Bool = true, L::Int = _SME_L)
    # ZA HOLDS 8L VECTORS AND A vgx4 GROUP INDEX IS REDUCED MOD 2L BY THE HARDWARE, so a group count
    # above 2L aliases onto low groups and silently sums the wrong rows into y. That is the same
    # wrap the slice-argument note above records, reached the other way. Refuse it here rather than
    # let a machine with a narrower streaming vector build a kernel that returns wrong answers.
    ng <= 2 * L || throw(ArgumentError(
        "SME gemv: $ng slice groups exceeds the $(2L) that ZA can address at $L lanes"))
    q = Char(34)
    T4 = "{ <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double> }"
    rb = 4 * L * ng                   # rows of y per block: ng groups of four L-lane vectors
    io = IOBuffer()
    print(io, """
declare void @llvm.aarch64.sme.za.enable()
declare void @llvm.aarch64.sme.za.disable()
declare void @llvm.aarch64.sme.zero(i32)
declare target($(q)aarch64.svcount$(q)) @llvm.aarch64.sve.ptrue.c64()
declare $T4 @llvm.aarch64.sve.ld1.pn.x4.nxv2f64(target($(q)aarch64.svcount$(q)), ptr)
declare void @llvm.aarch64.sme.fmla.single.vg1x4.nxv2f64(i32, <vscale x 2 x double>,
  <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>)
declare $T4 @llvm.aarch64.sme.read.vg1x4.nxv2f64(i32)

define void @entry(ptr %y, ptr %a, i64 %lda, ptr %x, i64 %n, double %alpha, i64 %nb) {
  call void @k(ptr %y, ptr %a, i64 %lda, ptr %x, i64 %n, double %alpha, i64 %nb)
  ret void
}

define internal void @k(ptr %y, ptr %a, i64 %lda, ptr %x, i64 %n, double %alpha, i64 %nb) #0 {
entry:
  call void @llvm.aarch64.sme.za.enable()
  %pn = call target($(q)aarch64.svcount$(q)) @llvm.aarch64.sve.ptrue.c64()
  %anyb = icmp sgt i64 %nb, 0
  br i1 %anyb, label %blk, label %fin

blk:
  %b = phi i64 [ 0, %entry ], [ %bn, %bend ]
  %boff = mul nsw i64 %b, $rb
  %ablk = getelementptr inbounds double, ptr %a, i64 %boff
  %yblk = getelementptr inbounds double, ptr %y, i64 %boff
  call void @llvm.aarch64.sme.zero(i32 255)
  %anyc = icmp sgt i64 %n, 0
  br i1 %anyc, label %col, label %rd

col:
  %j = phi i64 [ 0, %blk ], [ %jn, %col ]
  %xp = getelementptr inbounds double, ptr %x, i64 %j
  %xs = load double, ptr %xp, align 8
  %xa = fmul double %xs, %alpha
  %e0 = insertelement <vscale x 2 x double> poison, double %xa, i32 0
  %bc = shufflevector <vscale x 2 x double> %e0, <vscale x 2 x double> poison, <vscale x 2 x i32> zeroinitializer
  %co = mul nsw i64 %j, %lda
  %cp = getelementptr inbounds double, ptr %ablk, i64 %co
""")
    for g in 0:(ng-1)
        println(io, "  %p$g = getelementptr inbounds double, ptr %cp, i64 $(4 * L * g)")
        println(io, "  %r$g = call $T4 @llvm.aarch64.sve.ld1.pn.x4.nxv2f64(target($(q)aarch64.svcount$(q)) %pn, ptr %p$g)")
        for k in 0:3
            println(io, "  %w$(g)_$k = extractvalue $T4 %r$g, $k")
        end
        println(io, "  call void @llvm.aarch64.sme.fmla.single.vg1x4.nxv2f64(i32 $g,")
        println(io, "    <vscale x 2 x double> %w$(g)_0, <vscale x 2 x double> %w$(g)_1,")
        println(io, "    <vscale x 2 x double> %w$(g)_2, <vscale x 2 x double> %w$(g)_3, <vscale x 2 x double> %bc)")
    end
    print(io, """
  %jn = add nuw nsw i64 %j, 1
  %cdone = icmp eq i64 %jn, %n
  br i1 %cdone, label %rd, label %col

rd:
""")
    for g in 0:(ng-1)
        println(io, "  %o$g = call $T4 @llvm.aarch64.sme.read.vg1x4.nxv2f64(i32 $g)")
        for k in 0:3
            off = 4 * L * g + L * k
            println(io, "  %z$(g)_$k = extractvalue $T4 %o$g, $k")
            println(io, "  %yp$(g)_$k = getelementptr inbounds double, ptr %yblk, i64 $off")
            if acc
                println(io, "  %yv$(g)_$k = load <vscale x 2 x double>, ptr %yp$(g)_$k, align 8")
                println(io, "  %ys$(g)_$k = fadd <vscale x 2 x double> %yv$(g)_$k, %z$(g)_$k")
                println(io, "  store <vscale x 2 x double> %ys$(g)_$k, ptr %yp$(g)_$k, align 8")
            else
                # beta == 0: ZA already holds exactly alpha*A*x for these rows, so store it. The
                # accumulate form costs a y-load and an fadd per vector -- both z-register ops at
                # ~4 cycles -- and forces a fill!(y, 0) pass beforehand.
                println(io, "  store <vscale x 2 x double> %z$(g)_$k, ptr %yp$(g)_$k, align 8")
            end
        end
    end
    print(io, """
  br label %bend

bend:
  %bn = add nuw nsw i64 %b, 1
  %bdone = icmp eq i64 %bn, %nb
  br i1 %bdone, label %fin, label %blk

fin:
  call void @llvm.aarch64.sme.za.disable()
  ret void
}

attributes #0 = { noinline "aarch64_inout_za" "aarch64_pstate_sm_body"
  "target-features"="+sme,+sme2,+sme-f64f64" }
""")
    return String(take!(io))
end

# One kernel per group count, each looping over its own row blocks internally so streaming mode is
# entered ONCE per call rather than once per block. The remainder is handled by SMALLER ZA blocks,
# not a scalar loop: the scalar tail was the dominant cost whenever m was not a multiple of the
# block (0.08x Accelerate at m=256, 0.17x at m=768) and only a scrap below one vector group stays
# scalar now.
#
# Built even where SME is absent — the strings are constructed, never executed; every entry point
# is gated on `_SME_F64`, exactly as the gemm IR above.
# Rows per ZA vector group: four vectors of `_SME_L` doubles, the smallest block the kernel emits.
const _SME_GEMV_BLK = 4 * _SME_L

# THE LADDER IS BOUNDED BY WHAT ZA CAN ADDRESS, NOT BY A LITERAL. ZA holds 8L vectors and a vgx4
# group index is reduced MOD 2L, so `2L` groups is the largest kernel that addresses distinct rows;
# above it the groups alias and rows are summed into the wrong place, silently. At the 512-bit
# streaming length of an M4-class chip L is 8 and this is 16 groups of 32 rows — the 512-row block
# the ladder used to name as a literal. On a 256-bit part it is 8 groups, and the literal would have
# been wrong by a factor of two with no symptom but bad numbers.
const _SME_GEMV_NGMAX = 2 * _SME_L
const _SME_GEMV_NGS = Tuple(1 << i for i in 0:(ndigits(_SME_GEMV_NGMAX, base = 2) - 1))

const _SME_GEMV_IR = Dict{Tuple{Int, Bool}, String}(
    (ng, acc) => _gemv_ir(ng, acc) for ng in _SME_GEMV_NGS, acc in (true, false)
)

for ng in _SME_GEMV_NGS, acc in (true, false)
    nm = Symbol(acc ? "_sme_gemv_acc" : "_sme_gemv_sto", ng)
    ir = _SME_GEMV_IR[(ng, acc)]
    @eval @inline $nm(y, a, l, x, n, al, nb) = Base.llvmcall(($ir, "entry"), Cvoid,
        Tuple{Ptr{Float64}, Ptr{Float64}, Int64, Ptr{Float64}, Int64, Float64, Int64},
        y, a, Int64(l), x, Int64(n), al, Int64(nb))
end

# Dispatch on the GROUP COUNT, which is what the kernel is parameterised by; the caller converts a
# row block to groups. A chain of `==` over the detected set keeps this a compile-time ladder rather
# than a dynamic lookup, and it cannot drift from `_SME_GEMV_NGS` because it is generated from it.
@eval @inline function _sme_gemv_run_ng(ng::Int, y, a, l, x, n, al, nb, store::Bool)
    $(Expr(:block, (quote
        if ng == $g
            return store ? $(Symbol("_sme_gemv_sto", g))(y, a, l, x, n, al, nb) :
                           $(Symbol("_sme_gemv_acc", g))(y, a, l, x, n, al, nb)
        end
    end for g in _SME_GEMV_NGS)...))
    return nothing
end

@inline _sme_gemv_run(rb::Int, y, a, l, x, n, al, nb, store::Bool) =
    _sme_gemv_run_ng(rb ÷ _SME_GEMV_BLK, y, a, l, x, n, al, nb, store)

# The work below which a ZA-resident y costs more than it saves. The cost is the ZA fill and the
# readback, both O(m) and both paid whatever n is, against a stream of m*n elements — so the
# criterion is the PRODUCT, not either dimension. Measured against the SIMD column-panel kernel it
# displaces, forced SME arm, one process (`bench/probes/sme_gemv_min_crossover.jl`):
#
#     m \ n      4      8     16     32     64    128    256    512
#       32    0.09   0.19   0.30   0.48   0.75   1.12   1.46   1.77
#       96    0.17   0.31   0.55   0.94   1.49   2.02   3.49   3.46
#      128    0.37   0.67   1.09   1.70   2.63   3.18   6.52   7.50
#      512    0.90   1.49   2.32   4.07   8.69  11.77   9.51   8.82
#     1024    1.37   2.12   3.69   7.60  10.40  10.62  12.45   8.77
#
# DERIVING THIS ON ONE n WOULD HAVE BEEN WRONG: an m-only sweep at n=512 put the cut at 96 rows, and
# that admitted 35 shapes at n=64 that run as slow as 0.82 of the path they displaced.
#
# ⚠ THE CUT MOVES WITH THE DRIVER ABOVE IT, and that is why it is a QUARTER of L1 rather than the half
# the table above implies. The fixed cost this cut is paying for is per BLOCK, not per call, so a
# driver that covers a ragged `m` in two blocks instead of six has a different crossing from one that
# does not. Re-measured on square shapes in the gate's own methodology, SME against the SIMD path it
# displaces:
#
#     n           32    40    48    56    64    80    96   112   128
#     m*n       1024  1600  2304  3136  4096  6400  9216 12544 16384
#     SME/SIMD  0.46  0.59  0.64  1.31  1.55  1.38  1.83  2.24  4.05
#
# so the crossing is between 2304 and 3136 elements. A quarter of L1 is 4096, the first power of two
# clear of it, and it admits every square shape from n=64 up. ⚠ If the row driver or the trampoline
# cost changes again, this cut is a frozen comparison against the arm as it stood — re-derive it.
# PDM: Derived — formula over detected consts: a quarter of L1 in elements, `_L1_BYTES ÷ (4 * sizeof(Float64))`, the panel size at which the per-block ZA fill and readback disappear into the stream.
const _SME_GEMV_MINWORK = @load_preference("sme_gemv_minwork",
    _L1_BYTES ÷ (4 * sizeof(Float64)))::Int

# A NON-MULTIPLE IS ADMISSIBLE ONLY WITH beta == 0. Its trailing rows are covered by an overlapping
# full block, which recomputes rows it shares with the previous one -- sound when those rows are
# STORED with the value they already hold, and wrong when they are accumulated into, because the
# overlap would add alpha*A*x twice. The scalar loop that served the tail before dominated the call
# long before it was a small fraction of the rows (m=40 at 0.45 of the SIMD path, m=144 at 0.90).
#
# THEREFORE THIS KERNEL MAY NEVER BE GIVEN A PARTITIONED ROW RANGE. The overlap is sound only
# because the rows it recomputes already hold the value being rewritten, which is a statement about
# the WHOLE call: two row bands that each extend their own trailing block overlap each other's rows
# and both write them, and nothing about `beta == 0` makes that ordering safe. A threaded gemv-N
# splits rows — that is its write-disjoint axis, since rows of A map to elements of y — so the split
# and this kernel are mutually exclusive by construction.
#
# The rule that keeps them apart is the one `_sme_owns` states for gemm: decide SME ownership at the
# threaded ENTRY, before any split, and run an owned call serially. It is also the faster answer on
# this hardware — one coprocessor shared by the cluster, measured 503 owned vs 301 split at n=4096 —
# so there is no case where splitting an eligible call is worth reopening this.
#
# Two size-keyed terms below would ALSO misroute a chunk if one ever reached them: `m * n` can fall
# under `_SME_GEMV_MINWORK` for a slice of a call that clears it whole, and `m % _SME_GEMV_BLK`
# flips with the band height. Both are moot under the entry rule above, and neither is worth a route
# parameter — an unused argument on a hot predicate is not free (see `_trsm!`'s ninth-parameter note).
@inline _sme_gemv_shape_ok(m, n, beta) =
    m * n >= _SME_GEMV_MINWORK && m >= _SME_GEMV_BLK &&
        (m % _SME_GEMV_BLK == 0 || iszero(beta))

# The kernel reads y and A as raw column-major Float64 with unit row stride, and x contiguously.
# ⚠ THIS KERNEL'S RATE IS GOVERNED BY ROWS, NOT BY TOTAL WORK, AND THE FLOOR ABOVE DOES NOT SAY SO.
# Measured through `_sme_gemv!`, Float64, β=1, every block resident, GB/s of A:
#
#     rows \ cols      64    128    256    512   1024
#         32           75    123    165    198    229
#         64          225    175    261    475    510
#        128          126    218    336    503    629
#        256          511    672    783    862    908
#        512          647    790    891    953    986
#
# So m >= 256 is where it performs, and below that it gives up 2-5x however many columns follow. Two
# consequences worth knowing before tuning anything that calls it:
#
#   * A SMALL-n SQUARE gemv IS NOT SLOW BECAUSE IT IS SMALL. At m=n=64 the kernel reaches 225 GB/s and
#     at m=n=512 it reaches 953 — the same kernel, the same residency, four times the rows.
#   * NO TRIANGULAR COVER ESCAPES IT. A block of the strictly-upper triangle with m >= 256 must sit in
#     the top-right corner, so the blocks nearest the diagonal are short BY CONSTRUCTION. That is why
#     `_trmv_split!`'s cover aggregates 283 GB/s out of blocks that individually reach 725: at n=512 the
#     recursive cover is one 256x256 at 783 plus two 128x128 at 218 and four 64x64 at 225, and the
#     staircase alternative is worse (its panels are 64 columns wide, and narrow costs rows).
@inline _sme_gemv_eligible(::Type{T}, m, n, trans, cj, A, x, y, incx, incy, beta) where {T} =
    T === Float64 && _SME_F64 && !trans && !cj && incx == 1 && incy == 1 &&
        eltype(x) === Float64 && eltype(y) === Float64 &&
        _strided1(A) && _dense1(x) && _dense1(y) &&
        _sme_gemv_shape_ok(m, n, beta) && n > 0 && _SME_GEMV_ENTRY[] !== C_NULL


# THE KERNEL MUST NOT ENTER THE PACKAGE IMAGE, and a runtime guard cannot achieve that: codegen
# happens when the caller is COMPILED, not when it runs, so a concrete call from `level2.jl` is
# enough to make the precompile process emit SME2 intrinsics for a generic image CPU and abort with
# `Cannot select: intrinsic llvm.aarch64.sve.ptrue.c64`. Measured, not feared.
#
# So gemv takes the same barrier the gemm path uses: the body is reached only through a function
# POINTER resolved at load time, which inference sees as an opaque `Ptr`. Every argument is
# `Ptr`/`Int`/`Float64`, so the `ccall` boxes nothing.
#
# ⚠ THE CROSSING COSTS ~90-110 ns, AND IT IS THE C ABI MEETING A BODY THAT USES ZA. Calling
# `_sme_gemv_cabi` directly — same body, same kernels, every argument a runtime value so nothing folds:
#
#     m, n            64,64   256,256
#     through pointer  169.7     684.8 ns
#     called directly   67.5     594.6
#
# A bare `za.enable` + `ptrue.c64` + `za.disable` region as a direct `llvmcall` is 12.5 ns, so ZA
# itself is nearly free. What costs is crossing INTO a body that then uses it: a `@cfunction` over a
# body with no ZA work costs +8 ns, and over a body with ONE real kernel +97. Six candidates are
# eliminated — do not re-run them:
#
#   * NOT constant folding: runtime versus literal args moves the direct arm 2 ns.
#   * NOT inlining: `_sme_gemv_cabi` marked `@noinline` still measures 67.5 direct against 169.7.
#   * NOT argument count: +8 ns at one argument and +8 at eight.
#   * NOT the signature: a trivial body under the real mixed signature costs +8.3.
#   * NOT the number of streaming regions: one reachable kernel costs +96.9, all ten +95.7.
#   * NOT the witness counter alone — but it is a SECOND ~95 ns cost that OVERLAPS this one. A
#     `Threads.atomic_add!` on a counter the kernel's stream has evicted costs +69 to +135 ns; a plain
#     `Ref{Int}` increment costs 0. Removing EITHER cost alone changes nothing end to end, because each
#     hides the other; both must go.
#
# ⛔ AND BOTH WAYS OUT ARE BLOCKED, BY DIFFERENT PARTS OF THE BUILD. Measured, not assumed:
#
#   * A DIRECT call is what inference resolves, and precompilation then codegens the kernel for the
#     generic image CPU: `LLVM ERROR: Cannot select: intrinsic llvm.aarch64.sve.ptrue.c64`. A
#     `Ref{typeof(f)}` is the same thing — its eltype is a singleton, so `code_typed` shows one static
#     `:invoke` — and aborts identically.
#   * A DYNAMIC call through a `Ref{Any}` precompiles fine and is as fast as a direct one, and with the
#     arguments written into ONE preallocated per-thread struct it allocates nothing. With both costs
#     removed that way, gemv-N at m=n=64 measured 168 -> 71 ns and n=128 261 -> 180. But
#     `juliac --trim` cannot resolve a dynamic dispatch and the authoritative build fails.
#
# So the prize is ~2.4x on every small-n SME call — gemv-N at n=64 gates 0.48 and would gate above 1 —
# and taking it needs a call that inference cannot resolve at precompile time but `--trim` can resolve
# at build time. That is a toolchain problem, not a kernel one.
function _sme_gemv_cabi(
        y::Ptr{Float64}, a::Ptr{Float64}, lda::Int, x::Ptr{Float64},
        m::Int, n::Int, alpha::Float64, store::Int
    )
    st = store != 0
    ib = 0
    # ONE BLOCK SIZE, THEN ONE OVERLAPPING BLOCK — not a halving ladder, whenever the rows are being
    # STORED. A ladder covers a ragged `m` with progressively narrower blocks, and a narrow block is
    # slow twice over: its group count is its memory-level parallelism (measured 230 GB/s at two
    # groups, 608 at four, 845 at eight, 1009 at sixteen), and every extra block pays another ZA
    # prologue. At m=1000 the ladder issued SIX blocks (512+256+128+64+32+overlap) and at m=100 three
    # (64+32+overlap); the widest block plus one overlapping copy of itself covers either in TWO, at
    # the widest group count throughout. Measured against the ladder, same kernels, no other change:
    # m=100 1.71x, m=1000 1.28x, m=64 2.64x, and unchanged where the ladder already issued one block.
    #
    # Sound only in STORE mode, for the same reason the tail below is: the overlapped rows are
    # rewritten with the value they already hold. Accumulate mode keeps the ladder, where `m` is a
    # multiple of the block anyway because `_sme_gemv_shape_ok` demands it once beta is nonzero.
    if st
        ng = 1
        while 2 * ng * _SME_GEMV_BLK <= m && 2 * ng <= _SME_GEMV_NGMAX
            ng *= 2
        end
        rb = ng * _SME_GEMV_BLK
        nb = m ÷ rb
        if nb > 0
            _sme_gemv_run(rb, y, a, lda, x, n, alpha, nb, true)
            ib = nb * rb
        end
        if ib < m
            off = m - rb
            _sme_gemv_run(rb, y + off * 8, a + off * 8, lda, x, n, alpha, 1, true)
            ib = m
        end
    else
        rb = _SME_GEMV_BLK * _SME_GEMV_NGMAX
        while rb >= _SME_GEMV_BLK
            nb = (m - ib) ÷ rb
            if nb > 0
                _sme_gemv_run(rb, y + ib * 8, a + ib * 8, lda, x, n, alpha, nb, false)
                ib += nb * rb
            end
            rb >>= 1
        end
    end
    if ib < m
        # THE TAIL IS AN OVERLAPPING BLOCK, NOT A SCALAR LOOP. A remainder below one vector group
        # used to run scalar, and that loop dominated the call long before it was a small fraction
        # of the rows: m=40 measured 0.45 of the SIMD path, m=72 0.87, m=144 0.90.
        #
        # Instead run one more full block at `m - BLK`, which recomputes the rows it overlaps. That
        # is only sound in STORE mode, where a row is written with the value it already holds; in
        # accumulate mode the overlap would add alpha*A*x to those rows twice, which is why
        # `_sme_gemv_eligible` requires beta == 0 for a shape that is not an exact multiple.
        if st
            _sme_gemv_run(_SME_GEMV_BLK, y + (m - _SME_GEMV_BLK) * 8,
                          a + (m - _SME_GEMV_BLK) * 8, lda, x, n, alpha, 1, true)
        else
            for j in 0:(n - 1)
                s = alpha * unsafe_load(x + j * 8)
                aj = a + j * lda * 8
                for i in ib:(m - 1)
                    q = y + i * 8
                    unsafe_store!(q, muladd(s, unsafe_load(aj + i * 8), unsafe_load(q)))
                end
            end
        end
    end
    return nothing
end

const _SME_GEMV_TRAMPOLINE = Ref{Any}(nothing)     # roots the closure against collection
const _SME_GEMV_ENTRY = Ref{Ptr{Cvoid}}(C_NULL)

# ══ gemv-T: y += alpha*A'x ══════════════════════════════════════════════════════════════════════
#
# THE TWO FORMS STREAM A IDENTICALLY — column by column, contiguous — and differ only in the
# arithmetic. gemv-N multiplies a column by a BROADCAST SCALAR and accumulates into a vector
# (`fmla.single.vg1x4`); gemv-T multiplies a column by a VECTOR and accumulates into a scalar, so it
# takes the multi-vector `fmla.vg1x4` and reduces at the end. Because the traffic is the same, the
# bandwidth the N form reaches is available here too, and it is bandwidth that this routine wants:
# without it gemvT sits at DRAM speed while its operand is in L2.
#
# `NC` COLUMNS ARE IN FLIGHT AT ONCE, each owning one vg1x4 group of four ZA slices. That buys three
# things at once: `x` is loaded once per 4L rows and reused NC times, the horizontal reduction is
# paid once per NC columns rather than once per column, and there are 4*NC independent accumulator
# chains — the quantity the N form measured as decisive (256.7 GB/s at one slice group, 1013 at
# four). It costs NC concurrent A streams, which is why the widest NC is not the fastest.
#
# ACCUMULATES into y, as the N form does, so `beta` is applied by the caller before the call.
function _gemvt_ir(nc::Int, defer::Bool)
    nc <= 2 * _SME_L || throw(ArgumentError(
        "SME gemv-T: $nc column groups exceeds the $(2 * _SME_L) that ZA can address at $(_SME_L) lanes"))
    L = _SME_L
    q = Char(34)
    T4 = "{ <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double> }"
    io = IOBuffer()
    print(io, """
declare void @llvm.aarch64.sme.za.enable()
declare void @llvm.aarch64.sme.za.disable()
declare void @llvm.aarch64.sme.zero(i32)
declare target($(q)aarch64.svcount$(q)) @llvm.aarch64.sve.ptrue.c64()
declare $T4 @llvm.aarch64.sve.ld1.pn.x4.nxv2f64(target($(q)aarch64.svcount$(q)), ptr)
declare void @llvm.aarch64.sme.fmla.vg1x4.nxv2f64(i32,
  <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>,
  <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>)
declare $T4 @llvm.aarch64.sme.read.vg1x4.nxv2f64(i32)
declare double @llvm.vector.reduce.fadd.nxv2f64(double, <vscale x 2 x double>)

define void @entry(ptr %y, ptr %a, i64 %lda, ptr %x, i64 %m, double %alpha, i64 %nblk) {
  call void @k(ptr %y, ptr %a, i64 %lda, ptr %x, i64 %m, double %alpha, i64 %nblk)
  ret void
}

define internal void @k(ptr %y, ptr %a, i64 %lda, ptr %x, i64 %m, double %alpha, i64 %nblk) #0 {
entry:
  call void @llvm.aarch64.sme.za.enable()
  %pn = call target($(q)aarch64.svcount$(q)) @llvm.aarch64.sve.ptrue.c64()
  %anyb = icmp sgt i64 %nblk, 0
  br i1 %anyb, label %blk, label %fin

blk:
  %b = phi i64 [ 0, %entry ], [ %bn, %bend ]
  %jb = mul nsw i64 %b, $nc
  %yblk = getelementptr inbounds double, ptr %y, i64 %jb
  %aoff = mul nsw i64 %jb, %lda
  %ablk = getelementptr inbounds double, ptr %a, i64 %aoff
  call void @llvm.aarch64.sme.zero(i32 255)
  %anyr = icmp sgt i64 %m, 0
  br i1 %anyr, label %row, label %rd

row:
  %i = phi i64 [ 0, %blk ], [ %in, %row ]
  %xp = getelementptr inbounds double, ptr %x, i64 %i
  %xr = call $T4 @llvm.aarch64.sve.ld1.pn.x4.nxv2f64(target($(q)aarch64.svcount$(q)) %pn, ptr %xp)
  %xv0 = extractvalue $T4 %xr, 0
  %xv1 = extractvalue $T4 %xr, 1
  %xv2 = extractvalue $T4 %xr, 2
  %xv3 = extractvalue $T4 %xr, 3
""")
    for c in 0:(nc - 1)
        println(io, "  %co$c = mul nsw i64 %lda, $c")
        println(io, "  %cb$c = getelementptr inbounds double, ptr %ablk, i64 %co$c")
        println(io, "  %ap$c = getelementptr inbounds double, ptr %cb$c, i64 %i")
        println(io, "  %ar$c = call $T4 @llvm.aarch64.sve.ld1.pn.x4.nxv2f64(target($(q)aarch64.svcount$(q)) %pn, ptr %ap$c)")
        for k in 0:3
            println(io, "  %av$(c)_$k = extractvalue $T4 %ar$c, $k")
        end
        println(io, "  call void @llvm.aarch64.sme.fmla.vg1x4.nxv2f64(i32 $c,")
        println(io, "    <vscale x 2 x double> %av$(c)_0, <vscale x 2 x double> %av$(c)_1,")
        println(io, "    <vscale x 2 x double> %av$(c)_2, <vscale x 2 x double> %av$(c)_3,")
        println(io, "    <vscale x 2 x double> %xv0, <vscale x 2 x double> %xv1,")
        println(io, "    <vscale x 2 x double> %xv2, <vscale x 2 x double> %xv3)")
    end
    print(io, """
  %in = add nuw nsw i64 %i, $(4 * L)
  %rdone = icmp sge i64 %in, %m
  br i1 %rdone, label %rd, label %row

rd:
""")
    for c in 0:(nc - 1)
        println(io, "  %o$c = call $T4 @llvm.aarch64.sme.read.vg1x4.nxv2f64(i32 $c)")
        for k in 0:3
            println(io, "  %zv$(c)_$k = extractvalue $T4 %o$c, $k")
        end
        # The four slice vectors of a group are partial sums of ONE column, so folding them and the
        # lanes is a reduction of that column's dot product; `reassoc` lets it be a tree rather than
        # the sequential FADDA, which is the whole point of holding four chains.
        println(io, "  %s$(c)_a = fadd reassoc <vscale x 2 x double> %zv$(c)_0, %zv$(c)_1")
        println(io, "  %s$(c)_b = fadd reassoc <vscale x 2 x double> %zv$(c)_2, %zv$(c)_3")
        println(io, "  %s$(c)_c = fadd reassoc <vscale x 2 x double> %s$(c)_a, %s$(c)_b")
        if defer
            # DEFERRED: store the folded vector and let the caller sum its lanes outside streaming
            # mode. `faddv` here is two thirds of the per-column cost (see `_SME_GEMVT_DEFER_MAX`),
            # and `alpha` is applied by the caller along with the lane sum.
            println(io, "  %sb$c = mul nsw i64 %jb, $L")
            println(io, "  %si$c = add nsw i64 %sb$c, $(c * L)")
            println(io, "  %sp$c = getelementptr inbounds double, ptr %y, i64 %si$c")
            println(io, "  store <vscale x 2 x double> %s$(c)_c, ptr %sp$c, align 8")
        else
            println(io, "  %h$c = call reassoc double @llvm.vector.reduce.fadd.nxv2f64(double 0.0, <vscale x 2 x double> %s$(c)_c)")
            println(io, "  %yp$c = getelementptr inbounds double, ptr %yblk, i64 $c")
            println(io, "  %yo$c = load double, ptr %yp$c, align 8")
            println(io, "  %r$c = call double @llvm.fmuladd.f64(double %h$c, double %alpha, double %yo$c)")
            println(io, "  store double %r$c, ptr %yp$c, align 8")
        end
    end
    print(io, """
  br label %bend

bend:
  %bn = add nuw nsw i64 %b, 1
  %bdone = icmp eq i64 %bn, %nblk
  br i1 %bdone, label %fin, label %blk

fin:
  call void @llvm.aarch64.sme.za.disable()
  ret void
}

declare double @llvm.fmuladd.f64(double, double, double)
""")
    print(io, _SME_ATTRS)
    return String(take!(io))
end

# Column groups in flight. Bounded above by the vg1x4 groups ZA can address; the useful range is
# narrower and is chosen per call by `_sme_gemvt_nc`.
const _SME_GEMVT_NCS = (2, 4, 8)
const _SME_GEMVT_IR = Dict{Int, String}(nc => _gemvt_ir(nc, false) for nc in _SME_GEMVT_NCS)
# Same kernel, epilogue deferred: it stores each column's folded vector into a scratch strip instead
# of reducing it in streaming mode. See `_SME_GEMVT_DEFER_MAX` for when that is the faster shape.
const _SME_GEMVT_DEFER_IR = Dict{Int, String}(nc => _gemvt_ir(nc, true) for nc in _SME_GEMVT_NCS)

for nc in _SME_GEMVT_NCS
    @eval @inline $(Symbol("_sme_gemvt", nc))(y, a, l, x, m, al, nb) =
        Base.llvmcall(($(_SME_GEMVT_IR[nc]), "entry"), Cvoid,
            Tuple{Ptr{Float64}, Ptr{Float64}, Int64, Ptr{Float64}, Int64, Float64, Int64},
            y, a, Int64(l), x, Int64(m), al, Int64(nb))
    @eval @inline $(Symbol("_sme_gemvtd", nc))(s, a, l, x, m, al, nb) =
        Base.llvmcall(($(_SME_GEMVT_DEFER_IR[nc]), "entry"), Cvoid,
            Tuple{Ptr{Float64}, Ptr{Float64}, Int64, Ptr{Float64}, Int64, Float64, Int64},
            s, a, Int64(l), x, Int64(m), al, Int64(nb))
end

@inline function _sme_gemvt_run(nc::Int, y, a, l, x, m, al, nb)
    nc == 2 && return _sme_gemvt2(y, a, l, x, m, al, nb)
    nc == 4 && return _sme_gemvt4(y, a, l, x, m, al, nb)
    return _sme_gemvt8(y, a, l, x, m, al, nb)
end

# Deferred arm. `s` is the scratch strip, `nb * _SME_L` doubles, NOT y.
@inline function _sme_gemvt_drun(nc::Int, s, a, l, x, m, al, nb)
    nc == 2 && return _sme_gemvtd2(s, a, l, x, m, al, nb)
    nc == 4 && return _sme_gemvtd4(s, a, l, x, m, al, nb)
    return _sme_gemvtd8(s, a, l, x, m, al, nb)
end

# One scratch strip per thread, grow-only through `_ws_grow!`, so the deferred arm allocates only on a
# thread's first call through it. `_sme_gemvt_cabi` claims it and drops it within the same call and
# makes no public Level-3 call, so it cannot be live across a threaded join — the same argument the
# trsv reciprocal caches carry in `test/perthread_lint_baseline.txt`.
const _SME_GEMVT_SCR = Base.OncePerThread{Vector{Float64}}(() -> Float64[])
# ⚠ THE HORIZONTAL REDUCTION IS TWO THIRDS OF THE PER-COLUMN COST, AND IT IS WHY THERE ARE TWO ARMS.
#
# Each column block ends with a fold of the four ZA slices plus, in the fused arm, one
# `llvm.vector.reduce.fadd` per column — a streaming-mode `faddv`. Holding one column block and
# sweeping m separates the three costs a square sweep leaves entangled: about 150 ns per call, about
# 40 ns per NC=4 block, of which only 14 ns is the column stream itself at 583 GB/s. Pricing the
# reduce by replacing it with a lane-0 extract — wrong answer, timing only:
#
#     m = n          256    512   1024  |  m=256, n=64
#     with faddv    2608   6017  17541  |       811 ns
#     without       1366   4244  16666  |       349 ns
#     ratio         1.91   1.42   1.05  |      2.33
#
# THE DEFERRED ARM takes that out of streaming mode: fold the four slices as before, store the folded
# vector into a per-column strip of `_SME_GEMVT_SCR`, and sum its `_SME_L` lanes in the caller. It is
# correct to 3e-16 and it is what `_SME_GEMVT_DEFER_MAX` selects.
#
# ⛔ SVE `faddp` IS NOT AN ALTERNATIVE, and the shape of the failure is worth keeping: pairwise adds
# are SEGMENT-WISE — they pair lanes inside each 128-bit segment and never cross one — so three of
# them leave four partial sums in a 512-bit vector instead of one total. The variant measures
# 1.05-1.62x and returns the wrong answer. That timing is a floor on what a correct cross-segment
# sequence would have to beat, not a result.
#
# ⚠ TAKE NO m=128 NUMBER FROM A SINGLE PROCESS. One run of the deferred arm read 518 ns there and
# three later runs read 1385 ns for the same code. At m=128 the leading dimension is 1024 B, so four
# columns span exactly one page and the scratch can alias them. m=128 is below `_SME_GEMVT_MINM`
# either way: the deferred arm is 0.61x of the NEON path there, so neither arm lowers that floor.
#
# ⛔ AND VECTORISING THE SCALAR ROW TAIL IS NOT THE WIN IT LOOKS LIKE. The tail runs once per COLUMN
# and costs 28% of the call at m=1016 against m=992 (9875 ns against 7729), which reads like a
# scalar-arithmetic problem. It is not. Replacing it with `_dot_simd` per column, and then with ONE
# blocked `_gemv_t_simd!` over the whole tail block, both measure NEUTRAL-TO-WORSE in a single-session
# A/B: m=1000 n=1000 went 16833 -> 17625 ns blocked. The cost is a second, scattered pass over A —
# one short strided chunk per column — so it is memory-bound and no arithmetic rewrite reaches it.
# Removing it means predicating the tail rows inside the main kernel so A is read once.


#
# ⛔ MORE ZA GROUPS PER COLUMN DOES NOT LIFT THIS KERNEL'S RATE, AND THE gemv-N TABLE DOES NOT TRANSFER.
# gemv-T at n=1024 reaches 514 GB/s where gemv-N reaches 988 on the same bytes, and the obvious read of
# the note above `_sme_gemv_eligible` — group count IS memory-level parallelism, 230 GB/s at two groups
# rising to 1009 at sixteen — says to give each column more than one. It measures the other way.
# Prototyped as nc columns x G ZA row-groups, row step 4L*G, against the shipped kernel:
#
#     n        nc4 G1  nc4 G2  nc4 G4  nc2 G8  nc8 G2
#      256       0.79    0.66    0.42    0.23    0.60
#      512       0.85    0.69    0.47    0.24    0.61
#     1024       0.98    0.80    0.58    0.22    0.71
#
# The load count per element is IDENTICAL across G — G x-loads and 4G A-loads cover 4L*G rows of nc
# columns either way — so this is not traffic. The live set is what changes: G=4 at nc=4 needs sixteen
# x vectors and sixteen A vectors live per iteration, which is the whole register file.
#
# What the gemv-N table actually measures is CONTIGUITY PER COLUMN, not ZA groups: more groups there
# means a taller row block, so more consecutive bytes are read from each column. In the T form the
# columns are already swept whole, so there is nothing for extra groups to make more contiguous.
#
# The 1.9x against gemv-N is therefore still unexplained. A quarter of it is accounted: the T form
# re-reads x once per column block, which at nc=4 is m*n/4 elements — 2 MB against A's 8.4 at n=1024.
# Measured on square Float64 through `gemv!`, best NC per size (bench/probes/sme_gemvt_proto.jl):
#     n        128   256   512  1024  2048
#     NC=2    1.00  2.69  3.79  4.14  1.94
#     NC=4    1.00  2.64  3.93  4.30  2.05
#     NC=8    0.99  2.31  2.78  3.53  1.99
# Four is the best or within noise of it everywhere, and eight is worse at every size — the extra
# concurrent A streams cost more than the x traffic they save. So NC is not swept per call.
# PDM: Literal — a falsified-derivation literal: the criterion would be "widest NC that still saves x traffic", and it predicts 8, which measures WORSE at every size. Four is what the table above says. | tune: candidate, (2,4,8)
const _SME_GEMVT_NC = @load_preference("sme_gemvt_nc", 4)::Int   # req8-ok: falsified-derivation literal, see table above

# THE FLOOR IS ON m, NOT ON m*n, and that is the whole difference from the N form's cut. The fixed
# cost here is per COLUMN — zero ZA, read four slices back, fold — and the work a column does is
# proportional to m, so the overhead ratio falls as 1/m and does not care how many columns follow.
#
# Measured in ONE process, both arms on the same arrays, `_sme_gemvt!` (every tail included) against
# the `_gemv_t_simd!` NEON path it displaces. Two runs, Float64 square:
#
#     m          136   144   152  |  160   168   176   184   192   200   208   216   224   232   256
#     m % 4L       8    16    24  |    0     8    16    24     0     8    16    24     0     8     0
#     SME/NEON  0.69  0.78  0.80  | 1.10  1.19  1.26  1.17  1.60  1.70  1.63  1.52  1.90  1.92  2.23
#
# so it loses below 160 and wins at EVERY m from 160 up, and the floor sits at the break.
#
# ⚠ THE BAND USED TO BE HALF SPEED AT EVERY NON-MULTIPLE OF 4L, and it is not any more. The row tail
# is still scalar and still runs once per column, but it was never the dominant term: the streaming
# `faddv` was, and `_SME_GEMVT_DEFER_MAX` removes it below 640. With that gone the tail is visible as
# a dip (216 and 248 sag against their neighbours) rather than as a cliff, and 168, 200, 232 — all
# non-multiples — win by 1.19x to 1.92x. The floor moved because the reduction moved, NOT because
# the tail was fixed; vectorising the tail was tried and measured neutral-to-worse, see the note
# above `_SME_GEMVT_SCR`.
#
# ⛔ STILL DO NOT SET THIS FLOOR FROM A SWEEP THAT ONLY LANDS ON MULTIPLES OF 4L, and do not set it
# from one process. The same prototype read m=160 at 1475 ns in one run and 621 ns in the next; at
# these sizes the leading dimension is a small multiple of a page and the operands alias. Every
# number in the table above is a median of five, taken in-process against a same-array control, and
# it reproduced across two runs.
#
# ⛔ m=128 IS AT THE NEON CEILING AND THE MATRIX UNIT CANNOT REACH IT. This is the cell that binds
# gemv-T's gate, so read this before trying to move it. The NEON path runs 128 KB in 844 ns, which is
# 155 GB/s — ABOVE what any other load-and-reduce kernel here reaches on L1-resident data:
#
#     dot 32 KB x2  168 GB/s    dot 128 KB x2  146    asum 64 KB  146    asum 128 KB  146
#
# so the displaced path is not leaving anything on the table. The SME arm cannot take the cell either,
# and the reason is structural rather than tunable: one ZA drain per column is irreducible, it costs
# about 7.8 ns, and at 128 rows a column's own stream is only 4 KB. Narrowing the drain from four ZA
# groups to one would remove three folds out of five ops and still land above the NEON time.
# Accelerate reaches about 457 GB/s here, which is matrix-unit throughput at a per-column cost this
# kernel shape does not have. Moving this cell needs a different decomposition, not a better constant.
# PDM: Derived — the per-column ZA fill and readback is O(1) against O(m) of streamed column, so the crossover is a row count; it sits at the measured break against the NEON path it displaces. | tune: sweep m at fixed n
const _SME_GEMVT_MINM = @load_preference("sme_gemvt_minm", 5 * _SME_GEMV_BLK)::Int

# The m below which the deferred epilogue wins. Both what it removes (one `faddv` per column) and what
# it adds (`_SME_L` doubles of scratch, written then read) are per COLUMN and O(1) in m, so the
# crossover should not depend on m at all. It does, monotonically, which falsifies that criterion:
#
#     m = n      192   256   320   384   448   512   640   768   896  1024  1280
#     deferred  1.16  1.26  1.25  1.23  1.17  1.17  1.02  1.02  0.94  0.97  0.91
#
# so the cut goes at the end of the winning band, below the neutral pair at 640 and 768.
# PDM: Literal — a falsified-derivation literal: the criterion would be "both costs are per column and m-independent, so the crossover is m-independent", which the table above contradicts. | tune: candidate, sweep m at fixed n
const _SME_GEMVT_DEFER_MAX = @load_preference("sme_gemvt_defer_max", 640)::Int   # req8-ok: falsified-derivation literal, see table above

# y += alpha*A'x. Bulk columns and whole 4L-row groups run on ZA; the two tails are scalar, and both
# are bounded: the row tail is under 4L rows of every column, the column tail under NC whole columns.
function _sme_gemvt_cabi(
        y::Ptr{Float64}, a::Ptr{Float64}, lda::Int, x::Ptr{Float64},
        m::Int, n::Int, alpha::Float64
    )
    nc = _SME_GEMVT_NC
    rb = 4 * _SME_L
    mb = (m ÷ rb) * rb                    # rows the kernel covers
    nb = (n ÷ nc) * nc                    # columns the kernel covers
    if mb > 0 && nb > 0
        if mb < _SME_GEMVT_DEFER_MAX
            # Deferred epilogue: the kernel writes one `_SME_L`-lane strip per column into scratch and
            # the lane sums, with alpha, are applied here — outside streaming mode.
            scr = _ws_grow!(_SME_GEMVT_SCR(), nb * _SME_L)
            GC.@preserve scr begin
                _sme_gemvt_drun(nc, pointer(scr), a, lda, x, mb, alpha, nb ÷ nc)
            end
            @inbounds for j in 0:(nb - 1)
                b = j * _SME_L
                s0 = 0.0; s1 = 0.0
                for k in 1:2:_SME_L
                    s0 += scr[b + k]; s1 += scr[b + k + 1]
                end
                q = y + j * 8
                unsafe_store!(q, muladd(alpha, s0 + s1, unsafe_load(q)))
            end
        else
            _sme_gemvt_run(nc, y, a, lda, x, mb, alpha, nb ÷ nc)
        end
    end
    # Row tail, for the columns the kernel handled. It ACCUMULATES, matching the kernel.
    if mb < m
        for j in 0:(nb - 1)
            aj = a + j * lda * 8
            s = 0.0
            for i in mb:(m - 1)
                s = muladd(unsafe_load(aj + i * 8), unsafe_load(x + i * 8), s)
            end
            q = y + j * 8
            unsafe_store!(q, muladd(alpha, s, unsafe_load(q)))
        end
    end
    # Column tail: whole columns past the last group.
    for j in nb:(n - 1)
        aj = a + j * lda * 8
        s = 0.0
        for i in 0:(m - 1)
            s = muladd(unsafe_load(aj + i * 8), unsafe_load(x + i * 8), s)
        end
        q = y + j * 8
        unsafe_store!(q, muladd(alpha, s, unsafe_load(q)))
    end
    return nothing
end

const _SME_GEMVT_TRAMPOLINE = Ref{Any}(nothing)    # roots the closure against collection
const _SME_GEMVT_ENTRY = Ref{Ptr{Cvoid}}(C_NULL)

# `y` is n long and `x` is m long here — the mirror of the N form — and the kernel reads A as raw
# column-major Float64 with unit row stride.
@inline _sme_gemvt_eligible(::Type{T}, m, n, trans, cj, A, x, y, incx, incy) where {T} =
    T === Float64 && _SME_F64 && trans && !cj && !(@inbounds _EXPFLAG[_EXP19]) && incx == 1 && incy == 1 &&
        eltype(x) === Float64 && eltype(y) === Float64 &&
        _strided1(A) && _dense1(x) && _dense1(y) &&
        m >= _SME_GEMVT_MINM && n > 0 && _SME_GEMVT_ENTRY[] !== C_NULL

# Mode 1 hands `x` and `y` as RAW POINTERS (`cabi_l2.jl` wraps only A, as a `PtrMatrix`), and
# `_dense1` admits those by design, so the operands here are `Ptr` or `Vector` depending on which
# mode called. `pointer` covers the array cases; the `Ptr` identity covers the C ABI.
@inline _sme_p(v::Ptr{Float64}) = v
@inline _sme_p(v) = pointer(v)

@noinline function _sme_gemv!(m::Int, n::Int, alpha::Float64, A, x, beta::Float64, y)
    Threads.atomic_add!(_SME_GEMV_CALLS, 1)
    lda = stride(A, 2)
    GC.@preserve y A x begin
        py = _sme_p(y)
        if !iszero(beta) && !isone(beta)
            for i in 0:(m - 1)
                q = py + i * 8
                unsafe_store!(q, beta * unsafe_load(q))
            end
        end
        ccall(
            _SME_GEMV_ENTRY[], Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Int, Ptr{Float64}, Int, Int, Float64, Int),
            py, _sme_p(A), lda, _sme_p(x), m, n, alpha, iszero(beta) ? 1 : 0
        )
    end
    return y
end

const _SME_CALLS = Threads.Atomic{Int}(0)

# The gemv kernel's own witness, separate from `_SME_CALLS` so the threading liveness gate that
# reads that one is not perturbed by a gemv.
#
# IT EXISTS BECAUSE A ROUTE WITH NO WITNESS CANNOT BE A/B'd: a predicate that silently declines and
# a kernel that is genuinely no faster produce the same null result. Asking `_SME_CALLS` about a
# gemv answers zero whatever happens, and a control is the only thing that catches that — a plain
# `gemv!` reports zero there too, which is impossible if the counter covered it.
const _SME_GEMV_CALLS = Threads.Atomic{Int}(0)

# The transposed kernel's own witness, for the same reason the N form has one: a predicate that
# silently declines and a kernel that is genuinely no faster produce the same null result.
const _SME_GEMVT_CALLS = Threads.Atomic{Int}(0)

# y := alpha*A'x + beta*y. The kernel ACCUMULATES, so beta is applied here first — over `n` elements,
# which is the short side, so the extra pass is negligible against the m*n stream.
@noinline function _sme_gemvt!(m::Int, n::Int, alpha::Float64, A, x, beta::Float64, y)
    Threads.atomic_add!(_SME_GEMVT_CALLS, 1)
    lda = stride(A, 2)
    GC.@preserve y A x begin
        py = _sme_p(y)
        if iszero(beta)
            for j in 0:(n - 1)
                unsafe_store!(py + j * 8, 0.0)
            end
        elseif !isone(beta)
            for j in 0:(n - 1)
                q = py + j * 8
                unsafe_store!(q, beta * unsafe_load(q))
            end
        end
        ccall(
            _SME_GEMVT_ENTRY[], Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Int, Ptr{Float64}, Int, Int, Float64),
            py, _sme_p(A), lda, _sme_p(x), m, n, alpha
        )
    end
    return y
end

# ══ ger: A += alpha * x * y' ON ZA ══════════════════════════════════════════════════════════════
#
# A RANK-1 UPDATE IS `axpy` IN TWO DIMENSIONS: column j is `A[:,j] += (alpha*y[j]) * x`, one fused
# multiply-add per element, read and written. That is the shape `_sme_rmw_ir` already serves in
# BLAS-1, and it is capped the same way — a streaming-mode copy of a resident buffer runs 404 GB/s
# and one `fmul` per vector in z registers drops it to 138, so the wall is z-register arithmetic and
# ZA is what removes it.
#
# ⚠ THE GROUPS GO ACROSS COLUMNS, NOT DOWN ROWS, and that is the whole design. Splitting the rows of
# ONE column across G groups looks natural and fails on short matrices: a column shorter than `4L*G`
# leaves the main loop with no trips, every block falls to a single-group tail, and `zero {za}` is
# still paid once per column. Measured, that arrangement took ger@50 from 0.99 to 0.096 and ger@100
# from 0.65 to 0.146 while helping only n >= 128. Giving each group its OWN COLUMN of the same row
# block instead makes the parallelism independent of `m`, amortizes one `zero {za}` over G columns,
# and loads the block of `x` once for all G of them rather than once per column.
#
# THE WHOLE MATRIX LIVES IN ONE STREAMING REGION. Calling the BLAS-1 `axpy` kernel once per column
# would pay the ~50 ns function-pointer barrier and a ZA prologue n times over, which at n=1024 costs
# more than the entire operation; and ordinary floating-point work between streaming calls carries
# its own ~50 ns whichever side it sits on.
#
# `alpha*y[j]` is hoisted and applied as one fused multiply-add, which is exactly what `_ger_simd!`
# does through `_axpy_simd!`, so the result is bit-for-bit the shipped arithmetic.
function _sme_ger_ir(G::Int)
    q = Char(34)
    SC = "target($(q)aarch64.svcount$(q))"
    T4 = "{ <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double> }"
    io = IOBuffer()
    # `alpha*y[j]` broadcast for the column named by `jexpr`.
    bcast = function (jexpr, sfx)
        println(io, "  %yp$sfx = getelementptr inbounds double, ptr %y, i64 $jexpr")
        println(io, "  %ys$sfx = load double, ptr %yp$sfx, align 8")
        println(io, "  %ya$sfx = fmul double %ys$sfx, %alpha")
        println(io, "  %be$sfx = insertelement <vscale x 2 x double> poison, double %ya$sfx, i32 0")
        println(io, "  %bc$sfx = shufflevector <vscale x 2 x double> %be$sfx, <vscale x 2 x double> poison, <vscale x 2 x i32> zeroinitializer")
        println(io, "  %cf$sfx = mul nsw i64 $jexpr, %lda")
        println(io, "  %ac$sfx = getelementptr inbounds double, ptr %a, i64 %cf$sfx")
    end
    # One whole block of column `sfx` through ZA group `g`, with the block of x already in %xv*.
    blk = function (g, sfx, off)
        println(io, "  %pa$sfx = getelementptr inbounds double, ptr %ac$sfx, i64 $off")
        println(io, "  %ra$sfx = call $T4 @llvm.aarch64.sve.ld1.pn.x4.nxv2f64($SC %pn, ptr %pa$sfx)")
        for k in 0:3
            println(io, "  %av$(sfx)_$k = extractvalue $T4 %ra$sfx, $k")
        end
        println(io, "  call void @llvm.aarch64.sme.fmla.single.vg1x4.nxv2f64(i32 $g,")
        println(io, "    <vscale x 2 x double> %av$(sfx)_0, <vscale x 2 x double> %av$(sfx)_1,")
        println(io, "    <vscale x 2 x double> %av$(sfx)_2, <vscale x 2 x double> %av$(sfx)_3, <vscale x 2 x double> %one)")
        println(io, "  call void @llvm.aarch64.sme.fmla.single.vg1x4.nxv2f64(i32 $g,")
        println(io, "    <vscale x 2 x double> %xv0, <vscale x 2 x double> %xv1,")
        println(io, "    <vscale x 2 x double> %xv2, <vscale x 2 x double> %xv3, <vscale x 2 x double> %bc$sfx)")
        println(io, "  %z$sfx = call $T4 @llvm.aarch64.sme.read.vg1x4.nxv2f64(i32 $g)")
        for k in 0:3
            println(io, "  %q$(sfx)_$k = extractvalue $T4 %z$sfx, $k")
        end
        println(io, "  call void @llvm.aarch64.sve.st1.pn.x4.nxv2f64(<vscale x 2 x double> %q$(sfx)_0, <vscale x 2 x double> %q$(sfx)_1, <vscale x 2 x double> %q$(sfx)_2, <vscale x 2 x double> %q$(sfx)_3, $SC %pn, ptr %pa$sfx)")
    end
    # Rows below one whole block, for column `sfx`, predicated.
    scrap = function (sfx, lbl, pred)
        println(io, "$lbl:")
        println(io, "  %si$sfx = phi i64 [ %whole, %$pred ], [ %sn$sfx, %$lbl ]")
        println(io, "  %sg$sfx = call <vscale x 2 x i1> @llvm.aarch64.sve.whilelt.nxv2i1.i64(i64 %si$sfx, i64 %m)")
        println(io, "  %sa$sfx = getelementptr inbounds double, ptr %ac$sfx, i64 %si$sfx")
        println(io, "  %sx$sfx = getelementptr inbounds double, ptr %x, i64 %si$sfx")
        println(io, "  %sav$sfx = call <vscale x 2 x double> @llvm.aarch64.sve.ld1.nxv2f64(<vscale x 2 x i1> %sg$sfx, ptr %sa$sfx)")
        println(io, "  %sxv$sfx = call <vscale x 2 x double> @llvm.aarch64.sve.ld1.nxv2f64(<vscale x 2 x i1> %sg$sfx, ptr %sx$sfx)")
        println(io, "  %sr$sfx = call <vscale x 2 x double> @llvm.fma.nxv2f64(<vscale x 2 x double> %bc$sfx, <vscale x 2 x double> %sxv$sfx, <vscale x 2 x double> %sav$sfx)")
        println(io, "  call void @llvm.aarch64.sve.st1.nxv2f64(<vscale x 2 x double> %sr$sfx, <vscale x 2 x i1> %sg$sfx, ptr %sa$sfx)")
        println(io, "  %sn$sfx = add nsw i64 %si$sfx, %lanes")
        println(io, "  %sd$sfx = icmp slt i64 %sn$sfx, %m")
    end
    print(io, """
declare void @llvm.aarch64.sme.za.enable()
declare void @llvm.aarch64.sme.za.disable()
declare void @llvm.aarch64.sme.zero(i32)
declare $SC @llvm.aarch64.sve.ptrue.c64()
declare <vscale x 2 x i1> @llvm.aarch64.sve.whilelt.nxv2i1.i64(i64, i64)
declare $T4 @llvm.aarch64.sve.ld1.pn.x4.nxv2f64($SC, ptr)
declare void @llvm.aarch64.sve.st1.pn.x4.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, $SC, ptr)
declare <vscale x 2 x double> @llvm.aarch64.sve.ld1.nxv2f64(<vscale x 2 x i1>, ptr)
declare void @llvm.aarch64.sve.st1.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, ptr)
declare void @llvm.aarch64.sme.fmla.single.vg1x4.nxv2f64(i32, <vscale x 2 x double>,
  <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>)
declare $T4 @llvm.aarch64.sme.read.vg1x4.nxv2f64(i32)
declare <vscale x 2 x double> @llvm.fma.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>)
declare i64 @llvm.vscale.i64()

define void @entry(ptr %a, i64 %lda, ptr %x, ptr %y, i64 %m, i64 %n, double %alpha) {
  call void @k(ptr %a, i64 %lda, ptr %x, ptr %y, i64 %m, i64 %n, double %alpha)
  ret void
}
define internal void @k(ptr %a, i64 %lda, ptr %x, ptr %y, i64 %m, i64 %n, double %alpha) #0 {
entry:
  call void @llvm.aarch64.sme.za.enable()
  %pn = call $SC @llvm.aarch64.sve.ptrue.c64()
  %vs = call i64 @llvm.vscale.i64()
  %lanes = shl i64 %vs, 1
  %w = shl i64 %vs, 3
  %e1 = insertelement <vscale x 2 x double> poison, double 1.0, i32 0
  %one = shufflevector <vscale x 2 x double> %e1, <vscale x 2 x double> poison, <vscale x 2 x i32> zeroinitializer
  %nb = sdiv i64 %m, %w
  %whole = mul nsw i64 %nb, %w
  %hasrow = icmp sgt i64 %nb, 0
  %hasscrap = icmp slt i64 %whole, %m
  %cg = sdiv i64 %n, $G
  %cov = mul nsw i64 %cg, $G
  %anyg = icmp sgt i64 %cg, 0
  br i1 %anyg, label %cgrp, label %ctailchk
cgrp:
  %jg = phi i64 [ 0, %entry ], [ %jgn, %cgend ]
  %j0 = mul nsw i64 %jg, $G
""")
    for g in 0:(G - 1)
        println(io, "  %jj$g = add nsw i64 %j0, $g")
        bcast("%jj$g", "m$g")
    end
    print(io, """
  br i1 %hasrow, label %row, label %mscrapchk
row:
  %b = phi i64 [ 0, %cgrp ], [ %bn, %row ]
  %ro = mul nsw i64 %b, %w
  call void @llvm.aarch64.sme.zero(i32 255)
  %pxb = getelementptr inbounds double, ptr %x, i64 %ro
  %rxb = call $T4 @llvm.aarch64.sve.ld1.pn.x4.nxv2f64($SC %pn, ptr %pxb)
  %xv0 = extractvalue $T4 %rxb, 0
  %xv1 = extractvalue $T4 %rxb, 1
  %xv2 = extractvalue $T4 %rxb, 2
  %xv3 = extractvalue $T4 %rxb, 3
""")
    for g in 0:(G - 1)
        blk(g, "m$g", "%ro")
    end
    print(io, """
  %bn = add nuw nsw i64 %b, 1
  %bd = icmp eq i64 %bn, %nb
  br i1 %bd, label %mscrapchk, label %row
mscrapchk:
  br i1 %hasscrap, label %mscrap0, label %cgend
""")
    for g in 0:(G - 1)
        lbl = "mscrap$g"
        pred = g == 0 ? "mscrapchk" : "mscrap$(g)chk"
        g > 0 && println(io, "$pred:\n  br label %$lbl")
        scrap("m$g", lbl, pred)
        nxt = g == G - 1 ? "cgend" : "mscrap$(g+1)chk"
        println(io, "  br i1 %sdm$g, label %$lbl, label %$nxt")
    end
    print(io, """
cgend:
  %jgn = add nuw nsw i64 %jg, 1
  %jgd = icmp eq i64 %jgn, %cg
  br i1 %jgd, label %ctailchk, label %cgrp
ctailchk:
  %hastailc = icmp slt i64 %cov, %n
  br i1 %hastailc, label %ctail, label %fin
ctail:
  %jt = phi i64 [ %cov, %ctailchk ], [ %jtn, %ctend ]
""")
    bcast("%jt", "t")
    print(io, """
  br i1 %hasrow, label %trow, label %tscrapchk
trow:
  %tb = phi i64 [ 0, %ctail ], [ %tbn, %trow ]
  %tro = mul nsw i64 %tb, %w
  call void @llvm.aarch64.sme.zero(i32 255)
  %tpx = getelementptr inbounds double, ptr %x, i64 %tro
  %trx = call $T4 @llvm.aarch64.sve.ld1.pn.x4.nxv2f64($SC %pn, ptr %tpx)
  %xv0t = extractvalue $T4 %trx, 0
  %xv1t = extractvalue $T4 %trx, 1
  %xv2t = extractvalue $T4 %trx, 2
  %xv3t = extractvalue $T4 %trx, 3
  %tpa = getelementptr inbounds double, ptr %act, i64 %tro
  %tra = call $T4 @llvm.aarch64.sve.ld1.pn.x4.nxv2f64($SC %pn, ptr %tpa)
  %tav0 = extractvalue $T4 %tra, 0
  %tav1 = extractvalue $T4 %tra, 1
  %tav2 = extractvalue $T4 %tra, 2
  %tav3 = extractvalue $T4 %tra, 3
  call void @llvm.aarch64.sme.fmla.single.vg1x4.nxv2f64(i32 0,
    <vscale x 2 x double> %tav0, <vscale x 2 x double> %tav1,
    <vscale x 2 x double> %tav2, <vscale x 2 x double> %tav3, <vscale x 2 x double> %one)
  call void @llvm.aarch64.sme.fmla.single.vg1x4.nxv2f64(i32 0,
    <vscale x 2 x double> %xv0t, <vscale x 2 x double> %xv1t,
    <vscale x 2 x double> %xv2t, <vscale x 2 x double> %xv3t, <vscale x 2 x double> %bct)
  %tz = call $T4 @llvm.aarch64.sme.read.vg1x4.nxv2f64(i32 0)
  %tq0 = extractvalue $T4 %tz, 0
  %tq1 = extractvalue $T4 %tz, 1
  %tq2 = extractvalue $T4 %tz, 2
  %tq3 = extractvalue $T4 %tz, 3
  call void @llvm.aarch64.sve.st1.pn.x4.nxv2f64(<vscale x 2 x double> %tq0, <vscale x 2 x double> %tq1, <vscale x 2 x double> %tq2, <vscale x 2 x double> %tq3, $SC %pn, ptr %tpa)
  %tbn = add nuw nsw i64 %tb, 1
  %tbd = icmp eq i64 %tbn, %nb
  br i1 %tbd, label %tscrapchk, label %trow
tscrapchk:
  br i1 %hasscrap, label %tscrap, label %ctend
""")
    scrap("t", "tscrap", "tscrapchk")
    print(io, """
  br i1 %sdt, label %tscrap, label %ctend
ctend:
  %jtn = add nuw nsw i64 %jt, 1
  %jtd = icmp eq i64 %jtn, %n
  br i1 %jtd, label %fin, label %ctail
fin:
  call void @llvm.aarch64.sme.za.disable()
  ret void
}
""")
    print(io, _SME_ATTRS)
    return String(take!(io))
end
# Columns handled at once, one ZA group each. Every group carries its own load, its own pair of ZA
# accumulates and its own store, so groups buy independent memory streams as well as independent
# dependency chains. Measured on the rank-1 update, GB/s over 2n^2 bytes:
#
#     n          128   256   512  1024
#     G=1        240   240   240   240
#     G=2        378   381   381   380
#     G=4        441   450   447   422
#     G=8          -   479   402   435
#
# Four is the best or within noise of it at every size and the ladder is flat from there.
# PDM: Measured — each group carries its own load, accumulates and store, so this is stream count and dependency depth together, not residency or width. | tune: candidate, (1,2,4,8)
const _SME_GER_G = @load_preference("sme_ger_groups", 4)::Int   # req8-ok: swept, see the table above

# The groups span COLUMNS, so a matrix narrower than G needs a narrower kernel; the driver picks the
# widest power of two the columns support from this ladder.
const _SME_GER_GS = Tuple(1 << i for i in 0:Int(log2(_SME_GER_G)))
const _SME_GER_IRS = Dict{Int, String}(g => _sme_ger_ir(g) for g in _SME_GER_GS)

for g in _SME_GER_GS
    @eval @inline $(Symbol("_sme_ger_k", g))(a::Ptr{Float64}, lda::Int, x::Ptr{Float64},
            y::Ptr{Float64}, m::Int, n::Int, alpha::Float64) =
        Base.llvmcall(($(_SME_GER_IRS[g]), "entry"), Cvoid,
                      Tuple{Ptr{Float64}, Int64, Ptr{Float64}, Ptr{Float64}, Int64, Int64, Float64},
                      a, Int64(lda), x, y, Int64(m), Int64(n), alpha)
end

# A compile-time ladder over the detected set, generated from it so the two cannot drift.
@eval @inline function _sme_ger_run(g::Int, a, lda, x, y, m, n, alpha)
    $(Expr(:block, (quote
        if g == $gg
            return $(Symbol("_sme_ger_k", gg))(a, lda, x, y, m, n, alpha)
        end
    end for gg in _SME_GER_GS)...))
    return nothing
end

const _SME_GER_CALLS = Threads.Atomic{Int}(0)
const _SME_GER_TRAMPOLINE = Ref{Any}(nothing)
const _SME_GER_ENTRY      = Ref{Ptr{Cvoid}}(C_NULL)

function _sme_ger_cabi(a::Ptr{Float64}, lda::Int, x::Ptr{Float64}, y::Ptr{Float64},
                       m::Int, n::Int, alpha::Float64)
    g = 1
    while 2 * g <= n && 2 * g <= _SME_GER_G
        g *= 2
    end
    _sme_ger_run(g, a, lda, x, y, m, n, alpha)
    return nothing
end

# ⛔ WHOLE ZA BLOCKS OF ROWS ONLY, AND THE REMAINDER IS AN OPEN PROBLEM.
#
# Rows below a whole block have to be finished somehow, and every way of doing it separately costs
# ONE PASS PER COLUMN, which is fatal here: at n=100 a 4-row remainder is 4% of the elements and
# roughly half the run time, because n small per-column calls at a few ns each rival the whole
# operation (Accelerate does the entire n=100 update in 0.57 us). Measured, m=100 against m=96:
# 83 GB/s with a NEON per-column remainder, 62 with a scalar one, 475 with none.
#
# Fusing it into the column body instead needs a PARTIAL `whilelt` predicate, and those are what
# collapse: with the remainder a whole number of vectors the kernel is fine (m=120, a 24-row
# remainder, 335 GB/s), with a partial one it is not (m=100 62, m=97 49, m=129 85).
#
# So the route is taken only where every predicate it issues is FULL, which means `m` a whole number
# of vectors. Rows below a whole ZA block are finished by the kernel's own vector loop, and that
# costs little when there are few of them (m=1000 leaves exactly one vector), which is why the
# condition is `_SME_L` and not the block. n=50 and n=100 are not whole vectors and stay on the SIMD
# path at 0.98 and 0.64.
# PDM: Derived — formula over detected consts: the kernel's own row granularity, `4 * _SME_L`, below and outside of which it has nothing to run.
const _SME_GER_MINM = @load_preference("sme_ger_minm", 4 * _SME_L)::Int

# ⚠ `m % _SME_L == 0` IS WHY ger's WORST CELL IS WORST, AND SPLITTING THE ROWS DOES NOT FIX IT.
# The predicate below sends every m that is not a whole number of vectors to the NEON path, and the
# two rates are far apart. Measured square, Float64, alpha tiny so A stays stable, no copy in the
# timed body, traffic counted as one read plus one write of A:
#
#     n          96   100   104   120   128   132   160   200   256   300
#     n % 8       0     4     0     0     0     4     0     0     0     4
#     GB/s      385   184   336   298   426   150   444   366   458   140
#     path      SME  NEON   SME   SME   SME  NEON   SME   SME   SME  NEON
#
# For scale, `scal` on the same read-modify-write traffic reaches 262-264 GB/s at 72-156 KB, so the
# SME arm is above that roofline and the NEON arm is well under it. ger@100 gates 0.638 while its
# neighbours at 128 and 256 gate 0.961 and 0.942 — the cell is not small-n overhead, it is this cut.
#
# ⛔ DO NOT FIX IT BY SPLITTING THE ROWS. Running the kernel over `(m ÷ L) * L` rows and finishing the
# scrap with a separate `_ger_simd!` measures WORSE than NEON on the whole matrix at every size that
# needs it: n=100 0.45x, 132 0.60x, 156 0.69x, 300 0.66x, 500 0.65x. The scrap pass is short per
# column, and the truncated call still carries the FULL matrix's leading dimension — which at exactly
# the sizes this route exists to serve is not a multiple of `_SME_L`, and that is the cliff below.
#
# ⚠ `stride(A, 2) % _SME_L == 0` IS A SEPARATE REQUIREMENT FROM `m % _SME_L == 0`, and it is only the
# same thing by accident. A plain `Matrix` has `lda == m`, so the row test covers the stride test and
# the second one looks redundant — until the operand is a VIEW. Scanned at m=n=96 against lda=96,
# every lda from 96 to 200:
#
#     lda % _SME_L == 0    1.00x        (96, 104, 112, 120, 128, 160, 176, 184, 192)
#     otherwise            2.4x - 3.1x  (91 of the 105 values tested)
#
# The kernel loads and STORES whole 512-bit vectors down each column, so a column stride that is not a
# whole number of them puts every column's writes at a different offset inside the line. Reads barely
# notice — the same scan through `_sme_gemv!` costs 4-16%, and through `_sme_gemvt!` 13-18% — but ger
# writes A, and that is the whole difference.
#
# It was reachable: `ger!` on `view(B, 1:96, 1:96)` with `B` 100x100 took the SME path at 153 GB/s
# where the plain 96x96 matrix runs 388, and at lda=129 it fell to 123, BELOW the NEON path it
# displaced. With the stride test in place those route to NEON: 963 -> 812 ns and 1196 -> 833.
#
# ⛔ AND THE KERNEL'S OWN PREDICATED SCRAP IS NOT THE ANSWER EITHER — it already exists (`mscrap0`..
# `mscrapG` below), it is CORRECT at every m, and it is slower than NEON everywhere it would be used:
#
#     n        100   101   132   156   255   300   500  1001
#     SME/NEON 0.56  0.45  0.63  0.43  0.44  0.55  0.56  0.66
#
# so relaxing the predicate is not a one-line win. The scrap runs once per COLUMN GROUP: at m=100 it
# adds 555 ns over the same call truncated to 96 rows, for 400 elements — 1.4 ns per element against
# 0.02 on the whole-vector path, about 7 cycles per predicated operation. That is the per-column
# remainder cost this file documents for gemv-T and `_sme_ger_ir`, and it is architectural.
#
# ⛔ AN OVERLAPPING FINAL VECTOR CANNOT RESCUE IT, unlike the N-form gemv store path. ger ACCUMULATES
# into A, so a last vector placed at `m - L` re-applies the update to the `L - m % L` rows it overlaps.
# Overlap is only sound when the kernel STORES.
@inline _sme_ger_eligible(::Type{T}, m, n, cj, A, x, y, incx, incy) where {T} =
    T === Float64 && _SME_F64 && !cj && incx == 1 && incy == 1 &&
        eltype(x) === Float64 && eltype(y) === Float64 &&
        _strided1(A) && _dense1(x) && _dense1(y) &&
        m >= _SME_GER_MINM && m % _SME_L == 0 && stride(A, 2) % _SME_L == 0 &&
        n > 0 && _SME_GER_ENTRY[] !== C_NULL

@noinline function _sme_ger!(m::Int, n::Int, alpha::Float64, x, y, A)
    Threads.atomic_add!(_SME_GER_CALLS, 1)
    lda = stride(A, 2)
    GC.@preserve A x y begin
        ccall(
            _SME_GER_ENTRY[], Cvoid,
            (Ptr{Float64}, Int, Ptr{Float64}, Ptr{Float64}, Int, Int, Float64),
            _sme_p(A), lda, _sme_p(x), _sme_p(y), m, n, alpha
        )
    end
    return A
end

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

# The self-test's failure text, built at PRECOMPILE from consts rather than interpolated where it is
# thrown. Every value in it is a compile-time constant, so there is nothing to defer; and `__init__`
# is in the trim graph, where an interpolated `error("...$x...")` lowers to `print_to_string` over a
# `Vararg{Any}` tuple and `--trim=safe` rejects it as an unresolved call. It lives at the END of the
# file because a `const` is evaluated where it is written, and the gemv geometry below `_sme_init!`
# would not exist yet at that point; `_sme_init!` only reads it when `__init__` runs, by which time
# the module is fully loaded.
const _SME_SELFTEST_MSG = """
    SME self-test failed against a scalar reference. Call `PureBLAS._sme_selftest()` for the
    measured relative error. The kernels built and ran, so this is a geometry assumption that
    does not hold on this machine -- lanes=$(_SME_LANES), L=$(_SME_L), MR=$(_SME_MR), NR=$(_SME_NR), gemv blocks up
    to $(_SME_GEMV_BLK * _SME_GEMV_NGMAX) rows. Please report it.
    Set `sme_f64 = false` in LocalPreferences.toml to keep the SIMD path meanwhile."""

# ⛔ THE BARRIER COSTS ~50 ns PER CALL AND CANNOT BE SKIPPED BY TESTING THE `-C` TARGET.
#
# Measured on `gemv`, the same kernel reached both ways: 607 against 794 GB/s at n=128, 845 against
# 918 at n=256, 1007 against 1032 at n=512 — a flat ~50 ns, which is 30% of the call at n=128 and is
# what holds gemv's small-n cells under the gate. It is not the call itself (an empty Julia cfunction
# is 3.3 ns, and an empty streaming body through one costs 7.6 ns more than reached directly), so it
# is the boundary into a kernel that holds ZA.
#
# The obvious escape does not work. `Base.JLOptions().cpu_target == C_NULL` looks like it means "this
# host", which would let an ordinary session call the kernel directly and leave the barrier to
# juliac's stripped pass. It does not: an ordinary `Pkg.precompile()` ALSO compiles for a generic
# image CPU, and routing `_sme_gemv!` straight at `_sme_gemv_cabi` under that test aborts the package
# build with `Cannot select: intrinsic llvm.aarch64.sve.ptrue.c64`. Tried 2026-09-30.

# ══ TRAMPOLINE CONSTRUCTION — TWO FORMS, CHOSEN BY THE COMPILING PROCESS'S OWN TARGET ═══════════
#
# The barrier below exists because AOT codegen OVERRIDES a function's own `target-features` with the
# process `-C` target, while the JIT leaves them alone. So the kernel's `+sme` survives in a normal
# session and is stripped in an ahead-of-time one, where `rdsvl` then has no pattern and the build
# aborts. `inferencebarrier` hides the callee so the kernel is never reached at all.
#
# That barrier is also why `--trim=safe` rejects the build: `Compiler/src/verifytrim.jl` resolves a
# `:cfunction` purely from the INFERRED TYPE of its target, and the barrier is what erases it. The
# two requirements are the same property seen twice, and no annotation squares them.
#
# What squares them is noticing the barrier is only needed when the target LACKS `+sme`. juliac runs
# two processes: `Pkg.precompile()` with the features stripped, then `--output-o` with the full
# target and `--output-incremental=no`, which makes it evaluate this source fresh rather than load
# the first one's image. So each process can take the form it needs.
#
# Measured on an M6: with `-C apple-m1,+sme,+sme2,+sme-f64f64` the constant form compiles the kernel
# INTO the image and the built dylib carries the `_jlcapi_` adapters, 20 `fmopa` and 36
# `smstart`/`smstop` pairs; with the default target the same source aborts on
# `Cannot select: intrinsic llvm.aarch64.sve.ptrue.c64`.
#
# These live after `_sme_gemv_cabi` because the CONSTANT form resolves its callee when the enclosing
# method is DEFINED, not when it runs — placed earlier they are an `UndefVarError` at load.
const _SME_STATIC = _SME_F64 && let p = Base.JLOptions().cpu_target   # C_NULL / "native" when no -C
    p != C_NULL && occursin("+sme", unsafe_string(p))
end

@static if !_SME_F64
    # OFF SME HARDWARE THESE MUST NOT EXIST IN ANY FORM. `_sme_init!` returns at its first line
    # there, so they are unreachable — but a top-level definition is still INFERRED, and inferring
    # `inferencebarrier(_sme_entry_cabi)` + `@cfunction` drags the cabi entry and `_gemm_core!`
    # behind it. That is not hypothetical: hoisting these out of `_sme_init!`'s dead body, where
    # they had been compiled away, is what made two StrictMode dogfood items report
    # `OptimizationFailureReport in PureBLAS._gemm_core!` on x86.
    _sme_entry_cf() = throw(AssertionError("SME trampoline requested without SME"))
    _sme_gemv_cf() = throw(AssertionError("SME trampoline requested without SME"))
    _sme_gemvt_cf() = throw(AssertionError("SME trampoline requested without SME"))
    _sme_ger_cf() = throw(AssertionError("SME trampoline requested without SME"))
    _sme_dot_cf() = throw(AssertionError("SME trampoline requested without SME"))
    _sme_asum_cf() = throw(AssertionError("SME trampoline requested without SME"))
    _sme_axpy_cf() = throw(AssertionError("SME trampoline requested without SME"))
    _sme_scal_cf() = throw(AssertionError("SME trampoline requested without SME"))
elseif _SME_STATIC
    _sme_entry_cf() = @cfunction(_sme_entry_cabi, Cvoid,
        (Ptr{Float64}, Int, Ptr{Float64}, Int, Ptr{Float64}, Int,
         Int, Int, Int, Float64, Float64, Int, Int,
         Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Int, Int, Int))
    _sme_gemv_cf() = @cfunction(_sme_gemv_cabi, Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Int, Ptr{Float64}, Int, Int, Float64, Int))
    _sme_gemvt_cf() = @cfunction(_sme_gemvt_cabi, Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Int, Ptr{Float64}, Int, Int, Float64))
    _sme_ger_cf() = @cfunction(_sme_ger_cabi, Cvoid,
        (Ptr{Float64}, Int, Ptr{Float64}, Ptr{Float64}, Int, Int, Float64))
    _sme_dot_cf() = @cfunction(_sme_dot_cabi, Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Int))
    _sme_asum_cf() = @cfunction(_sme_asum_cabi, Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Int))
    _sme_axpy_cf() = @cfunction(_sme_axpy_cabi, Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Float64, Int))
    _sme_scal_cf() = @cfunction(_sme_scal_cabi, Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Float64, Int))
else
    # `getfield(@__MODULE__, :name)` is NOT opaque — module and symbol are both constants, so
    # inference folds it back to the concrete function and walks into the kernel anyway.
    # `inferencebarrier` forces the value to `Any`, which is what actually stops it.
    function _sme_entry_cf()
        f = Base.inferencebarrier(_sme_entry_cabi)
        return @cfunction($f, Cvoid,
            (Ptr{Float64}, Int, Ptr{Float64}, Int, Ptr{Float64}, Int,
             Int, Int, Int, Float64, Float64, Int, Int,
             Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Int, Int, Int))
    end
    function _sme_gemv_cf()
        g = Base.inferencebarrier(_sme_gemv_cabi)
        return @cfunction($g, Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Int, Ptr{Float64}, Int, Int, Float64, Int))
    end
    function _sme_gemvt_cf()
        h = Base.inferencebarrier(_sme_gemvt_cabi)
        return @cfunction($h, Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Int, Ptr{Float64}, Int, Int, Float64))
    end
    function _sme_ger_cf()
        gr = Base.inferencebarrier(_sme_ger_cabi)
        return @cfunction($gr, Cvoid,
            (Ptr{Float64}, Int, Ptr{Float64}, Ptr{Float64}, Int, Int, Float64))
    end
    function _sme_dot_cf()
        d = Base.inferencebarrier(_sme_dot_cabi)
        return @cfunction($d, Cvoid, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Int))
    end
    function _sme_asum_cf()
        e = Base.inferencebarrier(_sme_asum_cabi)
        return @cfunction($e, Cvoid, (Ptr{Float64}, Ptr{Float64}, Int))
    end
    function _sme_axpy_cf()
        p = Base.inferencebarrier(_sme_axpy_cabi)
        return @cfunction($p, Cvoid, (Ptr{Float64}, Ptr{Float64}, Float64, Int))
    end
    function _sme_scal_cf()
        r = Base.inferencebarrier(_sme_scal_cabi)
        return @cfunction($r, Cvoid, (Ptr{Float64}, Ptr{Float64}, Float64, Int))
    end
end
