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
# against in place (`bench/probes/sme_inplace_worstcase.jl`, _EXP9 A/B in one process):
#
#     A block / L1  0.25   0.56   1.00   1.56   2.25   3.06   4.00   5.06   6.25   9.00  16.00
#     packed/inplc  1.939  1.599  1.418  1.111  1.102  1.053  1.057  0.968  0.982  0.959  0.949
#
# so in place pays up to four L1-fuls and loses beyond five. With a FRIENDLY stride (the same sizes
# at lda+1) it wins everywhere measured, 1.14-1.91, and `sme_inplace_alias.jl` isolates that to the
# stride alone: at n=512 the ratio is 0.967 at lda=512 and 1.143 at lda=513, nothing else changed.
#
# WHY THE CUT IS RESIDENCY AND NOT AN ALIASING PREDICATE. `_alias_ld` exists for this class of
# problem, but its period is `_L1_WAY_D` = L1 ÷ associativity, and associativity comes from a CPUID
# leaf that does not exist on aarch64 — `_L1D_ASSOC` is the fallback 8 here, not a detected value. So
# `_alias_ld` is false for every stride that actually loses on this machine (lda = 256, 512, 768,
# 1024), and a set-conflict criterion cannot be founded on a number we are guessing. A residency cut
# holds for any stride; widening it to admit friendly strides is left open, and the 7-20% it gives up
# at blocks of 5-30 L1-fuls is recorded above rather than claimed.
# req8-ok: a coefficient over the DETECTED L1 with the falsifying table above, not a fitted magic
# number — the criterion is A-block residency and the coefficient is where it was measured to flip.
# PDM: Derived — A-block residency against the detected L1: an in-place walk stays as cheap as a contiguous stream while the block the kernel re-reads is a few L1-fuls, and the coefficient is where that was measured to flip under a worst-case stride. | tune: n/a, follows _L1_BYTES
const _SME_INPLACE_MAX =
    @load_preference("sme_inplace_max", 4 * (_L1_BYTES ÷ sizeof(Float64)))::Int

@inline function _sme_inplace_cap()
    ov = @inbounds _EXPINT[4]           # sweep override; 0 = the shipped cut
    return ov > 0 ? ov : _SME_INPLACE_MAX
end

@inline function _sme_inplace_a(mce::Int, kce::Int, alpha::Float64, tA::Bool)
    (!tA && alpha == 1.0 && !(@inbounds _EXPFLAG[_EXP9])) || return false
    mce % _SME_MR == 0 || return false
    return mce * kce <= _sme_inplace_cap()
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
            if tB
                _sme_pack_A!(Bp, B, ldb, jc, pc, nce, kce, kpad, bscale)
            elseif kce == kpad && nce == npad
                _sme_packb!(Bp, B + (pc + jc * ldb) * 8, ldb, kpad, npad ÷ _SME_L)
                bscale == 1.0 || _sme_scale_panel!(Bp, npad * kpad, bscale)
            else
                _sme_pack_B_edge!(Bp, B, ldb, pc, jc, kce, nce, kpad)
                bscale == 1.0 || _sme_scale_panel!(Bp, npad * kpad, bscale)
            end
            bjp = kpad * NR
            bks = NR
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
                    C, ldc, pa, Bp, Cs, ic, jc, mce, nce, mpad, npad, kce, kpad,
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
    # Both pointers are live now, so the kernels can be asked a question with a known answer.
    if _SME_ENTRY[] !== C_NULL && _SME_GEMV_ENTRY[] !== C_NULL
        err = try
            _sme_selftest()
        catch e
            _SME_ENTRY[] = C_NULL; _SME_GEMV_ENTRY[] = C_NULL
            rethrow(e)
        end
        if !(err < 1e-12)
            _SME_ENTRY[] = C_NULL; _SME_GEMV_ENTRY[] = C_NULL
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
@inline function _sme_tile_ok(m, n)
    min(m, n) >= 2 * _SME_MR || return false
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
# Every cell at or above 8192 elements wins and every loser is below it, and 8192 doubles is half of
# this machine's L1 — the panel has to be at least that before the fixed ZA cost disappears into it.
# DERIVING THIS ON ONE n WOULD HAVE BEEN WRONG: an m-only sweep at n=512 put the cut at 96 rows, and
# that admitted 35 shapes at n=64 that run as slow as 0.82 of the path they displaced.
# PDM: Derived — formula over detected consts: half of L1 in elements, `_L1_BYTES ÷ (2 * sizeof(Float64))`, the panel size at which the O(m) ZA fill and readback disappear into the stream.
const _SME_GEMV_MINWORK = @load_preference("sme_gemv_minwork",
    _L1_BYTES ÷ (2 * sizeof(Float64)))::Int

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
function _sme_gemv_cabi(
        y::Ptr{Float64}, a::Ptr{Float64}, lda::Int, x::Ptr{Float64},
        m::Int, n::Int, alpha::Float64, store::Int
    )
    st = store != 0
    ib = 0
    rb = 512
    # LARGEST BLOCK FIRST, HALVING. An earlier version capped this at half the rows because a lone
    # block seemed to have nothing to overlap against. What it was overlapping WAS the readback, and
    # once that became a direct store the rule was pure cost: removing it took m=512 from 0.96 to
    # 1.05 of Accelerate and m=768 from 0.99 to 1.04.
    while rb >= 32
        nb = (m - ib) ÷ rb
        if nb > 0
            _sme_gemv_run(rb, y + ib * 8, a + ib * 8, lda, x, n, alpha, nb, st)
            ib += nb * rb
        end
        rb >>= 1
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
elseif _SME_STATIC
    _sme_entry_cf() = @cfunction(_sme_entry_cabi, Cvoid,
        (Ptr{Float64}, Int, Ptr{Float64}, Int, Ptr{Float64}, Int,
         Int, Int, Int, Float64, Float64, Int, Int,
         Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Int, Int, Int))
    _sme_gemv_cf() = @cfunction(_sme_gemv_cabi, Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Int, Ptr{Float64}, Int, Int, Float64, Int))
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
end
