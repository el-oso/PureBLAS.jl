# ══ BLAS-1 REDUCTIONS ON SME ════════════════════════════════════════════════════════════════════
#
# THE WALL IS THE ACCUMULATOR, NOT THE LOADS. A `dot` or `asum` written any ordinary way tops out
# near 105-150 GB/s on this part however it is arranged, and that ceiling does not move with the
# cache hierarchy — the same rate whether the operand is 2 MB or 320 MB. Measured here, all
# falsified: concurrent stream count (plateaus at two chunks), software prefetch (0.5x — the
# hardware streamer already does it), 512-bit streaming vectors (0.6x on a read), NEON multi-register
# `ld1x4` (1.00x), and extra accumulator chains on `dot` (0.99x, `@simd` already had them).
#
# `apple-sme-gemv-za-accumulate` names the cause: every `fmla` into the SAME ZA slice is one serial
# dependency chain, and moving the accumulation out of z registers into ZA took the same traffic from
# 256.7 GB/s to 1013. That is what these kernels do. The range is split across TWO ZA groups so the
# two chains are independent; `dot` then reads
#
#     n          1024   2048   4096   16384  65536  249984  976000
#     shipped   175.5  169.4  167.2   152.2  151.4   139.4   134.5   GB/s
#     ZA        210.1  305.3  564.6   809.6  924.7   989.2   998.1
#
# which is 7.4x at the top of the ladder, and 97% of what Accelerate reaches on the same operand.
# Past L2 both converge on the memory rate, as they must.
#
# GROUP COUNT IS THE DIMENSION THAT MATTERS, not load width: one group reads 566 GB/s, two 1000,
# four 973, eight 628. Two is the peak and the shape matches the gemv finding's.
#
# The kernels are reached through a function POINTER for the reason `sme_kernel.jl` documents at
# length: a function holding ZA state needs `rdsvl` to size its frame, which a generic-CPU package
# image cannot select, and inference walks through a call boundary even when execution does not.

# Elements one ZA group consumes per step: four 512-bit vectors.
const _SME_L1_BLK = 4 * _SME_L
# Independent ZA groups the range is split across. PDM: Measured — every fmla into one slice is a
# serial chain, so this is a dependency-depth choice, not a residency or width one; the optimum
# inverts either side of two (566 / 1000 / 973 / 628 GB/s at one / two / four / eight).
# | tune: candidate, (1,2,4,8)
# PDM: Measured — every fmla into one ZA slice is a serial chain, so this is dependency depth, not residency or width; the optimum inverts either side of two. | tune: candidate, (1,2,4,8)
const _SME_L1_G = @load_preference("sme_l1_groups", 2)::Int   # req8-ok: swept, see the table above

function _sme_l1_ir(G::Int, kind::Symbol)
    dotform = kind === :dot
    q = Char(34)
    T4 = "{ <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double> }"
    args = dotform ? "ptr %o, ptr %x, ptr %y, i64 %c" : "ptr %o, ptr %x, i64 %c"
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
declare void @llvm.aarch64.sme.add.za64.vg1x4.nxv2f64(i32,
  <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>)
declare $T4 @llvm.aarch64.sme.read.vg1x4.nxv2f64(i32)
declare double @llvm.vector.reduce.fadd.nxv2f64(double, <vscale x 2 x double>)
declare <vscale x 2 x double> @llvm.fabs.nxv2f64(<vscale x 2 x double>)

define void @entry($args) {
  call void @k($args)
  ret void
}
define internal void @k($args) #0 {
entry:
  call void @llvm.aarch64.sme.za.enable()
  call void @llvm.aarch64.sme.zero(i32 255)
  %pn = call target($(q)aarch64.svcount$(q)) @llvm.aarch64.sve.ptrue.c64()
  %oe = insertelement <vscale x 2 x double> poison, double 1.0, i32 0
  %one = shufflevector <vscale x 2 x double> %oe, <vscale x 2 x double> poison, <vscale x 2 x i32> zeroinitializer
  %go = icmp sgt i64 %c, 0
  br i1 %go, label %loop, label %rd
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
""")
    for j in 0:(G - 1)
        println(io, "  %ox$(j) = mul nsw i64 %c, $j")
        println(io, "  %qx$(j) = add nsw i64 %i, %ox$(j)")
        println(io, "  %px$(j) = getelementptr inbounds double, ptr %x, i64 %qx$(j)")
        println(io, "  %rx$(j) = call $T4 @llvm.aarch64.sve.ld1.pn.x4.nxv2f64(target($(q)aarch64.svcount$(q)) %pn, ptr %px$(j))")
        for k in 0:3
            println(io, "  %xv$(j)_$(k) = extractvalue $T4 %rx$(j), $k")
        end
        if dotform
            println(io, "  %py$(j) = getelementptr inbounds double, ptr %y, i64 %qx$(j)")
            println(io, "  %ry$(j) = call $T4 @llvm.aarch64.sve.ld1.pn.x4.nxv2f64(target($(q)aarch64.svcount$(q)) %pn, ptr %py$(j))")
            for k in 0:3
                println(io, "  %yv$(j)_$(k) = extractvalue $T4 %ry$(j), $k")
            end
        else
            for k in 0:3
                # FALSIFIED 2026-09-28: replacing this with a bitwise sign-bit clear (bitcast, AND with
            # 0x7fff..., bitcast back) measured WORSE — gate 0.847 against 0.877 — so LLVM already
            # lowers `fabs` to the right thing and the FP pipeline is not the contention. Removing
            # the operation ENTIRELY runs 2.44x faster at n=100000, which prices the load-to-ZA
            # dependency, not the absolute value. Do not retry the bitwise form.
            println(io, "  %av$(j)_$(k) = call <vscale x 2 x double> @llvm.fabs.nxv2f64(<vscale x 2 x double> %xv$(j)_$(k))")
            end
        end
        if dotform
            println(io, "  call void @llvm.aarch64.sme.fmla.vg1x4.nxv2f64(i32 $j,")
            println(io, "    <vscale x 2 x double> %xv$(j)_0, <vscale x 2 x double> %xv$(j)_1,")
            println(io, "    <vscale x 2 x double> %xv$(j)_2, <vscale x 2 x double> %xv$(j)_3,")
            println(io, "    <vscale x 2 x double> %yv$(j)_0, <vscale x 2 x double> %yv$(j)_1,")
            println(io, "    <vscale x 2 x double> %yv$(j)_2, <vscale x 2 x double> %yv$(j)_3)")
        else
            # Plain add-to-ZA: no ones vector and no multiply, which is four fewer operand
            # registers and one less op per step than the fmla form this replaced.
            println(io, "  call void @llvm.aarch64.sme.add.za64.vg1x4.nxv2f64(i32 $j,")
            println(io, "    <vscale x 2 x double> %av$(j)_0, <vscale x 2 x double> %av$(j)_1,")
            println(io, "    <vscale x 2 x double> %av$(j)_2, <vscale x 2 x double> %av$(j)_3)")
        end
    end
    print(io, """
  %in = add nuw nsw i64 %i, $(4 * _SME_L)
  %d = icmp sge i64 %in, %c
  br i1 %d, label %rd, label %loop
rd:
""")
    parts = String[]
    for j in 0:(G - 1)
        println(io, "  %g$(j) = call $T4 @llvm.aarch64.sme.read.vg1x4.nxv2f64(i32 $j)")
        for k in 0:3
            println(io, "  %z$(j)_$(k) = extractvalue $T4 %g$(j), $k")
            push!(parts, "%z$(j)_$(k)")
        end
    end
    prev = parts[1]
    for (i, p) in enumerate(parts[2:end])
        println(io, "  %s$(i) = fadd reassoc <vscale x 2 x double> $prev, $p")
        prev = "%s$(i)"
    end
    println(io, "  %tot = call reassoc double @llvm.vector.reduce.fadd.nxv2f64(double 0.0, <vscale x 2 x double> $prev)")
    print(io, """
  store double %tot, ptr %o, align 8
  call void @llvm.aarch64.sme.za.disable()
  ret void
}
""")
    print(io, _SME_ATTRS)
    return String(take!(io))
end

const _SME_DOT_IR  = _sme_l1_ir(_SME_L1_G, :dot)
const _SME_ASUM_IR = _sme_l1_ir(_SME_L1_G, :asum)

@inline _sme_dot_k(o::Ptr{Float64}, x::Ptr{Float64}, y::Ptr{Float64}, c::Int) =
    Base.llvmcall((_SME_DOT_IR, "entry"), Cvoid,
                  Tuple{Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Int64}, o, x, y, Int64(c))
@inline _sme_asum_k(o::Ptr{Float64}, x::Ptr{Float64}, c::Int) =
    Base.llvmcall((_SME_ASUM_IR, "entry"), Cvoid,
                  Tuple{Ptr{Float64}, Ptr{Float64}, Int64}, o, x, Int64(c))

const _SME_DOT_CALLS  = Threads.Atomic{Int}(0)
const _SME_ASUM_CALLS = Threads.Atomic{Int}(0)

# THE FLOOR IS A COLD CROSSING, NOT A WARM ONE, and that distinction is the whole of it. Warm, the
# ZA prologue amortizes over repeated passes and the route turns at n = 1024; cold it is paid once,
# on a single pass, which is what BLAS-1 is graded on. Measured cold, ZA against the kernel it
# displaces (`bench/probes/l1_za_cold_floor.jl`):
#
#     n         4096   8192  16384  32768  65536  131072  262144  976000
#     dot       0.36   0.70   0.84   0.92   1.75    3.23    3.28    1.41
#     asum      0.21   0.33   0.59   0.66   0.77    1.22    1.46    1.43
#
# so dot turns near 65536 and asum near 131072 — 32x and 16x above where the warm crossings sit.
# Floors placed one step above each, because a floor set from the wrong regime is what put dot
# at 0.842 of OpenBLAS on the first attempt.
# PDM: Measured — where a fixed ZA prologue disappears into the stream; a ratio between two kernels, not a residency criterion. | tune: sweep n
# PDM: Measured — where a fixed ZA prologue disappears into the stream; a ratio between two kernels, not a residency criterion. | tune: sweep n
const _SME_DOT_MIN  = @load_preference("sme_dot_min", 65536)::Int   # req8-ok: measured crossover, table above
# PDM: Measured — the same crossing for the one-stream form, which turns later because half the outstanding requests. | tune: sweep n
const _SME_ASUM_MIN = @load_preference("sme_asum_min", 65536)::Int  # req8-ok: measured crossover, table above

const _SME_DOT_TRAMPOLINE  = Ref{Any}(nothing)
const _SME_DOT_ENTRY       = Ref{Ptr{Cvoid}}(C_NULL)
const _SME_ASUM_TRAMPOLINE = Ref{Any}(nothing)
const _SME_ASUM_ENTRY      = Ref{Ptr{Cvoid}}(C_NULL)

# `c` is the per-group chunk length; the kernel covers `c * G` elements and the caller finishes the
# remainder. Both are whole multiples of the block, so no partial vector is ever loaded.
function _sme_dot_cabi(o::Ptr{Float64}, x::Ptr{Float64}, y::Ptr{Float64}, c::Int)
    _sme_dot_k(o, x, y, c)
    return nothing
end
function _sme_asum_cabi(o::Ptr{Float64}, x::Ptr{Float64}, c::Int)
    _sme_asum_k(o, x, c)
    return nothing
end

@inline _sme_l1_chunk(n::Int) = ((n ÷ _SME_L1_G) ÷ _SME_L1_BLK) * _SME_L1_BLK

@inline _sme_dot_ok(::Type{T}, n::Int, x, y) where {T} =
    T === Float64 && _SME_F64 && n >= _SME_DOT_MIN &&
        _dense1(x) && _dense1(y) && _SME_DOT_ENTRY[] !== C_NULL
@inline _sme_asum_ok(::Type{T}, n::Int, x) where {T} =
    T === Float64 && _SME_F64 && n >= _SME_ASUM_MIN &&
        _dense1(x) && _SME_ASUM_ENTRY[] !== C_NULL

@noinline function _sme_dot(n::Int, x, y)
    Threads.atomic_add!(_SME_DOT_CALLS, 1)
    c = _sme_l1_chunk(n)
    m = c * _SME_L1_G
    o = Ref(0.0)
    GC.@preserve x y o begin
        po = Base.unsafe_convert(Ptr{Float64}, o)
        ccall(_SME_DOT_ENTRY[], Cvoid, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Int),
              po, _sme_p(x), _sme_p(y), c)
    end
    s = o[]
    px = _sme_p(x); py = _sme_p(y)
    GC.@preserve x y begin
        for i in m:(n - 1)
            s = muladd(unsafe_load(px + i * 8), unsafe_load(py + i * 8), s)
        end
    end
    return s
end

@noinline function _sme_asum(n::Int, x)
    Threads.atomic_add!(_SME_ASUM_CALLS, 1)
    c = _sme_l1_chunk(n)
    m = c * _SME_L1_G
    o = Ref(0.0)
    GC.@preserve x o begin
        po = Base.unsafe_convert(Ptr{Float64}, o)
        ccall(_SME_ASUM_ENTRY[], Cvoid, (Ptr{Float64}, Ptr{Float64}, Int), po, _sme_p(x), c)
    end
    s = o[]
    px = _sme_p(x)
    GC.@preserve x begin
        for i in m:(n - 1)
            s += abs(unsafe_load(px + i * 8))
        end
    end
    return s
end

# ── scal and axpy: ZA AS AN ARITHMETIC UNIT, NOT AS AN ACCUMULATOR ──────────────────────────────
#
# `x .*= a` and `y .+= a.*x` accumulate nothing, so the reduction argument above does not apply to
# them — and they are still capped by the same wall. Measured on an 8 MB resident buffer, a
# streaming-mode COPY of it runs 404 GB/s and adding ONE `fmul` per vector in z registers drops it
# to 138 (`bench/probes/rmw_ceiling.jl`). The 2.9x is arithmetic that moves no extra bytes: the wall
# is z-register FP throughput in streaming mode, which is what `apple-sme-gemv-za-accumulate` says
# it is. There is no multi-vector `fmul` in this LLVM, so ZA is the only way to get the multiply out
# of z registers, and it works on a non-accumulating kernel too:
#
#     `zero {za}` the group, `fmla.single` lands `a*x` in it, read the group back, store it.
#
#     n            1000  3000  10000  30000  100000  300000  1000000
#     scal NEON     288   272    275    164     167     168      167   GB/s
#     scal ZA       236   509    547    558     553     550      563
#     Accelerate    496   541    554    559     549     543      556
#     axpy NEON     294   294    171    173     213     214      213
#     axpy ZA       344   622    662    668     669     727      303
#
# which is where the floors below come from. `axpy` seeds the cleared
# group with `y` at weight one and then accumulates `a*x` into it, so both operands arrive through
# the same fused multiply-add the NEON kernel uses and the result is bit-for-bit `muladd(a, x, y)`.
#
# ⛔ THE KERNEL COVERS THE WHOLE VECTOR, DOWN TO THE LAST ELEMENT, and that is a performance
# requirement, not tidiness. Ordinary floating-point work in a function that also calls a
# streaming-mode kernel costs ~50 ns whichever side of the call it sits on: a 16-element remainder
# loop is 50 ns of stall for 2.8 ns of work, which at n=10000 is the whole difference between 1.01x
# and 0.81x of Accelerate. A remainder loop that runs ZERO iterations costs nothing, so the penalty
# is the mixing, not the branch. Hence three phases inside ONE streaming
# region: G groups over the blocks they cover, a single-group loop over the blocks past that, and a
# `whilelt`-predicated loop over the scrap below one block.
#
# GROUP COUNT is not the same optimum as `dot`'s, because each group here carries a load AND a
# store, so groups buy independent memory streams rather than only independent dependency chains:
# scal reads 477 / 563 / 550 GB/s at two / four / eight groups, axpy 561 / 727 / 671.
# PDM: Measured — each group carries its own load and store, so this is stream count and dependency depth together, not residency or width. | tune: candidate, (1,2,4,8)
const _SME_RMW_G = @load_preference("sme_rmw_groups", 4)::Int   # req8-ok: swept, see the rates above

# `kind` is `:scal` (`y = a*x`, called in place) or `:axpy` (`y = y + a*x`).
function _sme_rmw_ir(G::Int, kind::Symbol)
    axpy = kind === :axpy
    q = Char(34)
    SC = "target($(q)aarch64.svcount$(q))"
    T4 = "{ <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double> }"
    io = IOBuffer()
    # One whole block through ZA group `g`, at the block index named by `bexpr`.
    blk = function (g, bexpr, sfx)
        println(io, "  %o$sfx = mul nsw i64 $bexpr, %w")
        println(io, "  %px$sfx = getelementptr inbounds double, ptr %x, i64 %o$sfx")
        println(io, "  %py$sfx = getelementptr inbounds double, ptr %y, i64 %o$sfx")
        if axpy
            println(io, "  %ry$sfx = call $T4 @llvm.aarch64.sve.ld1.pn.x4.nxv2f64($SC %pn, ptr %py$sfx)")
            for k in 0:3
                println(io, "  %yv$(sfx)_$k = extractvalue $T4 %ry$sfx, $k")
            end
            println(io, "  call void @llvm.aarch64.sme.fmla.single.vg1x4.nxv2f64(i32 $g,")
            println(io, "    <vscale x 2 x double> %yv$(sfx)_0, <vscale x 2 x double> %yv$(sfx)_1,")
            println(io, "    <vscale x 2 x double> %yv$(sfx)_2, <vscale x 2 x double> %yv$(sfx)_3, <vscale x 2 x double> %one)")
        end
        println(io, "  %r$sfx = call $T4 @llvm.aarch64.sve.ld1.pn.x4.nxv2f64($SC %pn, ptr %px$sfx)")
        for k in 0:3
            println(io, "  %v$(sfx)_$k = extractvalue $T4 %r$sfx, $k")
        end
        println(io, "  call void @llvm.aarch64.sme.fmla.single.vg1x4.nxv2f64(i32 $g,")
        println(io, "    <vscale x 2 x double> %v$(sfx)_0, <vscale x 2 x double> %v$(sfx)_1,")
        println(io, "    <vscale x 2 x double> %v$(sfx)_2, <vscale x 2 x double> %v$(sfx)_3, <vscale x 2 x double> %av)")
        println(io, "  %z$sfx = call $T4 @llvm.aarch64.sme.read.vg1x4.nxv2f64(i32 $g)")
        for k in 0:3
            println(io, "  %q$(sfx)_$k = extractvalue $T4 %z$sfx, $k")
        end
        println(io, "  call void @llvm.aarch64.sve.st1.pn.x4.nxv2f64(<vscale x 2 x double> %q$(sfx)_0, <vscale x 2 x double> %q$(sfx)_1, <vscale x 2 x double> %q$(sfx)_2, <vscale x 2 x double> %q$(sfx)_3, $SC %pn, ptr %py$sfx)")
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

define void @entry(ptr %y, ptr %x, double %a, i64 %n) {
  call void @k(ptr %y, ptr %x, double %a, i64 %n)
  ret void
}
define internal void @k(ptr %y, ptr %x, double %a, i64 %n) #0 {
entry:
  call void @llvm.aarch64.sme.za.enable()
  %pn = call $SC @llvm.aarch64.sve.ptrue.c64()
  %vs = call i64 @llvm.vscale.i64()
  %lanes = shl i64 %vs, 1
  %w = shl i64 %vs, 3
  %e0 = insertelement <vscale x 2 x double> poison, double %a, i32 0
  %av = shufflevector <vscale x 2 x double> %e0, <vscale x 2 x double> poison, <vscale x 2 x i32> zeroinitializer
  %e1 = insertelement <vscale x 2 x double> poison, double 1.0, i32 0
  %one = shufflevector <vscale x 2 x double> %e1, <vscale x 2 x double> poison, <vscale x 2 x i32> zeroinitializer
  %nb = sdiv i64 %n, %w
  %whole = mul nsw i64 %nb, %w
  %cb = sdiv i64 %nb, $G
  %cov = mul nsw i64 %cb, $G
  %go = icmp sgt i64 %cb, 0
  br i1 %go, label %loop, label %tailchk
loop:
  %b = phi i64 [ 0, %entry ], [ %bn, %loop ]
  call void @llvm.aarch64.sme.zero(i32 255)
""")
    for g in 0:(G - 1)
        println(io, "  %m$g = mul nsw i64 %cb, $g")
        println(io, "  %s$g = add nsw i64 %b, %m$g")
        blk(g, "%s$g", "g$g")
    end
    print(io, """
  %bn = add nuw nsw i64 %b, 1
  %ld = icmp eq i64 %bn, %cb
  br i1 %ld, label %tailchk, label %loop
tailchk:
  %hastail = icmp slt i64 %cov, %nb
  br i1 %hastail, label %tail, label %scrapchk
tail:
  %j = phi i64 [ %cov, %tailchk ], [ %jn, %tail ]
  call void @llvm.aarch64.sme.zero(i32 255)
""")
    blk(0, "%j", "t")
    print(io, """
  %jn = add nuw nsw i64 %j, 1
  %td = icmp eq i64 %jn, %nb
  br i1 %td, label %scrapchk, label %tail
scrapchk:
  %hasscrap = icmp slt i64 %whole, %n
  br i1 %hasscrap, label %scrap, label %done
scrap:
  %i = phi i64 [ %whole, %scrapchk ], [ %in, %scrap ]
  %pg = call <vscale x 2 x i1> @llvm.aarch64.sve.whilelt.nxv2i1.i64(i64 %i, i64 %n)
  %sx = getelementptr inbounds double, ptr %x, i64 %i
  %sy = getelementptr inbounds double, ptr %y, i64 %i
  %sv = call <vscale x 2 x double> @llvm.aarch64.sve.ld1.nxv2f64(<vscale x 2 x i1> %pg, ptr %sx)
""")
    if axpy
        print(io, """
  %syv = call <vscale x 2 x double> @llvm.aarch64.sve.ld1.nxv2f64(<vscale x 2 x i1> %pg, ptr %sy)
  %sr = call <vscale x 2 x double> @llvm.fma.nxv2f64(<vscale x 2 x double> %av, <vscale x 2 x double> %sv, <vscale x 2 x double> %syv)
""")
    else
        println(io, "  %sr = fmul <vscale x 2 x double> %sv, %av")
    end
    print(io, """
  call void @llvm.aarch64.sve.st1.nxv2f64(<vscale x 2 x double> %sr, <vscale x 2 x i1> %pg, ptr %sy)
  %in = add nsw i64 %i, %lanes
  %sd = icmp slt i64 %in, %n
  br i1 %sd, label %scrap, label %done
done:
  call void @llvm.aarch64.sme.za.disable()
  ret void
}
""")
    print(io, _SME_ATTRS)
    return String(take!(io))
end

const _SME_SCAL_IR = _sme_rmw_ir(_SME_RMW_G, :scal)
const _SME_AXPY_IR = _sme_rmw_ir(_SME_RMW_G, :axpy)

@inline _sme_scal_k(y::Ptr{Float64}, x::Ptr{Float64}, a::Float64, n::Int) =
    Base.llvmcall((_SME_SCAL_IR, "entry"), Cvoid,
                  Tuple{Ptr{Float64}, Ptr{Float64}, Float64, Int64}, y, x, a, Int64(n))
@inline _sme_axpy_k(y::Ptr{Float64}, x::Ptr{Float64}, a::Float64, n::Int) =
    Base.llvmcall((_SME_AXPY_IR, "entry"), Cvoid,
                  Tuple{Ptr{Float64}, Ptr{Float64}, Float64, Int64}, y, x, a, Int64(n))

const _SME_SCAL_CALLS = Threads.Atomic{Int}(0)
const _SME_AXPY_CALLS = Threads.Atomic{Int}(0)
const _SME_SCAL_TRAMPOLINE = Ref{Any}(nothing)
const _SME_SCAL_ENTRY      = Ref{Ptr{Cvoid}}(C_NULL)
const _SME_AXPY_TRAMPOLINE = Ref{Any}(nothing)
const _SME_AXPY_ENTRY      = Ref{Ptr{Cvoid}}(C_NULL)

function _sme_scal_cabi(y::Ptr{Float64}, x::Ptr{Float64}, a::Float64, n::Int)
    _sme_scal_k(y, x, a, n)
    return nothing
end
function _sme_axpy_cabi(y::Ptr{Float64}, x::Ptr{Float64}, a::Float64, n::Int)
    _sme_axpy_k(y, x, a, n)
    return nothing
end

# Where the ZA route overtakes the NEON kernel it displaces, measured in the gate's own L1 setup —
# `randn(n)` per sample, then `clamp(8_000_000/n, 30, 20000)` calls on that buffer
# (`bench/probes/l1_za_rmw_floor.jl`), ZA / shipped:
#
#     n       512   768  1024  1536  2048  3072  4096   8192  32768
#     scal   0.42  0.63  0.87  1.36  1.79  1.90  1.89   1.99   3.41
#     axpy   0.62  0.97  1.31  1.89  2.07  2.12  2.16   2.20   3.65
#
# so scal turns between 1024 and 1536 and axpy between 768 and 1024. Each floor is the first power
# of two on the winning side of its own crossing — they are different numbers because `scal` moves
# two streams to `axpy`'s three, so the same fixed cost buys less.
# PDM: Measured — where the ZA prologue and its group-clear disappear into the stream; a ratio between two kernels, not a residency criterion. | tune: sweep n
const _SME_SCAL_MIN = @load_preference("sme_scal_min", 2048)::Int   # req8-ok: measured crossover, table above
# PDM: Measured — the same crossing for the three-stream form, which turns earlier because each call carries more work per unit of fixed cost. | tune: sweep n
const _SME_AXPY_MIN = @load_preference("sme_axpy_min", 1024)::Int   # req8-ok: measured crossover, table above

@inline _sme_scal_ok(::Type{T}, n::Int, x) where {T} =
    T === Float64 && _SME_F64 && n >= _SME_SCAL_MIN &&
        _dense1(x) && _SME_SCAL_ENTRY[] !== C_NULL
@inline _sme_axpy_ok(::Type{T}, n::Int, x, y) where {T} =
    T === Float64 && _SME_F64 && n >= _SME_AXPY_MIN &&
        _dense1(x) && _dense1(y) && _SME_AXPY_ENTRY[] !== C_NULL

@noinline function _sme_scal!(n::Int, a::Float64, x)
    Threads.atomic_add!(_SME_SCAL_CALLS, 1)
    GC.@preserve x begin
        p = _sme_p(x)
        ccall(_SME_SCAL_ENTRY[], Cvoid, (Ptr{Float64}, Ptr{Float64}, Float64, Int), p, p, a, n)
    end
    return x
end

@noinline function _sme_axpy!(n::Int, a::Float64, x, y)
    Threads.atomic_add!(_SME_AXPY_CALLS, 1)
    GC.@preserve x y begin
        ccall(_SME_AXPY_ENTRY[], Cvoid, (Ptr{Float64}, Ptr{Float64}, Float64, Int),
              _sme_p(y), _sme_p(x), a, n)
    end
    return y
end
