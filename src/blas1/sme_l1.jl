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
