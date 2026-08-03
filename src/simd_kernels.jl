# SIMD.jl fast paths for the bandwidth-bound Level-1 kernels, real (`Float32`/`Float64`),
# unit-stride, dense memory. Everything else falls back to the generic scalar loops in
# level1.jl. Pointer-based `vload`/`vstore` over `GC.@preserve`d buffers; masked scalar tail.

using SIMD: Vec, vload, vstore, vifelse, shufflevector

# Vec type for element `T` at the detected register width (folds to a concrete type — see cpuinfo.jl).
@inline _vec(::Type{T}) where {T} = Vec{_vwidth(T), T}

# Eligibility: unit-stride + dense + real. `Ptr` inputs come from the C ABI (already raw); dense
# arrays expose a pointer. Complex deliberately excluded (interleaved re/im SIMD is M2 work).
@inline _simd1(::Ptr{T}) where {T <: BlasReal} = true
@inline _simd1(::DenseArray{T}) where {T <: BlasReal} = true
@inline _simd1(@nospecialize(_)) = false
@inline _simd2(::Ptr{T}, ::Ptr{T}) where {T <: BlasReal} = true
@inline _simd2(::DenseArray{T}, ::DenseArray{T}) where {T <: BlasReal} = true
@inline _simd2(@nospecialize(_), @nospecialize(_)) = false

# Complex unit-stride dense/Ptr → the underlying interleaved [re im re im …] buffer IS a contiguous
# 2n-real array. For the two reductions that are grouping-invariant — nrm2 (Σ|xᵢ|² = Σ over 2n reals r² )
# and asum (dzasum = Σ|Re|+|Im| = Σ over 2n reals |r|) — a complex op reduces EXACTLY to the real SIMD
# kernel over that reinterpreted buffer. `_reptr` gives the real Ptr (caller GC.@preserves the array).
@inline _cplx_re(::Ptr{Complex{T}}) where {T <: BlasReal} = true
@inline _cplx_re(::DenseArray{Complex{T}}) where {T <: BlasReal} = true
@inline _cplx_re(@nospecialize(_)) = false
@inline _reptr(x::Ptr{Complex{T}}) where {T <: BlasReal} = Ptr{T}(x)
@inline _reptr(x::DenseArray{Complex{T}}) where {T <: BlasReal} = Ptr{T}(pointer(x))
const _CplxArg{T} = Union{Ptr{Complex{T}}, DenseArray{Complex{T}}}
@inline _cplx2(::_CplxArg{T}, ::_CplxArg{T}) where {T <: BlasReal} = true      # both complex, same real T
@inline _cplx2(@nospecialize(_), @nospecialize(_)) = false

@inline _ptr(p::Ptr) = p
@inline _ptr(a) = pointer(a)

# Elementwise kernels are 4-way unrolled (4 vectors / iteration) to keep load/store ports busy and
# give the prefetcher a longer stride — a single vector/iteration is throughput-starved in the
# L2-resident regime. `_UNROLL` is defined with the reductions below. Pattern: unrolled body, then
# a W-at-a-time pass, then a scalar tail.
# `pf` = software-prefetch distance (elements ahead) for the OUTPUT stream `y`. Default 0 → the whole
# prefetch block const-folds away, so the L1 `axpy` path (and every other 4-arg caller) is byte-identical.
# `ger` passes `pf>0`: its `y` is a full A column, so at large m the sequential read-modify-write is
# memory-latency-bound on high-latency memory (e.g. LPDDR5x) — one prefetch PER CACHE LINE across the
# unrolled step (the HW prefetcher can't be relied on there) hides it (measured: neuromancer ger n=4096
# 0.88→~1.0). The prefetch may reach up to `pf` elements past the column end; `llvm.prefetch` lowers to a
# non-faulting `prefetcht0`, so that's safe. Distance `pf` is a derived const (see `_GER_PF_BYTES`).
@inline function _axpy_simd!(n::Int, a::T, x, y, pf::Int = 0) where {T <: BlasReal}
    px = _ptr(x); py = _ptr(y); V = _vec(T); W = _vwidth(T); sz = sizeof(T); step = _UNROLL * W
    GC.@preserve x y begin
        va = V(a)
        i = 0
        while i + step <= n
            o = i * sz
            if pf > 0                                 # const-folds OFF when pf==0 (default / axpy)
                pb = py + (i + pf) * sz
                for c in 0:_CACHELINE:(step * sz - 1)
                    _prefetch(pb + c)
                end   # one prefetch per line
            end
            vstore(muladd(va, vload(V, px + o), vload(V, py + o)), py + o)
            vstore(muladd(va, vload(V, px + o + W * sz), vload(V, py + o + W * sz)), py + o + W * sz)
            vstore(muladd(va, vload(V, px + o + 2W * sz), vload(V, py + o + 2W * sz)), py + o + 2W * sz)
            vstore(muladd(va, vload(V, px + o + 3W * sz), vload(V, py + o + 3W * sz)), py + o + 3W * sz)
            i += step
        end
        while i + W <= n
            o = i * sz
            vstore(muladd(va, vload(V, px + o), vload(V, py + o)), py + o)
            i += W
        end
        while i < n
            j = i + 1
            unsafe_store!(py, muladd(a, unsafe_load(px, j), unsafe_load(py, j)), j)
            i += 1
        end
    end
    return y
end

@inline function _scal_simd!(n::Int, a::T, x) where {T <: BlasReal}
    px = _ptr(x); V = _vec(T); W = _vwidth(T); sz = sizeof(T); step = _UNROLL * W
    GC.@preserve x begin
        va = V(a)
        i = 0
        # Beyond L1, HAND-UNROLLING LOSES TO `@simd ivdep`. Measured 2026-07-30, wintermute freq-locked,
        # one process, back-to-back, plots.jl's own L1 regime (`_L1REP` reps on the same vector) — GB/s of
        # this kernel vs the identical pointer loop written as `@simd ivdep for i in 1:n`:
        #     n=1e3  0.99   n=3e3  0.98   n=1e4  1.01   n=3e4  1.00
        #     n=1e5  1.04   n=3e5  1.03   n=1e6  1.08
        # The crossover is L1 residency: 3e3·8 = 24 KB fits a 32 KB L1 (manual unroll wins), 1e4·8 = 80 KB
        # does not (ivdep wins). So keep the explicit 4× unroll while resident — it pipelines a short loop
        # better — and hand the rest to LLVM, which picks its own unroll/addressing and does better.
        # This is also the ACTUAL scal gate gap, and it is NOT a memory-parallelism story: at n=1e6 (8 MB
        # in a 16 MB L3) a plain Julia `x[i] *= a` loop reaches 152.8 GB/s where this kernel reached 141.6
        # — PB/raw = 0.927 against the gate's PB/AOCL = 0.923, i.e. AOCL is simply achieving what the naive
        # loop achieves. The genuine DRAM regime (n=4e6, 32 MB) sits at 76-79 GB/s with PB/raw = 0.973.
        # PDM: DERIVE tier (L1 residency over a detected const), no new knob.
        if n * sizeof(T) > _L1_BYTES
            @inbounds @simd ivdep for j in 1:n
                unsafe_store!(px, a * unsafe_load(px, j), j)
            end
            return x
        end
        # FALSIFIED 2026-07-30 (wintermute, freq-locked, plots.jl op=scal): scal misses AOCL ONLY at
        # n=1e6 (8 MB against a 16 MB L3, doubled by RMW read + dirty-writeback) and gates everywhere
        # smaller. Deepening the unroll to 8 for that regime — i.e. 8 cache lines in flight instead of
        # 4 — moved it 0.91 → 0.92, noise. So lines-in-flight is NOT the mechanism; do not re-try depth.
        # Also do not re-try NON-TEMPORAL stores: this is a load+store to the SAME address (an RMW), the
        # line is already resident from the load so there is no RFO to skip, and NT on the identical
        # structure was measured at ger n=1024 0.99 → 0.45 (see kb pureblas-zen5-ger).
        # Note `_UNROLL = 4`'s upstream justification ("reductions use 4 independent accumulators so the
        # FMA/add latency is hidden") does NOT apply here — `_scal_simd!` has no accumulator — but the
        # measurement above says the value is not what is costing us, so it stays.
        while i + step <= n
            o = i * sz
            vstore(va * vload(V, px + o), px + o)
            vstore(va * vload(V, px + o + W * sz), px + o + W * sz)
            vstore(va * vload(V, px + o + 2W * sz), px + o + 2W * sz)
            vstore(va * vload(V, px + o + 3W * sz), px + o + 3W * sz)
            i += step
        end
        while i + W <= n
            o = i * sz
            vstore(va * vload(V, px + o), px + o)
            i += W
        end
        while i < n
            j = i + 1
            unsafe_store!(px, a * unsafe_load(px, j), j)
            i += 1
        end
    end
    return x
end

# Complex scal: x .*= (alr + i·ali). Bandwidth-bound (read-modify-write), so minimise the shuffle chain:
# for a SCALAR multiplier, one swap-adjacent-pairs shuffle suffices — result = v·alr + swap(v)·[−ali,+ali…]
# (= [r·alr − i·ali, i·alr + r·ali, …] on the interleaved [r i r i…] buffer). One shuffle/vector (vs 3 for
# deinterleave+interleave), 4× unrolled to saturate memory bandwidth. `n` counts COMPLEX elements.
@generated function _scal_cmplx_simd!(n::Int, alr::T, ali::T, x) where {T <: BlasReal}
    W = _vwidth(T); V2 = Vec{2W, T}; sz = sizeof(T); Wc = 2 * W          # reals per Vec = 2W; W complex
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(2W - 1))...)     # swap adjacent (re,im)
    sgn = :($V2($(Expr(:tuple, (iseven(l) ? :(-ali) : :ali for l in 0:(2W - 1))...))))  # [−ali,ali,…]
    return quote
        px = _reptr(x); arv = $V2(alr); sv = $sgn; step = 4 * $W          # 4 vectors = 4W complex/step
        GC.@preserve x begin
            i = 0
            while i + step <= n
                @inbounds for u in 0:3
                    o = (i + u * $W) * 2 * $sz; v = vload($V2, px + o)
                    vstore(muladd(shufflevector(v, Val($swp)), sv, v * arv), px + o)
                end
                i += step
            end
            while i + $W <= n
                o = i * 2 * $sz; v = vload($V2, px + o)
                vstore(muladd(shufflevector(v, Val($swp)), sv, v * arv), px + o)
                i += $W
            end
            while i < n
                j = i + 1; re = unsafe_load(px, 2j - 1); im = unsafe_load(px, 2j)
                unsafe_store!(px, alr * re - ali * im, 2j - 1); unsafe_store!(px, alr * im + ali * re, 2j)
                i += 1
            end
        end
        return x
    end
end

# Complex axpy: y .+= (alr + i·ali)·x. Swap-pairs complex-multiply of x, fused straight into y.
# DECOMPOSITION (galen Zen3/AVX2, n=1M ≈ 32MB footprint = L3 edge, the worst point): the complex kernel
# under-extracted bandwidth with NO register spills and no prefetch/unroll sensitivity (all falsified) — so
# the residual was the compute critical path at the L3→DRAM transition, not memory. The old body was 4 vector
# ops/lane (mul xv·arv, shuffle, FMA, then a standalone `y + ax` add). Folding the y-load into the FIRST FMA
# as its addend drops it to 1 shuffle + 2 FMAs = 3 ops/lane (−25%), identical numerics, shorter dep chain
# (yv feeds the FMA directly): t = yv + xv·arv ; result = t + swap(xv)·sv. ISA-neutral. Controlled same-process
# A/B @1M: 96.5→99.1 GB/s, PB/OB 0.937→0.958 (5/5 trials). Residual vs OB is complex-specific L3-edge
# scheduling (shuffle+2FMA density limits bandwidth for OB's complex too: real 111 vs complex-OB 104 GB/s at
# matched 32MB/bytes). 4× unrolled (bandwidth-bound: reads x+y, writes y). `n` counts COMPLEX elements.
# PAST-L2 VARIANT: 256-bit vectors, and the iteration PHASE-SEPARATED (all x loads, all y loads, all
# compute, all stores) — the shape OpenBLAS's zaxpy_kernel_4 uses (verified by disassembly: it is a ymm
# kernel even under the Cooperlake dispatch, with all loads issued up front and the stores batched).
#
# BOTH halves are required, and only past L2. Measured Zen4, freq-locked (GB/s, shipped / narrow+phase /
# native+phase / OpenBLAS):
#   n=1e3   129.7 /  78.6 /  77.3 / 119.7      n=1e5  108.0 / 111.9 / 106.7 / 112.5
#   n=4096  130.0 / 113.4 / 112.8 / 129.2      n=3e5  105.0 / 110.5 / 103.2 / 109.9
#   n=1e4   128.8 / 117.4 / 118.6 / 126.4      n=1e6   69.5 /  69.9 /  69.2 /  70.7
#   n=3e4   114.2 / 106.9 / 107.8 / 111.8
# The phase shape is CATASTROPHIC while resident (78.6 vs 129.7 at n=1e3) and the narrow width only pays
# once a stream leaves L2 — applying it globally cost zaxpy 0.952→0.877 and, because complex `ger` calls
# this per COLUMN at exactly those short lengths, zgeru 0.909→0.754. Do not hoist the switch.
@generated function _axpy_cmplx_phase!(::Val{LANES}, n::Int, alr::T, ali::T, x, y) where {LANES, T <: BlasReal}
    V = Vec{LANES, T}; sz = sizeof(T); cpx = LANES ÷ 2; U = 4
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(LANES - 1))...)
    sgn = :($V($(Expr(:tuple, (iseven(l) ? :(-ali) : :ali for l in 0:(LANES - 1))...))))
    lx = [:($(Symbol(:xv, u)) = vload($V, px + (i + $u * $cpx) * 2 * $sz)) for u in 0:(U - 1)]
    ly = [:($(Symbol(:yv, u)) = vload($V, py + (i + $u * $cpx) * 2 * $sz)) for u in 0:(U - 1)]
    cp = [
        :(
            $(Symbol(:rv, u)) = muladd(
                shufflevector($(Symbol(:xv, u)), Val($swp)), sv,
                muladd($(Symbol(:xv, u)), arv, $(Symbol(:yv, u)))
            )
        ) for u in 0:(U - 1)
    ]
    st = [:(vstore($(Symbol(:rv, u)), py + (i + $u * $cpx) * 2 * $sz)) for u in 0:(U - 1)]
    return quote
        $(Expr(:meta, :inline))
        px = _reptr(x); py = _reptr(y); arv = $V(alr); sv = $sgn; step = $U * $cpx
        GC.@preserve x y begin
            i = 0
            @inbounds while i + step <= n
                $(lx...)                       # phase 1: x stream
                $(ly...)                       # phase 2: y stream
                $(cp...)                       # phase 3: compute
                $(st...)                       # phase 4: stores batched
                i += step
            end
            @inbounds while i + $cpx <= n
                o = i * 2 * $sz; xv = vload($V, px + o)
                t = muladd(xv, arv, vload($V, py + o))
                vstore(muladd(shufflevector(xv, Val($swp)), sv, t), py + o); i += $cpx
            end
            @inbounds while i < n
                j = i + 1; xr = unsafe_load(px, 2j - 1); xi = unsafe_load(px, 2j)
                unsafe_store!(py, unsafe_load(py, 2j - 1) + alr * xr - ali * xi, 2j - 1)
                unsafe_store!(py, unsafe_load(py, 2j) + alr * xi + ali * xr, 2j)
                i += 1
            end
        end
        return y
    end
end

# Is the past-L2 variant actually faster HERE? PDM **Measure** tier: it is a datapath property (Zen4
# double-pumps 512-bit ops over a 256-bit path, so the wide vector buys no bandwidth on a memory-bound
# kernel; a true 512-bit datapath can flip the sign), which no formula over the detected consts predicts
# — the req#8b tell. The WINDOW, by contrast, is Derived: one stream leaving L2.
@inline _zaxpy_narrow_lanes(::Type{T}) where {T} = 32 ÷ sizeof(T)      # 256 bits
const _ZAXPY_NARROW_PREF = @load_preference("zaxpy_narrow", nothing)
@static if isnothing(_ZAXPY_NARROW_PREF)
    function _measure_zaxpy_narrow()::Bool                 # straight-line, no closures, TOTAL
        Base.generating_output() && return false           # never burn a measure during precompile
        try
            nn = _zaxpy_narrow_lanes(Float64); nw = 2 * _vwidth(Float64)
            nn == nw && return false
            n = max(4096, 3 * _L2_BYTES ÷ sizeof(ComplexF64))   # inside the window: stream > L2
            x = fill(ComplexF64(1.0, 0.5), n); y = fill(ComplexF64(0.25, -0.75), n)
            tn = typemax(UInt64); tw = typemax(UInt64)
            for r in 0:3                                   # r=0 untimed warmup; ABBA after
                s = time_ns()
                _axpy_cmplx_phase!(Val(_zaxpy_narrow_lanes(Float64)), n, 1.0e-9, 0.0, x, y)
                e = time_ns() - s; r > 0 && (tn = min(tn, e))
                s = time_ns()
                _axpy_cmplx_wide!(n, 1.0e-9, 0.0, x, y)
                e = time_ns() - s; r > 0 && (tw = min(tw, e))
            end
            return tn < tw
        catch
            return false
        end
    end
    const _ZAXPY_NARROW_ONCE = Base.OncePerProcess{Bool}(_measure_zaxpy_narrow)
    @inline _zaxpy_narrow() = _ZAXPY_NARROW_ONCE()
else
    @inline _zaxpy_narrow() = _ZAXPY_NARROW_PREF::Bool
end

# The size test comes FIRST so short calls (complex `ger`, per column) never even reach the
# OncePerProcess lookup and run byte-identical code to before.
@inline function _axpy_cmplx_simd!(n::Int, alr::T, ali::T, x, y) where {T <: BlasReal}
    if n * 2 * sizeof(T) > _L2_BYTES && _zaxpy_narrow()
        return _axpy_cmplx_phase!(Val(_zaxpy_narrow_lanes(T)), n, alr, ali, x, y)
    end
    return _axpy_cmplx_wide!(n, alr, ali, x, y)
end

@generated function _axpy_cmplx_wide!(n::Int, alr::T, ali::T, x, y) where {T <: BlasReal}
    W = _vwidth(T); V2 = Vec{2W, T}; sz = sizeof(T)
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(2W - 1))...)
    sgn = :($V2($(Expr(:tuple, (iseven(l) ? :(-ali) : :ali for l in 0:(2W - 1))...))))
    return quote
        px = _reptr(x); py = _reptr(y); arv = $V2(alr); sv = $sgn; step = 4 * $W
        GC.@preserve x y begin
            i = 0
            while i + step <= n
                @inbounds for u in 0:3
                    o = (i + u * $W) * 2 * $sz; xv = vload($V2, px + o)
                    t = muladd(xv, arv, vload($V2, py + o))                      # y + arv·x
                    vstore(muladd(shufflevector(xv, Val($swp)), sv, t), py + o)  # + swap(x)·sv
                end
                i += step
            end
            while i + $W <= n
                o = i * 2 * $sz; xv = vload($V2, px + o)
                t = muladd(xv, arv, vload($V2, py + o))
                vstore(muladd(shufflevector(xv, Val($swp)), sv, t), py + o); i += $W
            end
            while i < n
                j = i + 1; xr = unsafe_load(px, 2j - 1); xi = unsafe_load(px, 2j)
                unsafe_store!(py, unsafe_load(py, 2j - 1) + alr * xr - ali * xi, 2j - 1)
                unsafe_store!(py, unsafe_load(py, 2j) + alr * xi + ali * xr, 2j)
                i += 1
            end
        end
        return y
    end
end

# Complex dot: Σ (CJ ? conj(xᵢ) : xᵢ)·yᵢ. NO per-iteration deinterleave (too many shuffles on AVX2 — cost
# dotu 0.70 there). Instead accumulate two INTERLEAVED products: p = Σ x·y = [Σxr·yr, Σxi·yi, …] and
# q = Σ x·swap(y) = [Σxr·yi, Σxi·yr, …] — ONE shuffle (swap y) + 2 FMAs/iter, identical for dotu/dotc.
# Deinterleave only the 2 accumulators ONCE at the end; CJ flips two combine signs. 4× unrolled for the
# FMA-reduction latency. Returns Complex{T}. `n` counts COMPLEX elements.
@generated function _dot_cmplx_simd(n::Int, x, y, ::Type{T}, ::Val{CJ}) where {T <: BlasReal, CJ}
    W = _vwidth(T); V2 = Vec{2W, T}; sz = sizeof(T)
    # Unroll from the REGISTER BUDGET (req#8), not a literal: the 2·UNR `Vec{2W}` accumulators each take 2
    # physical vector registers, so keep 4·UNR ≲ Nregs−6 (leave ~6 for the x/y/swap loads). AVX2 has 16
    # vector regs, AVX-512 has 32 — the old hardcoded 4× put all 16 YMM into accumulators on AVX2 → spill
    # (dotc/dotu small-n 0.75×); this gives 2× on AVX2 / 4× on AVX-512, register-resident on both.
    UNR = clamp(((_SIMD_BYTES == 64 ? 32 : 16) - 6) ÷ 4, 1, 4)
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(2W - 1))...)
    ps = [Symbol(:p, u) for u in 0:(UNR - 1)]; qs = [Symbol(:q, u) for u in 0:(UNR - 1)]
    init = Expr(:block, (:($(ps[u + 1]) = zero($V2); $(qs[u + 1]) = zero($V2)) for u in 0:(UNR - 1))...)
    psum = foldl((a, b) -> :($a + $b), ps); qsum = foldl((a, b) -> :($a + $b), qs)
    body = Expr(:block)
    for u in 0:(UNR - 1)
        o = :((i + $u * $W) * 2 * $sz)
        push!(
            body.args, quote
                xv = vload($V2, px + $o); yv = vload($V2, py + $o)
                $(ps[u + 1]) = muladd(xv, yv, $(ps[u + 1])); $(qs[u + 1]) = muladd(xv, shufflevector(yv, Val($swp)), $(qs[u + 1]))
            end
        )
    end
    return quote
        px = _reptr(x); py = _reptr(y); step = $UNR * $W
        $init
        GC.@preserve x y begin
            i = 0
            while i + step <= n
                @inbounds begin
                    $body
                end
                i += step
            end
            while i + $W <= n
                @inbounds begin
                    xv = vload($V2, px + i * 2 * $sz); yv = vload($V2, py + i * 2 * $sz)
                    p0 = muladd(xv, yv, p0); q0 = muladd(xv, shufflevector(yv, Val($swp)), q0)
                end
                i += $W
            end
            # dotu: real=Σxr·yr−Σxi·yi, imag=Σxr·yi+Σxi·yr ;  dotc (conj x): signs of the xi terms flip.
            # Parity-preserving fold → [Σeven, Σodd] instead of deint + two full horizontal sums (see gemm.jl).
            pfld = _fold2_cmplx($psum)         # [Σxr·yr, Σxi·yi]  (unique names: $psum reads p0..p_{UNR-1})
            qfld = _fold2_cmplx($qsum)         # [Σxr·yi, Σxi·yr]
            sr = pfld[1] + $(CJ ? :(pfld[2]) : :(-pfld[2]))
            si = qfld[1] + $(CJ ? :(-qfld[2]) : :(qfld[2]))
            @inbounds while i < n
                j = i + 1; xr = unsafe_load(px, 2j - 1); xi = unsafe_load(px, 2j); yr = unsafe_load(py, 2j - 1); yi = unsafe_load(py, 2j)
                sr += xr * yr + $(CJ ? :(xi * yi) : :(-xi * yi)); si += xr * yi + $(CJ ? :(-xi * yr) : :(xi * yr))
                i += 1
            end
            return Complex{$T}(sr, si)
        end
    end
end

# Fused rank-2 complex conj-dot for the QR panel: returns (Σ conj(v0)·c, Σ conj(v1)·c) in ONE pass over
# c — the two reflectors share the c-load (halves the panel's level-2 read traffic vs two _dot_cmplx_simd
# calls). Same interleaved swap-adjacent idiom as _dot_cmplx_simd (CJ=true). `n` counts COMPLEX elements.
@generated function _qr_dot2c_cmplx(n::Int, v0, v1, c, ::Type{T}) where {T <: BlasReal}
    W = _vwidth(T); V2 = Vec{2W, T}; sz = sizeof(T)
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(2W - 1))...)
    return quote
        pv0 = _reptr(v0); pv1 = _reptr(v1); pc = _reptr(c)
        p0 = zero($V2); q0 = zero($V2); p1 = zero($V2); q1 = zero($V2); i = 0
        GC.@preserve v0 v1 c begin
            while i + $W <= n
                @inbounds begin
                    o = i * 2 * $sz
                    cv = vload($V2, pc + o); cs = shufflevector(cv, Val($swp))
                    a0 = vload($V2, pv0 + o); a1 = vload($V2, pv1 + o)
                    p0 = muladd(a0, cv, p0); q0 = muladd(a0, cs, q0)
                    p1 = muladd(a1, cv, p1); q1 = muladd(a1, cs, q1)
                end
                i += $W
            end
            pf0 = _fold2_cmplx(p0); qf0 = _fold2_cmplx(q0); pf1 = _fold2_cmplx(p1); qf1 = _fold2_cmplx(q1)
            d0r = pf0[1] + pf0[2]; d0i = qf0[1] - qf0[2]; d1r = pf1[1] + pf1[2]; d1i = qf1[1] - qf1[2]
            @inbounds while i < n
                j = i + 1
                v0r = unsafe_load(pv0, 2j - 1); v0i = unsafe_load(pv0, 2j)
                v1r = unsafe_load(pv1, 2j - 1); v1i = unsafe_load(pv1, 2j)
                cr = unsafe_load(pc, 2j - 1); ci = unsafe_load(pc, 2j)
                d0r += v0r * cr + v0i * ci; d0i += v0r * ci - v0i * cr
                d1r += v1r * cr + v1i * ci; d1i += v1r * ci - v1i * cr
                i += 1
            end
            return (Complex{$T}(d0r, d0i), Complex{$T}(d1r, d1i))
        end
    end
end

# Fused rank-2 complex axpy for the QR panel: c .+= k0·v0 + k1·v1 in ONE pass over c (shares the c
# read/write across both reflectors). Swap-pairs complex-multiply, same idiom as _axpy_cmplx_simd.
@generated function _qr_axpy2_cmplx!(n::Int, k0r::T, k0i::T, k1r::T, k1i::T, v0, v1, c) where {T <: BlasReal}
    W = _vwidth(T); V2 = Vec{2W, T}; sz = sizeof(T)
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(2W - 1))...)
    sgn0 = :($V2($(Expr(:tuple, (iseven(l) ? :(-k0i) : :k0i for l in 0:(2W - 1))...))))
    sgn1 = :($V2($(Expr(:tuple, (iseven(l) ? :(-k1i) : :k1i for l in 0:(2W - 1))...))))
    return quote
        pv0 = _reptr(v0); pv1 = _reptr(v1); pc = _reptr(c)
        ar0 = $V2(k0r); ar1 = $V2(k1r); s0 = $sgn0; s1 = $sgn1; i = 0
        GC.@preserve v0 v1 c begin
            while i + $W <= n
                @inbounds begin
                    o = i * 2 * $sz
                    a0 = vload($V2, pv0 + o); a1 = vload($V2, pv1 + o); t = vload($V2, pc + o)
                    t = muladd(a0, ar0, t); t = muladd(shufflevector(a0, Val($swp)), s0, t)
                    t = muladd(a1, ar1, t); t = muladd(shufflevector(a1, Val($swp)), s1, t)
                    vstore(t, pc + o)
                end
                i += $W
            end
            @inbounds while i < n
                j = i + 1
                v0r = unsafe_load(pv0, 2j - 1); v0i = unsafe_load(pv0, 2j)
                v1r = unsafe_load(pv1, 2j - 1); v1i = unsafe_load(pv1, 2j)
                cr = unsafe_load(pc, 2j - 1); ci = unsafe_load(pc, 2j)
                unsafe_store!(pc, cr + k0r * v0r - k0i * v0i + k1r * v1r - k1i * v1i, 2j - 1)
                unsafe_store!(pc, ci + k0r * v0i + k0i * v0r + k1r * v1i + k1i * v1r, 2j)
                i += 1
            end
        end
        return c
    end
end

@inline function _copy_simd!(n::Int, x, y) # T inferred from the pointer/array element type
    T = _et(x); px = _ptr(x); py = _ptr(y); V = _vec(T); W = _vwidth(T); sz = sizeof(T); step = _UNROLL * W
    GC.@preserve x y begin
        i = 0
        while i + step <= n
            o = i * sz
            vstore(vload(V, px + o), py + o)
            vstore(vload(V, px + o + W * sz), py + o + W * sz)
            vstore(vload(V, px + o + 2W * sz), py + o + 2W * sz)
            vstore(vload(V, px + o + 3W * sz), py + o + 3W * sz)
            i += step
        end
        while i + W <= n
            o = i * sz
            vstore(vload(V, px + o), py + o)
            i += W
        end
        while i < n
            j = i + 1
            unsafe_store!(py, unsafe_load(px, j), j)
            i += 1
        end
    end
    return y
end

@inline function _swap_simd!(n::Int, x, y)
    T = _et(x); px = _ptr(x); py = _ptr(y); V = _vec(T); W = _vwidth(T); sz = sizeof(T); step = _UNROLL * W
    GC.@preserve x y begin
        i = 0
        while i + step <= n
            o = i * sz
            for u in 0:(_UNROLL - 1)
                oo = o + u * W * sz
                vx = vload(V, px + oo); vy = vload(V, py + oo)
                vstore(vy, px + oo); vstore(vx, py + oo)
            end
            i += step
        end
        while i + W <= n
            o = i * sz
            vx = vload(V, px + o); vy = vload(V, py + o)
            vstore(vy, px + o); vstore(vx, py + o)
            i += W
        end
        while i < n
            j = i + 1
            t = unsafe_load(px, j)
            unsafe_store!(px, unsafe_load(py, j), j)
            unsafe_store!(py, t, j)
            i += 1
        end
    end
    return nothing
end

# Reductions use 4 independent accumulators so the FMA/add latency is hidden — a single
# accumulator is latency-bound (serial dependency) and leaves the pipeline idle at L1-resident
# sizes. 4 chains × W lanes per iteration; then a W-at-a-time pass, then a scalar tail.
const _UNROLL = 4

@inline function _dot_simd(n::Int, x, y, ::Type{T}) where {T <: BlasReal}
    px = _ptr(x); py = _ptr(y); V = _vec(T); W = _vwidth(T); sz = sizeof(T); step = _UNROLL * W
    GC.@preserve x y begin
        a0 = zero(V); a1 = zero(V); a2 = zero(V); a3 = zero(V)
        i = 0
        while i + step <= n
            o = i * sz
            a0 = muladd(vload(V, px + o), vload(V, py + o), a0)
            a1 = muladd(vload(V, px + o + W * sz), vload(V, py + o + W * sz), a1)
            a2 = muladd(vload(V, px + o + 2W * sz), vload(V, py + o + 2W * sz), a2)
            a3 = muladd(vload(V, px + o + 3W * sz), vload(V, py + o + 3W * sz), a3)
            i += step
        end
        acc = (a0 + a1) + (a2 + a3)
        while i + W <= n
            o = i * sz
            acc = muladd(vload(V, px + o), vload(V, py + o), acc)
            i += W
        end
        s = sum(acc)
        while i < n
            j = i + 1
            s += unsafe_load(px, j) * unsafe_load(py, j)
            i += 1
        end
    end
    return s
end

@inline function _asum_simd(n::Int, x, ::Type{T}) where {T <: BlasReal}
    px = _ptr(x); V = _vec(T); W = _vwidth(T); sz = sizeof(T); step = _UNROLL * W
    GC.@preserve x begin
        a0 = zero(V); a1 = zero(V); a2 = zero(V); a3 = zero(V)
        i = 0
        while i + step <= n
            o = i * sz
            a0 += abs(vload(V, px + o))
            a1 += abs(vload(V, px + o + W * sz))
            a2 += abs(vload(V, px + o + 2W * sz))
            a3 += abs(vload(V, px + o + 3W * sz))
            i += step
        end
        acc = (a0 + a1) + (a2 + a3)
        while i + W <= n
            acc += abs(vload(V, px + i * sz))
            i += W
        end
        s = sum(acc)
        while i < n
            j = i + 1
            s += abs(unsafe_load(px, j))
            i += 1
        end
    end
    return s
end

# Sum of squares, SIMD, 4 accumulators. Fast path for nrm2 — may overflow to Inf or underflow to 0
# on extreme inputs; the caller (_nrm2) detects that and falls back to scaled lassq.
@inline function _sumsq_simd(n::Int, x, ::Type{T}) where {T <: BlasReal}
    px = _ptr(x); V = _vec(T); W = _vwidth(T); sz = sizeof(T); step = _UNROLL * W
    GC.@preserve x begin
        a0 = zero(V); a1 = zero(V); a2 = zero(V); a3 = zero(V)
        i = 0
        while i + step <= n
            o = i * sz
            v0 = vload(V, px + o); v1 = vload(V, px + o + W * sz)
            v2 = vload(V, px + o + 2W * sz); v3 = vload(V, px + o + 3W * sz)
            a0 = muladd(v0, v0, a0); a1 = muladd(v1, v1, a1)
            a2 = muladd(v2, v2, a2); a3 = muladd(v3, v3, a3)
            i += step
        end
        acc = (a0 + a1) + (a2 + a3)
        while i + W <= n
            v = vload(V, px + i * sz); acc = muladd(v, v, acc); i += W
        end
        s = sum(acc)
        while i < n
            j = i + 1; v = unsafe_load(px, j); s = muladd(v, v, s); i += 1
        end
    end
    return s
end

# SIMD argmax for BLAS iamax: 1-based index of the first element with maximal |x|. Real unit-stride;
# assumes n ≥ 4W (caller routes shorter / strided / complex to the scalar loop). Two implementations,
# selected by ISA at build time (`_SIMD_BYTES` const-folds → trim-safe, no runtime branch):
#
#  • AVX2 + AVX-512 (`_iamax_thresh!`): DEPENDENCY-FREE THRESHOLD SCAN. The hot loop keeps a broadcast
#    running max `thr` (invariant except on the rare advance) and does FOUR INDEPENDENT `|xⱼ| > thr`
#    compares per iter, OR-ing the masks into ONE `any` branch — no loop-carried accumulator, so it
#    out-throughputs even a bare vmaxpd reduction. The predicted-not-taken cold path (max advances
#    ~O(log n) for random data) scalar-scans the blocks for the new max + its first lane. Tracks NO index
#    in the hot path, unlike the chain kernel below. WIDTH-GENERAL (W=4 f64 / W=8 f32). Beats OpenBLAS
#    1.6–2.1× on Zen4. (Old AVX2 chain was the worst AOCL miss: f64 0.55×, f32 0.32× — git log.)
#
#    It does NOT gate AOCL-BLIS. Measured 2026-08-03, wintermute, freq-locked, bench/plots.jl arms=pb,
#    pb/AOCL by size: n=1e3 0.967 | 3e3 1.039 | 1e4 0.957 | 3e4 0.974 | 1e5 0.937 | 3e5 1.03 | 1e6 1.016
#    ⇒ gate 0.937. Two claims that used to stand here are FALSE and were removed: "gates AOCL at EVERY
#    size" (unreproducible — it came from bench/iamax_nb.jl, which the kb already flags for stale seeding),
#    and "at n≥1e6 both sit on the shared DRAM roofline" (the harness runs `_L1REP(1e6)=30` reps over one
#    8 MB buffer, so it is L3-resident; 73–77 GB/s is far above Zen4 single-thread DRAM. n=1e6 is an
#    L3-bandwidth cell at the half-L3 edge, not a roofline).
#
#    THE GAP vs AOCL, read from `bli_damaxv_zen_int_avx512`'s disassembly: it issues TWO ops per vector —
#    `vandnpd (mem),zmm_sign,zmmX` (abs FUSED into the load) then `vmaxpd` into 4 accumulators — and finds
#    only the max VALUE, taking a SECOND PASS for the index. This scan issues three (vload, vandpd,
#    vcmppd). A ceiling probe deleting the abs measured +21% at n=1e3, +2% at n=1e5, ~0 at n=1e6, so the
#    third op costs real time only where the loop is issue-bound. AOCL can afford its second pass because
#    inside the gate's size range the re-read is L3-resident; it would lose in a true DRAM regime, which
#    the gate does not currently measure at all.
#    FALSIFIED 2026-08-03 (do not rebuild): porting that as a CHUNKED single-pass max-accumulate —
#    accumulate 2 ops/vector over an L1-resident chunk, re-walk only a chunk whose max beats `gmax` —
#    was correct (netlib NaN/tie intact: `max(::Vec,::Vec)` is llvm.maxnum and drops NaN, while
#    `maximum(::Vec)` and scalar `max` propagate, so the fold and tail used the strict `>` walk) and MUCH
#    slower: pb/AOCL 0.435/0.486/0.599/0.697/0.827/0.897/0.921 across n=1e3…1e6. Reason is RE-WALK
#    GRANULARITY: this scan detects per BLOCK and rescans ~NB·W elements, a chunk is 64× coarser, and at
#    small n the chunk IS the array so it degenerates to a literal two-pass. Any revival must detect at
#    block granularity — which is what this kernel already does.
#  • SSE / NEON (narrower than 32 B, unvalidated) (`_iamax_chain4!`): the original 4-chain lane-parallel
#    running max + parallel index vector (loop-carried, latency-bound → 4 chains). Kept as the fallback.
#
# Tie rule everywhere: equal value → keep the SMALLER index ⇒ BLAS first-occurrence semantics (strict `>`).
# NB: OB/AOCL idamax are alignment-volatile (time swings ~60% with the array's address) while these are
# stable — sample many fresh allocations and take the median (see bench/plots.jl iamax entry).
@inline _amax_up(m, i, nv, ni) = (t = nv > m; (vifelse(t, nv, m), vifelse(t, ni, i)))
# Fold across chains: indices interleave, so the tie must pick the smaller index explicitly.
@inline _amax_merge(m0, i0, m1, i1) = begin
    take = (m1 > m0) | ((m1 == m0) & (i1 < i0))
    (vifelse(take, m1, m0), vifelse(take, i1, i0))
end

# NB = independent |xⱼ|>thr compare-chains per hot-loop iteration. It is a REAL parameter (`Val{NB}`,
# body generated from it). It used to be `const _IAMAX_NB = min(4, _NVREG - 1)`, advertised as the
# tunable unroll but INERT and a trap: it fed only `step` while the body loaded four blocks
# unconditionally, so setting it to 2 made the loop re-read every element AND read up to 2W past the
# guard, and 8 made it SKIP half the data and return wrong answers. (The NB numbers in the old comment
# did not come from turning it — bench/iamax_nb.jl @evals its own unrolled kernels; those are suspect
# for a different reason, the kb records that file as carrying stale seeding.)
#
# NB IS CHOSEN BY L2 RESIDENCY — PDM **Derive** tier, no knob, no preference, no runtime measurement.
# Criterion: while the stream fits L2, load latency is low and two lines in flight cover it, so the
# shorter loop (less per-iteration overhead, fewer live registers) wins; once the stream leaves L2 the
# misses must be hidden by more outstanding lines, and 4 beats 2. Measured on wintermute (Zen4, L2 = 1 MB,
# freq-locked, plots.jl's own fresh-allocation regime, 90 samples, GB/s median) — NB=2 over NB=4:
#     7 KB +12.7% | 23 KB +8.8% | 78 KB +5.1% | 234 KB +4.2% | 781 KB +4.2% | 1024 KB (=L2) +1.5%
#     1562 KB −5.5% | 2343 KB −0.4% | 4.6 MB −7.6% | 7.6 MB −11.1% | 15 MB −9.5% | 30 MB −9.9%
# The sign flips exactly at L2, which is why the criterion is `n*sizeof(T) <= _L2_BYTES` and not a fitted
# constant. NOTE THE STATISTIC: at n≥600 KB the NB=2 loss is a TAIL, not a slower best case (n=1e6:
# identical min, ~15% worse median, p90/min 1.20 vs 1.06) — consistent with fewer outstanding lines
# absorbing an L3 miss worse. A 24-sample A/B could not see it and ranked NB=2 ahead at every size; it
# took ~90 samples. Do not re-rank this knob from a small sample.
# RESIDENT unroll — **Derive**: hold TWO CACHE LINES in flight per iteration. While the stream fits L2 the
# load latency is low, so extra chains buy nothing and the shorter loop's lower per-iteration overhead
# wins; two lines is what keeps the load stream fed without lengthening the body. The ISA sets how many
# blocks that is, which is the whole point of deriving it — AVX-512 (_SIMD_BYTES=64) → 2, AVX2 (32) → 4.
# BOTH are the measured optimum on their box, and they are DIFFERENT NUMBERS: shipping the Zen4 value (2)
# as a literal regressed galen's iamax from gate 1.022 (PASS) to 0.934 (FAIL), −18/−19% at n=1e3/3e3.
# That is the req#8 lesson in one line: the constant that looked µarch-specific was a fixed byte budget.
const _IAMAX_NB_RESIDENT = max(1, 2 * _CACHELINE ÷ _SIMD_BYTES)
# STREAMING unroll — past L2 the criterion changes from bytes to ILP: independent compare chains to cover
# the miss latency. Four is ISA-invariant here and is also the incumbent value, now validated BOTH ways —
# Zen4 4 blocks = 4 lines (n=1e6: 72.3 vs 64.3 GB/s for 2), Zen3 4 blocks = 2 lines (78.5, beating both 2
# at 75.0 and 8 at 77.4). Deriving this as a line budget like the resident arm gives 8 on AVX2, which
# MEASURED WORSE on galen at every size — so it is a chain count, not a byte budget.
# req8-ok: ILP chain count, ISA-invariant, incumbent value measured optimal on Zen3 and Zen4 (see above)
const _IAMAX_NB_STREAM = 4

@generated function _iamax_thresh!(::Val{NB}, n::Int, xp::Ptr{T}) where {NB, T <: BlasReal}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T)
    vs = [Symbol(:v, j) for j in 0:(NB - 1)]
    cs = [Symbol(:c, j) for j in 0:(NB - 1)]
    loads = [:($(vs[j + 1]) = abs(vload($V, xp + (o + $(j * W)) * $sz))) for j in 0:(NB - 1)]
    cmps = [:($(cs[j + 1]) = $(vs[j + 1]) > thr) for j in 0:(NB - 1)]
    ortree = reduce((a, b) -> :($a | $b), cs)
    walks = [quote
            if any($(cs[j + 1]))
                bm = gmax; bl = 0
                for l in 1:$W
                    $(vs[j + 1])[l] > bm && (bm = $(vs[j + 1])[l]; bl = l)   # strict > ⇒ first lane on ties
                end
                bl != 0 && (gmax = bm; bi = o + $(j * W) + bl; thr = $V(gmax))
            end
        end for j in 0:(NB - 1)]
    # Straight-line only: no ntuple, no closure, no tuple return. A closure in this loop body measured
    # 160× slower on 2026-07-31 and 300× on 2026-08-03. `Expr(:meta, :inline)` is emitted because
    # `@inline` does NOT propagate into @generated CodeInfo on 1.12.
    #
    # FALSIFIED 2026-08-03, do NOT rebuild — CHUNKED MAX-ACCUMULATE. AOCL's `bli_damaxv_zen_int_avx512`
    # issues 2 ops/vector (`vandnpd (mem),zmm_sign,zmmX` — abs fused into the load — then `vmaxpd` into 4
    # accumulators, no index in the hot loop, index on a SECOND pass) where this scan issues 3 (vload,
    # vandpd, vcmppd). Porting it single-pass — accumulate over an L1-resident chunk, re-walk only a chunk
    # whose max beats `gmax` — was CORRECT (netlib NaN/tie intact: `max(::Vec,::Vec)` is llvm.maxnum and
    # drops NaN, while `maximum(::Vec)` and scalar `max` propagate, so the fold and tail used the strict
    # `>` walk) and MUCH slower: pb/AOCL 0.435/0.486/0.599/0.697/0.827/0.897/0.921 at n=1e3…1e6. Reason is
    # RE-WALK GRANULARITY — this scan detects per BLOCK and rescans ~NB·W elements, a chunk is 64× coarser,
    # and at small n the chunk IS the array so it degenerates into a literal two-pass. Any revival must
    # detect at block granularity, which is what this already does. AOCL can afford its second pass only
    # because inside the gate's size range the re-read is L3-resident.
    return quote
        $(Expr(:meta, :inline))
        step = $(NB * W)
    # Seed from |x[1]|, NOT typemin — reference netlib `idamax` starts `DMAX = DABS(DX(1))`, and seeding
    # lower silently changed the NaN contract WITH VECTOR LENGTH: `[NaN, 1.0]` returned 1 on the scalar
    # path (n < 4W) but the index of the true max on this one, because a NaN never beats typemin and so
    # got skipped, whereas seeding from x[1] makes every later compare-with-NaN false and pins the answer
    # at 1. Same routine, two different semantics either side of an internal threshold (found 2026-07-31
    # by the new netlib-oracle NaN testitem). For finite data this is identical — element 1 is simply
    # accounted for up front instead of via the first compare — and it can only REDUCE cold-path entries,
    # since the starting threshold is now a real element rather than typemin.
    # `_iamax_simd_try` gates on n >= 4W, so x[1] always exists here.
        gmax = abs(unsafe_load(xp, 1)); bi = 1; thr = $V(gmax); o = 0
        @inbounds while o + step <= n                 # dependency-free: NB independent compares vs `thr`
            $(loads...)
            # FALSIFIED 2026-07-31: rescanning ONLY the blocks whose mask is non-empty via a small closure
            # returning the updated `(gmax, bi, thr)` is semantically exact and looks like an NB× cut of
            # cold-path work — but it measured 0.006 vs AOCL, ~160× SLOWER, at every size. The closure
            # defeats whatever keeps this loop in registers. That is why the walks below are generated
            # straight-line instead.
            $(cmps...)
            if any($ortree)                                              # rare (running max advances)
                $(walks...)
            end
            o += step
        end
        @inbounds while o + $W <= n                    # leftover full blocks
            u0 = abs(vload($V, xp + o * $sz))
            if any(u0 > thr)
                bm = gmax; bl = 0
                for l in 1:$W
                    u0[l] > bm && (bm = u0[l]; bl = l)
                end
                bl != 0 && (gmax = bm; bi = o + bl; thr = $V(gmax))
            end
            o += $W
        end
        @inbounds for k in (o + 1):n                   # scalar remainder (no OOB / masked read needed)
            a = abs(unsafe_load(xp, k)); a > gmax && (gmax = a; bi = k)
        end
        return bi
    end
end

@inline function _iamax_chain4!(n::Int, xp::Ptr{T}) where {T <: BlasReal}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T); step = 4W
    lane = Vec(ntuple(i -> i, Val(W)))               # 1,2,…,W
    ld(o) = abs(vload(V, xp + o * sz))
    m0 = ld(0); m1 = ld(W); m2 = ld(2W); m3 = ld(3W)
    i0 = lane; i1 = lane + W; i2 = lane + 2W; i3 = lane + 3W
    o = step
    @inbounds while o + step <= n                    # 4 independent chains (cheap inner update)
        (m0, i0) = _amax_up(m0, i0, ld(o), lane + o)
        (m1, i1) = _amax_up(m1, i1, ld(o + W), lane + (o + W))
        (m2, i2) = _amax_up(m2, i2, ld(o + 2W), lane + (o + 2W))
        (m3, i3) = _amax_up(m3, i3, ld(o + 3W), lane + (o + 3W))
        o += step
    end
    (m0, i0) = _amax_merge(m0, i0, m1, i1); (m2, i2) = _amax_merge(m2, i2, m3, i3)
    (m0, i0) = _amax_merge(m0, i0, m2, i2)           # fold 4 chains → 1
    @inbounds while o + W <= n                        # leftover full blocks
        (m0, i0) = _amax_up(m0, i0, ld(o), lane + o); o += W
    end
    if o < n                                          # masked remainder (no OOB read)
        msk = lane <= (n - o)
        v = abs(vload(V, xp + o * sz, msk)); take = (v > m0) & msk
        m0 = vifelse(take, v, m0); i0 = vifelse(take, lane + o, i0)
    end
    mx = m0[1]; bi = i0[1]
    @inbounds for l in 2:W
        ml = m0[l]
        (ml > mx || (ml == mx && i0[l] < bi)) && (mx = ml; bi = i0[l])
    end
    return bi
end

# ISA-selected at build time: the `_SIMD_BYTES` compare const-folds, so the dead arm is eliminated
# (trim-safe, no runtime dispatch). AVX2 + AVX-512 (≥32 B) → threshold scan; narrower unvalidated ISAs
# (SSE/NEON) keep the 4-chain.
#
# The unroll is picked by L2 RESIDENCY — PDM Derive tier, see `_iamax_thresh!`. Both `Val`s are LITERAL,
# so this stays trim-safe (no runtime→Val) and both specializations are statically reachable; the test is
# a single compare against a const-folded `_L2_BYTES`.
@inline _iamax_simd!(n::Int, xp::Ptr{T}) where {T <: BlasReal} =
    _SIMD_BYTES >= 32 ?
    (n * sizeof(T) <= _L2_BYTES ?
     _iamax_thresh!(Val(_IAMAX_NB_RESIDENT), n, xp) : _iamax_thresh!(Val(_IAMAX_NB_STREAM), n, xp)) :
    _iamax_chain4!(n, xp)

# Complex iamax (icamax/izamax): 1-based index of the first element with maximal |re|+|im|. Same 4-chain
# argmax machinery as the real kernel, but each Vec{2W} load is W complex elements — deinterleave → re/im,
# magnitude = |re|+|im| → Vec{W}. `n` counts COMPLEX elements; xp points at the interleaved [r i r i…]
# buffer (Ptr{T}, T the real type). Was a scalar loop (~0.6× OB); this vectorizes the magnitude + argmax.
# |re|+|im| in the INTERLEAVED domain: abs, then add each real to its swapped partner (within-128-bit-lane
# shuffle — far cheaper than the cross-lane deinterleave). Result Vec{2W} has magnitude mₖ=|rₖ|+|iₖ|
# DUPLICATED in each (re,im) lane-pair: [m0,m0,m1,m1,…]. Argmax then runs over 2W lanes with the complex
# index duplicated per pair, so no deinterleave/extract shuffle at all — memory-bandwidth-bound like OB.
@inline @generated function _cmag2(v::Vec{N, T}) where {N, T}
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(N - 1))...)   # swap adjacent re↔im
    return :($(Expr(:meta, :inline)); (av = abs(v); av + shufflevector(av, Val($swp))))
end
@inline function _iamax_cmplx_simd!(n::Int, xp::Ptr{T}) where {T <: BlasReal}
    W = _vwidth(T); V = Vec{2W, T}; sz = sizeof(T); step = 4W
    clane = Vec(ntuple(i -> (i + 1) ÷ 2, Val(2W)))        # 1,1,2,2,…,W,W (complex index per real lane)
    magc(c) = _cmag2(vload(V, xp + 2c * sz))              # Vec{2W}, mₖ duplicated per pair
    m0 = magc(0); m1 = magc(W); m2 = magc(2W); m3 = magc(3W)
    i0 = clane; i1 = clane + W; i2 = clane + 2W; i3 = clane + 3W
    c = step
    @inbounds while c + step <= n                         # 4 independent chains (loop-carried max latency)
        (m0, i0) = _amax_up(m0, i0, magc(c), clane + c)
        (m1, i1) = _amax_up(m1, i1, magc(c + W), clane + (c + W))
        (m2, i2) = _amax_up(m2, i2, magc(c + 2W), clane + (c + 2W))
        (m3, i3) = _amax_up(m3, i3, magc(c + 3W), clane + (c + 3W))
        c += step
    end
    (m0, i0) = _amax_merge(m0, i0, m1, i1); (m2, i2) = _amax_merge(m2, i2, m3, i3)
    (m0, i0) = _amax_merge(m0, i0, m2, i2)                # fold 4 chains → 1
    @inbounds while c + W <= n                             # leftover full blocks (W complex each)
        (m0, i0) = _amax_up(m0, i0, magc(c), clane + c); c += W
    end
    if c < n                                               # masked remainder (no OOB read)
        rem = n - c
        rmsk = Vec(ntuple(i -> i, Val(2W))) <= 2 * rem     # real-lane mask (2 reals / complex)
        av = abs(vload(V, xp + 2c * sz, rmsk)); mag = _cmag2_masked(av)
        cmsk = clane <= rem; take = (mag > m0) & cmsk
        m0 = vifelse(take, mag, m0); i0 = vifelse(take, clane + c, i0)
    end
    mx = m0[1]; bi = i0[1]
    @inbounds for l in 2:2W                                # reduce 2W lanes; strict > keeps first occurrence
        ml = m0[l]
        (ml > mx || (ml == mx && i0[l] < bi)) && (mx = ml; bi = i0[l])
    end
    return bi
end
# masked remainder magnitude: masked-out reals load as 0, so |·| pairs sum correctly (0 lanes stay 0).
@inline @generated function _cmag2_masked(av::Vec{N, T}) where {N, T}
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(N - 1))...)
    return :($(Expr(:meta, :inline)); av + shufflevector(av, Val($swp)))
end
