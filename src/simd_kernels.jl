# SIMD.jl fast paths for the bandwidth-bound Level-1 kernels, real (`Float32`/`Float64`),
# unit-stride, dense memory. Everything else falls back to the generic scalar loops in
# level1.jl. Pointer-based `vload`/`vstore` over `GC.@preserve`d buffers; masked scalar tail.

using SIMD: Vec, vload, vstore, vifelse, shufflevector

# Vec type for element `T` at the detected register width (folds to a concrete type — see cpuinfo.jl).
@inline _vec(::Type{T}) where {T} = Vec{_vwidth(T), T}

# Eligibility: unit-stride + dense + real. `Ptr` inputs come from the C ABI (already raw); dense
# arrays expose a pointer. Complex deliberately excluded (interleaved re/im SIMD is M2 work).
@inline _simd1(::Ptr{T}) where {T<:BlasReal} = true
@inline _simd1(::DenseArray{T}) where {T<:BlasReal} = true
@inline _simd1(@nospecialize(_)) = false
@inline _simd2(::Ptr{T}, ::Ptr{T}) where {T<:BlasReal} = true
@inline _simd2(::DenseArray{T}, ::DenseArray{T}) where {T<:BlasReal} = true
@inline _simd2(@nospecialize(_), @nospecialize(_)) = false

# Complex unit-stride dense/Ptr → the underlying interleaved [re im re im …] buffer IS a contiguous
# 2n-real array. For the two reductions that are grouping-invariant — nrm2 (Σ|xᵢ|² = Σ over 2n reals r² )
# and asum (dzasum = Σ|Re|+|Im| = Σ over 2n reals |r|) — a complex op reduces EXACTLY to the real SIMD
# kernel over that reinterpreted buffer. `_reptr` gives the real Ptr (caller GC.@preserves the array).
@inline _cplx_re(::Ptr{Complex{T}}) where {T<:BlasReal} = true
@inline _cplx_re(::DenseArray{Complex{T}}) where {T<:BlasReal} = true
@inline _cplx_re(@nospecialize(_)) = false
@inline _reptr(x::Ptr{Complex{T}}) where {T<:BlasReal} = Ptr{T}(x)
@inline _reptr(x::DenseArray{Complex{T}}) where {T<:BlasReal} = Ptr{T}(pointer(x))

@inline _ptr(p::Ptr) = p
@inline _ptr(a) = pointer(a)

# Elementwise kernels are 4-way unrolled (4 vectors / iteration) to keep load/store ports busy and
# give the prefetcher a longer stride — a single vector/iteration is throughput-starved in the
# L2-resident regime. `_UNROLL` is defined with the reductions below. Pattern: unrolled body, then
# a W-at-a-time pass, then a scalar tail.
@inline function _axpy_simd!(n::Int, a::T, x, y) where {T<:BlasReal}
    px = _ptr(x); py = _ptr(y); V = _vec(T); W = _vwidth(T); sz = sizeof(T); step = _UNROLL * W
    GC.@preserve x y begin
        va = V(a)
        i = 0
        while i + step <= n
            o = i * sz
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

@inline function _scal_simd!(n::Int, a::T, x) where {T<:BlasReal}
    px = _ptr(x); V = _vec(T); W = _vwidth(T); sz = sizeof(T); step = _UNROLL * W
    GC.@preserve x begin
        va = V(a)
        i = 0
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

@inline function _dot_simd(n::Int, x, y, ::Type{T}) where {T<:BlasReal}
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

@inline function _asum_simd(n::Int, x, ::Type{T}) where {T<:BlasReal}
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
@inline function _sumsq_simd(n::Int, x, ::Type{T}) where {T<:BlasReal}
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
# assumes n ≥ W (caller routes shorter / strided / complex to the scalar loop). Lane-parallel running
# max + parallel index vector; strict `>` updates keep the earliest index per lane. The running max is
# a loop-carried dependency (latency-bound), so it runs 4 independent chains (else ~0.3–0.8× at L1/RAM
# sizes). Tie rule everywhere: equal value → keep the smaller index ⇒ BLAS first-occurrence semantics.
# Gate: median ≥1.06× OpenBLAS at every size (n=64…1e6). NB: OB's idamax is alignment-volatile (its
# time swings ~60% with the array's address) while this kernel is stable — so single-allocation ratios
# are noisy; sample many fresh allocations and take the median. A two-pass (max-only + locate) variant
# was tried and is slower (locate's extra array read costs more than the in-loop index selects save).
# Inner update: new block's indices are always larger than the chain's, so strict `>` keeps the
# earliest index on ties — no index compare needed (cheap, 1 cmp + 2 selects).
@inline _amax_up(m, i, nv, ni) = (t = nv > m; (vifelse(t, nv, m), vifelse(t, ni, i)))
# Fold across chains: indices interleave, so the tie must pick the smaller index explicitly.
@inline _amax_merge(m0, i0, m1, i1) = begin
    take = (m1 > m0) | ((m1 == m0) & (i1 < i0))
    (vifelse(take, m1, m0), vifelse(take, i1, i0))
end
@inline function _iamax_simd!(n::Int, xp::Ptr{T}) where {T<:BlasReal}
    W = _vwidth(T); V = Vec{W,T}; sz = sizeof(T); step = 4W
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
