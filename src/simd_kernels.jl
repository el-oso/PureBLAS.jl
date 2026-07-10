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
const _CplxArg{T} = Union{Ptr{Complex{T}}, DenseArray{Complex{T}}}
@inline _cplx2(::_CplxArg{T}, ::_CplxArg{T}) where {T<:BlasReal} = true      # both complex, same real T
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
@inline function _axpy_simd!(n::Int, a::T, x, y, pf::Int = 0) where {T<:BlasReal}
    px = _ptr(x); py = _ptr(y); V = _vec(T); W = _vwidth(T); sz = sizeof(T); step = _UNROLL * W
    GC.@preserve x y begin
        va = V(a)
        i = 0
        while i + step <= n
            o = i * sz
            if pf > 0                                 # const-folds OFF when pf==0 (default / axpy)
                pb = py + (i + pf) * sz
                for c in 0:_CACHELINE:(step * sz - 1); _prefetch(pb + c); end   # one prefetch per line
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

# Complex scal: x .*= (alr + i·ali). Bandwidth-bound (read-modify-write), so minimise the shuffle chain:
# for a SCALAR multiplier, one swap-adjacent-pairs shuffle suffices — result = v·alr + swap(v)·[−ali,+ali…]
# (= [r·alr − i·ali, i·alr + r·ali, …] on the interleaved [r i r i…] buffer). One shuffle/vector (vs 3 for
# deinterleave+interleave), 4× unrolled to saturate memory bandwidth. `n` counts COMPLEX elements.
@generated function _scal_cmplx_simd!(n::Int, alr::T, ali::T, x) where {T<:BlasReal}
    W = _vwidth(T); V2 = Vec{2W, T}; sz = sizeof(T); Wc = 2 * W          # reals per Vec = 2W; W complex
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(2W - 1))...)     # swap adjacent (re,im)
    sgn = :($V2($(Expr(:tuple, (iseven(l) ? :(-ali) : :ali for l in 0:(2W - 1))...))))  # [−ali,ali,…]
    quote
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

# Complex axpy: y .+= (alr + i·ali)·x. Same swap-pairs complex-multiply of x as scal, then add into y.
# One shuffle/vector, 4× unrolled (bandwidth-bound: reads x+y, writes y). `n` counts COMPLEX elements.
@generated function _axpy_cmplx_simd!(n::Int, alr::T, ali::T, x, y) where {T<:BlasReal}
    W = _vwidth(T); V2 = Vec{2W, T}; sz = sizeof(T)
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(2W - 1))...)
    sgn = :($V2($(Expr(:tuple, (iseven(l) ? :(-ali) : :ali for l in 0:(2W - 1))...))))
    quote
        px = _reptr(x); py = _reptr(y); arv = $V2(alr); sv = $sgn; step = 4 * $W
        GC.@preserve x y begin
            i = 0
            while i + step <= n
                @inbounds for u in 0:3
                    o = (i + u * $W) * 2 * $sz; xv = vload($V2, px + o)
                    ax = muladd(shufflevector(xv, Val($swp)), sv, xv * arv)   # a·x
                    vstore(vload($V2, py + o) + ax, py + o)                    # y += a·x
                end
                i += step
            end
            while i + $W <= n
                o = i * 2 * $sz; xv = vload($V2, px + o)
                ax = muladd(shufflevector(xv, Val($swp)), sv, xv * arv)
                vstore(vload($V2, py + o) + ax, py + o); i += $W
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
@generated function _dot_cmplx_simd(n::Int, x, y, ::Type{T}, ::Val{CJ}) where {T<:BlasReal, CJ}
    W = _vwidth(T); V2 = Vec{2W, T}; sz = sizeof(T)
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(2W - 1))...)
    ps = [Symbol(:p, u) for u in 0:3]; qs = [Symbol(:q, u) for u in 0:3]
    init = Expr(:block, (:( $(ps[u+1]) = zero($V2); $(qs[u+1]) = zero($V2) ) for u in 0:3)...)
    body = Expr(:block)
    for u in 0:3
        o = :((i + $u * $W) * 2 * $sz)
        push!(body.args, quote
            xv = vload($V2, px + $o); yv = vload($V2, py + $o)
            $(ps[u+1]) = muladd(xv, yv, $(ps[u+1])); $(qs[u+1]) = muladd(xv, shufflevector(yv, Val($swp)), $(qs[u+1]))
        end)
    end
    quote
        px = _reptr(x); py = _reptr(y); step = 4 * $W
        $init
        GC.@preserve x y begin
            i = 0
            while i + step <= n
                @inbounds begin $body end
                i += step
            end
            while i + $W <= n
                @inbounds begin
                    xv = vload($V2, px + i * 2 * $sz); yv = vload($V2, py + i * 2 * $sz)
                    p0 = muladd(xv, yv, p0); q0 = muladd(xv, shufflevector(yv, Val($swp)), q0)
                end
                i += $W
            end
            pr, pi = _deint_cmplx((p0 + p1) + (p2 + p3))       # pr = Σxr·yr, pi = Σxi·yi
            qr, qi = _deint_cmplx((q0 + q1) + (q2 + q3))       # qr = Σxr·yi, qi = Σxi·yr
            # dotu: real=Σxr·yr−Σxi·yi, imag=Σxr·yi+Σxi·yr ;  dotc (conj x): signs of the xi terms flip
            sr = sum(pr) + $(CJ ? :(sum(pi)) : :(-sum(pi)))
            si = sum(qr) + $(CJ ? :(-sum(qi)) : :(sum(qi)))
            @inbounds while i < n
                j = i + 1; xr = unsafe_load(px, 2j - 1); xi = unsafe_load(px, 2j); yr = unsafe_load(py, 2j - 1); yi = unsafe_load(py, 2j)
                sr += xr * yr + $(CJ ? :(xi * yi) : :(-xi * yi)); si += xr * yi + $(CJ ? :(-xi * yr) : :(xi * yr))
                i += 1
            end
            return Complex{$T}(sr, si)
        end
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
    :((av = abs(v); av + shufflevector(av, Val($swp))))
end
@inline function _iamax_cmplx_simd!(n::Int, xp::Ptr{T}) where {T<:BlasReal}
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
    :(av + shufflevector(av, Val($swp)))
end
