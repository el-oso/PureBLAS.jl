# Low-level BLAS Level-1 kernels in BLAS-native `(n, …, inc)` form. These are the single shared
# implementation behind both the C-ABI wrappers (cabi.jl) and the native/backend API
# (backend.jl, native.jl). Real unit-stride dense inputs take the SIMD.jl fast path; every other
# `T<:Number` (complex, ForwardDiff.Dual, …) and any strided/negative increment uses the generic
# scalar loop, which is exactly what makes Mode 2 differentiable.

# y .= x
@inline function _copy!(n::Integer, x, incx::Integer, y, incy::Integer)
    n <= 0 && return y
    (incx == 1 && incy == 1 && _simd2(x, y)) && return _copy_simd!(Int(n), x, y)
    # Complex and Dual vectors are contiguous 2n-real buffers and copy involves no arithmetic, so the real
    # SIMD kernel serves them as-is. LLVM does NOT vectorize the scalar loop below for ComplexF64 — it emits
    # a `memmove` call (bench/probes/cplx_copy_vec.jl: `_vectorized=false`), 1.9x slower than `_copy_simd!`
    # while L1-resident (89 vs 170 GB/s at n=1e3) and level from L2 on (within 2% at n=1e4/1e5).
    if incx == 1 && incy == 1 && (_cplx2(x, y) || _pair2(x, y))
        GC.@preserve x y _copy_simd!(2 * Int(n), _pairreal(x), _pairreal(y))
        return y
    end
    ix = _start(n, incx); iy = _start(n, incy)
    @inbounds for _ in 1:n
        _st!(y, iy, _ld(x, ix)); ix += incx; iy += incy
    end
    return y
end

# x ⇄ y
@inline function _swap!(n::Integer, x, incx::Integer, y, incy::Integer)
    n <= 0 && return nothing
    (incx == 1 && incy == 1 && _simd2(x, y)) && return _swap_simd!(Int(n), x, y)
    # Same reasoning as `_copy!`; here the scalar loop lowers to memcpy+memmove through a temporary and runs
    # 2.4-3.1x slower than `_swap_simd!` at every size measured (n=1e3..1e5, same probe).
    if incx == 1 && incy == 1 && (_cplx2(x, y) || _pair2(x, y))
        GC.@preserve x y _swap_simd!(2 * Int(n), _pairreal(x), _pairreal(y))
        return nothing
    end
    ix = _start(n, incx); iy = _start(n, incy)
    @inbounds for _ in 1:n
        t = _ld(x, ix); _st!(x, ix, _ld(y, iy)); _st!(y, iy, t)
        ix += incx; iy += incy
    end
    return nothing
end

# x .*= a
@inline function _scal!(n::Integer, a::Number, x, incx::Integer)
    n <= 0 && return x
    (incx == 1 && _simd1(x)) && return _scal_simd!(Int(n), convert(_et(x), a), x)
    if incx == 1 && _cplx_re(x)
        ac = convert(_et(x), a)
        if iszero(imag(ac))                                # real scalar × complex vec = real scal over 2n
            GC.@preserve x _scal_simd!(2 * Int(n), real(ac), _reptr(x))   # reals (OB fast-paths this too)
            return x
        end
        return _scal_cmplx_simd!(Int(n), real(ac), imag(ac), x)   # true complex → interleaved swap-multiply
    end
    # Dual vector (ForwardDiff extension loaded): a REAL alpha scales value and partial alike, so it is the
    # real scal over the 2n-real buffer — exactly the complex bypass above. A dual alpha (nonzero partial)
    # multiplies element by element and takes the generic loop below (its SIMD body is step 3 of the design).
    if incx == 1 && _pairalg(x)
        av, ap = _parts(convert(_et(x), a))
        if iszero(ap)
            GC.@preserve x _scal_simd!(2 * Int(n), av, _pairreal(x))
            return x
        end
    end
    ix = _start(n, incx)
    @inbounds for _ in 1:n
        _st!(x, ix, a * _ld(x, ix)); ix += incx
    end
    return x
end

# y .+= a .* x
#
# `@inline` is a MEASURED gate lever, not a style choice. The Zen4 BLAS-1 gate had six ops sitting a
# hair under 1.0 (axpy 0.999, dot 0.992, asum 0.990, scal 0.992, zaxpy 0.981, zdotc 0.973) — six
# independent kernel deficits of identical tiny size is implausible; one shared per-call cost is not.
# Decomposing the ladder (bench/probes/axpy_entry.jl, Chairmarks median) named it at n=1e4 on Zen4:
#   ob 1816.5 ns | raw kernel 1801.6 (1.0083 vs OB) | +knob 0.6 | +shape ladder 6.9 | +ENTRY 28.7
# i.e. the kernel already BEATS OpenBLAS and the public wrapper gives the win back. The knob lookup
# was the obvious suspect and is falsified at +0.6 ns. Out-of-line, this call cannot see that its
# arguments are a concrete `Vector{Float64}` with unit strides, so the whole branch chain below
# (`_simd2`/`_cplx2`/stride tests) stays live and the `convert` is a real call; inlined into a
# concrete call site every one of those tests const-folds to the single surviving branch.
@inline function _axpy!(n::Integer, a::Number, x, incx::Integer, y, incy::Integer)
    n <= 0 && return y
    (incx == 1 && incy == 1 && _simd2(x, y)) && return _axpy_simd!(Int(n), convert(_et(x), a), x, y)
    if incx == 1 && incy == 1 && _cplx2(x, y)
        ac = convert(_et(x), a)
        if iszero(imag(ac))                                # real scalar × complex vecs = real axpy over 2n
            GC.@preserve x y _axpy_simd!(2 * Int(n), real(ac), _reptr(x), _reptr(y))   # mirrors _scal!
            return y
        end
        return _axpy_cmplx_simd!(Int(n), real(ac), imag(ac), x, y)   # interleaved-complex SIMD axpy
    end
    # Dual vectors, real alpha: y_v += a·x_v and y_p += a·x_p are one real axpy over the 2n-real buffer
    # (see `_scal!`). A dual alpha runs the complex axpy body under the dual multiply rule (`_pair_shuf`).
    if incx == 1 && incy == 1 && _pair2(x, y)
        av, ap = _parts(convert(_et(x), a))
        if iszero(ap)
            GC.@preserve x y _axpy_simd!(2 * Int(n), av, _pairreal(x), _pairreal(y))
            return y
        end
        return _axpy_pair_simd!(Val(:dual), Int(n), av, ap, x, y)
    end
    ix = _start(n, incx); iy = _start(n, incy)
    @inbounds for _ in 1:n
        _st!(y, iy, muladd(a, _ld(x, ix), _ld(y, iy))); ix += incx; iy += incy
    end
    return y
end

# Σ (conjx ? conj(xᵢ) : xᵢ) · yᵢ
@inline function _dot_generic(n::Integer, x, incx::Integer, y, incy::Integer, conjx::Bool)
    s = zero(_et(x)) * zero(_et(y))
    n <= 0 && return s
    ix = _start(n, incx); iy = _start(n, incy)
    @inbounds for _ in 1:n
        xi = _ld(x, ix); yi = _ld(y, iy)
        s += (conjx ? conj(xi) : xi) * yi
        ix += incx; iy += incy
    end
    return s
end

# Unconjugated dot (BLAS ?dot / ?dotu).
@inline function _dotu(n::Integer, x, incx::Integer, y, incy::Integer)
    (incx == 1 && incy == 1 && _simd2(x, y)) && return _dot_simd(Int(n), x, y, _et(x))
    (incx == 1 && incy == 1 && _cplx2(x, y)) && return _dot_cmplx_simd(Int(n), x, y, real(_et(x)), Val(false))
    # Dual vectors (ForwardDiff extension loaded): the complex dot body under the dual multiply rule; it hands
    # back (value, partial) and `_mkpair` builds the Dual with x's own tag. dotc is the same (no conj on a Real).
    (incx == 1 && incy == 1 && _pair2(x, y)) && return _mkpair(_et(x), _dot_pair_simd(Val(:dual), Int(n), x, y, _pairv(x), Val(false))...)
    return _dot_generic(n, x, incx, y, incy, false)
end

# Conjugated dot (BLAS ?dotc). For real T this equals `_dotu`.
@inline function _dotc(n::Integer, x, incx::Integer, y, incy::Integer)
    (incx == 1 && incy == 1 && _simd2(x, y)) && return _dot_simd(Int(n), x, y, _et(x))
    (incx == 1 && incy == 1 && _cplx2(x, y)) && return _dot_cmplx_simd(Int(n), x, y, real(_et(x)), Val(true))
    (incx == 1 && incy == 1 && _pair2(x, y)) && return _mkpair(_et(x), _dot_pair_simd(Val(:dual), Int(n), x, y, _pairv(x), Val(false))...)
    return _dot_generic(n, x, incx, y, incy, true)
end

# Euclidean norm. Fast path: SIMD sum-of-squares (real, unit-stride dense). If that overflows to
# Inf or underflows to 0 with a nonzero input, fall back to the overflow/underflow-safe scaled
# accumulation (LAPACK lassq) — the correctness boundary. Returns a real scalar.
@inline function _nrm2(n::Integer, x, incx::Integer)
    R = real(_et(x))
    n <= 0 && return zero(R)
    if incx == 1 && _simd1(x)
        ss = _sumsq_simd(Int(n), x, _et(x))
        (isfinite(ss) && !iszero(ss)) && return sqrt(ss)
        # ss is Inf (overflow) or 0 (all-zero, or underflow of tiny values) → use safe path
    elseif incx == 1 && _cplx_re(x)
        GC.@preserve x begin                               # Σ|xᵢ|² over the interleaved 2n-real buffer
            ss = _sumsq_simd(2 * Int(n), _reptr(x), R)
            (isfinite(ss) && !iszero(ss)) && return sqrt(ss)
        end                                                # non-finite/zero → complex lassq fallback below
    elseif incx == 1 && _pairalg(x)
        # Dual: ‖x‖ = √(Σ x_v²) with partial Σ x_v·x_p / ‖x‖ — one `dupEven` FMA kernel gives both sums
        # (simd_kernels.jl). Guarded on BOTH sums: overflow/underflow of either takes the scaled slow path.
        GC.@preserve x begin
            xp = _pairreal(x)
            ss, sp = _sumsq_dual_simd(Int(n), xp)
            if isfinite(ss) && !iszero(ss) && isfinite(sp)
                r = sqrt(ss)
                return _mkpair(_et(x), r, sp / r)
            end
            # OVERFLOW/UNDERFLOW: NOT the generic Dual lassq loop below. The design (docs/src/dual.md) said to
            # fall back to it, and the 1e200-scale test it prescribed showed that loop is numerically wrong
            # there: `_lassq` divides Duals, and ForwardDiff's quotient rule squares the denominator, so at
            # |x| ~ 1e200 the partial is garbage (measured −1.66 against the analytic −1.39; LinearAlgebra's
            # own `norm` over Duals gave −1.47 — the oracle is broken the same way). The value is fine either
            # way; only the derivative is lost. So scale on VALUES only — real lassq, no Dual division — and
            # take the partial as Σ (x_v/scale)·x_p / √ssq, every term of which is bounded by |x_p|.
            return _nrm2_dual_scaled(Int(n), xp, _et(x))
        end
    end
    scale = zero(R); ssq = one(R)
    ix = _start(n, incx)
    @inbounds for _ in 1:n
        scale, ssq = _nrm2_acc(scale, ssq, _ld(x, ix)); ix += incx
    end
    return scale * sqrt(ssq)
end

# Dual nrm2, overflow/underflow-safe slow path (see the note in `_nrm2`). `xp` is the real Ptr over the
# interleaved [v p …] buffer; `D` the vector's Dual type. Two scalar passes: real lassq over the values,
# then the partial Σ (x_v/scale)·x_p / √ssq — no Dual arithmetic, so no Dual quotient rule to overflow.
# All-zero input returns (0, 0), as the generic loop did (`Dual(0,0)·√Dual(1,0)`).
@noinline function _nrm2_dual_scaled(n::Int, xp::Ptr{V}, ::Type{D}) where {V <: BlasReal, D}
    scale = zero(V); ssq = one(V)
    @inbounds for j in 1:n
        scale, ssq = _lassq(scale, ssq, unsafe_load(xp, 2j - 1))
    end
    iszero(scale) && return _mkpair(D, zero(V), zero(V))
    sp = zero(V)
    @inbounds for j in 1:n
        sp = muladd(unsafe_load(xp, 2j - 1) / scale, unsafe_load(xp, 2j), sp)
    end
    rs = sqrt(ssq)
    return _mkpair(D, scale * rs, sp / rs)
end

# Σ |xᵢ|  (complex: Σ |Re|+|Im|). Returns a real scalar.
@inline function _asum(n::Integer, x, incx::Integer)
    R = real(_et(x))
    n <= 0 && return zero(R)
    (incx == 1 && _simd1(x)) && return _asum_simd(Int(n), x, _et(x))
    (incx == 1 && _cplx_re(x)) &&                          # dzasum = Σ|Re|+|Im| = asum over the 2n reals
        (GC.@preserve x return _asum_simd(2 * Int(n), _reptr(x), R))
    if incx == 1 && _pairalg(x)                            # Dual: Σ|x_v| with partial Σ flipsign(x_p, x_v)
        GC.@preserve x begin
            sa, sp = _asum_dual_simd(Int(n), _pairreal(x))
            return _mkpair(_et(x), sa, sp)
        end
    end
    s = zero(R); ix = _start(n, incx)
    @inbounds for _ in 1:n
        s += _l1(_ld(x, ix)); ix += incx
    end
    return s
end

# 1-based index of the first element maximising |xᵢ| (complex: |Re|+|Im|). 0 if n ≤ 0.
# Real unit-stride → SIMD argmax; complex unit-stride → complex SIMD argmax; else (strided, short, other
# T) → scalar below.
# The generic fallback also catches a dense Dual vector (ForwardDiff extension loaded): `_pairalg` is a
# compile-time `false` for every other type, so this stays the `= 0` it always was for them. The dual
# magnitude is |value| in both lanes of the pair (`_pmag2(Val(:dual), …)`) on the complex argmax scaffold.
@inline function _iamax_simd_try(n::Integer, x)
    _pairalg(x) || return 0
    xp = _pairreal(x)
    n < 4 * _vwidth(_et(xp)) && return 0
    GC.@preserve x return _iamax_pair_simd!(Val(:dual), Int(n), xp)
end
@inline _iamax_simd_try(n::Integer, x::Ptr{T}) where {T <: BlasReal} =
    n < 4 * _vwidth(T) ? 0 : _iamax_simd!(Int(n), x)
# Ptr{Complex} — the Mode-1 C-ABI shape. Without this method `icamax_64_`/`izamax_64_` fell through to
# the generic `= 0` above and ran the SCALAR loop, while the identical call on a `StridedVector{Complex}`
# (Mode 2, and everything bench/plots.jl measures) took the SIMD path. A wire-the-fastest-path miss that
# no gate row could see. Same n ≥ 4W guard as the sibling methods.
@inline _iamax_simd_try(n::Integer, x::Ptr{Complex{T}}) where {T <: BlasReal} =
    n < 4 * _vwidth(T) ? 0 : _iamax_cmplx_simd!(Int(n), Ptr{T}(x))
@inline function _iamax_simd_try(n::Integer, x::StridedVector{T}) where {T <: BlasReal}
    (stride(x, 1) == 1 && n >= 4 * _vwidth(T)) || return 0
    GC.@preserve x return _iamax_simd!(Int(n), pointer(x))
end
@inline function _iamax_simd_try(n::Integer, x::StridedVector{Complex{T}}) where {T <: BlasReal}
    (stride(x, 1) == 1 && n >= 4 * _vwidth(T)) || return 0
    GC.@preserve x return _iamax_cmplx_simd!(Int(n), Ptr{T}(pointer(x)))
end
# An arena borrow (`borrow!(s, T, n)`, arena.jl) is a `PtrVector`, which is neither a `Ptr` nor in the
# `StridedVector` union — so a converted routine's scratch would fall to the generic `= 0` above and run
# the SCALAR argmax, where the `view(ws.field, 1:n)` it replaces took the StridedVector method. That is
# the izamax miss recorded three lines up, re-arriving through the workspace refactor. ONE forwarding
# method rather than a real/complex pair: `pointer` re-dispatches onto whichever `Ptr` method fits, and
# a non-`BlasReal` element type still lands on the generic `= 0`. No `GC.@preserve` — a PtrVector's
# buffer is owned by the arena slab (or by a C caller), never by this argument.
@inline _iamax_simd_try(n::Integer, x::PtrVector) = _iamax_simd_try(n, pointer(x))

@inline function _iamax(n::Integer, x, incx::Integer)
    n <= 0 && return 0
    if incx == 1
        v = _iamax_simd_try(n, x)
        v > 0 && return v
    end
    ix = _start(n, incx)
    best = _l1v(_ld(x, ix)); bi = 1; ix += incx           # `_l1v`: |value| for Dual, `_l1` otherwise
    @inbounds for k in 2:n
        v = _l1v(_ld(x, ix))
        if v > best
            best = v; bi = k
        end
        ix += incx
    end
    return bi
end
