# Low-level BLAS Level-1 kernels in BLAS-native `(n, …, inc)` form. These are the single shared
# implementation behind both the C-ABI wrappers (cabi.jl) and the native/backend API
# (backend.jl, native.jl). Real unit-stride dense inputs take the SIMD.jl fast path; every other
# `T<:Number` (complex, ForwardDiff.Dual, …) and any strided/negative increment uses the generic
# scalar loop, which is exactly what makes Mode 2 differentiable.

# y .= x
function _copy!(n::Integer, x, incx::Integer, y, incy::Integer)
    n <= 0 && return y
    (incx == 1 && incy == 1 && _simd2(x, y)) && return _copy_simd!(Int(n), x, y)
    ix = _start(n, incx); iy = _start(n, incy)
    @inbounds for _ in 1:n
        _st!(y, iy, _ld(x, ix)); ix += incx; iy += incy
    end
    return y
end

# x ⇄ y
function _swap!(n::Integer, x, incx::Integer, y, incy::Integer)
    n <= 0 && return nothing
    (incx == 1 && incy == 1 && _simd2(x, y)) && return _swap_simd!(Int(n), x, y)
    ix = _start(n, incx); iy = _start(n, incy)
    @inbounds for _ in 1:n
        t = _ld(x, ix); _st!(x, ix, _ld(y, iy)); _st!(y, iy, t)
        ix += incx; iy += incy
    end
    return nothing
end

# x .*= a
function _scal!(n::Integer, a::Number, x, incx::Integer)
    n <= 0 && return x
    (incx == 1 && _simd1(x)) && return _scal_simd!(Int(n), convert(_et(x), a), x)
    ix = _start(n, incx)
    @inbounds for _ in 1:n
        _st!(x, ix, a * _ld(x, ix)); ix += incx
    end
    return x
end

# y .+= a .* x
function _axpy!(n::Integer, a::Number, x, incx::Integer, y, incy::Integer)
    n <= 0 && return y
    (incx == 1 && incy == 1 && _simd2(x, y)) && return _axpy_simd!(Int(n), convert(_et(x), a), x, y)
    ix = _start(n, incx); iy = _start(n, incy)
    @inbounds for _ in 1:n
        _st!(y, iy, muladd(a, _ld(x, ix), _ld(y, iy))); ix += incx; iy += incy
    end
    return y
end

# Σ (conjx ? conj(xᵢ) : xᵢ) · yᵢ
function _dot_generic(n::Integer, x, incx::Integer, y, incy::Integer, conjx::Bool)
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
function _dotu(n::Integer, x, incx::Integer, y, incy::Integer)
    (incx == 1 && incy == 1 && _simd2(x, y)) && return _dot_simd(Int(n), x, y, _et(x))
    return _dot_generic(n, x, incx, y, incy, false)
end

# Conjugated dot (BLAS ?dotc). For real T this equals `_dotu`.
function _dotc(n::Integer, x, incx::Integer, y, incy::Integer)
    (incx == 1 && incy == 1 && _simd2(x, y)) && return _dot_simd(Int(n), x, y, _et(x))
    return _dot_generic(n, x, incx, y, incy, true)
end

# Euclidean norm. Fast path: SIMD sum-of-squares (real, unit-stride dense). If that overflows to
# Inf or underflows to 0 with a nonzero input, fall back to the overflow/underflow-safe scaled
# accumulation (LAPACK lassq) — the correctness boundary. Returns a real scalar.
function _nrm2(n::Integer, x, incx::Integer)
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
    end
    scale = zero(R); ssq = one(R)
    ix = _start(n, incx)
    @inbounds for _ in 1:n
        scale, ssq = _nrm2_acc(scale, ssq, _ld(x, ix)); ix += incx
    end
    return scale * sqrt(ssq)
end

# Σ |xᵢ|  (complex: Σ |Re|+|Im|). Returns a real scalar.
function _asum(n::Integer, x, incx::Integer)
    R = real(_et(x))
    n <= 0 && return zero(R)
    (incx == 1 && _simd1(x)) && return _asum_simd(Int(n), x, _et(x))
    (incx == 1 && _cplx_re(x)) &&                          # dzasum = Σ|Re|+|Im| = asum over the 2n reals
        (GC.@preserve x return _asum_simd(2 * Int(n), _reptr(x), R))
    s = zero(R); ix = _start(n, incx)
    @inbounds for _ in 1:n
        s += _l1(_ld(x, ix)); ix += incx
    end
    return s
end

# 1-based index of the first element maximising |xᵢ| (complex: |Re|+|Im|). 0 if n ≤ 0.
# Real unit-stride → SIMD argmax; everything else (complex, strided, short, other T) → scalar below.
@inline _iamax_simd_try(n::Integer, x) = 0
@inline _iamax_simd_try(n::Integer, x::Ptr{T}) where {T<:BlasReal} =
    n < 4 * _vwidth(T) ? 0 : _iamax_simd!(Int(n), x)
@inline function _iamax_simd_try(n::Integer, x::StridedVector{T}) where {T<:BlasReal}
    (stride(x, 1) == 1 && n >= 4 * _vwidth(T)) || return 0
    GC.@preserve x return _iamax_simd!(Int(n), pointer(x))
end

function _iamax(n::Integer, x, incx::Integer)
    n <= 0 && return 0
    if incx == 1
        v = _iamax_simd_try(n, x)
        v > 0 && return v
    end
    ix = _start(n, incx)
    best = _l1(_ld(x, ix)); bi = 1; ix += incx
    @inbounds for k in 2:n
        v = _l1(_ld(x, ix))
        if v > best
            best = v; bi = k
        end
        ix += incx
    end
    return bi
end
