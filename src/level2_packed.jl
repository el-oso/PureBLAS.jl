# BLAS Level-2 packed storage: spmv/hpmv (symmetric/Hermitian) and tpmv/tpsv (triangular mul/solve).
# The triangle is stored linearly in a vector AP. Crucially each *column* is a CONTIGUOUS segment, so
# these reduce to the same per-column kernels as the full versions (`_symv_col!`, `_axpy_simd!`,
# `_dot_simd`) — only the column base pointer changes (packed offset instead of (j-1)·lda). No 2D
# array ⇒ no blocked-gemv reuse, so the SIMD path is per-column (packed is a memory-saving niche
# format; OpenBLAS's packed routines are likewise unblocked). Real dense unit-stride → SIMD; complex /
# strided / Dual → generic (AD-traceable).
#
# Packed layout (1-based AP index of A[i,j]):
#   uplo='U' (i≤j): col j is A[1:j, j] at AP[ _pkU(j)+1 : _pkU(j)+j ],   _pkU(j) = j(j-1)/2
#   uplo='L' (i≥j): col j is A[j:n, j] at AP[ _pkL(j,n)+1 : _pkL(j,n)+(n-j+1) ], _pkL(j,n)=(j-1)(2n-j+2)/2

@inline _pkU(j::Int) = (j * (j - 1)) >>> 1
@inline _pkL(j::Int, n::Int) = ((j - 1) * (2n - j + 2)) >>> 1

@inline function _pk_simd_ok(AP, x, incx::Integer)
    T = eltype(AP)
    return incx == 1 && T <: BlasReal && eltype(x) === T &&
        AP isa StridedVector && stride(AP, 1) == 1 && x isa StridedVector && stride(x, 1) == 1
end
@inline function _pk2_simd_ok(AP, x, y, incx::Integer, incy::Integer)
    T = eltype(AP)
    return incx == 1 && incy == 1 && T <: BlasReal && eltype(x) === T && eltype(y) === T &&
        AP isa StridedVector && stride(AP, 1) == 1 &&
        x isa StridedVector && stride(x, 1) == 1 && y isa StridedVector && stride(y, 1) == 1
end

# ── spmv: y := α·A·x + β·y, A symmetric packed ─────────────────────────────────────────────────
@inline function _spmv_simd!(up::Bool, n::Int, α::T, AP, x, y) where {T<:BlasReal}
    GC.@preserve AP x y begin
        Ap = pointer(AP); xp = pointer(x); yp = pointer(y); sz = sizeof(T)
        @inbounds for j in 1:n
            axj = α * unsafe_load(xp, j)
            if up                                  # col j = A[1:j, j]; diag last
                cp = Ap + _pkU(j) * sz
                s = _symv_col!(j - 1, axj, cp, xp, yp)
                ajj = unsafe_load(cp + (j - 1) * sz)
            else                                   # col j = A[j:n, j]; diag first
                cp = Ap + _pkL(j, n) * sz
                ajj = unsafe_load(cp)
                s = _symv_col!(n - j, axj, cp + sz, xp + j * sz, yp + j * sz)
            end
            unsafe_store!(yp, unsafe_load(yp, j) + axj * ajj + α * s, j)
        end
    end
    return y
end

function _spmv!(up::Bool, n::Integer, α::Number, AP, x, incx::Integer, β::Number, y, incy::Integer)
    _scale_y!(Int(n), β, y, incy)
    iszero(α) && return y
    if _pk2_simd_ok(AP, x, y, incx, incy)
        return _spmv_simd!(up, Int(n), convert(eltype(AP), α), AP, x, y)
    end
    n = Int(n); sx = _start(n, incx); sy = _start(n, incy); s0 = zero(_et(AP)) * zero(_et(x))
    @inbounds for j in 1:n
        tmp = α * _ld(x, sx + (j - 1) * incx); s = s0
        base = up ? _pkU(j) : _pkL(j, n)
        rng = up ? (1:(j - 1)) : ((j + 1):n)
        for i in rng
            aij = _ld(AP, up ? base + i : base + (i - j) + 1)
            _st!(y, sy + (i - 1) * incy, _ld(y, sy + (i - 1) * incy) + tmp * aij)
            s += aij * _ld(x, sx + (i - 1) * incx)
        end
        ajj = _ld(AP, up ? base + j : base + 1)
        _st!(y, sy + (j - 1) * incy, _ld(y, sy + (j - 1) * incy) + tmp * ajj + α * s)
    end
    return y
end

# ── hpmv: y := α·A·x + β·y, A Hermitian packed (complex; generic path, real diagonal) ───────────
function _hpmv!(up::Bool, n::Integer, α::Number, AP, x, incx::Integer, β::Number, y, incy::Integer)
    _scale_y!(Int(n), β, y, incy)
    iszero(α) && return y
    n = Int(n); sx = _start(n, incx); sy = _start(n, incy); s0 = zero(_et(AP)) * zero(_et(x))
    @inbounds for j in 1:n
        tmp = α * _ld(x, sx + (j - 1) * incx); s = s0
        base = up ? _pkU(j) : _pkL(j, n)
        rng = up ? (1:(j - 1)) : ((j + 1):n)
        for i in rng
            aij = _ld(AP, up ? base + i : base + (i - j) + 1)
            _st!(y, sy + (i - 1) * incy, _ld(y, sy + (i - 1) * incy) + tmp * aij)
            s += conj(aij) * _ld(x, sx + (i - 1) * incx)
        end
        ajj = _ld(AP, up ? base + j : base + 1)
        _st!(y, sy + (j - 1) * incy, _ld(y, sy + (j - 1) * incy) + tmp * real(ajj) + α * s)
    end
    return y
end

# ── tpmv: x := op(A)·x, A triangular packed ────────────────────────────────────────────────────
@inline function _tpmv_simd!(up::Bool, tr::Bool, unit::Bool, n::Int, AP, x) where {}
    T = eltype(AP)
    GC.@preserve AP x begin
        Ap = pointer(AP); xp = pointer(x); sz = sizeof(T)
        if !tr
            if up                                  # U,N ascending
                @inbounds for j in 1:n
                    cp = Ap + _pkU(j) * sz; t = unsafe_load(xp, j)
                    _axpy_simd!(j - 1, t, cp, xp)
                    unit || unsafe_store!(xp, t * unsafe_load(cp + (j - 1) * sz), j)
                end
            else                                   # L,N descending
                @inbounds for j in n:-1:1
                    cp = Ap + _pkL(j, n) * sz; t = unsafe_load(xp, j)
                    _axpy_simd!(n - j, t, cp + sz, xp + j * sz)
                    unit || unsafe_store!(xp, t * unsafe_load(cp), j)
                end
            end
        else
            if up                                  # U,T descending
                @inbounds for j in n:-1:1
                    cp = Ap + _pkU(j) * sz; xj = unsafe_load(xp, j)
                    s = _dot_simd(j - 1, cp, xp, T)
                    unsafe_store!(xp, (unit ? xj : xj * unsafe_load(cp + (j - 1) * sz)) + s, j)
                end
            else                                   # L,T ascending
                @inbounds for j in 1:n
                    cp = Ap + _pkL(j, n) * sz; xj = unsafe_load(xp, j)
                    s = _dot_simd(n - j, cp + sz, xp + j * sz, T)
                    unsafe_store!(xp, (unit ? xj : xj * unsafe_load(cp)) + s, j)
                end
            end
        end
    end
    return x
end

function _tpmv!(up::Bool, tr::Bool, cj::Bool, unit::Bool, n::Integer, AP, x, incx::Integer)
    if _pk_simd_ok(AP, x, incx)
        return _tpmv_simd!(up, tr, unit, Int(n), AP, x)
    end
    n = Int(n); sx = _start(n, incx)
    el = (i, j, base) -> (v = _ld(AP, up ? base + i : base + (i - j) + 1); cj ? conj(v) : v)
    dg = (j, base) -> (v = _ld(AP, up ? base + j : base + 1); cj ? conj(v) : v)
    if !tr
        if up
            @inbounds for j in 1:n
                base = _pkU(j); xj = _ld(x, sx + (j - 1) * incx)
                for i in 1:(j - 1); _st!(x, sx + (i - 1) * incx, _ld(x, sx + (i - 1) * incx) + xj * _ld(AP, base + i)); end
                unit || _st!(x, sx + (j - 1) * incx, xj * _ld(AP, base + j))
            end
        else
            @inbounds for j in n:-1:1
                base = _pkL(j, n); xj = _ld(x, sx + (j - 1) * incx)
                for i in n:-1:(j + 1); _st!(x, sx + (i - 1) * incx, _ld(x, sx + (i - 1) * incx) + xj * _ld(AP, base + (i - j) + 1)); end
                unit || _st!(x, sx + (j - 1) * incx, xj * _ld(AP, base + 1))
            end
        end
    else
        if up
            @inbounds for j in n:-1:1
                base = _pkU(j); s = unit ? _ld(x, sx + (j - 1) * incx) : _ld(x, sx + (j - 1) * incx) * dg(j, base)
                for i in 1:(j - 1); s += el(i, j, base) * _ld(x, sx + (i - 1) * incx); end
                _st!(x, sx + (j - 1) * incx, s)
            end
        else
            @inbounds for j in 1:n
                base = _pkL(j, n); s = unit ? _ld(x, sx + (j - 1) * incx) : _ld(x, sx + (j - 1) * incx) * dg(j, base)
                for i in (j + 1):n; s += el(i, j, base) * _ld(x, sx + (i - 1) * incx); end
                _st!(x, sx + (j - 1) * incx, s)
            end
        end
    end
    return x
end

# ── tpsv: x := op(A)⁻¹·x, A triangular packed (solve) ──────────────────────────────────────────
@inline function _tpsv_simd!(up::Bool, tr::Bool, unit::Bool, n::Int, AP, x) where {}
    T = eltype(AP)
    GC.@preserve AP x begin
        Ap = pointer(AP); xp = pointer(x); sz = sizeof(T)
        if !tr
            if up                                  # U,N back: j descending
                @inbounds for j in n:-1:1
                    cp = Ap + _pkU(j) * sz
                    unit || unsafe_store!(xp, unsafe_load(xp, j) / unsafe_load(cp + (j - 1) * sz), j)
                    _axpy_simd!(j - 1, -unsafe_load(xp, j), cp, xp)
                end
            else                                   # L,N forward: j ascending
                @inbounds for j in 1:n
                    cp = Ap + _pkL(j, n) * sz
                    unit || unsafe_store!(xp, unsafe_load(xp, j) / unsafe_load(cp), j)
                    _axpy_simd!(n - j, -unsafe_load(xp, j), cp + sz, xp + j * sz)
                end
            end
        else
            if up                                  # U,T forward: j ascending
                @inbounds for j in 1:n
                    cp = Ap + _pkU(j) * sz
                    t = unsafe_load(xp, j) - _dot_simd(j - 1, cp, xp, T)
                    unit || (t /= unsafe_load(cp + (j - 1) * sz))
                    unsafe_store!(xp, t, j)
                end
            else                                   # L,T back: j descending
                @inbounds for j in n:-1:1
                    cp = Ap + _pkL(j, n) * sz
                    t = unsafe_load(xp, j) - _dot_simd(n - j, cp + sz, xp + j * sz, T)
                    unit || (t /= unsafe_load(cp))
                    unsafe_store!(xp, t, j)
                end
            end
        end
    end
    return x
end

function _tpsv!(up::Bool, tr::Bool, cj::Bool, unit::Bool, n::Integer, AP, x, incx::Integer)
    if _pk_simd_ok(AP, x, incx)
        return _tpsv_simd!(up, tr, unit, Int(n), AP, x)
    end
    n = Int(n); sx = _start(n, incx)
    el = (i, j, base) -> (v = _ld(AP, up ? base + i : base + (i - j) + 1); cj ? conj(v) : v)
    dg = (j, base) -> (v = _ld(AP, up ? base + j : base + 1); cj ? conj(v) : v)
    if !tr
        if up
            @inbounds for j in n:-1:1
                base = _pkU(j)
                unit || _st!(x, sx + (j - 1) * incx, _ld(x, sx + (j - 1) * incx) / _ld(AP, base + j))
                xj = _ld(x, sx + (j - 1) * incx)
                for i in 1:(j - 1); _st!(x, sx + (i - 1) * incx, _ld(x, sx + (i - 1) * incx) - xj * _ld(AP, base + i)); end
            end
        else
            @inbounds for j in 1:n
                base = _pkL(j, n)
                unit || _st!(x, sx + (j - 1) * incx, _ld(x, sx + (j - 1) * incx) / _ld(AP, base + 1))
                xj = _ld(x, sx + (j - 1) * incx)
                for i in (j + 1):n; _st!(x, sx + (i - 1) * incx, _ld(x, sx + (i - 1) * incx) - xj * _ld(AP, base + (i - j) + 1)); end
            end
        end
    else
        if up
            @inbounds for j in 1:n
                base = _pkU(j); s = _ld(x, sx + (j - 1) * incx)
                for i in 1:(j - 1); s -= el(i, j, base) * _ld(x, sx + (i - 1) * incx); end
                unit || (s /= dg(j, base))
                _st!(x, sx + (j - 1) * incx, s)
            end
        else
            @inbounds for j in n:-1:1
                base = _pkL(j, n); s = _ld(x, sx + (j - 1) * incx)
                for i in (j + 1):n; s -= el(i, j, base) * _ld(x, sx + (i - 1) * incx); end
                unit || (s /= dg(j, base))
                _st!(x, sx + (j - 1) * incx, s)
            end
        end
    end
    return x
end
