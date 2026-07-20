# BLAS Level-2 band storage: gbmv (general), sbmv/hbmv (symmetric/Hermitian), tbmv/tbsv (triangular
# mul/solve). A's band is packed column-by-column into AB (column-major, leading dim ldb). Each AB
# column holds that matrix-column's band entries CONTIGUOUSLY, so these reduce to the per-column
# kernels (`_axpy_simd!`, `_dot_simd`, `_symv_col!`) over the band segment. Banded work is O(n·band)
# (cheap); real dense unit-stride → SIMD, else generic (complex/strided/AD).
#
# AB layout (1-based AB row of A[i,j]):
#   gbmv (kl sub, ku super):        AB[ku+1+i-j, j]   for max(1,j-ku) ≤ i ≤ min(m,j+kl)
#   uplo='U' band (k super-diags):  AB[k+1+i-j, j]    for max(1,j-k) ≤ i ≤ j      (diag at AB[k+1,j])
#   uplo='L' band (k sub-diags):    AB[1+i-j, j]      for j ≤ i ≤ min(n,j+k)      (diag at AB[1,j])

# ── gbmv: y := α·op(A)·x + β·y, A general banded (m×n) ──────────────────────────────────────────
# gbmv-N: band ≤ CONV_MAX → convolution kernel (keeps y in registers, re-reads AB ~band/W times via
# masked loads); wider band → per-column axpy (reads AB once, re-streams cache-resident y). On W=8 the
# masked conv wins up to band 48; on AVX2 (W=4) the masked-load conv loses to axpy above band ~17
# (measured crossover conv 1.05→axpy 1.09 at band 25, stable across n=256…4096). Overridable per machine.
const _GBMV_CONV_MAX = @load_preference("gbmv_conv_max", _vwidth(Float64) == 4 ? 20 : 48)::Int

# gbmv-N "convolution" kernel: tile the OUTPUT y into W-row blocks kept in ONE register (no y-window
# re-stream, unlike per-column which re-reads the overlapping window ~band times). For each column
# touching the block, FMA a contiguous masked AB-segment scaled by x[j]. Best for narrow/medium band
# (A is re-read ~band/W times — cheap there; per-column wins once band ≫ W). y must be pre-scaled by β.
@inline function _gbmv_conv!(m::Int, n::Int, kl::Int, ku::Int, α::T, AB, x, y) where {T <: BlasReal}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T); b = kl + ku + 1
    lanes = Vec{W, Int}(ntuple(l -> l - 1, Val(W)))
    GC.@preserve AB x y begin
        Ap = pointer(AB); xp = pointer(x); yp = pointer(y); ldb = stride(AB, 2)
        i0 = 0
        @inbounds while i0 < m
            mm = min(W, m - i0); orow = lanes < mm
            acc = vload(V, yp + i0 * sz, orow)              # pre-scaled y-block (β·y)
            jlo = max(1, i0 + 1 - kl); jhi = min(n, i0 + mm + ku)
            for j in jlo:jhi
                r1 = ku + i0 + 2 - j                        # AB row (1-based) for output row i0+1
                msk = orow & (lanes >= (1 - r1)) & (lanes <= (b - r1))
                av = vload(V, Ap + ((r1 - 1) + (j - 1) * ldb) * sz, msk)
                acc = muladd(V(α * unsafe_load(xp, j)), av, acc)
            end
            vstore(acc, yp + i0 * sz, orow)
            i0 += W
        end
    end
    return y
end

# gbmv-T narrow band (band < SIMD width): scalar-accumulate per-column dot. `_dot_simd` forces a
# 4-accumulator + horizontal-sum-of-zeros that's pure overhead for a sub-W dot; letting LLVM size the
# reduction to the (tiny) band — scalar, no horizontal sum — is faster. (Wider band → `_dot_simd`.)
# gbmv-N (y pre-scaled by β): conv kernel for narrow band, per-column axpy for wide.
@inline function _gbmv_n_simd!(m::Int, n::Int, kl::Int, ku::Int, α::T, AB, x, y) where {T <: BlasReal}
    (kl + ku + 1) <= _GBMV_CONV_MAX && return _gbmv_conv!(m, n, kl, ku, α, AB, x, y)
    GC.@preserve AB x y begin
        Ap = pointer(AB); xp = pointer(x); yp = pointer(y); ldb = stride(AB, 2); sz = sizeof(T)
        @inbounds for j in 1:n
            ilo = max(1, j - ku); ihi = min(m, j + kl); len = ihi - ilo + 1
            len <= 0 && continue
            segp = Ap + ((ku + ilo - j) + (j - 1) * ldb) * sz
            _axpy_simd!(len, α * unsafe_load(xp, j), segp, yp + (ilo - 1) * sz)
        end
    end
    return y
end
# Complex gbmv-N: each column is a complex axpy of α·x[j] into the band segment of y (mirrors the real
# kernel with `_axpy_cmplx_simd!`). y pre-scaled by β outside; kernel accumulates. `sz` is the COMPLEX
# element size (band segments are contiguous complex runs); pointers reinterpreted to the real type.
@inline function _gbmv_n_cmplx_simd!(m::Int, n::Int, kl::Int, ku::Int, α::T, AB, x, y) where {T <: BlasComplex}
    GC.@preserve AB x y begin
        Ap = pointer(AB); xp = pointer(x); yp = pointer(y)   # Ptr{Complex} (kernel reinterprets via _reptr)
        ldb = stride(AB, 2); sz = sizeof(T)                  # complex element size (bytes)
        @inbounds for j in 1:n
            ilo = max(1, j - ku); ihi = min(m, j + kl); len = ihi - ilo + 1
            len <= 0 && continue
            c = α * unsafe_load(xp, j)                        # complex scalar α·x[j]
            (real(c) == 0 && imag(c) == 0) && continue
            segp = Ap + ((ku + ilo - j) + (j - 1) * ldb) * sz
            _axpy_cmplx_simd!(len, real(c), imag(c), segp, yp + (ilo - 1) * sz)
        end
    end
    return y
end

# One gbmv-T column dot (band of column j against x). Plain function (not a closure) so the 4-column
# unroll below stays allocation-free.
@inline function _gbt_dot(Ap::Ptr{T}, xp::Ptr{T}, ldb::Int, sz::Int, m::Int, kl::Int, ku::Int, j::Int) where {T <: BlasReal}
    ilo = max(1, j - ku); ihi = min(m, j + kl)
    return ihi < ilo ? zero(T) : _dot_simd(ihi - ilo + 1, Ap + ((ku + ilo - j) + (j - 1) * ldb) * sz, xp + (ilo - 1) * sz, T)
end

# One gbmv-T column j with β fused.
@inline function _gbt_one!(yp::Ptr{T}, Ap::Ptr{T}, xp::Ptr{T}, ldb::Int, sz::Int, m::Int, kl::Int, ku::Int, α::T, β::T, j::Int) where {T <: BlasReal}
    s = _gbt_dot(Ap, xp, ldb, sz, m, kl, ku, j)
    unsafe_store!(yp, (iszero(β) ? zero(T) : β * unsafe_load(yp, j)) + α * s, j)
    return nothing
end

# gbmv-T convolution block (BLASFEO-style): W full-band interior columns j..j+W-1 at once. Load the
# overlapping x super-window ONCE per band-chunk; each column's W-slice is a register SHIFT
# (shufflevector by immediate c) — x is reused from registers, never re-read (the per-column dot's
# weakness at wide band). AB streamed contiguous. Caller guarantees the super-window is in bounds
# (columns ku+1 … n-kl-2W). β fused. Last band-chunk masked (band not a multiple of W).
@generated function _gbmv_t_conv_block!(
        yp::Ptr{T}, Ap::Ptr{T}, xp::Ptr{T}, ldb::Int, sz::Int,
        b::Int, ku::Int, α::T, β::T, j::Int, ::Val{W}
    ) where {T, W}
    V = Vec{W, T}
    body = quote end
    for c in 0:(W - 1)
        push!(body.args, :($(Symbol(:acc, c)) = zero($V)))
    end
    fb = quote end
    push!(fb.args, :(xb = (j - ku - 1) + p))
    push!(fb.args, :(sv0 = vload($V, xp + xb * sz)))
    push!(fb.args, :(sv1 = vload($V, xp + (xb + $W) * sz)))
    for c in 0:(W - 1)
        sh = Expr(:tuple, (c:(c + W - 1))...)
        push!(fb.args, :($(Symbol(:a, c)) = vload($V, Ap + (p + (j + $c - 1) * ldb) * sz)))
        push!(fb.args, :($(Symbol(:acc, c)) = muladd($(Symbol(:a, c)), shufflevector(sv0, sv1, Val($sh)), $(Symbol(:acc, c)))))
    end
    push!(
        body.args, :(
            nf = b - rem(b, $W); p = 0; while p < nf
                $fb; p += $W
            end
        )
    )
    mb = quote end
    push!(mb.args, :(lanes = Vec{$W, Int}($(Expr(:tuple, (0:(W - 1))...)))))
    push!(mb.args, :(msk = lanes < (b - nf)))
    push!(mb.args, :(xb = (j - ku - 1) + nf))
    push!(mb.args, :(sv0 = vload($V, xp + xb * sz)))
    push!(mb.args, :(sv1 = vload($V, xp + (xb + $W) * sz)))
    for c in 0:(W - 1)
        sh = Expr(:tuple, (c:(c + W - 1))...)
        push!(mb.args, :($(Symbol(:a, c)) = vload($V, Ap + (nf + (j + $c - 1) * ldb) * sz, msk)))
        push!(mb.args, :($(Symbol(:acc, c)) = muladd($(Symbol(:a, c)), shufflevector(sv0, sv1, Val($sh)), $(Symbol(:acc, c)))))
    end
    push!(
        body.args, :(
            if rem(b, $W) != 0
                $mb
            end
        )
    )
    for c in 0:(W - 1)
        push!(body.args, :(unsafe_store!(yp, (iszero(β) ? zero($T) : β * unsafe_load(yp, j + $c)) + α * sum($(Symbol(:acc, c))), j + $c)))
    end
    push!(body.args, :(return nothing))
    return body
end

# gbmv-T with β FUSED (overwrite for β=0, no separate _scale_y! pass — that pass is ~1/band of the
# work, dominant for narrow band). Narrow band (< W): scalar dot (no horizontal sum). Wider: _dot_simd.
@inline function _gbmv_t_simd!(m::Int, n::Int, kl::Int, ku::Int, α::T, AB, x, β::T, y) where {T <: BlasReal}
    GC.@preserve AB x y begin
        Ap = pointer(AB); xp = pointer(x); yp = pointer(y); ldb = stride(AB, 2); sz = sizeof(T)
        if (kl + ku + 1) < _vwidth(T)
            @inbounds for j in 1:n
                ilo = max(1, j - ku); ihi = min(m, j + kl); base = (ku + ilo - j) + (j - 1) * ldb; s = zero(T)
                for i in ilo:ihi
                    s = muladd(unsafe_load(Ap, base + (i - ilo) + 1), unsafe_load(xp, i), s)
                end
                unsafe_store!(yp, (iszero(β) ? zero(T) : β * unsafe_load(yp, j)) + α * s, j)
            end
        else   # wide band: conv (BLASFEO-style x-reuse) on full-band interior, per-column elsewhere
            W = _vwidth(T); b = kl + ku + 1; clo = ku + 1; chi = (m - kl) - 2W
            j = 1
            @inbounds if chi >= clo + W - 1     # at least one conv block fits in-bounds
                while j < clo
                    _gbt_one!(yp, Ap, xp, ldb, sz, m, kl, ku, α, β, j); j += 1
                end
                while j + W - 1 <= chi
                    _gbmv_t_conv_block!(yp, Ap, xp, ldb, sz, b, ku, α, β, j, Val(W)); j += W
                end
            end
            @inbounds while j <= n
                _gbt_one!(yp, Ap, xp, ldb, sz, m, kl, ku, α, β, j); j += 1
            end
        end
    end
    return y
end

function _gbmv!(tr::Bool, cj::Bool, m::Integer, n::Integer, kl::Integer, ku::Integer, α::Number, AB, x, incx::Integer, β::Number, y, incy::Integer)
    m = Int(m); n = Int(n); kl = Int(kl); ku = Int(ku)
    ylen = tr ? n : m
    if !cj && _l2_simd_ok(AB, x, y, incx, incy)
        αT = convert(eltype(AB), α); βT = convert(eltype(AB), β)
        if tr
            return _gbmv_t_simd!(m, n, kl, ku, αT, AB, x, βT, y)         # β fused (no pre-scale pass)
        else
            _scale_y!(ylen, βT, y, 1); iszero(α) && return y
            return _gbmv_n_simd!(m, n, kl, ku, αT, AB, x, y)            # y pre-scaled, kernel accumulates
        end
    elseif !tr && !cj && _l2c_simd_ok(AB, x, y, incx, incy)              # complex non-trans → complex axpy band
        αT = convert(eltype(AB), α); βT = convert(eltype(AB), β)
        _scale_y!(ylen, βT, y, 1); iszero(α) && return y
        return _gbmv_n_cmplx_simd!(m, n, kl, ku, αT, AB, x, y)
    end
    _scale_y!(ylen, β, y, incy)
    iszero(α) && return y
    sx = _start(tr ? m : n, incx); sy = _start(ylen, incy); s0 = zero(_et(AB)) * zero(_et(x))
    @inbounds for j in 1:n
        ilo = max(1, j - ku); ihi = min(m, j + kl)
        if !tr
            tmp = α * _ld(x, sx + (j - 1) * incx)
            for i in ilo:ihi
                aij = AB[ku + 1 + i - j, j]
                _st!(y, sy + (i - 1) * incy, _ld(y, sy + (i - 1) * incy) + tmp * aij)
            end
        else
            s = s0
            for i in ilo:ihi
                aij = AB[ku + 1 + i - j, j]; aij = cj ? conj(aij) : aij
                s += aij * _ld(x, sx + (i - 1) * incx)
            end
            _st!(y, sy + (j - 1) * incy, _ld(y, sy + (j - 1) * incy) + α * s)
        end
    end
    return y
end

# ── sbmv: y := α·A·x + β·y, A symmetric banded (k diagonals on `up` side) ───────────────────────
@inline function _sbmv_simd!(up::Bool, n::Int, k::Int, α::T, AB, x, y) where {T <: BlasReal}
    GC.@preserve AB x y begin
        Ap = pointer(AB); xp = pointer(x); yp = pointer(y); ldb = stride(AB, 2); sz = sizeof(T)
        @inbounds for j in 1:n
            axj = α * unsafe_load(xp, j)
            if up                                       # band A[max(1,j-k):j, j]; diag AB[k+1,j]
                ilo = max(1, j - k); len = j - ilo
                ajj = unsafe_load(Ap + (k + (j - 1) * ldb) * sz)
                segp = Ap + ((k + ilo - j) + (j - 1) * ldb) * sz       # AB[k+1+ilo-j, j]
                s = _symv_col!(len, axj, segp, xp + (ilo - 1) * sz, yp + (ilo - 1) * sz)
            else                                        # band A[j:min(n,j+k), j]; diag AB[1,j]
                ihi = min(n, j + k); len = ihi - j
                ajj = unsafe_load(Ap + (j - 1) * ldb * sz)
                segp = Ap + (1 + (j - 1) * ldb) * sz                   # AB[2, j]
                s = _symv_col!(len, axj, segp, xp + j * sz, yp + j * sz)
            end
            unsafe_store!(yp, unsafe_load(yp, j) + axj * ajj + α * s, j)
        end
    end
    return y
end

# Complex Hermitian banded mv: the banded analog of _hemv_cmplx! — per column, run the fused two-sided
# _hemv_col_cmplx! over the (contiguous) band segment, then add the real-diagonal + conj-dot term. β·y
# pre-scaled by the driver. `up`: band above diag stored AB[k+1+i-j,j], diag AB[k+1,j]; `lo`: mirror.
@inline function _hbmv_cmplx_simd!(up::Bool, n::Int, k::Int, α::T, AB, x, y) where {T <: BlasComplex}
    Tr = real(T)
    GC.@preserve AB x y begin
        Ap = Ptr{Tr}(pointer(AB)); xp = Ptr{Tr}(pointer(x)); yp = Ptr{Tr}(pointer(y))
        Apc = pointer(AB); xpc = pointer(x); ypc = pointer(y)     # Ptr{Complex}
        ldb = stride(AB, 2); szr = sizeof(Tr)
        @inbounds for j in 1:n
            tmp = α * unsafe_load(xpc, j); sr = zero(Tr); si = zero(Tr)
            if up
                ilo = max(1, j - k); L = j - ilo
                if L > 0
                    off = (k + ilo - j + (j - 1) * ldb) * 2; seg = (ilo - 1) * 2        # AB[k+1+ilo-j,j]; x/y[ilo]
                    sr, si = _hemv_col_cmplx!(L, real(tmp), imag(tmp), Ap + off * szr, xp + seg * szr, yp + seg * szr)
                end
                ajj = unsafe_load(Ap, (k + (j - 1) * ldb) * 2 + 1)                       # real(AB[k+1,j])
            else
                ihi = min(n, j + k); L = ihi - j
                if L > 0
                    off = (1 + (j - 1) * ldb) * 2; seg = j * 2                           # AB[2,j]; x/y[j+1]
                    sr, si = _hemv_col_cmplx!(L, real(tmp), imag(tmp), Ap + off * szr, xp + seg * szr, yp + seg * szr)
                end
                ajj = unsafe_load(Ap, ((j - 1) * ldb) * 2 + 1)                           # real(AB[1,j])
            end
            unsafe_store!(ypc, unsafe_load(ypc, j) + tmp * ajj + α * Complex{Tr}(sr, si), j)
        end
    end
    return y
end

# Generic symmetric/Hermitian banded (herm=true ⇒ conj + real diagonal). Covers sbmv complex & hbmv.
function _sbmv_generic!(up::Bool, herm::Bool, n::Int, k::Int, α::Number, AB, x, incx::Integer, β::Number, y, incy::Integer)
    _scale_y!(n, β, y, incy)
    iszero(α) && return y
    sx = _start(n, incx); sy = _start(n, incy); s0 = zero(_et(AB)) * zero(_et(x))
    @inbounds for j in 1:n
        tmp = α * _ld(x, sx + (j - 1) * incx); s = s0
        rng = up ? (max(1, j - k):(j - 1)) : ((j + 1):min(n, j + k))
        for i in rng
            aij = up ? AB[k + 1 + i - j, j] : AB[1 + i - j, j]
            _st!(y, sy + (i - 1) * incy, _ld(y, sy + (i - 1) * incy) + tmp * aij)
            s += (herm ? conj(aij) : aij) * _ld(x, sx + (i - 1) * incx)
        end
        ajj = up ? AB[k + 1, j] : AB[1, j]
        _st!(y, sy + (j - 1) * incy, _ld(y, sy + (j - 1) * incy) + tmp * (herm ? real(ajj) : ajj) + α * s)
    end
    return y
end

function _sbmv!(up::Bool, n::Integer, k::Integer, α::Number, AB, x, incx::Integer, β::Number, y, incy::Integer)
    n = Int(n); k = Int(k)
    if _l2_simd_ok(AB, x, y, incx, incy)
        _scale_y!(n, β, y, incy)
        iszero(α) && return y
        return _sbmv_simd!(up, n, k, convert(eltype(AB), α), AB, x, y)
    end
    return _sbmv_generic!(up, false, n, k, α, AB, x, incx, β, y, incy)
end
function _hbmv!(up, n, k, α, AB, x, incx, β, y, incy)
    if _l2c_simd_ok(AB, x, y, incx, incy)                        # complex contiguous → fused hemv-col band kernel
        _scale_y!(Int(n), convert(eltype(AB), β), y, 1); iszero(α) && return y
        return _hbmv_cmplx_simd!(up, Int(n), Int(k), convert(eltype(AB), α), AB, x, y)
    end
    return _sbmv_generic!(up, true, Int(n), Int(k), α, AB, x, incx, β, y, incy)
end

# ── tbmv / tbsv: x := op(A)·x  /  op(A)⁻¹·x, A triangular banded ────────────────────────────────
# Band-segment helpers per column j (returns 0-based AB-row offset of the segment start + length of
# the strictly off-diagonal part + 0-based offset of the diagonal element within the column).
@inline function _tb_simd!(solve::Bool, up::Bool, tr::Bool, unit::Bool, n::Int, k::Int, AB, x)
    T = eltype(AB)
    GC.@preserve AB x begin
        Ap = pointer(AB); xp = pointer(x); ldb = stride(AB, 2); sz = sizeof(T)
        # column j: up ⇒ off-diag A[ilo:j-1,j] at AB[k+1+ilo-j,j], diag AB[k+1,j];
        #           !up ⇒ off-diag A[j+1:ihi,j] at AB[2,j],         diag AB[1,j]
        cofs(j) = up ? (max(1, j - k), Ap + ((k + max(1, j - k) - j) + (j - 1) * ldb) * sz, j - max(1, j - k), Ap + (k + (j - 1) * ldb) * sz, xp + (max(1, j - k) - 1) * sz) :
            (j, Ap + (1 + (j - 1) * ldb) * sz, min(n, j + k) - j, Ap + (j - 1) * ldb * sz, xp + j * sz)
        if !tr                                          # multiply/solve N
            order = (up != solve) ? (1:n) : (n:-1:1)    # trmv: U→asc,L→desc ; trsv: U→desc,L→asc
            @inbounds for j in order
                _, segp, len, dgp, xseg = cofs(j)
                if solve
                    unit || unsafe_store!(xp, unsafe_load(xp, j) / unsafe_load(dgp), j)
                    _axpy_simd!(len, -unsafe_load(xp, j), segp, xseg)
                else
                    t = unsafe_load(xp, j)
                    _axpy_simd!(len, t, segp, xseg)
                    unit || unsafe_store!(xp, t * unsafe_load(dgp), j)
                end
            end
        else                                            # transpose: dot form
            order = (up != solve) ? (n:-1:1) : (1:n)
            @inbounds for j in order
                _, segp, len, dgp, xseg = cofs(j)
                if solve
                    t = unsafe_load(xp, j) - _dot_simd(len, segp, xseg, T)
                    unit || (t /= unsafe_load(dgp))
                    unsafe_store!(xp, t, j)
                else
                    xj = unsafe_load(xp, j)
                    s = _dot_simd(len, segp, xseg, T)
                    unsafe_store!(xp, (unit ? xj : xj * unsafe_load(dgp)) + s, j)
                end
            end
        end
    end
    return x
end

function _tb_generic!(solve::Bool, up::Bool, tr::Bool, cj::Bool, unit::Bool, n::Int, k::Int, AB, x, incx::Integer)
    sx = _start(n, incx)
    abij = (i, j) -> (v = up ? AB[k + 1 + i - j, j] : AB[1 + i - j, j]; cj ? conj(v) : v)
    abjj = (j) -> (v = up ? AB[k + 1, j] : AB[1, j]; cj ? conj(v) : v)
    rng(j) = up ? (max(1, j - k):(j - 1)) : ((j + 1):min(n, j + k))
    if !tr
        order = (up != solve) ? (1:n) : (n:-1:1)
        @inbounds for j in order
            if solve
                unit || _st!(x, sx + (j - 1) * incx, _ld(x, sx + (j - 1) * incx) / abjj(j))
                xj = _ld(x, sx + (j - 1) * incx)
                for i in rng(j)
                    _st!(x, sx + (i - 1) * incx, _ld(x, sx + (i - 1) * incx) - xj * abij(i, j))
                end
            else
                xj = _ld(x, sx + (j - 1) * incx)
                for i in rng(j)
                    _st!(x, sx + (i - 1) * incx, _ld(x, sx + (i - 1) * incx) + xj * abij(i, j))
                end
                unit || _st!(x, sx + (j - 1) * incx, xj * abjj(j))
            end
        end
    else
        order = (up != solve) ? (n:-1:1) : (1:n)
        @inbounds for j in order
            s = _ld(x, sx + (j - 1) * incx)
            if solve
                for i in rng(j)
                    s -= abij(i, j) * _ld(x, sx + (i - 1) * incx)
                end
                unit || (s /= abjj(j))
                _st!(x, sx + (j - 1) * incx, s)
            else
                s = unit ? s : s * abjj(j)
                for i in rng(j)
                    s += abij(i, j) * _ld(x, sx + (i - 1) * incx)
                end
                _st!(x, sx + (j - 1) * incx, s)
            end
        end
    end
    return x
end

function _tbmv!(up::Bool, tr::Bool, cj::Bool, unit::Bool, n::Integer, k::Integer, AB, x, incx::Integer)
    if !cj && _l2v_simd_ok(AB, x, incx)
        return _tb_simd!(false, up, tr, unit, Int(n), Int(k), AB, x)
    end
    return _tb_generic!(false, up, tr, cj, unit, Int(n), Int(k), AB, x, incx)
end
function _tbsv!(up::Bool, tr::Bool, cj::Bool, unit::Bool, n::Integer, k::Integer, AB, x, incx::Integer)
    if !cj && _l2v_simd_ok(AB, x, incx)
        return _tb_simd!(true, up, tr, unit, Int(n), Int(k), AB, x)
    end
    return _tb_generic!(true, up, tr, cj, unit, Int(n), Int(k), AB, x, incx)
end
