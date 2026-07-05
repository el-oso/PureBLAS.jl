# BLAS Level-2: matrix-vector. gemv (y = α·op(A)·x + β·y) and ger (A += α·x·yᵀ). Both reduce to the
# Level-1 column kernels: gemv-N and ger are column axpys (`_axpy_simd!`), gemv-T/C are column dots
# (`_dot_simd`). Real dense unit-stride takes the SIMD path; complex / Dual / strided take a generic
# scalar loop (AD-traceable). A is column-major; for the SIMD path A[:,j] is a contiguous segment.

# y .*= β, with β==0 → 0 (overwrite, ignoring NaN/Inf per BLAS) and β==1 → no-op.
@inline function _scale_y!(n::Int, β::Number, y, incy::Integer)
    if iszero(β)
        iy = _start(n, incy)
        @inbounds for _ in 1:n
            _st!(y, iy, zero(_et(y))); iy += incy
        end
    elseif !isone(β)
        _scal!(n, β, y, incy)
    end
    return
end

# SIMD eligibility: A, x, y all the SAME real (Float32/Float64) eltype, A dense with unit column
# stride, x/y dense unit-stride. Checking x/y eltypes (not just A's) is essential — gemv(real A,
# Dual x) from AD must take the generic path, not the SIMD one (which would MethodError on Dual).
@inline function _l2_simd_ok(A, x, y, incx::Integer, incy::Integer)
    T = eltype(A)
    return incx == 1 && incy == 1 && T <: BlasReal && eltype(x) === T && eltype(y) === T &&
        _strided1(A) &&
        x isa StridedVector && stride(x, 1) == 1 && y isa StridedVector && stride(y, 1) == 1
end

const _GEMV_MR = _vwidth(Float64) == 4 ? 8 : 4   # gemv-N row-block height in vectors (mr = _GEMV_MR·W rows). AVX2: 8 accs feed both FMA units (~5-cyc latency) — MR=4 half-fills the pipe at cache-resident mid-n; AVX-512 (32 regs, already ≥gate) stays 4.

const _GEMV_NP = 8             # gemv-N column-panel width
const _GEMVN_RB = @load_preference("gemvn_rb", _vwidth(Float64) == 4 ? 64 : 448)::Int  # gemv-N: n ≤ this → row-block; larger → column-panel. AVX2 cut dropped 192→64: with _GEMV_MR=8 the sequential-streaming panel path now beats strided row-block for all n≥96 (128: 0.92→1.0); row-block only wins at n≤64 where panel's m<mr all-masked tail dominates. Zen4 1MB L2 → 448.
#                                unmasked full-block kernel, dominates per-column at every n ≥ 512,
#                                incl. the n=512 power-of-2 / just-over-L2 case → 0.96×).
# gemv-N (column-major A makes it transpose-like — see kb finding): two regimes —
#   small n: row-block (y in registers across all cols; A strided but cache-resident),
#   else:    column-panel (accumulate _GEMV_NP cols/pass → y re-streamed n/_GEMV_NP times, A in
#            _GEMV_NP sequential streams; an unmasked full-block kernel makes it dominate per-column
#            at every n ≥ 512, incl. the huge-n y-restream that the per-column path lost at n=4096).

# Row-block: full block (β folded) + masked remainder for any m.
@generated function _gemv_n_block!(yb::Ptr{T}, Ab::Ptr{T}, lda::Int, xp::Ptr{T}, n::Int,
        α::T, β::T, ::Val{MR}, ::Val{B0}) where {T, MR, B0}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T)
    body = quote end
    if B0
        for v in 1:MR; push!(body.args, :($(Symbol(:c, v)) = zero($V))); end
    else
        push!(body.args, :(bv = $V(β)))
        for v in 1:MR; push!(body.args, :($(Symbol(:c, v)) = bv * vload($V, yb + $((v - 1) * W * sz)))); end
    end
    push!(body.args, :(av = $V(α)))
    inner = quote end
    push!(inner.args, :(xj = av * $V(unsafe_load(xp, j + 1))))
    for v in 1:MR
        push!(inner.args, :($(Symbol(:c, v)) = muladd(vload($V, Ab + ($((v - 1) * W) + j * lda) * $sz), xj, $(Symbol(:c, v)))))
    end
    push!(body.args, :(for j in 0:(n - 1); $inner; end))
    for v in 1:MR; push!(body.args, :(vstore($(Symbol(:c, v)), yb + $((v - 1) * W * sz)))); end
    push!(body.args, :(return nothing))
    return body
end

@generated function _gemv_n_block_masked!(yb::Ptr{T}, Ab::Ptr{T}, lda::Int, xp::Ptr{T}, n::Int,
        α::T, β::T, mrows::Int, ::Val{NV}, ::Val{B0}) where {T, NV, B0}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T)
    body = quote end
    lanetuple = Expr(:tuple, (0:(W - 1))...)
    push!(body.args, :(lanes = Vec{$W, Int}($lanetuple)))
    for v in 1:NV; push!(body.args, :($(Symbol(:k, v)) = (lanes + $((v - 1) * W)) < mrows)); end
    if B0
        for v in 1:NV; push!(body.args, :($(Symbol(:c, v)) = zero($V))); end
    else
        push!(body.args, :(bv = $V(β)))
        for v in 1:NV; push!(body.args, :($(Symbol(:c, v)) = bv * vload($V, yb + $((v - 1) * W * sz), $(Symbol(:k, v))))); end
    end
    push!(body.args, :(av = $V(α)))
    inner = quote end
    push!(inner.args, :(xj = av * $V(unsafe_load(xp, j + 1))))
    for v in 1:NV
        push!(inner.args, :($(Symbol(:c, v)) = muladd(vload($V, Ab + ($((v - 1) * W) + j * lda) * $sz, $(Symbol(:k, v))), xj, $(Symbol(:c, v)))))
    end
    push!(body.args, :(for j in 0:(n - 1); $inner; end))
    for v in 1:NV; push!(body.args, :(vstore($(Symbol(:c, v)), yb + $((v - 1) * W * sz), $(Symbol(:k, v))))); end
    push!(body.args, :(return nothing))
    return body
end

@inline function _gemv_n_rowblock!(m::Int, n::Int, α::T, A, x, y, β::T, ::Val{B0}) where {T<:BlasReal, B0}
    W = _vwidth(T); mr = _GEMV_MR * W
    GC.@preserve A x y begin
        Aptr = pointer(A); yptr = pointer(y); xptr = pointer(x); lda = stride(A, 2); sz = sizeof(T)
        i0 = 0
        while i0 + mr <= m
            _gemv_n_block!(yptr + i0 * sz, Aptr + i0 * sz, lda, xptr, n, α, β, Val(_GEMV_MR), Val(B0))
            i0 += mr
        end
        mre = m - i0
        if mre > 0
            yb = yptr + i0 * sz; Ab = Aptr + i0 * sz; nv = cld(mre, W)
            if nv == 1
                _gemv_n_block_masked!(yb, Ab, lda, xptr, n, α, β, mre, Val(1), Val(B0))
            elseif nv == 2
                _gemv_n_block_masked!(yb, Ab, lda, xptr, n, α, β, mre, Val(2), Val(B0))
            elseif nv == 3
                _gemv_n_block_masked!(yb, Ab, lda, xptr, n, α, β, mre, Val(3), Val(B0))
            else
                _gemv_n_block_masked!(yb, Ab, lda, xptr, n, α, β, mre, Val(_GEMV_MR), Val(B0))
            end
        end
    end
    return y
end

# Column-panel × masked-row-block: accumulate Peff columns of one panel into a y-block, RMW y
# once per (panel, row-block) → y re-streamed n/_GEMV_NP times, A read as _GEMV_NP sequential
# streams. y pre-scaled by β by the driver.
@generated function _gemv_n_panel!(yb::Ptr{T}, Ab::Ptr{T}, lda::Int, xp::Ptr{T}, jc::Int, Peff::Int,
        mrows::Int, α::T, ::Val{NV}) where {T, NV}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T)
    body = quote end
    lanetuple = Expr(:tuple, (0:(W - 1))...)
    push!(body.args, :(lanes = Vec{$W, Int}($lanetuple)))
    for v in 1:NV; push!(body.args, :($(Symbol(:k, v)) = (lanes + $((v - 1) * W)) < mrows)); end
    push!(body.args, :(av = $V(α)))
    for v in 1:NV; push!(body.args, :($(Symbol(:c, v)) = vload($V, yb + $((v - 1) * W * sz), $(Symbol(:k, v))))); end
    inner = quote end
    push!(inner.args, :(xj = av * $V(unsafe_load(xp, jc + cc + 1))))
    for v in 1:NV
        push!(inner.args, :($(Symbol(:c, v)) = muladd(vload($V, Ab + ($((v - 1) * W) + (jc + cc) * lda) * $sz, $(Symbol(:k, v))), xj, $(Symbol(:c, v)))))
    end
    push!(body.args, :(for cc in 0:(Peff - 1); $inner; end))
    for v in 1:NV; push!(body.args, :(vstore($(Symbol(:c, v)), yb + $((v - 1) * W * sz), $(Symbol(:k, v))))); end
    push!(body.args, :(return nothing))
    return body
end

# Unmasked full-row-block variant (mre == mr): the common case — no mask overhead. Needed so the
# panel path is competitive at mid n (e.g. n=512), where the masked version's per-block overhead cost
# ~12% vs per-column.
@generated function _gemv_n_panel_full!(yb::Ptr{T}, Ab::Ptr{T}, lda::Int, xp::Ptr{T}, jc::Int,
        Peff::Int, α::T, ::Val{MR}) where {T, MR}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T)
    body = quote end
    push!(body.args, :(av = $V(α)))
    for v in 1:MR; push!(body.args, :($(Symbol(:c, v)) = vload($V, yb + $((v - 1) * W * sz)))); end
    inner = quote end
    push!(inner.args, :(xj = av * $V(unsafe_load(xp, jc + cc + 1))))
    for v in 1:MR
        push!(inner.args, :($(Symbol(:c, v)) = muladd(vload($V, Ab + ($((v - 1) * W) + (jc + cc) * lda) * $sz), xj, $(Symbol(:c, v)))))
    end
    push!(body.args, :(for cc in 0:(Peff - 1); $inner; end))
    for v in 1:MR; push!(body.args, :(vstore($(Symbol(:c, v)), yb + $((v - 1) * W * sz)))); end
    push!(body.args, :(return nothing))
    return body
end

@inline function _gemv_n_paneldrv!(m::Int, n::Int, α::T, A, x, y, β::T, ::Val{B0}) where {T<:BlasReal, B0}
    W = _vwidth(T); mr = _GEMV_MR * W
    GC.@preserve A x y begin
        Aptr = pointer(A); yptr = pointer(y); xptr = pointer(x); lda = stride(A, 2); sz = sizeof(T)
        if B0
            @inbounds for i in 1:m; unsafe_store!(yptr, zero(T), i); end
        elseif β != one(T)
            @inbounds for i in 1:m; unsafe_store!(yptr, β * unsafe_load(yptr, i), i); end
        end
        jc = 0
        while jc < n
            Peff = min(_GEMV_NP, n - jc)
            i0 = 0
            while i0 + mr <= m   # full row-blocks: unmasked (no per-block mask overhead)
                _gemv_n_panel_full!(yptr + i0 * sz, Aptr + i0 * sz, lda, xptr, jc, Peff, α, Val(_GEMV_MR))
                i0 += mr
            end
            mre = m - i0          # masked remainder
            if mre > 0
                yb = yptr + i0 * sz; Ab = Aptr + i0 * sz; nv = cld(mre, W)
                if nv == 1
                    _gemv_n_panel!(yb, Ab, lda, xptr, jc, Peff, mre, α, Val(1))
                elseif nv == 2
                    _gemv_n_panel!(yb, Ab, lda, xptr, jc, Peff, mre, α, Val(2))
                elseif nv == 3
                    _gemv_n_panel!(yb, Ab, lda, xptr, jc, Peff, mre, α, Val(3))
                else
                    _gemv_n_panel!(yb, Ab, lda, xptr, jc, Peff, mre, α, Val(_GEMV_MR))
                end
            end
            jc += _GEMV_NP
        end
    end
    return y
end

@inline function _gemv_n_simd!(m::Int, n::Int, α::T, A, x, y, β::T, ::Val{B0}) where {T<:BlasReal, B0}
    if n <= _GEMVN_RB
        _gemv_n_rowblock!(m, n, α, A, x, y, β, Val(B0))
    else
        _gemv_n_paneldrv!(m, n, α, A, x, y, β, Val(B0))
    end
    return y
end

# gemv-T column-block: NC column-dots accumulated together, reusing each x W-chunk across the NC
# columns (one set of horizontal sums per block) — cuts per-column overhead for small n. β folded
# in; masked tail for the row remainder.
@generated function _gemv_t_block!(yp::Ptr{T}, Ab::Ptr{T}, lda::Int, xp::Ptr{T}, m::Int,
        α::T, β::T, ::Val{NC}, ::Val{B0}) where {T, NC, B0}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T)
    body = quote end
    lanetuple = Expr(:tuple, (0:(W - 1))...)
    push!(body.args, :(lanes = Vec{$W, Int}($lanetuple)))
    for c in 1:NC; push!(body.args, :($(Symbol(:a, c)) = zero($V))); end
    full = quote end
    push!(full.args, :(xc = vload($V, xp + i * $sz)))
    for c in 1:NC
        push!(full.args, :($(Symbol(:a, c)) = muladd(vload($V, Ab + (i + $(c - 1) * lda) * $sz), xc, $(Symbol(:a, c)))))
    end
    push!(body.args, :(nfull = m - rem(m, $W); i = 0; while i < nfull; $full; i += $W; end))
    rmd = quote end
    push!(rmd.args, :(msk = lanes < (m - i)))
    push!(rmd.args, :(xc = vload($V, xp + i * $sz, msk)))
    for c in 1:NC
        push!(rmd.args, :($(Symbol(:a, c)) = muladd(vload($V, Ab + (i + $(c - 1) * lda) * $sz, msk), xc, $(Symbol(:a, c)))))
    end
    push!(body.args, :(if i < m; $rmd; end))
    for c in 1:NC
        st = B0 ? :(unsafe_store!(yp, α * sc, $c)) :
            :(unsafe_store!(yp, muladd(β, unsafe_load(yp, $c), α * sc), $c))
        push!(body.args, :(sc = sum($(Symbol(:a, c))); $st))
    end
    push!(body.args, :(return nothing))
    return body
end

# gemv-T: column-block (4 cols/pass sharing each x W-chunk) for all n. Sharing x cuts both the small-n
# per-column overhead AND the huge-n x-restream (x exceeds L1 at n≈4096; per-column re-read it n times).
@inline function _gemv_t_simd!(m::Int, n::Int, α::T, A, x, β::T, y, ::Val{B0}) where {T<:BlasReal, B0}
    GC.@preserve A x y begin
        Aptr = pointer(A); xptr = pointer(x); yptr = pointer(y); lda = stride(A, 2); sz = sizeof(T)
        j = 0
        while j + 4 <= n
            _gemv_t_block!(yptr + j * sz, Aptr + j * lda * sz, lda, xptr, m, α, β, Val(4), Val(B0))
            j += 4
        end
        @inbounds while j < n           # remainder columns: per-column dot
            s = _dot_simd(m, Aptr + j * lda * sz, xptr, T)
            yj = unsafe_load(yptr, j + 1)
            unsafe_store!(yptr, (B0 ? zero(T) : β * yj) + α * s, j + 1)
            j += 1
        end
    end
    return y
end

# y := β·y + α·op(A)·x. trans: false=N, true=T/C; cj: conjugate (op='C').
# Complex unit-stride L2 eligibility (mirror _l2_simd_ok for the complex SIMD paths).
@inline function _l2c_ok(A, x, y, incx::Integer, incy::Integer)
    T = eltype(A)
    return incx == 1 && incy == 1 && T <: BlasComplex && eltype(x) === T && eltype(y) === T &&
        _strided1(A) &&
        x isa StridedVector && stride(x, 1) == 1 && y isa StridedVector && stride(y, 1) == 1
end

# Complex gemv-N row-tile height (in W-complex vectors). Vec{2W} accumulators (AVX2 → 2 regs each), so a
# smaller MR than the real kernel. Preferences-tunable; swept per fleet box.
const _CGEMV_MR = @load_preference("cgemv_mr", _vwidth(Float64) == 4 ? 3 : 4)::Int
const _CGEMV_NP = 8                                 # column-panel width when A doesn't fit cache
# When A (m×n complex) fits ~L2, sweep all n columns in ONE panel (row-tile mode: A cache-resident, no
# panel/y-restream overhead — faster at small n). Above, width-_CGEMV_NP panels stream A sequentially.
const _CGEMV_RB = @load_preference("cgemv_rb", 65536)::Int   # m·n complex threshold for one-panel mode

# Complex gemv-N panel block: accumulate columns [jc, jc+Peff) of A into MR row-tiles of W complex, RMW
# into y (y pre-scaled by β by the driver). Interleaved Vec{2W} accumulators; per column cⱼ=α·x[j],
# c += A·cr + swap(A)·[−ci,ci] (swap-pairs complex multiply). Panel loop reads each A-column sequentially.
# NOTE (AVX2): Vec{2W} legalizes to 2 regs (MR small); still ~0.5–0.7 on AVX2 — gemvN there is
# shuffle/throughput-bound, an AVX2 TUNING residual (the fma/muladd primitives suffice; a split-Vec{W}
# variant measured worse). Gates AVX-512 (Vec{2W}=1 reg). See ROADMAP M5.
@generated function _gemv_n_block_cmplx!(yb::Ptr{T}, Ab::Ptr{T}, ldc::Int, xp::Ptr{T}, jc::Int, Peff::Int,
        αr::T, αi::T, ::Val{MR}) where {T, MR}
    W = _vwidth(T); V2 = Vec{2W, T}; sz = sizeof(T)
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(2W - 1))...)
    body = quote end
    for v in 1:MR; push!(body.args, :($(Symbol(:c, v)) = vload($V2, yb + $((v - 1) * 2W * sz)))); end
    inner = quote end
    push!(inner.args, :(jj = jc + cc))
    push!(inner.args, :(xr = unsafe_load(xp, 2jj + 1); xi = unsafe_load(xp, 2jj + 2)))
    push!(inner.args, :(cr = αr * xr - αi * xi; ci = αr * xi + αi * xr))                # cⱼ = α·x[jj]
    push!(inner.args, :(crv = $V2(cr)))
    push!(inner.args, :(csgn = $V2($(Expr(:tuple, (iseven(l) ? :(-ci) : :ci for l in 0:(2W - 1))...)))))
    for v in 1:MR
        av = Symbol(:av, v)
        push!(inner.args, :($av = vload($V2, Ab + ($((v - 1) * 2W) + jj * 2 * ldc) * $sz)))
        push!(inner.args, :($(Symbol(:c, v)) = muladd($av, crv, $(Symbol(:c, v)))))
        push!(inner.args, :($(Symbol(:c, v)) = muladd(shufflevector($av, Val($swp)), csgn, $(Symbol(:c, v)))))
    end
    push!(body.args, :(for cc in 0:(Peff - 1); $inner; end))
    for v in 1:MR; push!(body.args, :(vstore($(Symbol(:c, v)), yb + $((v - 1) * 2W * sz)))); end
    push!(body.args, :(return nothing))
    return body
end

# Driver: pre-scale y by β once, then column-panels × row-tiles accumulate (RMW y). W-remainder blocks
# (Val(1)) + scalar tail per panel handle m not a multiple of MR·W.
function _gemv_n_cmplx!(m::Int, n::Int, α::Complex{T}, A, x, y, β::Complex{T}, ::Val{B0}) where {T<:BlasReal, B0}
    W = _vwidth(T); mr = _CGEMV_MR * W; sz = sizeof(T); αr = real(α); αi = imag(α)
    GC.@preserve A x y begin
        Ap = Ptr{T}(pointer(A)); yp = Ptr{T}(pointer(y)); xp = Ptr{T}(pointer(x)); ldc = stride(A, 2)
        if B0
            @inbounds for i in 1:(2m); unsafe_store!(yp, zero(T), i); end       # y := 0
        elseif !isone(β)
            _scal_cmplx_simd!(m, real(β), imag(β), y)                           # y := β·y
        end
        np = m * n <= _CGEMV_RB ? n : _CGEMV_NP     # one wide panel if A fits cache, else stream
        jc = 0
        while jc < n
            Peff = min(np, n - jc)
            i0 = 0
            while i0 + mr <= m
                _gemv_n_block_cmplx!(yp + i0 * 2 * sz, Ap + i0 * 2 * sz, ldc, xp, jc, Peff, αr, αi, Val(_CGEMV_MR)); i0 += mr
            end
            while i0 + W <= m
                _gemv_n_block_cmplx!(yp + i0 * 2 * sz, Ap + i0 * 2 * sz, ldc, xp, jc, Peff, αr, αi, Val(1)); i0 += W
            end
            @inbounds for i in (i0 + 1):m                                       # scalar tail (< W rows)
                s = zero(Complex{T})
                for cc in 0:(Peff - 1); s += A[i, jc + cc + 1] * x[jc + cc + 1]; end
                y[i] += α * s
            end
            jc += Peff
        end
    end
    return y
end

# Complex gemv trans='T'/'C': y[j] := β·y[j] + α·Σ_i (CJ ? conj(A[i,j]) : A[i,j])·x[i]. Each output is
# one complex dot of A's (contiguous) column j with x — reuses the L1 _dot_cmplx_simd kernel directly.
function _gemv_tc_cmplx!(m::Int, n::Int, α::Complex{T}, A, x, β::Complex{T}, y, ::Val{CJ}) where {T<:BlasReal, CJ}
    z = iszero(β); csz = sizeof(Complex{T})
    GC.@preserve A x begin
        Ap = pointer(A); lda = stride(A, 2); xp = pointer(x)
        @inbounds for j in 1:n
            colp = Ap + (j - 1) * lda * csz                       # Ptr{Complex{T}} → column j (unit-stride)
            s = _dot_cmplx_simd(m, colp, xp, T, Val(CJ))
            yj = y[j]; y[j] = (z ? zero(yj) : β * yj) + α * s
        end
    end
    return y
end

function _gemv!(trans::Bool, cj::Bool, m::Integer, n::Integer, α::Number, A, x, incx::Integer,
        β::Number, y, incy::Integer)
    if !trans
        if iszero(α)
            _scale_y!(Int(m), β, y, incy); return y
        end
        if _l2_simd_ok(A, x, y, incx, incy)   # column-panel kernel handles all n; β folded in
            αT = convert(eltype(A), α); βT = convert(eltype(A), β)
            return iszero(β) ? _gemv_n_simd!(Int(m), Int(n), αT, A, x, y, βT, Val(true)) :
                _gemv_n_simd!(Int(m), Int(n), αT, A, x, y, βT, Val(false))
        end
        if _l2c_ok(A, x, y, incx, incy)       # complex N → row-tiled SIMD (y in registers over columns)
            αc = convert(eltype(A), α); βc = convert(eltype(A), β)
            return iszero(β) ? _gemv_n_cmplx!(Int(m), Int(n), αc, A, x, y, βc, Val(true)) :
                _gemv_n_cmplx!(Int(m), Int(n), αc, A, x, y, βc, Val(false))
        end
        _scale_y!(Int(m), β, y, incy)
        ix = _start(n, incx)
        @inbounds for j in 1:n
            axj = α * _ld(x, ix); ix += incx
            if !iszero(axj)
                iy = _start(m, incy)
                for i in 1:m
                    _st!(y, iy, _ld(y, iy) + axj * A[i, j]); iy += incy
                end
            end
        end
    else
        if _l2_simd_ok(A, x, y, incx, incy)
            αT = convert(eltype(A), α); βT = convert(eltype(A), β)
            return iszero(β) ? _gemv_t_simd!(Int(m), Int(n), αT, A, x, βT, y, Val(true)) :
                _gemv_t_simd!(Int(m), Int(n), αT, A, x, βT, y, Val(false))
        end
        if _l2c_ok(A, x, y, incx, incy)                          # complex T/C → per-column SIMD dot
            αc = convert(eltype(A), α); βc = convert(eltype(A), β)
            return cj ? _gemv_tc_cmplx!(Int(m), Int(n), αc, A, x, βc, y, Val(true)) :
                _gemv_tc_cmplx!(Int(m), Int(n), αc, A, x, βc, y, Val(false))
        end
        s0 = zero(_et(A)) * zero(_et(x))
        iy = _start(n, incy)
        @inbounds for i in 1:n
            s = s0
            ix = _start(m, incx)
            for jj in 1:m
                aij = cj ? conj(A[jj, i]) : A[jj, i]
                s += aij * _ld(x, ix); ix += incx
            end
            yi = _ld(y, iy)
            _st!(y, iy, (iszero(β) ? zero(yi) : β * yi) + α * s); iy += incy
        end
    end
    return y
end

@inline function _ger_simd!(m::Int, n::Int, α::T, x, y, A) where {T<:BlasReal}
    GC.@preserve A x y begin
        Aptr = pointer(A); xptr = pointer(x); yptr = pointer(y); lda = stride(A, 2); sz = sizeof(T)
        @inbounds for j in 1:n
            ayj = α * unsafe_load(yptr, j)
            iszero(ayj) || _axpy_simd!(m, ayj, xptr, Aptr + (j - 1) * lda * sz)  # A[:,j] += ayj·x
        end
    end
    return A
end

# Complex rank-1: A[:,j] += (α·(cj ? conj(y[j]) : y[j]))·x — one complex axpy of x into each contiguous
# column, reusing the L1 _axpy_cmplx_simd! kernel (like the real _ger_simd! reuses _axpy_simd!).
function _ger_cmplx!(m::Int, n::Int, α::Complex{T}, x, y, A, cj::Bool) where {T<:BlasReal}
    csz = sizeof(Complex{T})
    GC.@preserve A x y begin
        Aptr = pointer(A); xptr = pointer(x); yptr = pointer(y); lda = stride(A, 2)
        @inbounds for j in 1:n
            yj = cj ? conj(unsafe_load(yptr, j)) : unsafe_load(yptr, j)
            ayj = α * yj
            iszero(ayj) || _axpy_cmplx_simd!(m, real(ayj), imag(ayj), xptr, Aptr + (j - 1) * lda * csz)
        end
    end
    return A
end

# A := α·x·yᵀ + A  (geru); cj=true gives α·x·yᴴ (gerc).
function _ger!(cj::Bool, m::Integer, n::Integer, α::Number, x, incx::Integer, y, incy::Integer, A)
    iszero(α) && return A
    if _l2_simd_ok(A, x, y, incx, incy)
        return _ger_simd!(Int(m), Int(n), convert(eltype(A), α), x, y, A)
    end
    if _l2c_ok(A, x, y, incx, incy)
        return _ger_cmplx!(Int(m), Int(n), convert(eltype(A), α), x, y, A, cj)
    end
    iy = _start(n, incy)
    @inbounds for j in 1:n
        yj = cj ? conj(_ld(y, iy)) : _ld(y, iy); iy += incy
        ayj = α * yj
        if !iszero(ayj)
            ix = _start(m, incx)
            for i in 1:m
                A[i, j] += ayj * _ld(x, ix); ix += incx
            end
        end
    end
    return A
end

# ── symv / hemv (symmetric / Hermitian matrix-vector) ──────────────────────────────────────────
# symv: y := α·A·x + β·y, A symmetric, only the `up` triangle stored. Each stored A[i,j] feeds BOTH
# y[i] (via x[j]) and y[j] (via x[i], since A[j,i]=A[i,j]) — so one pass over the triangle does an
# axpy AND a dot per column, reading the column ONCE (symv is memory-bound: 2× A traffic would fail
# the gate). Fused single-read column kernel below; scalar remainder (< W per column, cheap).

# Over a column segment of length L: y[k] += axj·a[k]  AND  return Σ a[k]·x[k].  (a = A-segment.)
@inline function _symv_col!(L::Int, axj::T, aptr::Ptr{T}, xptr::Ptr{T}, yptr::Ptr{T}) where {T<:BlasReal}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T)
    acc = zero(V); axv = V(axj); i = 0; nf = L - rem(L, W)
    while i < nf
        av = vload(V, aptr + i * sz)
        vstore(muladd(axv, av, vload(V, yptr + i * sz)), yptr + i * sz)   # y += axj·a
        acc = muladd(av, vload(V, xptr + i * sz), acc)                    # acc += a·x
        i += W
    end
    s = sum(acc)
    @inbounds while i < L            # scalar remainder (< W)
        a = unsafe_load(aptr, i + 1)
        unsafe_store!(yptr, muladd(axj, a, unsafe_load(yptr, i + 1)), i + 1)
        s += a * unsafe_load(xptr, i + 1)
        i += 1
    end
    return s
end

const _SYMV_NB = 8   # symv column-panel width (= # of gemv-T dot accumulators in the microkernel)
# symv row-panel height in vectors — its OWN const, NOT _GEMV_MR: symv's off-block fuses NB axpy +
# NB dot accumulators per column, so it is far more register-hungry than plain gemv-N. 4 fits AVX2's
# 16 ymm; the gemv-N MR=8 bump spilled symv (galen 1.13→0.86). AVX-512 kept 4 before, keeps 4 here.
const _SYMV_MR = 4

# Codegen helper (runs at @generated expansion): emit a K-vector off-diagonal row-block at row `i`,
# accumulating gemv-N into y (yp) and gemv-T into the d_c (one A load feeds both). masked ⇒ guard
# each vector by k_v < rmn. Shared by the lower/upper panel kernels.
function _symv_offblk_expr(W, V, sz, NB, K, masked)
    q = Expr(:block)
    masked && for v in 1:K
        push!(q.args, :($(Symbol(:k, v)) = (lanes + $((v - 1) * W)) < rmn))
    end
    ld = (p, v) -> masked ? :(vload($V, $p + (i + $((v - 1) * W)) * $sz, $(Symbol(:k, v)))) :
        :(vload($V, $p + (i + $((v - 1) * W)) * $sz))
    for v in 1:K; push!(q.args, :($(Symbol(:yy, v)) = $(ld(:yp, v)))); end
    for v in 1:K; push!(q.args, :($(Symbol(:xx, v)) = $(ld(:xp, v)))); end
    for c in 1:NB
        for v in 1:K
            ap = masked ? :(vload($V, Ap + (i + $((v - 1) * W) + $(c - 1) * lda) * $sz, $(Symbol(:k, v)))) :
                :(vload($V, Ap + (i + $((v - 1) * W) + $(c - 1) * lda) * $sz))
            push!(q.args, :($(Symbol(:aa, v)) = $ap))
        end
        for v in 1:K
            push!(q.args, :($(Symbol(:yy, v)) = muladd($(Symbol(:aa, v)), $(Symbol(:xj, c)), $(Symbol(:yy, v)))))
            push!(q.args, :($(Symbol(:d, c)) = muladd($(Symbol(:aa, v)), $(Symbol(:xx, v)), $(Symbol(:d, c)))))
        end
    end
    for v in 1:K
        push!(q.args, masked ? :(vstore($(Symbol(:yy, v)), yp + (i + $((v - 1) * W)) * $sz, $(Symbol(:k, v)))) :
            :(vstore($(Symbol(:yy, v)), yp + (i + $((v - 1) * W)) * $sz)))
    end
    return q
end

# Unified LOWER panel: cols 0:NB-1, rows 0:M-1 (stored r≥c). The triangular diagonal block (rows
# 0:NB-1, one masked vector) feeds the SAME d_c vector accumulators as the off-diagonal rectangle
# (rows NB:M-1) → ONE reduction per column and a vectorized diagonal (the small-n win). The diagonal
# entry is taken once via the axpy (mask r≥c); the dot (mask r>c) is strictly-lower. α in xj_c.
@generated function _symv_lowerpanel!(M::Int, α::T, Ap::Ptr{T}, lda::Int, xp::Ptr{T}, yp::Ptr{T},
        ::Val{MR}, ::Val{NB}) where {T, MR, NB}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T); mr = MR * W
    body = quote end
    push!(body.args, :(lanes = Vec{$W, Int}($(Expr(:tuple, (0:(W - 1))...)))))
    push!(body.args, :(zv = zero($V)))
    for c in 1:NB
        push!(body.args, :($(Symbol(:xj, c)) = $V(α * unsafe_load(xp, $c))))
        push!(body.args, :($(Symbol(:d, c)) = zero($V)))
    end
    push!(body.args, :(mblk = lanes < $NB))
    push!(body.args, :(yblk = vload($V, yp, mblk)))
    push!(body.args, :(xblk = vload($V, xp, mblk)))
    for c in 1:NB
        push!(body.args, :(acd = vload($V, Ap + $(c - 1) * lda * $sz, mblk)))
        push!(body.args, :(yblk = vifelse(lanes >= $(c - 1), muladd($(Symbol(:xj, c)), acd, yblk), yblk)))
        push!(body.args, :($(Symbol(:d, c)) = muladd(vifelse(lanes > $(c - 1), acd, zv), xblk, $(Symbol(:d, c)))))
    end
    push!(body.args, :(vstore(yblk, yp, mblk)))
    push!(body.args, :(i = $NB; nfull = M - rem(M - $NB, $mr); while i < nfull; $(_symv_offblk_expr(W, V, sz, NB, MR, false)); i += $mr; end))
    branches = _symv_offblk_expr(W, V, sz, NB, 1, true)
    for k in 2:MR; branches = Expr(:if, :(rmn > $((k - 1) * W)), _symv_offblk_expr(W, V, sz, NB, k, true), branches); end
    push!(body.args, :(if i < M; rmn = M - i; $branches; end))
    for c in 1:NB; push!(body.args, :(unsafe_store!(yp, muladd(α, sum($(Symbol(:d, c))), unsafe_load(yp, $c)), $c))); end
    push!(body.args, :(return nothing))
    return body
end

# Unified UPPER panel: cols 0:NB-1 (global jb+c), rows 0:M-1 with M=jb+NB (stored r≤jb+c). Off-diagonal
# rectangle rows 0:dboff-1 (dboff=M-NB=jb) THEN triangular diagonal block rows dboff:dboff+NB-1.
@generated function _symv_upperpanel!(M::Int, α::T, Ap::Ptr{T}, lda::Int, xp::Ptr{T}, yp::Ptr{T},
        ::Val{MR}, ::Val{NB}) where {T, MR, NB}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T); mr = MR * W
    body = quote end
    push!(body.args, :(lanes = Vec{$W, Int}($(Expr(:tuple, (0:(W - 1))...)))))
    push!(body.args, :(zv = zero($V)))
    push!(body.args, :(dboff = M - $NB))
    for c in 1:NB
        push!(body.args, :($(Symbol(:xj, c)) = $V(α * unsafe_load(xp, dboff + $c))))
        push!(body.args, :($(Symbol(:d, c)) = zero($V)))
    end
    push!(body.args, :(i = 0; nfull = dboff - rem(dboff, $mr); while i < nfull; $(_symv_offblk_expr(W, V, sz, NB, MR, false)); i += $mr; end))
    branches = _symv_offblk_expr(W, V, sz, NB, 1, true)
    for k in 2:MR; branches = Expr(:if, :(rmn > $((k - 1) * W)), _symv_offblk_expr(W, V, sz, NB, k, true), branches); end
    push!(body.args, :(if i < dboff; rmn = dboff - i; $branches; end))
    push!(body.args, :(mblk = lanes < $NB))
    push!(body.args, :(yblk = vload($V, yp + dboff * $sz, mblk)))
    push!(body.args, :(xblk = vload($V, xp + dboff * $sz, mblk)))
    for c in 1:NB
        push!(body.args, :(acd = vload($V, Ap + (dboff + $(c - 1) * lda) * $sz, mblk)))
        push!(body.args, :(yblk = vifelse(lanes <= $(c - 1), muladd($(Symbol(:xj, c)), acd, yblk), yblk)))
        push!(body.args, :($(Symbol(:d, c)) = muladd(vifelse(lanes < $(c - 1), acd, zv), xblk, $(Symbol(:d, c)))))
    end
    push!(body.args, :(vstore(yblk, yp + dboff * $sz, mblk)))
    for c in 1:NB; push!(body.args, :(unsafe_store!(yp, muladd(α, sum($(Symbol(:d, c))), unsafe_load(yp, dboff + $c)), dboff + $c))); end
    push!(body.args, :(return nothing))
    return body
end

@inline function _symv_simd!(up::Bool, n::Int, α::T, A, x, y) where {T<:BlasReal}
    # NB must not exceed the vector width: the panel kernels handle the NB×NB diagonal block as ONE
    # masked vector (`lanes < NB`). NB=8 on W=4 (AVX2 F64) silently truncated the block → WRONG RESULTS
    # (latent bug caught by CI's AVX2 runner lottery; W and _SYMV_NB are consts, so this folds statically).
    NB = min(_SYMV_NB, _vwidth(T))
    GC.@preserve A x y begin
        base = pointer(A); xp = pointer(x); yp = pointer(y); lda = stride(A, 2); sz = sizeof(T)
        jb = 0
        while jb + NB <= n                                  # full column panels (unified kernel)
            if up
                _symv_upperpanel!(jb + NB, α, base + jb * lda * sz, lda, xp, yp, Val(_SYMV_MR), Val(NB))
            else
                _symv_lowerpanel!(n - jb, α, base + (jb + jb * lda) * sz, lda, xp + jb * sz, yp + jb * sz, Val(_SYMV_MR), Val(NB))
            end
            jb += NB
        end
        @inbounds while jb < n                              # last partial panel: naive full column
            axj = α * unsafe_load(xp, jb + 1)
            colp = base + jb * lda * sz                     # A[0,jb]
            ajj = unsafe_load(colp + jb * sz)               # A[jb,jb]
            if up
                s = _symv_col!(jb, axj, colp, xp, yp)
            else
                s = _symv_col!(n - 1 - jb, axj, colp + (jb + 1) * sz, xp + (jb + 1) * sz, yp + (jb + 1) * sz)
            end
            unsafe_store!(yp, unsafe_load(yp, jb + 1) + axj * ajj + α * s, jb + 1)
            jb += 1
        end
    end
    return y
end

# y := α·A·x + β·y, A symmetric (`up` ⇒ upper triangle stored). Real dense → fused SIMD; else generic.
function _symv!(up::Bool, n::Integer, α::Number, A, x, incx::Integer, β::Number, y, incy::Integer)
    _scale_y!(Int(n), β, y, incy)
    iszero(α) && return y
    if _l2_simd_ok(A, x, y, incx, incy)
        return _symv_simd!(up, Int(n), convert(eltype(A), α), A, x, y)
    end
    sx = _start(Int(n), incx); sy = _start(Int(n), incy)
    s0 = zero(_et(A)) * zero(_et(x))
    @inbounds for j in 1:n
        tmp = α * _ld(x, sx + (j - 1) * incx)
        s = s0
        rng = up ? (1:(j - 1)) : ((j + 1):n)
        for i in rng
            aij = A[i, j]
            _st!(y, sy + (i - 1) * incy, _ld(y, sy + (i - 1) * incy) + tmp * aij)
            s += aij * _ld(x, sx + (i - 1) * incx)
        end
        _st!(y, sy + (j - 1) * incy, _ld(y, sy + (j - 1) * incy) + tmp * A[j, j] + α * s)
    end
    return y
end

# y := α·A·x + β·y, A Hermitian (`up` ⇒ upper stored; A[j,i]=conj(A[i,j]); diagonal taken real).
# Complex → generic path (complex SIMD deferred, per project convention); correct for real too.
function _hemv!(up::Bool, n::Integer, α::Number, A, x, incx::Integer, β::Number, y, incy::Integer)
    _scale_y!(Int(n), β, y, incy)
    iszero(α) && return y
    sx = _start(Int(n), incx); sy = _start(Int(n), incy)
    s0 = zero(_et(A)) * zero(_et(x))
    @inbounds for j in 1:n
        tmp = α * _ld(x, sx + (j - 1) * incx)
        s = s0
        rng = up ? (1:(j - 1)) : ((j + 1):n)
        for i in rng
            aij = A[i, j]
            _st!(y, sy + (i - 1) * incy, _ld(y, sy + (i - 1) * incy) + tmp * aij)
            s += conj(aij) * _ld(x, sx + (i - 1) * incx)
        end
        _st!(y, sy + (j - 1) * incy, _ld(y, sy + (j - 1) * incy) + tmp * real(A[j, j]) + α * s)
    end
    return y
end

# ── trmv / trsv (triangular matrix-vector multiply / solve, in-place on x) ──────────────────────
# A is triangular (`up` ⇒ upper stored). `tr` ⇒ operate with op(A)ᵀ; `cj` ⇒ conjugate (op='C').
# `unit` ⇒ unit diagonal (A[j,j] implied 1, not read). Both reduce to per-column kernels: the "N"
# forms are column axpys (reuse `_axpy_simd!`), the "T/C" forms column dots (`_dot_simd`); a scalar
# diagonal multiply (trmv) / divide (trsv). Real dense unit-stride → SIMD; else generic (AD/complex).

# Real-dense unit-stride eligibility for a single in-place vector op.
@inline function _l2v_simd_ok(A, x, incx::Integer)
    T = eltype(A)
    return incx == 1 && T <: BlasReal && eltype(x) === T &&
        _strided1(A) && x isa StridedVector && stride(x, 1) == 1
end

@inline function _trmv_simd!(up::Bool, tr::Bool, unit::Bool, n::Int, A, x)
    T = eltype(A)
    GC.@preserve A x begin
        Ap = pointer(A); xp = pointer(x); lda = stride(A, 2); sz = sizeof(T)
        if !tr                                   # x := A·x  (column axpy)
            if up                                # U,N: j ascending
                @inbounds for j in 1:n
                    cp = Ap + (j - 1) * lda * sz; t = unsafe_load(xp, j)
                    _axpy_simd!(j - 1, t, cp, xp)
                    unit || unsafe_store!(xp, t * unsafe_load(cp + (j - 1) * sz), j)
                end
            else                                 # L,N: j descending
                @inbounds for j in n:-1:1
                    cp = Ap + (j - 1) * lda * sz; t = unsafe_load(xp, j)
                    _axpy_simd!(n - j, t, cp + j * sz, xp + j * sz)
                    unit || unsafe_store!(xp, t * unsafe_load(cp + (j - 1) * sz), j)
                end
            end
        else                                     # x := Aᵀ·x  (column dot)
            if up                                # U,T: j descending
                @inbounds for j in n:-1:1
                    cp = Ap + (j - 1) * lda * sz; xj = unsafe_load(xp, j)
                    s = _dot_simd(j - 1, cp, xp, T)
                    unsafe_store!(xp, (unit ? xj : xj * unsafe_load(cp + (j - 1) * sz)) + s, j)
                end
            else                                 # L,T: j ascending
                @inbounds for j in 1:n
                    cp = Ap + (j - 1) * lda * sz; xj = unsafe_load(xp, j)
                    s = _dot_simd(n - j, cp + j * sz, xp + j * sz, T)
                    unsafe_store!(xp, (unit ? xj : xj * unsafe_load(cp + (j - 1) * sz)) + s, j)
                end
            end
        end
    end
    return x
end

@inline function _trsv_simd!(up::Bool, tr::Bool, unit::Bool, n::Int, A, x)
    T = eltype(A)
    GC.@preserve A x begin
        Ap = pointer(A); xp = pointer(x); lda = stride(A, 2); sz = sizeof(T)
        if !tr                                   # solve A·x = b  (column axpy, subtract)
            if up                                # U,N: back-substitution, j descending
                @inbounds for j in n:-1:1
                    cp = Ap + (j - 1) * lda * sz
                    unit || unsafe_store!(xp, unsafe_load(xp, j) / unsafe_load(cp + (j - 1) * sz), j)
                    _axpy_simd!(j - 1, -unsafe_load(xp, j), cp, xp)
                end
            else                                 # L,N: forward, j ascending
                @inbounds for j in 1:n
                    cp = Ap + (j - 1) * lda * sz
                    unit || unsafe_store!(xp, unsafe_load(xp, j) / unsafe_load(cp + (j - 1) * sz), j)
                    _axpy_simd!(n - j, -unsafe_load(xp, j), cp + j * sz, xp + j * sz)
                end
            end
        else                                     # solve Aᵀ·x = b  (column dot)
            if up                                # U,T: forward, j ascending
                @inbounds for j in 1:n
                    cp = Ap + (j - 1) * lda * sz
                    t = unsafe_load(xp, j) - _dot_simd(j - 1, cp, xp, T)
                    unit || (t /= unsafe_load(cp + (j - 1) * sz))
                    unsafe_store!(xp, t, j)
                end
            else                                 # L,T: back, j descending
                @inbounds for j in n:-1:1
                    cp = Ap + (j - 1) * lda * sz
                    t = unsafe_load(xp, j) - _dot_simd(n - j, cp + j * sz, xp + j * sz, T)
                    unit || (t /= unsafe_load(cp + (j - 1) * sz))
                    unsafe_store!(xp, t, j)
                end
            end
        end
    end
    return x
end

const _TRI_NB = 64       # triangular block size (diagonal block per-column; off-diagonal via gemv)
const _TRI_T_UNB = 1024  # trsv-T: unblocked up to here (blocked only helps the huge-n x-restream).
#                          trmv-T blocks at _TRI_NB (its unblocked L-form dips mid-n); N forms at _TRI_NB.

# Blocked trmv/trsv (real dense): the per-column kernel re-streams x from memory at large n. Block it
# — each diagonal NB×NB block uses the per-column kernel (cache-resident), and the off-diagonal block
# uses the already-fast gemv (reads A once, no re-stream). Processing order keeps x_J unmodified when
# block I needs it. trmv: diagonal-then-gemv(+); trsv: gemv(−)-then-solve. Sub-blocks are contiguous
# views (unit-stride → SIMD gemv via the relaxed _l2_simd_ok).
# Force the column-panel gemv-N (β=1 accumulate) for the tall off-diagonal scatter `y += α·Av·xv`.
# Going through the dispatcher would pick the row-block path (n=NB cols → NB strided streams), which
# thrashes because a sub-block's columns are a full parent-lda apart; the panel uses _GEMV_NP streams.
@inline _tri_scat!(yv, Av, xv, α) = _gemv_n_paneldrv!(size(Av, 1), size(Av, 2), α, Av, xv, yv, one(α), Val(false))
# T-form off-diagonal: gemv-T kernel directly (no backend kwarg layer — ~200 ns/call dominated the
# few off-diagonal calls at mid n). y_I += α·Avᵀ·xv  (β=1 accumulate).
@inline _tri_scatT!(yv, Av, xv, α) = _gemv_t_simd!(size(Av, 1), size(Av, 2), α, Av, xv, one(α), yv, Val(false))

@inline function _trmv_blk!(up::Bool, tr::Bool, unit::Bool, n::Int, A, x)
    NB = _TRI_NB
    n <= NB && return _trmv_simd!(up, tr, unit, n, A, x)
    # N forms use column-block J so the off-diagonal scatter is a TALL gemv-N (good A locality).
    @inbounds if !tr && up               # U,N: J ascending; tall scatter UP then diag
        ib = 0
        while ib < n
            nb = min(NB, n - ib); J = (ib + 1):(ib + nb)
            ib > 0 && _tri_scat!(view(x, 1:ib), view(A, 1:ib, J), view(x, J), one(eltype(A)))
            _trmv_simd!(true, false, unit, nb, view(A, J, J), view(x, J))
            ib += NB
        end
    elseif !tr && !up                    # L,N: J descending; tall scatter DOWN then diag
        ib = (cld(n, NB) - 1) * NB
        while ib >= 0
            nb = min(NB, n - ib); J = (ib + 1):(ib + nb); je = ib + nb
            je < n && _tri_scat!(view(x, (je + 1):n), view(A, (je + 1):n, J), view(x, J), one(eltype(A)))
            _trmv_simd!(false, false, unit, nb, view(A, J, J), view(x, J))
            ib -= NB
        end
    elseif tr && up                      # U,T: I descending; diag(ᵀ) then gemv-T (rows above)
        ib = (cld(n, NB) - 1) * NB
        while ib >= 0
            nb = min(NB, n - ib); I = (ib + 1):(ib + nb)
            _trmv_simd!(true, true, unit, nb, view(A, I, I), view(x, I))
            ib > 0 && _tri_scatT!(view(x, I), view(A, 1:ib, I), view(x, 1:ib), one(eltype(A)))
            ib -= NB
        end
    else                                 # L,T: I ascending; diag(ᵀ) then gemv-T (rows below)
        ib = 0
        while ib < n
            nb = min(NB, n - ib); I = (ib + 1):(ib + nb)
            _trmv_simd!(false, true, unit, nb, view(A, I, I), view(x, I))
            ib + nb < n && _tri_scatT!(view(x, I), view(A, (ib + nb + 1):n, I), view(x, (ib + nb + 1):n), one(eltype(A)))
            ib += NB
        end
    end
    return x
end

@inline function _trsv_blk!(up::Bool, tr::Bool, unit::Bool, n::Int, A, x)
    NB = _TRI_NB
    # trsv-T (forward/back substitution by dots): unblocked is faster at mid n (x cached), blocking
    # only pays off for the huge-n x-restream. (trmv-T differs — it blocks at NB; see _TRI_T_UNB.)
    (n <= NB || (tr && n <= _TRI_T_UNB)) && return _trsv_simd!(up, tr, unit, n, A, x)
    # N forms use column-block J so the off-diagonal scatter is a TALL gemv-N (good A locality).
    @inbounds if !tr && up               # U,N back: J descending; solve diag then tall scatter UP
        ib = (cld(n, NB) - 1) * NB
        while ib >= 0
            nb = min(NB, n - ib); J = (ib + 1):(ib + nb)
            _trsv_simd!(true, false, unit, nb, view(A, J, J), view(x, J))
            ib > 0 && _tri_scat!(view(x, 1:ib), view(A, 1:ib, J), view(x, J), -one(eltype(A)))
            ib -= NB
        end
    elseif !tr && !up                    # L,N fwd: J ascending; solve diag then tall scatter DOWN
        ib = 0
        while ib < n
            nb = min(NB, n - ib); J = (ib + 1):(ib + nb); je = ib + nb
            _trsv_simd!(false, false, unit, nb, view(A, J, J), view(x, J))
            je < n && _tri_scat!(view(x, (je + 1):n), view(A, (je + 1):n, J), view(x, J), -one(eltype(A)))
            ib += NB
        end
    elseif tr && up                      # U,T forward: I ascending; gemv-T(−, rows above) then solve(ᵀ)
        ib = 0
        while ib < n
            nb = min(NB, n - ib); I = (ib + 1):(ib + nb)
            ib > 0 && _tri_scatT!(view(x, I), view(A, 1:ib, I), view(x, 1:ib), -one(eltype(A)))
            _trsv_simd!(true, true, unit, nb, view(A, I, I), view(x, I))
            ib += NB
        end
    else                                 # L,T back: I descending; gemv-T(−, rows below) then solve(ᵀ)
        ib = (cld(n, NB) - 1) * NB
        while ib >= 0
            nb = min(NB, n - ib); I = (ib + 1):(ib + nb)
            ib + nb < n && _tri_scatT!(view(x, I), view(A, (ib + nb + 1):n, I), view(x, (ib + nb + 1):n), -one(eltype(A)))
            _trsv_simd!(false, true, unit, nb, view(A, I, I), view(x, I))
            ib -= NB
        end
    end
    return x
end

# x := op(A)·x, A triangular. trans: false=N, true=T/C; cj: conjugate; unit: unit diagonal.
function _trmv!(up::Bool, tr::Bool, cj::Bool, unit::Bool, n::Integer, A, x, incx::Integer)
    if _l2v_simd_ok(A, x, incx)
        return _trmv_blk!(up, tr, unit, Int(n), A, x)
    end
    n = Int(n); sx = _start(n, incx)
    el = (i, j) -> cj ? conj(A[i, j]) : A[i, j]
    if !tr                                       # x := A·x
        if up
            @inbounds for j in 1:n
                xj = _ld(x, sx + (j - 1) * incx)
                for i in 1:(j - 1); _st!(x, sx + (i - 1) * incx, _ld(x, sx + (i - 1) * incx) + xj * A[i, j]); end
                unit || _st!(x, sx + (j - 1) * incx, xj * A[j, j])
            end
        else
            @inbounds for j in n:-1:1
                xj = _ld(x, sx + (j - 1) * incx)
                for i in n:-1:(j + 1); _st!(x, sx + (i - 1) * incx, _ld(x, sx + (i - 1) * incx) + xj * A[i, j]); end
                unit || _st!(x, sx + (j - 1) * incx, xj * A[j, j])
            end
        end
    else                                         # x := op(A)ᵀ·x
        if up
            @inbounds for j in n:-1:1
                s = unit ? _ld(x, sx + (j - 1) * incx) : _ld(x, sx + (j - 1) * incx) * (cj ? conj(A[j, j]) : A[j, j])
                for i in 1:(j - 1); s += el(i, j) * _ld(x, sx + (i - 1) * incx); end
                _st!(x, sx + (j - 1) * incx, s)
            end
        else
            @inbounds for j in 1:n
                s = unit ? _ld(x, sx + (j - 1) * incx) : _ld(x, sx + (j - 1) * incx) * (cj ? conj(A[j, j]) : A[j, j])
                for i in (j + 1):n; s += el(i, j) * _ld(x, sx + (i - 1) * incx); end
                _st!(x, sx + (j - 1) * incx, s)
            end
        end
    end
    return x
end

# x := op(A)⁻¹·x, A triangular (solve). Same parameterization as `_trmv!`.
function _trsv!(up::Bool, tr::Bool, cj::Bool, unit::Bool, n::Integer, A, x, incx::Integer)
    if _l2v_simd_ok(A, x, incx)
        return _trsv_blk!(up, tr, unit, Int(n), A, x)
    end
    n = Int(n); sx = _start(n, incx)
    el = (i, j) -> cj ? conj(A[i, j]) : A[i, j]
    if !tr                                       # solve A·x = b
        if up
            @inbounds for j in n:-1:1
                unit || _st!(x, sx + (j - 1) * incx, _ld(x, sx + (j - 1) * incx) / A[j, j])
                xj = _ld(x, sx + (j - 1) * incx)
                for i in 1:(j - 1); _st!(x, sx + (i - 1) * incx, _ld(x, sx + (i - 1) * incx) - xj * A[i, j]); end
            end
        else
            @inbounds for j in 1:n
                unit || _st!(x, sx + (j - 1) * incx, _ld(x, sx + (j - 1) * incx) / A[j, j])
                xj = _ld(x, sx + (j - 1) * incx)
                for i in (j + 1):n; _st!(x, sx + (i - 1) * incx, _ld(x, sx + (i - 1) * incx) - xj * A[i, j]); end
            end
        end
    else                                         # solve op(A)ᵀ·x = b
        if up
            @inbounds for j in 1:n
                s = _ld(x, sx + (j - 1) * incx)
                for i in 1:(j - 1); s -= el(i, j) * _ld(x, sx + (i - 1) * incx); end
                unit || (s /= (cj ? conj(A[j, j]) : A[j, j]))
                _st!(x, sx + (j - 1) * incx, s)
            end
        else
            @inbounds for j in n:-1:1
                s = _ld(x, sx + (j - 1) * incx)
                for i in (j + 1):n; s -= el(i, j) * _ld(x, sx + (i - 1) * incx); end
                unit || (s /= (cj ? conj(A[j, j]) : A[j, j]))
                _st!(x, sx + (j - 1) * incx, s)
            end
        end
    end
    return x
end
