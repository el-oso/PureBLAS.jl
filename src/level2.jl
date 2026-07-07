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
# Complex analog: unit-stride, contiguous, matching complex eltypes → the complex-SIMD L2 kernels apply.
@inline function _l2c_simd_ok(A, x, y, incx::Integer, incy::Integer)
    T = eltype(A)
    return incx == 1 && incy == 1 && T <: BlasComplex && eltype(x) === T && eltype(y) === T &&
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

# Complex gemv-N row-tile height (in W-complex vectors). Each y-tile is a Vec{2W} accumulator (AVX2 →
# 2 ymm each). gemvN is ILP-bound: more independent tiles hide Zen3's fma latency — but 2W-wide accs
# eat the 16-ymm file fast, so MR=4 (4 chains, ~14 ymm) is the measured AVX2 optimum (MR=5 spills,
# split-accumulators need MR too small to amortize A). AVX-512's 32 zmm has ample room. Swept per box.
const _CGEMV_MR = @load_preference("cgemv_mr", 4)::Int
# Off-diagonal tri-scatter is a TALL skinny gemv (m≫n=NB); MR=4 that suits square gemvN regresses it
# (worse panel/cache behavior), so the scatter uses its own smaller MR. Decoupled per shape.
const _CGEMV_MR_SCAT = @load_preference("cgemv_mr_scat", 3)::Int
# Complex gemv-T/C column-block width (cols/pass). Each col needs 2 Vec{2W} accs (p,q); AVX2's 16 regs
# → 2, AVX-512 → 4. Sharing xc + its swap across the block is the win (fewer shuffles, x streamed once).
const _CGEMVT_NC = @load_preference("cgemvt_nc", _vwidth(Float64) == 4 ? 2 : 4)::Int
const _CGEMV_NP = 8                                 # column-panel width when A doesn't fit cache
# When A (m×n complex) fits ~L2, sweep all n columns in ONE panel (row-tile mode: A cache-resident, no
# panel/y-restream overhead — faster at small n). Above, width-_CGEMV_NP panels stream A sequentially.
# Threshold keyed to detected L2 (A fits when m·n·sizeof(ComplexF64) ≤ L2) — NOT hardcoded, so Zen3's
# 512 KiB L2 doesn't inherit Zen4's 1 MiB assumption and thrash mid-n (one-panel row-tile re-reads A).
const _CGEMV_RB = @load_preference("cgemv_rb", _L2_BYTES ÷ 16)::Int   # m·n complex threshold for one-panel mode
# AVX2 complex gemvN: OpenBLAS-structure kernel (Fable-decomposed 2026-07-06). The mid-n valley (n=1024
# 0.735) was NOT the shuffle (kb hypothesis, refuted by measurement) nor memory (PB's access streams at
# the L3 ceiling) — it was the per-(row-tile×column) α·x scalar work stealing FMA-port slots + the two
# serial muladds/column forming an 8-cyc loop-carried chain. Fix: NC columns OUTER, rows inner, FRESH
# Pv/Qv accumulators each row-iter (the y-RMW breaks all dep chains), α folded ONCE per column into the
# hoisted x-broadcast (cr,ci = α·x[jj] — NC mults/panel, amortized over m rows, not per row-tile), and a
# +192 B prefetch on each A stream. Measured galen: n=1024 0.735→1.03, sweep 1.00–1.24× OB. Only 2
# shuffles/row-iter (on Q, off the FMA ports). AVX2 only; AVX-512 keeps the row-tile path (already gates).
const _CGEMVN_NC = @load_preference("cgemvn_nc", 4)::Int             # columns per panel (OB uses 4)
const _CGEMVN_PF = @load_preference("cgemvn_pf", _vwidth(Float64) == 4)::Bool  # A-stream prefetch (AVX2)
@generated function _gemv_n_ri_panel!(yp::Ptr{T}, Ab::Ptr{T}, ldc::Int, xp::Ptr{T}, jc::Int, m::Int,
        αr::T, αi::T, ::Val{NC}, ::Val{PF}) where {T, NC, PF}
    W = _vwidth(T); V2 = Vec{2W, T}; sz = sizeof(T)
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(2W - 1))...)
    sgn = Expr(:tuple, (iseven(l) ? -one(T) : one(T) for l in 0:(2W - 1))...)
    body = quote sgnv = $V2($sgn) end
    for c in 1:NC                                   # hoist: α·x[jj] once per column, broadcast re/im
        push!(body.args, :($(Symbol(:b, c)) = Ab + (jc + $(c - 1)) * 2 * ldc * $sz))
        push!(body.args, :(xr = unsafe_load(xp, 2 * (jc + $(c - 1)) + 1); xi = unsafe_load(xp, 2 * (jc + $(c - 1)) + 2)))
        push!(body.args, :($(Symbol(:cr, c)) = $V2(αr * xr - αi * xi)))
        push!(body.args, :($(Symbol(:ci, c)) = $V2(αr * xi + αi * xr)))
    end
    inner = quote off = i * 2 * $sz end
    for c in 1:NC
        av = Symbol(:av, c)
        push!(inner.args, :($av = vload($V2, $(Symbol(:b, c)) + off)))
        PF && push!(inner.args, :(_prefetch($(Symbol(:b, c)) + off + $(3 * 2W * sz))))   # +2W complex (192 B @W=4)
        if c == 1
            push!(inner.args, :(Pv = $av * cr1; Qv = $av * ci1))     # FRESH accumulators (break dep chains)
        else
            push!(inner.args, :(Pv = muladd($av, $(Symbol(:cr, c)), Pv)))
            push!(inner.args, :(Qv = muladd($av, $(Symbol(:ci, c)), Qv)))
        end
    end
    push!(inner.args, :(yv = vload($V2, yp + off)))
    push!(inner.args, :(yv = muladd(shufflevector(Qv, Val($swp)), sgnv, yv + Pv)))
    push!(inner.args, :(vstore(yv, yp + off)))
    tail = quote acr = zero($T); aci = zero($T) end                  # scalar row tail (m % W complex rows)
    for c in 1:NC
        push!(tail.args, quote
            arr = unsafe_load($(Symbol(:b, c)), 2i + 1); aii = unsafe_load($(Symbol(:b, c)), 2i + 2)
            acr += arr * $(Symbol(:cr, c))[1] - aii * $(Symbol(:ci, c))[1]
            aci += arr * $(Symbol(:ci, c))[1] + aii * $(Symbol(:cr, c))[1]
        end)
    end
    push!(tail.args, quote
        unsafe_store!(yp, unsafe_load(yp, 2i + 1) + acr, 2i + 1)
        unsafe_store!(yp, unsafe_load(yp, 2i + 2) + aci, 2i + 2)
    end)
    push!(body.args, :(i = 0))
    push!(body.args, :(while i + $W <= m; $inner; i += $W; end))
    push!(body.args, :(while i < m; $tail; i += 1; end))
    push!(body.args, :(return nothing))
    return body
end
# AVX2 complex gemvN driver: β-prescale y, then NC-column panels over full m (no m-blocking — full-m
# column streams prefetch best; tall y-beyond-L2 shapes stay on the row-tile path via the caller).
function _gemv_n_ri_cmplx!(m::Int, n::Int, α::Complex{T}, A, x, y, β::Complex{T}, ::Val{B0}) where {T<:BlasReal, B0}
    NC = _CGEMVN_NC; sz = sizeof(T); αr = real(α); αi = imag(α)
    # y is restreamed once per NC-column panel (n/NC times) → block m so the y-block fits ~½ L2 for tall
    # shapes; square mid-n (16m ≤ ½L2) runs one block (NB=m), which measured fastest (prefetch continuity).
    NB = (2 * m * sz <= _L2_BYTES ÷ 2) ? m : max(NC, (_L2_BYTES ÷ 2) ÷ (2 * sz))
    GC.@preserve A x y begin
        Ap = Ptr{T}(pointer(A)); yp = Ptr{T}(pointer(y)); xp = Ptr{T}(pointer(x)); ldc = stride(A, 2)
        if B0
            @inbounds for i in 1:(2m); unsafe_store!(yp, zero(T), i); end
        elseif !isone(β)
            _scal_cmplx_simd!(m, real(β), imag(β), y)
        end
        i0 = 0
        while i0 < m
            mb = min(NB, m - i0); ypb = yp + i0 * 2 * sz; Apb = Ap + i0 * 2 * sz
            jc = 0
            while jc + NC <= n
                _CGEMVN_PF ? _gemv_n_ri_panel!(ypb, Apb, ldc, xp, jc, mb, αr, αi, Val(_CGEMVN_NC), Val(true)) :
                             _gemv_n_ri_panel!(ypb, Apb, ldc, xp, jc, mb, αr, αi, Val(_CGEMVN_NC), Val(false))
                jc += NC
            end
            while jc < n
                _gemv_n_ri_panel!(ypb, Apb, ldc, xp, jc, mb, αr, αi, Val(1), Val(false))
                jc += 1
            end
            i0 += mb
        end
    end
    return y
end

# Complex gemv-N panel block: accumulate columns [jc, jc+Peff) of A into MR row-tiles of W complex, RMW

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
function _gemv_n_cmplx!(m::Int, n::Int, α::Complex{T}, A, x, y, β::Complex{T}, ::Val{B0},
        ::Val{MR} = Val(_CGEMV_MR)) where {T<:BlasReal, B0, MR}
    W = _vwidth(T); mr = MR * W; sz = sizeof(T); αr = real(α); αi = imag(α)
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
                _gemv_n_block_cmplx!(yp + i0 * 2 * sz, Ap + i0 * 2 * sz, ldc, xp, jc, Peff, αr, αi, Val(MR)); i0 += mr
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
# One column-block of gemv-T/C: NC columns share each x W-chunk AND its swap (1 shuffle feeds NC cols),
# and x is streamed once per block instead of re-read per column. Reduction mirrors _dot_cmplx_simd.
@generated function _gemv_tc_block_cmplx!(yp::Ptr{Complex{T}}, Ab::Ptr{Complex{T}}, lda::Int,
        xp::Ptr{Complex{T}}, m::Int, α::Complex{T}, β::Complex{T}, z::Bool,
        ::Val{NC}, ::Val{CJ}) where {T, NC, CJ}
    W = _vwidth(T); V2 = Vec{2W, T}; sz = sizeof(T)
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(2W - 1))...)
    body = quote
        xr = Ptr{$T}(xp); Ar = Ptr{$T}(Ab)
    end
    for c in 1:NC; push!(body.args, :($(Symbol(:p, c)) = zero($V2); $(Symbol(:q, c)) = zero($V2))); end
    main = quote
        xc = vload($V2, xr + i * 2 * $sz); xcs = shufflevector(xc, Val($swp))
    end
    for c in 1:NC
        av = Symbol(:av, c)
        push!(main.args, :($av = vload($V2, Ar + (i + $(c - 1) * lda) * 2 * $sz)))
        push!(main.args, :($(Symbol(:p, c)) = muladd($av, xc, $(Symbol(:p, c)))))
        push!(main.args, :($(Symbol(:q, c)) = muladd($av, xcs, $(Symbol(:q, c)))))
    end
    push!(body.args, :(nfull = m - rem(m, $W); i = 0; @inbounds while i < nfull; $main; i += $W; end))
    for c in 1:NC
        push!(body.args, quote
            pr, pii = _deint_cmplx($(Symbol(:p, c))); qr, qi = _deint_cmplx($(Symbol(:q, c)))
            $(Symbol(:sr, c)) = sum(pr) + $(CJ ? :(sum(pii)) : :(-sum(pii)))
            $(Symbol(:si, c)) = sum(qr) + $(CJ ? :(-sum(qi)) : :(sum(qi)))
        end)
    end
    tail = quote xrr = unsafe_load(xr, 2i + 1); xii = unsafe_load(xr, 2i + 2) end
    for c in 1:NC
        push!(tail.args, quote
            arr = unsafe_load(Ar, 2 * (i + $(c - 1) * lda) + 1); aii = unsafe_load(Ar, 2 * (i + $(c - 1) * lda) + 2)
            $(Symbol(:sr, c)) += arr * xrr + $(CJ ? :(aii * xii) : :(-aii * xii))
            $(Symbol(:si, c)) += arr * xii + $(CJ ? :(-aii * xrr) : :(aii * xrr))
        end)
    end
    push!(body.args, :(@inbounds while i < m; $tail; i += 1; end))
    for c in 1:NC
        push!(body.args, quote
            s = Complex($(Symbol(:sr, c)), $(Symbol(:si, c)))
            yj = unsafe_load(yp, $c)
            unsafe_store!(yp, z ? α * s : muladd(β, yj, α * s), $c)
        end)
    end
    push!(body.args, :(return nothing))
    body
end

function _gemv_tc_cmplx!(m::Int, n::Int, α::Complex{T}, A, x, β::Complex{T}, y, ::Val{CJ}) where {T<:BlasReal, CJ}
    z = iszero(β); csz = sizeof(Complex{T}); NC = _CGEMVT_NC
    GC.@preserve A x y begin
        Ap = pointer(A); lda = stride(A, 2); xp = pointer(x); yp = pointer(y)
        j = 0
        while j + NC <= n                                         # NC-column blocks (shared x + swap)
            _gemv_tc_block_cmplx!(yp + j * csz, Ap + j * lda * csz, lda, xp, m, α, β, z, Val(NC), Val(CJ))
            j += NC
        end
        @inbounds while j < n                                     # remainder columns: per-column dot
            colp = Ap + j * lda * csz
            s = _dot_cmplx_simd(m, colp, xp, T, Val(CJ))
            yj = y[j + 1]; y[j + 1] = (z ? zero(yj) : β * yj) + α * s
            j += 1
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
        if _l2c_ok(A, x, y, incx, incy)       # complex N
            αc = convert(eltype(A), α); βc = convert(eltype(A), β)
            if _vwidth(real(eltype(A))) == 4  # AVX2 → OB-structure ri kernel (fresh accs, α-in-broadcast)
                return iszero(β) ? _gemv_n_ri_cmplx!(Int(m), Int(n), αc, A, x, y, βc, Val(true)) :
                    _gemv_n_ri_cmplx!(Int(m), Int(n), αc, A, x, y, βc, Val(false))
            end
            return iszero(β) ? _gemv_n_cmplx!(Int(m), Int(n), αc, A, x, y, βc, Val(true)) :   # AVX-512 row-tile
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
# Fused hemv column-segment kernel: over L complex of A-column ap (with x-segment xp, y-segment yp),
# does y += tmp·a (complex axpy, swap-pairs) AND accumulates s += conj(a)·x (interleaved products),
# reading a ONCE. Returns (sr, si) = the conj-dot. tmp = α·x[j] (complex scalar).
@generated function _hemv_col_cmplx!(L::Int, tmpr::T, tmpi::T, ap::Ptr{T}, xp::Ptr{T}, yp::Ptr{T}) where {T<:BlasReal}
    W = _vwidth(T); V2 = Vec{2W, T}; sz = sizeof(T)
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(2W - 1))...)
    quote
        tmprv = $V2(tmpr); tmpsgn = $V2($(Expr(:tuple, (iseven(l) ? :(-tmpi) : :tmpi for l in 0:(2W - 1))...)))
        pacc = zero($V2); qacc = zero($V2); i = 0
        while i + $W <= L
            o = i * 2 * $sz; av = vload($V2, ap + o)
            yv = vload($V2, yp + o)
            yv = muladd(av, tmprv, yv); yv = muladd(shufflevector(av, Val($swp)), tmpsgn, yv)
            vstore(yv, yp + o)
            xv = vload($V2, xp + o)
            pacc = muladd(av, xv, pacc); qacc = muladd(av, shufflevector(xv, Val($swp)), qacc)
            i += $W
        end
        pr, pi = _deint_cmplx(pacc); qr, qi = _deint_cmplx(qacc)
        sr = sum(pr) + sum(pi); si = sum(qr) - sum(qi)          # conj(a)·x = (ar·xr+ai·xi) + i(ar·xi−ai·xr)
        while i < L
            j2 = 2i
            ar = unsafe_load(ap, j2 + 1); ai = unsafe_load(ap, j2 + 2)
            xr = unsafe_load(xp, j2 + 1); xi = unsafe_load(xp, j2 + 2)
            unsafe_store!(yp, unsafe_load(yp, j2 + 1) + tmpr * ar - tmpi * ai, j2 + 1)
            unsafe_store!(yp, unsafe_load(yp, j2 + 2) + tmpr * ai + tmpi * ar, j2 + 2)
            sr += ar * xr + ai * xi; si += ar * xi - ai * xr
            i += 1
        end
        return (sr, si)
    end
end

# Complex hemv (A Hermitian, `up` triangle stored, diagonal real): y := β·y + α·A·x. One fused pass over
# each stored column (off-diagonal axpy+conj-dot), then the real-diagonal term. Reuses the fused kernel.
function _hemv_cmplx!(up::Bool, n::Int, α::Complex{T}, A, x, β::Complex{T}, y) where {T<:BlasReal}
    _scale_y!(n, β, y, 1)                                       # β·y (complex scal via _scal_cmplx_simd!)
    iszero(α) && return y
    sz = sizeof(T)
    GC.@preserve A x y begin
        Ap = Ptr{T}(pointer(A)); xp = Ptr{T}(pointer(x)); yp = Ptr{T}(pointer(y))
        Apc = Ptr{Complex{T}}(pointer(A)); xpc = Ptr{Complex{T}}(pointer(x)); ypc = Ptr{Complex{T}}(pointer(y))
        ldc = stride(A, 2)
        @inbounds for j in 1:n
            tmp = α * unsafe_load(xpc, j)
            L = up ? (j - 1) : (n - j)
            sr = zero(T); si = zero(T)
            if L > 0
                off = up ? (j - 1) * 2 * ldc : ((j - 1) * ldc + j) * 2   # A[1,j] (up) or A[j+1,j] (lo)
                seg = up ? 0 : j * 2                                     # x/y segment start (reals)
                sr, si = _hemv_col_cmplx!(L, real(tmp), imag(tmp), Ap + off * sz, xp + seg * sz, yp + seg * sz)
            end
            ajj = unsafe_load(Ap, ((j - 1) * ldc + (j - 1)) * 2 + 1)     # real(A[j,j])
            unsafe_store!(ypc, unsafe_load(ypc, j) + tmp * ajj + α * Complex{T}(sr, si), j)
        end
    end
    return y
end

function _hemv!(up::Bool, n::Integer, α::Number, A, x, incx::Integer, β::Number, y, incy::Integer)
    if incx == 1 && incy == 1 && _l2c_ok(A, x, y, incx, incy)
        return _hemv_cmplx!(up, Int(n), convert(eltype(A), α), A, x, convert(eltype(A), β), y)
    end
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
# COMPLEX tri unblocked threshold. The blocked off-diagonal scatter goes through the complex gemv; on
# AVX-512 its per-call/shuffle overhead made per-column faster ≤1024. On AVX2 the scatter now uses the
# fast OB-structure ri gemv (see _tri_scat_cmplx!), so blocking wins earlier — the unblocked column-axpy
# re-streams x and dips at n=1024–2048 (0.83–0.86); route those to blocked+ri. n≤512 stays unblocked
# (gates 0.96–1.53). Sweep the crossover per box via the knob.
const _TRI_C_BLK_MIN = @load_preference("tri_c_blk_min", _vwidth(Float64) == 4 ? 256 : 1024)::Int

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
# Complex unit-stride single-vector-op eligibility (trmv/trsv).
@inline function _l2vc_ok(A, x, incx::Integer)
    T = eltype(A)
    return incx == 1 && T <: BlasComplex && eltype(x) === T &&
        _strided1(A) && x isa StridedVector && stride(x, 1) == 1
end

# Barrier: resolve the runtime conj flag to a compile-time Val so _dot_cmplx_simd (@generated on Val{CJ})
# doesn't dynamic-dispatch. Both branches return Complex{T} → type-stable.
@inline _dot_cmplx_disp(L::Int, ap, xp, ::Type{T}, cj::Bool) where {T<:BlasReal} =
    cj ? _dot_cmplx_simd(L, ap, xp, T, Val(true)) : _dot_cmplx_simd(L, ap, xp, T, Val(false))

# Complex trmv (x := op(A)·x, A triangular, in place). N forms are per-column complex axpys into x; T/C
# forms are per-column complex dots — reusing the gating L1 kernels (like ger/gemv-T). Diagonal scalar.
function _trmv_cmplx!(up::Bool, tr::Bool, cj::Bool, unit::Bool, n::Int, A, x) where {}
    T = real(eltype(A)); csz = sizeof(Complex{T})
    GC.@preserve A x begin
        Ap = Ptr{Complex{T}}(pointer(A)); xp = Ptr{Complex{T}}(pointer(x)); ldc = stride(A, 2)
        djj(j) = (a = unsafe_load(Ap, (j - 1) * ldc + j); cj ? conj(a) : a)
        colp(r, j) = Ap + ((j - 1) * ldc + (r - 1)) * csz
        if !tr                                               # x := A·x, column axpy
            if up
                @inbounds for j in 1:n
                    xj = unsafe_load(xp, j)
                    j > 1 && _axpy_cmplx_simd!(j - 1, real(xj), imag(xj), colp(1, j), xp)
                    unit || unsafe_store!(xp, xj * unsafe_load(Ap, (j - 1) * ldc + j), j)
                end
            else
                @inbounds for j in n:-1:1
                    xj = unsafe_load(xp, j)
                    j < n && _axpy_cmplx_simd!(n - j, real(xj), imag(xj), colp(j + 1, j), xp + j * csz)
                    unit || unsafe_store!(xp, xj * unsafe_load(Ap, (j - 1) * ldc + j), j)
                end
            end
        else                                                 # x := op(A)ᵀ·x, column dot
            if up
                @inbounds for j in n:-1:1
                    s = unit ? unsafe_load(xp, j) : unsafe_load(xp, j) * djj(j)
                    j > 1 && (s += _dot_cmplx_disp(j - 1, colp(1, j), xp, T, cj))
                    unsafe_store!(xp, s, j)
                end
            else
                @inbounds for j in 1:n
                    s = unit ? unsafe_load(xp, j) : unsafe_load(xp, j) * djj(j)
                    j < n && (s += _dot_cmplx_disp(n - j, colp(j + 1, j), xp + j * csz, T, cj))
                    unsafe_store!(xp, s, j)
                end
            end
        end
    end
    return x
end

# Reciprocal-of-diagonal scratch for complex trsv (per real type; single-thread — no MT here). The
# in-loop complex DIVIDE sits on the sequential substitution CRITICAL PATH (each x[j] feeds the next
# column's axpy/dot) — Julia's Complex `/` (Smith's robust algorithm) latency is fully exposed and
# dominates small n (ztrsv n≤256 was 0.65–0.96 vs OB; ztrmv, SAME axpy but NO divide, gates). Precompute
# r[j]=1/diag up front (all independent → pipelined, throughput-bound) with a NAIVE reciprocal (the trsv
# diagonal is well-conditioned; BLAS doesn't overflow-guard the inner divide), then MULTIPLY in the loop.
# n ≤ _TRI_C_BLK_MIN (256) unblocked / 64-block ⇒ 512 covers it (else fall back to the divide).
const _TRSV_RCP64 = Vector{ComplexF64}(undef, 512)
const _TRSV_RCP32 = Vector{ComplexF32}(undef, 512)
@inline _trsv_rcpbuf(::Type{Float64}) = _TRSV_RCP64
@inline _trsv_rcpbuf(::Type{Float32}) = _TRSV_RCP32
@inline _crecip(d::Complex) = (r = real(d); i = imag(d); s = inv(muladd(r, r, i * i)); Complex(r * s, -i * s))

# Complex trsv (solve op(A)·x = x in place). N forms = column substitution (axpy of −xⱼ into the rest);
# T/C forms = dot-based row substitution. Diagonal reciprocals precomputed off the critical path.
function _trsv_cmplx!(up::Bool, tr::Bool, cj::Bool, unit::Bool, n::Int, A, x) where {}
    T = real(eltype(A)); csz = sizeof(Complex{T})
    userc = !unit && n <= 512                                # precompute reciprocals off the crit path
    GC.@preserve A x begin
        Ap = Ptr{Complex{T}}(pointer(A)); xp = Ptr{Complex{T}}(pointer(x)); ldc = stride(A, 2)
        djj(j) = (a = unsafe_load(Ap, (j - 1) * ldc + j); cj ? conj(a) : a)
        colp(r, j) = Ap + ((j - 1) * ldc + (r - 1)) * csz
        rcp = _trsv_rcpbuf(T)
        if userc                                             # r[j] = 1/diag (naive, pipelined)
            if !tr
                @inbounds for j in 1:n; rcp[j] = _crecip(unsafe_load(Ap, (j - 1) * ldc + j)); end
            else
                @inbounds for j in 1:n; rcp[j] = _crecip(djj(j)); end
            end
        end
        if !tr                                               # op = A: column-oriented substitution
            if up                                            # back-substitution (j descending)
                @inbounds for j in n:-1:1
                    unit || unsafe_store!(xp, userc ? unsafe_load(xp, j) * rcp[j] : unsafe_load(xp, j) / unsafe_load(Ap, (j - 1) * ldc + j), j)
                    xj = unsafe_load(xp, j)
                    j > 1 && _axpy_cmplx_simd!(j - 1, real(-xj), imag(-xj), colp(1, j), xp)
                end
            else                                             # forward-substitution (j ascending)
                @inbounds for j in 1:n
                    unit || unsafe_store!(xp, userc ? unsafe_load(xp, j) * rcp[j] : unsafe_load(xp, j) / unsafe_load(Ap, (j - 1) * ldc + j), j)
                    xj = unsafe_load(xp, j)
                    j < n && _axpy_cmplx_simd!(n - j, real(-xj), imag(-xj), colp(j + 1, j), xp + j * csz)
                end
            end
        else                                                 # op = Aᵀ: dot-based row substitution
            if up                                            # forward (j ascending)
                @inbounds for j in 1:n
                    s = unsafe_load(xp, j)
                    j > 1 && (s -= _dot_cmplx_disp(j - 1, colp(1, j), xp, T, cj))
                    unsafe_store!(xp, unit ? s : (userc ? s * rcp[j] : s / djj(j)), j)
                end
            else                                             # backward (j descending)
                @inbounds for j in n:-1:1
                    s = unsafe_load(xp, j)
                    j < n && (s -= _dot_cmplx_disp(n - j, colp(j + 1, j), xp + j * csz, T, cj))
                    unsafe_store!(xp, unit ? s : (userc ? s * rcp[j] : s / djj(j)), j)
                end
            end
        end
    end
    return x
end

# Complex off-diagonal scatters for blocked trmv/trsv: y += α·op(Av)·xv (β=1 accumulate), reusing the
# gating complex gemv kernels. N → gemv-N; T/C → gemv-T/C with cj resolved to a compile-time Val.
@inline _tri_scat_cmplx!(yv, Av, xv, α::T) where {T} =
    _vwidth(real(T)) == 4 ?     # AVX2: the OB-structure ri gemv (α folded, fresh accs, prefetch, m-blocked)
        _gemv_n_ri_cmplx!(size(Av, 1), size(Av, 2), α, Av, xv, yv, one(T), Val(false)) :
        _gemv_n_cmplx!(size(Av, 1), size(Av, 2), α, Av, xv, yv, one(T), Val(false), Val(_CGEMV_MR_SCAT))
@inline _tri_scatT_cmplx!(yv, Av, xv, α::T, cj::Bool) where {T} =
    cj ? _gemv_tc_cmplx!(size(Av, 1), size(Av, 2), α, Av, xv, one(T), yv, Val(true)) :
         _gemv_tc_cmplx!(size(Av, 1), size(Av, 2), α, Av, xv, one(T), yv, Val(false))

# Blocked complex trmv/trsv (mirror of the real _trmv_blk!/_trsv_blk!): per-column kernel re-streams x
# at large n and serializes across columns. Block it — NB×NB diagonal via the per-column complex kernel
# (cache-resident), off-diagonal via the gating complex gemv (reads A once, no re-stream, not serialized).
@inline function _trmv_cmplx_blk!(up::Bool, tr::Bool, cj::Bool, unit::Bool, n::Int, A, x)
    NB = _TRI_NB
    n <= _TRI_C_BLK_MIN && return _trmv_cmplx!(up, tr, cj, unit, n, A, x)
    T = eltype(A)
    @inbounds if !tr && up               # U,N: J ascending; tall scatter UP then diag
        ib = 0
        while ib < n
            nb = min(NB, n - ib); J = (ib + 1):(ib + nb)
            ib > 0 && _tri_scat_cmplx!(view(x, 1:ib), view(A, 1:ib, J), view(x, J), one(T))
            _trmv_cmplx!(true, false, cj, unit, nb, view(A, J, J), view(x, J))
            ib += NB
        end
    elseif !tr && !up                    # L,N: J descending; tall scatter DOWN then diag
        ib = (cld(n, NB) - 1) * NB
        while ib >= 0
            nb = min(NB, n - ib); J = (ib + 1):(ib + nb); je = ib + nb
            je < n && _tri_scat_cmplx!(view(x, (je + 1):n), view(A, (je + 1):n, J), view(x, J), one(T))
            _trmv_cmplx!(false, false, cj, unit, nb, view(A, J, J), view(x, J))
            ib -= NB
        end
    elseif tr && up                      # U,T/C: I descending; diag(op) then gemv-T/C (rows above)
        ib = (cld(n, NB) - 1) * NB
        while ib >= 0
            nb = min(NB, n - ib); I = (ib + 1):(ib + nb)
            _trmv_cmplx!(true, true, cj, unit, nb, view(A, I, I), view(x, I))
            ib > 0 && _tri_scatT_cmplx!(view(x, I), view(A, 1:ib, I), view(x, 1:ib), one(T), cj)
            ib -= NB
        end
    else                                 # L,T/C: I ascending; diag(op) then gemv-T/C (rows below)
        ib = 0
        while ib < n
            nb = min(NB, n - ib); I = (ib + 1):(ib + nb)
            _trmv_cmplx!(false, true, cj, unit, nb, view(A, I, I), view(x, I))
            ib + nb < n && _tri_scatT_cmplx!(view(x, I), view(A, (ib + nb + 1):n, I), view(x, (ib + nb + 1):n), one(T), cj)
            ib += NB
        end
    end
    return x
end

@inline function _trsv_cmplx_blk!(up::Bool, tr::Bool, cj::Bool, unit::Bool, n::Int, A, x)
    NB = _TRI_NB
    (n <= _TRI_C_BLK_MIN || (tr && n <= _TRI_T_UNB)) && return _trsv_cmplx!(up, tr, cj, unit, n, A, x)
    T = eltype(A)
    @inbounds if !tr && up               # U,N back: J descending; solve diag then tall scatter UP (−)
        ib = (cld(n, NB) - 1) * NB
        while ib >= 0
            nb = min(NB, n - ib); J = (ib + 1):(ib + nb)
            _trsv_cmplx!(true, false, cj, unit, nb, view(A, J, J), view(x, J))
            ib > 0 && _tri_scat_cmplx!(view(x, 1:ib), view(A, 1:ib, J), view(x, J), -one(T))
            ib -= NB
        end
    elseif !tr && !up                    # L,N fwd: J ascending; solve diag then tall scatter DOWN (−)
        ib = 0
        while ib < n
            nb = min(NB, n - ib); J = (ib + 1):(ib + nb); je = ib + nb
            _trsv_cmplx!(false, false, cj, unit, nb, view(A, J, J), view(x, J))
            je < n && _tri_scat_cmplx!(view(x, (je + 1):n), view(A, (je + 1):n, J), view(x, J), -one(T))
            ib += NB
        end
    elseif tr && up                      # U,T/C fwd: I ascending; gemv-T/C(−, above) then solve(op)
        ib = 0
        while ib < n
            nb = min(NB, n - ib); I = (ib + 1):(ib + nb)
            ib > 0 && _tri_scatT_cmplx!(view(x, I), view(A, 1:ib, I), view(x, 1:ib), -one(T), cj)
            _trsv_cmplx!(true, true, cj, unit, nb, view(A, I, I), view(x, I))
            ib += NB
        end
    else                                 # L,T/C back: I descending; gemv-T/C(−, below) then solve(op)
        ib = (cld(n, NB) - 1) * NB
        while ib >= 0
            nb = min(NB, n - ib); I = (ib + 1):(ib + nb)
            ib + nb < n && _tri_scatT_cmplx!(view(x, I), view(A, (ib + nb + 1):n, I), view(x, (ib + nb + 1):n), -one(T), cj)
            _trsv_cmplx!(false, true, cj, unit, nb, view(A, I, I), view(x, I))
            ib -= NB
        end
    end
    return x
end

function _trmv!(up::Bool, tr::Bool, cj::Bool, unit::Bool, n::Integer, A, x, incx::Integer)
    if _l2v_simd_ok(A, x, incx)
        return _trmv_blk!(up, tr, unit, Int(n), A, x)
    end
    _l2vc_ok(A, x, incx) && return _trmv_cmplx_blk!(up, tr, cj, unit, Int(n), A, x)
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
    _l2vc_ok(A, x, incx) && return _trsv_cmplx_blk!(up, tr, cj, unit, Int(n), A, x)
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
