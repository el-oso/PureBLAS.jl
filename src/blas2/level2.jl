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

# gemv-N row-block height in vectors (mr = _GEMV_MR·W rows). MR=8 keeps 8 accumulators feeding both FMA
# units to cover the ~5-cyc latency; MR=4 half-fills the pipe at cache-resident mid-n. Double-pumped Zen4
# occupies each 512-bit pipe TWICE, self-hiding the latency → MR=4 suffices. AVX2 (Zen3) and NATIVE-512
# (Zen5) re-expose it → MR=8 (Zen5 gemvN@256 was 0.86 at MR=4). Keyed on _double_pumped (silicon fact).
const _GEMV_MR = _double_pumped(_HW) ? 4 : (_vwidth(Float64) >= 4 ? 8 : 4)

const _GEMV_NP = 8             # gemv-N column-panel width
# ── gemvN m-inner panel (OpenBLAS dgemv_n shape; see _gemv_n_paneldrv_minner!). The old panel path holds
# a full row-block's y in registers and sweeps columns inner, so each of NP=8 A-columns is read in mr-row
# bursts with big gaps → the HW prefetcher can't sustain 8 strided streams (dips at first L3-resident size:
# galen 512 0.92, Zen5 2048 0.91). OB inverts it: m-block small enough that the y-block stays L1-resident,
# columns grouped per panel, inner loop streams m DOWN each column (tight, continuous per-column streams
# the prefetchers lock onto), x broadcasts hoisted, y RMW'd from L1. HELPS on AVX2 (narrow → the extra
# per-column streams win) and on double-pumped-512 Zen4 (each 512-bit pipe is occupied twice, hiding the
# y-restream cost); REGRESSES on NATIVE-512 Zen5 (single-pumped ports already saturate, so the y-restream
# is pure overhead — full-sweep worst n=1024 0.91→0.85, both geomean AND worst below the old NP=8 path).
# So gate on the DATAPATH, not a flat default: minner ON iff the vector unit is AVX2 (W<8) OR double-pumped,
# OFF on native-512 (physical criterion over detected consts — CLAUDE.md req#7/#8; validated full-sweep on
# the fleet: Zen3/Zen4 keep the gains, Zen5 reverts to the old path). Panel-width regimes below apply where on.
const _GEMVN_MINNER = @load_preference("gemvn_minner", _vwidth(Float64) < 8 || _double_pumped(_HW))::Bool
const _GEMVN_MINNER_U = 4    # row-vector unroll (U·W rows/step): independent y-accumulators to cover FMA latency (ILP)
# Panel width (columns/panel = concurrent A-read streams; y re-streamed n/NP times) — three regimes:
#  narrow  A ≤ 2·L2 (partially L2-resident band, e.g. f64 n=512 = 2 MB): few streams win. NP8 = 0.95 vs OB,
#          NP5/6 ≈ 1.00 (16-round gate check; 5 vs 6 within noise). Mechanism: with A partly L2-resident the
#          L2→L1 feed, not MLP, limits — more concurrent streams only thrash the DL1/its prefetcher.
#          Empirical width: assoc−2 (streams + y + slack fit the 8-way L1); Preferences-overridable.
#  aliased lda·sizeof ≡ 0 mod L1-way (po2 lda: 1024/1536/2048 f64): ALL NP streams index the SAME L1 set,
#          so NP is capped at the associativity (8-way: NP12 1.027/1.264 vs NP8 1.067/1.283 @1024/2048).
#          Proof it's aliasing, not size: de-aliased via lda+8 pad @1024, NP12 1.107 > NP8 1.079 flips back.
#  wide    otherwise: NP12 — fewer y-RMW re-streams; the extra streams cost nothing (streams spread over
#          ≥2 sets: worst case s=way/2 → 12/2+1 = 7 ≤ 8 ways). n=768: NP8 0.978 → NP12 1.019 (A-only read
#          runs 69-71 GB/s vs OB's 67 total, so the y-restream tax was the whole gap). Register-capped:
#          NP+U+1 live vectors = 17 ≤ 32 on AVX-512; on 16-reg ISAs 12+4+1 spills → cap at assoc (=8, the
#          fleet-verified AVX2 config).
const _GEMVN_NP_NARROW = @load_preference("gemvn_np_narrow", max(2, _L1D_ASSOC - 2))::Int
const _GEMVN_NP_WIDE = @load_preference(
    "gemvn_np_wide",
    _NVREG >= 32 ? min(12, _NVREG - _GEMVN_MINNER_U - 1) : _L1D_ASSOC
)::Int
@inline function _gemvn_minner_np(m::Int, n::Int, lda::Int, ::Type{T}) where {T}
    m * n * sizeof(T) <= 2 * _L2_BYTES && return _GEMVN_NP_NARROW
    (lda * sizeof(T)) % _L1_WAY_BYTES == 0 && return min(_L1D_ASSOC, _GEMVN_NP_WIDE)
    return _GEMVN_NP_WIDE
end
const _GEMVN_MB = @load_preference("gemvn_mb", max(_vwidth(Float64), _L1_BYTES ÷ 2 ÷ sizeof(Float64)))::Int  # m-block: y-block ≤ ½L1 stays resident while sweeping all n columns
# minner helps the mid-n/L3 regime (measured Zen4 PB-self: n=512-2048 ~8-10% faster) but the y-restream
# regresses deep-DRAM n (4096 ~16% slower, where the old NP=8 path already gates 1.31×). So cap minner to
# A ≲ a few × L3 and fall back to the old panel path beyond. Crossover is unmeasured on locked HW → tune.
const _GEMVN_MINNER_MAXA = @load_preference("gemvn_minner_maxa", 4 * _L3_BYTES)::Int  # max A bytes (m·n·sizeof) for minner
const _GEMVN_RB = @load_preference("gemvn_rb", _vwidth(Float64) == 4 ? 64 : 448)::Int  # gemv-N: n ≤ this → row-block; larger → column-panel. AVX2 cut dropped 192→64: with _GEMV_MR=8 the sequential-streaming panel path now beats strided row-block for all n≥96 (128: 0.92→1.0); row-block only wins at n≤64 where panel's m<mr all-masked tail dominates. Zen4 1MB L2 → 448.
#                                unmasked full-block kernel, dominates per-column at every n ≥ 512,
#                                incl. the n=512 power-of-2 / just-over-L2 case → 0.96×).
# gemv-N (column-major A makes it transpose-like — see kb finding): two regimes —
#   small n: row-block (y in registers across all cols; A strided but cache-resident),
#   else:    column-panel (accumulate _GEMV_NP cols/pass → y re-streamed n/_GEMV_NP times, A in
#            _GEMV_NP sequential streams; an unmasked full-block kernel makes it dominate per-column
#            at every n ≥ 512, incl. the huge-n y-restream that the per-column path lost at n=4096).

# Row-block: full block (β folded) + masked remainder for any m.
@generated function _gemv_n_block!(
        yb::Ptr{T}, Ab::Ptr{T}, lda::Int, xp::Ptr{T}, n::Int,
        α::T, β::T, ::Val{MR}, ::Val{B0}
    ) where {T, MR, B0}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T)
    body = quote end
    if B0
        for v in 1:MR
            push!(body.args, :($(Symbol(:c, v)) = zero($V)))
        end
    else
        push!(body.args, :(bv = $V(β)))
        for v in 1:MR
            push!(body.args, :($(Symbol(:c, v)) = bv * vload($V, yb + $((v - 1) * W * sz))))
        end
    end
    push!(body.args, :(av = $V(α)))
    inner = quote end
    push!(inner.args, :(xj = av * $V(unsafe_load(xp, j + 1))))
    for v in 1:MR
        push!(inner.args, :($(Symbol(:c, v)) = muladd(vload($V, Ab + ($((v - 1) * W) + j * lda) * $sz), xj, $(Symbol(:c, v)))))
    end
    push!(
        body.args, :(
            for j in 0:(n - 1)
                $inner
            end
        )
    )
    for v in 1:MR
        push!(body.args, :(vstore($(Symbol(:c, v)), yb + $((v - 1) * W * sz))))
    end
    push!(body.args, :(return nothing))
    return body
end

@generated function _gemv_n_block_masked!(
        yb::Ptr{T}, Ab::Ptr{T}, lda::Int, xp::Ptr{T}, n::Int,
        α::T, β::T, mrows::Int, ::Val{NV}, ::Val{B0}
    ) where {T, NV, B0}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T)
    body = quote end
    lanetuple = Expr(:tuple, (0:(W - 1))...)
    push!(body.args, :(lanes = Vec{$W, Int}($lanetuple)))
    for v in 1:NV
        push!(body.args, :($(Symbol(:k, v)) = (lanes + $((v - 1) * W)) < mrows))
    end
    if B0
        for v in 1:NV
            push!(body.args, :($(Symbol(:c, v)) = zero($V)))
        end
    else
        push!(body.args, :(bv = $V(β)))
        for v in 1:NV
            push!(body.args, :($(Symbol(:c, v)) = bv * vload($V, yb + $((v - 1) * W * sz), $(Symbol(:k, v)))))
        end
    end
    push!(body.args, :(av = $V(α)))
    inner = quote end
    push!(inner.args, :(xj = av * $V(unsafe_load(xp, j + 1))))
    for v in 1:NV
        push!(inner.args, :($(Symbol(:c, v)) = muladd(vload($V, Ab + ($((v - 1) * W) + j * lda) * $sz, $(Symbol(:k, v))), xj, $(Symbol(:c, v)))))
    end
    push!(
        body.args, :(
            for j in 0:(n - 1)
                $inner
            end
        )
    )
    for v in 1:NV
        push!(body.args, :(vstore($(Symbol(:c, v)), yb + $((v - 1) * W * sz), $(Symbol(:k, v)))))
    end
    push!(body.args, :(return nothing))
    return body
end

@inline function _gemv_n_rowblock!(m::Int, n::Int, α::T, A, x, y, β::T, ::Val{B0}) where {T <: BlasReal, B0}
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
@generated function _gemv_n_panel!(
        yb::Ptr{T}, Ab::Ptr{T}, lda::Int, xp::Ptr{T}, jc::Int, Peff::Int,
        mrows::Int, α::T, ::Val{NV}
    ) where {T, NV}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T)
    body = quote end
    lanetuple = Expr(:tuple, (0:(W - 1))...)
    push!(body.args, :(lanes = Vec{$W, Int}($lanetuple)))
    for v in 1:NV
        push!(body.args, :($(Symbol(:k, v)) = (lanes + $((v - 1) * W)) < mrows))
    end
    push!(body.args, :(av = $V(α)))
    for v in 1:NV
        push!(body.args, :($(Symbol(:c, v)) = vload($V, yb + $((v - 1) * W * sz), $(Symbol(:k, v)))))
    end
    inner = quote end
    push!(inner.args, :(xj = av * $V(unsafe_load(xp, jc + cc + 1))))
    for v in 1:NV
        push!(inner.args, :($(Symbol(:c, v)) = muladd(vload($V, Ab + ($((v - 1) * W) + (jc + cc) * lda) * $sz, $(Symbol(:k, v))), xj, $(Symbol(:c, v)))))
    end
    push!(
        body.args, :(
            for cc in 0:(Peff - 1)
                $inner
            end
        )
    )
    for v in 1:NV
        push!(body.args, :(vstore($(Symbol(:c, v)), yb + $((v - 1) * W * sz), $(Symbol(:k, v)))))
    end
    push!(body.args, :(return nothing))
    return body
end

# Unmasked full-row-block variant (mre == mr): the common case — no mask overhead. Needed so the
# panel path is competitive at mid n (e.g. n=512), where the masked version's per-block overhead cost
# ~12% vs per-column.
@generated function _gemv_n_panel_full!(
        yb::Ptr{T}, Ab::Ptr{T}, lda::Int, xp::Ptr{T}, jc::Int,
        Peff::Int, α::T, ::Val{MR}
    ) where {T, MR}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T)
    body = quote end
    push!(body.args, :(av = $V(α)))
    for v in 1:MR
        push!(body.args, :($(Symbol(:c, v)) = vload($V, yb + $((v - 1) * W * sz))))
    end
    inner = quote end
    push!(inner.args, :(xj = av * $V(unsafe_load(xp, jc + cc + 1))))
    for v in 1:MR
        push!(inner.args, :($(Symbol(:c, v)) = muladd(vload($V, Ab + ($((v - 1) * W) + (jc + cc) * lda) * $sz), xj, $(Symbol(:c, v)))))
    end
    push!(
        body.args, :(
            for cc in 0:(Peff - 1)
                $inner
            end
        )
    )
    for v in 1:MR
        push!(body.args, :(vstore($(Symbol(:c, v)), yb + $((v - 1) * W * sz))))
    end
    push!(body.args, :(return nothing))
    return body
end

@inline function _gemv_n_paneldrv!(m::Int, n::Int, α::T, A, x, y, β::T, ::Val{B0}) where {T <: BlasReal, B0}
    W = _vwidth(T); mr = _GEMV_MR * W
    GC.@preserve A x y begin
        Aptr = pointer(A); yptr = pointer(y); xptr = pointer(x); lda = stride(A, 2); sz = sizeof(T)
        if B0
            @inbounds for i in 1:m
                unsafe_store!(yptr, zero(T), i)
            end
        elseif β != one(T)
            @inbounds for i in 1:m
                unsafe_store!(yptr, β * unsafe_load(yptr, i), i)
            end
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

# m-inner panel (OB dgemv_n shape): yb[0:mb) += Σ_{c=0}^{NP-1} (α·x[jc+c])·A[:,jc+c], streaming m DOWN
# each column. NP x-broadcasts hoisted; U row-vectors held in registers per step for ILP; masked W-tail.
@generated function _gemv_n_panel_minner!(
        yb::Ptr{T}, Ab::Ptr{T}, lda::Int, xp::Ptr{T},
        jc::Int, mb::Int, α::T, ::Val{NP}, ::Val{U}
    ) where {T, NP, U}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T)
    lanetuple = Expr(:tuple, (0:(W - 1))...)
    body = quote
        av = $V(α)
        lanes = Vec{$W, Int}($lanetuple)
    end
    for c in 1:NP
        push!(body.args, :($(Symbol(:xb, c)) = av * $V(unsafe_load(xp, jc + $c))))
    end   # α·x[jc+c-1]
    main = quote end
    for u in 1:U
        push!(main.args, :($(Symbol(:y, u)) = vload($V, yb + (i + $((u - 1) * W)) * $sz)))
    end
    for c in 1:NP, u in 1:U
        push!(
            main.args, :(
                $(Symbol(:y, u)) = muladd(
                    vload($V, Ab + (i + $((u - 1) * W) + (jc + $(c - 1)) * lda) * $sz), $(Symbol(:xb, c)), $(Symbol(:y, u))
                )
            )
        )
    end
    for u in 1:U
        push!(main.args, :(vstore($(Symbol(:y, u)), yb + (i + $((u - 1) * W)) * $sz)))
    end
    push!(body.args, :(i = 0))
    push!(
        body.args, :(
            while i + $(U * W) <= mb
                $main; i += $(U * W)
            end
        )
    )
    tail = quote
        msk = lanes < (mb - i)
        yt = vload($V, yb + i * $sz, msk)
    end
    for c in 1:NP
        push!(tail.args, :(yt = muladd(vload($V, Ab + (i + (jc + $(c - 1)) * lda) * $sz, msk), $(Symbol(:xb, c)), yt)))
    end
    push!(tail.args, :(vstore(yt, yb + i * $sz, msk)))
    push!(
        body.args, :(
            while i < mb
                $tail; i += $W
            end
        )
    )
    push!(body.args, :(return nothing))
    return body
end

# m-BLOCKED driver: each m-block's y stays ≤½L1 resident while all n columns stream through it once,
# NP columns per panel, m streamed inner (tight per-column streams). β pre-applied per y-block, then
# panels pure-accumulate. This is OpenBLAS's dgemv_n structure (NBMAX m-block × 4-col groups).
# One m-block × all its column panels at compile-time width NP (Val-dispatched by the driver below).
@inline function _gemv_n_mblock_minner!(
        yb::Ptr{T}, Ab0::Ptr{T}, lda::Int, xptr::Ptr{T}, n::Int,
        mb::Int, α::T, ::Val{NP}
    ) where {T, NP}
    jc = 0
    while jc + NP <= n
        _gemv_n_panel_minner!(yb, Ab0, lda, xptr, jc, mb, α, Val(NP), Val(_GEMVN_MINNER_U))
        jc += NP
    end
    while jc < n                                 # column remainder (< NP): 1-column panels
        _gemv_n_panel_minner!(yb, Ab0, lda, xptr, jc, mb, α, Val(1), Val(_GEMVN_MINNER_U))
        jc += 1
    end
    return
end

@inline function _gemv_n_paneldrv_minner!(m::Int, n::Int, α::T, A, x, y, β::T, ::Val{B0}) where {T <: BlasReal, B0}
    GC.@preserve A x y begin
        Aptr = pointer(A); yptr = pointer(y); xptr = pointer(x); lda = stride(A, 2); sz = sizeof(T)
        np = _gemvn_minner_np(m, n, lda, T)      # regime-selected panel width (consts → Val below is static)
        i0 = 0
        while i0 < m
            mb = min(_GEMVN_MB, m - i0)
            yb = yptr + i0 * sz; Ab0 = Aptr + i0 * sz
            if B0                                        # β pre-scale this y-block once; panels then accumulate
                @inbounds for i in 1:mb
                    unsafe_store!(yb, zero(T), i)
                end
            elseif β != one(T)
                @inbounds for i in 1:mb
                    unsafe_store!(yb, β * unsafe_load(yb, i), i)
                end
            end
            if np == _GEMVN_NP_WIDE
                _gemv_n_mblock_minner!(yb, Ab0, lda, xptr, n, mb, α, Val(_GEMVN_NP_WIDE))
            elseif np == _GEMVN_NP_NARROW
                _gemv_n_mblock_minner!(yb, Ab0, lda, xptr, n, mb, α, Val(_GEMVN_NP_NARROW))
            else                                         # aliased-lda cap (≤ L1 associativity)
                _gemv_n_mblock_minner!(yb, Ab0, lda, xptr, n, mb, α, Val(min(_L1D_ASSOC, _GEMVN_NP_WIDE)))
            end
            i0 += mb
        end
    end
    return y
end

@inline function _gemv_n_simd!(m::Int, n::Int, α::T, A, x, y, β::T, ::Val{B0}) where {T <: BlasReal, B0}
    if n <= _GEMVN_RB
        _gemv_n_rowblock!(m, n, α, A, x, y, β, Val(B0))
    elseif _GEMVN_MINNER && m * n * sizeof(T) <= _GEMVN_MINNER_MAXA   # mid-n/L3 regime; large-n DRAM → old path (already gates)
        _gemv_n_paneldrv_minner!(m, n, α, A, x, y, β, Val(B0))
    else
        _gemv_n_paneldrv!(m, n, α, A, x, y, β, Val(B0))
    end
    return y
end

# gemv-T column-block: NC column-dots accumulated together, reusing each x W-chunk across the NC
# columns (one set of horizontal sums per block) — cuts per-column overhead for small n. β folded
# in; masked tail for the row remainder.
@generated function _gemv_t_block!(
        yp::Ptr{T}, Ab::Ptr{T}, lda::Int, xp::Ptr{T}, m::Int,
        α::T, β::T, ::Val{NC}, ::Val{B0}, ::Val{U}
    ) where {T, NC, B0, U}
    # ⚠ NO DEFAULT ARG on a @generated function — juliac --trim fails on the default-arg trampoline
    # (it lowers to an `invoke ::Any`). Every call site passes Val(U) explicitly.
    # ⚠ M-UNROLL. At U=1 the loop does W rows per iteration: ONE x-load, NC FMAs, then increment and
    # compare — bookkeeping amortized over NC=4 useful instructions. AOCL's dgemv-T does 32 rows per
    # iteration (4 chunks of 8) so its overhead rides on 32 FMAs. That is the structural delta that
    # SURVIVED after the column count was falsified on both boxes (see the NC tables in
    # `_measure_gemvt_nc`): more columns hurt, so the remaining suspect is the row loop.
    # Each (column, chunk) gets its OWN accumulator rather than folding chunks into one, which is where
    # this differs from AOCL: shared accumulators would make each iteration a U-deep dependent chain per
    # column, trading loop overhead for latency. Independent accumulators cost NC·U registers and give
    # NC·U independent chains; the caller's `_gemvt_u_max` keeps that inside `_NVREG`.
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T)
    body = quote end
    lanetuple = Expr(:tuple, (0:(W - 1))...)
    push!(body.args, :(lanes = Vec{$W, Int}($lanetuple)))
    for c in 1:NC, u in 1:U
        push!(body.args, :($(Symbol(:a, c, :_, u)) = zero($V)))
    end
    full = quote end
    for u in 1:U
        push!(full.args, :($(Symbol(:xc, u)) = vload($V, xp + (i + $((u - 1) * W)) * $sz)))
    end
    for c in 1:NC, u in 1:U
        push!(
            full.args,
            :($(Symbol(:a, c, :_, u)) = muladd(
                vload($V, Ab + (i + $((u - 1) * W) + $(c - 1) * lda) * $sz), $(Symbol(:xc, u)),
                $(Symbol(:a, c, :_, u))
            ))
        )
    end
    push!(
        body.args, :(
            nfull = m - rem(m, $(U * W)); i = 0; while i < nfull
                $full; i += $(U * W)
            end
        )
    )
    # U>1 leaves up to U-1 whole W-chunks before the masked tail — sweep them at U=1 shape.
    if U > 1
        one_ = quote end
        push!(one_.args, :(xc1 = vload($V, xp + i * $sz)))
        for c in 1:NC
            push!(one_.args, :($(Symbol(:a, c, :_, 1)) = muladd(vload($V, Ab + (i + $(c - 1) * lda) * $sz), xc1, $(Symbol(:a, c, :_, 1)))))
        end
        push!(
            body.args, :(
                n1 = m - rem(m, $W); while i < n1
                    $one_; i += $W
                end
            )
        )
    end
    rmd = quote end
    push!(rmd.args, :(msk = lanes < (m - i)))
    push!(rmd.args, :(xcm = vload($V, xp + i * $sz, msk)))
    for c in 1:NC
        push!(rmd.args, :($(Symbol(:a, c, :_, 1)) = muladd(vload($V, Ab + (i + $(c - 1) * lda) * $sz, msk), xcm, $(Symbol(:a, c, :_, 1)))))
    end
    push!(
        body.args, :(
            if i < m
                $rmd
            end
        )
    )
    for c in 1:NC
        acc = Symbol(:a, c, :_, 1)                      # fold the chunk accumulators, then reduce
        red = U == 1 ? :($acc) : foldl((p, u) -> :($p + $(Symbol(:a, c, :_, u))), 2:U; init = :($acc))
        st = B0 ? :(unsafe_store!(yp, α * sc, $c)) :
            :(unsafe_store!(yp, muladd(β, unsafe_load(yp, $c), α * sc), $c))
        push!(body.args, :(sc = sum($red); $st))
    end
    push!(body.args, :(return nothing))
    return body
end

# gemv-T: column-block (4 cols/pass sharing each x W-chunk) for all n. Sharing x cuts both the small-n
# per-column overhead AND the huge-n x-restream (x exceeds L1 at n≈4096; per-column re-read it n times).
# Route gemv-T to the per-column dot instead of the NC-blocked kernel, inside a cache-geometry WINDOW,
# and only on boxes where that actually wins. PDM: the window is DERIVE, the decision to use it is
# MEASURE — because the sign is µarch-dependent and no formula over detected consts predicts it.
#
# WINDOW (Derive, no literal). Blocking amortises the x RE-READ across NC columns; it can only matter
# when x is not already free to re-read and the op is not compute-bound:
#   `m*n*sizeof(T) > _L2_BYTES`  — past L2 the NC independent accumulators stop being the binding
#                                  resource (below it the op is compute-bound and blocking's ILP wins).
#   `m*sizeof(T) <= _L1_BYTES÷2` — x (length m for the T form) still fits L1 beside the streaming A
#                                  column, so per-column re-reads are free. The ÷2 leaves room for the
#                                  A stream; once x approaches L1 the A traffic evicts it and
#                                  amortising x is the whole game again.
#
# DECISION (Measure, req#8b — a one-box derivation was caught here). Inside that window, per-column
# beats blocked NC=4 by up to 1.35× on wintermute (Zen4, 16 MB L3, mobile) and NEVER wins on galen
# (Zen3, 32 MB L3/CCX, desktop) — measured per-column ÷ blocked, GB/s:
#     n=      128   256   512   768  1024  1536  2048  3072  4096
#   Zen4     0.83  0.90  1.10  1.09  1.22  1.30  1.13  0.80  0.71   ⇒ window real
#   Zen3     0.81  0.86  0.92  0.90  0.97  0.84  0.74  0.74  0.80   ⇒ never
# Shipping the Derive-only version regressed Zen3 gemvT from 1.04/0.94 to **1.00/0.69** vs AOCL. Same
# class as `_ger_np` ("intrinsic per-core property, NO derivable formula, OPPOSITE sign across µarchs"),
# so it gets the same treatment: Preference-pinnable, else auto-measured once per process.
# MEASURE tier (PDM): this is NOT physically predictable, so it must be measured, not guessed.
# Per-column beats blocked NC=4 inside the window by up to 1.35× on wintermute (Zen4) and NEVER wins on
# galen (Zen3) — per-column ÷ blocked, GB/s:
#     n=      128   256   512   768  1024  1536  2048  3072  4096
#   Zen4     0.83  0.90  1.10  1.09  1.22  1.30  1.13  0.80  0.71
#   Zen3     0.81  0.86  0.92  0.90  0.97  0.84  0.74  0.74  0.80
# Opposite sign across µarchs with no formula over detected consts predicting it — the `_ger_np` shape
# ("intrinsic per-core property, NO derivable formula, OPPOSITE sign across µarchs"). A Derive-only
# version of this route regressed galen gemvT to 0.69 vs AOCL, and a `_double_pumped` default would be a
# 2-point fit dressed up as physics. So: auto-measured once per process, Preference only as the user's
# override AFTER measuring, never as the shipped mechanism.
#
# TWO CI CONSTRAINTS the probe must satisfy — both learned by turning the pipeline red (see
# kb/findings/pureblas-measure-tier-ci-constraints.md):
#   1. TRIM: the probe must be CLOSURE-FREE. Writing its arms as closures over a mutated `j` boxes the
#      capture, and the Box leaks into the trim call graph of every @ccallable reaching gemv-T
#      (dgbtrf/dgels/dgecon/dtrcon/dpocon: "unresolved call … Core.getfield(%new()::Box, :contents)").
#      The juliac pin does NOT protect you — CI validates trim WITHOUT the preference set.
#   2. @noalloc: `trmv!` carries a @strict_contract AllocCheck all-paths proof, and OncePerProcess's
#      one-time-init provably allocates its calibration buffer. The fix is NOT to reshape the probe —
#      it is to PIN the knob in the environments whose hardware is known, exactly as `ger_panel_np` is
#      pinned in `test/Project.toml` ("Freeze ger's panel stream-count so the StrictMode dogfood
#      exercises the SHIPPING path … the dogfood checks alloc-freeness, not perf") and in
#      `juliac/build.jl` for trim. Pinned there, the `@static if` compiles this branch out entirely and
#      the all-paths proof sees only the shipping path. Auto-measurement remains the mechanism for real
#      users on unknown hardware, which is the whole point of the Measure tier.
# ── gemv-T blocked-kernel column count (NC) ─────────────────────────────────────────────────────────
# Candidate set DERIVED from the register file: NC dot accumulators + the shared x vector + one A temp.
# AVX-512 (_NVREG=32) → {4,8,16}; AVX2 (16) → {4,8}. The SELECTION is Measure tier: whether more chains
# pay depends on 512-bit FMA throughput vs latency, which inverts between double-pumped and native-512
# silicon and is not derivable (the `_ger_np` shape). See the routing note in `_gemv_t_simd!` for why
# the existing Zen4 NC sweep does not settle this.
const _GEMVT_NC_CANDIDATES = Tuple(c for c in (4, 8, 16) if c + 2 <= _NVREG)::Tuple{Vararg{Int}}
# m-unroll for the blocked gemv-T kernel. DERIVED cap: NC·U accumulators + U x-vectors + ~2 temps must
# fit `_NVREG`. At the shipped NC=4 that gives U ≤ 4 on AVX-512 (24 regs) and U ≤ 2 on AVX2 (14).
@inline _gemvt_u_max(nc::Int) = (u = 4; while u > 1 && nc * u + u + 2 > _NVREG; u ÷= 2; end; u)
const _GEMVT_U_PREF = @load_preference("gemvt_u", nothing)
const _GEMVT_U = something(_GEMVT_U_PREF, 1)::Int   # req8-ok: shipped default until the gate says move it
# Runtime-resolved so the force hook can reach it (a const is baked at load and cannot be forced) —
# but resolved ONCE PER PROCESS, never per call.
# ⚠ `_force_knob` reads ENV, and ENV lookup in a BLAS-2 hot path is not free. The first cut called it on
# every gemv, and the gate caught it immediately: Zen5 gemvT n=64 fell 0.959 -> 0.767 and n=128
# 0.860 -> 0.813 with U unchanged at 1, i.e. the instrument itself was the regression, and it also
# compressed the very U differences the experiment existed to resolve. Small-n BLAS-2 amortizes a
# per-call dictionary lookup over very little work — the same class as the kwarg-overhead finding in
# kb/findings/pureblas-gemv.md, where ~200 ns of calling convention dominated a 33 ns kernel.
const _GEMVT_U_ONCE = Base.OncePerProcess{Int}() do
    f = _force_knob("gemvt_u")
    return f >= 1 ? min(f, _gemvt_u_max(4)) : _GEMVT_U
end
@inline _gemvt_u() = _GEMVT_U_ONCE()
const _GEMVT_NC_PREF = @load_preference("gemvt_nc", nothing)
@static if isnothing(_GEMVT_NC_PREF)
    function _measure_gemvt_nc()::Int
        Base.generating_output() && return 4                      # never burn a measure at precompile
        _f = _force_knob("gemvt_nc"); _f >= 0 && return _f         # instrument only, see _force_knob
        try
            T = Float64
            # PROBE IN THE BLOCKED REGIME THIS KNOB GOVERNS: square A at ~½L2, which is where the gate's
            # missing cells live (n=128/256 → A = 128 KB/512 KB, both ≤ L2) and where the loop is
            # compute-bound. Probing past L2 would measure a regime that routes per-column on Zen4 and is
            # memory-bound everywhere — the axpy_band lesson: a probe where the arms tie cannot resolve.
            n = _avoid_po2(max(32, isqrt(_L2_BYTES ÷ (2 * sizeof(T)))), _vwidth(T))
            nb = 3
            As = [fill(one(T), n, n) for _ in 1:nb]                # rotated: fresh draw per round
            x = fill(one(T), n); y = fill(zero(T), n)
            Ar = Ref(As[1]); rot(r) = (Ar[] = As[mod1(r, nb)]; nothing)
            run(c) = _gemv_t_sweep_nc!(n, n, one(T), Ar[], x, zero(T), y, c)
            inc() = run(4)
            _REP_BUDGET_NS = 20_000_000   # req8-ok: measurement-precision budget, selects no kernel
            nrep = clamp(Int(_REP_BUDGET_NS ÷ max(_tune_one(inc; reps = 3), UInt64(1))), 5, 40)
            # ⚠⚠ THE DUEL IS DISABLED, AND THE GATE IS WHY. Measured 2026-08-08 on wintermute,
            # freq-locked, all arms same-run, forcing each arm through the REAL entry path:
            #     n=      64     128    256     512     1024    2048    4096
            #   NC=4    1.024  1.026  0.978   1.162   1.207   1.043   1.102
            #   NC=8    1.140  1.031  0.995   0.989   0.989   1.013   0.998
            # NC=8 takes THREE passing cells to misses (512, 1024, 4096) and does not even close 256.
            # This probe — square A at ~½L2, rotated operands, per-round medians, time-budgeted reps —
            # nonetheless resolved 8 on this box. That is the THIRD knob today whose standalone probe
            # ranked the arms against the gate (see the axpy_band post-mortem and `_cgemvt_cfg`), and the
            # rule this project keeps re-learning is that a probe disagreeing with the gate is evidence
            # about the PROBE. So the selection is off until a probe exists that reproduces the table
            # above; the plumbing (candidate set, Val ladder, `PUREBLAS_FORCE_gemvt_nc`) stays, because
            # forcing an arm through `bench/plots.jl` is how that table was produced in the first place.
            # Do NOT re-enable this by "fixing" the probe's noise — fix its REGIME, then show it
            # reproduces the gate ordering on at least two boxes.
            # ⚠⚠ AND NC IS FALSIFIED AS THE LEVER FOR THE WORST CELLS — measured on ZEN5 (native 512,
            # where `_gemvt_perscan` is FALSE so EVERY size takes this blocked kernel), freq-locked,
            # all arms same-run, each NC forced through the real entry path:
            #     n=      64     128    256     512     1024    2048    4096
            #   NC=4    0.959  0.860  0.723   1.010   1.026   0.975   1.044
            #   NC=8    1.164  0.850  0.717   0.977   1.008   1.006   0.968
            #   NC=16   1.075  0.829  0.695   0.768   0.867   0.866   0.975
            # More chains make n=128 and n=256 SLIGHTLY WORSE, monotonically. The "native-512 Zen5 has
            # twice the 512-bit FMA throughput so 4 dependent chains starve it" model predicted the
            # opposite and is dead. NC=4 is right on every box and every size that matters.
            # ⚠⚠ AND THE M-UNROLL IS FALSIFIED TOO (2026-08-08, Zen5, forced through the real entry
            # path, ENV-overhead bug fixed first so U=1 reproduces the baseline):
            #     n=      64     128    256     512     1024    2048    4096
            #   U=1     0.951  0.856  0.728   0.996   1.014   0.974   1.040
            #   U=2     0.953  0.851  0.726   1.017   1.034   0.973   1.053
            #   U=4     1.042  0.853  0.721   1.009   0.993   0.983   1.044
            # Dead flat at the two cells that matter. BOTH structural deltas vs AOCL's dgemv-T — the
            # column count AND the m-unroll — are now measured and neither is the mechanism. The Zen5
            # n=128/256 deficit is NOT in this loop's shape.
            # What has NOT been tested: whether the blocked kernel should run AT ALL there. The
            # blocked-vs-per-column routing (`_gemvt_perscan`, this file) sends A <= L2 to BLOCKED
            # unconditionally, and that assumption was measured on Zen4. The per-column arm
            # (`_dot_simd` per column) is right there in the remainder loop and is not currently
            # reachable at A <= L2 on any box. THAT is the next experiment: make the residency arm of
            # the routing forceable and try per-column at n=128/256 on Zen5.
            return 4                                               # req8-ok: gate-measured incumbent
            # (unreachable while the duel is disabled; kept so the candidate arms stay compiled+tested)
            for c in (16, 8)                                       # widest-effect first
                c in _GEMVT_NC_CANDIDATES || continue
                _tune_wins_it(_tune_duel(inc, () -> run(c); reps = nrep, refresh = rot)) && return c  # req8-ok: candidate arm
            end
            return 4
        catch
            return 4
        end
    end
    const _GEMVT_NC_ONCE = Base.OncePerProcess{Int}(_measure_gemvt_nc)
    @inline _gemvt_nc() = _GEMVT_NC_ONCE()
else
    @inline _gemvt_nc() = _GEMVT_NC_PREF::Int
end

# gemv-T blocked-vs-per-column ROUTE, as a 3-valued MODE (not a Bool) so the RESIDENCY arm is itself
# forceable/shippable:
#   0 = blocked kernel at every size (the residency window never opens)
#   1 = per-column inside the derived window `A > L2 && x ≤ L1/2`, blocked outside   ← the old `true`
#   2 = per-column at every size (the residency guard is DISABLED)
# Mode 2 exists because the guard `A ≤ L2 ⇒ blocked` was itself only ever measured on Zen4, and Zen5's
# two worst gemvT cells (n=128/256) sit inside it — see the falsification block below.
# A Bool Preference (the historical spelling, still in test/Project.toml and juliac/build.jl) maps
# false→0, true→1, so pinned/trim builds keep their exact behaviour.
const _GEMVT_PERSCAN_PREF = (_p = @load_preference("gemvt_perscan", nothing);
                             _p isa Bool ? (_p ? 1 : 0) : _p)
@static if isnothing(_GEMVT_PERSCAN_PREF)
    function _measure_gemvt_perscan()::Int
        Base.generating_output() && return 0         # never burn a measure during precompile
        _f = _force_knob("gemvt_perscan"); _f >= 0 && return _f  # instrument only, see _force_knob
        try
            # Probe INSIDE the window, derived: A ≈ 4×L2 (clearly past L2), m capped so x ≤ L1/2.
            m = min(isqrt(4 * _L2_BYTES ÷ sizeof(Float64)), _L1_BYTES ÷ 2 ÷ sizeof(Float64))
            m -= m % 4; m < 64 && return 0
            n = m
            A = fill(1.0, m, n); x = fill(1.0, m); y = fill(0.0, n)
            GC.@preserve A x y begin
                pA = pointer(A); px = pointer(x); py = pointer(y); ld = stride(A, 2); sz = sizeof(Float64)
                # MEDIAN, not min. Kept straight-line (constraint 1: no closures here), so the samples go
                # into vectors rather than through `_tune_ab`. min-of-3 was the shipped estimator until
                # 2026-08-05; it is optimistic and tail-blind, and it selects which gemv-T path SHIPS.
                nrep = 5
                tbs = Vector{UInt64}(undef, nrep); tps = Vector{UInt64}(undef, nrep)
                for r in 0:nrep                              # r=0 untimed warmup; interleaved after
                    s = time_ns()
                    j = 0
                    while j + 4 <= n
                        _gemv_t_block!(py + j * sz, pA + j * ld * sz, ld, px, m, 1.0, 0.0, Val(4), Val(true), Val(_GEMVT_U))
                        j += 4
                    end
                    e = time_ns() - s; r > 0 && (tbs[r] = e)
                    s = time_ns()
                    @inbounds for jj in 0:(n - 1)
                        unsafe_store!(py, _dot_simd(m, pA + jj * ld * sz, px, Float64), jj + 1)
                    end
                    e = time_ns() - s; r > 0 && (tps[r] = e)
                end
                sort!(tbs); sort!(tps)
                mid = (nrep + 1) ÷ 2
                # This duel only ever probes INSIDE the window, so it can resolve 0 vs 1 and nothing
                # else. Mode 2 is reachable by force/Pin only, deliberately: nothing ships it yet.
                return tps[mid] < tbs[mid] ? 1 : 0
            end
        catch
            return 0
        end
    end
    const _GEMVT_PERSCAN_ONCE = Base.OncePerProcess{Int}(_measure_gemvt_perscan)
    @inline _gemvt_perscan_mode() = _GEMVT_PERSCAN_ONCE()
else
    @inline _gemvt_perscan_mode() = _GEMVT_PERSCAN_PREF::Int
end
@inline function _gemvt_perscan(m::Int, n::Int, ::Type{T}) where {T}
    mode = _gemvt_perscan_mode()                     # ONE resolution per call, as before
    mode >= 2 && return true                         # per-column everywhere (instrument arm)
    return mode == 1 && m * n * sizeof(T) > _L2_BYTES && m * sizeof(T) <= _L1_BYTES ÷ 2
end

@inline _gemv_t_simd!(m::Int, n::Int, α::T, A, x, β::T, y, b0::Val{B0}) where {T <: BlasReal, B0} =
    _gemv_t_simd!(m, n, α, A, x, β, y, b0, !_gemvt_perscan(m, n, T))

# `blk`-explicit entry: `true` forces the NC=4 blocked kernel regardless of `_gemvt_perscan`.
# Exists for callers whose regime the perscan probe does NOT represent — the probe measures a CLEAN
# standalone sweep, but `_laqps!`'s F-build re-sweeps the SAME trailing block once per panel column;
# in that repeated-sweep regime blocked won on Zen4 at EVERY size measured (in-context replay
# 38 vs 30 GB/s at the n=2048 trailing shape; live geqp3 1.03-1.34x at n=256..2048) even though the
# clean standalone probe ranks per-column ahead there (probe-regime-must-match-live).
# Knob-free blocked gemv-T sweep at an explicit NC — what `_measure_gemvt_nc` duels. Kept separate from
# `_gemv_t_simd!` so the harness exercises the SAME block kernel the shipped path uses, without the knob
# lookup inside the timed region.
function _gemv_t_sweep_nc!(m::Int, n::Int, α::T, A, x, β::T, y, nc::Int) where {T <: BlasReal}
    GC.@preserve A x y begin
        Aptr = pointer(A); xptr = pointer(x); yptr = pointer(y); lda = stride(A, 2); sz = sizeof(T)
        j = 0
        while j + nc <= n
            if nc >= 16
                _gemv_t_block!(yptr + j * sz, Aptr + j * lda * sz, lda, xptr, m, α, β, Val(16), Val(true), Val(_GEMVT_U))  # req8-ok: candidate arm
            elseif nc >= 8
                _gemv_t_block!(yptr + j * sz, Aptr + j * lda * sz, lda, xptr, m, α, β, Val(8), Val(true), Val(_GEMVT_U))  # req8-ok: candidate arm
            else
                _gemv_t_block!(yptr + j * sz, Aptr + j * lda * sz, lda, xptr, m, α, β, Val(4), Val(true), Val(_GEMVT_U))  # req8-ok: candidate arm
            end
            j += nc
        end
        @inbounds while j < n
            s = _dot_simd(m, Aptr + j * lda * sz, xptr, T)
            unsafe_store!(yptr, α * s, j + 1)
            j += 1
        end
    end
    return y
end

@inline function _gemv_t_simd!(
        m::Int, n::Int, α::T, A, x, β::T, y, ::Val{B0}, blk::Bool
    ) where {T <: BlasReal, B0}
    GC.@preserve A x y begin
        Aptr = pointer(A); xptr = pointer(x); yptr = pointer(y); lda = stride(A, 2); sz = sizeof(T)
        # Column blocking exists to amortise the x RE-READ across NC columns. It only pays in two
        # regimes, and LOSES between them — route on the physics rather than blocking unconditionally.
        #   • A ≤ L2: compute-bound, and the NC independent dot accumulators supply the ILP. Blocked.
        #   • x no longer L1-resident against the A stream: per-column would re-stream x from L2/DRAM
        #     n times (x traffic ≈ A traffic at n=4096!), so amortising it is the whole game. Blocked.
        #   • in between — A past L2 but x still comfortably L1-hot — re-reading x is free, and the
        #     blocked kernel's extra concurrent A-column streams only cost prefetch/TLB. Per-column.
        # Measured (wintermute, freq-locked, GB/s; per-column ÷ blocked NC=4):
        #     n=      128    256    512    768   1024   1536   2048   3072   4096
        #   ratio    0.83   0.90   1.10   1.09   1.22   1.30   1.13   0.80   0.71
        # `_gemvt_perscan` below reproduces every one of those crossovers on this box: blocked at
        # 128/256 (A ≤ L2) and at 3072/4096 (x > L1/2), per-column at 512…2048. DERIVE tier — pure
        # cache geometry over detected consts, no new knob. The per-column arm reuses this function's
        # OWN remainder path (`_dot_simd` per column), so it is a routing change, not a new kernel:
        # isolating it showed per-column `_dot_simd` already recovers 1.08-1.21x of the 1.09-1.35x, i.e.
        # the deficit is the BLOCKING, not the kernel.
        # NC=4 columns (⇒ 4 dot accumulators) per block. MEASURED-AND-KEPT, not an unexamined literal:
        # the standing hypothesis was that 4 chains half-fill Zen4's 2 FMA/cyc × ~4-cyc-latency pipe and
        # that ≥8 would close the mid-n gemvT miss. FALSIFIED 2026-07-31 (wintermute, freq-locked, GF/s,
        # same-process A/B over the @generated Val{NC}):
        #     n=  256   512  1024  2048  4096
        #  NC=2  20.59 15.88 14.27  7.28  7.13
        #  NC=4  20.71 16.61 14.04  8.43  8.06   ← ships
        #  NC=8  21.07 15.66 14.84  8.36  6.89
        #  NC=16 18.92 11.58 10.87  7.23  7.24
        # NC=8 wins only at 256 (+1.7%) and 1024 (+5.7%) and LOSES at 512 (−5.7%), 2048 and 4096 (−14.5%);
        # NC=16 is uniformly worse. So this is not ILP-starved, and a size-switched NC would buy the
        # n=1024 cell (0.948 vs AOCL) by giving back others — a new tuning knob for a net wash. Left at 4.
        # NOTE this also retires the `_gemvt_nc` Measure-tier constant referenced in comments elsewhere
        # (e.g. banded_chol.jl): it was never implemented, and the measurement above says it should not be.
        # ⚠ THE SWEEP ABOVE IS ZEN4-ONLY, AND ON ZEN4 THREE OF ITS FIVE COLUMNS ARE NOT LIVE. Re-read it
        # against the routing: `_gemvt_perscan` is TRUE on Zen4, so n=512…2048 take the PER-COLUMN arm
        # there — NC never applies. The cells where NC=8 actually loses on Zen4 are exactly the ones the
        # blocked kernel does not run. Where blocked IS live on Zen4 (n=256, A ≤ L2) NC=8 WINS (+1.7%),
        # and n=256 is Zen4's only gemvT miss (0.978).
        # And perscan is FALSE on Zen5 and Zen3, so those boxes take blocked at EVERY size — a regime the
        # sweep never covered. Zen5 is where this hurts: 0.863 at n=128 and 0.747 at n=256, the worst
        # gemv cell on the fleet, both A ≤ L2 and both on this NC=4 kernel that Zen4 runs at 0.978.
        # AOCL (disassembled 2026-08-08, `bli_dgemv_t_zen_int_avx512`): NC=8 accumulators, m unrolled ×4
        # (32 rows/iteration, 1 x-load per 8 FMAs), no prefetch and no non-temporal stores anywhere in
        # its 2.6 KB. Our NC=4 with no m-unroll is the whole structural delta. Native-512 Zen5 has twice
        # Zen4's 512-bit FMA throughput, so 4 dependent chains cover Zen4's pipe and starve Zen5's —
        # consistent with the collapse being confined to the compute-bound A ≤ L2 regime and recovering
        # to 0.99–1.06 past L2 where memory hides it.
        # So NC is Measure tier (it inverts across µarchs — the `_ger_np` shape), with the candidate set
        # DERIVED from the register file. This does NOT re-open the sweep's finding: that measured Zen4
        # cells which route per-column, on one box.
        j = 0
        # NC is FIXED AT 4 — falsified as a lever on every box (tables in `_measure_gemvt_nc`). The live
        # parameter is the m-unroll U, resolved at runtime so `PUREBLAS_FORCE_gemvt_u` can force it
        # through this real entry path; a compile-time const cannot be forced, which is how the first
        # attempt at this experiment silently measured U=1 three times.
        u = _gemvt_u()
        while blk && j + 4 <= n
            p = yptr + j * sz; q = Aptr + j * lda * sz
            if u >= 4
                _gemv_t_block!(p, q, lda, xptr, m, α, β, Val(4), Val(B0), Val(4))  # req8-ok: candidate arm
            elseif u >= 2
                _gemv_t_block!(p, q, lda, xptr, m, α, β, Val(4), Val(B0), Val(2))  # req8-ok: candidate arm
            else
                _gemv_t_block!(p, q, lda, xptr, m, α, β, Val(4), Val(B0), Val(1))  # req8-ok: candidate arm
            end
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
# Complex gemv-T/C column-block width (cols/pass). NC=4 both ISAs: AVX2 via half-width Vec{W} accs (see
# _CGEMVT_HALF below), AVX-512 via full-width Vec{2W}. Sharing xc + its swap across the block is the win
# (1 shuffle feeds NC cols, x streamed once per block).
const _CGEMVT_NC = @load_preference("cgemvt_nc", 4)::Int   # legacy pin; superseded by _cgemvt_cfg()
# AVX2: accumulate gemvT/C in native ymm (Vec{W}) so NC=4 columns fit → 4 concurrent load streams (see
# _gemv_tc_block_cmplx!). AVX-512 keeps full-width Vec{2W} (32 zmm has room, already gates).
# ⚠ MOVED UP from below `_cgemvt_cfg`: the config knob's default is `NC + (HALF ? 100 : 0)`, so both
# consts must exist before it. Julia consts are load-ordered — the earlier arrangement referenced
# `_CGEMVT_HALF` ~45 lines before its definition.
const _CGEMVT_HALF = @load_preference("cgemvt_half", _vwidth(Float64) == 4)::Bool
# Once A spills L2 (n≳768), gemvT/C is bandwidth-bound, not FMA-latency-bound (measured galen: n≥1024
# both PB & OB run at L3/DRAM bandwidth, PB only ~92-94% of OB's). Same +192B A-stream prefetch that
# fixed the gemvN ri valley saturates it here. AVX2-gated (AVX-512 gemvT already gates); Preferences knob.
const _CGEMVT_PF = @load_preference("cgemvt_pf", _vwidth(Float64) == 4)::Bool
# ── gemv-T/C column-block CONFIG: (NC, HALF) as ONE Measure knob ────────────────────────────────────
# Encoding: `NC + (HALF ? 100 : 0)`. One knob because the two are not independent — HALF halves the
# registers per column, which is exactly what buys a larger NC, so sweeping them separately would test
# combinations the register file cannot hold.
#
# WHY THIS EXISTS. `_CGEMVT_NC = 4` was a bare literal whose comment said "swept per box" — a req#8b
# violation with no Derive formula and no Measure harness (real gemv-T has `_gemvt_nc`; the complex path
# never got one). It is also measurably wrong: disassembling AOCL 2026-08-07 shows `zgemv` trans routes
# to `bli_zdotxf_zen_int_8_avx512` — a fused dot with FUSING FACTOR 8, i.e. eight concurrent column read
# streams against our four — and the miss is broad rather than a single cell (Zen4 0.963/0.895/0.992 and
# Zen3 0.980/0.945/0.945 at n=512/1024/2048 vs AOCL), which is the signature of a stream-count deficit
# rather than one bad size. AOCL uses no prefetch and no non-temporal stores anywhere in that path
# (verified: 64 vfmadd231pd, 28 vpermilpd, zero prefetch*/movnt*), so stream count is what is left.
#
# CANDIDATES ARE DERIVED FROM THE REGISTER FILE, not enumerated by hand. Each column holds p and q
# accumulators of `Vec{lanes}` with lanes = HALF ? W : 2W, and a `Vec{lanes}` is `lanes*sizeof(T) /
# _SIMD_BYTES` native registers — 1 when HALF, 2 when not. Plus xc and xcs at the same width, plus a
# few for A-loads and addresses:
#     regs(NC, HALF) = (2*NC + 2) * (HALF ? 1 : 2) + reserve
# AVX-512 (_NVREG=32): (4,wide)=20+r, (8,half)=18+r, (4,half)=10+r  → all three fit.
# AVX2    (_NVREG=16): (4,half)=10+r, (2,wide)=12+r fit; (8,half)=18+r does NOT.
# Tier: MEASURE — stream count is the `_ger_np` class (port/prefetcher-dependent, sign-inverts across
# µarchs), so it cannot be Derived; only the BOUNDS are derived, which is what req#8b asks for.
@inline _cgemvt_regs(nc::Int, half::Bool) = (2 * nc + 2) * (half ? 1 : 2) + 6
_cgemvt_fits(nc::Int, half::Bool) = _cgemvt_regs(nc, half) <= _NVREG
# Incumbent for the L2-RESIDENT regime = today's shipped behaviour, so a tie keeps what ships.
const _CGEMVT_CFG_DEFAULT = _CGEMVT_NC + (_CGEMVT_HALF ? 100 : 0)
# Incumbent for the PAST-L2 regime = DERIVED: the most concurrent column streams the register file can
# hold in the narrow layout. Physical criterion — past L2 the loop is bandwidth/MLP-bound, so streams
# are the resource, and the narrow layout is what makes them affordable. AVX-512 → 8 (matching AOCL's
# fused zdotxf factor of 8, arrived at independently from the register budget), AVX2 → 4.
#
# ⚠ WHY THE INCUMBENT MOVED HERE INSTEAD OF LEAVING IT TO THE DUEL. Measured 2026-08-07: with (4,wide)
# as incumbent the duel resolved 108/4/108/4/108/4/108/4 across eight fresh processes — a systematic
# alternation, not noise — while the GATE says (8,half) is decisively right past L2 on this box
# (n=512 0.955→1.006, n=1024 0.912→1.017 when forced). Shipping a coin flip between a passing and a
# missing kernel is the exact failure the duel rule exists to prevent, so the derived value becomes the
# incumbent and ties go to it. The knob REMAINS Measure tier: a box whose optimum really is fewer
# streams (Zen5 resolves `_ger_np()` = 1, so this is not hypothetical) can still displace it, but must
# now clear the supermajority AND the regret bound to do so.
const _CGEMVT_CFG_BIG = (
    n = _cgemvt_fits(8, true) ? 8 : _cgemvt_fits(4, true) ? 4 : 2;
    n + 100
)
const _CGEMVT_CFG_PREF = @load_preference("cgemvt_cfg", nothing)
@static if isnothing(_CGEMVT_CFG_PREF)
    function _measure_cgemvt_cfg()::Int
        Base.generating_output() && return _CGEMVT_CFG_BIG     # never burn a measure at precompile
        _f = _force_knob("cgemvt_cfg"); _f >= 0 && return _f       # instrument only, see _force_knob
        try
            T = Float64; W = _vwidth(T)
            # PROBE WHERE THE KNOB IS DISPATCHED AND WHERE THE ARMS SEPARATE. Square A sized so the
            # matrix lands at L3 — that is the worst measured cell on both locked boxes and it sits
            # inside the broad 512..2048 miss band, so a winner here is a winner across the band. This
            # is the axpy_band lesson: a probe in a regime where the arms tie cannot resolve the knob.
            n = _avoid_po2(max(64, isqrt(_L3_BYTES ÷ (2 * sizeof(Complex{T})))), W)
            nb = 3
            As = [fill(Complex{T}(1.0e-3, 2.0e-3), n, n) for _ in 1:nb]   # rotated: fresh draw per round
            x = fill(Complex{T}(0.5, -0.25), n)
            y = fill(Complex{T}(0.0, 0.0), n)
            Ar = Ref(As[1])
            rot(r) = (Ar[] = As[mod1(r, nb)]; nothing)
            α = Complex{T}(1.0, 0.0); β = Complex{T}(0.0, 0.0)
            run(nc, half) = _gemv_tc_run!(n, n, α, Ar[], x, β, y, Val(false), nc, half)
            inc() = run(Val(_CGEMVT_CFG_BIG - 100), Val(true))   # derived incumbent
            # req8-ok: a MEASUREMENT-PRECISION budget in nanoseconds, not a machine value — selects no
            # kernel, appears in no shipped path. Same fix `_measure_axpy_unroll` needed: a fixed 5-rep
            # median rests on too little signal at this size and the duel under-resolves an effect the
            # GATE measures at 11% (n=1024: 0.912 → 1.017 when the arm is forced). Deriving `nrep` from
            # a time budget makes every box spend the same TIME rather than the same rep count.
            _REP_BUDGET_NS = 20_000_000
            nrep = clamp(Int(_REP_BUDGET_NS ÷ max(_tune_one(inc; reps = 3), UInt64(1))), 5, 40)
            # Ordered widest-effect first; first to earn a supermajority wins. Guards const-fold, so a
            # candidate the register file cannot hold is not even compiled on that ISA.
            # Candidates are the alternatives to the DERIVED incumbent (fewer/wider streams). A box whose
            # optimum really is fewer streams displaces it; a tie keeps the derived value.
            if _CGEMVT_CFG_BIG != 104 && _cgemvt_fits(4, true)
                _tune_wins_it(_tune_duel(inc, () -> run(Val(4), Val(true)); reps = nrep, refresh = rot)) && return 104  # req8-ok: candidate arm
            end
            if _CGEMVT_CFG_BIG != 4 && _cgemvt_fits(4, false)
                _tune_wins_it(_tune_duel(inc, () -> run(Val(4), Val(false)); reps = nrep, refresh = rot)) && return 4  # req8-ok: candidate arm
            end
            if _CGEMVT_CFG_BIG != 2 && _cgemvt_fits(2, false)
                _tune_wins_it(_tune_duel(inc, () -> run(Val(2), Val(false)); reps = nrep, refresh = rot)) && return 2  # req8-ok: candidate arm
            end
            return _CGEMVT_CFG_BIG
        catch
            return _CGEMVT_CFG_BIG
        end
    end
    const _CGEMVT_CFG_ONCE = Base.OncePerProcess{Int}(_measure_cgemvt_cfg)
    @inline _cgemvt_cfg() = _CGEMVT_CFG_ONCE()
else
    @inline _cgemvt_cfg() = _CGEMVT_CFG_PREF::Int
end
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
@generated function _gemv_n_ri_panel!(
        yp::Ptr{T}, Ab::Ptr{T}, ldc::Int, xp::Ptr{T}, jc::Int, m::Int,
        αr::T, αi::T, ::Val{NC}, ::Val{PF}
    ) where {T, NC, PF}
    W = _vwidth(T); V2 = Vec{2W, T}; sz = sizeof(T)
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(2W - 1))...)
    # Sign-fold: pre-multiply the ci broadcast by [+1,-1,+1,-1,…] (hoisted, once/column) so the per-row
    # epilogue becomes a plain FADD `(y+Pv)+shuffle(Qv)` instead of `muladd(shuffle(Qv), sgnv, y+Pv)` —
    # drops one FMA-port op per row-iter (the inner loop is FMA-bound). altv[swp[k]]·shuffle folds the
    # sgnv=[-1,+1,…] pattern into Qv; scalar-tail uses lane[1] (=+ci, unaffected). Fable-P2 2026-07-14.
    altv = Expr(:tuple, (iseven(l) ? one(T) : -one(T) for l in 0:(2W - 1))...)
    body = quote
        altv = $V2($altv)
    end
    for c in 1:NC                                   # hoist: α·x[jj] once per column, broadcast re/im
        push!(body.args, :($(Symbol(:b, c)) = Ab + (jc + $(c - 1)) * 2 * ldc * $sz))
        push!(body.args, :(xr = unsafe_load(xp, 2 * (jc + $(c - 1)) + 1); xi = unsafe_load(xp, 2 * (jc + $(c - 1)) + 2)))
        push!(body.args, :($(Symbol(:cr, c)) = $V2(αr * xr - αi * xi)))
        push!(body.args, :($(Symbol(:ci, c)) = $V2(αr * xi + αi * xr) * altv))   # sign-folded ci broadcast
    end
    inner = quote
        off = i * 2 * $sz
    end
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
    push!(inner.args, :(yv = (yv + Pv) + shufflevector(Qv, Val($swp))))   # sign pre-folded into Qv
    push!(inner.args, :(vstore(yv, yp + off)))
    tail = quote
        acr = zero($T); aci = zero($T)
    end                  # scalar row tail (m % W complex rows)
    for c in 1:NC
        push!(
            tail.args, quote
                arr = unsafe_load($(Symbol(:b, c)), 2i + 1); aii = unsafe_load($(Symbol(:b, c)), 2i + 2)
                acr += arr * $(Symbol(:cr, c))[1] - aii * $(Symbol(:ci, c))[1]
                aci += arr * $(Symbol(:ci, c))[1] + aii * $(Symbol(:cr, c))[1]
            end
        )
    end
    push!(
        tail.args, quote
            unsafe_store!(yp, unsafe_load(yp, 2i + 1) + acr, 2i + 1)
            unsafe_store!(yp, unsafe_load(yp, 2i + 2) + aci, 2i + 2)
        end
    )
    push!(body.args, :(i = 0))
    push!(
        body.args, :(
            while i + $W <= m
                $inner; i += $W
            end
        )
    )
    push!(
        body.args, :(
            while i < m
                $tail; i += 1
            end
        )
    )
    push!(body.args, :(return nothing))
    return body
end
# NC IS A MEMORY-LEVEL-PARALLELISM KNOB, and it was a bare literal (`4`, justified by "OB uses 4" —
# OpenBLAS's choice for OpenBLAS's kernel is not a derivation for ours). NC is how many A-column streams
# run concurrently: while A is cache-resident there is no fill latency to hide and a wide panel only
# costs register pressure and y-restreaming, but once A outgrows the caches more concurrent streams keep
# more misses outstanding. So the optimum RISES with A's residency, and one constant is wrong at one end.
#
# Measured 2026-08-06, both boxes freq-locked, in plots.jl's own `_L2REP` regime, 3 rotated rounds with
# the anchor arm duplicated as the run's noise floor (0.05-1.4%). Ratios vs the shipping NC=4:
#   Zen4 (W=8, L2=1 MB, L3=16 MB)          Zen3 (W=4, L2=512 KB)
#    n     A     A/L2   4     8    12   16      n     A    A/L2   2     4     6     8
#   256  1.0MB    1   1.000 0.924 0.743 0.576  256  1.0MB   2  0.852 1.000 0.995 0.946
#   512  4.2MB    4   0.999 1.076 0.972 0.845  512  4.2MB   8  0.870 0.993 1.029 0.878
#  1024 16.8MB   16   0.980 1.090 1.132 1.080 1024 16.8MB  32  0.827 0.980 1.032 0.880
#  2048 67.1MB   64   0.996 1.034 1.058 0.930 2048 67.1MB 128  0.787 0.988 1.036 0.930
#
# WHY MEASURE AND NOT DERIVE. The tier SHAPE is shared — NC rises with A/L2 and saturates at 3W/2, and
# 2W is worse than the shipping value on both boxes, which is what bounds the candidate set above. But
# the values are NOT: at A ≈ L2 Zen4 wants W/2 while the same rule gives Zen3 W/2 = 2, which measures
# 0.852 there — a 15% regression. A residency formula fitted to Zen4 mispredicts a box we own, which is
# precisely req#8b's tell for Measure tier. Only the LARGE-A tier is common ground: both boxes put the
# optimum at 3W/2 once A > L3/2, and both put it above the shipping value.
#
# SO ONLY THAT TIER MOVES. Below L3/2 the shipping `_CGEMVN_NC` is kept byte-for-byte — no size on
# either box regresses (Zen4 n=256/512 measure 1.000/0.999 at NC=4). Above it, the value is measured
# on-host over the DERIVED candidate set {W, 3W/2}. The boundary (L3 residency) is Derive tier; the
# value inside it is Measure tier. The A ≤ L3/2 tiers are left as a known-unclaimed +7.6% on Zen4 at
# n=512 rather than shipping a rule that regresses Zen3 — widening the harness to three measured tiers
# is the follow-up, and it needs its own fleet validation.
@inline function _cgemvn_abytes(::Type{T}, m::Int, n::Int) where {T}
    return 2 * m * n * sizeof(T)                    # A is complex: two reals per element
end

# Complex gemvN driver with NC fixed at COMPILE time: β-prescale y, then NC-column panels over full m
# (no m-blocking — full-m column streams prefetch best; tall y-beyond-L2 shapes stay on the row-tile
# path via the caller). NC is a `Val` rather than an Int so the @generated panel specializes per
# candidate; a runtime `Val(nc)` would dispatch dynamically on every panel, and the tuner below needs to
# call this at each candidate anyway.
function _gemv_n_ri_run!(
        m::Int, n::Int, α::Complex{T}, A, x, y, β::Complex{T}, ::Val{B0}, ::Val{NC}, ::Val{PF}
    ) where {T <: BlasReal, B0, NC, PF}
    sz = sizeof(T); αr = real(α); αi = imag(α)
    # y is restreamed once per NC-column panel (n/NC times) → block m so the y-block fits ~½ L2 for tall
    # shapes; square mid-n (16m ≤ ½L2) runs one block (NB=m), which measured fastest (prefetch continuity).
    NB = (2 * m * sz <= _L2_BYTES ÷ 2) ? m : max(NC, (_L2_BYTES ÷ 2) ÷ (2 * sz))
    GC.@preserve A x y begin
        Ap = Ptr{T}(pointer(A)); yp = Ptr{T}(pointer(y)); xp = Ptr{T}(pointer(x)); ldc = stride(A, 2)
        if B0
            @inbounds for i in 1:(2m)
                unsafe_store!(yp, zero(T), i)
            end
        elseif !isone(β)
            _scal_cmplx_simd!(m, real(β), imag(β), y)
        end
        i0 = 0
        while i0 < m
            mb = min(NB, m - i0); ypb = yp + i0 * 2 * sz; Apb = Ap + i0 * 2 * sz
            jc = 0
            while jc + NC <= n
                _gemv_n_ri_panel!(ypb, Apb, ldc, xp, jc, mb, αr, αi, Val(NC), Val(PF))
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

# NC FOR THE LARGE-A TIER: **DERIVE**, not Measure. This was a Measure-tier OncePerProcess until
# 2026-08-07, and the demotion is the point — the knob never earned that tier, it inherited it by
# analogy to `_ger_np` ("same class"), and its own data says otherwise.
#
# THE LITMUS TEST for Derive-vs-Measure: does the measured optimum SCALE WITH A DETECTED CONST across
# boxes, or does it move while every detected const stays fixed? Scaling means the binding resource is
# core structure, which the ISA/cache consts describe. Moving at fixed consts means the resource is
# outside the detected set (memory latency, prefetcher depth, store-buffer drain) and no formula over
# that set can reproduce it without a family branch — a per-µarch literal in disguise, which req#8b
# bans. Measured winners for this knob, in the regime it governs (A > L3/2), square operands:
#     Zen4 (W=8):  12   = 3W/2      [n=1024 1.132 and n=2048 1.058 vs shipping; gate 0.849 -> 0.989]
#     Zen3 (W=4):   6   = 3W/2      [n=1024 1.032, n=2048 1.036 vs shipping]
# The SAME formula on both, and the value tracks W — the signature of a core-structure bound. Note it
# is specifically NOT the memory-level-parallelism story I first assumed: MAB/LFB counts do not double
# when AVX-512 is enabled, so an MLP-bound optimum would not track W at all.
#
# WHY 3W/2 AND NOT 2W: 2W was measured WORSE than the shipping value on both boxes (Zen4 0.930 at
# n=2048, Zen3 0.930 at n=2048) — each column holds a cr/ci broadcast pair alongside the Pv/Qv
# accumulators, so past 3W/2 the panel spills. That is the register-file bound, and it is why the
# candidate set was derived from W in the first place.
#
# WHAT THIS DELETES, all of it pure benefit: the OncePerProcess (so no first-call benchmark inside a
# library, no @noalloc hazard on the paths that reach this, no pin required for the trim build), the
# duel, and the nondeterminism that had it resolving 8/12/8/8 across fresh processes before the duel
# and 12 after. A derived const is deterministic by construction and const-folds.
#
# ⚠ ZEN5 IS PREDICTED, NOT MEASURED (W=8 ⇒ 12). req#8b(b) requires a derived formula to reproduce the
# fleet's measured optima before it is trusted to extrapolate, and neuromancer has not been run for
# this knob. Acceptance test, to run freq-locked with plots.jl methodology and `arms=pb`: zgemvN square
# at A≈L3 and A≈4×L3, Val(8) vs Val(12) arms of `_gemv_n_ri_run!`, median estimator. If Zen5 prefers 8,
# that is the req#8b tell and this goes back to Measure — the Preference below is the escape hatch
# meanwhile.
const _CGEMVN_NC_BIG = @load_preference("cgemvn_nc_big", 3 * _vwidth(Float64) ÷ 2)::Int
@inline _cgemvn_nc_big() = _CGEMVN_NC_BIG

# The shipping panel, with NO reference to the measured knob. Internal BLAS-2 callers use THIS, not the
# tier-selecting entry below — see the @noalloc note there.
@inline function _gemv_n_ri_ship!(m::Int, n::Int, α::Complex{T}, A, x, y, β::Complex{T}, ::Val{B0}) where {T <: BlasReal, B0}
    return _CGEMVN_PF ?
        _gemv_n_ri_run!(m, n, α, A, x, y, β, Val(B0), Val(_CGEMVN_NC), Val(true)) :
        _gemv_n_ri_run!(m, n, α, A, x, y, β, Val(B0), Val(_CGEMVN_NC), Val(false))
end

# ⚠ THE MEASURED KNOB LIVES HERE AND NOWHERE UPSTREAM OF A @noalloc CONTRACT. `Base.OncePerProcess`'s
# first-call init allocates (`jl_set_precompile_field_replace`), and `@assert_noalloc` is an ALL-PATHS
# static proof — it does not care that a branch is dynamically unreachable. `trmv!`/`trsv!` carry that
# contract and reach the complex gemvN kernel through `_tri_scat_cmplx!`, so routing them through this
# function put 72 allocation sites into their proof and failed StrictMode. Same class as the axpy
# OncePerProcess that had to leave the BLAS-2 path.
# It is also the right split on the merits, not just to satisfy the checker: the triangular scatter's
# operand is m×NB — a tall-skinny panel — never the large SQUARE regime this knob was measured in, so
# consulting it there would be a probe-regime error even if it were free.
function _gemv_n_ri_cmplx!(m::Int, n::Int, α::Complex{T}, A, x, y, β::Complex{T}, ::Val{B0}) where {T <: BlasReal, B0}
    # Below L3/2 keep the shipping panel byte-for-byte (no size on either box regresses there); above it,
    # take the measured value. The BOUNDARY is Derive tier (L3 residency), the VALUE inside it is Measure.
    if _cgemvn_abytes(T, m, n) > _L3_BYTES ÷ 2
        nc = _cgemvn_nc_big()
        # `nc` is one of three compile-time-known values, so branch to a specialization rather than
        # building `Val(nc)` — and note the branch must be exhaustive over what the tuner can return,
        # since falling through silently applies `_CGEMVN_NC` instead (correct, just not what was tuned).
        if nc == 3 * _vwidth(T) ÷ 2
            return _gemv_n_ri_run!(m, n, α, A, x, y, β, Val(B0), Val(3 * _vwidth(T) ÷ 2), Val(false))
        elseif nc == _vwidth(T)
            return _gemv_n_ri_run!(m, n, α, A, x, y, β, Val(B0), Val(_vwidth(T)), Val(false))
        end
    end
    return _gemv_n_ri_ship!(m, n, α, A, x, y, β, Val(B0))
end

# Complex gemv-N panel block: accumulate columns [jc, jc+Peff) of A into MR row-tiles of W complex, RMW

# Complex gemv-N panel block: accumulate columns [jc, jc+Peff) of A into MR row-tiles of W complex, RMW
# into y (y pre-scaled by β by the driver). Interleaved Vec{2W} accumulators; per column cⱼ=α·x[j],
# c += A·cr + swap(A)·[−ci,ci] (swap-pairs complex multiply). Panel loop reads each A-column sequentially.
# NOTE (AVX2): Vec{2W} legalizes to 2 regs (MR small); still ~0.5–0.7 on AVX2 — gemvN there is
# shuffle/throughput-bound, an AVX2 TUNING residual (the fma/muladd primitives suffice; a split-Vec{W}
# variant measured worse). Gates AVX-512 (Vec{2W}=1 reg). See ROADMAP M5.
@generated function _gemv_n_block_cmplx!(
        yb::Ptr{T}, Ab::Ptr{T}, ldc::Int, xp::Ptr{T}, jc::Int, Peff::Int,
        αr::T, αi::T, ::Val{MR}
    ) where {T, MR}
    W = _vwidth(T); V2 = Vec{2W, T}; sz = sizeof(T)
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(2W - 1))...)
    body = quote end
    for v in 1:MR
        push!(body.args, :($(Symbol(:c, v)) = vload($V2, yb + $((v - 1) * 2W * sz))))
    end
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
    push!(
        body.args, :(
            for cc in 0:(Peff - 1)
                $inner
            end
        )
    )
    for v in 1:MR
        push!(body.args, :(vstore($(Symbol(:c, v)), yb + $((v - 1) * 2W * sz))))
    end
    push!(body.args, :(return nothing))
    return body
end

# Driver: pre-scale y by β once, then column-panels × row-tiles accumulate (RMW y). W-remainder blocks
# (Val(1)) + scalar tail per panel handle m not a multiple of MR·W.
function _gemv_n_cmplx!(
        m::Int, n::Int, α::Complex{T}, A, x, y, β::Complex{T}, ::Val{B0},
        ::Val{MR} = Val(_CGEMV_MR)
    ) where {T <: BlasReal, B0, MR}
    W = _vwidth(T); mr = MR * W; sz = sizeof(T); αr = real(α); αi = imag(α)
    GC.@preserve A x y begin
        Ap = Ptr{T}(pointer(A)); yp = Ptr{T}(pointer(y)); xp = Ptr{T}(pointer(x)); ldc = stride(A, 2)
        if B0
            @inbounds for i in 1:(2m)
                unsafe_store!(yp, zero(T), i)
            end       # y := 0
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
                for cc in 0:(Peff - 1)
                    s += A[i, jc + cc + 1] * x[jc + cc + 1]
                end
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
# HALF: accumulate in the native-ymm Vec{W} (1 reg) rather than Vec{2W} (2 regs). Large-n gemvT/C is
# bandwidth/MLP-bound (measured galen: n≥1024 both PB & OB run at L3/DRAM bw); Vec{2W} at NC=2 already
# eats all 16 ymm, capping concurrent column streams at 2. Vec{W} at NC=4 → 4 independent streams (OB's
# AVX2 blocking) → more memory-level parallelism → saturates bw. Full-width kept for AVX-512 (32 regs).
@inline @generated function _gemv_tc_block_cmplx!(
        yp::Ptr{Complex{T}}, Ab::Ptr{Complex{T}}, lda::Int,
        xp::Ptr{Complex{T}}, m::Int, α::Complex{T}, β::Complex{T}, z::Bool,
        ::Val{NC}, ::Val{CJ}, ::Val{HALF} = Val(false)
    ) where {T, NC, CJ, HALF}
    W = _vwidth(T); lanes = HALF ? W : 2W; cstep = lanes ÷ 2; V2 = Vec{lanes, T}; sz = sizeof(T)
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(lanes - 1))...)
    body = quote
        xr = Ptr{$T}(xp); Ar = Ptr{$T}(Ab)
    end
    for c in 1:NC
        push!(body.args, :($(Symbol(:p, c)) = zero($V2); $(Symbol(:q, c)) = zero($V2)))
    end
    main = quote
        xc = vload($V2, xr + i * 2 * $sz); xcs = shufflevector(xc, Val($swp))
    end
    for c in 1:NC
        av = Symbol(:av, c)
        push!(main.args, :($av = vload($V2, Ar + (i + $(c - 1) * lda) * 2 * $sz)))
        _CGEMVT_PF && push!(main.args, :(_prefetch(Ar + (i + $(c - 1) * lda) * 2 * $sz + $(3 * lanes * sz))))
        push!(main.args, :($(Symbol(:p, c)) = muladd($av, xc, $(Symbol(:p, c)))))
        push!(main.args, :($(Symbol(:q, c)) = muladd($av, xcs, $(Symbol(:q, c)))))
    end
    push!(
        body.args, :(
            nfull = m - rem(m, $cstep); i = 0; @inbounds while i < nfull
                $main; i += $cstep
            end
        )
    )
    for c in 1:NC
        pf = Symbol(:pf, c); qf = Symbol(:qf, c)   # unique names — must NOT collide with accumulators p$c/q$c
        push!(
            body.args, quote
                $pf = _fold2_cmplx($(Symbol(:p, c)))   # [Σ ar·xr, Σ ai·xi]  (parity-preserving fold; see gemm.jl)
                $qf = _fold2_cmplx($(Symbol(:q, c)))   # [Σ ar·xi, Σ ai·xr]
                $(Symbol(:sr, c)) = $pf[1] + $(CJ ? :($pf[2]) : :(-$pf[2]))
                $(Symbol(:si, c)) = $qf[1] + $(CJ ? :(-$qf[2]) : :($qf[2]))
            end
        )
    end
    tail = quote
        xrr = unsafe_load(xr, 2i + 1); xii = unsafe_load(xr, 2i + 2)
    end
    for c in 1:NC
        push!(
            tail.args, quote
                arr = unsafe_load(Ar, 2 * (i + $(c - 1) * lda) + 1); aii = unsafe_load(Ar, 2 * (i + $(c - 1) * lda) + 2)
                $(Symbol(:sr, c)) += arr * xrr + $(CJ ? :(aii * xii) : :(-aii * xii))
                $(Symbol(:si, c)) += arr * xii + $(CJ ? :(-aii * xrr) : :(aii * xrr))
            end
        )
    end
    push!(
        body.args, :(
            @inbounds while i < m
                $tail; i += 1
            end
        )
    )
    for c in 1:NC
        push!(
            body.args, quote
                s = Complex($(Symbol(:sr, c)), $(Symbol(:si, c)))
                yj = unsafe_load(yp, $c)
                unsafe_store!(yp, z ? α * s : muladd(β, yj, α * s), $c)
            end
        )
    end
    push!(body.args, :(return nothing))
    return body
end

# Knob-free runner: (NC, HALF) arrive as compile-time `Val`s so the block kernel specializes and the
# hot loop carries no dynamic dispatch. The measure harness calls this directly with each candidate.
function _gemv_tc_run!(
        m::Int, n::Int, α::Complex{T}, A, x, β::Complex{T}, y, ::Val{CJ}, ::Val{NC}, ::Val{HALF}
    ) where {T <: BlasReal, CJ, NC, HALF}
    z = iszero(β); csz = sizeof(Complex{T})
    GC.@preserve A x y begin
        Ap = pointer(A); lda = stride(A, 2); xp = pointer(x); yp = pointer(y)
        j = 0
        while j + NC <= n                                         # NC-column blocks (shared x + swap)
            _gemv_tc_block_cmplx!(yp + j * csz, Ap + j * lda * csz, lda, xp, m, α, β, z, Val(NC), Val(CJ), Val(HALF))
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

# Static ladder: the resolved (NC, HALF) config → compile-time `Val`s. One branch, each arm statically
# dispatched, so the public complex gemv path stays allocation-free and StrictMode-clean (a dynamic
# `Val(nc)` here would reintroduce the runtime dispatch that has broken the @typestable contract twice).
# Arms the register file cannot hold on this ISA const-fold away.
@inline function _gemv_tc_cmplx!(
        m::Int, n::Int, α::Complex{T}, A, x, β::Complex{T}, y, ::Val{CJ}
    ) where {T <: BlasReal, CJ}
    # ⚠ THE OPTIMUM IS SIZE-DEPENDENT, SO THE RESIDENCY SPLIT IS PART OF THE KNOB — one global config is
    # measurably wrong at one end or the other. Measured 2026-08-07 on wintermute, freq-locked, all arms
    # SAME-RUN, forced via PUREBLAS_FORCE_cgemvt_cfg, vs AOCL:
    #     n      A vs L2      NC=4 wide      NC=8 half
    #     64     ≪ L2         1.007          0.986
    #     128    ≪ L2         0.998          0.985
    #     256    = L2         1.148          1.037
    #     512    4×L2         0.955 MISS     1.006 PASS
    #     1024   16×L2        0.912 MISS     1.017 PASS
    #     2048   ≫ L2         1.004          1.004
    # The crossover sits exactly where A stops fitting L2, which is the physical story this file already
    # told: while A is L2-resident the loop is FMA-latency-bound and wants fewer, wider accumulators;
    # once A streams from L3/DRAM it is bandwidth/MLP-bound and wants more concurrent column streams
    # (AOCL's fused zdotxf runs 8 — see the `_cgemvt_cfg` note). Shipping NC=8 everywhere would have
    # traded two mid-band misses for two small-n ones.
    # DERIVE tier: the SPLIT is a residency criterion over `_L2_BYTES`; only the past-L2 arm is Measured.
    if m * n * sizeof(Complex{T}) <= _L2_BYTES
        return _gemv_tc_run!(m, n, α, A, x, β, y, Val(CJ), Val(_CGEMVT_NC), Val(_CGEMVT_HALF))
    end
    cfg = _cgemvt_cfg()
    if _cgemvt_fits(8, true) && cfg == 108
        return _gemv_tc_run!(m, n, α, A, x, β, y, Val(CJ), Val(8), Val(true), Val(_GEMVT_U))    # req8-ok: candidate arm
    elseif cfg == 104
        return _gemv_tc_run!(m, n, α, A, x, β, y, Val(CJ), Val(4), Val(true), Val(_GEMVT_U))    # req8-ok: candidate arm
    elseif cfg == 2
        return _gemv_tc_run!(m, n, α, A, x, β, y, Val(CJ), Val(2), Val(false))   # req8-ok: candidate arm
    end
    return _gemv_tc_run!(m, n, α, A, x, β, y, Val(CJ), Val(_CGEMVT_NC), Val(_CGEMVT_HALF))
end

function _gemv!(
        trans::Bool, cj::Bool, m::Integer, n::Integer, α::Number, A, x, incx::Integer,
        β::Number, y, incy::Integer
    )
    if !trans
        if iszero(α)
            _scale_y!(Int(m), β, y, incy); return y
        end
        if _l2_simd_ok(A, x, y, incx, incy)   # column-panel kernel handles all n; β folded in
            αT = convert(eltype(A), α); βT = convert(eltype(A), β)
            return iszero(β) ? _gemv_n_simd!(Int(m), Int(n), αT, A, x, y, βT, Val(true)) :
                _gemv_n_simd!(Int(m), Int(n), αT, A, x, y, βT, Val(false))
        end
        if _l2c_ok(A, x, y, incx, incy)       # complex N → OB-structure ri kernel (column-streaming; measured
            αc = convert(eltype(A), α); βc = convert(eltype(A), β)   # ≥ the row-tile at every n on BOTH AVX2 and
            # AVX-512 — the row-tile's 8-col-panel restreaming of y is L3-hostile at mid-n, e.g. Zen4 n=1024
            # 0.77→1.00, n=512 1.11→1.31. (Row-tile _gemv_n_cmplx! is kept: hemv still uses it internally.)
            return iszero(β) ? _gemv_n_ri_cmplx!(Int(m), Int(n), αc, A, x, y, βc, Val(true)) :
                _gemv_n_ri_cmplx!(Int(m), Int(n), αc, A, x, y, βc, Val(false))
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
        # α==0 ⇒ y := βy with A and x NOT referenced (reference ?gemv). The !trans branch above has had
        # this since forever; the trans branch did not, so `gemv 'T'` with α=0 over an A holding Inf/NaN
        # produced NaN in y. Note the output length here is n, not m. Found 2026-08-01 by review.
        if iszero(α)
            _scale_y!(Int(n), β, y, incy); return y
        end
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

# ── ger DRAM path: NP-column m-inner panel (BLASFEO dger shape), NP = concurrent wide-SIMD A-column RMW
# streams. NP is CALIBRATED per box (bench/calibrate.jl) because the optimal stream count is an intrinsic
# per-core property with NO derivable formula and opposite sign across µarchs — MEASURED (prefetch off, both
# DRAM sizes): Zen5→1, Zen3→4, Zen4→8. Every external cause was eliminated (memory subsystem scales fine on
# both; DIMMs rank-matched; OS/clock cancel in the PB/OB ratio; 4K-aliasing padded out; LLVM znver4≡znver5
# codegen on the same silicon) — so this is a genuine tuning knob, not a µarch hack. Default 4 (a safe middle).
const _GER_PANEL_U = 4                                          # x-vector unroll (ILP)
# Complex ger panel budget — DERIVED from the register file, not swept. A complex column holds 2
# `Vec{2W,T}` coefficient vectors, and `Vec{2W,T}` occupies exactly 2 native registers on every ISA
# (2W·sizeof(T) = 2·_SIMD_BYTES by construction), so a column costs 4 registers; each unrolled x step
# costs another 4 (the vector and its swap-adjacent partner). Reserving ~4 for addresses/temporaries:
#     NP_max = (_NVREG - 4 - 4·U) ÷ 4
# AVX-512 (_NVREG=32, U=2) → 5, so the ladder's 4 fits with room; AVX2 (_NVREG=16, U=1) → 2. Halving U
# on the narrow file is what keeps NP ≥ 2 there at all — with U=2 an AVX2 box could only afford NP=1,
# i.e. no panel. See [[register-ceiling-vs-structure]]: count the vectors actually held live.
const _CGER_U = _NVREG >= 32 ? 2 : 1
# ⚠ SNAPPED DOWN TO A POWER OF TWO, and that is a correctness requirement, not tidiness: the driver's
# `Val` ladder only instantiates NP ∈ {2,4,8}, but advances `jc += np`. A raw budget of 5 selected the
# Val(8) arm while striding 5, which reprocesses columns and reads past the last panel. Caught by the
# correctness check, not by review.
#
# A `Vec{lanes,T}` costs `lanes*sizeof(T) ÷ _SIMD_BYTES` native registers — 2 wide (lanes=2W), 1 narrow
# (lanes=W). Per column: 2 coefficient vectors. Per unroll step: the x vector and its swap partner.
# Reserve ~4 for addresses/temporaries.
@inline _cger_np_max(half::Bool) = (
    v = half ? 1 : 2;
    r = (_NVREG - 4 - 2 * _CGER_U * v) ÷ (2 * v);
    r >= 8 ? 8 : r >= 4 ? 4 : r >= 2 ? 2 : 1
)
const _CGER_NP_MAX_WIDE = _cger_np_max(false)
const _CGER_NP_MAX_HALF = _cger_np_max(true)
# ⚠⚠ FALSIFIED BY THE GATE 2026-08-08 — DO NOT RE-CHASE "MORE STREAMS VIA THE NARROW LAYOUT".
# The idea (from the AOCL disassembly: its zaxpyv uses one `vbroadcastsd` per part, 2 registers per
# column, which is how a competitor affords 8 streams where our wide layout affords 4) was to switch
# the panel to `Vec{W}` coefficients and let NP reach the box's measured `_ger_np()` = 8. It is a
# clean derivation and it MADE THINGS MUCH WORSE. Measured on wintermute, freq-locked, all arms
# same-run, zgeru vs AOCL:
#     n=1024   0.901  ->  0.742     <- severe regression at the L3 boundary
#     n=2048   1.355  ->  1.334
#     n=4096   1.401  ->  1.342
# So the narrow layout does not simply buy streams: halving the vector halves the work per instruction
# and doubles the instruction count on a kernel that is already store-bound, and the extra streams do
# not pay for it. AOCL affording 8 streams is not evidence that 8 streams is what makes AOCL fast here
# — it wins in the A≈L3 band with a ONE-stream axpyv (see `_axpy_cmplx_cold!`), so its register layout
# was never the mechanism. Reading a competitor's shape is a hypothesis generator, not a result.
# The kernel keeps its `Val{HALF}` parameter (it costs nothing and the arm stays measurable), but the
# shipped layout is WIDE and NP stays capped by the wide budget.
@inline _cger_half() = false
@inline _cger_np() = min(_ger_np(), _CGER_NP_MAX_WIDE)
# NP resolution. A Preference (`ger_panel_np`, written by bench/calibrate.jl or the juliac build) PINS it;
# else it is auto-measured ONCE per process on the first DRAM ger via `OncePerProcess` — no __init__, so a
# trimmed .so never runs a benchmark at load. `@static if` (not DCE-by-faith): when the pref IS set (every
# trim/.so build MUST set it), the auto path is NEVER DEFINED → trivially trim-clean.
const _GER_NP_PREF = @load_preference("ger_panel_np", nothing)
@static if isnothing(_GER_NP_PREF)
    # Base-only, TOTAL (OncePerProcess poisons the whole process if the initializer throws) → catch → 4.
    function _measure_ger_np()::Int
        Base.generating_output() && return 4                     # don't burn a measure during precompilation
        _f = _force_knob("ger_np"); _f >= 0 && return _f          # instrument only, see _force_knob
        try
            # ⚠ SQUARE, matching the shape the GATE scores. This probed n=64 columns against
            # m = 2·L3/64 rows — a 65536×64 tall-skinny panel — while bench/plots.jl measures `ger` at
            # `randn(s,s)`. That is the same probe-regime defect found in `_measure_cgemvn_nc_big` the
            # day before (32768×32 vs square), and here it had a measured cost: with the tall-skinny
            # probe made REPRODUCIBLE by rotation the knob settled on 4, and the gate then read
            # ger@2048 0.989 — a regression from 1.199, because np=8 is right for the square shape.
            # Making a wrong-shape measurement stable just ships the wrong answer consistently.
            # Square at ~2×L3 puts m = n ≈ 2048 on Zen4, which is exactly the cell that regressed.
            n = max(64, isqrt(2 * _L3_BYTES ÷ sizeof(Float64)))
            n -= n % 8                                           # multiple of the widest NP candidate
            m = n
            # NB OPERAND SETS, ROTATED PER ROUND. Fleet audit 2026-08-07 (one knob per fresh process):
            # this knob resolved 4/2 on Zen3 and 8/4 on Zen4 — a coin toss on two of three boxes — while
            # `axpy_band`, the only knob with rotation, was the ONLY one stable on all three. Duelling
            # resamples time; when the winner depends on state fixed once per process (page placement,
            # THP, allocator addresses) every round re-measures the same draw. `A` here is ~2×L3, so its
            # placement is exactly the kind of per-process accident that decides a DRAM stream count.
            nb = 3                                               # 3 × 2×L3 of scratch; enough to vary placement
            As = [fill(1.0, m, n) for _ in 1:nb]
            x = fill(1.0, m); y = fill(1.0, n)                   # pre-touched (no first-touch bias)
            # MEDIAN per candidate, not min (see cpuinfo.jl `_tune_one`). This knob has OPPOSITE SIGN
            # across µarchs (Zen5→1, Zen3→4, Zen4→8), so it is exactly the case where the luckiest
            # window of one candidate must not decide what ships.
            # ⚠ SEED `bt` WITH THE DEFAULT'S OWN MEASURED TIME, not `typemax`. With `bt = typemax`,
            # `_tune_better(t, bt)` is `t*100 < typemax*95`, and `typemax*95` WRAPS to 2^64-95 — still
            # astronomically above any real time, so the FIRST swept candidate always displaces and the
            # declared default (`best = 4`) can never win. The effective incumbent was np=1 purely
            # because it is first in the list, which silently voided the invariant the margin exists for
            # ("ties go to the incumbent, which is the derived default", cpuinfo.jl). It bites hardest on
            # exactly the noisy boxes the margin was built for: where all candidates sit within 5%, the
            # shipped kernel was "first in the list", undocumented.
            # Seeded this way the loop is extensionally identical to `_tune_pick` (bt starts at the
            # incumbent and only decreases, so every displacement also clears the incumbent) — see the
            # proof note there, and test/tuner_tests.jl for the pinned semantics.
            # DUELS + ROTATION, replacing the margin. `Ar` names the operand set for the current round;
            # `rot` advances it, so each round is a fresh placement draw and the sign count aggregates
            # over placements — which is what a caller sees, since callers do not share one allocation.
            Ar = Ref(As[1])
            rot(r) = (Ar[] = As[mod1(r, nb)]; nothing)
            inc() = _ger_paneldrv_np(m, n, 1.0, x, y, Ar[], 4)
            for np in (8, 2, 1)                        # widest first; the default (4) is the incumbent
                _tune_wins_it(_tune_duel(inc, () -> _ger_paneldrv_np(m, n, 1.0, x, y, Ar[], np); refresh = rot)) && return np
            end
            return 4
        catch
            return 4
        end
    end
    const _GER_NP_ONCE = Base.OncePerProcess{Int}(_measure_ger_np)
    @inline _ger_np() = _GER_NP_ONCE()
else
    @inline _ger_np() = _GER_NP_PREF::Int
end

@generated function _ger_panel!(
        Aptr::Ptr{T}, lda::Int, xp::Ptr{T}, yp::Ptr{T},
        jc::Int, m::Int, α::T, pf::Int, ::Val{NP}, ::Val{U}
    ) where {T, NP, U}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T); step = U * W
    body = quote
        lanes = Vec{$W, Int}($(Expr(:tuple, (0:(W - 1))...)))
    end
    for c in 1:NP
        push!(body.args, :($(Symbol(:ay, c)) = $V(α * unsafe_load(yp, jc + $c))))       # α·y[jc+c-1]
        push!(body.args, :($(Symbol(:ac, c)) = Aptr + (jc + $(c - 1)) * lda * $sz))      # column base
    end
    main = quote end
    for u in 1:U
        push!(main.args, :($(Symbol(:xv, u)) = vload($V, xp + (i + $((u - 1) * W)) * $sz)))
    end
    for c in 1:NP
        push!(
            main.args, quote
                if pf > 0                                                                    # dense prefetch of this A-column pf ahead
                    pb = $(Symbol(:ac, c)) + (i + pf) * $sz
                    for cl in 0:$_CACHELINE:$(step * sz - 1)
                        _prefetch(pb + cl)
                    end
                end
            end
        )
        for u in 1:U
            push!(main.args, :(p = $(Symbol(:ac, c)) + (i + $((u - 1) * W)) * $sz))
            push!(main.args, :(vstore(muladd($(Symbol(:ay, c)), $(Symbol(:xv, u)), vload($V, p)), p)))
        end
    end
    push!(body.args, :(i = 0))
    push!(
        body.args, :(
            while i + $step <= m
                $main; i += $step
            end
        )
    )
    tail = quote
        msk = lanes < (m - i); xt = vload($V, xp + i * $sz, msk)
    end
    for c in 1:NP
        push!(tail.args, :(pt = $(Symbol(:ac, c)) + i * $sz))
        push!(tail.args, :(vstore(muladd($(Symbol(:ay, c)), xt, vload($V, pt, msk)), pt, msk)))
    end
    push!(
        body.args, :(
            while i < m
                $tail; i += $W
            end
        )
    )
    push!(body.args, :(return nothing))
    return body
end

@inline function _ger_panel_driver!(m::Int, n::Int, α::T, x, y, A, ::Val{NP}) where {T <: BlasReal, NP}
    GC.@preserve A x y begin
        Aptr = pointer(A); xptr = pointer(x); yptr = pointer(y); lda = stride(A, 2); sz = sizeof(T)
        jc = 0
        while jc + NP <= n
            _ger_panel!(Aptr, lda, xptr, yptr, jc, m, α, 0, Val(NP), Val(_GER_PANEL_U))   # pf=0: NP is the lever
            jc += NP
        end
        @inbounds while jc < n                                                            # remainder columns (< NP)
            ayj = α * unsafe_load(yptr, jc + 1)
            iszero(ayj) || _axpy_simd!(m, ayj, xptr, Aptr + jc * lda * sz, 0)
            jc += 1
        end
    end
    return A
end

# Static Val ladder: runtime NP (from _ger_np()) → a compile-time Val{NP} driver call. One branch, each arm
# statically dispatched (no dynamic Val(NP) in the hot loop → zero-alloc, StrictMode-clean).
@inline function _ger_paneldrv_np(m::Int, n::Int, α::T, x, y, A, np::Int) where {T <: BlasReal}
    return np == 1 ? _ger_panel_driver!(m, n, α, x, y, A, Val(1)) :
        np == 2 ? _ger_panel_driver!(m, n, α, x, y, A, Val(2)) :
        np == 4 ? _ger_panel_driver!(m, n, α, x, y, A, Val(4)) :
        _ger_panel_driver!(m, n, α, x, y, A, Val(8))
end

@inline function _ger_simd!(m::Int, n::Int, α::T, x, y, A) where {T <: BlasReal}
    # DRAM-bound (A > L3) → m-inner panel with a per-box stream count (`_ger_np()`: Preference or auto-measured). The optimal number
    # of concurrent wide-SIMD write-streams is an intrinsic per-core property with NO derivable formula and
    # OPPOSITE sign across µarchs (measured, prefetch off: Zen5→NP1, Zen3→NP4, Zen4→NP8; all external causes —
    # memory, DIMMs, OS, codegen, aliasing — eliminated). So it's calibrated per box (see bench/calibrate.jl),
    # not gated by a µarch `if`. Cache-resident A stays on the simple per-column axpy below (gates small-n).
    m * n * sizeof(T) >= _L3_BYTES && return _ger_paneldrv_np(m, n, α, x, y, A, _ger_np())  # ≥: A that fills L3 leaves no room for x/y ⇒ panel (galen n=2048: A=L3 exactly, per-column 0.97 → panel 1.04)
    pf = 0                                               # cache-resident: prefetch never helped (regressed n=512)
    GC.@preserve A x y begin
        Aptr = pointer(A); xptr = pointer(x); yptr = pointer(y); lda = stride(A, 2); sz = sizeof(T)
        @inbounds for j in 1:n
            ayj = α * unsafe_load(yptr, j)
            iszero(ayj) || _axpy_simd!(m, ayj, xptr, Aptr + (j - 1) * lda * sz, pf)  # A[:,j] += ayj·x
        end
    end
    return A
end

# Complex rank-1 PANEL: NP columns per pass, x loaded once and reused across all of them — the complex
# mirror of `_ger_panel!`. Structure and register accounting are the only differences worth stating:
# a complex coefficient needs TWO live vectors per column (the α·y[j] broadcast and its sign-alternated
# partner for the swap-adjacent product) against the real kernel's one, and each is `Vec{2W}` = two
# native registers, so a column costs 4 registers here versus 1 there. That is why NP is capped by
# `_NVREG` at the call site rather than inheriting the real path's value unexamined.
@generated function _ger_panel_cmplx!(
        Ap::Ptr{Complex{T}}, lda::Int, xp::Ptr{Complex{T}}, yp::Ptr{Complex{T}},
        jc::Int, m::Int, α::Complex{T}, ::Val{NP}, ::Val{CJ}, ::Val{U}, ::Val{HALF}
    ) where {T <: BlasReal, NP, CJ, U, HALF}
    # HALF picks the accumulator/coefficient width. `Vec{2W}` is 2 native registers, `Vec{W}` is 1, so
    # a column costs 4 registers wide and 2 narrow — and that is precisely what caps NP. AOCL's zaxpyv
    # uses the narrow layout (one `vbroadcastsd` per part), which is how it affords 8 streams where our
    # wide layout could only afford 4. Same trick `_CGEMVT_HALF` already plays for gemv-T/C.
    W = _vwidth(T); lanes = HALF ? W : 2W; cstep = lanes ÷ 2   # complex elements per vector
    V2 = Vec{lanes, T}; sz = sizeof(T); step = U * cstep
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(lanes - 1))...)
    body = quote
        pxr = Ptr{$T}(xp)                                  # real-interleaved views for the vector loads
        par = Ptr{$T}(Ap)
    end
    for c in 1:NP
        # ay = α·(cj ? conj(y[j]) : y[j]) — the SAME scalar the per-column path forms, so the two paths
        # are bit-identical per element and the panel is a pure scheduling change.
        push!(body.args, :($(Symbol(:yv, c)) = unsafe_load(yp, jc + $c)))
        push!(body.args, :($(Symbol(:ay, c)) = α * $(CJ ? :(conj($(Symbol(:yv, c)))) : Symbol(:yv, c))))
        push!(body.args, :($(Symbol(:ar, c)) = $V2(real($(Symbol(:ay, c))))))
        push!(
            body.args, :(
                $(Symbol(:si, c)) = $V2($(Expr(:tuple, (iseven(l) ? :(-imag($(Symbol(:ay, c)))) :
                                                        :(imag($(Symbol(:ay, c)))) for l in 0:(lanes - 1))...)))
            )
        )
        push!(body.args, :($(Symbol(:ac, c)) = par + (jc + $(c - 1)) * lda * 2 * $sz))
    end
    main = quote end
    for u in 1:U
        push!(main.args, :($(Symbol(:xv, u)) = vload($V2, pxr + (i + $((u - 1) * cstep)) * 2 * $sz)))
        push!(main.args, :($(Symbol(:xs, u)) = shufflevector($(Symbol(:xv, u)), Val($swp))))
    end
    for c in 1:NP, u in 1:U
        push!(main.args, :(p = $(Symbol(:ac, c)) + (i + $((u - 1) * cstep)) * 2 * $sz))
        push!(main.args, :(t = muladd($(Symbol(:xv, u)), $(Symbol(:ar, c)), vload($V2, p))))
        push!(main.args, :(vstore(muladd($(Symbol(:xs, u)), $(Symbol(:si, c)), t), p)))
    end
    tail = quote end
    for c in 1:NP
        push!(
            tail.args, quote
                j = i + 1
                xr = unsafe_load(pxr, 2j - 1); xi = unsafe_load(pxr, 2j)
                q = $(Symbol(:ac, c)); ar = real($(Symbol(:ay, c))); ai = imag($(Symbol(:ay, c)))
                unsafe_store!(q, unsafe_load(q, 2j - 1) + ar * xr - ai * xi, 2j - 1)
                unsafe_store!(q, unsafe_load(q, 2j) + ar * xi + ai * xr, 2j)
            end
        )
    end
    push!(body.args, :(i = 0))
    push!(body.args, :(while i + $step <= m
        $main; i += $step
    end))
    push!(body.args, :(while i < m
        $tail; i += 1
    end))
    push!(body.args, :(return nothing))
    return body
end

# Static Val ladder: runtime NP → compile-time Val{NP}, one branch, each arm statically dispatched (no
# dynamic Val in the hot path → allocation-free, StrictMode-clean). Mirrors `_ger_paneldrv_np`.
@inline function _ger_paneldrv_cmplx!(
        m::Int, n::Int, α::Complex{T}, x, y, A, cj::Bool, np::Int
    ) where {T <: BlasReal}
    h = _cger_half()
    return cj ?
        (h ? _ger_pdc_cj!(m, n, α, x, y, A, np, Val(true), Val(true)) :
        _ger_pdc_cj!(m, n, α, x, y, A, np, Val(true), Val(false))) :
        (h ? _ger_pdc_cj!(m, n, α, x, y, A, np, Val(false), Val(true)) :
        _ger_pdc_cj!(m, n, α, x, y, A, np, Val(false), Val(false)))
end
@inline function _ger_pdc_cj!(
        m::Int, n::Int, α::Complex{T}, x, y, A, np::Int, ::Val{CJ}, ::Val{HALF}
    ) where {T <: BlasReal, CJ, HALF}
    GC.@preserve A x y begin
        Ap = pointer(A); xp = pointer(x); yp = pointer(y); lda = stride(A, 2); csz = sizeof(Complex{T})
        jc = 0
        while jc + np <= n
            if np == 2
                _ger_panel_cmplx!(Ap, lda, xp, yp, jc, m, α, Val(2), Val(CJ), Val(_CGER_U), Val(HALF))  # req8-ok: candidate arm
            elseif np == 4
                _ger_panel_cmplx!(Ap, lda, xp, yp, jc, m, α, Val(4), Val(CJ), Val(_CGER_U), Val(HALF))  # req8-ok: candidate arm
            else
                _ger_panel_cmplx!(Ap, lda, xp, yp, jc, m, α, Val(8), Val(CJ), Val(_CGER_U), Val(HALF))  # req8-ok: candidate arm
            end
            jc += np
        end
        @inbounds while jc < n                                              # remainder columns (< np)
            yj = CJ ? conj(unsafe_load(yp, jc + 1)) : unsafe_load(yp, jc + 1)
            ayj = α * yj
            # `_cold!`, not `_simd!`: the driver only runs when A ≥ L3, so a remainder column is just
            # as non-resident as a panel one. Sizing residency from the column's own bytes here would
            # reintroduce the exact bug this path exists to fix.
            iszero(ayj) ||
                _axpy_cmplx_cold!(m, real(ayj), imag(ayj), xp, Ap + jc * lda * csz)
            jc += 1
        end
    end
    return A
end

# Complex rank-1: A[:,j] += (α·(cj ? conj(y[j]) : y[j]))·x — one complex axpy of x into each contiguous
# column, reusing the L1 _axpy_cmplx_simd! kernel (like the real _ger_simd! reuses _axpy_simd!).
function _ger_cmplx!(m::Int, n::Int, α::Complex{T}, x, y, A, cj::Bool) where {T <: BlasReal}
    csz = sizeof(Complex{T})
    # ⚠ DRAM-BOUND A → PANEL, exactly as the real path does. This asymmetry was a measured gate failure:
    # real `ger` has routed A ≥ L3 to `_ger_paneldrv_np` (NP concurrent write streams) since the stream
    # count was calibrated per box, and the complex path never got it — it ran ONE read+write stream at
    # every size. Measured 2026-08-07 on wintermute, freq-locked, ALL ARMS SAME-RUN (so this is not the
    # cross-run drift that manufactured a phantom miss on Zen3 the same day), zgeru vs AOCL:
    #     n=512  A=4.2 MB  < L3   1.056 PASS
    #     n=1024 A=16.8 MB ≥ L3   0.885 MISS
    #     n=2048 A=67 MB   ≥ L3   0.883 MISS
    #     n=4096 A=268 MB  ≥ L3   0.915 MISS
    # Every cell above L3 misses and every cell below it passes — the split is exactly the threshold the
    # real path switches at. Once A is DRAM-resident the op is bound by memory-level parallelism, and one
    # stream cannot keep enough misses in flight; that is the same physics `_ger_np`'s own comment
    # records ("optimal number of concurrent wide-SIMD write-streams ... intrinsic per-core property").
    #
    # NP is the per-box Measure knob CAPPED BY A DERIVED REGISTER BUDGET, not inherited outright: a
    # complex column holds two `Vec{2W}` coefficient vectors (the α·y broadcast and its sign-alternated
    # partner) against the real kernel's one, and `Vec{2W,T}` is always exactly 2 native registers
    # (2W·sizeof(T) = 2·_SIMD_BYTES by construction), so a column costs 4 registers here versus 1 there.
    # Spending the real path's NP=8 would be 32 registers of coefficients alone and spill on every ISA.
    cold = m * n * csz >= _L3_BYTES        # the CALLER's footprint — see `_axpy_cmplx_cold!`
    if cold
        np = _cger_np()          # measured stream count, capped by the chosen layout's register budget
        np >= 2 && return _ger_paneldrv_cmplx!(m, n, α, x, y, A, cj, np)
        # np == 1 IS the per-column path below (one stream) — no separate arm needed. This is the LIVE
        # path on any box whose measured stream optimum is 1 (Zen5 resolves `_ger_np()` = 1), so the
        # cold-column fix below is that box's whole remedy, not an edge case.
    end
    # Val, not Bool: `cold` is loop-invariant, so a runtime flag would put a branch inside the
    # per-column hot loop and block specialization of the axpy arm it selects.
    return cold ? _ger_cmplx_percol!(m, n, α, x, y, A, cj, Val(true)) :
        _ger_cmplx_percol!(m, n, α, x, y, A, cj, Val(false))
end

# Per-column complex ger. `cold` says the CALLER is sweeping past L3, so each column is non-resident
# however small the column itself is — without it the callee sizes residency from its own 16 KiB
# argument and picks the L1-resident arm for a stone-cold stream (see `_axpy_cmplx_cold!` for the full
# post-mortem and the AOCL disassembly that found it).
@inline function _ger_cmplx_percol!(
        m::Int, n::Int, α::Complex{T}, x, y, A, cj::Bool, ::Val{COLD}
    ) where {T <: BlasReal, COLD}
    csz = sizeof(Complex{T})
    GC.@preserve A x y begin
        Aptr = pointer(A); xptr = pointer(x); yptr = pointer(y); lda = stride(A, 2)
        @inbounds for j in 1:n
            yj = cj ? conj(unsafe_load(yptr, j)) : unsafe_load(yptr, j)
            ayj = α * yj
            iszero(ayj) && continue
            p = Aptr + (j - 1) * lda * csz
            COLD ? _axpy_cmplx_cold!(m, real(ayj), imag(ayj), xptr, p) :
                _axpy_cmplx_simd!(m, real(ayj), imag(ayj), xptr, p)
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
@inline function _symv_col!(L::Int, axj::T, aptr::Ptr{T}, xptr::Ptr{T}, yptr::Ptr{T}) where {T <: BlasReal}
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
    for v in 1:K
        push!(q.args, :($(Symbol(:yy, v)) = $(ld(:yp, v))))
    end
    for v in 1:K
        push!(q.args, :($(Symbol(:xx, v)) = $(ld(:xp, v))))
    end
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
        push!(
            q.args, masked ? :(vstore($(Symbol(:yy, v)), yp + (i + $((v - 1) * W)) * $sz, $(Symbol(:k, v)))) :
                :(vstore($(Symbol(:yy, v)), yp + (i + $((v - 1) * W)) * $sz))
        )
    end
    return q
end

# Unified LOWER panel: cols 0:NB-1, rows 0:M-1 (stored r≥c). The triangular diagonal block (rows
# 0:NB-1, one masked vector) feeds the SAME d_c vector accumulators as the off-diagonal rectangle
# (rows NB:M-1) → ONE reduction per column and a vectorized diagonal (the small-n win). The diagonal
# entry is taken once via the axpy (mask r≥c); the dot (mask r>c) is strictly-lower. α in xj_c.
@generated function _symv_lowerpanel!(
        M::Int, α::T, Ap::Ptr{T}, lda::Int, xp::Ptr{T}, yp::Ptr{T},
        ::Val{MR}, ::Val{NB}
    ) where {T, MR, NB}
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
    push!(
        body.args, :(
            i = $NB; nfull = M - rem(M - $NB, $mr); while i < nfull
                $(_symv_offblk_expr(W, V, sz, NB, MR, false)); i += $mr
            end
        )
    )
    branches = _symv_offblk_expr(W, V, sz, NB, 1, true)
    for k in 2:MR
        branches = Expr(:if, :(rmn > $((k - 1) * W)), _symv_offblk_expr(W, V, sz, NB, k, true), branches)
    end
    push!(
        body.args, :(
            if i < M
                rmn = M - i; $branches
            end
        )
    )
    for c in 1:NB
        push!(body.args, :(unsafe_store!(yp, muladd(α, sum($(Symbol(:d, c))), unsafe_load(yp, $c)), $c)))
    end
    push!(body.args, :(return nothing))
    return body
end

# Unified UPPER panel: cols 0:NB-1 (global jb+c), rows 0:M-1 with M=jb+NB (stored r≤jb+c). Off-diagonal
# rectangle rows 0:dboff-1 (dboff=M-NB=jb) THEN triangular diagonal block rows dboff:dboff+NB-1.
@generated function _symv_upperpanel!(
        M::Int, α::T, Ap::Ptr{T}, lda::Int, xp::Ptr{T}, yp::Ptr{T},
        ::Val{MR}, ::Val{NB}
    ) where {T, MR, NB}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T); mr = MR * W
    body = quote end
    push!(body.args, :(lanes = Vec{$W, Int}($(Expr(:tuple, (0:(W - 1))...)))))
    push!(body.args, :(zv = zero($V)))
    push!(body.args, :(dboff = M - $NB))
    for c in 1:NB
        push!(body.args, :($(Symbol(:xj, c)) = $V(α * unsafe_load(xp, dboff + $c))))
        push!(body.args, :($(Symbol(:d, c)) = zero($V)))
    end
    push!(
        body.args, :(
            i = 0; nfull = dboff - rem(dboff, $mr); while i < nfull
                $(_symv_offblk_expr(W, V, sz, NB, MR, false)); i += $mr
            end
        )
    )
    branches = _symv_offblk_expr(W, V, sz, NB, 1, true)
    for k in 2:MR
        branches = Expr(:if, :(rmn > $((k - 1) * W)), _symv_offblk_expr(W, V, sz, NB, k, true), branches)
    end
    push!(
        body.args, :(
            if i < dboff
                rmn = dboff - i; $branches
            end
        )
    )
    push!(body.args, :(mblk = lanes < $NB))
    push!(body.args, :(yblk = vload($V, yp + dboff * $sz, mblk)))
    push!(body.args, :(xblk = vload($V, xp + dboff * $sz, mblk)))
    for c in 1:NB
        push!(body.args, :(acd = vload($V, Ap + (dboff + $(c - 1) * lda) * $sz, mblk)))
        push!(body.args, :(yblk = vifelse(lanes <= $(c - 1), muladd($(Symbol(:xj, c)), acd, yblk), yblk)))
        push!(body.args, :($(Symbol(:d, c)) = muladd(vifelse(lanes < $(c - 1), acd, zv), xblk, $(Symbol(:d, c)))))
    end
    push!(body.args, :(vstore(yblk, yp + dboff * $sz, mblk)))
    for c in 1:NB
        push!(body.args, :(unsafe_store!(yp, muladd(α, sum($(Symbol(:d, c))), unsafe_load(yp, dboff + $c)), dboff + $c)))
    end
    push!(body.args, :(return nothing))
    return body
end

@inline function _symv_simd!(up::Bool, n::Int, α::T, A, x, y) where {T <: BlasReal}
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
@generated function _hemv_col_cmplx!(L::Int, tmpr::T, tmpi::T, ap::Ptr{T}, xp::Ptr{T}, yp::Ptr{T}) where {T <: BlasReal}
    W = _vwidth(T); V2 = Vec{2W, T}; sz = sizeof(T)
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(2W - 1))...)
    return quote
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

# Blocked off-diagonal rectangle for complex hemv: NB stored columns processed TOGETHER so each x/y
# row-tile is read ONCE across all NB columns (the per-column kernel above re-reads x/y once PER column
# — fine while x,y fit L1, but it collapses when they spill, e.g. n≥1024: measured PB 33→23 GB/s vs
# AOCL's ~38 as n grows). Each A element is still read once, feeding BOTH its column axpy (y += tmp_c·a)
# and its conj-dot (s_c += conj(a)·x_row) — same swap-adjacent idiom as `_hemv_col_cmplx!`. `arp` → the
# rectangle's top-left A element (reals); M rows; `ldc` = A complex column stride; `xrp`/`yrp` → the
# x/y row-segment; `xcp` → x at the NB panel columns. Returns ((sr₁,si₁),…) partial conj-dots per col.
# A-stream software prefetch in the blocked rect kernel — AVX2 only (matches `_CGEMVN_PF`: the narrow
# machine's HW prefetcher can't sustain the NB concurrent A column streams; +192 B prefetch hides the
# DRAM latency that leaves PB below AOCL at n≥2048. AVX-512 boxes don't need it — override via Preference.
const _ZHEMV_PF = @load_preference("zhemv_pf", _vwidth(Float64) == 4)::Bool
# Prefetch distance in row-tiles (1 tile = W complex = one 64 B line @ W=4). The large-n A read sits
# deep in DRAM (n≥2048 ⇒ A ≫ L3), so hemv wants a longer look-ahead than gemvN's 3 lines to cover the
# ~hundreds-of-cycles DRAM latency; empirical (a latency behaviour, not a datasheet number) — Preference.
const _ZHEMV_PF_TILES = @load_preference("zhemv_pf_tiles", 8)::Int

@generated function _hemv_rect_cmplx!(
        M::Int, arp::Ptr{T}, ldc::Int, xrp::Ptr{T}, yrp::Ptr{T},
        xcp::Ptr{T}, αr::T, αi::T, ::Val{NB}
    ) where {T <: BlasReal, NB}
    W = _vwidth(T); V2 = Vec{2W, T}; sz = sizeof(T)
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(2W - 1))...)
    body = quote end
    for c in 1:NB                                              # tmp_c = α·x[col_c]; conj-dot accumulators
        push!(
            body.args, quote
                xcr = unsafe_load(xcp, $(2c - 1)); xci = unsafe_load(xcp, $(2c))
                $(Symbol(:cr, c)) = αr * xcr - αi * xci; $(Symbol(:ci, c)) = αr * xci + αi * xcr
                $(Symbol(:crv, c)) = $V2($(Symbol(:cr, c)))
                $(Symbol(:csgn, c)) = $V2($(Expr(:tuple, (iseven(l) ? :(-$(Symbol(:ci, c))) : Symbol(:ci, c) for l in 0:(2W - 1))...)))
                $(Symbol(:pac, c)) = zero($V2); $(Symbol(:qac, c)) = zero($V2)
            end
        )
    end
    loop = quote                                              # one W-complex row-tile, all NB columns
        o = i * 2 * $sz
        yv = vload($V2, yrp + o); xv = vload($V2, xrp + o); xsw = shufflevector(xv, Val($swp))
    end
    for c in 1:NB
        push!(
            loop.args, quote
                av = vload($V2, arp + o + $((c - 1)) * ldc * 2 * $sz)
                yv = muladd(av, $(Symbol(:crv, c)), yv); yv = muladd(shufflevector(av, Val($swp)), $(Symbol(:csgn, c)), yv)
                $(Symbol(:pac, c)) = muladd(av, xv, $(Symbol(:pac, c))); $(Symbol(:qac, c)) = muladd(av, xsw, $(Symbol(:qac, c)))
            end
        )
        # A-stream prefetch, +3 complex-tiles (192 B @ W=4) ahead per column, like the gemvN panel's
        # _CGEMVN_PF: hides the DRAM latency of the large-n A read (the residual vs AOCL at n≥2048).
        _ZHEMV_PF && push!(loop.args, :(_prefetch(arp + o + $((c - 1)) * ldc * 2 * $sz + $(_ZHEMV_PF_TILES * 2W * sz))))
    end
    push!(loop.args, :(vstore(yv, yrp + o)))
    push!(
        body.args, :(
            i = 0; while i + $W <= M
                $loop; i += $W
            end
        )
    )
    for c in 1:NB                                              # reduce: conj(a)·x = (ar·xr+ai·xi)+i(ar·xi−ai·xr)
        push!(
            body.args, quote
                (prc, pic) = _deint_cmplx($(Symbol(:pac, c))); (qrc, qic) = _deint_cmplx($(Symbol(:qac, c)))
                $(Symbol(:sr, c)) = sum(prc) + sum(pic); $(Symbol(:si, c)) = sum(qrc) - sum(qic)
            end
        )
    end
    rem = quote                                               # scalar row remainder (< W rows), all NB cols
        j2 = 2i; xrr = unsafe_load(xrp, j2 + 1); xri = unsafe_load(xrp, j2 + 2)
    end
    for c in 1:NB
        push!(
            rem.args, quote
                ar = unsafe_load(arp, j2 + 1 + $((c - 1)) * ldc * 2); ai = unsafe_load(arp, j2 + 2 + $((c - 1)) * ldc * 2)
                unsafe_store!(yrp, unsafe_load(yrp, j2 + 1) + $(Symbol(:cr, c)) * ar - $(Symbol(:ci, c)) * ai, j2 + 1)
                unsafe_store!(yrp, unsafe_load(yrp, j2 + 2) + $(Symbol(:cr, c)) * ai + $(Symbol(:ci, c)) * ar, j2 + 2)
                $(Symbol(:sr, c)) += ar * xrr + ai * xri; $(Symbol(:si, c)) += ar * xri - ai * xrr
            end
        )
    end
    push!(
        body.args, :(
            @inbounds while i < M
                $rem; i += 1
            end
        )
    )
    push!(body.args, Expr(:return, Expr(:tuple, (Expr(:tuple, Symbol(:sr, c), Symbol(:si, c)) for c in 1:NB)...)))
    return body
end

# Complex hemv panel width: 2 conj-dot Vec{2W} accumulators per column ⇒ 4 physical vector regs/column
# (a Vec{2W} legalizes to 2 regs on both AVX2 and AVX-512). Reserve half the architectural vector regs
# (_NVREG: 16 AVX2 / 32 AVX-512) for accumulators, rest for the x/y/A working set ⇒ AVX2→2, AVX-512→4.
const _ZHEMV_NB = clamp((_NVREG ÷ 2) ÷ 4, 1, 4)

# Complex hemv (A Hermitian, `up` triangle stored, diagonal real): y := β·y + α·A·x. Off-diagonal work
# is done in NB-column panels (`_hemv_rect_cmplx!`, blocked → x/y-traffic-efficient at large n); the tiny
# NB×NB diagonal triangle + real-diagonal term are finished per column (mirrors the reference recurrence).
function _hemv_cmplx!(up::Bool, n::Int, α::Complex{T}, A, x, β::Complex{T}, y) where {T <: BlasReal}
    _scale_y!(n, β, y, 1)                                       # β·y (complex scal via _scal_cmplx_simd!)
    iszero(α) && return y
    NB = _ZHEMV_NB; sz = sizeof(T); αr = real(α); αi = imag(α)
    GC.@preserve A x y begin
        Ap = Ptr{T}(pointer(A)); xp = Ptr{T}(pointer(x)); yp = Ptr{T}(pointer(y))
        Apc = Ptr{Complex{T}}(pointer(A)); xpc = Ptr{Complex{T}}(pointer(x)); ypc = Ptr{Complex{T}}(pointer(y))
        ldc = stride(A, 2)
        jb = 0
        @inbounds while jb + NB <= n                           # full NB-column panels
            if up                                              # rectangle = rows 0:jb-1 (above the block)
                s = _hemv_rect_cmplx!(jb, Ap + jb * 2 * ldc * sz, ldc, xp, yp, xp + jb * 2 * sz, αr, αi, Val(NB))
            else                                               # rectangle = rows jb+NB:n-1 (below the block)
                r0 = jb + NB
                s = _hemv_rect_cmplx!(
                    n - r0, Ap + (r0 + jb * ldc) * 2 * sz, ldc, xp + r0 * 2 * sz, yp + r0 * 2 * sz,
                    xp + jb * 2 * sz, αr, αi, Val(NB)
                )
            end
            for c in 1:NB                                      # NB×NB diagonal triangle + real diagonal, per col
                j = jb + c                                     # 1-based column index
                tmp = α * unsafe_load(xpc, j)
                sc = Complex{T}(s[c][1], s[c][2])
                if up
                    for i in (jb + 1):(j - 1)                  # intra-block strictly-upper rows
                        aij = unsafe_load(Apc, (j - 1) * ldc + i)
                        unsafe_store!(ypc, unsafe_load(ypc, i) + tmp * aij, i); sc += conj(aij) * unsafe_load(xpc, i)
                    end
                else
                    for i in (j + 1):(jb + NB)                 # intra-block strictly-lower rows
                        aij = unsafe_load(Apc, (j - 1) * ldc + i)
                        unsafe_store!(ypc, unsafe_load(ypc, i) + tmp * aij, i); sc += conj(aij) * unsafe_load(xpc, i)
                    end
                end
                ajj = unsafe_load(Ap, ((j - 1) * ldc + (j - 1)) * 2 + 1)     # real(A[j,j])
                unsafe_store!(ypc, unsafe_load(ypc, j) + tmp * ajj + α * sc, j)
            end
            jb += NB
        end
        @inbounds while jb < n                                 # tail columns (< NB): per-column kernel
            j = jb + 1
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
            jb += 1
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
    rcp = _trsv_rrcpbuf(T)
    # Take the divide OFF the sequential substitution critical path (see _TRSV_RRCP64 above). `userc` is
    # loop-invariant, so the in-loop ternary predicts perfectly / unswitches — same shape the complex
    # `_trsv_cmplx!` already uses, which is why this needs no duplicate branch bodies.
    userc = !unit && n <= length(rcp)
    GC.@preserve A x begin
        Ap = pointer(A); xp = pointer(x); lda = stride(A, 2); sz = sizeof(T)
        dp(j) = Ap + ((j - 1) * lda + (j - 1)) * sz          # &A[j,j]
        if userc                                 # r[j] = 1/A[j,j]: all independent ⇒ pipelined
            @inbounds @simd ivdep for j in 1:n
                rcp[j] = inv(unsafe_load(dp(j)))
            end
        end
        if !tr                                   # solve A·x = b  (column axpy, subtract)
            if up                                # U,N: back-substitution, j descending
                @inbounds for j in n:-1:1
                    cp = Ap + (j - 1) * lda * sz
                    unit || unsafe_store!(xp, userc ? unsafe_load(xp, j) * rcp[j] :
                                              unsafe_load(xp, j) / unsafe_load(cp + (j - 1) * sz), j)
                    _axpy_simd!(j - 1, -unsafe_load(xp, j), cp, xp)
                end
            else                                 # L,N: forward, j ascending
                @inbounds for j in 1:n
                    cp = Ap + (j - 1) * lda * sz
                    unit || unsafe_store!(xp, userc ? unsafe_load(xp, j) * rcp[j] :
                                              unsafe_load(xp, j) / unsafe_load(cp + (j - 1) * sz), j)
                    _axpy_simd!(n - j, -unsafe_load(xp, j), cp + j * sz, xp + j * sz)
                end
            end
        else                                     # solve Aᵀ·x = b  (column dot)
            if up                                # U,T: forward, j ascending
                @inbounds for j in 1:n
                    cp = Ap + (j - 1) * lda * sz
                    t = unsafe_load(xp, j) - _dot_simd(j - 1, cp, xp, T)
                    unit || (t = userc ? t * rcp[j] : t / unsafe_load(cp + (j - 1) * sz))
                    unsafe_store!(xp, t, j)
                end
            else                                 # L,T: back, j descending
                @inbounds for j in n:-1:1
                    cp = Ap + (j - 1) * lda * sz
                    t = unsafe_load(xp, j) - _dot_simd(n - j, cp + j * sz, xp + j * sz, T)
                    unit || (t = userc ? t * rcp[j] : t / unsafe_load(cp + (j - 1) * sz))
                    unsafe_store!(xp, t, j)
                end
            end
        end
    end
    return x
end

# triangular block size for blocked trmv/trsv: the NB×NB diagonal block is walked per-column while the
# off-diagonal goes through gemv, so the diagonal block must stay L1-resident (NB²·8 ≤ L1 ⇒ NB ≤ √(L1/8)).
# req#8: DERIVED as an L1-residency CLAMP. NOT a flat crossover — the fleet A/B (trmv/trsv N+T, boost-locked)
# shows NB=128 REGRESSES mid-n 3–8% on Zen4 (128²·8 = 128 KB SPILLS the 32 KB L1) while NB=32/64 tie ≤2%;
# so the block is capped at √(L1/8) and shrunk further on a smaller L1. Fleet L1 = 32 KB → √4096 = 64 EXACT
# (no-op vs the old literal); a ≤16 KB-L1 box gets a smaller, still-resident block. `tri_nb` pref pins it.
const _TRI_NB = @load_preference("tri_nb", clamp(_round_dn(isqrt(_L1_BYTES ÷ 8), 16), 16, 64))::Int
# REAL trsv-T only (complex has its own `_TRI_C_T_UNB` below): unblocked (scalar back/forward-substitution
# dots) up to here, blocked above. req#8: the blocked path offloads the bulk O(n²) off-diagonal work to the
# vectorized gemv-T kernel and only substitutes the NB×NB diagonal per-column, so it wins once n is large
# enough to amortize the gemv-T calls. FLEET A/B (boost-locked, same-process interleaved): the crossover
# MOVES with ISA — AVX2 Zen4 ties at ~512 / Zen3 at ~640 (unblocked wins ≤3% at n≤512, blocked wins 6–40%
# at n≥768), and AVX-512 Zen5 blocks even at n=128 (blk wins 1–6% ≤512). 512 = the fleet MAX crossover:
# it keeps the AVX2 boxes' small-n unblocked optimum (where they'd lose ~3% blocked) while Zen5 still gates
# unblocked ≤512 (no miss, ~1–6% left on the table). Spread is within a size-step ⇒ literal, not a W-formula
# (a formula would inject unmeasured µarch variation). The old 1024 was MISTUNED: it forced the slower
# unblocked path at n=768/1024, and at n=1024 unblocked is a gate MISS on both AVX2 boxes (Zen4 1.03×, Zen3
# 1.09× OB) that blocking FIXES (→0.95/0.87×). `tri_t_unb` pref pins it.
const _TRI_T_UNB = @load_preference("tri_t_unb", 512)::Int
#                          trmv-T blocks at _TRI_NB (its unblocked L-form dips mid-n); N forms at _TRI_NB.

# Reciprocal-of-diagonal scratch for REAL trsv (per type; single-thread — no MT here). Same mechanism the
# complex path already ships (see _TRSV_RCP64 / _crecip below): the in-loop scalar DIVIDE sits on the
# sequential substitution CRITICAL PATH — x[j] feeds the next column's axpy/dot — so its ~13.5-cyc Zen4
# latency is fully exposed, ×n columns, while a multiply is ~4. Precompute r[j]=1/diag up front (all
# independent ⇒ pipelined, throughput-bound), then MULTIPLY in the loop. The complex finding that motivates
# this reads: "ztrsv n≤256 was 0.65-0.96 vs OB; ztrmv, SAME axpy but NO divide, gates" — and the real
# Zen4/AOCL split has the same shape (trmv 0.95 worst, trsv 0.88, identical blocking and identical axpy).
# Naive reciprocal, no overflow guard: reference `dtrsm` itself does `temp = one/a(k,k)` then multiplies, so
# the one extra rounding is BLAS-conformant practice, not a liberty.
# Size DERIVED, not a literal: `_trsv_blk!` hands the per-column kernel at most max(_TRI_NB, _TRI_T_UNB)
# columns (unblocked dispatch at :1695, diagonal blocks ≤ _TRI_NB elsewhere), and it is the only caller.
# The `n <= length` guard below keeps a pinned-larger `tri_t_unb` correct rather than out-of-bounds.
const _TRSV_RRCP_N = max(_TRI_NB, _TRI_T_UNB)
const _TRSV_RRCP64 = Vector{Float64}(undef, _TRSV_RRCP_N)
const _TRSV_RRCP32 = Vector{Float32}(undef, _TRSV_RRCP_N)
@inline _trsv_rrcpbuf(::Type{Float64}) = _TRSV_RRCP64
@inline _trsv_rrcpbuf(::Type{Float32}) = _TRSV_RRCP32
# COMPLEX tri unblocked threshold. The blocked off-diagonal scatter goes through the complex gemv; on
# AVX-512 its per-call/shuffle overhead made per-column faster ≤1024. On AVX2 the scatter now uses the
# fast OB-structure ri gemv (see _tri_scat_cmplx!), so blocking wins earlier — the unblocked column-axpy
# re-streams x and dips at n=1024–2048 (0.83–0.86); route those to blocked+ri. n≤512 stays unblocked
# (gates 0.96–1.53). Sweep the crossover per box via the knob.
const _TRI_C_BLK_MIN = @load_preference("tri_c_blk_min", _vwidth(Float64) == 4 ? 256 : 1024)::Int
# COMPLEX trsv-T unblocked threshold — SEPARATE from real `_TRI_T_UNB` (which the real-only fleet A/B retuned
# to 512). Kept at 1024 (the pre-retune value) so the complex path is byte-unchanged: complex ztrsv-T stays
# unblocked ≤1024 as before. A complex-specific crossover sweep (the AVX2 n=1024–2048 dip noted above) is
# DEFERRED to the parked complex batch; decoupling avoids silently retuning a gated complex op off a real A/B.
const _TRI_C_T_UNB = @load_preference("tri_c_t_unb", 1024)::Int

# Blocked trmv/trsv (real dense): the per-column kernel re-streams x from memory at large n. Block it
# — each diagonal NB×NB block uses the per-column kernel (cache-resident), and the off-diagonal block
# uses the already-fast gemv (reads A once, no re-stream). Processing order keeps x_J unmodified when
# block I needs it. trmv: diagonal-then-gemv(+); trsv: gemv(−)-then-solve. Sub-blocks are contiguous
# views (unit-stride → SIMD gemv via the relaxed _l2_simd_ok).
# Force the column-panel gemv-N (β=1 accumulate) for the tall off-diagonal scatter `y += α·Av·xv`.
# Going through the dispatcher would pick the row-block path (n=NB cols → NB strided streams), which
# thrashes because a sub-block's columns are a full parent-lda apart; so we keep calling a panel
# driver DIRECTLY (also dodging the ~200 ns/call kwarg layer).
# But pick the SAME panel driver the dispatcher would (`_gemv_n_simd!`, minus the row-block branch we
# must avoid): `_gemv_n_paneldrv_minner!` in the mid-n/L3 regime, the old NP=8 panel beyond it. Two
# reasons, and the second is the one that matters here:
#   (a) minner measured ~8–10% faster PB-self at n=512–2048 on Zen4;
#   (b) `_gemvn_minner_np` is **lda-aliasing-aware** — at a way-stride lda it drops the stream count to
#       `min(_L1D_ASSOC, _GEMVN_NP_WIDE)`, whereas the old path's bare `_GEMV_NP = 8` is not. The
#       scatter's A-view inherits the PARENT's lda, and every size in the trmv/trsv gate rows is a
#       power of two, so the aliasing branch is exactly the live case.
# This is why `gemvN` gates while `trmv`/`trsv` missed at the SAME sizes: gemvN got minner, this
# scatter never did. No new tuning constant — `_GEMVN_MINNER` (Derive: datapath-gated, OFF on
# native-512 Zen5) and `_GEMVN_MINNER_MAXA` are reused verbatim from the dispatcher.
# Always the plain NP=8 panel driver — NOT the dispatcher, and NOT minner.
#   • not the dispatcher: at n = NB = 64 columns it would take the row-block path, measured 49.8 GB/s
#     at m=4096 against the panel's 63.5 (−22%), because a sub-block's columns are a full parent-lda
#     apart. That is the original reason this call is direct.
#   • not minner: minner wins on SQUARE mid-n (which is why the dispatcher prefers it there), but the
#     scatter is TALL-SKINNY and it loses at that shape. Measured GB/s, n=64 cols, β=1 accumulate:
#         m=       1024   2048   4096   8192
#       paneldrv  77.58  69.92  63.49  63.29   ← ships
#       minner    74.87  67.78  60.95  62.95
#     Consistent 3–4% for paneldrv at every height. Routing this call through minner (briefly done on
#     a "small PB-self gain" reading of a whole-op sweep) is therefore a small REGRESSION here, and the
#     scatter is 56–58% of blocked trsv's runtime, so it is not free. Shape matters more than the
#     residency window: measure the kernel at the shape the CALLER issues, not at a square one.
@inline _tri_scat!(yv, Av, xv, α) = _gemv_n_paneldrv!(size(Av, 1), size(Av, 2), α, Av, xv, yv, one(α), Val(false))
# T-form off-diagonal: gemv-T kernel directly (no backend kwarg layer — ~200 ns/call dominated the
# few off-diagonal calls at mid n). y_I += α·Avᵀ·xv  (β=1 accumulate).
@inline _tri_scatT!(yv, Av, xv, α) = _gemv_t_simd!(size(Av, 1), size(Av, 2), α, Av, xv, one(α), yv, Val(false))

# ── real trmv: the n at which `_trmv_simd!` hands over (to `_trmv_fused8!` for N, blocked for T) ─────
# DERIVE tier (cache residency), and BYTE-IDENTICAL to the predicate it replaces — see the block comment
# inside `_trmv_blk!` for the A/B that fixed the criterion at HALF L2:
#     n <= NB || 2·n²·sizeof(T) <= _L2_BYTES   ⟺   n <= max(_TRI_NB, isqrt(_L2_BYTES ÷ (2·sizeof(T))))
# (n integer ⇒ n² <= L2/(2s) ⟺ n² <= L2 ÷ (2s) ⟺ n <= isqrt(L2 ÷ (2s)); exact, no float.)
# Fleet: L2 = 1 MiB ⇒ 257 for Float64 / 363 for Float32 (Zen4 wintermute, Zen5 neuromancer);
#        L2 = 512 KiB ⇒ 182 / 257 (Zen3 galen). `_TRI_NB` (=64 fleet-wide) is the floor.
#
# WHY IT IS A KNOB AND NOT JUST THE FORMULA: the A/B that validated this crossover (f552f13, 2026-07-31)
# PREDATES `_trmv_fused8!` (447c46a, 2026-08-01). Below the threshold the loser was the OLD blocked
# diagonal+tall-scatter structure, which no longer exists on the N path — so the n ≤ threshold side has
# never been re-litigated against the fused kernel. `PUREBLAS_FORCE_trmv_fused_min=0` runs fused8 at every
# n, a huge value runs `_trmv_simd!` at every n, and the crossover can be swept without editing source.
const _TRMV_FUSED_MIN_PREF = @load_preference("trmv_fused_min", nothing)
# Resolved at RUNTIME so the force hook can reach it (a const is baked at load/precompile and cannot be
# forced), but ONCE PER PROCESS, never per call: `_force_knob` reads ENV, and an ENV dictionary lookup in
# a BLAS-2 hot path is itself a regression — the gemv-T m-unroll instrument cost Zen5 gemvT n=64
# 0.959 -> 0.767 with the value unchanged (see `_GEMVT_U_ONCE` above, whose shape this mirrors exactly,
# including being reachable from `trmv!`'s all-paths @noalloc proof). Pin wins over force (PDM: P first).
const _TRMV_FUSED_MIN_ONCE = Base.OncePerProcess{Int}() do
    something(_TRMV_FUSED_MIN_PREF, _force_knob("trmv_fused_min"))
end
# < 0 (the unset sentinel from `_force_knob`) ⇒ the derived default.
@inline function _trmv_fused_min(::Type{T}) where {T}
    f = _TRMV_FUSED_MIN_ONCE()
    f >= 0 && return f
    return max(_TRI_NB, isqrt(_L2_BYTES ÷ (2 * sizeof(T)))) + 1
end

@inline function _trmv_blk!(up::Bool, tr::Bool, unit::Bool, n::Int, A, x)
    NB = _TRI_NB
    # Block only once the triangle outgrows L2. Blocking exists to stop the per-column kernel
    # re-streaming x from memory; while A is L2-resident that re-stream is served from L2 and costs
    # little, so the blocking overhead (a gemv call and its bookkeeping per block) is not repaid.
    # The triangle must fit L2 with HEADROOM (hence 2·A <= L2, not A <= L2): x, the output and the
    # streaming share that cache, so a triangle that exactly fills L2 already thrashes it.
    # `A <= L2` alone was fitted on Zen4 and REGRESSED Zen3, where n=256 gives a triangle of exactly
    # 512 KB = its entire L2. Interleaved A/B, 5 fresh processes per arm, alternating arms, separate
    # precompile, variant verified present/absent every round. Ratio vs AOCL (min..max over 5):
    #             ORIGINAL          HALF-L2           verdict
    #   Zen4 128  1.010..1.017      1.169..1.176      BETTER, disjoint
    #   Zen4 256  0.950..0.958      1.021..1.034      BETTER, disjoint   <- was the failing cell
    #   Zen3 128  1.080..1.091      1.115..1.120      BETTER, disjoint
    #   Zen3 256  1.003..1.051      1.011..1.037      overlap  (A <= L2 made this WORSE, disjoint)
    #   both 512..4096                                overlap  (criterion cannot reach them)
    # Sizes the criterion cannot touch overlap on both boxes - the controls are part of the result.
    # DERIVE tier: cache residency over a detected const, validated on both microarchitectures (req#8b).
    # NOT shared with trsv, deliberately: the same sweep gives trsv 1.009 / 0.986 / 0.960 / 0.967, i.e.
    # blocking pays from n≈192 there, because trsv's per-column path carries the serial substitution
    # dependency and is latency-bound, so offloading the off-diagonal to gemv repays much earlier.
    # Two routines, one shape, different crossovers — do not unify them.
    # Same predicate, hoisted into `_trmv_fused_min` so it is forceable — see the note above it.
    n < _trmv_fused_min(eltype(A)) && return _trmv_simd!(up, tr, unit, n, A, x)
    # N forms: the fused F=8 panel sweep (see `_trmv_fused8!`) replaces the blocked
    # diagonal + tall-scatter structure. Requires unit-stride columns for the vector loads.
    if !tr && eltype(A) <: BlasReal && _strided1(A) && x isa StridedVector && stride(x, 1) == 1
        return _trmv_fused8!(up, unit, n, A, x)
    end
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

# FUSED f-column trmv for the N forms (BLIS `trmv_unf_var2` / `axpyf` shape), F fixed at 8.
#
# Same panel structure as `_trsv_fused8!` below, minus the divide: per panel of F columns, the F
# ORIGINAL x values are read into scalars first (the triangle overwrites them), then ONE fused pass
# over the off-panel rows does F fmas per vector chunk, then the F×F triangle runs in registers.
#
# WHY: the per-column sweep this replaces re-reads AND re-writes the whole x prefix once per column
# — n²/2 loads + n²/2 stores of x against n²/2 loads of A, i.e. 3 streams per A element. F=8 cuts
# the x traffic 8×, to 1.25 streams. That is precisely the fusing factor BLIS's `axpyf` exists for.
# Ordering note: for U,N the triangle must run j ASCENDING with its `x[j] = t·A[j,j]` store in the
# loop, so later columns of the same panel accumulate onto it; L,N mirrors with j descending.
#
# PANEL ALIGNMENT IS THE MIRROR OF trsv's, AND GETTING IT BACKWARDS COSTS THE WHOLE WIN. trmv's
# off-panel update GROWS with panel index (U,N updates rows 1:lo-1), where trsv's SHRINKS. Putting
# the ragged n-mod-8 panel last — the natural loop shape, and what trsv does — would drop the single
# LARGEST update onto a per-column fallback. Anchoring the remainder in the FIRST panel instead puts
# it exactly where the off-panel work is zero, so every panel that does off-panel work is full and
# the ragged fallback disappears entirely (hence no `_axpy_simd!` branch here, unlike `_trsv_fused8!`).
@inline function _trmv_fused8!(up::Bool, unit::Bool, n::Int, A, x)
    T = eltype(A); W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T)
    GC.@preserve A x begin
        Ap = pointer(A); xp = pointer(x); lda = stride(A, 2)
        r = n & 7; r = r == 0 ? 8 : r             # remainder goes in the FIRST panel — see note above
        if up                                     # U,N: panels forward from column 1
            lo = 1
            @inbounds while lo <= n
                hi = lo == 1 ? r : lo + 7; top = lo - 1
                if top > 0                        # fused pass over rows 1:lo-1, originals hoisted
                    c0 = Ap + (lo - 1) * lda * sz; c1 = c0 + lda * sz; c2 = c1 + lda * sz
                    c3 = c2 + lda * sz; c4 = c3 + lda * sz; c5 = c4 + lda * sz
                    c6 = c5 + lda * sz; c7 = c6 + lda * sz
                    b0 = V(unsafe_load(xp, lo));     b1 = V(unsafe_load(xp, lo + 1))
                    b2 = V(unsafe_load(xp, lo + 2)); b3 = V(unsafe_load(xp, lo + 3))
                    b4 = V(unsafe_load(xp, lo + 4)); b5 = V(unsafe_load(xp, lo + 5))
                    b6 = V(unsafe_load(xp, lo + 6)); b7 = V(unsafe_load(xp, lo + 7))
                    i = 0
                    while i + W <= top
                        o = i * sz; acc = vload(V, xp + o)
                        acc = muladd(b0, vload(V, c0 + o), acc); acc = muladd(b1, vload(V, c1 + o), acc)
                        acc = muladd(b2, vload(V, c2 + o), acc); acc = muladd(b3, vload(V, c3 + o), acc)
                        acc = muladd(b4, vload(V, c4 + o), acc); acc = muladd(b5, vload(V, c5 + o), acc)
                        acc = muladd(b6, vload(V, c6 + o), acc); acc = muladd(b7, vload(V, c7 + o), acc)
                        vstore(acc, xp + o); i += W
                    end
                    while i < top
                        s = unsafe_load(xp, i + 1)
                        for k in lo:hi
                            s += unsafe_load(Ap + ((k - 1) * lda + i) * sz) * unsafe_load(xp, k)
                        end
                        unsafe_store!(xp, s, i + 1); i += 1
                    end
                end
                for j in lo:hi                    # F×F triangle, in registers
                    cp = Ap + (j - 1) * lda * sz; t = unsafe_load(xp, j)
                    for i in lo:(j - 1)
                        unsafe_store!(xp, unsafe_load(xp, i) + t * unsafe_load(cp, i), i)
                    end
                    unit || unsafe_store!(xp, t * unsafe_load(cp + (j - 1) * sz), j)
                end
                lo = hi + 1
            end
        else                                      # L,N: panels backward from column n
            hi = n
            @inbounds while hi > 0
                lo = hi == n ? n - r + 1 : hi - 7; rest = n - hi
                if rest > 0
                    c0 = Ap + ((lo - 1) * lda + hi) * sz; c1 = c0 + lda * sz; c2 = c1 + lda * sz
                    c3 = c2 + lda * sz; c4 = c3 + lda * sz; c5 = c4 + lda * sz
                    c6 = c5 + lda * sz; c7 = c6 + lda * sz
                    b0 = V(unsafe_load(xp, lo));     b1 = V(unsafe_load(xp, lo + 1))
                    b2 = V(unsafe_load(xp, lo + 2)); b3 = V(unsafe_load(xp, lo + 3))
                    b4 = V(unsafe_load(xp, lo + 4)); b5 = V(unsafe_load(xp, lo + 5))
                    b6 = V(unsafe_load(xp, lo + 6)); b7 = V(unsafe_load(xp, lo + 7))
                    yp = xp + hi * sz
                    i = 0
                    while i + W <= rest
                        o = i * sz; acc = vload(V, yp + o)
                        acc = muladd(b0, vload(V, c0 + o), acc); acc = muladd(b1, vload(V, c1 + o), acc)
                        acc = muladd(b2, vload(V, c2 + o), acc); acc = muladd(b3, vload(V, c3 + o), acc)
                        acc = muladd(b4, vload(V, c4 + o), acc); acc = muladd(b5, vload(V, c5 + o), acc)
                        acc = muladd(b6, vload(V, c6 + o), acc); acc = muladd(b7, vload(V, c7 + o), acc)
                        vstore(acc, yp + o); i += W
                    end
                    while i < rest
                        s = unsafe_load(yp, i + 1)
                        for k in lo:hi
                            s += unsafe_load(Ap + ((k - 1) * lda + hi + i) * sz) * unsafe_load(xp, k)
                        end
                        unsafe_store!(yp, s, i + 1); i += 1
                    end
                end
                for j in hi:-1:lo
                    cp = Ap + (j - 1) * lda * sz; t = unsafe_load(xp, j)
                    for i in (j + 1):hi
                        unsafe_store!(xp, unsafe_load(xp, i) + t * unsafe_load(cp, i), i)
                    end
                    unit || unsafe_store!(xp, t * unsafe_load(cp + (j - 1) * sz), j)
                end
                hi = lo - 1
            end
        end
    end
    return x
end

# FUSED f-column trsv for the N forms (BLIS `trsv_unf_var2` shape), F fixed at 8.
#
# Per panel of F columns: (1) an F×F scalar triangle — dependent chain only F long, not n; (2) ONE
# fused pass over the remaining rows doing F fmas per vector chunk, with the F solved values HOISTED
# into F broadcast registers before the loop. Zero kernel calls in the whole routine.
#
# WHY THIS AND NOT THE BLOCKED PATH. Decomposing blocked trsv showed the off-diagonal scatter is
# 75% (n=512) to 93% (n=4096) of runtime — but a vendor head-to-head at the scatter's own shape has
# our gemv-N BEATING both AOCL and OpenBLAS (77.9/72.5/69.0 GB/s at m=1024). Working the arithmetic
# back through the split: at n=512 our scatter ALONE costs 0.773× AOCL's entire trsv. The deficit was
# never the scatter — it was the 64-long dependent chain in the diagonal solve, which this removes.
# AOCL is BLIS, and BLIS never cache-blocks level-2: its exported symbols are `bli_dtrsv_unf_var1/2`
# over fused `axpyf`/`dotxf`, with no blocked variant at all. This is that structure.
#
# Measured vs the blocked path it replaces (Zen4, freq-locked, upper/N/non-unit F64):
#     n=      256    512   1024   2048   4096
#   ratio    1.01   1.06   1.14   1.07   1.00
#
# THREE EARLIER ATTEMPTS AT THIS REGION FAILED, and the reasons are the design constraints here:
#   • sub-blocking with hand-written update loops — 0.47-0.53×
#   • sub-blocking calling `_axpy_simd!` per column — 0.59-0.66×
#     Both issued ~2n kernel calls where the shipped sweep issues n; entry cost ate the shorter chain.
#     THE FUSED FORM ISSUES ZERO CALLS — that is the whole point, not an optimisation detail.
#   • the same fused loop with the F broadcasts left INSIDE the chunk loop — 0.54-0.62×. Hoisting
#     them into registers took it to 0.71/0.84/1.06 at n=64/128/256. Runtime-bounded inner loops do
#     not get unrolled, so the broadcasts must be written out explicitly.
# Numerics: reassociated vs the per-column sweep; residual-checked, not bitwise.
@inline function _trsv_fused8!(up::Bool, unit::Bool, n::Int, A, x)
    T = eltype(A); W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T)
    GC.@preserve A x begin
        Ap = pointer(A); xp = pointer(x); lda = stride(A, 2)
        if up                                     # U,N: panels backward from column n
            hi = n
            @inbounds while hi > 0
                lo = max(1, hi - 7)
                for j in hi:-1:lo                 # F×F triangle: chain is F, not n
                    cp = Ap + (j - 1) * lda * sz
                    xj = unit ? unsafe_load(xp, j) :
                         unsafe_load(xp, j) / unsafe_load(cp + (j - 1) * sz)
                    unsafe_store!(xp, xj, j)
                    for i in lo:(j - 1)
                        unsafe_store!(xp, unsafe_load(xp, i) - xj * unsafe_load(cp, i), i)
                    end
                end
                top = lo - 1
                if top > 0
                    if hi - lo + 1 == 8           # full panel: fused, broadcasts hoisted
                        c0 = Ap + (lo - 1) * lda * sz; c1 = c0 + lda * sz; c2 = c1 + lda * sz
                        c3 = c2 + lda * sz; c4 = c3 + lda * sz; c5 = c4 + lda * sz
                        c6 = c5 + lda * sz; c7 = c6 + lda * sz
                        b0 = V(-unsafe_load(xp, lo));     b1 = V(-unsafe_load(xp, lo + 1))
                        b2 = V(-unsafe_load(xp, lo + 2)); b3 = V(-unsafe_load(xp, lo + 3))
                        b4 = V(-unsafe_load(xp, lo + 4)); b5 = V(-unsafe_load(xp, lo + 5))
                        b6 = V(-unsafe_load(xp, lo + 6)); b7 = V(-unsafe_load(xp, lo + 7))
                        i = 0
                        while i + W <= top
                            o = i * sz; acc = vload(V, xp + o)
                            acc = muladd(b0, vload(V, c0 + o), acc); acc = muladd(b1, vload(V, c1 + o), acc)
                            acc = muladd(b2, vload(V, c2 + o), acc); acc = muladd(b3, vload(V, c3 + o), acc)
                            acc = muladd(b4, vload(V, c4 + o), acc); acc = muladd(b5, vload(V, c5 + o), acc)
                            acc = muladd(b6, vload(V, c6 + o), acc); acc = muladd(b7, vload(V, c7 + o), acc)
                            vstore(acc, xp + o); i += W
                        end
                        @inbounds while i < top
                            s = unsafe_load(xp, i + 1)
                            for k in lo:hi
                                s -= unsafe_load(Ap + ((k - 1) * lda + i) * sz) * unsafe_load(xp, k)
                            end
                            unsafe_store!(xp, s, i + 1); i += 1
                        end
                    else                          # ragged final panel
                        @inbounds for k in lo:hi
                            _axpy_simd!(top, -unsafe_load(xp, k), Ap + (k - 1) * lda * sz, xp)
                        end
                    end
                end
                hi = lo - 1
            end
        else                                      # L,N: panels forward from column 1
            lo = 1
            @inbounds while lo <= n
                hi = min(n, lo + 7)
                for j in lo:hi
                    cp = Ap + (j - 1) * lda * sz
                    xj = unit ? unsafe_load(xp, j) :
                         unsafe_load(xp, j) / unsafe_load(cp + (j - 1) * sz)
                    unsafe_store!(xp, xj, j)
                    for i in (j + 1):hi
                        unsafe_store!(xp, unsafe_load(xp, i) - xj * unsafe_load(cp, i), i)
                    end
                end
                rest = n - hi
                if rest > 0
                    if hi - lo + 1 == 8
                        c0 = Ap + ((lo - 1) * lda + hi) * sz; c1 = c0 + lda * sz; c2 = c1 + lda * sz
                        c3 = c2 + lda * sz; c4 = c3 + lda * sz; c5 = c4 + lda * sz
                        c6 = c5 + lda * sz; c7 = c6 + lda * sz
                        b0 = V(-unsafe_load(xp, lo));     b1 = V(-unsafe_load(xp, lo + 1))
                        b2 = V(-unsafe_load(xp, lo + 2)); b3 = V(-unsafe_load(xp, lo + 3))
                        b4 = V(-unsafe_load(xp, lo + 4)); b5 = V(-unsafe_load(xp, lo + 5))
                        b6 = V(-unsafe_load(xp, lo + 6)); b7 = V(-unsafe_load(xp, lo + 7))
                        yp = xp + hi * sz
                        i = 0
                        while i + W <= rest
                            o = i * sz; acc = vload(V, yp + o)
                            acc = muladd(b0, vload(V, c0 + o), acc); acc = muladd(b1, vload(V, c1 + o), acc)
                            acc = muladd(b2, vload(V, c2 + o), acc); acc = muladd(b3, vload(V, c3 + o), acc)
                            acc = muladd(b4, vload(V, c4 + o), acc); acc = muladd(b5, vload(V, c5 + o), acc)
                            acc = muladd(b6, vload(V, c6 + o), acc); acc = muladd(b7, vload(V, c7 + o), acc)
                            vstore(acc, yp + o); i += W
                        end
                        @inbounds while i < rest
                            s = unsafe_load(yp, i + 1)
                            for k in lo:hi
                                s -= unsafe_load(Ap + ((k - 1) * lda + hi + i) * sz) * unsafe_load(xp, k)
                            end
                            unsafe_store!(yp, s, i + 1); i += 1
                        end
                    else
                        @inbounds for k in lo:hi
                            _axpy_simd!(rest, -unsafe_load(xp, k),
                                        Ap + ((k - 1) * lda + hi) * sz, xp + hi * sz)
                        end
                    end
                end
                lo = hi + 1
            end
        end
    end
    return x
end

@inline function _trsv_blk!(up::Bool, tr::Bool, unit::Bool, n::Int, A, x)
    NB = _TRI_NB
    # trsv-T (forward/back substitution by dots): unblocked wins only at small n (n≤_TRI_T_UNB); above
    # that, blocking offloads the O(n²) off-diagonal to gemv-T and wins (fleet-measured). See _TRI_T_UNB.
    (n <= NB || (tr && n <= _TRI_T_UNB)) && return _trsv_simd!(up, tr, unit, n, A, x)
    # N forms: the fused F=8 panel sweep (see `_trsv_fused8!`) replaces the blocked
    # diagonal-solve + tall-scatter structure entirely. Measured vs the blocked path it supersedes,
    # Zen4 upper/N/non-unit F64: n=256 1.01×, 512 1.06×, 1024 1.14×, 2048 1.07×, 4096 1.00×.
    # Requires unit-stride columns for the vector loads; anything else keeps the blocked path below.
    if !tr && eltype(A) <: BlasReal && _strided1(A) && x isa StridedVector && stride(x, 1) == 1
        return _trsv_fused8!(up, unit, n, A, x)
    end
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
@inline _dot_cmplx_disp(L::Int, ap, xp, ::Type{T}, cj::Bool) where {T <: BlasReal} =
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
                @inbounds for j in 1:n
                    rcp[j] = _crecip(unsafe_load(Ap, (j - 1) * ldc + j))
                end
            else
                @inbounds for j in 1:n
                    rcp[j] = _crecip(djj(j))
                end
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
# The OB-structure ri gemv (α folded, fresh accs, prefetch, m-blocked) beats the row-tile scatter on
# BOTH ISAs for the tall off-diagonal shape (m≫k=NB, β=1) — measured 0.71–0.96× row-tile across m on
# AVX-512, and it's the same kernel zgemvN already rides. (Was AVX-512→row-tile; that predated the ri tune.)
# `_gemv_n_ri_ship!`, NOT `_gemv_n_ri_cmplx!`: trmv!/trsv! carry a @noalloc contract and the latter
# consults a OncePerProcess whose init allocates — an all-paths proof fails on it even though this
# operand (m×NB, tall-skinny) can never reach the large-square tier that knob selects for.
@inline _tri_scat_cmplx!(yv, Av, xv, α::T) where {T} =
    _gemv_n_ri_ship!(size(Av, 1), size(Av, 2), α, Av, xv, yv, one(T), Val(false))
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
    (n <= _TRI_C_BLK_MIN || (tr && n <= _TRI_C_T_UNB)) && return _trsv_cmplx!(up, tr, cj, unit, n, A, x)
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
                for i in 1:(j - 1)
                    _st!(x, sx + (i - 1) * incx, _ld(x, sx + (i - 1) * incx) + xj * A[i, j])
                end
                unit || _st!(x, sx + (j - 1) * incx, xj * A[j, j])
            end
        else
            @inbounds for j in n:-1:1
                xj = _ld(x, sx + (j - 1) * incx)
                for i in n:-1:(j + 1)
                    _st!(x, sx + (i - 1) * incx, _ld(x, sx + (i - 1) * incx) + xj * A[i, j])
                end
                unit || _st!(x, sx + (j - 1) * incx, xj * A[j, j])
            end
        end
    else                                         # x := op(A)ᵀ·x
        if up
            @inbounds for j in n:-1:1
                s = unit ? _ld(x, sx + (j - 1) * incx) : _ld(x, sx + (j - 1) * incx) * (cj ? conj(A[j, j]) : A[j, j])
                for i in 1:(j - 1)
                    s += el(i, j) * _ld(x, sx + (i - 1) * incx)
                end
                _st!(x, sx + (j - 1) * incx, s)
            end
        else
            @inbounds for j in 1:n
                s = unit ? _ld(x, sx + (j - 1) * incx) : _ld(x, sx + (j - 1) * incx) * (cj ? conj(A[j, j]) : A[j, j])
                for i in (j + 1):n
                    s += el(i, j) * _ld(x, sx + (i - 1) * incx)
                end
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
                for i in 1:(j - 1)
                    _st!(x, sx + (i - 1) * incx, _ld(x, sx + (i - 1) * incx) - xj * A[i, j])
                end
            end
        else
            @inbounds for j in 1:n
                unit || _st!(x, sx + (j - 1) * incx, _ld(x, sx + (j - 1) * incx) / A[j, j])
                xj = _ld(x, sx + (j - 1) * incx)
                for i in (j + 1):n
                    _st!(x, sx + (i - 1) * incx, _ld(x, sx + (i - 1) * incx) - xj * A[i, j])
                end
            end
        end
    else                                         # solve op(A)ᵀ·x = b
        if up
            @inbounds for j in 1:n
                s = _ld(x, sx + (j - 1) * incx)
                for i in 1:(j - 1)
                    s -= el(i, j) * _ld(x, sx + (i - 1) * incx)
                end
                unit || (s /= (cj ? conj(A[j, j]) : A[j, j]))
                _st!(x, sx + (j - 1) * incx, s)
            end
        else
            @inbounds for j in n:-1:1
                s = _ld(x, sx + (j - 1) * incx)
                for i in (j + 1):n
                    s -= el(i, j) * _ld(x, sx + (i - 1) * incx)
                end
                unit || (s /= (cj ? conj(A[j, j]) : A[j, j]))
                _st!(x, sx + (j - 1) * incx, s)
            end
        end
    end
    return x
end
