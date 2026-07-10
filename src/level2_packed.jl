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

# ── Packed symv PANEL kernels (dense-symv fused n+t panel ported to packed via per-column base pointers).
# Amortizes the x/y re-stream over NB columns — the dense `_symv_col!` per-column path re-reads the x/y
# prefix EVERY column (n·n/2 extra traffic → the large-n decline). BLASFEO's dsymv is this same fused
# panel. `bc` = NB per-column base Ptrs with bc[c] + (panel-local row i)·sz = A[row, col_c]; that identity
# is what makes packed's variable column spacing address-uniform in the row (mirrors `_symv_*panel!`).
const _SPMV_PANEL = @load_preference("spmv_panel", false)::Bool   # default OFF pending fleet A/B
# The panel wins only once x+y spill L1 (the re-stream then hits L2/L3 — measured Zen4 PB-self: n=2048
# +21%, n=4096 +42%, but n=512 −18% where the single-column path's lower setup wins with x/y L1-resident).
# So size-gate: panel above, per-column below. Crossover ≈ x+y = L1; tune on the locked fleet.
const _SPMV_PANEL_MINXY = @load_preference("spmv_panel_minxy", _L1_BYTES)::Int  # min x+y bytes (2·n·sizeof) for the panel

function _spmv_offblk_packed_expr(W, V, sz, NB, K, masked)
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
            ap = masked ? :(vload($V, bc[$c] + (i + $((v - 1) * W)) * $sz, $(Symbol(:k, v)))) :
                :(vload($V, bc[$c] + (i + $((v - 1) * W)) * $sz))
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

# LOWER panel: NB×NB masked diagonal block (rows 0:NB-1) then off-block below (rows NB:M-1).
@generated function _spmv_lpanel!(M::Int, α::T, bc::NTuple{NB, Ptr{T}}, xp::Ptr{T}, yp::Ptr{T},
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
        push!(body.args, :(acd = vload($V, bc[$c], mblk)))
        push!(body.args, :(yblk = vifelse(lanes >= $(c - 1), muladd($(Symbol(:xj, c)), acd, yblk), yblk)))
        push!(body.args, :($(Symbol(:d, c)) = muladd(vifelse(lanes > $(c - 1), acd, zv), xblk, $(Symbol(:d, c)))))
    end
    push!(body.args, :(vstore(yblk, yp, mblk)))
    push!(body.args, :(i = $NB; nfull = M - rem(M - $NB, $mr); while i < nfull; $(_spmv_offblk_packed_expr(W, V, sz, NB, MR, false)); i += $mr; end))
    branches = _spmv_offblk_packed_expr(W, V, sz, NB, 1, true)
    for k in 2:MR; branches = Expr(:if, :(rmn > $((k - 1) * W)), _spmv_offblk_packed_expr(W, V, sz, NB, k, true), branches); end
    push!(body.args, :(if i < M; rmn = M - i; $branches; end))
    for c in 1:NB; push!(body.args, :(unsafe_store!(yp, muladd(α, sum($(Symbol(:d, c))), unsafe_load(yp, $c)), $c))); end
    push!(body.args, :(return nothing))
    return body
end

# UPPER panel: off-block rows 0:dboff-1 (dboff=M-NB) then NB×NB masked diagonal block at rows dboff:M-1.
@generated function _spmv_upanel!(M::Int, α::T, bc::NTuple{NB, Ptr{T}}, xp::Ptr{T}, yp::Ptr{T},
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
    push!(body.args, :(i = 0; nfull = dboff - rem(dboff, $mr); while i < nfull; $(_spmv_offblk_packed_expr(W, V, sz, NB, MR, false)); i += $mr; end))
    branches = _spmv_offblk_packed_expr(W, V, sz, NB, 1, true)
    for k in 2:MR; branches = Expr(:if, :(rmn > $((k - 1) * W)), _spmv_offblk_packed_expr(W, V, sz, NB, k, true), branches); end
    push!(body.args, :(if i < dboff; rmn = dboff - i; $branches; end))
    push!(body.args, :(mblk = lanes < $NB))
    push!(body.args, :(yblk = vload($V, yp + dboff * $sz, mblk)))
    push!(body.args, :(xblk = vload($V, xp + dboff * $sz, mblk)))
    for c in 1:NB
        push!(body.args, :(acd = vload($V, bc[$c] + dboff * $sz, mblk)))
        push!(body.args, :(yblk = vifelse(lanes <= $(c - 1), muladd($(Symbol(:xj, c)), acd, yblk), yblk)))
        push!(body.args, :($(Symbol(:d, c)) = muladd(vifelse(lanes < $(c - 1), acd, zv), xblk, $(Symbol(:d, c)))))
    end
    push!(body.args, :(vstore(yblk, yp + dboff * $sz, mblk)))
    for c in 1:NB; push!(body.args, :(unsafe_store!(yp, muladd(α, sum($(Symbol(:d, c))), unsafe_load(yp, dboff + $c)), dboff + $c))); end
    push!(body.args, :(return nothing))
    return body
end

# Panel driver: full NB-column panels via the fused kernels, per-column bases from _pkU/_pkL; partial last
# panel falls back to the per-column path (same as _spmv_simd!). bc[c] + row·sz = A[row, panel-col c].
@inline function _spmv_panel_driver!(up::Bool, n::Int, α::T, AP, x, y) where {T<:BlasReal}
    NB = min(_SYMV_NB, _vwidth(T))
    GC.@preserve AP x y begin
        base = pointer(AP); xp = pointer(x); yp = pointer(y); sz = sizeof(T)
        jb = 0
        while jb + NB <= n
            if up
                bc = ntuple(c -> base + _pkU(jb + c) * sz, Val(NB))                 # A[i,jb+c] (1-based i≤jb+c) = base+(_pkU(jb+c)+i-1); 0-based row → +i
                _spmv_upanel!(jb + NB, α, bc, xp, yp, Val(_SYMV_MR), Val(NB))
            else
                bc = ntuple(c -> base + (_pkL(jb + c, n) + 1 - c) * sz, Val(NB))     # A[jb+i+1,jb+c] = base+(_pkL(jb+c,n)+1-c)+i
                _spmv_lpanel!(n - jb, α, bc, xp + jb * sz, yp + jb * sz, Val(_SYMV_MR), Val(NB))
            end
            jb += NB
        end
        @inbounds while jb < n                                                       # partial last panel (per-column)
            axj = α * unsafe_load(xp, jb + 1)
            if up
                cp = base + _pkU(jb + 1) * sz; ajj = unsafe_load(cp + jb * sz)
                s = _symv_col!(jb, axj, cp, xp, yp)
            else
                cp = base + _pkL(jb + 1, n) * sz; ajj = unsafe_load(cp)
                s = _symv_col!(n - 1 - jb, axj, cp + sz, xp + (jb + 1) * sz, yp + (jb + 1) * sz)
            end
            unsafe_store!(yp, unsafe_load(yp, jb + 1) + axj * ajj + α * s, jb + 1)
            jb += 1
        end
    end
    return y
end

function _spmv!(up::Bool, n::Integer, α::Number, AP, x, incx::Integer, β::Number, y, incy::Integer)
    _scale_y!(Int(n), β, y, incy)
    iszero(α) && return y
    if _pk2_simd_ok(AP, x, y, incx, incy)
        T = eltype(AP)                                        # panel only where x+y spill L1 (large-n flatten); else per-column
        return (_SPMV_PANEL && 2 * Int(n) * sizeof(T) >= _SPMV_PANEL_MINXY) ?
            _spmv_panel_driver!(up, Int(n), convert(T, α), AP, x, y) :
            _spmv_simd!(up, Int(n), convert(T, α), AP, x, y)
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
# Complex Hermitian packed mv: packed analog of _hemv_cmplx! — the off-diagonal of column j is a CONTIGUOUS
# packed run, so run the fused two-sided _hemv_col_cmplx! over it + real-diagonal term. β·y pre-scaled.
@inline function _hpmv_cmplx_simd!(up::Bool, n::Int, α::T, AP, x, y) where {T<:BlasComplex}
    Tr = real(T)
    GC.@preserve AP x y begin
        Ap = Ptr{Tr}(pointer(AP)); xp = Ptr{Tr}(pointer(x)); yp = Ptr{Tr}(pointer(y))
        xpc = pointer(x); ypc = pointer(y); szr = sizeof(Tr)
        @inbounds for j in 1:n
            tmp = α * unsafe_load(xpc, j); sr = zero(Tr); si = zero(Tr)
            base = up ? _pkU(j) : _pkL(j, n)                      # 0-based complex index before column j
            if up
                L = j - 1
                L > 0 && ((sr, si) = _hemv_col_cmplx!(L, real(tmp), imag(tmp), Ap + (base * 2) * szr, xp, yp))
                ajj = unsafe_load(Ap, (base + j - 1) * 2 + 1)     # real(A[j,j]) = AP[base+j]
            else
                L = n - j
                L > 0 && ((sr, si) = _hemv_col_cmplx!(L, real(tmp), imag(tmp), Ap + (base * 2 + 2) * szr, xp + (j * 2) * szr, yp + (j * 2) * szr))
                ajj = unsafe_load(Ap, base * 2 + 1)               # real(A[j,j]) = AP[base+1]
            end
            unsafe_store!(ypc, unsafe_load(ypc, j) + tmp * ajj + α * Complex{Tr}(sr, si), j)
        end
    end
    return y
end
function _hpmv!(up::Bool, n::Integer, α::Number, AP, x, incx::Integer, β::Number, y, incy::Integer)
    if incx == 1 && incy == 1 && eltype(AP) <: BlasComplex && eltype(x) === eltype(AP) && eltype(y) === eltype(AP) &&
            AP isa StridedVector && x isa StridedVector && y isa StridedVector
        _scale_y!(Int(n), convert(eltype(AP), β), y, 1); iszero(α) && return y
        return _hpmv_cmplx_simd!(up, Int(n), convert(eltype(AP), α), AP, x, y)
    end
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

# ── spr / spr2 (symmetric packed rank-1/2, real + AD) · hpr / hpr2 (Hermitian packed, complex) ───
# Rank updates on packed triangular storage. Real spr/spr2 reuse the contiguous packed-column axpy
# (`_axpy_simd!`); the Hermitian hpr/hpr2 use the generic scalar path (complex SIMD deferred) and force
# the diagonal real. Convention matches the packed mat-vecs: A[i,j] at `_pkU(j)+i` (up) / `_pkL(j,n)+(i-j)+1`.

# spr:  A := α·x·xᵀ + A
@inline function _spr_simd!(up::Bool, n::Int, α::T, AP, x) where {T<:BlasReal}
    sz = sizeof(T)
    GC.@preserve AP x begin
        Ap = pointer(AP); xp = pointer(x)
        if up
            @inbounds for j in 1:n
                xj = unsafe_load(xp, j)
                iszero(xj) || _axpy_simd!(j, α * xj, xp, Ap + _pkU(j) * sz)
            end
        else
            @inbounds for j in 1:n
                xj = unsafe_load(xp, j)
                iszero(xj) || _axpy_simd!(n - j + 1, α * xj, xp + (j - 1) * sz, Ap + _pkL(j, n) * sz)
            end
        end
    end
    return AP
end
function _spr!(up::Bool, n::Integer, α::Number, x, incx::Integer, AP)
    iszero(α) && return AP
    _pk_simd_ok(AP, x, incx) && return _spr_simd!(up, Int(n), convert(eltype(AP), α), AP, x)
    n = Int(n); sx = _start(n, incx)
    @inbounds for j in 1:n
        xj = _ld(x, sx + (j - 1) * incx)
        if !iszero(xj)
            tmp = α * xj; base = up ? _pkU(j) : _pkL(j, n)
            for i in (up ? (1:j) : (j:n))
                k = up ? base + i : base + (i - j) + 1
                _st!(AP, k, _ld(AP, k) + _ld(x, sx + (i - 1) * incx) * tmp)
            end
        end
    end
    return AP
end

# spr2:  A := α·x·yᵀ + α·y·xᵀ + A
@inline function _spr2_simd!(up::Bool, n::Int, α::T, AP, x, y) where {T<:BlasReal}
    sz = sizeof(T)
    GC.@preserve AP x y begin
        Ap = pointer(AP); xp = pointer(x); yp = pointer(y)
        @inbounds for j in 1:n
            xj = unsafe_load(xp, j); yj = unsafe_load(yp, j)
            (iszero(xj) && iszero(yj)) && continue
            if up
                cp = Ap + _pkU(j) * sz
                _axpy_simd!(j, α * yj, xp, cp); _axpy_simd!(j, α * xj, yp, cp)
            else
                off = (j - 1) * sz; cp = Ap + _pkL(j, n) * sz; len = n - j + 1
                _axpy_simd!(len, α * yj, xp + off, cp); _axpy_simd!(len, α * xj, yp + off, cp)
            end
        end
    end
    return AP
end
function _spr2!(up::Bool, n::Integer, α::Number, x, incx::Integer, y, incy::Integer, AP)
    iszero(α) && return AP
    _pk2_simd_ok(AP, x, y, incx, incy) && return _spr2_simd!(up, Int(n), convert(eltype(AP), α), AP, x, y)
    n = Int(n); sx = _start(n, incx); sy = _start(n, incy)
    @inbounds for j in 1:n
        xj = _ld(x, sx + (j - 1) * incx); yj = _ld(y, sy + (j - 1) * incy)
        if !(iszero(xj) && iszero(yj))
            t1 = α * yj; t2 = α * xj; base = up ? _pkU(j) : _pkL(j, n)
            for i in (up ? (1:j) : (j:n))
                k = up ? base + i : base + (i - j) + 1
                _st!(AP, k, _ld(AP, k) + _ld(x, sx + (i - 1) * incx) * t1 + _ld(y, sy + (i - 1) * incy) * t2)
            end
        end
    end
    return AP
end

# hpr:  A := α·x·xᴴ + A  (α real, A Hermitian; diagonal forced real)
function _hpr!(up::Bool, n::Integer, α::Number, x, incx::Integer, AP)
    n = Int(n); sx = _start(n, incx); a = real(α)
    @inbounds for j in 1:n
        xj = _ld(x, sx + (j - 1) * incx)
        base = up ? _pkU(j) : _pkL(j, n); kd = up ? base + j : base + 1
        if !iszero(xj)
            tmp = a * conj(xj)
            up && for i in 1:j-1
                k = base + i; _st!(AP, k, _ld(AP, k) + _ld(x, sx + (i - 1) * incx) * tmp)
            end
            _st!(AP, kd, real(_ld(AP, kd)) + real(xj * tmp))
            !up && for i in j+1:n
                k = base + (i - j) + 1; _st!(AP, k, _ld(AP, k) + _ld(x, sx + (i - 1) * incx) * tmp)
            end
        else
            _st!(AP, kd, real(_ld(AP, kd)))
        end
    end
    return AP
end

# hpr2:  A := α·x·yᴴ + ᾱ·y·xᴴ + A  (A Hermitian; diagonal forced real)
function _hpr2!(up::Bool, n::Integer, α::Number, x, incx::Integer, y, incy::Integer, AP)
    iszero(α) && return AP
    n = Int(n); sx = _start(n, incx); sy = _start(n, incy)
    @inbounds for j in 1:n
        xj = _ld(x, sx + (j - 1) * incx); yj = _ld(y, sy + (j - 1) * incy)
        base = up ? _pkU(j) : _pkL(j, n); kd = up ? base + j : base + 1
        if !(iszero(xj) && iszero(yj))
            t1 = α * conj(yj); t2 = conj(α * xj)
            up && for i in 1:j-1
                k = base + i
                _st!(AP, k, _ld(AP, k) + _ld(x, sx + (i - 1) * incx) * t1 + _ld(y, sy + (i - 1) * incy) * t2)
            end
            _st!(AP, kd, real(_ld(AP, kd)) + real(xj * t1 + yj * t2))
            !up && for i in j+1:n
                k = base + (i - j) + 1
                _st!(AP, k, _ld(AP, k) + _ld(x, sx + (i - 1) * incx) * t1 + _ld(y, sy + (i - 1) * incy) * t2)
            end
        else
            _st!(AP, kd, real(_ld(AP, kd)))
        end
    end
    return AP
end
