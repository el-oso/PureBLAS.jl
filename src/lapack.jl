using LinearAlgebra: PosDefException

# LAPACK-level routines built on the gated Level-3 BLAS. First: Cholesky (potrf).
# Real symmetric positive-definite A = L·Lᵀ (uplo='L') or A = Uᵀ·U (uplo='U'), factored in place into
# the `uplo` triangle. Right-looking BLOCKED algorithm: each NB diagonal block is factored by the
# unblocked `_potf2` base, then the gated trsm (panel solve) + syrk (trailing rank-NB update) carry the
# bulk. Generic over T<:Real (the unblocked base + the generic trsm/syrk path make it ForwardDiff-
# traceable); BlasReal hits the SIMD trsm/syrk. ponytail: NB hand-set for Zen4; lift to a knob if tuning.

const _POTRF_BASE = 512    # recurse above this; below, the unblocked base (potf2, vectorized inner loop).
# Measured sweet spot on Zen4: smaller bases pay more small-k trsm/syrk overhead, larger pay a
# memory-bound unblocked panel. ponytail: hand-set; revisit when tuning Cholesky to the gate.

# Contiguous scratch for the diagonal base block: the recursion's base is a view(A, js, js) whose
# columns are parent_ld apart (poor locality, the memory-bound potf2). Copying it to a contiguous
# buffer, factoring there, and copying back streams contiguous memory (better prefetch/TLB).
# _potf2_buf (potrf diagonal-base contiguous buffer) is the per-type L3Workspace `potf2` field
# (see src/workspace.jl).

# Unblocked right-looking Cholesky of an n×n block, lower triangle. Throws PosDefException at the first
# non-positive pivot (LAPACK's info>0). Reads/writes only the lower triangle.
function _potf2_lower!(A, n::Int)
    @inbounds for j in 1:n
        d = A[j, j]
        d > 0 || throw(PosDefException(j))
        ajj = sqrt(d); A[j, j] = ajj; invd = inv(ajj)
        for i in (j + 1):n; A[i, j] *= invd; end                 # scale column j below the diagonal
        for k in (j + 1):n                                        # rank-1 update of the lower trailing
            akj = A[k, j]
            for i in k:n; A[i, k] -= A[i, j] * akj; end
        end
    end
    return A
end
# Unblocked, upper triangle: A = Uᵀ·U.
function _potf2_upper!(A, n::Int)
    @inbounds for j in 1:n
        d = A[j, j]
        d > 0 || throw(PosDefException(j))
        ajj = sqrt(d); A[j, j] = ajj; invd = inv(ajj)
        for i in (j + 1):n; A[j, i] *= invd; end                 # scale row j right of the diagonal
        for k in (j + 1):n                                        # rank-1 update of the upper trailing
            ajk = A[j, k]
            for i in (j + 1):k; A[i, k] -= A[j, i] * ajk; end
        end
    end
    return A
end

# Recursive (cache-oblivious) Cholesky. Lower: split 2×2 — factor A11, solve the off-diagonal panel
# A21·L11⁻ᵀ (trsm), downdate A22 -= A21·A21ᵀ (syrk), recurse A22. The top-level trsm/syrk are large-k
# (half-matrix → the gated packed L3 paths); only the ≤_POTRF_BASE diagonal base is scalar potf2.
# Factor a base block, via a contiguous buffer when A is a strided sub-block (better locality).
@inline function _potf2b_lower!(A, n::Int)
    if n >= 128 && A isa SubArray && stride(A, 2) > n
        buf = _potf2_buf(eltype(A), n); copyto!(buf, A); _potf2_lower!(buf, n); copyto!(A, buf); return A
    end
    return _potf2_lower!(A, n)
end
@inline function _potf2b_upper!(A, n::Int)
    if n >= 128 && A isa SubArray && stride(A, 2) > n
        buf = _potf2_buf(eltype(A), n); copyto!(buf, A); _potf2_upper!(buf, n); copyto!(A, buf); return A
    end
    return _potf2_upper!(A, n)
end

function _potrf_lower!(A, n::Int, base::Int = _POTRF_BASE)
    n <= base && return _potf2b_lower!(A, n)
    h = n ÷ 2
    _potrf_lower!(view(A, 1:h, 1:h), h, base)
    A21 = view(A, (h + 1):n, 1:h)
    trsm!(A21, view(A, 1:h, 1:h); side = 'R', uplo = 'L', transA = 'T', diag = 'N', alpha = true)
    syrk!(view(A, (h + 1):n, (h + 1):n), A21; uplo = 'L', trans = 'N', alpha = -1, beta = 1)
    _potrf_lower!(view(A, (h + 1):n, (h + 1):n), n - h, base)
    return A
end
# Upper: A = Uᵀ·U. Off-diagonal panel A12 = U11⁻ᵀ·A12 (trsm side-L), downdate A22 -= A12ᵀ·A12 (syrk).
function _potrf_upper!(A, n::Int)
    n <= _POTRF_BASE && return _potf2b_upper!(A, n)
    h = n ÷ 2
    _potrf_upper!(view(A, 1:h, 1:h), h)
    A12 = view(A, 1:h, (h + 1):n)
    trsm!(A12, view(A, 1:h, 1:h); side = 'L', uplo = 'U', transA = 'T', diag = 'N', alpha = true)
    syrk!(view(A, (h + 1):n, (h + 1):n), A12; uplo = 'U', trans = 'T', alpha = -1, beta = 1)
    _potrf_upper!(view(A, (h + 1):n, (h + 1):n), n - h)
    return A
end

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# Float64 LOWER fast path — faithful port of faer 0.24.1 `cholesky_recursion_right_looking`
# (github.com/el-oso/BlazingPorts.jl, src/Factorizations.jl). Custom register-blocked SIMD kernels
# (left-looking base, fused trsm, fused syrk) — that fusion beats the gate where the generic recursion
# above paid the general trsm!/syrk! overhead at small Cholesky block sizes. Overwrites the lower
# triangle with L; the upper is scratch (computed into never-read memory). Float64-only; everything
# else (Float32/complex/Dual/upper) stays on the generic AD-traceable path. Returns false on a
# non-positive pivot. SIMD via PureBLAS's SIMD.jl layer at the detected width.
const _CVF = Vec{_vwidth(Float64), Float64}     # vector type at host width (concrete const)
const _CHOLW = _vwidth(Float64)
const _CHOL_THRESHOLD = 64                        # faer LdltParams::auto
const _CHOL_BLOCK = 128
const _CHOL_NB = 4                                # trsm panel column block
const _CHOL_NC = 4                                # syrk column block
@inline _clidx(i, k, ld) = (k - 1) * ld + i                              # 1-based linear index
@inline _cvptr(p, i, k, ld) = p + (((k - 1) * ld + (i - 1)) * sizeof(Float64))   # byte Ptr to [i,k]

# base case (n ≤ threshold): left-looking SIMD panel Cholesky, ascending-k FMA, scale by 1/√diag.
function _chol_base_f64!(p::Ptr{Float64}, n::Int, ld::Int)
    @inbounds for j in 1:n
        i = j
        while i + _CHOLW - 1 <= n
            base = _cvptr(p, i, j, ld); acc = vload(_CVF, base)
            for k in 1:j-1
                acc = muladd(_CVF(-unsafe_load(p, _clidx(j, k, ld))), vload(_CVF, _cvptr(p, i, k, ld)), acc)
            end
            vstore(acc, base); i += _CHOLW
        end
        while i <= n
            s = unsafe_load(p, _clidx(i, j, ld))
            for k in 1:j-1
                s = muladd(-unsafe_load(p, _clidx(j, k, ld)), unsafe_load(p, _clidx(i, k, ld)), s)
            end
            unsafe_store!(p, s, _clidx(i, j, ld)); i += 1
        end
        d = unsafe_load(p, _clidx(j, j, ld))
        (d > 0.0) || return false
        invd = 1.0 / sqrt(d); vinv = _CVF(invd)
        i = j
        while i + _CHOLW - 1 <= n
            base = _cvptr(p, i, j, ld); vstore(vload(_CVF, base) * vinv, base); i += _CHOLW
        end
        while i <= n
            unsafe_store!(p, unsafe_load(p, _clidx(i, j, ld)) * invd, _clidx(i, j, ld)); i += 1
        end
    end
    return true
end

# panel solve column cc: A10[:,cc] -= Σ_{k<c0} L00[cc,k]·A10[:,k]  (remainder / non-full-NB path).
@inline function _trsm_gemm_col_f64!(p00, p10, cc::Int, c0::Int, m::Int, ld::Int)
    i = 1
    @inbounds while i + _CHOLW - 1 <= m
        o = _cvptr(p10, i, cc, ld); a = vload(_CVF, o)
        for k in 1:c0-1
            a = muladd(_CVF(-unsafe_load(p00, _clidx(cc, k, ld))), vload(_CVF, _cvptr(p10, i, k, ld)), a)
        end
        vstore(a, o); i += _CHOLW
    end
    @inbounds while i <= m
        s = unsafe_load(p10, _clidx(i, cc, ld))
        for k in 1:c0-1
            s = muladd(-unsafe_load(p00, _clidx(cc, k, ld)), unsafe_load(p10, _clidx(i, k, ld)), s)
        end
        unsafe_store!(p10, s, _clidx(i, cc, ld)); i += 1
    end
end

# panel solve: L10 (m×bs) from L10·L00ᵀ = A10, in place on A10. Register-blocked in NB-column panels.
function _trsm_right_lower_f64!(p00::Ptr{Float64}, p10::Ptr{Float64}, bs::Int, m::Int, ld::Int)
    c0 = 1
    @inbounds while c0 <= bs
        nb = min(_CHOL_NB, bs - c0 + 1)
        if c0 > 1
            if nb == _CHOL_NB
                i = 1
                while i + _CHOLW - 1 <= m
                    o0 = _cvptr(p10, i, c0, ld);     a0 = vload(_CVF, o0)
                    o1 = _cvptr(p10, i, c0 + 1, ld); a1 = vload(_CVF, o1)
                    o2 = _cvptr(p10, i, c0 + 2, ld); a2 = vload(_CVF, o2)
                    o3 = _cvptr(p10, i, c0 + 3, ld); a3 = vload(_CVF, o3)
                    for k in 1:c0-1
                        vk = vload(_CVF, _cvptr(p10, i, k, ld))
                        a0 = muladd(_CVF(-unsafe_load(p00, _clidx(c0, k, ld))), vk, a0)
                        a1 = muladd(_CVF(-unsafe_load(p00, _clidx(c0 + 1, k, ld))), vk, a1)
                        a2 = muladd(_CVF(-unsafe_load(p00, _clidx(c0 + 2, k, ld))), vk, a2)
                        a3 = muladd(_CVF(-unsafe_load(p00, _clidx(c0 + 3, k, ld))), vk, a3)
                    end
                    vstore(a0, o0); vstore(a1, o1); vstore(a2, o2); vstore(a3, o3); i += _CHOLW
                end
                while i <= m
                    for dj in 0:_CHOL_NB-1
                        cc = c0 + dj; s = unsafe_load(p10, _clidx(i, cc, ld))
                        for k in 1:c0-1
                            s = muladd(-unsafe_load(p00, _clidx(cc, k, ld)), unsafe_load(p10, _clidx(i, k, ld)), s)
                        end
                        unsafe_store!(p10, s, _clidx(i, cc, ld))
                    end
                    i += 1
                end
            else
                for dj in 0:nb-1
                    _trsm_gemm_col_f64!(p00, p10, c0 + dj, c0, m, ld)
                end
            end
        end
        for dj in 0:nb-1                                  # within-panel triangular solve + scale
            c = c0 + dj; invc = 1.0 / unsafe_load(p00, _clidx(c, c, ld)); vinv = _CVF(invc)
            i = 1
            while i + _CHOLW - 1 <= m
                o = _cvptr(p10, i, c, ld); a = vload(_CVF, o)
                for k in c0:c-1
                    a = muladd(_CVF(-unsafe_load(p00, _clidx(c, k, ld))), vload(_CVF, _cvptr(p10, i, k, ld)), a)
                end
                vstore(a * vinv, o); i += _CHOLW
            end
            while i <= m
                s = unsafe_load(p10, _clidx(i, c, ld))
                for k in c0:c-1
                    s = muladd(-unsafe_load(p00, _clidx(c, k, ld)), unsafe_load(p10, _clidx(i, k, ld)), s)
                end
                unsafe_store!(p10, s * invc, _clidx(i, c, ld)); i += 1
            end
        end
        c0 += _CHOL_NB
    end
    return nothing
end

# one trailing column j: A11[i,j] -= Σ_c L10[j,c]·L10[i,c]  (remainder / <NC-column path).
@inline function _syrk_panel_f64!(p11, p10, j::Int, m::Int, bs::Int, ld::Int)
    i = ((j - 1) ÷ _CHOLW) * _CHOLW + 1
    @inbounds while i + _CHOLW - 1 <= m
        b = _cvptr(p11, i, j, ld); a = vload(_CVF, b)
        for c in 1:bs
            a = muladd(_CVF(-unsafe_load(p10, _clidx(j, c, ld))), vload(_CVF, _cvptr(p10, i, c, ld)), a)
        end
        vstore(a, b); i += _CHOLW
    end
    @inbounds while i <= m
        s = unsafe_load(p11, _clidx(i, j, ld))
        for c in 1:bs
            s = muladd(-unsafe_load(p10, _clidx(j, c, ld)), unsafe_load(p10, _clidx(i, c, ld)), s)
        end
        unsafe_store!(p11, s, _clidx(i, j, ld)); i += 1
    end
end

# trailing symmetric rank-bs update A11 (m×m) −= L10·L10ᵀ. Register-blocked MR rows × NC cols.
function _syrk_lower_f64!(p11::Ptr{Float64}, p10::Ptr{Float64}, m::Int, bs::Int, ld::Int)
    j = 1
    @inbounds while j + _CHOL_NC - 1 <= m
        i = ((j - 1) ÷ _CHOLW) * _CHOLW + 1                # W-aligned triangular start (skip upper blocks)
        while i + 3_CHOLW - 1 <= m                          # MR=3 × NC=4 = 12 accumulators
            r1 = i + _CHOLW; r2 = i + 2_CHOLW
            e00 = _cvptr(p11, i, j, ld);      A00 = vload(_CVF, e00)
            e10 = _cvptr(p11, r1, j, ld);     C00 = vload(_CVF, e10)
            e20 = _cvptr(p11, r2, j, ld);     D00 = vload(_CVF, e20)
            e01 = _cvptr(p11, i, j + 1, ld);  A01 = vload(_CVF, e01)
            e11 = _cvptr(p11, r1, j + 1, ld); C01 = vload(_CVF, e11)
            e21 = _cvptr(p11, r2, j + 1, ld); D01 = vload(_CVF, e21)
            e02 = _cvptr(p11, i, j + 2, ld);  A02 = vload(_CVF, e02)
            e12 = _cvptr(p11, r1, j + 2, ld); C02 = vload(_CVF, e12)
            e22 = _cvptr(p11, r2, j + 2, ld); D02 = vload(_CVF, e22)
            e03 = _cvptr(p11, i, j + 3, ld);  A03 = vload(_CVF, e03)
            e13 = _cvptr(p11, r1, j + 3, ld); C03 = vload(_CVF, e13)
            e23 = _cvptr(p11, r2, j + 3, ld); D03 = vload(_CVF, e23)
            for c in 1:bs
                v0 = vload(_CVF, _cvptr(p10, i, c, ld)); v1 = vload(_CVF, _cvptr(p10, r1, c, ld)); v2 = vload(_CVF, _cvptr(p10, r2, c, ld))
                g0 = _CVF(-unsafe_load(p10, _clidx(j, c, ld)));     A00 = muladd(g0, v0, A00); C00 = muladd(g0, v1, C00); D00 = muladd(g0, v2, D00)
                g1 = _CVF(-unsafe_load(p10, _clidx(j + 1, c, ld))); A01 = muladd(g1, v0, A01); C01 = muladd(g1, v1, C01); D01 = muladd(g1, v2, D01)
                g2 = _CVF(-unsafe_load(p10, _clidx(j + 2, c, ld))); A02 = muladd(g2, v0, A02); C02 = muladd(g2, v1, C02); D02 = muladd(g2, v2, D02)
                g3 = _CVF(-unsafe_load(p10, _clidx(j + 3, c, ld))); A03 = muladd(g3, v0, A03); C03 = muladd(g3, v1, C03); D03 = muladd(g3, v2, D03)
            end
            vstore(A00, e00); vstore(A01, e01); vstore(A02, e02); vstore(A03, e03)
            vstore(C00, e10); vstore(C01, e11); vstore(C02, e12); vstore(C03, e13)
            vstore(D00, e20); vstore(D01, e21); vstore(D02, e22); vstore(D03, e23)
            i += 3_CHOLW
        end
        while i + 2_CHOLW - 1 <= m                          # MR=2 × NC=4 = 8 accumulators
            r1 = i + _CHOLW
            d00 = _cvptr(p11, i, j, ld);     A00 = vload(_CVF, d00)
            d10 = _cvptr(p11, r1, j, ld);    B00 = vload(_CVF, d10)
            d01 = _cvptr(p11, i, j + 1, ld); A01 = vload(_CVF, d01)
            d11 = _cvptr(p11, r1, j + 1, ld); B01 = vload(_CVF, d11)
            d02 = _cvptr(p11, i, j + 2, ld); A02 = vload(_CVF, d02)
            d12 = _cvptr(p11, r1, j + 2, ld); B02 = vload(_CVF, d12)
            d03 = _cvptr(p11, i, j + 3, ld); A03 = vload(_CVF, d03)
            d13 = _cvptr(p11, r1, j + 3, ld); B03 = vload(_CVF, d13)
            for c in 1:bs
                v0 = vload(_CVF, _cvptr(p10, i, c, ld)); v1 = vload(_CVF, _cvptr(p10, r1, c, ld))
                g0 = _CVF(-unsafe_load(p10, _clidx(j, c, ld)));     A00 = muladd(g0, v0, A00); B00 = muladd(g0, v1, B00)
                g1 = _CVF(-unsafe_load(p10, _clidx(j + 1, c, ld))); A01 = muladd(g1, v0, A01); B01 = muladd(g1, v1, B01)
                g2 = _CVF(-unsafe_load(p10, _clidx(j + 2, c, ld))); A02 = muladd(g2, v0, A02); B02 = muladd(g2, v1, B02)
                g3 = _CVF(-unsafe_load(p10, _clidx(j + 3, c, ld))); A03 = muladd(g3, v0, A03); B03 = muladd(g3, v1, B03)
            end
            vstore(A00, d00); vstore(A01, d01); vstore(A02, d02); vstore(A03, d03)
            vstore(B00, d10); vstore(B01, d11); vstore(B02, d12); vstore(B03, d13)
            i += 2_CHOLW
        end
        while i + _CHOLW - 1 <= m
            b0 = _cvptr(p11, i, j, ld);     a0 = vload(_CVF, b0)
            b1 = _cvptr(p11, i, j + 1, ld); a1 = vload(_CVF, b1)
            b2 = _cvptr(p11, i, j + 2, ld); a2 = vload(_CVF, b2)
            b3 = _cvptr(p11, i, j + 3, ld); a3 = vload(_CVF, b3)
            for c in 1:bs
                lic = vload(_CVF, _cvptr(p10, i, c, ld))
                a0 = muladd(_CVF(-unsafe_load(p10, _clidx(j, c, ld))), lic, a0)
                a1 = muladd(_CVF(-unsafe_load(p10, _clidx(j + 1, c, ld))), lic, a1)
                a2 = muladd(_CVF(-unsafe_load(p10, _clidx(j + 2, c, ld))), lic, a2)
                a3 = muladd(_CVF(-unsafe_load(p10, _clidx(j + 3, c, ld))), lic, a3)
            end
            vstore(a0, b0); vstore(a1, b1); vstore(a2, b2); vstore(a3, b3); i += _CHOLW
        end
        while i <= m
            for dj in 0:_CHOL_NC-1
                s = unsafe_load(p11, _clidx(i, j + dj, ld))
                for c in 1:bs
                    s = muladd(-unsafe_load(p10, _clidx(j + dj, c, ld)), unsafe_load(p10, _clidx(i, c, ld)), s)
                end
                unsafe_store!(p11, s, _clidx(i, j + dj, ld))
            end
            i += 1
        end
        j += _CHOL_NC
    end
    while j <= m
        _syrk_panel_f64!(p11, p10, j, m, bs, ld); j += 1
    end
    return nothing
end

# right-looking recursive driver (faer cholesky_recursion_right_looking).
function _chol_rl_f64!(p::Ptr{Float64}, n::Int, ld::Int, block_size::Int, threshold::Int)
    n <= threshold && return _chol_base_f64!(p, n, ld)
    bs_outer = min(nextpow(2, n) ÷ 2, block_size)
    j = 0
    while j < n
        bs = min(bs_outer, n - j)
        _chol_rl_f64!(_cvptr(p, j + 1, j + 1, ld), bs, ld, block_size, threshold) || return false
        m = n - j - bs
        if m > 0
            p10 = _cvptr(p, j + bs + 1, j + 1, ld); p11 = _cvptr(p, j + bs + 1, j + bs + 1, ld)
            _trsm_right_lower_f64!(_cvptr(p, j + 1, j + 1, ld), p10, bs, m, ld)
            _syrk_lower_f64!(p11, p10, m, bs, ld)
        end
        j += bs
    end
    return true
end

# A power-of-two leading dimension aliases columns into the same cache sets (the LDA=2^k conflict,
# ~1.3–1.5× slower at n≥512). When A's stride is a po2, factor in a padded (ld+8) scratch and copy
# back — bit-identical, ld is pure addressing. Reusable buffer (single-thread; project defers MT).
const _CHOL_PAD = Ref(Matrix{Float64}(undef, 0, 0))
# ≤ this → faer kernels; above → halve, routing the O(n³) trailing update through the cache-blocked
# gating syrk!/trsm!. AVX-512 (32 regs) runs the faer syrk to 1024 where it still wins; AVX2 (16 regs)
# has no cache-blocked faer syrk so it fades by n≈256 — drop the base to 128 so large-n rides gating
# syrk! (measured: n=1024 0.70→0.87, n=2048 0.85→0.91 on Zen3). ponytail: per-ISA knob.
const _CHOL_FAER_BASE = _CHOLW == 8 ? 1024 : 128
# Pad when columns alias L1 sets: Zen4 L1 = 64 sets × 64 B, so stride·8 a multiple of 64·64=4096 B
# (stride % 512 == 0) maps every column to the same sets. Covers po2 ≥512 AND 1536, 2560, … —
# faer's plain ispow2 missed the non-po2 multiples. WIDENED to %256 (half-period, 2 sets/column):
# lda=256 measured a 1.77× penalty in the faer right-looking kernel (287µs → 162µs at ld=264).
@inline _chol_needs_pad(A, n) = n >= 128 && stride(A, 2) % 256 == 0

# Hybrid driver: the faer kernels are fastest at small/medium n but their syrk isn't cache-blocked, so
# they fade at large n (panel re-streamed). Recurse by halving — the big off-diagonal blocks go through
# PureBLAS's cache-blocked trsm!/syrk! (which gate at large k) — and bottom out in the faer kernels.
function _chol_hyb_f64!(M, n::Int, base::Int)
    if n <= base
        ok = GC.@preserve M _chol_rl_f64!(pointer(M), n, stride(M, 2), _CHOL_BLOCK, _CHOL_THRESHOLD)
        ok || throw(PosDefException(1))   # ponytail: faer returns Bool; exact pivot column not threaded
        return M
    end
    h = n ÷ 2
    _chol_hyb_f64!(view(M, 1:h, 1:h), h, base)
    A21 = view(M, (h + 1):n, 1:h)
    trsm!(A21, view(M, 1:h, 1:h); side = 'R', uplo = 'L', transA = 'T', diag = 'N', alpha = true)
    syrk!(view(M, (h + 1):n, (h + 1):n), A21; uplo = 'L', trans = 'N', alpha = -1, beta = 1)
    _chol_hyb_f64!(view(M, (h + 1):n, (h + 1):n), n - h, base)
    return M
end

function _potrf_f64_lower!(A::StridedMatrix{Float64}, base::Int = _CHOL_FAER_BASE)
    n = size(A, 1)
    n == 0 && return A
    if _chol_needs_pad(A, n)                      # factor in a non-conflicting (ld = n+8) scratch, copy back
        R = n + 8
        b = _CHOL_PAD[]
        (size(b, 1) < R || size(b, 2) < n) && (b = _CHOL_PAD[] = Matrix{Float64}(undef, R, n))
        Mw = view(b, 1:n, 1:n)
        # explicit contiguous per-column copies — copyto! on SubArrays is elementwise (the LU pad lesson)
        lda = stride(A, 2); ldb = size(b, 1)
        GC.@preserve A b begin
            pa = pointer(A); pb = pointer(b)
            @inbounds for j in 0:(n - 1)
                unsafe_copyto!(pb + j * ldb * 8, pa + j * lda * 8, n)
            end
            _chol_hyb_f64!(Mw, n, base)
            @inbounds for j in 0:(n - 1)
                unsafe_copyto!(pa + j * lda * 8, pb + j * ldb * 8, n)
            end
        end
    else
        _chol_hyb_f64!(A, n, base)
    end
    return A
end

# Public: Cholesky factor A in place into its `uplo` triangle. Returns A. Throws PosDefException if A is
# not positive definite. Float64 lower → faer fast path; else the generic AD-traceable recursion.
function potrf!(A::AbstractMatrix; uplo::Char = 'L')
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("potrf!: A must be square"))
    if uplo == 'L' && A isa StridedMatrix{Float64} && stride(A, 1) == 1
        return _potrf_f64_lower!(A)
    end
    uplo == 'L' ? _potrf_lower!(A, n) : _potrf_upper!(A, n)
    return A
end
