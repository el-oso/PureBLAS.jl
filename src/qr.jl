# LAPACK QR (geqrf) — Householder, no pivoting. Port of faer 0.24.1's unblocked panel reduction
# (el-oso/BlazingPorts.jl `src/Factorizations.jl`) onto PureBLAS's SIMD.jl layer, driven by a blocked
# compact-WY (LAPACK dlarft/dlarfb) loop whose trailing update is PureBLAS's gated `gemm!` — so faer's
# bespoke packed BLIS gemm is unneeded (we have one). faer convention: H_k = I − v_k v_kᵀ/τ_k, τ_k=Inf ⇒
# identity; on output the upper triangle of A is R, the essential v_k (implicit v_k[k]=1) sit below the
# diagonal of column k, τ[k] the coefficient. Float64-only (the proven-fast path); reuses the Cholesky
# SIMD helpers (_CVF/_CHOLW/_clidx/_cvptr from lapack.jl). ponytail: generic/AD QR deferred.

# Compute the Householder reflector of column `col` in place (faer convention: H=I−τ·v·vᵀ, v[col]=1
# implicit + essential below, R diagonal = β on output). Returns tinv (=τ), or Inf for a trivial reflector.
@inline function _qr_reflect_f64!(p::Ptr{Float64}, m::Int, col::Int, ld::Int)
    tn = 0.0; i = col + 1
    @inbounds while i + _CHOLW - 1 <= m; x = vload(_CVF, _cvptr(p, i, col, ld)); tn += sum(x * x); i += _CHOLW; end
    @inbounds while i <= m; x = unsafe_load(p, _clidx(i, col, ld)); tn = muladd(x, x, tn); i += 1; end
    tail = sqrt(tn); @inbounds head = unsafe_load(p, _clidx(col, col, ld))
    tail < floatmin(Float64) && return Inf
    nrm = hypot(abs(head), tail); sn = head >= 0.0 ? nrm : -nrm; hwb = head + sn; hi = 1.0 / hwb; vi = _CVF(hi)
    i = col + 1
    @inbounds while i + _CHOLW - 1 <= m; b = _cvptr(p, i, col, ld); vstore(vload(_CVF, b) * vi, b); i += _CHOLW; end
    @inbounds while i <= m; unsafe_store!(p, unsafe_load(p, _clidx(i, col, ld)) * hi, _clidx(i, col, ld)); i += 1; end
    @inbounds unsafe_store!(p, -sn, _clidx(col, col, ld))
    return 1.0 / (0.5 * (1.0 + (tail * abs(hi))^2))              # tinv = 1/τ
end
# Apply a single reflector H_col (rank-1) to trailing column j: A[:,j] += (−tinv·v_colᵀA[:,j])·v_col.
@inline function _qr_apply1_f64!(p::Ptr{Float64}, m::Int, col::Int, j::Int, ld::Int, tinv::Float64)
    @inbounds begin
        acc = _CVF(0.0); dot = unsafe_load(p, _clidx(col, j, ld)); i = col + 1
        while i + _CHOLW - 1 <= m; acc = muladd(vload(_CVF, _cvptr(p, i, col, ld)), vload(_CVF, _cvptr(p, i, j, ld)), acc); i += _CHOLW; end
        dot += sum(acc)
        while i <= m; dot = muladd(unsafe_load(p, _clidx(i, col, ld)), unsafe_load(p, _clidx(i, j, ld)), dot); i += 1; end
        kk = -dot * tinv; unsafe_store!(p, unsafe_load(p, _clidx(col, j, ld)) + kk, _clidx(col, j, ld)); vk = _CVF(kk); i = col + 1
        while i + _CHOLW - 1 <= m; bj = _cvptr(p, i, j, ld); vstore(muladd(vk, vload(_CVF, _cvptr(p, i, col, ld)), vload(_CVF, bj)), bj); i += _CHOLW; end
        while i <= m; unsafe_store!(p, muladd(kk, unsafe_load(p, _clidx(i, col, ld)), unsafe_load(p, _clidx(i, j, ld))), _clidx(i, j, ld)); i += 1; end
    end
end
# Unblocked panel reduction (faer port). RANK-2: factor columns in PAIRS — the two reflectors + the H0→col+1
# cross-apply are the rank-1 path, but the BULK trailing columns (j≥col+2) get a FUSED rank-2 apply (one read
# pass for both dots + one read/write pass for the rank-2 axpy) instead of two rank-1 sweeps, halving the
# level-2 trailing traffic. g=v0ᵀv1 (once/pair) decouples the 2nd dot. Measured galen panel +57-74% (18→31
# GFlops), bit-identical to rank-1. Odd tail column / trivial reflectors fall back to rank-1.
function qr_unblocked!(A::AbstractMatrix{Float64}, tau::AbstractVector{Float64})
    m, n = size(A); ld = stride(A, 2); k = min(m, n)
    GC.@preserve A begin
        p = pointer(A); col = 1
        @inbounds while col <= k
            t0 = _qr_reflect_f64!(p, m, col, ld); tau[col] = isinf(t0) ? Inf : 1.0 / t0
            if col + 1 <= k && !isinf(t0)
                _qr_apply1_f64!(p, m, col, col + 1, ld, t0)         # H0 → col+1
                t1 = _qr_reflect_f64!(p, m, col + 1, ld); tau[col + 1] = isinf(t1) ? Inf : 1.0 / t1
                if isinf(t1)
                    for j in col+2:n; _qr_apply1_f64!(p, m, col, j, ld, t0); end
                else
                    g = unsafe_load(p, _clidx(col + 1, col, ld)); gacc = _CVF(0.0); i = col + 2   # g = v0ᵀv1
                    while i + _CHOLW - 1 <= m; gacc = muladd(vload(_CVF, _cvptr(p, i, col, ld)), vload(_CVF, _cvptr(p, i, col + 1, ld)), gacc); i += _CHOLW; end
                    g += sum(gacc)
                    while i <= m; g = muladd(unsafe_load(p, _clidx(i, col, ld)), unsafe_load(p, _clidx(i, col + 1, ld)), g); i += 1; end
                    v10 = unsafe_load(p, _clidx(col + 1, col, ld))                        # v0[col+1]
                    for j in col+2:n                                                      # FUSED rank-2 apply
                        a0 = unsafe_load(p, _clidx(col, j, ld)); a1 = unsafe_load(p, _clidx(col + 1, j, ld))
                        d0 = muladd(v10, a1, a0); d1 = a1; ac0 = _CVF(0.0); ac1 = _CVF(0.0); i = col + 2
                        while i + _CHOLW - 1 <= m
                            aj = vload(_CVF, _cvptr(p, i, j, ld))
                            ac0 = muladd(vload(_CVF, _cvptr(p, i, col, ld)), aj, ac0); ac1 = muladd(vload(_CVF, _cvptr(p, i, col + 1, ld)), aj, ac1); i += _CHOLW
                        end
                        d0 += sum(ac0); d1 += sum(ac1)
                        while i <= m; aj = unsafe_load(p, _clidx(i, j, ld)); d0 = muladd(unsafe_load(p, _clidx(i, col, ld)), aj, d0); d1 = muladd(unsafe_load(p, _clidx(i, col + 1, ld)), aj, d1); i += 1; end
                        kk0 = -d0 * t0; kk1 = -(d1 + kk0 * g) * t1
                        unsafe_store!(p, a0 + kk0, _clidx(col, j, ld))
                        unsafe_store!(p, muladd(kk0, v10, a1) + kk1, _clidx(col + 1, j, ld))
                        vk0 = _CVF(kk0); vk1 = _CVF(kk1); i = col + 2
                        while i + _CHOLW - 1 <= m
                            bj = _cvptr(p, i, j, ld)
                            vstore(muladd(vk1, vload(_CVF, _cvptr(p, i, col + 1, ld)), muladd(vk0, vload(_CVF, _cvptr(p, i, col, ld)), vload(_CVF, bj))), bj); i += _CHOLW
                        end
                        while i <= m; unsafe_store!(p, muladd(kk1, unsafe_load(p, _clidx(i, col + 1, ld)), muladd(kk0, unsafe_load(p, _clidx(i, col, ld)), unsafe_load(p, _clidx(i, j, ld)))), _clidx(i, j, ld)); i += 1; end
                    end
                end
                col += 2
            else
                for j in col+1:n; isinf(t0) || _qr_apply1_f64!(p, m, col, j, ld, t0); end
                col += 1
            end
        end
    end
    return true
end

# Complex Householder reflector of column `col` (LAPACK zlarfg convention: Hⱼ = I − τⱼ·vⱼ·vⱼᴴ, τ COMPLEX,
# β REAL on the diagonal = −sign(re α)·‖col‖, vⱼ[col]=1 implicit + scaled essential below). Returns τ
# (zero(T) ⇒ trivial reflector, H=I). `p` = Ptr{Complex{R}} to A[1,1] of the view; `ld` = col stride.
@inline function _zqr_reflect!(p::Ptr{Complex{R}}, m::Int, col::Int, ld::Int, csz::Int) where {R}
    mt0 = m - col                                                  # tail length (rows col+1:m)
    # ‖tail‖ via _nrm2 (SIMD sum-of-squares fast path + lassq fallback — robust AND fast; req#6). The old
    # per-element hypot(xnorm, abs(xᵢ)) did TWO transcendentals/element (abs is itself a hypot).
    xnorm = mt0 > 0 ? _nrm2(mt0, p + ((col - 1) * ld + col) * csz, 1) : zero(R)
    @inbounds alpha = unsafe_load(p, _clidx(col, col, ld))
    (xnorm == 0 && imag(alpha) == 0) && return zero(Complex{R})    # trivial reflector
    beta = -copysign(hypot(abs(alpha), xnorm), real(alpha))
    tau = complex((beta - real(alpha)) / beta, -imag(alpha) / beta)
    sc = one(Complex{R}) / (alpha - beta)                          # v_essential = tail / (α − β)
    vp = p + ((col - 1) * ld + col) * csz                          # &A[col+1, col]
    mt0 > 0 && _scal_cmplx_simd!(mt0, real(sc), imag(sc), vp)
    @inbounds unsafe_store!(p, complex(beta, zero(R)), _clidx(col, col, ld))
    return tau
end
# Apply a single reflector H_col (v[col]=1 + essential below) to trailing column j with coefficient
# tc = conj(τ): C[:,j] := C[:,j] − tc·(vᴴ·C[:,j])·v. SIMD dotc + axpy over the tail rows col+1:m.
@inline function _zqr_apply1!(p::Ptr{Complex{R}}, m::Int, col::Int, j::Int, ld::Int, csz::Int, tc::Complex{R}) where {R}
    mt = m - col; vp = p + ((col - 1) * ld + col) * csz; cjp = p + ((j - 1) * ld + col) * csz
    @inbounds w = unsafe_load(p, _clidx(col, j, ld))               # w = vᴴ·C[:,j] (v[col]=1)
    mt > 0 && (w += _dot_cmplx_simd(mt, vp, cjp, R, Val(true)))    # + Σ conj(v)·C[:,j]
    twc = tc * w
    @inbounds unsafe_store!(p, unsafe_load(p, _clidx(col, j, ld)) - twc, _clidx(col, j, ld))
    mt > 0 && _axpy_cmplx_simd!(mt, -real(twc), -imag(twc), vp, cjp)
end
# Complex unblocked QR panel — LAPACK zgeqr2 + zlarfg. RANK-2 (mirror of the real path): factor columns in
# PAIRS. The two reflectors + the H0→col+1 cross-apply are rank-1; the BULK trailing columns (j≥col+2) get a
# FUSED rank-2 apply (one c-read pass for both dotc's + one read/write pass for the rank-2 axpy) instead of
# two rank-1 sweeps — halving the level-2 trailing traffic. g = v1ᴴ·v0 (once/pair) decouples the 2nd dot.
# Apply uses CONJ(τ) (the zgeqr2 convention — the τ-vs-conj(τ) placement is the bug magnet). Odd tail column /
# trivial reflectors fall back to rank-1. Matches LinearAlgebra.qr reconstruction.
function qr_unblocked!(A::AbstractMatrix{T}, tau::AbstractVector{T}) where {T<:BlasComplex}
    m, n = size(A); R = real(T); ld = stride(A, 2); csz = sizeof(T); k = min(m, n)
    GC.@preserve A begin
        p = pointer(A); col = 1                                   # Ptr{Complex{R}}; SIMD helpers take complex Ptr
        @inbounds while col <= k
            t0 = _zqr_reflect!(p, m, col, ld, csz); tau[col] = t0
            if col + 1 <= k && t0 != zero(T)
                _zqr_apply1!(p, m, col, col + 1, ld, csz, conj(t0))    # H0 → col+1
                t1 = _zqr_reflect!(p, m, col + 1, ld, csz); tau[col + 1] = t1
                if t1 == zero(T)
                    for j in col+2:n; _zqr_apply1!(p, m, col, j, ld, csz, conj(t0)); end
                else
                    tc0 = conj(t0); tc1 = conj(t1)
                    w10 = unsafe_load(p, _clidx(col + 1, col, ld))     # v0[col+1] essential
                    mt2 = m - (col + 1)                               # rows col+2:m
                    v0p = p + ((col - 1) * ld + (col + 1)) * csz      # &A[col+2, col]
                    v1p = p + (col * ld + (col + 1)) * csz            # &A[col+2, col+1]
                    g = w10                                          # g = v1ᴴ·v0 = w10 + Σ conj(v1)·v0
                    mt2 > 0 && (g += _dot_cmplx_simd(mt2, v1p, v0p, R, Val(true)))
                    for j in col+2:n                                  # FUSED rank-2 apply
                        cp = p + ((j - 1) * ld + (col + 1)) * csz     # &A[col+2, j]
                        a0 = unsafe_load(p, _clidx(col, j, ld)); a1 = unsafe_load(p, _clidx(col + 1, j, ld))
                        d0 = a0 + conj(w10) * a1; d1 = a1                # heads: d0 += conj(v0[col+1])·C[col+1,j]
                        if mt2 > 0
                            e0, e1 = _qr_dot2c_cmplx(mt2, v0p, v1p, cp, R)
                            d0 += e0; d1 += e1
                        end
                        k0 = -tc0 * d0; k1 = -tc1 * (d1 + k0 * g)
                        unsafe_store!(p, a0 + k0, _clidx(col, j, ld))
                        unsafe_store!(p, muladd(k0, w10, a1) + k1, _clidx(col + 1, j, ld))
                        mt2 > 0 && _qr_axpy2_cmplx!(mt2, real(k0), imag(k0), real(k1), imag(k1), v0p, v1p, cp)
                    end
                end
                col += 2
            else
                for j in col+1:n; t0 == zero(T) || _zqr_apply1!(p, m, col, j, ld, csz, conj(t0)); end
                col += 1
            end
        end
    end
    return true
end
# Blocked-QR panel width (real), DERIVED (req#8; was a flat literal 32). The unblocked rank-1/2 panel is
# BLAS-2 (cost ∝ m·nb² per panel); the blocked trailing update is gemm (efficient once nb ≳ the register
# tile ≈ _vwidth). So the panel wants the SMALLEST nb that keeps the trailing gemm efficient (= _vwidth),
# growing only when the matrix spills L2 and the trailing re-streams from DRAM (sweep traffic ∝ k/nb) —
# one _vwidth per L2-overflow, capped at 4·_vwidth where the BLAS-2 panel share (≈0.75·nb/n) starts to
# bite. MEASURED Zen4 (vw=8, L2=1MB): 8 (n≤256) → 16 (384–512) → 32 (≥768), each within ~2% of the
# nb-sweep optimum, vs the old flat nb=32 which lost 20–50% at n=48–256 (and caused the n=48 AOCL gate
# miss). Tall (large m·n) caps at 32 = the old value (no regression — [[pureblas-getrf-campaign]] found
# growth ABOVE 32 hurts tall). Keyed _vwidth + _L2_BYTES. NEEDS Zen3/Zen5 fleet validation (req#8b).
@inline _qr_nb(m::Int, n::Int) = clamp(_vwidth(Float64) * cld(m * n * sizeof(Float64), _L2_BYTES),
    _vwidth(Float64), 4 * _vwidth(Float64))
# Unblocked→blocked crossover (real): the SIMD rank-2 unblocked panel beats the blocked WY path only while
# the matrix is TINY (n≤32 on Zen4) — real blocked-nb=_vwidth is efficient enough to win from n≈48 up.
# (Contrast the complex path at qr.jl:_zqr_unblk_max, whose stronger BLAS-2 panel stays ahead to ~½L3.)
# Pinned at 4·_vwidth = the old flat _QR_NB (=32 Zen4), preserving the measured n≤32 unblocked win,
# decoupled from the now-variable nb. Dim-wise (not m·n) so a tall-skinny front stays unblocked.
const _QR_UNBLK_MAX = 4 * _vwidth(Float64)
# Complex panel width: the unblocked complex panel (SIMD zlarf: dotc+axpy per trailing col) is per-column-
# call-bound, so a narrow panel hands the O(n²k) trailing update to the gating blocked complex gemm sooner.
# Keyed via Preferences per box (Zen4 sweet spot measured).
const _QR_NB_C = @load_preference("qr_nb_c", 32)::Int
# Unblocked→blocked crossover (COMPLEX). The SIMD rank-2 unblocked panel (qr_unblocked!) is BLAS-2 but
# strong — it BEATS the blocked WY path while the matrix stays cache-resident (its O(n³) re-streams hit cache,
# not DRAM). req#8: derive from residency, not a size literal. On the wide-SIMD box (AVX-512, _CGEMM_3M off)
# unblocked wins to matrix ≈ ½L3 (neuromancer Zen5: single to n≈700, collapses n≥1536). On the AVX2-complex
# box (_CGEMM_3M) the BLAS-2 panel is bandwidth-bound AND the chunked blocked path is efficient from small n,
# so unblocked only wins while the matrix is ~L2-resident (≤2·L2 ⇒ galen n≤256; n≥320 must reach blocked).
@inline _zqr_unblk_max(::Type{T}) where {T} = _CGEMM_3M ? 2 * _L2_BYTES : _L3_BYTES ÷ 2
# Blocked panel width (COMPLEX), derived: the trailing gemm C−=V·W contracts over pb=nb, and once the matrix
# spills cache each panel re-streams the trailing matrix from DRAM (total sweep traffic ∝ k/nb). Grow nb with
# how many times the matrix exceeds ¼L3 so DRAM-sweep bytes stay bounded, capped at 4× where the BLAS-2 panel
# share (≈0.75·nb/n) starts to bite. Measured galen: 32 (n≤512) → 64 (768) → 96 (1024) → 128 (≥1280), gating
# the whole range (nb=32 dips at n≥896, nb=64 dips at n≈320). Overridable base "qr_nb_c".
@inline _zqr_nb(::Type{T}, m::Int, n::Int) where {T} =
    clamp(_QR_NB_C * cld(m * n * sizeof(T), _L3_BYTES ÷ 4), _QR_NB_C, 4 * _QR_NB_C)
# Complex blocked-QR workspace (GKH: a second owned Ref for ComplexF64, mirroring _QR_WS).
const _QR_WS_C = Ref{NTuple{5, Matrix{ComplexF64}}}(ntuple(_ -> Matrix{ComplexF64}(undef, 0, 0), 5))
@inline function _qr_ws_c(::Type{T}, m::Int, n::Int, nb::Int) where {T}
    V, Tm, G, Wb, Yb = _QR_WS_C[]
    if size(V, 1) < m || size(V, 2) < nb || size(Tm, 1) < nb || size(Wb, 2) < n
        V = Matrix{T}(undef, m, nb); Tm = Matrix{T}(undef, nb, nb)
        G = Matrix{T}(undef, nb, nb); Wb = Matrix{T}(undef, nb, n); Yb = Matrix{T}(undef, nb, n)
        _QR_WS_C[] = (V, Tm, G, Wb, Yb)
    end
    return V, Tm, G, Wb, Yb
end
# Complex blocked QR (zgeqrf): per nb-panel qr_unblocked! (complex zlarfg), build V + the LAPACK compact-WY
# T (zlarft: T[c,c]=τ, T[1:c-1,c] = T[1:c-1,1:c-1]·(−τ·G[1:c-1,c]), G=VᴴV), then apply the block Qᴴ to the
# trailing: C −= V·(Tᴴ·(Vᴴ·C)) — the two big products via complex gemm ('C' = compile-time conj signs).
function geqrf!(A::AbstractMatrix{T}, tau::AbstractVector{T}; nb::Int = 0) where {T<:BlasComplex}
    m, n = size(A); k = min(m, n)
    k == 0 && return A
    length(tau) >= k || throw(DimensionMismatch("geqrf!: length(tau) < min(size(A))"))
    # Single unblocked panel while the matrix is cache-resident (BLAS-2 rank-2 panel beats the blocked WY path
    # there); blocked only once re-streaming would spill to DRAM. Subsumes the tiny k≤nb case (no workspace).
    if m * n * sizeof(T) <= _zqr_unblk_max(T)
        qr_unblocked!(view(A, 1:m, 1:n), view(tau, 1:k)); return A
    end
    nb = clamp(nb > 0 ? nb : _zqr_nb(T, m, n), 1, k)
    V, Tm, G, Wb, Yb = _qr_ws_c(T, m, n, nb)
    pc = 1
    @inbounds while pc <= k
        pb = min(nb, k - pc + 1)
        qr_unblocked!(view(A, pc:m, pc:pc+pb-1), view(tau, pc:pc+pb-1))
        jt0 = pc + pb
        if jt0 <= n
            mp = m - pc + 1; nt = n - jt0 + 1
            Vv = view(V, 1:mp, 1:pb)
            for c in 1:pb, i in 1:mp
                Vv[i, c] = i == c ? one(T) : (i > c ? A[pc+i-1, pc+c-1] : zero(T))
            end
            Gv = view(G, 1:pb, 1:pb)
            herk!(Gv, Vv; uplo = 'U', trans = 'C', alpha = true, beta = false)  # G = Vᴴ V (upper only —
                                                                               # dlarft reads Gv[kk,c], kk<c; half flops)
            Tv = view(Tm, 1:pb, 1:pb)                                       # LAPACK zlarft T
            for c in 1:pb
                τc = tau[pc+c-1]; Tv[c, c] = τc
                for r in 1:c-1                                              # T[r,c] = Σ_{kk≥r} T[r,kk]·(−τc·G[kk,c])
                    s = zero(T)
                    for kk in r:c-1; s = muladd(Tv[r, kk], -τc * Gv[kk, c], s); end
                    Tv[r, c] = s
                end
            end
            # nc-CHUNK the trailing update over columns. On the AVX2-complex path (_CGEMM_3M) the two trailing
            # gemms route through Karatsuba-3M, which materializes O(mp·nt) real split/product scratch (~3× the
            # trailing matrix) — for nt≈mp≈n that spills L3 exactly when the matrix ITSELF fits L3 (the galen
            # n≈768–1100 dip band). Chunking columns bounds that scratch to ≤½L3 AND fuses each Cj's V·W downdate
            # while its Vᴴ·C is still L3-hot (read+write once vs ~5–6 matrix passes). ncb = nt (single chunk,
            # bit-identical) when 3M is off (neuromancer). req#8: ncb from the 3M live-set residency ≤ ½·L3,
            # live set ≈ (sizeof(T) + 3·sizeof(real(T)))·mp·ncb (C chunk + its real split/product buffers).
            ncb = _CGEMM_3M ? clamp(((_L3_BYTES ÷ 2) ÷ ((sizeof(T) + 3 * sizeof(real(T))) * mp)) & ~7, min(4 * _CNR, nt), nt) : nt
            jj = jt0
            while jj <= n
                cw = min(ncb, n - jj + 1)
                Cj = view(A, pc:m, jj:jj+cw-1)
                Wv = view(Wb, 1:pb, 1:cw); gemm!(Wv, Vv, Cj; transA = 'C', alpha = true, beta = false)  # W = Vᴴ Cj
                trmm!(Wv, Tv; side = 'L', uplo = 'U', transA = 'C')        # W := Tᴴ W (SIMD trmm)
                gemm!(Cj, Vv, Wv; alpha = -1, beta = true)                 # Cj −= V·(Tᴴ W)
                jj += cw
            end
        end
        pc += pb
    end
    return A
end
function geqrf!(A::StridedMatrix{T}) where {T<:BlasComplex}
    tau = Vector{T}(undef, min(size(A)...))
    geqrf!(A, tau)
    return A, tau
end

# Blocked compact-WY QR. Reduce each nb-panel (qr_unblocked!), build V (unit-lower-trapezoid) + the
# compact-WY T (dlarft), then apply Qᵀ to the trailing block: C −= V·(Tᵀ·(Vᵀ·C)) — the two big gemms via
# PureBLAS's cache-blocked gemm! (VᵀV and the trailing get gemm; Y=TᵀW is tiny → scalar).
# Cached blocked-QR workspace (V m×nb, Tm/G nb×nb, Wb/Yb nb×n) — a fresh 5-matrix alloc per call
# dominated geqrf at n=32–64. Regrown on demand; single-thread (like the other L3/LAPACK scratches).
const _QR_WS = Ref{NTuple{5, Matrix{Float64}}}((Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0),
    Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0)))
@inline function _qr_ws(m::Int, n::Int, nb::Int)
    V, Tm, G, Wb, Vt = _QR_WS[]                                   # 5th slot repurposed Yb→Vt (nb×m): the
    if size(V, 1) < m || size(V, 2) < nb || size(Tm, 1) < nb || size(Wb, 2) < n || size(Vt, 2) < m
        V = Matrix{Float64}(undef, m, nb); Tm = Matrix{Float64}(undef, nb, nb)   # transposed V for the skinny
        G = Matrix{Float64}(undef, nb, nb); Wb = Matrix{Float64}(undef, nb, n); Vt = Matrix{Float64}(undef, nb, m)
        _QR_WS[] = (V, Tm, G, Wb, Vt)                             # W=Vᵀ·C as an unpacked no-trans gemm
    end
    return V, Tm, G, Wb, Vt
end
function geqrf!(A::AbstractMatrix{Float64}, tau::AbstractVector{Float64}; nb::Int = 0)
    m, n = size(A); k = min(m, n)
    k == 0 && return A
    length(tau) >= k || throw(DimensionMismatch("geqrf!: length(tau) < min(size(A))"))
    # Tiny matrix → single unblocked panel (BLAS-2 rank-2 beats the blocked WY path there; no workspace).
    if k <= _QR_UNBLK_MAX && n <= _QR_UNBLK_MAX
        qr_unblocked!(view(A, 1:m, 1:n), view(tau, 1:k))
        return A
    end
    nb = clamp(nb > 0 ? nb : _qr_nb(m, n), 1, k)    # nb>0 = caller override (tuning); else derive
    V, Tm, G, Wb, Vt = _qr_ws(m, n, nb)
    pc = 1
    @inbounds while pc <= k
        pb = min(nb, k - pc + 1)
        qr_unblocked!(view(A, pc:m, pc:pc+pb-1), view(tau, pc:pc+pb-1))
        jt0 = pc + pb
        if jt0 <= n
            mp = m - pc + 1; nt = n - jt0 + 1
            Vv = view(V, 1:mp, 1:pb); Vtv = view(Vt, 1:pb, 1:mp)
            # Skinny W=Vᵀ·C via the unpacked path — the crossover is a µARCH SPLIT (req#8; measured fleet):
            #  • AVX2 (W=4): the packed transA skinny gemm is STABLE at large mp, so unpacked wins only while Vᵀ
            #    (pb·mp·8 B) stays L2-resident (else it re-fetches Vᵀ per n-tile). Gate: Vᵀ≤½L2 ⇔ 16·pb·mp≤L2
            #    (galen: unpacked +8-17% at mp≤1024/Vt≤256KB, -10-23% at mp≥1536).
            #  • AVX-512 (W=8): the packed transA skinny gemm COLLAPSES at large mp (neuro/wm: packed 16-28 vs
            #    unpacked ~30-42 GF for mp≥512), so unpacked wins for all non-tiny mp. Gate: mp>_GEMM_UNPACK_MAX.
            useskinny = _CHOLW == 4 ? (16 * pb * mp <= _L2_BYTES) : (mp > _GEMM_UNPACK_MAX)
            if useskinny
                for c in 1:pb, i in 1:mp                           # V and its transpose Vᵀ from A in ONE pass
                    val = i == c ? 1.0 : (i > c ? A[pc+i-1, pc+c-1] : 0.0)
                    Vv[i, c] = val; Vtv[c, i] = val
                end
            else
                for c in 1:pb, i in 1:mp
                    Vv[i, c] = i == c ? 1.0 : (i > c ? A[pc+i-1, pc+c-1] : 0.0)
                end
            end
            Gv = view(G, 1:pb, 1:pb)
            syrk!(Gv, Vv; uplo = 'U', trans = 'T', alpha = true, beta = false)  # G = Vᵀ V (upper only — dlarft
                                                                               # reads Gv[kk,c], kk<c; half the flops)
            Tv = view(Tm, 1:pb, 1:pb)                                       # compact-WY T (dlarft), λ=1/τ
            for c in 1:pb
                tc = tau[pc+c-1]; λ = isfinite(tc) ? 1.0 / tc : 0.0
                Tv[c, c] = λ
                for r in 1:c-1
                    s = 0.0
                    for kk in r:c-1; s = muladd(Tv[r, kk], Gv[kk, c], s); end
                    Tv[r, c] = -λ * s
                end
            end
            C = view(A, pc:m, jt0:n)
            Wv = view(Wb, 1:pb, 1:nt)                                     # W = Vᵀ·C: skinny m=pb. When Vᵀ is
            if useskinny                                                  # L2-resident, the UNPACKED no-trans path
                _gemm_unpacked!(Val(false), Val(true), pb, nt, mp, 1.0, Vtv, C, 0.0, Wv)  # (Vᵀ prebuilt, streams C
            else                                                          # in place) beats the packed transA gemm
                gemm!(Wv, Vv, C; transA = 'T', alpha = true, beta = false)   # +8-17%; else packed kc-blocked wins.
            end
            trmm!(Wv, Tv; side = 'L', uplo = 'U', transA = 'T')            # W := Tᵀ W (was a scalar latency-bound
                                                                          # triple loop — SIMD trmm instead)
            gemm!(C, Vv, Wv; alpha = -1, beta = true)                     # C −= V·(Tᵀ W)
        end
        pc += pb
    end
    return A
end

# Convenience: allocate tau, return (A overwritten with R + reflectors, tau).
function geqrf!(A::StridedMatrix{Float64})
    tau = Vector{Float64}(undef, min(size(A)...))
    geqrf!(A, tau)
    return A, tau
end
