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

# Complex unblocked QR panel — LAPACK zgeqr2 + zlarfg. Hⱼ = I − τⱼ·vⱼ·vⱼᴴ with τ COMPLEX, β REAL on the
# diagonal (β = −sign(re α)·‖col‖), vⱼ[j]=1 implicit + essential below. Apply to the trailing columns with
# CONJ(τ) (the zgeqr2 convention — the τ-vs-conj(τ) placement is the bug magnet). Scalar; the blocked
# driver rides complex gemm ('C') for the O(n³) trailing update. Matches LinearAlgebra.qr reconstruction.
function qr_unblocked!(A::AbstractMatrix{T}, tau::AbstractVector{T}) where {T<:BlasComplex}
    m, n = size(A); R = real(T); ld = stride(A, 2); csz = sizeof(T)
    GC.@preserve A begin
        p = pointer(A)                                          # Ptr{Complex{R}}; SIMD helpers take complex Ptr
        @inbounds for col in 1:min(m, n)
            row = col
            xnorm = zero(R)                                      # ‖tail‖ (rows row+1:m)
            for i in (row + 1):m; xnorm = hypot(xnorm, abs(A[i, col])); end
            alpha = A[row, col]
            if xnorm == 0 && imag(alpha) == 0
                tau[col] = zero(T); continue                     # trivial reflector
            end
            beta = -copysign(hypot(abs(alpha), xnorm), real(alpha))  # β = −sign(re α)·‖[α;x]‖ (real)
            tau[col] = complex((beta - real(alpha)) / beta, -imag(alpha) / beta)
            sc = one(T) / (alpha - beta)                         # v_essential = tail / (α − β)
            mt = m - row                                        # tail length (rows row+1:m)
            vp = p + ((col - 1) * ld + row) * csz               # &A[row+1, col]
            mt > 0 && _scal_cmplx_simd!(mt, real(sc), imag(sc), vp)   # SIMD: scale essential v
            A[row, col] = complex(beta, zero(R))                 # R diagonal = β
            tc = conj(tau[col])
            for j in (col + 1):n                                 # C := (I − conj(τ)·v·vᴴ)·C
                cjp = p + ((j - 1) * ld + row) * csz            # &A[row+1, j]
                w = A[row, j]                                    # w = vᴴ·C[:,j] (v[row]=1)
                mt > 0 && (w += _dot_cmplx_simd(mt, vp, cjp, R, Val(true)))   # SIMD dotc: Σ conj(v)·C[:,j]
                twc = tc * w
                A[row, j] -= twc
                mt > 0 && _axpy_cmplx_simd!(mt, -real(twc), -imag(twc), vp, cjp)  # SIMD: C[:,j] -= twc·v
            end
        end
    end
    return true
end
const _QR_NB = 32     # blocked panel width; ponytail: hand-set for Zen4, tune if needed
# Complex panel width: the unblocked complex panel (SIMD zlarf: dotc+axpy per trailing col) is per-column-
# call-bound, so a narrow panel hands the O(n²k) trailing update to the gating blocked complex gemm sooner.
# Keyed via Preferences per box (Zen4 sweet spot measured).
const _QR_NB_C = @load_preference("qr_nb_c", 32)::Int
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
function geqrf!(A::AbstractMatrix{T}, tau::AbstractVector{T}; nb::Int = _QR_NB_C) where {T<:BlasComplex}
    m, n = size(A); k = min(m, n)
    k == 0 && return A
    length(tau) >= k || throw(DimensionMismatch("geqrf!: length(tau) < min(size(A))"))
    nb = clamp(nb, 1, k)
    if k <= nb && n <= nb
        qr_unblocked!(view(A, 1:m, 1:n), view(tau, 1:k)); return A
    end
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
            gemm!(Gv, Vv, Vv; transA = 'C', alpha = true, beta = false)     # G = Vᴴ V
            Tv = view(Tm, 1:pb, 1:pb)                                       # LAPACK zlarft T
            for c in 1:pb
                τc = tau[pc+c-1]; Tv[c, c] = τc
                for r in 1:c-1                                              # T[r,c] = Σ_{kk≥r} T[r,kk]·(−τc·G[kk,c])
                    s = zero(T)
                    for kk in r:c-1; s = muladd(Tv[r, kk], -τc * Gv[kk, c], s); end
                    Tv[r, c] = s
                end
            end
            C = view(A, pc:m, jt0:n)
            Wv = view(Wb, 1:pb, 1:nt); gemm!(Wv, Vv, C; transA = 'C', alpha = true, beta = false)  # W = Vᴴ C
            Yv = view(Yb, 1:pb, 1:nt)
            for j in 1:nt, c in 1:pb                                        # Y = Tᴴ W (Tᴴ lower: r≤c → conj)
                s = zero(T)
                for r in 1:c; s = muladd(conj(Tv[r, c]), Wv[r, j], s); end
                Yv[c, j] = s
            end
            gemm!(C, Vv, Yv; alpha = -1, beta = true)                       # C −= V Y
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
    V, Tm, G, Wb, Yb = _QR_WS[]
    if size(V, 1) < m || size(V, 2) < nb || size(Tm, 1) < nb || size(Wb, 2) < n
        V = Matrix{Float64}(undef, m, nb); Tm = Matrix{Float64}(undef, nb, nb)
        G = Matrix{Float64}(undef, nb, nb); Wb = Matrix{Float64}(undef, nb, n); Yb = Matrix{Float64}(undef, nb, n)
        _QR_WS[] = (V, Tm, G, Wb, Yb)
    end
    return V, Tm, G, Wb, Yb
end
function geqrf!(A::AbstractMatrix{Float64}, tau::AbstractVector{Float64}; nb::Int = _QR_NB)
    m, n = size(A); k = min(m, n)
    k == 0 && return A
    length(tau) >= k || throw(DimensionMismatch("geqrf!: length(tau) < min(size(A))"))
    nb = clamp(nb, 1, k)
    if k <= nb && n <= nb                     # single panel, no trailing update — no workspace needed
        qr_unblocked!(view(A, 1:m, 1:n), view(tau, 1:k))
        return A
    end
    V, Tm, G, Wb, Yb = _qr_ws(m, n, nb)
    pc = 1
    @inbounds while pc <= k
        pb = min(nb, k - pc + 1)
        qr_unblocked!(view(A, pc:m, pc:pc+pb-1), view(tau, pc:pc+pb-1))
        jt0 = pc + pb
        if jt0 <= n
            mp = m - pc + 1; nt = n - jt0 + 1
            Vv = view(V, 1:mp, 1:pb)
            for c in 1:pb, i in 1:mp                            # dense unit-lower-trapezoid V
                Vv[i, c] = i == c ? 1.0 : (i > c ? A[pc+i-1, pc+c-1] : 0.0)
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
            Wv = view(Wb, 1:pb, 1:nt); gemm!(Wv, Vv, C; transA = 'T', alpha = true, beta = false)  # W = Vᵀ C
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
