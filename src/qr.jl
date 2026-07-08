# LAPACK QR (geqrf) — Householder, no pivoting. Port of faer 0.24.1's unblocked panel reduction
# (el-oso/BlazingPorts.jl `src/Factorizations.jl`) onto PureBLAS's SIMD.jl layer, driven by a blocked
# compact-WY (LAPACK dlarft/dlarfb) loop whose trailing update is PureBLAS's gated `gemm!` — so faer's
# bespoke packed BLIS gemm is unneeded (we have one). faer convention: H_k = I − v_k v_kᵀ/τ_k, τ_k=Inf ⇒
# identity; on output the upper triangle of A is R, the essential v_k (implicit v_k[k]=1) sit below the
# diagonal of column k, τ[k] the coefficient. Float64-only (the proven-fast path); reuses the Cholesky
# SIMD helpers (_CVF/_CHOLW/_clidx/_cvptr from lapack.jl). ponytail: generic/AD QR deferred.

# Unblocked panel reduction (faithful faer port): factor A's columns left-to-right, applying each
# reflector to the panel's own trailing columns. Vectorized over rows.
function qr_unblocked!(A::AbstractMatrix{Float64}, tau::AbstractVector{Float64})
    m, n = size(A); ld = stride(A, 2)
    GC.@preserve A begin
        p = pointer(A)
        @inbounds for col in 1:min(m, n)
            row = col
            tn = 0.0; i = row + 1                               # ‖tail‖²
            while i + _CHOLW - 1 <= m
                x = vload(_CVF, _cvptr(p, i, col, ld)); tn += sum(x * x); i += _CHOLW
            end
            while i <= m
                x = unsafe_load(p, _clidx(i, col, ld)); tn = muladd(x, x, tn); i += 1
            end
            tail_norm = sqrt(tn)
            head = unsafe_load(p, _clidx(row, col, ld)); head_norm = abs(head)
            if tail_norm < floatmin(Float64)
                tau[col] = Inf; continue                        # trivial reflector
            end
            nrm = hypot(head_norm, tail_norm)
            signed_norm = head >= 0.0 ? nrm : -nrm
            hwb = head + signed_norm; hwb_inv = 1.0 / hwb; vinv = _CVF(hwb_inv)
            i = row + 1                                          # v_essential = tail · (1/hwb)
            while i + _CHOLW - 1 <= m
                b = _cvptr(p, i, col, ld); vstore(vload(_CVF, b) * vinv, b); i += _CHOLW
            end
            while i <= m
                unsafe_store!(p, unsafe_load(p, _clidx(i, col, ld)) * hwb_inv, _clidx(i, col, ld)); i += 1
            end
            unsafe_store!(p, -signed_norm, _clidx(row, col, ld))     # R diagonal = β
            t = 0.5 * (1.0 + (tail_norm * abs(hwb_inv))^2); tau[col] = t; tinv = 1.0 / t
            for j in col+1:n                                    # apply H_col to trailing panel columns
                acc = _CVF(0.0); dscal = unsafe_load(p, _clidx(row, j, ld)); i = row + 1
                while i + _CHOLW - 1 <= m
                    acc = muladd(vload(_CVF, _cvptr(p, i, col, ld)), vload(_CVF, _cvptr(p, i, j, ld)), acc); i += _CHOLW
                end
                dot = dscal + sum(acc)
                while i <= m
                    dot = muladd(unsafe_load(p, _clidx(i, col, ld)), unsafe_load(p, _clidx(i, j, ld)), dot); i += 1
                end
                kk = -dot * tinv
                unsafe_store!(p, unsafe_load(p, _clidx(row, j, ld)) + kk, _clidx(row, j, ld))
                vk = _CVF(kk); i = row + 1
                while i + _CHOLW - 1 <= m
                    bj = _cvptr(p, i, j, ld)
                    vstore(muladd(vk, vload(_CVF, _cvptr(p, i, col, ld)), vload(_CVF, bj)), bj); i += _CHOLW
                end
                while i <= m
                    unsafe_store!(p, muladd(kk, unsafe_load(p, _clidx(i, col, ld)), unsafe_load(p, _clidx(i, j, ld))), _clidx(i, j, ld)); i += 1
                end
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
    m, n = size(A); R = real(T)
    @inbounds for col in 1:min(m, n)
        row = col
        xnorm = zero(R)                                          # ‖tail‖ (rows row+1:m)
        for i in (row + 1):m; xnorm = hypot(xnorm, abs(A[i, col])); end
        alpha = A[row, col]
        if xnorm == 0 && imag(alpha) == 0
            tau[col] = zero(T); continue                         # trivial reflector
        end
        beta = -copysign(hypot(abs(alpha), xnorm), real(alpha))  # β = −sign(re α)·‖[α;x]‖ (real)
        tau[col] = complex((beta - real(alpha)) / beta, -imag(alpha) / beta)
        sc = one(T) / (alpha - beta)                             # v_essential = tail / (α − β)
        for i in (row + 1):m; A[i, col] *= sc; end
        A[row, col] = complex(beta, zero(R))                     # R diagonal = β
        tc = conj(tau[col])
        for j in (col + 1):n                                     # C := (I − conj(τ)·v·vᴴ)·C
            w = A[row, j]                                        # w = vᴴ·C[:,j] (v[row]=1)
            for i in (row + 1):m; w += conj(A[i, col]) * A[i, j]; end
            twc = tc * w
            A[row, j] -= twc
            for i in (row + 1):m; A[i, j] -= twc * A[i, col]; end
        end
    end
    return true
end
const _QR_NB = 32     # blocked panel width; ponytail: hand-set for Zen4, tune if needed
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
function geqrf!(A::AbstractMatrix{T}, tau::AbstractVector{T}; nb::Int = _QR_NB) where {T<:BlasComplex}
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
            gemm!(Gv, Vv, Vv; transA = 'T', alpha = true, beta = false)     # G = Vᵀ V
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
            Yv = view(Yb, 1:pb, 1:nt)
            for j in 1:nt, c in 1:pb                            # Y = Tᵀ W (Tᵀ lower-tri; pb small → scalar)
                s = 0.0
                for r in 1:c; s = muladd(Tv[r, c], Wv[r, j], s); end
                Yv[c, j] = s
            end
            gemm!(C, Vv, Yv; alpha = -1, beta = true)                       # C −= V Y
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
