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

const _QR_NB = 32     # blocked panel width; ponytail: hand-set for Zen4, tune if needed

# Blocked compact-WY QR. Reduce each nb-panel (qr_unblocked!), build V (unit-lower-trapezoid) + the
# compact-WY T (dlarft), then apply Qᵀ to the trailing block: C −= V·(Tᵀ·(Vᵀ·C)) — the two big gemms via
# PureBLAS's cache-blocked gemm! (VᵀV and the trailing get gemm; Y=TᵀW is tiny → scalar).
function geqrf!(A::StridedMatrix{Float64}, tau::AbstractVector{Float64}; nb::Int = _QR_NB)
    m, n = size(A); k = min(m, n)
    k == 0 && return A
    length(tau) >= k || throw(DimensionMismatch("geqrf!: length(tau) < min(size(A))"))
    nb = clamp(nb, 1, k)
    V = Matrix{Float64}(undef, m, nb); Tm = Matrix{Float64}(undef, nb, nb)
    G = Matrix{Float64}(undef, nb, nb); Wb = Matrix{Float64}(undef, nb, n); Yb = Matrix{Float64}(undef, nb, n)
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
