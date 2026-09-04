# Column-pivoted QR (LAPACK geqp3 / geqpf):  A·P = Q·R, with P chosen so the R diagonal is
# non-increasing in magnitude (|R[1,1]| ≥ |R[2,2]| ≥ …) — rank-revealing. This is the UNBLOCKED
# core (LAPACK dgeqpf/zgeqpf; dgeqp3's blocked trailing update is a perf refinement over the SAME
# numerics), composed from PureBLAS's own Householder reflector kernels `_larfg!`/`_house_left!`
# (svd.jl, standard LAPACK τ convention) plus `_nrm2` (level1.jl, lassq-safe — req#6) for the
# partial-column-norm downdating. Generic over Float32/Float64/ComplexF32/ComplexF64.
#
# tau is standard LAPACK (H_i = I − τ_i·v_i·v_iᴴ, v_i[i]=1 implicit + essential below the diagonal
# of column i, R in the upper triangle) — the SAME convention `LinearAlgebra.LAPACK.geqp3!` returns.
# jpvt is 1-based: jpvt[k] = original index of the column now sitting in position k (so A_in[:,jpvt]
# = Q·R). Pivoting is the classic max-partial-norm rule with the dlaqps √-tolerance recompute test.

# op(H)-apply coefficient for a single reflector: for Qᴴ apply conj(τ) on complex (matches LAPACK
# zlarf w/ CONJG(TAU)); real is symmetric so τ passes through.
@inline _geqp3_tau_Qh(τ::T) where {T <: BlasReal} = τ
@inline _geqp3_tau_Qh(τ::T) where {T <: BlasComplex} = conj(τ)

# ── real-strided fast path for the unblocked loop ────────────────────────────────────────────────
# Phase decomposition of the n=32 call (Zen4 freq-locked, in-situ time_ns accumulators, 2026-08-02)
# put the reflector apply at ~41% (6.6 GF/s) and the norm DOWNDATE at ~25% — the downdate is two
# scalar divides + a sqrt per trailing column (≈20 cyc latency each visit), pure scalar throughput
# next to zero flops. The fix below: the apply pass collects each updated head A[i,j] into a
# contiguous buffer, and the downdate becomes ONE SIMD pass (vdivpd/vsqrtpd are full-width) with a
# scalar fixup chunk only where the tol3z recompute test fires (rare). IEEE div/sqrt vectorize
# exactly, so results are BITWISE identical to the scalar order. Measured (same harness): n=32
# 13.8→11.2 µs (1.23×), n=8 1.07× — the shipped path LOST the n=32 gate cell to AOCL at 0.848.

# Scalar downdate of one column: the LAPACK dlaqp2/dlaqps formula, verbatim (recompute via lassq-safe
# _nrm2, req#6). `h` is the just-updated head A[i,jj].
@inline function _geqp3_dd1!(A, i::Int, m::Int, jj::Int, h::T, vn1, vn2, tol3z::T) where {T <: BlasReal}
    @inbounds if !iszero(vn1[jj])
        temp = one(T) - (abs(h) / vn1[jj])^2
        temp = max(temp, zero(T))
        temp2 = temp * (vn1[jj] / vn2[jj])^2
        if temp2 <= tol3z
            if i < m
                nrm = _nrm2(m - i, view(A, (i + 1):m, jj), 1); vn1[jj] = nrm; vn2[jj] = nrm
            else
                vn1[jj] = zero(T); vn2[jj] = zero(T)
            end
        else
            vn1[jj] = vn1[jj] * sqrt(temp)
        end
    end
    return
end

# One reflector step of the unblocked loop: apply H = I − τ·v·vᵀ (v = A[i:m,i], head ≡ 1, holds β —
# never read) to the trailing columns AND downdate vn1/vn2. Per column the apply is the same
# _dot_simd/_axpy_simd! pair `_house_left!`'s SIMD branch issues (bitwise-identical results); the
# downdate runs as the SIMD pass described above. Below 2·_vwidth trailing columns there is at most
# one full vector + tail, so the two-pass structure has nothing to amortize — the downdate stays
# fused scalar per column instead (derived cutoff: pure ISA width, no tuning literal; this is also
# what keeps n=8 ahead of the old code, measured 0.91×→1.07×).
function _geqp3_house_dd!(
        A::AbstractMatrix{T}, i::Int, m::Int, n::Int, τ::T,
        # `AbstractVector`, not `Vector`: these arrive as arena `PtrVector` borrows (they were contiguous
        # workspace views before). A `::Vector{T}` annotation would not match either, so the call would
        # fail to dispatch rather than silently deoptimise — but widen it deliberately, and note the fast
        # paths downstream gate on `_strided1`/`_dense1`. A contiguous view satisfies `_dense1` by being
        # a `StridedVector`; a `PtrVector` satisfies it only through the explicit method in ptrmat.jl —
        # without that it would take the SCALAR path here, silently.
        hbuf::AbstractVector{T}, vn1::AbstractVector{T}, vn2::AbstractVector{T}, tol3z::T
    ) where {T <: BlasReal}
    len = m - i + 1; nc = n - i
    W = _vwidth(T)
    sz = sizeof(T)
    GC.@preserve A hbuf begin
        pa = pointer(A); ld = stride(A, 2)
        pv = pa + ((i - 1) * ld + (i - 1)) * sz          # column i, row i: the reflector v
        if nc < 2 * W
            @inbounds for j in (i + 1):n
                cp = pa + ((j - 1) * ld + (i - 1)) * sz
                c1 = unsafe_load(cp, 1)
                w = iszero(τ) ? zero(T) : τ * (c1 + _dot_simd(len - 1, cp + sz, pv + sz, T))
                c1new = c1 - w
                unsafe_store!(cp, c1new, 1)
                iszero(w) || _axpy_simd!(len - 1, -w, pv + sz, cp + sz)
                _geqp3_dd1!(A, i, m, j, c1new, vn1, vn2, tol3z)
            end
            return
        end
        @inbounds for j in (i + 1):n
            cp = pa + ((j - 1) * ld + (i - 1)) * sz
            c1 = unsafe_load(cp, 1)
            w = iszero(τ) ? zero(T) : τ * (c1 + _dot_simd(len - 1, cp + sz, pv + sz, T))
            c1new = c1 - w
            unsafe_store!(cp, c1new, 1)
            hbuf[j - i] = c1new                          # contiguous heads for the SIMD downdate
            iszero(w) || _axpy_simd!(len - 1, -w, pv + sz, cp + sz)
        end
    end
    V = Vec{W, T}
    j = 1
    # POINTER form, not SIMD.jl's array form. `vload(V, vn1, k)` only accepts the `DenseVector` /
    # contiguous-`SubArray` union, so it MethodErrors on the `PtrVector` these now arrive as — a hard
    # failure, not a deoptimisation, and it is the shape the C-ABI would have hit too if it ever reached
    # this kernel (it does not: `_geqp3_house_dd!` is only called from `geqp3!`, which is why the
    # container was never exercised). `pointer(x, k)` is defined for Vector, contiguous SubArray and
    # PtrVector alike, and lowers to the same address arithmetic the array form does internally.
    GC.@preserve vn1 vn2 hbuf @inbounds begin
        while j + W - 1 <= nc
            v1 = vload(V, pointer(vn1, i + j)); v2 = vload(V, pointer(vn2, i + j))
            h = vload(V, pointer(hbuf, j))
            q = abs(h) / v1
            temp = max(one(T) - q * q, zero(T))
            r = v1 / v2
            t2 = temp * (r * r)
            zero1 = v1 == zero(T)                        # vn1==0 lanes: q=Inf ⇒ temp=0 ⇒ t2=0 would
            flag = (t2 <= tol3z) & !zero1                # false-flag; mask them (they stay 0)
            if any(flag)
                for l in 0:(W - 1)                       # exact LAPACK path for the whole chunk
                    _geqp3_dd1!(A, i, m, i + j + l, hbuf[j + l], vn1, vn2, tol3z)
                end
            else
                vstore(vifelse(zero1, v1, v1 * sqrt(temp)), pointer(vn1, i + j))
            end
            j += W
        end
        while j <= nc
            _geqp3_dd1!(A, i, m, i + j, hbuf[j], vn1, vn2, tol3z)
            j += 1
        end
    end
    return
end

# Initial column norms, real-strided fast path: one plain-SIMD sum-of-squares pass per (contiguous)
# column with the lassq-safe _nrm2 as the over/underflow fallback — the SAME guard pattern _larfg!'s
# fast path uses (reroute when ss is non-finite or below floatmin; req#6 honoured via the fallback).
# 32 separate _nrm2 entries were ~14% of the n=32 call (measured); this pass is ~4×.
function _geqp3_norms!(A::AbstractMatrix{T}, m::Int, n::Int, vn1, vn2) where {T <: BlasReal}
    @inbounds for j in 1:n
        ss = zero(T)
        @simd for r in 1:m
            ss = muladd(A[r, j], A[r, j], ss)
        end
        nrm = sqrt(ss)
        if !isfinite(nrm) || ss < floatmin(T)
            nrm = _nrm2(m, view(A, :, j), 1)
        end
        vn1[j] = nrm; vn2[j] = nrm
    end
    return
end

function geqp3!(
        A::AbstractMatrix{T}, jpvt::AbstractVector{<:Integer},
        tau::AbstractVector{T}
    ) where {T <: BlasFloat}
    m, n = size(A); k = min(m, n); R = real(T)
    length(jpvt) >= n || _throw_len_jpvt(:geqp3!, n)
    length(tau) >= k || _throw_len_tau_mn(:geqp3!, k)
    @inbounds for j in 1:n
        jpvt[j] = j
    end
    n == 0 && return A, jpvt, tau
    tol3z = sqrt(eps(R))                                   # dlaqps recompute threshold (√ machine-eps)
    # ONE scope for the whole routine. vn1/vn2/hbuf are REAL and are borrowed together because all three
    # are needed on EVERY path; the blocked-panel trio is borrowed from the SAME token further down,
    # inside the `nb > 1` arm, so it costs nothing on the paths that never take it. Being REAL next to
    # element-typed borrows needs no argument now — one arena, disjoint byte ranges — where the fields
    # needed "they live on a different `_l3ws` owner".
    # ESCAPE AUDIT: every handle is passed into `_geqp3_norms!` / `_laqps!` / `_geqp3_house_dd!` and
    # written through; `geqp3!` returns `A, jpvt, tau`, all caller-owned.
    @scope arn begin
        vn1 = borrow!(arn, R, n)
        vn2 = borrow!(arn, R, n)
        hbuf = borrow!(arn, R, n)
        if T <: BlasReal && R === T && _strided1(A)
            _geqp3_norms!(A, m, n, vn1, vn2)                   # fused fast pass, _nrm2 fallback guard
        else
            @inbounds for j in 1:n
                nrm = _nrm2(m, view(A, :, j), 1); vn1[j] = nrm; vn2[j] = nrm
            end
        end
        # blocked dlaqps panels (laqps.jl) — real strided above the unblocked crossover; unblocked loop
        # below factors the tail and remains the fallback (also the complex / non-strided path).
        j0 = 1
        if T <: BlasReal && R === T && _strided1(A) && k > _QR_UNBLK_MAX
            nb = clamp(_qr_nb(m, n), 1, k)
            if nb > 1
                # Blocked-panel scratch, borrowed from the enclosing scope. Reached only when `nb > 1`,
                # which is why an allocation audit at a single small n cannot see it (F/auxv/wrow are
                # ~94% of the bytes at n=129). ABOVE the `while` panel loop, where it already sat — the
                # shape does not depend on the panel index, so nothing needed hoisting. `F` is EXACTLY
                # n×nb (`ld == n`), matching the incumbent field's unpadded stride policy.
                F = borrow!(arn, T, n, nb)
                auxv = borrow!(arn, T, nb)
                wrow = borrow!(arn, T, n)
                while k - j0 + 1 > nb
                    jb = min(nb, k - j0 + 1)
                    fjb = _laqps!(
                        m, n - j0 + 1, j0 - 1, jb, view(A, 1:m, j0:n),
                        view(jpvt, j0:n), view(tau, j0:k),
                        view(vn1, j0:n), view(vn2, j0:n), auxv, view(F, 1:(n - j0 + 1), 1:jb), wrow
                    )
                    fjb <= 0 && break
                    j0 += fjb
                end
            end
        end
        if T <: BlasReal && R === T && _strided1(A)
            # fast unblocked loop: same pivot/swap/larfg, apply+downdate via _geqp3_house_dd! (SIMD
            # downdate, bitwise-identical results — see the kernel header for the measurements).
            # NOTE (measured 2026-08-02, Zen4): with this loop the UNBLOCKED path also beats the blocked
            # path at n=48/64/96 (1.33×/1.24×/1.05×), crossing only at n=128 (0.96×). _QR_UNBLK_MAX
            # stays 32 — no gate cell sits in 48..96 and the crossover is unmeasured off Zen4; a
            # geqp3-specific derived crossover is a follow-up lever, not assumed here.
            @inbounds for i in j0:k
                pvt = i; maxn = vn1[i]
                for j in (i + 1):n
                    if vn1[j] > maxn
                        maxn = vn1[j]; pvt = j
                    end
                end
                if pvt != i
                    for r in 1:m
                        t = A[r, i]; A[r, i] = A[r, pvt]; A[r, pvt] = t
                    end
                    jpvt[i], jpvt[pvt] = jpvt[pvt], jpvt[i]
                    vn1[pvt] = vn1[i]; vn2[pvt] = vn2[i]
                end
                β, τ = _larfg!(view(A, i:m, i)); tau[i] = τ
                A[i, i] = β                                    # _larfg! leaves x[1]=α; place R's diagonal
                i < n && _geqp3_house_dd!(A, i, m, n, τ, hbuf, vn1, vn2, tol3z)
            end
            return A, jpvt, tau
        end
        @inbounds for i in j0:k
            # ---- pivot: column of maximal partial norm over i:n → swap to position i ----
            pvt = i; maxn = vn1[i]
            for j in (i + 1):n
                if vn1[j] > maxn
                    maxn = vn1[j]; pvt = j
                end
            end
            if pvt != i
                for r in 1:m
                    t = A[r, i]; A[r, i] = A[r, pvt]; A[r, pvt] = t
                end
                jpvt[i], jpvt[pvt] = jpvt[pvt], jpvt[i]
                vn1[pvt] = vn1[i]; vn2[pvt] = vn2[i]
            end
            # ---- reflector for column i (rows i:m); β lands on the diagonal ----
            β, τ = _larfg!(view(A, i:m, i)); tau[i] = τ
            A[i, i] = β                                        # _larfg! leaves x[1]=α; place R's diagonal
            # ---- apply H_iᴴ to the trailing columns; then downdate their partial norms ----
            if i < n
                _house_left!(view(A, i:m, (i + 1):n), view(A, i:m, i), _geqp3_tau_Qh(τ))
                for j in (i + 1):n
                    if !iszero(vn1[j])
                        temp = one(R) - (abs(A[i, j]) / vn1[j])^2
                        temp = max(temp, zero(R))
                        temp2 = temp * (vn1[j] / vn2[j])^2
                        if temp2 <= tol3z                      # downdated norm degraded → recompute exactly
                            if i < m
                                nrm = _nrm2(m - i, view(A, (i + 1):m, j), 1); vn1[j] = nrm; vn2[j] = nrm
                            else
                                vn1[j] = zero(R); vn2[j] = zero(R)
                            end
                        else
                            vn1[j] = vn1[j] * sqrt(temp)
                        end
                    end
                end
            end
        end
        return A, jpvt, tau
    end
end

# Convenience: allocate jpvt + tau, return (A overwritten, jpvt, tau).
function geqp3!(A::AbstractMatrix{T}) where {T <: BlasFloat}
    m, n = size(A)
    jpvt = Vector{Int}(undef, n); tau = Vector{T}(undef, min(m, n))
    geqp3!(A, jpvt, tau)
    return A, jpvt, tau
end
