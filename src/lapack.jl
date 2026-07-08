using LinearAlgebra: PosDefException

# LAPACK-level routines built on the gated Level-3 BLAS. First: Cholesky (potrf).
# Real symmetric positive-definite A = L·Lᵀ (uplo='L') or A = Uᵀ·U (uplo='U'), factored in place into
# the `uplo` triangle. Right-looking BLOCKED algorithm: each NB diagonal block is factored by the
# unblocked `_potf2` base, then the gated trsm (panel solve) + syrk (trailing rank-NB update) carry the
# bulk. Generic over T<:Real (the unblocked base + the generic trsm/syrk path make it ForwardDiff-
# traceable); BlasReal hits the SIMD trsm/syrk. ponytail: NB hand-set for Zen4; lift to a knob if tuning.

const _POTRF_BASE = 512    # recurse above this; below, the unblocked base (potf2, vectorized inner loop).
# Complex has NO fast SIMD base (the scalar potf2 above), so a 512 base = the whole factorization is scalar
# (measured: zpotrf n≤512 = 0.15-0.49× — all base, no recursion). A small base hands the bulk to the fast
# complex ztrsm!/zherk! recursion. Retune per box; Preferences knob "cpotrf_base".
const _CPOTRF_BASE = @load_preference("cpotrf_base", _vwidth(Float64) == 4 ? 48 : 64)::Int
# n≤base ⇒ ONE vectorized `_cpotf2_lower!` call; n>base ⇒ right-looking blocked (see _cpotrf_rl_lower!).
# Base sweet spot is where the unblocked SIMD base still gates: AVX-512 (W=8) rides it to 64 (n=64 gates
# 1.09), but on AVX2 (W=4) the narrower datapath makes the n=64 base memory-bound (0.76), so cap it at 32
# (n≤32 gates, n>32 → rl). Keyed on _vwidth like the sibling cuts. Larger bases go memory-bound unblocked
# unblocked (base=128 → n=128 0.72), smaller pay recursion/small-k overhead — mirrors the real path's
# _CHOL_THRESHOLD=64. ponytail: flat 64 like the real threshold; galen(AVX2)/zen5 calibration via the knob.

# Contiguous scratch for the diagonal base block: the recursion's base is a view(A, js, js) whose
# columns are parent_ld apart (poor locality, the memory-bound potf2). Copying it to a contiguous
# buffer, factoring there, and copying back streams contiguous memory (better prefetch/TLB).
# _potf2_buf (potrf diagonal-base contiguous buffer) is the per-type L3Workspace `potf2` field
# (see src/workspace.jl).

# Unblocked right-looking Cholesky of an n×n block, lower triangle. Throws PosDefException at the first
# non-positive pivot (LAPACK's info>0). Reads/writes only the lower triangle.
# Hermitian-aware: `real(A[j,j])` (the diagonal is real for A=LLᴴ) and `conj` on the downdate operand.
# On T<:Real both are identities (compile away) → the real/ForwardDiff path is byte-identical; on
# T<:Complex this is the Hermitian Cholesky (zpotrf), matching LAPACK zpotf2.
function _potf2_lower!(A, n::Int)
    @inbounds for j in 1:n
        d = real(A[j, j])
        d > 0 || throw(PosDefException(j))
        ajj = sqrt(d); A[j, j] = ajj; invd = inv(ajj)
        for i in (j + 1):n; A[i, j] *= invd; end                 # scale column j below the diagonal
        for k in (j + 1):n                                        # rank-1 update of the lower trailing
            akj = conj(A[k, j])                                   # Hermitian: L[i,j]·conj(L[k,j])
            for i in k:n; A[i, k] -= A[i, j] * akj; end
        end
    end
    return A
end
# Unblocked, upper triangle: A = Uᴴ·U (Hermitian). conj on the mirror operand A[j,i].
function _potf2_upper!(A, n::Int)
    @inbounds for j in 1:n
        d = real(A[j, j])
        d > 0 || throw(PosDefException(j))
        ajj = sqrt(d); A[j, j] = ajj; invd = inv(ajj)
        for i in (j + 1):n; A[j, i] *= invd; end                 # scale row j right of the diagonal
        for k in (j + 1):n                                        # rank-1 update of the upper trailing
            ajk = A[j, k]
            for i in (j + 1):k; A[i, k] -= conj(A[j, i]) * ajk; end   # Hermitian: conj(U[j,i])·U[j,k]
        end
    end
    return A
end

# Base-update row-unroll (MR). MR=2 (two W-blocks/step, 1 L[j,k] load / 2 blocks → more ILP) LIFTS the
# memory-bound base on DOUBLE-PUMPED AVX-512 (Zen4) + AVX2 (Zen3) — n=48 0.97→1.18, n=64 1.11→1.17 — but
# REGRESSES native-512 (Zen5) where MR=1 already saturates. The discriminator is L1D size: 32K on the
# double-pump/AVX2 boxes, 48K on Zen5 (and Intel Tiger/Ice Lake+) native-512 — so key MR on `_L1_BYTES`
# (Zen4 fam 25 / Zen5 fam 26 also differ, but L1D is the causal-adjacent cache signal + already a const).
const _CPOTF2_MR = @load_preference("cpotf2_mr", _L1_BYTES < 49152 ? 2 : 1)::Int

# Vectorized complex Hermitian Cholesky base (lower, A = L·Lᴴ). Complex analogue of `_chol_base_f64!`:
# left-looking, SIMD over i (W complex per step, deinterleaved re/im FMA chains), scalar tail. Column j:
# A[i,j] -= Σ_{k<j} L[i,k]·conj(L[j,k]) (i=j..n), then real diagonal d=A[j,j], scale below-diag by 1/√d.
# Reads only the lower triangle. Throws PosDefException at the first non-positive pivot (LAPACK zpotf2).
# The scalar `_potf2_lower!` above stays for Float32/Dual/other T (this method is BlasComplex-only).
function _cpotf2_lower!(A::AbstractMatrix{Tc}, n::Int) where {Tc <: BlasComplex}
    Tr = real(Tc); W = _vwidth(Tr); V = Vec{W, Tr}; V2 = Vec{2W, Tr}; sz = sizeof(Tr); ld = stride(A, 2)
    GC.@preserve A begin
        p = Ptr{Tr}(pointer(A))
        cx(i, k) = p + ((k - 1) * ld + (i - 1)) * 2 * sz                 # byte Ptr to the re-part of A[i,k]
        @inbounds for j in 1:n
            i = j
            if _CPOTF2_MR >= 2                                           # MR=2 (double-pump/AVX2): 2 W-blocks / step
                while i + 2W - 1 <= n
                    b0 = cx(i, j); b1 = cx(i + W, j)
                    (ar0, ai0) = _deint_cmplx(vload(V2, b0)); (ar1, ai1) = _deint_cmplx(vload(V2, b1))
                    for k in 1:j-1
                        l = cx(j, k); sr = V(unsafe_load(l)); si = V(unsafe_load(l + sz))   # L[j,k], 1 load / 2 blocks
                        (vr0, vi0) = _deint_cmplx(vload(V2, cx(i, k))); (vr1, vi1) = _deint_cmplx(vload(V2, cx(i + W, k)))
                        ar0 = muladd(vr0, -sr, ar0); ar0 = muladd(vi0, -si, ar0); ai0 = muladd(vi0, -sr, ai0); ai0 = muladd(vr0, si, ai0)
                        ar1 = muladd(vr1, -sr, ar1); ar1 = muladd(vi1, -si, ar1); ai1 = muladd(vi1, -sr, ai1); ai1 = muladd(vr1, si, ai1)
                    end
                    vstore(_intlv_cmplx(ar0, ai0), b0); vstore(_intlv_cmplx(ar1, ai1), b1); i += 2W
                end
            end
            while i + W - 1 <= n                                         # SIMD: W complex rows / step
                base = cx(i, j); (ar, ai) = _deint_cmplx(vload(V2, base))
                for k in 1:j-1
                    l = cx(j, k); sr = V(unsafe_load(l)); si = V(unsafe_load(l + sz))   # L[j,k]
                    (vr, vi) = _deint_cmplx(vload(V2, cx(i, k)))         # -v·conj(L[j,k])
                    ar = muladd(vr, -sr, ar); ar = muladd(vi, -si, ar)
                    ai = muladd(vi, -sr, ai); ai = muladd(vr, si, ai)
                end
                vstore(_intlv_cmplx(ar, ai), base); i += W
            end
            while i <= n                                                # scalar tail
                sr = unsafe_load(cx(i, j)); si = unsafe_load(cx(i, j) + sz)
                for k in 1:j-1
                    l = cx(j, k); jr = unsafe_load(l); ji = unsafe_load(l + sz)
                    vr = unsafe_load(cx(i, k)); vi = unsafe_load(cx(i, k) + sz)
                    sr -= vr * jr + vi * ji; si -= vi * jr - vr * ji
                end
                unsafe_store!(cx(i, j), sr); unsafe_store!(cx(i, j) + sz, si); i += 1
            end
            d = unsafe_load(cx(j, j))                                    # diagonal is real (Hermitian)
            d > 0 || throw(PosDefException(j))
            ajj = sqrt(d); invd = inv(ajj)
            unsafe_store!(cx(j, j), ajj); unsafe_store!(cx(j, j) + sz, zero(Tr))
            i = j + 1                                                   # scale below-diag by 1/√d (real)
            while i + W - 1 <= n
                base = cx(i, j); vstore(vload(V2, base) * V2(invd), base); i += W
            end
            while i <= n
                unsafe_store!(cx(i, j), unsafe_load(cx(i, j)) * invd)
                unsafe_store!(cx(i, j) + sz, unsafe_load(cx(i, j) + sz) * invd); i += 1
            end
        end
    end
    return A
end

# Recursive (cache-oblivious) Cholesky. Lower: split 2×2 — factor A11, solve the off-diagonal panel
# A21·L11⁻ᵀ (trsm), downdate A22 -= A21·A21ᵀ (syrk), recurse A22. The top-level trsm/syrk are large-k
# (half-matrix → the gated packed L3 paths); only the ≤_POTRF_BASE diagonal base is scalar potf2.
# Factor a base block, via a contiguous buffer when A is a strided sub-block (better locality).
@inline function _potf2b_lower!(A, n::Int)
    if eltype(A) <: BlasComplex                                         # SIMD Hermitian base (via contig buf if strided)
        if A isa SubArray && stride(A, 2) > n && n >= 8
            buf = _potf2_buf(eltype(A), n); copyto!(buf, A); _cpotf2_lower!(buf, n); copyto!(A, buf); return A
        end
        return _cpotf2_lower!(A, n)
    end
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
    if eltype(A) <: Complex                                    # Hermitian: A21·L11⁻ᴴ + A22 -= A21·A21ᴴ
        trsm!(A21, view(A, 1:h, 1:h); side = 'R', uplo = 'L', transA = 'C', diag = 'N', alpha = true)
        herk!(view(A, (h + 1):n, (h + 1):n), A21; uplo = 'L', trans = 'N', alpha = -1.0, beta = 1.0)
    else
        trsm!(A21, view(A, 1:h, 1:h); side = 'R', uplo = 'L', transA = 'T', diag = 'N', alpha = true)
        syrk!(view(A, (h + 1):n, (h + 1):n), A21; uplo = 'L', trans = 'N', alpha = -1, beta = 1)
    end
    _potrf_lower!(view(A, (h + 1):n, (h + 1):n), n - h, base)
    return A
end
# Upper: A = Uᵀ·U. Off-diagonal panel A12 = U11⁻ᵀ·A12 (trsm side-L), downdate A22 -= A12ᵀ·A12 (syrk).
function _potrf_upper!(A, n::Int, base::Int = _POTRF_BASE)
    n <= base && return _potf2b_upper!(A, n)
    h = n ÷ 2
    _potrf_upper!(view(A, 1:h, 1:h), h)
    A12 = view(A, 1:h, (h + 1):n)
    if eltype(A) <: Complex                                    # Hermitian: U11⁻ᴴ·A12 + A22 -= A12ᴴ·A12
        trsm!(A12, view(A, 1:h, 1:h); side = 'L', uplo = 'U', transA = 'C', diag = 'N', alpha = true)
        herk!(view(A, (h + 1):n, (h + 1):n), A12; uplo = 'U', trans = 'C', alpha = -1.0, beta = 1.0)
    else
        trsm!(A12, view(A, 1:h, 1:h); side = 'L', uplo = 'U', transA = 'T', diag = 'N', alpha = true)
        syrk!(view(A, (h + 1):n, (h + 1):n), A12; uplo = 'U', trans = 'T', alpha = -1, beta = 1)
    end
    _potrf_upper!(view(A, (h + 1):n, (h + 1):n), n - h, base)
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
# Split the base k-reduction into 6 independent FMA chains (vs 3) — pays off only where the reduction is
# latency-bound: Haswell-class Intel AVX2 (narrow OOO). Auto-on there, off on Zen/AVX-512 (their OOO hides
# the chain — measured slight regression), overridable. See [[_INTEL_AVX2]] in cpuinfo.jl.
const _CHOL_BASE_SPLIT = @load_preference("chol_base_split", _INTEL_AVX2)::Bool
@inline _clidx(i, k, ld) = (k - 1) * ld + i                              # 1-based linear index
@inline _cvptr(p, i, k, ld) = p + (((k - 1) * ld + (i - 1)) * sizeof(Float64))   # byte Ptr to [i,k]

# base case (n ≤ threshold): left-looking SIMD panel Cholesky, ascending-k FMA, scale by 1/√diag.
function _chol_base_f64!(p::Ptr{Float64}, n::Int, ld::Int)
    @inbounds for j in 1:n
        i = j
        while i + 3_CHOLW - 1 <= n                              # MR=3 row-vectors
            b0 = _cvptr(p, i, j, ld); b1 = _cvptr(p, i + _CHOLW, j, ld); b2 = _cvptr(p, i + 2_CHOLW, j, ld)
            a0 = vload(_CVF, b0); a1 = vload(_CVF, b1); a2 = vload(_CVF, b2)
            if _CHOL_BASE_SPLIT
                # Haswell-class: split each row-block's k-reduction into even/odd partials → 6 independent
                # FMA chains so the 2 units aren't starved by the 5-cyc latency (llvm-mca: 10→5 cyc/iter).
                # Reassociates the dot product (not bit-identical to faer, still OpenBLAS-correct).
                d0 = zero(_CVF); d1 = zero(_CVF); d2 = zero(_CVF)
                kk = 1
                while kk + 1 <= j - 1
                    g = _CVF(-unsafe_load(p, _clidx(j, kk, ld))); h = _CVF(-unsafe_load(p, _clidx(j, kk + 1, ld)))
                    a0 = muladd(g, vload(_CVF, _cvptr(p, i, kk, ld)), a0)
                    d0 = muladd(h, vload(_CVF, _cvptr(p, i, kk + 1, ld)), d0)
                    a1 = muladd(g, vload(_CVF, _cvptr(p, i + _CHOLW, kk, ld)), a1)
                    d1 = muladd(h, vload(_CVF, _cvptr(p, i + _CHOLW, kk + 1, ld)), d1)
                    a2 = muladd(g, vload(_CVF, _cvptr(p, i + 2_CHOLW, kk, ld)), a2)
                    d2 = muladd(h, vload(_CVF, _cvptr(p, i + 2_CHOLW, kk + 1, ld)), d2)
                    kk += 2
                end
                if kk <= j - 1                                  # odd tail k
                    g = _CVF(-unsafe_load(p, _clidx(j, kk, ld)))
                    a0 = muladd(g, vload(_CVF, _cvptr(p, i, kk, ld)), a0)
                    a1 = muladd(g, vload(_CVF, _cvptr(p, i + _CHOLW, kk, ld)), a1)
                    a2 = muladd(g, vload(_CVF, _cvptr(p, i + 2_CHOLW, kk, ld)), a2)
                end
                a0 += d0; a1 += d1; a2 += d2
            else                                               # Zen / AVX-512: OOO hides the chain → keep 3
                for k in 1:j-1
                    g = _CVF(-unsafe_load(p, _clidx(j, k, ld)))
                    a0 = muladd(g, vload(_CVF, _cvptr(p, i, k, ld)), a0)
                    a1 = muladd(g, vload(_CVF, _cvptr(p, i + _CHOLW, k, ld)), a1)
                    a2 = muladd(g, vload(_CVF, _cvptr(p, i + 2_CHOLW, k, ld)), a2)
                end
            end
            vstore(a0, b0); vstore(a1, b1); vstore(a2, b2); i += 3_CHOLW
        end
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
# Pad when columns alias L1 sets: Zen L1 = 64 sets × 64 B, so stride·8 a multiple of 64·64=4096 B
# (stride % 512 == 0) maps every column to the same sets. %256 = half-period (2 cols/set), %128 =
# quarter-period (4 cols/set) — both thrash L1. But the pad is an n² copy round-trip, so it only wins
# when it's cheap vs the aliasing it removes: strong (%256) aliasing always, OR quarter-period (%128)
# only when the matrix is L2-resident (copy cheap, L1-aliasing-dominated). Measured on Zen3: n=128
# (128 KB, %128) 0.887→1.018 (pad wins); n=384 (1.1 MB > L2, %128 not %256) 0.918→0.899 (pad LOSES).
@inline _chol_needs_pad(A, n) = n >= 128 && stride(A, 2) % 128 == 0 &&
    (stride(A, 2) % 256 == 0 || n * n <= _L2_BYTES ÷ 8)

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

# ── Fused panel driver: po2-strided AVX2 potrf without the whole-matrix pad round-trip ──────────────
# Measured (galen/Zen3, kb pureblas-cholesky): the po2-stride tax lives in the trsm B-panel and the
# faer base reading A directly (syrk! packs both operands — stride-immune), and the whole-pad fix pays
# an n² copy round-trip that IS the residual gate gap at n=256–1024. Fix: per NB=128 block, (1) factor
# the diagonal block in a conflict-free scratch D, (2) solve the panel INTO a conflict-free workspace T
# with a split-ld trsm whose FIRST TOUCH reads po2 A21 as its initial operand load (the copy-in is
# fused away — zero extra traffic), (3) update the trailing from T (cache-resident @inline syrk when
# the T slab fits L2, packed syrk! reading T otherwise), (4) stream T back to A21 exactly ONCE (the
# factor's own output write). The @inline split kernels compose in ONE function body so T never
# round-trips through A21 between the trsm and the syrk. AVX-512 and non-po2 stay on the paths above.

# Split-ld faer trsm panel solve: L10·L00ᵀ = A10 with p00 (diag factor) at ld0, the panel SOLVED INTO
# pT at ldt, and the po2 source psrc at lds read exactly once (each column's initial load, before its
# first update). Same math/order as _trsm_right_lower_f64! — the c0==1 register pass degenerates to
# the first-touch copy (empty k-loop), fusing the copy-in.
@inline function _trsm_rl_split_f64!(p00::Ptr{Float64}, ld0::Int, psrc::Ptr{Float64}, lds::Int,
                                     pT::Ptr{Float64}, ldt::Int, bs::Int, m::Int)
    c0 = 1
    @inbounds while c0 <= bs
        nb = min(_CHOL_NB, bs - c0 + 1)
        if nb == _CHOL_NB
            i = 1
            while i + 3_CHOLW - 1 <= m                            # MR=3 × NC=4 = 12 accumulators —
                r1 = i + _CHOLW; r2 = i + 2_CHOLW                 # 7 loads/12 FMAs (FMA-bound; the
                a00 = vload(_CVF, _cvptr(psrc, i, c0, lds))       # MR=1 pass was 5 loads/4 FMAs,
                a01 = vload(_CVF, _cvptr(psrc, i, c0 + 1, lds))   # load-port-bound — measured +25%
                a02 = vload(_CVF, _cvptr(psrc, i, c0 + 2, lds))   # vs packed trsm!). First touch:
                a03 = vload(_CVF, _cvptr(psrc, i, c0 + 3, lds))   # po2 A21, read exactly once.
                a10 = vload(_CVF, _cvptr(psrc, r1, c0, lds))
                a11 = vload(_CVF, _cvptr(psrc, r1, c0 + 1, lds))
                a12 = vload(_CVF, _cvptr(psrc, r1, c0 + 2, lds))
                a13 = vload(_CVF, _cvptr(psrc, r1, c0 + 3, lds))
                a20 = vload(_CVF, _cvptr(psrc, r2, c0, lds))
                a21 = vload(_CVF, _cvptr(psrc, r2, c0 + 1, lds))
                a22 = vload(_CVF, _cvptr(psrc, r2, c0 + 2, lds))
                a23 = vload(_CVF, _cvptr(psrc, r2, c0 + 3, lds))
                for k in 1:c0-1                                   # solved columns: conflict-free T
                    v0 = vload(_CVF, _cvptr(pT, i, k, ldt)); v1 = vload(_CVF, _cvptr(pT, r1, k, ldt)); v2 = vload(_CVF, _cvptr(pT, r2, k, ldt))
                    g = _CVF(-unsafe_load(p00, _clidx(c0, k, ld0)));     a00 = muladd(g, v0, a00); a10 = muladd(g, v1, a10); a20 = muladd(g, v2, a20)
                    g = _CVF(-unsafe_load(p00, _clidx(c0 + 1, k, ld0))); a01 = muladd(g, v0, a01); a11 = muladd(g, v1, a11); a21 = muladd(g, v2, a21)
                    g = _CVF(-unsafe_load(p00, _clidx(c0 + 2, k, ld0))); a02 = muladd(g, v0, a02); a12 = muladd(g, v1, a12); a22 = muladd(g, v2, a22)
                    g = _CVF(-unsafe_load(p00, _clidx(c0 + 3, k, ld0))); a03 = muladd(g, v0, a03); a13 = muladd(g, v1, a13); a23 = muladd(g, v2, a23)
                end
                vstore(a00, _cvptr(pT, i, c0, ldt));      vstore(a01, _cvptr(pT, i, c0 + 1, ldt))
                vstore(a02, _cvptr(pT, i, c0 + 2, ldt));  vstore(a03, _cvptr(pT, i, c0 + 3, ldt))
                vstore(a10, _cvptr(pT, r1, c0, ldt));     vstore(a11, _cvptr(pT, r1, c0 + 1, ldt))
                vstore(a12, _cvptr(pT, r1, c0 + 2, ldt)); vstore(a13, _cvptr(pT, r1, c0 + 3, ldt))
                vstore(a20, _cvptr(pT, r2, c0, ldt));     vstore(a21, _cvptr(pT, r2, c0 + 1, ldt))
                vstore(a22, _cvptr(pT, r2, c0 + 2, ldt)); vstore(a23, _cvptr(pT, r2, c0 + 3, ldt))
                i += 3_CHOLW
            end
            while i + _CHOLW - 1 <= m                             # MR=1 tail rows
                a0 = vload(_CVF, _cvptr(psrc, i, c0, lds))
                a1 = vload(_CVF, _cvptr(psrc, i, c0 + 1, lds))
                a2 = vload(_CVF, _cvptr(psrc, i, c0 + 2, lds))
                a3 = vload(_CVF, _cvptr(psrc, i, c0 + 3, lds))
                for k in 1:c0-1
                    vk = vload(_CVF, _cvptr(pT, i, k, ldt))
                    a0 = muladd(_CVF(-unsafe_load(p00, _clidx(c0, k, ld0))), vk, a0)
                    a1 = muladd(_CVF(-unsafe_load(p00, _clidx(c0 + 1, k, ld0))), vk, a1)
                    a2 = muladd(_CVF(-unsafe_load(p00, _clidx(c0 + 2, k, ld0))), vk, a2)
                    a3 = muladd(_CVF(-unsafe_load(p00, _clidx(c0 + 3, k, ld0))), vk, a3)
                end
                vstore(a0, _cvptr(pT, i, c0, ldt));     vstore(a1, _cvptr(pT, i, c0 + 1, ldt))
                vstore(a2, _cvptr(pT, i, c0 + 2, ldt)); vstore(a3, _cvptr(pT, i, c0 + 3, ldt))
                i += _CHOLW
            end
            while i <= m
                for dj in 0:_CHOL_NB-1
                    cc = c0 + dj; s = unsafe_load(psrc, _clidx(i, cc, lds))
                    for k in 1:c0-1
                        s = muladd(-unsafe_load(p00, _clidx(cc, k, ld0)), unsafe_load(pT, _clidx(i, k, ldt)), s)
                    end
                    unsafe_store!(pT, s, _clidx(i, cc, ldt))
                end
                i += 1
            end
        else
            for dj in 0:nb-1                                      # remainder columns (<NB)
                cc = c0 + dj
                i = 1
                while i + _CHOLW - 1 <= m
                    a = vload(_CVF, _cvptr(psrc, i, cc, lds))
                    for k in 1:c0-1
                        a = muladd(_CVF(-unsafe_load(p00, _clidx(cc, k, ld0))), vload(_CVF, _cvptr(pT, i, k, ldt)), a)
                    end
                    vstore(a, _cvptr(pT, i, cc, ldt)); i += _CHOLW
                end
                while i <= m
                    s = unsafe_load(psrc, _clidx(i, cc, lds))
                    for k in 1:c0-1
                        s = muladd(-unsafe_load(p00, _clidx(cc, k, ld0)), unsafe_load(pT, _clidx(i, k, ldt)), s)
                    end
                    unsafe_store!(pT, s, _clidx(i, cc, ldt)); i += 1
                end
            end
        end
        for dj in 0:nb-1                                  # within-panel triangular solve + scale, on T
            c = c0 + dj; invc = 1.0 / unsafe_load(p00, _clidx(c, c, ld0)); vinv = _CVF(invc)
            i = 1
            while i + _CHOLW - 1 <= m
                o = _cvptr(pT, i, c, ldt); a = vload(_CVF, o)
                for k in c0:c-1
                    a = muladd(_CVF(-unsafe_load(p00, _clidx(c, k, ld0))), vload(_CVF, _cvptr(pT, i, k, ldt)), a)
                end
                vstore(a * vinv, o); i += _CHOLW
            end
            while i <= m
                s = unsafe_load(pT, _clidx(i, c, ldt))
                for k in c0:c-1
                    s = muladd(-unsafe_load(p00, _clidx(c, k, ld0)), unsafe_load(pT, _clidx(i, k, ldt)), s)
                end
                unsafe_store!(pT, s * invc, _clidx(i, c, ldt)); i += 1
            end
        end
        c0 += _CHOL_NB
    end
    return nothing
end

# Split-ld faer syrk column j: A11[i,j] (at ld1) −= Σ_c T[j,c]·T[i,c] (T at ldt).
@inline function _syrk_panel_split_f64!(p11, ld1::Int, pT, ldt::Int, j::Int, m::Int, bs::Int)
    i = ((j - 1) ÷ _CHOLW) * _CHOLW + 1
    @inbounds while i + _CHOLW - 1 <= m
        b = _cvptr(p11, i, j, ld1); a = vload(_CVF, b)
        for c in 1:bs
            a = muladd(_CVF(-unsafe_load(pT, _clidx(j, c, ldt))), vload(_CVF, _cvptr(pT, i, c, ldt)), a)
        end
        vstore(a, b); i += _CHOLW
    end
    @inbounds while i <= m
        s = unsafe_load(p11, _clidx(i, j, ld1))
        for c in 1:bs
            s = muladd(-unsafe_load(pT, _clidx(j, c, ldt)), unsafe_load(pT, _clidx(i, c, ldt)), s)
        end
        unsafe_store!(p11, s, _clidx(i, j, ld1)); i += 1
    end
end

# Split-ld faer trailing update: A11 (m×m at ld1, po2 is fine — registers carry the RMW across the
# k-loop) −= T·Tᵀ with the PANEL read from conflict-free T at ldt. Body = _syrk_lower_f64! with the
# two operands' lds split.
@inline function _syrk_lower_split_f64!(p11::Ptr{Float64}, ld1::Int, pT::Ptr{Float64}, ldt::Int,
                                        m::Int, bs::Int)
    j = 1
    @inbounds while j + _CHOL_NC - 1 <= m
        i = ((j - 1) ÷ _CHOLW) * _CHOLW + 1
        while i + 3_CHOLW - 1 <= m                          # MR=3 × NC=4 = 12 accumulators
            r1 = i + _CHOLW; r2 = i + 2_CHOLW
            e00 = _cvptr(p11, i, j, ld1);      A00 = vload(_CVF, e00)
            e10 = _cvptr(p11, r1, j, ld1);     C00 = vload(_CVF, e10)
            e20 = _cvptr(p11, r2, j, ld1);     D00 = vload(_CVF, e20)
            e01 = _cvptr(p11, i, j + 1, ld1);  A01 = vload(_CVF, e01)
            e11 = _cvptr(p11, r1, j + 1, ld1); C01 = vload(_CVF, e11)
            e21 = _cvptr(p11, r2, j + 1, ld1); D01 = vload(_CVF, e21)
            e02 = _cvptr(p11, i, j + 2, ld1);  A02 = vload(_CVF, e02)
            e12 = _cvptr(p11, r1, j + 2, ld1); C02 = vload(_CVF, e12)
            e22 = _cvptr(p11, r2, j + 2, ld1); D02 = vload(_CVF, e22)
            e03 = _cvptr(p11, i, j + 3, ld1);  A03 = vload(_CVF, e03)
            e13 = _cvptr(p11, r1, j + 3, ld1); C03 = vload(_CVF, e13)
            e23 = _cvptr(p11, r2, j + 3, ld1); D03 = vload(_CVF, e23)
            for c in 1:bs
                v0 = vload(_CVF, _cvptr(pT, i, c, ldt)); v1 = vload(_CVF, _cvptr(pT, r1, c, ldt)); v2 = vload(_CVF, _cvptr(pT, r2, c, ldt))
                g0 = _CVF(-unsafe_load(pT, _clidx(j, c, ldt)));     A00 = muladd(g0, v0, A00); C00 = muladd(g0, v1, C00); D00 = muladd(g0, v2, D00)
                g1 = _CVF(-unsafe_load(pT, _clidx(j + 1, c, ldt))); A01 = muladd(g1, v0, A01); C01 = muladd(g1, v1, C01); D01 = muladd(g1, v2, D01)
                g2 = _CVF(-unsafe_load(pT, _clidx(j + 2, c, ldt))); A02 = muladd(g2, v0, A02); C02 = muladd(g2, v1, C02); D02 = muladd(g2, v2, D02)
                g3 = _CVF(-unsafe_load(pT, _clidx(j + 3, c, ldt))); A03 = muladd(g3, v0, A03); C03 = muladd(g3, v1, C03); D03 = muladd(g3, v2, D03)
            end
            vstore(A00, e00); vstore(A01, e01); vstore(A02, e02); vstore(A03, e03)
            vstore(C00, e10); vstore(C01, e11); vstore(C02, e12); vstore(C03, e13)
            vstore(D00, e20); vstore(D01, e21); vstore(D02, e22); vstore(D03, e23)
            i += 3_CHOLW
        end
        while i + 2_CHOLW - 1 <= m                          # MR=2 × NC=4 = 8 accumulators
            r1 = i + _CHOLW
            d00 = _cvptr(p11, i, j, ld1);      A00 = vload(_CVF, d00)
            d10 = _cvptr(p11, r1, j, ld1);     B00 = vload(_CVF, d10)
            d01 = _cvptr(p11, i, j + 1, ld1);  A01 = vload(_CVF, d01)
            d11 = _cvptr(p11, r1, j + 1, ld1); B01 = vload(_CVF, d11)
            d02 = _cvptr(p11, i, j + 2, ld1);  A02 = vload(_CVF, d02)
            d12 = _cvptr(p11, r1, j + 2, ld1); B02 = vload(_CVF, d12)
            d03 = _cvptr(p11, i, j + 3, ld1);  A03 = vload(_CVF, d03)
            d13 = _cvptr(p11, r1, j + 3, ld1); B03 = vload(_CVF, d13)
            for c in 1:bs
                v0 = vload(_CVF, _cvptr(pT, i, c, ldt)); v1 = vload(_CVF, _cvptr(pT, r1, c, ldt))
                g0 = _CVF(-unsafe_load(pT, _clidx(j, c, ldt)));     A00 = muladd(g0, v0, A00); B00 = muladd(g0, v1, B00)
                g1 = _CVF(-unsafe_load(pT, _clidx(j + 1, c, ldt))); A01 = muladd(g1, v0, A01); B01 = muladd(g1, v1, B01)
                g2 = _CVF(-unsafe_load(pT, _clidx(j + 2, c, ldt))); A02 = muladd(g2, v0, A02); B02 = muladd(g2, v1, B02)
                g3 = _CVF(-unsafe_load(pT, _clidx(j + 3, c, ldt))); A03 = muladd(g3, v0, A03); B03 = muladd(g3, v1, B03)
            end
            vstore(A00, d00); vstore(A01, d01); vstore(A02, d02); vstore(A03, d03)
            vstore(B00, d10); vstore(B01, d11); vstore(B02, d12); vstore(B03, d13)
            i += 2_CHOLW
        end
        while i + _CHOLW - 1 <= m
            b0 = _cvptr(p11, i, j, ld1);     a0 = vload(_CVF, b0)
            b1 = _cvptr(p11, i, j + 1, ld1); a1 = vload(_CVF, b1)
            b2 = _cvptr(p11, i, j + 2, ld1); a2 = vload(_CVF, b2)
            b3 = _cvptr(p11, i, j + 3, ld1); a3 = vload(_CVF, b3)
            for c in 1:bs
                lic = vload(_CVF, _cvptr(pT, i, c, ldt))
                a0 = muladd(_CVF(-unsafe_load(pT, _clidx(j, c, ldt))), lic, a0)
                a1 = muladd(_CVF(-unsafe_load(pT, _clidx(j + 1, c, ldt))), lic, a1)
                a2 = muladd(_CVF(-unsafe_load(pT, _clidx(j + 2, c, ldt))), lic, a2)
                a3 = muladd(_CVF(-unsafe_load(pT, _clidx(j + 3, c, ldt))), lic, a3)
            end
            vstore(a0, b0); vstore(a1, b1); vstore(a2, b2); vstore(a3, b3); i += _CHOLW
        end
        while i <= m
            for dj in 0:_CHOL_NC-1
                s = unsafe_load(p11, _clidx(i, j + dj, ld1))
                for c in 1:bs
                    s = muladd(-unsafe_load(pT, _clidx(j + dj, c, ldt)), unsafe_load(pT, _clidx(i, c, ldt)), s)
                end
                unsafe_store!(p11, s, _clidx(i, j + dj, ld1))
            end
            i += 1
        end
        j += _CHOL_NC
    end
    while j <= m
        _syrk_panel_split_f64!(p11, ld1, pT, ldt, j, m, bs); j += 1
    end
    return nothing
end

# Owned conflict-free scratches (GKH ownership; single-thread — MT deferred project-wide).
const _CHOL_D = Matrix{Float64}(undef, _CHOL_BLOCK + 8, _CHOL_BLOCK)  # diag block, fixed 136×128
const _CHOL_T = Ref(Matrix{Float64}(undef, 0, 0))                     # panel workspace, grows (n+8)×NB
# trsm row chunk: the mc×NB T slab the k-repasses re-read stays L2-resident (slab ≤ L2/2).
const _CHOL_MC = max(_CHOLW, (_L2_BYTES ÷ 2) ÷ (_CHOL_BLOCK * 8))

function _chol_panel_f64!(A, n::Int)
    lda = stride(A, 2)
    Tb = _CHOL_T[]
    if size(Tb, 1) < n + 8
        R = (n + 8) % 128 == 0 ? n + 16 : n + 8               # keep ldT itself alias-free
        Tb = _CHOL_T[] = Matrix{Float64}(undef, R, _CHOL_BLOCK)
    end
    ldT = size(Tb, 1); D = _CHOL_D; ldD = size(D, 1)
    GC.@preserve A Tb D begin
        pa = pointer(A); pT = pointer(Tb); pD = pointer(D)
        j = 0
        @inbounds while j < n
            bs = min(_CHOL_BLOCK, n - j)
            pjj = _cvptr(pa, j + 1, j + 1, lda)
            for c in 0:bs-1                                   # diag block lower triangle → D (L1/L2)
                unsafe_copyto!(pD + (c * ldD + c) * 8, pjj + (c * lda + c) * 8, bs - c)
            end
            _chol_rl_f64!(pD, bs, ldD, _CHOL_BLOCK, _CHOL_THRESHOLD) || throw(PosDefException(j + 1))
            for c in 0:bs-1                                   # factored diag back (tiny)
                unsafe_copyto!(pjj + (c * lda + c) * 8, pD + (c * ldD + c) * 8, bs - c)
            end
            m = n - j - bs
            if m > 0
                p21 = _cvptr(pa, j + bs + 1, j + 1, lda)
                i0 = 0                                        # fused panel solve → T, MC row chunks
                while i0 < m
                    mc = min(_CHOL_MC, m - i0)
                    _trsm_rl_split_f64!(pD, ldD, p21 + i0 * 8, lda, pT + i0 * 8, ldT, bs, mc)
                    i0 += mc
                end
                p22 = _cvptr(pa, j + bs + 1, j + bs + 1, lda)
                if m * bs * 8 <= _L2_BYTES ÷ 2                # T slab L2-resident: fused inline syrk
                    _syrk_lower_split_f64!(p22, lda, pT, ldT, m, bs)
                else                                          # big trailing: cache-blocked syrk! reads T
                    syrk!(view(A, (j + bs + 1):n, (j + bs + 1):n), view(Tb, 1:m, 1:bs);
                          uplo = 'L', trans = 'N', alpha = -1, beta = 1)
                end
                for c in 0:bs-1                               # stream the factor back to A21 ONCE
                    unsafe_copyto!(p21 + c * lda * 8, pT + c * ldT * 8, m)
                end
            end
            j += bs
        end
    end
    return A
end

function _potrf_f64_lower!(A, base::Int = _CHOL_FAER_BASE)
    n = size(A, 1)
    n == 0 && return A
    if _CHOLW == 4 && n > _CHOL_BLOCK
        # AVX2: the fused panel driver beats the hybrid/whole-pad path at EVERY size (measured galen/Zen3,
        # 200–4000: transition dips 384/448/640 0.91-0.94→1.01-1.03, non-po2 large 0.98→1.00-1.03). It was
        # originally gated to po2-aliased strides only (its raison d'être was dodging the po2 pad round-trip),
        # but it's a better-composed blocked driver everywhere — the hybrid's generic trsm!(side=R,transA=T)
        # is the side-R-T laggard the panel driver's fused split-ld trsm avoids. (AVX-512 W=8 stays below.)
        return _chol_panel_f64!(A, n)
    end
    if _chol_needs_pad(A, n)                      # factor in a non-conflicting (ld = n+8) scratch, copy back
        R = n + 8
        b = _CHOL_PAD[]
        (size(b, 1) < R || size(b, 2) < n) && (b = _CHOL_PAD[] = Matrix{Float64}(undef, R, n))
        Mw = view(b, 1:n, 1:n)
        # explicit contiguous per-column copies — copyto! on SubArrays is elementwise (the LU pad lesson)
        lda = stride(A, 2); ldb = size(b, 1)
        GC.@preserve A b begin
            pa = pointer(A); pb = pointer(b)
            # Lower triangle only: the faer lower path reads/writes exclusively the lower triangle + diagonal
            # (base kernel loads rows ≥ j, cols < j), and the scratch upper is never-read workspace. Copying
            # column j from its diagonal down (n-j elts) halves the copy — the copy is the WHOLE pad overhead
            # (~16 MB at n=1024), so this lifts the po2-input gate directly (2n²→n² moved).
            @inbounds for j in 0:(n - 1)
                unsafe_copyto!(pb + (j * ldb + j) * 8, pa + (j * lda + j) * 8, n - j)
            end
            _chol_hyb_f64!(Mw, n, base)
            @inbounds for j in 0:(n - 1)
                unsafe_copyto!(pa + (j * lda + j) * 8, pb + (j * ldb + j) * 8, n - j)
            end
        end
    else
        _chol_hyb_f64!(A, n, base)
    end
    return A
end

# Recursive right-looking blocked complex Hermitian Cholesky (lower). Panel width nb ∝ n/4 (OpenBLAS's own
# policy — see potrf_L_single.c: nb = min(n/4, GEMM_Q)), capped at `_CPOTRF_NBMAX`. This keeps the panel
# COUNT ~constant (≈4) and block shapes SCALING CONTINUOUSLY with n — the two things that make the curve
# smooth. A fixed nb=32 instead made panel count jump 2→4→8→16 (large-n falloff: AVX2 0.82 at n≥1024) and
# a big base cutoff added a discrete base→blocked step; nb=n/4 removes both. Per panel: factor the diagonal
# jb-block RECURSIVELY (jb>base ⇒ blocks again; jb≤base ⇒ the vectorized unblocked base), trsm side-R 'C'
# panel solve, herk 'N' rank-jb trailing downdate — all gating L3. BlasComplex only (Dual/upper → generic).
const _CPOTRF_NBMAX = @load_preference("cpotrf_nbmax", _vwidth(Float64) == 4 ? 128 : 192)::Int
@inline _chol_nb(n::Int) = clamp((n >> 2) & ~15, 32, _CPOTRF_NBMAX)     # ~n/4, rounded to a multiple of 16
function _cpotrf_lower!(A, n::Int)
    n <= _CPOTRF_BASE && return _potf2b_lower!(A, n)                    # unblocked vectorized base
    nb = _chol_nb(n)
    j = 1
    @inbounds while j <= n
        jb = min(nb, n - j + 1)
        _cpotrf_lower!(view(A, j:(j + jb - 1), j:(j + jb - 1)), jb)     # factor diagonal jb-block (recurse)
        if j + jb <= n
            db = view(A, j:(j + jb - 1), j:(j + jb - 1)); pan = view(A, (j + jb):n, j:(j + jb - 1))
            trsm!(pan, db; side = 'R', uplo = 'L', transA = 'C', diag = 'N', alpha = true)  # L21 = A21·L11⁻ᴴ
            herk!(view(A, (j + jb):n, (j + jb):n), pan; uplo = 'L', trans = 'N', alpha = -1.0, beta = 1.0)  # A22 -= L21·L21ᴴ
        end
        j += jb
    end
    return A
end

# Public: Cholesky factor A in place into its `uplo` triangle. Returns A. Throws PosDefException if A is
# not positive definite. Float64 lower → faer fast path; complex lower → right-looking blocked; else the
# generic AD-traceable recursion.
function potrf!(A::AbstractMatrix; uplo::Char = 'L')
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("potrf!: A must be square"))
    if uplo == 'L' && _strided1(A)
        eltype(A) === Float64 && return _potrf_f64_lower!(A)
        # n≤base: one vectorized base call (fast ≤64, single contiguous factor). n>base: right-looking
        # blocked (small-nb panels → big amortizing trailing herks). Splitting n≤64 into panels regressed it.
        eltype(A) <: BlasComplex && return (_cpotrf_lower!(A, n); A)   # recursive nb=n/4 (base handled inside)
    end
    base = eltype(A) <: Complex ? _CPOTRF_BASE : _POTRF_BASE   # complex upper / Dual: small base → fast recursion
    uplo == 'L' ? _potrf_lower!(A, n, base) : _potrf_upper!(A, n, base)
    return A
end
