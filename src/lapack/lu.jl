# LAPACK LU (getrf) — partial pivoting, blocked right-looking. FROM SCRATCH (BlazingPorts has no LU
# source — only bench data), but the same recipe as the faer ports: a simple panel kernel + PureBLAS's
# gated trsm!/gemm! for the trailing. A = P·L·U (L unit-lower, U upper; ipiv[i] = global row swapped to
# position i, LAPACK convention). Float64. ponytail: generic/AD LU deferred (pivoting is data-dependent).

const _LU_NB = 48       # blocked panel width base (small nb trims the panel/trsm BLAS-2 cost; large nb fattens
# the rank-nb trailing gemm toward peak). At small n the panel factor dominates → nb=48; at large n the gemm
# dominates → grow nb so k=nb isn't a skinny gemm. req#8 (2026-07-16): fleet nb-sweep (Zen4+Zen3, full nb×n
# grid) shows this is a FLAT, roughly µarch-INVARIANT optimum — NOT a clean cache-residency scale:
#   • FLOOR 48 — VALIDATED: n=256 wants exactly 48 on BOTH boxes (a real dip, nb64 is +2.4% Zen4 / +4.4% Zen3).
#   • SLOPE ÷8 / CAP 128 — VALIDATED: for n≥384 nb∈[96,192] are all within ~1% (flat band); 128 sits in it on
#     both boxes. The curve is parity-BUMPY (multiples of 64 win; e.g. 128,192 beat 168), so it does NOT track
#     a residency formula: the derived `_l1_block(F64,_MR·W)` (=128 Zen4 / 168 Zen3, the A-micropanel ½·L1
#     bound) was FLEET-FALSIFIED — 168 is a trough on Zen3 (+0.4–2.6% vs 128), worse than the literal. The true
#     large-n optimum is 64-aligned + per-µarch (Zen4 256, Zen3 192) — no clean formula hits it, and it's only
#     ~0.5% (Zen4) / ~1.5% (Zen3, n≥1536) above 128. A formula here would add spurious variation (req#8(b)).
# So: measured-validated literals kept. (The old "+4–5% large-n → BLASFEO parity" note stands as the win vs a
# flat nb=48; this sweep confirms 128 is near the flat optimum, not that a bigger derived cap would help.)
_lu_nb(n::Int) = clamp((n ÷ 8) & ~15, _LU_NB, 128)

# Complex getrf panel width. Grows with n so the trailing rank-nb zgemm is compute-bound — measured
# (galen): a rank-k zgemm gates only at k≳96 on AVX2 (k=48 → 0.85), so the complex panel must grow faster
# than the real one (÷5 vs ÷8) — and is CAPPED at the complex-gemm kc micropanel `_clu_cap` (== `_CKC` for
# ComplexF64: L1-residency `kc·nr·sizeof(T) ≤ ½L1`, per-T via `_l1_block`), beyond which the trailing zgemm
# re-blocks k internally anyway. Floor keeps the panel lean at small n (the rank-2 panel factor dominates
# there). Reproduces the rank-2-panel nb-sweep optima (galen Zen3/AVX2): 32@128, 48@256, 96@512, cap@≥1024.
@inline _clu_cap(::Type{T}) where {T <: BlasComplex} = _l1_block(_HW, T, max(_CNR, _CNR_SMALL))
@inline _clu_nb(n::Int, ::Type{T}) where {T <: BlasComplex} = clamp((n ÷ 5) & ~15, 32, _clu_cap(T))

# Panels wider than this recurse (see _getf2_blocked!). The flat rank-1 sweep below rewrites the trailing
# panel once per pivot column → O(pb²·mp) stores (store-bound BLAS-2, ~40% of getrf(256) on AVX2). A
# single split does the cross-half update ONCE via BLAS-3 gemm, halving that store traffic.
const _GETF2_BASE = 16   # ≤ this ⇒ store-bound rank-1 sweep; above ⇒ BLAS-3 split. 24→16: +2-4.5% at n=64-384 (less store-bound rank-1 in the cliff zone). req#8: INVARIANT — a store-traffic algorithm-switch crossover (not a cache-residency block), validated-by-gate (getrf gates vs OB+AOCL); the complex sibling `_CGETF2_BASE` is sizeof-derived. Literal retained (a residency formula would add spurious variation; cf. falsified _LU_NB/_TRMM_RPACK derivations).
# Complex base is WIDER than the real one, by the complex/real element-size ratio (=2): the complex base
# is a rank-2 SIMD sweep (already halves the store traffic the recursion's cross-half gemm targets), so the
# BLAS-3 split's benefit shrinks while its cost (skinny-k zgemm/ztrsm cross-updates on a TALL panel) grows,
# pushing the flat/recurse crossover ~2× wider. Measured (galen): 32 beats both 16 and 64 for zgetrf 256/
# 1024 (panel 0.91→0.96 at m=1024). Derived from sizeof so cgetf2 (ComplexF32) tracks the same criterion.
const _CGETF2_BASE = _GETF2_BASE * (sizeof(ComplexF64) ÷ sizeof(Float64))   # = 32

# Apply the sequential row interchanges recorded in ipiv[ip0+1 : ip0+np] to columns j1:j2 of panel view V,
# LOCAL to V (ipiv holds GLOBAL rows = roff + local). Pivot t swaps V-row (rowbase+t) ↔ V-row (ipiv[ip0+t]
# - roff). Used only inside _getf2_blocked! for the cross-half swaps; the flat base does its own swaps.
@inline function _laswp_local!(V, ipiv, ip0::Int, np::Int, rowbase::Int, j1::Int, j2::Int, roff::Int)
    return @inbounds for t in 1:np
        pos = rowbase + t; r = ipiv[ip0 + t] - roff
        if r != pos
            for j in j1:j2
                V[pos, j], V[r, j] = V[r, j], V[pos, j]
            end
        end
    end
end

# Blocked (recursive) panel LU: split the pb columns once, factor the left half, propagate its pivots to
# the right half, solve U12 = L11⁻¹A12 (trsm) and downdate A22 -= L21·U12 (gemm — the BLAS-3 cross-half
# update that replaces pb1 rank-1 passes), factor the right half, then propagate its pivots to the left
# half. Net result is bit-identical to the flat sweep; only the store traffic is halved. Pivoting is a
# correctness boundary — the two _laswp_local! calls carry the interchanges the flat sweep applied inline.
function _getf2_blocked!(A, mp::Int, pb::Int, roff::Int, ipiv, ioff::Int)
    _base = eltype(A) <: BlasComplex ? _CGETF2_BASE : _GETF2_BASE      # complex base is wider (flat rank-2)
    pb <= _base && return _getf2!(A, mp, pb, roff, ipiv, ioff)         # base: flat rank-1/rank-2 sweep
    pb1 = pb ÷ 2; pb2 = pb - pb1
    info = _getf2_blocked!(view(A, :, 1:pb1), mp, pb1, roff, ipiv, ioff)     # factor left half
    _laswp_local!(A, ipiv, ioff, pb1, 0, pb1 + 1, pb, roff)                  # left pivots → right cols
    trsm!(
        view(A, 1:pb1, (pb1 + 1):pb), view(A, 1:pb1, 1:pb1);
        side = 'L', uplo = 'L', transA = 'N', diag = 'U', alpha = true
    )    # U12 = L11⁻¹ A12
    gemm!(
        view(A, (pb1 + 1):mp, (pb1 + 1):pb), view(A, (pb1 + 1):mp, 1:pb1), view(A, 1:pb1, (pb1 + 1):pb);
        alpha = -1, beta = true
    )                                          # A22 -= L21 U12
    info2 = _getf2_blocked!(view(A, (pb1 + 1):mp, (pb1 + 1):pb), mp - pb1, pb2, roff + pb1, ipiv, ioff + pb1)
    (info == 0 && info2 != 0) && (info = info2)                             # factor right half
    _laswp_local!(A, ipiv, ioff + pb1, pb2, pb1, 1, pb1, roff)              # right pivots → left cols
    return info
end

# Unblocked panel LU with partial pivoting (LAPACK dgetf2) on an mp×pb panel whose rows are global
# (offset roff). Fills ipiv[ioff+1 : ioff+pb] with GLOBAL 1-based pivot rows. Returns the first
# zero-pivot global column. Float64 contiguous-column panel → SIMD via PureBLAS's layer (vectorized
# column scale + rank-1 update over contiguous rows; scalar argmax); else the generic fallback.
# Pivoting is a correctness boundary — do not simplify.
function _getf2!(A, mp::Int, pb::Int, roff::Int, ipiv, ioff::Int)
    _base = eltype(A) <: BlasComplex ? _CGETF2_BASE : _GETF2_BASE            # complex base is wider (rank-2)
    if pb > _base && A isa StridedMatrix && stride(A, 1) == 1 &&              # wide → BLAS-3 split (generic:
            (eltype(A) === Float64 || eltype(A) <: BlasComplex)               # rides trsm!/gemm!, incl. complex)
        return _getf2_blocked!(A, mp, pb, roff, ipiv, ioff)
    end
    if A isa StridedMatrix{Float64} && stride(A, 1) == 1
        return GC.@preserve A _getf2_simd!(pointer(A), stride(A, 2), mp, pb, roff, ipiv, ioff)
    end
    if A isa StridedMatrix && stride(A, 1) == 1 && eltype(A) <: BlasComplex
        return GC.@preserve A _cgetf2_simd!(pointer(A), stride(A, 2), mp, pb, roff, ipiv, ioff)
    end
    info = 0
    @inbounds for jl in 1:pb
        piv = jl; pmax = abs(A[jl, jl])                  # partial pivot: max |·| in column jl, rows jl:mp
        for il in (jl + 1):mp
            a = abs(A[il, jl]); a > pmax && (pmax = a; piv = il)
        end
        ipiv[ioff + jl] = roff + piv
        if A[piv, jl] != 0.0
            if piv != jl                                  # swap rows jl ↔ piv across the panel
                for jc in 1:pb
                    A[jl, jc], A[piv, jc] = A[piv, jc], A[jl, jc]
                end
            end
            d = 1.0 / A[jl, jl]
            for il in (jl + 1):mp
                A[il, jl] *= d
            end     # scale column below the diagonal
        elseif info == 0
            info = roff + jl
        end
        for jc in (jl + 1):pb                             # rank-1 update of the panel trailing
            ajc = A[jl, jc]
            for il in (jl + 1):mp
                A[il, jc] -= A[il, jl] * ajc
            end
        end
    end
    return info
end

const _LU_IOFF = Vec{_CHOLW, Int}(ntuple(k -> k - 1, _CHOLW))    # lane row offsets for the SIMD idamax

# SIMD partial-pivot argmax: index of max |·| over rows jl:mp of column jl (LAPACK: first max on ties).
@inline function _idamax_col(p::Ptr{Float64}, ld::Int, mp::Int, jl::Int)
    vm = _CVF(-1.0); vi = Vec{_CHOLW, Int}(jl); i = jl
    @inbounds while i + _CHOLW - 1 <= mp
        v = abs(vload(_CVF, _cvptr(p, i, jl, ld))); idx = _LU_IOFF + Vec{_CHOLW, Int}(i)
        gt = v > vm; vm = vifelse(gt, v, vm); vi = vifelse(gt, idx, vi); i += _CHOLW
    end
    m = -1.0; piv = jl
    @inbounds for l in 1:_CHOLW
        (vm[l] > m) && (m = vm[l]; piv = vi[l])
    end   # lane reduce (first max)
    @inbounds while i <= mp
        a = abs(unsafe_load(p, _clidx(i, jl, ld))); a > m && (m = a; piv = i); i += 1
    end
    return piv
end

# factor column jl: SIMD argmax + swap rows jl↔piv across the whole panel + scale below diagonal. Returns
# true on a zero pivot. Pivoting is a correctness boundary — do not simplify.
@inline function _lu_fact1!(p::Ptr{Float64}, ld::Int, mp::Int, jl::Int, pb::Int, roff::Int, ipiv, ioff::Int)
    piv = _idamax_col(p, ld, mp, jl); ipiv[ioff + jl] = roff + piv
    zero_piv = unsafe_load(p, _clidx(piv, jl, ld)) == 0.0
    @inbounds if piv != jl
        for c in 1:pb
            a = unsafe_load(p, _clidx(jl, c, ld))
            unsafe_store!(p, unsafe_load(p, _clidx(piv, c, ld)), _clidx(jl, c, ld)); unsafe_store!(p, a, _clidx(piv, c, ld))
        end
    end
    @inbounds if !zero_piv
        invd = 1.0 / unsafe_load(p, _clidx(jl, jl, ld)); vinv = _CVF(invd); i = jl + 1
        while i + _CHOLW - 1 <= mp
            b = _cvptr(p, i, jl, ld); vstore(vload(_CVF, b) * vinv, b); i += _CHOLW
        end
        while i <= mp
            unsafe_store!(p, unsafe_load(p, _clidx(i, jl, ld)) * invd, _clidx(i, jl, ld)); i += 1
        end
    end
    return zero_piv
end

# SIMD panel (Float64) — RANK-2 blocked with SIMD idamax. Bit-identical pivots/result to the flat rank-1
# sweep, but the trailing update touches each element ONCE per 2 columns (the rank-1 sweep is store-bound
# BLAS-2, ~40% of getrf(256)) and the pivot argmax is vectorized (was ~30% scalar). Measured +48–82% on the
# base (galen). Columns processed in pairs: factor jl, update col jl+1 by jl, factor jl+1, then one fused
# rank-2 update of the trailing (with the U[jl+1,·] row correction). Pivoting is a correctness boundary.
function _getf2_simd!(p::Ptr{Float64}, ld::Int, mp::Int, pb::Int, roff::Int, ipiv, ioff::Int)
    info = 0; jl = 1
    @inbounds while jl <= pb
        if jl == pb
            _lu_fact1!(p, ld, mp, jl, pb, roff, ipiv, ioff) && info == 0 && (info = roff + jl); break
        end
        _lu_fact1!(p, ld, mp, jl, pb, roff, ipiv, ioff) && info == 0 && (info = roff + jl)
        u = -unsafe_load(p, _clidx(jl, jl + 1, ld)); vu = _CVF(u); i = jl + 1     # update col jl+1 by jl
        while i + _CHOLW - 1 <= mp
            bj = _cvptr(p, i, jl + 1, ld); vstore(muladd(vu, vload(_CVF, _cvptr(p, i, jl, ld)), vload(_CVF, bj)), bj); i += _CHOLW
        end
        while i <= mp
            unsafe_store!(p, muladd(u, unsafe_load(p, _clidx(i, jl, ld)), unsafe_load(p, _clidx(i, jl + 1, ld))), _clidx(i, jl + 1, ld)); i += 1
        end
        _lu_fact1!(p, ld, mp, jl + 1, pb, roff, ipiv, ioff) && info == 0 && (info = roff + jl + 1)
        for jc in (jl + 2):pb                                                    # fused rank-2 trailing update
            u0 = -unsafe_load(p, _clidx(jl, jc, ld))
            u1c = unsafe_load(p, _clidx(jl + 1, jc, ld)) + u0 * unsafe_load(p, _clidx(jl + 1, jl, ld))  # U[jl+1,jc]
            unsafe_store!(p, u1c, _clidx(jl + 1, jc, ld)); u1 = -u1c
            vu0 = _CVF(u0); vu1 = _CVF(u1); i = jl + 2
            while i + _CHOLW - 1 <= mp
                bj = _cvptr(p, i, jc, ld); acc = vload(_CVF, bj)
                acc = muladd(vu0, vload(_CVF, _cvptr(p, i, jl, ld)), acc); acc = muladd(vu1, vload(_CVF, _cvptr(p, i, jl + 1, ld)), acc)
                vstore(acc, bj); i += _CHOLW
            end
            while i <= mp
                s = unsafe_load(p, _clidx(i, jc, ld)); s = muladd(u0, unsafe_load(p, _clidx(i, jl, ld)), s); s = muladd(u1, unsafe_load(p, _clidx(i, jl + 1, ld)), s); unsafe_store!(p, s, _clidx(i, jc, ld)); i += 1
            end
        end
        jl += 2
    end
    return info
end

# factor complex column jl: SIMD izamax (cabs1, LAPACK's izamax metric) + swap rows jl↔piv across the whole
# panel + scale below diagonal by 1/pivot. Returns true on a zero pivot. Pivoting is a correctness boundary.
@inline function _clu_fact1!(p::Ptr{Tc}, ld::Int, mp::Int, jl::Int, pb::Int, roff::Int, ipiv, ioff::Int) where {Tc <: BlasComplex}
    R = real(Tc); csz = sizeof(Tc)
    lidx(i, k) = (k - 1) * ld + i
    cptr(i, k) = p + ((k - 1) * ld + (i - 1)) * csz
    nrows = mp - jl + 1
    rel = if nrows >= 4 * _vwidth(R)                                # SIMD izamax needs n ≥ 4W (else OOB lanes)
        _iamax_cmplx_simd!(nrows, Ptr{R}(cptr(jl, jl)))
    else
        b = 1; m = -one(R)                                          # scalar cabs1 argmax (LAPACK izamax metric)
        for t in 1:nrows
            v = unsafe_load(p, lidx(jl - 1 + t, jl)); a = abs(real(v)) + abs(imag(v))
            a > m && (m = a; b = t)
        end
        b
    end
    piv = jl - 1 + rel
    ipiv[ioff + jl] = roff + piv
    zero_piv = unsafe_load(p, lidx(piv, jl)) == zero(Tc)
    @inbounds if piv != jl                                          # swap rows jl ↔ piv across the panel
        for jc in 1:pb
            a = unsafe_load(p, lidx(jl, jc))
            unsafe_store!(p, unsafe_load(p, lidx(piv, jc)), lidx(jl, jc)); unsafe_store!(p, a, lidx(piv, jc))
        end
    end
    @inbounds if !zero_piv
        mt = mp - jl
        r = _crecip(unsafe_load(p, lidx(jl, jl)))                   # 1/pivot; scale column below the diagonal
        mt > 0 && _scal_cmplx_simd!(mt, real(r), imag(r), cptr(jl + 1, jl))
    end
    return zero_piv
end

# Complex SIMD panel (zgetf2) — RANK-2 blocked (mirror of the real _getf2_simd!): SIMD izamax argmax
# (`_iamax_cmplx_simd!`), and the trailing update touches each element ONCE per 2 columns via the fused
# 2-source complex axpy (`_qr_axpy2_cmplx!`) instead of two rank-1 passes — halving the store traffic that
# dominates the store-bound BLAS-2 panel. Columns in pairs: factor jl, update col jl+1 by jl, factor jl+1,
# then one fused rank-2 update of the trailing (with the U[jl+1,·] row correction). Bit-equivalent pivots/
# result to the flat rank-1 sweep. Pivoting is a correctness boundary — do not simplify.
function _cgetf2_simd!(p::Ptr{Tc}, ld::Int, mp::Int, pb::Int, roff::Int, ipiv, ioff::Int) where {Tc <: BlasComplex}
    csz = sizeof(Tc); info = 0
    lidx(i, k) = (k - 1) * ld + i
    cptr(i, k) = p + ((k - 1) * ld + (i - 1)) * csz
    jl = 1
    @inbounds while jl <= pb
        if jl == pb                                                # odd tail: single column
            _clu_fact1!(p, ld, mp, jl, pb, roff, ipiv, ioff) && info == 0 && (info = roff + jl); break
        end
        _clu_fact1!(p, ld, mp, jl, pb, roff, ipiv, ioff) && info == 0 && (info = roff + jl)
        u = unsafe_load(p, lidx(jl, jl + 1)); mt = mp - jl          # update col jl+1 by jl (rank-1 axpy)
        mt > 0 && _axpy_cmplx_simd!(mt, -real(u), -imag(u), cptr(jl + 1, jl), cptr(jl + 1, jl + 1))
        _clu_fact1!(p, ld, mp, jl + 1, pb, roff, ipiv, ioff) && info == 0 && (info = roff + jl + 1)
        mt2 = mp - jl - 1
        for jc in (jl + 2):pb                                      # fused rank-2 trailing update
            a0 = unsafe_load(p, lidx(jl, jc)); u0 = -a0
            uc = unsafe_load(p, lidx(jl + 1, jc)) - a0 * unsafe_load(p, lidx(jl + 1, jl))  # U[jl+1,jc]
            unsafe_store!(p, uc, lidx(jl + 1, jc)); u1 = -uc
            mt2 > 0 && _qr_axpy2_cmplx!(
                mt2, real(u0), imag(u0), real(u1), imag(u1),
                cptr(jl + 2, jl), cptr(jl + 2, jl + 1), cptr(jl + 2, jc)
            )
        end
        jl += 2
    end
    return info
end

# Apply row interchanges ipiv[k1:k2] to columns j1:j2 (LAPACK dlaswp), in sequence. Size-adaptive:
# small m → column-outer (each contiguous column is L1-resident through all its swaps); large m → 32-col
# blocked, pivots inner (each block ≤½ L2, pivot index/branch hoisted across it). Measured: small n wants
# column-outer, large n wants blocked.
function _laswp!(A, ipiv, k1::Int, k2::Int, j1::Int, j2::Int)
    j1 > j2 && return
    return if size(A, 1) < 1024
        @inbounds for j in j1:j2
            for i in k1:k2
                ip = ipiv[i]
                if ip != i
                    A[i, j], A[ip, j] = A[ip, j], A[i, j]
                end
            end
        end
    else
        @inbounds for jb in j1:32:j2
            je = min(jb + 31, j2)
            for i in k1:k2
                ip = ipiv[i]
                if ip != i
                    for j in jb:je
                        A[i, j], A[ip, j] = A[ip, j], A[i, j]
                    end
                end
            end
        end
    end
end

# Reusable padded scratch (like Cholesky): a po2 / stride%512==0 leading dim aliases cache sets, slowing
# the panel + laswp at large n; factor in an ld=m+8 buffer and copy back.
const _LU_PAD = Ref(Matrix{Float64}(undef, 0, 0))
@inline _lu_needs_pad(A, m) = m >= 512 && stride(A, 2) % 512 == 0

# Blocked right-looking LU (LAPACK dgetrf's algorithm — the reference, faster here than a recursive LU
# which over-decomposes into many small gemm! calls). Factor each nb-panel (getf2), swap the rest of the
# rows (laswp), solve the row panel (trsm, L11⁻¹·A12), downdate the trailing (gemm, A22 −= L21·U12).
# The cheap unblocked panel + ONE big rank-nb trailing gemm per step is the win. Returns (A, ipiv, info).
function getrf!(A::AbstractMatrix{Float64}, ipiv::AbstractVector{<:Integer}; nb::Int = _lu_nb(min(size(A)...)))
    m, n = size(A); k = min(m, n)
    k == 0 && return A, ipiv, 0
    length(ipiv) >= k || throw(DimensionMismatch("getrf!: length(ipiv) < min(size(A))"))
    if _lu_needs_pad(A, m)                                # factor in a non-conflicting (ld=m+8) scratch
        R = m + 8
        b = _LU_PAD[]
        (size(b, 1) < R || size(b, 2) < n) && (b = _LU_PAD[] = Matrix{Float64}(undef, R, n))
        Mw = view(b, 1:m, 1:n)
        ld = stride(A, 2); sz = sizeof(eltype(A))
        info = GC.@preserve A b begin
            pA = pointer(A); pB = pointer(b)
            @inbounds for j in 0:(n - 1)                       # contiguous per-column copy in (A → scratch)
                unsafe_copyto!(pB + j * R * sz, pA + j * ld * sz, m)
            end
            (_, _, i3) = _getrf_core!(Mw, ipiv, nb)
            @inbounds for j in 0:(n - 1)                       # and back
                unsafe_copyto!(pA + j * ld * sz, pB + j * R * sz, m)
            end
            i3
        end
        return A, ipiv, info
    end
    return _getrf_core!(A, ipiv, nb)
end

function _getrf_core!(A, ipiv, nb::Int)
    m, n = size(A); k = min(m, n)
    nb = clamp(nb, 1, k)
    info = 0; pc = 1
    @inbounds while pc <= k
        pb = min(nb, k - pc + 1); mp = m - pc + 1
        pinfo = _getf2!(view(A, pc:m, pc:(pc + pb - 1)), mp, pb, pc - 1, ipiv, pc - 1)
        (info == 0 && pinfo != 0) && (info = pinfo)
        jt0 = pc + pb
        if jt0 <= n
            _laswp!(A, ipiv, pc, pc + pb - 1, jt0, n)                # swap trailing columns (needed now)
            trsm!(
                view(A, pc:(pc + pb - 1), jt0:n), view(A, pc:(pc + pb - 1), pc:(pc + pb - 1));
                side = 'L', uplo = 'L', transA = 'N', diag = 'U', alpha = true
            )   # U12 = L11⁻¹ A12
            if pc + pb <= m
                gemm!(
                    view(A, (pc + pb):m, jt0:n), view(A, (pc + pb):m, pc:(pc + pb - 1)), view(A, pc:(pc + pb - 1), jt0:n);
                    alpha = -1, beta = true
                )                       # A22 −= L21 U12
            end
        end
        pc += pb
    end
    # DEFERRED left-block pivots: each panel's columns get the LATER pivots, applied once at the end
    # (cache-friendly — the in-loop version re-touched cold left columns at every subsequent panel, the
    # large-n laswp killer). Same permutation, reordered: column j gets ipiv from panels after its own.
    pc = 1
    @inbounds while pc <= k
        pb = min(nb, k - pc + 1); jt0 = pc + pb
        jt0 <= k && _laswp!(A, ipiv, jt0, k, pc, pc + pb - 1)
        pc += pb
    end
    return A, ipiv, info
end

# Complex padded scratch (mirror of the real `_LU_PAD`): a po2 leading dim aliases cache sets — the L1 page
# is 4096 B, so `stride·sizeof(T) % 4096 == 0` maps every column onto the same set, thrashing the per-pivot
# row-swaps + trsm/gemm view reads. MEASURED (galen): a sharp dip at n=256 (0.94 vs 1.02 at n=252/260) and
# n=1024 (0.97 vs 1.10 at n=1020/1028) — exactly the po2 sizes; the non-po2 neighbours gate. Factor in an
# ld=m+8 buffer (breaks the aliasing) and copy back. Per-type owned scratch (GKH ownership; trim-safe).
const _CLU_PAD64 = Ref(Matrix{ComplexF64}(undef, 0, 0))
const _CLU_PAD32 = Ref(Matrix{ComplexF32}(undef, 0, 0))
@inline _clu_pad(::Type{ComplexF64}) = _CLU_PAD64
@inline _clu_pad(::Type{ComplexF32}) = _CLU_PAD32
@inline _clu_needs_pad(A, m, ::Type{T}) where {T} =
    m >= 256 && A isa StridedMatrix && stride(A, 1) == 1 && (stride(A, 2) * sizeof(T)) % 4096 == 0

# Complex LU (zgetrf): identical structure — no conj anywhere in L·U. `_getf2!`'s rank-2 SIMD panel pivots
# on |·| (cabs1 = LAPACK izamax) and divides by the complex pivot correctly; `_getrf_core!` rides complex
# trsm!/gemm! (the 3M path) for the trailing update. Pad-scratch dodges po2 lda aliasing (see `_clu_pad`).
# (Oracle tests compare the PA=LU residual, not entries.)
function getrf!(A::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer}; nb::Int = _clu_nb(min(size(A)...), T)) where {T <: BlasComplex}
    m, n = size(A); k = min(m, n)
    k == 0 && return A, ipiv, 0
    length(ipiv) >= k || throw(DimensionMismatch("getrf!: length(ipiv) < min(size(A))"))
    if _clu_needs_pad(A, m, T)                            # factor in a non-conflicting scratch
        R = m + 8                                          # +8 breaks the set-aliasing (measured: offset ≥8 saturates)
        pref = _clu_pad(T); b = pref[]
        (size(b, 1) < R || size(b, 2) < n) && (b = pref[] = Matrix{T}(undef, R, n))
        Mw = view(b, 1:m, 1:n)
        ld = stride(A, 2); sz = sizeof(T)
        info = GC.@preserve A b begin
            pA = pointer(A); pB = pointer(b)
            @inbounds for j in 0:(n - 1)                       # contiguous per-column copy in (A → scratch)
                unsafe_copyto!(pB + j * R * sz, pA + j * ld * sz, m)
            end
            (_, _, i3) = _getrf_core!(Mw, ipiv, nb)
            @inbounds for j in 0:(n - 1)                       # and back
                unsafe_copyto!(pA + j * ld * sz, pB + j * R * sz, m)
            end
            i3
        end
        return A, ipiv, info
    end
    return _getrf_core!(A, ipiv, nb)
end
# Convenience: allocate ipiv, return (A overwritten with L\U, ipiv, info).
function getrf!(A::StridedMatrix{Float64})
    ipiv = Vector{Int}(undef, min(size(A)...))
    return getrf!(A, ipiv)
end
function getrf!(A::StridedMatrix{T}) where {T <: BlasComplex}
    ipiv = Vector{Int}(undef, min(size(A)...))
    return getrf!(A, ipiv)
end
