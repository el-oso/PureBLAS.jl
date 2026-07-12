# LAPACK LU (getrf) — partial pivoting, blocked right-looking. FROM SCRATCH (BlazingPorts has no LU
# source — only bench data), but the same recipe as the faer ports: a simple panel kernel + PureBLAS's
# gated trsm!/gemm! for the trailing. A = P·L·U (L unit-lower, U upper; ipiv[i] = global row swapped to
# position i, LAPACK convention). Float64. ponytail: generic/AD LU deferred (pivoting is data-dependent).

const _LU_NB = 48       # blocked panel width base (small nb trims the panel/trsm BLAS-2 cost; large nb fattens
# the rank-nb trailing gemm toward peak). At small n the panel factor dominates → nb=48; at large n the gemm
# dominates → grow nb so k=nb isn't a skinny gemm. Measured (galen/fleet): nb=48 to n≈384, 64@512, 128@≥1024
# gives +4–5% large-n (getrf 2048 45.7→47.8 = BLASFEO parity, OB gate 1.01→1.06). ponytail/req#8: the /8 +
# clamp bounds are measured literals to re-derive from the gemm k-block (L1) / trailing residency (L2).
_lu_nb(n::Int) = clamp((n ÷ 8) & ~15, _LU_NB, 128)

# Panels wider than this recurse (see _getf2_blocked!). The flat rank-1 sweep below rewrites the trailing
# panel once per pivot column → O(pb²·mp) stores (store-bound BLAS-2, ~40% of getrf(256) on AVX2). A
# single split does the cross-half update ONCE via BLAS-3 gemm, halving that store traffic.
const _GETF2_BASE = 16   # ≤ this ⇒ store-bound rank-1 sweep; above ⇒ BLAS-3 split. 24→16: +2-4.5% at n=64-384 (less store-bound rank-1 in the cliff zone). req#8: derive from store-BW/L1.

# Apply the sequential row interchanges recorded in ipiv[ip0+1 : ip0+np] to columns j1:j2 of panel view V,
# LOCAL to V (ipiv holds GLOBAL rows = roff + local). Pivot t swaps V-row (rowbase+t) ↔ V-row (ipiv[ip0+t]
# - roff). Used only inside _getf2_blocked! for the cross-half swaps; the flat base does its own swaps.
@inline function _laswp_local!(V, ipiv, ip0::Int, np::Int, rowbase::Int, j1::Int, j2::Int, roff::Int)
    @inbounds for t in 1:np
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
    pb <= _GETF2_BASE && return _getf2!(A, mp, pb, roff, ipiv, ioff)   # base: flat rank-1 sweep
    pb1 = pb ÷ 2; pb2 = pb - pb1
    info = _getf2_blocked!(view(A, :, 1:pb1), mp, pb1, roff, ipiv, ioff)     # factor left half
    _laswp_local!(A, ipiv, ioff, pb1, 0, pb1 + 1, pb, roff)                  # left pivots → right cols
    trsm!(view(A, 1:pb1, pb1+1:pb), view(A, 1:pb1, 1:pb1);
          side = 'L', uplo = 'L', transA = 'N', diag = 'U', alpha = true)    # U12 = L11⁻¹ A12
    gemm!(view(A, pb1+1:mp, pb1+1:pb), view(A, pb1+1:mp, 1:pb1), view(A, 1:pb1, pb1+1:pb);
          alpha = -1, beta = true)                                          # A22 -= L21 U12
    info2 = _getf2_blocked!(view(A, pb1+1:mp, pb1+1:pb), mp - pb1, pb2, roff + pb1, ipiv, ioff + pb1)
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
    if pb > _GETF2_BASE && A isa StridedMatrix && stride(A, 1) == 1 &&        # wide → BLAS-3 split (generic:
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
            for il in (jl + 1):mp; A[il, jl] *= d; end     # scale column below the diagonal
        elseif info == 0
            info = roff + jl
        end
        for jc in (jl + 1):pb                             # rank-1 update of the panel trailing
            ajc = A[jl, jc]
            for il in (jl + 1):mp; A[il, jc] -= A[il, jl] * ajc; end
        end
    end
    return info
end

# SIMD panel (Float64, ld-strided columns, p = &panel[1,1]). Same math as above, vectorized over rows.
function _getf2_simd!(p::Ptr{Float64}, ld::Int, mp::Int, pb::Int, roff::Int, ipiv, ioff::Int)
    info = 0
    @inbounds for jl in 1:pb
        piv = jl; pmax = abs(unsafe_load(p, _clidx(jl, jl, ld)))      # argmax |·| in column jl, rows jl:mp
        for il in (jl + 1):mp
            a = abs(unsafe_load(p, _clidx(il, jl, ld))); a > pmax && (pmax = a; piv = il)
        end
        ipiv[ioff + jl] = roff + piv
        d = unsafe_load(p, _clidx(piv, jl, ld))
        if d != 0.0
            if piv != jl                                              # swap rows jl ↔ piv across the panel
                for jc in 1:pb
                    a = unsafe_load(p, _clidx(jl, jc, ld))
                    unsafe_store!(p, unsafe_load(p, _clidx(piv, jc, ld)), _clidx(jl, jc, ld))
                    unsafe_store!(p, a, _clidx(piv, jc, ld))
                end
            end
            invd = 1.0 / unsafe_load(p, _clidx(jl, jl, ld)); vinv = _CVF(invd)
            i = jl + 1                                                # scale column below the diagonal
            while i + _CHOLW - 1 <= mp
                b = _cvptr(p, i, jl, ld); vstore(vload(_CVF, b) * vinv, b); i += _CHOLW
            end
            while i <= mp; unsafe_store!(p, unsafe_load(p, _clidx(i, jl, ld)) * invd, _clidx(i, jl, ld)); i += 1; end
        elseif info == 0
            info = roff + jl
        end
        for jc in (jl + 1):pb                                         # rank-1 update of the panel trailing
            ajc = -unsafe_load(p, _clidx(jl, jc, ld)); vajc = _CVF(ajc)
            i = jl + 1
            while i + _CHOLW - 1 <= mp
                bj = _cvptr(p, i, jc, ld)
                vstore(muladd(vajc, vload(_CVF, _cvptr(p, i, jl, ld)), vload(_CVF, bj)), bj); i += _CHOLW
            end
            while i <= mp; unsafe_store!(p, muladd(ajc, unsafe_load(p, _clidx(i, jl, ld)), unsafe_load(p, _clidx(i, jc, ld))), _clidx(i, jc, ld)); i += 1; end
        end
    end
    return info
end

# Complex SIMD panel (zgetf2): same math as the scalar fallback, vectorized over rows via the L1 complex
# kernels (`_scal_cmplx_simd!` column scale, `_axpy_cmplx_simd!` rank-1 trailing update). Scalar argmax +
# row swap (data-dependent). Contiguous-column complex panel, p = &panel[1,1]. Pivoting is a correctness
# boundary — do not simplify.
function _cgetf2_simd!(p::Ptr{Tc}, ld::Int, mp::Int, pb::Int, roff::Int, ipiv, ioff::Int) where {Tc <: BlasComplex}
    R = real(Tc); csz = sizeof(Tc); info = 0
    lidx(i, k) = (k - 1) * ld + i                                   # 1-based linear (complex) index
    cptr(i, k) = p + ((k - 1) * ld + (i - 1)) * csz                # &A[i,k] (complex Ptr)
    @inbounds for jl in 1:pb
        piv = jl; pmax = abs(unsafe_load(p, lidx(jl, jl)))         # argmax |·| in column jl, rows jl:mp
        for il in (jl + 1):mp
            a = abs(unsafe_load(p, lidx(il, jl))); a > pmax && (pmax = a; piv = il)
        end
        ipiv[ioff + jl] = roff + piv
        if unsafe_load(p, lidx(piv, jl)) != zero(Tc)
            if piv != jl                                            # swap rows jl ↔ piv across the panel
                for jc in 1:pb
                    a = unsafe_load(p, lidx(jl, jc))
                    unsafe_store!(p, unsafe_load(p, lidx(piv, jc)), lidx(jl, jc))
                    unsafe_store!(p, a, lidx(piv, jc))
                end
            end
            mt = mp - jl
            r = _crecip(unsafe_load(p, lidx(jl, jl)))               # scale column below diagonal by 1/pivot
            mt > 0 && _scal_cmplx_simd!(mt, real(r), imag(r), cptr(jl + 1, jl))
        elseif info == 0
            info = roff + jl
        end
        mt = mp - jl
        for jc in (jl + 1):pb                                       # rank-1 update: A[jl+1:,jc] -= A[jl,jc]·A[jl+1:,jl]
            ajc = unsafe_load(p, lidx(jl, jc))
            mt > 0 && _axpy_cmplx_simd!(mt, -real(ajc), -imag(ajc), cptr(jl + 1, jl), cptr(jl + 1, jc))
        end
    end
    return info
end

# Apply row interchanges ipiv[k1:k2] to columns j1:j2 (LAPACK dlaswp), in sequence. Size-adaptive:
# small m → column-outer (each contiguous column is L1-resident through all its swaps); large m → 32-col
# blocked, pivots inner (each block ≤½ L2, pivot index/branch hoisted across it). Measured: small n wants
# column-outer, large n wants blocked.
function _laswp!(A, ipiv, k1::Int, k2::Int, j1::Int, j2::Int)
    j1 > j2 && return
    if size(A, 1) < 1024
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
            @inbounds for j in 0:n-1                       # contiguous per-column copy in (A → scratch)
                unsafe_copyto!(pB + j * R * sz, pA + j * ld * sz, m)
            end
            (_, _, i3) = _getrf_core!(Mw, ipiv, nb)
            @inbounds for j in 0:n-1                       # and back
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
        pinfo = _getf2!(view(A, pc:m, pc:pc+pb-1), mp, pb, pc - 1, ipiv, pc - 1)
        (info == 0 && pinfo != 0) && (info = pinfo)
        jt0 = pc + pb
        if jt0 <= n
            _laswp!(A, ipiv, pc, pc + pb - 1, jt0, n)                # swap trailing columns (needed now)
            trsm!(view(A, pc:pc+pb-1, jt0:n), view(A, pc:pc+pb-1, pc:pc+pb-1);
                  side = 'L', uplo = 'L', transA = 'N', diag = 'U', alpha = true)   # U12 = L11⁻¹ A12
            if pc + pb <= m
                gemm!(view(A, pc+pb:m, jt0:n), view(A, pc+pb:m, pc:pc+pb-1), view(A, pc:pc+pb-1, jt0:n);
                      alpha = -1, beta = true)                       # A22 −= L21 U12
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

# Complex LU (zgetrf): identical structure — no conj anywhere in L·U. `_getf2!`'s generic panel pivots on
# |·| and divides by the complex pivot correctly; `_getrf_core!` rides complex trsm!/gemm! (the 3M path)
# for the trailing update. The Float64 pad-scratch optimization (`_LU_PAD`) is real-only, so complex routes
# straight to the core. (Partial-pivot metric is |·| like LAPACK's cabs2, not cabs1 — both valid; oracle
# tests must compare the PA=LU residual, not entries.)
function getrf!(A::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer}; nb::Int = _LU_NB) where {T<:BlasComplex}
    m, n = size(A); k = min(m, n)
    k == 0 && return A, ipiv, 0
    length(ipiv) >= k || throw(DimensionMismatch("getrf!: length(ipiv) < min(size(A))"))
    return _getrf_core!(A, ipiv, nb)
end
# Convenience: allocate ipiv, return (A overwritten with L\U, ipiv, info).
function getrf!(A::StridedMatrix{Float64})
    ipiv = Vector{Int}(undef, min(size(A)...))
    return getrf!(A, ipiv)
end
function getrf!(A::StridedMatrix{T}) where {T<:BlasComplex}
    ipiv = Vector{Int}(undef, min(size(A)...))
    return getrf!(A, ipiv)
end
