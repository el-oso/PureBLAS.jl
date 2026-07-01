# BLAS-3 beyond gemm: trmm/trsm (triangular ·/solve), and later syrk/herk/syr2k/her2k/symm/hemm.
# Strategy — recursive 2×2 blocking that reuses the gate-passing `gemm!` for the off-diagonal update
# and bottoms out (block ≤ _TRMM_BASE) in the L2 kernels (trmv/trsv per B-column for side L;
# column-axpy/solve for side R). This is the L3 analogue of the trsv/trmv "diagonal block + gemv"
# decomposition: gemm carries the flops, the small triangular base carries the structure. α is applied
# as a final scale (kept out of the recursion). Generic `T<:Number` path via the L2 generic kernels.

const _TRMM_BASE = 128        # ≤ this → _trmm_small! directly (capped by _L3_NB=128 M scratch)
const _TRMM_RPANEL = 512
const _TRMM_RKC = 384
const _TRMM_RPACK = 448        # > this → packed single-pass side-R (mirrors the L cut at _GEMM_UNPACK_MAX)       # side-R flat-loop panel width (fat off-diag gemms; diagonal recurses)
@inline _trsplit(k::Int) = (k ÷ 2)                 # 2×2 split point
@inline _opchar(tr::Bool, cj::Bool) = tr ? (cj ? 'C' : 'T') : 'N'

# off-diagonal update C += op(A)·B — straight to the dispatch core (skip gemm!'s kwarg/check layer;
# the recursion guarantees the shapes).
@inline _gemm_acc!(C, A, B, tr::Bool, cj::Bool) =
    _gemm_core!(C, A, B, one(eltype(C)), one(eltype(C)), tr, false, cj, false)

# ── trmm side='L':  B := op(A)·B,  A k×k triangular (k=size(B,1)), unscaled ──────────────────────
# NOTE: trmm! routes large real side-L to the single-pass `_trmm_packed!` (the proven-fastest path); a
# cache-oblivious recursion here was measured SLOWER at every size, so this recursion is only the
# fallback for complex / side-R / small. (See memory anchor-fastest-path.)
# Dense small-k trmm base (side L): pivot-outer over contiguous columns of A and B — no per-column
# view/call. N-cases in axpy form (B[i,c]'s contribution scattered to its column band BEFORE it is
# overwritten: upper ascending / lower descending pivots); T-cases in dot form (B[i,c] rebuilt from the
# still-original band: upper descending / lower ascending). Both hit the gated SIMD L1 kernels. Real only.
function _trmm_dense_L!(up::Bool, tr::Bool, unit::Bool, A, B)
    k = size(A, 1); n = size(B, 2); T = eltype(B); sz = sizeof(T)
    lda = stride(A, 2); ldb = stride(B, 2)
    GC.@preserve A B begin
        pA = pointer(A); pB = pointer(B)
        @inbounds if !tr
            for i in (up ? (1:k) : (k:-1:1))
                len = up ? (i - 1) : (k - i); rs = up ? 1 : (i + 1)
                aptr = pA + ((i - 1) * lda + (rs - 1)) * sz
                d = unit ? one(T) : A[i, i]
                for c in 1:n
                    t = B[i, c]
                    len > 0 && _axpy_simd!(len, t, aptr, pB + ((c - 1) * ldb + (rs - 1)) * sz)
                    B[i, c] = d * t
                end
            end
        else
            for i in (up ? (k:-1:1) : (1:k))
                len = up ? (i - 1) : (k - i); rs = up ? 1 : (i + 1)
                aptr = pA + ((i - 1) * lda + (rs - 1)) * sz
                d = unit ? one(T) : A[i, i]
                for c in 1:n
                    s = len > 0 ? _dot_simd(len, aptr, pB + ((c - 1) * ldb + (rs - 1)) * sz, T) : zero(T)
                    B[i, c] = muladd(d, B[i, c], s)
                end
            end
        end
    end
    return B
end
# Materialized-triangle base: copy op(A)'s stored triangle into a dense scratch (other half zero, unit
# diag → 1) and run ONE gemm — 2× the triangle's flops but at gemm throughput, with no per-column calls.
# OB's trmm base is throughput-bound (a multiply, unlike trsm's sequential solve), so this is the base
# that keeps up; the recursion's true gemm off-diagonals bound the waste to ~base/k. Real non-conj.
function _mat_tri!(M, A, k::Int, up::Bool, tr::Bool, unit::Bool)
    T = eltype(M)
    @inbounds if !tr
        for j in 1:k                             # N: per column, copy the stored segment + zero the rest
            lo = up ? 1 : j; hi = up ? j : k     # (contiguous — the compiler vectorizes both loops)
            @simd for i in 1:(lo - 1); M[i, j] = zero(T); end
            @simd for i in lo:hi; M[i, j] = A[i, j]; end
            @simd for i in (hi + 1):k; M[i, j] = zero(T); end
            unit && (M[j, j] = one(T))
        end
    else                                         # T: transpose-on-store (strided source, scalar)
        for j in 1:k
            lo = up ? j : 1; hi = up ? k : j     # M column j = op(A) col j = A row j, stored part
            @simd for i in 1:(lo - 1); M[i, j] = zero(T); end
            for i in lo:hi; M[i, j] = A[j, i]; end
            @simd for i in (hi + 1):k; M[i, j] = zero(T); end
            unit && (M[j, j] = one(T))
        end
    end
    return M
end
const _TRMM_DDIRECT = 4      # ≤ this → dense substitution kernel (beats everything at tiny k)
# Small-k trmm at HALF flops and gemm throughput: materialize op(A) into a dense scratch (zeros in the
# unstored half make every read safe), copy B to scratch (in-place source), then run the UNPACKED gemm
# micro-kernels with a per-tile K-TRIM — each C-tile contracts only the p-range where M's triangle is
# nonzero, so the only waste is the mr×mr (or nr×nr) diagonal straddle. No packing, no per-column calls.
# Requires k ≤ _L3_NB (the M scratch); real non-conj.
function _trmm_small!(side_left::Bool, up::Bool, tr::Bool, unit::Bool, A, B)
    T = eltype(B); k = size(A, 1)
    upM = (up != tr)                                     # op(A)'s triangle after the on-store transpose
    M = _l3_tmp(T); _mat_tri!(M, A, k, up, tr, unit)     # full matrix scratch: ldM = _L3_NB, no view
    W = _vwidth(T); mr = _MR * W; nr = _NR; sz = sizeof(T)
    ldM = _L3_NB; ldb = stride(B, 2)
    if side_left                                         # B(k×n) := M·B, IN PLACE, dependency-ordered:
        n = size(B, 2)                                   # upM → top-down row-tiles (each reads rows ≥ its
        GC.@preserve M B begin                           # start, still untouched; registers hold the tile
            Mp = pointer(M); Bp = pointer(B)             # between read and store). lower → bottom-up.
            nt = cld(k, mr)
            for t in (upM ? (0:(nt - 1)) : ((nt - 1):-1:0))
                ir = t * mr; mre = min(mr, k - ir)
                plo = upM ? ir : 0
                phi = upM ? k : min(k, ir + mre)
                Ap = Mp + plo * ldM * sz; kc = phi - plo
                jr = 0
                while jr < n
                    nre = min(nr, n - jr)
                    # The B-operand aliases the store target. Full-strip kernels hold the whole tile in
                    # registers (safe). The EDGE kernel is W-row-block serial: for upper M the zero triangle
                    # exactly masks the stale rows; for lower M it does NOT — copy the strip's source
                    # columns to scratch first.
                    if mre == mr && nre == nr
                        _microkernel_unpacked!(Bp, ldb, Ap, ldM, ir, Bp + plo * sz, ldb, jr, kc,
                            one(T), zero(T), Val(_MR), Val(_NR), Val(false), Val(true))
                    elseif nre == nr
                        _microkernel_unpacked_mrows!(Bp, ldb, Ap, ldM, ir, Bp + plo * sz, ldb, jr, kc,
                            one(T), zero(T), mre, cld(mre, W) == 1 ? Val(1) : Val(_MR),
                            Val(_NR), Val(false), Val(true))
                    elseif upM
                        _microkernel_unpacked_edge!(Bp, ldb, Ap, ldM, ir, Bp + plo * sz, ldb, jr, kc,
                            one(T), zero(T), mre, nre, false, true)
                    else
                        Ec = _trsm_tmp(T, _L3_NB, nr)    # kc×nre source copy (dodges the serial aliasing)
                        lde = size(Ec, 1)
                        GC.@preserve Ec begin
                            Ep = pointer(Ec)
                            @inbounds for j in 0:(nre - 1), p in 0:(kc - 1)
                                unsafe_store!(Ep, unsafe_load(Bp, plo + p + (jr + j) * ldb + 1), p + j * lde + 1)
                            end
                            _microkernel_unpacked_edge!(Bp, ldb, Ap, ldM, ir, Ep - jr * lde * sz, lde, jr, kc,
                                one(T), zero(T), mre, nre, false, true)
                        end
                    end
                    jr += nr
                end
            end
        end
    else                                                 # B(m×k) := B·M, IN PLACE: upM → column-tiles
        m = size(B, 1)                                   # right-to-left (each reads cols ≤ its end, i.e.
        GC.@preserve M B begin                           # untouched to its left); lower → left-to-right.
            Mp = pointer(M); Bp = pointer(B)
            nt = cld(k, nr)
            # Row-blocks OUTER: the in-place hazard is row-local (each tile reads/writes only its own
            # rows), so row-blocks are independent — hoisting them keeps the 16×k A-slab L1-resident
            # across its column tiles instead of re-streaming all m×k per tile (the wide-m killer).
            ir = 0
            while ir < m
                mre = min(mr, m - ir)
                for t in (upM ? ((nt - 1):-1:0) : (0:(nt - 1)))
                    jr = t * nr; nre = min(nr, k - jr)
                    plo = upM ? 0 : jr
                    phi = upM ? min(k, jr + nre) : k
                    Bsp = Mp + plo * sz; kc = phi - plo
                    if mre == mr && nre == nr
                        _microkernel_unpacked!(Bp, ldb, Bp + plo * ldb * sz, ldb, ir, Bsp, ldM, jr, kc,
                            one(T), zero(T), Val(_MR), Val(_NR), Val(false), Val(true))
                    elseif nre == nr
                        _microkernel_unpacked_mrows!(Bp, ldb, Bp + plo * ldb * sz, ldb, ir, Bsp, ldM, jr, kc,
                            one(T), zero(T), mre, cld(mre, W) == 1 ? Val(1) : Val(_MR),
                            Val(_NR), Val(false), Val(true))
                    else
                        # Edge kernel is COLUMN-serial and the A-operand is B itself: column j+1's
                        # contraction re-reads columns already stored (they're inside [plo,phi) on both
                        # uplos). Compute the strip into a dest scratch, copy back after.
                        Ec = _trsm_tmp(T, mr, nr); lde = size(Ec, 1)
                        GC.@preserve Ec begin
                            Ep = pointer(Ec)
                            _microkernel_unpacked_edge!(Ep - (ir + jr * lde) * sz, lde,
                                Bp + plo * ldb * sz, ldb, ir, Bsp, ldM, jr, kc,
                                one(T), zero(T), mre, nre, false, true)
                            @inbounds for j in 0:(nre - 1), r in 0:(mre - 1)
                                unsafe_store!(Bp, unsafe_load(Ep, r + j * lde + 1),
                                    ir + r + (jr + j) * ldb + 1)
                            end
                        end
                    end
                end
                ir += mr
            end
        end
    end
    return B
end
function _trmm_left!(up::Bool, tr::Bool, cj::Bool, unit::Bool, A, B)
    k = size(A, 1)
    if eltype(B) <: BlasReal && !cj && k <= _TRMM_BASE
        return k <= _TRMM_DDIRECT ? _trmm_dense_L!(up, tr, unit, A, B) :
                                    _trmm_small!(true, up, tr, unit, A, B)
    elseif k <= _TRMM_BASE                          # complex/AD: trmv on each B column (contiguous)
        @inbounds for c in axes(B, 2)
            _trmv!(up, tr, cj, unit, k, A, view(B, :, c), 1)
        end
        return B
    end
    h = _trsplit(k)
    A11 = view(A, 1:h, 1:h); A22 = view(A, (h + 1):k, (h + 1):k)
    B1 = view(B, 1:h, :); B2 = view(B, (h + 1):k, :)
    # up≠tr → the off-diagonal feeds B1 (process B1's diagonal first, then gemm B1+=off·B2, then B2).
    # up==tr → it feeds B2. The off-diagonal A-block is A12 (above diag) or A21 (below), and gemm's
    # transA carries op. Verified against all four (uplo×trans) cases.
    if up != tr
        off = tr ? view(A, (h + 1):k, 1:h) : view(A, 1:h, (h + 1):k)
        _trmm_left!(up, tr, cj, unit, A11, B1)
        _gemm_acc!(B1, off, B2, tr, cj)
        _trmm_left!(up, tr, cj, unit, A22, B2)
    else
        off = tr ? view(A, 1:h, (h + 1):k) : view(A, (h + 1):k, 1:h)
        _trmm_left!(up, tr, cj, unit, A22, B2)
        _gemm_acc!(B2, off, B1, tr, cj)
        _trmm_left!(up, tr, cj, unit, A11, B1)
    end
    return B
end

# ── trmm side='R':  B := B·op(A),  A k×k triangular (k=size(B,2)), unscaled ───────────────────────
# Base: column-axpy on B's columns. For upper-N, B[:,j] := A[j,j]·B[:,j] + Σ_{i<j} A[i,j]·B[:,i]
# (j descending so B[:,i], i<j, are still original). The four combos mirror trmv's column structure.
function _trmm_right_base!(up::Bool, tr::Bool, cj::Bool, unit::Bool, k::Int, A, B)
    m = size(B, 1)
    # out[:,j] = Σ_i (op A)[i,j]·B[:,i].  (op A)[i,j] = A[i,j] (N) or A[j,i] (T/C). up≠tr ⇒ feeds are
    # the lower-index columns (j descending keeps them original); up==tr ⇒ higher-index (j ascending).
    coef(i, j) = tr ? (cj ? conj(A[j, i]) : A[j, i]) : A[i, j]
    @inbounds if up != tr
        for j in k:-1:1
            unit || _scal_col!(B, j, coef(j, j), m)
            for i in 1:(j - 1); _axpy_col!(B, j, coef(i, j), i, m); end
        end
    else
        for j in 1:k
            unit || _scal_col!(B, j, coef(j, j), m)
            for i in (j + 1):k; _axpy_col!(B, j, coef(i, j), i, m); end
        end
    end
    return B
end
# B[:,j] *= s   and   B[:,j] += a·B[:,i]  on contiguous columns (SIMD where eligible).
@inline function _scal_col!(B, j, s, m)
    if B isa StridedMatrix && stride(B, 1) == 1 && eltype(B) <: BlasReal
        GC.@preserve B (_scal_simd_ptr!(pointer(B) + (j - 1) * stride(B, 2) * sizeof(eltype(B)), m, s))
    else
        @inbounds for r in 1:m; B[r, j] *= s; end
    end
end
@inline function _axpy_col!(B, j, a, i, m)
    if B isa StridedMatrix && stride(B, 1) == 1 && eltype(B) <: BlasReal
        T = eltype(B); sz = sizeof(T); ldb = stride(B, 2)
        GC.@preserve B _axpy_simd!(m, T(a), pointer(B) + (i - 1) * ldb * sz, pointer(B) + (j - 1) * ldb * sz)
    else
        @inbounds for r in 1:m; B[r, j] += a * B[r, i]; end
    end
end
@inline function _scal_simd_ptr!(p::Ptr{T}, n::Int, s::T) where {T<:BlasReal}
    _scal!(n, s, p, 1)   # _scal! accepts a pointer (level1)
end

# Packed single-pass trmm side-R: B := B·op(A) as ONE K-trimmed blocked-gemm sweep (mirror of
# _trmm_packed!). B is copied once to a contiguous scratch (kills the in-place aliasing outright,
# O(mk) ≪ O(mk²/2)); op(A) packs per pc-block into nr-panels with zeros outside the triangle and only
# the rows each column-tile actually contracts (per-tile K-trim at nr granularity — the trim that kept
# the flat/recursion versions from gemm efficiency lived at panel granularity). Real non-conj.
const _TRMM_BCR = Ref(Matrix{Float64}(undef, 0, 0))
function _pack_B_triR!(Bp::Vector{T}, A, pc::Int, kce::Int, k::Int, upM::Bool, tr::Bool,
        unit::Bool, nr::Int) where {T}
    np = cld(k, nr)
    @inbounds for jp in 0:(np - 1)
        j0 = jp * nr
        plo = upM ? 0 : max(0, j0 - pc)                              # rows this panel's tiles contract
        phi = upM ? clamp(j0 + nr - pc, 0, kce) : kce
        plo >= phi && continue
        base = jp * nr * kce
        for p in plo:(phi - 1)
            gp = pc + p
            for c in 0:(nr - 1)
                gj = j0 + c
                v = if gj < k && (upM ? (gp <= gj) : (gp >= gj))
                    (unit && gp == gj) ? one(T) : (tr ? A[gj + 1, gp + 1] : A[gp + 1, gj + 1])
                else
                    zero(T)
                end
                Bp[base + p * nr + c + 1] = v
            end
        end
    end
    return
end
function _trmm_packedR!(up::Bool, tr::Bool, unit::Bool, A, B, ::Type{T}) where {T<:BlasReal}
    m, k = size(B); W = _vwidth(T); mr = _MR * W; nr = _NR
    upM = (up != tr)
    kc = min(_TRMM_RKC, k); mc = min(max(mr, (_MC ÷ mr) * mr), cld(m, mr) * mr)
    Ap, Bp = _gemm_scratch(T, cld(mc, mr) * mr * kc, cld(k, nr) * nr * kc)
    Bc = _TRMM_BCR[]
    (size(Bc, 1) < m || size(Bc, 2) < k) && (Bc = _TRMM_BCR[] = Matrix{Float64}(undef, m, k))
    ldb = stride(B, 2); ldc = stride(Bc, 2); sz = sizeof(T)
    GC.@preserve B Bc Ap Bp begin
        pB = pointer(B); pBc = pointer(Bc)
        @inbounds for j in 0:(k - 1)                                 # capture B (contiguous per-column)
            unsafe_copyto!(pBc + j * ldc * sz, pB + j * ldb * sz, m)
        end
        App = pointer(Ap); Bpp = pointer(Bp)
        pc = 0; pb = 0
        while pc < k
            kce = min(kc, k - pc)
            _pack_B_triR!(Bp, A, pc, kce, k, upM, tr, unit, nr)
            ic = 0
            while ic < m
                mce = min(mc, m - ic)
                _pack_A!(Ap, Bc, ic, pc, mce, kce, false, one(T), mr)
                jr = 0
                while jr < k
                    nre = min(nr, k - jr)
                    plo = upM ? 0 : max(0, jr - pc)                  # per-tile K-trim
                    phi = upM ? min(kce, jr + nre - pc) : kce
                    cnt = phi - plo
                    if cnt > 0
                        ow = upM ? (pb == 0) : (pb == div(jr, kc))   # first contributing block → β=0
                        ir = 0
                        while ir < mce
                            mre = min(mr, mce - ir)
                            Apanel = App + (div(ir, mr) * mr * kce + plo * mr) * sz
                            Bpanel = Bpp + (div(jr, nr) * nr * kce + plo * nr) * sz
                            Cblk = pB + ((ic + ir) + jr * ldb) * sz
                            if mre == mr && nre == nr
                                ow ? _microkernel!(Ptr{T}(Cblk), ldb, Ptr{T}(Apanel), Ptr{T}(Bpanel), cnt, Val(_MR), Val(_NR), Val(true)) :
                                     _microkernel!(Ptr{T}(Cblk), ldb, Ptr{T}(Apanel), Ptr{T}(Bpanel), cnt, Val(_MR), Val(_NR), Val(false))
                            else
                                ow ? _microkernel_masked!(Ptr{T}(Cblk), ldb, Ptr{T}(Apanel), Ptr{T}(Bpanel), cnt, mre, nre, Val(_MR), Val(_NR), Val(true)) :
                                     _microkernel_masked!(Ptr{T}(Cblk), ldb, Ptr{T}(Apanel), Ptr{T}(Bpanel), cnt, mre, nre, Val(_MR), Val(_NR), Val(false))
                            end
                            ir += mr
                        end
                    end
                    jr += nr
                end
                ic += mc
            end
            pc += kc; pb += 1
        end
    end
    return B
end
function _trmm_right!(up::Bool, tr::Bool, cj::Bool, unit::Bool, A, B)
    k = size(A, 1)
    if eltype(B) <: BlasReal && !cj && k <= _TRMM_BASE
        return k <= _TRMM_DDIRECT ? _trmm_right_base!(up, tr, cj, unit, k, A, B) :
                                    _trmm_small!(false, up, tr, unit, A, B)
    elseif B isa StridedMatrix{Float64} && stride(B, 1) == 1 && !cj && k > _TRMM_RPACK
        return _trmm_packedR!(up, tr, unit, A, B, Float64)
    elseif eltype(B) <: BlasReal && !cj
        # FLAT panel loop: each _TRMM_RPANEL-column panel of B gets ONE fat off-diagonal gemm on a
        # STORED rectangular A-view (transB carries op; no materialize) + a diagonal solved by the
        # halving recursion (→ _trmm_small! bases). Big panels keep the gemms fat (skinny n=128 gemms
        # measured 0.85 at 2048); the flat level touches B only twice. Diagonal FIRST (consumes the
        # panel's ORIGINAL values), then += off-diagonal (reads other, still-original panels).
        # upM → right-to-left; lower → left-to-right.
        upM = (up != tr); P = _TRMM_RPANEL
        np = cld(k, P)
        for t in (upM ? ((np - 1):-1:0) : (0:(np - 1)))
            jc = t * P; pc = min(P, k - jc)
            Bpan = view(B, :, (jc + 1):(jc + pc))
            Adia = view(A, (jc + 1):(jc + pc), (jc + 1):(jc + pc))
            _trmm_right_recur!(up, tr, cj, unit, Adia, Bpan)
            if upM && jc > 0                     # off-diag: Bpan += B[:,1:jc]·op(A)[1:jc, jc+1:jc+pc]
                Ablk = tr ? view(A, (jc + 1):(jc + pc), 1:jc) : view(A, 1:jc, (jc + 1):(jc + pc))
                _gemm_core!(Bpan, view(B, :, 1:jc), Ablk, one(eltype(B)), one(eltype(B)),
                    false, tr, false, false)
            elseif !upM && jc + pc < k           # off-diag: Bpan += B[:,jc+pc+1:k]·op(A)[jc+pc+1:k, …]
                Ablk = tr ? view(A, (jc + 1):(jc + pc), (jc + pc + 1):k) :
                            view(A, (jc + pc + 1):k, (jc + 1):(jc + pc))
                _gemm_core!(Bpan, view(B, :, (jc + pc + 1):k), Ablk, one(eltype(B)), one(eltype(B)),
                    false, tr, false, false)
            end
        end
        return B
    elseif k <= _TRMM_BASE
        return _trmm_right_base!(up, tr, cj, unit, k, A, B)
    end
    return _trmm_right_recur!(up, tr, cj, unit, A, B)
end
# Halving recursion (diagonal blocks of the flat loop + the complex/AD path).
function _trmm_right_recur!(up::Bool, tr::Bool, cj::Bool, unit::Bool, A, B)
    k = size(A, 1)
    if k <= _TRMM_BASE
        if eltype(B) <: BlasReal && !cj
            return k <= _TRMM_DDIRECT ? _trmm_right_base!(up, tr, cj, unit, k, A, B) :
                                        _trmm_small!(false, up, tr, unit, A, B)
        end
        return _trmm_right_base!(up, tr, cj, unit, k, A, B)
    end
    h = _trsplit(k)
    A11 = view(A, 1:h, 1:h); A22 = view(A, (h + 1):k, (h + 1):k)
    B1 = view(B, :, 1:h); B2 = view(B, :, (h + 1):k)
    # Mirror of left, with B column-blocks and transB carrying op(A). up≠tr → the off-diagonal feeds B2
    # (process B2's diagonal first, gemm B2+=B1·off, then B1); up==tr → feeds B1.
    if up != tr
        off = tr ? view(A, (h + 1):k, 1:h) : view(A, 1:h, (h + 1):k)
        _trmm_right_recur!(up, tr, cj, unit, A22, B2)
        _gemm_accR!(B2, B1, off, tr, cj)
        _trmm_right_recur!(up, tr, cj, unit, A11, B1)
    else
        off = tr ? view(A, 1:h, (h + 1):k) : view(A, (h + 1):k, 1:h)
        _trmm_right_recur!(up, tr, cj, unit, A11, B1)
        _gemm_accR!(B1, B2, off, tr, cj)
        _trmm_right_recur!(up, tr, cj, unit, A22, B2)
    end
    return B
end
# C += B·op(A): straight to the dispatch core (transB carries op; shapes guaranteed by the recursion).
@inline _gemm_accR!(C, Bmat, A, tr::Bool, cj::Bool) =
    _gemm_core!(C, Bmat, A, one(eltype(C)), one(eltype(C)), false, tr, false, cj)

# x := op(A)·x / op(A)⁻¹·x entry: B := α·op(A)·B (side L) or α·B·op(A) (side R), A triangular.
function _trmm!(side_left::Bool, up::Bool, tr::Bool, cj::Bool, unit::Bool, α::Number, A, B)
    if side_left
        _trmm_left!(up, tr, cj, unit, A, B)
    else
        _trmm_right!(up, tr, cj, unit, A, B)
    end
    isone(α) || _scal_all!(B, α)
    return B
end
@inline function _scal_all!(B, α)
    if B isa StridedMatrix && stride(B, 1) == 1
        GC.@preserve B _scal!(length(B), convert(eltype(B), α), pointer(B), 1)
    else
        B .*= α
    end
end

# Pack a triangular op(A) panel: zero the non-stored half, write the diagonal (unit ⇒ 1). packed_upper
# = the packed op(A) is upper-triangular (zero where gi>gp). Used only for diagonal-straddling A-panels
# (off-diagonal panels are fully stored → plain _pack_A!, fully-zero panels are skipped by the driver).
function _pack_A_tri!(Ap::Vector{T}, A, ic::Int, pc::Int, mce::Int, kce::Int, tA::Bool, unit::Bool,
        packed_upper::Bool, alpha::T, mr::Int) where {T}
    if !tA && A isa StridedMatrix{T} && stride(A, 1) == 1 && T <: BlasReal
        return _pack_A_tri_simd!(Ap, A, ic, pc, mce, kce, unit, packed_upper, alpha, mr)
    end
    np = cld(mce, mr)
    @inbounds for pi in 0:(np - 1)
        base = pi * mr * kce
        for p in 0:(kce - 1)
            for r in 0:(mr - 1)
                lr = pi * mr + r
                Ap[base + p * mr + r + 1] = if lr < mce
                    gi = ic + lr; gp = pc + p
                    if gi == gp
                        unit ? alpha : alpha * (tA ? A[gp + 1, gi + 1] : A[gi + 1, gp + 1])
                    elseif (packed_upper ? (gi > gp) : (gi < gp))
                        zero(T)
                    else
                        alpha * (tA ? A[gp + 1, gi + 1] : A[gi + 1, gp + 1])
                    end
                else
                    zero(T)
                end
            end
        end
    end
    return
end

# SIMD triangular A-pack (tA='N', dense unit-stride): per (mr-sub-panel, k-column) the stored rows are
# a prefix/suffix vs the diagonal threshold → vector load+scale + masked select (vifelse) to zero the
# rest; diagonal unit-fix is one scalar store. Lifts the straddling-panel packing to ~SIMD speed.
@inline function _pack_A_tri_simd!(Ap::Vector{T}, A, ic::Int, pc::Int, mce::Int, kce::Int, unit::Bool,
        packed_upper::Bool, alpha::T, mr::Int) where {T<:BlasReal}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T); lda = stride(A, 2); MR = mr ÷ W
    np = cld(mce, mr); lanes = Vec(ntuple(i -> i - 1, Val(W)))
    GC.@preserve A Ap begin
        Aptr = pointer(A); App = pointer(Ap); av = V(alpha); zv = zero(V)
        @inbounds for pi in 0:(np - 1)
            base = pi * mr * kce; r0g = ic + pi * mr; full = pi * mr + mr <= mce
            for p in 0:(kce - 1)
                gp = pc + p; rthr = gp - r0g; dst = App + (base + p * mr) * sz
                if full
                    src = Aptr + (r0g + gp * lda) * sz
                    for vi in 0:(MR - 1)
                        rows = lanes + vi * W
                        m = packed_upper ? (rows <= rthr) : (rows >= rthr)
                        vstore(vifelse(m, av * vload(V, src + vi * W * sz), zv), dst + vi * W * sz)
                    end
                else
                    for r in 0:(mr - 1)
                        lr = pi * mr + r; gi = ic + lr
                        Ap[base + p * mr + r + 1] = (lr < mce && (packed_upper ? gi <= gp : gi >= gp)) ?
                            alpha * A[gi + 1, gp + 1] : zero(T)
                    end
                end
                (unit && 0 <= rthr < mr && pi * mr + rthr < mce) && (Ap[base + p * mr + rthr + 1] = alpha)
            end
        end
    end
    return
end

# Single-pass packed trmm, side 'L': B := α·op(A)·B, A triangular m×m. = gemm(op(A_triangle), B) with
# A's non-stored half packed as zero (correct flops with the gemm microkernel). Per A-panel: skip
# fully-zero, plain _pack_A! fully-stored, _pack_A_tri! diagonal-straddling. Real only; α into the pack.
# IN-PLACE (no full B-copy): trmm-L columns are independent, so per jc column-panel we pack ALL of its
# pc-blocks into Bpf (capturing the input) BEFORE zeroing that panel of B — the pack itself is the copy,
# so the separate Bc scratch is gone. (Bpf holds the whole panel: nblk pc-blocks × one packed block.)
const _TRMM_BPF = IdDict{DataType, Vector}()
function _trmm_bpf(::Type{T}, len::Int) where {T}
    v = get!(() -> T[], _TRMM_BPF, T)::Vector{T}
    length(v) < len && resize!(v, len)
    return v
end
function _trmm_packed!(up::Bool, tr::Bool, unit::Bool, α::T, A, B, ::Val{MRV} = Val(_MR)) where {T<:BlasReal, MRV}
    m = size(B, 1); n = size(B, 2); W = _vwidth(T); mr = MRV * W; nr = _NR
    packed_upper = (up != tr)
    kc = min(_KC, m); mc = min(max(mr, (_MC ÷ mr) * mr), cld(m, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(n, nr) * nr)
    nblk = cld(m, kc); bpf_blk = cld(nc, nr) * nr * kc          # one packed pc-block slot (padded to kc)
    Ap, _ = _gemm_scratch(T, cld(mc, mr) * mr * kc, 1)
    Bpf = _trmm_bpf(T, nblk * bpf_blk)
    ldc = stride(B, 2); sz = sizeof(T)
    GC.@preserve B Ap Bpf begin
        Cp0 = pointer(B); App = pointer(Ap); Bfp = pointer(Bpf)
        jc = 0
        while jc < n
            nce = min(nc, n - jc)
            pc = 0; pb = 0                                       # Phase 1: pack whole jc-panel of B
            while pc < m
                kce = min(kc, m - pc)
                _pack_B!(Bpf, B, pc, jc, kce, nce, false, nr, pb * bpf_blk)
                pc += kc; pb += 1
            end
            pc = 0; pb = 0                                       # Phase 2: compute from Bpf (no zero pass:
            while pc < m                                          # each tile's FIRST contribution overwrites)
                kce = min(kc, m - pc)
                ic = 0
                while ic < m
                    mce = min(mc, m - ic); a_hi = ic + mce - 1; p_hi = pc + kce - 1
                    zpanel = packed_upper ? (ic > p_hi) : (a_hi < pc)
                    if !zpanel
                        stored = packed_upper ? (a_hi < pc) : (ic > p_hi)
                        stored ? _pack_A!(Ap, A, ic, pc, mce, kce, tr, α, mr) :
                                 _pack_A_tri!(Ap, A, ic, pc, mce, kce, tr, unit, packed_upper, α, mr)
                        jr = 0
                        while jr < nce
                            nre = min(nr, nce - jr); ir = 0
                            while ir < mce
                                mre = min(mr, mce - ir); r0 = ic + ir
                                plo = stored ? 0 : (packed_upper ? max(0, r0 - pc) : 0)
                                cnt = stored ? kce : (packed_upper ? kce - plo : min(kce, r0 + mre - pc))
                                if cnt > 0
                                    # this pc-block is the tile's FIRST contribution → overwrite (β=0),
                                    # no zero pass + no C read. upper: first block = div(r0,kc); lower: pb 0.
                                    ow = packed_upper ? (pb == div(r0, kc)) : (pb == 0)
                                    Apanel = App + (div(ir, mr) * mr * kce + plo * mr) * sz
                                    Bpanel = Bfp + (pb * bpf_blk + div(jr, nr) * nr * kce + plo * nr) * sz
                                    Cblk = Ptr{T}(Cp0 + ((ic + ir) + (jc + jr) * ldc) * sz)
                                    if mre == mr && nre == nr
                                        ow ? _microkernel!(Cblk, ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), cnt, Val(MRV), Val(_NR), Val(true)) :
                                             _microkernel!(Cblk, ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), cnt, Val(MRV), Val(_NR), Val(false))
                                    else
                                        ow ? _microkernel_masked!(Cblk, ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), cnt, mre, nre, Val(MRV), Val(_NR), Val(true)) :
                                             _microkernel_masked!(Cblk, ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), cnt, mre, nre, Val(MRV), Val(_NR), Val(false))
                                    end
                                end
                                ir += mr
                            end
                            jr += nr
                        end
                    end
                    ic += mc
                end
                pc += kc; pb += 1
            end
            jc += nc
        end
    end
    return B
end

# Public: B := α·op(A)·B (side 'L') or α·B·op(A) (side 'R'); A k×k triangular (uplo/transA/diag).
function trmm!(B::AbstractMatrix, A::AbstractMatrix; side::Char = 'L', uplo::Char = 'U',
        transA::Char = 'N', diag::Char = 'N', alpha::Number = true)
    sl = side == 'L'
    k = sl ? size(B, 1) : size(B, 2)
    (size(A, 1) == size(A, 2) == k) || throw(DimensionMismatch("trmm!: A must be $k×$k"))
    # side-L real large → K-range-trimmed single-pass packed (the straddling tile contracts only its
    # nonzero p-band, not the full kc zero-band — that band was the ~kc/k waste that capped the naive
    # packed trmm). Else (side R, complex/AD, small) → recursion-over-gemm! (no regression).
    if sl && eltype(B) <: BlasReal && transA != 'C' && k > _GEMM_UNPACK_MAX
        # 8×8 tile (Val(1), unified W==_NR): finer K-trim staircase + smaller within-tile zero triangle;
        # the proven-fastest, most consistent path across sizes. (A 16×8 bulk helped N-cases at large k
        # but regressed k=768 and the public po2 A-pad path — non-robust, not worth the split.)
        mrv = _unified_ok(eltype(B)) ? Val(1) : Val(_MR)
        # NOTE: no A-pad here (unlike trsm). trmm's po2-ld conflict is mild (~2%); the O(k²) A-copy to
        # pad it costs about the same, so padding is net-negative for trmm — measured. (trsm's conflict
        # was catastrophic 0.78→1.12, there the copy pays.)
        _trmm_packed!(uplo == 'U', transA != 'N', diag == 'U', convert(eltype(B), alpha), A, B, mrv)
    else
        _trmm!(sl, uplo == 'U', transA != 'N', transA == 'C', diag == 'U', alpha, A, B)
    end
    return B
end

# ──────────────────────────────────────────────────────────────────────────────────────────────
# trsm: B := α·op(A)⁻¹·B (side 'L') / α·B·op(A)⁻¹ (side 'R'), A triangular. Same recursive blocking
# as trmm, but: (1) solve the independent block FIRST, (2) the off-diagonal update SUBTRACTS the
# already-solved block (gemm α=-1,β=1), (3) the base is a triangular solve (trsv per column / column
# substitution). α is applied to B up front (B := α·op(A)⁻¹·B = op(A)⁻¹·(αB)).
@inline _gemm_sub!(C, A, B, tr::Bool, cj::Bool) =                                    # C -= op(A)·B
    _gemm_core!(C, A, B, -one(eltype(C)), one(eltype(C)), tr, false, cj, false)
@inline _gemm_subR!(C, Bmat, A, tr::Bool, cj::Bool) =                                # C -= B·op(A)
    _gemm_core!(C, Bmat, A, -one(eltype(C)), one(eltype(C)), false, tr, false, cj)

# trsm base via small triangular INVERSE + gemm (BLIS-style): a block ≤ _TRSM_BASE is solved by
# inverting its NB×NB triangle once (O(NB³/6), tiny) then applying op(inv) as a gemm — so the diagonal
# solve runs at gemm speed instead of scalar back-substitution. The recursion's off-diagonal updates
# are already gemm!. Real only (stability fine for the well-conditioned diagonal blocks trsm assumes);
# complex/conj keep the scalar trsv base.
const _TRSM_BASE = 32
# Small real triangular inverse: V (same uplo as A) = inv(A). V[i,i]=1/A[i,i] (or 1 if unit);
# off-diagonal V[i,j] = -V[i,i]·Σ A[i,l]V[l,j] over the stored band between i and j.
function _trtri!(V, A, nb::Int, up::Bool, unit::Bool) where {}
    T = eltype(V)
    if up
        @inbounds for j in 1:nb
            for i in (j + 1):nb; V[i, j] = zero(T); end     # zero the non-stored (lower) half
            V[j, j] = unit ? one(T) : inv(A[j, j])
            for i in (j - 1):-1:1
                s = zero(T)
                for l in (i + 1):j; s += A[i, l] * V[l, j]; end
                V[i, j] = -V[i, i] * s
            end
        end
    else
        @inbounds for j in nb:-1:1
            for i in 1:(j - 1); V[i, j] = zero(T); end       # zero the non-stored (upper) half
            V[j, j] = unit ? one(T) : inv(A[j, j])
            for i in (j + 1):nb
                s = zero(T)
                for l in j:(i - 1); s += A[i, l] * V[l, j]; end
                V[i, j] = -V[i, i] * s
            end
        end
    end
    return V
end
const _TRSM_TMP = IdDict{DataType, Matrix}()
function _trsm_tmp(::Type{T}, m::Int, n::Int) where {T}
    b = get(_TRSM_TMP, T, nothing)
    if isnothing(b) || size(b, 1) < m || size(b, 2) < n; b = Matrix{T}(undef, m, n); _TRSM_TMP[T] = b; end
    return b
end
# side L base: B := op(A)⁻¹·B = op(inv(A))·B (gemm with transA=op into temp, copy back).
function _trsm_base_invL!(up::Bool, tr::Bool, unit::Bool, A, B)
    nb = size(A, 1); n = size(B, 2); T = eltype(B)
    iv = view(_l3_tmp(T), 1:nb, 1:nb); _trtri!(iv, A, nb, up, unit)
    tmp = view(_trsm_tmp(T, nb, n), 1:nb, 1:n)
    gemm!(tmp, iv, B; alpha = true, beta = false, transA = tr ? 'T' : 'N')
    copyto!(B, tmp); return B
end
# side R base: B := B·op(A)⁻¹ = B·op(inv(A)) (gemm with transB=op into temp, copy back).
function _trsm_base_invR!(up::Bool, tr::Bool, unit::Bool, A, B)
    nb = size(A, 1); m = size(B, 1); T = eltype(B)
    iv = view(_l3_tmp(T), 1:nb, 1:nb); _trtri!(iv, A, nb, up, unit)
    tmp = view(_trsm_tmp(T, m, nb), 1:m, 1:nb)
    gemm!(tmp, B, iv; alpha = true, beta = false, transB = tr ? 'T' : 'N')
    copyto!(B, tmp); return B
end

# Direct triangular solve base (side L): rank-1 substitution, the eliminate-rows axpy dispatched to the gated
# 4-way-unrolled `_axpy_simd!` (no-trans; trans strided → scalar). n³/2 flops (half of invert+gemm), no gemm
# dispatch. Real non-conj; forward when up==tr. Used as the base ONLY when B is narrow — the per-column axpy
# count grows with n, so for wide B the invL/gemm base wins (routed by _TRSM_NCUT below).
const _TRSM_NCUT = 96          # side-L: B width cut
const _TRSM_NCUT_R = 128       # side-R: B height cut (R's narrow path is stronger than L's — measured, 128 rides it at 1.7×)
const _TRSM_DBASE = 16
function _trsm_dense_L!(up::Bool, tr::Bool, unit::Bool, A, B)
    k = size(A, 1); n = size(B, 2); T = eltype(B); sz = sizeof(T)
    lda = stride(A, 2); ldb = stride(B, 2); fwd = (up == tr)
    GC.@preserve A B begin
        pA = pointer(A); pB = pointer(B)
        @inbounds for i in (fwd ? (1:k) : (k:-1:1))
            if !unit
                d = inv(A[i, i]); for c in 1:n; B[i, c] *= d; end
            end
            rlen = fwd ? (k - i) : (i - 1); rlen == 0 && continue
            rs = fwd ? (i + 1) : 1
            if tr
                rows = fwd ? ((i + 1):k) : (1:(i - 1))
                for c in 1:n; bic = B[i, c]; @simd for r in rows; B[r, c] -= A[i, r] * bic; end; end
            else
                aptr = pA + ((i - 1) * lda + (rs - 1)) * sz
                for c in 1:n
                    _axpy_simd!(rlen, -B[i, c], aptr, pB + ((c - 1) * ldb + (rs - 1)) * sz)
                end
            end
        end
    end
    return B
end
# side 'L': B := op(A)⁻¹·B, A k×k (k=size(B,1)), unscaled.
function _trsm_left!(up::Bool, tr::Bool, cj::Bool, unit::Bool, A, B)
    k = size(A, 1)
    if eltype(B) <: BlasReal && !cj
        # narrow B → dense base (few axpy calls); wide B → invL base (gemm-efficient). n is invariant under
        # the side-L row split, so the choice is consistent through the recursion.
        if size(B, 2) <= _TRSM_NCUT
            k <= _TRSM_DBASE && return _trsm_dense_L!(up, tr, unit, A, B)
        elseif k <= _TRSM_BASE
            return _trsm_base_invL!(up, tr, unit, A, B)
        end
    elseif k <= _TRMM_BASE
        @inbounds for c in axes(B, 2)
            _trsv!(up, tr, cj, unit, k, A, view(B, :, c), 1)
        end
        return B
    end
    h = _trsplit(k)
    A11 = view(A, 1:h, 1:h); A22 = view(A, (h + 1):k, (h + 1):k)
    B1 = view(B, 1:h, :); B2 = view(B, (h + 1):k, :)
    if up != tr                                       # solve B2, then B1 -= off·B2, then solve B1
        off = tr ? view(A, (h + 1):k, 1:h) : view(A, 1:h, (h + 1):k)
        _trsm_left!(up, tr, cj, unit, A22, B2)
        _gemm_sub!(B1, off, B2, tr, cj)
        _trsm_left!(up, tr, cj, unit, A11, B1)
    else                                              # solve B1, then B2 -= off·B1, then solve B2
        off = tr ? view(A, 1:h, (h + 1):k) : view(A, (h + 1):k, 1:h)
        _trsm_left!(up, tr, cj, unit, A11, B1)
        _gemm_sub!(B2, off, B1, tr, cj)
        _trsm_left!(up, tr, cj, unit, A22, B2)
    end
    return B
end

# side 'R' base: X·op(A)=B by column substitution. up≠tr ⇒ ascending (feeds lower-index columns);
# up==tr ⇒ descending. Subtract solved columns, then divide by the diagonal (unless unit).
function _trsm_right_base!(up::Bool, tr::Bool, cj::Bool, unit::Bool, k::Int, A, B)
    m = size(B, 1)
    coef(i, j) = tr ? (cj ? conj(A[j, i]) : A[j, i]) : A[i, j]
    @inbounds if up != tr
        for j in 1:k
            for i in 1:(j - 1); _axpy_col!(B, j, -coef(i, j), i, m); end
            unit || _scal_col!(B, j, inv(coef(j, j)), m)
        end
    else
        for j in k:-1:1
            for i in (j + 1):k; _axpy_col!(B, j, -coef(i, j), i, m); end
            unit || _scal_col!(B, j, inv(coef(j, j)), m)
        end
    end
    return B
end

# Dense side-R base: solve X·op(A)=B column by column. Eliminate a solved column via _axpy_simd! over the
# contiguous B columns (closure-free, unlike _trsm_right_base!); divide by the diagonal via _scal. Real
# non-conj; ascending columns when up≠tr. Used only for narrow B (few rows) — few axpy calls.
function _trsm_dense_R!(up::Bool, tr::Bool, unit::Bool, A, B)
    m = size(B, 1); k = size(A, 2); T = eltype(B); sz = sizeof(T); ldb = stride(B, 2)
    asc = (up != tr)
    GC.@preserve A B begin
        pB = pointer(B)
        @inbounds for j in (asc ? (1:k) : (k:-1:1))
            pj = pB + (j - 1) * ldb * sz
            for l in (asc ? (1:(j - 1)) : ((j + 1):k))
                coef = tr ? A[j, l] : A[l, j]
                coef == zero(T) || _axpy_simd!(m, -coef, pB + (l - 1) * ldb * sz, pj)
            end
            unit || _scal_simd_ptr!(pj, m, inv(tr ? A[j, j] : A[j, j]))
        end
    end
    return B
end
# side 'R': B := B·op(A)⁻¹, A k×k (k=size(B,2)), unscaled.
function _trsm_right!(up::Bool, tr::Bool, cj::Bool, unit::Bool, A, B)
    k = size(A, 1)
    if eltype(B) <: BlasReal && !cj
        # narrow B (few rows) → dense column-substitution base; wide → invR/gemm base. m is invariant
        # under the side-R column split. (Same dense/gemm split as side L, routed by _TRSM_NCUT_R.)
        if size(B, 1) <= _TRSM_NCUT_R
            k <= _TRSM_DBASE && return _trsm_dense_R!(up, tr, unit, A, B)
        elseif k <= _TRSM_BASE
            return _trsm_base_invR!(up, tr, unit, A, B)
        end
    elseif k <= _TRMM_BASE
        return _trsm_right_base!(up, tr, cj, unit, k, A, B)
    end
    h = _trsplit(k)
    A11 = view(A, 1:h, 1:h); A22 = view(A, (h + 1):k, (h + 1):k)
    B1 = view(B, :, 1:h); B2 = view(B, :, (h + 1):k)
    if up != tr                                       # solve B1, then B2 -= B1·off, then solve B2
        off = tr ? view(A, (h + 1):k, 1:h) : view(A, 1:h, (h + 1):k)
        _trsm_right!(up, tr, cj, unit, A11, B1)
        _gemm_subR!(B2, B1, off, tr, cj)
        _trsm_right!(up, tr, cj, unit, A22, B2)
    else                                              # solve B2, then B1 -= B2·off, then solve B1
        off = tr ? view(A, 1:h, (h + 1):k) : view(A, (h + 1):k, 1:h)
        _trsm_right!(up, tr, cj, unit, A22, B2)
        _gemm_subR!(B1, B2, off, tr, cj)
        _trsm_right!(up, tr, cj, unit, A11, B1)
    end
    return B
end

function _trsm!(side_left::Bool, up::Bool, tr::Bool, cj::Bool, unit::Bool, α::Number, A, B)
    isone(α) || _scal_all!(B, α)
    side_left ? _trsm_left!(up, tr, cj, unit, A, B) : _trsm_right!(up, tr, cj, unit, A, B)
    return B
end

# When A's leading dim is a pure power of 2 (≥512), packing its triangular sub-views thrashes one cache
# set (column starts alias) — measured trsm 0.78–0.94 at ld∈{1024,2048} vs 1.0–1.12 at non-po2. Copying
# A once into a padded-ld scratch (ld=k+8) removes the conflict (B-padding doesn't help — it's the A
# sub-view packing). ponytail: only A needs it; B is solved in place. Cost O(k²) ≪ trsm O(k²n).
@inline _badld(ld::Int) = ld >= 512 && (ld & (ld - 1)) == 0
const _L3_APAD = IdDict{DataType, Matrix}()
function _l3_apad(::Type{T}, k::Int) where {T}
    b = get(_L3_APAD, T, nothing)
    if isnothing(b) || size(b, 1) < k + 8 || size(b, 2) < k; b = Matrix{T}(undef, k + 8, k); _L3_APAD[T] = b; end
    return view(b, 1:k, 1:k)        # ld = k+8 (non-power-of-2)
end

# Public: B := α·op(A)⁻¹·B (side 'L') or α·B·op(A)⁻¹ (side 'R'); A k×k triangular (uplo/transA/diag).
function trsm!(B::AbstractMatrix, A::AbstractMatrix; side::Char = 'L', uplo::Char = 'U',
        transA::Char = 'N', diag::Char = 'N', alpha::Number = true)
    sl = side == 'L'
    k = sl ? size(B, 1) : size(B, 2)
    (size(A, 1) == size(A, 2) == k) || throw(DimensionMismatch("trsm!: A must be $k×$k"))
    # tiny-k fast path: skip the _trsm!/_trsm_left!/_trsm_right! dispatch chain (~3 non-inlined calls ≈ 60ns,
    # which dominates when the solve itself is only ~100ns) and go straight to the dense base kernel.
    if k <= _TRSM_DBASE && eltype(B) <: BlasReal && transA != 'C' && isone(alpha)
        up = uplo == 'U'; tr = transA != 'N'; unit = diag == 'U'
        return sl ? _trsm_dense_L!(up, tr, unit, A, B) : _trsm_dense_R!(up, tr, unit, A, B)
    end
    if eltype(B) <: BlasReal && transA != 'C' && k > _GEMM_UNPACK_MAX &&
       A isa StridedMatrix && stride(A, 1) == 1 && _badld(stride(A, 2))
        Apad = _l3_apad(eltype(B), k); copyto!(Apad, A)
        _trsm!(sl, uplo == 'U', transA != 'N', false, diag == 'U', alpha, Apad, B)
    else
        _trsm!(sl, uplo == 'U', transA != 'N', transA == 'C', diag == 'U', alpha, A, B)
    end
    return B
end

# ──────────────────────────────────────────────────────────────────────────────────────────────
# syrk/herk: C := α·op(A)·op(A)ᴴ + β·C, C n×n, only the `uplo` triangle referenced/updated.
# trans 'N': op(A)=A (n×k) ⇒ A·Aᴴ. trans 'T'/'C': op(A)=Aᴴ (A k×n) ⇒ Aᴴ·A. syrk: ᵀ (no conj),
# any T<:Number. herk: Hermitian (conj), real α/β, diagonal forced real. Recursive: diagonal blocks
# recurse (scalar base), the off-diagonal block is a full gemm! — breadth-first correctness (gate later).
const _SYRK_BASE = 48

@inline _symstored(up::Bool, i, j) = up ? (i <= j) : (i >= j)
function _syrk_scaleC!(C, up::Bool, β)
    @inbounds for j in axes(C, 2), i in axes(C, 1)
        _symstored(up, i, j) && (C[i, j] = iszero(β) ? zero(eltype(C)) : β * C[i, j])
    end
end
# Reusable NB×NB scratch for L3 diagonal blocks (one per element type; thread-unsafe — single-thread
# project for now). ponytail: per-type cached buffer; revisit for multithreading.
const _L3_NB = 128
const _L3_TMP = IdDict{DataType, Matrix}()
_l3_tmp(::Type{T}) where {T} = get!(() -> Matrix{T}(undef, _L3_NB, _L3_NB), _L3_TMP, T)::Matrix{T}
# The IdDict lookup costs ~130 ns — more than an entire tiny trmm. Const-dispatch the gated types.
const _L3_TMP_F64 = Matrix{Float64}(undef, _L3_NB, _L3_NB)
const _L3_TMP_F32 = Matrix{Float32}(undef, _L3_NB, _L3_NB)
@inline _l3_tmp(::Type{Float64}) = _L3_TMP_F64
@inline _l3_tmp(::Type{Float32}) = _L3_TMP_F32

# Triangular-store microkernel: same FMA as the gemm masked microkernel, but on store keeps only the
# stored-triangle entries — for a diagonal-straddling C-tile whose top-left global offset is (r0,c0),
# d0=c0-r0; upper keeps local row ≤ d0+j, lower keeps row ≥ d0+j (j = 0-based column). Accumulates into
# C, so K-accumulation across the gemm pc-loop stays correct (no temp needed).
@generated function _microkernel_tri!(C::Ptr{T}, ldc::Int, Ap::Ptr{T}, Bp::Ptr{T}, kc::Int,
        mre::Int, nre::Int, d0::Int, upper::Bool, ::Val{MR}, ::Val{NR}) where {T, MR, NR}
    W = _vwidth(T); sz = sizeof(T); V = Vec{W, T}
    body = quote end
    push!(body.args, :(lanes = Vec{$W, Int}($(Expr(:tuple, (0:(W - 1))...)))))
    for mi in 1:MR, j in 1:NR; push!(body.args, :($(Symbol(:c, mi, :_, j)) = zero($V))); end
    inner = quote end
    for mi in 1:MR
        push!(inner.args, :($(Symbol(:a, mi)) = vload($V, Ap + (p * $MR + $(mi - 1)) * $(W * sz))))
    end
    for j in 1:NR
        push!(inner.args, :($(Symbol(:b, j)) = $V(unsafe_load(Bp + (p * $NR + $(j - 1)) * $sz))))
        for mi in 1:MR
            cs = Symbol(:c, mi, :_, j); push!(inner.args, :($cs = muladd($(Symbol(:a, mi)), $(Symbol(:b, j)), $cs)))
        end
    end
    push!(body.args, :(for p in 0:(kc - 1); $inner; end))
    for j in 1:NR
        stores = quote end
        push!(stores.args, :(colp = C + $(j - 1) * ldc * $sz)); push!(stores.args, :(thr = d0 + $(j - 1)))
        for mi in 1:MR
            cs = Symbol(:c, mi, :_, j)
            push!(stores.args, :(let base = $((mi - 1) * W), q = colp + $((mi - 1) * W * sz)
                rows = lanes + base
                mk = (rows < mre) & (upper ? (rows <= thr) : (rows >= thr))
                vstore(vload($V, q, mk) + $cs, q, mk)
            end))
        end
        push!(body.args, :(if $(j - 1) < nre; $stores; end))
    end
    push!(body.args, :(return nothing))
    return body
end

# Two-product fused microkernel for syr2k: C += op(X1)op(Y1) + op(X2)op(Y2) for ONE C-tile, with a
# SINGLE C read-modify-write (not two). Both products' panels share the kc accumulation in the same
# registers; only at the end is C touched. This halves C traffic and (in :tri mode) the masked store
# vs running two separate microkernels per tile. MODE picks the store: :full / :masked / :tri.
@generated function _microkernel2!(C::Ptr{T}, ldc::Int, Ap1::Ptr{T}, Bp1::Ptr{T}, Ap2::Ptr{T},
        Bp2::Ptr{T}, kc::Int, alpha::T, mre::Int, nre::Int, d0::Int, upper::Bool,
        ::Val{MR}, ::Val{NR}, ::Val{MODE}) where {T, MR, NR, MODE}
    W = _vwidth(T); sz = sizeof(T); V = Vec{W, T}
    body = quote end
    push!(body.args, :(av = $V(alpha)))
    if MODE !== :full
        push!(body.args, :(lanes = Vec{$W, Int}($(Expr(:tuple, (0:(W - 1))...)))))
    end
    for j in 1:NR; push!(body.args, :(_prefetch(C + $(j - 1) * ldc * $sz))); end
    for mi in 1:MR, j in 1:NR; push!(body.args, :($(Symbol(:c, mi, :_, j)) = zero($V))); end
    inner = quote end
    for mi in 1:MR
        push!(inner.args, :($(Symbol(:a, mi)) = vload($V, Ap1 + (p * $MR + $(mi - 1)) * $(W * sz))))
        push!(inner.args, :($(Symbol(:e, mi)) = vload($V, Ap2 + (p * $MR + $(mi - 1)) * $(W * sz))))
    end
    for j in 1:NR
        push!(inner.args, :($(Symbol(:b, j)) = $V(unsafe_load(Bp1 + (p * $NR + $(j - 1)) * $sz))))
        push!(inner.args, :($(Symbol(:f, j)) = $V(unsafe_load(Bp2 + (p * $NR + $(j - 1)) * $sz))))
        for mi in 1:MR
            cs = Symbol(:c, mi, :_, j)
            push!(inner.args, :($cs = muladd($(Symbol(:a, mi)), $(Symbol(:b, j)), $cs)))
            push!(inner.args, :($cs = muladd($(Symbol(:e, mi)), $(Symbol(:f, j)), $cs)))
        end
    end
    push!(body.args, :(for p in 0:(kc - 1); $inner; end))
    for j in 1:NR
        stores = quote end
        push!(stores.args, :(colp = C + $(j - 1) * ldc * $sz))
        MODE === :tri && push!(stores.args, :(thr = d0 + $(j - 1)))
        for mi in 1:MR
            cs = Symbol(:c, mi, :_, j)
            if MODE === :full
                push!(stores.args, :(let q = colp + $((mi - 1) * W * sz); vstore(muladd(av, $cs, vload($V, q)), q); end))
            elseif MODE === :masked
                push!(stores.args, :(let q = colp + $((mi - 1) * W * sz)
                    mk = (lanes + $((mi - 1) * W)) < mre; vstore(muladd(av, $cs, vload($V, q, mk)), q, mk); end))
            else # :tri
                push!(stores.args, :(let q = colp + $((mi - 1) * W * sz)
                    rows = lanes + $((mi - 1) * W)
                    mk = (rows < mre) & (upper ? (rows <= thr) : (rows >= thr))
                    vstore(muladd(av, $cs, vload($V, q, mk)), q, mk); end))
            end
        end
        if MODE === :full
            push!(body.args, stores)
        else
            push!(body.args, :(if $(j - 1) < nre; $stores; end))
        end
    end
    push!(body.args, :(return nothing))
    return body
end

# Single-product α-at-store microkernel for the UNIFIED syrk path: C += α·(A·B), with α applied at the
# store (the unified path packs A once and reads it as both operands, so α cannot be folded into the
# pack — both operands would pick it up, giving α²). MODE: :full / :masked / :tri.
@generated function _microkernel_u!(C::Ptr{T}, ldc::Int, Ap::Ptr{T}, Bp::Ptr{T}, kc::Int, alpha::T,
        mre::Int, nre::Int, d0::Int, upper::Bool, ::Val{MR}, ::Val{NR}, ::Val{MODE}) where {T, MR, NR, MODE}
    W = _vwidth(T); sz = sizeof(T); V = Vec{W, T}
    body = quote end
    push!(body.args, :(av = $V(alpha)))
    if MODE !== :full
        push!(body.args, :(lanes = Vec{$W, Int}($(Expr(:tuple, (0:(W - 1))...)))))
    end
    for j in 1:NR; push!(body.args, :(_prefetch(C + $(j - 1) * ldc * $sz))); end
    for mi in 1:MR, j in 1:NR; push!(body.args, :($(Symbol(:c, mi, :_, j)) = zero($V))); end
    inner = quote end
    for mi in 1:MR
        push!(inner.args, :($(Symbol(:a, mi)) = vload($V, Ap + (p * $MR + $(mi - 1)) * $(W * sz))))
    end
    for j in 1:NR
        push!(inner.args, :($(Symbol(:b, j)) = $V(unsafe_load(Bp + (p * $NR + $(j - 1)) * $sz))))
        for mi in 1:MR
            cs = Symbol(:c, mi, :_, j)
            push!(inner.args, :($cs = muladd($(Symbol(:a, mi)), $(Symbol(:b, j)), $cs)))
        end
    end
    push!(body.args, :(for p in 0:(kc - 1); $inner; end))
    for j in 1:NR
        stores = quote end
        push!(stores.args, :(colp = C + $(j - 1) * ldc * $sz))
        MODE === :tri && push!(stores.args, :(thr = d0 + $(j - 1)))
        for mi in 1:MR
            cs = Symbol(:c, mi, :_, j)
            if MODE === :full
                push!(stores.args, :(let q = colp + $((mi - 1) * W * sz); vstore(muladd(av, $cs, vload($V, q)), q); end))
            elseif MODE === :masked
                push!(stores.args, :(let q = colp + $((mi - 1) * W * sz)
                    mk = (lanes + $((mi - 1) * W)) < mre; vstore(muladd(av, $cs, vload($V, q, mk)), q, mk); end))
            else # :tri
                push!(stores.args, :(let q = colp + $((mi - 1) * W * sz)
                    rows = lanes + $((mi - 1) * W)
                    mk = (rows < mre) & (upper ? (rows <= thr) : (rows >= thr))
                    vstore(muladd(av, $cs, vload($V, q, mk)), q, mk); end))
            end
        end
        if MODE === :full
            push!(body.args, stores)
        else
            push!(body.args, :(if $(j - 1) < nre; $stores; end))
        end
    end
    push!(body.args, :(return nothing))
    return body
end

# Single-pass packed syrk (the gate path for large n): syrk = gemm(A, Aᴴ) with a triangular C. Reuses
# gemm's packing + microkernel; classifies each micro-tile vs the diagonal — skip below-diagonal,
# regular/masked microkernel fully-stored, triangular-store microkernel straddling. Packs A once per
# (ic,pc) panel (reads A like a single gemm — no recursion re-reads). Real (BlasReal) only; α folded
# into the packed A by _pack_A!. C's stored triangle must be β-pre-scaled by the caller.
# General triangular-C gemm: C[uplo-triangle] += α·op(X)·op(Y) (X→A-operand, Y→B-operand), n×n result.
# The reusable core behind syrk (Y=X) and syr2k (two passes). Real only; α folded into packed X.
function _trgemm_packed!(up::Bool, α::T, X, tXp::Bool, Y, tYp::Bool, C, k::Int) where {T<:BlasReal}
    n = size(C, 1); W = _vwidth(T); mr = _MR * W; nr = _NR
    kc = min(_KC, k); mc = min(max(mr, (_MC ÷ mr) * mr), cld(n, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(n, nr) * nr)
    Ap, Bp = _gemm_scratch(T, cld(mc, mr) * mr * kc, cld(nc, nr) * nr * kc)
    ldc = stride(C, 2); sz = sizeof(T)
    GC.@preserve C Ap Bp begin
        Cp0 = pointer(C); App = pointer(Ap); Bpp = pointer(Bp)
        jc = 0
        while jc < n
            nce = min(nc, n - jc); pc = 0
            while pc < k
                kce = min(kc, k - pc)
                _pack_B!(Bp, Y, pc, jc, kce, nce, tYp, nr)
                ic = 0
                while ic < n
                    mce = min(mc, n - ic)
                    _pack_A!(Ap, X, ic, pc, mce, kce, tXp, α, mr)
                    jr = 0
                    while jr < nce
                        nre = min(nr, nce - jr); ir = 0
                        while ir < mce
                            mre = min(mr, mce - ir); r0 = ic + ir; c0 = jc + jr
                            skip = up ? (r0 > c0 + nre - 1) : (r0 + mre - 1 < c0)
                            if !skip
                                Apanel = App + (div(ir, mr) * mr * kce) * sz
                                Bpanel = Bpp + (div(jr, nr) * nr * kce) * sz
                                Cblk = Cp0 + (r0 + c0 * ldc) * sz
                                full = up ? (r0 + mre - 1 <= c0) : (r0 >= c0 + nre - 1)
                                if full && mre == mr && nre == nr
                                    _microkernel!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, Val(_MR), Val(_NR))
                                elseif full
                                    _microkernel_masked!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, mre, nre, Val(_MR), Val(_NR))
                                else
                                    _microkernel_tri!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, mre, nre, c0 - r0, up, Val(_MR), Val(_NR))
                                end
                            end
                            ir += mr
                        end
                        jr += nr
                    end
                    ic += mc
                end
                pc += kc
            end
            jc += nc
        end
    end
    return C
end

# Add S's `uplo` triangle into C's; herm → force the diagonal real.
function _add_tri!(C, S, up::Bool, herm::Bool, b::Int)
    @inbounds for j in 1:b, i in (up ? (1:j) : (j:b)); C[i, j] += S[i, j]; end
    herm && @inbounds for i in 1:b; C[i, i] = real(C[i, i]); end
    return C
end
# syrk = one triangular-C gemm (Y = X = A). syr2k = two (A·Bᴴ + B·Aᴴ); real ⇒ both use α.
@inline _syrk_packed!(up::Bool, tr::Bool, α::T, A, C, k::Int) where {T<:BlasReal} =
    _unified_ok(T) ? _trgemm_packed_u!(up, α, A, tr, C, k) :
        _trgemm_packed!(up, α, A, tr, A, !tr, C, k)

# Four-buffer scratch for the fused two-product syr2k driver (two A-packs, two B-packs). ponytail:
# single global, single-thread; make task-local with the gemm scratch for M4 threading.
const _SYR2K_SCRATCH = Dict{DataType, NTuple{4, Vector}}()
function _syr2k_scratch(::Type{T}, lenA::Int, lenB::Int) where {T}
    t = get!(() -> (T[], T[], T[], T[]), _SYR2K_SCRATCH, T)::NTuple{4, Vector{T}}
    length(t[1]) < lenA && (resize!(t[1], lenA); resize!(t[3], lenA))
    length(t[2]) < lenB && (resize!(t[2], lenB); resize!(t[4], lenB))
    return t
end

# Fused two-product triangular-C gemm: C[tri] += α·op(X1)·op(Y1) + α·op(X2)·op(Y2). The core of syr2k.
# Both products are packed (X1,Y1,X2,Y2) and each C-tile is visited ONCE: _microkernel2! accumulates
# both products in registers and does a single C read-modify-write. Running two _trgemm_packed! passes
# instead touches every C-tile twice (the microkernel loads/stores C per call) — measured 2.05× a syrk
# vs OpenBLAS's ~1.93×. This fused tile-pass removes that second C round-trip.
function _trgemm_packed2!(up::Bool, α::T, X1, tX1::Bool, Y1, tY1::Bool,
        X2, tX2::Bool, Y2, tY2::Bool, C, k::Int, ::Val{MRV} = Val(_MR)) where {T<:BlasReal, MRV}
    n = size(C, 1); W = _vwidth(T); mr = MRV * W; nr = _NR
    kc = min(_KC, k); mc = min(max(mr, (_MC ÷ mr) * mr), cld(n, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(n, nr) * nr)
    Ap1, Bp1, Ap2, Bp2 = _syr2k_scratch(T, cld(mc, mr) * mr * kc, cld(nc, nr) * nr * kc)
    ldc = stride(C, 2); sz = sizeof(T)
    GC.@preserve C Ap1 Bp1 Ap2 Bp2 begin
        Cp0 = pointer(C); A1p = pointer(Ap1); B1p = pointer(Bp1); A2p = pointer(Ap2); B2p = pointer(Bp2)
        jc = 0
        while jc < n
            nce = min(nc, n - jc); pc = 0
            while pc < k
                kce = min(kc, k - pc)
                _pack_B!(Bp1, Y1, pc, jc, kce, nce, tY1, nr)
                _pack_B!(Bp2, Y2, pc, jc, kce, nce, tY2, nr)
                ic = 0
                while ic < n
                    mce = min(mc, n - ic)
                    _pack_A!(Ap1, X1, ic, pc, mce, kce, tX1, one(T), mr)
                    _pack_A!(Ap2, X2, ic, pc, mce, kce, tX2, one(T), mr)
                    jr = 0
                    while jr < nce
                        nre = min(nr, nce - jr); ir = 0
                        while ir < mce
                            mre = min(mr, mce - ir); r0 = ic + ir; c0 = jc + jr
                            skip = up ? (r0 > c0 + nre - 1) : (r0 + mre - 1 < c0)
                            if !skip
                                off = div(ir, mr) * mr * kce; boff = div(jr, nr) * nr * kce
                                a1 = A1p + off * sz; b1 = B1p + boff * sz
                                a2 = A2p + off * sz; b2 = B2p + boff * sz
                                Cblk = Ptr{T}(Cp0 + (r0 + c0 * ldc) * sz)
                                full = up ? (r0 + mre - 1 <= c0) : (r0 >= c0 + nre - 1)
                                if full && mre == mr && nre == nr
                                    _microkernel2!(Cblk, ldc, Ptr{T}(a1), Ptr{T}(b1), Ptr{T}(a2), Ptr{T}(b2), kce, α, mre, nre, 0, up, Val(MRV), Val(_NR), Val(:full))
                                elseif full
                                    _microkernel2!(Cblk, ldc, Ptr{T}(a1), Ptr{T}(b1), Ptr{T}(a2), Ptr{T}(b2), kce, α, mre, nre, 0, up, Val(MRV), Val(_NR), Val(:masked))
                                else
                                    _microkernel2!(Cblk, ldc, Ptr{T}(a1), Ptr{T}(b1), Ptr{T}(a2), Ptr{T}(b2), kce, α, mre, nre, c0 - r0, up, Val(MRV), Val(_NR), Val(:tri))
                                end
                            end
                            ir += mr
                        end
                        jr += nr
                    end
                    ic += mc
                end
                pc += kc
            end
            jc += nc
        end
    end
    return C
end
# Unified single-pack works only when the A-pack panel width (mr=W) equals the B-pack width (nr=_NR),
# i.e. an 8×8 tile — true for Float64 on AVX-512 (W=8=_NR). Other ISAs/types fall back to the 16×8
# multi-pack drivers.
@inline _unified_ok(::Type{T}) where {T} = _vwidth(T) == _NR

# Unified single-pack syrk: pack A ONCE into W-row panels; the A-operand (vector load, panel ir) and
# the B-operand (scalar broadcast, panel jr) both read that one buffer. 8×8 tile (MR=1) so both packs'
# layouts coincide; α applied at the store (shared buffer ⇒ can't fold α into the pack).
function _trgemm_packed_u!(up::Bool, α::T, A, tAp::Bool, C, k::Int) where {T<:BlasReal}
    n = size(C, 1); W = _vwidth(T); mr = W; nr = _NR
    kc = min(_KC, k); mc = min(max(mr, (_MC ÷ mr) * mr), cld(n, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(n, nr) * nr)
    plen = cld(n, mr) * mr * kc
    pk = _syr2k_scratch(T, plen, plen); packA = pk[1]
    ldc = stride(C, 2); sz = sizeof(T)
    GC.@preserve C packA begin
        Cp0 = pointer(C); PA = pointer(packA)
        jc = 0
        while jc < n
            nce = min(nc, n - jc); pc = 0
            while pc < k
                kce = min(kc, k - pc); pstr = mr * kce
                _pack_A!(packA, A, 0, pc, n, kce, tAp, one(T), mr)
                ic = 0
                while ic < n
                    mce = min(mc, n - ic); jr = 0
                    while jr < nce
                        nre = min(nr, nce - jr); ir = 0
                        while ir < mce
                            mre = min(mr, mce - ir); r0 = ic + ir; c0 = jc + jr
                            skip = up ? (r0 > c0 + nre - 1) : (r0 + mre - 1 < c0)
                            if !skip
                                a = PA + div(r0, mr) * pstr * sz; b = PA + div(c0, nr) * pstr * sz
                                Cblk = Ptr{T}(Cp0 + (r0 + c0 * ldc) * sz)
                                full = up ? (r0 + mre - 1 <= c0) : (r0 >= c0 + nre - 1)
                                if full && mre == mr && nre == nr
                                    _microkernel_u!(Cblk, ldc, Ptr{T}(a), Ptr{T}(b), kce, α, mre, nre, 0, up, Val(1), Val(_NR), Val(:full))
                                elseif full
                                    _microkernel_u!(Cblk, ldc, Ptr{T}(a), Ptr{T}(b), kce, α, mre, nre, 0, up, Val(1), Val(_NR), Val(:masked))
                                else
                                    _microkernel_u!(Cblk, ldc, Ptr{T}(a), Ptr{T}(b), kce, α, mre, nre, c0 - r0, up, Val(1), Val(_NR), Val(:tri))
                                end
                            end
                            ir += mr
                        end
                        jr += nr
                    end
                    ic += mc
                end
                pc += kc
            end
            jc += nc
        end
    end
    return C
end

# Unified single-pack syr2k: pack A and B ONCE each (W-row panels); the two products read them in
# swapped roles (A·Bᵀ: packA-rows·packB-cols; B·Aᵀ: packB-rows·packA-cols). 2 packs, not 4.
function _trgemm_packed2_u!(up::Bool, α::T, A, tAp::Bool, Bm, tBp::Bool, C, k::Int) where {T<:BlasReal}
    n = size(C, 1); W = _vwidth(T); mr = W; nr = _NR
    kc = min(_KC, k); mc = min(max(mr, (_MC ÷ mr) * mr), cld(n, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(n, nr) * nr)
    plen = cld(n, mr) * mr * kc
    pk = _syr2k_scratch(T, plen, plen); packA = pk[1]; packB = pk[2]
    ldc = stride(C, 2); sz = sizeof(T)
    GC.@preserve C packA packB begin
        Cp0 = pointer(C); PA = pointer(packA); PB = pointer(packB)
        jc = 0
        while jc < n
            nce = min(nc, n - jc); pc = 0
            while pc < k
                kce = min(kc, k - pc); pstr = mr * kce
                _pack_A!(packA, A, 0, pc, n, kce, tAp, one(T), mr)
                _pack_A!(packB, Bm, 0, pc, n, kce, tBp, one(T), mr)
                ic = 0
                while ic < n
                    mce = min(mc, n - ic); jr = 0
                    while jr < nce
                        nre = min(nr, nce - jr); ir = 0
                        while ir < mce
                            mre = min(mr, mce - ir); r0 = ic + ir; c0 = jc + jr
                            skip = up ? (r0 > c0 + nre - 1) : (r0 + mre - 1 < c0)
                            if !skip
                                ip = div(r0, mr) * pstr * sz; jp = div(c0, nr) * pstr * sz
                                a1 = PA + ip; b1 = PB + jp; a2 = PB + ip; b2 = PA + jp
                                Cblk = Ptr{T}(Cp0 + (r0 + c0 * ldc) * sz)
                                full = up ? (r0 + mre - 1 <= c0) : (r0 >= c0 + nre - 1)
                                if full && mre == mr && nre == nr
                                    _microkernel2!(Cblk, ldc, Ptr{T}(a1), Ptr{T}(b1), Ptr{T}(a2), Ptr{T}(b2), kce, α, mre, nre, 0, up, Val(1), Val(_NR), Val(:full))
                                elseif full
                                    _microkernel2!(Cblk, ldc, Ptr{T}(a1), Ptr{T}(b1), Ptr{T}(a2), Ptr{T}(b2), kce, α, mre, nre, 0, up, Val(1), Val(_NR), Val(:masked))
                                else
                                    _microkernel2!(Cblk, ldc, Ptr{T}(a1), Ptr{T}(b1), Ptr{T}(a2), Ptr{T}(b2), kce, α, mre, nre, c0 - r0, up, Val(1), Val(_NR), Val(:tri))
                                end
                            end
                            ir += mr
                        end
                        jr += nr
                    end
                    ic += mc
                end
                pc += kc
            end
            jc += nc
        end
    end
    return C
end

@inline _syr2k_packed!(up::Bool, tr::Bool, α::T, A, Bm, C, k::Int) where {T<:BlasReal} =
    _unified_ok(T) ? _trgemm_packed2_u!(up, α, A, tr, Bm, tr, C, k) :
        (tr ? _trgemm_packed2!(up, α, A, true, Bm, false, Bm, true, A, false, C, k) :
              _trgemm_packed2!(up, α, A, false, Bm, true, Bm, false, A, true, C, k))

# Recursive blocked syrk/herk (the gate path): split into 2×2; the two diagonal blocks recurse and the
# off-diagonal block is one large gemm! written straight into C's stored triangle (correct flops, no
# temp). Only the small diagonal BASE (≤ _SYRK_DBASE) goes through a gemm→temp + triangle-add, so the
# unavoidable "compute the full b×b but keep the triangle" waste is confined to tiny base blocks
# (≈ 2·DBASE/n of the flops). Large off-diagonal gemms keep the bulk at peak.
const _SYRK_DBASE = 32
# Large real syrk → single-pass packed (gate); small / complex / herk → recursion (gemm-temp base).
function _syrk_blocked!(up::Bool, tr::Bool, herm::Bool, α, A, C, k::Int)
    if !herm && eltype(C) <: BlasReal && size(C, 1) > _GEMM_UNPACK_MAX && k > 0
        return _syrk_packed!(up, tr, convert(eltype(C), α), A, C, k)
    end
    _syrk_rec!(up, tr, herm, α, A, C, k, _l3_tmp(eltype(C)))
end
function _syrk_rec!(up::Bool, tr::Bool, herm::Bool, α, A, C, k::Int, scr)
    n = size(C, 1); tB = herm ? 'C' : 'T'
    if n <= _SYRK_DBASE
        tmp = view(scr, 1:n, 1:n)
        tr ? gemm!(tmp, A, A; alpha = α, beta = false, transA = tB) :
             gemm!(tmp, A, A; alpha = α, beta = false, transB = tB)
        return _add_tri!(C, tmp, up, herm, n)
    end
    h = _trsplit(n)
    C11 = view(C, 1:h, 1:h); C22 = view(C, (h + 1):n, (h + 1):n)
    A1 = tr ? view(A, :, 1:h) : view(A, 1:h, :); A2 = tr ? view(A, :, (h + 1):n) : view(A, (h + 1):n, :)
    _syrk_rec!(up, tr, herm, α, A1, C11, k, scr)
    _syrk_rec!(up, tr, herm, α, A2, C22, k, scr)
    if up
        Coff = view(C, 1:h, (h + 1):n)
        tr ? gemm!(Coff, A1, A2; alpha = α, beta = true, transA = tB) :
             gemm!(Coff, A1, A2; alpha = α, beta = true, transB = tB)
    else
        Coff = view(C, (h + 1):n, 1:h)
        tr ? gemm!(Coff, A2, A1; alpha = α, beta = true, transA = tB) :
             gemm!(Coff, A2, A1; alpha = α, beta = true, transB = tB)
    end
    return C
end

_syrk_dims(C, A, trans) = (n = size(C, 1); size(C, 2) == n ||
        throw(DimensionMismatch("syrk!: C must be square")); k = trans == 'N' ? size(A, 2) : size(A, 1);
        (trans == 'N' ? size(A, 1) : size(A, 2)) == n || throw(DimensionMismatch("syrk!: op(A) rows ≠ n")); (n, k))

function syrk!(C::AbstractMatrix, A::AbstractMatrix; uplo::Char = 'U', trans::Char = 'N',
        alpha::Number = true, beta::Number = false)
    n, k = _syrk_dims(C, A, trans); up = uplo == 'U'
    _syrk_scaleC!(C, up, beta)
    _syrk_blocked!(up, trans != 'N', false, alpha, A, C, k)
    return C
end
function herk!(C::AbstractMatrix, A::AbstractMatrix; uplo::Char = 'U', trans::Char = 'N',
        alpha::Real = true, beta::Real = false)
    n, k = _syrk_dims(C, A, trans); up = uplo == 'U'
    _syrk_scaleC!(C, up, beta)
    _syrk_blocked!(up, trans != 'N', true, alpha, A, C, k)
    return C
end

# ──────────────────────────────────────────────────────────────────────────────────────────────
# symm/hemm: C := α·A·B + β·C (side 'L', A symmetric/Hermitian n×n) or C := α·B·A + β·C (side 'R',
# A m×m). Only the `uplo` triangle of A is referenced. Diagonal blocks recurse; the off-diagonal A
# block feeds both halves (once as-is, once transposed) via gemm! — the matrix analogue of symv.
@inline function _asym(up::Bool, herm::Bool, A, i, l)
    i == l && return herm ? oftype(A[i, i], real(A[i, i])) : A[i, i]
    _symstored(up, i, l) ? A[i, l] : (herm ? conj(A[l, i]) : A[l, i])
end
# symm's output C is a FULL matrix (no triangle), so symm = gemm with a materialized full symmetric
# A — correct flops (NO 2× waste, unlike syrk). Materialize the symmetric/Hermitian A into a dense
# scratch (O(n²), amortized over the O(n²·m) gemm), then one gemm! carries α and β directly.
const _SYMM_SCR = IdDict{DataType, Matrix}()
function _symm_scr(::Type{T}, n::Int) where {T}
    m = get(_SYMM_SCR, T, nothing)
    if isnothing(m) || size(m, 1) < n; m = Matrix{T}(undef, n, n); _SYMM_SCR[T] = m; end
    return m
end
# Const-dispatch the gated types (the IdDict get costs ~130 ns — dominates tiny symm).
const _SYMM_SCR_F64 = Ref(Matrix{Float64}(undef, 0, 0))
const _SYMM_SCR_F32 = Ref(Matrix{Float32}(undef, 0, 0))
@inline function _symm_scr(::Type{Float64}, n::Int)
    m = _SYMM_SCR_F64[]
    size(m, 1) < n && (m = Matrix{Float64}(undef, n, n); _SYMM_SCR_F64[] = m)
    return m
end
@inline function _symm_scr(::Type{Float32}, n::Int)
    m = _SYMM_SCR_F32[]
    size(m, 1) < n && (m = Matrix{Float32}(undef, n, n); _SYMM_SCR_F32[] = m)
    return m
end
# Branch-free symmetric/Hermitian → dense fill: copy the stored triangle (contiguous column segments),
# then mirror it to the other triangle. No per-element uplo/diagonal branch in the hot path.
function _symm_materialize!(Ad, up::Bool, herm::Bool, A, n::Int)
    @inbounds if up
        for j in 1:n
            for i in 1:j; Ad[i, j] = A[i, j]; end                       # stored upper col (contiguous)
            for i in 1:(j - 1); Ad[j, i] = herm ? conj(A[i, j]) : A[i, j]; end   # mirror to lower
        end
    else
        for j in 1:n
            for i in j:n; Ad[i, j] = A[i, j]; end                       # stored lower col (contiguous)
            for i in (j + 1):n; Ad[j, i] = herm ? conj(A[i, j]) : A[i, j]; end   # mirror to upper
        end
    end
    herm && @inbounds for i in 1:n; Ad[i, i] = real(Ad[i, i]); end
    return Ad
end
# Symmetric A-pack for a diagonal-straddling panel (real symm): stored side reads A[gi,gp], mirror
# side A[gp,gi]. Off-diagonal panels use plain _pack_A! (stored: tA=false SIMD; mirror: tA=true).
function _pack_A_sym!(Ap::Vector{T}, A, ic::Int, pc::Int, mce::Int, kce::Int, up::Bool, alpha::T, mr::Int) where {T}
    np = cld(mce, mr)
    @inbounds for pi in 0:(np - 1)
        base = pi * mr * kce
        for p in 0:(kce - 1)
            for r in 0:(mr - 1)
                lr = pi * mr + r
                Ap[base + p * mr + r + 1] = if lr < mce
                    gi = ic + lr; gp = pc + p
                    alpha * ((up ? gi <= gp : gi >= gp) ? A[gi + 1, gp + 1] : A[gp + 1, gi + 1])
                else
                    zero(T)
                end
            end
        end
    end
    return
end
# Symmetric B-pack for a diagonal-straddling panel (real symm side R): the symmetric matrix is the
# gemm's RIGHT operand. Stored side reads A[gp,gj], mirror side A[gj,gp]. Off-diagonal panels use
# plain _pack_B! (stored: tB=false; mirror: tB=true). No α here — α rides on the left operand's pack.
function _pack_B_sym!(Bp::Vector{T}, A, pc::Int, jc::Int, kce::Int, nce::Int, up::Bool, nr::Int) where {T}
    np = cld(nce, nr)
    @inbounds for ji in 0:(np - 1)
        base = ji * nr * kce
        for p in 0:(kce - 1)
            for c in 0:(nr - 1)
                lc = ji * nr + c
                Bp[base + p * nr + c + 1] = if lc < nce
                    gj = jc + lc; gp = pc + p
                    (up ? gp <= gj : gp >= gj) ? A[gp + 1, gj + 1] : A[gj + 1, gp + 1]
                else
                    zero(T)
                end
            end
        end
    end
    return
end

# Single-pass packed symm (side L, real): C := α·A_sym·B + β·C as one gemm, packing A's symmetric
# panels directly (no n² materialize). M=n, N=m, K=n; classify each A-panel: stored / mirror / straddle.
function _symm_packed_L!(up::Bool, α::T, β::T, A, B, C) where {T<:BlasReal}
    n = size(C, 1); m = size(C, 2); W = _vwidth(T); mr = _MR * W; nr = _NR
    kc = min(_KC, n); mc = min(max(mr, (_MC ÷ mr) * mr), cld(n, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(m, nr) * nr)
    Ap, Bp = _gemm_scratch(T, cld(mc, mr) * mr * kc, cld(nc, nr) * nr * kc)
    _scale_C!(C, n, m, β); ldc = stride(C, 2); sz = sizeof(T)
    GC.@preserve C Ap Bp begin
        Cp0 = pointer(C); App = pointer(Ap); Bpp = pointer(Bp)
        jc = 0
        while jc < m
            nce = min(nc, m - jc); pc = 0
            while pc < n
                kce = min(kc, n - pc)
                _pack_B!(Bp, B, pc, jc, kce, nce, false, nr)
                ic = 0
                while ic < n
                    mce = min(mc, n - ic); a_hi = ic + mce - 1; p_hi = pc + kce - 1
                    stored = up ? (a_hi <= pc) : (ic >= p_hi)
                    mirror = up ? (ic > p_hi) : (a_hi < pc)
                    stored ? _pack_A!(Ap, A, ic, pc, mce, kce, false, α, mr) :
                        mirror ? _pack_A!(Ap, A, ic, pc, mce, kce, true, α, mr) :
                                 _pack_A_sym!(Ap, A, ic, pc, mce, kce, up, α, mr)
                    jr = 0
                    while jr < nce
                        nre = min(nr, nce - jr); ir = 0
                        while ir < mce
                            mre = min(mr, mce - ir)
                            Apanel = App + (div(ir, mr) * mr * kce) * sz
                            Bpanel = Bpp + (div(jr, nr) * nr * kce) * sz
                            Cblk = Cp0 + ((ic + ir) + (jc + jr) * ldc) * sz
                            if mre == mr && nre == nr
                                _microkernel!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, Val(_MR), Val(_NR))
                            else
                                _microkernel_masked!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, mre, nre, Val(_MR), Val(_NR))
                            end
                            ir += mr
                        end
                        jr += nr
                    end
                    ic += mc
                end
                pc += kc
            end
            jc += nc
        end
    end
    return C
end

# Single-pass packed symm (side R, real): C := α·B·A_sym + β·C. A_sym is the gemm's RIGHT operand.
# M=size(C,1), N=K=n; classify each A_sym panel (pc..K, jc..N): stored / mirror / straddle.
function _symm_packed_R!(up::Bool, α::T, β::T, B, A, C) where {T<:BlasReal}
    M = size(C, 1); n = size(A, 1); W = _vwidth(T); mr = _MR * W; nr = _NR
    kc = min(_KC, n); mc = min(max(mr, (_MC ÷ mr) * mr), cld(M, mr) * mr)
    nc = min(max(nr, (_NC ÷ nr) * nr), cld(n, nr) * nr)
    Ap, Bp = _gemm_scratch(T, cld(mc, mr) * mr * kc, cld(nc, nr) * nr * kc)
    _scale_C!(C, M, n, β); ldc = stride(C, 2); sz = sizeof(T)
    GC.@preserve C Ap Bp begin
        Cp0 = pointer(C); App = pointer(Ap); Bpp = pointer(Bp)
        jc = 0
        while jc < n
            nce = min(nc, n - jc); j_hi = jc + nce - 1; pc = 0
            while pc < n
                kce = min(kc, n - pc); p_hi = pc + kce - 1
                stored = up ? (p_hi <= jc) : (pc >= j_hi)
                mirror = up ? (pc > j_hi) : (p_hi < jc)
                stored ? _pack_B!(Bp, A, pc, jc, kce, nce, false, nr) :
                    mirror ? _pack_B!(Bp, A, pc, jc, kce, nce, true, nr) :
                             _pack_B_sym!(Bp, A, pc, jc, kce, nce, up, nr)
                ic = 0
                while ic < M
                    mce = min(mc, M - ic)
                    _pack_A!(Ap, B, ic, pc, mce, kce, false, α, mr)
                    jr = 0
                    while jr < nce
                        nre = min(nr, nce - jr); ir = 0
                        while ir < mce
                            mre = min(mr, mce - ir)
                            Apanel = App + (div(ir, mr) * mr * kce) * sz
                            Bpanel = Bpp + (div(jr, nr) * nr * kce) * sz
                            Cblk = Cp0 + ((ic + ir) + (jc + jr) * ldc) * sz
                            if mre == mr && nre == nr
                                _microkernel!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, Val(_MR), Val(_NR))
                            else
                                _microkernel_masked!(Ptr{T}(Cblk), ldc, Ptr{T}(Apanel), Ptr{T}(Bpanel), kce, mre, nre, Val(_MR), Val(_NR))
                            end
                            ir += mr
                        end
                        jr += nr
                    end
                    ic += mc
                end
                pc += kc
            end
            jc += nc
        end
    end
    return C
end

function _symm!(side_left::Bool, up::Bool, herm::Bool, α, β, A, B, C)
    n = size(A, 1)
    if !herm && eltype(C) <: BlasReal && n > _GEMM_UNPACK_MAX
        return side_left ?
            _symm_packed_L!(up, convert(eltype(C), α), convert(eltype(C), β), A, B, C) :
            _symm_packed_R!(up, convert(eltype(C), α), convert(eltype(C), β), B, A, C)
    end
    Ad = view(_symm_scr(eltype(C), n), 1:n, 1:n)
    _symm_materialize!(Ad, up, herm, A, n)
    if eltype(C) <: BlasReal && !herm     # real: straight to the dispatch core (skip the kwarg layer)
        T = eltype(C)
        side_left ? _gemm_core!(C, Ad, B, T(α), T(β), false, false, false, false) :
                    _gemm_core!(C, B, Ad, T(α), T(β), false, false, false, false)
    else
        side_left ? gemm!(C, Ad, B; alpha = α, beta = β) : gemm!(C, B, Ad; alpha = α, beta = β)
    end
    return C
end
function _symm_check(side_left, A, B, C)
    (size(C) == size(B)) || throw(DimensionMismatch("symm!: C and B must match"))
    k = side_left ? size(B, 1) : size(B, 2)
    (size(A, 1) == size(A, 2) == k) || throw(DimensionMismatch("symm!: A must be $k×$k"))
end
function symm!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix; side::Char = 'L',
        uplo::Char = 'U', alpha::Number = true, beta::Number = false)
    _symm_check(side == 'L', A, B, C); _symm!(side == 'L', uplo == 'U', false, alpha, beta, A, B, C); C
end
function hemm!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix; side::Char = 'L',
        uplo::Char = 'U', alpha::Number = true, beta::Number = false)
    _symm_check(side == 'L', A, B, C); _symm!(side == 'L', uplo == 'U', true, alpha, beta, A, B, C); C
end

# ──────────────────────────────────────────────────────────────────────────────────────────────
# syr2k/her2k: C := α·op(A)·op(B)ᴴ + (α or ᾱ)·op(B)·op(A)ᴴ + β·C, only `uplo` triangle of C.
# trans 'N': A,B are n×k (A·Bᴴ + B·Aᴴ). 'T'/'C': A,B are k×n (Aᴴ·B + Bᴴ·A). her2k: conj + β real +
# diagonal real, second term uses ᾱ. Diagonal blocks recurse; off-diagonal = two gemm!s.
# gemm-temp base (the gate path): both rank-k products into an n×n temp, then triangle-add.
function _syr2k_base!(up::Bool, tr::Bool, herm::Bool, α, A, B, C, n::Int, scr)
    α2 = herm ? conj(α) : α; tX = herm ? 'C' : 'T'; tmp = view(scr, 1:n, 1:n)
    if !herm && eltype(C) <: BlasReal
        # Real: the second product IS the transpose of the first (B·Aᵀ = (A·Bᵀ)ᵀ, and Bᵀ·A = (Aᵀ·B)ᵀ),
        # so ONE gemm + a symmetrized triangle-add — halves the base's gemm work. Dispatch-core direct.
        T = eltype(C)
        tr ? _gemm_core!(tmp, A, B, T(α), zero(T), true, false, false, false) :
             _gemm_core!(tmp, A, B, T(α), zero(T), false, true, false, false)
        @inbounds for j in 1:n, i in (up ? (1:j) : (j:n))
            C[i, j] += tmp[i, j] + tmp[j, i]
        end
        return C
    elseif !tr
        gemm!(tmp, A, B; alpha = α, beta = false, transB = tX); gemm!(tmp, B, A; alpha = α2, beta = true, transB = tX)
    else
        gemm!(tmp, A, B; alpha = α, beta = false, transA = tX); gemm!(tmp, B, A; alpha = α2, beta = true, transA = tX)
    end
    return _add_tri!(C, tmp, up, herm, n)
end
function _syr2k_acc!(up::Bool, tr::Bool, herm::Bool, α, A, B, C, k::Int, scr = _l3_tmp(eltype(C)))
    n = size(C, 1)
    n <= _SYRK_DBASE && return _syr2k_base!(up, tr, herm, α, A, B, C, n, scr)
    h = _trsplit(n); tX = herm ? 'C' : 'T'; α2 = herm ? conj(α) : α
    C11 = view(C, 1:h, 1:h); C22 = view(C, (h + 1):n, (h + 1):n)
    A1 = tr ? view(A, :, 1:h) : view(A, 1:h, :); A2 = tr ? view(A, :, (h + 1):n) : view(A, (h + 1):n, :)
    B1 = tr ? view(B, :, 1:h) : view(B, 1:h, :); B2 = tr ? view(B, :, (h + 1):n) : view(B, (h + 1):n, :)
    _syr2k_acc!(up, tr, herm, α, A1, B1, C11, k, scr); _syr2k_acc!(up, tr, herm, α, A2, B2, C22, k, scr)
    if up
        Coff = view(C, 1:h, (h + 1):n)            # C12 += α·op(A1)op(B2)ᴴ + α2·op(B1)op(A2)ᴴ
        if !tr
            gemm!(Coff, A1, B2; alpha = α, beta = true, transB = tX); gemm!(Coff, B1, A2; alpha = α2, beta = true, transB = tX)
        else
            gemm!(Coff, A1, B2; alpha = α, beta = true, transA = tX); gemm!(Coff, B1, A2; alpha = α2, beta = true, transA = tX)
        end
    else
        Coff = view(C, (h + 1):n, 1:h)            # C21 += α·op(A2)op(B1)ᴴ + α2·op(B2)op(A1)ᴴ
        if !tr
            gemm!(Coff, A2, B1; alpha = α, beta = true, transB = tX); gemm!(Coff, B2, A1; alpha = α2, beta = true, transB = tX)
        else
            gemm!(Coff, A2, B1; alpha = α, beta = true, transA = tX); gemm!(Coff, B2, A1; alpha = α2, beta = true, transA = tX)
        end
    end
    return C
end
function _syr2k_dims(C, A, B, trans)
    n = size(C, 1); (size(C, 2) == n && size(A) == size(B)) || throw(DimensionMismatch("syr2k!: shapes"))
    k = trans == 'N' ? size(A, 2) : size(A, 1)
    (trans == 'N' ? size(A, 1) : size(A, 2)) == n || throw(DimensionMismatch("syr2k!: op(A) rows ≠ n"))
    (n, k)
end
function syr2k!(C::AbstractMatrix, A::AbstractMatrix, Bm::AbstractMatrix; uplo::Char = 'U',
        trans::Char = 'N', alpha::Number = true, beta::Number = false)
    n, k = _syr2k_dims(C, A, Bm, trans); up = uplo == 'U'; _syrk_scaleC!(C, up, beta)
    if eltype(C) <: BlasReal && n > _GEMM_UNPACK_MAX && k > 0
        _syr2k_packed!(up, trans != 'N', convert(eltype(C), alpha), A, Bm, C, k)
    else
        _syr2k_acc!(up, trans != 'N', false, alpha, A, Bm, C, k)
    end
    C
end
function her2k!(C::AbstractMatrix, A::AbstractMatrix, Bm::AbstractMatrix; uplo::Char = 'U',
        trans::Char = 'N', alpha::Number = true, beta::Real = false)
    n, k = _syr2k_dims(C, A, Bm, trans); up = uplo == 'U'
    _syrk_scaleC!(C, up, beta); _syr2k_acc!(up, trans != 'N', true, alpha, A, Bm, C, k); C
end
