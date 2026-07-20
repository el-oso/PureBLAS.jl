# Mode 1 C/Fortran-ABI boundary for LAPACK — `@ccallable` entry points forwarded via libblastrampoline.
#
# Same ABI as cabi.jl (GEMM): char args are `Ptr{UInt8}` FIRST (deref + byte-uppercase via `_cabi_char`
# from cabi.jl), then by-ref `Ptr` scalars/arrays, then TRAILING hidden Fortran string-length `Clong`s —
# one per char arg. Column-major, ILP64 (`Int64`). Matrix bridge = the GEMM `view(unsafe_wrap(...))` idiom.
#
# LAPACK-specific ABI (vs BLAS): an `info` OUTPUT arg (`Ptr{Int64}`, last positional before the hidden
# lengths) — `unsafe_store!(info, 0)` on success or the real LAPACK info. Routines with a workspace query
# (`work`, `lwork`): when `lwork == -1` return the optimal work size in `work[1]` and do nothing else.
# PureBLAS manages its OWN workspace, so the wrapper IGNORES the provided `work` buffer but HONORS the
# query protocol (returns 1 and info=0).
#
# Element types: `d` (Float64) for all four factorizations; `s` (Float32) ONLY for potrf — PureBLAS's
# potrf! has a generic T<:Real path (Float32-capable), but getrf!/geqrf!/gesvd! are Float64-only kernels,
# so sgetrf/sgeqrf/sgesvd are intentionally NOT provided (no Float32 factorization path to forward to).

# ── potrf: Cholesky (real SPD; complex Hermitian PD) ─────────────────────────────────────────────────
# `{s,d,c,z}potrf_64_(uplo, n, A, lda, info, len_uplo)` — 1 char. potrf! throws PosDefException on a
# non-(S/H)PD input; CATCH it at the boundary and report the failing minor via info>0 (LAPACK convention)
# rather than letting a Julia exception unwind through the C-ABI. (Note: the Float64 lower faer fast path
# reports its non-SPD as PosDefException(1) — info=1, not the exact minor; correctness-wise info>0 is all
# LAPACK asks.) potrf! is generic over the element type, so the complex (Hermitian) path is the same code.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "potrf_64_"))(
            uplo::Ptr{UInt8}, n::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64},
            info::Ptr{Int64}, len_uplo::Clong)::Cvoid
        ul = _cabi_char(uplo)
        N = Int(unsafe_load(n)); ld = Int(unsafe_load(lda))
        Av = PtrMatrix(A, N, N, ld)
        try
            potrf!(Av; uplo = ul)
            unsafe_store!(info, Int64(0))
        catch e
            e isa PosDefException || rethrow()
            unsafe_store!(info, Int64(e.info))
        end
        return
    end
end

# ── getrf: LU with partial pivoting (real + complex) ─────────────────────────────────────────────────
# `{d,c,z}getrf_64_(m, n, A, lda, ipiv, info)` — 0 chars, no hidden lengths. ipiv is `Ptr{Int64}` OUTPUT
# (length min(m,n)); bridge as a Vector{Int64} (== Vector{Int} on 64-bit) — LAPACK 1-based pivot rows.
# getrf!(A, ipiv) has both a Float64 and a `T<:BlasComplex` method, so the complex path is the same wrapper.
# (No `s` — the Float32 getrf! kernel isn't implemented yet; sgetrf stays on the fallback backend.)
for (p, T) in (("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "getrf_64_"))(m::Ptr{Int64}, n::Ptr{Int64},
            A::Ptr{$T}, lda::Ptr{Int64}, ipiv::Ptr{Int64}, info::Ptr{Int64})::Cvoid
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); ld = Int(unsafe_load(lda))
        Av = PtrMatrix(A, M, N, ld)
        ip = PtrVector(ipiv, min(M, N))
        _, _, inf = getrf!(Av, ip)
        unsafe_store!(info, Int64(inf))
        return
    end
end

# ── geqrf: QR (Householder, no pivoting) ────────────────────────────────────────────────────────────
# `dgeqrf_64_(m, n, A, lda, tau, work, lwork, info)` — 0 chars. Honors the lwork==-1 query.
# PureBLAS's geqrf! stores the SAME reflector vectors v as reference LAPACK, but τ in the faer convention
# (τ_stored = 1/τ_LAPACK). We CONVERT τ back to the LAPACK convention at the C-ABI boundary so this symbol
# is a true drop-in (faer never crosses the ABI): a caller feeding this τ to reference/OpenBLAS orgqr/ormqr
# now gets a correct Q. (Complex zgeqrf! already returns LAPACK-convention τ — no conversion there.)
Base.@ccallable function dgeqrf_64_(m::Ptr{Int64}, n::Ptr{Int64}, A::Ptr{Float64}, lda::Ptr{Int64},
        tau::Ptr{Float64}, work::Ptr{Float64}, lwork::Ptr{Int64}, info::Ptr{Int64})::Cvoid
    if unsafe_load(lwork) == Int64(-1)                 # workspace query: report size 1, do nothing else
        unsafe_store!(work, 1.0); unsafe_store!(info, Int64(0)); return
    end
    M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); ld = Int(unsafe_load(lda)); k = min(M, N)
    Av = PtrMatrix(A, M, N, ld)
    tv = PtrVector(tau, k)
    geqrf!(Av, tv)
    @inbounds for i in 1:k                             # faer τ_stored=1/τ_LAPACK → LAPACK (trivial: Inf→0)
        t = tv[i]; tv[i] = (isfinite(t) && t != 0.0) ? 1.0 / t : 0.0
    end
    unsafe_store!(info, Int64(0))
    return
end

# cgeqrf/zgeqrf: native complex geqrf! (BlasComplex). Same faer-τ→LAPACK convert as dgeqrf (v is standard).
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "geqrf_64_"))(m::Ptr{Int64}, n::Ptr{Int64}, A::Ptr{$T},
            lda::Ptr{Int64}, tau::Ptr{$T}, work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64})::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, $T(1)); unsafe_store!(info, Int64(0)); return
        end
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); ld = Int(unsafe_load(lda)); k = min(M, N)
        geqrf!(PtrMatrix(A, M, N, ld), PtrVector(tau, k))
        tv = PtrVector(tau, k)
        @inbounds for i in 1:k
            t = tv[i]; tv[i] = (isfinite(t) && t != zero($T)) ? one($T) / t : zero($T)
        end
        unsafe_store!(info, Int64(0)); return
    end
end
# sgeqrf: no native Float32 geqrf! kernel — mixed-precision (promote→F64 geqrf!→demote), like sgeqrt_64_.
Base.@ccallable function sgeqrf_64_(m::Ptr{Int64}, n::Ptr{Int64}, A::Ptr{Float32}, lda::Ptr{Int64},
        tau::Ptr{Float32}, work::Ptr{Float32}, lwork::Ptr{Int64}, info::Ptr{Int64})::Cvoid
    if unsafe_load(lwork) == Int64(-1)
        unsafe_store!(work, 1.0f0); unsafe_store!(info, Int64(0)); return
    end
    M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); ld = Int(unsafe_load(lda)); k = min(M, N)
    Am = PtrMatrix(A, M, N, ld)
    Af = Matrix{Float64}(undef, M, N); _f32_to_f64!(Af, Am, M, N)
    tf = Vector{Float64}(undef, k)
    geqrf!(Af, tf)
    _f64_to_f32!(Am, Af, M, N)
    tv = PtrVector(tau, k)
    @inbounds for i in 1:k
        t = tf[i]; tv[i] = Float32((isfinite(t) && t != 0.0) ? 1.0 / t : 0.0)
    end
    unsafe_store!(info, Int64(0)); return
end

# ── geqrt / gemqrt: compact-WY QR — routes LinearAlgebra.qr() (QRCompactWY) to PureBLAS ────────────────
# Julia's qr(A) calls geqrt!; Q ops (Matrix(Q), Q*B, Q'*B, A*Q, qr(A)\b's Qᵀb) all go through gemqrt!.
# We compose PureBLAS's geqrf! (V + faer τ) + τ→LAPACK inversion + wy_t! (dlarft) for geqrt, and wy_apply!
# / a right-side mirror (dlarfb) for gemqrt. The T factor is built on the CALLER's nb grid in LAPACK-exact
# layout (T[1:ib, i:i+ib-1] per block) — det(Q::QRCompactWYQ) reads T's block-diagonal. Real Float64 only:
# no Float32 QR kernel, and complex needs a VᴴV (not VᵀV) T-build — a separate wrapper.

# Build the explicit unit-lower-trapezoidal reflector panel for block [i, i+ib) into dest[1:mp,1:ib].
@inline function _qr_vpanel!(dest::AbstractMatrix{T}, Vsrc::AbstractMatrix{T}, i::Int, ib::Int, mp::Int) where {T}
    @inbounds for c in 1:ib, r in 1:mp
        dest[r, c] = r == c ? one(T) : (r > c ? Vsrc[i + r - 1, i + c - 1] : zero(T))
    end
end
# Right-side block apply: C := C·Q (trans 'N') or C·Qᵀ ('T'), Q = I − V·Tblk·Vᵀ (V = mp×ib explicit unit).
@inline function _qr_apply_right!(trans::Char, C::AbstractMatrix{Float64}, Vp::AbstractMatrix{Float64},
        Tblk::AbstractMatrix{Float64})
    nr = size(C, 1); ib = size(Vp, 2)
    (nr == 0 || ib == 0) && return C
    W = Matrix{Float64}(undef, nr, ib)
    gemm!(W, C, Vp; alpha = true, beta = false)                       # W = C·V
    trmm!(W, Tblk; side = 'R', uplo = 'U', transA = trans)            # W := W·(T or Tᵀ)
    gemm!(C, W, Vp; transB = 'T', alpha = -1.0, beta = 1.0)           # C -= W·Vᵀ
    return C
end

Base.@ccallable function dgeqrt_64_(m::Ptr{Int64}, n::Ptr{Int64}, nb::Ptr{Int64}, A::Ptr{Float64},
        lda::Ptr{Int64}, T::Ptr{Float64}, ldt::Ptr{Int64}, work::Ptr{Float64}, info::Ptr{Int64})::Cvoid
    M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); NB = Int(unsafe_load(nb)); k = min(M, N)
    Av = PtrMatrix(A, M, N, Int(unsafe_load(lda)))
    Tm = PtrMatrix(T, NB, k, Int(unsafe_load(ldt)))
    τ = Vector{Float64}(undef, k)
    GC.@preserve τ begin
        geqrf!(Av, PtrVector(pointer(τ), k))
        @inbounds for i in 1:k; t = τ[i]; τ[i] = (isfinite(t) && t != 0.0) ? 1.0 / t : 0.0; end
        ws = WYApplyWorkspace{Float64}(M, NB, N)
        for i in 1:NB:k
            ib = min(NB, k - i + 1); mp = M - i + 1
            Vp = view(ws.V, 1:mp, 1:ib)
            _qr_vpanel!(Vp, Av, i, ib, mp)
            wy_t!(view(Tm, 1:ib, i:(i + ib - 1)), Vp, view(τ, i:(i + ib - 1)), view(ws.G, 1:ib, 1:ib))
        end
    end
    unsafe_store!(info, Int64(0)); return
end

Base.@ccallable function dgemqrt_64_(side::Ptr{UInt8}, trans::Ptr{UInt8}, m::Ptr{Int64}, n::Ptr{Int64},
        k::Ptr{Int64}, nb::Ptr{Int64}, V::Ptr{Float64}, ldv::Ptr{Int64}, T::Ptr{Float64}, ldt::Ptr{Int64},
        C::Ptr{Float64}, ldc::Ptr{Int64}, work::Ptr{Float64}, info::Ptr{Int64},
        len_s::Clong, len_t::Clong)::Cvoid
    sd = _cabi_char(side); tr = _cabi_char(trans)
    M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); K = Int(unsafe_load(k)); NB = Int(unsafe_load(nb))
    vrows = sd == 'L' ? M : N
    Vm = PtrMatrix(V, vrows, K, Int(unsafe_load(ldv)))
    Tm = PtrMatrix(T, NB, K, Int(unsafe_load(ldt)))
    Cm = PtrMatrix(C, M, N, Int(unsafe_load(ldc)))
    # Block sweep order (LAPACK dgemqrt): L+T / R+N forward; L+N / R+T reverse.
    forward = (sd == 'L' && tr != 'N') || (sd == 'R' && tr == 'N')
    starts = collect(1:NB:K); forward || reverse!(starts)
    ws = WYApplyWorkspace{Float64}(vrows, NB, max(M, N))
    for i in starts
        ib = min(NB, K - i + 1); mp = vrows - i + 1
        Vp = view(ws.V, 1:mp, 1:ib)
        _qr_vpanel!(Vp, Vm, i, ib, mp)
        Tblk = view(Tm, 1:ib, i:(i + ib - 1))
        if sd == 'L'
            wy_apply!(tr, view(Cm, i:M, 1:N), Vp, Tblk, ws)      # C[i:M, :]
        else
            _qr_apply_right!(tr, view(Cm, 1:M, i:N), Vp, Tblk)   # C[:, i:N]
        end
    end
    unsafe_store!(info, Int64(0)); return
end

# ── gesvd: SVD ──────────────────────────────────────────────────────────────────────────────────────
# `dgesvd_64_(jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, info, len_jobu, len_jobvt)`.
# The in-place gesvd!(A,U,S,Vᵀ) writes the factors DIRECTLY into the caller's buffers (0-alloc; A destroyed
# as LAPACK gesvd does). The SVD forms BOTH U and Vᵀ regardless of job, so 'N'/'O' — which supply no output
# buffer, or overwrite A — get an owned economy scratch (SVDWorkspace.cabi_U/cabi_Vt), copied into A for 'O'.
# Every operand is a PtrMatrix ⇒ one concrete gesvd! specialization ⇒ juliac --trim=safe (TrimCheck).
#
# jobu/jobvt COVERAGE (LAPACK dgesvd semantics):
#   'N' — vectors not computed (skip).
#   'S' — economy: U is m×min into the U buffer; Vᵀ is min×n into the VT buffer.
#   'A' — full square: U is m×m; Vᵀ is n×n. When the full factor exceeds economy (jobu='A' & m>n, or
#         jobvt='A' & n>m) gesvd!'s full_u/full_v forms the extra orthonormal complement.
#   'O' — overwrite A: jobu='O' writes U's economy columns into A; jobvt='O' writes Vᵀ's economy rows into
#         A. (LAPACK forbids jobu=jobvt='O' → info=-1.) A is destroyed by the factorization first, then the
#         fresh vectors are written into it — no aliasing with the returned Matrices.
# Shared SVD C-ABI core: given the already-deref'd job chars + raw pointers/dims, compute the SVD via
# PureBLAS's gesvd! driver (which picks bdsqr vs bdsdc divide-and-conquer by size) and write into the
# caller's buffers. Used by BOTH dgesvd_64_ (jobu,jobvt) and dgesdd_64_ (jobz → same (ju,jvt)).
@inline function _svd_cabi!(ju::Char, jvt::Char, M::Int, N::Int, A::Ptr{Float64}, ld::Int,
        S::Ptr{Float64}, U::Ptr{Float64}, ldu::Int, VT::Ptr{Float64}, ldvt::Int, info::Ptr{Int64})
    mn = min(M, N)
    u_ok = ju == 'N' || ju == 'S' || ju == 'A' || ju == 'O'
    v_ok = jvt == 'N' || jvt == 'S' || jvt == 'A' || jvt == 'O'
    if !(u_ok && v_ok) || (ju == 'O' && jvt == 'O')    # unknown job, or the illegal jobu=jobvt='O'
        unsafe_store!(info, Int64(-1)); return
    end
    Av = PtrMatrix(A, M, N, ld)
    if ju == 'N' && jvt == 'N'                         # values only → 0-alloc into caller's S
        gesvd_vals!(Av, PtrVector(S, mn))
        unsafe_store!(info, Int64(0)); return
    end
    # 0-alloc: gesvd! writes U/S/Vᵀ straight into caller buffers. The SVD forms BOTH factors regardless,
    # so 'N'/'O' (no caller buffer, or overwrite-A) get an owned economy scratch (ws.cabi_U/cabi_Vt); 'O'
    # is then copied into A. Everything stays a PtrMatrix ⇒ one concrete gesvd! specialization ⇒ trim-safe.
    full_u = ju == 'A' && M > N                         # extra m−n complement columns needed
    full_v = jvt == 'A' && N > M                        # extra n−m complement rows needed
    ncu = ju == 'A' ? M : mn
    ncv = jvt == 'A' ? N : mn
    ws = _svdws()
    uscr = ju == 'N' || ju == 'O'                      # need scratch U (economy M×mn)
    vscr = jvt == 'N' || jvt == 'O'                    # need scratch Vᵀ (economy mn×N)
    uscr && (ws.cabi_U = _gm(ws.cabi_U, M, mn))
    vscr && (ws.cabi_Vt = _gm(ws.cabi_Vt, mn, N))
    GC.@preserve ws begin
        Ut  = uscr ? PtrMatrix(pointer(ws.cabi_U), M, mn, size(ws.cabi_U, 1)) :
                     PtrMatrix(U, M, ncu, ldu)
        Vtt = vscr ? PtrMatrix(pointer(ws.cabi_Vt), mn, N, size(ws.cabi_Vt, 1)) :
                     PtrMatrix(VT, ncv, N, ldvt)
        gesvd!(Av, Ut, PtrVector(S, mn), Vtt; full_u = full_u, full_v = full_v)
        if ju == 'O'                                   # economy U columns overwrite A
            @inbounds for j in 1:mn, i in 1:M; Av[i, j] = Ut[i, j]; end
        end
        if jvt == 'O'                                  # economy Vᵀ rows overwrite A
            @inbounds for j in 1:N, i in 1:mn; Av[i, j] = Vtt[i, j]; end
        end
    end
    unsafe_store!(info, Int64(0))
    return
end

Base.@ccallable function dgesvd_64_(jobu::Ptr{UInt8}, jobvt::Ptr{UInt8}, m::Ptr{Int64}, n::Ptr{Int64},
        A::Ptr{Float64}, lda::Ptr{Int64}, S::Ptr{Float64}, U::Ptr{Float64}, ldu::Ptr{Int64},
        VT::Ptr{Float64}, ldvt::Ptr{Int64}, work::Ptr{Float64}, lwork::Ptr{Int64}, info::Ptr{Int64},
        len_jobu::Clong, len_jobvt::Clong)::Cvoid
    if unsafe_load(lwork) == Int64(-1)                 # workspace query
        unsafe_store!(work, 1.0); unsafe_store!(info, Int64(0)); return
    end
    _svd_cabi!(_cabi_char(jobu), _cabi_char(jobvt), Int(unsafe_load(m)), Int(unsafe_load(n)),
        A, Int(unsafe_load(lda)), S, U, Int(unsafe_load(ldu)), VT, Int(unsafe_load(ldvt)), info)
    return
end

# ── gesdd: divide-and-conquer SVD ─────────────────────────────────────────────────────────────────────
# `dgesdd_64_(jobz, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, iwork, info, len_jobz)` — 1 char + an
# extra Int64 `iwork` workspace arg (ignored; PureBLAS owns its scratch). Julia's `svd()`/`svdvals` route
# through gesdd (NOT gesvd), so this is what makes them use PureBLAS. gesdd's single `jobz` maps to gesvd's
# (jobu,jobvt) = (jobz,jobz) for N/S/A, dispatched to the same gesvd! driver (which does D&C above
# _SVD_DC_CROSS). jobz='O' (overwrite A) is NOT supported (info=-1) — Julia only ever uses N/S/A.
Base.@ccallable function dgesdd_64_(jobz::Ptr{UInt8}, m::Ptr{Int64}, n::Ptr{Int64}, A::Ptr{Float64},
        lda::Ptr{Int64}, S::Ptr{Float64}, U::Ptr{Float64}, ldu::Ptr{Int64}, VT::Ptr{Float64},
        ldvt::Ptr{Int64}, work::Ptr{Float64}, lwork::Ptr{Int64}, iwork::Ptr{Int64}, info::Ptr{Int64},
        len_jobz::Clong)::Cvoid
    if unsafe_load(lwork) == Int64(-1)                 # workspace query
        unsafe_store!(work, 1.0); unsafe_store!(info, Int64(0)); return
    end
    jz = _cabi_char(jobz)
    if jz == 'O'                                       # gesdd overwrite-A mode not supported
        unsafe_store!(info, Int64(-1)); return
    end
    _svd_cabi!(jz, jz, Int(unsafe_load(m)), Int(unsafe_load(n)),
        A, Int(unsafe_load(lda)), S, U, Int(unsafe_load(ldu)), VT, Int(unsafe_load(ldvt)), info)
    return
end

# ── COMPLEX gesvd/gesdd (ComplexF64 native + ComplexF32 mixed-precision) ───────────────────────────────
# Same job semantics as the real _svd_cabi! (N/S/A/O), but S/rwork are REAL ($Tr) while A/U/VT/work are
# COMPLEX; the LAPACK complex gesvd/gesdd ABI inserts an rwork block (real scratch) the real drivers lack —
# ignored here (PureBLAS owns its scratch). gesvd! forms BOTH factors regardless, so 'N'/'O' get an owned
# economy scratch (ws.cabi_U/cabi_Vt), 'O' then copied into A. One concrete gesvd! specialization ⇒ trim-safe.
@inline function _zsvd_cabi!(ju::Char, jvt::Char, M::Int, N::Int, A::Ptr{ComplexF64}, ld::Int,
        S::Ptr{Float64}, U::Ptr{ComplexF64}, ldu::Int, VT::Ptr{ComplexF64}, ldvt::Int, info::Ptr{Int64})
    mn = min(M, N)
    u_ok = ju == 'N' || ju == 'S' || ju == 'A' || ju == 'O'
    v_ok = jvt == 'N' || jvt == 'S' || jvt == 'A' || jvt == 'O'
    if !(u_ok && v_ok) || (ju == 'O' && jvt == 'O')
        unsafe_store!(info, Int64(-1)); return
    end
    Av = PtrMatrix(A, M, N, ld)
    if ju == 'N' && jvt == 'N'                         # values only → 0-alloc into caller's S
        gesvd_vals!(Av, PtrVector(S, mn))
        unsafe_store!(info, Int64(0)); return
    end
    full_u = ju == 'A' && M > N
    full_v = jvt == 'A' && N > M
    ncu = ju == 'A' ? M : mn
    ncv = jvt == 'A' ? N : mn
    ws = _svdws(ComplexF64)
    uscr = ju == 'N' || ju == 'O'
    vscr = jvt == 'N' || jvt == 'O'
    uscr && (ws.cabi_U = _gm(ws.cabi_U, M, mn))
    vscr && (ws.cabi_Vt = _gm(ws.cabi_Vt, mn, N))
    GC.@preserve ws begin
        Ut  = uscr ? PtrMatrix(pointer(ws.cabi_U), M, mn, size(ws.cabi_U, 1)) :
                     PtrMatrix(U, M, ncu, ldu)
        Vtt = vscr ? PtrMatrix(pointer(ws.cabi_Vt), mn, N, size(ws.cabi_Vt, 1)) :
                     PtrMatrix(VT, ncv, N, ldvt)
        gesvd!(Av, Ut, PtrVector(S, mn), Vtt; full_u = full_u, full_v = full_v)
        if ju == 'O'
            @inbounds for j in 1:mn, i in 1:M; Av[i, j] = Ut[i, j]; end
        end
        if jvt == 'O'
            @inbounds for j in 1:N, i in 1:mn; Av[i, j] = Vtt[i, j]; end
        end
    end
    unsafe_store!(info, Int64(0))
    return
end

# zgesvd_(jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info, len_jobu, len_jobvt).
Base.@ccallable function zgesvd_64_(jobu::Ptr{UInt8}, jobvt::Ptr{UInt8}, m::Ptr{Int64}, n::Ptr{Int64},
        A::Ptr{ComplexF64}, lda::Ptr{Int64}, S::Ptr{Float64}, U::Ptr{ComplexF64}, ldu::Ptr{Int64},
        VT::Ptr{ComplexF64}, ldvt::Ptr{Int64}, work::Ptr{ComplexF64}, lwork::Ptr{Int64},
        rwork::Ptr{Float64}, info::Ptr{Int64}, len_jobu::Clong, len_jobvt::Clong)::Cvoid
    if unsafe_load(lwork) == Int64(-1)
        unsafe_store!(work, ComplexF64(1)); unsafe_store!(info, Int64(0)); return
    end
    _zsvd_cabi!(_cabi_char(jobu), _cabi_char(jobvt), Int(unsafe_load(m)), Int(unsafe_load(n)),
        A, Int(unsafe_load(lda)), S, U, Int(unsafe_load(ldu)), VT, Int(unsafe_load(ldvt)), info)
    return
end

# zgesdd_(jobz, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, iwork, info, len_jobz) — Julia's
# svd()/svdvals(::Matrix{ComplexF64}) route here (D&C). rwork/iwork ignored. jobz='O' unsupported (info=-1).
Base.@ccallable function zgesdd_64_(jobz::Ptr{UInt8}, m::Ptr{Int64}, n::Ptr{Int64}, A::Ptr{ComplexF64},
        lda::Ptr{Int64}, S::Ptr{Float64}, U::Ptr{ComplexF64}, ldu::Ptr{Int64}, VT::Ptr{ComplexF64},
        ldvt::Ptr{Int64}, work::Ptr{ComplexF64}, lwork::Ptr{Int64}, rwork::Ptr{Float64},
        iwork::Ptr{Int64}, info::Ptr{Int64}, len_jobz::Clong)::Cvoid
    if unsafe_load(lwork) == Int64(-1)
        unsafe_store!(work, ComplexF64(1)); unsafe_store!(info, Int64(0)); return
    end
    jz = _cabi_char(jobz)
    if jz == 'O'
        unsafe_store!(info, Int64(-1)); return
    end
    _zsvd_cabi!(jz, jz, Int(unsafe_load(m)), Int(unsafe_load(n)),
        A, Int(unsafe_load(lda)), S, U, Int(unsafe_load(ldu)), VT, Int(unsafe_load(ldvt)), info)
    return
end

# ComplexF32 SVD via mixed precision: promote A→ComplexF64, run the ComplexF64 driver into F64 scratch,
# demote outputs (S→Float32, U/VT→ComplexF32). Mirrors _svd_cabi_f32! (the real F32 path).
@inline function _csvd_cabi_f32!(ju::Char, jvt::Char, M::Int, N::Int, A::Ptr{ComplexF32}, ld::Int,
        S::Ptr{Float32}, U::Ptr{ComplexF32}, ldu::Int, VT::Ptr{ComplexF32}, ldvt::Int, info::Ptr{Int64})
    mn = min(M, N)
    Af = Matrix{ComplexF64}(undef, M, N); Am = PtrMatrix(A, M, N, ld)
    @inbounds for j in 1:N, i in 1:M; Af[i, j] = ComplexF64(Am[i, j]); end
    Sf = Vector{Float64}(undef, mn)
    needU = ju != 'N'; needV = jvt != 'N'
    ncu = ju == 'A' ? M : mn; ncv = jvt == 'A' ? N : mn
    Uf = Matrix{ComplexF64}(undef, M, needU ? ncu : 1)
    Vf = Matrix{ComplexF64}(undef, needV ? ncv : 1, N)
    GC.@preserve Af Sf Uf Vf begin
        _zsvd_cabi!(ju, jvt, M, N, pointer(Af), M, pointer(Sf), pointer(Uf), M,
            pointer(Vf), needV ? ncv : 1, info)
        unsafe_load(info) == 0 || return
        Sm = PtrVector(S, mn); @inbounds for i in 1:mn; Sm[i] = Float32(Sf[i]); end
        if needU
            Um = PtrMatrix(U, M, ncu, ldu)
            @inbounds for j in 1:ncu, i in 1:M; Um[i, j] = ComplexF32(Uf[i, j]); end
        end
        if needV
            Vm = PtrMatrix(VT, ncv, N, ldvt)
            @inbounds for j in 1:N, i in 1:ncv; Vm[i, j] = ComplexF32(Vf[i, j]); end
        end
    end
    return
end

Base.@ccallable function cgesvd_64_(jobu::Ptr{UInt8}, jobvt::Ptr{UInt8}, m::Ptr{Int64}, n::Ptr{Int64},
        A::Ptr{ComplexF32}, lda::Ptr{Int64}, S::Ptr{Float32}, U::Ptr{ComplexF32}, ldu::Ptr{Int64},
        VT::Ptr{ComplexF32}, ldvt::Ptr{Int64}, work::Ptr{ComplexF32}, lwork::Ptr{Int64},
        rwork::Ptr{Float32}, info::Ptr{Int64}, len_jobu::Clong, len_jobvt::Clong)::Cvoid
    if unsafe_load(lwork) == Int64(-1)
        unsafe_store!(work, ComplexF32(1)); unsafe_store!(info, Int64(0)); return
    end
    _csvd_cabi_f32!(_cabi_char(jobu), _cabi_char(jobvt), Int(unsafe_load(m)), Int(unsafe_load(n)),
        A, Int(unsafe_load(lda)), S, U, Int(unsafe_load(ldu)), VT, Int(unsafe_load(ldvt)), info)
    return
end

Base.@ccallable function cgesdd_64_(jobz::Ptr{UInt8}, m::Ptr{Int64}, n::Ptr{Int64}, A::Ptr{ComplexF32},
        lda::Ptr{Int64}, S::Ptr{Float32}, U::Ptr{ComplexF32}, ldu::Ptr{Int64}, VT::Ptr{ComplexF32},
        ldvt::Ptr{Int64}, work::Ptr{ComplexF32}, lwork::Ptr{Int64}, rwork::Ptr{Float32},
        iwork::Ptr{Int64}, info::Ptr{Int64}, len_jobz::Clong)::Cvoid
    if unsafe_load(lwork) == Int64(-1)
        unsafe_store!(work, ComplexF32(1)); unsafe_store!(info, Int64(0)); return
    end
    jz = _cabi_char(jobz)
    if jz == 'O'
        unsafe_store!(info, Int64(-1)); return
    end
    _csvd_cabi_f32!(jz, jz, Int(unsafe_load(m)), Int(unsafe_load(n)),
        A, Int(unsafe_load(lda)), S, U, Int(unsafe_load(ldu)), VT, Int(unsafe_load(ldvt)), info)
    return
end

# ── syev / syevd / syevr: symmetric eigensolver (real Float64, M-E1) ──────────────────────────────────
# All three LAPACK drivers share ONE engine (_syev! = sytrd → steqr → ormtr): the driver contract is
# identical (eigenpairs to O(eps·‖A‖), orthonormal vectors, ascending). Julia's DEFAULT eigen(Symmetric)/
# eigvals(Symmetric) routes through dsyevr_ (RobustRepresentations); DivideAndConquer→dsyevd_, QRIteration→
# dsyev_. We compute the FULL spectrum and (for jobz='V') vectors from a COPY of the caller's A (cheap O(n²)
# vs the O(n³) solve; sidesteps PtrMatrix-view aliasing), then write back per each driver's ABI.
# Dispatch the right native engine by element type (const-folds per instantiation — trim-safe).
_ev_engine!(jobz::Char, uplo::Char, A::AbstractMatrix{<:Real})    = _syev!(jobz, uplo, A)
_ev_engine!(jobz::Char, uplo::Char, A::AbstractMatrix{<:Complex}) = _heev!(jobz, uplo, A)

# Copy the caller's A into a fresh Matrix{T} (unused triangle is garbage but unread) and solve. For the
# real drivers T is the ABI type; for the complex drivers _heev_solve promotes to ComplexF64 (native for
# z, MIXED precision for c — mirrors sgetrf/sgesvd). Returns (w::Vector, Z::Matrix, info::Int).
@inline function _syev_compute(jobz::Char, uplo::Char, N::Int, A::Ptr{T}, ld::Int) where {T}
    Aw = Matrix{T}(undef, N, N)
    Am = PtrMatrix(A, N, N, ld)
    @inbounds for j in 1:N, i in 1:N; Aw[i, j] = Am[i, j]; end
    return _ev_engine!(jobz, uplo, Aw)
end
@inline function _heev_solve(jobz::Char, uplo::Char, N::Int, A::Ptr{Tc}, ld::Int) where {Tc<:Complex}
    Aw = Matrix{ComplexF64}(undef, N, N)             # compute in ComplexF64 (z native, c mixed-precision)
    Am = PtrMatrix(A, N, N, ld)
    @inbounds for j in 1:N, i in 1:N; Aw[i, j] = ComplexF64(Am[i, j]); end
    return _heev!(jobz, uplo, Aw)                    # (w::Vector{Float64}, Z::Matrix{ComplexF64}, info)
end

# ── REAL syev/syevd/syevr (Float32 native + Float64) ──────────────────────────────────────────────────
# All three drivers share the _syev! engine (sytrd → stedc/sterf → ormtr): identical contract (eigenpairs
# to O(eps·‖A‖), orthonormal vectors, ascending). Julia's DEFAULT eigen(Symmetric)/eigvals(Symmetric) →
# dsyevr_ (RobustRepresentations); DivideAndConquer→dsyevd_, QRIteration→dsyev_. Float32 is NATIVE (the
# genericized kernels), not mixed precision. Vectors OVERWRITE A (syev/syevd) or go to the z buffer (syevr).
for (p, T) in (("s", Float32), ("d", Float64))
    @eval begin
        # $(p)syev_(jobz, uplo, n, A, lda, w, work, lwork, info, len_jobz, len_uplo)
        Base.@ccallable function $(Symbol(p, "syev_64_"))(jobz::Ptr{UInt8}, uplo::Ptr{UInt8}, n::Ptr{Int64},
                A::Ptr{$T}, lda::Ptr{Int64}, w::Ptr{$T}, work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64},
                len_jobz::Clong, len_uplo::Clong)::Cvoid
            if unsafe_load(lwork) == Int64(-1)                 # workspace query
                unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
            end
            jz = _cabi_char(jobz); ul = _cabi_char(uplo)
            N = Int(unsafe_load(n)); ld = Int(unsafe_load(lda))
            bad = (jz != 'N' && jz != 'V') ? -1 : (ul != 'U' && ul != 'L') ? -2 : (N < 0) ? -3 : 0
            !iszero(bad) && (unsafe_store!(info, Int64(bad)); return)
            wv, Z, inf = _syev_compute(jz, ul, N, A, ld)
            wm = PtrVector(w, N)
            @inbounds for i in eachindex(wm); wm[i] = wv[i]; end
            if jz == 'V'
                Am = PtrMatrix(A, N, N, ld)
                @inbounds for j in 1:N, i in 1:N; Am[i, j] = Z[i, j]; end
            end
            unsafe_store!(info, Int64(inf)); return
        end

        # $(p)syevd_(jobz, uplo, n, A, lda, w, work, lwork, iwork, liwork, info, len_jobz, len_uplo)
        Base.@ccallable function $(Symbol(p, "syevd_64_"))(jobz::Ptr{UInt8}, uplo::Ptr{UInt8}, n::Ptr{Int64},
                A::Ptr{$T}, lda::Ptr{Int64}, w::Ptr{$T}, work::Ptr{$T}, lwork::Ptr{Int64}, iwork::Ptr{Int64},
                liwork::Ptr{Int64}, info::Ptr{Int64}, len_jobz::Clong, len_uplo::Clong)::Cvoid
            if unsafe_load(lwork) == Int64(-1)                 # workspace query (report work + iwork sizes)
                unsafe_store!(work, one($T)); unsafe_store!(iwork, Int64(1)); unsafe_store!(info, Int64(0)); return
            end
            jz = _cabi_char(jobz); ul = _cabi_char(uplo)
            N = Int(unsafe_load(n)); ld = Int(unsafe_load(lda))
            bad = (jz != 'N' && jz != 'V') ? -1 : (ul != 'U' && ul != 'L') ? -2 : (N < 0) ? -3 : 0
            !iszero(bad) && (unsafe_store!(info, Int64(bad)); return)
            wv, Z, inf = _syev_compute(jz, ul, N, A, ld)
            wm = PtrVector(w, N)
            @inbounds for i in eachindex(wm); wm[i] = wv[i]; end
            if jz == 'V'
                Am = PtrMatrix(A, N, N, ld)
                @inbounds for j in 1:N, i in 1:N; Am[i, j] = Z[i, j]; end
            end
            unsafe_store!(info, Int64(inf)); return
        end

        # $(p)syevr_(jobz, range, uplo, n, A, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work,
        #            lwork, iwork, liwork, info, +3 lens) — vectors into the SEPARATE z buffer. range:
        #   'A'→all (m=N); 'I'→slice il:iu; 'V'→HALF-OPEN (vl,vu] band. isuppz[2i-1]=1,isuppz[2i]=N. abstol ignored.
        Base.@ccallable function $(Symbol(p, "syevr_64_"))(jobz::Ptr{UInt8}, range::Ptr{UInt8}, uplo::Ptr{UInt8},
                n::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, vl::Ptr{$T}, vu::Ptr{$T}, il::Ptr{Int64},
                iu::Ptr{Int64}, abstol::Ptr{$T}, m::Ptr{Int64}, w::Ptr{$T}, z::Ptr{$T}, ldz::Ptr{Int64},
                isuppz::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64}, iwork::Ptr{Int64}, liwork::Ptr{Int64},
                info::Ptr{Int64}, len_jobz::Clong, len_range::Clong, len_uplo::Clong)::Cvoid
            if unsafe_load(lwork) == Int64(-1)                 # workspace query
                unsafe_store!(work, one($T)); unsafe_store!(iwork, Int64(1)); unsafe_store!(info, Int64(0)); return
            end
            jz = _cabi_char(jobz); rg = _cabi_char(range); ul = _cabi_char(uplo)
            N = Int(unsafe_load(n)); ld = Int(unsafe_load(lda)); ldzz = Int(unsafe_load(ldz))
            wantz = jz == 'V'
            bad = (jz != 'N' && jz != 'V') ? -1 :
                  (rg != 'A' && rg != 'V' && rg != 'I') ? -2 :
                  (ul != 'U' && ul != 'L') ? -3 : (N < 0) ? -4 :
                  (rg == 'I' && N > 0 && !(1 <= Int(unsafe_load(il)) <= Int(unsafe_load(iu)) <= N)) ? -8 : 0
            if !iszero(bad)
                unsafe_store!(m, Int64(0)); unsafe_store!(info, Int64(bad)); return
            end
            wv, Z, inf = _syev_compute(jz, ul, N, A, ld)       # full spectrum (ascending) + vectors if wantz
            lo = 1; hi = N
            if rg == 'I'
                lo = Int(unsafe_load(il)); hi = Int(unsafe_load(iu))
            elseif rg == 'V'
                vlo = unsafe_load(vl); vhi = unsafe_load(vu)
                lo = N + 1; hi = 0
                @inbounds for i in 1:N
                    if vlo < wv[i] <= vhi
                        lo = min(lo, i); hi = max(hi, i)
                    end
                end
            end
            M = max(hi - lo + 1, 0)
            unsafe_store!(m, Int64(M))
            wm = PtrVector(w, max(M, 0))
            @inbounds for k in 1:M; wm[k] = wv[lo + k - 1]; end
            if wantz && M > 0
                Zm = PtrMatrix(z, N, M, ldzz)
                @inbounds for k in 1:M, i in 1:N; Zm[i, k] = Z[i, lo + k - 1]; end
            end
            ipz = PtrVector(isuppz, 2 * max(M, 0))
            @inbounds for k in 1:M; ipz[2k - 1] = Int64(1); ipz[2k] = Int64(N); end
            unsafe_store!(info, Int64(inf)); return
        end
    end
end

# ── COMPLEX heev/heevd/heevr (ComplexF64 native + ComplexF32 mixed-precision) ─────────────────────────
# Hermitian eigensolver via _heev! (hetrd → stedc/sterf → unmtr). ABI adds the LAPACK complex rwork block
# (and lrwork/iwork/liwork for heevd/heevr): w/rwork/vl/vu/abstol are REAL ($Tr); A/z/work are COMPLEX
# ($Tc). Julia's eigen(Hermitian) default → zheevr_/cheevr_. c-prefix promotes ComplexF32→ComplexF64.
for (p, Tc, Tr) in (("z", ComplexF64, Float64), ("c", ComplexF32, Float32))
    @eval begin
        # $(p)heev_(jobz, uplo, n, A, lda, w, work, lwork, rwork, info, len_jobz, len_uplo)
        Base.@ccallable function $(Symbol(p, "heev_64_"))(jobz::Ptr{UInt8}, uplo::Ptr{UInt8}, n::Ptr{Int64},
                A::Ptr{$Tc}, lda::Ptr{Int64}, w::Ptr{$Tr}, work::Ptr{$Tc}, lwork::Ptr{Int64},
                rwork::Ptr{$Tr}, info::Ptr{Int64}, len_jobz::Clong, len_uplo::Clong)::Cvoid
            if unsafe_load(lwork) == Int64(-1)                 # workspace query (only work[1] per zheev)
                unsafe_store!(work, one($Tc)); unsafe_store!(info, Int64(0)); return
            end
            jz = _cabi_char(jobz); ul = _cabi_char(uplo)
            N = Int(unsafe_load(n)); ld = Int(unsafe_load(lda))
            bad = (jz != 'N' && jz != 'V') ? -1 : (ul != 'U' && ul != 'L') ? -2 : (N < 0) ? -3 : 0
            !iszero(bad) && (unsafe_store!(info, Int64(bad)); return)
            wv, Z, inf = _heev_solve(jz, ul, N, A, ld)
            wm = PtrVector(w, N)
            @inbounds for i in eachindex(wm); wm[i] = $Tr(wv[i]); end
            if jz == 'V'
                Am = PtrMatrix(A, N, N, ld)
                @inbounds for j in 1:N, i in 1:N; Am[i, j] = $Tc(Z[i, j]); end
            end
            unsafe_store!(info, Int64(inf)); return
        end

        # $(p)heevd_(jobz, uplo, n, A, lda, w, work, lwork, rwork, lrwork, iwork, liwork, info, 2 lens)
        Base.@ccallable function $(Symbol(p, "heevd_64_"))(jobz::Ptr{UInt8}, uplo::Ptr{UInt8}, n::Ptr{Int64},
                A::Ptr{$Tc}, lda::Ptr{Int64}, w::Ptr{$Tr}, work::Ptr{$Tc}, lwork::Ptr{Int64},
                rwork::Ptr{$Tr}, lrwork::Ptr{Int64}, iwork::Ptr{Int64}, liwork::Ptr{Int64},
                info::Ptr{Int64}, len_jobz::Clong, len_uplo::Clong)::Cvoid
            if unsafe_load(lwork) == Int64(-1)                 # workspace query (work + rwork + iwork sizes)
                unsafe_store!(work, one($Tc)); unsafe_store!(rwork, one($Tr))
                unsafe_store!(iwork, Int64(1)); unsafe_store!(info, Int64(0)); return
            end
            jz = _cabi_char(jobz); ul = _cabi_char(uplo)
            N = Int(unsafe_load(n)); ld = Int(unsafe_load(lda))
            bad = (jz != 'N' && jz != 'V') ? -1 : (ul != 'U' && ul != 'L') ? -2 : (N < 0) ? -3 : 0
            !iszero(bad) && (unsafe_store!(info, Int64(bad)); return)
            wv, Z, inf = _heev_solve(jz, ul, N, A, ld)
            wm = PtrVector(w, N)
            @inbounds for i in eachindex(wm); wm[i] = $Tr(wv[i]); end
            if jz == 'V'
                Am = PtrMatrix(A, N, N, ld)
                @inbounds for j in 1:N, i in 1:N; Am[i, j] = $Tc(Z[i, j]); end
            end
            unsafe_store!(info, Int64(inf)); return
        end

        # $(p)heevr_(jobz, range, uplo, n, A, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work,
        #            lwork, rwork, lrwork, iwork, liwork, info, +3 lens) — vectors into the z buffer.
        Base.@ccallable function $(Symbol(p, "heevr_64_"))(jobz::Ptr{UInt8}, range::Ptr{UInt8}, uplo::Ptr{UInt8},
                n::Ptr{Int64}, A::Ptr{$Tc}, lda::Ptr{Int64}, vl::Ptr{$Tr}, vu::Ptr{$Tr}, il::Ptr{Int64},
                iu::Ptr{Int64}, abstol::Ptr{$Tr}, m::Ptr{Int64}, w::Ptr{$Tr}, z::Ptr{$Tc}, ldz::Ptr{Int64},
                isuppz::Ptr{Int64}, work::Ptr{$Tc}, lwork::Ptr{Int64}, rwork::Ptr{$Tr}, lrwork::Ptr{Int64},
                iwork::Ptr{Int64}, liwork::Ptr{Int64}, info::Ptr{Int64},
                len_jobz::Clong, len_range::Clong, len_uplo::Clong)::Cvoid
            if unsafe_load(lwork) == Int64(-1)                 # workspace query (work + rwork + iwork sizes)
                unsafe_store!(work, one($Tc)); unsafe_store!(rwork, one($Tr))
                unsafe_store!(iwork, Int64(1)); unsafe_store!(info, Int64(0)); return
            end
            jz = _cabi_char(jobz); rg = _cabi_char(range); ul = _cabi_char(uplo)
            N = Int(unsafe_load(n)); ld = Int(unsafe_load(lda)); ldzz = Int(unsafe_load(ldz))
            wantz = jz == 'V'
            bad = (jz != 'N' && jz != 'V') ? -1 :
                  (rg != 'A' && rg != 'V' && rg != 'I') ? -2 :
                  (ul != 'U' && ul != 'L') ? -3 : (N < 0) ? -4 :
                  (rg == 'I' && N > 0 && !(1 <= Int(unsafe_load(il)) <= Int(unsafe_load(iu)) <= N)) ? -10 : 0
            if !iszero(bad)
                unsafe_store!(m, Int64(0)); unsafe_store!(info, Int64(bad)); return
            end
            wv, Z, inf = _heev_solve(jz, ul, N, A, ld)         # full spectrum (ascending) + vectors if wantz
            lo = 1; hi = N
            if rg == 'I'
                lo = Int(unsafe_load(il)); hi = Int(unsafe_load(iu))
            elseif rg == 'V'
                vlo = unsafe_load(vl); vhi = unsafe_load(vu)
                lo = N + 1; hi = 0
                @inbounds for i in 1:N
                    if vlo < wv[i] <= vhi
                        lo = min(lo, i); hi = max(hi, i)
                    end
                end
            end
            M = max(hi - lo + 1, 0)
            unsafe_store!(m, Int64(M))
            wm = PtrVector(w, max(M, 0))
            @inbounds for k in 1:M; wm[k] = $Tr(wv[lo + k - 1]); end
            if wantz && M > 0
                Zm = PtrMatrix(z, N, M, ldzz)
                @inbounds for k in 1:M, i in 1:N; Zm[i, k] = $Tc(Z[i, lo + k - 1]); end
            end
            ipz = PtrVector(isuppz, 2 * max(M, 0))
            @inbounds for k in 1:M; ipz[2k - 1] = Int64(1); ipz[2k] = Int64(N); end
            unsafe_store!(info, Int64(inf)); return
        end
    end
end

# ── SOLVES on caller-provided factors: trtrs / potrs / getrs (compose trsm! + _laswp!) ────────────────
# Self-consistent under a mixed backend: they operate on the factors passed in (standard-convention L/U/
# Cholesky + LAPACK 1-based ipiv), so forwarding them is correct even if the factorization ran on OpenBLAS.
# This is what makes `A\b` / `ldiv!` route to PureBLAS after activate() (getrs is the solve step of `\`).

# trtrs: op(A)·X = B, A triangular n×n — a single trsm.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "trtrs_64_"))(uplo::Ptr{UInt8}, trans::Ptr{UInt8},
            diag::Ptr{UInt8}, n::Ptr{Int64}, nrhs::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64},
            B::Ptr{$T}, ldb::Ptr{Int64}, info::Ptr{Int64}, lu::Clong, lt::Clong, ld::Clong)::Cvoid
        N = Int(unsafe_load(n)); R = Int(unsafe_load(nrhs))
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        Bm = PtrMatrix(B, N, R, Int(unsafe_load(ldb)))
        trsm!(Bm, Am; side = 'L', uplo = _cabi_char(uplo), transA = _cabi_char(trans),
            diag = _cabi_char(diag), alpha = one($T))
        unsafe_store!(info, Int64(0)); return
    end
end

# potrs: A·X = B with A = Cholesky factor (L·Lᴴ if uplo='L', Uᴴ·U if 'U') — two trsm (conj-transpose for
# the complex Hermitian second solve).
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "potrs_64_"))(uplo::Ptr{UInt8}, n::Ptr{Int64},
            nrhs::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, B::Ptr{$T}, ldb::Ptr{Int64},
            info::Ptr{Int64}, lu::Clong)::Cvoid
        N = Int(unsafe_load(n)); R = Int(unsafe_load(nrhs))
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        Bm = PtrMatrix(B, N, R, Int(unsafe_load(ldb)))
        ct = $(T <: Complex ? 'C' : 'T')
        if _cabi_char(uplo) == 'L'
            trsm!(Bm, Am; side = 'L', uplo = 'L', transA = 'N', alpha = one($T))    # L·Y = B
            trsm!(Bm, Am; side = 'L', uplo = 'L', transA = ct, alpha = one($T))     # Lᴴ·X = Y
        else
            trsm!(Bm, Am; side = 'L', uplo = 'U', transA = ct, alpha = one($T))     # Uᴴ·Y = B
            trsm!(Bm, Am; side = 'L', uplo = 'U', transA = 'N', alpha = one($T))    # U·X = Y
        end
        unsafe_store!(info, Int64(0)); return
    end
end

# getrs: A·X = B with PA = LU (from getrf). trans='N': P·B then L\ then U\. trans='T'/'C': Uᵀ\, Lᵀ\, then
# the row interchanges in REVERSE (LAPACK dgetrs). ipiv is LAPACK 1-based.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "getrs_64_"))(trans::Ptr{UInt8}, n::Ptr{Int64},
            nrhs::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, ipiv::Ptr{Int64}, B::Ptr{$T},
            ldb::Ptr{Int64}, info::Ptr{Int64}, lt::Clong)::Cvoid
        N = Int(unsafe_load(n)); R = Int(unsafe_load(nrhs)); tr = _cabi_char(trans)
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        Bm = PtrMatrix(B, N, R, Int(unsafe_load(ldb)))
        ip = PtrVector(ipiv, N)
        if tr == 'N'
            _laswp!(Bm, ip, 1, N, 1, R)                                              # P·B (forward)
            trsm!(Bm, Am; side = 'L', uplo = 'L', transA = 'N', diag = 'U', alpha = one($T))  # L·Y = PB
            trsm!(Bm, Am; side = 'L', uplo = 'U', transA = 'N', diag = 'N', alpha = one($T))  # U·X = Y
        else
            trsm!(Bm, Am; side = 'L', uplo = 'U', transA = tr, diag = 'N', alpha = one($T))   # Uᵀ·Y = B
            trsm!(Bm, Am; side = 'L', uplo = 'L', transA = tr, diag = 'U', alpha = one($T))   # Lᵀ·Z = Y
            @inbounds for i in N:-1:1                                                 # reverse interchanges
                q = Int(ip[i])
                if q != i
                    for j in 1:R
                        Bm[i, j], Bm[q, j] = Bm[q, j], Bm[i, j]
                    end
                end
            end
        end
        unsafe_store!(info, Int64(0)); return
    end
end

# ── INVERSES: getri / potri / trtri (solve A·X = I on the factors, then copy back) ────────────────────
# Correctness-first: compute the inverse by solving A·X = I with the same trsm!/_laswp! machinery as the
# solves (self-consistent on any-backend factors). O(n³) scratch solve — fine (inverses aren't hot; prefer
# `\`). getri needs the LU + ipiv from getrf; potri/trtri need only the (Cholesky / triangular) factor.

# getri: A⁻¹ from the LU factors in A (+ ipiv). Solve A·X=I ⇒ laswp(I) then L\ then U\; copy X→A.
# The body is generic over T (trsm!/_laswp!), so Float32 (sgetri) is NATIVE via the F32 generic path
# (its getrf companion sgetrf routes via mixed precision, but the factors are standard-convention).
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "getri_64_"))(n::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64},
            ipiv::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64})::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        N = Int(unsafe_load(n)); Am = PtrMatrix(A, N, N, Int(unsafe_load(lda))); ip = PtrVector(ipiv, N)
        X = zeros($T, N, N)
        @inbounds for i in 1:N; X[i, i] = one($T); end
        GC.@preserve X begin
            Xm = PtrMatrix(pointer(X), N, N, N)
            _laswp!(Xm, ip, 1, N, 1, N)
            trsm!(Xm, Am; side = 'L', uplo = 'L', transA = 'N', diag = 'U', alpha = one($T))
            trsm!(Xm, Am; side = 'L', uplo = 'U', transA = 'N', diag = 'N', alpha = one($T))
            @inbounds for j in 1:N, i in 1:N; Am[i, j] = Xm[i, j]; end
        end
        unsafe_store!(info, Int64(0)); return
    end
end

# potri: A⁻¹ from the Cholesky factor in A (uplo). Solve A·X=I via the two triangular solves; copy the
# uplo triangle of X (A⁻¹ is Hermitian) back to A — LAPACK dpotri returns only the uplo triangle.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "potri_64_"))(uplo::Ptr{UInt8}, n::Ptr{Int64},
            A::Ptr{$T}, lda::Ptr{Int64}, info::Ptr{Int64}, lu::Clong)::Cvoid
        N = Int(unsafe_load(n)); ul = _cabi_char(uplo); Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        ct = $(T <: Complex ? 'C' : 'T')
        X = zeros($T, N, N)
        @inbounds for i in 1:N; X[i, i] = one($T); end
        GC.@preserve X begin
            Xm = PtrMatrix(pointer(X), N, N, N)
            if ul == 'L'
                trsm!(Xm, Am; side = 'L', uplo = 'L', transA = 'N', alpha = one($T))
                trsm!(Xm, Am; side = 'L', uplo = 'L', transA = ct, alpha = one($T))
                @inbounds for j in 1:N, i in j:N; Am[i, j] = Xm[i, j]; end     # lower triangle
            else
                trsm!(Xm, Am; side = 'L', uplo = 'U', transA = ct, alpha = one($T))
                trsm!(Xm, Am; side = 'L', uplo = 'U', transA = 'N', alpha = one($T))
                @inbounds for j in 1:N, i in 1:j; Am[i, j] = Xm[i, j]; end     # upper triangle
            end
        end
        unsafe_store!(info, Int64(0)); return
    end
end

# trtri: inverse of a triangular A (in place). Solve op(A)·X = I (one trsm), copy the uplo triangle back.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "trtri_64_"))(uplo::Ptr{UInt8}, diag::Ptr{UInt8},
            n::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, info::Ptr{Int64}, lu::Clong, ld::Clong)::Cvoid
        N = Int(unsafe_load(n)); ul = _cabi_char(uplo); dg = _cabi_char(diag)
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        X = zeros($T, N, N)
        @inbounds for i in 1:N; X[i, i] = one($T); end
        GC.@preserve X begin
            Xm = PtrMatrix(pointer(X), N, N, N)
            trsm!(Xm, Am; side = 'L', uplo = ul, transA = 'N', diag = dg, alpha = one($T))
            if ul == 'L'
                @inbounds for j in 1:N, i in j:N; Am[i, j] = Xm[i, j]; end
            else
                @inbounds for j in 1:N, i in 1:j; Am[i, j] = Xm[i, j]; end
            end
        end
        unsafe_store!(info, Int64(0)); return
    end
end

# ── Complex QR (zgeqrt / zgemqrt): same structure as real, but the compact-WY T uses VᴴV (herk) not VᵀV,
# τ is ALREADY LAPACK-convention (no inversion), and the block reflector is Q = I − V·T·Vᴴ (conjugate).
# Mirrors complex geqrf!'s proven zlarft + trailing-update pattern (qr.jl). ComplexF64 only for now.
@inline function _qr_t_cmplx!(Tm::AbstractMatrix{T}, Vp::AbstractMatrix{T}, tau::AbstractVector{T},
        G::AbstractMatrix{T}) where {T<:Complex}
    bs = length(tau); bs == 0 && return Tm
    m = size(Vp, 1)
    Vv = view(Vp, 1:m, 1:bs); Gv = view(G, 1:bs, 1:bs)
    herk!(Gv, Vv; uplo = 'U', trans = 'C', alpha = true, beta = false)   # G = VᴴV (upper)
    @inbounds for c in 1:bs
        τc = tau[c]; Tm[c, c] = τc
        for r in 1:(c - 1)
            s = zero(T)
            for kk in r:(c - 1); s = muladd(Tm[r, kk], -τc * Gv[kk, c], s); end
            Tm[r, c] = s
        end
        for r in (c + 1):bs; Tm[r, c] = zero(T); end
    end
    return Tm
end
# Left apply: C := op(Q)·C, op(Q)= Q (trans 'N') or Qᴴ (trans 'C'), Q = I − V·T·Vᴴ.
@inline function _qr_apply_left_cmplx!(trans::Char, C::AbstractMatrix{T}, Vp::AbstractMatrix{T},
        Tblk::AbstractMatrix{T}) where {T<:Complex}
    mp = size(C, 1); ib = size(Vp, 2); nc = size(C, 2)
    (ib == 0 || nc == 0 || mp == 0) && return C
    W = Matrix{T}(undef, ib, nc)
    gemm!(W, Vp, C; transA = 'C', alpha = true, beta = false)            # W = Vᴴ·C
    trmm!(W, Tblk; side = 'L', uplo = 'U', transA = trans)               # W := op(T)·W  (trans 'N'/'C')
    gemm!(C, Vp, W; alpha = -one(T), beta = one(T))                      # C -= V·W
    return C
end
# Right apply: C := C·op(Q).
@inline function _qr_apply_right_cmplx!(trans::Char, C::AbstractMatrix{T}, Vp::AbstractMatrix{T},
        Tblk::AbstractMatrix{T}) where {T<:Complex}
    nr = size(C, 1); ib = size(Vp, 2)
    (nr == 0 || ib == 0) && return C
    W = Matrix{T}(undef, nr, ib)
    gemm!(W, C, Vp; alpha = true, beta = false)                         # W = C·V
    trmm!(W, Tblk; side = 'R', uplo = 'U', transA = trans)              # W := W·op(T)
    gemm!(C, W, Vp; transB = 'C', alpha = -one(T), beta = one(T))       # C -= W·Vᴴ
    return C
end

# Generated for ComplexF64 (z, native) and ComplexF32 (c, native — geqrf!/herk!/gemm!/trmm! are all
# generic over T<:BlasComplex, so ComplexF32 QR needs no mixed precision). Routes qr(::Matrix{Complex*}).
for (p, T) in (("z", ComplexF64), ("c", ComplexF32))
    @eval Base.@ccallable function $(Symbol(p, "geqrt_64_"))(m::Ptr{Int64}, n::Ptr{Int64}, nb::Ptr{Int64},
            A::Ptr{$T}, lda::Ptr{Int64}, T::Ptr{$T}, ldt::Ptr{Int64}, work::Ptr{$T}, info::Ptr{Int64})::Cvoid
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); NB = Int(unsafe_load(nb)); k = min(M, N)
        Av = PtrMatrix(A, M, N, Int(unsafe_load(lda)))
        Tm = PtrMatrix(T, NB, k, Int(unsafe_load(ldt)))
        τ = Vector{$T}(undef, k)
        GC.@preserve τ begin
            geqrf!(Av, PtrVector(pointer(τ), k))            # complex τ already LAPACK-convention
            Vpan = Matrix{$T}(undef, M, NB); Gs = Matrix{$T}(undef, NB, NB)
            for i in 1:NB:k
                ib = min(NB, k - i + 1); mp = M - i + 1
                Vp = view(Vpan, 1:mp, 1:ib)
                _qr_vpanel!(Vp, Av, i, ib, mp)
                _qr_t_cmplx!(view(Tm, 1:ib, i:(i + ib - 1)), Vp, view(τ, i:(i + ib - 1)), Gs)
            end
        end
        unsafe_store!(info, Int64(0)); return
    end

    @eval Base.@ccallable function $(Symbol(p, "gemqrt_64_"))(side::Ptr{UInt8}, trans::Ptr{UInt8},
            m::Ptr{Int64}, n::Ptr{Int64}, k::Ptr{Int64}, nb::Ptr{Int64}, V::Ptr{$T}, ldv::Ptr{Int64},
            T::Ptr{$T}, ldt::Ptr{Int64}, C::Ptr{$T}, ldc::Ptr{Int64}, work::Ptr{$T}, info::Ptr{Int64},
            len_s::Clong, len_t::Clong)::Cvoid
        sd = _cabi_char(side); tr = _cabi_char(trans)
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); K = Int(unsafe_load(k)); NB = Int(unsafe_load(nb))
        vrows = sd == 'L' ? M : N
        Vm = PtrMatrix(V, vrows, K, Int(unsafe_load(ldv)))
        Tm = PtrMatrix(T, NB, K, Int(unsafe_load(ldt)))
        Cm = PtrMatrix(C, M, N, Int(unsafe_load(ldc)))
        forward = (sd == 'L' && tr != 'N') || (sd == 'R' && tr == 'N') # L+C/R+N forward; L+N/R+C reverse
        starts = collect(1:NB:K); forward || reverse!(starts)
        Vpan = Matrix{$T}(undef, vrows, NB)
        for i in starts
            ib = min(NB, K - i + 1); mp = vrows - i + 1
            Vp = view(Vpan, 1:mp, 1:ib)
            _qr_vpanel!(Vp, Vm, i, ib, mp)
            Tblk = view(Tm, 1:ib, i:(i + ib - 1))
            if sd == 'L'
                _qr_apply_left_cmplx!(tr, view(Cm, i:M, 1:N), Vp, Tblk)
            else
                _qr_apply_right_cmplx!(tr, view(Cm, 1:M, i:N), Vp, Tblk)
            end
        end
        unsafe_store!(info, Int64(0)); return
    end
end

# ── Float32 LAPACK via MIXED PRECISION (promote→Float64 kernel→demote) ────────────────────────────────
# PureBLAS has no native Float32 getrf/geqrf/gesvd kernels (only potrf). To route sgetrf/sgeqrt/sgemqrt/
# sgesdd/sgesvd to PureBLAS anyway, we compute in Float64 and round back to Float32 — correct to F32
# precision (actually a touch more accurate), reusing the tuned F64 kernels. Native F32 SIMD kernels (~2×)
# are a perf follow-up. `spotrf`/`strtrs`/`spotrs`/`sgetrs`/`s{tri}` already route via the generic F32 paths.
@inline function _f32_to_f64!(dst::Matrix{Float64}, src::PtrMatrix{Float32}, M::Int, N::Int)
    @inbounds for j in 1:N, i in 1:M; dst[i, j] = Float64(src[i, j]); end
end
@inline function _f64_to_f32!(dst::PtrMatrix{Float32}, src::Matrix{Float64}, M::Int, N::Int)
    @inbounds for j in 1:N, i in 1:M; dst[i, j] = Float32(src[i, j]); end
end

Base.@ccallable function sgetrf_64_(m::Ptr{Int64}, n::Ptr{Int64}, A::Ptr{Float32}, lda::Ptr{Int64},
        ipiv::Ptr{Int64}, info::Ptr{Int64})::Cvoid
    M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); Am = PtrMatrix(A, M, N, Int(unsafe_load(lda)))
    Af = Matrix{Float64}(undef, M, N); _f32_to_f64!(Af, Am, M, N)
    ip = PtrVector(ipiv, min(M, N))
    GC.@preserve Af begin
        _, _, inf = getrf!(PtrMatrix(pointer(Af), M, N, M), ip)
        _f64_to_f32!(Am, Af, M, N); unsafe_store!(info, Int64(inf))
    end
    return
end

Base.@ccallable function sgeqrt_64_(m::Ptr{Int64}, n::Ptr{Int64}, nb::Ptr{Int64}, A::Ptr{Float32},
        lda::Ptr{Int64}, T::Ptr{Float32}, ldt::Ptr{Int64}, work::Ptr{Float32}, info::Ptr{Int64})::Cvoid
    M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); NB = Int(unsafe_load(nb)); k = min(M, N)
    Am = PtrMatrix(A, M, N, Int(unsafe_load(lda))); Tm = PtrMatrix(T, NB, k, Int(unsafe_load(ldt)))
    Af = Matrix{Float64}(undef, M, N); _f32_to_f64!(Af, Am, M, N)
    Tf = Matrix{Float64}(undef, NB, k)
    GC.@preserve Af Tf begin
        Avf = PtrMatrix(pointer(Af), M, N, M); τ = Vector{Float64}(undef, k)
        GC.@preserve τ begin
            geqrf!(Avf, PtrVector(pointer(τ), k))
            @inbounds for i in 1:k; t = τ[i]; τ[i] = (isfinite(t) && t != 0.0) ? 1.0 / t : 0.0; end
            ws = WYApplyWorkspace{Float64}(M, NB, N)
            Tvf = PtrMatrix(pointer(Tf), NB, k, NB)
            for i in 1:NB:k
                ib = min(NB, k - i + 1); mp = M - i + 1
                Vp = view(ws.V, 1:mp, 1:ib); _qr_vpanel!(Vp, Avf, i, ib, mp)
                wy_t!(view(Tvf, 1:ib, i:(i + ib - 1)), Vp, view(τ, i:(i + ib - 1)), view(ws.G, 1:ib, 1:ib))
            end
        end
        _f64_to_f32!(Am, Af, M, N); _f64_to_f32!(Tm, Tf, NB, k)
    end
    unsafe_store!(info, Int64(0)); return
end

Base.@ccallable function sgemqrt_64_(side::Ptr{UInt8}, trans::Ptr{UInt8}, m::Ptr{Int64}, n::Ptr{Int64},
        k::Ptr{Int64}, nb::Ptr{Int64}, V::Ptr{Float32}, ldv::Ptr{Int64}, T::Ptr{Float32}, ldt::Ptr{Int64},
        C::Ptr{Float32}, ldc::Ptr{Int64}, work::Ptr{Float32}, info::Ptr{Int64},
        len_s::Clong, len_t::Clong)::Cvoid
    sd = _cabi_char(side); tr = _cabi_char(trans)
    M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); K = Int(unsafe_load(k)); NB = Int(unsafe_load(nb))
    vrows = sd == 'L' ? M : N
    Vm = PtrMatrix(V, vrows, K, Int(unsafe_load(ldv))); Tm = PtrMatrix(T, NB, K, Int(unsafe_load(ldt)))
    Cm = PtrMatrix(C, M, N, Int(unsafe_load(ldc)))
    Vf = Matrix{Float64}(undef, vrows, K); _f32_to_f64!(Vf, Vm, vrows, K)
    Tf = Matrix{Float64}(undef, NB, K); _f32_to_f64!(Tf, Tm, NB, K)
    Cf = Matrix{Float64}(undef, M, N); _f32_to_f64!(Cf, Cm, M, N)
    GC.@preserve Vf Tf Cf begin
        Vmf = PtrMatrix(pointer(Vf), vrows, K, vrows); Tmf = PtrMatrix(pointer(Tf), NB, K, NB)
        Cmf = PtrMatrix(pointer(Cf), M, N, M)
        forward = (sd == 'L' && tr != 'N') || (sd == 'R' && tr == 'N')
        starts = collect(1:NB:K); forward || reverse!(starts)
        ws = WYApplyWorkspace{Float64}(vrows, NB, max(M, N))
        for i in starts
            ib = min(NB, K - i + 1); mp = vrows - i + 1
            Vp = view(ws.V, 1:mp, 1:ib); _qr_vpanel!(Vp, Vmf, i, ib, mp)
            Tblk = view(Tmf, 1:ib, i:(i + ib - 1))
            sd == 'L' ? wy_apply!(tr, view(Cmf, i:M, 1:N), Vp, Tblk, ws) :
                        _qr_apply_right!(tr, view(Cmf, 1:M, i:N), Vp, Tblk)
        end
        _f64_to_f32!(Cm, Cf, M, N)
    end
    unsafe_store!(info, Int64(0)); return
end

# F32 SVD via mixed precision: promote A→F64, run the F64 gesvd! driver into F64 scratch, demote outputs.
@inline function _svd_cabi_f32!(ju::Char, jvt::Char, M::Int, N::Int, A::Ptr{Float32}, ld::Int,
        S::Ptr{Float32}, U::Ptr{Float32}, ldu::Int, VT::Ptr{Float32}, ldvt::Int, info::Ptr{Int64})
    mn = min(M, N)
    Af = Matrix{Float64}(undef, M, N); _f32_to_f64!(Af, PtrMatrix(A, M, N, ld), M, N)
    Sf = Vector{Float64}(undef, mn)
    needU = ju != 'N'; needV = jvt != 'N'
    ncu = ju == 'A' ? M : mn; ncv = jvt == 'A' ? N : mn
    Uf = Matrix{Float64}(undef, M, needU ? ncu : 1)
    Vf = Matrix{Float64}(undef, needV ? ncv : 1, N)
    GC.@preserve Af Sf Uf Vf begin
        _svd_cabi!(ju, jvt, M, N, pointer(Af), M, pointer(Sf), pointer(Uf), M,
            pointer(Vf), needV ? ncv : 1, info)
        unsafe_load(info) == 0 || return
        Sm = PtrVector(S, mn); @inbounds for i in 1:mn; Sm[i] = Float32(Sf[i]); end
        needU && _f64_to_f32!(PtrMatrix(U, M, ncu, ldu), Uf, M, ncu)
        needV && _f64_to_f32!(PtrMatrix(VT, ncv, N, ldvt), Vf, ncv, N)
    end
    return
end

Base.@ccallable function sgesvd_64_(jobu::Ptr{UInt8}, jobvt::Ptr{UInt8}, m::Ptr{Int64}, n::Ptr{Int64},
        A::Ptr{Float32}, lda::Ptr{Int64}, S::Ptr{Float32}, U::Ptr{Float32}, ldu::Ptr{Int64},
        VT::Ptr{Float32}, ldvt::Ptr{Int64}, work::Ptr{Float32}, lwork::Ptr{Int64}, info::Ptr{Int64},
        len_ju::Clong, len_jvt::Clong)::Cvoid
    if unsafe_load(lwork) == Int64(-1)
        unsafe_store!(work, 1.0f0); unsafe_store!(info, Int64(0)); return
    end
    _svd_cabi_f32!(_cabi_char(jobu), _cabi_char(jobvt), Int(unsafe_load(m)), Int(unsafe_load(n)),
        A, Int(unsafe_load(lda)), S, U, Int(unsafe_load(ldu)), VT, Int(unsafe_load(ldvt)), info)
    return
end

Base.@ccallable function sgesdd_64_(jobz::Ptr{UInt8}, m::Ptr{Int64}, n::Ptr{Int64}, A::Ptr{Float32},
        lda::Ptr{Int64}, S::Ptr{Float32}, U::Ptr{Float32}, ldu::Ptr{Int64}, VT::Ptr{Float32},
        ldvt::Ptr{Int64}, work::Ptr{Float32}, lwork::Ptr{Int64}, iwork::Ptr{Int64}, info::Ptr{Int64},
        len_jobz::Clong)::Cvoid
    if unsafe_load(lwork) == Int64(-1)
        unsafe_store!(work, 1.0f0); unsafe_store!(info, Int64(0)); return
    end
    jz = _cabi_char(jobz)
    if jz == 'O'; unsafe_store!(info, Int64(-1)); return; end
    _svd_cabi_f32!(jz, jz, Int(unsafe_load(m)), Int(unsafe_load(n)),
        A, Int(unsafe_load(lda)), S, U, Int(unsafe_load(ldu)), VT, Int(unsafe_load(ldvt)), info)
    return
end

# ══════════════════════════════════════════════════════════════════════════════════════════════════
# Batch 6: LQ / Bunch-Kaufman / pivoted-QR / least-squares / condition / Hessenberg reduction.
# All six kernels are GENERIC over s/d/c/z (native — NOT mixed precision), so every wrapper forwards
# straight to the generic kernel per element type. Char args → Ptr{UInt8} first (deref via _cabi_char),
# trailing Fortran string-length Clong(s) last (one per char). Routines with lwork honor the -1 query
# (report work[1]=1, info=0). Cross-checked vs LinearAlgebra/lapack.jl ccalls for arg order + hidden lens.
# ══════════════════════════════════════════════════════════════════════════════════════════════════

# ── LQ: gelqf / orglq·unglq / ormlq·unmlq — routes lq(A), Matrix(lq(A).Q), lq(A).Q ops ────────────────
# {s,d,c,z}gelqf_64_(m, n, A, lda, tau, work, lwork, info) — 0 chars.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "gelqf_64_"))(m::Ptr{Int64}, n::Ptr{Int64}, A::Ptr{$T},
            lda::Ptr{Int64}, tau::Ptr{$T}, work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64})::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); k = min(M, N)
        gelqf!(PtrMatrix(A, M, N, Int(unsafe_load(lda))), PtrVector(tau, k))
        unsafe_store!(info, Int64(0)); return
    end
end

# orglq (real) / unglq (complex): (m, n, k, A, lda, tau, work, lwork, info) — 0 chars. Reference name
# differs by type (dorglq vs zunglq) but the kernel is one generic orglq!.
for (nm, T) in (("sorglq", Float32), ("dorglq", Float64), ("cunglq", ComplexF32), ("zunglq", ComplexF64))
    @eval Base.@ccallable function $(Symbol(nm, "_64_"))(m::Ptr{Int64}, n::Ptr{Int64}, k::Ptr{Int64},
            A::Ptr{$T}, lda::Ptr{Int64}, tau::Ptr{$T}, work::Ptr{$T}, lwork::Ptr{Int64},
            info::Ptr{Int64})::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); K = Int(unsafe_load(k))
        orglq!(PtrMatrix(A, M, N, Int(unsafe_load(lda))), PtrVector(tau, K))
        unsafe_store!(info, Int64(0)); return
    end
end

# ormlq (real) / unmlq (complex): (side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info) — 2 chars.
# A is the gelqf reflector panel: K rows × nq cols (nq = m if side='L' else n).
for (nm, T) in (("sormlq", Float32), ("dormlq", Float64), ("cunmlq", ComplexF32), ("zunmlq", ComplexF64))
    @eval Base.@ccallable function $(Symbol(nm, "_64_"))(side::Ptr{UInt8}, trans::Ptr{UInt8},
            m::Ptr{Int64}, n::Ptr{Int64}, k::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, tau::Ptr{$T},
            C::Ptr{$T}, ldc::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64},
            len_s::Clong, len_t::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        sd = _cabi_char(side); tr = _cabi_char(trans)
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); K = Int(unsafe_load(k))
        nq = sd == 'L' ? M : N
        Am = PtrMatrix(A, K, nq, Int(unsafe_load(lda)))
        Cm = PtrMatrix(C, M, N, Int(unsafe_load(ldc)))
        ormlq!(sd, tr, Am, PtrVector(tau, K), Cm)
        unsafe_store!(info, Int64(0)); return
    end
end

# ── Bunch-Kaufman: sytrf/hetrf + sytrs/hetrs — routes bunchkaufman(Symmetric/Hermitian) + its solve ───
# {s,d,c,z}sytrf_64_(uplo, n, A, lda, ipiv, work, lwork, info, len_uplo) — 1 char. ipiv OUT (LAPACK enc).
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "sytrf_64_"))(uplo::Ptr{UInt8}, n::Ptr{Int64}, A::Ptr{$T},
            lda::Ptr{Int64}, ipiv::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64},
            lu::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        N = Int(unsafe_load(n))
        inf = sytrf!(PtrMatrix(A, N, N, Int(unsafe_load(lda))), PtrVector(ipiv, N); uplo = _cabi_char(uplo))
        unsafe_store!(info, Int64(inf)); return
    end
end
# {c,z}hetrf_64_ — Hermitian variant (complex only; real Hermitian bunchkaufman routes to sytrf).
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "hetrf_64_"))(uplo::Ptr{UInt8}, n::Ptr{Int64}, A::Ptr{$T},
            lda::Ptr{Int64}, ipiv::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64},
            lu::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        N = Int(unsafe_load(n))
        inf = hetrf!(PtrMatrix(A, N, N, Int(unsafe_load(lda))), PtrVector(ipiv, N); uplo = _cabi_char(uplo))
        unsafe_store!(info, Int64(inf)); return
    end
end
# {s,d,c,z}sytrs_64_(uplo, n, nrhs, A, lda, ipiv, B, ldb, info, len_uplo) — 1 char.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "sytrs_64_"))(uplo::Ptr{UInt8}, n::Ptr{Int64},
            nrhs::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, ipiv::Ptr{Int64}, B::Ptr{$T},
            ldb::Ptr{Int64}, info::Ptr{Int64}, lu::Clong)::Cvoid
        N = Int(unsafe_load(n)); R = Int(unsafe_load(nrhs))
        sytrs!(PtrMatrix(A, N, N, Int(unsafe_load(lda))), PtrVector(ipiv, N),
            PtrMatrix(B, N, R, Int(unsafe_load(ldb))); uplo = _cabi_char(uplo))
        unsafe_store!(info, Int64(0)); return
    end
end
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "hetrs_64_"))(uplo::Ptr{UInt8}, n::Ptr{Int64},
            nrhs::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, ipiv::Ptr{Int64}, B::Ptr{$T},
            ldb::Ptr{Int64}, info::Ptr{Int64}, lu::Clong)::Cvoid
        N = Int(unsafe_load(n)); R = Int(unsafe_load(nrhs))
        hetrs!(PtrMatrix(A, N, N, Int(unsafe_load(lda))), PtrVector(ipiv, N),
            PtrMatrix(B, N, R, Int(unsafe_load(ldb))); uplo = _cabi_char(uplo))
        unsafe_store!(info, Int64(0)); return
    end
end

# ── geqp3: pivoted QR — routes qr(A, ColumnNorm()) and non-square A\b ─────────────────────────────────
# real {s,d}geqp3_64_(m, n, A, lda, jpvt, tau, work, lwork, info) — 0 chars.
# complex {c,z}geqp3_64_(m, n, A, lda, jpvt, tau, work, lwork, rwork, info) — extra REAL rwork block.
# jpvt is Ptr{Int64} (1-based, both IN as the free-column mask AND OUT as the permutation). The kernel
# initializes jpvt itself (writes 1:n then permutes), matching LAPACK's behavior for the all-free case
# that qr(::Matrix, ColumnNorm()) uses (Julia passes jpvt = zeros).
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "geqp3_64_"))(m::Ptr{Int64}, n::Ptr{Int64}, A::Ptr{$T},
            lda::Ptr{Int64}, jpvt::Ptr{Int64}, tau::Ptr{$T}, work::Ptr{$T}, lwork::Ptr{Int64},
            info::Ptr{Int64})::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); k = min(M, N)
        geqp3!(PtrMatrix(A, M, N, Int(unsafe_load(lda))), PtrVector(jpvt, N), PtrVector(tau, k))
        unsafe_store!(info, Int64(0)); return
    end
end
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))
    Tr = T == ComplexF32 ? Float32 : Float64
    @eval Base.@ccallable function $(Symbol(p, "geqp3_64_"))(m::Ptr{Int64}, n::Ptr{Int64}, A::Ptr{$T},
            lda::Ptr{Int64}, jpvt::Ptr{Int64}, tau::Ptr{$T}, work::Ptr{$T}, lwork::Ptr{Int64},
            rwork::Ptr{$Tr}, info::Ptr{Int64})::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); k = min(M, N)
        geqp3!(PtrMatrix(A, M, N, Int(unsafe_load(lda))), PtrVector(jpvt, N), PtrVector(tau, k))
        unsafe_store!(info, Int64(0)); return
    end
end

# ── gels: least-squares / min-norm — {s,d,c,z}gels_64_(trans, m, n, nrhs, A, lda, B, ldb, work, lwork,
# info, len_trans) — 1 char. The SOLUTION is written into B (LAPACK ldb layout: B is max(m,n)×nrhs, first
# `cols(op A)` rows hold x, the tail rows hold Qᴴ·b's residual block). NOTE: the PureBLAS gels! kernel
# does NOT overwrite A with the QR/LQ factors (LAPACK does) — see the assembly report. For routing `\` and
# for the solution vector this is immaterial (Julia's non-square `\` goes through geqp3, not gels; the
# solution in B is the load-bearing output). A direct LAPACK.gels! call gets a correct B + residuals.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "gels_64_"))(trans::Ptr{UInt8}, m::Ptr{Int64},
            n::Ptr{Int64}, nrhs::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, B::Ptr{$T}, ldb::Ptr{Int64},
            work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64}, lt::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); R = Int(unsafe_load(nrhs))
        Am = PtrMatrix(A, M, N, Int(unsafe_load(lda)))
        Bm = PtrMatrix(B, max(M, N), R, Int(unsafe_load(ldb)))
        gels!(_cabi_char(trans), Am, Bm)
        unsafe_store!(info, Int64(0)); return
    end
end

# ── Condition estimation: gecon / trcon / pocon. rcond is a by-ref REAL OUTPUT scalar. anorm a by-ref
# REAL input. Complex drivers carry a REAL rwork block where the real ones carry an integer iwork — both
# ignored (PureBLAS owns its scratch), but the ABI slot type differs so the wrappers are split by type.
# gecon: (normtype, n, A, lda, anorm, rcond, work, iwork|rwork, info, len_norm) — 1 char. The LAPACK ABI
# passes NO ipiv (the 1-norm/∞-norm estimate is invariant under the LU row permutation P — permuting the
# operator's rows/cols leaves the max column-sum unchanged), so we feed the kernel an identity ipiv.
for (p, T, Tr, IW) in (("s", Float32, Float32, Int64), ("d", Float64, Float64, Int64),
                       ("c", ComplexF32, Float32, Float32), ("z", ComplexF64, Float64, Float64))
    @eval Base.@ccallable function $(Symbol(p, "gecon_64_"))(normtype::Ptr{UInt8}, n::Ptr{Int64},
            A::Ptr{$T}, lda::Ptr{Int64}, anorm::Ptr{$Tr}, rcond::Ptr{$Tr}, work::Ptr{$T},
            iwork::Ptr{$IW}, info::Ptr{Int64}, ln::Clong)::Cvoid
        N = Int(unsafe_load(n))
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        idp = Vector{Int64}(undef, N)
        @inbounds for i in 1:N; idp[i] = i; end
        rc = gecon!($Tr(unsafe_load(anorm)), Am, idp; norm = _cabi_char(normtype))
        unsafe_store!(rcond, $Tr(rc)); unsafe_store!(info, Int64(0)); return
    end
end
# trcon: (norm, uplo, diag, n, A, lda, rcond, work, iwork|rwork, info, len_norm, len_uplo, len_diag) — 3 chars.
for (p, T, Tr, IW) in (("s", Float32, Float32, Int64), ("d", Float64, Float64, Int64),
                       ("c", ComplexF32, Float32, Float32), ("z", ComplexF64, Float64, Float64))
    @eval Base.@ccallable function $(Symbol(p, "trcon_64_"))(norm::Ptr{UInt8}, uplo::Ptr{UInt8},
            diag::Ptr{UInt8}, n::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, rcond::Ptr{$Tr}, work::Ptr{$T},
            iwork::Ptr{$IW}, info::Ptr{Int64}, ln::Clong, lu::Clong, ld::Clong)::Cvoid
        N = Int(unsafe_load(n))
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        rc = trcon!(Am; uplo = _cabi_char(uplo), diag = _cabi_char(diag), norm = _cabi_char(norm))
        unsafe_store!(rcond, $Tr(rc)); unsafe_store!(info, Int64(0)); return
    end
end
# pocon: (uplo, n, A, lda, anorm, rcond, work, iwork|rwork, info, len_uplo) — 1 char.
for (p, T, Tr, IW) in (("s", Float32, Float32, Int64), ("d", Float64, Float64, Int64),
                       ("c", ComplexF32, Float32, Float32), ("z", ComplexF64, Float64, Float64))
    @eval Base.@ccallable function $(Symbol(p, "pocon_64_"))(uplo::Ptr{UInt8}, n::Ptr{Int64}, A::Ptr{$T},
            lda::Ptr{Int64}, anorm::Ptr{$Tr}, rcond::Ptr{$Tr}, work::Ptr{$T}, iwork::Ptr{$IW},
            info::Ptr{Int64}, lu::Clong)::Cvoid
        N = Int(unsafe_load(n))
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        rc = pocon!($Tr(unsafe_load(anorm)), Am; uplo = _cabi_char(uplo))
        unsafe_store!(rcond, $Tr(rc)); unsafe_store!(info, Int64(0)); return
    end
end

# ── Hessenberg reduction: gebal / gehrd / orghr·unghr — routes hessenberg(A) (gehrd) + reductions ─────
# gebal: (job, n, A, lda, ilo, ihi, scale, info, len_job) — 1 char. ilo/ihi Ptr{Int64} OUT, scale Ptr{real} OUT.
for (p, T, Tr) in (("s", Float32, Float32), ("d", Float64, Float64),
                   ("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval Base.@ccallable function $(Symbol(p, "gebal_64_"))(job::Ptr{UInt8}, n::Ptr{Int64}, A::Ptr{$T},
            lda::Ptr{Int64}, ilo::Ptr{Int64}, ihi::Ptr{Int64}, scale::Ptr{$Tr}, info::Ptr{Int64},
            lj::Clong)::Cvoid
        N = Int(unsafe_load(n))
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        k, l, scl = gebal!(Am; job = _cabi_char(job))
        sm = PtrVector(scale, N)
        @inbounds for i in 1:N; sm[i] = scl[i]; end
        unsafe_store!(ilo, Int64(k)); unsafe_store!(ihi, Int64(l)); unsafe_store!(info, Int64(0)); return
    end
end
# gehrd: (n, ilo, ihi, A, lda, tau, work, lwork, info) — 0 chars. tau length max(n-1,0).
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "gehrd_64_"))(n::Ptr{Int64}, ilo::Ptr{Int64},
            ihi::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, tau::Ptr{$T}, work::Ptr{$T},
            lwork::Ptr{Int64}, info::Ptr{Int64})::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        N = Int(unsafe_load(n))
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        gehrd!(Am, Int(unsafe_load(ilo)), Int(unsafe_load(ihi)), PtrVector(tau, max(N - 1, 0)))
        unsafe_store!(info, Int64(0)); return
    end
end
# orghr (real) / unghr (complex): (n, ilo, ihi, A, lda, tau, work, lwork, info) — 0 chars. Overwrites A with Q.
for (nm, T) in (("sorghr", Float32), ("dorghr", Float64), ("cunghr", ComplexF32), ("zunghr", ComplexF64))
    @eval Base.@ccallable function $(Symbol(nm, "_64_"))(n::Ptr{Int64}, ilo::Ptr{Int64}, ihi::Ptr{Int64},
            A::Ptr{$T}, lda::Ptr{Int64}, tau::Ptr{$T}, work::Ptr{$T}, lwork::Ptr{Int64},
            info::Ptr{Int64})::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        N = Int(unsafe_load(n))
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        orghr!(Am, Int(unsafe_load(ilo)), Int(unsafe_load(ihi)), PtrVector(tau, max(N - 1, 0)))
        unsafe_store!(info, Int64(0)); return
    end
end

# ── geev / geevx: general eigensolver (eigenvalues + right eigenvectors) ───────────────────────────────
# Routes eigen(A)/eigvals(A) (Julia's default is geevx! with sense='N', balanc='B'/'N'; geev is the plain
# driver + non-Julia hosts). jobvl='V' (left vectors) is unsupported by our trevc — reported via info<0.
# _geev_run! composes gebal→gehrd→orghr→hseqr→trevc→gebak→normalize and mutates A (→ Schur form) in place,
# returning freshly-allocated VR (real-packed for real A) copied into the caller's buffer; wr/wi (real) or
# w (complex) into their buffers. Honors the lwork==-1 query. Everything is a PtrMatrix ⇒ trim-safe.

# REAL {s,d}geev_(jobvl, jobvr, n, A, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info, +2 lens).
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "geev_64_"))(jobvl::Ptr{UInt8}, jobvr::Ptr{UInt8},
            n::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, wr::Ptr{$T}, wi::Ptr{$T}, vl::Ptr{$T},
            ldvl::Ptr{Int64}, vr::Ptr{$T}, ldvr::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64},
            info::Ptr{Int64}, ljl::Clong, ljr::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        jl = _cabi_char(jobvl); jr = _cabi_char(jobvr)
        if jl != 'N'                                       # left eigenvectors not implemented
            unsafe_store!(info, Int64(-1)); return
        end
        N = Int(unsafe_load(n)); ld = Int(unsafe_load(lda))
        Am = PtrMatrix(A, N, N, ld)
        wra, wia, _, VRm = _geev_run!('B', jl, jr, Am)
        wrp = PtrVector(wr, N); wip = PtrVector(wi, N)
        @inbounds for i in 1:N; wrp[i] = wra[i]; wip[i] = wia[i]; end
        if jr == 'V'
            VRp = PtrMatrix(vr, N, N, Int(unsafe_load(ldvr)))
            @inbounds for j in 1:N, i in 1:N; VRp[i, j] = VRm[i, j]; end
        end
        unsafe_store!(info, Int64(0)); return
    end
end

# COMPLEX {c,z}geev_(jobvl, jobvr, n, A, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info, +2 lens).
# The complex ABI has ONE w output (vs real's wr/wi) and inserts a REAL rwork block the real driver lacks.
for (p, T, R) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval Base.@ccallable function $(Symbol(p, "geev_64_"))(jobvl::Ptr{UInt8}, jobvr::Ptr{UInt8},
            n::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, w::Ptr{$T}, vl::Ptr{$T}, ldvl::Ptr{Int64},
            vr::Ptr{$T}, ldvr::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64}, rwork::Ptr{$R},
            info::Ptr{Int64}, ljl::Clong, ljr::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        jl = _cabi_char(jobvl); jr = _cabi_char(jobvr)
        if jl != 'N'
            unsafe_store!(info, Int64(-1)); return
        end
        N = Int(unsafe_load(n)); ld = Int(unsafe_load(lda))
        Am = PtrMatrix(A, N, N, ld)
        wa, _, VRm = _geev_run!('B', jl, jr, Am)
        wp = PtrVector(w, N)
        @inbounds for i in 1:N; wp[i] = wa[i]; end
        if jr == 'V'
            VRp = PtrMatrix(vr, N, N, Int(unsafe_load(ldvr)))
            @inbounds for j in 1:N, i in 1:N; VRp[i, j] = VRm[i, j]; end
        end
        unsafe_store!(info, Int64(0)); return
    end
end

# REAL {s,d}geevx_ — expert driver Julia's eigen!/eigvals! actually call (sense='N'). Adds balanc/sense
# chars + ilo/ihi/scale/abnrm outputs; rconde/rcondv (condition numbers) are only computed for sense∈
# {E,V,B} which Julia never requests — left untouched here. Same _geev_run! core, balanc passed through.
# (balanc, jobvl, jobvr, sense, n, A, lda, wr, wi, vl, ldvl, vr, ldvr, ilo, ihi, scale, abnrm, rconde,
#  rcondv, work, lwork, iwork, info, +4 lens)
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "geevx_64_"))(balanc::Ptr{UInt8}, jobvl::Ptr{UInt8},
            jobvr::Ptr{UInt8}, sense::Ptr{UInt8}, n::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64},
            wr::Ptr{$T}, wi::Ptr{$T}, vl::Ptr{$T}, ldvl::Ptr{Int64}, vr::Ptr{$T}, ldvr::Ptr{Int64},
            ilo::Ptr{Int64}, ihi::Ptr{Int64}, scale::Ptr{$T}, abnrm::Ptr{$T}, rconde::Ptr{$T},
            rcondv::Ptr{$T}, work::Ptr{$T}, lwork::Ptr{Int64}, iwork::Ptr{Int64}, info::Ptr{Int64},
            lb::Clong, ljl::Clong, ljr::Clong, ls::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        bl = _cabi_char(balanc); jl = _cabi_char(jobvl); jr = _cabi_char(jobvr)
        if jl != 'N'
            unsafe_store!(info, Int64(-2)); return
        end
        N = Int(unsafe_load(n)); ld = Int(unsafe_load(lda))
        Am = PtrMatrix(A, N, N, ld)
        wra, wia, _, VRm, il, ih, scl, anr = _geev_run!(bl, jl, jr, Am)
        wrp = PtrVector(wr, N); wip = PtrVector(wi, N)
        @inbounds for i in 1:N; wrp[i] = wra[i]; wip[i] = wia[i]; end
        if jr == 'V'
            VRp = PtrMatrix(vr, N, N, Int(unsafe_load(ldvr)))
            @inbounds for j in 1:N, i in 1:N; VRp[i, j] = VRm[i, j]; end
        end
        unsafe_store!(ilo, Int64(il)); unsafe_store!(ihi, Int64(ih))
        scp = PtrVector(scale, N); @inbounds for i in 1:N; scp[i] = scl[i]; end
        unsafe_store!(abnrm, $T(anr)); unsafe_store!(info, Int64(0)); return
    end
end

# COMPLEX {c,z}geevx_ — scale/abnrm/rconde/rcondv are REAL ($R); one w output; a REAL rwork block replaces
# real's integer iwork. (balanc, jobvl, jobvr, sense, n, A, lda, w, vl, ldvl, vr, ldvr, ilo, ihi, scale,
#  abnrm, rconde, rcondv, work, lwork, rwork, info, +4 lens)
for (p, T, R) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval Base.@ccallable function $(Symbol(p, "geevx_64_"))(balanc::Ptr{UInt8}, jobvl::Ptr{UInt8},
            jobvr::Ptr{UInt8}, sense::Ptr{UInt8}, n::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64},
            w::Ptr{$T}, vl::Ptr{$T}, ldvl::Ptr{Int64}, vr::Ptr{$T}, ldvr::Ptr{Int64}, ilo::Ptr{Int64},
            ihi::Ptr{Int64}, scale::Ptr{$R}, abnrm::Ptr{$R}, rconde::Ptr{$R}, rcondv::Ptr{$R},
            work::Ptr{$T}, lwork::Ptr{Int64}, rwork::Ptr{$R}, info::Ptr{Int64},
            lb::Clong, ljl::Clong, ljr::Clong, ls::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        bl = _cabi_char(balanc); jl = _cabi_char(jobvl); jr = _cabi_char(jobvr)
        if jl != 'N'
            unsafe_store!(info, Int64(-2)); return
        end
        N = Int(unsafe_load(n)); ld = Int(unsafe_load(lda))
        Am = PtrMatrix(A, N, N, ld)
        wa, _, VRm, il, ih, scl, anr = _geev_run!(bl, jl, jr, Am)
        wp = PtrVector(w, N); @inbounds for i in 1:N; wp[i] = wa[i]; end
        if jr == 'V'
            VRp = PtrMatrix(vr, N, N, Int(unsafe_load(ldvr)))
            @inbounds for j in 1:N, i in 1:N; VRp[i, j] = VRm[i, j]; end
        end
        unsafe_store!(ilo, Int64(il)); unsafe_store!(ihi, Int64(ih))
        scp = PtrVector(scale, N); @inbounds for i in 1:N; scp[i] = scl[i]; end
        unsafe_store!(abnrm, $R(anr)); unsafe_store!(info, Int64(0)); return
    end
end

# ── gees: Schur decomposition (Schur form + Schur vectors). Routes schur(A) (Julia calls gees!('V',A)). ──
# `sort`/`select`/`bwork` (eigenvalue sorting) are not supported — sort is always 'N' from Julia, so
# select/bwork are ignored (Ptr{Cvoid}) and sdim=0. _gees_run! uses permute-only balance so Z stays
# orthogonal. REAL {s,d}gees_(jobvs, sort, select, n, A, lda, sdim, wr, wi, vs, ldvs, work, lwork, bwork,
# info, +2 lens).
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "gees_64_"))(jobvs::Ptr{UInt8}, sort::Ptr{UInt8},
            select::Ptr{Cvoid}, n::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, sdim::Ptr{Int64},
            wr::Ptr{$T}, wi::Ptr{$T}, vs::Ptr{$T}, ldvs::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64},
            bwork::Ptr{Cvoid}, info::Ptr{Int64}, ljvs::Clong, ls::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        jvs = _cabi_char(jobvs)
        N = Int(unsafe_load(n)); ld = Int(unsafe_load(lda))
        Am = PtrMatrix(A, N, N, ld)
        wra, wia, VSm = _gees_run!(jvs, Am)
        wrp = PtrVector(wr, N); wip = PtrVector(wi, N)
        @inbounds for i in 1:N; wrp[i] = wra[i]; wip[i] = wia[i]; end
        if jvs == 'V'
            VSp = PtrMatrix(vs, N, N, Int(unsafe_load(ldvs)))
            @inbounds for j in 1:N, i in 1:N; VSp[i, j] = VSm[i, j]; end
        end
        unsafe_store!(sdim, Int64(0)); unsafe_store!(info, Int64(0)); return
    end
end

# COMPLEX {c,z}gees_(jobvs, sort, select, n, A, lda, sdim, w, vs, ldvs, work, lwork, rwork, bwork, info,
# +2 lens). One w output + a REAL rwork block (vs real's wr/wi).
for (p, T, R) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval Base.@ccallable function $(Symbol(p, "gees_64_"))(jobvs::Ptr{UInt8}, sort::Ptr{UInt8},
            select::Ptr{Cvoid}, n::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, sdim::Ptr{Int64},
            w::Ptr{$T}, vs::Ptr{$T}, ldvs::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64},
            rwork::Ptr{$R}, bwork::Ptr{Cvoid}, info::Ptr{Int64}, ljvs::Clong, ls::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        jvs = _cabi_char(jobvs)
        N = Int(unsafe_load(n)); ld = Int(unsafe_load(lda))
        Am = PtrMatrix(A, N, N, ld)
        wa, VSm = _gees_run!(jvs, Am)
        wp = PtrVector(w, N); @inbounds for i in 1:N; wp[i] = wa[i]; end
        if jvs == 'V'
            VSp = PtrMatrix(vs, N, N, Int(unsafe_load(ldvs)))
            @inbounds for j in 1:N, i in 1:N; VSp[i, j] = VSm[i, j]; end
        end
        unsafe_store!(sdim, Int64(0)); unsafe_store!(info, Int64(0)); return
    end
end

# ══════════════════════════════════════════════════════════════════════════════════════════════════
# Assembly batch: generalized eigen (sygvd/hegvd, ggev/ggev3, gges/gges3), tridiagonal (gtsv/gttrf/
# gttrs), tridiagonal-eigen (stev/stegr), band/packed Cholesky (pbtrf/pbtrs, pptrf/pptrs). Sigs
# cross-checked vs LinearAlgebra/lapack.jl ccalls (arg order, rwork for complex, hidden Clong count).
# ══════════════════════════════════════════════════════════════════════════════════════════════════

# ── sygvd / hegvd: generalized symmetric/Hermitian-definite eigensolver — routes eigen(Sym,Sym) ───────
# REAL {s,d}sygvd_(itype, jobz, uplo, n, A, lda, B, ldb, w, work, lwork, iwork, liwork, info, l_jobz,
# l_uplo) — 2 chars. w REAL. B not-PD → PosDefException caught → info>0. Honors lwork==-1 (work/iwork=1).
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "sygvd_64_"))(itype::Ptr{Int64}, jobz::Ptr{UInt8},
            uplo::Ptr{UInt8}, n::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, B::Ptr{$T}, ldb::Ptr{Int64},
            w::Ptr{$T}, work::Ptr{$T}, lwork::Ptr{Int64}, iwork::Ptr{Int64}, liwork::Ptr{Int64},
            info::Ptr{Int64}, l_jobz::Clong, l_uplo::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(iwork, Int64(1)); unsafe_store!(info, Int64(0)); return
        end
        it = Int(unsafe_load(itype)); jz = _cabi_char(jobz); ul = _cabi_char(uplo); N = Int(unsafe_load(n))
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda))); Bm = PtrMatrix(B, N, N, Int(unsafe_load(ldb)))
        wm = PtrVector(w, N)
        try
            wv = sygvd!(it, jz, ul, Am, Bm)[1]
            @inbounds for i in 1:N; wm[i] = wv[i]; end
            unsafe_store!(info, Int64(0))
        catch e
            e isa PosDefException || rethrow()
            unsafe_store!(info, Int64(N + e.info))     # LAPACK: info>n signals B-Cholesky failure
        end
        return
    end
end
# COMPLEX {c,z}hegvd_(itype, jobz, uplo, n, A, lda, B, ldb, w, work, lwork, rwork, lrwork, iwork, liwork,
# info, l_jobz, l_uplo) — 2 chars; w/rwork REAL ($Tr), A/B/work COMPLEX.
for (p, Tc, Tr) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval Base.@ccallable function $(Symbol(p, "hegvd_64_"))(itype::Ptr{Int64}, jobz::Ptr{UInt8},
            uplo::Ptr{UInt8}, n::Ptr{Int64}, A::Ptr{$Tc}, lda::Ptr{Int64}, B::Ptr{$Tc}, ldb::Ptr{Int64},
            w::Ptr{$Tr}, work::Ptr{$Tc}, lwork::Ptr{Int64}, rwork::Ptr{$Tr}, lrwork::Ptr{Int64},
            iwork::Ptr{Int64}, liwork::Ptr{Int64}, info::Ptr{Int64}, l_jobz::Clong, l_uplo::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($Tc)); unsafe_store!(rwork, one($Tr))
            unsafe_store!(iwork, Int64(1)); unsafe_store!(info, Int64(0)); return
        end
        it = Int(unsafe_load(itype)); jz = _cabi_char(jobz); ul = _cabi_char(uplo); N = Int(unsafe_load(n))
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda))); Bm = PtrMatrix(B, N, N, Int(unsafe_load(ldb)))
        wm = PtrVector(w, N)
        try
            wv = hegvd!(it, jz, ul, Am, Bm)[1]
            @inbounds for i in 1:N; wm[i] = $Tr(wv[i]); end
            unsafe_store!(info, Int64(0))
        catch e
            e isa PosDefException || rethrow()
            unsafe_store!(info, Int64(N + e.info))
        end
        return
    end
end

# ── gtsv / gttrf / gttrs: tridiagonal solve/factor — {s,d,c,z}, 0-char (gttrs 1-char). ────────────────
# gtsv_(n, nrhs, dl, d, du, B, ldb, info). SingularException → info>0.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "gtsv_64_"))(n::Ptr{Int64}, nrhs::Ptr{Int64},
            dl::Ptr{$T}, d::Ptr{$T}, du::Ptr{$T}, B::Ptr{$T}, ldb::Ptr{Int64}, info::Ptr{Int64})::Cvoid
        N = Int(unsafe_load(n)); R = Int(unsafe_load(nrhs))
        dlv = PtrVector(dl, max(N - 1, 0)); dv = PtrVector(d, N); duv = PtrVector(du, max(N - 1, 0))
        Bm = PtrMatrix(B, N, R, Int(unsafe_load(ldb)))
        try
            gtsv!(dlv, dv, duv, Bm); unsafe_store!(info, Int64(0))
        catch e
            e isa SingularException || rethrow()
            unsafe_store!(info, Int64(e.info))
        end
        return
    end
end
# gttrf_(n, dl, d, du, du2, ipiv, info).
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "gttrf_64_"))(n::Ptr{Int64}, dl::Ptr{$T}, d::Ptr{$T},
            du::Ptr{$T}, du2::Ptr{$T}, ipiv::Ptr{Int64}, info::Ptr{Int64})::Cvoid
        N = Int(unsafe_load(n))
        dlv = PtrVector(dl, max(N - 1, 0)); dv = PtrVector(d, N); duv = PtrVector(du, max(N - 1, 0))
        du2v = PtrVector(du2, max(N - 2, 0)); ipv = PtrVector(ipiv, N)
        try
            gttrf!(dlv, dv, duv, du2v, ipv); unsafe_store!(info, Int64(0))
        catch e
            e isa SingularException || rethrow()
            unsafe_store!(info, Int64(e.info))
        end
        return
    end
end
# gttrs_(trans, n, nrhs, dl, d, du, du2, ipiv, B, ldb, info, l_trans) — 1 char.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "gttrs_64_"))(trans::Ptr{UInt8}, n::Ptr{Int64},
            nrhs::Ptr{Int64}, dl::Ptr{$T}, d::Ptr{$T}, du::Ptr{$T}, du2::Ptr{$T}, ipiv::Ptr{Int64},
            B::Ptr{$T}, ldb::Ptr{Int64}, info::Ptr{Int64}, l_trans::Clong)::Cvoid
        N = Int(unsafe_load(n)); R = Int(unsafe_load(nrhs))
        dlv = PtrVector(dl, max(N - 1, 0)); dv = PtrVector(d, N); duv = PtrVector(du, max(N - 1, 0))
        du2v = PtrVector(du2, max(N - 2, 0)); ipv = PtrVector(ipiv, N)
        Bm = PtrMatrix(B, N, R, Int(unsafe_load(ldb)))
        gttrs!(_cabi_char(trans), dlv, dv, duv, du2v, ipv, Bm)
        unsafe_store!(info, Int64(0)); return
    end
end

# ── stev / stegr: symmetric-tridiagonal eigensolver (real s/d only) — routes eigen/eigvals(SymTridiagonal).
# stev_(job, n, dv, ev, Z, ldz, work, info, l_job) — 1 char. job='N' values (dv←w); 'V' + vectors→Z.
# Wraps _sterf! (values) / _steqr!('I') (values+vectors); operates on COPIES so the kernels' destruction
# stays internal (dv is overwritten with eigenvalues, matching LAPACK).
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "stev_64_"))(job::Ptr{UInt8}, n::Ptr{Int64}, dv::Ptr{$T},
            ev::Ptr{$T}, Z::Ptr{$T}, ldz::Ptr{Int64}, work::Ptr{$T}, info::Ptr{Int64}, l_job::Clong)::Cvoid
        jb = _cabi_char(job); N = Int(unsafe_load(n))
        dvv = PtrVector(dv, N); evv = PtrVector(ev, max(N - 1, 0))
        d = Vector{$T}(undef, N); e = Vector{$T}(undef, max(N - 1, 0))
        @inbounds for i in 1:N; d[i] = dvv[i]; end
        @inbounds for i in 1:N-1; e[i] = evv[i]; end
        if jb == 'V'
            Zm = PtrMatrix(Z, N, N, Int(unsafe_load(ldz)))
            Zw = zeros($T, N, N); @inbounds for i in 1:N; Zw[i, i] = one($T); end   # I (steqr('I') skips init at n=1)
            _steqr!('I', d, e, Zw)
            @inbounds for j in 1:N, i in 1:N; Zm[i, j] = Zw[i, j]; end
        else
            _sterf!(d, e)
        end
        @inbounds for i in 1:N; dvv[i] = d[i]; end
        unsafe_store!(info, Int64(0)); return
    end
end
# stegr_(jobz, range, n, dv, ev, vl, vu, il, iu, abstol, m, w, Z, ldz, isuppz, work, lwork, iwork,
# liwork, info, l_jobz, l_range) — 2 chars. Compute the FULL spectrum (+vectors) then slice by range
# (mirrors syevr). range 'A' all, 'I' il:iu, 'V' half-open (vl,vu]. abstol ignored. Honors lwork==-1.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "stegr_64_"))(jobz::Ptr{UInt8}, range::Ptr{UInt8},
            n::Ptr{Int64}, dv::Ptr{$T}, ev::Ptr{$T}, vl::Ptr{$T}, vu::Ptr{$T}, il::Ptr{Int64},
            iu::Ptr{Int64}, abstol::Ptr{$T}, m::Ptr{Int64}, w::Ptr{$T}, Z::Ptr{$T}, ldz::Ptr{Int64},
            isuppz::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64}, iwork::Ptr{Int64}, liwork::Ptr{Int64},
            info::Ptr{Int64}, l_jobz::Clong, l_range::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(iwork, Int64(1)); unsafe_store!(info, Int64(0)); return
        end
        jz = _cabi_char(jobz); rg = _cabi_char(range); N = Int(unsafe_load(n)); wantz = jz == 'V'
        dvv = PtrVector(dv, N); evv = PtrVector(ev, max(N - 1, 0))
        d = Vector{$T}(undef, N); e = Vector{$T}(undef, max(N - 1, 0))
        @inbounds for i in 1:N; d[i] = dvv[i]; end
        @inbounds for i in 1:N-1; e[i] = evv[i]; end
        local Zw::Matrix{$T}
        if wantz
            Zw = zeros($T, N, N); @inbounds for i in 1:N; Zw[i, i] = one($T); end   # I (steqr('I') skips init at n=1)
            _steqr!('I', d, e, Zw)
        else
            _sterf!(d, e); Zw = Matrix{$T}(undef, 0, 0)
        end
        lo = 1; hi = N
        if rg == 'I'
            lo = Int(unsafe_load(il)); hi = Int(unsafe_load(iu))
        elseif rg == 'V'
            vlo = unsafe_load(vl); vhi = unsafe_load(vu); lo = N + 1; hi = 0
            @inbounds for i in 1:N
                if vlo < d[i] <= vhi; lo = min(lo, i); hi = max(hi, i); end
            end
        end
        M = max(hi - lo + 1, 0); unsafe_store!(m, Int64(M))
        wm = PtrVector(w, max(M, 0))
        @inbounds for k in 1:M; wm[k] = d[lo + k - 1]; end
        if wantz && M > 0
            Zm = PtrMatrix(Z, N, M, Int(unsafe_load(ldz)))
            @inbounds for k in 1:M, i in 1:N; Zm[i, k] = Zw[i, lo + k - 1]; end
        end
        ipz = PtrVector(isuppz, 2 * max(M, 0))
        @inbounds for k in 1:M; ipz[2k - 1] = Int64(1); ipz[2k] = Int64(N); end
        unsafe_store!(info, Int64(0)); return
    end
end

# ── pbtrf / pbtrs: band Cholesky (real+complex Hermitian). pbtrf_(uplo, n, kd, AB, ldab, info, l_uplo).
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "pbtrf_64_"))(uplo::Ptr{UInt8}, n::Ptr{Int64},
            kd::Ptr{Int64}, AB::Ptr{$T}, ldab::Ptr{Int64}, info::Ptr{Int64}, l_uplo::Clong)::Cvoid
        ul = _cabi_char(uplo); N = Int(unsafe_load(n)); KD = Int(unsafe_load(kd))
        ABm = PtrMatrix(AB, KD + 1, N, Int(unsafe_load(ldab)))
        try
            pbtrf!(ABm; uplo = ul, kd = KD); unsafe_store!(info, Int64(0))
        catch e
            e isa PosDefException || rethrow()
            unsafe_store!(info, Int64(e.info))
        end
        return
    end
end
# pbtrs_(uplo, n, kd, nrhs, AB, ldab, B, ldb, info, l_uplo).
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "pbtrs_64_"))(uplo::Ptr{UInt8}, n::Ptr{Int64},
            kd::Ptr{Int64}, nrhs::Ptr{Int64}, AB::Ptr{$T}, ldab::Ptr{Int64}, B::Ptr{$T}, ldb::Ptr{Int64},
            info::Ptr{Int64}, l_uplo::Clong)::Cvoid
        ul = _cabi_char(uplo); N = Int(unsafe_load(n)); KD = Int(unsafe_load(kd)); R = Int(unsafe_load(nrhs))
        ABm = PtrMatrix(AB, KD + 1, N, Int(unsafe_load(ldab))); Bm = PtrMatrix(B, N, R, Int(unsafe_load(ldb)))
        pbtrs!(ABm, Bm; uplo = ul, kd = KD)
        unsafe_store!(info, Int64(0)); return
    end
end
# ── pptrf / pptrs: packed Cholesky. pptrf_(uplo, n, AP, info, l_uplo); pptrs_(uplo, n, nrhs, AP, B, ldb,
# info, l_uplo). AP is the packed-triangle vector, length n(n+1)/2.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "pptrf_64_"))(uplo::Ptr{UInt8}, n::Ptr{Int64},
            AP::Ptr{$T}, info::Ptr{Int64}, l_uplo::Clong)::Cvoid
        ul = _cabi_char(uplo); N = Int(unsafe_load(n))
        APv = PtrVector(AP, (N * (N + 1)) >> 1)
        try
            pptrf!(APv; uplo = ul); unsafe_store!(info, Int64(0))
        catch e
            e isa PosDefException || rethrow()
            unsafe_store!(info, Int64(e.info))
        end
        return
    end
end
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "pptrs_64_"))(uplo::Ptr{UInt8}, n::Ptr{Int64},
            nrhs::Ptr{Int64}, AP::Ptr{$T}, B::Ptr{$T}, ldb::Ptr{Int64}, info::Ptr{Int64}, l_uplo::Clong)::Cvoid
        ul = _cabi_char(uplo); N = Int(unsafe_load(n)); R = Int(unsafe_load(nrhs))
        APv = PtrVector(AP, (N * (N + 1)) >> 1); Bm = PtrMatrix(B, N, R, Int(unsafe_load(ldb)))
        pptrs!(APv, Bm; uplo = ul)
        unsafe_store!(info, Int64(0)); return
    end
end

# ── ggev / ggev3: generalized eigensolver — routes eigen(A,B)/eigvals(A,B). ggev3 (blocked Hessenberg-
# triangular) has the IDENTICAL reference ABI to ggev; Julia (LAPACK≥3.6) calls ggev3, so both names are
# generated from one shared core. jobvl='V' unsupported (info<0). REAL: alphar/alphai/beta; COMPLEX: alpha
# + a REAL rwork block. _ggev_run! mutates A,B (→ Schur form), returns freshly-allocated VR.
@inline function _ggev_cabi_real!(::Type{T}, jl::Char, jr::Char, N::Int, A::Ptr{T}, lda::Int,
        B::Ptr{T}, ldb::Int, alphar::Ptr{T}, alphai::Ptr{T}, beta::Ptr{T}, vr::Ptr{T}, ldvr::Int,
        info::Ptr{Int64}) where {T<:Real}
    if jl != 'N'; unsafe_store!(info, Int64(-1)); return; end
    Am = PtrMatrix(A, N, N, lda); Bm = PtrMatrix(B, N, N, ldb)
    arr, aii, bee, VR = _ggev_run!(jl, jr, Am, Bm)
    arp = PtrVector(alphar, N); aip = PtrVector(alphai, N); bep = PtrVector(beta, N)
    @inbounds for i in 1:N; arp[i] = arr[i]; aip[i] = aii[i]; bep[i] = bee[i]; end
    if jr == 'V'
        vrp = PtrMatrix(vr, N, N, ldvr)
        @inbounds for j in 1:N, i in 1:N; vrp[i, j] = VR[i, j]; end
    end
    unsafe_store!(info, Int64(0)); return
end
@inline function _ggev_cabi_cmplx!(::Type{T}, ::Type{R}, jl::Char, jr::Char, N::Int, A::Ptr{T}, lda::Int,
        B::Ptr{T}, ldb::Int, alpha::Ptr{T}, beta::Ptr{T}, vr::Ptr{T}, ldvr::Int, info::Ptr{Int64}) where {T<:Complex,R}
    if jl != 'N'; unsafe_store!(info, Int64(-1)); return; end
    Am = PtrMatrix(A, N, N, lda); Bm = PtrMatrix(B, N, N, ldb)
    al, bee, VR = _ggev_run!(jl, jr, Am, Bm)
    alp = PtrVector(alpha, N); bep = PtrVector(beta, N)
    @inbounds for i in 1:N; alp[i] = al[i]; bep[i] = bee[i]; end
    if jr == 'V'
        vrp = PtrMatrix(vr, N, N, ldvr)
        @inbounds for j in 1:N, i in 1:N; vrp[i, j] = VR[i, j]; end
    end
    unsafe_store!(info, Int64(0)); return
end
for (p, T) in (("s", Float32), ("d", Float64)), nm in ("ggev", "ggev3")
    @eval Base.@ccallable function $(Symbol(p, nm, "_64_"))(jobvl::Ptr{UInt8}, jobvr::Ptr{UInt8},
            n::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, B::Ptr{$T}, ldb::Ptr{Int64}, alphar::Ptr{$T},
            alphai::Ptr{$T}, beta::Ptr{$T}, vl::Ptr{$T}, ldvl::Ptr{Int64}, vr::Ptr{$T}, ldvr::Ptr{Int64},
            work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64}, ljl::Clong, ljr::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        _ggev_cabi_real!($T, _cabi_char(jobvl), _cabi_char(jobvr), Int(unsafe_load(n)), A,
            Int(unsafe_load(lda)), B, Int(unsafe_load(ldb)), alphar, alphai, beta, vr,
            Int(unsafe_load(ldvr)), info)
        return
    end
end
for (p, T, R) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64)), nm in ("ggev", "ggev3")
    @eval Base.@ccallable function $(Symbol(p, nm, "_64_"))(jobvl::Ptr{UInt8}, jobvr::Ptr{UInt8},
            n::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, B::Ptr{$T}, ldb::Ptr{Int64}, alpha::Ptr{$T},
            beta::Ptr{$T}, vl::Ptr{$T}, ldvl::Ptr{Int64}, vr::Ptr{$T}, ldvr::Ptr{Int64}, work::Ptr{$T},
            lwork::Ptr{Int64}, rwork::Ptr{$R}, info::Ptr{Int64}, ljl::Clong, ljr::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        _ggev_cabi_cmplx!($T, $R, _cabi_char(jobvl), _cabi_char(jobvr), Int(unsafe_load(n)), A,
            Int(unsafe_load(lda)), B, Int(unsafe_load(ldb)), alpha, beta, vr, Int(unsafe_load(ldvr)), info)
        return
    end
end

# ── gges / gges3: generalized Schur — routes schur(A,B). select/bwork ignored (sort='N', sdim=0). Both
# names share one core (identical ABI; Julia≥3.6 calls gges3). REAL: alphar/alphai; COMPLEX: alpha + rwork.
@inline function _gges_cabi_real!(::Type{T}, jvsl::Char, jvsr::Char, N::Int, A::Ptr{T}, lda::Int,
        B::Ptr{T}, ldb::Int, alphar::Ptr{T}, alphai::Ptr{T}, beta::Ptr{T}, vsl::Ptr{T}, ldvsl::Int,
        vsr::Ptr{T}, ldvsr::Int, sdim::Ptr{Int64}, info::Ptr{Int64}) where {T<:Real}
    Am = PtrMatrix(A, N, N, lda); Bm = PtrMatrix(B, N, N, ldb)
    _, _, alC, bee, VSL, VSR = _gges_run!(Am, Bm)
    arp = PtrVector(alphar, N); aip = PtrVector(alphai, N); bep = PtrVector(beta, N)
    @inbounds for i in 1:N; arp[i] = real(alC[i]); aip[i] = imag(alC[i]); bep[i] = bee[i]; end
    if jvsl == 'V'
        m = PtrMatrix(vsl, N, N, ldvsl); @inbounds for j in 1:N, i in 1:N; m[i, j] = VSL[i, j]; end
    end
    if jvsr == 'V'
        m = PtrMatrix(vsr, N, N, ldvsr); @inbounds for j in 1:N, i in 1:N; m[i, j] = VSR[i, j]; end
    end
    unsafe_store!(sdim, Int64(0)); unsafe_store!(info, Int64(0)); return
end
@inline function _gges_cabi_cmplx!(::Type{T}, jvsl::Char, jvsr::Char, N::Int, A::Ptr{T}, lda::Int,
        B::Ptr{T}, ldb::Int, alpha::Ptr{T}, beta::Ptr{T}, vsl::Ptr{T}, ldvsl::Int, vsr::Ptr{T},
        ldvsr::Int, sdim::Ptr{Int64}, info::Ptr{Int64}) where {T<:Complex}
    Am = PtrMatrix(A, N, N, lda); Bm = PtrMatrix(B, N, N, ldb)
    _, _, al, bee, VSL, VSR = _gges_run!(Am, Bm)
    alp = PtrVector(alpha, N); bep = PtrVector(beta, N)
    @inbounds for i in 1:N; alp[i] = al[i]; bep[i] = bee[i]; end
    if jvsl == 'V'
        m = PtrMatrix(vsl, N, N, ldvsl); @inbounds for j in 1:N, i in 1:N; m[i, j] = VSL[i, j]; end
    end
    if jvsr == 'V'
        m = PtrMatrix(vsr, N, N, ldvsr); @inbounds for j in 1:N, i in 1:N; m[i, j] = VSR[i, j]; end
    end
    unsafe_store!(sdim, Int64(0)); unsafe_store!(info, Int64(0)); return
end
for (p, T) in (("s", Float32), ("d", Float64)), nm in ("gges", "gges3")
    @eval Base.@ccallable function $(Symbol(p, nm, "_64_"))(jobvsl::Ptr{UInt8}, jobvsr::Ptr{UInt8},
            sort::Ptr{UInt8}, select::Ptr{Cvoid}, n::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, B::Ptr{$T},
            ldb::Ptr{Int64}, sdim::Ptr{Int64}, alphar::Ptr{$T}, alphai::Ptr{$T}, beta::Ptr{$T},
            vsl::Ptr{$T}, ldvsl::Ptr{Int64}, vsr::Ptr{$T}, ldvsr::Ptr{Int64}, work::Ptr{$T},
            lwork::Ptr{Int64}, bwork::Ptr{Cvoid}, info::Ptr{Int64}, ljl::Clong, ljr::Clong, ls::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        _gges_cabi_real!($T, _cabi_char(jobvsl), _cabi_char(jobvsr), Int(unsafe_load(n)), A,
            Int(unsafe_load(lda)), B, Int(unsafe_load(ldb)), alphar, alphai, beta, vsl,
            Int(unsafe_load(ldvsl)), vsr, Int(unsafe_load(ldvsr)), sdim, info)
        return
    end
end
for (p, T, R) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64)), nm in ("gges", "gges3")
    @eval Base.@ccallable function $(Symbol(p, nm, "_64_"))(jobvsl::Ptr{UInt8}, jobvsr::Ptr{UInt8},
            sort::Ptr{UInt8}, select::Ptr{Cvoid}, n::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, B::Ptr{$T},
            ldb::Ptr{Int64}, sdim::Ptr{Int64}, alpha::Ptr{$T}, beta::Ptr{$T}, vsl::Ptr{$T},
            ldvsl::Ptr{Int64}, vsr::Ptr{$T}, ldvsr::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64},
            rwork::Ptr{$R}, bwork::Ptr{Cvoid}, info::Ptr{Int64}, ljl::Clong, ljr::Clong, ls::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        _gges_cabi_cmplx!($T, _cabi_char(jobvsl), _cabi_char(jobvsr), Int(unsafe_load(n)), A,
            Int(unsafe_load(lda)), B, Int(unsafe_load(ldb)), alpha, beta, vsl, Int(unsafe_load(ldvsl)),
            vsr, Int(unsafe_load(ldvsr)), sdim, info)
        return
    end
end

# ══════════════════════════════════════════════════════════════════════════════════════════════════
# Assembly batch 2: banded LU, SPD tridiagonal, sym-tridiag bisection/inverse-iteration, pivoted
# Cholesky, QL/RQ, RZ least-squares, SVD least-squares, symmetric-indefinite solve/inverse, Sylvester,
# Schur reorder, equality-constrained LS, generalized SVD (Float64). All kernels were ALREADY validated
# vs LAPACK to machine-eps (see sysv.jl/gbtrf.jl/pttrf.jl/stebz.jl/gelsd.jl/gelsy.jl/pstrf.jl/qlrq.jl/
# trsyl.jl/trsen.jl/gglse.jl/ggsvd.jl headers) — this batch is wiring only. Sigs cross-checked vs
# LinearAlgebra/lapack.jl ccalls (arg order, rwork for complex, hidden Clong count) — see per-routine notes.
# ══════════════════════════════════════════════════════════════════════════════════════════════════

# ── gbtrf / gbtrs: general banded LU with partial pivoting (real+complex) ───────────────────────────────
# {s,d,c,z}gbtrf_64_(m, n, kl, ku, AB, ldab, ipiv, info) — 0 chars. ipiv OUT (length min(m,n)); the
# kernel allocates its own ipiv internally (gbtrf!(kl,ku,m,AB)->(AB,ipiv,info)), copied to the caller's.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "gbtrf_64_"))(m::Ptr{Int64}, n::Ptr{Int64}, kl::Ptr{Int64},
            ku::Ptr{Int64}, AB::Ptr{$T}, ldab::Ptr{Int64}, ipiv::Ptr{Int64}, info::Ptr{Int64})::Cvoid
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); KL = Int(unsafe_load(kl)); KU = Int(unsafe_load(ku))
        LD = Int(unsafe_load(ldab))
        ABm = PtrMatrix(AB, LD, N, LD)
        _, ipv, inf = gbtrf!(KL, KU, M, ABm)
        ip = PtrVector(ipiv, min(M, N))
        @inbounds for i in eachindex(ipv); ip[i] = Int64(ipv[i]); end
        unsafe_store!(info, Int64(inf)); return
    end
end
# {s,d,c,z}gbtrs_64_(trans, n, kl, ku, nrhs, AB, ldab, ipiv, B, ldb, info, len_trans) — 1 char. The
# kernel's `m` parameter is a vestige of gbtrf!'s signature (unused in gbtrs! — the system is n×n).
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "gbtrs_64_"))(trans::Ptr{UInt8}, n::Ptr{Int64},
            kl::Ptr{Int64}, ku::Ptr{Int64}, nrhs::Ptr{Int64}, AB::Ptr{$T}, ldab::Ptr{Int64},
            ipiv::Ptr{Int64}, B::Ptr{$T}, ldb::Ptr{Int64}, info::Ptr{Int64}, lt::Clong)::Cvoid
        N = Int(unsafe_load(n)); KL = Int(unsafe_load(kl)); KU = Int(unsafe_load(ku)); R = Int(unsafe_load(nrhs))
        LD = Int(unsafe_load(ldab))
        ABm = PtrMatrix(AB, LD, N, LD)
        ip = PtrVector(ipiv, N)
        Bm = PtrMatrix(B, N, R, Int(unsafe_load(ldb)))
        gbtrs!(_cabi_char(trans), KL, KU, N, ABm, ip, Bm)
        unsafe_store!(info, Int64(0)); return
    end
end

# ── pttrf / pttrs / ptsv: SPD (real) / Hermitian-PD (complex) tridiagonal LDLᴴ ──────────────────────────
# ptsv_64_(n, nrhs, D, E, B, ldb, info) — 0 chars, ALL 4 types (reference LAPACK carries no uplo here;
# D is REAL even for complex T). SingularException-free (info>0 marks the first non-positive pivot).
for (p, T, Tr) in (("s", Float32, Float32), ("d", Float64, Float64),
                   ("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval Base.@ccallable function $(Symbol(p, "ptsv_64_"))(n::Ptr{Int64}, nrhs::Ptr{Int64}, D::Ptr{$Tr},
            E::Ptr{$T}, B::Ptr{$T}, ldb::Ptr{Int64}, info::Ptr{Int64})::Cvoid
        N = Int(unsafe_load(n)); R = Int(unsafe_load(nrhs))
        Dv = PtrVector(D, N); Ev = PtrVector(E, max(N - 1, 0))
        Bm = PtrMatrix(B, N, R, Int(unsafe_load(ldb)))
        _, _, _, inf = ptsv!(Dv, Ev, Bm)
        unsafe_store!(info, Int64(inf)); return
    end
end
# pttrf_64_(n, D, E, info) — 0 chars, ALL 4 types.
for (p, T, Tr) in (("s", Float32, Float32), ("d", Float64, Float64),
                   ("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval Base.@ccallable function $(Symbol(p, "pttrf_64_"))(n::Ptr{Int64}, D::Ptr{$Tr}, E::Ptr{$T},
            info::Ptr{Int64})::Cvoid
        N = Int(unsafe_load(n))
        Dv = PtrVector(D, N); Ev = PtrVector(E, max(N - 1, 0))
        _, _, inf = pttrf!(Dv, Ev)
        unsafe_store!(info, Int64(inf)); return
    end
end
# pttrs_64_ REAL (s,d): (n, nrhs, D, E, B, ldb, info) — 0 chars (no uplo; symmetric case).
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "pttrs_64_"))(n::Ptr{Int64}, nrhs::Ptr{Int64}, D::Ptr{$T},
            E::Ptr{$T}, B::Ptr{$T}, ldb::Ptr{Int64}, info::Ptr{Int64})::Cvoid
        N = Int(unsafe_load(n)); R = Int(unsafe_load(nrhs))
        Dv = PtrVector(D, N); Ev = PtrVector(E, max(N - 1, 0))
        Bm = PtrMatrix(B, N, R, Int(unsafe_load(ldb)))
        pttrs!(Dv, Ev, Bm)
        unsafe_store!(info, Int64(0)); return
    end
end
# pttrs_64_ COMPLEX (c,z): (uplo, n, nrhs, D, E, B, ldb, info, len_uplo) — 1 char (Hermitian sub/super).
for (p, T, Tr) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval Base.@ccallable function $(Symbol(p, "pttrs_64_"))(uplo::Ptr{UInt8}, n::Ptr{Int64},
            nrhs::Ptr{Int64}, D::Ptr{$Tr}, E::Ptr{$T}, B::Ptr{$T}, ldb::Ptr{Int64}, info::Ptr{Int64},
            lu::Clong)::Cvoid
        N = Int(unsafe_load(n)); R = Int(unsafe_load(nrhs))
        Dv = PtrVector(D, N); Ev = PtrVector(E, max(N - 1, 0))
        Bm = PtrMatrix(B, N, R, Int(unsafe_load(ldb)))
        pttrs!(Dv, Ev, Bm; uplo = _cabi_char(uplo))
        unsafe_store!(info, Int64(0)); return
    end
end

# ── stebz / stein: real-symmetric-tridiagonal eigenvalues (bisection) / eigenvectors (inverse iteration)
# — real s/d only. work/iwork are FIXED-size (no lwork query for either routine).
# stebz_64_(range, order, n, vl, vu, il, iu, abstol, dv, ev, m, nsplit, w, iblock, isplit, work, iwork,
# info, len_range, len_order) — 2 chars.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "stebz_64_"))(range::Ptr{UInt8}, order::Ptr{UInt8},
            n::Ptr{Int64}, vl::Ptr{$T}, vu::Ptr{$T}, il::Ptr{Int64}, iu::Ptr{Int64}, abstol::Ptr{$T},
            dv::Ptr{$T}, ev::Ptr{$T}, m::Ptr{Int64}, nsplit::Ptr{Int64}, w::Ptr{$T}, iblock::Ptr{Int64},
            isplit::Ptr{Int64}, work::Ptr{$T}, iwork::Ptr{Int64}, info::Ptr{Int64},
            lr::Clong, lo::Clong)::Cvoid
        N = Int(unsafe_load(n))
        dv_ = PtrVector(dv, N); ev_ = PtrVector(ev, max(N - 1, 0))
        wv, ibv, isv, inf = stebz!(_cabi_char(range), _cabi_char(order), unsafe_load(vl), unsafe_load(vu),
            Int(unsafe_load(il)), Int(unsafe_load(iu)), unsafe_load(abstol), dv_, ev_)
        M = length(wv)
        wm = PtrVector(w, N); ibm = PtrVector(iblock, N)
        @inbounds for i in 1:M; wm[i] = wv[i]; ibm[i] = Int64(ibv[i]); end
        NS = length(isv)
        ism = PtrVector(isplit, N)
        @inbounds for i in 1:NS; ism[i] = Int64(isv[i]); end
        unsafe_store!(m, Int64(M)); unsafe_store!(nsplit, Int64(NS)); unsafe_store!(info, Int64(inf))
        return
    end
end
# stein_64_(n, dv, ev, m, w, iblock, isplit, z, ldz, work, iwork, ifail, info) — 0 chars. `ifail`
# (convergence-failure flags) has no analogue in the pure-Julia kernel (fixed maxits, no explicit
# failure signal) — always written zero; see the assembler report for this coverage note.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "stein_64_"))(n::Ptr{Int64}, dv::Ptr{$T}, ev::Ptr{$T},
            m::Ptr{Int64}, w::Ptr{$T}, iblock::Ptr{Int64}, isplit::Ptr{Int64}, z::Ptr{$T}, ldz::Ptr{Int64},
            work::Ptr{$T}, iwork::Ptr{Int64}, ifail::Ptr{Int64}, info::Ptr{Int64})::Cvoid
        N = Int(unsafe_load(n)); M = Int(unsafe_load(m))
        dv_ = PtrVector(dv, N); ev_ = PtrVector(ev, max(N - 1, 0))
        wv = Vector{$T}(undef, M); @inbounds for i in 1:M; wv[i] = unsafe_load(w, i); end
        ibp = PtrVector(iblock, M); ibv = Vector{Int}(undef, M)
        @inbounds for i in 1:M; ibv[i] = Int(ibp[i]); end
        isp = PtrVector(isplit, N); isv = Vector{Int}(undef, N)
        @inbounds for i in 1:N; isv[i] = Int(isp[i]); end
        Z = stein!(dv_, ev_, wv, ibv, isv)
        Zm = PtrMatrix(z, N, M, Int(unsafe_load(ldz)))
        @inbounds for j in 1:M, i in 1:N; Zm[i, j] = Z[i, j]; end
        ifp = PtrVector(ifail, max(M, 1)); @inbounds for i in 1:M; ifp[i] = Int64(0); end
        unsafe_store!(info, Int64(0)); return
    end
end

# ── pstrf: pivoted (rank-revealing) Cholesky of a positive-SEMIdefinite matrix (real+complex) ───────────
# {s,d,c,z}pstrf_64_(uplo, n, A, lda, piv, rank, tol, work, info, len_uplo) — 1 char, NO lwork query
# (work is a fixed 2n-length LAPACK scratch array; PureBLAS owns its own — ignored).
for (p, T, Tr) in (("s", Float32, Float32), ("d", Float64, Float64),
                   ("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval Base.@ccallable function $(Symbol(p, "pstrf_64_"))(uplo::Ptr{UInt8}, n::Ptr{Int64}, A::Ptr{$T},
            lda::Ptr{Int64}, piv::Ptr{Int64}, rank::Ptr{Int64}, tol::Ptr{$Tr}, work::Ptr{$Tr},
            info::Ptr{Int64}, lu::Clong)::Cvoid
        N = Int(unsafe_load(n))
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        pv = PtrVector(piv, N)
        _, _, rk, inf = pstrf!(Am, pv, unsafe_load(tol); uplo = _cabi_char(uplo))
        unsafe_store!(rank, Int64(rk)); unsafe_store!(info, Int64(inf)); return
    end
end

# ── geqlf / gerqf: QL/RQ factorizations (real+complex; τ ALREADY LAPACK convention, no faer inversion) ──
# {s,d,c,z}geqlf_64_(m, n, A, lda, tau, work, lwork, info) — 0 chars.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "geqlf_64_"))(m::Ptr{Int64}, n::Ptr{Int64}, A::Ptr{$T},
            lda::Ptr{Int64}, tau::Ptr{$T}, work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64})::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); k = min(M, N)
        geqlf!(PtrMatrix(A, M, N, Int(unsafe_load(lda))), PtrVector(tau, k))
        unsafe_store!(info, Int64(0)); return
    end
end
# {s,d,c,z}gerqf_64_(m, n, A, lda, tau, work, lwork, info) — 0 chars.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "gerqf_64_"))(m::Ptr{Int64}, n::Ptr{Int64}, A::Ptr{$T},
            lda::Ptr{Int64}, tau::Ptr{$T}, work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64})::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); k = min(M, N)
        gerqf!(PtrMatrix(A, M, N, Int(unsafe_load(lda))), PtrVector(tau, k))
        unsafe_store!(info, Int64(0)); return
    end
end
# orgql (real) / ungql (complex): (m, n, k, A, lda, tau, work, lwork, info) — 0 chars.
for (nm, T) in (("sorgql", Float32), ("dorgql", Float64), ("cungql", ComplexF32), ("zungql", ComplexF64))
    @eval Base.@ccallable function $(Symbol(nm, "_64_"))(m::Ptr{Int64}, n::Ptr{Int64}, k::Ptr{Int64},
            A::Ptr{$T}, lda::Ptr{Int64}, tau::Ptr{$T}, work::Ptr{$T}, lwork::Ptr{Int64},
            info::Ptr{Int64})::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); K = Int(unsafe_load(k))
        orgql!(PtrMatrix(A, M, N, Int(unsafe_load(lda))), PtrVector(tau, K))
        unsafe_store!(info, Int64(0)); return
    end
end
# orgrq (real) / ungrq (complex): (m, n, k, A, lda, tau, work, lwork, info) — 0 chars.
for (nm, T) in (("sorgrq", Float32), ("dorgrq", Float64), ("cungrq", ComplexF32), ("zungrq", ComplexF64))
    @eval Base.@ccallable function $(Symbol(nm, "_64_"))(m::Ptr{Int64}, n::Ptr{Int64}, k::Ptr{Int64},
            A::Ptr{$T}, lda::Ptr{Int64}, tau::Ptr{$T}, work::Ptr{$T}, lwork::Ptr{Int64},
            info::Ptr{Int64})::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); K = Int(unsafe_load(k))
        orgrq!(PtrMatrix(A, M, N, Int(unsafe_load(lda))), PtrVector(tau, K))
        unsafe_store!(info, Int64(0)); return
    end
end
# ormql (real) / unmql (complex): (side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info) — 2
# chars. A holds the geqlf COLUMN-reflector panel: nq rows (nq = m if side='L' else n) × k cols.
for (nm, T) in (("sormql", Float32), ("dormql", Float64), ("cunmql", ComplexF32), ("zunmql", ComplexF64))
    @eval Base.@ccallable function $(Symbol(nm, "_64_"))(side::Ptr{UInt8}, trans::Ptr{UInt8},
            m::Ptr{Int64}, n::Ptr{Int64}, k::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, tau::Ptr{$T},
            C::Ptr{$T}, ldc::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64},
            len_s::Clong, len_t::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        sd = _cabi_char(side); tr = _cabi_char(trans)
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); K = Int(unsafe_load(k))
        nq = sd == 'L' ? M : N
        Am = PtrMatrix(A, nq, K, Int(unsafe_load(lda)))
        Cm = PtrMatrix(C, M, N, Int(unsafe_load(ldc)))
        ormql!(sd, tr, Am, PtrVector(tau, K), Cm)
        unsafe_store!(info, Int64(0)); return
    end
end
# ormrq (real) / unmrq (complex): (side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info) — 2
# chars. A holds the gerqf ROW-reflector panel: k rows × nq cols (nq = m if side='L' else n).
for (nm, T) in (("sormrq", Float32), ("dormrq", Float64), ("cunmrq", ComplexF32), ("zunmrq", ComplexF64))
    @eval Base.@ccallable function $(Symbol(nm, "_64_"))(side::Ptr{UInt8}, trans::Ptr{UInt8},
            m::Ptr{Int64}, n::Ptr{Int64}, k::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, tau::Ptr{$T},
            C::Ptr{$T}, ldc::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64},
            len_s::Clong, len_t::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        sd = _cabi_char(side); tr = _cabi_char(trans)
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); K = Int(unsafe_load(k))
        nq = sd == 'L' ? M : N
        Am = PtrMatrix(A, K, nq, Int(unsafe_load(lda)))
        Cm = PtrMatrix(C, M, N, Int(unsafe_load(ldc)))
        ormrq!(sd, tr, Am, PtrVector(tau, K), Cm)
        unsafe_store!(info, Int64(0)); return
    end
end

# ── tzrzf: RZ (upper-trapezoidal → upper-triangular) reduction — the "complete orthogonal" half of gelsy
# {s,d,c,z}tzrzf_64_(m, n, A, lda, tau, work, lwork, info) — 0 chars. tau length m (m ≤ n required).
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "tzrzf_64_"))(m::Ptr{Int64}, n::Ptr{Int64}, A::Ptr{$T},
            lda::Ptr{Int64}, tau::Ptr{$T}, work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64})::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n))
        tzrzf!(PtrMatrix(A, M, N, Int(unsafe_load(lda))), PtrVector(tau, M))
        unsafe_store!(info, Int64(0)); return
    end
end
# ormrz (real) / unmrz (complex): (side, trans, m, n, k, l, A, lda, tau, C, ldc, work, lwork, info) — 2
# chars. A is k×(k+l) (the tzrzf RZ reflector factor); k, l are BOTH explicit ABI args (not inferred).
for (nm, T) in (("sormrz", Float32), ("dormrz", Float64), ("cunmrz", ComplexF32), ("zunmrz", ComplexF64))
    @eval Base.@ccallable function $(Symbol(nm, "_64_"))(side::Ptr{UInt8}, trans::Ptr{UInt8},
            m::Ptr{Int64}, n::Ptr{Int64}, k::Ptr{Int64}, l::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64},
            tau::Ptr{$T}, C::Ptr{$T}, ldc::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64},
            info::Ptr{Int64}, len_s::Clong, len_t::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        K = Int(unsafe_load(k)); L = Int(unsafe_load(l))
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n))
        Am = PtrMatrix(A, K, K + L, Int(unsafe_load(lda)))
        Cm = PtrMatrix(C, M, N, Int(unsafe_load(ldc)))
        ormrz!(_cabi_char(side), _cabi_char(trans), Am, PtrVector(tau, K), Cm)
        unsafe_store!(info, Int64(0)); return
    end
end

# ── gelsy: rank-deficient least squares via complete-orthogonal (pivoted-QR + RZ) factorization ────────
# REAL {s,d}gelsy_64_(m, n, nrhs, A, lda, B, ldb, jpvt, rcond, rank, work, lwork, info) — 0 chars.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "gelsy_64_"))(m::Ptr{Int64}, n::Ptr{Int64}, nrhs::Ptr{Int64},
            A::Ptr{$T}, lda::Ptr{Int64}, B::Ptr{$T}, ldb::Ptr{Int64}, jpvt::Ptr{Int64}, rcond::Ptr{$T},
            rank::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64})::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); Rh = Int(unsafe_load(nrhs))
        Am = PtrMatrix(A, M, N, Int(unsafe_load(lda)))
        Bm = PtrMatrix(B, max(M, N), Rh, Int(unsafe_load(ldb)))
        _, rk = gelsy!(Am, Bm, PtrVector(jpvt, N), unsafe_load(rcond))
        unsafe_store!(rank, Int64(rk)); unsafe_store!(info, Int64(0)); return
    end
end
# COMPLEX {c,z}gelsy_64_(m, n, nrhs, A, lda, B, ldb, jpvt, rcond, rank, work, lwork, rwork, info) — 0
# chars. rcond REAL ($R); rwork a real scratch block (ignored — PureBLAS owns its own).
for (p, T, R) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval Base.@ccallable function $(Symbol(p, "gelsy_64_"))(m::Ptr{Int64}, n::Ptr{Int64}, nrhs::Ptr{Int64},
            A::Ptr{$T}, lda::Ptr{Int64}, B::Ptr{$T}, ldb::Ptr{Int64}, jpvt::Ptr{Int64}, rcond::Ptr{$R},
            rank::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64}, rwork::Ptr{$R}, info::Ptr{Int64})::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); Rh = Int(unsafe_load(nrhs))
        Am = PtrMatrix(A, M, N, Int(unsafe_load(lda)))
        Bm = PtrMatrix(B, max(M, N), Rh, Int(unsafe_load(ldb)))
        _, rk = gelsy!(Am, Bm, PtrVector(jpvt, N), unsafe_load(rcond))
        unsafe_store!(rank, Int64(rk)); unsafe_store!(info, Int64(0)); return
    end
end

# ── gelsd: rank-deficient least squares via SVD (bidiagonalize → SVD → rcond-threshold → solve) ────────
# REAL {s,d}gelsd_64_(m, n, nrhs, A, lda, B, ldb, S, rcond, rank, work, lwork, iwork, info) — 0 chars.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "gelsd_64_"))(m::Ptr{Int64}, n::Ptr{Int64}, nrhs::Ptr{Int64},
            A::Ptr{$T}, lda::Ptr{Int64}, B::Ptr{$T}, ldb::Ptr{Int64}, S::Ptr{$T}, rcond::Ptr{$T},
            rank::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64}, iwork::Ptr{Int64}, info::Ptr{Int64})::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(iwork, Int64(1)); unsafe_store!(info, Int64(0)); return
        end
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); Rh = Int(unsafe_load(nrhs)); mn = min(M, N)
        Am = PtrMatrix(A, M, N, Int(unsafe_load(lda)))
        Bm = PtrMatrix(B, max(M, N), Rh, Int(unsafe_load(ldb)))
        _, rk, sv = gelsd!(Am, Bm, unsafe_load(rcond))
        Sm = PtrVector(S, mn); @inbounds for i in 1:mn; Sm[i] = sv[i]; end
        unsafe_store!(rank, Int64(rk)); unsafe_store!(info, Int64(0)); return
    end
end
# COMPLEX {c,z}gelsd_64_(m, n, nrhs, A, lda, B, ldb, S, rcond, rank, work, lwork, rwork, iwork, info) —
# 0 chars. S/rcond REAL ($R); rwork a real scratch block (ignored).
for (p, T, R) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval Base.@ccallable function $(Symbol(p, "gelsd_64_"))(m::Ptr{Int64}, n::Ptr{Int64}, nrhs::Ptr{Int64},
            A::Ptr{$T}, lda::Ptr{Int64}, B::Ptr{$T}, ldb::Ptr{Int64}, S::Ptr{$R}, rcond::Ptr{$R},
            rank::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64}, rwork::Ptr{$R}, iwork::Ptr{Int64},
            info::Ptr{Int64})::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(rwork, one($R))
            unsafe_store!(iwork, Int64(1)); unsafe_store!(info, Int64(0)); return
        end
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); Rh = Int(unsafe_load(nrhs)); mn = min(M, N)
        Am = PtrMatrix(A, M, N, Int(unsafe_load(lda)))
        Bm = PtrMatrix(B, max(M, N), Rh, Int(unsafe_load(ldb)))
        _, rk, sv = gelsd!(Am, Bm, unsafe_load(rcond))
        Sm = PtrVector(S, mn); @inbounds for i in 1:mn; Sm[i] = sv[i]; end
        unsafe_store!(rank, Int64(rk)); unsafe_store!(info, Int64(0)); return
    end
end

# ── sysv / hesv: one-shot symmetric-indefinite / Hermitian solve (Bunch-Kaufman factor + solve) ────────
# {s,d,c,z}sysv_64_(uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info, len_uplo) — 1 char. Composed
# from the already-wired sytrf!/sytrs! (bunchkaufman.jl) driven straight into the caller's ipiv buffer.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "sysv_64_"))(uplo::Ptr{UInt8}, n::Ptr{Int64},
            nrhs::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, ipiv::Ptr{Int64}, B::Ptr{$T}, ldb::Ptr{Int64},
            work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64}, lu::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        N = Int(unsafe_load(n)); R = Int(unsafe_load(nrhs)); ul = _cabi_char(uplo)
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        Bm = PtrMatrix(B, N, R, Int(unsafe_load(ldb)))
        ip = PtrVector(ipiv, N)
        sytrf!(Am, ip; uplo = ul)
        sytrs!(Am, ip, Bm; uplo = ul)
        unsafe_store!(info, Int64(0)); return
    end
end
# {c,z}hesv_64_ — Hermitian variant (complex only; real Hermitian solve routes through sysv above).
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "hesv_64_"))(uplo::Ptr{UInt8}, n::Ptr{Int64},
            nrhs::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, ipiv::Ptr{Int64}, B::Ptr{$T}, ldb::Ptr{Int64},
            work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64}, lu::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        N = Int(unsafe_load(n)); R = Int(unsafe_load(nrhs)); ul = _cabi_char(uplo)
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        Bm = PtrMatrix(B, N, R, Int(unsafe_load(ldb)))
        ip = PtrVector(ipiv, N)
        hetrf!(Am, ip; uplo = ul)
        hetrs!(Am, ip, Bm; uplo = ul)
        unsafe_store!(info, Int64(0)); return
    end
end
# ── sytri / hetri: symmetric-indefinite / Hermitian matrix inverse from Bunch-Kaufman factors ──────────
# {s,d,c,z}sytri_64_(uplo, n, A, lda, ipiv, work, info, len_uplo) — 1 char, NO lwork query. NOTE: the
# pure-Julia kernel never signals a singular D-block via info (unlike reference dsytri) — always 0.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "sytri_64_"))(uplo::Ptr{UInt8}, n::Ptr{Int64}, A::Ptr{$T},
            lda::Ptr{Int64}, ipiv::Ptr{Int64}, work::Ptr{$T}, info::Ptr{Int64}, lu::Clong)::Cvoid
        N = Int(unsafe_load(n))
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        sytri!(Am, PtrVector(ipiv, N); uplo = _cabi_char(uplo))
        unsafe_store!(info, Int64(0)); return
    end
end
# {c,z}hetri_64_ — Hermitian variant (complex only).
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "hetri_64_"))(uplo::Ptr{UInt8}, n::Ptr{Int64}, A::Ptr{$T},
            lda::Ptr{Int64}, ipiv::Ptr{Int64}, work::Ptr{$T}, info::Ptr{Int64}, lu::Clong)::Cvoid
        N = Int(unsafe_load(n))
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        hetri!(Am, PtrVector(ipiv, N); uplo = _cabi_char(uplo))
        unsafe_store!(info, Int64(0)); return
    end
end

# ── trsyl: triangular Sylvester solve — op(A)·X ± X·op(B) = scale·C (backs sylvester/lyap) ──────────────
# {s,d,c,z}trsyl_64_(transa, transb, isgn, m, n, A, lda, B, ldb, C, ldc, scale, info, len_ta, len_tb) —
# 2 chars, NO work array / lwork query (reference dtrsyl carries no workspace).
for (p, T, Tr) in (("s", Float32, Float32), ("d", Float64, Float64),
                   ("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval Base.@ccallable function $(Symbol(p, "trsyl_64_"))(transa::Ptr{UInt8}, transb::Ptr{UInt8},
            isgn::Ptr{Int64}, m::Ptr{Int64}, n::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, B::Ptr{$T},
            ldb::Ptr{Int64}, C::Ptr{$T}, ldc::Ptr{Int64}, scale::Ptr{$Tr}, info::Ptr{Int64},
            lta::Clong, ltb::Clong)::Cvoid
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); ig = Int(unsafe_load(isgn))
        Am = PtrMatrix(A, M, M, Int(unsafe_load(lda)))
        Bm = PtrMatrix(B, N, N, Int(unsafe_load(ldb)))
        Cm = PtrMatrix(C, M, N, Int(unsafe_load(ldc)))
        _, sc = trsyl!(_cabi_char(transa), _cabi_char(transb), ig, Am, Bm, Cm)
        unsafe_store!(scale, $Tr(sc)); unsafe_store!(info, Int64(0)); return
    end
end

# ── trexc / trsen: Schur reordering (backs ordschur) ────────────────────────────────────────────────────
# Uses the internal `_trexc_dispatch!`/`_trsen_dispatch!` (trsen.jl) directly rather than the public
# `trexc!`/`trsen!` wrappers — the public wrappers discard `info` and (for trexc) the block-snapped final
# `ilst`, both of which the C-ABI must report/write back (LAPACK OUTPUT semantics). Same module, same
# already-validated kernels — this is not a reimplementation.
#
# REAL {s,d}trexc_64_(compq, n, T, ldt, Q, ldq, ifst, ilst, work, info, len_compq) — 1 char.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "trexc_64_"))(compq::Ptr{UInt8}, n::Ptr{Int64},
            Tm_::Ptr{$T}, ldt::Ptr{Int64}, Q::Ptr{$T}, ldq::Ptr{Int64}, ifst::Ptr{Int64}, ilst::Ptr{Int64},
            work::Ptr{$T}, info::Ptr{Int64}, lc::Clong)::Cvoid
        N = Int(unsafe_load(n)); wantq = _cabi_char(compq) == 'V'
        Tm = PtrMatrix(Tm_, N, N, Int(unsafe_load(ldt)))
        Qm = PtrMatrix(Q, N, N, Int(unsafe_load(ldq)))
        inf, here = _trexc_dispatch!(wantq, Tm, Qm, Int(unsafe_load(ifst)), Int(unsafe_load(ilst)))
        unsafe_store!(ilst, Int64(here)); unsafe_store!(info, Int64(inf)); return
    end
end
# COMPLEX {c,z}trexc_64_(compq, n, T, ldt, Q, ldq, ifst, ilst, info, len_compq) — 1 char, NO work array.
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "trexc_64_"))(compq::Ptr{UInt8}, n::Ptr{Int64},
            Tm_::Ptr{$T}, ldt::Ptr{Int64}, Q::Ptr{$T}, ldq::Ptr{Int64}, ifst::Ptr{Int64}, ilst::Ptr{Int64},
            info::Ptr{Int64}, lc::Clong)::Cvoid
        N = Int(unsafe_load(n)); wantq = _cabi_char(compq) == 'V'
        Tm = PtrMatrix(Tm_, N, N, Int(unsafe_load(ldt)))
        Qm = PtrMatrix(Q, N, N, Int(unsafe_load(ldq)))
        inf, here = _trexc_dispatch!(wantq, Tm, Qm, Int(unsafe_load(ifst)), Int(unsafe_load(ilst)))
        unsafe_store!(ilst, Int64(here)); unsafe_store!(info, Int64(inf)); return
    end
end
# REAL {s,d}trsen_64_(job, compq, select, n, T, ldt, Q, ldq, wr, wi, m, s, sep, work, lwork, iwork,
# liwork, info, len_job, len_compq) — 2 chars. `select` is Ptr{Int64} (LAPACK LOGICAL-as-BlasInt), one
# entry per column (either half of a conjugate pair selects the whole 2×2 block, per trsen! semantics).
# `m` (selected-subspace dimension) is written as a best-effort popcount of `select` (the kernel doesn't
# separately expose the block-aware count; unused by Julia's own high-level callers either).
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "trsen_64_"))(job::Ptr{UInt8}, compq::Ptr{UInt8},
            select::Ptr{Int64}, n::Ptr{Int64}, Tm_::Ptr{$T}, ldt::Ptr{Int64}, Q::Ptr{$T}, ldq::Ptr{Int64},
            wr::Ptr{$T}, wi::Ptr{$T}, m::Ptr{Int64}, s::Ptr{$T}, sep::Ptr{$T}, work::Ptr{$T},
            lwork::Ptr{Int64}, iwork::Ptr{Int64}, liwork::Ptr{Int64}, info::Ptr{Int64},
            lj::Clong, lc::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(iwork, Int64(1)); unsafe_store!(info, Int64(0)); return
        end
        N = Int(unsafe_load(n)); jb = _cabi_char(job); wantq = _cabi_char(compq) == 'V'
        Tm = PtrMatrix(Tm_, N, N, Int(unsafe_load(ldt)))
        Qm = PtrMatrix(Q, N, N, Int(unsafe_load(ldq)))
        selp = PtrVector(select, N)
        sel = Bool[selp[i] != 0 for i in 1:N]
        _, _, w, sv, sepv, inf = _trsen_dispatch!(jb, wantq, sel, Tm, Qm)
        wrp = PtrVector(wr, N); wip = PtrVector(wi, N)
        @inbounds for i in 1:N; wrp[i] = real(w[i]); wip[i] = imag(w[i]); end
        unsafe_store!(s, $T(sv)); unsafe_store!(sep, $T(sepv))
        unsafe_store!(m, Int64(count(!iszero, sel))); unsafe_store!(info, Int64(inf)); return
    end
end
# COMPLEX {c,z}trsen_64_(job, compq, select, n, T, ldt, Q, ldq, w, m, s, sep, work, lwork, info, len_job,
# len_compq) — 2 chars. ONE w output (vs real's wr/wi); s/sep REAL ($R); NO iwork.
for (p, T, R) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval Base.@ccallable function $(Symbol(p, "trsen_64_"))(job::Ptr{UInt8}, compq::Ptr{UInt8},
            select::Ptr{Int64}, n::Ptr{Int64}, Tm_::Ptr{$T}, ldt::Ptr{Int64}, Q::Ptr{$T}, ldq::Ptr{Int64},
            w::Ptr{$T}, m::Ptr{Int64}, s::Ptr{$R}, sep::Ptr{$R}, work::Ptr{$T}, lwork::Ptr{Int64},
            info::Ptr{Int64}, lj::Clong, lc::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        N = Int(unsafe_load(n)); jb = _cabi_char(job); wantq = _cabi_char(compq) == 'V'
        Tm = PtrMatrix(Tm_, N, N, Int(unsafe_load(ldt)))
        Qm = PtrMatrix(Q, N, N, Int(unsafe_load(ldq)))
        selp = PtrVector(select, N)
        sel = Bool[selp[i] != 0 for i in 1:N]
        _, _, wv, sv, sepv, inf = _trsen_dispatch!(jb, wantq, sel, Tm, Qm)
        wp = PtrVector(w, N); @inbounds for i in 1:N; wp[i] = wv[i]; end
        unsafe_store!(s, $R(sv)); unsafe_store!(sep, $R(sepv))
        unsafe_store!(m, Int64(count(!iszero, sel))); unsafe_store!(info, Int64(inf)); return
    end
end

# ── gglse: equality-constrained least squares (min‖A·x−c‖ s.t. B·x=d) — real+complex ────────────────────
# {s,d,c,z}gglse_64_(m, n, p, A, lda, B, ldb, c, d, X, work, lwork, info) — 0 chars.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "gglse_64_"))(m::Ptr{Int64}, n::Ptr{Int64}, pp::Ptr{Int64},
            A::Ptr{$T}, lda::Ptr{Int64}, B::Ptr{$T}, ldb::Ptr{Int64}, cc::Ptr{$T}, dd::Ptr{$T}, X::Ptr{$T},
            work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64})::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); P = Int(unsafe_load(pp))
        Am = PtrMatrix(A, M, N, Int(unsafe_load(lda)))
        Bm = PtrMatrix(B, P, N, Int(unsafe_load(ldb)))
        cv = PtrVector(cc, M); dv = PtrVector(dd, P)
        x, _ = gglse!(Am, cv, Bm, dv)
        Xv = PtrVector(X, N); @inbounds for i in 1:N; Xv[i] = x[i]; end
        unsafe_store!(info, Int64(0)); return
    end
end

# ── ggsvd / ggsvd3: generalized SVD of the pair (A,B) — RANK-DEFICIENT-capable, ALL FOUR types (the
# ggsvd.jl kernel now does dggsvp rank-revealing preprocessing + dtgsja/ztgsja). The kernel mutates A/B
# IN PLACE to the LAPACK on-exit layout (R lives in A, or A+B for m<n) — the C caller reads R from there
# — so this helper only copies U/V/Q/alpha/beta/k/l out. alpha/beta are REAL. 'N' jobs skip that output.
@inline function _ggsvd_cabi!(::Type{T}, ::Type{Tr}, ju::Char, jv::Char, jq::Char, M::Int, N::Int, P::Int,
        A::Ptr{T}, lda::Int, B::Ptr{T}, ldb::Int, alpha::Ptr{Tr}, beta::Ptr{Tr},
        U::Ptr{T}, ldu::Int, V::Ptr{T}, ldv::Int, Q::Ptr{T}, ldq::Int,
        k::Ptr{Int64}, l::Ptr{Int64}, info::Ptr{Int64}) where {T,Tr}
    Am = PtrMatrix(A, M, N, lda); Bm = PtrMatrix(B, P, N, ldb)
    Uo, Vo, Qo, ao, bo, ko, lo, _ = ggsvd!(ju, jv, jq, Am, Bm)   # mutates Am/Bm in place → R lands in A/B
    ap = PtrVector(alpha, N); bp = PtrVector(beta, N)
    @inbounds for i in 1:N; ap[i] = ao[i]; bp[i] = bo[i]; end
    if ju == 'U'
        Um = PtrMatrix(U, M, M, ldu); @inbounds for j in 1:M, i in 1:M; Um[i, j] = Uo[i, j]; end
    end
    if jv == 'V'
        Vm = PtrMatrix(V, P, P, ldv); @inbounds for j in 1:P, i in 1:P; Vm[i, j] = Vo[i, j]; end
    end
    if jq == 'Q'
        Qm = PtrMatrix(Q, N, N, ldq); @inbounds for j in 1:N, i in 1:N; Qm[i, j] = Qo[i, j]; end
    end
    unsafe_store!(k, Int64(ko)); unsafe_store!(l, Int64(lo)); unsafe_store!(info, Int64(0))
    return
end
# {s,d}ggsvd_64_(jobu,jobv,jobq,m,n,p,k,l,A,lda,B,ldb,alpha,beta,U,ldu,V,ldv,Q,ldq,work,iwork,info,+3 lens)
# REAL: work real, trailing iwork. — 3 chars, NO lwork query (PureBLAS owns its own scratch).
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "ggsvd_64_"))(jobu::Ptr{UInt8}, jobv::Ptr{UInt8},
            jobq::Ptr{UInt8}, m::Ptr{Int64}, n::Ptr{Int64}, pp::Ptr{Int64}, k::Ptr{Int64}, l::Ptr{Int64},
            A::Ptr{$T}, lda::Ptr{Int64}, B::Ptr{$T}, ldb::Ptr{Int64}, alpha::Ptr{$T}, beta::Ptr{$T},
            U::Ptr{$T}, ldu::Ptr{Int64}, V::Ptr{$T}, ldv::Ptr{Int64}, Q::Ptr{$T}, ldq::Ptr{Int64},
            work::Ptr{$T}, iwork::Ptr{Int64}, info::Ptr{Int64}, lju::Clong, ljv::Clong, ljq::Clong)::Cvoid
        _ggsvd_cabi!($T, $T, _cabi_char(jobu), _cabi_char(jobv), _cabi_char(jobq), Int(unsafe_load(m)),
            Int(unsafe_load(n)), Int(unsafe_load(pp)), A, Int(unsafe_load(lda)), B, Int(unsafe_load(ldb)),
            alpha, beta, U, Int(unsafe_load(ldu)), V, Int(unsafe_load(ldv)), Q, Int(unsafe_load(ldq)),
            k, l, info)
        return
    end
    @eval Base.@ccallable function $(Symbol(p, "ggsvd3_64_"))(jobu::Ptr{UInt8}, jobv::Ptr{UInt8},
            jobq::Ptr{UInt8}, m::Ptr{Int64}, n::Ptr{Int64}, pp::Ptr{Int64}, k::Ptr{Int64}, l::Ptr{Int64},
            A::Ptr{$T}, lda::Ptr{Int64}, B::Ptr{$T}, ldb::Ptr{Int64}, alpha::Ptr{$T}, beta::Ptr{$T},
            U::Ptr{$T}, ldu::Ptr{Int64}, V::Ptr{$T}, ldv::Ptr{Int64}, Q::Ptr{$T}, ldq::Ptr{Int64},
            work::Ptr{$T}, lwork::Ptr{Int64}, iwork::Ptr{Int64}, info::Ptr{Int64},
            lju::Clong, ljv::Clong, ljq::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        _ggsvd_cabi!($T, $T, _cabi_char(jobu), _cabi_char(jobv), _cabi_char(jobq), Int(unsafe_load(m)),
            Int(unsafe_load(n)), Int(unsafe_load(pp)), A, Int(unsafe_load(lda)), B, Int(unsafe_load(ldb)),
            alpha, beta, U, Int(unsafe_load(ldu)), V, Int(unsafe_load(ldv)), Q, Int(unsafe_load(ldq)),
            k, l, info)
        return
    end
end
# COMPLEX: alpha/beta REAL; extra `rwork` (real) before iwork.
for (p, T, Tr) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval Base.@ccallable function $(Symbol(p, "ggsvd_64_"))(jobu::Ptr{UInt8}, jobv::Ptr{UInt8},
            jobq::Ptr{UInt8}, m::Ptr{Int64}, n::Ptr{Int64}, pp::Ptr{Int64}, k::Ptr{Int64}, l::Ptr{Int64},
            A::Ptr{$T}, lda::Ptr{Int64}, B::Ptr{$T}, ldb::Ptr{Int64}, alpha::Ptr{$Tr}, beta::Ptr{$Tr},
            U::Ptr{$T}, ldu::Ptr{Int64}, V::Ptr{$T}, ldv::Ptr{Int64}, Q::Ptr{$T}, ldq::Ptr{Int64},
            work::Ptr{$T}, rwork::Ptr{$Tr}, iwork::Ptr{Int64}, info::Ptr{Int64},
            lju::Clong, ljv::Clong, ljq::Clong)::Cvoid
        _ggsvd_cabi!($T, $Tr, _cabi_char(jobu), _cabi_char(jobv), _cabi_char(jobq), Int(unsafe_load(m)),
            Int(unsafe_load(n)), Int(unsafe_load(pp)), A, Int(unsafe_load(lda)), B, Int(unsafe_load(ldb)),
            alpha, beta, U, Int(unsafe_load(ldu)), V, Int(unsafe_load(ldv)), Q, Int(unsafe_load(ldq)),
            k, l, info)
        return
    end
    @eval Base.@ccallable function $(Symbol(p, "ggsvd3_64_"))(jobu::Ptr{UInt8}, jobv::Ptr{UInt8},
            jobq::Ptr{UInt8}, m::Ptr{Int64}, n::Ptr{Int64}, pp::Ptr{Int64}, k::Ptr{Int64}, l::Ptr{Int64},
            A::Ptr{$T}, lda::Ptr{Int64}, B::Ptr{$T}, ldb::Ptr{Int64}, alpha::Ptr{$Tr}, beta::Ptr{$Tr},
            U::Ptr{$T}, ldu::Ptr{Int64}, V::Ptr{$T}, ldv::Ptr{Int64}, Q::Ptr{$T}, ldq::Ptr{Int64},
            work::Ptr{$T}, lwork::Ptr{Int64}, rwork::Ptr{$Tr}, iwork::Ptr{Int64},
            info::Ptr{Int64}, lju::Clong, ljv::Clong, ljq::Clong)::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        _ggsvd_cabi!($T, $Tr, _cabi_char(jobu), _cabi_char(jobv), _cabi_char(jobq), Int(unsafe_load(m)),
            Int(unsafe_load(n)), Int(unsafe_load(pp)), A, Int(unsafe_load(lda)), B, Int(unsafe_load(ldb)),
            alpha, beta, U, Int(unsafe_load(ldu)), V, Int(unsafe_load(ldv)), Q, Int(unsafe_load(ldq)),
            k, l, info)
        return
    end
end
