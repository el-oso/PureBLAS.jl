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
for (p, T) in (("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
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

Base.@ccallable function zgeqrt_64_(m::Ptr{Int64}, n::Ptr{Int64}, nb::Ptr{Int64}, A::Ptr{ComplexF64},
        lda::Ptr{Int64}, T::Ptr{ComplexF64}, ldt::Ptr{Int64}, work::Ptr{ComplexF64}, info::Ptr{Int64})::Cvoid
    M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); NB = Int(unsafe_load(nb)); k = min(M, N)
    Av = PtrMatrix(A, M, N, Int(unsafe_load(lda)))
    Tm = PtrMatrix(T, NB, k, Int(unsafe_load(ldt)))
    τ = Vector{ComplexF64}(undef, k)
    GC.@preserve τ begin
        geqrf!(Av, PtrVector(pointer(τ), k))                # complex τ already LAPACK-convention
        Vpan = Matrix{ComplexF64}(undef, M, NB); Gs = Matrix{ComplexF64}(undef, NB, NB)
        for i in 1:NB:k
            ib = min(NB, k - i + 1); mp = M - i + 1
            Vp = view(Vpan, 1:mp, 1:ib)
            _qr_vpanel!(Vp, Av, i, ib, mp)
            _qr_t_cmplx!(view(Tm, 1:ib, i:(i + ib - 1)), Vp, view(τ, i:(i + ib - 1)), Gs)
        end
    end
    unsafe_store!(info, Int64(0)); return
end

Base.@ccallable function zgemqrt_64_(side::Ptr{UInt8}, trans::Ptr{UInt8}, m::Ptr{Int64}, n::Ptr{Int64},
        k::Ptr{Int64}, nb::Ptr{Int64}, V::Ptr{ComplexF64}, ldv::Ptr{Int64}, T::Ptr{ComplexF64},
        ldt::Ptr{Int64}, C::Ptr{ComplexF64}, ldc::Ptr{Int64}, work::Ptr{ComplexF64}, info::Ptr{Int64},
        len_s::Clong, len_t::Clong)::Cvoid
    sd = _cabi_char(side); tr = _cabi_char(trans)
    M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); K = Int(unsafe_load(k)); NB = Int(unsafe_load(nb))
    vrows = sd == 'L' ? M : N
    Vm = PtrMatrix(V, vrows, K, Int(unsafe_load(ldv)))
    Tm = PtrMatrix(T, NB, K, Int(unsafe_load(ldt)))
    Cm = PtrMatrix(C, M, N, Int(unsafe_load(ldc)))
    forward = (sd == 'L' && tr != 'N') || (sd == 'R' && tr == 'N')     # L+C/R+N forward; L+N/R+C reverse
    starts = collect(1:NB:K); forward || reverse!(starts)
    Vpan = Matrix{ComplexF64}(undef, vrows, NB)
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
