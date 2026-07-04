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

# ── potrf: Cholesky ─────────────────────────────────────────────────────────────────────────────────
# `{d,s}potrf_64_(uplo, n, A, lda, info, len_uplo)` — 1 char. potrf! throws PosDefException on a non-SPD
# input; CATCH it at the boundary and report the failing minor via info>0 (LAPACK convention) rather than
# letting a Julia exception unwind through the C-ABI. (Note: the Float64 lower faer fast path reports its
# non-SPD as PosDefException(1) — info=1, not the exact minor; correctness-wise info>0 is all LAPACK asks.)
for (p, T) in (("s", Float32), ("d", Float64))
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

# ── getrf: LU with partial pivoting ─────────────────────────────────────────────────────────────────
# `dgetrf_64_(m, n, A, lda, ipiv, info)` — 0 chars, no hidden lengths. ipiv is `Ptr{Int64}` OUTPUT
# (length min(m,n)); bridge as a Vector{Int64} (== Vector{Int} on 64-bit) — LAPACK 1-based pivot rows.
Base.@ccallable function dgetrf_64_(m::Ptr{Int64}, n::Ptr{Int64}, A::Ptr{Float64}, lda::Ptr{Int64},
        ipiv::Ptr{Int64}, info::Ptr{Int64})::Cvoid
    M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); ld = Int(unsafe_load(lda))
    Av = PtrMatrix(A, M, N, ld)
    ip = PtrVector(ipiv, min(M, N))
    _, _, inf = getrf!(Av, ip)
    unsafe_store!(info, Int64(inf))
    return
end

# ── geqrf: QR (Householder, no pivoting) ────────────────────────────────────────────────────────────
# `dgeqrf_64_(m, n, A, lda, tau, work, lwork, info)` — 0 chars. Honors the lwork==-1 query.
# IMPORTANT: PureBLAS's τ uses the faer convention H = I − v·vᵀ/τ (τ = 1/τ_LAPACK); this native convention
# is what the symbol returns, so a caller using PureBLAS-as-LAPACK gets PureBLAS's τ (NOT reference τ).
Base.@ccallable function dgeqrf_64_(m::Ptr{Int64}, n::Ptr{Int64}, A::Ptr{Float64}, lda::Ptr{Int64},
        tau::Ptr{Float64}, work::Ptr{Float64}, lwork::Ptr{Int64}, info::Ptr{Int64})::Cvoid
    if unsafe_load(lwork) == Int64(-1)                 # workspace query: report size 1, do nothing else
        unsafe_store!(work, 1.0); unsafe_store!(info, Int64(0)); return
    end
    M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); ld = Int(unsafe_load(lda))
    Av = PtrMatrix(A, M, N, ld)
    tv = PtrVector(tau, min(M, N))
    geqrf!(Av, tv)
    unsafe_store!(info, Int64(0))
    return
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
Base.@ccallable function dgesvd_64_(jobu::Ptr{UInt8}, jobvt::Ptr{UInt8}, m::Ptr{Int64}, n::Ptr{Int64},
        A::Ptr{Float64}, lda::Ptr{Int64}, S::Ptr{Float64}, U::Ptr{Float64}, ldu::Ptr{Int64},
        VT::Ptr{Float64}, ldvt::Ptr{Int64}, work::Ptr{Float64}, lwork::Ptr{Int64}, info::Ptr{Int64},
        len_jobu::Clong, len_jobvt::Clong)::Cvoid
    if unsafe_load(lwork) == Int64(-1)                 # workspace query
        unsafe_store!(work, 1.0); unsafe_store!(info, Int64(0)); return
    end
    ju = _cabi_char(jobu); jvt = _cabi_char(jobvt)
    M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); mn = min(M, N)
    u_ok = ju == 'N' || ju == 'S' || ju == 'A' || ju == 'O'
    v_ok = jvt == 'N' || jvt == 'S' || jvt == 'A' || jvt == 'O'
    if !(u_ok && v_ok) || (ju == 'O' && jvt == 'O')    # unknown job, or the illegal jobu=jobvt='O'
        unsafe_store!(info, Int64(-1)); return
    end
    ld = Int(unsafe_load(lda))
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
                     PtrMatrix(U, M, ncu, Int(unsafe_load(ldu)))
        Vtt = vscr ? PtrMatrix(pointer(ws.cabi_Vt), mn, N, size(ws.cabi_Vt, 1)) :
                     PtrMatrix(VT, ncv, N, Int(unsafe_load(ldvt)))
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
