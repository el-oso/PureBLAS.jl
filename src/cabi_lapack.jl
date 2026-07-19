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
