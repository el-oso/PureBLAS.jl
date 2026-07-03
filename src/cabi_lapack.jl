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
        Av = view(unsafe_wrap(Array, A, (ld, N)), 1:N, 1:N)
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
    Av = view(unsafe_wrap(Array, A, (ld, N)), 1:M, 1:N)
    ip = unsafe_wrap(Array, ipiv, min(M, N))
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
    Av = view(unsafe_wrap(Array, A, (ld, N)), 1:M, 1:N)
    tv = unsafe_wrap(Array, tau, min(M, N))
    geqrf!(Av, tv)
    unsafe_store!(info, Int64(0))
    return
end

# ── gesvd: SVD ──────────────────────────────────────────────────────────────────────────────────────
# `dgesvd_64_(jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, info, len_jobu, len_jobvt)`.
# PureBLAS's gesvd! returns FRESH ECONOMY factors: U (m × min), S (min), Vᵀ (min × n), destroying A (as
# LAPACK gesvd does). We COPY them into the caller's S/U/VT buffers.
#
# SUPPORTED jobu/jobvt: 'N' (skip) and 'S' (economy — a direct copy). 'A' is supported ONLY when the full
# factor coincides with the economy one — jobu='A' needs U m×m so requires min==m (m≤n); jobvt='A' needs
# VT n×n so requires min==n (n≤m). For a square A ('A'=='S') this is the common all-vectors case.
# UNIMPLEMENTED (info set to -1, buffers untouched): jobu='O'/jobvt='O' (overwrite-A mode), jobu='A' with
# m>n, jobvt='A' with n>m — PureBLAS computes only the economy (min) vectors, so the extra columns/rows of
# the FULL square factor are unavailable; shipping a partly-filled buffer would be subtly wrong.
#
# TRIM LIMITATION (unlike potrf/getrf/geqrf, this symbol does NOT pass TrimCheck.@validate): gesvd! has a
# type-UNSTABLE return — U is a SubArray on the m≥n path but a Matrix on the m<n transpose-recursion path,
# and want_vectors unions a 1-tuple with a 3-tuple — plus internals (permutedims, bdsdc! divide-and-conquer)
# that aren't trim-clean. That Union flows into the copyto! below as an ::Any source. So dgesvd_64_ is
# correct (validated by reconstruction) but can't yet be compiled into libpureblas.so via juliac --trim;
# making it trim-safe needs gesvd! itself refactored to a single concrete return type (a separate pass).
# GATED OUT of the trimmed .so: defined as a PLAIN function (no `Base.@ccallable`) so --compile-ccallable
# does not export it and --trim=safe does not choke on it. Re-add `Base.@ccallable` once gesvd! is
# refactored to a concrete return. The body is unchanged and still callable/validatable in-Julia.
function dgesvd_64_(jobu::Ptr{UInt8}, jobvt::Ptr{UInt8}, m::Ptr{Int64}, n::Ptr{Int64},
        A::Ptr{Float64}, lda::Ptr{Int64}, S::Ptr{Float64}, U::Ptr{Float64}, ldu::Ptr{Int64},
        VT::Ptr{Float64}, ldvt::Ptr{Int64}, work::Ptr{Float64}, lwork::Ptr{Int64}, info::Ptr{Int64},
        len_jobu::Clong, len_jobvt::Clong)::Cvoid
    if unsafe_load(lwork) == Int64(-1)                 # workspace query
        unsafe_store!(work, 1.0); unsafe_store!(info, Int64(0)); return
    end
    ju = _cabi_char(jobu); jvt = _cabi_char(jobvt)
    M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); mn = min(M, N)
    u_ok = ju == 'N' || ju == 'S' || (ju == 'A' && M <= N)
    v_ok = jvt == 'N' || jvt == 'S' || (jvt == 'A' && N <= M)
    if !(u_ok && v_ok)                                 # 'O' or an unsupported full 'A' → flag, don't ship
        unsafe_store!(info, Int64(-1)); return
    end
    ld = Int(unsafe_load(lda))
    Av = view(unsafe_wrap(Array, A, (ld, N)), 1:M, 1:N)
    if ju == 'N' && jvt == 'N'
        (Sc,) = gesvd!(Av; want_vectors = false)
        copyto!(unsafe_wrap(Array, S, mn), Sc)
    else
        Uc, Sc, Vtc = gesvd!(Av; want_vectors = true)   # Uc: M×mn, Sc: mn, Vtc: mn×N
        copyto!(unsafe_wrap(Array, S, mn), Sc)
        if ju != 'N'
            Uw = view(unsafe_wrap(Array, U, (Int(unsafe_load(ldu)), mn)), 1:M, 1:mn)
            copyto!(Uw, Uc)
        end
        if jvt != 'N'
            Vw = view(unsafe_wrap(Array, VT, (Int(unsafe_load(ldvt)), N)), 1:mn, 1:N)
            copyto!(Vw, Vtc)
        end
    end
    unsafe_store!(info, Int64(0))
    return
end
