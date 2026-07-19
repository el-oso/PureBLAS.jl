module PureBLAS

# Pure-Julia BLAS for the Pure ecosystem. Milestone 1: BLAS Level-1, all four element types via
# generic `T<:Number` kernels, plugged into Julia two ways — directly (AD-friendly native API)
# and as a libblastrampoline drop-in (juliac --trim → libpureblas.so). See ROADMAP.md for L2/L3.

include("core.jl")          # type aliases, _ld/_st! accessors, lassq, |·|
include("ptrmat.jl")        # PtrMatrix/PtrVector: isbits Ptr-backed operands for the C-ABI boundary
include("cpuinfo.jl")       # SIMD width detection (const-folded, trim-safe)
include("simd_kernels.jl")  # SIMD.jl fast paths (real, unit-stride, dense)
include("level1.jl")        # low-level (n,…,inc) kernels — shared by both modes
include("level2.jl")        # Level-2 gemv/ger/symv/hemv/trmv/trsv kernels (on the L1 column kernels)
include("level2_packed.jl") # Level-2 packed storage: spmv/hpmv/tpmv/tpsv
include("level2_banded.jl") # Level-2 band storage: gbmv/sbmv/hbmv/tbmv/tbsv
include("contracts.jl")     # TypeContracts AbstractBLAS1 / AbstractBLAS2 interfaces
include("backend.jl")       # SIMDBackend: high-level AbstractVector ops (Mode 2)
include("native.jl")        # bare native API → default backend
include("workspace.jl")     # L3Workspace: owned per-type Level-3/LAPACK scratch (replaces global caches)
include("gemm.jl")          # Level-3 GEMM (BLIS 5-loop + SIMD microkernel; generic fallback)
include("level3.jl")        # Level-3 trmm/trsm (recursive blocking, reuses gemm!)
include("lapack.jl")        # LAPACK: Cholesky (potrf) on the gated L3
include("qr.jl")            # LAPACK: QR (geqrf) — faer panel reduction + gemm! dlarfb
include("wy.jl")            # compact-WY block-reflector kernels (dlarft/dlarfb roles), caller-
                             # owned workspace — PureSparse.jl M5b multifrontal QR's P1a/P1b
include("lu.jl")            # LAPACK: LU (getrf) — pivoted panel + gemm!/trsm! trailing
include("svd_dqds.jl")      # LAPACK: dqds (dlasq1-6) — fast bidiagonal singular VALUES (values-only path)
include("svd.jl")           # LAPACK: SVD (gesvd) — gebrd + bidiagonal implicit-QR + back-transform
include("svd_dc.jl")        # LAPACK: SVD divide-and-conquer bidiagonal solver (bdsdc, faer port)
include("eigen.jl")         # LAPACK: symmetric/Hermitian eigensolver (syev/heev) — sytrd/hetrd + steqr + ormtr/unmtr
include("eigen_dc.jl")      # LAPACK: symmetric tridiagonal divide-and-conquer (stedc, Cuppen) — jobz='V' path
include("lq.jl")            # LAPACK: LQ (gelqf/orglq/ormlq) — row-wise dual of QR, generic s/d/c/z
include("bunchkaufman.jl")  # LAPACK: Bunch-Kaufman (sytrf/hetrf + sytrs/hetrs) symmetric-indefinite/Hermitian
include("geqp3.jl")         # LAPACK: column-pivoted QR (geqp3) — rank-revealing, generic s/d/c/z
include("gels.jl")          # LAPACK: least-squares / min-norm solve (gels) over QR/LQ
include("gecon.jl")         # LAPACK: condition estimation (gecon/trcon/pocon) — Higham–Hager estimator
include("hessenberg.jl")    # LAPACK: Hessenberg reduction (gebal/gehrd/orghr) — nonsymmetric-eigen front half
include("hseqr.jl")         # LAPACK: Schur decomposition of upper-Hessenberg (hseqr, Francis double-shift QR)
include("trevc.jl")         # LAPACK: right eigenvectors of Schur form (trevc, back-substitution)
include("geev.jl")          # LAPACK: general eigensolver drivers (geev/gees + gebak) — eigen/eigvals/schur
include("verify.jl")        # precompile-time @verify_strict SIMDBackend (needs all ops defined first)
include("cabi.jl")          # @ccallable Fortran-ABI symbols (Mode 1): BLAS-1 + gemm
include("cabi_l2.jl")       # Mode 1: BLAS-2 (gemv/ger/symv/…, packed, banded)
include("cabi_l3.jl")       # Mode 1: BLAS-3 rest (symm/syrk/trmm/trsm/…)
include("cabi_lapack.jl")   # Mode 1: LAPACK (potrf/getrf/geqrf/gesvd)
include("cabi_forward.jl")  # in-process LBT forward registry (@cfunction pointers to the above)
include("lbt.jl")           # activate/deactivate via BLAS.lbt_set_forward

# ── Precompile workload ────────────────────────────────────────────────────────────────────────────
# Run the hot native kernels once at precompile time so their native code is cached in the .ji — first-call
# gemm!/potrf!/eigen drops from ~3.5ms (base inference only) to ~0.06ms. Precompile-time only (the macros
# guard on jl_generating_output); NOT reachable from the @ccallable roots, so juliac --trim excludes it
# (verified: the .so still builds). Small n keeps the cost low (~18s of the ~140s module precompile — the
# base kernel surface dominates, not this). ALWAYS ON: an env/@static gate on the workload is cache-UNSAFE
# (ENV is not part of the .ji key → a .ji built once with the workload skipped is silently reused later),
# and the CI saving was only ~18s, so it isn't worth gating. Covers L1/L2/L3 + Cholesky/LU/QR + sym/Herm
# eigen across Float64/Float32/ComplexF64.
using PrecompileTools: @setup_workload, @compile_workload
@setup_workload begin
    n = 8
    @compile_workload begin
        for T in (Float64, Float32, ComplexF64)
            x = ones(T, n); y = fill(T(2), n)
            axpy!(copy(y), one(T), x); dot(x, y); dotu(x, y); nrm2(x)
            A = reshape(collect(T, 1:n * n), n, n) ./ T(n)
            gemv(A, x); gemv(A, x; trans = 'T')
            C = zeros(T, n, n)
            gemm!(C, A, A; alpha = one(T), beta = zero(T))
            gemm!(copy(C), A, A; alpha = one(T), beta = one(T), transA = 'T', transB = 'N')
        end
        for T in (Float64, ComplexF64)
            A = reshape(collect(T, 1:n * n), n, n) ./ T(n)
            S = A * A'; @inbounds for i in 1:n; S[i, i] += T(n); end   # Hermitian PD
            potrf!(copy(S); uplo = 'L')
            G = copy(A); @inbounds for i in 1:n; G[i, i] += T(n); end  # nonsingular
            getrf!(G)
            tau = zeros(T, n); geqrf!(copy(A), tau)
        end
        As = reshape(collect(Float64, 1:n * n), n, n); As = As .+ As'
        _syev!('V', 'L', copy(As))
        Ah = reshape(collect(ComplexF64, 1:n * n), n, n); Ah = Ah .+ Ah'
        _heev!('V', 'L', copy(Ah))
    end
end

end # module
