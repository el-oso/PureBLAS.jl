module PureBLAS

# Pure-Julia BLAS for the Pure ecosystem. Milestone 1: BLAS Level-1, all four element types via
# generic `T<:Number` kernels, plugged into Julia two ways — directly (AD-friendly native API)
# and as a libblastrampoline drop-in (juliac --trim → libpureblas.so). See ROADMAP.md for L2/L3.

include("core.jl")          # type aliases, _ld/_st! accessors, lassq, |·|
include("ptrmat.jl")        # PtrMatrix/PtrVector: isbits Ptr-backed operands for the C-ABI boundary
include("cpuinfo.jl")       # SIMD width detection (const-folded, trim-safe)
include("simd_kernels.jl")  # SIMD.jl fast paths (real, unit-stride, dense)
include("blas1/level1.jl")        # low-level (n,…,inc) kernels — shared by both modes
include("blas2/level2.jl")        # Level-2 gemv/ger/symv/hemv/trmv/trsv kernels (on the L1 column kernels)
include("blas2/level2_packed.jl") # Level-2 packed storage: spmv/hpmv/tpmv/tpsv
include("blas2/level2_banded.jl") # Level-2 band storage: gbmv/sbmv/hbmv/tbmv/tbsv
include("contracts.jl")     # TypeContracts AbstractBLAS1 / AbstractBLAS2 interfaces
include("backend.jl")       # SIMDBackend: high-level AbstractVector ops (Mode 2)
include("native.jl")        # bare native API → default backend
include("workspace.jl")     # L3Workspace: owned per-type Level-3/LAPACK scratch (replaces global caches)
include("blas3/gemm.jl")          # Level-3 GEMM (BLIS 5-loop + SIMD microkernel; generic fallback)
include("blas3/level3.jl")        # Level-3 trmm/trsm (recursive blocking, reuses gemm!)
include("lapack/lapack.jl")        # LAPACK: Cholesky (potrf) on the gated L3
include("lapack/qr.jl")            # LAPACK: QR (geqrf) — faer panel reduction + gemm! dlarfb
include("lapack/wy.jl")            # compact-WY block-reflector kernels (dlarft/dlarfb roles), caller-
# owned workspace — PureSparse.jl M5b multifrontal QR's P1a/P1b
include("lapack/lu.jl")            # LAPACK: LU (getrf) — pivoted panel + gemm!/trsm! trailing
include("lapack/svd_dqds.jl")      # LAPACK: dqds (dlasq1-6) — fast bidiagonal singular VALUES (values-only path)
include("lapack/svd.jl")           # LAPACK: SVD (gesvd) — gebrd + bidiagonal implicit-QR + back-transform
include("lapack/svd_dc.jl")        # LAPACK: SVD divide-and-conquer bidiagonal solver (bdsdc, faer port)
include("lapack/eigen.jl")         # LAPACK: symmetric/Hermitian eigensolver (syev/heev) — sytrd/hetrd + steqr + ormtr/unmtr
include("lapack/eigen_dc.jl")      # LAPACK: symmetric tridiagonal divide-and-conquer (stedc, Cuppen) — jobz='V' path
include("lapack/lq.jl")            # LAPACK: LQ (gelqf/orglq/ormlq) — row-wise dual of QR, generic s/d/c/z
include("lapack/bunchkaufman.jl")  # LAPACK: Bunch-Kaufman (sytrf/hetrf + sytrs/hetrs) symmetric-indefinite/Hermitian
include("lapack/geqp3.jl")         # LAPACK: column-pivoted QR (geqp3) — rank-revealing, generic s/d/c/z
include("lapack/gels.jl")          # LAPACK: least-squares / min-norm solve (gels) over QR/LQ
include("lapack/gecon.jl")         # LAPACK: condition estimation (gecon/trcon/pocon) — Higham–Hager estimator
include("lapack/hessenberg.jl")    # LAPACK: Hessenberg reduction (gebal/gehrd/orghr) — nonsymmetric-eigen front half
include("lapack/hseqr.jl")         # LAPACK: Schur decomposition of upper-Hessenberg (hseqr, Francis double-shift QR)
include("lapack/trevc.jl")         # LAPACK: right eigenvectors of Schur form (trevc, back-substitution)
include("lapack/geev.jl")          # LAPACK: general eigensolver drivers (geev/gees + gebak) — eigen/eigvals/schur
include("lapack/sygvd.jl")         # LAPACK: generalized sym/Herm-definite eigensolver (sygvd/hegvd) — eigen(Sym,Sym)
include("lapack/tridiag.jl")       # LAPACK: tridiagonal solvers (gtsv/gttrf/gttrs) + _gt_asmat helper
include("lapack/banded_chol.jl")   # LAPACK: band Cholesky (pbtrf/pbtrs)
include("lapack/packed_chol.jl")   # LAPACK: packed Cholesky (pptrf/pptrs)
include("lapack/qz.jl")            # LAPACK: generalized-eigen QZ kernels (gghrd/hgeqz + auxiliaries)
include("lapack/tgevc_gen.jl")     # LAPACK: generalized right eigenvectors (tgevc) — needs qz.jl first
include("lapack/ggev.jl")          # LAPACK: generalized eigensolver drivers (ggev/gges) — eigen(A,B)/eigvals/schur(A,B)
include("lapack/sysv.jl")          # LAPACK: symmetric-indefinite/Hermitian solve+inverse (sysv/hesv/sytri/hetri)
include("lapack/gbtrf.jl")         # LAPACK: general banded LU (gbtrf/gbtrs)
include("lapack/pttrf.jl")         # LAPACK: SPD tridiagonal LDLᴴ (pttrf/pttrs/ptsv)
include("lapack/stebz.jl")         # LAPACK: sym-tridiag eigvals by bisection (stebz) / eigvecs by inverse iteration (stein)
include("lapack/pstrf.jl")         # LAPACK: pivoted/semidefinite Cholesky (pstrf)
include("lapack/qlrq.jl")          # LAPACK: QL/RQ factorizations (geqlf/gerqf/orgql/orgrq/ormql/ormrq + complex duals)
include("lapack/gelsy.jl")         # LAPACK: rank-deficient LS via RZ (gelsy/tzrzf/ormrz) — needs geqp3.jl + gels.jl
include("lapack/gelsd.jl")         # LAPACK: rank-deficient LS via SVD (gelsd) — needs svd.jl
include("lapack/trsyl.jl")         # LAPACK: Sylvester solve (trsyl) — standalone
include("lapack/trsen.jl")         # LAPACK: Schur reorder (trexc/trsen) — needs trsyl.jl
include("lapack/gglse.jl")         # LAPACK: equality-constrained LS (gglse)
include("lapack/ggsvd.jl")         # LAPACK: generalized SVD (ggsvd) — rank-deficient-capable, all s/d/c/z (dggsvp + dtgsja)
include("lapack/syconv.jl")        # LAPACK: Bunch-Kaufman factorization convert (syconv)
include("lapack/trrfs.jl")         # LAPACK: triangular-solve forward/backward error bounds (trrfs)
include("lapack/tgsen.jl")         # LAPACK: generalized Schur reorder (tgsen) — complex complete; real all-real-λ only
include("lapack/gesvx.jl")         # LAPACK: expert general solve (gesvx) — equilibrate + LU + refine + error bounds
include("verify.jl")        # precompile-time @verify_strict SIMDBackend (needs all ops defined first)
include("cabi/cabi.jl")          # @ccallable Fortran-ABI symbols (Mode 1): BLAS-1 + gemm
include("cabi/cabi_cdot.jl")     # Mode 1: complex BLAS-1 dot (c/zdotu, c/zdotc) — needs _dotu/_dotc in scope from cabi.jl
include("cabi/cabi_l2.jl")       # Mode 1: BLAS-2 (gemv/ger/symv/…, packed, banded)
include("cabi/cabi_l3.jl")       # Mode 1: BLAS-3 rest (symm/syrk/trmm/trsm/…)
include("cabi/cabi_lapack.jl")   # Mode 1: LAPACK (potrf/getrf/geqrf/gesvd)
include("cabi/cabi_lapack2.jl")  # Mode 1: LAPACK batch 2 (gesv/posv/lacpy/larfg/larf/gebak/hseqr/trevc/
# sytrd·hetrd/orgtr·ungtr/ormtr·unmtr/orgqr·ungqr/ormqr·unmqr/ormhr·unmhr/
# gebrd/bdsqr/bdsdc) — OpenBLAS-removal ratchet follow-up
include("cabi/cabi_lapack3.jl")  # Mode 1: LAPACK batch 3 (syconv/trrfs; tgsen/ggsvd-complex to follow)
include("cabi/cabi_forward.jl")  # in-process LBT forward registry (@cfunction pointers to the above)
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
            A = reshape(collect(T, 1:(n * n)), n, n) ./ T(n)
            gemv(A, x); gemv(A, x; trans = 'T')
            C = zeros(T, n, n)
            gemm!(C, A, A; alpha = one(T), beta = zero(T))
            gemm!(copy(C), A, A; alpha = one(T), beta = one(T), transA = 'T', transB = 'N')
        end
        for T in (Float64, ComplexF64)
            A = reshape(collect(T, 1:(n * n)), n, n) ./ T(n)
            S = A * A'; @inbounds for i in 1:n
                S[i, i] += T(n)
            end   # Hermitian PD
            potrf!(copy(S); uplo = 'L')
            G = copy(A); @inbounds for i in 1:n
                G[i, i] += T(n)
            end  # nonsingular
            getrf!(G)
            tau = zeros(T, n); geqrf!(copy(A), tau)
        end
        As = reshape(collect(Float64, 1:(n * n)), n, n); As = As .+ As'
        _syev!('V', 'L', copy(As))
        Ah = reshape(collect(ComplexF64, 1:(n * n)), n, n); Ah = Ah .+ Ah'
        _heev!('V', 'L', copy(Ah))
    end
end

end # module
