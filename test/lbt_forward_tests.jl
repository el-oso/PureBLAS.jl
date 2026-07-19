# In-process LBT forwarding dogfood: PureBLAS.activate() must reroute LinearAlgebra's BLAS/LAPACK to
# PureBLAS's native kernels (via lbt_set_forward @cfunction pointers — cabi_forward.jl), inside a live
# Julia process, with correct results; deactivate() must restore OpenBLAS. Regression guard for the
# cabi_forward.jl @cfunction signatures (a mismatch → wrong result) AND for the "only self-consistent
# LAPACK symbols forwarded" invariant (a forwarded factorization whose companion stays on OpenBLAS must
# still be correct — e.g. geqrf is deliberately NOT forwarded: its faer-τ breaks OpenBLAS orgqr).

@testitem "LBT in-process forward: activate reroutes BLAS/LAPACK to PureBLAS" tags = [:lbt] begin
    using PureBLAS, LinearAlgebra, Random
    import LinearAlgebra.BLAS as B
    import LinearAlgebra.LAPACK as LA
    Random.seed!(0xBEEF)
    n = 96
    A = randn(n, n); Bm = randn(n, n); x = randn(n); y = randn(n); b = randn(n)
    SPD = A * A' + n * I
    At = A + n * I                # well-conditioned source for triangular solves (UpperTriangular(A) is near-singular)
    Az = randn(ComplexF64, n, n); zb = randn(ComplexF64, n); HPD = Az * Az' + n * I

    # OpenBLAS oracles, captured BEFORE activate (independent reference).
    ref = (mul = A * Bm, gemv = A * x, gemvt = transpose(A) * x, dot = dot(x, y), nrm2 = norm(x),
           symm = Symmetric(A) * Bm, trsm = UpperTriangular(At) \ Bm,
           chol = Matrix(cholesky(copy(SPD)).U), lusol = lu(copy(A)) \ b, chsol = cholesky(copy(SPD)) \ b,
           atsol = transpose(A) \ b, trsol = UpperTriangular(At) \ b,
           ichol = inv(cholesky(copy(SPD))), itri = inv(UpperTriangular(At)),
           qrsol = qr(copy(A)) \ b, qrdet = det(qr(copy(A)).Q),
           svals = svdvals!(copy(A)),   # svdvals! → gesdd (stays OpenBLAS); here just a numeric oracle
           zchol = Matrix(cholesky(Hermitian(copy(HPD))).U), zlusol = lu(copy(Az)) \ zb)

    @test length(PureBLAS._LBT_REGISTRARS) > 100
    fwd(s) = B.lbt_get_forward(s, Int32(B.LBT_INTERFACE_ILP64), Int32(B.LBT_F2C_PLAIN))
    gemm_before = fwd("dgemm_"); geqrf_before = fwd("dgeqrf_")
    zpotrf_before = fwd("zpotrf_"); zgetrf_before = fwd("zgetrf_"); gesdd_before = fwd("dgesdd_")
    getrs_before = fwd("dgetrs_"); potrs_before = fwd("dpotrs_"); trtrs_before = fwd("dtrtrs_")
    getri_before = fwd("dgetri_"); potri_before = fwd("dpotri_"); trtri_before = fwd("dtrtri_")
    geqrt_before = fwd("dgeqrt_"); gemqrt_before = fwd("dgemqrt_")
    PureBLAS.activate()
    try
        # BLAS — all route to PureBLAS.
        @test fwd("dgemm_") != gemm_before                         # actually forwarded (pointer changed)
        @test maximum(abs, (A * Bm) .- ref.mul) < 1e-9             # gemm
        @test maximum(abs, (A * x) .- ref.gemv) < 1e-10           # gemv N
        @test maximum(abs, (transpose(A) * x) .- ref.gemvt) < 1e-10 # gemv T
        @test abs(dot(x, y) - ref.dot) < 1e-11                     # dot
        @test abs(norm(x) - ref.nrm2) < 1e-12                      # nrm2
        @test maximum(abs, (Symmetric(A) * Bm) .- ref.symm) < 1e-9 # symm
        @test maximum(abs, (UpperTriangular(At) \ Bm) .- ref.trsm) < 1e-9  # trsm

        # LAPACK that routes to PureBLAS (self-consistent under a mixed backend).
        C = cholesky(copy(SPD)); @test maximum(abs, Matrix(C.U) .- ref.chol) < 1e-8   # potrf
        F = lu(copy(A))                                            # getrf (LAPACK-convention ipiv/factors)
        @test maximum(abs, (F.L * F.U) .- A[F.p, :]) < 1e-9
        # Solves now route to PureBLAS too (getrs/potrs/trtrs) — the solve step of `\`.
        @test fwd("dgetrs_") != getrs_before && fwd("dpotrs_") != potrs_before && fwd("dtrtrs_") != trtrs_before
        @test maximum(abs, (lu(copy(A)) \ b) .- ref.lusol) < 1e-9          # getrf + getrs, both PureBLAS
        @test maximum(abs, (transpose(A) \ b) .- ref.atsol) < 1e-9         # getrs trans='T' (reverse pivots)
        @test maximum(abs, (cholesky(copy(SPD)) \ b) .- ref.chsol) < 1e-9  # potrf + potrs
        @test maximum(abs, (UpperTriangular(At) \ b) .- ref.trsol) < 1e-9   # trtrs
        # Inverses now route (getri/potri/trtri) — inv() uses PureBLAS.
        @test fwd("dgetri_") != getri_before && fwd("dpotri_") != potri_before && fwd("dtrtri_") != trtri_before
        @test maximum(abs, inv(copy(A)) * A - I) < 1e-8                     # getri
        @test maximum(abs, inv(cholesky(copy(SPD))) - ref.ichol) < 1e-9    # potri
        @test maximum(abs, inv(UpperTriangular(At)) - ref.itri) < 1e-9     # trtri
        # gesvd routes (direct LAPACK call).
        S = LA.gesvd!('N', 'N', copy(A))[2]
        @test maximum(abs, S .- ref.svals) < 1e-9
        # gesdd routes → svd()/svdvals (Julia's default SVD path) now use PureBLAS.
        @test fwd("dgesdd_") != gesdd_before
        @test maximum(abs, svdvals(copy(A)) .- ref.svals) < 1e-9
        Fs = svd(copy(A)); @test maximum(abs, Fs.U * Diagonal(Fs.S) * Fs.Vt .- A) < 1e-9

        # Complex LAPACK — potrf/getrf route (self-consistent: standard-convention factors).
        @test fwd("zpotrf_") != zpotrf_before
        @test fwd("zgetrf_") != zgetrf_before
        Cz = cholesky(Hermitian(copy(HPD)))
        @test maximum(abs, Matrix(Cz.U) .- ref.zchol) < 1e-8               # zpotrf
        Fz = lu(copy(Az))
        @test maximum(abs, (Fz.L * Fz.U) .- Az[Fz.p, :]) < 1e-9            # zgetrf (valid factorization)
        @test maximum(abs, (lu(copy(Az)) \ zb) .- ref.zlusol) < 1e-8       # zgetrf + zgetrs, both PureBLAS

        # QR routes to PureBLAS via geqrt+gemqrt (Julia's qr() is QRCompactWY). geqrf is now forwarded too
        # (its wrapper converts faer τ → LAPACK, so geqrf!+OpenBLAS-orgqr is correct — no more NaN-Q hazard).
        @test fwd("dgeqrt_") != geqrt_before && fwd("dgemqrt_") != gemqrt_before && fwd("dgeqrf_") != geqrf_before
        Q = qr(copy(A))
        @test maximum(abs, Matrix(Q.Q)' * Matrix(Q.Q) - I) < 1e-9   # gemqrt: Q orthonormal
        @test maximum(abs, Matrix(Q.Q) * Q.R - A) < 1e-9            # geqrt+gemqrt: Q·R = A
        @test maximum(abs, (qr(copy(A)) \ b) .- ref.qrsol) < 1e-8   # geqrt+gemqrt+trtrs
        @test abs(det(qr(copy(A)).Q) - ref.qrdet) < 1e-9            # det(Q) reads T's diagonal (LAPACK-exact T)
        # geqrf τ-conversion: geqrf!+OpenBLAS orgqr now gives a valid Q (faer never crosses the ABI).
        A2 = copy(A); _, tau = LA.geqrf!(A2); Qo = LA.orgqr!(copy(A2), tau)
        @test maximum(abs, Qo'Qo - I) < 1e-9
    finally
        PureBLAS.deactivate()   # ALWAYS restore OpenBLAS so later testitems keep their oracle
    end
    @test A * Bm == ref.mul     # after deactivate the OpenBLAS oracle is back (bit-identical)
end
