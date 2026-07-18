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

    # OpenBLAS oracles, captured BEFORE activate (independent reference).
    ref = (mul = A * Bm, gemv = A * x, gemvt = transpose(A) * x, dot = dot(x, y), nrm2 = norm(x),
           symm = Symmetric(A) * Bm, trsm = UpperTriangular(A) \ Bm,
           chol = Matrix(cholesky(copy(SPD)).U), lusol = lu(copy(A)) \ b, chsol = cholesky(copy(SPD)) \ b,
           svals = svdvals!(copy(A)))   # svdvals! → gesdd (stays OpenBLAS); here just a numeric oracle

    @test length(PureBLAS._LBT_REGISTRARS) > 100
    fwd(s) = B.lbt_get_forward(s, Int32(B.LBT_INTERFACE_ILP64), Int32(B.LBT_F2C_PLAIN))
    gemm_before = fwd("dgemm_"); geqrf_before = fwd("dgeqrf_")
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
        @test maximum(abs, (UpperTriangular(A) \ Bm) .- ref.trsm) < 1e-9  # trsm

        # LAPACK that routes to PureBLAS (self-consistent under a mixed backend).
        C = cholesky(copy(SPD)); @test maximum(abs, Matrix(C.U) .- ref.chol) < 1e-8   # potrf
        F = lu(copy(A))                                            # getrf (LAPACK-convention ipiv/factors)
        @test maximum(abs, (F.L * F.U) .- A[F.p, :]) < 1e-9
        @test maximum(abs, (lu(copy(A)) \ b) .- ref.lusol) < 1e-9  # getrf(PB) + getrs(OpenBLAS) consistent
        @test maximum(abs, (cholesky(copy(SPD)) \ b) .- ref.chsol) < 1e-9  # potrf(PB) + potrs(OpenBLAS)
        # gesvd routes (direct LAPACK call; svd()/svdvals default to gesdd, which stays on OpenBLAS).
        S = LA.gesvd!('N', 'N', copy(A))[2]
        @test maximum(abs, S .- ref.svals) < 1e-9

        # HAZARD GUARD: geqrf must NOT be forwarded (faer-τ would break OpenBLAS orgqr → NaN Q). With it
        # unforwarded, geqrf!+orgqr! are both OpenBLAS and produce a valid orthonormal Q.
        @test fwd("dgeqrf_") == geqrf_before                       # dgeqrf_ NOT redirected (stays OpenBLAS)
        A2 = copy(A); _, tau = LA.geqrf!(A2); Q = LA.orgqr!(copy(A2), tau)
        @test maximum(abs, Q'Q - I) < 1e-9
    finally
        PureBLAS.deactivate()   # ALWAYS restore OpenBLAS so later testitems keep their oracle
    end
    @test A * Bm == ref.mul     # after deactivate the OpenBLAS oracle is back (bit-identical)
end
