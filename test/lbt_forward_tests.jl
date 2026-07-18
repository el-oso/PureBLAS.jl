# In-process LBT forwarding dogfood: PureBLAS.activate() must reroute LinearAlgebra's BLAS/LAPACK to
# PureBLAS's native kernels (via lbt_set_forward @cfunction pointers — cabi_forward.jl), inside a live
# Julia process, with correct results across the whole surface; deactivate() must restore OpenBLAS.
# This is the regression guard for the cabi_forward.jl @cfunction signatures (a mismatch → wrong result).

@testitem "LBT in-process forward: activate reroutes BLAS/LAPACK to PureBLAS" tags = [:lbt] begin
    using PureBLAS, LinearAlgebra, Random
    Random.seed!(0xBEEF)
    n = 96
    A = randn(n, n); B = randn(n, n); x = randn(n); y = randn(n)
    SPD = A * A' + n * I
    band = 3

    # OpenBLAS oracles, captured BEFORE activate (independent reference).
    ref = (
        mul   = A * B,
        gemv  = A * x,
        gemvt = transpose(A) * x,
        dot   = dot(x, y),
        nrm2  = norm(x),
        asum  = sum(abs, x),
        iamax = argmax(abs.(x)),
        symm  = Symmetric(A) * B,
        chol  = Matrix(cholesky(copy(SPD)).U),
        lu    = (F = lu(copy(A)); (L = F.L, U = F.U, p = F.p)),
        qrR   = qr(copy(A)).R,
        svals = svdvals(copy(A)),
        trsm  = UpperTriangular(A) \ B,
    )

    @test length(PureBLAS._LBT_REGISTRARS) > 100
    PureBLAS.activate()
    try
        @test maximum(abs, (A * B) .- ref.mul) < 1e-9              # gemm
        @test maximum(abs, (A * x) .- ref.gemv) < 1e-10           # gemv N
        @test maximum(abs, (transpose(A) * x) .- ref.gemvt) < 1e-10 # gemv T
        @test abs(dot(x, y) - ref.dot) < 1e-11                     # dot
        @test abs(norm(x) - ref.nrm2) < 1e-12                      # nrm2
        @test maximum(abs, (Symmetric(A) * B) .- ref.symm) < 1e-9  # symm
        C = cholesky(copy(SPD)); @test maximum(abs, Matrix(C.U) .- ref.chol) < 1e-8   # potrf
        F = lu(copy(A))                                            # getrf
        @test F.p == ref.lu.p
        @test maximum(abs, (F.L * F.U) .- A[F.p, :]) < 1e-9
        Q = qr(copy(A)); @test maximum(abs, abs.(Q.R) .- abs.(ref.qrR)) < 1e-9        # geqrf
        @test maximum(abs, svdvals(copy(A)) .- ref.svals) < 1e-9   # gesvd
        @test maximum(abs, (UpperTriangular(A) \ B) .- ref.trsm) < 1e-9   # trsm
    finally
        PureBLAS.deactivate()   # ALWAYS restore OpenBLAS so later testitems keep their oracle
    end
    # After deactivate the oracle is back (bit-identical to the pre-activate OpenBLAS result).
    @test A * B == ref.mul
end
