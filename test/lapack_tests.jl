@testitem "potrf (Cholesky) vs LAPACK — lower/upper, sizes" begin
    using PureBLAS, LinearAlgebra
    spd(T, n) = (M = randn(T, n, n); M * M' + n * I)
    tol(::Type{T}) where {T} = sqrt(eps(real(T))) * 50
    @testset "$T n=$n" for T in (Float32, Float64),
        n in (1, 2, 5, 31, 96, 256, 512, 513, 700, 1025)

        A = Matrix{T}(spd(T, n)); F = cholesky(A)              # LAPACK reference
        L = PureBLAS.potrf!(copy(A); uplo = 'L')
        U = PureBLAS.potrf!(copy(A); uplo = 'U')
        @test norm(tril(L) - Matrix(F.L)) <= tol(T) * (norm(F.L) + 1)
        @test norm(triu(U) - Matrix(F.U)) <= tol(T) * (norm(F.U) + 1)
        @test norm(tril(L) * tril(L)' - A) <= tol(T) * (norm(A) + 1)   # reconstruction
    end
end

@testitem "potrf — non-positive-definite throws, shape check" begin
    using PureBLAS, LinearAlgebra
    @test_throws PosDefException PureBLAS.potrf!([1.0 2.0; 2.0 1.0]; uplo = 'L')   # indefinite
    @test_throws PosDefException PureBLAS.potrf!(zeros(3, 3); uplo = 'U')          # singular
    @test_throws DimensionMismatch PureBLAS.potrf!(randn(3, 4))
end

@testitem "potrf — ForwardDiff AD through the factor" begin
    using PureBLAS, LinearAlgebra, ForwardDiff
    # L[1,1] of [[x+4, 1],[1, 3]] = sqrt(x+4); d/dx = 1/(2 sqrt(x+4))
    f(x) = (A = [x + 4.0 1.0; 1.0 3.0]; PureBLAS.potrf!(A; uplo = 'L')[1, 1])
    @test ForwardDiff.derivative(f, 1.0) ≈ 0.5 / sqrt(5.0) rtol = 1e-10
    # gradient of logdet via Cholesky: logdet(A) = 2 Σ log(L[i,i])
    g(v) = (A = [v[1]+5.0 1.0; 1.0 v[2]+5.0]; L = PureBLAS.potrf!(A; uplo = 'L'); 2*(log(L[1,1]) + log(L[2,2])))
    gr = ForwardDiff.gradient(g, [1.0, 2.0])
    @test all(isfinite, gr)
end
