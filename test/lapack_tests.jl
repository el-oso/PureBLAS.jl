@testitem "potrf (Cholesky) vs LAPACK — lower/upper, sizes" begin
    using PureBLAS, LinearAlgebra
    spd(T, n) = (M = randn(T, n, n); M * M' + n * I)
    tol(::Type{T}) where {T} = sqrt(eps(real(T))) * 50
    @testset "$T n=$n" for T in (Float32, Float64),
        n in (1, 2, 5, 31, 96, 256, 512, 513, 700, 1025, 1536, 2049)

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

@testitem "geqrf (QR) vs LAPACK — square/tall/wide, R + reconstruction" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.LAPACK
    maxe(x, y) = maximum(abs.(x .- y)) / max(maximum(abs.(y)), 1e-300)
    # reconstruct A = Q*R from faer-convention factored output (H_k = I − v vᵀ/τ, τ=Inf ⇒ identity)
    function recon(F, tau)
        m, n = size(F); k = min(m, n)
        R = [i <= j ? F[i, j] : 0.0 for i in 1:m, j in 1:n]
        for kk in k:-1:1
            isfinite(tau[kk]) || continue
            v = zeros(m); v[kk] = 1.0; v[kk+1:m] = F[kk+1:m, kk]
            R .-= (v * (v' * R)) ./ tau[kk]
        end
        R
    end
    @testset "$m×$n" for (m, n) in ((1, 1), (8, 8), (40, 25), (64, 64), (129, 96), (200, 200), (96, 160), (600, 513))
        A0 = randn(m, n)
        F = copy(A0); tau = zeros(min(m, n)); PureBLAS.geqrf!(F, tau)
        Fl = copy(A0); LAPACK.geqrf!(Fl)                                   # LAPACK reference
        k = min(m, n)
        @test maxe(abs.(triu(F)[1:k, :]), abs.(triu(Fl)[1:k, :])) < 1e-11   # |R| matches up to sign
        @test maxe(recon(F, tau), A0) < 1e-11                              # Q·R ≈ A
    end
    @test_throws DimensionMismatch PureBLAS.geqrf!(randn(5, 5), zeros(2))
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
