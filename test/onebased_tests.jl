@testmodule ZeroBased begin
    using LinearAlgebra
    export Zb, zb

    # Minimal array whose axes start at 0: `a[i]` reads `parent[i + 1]`. Its buffer is the parent array's, so
    # a kernel that indexed it from 1 or took `pointer` would touch the wrong elements.
    struct Zb{T, N, P <: AbstractArray{T, N}} <: AbstractArray{T, N}
        parent::P
    end
    Base.size(a::Zb) = size(a.parent)
    Base.axes(a::Zb) = map(r -> (first(r) - 1):(last(r) - 1), axes(a.parent))
    Base.getindex(a::Zb{T, N}, I::Vararg{Int, N}) where {T, N} = a.parent[(I .+ 1)...]
    Base.setindex!(a::Zb{T, N}, v, I::Vararg{Int, N}) where {T, N} = (a.parent[(I .+ 1)...] = v; a)
    zb(p) = Zb{eltype(p), ndims(p), typeof(p)}(p)
end

@testitem "offset axes are rejected before any kernel runs" setup = [ZeroBased] begin
    using PureBLAS, LinearAlgebra
    msg = "offset arrays are not supported"
    n = 6
    @test !isempty(axes(zb(zeros(n)), 1)) && first(axes(zb(zeros(n)), 1)) == 0

    # `throws(f, parents...)`: wrap each parent in turn, expect an ArgumentError, and expect every parent
    # buffer unchanged afterwards.
    function throws(f, parents...)
        for k in eachindex(parents)
            ps = map(copy, parents)
            before = map(copy, ps)
            args = ntuple(i -> i == k ? zb(ps[i]) : ps[i], length(ps))
            @test_throws ArgumentError f(args...)
            @test_throws msg f(args...)
            @test all(ps .== before)
        end
    end
    spd(T, n) = (A = randn(T, n, n); A * A' + n * I)

    @testset "BLAS-1 $T" for T in (Float64, ComplexF64)
        x = randn(T, n); y = randn(T, n); a = randn(T)
        throws((y, x) -> PureBLAS.axpy!(y, a, x), y, x)
        throws((x, y) -> PureBLAS.dot(x, y), x, y)
        throws((x, y) -> PureBLAS.dotu(x, y), x, y)
        throws(x -> PureBLAS.nrm2(x), x)
        throws(x -> PureBLAS.asum(x), x)
        throws(x -> PureBLAS.iamax(x), x)
        throws(x -> PureBLAS.scal!(a, x), x)
        throws((y, x) -> PureBLAS.blascopy!(y, x), y, x)
        throws((x, y) -> PureBLAS.swap!(x, y), x, y)
    end

    @testset "BLAS-2 $T" for T in (Float64, ComplexF64)
        A = randn(T, n, n); x = randn(T, n); y = randn(T, n); a = randn(T)
        throws((y, A, x) -> PureBLAS.gemv!(y, A, x), y, A, x)
        throws((A, x) -> PureBLAS.gemv(A, x), A, x)
        throws((x, y, A) -> PureBLAS.ger!(a, x, y, A), x, y, A)
        throws((A, x) -> PureBLAS.trsv!(A, x), A, x)
        throws((A, x) -> PureBLAS.trmv!(A, x), A, x)
        throws((y, A, x) -> PureBLAS.symv!(y, A, x), y, A, x)
        AP = randn(T, n * (n + 1) ÷ 2)
        throws((y, AP, x) -> PureBLAS.spmv!(y, AP, x), y, AP, x)
        AB = randn(T, 3, n)
        throws((y, AB, x) -> PureBLAS.sbmv!(y, AB, x), y, AB, x)
    end

    @testset "BLAS-3 $T" for T in (Float64, ComplexF64)
        A = randn(T, n, n); B = randn(T, n, n); C = randn(T, n, n)
        throws((C, A, B) -> PureBLAS.gemm!(C, A, B), C, A, B)
        throws((A, B) -> PureBLAS.gemm(A, B), A, B)
        throws((B, A) -> PureBLAS.trsm!(B, A), B, A)
        throws((B, A) -> PureBLAS.trmm!(B, A), B, A)
        throws((C, A) -> PureBLAS.syrk!(C, A), C, A)
        throws((C, A, B) -> PureBLAS.symm!(C, A, B), C, A, B)
        throws((C, A, B) -> PureBLAS.syr2k!(C, A, B), C, A, B)
    end

    @testset "LAPACK $T" for T in (Float64, ComplexF64)
        A = randn(T, n, n); ipiv = zeros(Int, n); tau = zeros(T, n)
        # The one-argument complex forms accept only `StridedMatrix`, so an `AbstractArray` wrapper is a
        # `MethodError` there rather than reaching the guard.
        T <: Real && throws(A -> PureBLAS.getrf!(A), A)
        throws((A, ipiv) -> PureBLAS.getrf!(A, ipiv), A, ipiv)
        throws(A -> PureBLAS.potrf!(A), spd(T, n))
        throws((A, tau) -> PureBLAS.geqrf!(A, tau), A, tau)
        T <: Real && throws(A -> PureBLAS.geqrf!(A), A)
        throws(A -> PureBLAS.gesvd!(A), A)
        throws((A, B) -> PureBLAS.getrs!(A, ipiv, B), A, randn(T, n, 2))
        throws(A -> PureBLAS.trtri!(A), A)
        throws(A -> PureBLAS.potri!(A), spd(T, n))
    end

    # The backend-layer forwards carry the same guard.
    throws((y, x) -> PureBLAS.axpy!(PureBLAS.DEFAULT_BACKEND, y, 2.0, x), randn(n), randn(n))
    throws((C, A, B) -> PureBLAS.gemm!(PureBLAS.DEFAULT_BACKEND, C, A, B), randn(n, n), randn(n, n), randn(n, n))
end

@testitem "plain arrays still work after the offset guard" begin
    using PureBLAS, LinearAlgebra
    n = 6
    @testset "$T" for T in (Float64, ComplexF64)
        x = randn(T, n); y = randn(T, n); a = randn(T)
        A = randn(T, n, n); B = randn(T, n, n); C = randn(T, n, n)
        yy = copy(y); PureBLAS.axpy!(yy, a, x)
        @test yy ≈ y + a * x
        @test PureBLAS.dot(x, y) ≈ dot(x, y)
        @test PureBLAS.nrm2(x) ≈ norm(x)
        xx = copy(x); PureBLAS.scal!(a, xx)
        @test xx ≈ a * x
        @test PureBLAS.gemv(A, x) ≈ A * x
        yy = zeros(T, n); PureBLAS.gemv!(yy, A, x)
        @test yy ≈ A * x
        AA = copy(A); PureBLAS.ger!(a, x, y, AA)
        @test AA ≈ A + a * x * transpose(y)
        U = triu(A) + n * I
        xx = copy(x); PureBLAS.trsv!(U, xx)
        @test U * xx ≈ x
        CC = copy(C); PureBLAS.gemm!(CC, A, B; alpha = one(T), beta = zero(T))
        @test CC ≈ A * B
        BB = copy(B); PureBLAS.trsm!(BB, U)
        @test U * BB ≈ B
        CC = copy(C); PureBLAS.syrk!(CC, A; uplo = 'U', trans = 'N', alpha = one(T), beta = zero(T))
        @test triu(CC) ≈ triu(A * transpose(A))
        G = copy(A); _, ipiv, info = PureBLAS.getrf!(G)
        F = lu(A)
        @test info == 0
        @test ipiv == F.ipiv
        @test G ≈ F.factors
        S = A * A' + n * I
        L = copy(S); PureBLAS.potrf!(L; uplo = 'L')
        @test tril(L) * tril(L)' ≈ S
        Q = copy(A); tau = zeros(T, n); PureBLAS.geqrf!(Q, tau)
        @test abs.(diag(Q)) ≈ abs.(diag(qr(A).R))
        sv = PureBLAS.gesvd!(copy(A); want_vectors = false)[1]
        @test sv ≈ svdvals(A)
    end
end
