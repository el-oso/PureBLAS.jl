@testitem "Native API: conj/unconj dot semantics" begin
    using PureBLAS
    x = randn(ComplexF64, 64); y = randn(ComplexF64, 64)
    @test PureBLAS.dot(x, y) ≈ sum(conj.(x) .* y)   # conjugated (LinearAlgebra.dot convention)
    @test PureBLAS.dotu(x, y) ≈ sum(x .* y)         # unconjugated
end

@testitem "Native API: generic scalar path on strided views" begin
    using PureBLAS
    # A strided SubArray is not a DenseArray, so it takes the generic (non-SIMD) loop.
    base = randn(Float64, 200)
    x = @view base[1:2:end]        # 100 elements, stride 2 — not dense
    y = randn(Float64, length(x))
    yref = y .+ 3.0 .* collect(x)
    PureBLAS.axpy!(y, 3.0, x)
    @test y ≈ yref
    @test PureBLAS.nrm2(x) ≈ sqrt(sum(abs2, collect(x)))
end

@testitem "AD: ForwardDiff differentiates the native kernels (Mode 2 value prop)" begin
    using PureBLAS, ForwardDiff, LinearAlgebra
    x = randn(Float64, 128); v = randn(Float64, 128); w = randn(Float64, 128)
    # nrm2 along a direction: d/dt ||x + t v|| = <x+tv, v> / ||x+tv||
    f(t) = PureBLAS.nrm2(x .+ t .* v)
    @test ForwardDiff.derivative(f, 0.0) ≈ dot(x, v) / norm(x)
    # dot is bilinear: d/da dot(a*x, w) = dot(x, w)
    g(a) = PureBLAS.dot(a .* x, w)
    @test ForwardDiff.derivative(g, 1.0) ≈ dot(x, w)
    # gradient of asum (Σ|xᵢ|) is sign.(x)
    @test ForwardDiff.gradient(PureBLAS.asum, x) ≈ sign.(x)
end

@testitem "Native API: dimension mismatch is caught" begin
    using PureBLAS
    @test_throws DimensionMismatch PureBLAS.axpy!(zeros(3), 1.0, zeros(4))
    @test_throws DimensionMismatch PureBLAS.dot(zeros(3), zeros(4))
end
