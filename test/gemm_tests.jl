@testsetup module GemmOracle
using LinearAlgebra
export gerr, gtol
gerr(a, b) = norm(a .- b) / max(norm(b), eps(Float64))
gtol(::Type{T}) where {T} = T <: Union{Float32, ComplexF32} ? 1.0e-3 : 1.0e-11
end

@testitem "GEMM real (blocked) vs OpenBLAS" setup = [GemmOracle] begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    @testset "$T $tA$tB m=$m n=$n k=$k" for T in (Float32, Float64),
            (tA, tB) in (('N', 'N'), ('T', 'N'), ('N', 'T'), ('T', 'T')),
            m in (1, 16, 17, 40), n in (1, 6, 7, 31), k in (1, 16, 33)

        A = tA == 'N' ? randn(T, m, k) : randn(T, k, m)
        Bm = tB == 'N' ? randn(T, k, n) : randn(T, n, k)
        C0 = randn(T, m, n)
        for (al, be) in ((one(T), zero(T)), (T(0.5), T(2)), (zero(T), T(1.5)))
            Cref = copy(C0); B.gemm!(tA, tB, al, A, Bm, be, Cref)
            Cp = copy(C0); PureBLAS.gemm!(Cp, A, Bm; alpha = al, beta = be, transA = tA, transB = tB)
            @test gerr(Cp, Cref) < gtol(T)
        end
    end
end

@testitem "GEMM blocked path (>unpack threshold) vs OpenBLAS" setup = [GemmOracle] begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    # Sizes above the unpacked threshold (96) exercise the blocked path incl. its masked edge.
    @testset "$T $tA$tB m=$m n=$n k=$k" for T in (Float32, Float64),
            (tA, tB) in (('N', 'N'), ('T', 'N'), ('N', 'T')),
            (m, n, k) in ((100, 100, 100), (130, 97, 113), (160, 200, 128), (97, 150, 99))

        A = tA == 'N' ? randn(T, m, k) : randn(T, k, m)
        Bm = tB == 'N' ? randn(T, k, n) : randn(T, n, k)
        C0 = randn(T, m, n)
        for (al, be) in ((one(T), zero(T)), (T(0.7), T(1.3)))
            Cref = copy(C0); B.gemm!(tA, tB, al, A, Bm, be, Cref)
            Cp = copy(C0); PureBLAS.gemm!(Cp, A, Bm; alpha = al, beta = be, transA = tA, transB = tB)
            @test gerr(Cp, Cref) < gtol(T)
        end
    end
end

@testitem "GEMM beta=0 ignores NaN in C (BLAS semantics)" setup = [GemmOracle] begin
    using PureBLAS
    A = randn(64, 48); B = randn(48, 32)
    C = fill(NaN, 64, 32)
    PureBLAS.gemm!(C, A, B; alpha = 1.0, beta = 0.0)
    @test all(isfinite, C) && gerr(C, A * B) < gtol(Float64)
end

@testitem "GEMM complex (SIMD split-pack + unpacked) vs OpenBLAS" setup = [GemmOracle] begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    # sizes span the routing: tiny→generic (5), unpacked small-n (23,40 — >nr cols & >mr rows to
    # exercise the jr/ir tiling), and >_CGEMM_UNPACK_MAX→blocked (100). n>nr with jr>0 was a real bug.
    @testset "$T $tA$tB $m×$n×$k" for T in (ComplexF32, ComplexF64),
            (tA, tB) in (('N', 'N'), ('C', 'N'), ('N', 'C'), ('T', 'T'), ('C', 'C')),
            (m, n, k) in ((23, 17, 19), (5, 5, 5), (40, 48, 33), (100, 96, 77))

        A = tA == 'N' ? randn(T, m, k) : randn(T, k, m)
        Bm = tB == 'N' ? randn(T, k, n) : randn(T, n, k)
        C0 = randn(T, m, n); al = T(0.7, -0.3); be = T(1.4, 0.2)
        Cref = copy(C0); B.gemm!(tA, tB, al, A, Bm, be, Cref)
        Cp = copy(C0); PureBLAS.gemm!(Cp, A, Bm; alpha = al, beta = be, transA = tA, transB = tB)
        @test gerr(Cp, Cref) < gtol(T)
    end
end

@testitem "GEMM allocating gemm(A,B) == A*B" setup = [GemmOracle] begin
    using PureBLAS
    A = randn(50, 30); B = randn(30, 40)
    @test gerr(PureBLAS.gemm(A, B), A * B) < gtol(Float64)
    @test gerr(PureBLAS.gemm(A, A; transA = 'T'), A' * A) < gtol(Float64)
end

@testitem "GEMM is AD-traceable (generic path, ForwardDiff)" begin
    using PureBLAS, ForwardDiff, LinearAlgebra
    A = randn(8, 5); dA = randn(8, 5); B = randn(5, 6)
    # d/dt sum((A + t·dA)·B) = sum(dA·B)
    f(t) = sum(PureBLAS.gemm(A .+ t .* dA, B))
    @test ForwardDiff.derivative(f, 0.0) ≈ sum(dA * B)
end

@testitem "GEMM steady-state is allocation-free (driver level)" begin
    using PureBLAS
    # unpacked (small, max dim ≤ 96) path — no buffers at all
    A = randn(48, 48); B = randn(48, 48); C = zeros(48, 48)
    PureBLAS.gemm!(C, A, B; alpha = 1.0, beta = 0.0)            # warmup/compile
    @test (@allocated PureBLAS.gemm!(C, A, B; alpha = 1.0, beta = 0.0)) == 0
    @test (@allocated PureBLAS.gemm!(C, A, B; alpha = 2.0, beta = 1.0)) == 0  # beta≠0 branch
    # blocked (large) path — scratch allocated on first call, then reused → 0 thereafter
    Al = randn(300, 300); Bl = randn(300, 300); Cl = zeros(300, 300)
    PureBLAS.gemm!(Cl, Al, Bl; beta = 0.0)                       # warmup (allocates scratch)
    @test (@allocated PureBLAS.gemm!(Cl, Al, Bl; beta = 0.0)) == 0
end

@testitem "GEMM dimension mismatch is caught" begin
    using PureBLAS
    @test_throws DimensionMismatch PureBLAS.gemm!(zeros(3, 3), zeros(3, 4), zeros(5, 3))
end
