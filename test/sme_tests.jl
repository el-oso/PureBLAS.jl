@testitem "SME detection is consistent" begin
    using PureBLAS
    const P = PureBLAS
    # `_SME_F64` gates the Float64 path and must imply a usable tile geometry. A machine that
    # reports the feature but no vector length would otherwise derive a zero-sized tile.
    if P._SME_F64
        @test P._SME_LANES > 0
        @test P._SME_MR == 2 * P._SME_L
        @test P._SME_NR == 2 * P._SME_L
        @test P._SME_MIN >= P._SME_MR
    else
        # Geometry still has to be well formed so the IR strings build on machines without SME.
        @test P._SME_L == 8
    end
end

@testitem "SME block sizes obey their derivation" begin
    using PureBLAS
    const P = PureBLAS
    for (m, n, k) in ((2048, 2048, 2048), (512, 512, 512), (100, 37, 53), (4096, 4096, 4096))
        MC, NC, KC = P._sme_blocks(m, n, k)
        @test 0 < MC <= max(m, P._SME_MR)
        @test 0 < NC
        @test 0 < KC <= k
        @test MC % P._SME_MR == 0
        @test KC % P._SME_L == 0
        # NC spans n so A is packed exactly once -- the reason this path beats the SIMD blocking.
        @test NC >= n
        # The packed B panel stays inside the budget that bounds C re-streaming.
        @test KC * NC * sizeof(Float64) <= P._SME_PANEL_BUDGET + KC * P._SME_NR * sizeof(Float64)
    end
end

@testitem "SME gemm! matches the oracle on ragged shapes" begin
    using PureBLAS, LinearAlgebra
    const P = PureBLAS
    if !P._SME_F64 || P._SME_ENTRY[] === C_NULL
        @test_skip "no SME on this machine"
    else
        # Sizes deliberately off the tile so the zero-filled panel edges and the scratch-tile
        # path are exercised, not just the straight-through one. beta == 0 takes the overwrite
        # path, which writes C directly instead of zeroing it first.
        @testset "m=$m n=$n k=$k alpha=$al beta=$be" for (m, n, k) in (
                (64, 64, 64), (65, 67, 69), (100, 37, 53), (129, 130, 131), (300, 200, 150),
            ),
                (al, be) in ((1.0, 0.0), (1.0, 1.0), (-2.5, 0.75), (0.0, 2.0), (3.0, -1.0))
            A = randn(m, k); B = randn(k, n); C0 = randn(m, n)
            C = copy(C0)
            PureBLAS.gemm!(C, A, B; alpha = al, beta = be)
            want = be .* C0 .+ al .* (A * B)
            @test C ≈ want rtol = 1.0e-12
        end
    end
end

@testitem "SME path is allocation-free at steady state" begin
    using PureBLAS
    const P = PureBLAS
    if !P._SME_F64 || P._SME_ENTRY[] === C_NULL
        @test_skip "no SME on this machine"
    else
        n = max(128, P._SME_MIN)
        A = rand(n, n); B = rand(n, n); C = zeros(n, n)
        f!(C, A, B) = PureBLAS.gemm!(C, A, B; alpha = 1.0, beta = 0.0)
        f!(C, A, B)                       # first call grows the workspace
        @test (@allocated f!(C, A, B)) == 0
    end
end

@testitem "SME eligibility declines what it cannot handle" begin
    using PureBLAS
    const P = PureBLAS
    n = max(128, P._SME_MIN)
    C = zeros(n, n); A = rand(n, n); B = rand(n, n)
    elig(tA, tB, cA, cB) = P._sme_eligible(Float64, n, n, n, tA, tB, cA, cB, C, A, B)
    # The packers read untransposed, unit-row-stride, real Float64 operands only; everything
    # else must fall through to the SIMD path rather than produce a wrong answer.
    @test !elig(true, false, false, false)
    @test !elig(false, true, false, false)
    @test !P._sme_eligible(Float32, n, n, n, false, false, false, false,
                           zeros(Float32, n, n), rand(Float32, n, n), rand(Float32, n, n))
    # Below the crossover the packed panels do not pay for themselves.
    small = P._SME_MIN - 1
    @test !P._sme_eligible(Float64, small, small, small, false, false, false, false,
                           zeros(small, small), rand(small, small), rand(small, small))
end
