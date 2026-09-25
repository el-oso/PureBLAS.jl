    # Non-unit increments: the kernel indexes both vectors contiguously.
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
    # THE POSITIVE CASES NEED THE HARDWARE. Off SME, `_SME_F64` is a compile-time false and every
    # predicate here correctly answers `false`, so `@test elig(...)` asserts the opposite of what
    # the machine can do. The negative cases would still hold, but they prove nothing once the
    # guard they are meant to exercise has already short-circuited.
    if !P._SME_F64
        @test_skip "no SME F64 on this machine ($(Sys.ARCH))"
    else
        n = max(128, P._SME_MIN)
        C = zeros(n, n); A = rand(n, n); B = rand(n, n)
        elig(tA, tB, cA, cB) = P._sme_eligible(Float64, n, n, n, tA, tB, cA, cB, C, A, B)
        # Transposed operands ARE handled: the packers read either orientation, which is what lets the
        # triangular and symmetric routines reach this path at all -- `_rank_k` always issues one
        # transposed operand, and symm issues `(false, tr)`.
        @test elig(true, false, false, false)
        @test elig(false, true, false, false)
        @test elig(true, true, false, false)
        # Conjugation is not: the kernel is real Float64, so a conjugated operand must fall through to
        # the SIMD path rather than produce a wrong answer.
        @test !elig(false, false, true, false)
        @test !elig(false, false, false, true)
        @test !P._sme_eligible(Float32, n, n, n, false, false, false, false,
                               zeros(Float32, n, n), rand(Float32, n, n), rand(Float32, n, n))
        # Below the crossover the packed panels do not pay for themselves.
        small = P._SME_MIN - 1
        @test !P._sme_eligible(Float64, small, small, small, false, false, false, false,
                               zeros(small, small), rand(small, small), rand(small, small))
    end
end

@testitem "SME computes every transpose combination correctly" begin
    using PureBLAS, LinearAlgebra
    const P = PureBLAS
    if !P._SME_F64 || P._SME_ENTRY[] === C_NULL
        @test_skip "no SME F64 on this machine"
    else
        # Ragged as well as square: the edge macrokernel handles the m/n/k remainders, and a shape
        # that divides the block sizes exercises none of it.
        for (m, n, k) in ((300, 300, 300), (257, 193, 129), (512, 128, 320), (129, 512, 97))
            for ta in ('N', 'T'), tb in ('N', 'T')
                A = ta == 'N' ? randn(m, k) : randn(k, m)
                B = tb == 'N' ? randn(k, n) : randn(n, k)
                C = zeros(m, n)
                P.gemm!(C, A, B; transA = ta, transB = tb, alpha = 1.0, beta = 0.0)
                R = (ta == 'N' ? A : transpose(A)) * (tb == 'N' ? B : transpose(B))
                @test maximum(abs, C .- R) / maximum(abs, R) < 1e-13
            end
        end
    end
end

@testitem "SME does not disturb Mode 2 differentiability" begin
    using PureBLAS, ForwardDiff, LinearAlgebra
    const P = PureBLAS
    # Mode 2 differentiates by running `Dual` through the real kernels as an interleaved pair,
    # not through a reverse rule. The SME path is Float64-only, so `Dual` operands must keep
    # taking the generic path -- at a size where Float64 WOULD be routed to SME, to catch a
    # future eligibility test that forgets to check the element type.
    n = max(96, P._SME_MIN + 16)
    A0 = randn(n, n); B0 = randn(n, n)
    @test !P._sme_eligible(eltype(ForwardDiff.Dual{Nothing, Float64, 1}[]), n, n, n,
                           false, false, false, false, A0, A0, B0) skip = false

    # d/dt tr(A(t) * B) at t = 0 with A(t) = A0 + t*E is tr(E * B), computed independently.
    E = randn(n, n)
    f(t) = begin
        A = A0 .+ t .* E
        C = zeros(typeof(t), n, n)
        PureBLAS.gemm!(C, A, Matrix{typeof(t)}(B0); alpha = one(t), beta = zero(t))
        tr(C)
    end
    got = ForwardDiff.derivative(f, 0.0)
    want = tr(E * B0)
    @test got ≈ want rtol = 1.0e-10

    # And the value at t = 0 still matches the Float64 product, which DOES take the SME path.
    C64 = zeros(n, n)
    PureBLAS.gemm!(C64, A0, B0; alpha = 1.0, beta = 0.0)
    @test tr(C64) ≈ tr(A0 * B0) rtol = 1.0e-12
end

@testitem "SME gemv: eligibility declines what it cannot handle" begin
    using PureBLAS
    const P = PureBLAS
    # THE POSITIVE CASES NEED THE HARDWARE. Off SME, `_SME_F64` is a compile-time false and every
    # predicate here correctly answers `false`, so `@test elig(...)` asserts the opposite of what
    # the machine can do. The negative cases would still hold, but they prove nothing once the
    # guard they are meant to exercise has already short-circuited.
    if !P._SME_F64
        @test_skip "no SME F64 on this machine ($(Sys.ARCH))"
    else
        m = 4 * P._SME_GEMV_BLK; n = max(64, P._SME_GEMV_MINWORK ÷ m + 1)
        A = rand(m, n); x = rand(n); y = zeros(m)
        el(mm, nn, tr, cj, b) = P._sme_gemv_eligible(Float64, mm, nn, tr, cj, A, x, y, 1, 1, b)
        @test el(m, n, false, false, 0.0)
        # Transposed and conjugated forms read A the other way; the kernel streams columns only.
        @test !el(m, n, true, false, 0.0)
        @test !el(m, n, false, true, 0.0)
        # Float32 and complex share no kernel with this path.
        @test !P._sme_gemv_eligible(Float32, m, n, false, false,
                                    rand(Float32, m, n), rand(Float32, n), zeros(Float32, m), 1, 1, 0.0)
        # Non-unit increments: the kernel indexes both vectors contiguously.
        @test !P._sme_gemv_eligible(Float64, m, n, false, false, A, x, y, 2, 1, 0.0)
        @test !P._sme_gemv_eligible(Float64, m, n, false, false, A, x, y, 1, 2, 0.0)
        # Below the work floor the ZA fill and readback are not amortized.
        @test !el(P._SME_GEMV_BLK, 1, false, false, 0.0)
        # A row count that is not a whole number of blocks needs beta == 0: its tail is an OVERLAPPING
        # block, which recomputes shared rows, and that is only sound when they are stored not added.
        mr = m + 1
        @test P._sme_gemv_eligible(Float64, mr, n, false, false, A, x, y, 1, 1, 0.0)
        @test !P._sme_gemv_eligible(Float64, mr, n, false, false, A, x, y, 1, 1, 1.0)
    end
end

@testitem "SME gemv matches the reference over shapes, alphas and betas" begin
    using PureBLAS, LinearAlgebra
    const P = PureBLAS
    if !P._SME_F64 || P._SME_GEMV_ENTRY[] === C_NULL
        @test_skip "no SME F64 on this machine"
    else
        # Row counts on both sides of the block boundary, since the tail is a separate path, and a
        # beta that is not a power of two so an alpha/beta placement difference cannot hide.
        for m in (256, 257, 300, 512, 1000), n in (32, 64, 257)
            for (al, be) in ((1.0, 0.0), (0.75, 0.0), (0.75, 1.0), (2.5, -0.3), (0.0, 0.5))
                A = randn(m, n); x = randn(n); y0 = randn(m)
                want = al .* (A * x) .+ be .* y0
                y = copy(y0); P.gemv!(y, A, x; alpha = al, beta = be)
                @test maximum(abs, y .- want) / max(1e-300, maximum(abs, want)) < 1e-12
            end
        end
    end
end

@testitem "SME gemv is allocation-free at steady state" begin
    using PureBLAS
    const P = PureBLAS
    # Through a function barrier, per req#10: at module scope the operands are `Any`-typed and the
    # first compile of the call site caches a method instance no real caller pays for.
    steady(f, args...; kw...) = (f(args...; kw...); @allocated f(args...; kw...))
    A = randn(1024, 1024); x = randn(1024); y = zeros(1024)
    @test steady(P.gemv!, y, A, x; alpha = 1.0, beta = 0.0) == 0
    @test steady(P.gemv!, y, A, x; alpha = 0.75, beta = 1.0) == 0
end

@testitem "SME IR adapts to the streaming vector length" begin
    using PureBLAS
    const P = PureBLAS
    # Runs on EVERY machine, SME or not: the generators are pure string building and the geometry is
    # an argument. This is the only check of the 256- and 1024-bit shapes that needs no such
    # hardware, and it is where a hardcoded offset shows up -- the gemv IR carried the 8-lane
    # geometry as literals (`32*g`, `8*k`) while the gemm IR interpolated it, so a machine with a
    # different vector length would have addressed half the stride and returned wrong numbers.
    for L in (4, 8, 16)
        ngmax = 2L                      # ZA holds 8L vectors; a vgx4 group index is reduced mod 2L
        ir = P._gemv_ir(ngmax, true, L)
        # Last group's column pointer and last readback slot must both scale with L.
        @test occursin("i64 $(4L * (ngmax - 1))", ir)
        @test occursin("i64 $(4L * (ngmax - 1) + 3L)", ir)
        # One group per `fmla`, addressed consecutively from zero — NOT strided by four, which is
        # the wrap this bound exists to prevent.
        for g in 0:(ngmax - 1)
            @test occursin("fmla.single.vg1x4.nxv2f64(i32 $g,", ir)
        end
        # A group count above what ZA can address must be refused, not silently aliased.
        @test_throws ArgumentError P._gemv_ir(2 * ngmax, true, L)
    end
    # The ladder itself is derived, not a literal list.
    @test P._SME_GEMV_NGMAX == 2 * P._SME_L
    @test P._SME_GEMV_BLK == 4 * P._SME_L
    @test all(<=(P._SME_GEMV_NGMAX), P._SME_GEMV_NGS)
    @test P._SME_GEMV_NGMAX in P._SME_GEMV_NGS
end

@testitem "SME self-test runs and can fail" begin
    using PureBLAS
    const P = PureBLAS
    if !P._SME_F64 || P._SME_ENTRY[] === C_NULL
        @test_skip "no SME F64 on this machine"
    else
        # `_sme_init!` runs this before publishing the pointers; a kernel that builds is not a
        # kernel that is correct, and a wrong geometry produces numbers rather than an exception.
        @test P._sme_selftest() < 1e-12
        # POSITIVE CONTROL: the comparison must be able to see a row permutation, which is what
        # asymmetric `i + 1000j` data is for. Symmetric data hides it entirely.
        m = P._SME_GEMV_BLK * 3; n = 7
        A = [i + 1000.0 * j for i in 1:m, j in 1:n]; x = [1.0 + 0.25j for j in 1:n]
        want = A * x
        swapped = copy(want); swapped[1], swapped[2] = swapped[2], swapped[1]
        @test maximum(abs, swapped .- want) / maximum(abs, want) > 1e-12
    end
end
