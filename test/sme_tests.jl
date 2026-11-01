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
        # MC, like KC below, is tile-aligned ONLY when it is a genuine split: a single block takes the
        # true m and the kernel predicates its last row tile, so m=100 here must NOT be rounded to 96.
        @test MC == m || MC % P._SME_MR == 0
        # KC is lane-aligned ONLY when it is a genuine split. A single block takes the true k, and
        # rounding it there would turn any `k % _SME_L != 0` into two blocks — a second pack, a
        # second pass over C, and every ragged edge tile again. The pack zero-fills `kce` to `kpad`,
        # so an unrounded single block is exact; k=53 here is the case that must NOT be rounded.
        @test KC == k || KC % P._SME_L == 0
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
        # Sizes deliberately off the tile so the zero-filled panel edges and the predicated edge
        # tiles are exercised, not just the straight-through one. beta == 0 takes the overwrite
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
        # FLOAT32 HAS ITS OWN ROUTE, with its own feature gate, its own tile floor and its own entry
        # pointer, so the predicate admits it on its own terms instead of falling through. `n` here
        # is a multiple of both register tiles and above both floors.
        f32(cA, cB, nn) = P._sme_eligible(Float32, nn, nn, nn, false, false, cA, cB,
            zeros(Float32, nn, nn), rand(Float32, nn, nn), rand(Float32, nn, nn))
        if P._SME_F32 && P._SME_ENTRY32[] !== C_NULL
            @test f32(false, false, n)
            # Conjugation is declined at BOTH precisions: these kernels are real.
            @test !f32(true, false, n)
            @test !f32(false, true, n)
            # THE TWO FLOORS ARE NOT THE SAME FLOOR. A 32-bit register tile is twice as wide, so half
            # the extents that divide the 64-bit tile leave the 32-bit one ragged -- and a ragged
            # Float32 shape must clear a higher floor. At `_SME_MR32 + _SME_MR` the Float64 predicate
            # takes its tile-exact path while Float32 declines, which is what makes borrowing either
            # precision's guard for the other a wrong answer.
            sq = P._SME_MR32 + P._SME_MR
            @test P._sme_tile_ok(sq, sq)
            @test !f32(false, false, sq)
        else
            @test !f32(false, false, n)
        end
        # Below the crossover the packed panels do not pay for themselves. The crossover is
        # `_SME_MIN_EXACT` for EVERY operand, ragged or not: a ragged block is one macrokernel call
        # whose edge tiles are predicated, so it no longer pays a prologue per edge tile and no
        # longer needs a higher floor. Measured at the sizes that separates — n=40
        # loses 0.81x and stays out, n=50 wins 1.25x and comes in.
        small = P._SME_MIN_EXACT - 1
        @test !P._sme_eligible(Float64, small, small, small, false, false, false, false,
                               zeros(small, small), rand(small, small), rand(small, small))
        # A shape too NARROW for the tile geometry is still declined however large its other side:
        # `min >= 2*_SME_MR` is a geometry bound, not a crossover, and padding does not lift it.
        @test !P._sme_eligible(Float64, 8, 4096, 4096, false, false, false, false,
                               zeros(8, 4096), rand(8, 4096), rand(4096, 4096))
    end
end

@testitem "SME computes every transpose combination correctly" begin
    using PureBLAS, LinearAlgebra
    const P = PureBLAS
    if !P._SME_F64 || P._SME_ENTRY[] === C_NULL
        @test_skip "no SME F64 on this machine"
    else
        # Ragged as well as square: the predicated edge tiles handle the m/n/k remainders, and a shape
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
        # A row count that is not a whole number of blocks is admitted for EITHER beta. Store mode
        # finishes with an overlapping block, which recomputes shared rows and is sound only because
        # it writes them; accumulate mode finishes with a scalar pass that adds each leftover row once.
        # Requiring beta == 0 here used to cost 4-7x on `y += A*x` at a ragged m — see the predicate.
        mr = m + 1
        @test P._sme_gemv_eligible(Float64, mr, n, false, false, A, x, y, 1, 1, 0.0)
        @test P._sme_gemv_eligible(Float64, mr, n, false, false, A, x, y, 1, 1, 1.0)
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

# A ROUTE WITH NO WITNESS CANNOT BE A/B'd. A predicate that silently declines and a kernel that is
# genuinely no faster produce the same null result, so a timing comparison over the gemv path means
# nothing without a counter that says the coprocessor ran. `_SME_CALLS` does NOT serve: it is bumped
# only inside `_gemm_sme!`, and asking it about a gemv answers zero whatever happens.
#
# The negative case is the control, and it is the half that matters. A witness that fires for
# everything is worthless evidence, and one that fires for nothing reads exactly like a route that
# is never taken — which is how the missing counter cost a day of trmv measurement.
@testitem "SME gemv carries its own execution witness" tags = [:checks] begin
    using PureBLAS
    P = PureBLAS
    if !P._SME_F64
        @test P._SME_GEMV_CALLS[] == 0                 # off SME the kernel does not exist to run
    else
        n = 1024
        A = randn(n, n); x = randn(n); y = zeros(n)
        P.gemv!(y, A, x)                               # warm, and prove the route is reachable
        b = P._SME_GEMV_CALLS[]
        P.gemv!(y, A, x)
        @test P._SME_GEMV_CALLS[] == b + 1             # POSITIVE: an eligible shape bumps it once

        # NEGATIVE control: transposed is outside `_sme_gemv_eligible` (the kernel reads columns of
        # A directly), so the counter must NOT move. Without this the positive case above would
        # still pass if the counter were bumped unconditionally somewhere upstream.
        At = collect(transpose(A))
        P.gemv!(y, At, x; trans = 'T')
        b2 = P._SME_GEMV_CALLS[]
        P.gemv!(y, At, x; trans = 'T')
        @test P._SME_GEMV_CALLS[] == b2                # NEGATIVE: an ineligible shape does not

        # And the witness agrees with the predicate rather than being a second opinion.
        @test P._sme_gemv_eligible(Float64, n, n, false, false, A, x, y, 1, 1, 0.0)
        @test !P._sme_gemv_eligible(Float64, n, n, true, false, At, x, y, 1, 1, 0.0)
    end
end

# ── Float32 route ──────────────────────────────────────────────────────────────────────────────
@testitem "SME Float32 geometry follows the detected lane count" begin
    using PureBLAS
    const P = PureBLAS
    # ZA is `svl` x `svl` BYTES, so a 32-bit tile is twice the dimension of a 64-bit one on the
    # same hardware and the register tile is twice as wide in each direction.
    @test P._SME_MR32 == 2 * P._SME_L32
    @test P._SME_NR32 == 2 * P._SME_L32
    if P._SME_F32
        @test P._SME_LANES_F32 > 0
        @test P._SME_LANES_F32 == 2 * P._SME_LANES
        @test P._SME_L32 == 2 * P._SME_L
        # The floor is a multiple of the register tile, not a Float32 literal, so it tracks it.
        @test P._SME_MIN_EXACT32 == 3 * P._SME_MR32
    else
        # Geometry stays well formed off the feature so the IR strings still build.
        @test P._SME_L32 == 16
    end
end

@testitem "SME Float32 block sizes obey the same derivation" begin
    using PureBLAS
    const P = PureBLAS
    # The SAME formulas as the Float64 case, with `sizeof(T)` and the register tile substituted —
    # which is the point: one derivation serves both precisions, so neither carries a literal.
    for (m, n, k) in ((2048, 2048, 2048), (512, 512, 512), (100, 37, 53), (4096, 4096, 4096))
        MC, NC, KC = P._sme_blocks(Float32, m, n, k)
        @test 0 < MC <= max(m, P._SME_MR32)
        @test 0 < KC <= k
        # Tile-aligned ONLY when the block is a genuine split; a single block takes the true extent
        # and the kernel predicates its last tile. m=100 and k=53 are the cases that must not round.
        @test MC == m || MC % P._SME_MR32 == 0
        @test KC == k || KC % P._SME_L32 == 0
        @test NC >= n
        @test KC * NC * sizeof(Float32) <= P._SME_PANEL_BUDGET + KC * P._SME_NR32 * sizeof(Float32)
    end
end

@testitem "SME Float32 tile floor admits one whole tile and declines below it" begin
    using PureBLAS
    const P = PureBLAS
    MR = P._SME_MR32
    # ONE whole register tile each way is the admission criterion for a tile-exact shape, measured
    # at k = 10, 64 and 256 (the table on `_sme_tile_ok32`): one tile wins at every depth and
    # nothing below it does. This is where the Float32 floor parts company with the Float64 one,
    # which needs two tiles -- so it is asserted rather than assumed to follow.
    @test P._sme_tile_ok32(MR, MR)
    @test P._sme_tile_ok32(2 * MR, 2 * MR)
    @test !P._sme_tile_ok32(MR - 1, 4096)
    # A RAGGED shape has to clear the higher floor instead, because its predicated edge tile can
    # cost more than the tile earns at small k -- 1.25 and 1.5 tiles measure 0.34x and 0.48x there.
    @test !P._sme_tile_ok32(MR + 8, MR + 8)
    @test !P._sme_tile_ok32(2 * MR + 1, 2 * MR + 1)
    @test P._sme_tile_ok32(MR + 8, P._SME_MIN_EXACT32)
    # THE TWO FLOORS DIVERGE, and the clearest witness is an extent that is tile-exact for Float64
    # and ragged for Float32 -- a 32-bit tile is twice as wide, so half the extents that divide the
    # 64-bit tile do not divide it. At `_SME_MR32 + _SME_MR` the Float64 predicate takes its exact
    # path and the Float32 one falls to its ragged floor and declines.
    @test P._sme_tile_ok(P._SME_MR32 + P._SME_MR, P._SME_MR32 + P._SME_MR)
    @test !P._sme_tile_ok32(P._SME_MR32 + P._SME_MR, P._SME_MR32 + P._SME_MR)
end

@testitem "SME Float32 gemm! matches the oracle on ragged shapes" begin
    using PureBLAS, LinearAlgebra
    const P = PureBLAS
    if !P._SME_F32 || P._SME_ENTRY32[] === C_NULL
        @test_skip "no SME Float32 route on this machine"
    else
        # The reference accumulates in Float64, so this compares against the EXACT product rather
        # than against another Float32 summation order — a kernel that merely reorders the sum
        # would otherwise pass. The tolerance is a Float32 one, scaled by depth.
        MR = P._SME_MR32
        @testset "m=$m n=$n k=$k tA=$tA alpha=$al beta=$be" for (m, n, k) in (
                    (2MR, 2MR, P._SME_L32), (2MR + 1, 2MR + 3, P._SME_L32 + 1),
                    (3MR, 2MR, 10), (2MR, 3MR + 5, 64), (2MR + 7, 2MR + 11, 17),
                ),
                tA in (false, true),
                (al, be) in ((1.0f0, 0.0f0), (1.5f0, 0.5f0), (0.0f0, 1.0f0))
            A = tA ? rand(Float32, k, m) : rand(Float32, m, k)
            B = rand(Float32, k, n)
            C = rand(Float32, m, n)
            got = copy(C)
            P.gemm!(got, A, B; alpha = al, beta = be, transA = tA ? 'T' : 'N', transB = 'N')
            opA = tA ? transpose(Float64.(A)) : Float64.(A)
            want = Float64(al) .* (opA * Float64.(B)) .+ Float64(be) .* Float64.(C)
            den = maximum(abs, want)
            @test maximum(abs.(Float64.(got) .- want)) <= 1.0e-5 * max(den, 1.0)
        end
    end
end

@testitem "SME Float32 eligible calls decline the worker pool" begin
    using PureBLAS
    const P = PureBLAS
    if !P._SME_F32 || P._SME_ENTRY32[] === C_NULL
        @test_skip "no SME Float32 route on this machine"
    else
        # The matrix unit is shared by the cluster, so splitting an SME call across workers queues
        # them behind one unit instead of dividing the work. `_sme_owns` at the threaded entry is
        # the only place that can be decided — a guard inside the chunk body runs after the split.
        MR = P._SME_MR32
        C = zeros(Float32, 4MR, 4MR); A = rand(Float32, 4MR, 64); B = rand(Float32, 64, 4MR)
        @test P._sme_owns(Float32, 4MR, 4MR, 64, false, false, false, false, C, A, B)
        # One whole tile each way is ON the floor, so the route owns it and the pool stays out.
        @test P._sme_owns(Float32, MR, MR, 64, false, false, false, false, C, A, B)
        # Below the floor the route declines and the pool is free to split as usual.
        @test !P._sme_owns(Float32, MR - 1, MR - 1, 64, false, false, false, false, C, A, B)
        # A conjugated operand is not this kernel's shape at either precision.
        @test !P._sme_owns(Float32, 4MR, 4MR, 64, false, false, true, false, C, A, B)
        # THE GUARD REACHES BEYOND gemm, AND THAT IS THE POINT RATHER THAN A SIDE EFFECT. `symm`
        # asks the same predicate before sizing its pool, and its own test already admits Float32,
        # so a Float32 symm whose sub-gemm the route owns declines the split for the same reason:
        # otherwise each worker chunk reaches the route separately and contends for one shared unit.
        Cs = zeros(Float32, 4MR, 4MR); Bs = rand(Float32, 4MR, 4MR)
        @test P._symm_sme_owns(Float32, 4MR, 4MR, 4MR, Cs, Bs)
        @test !P._symm_sme_owns(Float32, MR - 1, MR - 1, MR - 1,
                                zeros(Float32, MR - 1, MR - 1), rand(Float32, MR - 1, MR - 1))
    end
end

@testitem "SME Float32 results are identical across worker counts" begin
    using PureBLAS
    const P = PureBLAS
    if !P._SME_F32 || P._SME_ENTRY32[] === C_NULL
        @test_skip "no SME Float32 route on this machine"
    else
        # req#11: the result may not depend on how many workers ran it. Adding a route is exactly
        # the change that breaks this, by keying a predicate on a chunk's own extent instead of on
        # the route token — so both a shape the route OWNS and one just under its floor are checked,
        # because the second is the one that actually gets split.
        MR = P._SME_MR32
        using Base.Threads: nthreads
        if nthreads() < 2
            @test_skip "needs >= 2 julia threads to split anything"
        else
            for (m, n, k) in ((4MR, 4MR, 64), (2MR - 1, 4MR, 64))
                A = rand(Float32, m, k); B = rand(Float32, k, n)
                P.set_num_threads(1)
                base = zeros(Float32, m, n)
                P.gemm!(base, A, B)
                for nt in 2:min(4, nthreads())
                    P.set_num_threads(nt)
                    got = zeros(Float32, m, n)
                    P.gemm!(got, A, B)
                    @test got == base
                end
            end
            P.set_num_threads(1)
        end
    end
end
