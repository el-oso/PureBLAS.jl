@testmodule L2Oracle begin
using LinearAlgebra
export l2err, l2tol
l2err(a, b) = norm(a .- b) / max(norm(b), eps(Float64))
l2tol(::Type{T}) where {T} = T <: Union{Float32, ComplexF32} ? 1.0e-3 : 1.0e-10
end

@testitem "gemv vs OpenBLAS" setup = [L2Oracle] begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    @testset "$T $tr m=$m n=$n" for T in (Float32, Float64, ComplexF32, ComplexF64),
            tr in ('N', 'T', 'C'), m in (1, 5, 16, 17, 40), n in (1, 7, 16, 33)

        A = randn(T, m, n)
        xlen = tr == 'N' ? n : m; ylen = tr == 'N' ? m : n
        x = randn(T, xlen); y0 = randn(T, ylen)
        for (al, be) in ((one(T), zero(T)), (T(0.7), T(1.3)), (zero(T), T(2)))
            yref = copy(y0); B.gemv!(tr, al, A, x, be, yref)
            yp = copy(y0); PureBLAS.gemv!(yp, A, x; alpha = al, beta = be, trans = tr)
            @test l2err(yp, yref) < l2tol(T)
        end
    end
end

# REGRESSION. Complex gemv trans='T'/'C' threw a MethodError for every A LARGER THAN L2 on any box
# where `_cgemvt_fits(8, true)` holds -- i.e. by default on AVX-512, not as a rare tuner outcome.
# `_gemv_tc_cmplx!` dispatched `_gemv_tc_run!(..., Val(NC), Val(HALF), Val(_GEMVT_U), Val(_GEMVT_PF))`,
# but U and PF are knobs of the REAL blocked kernel: the complex block kernel takes only (NC, CJ, HALF)
# and no such `_gemv_tc_run!` method has ever existed. Live since 2026-08-08 (cfa09a4 added U to these
# call sites, a411e57 added PF) and invisible to the suite because every existing gemv case is at most
# 40x33 -- comfortably L2-resident, so the ladder never left its first branch. The tuner's own harness
# calls the 10-arg form, so it did not trip either. Found only when a full fleet sweep dropped
# zgemvT/zgemvC out of the cache entirely.
# The size here is DERIVED from the detected L2 so the test stays past the branch on every box.
@testitem "gemv complex T/C past L2 (tuner arm dispatch)" setup = [L2Oracle] begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    @testset "$T $tr" for T in (ComplexF32, ComplexF64), tr in ('T', 'C')
        n = max(64, ceil(Int, sqrt(2 * PureBLAS._L2_BYTES / sizeof(T))))   # A is ~2x L2 => past the split
        @test n * n * sizeof(T) > PureBLAS._L2_BYTES                        # the branch is actually taken
        A = randn(T, n, n); x = randn(T, n); y0 = randn(T, n)
        for (al, be) in ((one(T), zero(T)), (T(0.7), T(1.3)))
            yref = copy(y0); B.gemv!(tr, al, A, x, be, yref)
            yp = copy(y0); PureBLAS.gemv!(yp, A, x; alpha = al, beta = be, trans = tr)
            @test l2err(yp, yref) < l2tol(T)
        end
    end
    # Every arm the ladder can dispatch must HAVE a method. This is the invariant the bug violated, and
    # it is checked statically so it holds regardless of which cfg this process's tuner happens to pick.
    @testset "all cfg arms are callable" begin
        T = ComplexF64
        A = randn(T, 4, 4); x = randn(T, 4); y = zeros(T, 4); a = one(T); b = zero(T)
        for (nc, half) in ((8, true), (4, true), (4, false), (2, false),
                (PureBLAS._CGEMVT_NC, PureBLAS._CGEMVT_HALF)), cj in (false, true)
            @test applicable(PureBLAS._gemv_tc_run!, 4, 4, a, A, x, b, y, Val(cj), Val(nc), Val(half))
        end
    end
end

@testitem "ger (geru/gerc) vs explicit outer product" setup = [L2Oracle] begin
    using PureBLAS, LinearAlgebra
    # Oracle is the explicit rank-1 update (LinearAlgebra.BLAS has geru! but not gerc!):
    #   geru: A += α·x·yᵀ (transpose) ;  gerc: A += α·x·yᴴ (adjoint, conjugates y).
    @testset "$T m=$m n=$n" for T in (Float32, Float64, ComplexF32, ComplexF64),
            m in (1, 16, 17, 40), n in (1, 7, 16, 33)

        x = randn(T, m); y = randn(T, n); A0 = randn(T, m, n); al = randn(T)
        Ap = copy(A0); PureBLAS.ger!(al, x, y, Ap)                 # geru
        @test l2err(Ap, A0 .+ al .* (x * transpose(y))) < l2tol(T)
        Ap2 = copy(A0); PureBLAS.ger!(al, x, y, Ap2; conj = true)  # gerc
        @test l2err(Ap2, A0 .+ al .* (x * y')) < l2tol(T)
    end
end

@testitem "gemv beta=0 ignores NaN; allocating gemv == op(A)·x" setup = [L2Oracle] begin
    using PureBLAS
    A = randn(40, 24); x = randn(24); xt = randn(40)
    y = fill(NaN, 40)
    PureBLAS.gemv!(y, A, x; alpha = 1.0, beta = 0.0)
    @test all(isfinite, y) && l2err(y, A * x) < l2tol(Float64)
    @test l2err(PureBLAS.gemv(A, x), A * x) < l2tol(Float64)
    @test l2err(PureBLAS.gemv(A, xt; trans = 'T'), A' * xt) < l2tol(Float64)
end

@testitem "gemv generic path (strided / non-dense)" setup = [L2Oracle] begin
    using PureBLAS, LinearAlgebra
    A = randn(50, 30)
    x = @view randn(60)[1:2:end]   # 30-elt strided view → generic path
    y = zeros(50)
    PureBLAS.gemv!(y, A, x; alpha = 2.0, beta = 0.0)
    @test l2err(y, 2.0 .* (A * collect(x))) < l2tol(Float64)
end

@testitem "gemv/ger AD-traceable (ForwardDiff)" begin
    using PureBLAS, ForwardDiff, LinearAlgebra
    A = randn(8, 5); x = randn(5); dx = randn(5)
    @test ForwardDiff.derivative(t -> sum(PureBLAS.gemv(A, x .+ t .* dx)), 0.0) ≈ sum(A * dx)
    # ger: d/dt sum(α·x·(y+t·dy)ᵀ) = α·(Σxᵢ)(Σdyⱼ)
    xx = randn(8); y = randn(7); dy = randn(7); a = 1.3
    h(t) = (M = zeros(typeof(t), 8, 7); PureBLAS.ger!(a, xx, y .+ t .* dy, M); sum(M))
    @test ForwardDiff.derivative(h, 0.0) ≈ a * sum(xx) * sum(dy)
end

@testitem "gemv/ger dimension mismatch is caught" begin
    using PureBLAS
    @test_throws DimensionMismatch PureBLAS.gemv!(zeros(3), zeros(3, 4), zeros(3))
    @test_throws DimensionMismatch PureBLAS.ger!(1.0, zeros(3), zeros(4), zeros(3, 5))
end

@testitem "symv vs Symmetric·x" setup = [L2Oracle] begin
    using PureBLAS, LinearAlgebra
    @testset "$T uplo=$ul n=$n" for T in (Float32, Float64, ComplexF32, ComplexF64),
            ul in ('U', 'L'), n in (1, 5, 16, 17, 40)

        A = randn(T, n, n); x = randn(T, n); y0 = randn(T, n)
        S = Symmetric(A, ul == 'U' ? :U : :L)   # oracle reads the same triangle PureBLAS does
        for (al, be) in ((one(T), zero(T)), (T(0.7), T(1.3)), (zero(T), T(2)))
            yp = copy(y0); PureBLAS.symv!(yp, A, x; uplo = ul, alpha = al, beta = be)
            @test l2err(yp, al .* (S * x) .+ be .* y0) < l2tol(T)
        end
    end
end

@testitem "hemv vs Hermitian·x" setup = [L2Oracle] begin
    using PureBLAS, LinearAlgebra
    @testset "$T uplo=$ul n=$n" for T in (ComplexF32, ComplexF64, Float64),
            ul in ('U', 'L'), n in (1, 5, 16, 17, 40)

        A = randn(T, n, n); x = randn(T, n); y0 = randn(T, n)
        H = Hermitian(A, ul == 'U' ? :U : :L)   # Hermitian forces real diagonal — hemv matches
        for (al, be) in ((one(T), zero(T)), (T(0.7), T(1.3)), (zero(T), T(2)))
            yp = copy(y0); PureBLAS.hemv!(yp, A, x; uplo = ul, alpha = al, beta = be)
            @test l2err(yp, al .* (H * x) .+ be .* y0) < l2tol(T)
        end
    end
end

@testitem "symv beta=0 ignores NaN; generic strided path; AD" setup = [L2Oracle] begin
    using PureBLAS, LinearAlgebra, ForwardDiff
    A = randn(40, 40); x = randn(40)
    y = fill(NaN, 40); PureBLAS.symv!(y, A, x; uplo = 'U', alpha = 1.0, beta = 0.0)
    @test all(isfinite, y) && l2err(y, Symmetric(A, :U) * x) < l2tol(Float64)
    xs = @view randn(80)[1:2:end]              # strided x → generic path
    ys = zeros(40); PureBLAS.symv!(ys, A, xs; uplo = 'L', alpha = 2.0, beta = 0.0)
    @test l2err(ys, 2.0 .* (Symmetric(A, :L) * collect(xs))) < l2tol(Float64)
    dx = randn(40)                              # ForwardDiff through symv (generic scalar path)
    @test ForwardDiff.derivative(t -> sum(PureBLAS.symv!(zeros(eltype(t), 40), A, x .+ t .* dx; uplo = 'U')), 0.0) ≈
        sum(Symmetric(A, :U) * dx)
end

@testitem "symv/hemv dimension mismatch is caught" begin
    using PureBLAS
    @test_throws DimensionMismatch PureBLAS.symv!(zeros(3), zeros(3, 4), zeros(3))
    @test_throws DimensionMismatch PureBLAS.hemv!(zeros(3), zeros(4, 4), zeros(3))
end

@testitem "trmv vs OpenBLAS" setup = [L2Oracle] begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    @testset "$T $ul$tr$dg n=$n" for T in (Float32, Float64, ComplexF32, ComplexF64),
            ul in ('U', 'L'), tr in ('N', 'T', 'C'), dg in ('N', 'U'), n in (1, 5, 16, 17, 40)

        A = randn(T, n, n); x0 = randn(T, n)
        xr = copy(x0); B.trmv!(ul, tr, dg, A, xr)
        xp = copy(x0); PureBLAS.trmv!(A, xp; uplo = ul, trans = tr, diag = dg)
        @test l2err(xp, xr) < l2tol(T)
    end
end

@testitem "trsv vs OpenBLAS" setup = [L2Oracle] begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    @testset "$T $ul$tr$dg n=$n" for T in (Float32, Float64, ComplexF32, ComplexF64),
            ul in ('U', 'L'), tr in ('N', 'T', 'C'), dg in ('N', 'U'), n in (1, 5, 16, 17, 40)

        A = randn(T, n, n) ./ T(2n)                       # well-conditioned: near-identity triangle
        for i in 1:n
            A[i, i] = one(T) + abs(real(A[i, i]))
        end
        b = randn(T, n)
        xr = copy(b); B.trsv!(ul, tr, dg, A, xr)
        xp = copy(b); PureBLAS.trsv!(A, xp; uplo = ul, trans = tr, diag = dg)
        @test l2err(xp, xr) < l2tol(T)
    end
end

@testitem "trmv/trsv blocked (large n) vs OpenBLAS" setup = [L2Oracle] begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B   # n > _TRI_NB(=64) exercises the blocked diagonal+gemv path
    @testset "$T $ul$tr$dg n=$n" for T in (Float32, Float64, ComplexF32, ComplexF64),
            ul in ('U', 'L'), tr in ('N', 'T', 'C'), dg in ('N', 'U'), n in (65, 100, 129, 257)

        A = randn(T, n, n); x0 = randn(T, n)
        xr = copy(x0); B.trmv!(ul, tr, dg, A, xr)
        xp = copy(x0); PureBLAS.trmv!(A, xp; uplo = ul, trans = tr, diag = dg)
        @test l2err(xp, xr) < l2tol(T)
        As = randn(T, n, n) ./ T(2n); for i in 1:n
            As[i, i] = one(T) + abs(real(As[i, i]))
        end
        b = randn(T, n)
        br = copy(b); B.trsv!(ul, tr, dg, As, br)
        bp = copy(b); PureBLAS.trsv!(As, bp; uplo = ul, trans = tr, diag = dg)
        @test l2err(bp, br) < l2tol(T)
    end
end

# `_trmv_fused_min` hoisted trmv's unblocked-vs-fused8 predicate out of `_trmv_blk!` so the crossover is
# forceable (PUREBLAS_FORCE_trmv_fused_min). The whole point is that the SHIPPED routing is byte-identical
# to the predicate it replaced — assert that directly rather than hope, over every n the routine can see.
@testitem "trmv fused-vs-unblocked threshold == the predicate it replaced" begin
    using PureBLAS
    NB = PureBLAS._TRI_NB
    L2 = PureBLAS._L2_BYTES
    # Skip under a deliberate override — a forced/pinned sweep must not read as a red suite.
    forced = !isnothing(PureBLAS._TRMV_FUSED_MIN_PREF) || haskey(ENV, "PUREBLAS_FORCE_trmv_fused_min")
    @testset "$T" for T in (Float32, Float64)
        fmin = PureBLAS._trmv_fused_min(T)
        @test forced || fmin == max(NB, isqrt(L2 ÷ (2 * sizeof(T)))) + 1
        @test forced || all(n -> (n <= NB || 2 * n * n * sizeof(T) <= L2) == (n < fmin), 1:16384)
    end
end

@testitem "trmv/trsv strided + round-trip + AD + dim" setup = [L2Oracle] begin
    using PureBLAS, LinearAlgebra, ForwardDiff
    import LinearAlgebra.BLAS as B
    A = randn(20, 20); xfull = randn(40); xs = @view xfull[1:2:end]   # strided → generic path
    xr = collect(xs); B.trmv!('U', 'N', 'N', A, xr)
    PureBLAS.trmv!(A, xs; uplo = 'U')
    @test l2err(collect(xs), xr) < l2tol(Float64)
    # round-trip: trsv ∘ trmv == identity
    L = randn(15, 15); for i in 1:15
        L[i, i] += 15
    end
    xt = randn(15); y = copy(xt)
    PureBLAS.trmv!(L, y; uplo = 'L'); PureBLAS.trsv!(L, y; uplo = 'L')
    @test l2err(y, xt) < 1.0e-9
    # AD through trsv (generic Dual path): d/dt Σ L⁻¹(xt+t·dx) = Σ L⁻¹ dx
    dx = randn(15)
    f(t) = (b = xt .+ t .* dx; PureBLAS.trsv!(L, b; uplo = 'L'); sum(b))
    @test ForwardDiff.derivative(f, 0.0) ≈ sum(LowerTriangular(L) \ dx)
    @test_throws DimensionMismatch PureBLAS.trmv!(zeros(3, 4), zeros(3))
    @test_throws DimensionMismatch PureBLAS.trsv!(zeros(3, 3), zeros(4))
end

# The DRAM ger panel driver only fires at A ≥ L3 (level2.jl:776), so the public ger test (small n) never
# reaches it. Call it directly at every NP with awkward m (masked W-tail) and n (< NP column remainder).
@testitem "ger panel driver (_ger_paneldrv_np) direct: all NP, m-tails + column remainders" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    tol(::Type{T}) where {T} = T <: Float32 ? 1.0f-4 : 1.0e-11
    @testset "T=$T np=$np m=$m n=$n" for T in (Float32, Float64), np in (1, 2, 4, 8),
            m in (1, 7, 16, 31), n in (1, 3, 8, 13)

        α = T(0.7); x = randn(T, m); y = randn(T, n); A0 = randn(T, m, n)
        Ap = copy(A0); PureBLAS._ger_paneldrv_np(m, n, α, x, y, Ap, np)   # A += α·x·yᵀ
        Ar = copy(A0); B.ger!(α, x, y, Ar)                               # OpenBLAS geru oracle
        @test norm(Ap .- Ar) / max(norm(Ar), eps(Float64)) < tol(T)
    end
end

@testitem "complex ger panel np resolves to a Val the kernel implements" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    # `_ger_pdc_cj!` strides its panel loop by the RUNTIME np but dispatches a COMPILE-TIME Val, and its
    # `else` arm is Val(8). Any np without an arm (3/5/6/7) therefore advances by np while WRITING 8
    # columns — a heap overwrite past A's last column, reproduced at 4096 stray elements before the
    # 2026-08-16 snap. Unlike the real `_ger_paneldrv_np` (one Val{NP} drives both bound and stride, so
    # every np is safe), the complex driver is only safe on the ladder. Assert the invariant at the
    # single resolution point rather than trusting the measure candidates: the danger case is a hand-set
    # `ger_panel_np` Preference, which no candidate-set reasoning covers.
    @test PureBLAS._cger_np() in (1, 2, 4, 8)

    tol(::Type{T}) where {T} = T <: Float32 ? 1.0f-4 : 1.0e-11
    @testset "T=$T cj=$cj np=$np m=$m n=$n" for T in (ComplexF32, ComplexF64),
            cj in (false, true), np in (2, 4, 8), m in (1, 7, 16), n in (1, 3, 8, 13)

        α = T(0.7, -0.3); x = randn(T, m); y = randn(T, n); A0 = randn(T, m, n)
        Ap = copy(A0); PureBLAS._ger_paneldrv_cmplx!(m, n, α, x, y, Ap, cj, np)
        Ar = A0 .+ α .* x .* transpose(cj ? conj.(y) : y)   # explicit outer product, as at :60
        @test norm(Ap .- Ar) / max(norm(Ar), eps(Float64)) < tol(real(T))
    end
end
