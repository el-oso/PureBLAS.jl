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

    # The snap itself, over every value any budget or Preference could produce. This is the axis the
    # suite otherwise does not cover: BOTH memory-safety bugs found on 2026-08-16 were reachable only
    # through a NON-DEFAULT knob, and every test here runs at defaults. Testing `_snap_np` directly
    # covers all of them at once without spawning a process per pin.
    # Verified out-of-band with the real env override (PUREBLAS_FORCE_ger_np): 3 -> 2, 5 -> 4, 6 -> 4,
    # 7 -> 4. np=3 is the value that wrote 4096 elements past A before 76ffd06.
    @test all(PureBLAS._snap_np(r) in (1, 2, 4, 8) for r in 1:64)
    @test all(PureBLAS._snap_np(r) <= r for r in 1:64)          # snaps DOWN — stays inside the budget
    @test all(PureBLAS._snap_np(r) == r for r in (1, 2, 4, 8))  # exact on the ladder itself

    tol(::Type{T}) where {T} = T <: Float32 ? 1.0f-4 : 1.0e-11
    @testset "T=$T cj=$cj np=$np m=$m n=$n" for T in (ComplexF32, ComplexF64),
            cj in (false, true), np in (2, 4, 8), m in (1, 7, 16), n in (1, 3, 8, 13)

        α = T(0.7, -0.3); x = randn(T, m); y = randn(T, n); A0 = randn(T, m, n)
        Ap = copy(A0); PureBLAS._ger_paneldrv_cmplx!(m, n, α, x, y, Ap, cj, np)
        Ar = A0 .+ α .* x .* transpose(cj ? conj.(y) : y)   # explicit outer product, as at :60
        @test norm(Ap .- Ar) / max(norm(Ar), eps(Float64)) < tol(real(T))
    end
end

# ── req#11 FOR ger: bitwise reproducibility across thread counts ───────────────────────────────────
#
# ger is the one threaded op whose invariance needs no argument about accumulation order: there is no
# reduction anywhere, every `A[i,j]` is written exactly once, and both route arms fold α into `y[j]`
# before touching `x`, so a column partition cannot change a single bit. The test still earns its
# place, because the things that CAN go wrong are not about arithmetic:
#
#   * a column split that overlaps (a write race) or leaves a gap (a stale element), which is what
#     `_gemm_chunk`'s `_NR` rounding has to get right at the ragged end;
#   * the ROUTE, which is keyed on the A byte count and so moves under a split — bits are safe either
#     way, but the sizes below straddle `_L3_BYTES` deliberately so both arms are exercised;
#   * the lost-claim fallback, which must reach the same arm as the winner's chunks.
#
# Sizes: `n` large enough that `_ger_workers` admits every worker, and an `n` that is NOT a multiple
# of `_NR` so the last chunk is ragged.
@testitem "ger: bit-identical at every thread count" tags = [:checks] begin
    using PureBLAS, LinearAlgebra
    using Base.Threads: nthreads
    const P = PureBLAS
    nt = min(6, nthreads())
    if nt < 2
        @test_skip "needs >=2 julia threads"
    else
        bitsame(X, Y) = length(X) == length(Y) && all(i -> bitstring(X[i]) == bitstring(Y[i]), eachindex(X))
        @testset "$T" for T in (Float64, Float32)
            L3 = P._L3_BYTES
            # One shape comfortably inside L3 (the per-column arm) and one past it (the panel arm),
            # each in a square-ish and a ragged-n variant.
            # `s` is the square side whose A is exactly L3, so `s ÷ 2` is a quarter of L3 — safely on
            # the per-column arm — and `s` itself trips the `>= _L3_BYTES` panel predicate. Kept AT
            # the boundary rather than a multiple past it: four copies of a 4x-L3 matrix is half a
            # gigabyte and buys no coverage the boundary shape does not already give.
            mns = let s = isqrt(L3 ÷ sizeof(T))
                ((s ÷ 2, s ÷ 2), (s ÷ 2, s ÷ 2 + 7), (s, s), (s, s + 7))
            end
            @testset "m=$m n=$n" for (m, n) in mns
                A0 = randn(T, m, n)
                x = randn(T, m)
                y = randn(T, n)
                α = T(0.75)                      # NOT a power of two
                P.set_num_threads(1)
                want = (t = copy(A0); P.ger!(α, x, y, t); t)
                for nw in (2, nt)
                    P.set_num_threads(nw)
                    @test bitsame((t = copy(A0); P.ger!(α, x, y, t); t), want)
                end
                P.set_num_threads(1)
            end
        end
    end
end

# The invariant item above passes for a library that ignores every thread it is given, so it needs a
# liveness gate — the same `p.gen` witness the L1 items use — and a concurrent exercise of the
# lost-claim path, which is the branch a worker count can reach that is NOT the happy one.
@testitem "ger: the pool runs, and losing the claim gives the same answer" tags = [:checks] begin
    using PureBLAS, LinearAlgebra
    using Base.Threads: nthreads, @spawn
    const P = PureBLAS
    nt = min(6, nthreads())
    if nt < 2
        @test_skip "needs >=2 julia threads"
    else
        P.set_num_threads(nt)
        p = P._gemm_pool(Float64)
        # AT the L3 boundary, with a ragged `n`: enough to admit every worker and to take the panel
        # arm, while keeping one copy near L3 rather than a multiple of it. The concurrent block
        # below holds `nt + 1` copies live at once, so the per-copy footprint is the binding cost.
        s = isqrt(P._L3_BYTES ÷ sizeof(Float64))
        m, n = s, s + 7
        A0 = randn(m, n); x = randn(m); y = randn(n)
        g0 = @atomic p.gen
        A1 = copy(A0); P.ger!(0.75, x, y, A1)
        @test (@atomic p.gen) != g0                    # the pool was actually dispatched
        P.set_num_threads(1)
        A2 = copy(A0); P.ger!(0.75, x, y, A2)          # serial reference
        @test A1 == A2
        # CONCURRENT: several callers at once, so all but one LOSE the claim and run the fallback.
        # Every result must equal the serial one, which is what pins the fallback to the same arm.
        P.set_num_threads(nt)
        tasks = [@spawn (t = copy(A0); P.ger!(0.75, x, y, t); t) for _ in 1:(nt + 1)]
        for t in tasks
            @test fetch(t) == A2
        end
        P.set_num_threads(1)
    end
end

# ── req#11 FOR gemv-N: bitwise reproducibility across thread counts ────────────────────────────────
#
# UNLIKE ger, THIS ONE IS A REAL CORRECTNESS GATE, and the sizes are chosen to make it bite. gemv-N
# splits ROWS, and two of its route decisions are keyed on the A byte count `m * n * sizeof(T)`:
#
#   * `_gemvn_minner() && m*n*sizeof(T) <= _GEMVN_MINNER_MAXA` picks between the minner driver (panel
#     width from `_gemvn_minner_np`) and the old one (a flat `_GEMV_NP`). Different column chunking,
#     so each `y[i]` accumulates its n-chain in a different ORDER.
#   * inside minner, `_gemvn_minner_np` itself flips NARROW/WIDE at `2 * _L2_BYTES` — again a panel
#     width, again the n-chain's grouping.
#
# A row band holds 1/nw of the bytes, so without the `mroute` token a worker flips either predicate
# and returns different bits from the undivided problem. The two "flip" sizes below sit JUST ABOVE
# each threshold, so the whole problem is on one side and every band is on the other — which is
# exactly the case that fails if the token is dropped. The row-block size is there because
# `n <= _gemvn_rb()` is keyed on `n`, which a row split does NOT move: it must keep working, and
# `_gemvn_rowblock_mr` picking a different block height per band must stay bit-neutral.
#
# β is swept because each band pre-scales its OWN rows. Two workers double-scaling, or a band
# skipping the scale, shows up here and nowhere else.
@testitem "gemv-N: bit-identical at every thread count" tags = [:checks] begin
    using PureBLAS, LinearAlgebra
    using Base.Threads: nthreads
    const P = PureBLAS
    nt = min(6, nthreads())
    if nt < 2
        @test_skip "needs >=2 julia threads"
    else
        bitsame(X, Y) = length(X) == length(Y) && all(i -> bitstring(X[i]) == bitstring(Y[i]), eachindex(X))
        @testset "$T" for T in (Float64, Float32)
            sz = sizeof(T)
            # Just past `2 * _L2_BYTES` with a modest `n`, so the undivided problem is WIDE and every
            # band is NARROW.
            n_np = 160
            m_np = (5 * P._L2_BYTES) ÷ (2 * n_np * sz)
            # Just past `_GEMVN_MINNER_MAXA` (= 4·L3), so the undivided problem leaves minner and
            # every band would re-enter it. One allocation of a little over 4·L3 is the price of
            # covering this predicate at all; there is no smaller shape that straddles it.
            n_mx = 1024
            m_mx = (P._GEMVN_MINNER_MAXA ÷ (n_mx * sz)) + 64
            cases = (
                ("rowblock n<=rb", 8192, min(64, P._gemvn_rb())),
                ("np narrow/wide flip", m_np, n_np),
                ("minner/paneldrv flip", m_mx, n_mx),
            )
            @testset "$nm m=$m n=$n" for (nm, m, n) in cases
                A = randn(T, m, n)
                x = randn(T, n)
                y0 = randn(T, m)
                α = T(0.75)                                  # NOT a power of two
                @testset "beta=$β" for β in (zero(T), one(T), T(2.5))
                    P.set_num_threads(1)
                    want = (t = copy(y0); P.gemv!(t, A, x; alpha = α, beta = β); t)
                    for nw in (2, nt)
                        P.set_num_threads(nw)
                        @test bitsame((t = copy(y0); P.gemv!(t, A, x; alpha = α, beta = β); t), want)
                    end
                    P.set_num_threads(1)
                end
            end
        end
    end
end

# Liveness plus the lost-claim path, as for ger. The extra assertion here is that `trmv` must NOT
# reach the pool: `_tri_scat!` calls `_gemv_n_paneldrv!` directly precisely so the `_TRMV_ACC*` and
# `_TRSV_*` per-thread owners in `test/perthread_lint_baseline.txt` stay un-forked, and that
# justification is now load-bearing rather than incidental.
@testitem "gemv-N: the pool runs, trmv stays out of it, and losing the claim agrees" tags = [:checks] begin
    using PureBLAS, LinearAlgebra
    using Base.Threads: nthreads, @spawn
    const P = PureBLAS
    nt = min(6, nthreads())
    if nt < 2
        @test_skip "needs >=2 julia threads"
    else
        P.set_num_threads(nt)
        p = P._gemm_pool(Float64)
        m, n = 8192, 512
        A = randn(m, n); x = randn(n); y0 = randn(m)
        ran(f) = (g0 = @atomic p.gen; f(); (@atomic p.gen) != g0)
        @test ran(() -> P.gemv!(copy(y0), A, x; alpha = 0.75, beta = 2.5))
        # trmv over a matrix big enough that a threaded gemv-N WOULD have been admitted. If this ever
        # reports true, the four per-thread baseline entries lost their justification.
        Tri = randn(2048, 2048) + 2048I
        v = randn(2048)
        @test !ran(() -> P.trmv!(Tri, copy(v)))
        y1 = copy(y0); P.gemv!(y1, A, x; alpha = 0.75, beta = 2.5)
        P.set_num_threads(1)
        y2 = copy(y0); P.gemv!(y2, A, x; alpha = 0.75, beta = 2.5)
        @test y1 == y2
        # CONCURRENT: all but one caller loses the claim and runs the fallback, which must reach the
        # same route as the winner's chunks.
        P.set_num_threads(nt)
        tasks = [@spawn (t = copy(y0); P.gemv!(t, A, x; alpha = 0.75, beta = 2.5); t) for _ in 1:(2 * nt)]
        for t in tasks
            @test fetch(t) == y2
        end
        P.set_num_threads(1)
    end
end

# ── req#11 FOR gemv-T: bitwise reproducibility across thread counts ────────────────────────────────
#
# The mirror of the gemv-N item. gemv-T splits COLUMNS, and two decisions read the A byte count that a
# column split shrinks:
#
#   * `_gemvt_perscan` — `m*n*sizeof(T) > _GEMVT_PERCOL_AMIN` chooses per-column over NC-blocked, and
#     the two reduce each `y[j]` by different arithmetic, not merely at a different speed;
#   * `_gemvt_deep` — `m*n*sizeof(T) <= _L2_BYTES` switches the ACCUMULATOR COUNT
#     (`_GEMVT_NC_DEEP`/`_GEMVT_U_DEEP`), which is a different fold tree.
#
# The sizes sit JUST ABOVE each threshold so the undivided problem is on one side and every band on
# the other. Both predicates also carry an m-only term (`m*sizeof(T) <= _GEMVT_PERCOL_XMAX`,
# `_gemvt_x_l1_resident(T, m, lda)`) which a column split does not move — so `m` is kept inside those
# bounds, otherwise the case is decided by the m term and the n flip is never reached.
#
# A µarch where `_gemvt_perscan_mode() != 1` makes the first predicate constant, and a pinned
# `gemvt_deep = false` does the same for the second. The item still asserts invariance there; it just
# stops being the test that would catch a dropped token. That is why BOTH are present.
@testitem "gemv-T: bit-identical at every thread count" tags = [:checks] begin
    using PureBLAS, LinearAlgebra
    using Base.Threads: nthreads
    const P = PureBLAS
    nt = min(6, nthreads())
    if nt < 2
        @test_skip "needs >=2 julia threads"
    else
        bitsame(X, Y) = length(X) == length(Y) && all(i -> bitstring(X[i]) == bitstring(Y[i]), eachindex(X))
        @testset "$T" for T in (Float64, Float32)
            sz = sizeof(T)
            # Keep m under _GEMVT_PERCOL_XMAX so the perscan decision turns on the n term.
            m_ps = min(4096, P._GEMVT_PERCOL_XMAX ÷ sz)
            n_ps = max(2 * P._NR, (5 * P._GEMVT_PERCOL_AMIN) ÷ (4 * m_ps * sz))
            # Just past L2, so the undivided problem leaves `_gemvt_deep`'s size branch and every band
            # re-enters it.
            m_dp = 2048
            n_dp = max(2 * P._NR, (5 * P._L2_BYTES) ÷ (4 * m_dp * sz))
            cases = (("perscan flip", m_ps, n_ps), ("deep flip", m_dp, n_dp))
            @testset "$nm m=$m n=$n" for (nm, m, n) in cases
                A = randn(T, m, n)
                x = randn(T, m)            # gemv-T consumes m and produces n
                y0 = randn(T, n)
                α = T(0.75)
                @testset "beta=$β" for β in (zero(T), one(T), T(2.5))
                    P.set_num_threads(1)
                    want = (t = copy(y0); P.gemv!(t, A, x; alpha = α, beta = β, trans = 'T'); t)
                    for nw in (2, nt)
                        P.set_num_threads(nw)
                        @test bitsame((t = copy(y0); P.gemv!(t, A, x; alpha = α, beta = β, trans = 'T'); t), want)
                    end
                    P.set_num_threads(1)
                end
            end
        end
    end
end

# Liveness, the trmv exclusion, and the lost-claim path — as for gemv-N. The trmv assertion matters
# MORE here than there: `_tri_scatT!` calls `_gemv_t_simd!` itself, so the thread seam had to go one
# level up into `_gemv!`. If that ever slips back down, this is what notices.
@testitem "gemv-T: the pool runs, trmv-T stays out, and losing the claim agrees" tags = [:checks] begin
    using PureBLAS, LinearAlgebra
    using Base.Threads: nthreads, @spawn
    const P = PureBLAS
    nt = min(6, nthreads())
    if nt < 2
        @test_skip "needs >=2 julia threads"
    else
        P.set_num_threads(nt)
        p = P._gemm_pool(Float64)
        m, n = 4096, 1024
        A = randn(m, n); x = randn(m); y0 = randn(n)
        ran(f) = (g0 = @atomic p.gen; f(); (@atomic p.gen) != g0)
        @test ran(() -> P.gemv!(copy(y0), A, x; alpha = 0.75, beta = 2.5, trans = 'T'))
        # trmv with trans='T' routes through `_tri_scatT!`, which calls `_gemv_t_simd!` directly. It
        # must not reach the pool, or the `_TRMV_ACC*` / `_TRSV_*` per-thread owners in
        # test/perthread_lint_baseline.txt lose the justification they are listed under.
        Tri = randn(2048, 2048) + 2048I
        v = randn(2048)
        @test !ran(() -> P.trmv!(Tri, copy(v); trans = 'T'))
        @test !ran(() -> P.trsv!(Tri, copy(v); trans = 'T'))
        y1 = copy(y0); P.gemv!(y1, A, x; alpha = 0.75, beta = 2.5, trans = 'T')
        P.set_num_threads(1)
        y2 = copy(y0); P.gemv!(y2, A, x; alpha = 0.75, beta = 2.5, trans = 'T')
        @test y1 == y2
        P.set_num_threads(nt)
        tasks = [@spawn (t = copy(y0); P.gemv!(t, A, x; alpha = 0.75, beta = 2.5, trans = 'T'); t) for _ in 1:(2 * nt)]
        for t in tasks
            @test fetch(t) == y2
        end
        P.set_num_threads(1)
    end
end
