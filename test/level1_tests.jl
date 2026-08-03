@testsetup module Oracle
using LinearAlgebra
import LinearAlgebra.BLAS as B
export relerr, tol, B, TYPES, SIZES
relerr(a, b) = norm(a .- b) / max(norm(b), eps(Float64))
tol(::Type{T}) where {T} = T <: Union{Float32, ComplexF32} ? 1.0e-3 : 1.0e-10
const TYPES = (Float32, Float64, ComplexF32, ComplexF64)
const SIZES = (1, 2, 7, 8, 9, 16, 31, 1000)
end

@testitem "Level-1 contiguous vs OpenBLAS" setup = [Oracle] begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    @testset "$T n=$n" for T in TYPES, n in SIZES
        x = randn(T, n); y = randn(T, n); a = randn(T)
        # axpy! (native API)
        yb = copy(y); B.axpy!(a, x, yb)
        yp = copy(y); PureBLAS.axpy!(yp, a, x)
        @test relerr(yp, yb) < tol(T)
        # scal!
        xb = copy(x); B.scal!(a, xb)
        xp = copy(x); PureBLAS.scal!(a, xp)
        @test relerr(xp, xb) < tol(T)
        # blascopy!
        yp = similar(y); PureBLAS.blascopy!(yp, x)
        @test yp == x
        # swap!
        xp = copy(x); yp = copy(y); PureBLAS.swap!(xp, yp)
        @test xp == y && yp == x
        # dot (conjugated, matches LinearAlgebra.dot) and dotu (unconjugated)
        @test relerr(PureBLAS.dot(x, y), dot(x, y)) < tol(T)
        duref = T <: Complex ? B.dotu(x, y) : dot(x, y)
        @test relerr(PureBLAS.dotu(x, y), duref) < tol(T)
        # nrm2 / asum / iamax
        @test relerr(PureBLAS.nrm2(x), B.nrm2(x)) < tol(T)
        @test relerr(PureBLAS.asum(x), B.asum(x)) < tol(T)
        @test PureBLAS.iamax(x) == B.iamax(x)
    end
end

@testitem "Level-1 empty (n=0) edge cases" begin
    using PureBLAS
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        x = T[]; y = T[]
        @test PureBLAS.axpy!(copy(y), one(T), x) == y     # no-op
        @test PureBLAS.nrm2(x) == 0
        @test PureBLAS.asum(x) == 0
        @test PureBLAS.iamax(x) == 0
        @test PureBLAS.dot(x, y) == 0
    end
end

# Two-vector ops (axpy, dot, dotu) support negative/mismatched increments per the BLAS spec.
@testitem "nrm2 overflow/underflow safety vs OpenBLAS" setup = [Oracle] begin
    using PureBLAS
    import LinearAlgebra.BLAS as B
    @testset "$T" for T in (Float32, Float64)
        big = T <: Float32 ? T(1.0f30) : T(1.0e200)    # naive Σx² overflows to Inf
        tiny = T <: Float32 ? T(1.0f-30) : T(1.0e-200)  # naive Σx² underflows to 0
        for v in (big, tiny)
            x = fill(v, 100)
            @test PureBLAS.nrm2(x) ≈ B.nrm2(x) rtol = tol(T)
            @test isfinite(PureBLAS.nrm2(x)) && PureBLAS.nrm2(x) > 0
        end
    end
end

@testitem "Level-1 strided two-vector vs OpenBLAS" setup = [Oracle] begin
    using PureBLAS
    import LinearAlgebra.BLAS as B
    P = PureBLAS
    @testset "$T inc=($ix,$iy)" for T in TYPES,
            (ix, iy) in ((2, 1), (1, 3), (2, 3), (-1, 1), (-2, -3))

        n = 50
        lenx = 1 + (n - 1) * abs(ix); leny = 1 + (n - 1) * abs(iy)
        x = randn(T, lenx); y = randn(T, leny); a = randn(T)
        yb = copy(y); B.axpy!(n, a, x, ix, yb, iy)
        yp = copy(y); P._axpy!(n, a, x, ix, yp, iy)
        @test relerr(yp, yb) < tol(T)
        # OpenBLAS exposes ?dot for real, ?dotu/?dotc for complex.
        dref = T <: Complex ? B.dotu(n, x, ix, y, iy) : B.dot(n, x, ix, y, iy)
        @test relerr(P._dotu(n, x, ix, y, iy), dref) < tol(T)
        dcref = T <: Complex ? B.dotc(n, x, ix, y, iy) : B.dot(n, x, ix, y, iy)
        @test relerr(P._dotc(n, x, ix, y, iy), dcref) < tol(T)
    end
end

# Single-vector ops (scal, nrm2, asum, iamax) are spec'd for incx ≥ 1 (reference BLAS returns 0
# for incx < 1); test against OpenBLAS with positive strides only.
@testitem "Level-1 strided single-vector vs OpenBLAS" setup = [Oracle] begin
    using PureBLAS
    import LinearAlgebra.BLAS as B
    P = PureBLAS
    @testset "$T inc=$ix" for T in TYPES, ix in (1, 2, 3)
        n = 50; lenx = 1 + (n - 1) * ix
        x = randn(T, lenx); a = randn(T)
        @test relerr(P._nrm2(n, x, ix), B.nrm2(n, x, ix)) < tol(T)
        @test relerr(P._asum(n, x, ix), B.asum(n, x, ix)) < tol(T)
        @test P._iamax(n, x, ix) == B.iamax(n, x, ix)
        xb = copy(x); B.scal!(n, a, xb, ix)
        xp = copy(x); P._scal!(n, a, xp, ix)
        @test relerr(xp, xb) < tol(T)
    end
end

# iamax with NaN/Inf had NO coverage before 2026-07-31, which is why two proposed optimisations of
# `_iamax_thresh4!` had to be rejected on review rather than caught by a test: a value max-tree in the
# hot path lets a NaN lane HIDE a real maximum in the same lane of another block, and the cold path's
# `maximum(::Vec)` resolves to `fmax` or `fmaximum` depending on SIMD.jl's SUPPORTS_FMAXIMUM_FMINIMUM,
# so its NaN semantics are not even stable across dependency versions.
#
# The oracle is reference netlib `idamax` transcribed directly (init to |x[1]|, sequential strict `>`),
# NOT `BLAS.iamax`: OpenBLAS's vectorised kernel DISAGREES with netlib on NaN — for [1.0, NaN, 2.0] it
# returns 2 (picks the NaN) where netlib and PureBLAS return 3 (NaN fails `>`, so it is skipped).
# Pinning against OpenBLAS here would therefore encode a bug and forbid the correct behaviour.
# Note the asymmetry this exposes and deliberately preserves: a NaN at position 1 poisons the whole
# scan (result 1, because DMAX starts as NaN and nothing compares greater), while a NaN anywhere else
# is simply skipped. That is reference behaviour, not an accident.
@testitem "iamax NaN/Inf semantics (netlib reference oracle)" begin
    using PureBLAS

    function ref_iamax(x)                      # netlib idamax, verbatim
        n = length(x)
        n < 1 && return 0
        n == 1 && return 1
        dmax = abs(x[1]); ix = 1
        for k in 2:n
            if abs(x[k]) > dmax
                dmax = abs(x[k]); ix = k
            end
        end
        ix
    end

    for T in (Float32, Float64)
        nan, inf = T(NaN), T(Inf)
        cases = Vector{T}[
            T[nan, 1, 2], T[1, nan, 2], T[1, 2, nan], T[nan, nan, nan],
            T[nan, 1, 9, 3], T[1, inf, 2], T[1, -inf, 2], T[inf, nan, 1],
            T[nan, inf, 1], T[3, -3, 3],
            # NaN placed in each SIMD lane/block position, with the true max after it
            vcat(fill(T(1), 8), T[nan], fill(T(2), 8), T[9]),
            vcat(fill(T(1), 31), T[nan], fill(T(2), 31), T[9]),
            vcat(fill(T(1), 64), T[nan], fill(T(2), 64), T[9]),
            vcat(T[nan], fill(T(1), 128), T[9]),               # NaN first, long ⇒ SIMD path
            vcat(fill(T(1), 100), T[inf], fill(T(2), 100)),
            vcat(fill(T(1), 100), T[nan], fill(T(2), 100), T[inf]),
        ]
        for x in cases
            @test PureBLAS.iamax(x) == ref_iamax(x)
        end
        # every single-NaN position in a vector long enough to exercise all three kernel tiers
        for pos in (1, 2, 8, 9, 17, 33, 64, 65, 100, 127, 128)
            x = collect(T, 1:128); x[pos] = nan
            @test PureBLAS.iamax(x) == ref_iamax(x)
        end
        # NaN AND the true max in the same lane of different blocks — the case a max-tree would break
        for W in (4, 8), blk in (1, 2, 3)
            x = fill(T(1), 4W); x[W ÷ 2] = nan; x[blk * W + W ÷ 2] = T(99)
            @test PureBLAS.iamax(x) == ref_iamax(x)
        end
        # NaN immediately followed by the UNIQUE global max, at every alignment. This is the case that
        # discriminates a block-local rescan seed from netlib's running-max walk: if the rescan seeds its
        # block max from lane 1 and that lane is the NaN, the block is silently dropped and the true max
        # is never seen. The `collect(1:128)` sweep above cannot catch it — there a LATER block always
        # holds a bigger element, so the answer comes out right despite the dropped block. Sweeping every
        # p covers lane 1 of every block for every vector width. Regression for the 2026-08-01 fix.
        for p in 1:127
            x = fill(T(1), 128); x[p] = nan; x[p + 1] = T(99)
            @test PureBLAS.iamax(x) == ref_iamax(x)
        end
    end
end

# FIRST-OCCURRENCE (tie) semantics — the companion hazard to the NaN item above, and the one a
# BLIS-style restructure would break. `bli_damaxv_zen_int_avx512` finds the maximum VALUE with a
# vmaxpd merge tree, records only which BLOCK won, and recovers the index at the end by a scalar
# equality-rescan of that block. If the same magnitude also occurs in an EARLIER block, that shape
# returns the later index; netlib idamax returns the FIRST (its update is a strict `>`, so an equal
# element never displaces the incumbent). The existing cases above are almost all UNIQUE-max vectors,
# so they cannot see this: `T[3, -3, 3]` is the only tie, and it is 3 elements long — below every
# SIMD tier. These sweep ties across lanes, across blocks, and at every alignment.
#
# Also pinned here: |−0.0| == |0.0| is a tie (must give index 1, not 2), and ±x are a tie under abs.
@testitem "iamax first-occurrence on ties (netlib reference oracle)" begin
    using PureBLAS

    function ref_iamax(x)                      # netlib idamax, verbatim — strict `>`, so ties keep first
        n = length(x)
        n < 1 && return 0
        n == 1 && return 1
        dmax = abs(x[1]); ix = 1
        for k in 2:n
            if abs(x[k]) > dmax
                dmax = abs(x[k]); ix = k
            end
        end
        ix
    end

    for T in (Float32, Float64)
        nan = T(NaN)
        cases = Vector{T}[
            T[0, -0.0], T[-0.0, 0], T[5, -5], T[-5, 5],          # sign/zero ties
            fill(T(7), 3), fill(T(7), 8), fill(T(7), 64), fill(T(7), 300),   # all-equal, every tier
        ]
        for x in cases
            @test PureBLAS.iamax(x) == ref_iamax(x)
        end
        # the SAME maximum magnitude in two different blocks: the first must win. Sweep both positions
        # across lane and block boundaries for both vector widths.
        for n in (64, 128, 300), p in (1, 2, 4, 8, 9, 16, 17, 32, 33, 63, 64)
            p < n || continue
            for q in (p + 1, p + 8, p + 16, p + 63)
                q <= n || continue
                x = fill(T(1), n); x[p] = T(99); x[q] = T(99)     # exact tie, p < q ⇒ answer is p
                @test PureBLAS.iamax(x) == ref_iamax(x) == p
                x2 = fill(T(1), n); x2[p] = T(99); x2[q] = T(-99) # tie under abs
                @test PureBLAS.iamax(x2) == ref_iamax(x2) == p
            end
        end
        # a tie whose duplicate sits in a LATER block than a NaN — combines both hazards
        for p in (2, 9, 17, 65)
            x = fill(T(1), 128); x[p] = nan; x[p + 1] = T(50); x[p + 40] = T(50)
            @test PureBLAS.iamax(x) == ref_iamax(x)
        end
    end
end

# SAME-LANE, CROSS-BLOCK NaN/max pairs — the input class that breaks a max-TREE detect, which is what
# `_iamax_thresh4!` uses below L2. The tree computes m = vmax(vmax(v0,v1), vmax(v2,v3)) over the four
# blocks of one iteration and then asks whether any lane of m exceeds the running threshold. vmaxpd
# PROPAGATES a NaN over a real maximum in the same lane of another block, so an ORDERED compare
# (`any(m > thr)`) would swallow the real max and skip the whole iteration. The shipped detect uses the
# UNORDERED form (`any(!(m <= thr))`), under which a NaN lane conservatively fires the cold path and the
# byte-for-byte-unchanged gmax-seeded walk then resolves the true answer.
#
# These pin that. Every pair is placed at the SAME lane of two different blocks within ONE 4W iteration,
# in both orders and at all three tree levels (blocks 0-1, 0-2, 0-3), for both vector widths. Also the
# leftover-W and scalar-tail regions, the n = 4W routing threshold, and ties across blocks.
@testitem "iamax max-tree detect: same-lane cross-block NaN/max (netlib reference oracle)" begin
    using PureBLAS

    function ref_iamax(x)
        n = length(x)
        n < 1 && return 0
        n == 1 && return 1
        dmax = abs(x[1]); ix = 1
        for k in 2:n
            if abs(x[k]) > dmax
                dmax = abs(x[k]); ix = k
            end
        end
        ix
    end

    for T in (Float32, Float64)
        nan, inf = T(NaN), T(Inf)
        for W in (4, 8, 16)                       # cover both real widths and F32's W=16
            n = 8W
            for l in 1:W, d in (W, 2W, 3W)        # same lane l, blocks 1 apart / 2 apart / 3 apart
                l + d <= n || continue
                x = ones(T, n); x[l] = nan; x[l + d] = T(99)       # NaN before the max
                @test PureBLAS.iamax(x) == ref_iamax(x)
                y = ones(T, n); y[l] = T(99); y[l + d] = nan       # max before the NaN
                @test PureBLAS.iamax(y) == ref_iamax(y)
            end
            # the true max in a DIFFERENT block of a NaN-poisoned iteration
            for l in (1, W ÷ 2, W)
                x = ones(T, n); x[l] = nan; x[l + W + 1] = T(55)
                @test PureBLAS.iamax(x) == ref_iamax(x)
            end
        end
        # NaN at position 1 on both sides of the n = 4W routing threshold, all-NaN, and the tail regions
        for n in (15, 16, 17, 31, 32, 33, 63, 64, 65, 100)
            x = collect(T, 1:n); x[1] = nan
            @test PureBLAS.iamax(x) == ref_iamax(x)
            y = fill(nan, n)
            @test PureBLAS.iamax(y) == ref_iamax(y)
            z = collect(T, 1:n); z[n] = nan                        # NaN in the scalar tail
            @test PureBLAS.iamax(z) == ref_iamax(z)
        end
        # NaN in the leftover-W region specifically (n just past a 4W multiple), and +-Inf with NaN
        for n in (36, 68, 132)
            x = ones(T, n); x[n - 2] = nan; x[n - 1] = T(9)
            @test PureBLAS.iamax(x) == ref_iamax(x)
            y = ones(T, n); y[3] = inf; y[n - 3] = nan; y[n - 1] = -inf
            @test PureBLAS.iamax(y) == ref_iamax(y)
        end
        # ties across blocks WITHIN one iteration, and across iterations — first must win either way
        let x = ones(T, 128)
            x[33] = T(7); x[41] = T(7)
            @test PureBLAS.iamax(x) == ref_iamax(x) == 33
        end
        let x = ones(T, 512)
            x[5] = T(7); x[300] = T(7)
            @test PureBLAS.iamax(x) == ref_iamax(x) == 5
        end
    end
end
