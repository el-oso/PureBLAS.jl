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
    end
end
