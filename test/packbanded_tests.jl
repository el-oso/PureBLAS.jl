@testsetup module PackBand
    using LinearAlgebra
    export l2err, l2tol, packtri_U, packtri_L, packband
    l2err(a, b) = norm(a .- b) / max(norm(b), eps(Float64))
    l2tol(::Type{T}) where {T} = T <: Union{Float32, ComplexF32} ? 1.0e-3 : 1.0e-10
    # Pack a dense triangle into linear packed storage AP (1-based BLAS layout).
    function packtri_U(A, n)
        AP = zeros(eltype(A), (n * (n + 1)) ÷ 2)
        for j in 1:n, i in 1:j; AP[i + (j * (j - 1)) ÷ 2] = A[i, j]; end
        AP
    end
    function packtri_L(A, n)
        AP = zeros(eltype(A), (n * (n + 1)) ÷ 2)
        for j in 1:n, i in j:n; AP[i + ((j - 1) * (2n - j)) ÷ 2] = A[i, j]; end
        AP
    end
    # Pack a dense triangular band into AB ((k+1)×n).
    function packband(up, A, n, k)
        AB = zeros(eltype(A), k + 1, n)
        if up
            for j in 1:n, i in max(1, j - k):j; AB[k + 1 + i - j, j] = A[i, j]; end
        else
            for j in 1:n, i in j:min(n, j + k); AB[1 + i - j, j] = A[i, j]; end
        end
        AB
    end
end

@testitem "spmv/hpmv vs OpenBLAS packed" setup = [PackBand] begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    @testset "spmv $T $ul n=$n" for T in (Float32, Float64), ul in ('U', 'L'), n in (1, 5, 16, 33, 64)
        AP = randn(T, (n * (n + 1)) ÷ 2); x = randn(T, n); y0 = randn(T, n)
        for (al, be) in ((one(T), zero(T)), (T(0.7), T(1.3)))
            yr = copy(y0); B.spmv!(ul, al, AP, x, be, yr)
            yp = copy(y0); PureBLAS.spmv!(yp, AP, x; uplo = ul, alpha = al, beta = be)
            @test l2err(yp, yr) < l2tol(T)
        end
    end
    @testset "hpmv $T $ul n=$n" for T in (ComplexF32, ComplexF64), ul in ('U', 'L'), n in (1, 5, 16, 33, 64)
        AP = randn(T, (n * (n + 1)) ÷ 2); x = randn(T, n); y0 = randn(T, n)
        for (al, be) in ((one(T), zero(T)), (T(0.7) + im, T(1.3)))
            yr = copy(y0); B.hpmv!(ul, al, AP, x, be, yr)
            yp = copy(y0); PureBLAS.hpmv!(yp, AP, x; uplo = ul, alpha = al, beta = be)
            @test l2err(yp, yr) < l2tol(T)
        end
    end
end

@testitem "tpmv/tpsv packed vs dense trmv/trsv" setup = [PackBand] begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B   # no BLAS packed-triangular wrapper → oracle on the dense matrix
    @testset "$T $ul$tr$dg n=$n" for T in (Float32, Float64, ComplexF32, ComplexF64),
        ul in ('U', 'L'), tr in ('N', 'T', 'C'), dg in ('N', 'U'), n in (1, 5, 16, 33, 65, 100)

        A = randn(T, n, n) ./ T(2n); for i in 1:n; A[i, i] = one(T) + abs(real(A[i, i])); end
        AP = ul == 'U' ? packtri_U(A, n) : packtri_L(A, n)
        x0 = randn(T, n)
        xr = copy(x0); B.trmv!(ul, tr, dg, A, xr)
        xp = copy(x0); PureBLAS.tpmv!(AP, xp; uplo = ul, trans = tr, diag = dg)
        @test l2err(xp, xr) < l2tol(T)
        br = copy(x0); B.trsv!(ul, tr, dg, A, br)
        bp = copy(x0); PureBLAS.tpsv!(AP, bp; uplo = ul, trans = tr, diag = dg)
        @test l2err(bp, br) < l2tol(T)
    end
end

@testitem "gbmv vs OpenBLAS banded" setup = [PackBand] begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    @testset "$T $tr m=$m n=$n kl=$kl ku=$ku" for T in (Float32, Float64, ComplexF32, ComplexF64),
        tr in ('N', 'T', 'C'), (m, n) in ((16, 16), (20, 12), (12, 20)), (kl, ku) in ((0, 0), (2, 1), (3, 4))

        AB = randn(T, kl + ku + 1, n)
        xlen = tr == 'N' ? n : m; ylen = tr == 'N' ? m : n
        x = randn(T, xlen); y0 = randn(T, ylen)
        for (al, be) in ((one(T), zero(T)), (T(0.7), T(1.3)))
            yr = copy(y0); B.gbmv!(tr, m, kl, ku, al, AB, x, be, yr)
            yp = copy(y0); PureBLAS.gbmv!(yp, AB, x, m, kl, ku; trans = tr, alpha = al, beta = be)
            @test l2err(yp, yr) < l2tol(T)
        end
    end
end

@testitem "gbmv wide-band vs OpenBLAS (conv path)" setup = [PackBand] begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B   # wide band (≥ SIMD width) exercises gbmv-T conv + per-column boundary
    @testset "$T $tr kl=$kl ku=$ku n=$n" for T in (Float32, Float64),
        tr in ('N', 'T'), (kl, ku) in ((7, 8), (15, 16), (31, 32), (63, 64), (95, 96)),
        n in (40, 100, 200, 500)

        AB = randn(T, kl + ku + 1, n); x = randn(T, n); y0 = randn(T, n)
        for (al, be) in ((one(T), zero(T)), (T(0.7), T(1.3)))
            yr = copy(y0); B.gbmv!(tr, n, kl, ku, al, AB, x, be, yr)
            yp = copy(y0); PureBLAS.gbmv!(yp, AB, x, n, kl, ku; trans = tr, alpha = al, beta = be)
            @test l2err(yp, yr) < l2tol(T)
        end
    end
end

@testitem "sbmv/hbmv vs OpenBLAS banded" setup = [PackBand] begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    @testset "sbmv $T $ul n=$n k=$k" for T in (Float32, Float64), ul in ('U', 'L'), n in (1, 8, 16, 33), k in (0, 2, 5)
        k >= n && continue
        AB = randn(T, k + 1, n); x = randn(T, n); y0 = randn(T, n)
        for (al, be) in ((one(T), zero(T)), (T(0.7), T(1.3)))
            yr = copy(y0); B.sbmv!(ul, k, al, AB, x, be, yr)
            yp = copy(y0); PureBLAS.sbmv!(yp, AB, x; uplo = ul, alpha = al, beta = be)
            @test l2err(yp, yr) < l2tol(T)
        end
    end
    @testset "hbmv $T $ul n=$n k=$k" for T in (ComplexF32, ComplexF64), ul in ('U', 'L'), n in (1, 8, 16, 33), k in (0, 2, 5)
        k >= n && continue
        AB = randn(T, k + 1, n); x = randn(T, n); y0 = randn(T, n)
        for (al, be) in ((one(T), zero(T)), (T(0.7) + im, T(1.3)))
            yr = copy(y0); B.hbmv!(ul, k, al, AB, x, be, yr)
            yp = copy(y0); PureBLAS.hbmv!(yp, AB, x; uplo = ul, alpha = al, beta = be)
            @test l2err(yp, yr) < l2tol(T)
        end
    end
end

@testitem "tbmv/tbsv banded vs dense trmv/trsv" setup = [PackBand] begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B   # no BLAS banded-triangular wrapper → oracle on the dense matrix
    @testset "$T $ul$tr$dg n=$n k=$k" for T in (Float32, Float64, ComplexF32, ComplexF64),
        ul in ('U', 'L'), tr in ('N', 'T', 'C'), dg in ('N', 'U'), n in (5, 16, 33, 65), k in (0, 2, 5)

        k >= n && continue
        A = zeros(T, n, n)                      # dense triangular band, diagonally dominant for trsv
        if ul == 'U'
            for j in 1:n, i in max(1, j - k):j; A[i, j] = randn(T) / T(2n); end
        else
            for j in 1:n, i in j:min(n, j + k); A[i, j] = randn(T) / T(2n); end
        end
        for i in 1:n; A[i, i] = one(T) + abs(real(A[i, i])); end
        AB = packband(ul == 'U', A, n, k); x0 = randn(T, n)
        xr = copy(x0); B.trmv!(ul, tr, dg, A, xr)
        xp = copy(x0); PureBLAS.tbmv!(AB, xp; uplo = ul, trans = tr, diag = dg)
        @test l2err(xp, xr) < l2tol(T)
        br = copy(x0); B.trsv!(ul, tr, dg, A, br)
        bp = copy(x0); PureBLAS.tbsv!(AB, bp; uplo = ul, trans = tr, diag = dg)
        @test l2err(bp, br) < l2tol(T)
    end
end

@testitem "packed/banded dim mismatch + AD" setup = [PackBand] begin
    using PureBLAS, ForwardDiff, LinearAlgebra
    @test_throws DimensionMismatch PureBLAS.spmv!(zeros(3), zeros(5), zeros(3))   # AP too short (need 6)
    @test_throws DimensionMismatch PureBLAS.tbmv!(zeros(2, 4), zeros(5))
    # AD through tpsv (generic Dual path)
    n = 12; A = randn(n, n) ./ (2n); for i in 1:n; A[i, i] = 1 + abs(A[i, i]); end
    AP = packtri_L(A, n); dx = randn(n); b0 = randn(n)
    f(t) = (b = b0 .+ t .* dx; PureBLAS.tpsv!(AP, b; uplo = 'L'); sum(b))
    @test ForwardDiff.derivative(f, 0.0) ≈ sum(LowerTriangular(A) \ dx)
end

@testitem "spr/spr2/hpr/hpr2 packed rank updates" setup = [PackBand] begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    pkU(j) = (j * (j - 1)) ÷ 2; pkL(j, n) = ((j - 1) * (2n - j)) ÷ 2
    function unpk(AP, n, up, herm)
        A = zeros(eltype(AP), n, n)
        for j in 1:n, i in (up ? (1:j) : (j:n))
            v = AP[up ? i + pkU(j) : i + pkL(j, n)]; A[i, j] = v; A[j, i] = herm ? conj(v) : v
        end
        A
    end
    @testset "spr/spr2 $T $ul n=$n" for T in (Float32, Float64), ul in ('U', 'L'), n in (1, 5, 16, 33)
        up = ul == 'U'; pack = up ? packtri_U : packtri_L
        As = Matrix(Symmetric(randn(T, n, n))); x = randn(T, n); y = randn(T, n); al = T(0.7)
        AP = pack(As, n); PureBLAS.spr!(al, x, AP; uplo = ul)
        @test l2err(unpk(AP, n, up, false), As .+ al .* (x * x')) < l2tol(T)
        APb = pack(As, n); B.spr!(ul, al, x, APb)                 # vs OpenBLAS spr
        @test l2err(AP, APb) < l2tol(T)
        AP2 = pack(As, n); PureBLAS.spr2!(al, x, y, AP2; uplo = ul)
        @test l2err(unpk(AP2, n, up, false), As .+ al .* (x * y' .+ y * x')) < l2tol(T)
    end
    @testset "hpr/hpr2 $T $ul n=$n" for T in (ComplexF32, ComplexF64), ul in ('U', 'L'), n in (1, 5, 16, 33)
        up = ul == 'U'; pack = up ? packtri_U : packtri_L
        Hd = Matrix(Hermitian(randn(T, n, n))); x = randn(T, n); y = randn(T, n); ar = real(T)(0.6)
        APh = pack(Hd, n); PureBLAS.hpr!(ar, x, APh; uplo = ul)
        @test l2err(unpk(APh, n, up, true), Hd .+ ar .* (x * x')) < l2tol(T)
        ac = T(0.5 + 0.3im); APh2 = pack(Hd, n); PureBLAS.hpr2!(ac, x, y, APh2; uplo = ul)
        @test l2err(unpk(APh2, n, up, true), Hd .+ ac .* (x * y') .+ conj(ac) .* (y * x')) < l2tol(T)
    end
    # dim checks + ForwardDiff (generic path) through spr
    @test_throws DimensionMismatch PureBLAS.spr!(1.0, randn(5), zeros(9))          # AP too short (need 15)
    @test_throws DimensionMismatch PureBLAS.spr2!(1.0, randn(5), randn(4), zeros(15))
    let
        using ForwardDiff
        f(t) = (AP = zeros(eltype(t), 3); PureBLAS.spr!(t, [1.0, 2.0], AP; uplo = 'U'); AP[3])  # = t·x2² = 4t
        @test ForwardDiff.derivative(f, 1.0) ≈ 4.0
    end
end

# The spmv panel driver only fires when 2·n·sizeof ≥ _SPMV_PANEL_MINXY (level2_packed.jl:185), so the
# public spmv test's small n mostly takes the per-column path. Call it directly across U/L and n that
# leave an NB remainder + a partial last panel. Driver does y += α·A·x (no β) ⇒ oracle β=1.
@testitem "spmv panel driver (_spmv_panel_driver!) direct: U/L, NB remainders + partial panel" setup = [PackBand] begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    @testset "spmv-panel $T up=$up n=$n" for T in (Float32, Float64), up in (true, false),
        n in (1, 3, 5, 9, 17, 33, 48)

        ul = up ? 'U' : 'L'
        AP = randn(T, (n * (n + 1)) ÷ 2); x = randn(T, n); y0 = randn(T, n); α = T(0.7)
        yp = copy(y0); PureBLAS._spmv_panel_driver!(up, n, α, AP, x, yp)   # y += α·A·x (symmetric)
        yr = copy(y0); B.spmv!(ul, α, AP, x, one(T), yr)                  # β=1 accumulate oracle
        @test l2err(yp, yr) < l2tol(T)
    end
end
