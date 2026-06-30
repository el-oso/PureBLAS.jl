@testitem "trmm vs OpenBLAS (all side/uplo/trans/diag)" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    tol(::Type{T}) where {T} = sqrt(eps(real(T))) * 50
    @testset "$T s=$side u=$ul t=$ta d=$dg ($m×$n)" for T in (Float32, Float64, ComplexF64),
        side in ('L', 'R'), ul in ('U', 'L'), ta in ('N', 'T', 'C'), dg in ('N', 'U'),
        (m, n) in ((4, 3), (33, 33), (80, 50), (130, 96))

        (T <: Real && ta == 'C') && continue
        k = side == 'L' ? m : n
        A = randn(T, k, k); X = randn(T, m, n)
        for al in (one(T), T(0.7))
            Br = copy(X); B.trmm!(side, ul, ta, dg, al, A, Br)
            Bp = copy(X); PureBLAS.trmm!(Bp, A; side, uplo = ul, transA = ta, diag = dg, alpha = al)
            @test norm(Bp - Br) <= tol(T) * (norm(Br) + 1)
        end
    end
end

@testitem "trsm vs OpenBLAS (all side/uplo/trans/diag)" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    tol(::Type{T}) where {T} = sqrt(eps(real(T))) * 200   # solves amplify; looser than trmm
    @testset "$T s=$side u=$ul t=$ta d=$dg ($m×$n)" for T in (Float32, Float64, ComplexF64),
        side in ('L', 'R'), ul in ('U', 'L'), ta in ('N', 'T', 'C'), dg in ('N', 'U'),
        (m, n) in ((4, 3), (33, 33), (80, 50), (130, 96))

        (T <: Real && ta == 'C') && continue
        k = side == 'L' ? m : n
        A = randn(T, k, k)                                # diagonally dominant ⇒ well-conditioned
        for i in 1:k; A[i, i] += (3 + k) * sign(real(A[i, i]) + 0.1); end
        X = randn(T, m, n)
        for al in (one(T), T(0.8))
            Br = copy(X); B.trsm!(side, ul, ta, dg, al, A, Br)
            Bp = copy(X); PureBLAS.trsm!(Bp, A; side, uplo = ul, transA = ta, diag = dg, alpha = al)
            @test norm(Bp - Br) <= tol(T) * (norm(Br) + 1)
        end
    end
end

@testitem "trmm/trsm dimension checks + AD" begin
    using PureBLAS, LinearAlgebra, ForwardDiff
    @test_throws DimensionMismatch PureBLAS.trmm!(randn(4, 3), randn(5, 5))           # A k≠4
    @test_throws DimensionMismatch PureBLAS.trsm!(randn(4, 3), randn(4, 4); side = 'R')  # A k≠3
    # AD-traceable through the generic path: differentiate the transformed matrix B (Dual flows
    # through; A constant). B must carry the Dual type to hold op(A)·B / op(A)⁻¹·B.
    A = randn(5, 5) + 6I; X = randn(10)
    gm = ForwardDiff.gradient(x -> sum(PureBLAS.trmm!(reshape(copy(x), 5, 2), A; uplo = 'U')), X)
    gs = ForwardDiff.gradient(x -> sum(PureBLAS.trsm!(reshape(copy(x), 5, 2), A; uplo = 'U')), X)
    @test length(gm) == 10 && all(isfinite, gm)
    @test length(gs) == 10 && all(isfinite, gs)
end

@testitem "syrk/herk vs OpenBLAS" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    tri(M, up) = up ? triu(M) : tril(M)
    @testset "syrk $T $ul $tr ($n,$k)" for T in (Float32, Float64, ComplexF64),
        ul in ('U', 'L'), tr in ('N', 'T'), (n, k) in ((5, 7), (60, 40), (100, 100), (130, 70))
        A = tr == 'N' ? randn(T, n, k) : randn(T, k, n)
        C0 = randn(T, n, n); C0 = C0 + transpose(C0); al = T(0.7); be = T(0.3)
        Cr = copy(C0); B.syrk!(ul, tr, al, A, be, Cr)
        Cp = copy(C0); PureBLAS.syrk!(Cp, A; uplo = ul, trans = tr, alpha = al, beta = be)
        @test norm(tri(Cr - Cp, ul == 'U')) <= sqrt(eps(real(T))) * 100 * (norm(Cr) + 1)
    end
    @testset "herk $ul $tr ($n,$k)" for ul in ('U', 'L'), tr in ('N', 'C'),
        (n, k) in ((5, 7), (80, 50), (120, 120))
        A = tr == 'N' ? randn(ComplexF64, n, k) : randn(ComplexF64, k, n)
        C0 = randn(ComplexF64, n, n); C0 = C0 + adjoint(C0); al = 0.7; be = 0.4
        Cr = copy(C0); B.herk!(ul, tr, al, A, be, Cr)
        Cp = copy(C0); PureBLAS.herk!(Cp, A; uplo = ul, trans = tr, alpha = al, beta = be)
        @test norm(tri(Cr - Cp, ul == 'U')) <= 1e-9 * (norm(Cr) + 1)
    end
end

@testitem "symm/hemm vs OpenBLAS" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    @testset "symm $T $side $ul ($n,$m)" for T in (Float32, Float64, ComplexF64),
        side in ('L', 'R'), ul in ('U', 'L'), (n, m) in ((5, 4), (60, 40), (100, 100), (130, 70))
        k = side == 'L' ? n : m
        A = randn(T, k, k); A = A + transpose(A)
        Bm = randn(T, n, m); C0 = randn(T, n, m); al = T(0.7); be = T(0.3)
        Cr = copy(C0); B.symm!(side, ul, al, A, Bm, be, Cr)
        Cp = copy(C0); PureBLAS.symm!(Cp, A, Bm; side, uplo = ul, alpha = al, beta = be)
        @test norm(Cr - Cp) <= sqrt(eps(real(T))) * 100 * (norm(Cr) + 1)
    end
    @testset "hemm $side $ul ($n,$m)" for side in ('L', 'R'), ul in ('U', 'L'),
        (n, m) in ((5, 4), (80, 50), (120, 120))
        k = side == 'L' ? n : m
        A = randn(ComplexF64, k, k); A = A + adjoint(A)
        Bm = randn(ComplexF64, n, m); C0 = randn(ComplexF64, n, m); al = 0.7 + 0.2im; be = 0.3 - 0.1im
        Cr = copy(C0); B.hemm!(side, ul, al, A, Bm, be, Cr)
        Cp = copy(C0); PureBLAS.hemm!(Cp, A, Bm; side, uplo = ul, alpha = al, beta = be)
        @test norm(Cr - Cp) <= 1e-9 * (norm(Cr) + 1)
    end
end

@testitem "syr2k/her2k vs OpenBLAS" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    tri(M, up) = up ? triu(M) : tril(M)
    @testset "syr2k $T $ul $tr ($n,$k)" for T in (Float32, Float64, ComplexF64),
        ul in ('U', 'L'), tr in ('N', 'T'), (n, k) in ((5, 7), (60, 40), (100, 100), (130, 70))
        A = tr == 'N' ? randn(T, n, k) : randn(T, k, n); Bm = tr == 'N' ? randn(T, n, k) : randn(T, k, n)
        C0 = randn(T, n, n); C0 = C0 + transpose(C0); al = T(0.7); be = T(0.3)
        Cr = copy(C0); B.syr2k!(ul, tr, al, A, Bm, be, Cr)
        Cp = copy(C0); PureBLAS.syr2k!(Cp, A, Bm; uplo = ul, trans = tr, alpha = al, beta = be)
        @test norm(tri(Cr - Cp, ul == 'U')) <= sqrt(eps(real(T))) * 100 * (norm(Cr) + 1)
    end
    @testset "her2k $ul $tr ($n,$k)" for ul in ('U', 'L'), tr in ('N', 'C'),
        (n, k) in ((5, 7), (80, 50), (120, 120))
        A = tr == 'N' ? randn(ComplexF64, n, k) : randn(ComplexF64, k, n)
        Bm = tr == 'N' ? randn(ComplexF64, n, k) : randn(ComplexF64, k, n)
        C0 = randn(ComplexF64, n, n); C0 = C0 + adjoint(C0); al = 0.7 + 0.2im; be = 0.4
        Cr = copy(C0); B.her2k!(ul, tr, al, A, Bm, be, Cr)
        Cp = copy(C0); PureBLAS.her2k!(Cp, A, Bm; uplo = ul, trans = tr, alpha = al, beta = be)
        @test norm(tri(Cr - Cp, ul == 'U')) <= 1e-9 * (norm(Cr) + 1)
    end
end

@testitem "syrk/syr2k large-n packed path vs OpenBLAS" begin
    using PureBLAS, LinearAlgebra            # n>448 triggers the single-pass packed kernel + _microkernel_tri!
    import LinearAlgebra.BLAS as B
    tri(M, up) = up ? triu(M) : tril(M)
    @testset "$op $ul $tr ($n,$k)" for op in (:syrk, :syr2k), ul in ('U', 'L'), tr in ('N', 'T'),
        (n, k) in ((500, 300), (512, 512), (700, 128), (513, 600))
        A = tr == 'N' ? randn(n, k) : randn(k, n)
        C0 = randn(n, n); C0 = C0 + transpose(C0); al = 0.8; be = 0.3
        if op === :syrk
            Cr = copy(C0); B.syrk!(ul, tr, al, A, be, Cr)
            Cp = copy(C0); PureBLAS.syrk!(Cp, A; uplo = ul, trans = tr, alpha = al, beta = be)
            @test norm(tri(Cr - Cp, ul == 'U')) <= 1e-8 * (norm(Cr) + 1)
        else
            Bm = tr == 'N' ? randn(n, k) : randn(k, n)
            Cr = copy(C0); B.syr2k!(ul, tr, al, A, Bm, be, Cr)
            Cp = copy(C0); PureBLAS.syr2k!(Cp, A, Bm; uplo = ul, trans = tr, alpha = al, beta = be)
            @test norm(tri(Cr - Cp, ul == 'U')) <= 1e-8 * (norm(Cr) + 1)
        end
    end
end
