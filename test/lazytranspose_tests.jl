# Lazy transpose operands (`A'`, `transpose(A)`) in BLAS-3.
#
# Two separate guarantees, and they failed in different ways before:
#
#   CORRECTNESS. `A'` is `Adjoint`: a plain transpose for real, the CONJUGATE transpose for complex.
#   `transpose(A)` is never conjugated. The ops disagree about which they can express -- syrk/syr2k are
#   symmetric ('N'/'T'), herk/her2k are Hermitian ('N'/'C'), gemm has all three -- so folding a wrapper
#   into a trans char is only valid where the char exists. Folding a complex `A'` into syrk's 'T' would
#   silently drop the conjugation, which is why the fold tests for a specific char rather than for
#   "is it wrapped". A wrong fold here is a wrong NUMBER, not a crash, so every pairing is checked.
#
#   REACHABILITY. The complex small-n rank-k route (`_ctri_unpacked!` / `_ctri2_unpacked!`) reads its
#   operands directly with `stride`/`pointer`. A complex `Adjoint` is not a strided view -- Base defines
#   no `strides` for it, since it conjugates -- so `herk!(C, A')` threw `MethodError(strides)` at every
#   n <= 64 while n >= 128 escaped to the packed path. Real never hit it: a real adjoint IS strided.

@testitem "BLAS-3 accepts lazy transpose operands and gets them right" begin
    using PureBLAS, LinearAlgebra
    wrappers(M) = (("plain", M), ("adjoint", M'), ("transpose", transpose(M)))

    for T in (Float64, ComplexF64), n in (8, 32, 128)
        A0 = randn(T, n, n); B0 = randn(T, n, n)

        for (_, Aw) in wrappers(A0), (_, Bw) in wrappers(B0)
            C = zeros(T, n, n)
            PureBLAS.gemm!(C, Aw, Bw)
            @test C ≈ collect(Aw) * collect(Bw)
        end

        for (_, Aw) in wrappers(A0)
            a = collect(Aw)
            C = zeros(T, n, n); PureBLAS.syrk!(C, Aw; uplo = 'U', trans = 'N')
            @test triu(C) ≈ triu(a * transpose(a))
            C = zeros(T, n, n); PureBLAS.herk!(C, Aw; uplo = 'U', trans = 'N')
            @test triu(C) ≈ triu(a * a')

            for (_, Bw) in wrappers(B0)
                b = collect(Bw)
                C = zeros(T, n, n); PureBLAS.syr2k!(C, Aw, Bw; uplo = 'U', trans = 'N')
                @test triu(C) ≈ triu(a * transpose(b) + b * transpose(a))
                C = zeros(T, n, n); PureBLAS.her2k!(C, Aw, Bw; uplo = 'U', trans = 'N')
                @test triu(C) ≈ triu(a * b' + b * a')
            end
        end
    end
end

@testitem "complex rank-k small-n path is not reachable by a non-strided operand" tags = [:unit] begin
    using PureBLAS, LinearAlgebra
    # n <= 64 is the window that routed into the direct-read kernel; 128 escaped it anyway, so a
    # regression would show up only below the cut. A complex adjoint has no `strides` method, which is
    # exactly the property the guard tests via `_strided1`.
    @test !PureBLAS._strided1(randn(ComplexF64, 32, 32)')
    @test PureBLAS._strided1(randn(ComplexF64, 32, 32))

    for n in (8, 16, 32, 48, 64)
        A = randn(ComplexF64, n, n); B = randn(ComplexF64, n, n)
        # `trans='C'` keeps the wrapper unfolded (the fold only fires on 'N'), so this reaches the
        # routing decision with a genuinely non-strided operand. Must not throw.
        # NOTE the direction: with trans='C' these are XᴴX and XᴴY+YᴴX, NOT the trans='N' forms XXᴴ
        # and XYᴴ+YXᴴ. Cross-checked against BLAS.herk!/her2k! rather than derived by hand.
        X = collect(A'); Y = collect(B')
        C = zeros(ComplexF64, n, n)
        PureBLAS.herk!(C, A'; uplo = 'U', trans = 'C')
        @test triu(C) ≈ triu(X' * X)
        C .= 0
        PureBLAS.her2k!(C, A', B'; uplo = 'U', trans = 'C')
        @test triu(C) ≈ triu(X' * Y + Y' * X)
    end
end

@testitem "_lazyop names the operation, per element type" tags = [:unit] begin
    using PureBLAS, LinearAlgebra
    # Adjoint is transpose for real and conjugate transpose for complex; Transpose is never conjugated.
    # This distinction is what makes the fold legal for some (op, wrapper) pairs and not others.
    @test PureBLAS._lazyop(randn(4, 4)') == 'T'
    @test PureBLAS._lazyop(randn(ComplexF64, 4, 4)') == 'C'
    @test PureBLAS._lazyop(transpose(randn(4, 4))) == 'T'
    @test PureBLAS._lazyop(transpose(randn(ComplexF64, 4, 4))) == 'T'
    @test PureBLAS._lazyop(randn(4, 4)) == 'N'
end
