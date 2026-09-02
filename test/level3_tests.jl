@testitem "trmm vs OpenBLAS (all side/uplo/trans/diag)" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    tol(::Type{T}) where {T} = sqrt(eps(real(T))) * 50
    @testset "$T s=$side u=$ul t=$ta d=$dg ($m×$n)" for T in (Float32, Float64, ComplexF64),
            side in ('L', 'R'), ul in ('U', 'L'), ta in ('N', 'T', 'C'), dg in ('N', 'U'),
            # last 3 sizes exercise the packed complex-trmm remainder column-tile (k%_CNR ∈ {2,4}) at
            # both uplo (upper-N deep-remainder / lower-N shallow) with a non-mr-multiple m (bottom-row mask).
            (m, n) in ((4, 3), (33, 33), (80, 50), (130, 96), (50, 56), (66, 112), (128, 64))

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
        for i in 1:k
            A[i, i] += (3 + k) * sign(real(A[i, i]) + 0.1)
        end
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
        @test norm(tri(Cr - Cp, ul == 'U')) <= 1.0e-9 * (norm(Cr) + 1)
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
        @test norm(Cr - Cp) <= 1.0e-9 * (norm(Cr) + 1)
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
        @test norm(tri(Cr - Cp, ul == 'U')) <= 1.0e-9 * (norm(Cr) + 1)
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
            @test norm(tri(Cr - Cp, ul == 'U')) <= 1.0e-8 * (norm(Cr) + 1)
        else
            Bm = tr == 'N' ? randn(n, k) : randn(k, n)
            Cr = copy(C0); B.syr2k!(ul, tr, al, A, Bm, be, Cr)
            Cp = copy(C0); PureBLAS.syr2k!(Cp, A, Bm; uplo = ul, trans = tr, alpha = al, beta = be)
            @test norm(tri(Cr - Cp, ul == 'U')) <= 1.0e-8 * (norm(Cr) + 1)
        end
    end
end

# Regression: the complex trsm pointer bases (`_trsm_cgt_L!`, `_trsm_cmplx_dLN!`, `_dRN!`, `_dRC!`) used
# to be gated on `eltype(B)` ALONE. They reinterpret A as `Ptr{eltype(B)}`, so a Float64 or ComplexF32 A
# paired with a ComplexF64 B was read at the wrong element width and ran ~2x off A's allocation, quietly
# returning NaN. LBT/Mode-1 callers always match eltypes, so nothing in the suite exercised it — it was
# found by adversarial review (2026-08-02). Mismatched eltypes must fall through to the generic path and
# still produce the promoted answer.
@testitem "trsm mixed-eltype A/B does not take the complex pointer bases" begin
    using LinearAlgebra, PureBLAS
    T = ComplexF64
    for (ul, sd, ta) in (
            ('U', 'L', 'N'), ('L', 'L', 'N'), ('U', 'R', 'N'),
            ('L', 'R', 'N'), ('U', 'R', 'C'), ('L', 'R', 'C'),
        ), At in (Float64, ComplexF32)
        k, n = 64, 8                                    # k > _ZGT_W and > any direct-base cutoff
        A = At <: Real ? triu(rand(At, k, k) .+ At(4)) : (rand(At, k, k) + At(4) * I)
        B = sd == 'L' ? rand(T, k, n) : rand(T, n, k)
        X = copy(B)
        PureBLAS.trsm!(X, A; side = sd, uplo = ul, transA = ta)
        @test all(isfinite, X)
        R = copy(B)
        LinearAlgebra.BLAS.trsm!(sd, ul, ta, 'N', one(T), Matrix{T}(A), R)
        @test norm(X - R) <= 1.0e-10 * (norm(R) + 1)
    end
end

# Regression: the side-L ragged-column-tail arm (level3.jl, the `1 < nrhs < _vwidth` widening branch)
# used to stage B into the SAME owned buffer that the complex `_trsm!` path takes for `_trtri!`'s
# output. `_trtri!` then wrote A⁻¹ over the staged right-hand side and trsm! returned garbage —
# measured relative error ~1.0, not a rounding difference.
#
# Two properties made it invisible to the existing tests, and both are why this item is shaped the way
# it is. It needs nrhs STRICTLY BETWEEN 1 and `_vwidth(T)` (so ComplexF32 with its 8 lanes, nrhs 2..7 —
# the common nrhs=1 and nrhs≥8 both miss it), and it is HISTORY-DEPENDENT on the scratch buffer's grown
# size: nrhs=5/transA='T' was correct on the first call and wrong on the second. So each case is solved
# TWICE against the same oracle, and a single-call check would not have caught the original bug.
@testitem "trsm! side-L ragged column tail does not alias the trtri scratch (both calls)" begin
    using PureBLAS, LinearAlgebra
    for T in (ComplexF32, ComplexF64, Float32, Float64)
        w = PureBLAS._vwidth(T)
        w <= 2 && continue
        for k in (32, 48), nrhs in 2:min(w - 1, 6), tA in ('N', 'T', 'C')
            T <: Real && tA == 'C' && continue
            A = tril(randn(T, k, k)) + T(k) * I
            B0 = randn(T, k, nrhs)
            opA = tA == 'N' ? A : (tA == 'T' ? transpose(A) : adjoint(A))
            oracle = opA \ B0
            tol = (T <: Union{Float32, ComplexF32}) ? 1.0f-3 : 1e-9
            for rep in 1:2                      # the 2nd call is the one that used to corrupt
                B = copy(B0)
                PureBLAS.trsm!(B, A; side = 'L', uplo = 'L', transA = tA, diag = 'N', alpha = one(T))
                @test isapprox(B, oracle; rtol = tol)
            end
        end
    end
end

# Regression: the side-R pad arm (level3.jl, the `_potrf_needs_pad` branch of `_trsm_right!`) copied A's
# lower triangle into `_trsm_rpack`'s buffer and then held that view ACROSS `_trsm_rl_fused_drv!`, which
# claims `_trsm_rpack` AGAIN as its own solve scratch (`scratch=true`). Same field, same base pointer, so
# the leaf wrote the solution over the coefficients it was still reading. Fixed by the `rpad` /
# `_trsm_rpad` field split — same shape, and same fix, as the `trsmw` bug above.
#
# Two properties shape this item. (1) The arm is gated on `_RL_MR_LIVE > _NVREG` — the leaf only spills,
# and so only pays for the de-aliasing copy, on a 16-register AVX2 file (18 > 16); with AVX-512's 32 it is
# DEAD CODE. This item therefore cannot execute the bug on an AVX-512 box and says so via @test_skip
# instead of passing vacuously. (2) Like `trsmw` it is history-dependent: the inner claim asks for
# mc0 = max(_vwidth, (_L2_BYTES÷2)÷(k·8)) rows, which on a 512 KiB-L2 part exceeds the outer claim and
# REALLOCATES on the first call — only the second call finds the buffer already grown and aliases. Hence
# two solves per shape; a single-call check would not have caught it.
@testitem "trsm! side-R pad arm does not alias the fused-leaf scratch (AVX2 only)" begin
    using PureBLAS, LinearAlgebra
    if PureBLAS._RL_MR_LIVE <= PureBLAS._NVREG
        @info "side-R pad arm is dead on this CPU — it needs a spilling leaf (_RL_MR_LIVE > _NVREG), \
            true only on a 16-register AVX2 file. No runtime test can reach the branch here; the static \
            guard is test/workspace_lint_tests.jl." _RL_MR_LIVE = PureBLAS._RL_MR_LIVE _NVREG = PureBLAS._NVREG
        @test_skip PureBLAS._RL_MR_LIVE > PureBLAS._NVREG
    else
        # Shapes DERIVED from the guards, not hardcoded, so this reaches the arm on any AVX2 box:
        # `_potrf_needs_pad` wants lda·8 a multiple of a quarter L1 way and k ≥ 128; `_alias_ld` (which
        # selects the aliasing `scratch=true` leaf) wants ldb a multiple of a whole L1 way in doubles.
        ks = max(1, PureBLAS._L1_WAY_BYTES >> 5)          # lda granularity, in Float64 elements
        k = ks * cld(128, ks)
        m = PureBLAS._L1_WAY_D
        A = tril(randn(k, k)) + k * I
        B0 = randn(m, k)
        # Witnesses: assert the branch is actually reached before trusting the numbers below.
        @test PureBLAS._potrf_needs_pad(A, k)
        @test PureBLAS._alias_ld(stride(B0, 2))
        @test k > PureBLAS._trsm_dbase() && k <= PureBLAS._trsm_r_fuse() && m > PureBLAS._trsm_ncut_r()
        oracle = B0 / transpose(A)
        for rep in 1:2                                     # the 2nd call is the one that used to corrupt
            B = copy(B0)
            PureBLAS.trsm!(B, A; side = 'R', uplo = 'L', transA = 'T', diag = 'N', alpha = 1.0)
            @test isapprox(B, oracle; rtol = 1.0e-9)
        end
    end
end
