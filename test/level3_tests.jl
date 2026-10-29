# α for the trmm threading items. `trmm!` computes `B := α·op(A)·B`, and the serial entry and the
# threaded bands both apply α AFTER the product, elementwise (`_scal_all!`), so they agree bit for
# bit at any α. The items pass an α that is not ±2^j so that they would still see a path applying α
# BEFORE the product, as `_gemm_core!` does by folding it into `_pack_A!`: `Σ (α·aᵢ)·bᵢ` and
# `α·Σ (aᵢ·bᵢ)` differ in the last bit unless α is a power of two, where the multiplication is exact
# and the two are identical. α = 1 would also skip every `isone(alpha) || _scal_all!` line, so a
# loser branch that dropped its scale would pass.
#
# THE VALUE IS IMMATERIAL BEYOND THAT PROPERTY, and the guard is the whole specification — there is
# no better admissible α to find, because the criterion is saturated rather than under-determined. A
# misplaced α moves an affected entry's bits with probability above 0.2 for every non-power-of-two α
# (measured 0.24-0.35 per product and 0.89-0.95 per length-512 dot, values adjacent to a power of two
# sitting at the low end), while each item compares hundreds to hundreds of thousands of entries and
# a misplaced α touches every entry of the affected band. The chance of the defect hiding is then
# below 0.76^500 ≈ 1e-59 for ANY admissible value, so no admissible value is better than another and
# the spread between them decides nothing.
#
# `significand` is taken in absolute value on purpose: it carries the sign, so `significand(-4.0)` is
# -1.0 and an unsigned comparison would admit the negative powers of two.
@testmodule L3Alpha begin
    const ALPHA = 3.7
    (isfinite(ALPHA) && !iszero(ALPHA) && abs(significand(ALPHA)) != 1) || error(
        "L3Alpha.ALPHA = $ALPHA is zero, non-finite, or ±2^j: scaling by it is exact, so a " *
        "misplaced alpha returns the same bits and the trmm threading items cannot see it"
    )
end

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

# The complex gemmtrsm leaf solves LOWER natively (`Val(true)`). It must read only the strictly-lower part
# (plus the diagonal when non-unit): an A poisoned with NaN above the diagonal must still give the right
# answer, and poisoning one strictly-lower entry must bite (positive control — the test can fail).
@testitem "complex gemmtrsm leaf: native lower is correct and never reads above the diagonal" begin
    using PureBLAS, LinearAlgebra
    # THE LEAF IS WIDTH-GATED. It needs an in-register WxW transpose and only `_tr8x8`/`_tr4x4`
    # exist, so `_ZGT_ON` is false at any other vector width and the dLN base runs instead. Calling
    # `_trsm_cgt_L!` there raises from the missing transpose rather than testing anything.
    if !PureBLAS._ZGT_ON
        @test_skip "complex gemmtrsm leaf is off at _ZGT_W=$(PureBLAS._ZGT_W)"
    else
        T = ComplexF64
        for k in (1, 7, 8, 9, 17, 32, 64), m in (1, 5, 8, 18), unit in (true, false)
            L = tril(randn(T, k, k), -1) + 4I
            Lp = copy(L)
            for j in 1:k, i in 1:(j - 1)
                Lp[i, j] = T(NaN, NaN)
            end
            unit && for i in 1:k
                Lp[i, i] = T(NaN, NaN)
            end
            B0 = randn(T, k, m)
            R = copy(B0); LinearAlgebra.BLAS.trsm!('L', 'L', 'N', unit ? 'U' : 'N', one(T), unit ? tril(L, -1) + I : L, R)
            X = copy(B0); PureBLAS._trsm_cgt_L!(Val(true), unit, k, Lp, X)
            @test all(isfinite, X)
            @test norm(X - R) <= 1.0e-10 * (norm(R) + 1)
            Xp = copy(B0); PureBLAS.trsm!(Xp, Lp; side = 'L', uplo = 'L', transA = 'N', diag = unit ? 'U' : 'N')
            @test norm(Xp - R) <= 1.0e-10 * (norm(R) + 1)
            if k > 1
                C = copy(Lp); C[k, 1] = T(NaN, NaN)
                Y = copy(B0); PureBLAS._trsm_cgt_L!(Val(true), unit, k, C, Y)
                @test !all(isfinite, Y)
            end
        end
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

# A side-R worker is handed a ROW BAND of B and routes from `nroute`, the whole problem's row count,
# so it takes the algorithm the unsplit solve takes. Bands round to `_MR·W`, which is FINER than the
# `_SME_MR` modulus the SME recursion asks for, so a band's own height can fail that modulus where
# the unsplit height passes it — at m=2048 with three workers the bands are 680/684/684. A band
# routing on its own height takes the fused leaf while the unsplit call recurses, and leaf and
# recursion sum in a different order, so the result would depend on the worker count.
#
# THE WORKER COUNT, NOT A RAGGED `m`, IS WHAT REACHES THIS: every band is a whole `_MR·W` block, and
# only some worker counts make them whole `_SME_MR` blocks as well. `k` must be at least 64, because
# `_sme_tile_ok` needs `min(m, k÷2) ≥ 2·_SME_MR` before the recursion is eligible at all.
@testitem "trmm side-L: a concurrent caller that loses the claim still multiplies a TRIANGLE" tags = [:checks] setup = [L3Alpha] begin
    using PureBLAS, LinearAlgebra, Base.Threads
    P = PureBLAS
    if Threads.nthreads() < 2
        @test_skip Threads.nthreads() >= 2
    else
        # WHAT THIS CATCHES, and it caught it: the side-L column route had no lost-claim branch, so a
        # loser fell through to the pool's generic `_gemm_core!` fallback — a FULL matrix product. It
        # read A's unreferenced triangle as data and could not honour an implicit unit diagonal, so a
        # caller that merely lost the claim got a wrong answer rather than a slow one. Only
        # concurrency reaches it: with one caller the claim always succeeds, and every
        # single-threaded test passed over it.
        #
        # The operand is built to make that failure loud. `uplo='L'` never references the STRICTLY
        # UPPER triangle, so it carries a sentinel: a full product reads it as data and lands far
        # outside the tolerance band.
        #
        # SENTINEL PROVENANCE. A wrong full product adds a spurious `_SENTINEL * B[j,c]` to an entry
        # whose true value is O(k·|B|), so it is detected once `_SENTINEL / k > rtol`; with the
        # `rtol` below that is `_SENTINEL > 8·k²·eps`, about 5e-10 at k = 512. The value is round and
        # far above that bound rather than tight to it, so the margin survives a change to `rtol`, to
        # `k`, or to `B`'s scale — and the POSITIVE CONTROL below computes the wrong answer and
        # asserts it is rejected, so the margin is demonstrated per run instead of derived once here.
        #
        # ⚠ NOT NaN, deliberately. A NaN sentinel would also catch an implementation that packs the
        # full tile and masks the unreferenced part (`NaN·0` poisons, `_SENTINEL·0` does not), which
        # is a real spec violation. It is rejected anyway: the pack buffers and the arena are shared
        # and reused across calls, so a NaN that reached one could surface in an unrelated item with
        # nothing pointing back here. A masked read is left to the guard-page checks that already
        # cover it.
        #
        # THE REFERENCE IS BUILT FROM THE CLEAN MATRIX. Taken from the sentinel matrix instead, this
        # item would be asserting that OpenBLAS also never touches the unreferenced triangle — true
        # by the spec, but an implementation that packs a full block and masks it would move the
        # reference too and the item would fail for the wrong reason.
        #
        # `diag = 'U'` is exercised because a gemm cannot honour an implicit unit diagonal at all.
        # With the lost-claim branch removed, every caller that loses the claim fails; how many that
        # is varies with the race, so the count is not asserted.
        _SENTINEL = 1.0e6
        nt = min(4, Threads.nthreads())
        P.set_num_threads(nt)
        k = clamp(4 * P._TRMM_BASE_R, P._TRMM_BASE_R + 1, P._TRMM_PACK_MIN)
        n = max(2 * P._NR, 2 * cld(P._GEMM_MT_WORK, k * k))
        Aref = tril(randn(k, k)) + k * I          # referenced half only; the other half is zero
        A = copy(Aref)
        for c in 1:k, r in 1:(c - 1)
            A[r, c] = _SENTINEL                   # strictly upper: never referenced for uplo='L'
        end
        B0 = randn(k, n)
        # A length-k dot product carries a relative error of order k·eps, and this is PureBLAS against
        # OpenBLAS summing in a different order — so the band is derived from k and moves with it. The
        # sibling thread-invariance item compares bit patterns instead, because there both sides are
        # PureBLAS and any difference at all is the defect.
        rtol = 8 * k * eps(Float64)
        # `L3Alpha.ALPHA` rather than 1, and for a DIFFERENT reason than the bit-identical items
        # have: at α = 1 the loser branch's `isone(alpha) || _scal_all!` never runs, so a loser that
        # dropped its scale entirely would still pass here. α PLACEMENT is invisible to this item —
        # it compares against OpenBLAS at `rtol` and a placement difference is O(eps) — so only the
        # bit-identical items below can see that half.
        alpha = L3Alpha.ALPHA
        for dg in ('N', 'U')
            ref = copy(B0)
            BLAS.trmm!('L', 'L', 'N', dg, alpha, Aref, ref)
            @test all(isfinite, ref)              # the sentinel can only fire on a read of A's upper half
            # POSITIVE CONTROL: the full product is what the missing branch fell through to, so
            # computing it here proves the operand and the band can tell the two apart. Without this
            # the item could pass by being blind rather than by being right.
            bad = similar(ref)
            BLAS.gemm!('N', 'N', alpha, A, B0, 0.0, bad)
            @test !isapprox(bad, ref; rtol)
            outs = [copy(B0) for _ in 1:nt]
            Threads.@threads :static for t in 1:nt
                for _ in 1:2
                    copyto!(outs[t], B0)
                    P.trmm!(outs[t], A; side = 'L', uplo = 'L', transA = 'N', diag = dg, alpha = alpha)
                end
            end
            for t in 1:nt
                @test isapprox(outs[t], ref; rtol)
            end
        end
        P.set_num_threads(1)
    end
end

@testitem "trmm side-L: bit-identical at every thread count" tags = [:checks] setup = [L3Alpha] begin
    using PureBLAS, LinearAlgebra
    using Base.Threads: nthreads
    const P = PureBLAS
    if nthreads() < 2
        @test_skip "needs ≥2 julia threads"
    else
        bitsame(X, Y) = all(i -> reinterpret(UInt64, X[i]) === reinterpret(UInt64, Y[i]), eachindex(X))
        # Side L splits the COLUMNS of B, and the one thing that could make a band disagree with the
        # unsplit call is the off-diagonal `_gemm_acc!`, whose kernel choice reads `n`. The band
        # carries the parent's width as a route token, so these must agree bit for bit. k is chosen
        # inside the band the column route is armed for — above `_TRMM_BASE_R`, at or below
        # `_TRMM_PACK_MIN` — because outside it a different driver owns the call.
        #
        # liveness-ok: this route's admission is decided by `_trsm_workers` against `_GEMM_MT_WORK`,
        # not by `_L1_MT_MIN` (a BLAS-1 byte floor) and not by an SME predicate (side L has none).
        # The item asks `_trsm_workers` DIRECTLY below and skips when it declines, so the gate is
        # consulted rather than guessed — which is what the lint is for. The sizes are derived from
        # the two thresholds that bound the route, not chosen.
        # THE BAND IS NOT THE SAME WIDTH ON EVERY µARCH and it can be narrow: `_TRMM_PACK_MIN` is
        # `(5 * _GEMM_UNPACK_MAX) ÷ 2`, which is 1120 where `_GEMM_UNPACK_MAX` is 448 and 240 where it
        # is 96. A k derived from `_TRMM_BASE_R` alone overshoots the top on the narrow hosts — this
        # item asserted itself into that and failed on AVX2 — so k is clamped INTO the band and the
        # band being empty is a skip rather than a false assertion.
        lo = P._TRMM_BASE_R + 1
        hi = P._TRMM_PACK_MIN
        k = clamp(4 * P._TRMM_BASE_R, lo, hi)
        # the narrowest B that clears the flop floor, doubled so the floor is not the thing under test
        n = max(2 * P._NR, 2 * cld(P._GEMM_MT_WORK, k * k))
        # LIVENESS, read from the pool's own publish counter rather than from a buffer. `gen` is
        # bumped to publish a job and its low bits carry the worker count, so a bumped `gen` with
        # nw > 1 means a split actually ran. This cannot be satisfied by a build that threads
        # nothing, which is how this class of item goes quiet — and unlike a pack-slot count it does
        # not assume WHICH route serves the call, so it needs no per-host guard.
        #
        # It is still guarded on whether the entry ARMS, using the entry's own worker function: a
        # host whose floor declines this shape would otherwise fail for threading correctly.
        # ARMED MEANS THE ENTRY'S WHOLE CONDITION, not just the worker count. The route also requires
        # k to be inside the band, and asking only `_trsm_workers` is how this item first failed on a
        # host whose band does not reach the k it had chosen: the workers said yes, the entry said no,
        # and the witness then asserted that a job it had declined to publish had been published.
        P.set_num_threads(max(2, nthreads()))
        armed = lo <= hi && k > P._TRMM_BASE_R && k <= P._TRMM_PACK_MIN && P._trsm_workers(k, n) > 1
        A = triu(randn(k, k)) + k * I
        B0 = randn(k, n)
        P.set_num_threads(1)
        r = copy(B0); P.trmm!(r, A; side = 'L', uplo = 'U', transA = 'N', diag = 'N', alpha = L3Alpha.ALPHA)
        if armed
            pool = P._gemm_pool(Float64)
            P.set_num_threads(max(2, nthreads()))
            g0 = @atomic pool.gen
            Bw = copy(B0); P.trmm!(Bw, A; side = 'L', uplo = 'U', transA = 'N', diag = 'N', alpha = L3Alpha.ALPHA)
            g1 = @atomic pool.gen
            @test g1 != g0                               # a job was published
            @test P._mt_gen_nw(g1) > 1                   # and with more than one worker
        else
            @test_skip "the amortisation floor declines this shape on this host"
        end
        # `L3Alpha.ALPHA` rather than 1: the scaling has to ROUND or a misplaced alpha returns the
        # same bits and this item cannot see it. That requirement is enforced in `L3Alpha`.
        for up in ('L', 'U'), tr in ('N', 'T'), dg in ('N', 'U')
            G = up == 'U' ? triu(randn(k, k)) + k * I : tril(randn(k, k)) + k * I
            X0 = randn(k, n)
            P.set_num_threads(1)
            ref = copy(X0); P.trmm!(ref, G; side = 'L', uplo = up, transA = tr, diag = dg, alpha = L3Alpha.ALPHA)
            for nt in 2:nthreads()
                P.set_num_threads(nt)
                g = copy(X0); P.trmm!(g, G; side = 'L', uplo = up, transA = tr, diag = dg, alpha = L3Alpha.ALPHA)
                @test bitsame(g, ref)
            end
        end
        P.set_num_threads(1)
    end
end

@testitem "trmm side-R: bit-identical at every thread count" tags = [:checks] setup = [L3Alpha] begin
    using PureBLAS, LinearAlgebra
    using Base.Threads: nthreads
    const P = PureBLAS
    if nthreads() < 2
        @test_skip "needs ≥2 julia threads"
    else
        bitsame(X, Y) = all(i -> reinterpret(UInt64, X[i]) === reinterpret(UInt64, Y[i]), eachindex(X))
        # THIS IS A REGRESSION NET, NOT A WITNESS OF THE MECHANISM, and the difference is stated because
        # the stronger claim is the one a reader will assume. The side-R entry carries no size floor:
        # every path a band takes receives B's undivided row count, so a band and the unsplit call key
        # `_use_unpacked` on the same extent. The ROUTES provably diverge without that — at m = 2048,
        # k = 256 the flat loop's off-diagonal gemm has `m` = B's rows, and
        #
        #     _use_unpacked(2048, …) = false    the unsplit call takes the BLOCKED path
        #     _use_unpacked( 336, …) = true     a band takes the UNPACKED path
        #
        # because 2048 is above `_GEMM_UNPACK_MAX` and a 336-row band is below it.
        #
        # THE BITS DO NOT FOLLOW, THOUGH, AND THAT WAS MEASURED RATHER THAN ASSUMED. With the token
        # hand-off deliberately broken, 144 armed shapes — m in 512..4096, k in 64..1536, both
        # triangles, both transposes, every worker count 2..6 — agree bit for bit with the serial call.
        # So on this microarchitecture the two routes compute the same bits here: `_gemm_accR!` passes
        # alpha = beta = 1, which makes alpha placement and beta handling no-ops, and the inner k stays
        # at or under `_KC` so neither route chunks the reduction. No positive control for the token
        # exists on this box. It is kept because making the routes agree is the invariant req#11 asks
        # for, not because a divergence was reproduced — and a reader must not be told otherwise.
        #
        # What this item DOES certify is that a worker ran and the answer did not move.
        m, k = 2048, 256
        nws = map(2:nthreads()) do nt
            P.set_num_threads(nt)
            P._trsm_workers_r(k, m, Float64)
        end
        @test maximum(nws) > 1
        # LIVENESS, and it reads the driver rather than the admission predicate. `_trmmr_prefit!` grows
        # one `_gpack_shared_pool` slot per worker, from the driver, inside the claim — so a grown slot
        # count means a job was published. Without this, "every thread count agrees" would be satisfied
        # by a build where nothing threaded at all, which is how this class of item goes quiet.
        #
        # THE POOL IS ONLY THE THREADED PATH WHERE THE PACK DRIVER OWNS THE CALL. A host whose matrix
        # coprocessor takes this shape never reaches `_trmmr_prefit!`, so the slot count stays at zero
        # and an unguarded assertion there asserts the ABSENCE of that coprocessor instead of liveness
        # — measured on an M6 at this shape: slots 0 with the coprocessor entry live, 12 with it
        # cleared. The guard therefore asks `_trmmr_sme_owns`, which is the predicate the arming site
        # itself consults, rather than a proxy for it: a guard that is not the code's own question can
        # disagree with the code. The bit-identity assertions below are NOT guarded — they must hold on
        # every route, which is the invariant req#11 actually asks for.
        P.set_num_threads(1)
        Aw = triu(randn(k, k)) + k * I
        Bw = randn(m, k); P.trmm!(Bw, Aw; side = 'R', uplo = 'U')
        serial_slots = length(P._gpack_shared_pool(Float64))
        P.set_num_threads(maximum(2:nthreads()))
        Bw2 = randn(m, k); P.trmm!(Bw2, Aw; side = 'R', uplo = 'U')
        if P._trmmr_sme_owns(Float64, k, m, Aw, Bw2)
            @test_skip "coprocessor owns this shape — the pack driver is not the threaded path here"
        else
            @test length(P._gpack_shared_pool(Float64)) > serial_slots
        end
        # `L3Alpha.ALPHA` rather than 1: the scaling has to ROUND or a misplaced alpha returns the
        # same bits and this item cannot see it. That requirement is enforced in `L3Alpha`.
        for up in ('L', 'U'), tr in ('N', 'T'), dg in ('N', 'U')
            G = randn(k, k)
            A = (up == 'U' ? triu(G) : tril(G)) + k * I
            B0 = randn(m, k)
            P.set_num_threads(1)
            r = copy(B0); P.trmm!(r, A; side = 'R', uplo = up, transA = tr, diag = dg, alpha = L3Alpha.ALPHA)
            for nt in 2:nthreads()
                P.set_num_threads(nt)
                g = copy(B0); P.trmm!(g, A; side = 'R', uplo = up, transA = tr, diag = dg, alpha = L3Alpha.ALPHA)
                @test bitsame(g, r)
            end
        end
        P.set_num_threads(1)
    end
end

@testitem "trsm side-R: bit-identical at every thread count" tags = [:checks] begin
    using PureBLAS, LinearAlgebra
    using Base.Threads: nthreads
    const P = PureBLAS
    m, k = 2048, 64
    if nthreads() < 2
        @test_skip "needs ≥2 julia threads"
    else
        bitsame(X, Y) = all(i -> reinterpret(UInt64, X[i]) === reinterpret(UInt64, Y[i]), eachindex(X))
        # Liveness first: the band split has to be armed at this shape or the comparison is vacuous.
        # Then the hazard itself — a worker count whose bands are NOT whole `_SME_MR` blocks while the
        # unsplit height is. A box that cannot reach one says so rather than passing for free.
        nws = map(2:nthreads()) do nt
            P.set_num_threads(nt)
            P._trsm_workers_r(k, m, Float64)
        end
        bandlens(nw) = (P._trsm_rchunk(m, nw, i, Float64)[2] for i in 1:nw)
        ragged = any(nw -> nw > 1 && any(l -> l % P._SME_MR != 0, bandlens(nw)), nws)
        @test maximum(nws) > 1
        @test m % P._SME_MR == 0
        if !(ragged && P._SME_F64)
            @test_skip "no reachable worker count splits m=$m into a band that is not a whole \
                _SME_MR block, or f64 SME is off (_SME_F64 = $(P._SME_F64))"
        end
        for (up, tr) in (('L', 'N'), ('L', 'T'), ('U', 'N'), ('U', 'T'))
            G = randn(k, k)
            A = (up == 'U' ? triu(G) : tril(G)) + k * I
            B0 = randn(m, k)
            P.set_num_threads(1)
            r = copy(B0); P.trsm!(r, A; side = 'R', uplo = up, transA = tr)
            for nt in 2:nthreads()
                P.set_num_threads(nt)
                g = copy(B0); P.trsm!(g, A; side = 'R', uplo = up, transA = tr)
                @test bitsame(g, r)
            end
        end
        P.set_num_threads(1)
    end
end

# Regression: the complex `hemm!` PACKED path copied the Hermitian diagonal's imaginary part verbatim
# into the packed panel. Reference zhemm never READS imag(A[i,i]) — the Hermitian convention says it is
# zero — so a caller may legitimately leave anything there, and LinearAlgebra's own `Hermitian` wrapper
# does exactly that. PureBLAS then disagreed with OpenBLAS by ~2-4.5e-2.
#
# Two things made this survive: it needs junk in the diagonal's imaginary part (a matrix built as
# `(A + A')/2` has an exactly-real diagonal and cannot expose it), AND n above the packing crossover —
# n <= 32 takes the unpacked route and was always correct. `hemv!` and `_symm_materialize!` both already
# realified, so `hemm!` was the odd one out; found by an adversarial review, not by the suite.
@testitem "hemm! ignores the Hermitian diagonal's imaginary part (packed path, vs OpenBLAS)" begin
    using PureBLAS, LinearAlgebra
    for T in (ComplexF64, ComplexF32), side in ('L', 'R'), uplo in ('L', 'U'), n in (8, 16, 33, 64, 128)
        nrhs = 3
        A = randn(T, n, n); A = (A + A') / 2
        for i in 1:n
            A[i, i] += T(0, 0.5)          # junk the convention says must be ignored
        end
        B = side == 'L' ? randn(T, n, nrhs) : randn(T, nrhs, n)
        pb = zeros(T, size(B))
        PureBLAS.hemm!(pb, A, B; side = side, uplo = uplo, alpha = one(T), beta = zero(T))
        ob = zeros(T, size(B))
        BLAS.hemm!(side, uplo, one(T), A, B, zero(T), ob)
        @test isapprox(pb, ob; rtol = T === ComplexF32 ? 1.0f-4 : 1e-12)
    end
end
