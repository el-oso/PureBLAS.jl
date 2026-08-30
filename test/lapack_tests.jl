@testitem "potrf (Cholesky) vs LAPACK — lower/upper, sizes" begin
    using PureBLAS, LinearAlgebra
    spd(T, n) = (M = randn(T, n, n); M * M' + n * I)
    tol(::Type{T}) where {T} = sqrt(eps(real(T))) * 50
    @testset "$T n=$n" for T in (Float32, Float64),
            n in (1, 2, 5, 31, 96, 256, 512, 513, 700, 1025, 1536, 2049)

        A = Matrix{T}(spd(T, n)); F = cholesky(A)              # LAPACK reference
        L = PureBLAS.potrf!(copy(A); uplo = 'L')
        U = PureBLAS.potrf!(copy(A); uplo = 'U')
        @test norm(tril(L) - Matrix(F.L)) <= tol(T) * (norm(F.L) + 1)
        @test norm(triu(U) - Matrix(F.U)) <= tol(T) * (norm(F.U) + 1)
        @test norm(tril(L) * tril(L)' - A) <= tol(T) * (norm(A) + 1)   # reconstruction
    end
end

@testitem "potrf — non-positive-definite throws, shape check" begin
    using PureBLAS, LinearAlgebra
    @test_throws PosDefException PureBLAS.potrf!([1.0 2.0; 2.0 1.0]; uplo = 'L')   # indefinite
    @test_throws PosDefException PureBLAS.potrf!(zeros(3, 3); uplo = 'U')          # singular
    @test_throws DimensionMismatch PureBLAS.potrf!(randn(3, 4))
end

@testitem "pbtrf/pbtrs (banded Cholesky) — both triangles, kd across the blocked crossover" begin
    using PureBLAS, LinearAlgebra, Random
    # pbtrf had NO correctness coverage — only a StrictMode trim check at uplo='L', kd=2. Everything at
    # kd >= _pbtrf_cross (the BLOCKED kernel, both triangles, the panel/corner blocks and the work-array
    # copy-back) was untested, so a broken blocked kernel passed the whole suite. Julia's stdlib has no
    # pbtrf wrapper, so the oracle here is self-contained: factor the band, reconstruct UᴴU / LLᴴ, and
    # compare against the dense matrix it came from. kd is chosen to straddle the crossover (32 on Zen4)
    # and to include a power-of-two and the nb boundary.
    function dense_from_band(AB, n, kd, uplo)
        A = zeros(eltype(AB), n, n)
        if uplo == 'L'
            for j in 1:n, i in j:min(n, j + kd); A[i, j] = AB[1 + i - j, j]; end
        else
            for j in 1:n, i in max(1, j - kd):j; A[i, j] = AB[kd + 1 + i - j, j]; end
        end
        A
    end
    @testset "$T uplo=$uplo kd=$kd n=$n" for T in (Float64, Float32, ComplexF64, ComplexF32),
            uplo in ('L', 'U'), kd in (1, 2, 8, 31, 32, 33, 40, 64), n in (65, 300)

        kd >= n && continue
        R = real(T)
        Random.seed!(hash((T, uplo, kd, n)))
        AB = zeros(T, kd + 1, n)                       # diagonally dominant ⇒ SPD/HPD
        for j in 1:n
            if uplo == 'L'
                AB[1, j] = T(2kd + 4) + abs(randn(R))
                for i in 1:min(kd, n - j); AB[1 + i, j] = randn(T) * R(0.3); end
            else
                AB[kd + 1, j] = T(2kd + 4) + abs(randn(R))
                for i in 1:min(kd, j - 1); AB[kd + 1 - i, j] = randn(T) * R(0.3); end
            end
        end
        Aband = dense_from_band(AB, n, kd, uplo)
        Afull = uplo == 'L' ? Matrix(Hermitian(Aband, :L)) : Matrix(Hermitian(Aband, :U))
        F = PureBLAS.pbtrf!(copy(AB); uplo = uplo, kd = kd)
        Fd = dense_from_band(F, n, kd, uplo)
        rec = uplo == 'L' ? Fd * Fd' : Fd' * Fd        # A = L·Lᴴ  /  A = Uᴴ·U
        @test maximum(abs, rec - Afull) < sqrt(eps(R)) * 50 * (norm(Afull) + 1)
        # and the solve round-trips
        B = randn(T, n, 2)
        X = PureBLAS.pbtrs!(F, copy(B); uplo = uplo, kd = kd)
        @test maximum(abs, Afull * X - B) < sqrt(eps(R)) * 200 * (norm(Afull) + 1)

        # uplo='U' has TWO blocked kernels and pbtrf! picks between them at a MEASURED bandwidth
        # crossover (_pbtrf_ucross), so which one the loop above exercised depends on the host —
        # on this box it is 256 for Float64 and typemax for Float32, i.e. the native kernel would
        # never be reached by any kd this test can afford. Call both directly so each is covered
        # on every machine regardless of what the harness measured.
        # NOT guarded on _pbtrf_cross(T): that is a per-process measurement, so gating on it made
        # the number of assertions vary between runs (336 one run, 332 the next). kd ≥ 4 is the
        # blocked kernels' own documented precondition (reference dpbtrf takes the blocked branch
        # only for 1 < NB ≤ KD, and nb = min(nb_tuned, kd) here).
        if uplo == 'U' && T <: PureBLAS.BlasFloat && kd >= 4
            for k! in (PureBLAS._pbtrf_repack_U!, PureBLAS._pbtrf_blocked_U!)
                Fk = k!(copy(AB), n, kd)
                Fkd = dense_from_band(Fk, n, kd, 'U')
                @test maximum(abs, Fkd' * Fkd - Afull) < sqrt(eps(R)) * 50 * (norm(Afull) + 1)
            end
        end
    end
end

@testitem "getrs/potrs/trtrs (solves on given factors) vs LAPACK" begin
    using PureBLAS, LinearAlgebra, Random
    # These back `lu(A) \ b`, `cholesky(A) \ b` and triangular `\`. Until now they existed ONLY as
    # C-ABI shims, so there was no native entry point to test or to gate — and gating them turned up
    # getrs at 0.07-0.27 vs OpenBLAS for a single RHS. nrhs=1 is covered explicitly because that is
    # the `\` case and the one that was broken.
    @testset "$T n=$n nrhs=$nrhs" for T in (Float64, Float32, ComplexF64, ComplexF32),
            n in (1, 7, 64, 129), nrhs in (1, 3)

        R = real(T); tol = sqrt(eps(R)) * 100
        Random.seed!(hash((T, n, nrhs)))
        A = randn(T, n, n) + n * I
        B = randn(T, n, nrhs)
        # Factors come from LAPACK on purpose: these solves are specified to operate on
        # standard-convention factors from ANY backend (that is what makes forwarding them correct
        # under a mixed backend), so testing against reference-produced factors is the faithful check.
        F, ipiv, _ = LinearAlgebra.LAPACK.getrf!(copy(A))
        for tr in ('N', 'T', 'C')
            x1 = PureBLAS.getrs!(copy(F), ipiv, copy(B); trans = tr)
            x2 = LinearAlgebra.LAPACK.getrs!(tr, copy(F), ipiv, copy(B))
            @test maximum(abs, x1 - x2) < tol * (norm(x2) + 1)
        end
        S = A * A' + n * I                                  # HPD for the Cholesky solve
        for u in ('L', 'U')
            C = copy(S); PureBLAS.potrf!(C; uplo = u)
            y1 = PureBLAS.potrs!(copy(C), copy(B); uplo = u)
            y2 = LinearAlgebra.LAPACK.potrs!(u, copy(C), copy(B))
            @test maximum(abs, y1 - y2) < tol * (norm(y2) + 1)
        end
        for u in ('L', 'U'), tr in ('N', 'T', 'C'), dg in ('N', 'U')
            Tm = u == 'L' ? tril(A) : triu(A)
            z1 = PureBLAS.trtrs!(copy(Tm), copy(B); uplo = u, trans = tr, diag = dg)
            z2 = LinearAlgebra.LAPACK.trtrs!(u, tr, dg, copy(Tm), copy(B))
            @test maximum(abs, z1 - z2) < tol * (norm(z2) + 1)
        end
    end
end

@testitem "pptrf/pptrs (packed Cholesky) — both triangles, s/d/c/z" begin
    using PureBLAS, LinearAlgebra, Random
    # Same coverage hole pbtrf had: pptrf's only test was a StrictMode trim check at uplo='L', so
    # neither triangle was ever checked for correctness. Julia's stdlib has no pptrf! wrapper either,
    # so the oracle is again self-contained — factor, reconstruct, compare to the dense original.
    packL(M, n) = [M[i, j] for j in 1:n for i in j:n]     # column-packed lower
    packU(M, n) = [M[i, j] for j in 1:n for i in 1:j]     # column-packed upper
    function unpack(AP, n, uplo)
        M = zeros(eltype(AP), n, n); k = 1
        if uplo == 'L'
            for j in 1:n, i in j:n; M[i, j] = AP[k]; k += 1; end
        else
            for j in 1:n, i in 1:j; M[i, j] = AP[k]; k += 1; end
        end
        M
    end
    @testset "$T uplo=$uplo n=$n" for T in (Float64, Float32, ComplexF64, ComplexF32),
            uplo in ('L', 'U'), n in (1, 2, 7, 32, 33, 64, 129)

        R = real(T)
        Random.seed!(hash((T, uplo, n)))
        B = randn(T, n, n); A = B * B' + n * I           # HPD
        A = (A + A') / 2
        Af = Matrix{T}(A)
        AP = uplo == 'L' ? packL(Af, n) : packU(Af, n)
        F = PureBLAS.pptrf!(copy(AP); uplo = uplo)
        Fd = unpack(F, n, uplo)
        rec = uplo == 'L' ? Fd * Fd' : Fd' * Fd
        @test maximum(abs, rec - Af) < sqrt(eps(R)) * 50 * (norm(Af) + 1)
        b = randn(T, n, 2)
        x = PureBLAS.pptrs!(copy(F), copy(b); uplo = uplo)
        @test maximum(abs, Af * x - b) < sqrt(eps(R)) * 500 * (norm(Af) + 1)
    end
end

@testitem "potrf — PosDefException reports the FIRST failing column (LAPACK info)" begin
    using PureBLAS, LinearAlgebra, Random
    import LinearAlgebra.LAPACK as LA
    # LAPACK's dpotrf sets info = the first column whose pivot is non-positive; `cholesky(A; check=true)`
    # surfaces it, so a wrong value is user-visible through the C-ABI. The blocked/hybrid drivers factor
    # sub-blocks of VIEWS, so the base kernel's column index has to be lifted back through every recursion
    # level — this pins that. (Previously every failure reported column 1 regardless.)
    function pb_info(A, uplo)
        B = copy(A)
        try
            PureBLAS.potrf!(B; uplo = uplo); return 0
        catch e
            e isa PosDefException && return e.info
            rethrow()
        end
    end
    ref_info(A, uplo) = LA.potrf!(uplo, copy(A))[2]        # LAPACK.potrf! RETURNS (A, info); it does not throw
    @testset "$T uplo=$uplo n=$n" for T in (Float64, Float32, ComplexF64, ComplexF32),
            uplo in ('L', 'U'), n in (8, 17, 33, 64, 129, 257)

        Random.seed!(hash((T, n)))
        X = randn(T, n, n)
        A0 = Matrix(T <: Complex ? Hermitian(X'X + real(T)(n) * I, :L) : Symmetric(X'X + T(n) * I, :L))
        @test pb_info(A0, uplo) == 0                                  # SPD ⇒ no throw
        for col in unique(clamp.([1, 2, n ÷ 2, n - 1, n], 1, n))
            A = copy(A0)
            A[col, col] = -abs(A[col, col]) * T(1.0e-3)                # poison one pivot
            @test pb_info(A, uplo) == ref_info(A, uplo)
        end
    end
end

@testitem "geqrf (QR) vs LAPACK — square/tall/wide, R + reconstruction" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.LAPACK
    maxe(x, y) = maximum(abs.(x .- y)) / max(maximum(abs.(y)), 1.0e-300)
    # reconstruct A = Q*R from faer-convention factored output (H_k = I − v vᵀ/τ, τ=Inf ⇒ identity)
    function recon(F, tau)
        m, n = size(F); k = min(m, n)
        R = [i <= j ? F[i, j] : 0.0 for i in 1:m, j in 1:n]
        for kk in k:-1:1
            isfinite(tau[kk]) || continue
            v = zeros(m); v[kk] = 1.0; v[(kk + 1):m] = F[(kk + 1):m, kk]
            R .-= (v * (v' * R)) ./ tau[kk]
        end
        R
    end
    @testset "$m×$n" for (m, n) in ((1, 1), (8, 8), (40, 25), (64, 64), (129, 96), (200, 200), (96, 160), (600, 513))
        A0 = randn(m, n)
        F = copy(A0); tau = zeros(min(m, n)); PureBLAS.geqrf!(F, tau)
        Fl = copy(A0); LAPACK.geqrf!(Fl)                                   # LAPACK reference
        k = min(m, n)
        @test maxe(abs.(triu(F)[1:k, :]), abs.(triu(Fl)[1:k, :])) < 1.0e-11   # |R| matches up to sign
        @test maxe(recon(F, tau), A0) < 1.0e-11                              # Q·R ≈ A
    end
    @test_throws DimensionMismatch PureBLAS.geqrf!(randn(5, 5), zeros(2))
end

@testmodule WYHelpers begin
using PureBLAS, LinearAlgebra
export explicit_v_panel, lapack_tau, wy_qtc!, wy_qc!

# Build the EXPLICIT unit-diagonal panel wy_t!/wy_apply! require (see src/wy.jl's header:
# `Apanel` must NOT be a raw post-factorization view — its diagonal/above-diagonal entries
# hold R values, not the implicit 0/1 structure VᵀV needs).
function explicit_v_panel(Af::AbstractMatrix{Float64}, k::Int)
    m = size(Af, 1)
    V = zeros(m, k)
    for c in 1:k, i in 1:m
        V[i, c] = i == c ? 1.0 : (i > c ? Af[i, c] : 0.0)
    end
    return V
end

# qr_unblocked! stores tau in the inverted "faer" convention (H = I - vvᵀ/tau); wy_t!/wy_apply!
# take standard LAPACK convention (H = I - tau·vvᵀ) — the one-documented-convention P1 requirement.
lapack_tau(tau_stored) = [isfinite(t) ? 1.0 / t : 0.0 for t in tau_stored]

# Multi-block Qᵀ·C / Q·C sweeps — caller-side looping is wy_apply!'s documented contract
# (single-block kernel; forward order for 'T', reverse for 'N').
function wy_qtc!(C::AbstractMatrix{Float64}, V::AbstractMatrix{Float64}, tau::AbstractVector{Float64}, nb::Int)
    m, k = size(V)
    nblk = cld(k, nb)
    for b in 1:nblk
        pc = (b - 1) * nb + 1
        pb = min(nb, k - pc + 1)
        Vv = view(V, pc:m, pc:(pc + pb - 1))
        Tv = zeros(pb, pb); G = zeros(pb, pb)
        PureBLAS.wy_t!(Tv, Vv, view(tau, pc:(pc + pb - 1)), G)
        Cb = view(C, pc:m, :)
        ws = PureBLAS.WYApplyWorkspace{Float64}(size(Cb, 1), pb, size(Cb, 2))
        PureBLAS.wy_apply!('T', Cb, Vv, Tv, ws)
    end
    return C
end

function wy_qc!(C::AbstractMatrix{Float64}, V::AbstractMatrix{Float64}, tau::AbstractVector{Float64}, nb::Int)
    m, k = size(V)
    nblk = cld(k, nb)
    for b in nblk:-1:1
        pc = (b - 1) * nb + 1
        pb = min(nb, k - pc + 1)
        Vv = view(V, pc:m, pc:(pc + pb - 1))
        Tv = zeros(pb, pb); G = zeros(pb, pb)
        PureBLAS.wy_t!(Tv, Vv, view(tau, pc:(pc + pb - 1)), G)
        Cb = view(C, pc:m, :)
        ws = PureBLAS.WYApplyWorkspace{Float64}(size(Cb, 1), pb, size(Cb, 2))
        PureBLAS.wy_apply!('N', Cb, Vv, Tv, ws)
    end
    return C
end
end

@testitem "wy_t!/wy_apply! — Qᵀ·A reconstructs R, Q·(Qᵀ·A) reconstructs A, single+multi block" setup = [WYHelpers] begin
    using PureBLAS, LinearAlgebra, Random
    Random.seed!(303)
    maxe(A, B) = maximum(abs.(A .- B))
    @testset "$m×$n, nb=$nb" for (m, n) in ((1, 1), (8, 8), (30, 12), (64, 64), (96, 40), (200, 150)),
            nb in (min(m, n), 5, 16)
        A0 = randn(m, n)
        k = min(m, n)
        Af = copy(A0); tau_stored = zeros(k)
        PureBLAS.qr_unblocked!(Af, tau_stored)
        V = explicit_v_panel(Af, k)
        tau = lapack_tau(tau_stored)

        C = copy(A0)
        wy_qtc!(C, V, tau, nb)
        @test maxe(triu(C), triu(Af)) < 1.0e-10          # Qᵀ·A upper part == R

        C2 = copy(C)
        wy_qc!(C2, V, tau, nb)
        @test maxe(C2, A0) < 1.0e-10                      # Q·(Qᵀ·A) == A
    end
end

@testitem "wy_apply!: trans argument validation" begin
    using PureBLAS, LinearAlgebra
    V = Matrix{Float64}(I, 4, 2)
    Tm = zeros(2, 2)
    C = zeros(4, 3)
    ws = PureBLAS.WYApplyWorkspace{Float64}(4, 2, 3)
    @test_throws ArgumentError PureBLAS.wy_apply!('X', C, V, Tm, ws)
end

@testitem "wy_t!/wy_apply!: zero-allocation after warmup" setup = [WYHelpers] begin
    using PureBLAS, Random
    Random.seed!(404)
    m, n = 64, 20
    A0 = randn(m, n)
    Af = copy(A0); tau_stored = zeros(n)
    PureBLAS.qr_unblocked!(Af, tau_stored)
    V = explicit_v_panel(Af, n)
    tau = lapack_tau(tau_stored)
    Tm = zeros(n, n); G = zeros(n, n)
    C = copy(A0)
    ws = PureBLAS.WYApplyWorkspace{Float64}(m, n, n)

    PureBLAS.wy_t!(Tm, V, tau, G)                    # warm up
    PureBLAS.wy_apply!('T', C, V, Tm, ws)
    @test (@allocated PureBLAS.wy_t!(Tm, V, tau, G)) == 0
    @test (@allocated PureBLAS.wy_apply!('T', C, V, Tm, ws)) == 0
    @test (@allocated PureBLAS.wy_apply!('N', C, V, Tm, ws)) == 0
end

@testitem "getrf (LU) vs LAPACK — square/tall/wide, factor + ipiv + P·A=L·U" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.LAPACK
    maxe(x, y) = maximum(abs.(x .- y)) / max(maximum(abs.(y)), 1.0e-300)
    function recon(A0, F, ipiv)                       # P·A ≈ L·U
        m, n = size(F); k = min(m, n)
        L = [i > j ? F[i, j] : (i == j ? 1.0 : 0.0) for i in 1:m, j in 1:k]
        U = [i <= j ? F[i, j] : 0.0 for i in 1:k, j in 1:n]
        PA = copy(A0)
        for i in 1:k
            ip = ipiv[i]
            if ip != i
                tmp = PA[i, :]; PA[i, :] = PA[ip, :]; PA[ip, :] = tmp
            end
        end
        maxe(L * U, PA)
    end
    @testset "$m×$n" for (m, n) in ((1, 1), (8, 8), (40, 25), (64, 64), (129, 96), (200, 200), (96, 160), (600, 513))
        A0 = randn(m, n)
        F = copy(A0); ip = zeros(Int, min(m, n)); PureBLAS.getrf!(F, ip)
        Fl = copy(A0); _, ipl, _ = LAPACK.getrf!(Fl)                  # LAPACK reference
        @test maxe(F, Fl) < 1.0e-11                                    # same factorization
        @test ip == ipl                                             # same pivot sequence
        @test recon(A0, F, ip) < 1.0e-11                              # P·A = L·U
    end
    @test_throws DimensionMismatch PureBLAS.getrf!(randn(5, 5), zeros(Int, 2))
end

@testitem "potrf — ForwardDiff AD through the factor" begin
    using PureBLAS, LinearAlgebra, ForwardDiff
    # L[1,1] of [[x+4, 1],[1, 3]] = sqrt(x+4); d/dx = 1/(2 sqrt(x+4))
    f(x) = (A = [x + 4.0 1.0; 1.0 3.0]; PureBLAS.potrf!(A; uplo = 'L')[1, 1])
    @test ForwardDiff.derivative(f, 1.0) ≈ 0.5 / sqrt(5.0) rtol = 1.0e-10
    # gradient of logdet via Cholesky: logdet(A) = 2 Σ log(L[i,i])
    g(v) = (A = [v[1] + 5.0 1.0; 1.0 v[2] + 5.0]; L = PureBLAS.potrf!(A; uplo = 'L'); 2 * (log(L[1, 1]) + log(L[2, 2])))
    gr = ForwardDiff.gradient(g, [1.0, 2.0])
    @test all(isfinite, gr)
end

@testitem "gesvd (SVD) vs LAPACK — square/tall/wide, σ + reconstruction + orthogonality" begin
    using PureBLAS, LinearAlgebra, Random
    Random.seed!(20)
    maxe(A, B) = maximum(abs.(A .- B))
    @testset "$m×$n" for (m, n) in (
            (1, 1), (2, 2), (8, 8), (40, 25), (64, 64),
            (129, 96), (200, 200), (96, 160), (300, 257),
        )
        A0 = randn(m, n)
        sref = svdvals(A0)
        # full factorization
        U, S, Vt = PureBLAS.gesvd!(copy(A0))
        @test maxe(S, sref) / maximum(sref) < 1.0e-11                  # singular values match LAPACK
        @test maxe(U * Diagonal(S) * Vt, A0) < 1.0e-10                 # A = U Σ Vᵀ
        @test maxe(U' * U, Matrix(I, size(U, 2), size(U, 2))) < 1.0e-10   # U orthonormal columns
        @test maxe(Vt * Vt', Matrix(I, size(Vt, 1), size(Vt, 1))) < 1.0e-10  # Vᵀ orthonormal rows
        # values-only path
        Sv = PureBLAS.gesvd!(copy(A0); want_vectors = false)[1]
        @test maxe(Sv, sref) / maximum(sref) < 1.0e-11
    end
end

@testitem "zgesvd (complex SVD) vs LAPACK — U/S/Vᴴ residual + orthonormality, scaled + rank-deficient" begin
    using PureBLAS, LinearAlgebra, Random
    Random.seed!(202)
    # residual ‖A−U·Σ·Vᴴ‖/(‖A‖·max(m,n)·eps), orthonormality ‖UᴴU−I‖, ‖VᴴV−I‖ (all ≲ O(1)·eps units).
    function chk(A)
        m, n = size(A); mn = min(m, n); T = eltype(A); R = real(T)
        U, S, Vt = PureBLAS.gesvd!(copy(A))
        sref = svdvals(A)
        sc = max(opnorm(A), floatmin(R)) * max(m, n) * eps(R)
        resid = opnorm(U * Diagonal(S) * Vt - A) / sc
        orthU = opnorm(U' * U - I) / (mn * eps(R))
        orthV = opnorm(Vt * Vt' - I) / (mn * eps(R))
        verr = maximum(abs, S .- sref) / max(maximum(sref), floatmin(R))
        return resid, orthU, orthV, verr
    end
    @testset "$T $m×$n" for T in (ComplexF64, ComplexF32),
            (m, n) in (
                (2, 2), (4, 4), (8, 8), (32, 32), (64, 64), (128, 128),
                (8, 4), (4, 8), (64, 40), (40, 64), (129, 96), (96, 129),
            )
        tol = T === ComplexF64 ? 64.0 : 512.0
        resid, orthU, orthV, verr = chk(randn(T, m, n))
        @test resid < tol
        @test orthU < tol
        @test orthV < tol
        @test verr < tol * eps(real(T))
    end
    @testset "$T scaled ‖A‖=$s" for T in (ComplexF64, ComplexF32), s in (1.0e-10, 1.0e-14, 1.0e8)
        resid, orthU, orthV, verr = chk(randn(T, 48, 48) .* T(s))
        @test resid < 128 && orthU < 128 && orthV < 128
    end
    @testset "$T rank-deficient / large-imag" for T in (ComplexF64, ComplexF32)
        A = randn(T, 32, 20); A[:, 12:20] .= 0                       # exact zero singular values
        r = chk(A); @test r[1] < 128 && r[2] < 128 && r[3] < 128
        # Repeated/exact-clustered σ: routed through the D&C path (n>_SVD_DC_CROSS). The simplified
        # forward-only bdsqr! (n≤96) does NOT converge on tightly-clustered σ — a PRE-EXISTING limitation
        # shared by the real Float64 gesvd path (bdsqr! is uarch-shared); D&C (bdsdc!) handles them.
        Q1 = Matrix(qr(randn(T, 128, 128)).Q); Q2 = Matrix(qr(randn(T, 128, 128)).Q)
        Arep = Q1 * Diagonal(T[fill(3.0, 40); fill(1.0, 48); fill(0.05, 40)]) * Q2'   # repeated σ
        r = chk(Arep); @test r[1] < 256 && r[2] < 256 && r[3] < 256
        Abig = randn(T, 32, 32) .+ T(0, 100) .* randn(T, 32, 32)      # large imaginary parts
        r = chk(Abig); @test r[1] < 128 && r[2] < 128 && r[3] < 128
    end
    @testset "values-only path matches" begin
        A = randn(ComplexF64, 50, 30)
        Sv = PureBLAS.gesvd!(copy(A); want_vectors = false)[1]
        @test maximum(abs, Sv .- svdvals(A)) / maximum(svdvals(A)) < 1.0e-11
    end
end

@testitem "syev (symmetric eigen) vs LAPACK — eigenvalues + residual + orthonormality, both uplo + stress" begin
    using PureBLAS, LinearAlgebra, Random
    Random.seed!(21)
    # ‖A·V − V·Diag(w)‖ / (‖A‖·n·eps) and ‖V'V − I‖ / (n·eps): sidesteps eigenvector non-uniqueness.
    function checkpair(Afull, w, Z)
        n = size(Afull, 1); nA = opnorm(Afull); sc = max(nA, 1.0) * n * eps()
        resid = opnorm(Afull * Z - Z * Diagonal(w)) / sc
        orth = opnorm(Z' * Z - I) / (n * eps())
        return resid, orth
    end
    @testset "$(uplo) n=$n" for n in (4, 16, 64, 128), uplo in ('L', 'U')
        A0 = randn(n, n); A0 = A0 + A0'
        S = Symmetric(A0, Symbol(uplo))
        Afull = Matrix(S)
        wref = eigvals(S)
        # jobz='V' — eigenvalues + vectors
        w, Z, info = PureBLAS._syev!('V', uplo, copy(Afull))
        @test info == 0
        @test maximum(abs, w .- wref) / max(1.0, maximum(abs, wref)) < 1.0e-12   # eigenvalues match LAPACK
        @test issorted(w)                                                       # ascending
        resid, orth = checkpair(Afull, w, Z)
        @test resid < 32                                                        # residual ≲ 32·n·eps·‖A‖
        @test orth < 32                                                         # orthonormality ≲ 32·n·eps
        # jobz='N' — eigenvalues only (same values)
        wN, _, iN = PureBLAS._syev!('N', uplo, copy(Afull))
        @test iN == 0
        @test maximum(abs, wN .- wref) / max(1.0, maximum(abs, wref)) < 1.0e-12
    end
    @testset "stress: eps-clustered spectrum" begin
        n = 60
        Q = Matrix(qr(randn(n, n)).Q)                         # random orthogonal
        dvals = [1.0 + (k - 1) * eps() for k in 1:n]          # spacings = eps (deflation-tolerance stress)
        A = Symmetric(Q * Diagonal(dvals) * Q')
        Afull = Matrix(A)
        w, Z, info = PureBLAS._syev!('V', 'L', copy(Afull))
        @test info == 0
        @test maximum(abs, w .- eigvals(A)) < 1.0e-12
        resid, orth = checkpair(Afull, w, Z)
        @test resid < 64 && orth < 64
    end
    @testset "stress: glued Wilkinson W21+" begin
        # Two glued Wilkinson W21+ blocks: many pathologically-close eigenvalue pairs.
        m = 21
        wdiag = Float64[abs(k - (m + 1) ÷ 2) for k in 1:m]    # [10,9,…,1,0,1,…,9,10]
        n = 2m
        d = vcat(wdiag, wdiag)
        A = zeros(n, n)
        for i in 1:n
            A[i, i] = d[i]
        end
        for i in 1:(n - 1)
            i == m && continue                                # weak glue link between the two blocks
            A[i + 1, i] = 1.0; A[i, i + 1] = 1.0
        end
        A[m + 1, m] = 1.0e-3; A[m, m + 1] = 1.0e-3                    # faint coupling
        S = Symmetric(A, :L); Afull = Matrix(S)
        w, Z, info = PureBLAS._syev!('V', 'L', copy(Afull))
        @test info == 0
        @test maximum(abs, w .- eigvals(S)) < 1.0e-10
        resid, orth = checkpair(Afull, w, Z)
        @test resid < 128 && orth < 128
    end
end

@testitem "syev Float32 (native) vs LAPACK — eigenvalues + residual + orthonormality, both uplo" begin
    using PureBLAS, LinearAlgebra, Random
    Random.seed!(31)
    function checkpair(Afull, w, Z, R)
        n = size(Afull, 1); nA = opnorm(Afull); sc = max(nA, one(R)) * n * eps(R)
        resid = opnorm(Afull * Z - Z * Diagonal(w)) / sc
        orth = opnorm(Z' * Z - I) / (n * eps(R))
        return resid, orth
    end
    @testset "$(uplo) n=$n" for n in (4, 17, 64, 128), uplo in ('L', 'U')
        A0 = randn(Float32, n, n); A0 = A0 + A0'
        S = Symmetric(A0, Symbol(uplo)); Afull = Matrix(S)
        wref = eigvals(Symmetric(Float64.(Afull)))                     # F64 LAPACK reference
        w, Z, info = PureBLAS._syev!('V', uplo, copy(Afull))
        @test info == 0
        @test eltype(w) === Float32 && eltype(Z) === Float32
        @test issorted(w)
        @test maximum(abs, Float64.(w) .- wref) / max(1.0, maximum(abs, wref)) < 1.0e-4
        resid, orth = checkpair(Afull, w, Z, Float32)
        @test resid < 64 && orth < 64
        wN, _, iN = PureBLAS._syev!('N', uplo, copy(Afull))            # values-only (sterf)
        @test iN == 0
        @test maximum(abs, Float64.(wN) .- wref) / max(1.0, maximum(abs, wref)) < 1.0e-4
    end
end

@testitem "heev (Hermitian eigen) vs LAPACK — ComplexF64/F32, residual + orthonormality, both uplo" begin
    using PureBLAS, LinearAlgebra, Random
    Random.seed!(32)
    # _heev! is generic: on ComplexF32 it computes natively (Float32 eigenvalues, ComplexF32 vectors) —
    # the mixed-precision promotion is only in the C-ABI cheev shim, not the engine. Check at native eps(R).
    function checkpair(Afull, w, Z, R)
        n = size(Afull, 1); nA = opnorm(Afull); sc = max(nA, one(R)) * n * eps(R)
        resid = opnorm(Afull * Z - Z * Diagonal(complex(w))) / sc
        orth = opnorm(Z' * Z - I) / (n * eps(R))
        return resid, orth
    end
    @testset "$T $(uplo) n=$n" for T in (ComplexF64, ComplexF32),
            n in (2, 4, 16, 64, 128), uplo in ('L', 'U')

        B0 = randn(T, n, n); A0 = B0 + B0'
        S = Hermitian(A0, Symbol(uplo)); Afull = Matrix(S)
        R = real(T)
        wref = eigvals(Hermitian(ComplexF64.(Afull)))                 # F64 LAPACK reference
        w, Z, info = PureBLAS._heev!('V', uplo, copy(Afull))
        @test info == 0
        @test eltype(w) === R && eltype(Z) === T                       # native element type
        @test issorted(w)
        vtol = R === Float64 ? 1.0e-11 : 1.0e-4
        @test maximum(abs, Float64.(w) .- wref) / max(1.0, maximum(abs, wref)) < vtol
        resid, orth = checkpair(Afull, w, Z, R)
        @test resid < 64 && orth < 64
        wN, _, iN = PureBLAS._heev!('N', uplo, copy(Afull))           # values-only (sterf)
        @test iN == 0
        @test maximum(abs, Float64.(wN) .- wref) / max(1.0, maximum(abs, wref)) < vtol
    end
end

@testitem "sterf (tridiagonal eigenvalues, values-only) vs LAPACK" begin
    using PureBLAS, LinearAlgebra, Random
    Random.seed!(33)
    @testset "$T n=$n" for T in (Float64, Float32), n in (1, 2, 10, 31, 128, 257)
        A = randn(T, n, n); A = A + A'
        d = Vector{T}(undef, n); e = Vector{T}(undef, max(n - 1, 1)); tau = Vector{T}(undef, max(n - 1, 1))
        PureBLAS._sytd2_lower!(copy(Symmetric(A, :L) |> Matrix), d, e, tau)   # tridiagonalize
        info = PureBLAS._sterf!(d, e)
        @test info == 0
        @test issorted(d)
        wref = eigvals(Symmetric(Float64.(A)))
        tol = T === Float64 ? 1.0e-11 : 1.0e-4
        @test maximum(abs, Float64.(d) .- wref) / max(1.0, maximum(abs, wref)) < tol
    end
end

@testitem "lq (LQ) vs LAPACK — gelqf L·Q reconstruct, orglq orthonormal rows, ormlq apply" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.LAPACK
    maxe(x, y) = maximum(abs.(x .- y)) / max(maximum(abs.(y)), 1.0e-300)
    @testset "$T $m×$n" for T in (Float32, Float64, ComplexF32, ComplexF64),
            (m, n) in ((1, 1), (8, 8), (25, 40), (64, 64), (96, 129), (60, 60))

        A0 = randn(T, m, n)
        F = copy(A0); tau = zeros(T, min(m, n)); PureBLAS.gelqf!(F, tau)
        # LAPACK reference: gelqf produces the SAME τ convention → compare L directly (lower trapezoid).
        Fl = copy(A0); tl = LAPACK.gelqf!(Fl)[2]
        k = min(m, n)
        @test maxe(abs.(tril(F)[:, 1:k]), abs.(tril(Fl)[:, 1:k])) < 100 * eps(real(T))
        # Reconstruct A = L·Q by forming Q (orglq) on the min(m,n)×n reflector rows.
        mq = k
        Qrows = copy(F)[1:mq, :]                        # first mq rows hold reflectors + tau[1:mq]
        Q = PureBLAS.orglq!(Qrows, tau[1:mq])
        @test maxe(Q * Q', Matrix{T}(I, mq, mq)) < 200 * eps(real(T))   # orthonormal rows
        L = tril(F)[:, 1:k]
        @test maxe(L * Q, A0) < 200 * eps(real(T))                      # L·Q = A
        # ormlq: apply Qᴴ then Q to a matrix, round-trips. side='L' applies the ORDER-n Q (of which orglq's
        # thin mq×n is the leading rows), so C is n×7 and Q·Qᴴ·C = C (full n×n orthogonal round-trip).
        C = randn(T, n, 7)
        trN = 'N'; trC = T <: Complex ? 'C' : 'T'
        C1 = PureBLAS.ormlq!('L', trC, copy(F)[1:mq, :], tau[1:mq], copy(C))
        C2 = PureBLAS.ormlq!('L', trN, copy(F)[1:mq, :], tau[1:mq], copy(C1))
        @test maxe(C2, C) < 200 * eps(real(T))
    end
end

# Bunch-Kaufman is a PIVOTED factorization, so the test has to constrain the PIVOTS, and until this
# item was rewritten nothing did. The old version factored with PureBLAS and solved with PureBLAS,
# then checked ‖A·X−B‖ — a closed loop that ANY self-consistent pivot sequence passes, including a
# numerically terrible one. It also used only well-conditioned randoms, so the 2×2-pivot branches of
# both `_sytf2_*` and `_sytrs_*` were barely reached. Three additions fix that, and they are what a
# blocked `dlasyf` port has to be built against:
#
#   1. CROSS-SOLVE, both directions. PureBLAS factor → `LAPACK.sytrs!`, and `LAPACK.sytrf!` →
#      PureBLAS solve. This IS a P·L·D·Lᵀ·Pᵀ reconstruction, performed by an independent
#      implementation, so it pins the `ipiv` encoding in BOTH the writing and the reading direction
#      without a hand-written extractor that would just reimplement the convention under test.
#   2. MULTIPLIER + BACKWARD-ERROR BOUNDS, both calibrated against LAPACK on the same input.
#      Cross-solve alone still admits a *valid but poor* pivot — one that stays self-consistent but
#      divides by a near-zero and blows up ‖L‖. NOTE the bound here is comparative, not the textbook
#      |l_ij| ≤ 1/(1−α) ≈ 2.781: that is the Bunch-PARLETT (complete-pivoting) bound. Bunch-KAUFMAN
#      *partial* pivoting bounds the growth factor but leaves ‖L‖ genuinely unbounded — which is the
#      entire reason `sytrf_rook` exists. Asserting 2.781 fails on ordinary random input (measured
#      2.87–4.94 against the known-good unblocked kernel). What IS true, measured over every cell
#      below: PureBLAS's max|L| equals LAPACK's to a ratio of 1.00, and the solve backward error is
#      0.05·n·eps. The asserted slack (4× and 100·n·eps) is ~1000× looser than observed, so it can't
#      flake, while a mis-selected pivot moves both by orders of magnitude. The D entries of a 2×2
#      block are not multipliers, so they're skipped via `ipiv`.
#   3. HARD MATRIX CLASSES. `hollow` (zero diagonal) forces EVERY pivot to 2×2 with a search that
#      reaches an arbitrary row — the decisive class for a blocked kernel's panel boundary. `rankdef`
#      reaches the `max(absakk,colmax)==0 ⇒ info=k` branch, which no test touched before.
#
# Bit-exact `ipiv` vs LAPACK is deliberately NOT asserted: a blocked path re-associates the column
# update (one length-(k−1) gemv reduction vs k−1 successive axpys), so near-ties can rank
# differently, and PureBLAS's nb differs from OpenBLAS's ILAENV value regardless. It would be a flaky
# test on the one routine whose entire point is pivot selection.
@testitem "bunchkaufman (sytrf/hetrf) — cross-solve vs LAPACK, growth bound, hollow + rank-deficient" begin
    using PureBLAS, LinearAlgebra, Random
    import LinearAlgebra.LAPACK
    # Backward-error tolerance, NOT the usual sqrt(eps)*100. Measured worst over every cell here is
    # 0.05·n·eps, so 100·n·eps keeps ~2000× headroom while being ~5 orders of magnitude tighter than
    # sqrt(eps)*100 — tight enough that a factorization with real growth cannot slip through.
    tol(::Type{T}, n) where {T} = 100 * max(n, 4) * eps(real(T))

    # Largest |L| entry, skipping the D entries of every 2×2 block (read off ipiv, both uplo).
    function maxmult(LD, ipiv, uplo, n)
        m = 0.0
        if uplo == 'L'
            k = 1
            while k <= n
                if ipiv[k] > 0                          # 1×1: column k below the diagonal is all L
                    for i in (k + 1):n; m = max(m, abs(LD[i, k])); end
                    k += 1
                else                                    # 2×2 at (k,k+1): LD[k+1,k] is D, not L
                    for i in (k + 2):n; m = max(m, abs(LD[i, k]), abs(LD[i, k + 1])); end
                    k += 2
                end
            end
        else
            k = n
            while k >= 1
                if ipiv[k] > 0
                    for i in 1:(k - 1); m = max(m, abs(LD[i, k])); end
                    k -= 1
                else                                    # 2×2 at (k-1,k): LD[k-1,k] is D, not L
                    for i in 1:(k - 2); m = max(m, abs(LD[i, k - 1]), abs(LD[i, k])); end
                    k -= 2
                end
            end
        end
        return m
    end

    # Every negative ipiv entry must come as an adjacent equal pair, oriented by uplo.
    function ipiv_wellformed(ipiv, uplo, n)
        k = uplo == 'L' ? 1 : n
        while uplo == 'L' ? k <= n : k >= 1
            p = ipiv[k]
            (1 <= abs(p) <= n) || return false
            if p > 0
                k += uplo == 'L' ? 1 : -1
            else
                j = uplo == 'L' ? k + 1 : k - 1
                (1 <= j <= n && ipiv[j] == p) || return false
                k += uplo == 'L' ? 2 : -2
            end
        end
        return true
    end

    @testset "$T n=$n uplo=$uplo herm=$herm cls=$cls" for T in (Float32, Float64, ComplexF32, ComplexF64),
            n in (1, 2, 5, 17, 49, 64, 100, 129, 200), uplo in ('L', 'U'),
            herm in (false, true), cls in (:generic, :hollow)

        (herm && !(T <: Complex)) && continue           # herm==sym for real; skip dup
        Random.seed!(hash((T, n, uplo, herm, cls)))
        M = randn(T, n, n)
        A = herm ? (M + M') : (M + transpose(M))        # Hermitian / symmetric indefinite
        # hollow: a zero diagonal makes absakk==0 at every step, so the 1×1 tests can never fire and
        # EVERY pivot is 2×2 with imax an arbitrary row — the panel-boundary stress case.
        cls === :hollow && (A[diagind(A)] .= zero(T))

        ipiv = zeros(Int, n); LD = copy(A)
        info = herm ? PureBLAS.hetrf!(LD, ipiv; uplo = uplo) :
                      PureBLAS.sytrf!(LD, ipiv; uplo = uplo)
        LDr = copy(A); ipr = zeros(Int, n)               # LAPACK's factor of the SAME matrix
        herm ? LAPACK.hetrf!(uplo, LDr, ipr) : LAPACK.sytrf!(uplo, LDr, ipr)

        @test ipiv_wellformed(ipiv, uplo, n)
        # Multiplier bound, calibrated against LAPACK's own pivot sequence on this exact input
        # (measured ratio 1.00 everywhere; 4× is slack, not a fitted threshold).
        @test maxmult(LD, ipiv, uplo, n) <=
              4 * maxmult(LDr, ipr, uplo, n) + sqrt(eps(real(T)))
        # zhetrf's D must stay real; a missing real() site shows up here exactly, no tolerance.
        herm && @test all(isreal, diag(LD))

        if info == 0
            B = randn(T, n, 3)
            # (1) PureBLAS factor → LAPACK solve. Independent reconstruction of P·L·D·Lᵀ·Pᵀ.
            Xl = copy(B)
            herm ? LAPACK.hetrs!(uplo, LD, ipiv, Xl) : LAPACK.sytrs!(uplo, LD, ipiv, Xl)
            @test norm(A * Xl - B) <= tol(T, n) * (norm(A) * norm(Xl) + norm(B))
            # (2) LAPACK factor → PureBLAS solve. Pins the READING side of the ipiv convention.
            Xr = copy(B)
            herm ? PureBLAS.hetrs!(LDr, ipr, Xr; uplo = uplo) :
                   PureBLAS.sytrs!(LDr, ipr, Xr; uplo = uplo)
            @test norm(A * Xr - B) <= tol(T, n) * (norm(A) * norm(Xr) + norm(B))
        end
    end

    # BLOCKED panel, called BY NAME with an explicit nb. Going through `sytrf!` would tie coverage to
    # the host's measured nb and to n crossing it; this sweeps `nb in 2:12`, which hits EVERY panel-
    # boundary parity — so the k=nb-1/kstep=2 case (the one that touches W's last column) and the
    # "2×2 deferred to the next panel" case both occur many times with no per-cell reasoning. The
    # hollow class matters most here: a zero diagonal forces every pivot to 2×2 with a search that
    # reaches an arbitrary row, which is exactly what a panel boundary can get wrong.
    @testset "blocked $T n=$n nb=$nb cls=$cls uplo=$uplo herm=$hm" for
            T in (Float64, Float32, ComplexF64, ComplexF32),
            n in (7, 13, 40, 100, 129), nb in 2:12, cls in (:generic, :hollow, :rankdef),
            uplo in ('L', 'U'), hm in (false, true)

        nb >= n && continue
        (hm && !(T <: Complex)) && continue      # herm == sym for real; skip the duplicate
        Random.seed!(hash((T, n, nb, cls, hm)))
        M = randn(T, n, n); A = hm ? (M + M') : (M + transpose(M))
        cls === :hollow && (A[diagind(A)] .= zero(T))
        if cls === :rankdef
            B0 = randn(T, n, max(1, n - 3))
            A = hm ? (B0 * B0') : (B0 * transpose(B0))
        end

        ip = zeros(Int, n); LD = copy(A)
        info = uplo == 'L' ? PureBLAS._sytrf_blocked_lower!(LD, ip, nb, hm) :
                             PureBLAS._sytrf_blocked_upper!(LD, ip, nb, hm)
        @test ipiv_wellformed(ip, uplo, n)
        @test all(isfinite, LD)
        # zhetrf's D must be EXACTLY real. Measured 0.0 against LAPACK on every cell, so no
        # tolerance — this is the assertion that catches a missing real() site, and it is the same
        # class of bug the unblocked kernel was carrying (see the header of this testitem).
        hm && @test all(isreal, diag(LD))
        if info == 0
            Bv = randn(T, n, 3); X = copy(Bv)
            # independent P·L·D·Lᴴ·Pᵀ reconstruction, performed by LAPACK
            hm ? LAPACK.hetrs!(uplo, LD, ip, X) : LAPACK.sytrs!(uplo, LD, ip, X)
            @test norm(A * X - Bv) <= tol(T, n) * (norm(A) * norm(X) + norm(Bv))
        end
    end

    # Rank-deficient / exactly-singular: reaches the `max(absakk,colmax)==0 ⇒ info=k` branch, which
    # nothing else in the suite touches (every other item adds n*I to stay well away from it).
    @testset "singular $T n=$n uplo=$uplo" for T in (Float32, Float64, ComplexF64),
            n in (8, 60), uplo in ('L', 'U')

        Random.seed!(hash((T, n, uplo, :sing)))
        B = randn(T, n, n - 3); A = B * transpose(B)     # rank n-3, symmetric
        LD = copy(A); ipiv = zeros(Int, n)
        info = PureBLAS.sytrf!(LD, ipiv; uplo = uplo)
        @test all(isfinite, LD)                          # must not NaN/Inf out
        @test ipiv_wellformed(ipiv, uplo, n)

        c = n ÷ 2                                        # exactly-zero row/column ⇒ info names it
        A2 = randn(T, n, n); A2 = A2 + transpose(A2)
        A2[:, c] .= zero(T); A2[c, :] .= zero(T)
        LD2 = copy(A2); ip2 = zeros(Int, n)
        info2 = PureBLAS.sytrf!(LD2, ip2; uplo = uplo)
        @test info2 > 0
        @test all(isfinite, LD2)
    end
end

@testitem "geqp3 (pivoted QR) vs LAPACK — A·P = Q·R reconstruct, |R| non-increasing" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.LAPACK
    maxe(x, y) = maximum(abs.(x .- y)) / max(maximum(abs.(y)), 1.0e-300)
    @testset "$T $m×$n" for T in (Float32, Float64, ComplexF32, ComplexF64),
            (m, n) in ((6, 6), (40, 25), (25, 40), (64, 64))

        A0 = randn(T, m, n)
        F = copy(A0); jpvt = zeros(Int, n); tau = zeros(T, min(m, n))
        PureBLAS.geqp3!(F, jpvt, tau)
        k = min(m, n)
        # Rebuild Q from the reflectors (standard τ: H_i = I − τ v vᴴ) and check A[:,jpvt] = Q·R.
        R = [i <= j ? F[i, j] : zero(T) for i in 1:m, j in 1:n]
        Q = Matrix{T}(I, m, m)
        for i in 1:k                                    # Q = H_1·H_2···H_k (right-multiply from I, in order)
            v = zeros(T, m); v[i] = one(T); v[(i + 1):m] = F[(i + 1):m, i]
            Q = Q - (tau[i] * Q * v) * v'
        end
        @test maxe(Q * R, A0[:, jpvt]) < 300 * eps(real(T))
        rdiag = [abs(R[i, i]) for i in 1:k]
        @test issorted(rdiag; rev = true) || maximum(diff(rdiag)) < 1.0e-6 * rdiag[1]  # non-increasing
    end
end

@testitem "gels (least-squares / min-norm) vs LAPACK — overdetermined + underdetermined residual" begin
    using PureBLAS, LinearAlgebra
    tol(::Type{T}) where {T} = sqrt(eps(real(T))) * 100
    @testset "$T $m×$n trans=$trans" for T in (Float32, Float64, ComplexF32, ComplexF64),
            (m, n) in ((40, 20), (20, 40), (30, 30)), trans in ('N',)

        A = randn(T, m, n)
        p, q = size(A)                                  # op = A (trans N): p×q
        nrhs = 2
        B = zeros(T, max(p, q), nrhs); b0 = randn(T, p, nrhs); B[1:p, :] = b0
        PureBLAS.gels!(trans, copy(A), B)
        X = B[1:q, :]
        if p >= q                                       # overdetermined → normal equations Aᴴ(Ax−b)=0
            @test norm(A' * (A * X - b0)) <= tol(T) * (norm(A)^2 * norm(X) + norm(A) * norm(b0))
        else                                            # underdetermined → A·x = b exactly, min-norm
            @test norm(A * X - b0) <= tol(T) * (norm(A) * norm(X) + norm(b0))
        end
    end
end

@testitem "gecon/trcon/pocon (condition estimate) vs LAPACK" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.LAPACK
    @testset "$T n=$n" for T in (Float32, Float64, ComplexF32, ComplexF64), n in (5, 20, 50)
        A = randn(T, n, n) + n * I
        # gecon: estimate from LU vs LAPACK.gecon! on the same factors.
        for nrm in ('1', 'I')
            an = opnorm(A, nrm == '1' ? 1 : Inf)
            LU = lu(copy(A))
            LUf = Matrix(LU.factors)
            rc = PureBLAS.gecon!(an, LUf, LU.ipiv; norm = nrm)
            rc_ref = LAPACK.gecon!(nrm, copy(LUf), real(T)(an))   # Julia's LAPACK wrapper wants '1'/'I' (not 'O')
            @test isapprox(rc, rc_ref; rtol = 1.0e-3)
        end
        # trcon: triangular condition vs LAPACK.trcon!.
        U = triu(A)
        rct = PureBLAS.trcon!(copy(U); uplo = 'U', diag = 'N', norm = '1')
        rct_ref = LAPACK.trcon!('1', 'U', 'N', copy(U))   # Julia's LAPACK wrapper wants '1'/'I' (not 'O')
        @test isapprox(rct, rct_ref; rtol = 1.0e-3)
        # pocon: SPD Cholesky condition; compare to reciprocal true condition (loose).
        SPD = A * A' + n * I
        Cf = PureBLAS.potrf!(copy(SPD); uplo = 'L')
        an = opnorm(SPD, 1)
        rcp = PureBLAS.pocon!(an, Cf; uplo = 'L')
        @test 0 < rcp <= 1
        @test isapprox(rcp, 1 / cond(SPD, 1); rtol = 0.5)   # estimator within ~2× of true rcond
    end
end

@testitem "gehrd (Hessenberg reduction) vs LAPACK — H = Qᴴ·A·Q reconstruct, orghr Q unitary" begin
    using PureBLAS, LinearAlgebra
    maxe(x, y) = maximum(abs.(x .- y)) / max(maximum(abs.(y)), 1.0e-300)
    @testset "$T n=$n" for T in (Float32, Float64, ComplexF32, ComplexF64), n in (1, 2, 5, 20, 64)
        A0 = randn(T, n, n)
        H = copy(A0); tau = zeros(T, max(n - 1, 0)); PureBLAS.gehrd!(H, 1, n, tau)
        Q = PureBLAS.orghr!(copy(H), 1, n, tau)
        @test maxe(Q' * Q, Matrix{T}(I, n, n)) < 500 * eps(real(T))         # Q unitary
        Hu = triu(H, -1)                                                     # upper Hessenberg part
        @test maxe(Q * Hu * Q', A0) < 500 * eps(real(T))                    # Q·H·Qᴴ = A
    end
end

@testitem "geev (general eigen) vs LAPACK — eigenvalues + A·V=V·Λ residual, real+complex, conj-pairs" begin
    using PureBLAS, LinearAlgebra, Random
    Random.seed!(0xEE)
    evmatch(a, b) = (
        a = sort(collect(a); by = x -> (real(x), imag(x)));
        b = sort(collect(b); by = x -> (real(x), imag(x)));
        maximum(abs.(a .- b)) / max(maximum(abs.(b)), 1)
    )
    # reconstruct the complex eigenvector matrix from real-packed VR (LAPACK conj-pair convention)
    function evecs_real(wr, wi, VR)
        n = length(wr); E = zeros(ComplexF64, n, n); j = 1
        while j <= n
            if wi[j] == 0
                E[:, j] = VR[:, j]
            else
                for i in 1:n
                    E[i, j] = VR[i, j] + im * VR[i, j + 1]; E[i, j + 1] = VR[i, j] - im * VR[i, j + 1]
                end
                j += 1
            end
            j += 1
        end
        E
    end
    @testset "real n=$n" for n in (2, 4, 8, 16, 32, 64)
        A = randn(n, n)
        wr, wi, VL, VR = PureBLAS.geev!('N', 'V', copy(A))
        λ = complex.(wr, wi); E = evecs_real(wr, wi, VR)
        @test maximum(abs.(A * E - E * Diagonal(λ))) / (opnorm(A, 1) * n * eps()) < 100
        @test evmatch(λ, eigvals(copy(A))) < 1.0e-8            # vs LinearAlgebra (OpenBLAS)
        wr2, wi2 = PureBLAS.geev!('N', 'N', copy(A))         # values-only path
        @test evmatch(complex.(wr2, wi2), λ) < 1.0e-10
    end
    @testset "conj-pair real n=$n" for n in (4, 8, 16)       # block rotations → guaranteed complex pairs
        B = zeros(n, n); i = 1
        while i + 1 <= n
            θ = randn(); a = randn()
            B[i, i] = a; B[i + 1, i + 1] = a; B[i, i + 1] = θ; B[i + 1, i] = -θ; i += 2
        end
        Q, _ = qr(randn(n, n)); A = Matrix(Q) * B * Matrix(Q)'
        wr, wi, VL, VR = PureBLAS.geev!('N', 'V', copy(A))
        @test count(!iszero, wi) > 0                          # actually has conjugate pairs
        E = evecs_real(wr, wi, VR)
        @test maximum(abs.(A * E - E * Diagonal(complex.(wr, wi)))) / (opnorm(A, 1) * n * eps()) < 100
        @test evmatch(complex.(wr, wi), eigvals(copy(A))) < 1.0e-8
    end
    @testset "complex n=$n" for n in (2, 4, 8, 16, 32)
        A = randn(ComplexF64, n, n)
        w, VL, VR = PureBLAS.geev!('N', 'V', copy(A))
        @test maximum(abs.(A * VR - VR * Diagonal(w))) / (opnorm(A, 1) * n * eps()) < 100
        @test evmatch(w, eigvals(copy(A))) < 1.0e-8
        w2, = PureBLAS.geev!('N', 'N', copy(A))
        @test evmatch(w2, w) < 1.0e-10
    end
    @test_throws ArgumentError PureBLAS.geev!('V', 'N', randn(4, 4))   # left vectors not implemented
end

@testitem "gees (Schur) vs LAPACK — Z·T·Zᴴ=A reconstruction, Z orthonormal, real+complex" begin
    using PureBLAS, LinearAlgebra, Random
    Random.seed!(0x5C)
    evmatch(a, b) = (
        a = sort(collect(a); by = x -> (real(x), imag(x)));
        b = sort(collect(b); by = x -> (real(x), imag(x)));
        maximum(abs.(a .- b)) / max(maximum(abs.(b)), 1)
    )
    @testset "real n=$n" for n in (2, 4, 8, 16, 32)
        A = randn(n, n)
        T, Z, w = PureBLAS.gees!('V', copy(A))
        @test maximum(abs.(Z * T * Z' - A)) / max(opnorm(A, 1), 1) < 1.0e-11    # A = Z·T·Zᵀ
        @test maximum(abs.(Z' * Z - I)) < 1.0e-12                                # Z orthonormal
        @test evmatch(w, eigvals(copy(A))) < 1.0e-8
        @test all(iszero, [T[i, j] for j in 1:n for i in (j + 2):n])              # quasi-upper-triangular
    end
    @testset "conj-pair real n=$n" for n in (4, 8, 16)
        B = zeros(n, n); i = 1
        while i + 1 <= n
            θ = randn(); a = randn(); B[i, i] = a; B[i + 1, i + 1] = a; B[i, i + 1] = θ; B[i + 1, i] = -θ; i += 2
        end
        Q, _ = qr(randn(n, n)); A = Matrix(Q) * B * Matrix(Q)'
        T, Z, w = PureBLAS.gees!('V', copy(A))
        @test maximum(abs.(Z * T * Z' - A)) / max(opnorm(A, 1), 1) < 1.0e-11
        @test maximum(abs.(Z' * Z - I)) < 1.0e-12
    end
    @testset "complex n=$n" for n in (2, 4, 8, 16, 32)
        A = randn(ComplexF64, n, n)
        T, Z, w = PureBLAS.gees!('V', copy(A))
        @test maximum(abs.(Z * T * Z' - A)) / max(opnorm(A, 1), 1) < 1.0e-11
        @test maximum(abs.(Z' * Z - I)) < 1.0e-12
        @test evmatch(w, eigvals(copy(A))) < 1.0e-8
        @test all(iszero, [T[i, j] for j in 1:n for i in (j + 1):n])              # upper-triangular (complex)
    end
end

@testitem "ggev (generalized eigen) vs LAPACK — (βA−αB)x residual, eigval match, conj-pair + infinite-eig" begin
    using PureBLAS, LinearAlgebra, Random
    import LinearAlgebra.LAPACK as LA
    Random.seed!(0x6617)
    # chordal metric on (α,β) — robust to infinite/huge eigenvalues (α/β on the Riemann sphere)
    chord(a, b, c, d) = abs(a * d - c * b) / (sqrt(abs2(a) + abs2(b)) * sqrt(abs2(c) + abs2(d)))
    function evmatch(a1, b1, a2, b2)
        n = length(a1); used = falses(n); w = 0.0
        for i in 1:n
            best = Inf; bj = 0
            for j in 1:n
                used[j] && continue
                d = chord(a1[i], b1[i], a2[j], b2[j]); d < best && (best = d; bj = j)
            end
            used[bj] = true; w = max(w, best)
        end
        w
    end
    packR(vr, ai) = (
        n = size(vr, 1); VC = zeros(ComplexF64, n, n); j = 1;
        while j <= n
            if iszero(ai[j])
                VC[:, j] = vr[:, j]; j += 1
            else
                VC[:, j] = vr[:, j] .+ im .* vr[:, j + 1]; VC[:, j + 1] = vr[:, j] .- im .* vr[:, j + 1]; j += 2
            end
        end; VC
    )
    function resid(A, B, al, be, VC)
        n = size(A, 1); w = 0.0
        for j in 1:n
            x = VC[:, j]
            d = norm(x) * (abs(be[j]) * opnorm(A, 1) + abs(al[j]) * opnorm(B, 1))
            d < eps() && continue
            w = max(w, norm(be[j] * (A * x) - al[j] * (B * x)) / d)
        end
        w
    end

    @testset "real random n=$n" for n in (6, 20, 40)
        A = randn(n, n); B = randn(n, n)
        ar, ai, be, vl, vr = PureBLAS.ggev!('N', 'V', copy(A), copy(B))
        @test size(vl, 2) == 0
        VC = packR(vr, ai)
        @test resid(A, B, complex.(ar, ai), be, VC) < 1.0e-11
        ar2, ai2, be2, _, _ = LA.ggev3!('N', 'V', copy(A), copy(B))
        @test evmatch(complex.(ar, ai), complex.(be), complex.(ar2, ai2), complex.(be2)) < 1.0e-10
    end
    @testset "real guaranteed conj-pairs n=$n" for n in (8, 20)
        Bp = zeros(n, n); i = 1
        while i + 1 <= n
            Bp[i, i] = randn(); Bp[i + 1, i + 1] = Bp[i, i]; Bp[i, i + 1] = randn(); Bp[i + 1, i] = -Bp[i, i + 1]; i += 2
        end
        Qr, _ = qr(randn(n, n)); A = Matrix(Qr) * Bp * Matrix(Qr)'
        M = randn(n, n); B = M'M + I                        # SPD, well conditioned
        ar, ai, be, _, vr = PureBLAS.ggev!('N', 'V', copy(A), copy(B))
        @test count(!iszero, ai) > 0                        # actually produced conjugate pairs
        @test resid(A, B, complex.(ar, ai), be, packR(vr, ai)) < 1.0e-11
    end
    @testset "real infinite eigenvalue (singular B)" begin
        n = 12
        A = randn(n, n); B = randn(n, n); B[:, n] .= 0; B[n, :] .= 0    # rank-deficient B → infinite eig
        ar, ai, be, _, vr = PureBLAS.ggev!('N', 'V', copy(A), copy(B))
        @test count(x -> abs(x) < 1.0e-10, be) >= 1           # at least one infinite eigenvalue (β≈0)
        @test resid(A, B, complex.(ar, ai), be, packR(vr, ai)) < 1.0e-9
    end
    @testset "complex n=$n" for n in (6, 20, 40)
        A = randn(ComplexF64, n, n); B = randn(ComplexF64, n, n)
        al, be, vl, vr = PureBLAS.ggev!('N', 'V', copy(A), copy(B))
        @test size(vl, 2) == 0
        @test resid(A, B, al, be, vr) < 1.0e-11
        al2, be2, _, _ = LA.ggev3!('N', 'V', copy(A), copy(B))
        @test evmatch(al, be, al2, be2) < 1.0e-10
    end
    @testset "eigenvalues-only path (jobvr='N') n=$n" for n in (10, 24)
        A = randn(n, n); B = randn(n, n)
        ar, ai, be, _, vr = PureBLAS.ggev!('N', 'N', copy(A), copy(B))
        @test size(vr, 2) == 0
        ar2, ai2, be2, _, _ = LA.ggev3!('N', 'N', copy(A), copy(B))
        @test evmatch(complex.(ar, ai), complex.(be), complex.(ar2, ai2), complex.(be2)) < 1.0e-10
    end
    @test_throws ArgumentError PureBLAS.ggev!('V', 'V', randn(4, 4), randn(4, 4))   # left vectors unsupported
end

@testitem "gges (generalized Schur) vs LAPACK — A=Q·S·Zᴴ, B=Q·P·Zᴴ, Q/Z orthonormal, real+complex" begin
    using PureBLAS, LinearAlgebra, Random
    Random.seed!(0x9a55)
    @testset "$T n=$n" for T in (Float64, ComplexF64), n in (5, 16, 33)
        A = randn(T, n, n); B = randn(T, n, n)
        S, P, al, be, Q, Z = PureBLAS.gges!('V', 'V', copy(A), copy(B))
        tol = 1.0e-11 * (opnorm(A, 1) + opnorm(B, 1) + 1)
        @test maximum(abs, Q * S * Z' - A) < tol
        @test maximum(abs, Q * P * Z' - B) < tol
        @test maximum(abs, Q' * Q - I) < 1.0e-12
        @test maximum(abs, Z' * Z - I) < 1.0e-12
        @test length(al) == n && length(be) == n
    end
    @testset "singular B (infinite eig) n=12" begin
        n = 12; A = randn(n, n); B = randn(n, n); B[:, n] .= 0; B[n, :] .= 0
        S, P, al, be, Q, Z = PureBLAS.gges!('V', 'V', copy(A), copy(B))
        @test maximum(abs, Q * S * Z' - A) < 1.0e-10 * (opnorm(A, 1) + 1)
        @test maximum(abs, Q * P * Z' - B) < 1.0e-12
    end
end

@testitem "sygvd/hegvd (generalized sym/Herm-definite eigen) vs LAPACK — itype 1/2/3, both uplo" begin
    using PureBLAS, LinearAlgebra, Random
    import LinearAlgebra.LAPACK as LA
    Random.seed!(0x5aa5)
    @testset "$T itype=$it uplo=$ul n=$n" for T in (Float64, ComplexF64), it in (1, 2, 3),
            ul in ('L', 'U'), n in (4, 13, 40)

        M = randn(T, n, n); A = M + M'                     # Hermitian
        N = randn(T, n, n); B = N * N' + n * I             # Hermitian positive definite
        gvd! = T <: Complex ? PureBLAS.hegvd! : PureBLAS.sygvd!
        w, Z = gvd!(it, 'V', ul, copy(A), copy(B))
        wref = LA.sygvd!(it, 'V', ul, copy(A), copy(B))[1]
        @test maximum(abs, w .- wref) < 1.0e-9 * (norm(w) + 1)
        # verify the defining relation per itype
        if it == 1                                         # A z = λ B z
            R = A * Z - B * Z * Diagonal(w)
        elseif it == 2                                     # A B z = λ z
            R = A * (B * Z) - Z * Diagonal(w)
        else                                               # itype 3: B A z = λ z
            R = B * (A * Z) - Z * Diagonal(w)
        end
        @test maximum(abs, R) < 1.0e-8 * (opnorm(A, 1) * opnorm(B, 1) + 1)
        wN = gvd!(it, 'N', ul, copy(A), copy(B))[1]        # values-only path
        @test maximum(abs, wN .- wref) < 1.0e-9 * (norm(w) + 1)
    end
    @test_throws PosDefException PureBLAS.sygvd!(1, 'V', 'L', [2.0 0; 0 2], [1.0 0; 0 -1.0])  # B not PD
end

@testitem "gtsv/gttrf/gttrs (tridiagonal solve) vs LAPACK — all four types, multi-RHS, trans" begin
    using PureBLAS, LinearAlgebra, Random
    import LinearAlgebra.LAPACK as LA
    Random.seed!(0x7331)
    @testset "$T n=$n" for T in (Float32, Float64, ComplexF32, ComplexF64), n in (1, 2, 7, 40)
        dl = randn(T, n - 1); d = randn(T, n) .+ T(3); du = randn(T, n - 1)   # diag-dominant → nonsingular
        A = diagm(-1 => dl, 0 => d, 1 => du)
        for nrhs in (1, 3)
            B = randn(T, n, nrhs)
            X = PureBLAS.gtsv!(copy(dl), copy(d), copy(du), copy(B))            # combined factor+solve
            @test maximum(abs, A * X - B) < sqrt(eps(real(T))) * 50 * (norm(A) + 1)
        end
        # factor once, solve with trans variants
        dl2, d2, du2, du22, ipiv = PureBLAS.gttrf!(
            copy(dl), copy(d), copy(du), Vector{T}(undef, max(n - 2, 0)),
            Vector{Int}(undef, n)
        )
        for (tr, Aop) in (('N', A), ('T', transpose(A)), ('C', A'))
            B = randn(T, n, 2)
            X = PureBLAS.gttrs!(tr, dl2, d2, du2, du22, ipiv, copy(B))
            @test maximum(abs, Aop * X - B) < sqrt(eps(real(T))) * 100 * (norm(A) + 1)
        end
    end
    @test_throws LinearAlgebra.SingularException PureBLAS.gtsv!([0.0], [0.0, 0.0], [1.0], [1.0, 1.0])  # [[0 1];[0 0]] singular

    # Row-interchange coverage. Everything above is diagonally dominant, so it only ever takes the
    # no-interchange branch — which leaves gtsv!'s fast loop → general loop transition, its deferred
    # dl prefix fill, and its 2-term/3-term back-solve selection untested. "pivot" makes |d| ≪ |dl| so
    # nearly every step swaps; "mixed" is dominant for the first half and swap-heavy for the second,
    # which is the case that actually crosses between the two loops mid-factorization.
    @testset "interchange $T n=$n $tag" for T in (Float32, Float64, ComplexF32, ComplexF64),
            n in (2, 7, 40, 129), tag in ("pivot", "mixed")

        # VERSION-STABLE SEED. `hash` is NOT stable across Julia releases, so seeding with it drew a
        # different matrix on CI (1.12) than locally (1.13): the case was unreproducible by
        # construction, and that is why master's red CI read as an AVX2-only kernel bug for two days
        # when it was neither AVX2-specific nor a kernel bug.
        Random.seed!(1000 * findfirst(==(T), (Float32, Float64, ComplexF32, ComplexF64)) +
                     10 * n + (tag == "pivot" ? 1 : 2))
        dl = randn(T, n - 1); du = randn(T, n - 1)
        d = tag == "pivot" ? randn(T, n) .* real(T)(1.0e-3) :
            [i <= n ÷ 2 ? randn(T) + T(4) : randn(T) * real(T)(1.0e-3) for i in 1:n]
        A = diagm(-1 => dl, 0 => d, 1 => du)
        B = randn(T, n, 2)
        # solution AND the overwritten factor arrays must both match LAPACK (dl is a documented gtsv
        # output — the deferred fill must reproduce it exactly, not just leave B correct)
        pdl, pd, pdu, pB = copy(dl), copy(d), copy(du), copy(B)
        rdl, rd, rdu, rB = copy(dl), copy(d), copy(du), copy(B)
        PureBLAS.gtsv!(pdl, pd, pdu, pB)
        LA.gtsv!(rdl, rd, rdu, rB)
        tol = sqrt(eps(real(T))) * 100 * (norm(A) + 1)
        # BACKWARD error, not a forward residual. A solver's contract is backward stability; a small
        # forward residual is NOT implied by it and is not the solver's to promise. With the "pivot"
        # tag the diagonal is ~1e-3, and Float32 at n=129 can draw cond(A) ~ 1e11 — there a forward
        # residual of 724 against a tol of 0.57 is the MATRIX's doing, and LAPACK's residual is
        # identical to the last bit (measured: |pB - rB| == 0 exactly, every seed).
        #
        # ⚠ AND THE RESIDUAL IS LARGE BECAUSE THE SOLUTION IS LARGE — NOT because accuracy was lost.
        # Measured against a BigFloat solve of the same A,B at cond(A) = 8.9e10: relative forward
        # error 9.7e-08 for BOTH PureBLAS and OpenBLAS, i.e. essentially full Float32 accuracy, where
        # the classical cond*eps bound would permit 1.1e+04. (cond*eps is a worst case over all
        # right-hand sides; a random B rarely aligns with the offending singular direction, and
        # tridiagonal partial pivoting is governed by componentwise conditioning, far smaller than the
        # normwise cond for this structure.) Near-singular A simply makes ‖x‖ enormous, so an absolute
        # residual ‖Ax - b‖ is large for a CORRECT answer. The old assertion penalised the magnitude
        # of a right answer. Normalising by
        # ‖A‖·‖x‖ + ‖b‖ removes the conditioning. BOUND CHOSEN FROM MEASUREMENT, not taste: the worst
        # backward error over this exact grid (4 types × 4 n × 2 tags, these seeds) is 0.35·eps
        # (ComplexF32 n=2 pivot), so 16·eps is 46× headroom — enough to absorb platform and LLVM
        # variation, tight enough to still fail a solver that is actually broken. 100·eps was the
        # first draft and was a 286× rubber stamp.
        nrmA = opnorm(A, Inf); nrmB = norm(B, Inf)
        @test maximum(abs, A * pB - B) / (nrmA * norm(pB, Inf) + nrmB) < 16 * eps(real(T))
        @test maximum(abs, pB - rB) < tol
        @test maximum(abs, pd - rd) < tol
        n > 1 && @test maximum(abs, pdu - rdu) < tol
        n > 2 && @test maximum(abs, pdl[1:(n - 2)] - rdl[1:(n - 2)]) < tol
        # gttrf's pivot bookkeeping (ipiv/du2 defaults now written inside the branches, not prologues)
        pf = PureBLAS.gttrf!(copy(dl), copy(d), copy(du), Vector{T}(undef, max(n - 2, 0)), Vector{Int}(undef, n))
        rf = LA.gttrf!(copy(dl), copy(d), copy(du))
        @test pf[5] == rf[5]                                       # ipiv identical
        @test maximum(abs, pf[2] - rf[2]) < tol                    # U diagonal
        n > 2 && @test maximum(abs, pf[4] - rf[4]) < tol           # du2 second superdiagonal
        Xg = PureBLAS.gttrs!('N', pf[1], pf[2], pf[3], pf[4], pf[5], copy(B))
        @test maximum(abs, A * Xg - B) / (nrmA * norm(Xg, Inf) + nrmB) < 16 * eps(real(T))
    end
end

@testitem "stev engine (_sterf!/_steqr!) vs LAPACK — SymTridiagonal values + vectors (stev C-ABI core)" begin
    using PureBLAS, LinearAlgebra, Random
    Random.seed!(0x1234)
    # The stev/stegr C-ABI wrappers compose these native kernels: _sterf! (values) and _steqr!('I',
    # values + eigenvectors). Test them directly (there is no public PureBLAS.stev! — it is C-ABI only).
    @testset "$T n=$n" for T in (Float32, Float64), n in (1, 2, 8, 50)
        dv = randn(T, n); ev = randn(T, n - 1)
        A = SymTridiagonal(dv, ev)
        wref = eigvals(A)                                            # LAPACK stev/steqr reference
        d = copy(dv); e = copy(ev); PureBLAS._sterf!(d, e)          # values-only (stev job='N')
        @test maximum(abs, sort(d) .- sort(wref)) < sqrt(eps(T)) * 50 * (norm(dv) + norm(ev) + 1)
        dv2 = copy(dv); ev2 = copy(ev); Z = Matrix{T}(I, n, n)     # init I (steqr('I') skips init at n=1)
        PureBLAS._steqr!('I', dv2, ev2, Z)                          # values + vectors (stev job='V')
        @test maximum(abs, sort(dv2) .- sort(wref)) < sqrt(eps(T)) * 50 * (norm(dv) + norm(ev) + 1)
        @test maximum(abs, Z' * Z - I) < sqrt(eps(T)) * 50           # orthonormal vectors
        @test maximum(abs, Matrix(A) * Z - Z * Diagonal(dv2)) < sqrt(eps(T)) * 100 * (norm(dv) + norm(ev) + 1)
    end
end

@testitem "sysv/hesv (symmetric-indefinite/Hermitian solve) + sytri/hetri (inverse) vs LAPACK" begin
    using PureBLAS, LinearAlgebra
    tol(::Type{T}) where {T} = sqrt(eps(real(T))) * 200
    @testset "$T n=$n uplo=$uplo herm=$herm" for T in (Float32, Float64, ComplexF32, ComplexF64),
            n in (1, 2, 5, 17, 40), uplo in ('L', 'U'), herm in (false, true)

        (herm && !(T <: Complex)) && continue
        M = randn(T, n, n)
        A = herm ? (M + M') : (M + transpose(M))
        A += n * I     # keep well away from exact singularity (sytri divides by pivots)
        B = randn(T, n, 3)
        Asol = copy(A); Bsol = copy(B)
        herm ? PureBLAS.hesv!(uplo, Asol, Bsol) : PureBLAS.sysv!(uplo, Asol, Bsol)
        @test norm(A * Bsol - B) <= tol(T) * (norm(A) * norm(Bsol) + norm(B))

        # inverse: sytrf/hetrf then sytri/hetri; A*Ainv ≈ I (only the uplo triangle of Ainv is filled).
        ipiv = zeros(Int, n)
        LD = copy(A)
        herm ? PureBLAS.hetrf!(LD, ipiv; uplo = uplo) : PureBLAS.sytrf!(LD, ipiv; uplo = uplo)
        Ainv = copy(LD)
        herm ? PureBLAS.hetri!(Ainv, ipiv; uplo = uplo) : PureBLAS.sytri!(Ainv, ipiv; uplo = uplo)
        Afull = uplo == 'L' ? (herm ? Hermitian(Ainv, :L) : Symmetric(Ainv, :L)) :
            (herm ? Hermitian(Ainv, :U) : Symmetric(Ainv, :U))
        @test norm(A * Matrix(Afull) - I) <= tol(T) * n
    end
end

# `gbtrf` is a PIVOTED factorization, and until this item was widened its inputs could not pivot: it
# built the band then did `A += n*I`, so every column's argmax was the diagonal and `ipiv[j] == j`
# throughout. The bit-exact `ipiv == ipivl` assertion was therefore true but vacuous — it never once
# compared a nontrivial pivot. Three generators fix that, and the third is the one that matters:
#
#   :dominant — the old behaviour. `ju` never runs ahead, so the fill-in/deep-swap machinery is idle.
#   :pivot    — random band with a shrunk diagonal. Most columns now pivot; the `nontrivial > half`
#               assertion below FAILS LOUDLY if a future change makes the generator stop pivoting.
#   :deep     — forces the argmax onto the LAST subdiagonal of every column, so `jp == km+1`, `ju`
#               runs maximally ahead, and the swap reaches its furthest column. This is the only
#               generator that exercises the deep-pivot path at all.
#
# Bit-exact `ipiv` IS a sound oracle here (unlike Bunch-Kaufman): partial pivoting is a single
# well-defined argmax per column with no re-association. If a blocked port ever makes a cell flake on
# a near-tie, the fix is a different seed, not a weaker assertion.
@testitem "gbtrf/gbtrs (general banded LU) vs LAPACK — pivoting, deep pivots, rectangular, padded ldab" begin
    using PureBLAS, LinearAlgebra, Random
    import LinearAlgebra.LAPACK as LA

    # Dense band -> LAPACK GB storage with a caller-chosen ldab (>= 2kl+ku+1).
    function tob(A, m, n, kl, ku, ldab)
        AB = zeros(eltype(A), ldab, n)
        for j in 1:n, i in max(1, j - ku):min(m, j + kl)
            AB[kl + ku + 1 + i - j, j] = A[i, j]
        end
        return AB
    end

    @testset "$T $m×$n kl=$kl ku=$ku gen=$gen pad=$pad" for T in (Float32, Float64, ComplexF32, ComplexF64),
            (m, n, kl, ku) in ((8, 8, 1, 1), (20, 20, 2, 3), (35, 35, 3, 1),
                               (48, 48, 12, 6), (50, 32, 10, 6), (32, 50, 10, 6)),
            gen in (:dominant, :pivot, :deep), pad in (0, 3)

        Random.seed!(hash((T, m, n, kl, ku, gen, pad)))
        A = zeros(T, m, n)
        for j in 1:n, i in max(1, j - ku):min(m, j + kl)
            A[i, j] = randn(T)
        end
        # :pivot is just the plain random band — with no diagonal boost the argmax lands off the
        # diagonal most of the time. :deep RAISES the last subdiagonal rather than shrinking the
        # diagonal: shrinking it (tried first) forces the pivots but also drives cond(A) through the
        # roof, and the solve residuals then blow past any backward-error bound for reasons that have
        # nothing to do with gbtrf. Raising the subdiagonal forces jp = km+1 while the matrix stays
        # generically conditioned.
        if gen === :dominant
            for j in 1:min(m, n); A[j, j] += T(4 * (kl + ku + 1)); end
        elseif gen === :deep && kl > 0
            for j in 1:n
                i = min(m, j + kl)
                i > j && (A[i, j] = T(100))                      # argmax onto the LAST subdiagonal
            end
        end

        ldab = 2kl + ku + 1 + pad                                # pad ⇒ the port must use stride(AB,2)
        ABp = tob(A, m, n, kl, ku, ldab)
        # Sentinel below the band: dgbtrf never touches storage rows kv+kl+2..ldab. An ld-1 view whose
        # declared rectangle overruns writes here, and nothing else in the suite would notice.
        pad > 0 && (ABp[(kl + ku + kl + 2):ldab, :] .= T(-12345))
        ABl = copy(ABp)

        _, ipiv, info = PureBLAS.gbtrf!(kl, ku, m, ABp)
        @test info == 0
        _, ipivl = LA.gbtrf!(kl, ku, m, ABl)
        @test ipiv == ipivl                                      # bit-exact pivot sequence
        @test maximum(abs, ABp .- ABl) < 200 * eps(real(T)) * maximum(abs, ABl)
        pad > 0 && @test all(==(T(-12345)), ABp[(kl + ku + kl + 2):ldab, :])

        mn = min(m, n)
        # Fails loudly if a future change makes the generator stop pivoting — the exact hole that let
        # the old `A += n*I` version assert `ipiv == ipivl` while every pivot was trivially j.
        #
        # RESTRICTED TO kl >= 2, and that restriction IS the fix for a real flake (2026-08-16). A column
        # pivots with probability ≈ kl/(kl+1), so the non-trivial count is ≈ Binomial(mn-1, kl/(kl+1)).
        # At kl=1 that is a fair coin: mn=8 gives mean 3.5, and the previous `>= mn÷3` (= 2) threshold
        # failed whenever the draw came in at 0 or 1 — P = 6.25% per cell, over 8 such cells (4 types ×
        # 2 pads, the (8,8,1,1) shape being the only kl=1 one). No threshold at kl=1 is both meaningful
        # and reliable, so the check now rests on the kl>=2 shapes, where it sits far from the tail
        # (kl=2, mn=20: mean 12.7 against a threshold of 5, ≈3.8σ down).
        #
        # WHY IT SURFACED ONLY NOW, and why it looked like a gbtrf bug: the seed is
        # `hash((T, m, n, kl, ku, gen, pad))`, which hashes a DataType — and `hash(::DataType)` is NOT
        # stable across Julia versions. Measured for this very cell: 1.12 → 16763250882009543865,
        # 1.13 → 263051053535568488. So every Julia upgrade redraws every matrix and reshuffles which
        # cells land in the tail. That is why 1.12 CI failed `ComplexF32 pad=0` while 1.13 failed
        # `ComplexF64 pad=3` — one defect, different victim, and neither was a factorization error.
        if gen !== :dominant && kl >= 2
            @test count(!=(0), ipiv .- (1:mn)) >= mn ÷ 4
        end

        if m == n
            for tr in ('N', 'T', 'C')
                Bv = randn(T, n, 2); Bp = copy(Bv)
                PureBLAS.gbtrs!(tr, kl, ku, n, ABp, ipiv, Bp)
                Aop = tr == 'N' ? A : (tr == 'T' ? transpose(A) : A')
                # Backward-error form: the ‖A‖·‖X‖ term is what makes this conditioning-independent.
                # Without it (the old bound) an ill-conditioned cell fails for reasons unrelated to
                # gbtrf, because the residual scales with ‖X‖ but the bound did not.
                @test norm(Aop * Bp - Bv) <=
                      200 * n * eps(real(T)) * (norm(A) * norm(Bp) + norm(Bv))
            end
        end
    end

    # BLOCKED kernel, called BY NAME with an explicit nb. Going through `gbtrf!` would not do: its
    # gate is kl ≥ 2·nb with nb = 48, so reaching the blocked path through the front door needs
    # kl ≥ 96 and n in the thousands — far too slow for a correctness item, and the coverage would
    # silently depend on the host's measured nb. This is the `pbtrf` precedent (see :24-31), whose
    # header is a post-mortem of a blocked kernel that shipped with no test reaching it.
    #
    # ORACLE is `_gbtf2!` on the same input, which is sharper than comparing against stdlib: a wrong
    # undo loop or a mis-offset ipiv produces a STRUCTURALLY different factor, not a roundoff-level
    # one. Measured separation is ~13 orders of magnitude (valid cells 1e-13, nb>kl cells 1e0-1e4).
    # `ipiv` is asserted bit-exact — partial pivoting is one argmax per column, no re-association.
    # The FACTOR tolerance is loose (1e4·eps) on purpose: blocked and unblocked reassociate the
    # arithmetic (rank-1 sweep vs rank-nb gemm), so they legitimately differ by roundoff × growth.
    @testset "blocked $T $m×$n kl=$kl ku=$ku nb=$nb gen=$gen pad=$pad" for
            T in (Float32, Float64, ComplexF32, ComplexF64),
            (m, n, kl, ku) in ((24, 24, 8, 4), (64, 64, 16, 16), (40, 40, 8, 8),
                               (40, 40, 12, 0), (50, 32, 10, 6), (32, 50, 10, 6)),
            nb in (1, 2, 3, 5, 8), gen in (:dominant, :pivot, :deep), pad in (0, 3)

        nb > kl && continue                                  # structural precondition of dgbtrf
        Random.seed!(hash((T, m, n, kl, ku, gen, pad)))
        A = zeros(T, m, n)
        for j in 1:n, i in max(1, j - ku):min(m, j + kl); A[i, j] = randn(T); end
        if gen === :dominant
            for j in 1:min(m, n); A[j, j] += T(4 * (kl + ku + 1)); end
        elseif gen === :deep && kl > 0
            for j in 1:n; i = min(m, j + kl); i > j && (A[i, j] = T(100)); end
        end

        ldab = 2kl + ku + 1 + pad
        ABb = tob(A, m, n, kl, ku, ldab); ABu = copy(ABb)
        if pad > 0                                           # out-of-band canary on BOTH copies
            ABb[(kl + ku + kl + 2):ldab, :] .= T(-12345)
            ABu[(kl + ku + kl + 2):ldab, :] .= T(-12345)
        end
        ipb = zeros(Int, min(m, n)); ipu = zeros(Int, min(m, n))
        infb = PureBLAS._gbtrf_blocked!(kl, ku, m, ABb, ipb, nb)
        infu = PureBLAS._gbtf2!(kl, ku, m, ABu, ipu)

        @test ipb == ipu
        @test infb == infu
        # The factor bound scales with kl, not just with ‖U‖ (fixed 2026-08-16). A band entry
        # accumulates up to `kl` rank-1 updates in `_gbtf2!`, which the blocked driver re-associates
        # into ⌈kl/nb⌉ rank-nb gemms, so the legitimate divergence grows with the number of
        # accumulation steps — a flat constant silently tightens as kl rises. The old flat `1e4·eps·‖U‖`
        # was calibrated on the narrower shapes and left `Float32 40×40 kl=12 ku=0 gen=pivot` over by
        # 1.45×/1.07×/1.03× at nb=2/3/5 while PASSING at nb=1 and nb=8 — exactly the signature of
        # re-association, not of a factorization error: `ipiv` is bit-exact in every one of those cells,
        # so both paths chose an identical pivot sequence and only the summation order differed.
        @test maximum(abs, ABb .- ABu) <=
              2.0e3 * (kl + 1) * eps(real(T)) * max(maximum(abs, ABu), one(real(T)))
        pad > 0 && @test all(==(T(-12345)), ABb[(kl + ku + kl + 2):ldab, :])
    end

    # nb > kl is outside the algorithm's domain (i2 = kl-jb goes negative) — must throw, not
    # silently return a wrong factor. Verified: those cells diverge from _gbtf2! by O(1).
    @test_throws ArgumentError PureBLAS._gbtrf_blocked!(4, 2, 20, zeros(Float64, 11, 20),
        zeros(Int, 20), 8)

    # Singular: an exactly-zero band column ⇒ info names it. Note LA.gbtrf! runs chklapackerror and
    # THROWS on info>0, so it cannot be used as an oracle here — assert against PureBLAS directly.
    @testset "singular $T n=$n kl=$kl ku=$ku" for T in (Float64, ComplexF64),
            (n, kl, ku) in ((20, 2, 3), (48, 12, 6))

        Random.seed!(hash((T, n, kl, ku, :sing)))
        A = zeros(T, n, n)
        for j in 1:n, i in max(1, j - ku):min(n, j + kl); A[i, j] = randn(T); end
        for j in 1:n; A[j, j] += T(4 * (kl + ku + 1)); end
        c = n ÷ 2
        for i in max(1, c - ku):min(n, c + kl); A[i, c] = zero(T); end
        AB = zeros(T, 2kl + ku + 1, n)
        for j in 1:n, i in max(1, j - ku):min(n, j + kl)
            AB[kl + ku + 1 + i - j, j] = A[i, j]
        end
        _, _, info = PureBLAS.gbtrf!(kl, ku, n, AB)
        @test info == c
        @test all(isfinite, AB)
    end
end

@testitem "pttrf/pttrs/ptsv (SPD/Hermitian-PD tridiagonal) vs LAPACK" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.LAPACK as LA
    @testset "$T n=$n" for T in (Float32, Float64, ComplexF32, ComplexF64), n in (1, 2, 8, 40)
        R = real(T)
        d = rand(R, n) .+ R(n + 2)                              # diagonally dominant → SPD/HPD
        e = randn(T, max(n - 1, 0)) .* R(0.3)
        A = zeros(T, n, n)
        for i in 1:n
            A[i, i] = d[i]
        end
        for i in 1:(n - 1)
            A[i + 1, i] = e[i]; A[i, i + 1] = conj(e[i])
        end
        d1 = copy(d); e1 = copy(e)
        _, _, info = PureBLAS.pttrf!(d1, e1)
        @test info == 0
        dl = copy(d); el = copy(e); LA.pttrf!(dl, el)
        @test maximum(abs, d1 .- dl) < 200 * eps(R) * maximum(d)
        n > 1 && @test maximum(abs, e1 .- el) < 200 * eps(R) * (maximum(abs, e1) + 1)
        Bv = randn(T, n, 3)
        Bp = copy(Bv); PureBLAS.pttrs!(d1, e1, Bp; uplo = 'L')
        @test norm(A * Bp - Bv) < sqrt(eps(R)) * 200 * (norm(A) + 1)
        d2 = copy(d); e2 = copy(e); B2 = copy(Bv)
        _, _, _, info2 = PureBLAS.ptsv!(d2, e2, B2)
        @test info2 == 0
        @test norm(A * B2 - Bv) < sqrt(eps(R)) * 200 * (norm(A) + 1)
    end
end

@testitem "stebz/stein (sym-tridiag eigenvalues by bisection / eigenvectors by inverse iteration) vs LAPACK" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.LAPACK as LA
    @testset "$T n=$n" for T in (Float32, Float64), n in (2, 8, 30)
        d = randn(T, n); e = randn(T, max(n - 1, 0))
        A = SymTridiagonal(d, e)
        wref = eigvals(A)
        w, iblock, isplit, info = PureBLAS.stebz!('A', 'E', T(0), T(0), 0, 0, T(-1), copy(d), copy(e))
        @test info == 0
        @test length(w) == n
        @test maximum(abs, sort(w) .- sort(wref)) < sqrt(eps(T)) * 50 * (norm(d) + norm(e) + 1)
        Z = PureBLAS.stein!(d, e, w, iblock, isplit)
        @test size(Z) == (n, n)
        @test maximum(abs, Matrix(A) * Z - Z * Diagonal(w)) < sqrt(eps(T)) * 400 * (norm(d) + norm(e) + 1)
        for j in 1:n
            @test abs(norm(view(Z, :, j)) - 1) < sqrt(eps(T)) * 10
        end
        wl, ibl, isl = LA.stebz!('A', 'E', T(0), T(0), 0, 0, T(-1), copy(d), copy(e))
        @test maximum(abs, sort(w) .- sort(wl)) < sqrt(eps(T)) * 50 * (norm(d) + norm(e) + 1)
    end
end

@testitem "gelsd (rank-deficient LS via SVD) vs LAPACK — over/under-determined + rank-deficient" begin
    using PureBLAS, LinearAlgebra
    tol(::Type{T}) where {T} = sqrt(eps(real(T))) * 200
    @testset "$T $m×$n" for T in (Float32, Float64, ComplexF32, ComplexF64),
            (m, n) in ((30, 15), (15, 30), (20, 20))

        A = randn(T, m, n)
        nrhs = 2
        Bv = zeros(T, max(m, n), nrhs); b0 = randn(T, m, nrhs); Bv[1:m, :] = b0
        Bsol, rk, s = PureBLAS.gelsd!(copy(A), Bv, -1.0)
        @test rk == min(m, n)
        X = Bsol[1:n, :]
        sref = svdvals(A)
        @test maximum(abs, s .- sref) / maximum(sref) < 1.0e-3
        if m >= n
            @test norm(A' * (A * X - b0)) <= tol(T) * (norm(A)^2 * norm(X) + norm(A) * norm(b0))
        else
            @test norm(A * X - b0) <= tol(T) * (norm(A) * norm(X) + norm(b0))
        end
    end
    @testset "rank-deficient $T" for T in (Float64, ComplexF64)
        m, n = 20, 10
        U = Matrix(qr(randn(T, m, m)).Q); V = Matrix(qr(randn(T, n, n)).Q)
        s = Float64[5, 4, 3, 0, 0, 0, 0, 0, 0, 0]
        A = U[:, 1:n] * Diagonal(T.(s)) * V'
        Bv = zeros(T, m, 1); Bv[:, 1] = randn(T, m)
        _, rk, _ = PureBLAS.gelsd!(copy(A), copy(Bv), 1.0e-8)
        @test rk == 3
    end
end

@testitem "gelsy (rank-deficient LS via RZ) + tzrzf/ormrz vs LAPACK" begin
    using PureBLAS, LinearAlgebra
    tol(::Type{T}) where {T} = sqrt(eps(real(T))) * 200
    @testset "$T $m×$n" for T in (Float32, Float64, ComplexF32, ComplexF64),
            (m, n) in ((30, 15), (15, 30), (20, 20))

        A = randn(T, m, n)
        nrhs = 2
        Bv = zeros(T, max(m, n), nrhs); b0 = randn(T, m, nrhs); Bv[1:m, :] = b0
        jpvt = zeros(Int, n)
        Bsol, rk = PureBLAS.gelsy!(copy(A), Bv, jpvt, -1.0)
        @test rk == min(m, n)
        X = Bsol[1:n, :]
        if m >= n
            @test norm(A' * (A * X - b0)) <= tol(T) * (norm(A)^2 * norm(X) + norm(A) * norm(b0))
        else
            @test norm(A * X - b0) <= tol(T) * (norm(A) * norm(X) + norm(b0))
        end
    end
    # tzrzf/ormrz directly: reduce an already-upper-trapezoidal (m≤n) A to upper-triangular R via a
    # Householder-Z from the right, then round-trip through ormrz to recover A (Z orthogonal/unitary).
    @testset "tzrzf/ormrz $T m=$m n=$n" for T in (Float32, Float64, ComplexF32, ComplexF64), (m, n) in ((5, 5), (4, 9))
        A0 = triu(randn(T, m, n))
        F = copy(A0); tau = zeros(T, m)
        PureBLAS.tzrzf!(F, tau)
        R = triu(F[:, 1:m])
        C = zeros(T, m, n); C[:, 1:m] = R
        PureBLAS.ormrz!('R', 'N', F, tau, C)
        @test maximum(abs, C .- A0) < 2000 * eps(real(T)) * max(maximum(abs, A0), 1)
        trH = T <: Complex ? 'C' : 'T'
        C2 = copy(C)
        PureBLAS.ormrz!('R', trH, F, tau, C2)
        @test maximum(abs, C2[:, 1:m] .- R) < 2000 * eps(real(T)) * max(maximum(abs, R), 1)
    end
end

@testitem "pstrf (pivoted/semidefinite Cholesky) vs LAPACK — full-rank + rank-deficient, both uplo" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.LAPACK as LA
    @testset "$T n=$n uplo=$uplo" for T in (Float32, Float64, ComplexF32, ComplexF64), n in (1, 5, 20), uplo in ('L', 'U')
        M = randn(T, n, n); A = M * M' + n * I    # full-rank PD
        piv = zeros(Int, n)
        F, pv, rk, info = PureBLAS.pstrf!(copy(A), piv, -1.0; uplo = uplo)
        @test info == 0
        @test rk == n
        Aperm = A[pv, pv]
        recon = uplo == 'L' ? tril(F) * tril(F)' : triu(F)' * triu(F)
        @test maximum(abs, recon .- Aperm) < 2000 * eps(real(T)) * maximum(abs, A)
        Fl, pvl, rankl, infol = LA.pstrf!(uplo, copy(A), -1.0)
        @test rankl == rk
    end
    @testset "rank-deficient $T" for T in (Float64, ComplexF64)
        n = 10; k = 4
        M = randn(T, n, k); A = M * M'             # rank k ≤ n, PSD
        piv = zeros(Int, n)
        F, pv, rk, info = PureBLAS.pstrf!(copy(A), piv, -1.0; uplo = 'L')
        @test rk == k
        Aperm = A[pv, pv]
        Lr = tril(F)[:, 1:rk]
        @test maximum(abs, Lr * Lr' .- Aperm) < 1.0e-6 * maximum(abs, A)
    end
end

@testitem "QL/RQ (geqlf/gerqf/orgql/orgrq/ormql/ormrq) vs LAPACK — reconstruction + orthonormal Q + apply" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.LAPACK as LA
    maxe(x, y) = maximum(abs.(x .- y)) / max(maximum(abs.(y)), 1.0e-300)
    @testset "QL $T $m×$n" for T in (Float32, Float64, ComplexF32, ComplexF64), (m, n) in ((8, 5), (8, 8), (30, 18))
        A0 = randn(T, m, n); k = min(m, n)
        F = copy(A0); tau = zeros(T, k); PureBLAS.geqlf!(F, tau)
        Fl = copy(A0); taul = zeros(T, k); LA.geqlf!(Fl, taul)     # LAPACK reference, SAME τ convention
        @test maxe(tril(F[(m - k + 1):m, (n - k + 1):n]), tril(Fl[(m - k + 1):m, (n - k + 1):n])) < 400 * eps(real(T))
        @test maxe(tau, taul) < 400 * eps(real(T))
        Q = PureBLAS.orgql!(copy(F), copy(tau))
        @test maxe(Q' * Q, Matrix{T}(I, n, n)) < 800 * eps(real(T))
        Lql = tril(F[(m - n + 1):m, :])          # economy L: n×n lower-tri bottom block, A = Q(m×n)·L(n×n)
        @test maxe(Q * Lql, A0) < 800 * eps(real(T))
        C = randn(T, m, 6); trH = T <: Complex ? 'C' : 'T'
        C1 = PureBLAS.ormql!('L', trH, copy(F), tau, copy(C))
        C2 = PureBLAS.ormql!('L', 'N', copy(F), tau, copy(C1))
        @test maxe(C2, C) < 800 * eps(real(T))
    end
    @testset "RQ $T $m×$n" for T in (Float32, Float64, ComplexF32, ComplexF64), (m, n) in ((5, 8), (8, 8), (18, 30))
        A0 = randn(T, m, n); k = min(m, n)
        F = copy(A0); tau = zeros(T, k); PureBLAS.gerqf!(F, tau)
        Fl = copy(A0); taul = zeros(T, k); LA.gerqf!(Fl, taul)
        @test maxe(triu(F[:, (n - m + 1):n]), triu(Fl[:, (n - m + 1):n])) < 400 * eps(real(T))
        @test maxe(tau, taul) < 400 * eps(real(T))
        Q = PureBLAS.orgrq!(copy(F), copy(tau))
        @test maxe(Q * Q', Matrix{T}(I, m, m)) < 800 * eps(real(T))
        Rrq = triu(F[:, (n - m + 1):n])          # economy R: m×m upper-tri right block, A = R(m×m)·Q(m×n)
        @test maxe(Rrq * Q, A0) < 800 * eps(real(T))
        C = randn(T, 6, n); trH = T <: Complex ? 'C' : 'T'
        C1 = PureBLAS.ormrq!('R', 'N', copy(F), tau, copy(C))
        C2 = PureBLAS.ormrq!('R', trH, copy(F), tau, copy(C1))
        @test maxe(C2, C) < 800 * eps(real(T))
    end
end

@testitem "trsyl (triangular Sylvester solve) vs LAPACK — op(A)X ± X op(B) = scale·C, all trans/isgn" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.LAPACK as LA
    @testset "$T m=$m n=$n ta=$ta tb=$tb isgn=$isgn" for T in (Float32, Float64, ComplexF32, ComplexF64),
            (m, n) in ((5, 5), (8, 6)), ta in (T <: Complex ? ('N', 'C') : ('N', 'T')),
            tb in (T <: Complex ? ('N', 'C') : ('N', 'T')), isgn in (1, -1)

        A0 = triu(randn(T, m, m)) + m * I           # well-separated diagonal (avoid a near-singular Sylvester op)
        B0 = triu(randn(T, n, n)) .* T(0.3) + 3m * I
        C0 = randn(T, m, n)
        Ap = copy(A0); Bp = copy(B0); Cp = copy(C0)
        Xp, scale = PureBLAS.trsyl!(ta, tb, isgn, Ap, Bp, Cp)
        opA = ta == 'N' ? A0 : (ta == 'T' ? transpose(A0) : A0')
        opB = tb == 'N' ? B0 : (tb == 'T' ? transpose(B0) : B0')
        resid = opA * Xp + isgn * Xp * opB - scale * C0
        @test norm(resid) < sqrt(eps(real(T))) * 400 * (norm(A0) * norm(B0) * norm(Xp) + norm(C0))
        Al = copy(A0); Bl = copy(B0); Cl = copy(C0)
        Xl, scalel = LA.trsyl!(ta, tb, Al, Bl, Cl, isgn)
        @test norm(Xp .- Xl) < sqrt(eps(real(T))) * 400 * (norm(Xl) + 1)
    end
end

@testitem "trexc/trsen (Schur reorder) vs LAPACK — similarity-preserving block swap, condition numbers" begin
    using PureBLAS, LinearAlgebra
    @testset "trexc $T n=$n" for T in (Float64, ComplexF64), n in (5, 12)
        A0 = randn(T, n, n)
        S = schur(A0)
        Torig = Matrix(S.T); Q0 = Matrix(S.Z)
        Tm = copy(Torig); Qm = copy(Q0)
        ifst, ilst = 1, min(3, n)
        PureBLAS.trexc!('V', Tm, Qm, ifst, ilst)
        @test maximum(abs, Qm * Tm * Qm' - A0) < 1.0e-9 * (opnorm(A0, 1) + 1)
        @test maximum(abs, Qm' * Qm - I) < 1.0e-9
    end
    @testset "trsen $T n=$n" for T in (Float64, ComplexF64), n in (6, 14)
        A0 = randn(T, n, n)
        S = schur(A0)
        Torig = Matrix(S.T); Q0 = Matrix(S.Z)
        Tm = copy(Torig); Qm = copy(Q0)
        sel = falses(n); sel[1] = true               # select the leading eigenvalue('s conjugate-pair block)
        Tr, Qr, w, s, sep = PureBLAS.trsen!('B', 'V', sel, Tm, Qm)
        @test maximum(abs, Qr * Tr * Qr' - A0) < 1.0e-8 * (opnorm(A0, 1) + 1)
        @test maximum(abs, Qr' * Qr - I) < 1.0e-8
        @test 0 < s <= 1 + 1.0e-8
        @test sep >= 0
    end
end

@testitem "gglse (equality-constrained least squares) vs LAPACK — Bx=d exactly, ‖Ax−c‖ residual" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.LAPACK as LA
    @testset "$T m=$m n=$n p=$p" for T in (Float32, Float64, ComplexF32, ComplexF64),
            (m, n, p) in ((10, 6, 3), (8, 8, 4), (12, 7, 7))

        A = randn(T, m, n); c = randn(T, m)
        B = randn(T, p, n); d = B * randn(T, n)     # ensure B·x=d is consistent
        x, res = PureBLAS.gglse!(copy(A), copy(c), copy(B), copy(d))
        @test norm(B * x - d) < sqrt(eps(real(T))) * 400 * (norm(B) * norm(x) + norm(d) + 1)
        @test abs(res - norm(A * x - c)) < sqrt(eps(real(T))) * 400 * (norm(A) * norm(x) + norm(c) + 1)
        xl, resl = LA.gglse!(copy(A), copy(c), copy(B), copy(d))
        @test norm(x .- xl) < sqrt(eps(real(T))) * 1000 * (norm(xl) + 1)
    end
end

@testitem "ggsvd (generalized SVD, rank-deficient-capable, s/d/c/z) vs LAPACK — UᴴAQ=D1·[0 R], VᴴBQ=D2·[0 R]" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.LAPACK as LA
    # D1/D2 block forms from the LAPACK dggsvd doc
    function d1d2(::Type{T}, m, p, k, l, alpha, beta) where {T}
        kl = k + l
        D1 = zeros(T, m, kl); D2 = zeros(T, p, kl)
        for i in 1:k
            D1[i, i] = 1
        end
        if m - kl >= 0
            for i in 1:l
                D1[k + i, k + i] = alpha[k + i]; D2[i, k + i] = beta[k + i]
            end
        else
            for i in (k + 1):m
                D1[i, i] = alpha[i]
            end
            for i in 1:(m - k)
                D2[i, k + i] = beta[k + i]
            end
            for i in (m - k + 1):l
                D2[i, k + i] = 1
            end
        end
        return D1, D2
    end
    @testset "T=$T m=$m p=$p n=$n rA=$rA rB=$rB" for T in (Float64, Float32, ComplexF64, ComplexF32),
            (m, p, n) in ((10, 8, 6), (6, 10, 6), (4, 5, 9)), (rA, rB) in ((99, 99), (2, 2))

        RT = real(T)
        A = rA >= min(m, n) ? randn(T, m, n) : randn(T, m, rA) * randn(T, rA, n)
        B = rB >= min(p, n) ? randn(T, p, n) : randn(T, p, rB) * randn(T, rB, n)
        U, V, Q, al, be, k, l, R = PureBLAS.ggsvd!('U', 'V', 'Q', copy(A), copy(B))
        _, _, _, alo, beo, ko, lo, _ = LA.ggsvd3!('U', 'V', 'Q', copy(A), copy(B))
        @test (k, l) == (ko, lo)                       # rank parameters match LAPACK exactly
        @test sort(al) ≈ sort(alo) atol = (RT === Float64 ? 1.0e-9 : 1.0e-4)
        @test sort(be) ≈ sort(beo) atol = (RT === Float64 ? 1.0e-9 : 1.0e-4)
        otol = 100 * max(m, p, n) * eps(RT)
        @test opnorm(U'U - I) < otol
        @test opnorm(V'V - I) < otol
        @test opnorm(Q'Q - I) < otol
        D1, D2 = d1d2(T, m, p, k, l, al, be)
        ZR = [zeros(T, k + l, n - k - l) R]
        rtol = 500 * max(m, p, n) * eps(RT)
        @test opnorm(U' * A * Q - D1 * ZR) < rtol * (opnorm(A) + 1)
        @test opnorm(V' * B * Q - D2 * ZR) < rtol * (opnorm(B) + 1)
        @test maximum(abs, al .^ 2 .+ be .^ 2 .- vcat(ones(RT, k + l), zeros(RT, n - k - l))) < otol
        # job='N' variants reproduce the same k, l, alpha, beta, R
        _, _, _, a2, b2, k2, l2, R2 = PureBLAS.ggsvd!('N', 'N', 'N', copy(A), copy(B))
        @test (k2, l2) == (k, l) && a2 == al && b2 == be && R2 == R
    end
end

# ── generic `T<:Real` factorizations (req#3: Mode 2 must stay differentiable) ──────────────────────────
# WHY THESE EXIST AS TESTS AND NOT PROBES. getrf!/geqrf!/gesvd! were widened from Float64 to T<:Real on
# 2026-08-30 so a ForwardDiff.Dual matrix can reach LU/QR/SVD at all. Nothing in the suite covered a
# non-Float64 Real, so the guarantee had NO CI protection: re-narrowing a signature, or reintroducing an
# unguarded `pointer()` fast path, would have gone unnoticed. The latter is not hypothetical — a
# `_strided1` that tested unit-stride without `isbitstype(eltype)` handed a `Matrix{BigFloat}` to
# `pointerref` and SEGFAULTED Julia's codegen; it took a segfault to find, because no test looked.
#
# THE THREE ELEMENT TYPES ARE NOT INTERCHANGEABLE — each covers a different path:
#   Float32   rides the SIMD kernels (it is a BlasReal) and so proves NOTHING about genericity;
#   Float16   is a bits Real that is NOT a BlasReal ⇒ the generic scalar path a Dual takes;
#   BigFloat  is NON-BITS ⇒ exercises every `pointer()` guard on the route.
# Float64 is included so a regression in the fast path shows up here too.

@testitem "getrf! — generic T<:Real: PA = LU across Float64/Float32/Float16/BigFloat" begin
    using PureBLAS, LinearAlgebra, Random
    Random.seed!(4242)
    function recon_err(A0, A, ipiv)
        m, n = size(A); k = min(m, n); T = eltype(A)
        L = Matrix(LowerTriangular(A))
        for i in 1:min(m, n)
            L[i, i] = one(T)
        end
        U = Matrix(UpperTriangular(A))
        p = collect(1:m)
        for i in 1:k
            p[i], p[ipiv[i]] = p[ipiv[i]], p[i]
        end
        return maximum(abs, (L * U)[:, 1:n] .- A0[p, :])
    end
    @testset "$T" for (T, tol) in ((Float64, 1.0e-12), (Float32, 1.0e-4), (Float16, 0.5), (BigFloat, 1.0e-60))
        @testset "n=$n" for n in (8, 16, 33)
            A0 = T.(randn(n, n) + n * I)
            A = copy(A0); ipiv = Vector{Int}(undef, n)
            PureBLAS.getrf!(A, ipiv)
            @test recon_err(A0, A, ipiv) < tol
        end
        # the 1-arg convenience form must accept a non-Float64 Real too
        _, ip, info = PureBLAS.getrf!(T.(randn(12, 12) + 12 * I))
        @test info == 0 && length(ip) == 12
    end
end

@testitem "geqrf! — generic T<:Real: A = QR under the faer tau convention" begin
    using PureBLAS, LinearAlgebra, Random
    Random.seed!(31415)
    # faer convention (NOT LAPACK's): H_k = I − v·vᵀ/τ, and τ=Inf means the identity reflector. Storing
    # LAPACK's τ here instead would make the convention depend on the ELEMENT TYPE — the mismatch class
    # that got a faer-τ/orgqr pairing retracted once already.
    function recon(F, tau, ::Type{T}) where {T}
        m, n = size(F); k = min(m, n)
        R = T[i <= j ? F[i, j] : zero(T) for i in 1:m, j in 1:n]
        for kk in k:-1:1
            isfinite(tau[kk]) || continue
            v = zeros(T, m); v[kk] = one(T); v[(kk + 1):m] = F[(kk + 1):m, kk]
            R .-= (v * (transpose(v) * R)) ./ tau[kk]
        end
        return R
    end
    @testset "$T" for (T, tol) in ((Float64, 1.0e-11), (Float32, 1.0e-4), (Float16, 0.1), (BigFloat, 1.0e-60))
        @testset "$(m)x$(n)" for (m, n) in ((1, 1), (8, 8), (16, 8), (8, 16), (33, 20))
            A0 = T.(randn(m, n))
            F = copy(A0); tau = zeros(T, min(m, n))
            PureBLAS.geqrf!(F, tau)
            rel = maximum(abs, recon(F, tau, T) .- A0) / max(maximum(abs, A0), eps(T))
            @test rel < tol
        end
    end
end

@testitem "gesvd! — generic T<:Real: singular values, and vectors refuse loudly" begin
    using PureBLAS, LinearAlgebra, Random
    Random.seed!(2718)
    @testset "$T" for (T, tol) in ((Float64, 1.0e-12), (Float32, 1.0e-5), (Float16, 5.0e-2), (BigFloat, 1.0e-12))
        @testset "$(m)x$(n)" for (m, n) in ((1, 1), (6, 6), (12, 7), (7, 12), (20, 20))
            A64 = randn(m, n)
            ref = svdvals(A64)                       # Float64 oracle — bounds BigFloat's achievable error
            S = PureBLAS.gesvd!(T.(A64); want_vectors = false)[1]
            @test maximum(abs, Float64.(S) .- ref) / ref[1] < tol
        end
    end
    # Vectors are Float64/complex only. Silently returning values where vectors were asked for is the
    # worse failure, so the generic path must throw.
    @test_throws ArgumentError PureBLAS.gesvd!(Float16.(randn(4, 4)))
    @test_throws ArgumentError PureBLAS.gesvd!(BigFloat.(randn(4, 4)); want_vectors = true)
end
