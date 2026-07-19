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

@testitem "geqrf (QR) vs LAPACK — square/tall/wide, R + reconstruction" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.LAPACK
    maxe(x, y) = maximum(abs.(x .- y)) / max(maximum(abs.(y)), 1e-300)
    # reconstruct A = Q*R from faer-convention factored output (H_k = I − v vᵀ/τ, τ=Inf ⇒ identity)
    function recon(F, tau)
        m, n = size(F); k = min(m, n)
        R = [i <= j ? F[i, j] : 0.0 for i in 1:m, j in 1:n]
        for kk in k:-1:1
            isfinite(tau[kk]) || continue
            v = zeros(m); v[kk] = 1.0; v[kk+1:m] = F[kk+1:m, kk]
            R .-= (v * (v' * R)) ./ tau[kk]
        end
        R
    end
    @testset "$m×$n" for (m, n) in ((1, 1), (8, 8), (40, 25), (64, 64), (129, 96), (200, 200), (96, 160), (600, 513))
        A0 = randn(m, n)
        F = copy(A0); tau = zeros(min(m, n)); PureBLAS.geqrf!(F, tau)
        Fl = copy(A0); LAPACK.geqrf!(Fl)                                   # LAPACK reference
        k = min(m, n)
        @test maxe(abs.(triu(F)[1:k, :]), abs.(triu(Fl)[1:k, :])) < 1e-11   # |R| matches up to sign
        @test maxe(recon(F, tau), A0) < 1e-11                              # Q·R ≈ A
    end
    @test_throws DimensionMismatch PureBLAS.geqrf!(randn(5, 5), zeros(2))
end

@testsetup module WYHelpers
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
        Vv = view(V, pc:m, pc:pc+pb-1)
        Tv = zeros(pb, pb); G = zeros(pb, pb)
        PureBLAS.wy_t!(Tv, Vv, view(tau, pc:pc+pb-1), G)
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
        Vv = view(V, pc:m, pc:pc+pb-1)
        Tv = zeros(pb, pb); G = zeros(pb, pb)
        PureBLAS.wy_t!(Tv, Vv, view(tau, pc:pc+pb-1), G)
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
        @test maxe(triu(C), triu(Af)) < 1e-10          # Qᵀ·A upper part == R

        C2 = copy(C)
        wy_qc!(C2, V, tau, nb)
        @test maxe(C2, A0) < 1e-10                      # Q·(Qᵀ·A) == A
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
    maxe(x, y) = maximum(abs.(x .- y)) / max(maximum(abs.(y)), 1e-300)
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
        @test maxe(F, Fl) < 1e-11                                    # same factorization
        @test ip == ipl                                             # same pivot sequence
        @test recon(A0, F, ip) < 1e-11                              # P·A = L·U
    end
    @test_throws DimensionMismatch PureBLAS.getrf!(randn(5, 5), zeros(Int, 2))
end

@testitem "potrf — ForwardDiff AD through the factor" begin
    using PureBLAS, LinearAlgebra, ForwardDiff
    # L[1,1] of [[x+4, 1],[1, 3]] = sqrt(x+4); d/dx = 1/(2 sqrt(x+4))
    f(x) = (A = [x + 4.0 1.0; 1.0 3.0]; PureBLAS.potrf!(A; uplo = 'L')[1, 1])
    @test ForwardDiff.derivative(f, 1.0) ≈ 0.5 / sqrt(5.0) rtol = 1e-10
    # gradient of logdet via Cholesky: logdet(A) = 2 Σ log(L[i,i])
    g(v) = (A = [v[1]+5.0 1.0; 1.0 v[2]+5.0]; L = PureBLAS.potrf!(A; uplo = 'L'); 2*(log(L[1,1]) + log(L[2,2])))
    gr = ForwardDiff.gradient(g, [1.0, 2.0])
    @test all(isfinite, gr)
end

@testitem "gesvd (SVD) vs LAPACK — square/tall/wide, σ + reconstruction + orthogonality" begin
    using PureBLAS, LinearAlgebra, Random
    Random.seed!(20)
    maxe(A, B) = maximum(abs.(A .- B))
    @testset "$m×$n" for (m, n) in ((1, 1), (2, 2), (8, 8), (40, 25), (64, 64),
            (129, 96), (200, 200), (96, 160), (300, 257))
        A0 = randn(m, n)
        sref = svdvals(A0)
        # full factorization
        U, S, Vt = PureBLAS.gesvd!(copy(A0))
        @test maxe(S, sref) / maximum(sref) < 1e-11                  # singular values match LAPACK
        @test maxe(U * Diagonal(S) * Vt, A0) < 1e-10                 # A = U Σ Vᵀ
        @test maxe(U' * U, Matrix(I, size(U, 2), size(U, 2))) < 1e-10   # U orthonormal columns
        @test maxe(Vt * Vt', Matrix(I, size(Vt, 1), size(Vt, 1))) < 1e-10  # Vᵀ orthonormal rows
        # values-only path
        Sv = PureBLAS.gesvd!(copy(A0); want_vectors = false)[1]
        @test maxe(Sv, sref) / maximum(sref) < 1e-11
    end
end

@testitem "syev (symmetric eigen) vs LAPACK — eigenvalues + residual + orthonormality, both uplo + stress" begin
    using PureBLAS, LinearAlgebra, Random
    Random.seed!(21)
    # ‖A·V − V·Diag(w)‖ / (‖A‖·n·eps) and ‖V'V − I‖ / (n·eps): sidesteps eigenvector non-uniqueness.
    function checkpair(Afull, w, Z)
        n = size(Afull, 1); nA = opnorm(Afull); sc = max(nA, 1.0) * n * eps()
        resid = opnorm(Afull * Z - Z * Diagonal(w)) / sc
        orth  = opnorm(Z' * Z - I) / (n * eps())
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
        @test maximum(abs, w .- wref) / max(1.0, maximum(abs, wref)) < 1e-12   # eigenvalues match LAPACK
        @test issorted(w)                                                       # ascending
        resid, orth = checkpair(Afull, w, Z)
        @test resid < 32                                                        # residual ≲ 32·n·eps·‖A‖
        @test orth < 32                                                         # orthonormality ≲ 32·n·eps
        # jobz='N' — eigenvalues only (same values)
        wN, _, iN = PureBLAS._syev!('N', uplo, copy(Afull))
        @test iN == 0
        @test maximum(abs, wN .- wref) / max(1.0, maximum(abs, wref)) < 1e-12
    end
    @testset "stress: eps-clustered spectrum" begin
        n = 60
        Q = Matrix(qr(randn(n, n)).Q)                         # random orthogonal
        dvals = [1.0 + (k - 1) * eps() for k in 1:n]          # spacings = eps (deflation-tolerance stress)
        A = Symmetric(Q * Diagonal(dvals) * Q')
        Afull = Matrix(A)
        w, Z, info = PureBLAS._syev!('V', 'L', copy(Afull))
        @test info == 0
        @test maximum(abs, w .- eigvals(A)) < 1e-12
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
        for i in 1:n; A[i, i] = d[i]; end
        for i in 1:n-1
            i == m && continue                                # weak glue link between the two blocks
            A[i+1, i] = 1.0; A[i, i+1] = 1.0
        end
        A[m+1, m] = 1e-3; A[m, m+1] = 1e-3                    # faint coupling
        S = Symmetric(A, :L); Afull = Matrix(S)
        w, Z, info = PureBLAS._syev!('V', 'L', copy(Afull))
        @test info == 0
        @test maximum(abs, w .- eigvals(S)) < 1e-10
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
        orth  = opnorm(Z' * Z - I) / (n * eps(R))
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
        @test maximum(abs, Float64.(w) .- wref) / max(1.0, maximum(abs, wref)) < 1e-4
        resid, orth = checkpair(Afull, w, Z, Float32)
        @test resid < 64 && orth < 64
        wN, _, iN = PureBLAS._syev!('N', uplo, copy(Afull))            # values-only (sterf)
        @test iN == 0
        @test maximum(abs, Float64.(wN) .- wref) / max(1.0, maximum(abs, wref)) < 1e-4
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
        orth  = opnorm(Z' * Z - I) / (n * eps(R))
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
        vtol = R === Float64 ? 1e-11 : 1e-4
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
        tol = T === Float64 ? 1e-11 : 1e-4
        @test maximum(abs, Float64.(d) .- wref) / max(1.0, maximum(abs, wref)) < tol
    end
end
