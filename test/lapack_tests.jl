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
        verr  = maximum(abs, S .- sref) / max(maximum(sref), floatmin(R))
        return resid, orthU, orthV, verr
    end
    @testset "$T $m×$n" for T in (ComplexF64, ComplexF32),
            (m, n) in ((2, 2), (4, 4), (8, 8), (32, 32), (64, 64), (128, 128),
                       (8, 4), (4, 8), (64, 40), (40, 64), (129, 96), (96, 129))
        tol = T === ComplexF64 ? 64.0 : 512.0
        resid, orthU, orthV, verr = chk(randn(T, m, n))
        @test resid < tol
        @test orthU < tol
        @test orthV < tol
        @test verr < tol * eps(real(T))
    end
    @testset "$T scaled ‖A‖=$s" for T in (ComplexF64, ComplexF32), s in (1e-10, 1e-14, 1e8)
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
        @test maximum(abs, Sv .- svdvals(A)) / maximum(svdvals(A)) < 1e-11
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

@testitem "lq (LQ) vs LAPACK — gelqf L·Q reconstruct, orglq orthonormal rows, ormlq apply" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.LAPACK
    maxe(x, y) = maximum(abs.(x .- y)) / max(maximum(abs.(y)), 1e-300)
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

@testitem "bunchkaufman (sytrf/hetrf + solve) vs LAPACK — factor solve residual, both uplo" begin
    using PureBLAS, LinearAlgebra
    tol(::Type{T}) where {T} = sqrt(eps(real(T))) * 100
    @testset "$T n=$n uplo=$uplo herm=$herm" for T in (Float32, Float64, ComplexF32, ComplexF64),
        n in (1, 2, 5, 17, 64), uplo in ('L', 'U'), herm in (false, true)

        (herm && !(T <: Complex)) && continue           # herm==sym for real; skip dup
        M = randn(T, n, n)
        A = herm ? (M + M') : (M + transpose(M))        # Hermitian / symmetric indefinite
        ipiv = zeros(Int, n)
        LD = copy(A)
        herm ? PureBLAS.hetrf!(LD, ipiv; uplo = uplo) : PureBLAS.sytrf!(LD, ipiv; uplo = uplo)
        B = randn(T, n, 3); X = copy(B)
        herm ? PureBLAS.hetrs!(LD, ipiv, X; uplo = uplo) : PureBLAS.sytrs!(LD, ipiv, X; uplo = uplo)
        @test norm(A * X - B) <= tol(T) * (norm(A) * norm(X) + norm(B))
    end
end

@testitem "geqp3 (pivoted QR) vs LAPACK — A·P = Q·R reconstruct, |R| non-increasing" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.LAPACK
    maxe(x, y) = maximum(abs.(x .- y)) / max(maximum(abs.(y)), 1e-300)
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
            v = zeros(T, m); v[i] = one(T); v[i+1:m] = F[i+1:m, i]
            Q = Q - (tau[i] * Q * v) * v'
        end
        @test maxe(Q * R, A0[:, jpvt]) < 300 * eps(real(T))
        rdiag = [abs(R[i, i]) for i in 1:k]
        @test issorted(rdiag; rev = true) || maximum(diff(rdiag)) < 1e-6 * rdiag[1]  # non-increasing
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
            @test isapprox(rc, rc_ref; rtol = 1e-3)
        end
        # trcon: triangular condition vs LAPACK.trcon!.
        U = triu(A)
        rct = PureBLAS.trcon!(copy(U); uplo = 'U', diag = 'N', norm = '1')
        rct_ref = LAPACK.trcon!('1', 'U', 'N', copy(U))   # Julia's LAPACK wrapper wants '1'/'I' (not 'O')
        @test isapprox(rct, rct_ref; rtol = 1e-3)
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
    maxe(x, y) = maximum(abs.(x .- y)) / max(maximum(abs.(y)), 1e-300)
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
    evmatch(a, b) = (a = sort(collect(a); by = x -> (real(x), imag(x)));
                     b = sort(collect(b); by = x -> (real(x), imag(x)));
                     maximum(abs.(a .- b)) / max(maximum(abs.(b)), 1))
    # reconstruct the complex eigenvector matrix from real-packed VR (LAPACK conj-pair convention)
    function evecs_real(wr, wi, VR)
        n = length(wr); E = zeros(ComplexF64, n, n); j = 1
        while j <= n
            if wi[j] == 0
                E[:, j] = VR[:, j]
            else
                for i in 1:n
                    E[i, j] = VR[i, j] + im * VR[i, j+1]; E[i, j+1] = VR[i, j] - im * VR[i, j+1]
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
        @test evmatch(λ, eigvals(copy(A))) < 1e-8            # vs LinearAlgebra (OpenBLAS)
        wr2, wi2 = PureBLAS.geev!('N', 'N', copy(A))         # values-only path
        @test evmatch(complex.(wr2, wi2), λ) < 1e-10
    end
    @testset "conj-pair real n=$n" for n in (4, 8, 16)       # block rotations → guaranteed complex pairs
        B = zeros(n, n); i = 1
        while i + 1 <= n
            θ = randn(); a = randn()
            B[i, i] = a; B[i+1, i+1] = a; B[i, i+1] = θ; B[i+1, i] = -θ; i += 2
        end
        Q, _ = qr(randn(n, n)); A = Matrix(Q) * B * Matrix(Q)'
        wr, wi, VL, VR = PureBLAS.geev!('N', 'V', copy(A))
        @test count(!iszero, wi) > 0                          # actually has conjugate pairs
        E = evecs_real(wr, wi, VR)
        @test maximum(abs.(A * E - E * Diagonal(complex.(wr, wi)))) / (opnorm(A, 1) * n * eps()) < 100
        @test evmatch(complex.(wr, wi), eigvals(copy(A))) < 1e-8
    end
    @testset "complex n=$n" for n in (2, 4, 8, 16, 32)
        A = randn(ComplexF64, n, n)
        w, VL, VR = PureBLAS.geev!('N', 'V', copy(A))
        @test maximum(abs.(A * VR - VR * Diagonal(w))) / (opnorm(A, 1) * n * eps()) < 100
        @test evmatch(w, eigvals(copy(A))) < 1e-8
        w2, = PureBLAS.geev!('N', 'N', copy(A))
        @test evmatch(w2, w) < 1e-10
    end
    @test_throws ArgumentError PureBLAS.geev!('V', 'N', randn(4, 4))   # left vectors not implemented
end

@testitem "gees (Schur) vs LAPACK — Z·T·Zᴴ=A reconstruction, Z orthonormal, real+complex" begin
    using PureBLAS, LinearAlgebra, Random
    Random.seed!(0x5C)
    evmatch(a, b) = (a = sort(collect(a); by = x -> (real(x), imag(x)));
                     b = sort(collect(b); by = x -> (real(x), imag(x)));
                     maximum(abs.(a .- b)) / max(maximum(abs.(b)), 1))
    @testset "real n=$n" for n in (2, 4, 8, 16, 32)
        A = randn(n, n)
        T, Z, w = PureBLAS.gees!('V', copy(A))
        @test maximum(abs.(Z * T * Z' - A)) / max(opnorm(A, 1), 1) < 1e-11    # A = Z·T·Zᵀ
        @test maximum(abs.(Z' * Z - I)) < 1e-12                                # Z orthonormal
        @test evmatch(w, eigvals(copy(A))) < 1e-8
        @test all(iszero, [T[i, j] for j in 1:n for i in j+2:n])              # quasi-upper-triangular
    end
    @testset "conj-pair real n=$n" for n in (4, 8, 16)
        B = zeros(n, n); i = 1
        while i + 1 <= n
            θ = randn(); a = randn(); B[i, i] = a; B[i+1, i+1] = a; B[i, i+1] = θ; B[i+1, i] = -θ; i += 2
        end
        Q, _ = qr(randn(n, n)); A = Matrix(Q) * B * Matrix(Q)'
        T, Z, w = PureBLAS.gees!('V', copy(A))
        @test maximum(abs.(Z * T * Z' - A)) / max(opnorm(A, 1), 1) < 1e-11
        @test maximum(abs.(Z' * Z - I)) < 1e-12
    end
    @testset "complex n=$n" for n in (2, 4, 8, 16, 32)
        A = randn(ComplexF64, n, n)
        T, Z, w = PureBLAS.gees!('V', copy(A))
        @test maximum(abs.(Z * T * Z' - A)) / max(opnorm(A, 1), 1) < 1e-11
        @test maximum(abs.(Z' * Z - I)) < 1e-12
        @test evmatch(w, eigvals(copy(A))) < 1e-8
        @test all(iszero, [T[i, j] for j in 1:n for i in j+1:n])              # upper-triangular (complex)
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
    packR(vr, ai) = (n = size(vr, 1); VC = zeros(ComplexF64, n, n); j = 1;
        while j <= n
            if iszero(ai[j]); VC[:, j] = vr[:, j]; j += 1
            else VC[:, j] = vr[:, j] .+ im .* vr[:, j+1]; VC[:, j+1] = vr[:, j] .- im .* vr[:, j+1]; j += 2 end
        end; VC)
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
        @test resid(A, B, complex.(ar, ai), be, VC) < 1e-11
        ar2, ai2, be2, _, _ = LA.ggev3!('N', 'V', copy(A), copy(B))
        @test evmatch(complex.(ar, ai), complex.(be), complex.(ar2, ai2), complex.(be2)) < 1e-10
    end
    @testset "real guaranteed conj-pairs n=$n" for n in (8, 20)
        Bp = zeros(n, n); i = 1
        while i + 1 <= n
            Bp[i, i] = randn(); Bp[i+1, i+1] = Bp[i, i]; Bp[i, i+1] = randn(); Bp[i+1, i] = -Bp[i, i+1]; i += 2
        end
        Qr, _ = qr(randn(n, n)); A = Matrix(Qr) * Bp * Matrix(Qr)'
        M = randn(n, n); B = M'M + I                        # SPD, well conditioned
        ar, ai, be, _, vr = PureBLAS.ggev!('N', 'V', copy(A), copy(B))
        @test count(!iszero, ai) > 0                        # actually produced conjugate pairs
        @test resid(A, B, complex.(ar, ai), be, packR(vr, ai)) < 1e-11
    end
    @testset "real infinite eigenvalue (singular B)" begin
        n = 12
        A = randn(n, n); B = randn(n, n); B[:, n] .= 0; B[n, :] .= 0    # rank-deficient B → infinite eig
        ar, ai, be, _, vr = PureBLAS.ggev!('N', 'V', copy(A), copy(B))
        @test count(x -> abs(x) < 1e-10, be) >= 1           # at least one infinite eigenvalue (β≈0)
        @test resid(A, B, complex.(ar, ai), be, packR(vr, ai)) < 1e-9
    end
    @testset "complex n=$n" for n in (6, 20, 40)
        A = randn(ComplexF64, n, n); B = randn(ComplexF64, n, n)
        al, be, vl, vr = PureBLAS.ggev!('N', 'V', copy(A), copy(B))
        @test size(vl, 2) == 0
        @test resid(A, B, al, be, vr) < 1e-11
        al2, be2, _, _ = LA.ggev3!('N', 'V', copy(A), copy(B))
        @test evmatch(al, be, al2, be2) < 1e-10
    end
    @testset "eigenvalues-only path (jobvr='N') n=$n" for n in (10, 24)
        A = randn(n, n); B = randn(n, n)
        ar, ai, be, _, vr = PureBLAS.ggev!('N', 'N', copy(A), copy(B))
        @test size(vr, 2) == 0
        ar2, ai2, be2, _, _ = LA.ggev3!('N', 'N', copy(A), copy(B))
        @test evmatch(complex.(ar, ai), complex.(be), complex.(ar2, ai2), complex.(be2)) < 1e-10
    end
    @test_throws ArgumentError PureBLAS.ggev!('V', 'V', randn(4, 4), randn(4, 4))   # left vectors unsupported
end

@testitem "gges (generalized Schur) vs LAPACK — A=Q·S·Zᴴ, B=Q·P·Zᴴ, Q/Z orthonormal, real+complex" begin
    using PureBLAS, LinearAlgebra, Random
    Random.seed!(0x9a55)
    @testset "$T n=$n" for T in (Float64, ComplexF64), n in (5, 16, 33)
        A = randn(T, n, n); B = randn(T, n, n)
        S, P, al, be, Q, Z = PureBLAS.gges!('V', 'V', copy(A), copy(B))
        tol = 1e-11 * (opnorm(A, 1) + opnorm(B, 1) + 1)
        @test maximum(abs, Q * S * Z' - A) < tol
        @test maximum(abs, Q * P * Z' - B) < tol
        @test maximum(abs, Q' * Q - I) < 1e-12
        @test maximum(abs, Z' * Z - I) < 1e-12
        @test length(al) == n && length(be) == n
    end
    @testset "singular B (infinite eig) n=12" begin
        n = 12; A = randn(n, n); B = randn(n, n); B[:, n] .= 0; B[n, :] .= 0
        S, P, al, be, Q, Z = PureBLAS.gges!('V', 'V', copy(A), copy(B))
        @test maximum(abs, Q * S * Z' - A) < 1e-10 * (opnorm(A, 1) + 1)
        @test maximum(abs, Q * P * Z' - B) < 1e-12
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
        @test maximum(abs, w .- wref) < 1e-9 * (norm(w) + 1)
        # verify the defining relation per itype
        if it == 1                                         # A z = λ B z
            R = A * Z - B * Z * Diagonal(w)
        elseif it == 2                                     # A B z = λ z
            R = A * (B * Z) - Z * Diagonal(w)
        else                                               # itype 3: B A z = λ z
            R = B * (A * Z) - Z * Diagonal(w)
        end
        @test maximum(abs, R) < 1e-8 * (opnorm(A, 1) * opnorm(B, 1) + 1)
        wN = gvd!(it, 'N', ul, copy(A), copy(B))[1]        # values-only path
        @test maximum(abs, wN .- wref) < 1e-9 * (norm(w) + 1)
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
        dl2, d2, du2, du22, ipiv = PureBLAS.gttrf!(copy(dl), copy(d), copy(du), Vector{T}(undef, max(n - 2, 0)),
            Vector{Int}(undef, n))
        for (tr, Aop) in (('N', A), ('T', transpose(A)), ('C', A'))
            B = randn(T, n, 2)
            X = PureBLAS.gttrs!(tr, dl2, d2, du2, du22, ipiv, copy(B))
            @test maximum(abs, Aop * X - B) < sqrt(eps(real(T))) * 100 * (norm(A) + 1)
        end
    end
    @test_throws LinearAlgebra.SingularException PureBLAS.gtsv!([0.0], [0.0, 0.0], [1.0], [1.0, 1.0])  # [[0 1];[0 0]] singular
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

@testitem "gbtrf/gbtrs (general banded LU) vs LAPACK — band storage, all four types, trans variants" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.LAPACK as LA
    @testset "$T n=$n kl=$kl ku=$ku" for T in (Float32, Float64, ComplexF32, ComplexF64),
        (n, kl, ku) in ((8, 1, 1), (20, 2, 3), (35, 3, 1))

        A = zeros(T, n, n)
        for j in 1:n, i in max(1, j - ku):min(n, j + kl)
            A[i, j] = randn(T)
        end
        A += n * I     # diagonally dominant-ish, keep nonsingular
        ldab = 2kl + ku + 1
        AB = zeros(T, ldab, n)
        for j in 1:n, i in max(1, j - ku):min(n, j + kl)
            AB[kl + ku + 1 + i - j, j] = A[i, j]
        end
        ABp = copy(AB)
        _, ipiv, info = PureBLAS.gbtrf!(kl, ku, n, ABp)
        @test info == 0
        ABl = copy(AB); _, ipivl = LA.gbtrf!(kl, ku, n, ABl)
        @test ipiv == ipivl
        @test maximum(abs, ABp .- ABl) < 200 * eps(real(T)) * maximum(abs, ABl)
        for tr in ('N', 'T', 'C')
            Bv = randn(T, n, 2)
            Bp = copy(Bv)
            PureBLAS.gbtrs!(tr, kl, ku, n, ABp, ipiv, Bp)
            Aop = tr == 'N' ? A : (tr == 'T' ? transpose(A) : A')
            @test norm(Aop * Bp - Bv) < sqrt(eps(real(T))) * 200 * (norm(A) + 1)
        end
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
        for i in 1:n; A[i, i] = d[i]; end
        for i in 1:n-1; A[i + 1, i] = e[i]; A[i, i + 1] = conj(e[i]); end
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
        @test maximum(abs, s .- sref) / maximum(sref) < 1e-3
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
        _, rk, _ = PureBLAS.gelsd!(copy(A), copy(Bv), 1e-8)
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
        @test maximum(abs, Lr * Lr' .- Aperm) < 1e-6 * maximum(abs, A)
    end
end

@testitem "QL/RQ (geqlf/gerqf/orgql/orgrq/ormql/ormrq) vs LAPACK — reconstruction + orthonormal Q + apply" begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.LAPACK as LA
    maxe(x, y) = maximum(abs.(x .- y)) / max(maximum(abs.(y)), 1e-300)
    @testset "QL $T $m×$n" for T in (Float32, Float64, ComplexF32, ComplexF64), (m, n) in ((8, 5), (8, 8), (30, 18))
        A0 = randn(T, m, n); k = min(m, n)
        F = copy(A0); tau = zeros(T, k); PureBLAS.geqlf!(F, tau)
        Fl = copy(A0); taul = zeros(T, k); LA.geqlf!(Fl, taul)     # LAPACK reference, SAME τ convention
        @test maxe(tril(F[m-k+1:m, n-k+1:n]), tril(Fl[m-k+1:m, n-k+1:n])) < 400 * eps(real(T))
        @test maxe(tau, taul) < 400 * eps(real(T))
        Q = PureBLAS.orgql!(copy(F), copy(tau))
        @test maxe(Q' * Q, Matrix{T}(I, n, n)) < 800 * eps(real(T))
        Lql = tril(F[m-n+1:m, :])          # economy L: n×n lower-tri bottom block, A = Q(m×n)·L(n×n)
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
        @test maxe(triu(F[:, n-m+1:n]), triu(Fl[:, n-m+1:n])) < 400 * eps(real(T))
        @test maxe(tau, taul) < 400 * eps(real(T))
        Q = PureBLAS.orgrq!(copy(F), copy(tau))
        @test maxe(Q * Q', Matrix{T}(I, m, m)) < 800 * eps(real(T))
        Rrq = triu(F[:, n-m+1:n])          # economy R: m×m upper-tri right block, A = R(m×m)·Q(m×n)
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
        @test maximum(abs, Qm * Tm * Qm' - A0) < 1e-9 * (opnorm(A0, 1) + 1)
        @test maximum(abs, Qm' * Qm - I) < 1e-9
    end
    @testset "trsen $T n=$n" for T in (Float64, ComplexF64), n in (6, 14)
        A0 = randn(T, n, n)
        S = schur(A0)
        Torig = Matrix(S.T); Q0 = Matrix(S.Z)
        Tm = copy(Torig); Qm = copy(Q0)
        sel = falses(n); sel[1] = true               # select the leading eigenvalue('s conjugate-pair block)
        Tr, Qr, w, s, sep = PureBLAS.trsen!('B', 'V', sel, Tm, Qm)
        @test maximum(abs, Qr * Tr * Qr' - A0) < 1e-8 * (opnorm(A0, 1) + 1)
        @test maximum(abs, Qr' * Qr - I) < 1e-8
        @test 0 < s <= 1 + 1e-8
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
        for i in 1:k; D1[i, i] = 1; end
        if m - kl >= 0
            for i in 1:l; D1[k + i, k + i] = alpha[k + i]; D2[i, k + i] = beta[k + i]; end
        else
            for i in (k + 1):m; D1[i, i] = alpha[i]; end
            for i in 1:(m - k); D2[i, k + i] = beta[k + i]; end
            for i in (m - k + 1):l; D2[i, k + i] = 1; end
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
        @test sort(al) ≈ sort(alo) atol = (RT === Float64 ? 1e-9 : 1e-4)
        @test sort(be) ≈ sort(beo) atol = (RT === Float64 ? 1e-9 : 1e-4)
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
