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
