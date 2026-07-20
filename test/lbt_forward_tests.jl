# In-process LBT forwarding dogfood: PureBLAS.activate() must reroute LinearAlgebra's BLAS/LAPACK to
# PureBLAS's native kernels (via lbt_set_forward @cfunction pointers — cabi_forward.jl), inside a live
# Julia process, with correct results; deactivate() must restore OpenBLAS. Regression guard for the
# cabi_forward.jl @cfunction signatures (a mismatch → wrong result) AND for the "only self-consistent
# LAPACK symbols forwarded" invariant (a forwarded factorization whose companion stays on OpenBLAS must
# still be correct — e.g. geqrf is deliberately NOT forwarded: its faer-τ breaks OpenBLAS orgqr).

@testitem "LBT in-process forward: activate reroutes BLAS/LAPACK to PureBLAS" tags = [:lbt] begin
    using PureBLAS, LinearAlgebra, Random
    import LinearAlgebra.BLAS as B
    import LinearAlgebra.LAPACK as LA
    Random.seed!(0xBEEF)
    n = 96
    A = randn(n, n); Bm = randn(n, n); x = randn(n); y = randn(n); b = randn(n)
    SPD = A * A' + n * I
    At = A + n * I                # well-conditioned source for triangular solves (UpperTriangular(A) is near-singular)
    Az = randn(ComplexF64, n, n); zb = randn(ComplexF64, n); HPD = Az * Az' + n * I

    # OpenBLAS oracles, captured BEFORE activate (independent reference).
    ref = (
        mul = A * Bm, gemv = A * x, gemvt = transpose(A) * x, dot = dot(x, y), nrm2 = norm(x),
        symm = Symmetric(A) * Bm, trsm = UpperTriangular(At) \ Bm,
        chol = Matrix(cholesky(copy(SPD)).U), lusol = lu(copy(A)) \ b, chsol = cholesky(copy(SPD)) \ b,
        atsol = transpose(A) \ b, trsol = UpperTriangular(At) \ b,
        ichol = inv(cholesky(copy(SPD))), itri = inv(UpperTriangular(At)),
        qrsol = qr(copy(A)) \ b, qrdet = det(qr(copy(A)).Q),
        svals = svdvals!(copy(A)),   # svdvals! → gesdd (stays OpenBLAS); here just a numeric oracle
        zchol = Matrix(cholesky(Hermitian(copy(HPD))).U), zlusol = lu(copy(Az)) \ zb,
        zqrsol = qr(copy(Az)) \ zb,
        evals = eigvals(Symmetric(A + A')),         # symmetric eigen values (uplo='U' default)
        zevals = eigvals(Hermitian(Az + Az')),
    )      # Hermitian complex eigen values (uplo='U' default)
    Asym = Symmetric(A + A'); Asymf = Matrix(Asym)     # dense symmetric for the eigen residual check
    Aherm = Hermitian(Az + Az'); Ahermf = Matrix(Aherm)  # dense Hermitian for the complex eigen residual check

    @test length(PureBLAS._LBT_REGISTRARS) > 100
    fwd(s) = B.lbt_get_forward(s, Int32(B.LBT_INTERFACE_ILP64), Int32(B.LBT_F2C_PLAIN))
    gemm_before = fwd("dgemm_"); geqrf_before = fwd("dgeqrf_")
    zpotrf_before = fwd("zpotrf_"); zgetrf_before = fwd("zgetrf_"); gesdd_before = fwd("dgesdd_")
    zgesdd_before = fwd("zgesdd_")
    getrs_before = fwd("dgetrs_"); potrs_before = fwd("dpotrs_"); trtrs_before = fwd("dtrtrs_")
    getri_before = fwd("dgetri_"); potri_before = fwd("dpotri_"); trtri_before = fwd("dtrtri_")
    geqrt_before = fwd("dgeqrt_"); gemqrt_before = fwd("dgemqrt_")
    zgeqrt_before = fwd("zgeqrt_"); zgemqrt_before = fwd("zgemqrt_")
    sgetrf_before = fwd("sgetrf_"); sgeqrt_before = fwd("sgeqrt_"); sgesdd_before = fwd("sgesdd_")
    syevr_before = fwd("dsyevr_"); zheevr_before = fwd("zheevr_")
    Af = randn(Float32, n, n); f32svals = svdvals(copy(Af))
    PureBLAS.activate()
    try
        # BLAS — all route to PureBLAS.
        @test fwd("dgemm_") != gemm_before                         # actually forwarded (pointer changed)
        @test maximum(abs, (A * Bm) .- ref.mul) < 1.0e-9             # gemm
        @test maximum(abs, (A * x) .- ref.gemv) < 1.0e-10           # gemv N
        @test maximum(abs, (transpose(A) * x) .- ref.gemvt) < 1.0e-10 # gemv T
        @test abs(dot(x, y) - ref.dot) < 1.0e-11                     # dot
        @test abs(norm(x) - ref.nrm2) < 1.0e-12                      # nrm2
        @test maximum(abs, (Symmetric(A) * Bm) .- ref.symm) < 1.0e-9 # symm
        @test maximum(abs, (UpperTriangular(At) \ Bm) .- ref.trsm) < 1.0e-9  # trsm

        # LAPACK that routes to PureBLAS (self-consistent under a mixed backend).
        C = cholesky(copy(SPD)); @test maximum(abs, Matrix(C.U) .- ref.chol) < 1.0e-8   # potrf
        F = lu(copy(A))                                            # getrf (LAPACK-convention ipiv/factors)
        @test maximum(abs, (F.L * F.U) .- A[F.p, :]) < 1.0e-9
        # Solves now route to PureBLAS too (getrs/potrs/trtrs) — the solve step of `\`.
        @test fwd("dgetrs_") != getrs_before && fwd("dpotrs_") != potrs_before && fwd("dtrtrs_") != trtrs_before
        @test maximum(abs, (lu(copy(A)) \ b) .- ref.lusol) < 1.0e-9          # getrf + getrs, both PureBLAS
        @test maximum(abs, (transpose(A) \ b) .- ref.atsol) < 1.0e-9         # getrs trans='T' (reverse pivots)
        @test maximum(abs, (cholesky(copy(SPD)) \ b) .- ref.chsol) < 1.0e-9  # potrf + potrs
        @test maximum(abs, (UpperTriangular(At) \ b) .- ref.trsol) < 1.0e-9   # trtrs
        # Inverses now route (getri/potri/trtri) — inv() uses PureBLAS.
        @test fwd("dgetri_") != getri_before && fwd("dpotri_") != potri_before && fwd("dtrtri_") != trtri_before
        @test maximum(abs, inv(copy(A)) * A - I) < 1.0e-8                     # getri
        @test maximum(abs, inv(cholesky(copy(SPD))) - ref.ichol) < 1.0e-9    # potri
        @test maximum(abs, inv(UpperTriangular(At)) - ref.itri) < 1.0e-9     # trtri
        # gesvd routes (direct LAPACK call).
        S = LA.gesvd!('N', 'N', copy(A))[2]
        @test maximum(abs, S .- ref.svals) < 1.0e-9
        # gesdd routes → svd()/svdvals (Julia's default SVD path) now use PureBLAS.
        @test fwd("dgesdd_") != gesdd_before
        @test maximum(abs, svdvals(copy(A)) .- ref.svals) < 1.0e-9
        Fs = svd(copy(A)); @test maximum(abs, Fs.U * Diagonal(Fs.S) * Fs.Vt .- A) < 1.0e-9

        # Complex LAPACK — potrf/getrf route (self-consistent: standard-convention factors).
        @test fwd("zpotrf_") != zpotrf_before
        @test fwd("zgetrf_") != zgetrf_before
        Cz = cholesky(Hermitian(copy(HPD)))
        @test maximum(abs, Matrix(Cz.U) .- ref.zchol) < 1.0e-8               # zpotrf
        Fz = lu(copy(Az))
        @test maximum(abs, (Fz.L * Fz.U) .- Az[Fz.p, :]) < 1.0e-9            # zgetrf (valid factorization)
        @test maximum(abs, (lu(copy(Az)) \ zb) .- ref.zlusol) < 1.0e-8       # zgetrf + zgetrs, both PureBLAS
        @test fwd("zgeqrt_") != zgeqrt_before && fwd("zgemqrt_") != zgemqrt_before   # complex QR routes
        Qz = qr(copy(Az))
        @test maximum(abs, Matrix(Qz.Q)' * Matrix(Qz.Q) - I) < 1.0e-9        # zgemqrt: Q unitary
        @test maximum(abs, Matrix(Qz.Q) * Qz.R - Az) < 1.0e-9                # zgeqrt+zgemqrt: Q·R = Az
        @test maximum(abs, (qr(copy(Az)) \ zb) .- ref.zqrsol) < 1.0e-8       # complex qr()\b

        # Complex SVD routes → svd()/svdvals(::Matrix{ComplexF64}) now use PureBLAS (zgesdd).
        @test fwd("zgesdd_") != zgesdd_before
        @test maximum(abs, svdvals(copy(Az)) .- svdvals(Az)) < 1.0e-9         # zgesdd values
        Fz = svd(copy(Az))                                                  # zgesdd U/S/Vᴴ
        @test maximum(abs, Fz.U * Diagonal(Fz.S) * Fz.Vt .- Az) < 1.0e-9      # reconstruction
        @test maximum(abs, Fz.U' * Fz.U - I) < 1.0e-9                         # U unitary columns
        @test maximum(abs, Fz.Vt * Fz.Vt' - I) < 1.0e-9                       # Vᴴ unitary rows

        # Float32 LAPACK via mixed precision (compute F64, store F32) — lu/qr/svd route, correct to F32.
        @test fwd("sgetrf_") != sgetrf_before && fwd("sgeqrt_") != sgeqrt_before && fwd("sgesdd_") != sgesdd_before
        Ff = lu(copy(Af)); @test maximum(abs, (Ff.L * Ff.U) .- Af[Ff.p, :]) < 1.0f-4   # sgetrf
        Qf = qr(copy(Af))
        @test maximum(abs, Matrix(Qf.Q)' * Matrix(Qf.Q) - I) < 1.0f-4          # sgeqrt+sgemqrt
        @test maximum(abs, Matrix(Qf.Q) * Qf.R - Af) < 1.0f-4
        @test maximum(abs, svdvals(copy(Af)) .- f32svals) < 1.0f-3             # sgesdd (loose F32 tol)

        # QR routes to PureBLAS via geqrt+gemqrt (Julia's qr() is QRCompactWY). geqrf is now forwarded too
        # (its wrapper converts faer τ → LAPACK, so geqrf!+OpenBLAS-orgqr is correct — no more NaN-Q hazard).
        @test fwd("dgeqrt_") != geqrt_before && fwd("dgemqrt_") != gemqrt_before && fwd("dgeqrf_") != geqrf_before
        Q = qr(copy(A))
        @test maximum(abs, Matrix(Q.Q)' * Matrix(Q.Q) - I) < 1.0e-9   # gemqrt: Q orthonormal
        @test maximum(abs, Matrix(Q.Q) * Q.R - A) < 1.0e-9            # geqrt+gemqrt: Q·R = A
        @test maximum(abs, (qr(copy(A)) \ b) .- ref.qrsol) < 1.0e-8   # geqrt+gemqrt+trtrs
        @test abs(det(qr(copy(A)).Q) - ref.qrdet) < 1.0e-9            # det(Q) reads T's diagonal (LAPACK-exact T)
        # geqrf τ-conversion: geqrf!+OpenBLAS orgqr now gives a valid Q (faer never crosses the ABI).
        A2 = copy(A); _, tau = LA.geqrf!(A2); Qo = LA.orgqr!(copy(A2), tau)
        @test maximum(abs, Qo'Qo - I) < 1.0e-9

        # Symmetric eigensolver routes: eigen(Symmetric)/eigvals(Symmetric) → dsyevr_ (Julia's default).
        @test fwd("dsyevr_") != syevr_before                              # actually forwarded
        Fe = eigen(Asym)                                                  # dsyevr_ jobz='V' range='A'
        n_ = size(Asymf, 1)
        @test maximum(abs, Fe.values .- ref.evals) < 1.0e-9                 # eigenvalues match OpenBLAS
        @test opnorm(Asymf * Fe.vectors - Fe.vectors * Diagonal(Fe.values)) < 1.0e-8 * opnorm(Asymf)
        @test opnorm(Fe.vectors' * Fe.vectors - I) < 1.0e-9                 # orthonormal vectors
        @test maximum(abs, eigvals(Asym) .- ref.evals) < 1.0e-9            # eigvals(Symmetric) routes too

        # Hermitian complex eigensolver routes: eigen(Hermitian)/eigvals(Hermitian) → zheevr_ (default).
        @test fwd("zheevr_") != zheevr_before                             # actually forwarded
        Fze = eigen(Aherm)                                               # zheevr_ jobz='V' range='A'
        @test maximum(abs, Fze.values .- ref.zevals) < 1.0e-9              # eigenvalues match OpenBLAS
        @test opnorm(Ahermf * Fze.vectors - Fze.vectors * Diagonal(Fze.values)) < 1.0e-8 * opnorm(Ahermf)
        @test opnorm(Fze.vectors' * Fze.vectors - I) < 1.0e-9             # orthonormal (unitary) vectors
        @test maximum(abs, eigvals(Aherm) .- ref.zevals) < 1.0e-9        # eigvals(Hermitian) routes too
    finally
        PureBLAS.deactivate()   # ALWAYS restore OpenBLAS so later testitems keep their oracle
    end
    @test A * Bm == ref.mul     # after deactivate the OpenBLAS oracle is back (bit-identical)
end

@testitem "LBT in-process forward: batch-6 LAPACK (lq/bunchkaufman/qr-pivot/cond/hessenberg)" tags = [:lbt] begin
    using PureBLAS, LinearAlgebra, Random
    import LinearAlgebra.BLAS as B
    import LinearAlgebra.LAPACK as LA
    Random.seed!(0xF00D)
    n = 64
    A = randn(n, n); Az = randn(ComplexF64, n, n)
    SI = A + transpose(A); HI = Az + Az'          # symmetric / Hermitian indefinite
    U = triu(A) + n * I                            # well-conditioned upper-triangular
    tall = randn(90, n); b = randn(90)

    # OpenBLAS oracles captured BEFORE activate.
    ref = (
        lqrec = (F = lq(copy(A)); Matrix(F.L) * Matrix(F.Q)),
        bksol = bunchkaufman(Symmetric(copy(SI))) \ ones(n),
        zbksol = bunchkaufman(Hermitian(copy(HI))) \ ones(ComplexF64, n),
        qpp = (Q = qr(copy(A), ColumnNorm()); Matrix(Q.Q) * Matrix(Q.R) - A[:, Q.p]),
        tallsol = tall \ b,
        trcond = cond(UpperTriangular(copy(U)), 1),
        hess = (H = hessenberg(copy(A)); Matrix(H.Q) * Matrix(H.H) * Matrix(H.Q)' - A),
    )

    fwd(s) = B.lbt_get_forward(s, Int32(B.LBT_INTERFACE_ILP64), Int32(B.LBT_F2C_PLAIN))
    gelqf_b = fwd("dgelqf_"); orglq_b = fwd("dorglq_"); sytrf_b = fwd("dsytrf_")
    zhetrf_b = fwd("zhetrf_"); geqp3_b = fwd("dgeqp3_"); trcon_b = fwd("dtrcon_"); gehrd_b = fwd("dgehrd_")
    PureBLAS.activate()
    try
        # Pointers actually changed (forwarded to PureBLAS).
        @test fwd("dgelqf_") != gelqf_b && fwd("dorglq_") != orglq_b
        @test fwd("dsytrf_") != sytrf_b && fwd("zhetrf_") != zhetrf_b
        @test fwd("dgeqp3_") != geqp3_b && fwd("dtrcon_") != trcon_b && fwd("dgehrd_") != gehrd_b
        # Correct under the mixed backend.
        F = lq(copy(A)); @test maximum(abs, (Matrix(F.L) * Matrix(F.Q)) .- ref.lqrec) < 1.0e-9
        @test maximum(abs, (bunchkaufman(Symmetric(copy(SI))) \ ones(n)) .- ref.bksol) < 1.0e-8
        @test maximum(abs, (bunchkaufman(Hermitian(copy(HI))) \ ones(ComplexF64, n)) .- ref.zbksol) < 1.0e-8
        Q = qr(copy(A), ColumnNorm()); @test maximum(abs, (Matrix(Q.Q) * Matrix(Q.R) - A[:, Q.p]) .- ref.qpp) < 1.0e-9
        @test maximum(abs, (tall \ b) .- ref.tallsol) < 1.0e-8       # non-square \ (geqp3 path)
        @test isapprox(cond(UpperTriangular(copy(U)), 1), ref.trcond; rtol = 1.0e-6)   # trcon
        Hh = hessenberg(copy(A))
        @test maximum(abs, (Matrix(Hh.Q) * Matrix(Hh.H) * Matrix(Hh.Q)' - A) .- ref.hess) < 1.0e-8
    finally
        PureBLAS.deactivate()
    end
end

@testitem "LBT in-process forward: general eigensolver (eigen/eigvals/schur → geev/geevx/gees)" tags = [:lbt] begin
    using PureBLAS, LinearAlgebra, Random
    import LinearAlgebra.BLAS as B
    Random.seed!(0xBEEF)
    n = 48
    A = randn(n, n); Az = randn(ComplexF64, n, n)     # general NON-symmetric / non-Hermitian
    # A matrix guaranteed to have complex-conjugate eigenvalue pairs (real-packed VR path).
    Bp = zeros(n, n); i = 1
    while i + 1 <= n
        Bp[i, i] = randn(); Bp[i + 1, i + 1] = Bp[i, i]; Bp[i, i + 1] = randn(); Bp[i + 1, i] = -Bp[i, i + 1]; i += 2
    end
    Qr, _ = qr(randn(n, n)); Acp = Matrix(Qr) * Bp * Matrix(Qr)'

    evsort(v) = sort(v; by = x -> (real(x), imag(x)))
    # OpenBLAS oracles captured BEFORE activate.
    ref = (
        ev = evsort(eigvals(copy(A))), zev = evsort(eigvals(copy(Az))),
        cpev = evsort(eigvals(copy(Acp))),
        eF = eigen(copy(A)), zeF = eigen(copy(Az)),
        sch = (F = schur(copy(A)); Matrix(F.Z) * Matrix(F.T) * Matrix(F.Z)' - A),
    )

    fwd(s) = B.lbt_get_forward(s, Int32(B.LBT_INTERFACE_ILP64), Int32(B.LBT_F2C_PLAIN))
    geev_b = fwd("dgeev_"); geevx_b = fwd("dgeevx_"); zgeev_b = fwd("zgeev_")
    zgeevx_b = fwd("zgeevx_"); gees_b = fwd("dgees_"); zgees_b = fwd("zgees_")
    PureBLAS.activate()
    try
        # Pointers actually changed (forwarded to PureBLAS). eigen/eigvals route via geevx; schur via gees.
        @test fwd("dgeev_") != geev_b && fwd("dgeevx_") != geevx_b
        @test fwd("zgeev_") != zgeev_b && fwd("zgeevx_") != zgeevx_b
        @test fwd("dgees_") != gees_b && fwd("zgees_") != zgees_b
        # eigvals / eigen correct under PureBLAS.
        @test maximum(abs, evsort(eigvals(copy(A))) .- ref.ev) < 1.0e-9          # real eigvals (geevx)
        @test maximum(abs, evsort(eigvals(copy(Az))) .- ref.zev) < 1.0e-9        # complex eigvals
        @test maximum(abs, evsort(eigvals(copy(Acp))) .- ref.cpev) < 1.0e-9      # conj-pair spectrum
        Fe = eigen(copy(A))                                                    # real eigen (geevx jobvr='V')
        @test maximum(abs, evsort(Fe.values) .- ref.ev) < 1.0e-9
        @test maximum(abs, A * Fe.vectors - Fe.vectors * Diagonal(Fe.values)) / (opnorm(A, 1) * n * eps()) < 200
        Fcp = eigen(copy(Acp))                                                 # conj-pair eigen residual
        @test maximum(abs, Acp * Fcp.vectors - Fcp.vectors * Diagonal(Fcp.values)) / (opnorm(Acp, 1) * n * eps()) < 200
        Fze = eigen(copy(Az))                                                  # complex eigen
        @test maximum(abs, Az * Fze.vectors - Fze.vectors * Diagonal(Fze.values)) / (opnorm(Az, 1) * n * eps()) < 200
        Fs = schur(copy(A))                                                    # schur (gees)
        @test maximum(abs, (Matrix(Fs.Z) * Matrix(Fs.T) * Matrix(Fs.Z)' - A) .- ref.sch) < 1.0e-8
        @test maximum(abs, Matrix(Fs.Z)' * Matrix(Fs.Z) - I) < 1.0e-10
    finally
        PureBLAS.deactivate()
    end
end

@testitem "LBT in-process forward: generalized/tridiagonal eigen (eigen(A,B)/eigen(Sym,Sym)/SymTridiagonal/gtsv)" tags = [:lbt] begin
    using PureBLAS, LinearAlgebra, Random
    import LinearAlgebra.BLAS as B
    import LinearAlgebra.LAPACK as LA
    Random.seed!(0xC0DE)
    n = 40
    A = randn(n, n); Az = randn(ComplexF64, n, n)
    Ms = randn(n, n); As = Ms + Ms'                        # symmetric
    Ns = randn(n, n); Bs = Ns * Ns' + n * I                # symmetric positive definite
    Bw = (M = randn(n, n); M'M + n * I)                    # well-conditioned general B (all finite eigs)
    Bwz = (M = randn(ComplexF64, n, n); M'M + n * I)
    dv = randn(n); ev = randn(n - 1); ST = SymTridiagonal(dv, ev)
    # tridiagonal system for a direct LAPACK.gtsv! routing check (Tridiagonal\b is native Julia, not gtsv)
    dl = randn(n - 1); dd = randn(n) .+ 4; du = randn(n - 1)
    Td = diagm(-1 => dl, 0 => dd, 1 => du); rhs = randn(n)

    # OpenBLAS oracles captured BEFORE activate.
    evsort(v) = sort(v; by = x -> (real(x), imag(x)))
    # Set-match (nearest neighbour): robust to eigenvalue ORDER (sorting complex-conjugate pairs by
    # (re,im) is unstable across backends when real parts coincide — an absolute sorted diff spuriously
    # blows up though the spectra are identical). Correctness of magnitudes is what this asserts.
    setmatch(p, q) = maximum(i -> minimum(j -> abs(p[i] - q[j]), eachindex(q)), eachindex(p))
    ref = (
        gab = eigvals(copy(A), copy(Bw)), gabz = eigvals(copy(Az), copy(Bwz)),
        gsym = eigvals(Symmetric(copy(As)), Symmetric(copy(Bs))),
        stev = eigvals(ST), gtsv = LA.gtsv!(copy(dl), copy(dd), copy(du), copy(rhs)),
    )
    Fst_ref = eigen(ST)

    fwd(s) = B.lbt_get_forward(s, Int32(B.LBT_INTERFACE_ILP64), Int32(B.LBT_F2C_PLAIN))
    ggev3_b = fwd("dggev3_"); zggev3_b = fwd("zggev3_"); gges3_b = fwd("dgges3_")
    sygvd_b = fwd("dsygvd_"); stev_b = fwd("dstev_"); stegr_b = fwd("dstegr_"); gtsv_b = fwd("dgtsv_")
    PureBLAS.activate()
    try
        @test fwd("dggev3_") != ggev3_b && fwd("zggev3_") != zggev3_b && fwd("dgges3_") != gges3_b
        @test fwd("dsygvd_") != sygvd_b && fwd("dstev_") != stev_b && fwd("dstegr_") != stegr_b
        @test fwd("dgtsv_") != gtsv_b
        # eigen(A,B) / eigvals(A,B) → ggev3 (set-match: robust to conjugate-pair ordering)
        @test setmatch(eigvals(copy(A), copy(Bw)), ref.gab) < 1.0e-8
        @test setmatch(eigvals(copy(Az), copy(Bwz)), ref.gabz) < 1.0e-8
        Fg = eigen(copy(A), copy(Bw))                       # generalized eigen residual (finite eigs)
        @test maximum(abs, A * Fg.vectors - Bw * Fg.vectors * Diagonal(Fg.values)) /
            ((opnorm(A, 1) + opnorm(Bw, 1)) * n) < 1.0e-10
        # eigen(Symmetric,Symmetric) → sygvd
        @test maximum(abs, eigvals(Symmetric(copy(As)), Symmetric(copy(Bs))) .- ref.gsym) < 1.0e-8
        Fgs = eigen(Symmetric(copy(As)), Symmetric(copy(Bs)))
        @test maximum(abs, As * Fgs.vectors - Bs * Fgs.vectors * Diagonal(Fgs.values)) /
            (opnorm(As, 1) * opnorm(Bs, 1) * n) < 1.0e-10
        # schur(A,B) → gges3
        Fsc = schur(copy(A), copy(Bw))
        @test maximum(abs, Fsc.Q * Fsc.S * Fsc.Z' - A) < 1.0e-9 * (opnorm(A, 1) + 1)
        @test maximum(abs, Fsc.Q * Fsc.T * Fsc.Z' - Bw) < 1.0e-9 * (opnorm(Bw, 1) + 1)
        # eigen(SymTridiagonal) → stegr; eigvals(SymTridiagonal) → stev
        @test maximum(abs, eigvals(ST) .- ref.stev) < 1.0e-9
        Fst = eigen(ST)
        @test maximum(abs, sort(Fst.values) .- sort(Fst_ref.values)) < 1.0e-9
        @test maximum(abs, Matrix(ST) * Fst.vectors - Fst.vectors * Diagonal(Fst.values)) < 1.0e-8 * (norm(dv) + norm(ev))
        # direct LAPACK.gtsv! routes (Tridiagonal\b itself is native-Julia, not a BLAS/LAPACK call)
        x = LA.gtsv!(copy(dl), copy(dd), copy(du), copy(rhs))
        @test maximum(abs, Td * x - rhs) < 1.0e-9
        @test maximum(abs, x .- ref.gtsv) < 1.0e-9
    finally
        PureBLAS.deactivate()
    end
end

@testitem "LBT in-process forward: batch-7 assembly (sysv/hesv/sytri/hetri, gbtrf/gbtrs, pttrf/pttrs/ptsv, stebz/stein, pstrf, QL/RQ, gelsy/tzrzf/ormrz, gelsd, trsyl, trexc/trsen, gglse, ggsvd)" tags = [:lbt] begin
    using PureBLAS, LinearAlgebra, Random
    import LinearAlgebra.BLAS as B
    import LinearAlgebra.LAPACK as LA
    Random.seed!(0xA55E)
    fwd(s) = B.lbt_get_forward(s, Int32(B.LBT_INTERFACE_ILP64), Int32(B.LBT_F2C_PLAIN))

    # None of this batch's routines are reached through Julia's high-level `\`/factorize/eigen/svd
    # dispatch (verified by grep over LinearAlgebra/src: sysv!/hesv!/gelsd!/gelsy! are never called
    # outside lapack.jl; bunchkaufman routes through sytrf!/sytrs! instead; qr.jl's non-square `\`
    # implements its own gelsy-like algorithm natively rather than calling LAPACK.gelsy!/gelsd!). So
    # every check here is a DIRECT `LinearAlgebra.LAPACK.<name>!` call (mirrors the gtsv routing test).
    before = Dict(
        s => fwd(s) for s in (
                "dsysv_", "zhesv_", "dsytri_", "zhetri_", "dgbtrf_", "dgbtrs_", "dpttrf_", "dpttrs_", "dptsv_",
                "dstebz_", "dstein_", "dpstrf_", "dgeqlf_", "dgerqf_", "dorgql_", "dorgrq_", "dormql_", "dormrq_",
                "dgelsy_", "dtzrzf_", "dormrz_", "dgelsd_", "dtrsyl_", "dtrexc_", "dtrsen_", "dgglse_", "dggsvd3_",
            )
    )
    PureBLAS.activate()
    try
        @test all(s -> fwd(s) != before[s], keys(before))    # every symbol actually rerouted

        n = 20
        # sysv / hesv / sytri / hetri
        M = randn(n, n); A = M + M' + n * I; Bv = randn(n, 3)
        Xr, _, _ = LA.sysv!('L', copy(A), copy(Bv))
        @test norm(A * Xr - Bv) < 1.0e-8 * (norm(A) * norm(Xr) + norm(Bv))
        Mz = randn(ComplexF64, n, n); Az = Mz + Mz' + n * I; Bz = randn(ComplexF64, n, 3)
        Xz, _, _ = LA.hesv!('L', copy(Az), copy(Bz))
        @test norm(Az * Xz - Bz) < 1.0e-8 * (norm(Az) * norm(Xz) + norm(Bz))
        LD, ipiv, _ = LA.sytrf!('L', copy(A))
        Ainv = LA.sytri!('L', copy(LD), ipiv)
        @test norm(A * Symmetric(Ainv, :L) - I) < 1.0e-6 * n

        # gbtrf / gbtrs (banded LU)
        kl, ku = 2, 1
        Ab = zeros(n, n)
        for j in 1:n, i in max(1, j - ku):min(n, j + kl)
            Ab[i, j] = randn()
        end
        Ab += n * I
        ldab = 2kl + ku + 1
        AB = zeros(ldab, n)
        for j in 1:n, i in max(1, j - ku):min(n, j + kl)
            AB[kl + ku + 1 + i - j, j] = Ab[i, j]
        end
        ABf, ipivb = LA.gbtrf!(kl, ku, n, copy(AB))
        bb = randn(n)
        xb = LA.gbtrs!('N', kl, ku, n, ABf, ipivb, copy(bb))
        @test norm(Ab * xb - bb) < 1.0e-8 * (norm(Ab) + 1)

        # pttrf / pttrs / ptsv (SPD tridiagonal)
        dv = rand(n) .+ (n + 2); ev = randn(n - 1) .* 0.3
        At = SymTridiagonal(dv, ev)
        d1, e1 = LA.pttrf!(copy(dv), copy(ev))
        bt = randn(n, 2)
        xt = LA.pttrs!(d1, e1, copy(bt))
        @test norm(Matrix(At) * xt - bt) < 1.0e-7 * (norm(Matrix(At)) + 1)
        xt2 = LA.ptsv!(copy(dv), copy(ev), copy(bt))
        @test norm(Matrix(At) * xt2 - bt) < 1.0e-7 * (norm(Matrix(At)) + 1)

        # stebz / stein (SymTridiagonal eigen, expert path)
        wb, ibb, isb = LA.stebz!('A', 'E', 0.0, 0.0, 0, 0, -1.0, copy(dv), copy(ev))
        @test maximum(abs, sort(wb) .- sort(eigvals(At))) < 1.0e-8 * (norm(dv) + norm(ev) + 1)
        Zb = LA.stein!(dv, ev, wb, ibb, isb)
        @test maximum(abs, Matrix(At) * Zb - Zb * Diagonal(wb)) < 1.0e-6 * (norm(dv) + norm(ev) + 1)

        # pstrf (pivoted Cholesky)
        Mp = randn(n, n); Ap = Mp * Mp' + n * I
        Fp, pv, rk, infop = LA.pstrf!('L', copy(Ap), -1.0)
        @test infop == 0 && rk == n
        @test maximum(abs, tril(Fp) * tril(Fp)' .- Ap[pv, pv]) < 1.0e-6 * maximum(abs, Ap)

        # QL / RQ
        Aql = randn(n + 4, n)               # QL needs rows ≥ cols
        Fql, tql = LA.geqlf!(copy(Aql))
        Qql = LA.orgql!(copy(Fql), tql)
        @test maximum(abs, Qql' * Qql - I) < 1.0e-8
        Arq = randn(n, n + 4)               # RQ needs rows ≤ cols
        Frq, trq = LA.gerqf!(copy(Arq))
        Qrq = LA.orgrq!(copy(Frq), trq)
        @test maximum(abs, Qrq * Qrq' - I) < 1.0e-8
        Cql = randn(n + 4, 3)
        Cql1 = LA.ormql!('L', 'T', copy(Fql), tql, copy(Cql))
        Cql2 = LA.ormql!('L', 'N', copy(Fql), tql, copy(Cql1))
        @test maximum(abs, Cql2 .- Cql) < 1.0e-6

        # gelsy / tzrzf / ormrz (rank-deficient LS via RZ) and gelsd (via SVD)
        m2, n2 = 25, 15
        A2 = randn(m2, n2); b2 = randn(m2)
        B2 = zeros(max(m2, n2), 1); B2[1:m2, 1] = b2
        Xy, rky = LA.gelsy!(copy(A2), copy(B2))
        @test rky == n2
        @test norm(A2' * (A2 * Xy[1:n2, :] - b2)) < 1.0e-6 * (norm(A2)^2 * norm(Xy) + 1)
        B2d = zeros(max(m2, n2), 1); B2d[1:m2, 1] = b2
        Xd, rkd = LA.gelsd!(copy(A2), copy(B2d))
        @test rkd == n2
        @test norm(Xd[1:n2, :] .- Xy[1:n2, :]) < 1.0e-5 * (norm(Xy) + 1)

        # trsyl (triangular Sylvester)
        Atri = triu(randn(n, n)) + n * I
        Btri = triu(randn(6, 6)) .* 0.3 + 3n * I
        Ctri = randn(n, 6)
        Xtri, sctri = LA.trsyl!('N', 'N', copy(Atri), copy(Btri), copy(Ctri))
        @test norm(Atri * Xtri + Xtri * Btri - sctri * Ctri) < 1.0e-6 * (norm(Atri) * norm(Btri) * norm(Xtri) + norm(Ctri))

        # trexc / trsen (Schur reorder)
        Ssch = schur(randn(n, n))
        Torig = Matrix(Ssch.T); Q0 = Matrix(Ssch.Z)
        Aorig = Q0 * Torig * Q0'
        Te, Qe = LA.trexc!('V', 1, min(3, n), copy(Torig), copy(Q0))
        @test maximum(abs, Qe * Te * Qe' - Aorig) < 1.0e-7 * (opnorm(Aorig, 1) + 1)
        sel = zeros(Int, n); sel[1] = 1
        Ts, Qs, ws, ss, seps = LA.trsen!('N', 'V', sel, copy(Torig), copy(Q0))
        @test maximum(abs, Qs * Ts * Qs' - Aorig) < 1.0e-7 * (opnorm(Aorig, 1) + 1)

        # gglse (equality-constrained LS)
        Ag = randn(10, 6); cg = randn(10); Bg = randn(3, 6); dg = Bg * randn(6)
        xg, resg = LA.gglse!(copy(Ag), copy(cg), copy(Bg), copy(dg))
        @test norm(Bg * xg - dg) < 1.0e-6 * (norm(Bg) * norm(xg) + norm(dg) + 1)

        # ggsvd3 (generalized SVD, Float64 full-rank)
        Ag2 = randn(10, 6); Bg2 = randn(8, 6)
        Ug, Vg, Qg, alphag, betag, kg, lg, Rg = LA.ggsvd3!('U', 'V', 'Q', copy(Ag2), copy(Bg2))
        @test maximum(abs, Ug' * Ug - I) < 1.0e-8
        @test maximum(abs, Vg' * Vg - I) < 1.0e-8
        @test length(alphag) == 6 && length(betag) == 6
    finally
        PureBLAS.deactivate()
    end
end

@testitem "LBT in-process forward: batch-8 (gesv/posv/lacpy/larfg/larf)" tags = [:lbt] begin
    using PureBLAS, LinearAlgebra, Random
    import LinearAlgebra.BLAS as B
    import LinearAlgebra.LAPACK as LA
    Random.seed!(0xB8)
    fwd(s) = B.lbt_get_forward(s, Int32(B.LBT_INTERFACE_ILP64), Int32(B.LBT_F2C_PLAIN))
    n = 12

    # None of this batch's routines are reached through Julia's high-level `\`/factorize dispatch (gesv/
    # posv are one-shot combined factor+solve LAPACK drivers; Julia's own `\` composes getrf!+getrs! /
    # potrf!+potrs! separately) — every check is a DIRECT `LinearAlgebra.LAPACK.<name>!` call, oracle
    # captured BEFORE activate() (mirrors the batch-7/gtsv style).
    A = randn(n, n) + n * I; Bv = randn(n, 3)
    Az = randn(ComplexF64, n, n) + n * I; Bz = randn(ComplexF64, n, 2)
    Af = randn(Float32, n, n) + Float32(n) * I; Bf = randn(Float32, n, 2)
    M = randn(n, n); SPD = M * M' + n * I; Bp = randn(n, 2)
    Mz = randn(ComplexF64, n, n); SPDz = Mz * Mz' + n * I; Bpz = randn(ComplexF64, n, 2)
    Ac = randn(n, n)
    x = randn(n); xz = randn(ComplexF64, n)

    ref = (
        gesv = LA.gesv!(copy(A), copy(Bv))[1], zgesv = LA.gesv!(copy(Az), copy(Bz))[1],
        sgesv = LA.gesv!(copy(Af), copy(Bf))[1],
        posv = LA.posv!('L', copy(SPD), copy(Bp))[2], zposv = LA.posv!('L', copy(SPDz), copy(Bpz))[2],
        lacpyU = triu(Ac), lacpyL = tril(Ac),
    )

    # larfg!/larf! oracles (OpenBLAS, pre-activate). larfg! mutates x → x[1]=1, x[2:]=essential v.
    # larf!('L') applies H (=I−τ·v·vᴴ, the LAPACK zlarf op) to C. Note LAPACK zlarfg makes Hᴴ (not H)
    # zero the tail, so the convention-agnostic correctness test is "identical to OpenBLAS", not "zeros".
    xr_o = copy(x); taur_o = LA.larfg!(xr_o)
    Cr0 = randn(n, 3); Cr_o = copy(Cr0); LA.larf!('L', xr_o, taur_o, Cr_o)
    xz_o = copy(xz); tauz_o = LA.larfg!(xz_o)
    Cz0 = randn(ComplexF64, n, 3); Cz_o = copy(Cz0); LA.larf!('L', xz_o, tauz_o, Cz_o)

    before = Dict(
        s => fwd(s) for s in (
                "sgesv_", "dgesv_", "cgesv_", "zgesv_", "sposv_", "dposv_", "cposv_", "zposv_",
                "slacpy_", "dlacpy_", "clacpy_", "zlacpy_", "slarfg_", "dlarfg_", "clarfg_", "zlarfg_",
                "slarf_", "dlarf_", "clarf_", "zlarf_",
            )
    )
    PureBLAS.activate()
    try
        @test all(s -> fwd(s) != before[s], keys(before))

        Xpb, _, _ = LA.gesv!(copy(A), copy(Bv))
        @test maximum(abs, Xpb .- ref.gesv) < 1.0e-8 * (norm(ref.gesv) + 1)
        @test maximum(abs, A * Xpb - Bv) < 1.0e-8 * (norm(A) * norm(Xpb) + norm(Bv))          # residual too
        Xzpb, _, _ = LA.gesv!(copy(Az), copy(Bz))
        @test maximum(abs, Xzpb .- ref.zgesv) < 1.0e-8 * (norm(ref.zgesv) + 1)
        Xfpb, _, _ = LA.gesv!(copy(Af), copy(Bf))
        @test maximum(abs, Xfpb .- ref.sgesv) < 1.0f-3 * (norm(ref.sgesv) + 1)                # F32 mixed-prec

        _, Xppb = LA.posv!('L', copy(SPD), copy(Bp))     # posv! returns (A_factor, X) — X is [2]
        @test maximum(abs, Xppb .- ref.posv) < 1.0e-8 * (norm(ref.posv) + 1)
        @test maximum(abs, SPD * Xppb - Bp) < 1.0e-8 * (norm(SPD) * norm(Xppb) + norm(Bp))
        _, Xpzpb = LA.posv!('L', copy(SPDz), copy(Bpz))
        @test maximum(abs, Xpzpb .- ref.zposv) < 1.0e-8 * (norm(ref.zposv) + 1)

        Bu = zeros(n, n); LA.lacpy!(Bu, Ac, 'U'); @test Bu ≈ ref.lacpyU
        Bl = zeros(n, n); LA.lacpy!(Bl, Ac, 'L'); @test Bl ≈ ref.lacpyL
        Bfull = zeros(n, n); LA.lacpy!(Bfull, Ac, 'A'); @test Bfull ≈ Ac

        # larfg!/larf! must reproduce OpenBLAS exactly (tau + essential v + the applied C).
        xr_p = copy(x); taur_p = LA.larfg!(xr_p)
        @test abs(taur_p - taur_o) < 1.0e-10 && maximum(abs, xr_p .- xr_o) < 1.0e-10
        Cr_p = copy(Cr0); LA.larf!('L', xr_p, taur_p, Cr_p)
        @test maximum(abs, Cr_p .- Cr_o) < 1.0e-9 * (norm(Cr_o) + 1)
        xz_p = copy(xz); tauz_p = LA.larfg!(xz_p)
        @test abs(tauz_p - tauz_o) < 1.0e-10 && maximum(abs, xz_p .- xz_o) < 1.0e-10
        Cz_p = copy(Cz0); LA.larf!('L', xz_p, tauz_p, Cz_p)
        @test maximum(abs, Cz_p .- Cz_o) < 1.0e-9 * (norm(Cz_o) + 1)
    finally
        PureBLAS.deactivate()
    end
end

@testitem "LBT in-process forward: batch-9 (gebak/hseqr/trevc, sytrd·hetrd/orgtr·ungtr/ormtr·unmtr)" tags = [:lbt] begin
    using PureBLAS, LinearAlgebra, Random
    import LinearAlgebra.BLAS as B
    import LinearAlgebra.LAPACK as LA
    Random.seed!(0xB9)
    fwd(s) = B.lbt_get_forward(s, Int32(B.LBT_INTERFACE_ILP64), Int32(B.LBT_F2C_PLAIN))
    n = 10

    # None reached via Julia's high-level dispatch for a PLAIN Matrix (eigen/schur route through the
    # geevx_/gees_ ABI symbols, whose _geev_run!/_gees_run! composition calls hseqr!/trevc!/gebak! as
    # Julia FUNCTIONS, not through these ABI POINTERS) — direct `LinearAlgebra.LAPACK.<name>!` calls,
    # oracle captured BEFORE activate().
    A = randn(n, n)
    ilo, ihi, scale = LA.gebal!('B', copy(A))
    V = randn(n, n)
    Href = copy(A); tauref = LA.gehrd!(Href)[2]; Qref = LA.orghr!(1, n, copy(Href), tauref)
    for i in 1:(n - 1), r in (i + 2):n
        Href[r, i] = 0.0
    end        # clean Hessenberg
    Hs, Zs, ws = LA.hseqr!('S', 'V', 1, n, copy(Href), copy(Qref))   # Schur form for the trevc oracle
    selref = zeros(Int, n)

    Msy = randn(n, n); Asy = Msy + Msy'
    Mhe = randn(ComplexF64, n, n); Ahe = Mhe + Mhe'

    ref = (
        gebak = LA.gebak!('B', 'R', ilo, ihi, copy(scale), copy(V)),
        hseqr_w = sort(ws; by = x -> (real(x), imag(x))),
        trevcA = LA.trevc!('R', 'A', selref, copy(Hs))[1],
        trevcB = LA.trevc!('R', 'B', selref, copy(Hs), similar(Hs), copy(Zs)),
        sytrd = LA.hetrd!('L', copy(Asy)), hetrd = LA.hetrd!('L', copy(Ahe)),
    )
    Qsy = LA.orgtr!('L', copy(ref.sytrd[1]), ref.sytrd[2])
    Qhe = LA.orgtr!('L', copy(ref.hetrd[1]), ref.hetrd[2])
    Csy = randn(n, 3); Cherm = randn(ComplexF64, n, 3)
    ormtrref = LA.ormtr!('L', 'L', 'N', copy(ref.sytrd[1]), ref.sytrd[2], copy(Csy))
    unmtrref = LA.ormtr!('L', 'L', 'N', copy(ref.hetrd[1]), ref.hetrd[2], copy(Cherm))

    before = Dict(
        s => fwd(s) for s in (
                "sgebak_", "dgebak_", "cgebak_", "zgebak_", "shseqr_", "dhseqr_", "chseqr_", "zhseqr_",
                "strevc_", "dtrevc_", "ctrevc_", "ztrevc_", "ssytrd_", "dsytrd_", "chetrd_", "zhetrd_",
                "sorgtr_", "dorgtr_", "cungtr_", "zungtr_", "sormtr_", "dormtr_", "cunmtr_", "zunmtr_",
            )
    )
    PureBLAS.activate()
    try
        @test all(s -> fwd(s) != before[s], keys(before))

        Vpb = copy(V); LA.gebak!('B', 'R', ilo, ihi, copy(scale), Vpb)
        @test maximum(abs, Vpb .- ref.gebak) < 1.0e-9 * (norm(ref.gebak) + 1)

        Hpb, Zpb, wpb = LA.hseqr!('S', 'V', 1, n, copy(Href), copy(Qref))
        @test maximum(abs, sort(wpb; by = x -> (real(x), imag(x))) .- ref.hseqr_w) < 1.0e-8
        @test maximum(abs, Zpb * Hpb * Zpb' - A) < 1.0e-8 * (norm(A) + 1)        # Schur reconstruction
        @test maximum(abs, Zpb' * Zpb - I) < 1.0e-9

        VAp = LA.trevc!('R', 'A', selref, copy(Hs))[1]
        @test maximum(abs, VAp .- ref.trevcA) < 1.0e-8 * (norm(ref.trevcA) + 1)
        VBp = LA.trevc!('R', 'B', selref, copy(Hs), similar(Hs), copy(Zs))
        @test maximum(abs, VBp .- ref.trevcB) < 1.0e-8 * (norm(ref.trevcB) + 1)

        Asyf, tausy = LA.hetrd!('L', copy(Asy))
        @test maximum(abs, Asyf[1, 1] - ref.sytrd[1][1, 1]) < 1.0e-9    # sanity: same diagonal reduction
        Qsypb = LA.orgtr!('L', copy(Asyf), tausy)
        @test maximum(abs, Qsypb' * Qsypb - I) < 1.0e-9
        @test maximum(abs, Qsypb .- Qsy) < 1.0e-8 * (norm(Qsy) + 1)

        Ahef2, tauhe = LA.hetrd!('L', copy(Ahe))
        Qhepb = LA.orgtr!('L', copy(Ahef2), tauhe)
        @test maximum(abs, Qhepb' * Qhepb - I) < 1.0e-9
        @test maximum(abs, Qhepb .- Qhe) < 1.0e-8 * (norm(Qhe) + 1)

        Csypb = LA.ormtr!('L', 'L', 'N', copy(Asyf), tausy, copy(Csy))
        @test maximum(abs, Csypb .- ormtrref) < 1.0e-8 * (norm(ormtrref) + 1)
        Chepb = LA.ormtr!('L', 'L', 'N', copy(Ahef2), tauhe, copy(Cherm))
        @test maximum(abs, Chepb .- unmtrref) < 1.0e-8 * (norm(unmtrref) + 1)
    finally
        PureBLAS.deactivate()
    end
end

@testitem "LBT in-process forward: batch-10 (orgqr·ungqr/ormqr·unmqr/ormhr·unmhr, gebrd/bdsqr/bdsdc)" tags = [:lbt] begin
    using PureBLAS, LinearAlgebra, Random
    import LinearAlgebra.BLAS as B
    import LinearAlgebra.LAPACK as LA
    Random.seed!(0x0B10)
    fwd(s) = B.lbt_get_forward(s, Int32(B.LBT_INTERFACE_ILP64), Int32(B.LBT_F2C_PLAIN))
    n = 10

    # orgqr/ormqr are reached via a DIRECT LAPACK.orgqr!/ormqr! call (Julia's qr() uses geqrt+gemqrt, the
    # QRCompactWY path — already covered by the main forward testitem). gebrd/bdsqr/bdsdc/ormhr are also
    # direct-caller-only. Oracle captured BEFORE activate().
    A = randn(n, n - 3)
    Aref = copy(A); tauref = LA.geqrf!(Aref)[2]
    Qref = LA.orgqr!(copy(Aref), tauref)
    C = randn(n, 4)
    ormqrref = LA.ormqr!('L', 'N', copy(Aref), tauref, copy(C))

    Ah = randn(n, n)
    Href = copy(Ah); tauh = LA.gehrd!(Href)[2]
    Ch = randn(n, 3)
    ormhrref = LA.ormhr!('L', 'N', 1, n, copy(Href), tauh, copy(Ch))

    m2 = 14
    Ab = randn(m2, n)
    gebrdref = LA.gebrd!(copy(Ab))     # (A, d, e, tauq, taup)
    svref = svdvals(Ab)

    dbd = rand(8) .+ 0.5; ebd = randn(7) .* 0.6
    Uid = Matrix{Float64}(I, 8, 8); Vtid = Matrix{Float64}(I, 8, 8); Cempty = zeros(8, 0)
    bdsqrref = LA.bdsqr!('U', copy(dbd), copy(ebd), copy(Vtid), copy(Uid), copy(Cempty))
    bdsdcref = LA.bdsdc!('U', 'I', copy(dbd), copy(ebd))

    before = Dict(
        s => fwd(s) for s in (
                "sorgqr_", "dorgqr_", "cungqr_", "zungqr_", "sormqr_", "dormqr_", "cunmqr_", "zunmqr_",
                "sormhr_", "dormhr_", "cunmhr_", "zunmhr_", "sgebrd_", "dgebrd_", "cgebrd_", "zgebrd_",
                "sbdsqr_", "dbdsqr_", "sbdsdc_", "dbdsdc_",
            )
    )
    PureBLAS.activate()
    try
        @test all(s -> fwd(s) != before[s], keys(before))

        Apb = copy(A); taupb = LA.geqrf!(Apb)[2]
        Qpb = LA.orgqr!(copy(Apb), taupb)
        @test maximum(abs, Qpb' * Qpb - I) < 1.0e-9
        @test maximum(abs, Qpb .- Qref) < 1.0e-8 * (norm(Qref) + 1)
        Cpb = LA.ormqr!('L', 'N', copy(Apb), taupb, copy(C))
        @test maximum(abs, Cpb .- ormqrref) < 1.0e-8 * (norm(ormqrref) + 1)

        Hpb = copy(Ah); tauhpb = LA.gehrd!(Hpb)[2]
        Chpb = LA.ormhr!('L', 'N', 1, n, copy(Hpb), tauhpb, copy(Ch))
        @test maximum(abs, Chpb .- ormhrref) < 1.0e-8 * (norm(ormhrref) + 1)

        gpb = LA.gebrd!(copy(Ab))
        @test maximum(abs, gpb[2] .- gebrdref[2]) < 1.0e-8 * (norm(gebrdref[2]) + 1)   # d
        @test maximum(abs, sort(abs.(gpb[3])) .- sort(abs.(gebrdref[3]))) < 1.0e-6 * (norm(gebrdref[3]) + 1)  # e magnitude
        # independent oracle: the bidiagonal's own singular values must match svdvals(Ab) regardless of
        # sign/algorithm details in the reduction above.
        dchk = copy(gpb[2]); echk = copy(gpb[3])[1:(n - 1)]   # gebrd e is length k; bdsqr wants n-1 off-diags
        Uidm = Matrix{Float64}(I, m2, m2)[:, 1:n]; Vtidm = Matrix{Float64}(I, n, n)
        LA.bdsqr!('U', dchk, echk, Vtidm, Uidm, zeros(n, 0))
        @test maximum(abs, sort(dchk) .- sort(svref)) < 1.0e-7 * (norm(svref) + 1)

        dqpb = copy(dbd); eqpb = copy(ebd)
        bdsqrpb = LA.bdsqr!('U', dqpb, eqpb, copy(Vtid), copy(Uid), copy(Cempty))
        @test maximum(abs, sort(bdsqrpb[1]) .- sort(bdsqrref[1])) < 1.0e-8
        Bmat = diagm(0 => dbd, 1 => ebd)
        Upb2 = bdsqrpb[3]; Vtpb2 = bdsqrpb[2]; Spb2 = bdsqrpb[1]
        @test maximum(abs, Upb2 * Diagonal(Spb2) * Vtpb2 - Bmat) < 1.0e-7 * (norm(Bmat) + 1)

        dcpb = copy(dbd); ecpb = copy(ebd)
        bdsdcpb = LA.bdsdc!('U', 'I', dcpb, ecpb)
        @test maximum(abs, sort(bdsdcpb[1]) .- sort(bdsdcref[1])) < 1.0e-8
        Ubd = bdsdcpb[3]; Vtbd = bdsdcpb[4]
        @test maximum(abs, Ubd * Diagonal(bdsdcpb[1]) * Vtbd - Bmat) < 1.0e-7 * (norm(Bmat) + 1)

        # uplo='L' path (the Bᵀ swap-trick): reconstruct against a LOWER-bidiagonal B.
        dl = rand(8) .+ 0.5; el = randn(7) .* 0.6
        dlpb = copy(dl); elpb = copy(el)
        Ulpb = Matrix{Float64}(I, 8, 8); Vtlpb = Matrix{Float64}(I, 8, 8)
        bdlref = LA.bdsqr!('L', copy(dl), copy(el), copy(Vtlpb), copy(Ulpb), copy(Cempty))
        bdlpb = LA.bdsqr!('L', dlpb, elpb, Vtlpb, Ulpb, copy(Cempty))
        @test maximum(abs, sort(bdlpb[1]) .- sort(bdlref[1])) < 1.0e-8
        Bl_ = diagm(0 => dl, -1 => el)
        @test maximum(abs, Ulpb * Diagonal(bdlpb[1]) * Vtlpb - Bl_) < 1.0e-7 * (norm(Bl_) + 1)
    finally
        PureBLAS.deactivate()
    end
end

@testitem "LBT in-process forward: batch-11 (syconv, trrfs)" tags = [:lbt] begin
    using PureBLAS, LinearAlgebra, Random
    import LinearAlgebra.BLAS as B
    import LinearAlgebra.LAPACK as LA
    Random.seed!(0x0B11)
    fwd(s) = B.lbt_get_forward(s, Int32(B.LBT_INTERFACE_ILP64), Int32(B.LBT_F2C_PLAIN))
    n = 16

    # Neither routine is on a high-level dispatch path — both checked via DIRECT LinearAlgebra.LAPACK.<name>!
    # calls, OpenBLAS oracle captured BEFORE activate(). syconv converts a Bunch-Kaufman (sytrf) factor's
    # 2×2 off-diagonals into `work`; factor ONCE with OpenBLAS so both syconv calls see identical input.
    facs = map((Float64, Float32, ComplexF64, ComplexF32)) do T
        M = randn(T, n, n); A = M + transpose(M)          # symmetric (sytrf, not hetrf) for all 4 types
        LD, ip, _ = LA.sytrf!('L', copy(A)); (LD = LD, ip = ip)
    end
    sycref = map(f -> LA.syconv!('L', copy(f.LD), copy(f.ip))[2], facs)   # OpenBLAS `work` per type

    # trrfs error bounds for a well-conditioned triangular solve — Berr must stay ~eps.
    tris = map((Float64, Float32, ComplexF64, ComplexF32)) do T
        A = triu(randn(T, n, n)) + n * I; Bm = randn(T, n, 3); X = A \ Bm
        (A = A, B = Bm, X = X)
    end
    trref = map(t -> LA.trrfs!('U', 'N', 'N', t.A, t.B, t.X), tris)       # OpenBLAS (Ferr, Berr) per type

    before = Dict(
        s => fwd(s) for s in (
                "ssyconv_", "dsyconv_", "csyconv_", "zsyconv_",
                "strrfs_", "dtrrfs_", "ctrrfs_", "ztrrfs_",
            )
    )
    PureBLAS.activate()
    try
        @test all(s -> fwd(s) != before[s], keys(before))
        for (i, f) in enumerate(facs)
            wpb = LA.syconv!('L', copy(f.LD), copy(f.ip))[2]
            @test maximum(abs, wpb .- sycref[i]) < 1.0e-6 * (norm(sycref[i]) + 1)   # F32-tol covers all 4
        end
        for (i, t) in enumerate(tris)
            Fe, Be = LA.trrfs!('U', 'N', 'N', t.A, t.B, t.X)
            @test maximum(Be) < 1.0e-5                                             # well-conditioned → tiny Berr
            @test maximum(Fe) < 1.0                                              # sane forward bound
        end
    finally
        PureBLAS.deactivate()
    end
end

@testitem "LBT in-process forward: batch-12 (tgsen — generalized Schur reorder, real + complex)" tags = [:lbt] begin
    using PureBLAS, LinearAlgebra, Random
    import LinearAlgebra.BLAS as B
    import LinearAlgebra.LAPACK as LA
    Random.seed!(0x0B12)
    fwd(s) = B.lbt_get_forward(s, Int32(B.LBT_INTERFACE_ILP64), Int32(B.LBT_F2C_PLAIN))
    n = 10

    # ALL FOUR types: real dtgsen/stgsen now handle 2×2 conjugate-pair blocks (dtgex2), complex ctgsen/
    # ztgsen have none. Direct LAPACK.tgsen! call, OpenBLAS oracle pre-activate; select is BlasInt.
    cases = map((Float64, Float32, ComplexF64, ComplexF32)) do T
        A = randn(T, n, n); Bm = randn(T, n, n)
        F = schur(A, Bm)                                   # generalized: A = Q·S·Z', B = Q·T·Z'
        (T = T, S0 = F.S, T0 = F.T, Q0 = F.Q, Z0 = F.Z, sel = Int64[i > n ÷ 2 for i in 1:n])
    end
    oracle = map(cases) do c
        S, Tt, al, be, Q, Z = LA.tgsen!(c.sel, copy(c.S0), copy(c.T0), copy(c.Q0), copy(c.Z0))
        (ev = sort(al ./ be, by = x -> (real(x), imag(x))), rec = Q * S * Z')
    end

    before = Dict(s => fwd(s) for s in ("dtgsen_", "stgsen_", "ctgsen_", "ztgsen_"))
    PureBLAS.activate()
    try
        @test all(s -> fwd(s) != before[s], keys(before))
        for (i, c) in enumerate(cases)
            S, Tt, al, be, Q, Z = LA.tgsen!(c.sel, copy(c.S0), copy(c.T0), copy(c.Q0), copy(c.Z0))
            ev = sort(al ./ be, by = x -> (real(x), imag(x)))
            tol = real(c.T) == Float32 ? 1.0f-3 : 1.0e-9
            @test maximum(abs, ev .- oracle[i].ev) < tol * (norm(oracle[i].ev) + 1)   # same eigenvalues
            @test maximum(abs, Q * S * Z' .- oracle[i].rec) < tol * (norm(oracle[i].rec) + 1)  # pair invariant
        end
    finally
        PureBLAS.deactivate()
    end
end

@testitem "LBT in-process forward: batch-13 (gesvx — expert general solve)" tags = [:lbt] begin
    using PureBLAS, LinearAlgebra, Random
    import LinearAlgebra.BLAS as B
    import LinearAlgebra.LAPACK as LA
    Random.seed!(0x0B13)
    fwd(s) = B.lbt_get_forward(s, Int32(B.LBT_INTERFACE_ILP64), Int32(B.LBT_F2C_PLAIN))
    n = 14
    data = map((Float64, Float32, ComplexF64, ComplexF32)) do T
        (T = T, A = randn(T, n, n) + n * I, Bm = randn(T, n, 3))
    end
    ref = map(d -> LA.gesvx!(copy(d.A), copy(d.Bm))[1], data)   # OpenBLAS X (fact='N')

    before = Dict(s => fwd(s) for s in ("sgesvx_", "dgesvx_", "cgesvx_", "zgesvx_"))
    PureBLAS.activate()
    try
        @test all(s -> fwd(s) != before[s], keys(before))
        for (i, d) in enumerate(data)
            X, rc, fe, be, rp = LA.gesvx!(copy(d.A), copy(d.Bm))
            tol = real(d.T) == Float32 ? 1.0f-3 : 1.0e-8
            @test maximum(abs, X .- ref[i]) < tol * (norm(ref[i]) + 1)                # matches OpenBLAS X
            @test maximum(abs, d.A * X - d.Bm) < tol * (norm(d.A) * norm(X) + norm(d.Bm))  # residual
            @test maximum(be) < tol                                                   # backward error tiny
            @test 0 < rc <= 1                                                          # sane rcond
        end
    finally
        PureBLAS.deactivate()
    end
end

@testitem "LBT in-process forward: batch-14 (bdsqr — complex bidiagonal SVD)" tags = [:lbt] begin
    using PureBLAS, LinearAlgebra, Random
    import LinearAlgebra.BLAS as B
    import LinearAlgebra.LAPACK as LA
    Random.seed!(0x0B14)
    fwd(s) = B.lbt_get_forward(s, Int32(B.LBT_INTERFACE_ILP64), Int32(B.LBT_F2C_PLAIN))
    # real bidiagonals (clustered / graded / random), complex Vt/U/C accumulators.
    mk(T, m, kind) = (
        Tr = real(T); kind == :clustered ?
            (Tr[1 + 1.0e-10 * randn() for _ in 1:m], Tr[1.0e-8 * randn() for _ in 1:(m - 1)]) :
            kind == :graded ? (Tr[Tr(10)^(i - m ÷ 2) for i in 1:m], Tr[Tr(10)^(i - m ÷ 2) / 3 for i in 1:(m - 1)]) :
            (randn(Tr, m), randn(Tr, m - 1))
    )
    cases = [(T, m, k) for T in (ComplexF64, ComplexF32) for m in (6, 25) for k in (:clustered, :graded, :random)]
    ref = map(cases) do (T, m, k)
        d, e = mk(T, m, k)
        Id = Matrix{T}(I, m, m)
        (d0 = d, e0 = e, sv = sort(LA.bdsqr!('U', copy(d), copy(e), copy(Id), copy(Id), copy(Id))[1], rev = true))
    end

    before = Dict(s => fwd(s) for s in ("cbdsqr_", "zbdsqr_"))
    PureBLAS.activate()
    try
        @test all(s -> fwd(s) != before[s], keys(before))
        for (i, (T, m, k)) in enumerate(cases)
            Tr = real(T)
            d = copy(ref[i].d0); e = copy(ref[i].e0)
            Vt = Matrix{T}(I, m, m); U = Matrix{T}(I, m, m); C = Matrix{T}(I, m, m)
            d2, Vt2, U2, C2 = LA.bdsqr!('U', d, e, Vt, U, C)
            sv = sort(d2, rev = true)
            @test maximum(abs, sv .- ref[i].sv) < 1.0e-6 * (sv[1] + 1)                  # σ match OpenBLAS
            Bre = zeros(T, m, m)
            for j in 1:m
                Bre[j, j] = ref[i].d0[j]
            end
            for j in 1:(m - 1)
                Bre[j, j + 1] = ref[i].e0[j]
            end
            @test maximum(abs, U2 * Diagonal(d2) * Vt2 .- Bre) < 1.0e-5 * (maximum(abs, Bre) + 1)  # recon
        end
    finally
        PureBLAS.deactivate()
    end
end

@testitem "LBT in-process forward: batch-15 (ggsvd — rank-deficient generalized SVD, s/d/c/z)" tags = [:lbt] begin
    using PureBLAS, LinearAlgebra, Random
    import LinearAlgebra.BLAS as B
    import LinearAlgebra.LAPACK as LA
    Random.seed!(0x0B15)
    fwd(s) = B.lbt_get_forward(s, Int32(B.LBT_INTERFACE_ILP64), Int32(B.LBT_F2C_PLAIN))
    m, n, p = 8, 6, 7
    # full-rank AND rank-deficient (low-rank A) pairs. Direct LAPACK.ggsvd!, OpenBLAS oracle pre-activate.
    data = [(T, rd) for T in (Float64, Float32, ComplexF64, ComplexF32) for rd in (false, true)]
    mats = map(d -> (A = d[2] ? randn(d[1], m, 2) * randn(d[1], 2, n) : randn(d[1], m, n), Bm = randn(d[1], p, n)), data)
    oracle = map(mats) do mm
        _, _, _, a, b, k, l, _ = LA.ggsvd!('U', 'V', 'Q', copy(mm.A), copy(mm.Bm))
        (a = sort(a), b = sort(b), k = k, l = l)
    end
    before = Dict(s => fwd(s) for s in ("sggsvd_", "dggsvd_", "cggsvd_", "zggsvd_"))
    PureBLAS.activate()
    try
        @test all(s -> fwd(s) != before[s], keys(before))
        for (i, mm) in enumerate(mats)
            T = data[i][1]
            U, V, Q, a, b, k, l, R = LA.ggsvd!('U', 'V', 'Q', copy(mm.A), copy(mm.Bm))
            tol = real(T) == Float32 ? 1.0f-3 : 1.0e-8
            @test k == oracle[i].k && l == oracle[i].l                         # ranks exact vs OpenBLAS
            @test maximum(abs, sort(a) .- oracle[i].a) < tol                   # generalized singular values
            @test maximum(abs, sort(b) .- oracle[i].b) < tol
            @test opnorm(U'U - I) < tol && opnorm(V'V - I) < tol && opnorm(Q'Q - I) < tol  # unitary
        end
    finally
        PureBLAS.deactivate()
    end
end

# ── OpenBLAS-removal GATE (the contract) ──────────────────────────────────────────────────────────────
# Enumerates EVERY LAPACK symbol LinearAlgebra can ccall (parsed from stdlib lapack.jl) and asserts how
# many still fall through to OpenBLAS after activate() (pointer unchanged vs pre-activate). This is the
# machine-checkable definition of "removed OpenBLAS": the RATCHET only decreases. Tighten _FALLTHROUGH_MAX
# toward 0 as symbols are forwarded; a NEW fallthrough (regression) or exceeding the baseline fails CI.
# North-star: _FALLTHROUGH_MAX == 0 (nothing reaches OpenBLAS). Current baseline is the honest count.
@testitem "LBT: OpenBLAS-removal ratchet (fallthrough count)" tags = [:forward] begin
    using LinearAlgebra, PureBLAS
    const B = LinearAlgebra.BLAS
    _FALLTHROUGH_MAX = 0    # ★ NORTH STAR REACHED ★ Every LinearAlgebra-reachable LAPACK symbol forwards
    # to PureBLAS after activate(). Do not raise. 2026-07-20: 87 → 23 → 15 → 13 → 11 → 3 → 0. The final
    # wave forwarded gesv/posv/lacpy/larfg/larf, gebak/hseqr/trevc, sytrd·hetrd/orgtr·ungtr/ormtr·unmtr,
    # orgqr·ungqr/ormqr·unmqr/ormhr·unmhr, gebrd/bdsqr/bdsdc, syconv + trrfs (s/d/c/z), tgsen (complex +
    # real dtgex2 2×2 swap), gesvx s/d/c/z (geequ+getrf+getrs+gecon+gerfs), c/z bdsqr (robust Demmel–Kahan),
    # and c/s/z ggsvd (rank-deficient GSVD via dggsvp + dtgsja). The cstev_/zstev_ PHANTOM allowlist below
    # is the ONLY exclusion — they are not real LAPACK symbols (see the note). OpenBLAS fallback: REMOVED.
    # PHANTOM allowlist: cstev_/zstev_ appear ONLY in COMMENTED-OUT lines of stdlib lapack.jl (the s/d/c/z
    # macro loop explicitly drops the complex STEV — "Need to rewrite for ZHEEV"); `nm -D libopenblas`
    # confirms NO c/zstev_ export exists. They are regex-scan artifacts with no caller, not real gaps, so
    # they are excluded from the denominator — otherwise north-star == 0 would be unreachable by construction.
    _PHANTOM = Set(["cstev_", "zstev_"])
    lp = joinpath(Sys.STDLIB, "LinearAlgebra", "src", "lapack.jl")
    syms = Set{String}()
    for m in eachmatch(r":([a-z]{4,7}_),", read(lp, String))
        push!(syms, m.captures[1])
    end
    setdiff!(syms, _PHANTOM)
    gf(s) = B.lbt_get_forward(s, B.LBT_INTERFACE_ILP64, B.LBT_F2C_PLAIN)
    pre = Dict(s => gf(s) for s in syms)
    PureBLAS.activate()
    try
        resid = sort([s for s in syms if gf(s) == pre[s]])
        @info "OpenBLAS fallthrough" total = length(syms) forwarded = length(syms) - length(resid) still_openblas = length(resid) residual = join(resid, " ")
        @test length(resid) <= _FALLTHROUGH_MAX
    finally
        PureBLAS.deactivate()
    end
end
