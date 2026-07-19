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
    ref = (mul = A * Bm, gemv = A * x, gemvt = transpose(A) * x, dot = dot(x, y), nrm2 = norm(x),
           symm = Symmetric(A) * Bm, trsm = UpperTriangular(At) \ Bm,
           chol = Matrix(cholesky(copy(SPD)).U), lusol = lu(copy(A)) \ b, chsol = cholesky(copy(SPD)) \ b,
           atsol = transpose(A) \ b, trsol = UpperTriangular(At) \ b,
           ichol = inv(cholesky(copy(SPD))), itri = inv(UpperTriangular(At)),
           qrsol = qr(copy(A)) \ b, qrdet = det(qr(copy(A)).Q),
           svals = svdvals!(copy(A)),   # svdvals! → gesdd (stays OpenBLAS); here just a numeric oracle
           zchol = Matrix(cholesky(Hermitian(copy(HPD))).U), zlusol = lu(copy(Az)) \ zb,
           zqrsol = qr(copy(Az)) \ zb,
           evals = eigvals(Symmetric(A + A')),         # symmetric eigen values (uplo='U' default)
           zevals = eigvals(Hermitian(Az + Az')))      # Hermitian complex eigen values (uplo='U' default)
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
        @test maximum(abs, (A * Bm) .- ref.mul) < 1e-9             # gemm
        @test maximum(abs, (A * x) .- ref.gemv) < 1e-10           # gemv N
        @test maximum(abs, (transpose(A) * x) .- ref.gemvt) < 1e-10 # gemv T
        @test abs(dot(x, y) - ref.dot) < 1e-11                     # dot
        @test abs(norm(x) - ref.nrm2) < 1e-12                      # nrm2
        @test maximum(abs, (Symmetric(A) * Bm) .- ref.symm) < 1e-9 # symm
        @test maximum(abs, (UpperTriangular(At) \ Bm) .- ref.trsm) < 1e-9  # trsm

        # LAPACK that routes to PureBLAS (self-consistent under a mixed backend).
        C = cholesky(copy(SPD)); @test maximum(abs, Matrix(C.U) .- ref.chol) < 1e-8   # potrf
        F = lu(copy(A))                                            # getrf (LAPACK-convention ipiv/factors)
        @test maximum(abs, (F.L * F.U) .- A[F.p, :]) < 1e-9
        # Solves now route to PureBLAS too (getrs/potrs/trtrs) — the solve step of `\`.
        @test fwd("dgetrs_") != getrs_before && fwd("dpotrs_") != potrs_before && fwd("dtrtrs_") != trtrs_before
        @test maximum(abs, (lu(copy(A)) \ b) .- ref.lusol) < 1e-9          # getrf + getrs, both PureBLAS
        @test maximum(abs, (transpose(A) \ b) .- ref.atsol) < 1e-9         # getrs trans='T' (reverse pivots)
        @test maximum(abs, (cholesky(copy(SPD)) \ b) .- ref.chsol) < 1e-9  # potrf + potrs
        @test maximum(abs, (UpperTriangular(At) \ b) .- ref.trsol) < 1e-9   # trtrs
        # Inverses now route (getri/potri/trtri) — inv() uses PureBLAS.
        @test fwd("dgetri_") != getri_before && fwd("dpotri_") != potri_before && fwd("dtrtri_") != trtri_before
        @test maximum(abs, inv(copy(A)) * A - I) < 1e-8                     # getri
        @test maximum(abs, inv(cholesky(copy(SPD))) - ref.ichol) < 1e-9    # potri
        @test maximum(abs, inv(UpperTriangular(At)) - ref.itri) < 1e-9     # trtri
        # gesvd routes (direct LAPACK call).
        S = LA.gesvd!('N', 'N', copy(A))[2]
        @test maximum(abs, S .- ref.svals) < 1e-9
        # gesdd routes → svd()/svdvals (Julia's default SVD path) now use PureBLAS.
        @test fwd("dgesdd_") != gesdd_before
        @test maximum(abs, svdvals(copy(A)) .- ref.svals) < 1e-9
        Fs = svd(copy(A)); @test maximum(abs, Fs.U * Diagonal(Fs.S) * Fs.Vt .- A) < 1e-9

        # Complex LAPACK — potrf/getrf route (self-consistent: standard-convention factors).
        @test fwd("zpotrf_") != zpotrf_before
        @test fwd("zgetrf_") != zgetrf_before
        Cz = cholesky(Hermitian(copy(HPD)))
        @test maximum(abs, Matrix(Cz.U) .- ref.zchol) < 1e-8               # zpotrf
        Fz = lu(copy(Az))
        @test maximum(abs, (Fz.L * Fz.U) .- Az[Fz.p, :]) < 1e-9            # zgetrf (valid factorization)
        @test maximum(abs, (lu(copy(Az)) \ zb) .- ref.zlusol) < 1e-8       # zgetrf + zgetrs, both PureBLAS
        @test fwd("zgeqrt_") != zgeqrt_before && fwd("zgemqrt_") != zgemqrt_before   # complex QR routes
        Qz = qr(copy(Az))
        @test maximum(abs, Matrix(Qz.Q)' * Matrix(Qz.Q) - I) < 1e-9        # zgemqrt: Q unitary
        @test maximum(abs, Matrix(Qz.Q) * Qz.R - Az) < 1e-9                # zgeqrt+zgemqrt: Q·R = Az
        @test maximum(abs, (qr(copy(Az)) \ zb) .- ref.zqrsol) < 1e-8       # complex qr()\b

        # Complex SVD routes → svd()/svdvals(::Matrix{ComplexF64}) now use PureBLAS (zgesdd).
        @test fwd("zgesdd_") != zgesdd_before
        @test maximum(abs, svdvals(copy(Az)) .- svdvals(Az)) < 1e-9         # zgesdd values
        Fz = svd(copy(Az))                                                  # zgesdd U/S/Vᴴ
        @test maximum(abs, Fz.U * Diagonal(Fz.S) * Fz.Vt .- Az) < 1e-9      # reconstruction
        @test maximum(abs, Fz.U' * Fz.U - I) < 1e-9                         # U unitary columns
        @test maximum(abs, Fz.Vt * Fz.Vt' - I) < 1e-9                       # Vᴴ unitary rows

        # Float32 LAPACK via mixed precision (compute F64, store F32) — lu/qr/svd route, correct to F32.
        @test fwd("sgetrf_") != sgetrf_before && fwd("sgeqrt_") != sgeqrt_before && fwd("sgesdd_") != sgesdd_before
        Ff = lu(copy(Af)); @test maximum(abs, (Ff.L * Ff.U) .- Af[Ff.p, :]) < 1f-4   # sgetrf
        Qf = qr(copy(Af))
        @test maximum(abs, Matrix(Qf.Q)' * Matrix(Qf.Q) - I) < 1f-4          # sgeqrt+sgemqrt
        @test maximum(abs, Matrix(Qf.Q) * Qf.R - Af) < 1f-4
        @test maximum(abs, svdvals(copy(Af)) .- f32svals) < 1f-3             # sgesdd (loose F32 tol)

        # QR routes to PureBLAS via geqrt+gemqrt (Julia's qr() is QRCompactWY). geqrf is now forwarded too
        # (its wrapper converts faer τ → LAPACK, so geqrf!+OpenBLAS-orgqr is correct — no more NaN-Q hazard).
        @test fwd("dgeqrt_") != geqrt_before && fwd("dgemqrt_") != gemqrt_before && fwd("dgeqrf_") != geqrf_before
        Q = qr(copy(A))
        @test maximum(abs, Matrix(Q.Q)' * Matrix(Q.Q) - I) < 1e-9   # gemqrt: Q orthonormal
        @test maximum(abs, Matrix(Q.Q) * Q.R - A) < 1e-9            # geqrt+gemqrt: Q·R = A
        @test maximum(abs, (qr(copy(A)) \ b) .- ref.qrsol) < 1e-8   # geqrt+gemqrt+trtrs
        @test abs(det(qr(copy(A)).Q) - ref.qrdet) < 1e-9            # det(Q) reads T's diagonal (LAPACK-exact T)
        # geqrf τ-conversion: geqrf!+OpenBLAS orgqr now gives a valid Q (faer never crosses the ABI).
        A2 = copy(A); _, tau = LA.geqrf!(A2); Qo = LA.orgqr!(copy(A2), tau)
        @test maximum(abs, Qo'Qo - I) < 1e-9

        # Symmetric eigensolver routes: eigen(Symmetric)/eigvals(Symmetric) → dsyevr_ (Julia's default).
        @test fwd("dsyevr_") != syevr_before                              # actually forwarded
        Fe = eigen(Asym)                                                  # dsyevr_ jobz='V' range='A'
        n_ = size(Asymf, 1)
        @test maximum(abs, Fe.values .- ref.evals) < 1e-9                 # eigenvalues match OpenBLAS
        @test opnorm(Asymf * Fe.vectors - Fe.vectors * Diagonal(Fe.values)) < 1e-8 * opnorm(Asymf)
        @test opnorm(Fe.vectors' * Fe.vectors - I) < 1e-9                 # orthonormal vectors
        @test maximum(abs, eigvals(Asym) .- ref.evals) < 1e-9            # eigvals(Symmetric) routes too

        # Hermitian complex eigensolver routes: eigen(Hermitian)/eigvals(Hermitian) → zheevr_ (default).
        @test fwd("zheevr_") != zheevr_before                             # actually forwarded
        Fze = eigen(Aherm)                                               # zheevr_ jobz='V' range='A'
        @test maximum(abs, Fze.values .- ref.zevals) < 1e-9              # eigenvalues match OpenBLAS
        @test opnorm(Ahermf * Fze.vectors - Fze.vectors * Diagonal(Fze.values)) < 1e-8 * opnorm(Ahermf)
        @test opnorm(Fze.vectors' * Fze.vectors - I) < 1e-9             # orthonormal (unitary) vectors
        @test maximum(abs, eigvals(Aherm) .- ref.zevals) < 1e-9        # eigvals(Hermitian) routes too
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
    ref = (lqrec = (F = lq(copy(A)); Matrix(F.L) * Matrix(F.Q)),
           bksol = bunchkaufman(Symmetric(copy(SI))) \ ones(n),
           zbksol = bunchkaufman(Hermitian(copy(HI))) \ ones(ComplexF64, n),
           qpp = (Q = qr(copy(A), ColumnNorm()); Matrix(Q.Q) * Matrix(Q.R) - A[:, Q.p]),
           tallsol = tall \ b,
           trcond = cond(UpperTriangular(copy(U)), 1),
           hess = (H = hessenberg(copy(A)); Matrix(H.Q) * Matrix(H.H) * Matrix(H.Q)' - A))

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
        F = lq(copy(A)); @test maximum(abs, (Matrix(F.L) * Matrix(F.Q)) .- ref.lqrec) < 1e-9
        @test maximum(abs, (bunchkaufman(Symmetric(copy(SI))) \ ones(n)) .- ref.bksol) < 1e-8
        @test maximum(abs, (bunchkaufman(Hermitian(copy(HI))) \ ones(ComplexF64, n)) .- ref.zbksol) < 1e-8
        Q = qr(copy(A), ColumnNorm()); @test maximum(abs, (Matrix(Q.Q) * Matrix(Q.R) - A[:, Q.p]) .- ref.qpp) < 1e-9
        @test maximum(abs, (tall \ b) .- ref.tallsol) < 1e-8       # non-square \ (geqp3 path)
        @test isapprox(cond(UpperTriangular(copy(U)), 1), ref.trcond; rtol = 1e-6)   # trcon
        Hh = hessenberg(copy(A))
        @test maximum(abs, (Matrix(Hh.Q) * Matrix(Hh.H) * Matrix(Hh.Q)' - A) .- ref.hess) < 1e-8
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
        Bp[i, i] = randn(); Bp[i+1, i+1] = Bp[i, i]; Bp[i, i+1] = randn(); Bp[i+1, i] = -Bp[i, i+1]; i += 2
    end
    Qr, _ = qr(randn(n, n)); Acp = Matrix(Qr) * Bp * Matrix(Qr)'

    evsort(v) = sort(v; by = x -> (real(x), imag(x)))
    # OpenBLAS oracles captured BEFORE activate.
    ref = (ev = evsort(eigvals(copy(A))), zev = evsort(eigvals(copy(Az))),
           cpev = evsort(eigvals(copy(Acp))),
           eF = eigen(copy(A)), zeF = eigen(copy(Az)),
           sch = (F = schur(copy(A)); Matrix(F.Z) * Matrix(F.T) * Matrix(F.Z)' - A))

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
        @test maximum(abs, evsort(eigvals(copy(A))) .- ref.ev) < 1e-9          # real eigvals (geevx)
        @test maximum(abs, evsort(eigvals(copy(Az))) .- ref.zev) < 1e-9        # complex eigvals
        @test maximum(abs, evsort(eigvals(copy(Acp))) .- ref.cpev) < 1e-9      # conj-pair spectrum
        Fe = eigen(copy(A))                                                    # real eigen (geevx jobvr='V')
        @test maximum(abs, evsort(Fe.values) .- ref.ev) < 1e-9
        @test maximum(abs, A * Fe.vectors - Fe.vectors * Diagonal(Fe.values)) / (opnorm(A, 1) * n * eps()) < 200
        Fcp = eigen(copy(Acp))                                                 # conj-pair eigen residual
        @test maximum(abs, Acp * Fcp.vectors - Fcp.vectors * Diagonal(Fcp.values)) / (opnorm(Acp, 1) * n * eps()) < 200
        Fze = eigen(copy(Az))                                                  # complex eigen
        @test maximum(abs, Az * Fze.vectors - Fze.vectors * Diagonal(Fze.values)) / (opnorm(Az, 1) * n * eps()) < 200
        Fs = schur(copy(A))                                                    # schur (gees)
        @test maximum(abs, (Matrix(Fs.Z) * Matrix(Fs.T) * Matrix(Fs.Z)' - A) .- ref.sch) < 1e-8
        @test maximum(abs, Matrix(Fs.Z)' * Matrix(Fs.Z) - I) < 1e-10
    finally
        PureBLAS.deactivate()
    end
end
