# Precompile-time contract verification for the SIMDBackend. Included LAST (after every backend op is
# defined) because @verify_strict actually CALLS the ops — the L3 warm-up and @strict calls need
# gemm!/symm!/trmm!/trsm! from gemm.jl/level3.jl, which are included after backend.jl.
#
# @verify_strict is the single backend verifier: it runs TypeContracts.@verify SIMDBackend — method
# existence + declared return types over the whole chain (AbstractBLAS1 → BLAS2 → BLAS3 → LAPACK) —
# AND @strict on each representative call (type-stable + allocation-free), since AbstractBLAS1/2/3 are
# @strict_contracts. The @strict calls self-gate on the `checks_enabled` preference; the main package
# ships fast mode (runtime @allocated / @inferred, no AllocCheck/JET dep) so they fire at PureBLAS's
# OWN precompile. The guard skips the whole block under full-mode environments (e.g. the test project)
# where @strict demands the AllocCheck/JET backend not loaded during PureBLAS's own precompile — there
# the test suite's strictmode dogfood runs the interface + deep static proof at test runtime with the
# backend present. (Trim-safety of the C-ABI entries is covered exhaustively by TrimCheck.@validate.)
# Values live in a `let` — only their types matter, so `ones` (no Random dep).
# potrf! overwrites its argument and throws PosDefException if re-factored — but @strict calls its
# target repeatedly. This probe re-seeds the SPD source (in-place copyto!, no allocation) before each
# factorization, so it's a single 0-alloc, type-stable call @strict can invoke as many times as it likes.
_strict_potrf_probe(bk, Aw, Apd) = (copyto!(Aw, Apd); potrf!(bk, Aw; uplo = 'L'))
# getrf!/geqrf! are 0-alloc through their IN-PLACE (pre-allocated ipiv/τ) forms — the convenience forms
# allocate the pivot/τ output, which is inherent, not a bug. These probes re-seed the source and call the
# in-place kernel; being proper (statically-resolved) functions they also avoid the call-site tuple box a
# dynamically-dispatched call would add. Return nothing so the (A,ipiv,info)/(A,τ) tuple never escapes.
_strict_getrf_probe(Gw, G0, ipiv) = (copyto!(Gw, G0); getrf!(Gw, ipiv); nothing)
_strict_geqrf_probe(Gw, G0, tau) = (copyto!(Gw, G0); geqrf!(Gw, tau); nothing)

if StrictMode.analysis_mode() === :fast || StrictMode.backend_available()
    let bk = DEFAULT_BACKEND, n = 1000, m = 64,
        xd = ones(n), yd = ones(n), xz = ones(ComplexF64, n), yz = ones(ComplexF64, n),
        Ad = ones(m, m), Az = ones(ComplexF64, m, m), um = ones(m), vm = ones(m),
        uz = ones(ComplexF64, m), wz = ones(ComplexF64, m),
        C3 = ones(m, m), A3 = ones(m, m), B3 = ones(m, m), Bt = ones(m, m), At = ones(m, m),
        Cz3 = ones(ComplexF64, m, m), Az3 = ones(ComplexF64, m, m), Bz3 = ones(ComplexF64, m, m),
        # SPD/non-singular source (diagonally dominant: diag m+1, off-diag 1) + a working copy the
        # factorizations overwrite; pre-allocated pivot/τ so the in-place LU/QR kernels are 0-alloc.
        Apd = [i == j ? float(m) + 1.0 : 1.0 for i in 1:m, j in 1:m], Aw = zeros(m, m),
        ipiv = Vector{Int}(undef, m), tau = Vector{Float64}(undef, m)
        # Warm the per-type Level-3 / Cholesky workspace scratch (allocated once on first touch) so the
        # fast-mode runtime @noalloc below sees steady state. All L3 ops are 0-alloc after the offset
        # recursion refactor (rank-k/hemm sub-blocks no longer heap-box), so the whole matrix-matrix set
        # is verified below.
        gemm!(bk, C3, A3, B3); symm!(bk, C3, A3, B3)
        trmm!(bk, Bt, At; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0)
        trsm!(bk, Bt, At; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0)
        syrk!(bk, C3, A3; uplo = 'L', trans = 'N', alpha = 1.0, beta = 1.0)
        herk!(bk, Cz3, Az3; uplo = 'L', trans = 'N', alpha = 1.0, beta = 1.0)
        syr2k!(bk, C3, A3, B3; uplo = 'L', trans = 'N', alpha = 1.0, beta = 1.0)
        her2k!(bk, Cz3, Az3, Bz3; uplo = 'L', trans = 'N', alpha = 1.0 + 0im, beta = 1.0)
        hemm!(bk, Cz3, Az3, Bz3; side = 'L', uplo = 'L', alpha = 1.0 + 0im, beta = 1.0 + 0im)
        copyto!(Aw, Apd); potrf!(bk, Aw; uplo = 'L')
        copyto!(Aw, Apd); getrf!(Aw, ipiv); copyto!(Aw, Apd); geqrf!(Aw, tau)
        @verify_strict SIMDBackend begin
            # ── Level 1 (bandwidth-bound; SIMD real path + generic complex path)
            axpy!(bk, yd, 2.0, xd)
            scal!(bk, 2.0, xd)
            blascopy!(bk, yd, xd)
            swap!(bk, xd, yd)
            dot(bk, xd, yd)
            dotu(bk, xd, yd)
            nrm2(bk, xd)
            asum(bk, xd)
            iamax(bk, xd)
            axpy!(bk, yz, 2.0 + 1.0im, xz)
            dot(bk, xz, yz)
            nrm2(bk, xz)
            # ── Level 2 (dense hot paths; real + complex)
            gemv!(bk, vm, Ad, um; alpha = 2.0, beta = 1.0, trans = 'N')
            gemv!(bk, vm, Ad, um; alpha = 2.0, beta = 1.0, trans = 'T')
            ger!(bk, 1.5, um, vm, Ad)
            symv!(bk, vm, Ad, um)
            hemv!(bk, wz, Az, uz)
            trmv!(bk, Ad, um)
            trsv!(bk, Ad, um)
            # ── Level 3 (all matrix-matrix ops; scratch pre-warmed above; real + complex)
            gemm!(bk, C3, A3, B3)
            symm!(bk, C3, A3, B3)
            hemm!(bk, Cz3, Az3, Bz3; side = 'L', uplo = 'L', alpha = 1.0 + 0im, beta = 1.0 + 0im)
            syrk!(bk, C3, A3; uplo = 'L', trans = 'N', alpha = 1.0, beta = 1.0)
            herk!(bk, Cz3, Az3; uplo = 'L', trans = 'N', alpha = 1.0, beta = 1.0)
            syr2k!(bk, C3, A3, B3; uplo = 'L', trans = 'N', alpha = 1.0, beta = 1.0)
            her2k!(bk, Cz3, Az3, Bz3; uplo = 'L', trans = 'N', alpha = 1.0 + 0im, beta = 1.0)
            trmm!(bk, Bt, At; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0)
            trsm!(bk, Bt, At; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0)
            # ── LAPACK: potrf!/getrf!/geqrf! are 0-alloc (potrf via its own pointer kernels; LU/QR via
            # their in-place pre-allocated-ipiv/τ forms). Via re-seeding probes so repeated @strict calls
            # always factor a fresh source (potrf would otherwise throw PosDefException on its L-output).
            # gesvd! is NOT here — it allocates its U/S/Vᵀ outputs (inherent); it stays interface-@verify'd.
            _strict_potrf_probe(bk, Aw, Apd)
            _strict_getrf_probe(Aw, Apd, ipiv)
            _strict_geqrf_probe(Aw, Apd, tau)
        end
    end
end
