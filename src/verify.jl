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

if StrictMode.analysis_mode() === :fast || StrictMode.backend_available()
    let bk = DEFAULT_BACKEND, n = 1000, m = 64,
        xd = ones(n), yd = ones(n), xz = ones(ComplexF64, n), yz = ones(ComplexF64, n),
        Ad = ones(m, m), Az = ones(ComplexF64, m, m), um = ones(m), vm = ones(m),
        uz = ones(ComplexF64, m), wz = ones(ComplexF64, m),
        C3 = ones(m, m), A3 = ones(m, m), B3 = ones(m, m), Bt = ones(m, m), At = ones(m, m),
        # SPD source (diagonally dominant: diag m+1, off-diag 1) + a working copy potrf! overwrites.
        Apd = [i == j ? float(m) + 1.0 : 1.0 for i in 1:m, j in 1:m], Aw = zeros(m, m)
        # Warm the per-type Level-3 / Cholesky workspace scratch (allocated once on first touch) so the
        # fast-mode runtime @noalloc below sees steady state. Only the 0-alloc ops are verified — the
        # rank-k/hemm family still boxes recursion sub-blocks (see contracts.jl); they'll join once refactored.
        gemm!(bk, C3, A3, B3); symm!(bk, C3, A3, B3)
        trmm!(bk, Bt, At; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0)
        trsm!(bk, Bt, At; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0)
        copyto!(Aw, Apd); potrf!(bk, Aw; uplo = 'L')
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
            # ── Level 3 (the 0-alloc ops; scratch pre-warmed above)
            gemm!(bk, C3, A3, B3)
            symm!(bk, C3, A3, B3)
            trmm!(bk, Bt, At; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0)
            trsm!(bk, Bt, At; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0)
            # ── LAPACK: potrf! (0-alloc). Via the re-seeding probe so repeated @strict calls never
            # re-factor an already-triangular matrix (which would throw PosDefException).
            _strict_potrf_probe(bk, Aw, Apd)
        end
    end
end
