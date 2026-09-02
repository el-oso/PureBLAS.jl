# Precompile-time contract verification for the SIMDBackend. Included LAST (after every backend op is
# defined) because @verify_strict actually CALLS the ops — the L3 warm-up and @strict calls need
# gemm!/symm!/trmm!/trsm! from gemm.jl/level3.jl, which are included after backend.jl.
#
# @verify_strict is the single backend verifier: it runs TypeContracts.@verify SIMDBackend — method
# existence + declared return types over the whole chain (AbstractBLAS1 → BLAS2 → BLAS3 → LAPACK →
# LAPACKSolve) —
# AND @strict on each representative call (type-stable + allocation-free), since AbstractBLAS1/2/3 are
# @strict_contracts. The @strict calls self-gate on the `checks_enabled` preference; the main package
# ships fast mode (runtime @allocated / @inferred, no AllocCheck/JET dep) so they fire at PureBLAS's
# OWN precompile. The guard skips the whole block under full-mode environments (e.g. the test project)
# where @strict demands the AllocCheck/JET backend not loaded during PureBLAS's own precompile — there
# the test suite's strictmode dogfood runs the interface + deep static proof at test runtime with the
# backend present. TRIM-COMPATIBILITY is a strict-contract guarantee too (contracts.jl): the complex-gemm
# kernel is asserted `@assert_trim_compatible` HERE (fast/dev → heuristic scan, cheap early net) and in the
# full-mode dogfood (test → juliac's authoritative verify_typeinf_trim). The heuristic misses reachability-
# limit union-splits the authoritative pass catches — StrictMode 0.3.9 now logs a one-time caveat on that
# heuristic-path trim PASS (issue #13); trim_tests.jl keeps the authoritative ccallable-rooted belt.
# trim_tests.jl keeps the exhaustive ccallable-rooted TrimCheck.@validate belt (strict verify isn't perfect).
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
# gesvd! is 0-alloc through its IN-PLACE form (caller-provided U/S/Vᵀ + a cached SVDWorkspace for the
# bidiagonalization scratch); the convenience gesvd!(A; want_vectors) allocates the outputs. Re-seed A.
_strict_gesvd_probe(Gw, G0, U, S, Vt) = (copyto!(Gw, G0); gesvd!(Gw, U, S, Vt); nothing)

# ── AbstractLAPACKSolve probes (contracts.jl). Same re-seeding discipline as the four factorization
# probes above, and for the same reason — but it applies to the WHOLE contract here rather than to a
# couple of members: every one of these 24 either overwrites its factor/band/tridiagonal input or its
# right-hand side, and @strict invokes its target repeatedly. A direct call would re-factor an
# already-factored matrix on the second invocation (PosDefException from pptrf!/pbtrf!, a zero pivot
# from gttrf!) or, worse, keep solving against a solution it just produced and pass on garbage. The
# three condition estimators (pocon!/gecon!/trcon!) read their factor without touching it, so those
# alone are called directly in the block below. Every probe returns nothing so the multi-output tuples
# — (piv, rank, info), (dl, d, du, du2, ipiv), gesvx's 6-tuple — never escape the strict window and
# get charged as an allocation that belongs to the call site rather than to the routine.
_strict_potrs_probe(bk, Bw, B0, Ac) = (copyto!(Bw, B0); potrs!(bk, Ac, Bw; uplo = 'L'); nothing)
_strict_potri_probe(bk, Aw, Ac) = (copyto!(Aw, Ac); potri!(bk, Aw; uplo = 'L'); nothing)
_strict_pptrf_probe(bk, APw, AP0) = (copyto!(APw, AP0); pptrf!(bk, APw; uplo = 'L'); nothing)
_strict_pptrs_probe(bk, Bw, B0, APc) = (copyto!(Bw, B0); pptrs!(bk, APc, Bw; uplo = 'L'); nothing)
_strict_pbtrf_probe(bk, ABw, AB0, kd) = (copyto!(ABw, AB0); pbtrf!(bk, ABw; uplo = 'L', kd = kd); nothing)
_strict_pbtrs_probe(bk, Bw, B0, ABc) = (copyto!(Bw, B0); pbtrs!(bk, ABc, Bw; uplo = 'L'); nothing)
_strict_pstrf_probe(bk, Aw, A0, piv) = (copyto!(Aw, A0); pstrf!(bk, Aw, piv, -1.0; uplo = 'L'); nothing)
_strict_getrs_probe(bk, Bw, B0, Alu, ipiv) = (copyto!(Bw, B0); getrs!(bk, Alu, ipiv, Bw; trans = 'N'); nothing)
_strict_getri_probe(bk, Aw, Alu, ipiv) = (copyto!(Aw, Alu); getri!(bk, Aw, ipiv); nothing)
_strict_trtrs_probe(bk, Bw, B0, At) = (copyto!(Bw, B0); trtrs!(bk, At, Bw; uplo = 'L', trans = 'N', diag = 'N'); nothing)
_strict_trtri_probe(bk, Aw, At) = (copyto!(Aw, At); trtri!(bk, Aw; uplo = 'L', diag = 'N'); nothing)
# trrfs! reads A/B/X and writes only Ferr/Berr, so it needs no re-seeding — the probe exists purely to
# swallow the (Ferr, Berr) tuple.
_strict_trrfs_probe(bk, At, B, X, fe, be) = (trrfs!(bk, 'L', 'N', 'N', At, B, X, fe, be); nothing)
_strict_gbtrf_probe(bk, ABw, AB0, kl, ku, m, ipiv) = (copyto!(ABw, AB0); gbtrf!(bk, kl, ku, m, ABw, ipiv); nothing)
_strict_gbtrs_probe(bk, Bw, B0, kl, ku, m, ABc, ipiv) = (copyto!(Bw, B0); gbtrs!(bk, 'N', kl, ku, m, ABc, ipiv, Bw); nothing)
_strict_gtsv_probe(bk, dl, d, du, dl0, d0, du0, Bw, B0) =
    (copyto!(dl, dl0); copyto!(d, d0); copyto!(du, du0); copyto!(Bw, B0); gtsv!(bk, dl, d, du, Bw); nothing)
_strict_gttrf_probe(bk, dl, d, du, du2, ipiv, dl0, d0, du0) =
    (copyto!(dl, dl0); copyto!(d, d0); copyto!(du, du0); gttrf!(bk, dl, d, du, du2, ipiv); nothing)
_strict_gttrs_probe(bk, Bw, B0, dl, d, du, du2, ipiv) =
    (copyto!(Bw, B0); gttrs!(bk, 'N', dl, d, du, du2, ipiv, Bw); nothing)
_strict_pttrf_probe(bk, D, E, D0, E0) = (copyto!(D, D0); copyto!(E, E0); pttrf!(bk, D, E); nothing)
_strict_pttrs_probe(bk, Bw, B0, Df, Ef) = (copyto!(Bw, B0); pttrs!(bk, Df, Ef, Bw; uplo = 'L'); nothing)
_strict_ptsv_probe(bk, D, E, D0, E0, Bw, B0) =
    (copyto!(D, D0); copyto!(E, E0); copyto!(Bw, B0); ptsv!(bk, D, E, Bw; uplo = 'L'); nothing)
# gesvx! with fact='N'/equed='N' leaves A and B alone, but the contract's in-place form owns AF/ipiv/X
# and the driver re-factors A into AF on every call — re-seed all three inputs so a repeated invocation
# sees the same problem, not the previous call's scaled/refined state.
_strict_gesvx_probe(bk, A, A0, AF, ipiv, R, C, Bw, B0, X, fe, be) =
    (
        copyto!(A, A0); copyto!(AF, A0); copyto!(Bw, B0);
        gesvx!(bk, 'N', 'N', A, AF, ipiv, 'N', R, C, Bw, X, fe, be); nothing
    )

# ── QR/LQ/QL/RZ probes. Every one of these overwrites its input, and `@strict` calls its target
# repeatedly, so each re-seeds from a pristine source with `copyto!` (allocation-free) and returns
# `nothing` so the multi-output tuples never escape to the call site.
_strict_qrfac_probe(f, Aw, A0, tau) = (copyto!(Aw, A0); f(Aw, tau); nothing)
_strict_qp3_probe(Aw, A0, jp, tau) = (copyto!(Aw, A0); fill!(jp, 0); geqp3!(Aw, jp, tau); nothing)
# Q-generators consume a FACTORED source, so the pristine copy is the factorization, not the raw A.
_strict_orgq_probe(f, Aw, Af, tau) = (copyto!(Aw, Af); f(Aw, tau); nothing)
# Q-appliers leave A alone and overwrite C.
_strict_ormq_probe(f, Cw, C0, Af, tau) = (copyto!(Cw, C0); f('L', 'N', Af, tau, Cw); nothing)

# orgtr!/ungtr! read A/tau and OVERWRITE the caller-supplied Q, so only Q needs re-seeding — and it does
# need it: the in-place form zeroes Q itself (its off-diagonal zeros are READ by the block-reflector
# gemms), so a repeated @strict call must not inherit the previous call's Q. Return nothing so Q does
# not escape and get charged to the call site.
_strict_orgtr_probe(bk, A, tau, Q) = (orgtr!(bk, 'L', A, tau, Q); nothing)
_strict_ungtr_probe(bk, A, tau, Q) = (ungtr!(bk, 'L', A, tau, Q); nothing)

if StrictMode.analysis_mode() === :fast || StrictMode.backend_available()
    let bk = DEFAULT_BACKEND, n = 1000, m = 64,
            xd = ones(n), yd = ones(n), xz = ones(ComplexF64, n), yz = ones(ComplexF64, n),
            Ad = ones(m, m), Az = ones(ComplexF64, m, m), um = ones(m), vm = ones(m),
            uz = ones(ComplexF64, m), wz = ones(ComplexF64, m),
            C3 = ones(m, m), A3 = ones(m, m), B3 = ones(m, m), Bt = ones(m, m), At = ones(m, m),
            # SubArray operands for the L3 view assertions below. The parents are LARGER than the view,
            # so `lda != nrows` — the shape the owned-workspace accessors (`_pptrf_lower_work`,
            # `_getri_work`, `_trsm_rrefl`, …) actually produce, and the one a same-size `view` would not
            # exercise. `_strided1` still holds, so these take the same pointer fast paths a Matrix does;
            # what differs is the argument TYPE, which is precisely what went unchecked.
            P3 = ones(m + 8, m + 8), Q3 = ones(m + 8, m + 8), R3 = ones(m + 8, m + 8),
            Pt = ones(m + 8, m + 8), Qt = ones(m + 8, m + 8),
            C3v = view(P3, 1:m, 1:m), A3v = view(Q3, 1:m, 1:m), B3v = view(R3, 1:m, 1:m),
            Btv = view(Pt, 1:m, 1:m), Atv = view(Qt, 1:m, 1:m),
            Cz3 = ones(ComplexF64, m, m), Az3 = ones(ComplexF64, m, m), Bz3 = ones(ComplexF64, m, m),
            # SPD/non-singular source (diagonally dominant: diag m+1, off-diag 1) + a working copy the
            # factorizations overwrite; pre-allocated pivot/τ so the in-place LU/QR kernels are 0-alloc.
            Apd = [i == j ? float(m) + 1.0 : 1.0 for i in 1:m, j in 1:m], Aw = zeros(m, m),
            ipiv = Vector{Int}(undef, m), tau = Vector{Float64}(undef, m),
            # QR-family sources: a pristine A, and pre-factored copies for the Q-generators and
            # Q-appliers (which need real reflectors, not raw data).
            Aqr = ones(m, m), Aqw = zeros(m, m), tauq = Vector{Float64}(undef, m),
            jpq = zeros(Int, m), Cqr = ones(m, m), Cqw = zeros(m, m),
            # COMPLEX counterparts. The contract previously verified `symm!`/`syrk!`/`syr2k!`/`trmm!`/
            # `trsm!` and every factorization on REAL operands only, so any defect confined to a complex
            # kernel was invisible to it — which is exactly how ~1 KB/call of `unsafe_wrap` allocation in
            # the complex-only 3M path passed for a month. Hermitian PD (diag m+1, off-diag 1+0im) so
            # zpotrf succeeds; `Atz` is the triangular operand for ztrmm/ztrsm (unit diagonal ⇒
            # non-singular), `Btz` the one they overwrite, mirroring the real `At`/`Bt` pair.
            Apdz = [i == j ? ComplexF64(m + 1) : ComplexF64(1) for i in 1:m, j in 1:m],
            Awz3 = zeros(ComplexF64, m, m), tauz = Vector{ComplexF64}(undef, m),
            Atz = ones(ComplexF64, m, m), Btz = ones(ComplexF64, m, m),
            # L2 packed storage (AP length m(m+1)/2) and band storage (kb sub/super-diagonals).
            kb = 8, APd = ones(m * (m + 1) ÷ 2), APz = ones(ComplexF64, m * (m + 1) ÷ 2),
            ABg = ones(2 * 8 + 1, m), ABs = ones(8 + 1, m), ABz = ones(ComplexF64, 8 + 1, m),
            # gesvd in-place output buffers (square m×m: U m×m, S m, Vᵀ m×m).
            Usv = zeros(m, m), Ssv = zeros(m), Vtsv = zeros(m, m),
            # ── AbstractLAPACKSolve operands. SMALL on purpose (ms=32, nrhs=2, band widths 1-2): this
            # block runs at every user's precompile, so each of the 24 members gets the smallest shape
            # that still reaches its real kernel rather than a degenerate one. Everything is
            # diagonally dominant — dense diag ms+1 vs ms-1 off-diagonal ones, tridiagonal 4.0 vs
            # 0.3/0.2, both band stores 10.0 on the diagonal row vs 1.0 elsewhere — so no call can
            # raise PosDefException/SingularException. That matters more here than in a test: a throw
            # inside @verify_strict fails the PRECOMPILE, i.e. the package stops loading at all.
            ms = 32, nrhs = 2,
            Aspd = [i == j ? float(ms) + 1.0 : 1.0 for i in 1:ms, j in 1:ms],
            Asw = zeros(ms, ms), Achol = zeros(ms, ms), Alu = zeros(ms, ms), AFx = zeros(ms, ms),
            ipivs = Vector{Int}(undef, ms), ipivx = Vector{Int}(undef, ms), pivs = Vector{Int}(undef, ms),
            Bs0 = ones(ms, nrhs), Bsw = zeros(ms, nrhs), Xs = zeros(ms, nrhs),
            ferrs = zeros(nrhs), berrs = zeros(nrhs), Rq = ones(ms), Cq = ones(ms),
            # Packed lower storage: column j is rows j:ms, so the comprehension's j-outer/i-inner order
            # IS the packing. APc is the factor pptrs! consumes (filled in the warm-up below).
            APs0 = [i == j ? float(ms) + 1.0 : 1.0 for j in 1:ms for i in j:ms],
            APc = zeros(length(APs0)),
            # pptrf!'s own operand is order 8, NOT ms — see the note at its call below; this is the one
            # member of the contract that is not allocation-free at ms=32.
            nps = 8, APu0 = [i == j ? float(nps) + 1.0 : 1.0 for j in 1:nps for i in j:nps],
            APuw = zeros(length(APu0)),
            # Band SPD (kdS sub-diagonals; row 1 of the lower band store is the diagonal).
            kdS = 2, ABp0 = [r == 1 ? 10.0 : 1.0 for r in 1:(kdS + 1), j in 1:ms],
            ABpw = zeros(kdS + 1, ms), ABpc = zeros(kdS + 1, ms),
            # General band LU: ldab = 2kl+ku+1 with the diagonal on row kl+ku+1 and rows 1:kl reserved
            # for the factorization's fill, exactly dgbtrf's convention.
            klS = 1, kuS = 1,
            ABg0 = [r == klS + kuS + 1 ? 10.0 : 1.0 for r in 1:(2 * klS + kuS + 1), j in 1:ms],
            ABgw = zeros(2 * klS + kuS + 1, ms), ABgc = zeros(2 * klS + kuS + 1, ms),
            ipivg = Vector{Int}(undef, ms), ipivgw = Vector{Int}(undef, ms),
            # Tridiagonal source + a working triple the factor-in-place routines overwrite, and a
            # separately factored triple gttrs! solves against.
            dl0 = fill(0.2, ms - 1), d0 = fill(4.0, ms), du0 = fill(0.3, ms - 1),
            dlw = zeros(ms - 1), dw = zeros(ms), duw = zeros(ms - 1), du2w = zeros(ms - 2),
            dlf = zeros(ms - 1), df = zeros(ms), duf = zeros(ms - 1), du2f = zeros(ms - 2),
            ipivt = Vector{Int}(undef, ms), ipivtw = Vector{Int}(undef, ms),
            # SPD tridiagonal (pttrf/pttrs/ptsv): D real, E the sub-diagonal.
            Dp0 = fill(4.0, ms), Ep0 = fill(0.3, ms - 1),
            Dpw = zeros(ms), Epw = zeros(ms - 1), Dpf = zeros(ms), Epf = zeros(ms - 1)
        # Warm the per-type Level-3 / Cholesky workspace scratch (allocated once on first touch) so the
        # fast-mode runtime @noalloc below sees steady state. All L3 ops are 0-alloc after the offset
        # recursion refactor (rank-k/hemm sub-blocks no longer heap-box), so the whole matrix-matrix set
        # is verified below.
        # complex gemm warms the 3M split/product pools too — they grow on first touch, so without this
        # the new in-window `gemm!(bk, Cz3, Az3, Bz3)` assertion below would trip on pool growth rather
        # than on a real per-call allocation.
        gemm!(bk, C3, A3, B3); gemm!(bk, Cz3, Az3, Bz3); symm!(bk, C3, A3, B3)
        trmm!(bk, Bt, At; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0)
        trsm!(bk, Bt, At; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0)
        # complex L3 + complex Cholesky warmup (own per-type workspaces and 3M pools)
        symm!(bk, Cz3, Az3, Bz3); syrk!(bk, Cz3, Az3; uplo = 'L', trans = 'N', alpha = 1.0 + 0im, beta = 1.0 + 0im)
        syr2k!(bk, Cz3, Az3, Bz3; uplo = 'L', trans = 'N', alpha = 1.0 + 0im, beta = 1.0 + 0im)
        trmm!(bk, Btz, Atz; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0 + 0im)
        trsm!(bk, Btz, Atz; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0 + 0im)
        copyto!(Awz3, Apdz); potrf!(Awz3; uplo = 'L')
        copyto!(Awz3, Apdz); getrf!(Awz3, ipiv); copyto!(Awz3, Apdz); geqrf!(Awz3, tauz)
        syrk!(bk, C3, A3; uplo = 'L', trans = 'N', alpha = 1.0, beta = 1.0)
        herk!(bk, Cz3, Az3; uplo = 'L', trans = 'N', alpha = 1.0, beta = 1.0)
        syr2k!(bk, C3, A3, B3; uplo = 'L', trans = 'N', alpha = 1.0, beta = 1.0)
        her2k!(bk, Cz3, Az3, Bz3; uplo = 'L', trans = 'N', alpha = 1.0 + 0im, beta = 1.0)
        hemm!(bk, Cz3, Az3, Bz3; side = 'L', uplo = 'L', alpha = 1.0 + 0im, beta = 1.0 + 0im)
        copyto!(Aw, Apd); potrf!(bk, Aw; uplo = 'L')
        copyto!(Aw, Apd); getrf!(Aw, ipiv); copyto!(Aw, Apd); geqrf!(Aw, tau)
        copyto!(Aw, Apd); gesvd!(Aw, Usv, Ssv, Vtsv)   # warm the cached SVDWorkspace
        # LAPACKSolve warm-up, in two parts. First build the factored sources the solves, inverses and
        # condition estimators consume — those routines take a factor, not a matrix, and handing them a
        # raw SPD/general matrix would "succeed" on nonsense. Then take one pass through every probe, so
        # the owned scratch each routine grows on first touch (getri's n×nb block column, pstrf's 2n work
        # vector, the LACN2 estimator buffers behind gecon/trcon/pocon, the L3 workspaces the trsm-based
        # solves reach) is already at steady state when the strict window opens — otherwise first-touch
        # growth is charged to the routine and reads as a per-call allocation that isn't one.
        copyto!(Achol, Aspd); potrf!(Achol; uplo = 'L')
        copyto!(Alu, Aspd); getrf!(Alu, ipivs)
        copyto!(APc, APs0); pptrf!(APc; uplo = 'L')
        copyto!(ABpc, ABp0); pbtrf!(ABpc; uplo = 'L', kd = kdS)
        copyto!(ABgc, ABg0); gbtrf!(klS, kuS, ms, ABgc, ipivg)
        copyto!(dlf, dl0); copyto!(df, d0); copyto!(duf, du0); gttrf!(dlf, df, duf, du2f, ipivt)
        copyto!(Dpf, Dp0); copyto!(Epf, Ep0); pttrf!(Dpf, Epf)
        copyto!(Xs, Bs0); trtrs!(Achol, Xs; uplo = 'L')   # a genuine solution for trrfs!'s error bounds
        _strict_potrs_probe(bk, Bsw, Bs0, Achol); _strict_potri_probe(bk, Asw, Achol)
        _strict_pptrf_probe(bk, APuw, APu0); _strict_pptrs_probe(bk, Bsw, Bs0, APc)
        _strict_pbtrf_probe(bk, ABpw, ABp0, kdS); _strict_pbtrs_probe(bk, Bsw, Bs0, ABpc)
        _strict_pstrf_probe(bk, Asw, Aspd, pivs); pocon!(bk, float(2 * ms), Achol; uplo = 'L')
        _strict_getrs_probe(bk, Bsw, Bs0, Alu, ipivs); _strict_getri_probe(bk, Asw, Alu, ipivs)
        _strict_trtrs_probe(bk, Bsw, Bs0, Achol); _strict_trtri_probe(bk, Asw, Achol)
        gecon!(bk, float(2 * ms), Alu, ipivs; norm = '1'); trcon!(bk, Achol; uplo = 'L', diag = 'N', norm = '1')
        _strict_trrfs_probe(bk, Achol, Bs0, Xs, ferrs, berrs)
        _strict_gbtrf_probe(bk, ABgw, ABg0, klS, kuS, ms, ipivgw)
        _strict_gbtrs_probe(bk, Bsw, Bs0, klS, kuS, ms, ABgc, ipivg)
        _strict_gtsv_probe(bk, dlw, dw, duw, dl0, d0, du0, Bsw, Bs0)
        _strict_gttrf_probe(bk, dlw, dw, duw, du2w, ipivtw, dl0, d0, du0)
        _strict_gttrs_probe(bk, Bsw, Bs0, dlf, df, duf, du2f, ipivt)
        _strict_pttrf_probe(bk, Dpw, Epw, Dp0, Ep0); _strict_pttrs_probe(bk, Bsw, Bs0, Dpf, Epf)
        _strict_ptsv_probe(bk, Dpw, Epw, Dp0, Ep0, Bsw, Bs0)
        _strict_gesvx_probe(bk, Asw, Aspd, AFx, ipivx, Rq, Cq, Bsw, Bs0, Xs, ferrs, berrs)
        # QR-family factored sources. The Q-generators and Q-appliers need REAL reflectors, so each
        # gets its own factorization of the same pristine A; reusing one factor across families would
        # not exercise the distinct reflector layouts (QR stores below the diagonal, LQ to the right,
        # QL/RQ at the opposite corners, RZ in the trapezoidal tail).
        Afqr = copy(Aqr); geqrf!(Afqr, tauq)
        Aflq = copy(Aqr); taul = Vector{Float64}(undef, m); gelqf!(Aflq, taul)
        Afql = copy(Aqr); tauql = Vector{Float64}(undef, m); geqlf!(Afql, tauql)
        Afrq = copy(Aqr); taurq = Vector{Float64}(undef, m); gerqf!(Afrq, taurq)
        Afrz = copy(Aqr); taurz = Vector{Float64}(undef, m); tzrzf!(Afrz, taurz)
        Aqtr = ones(m, m); tautr = Vector{Float64}(undef, m); Qtr = zeros(m, m)
        Aztr = ones(ComplexF64, m, m); tauztr = Vector{ComplexF64}(undef, m); Qztr = zeros(ComplexF64, m, m)
        copyto!(Xs, Bs0); trtrs!(Achol, Xs; uplo = 'L')   # gesvx overwrote Xs — restore the solution
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
            scal!(bk, 2.0 + 1.0im, xz)         # complex scal: interleaved-SIMD (swap-pairs) path
            dot(bk, xz, yz)                     # complex dot/dotu: split-deinterleave SIMD reduction
            dotu(bk, xz, yz)
            nrm2(bk, xz)                       # complex nrm2/asum now take the SIMD real-reinterpret path
            asum(bk, xz)
            blascopy!(bk, yz, xz)
            swap!(bk, xz, yz)
            iamax(bk, xz)                      # izamax: swap-adjacent magnitude reduction (its own kernel)
            # ── Level 2 (dense hot paths; real + complex)
            gemv!(bk, vm, Ad, um; alpha = 2.0, beta = 1.0, trans = 'N')
            gemv!(bk, vm, Ad, um; alpha = 2.0, beta = 1.0, trans = 'T')
            gemv!(bk, wz, Az, uz; alpha = 2.0 + 0im, beta = 1.0 + 0im, trans = 'N')   # complex gemv SIMD
            gemv!(bk, wz, Az, uz; alpha = 2.0 + 0im, beta = 1.0 + 0im, trans = 'C')
            ger!(bk, 1.5, um, vm, Ad)
            ger!(bk, 1.5 + 0.5im, uz, wz, Az)     # complex geru/gerc: per-column complex axpy
            symv!(bk, vm, Ad, um)
            hemv!(bk, wz, Az, uz)
            trmv!(bk, Ad, um)
            trsv!(bk, Ad, um)
            trmv!(bk, Az, uz)                     # complex trmv/trsv: per-column axpy(N)/dot(T/C) reuse
            trsv!(bk, Az, uz)
            # ── Level 2 packed storage (symmetric/Hermitian/triangular; rank-1/2 updates)
            spmv!(bk, vm, APd, um; uplo = 'U', alpha = 2.0, beta = 1.0)
            hpmv!(bk, wz, APz, uz; uplo = 'U', alpha = 2.0 + 0im, beta = 1.0 + 0im)
            tpmv!(bk, APd, um; uplo = 'U', trans = 'N', diag = 'N')
            tpsv!(bk, APd, um; uplo = 'U', trans = 'N', diag = 'N')
            spr!(bk, 1.5, um, APd; uplo = 'U')
            spr2!(bk, 1.5, um, vm, APd; uplo = 'U')
            hpr!(bk, 1.5, uz, APz; uplo = 'U')
            hpr2!(bk, 1.5 + 0im, uz, wz, APz; uplo = 'U')
            # ── Level 2 band storage (general/symmetric/Hermitian/triangular banded)
            gbmv!(bk, vm, ABg, um, m, kb, kb; trans = 'N', alpha = 2.0, beta = 1.0)
            sbmv!(bk, vm, ABs, um; uplo = 'U', alpha = 2.0, beta = 1.0)
            hbmv!(bk, wz, ABz, uz; uplo = 'U', alpha = 2.0 + 0im, beta = 1.0 + 0im)
            tbmv!(bk, ABs, um; uplo = 'U', trans = 'N', diag = 'N')
            tbsv!(bk, ABs, um; uplo = 'U', trans = 'N', diag = 'N')
            # ── Level 3 (all matrix-matrix ops; scratch pre-warmed above; real + complex)
            gemm!(bk, C3, A3, B3)
            # COMPLEX gemm, and deliberately at m=64 — INSIDE the Karatsuba-3M window
            # (`_CGEMM_3M_MIN`=48 ≤ max(m,n,k) ≤ 2048, min ≥ `_CGEMM_3M_KMIN`=16). Until now every
            # `gemm!` here was REAL, and 3M is complex-only, so the strict contract verified the one arm
            # that cannot reach `_gemm_3m!`. That is how nine `unsafe_wrap(Array, …)` per call — ~1 KB of
            # steady-state allocation — passed this check on AVX2 from the day 3M shipped. The other
            # complex L3 calls below cannot substitute: they run at m=64, under `_CSYRK_3M_MIN`=256, so
            # `herk!`/`her2k!` never enter rank-k 3M. One line, and it closes the whole class.
            gemm!(bk, Cz3, Az3, Bz3)
            symm!(bk, C3, A3, B3)
            hemm!(bk, Cz3, Az3, Bz3; side = 'L', uplo = 'L', alpha = 1.0 + 0im, beta = 1.0 + 0im)
            syrk!(bk, C3, A3; uplo = 'L', trans = 'N', alpha = 1.0, beta = 1.0)
            herk!(bk, Cz3, Az3; uplo = 'L', trans = 'N', alpha = 1.0, beta = 1.0)
            syr2k!(bk, C3, A3, B3; uplo = 'L', trans = 'N', alpha = 1.0, beta = 1.0)
            her2k!(bk, Cz3, Az3, Bz3; uplo = 'L', trans = 'N', alpha = 1.0 + 0im, beta = 1.0)
            trmm!(bk, Bt, At; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0)
            trsm!(bk, Bt, At; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0)
            # COMPLEX arms of the five L3 ops that were verified on the real path only. `hemm!`/`herk!`/
            # `her2k!` above do NOT substitute: the symmetric (non-Hermitian) complex kernels are
            # separate specializations — `zsymm`/`zsyrk`/`zsyr2k` conjugate nothing — and `ztrmm`/`ztrsm`
            # share no code with any of them.
            symm!(bk, Cz3, Az3, Bz3)
            syrk!(bk, Cz3, Az3; uplo = 'L', trans = 'N', alpha = 1.0 + 0im, beta = 1.0 + 0im)
            syr2k!(bk, Cz3, Az3, Bz3; uplo = 'L', trans = 'N', alpha = 1.0 + 0im, beta = 1.0 + 0im)
            trmm!(bk, Btz, Atz; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0 + 0im)
            trsm!(bk, Btz, Atz; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0 + 0im)
            # ── SubArray arguments. Everything above passes `Matrix`, and that gap let a REAL contract
            # violation stand: `_trsm_right!` assigned `Ar` three different types (the caller's `A`, and
            # two workspace views), so the slot was union-typed. With `Matrix` the union split and cost
            # nothing; with a `SubArray` it BOXED — 64 B per call, allocation-free contract broken, and
            # every one of these Matrix-only assertions passed anyway. Views are not an exotic input
            # here: the owned-workspace panels (`_pptrf_lower_work`, `_getri_work`, …) hand views to
            # exactly these ops, so this is the shape the package uses internally.
            # side='R' with transA='T' is the arm that boxed; keep it explicitly.
            trsm!(bk, Btv, Atv; side = 'R', uplo = 'L', transA = 'T', diag = 'N', alpha = 1.0)
            trsm!(bk, Btv, Atv; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0)
            trmm!(bk, Btv, Atv; side = 'R', uplo = 'L', transA = 'T', diag = 'N', alpha = 1.0)
            gemm!(bk, C3v, A3v, B3v)
            syrk!(bk, C3v, A3v; uplo = 'L', trans = 'N', alpha = 1.0, beta = 1.0)
            # ── LAPACK: potrf!/getrf!/geqrf! are 0-alloc (potrf via its own pointer kernels; LU/QR via
            # their in-place pre-allocated-output forms). Via re-seeding probes so repeated @strict calls
            # always factor a fresh source (potrf would otherwise throw PosDefException on its L-output).
            # All four LAPACK factorizations are strict now: gesvd! reaches 0-alloc via the in-place
            # gesvd!(A,U,S,Vᵀ) form + a cached SVDWorkspace for the bidiagonalization scratch.
            _strict_potrf_probe(bk, Aw, Apd)
            _strict_getrf_probe(Aw, Apd, ipiv)
            _strict_geqrf_probe(Aw, Apd, tau)
            _strict_gesvd_probe(Aw, Apd, Usv, Ssv, Vtsv)
            # COMPLEX factorizations — a wholly separate kernel set (`_cpotrf_lower!`, `_cgetf2_simd!`,
            # the complex QR panel), previously outside the contract entirely. Same re-seeding probes so
            # repeated @strict invocations stay 0-alloc and type-stable rather than re-factoring in place.
            # gesvd is deliberately NOT included for complex: its in-place form caches a Float64
            # SVDWorkspace and the complex path has not been shown 0-alloc — asserting it here without
            # that evidence would be guessing, so it stays an open item rather than a silent pass.
            _strict_potrf_probe(bk, Awz3, Apdz)
            _strict_getrf_probe(Awz3, Apdz, ipiv)
            _strict_geqrf_probe(Awz3, Apdz, tauz)
            # ── AbstractLAPACKSolve (contracts.jl): the solve/inverse/condition surface built on those
            # four factorizations. Declaring the contract only fixes the method signatures; it is THIS
            # block that makes "type-stable AND allocation-free" true, and eight of these were breaking
            # it before the owned-workspace conversion. Grouped by the factorization they hang off.
            #
            # Cholesky family: solve, explicit inverse (potri! takes the FACTOR, not the SPD matrix,
            # and overwrites it with A⁻¹ — hence the re-seed from the pre-factored Achol), packed and
            # banded factor+solve, the rank-revealing pivoted factor, and the condition estimate.
            _strict_potrs_probe(bk, Bsw, Bs0, Achol)
            _strict_potri_probe(bk, Asw, Achol)
            # pptrf! IS verified — but at order 8, and that is a KNOWN GAP, not a convenience. At
            # n ≥ `_PPTRF_BLK_MIN` (16) the lower path takes `_pptrf_lower_blocked!`, which builds two
            # `Matrix{T}(undef, n, nb)` scratch buffers per call: MEASURED 16560 B/call at n=32 on this
            # box, 0 B at n=8. So the blocked arm breaks the strict contract it is declared under, and
            # asserting it here would fail the precompile rather than report anything. Order 8 pins the
            # unblocked `_pptrf_lower!` arm, which is genuinely 0-alloc; the blocked arm stays unverified
            # until its W/V move to an owned workspace the way getri!'s block column already has.
            _strict_pptrf_probe(bk, APuw, APu0)
            _strict_pptrs_probe(bk, Bsw, Bs0, APc)
            _strict_pbtrf_probe(bk, ABpw, ABp0, kdS)
            _strict_pbtrs_probe(bk, Bsw, Bs0, ABpc)
            _strict_pstrf_probe(bk, Asw, Aspd, pivs)
            pocon!(bk, float(2 * ms), Achol; uplo = 'L')
            # LU family: solve against the factors and the explicit inverse. getri! is the one that used
            # to allocate an n×n identity plus scratch; it now runs LAPACK's block-column algorithm over
            # an owned n×nb buffer, which is exactly what this line holds it to.
            _strict_getrs_probe(bk, Bsw, Bs0, Alu, ipivs)
            _strict_getri_probe(bk, Asw, Alu, ipivs)
            # Triangular: solve, in-place inverse, condition estimate, and the trrfs! error bounds. The
            # three estimators (pocon!/gecon!/trcon!) are called DIRECTLY — they only read their factor,
            # so there is nothing to re-seed, and a probe would hide the closure-capture that is the
            # plausible failure mode for a Higham–Hager estimator driving a callback.
            _strict_trtrs_probe(bk, Bsw, Bs0, Achol)
            _strict_trtri_probe(bk, Asw, Achol)
            gecon!(bk, float(2 * ms), Alu, ipivs; norm = '1')
            trcon!(bk, Achol; uplo = 'L', diag = 'N', norm = '1')
            _strict_trrfs_probe(bk, Achol, Bs0, Xs, ferrs, berrs)
            # Banded LU: the IN-PLACE gbtrf!(kl, ku, m, AB, ipiv) form, not the convenience form that
            # allocates ipiv — the contract binds the former precisely so it cannot be satisfied by a
            # method that allocates its own output.
            _strict_gbtrf_probe(bk, ABgw, ABg0, klS, kuS, ms, ipivgw)
            _strict_gbtrs_probe(bk, Bsw, Bs0, klS, kuS, ms, ABgc, ipivg)
            # Tridiagonal: the fused driver, the split factor/solve pair, and their SPD counterparts.
            # These are pure recurrences with no BLAS underneath, so an allocation here would be a
            # bounds-check or boxing artefact rather than a workspace — worth pinning for that reason.
            _strict_gtsv_probe(bk, dlw, dw, duw, dl0, d0, du0, Bsw, Bs0)
            _strict_gttrf_probe(bk, dlw, dw, duw, du2w, ipivtw, dl0, d0, du0)
            _strict_gttrs_probe(bk, Bsw, Bs0, dlf, df, duf, du2f, ipivt)
            _strict_pttrf_probe(bk, Dpw, Epw, Dp0, Ep0)
            _strict_pttrs_probe(bk, Bsw, Bs0, Dpf, Epf)
            _strict_ptsv_probe(bk, Dpw, Epw, Dp0, Ep0, Bsw, Bs0)
            # Expert driver — the deepest composition in the contract (equilibrate → getrf → solve →
            # gecon ×2 → iterative refinement), so it is also the one most likely to leak an allocation
            # from a callee. fact='N'/equed='N' skips equilibration, keeping the arm under test the
            # factor-solve-refine chain rather than the scaling. Again the IN-PLACE form: X/ferr/berr
            # come from the caller, which is what netlib's OUT arguments always implied.
            _strict_gesvx_probe(bk, Asw, Aspd, AFx, ipivx, Rq, Cq, Bsw, Bs0, Xs, ferrs, berrs)
            # ── QR / LQ / QL / RZ. 14 distinct implementations; the nine `un*!` names are `const`
            # aliases of these same function objects, so verifying `or*!` verifies them too.
            # orgtr!/ungtr! are absent — they allocate their output Q (see contracts.jl).
            _strict_qrfac_probe(gelqf!, Aqw, Aqr, tauq)
            _strict_qrfac_probe(geqlf!, Aqw, Aqr, tauq)
            _strict_qrfac_probe(gerqf!, Aqw, Aqr, tauq)
            _strict_qrfac_probe(tzrzf!, Aqw, Aqr, tauq)
            _strict_qp3_probe(Aqw, Aqr, jpq, tauq)
            _strict_orgq_probe(orgqr!, Aqw, Afqr, tauq)
            _strict_orgq_probe(orglq!, Aqw, Aflq, taul)
            _strict_orgq_probe(orgql!, Aqw, Afql, tauql)
            _strict_orgq_probe(orgrq!, Aqw, Afrq, taurq)
            _strict_ormq_probe(ormqr!, Cqw, Cqr, Afqr, tauq)
            _strict_ormq_probe(ormlq!, Cqw, Cqr, Aflq, taul)
            _strict_ormq_probe(ormql!, Cqw, Cqr, Afql, tauql)
            _strict_ormq_probe(ormrq!, Cqw, Cqr, Afrq, taurq)
            _strict_ormq_probe(ormrz!, Cqw, Cqr, Afrz, taurz)
            _strict_orgtr_probe(bk, Aqtr, tautr, Qtr)
            _strict_ungtr_probe(bk, Aztr, tauztr, Qztr)
        end
        # Trim-compatibility guarantee (contracts.jl): the complex unpacked gemm kernel is the trim-critical
        # path (its runtime bool→Val flags were the union-split that regressed zgemm_64_/cgemm_64_). Fast/dev
        # here → heuristic; the full-mode dogfood roots the same call under juliac's authoritative verifier.
        @assert_trim_compatible _gemm_cmplx_unpacked!(
            Val(1), Val(1), false, m, m, m,
            one(ComplexF64), Az3, Bz3, zero(ComplexF64), Cz3
        )
    end
end
