# Precompile-time contract verification for the SIMDBackend. Included LAST (after every backend op is
# defined) because @verify_strict actually CALLS the ops — the L3 warm-up and @strict calls need
# gemm!/symm!/trmm!/trsm! from gemm.jl/level3.jl, which are included after backend.jl.
#
# @verify_strict is the single backend verifier: it runs TypeContracts.@verify SIMDBackend — method
# existence + declared return types over the whole chain (AbstractBLAS1 → BLAS2 → BLAS3 → LAPACK →
# LAPACKSolve) —
# AND @strict on each representative call (type-stable + allocation-free), since AbstractBLAS1/2/3 are
# @strict_contracts. The @strict calls self-gate on the `checks_enabled` preference.
#
# WHICH HALF ACTUALLY RUNS, MEASURED — read this before adding a member. `LocalPreferences.toml` sets
# `[StrictMode] analysis = "full"`, and `StrictMode.backend_available()` is FALSE in the main env
# (AllocCheck/JET are test-only), so the `if` below is FALSE and the @strict half of this block does not
# execute at PureBLAS's own precompile. What DOES run every time is `TypeContracts.@verify SIMDBackend`
# — the method surface + declared return types of all 78 members — because that is inside @verify_strict
# but the `if` guards the whole let. The @strict half runs when the block is entered with the backend
# loaded, i.e. the test suite's strictmode dogfood, where `_noalloc_mode` is `:static` (AllocCheck's
# proof, falling back to empirical `@allocated`).
# Do NOT use StrictMode 0.3.10's `:fast` heuristic (`_alloc_signals`) as the eligibility test for a new
# member: the control run in bench/probes/strict_heuristic_control.jl flags EVERY already-contracted
# LAPACK member — gemm!, potrf!, getrf!, geqrf!, potrs!, getri!, pstrf!, the whole QR family — at a
# measured 0 B. The predicate that means something here is empirical @allocated after warm-up, plus a
# concrete inferred return type; that is what every member below was admitted on
# (bench/probes/strict_contract_eligibility.jl, strict_contract_dryrun.jl).
#
# TRIM-COMPATIBILITY is a strict-contract guarantee too (contracts.jl): the complex-gemm
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

# ── Probes for the 34 members added when the nonsymmetric-eigen / Bunch-Kaufman / QZ / SVD-front /
# least-squares families moved to owned workspace. Same discipline as everything above: re-seed every
# overwritten input with `copyto!` (allocation-free) because `@strict` calls its target repeatedly, and
# return `nothing` so the multi-output tuples are never charged to the call site. Each is routed through
# the BACKEND method, since that is the method the contract actually binds. The `job`/`uplo`/`side`/
# `trans` characters are passed as ARGUMENTS rather than baked in, so one probe covers every arm and the
# strict block can list the arms it wants without a probe per combination.
_strict_gebal_probe(bk, A, A0, scale) = (copyto!(A, A0); gebal!(bk, A, scale; job = 'B'); nothing)
_strict_gebak_probe(bk, V, V0, scale, ilo, ihi) =
    (copyto!(V, V0); gebak!(bk, 'B', 'R', ilo, ihi, scale, V); nothing)
_strict_gehrd_probe(bk, A, A0, ilo, ihi, tau) = (copyto!(A, A0); gehrd!(bk, A, ilo, ihi, tau); nothing)
# orghr! consumes the FACTORED source (gehrd's reflectors), not the raw A, and overwrites it with Q.
_strict_orghr_probe(bk, A, Af, ilo, ihi, tau) = (copyto!(A, Af); orghr!(bk, A, ilo, ihi, tau); nothing)
# ormhr! leaves A/tau alone (read-only, as in reference LAPACK) and overwrites C.
_strict_ormhr_probe(bk, C, C0, side, trans, ilo, ihi, Af, tau) =
    (copyto!(C, C0); ormhr!(bk, side, trans, ilo, ihi, Af, tau, C); nothing)
_strict_hseqr_probe(bk, H, H0, job, compz, ilo, ihi, w, Z) =
    (copyto!(H, H0); hseqr!(bk, job, compz, H, ilo, ihi, w, Z); nothing)
_strict_trevc_probe(bk, Tw, T0, VL, VR, VR0) =
    (copyto!(Tw, T0); copyto!(VR, VR0); trevc!(bk, 'R', 'B', Tw, VL, VR); nothing)
_strict_trexc_probe(bk, Tw, T0, Q, Q0, ifst, ilst) =
    (copyto!(Tw, T0); copyto!(Q, Q0); trexc!(bk, 'V', Tw, Q, ifst, ilst); nothing)
_strict_trsen_probe(bk, Tw, T0, Q, Q0, job, sel, w) =
    (copyto!(Tw, T0); copyto!(Q, Q0); trsen!(bk, job, 'V', sel, Tw, Q, w); nothing)
_strict_trsyl_probe(bk, C, C0, transa, transb, A, B) =
    (copyto!(C, C0); trsyl!(bk, transa, transb, 1, A, B, C); nothing)
_strict_geevr_probe(bk, A, A0, wr, wi, VL, VR, scale) =
    (copyto!(A, A0); geev!(bk, 'N', 'V', A, wr, wi, VL, VR, scale); nothing)
_strict_geevc_probe(bk, A, A0, w, VL, VR, scale) =
    (copyto!(A, A0); geev!(bk, 'N', 'V', A, w, VL, VR, scale); nothing)
_strict_gees_probe(bk, A, A0, w, VS, scale) = (copyto!(A, A0); gees!(bk, 'V', A, w, VS, scale); nothing)

_strict_sytrf_probe(bk, A, A0, ipiv, uplo) = (copyto!(A, A0); sytrf!(bk, A, ipiv; uplo = uplo); nothing)
_strict_hetrf_probe(bk, A, A0, ipiv, uplo) = (copyto!(A, A0); hetrf!(bk, A, ipiv; uplo = uplo); nothing)
_strict_sytrs_probe(bk, B, B0, Af, ipiv, uplo) =
    (copyto!(B, B0); sytrs!(bk, Af, ipiv, B; uplo = uplo); nothing)
_strict_hetrs_probe(bk, B, B0, Af, ipiv, uplo) =
    (copyto!(B, B0); hetrs!(bk, Af, ipiv, B; uplo = uplo); nothing)
# sytri!/hetri! take the FACTOR and overwrite it with the inverse — re-seed from the factorization.
_strict_sytri_probe(bk, A, Af, ipiv, uplo) = (copyto!(A, Af); sytri!(bk, A, ipiv; uplo = uplo); nothing)
_strict_hetri_probe(bk, A, Af, ipiv, uplo) = (copyto!(A, Af); hetri!(bk, A, ipiv; uplo = uplo); nothing)
_strict_sysv_probe(bk, A, A0, B, B0, ipiv, uplo) =
    (copyto!(A, A0); copyto!(B, B0); sysv!(bk, uplo, A, B, ipiv); nothing)
_strict_hesv_probe(bk, A, A0, B, B0, ipiv, uplo) =
    (copyto!(A, A0); copyto!(B, B0); hesv!(bk, uplo, A, B, ipiv); nothing)
_strict_syconv_probe(bk, A, Af, ipiv, work, uplo) =
    (copyto!(A, Af); syconv!(bk, uplo, A, ipiv, work); nothing)

_strict_gghrd_probe(bk, A, A0, B, B0, Q, Z) =
    (copyto!(A, A0); copyto!(B, B0); gghrd!(bk, 'I', 'I', A, B, Q, Z); nothing)
_strict_hgeqz_probe(bk, H, H0, Tm, T0, alpha, beta, Q, Z) =
    (copyto!(H, H0); copyto!(Tm, T0); hgeqz!(bk, 'S', 'I', 'I', H, Tm, alpha, beta, Q, Z); nothing)
_strict_tgevc_probe(bk, S, S0, Pm, P0, VL, VR, VR0) =
    (
    copyto!(S, S0); copyto!(Pm, P0); copyto!(VR, VR0);
    tgevc!(bk, 'R', 'B', S, Pm, VL, VR); nothing
)
_strict_tgsen_probe(bk, sel, S, S0, Tm, T0, Q, Q0, Z, Z0, alpha, beta) =
    (
    copyto!(S, S0); copyto!(Tm, T0); copyto!(Q, Q0); copyto!(Z, Z0);
    tgsen!(bk, sel, S, Tm, Q, Z, alpha, beta); nothing
)
_strict_ggevr_probe(bk, A, A0, B, B0, alphar, alphai, beta, vl, vr) =
    (copyto!(A, A0); copyto!(B, B0); ggev!(bk, 'N', 'V', A, B, alphar, alphai, beta, vl, vr); nothing)
_strict_ggevc_probe(bk, A, A0, B, B0, alpha, beta, vl, vr) =
    (copyto!(A, A0); copyto!(B, B0); ggev!(bk, 'N', 'V', A, B, alpha, beta, vl, vr); nothing)
_strict_gges_probe(bk, A, A0, B, B0, alpha, beta, vsl, vsr) =
    (copyto!(A, A0); copyto!(B, B0); gges!(bk, 'V', 'V', A, B, alpha, beta, vsl, vsr); nothing)

_strict_gebd2_probe(bk, A, A0, d, e, tauq, taup) =
    (copyto!(A, A0); gebd2!(bk, A, d, e, tauq, taup); nothing)
_strict_gebrd_probe(bk, A, A0, d, e, tauq, taup) =
    (copyto!(A, A0); gebrd!(bk, A, d, e, tauq, taup); nothing)
_strict_bdsdc_probe(bk, d, d0, e, e0, Lvec, Rvec) =
    (copyto!(d, d0); copyto!(e, e0); bdsdc!(bk, d, e, Lvec, Rvec); nothing)
_strict_bdsqr_probe(bk, d, d0, e, e0, U, V) =
    (copyto!(d, d0); copyto!(e, e0); bdsqr!(bk, d, e, U, V); nothing)
_strict_bdsqrc_probe(bk, d, d0, e, e0, Vt, U, C) =
    (copyto!(d, d0); copyto!(e, e0); bdsqr!(bk, 'U', d, e, Vt, U, C); nothing)
_strict_ggsvd_probe(bk, A, A0, B, B0, U, V, Q, alpha, beta, R) =
    (
    copyto!(A, A0); copyto!(B, B0);
    ggsvd!(bk, 'U', 'V', 'Q', A, B, U, V, Q, alpha, beta, R); nothing
)

_strict_gels_probe(bk, A, A0, B, B0) = (copyto!(A, A0); copyto!(B, B0); gels!(bk, 'N', A, B); nothing)
_strict_gelsy_probe(bk, A, A0, B, B0, jpvt, rcond) =
    (copyto!(A, A0); copyto!(B, B0); fill!(jpvt, 0); gelsy!(bk, A, B, jpvt, rcond); nothing)
_strict_gglse_probe(bk, A, A0, c, c0, B, B0, d, d0, x) =
    (
    copyto!(A, A0); copyto!(c, c0); copyto!(B, B0); copyto!(d, d0);
    gglse!(bk, A, c, B, d, x); nothing
)
# stebz!/stein! read d/e and write only their output buffers, so nothing needs re-seeding; the probes
# exist to swallow the (m, nsplit, info) tuple and the returned Z.
_strict_stebz_probe(bk, d, e, w, iblock, isplit) =
    (stebz!(bk, 'A', 'B', 0.0, 0.0, 1, 1, -1.0, d, e, w, iblock, isplit); nothing)
_strict_stein_probe(bk, d, e, w, iblock, isplit, Z) =
    (stein!(bk, d, e, w, iblock, isplit, Z); nothing)

# Source builders for the operands those probes take. Deterministic and Random-free (same reason the
# rest of this file uses `ones`), well-separated and diagonally dominant so no call in the strict
# window can raise — a throw here fails the PRECOMPILE, i.e. the package stops loading at all.
_strict_nonsym(::Type{T}, n) where {T} = T[i == j ? T(i) : T(1) / T(2 * (i + j)) for i in 1:n, j in 1:n]
_strict_ddom(::Type{T}, n, dg) where {T} = T[i == j ? T(dg) : T(1) / T(4 * (i + j)) for i in 1:n, j in 1:n]
_strict_utri(::Type{T}, n) where {T} =
    T[i <= j ? (i == j ? T(j + 1) : T(3) / T(10 * (i + j))) : zero(T) for i in 1:n, j in 1:n]
_strict_eye(::Type{T}, n) where {T} = T[i == j ? one(T) : zero(T) for i in 1:n, j in 1:n]
# A REAL Schur form carrying one 2×2 conjugate-pair block. Without it every diagonal block is 1×1 and
# `trexc!`/`trsyl!`/`trevc!` never reach their `dlaexc`/`dlasy2`/`dlaln2` 2×2 arms — half of each real
# kernel, and the half that owns the `t16` / `laexcd` workspace buffers.
function _strict_quasitri(n)
    Tq = _strict_utri(Float64, n)
    Tq[1, 1] = 1.0; Tq[2, 2] = 1.0; Tq[1, 2] = 1.0; Tq[2, 1] = -1.0
    return Tq
end
# gehrd! leaves its reflectors below the subdiagonal; hseqr! wants a clean Hessenberg.
function _strict_clean_hess!(H)
    @inbounds for j in axes(H, 2), i in (j + 2):size(H, 1)
        H[i, j] = zero(eltype(H))
    end
    return H
end

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
            # pptrf!'s own operand is order 32, deliberately ABOVE `_PPTRF_BLK_MIN` (16) so the strict
            # window covers `_pptrf_lower_blocked!` — see the note at its call below.
            nps = 32, APu0 = [i == j ? float(nps) + 1.0 : 1.0 for j in 1:nps for i in j:nps],
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
            # A SECOND, WIDE band for gbtrf! only. kl=ku=1 above reaches `_gbtf2!` and nothing else:
            # `gbtrf!` takes `_gbtrf_blocked!` (and the `_gbtrf_work` W13/W31/S scratch behind it) only
            # when `kl >= max(2·nb, _gbtrf_cross(T))`, and `_gbtrf_cross(Float64)` is 32 on a wide-SIMD
            # box and 64 otherwise (cpuinfo.jl:507). kl=64 clears the wider of the two, so this arm is
            # exercised on EVERY µarch rather than only where the crossover happens to be low. (A user
            # who pins `gbtrf_cross` above 64 falls back to the unblocked kernel here — still 0-alloc,
            # just less coverage.) gbtrs! is unaffected by blocking and stays on the narrow band above.
            klB = 64, kuB = 8, msB = 128,
            ABb0 = [r == klB + kuB + 1 ? 10.0 : 1.0 for r in 1:(2 * klB + kuB + 1), j in 1:msB],
            ABbw = zeros(2 * klB + kuB + 1, msB), ipivb = Vector{Int}(undef, msB),
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
        # ── Operands for the 34 members added with the nonsymmetric-eigen / Bunch-Kaufman / QZ /
        # SVD-front / least-squares owned-workspace conversion. Everything is order `ms` (=32) or
        # smaller: this block runs at every user's precompile, so each member gets the smallest shape
        # that still reaches its real kernel. `ev*`/`ez*` are the real/complex nonsymmetric-eigen sets,
        # `qz*`/`qzz*` the generalized pencils, `bkf*`/`bkz*` Bunch-Kaufman, `sv*`/`svz*` the SVD front
        # half and generalized SVD, `ls*`/`lsz*` least squares, `st*` the tridiagonal eigen pair.
        # Each `ev/ez` set carries four ms×ms scratch matrices because several members need two or three
        # distinct mutable targets live at once (trexc!'s T and Q, gghrd!'s A/B/Q/Z).
        evA = _strict_nonsym(Float64, ms); evB = _strict_ddom(Float64, ms, ms + 1)
        evW1 = zeros(ms, ms); evW2 = zeros(ms, ms); evW3 = zeros(ms, ms); evW4 = zeros(ms, ms)
        evsc = ones(ms); evtau = Vector{Float64}(undef, ms)
        copyto!(evW1, evA); evilo, evihi = gebal!(evW1, evsc; job = 'B')
        evV = ones(ms, ms); evV0 = ones(ms, ms); evC = ones(ms, ms)
        evAf = copy(evA); gehrd!(evAf, evilo, evihi, evtau)
        evH = _strict_clean_hess!(copy(evAf))
        evw = Vector{ComplexF64}(undef, ms); evwr = zeros(ms); evwi = zeros(ms)
        evT = _strict_quasitri(ms); evQI = _strict_eye(Float64, ms)
        # trsen!: `select` marks the leading cluster; `w` receives the eigenvalues (must not alias T).
        tsnsel = zeros(Int, ms); tsnsel[1:(ms ÷ 2)] .= 1
        tsnw = Vector{ComplexF64}(undef, ms)
        evVL = zeros(ms, 0); evU = _strict_utri(Float64, ms)
        ezA = _strict_nonsym(ComplexF64, ms); ezB = _strict_ddom(ComplexF64, ms, ms + 1)
        ezW1 = zeros(ComplexF64, ms, ms); ezW2 = zeros(ComplexF64, ms, ms)
        ezW3 = zeros(ComplexF64, ms, ms); ezW4 = zeros(ComplexF64, ms, ms)
        ezsc = ones(ms); eztau = Vector{ComplexF64}(undef, ms)
        copyto!(ezW1, ezA); ezilo, ezihi = gebal!(ezW1, ezsc; job = 'B')
        ezV = ones(ComplexF64, ms, ms); ezV0 = ones(ComplexF64, ms, ms); ezC = ones(ComplexF64, ms, ms)
        ezAf = copy(ezA); gehrd!(ezAf, ezilo, ezihi, eztau)
        ezH = _strict_clean_hess!(copy(ezAf))
        ezw = Vector{ComplexF64}(undef, ms)
        ezQI = _strict_eye(ComplexF64, ms); ezVL = zeros(ComplexF64, ms, 0)
        ezU = _strict_utri(ComplexF64, ms)
        # QZ: a Hessenberg-triangular pair for hgeqz!, and a generalized Schur pair for tgevc!/tgsen!.
        # Both must be genuine — handing the QZ iteration a raw pencil would "succeed" on nonsense.
        qzA = copy(evA); qzB = copy(evB); gghrd!('I', 'I', qzA, qzB, zeros(ms, ms), zeros(ms, ms))
        qzal = Vector{ComplexF64}(undef, ms); qzbe = zeros(ms)
        qzS = copy(qzA); qzP = copy(qzB); qzZ = zeros(ms, ms)
        hgeqz!('S', 'V', 'V', qzS, qzP, qzal, qzbe, zeros(ms, ms), qzZ)
        qzsel = zeros(Int, ms); qzsel[1] = 1; qzsel[2] = 1; qzsel[5] = 1
        qzzA = copy(ezA); qzzB = copy(ezB)
        gghrd!('I', 'I', qzzA, qzzB, zeros(ComplexF64, ms, ms), zeros(ComplexF64, ms, ms))
        qzzal = Vector{ComplexF64}(undef, ms); qzzbe = Vector{ComplexF64}(undef, ms)
        qzzS = copy(qzzA); qzzP = copy(qzzB); qzzZ = zeros(ComplexF64, ms, ms)
        hgeqz!('S', 'V', 'V', qzzS, qzzP, qzzal, qzzbe, zeros(ComplexF64, ms, ms), qzzZ)
        # Bunch-Kaufman: one source per (sy/he) × (L/U), each with its own pivots, because `_sytri_lower!`
        # / `_sytri_upper!` and the symmetric / Hermitian arms are four separate kernels.
        bkfA = _strict_ddom(Float64, ms, ms + 1)
        bkzA = [i == j ? ComplexF64(ms + 1) : ComplexF64(1, 0.25) / (i + j) for i in 1:ms, j in 1:ms]
        bkfip = Vector{Int}(undef, ms); bkfiph = Vector{Int}(undef, ms)
        bkzip = Vector{Int}(undef, ms); bkziph = Vector{Int}(undef, ms)
        bkfwk = Vector{Float64}(undef, ms); bkzwk = Vector{ComplexF64}(undef, ms)
        bkfb0 = ones(ms); bkfbw = zeros(ms); bkzb0 = ones(ComplexF64, ms); bkzbw = zeros(ComplexF64, ms)
        bkzB0 = ones(ComplexF64, ms, nrhs); bkzBw = zeros(ComplexF64, ms, nrhs)
        bkfLs = copy(bkfA); bkfLsp = Vector{Int}(undef, ms); sytrf!(bkfLs, bkfLsp; uplo = 'L')
        bkfUs = copy(bkfA); bkfUsp = Vector{Int}(undef, ms); sytrf!(bkfUs, bkfUsp; uplo = 'U')
        bkfLh = copy(bkfA); bkfLhp = Vector{Int}(undef, ms); hetrf!(bkfLh, bkfLhp; uplo = 'L')
        bkzLs = copy(bkzA); bkzLsp = Vector{Int}(undef, ms); sytrf!(bkzLs, bkzLsp; uplo = 'L')
        bkzUs = copy(bkzA); bkzUsp = Vector{Int}(undef, ms); sytrf!(bkzUs, bkzUsp; uplo = 'U')
        bkzLh = copy(bkzA); bkzLhp = Vector{Int}(undef, ms); hetrf!(bkzLh, bkzLhp; uplo = 'L')
        bkzUh = copy(bkzA); bkzUhp = Vector{Int}(undef, ms); hetrf!(bkzUh, bkzUhp; uplo = 'U')
        # SVD front half + generalized SVD. `p < n` for ggsvd! so the RQ branches (`_ggs_gerq2!` /
        # `_ggs_apply_rq_right!`, the two O(n²) sites the conversion actually targeted) are on the path.
        svm, svn, svp = 24, 16, 8
        svA = Float64[i == j ? 8.0 : 1.0 / (i + j) for i in 1:svm, j in 1:svn]
        svAw = zeros(svm, svn); svd = zeros(svn); sve = zeros(svn)
        svtq = Vector{Float64}(undef, svn); svtp = Vector{Float64}(undef, svn)
        svzA = ComplexF64[i == j ? ComplexF64(8) : ComplexF64(1, 0.3) / (i + j) for i in 1:svm, j in 1:svn]
        svzAw = zeros(ComplexF64, svm, svn)
        svztq = Vector{ComplexF64}(undef, svn); svztp = Vector{ComplexF64}(undef, svn)
        ggA = Float64[i == j ? 4.0 : 1.0 / (i + j) for i in 1:svm, j in 1:svn]
        ggB = Float64[i == j ? 3.0 : 1.0 / (2 * (i + j)) for i in 1:svp, j in 1:svn]
        ggAw = zeros(svm, svn); ggBw = zeros(svp, svn)
        ggU = zeros(svm, svm); ggV = zeros(svp, svp); ggQ = zeros(svn, svn)
        ggal = zeros(svn); ggbe = zeros(svn); ggR = zeros(svn, svn)
        ggzA = ComplexF64[i == j ? ComplexF64(4) : ComplexF64(1, 0.2) / (i + j) for i in 1:svm, j in 1:svn]
        ggzB = ComplexF64[i == j ? ComplexF64(3) : ComplexF64(0.5, 0.1) / (i + j) for i in 1:svp, j in 1:svn]
        ggzAw = zeros(ComplexF64, svm, svn); ggzBw = zeros(ComplexF64, svp, svn)
        ggzU = zeros(ComplexF64, svm, svm); ggzV = zeros(ComplexF64, svp, svp)
        ggzQ = zeros(ComplexF64, svn, svn); ggzR = zeros(ComplexF64, svn, svn)
        # Bidiagonal SVD driver: d/e are the bidiagonal itself, so no factored source is needed.
        bqd0 = [2.0 + i / svn for i in 1:svn]; bqe0 = fill(0.3, svn)
        bqd = zeros(svn); bqe = zeros(svn)
        bqU = _strict_eye(Float64, svn); bqV = _strict_eye(Float64, svn)
        bqVt = _strict_eye(ComplexF64, svn); bqUz = _strict_eye(ComplexF64, svn)
        bqC = zeros(ComplexF64, svn, 0)
        # bdsdc! writes both singular-vector sets into caller buffers (n×n each).
        bdLv = zeros(svn, svn); bdRv = zeros(svn, svn)
        # Least squares. B is max(m,n)×nrhs (LAPACK ldb): rows 1:m hold b on entry, 1:n hold X on exit.
        lsB0 = ones(max(svm, svn), nrhs); lsBw = zeros(max(svm, svn), nrhs); lsjp = zeros(Int, svn)
        lszB0 = ones(ComplexF64, max(svm, svn), nrhs); lszBw = zeros(ComplexF64, max(svm, svn), nrhs)
        lsc0 = ones(svm); lscw = zeros(svm); lsd0 = ones(svp); lsdw = zeros(svp); lsx = zeros(svn)
        lszc0 = ones(ComplexF64, svm); lszcw = zeros(ComplexF64, svm)
        lszd0 = ones(ComplexF64, svp); lszdw = zeros(ComplexF64, svp); lszx = zeros(ComplexF64, svn)
        # Symmetric-tridiagonal eigen. stein! wants stebz!'s own w/iblock/isplit, trimmed to the m
        # eigenvalues and nsplit blocks it actually found.
        std = [2.0 + i / ms for i in 1:ms]; ste = fill(0.4, ms - 1)
        stw = zeros(ms); stib = zeros(Int, ms); stisp = zeros(Int, ms)
        stm, stns, _ = stebz!('A', 'B', 0.0, 0.0, 1, 1, -1.0, std, ste, stw, stib, stisp)
        stwv = view(stw, 1:stm); stibv = view(stib, 1:stm); stispv = view(stisp, 1:stns)
        stZ = zeros(ms, stm)
        # SubArray operands over LARGER parents (lda != nrows), for the same reason the L3 block has
        # them: the owned-workspace accessors hand exactly this shape to exactly these routines, and a
        # union-typed slot that costs nothing on a `Matrix` boxes on a `SubArray`.
        vP1 = ones(ms + 8, ms + 8); vP2 = ones(ms + 8, ms + 8)
        vP3 = ones(ms + 8, ms + 8); vP4 = ones(ms + 8, ms + 8)
        vA = view(vP1, 1:ms, 1:ms); vB = view(vP2, 1:ms, 1:ms)
        vC = view(vP3, 1:ms, 1:ms); vD = view(vP4, 1:ms, 1:ms)
        vsc = view(ones(ms + 8), 1:ms); vtau = view(Vector{Float64}(undef, ms + 8), 1:ms)
        vwr = view(ones(ms + 8), 1:ms); vwi = view(ones(ms + 8), 1:ms)
        vwc = view(Vector{ComplexF64}(undef, ms + 8), 1:ms)
        vipP = Vector{Int}(undef, ms + 8); vip = view(vipP, 1:ms)
        vwk = view(ones(ms + 8), 1:ms)
        vAf = copy(evA); vtauP = Vector{Float64}(undef, ms + 8)
        gehrd!(vAf, 1, ms, vtauP); vtauf = view(vtauP, 1:ms)
        vH = _strict_clean_hess!(copy(vAf))
        vbkf = copy(bkfLs); copyto!(vip, bkfLsp)
        vBP = ones(ms + 8, nrhs + 4); vBv = view(vBP, 1:ms, 1:nrhs)
        vB0 = ones(ms, nrhs)
        vlsP = ones(svm + 8, svn + 8); vlsA = view(vlsP, 1:svm, 1:svn)
        vlsBP = ones(svm + 8, nrhs + 4); vlsB = view(vlsBP, 1:svm, 1:nrhs)
        vlsB0 = ones(svm, nrhs); vlsjp = view(vipP, 1:svn)
        vstP = ones(ms + 8, stm + 4); vstZ = view(vstP, 1:ms, 1:stm)
        # One pass through every new probe, so the workspace each routine grows on first touch is at
        # steady state before the strict window opens — otherwise first-touch growth is charged to the
        # routine and reads as a per-call allocation that isn't one.
        _strict_gebal_probe(bk, evW1, evA, evsc); _strict_gebak_probe(bk, evV, evV0, evsc, evilo, evihi)
        _strict_gehrd_probe(bk, evW1, evA, evilo, evihi, evtau)
        _strict_orghr_probe(bk, evW1, evAf, evilo, evihi, evtau)
        _strict_ormhr_probe(bk, evW2, evC, 'L', 'N', evilo, evihi, evAf, evtau)
        _strict_hseqr_probe(bk, evW1, evH, 'S', 'I', evilo, evihi, evw, evW3)
        _strict_trevc_probe(bk, evW1, evT, evVL, evW2, evQI)
        _strict_trexc_probe(bk, evW1, evT, evW2, evQI, 5, 1)
        _strict_trsyl_probe(bk, evW2, evC, 'N', 'N', evU, evU)
        _strict_geevr_probe(bk, evW1, evA, evwr, evwi, evVL, evW2, evsc)
        _strict_gees_probe(bk, evW1, evA, evw, evW2, evsc)
        _strict_gebal_probe(bk, ezW1, ezA, ezsc); _strict_gebak_probe(bk, ezV, ezV0, ezsc, ezilo, ezihi)
        _strict_gehrd_probe(bk, ezW1, ezA, ezilo, ezihi, eztau)
        _strict_orghr_probe(bk, ezW1, ezAf, ezilo, ezihi, eztau)
        _strict_ormhr_probe(bk, ezW2, ezC, 'L', 'N', ezilo, ezihi, ezAf, eztau)
        _strict_hseqr_probe(bk, ezW1, ezH, 'S', 'I', ezilo, ezihi, ezw, ezW3)
        _strict_trevc_probe(bk, ezW1, ezU, ezVL, ezW2, ezQI)
        _strict_trexc_probe(bk, ezW1, ezU, ezW2, ezQI, 5, 1)
        _strict_trsyl_probe(bk, ezW2, ezC, 'N', 'N', ezU, ezU)
        _strict_geevc_probe(bk, ezW1, ezA, ezw, ezVL, ezW2, ezsc)
        _strict_gees_probe(bk, ezW1, ezA, ezw, ezW2, ezsc)
        _strict_sytrf_probe(bk, evW1, bkfA, bkfip, 'L'); _strict_hetrf_probe(bk, evW1, bkfA, bkfiph, 'L')
        _strict_sytrs_probe(bk, Bsw, Bs0, bkfLs, bkfLsp, 'L')
        _strict_sytrs_probe(bk, bkfbw, bkfb0, bkfLs, bkfLsp, 'L')
        _strict_hetrs_probe(bk, Bsw, Bs0, bkfLh, bkfLhp, 'L')
        _strict_sytri_probe(bk, evW1, bkfLs, bkfLsp, 'L'); _strict_hetri_probe(bk, evW1, bkfLh, bkfLhp, 'L')
        _strict_sysv_probe(bk, evW1, bkfA, Bsw, Bs0, bkfip, 'L')
        _strict_hesv_probe(bk, evW1, bkfA, Bsw, Bs0, bkfiph, 'L')
        _strict_syconv_probe(bk, evW1, bkfLs, bkfLsp, bkfwk, 'L')
        _strict_sytrf_probe(bk, ezW1, bkzA, bkzip, 'L'); _strict_hetrf_probe(bk, ezW1, bkzA, bkziph, 'L')
        _strict_sytrs_probe(bk, bkzBw, bkzB0, bkzLs, bkzLsp, 'L')
        _strict_hetrs_probe(bk, bkzBw, bkzB0, bkzLh, bkzLhp, 'L')
        _strict_sytri_probe(bk, ezW1, bkzLs, bkzLsp, 'L'); _strict_hetri_probe(bk, ezW1, bkzLh, bkzLhp, 'L')
        _strict_sysv_probe(bk, ezW1, bkzA, bkzBw, bkzB0, bkzip, 'L')
        _strict_hesv_probe(bk, ezW1, bkzA, bkzBw, bkzB0, bkziph, 'L')
        _strict_syconv_probe(bk, ezW1, bkzLs, bkzLsp, bkzwk, 'L')
        _strict_gghrd_probe(bk, evW1, evA, evW2, evB, evW3, evW4)
        _strict_hgeqz_probe(bk, evW1, qzA, evW2, qzB, qzal, qzbe, evW3, evW4)
        _strict_tgevc_probe(bk, evW1, qzS, evW2, qzP, evVL, evW3, qzZ)
        _strict_tgsen_probe(bk, qzsel, evW1, qzS, evW2, qzP, evW3, qzZ, evW4, qzZ, qzal, qzbe)
        _strict_ggevr_probe(bk, evW1, evA, evW2, evB, evwr, evwi, qzbe, evVL, evW3)
        _strict_gges_probe(bk, evW1, evA, evW2, evB, qzal, qzbe, evW3, evW4)
        _strict_gghrd_probe(bk, ezW1, ezA, ezW2, ezB, ezW3, ezW4)
        _strict_hgeqz_probe(bk, ezW1, qzzA, ezW2, qzzB, qzzal, qzzbe, ezW3, ezW4)
        _strict_tgevc_probe(bk, ezW1, qzzS, ezW2, qzzP, ezVL, ezW3, qzzZ)
        _strict_tgsen_probe(bk, qzsel, ezW1, qzzS, ezW2, qzzP, ezW3, qzzZ, ezW4, qzzZ, qzzal, qzzbe)
        _strict_ggevc_probe(bk, ezW1, ezA, ezW2, ezB, qzzal, qzzbe, ezVL, ezW3)
        _strict_gges_probe(bk, ezW1, ezA, ezW2, ezB, ezw, qzzbe, ezW3, ezW4)
        _strict_gebd2_probe(bk, svAw, svA, svd, sve, svtq, svtp)
        _strict_gebd2_probe(bk, svzAw, svzA, svd, sve, svztq, svztp)
        _strict_bdsqr_probe(bk, bqd, bqd0, bqe, bqe0, bqU, bqV)
        _strict_bdsqr_probe(bk, bqd, bqd0, bqe, bqe0, nothing, nothing)
        _strict_bdsqrc_probe(bk, bqd, bqd0, bqe, bqe0, bqVt, bqUz, bqC)
        _strict_ggsvd_probe(bk, ggAw, ggA, ggBw, ggB, ggU, ggV, ggQ, ggal, ggbe, ggR)
        _strict_ggsvd_probe(bk, ggzAw, ggzA, ggzBw, ggzB, ggzU, ggzV, ggzQ, ggal, ggbe, ggzR)
        _strict_gels_probe(bk, svAw, svA, lsBw, lsB0)
        _strict_gels_probe(bk, svzAw, svzA, lszBw, lszB0)
        _strict_gelsy_probe(bk, svAw, svA, lsBw, lsB0, lsjp, 1.0e-12)
        _strict_gelsy_probe(bk, svzAw, svzA, lszBw, lszB0, lsjp, 1.0e-12)
        _strict_gglse_probe(bk, ggAw, ggA, lscw, lsc0, ggBw, ggB, lsdw, lsd0, lsx)
        _strict_gglse_probe(bk, ggzAw, ggzA, lszcw, lszc0, ggzBw, ggzB, lszdw, lszd0, lszx)
        _strict_stebz_probe(bk, std, ste, stw, stib, stisp)
        _strict_stein_probe(bk, std, ste, stwv, stibv, stispv, stZ)
        _strict_gbtrf_probe(bk, ABbw, ABb0, klB, kuB, msB, ipivb)
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
            # pptrf! is verified at order 32, i.e. ABOVE `_PPTRF_BLK_MIN` (16), so the arm under test is
            # `_pptrf_lower_blocked!`. That used to be the opposite: the probe ran at order 8 because the
            # blocked path built two `Matrix{T}(undef, n, nb)` scratch buffers per call (16 560 B at
            # n=32), so the one arm that broke the contract was the one arm the contract never saw. 7d2f44d
            # moved W/V onto `_pptrf_lower_work`'s owned scratch and closed that; re-measured here at
            # n ∈ {8, 16, 32, 64} — 0 B at every order, blocked and unblocked alike.
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
            # method that allocates its own output. TWO shapes, for the same reason pptrf! moved above
            # its blocking threshold: kl=ku=1 reaches `_gbtf2!` and nothing else, so the blocked kernel
            # and the `_gbtrf_work` W13/W31/S scratch behind it went unverified. kl=64 clears
            # `_gbtrf_cross(Float64)` on every µarch (32 wide-SIMD / 64 otherwise). Both measure 0 B.
            _strict_gbtrf_probe(bk, ABgw, ABg0, klS, kuS, ms, ipivgw)
            _strict_gbtrf_probe(bk, ABbw, ABb0, klB, kuB, msB, ipivb)
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
            # ── Nonsymmetric eigen: balance, Hessenberg reduction, Schur, eigenvectors, drivers.
            # REAL and COMPLEX arms both, because they are separate kernel sets throughout (dlahqr vs
            # zlahqr, the real quasi-triangular 2×2 machinery vs plain triangular back-substitution), and
            # a real-only assertion is exactly how ~1 KB/call in the complex 3M gemm passed for a month.
            # `trsen!` is deliberately ABSENT: measured 16 B/call at job∈{'V','B'} on BOTH types — the
            # `Base.RefValue` carrying `_dtrsyl!`'s scale out of the `_lacn2_estimate` callback. It goes
            # in when that Ref does; asserting it now would fail the build, and dropping the failing jobs
            # would make the member's guarantee a lie.
            _strict_gebal_probe(bk, evW1, evA, evsc)
            _strict_gebak_probe(bk, evV, evV0, evsc, evilo, evihi)
            _strict_gehrd_probe(bk, evW1, evA, evilo, evihi, evtau)
            _strict_orghr_probe(bk, evW1, evAf, evilo, evihi, evtau)
            # Both sides AND both trans arms: side='R' is a different gemm! shape, and it is the arm the
            # L3 block's own SubArray note records as the one that boxed when only side='L' was checked.
            _strict_ormhr_probe(bk, evW2, evC, 'L', 'N', evilo, evihi, evAf, evtau)
            _strict_ormhr_probe(bk, evW2, evC, 'R', 'T', evilo, evihi, evAf, evtau)
            # job='S'/compz='I' is the full Schur arm (T and Z formed); job='E'/compz='N' the
            # eigenvalues-only arm, which skips the Z accumulation entirely.
            _strict_hseqr_probe(bk, evW1, evH, 'S', 'I', evilo, evihi, evw, evW3)
            _strict_hseqr_probe(bk, evW1, evH, 'E', 'N', evilo, evihi, evw, evW3)
            _strict_trevc_probe(bk, evW1, evT, evVL, evW2, evQI)
            _strict_trexc_probe(bk, evW1, evT, evW2, evQI, 5, 1)
            # trsen! at all four jobs: 'N'/'E' skip the sep estimate entirely, while 'V'/'B' run the
            # `_lacn2_estimate` reverse-communication loop — the arm that carried the 16 B Ref.
            _strict_trsen_probe(bk, evW1, evT, evW2, evQI, 'N', tsnsel, tsnw)
            _strict_trsen_probe(bk, evW1, evT, evW2, evQI, 'E', tsnsel, tsnw)
            _strict_trsen_probe(bk, evW1, evT, evW2, evQI, 'V', tsnsel, tsnw)
            _strict_trsen_probe(bk, evW1, evT, evW2, evQI, 'B', tsnsel, tsnw)
            # transa='N' and 'T' are separate back-substitution orders; the third call feeds trsyl! a
            # quasi-triangular A so the `_syl_dlasy2` 2×2 arm (and its `t16` scratch) is on the path.
            _strict_trsyl_probe(bk, evW2, evC, 'N', 'N', evU, evU)
            _strict_trsyl_probe(bk, evW2, evC, 'T', 'N', evU, evU)
            _strict_trsyl_probe(bk, evW2, evC, 'N', 'N', evT, evU)
            _strict_geevr_probe(bk, evW1, evA, evwr, evwi, evVL, evW2, evsc)
            _strict_gees_probe(bk, evW1, evA, evw, evW2, evsc)
            _strict_gebal_probe(bk, ezW1, ezA, ezsc)
            _strict_gebak_probe(bk, ezV, ezV0, ezsc, ezilo, ezihi)
            _strict_gehrd_probe(bk, ezW1, ezA, ezilo, ezihi, eztau)
            _strict_orghr_probe(bk, ezW1, ezAf, ezilo, ezihi, eztau)
            _strict_ormhr_probe(bk, ezW2, ezC, 'L', 'N', ezilo, ezihi, ezAf, eztau)
            _strict_ormhr_probe(bk, ezW2, ezC, 'R', 'C', ezilo, ezihi, ezAf, eztau)
            _strict_hseqr_probe(bk, ezW1, ezH, 'S', 'I', ezilo, ezihi, ezw, ezW3)
            _strict_hseqr_probe(bk, ezW1, ezH, 'E', 'N', ezilo, ezihi, ezw, ezW3)
            _strict_trevc_probe(bk, ezW1, ezU, ezVL, ezW2, ezQI)
            _strict_trexc_probe(bk, ezW1, ezU, ezW2, ezQI, 5, 1)
            _strict_trsen_probe(bk, ezW1, ezU, ezW2, ezQI, 'N', tsnsel, tsnw)
            _strict_trsen_probe(bk, ezW1, ezU, ezW2, ezQI, 'B', tsnsel, tsnw)
            _strict_trsyl_probe(bk, ezW2, ezC, 'N', 'N', ezU, ezU)
            _strict_trsyl_probe(bk, ezW2, ezC, 'C', 'N', ezU, ezU)
            _strict_geevc_probe(bk, ezW1, ezA, ezw, ezVL, ezW2, ezsc)
            _strict_gees_probe(bk, ezW1, ezA, ezw, ezW2, ezsc)
            # ── Bunch-Kaufman. uplo='L' and 'U' are separate kernels throughout (`_sytri_lower!` /
            # `_sytri_upper!`, `_sytrs_lower!` / `_sytrs_upper!`), so both are asserted for COMPLEX,
            # where the Hermitian arm is genuinely distinct. For REAL eltype `hetrf!`/`hetrs!`/`hetri!`/
            # `hesv!` ARE `sytrf!`/… (herm=false), so their real arms are pinned once, at uplo='L', to
            # hold the dispatch rather than to re-verify the same kernel a second time.
            _strict_sytrf_probe(bk, evW1, bkfA, bkfip, 'L')
            _strict_sytrf_probe(bk, evW1, bkfA, bkfip, 'U')
            _strict_hetrf_probe(bk, evW1, bkfA, bkfiph, 'L')
            _strict_sytrs_probe(bk, Bsw, Bs0, bkfLs, bkfLsp, 'L')
            _strict_sytrs_probe(bk, Bsw, Bs0, bkfUs, bkfUsp, 'U')
            # VECTOR right-hand side. `sytrs!` used to `reshape(B, n, 1)` a vector RHS — 48 B/call of
            # Array header, and a `Base.ReshapedArray` (the trim-hostile `_throw_dmrs` path) when the RHS
            # was itself a view. The reshape is gone; this is the line that keeps it gone.
            _strict_sytrs_probe(bk, bkfbw, bkfb0, bkfLs, bkfLsp, 'L')
            _strict_hetrs_probe(bk, Bsw, Bs0, bkfLh, bkfLhp, 'L')
            _strict_sytri_probe(bk, evW1, bkfLs, bkfLsp, 'L')
            _strict_sytri_probe(bk, evW1, bkfUs, bkfUsp, 'U')
            _strict_hetri_probe(bk, evW1, bkfLh, bkfLhp, 'L')
            _strict_sysv_probe(bk, evW1, bkfA, Bsw, Bs0, bkfip, 'L')
            _strict_sysv_probe(bk, evW1, bkfA, Bsw, Bs0, bkfip, 'U')
            _strict_hesv_probe(bk, evW1, bkfA, Bsw, Bs0, bkfiph, 'L')
            _strict_syconv_probe(bk, evW1, bkfLs, bkfLsp, bkfwk, 'L')
            _strict_syconv_probe(bk, evW1, bkfUs, bkfUsp, bkfwk, 'U')
            _strict_sytrf_probe(bk, ezW1, bkzA, bkzip, 'L')
            _strict_sytrf_probe(bk, ezW1, bkzA, bkzip, 'U')
            _strict_hetrf_probe(bk, ezW1, bkzA, bkziph, 'L')
            _strict_hetrf_probe(bk, ezW1, bkzA, bkziph, 'U')
            _strict_sytrs_probe(bk, bkzBw, bkzB0, bkzLs, bkzLsp, 'L')
            _strict_sytrs_probe(bk, bkzBw, bkzB0, bkzUs, bkzUsp, 'U')
            _strict_sytrs_probe(bk, bkzbw, bkzb0, bkzLs, bkzLsp, 'L')
            _strict_hetrs_probe(bk, bkzBw, bkzB0, bkzLh, bkzLhp, 'L')
            _strict_hetrs_probe(bk, bkzBw, bkzB0, bkzUh, bkzUhp, 'U')
            _strict_sytri_probe(bk, ezW1, bkzLs, bkzLsp, 'L')
            _strict_sytri_probe(bk, ezW1, bkzUs, bkzUsp, 'U')
            _strict_hetri_probe(bk, ezW1, bkzLh, bkzLhp, 'L')
            _strict_hetri_probe(bk, ezW1, bkzUh, bkzUhp, 'U')
            _strict_sysv_probe(bk, ezW1, bkzA, bkzBw, bkzB0, bkzip, 'L')
            _strict_hesv_probe(bk, ezW1, bkzA, bkzBw, bkzB0, bkziph, 'L')
            _strict_hesv_probe(bk, ezW1, bkzA, bkzBw, bkzB0, bkziph, 'U')
            _strict_syconv_probe(bk, ezW1, bkzLs, bkzLsp, bkzwk, 'L')
            _strict_syconv_probe(bk, ezW1, bkzUs, bkzUsp, bkzwk, 'U')
            # ── QZ / generalized eigen. hgeqz!/tgevc!/tgsen! take a GENUINE Hessenberg-triangular or
            # generalized-Schur pair (built above): handing the QZ iteration a raw pencil would run a
            # different, shorter path and "succeed" on nonsense. `ggev!` appears at both arities — the
            # contract can only declare the real one (alphar, alphai, beta), so the complex form is held
            # to the guarantee here.
            _strict_gghrd_probe(bk, evW1, evA, evW2, evB, evW3, evW4)
            _strict_hgeqz_probe(bk, evW1, qzA, evW2, qzB, qzal, qzbe, evW3, evW4)
            _strict_tgevc_probe(bk, evW1, qzS, evW2, qzP, evVL, evW3, qzZ)
            _strict_tgsen_probe(bk, qzsel, evW1, qzS, evW2, qzP, evW3, qzZ, evW4, qzZ, qzal, qzbe)
            _strict_ggevr_probe(bk, evW1, evA, evW2, evB, evwr, evwi, qzbe, evVL, evW3)
            _strict_gges_probe(bk, evW1, evA, evW2, evB, qzal, qzbe, evW3, evW4)
            _strict_gghrd_probe(bk, ezW1, ezA, ezW2, ezB, ezW3, ezW4)
            _strict_hgeqz_probe(bk, ezW1, qzzA, ezW2, qzzB, qzzal, qzzbe, ezW3, ezW4)
            _strict_tgevc_probe(bk, ezW1, qzzS, ezW2, qzzP, ezVL, ezW3, qzzZ)
            _strict_tgsen_probe(bk, qzsel, ezW1, qzzS, ezW2, qzzP, ezW3, qzzZ, ezW4, qzzZ, qzzal, qzzbe)
            _strict_ggevc_probe(bk, ezW1, ezA, ezW2, ezB, qzzal, qzzbe, ezVL, ezW3)
            _strict_gges_probe(bk, ezW1, ezA, ezW2, ezB, ezw, qzzbe, ezW3, ezW4)
            # ── SVD front half + generalized SVD. `gebrd!`/`bdsdc!` are now asserted at their
            # workspace-free arity (the `ws`-taking forms stay internal); see contracts.jl. `bdsqr!` is
            # asserted at all three shapes it actually gets: real with U/V, real with `nothing`/`nothing`
            # (the values-only path, a Union argument and so the one that could union-split), and the
            # complex 6-argument kernel, which shares no code with either.
            _strict_gebd2_probe(bk, svAw, svA, svd, sve, svtq, svtp)
            _strict_gebd2_probe(bk, svzAw, svzA, svd, sve, svztq, svztp)
            _strict_gebrd_probe(bk, svAw, svA, svd, sve, svtq, svtp)
            _strict_gebrd_probe(bk, svzAw, svzA, svd, sve, svztq, svztp)
            _strict_bdsdc_probe(bk, bqd, bqd0, bqe, bqe0, bdLv, bdRv)
            _strict_bdsqr_probe(bk, bqd, bqd0, bqe, bqe0, bqU, bqV)
            _strict_bdsqr_probe(bk, bqd, bqd0, bqe, bqe0, nothing, nothing)
            _strict_bdsqrc_probe(bk, bqd, bqd0, bqe, bqe0, bqVt, bqUz, bqC)
            _strict_ggsvd_probe(bk, ggAw, ggA, ggBw, ggB, ggU, ggV, ggQ, ggal, ggbe, ggR)
            _strict_ggsvd_probe(bk, ggzAw, ggzA, ggzBw, ggzB, ggzU, ggzV, ggzQ, ggal, ggbe, ggzR)
            # ── Least squares. `gelsd!` is absent: MEASURED 100 096 B/call at ComplexF64 because its
            # complex arm goes through the complex `gesvd!`, the one factorization this file already
            # documents as not shown 0-alloc. Its Float64 arm is 0 B, so it becomes eligible the moment
            # complex gesvd! does — the same open item, not a second one.
            _strict_gels_probe(bk, svAw, svA, lsBw, lsB0)
            _strict_gels_probe(bk, svzAw, svzA, lszBw, lszB0)
            _strict_gelsy_probe(bk, svAw, svA, lsBw, lsB0, lsjp, 1.0e-12)
            _strict_gelsy_probe(bk, svzAw, svzA, lszBw, lszB0, lsjp, 1.0e-12)
            _strict_gglse_probe(bk, ggAw, ggA, lscw, lsc0, ggBw, ggB, lsdw, lsd0, lsx)
            _strict_gglse_probe(bk, ggzAw, ggzA, lszcw, lszc0, ggzBw, ggzB, lszdw, lszd0, lszx)
            # ── Symmetric-tridiagonal eigen. stein! is fed stebz!'s OWN w/iblock/isplit, trimmed to the
            # m eigenvalues and nsplit blocks found — views, therefore, which is also the SubArray shape
            # its five read-per-column inputs actually arrive in.
            _strict_stebz_probe(bk, std, ste, stw, stib, stisp)
            _strict_stein_probe(bk, std, ste, stwv, stibv, stispv, stZ)
            # ── SubArray arguments for the new families, same rationale as the L3 SubArray block above:
            # every assertion so far in this section passes a `Matrix`/`Vector`, and that is precisely the
            # gap that let `_trsm_right!`'s union-typed slot box 64 B/call while every Matrix-only
            # assertion passed. Parents are LARGER than the views, so `lda != nrows`.
            _strict_gebal_probe(bk, vA, evA, vsc)
            _strict_gehrd_probe(bk, vA, evA, 1, ms, vtau)
            _strict_orghr_probe(bk, vA, vAf, 1, ms, vtauf)
            _strict_ormhr_probe(bk, vC, evC, 'L', 'N', 1, ms, vAf, vtauf)
            _strict_hseqr_probe(bk, vA, vH, 'S', 'I', 1, ms, vwc, vB)
            _strict_trevc_probe(bk, vA, evT, evVL, vB, evQI)
            _strict_trexc_probe(bk, vA, evT, vB, evQI, 5, 1)
            _strict_trsyl_probe(bk, vC, evC, 'N', 'N', evU, evU)
            _strict_geevr_probe(bk, vA, evA, vwr, vwi, evVL, vB, vsc)
            _strict_gees_probe(bk, vA, evA, vwc, vB, vsc)
            _strict_gghrd_probe(bk, vA, evA, vB, evB, vC, vD)
            _strict_sytrf_probe(bk, vA, bkfA, vip, 'L')
            _strict_sytrs_probe(bk, vBv, vB0, vbkf, vip, 'L')
            _strict_sytri_probe(bk, vA, vbkf, vip, 'L')
            _strict_syconv_probe(bk, vA, vbkf, vip, vwk, 'L')
            _strict_gels_probe(bk, vlsA, svA, vlsB, vlsB0)
            _strict_gelsy_probe(bk, vlsA, svA, vlsB, vlsB0, vlsjp, 1.0e-12)
            _strict_stein_probe(bk, std, ste, stwv, stibv, stispv, vstZ)
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
