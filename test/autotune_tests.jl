# AutoTune (req#8) — the derivation formulas must REPRODUCE the fleet's measured-optimal tuning values from
# detected-hardware descriptors alone. This is the offline validation gate: derive → check it lands on the
# known-good values → trust it to extrapolate to new machines. Formulas are pure functions of a `hw`
# NamedTuple (src/cpuinfo.jl), so we feed synthetic fleet descriptors here.
@testitem "AutoTune: formulas reproduce fleet-measured optima" begin
    using PureBLAS
    P = PureBLAS
    hw(simd, l1, l2, l3, vendor, family, nvreg) =
        (simd = simd, l1 = l1, l2 = l2, l3 = l3, vendor = vendor, family = family, nvreg = nvreg)
    #                     simd        l1        l2         l3       vendor  fam   nvreg
    galen = hw(32, 32 * 1024, 512 * 1024, 32 * 1024^2, :AMD, 0x19, 16)  # Zen3, AVX2 (L2 512K, L3 1 CCD)
    wintermute = hw(64, 32 * 1024, 1024^2, 16 * 1024^2, :AMD, 0x19, 32)   # Zen4, double-pumped 512
    neuromancer = hw(64, 48 * 1024, 1024^2, 16 * 1024^2, :AMD, 0x1A, 32)   # Zen5, native 512
    tigerlake = hw(64, 48 * 1024, 1280 * 1024, 12 * 1024^2, :Intel, 0x06, 32)  # never benchmarked — prediction

    # ── Complex Cholesky (wired to derived defaults): must reproduce the measured optima ──────────────
    @test P._at_cpotrf_base(galen) == 48
    @test P._at_cpotrf_base(wintermute) == 64
    @test P._at_cpotrf_base(neuromancer) == 64
    @test P._at_cpotrf_nbmax(galen) == 128
    @test P._at_cpotrf_nbmax(wintermute) == 192
    @test P._at_cpotrf_nbmax(neuromancer) == 192
    @test P._at_cpotf2_mr(galen) == 2        # AVX2 → double-pump-equivalent 32B datapath
    @test P._at_cpotf2_mr(wintermute) == 2   # Zen4 double-pumped 512 → 32B datapath
    @test P._at_cpotf2_mr(neuromancer) == 1  # Zen5 native 512 → 64B datapath

    # ── gemm blocks (derivation validated here; wiring is a follow-up per-box gate A/B) ───────────────
    # Reproduce the wintermute-tuned literals exactly on Zen4:
    @test P._at_gemm_kc(wintermute) == 256   # kc·NR·8 = ½·32K
    @test P._at_gemm_mc(wintermute) == 144   # 30%·L2 / (kc·8), rounded to mr·W=16
    @test P._at_gemm_nc(wintermute) == 2040  # ¼·L3 / (kc·8) = 2048 → po2-dodge → 2040
    @test (P._at_gemm_mr(wintermute), P._at_gemm_nr(wintermute)) == (2, 8)
    @test (P._at_gemm_mr(galen), P._at_gemm_nr(galen)) == (3, 4)
    @test (P._at_gemm_mr(neuromancer), P._at_gemm_nr(neuromancer)) == (2, 8)

    # ── Complex mc (all L3 sites derive via _at_mc_kc with the COMPLEX element, 16 B/elt) ─────────────
    # Joint residency mc·kc·sizeof ≤ 30%·L2. Complex (16 B) halves the real mc; galen's 512K L2 halves
    # it again — the adaptation the Zen4-tuned _MC=144 literal could not do. Fleet-A/B validated 2026-07-08.
    @test P._at_mc_kc(wintermute, ComplexF64, 256, 16, 4096) == 64   # 30%·1M /(256·16)=76 → mr-round 64
    @test P._at_mc_kc(neuromancer, ComplexF64, 256, 16, 4096) == 64
    @test P._at_mc_kc(galen, ComplexF64, 256, 16, 4096) == 32        # 512K L2 → half of Zen4/Zen5
    @test P._at_mc_kc(wintermute, Float64, 256, 16, 4096) == 144     # real path unchanged (8 B/elt)

    # ── Pack crossovers (req#8): register-capacity cut for gemm/rank-k, L2-copy-residency for symm ─────
    @test P._at_gemm_unpack_max(galen) == 96          # 2·(16−4)·4 ; measured unpk/blk tie-band 80–112
    @test P._at_gemm_unpack_max(wintermute) == 448    # 2·(32−4)·8 ; EXACT match to the validated Zen4 literal
    @test P._at_gemm_unpack_max(neuromancer) == 448
    @test P._at_rank_k_pack_cut(galen) == 84          # AVX2 MULTI-pack path: (7·48)/4 ; n=80→recursion, 96→packed
    @test P._at_rank_k_pack_cut(wintermute) == 8      # AVX-512 UNIFIED single-pack: crossover ≈ W (the 392 the old
    @test P._at_rank_k_pack_cut(neuromancer) == 8     # formula gave mis-routed n≤256 → recursion → syrk n=128 miss)
    @test P._at_symm_mat_max(galen) == 256            # √(512K/8) ; measured mat≈pack tie exactly here
    @test P._at_symm_mat_max(wintermute) == 362       # √(1M/8) — predicted (down from the 448 placeholder)
    @test P._at_symm_mat_max(neuromancer) == 362
    @test P._at_symm_mat_max(tigerlake) == 404        # isqrt(1280K/8) — out-of-fleet auto-size, no crash
    # live wired consts equal the formula applied to the detected _HW
    @test P._GEMM_UNPACK_MAX == P._at_gemm_unpack_max(P._HW)
    @test P._SYRK_PACK_CUT == P._at_rank_k_pack_cut(P._HW)
    @test P._SYR2K_PACK_CUT == P._at_rank_k_pack_cut(P._HW)
    @test P._SYMM_PACK_CUT == P._at_symm_mat_max(P._HW)

    # ── Out-of-fleet auto-sizing (no crash, sane values) — the whole point of the mandate ─────────────
    @test P._at_cpotf2_mr(tigerlake) == 1    # Intel native-512
    @test P._at_cpotrf_base(tigerlake) == 64
    @test P._double_pumped(wintermute) && !P._double_pumped(neuromancer) && !P._double_pumped(tigerlake)

    # ── Real-axpy DRAM arm (replaced a OncePerProcess duel, 2026-08-19) ───────────────────────────────
    # Only the double-pumped part takes the narrow 256-bit phase arm; native-256 and native-512 both take
    # the interleaved arm. Measured wintermute/galen (17% / 4%); Zen5 + Intel are predictions.
    # ── Banded-Cholesky width split (2026-08-19): reproduces 6/6 processes on all THREE boxes ────────
    # wintermute(Zen4) and neuromancer(Zen5) agree on every row and galen(Zen3) differs on every one —
    # what the first two share is simd=64, and `_double_pumped` SEPARATES them, so the criterion is
    # vector width, not microarchitecture.
    @test P._at_pbtrf_nb(wintermute, Float32) == 8
    @test P._at_pbtrf_nb(neuromancer, Float32) == 8
    @test P._at_pbtrf_nb(galen, Float32) == 16
    @test P._at_pbtrf_nb(wintermute, ComplexF32) == 32 && P._at_pbtrf_nb(neuromancer, ComplexF32) == 32
    @test P._at_pbtrf_nb(galen, ComplexF32) == 24
    @test P._at_pbtrf_nb(wintermute, ComplexF64) == 32 && P._at_pbtrf_nb(neuromancer, ComplexF64) == 32
    @test P._at_pbtrf_nb(galen, ComplexF64) == 24
    # F32 small-panel is a FORMULA (one vector of lanes), not a table — it must track _lanes exactly.
    @test P._at_pbtrf_nbs(wintermute, Float32) == 16 == P._lanes(wintermute, Float32)
    @test P._at_pbtrf_nbs(neuromancer, Float32) == 16
    @test P._at_pbtrf_nbs(galen, Float32) == 8 == P._lanes(galen, Float32)
    @test P._at_pbtrf_nbs(wintermute, Float64) == 8 && P._at_pbtrf_nbs(neuromancer, Float64) == 8
    @test P._at_pbtrf_nbs(galen, Float64) == 16
    # ucross is a FORMULA over L2, and it separates the fleet correctly for a different reason than
    # width does: wintermute and neuromancer share a 1 MiB L2, galen has 512 KiB.
    @test P._at_pbtrf_ucross(wintermute) == 256 && P._at_pbtrf_ucross(neuromancer) == 256
    @test P._at_pbtrf_ucross(galen) == 128
    @test P._at_pbtrf_ucross(tigerlake) == 1280 * 1024 ÷ 4096   # out-of-fleet: scales, no crash

    # ── Same width criterion, second batch. Rows marked MODAL had ONE box flip; the 5-of-6 value is
    # used, and the assertion records which box was modal so the provenance is not lost.
    @test P._at_gbtrf_cross(wintermute, Float32) == 48 && P._at_gbtrf_cross(neuromancer, Float32) == 48
    @test P._at_gbtrf_cross(galen, Float32) == 64
    @test P._at_gbtrf_cross(wintermute, Float64) == 32 && P._at_gbtrf_cross(neuromancer, Float64) == 32
    @test P._at_gbtrf_cross(galen, Float64) == 64          # MODAL on galen (64,64,48,64,64,64)
    @test P._at_gbtrf_cross(wintermute, ComplexF64) == 8 && P._at_gbtrf_cross(neuromancer, ComplexF64) == 8
    @test P._at_gbtrf_cross(galen, ComplexF64) == 16
    @test P._at_pbtrf_cross(wintermute, Float32) == 32 && P._at_pbtrf_cross(neuromancer, Float32) == 32
    @test P._at_pbtrf_cross(galen, Float32) == 40          # MODAL on galen (40,40,32,40,40,40)
    @test P._at_pbtrf_cross(wintermute, ComplexF32) == 24  # MODAL on wintermute (24x5, 16)
    @test P._at_pbtrf_cross(neuromancer, ComplexF32) == 24 && P._at_pbtrf_cross(galen, ComplexF32) == 16
    # F64 ucross does NOT obey the `l2 ÷ 4096` formula its F32/C32 siblings do — galen measures 192
    # where that formula gives 128. Asserted as a table row so a future "unify these" edit fails here.
    @test P._at_pbtrf_ucross(wintermute, Float64) == 256 && P._at_pbtrf_ucross(neuromancer, Float64) == 256
    @test P._at_pbtrf_ucross(galen, Float64) == 192        # MODAL on galen (192x4, 256, 192)
    @test P._at_pbtrf_ucross(galen, Float64) != P._at_pbtrf_ucross(galen)   # the two rules disagree
    # brd_nb is a formula on all three boxes, each stable 6/6.
    # brd_nb is NOT asserted here: it is machine-INVARIANT (8 on all three boxes, two ISAs) and so
    # lives in svd.jl as `_BRD_NB`, PDM Exempt, not as an `_at_*` function of `hw`. It was briefly
    # `_at_brd_nb(hw) = 8` — a function taking a hardware descriptor and ignoring it, which is the fake
    # formula the PDM ladder forbids. This file asserts hardware DERIVATIONS; a constant belongs with
    # its kernel.

    # gemv-T route mode. A `_wide_simd` "derivation" was attempted on 2026-08-20 and FALSIFIED before
    # it shipped: it gives Zen5 mode 1, which takes per-column at n=1024 where blocked wins ~10% on
    # that box (0.865/0.903/0.912 across three independent processes). Zen4 and Zen5 are identical in
    # every detected const relevant here — L2, L3, `_NVREG`, SIMD width, working set — and still want
    # OPPOSITE arms at n=1024, so this stays a KEYED LITERAL and the label is accurate, not lazy.
    # Full evidence and the two weaker instruments that misled earlier attempts: cpuinfo.jl (c5).
    @test P._at_gemvt_perscan(wintermute) == 1    # Zen4: percol 1.113 @512, 1.25 @1024
    @test P._at_gemvt_perscan(neuromancer) == 0   # Zen5: percol LOSES ~10% @1024 — mode 1 regresses it
    @test P._at_gemvt_perscan(galen) == 0         # AVX2: blocked, forcing mode 1 costs it 29%
    @test P._at_gemvt_perscan(tigerlake) == 0     # unseen hardware gets the conservative arm

    # gemv-T per-column WINDOW BOUNDS, exposed 2026-08-21. BEHAVIOUR-NEUTRALITY IS THE POINT: the
    # defaults must be exactly the two values that were hardcoded in `_gemvt_perscan`, or an unpinned
    # build silently re-routes gemv-T on every box. Asserted, not asserted-in-prose.
    @test P._GEMVT_PERCOL_AMIN == P._L2_BYTES
    @test P._GEMVT_PERCOL_XMAX == P._L1_BYTES ÷ 2
    for hw in (wintermute, galen, neuromancer, tigerlake)
        @test P._at_gemvt_percol_amin(hw) == hw.l2
        @test P._at_gemvt_percol_xmax(hw) == hw.l1 ÷ 2
    end
    # The window CAN express each box's measured per-size optimum — that is why the knob was reshaped.
    # F64 square: per-column iff m*n*8 > AMIN && m*8 <= XMAX.
    percol(hw, n, amin, xmax) = (n * n * 8 > amin) && (n * 8 <= xmax)
    # wintermute: the DEFAULT bounds already are its optimum (percol 512..2048, blocked 4096)
    @test [percol(wintermute, n, wintermute.l2, wintermute.l1 ÷ 2) for n in (512, 1024, 2048, 4096)] ==
          [true, true, true, false]
    # galen wants percol at n=1024 ONLY -> AMIN 2 MiB (excludes 512), XMAX 8 KiB (excludes 2048)
    @test [percol(galen, n, 2 * 1024^2, 8192) for n in (512, 1024, 2048, 4096)] ==
          [false, true, false, false]
    # neuromancer wants percol at n=512 ONLY -> XMAX 4 KiB
    @test [percol(neuromancer, n, neuromancer.l2, 4096) for n in (512, 1024, 2048, 4096)] ==
          [true, false, false, false]

    @test P._at_axpy_dram(wintermute) == 208
    @test P._at_axpy_dram(galen) == 4
    @test P._at_axpy_dram(neuromancer) == 4
    @test P._at_axpy_dram(tigerlake) == 4
    # Band-regime knob: same predicate, separate knob. These values reproduce what the retired duel
    # resolved per box, so the conversion is behaviour-neutral — 208 on Zen4 is the arm that closes the
    # n=1e6 gate cell, and the whole BLAS-1 ladder is governed by THIS knob (ws = 2·n·8 < L3 at n ≤ 1e6).
    @test P._at_axpy_band(wintermute) == 208
    @test P._at_axpy_band(galen) == 4
    @test P._at_axpy_band(neuromancer) == 4   # narrow arm measured to LOSE 0.6-1.4% on Zen5
    @test P._at_axpy_band(tigerlake) == 4
    # NO live-machine equality assertion for these two. `test/Project.toml` pins `axpy_unroll = 4` and
    # `axpy_dram = 4`, so under Pkg.test() the consts read 4 while the formula says 208 on Zen4 — a
    # preference beating a default is the system working, not a stale const. (Those pins exist only to
    # compile out the OncePerProcess resolvers for the all-paths noalloc proof; both knobs are Derive
    # tier as of 2026-08-19, so there is no longer a resolver to remove and the pins are now dead.)
    # The formula itself is fully covered by the per-descriptor assertions above, which no pin can mask.

    # ── The live machine's wired consts equal the formula applied to the detected _HW ─────────────────
    @test P._CPOTRF_BASE == P._at_cpotrf_base(P._HW)
    @test P._CPOTRF_NBMAX == P._at_cpotrf_nbmax(P._HW)
    @test P._CPOTF2_MR == P._at_cpotf2_mr(P._HW)
# ── strassen_min / trmm_rpack: derived from the REAL DATAPATH WIDTH ──────────────────────────────
# Zen4 double-pumps 512-bit ops over a 256-bit path, so its effective datapath is 32 B — the SAME
# as native-AVX2 Zen3. Only Zen5 is a native 64 B datapath. That is exactly why Zen3 and Zen4
# measured FLAT on both knobs while Zen5 wants very different values, and it is why the predicate
# is `_datapath_bytes` and not `_vwidth` (which cannot tell Zen4 from Zen5) or `_double_pumped`
# (which cannot tell Zen3 from Zen5).
@test P._datapath_bytes(galen) == 32
@test P._datapath_bytes(wintermute) == 32
@test P._datapath_bytes(neuromancer) == 64
# Zen3/Zen4 keep the shipped values — the change must be a no-op off native-512.
@test P._at_strassen_min(galen) == 1024
@test P._at_strassen_min(wintermute) == 1024
@test P._at_trmm_rpack(galen) == 448
@test P._at_trmm_rpack(wintermute) == 448
# Zen5 takes the measured optima (freq-locked, 4 runs for strassen_min, 2 locked for trmm_rpack):
#   strassen_min 256  -> 1.0155/1.0756/1.0625/1.0565/1.0000 at n=256..4096, no losing cell
#   trmm_rpack   1792 -> 1.0011/1.0813/1.0703 at n=256/512/1024
@test P._at_strassen_min(neuromancer) == 256
@test P._at_trmm_rpack(neuromancer) == 1792
# Tigerlake is native-512 too, so it inherits the Zen5 side — a PREDICTION, never benchmarked.
@test P._at_strassen_min(tigerlake) == 256
@test P._at_trmm_rpack(tigerlake) == 1792
# Live machine agrees with the formula applied to its own detected _HW.
@test P._STRASSEN_MIN == P._at_strassen_min(P._HW)
@test P._TRMM_RPACK == P._at_trmm_rpack(P._HW)
end

@testitem "trsm_base is clamped to the scratch it indexes" tags = [:unit] begin
    using PureBLAS, LinearAlgebra
    const P = PureBLAS
    # `_trsm_base_invL!/_invR!` build the inverse in `view(_l3_tmp(T), 1:nb, 1:nb)`, a fixed
    # _L3_NB×_L3_NB buffer — so the recursion base MUST NOT exceed _L3_NB. While `_TRSM_BASE` was a
    # bare const the invariant was unreachable; making it a preference put it in the user's hands, and
    # a pin is the one tier that cannot be exercised from here. Unclamped, `trsm_base = 4096` threw a
    # BoundsError from inside the kernel naming neither the preference nor the limit (measured on all
    # three fleet boxes, 2026-08-21). Both the const and the force hook are clamped; this pins both.
    @test P._TRSM_BASE <= P._L3_NB
    n = 192                                    # > _L3_NB, so the recursion descends past the base
    A = randn(n, n) ./ (2n); for i in 1:n; A[i, i] = 1 + abs(A[i, i]); end
    B0 = randn(n, n)
    want = UpperTriangular(A) \ B0
    if PureBLAS._FORCE_HOOKS
        old = P._FK["trsm_base"][]
        try
            for v in (1, 2, 4096)              # a real pin lands inside these endpoints
                P._FK["trsm_base"][] = v
                @test P._trsm_base() <= P._L3_NB
                B = copy(B0); P.trsm!(B, A; side = 'L', uplo = 'U')
                @test norm(B - want, Inf) / max(1.0, norm(want, Inf)) < 1e-8
            end
        finally
            P._FK["trsm_base"][] = old
        end
    end
end

@testitem "chol_nb/chol_nc are structural, not tunable" tags = [:unit] begin
    using PureBLAS, LinearAlgebra
    const P = PureBLAS
    # The fused Cholesky base kernel is HAND-UNROLLED for exactly four columns (d0..d3, l10..l32,
    # indices c0..c0+3); the knob appears only as the guard `nb == _fh_chol_nb()`. Any other value does
    # not run slower, it runs WRONG — measured 2026-08-22 on an oracle-confirmed PD matrix:
    #   chol_nb=3  -> SILENTLY wrong, rel err 1.0e-01      chol_nb=1,2,8,16,32 -> PosDefException
    #   chol_nc=2,3,8..128 -> SILENTLY wrong, 9.0e-02 .. 2.4e-01 (no exception at all)
    # Both were `@load_preference` with a `tune: candidate` marker, so a pin silently corrupted every
    # Cholesky in the process. They now validate at load and carry NO force hook.
    @test P._CHOL_NB == 4
    @test P._CHOL_NC == 4
    @test P._fh_chol_nb() == 4
    @test P._fh_chol_nc() == 4
    # Not sweepable: a probe must not be able to reintroduce the corruption.
    @test !haskey(P._FK, "chol_nb")
    @test !haskey(P._FK, "chol_nc")
    # And the shipped configuration factors correctly.
    n = 256; A0 = randn(n, n); H = A0 * A0' + n * I
    M = Matrix(H); P.potrf!(M; uplo = 'L'); L = tril(M)
    @test norm(L * L' - H, Inf) / norm(H, Inf) < 1e-10
end

@testitem "generated-meta lint: Vec-taking @generated helpers carry the inline meta" tags = [:unit] begin
    include(joinpath(@__DIR__, "generated_meta_lint.jl"))
    v = generated_meta_scan()
    isempty(v) || @error "A @generated helper takes/returns a Vec without emitting Expr(:meta, :inline). \
        `@inline` does NOT propagate into @generated CodeInfo (Julia 1.12), so the Vec is passed BY \
        POINTER and the CALLER's accumulators are stack-demoted — the failure that cost zgemvC \
        0.26-0.64 vs OpenBLAS. Emit `\$(Expr(:meta, :inline))` as the first statement of the returned \
        body." offenders = v
    @test isempty(v)
end
