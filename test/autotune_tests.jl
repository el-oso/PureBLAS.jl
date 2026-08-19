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
    Zen3 = hw(32, 32 * 1024, 512 * 1024, 32 * 1024^2, :AMD, 0x19, 16)  # Zen3, AVX2 (L2 512K, L3 1 CCD)
    Zen4 = hw(64, 32 * 1024, 1024^2, 16 * 1024^2, :AMD, 0x19, 32)   # Zen4, double-pumped 512
    Zen5 = hw(64, 48 * 1024, 1024^2, 16 * 1024^2, :AMD, 0x1A, 32)   # Zen5, native 512
    tigerlake = hw(64, 48 * 1024, 1280 * 1024, 12 * 1024^2, :Intel, 0x06, 32)  # never benchmarked — prediction

    # ── Complex Cholesky (wired to derived defaults): must reproduce the measured optima ──────────────
    @test P._at_cpotrf_base(Zen3) == 48
    @test P._at_cpotrf_base(Zen4) == 64
    @test P._at_cpotrf_base(Zen5) == 64
    @test P._at_cpotrf_nbmax(Zen3) == 128
    @test P._at_cpotrf_nbmax(Zen4) == 192
    @test P._at_cpotrf_nbmax(Zen5) == 192
    @test P._at_cpotf2_mr(Zen3) == 2        # AVX2 → double-pump-equivalent 32B datapath
    @test P._at_cpotf2_mr(Zen4) == 2   # Zen4 double-pumped 512 → 32B datapath
    @test P._at_cpotf2_mr(Zen5) == 1  # Zen5 native 512 → 64B datapath

    # ── gemm blocks (derivation validated here; wiring is a follow-up per-box gate A/B) ───────────────
    # Reproduce the Zen4-tuned literals exactly on Zen4:
    @test P._at_gemm_kc(Zen4) == 256   # kc·NR·8 = ½·32K
    @test P._at_gemm_mc(Zen4) == 144   # 30%·L2 / (kc·8), rounded to mr·W=16
    @test P._at_gemm_nc(Zen4) == 2040  # ¼·L3 / (kc·8) = 2048 → po2-dodge → 2040
    @test (P._at_gemm_mr(Zen4), P._at_gemm_nr(Zen4)) == (2, 8)
    @test (P._at_gemm_mr(Zen3), P._at_gemm_nr(Zen3)) == (3, 4)
    @test (P._at_gemm_mr(Zen5), P._at_gemm_nr(Zen5)) == (2, 8)

    # ── Complex mc (all L3 sites derive via _at_mc_kc with the COMPLEX element, 16 B/elt) ─────────────
    # Joint residency mc·kc·sizeof ≤ 30%·L2. Complex (16 B) halves the real mc; Zen3's 512K L2 halves
    # it again — the adaptation the Zen4-tuned _MC=144 literal could not do. Fleet-A/B validated 2026-07-08.
    @test P._at_mc_kc(Zen4, ComplexF64, 256, 16, 4096) == 64   # 30%·1M /(256·16)=76 → mr-round 64
    @test P._at_mc_kc(Zen5, ComplexF64, 256, 16, 4096) == 64
    @test P._at_mc_kc(Zen3, ComplexF64, 256, 16, 4096) == 32        # 512K L2 → half of Zen4/Zen5
    @test P._at_mc_kc(Zen4, Float64, 256, 16, 4096) == 144     # real path unchanged (8 B/elt)

    # ── Pack crossovers (req#8): register-capacity cut for gemm/rank-k, L2-copy-residency for symm ─────
    @test P._at_gemm_unpack_max(Zen3) == 96          # 2·(16−4)·4 ; measured unpk/blk tie-band 80–112
    @test P._at_gemm_unpack_max(Zen4) == 448    # 2·(32−4)·8 ; EXACT match to the validated Zen4 literal
    @test P._at_gemm_unpack_max(Zen5) == 448
    @test P._at_rank_k_pack_cut(Zen3) == 84          # AVX2 MULTI-pack path: (7·48)/4 ; n=80→recursion, 96→packed
    @test P._at_rank_k_pack_cut(Zen4) == 8      # AVX-512 UNIFIED single-pack: crossover ≈ W (the 392 the old
    @test P._at_rank_k_pack_cut(Zen5) == 8     # formula gave mis-routed n≤256 → recursion → syrk n=128 miss)
    @test P._at_symm_mat_max(Zen3) == 256            # √(512K/8) ; measured mat≈pack tie exactly here
    @test P._at_symm_mat_max(Zen4) == 362       # √(1M/8) — predicted (down from the 448 placeholder)
    @test P._at_symm_mat_max(Zen5) == 362
    @test P._at_symm_mat_max(tigerlake) == 404        # isqrt(1280K/8) — out-of-fleet auto-size, no crash
    # live wired consts equal the formula applied to the detected _HW
    @test P._GEMM_UNPACK_MAX == P._at_gemm_unpack_max(P._HW)
    @test P._SYRK_PACK_CUT == P._at_rank_k_pack_cut(P._HW)
    @test P._SYR2K_PACK_CUT == P._at_rank_k_pack_cut(P._HW)
    @test P._SYMM_PACK_CUT == P._at_symm_mat_max(P._HW)

    # ── Out-of-fleet auto-sizing (no crash, sane values) — the whole point of the mandate ─────────────
    @test P._at_cpotf2_mr(tigerlake) == 1    # Intel native-512
    @test P._at_cpotrf_base(tigerlake) == 64
    @test P._double_pumped(Zen4) && !P._double_pumped(Zen5) && !P._double_pumped(tigerlake)

    # ── Real-axpy DRAM arm (replaced a OncePerProcess duel, 2026-08-19) ───────────────────────────────
    # Only the double-pumped part takes the narrow 256-bit phase arm; native-256 and native-512 both take
    # the interleaved arm. Measured Zen4/Zen3 (17% / 4%); Zen5 + Intel are predictions.
    @test P._at_axpy_dram(Zen4) == 208
    @test P._at_axpy_dram(Zen3) == 4
    @test P._at_axpy_dram(Zen5) == 4
    @test P._at_axpy_dram(tigerlake) == 4
    # Band-regime knob: same predicate, separate knob. These values reproduce what the retired duel
    # resolved per box, so the conversion is behaviour-neutral — 208 on Zen4 is the arm that closes the
    # n=1e6 gate cell, and the whole BLAS-1 ladder is governed by THIS knob (ws = 2·n·8 < L3 at n ≤ 1e6).
    @test P._at_axpy_band(Zen4) == 208
    @test P._at_axpy_band(Zen3) == 4
    @test P._at_axpy_band(Zen5) == 4   # narrow arm measured to LOSE 0.6-1.4% on Zen5
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
end
