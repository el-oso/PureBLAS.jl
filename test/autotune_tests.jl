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
    # 8 on EVERY box. The width formula `32 ÷ _lanes(hw,Float64)` was FALSIFIED by the gate on
    # 2026-08-20: wintermute gesvd 0.988 FAIL -> 1.058 PASS, neuromancer 1.053 -> 1.101. It had
    # reproduced the duel exactly (wm 4, galen 8, neuro 4, stable 6/6 each), which is precisely how a
    # wrong measurement propagates into a derivation that looks rigorous.
    @test P._at_brd_nb(wintermute) == 8 && P._at_brd_nb(neuromancer) == 8
    @test P._at_brd_nb(galen) == 8 && P._at_brd_nb(tigerlake) == 8

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
end
