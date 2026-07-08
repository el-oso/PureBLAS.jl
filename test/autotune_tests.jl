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
    galen       = hw(32, 32 * 1024, 1024^2, 32 * 1024^2, :AMD, 0x19, 16)   # Zen3, AVX2 (L3: 1 CCD)
    wintermute  = hw(64, 32 * 1024, 1024^2, 16 * 1024^2, :AMD, 0x19, 32)   # Zen4, double-pumped 512
    neuromancer = hw(64, 48 * 1024, 1024^2, 16 * 1024^2, :AMD, 0x1A, 32)   # Zen5, native 512
    tigerlake   = hw(64, 48 * 1024, 1280 * 1024, 12 * 1024^2, :Intel, 0x06, 32)  # never benchmarked — prediction

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

    # ── Out-of-fleet auto-sizing (no crash, sane values) — the whole point of the mandate ─────────────
    @test P._at_cpotf2_mr(tigerlake) == 1    # Intel native-512
    @test P._at_cpotrf_base(tigerlake) == 64
    @test P._double_pumped(wintermute) && !P._double_pumped(neuromancer) && !P._double_pumped(tigerlake)

    # ── The live machine's wired consts equal the formula applied to the detected _HW ─────────────────
    @test P._CPOTRF_BASE == P._at_cpotrf_base(P._HW)
    @test P._CPOTRF_NBMAX == P._at_cpotrf_nbmax(P._HW)
    @test P._CPOTF2_MR == P._at_cpotf2_mr(P._HW)
end
