# Hardware detection (src/cpuinfo.jl): a feature gate must be able to tell a MISSING sysctl key
# from a key that reads zero.
#
# WHY THIS FILE EXISTS. `_sysctl_int` answers 0 in both cases, which is right for a cache size —
# absent and zero both mean "fall back" — and wrong for a feature gate, where it turns a renamed or
# mistyped oid into a permanently false predicate: the route goes dead on every machine, nothing is
# raised, and the symptom reads as a kernel that never helps rather than as a detection fault.
# Darwin makes that easy to hit by publishing base-SME capabilities unprefixed (`SME_F32F32`) and
# extensions with `FEAT_` (`FEAT_SME_F64F64`), so a gate written by analogy with its neighbor names
# a key that exists nowhere.
@testitem "cpuinfo: a feature probe separates a missing key from a clear one" begin
    using PureBLAS
    P = PureBLAS

    bogus = "hw.optional.arm.pureblas_this_key_does_not_exist"
    # The key a Float32 gate written by analogy with `_SME_F64`'s would read. It is not an oid on
    # any machine; `_sysctl_int` cannot tell it from a feature that is switched off.
    by_analogy = "hw.optional.arm.FEAT_SME_F32F32"

    for name in (bogus, by_analogy)
        @test P._sysctl_probe(name) === (false, 0)
        @test !P._sysctl_feature(name)
        # The hazard itself, asserted rather than described in a comment.
        @test P._sysctl_int(name) == 0
    end

    if Sys.isapple() && Sys.ARCH === :aarch64
        # A PROBE THAT CANNOT FAIL IS NOT A GATE. Where the hardware publishes a key, a present key
        # and an absent one have to give different answers, or the gate is decoration.
        present = "hw.optional.arm.FEAT_SME2"
        if P._sysctl_probe(present)[1]
            @test P._sysctl_probe(present) != P._sysctl_probe(bogus)
            @test P._sysctl_feature(present) != P._sysctl_feature(bogus)
        end

        if P._SME_F32
            @test P._sysctl_probe("hw.optional.arm.SME_F32F32") === (true, 1)
            @test P._sysctl_probe("hw.optional.arm.SME_F32F32") != P._sysctl_probe(by_analogy)
            # ZA is `svl` x `svl` BYTES, so halving the element size doubles the tile dimension.
            @test P._SME_LANES_F32 == 2 * P._SME_LANES
        end

        # ⛔ THIS ASSERTION IS WHAT STOPS A MISTYPED KEY, and it must stay OUTSIDE any
        # `if P._SME_F32` guard: a wrong key makes the gate false, which would SKIP a check placed
        # inside it and the suite would pass vacuously. The Float32 gate is the WIDER one — the
        # 32-bit outer product is base SME and Float64 is the extension — so a machine detected as
        # Float64-capable must also detect as Float32-capable, and a typo breaks exactly that.
        @test !P._SME_F64 || P._SME_F32

        # SME2 advertised with no 32-bit capability key means a rename or a genuinely new part.
        @test !P._SME_F32_KEY_ABSENT
    else
        # Off Apple aarch64 every probe is `(false, 0)` by construction, so there is no distinction
        # to assert and the gate must be off.
        @test !P._SME_F32
        @test !P._SME_F32_KEY_ABSENT
    end
end

@testitem "cpuinfo: L1 geometry must describe a real cache" tags = [:unit] begin
    using PureBLAS
    P = PureBLAS

    # Real geometries: Zen4 / Zen3, Zen5, Apple perflevel0.
    @test isnothing(P._check_l1_geometry(32 * 1024, 8, 64))
    @test isnothing(P._check_l1_geometry(48 * 1024, 12, 64))
    @test isnothing(P._check_l1_geometry(128 * 1024, 8, 128))

    # 32 KiB / 12 ways = 2730.67 B: not a whole number of lines.
    @test_throws "L1 size 32768 B is not a whole multiple of associativity 12 x line 64 B" P._check_l1_geometry(
        32768, 12, 64
    )
    @test_throws "comes from the \"l1d_assoc\" preference" P._check_l1_geometry(
        32768, 12, 64; assoc_pinned = true
    )
    @test_throws "come from detection" P._check_l1_geometry(32768, 12, 64)
    @test_throws "inconsistent L1d cache geometry" P._check_l1_geometry(32768, 0, 64)

    # The live consts passed the same check at precompile time.
    @test isnothing(P._check_l1_geometry(P._L1_BYTES, P._L1D_ASSOC, P._CACHELINE))
end

@testitem "cpuinfo: L1 way is one memory page" tags = [:unit] begin
    using PureBLAS
    P = PureBLAS

    # Only meaningful for the detected default: a pinned associativity overrides it by design.
    if isnothing(P.load_preference(P, "l1d_assoc", nothing))
        if Sys.iswindows()
            # Sys.PAGESIZE is the 64 KiB allocation granularity there, above L1: one way spans all of L1.
            @test P._L1D_ASSOC == 1
            @test P._L1_WAY_BYTES == P._L1_BYTES
        elseif Sys.PAGESIZE <= P._L1_BYTES
            @test P._L1_WAY_BYTES == Sys.PAGESIZE
            @test P._L1D_ASSOC == P._L1_BYTES ÷ Sys.PAGESIZE
        end
    end
    # The detected values pass the consistency check on every OS.
    @test isnothing(P._check_l1_geometry(P._L1_BYTES, P._L1D_ASSOC, P._CACHELINE))
    # Positive control: the check can fail.
    @test_throws "inconsistent L1d cache geometry" P._check_l1_geometry(32768, 12, 64)
end

# The vector width, register budget and gemm tile follow the CODEGEN target, not the build machine:
# an `x86-64-v3` library built on an AVX-512 host must get the 256-bit tile, or its microkernel spills.
@testitem "cpuinfo: SIMD constants follow the codegen target" tags = [:checks] begin
    using PureBLAS
    P = PureBLAS

    # What an AVX-512 build machine detects, compiled for each target through the same functions
    # that produce `_SIMD_BYTES`, `_NVREG` and the gemm tile.
    tile(target) = let simd = P._target_simd_bytes(64, P._target_simd_cap(target))
        hw = (simd = simd, nvreg = P._nvreg(simd))
        (simd = simd, W = P._lanes(hw, Float64), nvreg = hw.nvreg,
         mr = P._at_gemm_mr(hw), nr = P._at_gemm_nr(hw))
    end

    @test tile("x86-64-v4").W == 8
    @test tile("x86-64-v3").W == 4
    @test tile("x86-64-v3,-avx512f").W == 4                      # modifiers do not change the level
    @test tile("x86-64-v3;znver4,clone_all").W == 4              # the baseline group decides
    @test tile("generic").W == 2
    @test tile("x86-64-v2").W == 2
    if Sys.ARCH === :x86_64
        @test tile("x86-64-v4").nvreg == 32
        @test tile("x86-64-v3").nvreg == 16
        @test tile("generic").nvreg == 16
        for t in ("x86-64-v4", "x86-64-v3", "generic")
            x = tile(t)
            # MR·NR accumulators plus one register per A row-block and one B broadcast.
            @test x.mr * x.nr + x.mr + 1 <= x.nvreg
        end
    end

    # Positive control: targets that do not name an ISA level leave detection alone, so the test
    # above is measuring the cap and not a width that was 256-bit all along.
    for t in ("native", "znver4", "apple-m1,+sme")
        @test isnothing(P._target_simd_cap(t))
        @test tile(t).W == 8
    end
    @test tile("x86-64-v3").mr * tile("x86-64-v3").nr * 4 < tile("native").mr * tile("native").nr * 8

    # The live build obeys its own target unless `simd_bytes` is pinned.
    if isnothing(P.load_preference(P, "simd_bytes", nothing))
        @test P._SIMD_BYTES <= something(P._TARGET_SIMD_CAP, typemax(Int))
    end
end
