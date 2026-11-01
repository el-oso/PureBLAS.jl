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
