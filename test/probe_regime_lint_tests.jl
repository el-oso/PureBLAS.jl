# Enforces the probe-regime declaration in the suite (#113, fifth recurrence). A probe's verdict is only
# valid in the regime it probed; nothing prevented consulting one from another regime except memory, and
# memory failed four times. The scanner + the pre-lint debt baseline live in probe_regime_lint.jl /
# probe_regime_baseline.txt (run standalone: `julia test/probe_regime_lint.jl`).
@testitem "probe-regime lint: every probe declares its regime" begin
    include(joinpath(@__DIR__, "probe_regime_lint.jl"))   # defines probe_regime_scan; CLI guard skips execution
    v = probe_regime_scan()
    isempty(v) || @error "probe-regime lint: probe(s) with no declared regime. Add a `# REGIME:` line naming \
        BOTH residency (L1/L2/L3/DRAM, freshly-written or not) and call structure (one call per sample vs a \
        rep loop, fresh vs persistent operands), or `# regime-ok: <reason>`." probes = v
    @test isempty(v)
end
