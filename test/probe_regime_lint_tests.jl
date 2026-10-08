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

# A probe must never LINK a reference BLAS — references are cache-only for benchmarks (`arms=pb`), and
# re-timing one both wastes the box and risks a published number whose two sides never saw the same
# machine state. Only bench/plots.jl may forward LBT. Legacy violators are grandfathered in
# test/probe_refblas_baseline.txt so this fails on NEW ones only.
@testitem "ref-BLAS lint: no probe links OpenBLAS/AOCL" begin
    include(joinpath(@__DIR__, "probe_refblas_lint.jl"))
    v = probe_refblas_scan()
    isempty(v) || @error "probe(s) link a reference BLAS; use the v3 cache (`arms=pb`), or mark the \
        file `# refblas-ok: <reason>` if it is genuinely diagnostic." probes = v
    @test isempty(v)
end

# A pool-liveness witness (`@test (@atomic p.gen) != g0`) must name the constant that decides
# admission: `_L1_MT_MIN`, the byte floor under which `_l1_workers` returns 1 and which scales with
# L1, or an `_sme_*` route predicate, since an SME-eligible call declines the pool deliberately. A
# size chosen by eye passes on the box it was written on and makes every assertion beside it vacuous
# on another; that has recurred four times. The scanner and the pre-lint debt baseline live in
# liveness_lint.jl / liveness_baseline.txt (run standalone: `julia test/liveness_lint.jl`).
@testitem "pool-liveness lint: every witness names its admission gate" begin
    include(joinpath(@__DIR__, "liveness_lint.jl"))
    v = liveness_scan()
    isempty(v) || @error "pool-liveness witness(es) with a guessed size. Derive the size from \
        `_L1_MT_MIN` (the grid's `_red_block` does NOT imply it), or assert the route as \
        `ran(f) == !sme` with the matching `_sme_*` predicate, or mark the item \
        `# liveness-ok: <reason>`." items = v
    @test isempty(v)
end
