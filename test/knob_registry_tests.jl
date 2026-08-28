# docs/src/knobs.md must match what test/knob_registry.jl generates from src/.
#
# Same contract as bench/check_artifacts_current.sh: a published artifact is a pure function of its
# source, so the check is "regenerate and compare", not "look plausible". Without this the registry is
# a snapshot that silently ages — which is the failure it exists to prevent.
@testitem "knob registry: marker binding is one-to-one" begin
    include(joinpath(@__DIR__, "knob_registry.jl"))
    # GROUND TRUTH, not a sanity check. Every failure of this scanner class has been MIS-attribution,
    # which passes any `!isempty(result)` test — so the binder is pinned to a source whose correct
    # answer is known by construction, with the exact shapes that broke it:
    #   * two ADJACENT marked knobs (the case that gave 4 rows a neighbour's justification);
    #   * a marked knob followed by an UNMARKED one (must not inherit);
    #   * prose containing "PDM:" that is NOT a marker (src/ really does contain these).
    fx = split("""
    # PDM: Derived — alpha reason. | tune: n/a
    const _A = @load_preference("alpha", _L1_BYTES ÷ 2)::Int
    # PDM: Literal — beta reason. | tune: candidate
    const _B = @load_preference("beta", 16)::Int
    const _C = @load_preference("gamma", 32)::Int
    # PDM: P = `@load_preference`; D = DERIVED — prose, not a marker
    const _D = @load_preference("delta", 8)::Int
    """, '\n')
    r = bind_knobs(fx)
    @test [x.key for x in r] == ["alpha", "beta", "gamma", "delta"]
    @test [x.tier for x in r] == ["Derived", "Literal", "Unaudited", "Unaudited"]
    @test r[1].pdm == "alpha reason. | tune: n/a"      # alpha keeps its own, not beta's
    @test r[3].pdm == ""                                # gamma does NOT inherit beta's marker
    @test r[4].tier == "Unaudited"                      # prose "PDM: P = …" must not bind
end

@testitem "knob registry: every knob carries a PDM marker" begin
    include(joinpath(@__DIR__, "knob_registry.jl"))
    # All 120 were classified on 2026-08-21, so "Unaudited" is now a REGRESSION, not a backlog: it
    # means a knob arrived without a justification or lost the one it had. Enforced rather than
    # reported, because a category that is allowed to be non-empty quietly fills up again.
    missing_marker = [r.key for r in knob_rows() if r.tier == "Unaudited"]
    isempty(missing_marker) || @error "knob(s) with no `# PDM:` marker. Every tuning knob must declare \
        its tier and why: `# PDM: <Derived|Measured|Literal|Exempt> — <one line> | tune: <cost>`. A \
        Literal additionally needs fleet evidence that one value suits every µarch (task #164)." knobs = missing_marker
    @test isempty(missing_marker)
end

@testitem "knob registry: docs/src/knobs.md is current" begin
    include(joinpath(@__DIR__, "knob_registry.jl"))
    want = knob_markdown()
    path = joinpath(@__DIR__, "..", "docs", "src", "knobs.md")
    if !isfile(path)
        @error "docs/src/knobs.md is missing — regenerate: julia --project=. test/knob_registry.jl"
        @test false
    else
        got = read(path, String)
        got == want || @error "docs/src/knobs.md is STALE w.r.t. src/. A knob was added, removed, or \
            its default/`# PDM:` marker changed. Regenerate: julia --project=. test/knob_registry.jl"
        @test got == want
    end
end
