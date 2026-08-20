# docs/src/knobs.md must match what test/knob_registry.jl generates from src/.
#
# Same contract as bench/check_artifacts_current.sh: a published artifact is a pure function of its
# source, so the check is "regenerate and compare", not "look plausible". Without this the registry is
# a snapshot that silently ages — which is the failure it exists to prevent.
@testitem "knob registry: docs/src/knobs.md is current" begin
    include(joinpath(@__DIR__, "knob_registry.jl"))
    want = knob_markdown()
    path = joinpath(@__DIR__, "..", "docs", "src", "knobs.md")
    if !isfile(path)
        @error "docs/src/knobs.md is missing — regenerate: julia --project=test test/knob_registry.jl"
        @test false
    else
        got = read(path, String)
        got == want || @error "docs/src/knobs.md is STALE w.r.t. src/. A knob was added, removed, or \
            its default/`# PDM:` marker changed. Regenerate: julia --project=test test/knob_registry.jl"
        @test got == want
    end
end
