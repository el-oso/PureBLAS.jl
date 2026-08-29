# The comparison behind bench/cache_staleness.sh's comment-only carve-out. It decides whether a
# published cell is stale, so a bug here either forces needless multi-hour fleet re-sweeps (harmless
# but expensive) or silently publishes numbers describing code that no longer ships (the real risk).
@testitem "src_semantic_diff: comments are trivia, code is not" begin
    include(joinpath(@__DIR__, "..", "bench", "src_semantic_diff.jl"))

    base = "# a comment\nf(x) = x + 1\n"
    # Same code, different comments, extra blank lines, different indentation.
    recomment = "# a DIFFERENT comment\n\n#= block =#\nf(x) =  x + 1\n"
    changed = "# a comment\nf(x) = x + 2\n"

    tb = parse_tree(base, "base.jl")
    tr = parse_tree(recomment, "recomment.jl")
    tc = parse_tree(changed, "changed.jl")
    @test !isnothing(tb) && !isnothing(tr) && !isnothing(tc)

    @test tb == tr          # comment/whitespace-only edit ⇒ NOT stale
    @test tb != tc          # one changed literal ⇒ stale

    # Unparseable input must come back `nothing`, which the caller treats as a real change. Both
    # forms matter: Meta.parseall EMBEDS these rather than throwing, so a try/catch alone misses them.
    @test isnothing(parse_tree("f(x) = )(\n", "err.jl"))        # :error
    @test isnothing(parse_tree("f(x) = (1\n", "incomplete.jl")) # :incomplete

    # Line numbers must not leak into the comparison: same code, shifted down by a comment line.
    @test parse_tree("f(x) = x\n", "a.jl") == parse_tree("#\n#\nf(x) = x\n", "b.jl")
end
