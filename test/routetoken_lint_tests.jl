# Enforces req#11 in the suite: a size-keyed predicate inside a token-bearing function must not judge the
# chunk's own extent, and must not judge a token slot the caller filled with another dimension. The scanner
# and its reviewed baseline live in routetoken_lint.jl / routetoken_lint_baseline.txt (run standalone:
# `julia test/routetoken_lint.jl`).
#
# WHY A TESTITEM AND NOT ONLY test/lints.sh. Nothing in CI invokes lints.sh, so a check that lives only
# there runs when someone remembers to run it. The defect class this scanner covers is invisible to every
# correctness test — both arms are individually correct and the divergence is in the last bit — so the
# scanner IS the gate, and a gate that depends on being remembered is not one.
@testitem "route-token lint: no size-keyed predicate reads a chunk's own extent" begin
    include(joinpath(@__DIR__, "routetoken_lint.jl"))
    r = routetoken_lint()
    isempty(r.new) || @error "NEW size-keyed predicate(s) reachable from a chunk body in src/. A \
        partitioned caller passes the UNDIVIDED extents, so the branch must read the token \
        (`_rt_m`/`_rt_n`/`_rt_k`) and not its own slice — otherwise `set_num_threads(n)` changes the \
        result's bits. Fix the predicate, or add the line to test/routetoken_lint_baseline.txt WITH A \
        WRITTEN REASON, or mark it `# route-ok:` if it is a FLOOR on the tile actually handed to a \
        kernel." new = r.new
    isempty(r.stale) || @error "Stale route-token baseline entr(ies): the line no longer occurs in src/. \
        Delete it — a fixed site must not leave a rubber stamp behind." stale = r.stale
    @test isempty(r.new)
    @test isempty(r.stale)
end
