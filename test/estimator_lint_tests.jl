# Enforces the approved measurement statistic in the suite. Companion to req8_lint_tests.jl, and for the
# same reason: an estimator swap does not error, it silently produces confident wrong answers. On
# 2026-08-03/04 probe harnesses used `minimum(@elapsed …)` while the gate used `median`, which inverted an
# iamax unroll ranking at n=1e6 and cost a day of chasing the contradiction.
@testitem "estimator lint: no unapproved timing reductions under bench/" begin
    include(joinpath(@__DIR__, "estimator_lint.jl"))
    v = estimator_scan()
    isempty(v) || @error "Unapproved timing reduction(s). Use bench/measure.jl `tstat` (median), or \
        annotate `# estimator-ok: <reason>` naming why this reduction is not a gate number." violations = v
    @test isempty(v)
end
