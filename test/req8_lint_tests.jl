# Enforces req#8 in the suite: no NEW machine-dependent tuning literal may ship without deriving it from a
# detected hardware const or carrying a `# req8-ok:` justification. The scanner + the debt baseline live in
# req8_lint.jl / req8_lint_baseline.txt (run standalone: `julia test/req8_lint.jl`). This catches the
# "derived selection, hardcoded materialization" trap — e.g. writing `Val(8)` instead of `Val(_GEMVT_NC)`.
@testitem "req#8 lint: no un-derived tuning literals beyond baseline" begin
    include(joinpath(@__DIR__, "req8_lint.jl"))   # defines req8_new_violations; the CLI guard skips execution
    nv = req8_new_violations()
    isempty(nv) || @error "req#8 lint: NEW un-derived tuning literal(s). Derive from a detected const (e.g. \
        Val(_SOME_CONST)), or annotate `# req8-ok: <reason>`, or — last resort, as reviewed debt — add the \
        line to test/req8_lint_baseline.txt." violations = nv
    @test isempty(nv)
end
