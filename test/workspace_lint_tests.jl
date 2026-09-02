# Enforces the workspace non-aliasing invariant in the suite: no L3Workspace field may be claimed again
# from inside a live claim of it. Two shipped bugs had exactly this shape (trsm_tmp, CONFIRMED wrong
# answers; rpack, confirmed by reading and fixed by the `rpad` split), both history-dependent and one of
# them AVX2-only — i.e. neither is reliably catchable by a numerical test on an AVX-512 gate box.
# Scanner + reviewed baseline: workspace_lint.jl / workspace_lint_baseline.txt (standalone:
# `julia test/workspace_lint.jl`, regenerate with `--baseline`).
@testitem "workspace lint: no claim of a workspace field inside a live claim of it" begin
    include(joinpath(@__DIR__, "workspace_lint.jl"))
    r = ws_lint()
    isempty(r.new) || @error "workspace lint: NEW self-alias path(s). Give that role its OWN field (as \
        `trsmw` and `rpad` did), or thread the buffer down as an argument (as `_syrk_rec!` does) — or, if \
        it is genuinely legal, add it to test/workspace_lint_baseline.txt WITH A WRITTEN REASON." new = r.new
    isempty(r.stale) || @error "workspace lint: STALE baseline entry — this pair no longer occurs. Delete \
        the line (and its reason) so the baseline cannot rot into a rubber stamp." stale = r.stale
    @test isempty(r.new)
    @test isempty(r.stale)
end
