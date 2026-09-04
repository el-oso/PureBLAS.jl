# Keeps pointer-backed operands on the fast path. Companion to req8_lint_tests.jl and
# estimator_lint_tests.jl, and for the same reason all three exist: the failure is SILENT. A
# `PtrMatrix`/`PtrVector` that misses an `isa StridedMatrix` gate still computes the right answer, so no
# correctness test fails, and the gate sweep never calls through that container, so no cell moves either.
# Five instances shipped before this lint existed — the worst sent every Mode-1/LBT `getrf` down the
# scalar generic tail. See ../kb/findings/strided-gates-drop-pointer-operands-to-scalar.md.
@testitem "fastpath lint: no unreviewed `isa Strided*` gates in src/" begin
    include(joinpath(@__DIR__, "fastpath_lint.jl"))
    r = fastpath_violations()
    isempty(r.new) || @error "Fast-path gate(s) written as a bare `isa Strided*` test. `PtrMatrix`/\
        `PtrVector` are NOT in that closed union, so a pointer-backed operand silently takes the generic \
        scalar path. Use `_strided1` (matrix) or `_dense1` (vector); if the narrow form is genuinely \
        required, add the line to test/fastpath_lint_baseline.txt WITH A WRITTEN REASON." new = r.new
    isempty(r.stale) || @error "Stale fastpath baseline entr(ies): the line no longer occurs in src/. \
        Delete it — a fixed site must not leave a rubber stamp behind." stale = r.stale
    @test isempty(r.new)
    @test isempty(r.stale)
end
