# Gates the experiment-table bounds invariant in the suite. The scanner lives in expint_lint.jl
# (standalone: `julia test/expint_lint.jl`).
#
# Why a gate and not a comment: `_EXPINT`/`_EXPFLAG` are read `@inbounds` everywhere, so indexing past
# the declared length is an out-of-bounds READ that produces no error, no allocation and no wrong
# answer — the stale value past the end reads as 0, which is indistinguishable from "knob off". It
# shipped once (bf44b8b) and was caught only because a probe happened to WRITE the same index.
@testitem "expint lint: every experiment-table index is within its declared array" begin
    include(joinpath(@__DIR__, "expint_lint.jl"))    # defines expint_scan; the CLI guard skips execution
    v = expint_scan()
    isempty(v) || @error "expint lint: experiment-table index out of range.\n" * join(v, "\n")
    @test isempty(v)
end
