# Protects per-thread scratch ownership. The failure this catches is SILENT: a kernel that yields lets a
# second task run on the same thread, and both then use that thread's arena and pools — overlapping
# borrows, packing into the same bytes, a `resize!` under a live pointer. Every answer can still look
# plausible, and no correctness test fails, because it only happens when two tasks interleave.
# See src/arena.jl (ownership), test/arena_tests.jl (the per-thread and concurrency items).
@testitem "yield lint: no unreviewed task-switch points in src/" begin
    include(joinpath(@__DIR__, "yield_lint.jl"))
    r = yield_violations()
    isempty(r.new) || @error "Task-switch point(s) in src/. Scratch is owned PER THREAD, so a yield \
        inside a call lets another task use this thread's arena and pools. Remove it from the kernel, or \
        add the line to test/yield_lint_baseline.txt WITH A WRITTEN REASON (it must hold no @scope). If a \
        kernel must yield, switch the owners to Base.OncePerTask instead." new = r.new
    isempty(r.stale) || @error "Stale yield baseline entr(ies): the line no longer occurs in src/. \
        Delete it — a fixed site must not leave a rubber stamp behind." stale = r.stale
    @test isempty(r.new)
    @test isempty(r.stale)
end
