@testitem "per-thread owner lint: no unreviewed OncePerThread owner in src/" tags = [:checks] begin
    include(joinpath(@__DIR__, "perthread_lint.jl"))
    r = perthread_violations()
    isempty(r.new) || @info "unreviewed OncePerThread owners" r.new
    isempty(r.stale) || @info "stale baseline entries" r.stale
    @test isempty(r.new)
    @test isempty(r.stale)
end
