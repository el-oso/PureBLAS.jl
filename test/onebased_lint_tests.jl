# Gates the one-based-indexing guard in the suite. The scanner lives in onebased_lint.jl
# (standalone: `julia test/onebased_lint.jl`).
@testitem "onebased lint: every public array-taking src method guards against offset axes" begin
    include(joinpath(@__DIR__, "onebased_lint.jl"))    # defines onebased_scan; the CLI guard skips execution
    v = onebased_scan()
    isempty(v) || @error "onebased lint: public methods without a one-based-indexing guard.\n" * join(v, "\n")
    @test isempty(v)
end

@testitem "onebased lint: flags an unguarded method and accepts guarded or forwarding ones" begin
    include(joinpath(@__DIR__, "onebased_lint.jl"))
    scan(src) = onebased_scan(["fake.jl" => src])
    bad = "function foo!(A::AbstractMatrix, x::AbstractVector)\n    x .= A[1, :]\nend\n"
    @test length(scan(bad)) == 1
    @test occursin("foo!", only(scan(bad)))
    # guard that omits one array argument
    @test length(scan("function foo!(A::AbstractMatrix, x::AbstractVector)\n    Base.require_one_based_indexing(A)\n    x\nend\n")) == 1
    # guard nested in a branch does not count
    @test length(scan("function foo!(A::AbstractMatrix)\n    if true\n        Base.require_one_based_indexing(A)\n    end\n    A\nend\n")) == 1
    # short-form and `where`-bounded type variables are seen
    @test length(scan("foo!(A::AbstractMatrix) = A\n")) == 1
    @test length(scan("function foo!(A::M) where {M <: AbstractMatrix}\n    A\nend\n")) == 1
    # guarded, forwarding, underscore-named, and array-free methods pass
    good = "function foo!(A::AbstractMatrix, x::AbstractVector)\n    Base.require_one_based_indexing(A, x)\n    x\nend\n"
    @test isempty(scan(good))
    @test isempty(scan(good * "foo!(A::AbstractMatrix) = foo!(A, A[:, 1])\n"))
    @test isempty(scan("function _foo!(A::AbstractMatrix)\n    A\nend\n"))
    @test isempty(scan("function foo!(n::Integer)\n    n\nend\n"))
end
