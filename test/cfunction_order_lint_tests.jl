# Protects requirement 4 (trim-compatibility) against a defect that NO other check in a normal run can
# see. A constant `@cfunction` resolves its callee when the enclosing method is defined, so a callee
# defined later in include order loads fine, passes every test, and breaks `juliac --trim` only.
# See test/cfunction_order_lint.jl for why StrictMode cannot reach it and why the authoritative
# juliac item (gated on PUREBLAS_JULIAC_BUILD=1) reports green without running.
@testitem "cfunction order lint: a constant @cfunction's callee is defined before it" begin
    include(joinpath(@__DIR__, "cfunction_order_lint.jl"))

    # POSITIVE CONTROL FIRST. A scan that reports clean for everything is worthless evidence, and this
    # one is lexical, so its patterns are the thing most likely to rot. Both shapes must be recognised:
    # the constant form that binds at definition, and the `$f` form that does not.
    @test !isnothing(match(_CO_REF, "_sme_dot_cf() = @cfunction(_sme_dot_cabi, Cvoid,"))
    @test isnothing(match(_CO_REF, "return @cfunction(\$f, Cvoid,"))
    @test occursin(_co_defpat("sgetrf_64_"), "Base.@ccallable function sgetrf_64_(")
    @test occursin(_co_defpat("_sme_dot_cabi"), "function _sme_dot_cabi(o::Ptr{Float64},")
    @test !occursin(_co_defpat("_sme_dot_cabi"), "    _sme_dot_cf() = throw(AssertionError(")

    # Include order must be read from the module, not assumed: a callee's position only means
    # anything relative to where it is referenced.
    files = cfunction_order_files()
    @test length(files) > 1
    @test any(f -> endswith(f, "PureBLAS.jl"), files)

    v = cfunction_order_violations()
    isempty(v) || @error "Constant `@cfunction` with a callee defined later in include order. The \
        module still loads and every test passes; `juliac --trim` fails with `UndefVarError` and \
        nothing else reports it. Move the trampoline after the callee, or the callee before it." violations = v
    @test isempty(v)
end
