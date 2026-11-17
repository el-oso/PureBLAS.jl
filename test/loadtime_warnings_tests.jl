# The StrictMode checks in src/verify.jl (`@verify_strict`, `@strict_policy`) run at every precompile.
# A gating failure there stops the build; a heuristic one only logs a warning. src/verify.jl records each such
# warning in `PureBLAS._LOADTIME_WARNINGS`, saved with the package image, so this item reads the warnings
# of the build under test without compiling anything, and fails on any one not listed in
# test/loadtime_warnings_baseline.txt.

@testitem "Load-time StrictMode block logs no new warnings" tags = [:checks] begin
    using PureBLAS, StrictMode
    # With checks disabled the checks expand to bare calls and logs nothing, so a pass would be vacuous.
    @test StrictMode.checks_enabled()
    file = joinpath(@__DIR__, "loadtime_warnings_baseline.txt")
    baseline = Set(l for l in eachline(file) if !isempty(strip(l)) && !startswith(l, '#'))
    # One key per warning, `<guarantee>\t<target>`, from the first two lines StrictMode's `showerror` prints.
    function key(msg)
        m = match(r"^StrictViolation \(@(\w+)\)[^\n]*\n\s*target:\s*([^\n]*)", msg)
        return isnothing(m) ? msg : m[1] * "\t" * strip(m[2])
    end
    new = filter(!in(baseline), map(key, PureBLAS._LOADTIME_WARNINGS))
    isempty(new) || println("load-time StrictMode warnings not in the baseline:\n  ", join(new, "\n  "))
    @test isempty(new)
end
