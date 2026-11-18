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
    # One key per finding, `<guarantee>\t<target>`. A `StrictViolation` names them in its first two
    # lines; a `StrictMode: N finding(s)` report has one `[✗ <guarantee> <code>] <target> — …` line per
    # finding. The source locations those messages carry are not part of the key.
    function keys_of(msg)
        m = match(r"^StrictViolation \(@(\w+)\)[^\n]*\n\s*target:\s*([^\n]*)", msg)
        isnothing(m) || return [m[1] * "\t" * strip(m[2])]
        ks = [f[1] * "\t" * f[2] for f in eachmatch(r"\[✗ (\w+) SM\d+\] ([^\n]*?) — ", msg)]
        return isempty(ks) ? [msg] : ks
    end
    new = filter(!in(baseline), unique(mapreduce(keys_of, vcat, PureBLAS._LOADTIME_WARNINGS; init = String[])))
    isempty(new) || println("load-time StrictMode warnings not in the baseline:\n  ", join(new, "\n  "))
    @test isempty(new)
end
