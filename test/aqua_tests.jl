# Package-quality gate (Aqua.jl): method ambiguities, unbound type parameters, stale/undeclared deps,
# compat-bound presence, and type piracy. A registration-readiness check — keep it green.
@testitem "Aqua: package quality" begin
    using Aqua, PureBLAS
    Aqua.test_all(PureBLAS)
end
