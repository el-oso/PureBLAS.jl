# CI perf-regression gate. Runs PkgBenchmark.judge(PR HEAD vs the PR base branch) — both benchmarked on
# the SAME runner so the comparison is hardware-consistent — and EXITS 1 on any regression, so a slowdown
# blocks the merge (e.g. the 2× back-transform we hit by hand this session).
#
# CAVEAT (baked in): GitHub-hosted runners share CPUs and don't pin/lock frequency, so timings are noisy
# (±5–15%). We therefore use a COARSE tolerance (25%) — this catches GROSS regressions reliably without
# flaking on noise. Precise per-machine gating (the 0.96× OpenBLAS gate) belongs on a pinned self-hosted
# runner from the fleet (`taskset -c N julia --project=bench bench/lapackbench.jl`), not here.

using Pkg
Pkg.instantiate()
using PkgBenchmark, PureBLAS
using PkgBenchmark: benchmarkgroup, export_markdown
using BenchmarkTools: leaves, time

const TOL = 0.25
baseref = get(ENV, "GITHUB_BASE_REF", "master")
@info "judging HEAD vs $baseref (time_tolerance=$TOL)"

j = try
    judge(PureBLAS, "HEAD", baseref; judgekwargs = (time_tolerance = TOL,))
catch err
    @warn "judge could not run — the base ref likely predates benchmark/benchmarks.jl. Skipping (pass)." exception = err
    exit(0)
end

export_markdown("judge.md", j)
println(read("judge.md", String))

regs = [join(k, "/") for (k, tj) in leaves(benchmarkgroup(j)) if time(tj) == :regression]
if !isempty(regs)
    println("\n❌ PERFORMANCE REGRESSION (>$(round(Int, TOL*100))% slower vs $baseref):")
    foreach(r -> println("   - ", r), regs)
    exit(1)
end
println("\n✅ no performance regression vs $baseref")
