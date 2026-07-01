# PkgBenchmark suite — the idiomatic self-regression guardrail. Measures ABSOLUTE times of PureBLAS
# routines so `PkgBenchmark.judge` can compare two states (a commit/branch or a saved result) and flag
# any that got slower beyond a tolerance — catching regressions that stay above the OpenBLAS gate (e.g.
# a back-transform slowdown). Complements `bench/svdbench.jl` (the ABSOLUTE gate vs OpenBLAS, ≥0.96×);
# `judge` here answers "did MY change make MY code slower?", which the gate check does not.
#
# Run/compare (from repo root):
#   using PkgBenchmark
#   base = benchmarkpkg(PureBLAS); PkgBenchmark.writeresults("benchmark/base.json", base)   # baseline
#   # …make changes…
#   j = judge(PureBLAS, "benchmark/base.json")            # or judge(PureBLAS, "HEAD", "HEAD~1")
#   PkgBenchmark.export_markdown("benchmark/judge.md", j) # :regression / :improvement per benchmark
# Tolerance: BenchmarkTools default 5% time; tighten via `judge(...; judgekwargs=(time_tolerance=0.03,))`.

using BenchmarkTools, PureBLAS, LinearAlgebra, Random
Random.seed!(1)
const SUITE = BenchmarkGroup()

# destructive routines: fresh copy per eval (evals=1, setup copies the pristine input) → pure-kernel time.
svd = SUITE["svd"] = BenchmarkGroup()
for n in (256, 512, 1024)
    A = randn(n, n)
    svd["gesvd_vec_$n"] = @benchmarkable PureBLAS.gesvd!(B; want_vectors = true) setup = (B = copy($A)) evals = 1
    svd["gesvd_val_$n"] = @benchmarkable PureBLAS.gesvd!(B; want_vectors = false) setup = (B = copy($A)) evals = 1
end

l3 = SUITE["l3"] = BenchmarkGroup()
for n in (512, 1024)
    A = randn(n, n); Bm = randn(n, n)
    l3["gemm_$n"] = @benchmarkable PureBLAS.gemm!(C, $A, $Bm) setup = (C = zeros($n, $n))
end

# Cholesky (potrf) / QR (geqrf) / LU (getrf) across their gate range (512–2048). evals=1 + a per-eval
# copy so each timed run factors a pristine input (these are destructive, in-place).
lap = SUITE["lapack"] = BenchmarkGroup()
for n in (512, 1024, 2048)
    A = randn(n, n); M = randn(n, n); SPD = M'M + n * I
    lap["getrf_$n"] = @benchmarkable PureBLAS.getrf!(B, ip) setup = (B = copy($A); ip = zeros(Int, $n)) evals = 1
    lap["geqrf_$n"] = @benchmarkable PureBLAS.geqrf!(B, t) setup = (B = copy($A); t = zeros(min($n, $n))) evals = 1
    lap["potrf_$n"] = @benchmarkable PureBLAS.potrf!(B; uplo = 'L') setup = (B = copy($SPD)) evals = 1
end
