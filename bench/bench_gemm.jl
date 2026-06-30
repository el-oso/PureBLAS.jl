# dgemm parity benchmark: PureBLAS.gemm! vs OpenBLAS, single-threaded, Float64, square.
# Interleaved (drift-robust) median timing; GEMM is compute-bound so less noisy than L1. Pin with
# `taskset -c 2`. Gate: ratio = OpenBLAS/PureBLAS ≥ 0.96. GFLOP/s = 2n³ / time.

using PureBLAS, LinearAlgebra, Printf, Statistics
import LinearAlgebra.BLAS as B
BLAS.set_num_threads(1)

const SINK = Ref(0.0)
@noinline _run(f) = f()

function paired(ob, pb; rounds)
    _run(ob); _run(pb)
    o = Vector{Float64}(undef, rounds); p = similar(o)
    for r in 1:rounds
        t0 = time_ns(); v1 = _run(ob)
        t1 = time_ns(); v2 = _run(pb)
        t2 = time_ns()
        o[r] = t1 - t0; p[r] = t2 - t1; SINK[] += v1 + v2
    end
    return median(o), median(p)
end

@printf("%6s %14s %14s %9s   %s\n", "n", "OpenBLAS GF/s", "PureBLAS GF/s", "ratio", "gate")
ratios = Float64[]
for n in (64, 128, 256, 512, 1024, 2048)
    A = randn(n, n); Bm = randn(n, n); C = zeros(n, n)
    rounds = n <= 256 ? 31 : (n <= 1024 ? 11 : 5)
    mo, mp = paired(
        () -> begin B.gemm!('N', 'N', 1.0, A, Bm, 0.0, C); C[1] end,
        () -> begin PureBLAS.gemm!(C, A, Bm; alpha = 1.0, beta = 0.0); C[1] end;
        rounds)
    gf = 2.0 * n^3
    r = mo / mp
    push!(ratios, r)
    @printf("%6d %14.1f %14.1f %8.2fx   %s\n", n, gf / mo, gf / mp, r, r >= 0.96 ? "PASS" : "FAIL")
end
@printf("\n geomean ratio = %.3fx   min = %.3fx\n", exp(mean(log.(ratios))), minimum(ratios))
println("(sink ", SINK[], ")")
