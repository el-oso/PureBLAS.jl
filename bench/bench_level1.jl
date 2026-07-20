# BLAS-1 parity benchmark: PureBLAS native API vs OpenBLAS, single-threaded, Float64.
#
# Noise-robust by INTERLEAVING: each round times m in-place reps of OpenBLAS, then immediately m
# reps of PureBLAS, so both see the same clock state — frequency drift (e.g. unpinned opportunistic
# boost) cancels in the per-round ratio. We report the ratio of medians over many rounds. Methodology
# per PureFFT CLAUDE.md: single-thread BLAS, @noinline function barrier around the timed region,
# in-place reps (data may grow to Inf — FP throughput is identical and far more stable than
# copy-per-rep), MEDIAN not min. Pin the core with `taskset -c 2`. Gate: ratio = OpenBLAS/PureBLAS
# median ≥ 0.96 (>1 means PureBLAS faster).

using PureBLAS, LinearAlgebra, Printf, Statistics
import LinearAlgebra.BLAS as B
BLAS.set_num_threads(1)

const SINK = Ref(0.0)
const SIZES = (1_000, 10_000, 100_000, 1_000_000)
m_for(n) = max(2, round(Int, 4_000_000 ÷ n))   # ~constant total work per round

@noinline _run(work) = work()   # function barrier: timed region is just the rep loop

function paired(obwork, pbwork; m, rounds)
    _run(obwork); _run(pbwork)                                   # warmup + compile
    obs = Vector{Float64}(undef, rounds); pbs = similar(obs)
    for r in 1:rounds
        t0 = time_ns(); v1 = _run(obwork)
        t1 = time_ns(); v2 = _run(pbwork)
        t2 = time_ns()
        @inbounds obs[r] = (t1 - t0) / m
        @inbounds pbs[r] = (t2 - t1) / m
        SINK[] += v1 + v2
    end
    return median(obs), median(pbs)
end

function report(op, n, ob, pb, acc)
    r = ob / pb
    push!(acc, r)
    return @printf("%-7s %9d %13.1f %13.1f %8.2fx   %s\n", op, n, ob, pb, r, r >= 0.96 ? "PASS" : "FAIL")
end

function run_table()
    acc = Float64[]
    @printf("%-7s %9s %13s %13s %9s   %s\n", "op", "n", "OpenBLAS(ns)", "PureBLAS(ns)", "ratio", "gate")
    for n in SIZES
        x = randn(n); y = randn(n); a = 1.0000001; m = m_for(n); R = 151
        report(
            "axpy", n, paired(
                () -> begin
                    for _ in 1:m
                        B.axpy!(a, x, y)
                    end; y[1]
                end,
                () -> begin
                    for _ in 1:m
                        PureBLAS.axpy!(y, a, x)
                    end; y[1]
                end; m, rounds = R
            )..., acc
        )
        report(
            "scal", n, paired(
                () -> begin
                    for _ in 1:m
                        B.scal!(a, x)
                    end; x[1]
                end,
                () -> begin
                    for _ in 1:m
                        PureBLAS.scal!(a, x)
                    end; x[1]
                end; m, rounds = R
            )..., acc
        )
        report(
            "copy", n, paired(
                () -> begin
                    for _ in 1:m
                        B.blascopy!(n, x, 1, y, 1)
                    end; y[1]
                end,
                () -> begin
                    for _ in 1:m
                        PureBLAS.blascopy!(y, x)
                    end; y[1]
                end; m, rounds = R
            )..., acc
        )
        report(
            "dot", n, paired(
                () -> begin
                    s = 0.0; for _ in 1:m
                        s += dot(x, y)
                    end; s
                end,
                () -> begin
                    s = 0.0; for _ in 1:m
                        s += PureBLAS.dot(x, y)
                    end; s
                end; m, rounds = R
            )..., acc
        )
        report(
            "nrm2", n, paired(
                () -> begin
                    s = 0.0; for _ in 1:m
                        s += B.nrm2(x)
                    end; s
                end,
                () -> begin
                    s = 0.0; for _ in 1:m
                        s += PureBLAS.nrm2(x)
                    end; s
                end; m, rounds = R
            )..., acc
        )
        report(
            "asum", n, paired(
                () -> begin
                    s = 0.0; for _ in 1:m
                        s += B.asum(x)
                    end; s
                end,
                () -> begin
                    s = 0.0; for _ in 1:m
                        s += PureBLAS.asum(x)
                    end; s
                end; m, rounds = R
            )..., acc
        )
    end
    geo = exp(mean(log.(acc)))
    return @printf(
        "\n geomean ratio = %.3fx   min = %.3fx   ops below 0.96 = %d/%d\n",
        geo, minimum(acc), count(<(0.96), acc), length(acc)
    )
end

for pass in 1:2
    println("\n===== pass $pass =====")
    run_table()
end
println("\n(sink ", SINK[], ")")
