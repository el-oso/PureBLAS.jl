# Authoritative L1-complex gate check on idle galen. Interleaved sweep (plots metric), high rounds.
using PureBLAS, LinearAlgebra, Statistics, Printf
import LinearAlgebra.BLAS as B
BLAS.set_num_threads(1); const SINK = Ref(0.0); @noinline _run(f) = f()
const T = ComplexF64
rep(s) = clamp(8_000_000 ÷ s, 30, 20000)
function sweep(mk, ob, pb, s; rounds = 40)
    reps = rep(s); rs = Float64[]
    for _ in 1:rounds
        c = mk(s)
        _run(() -> ob(c, 1)); _run(() -> pb(c, 1))
        t0 = time_ns(); v1 = _run(() -> ob(c, reps)); t1 = time_ns(); v2 = _run(() -> pb(c, reps)); t2 = time_ns()
        SINK[] += real(v1) + real(v2); push!(rs, (t1 - t0) / (t2 - t1))
    end
    return median(rs), std(rs)
end
a0 = T(1.0000001, 0.0)
xy(s) = (randn(T, s), randn(T, s)); x1(s) = (randn(T, s),)
println("=== zscal (real scalar a0) ===")
for s in (1000, 3000, 10000, 30000)
    r = sweep(
        x1, (c, m) -> (
            for _ in 1:m
                B.scal!(a0, c[1])
            end; c[1][1]
        ), (c, m) -> (
            for _ in 1:m
                PureBLAS.scal!(a0, c[1])
            end; c[1][1]
        ), s
    )
    @printf("  n=%-6d %.3f (σ%.3f)\n", s, r...)
end
println("=== zdotc ===")
for s in (1000, 3000, 10000, 30000, 100000)
    r = sweep(
        xy, (c, m) -> (
            s0 = zero(T); for _ in 1:m
                s0 += B.dotc(c[1], c[2])
            end; s0
        ), (c, m) -> (
            s0 = zero(T); for _ in 1:m
                s0 += PureBLAS.dot(c[1], c[2])
            end; s0
        ), s
    )
    @printf("  n=%-6d %.3f (σ%.3f)\n", s, r...)
end
println("=== dzasum ===")
for s in (1000, 3000, 10000, 30000, 100000)
    r = sweep(
        x1, (c, m) -> (
            s0 = 0.0; for _ in 1:m
                s0 += B.asum(c[1])
            end; s0
        ), (c, m) -> (
            s0 = 0.0; for _ in 1:m
                s0 += PureBLAS.asum(c[1])
            end; s0
        ), s
    )
    @printf("  n=%-6d %.3f (σ%.3f)\n", s, r...)
end
