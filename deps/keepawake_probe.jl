# Started by deps/build.jl with several threads. Finds the shortest idle gap after which a threaded
# job runs slower than it does after the shortest gap, and prints that gap in ns as the last line of
# stdout — `_MT_KEEPAWAKE_NONE` when no gap on the ladder slows it. The per-gap table goes to stderr, i.e. the build log.
#
# THE JOB is one read of a private buffer per thread, half that core's L2: resident while the core
# stays awake, refetched when an idle state flushed the core's caches, and the exit latency of a
# deeper state lands in it too. Between two jobs the driver spins for the gap, with no memory
# traffic, while every worker thread sleeps — Julia's scheduler, with no keep-awake of its own.
# Repeating one gap many times in a row lets the OS idle governor learn it, as it does between the
# calls of a real workload.

using Chairmarks: @be
using Statistics: median, quantile
using CPUSummary: cache_size

include(joinpath(@__DIR__, "..", "src", "blas3", "keepawake.jl"))

const NT = Threads.nthreads()
const LEN = max(Int(cache_size(Val(2))), Int(cache_size(Val(1)))) ÷ (2 * sizeof(Float64))
LEN > 0 || error("no cache size detected")

const BUFS = [Float64[] for _ in 1:NT]
const SINK = zeros(NT)

# `:static` puts iteration i on thread i, so each buffer is first touched, and later read, by the
# same thread.
Threads.@threads :static for i in 1:NT
    BUFS[i] = fill(1.0, LEN)
end

function job()
    Threads.@threads :static for i in 1:NT
        SINK[i] = sum(BUFS[i])
    end
    return nothing
end

function idle(g)
    t = time_ns() + g
    while time_ns() < t
    end
    return nothing
end

# Gaps from 0.5 ms in steps of √2, up to the first above `_MT_KEEPAWAKE_GAP_NS`: any onset beyond
# that gives window 0, as no onset does.
function ladder()
    gs = [500_000]
    while last(gs) <= _MT_KEEPAWAKE_GAP_NS
        push!(gs, round(Int, 500_000 * 2^(length(gs) / 2)))
    end
    return gs
end

# Samples per gap: the quartiles of 31 are stable to a few samples, and the whole ladder then costs
# about 31 × (sum of the gaps), under two seconds.
const SAMPLES = 31

times(g) = [s.time for s in (@be idle(g) _ -> job() evals = 1 samples = SAMPLES seconds = Inf).samples]

function onset()
    gs = ladder()
    ts = [times(g) for g in gs]
    # The kernel's own noise: the upper quartile after the shortest gap. A gap is slow when its
    # median is above that.
    bound = quantile(ts[1], 0.75)
    slow = [median(t) > bound for t in ts]
    println(stderr, "threads $NT, buffer $(LEN * sizeof(Float64)) B per thread, bound $(round(bound * 1.0e6; digits = 1)) us")
    for (g, t, s) in zip(gs, ts, slow)
        q1, q2, q3 = quantile(t, [0.25, 0.5, 0.75])
        println(stderr, "gap $(round(g / 1.0e6; digits = 2)) ms: median $(round(q2 * 1.0e6; digits = 1)) us [$(round(q1 * 1.0e6; digits = 1)), $(round(q3 * 1.0e6; digits = 1))]", s ? "  slow" : "")
    end
    # The onset is the shortest gap from which every longer gap is slow too, so one noisy gap does
    # not count.
    for i in 2:length(gs)
        all(slow[i:end]) && return gs[i]
    end
    return nothing
end

let o = onset()
    println(isnothing(o) ? _MT_KEEPAWAKE_NONE : o)
end
