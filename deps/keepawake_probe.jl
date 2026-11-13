# Started by deps/build.jl with several threads. Finds the shortest idle gap after which a threaded
# job runs slower than after the reference gap, and prints, as the last line of stdout, that gap in
# ns, `_MT_KEEPAWAKE_NONE` when every gap matches the reference, or `_MT_KEEPAWAKE_UNCLEAR` when
# neither holds. The per-gap table goes to stderr, i.e. the build log.
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

# THE REFERENCE GAP. An idle Julia thread spins for `JULIA_THREAD_SLEEP_THRESHOLD` ns before it
# sleeps, and deps/build.jl sets that for this process. Below it the workers never sleep and the
# job runs faster than at any longer gap whatever the core does, so the reference is the first gap
# clear of it: twice the threshold, where the workers have just gone to sleep and their cores have
# idled for at most about the threshold. A host already slow there reads as slow at every longer
# gap, which is an onset at the second rung and a window of nearly `_MT_KEEPAWAKE_GAP_NS`.
# Gaps then rise in steps of √2 to the first above `_MT_KEEPAWAKE_GAP_NS`: an onset beyond that
# gives window 0.
const SLEEP_NS = parse(Int, ENV["JULIA_THREAD_SLEEP_THRESHOLD"])
function ladder()
    gs = [2 * SLEEP_NS]
    while last(gs) <= _MT_KEEPAWAKE_GAP_NS
        push!(gs, round(Int, 2 * SLEEP_NS * 2^(length(gs) / 2)))
    end
    return gs
end

# Samples per gap: the quartiles of 31 are stable to a few samples, and the whole ladder then costs
# about 31 × (sum of the gaps), under two seconds.
const SAMPLES = 31

times(g) = [s.time for s in (@be idle(g) _ -> job() evals = 1 samples = SAMPLES seconds = Inf).samples]

# Each gap compared with the reference by quartiles: `:slow` when the two interquartile ranges do
# not overlap and the gap's is higher, `:same` when the gap's median is inside the reference's
# range, `:unclear` otherwise.
function classify(t, ref)
    r1, r3 = quantile(ref, [0.25, 0.75])
    q1, q2 = quantile(t, [0.25, 0.5])
    q1 > r3 && return :slow
    r1 <= q2 <= r3 && return :same
    return :unclear
end

# The onset in ns; `_MT_KEEPAWAKE_NONE` only when every gap matched the reference, which is the
# positive evidence of no penalty that window 0 needs; `nothing` when the evidence is mixed, so
# that nothing is stored and the default window stays.
function onset()
    gs = ladder()
    times(gs[1])   # discarded: the first block after start-up reads slow and wide on every part measured
    ts = [times(g) for g in gs]
    cls = [classify(t, ts[1]) for t in ts]
    println(stderr, "threads $NT, buffer $(LEN * sizeof(Float64)) B per thread, sleep threshold $SLEEP_NS ns")
    for (g, t, c) in zip(gs, ts, cls)
        q1, q2, q3 = quantile(t, [0.25, 0.5, 0.75])
        println(stderr, "gap $(round(g / 1.0e6; digits = 2)) ms: median $(round(q2 * 1.0e6; digits = 1)) us [$(round(q1 * 1.0e6; digits = 1)), $(round(q3 * 1.0e6; digits = 1))]  $c")
    end
    # The onset is the shortest gap from which every longer gap is slow, so one noisy gap does not
    # make one.
    for i in 2:length(gs)
        all(==(:slow), cls[i:end]) && return gs[i]
    end
    all(==(:same), cls[2:end]) && return _MT_KEEPAWAKE_NONE
    return nothing
end

let o = onset()
    println(isnothing(o) ? _MT_KEEPAWAKE_UNCLEAR : o)
end
