# THE approved measurement statistic, in ONE place. Every benchmark — plots.jl, gate_verdict.jl, and any
# throwaway probe — must reduce timings through here.
#
# WHY THIS FILE EXISTS. CLAUDE.md has said "**median** times (not min)" since the start, and plots.jl has
# always formed `median(ref_q ./ pb_q)`. On 2026-08-03/04 the probe harnesses nevertheless used
# `minimum(@elapsed …)` and an early gate_verdict.jl used `mean` of log-ratios. Nobody noticed, because
# the rule lived in prose and every probe re-implemented "the number" from scratch.
#
# THE COST WAS NOT HYPOTHETICAL. `min` is optimistic AND tail-blind. A warm A/B on minima ranked the iamax
# unroll NB=2 ABOVE NB=4 at n=1e6; the gate's median ranked NB=2 15% WORSE at that size (identical best
# case, far worse typical case). The contradiction was then "explained" with a tail hypothesis, an AOCL
# kernel port, and an entry-path investigation — hours of work and several wrong published conclusions,
# all downstream of a silently substituted estimator. Sample count was never the problem: with `min`, MORE
# samples makes the estimator drift further from the median.
#
# So: one implementation, one name, and the reported number carries its own provenance.

module Measure

using Statistics, Random

export tstat, ESTIMATOR

"""Human-readable name of the approved estimator. Print this next to every number you report."""
const ESTIMATOR = "median"

"""
    tstat(ts) -> Float64

Reduce a vector of timings (or of per-round ratios) to the approved statistic. This is `median` and it is
NOT a free choice: the median is insensitive to the window tails that `min` ignores entirely and `mean`
over-weights, which is exactly the property the gate depends on.

If you think you need `minimum` or `mean` here, you are proposing a methodology change — raise it, do not
substitute it locally.
"""
tstat(ts::AbstractVector{<:Real}) = median(ts)

"""
    report(label, ts; unit="ms", scale=1e3) -> String

Format a measurement so the estimator and sample count travel WITH the number. A bare `169.9` hides which
statistic produced it; `169.9 (median of 5)` does not, and that is the difference between catching an
estimator swap in the first message and catching it in the fortieth.
"""
function report(label, ts::AbstractVector{<:Real}; unit = "ms", scale = 1.0e3)
    return string(label, " ", round(tstat(ts) * scale; digits = 4), " ", unit,
                  " (", ESTIMATOR, " of ", length(ts), ")")
end

"""
    ab(arms; rounds=8, reps, setup) -> Vector{(name, ratio, lo, hi)}

Run an A/B (or A/B/C/…) and return each arm's speed RELATIVE TO THE FIRST, with a bootstrap interval.

**This exists so that no probe ever writes a timing loop.** A hand-rolled loop is not one decision, it is
five — estimator, round count, warm-up, arm ordering, and whether inputs are fresh per sample — and on
2026-08-03/04 I got every one of them wrong at some point in bespoke probes, each failure costing hours.
Choosing badly is only possible if choosing is possible, so this makes those five decisions once:

  * ROUND-ALTERNATED (ABBA) — arms are interleaved within each round, so machine drift is common-mode
    and cancels in the ratio. Cross-run comparison was measured at up to 3% of pure methodology.
  * FRESH SETUP PER ROUND — `setup()` is re-run each round, matching plots.jl's `@be … evals=1`. A warm
    single buffer ranked an unroll the wrong way once already.
  * MEDIAN, per round and across rounds. Never min, never mean.
  * BOOTSTRAP CI over the per-round ratios, so the answer carries its own resolution and a difference
    smaller than the interval is reported as such instead of being acted on.

`arms` is a vector of `name => f` where `f(ctx)` does the work; `ctx` is whatever `setup()` returned.
"""
function ab(arms::AbstractVector; rounds::Int = 8, reps::Int = 1, setup = () -> nothing,
            nboot::Int = 4000, seed::Int = 20260804)
    n = length(arms)
    ts = [Float64[] for _ in 1:n]
    ctx0 = setup()                                        # warm code paths, not data
    for (_, f) in arms, _ in 1:2
        f(ctx0)
    end
    for r in 1:rounds
        ctx = setup()
        order = isodd(r) ? (1:n) : reverse(1:n)           # ABBA: alternate arm order each round
        for i in order
            f = arms[i][2]
            t0 = time_ns()                                 # estimator-ok: raw clock read; the REDUCTION
            for _ in 1:reps                                # below is the median, which is the rule
                f(ctx)
            end
            push!(ts[i], (time_ns() - t0) * 1.0e-9)
        end
    end
    base = tstat(ts[1])
    out = NamedTuple[]
    rng = Random.MersenneTwister(seed)
    for i in 1:n
        rel = [ts[1][r] / ts[i][r] for r in 1:rounds]      # per-round paired ratio vs arm 1
        m = tstat(rel)
        bs = [tstat(rel[rand(rng, 1:rounds, rounds)]) for _ in 1:nboot]
        sort!(bs)
        push!(out, (name = arms[i][1], ratio = m,
                    lo = bs[max(1, round(Int, 0.025nboot))], hi = bs[min(nboot, round(Int, 0.975nboot))],
                    secs = tstat(ts[i]), rounds = rounds))
    end
    return out
end

"""Print an `ab` result so every number carries its estimator, sample count and interval."""
function abshow(res; label = "")
    isempty(label) || println(label)
    println("  ", rpad("arm", 12), rpad("rel", 8), rpad("95% CI", 18), "$(ESTIMATOR) of $(res[1].rounds) rounds")
    for r in res
        println("  ", rpad(r.name, 12), rpad(round(r.ratio; digits = 3), 8),
                rpad(string("[", round(r.lo; digits = 3), ", ", round(r.hi; digits = 3), "]"), 18),
                round(r.secs * 1e3; digits = 3), " ms")
    end
    return nothing
end

end # module
