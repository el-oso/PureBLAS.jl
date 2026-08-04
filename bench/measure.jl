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
using Chairmarks: @be

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

  * ARMS BACK-TO-BACK WITHIN A ROUND — seconds apart, so drift is common-mode and cancels. Order
    alternation was MEASURED UNNECESSARY (see abba_nulltest.jl); the adjacency is what matters, not the
    flipping. Cross-RUN comparison, by contrast, was measured at up to 3% of pure methodology.
  * FRESH SETUP PER ROUND — `setup()` is re-run each round, matching plots.jl's `@be … evals=1`. A warm
    single buffer ranked an unroll the wrong way once already.
  * MEDIAN, per round and across rounds. Never min, never mean.
  * BOOTSTRAP CI over the per-round ratios, so the answer carries its own resolution and a difference
    smaller than the interval is reported as such instead of being acted on.

`arms` is a vector of `name => f` where `f(ctx)` does the work; `ctx` is whatever `setup()` returned.
"""
function ab(arms::AbstractVector; rounds::Int = 8, reps::Int = 1, setup = () -> nothing,
            samples::Int = 48, seconds::Float64 = 0.5, nboot::Int = 4000, seed::Int = 20260804,
            alternate::Bool = false)   # measured unnecessary — see the null test cited below
    n = length(arms)
    ts = [Float64[] for _ in 1:n]                          # per-round MEDIAN of that window's samples
    for r in 1:rounds
        # NO ORDER ALTERNATION — measured unnecessary, not assumed. bench/probes/abba_nulltest.jl runs two
        # IDENTICAL arms; an unbiased harness must read 1.000. Fixed order does:
        #     n=1e3  1.0002 [0.9999, 1.0008]     n=1e5  0.9998 [0.9989, 1.0011]
        # and alternating does not improve it (at n=1e5 it is marginally worse: 1.0005 [0.9982, 1.0033]).
        # Arms run back-to-back within a round — seconds apart — so there is no drift for an order flip to
        # cancel. Alternation would only matter if arms were separated by minutes, which is the
        # CACHED-REFERENCE case, where the separation is days and no ordering fixes it.
        # `alternate=true` is kept solely so the null test can re-check this claim on new hardware.
        order = (alternate && iseven(r)) ? reverse(1:n) : (1:n)
        for i in order
            f = arms[i][2]
            # CHAIRMARKS, exactly as plots.jl uses it — `evals=1` re-runs `setup` per SAMPLE, so inputs
            # are fresh and the regime matches the gate. Hand-rolling this was a mistake twice over: it
            # is the documented "ad-hoc harness" trap (gate numbers must come from the approved path),
            # and a hand loop yields ONE timing per window where @be yields hundreds — which is why the
            # hand-rolled intervals were needlessly wide.
            b = @be setup() (c -> begin
                for _ in 1:reps
                    f(c)
                end
            end) evals = 1 samples = samples seconds = seconds
            push!(ts[i], tstat(Float64[s.time for s in b.samples]))
        end
    end
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
