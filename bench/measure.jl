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

using Statistics

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

end # module
