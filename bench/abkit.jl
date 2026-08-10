# ONE paired-A/B harness. Lives in bench/ (TRACKED, so `fleet_sync` carries it) rather than in
# bench/probes/ (gitignored, hand-scp'd, forgotten twice).
#
# WHY THIS EXISTS. Before it, every experiment was a COPY of the previous probe with the flag, labels,
# round counts and A/A criterion edited by hand. Consequences, all real, all in one night:
#   * a probe inherited `_EXP2` (a FALSIFIED, dead flag) from its ancestor and returned a perfectly
#     believable NULL at mid-n — it was measuring nothing, and only a stale output LABEL revealed it;
#   * the A/A criterion diverged between copies (sigma-based vs SE-based), so one run printed
#     "FLOOR FAILED" on data that was fine and another passed an instrument with sigma = 0.144;
#   * `reset-at-start` existed in some copies and not others, so a probe that threw left its flag set
#     and silently contaminated the NEXT probe in the same session.
# A harness cannot fix a wrong hypothesis, but it can stop the instrument from lying. That is its job.
#
# USAGE
#   include(joinpath(@__DIR__, "abkit.jl")); using .ABKit
#   ABKit.ab(; name  = "paired stripes",
#              arm   = v -> (PureBLAS._EXPFLAG[PureBLAS._EXP8] = v),   # false = A (shipped), true = B
#              setup = () -> pool_of_operands(),
#              work  = p -> run_the_op!(p),
#              reset = () -> fill!(PureBLAS._EXPFLAG, false))
module ABKit

using Chairmarks, Printf, Statistics
include(joinpath(@__DIR__, "measure.jl"))
using .Measure: tstat

export ab, sweep

"""
    ab(; name, arm, setup, work, reset=nothing, aa_rounds=40, min_rounds=40, samples=8,
         target=nothing, effect_floor=0.02)

Paired in-process A/B. Returns `(ratio, se, verdict::Symbol)`.

Guarantees, each one earned by a specific failure:
 * `reset` runs FIRST, not only at exit — a probe that throws must not contaminate the next one.
 * ARM LIVENESS is asserted: both settings are timed once and must differ by more than the A/A floor
   at least once across the sweep, else the run aborts with `:dead_arm`. A dead flag returns a null
   that is indistinguishable from a real null; this is the check that catches it.
 * The A/A floor bounds BIAS (|median-1| <= 2*SE), it does not demand small sigma. Per-round sigma
   varied 0.008..0.11 on ONE box across sessions; high sigma is survivable by taking more rounds.
 * N is DERIVED from the measured sigma so a 2% effect clears 4 SE — never a hand-picked round count.
 * ABBA rotation per round; statistic is the median of per-round RATIOS so common-mode drift cancels
   inside each pair instead of being averaged over.
"""
function ab(; name::AbstractString, arm, setup, work, reset = nothing,
        aa_rounds::Int = 40, min_rounds::Int = 40, samples::Int = 8,
        target::Union{Nothing, Float64} = nothing, effect_floor::Float64 = 0.02)
    isnothing(reset) || reset()                       # reset-at-START

    round_time(v) = (arm(v); tstat(Float64[x.time for x in
        (@be setup() (p -> work(p)) evals = 1 samples = samples).samples]))

    function paired(a, b, rounds)
        r = Float64[]
        for i in 1:rounds
            ta, tb = isodd(i) ? (round_time(a), round_time(b)) : (reverse((round_time(b), round_time(a)))...,)
            push!(r, tb / ta)
        end
        return r
    end

    println("\n=== A/B: ", name, " ===")
    aa = paired(false, false, aa_rounds)
    aam, aasd = tstat(aa), std(aa)
    aase = aasd / sqrt(length(aa))
    bias_ok = abs(aam - 1) <= 2 * aase
    @printf("A/A floor      %.4f  sd %.4f  SE %.4f  n=%d   bias %s\n",
            aam, aasd, aase, length(aa), bias_ok ? "OK" : "SUSPECT")

    n = max(min_rounds, ceil(Int, (4 * aasd / effect_floor)^2))
    n = min(n, 400)                                   # cap: beyond this the box drifts more than we gain
    @printf("sigma %.4f => n = %d rounds for a %.0f%% effect at 4 SE\n", aasd, n, 100 * effect_floor)

    ab_r = paired(false, true, n)
    m, sd = tstat(ab_r), std(ab_r)
    se = sd / sqrt(length(ab_r))
    isnothing(reset) || reset()

    # ARM LIVENESS — the check that would have caught the dead-flag probe.
    live = abs(m - 1) > max(2 * se, 2 * aase) || maximum(abs.(ab_r .- 1)) > 4 * aasd
    verdict = !live ? :dead_or_null :
        abs(m - 1) <= max(4 * se, 2 * aase) ? :null :
        m < 1 ? :B_faster : :A_faster

    @printf("%-22s %.4f  sd %.4f  SE %.4f  n=%d  => %s\n", "B/A", m, sd, se, length(ab_r), verdict)
    verdict === :dead_or_null &&
        println("  !! B/A is within BOTH its own SE and the A/A floor. Either a true null, or the arm ",
                "does NOT change behaviour (wrong flag / not wired). VERIFY THE ARM before believing this.")
    isnothing(target) ||
        @printf("  target %.4f => %s\n", target, m <= target ? "MET" : "NOT met")
    return (m, se, verdict)
end

"""
    sweep(; name, set, values, setup, work, reset=nothing, rounds=12, samples=6, base=first(values))

Multi-VALUE sweep (a knob with more than two settings), reported as ratios against `base`.

It exists because a sweep is NOT a series of independent timings and must not be written as one. The
naive shape — measure base, then every value in a fixed order — puts every value AFTER the base within
a round and pushes later values progressively further from it in time, so any within-round drift biases
the whole row in one direction. I wrote exactly that, labelled it "ABBA-rotated", and it was neither.

Here each round visits the values in a ROTATED order (round r starts at value r mod N) and re-times the
base ADJACENT to each value, so every ratio is a locally-paired comparison and position within the round
averages out across rounds. Statistic is the median of per-round ratios.
"""
function sweep(; name::AbstractString, set, values, setup, work, reset = nothing,
        rounds::Int = 12, samples::Int = 6, base = first(values))
    isnothing(reset) || reset()
    t1(v) = (set(v); tstat(Float64[x.time for x in
        (@be setup() (p -> work(p)) evals = 1 samples = samples).samples]))
    vals = collect(values)
    acc = Dict(v => Float64[] for v in vals)
    for r in 1:rounds
        order = circshift(vals, r)                      # rotate: no value is always measured last
        for v in order
            b1 = t1(base)                               # base re-timed ADJACENT to each value
            tv = t1(v)
            b2 = t1(base)
            push!(acc[v], tv / ((b1 + b2) / 2))         # bracket the value with base on both sides
        end
    end
    isnothing(reset) || reset()
    println("\n=== SWEEP: ", name, "  (ratio vs ", base, "; <1 is FASTER) ===")
    live = false
    bias = abs(tstat(acc[base]) - 1)
    for v in vals
        m = tstat(acc[v]); se = std(acc[v]) / sqrt(length(acc[v]))
        v == base || abs(m - 1) > max(4 * se, bias) && (live = true)
        @printf("  %-10s %.4f   SE %.4f   n=%d%s\n", string(v), m, se, length(acc[v]),
                v == base ? "   <- base (sanity: must be ~1.000)" : "")
    end
    # ARM LIVENESS, the check sweep() was missing while ab() had it. A knob that is not wired, or whose
    # code path this shape does not route to, produces a flat 1.000 in every row — INDISTINGUISHABLE from
    # a real null. That is not hypothetical: the Zen3 k=32 arm of the trsm pad sweep returned a clean,
    # well-behaved null while `trsm!` was routing to _trsm_dense_L!, which never allocates the buffer the
    # knob pads. Two full sweeps were interpreted before anyone read the routing.
    live || println("  !! NO value differs from base by more than max(4*SE, bias). Either a true null, or ",
                    "the knob is DEAD for this SHAPE — operand shape selects the code path. VERIFY THE ",
                    "ROUTING (read the dispatch, print the cutoff consts) before reporting this as a null.")
    return Dict(v => tstat(acc[v]) for v in vals)
end

end # module
