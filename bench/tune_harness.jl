# Robustness layer for per-machine tuning on HARDWARE WE DO NOT CONTROL.
#
# The fleet benchmarks run under `sudo bench/fleet_freqlock.sh lock` — governor pinned, boost off, base
# clock. A user's laptop offers none of that: the clock ramps through a boost window for the first tens of
# seconds, then sags as the package heats, and something else may start at any moment. A multi-minute
# sweep under those conditions measures early candidates on a fast machine and late ones on a slow one,
# which is a SYSTEMATIC bias toward whatever ran first — not noise that more samples would average out.
#
# Four defences, in order of how much they buy:
#
#  1. PAIRED WITHIN A ROUND (already in Measure.ab): arms run back-to-back, seconds apart, and the
#     statistic is the per-round RATIO. Any drift slower than a round is common-mode and cancels. This is
#     the big one and it is why we do not need to control the clock, only to detect when it moves fast.
#  2. STABILISE FIRST: burn the boost window before measuring anything, by loading the core until an
#     anchor workload stops getting slower. Measuring during the ramp is the one thing paired rounds
#     cannot fix, because the ramp is monotone across the whole run.
#  3. ANCHOR THE RUN: re-time the same fixed workload before and after each knob. If it moved more than
#     the tolerance, the machine changed underneath the measurement and the result is not adjudicable —
#     report it and DO NOT PIN. plots.jl uses the identical concept across runs; here it is within one.
#  4. AGREE ACROSS PROCESSES: a winner that does not reproduce in independent processes is a per-process
#     draw, which is exactly the failure that made the retired OncePerProcess tier ship wrong kernels
#     (one knob picked a 17%-slower arm in 7 of 9 processes). Handled by the caller.
#
# What we CANNOT do without root is stop the clock moving. What we can do is notice, and decline.

using Chairmarks: @be
using Statistics: median

const ANCHOR_N = 1 << 16                       # fixed workload, L2-resident, no allocation per call
# ⚠ CALIBRATED AGAINST THE ANCHOR'S OWN NOISE, not intuition. This was 3%, which is BELOW the
# instrument's repeatability — 10 idle freq-locked reads on 2026-08-20 gave wintermute 5.3%,
# neuromancer 6.0%, galen 22.2%. A guard tighter than its own instrument fires on nothing but noise:
# at 3% it refused a clean +20.6% ger_panel_np win because the anchor moved 3.7%. See
# kb/findings/pureblas-run-anchor-is-not-l2-resident-on-zen3.md.
const ANCHOR_TOL = 0.08                        # above every box's measured anchor repeatability
const STABILISE_MAX_S = 90.0                   # give up warming up after this and say so

function _anchor_secs()
    a = fill(1.0000001, ANCHOR_N)
    b = @be sum(abs2, a) evals = 1 samples = 64 seconds = 0.4
    return median(Float64[s.time for s in b.samples])   # MEDIAN, never min — the project's estimator rule
end

"Achieved kHz under load, max over cores. Readable WITHOUT root on Linux; 0 where unavailable."
function achieved_khz()
    best = 0
    isdir("/sys/devices/system/cpu") || return best
    for d in readdir("/sys/devices/system/cpu"; join = true)
        f = joinpath(d, "cpufreq", "scaling_cur_freq")
        isfile(f) || continue
        v = tryparse(Int, strip(read(f, String)))
        isnothing(v) || (best = max(best, v))
    end
    return best
end

"""
    freq_locked() -> (locked::Bool, why::String)

Is this box in the ONLY state a gate/tuning measurement is valid in — governor pinned, boost off,
`scaling_min_freq == scaling_max_freq` (i.e. `sudo bench/fleet_freqlock.sh lock`)?

⚠ THE TUNER DID NOT CHECK THIS UNTIL 2026-08-21, AND IT SHOULD HAVE BEEN FIRST. On that date
neuromancer ran a full calibration at an achieved 4843 MHz against a 2000 MHz base — its lock had
silently dropped (laptop, suspend/resume) — and produced an incoherent per-size verdict (per-column
winning at n=512, losing at 1024, winning again at 2048 and 4096). The run was rejected only
incidentally, because no window fitted that nonsense; had it fitted, the tuner would have PINNED a
value measured on a floating clock. This repeats a documented incident: a kb finding was once retracted
because "the box's frequency lock had silently dropped (4841 MHz against a 2000 MHz base)".
A floating clock does not merely add noise — it drifts BETWEEN the arms of an A/B, which is the one
error paired measurement cannot cancel.
"""
function freq_locked()
    isdir("/sys/devices/system/cpu/cpu0/cpufreq") || return (true, "no cpufreq (not Linux) — unchecked")
    b = tryparse(Int, strip(read("/sys/devices/system/cpu/cpufreq/boost", String)))
    if !isnothing(b) && b != 0
        return (false, "boost is ON (boost=$b) — run `sudo bench/fleet_freqlock.sh lock`")
    end
    lo = tryparse(Int, strip(read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq", String)))
    hi = tryparse(Int, strip(read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq", String)))
    (isnothing(lo) || isnothing(hi)) && return (true, "cpufreq unreadable — unchecked")
    lo == hi || return (false, "clock NOT pinned (min=$(lo÷1000) MHz, max=$(hi÷1000) MHz) — " *
                               "run `sudo bench/fleet_freqlock.sh lock`")
    return (true, "pinned at $(hi ÷ 1000) MHz, boost off")
end

"""
    contention() -> (loadavg, busy::Bool)

1-minute load average, and whether the machine is too busy to tune. A user's laptop is not a locked fleet
box: a browser, a compile job or a backup running alongside will move the anchor by percent, and a knob
decided against that is a knob decided on someone else's workload.

CALIBRATED, NOT GUESSED: with a test suite running here (load 4.87, one process at 214% CPU) consecutive
anchor readings drifted 3.8% median / 8.9% max, while this fleet's quiet cross-run anchors agree to
0.05%. So contention alone spans the entire decision margin, and refusing up front is much cheaper than
discovering it via `with_anchor` after minutes of measurement.

Threshold 1.5 allows for our own single-threaded load plus normal desktop idle ON A DEDICATED BOX.

⚠ IT IS NOT UNIVERSAL, AND AN ABSOLUTE THRESHOLD IS THE WRONG SHAPE FOR A LAPTOP. Measured on
neuromancer 2026-08-21: ZERO julia processes and a load average of 1.56 / 1.73 / 1.52 — a *persistent*
desktop baseline (browser, indexers, trackers), not a decaying transient. That box therefore refuses
to tune, forever, no matter how long anyone waits. Override with `PUREBLAS_TUNE_MAXLOAD`.

Raising it is defensible because this check is only a cheap PRE-FILTER: `with_anchor` re-times a fixed
workload around every measurement and refuses any knob whose machine state moved more than
`ANCHOR_TOL`, so genuine interference is still caught per-knob rather than per-run. What the raised
threshold buys is the chance to measure at all; what it does not buy is permission to trust a drifting
result, and the anchor still owns that decision.
"""
const MAXLOAD = something(tryparse(Float64, get(ENV, "PUREBLAS_TUNE_MAXLOAD", "")), 1.5)
function contention()
    la = try
        parse(Float64, first(split(read("/proc/loadavg", String))))
    catch
        0.0    # unreadable (non-Linux): do not block, `with_anchor` still guards each knob
    end
    return (la, la > MAXLOAD)
end

"""
    stabilise!(; tol = 0.02, maxsecs = STABILISE_MAX_S) -> (anchor, khz, steady::Bool)

Load the core until the anchor workload stops changing, i.e. until the CPU has left its boost window and
reached thermal steady state. Returns the steady anchor time, the achieved clock, and whether it actually
converged. Measuring before this is the one bias paired rounds cannot cancel: the boost ramp is monotone
across the entire run, so it does not look like noise, it looks like "the first candidate is fastest".
"""
function stabilise!(; tol::Float64 = 0.02, maxsecs::Float64 = STABILISE_MAX_S)
    t0 = time()
    prev = _anchor_secs()
    while time() - t0 < maxsecs
        cur = _anchor_secs()
        # Converged when two consecutive readings agree AND we are not still speeding up. A laptop coming
        # out of idle gets FASTER first (boost) then SLOWER (thermal); either direction disqualifies.
        if abs(cur - prev) / prev <= tol
            return (cur, achieved_khz(), true)
        end
        prev = cur
    end
    return (prev, achieved_khz(), false)
end

"""
    with_anchor(f; tol = ANCHOR_TOL) -> (result, ok::Bool, drift)

Run `f()` between two anchor measurements. `ok` is false when the machine state moved by more than `tol`
across it — in which case the enclosed measurement is NOT adjudicable and the caller must refuse to pin,
however clean the numbers look. This is the guard that makes tuning safe on an uncontrolled machine.
"""
function with_anchor(f; tol::Float64 = ANCHOR_TOL)
    a0 = _anchor_secs()
    r = f()
    a1 = _anchor_secs()
    drift = abs(a1 - a0) / a0
    return (r, drift <= tol, drift)
end

"""
    decide(res; delta = 0.02) -> (name, verdict)

Turn a `Measure.ab` result into a decision. `res[1]` is the incumbent; `ratio > 1` means that arm is
FASTER than the incumbent. A candidate wins only if BOTH:
  * its bootstrap CI excludes 1.0 (the difference is resolvable at all), and
  * its median beats the incumbent by more than `delta` (it is worth acting on).
Anything else is `:tie`, and a tie MUST NOT be pinned — the in-code default is then adequate here, an
unwritten pin is self-documenting, and it leaves the derived default free to be re-adjudicated when the
kernel changes. Ranking by median alone is what the retired duels did, and it is how a knob comes to
select on noise.
"""
function decide(res; delta::Float64 = 0.02)
    best = nothing
    for r in res[2:end]
        (r.lo > 1.0 && r.ratio > 1 + delta) || continue
        (isnothing(best) || r.ratio > best.ratio) && (best = r)
    end
    return isnothing(best) ? (res[1].name, :tie) : (best.name, :win)
end
