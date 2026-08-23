# THE FREQUENCY-LOCK CRITERION — SINGLE SOURCE OF TRUTH, mirroring bench/gatecrit.jl.
#
# CLAUDE.md: `sudo bench/fleet_freqlock.sh lock` (governor pinned, boost OFF, every core at base
# clock) is the ONLY state a gate or tuning measurement may be taken in; a run taken off-lock is
# INVALID and must be discarded, not rationalised. That rule was spelled out in TWO places with two
# different strictnesses, which is how it ended up half-enforced:
#
#   * bench/tune_harness.jl `freq_locked()` — added 2026-08-21 after neuromancer calibrated at an
#     achieved 4843 MHz against a 2000 MHz base. Guards the CALIBRATOR only, and reads cpu0 ALONE.
#   * bench/plots.jl — guarded NOTHING. The gate benchmark, the thing that produces every published
#     number, would happily measure a boosting box. On 2026-08-23 it did: a Zen5 LP sweep ran at
#     4809435 kHz against a 2000000 base and 67 cells had to be thrown away.
#
# CPU0 IS NOT ENOUGH, and that is not hypothetical either. A `taskset`-pinned benchmark shares the
# package power and thermal budget with every other core, so ONE core left free to ramp is exactly the
# drift this check exists to exclude. `lock_state` therefore aggregates as `lo = MIN over every core's
# floor` and `hi = MAX over every core's ceiling`, so `lo == hi` means one frequency EVERYWHERE.
# Getting that direction backwards (max-over-floors) reports a locked box whenever any single core
# happens to be pinned high — the first draft of this function did exactly that and passed a fixture
# with core 8 ranging 0.4-2.0 GHz.
module FreqLock

"""
    lock_state() -> (lo_khz, hi_khz, boost)

`lo` = lowest floor ANY core may fall to, `hi` = highest ceiling ANY core may reach, `boost` = the
global boost flag (-1 when absent). `lo == hi` ⟺ every core is pinned to one frequency.
Returns `(0, 0, -1)` where cpufreq does not exist (e.g. ARM), so a platform without the knob degrades
to "unchecked" rather than marking every cell invalid.
"""
function lock_state(root::AbstractString = "/sys/devices/system/cpu")
    lo = typemax(Int); hi = 0
    isdir(root) || return (0, 0, -1)
    for d in readdir(root; join = true)
        fmin = joinpath(d, "cpufreq", "scaling_min_freq")
        fmax = joinpath(d, "cpufreq", "scaling_max_freq")
        (isfile(fmin) && isfile(fmax)) || continue
        vmin = tryparse(Int, strip(read(fmin, String)))
        vmax = tryparse(Int, strip(read(fmax, String)))
        isnothing(vmin) || (lo = min(lo, vmin))
        isnothing(vmax) || (hi = max(hi, vmax))
    end
    lo == typemax(Int) && (lo = 0)
    bf = joinpath(root, "cpufreq", "boost")
    boost = isfile(bf) ? something(tryparse(Int, strip(read(bf, String))), -1) : -1
    return (lo, hi, boost)
end

"""
    freq_locked() -> (locked::Bool, why::String)

Is this box in the only state a gate/tuning measurement is valid in?
"""
function freq_locked(root::AbstractString = "/sys/devices/system/cpu")
    (lo, hi, boost) = lock_state(root)
    hi == 0 && return (true, "no cpufreq — unchecked")
    boost == 1 && return (false, "boost is ON (boost=1) — run `sudo bench/fleet_freqlock.sh lock`")
    lo == hi || return (false, "not every core is pinned to one frequency (floor $(lo ÷ 1000) MHz, " *
                               "ceiling $(hi ÷ 1000) MHz) — run `sudo bench/fleet_freqlock.sh lock`")
    return (true, "pinned at $(hi ÷ 1000) MHz, boost off")
end

"""
    require_lock(; what = "measure")

Refuse to proceed off-lock. The contention guard's own rationale — "a three-hour sweep deserves a
one-second check first" — applies harder here: a contended run is merely noisy, an off-lock run is
INVALID. `PUREBLAS_BENCH_NOLOCK=1` overrides for a deliberately un-gated exploration; the caller
still records the true state in its provenance.
"""
function require_lock(; what::AbstractString = "measure")
    get(ENV, "PUREBLAS_BENCH_NOLOCK", "") == "1" && return nothing
    (ok, why) = freq_locked()
    ok && return nothing
    error("""
        REFUSING to $what: the frequency lock is not in the only state a gate run may use.
          $why
        A run taken off-lock is INVALID and must be discarded, so this refuses up front rather than
        letting hours of measurement turn into a cache nobody can trust.
        Override for a non-gate exploration only: PUREBLAS_BENCH_NOLOCK=1.""")
end

end # module
