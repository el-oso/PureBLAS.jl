# Per-machine tuning: status, fingerprint, and the `tune!()` entry point.
#
# WHY THIS FILE CONTAINS NO BENCHMARK. The Measure tier (a `Base.OncePerProcess` duel per knob, resolved
# on first call) was removed from src/ entirely on 2026-08-19 — 47 objects, plus every `_measure_*` body.
# It had to go because a resolver that runs in EVERY process cannot afford enough samples: measured, one
# knob picked a 17%-SLOWER kernel arm in 7 of 9 processes, and another shipped a 27% regression in 1 of 6.
# Putting the sweep back here would undo that and three other things besides: `juliac --trim` would have
# benchmark code to compile out again, the all-paths AllocCheck proofs would go red (a one-time init
# allocates), and the pins that exist purely to gate those branches would come back.
#
# So `tune!()` SPAWNS `bench/calibrate.jl` in a fresh subprocess. That keeps src/ measurement-free, and it
# buys the property the in-process duels could never have: the winner is confirmed across INDEPENDENT
# PROCESSES. Per-process draws (page placement, allocator state) are exactly what made the old tier
# unreliable — the same knob genuinely resolved differently run to run — so replication across processes,
# not more reps inside one, is the thing that fixes it.

using Preferences: @load_preference, load_preference, set_preferences!

"""Identity of the machine + library state that a tuning run is valid for."""
function _tuning_fingerprint()
    return string(_CPU_VENDOR, "/", _CPU_FAMILY, "/simd", _SIMD_BYTES,
                  "/l1:", _L1_BYTES, "/l2:", _L2_BYTES, "/l3:", _L3_BYTES, "/nvreg", _NVREG)
end

# A FINGERPRINT, NOT A BOOLEAN. `tuned = true` would go stale silently: move the depot to another
# machine, change a BIOS cache setting, or run in a container with a different CPU exposed, and a boolean
# still says "tuned" while every pinned value now describes hardware that is not there. Comparing the
# stored string against the live detection makes staleness self-announcing. The cost is one string
# compare at load.
const _TUNED_FOR = @load_preference("tuned_for", nothing)

"""
    is_tuned() -> Bool

`true` when this machine has been tuned by [`tune!`](@ref) and the stored tuning still matches the
detected hardware. Cheap: compares a stored fingerprint string against the detected one.
"""
is_tuned() = !isnothing(_TUNED_FOR) && _TUNED_FOR == _tuning_fingerprint()

"""
    tuning_status() -> NamedTuple

`(tuned, stored, current, pins)` — whether this machine is tuned, the fingerprint the pins were written
for, the fingerprint detected now, and the tuned preference keys currently set.
"""
function tuning_status()
    cur = _tuning_fingerprint()
    pins = Dict{String, Any}()
    for k in _TUNABLE_KEYS
        v = load_preference(@__MODULE__, k)
        isnothing(v) || (pins[k] = v)
    end
    return (tuned = is_tuned(), stored = _TUNED_FOR, current = cur, pins = pins)
end

# The keys `bench/calibrate.jl` may write. Kept here so `tuning_status()` can report them without
# loading the tuner, and so a key rename fails loudly in one place.
const _TUNABLE_KEYS = ("ger_panel_np", "potrf_upper_direct_max", "gbtrf_cross", "gbtrf_nb",
                       "pbtrf_cross_kd", "pbtrf_u_native_kd", "pbtrf_nb", "pbtrf_nb_small",
                       "brd_nb", "sytrf_cmult")

"""
    tune!(; dryrun = false, repeats = 3, project = Base.active_project())

Measure this machine's optimal values for the tuning knobs that are NOT derivable from detected
hardware, and write them as Preferences pins. Reload PureBLAS afterwards for them to take effect.

Runs `bench/calibrate.jl` in `repeats` fresh subprocesses and pins only values that AGREE across all of
them; a knob whose winner differs between processes, or whose candidates are within measurement noise,
is left at its in-code default and reported. That is deliberate — an unwritten pin means "the default is
adequate here", keeps `LocalPreferences.toml` minimal, and lets a future code change re-adjudicate.

Takes minutes. Run it on an idle, frequency-locked machine (`sudo bench/fleet_freqlock.sh lock`); a
boosting or contended clock produces a wrong winner, which then gets pinned.

`dryrun = true` measures and prints without writing anything.
"""
function tune!(; dryrun::Bool = false, repeats::Int = 3, project = Base.active_project())
    root = normpath(joinpath(@__DIR__, ".."))
    script = joinpath(root, "bench", "calibrate.jl")
    isfile(script) || error("tune!: $script not found — tuning needs the full repository, not just " *
                            "an installed package. Clone PureBLAS.jl and run tune!() from there.")
    args = String[script]
    dryrun && push!(args, "dryrun")
    push!(args, "repeats=$(repeats)")
    # `--project` is load-bearing: preferences are PER PROJECT. The same key has read 108 under
    # `--project=.` and 4 under `--project=bench` in this repo, which cost two full gate runs before it
    # was understood. The tuner must write where the CALLER's PureBLAS resolves its preferences from.
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(project) $(args)`
    @info "PureBLAS.tune!: running $(repeats) independent calibration process(es); this takes minutes." project
    run(cmd)
    dryrun || @info "PureBLAS.tune!: done. RELOAD Julia for the new pins to take effect."
    return nothing
end
