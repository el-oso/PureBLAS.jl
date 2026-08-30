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
#
# RUNTIME read into a Ref, not `@load_preference` into a const. The macro reads at COMPILE time and
# registers the key via `Base.record_compiletime_preference`, so writing it would invalidate the
# pkgimage — and `tune!()` writes exactly this key. That is what made an install cost precompile + tune
# + RECOMPILE. Read at runtime, `load_preference` registers nothing, so tuning costs a restart.


"""
    is_tuned() -> Bool

`true` when this machine has been tuned by [`tune!`](@ref) and the stored tuning still matches the
detected hardware. Cheap: compares a stored fingerprint string against the detected one.
"""
is_tuned() = (f = load_preference(@__MODULE__, "tuned_for"); !isnothing(f) && f == _tuning_fingerprint())

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
    return (tuned = is_tuned(), stored = load_preference(@__MODULE__, "tuned_for"), current = cur, pins = pins)
end

# The keys `bench/calibrate.jl` may write. Kept here so `tuning_status()` can report them without
# loading the tuner, and so a key rename fails loudly in one place.
#
# ⚠ THIS LIST AND `bench/calibrate.jl`'s `KNOBS` HAD DRIFTED APART IN BOTH DIRECTIONS (found 2026-08-29).
# calibrate WRITES `gemvt_percol_window`, `gemvt_pf`, `trmv_fused_min` and `gbtrf_cmult`, none of which
# were listed here — so `tuning_status()` under-reported a tuned machine, and (once `bench/plots.jl`
# began accepting a tuned box as a valid gate subject) a legitimately tuned box would have been REFUSED
# because four of its own pins looked unowned. The four are added below.
# The reverse gap is real too and is NOT a bug to fix by deletion: `gbtrf_cross`, `gbtrf_nb`,
# `pbtrf_cross_kd`, `pbtrf_u_native_kd`, `pbtrf_nb`, `pbtrf_nb_small` and `brd_nb` are declared tunable
# but have NO calibrator, so `tune!()` can never set them. They are kept here (the key is still a valid
# user pin) but must not be counted as "covered by tune!()" — a plan that assumed they were mis-scoped
# pbtrfU, gbtrf and gesvd as user actions when they are code work.
# `test/tuner_tests.jl` now asserts `KNOBS ⊆ _TUNABLE_KEYS` so this cannot drift silently again.
const _TUNABLE_KEYS = ("ger_panel_np", "potrf_upper_direct_max", "gbtrf_cross", "gbtrf_nb",
                       "pbtrf_cross_kd", "pbtrf_u_native_kd", "pbtrf_nb", "pbtrf_nb_small",
                       "brd_nb", "sytrf_cmult",
                       # written by bench/calibrate.jl's KNOBS but previously unlisted here:
                       "gemvt_percol_window", "gemvt_pf", "trmv_fused_min", "gbtrf_cmult")

"""
Block until the 1-minute load average falls below calibrate.jl's contention threshold, or give up.
A calibration run leaves its own load behind, so consecutive runs must be spaced or the second one
refuses to measure. Bounded so a genuinely busy machine still makes progress rather than hanging.

⚠ THE BOUNDS ARE SET BY HOW loadavg DECAYS, NOT BY TASTE. The 1-minute average is a LAGGING
indicator with roughly a 1-minute time constant, so after a multi-minute run that pinned a core it
needs several minutes to fall back under calibrate.jl's 1.5 threshold. Measured on Zen3 2026-08-21:
zero julia processes running and the 1-min average still read 2.69 (5-min 1.71, 15-min 0.70). A 180 s
cap therefore expired while the box was still "busy" by the very guard this exists to satisfy, and the
next run refused. `below` sits just under calibrate.jl's threshold so satisfying this satisfies that.
"""
function _wait_idle(; below::Float64 = 1.4, maxsecs::Int = 600)
    t0 = time()
    while time() - t0 < maxsecs
        la = try
            parse(Float64, first(split(read("/proc/loadavg", String))))
        catch
            return nothing        # unreadable (non-Linux): calibrate.jl's own guard still applies
        end
        la < below && return nothing
        sleep(5)
    end
    @warn "tune!: machine still busy after $(maxsecs)s; the next run may refuse to measure."
    return nothing
end

"""
    tune!(; dryrun = false, repeats = 3, project = Base.active_project(), unlocked = false)

Measure this machine's optimal values for the tuning knobs that are NOT derivable from detected
hardware, and write them as Preferences pins. Reload PureBLAS afterwards for them to take effect.

Runs `bench/calibrate.jl` in `repeats` fresh subprocesses and pins only values that AGREE across all of
them; a knob whose winner differs between processes, or whose candidates are within measurement noise,
is left at its in-code default and reported. That is deliberate — an unwritten pin means "the default is
adequate here", keeps `LocalPreferences.toml` minimal, and lets a future code change re-adjudicate.

Takes a few minutes (measured ~2 min locked, ~3.5 min unlocked, for the current 5 knobs).

## Frequency

Best results come from a frequency-locked machine (`sudo bench/fleet_freqlock.sh lock`), because a
pinned clock removes a whole class of measurement error at the source.

**If you do not have root — the normal case — use `unlocked = true`.** It does not skip the safety
checks; it changes the unit of measurement. The A/B then reduces in CYCLES rather than wall time, and
cycles are invariant to boost: a faster clock shortens both arms' elapsed time and leaves the cycle
counts where they were, so the RATIO the decision rests on is unaffected. To compensate for the missing
lock it also tightens the machine-state drift tolerance (8% → 3%), raises the margin a candidate must
beat the incumbent by (2% → 5%), and takes more rounds (8 → 12). Pins written this way are stamped
`measured_unlocked = true` so their provenance is never ambiguous.

Either way, run it on an **idle** machine — contention is a separate problem from clock speed, and
calibration refuses outright on a busy box rather than pinning a knob decided against someone else's
workload. Close other applications first.

`dryrun = true` measures and prints without writing anything — worth doing first.
"""
function tune!(; dryrun::Bool = false, repeats::Int = 3, project = Base.active_project(),
               unlocked::Bool = false)
    root = normpath(joinpath(@__DIR__, ".."))
    script = joinpath(root, "bench", "calibrate.jl")
    isfile(script) || error("tune!: $script not found — tuning needs the full repository, not just an " *
                            "installed package. Clone PureBLAS.jl and run tune!() from there.")
    jl = Base.julia_cmd()
    # `unlocked` must be reachable from HERE, not only from the script. `tune!()` is the documented
    # entry point — the one `_GER_NP`'s comment tells users to run — so a flag that exists only on
    # calibrate.jl leaves the refusal in place for everyone who follows the documentation. The kwarg
    # is forwarded to every subprocess; see the UNLOCKED MODE note in bench/calibrate.jl for what it
    # trades (decides in CYCLES, which boost cannot bias, and tightens drift/margin/rounds to
    # compensate for the missing lock).
    extra = unlocked ? ["unlocked"] : String[]
    if dryrun
        run(`$jl --startup-file=no --project=$(project) $script dryrun $extra`)
        return nothing
    end

    # INDEPENDENT PROCESSES, then agreement. A per-process draw (page placement, allocator state) can
    # change which arm wins — that is not a hypothesis, it is what made the retired OncePerProcess tier
    # ship wrong kernels (one knob picked a 17%-SLOWER arm in 7 of 9 processes). Repeating inside one
    # process cannot detect it; only separate processes can. A knob that does not agree across all of
    # them is left at its in-code default and reported.
    @info "PureBLAS.tune!: $(repeats) independent calibration runs (minutes each)." project
    results = Dict{String, Any}[]
    mktempdir() do dir
        for k in 1:repeats
            f = joinpath(dir, "run$k.toml")
            # LET THE BOX SETTLE FIRST. calibrate.jl refuses to measure on a busy machine (1-min
            # loadavg > 1.5), and a run leaves its OWN load in that average — so back-to-back runs
            # refuse themselves. Observed 2026-08-21: runs 2..N produced nothing on both fleet boxes
            # while run 1 succeeded, and because a missing file was silently skipped, "unanimous
            # agreement" was then satisfied by a SINGLE process. That is the exact failure the
            # multi-process design exists to prevent, reintroduced by the guard it depends on.
            k > 1 && _wait_idle()
            try
                run(`$jl --startup-file=no --project=$(project) $script emit=$f $extra`)
            catch e
                @warn "tune!: calibration run $k failed; skipping it." e
                continue
            end
            # Plain `key=value` lines, parsed here — deliberately NOT TOML, because TOML is not a
# PureBLAS dependency and adding one to read three integers would be the wrong trade.
isfile(f) || continue
d = Dict{String, Any}()
for ln in eachline(f)
    kv = split(strip(ln), '='; limit = 2)
    length(kv) == 2 && (d[strip(kv[1])] = something(tryparse(Int, strip(kv[2])), strip(kv[2])))
end
push!(results, d)
        end
    end
    # EVERY run must have produced a result. "Agreement across N independent processes" is the whole
    # safeguard — a per-process draw is exactly what the retired OncePerProcess tier got wrong — and
    # agreement among the SURVIVORS is not that. With runs silently dropped, one process could satisfy
    # "unanimous" on its own, which is a single sample wearing the costume of a quorum.
    if length(results) < repeats
        @warn "tune!: only $(length(results)) of $(repeats) calibration runs produced a result — a " *
              "dropped run makes 'unanimous' meaningless, so NOTHING is pinned. Re-run on an idle, " *
              "frequency-locked machine." got = length(results) want = repeats
        return nothing
    end
    isempty(results) && (@warn "tune!: no calibration run produced a result; defaults retained"; return nothing)

    agreed = Dict{String, Any}()
    keys_ = union(keys.(results)...)
    for k in keys_
        vals = [get(r, k, nothing) for r in results]
        if all(==(first(vals)), vals) && !isnothing(first(vals)) && length(vals) == length(results)
            agreed[k] = first(vals)
        else
            @info "tune!: `$k` did not agree across runs $(vals) — leaving the default in place."
        end
    end

    if isempty(agreed)
        @info "tune!: nothing to pin (every knob tied or disagreed). The in-code defaults are adequate here."
        return nothing
    end
    # Writing the pins is the ONE action that must land in the caller's project, because preferences are
    # per project — the same key has read 108 under `--project=.` and 4 under `--project=bench` in this
    # repo, which cost two full gate runs before it was understood.
    # Written through Preferences, not by hand-editing the TOML: it resolves the right file for the
    # ACTIVE project, and preferences are PER PROJECT — the same key has read 108 under `--project=.`
    # and 4 under `--project=bench` in this repo, which cost two full gate runs before it was understood.
    for (k, v) in agreed
        set_preferences!(@__MODULE__, k => v; force = true)
    end
    set_preferences!(@__MODULE__, "tuned_for" => _tuning_fingerprint(); force = true)
    @info "PureBLAS.tune!: pinned $(length(agreed)) knob(s)" agreed
    @info "Restart Julia to pick them up. They are read at RUNTIME, so this costs no recompile."
    return agreed
end
