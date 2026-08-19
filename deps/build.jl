# Install-time tuning. Pkg runs this once when PureBLAS is added or explicitly rebuilt.
#
# WHY HERE AND NOT IN `__init__`. A few knobs are not derivable from detected hardware — `ger_panel_np`
# is measured opposite-sign across microarchitectures (Zen5→1, Zen3→4, Zen4→8) on boxes that share L2,
# L3, SIMD width and register count — so they have to be MEASURED on the machine that will run them. The
# obvious place to do that is first use, and that is wrong: it would put a multi-minute benchmark inside
# `using PureBLAS`, stalling every script, CI job and container build, and `set_preferences!` triggers a
# recompile, which is a hazard to perform from inside the load of the package being recompiled.
# Install time is the correct moment: it happens once, the user is already waiting on Pkg, and nothing
# else is loaded.
#
# WHAT MAKES THIS SAFE ON THE WRONG MACHINE. Installs do not always happen where the code runs — Docker
# build hosts, shared CI runners, network depots. `tune!()` stamps a `tuned_for` FINGERPRINT
# (vendor/family/simd/L1/L2/L3/nvreg) alongside the pins, and `PureBLAS.is_tuned()` compares it against
# detected hardware at load. A tune performed on different silicon therefore degrades to "not tuned" —
# the defaults resume — instead of silently pinning values that describe absent hardware. That property
# is what makes an install-time measurement acceptable at all.
#
# THIS MUST NEVER FAIL THE INSTALL. Tuning is an optimisation; the in-code defaults are correct
# everywhere, just not optimal. Every failure path below degrades to "not tuned" and returns 0.
#
#   PUREBLAS_NO_AUTOTUNE=1   skip entirely (CI, containers, reproducible builds, air-gapped rebuilds)
#   PUREBLAS_TUNE_REPEATS=n  independent processes that must agree before a value is pinned (default 3)

const SKIP = get(ENV, "PUREBLAS_NO_AUTOTUNE", "") == "1"

if SKIP
    @info "PureBLAS: PUREBLAS_NO_AUTOTUNE=1 — skipping install-time tuning; in-code defaults will be used."
else
    try
        # Refuse on a busy machine BEFORE paying for the load. A knob decided against someone else's
        # workload is a knob decided wrong, and installs frequently happen while other work is running —
        # measured here: with a test suite active, consecutive anchor readings drifted 3.8% median /
        # 8.9% max, versus 0.05% on the same box when quiet. That span exceeds every decision margin.
        la = try
            parse(Float64, first(split(read("/proc/loadavg", String))))
        catch
            0.0
        end
        if la > 1.5
            @warn "PureBLAS: machine is busy (1-min load $(la)); skipping install-time tuning. " *
                  "Run `PureBLAS.tune!()` later on an idle machine to pin this box's values."
        else
            @info "PureBLAS: measuring the few machine-specific tuning knobs (several minutes). " *
                  "Set PUREBLAS_NO_AUTOTUNE=1 to skip this on future installs."
            using PureBLAS
            reps = something(tryparse(Int, get(ENV, "PUREBLAS_TUNE_REPEATS", "")), 3)
            PureBLAS.tune!(; repeats = reps)
            @info "PureBLAS: tuning complete. " * string(PureBLAS.tuning_status())
        end
    catch e
        # Deliberately broad: a build script that throws makes the package look broken, and nothing here
        # is required for correctness. Report and carry on with defaults.
        @warn "PureBLAS: install-time tuning failed; falling back to in-code defaults. This is not an " *
              "error — the library is fully functional, just not tuned for this machine. Run " *
              "`PureBLAS.tune!()` manually to retry." exception = (e, catch_backtrace())
    end
end
