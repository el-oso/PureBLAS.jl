# PREFERENCE-SYNC lint — the two `LocalPreferences.toml` files must agree about every PureBLAS knob.
#
# WHY TWO FILES EXIST AT ALL. Preferences resolve against the ACTIVE PROJECT, and this repo is driven
# from two: `--project=.` for tests, probes and the tuner, `--project=bench` for every sweep. So
# `./LocalPreferences.toml` and `./bench/LocalPreferences.toml` are separate files that no mechanism
# joins, and a pin written in one is invisible to the other.
#
# WHAT IT COST (2026-09-29). They had diverged for weeks and nothing reported it:
#
#     knob                 root (tests)   bench (the GATE)
#     ger_panel_np         4              8
#     gemvt_perscan        false          1
#     gemvt_percol_amin    —              1048576
#     brd_nb, gemvt_pf, gemvt_u, gemvn_minner, pbtrf_*, …   pinned       ABSENT
#
# Two knobs held OPPOSITE values and thirteen were pinned for tests but not for the gate, so the numbers
# published and the numbers the suite proved came from different configurations of the library. Both
# files carried the SAME `tuned_for` fingerprint, which is why neither the tuner nor `plots.jl` noticed:
# the fingerprint covers cache sizes and ISA, never the contents of the other file.
#
# It also wasted a tuning run. `PureBLAS.tune!()` invoked under `--project=.` reported
# "SKIPPED: already pinned … clear the pin to re-tune" for knob after knob — reading, and then writing,
# a file no sweep consults.
#
# THE RULE: a knob pinned in one file must be pinned to the SAME VALUE in the other. Absent from both is
# fine and is the preferred state — every pin here is a Measure-tier knob that has not been converted to
# a derivation yet, so an empty table means the derived defaults were adequate (req#8b).
#
# `tuned_for` is exempt: it is a stamp, not a knob, and each file legitimately records the machine state
# its own pins were measured under.
#
# A knob that genuinely belongs to one environment only carries `# prefsync-ok: <reason>` on its line or
# the line above, the same escape the other lints in this directory use.
#
# Run standalone:  julia test/prefsync_lint.jl

const _PS_FILES = (joinpath(@__DIR__, "..", "LocalPreferences.toml"),
    joinpath(@__DIR__, "..", "bench", "LocalPreferences.toml"))
const _PS_OK = r"#\s*prefsync-ok:"i
const _PS_EXEMPT = ("tuned_for",)

"""
    prefs_of(path) -> Dict{String,String}

The `[PureBLAS]` table of one LocalPreferences.toml as name => value text, skipping keys marked
`# prefsync-ok`. Hand-parsed rather than via TOML so this lint keeps the zero-dependency,
run-standalone shape of its siblings, and so the `prefsync-ok` COMMENTS survive parsing at all — a
TOML reader drops them.
"""
function prefs_of(path::AbstractString)
    out = Dict{String, String}()
    isfile(path) || return out
    lines = readlines(path)
    inblk = false
    for (i, ln) in enumerate(lines)
        s = strip(split(ln, '#')[1])
        if startswith(s, "[")
            inblk = s == "[PureBLAS]"
            continue
        end
        inblk || continue
        m = match(r"^([A-Za-z0-9_]+)\s*=\s*(.+?)\s*$", s)
        isnothing(m) && continue
        occursin(_PS_OK, ln) && continue
        (i > 1 && occursin(_PS_OK, lines[i - 1])) && continue
        m.captures[1] in _PS_EXEMPT && continue
        out[m.captures[1]] = m.captures[2]
    end
    return out
end

"""
    prefsync_scan() -> Vector{String}

One line per knob the two files disagree about — a different value, or pinned in one and not the other.
"""
function prefsync_scan()
    a, b = prefs_of(_PS_FILES[1]), prefs_of(_PS_FILES[2])
    hits = String[]
    for k in sort(collect(union(keys(a), keys(b))))
        va, vb = get(a, k, nothing), get(b, k, nothing)
        va == vb && continue
        if isnothing(va)
            push!(hits, "$k  pinned in bench/LocalPreferences.toml ($vb), absent from LocalPreferences.toml")
        elseif isnothing(vb)
            push!(hits, "$k  pinned in LocalPreferences.toml ($va), absent from bench/LocalPreferences.toml")
        else
            push!(hits, "$k  LocalPreferences.toml=$va  bench/LocalPreferences.toml=$vb")
        end
    end
    return hits
end

# KNOBS WHOSE CONST IGNORES THEM OFF APPLE SILICON. `_TRMM_BASE_R` reads its preference only where
# `_SME_F64` holds, because the recursion it selects is faster only when the off-diagonal gemms run on
# a matrix unit — off SME it was MEASURED regressing, 35.5% slower at k=100. So a value set here on
# any other host has no effect, and dropping it with nothing reported is a silent-continue.
#
# The host test is deliberately coarse. `_SME_F64` needs `+sme` in the CPU target and this lint does
# not load PureBLAS (it reads TOML and nothing else, which is why it costs nothing). `aarch64 + Apple`
# is a SUPERSET of SME-capable, so this never false-alarms on a box where the knob is live; the
# residual gap is an Apple box without `+sme`, where the pin is ignored and this stays quiet.
const _PS_SME_ONLY = ("trmm_base_r",)
_ps_host_has_sme() = Sys.ARCH === :aarch64 && Sys.isapple()

"""
    prefsync_inert(files = _PS_FILES) -> Vector{String}

Knobs that are set in `files` but cannot take effect on this host.

`files` is a parameter so the POSITIVE CONTROL for this check can point at a temporary TOML. It must
not require writing a real `LocalPreferences.toml`: the pin tier belongs to the user, and a check
whose only control is "set a real pin and see if it fires" either goes unverified or gets verified by
touching a file nothing else may touch. (It went unverified the first way and then the second.)
"""
function prefsync_inert(files = _PS_FILES)
    _ps_host_has_sme() && return String[]
    out = String[]
    for path in files
        isfile(path) || continue
        p = prefs_of(path)
        for k in _PS_SME_ONLY
            haskey(p, k) && push!(out, "$(k) = $(p[k])  in $(relpath(path))")
        end
    end
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    inert = prefsync_inert()
    if !isempty(inert)
        println("preference-sync lint: FAIL — $(length(inert)) pin(s) set but INERT on this host:")
        foreach(x -> println("  ", x), inert)
        println()
        println("  These knobs are read only where SME is available; this host is not, so the value is")
        println("  discarded and the derived default stands. That is a silent difference between what")
        println("  the file says and what the library does. Remove the pin, or move it to the Apple box.")
        exit(1)
    end
    v = prefsync_scan()
    if isempty(v)
        println("preference-sync lint: PASS (both LocalPreferences.toml agree on every PureBLAS knob)")
    else
        println("preference-sync lint: FAIL — $(length(v)) knob(s) differ between the two files:")
        foreach(x -> println("  ", x), v)
        println()
        println("  Sweeps read bench/LocalPreferences.toml; tests, probes and the tuner read the root one.")
        println("  A knob that differs means the published numbers and the proved numbers come from")
        println("  different builds. Re-tune once and write the SAME result to both, or clear both and")
        println("  let the derived defaults stand. Setting a pin's VALUE is the user's call, not an agent's.")
        exit(1)
    end
end
