# PIN-COVERAGE lint — every Measure-tier knob MUST be pinned in juliac/build.jl.
#
# WHAT THIS ACTUALLY PROTECTS: **determinism of the shipped .so**, not merely trim hygiene. Pinning is
# the user's knob for building libpureblas.so against a SPECIFIED microarchitecture; the @noalloc
# contract exists so that resulting artifact is deterministic. The two constraints complement each
# other — pinning is what makes the no-allocation proof achievable, and the proof verifies the pin did
# its job. A Measure-tier knob left unpinned puts an on-host `Base.OncePerProcess` benchmark inside the
# library: the first call through an @ccallable symbol behaves differently from every later one, and
# what it decides depends on machine state at that instant. The allocation AllocCheck reports is the
# symptom; the nondeterminism is the defect.
#
# WHY A GATE AND NOT A HABIT (CLAUDE.md req#8b): a missing pin announces itself to NOBODY. It is caught
# only by ACCIDENT, when the measure body happens to contain something trim rejects. On 2026-08-05 the
# axpy knobs were caught exactly that way (a non-concrete `Val(u)` → 10 failing C-ABI symbols), while
# `sytrf_nb` sat unpinned and silently reachable: a C/Rust caller's FIRST `sytrf_64_` would have run a
# benchmark allocating a 1024×1024 matrix and picked its blocking factor from momentary machine state.
# Trim never objected — that candidate loop is over plain ints, hence type-stable and trim-clean. Its
# own source comment claimed "pinned (trim lands here)", which was false.
#
# HEURISTIC (deliberately boring, per Fable's review): a `const X = @load_preference("name", nothing)`
# is Measure-tier if the SAME FILE also mentions `Base.OncePerProcess`. Per-file co-occurrence is
# sufficient for every knob in the tree today and avoids regex-parsing block boundaries. A file that
# genuinely needs an exception gets `# pin-ok: <reason>` on the pref line or the line above.
#
# Run standalone:  julia test/pin_lint.jl

const _SRC = joinpath(@__DIR__, "..", "src")
const _BUILD = joinpath(@__DIR__, "..", "juliac", "build.jl")
const _PREF_NOTHING = r"@load_preference\(\s*\"([A-Za-z0-9_]+)\"\s*,\s*nothing\s*\)"
const _PIN_OK = r"#\s*pin-ok:"i

_jlfiles(dir) = sort!(reduce(vcat, [joinpath(r, f) for (r, _, fs) in walkdir(dir) for f in fs if endswith(f, ".jl")]; init = String[]))

"""
    pin_scan() -> Vector{String}

Names of Measure-tier preferences in src/ that juliac/build.jl does not pin.
"""
function pin_scan()
    pinned = Set{String}()
    if isfile(_BUILD)
        for ln in readlines(_BUILD)
            for m in eachmatch(r"set_preferences!\(\s*PUREBLAS_UUID\s*,\s*\"([A-Za-z0-9_]+)\"\s*=>", split(ln, '#')[1])
                push!(pinned, m.captures[1])
            end
        end
    end
    missing_pins = String[]
    for f in _jlfiles(_SRC)
        lines = readlines(f)
        # Measure tier == this file also runs an on-host benchmark. A pref with a nothing default but no
        # OncePerProcess anywhere in the file is a Pin/Derive-tier override (e.g. `simd_bytes`), which
        # needs no trim pin because nothing is benchmarked.
        # COMMENTS DO NOT COUNT — same rule as the per-line scan below, which has always stripped them.
        # Testing the raw text made a file "Measure tier" the moment someone MENTIONED the type in prose:
        # writing "replaced a OncePerProcess duel" in cpuinfo.jl (which runs no benchmark at all) made
        # this lint demand a pin for `simd_bytes`. Retiring the Measure tier means writing that sentence
        # a lot, so the false positive would have recurred with every conversion.
        codeonly = join((split(ln, '#')[1] for ln in lines), "\n")
        occursin("OncePerProcess", codeonly) || continue
        for (i, ln) in enumerate(lines)
            code = split(ln, '#')[1]
            occursin(_PIN_OK, ln) && continue
            (i > 1 && occursin(_PIN_OK, lines[i - 1])) && continue
            for m in eachmatch(_PREF_NOTHING, code)
                name = m.captures[1]
                name in pinned && continue
                push!(missing_pins, "$(relpath(f, joinpath(@__DIR__, ".."))):$i  \"$name\" is Measure-tier but not pinned in juliac/build.jl")
            end
        end
    end
    return missing_pins
end

# ── PIN-SET DRIFT ───────────────────────────────────────────────────────────────────────────────────
# `pin_scan` above answers "is every Measure knob pinned for the .so?". It says nothing about the OTHER
# pin set: `test/Project.toml`'s `[preferences.PureBLAS]`, which exists so the AllocCheck dogfood items
# can prove all paths — an unpinned Measure knob keeps a `OncePerProcess` reachable, and its one-time
# init allocates, so one live tuner reddens a whole item.
#
# Two ways that set goes wrong, and both had happened before this check existed:
#
# DRIFT — a knob pinned in one file and not the other. The test env then exercises a configuration the
# shipped library does not have, which is the opposite of what pinning the dogfood is for. This is the
# rule that catches `zaxpy_narrow`: juliac/build.jl dropped its pin when the knob became a derivation
# over `_datapath_bytes`, `test/Project.toml` kept it, and the pin forced the narrow arm on precisely
# the native-512 datapath the formula exists to get right. Its `@load_preference` default is still
# `nothing`, so only the disagreement between the two files exposed it.
#
# OVERRIDE — a knob pinned although its `@load_preference` default is not `nothing`. A non-`nothing`
# default is a Derive or Exempt tier: there is no tuner to compile out, so the pin buys no proof and
# only overrides the formula.
#
# Baselined like the other lints here: the current set is recorded with its reasoning, a NEW finding
# fails, and a baseline line whose finding is gone is stale and also fails, so the list shrinks.
const _TESTPROJ = joinpath(@__DIR__, "Project.toml")
const _PIN_BASELINE = joinpath(@__DIR__, "pin_lint_baseline.txt")

"""
    pin_defaults() -> Dict{String, Bool}

Every `@load_preference` key in `src/`, mapped to whether its default is `nothing` (Measure tier, so a
pin removes a runtime tuner) rather than a value (Derive/Exempt, so a pin only overrides a formula).
"""
function pin_defaults()
    d = Dict{String, Bool}()
    for f in _jlfiles(_SRC), ln in readlines(f)
        for m in eachmatch(r"@load_preference\(\s*\"([A-Za-z0-9_]+)\"\s*,\s*([^\n]*)", split(ln, '#')[1])
            d[m.captures[1]] = startswith(strip(m.captures[2]), "nothing")
        end
    end
    return d
end

_build_pins() = Set{String}(
    m.captures[1] for ln in (isfile(_BUILD) ? readlines(_BUILD) : String[])
    for m in eachmatch(r"set_preferences!\(\s*PUREBLAS_UUID\s*,\s*\"([A-Za-z0-9_]+)\"\s*=>", split(ln, '#')[1])
)

# Hand-parsed rather than via TOML so this file keeps its zero-dependency, run-standalone property:
# read the `[preferences.PureBLAS]` table's bare `key = value` lines until the next section header.
function _test_pins()
    out = Set{String}()
    isfile(_TESTPROJ) || return out
    inblk = false
    for ln in readlines(_TESTPROJ)
        s = strip(split(ln, '#')[1])
        if startswith(s, "[")
            inblk = s == "[preferences.PureBLAS]"
            continue
        end
        inblk || continue
        m = match(r"^([A-Za-z0-9_]+)\s*=", s)
        isnothing(m) || push!(out, m.captures[1])
    end
    return out
end

"""
    pin_drift_scan() -> Vector{String}

One stable line per finding: a knob pinned in only one of the two pin sets, or pinned anywhere though
its src default is a value rather than `nothing`.
"""
function pin_drift_scan()
    b, t, defs = _build_pins(), _test_pins(), pin_defaults()
    hits = String[]
    for n in sort!(collect(setdiff(t, b)))
        push!(hits, "DRIFT    $n  pinned in test/Project.toml, not in juliac/build.jl")
    end
    for n in sort!(collect(setdiff(b, t)))
        push!(hits, "DRIFT    $n  pinned in juliac/build.jl, not in test/Project.toml")
    end
    for (n, where) in vcat([(n, "test/Project.toml") for n in sort!(collect(t))],
        [(n, "juliac/build.jl") for n in sort!(collect(b))])
        get(defs, n, true) && continue          # `nothing` default, or a key src does not read
        push!(hits, "OVERRIDE $n  pinned in $where though its src default is a formula, not `nothing`")
    end
    return hits
end

_pin_baseline() = isfile(_PIN_BASELINE) ?
    Set(filter(l -> !isempty(l) && !startswith(l, "#"), strip.(readlines(_PIN_BASELINE)))) : Set{String}()

"""
    pin_drift() -> (new = …, stale = …)

`new`: a drift or override nobody has reasoned about — align the two files, drop the pin, or baseline
it with the argument. `stale`: a baselined finding that no longer holds — delete the entry.
"""
function pin_drift()
    got = Set(pin_drift_scan())
    base = _pin_baseline()
    return (new = sort!(collect(setdiff(got, base))), stale = sort!(collect(setdiff(base, got))))
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--baseline" in ARGS
        foreach(println, pin_drift_scan())
    else
        v = pin_scan()
        r = pin_drift()
        ok = isempty(v) && isempty(r.new) && isempty(r.stale)
        if ok
            println("pin lint: PASS (every Measure-tier knob is pinned for the trim build; pin sets agree)")
        else
            println("pin lint: FAIL")
            foreach(x -> println("  ", x), v)
            foreach(x -> println("  NEW   ", x), r.new)
            foreach(x -> println("  STALE ", x), r.stale)
            exit(1)
        end
    end
end
