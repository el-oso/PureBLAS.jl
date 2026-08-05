# PIN-COVERAGE lint — every Measure-tier knob MUST be pinned in juliac/build.jl.
#
# WHY THIS IS A GATE AND NOT A HABIT (CLAUDE.md req#8b): a Measure-tier knob defaults to an on-host
# `Base.OncePerProcess` benchmark. That benchmark is not trim-safe, so the trim/.so build pins the
# preference, which makes `@static if isnothing(pref)` delete the measure — and the OncePerProcess with
# it — at expansion time. If the pin is MISSING, nothing announces it: the knob is caught only by
# ACCIDENT, when its measure body happens to contain something trim rejects. On 2026-08-05 the axpy
# knobs were caught exactly that way (a non-concrete `Val(u)` → 10 failing C-ABI symbols), while
# `sytrf_nb` — whose measure body is plausibly type-stable — sat unpinned and would have shipped a
# first-call benchmark (allocating a 1024×1024 matrix) into libpureblas.so via the @ccallable
# `sytrf_64_`. Its own source comment claimed "pinned (trim lands here)", which was false.
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
        src = read(f, String)
        # Measure tier == this file also runs an on-host benchmark. A pref with a nothing default but no
        # OncePerProcess anywhere in the file is a Pin/Derive-tier override (e.g. `simd_bytes`), which
        # needs no trim pin because nothing is benchmarked.
        occursin("OncePerProcess", src) || continue
        lines = readlines(f)
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

if abspath(PROGRAM_FILE) == @__FILE__
    v = pin_scan()
    if isempty(v)
        println("pin lint: PASS (every Measure-tier knob is pinned for the trim build)")
    else
        println("pin lint: FAIL — $(length(v)) Measure-tier knob(s) missing a juliac/build.jl pin:")
        foreach(x -> println("  ", x), v)
        exit(1)
    end
end
