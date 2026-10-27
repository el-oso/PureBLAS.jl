# Fleet machine names must not enter the repository. This is a PUBLIC repo and the names are one
# operator's; they have been removed twice now, by hand, 162 occurrences the second time.
#
# THE LINT CANNOT CONTAIN THE NAMES IT LOOKS FOR — that would be the leak it exists to prevent. So it
# reads them from the box: `hostname`, plus the ssh aliases in `PB_BOXES` from the gitignored
# `bench/fleet.local.sh`. A machine that knows its own name can check for it; the repository never
# learns any of them.
#
# TWO CHECKS, BECAUSE THEY FAIL IN DIFFERENT PLACES.
#
#   1. NAMES. Needs the local config, so it is effective where commits are AUTHORED and inert in CI,
#      which has no fleet and cannot know the names. That is the honest limit: CI cannot check this,
#      and a check reporting success from zero knowledge would be worse than none. It says which of
#      the two it did.
#
#   2. SHAPE. A `case "$(hostname)"` table or a `gethostname()` lookup key is the construct that put
#      the names there both times: a per-box value keyed on a machine's name. This half needs no names
#      and therefore DOES work in CI. Catching the shape catches the next reintroduction even on a box
#      whose own name is not in the list.
#
# The replacement for the shape is in `bench/fleet.local.example.sh`: a box states its own topology in
# a gitignored file, so no table and no names are needed. `gethostname()` itself is fine where the
# value is DATA rather than a key — a cache filename is per-host by design, and those caches are
# gitignored.
const _FN_ROOT = normpath(joinpath(@__DIR__, ".."))
const _FN_LOCAL = joinpath(_FN_ROOT, "bench", "fleet.local.sh")

"""Names to look for: this machine, plus the ssh aliases it syncs to. Empty when neither is available."""
function fleet_names()
    names = String[]
    h = strip(try
        gethostname()
    catch
        ""
    end)
    # A bare hostname only. A fully-qualified name's leading label is the part that would appear in a
    # comment, and matching the whole string would miss it.
    isempty(h) || push!(names, String(first(split(h, '.'))))
    if isfile(_FN_LOCAL)
        for ln in eachline(_FN_LOCAL)
            m = match(r"^\s*PB_BOXES\s*=\s*\"?([^\"#]*)\"?", ln)
            isnothing(m) && continue
            for b in split(m.captures[1])
                isempty(b) || push!(names, String(b))
            end
        end
    end
    # Two-character names would match inside ordinary words; the shortest real alias here is far longer
    # and a name that short could not be checked without false positives anyway.
    return unique(filter(n -> length(n) >= 4, names))
end

"""Tracked files, from git — so gitignored caches and local config are out of scope by construction."""
function _fn_tracked()
    out = try
        readchomp(Cmd(`git ls-files`; dir = _FN_ROOT))
    catch
        return String[]
    end
    return filter(!isempty, split(out, '\n'))
end

# Binary and generated artifacts are skipped: an SVG is machine output, and a name inside one means the
# generator leaked it, which the generator's own source will show.
_fn_skip(p) = endswith(p, ".svg") || endswith(p, ".png") || endswith(p, ".ico") ||
    startswith(p, "docs/src/assets/") || p == "test/fleetname_lint.jl"

"""Occurrences of any fleet name in a tracked file, as `path:line  text`."""
function fleetname_hits(names = fleet_names())
    isempty(names) && return String[]
    rx = Regex("\\b(" * join(map(n -> replace(n, r"([.*+?()\[\]\\^\$|])" => s"\\\1"), names), "|") * ")\\b", "i")
    hits = String[]
    for rel in _fn_tracked()
        _fn_skip(rel) && continue
        path = joinpath(_FN_ROOT, rel)
        isfile(path) || continue
        for (i, ln) in enumerate(eachline(path; keep = false))
            occursin(rx, ln) && push!(hits, "$rel:$i  $(strip(ln))")
        end
    end
    return hits
end

# A per-box value keyed on the machine's name. `case "$(hostname)"` in shell and a `gethostname()`
# lookup in Julia are the two forms this repo has actually used. A `gethostname()` interpolated into a
# filename is NOT this — that is data, and the files it names are gitignored.
const _FN_SHAPES = [
    r"case\s+\"?\$\(\s*hostname\s*\)" => "a `case \$(hostname)` dispatch table",
    r"case\s+\"?\$HOSTNAME" => "a `case \$HOSTNAME` dispatch table",
    r"(get|ENV\[\")?hostname(\"\])?\s*\(\s*\)\s*(==|!=|in\b)" => "a `gethostname()` comparison",
    r"\[\s*gethostname\(\)\s*\]" => "a `gethostname()` lookup key",
]

"""Hostname-keyed per-box dispatch, which needs no name list and so works in CI too."""
function fleetshape_hits()
    hits = String[]
    for rel in _fn_tracked()
        _fn_skip(rel) && continue
        (endswith(rel, ".sh") || endswith(rel, ".jl")) || continue
        path = joinpath(_FN_ROOT, rel)
        isfile(path) || continue
        for (i, ln) in enumerate(eachline(path; keep = false))
            startswith(lstrip(ln), "#") && continue
            for (rx, what) in _FN_SHAPES
                occursin(rx, ln) && push!(hits, "$rel:$i  $what  —  $(strip(ln))")
            end
        end
    end
    return hits
end

function fleetname_lint()
    names = fleet_names()
    return (; names_checked = length(names), names = fleetname_hits(names), shapes = fleetshape_hits())
end

if abspath(PROGRAM_FILE) == @__FILE__
    r = fleetname_lint()
    if isempty(r.names) && isempty(r.shapes)
        if iszero(r.names_checked)
            println("fleet-name lint: SHAPE ONLY — no local name list (hostname and bench/fleet.local.sh \
                    gave none), so the name scan did NOT run. Shape check clean.")
        else
            println("fleet-name lint: PASS ($(r.names_checked) name(s) checked, no dispatch shapes)")
        end
    else
        println("fleet-name lint: FAIL")
        foreach(h -> println("  NAME   ", h, "\n           → name the MICROARCHITECTURE (Zen3/Zen4/Zen5/arm64), ",
                "not the machine"), r.names)
        foreach(h -> println("  SHAPE  ", h, "\n           → a box states its own values in bench/fleet.local.sh; ",
                "see bench/fleet.local.example.sh"), r.shapes)
        exit(1)
    end
end
