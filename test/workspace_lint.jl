# workspace self-alias lint — no L3Workspace FIELD may be claimed again from inside a live claim of it.
#
# Both real self-alias bugs have this exact static shape: an outer routine takes a workspace field, holds
# the view across a call, and something below re-derives THE SAME FIELD from the element type and writes
# it. (trsm_tmp, CONFIRMED 2026-09-01, relerr ~1.0 on ComplexF32 side='L'; rpack, confirmed by reading
# the two grow tests.) Both are history-dependent (right on call 1, wrong on call 2) and one of them is
# AVX2-only, so neither is reliably reachable by a numerical test on the gate box. This lint finds them
# by reading, from any box, in milliseconds.
#
# It replaces an audit that silently expires: "these two roles are never live at once" stops being true
# the moment someone adds a caller, and LAPACK-calls-public-BLAS makes that routine here.
#
# Method (deliberately the same regex+baseline shape as req8_lint.jl / estimator_lint.jl):
#   1. field names from `mutable struct L3Workspace` in src/workspace.jl;
#   2. an ACCESSOR is any src/ function whose body mentions `_l3ws(` and `<x>.<field>` — auto-discovered,
#      so a new accessor is covered without touching this file. `_trmm_bpf` (level3.jl, three module
#      globals + an IdDict — the 48th role, outside the struct) is declared by hand below;
#   3. a name-level call graph over src/**/*.jl. A function name passed as an ARGUMENT is an edge from
#      the callee to it, which is what puts the `_lacn2!`/`_lacn2_estimate` callbacks in the graph
#      instead of leaving them as an untracked hole;
#   4. finding = (field, claimant, transitively-reachable re-claimant). Conservative: it does not model
#      sequential re-claims, if/ternary arms, or claim-once-and-thread-it-down, so legal pairs land in
#      the baseline WITH A WRITTEN REASON. Both directions are enforced: a new pair fails, and a stale
#      baseline entry fails too (so the list cannot rot into a rubber stamp).

const _SRC = normpath(joinpath(@__DIR__, "..", "src"))
const _EXTRA_ACCESSORS = Dict("_trmm_bpf" => Set(["trmm_bpf"]))   # role outside L3Workspace (level3.jl:1091)

# Callback edges, DECLARED not exempted. A `::F` parameter is invisible to a name-level graph, so the two
# Higham–Hager estimators would otherwise be a silent hole: whatever their caller's closure does runs
# INSIDE a live claim of lacn*/lacn2*. Declaring the closure names puts every one of them (all methods, all
# callers, name-merged) in the graph. A new callback-taking routine that claims a field must be added here.
const _CALLBACK_EDGES = Dict("_lacn2!" => ["applyop!"],                  # gecon.jl:49  ← gecon!/trcon!/pocon!
                             "_lacn2_estimate" => ["apply!", "applyf"])  # trsen.jl:402 ← trsen!/trrfs!

_srcfiles() = sort!([joinpath(r, f) for (r, _, fs) in walkdir(_SRC) for f in fs if endswith(f, ".jl")])

function _ws_fields()
    src = read(joinpath(_SRC, "workspace.jl"), String)
    body = match(r"mutable struct L3Workspace\{T\}(.*?)\nend"s, src).captures[1]
    fs = Set(m.captures[1] for m in eachmatch(r"^\s{4}([a-z][a-z0-9_]*)::"im, body))
    # A field the name regex drops is a SILENT hole — the lint keeps passing while covering one role less.
    # (During development `[a-z0-9]*` dropped `trsm_tmp`, i.e. the one field with a confirmed bug.)
    all_fs = Set(m.captures[1] for m in eachmatch(r"^\s+([^\s:#]+)\s*::"m, body))
    isempty(setdiff(all_fs, fs)) || error("workspace lint: unparsed L3Workspace field(s) $(setdiff(all_fs, fs))")
    return fs
end

# name => body lines, merged over all methods of that name (over-approximates: more findings, not fewer)
function _bodies()
    b = Dict{String, Vector{String}}()
    defl = r"^(?:@\w+\s+)*function\s+([A-Za-z_][A-Za-z0-9_!]*)"
    for p in _srcfiles()
        cur = ""
        for l in readlines(p)
            s = strip(l)
            m = match(defl, s)
            nm = m === nothing ? _shortdef(s) : m.captures[1]
            nm === nothing || (cur = nm; get!(b, cur, String[]))
            (cur != "" && !startswith(s, "#")) && push!(b[cur], split(l, '#')[1])
        end
    end
    return b
end

# Short-form def `name(args) [where …] = rhs`, with the parens matched by COUNTING, not by regex. A regex
# here silently swallows ordinary code — `isone(alpha) || (B .*= alpha)` (level3.jl:4315) read as a def of
# `isone`, which stole the rest of trsm!'s body and made the lint blind to the confirmed trsm_tmp bug 28
# lines later. Base names are rejected outright: this file never needs to model a Base method.
function _shortdef(s::AbstractString)
    m = match(r"^(?:@\w+\s+)*([A-Za-z_][A-Za-z0-9_!]*)\(", s)
    m === nothing && return nothing
    nm = m.captures[1]
    isdefined(Base, Symbol(nm)) && return nothing
    d = 0; i = findnext('(', s, m.offset)
    for j in eachindex(s)
        j < i && continue
        c = s[j]; c == '(' && (d += 1); c == ')' && (d -= 1)
        if d == 0
            r = strip(s[nextind(s, j):end])
            startswith(r, "where") && (r = strip(replace(r, r"^where\s*(\{[^}]*\}|\S+)\s*" => "")))
            return (startswith(r, "=") && !startswith(r, "==")) ? nm : nothing
        end
    end
    return nothing
end

function ws_scan()
    fields = _ws_fields()
    bodies = _bodies()
    names = Set(keys(bodies))
    # accessor => field
    acc = Dict{String, Set{String}}(k => copy(v) for (k, v) in _EXTRA_ACCESSORS)
    for (f, lns) in bodies
        src = join(lns, '\n')
        occursin("_l3ws(", src) || continue
        for fld in fields
            occursin(Regex("\\.\\Q$fld\\E\\b"), src) && push!(get!(acc, f, Set{String}()), fld)
        end
    end
    claims = Dict{String, Set{String}}()   # function => fields it claims directly
    calls = Dict{String, Set{String}}()    # function => callees (name level)
    for (f, lns) in bodies
        cl, cs = Set{String}(), Set{String}()
        for l in lns
            heads = [m.captures[1] for m in eachmatch(r"\b([A-Za-z_][A-Za-z0-9_!]*)\(", l)]
            for nm in heads
                haskey(acc, nm) && union!(cs, acc[nm])
                (nm in names && nm != f) && push!(cl, nm)
            end
        end
        claims[f] = cs
        union!(get!(calls, f, Set{String}()), setdiff(cl, Set(n for n in cl if isdefined(Base, Symbol(n)))))
    end
    for (f, gs) in _CALLBACK_EDGES
        union!(get!(calls, f, Set{String}()), (g for g in gs if g in names))
    end
    reach(f) = (seen = Set{String}(); st = [f];
                while !isempty(st); x = pop!(st);
                    for y in get(calls, x, ()); (y in seen) || (push!(seen, y); push!(st, y)); end
                end; seen)
    viols = String[]
    for (f, cs) in claims, fld in cs, g in reach(f)
        haskey(acc, g) && continue                # the accessor itself IS f's claim, not a second one
        fld in get(claims, g, ()) && push!(viols, "$fld  $f -> $g")
    end
    return sort!(unique!(viols))
end

const _BASELINE = joinpath(@__DIR__, "workspace_lint_baseline.txt")
_key(s) = strip(split(s, '#')[1])

function ws_lint()
    cur = Set(ws_scan())
    base = Set(_key(l) for l in (isfile(_BASELINE) ? readlines(_BASELINE) : String[])
               if !isempty(strip(l)) && !startswith(strip(l), "#"))
    return (new = sort(collect(setdiff(cur, base))), stale = sort(collect(setdiff(base, cur))))
end

if abspath(PROGRAM_FILE) == @__FILE__
    if get(ARGS, 1, "") == "--baseline"
        open(_BASELINE, "w") do io
            println(io, "# workspace self-alias lint baseline: claim-inside-a-claim pairs REVIEWED AS LEGAL.")
            println(io, "# Every line needs a reason. A new pair fails the lint; so does a line here that no")
            println(io, "# longer occurs (regenerate with `julia test/workspace_lint.jl --baseline`).")
            foreach(v -> println(io, v, "  # REASON REQUIRED"), ws_scan())
        end
        println("wrote baseline: $(length(ws_scan())) pairs")
    else
        r = ws_lint()
        if isempty(r.new) && isempty(r.stale)
            println("workspace lint: PASS")
        else
            foreach(v -> println("NEW   ", v), r.new); foreach(v -> println("STALE ", v), r.stale)
            exit(1)
        end
    end
end
