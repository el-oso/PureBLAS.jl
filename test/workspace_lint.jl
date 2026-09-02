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
# EVERY NAME HERE MUST RESOLVE TO A DEFINITION IN src/ OR ws_scan() ERRORS — the edges below were silently
# dropped for their whole existence because `applyf`/`apply!` are anonymous-function assignments that
# `_bodies` did not record, and a filtered-away edge looks exactly like a working one.
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
        instr = false           # inside an open triple-quoted literal (docstring OR ordinary string)
        for l0 in readlines(p)
            # STRIP THE COMMENT TAIL BEFORE THE QUOTE MACHINE SEES IT. Counting `\"\"\"` on the raw
            # line lets a triple quote written inside a COMMENT flip parity for the rest of the file.
            # That is not hypothetical: src/lapack/eigen.jl:741 is a comment describing this very fix
            # and contains a backticked triple quote, which put the scanner into string mode and
            # swallowed EIGHT definitions after it — orgtr!/ungtr! (both forms), _sterf!, _syev!,
            # _heev! — so the `_syev! -> _ormtr!` edge (a real claimant of mtrv/mtrg/mtrw/mtrt) went
            # invisible, and the docstring below was mis-attributed as body lines. That is the exact
            # phantom/hidden-edge class this scanner was repaired to remove, reintroduced by the
            # repair itself, and in the HIDES-FINDINGS direction. The old `startswith(strip(l), …)`
            # form did not have it. A `#` inside a string literal would truncate that line early —
            # acceptable here, since the body of a string carries no calls worth graphing.
            l = split(l0, '#')[1]
            s = strip(l)
            # Skip the BODY of every triple-quoted literal, but kill attribution only for the ones that
            # are actually docstrings. Those are two different jobs and the old code conflated them:
            # it tested `strip(l)`, so an INDENTED `"""` — the llvmcall IR in `_prefetch`
            # (blas3/gemm.jl:68/75) — set doc-mode and dropped the rest of that function on the floor.
            # That direction HIDES findings, which is the dangerous way for this lint to be wrong.
            if instr
                instr = !occursin("\"\"\"", l)
                continue            # a closing line's tail (`""", "entry",`) carries no call worth keeping
            end
            # DOCSTRINGS TERMINATE ATTRIBUTION, and this is load-bearing, not tidiness. Lines are
            # attributed to the preceding definition until the next one, so a docstring written ABOVE a
            # function — the normal Julia placement — has its indented signature lines swallowed by
            # whatever was defined before it. Those lines look exactly like calls: a `"""` block above
            # `orgtr!` containing `orgtr!(uplo, A, tau, Q)` injected phantom `_unmtr! -> orgtr!` and
            # `_unmtr! -> ungtr!` edges, and the lint then reported 8 aliasing pairs that do not exist
            # (`_ormtr!` is real-only, `_unmtr!` complex-only; for complex T they do not even share an
            # `_l3ws` owner, and neither calls the other). It was a FALSE POSITIVE that only surfaced
            # once `_unmtr!` began claiming a field — so the edges had been there, silently wrong, since
            # the docstring was written. Every `"""`-above-a-signature in src/ (gesvx!, getrf!, …) is the
            # same shape, so this must be fixed here rather than by moving docstrings around the source.
            if occursin("\"\"\"", l)
                # An ODD count leaves the literal open; an even one (`"""text"""`, `x = """a""" `) closes
                # on the line. A DOCSTRING is one that starts at column 0 — `"""` or `raw"""`; anything
                # else (indented, or `x = """…`) is an ordinary string and must not touch attribution.
                isodd(count("\"\"\"", l)) && (instr = true)
                (startswith(l, "\"\"\"") || startswith(l, "raw\"\"\"")) && (cur = "")
                continue
            end
            m = match(defl, s)
            nm = m === nothing ? _anondef(s) : m.captures[1]
            nm === nothing || (cur = nm; get!(b, cur, String[]))
            (cur != "" && !startswith(s, "#")) && push!(b[cur], split(l, '#')[1])
        end
    end
    return b
end

# `name = function (args)` / `name = (args) -> …`. The two Higham–Hager callbacks are written this way
# (`applyf`, trrfs.jl:127; `apply!`, trsen.jl:560/622), and _shortdef never saw them — so the declared
# `_lacn2_estimate` edges below had nothing to resolve to and were dropped in silence.
const _ANONDEF = r"^([A-Za-z_][A-Za-z0-9_!]*)\s*=\s*(?:function\b|(?:\([^)]*\)|[A-Za-z_][A-Za-z0-9_!]*)\s*->)"
function _anondef(s::AbstractString)
    m = match(_ANONDEF, s)
    m === nothing && return _shortdef(s)
    return isdefined(Base, Symbol(m.captures[1])) ? nothing : m.captures[1]
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
    # A DECLARED EDGE THAT DOES NOT RESOLVE IS AN ERROR. Silently filtering to `g in names` is how both
    # `_lacn2_estimate` edges sat inert while the header advertised them as the thing that closes the
    # callback hole: a config entry that quietly does nothing is indistinguishable from one that works.
    for (f, gs) in _CALLBACK_EDGES
        f in names || error("workspace lint: declared callback edge source `$f` is not a definition in src/")
        bad = filter(g -> !(g in names), gs)
        isempty(bad) || error("workspace lint: declared callback edge(s) `$f` -> $bad do not resolve to a " *
                              "definition in src/ (renamed callback, or a def form _bodies cannot see)")
        union!(get!(calls, f, Set{String}()), gs)
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
