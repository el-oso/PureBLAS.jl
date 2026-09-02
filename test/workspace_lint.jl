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

# `src/verify.jl` is EXCLUDED, and that is a correctness fix, not a convenience. It is a probe harness:
# `@verify_strict SIMDBackend begin … end` calls a few hundred routines SEQUENTIALLY at one nesting level.
# A name-level call graph cannot tell that block apart from a function body, so every routine in it reads
# as calling every routine after it — which manufactured 48 false findings the moment SVDWorkspace and
# _GVDWork came under the lint (`gesvd!` "calling" `tbsv!` "calling" `stein!`). It defines no accessor and
# sits on no algorithm path, so dropping it loses no coverage.
# `src/contracts.jl` is excluded for the same reason in a different disguise. It is a wall of bare
# forward declarations (`function tbsv! end`), so `_bodies()` opens a body for each one and attributes
# every following line to it — including the `@strict_contract` member lists, which name every routine
# in the package. That made `tbsv!` look like it called `gesvd!`. It defines no accessor either.
const _SKIP = Set(["verify.jl", "contracts.jl"])
_srcfiles() = sort!([joinpath(r, f) for (r, _, fs) in walkdir(_SRC)
                     for f in fs if endswith(f, ".jl") && !(f in _SKIP)])

# EVERY GKH-owned scratch struct, not just L3Workspace. `SVDWorkspace`, `_DCWork`, `_EDCWork`, `_TRDWork`
# and `_GVDWork` hold reusable buffers with exactly the same self-alias hazard, and covering only
# L3Workspace left ~70 roles unwatched (kb `gkh-borrow-design`, ranked item #2). Roles are keyed
# `<Struct>.<field>` because field names COLLIDE across them — n, e, W, Z, tmp all recur — and merging
# them by bare name would invent findings across unrelated workspaces.
const _OWNER_CALLS = Dict("L3Workspace" => "_l3ws(",
                          "SVDWorkspace" => "_svdws(",
                          "_DCWork" => "_get_dcwork(",
                          "_EDCWork" => "_edcws(",
                          "_TRDWork" => "_trdws(",
                          "_GVDWork" => "_gvdws(")

# Struct => its field names. Two traps this must survive, both of them the SILENT-hole class:
#   * several fields per line (`ec::Vector{T}; z::Vector{T}`) — `_EDCWork` is written that way, and a
#     `^`-anchored regex sees only the first, so the line is split on `;` first;
#   * CAPITALISED field names (`Zsort`, `R`, `V`, `B`, `Vg`, `Ztmp`) — the old `[a-z]` anchor would have
#     dropped all six while the lint kept reporting clean.
# The `all_fs` cross-check below is what turns either mistake into an error instead of silent under-coverage.
function _ws_fields()
    out = Dict{String, Set{String}}()
    for p in _srcfiles()
        src = read(p, String)
        # `mutable` and the type parameters are both optional: `_DCWork` is a plain immutable `struct`
        # with no parameters, and it is exactly the recursion-shared workspace most at risk here.
        for m in eachmatch(r"(?:mutable\s+)?struct (\w+)(?:\{[^{}]*\})?\n(.*?)\nend"s, src)
            name, body = m.captures[1], m.captures[2]
            haskey(_OWNER_CALLS, name) || continue
            fs, all_fs = Set{String}(), Set{String}()
            for line in split(body, '\n')
                code = split(line, '#')[1]
                for part in split(code, ';')
                    mm = match(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*::", part)
                    mm === nothing || push!(fs, mm.captures[1])
                    ma = match(r"^\s*([^\s:#]+)\s*::", part)
                    ma === nothing || push!(all_fs, ma.captures[1])
                end
            end
            isempty(setdiff(all_fs, fs)) ||
                error("workspace lint: unparsed $name field(s) $(setdiff(all_fs, fs))")
            out[name] = fs
        end
    end
    missing_ = setdiff(Set(keys(_OWNER_CALLS)), Set(keys(out)))
    isempty(missing_) ||
        error("workspace lint: declared owner struct(s) $missing_ not found in src/ (renamed or removed)")
    return out
end

# name => body lines, merged over all methods of that name (over-approximates: more findings, not fewer)
function _bodies()
    b = Dict{String, Vector{String}}()
    defl = r"^(?:@\w+\s+)*function\s+([A-Za-z_][A-Za-z0-9_!]*)"
    for p in _srcfiles()
        cur = ""
        carry = 0               # lines still owed to a short-form def whose `=` ended the previous line
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
            isshort = false
            nm = if m === nothing
                x = _anondef(s)
                # `name = function (args)` and `name = (args) -> begin` are LONG forms that happen to
                # match the anon-def pattern; their bodies run to an `end`. Clearing `cur` after them
                # dropped the two Higham-Hager callbacks (`apply!`, `applyf`) entirely, which took the
                # declared `_lacn2!`/`_lacn2_estimate` edges with them — six baseline entries went stale.
                isshort = !isnothing(x) && !occursin(r"=\s*function\b", s) &&
                    !occursin(r"->\s*(?:begin)?\s*$", s)
                x
            else
                m.captures[1]                       # long form: `function f(...)`, body runs to its `end`
            end
            nm === nothing || (cur = nm; get!(b, cur, String[]))
            (cur != "" && !startswith(s, "#")) && push!(b[cur], split(l, '#')[1])
            # A SHORT-FORM def (`f(x) = rhs`) ends ON ITS LINE. Leaving `cur` set attributed the REST OF
            # THE FILE to it — native.jl:92 is `@inline tbsv!(AB, x; uplo=…) =` with the RHS on the next
            # line, so `tbsv!` absorbed every following one-liner in native.jl and appeared to call
            # `gesvd!`. That invented `tbsv! -> stein!`-shaped findings across SVDWorkspace roles.
            # A trailing `=` means the RHS is on the next line, so hold attribution for exactly one more.
            if isshort
                if endswith(s, "=")
                    carry = 1
                else
                    cur = ""
                end
            elseif carry > 0
                carry -= 1
                carry == 0 && (cur = "")
            end
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
            # A RETURN-TYPE ANNOTATION sits between the parens and the `=`: backend.jl writes every
            # contract forward as `f(::SIMDBackend, …)::AbstractMatrix = f(…)`. Without this the whole
            # file reads as ONE definition — the first `@inline function` in it absorbed 148 lines
            # naming `gesvd!`, `stein!`, … and manufactured `tbsv! -> _gesvd_core!`-shaped findings
            # across every SVDWorkspace role. Cut at the first `=` that is not `==`.
            if startswith(r, "::")
                k = findfirst(r"(?<![=!<>])=(?!=)", r)
                isnothing(k) && return nothing
                r = strip(r[first(k):end])
            end
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
        for (structname, ownercall) in _OWNER_CALLS
            occursin(ownercall, src) || continue          # attribute fields to the owner this body fetches
            for fld in fields[structname]
                occursin(Regex("\\.\\Q$fld\\E\\b"), src) &&
                    push!(get!(acc, f, Set{String}()), string(structname, '.', fld))
            end
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
