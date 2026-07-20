# req#8 lint — catches machine-dependent TUNING numbers hardcoded as literals instead of derived from
# detected hardware consts (`_vwidth`, `_L1_BYTES`, `_ILP_TARGET`, …). It exists because the trap is
# invisible in review: you can DERIVE a value (`_ILP_TARGET÷2`) yet still MATERIALIZE it as a literal
# `Val(8)` dispatch arm — the value is traceable, the code is not. See CLAUDE.md req#8.
#
# Two flagged patterns:
#   1. `Val(<int ≥ 2>)`         — a dispatch arm frozen to a literal (use `Val(<derived-const>)`).
#   2. `const _NAME = <bare int>` or `@load_preference("k", <bare int>)` — a tuning const that should be a
#      formula over detected consts (or an overridable pref whose DEFAULT is a formula).
#
# Escape hatch: a `# req8-ok: <reason>` comment on the same line or the line above, naming the invariant /
# algorithmic fact / measured crossover that justifies the literal (e.g. a 2×2 recursion split, a mask, a
# documented µarch-invariant). Anything flagged WITHOUT that annotation fails. The lint can't tell a legit
# algorithmic `4` from a tuning `4`; it flags candidates and forces an explicit, reviewer-signed justification.

const _SRCDIR = joinpath(@__DIR__, "..", "src")
const _ANNOT = r"#\s*req8-ok:"i
const _VAL_LIT = r"\bVal\(\s*(\d+)\s*\)"
const _CONST_DEF = r"^\s*const\s+(_[A-Z][A-Z0-9_]*)\s*=\s*(.*)$"

# A const RHS is "bare" (unjustified) if it's a plain integer, OR an @load_preference whose default is a plain
# integer, OR a pure-integer arithmetic expression with NO reference to a detected const (lower/underscore name).
_isbare(s) = (v = tryparse(Int, strip(s)); v !== nothing && v >= 2)   # 0/1 = disable-flag/identity, not tuning
function _bare_int_rhs(rhs::AbstractString)
    r = strip(split(rhs, '#')[1])
    _isbare(r) && return true
    m = match(r"^@load_preference\([^,]+,\s*([^)]+)\)", r)
    m !== nothing && _isbare(m.captures[1]) && return true   # @load_preference("k", 32) — bare-int default
    return false
end

function req8_scan()
    viols = String[]
    for f in sort(readdir(_SRCDIR; join = true))
        endswith(f, ".jl") || continue
        lines = readlines(f)
        for (i, ln) in enumerate(lines)
            (i > 1 && occursin(_ANNOT, lines[i - 1])) && continue
            occursin(_ANNOT, ln) && continue
            code = split(ln, '#')[1]
            for m in eachmatch(_VAL_LIT, code)
                v = parse(Int, m.captures[1])
                v >= 2 && push!(viols, "$(basename(f)):$i  Val($v)")
            end
            cm = match(_CONST_DEF, code)
            if cm !== nothing && _bare_int_rhs(cm.captures[2])
                push!(viols, "$(basename(f)):$i  const $(cm.captures[1]) = $(strip(split(cm.captures[2], '#')[1]))")
            end
        end
    end
    return viols
end

# Baseline = the known literals pending annotation/derivation (explicit debt). The lint FAILS only on flags
# NOT in the baseline — so a new `Val(8)`-style trap is caught immediately, while the existing set is whittled
# down over time (remove a line from the baseline once its literal is derived or `# req8-ok:`-annotated).
const _BASELINE = joinpath(@__DIR__, "req8_lint_baseline.txt")
# strip the "file:line" prefix so a baselined literal isn't defeated by unrelated line shifts — match on the
# (file, literal-text) pair.
_key(s) = (p = split(s, "  "; limit = 2); string(basename(strip(p[1])), "  ", length(p) > 1 ? strip(p[2]) : ""))
_stripline(s) = replace(s, r"^([^:]+):\d+" => s"\1")

function req8_new_violations()
    cur = Set(_key(_stripline(v)) for v in req8_scan())
    base = isfile(_BASELINE) ? Set(_key(strip(l)) for l in readlines(_BASELINE) if !isempty(strip(l)) && !startswith(strip(l), "#")) : Set{String}()
    return sort(collect(setdiff(cur, base)))
end

if abspath(PROGRAM_FILE) == @__FILE__
    if get(ARGS, 1, "") == "--baseline"        # regenerate the debt list
        open(_BASELINE, "w") do io
            println(io, "# req#8 lint baseline — known tuning literals pending derivation or `# req8-ok:` annotation.")
            println(io, "# The lint fails on any flag NOT listed here. Whittle this down; do not add to it without cause.")
            for v in sort(req8_scan())
                println(io, _stripline(v))
            end
        end
        println("wrote baseline: $(length(req8_scan())) entries")
    else
        nv = req8_new_violations()
        if isempty(nv)
            println("req8 lint: PASS (no NEW unjustified tuning literals beyond the baseline)")
        else
            println("req8 lint: FAIL — $(length(nv)) NEW tuning literal(s) not derived/annotated/baselined:")
            for x in nv
                println("  ", x)
            end
            exit(1)
        end
    end
end
