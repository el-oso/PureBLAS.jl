# THE KNOB REGISTRY — generator and verifier for docs/src/knobs.md.
#
# Every `@load_preference` key in src/ is a tuning knob, and the PDM ladder says each one is either
# DERIVED (default is a formula over detected consts) or MEASURED (it is not, and that needs a reason).
# Before this file there was no way to enumerate them: 118 keys across 14 files, five different spellings
# of the tier marker, and no list. That is how a knob comes to be duplicated, and how a wrong
# justification survives (see cpuinfo.jl (c5), where one lasted twelve days).
#
# DESIGN: the registry is GENERATED, never hand-maintained, so it cannot drift from the code. The
# justification lives in the SOURCE, next to the knob, in a structured marker:
#
#     # PDM: Measured — <why it cannot be derived> | tune: <cost in tune!()>
#     # PDM: Derived  — <the physical criterion>
#     # PDM: Exempt   — <why it is not a hardware knob at all: algorithm-intrinsic, flag, sentinel>
#
# placed on any comment line within 12 lines above the `@load_preference`. A knob with no marker is
# reported as UNAUDITED — that is not a failure, it is the worklist. The lint only fails if the
# generated table is out of date w.r.t. src/, which is what keeps the doc honest.
#
# Regenerate:  julia --project=test test/knob_registry.jl   (writes docs/src/knobs.md)

const _KR_ROOT = normpath(joinpath(@__DIR__, ".."))

"Owning routine family, from the file path — the coarse grouping readers actually navigate by."
function _kr_family(rel)
    d, f = splitdir(rel)
    f == "gemm.jl" && return "gemm (BLAS-3)"
    f == "level3.jl" && return "BLAS-3 (trmm/trsm/syrk/symm)"
    f == "level2.jl" && return "BLAS-2 (gemv/ger/trmv/trsv)"
    f == "level2_packed.jl" && return "BLAS-2 packed"
    f == "level2_banded.jl" && return "BLAS-2 banded"
    f == "simd_kernels.jl" && return "BLAS-1 SIMD kernels"
    f == "cpuinfo.jl" && return "CPU detection"
    f == "workspace.jl" && return "workspace"
    startswith(d, "lapack") && return "LAPACK · " * replace(f, ".jl" => "")
    return replace(f, ".jl" => "")
end

# THE ONLY TIERS. The PDM ladder asks one question — Derive, or Measure? — plus Exempt for knobs that
# are not hardware tuning at all (sentinels, capability flags).
#
# ⚠ AN EARLIER VERSION OF THIS FILE CLASSIFIED KNOBS SYNTACTICALLY, from what the default EXPRESSION
# looked like: Derived / Literal / Pref-gated / Predicate-keyed / Coupled / Flag / Other. That is not a
# taxonomy the project uses — it is a description of source text — and publishing it as "Tier" next to
# free-form marker labels (`Derived(sibling)`, `Literal(borrowed)`, `Measured(bounds)`, …) produced a
# zoo that buried the actual question. The syntactic guess is gone; the tier is now exactly what a
# human asserted in the marker, and nothing else.
const _KR_TIERS = ("Derived", "Measured", "Literal", "Exempt")

# What the default EXPRESSION reduces to. Mechanical, no judgement — and deliberately NOT called a
# tier, which is the mistake that produced the zoo. It is reported because it is the best available
# evidence about the 97 knobs nobody has classified yet: a `formula` default is almost certainly
# Derived, and without this column the summary reads "8 Derived" and implies the library is barely
# derived at all, when in fact 54 defaults are outright formulas.
function _kr_form(default)
    occursin(r"_at_[a-z]", default) && return "formula"
    occursin(r"_L1_BYTES|_L2_BYTES|_L3_BYTES|_vwidth|_NVREG|_lanes\(|_SIMD_BYTES|_CACHELINE|_l1_block|_acc_cap|_ILP_TARGET|_datapath_bytes|_SCALAR_FPREGS|_L1D_ASSOC|_W64|_double_pumped|_wide_simd|_INTEL_AVX2|_GT_NREG|_ZGT_W|isqrt|sizeof", default) && return "formula"
    occursin(r"^\s*nothing", default) && return "delegates"
    occursin(r"^\s*(true|false)", default) && return "flag"
    occursin(r"^\s*-?\d+", default) && return "literal"
    occursin(r"^\s*_[A-Z]", default) && return "sibling"
    return "other"
end

# STRICT marker syntax: `# PDM: <Tier> — <one concise line>`. Anything else is prose, not a marker.
# This matters: `src/` already contained comments like "# PDM: DERIVE tier — a residency criterion…"
# and "# PDM: P = `@load_preference`", written as prose long before the registry existed. A loose
# pattern scooped those up and attached them to whichever knob happened to follow.
const _KR_MARKER = Regex("#\\s*PDM:\\s*(" * join(_KR_TIERS, "|") * ")\\s+—\\s+(.+?)\\s*\$")

"""
    bind_knobs(lines, rel) -> Vector{NamedTuple}

THE ONE marker-binding implementation. Extracted so it can be fixture-tested — see
`test/knob_registry_tests.jl`, which pins it against a synthetic source whose answer is known by
construction, INCLUDING two adjacent marked knobs.

⚠ WHY THIS IS A NAMED FUNCTION AND NOT AN INLINE LOOP. Hand-rolled variants of exactly this scan have
mis-attributed FOUR times in this tree: the first registry binder (upward 12-line scan gave a marker to
every knob near it), `predicate_knob_lint` (state not reset at a definition boundary, blamed an
innocent knob), the one-off marker rewriter (took the earliest marker in its window, not the nearest),
and the audit skip-test (found a NEIGHBOUR's marker and silently skipped 16 knobs). Every one passed a
`!isempty(result)` sanity check, because every one produced plausible output about the wrong symbol.
If you need this logic somewhere else, CALL THIS — do not write the loop again.

A `# PDM:` marker binds to the NEXT `@load_preference` and is CONSUMED by it: forward-only, one-to-one.
"""
function bind_knobs(lines, rel = "?")
    rows = NamedTuple[]
    let
        pending, pending_at, pending_tier = "", 0, ""
        for (i, ln) in pairs(lines)
            mp = match(_KR_MARKER, ln)
            if !isnothing(mp)
                pending_tier = String(mp.captures[1])
                pending, pending_at = String(mp.captures[2]), i
                continue
            end
            m = match(r"@load_preference\(\"([^\"]+)\"\s*,?(.*)$", ln)
            isnothing(m) && continue
            key = m.captures[1]
            # Trim a trailing line comment: the capture runs to end-of-line, so `8)::Int  # req8-ok: …`
            # would publish the marker text inside the Default column. Only strip a ` #` that follows
            # whitespace, so a `#` inside a string default is left alone.
            default = strip(replace(String(m.captures[2]), r"\s+#.*$" => ""))
            cm = match(r"^\s*const\s+(\w+)", ln)
            # Consume a pending marker only if it sits within 12 lines — a marker further off belongs
            # to something else (or to nothing), and guessing is what caused the mis-binding above.
            near = !isempty(pending) && i - pending_at <= 12
            pdm = near ? pending : ""
            tier = near ? pending_tier : "Unaudited"
            pending, pending_at, pending_tier = "", 0, ""
            push!(rows, (; key, file = rel, family = _kr_family(rel),
                         const_name = isnothing(cm) ? "—" : String(cm.captures[1]),
                         default = isempty(default) ? "—" : default,
                         form = _kr_form(default), tier, pdm))
        end
    end
    return rows
end

"""
    knob_rows() -> Vector{NamedTuple}

Every `@load_preference` key under `src/`, with its file, owning family, const name, default, default
form, PDM tier and justification.
"""
function knob_rows()
    rows = NamedTuple[]
    for (root, _, files) in walkdir(joinpath(_KR_ROOT, "src")), f in files
        endswith(f, ".jl") || continue
        rel = relpath(joinpath(root, f), joinpath(_KR_ROOT, "src"))
        append!(rows, bind_knobs(readlines(joinpath(root, f)), rel))
    end
    return sort!(rows; by = r -> (r.family, r.key))
end

_kr_esc(s) = replace(strip(s), "|" => "\\|", "\n" => " ")

"Render the registry as the markdown of docs/src/knobs.md."
function knob_markdown()
    rows = knob_rows()
    io = IOBuffer()
    println(io, "# The Knob Registry")
    println(io)
    println(io, "!!! warning \"Generated file — do not edit\"")
    println(io, "    Produced by `test/knob_registry.jl` from `src/`. To change a row, change the code")
    println(io, "    or its `# PDM:` marker and regenerate. `test/knob_registry_tests.jl` fails if this")
    println(io, "    file is out of date.")
    println(io)
    println(io, "Every `@load_preference` key in `src/` — $(length(rows)) of them.")
    println(io)
    println(io, "| Tier | Meaning |")
    println(io, "|---|---|")
    println(io, "| **Derived** | Default is a formula over detected hardware consts. |")
    println(io, "| **Measured** | Not derivable — the optimum depends on something we cannot detect. Wants a `tune!()` pin. |")
    println(io, "| **Literal** | A fixed value: a proven invariant, or a derivation that was tried and falsified. |")
    println(io, "| **Exempt** | Not hardware tuning at all — a sentinel or a capability flag. |")
    println(io, "| **Unaudited** | Nobody has classified it yet. Debt, not a verdict. |")
    println(io)
    # summary
    tiers = Dict{String, Int}()
    for r in rows
        tiers[r.tier] = get(tiers, r.tier, 0) + 1
    end
    audited = count(r -> r.tier != "Unaudited", rows)
    forms = Dict{String, Int}()
    for r in rows
        forms[r.form] = get(forms, r.form, 0) + 1
    end
    print(io, "**Reviewed tier:** ")
    println(io, join(["$(get(tiers, t, 0)) $t" for t in (_KR_TIERS..., "Unaudited")], " · "), ".")
    print(io, "**Default form** (mechanical, all $(length(rows))): ")
    println(io, join(["$(get(forms, f, 0)) $f" for f in ("formula", "delegates", "sibling", "literal", "flag", "other")], " · "), ".")
    println(io)
    println(io, "**$audited / $(length(rows)) reviewed.** ⚠ READ THE TWO LINES ABOVE TOGETHER. `Unaudited`")
    println(io, "means *nobody has classified it yet* — NOT that it is un-derived. The **default form** column")
    println(io, "is the mechanical evidence about those knobs, and it says $(get(forms, "formula", 0)) of $(length(rows)) defaults are outright")
    println(io, "formulas over detected consts. Several of the `delegates` rows resolve through an `_at_*`")
    println(io, "formula under a different name too (e.g. `pbtrf_nb_small` → `_at_pbtrf_nbs`). So the library")
    println(io, "is *mostly derived*; the reviewed-tier counts are small because the review is young, and")
    println(io, "reading them alone understates it badly.")
    println(io)
    fam = ""
    for r in rows
        if r.family != fam
            fam = r.family
            println(io)
            println(io, "## $fam")
            println(io)
            println(io, "| Knob | Default | Tier | Why | `tune!()` |")
            println(io, "|---|---|---|---|---|")
        end
        # The marker's ` | tune: …` tail is a SEPARATE column, not text — escaping it into the prose
        # (`\|`) is what it looked like first, and it read as noise.
        why, tune = if occursin("|", r.pdm)
            a, b = split(r.pdm, "|"; limit = 2)
            (strip(a), replace(strip(b), r"^tune:\s*" => ""))
        else
            (r.pdm, "")
        end
        println(io, "| `$(r.key)` | $(r.form) | $(r.tier) | $(isempty(why) ? "—" : _kr_esc(why)) | ",
                isempty(tune) ? "—" : _kr_esc(tune), " |")
    end
    println(io)
    println(io, "---")
    println(io)
    println(io, "Const names, defaults and files are deliberately NOT tabulated: they are one `grep` away")
    println(io, "and made this table too wide to read. The knob key is the identifier that matters.")
    return String(take!(io))
end

if abspath(PROGRAM_FILE) == @__FILE__
    open(joinpath(_KR_ROOT, "docs", "src", "knobs.md"), "w") do io
        write(io, knob_markdown())
    end
    println("wrote docs/src/knobs.md ($(length(knob_rows())) knobs)")
end
