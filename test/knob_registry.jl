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

"Classify a knob's DEFAULT expression. This is syntactic and deliberately conservative."
function _kr_tier(default)
    occursin(r"_at_[a-z]", default) && return "Derived"
    occursin(r"_L1_BYTES|_L2_BYTES|_L3_BYTES|_vwidth|_NVREG|_lanes\(|_SIMD_BYTES|_CACHELINE|_l1_block|_acc_cap|_ILP_TARGET|_datapath_bytes|_SCALAR_FPREGS|_L1D_ASSOC|isqrt|sizeof", default) && return "Derived"
    occursin(r"_W64|_double_pumped|_wide_simd|_INTEL_AVX2|_GT_NREG|_ZGT_W", default) && return "Predicate-keyed"
    occursin(r"^\s*nothing", default) && return "Pref-gated"
    occursin(r"^\s*(true|false)\s*\)", default) && return "Flag"
    occursin(r"^\s*-?\d+\s*\)", default) && return "Literal"
    occursin(r"^\s*_[A-Z]", default) && return "Coupled"
    return "Other"
end

"""
    knob_rows() -> Vector{NamedTuple}

Scan src/ for every `@load_preference` key: its file, owning family, const name, default expression,
syntactic tier, and any `# PDM:` marker within 12 lines above it.
"""
function knob_rows()
    rows = NamedTuple[]
    for (root, _, files) in walkdir(joinpath(_KR_ROOT, "src")), f in files
        endswith(f, ".jl") || continue
        rel = relpath(joinpath(root, f), joinpath(_KR_ROOT, "src"))
        lines = readlines(joinpath(root, f))
        # A `# PDM:` marker binds to the NEXT `@load_preference`, and is CONSUMED by it.
        # ⚠ The first version scanned UPWARD 12 lines from each knob, which silently gave a marker to
        # every knob within 12 lines of it — 4 of 15 "audited" rows were a neighbour's justification
        # (cgemv_rb, syr2k_2pass, cpotrf_base, pptrf_blk_min). A registry that mislabels a knob is
        # worse than one that admits it is unaudited, so binding is now one-to-one and forward-only.
        pending, pending_at = "", 0
        for (i, ln) in pairs(lines)
            mp = match(r"#\s*PDM:\s*(.+?)\s*$", ln)
            if !isnothing(mp)
                pending, pending_at = String(mp.captures[1]), i
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
            pdm = (!isempty(pending) && i - pending_at <= 12) ? pending : ""
            pending, pending_at = "", 0
            push!(rows, (; key, file = rel, family = _kr_family(rel),
                         const_name = isnothing(cm) ? "—" : String(cm.captures[1]),
                         default = isempty(default) ? "—" : default,
                         tier = _kr_tier(default), pdm))
        end
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
    println(io, "Every `@load_preference` key in `src/` — $(length(rows)) of them. The PDM ladder")
    println(io, "(`docs/src/tuning.md`) requires each to be **Derived** (default is a formula over detected")
    println(io, "consts) or **Measured** (it is not — which needs a justification and a `tune!()` cost).")
    println(io)
    println(io, "**Syntactic tier** is what the generator can see in the default expression; it is a")
    println(io, "*classification aid, not a verdict*. `Literal` means \"a bare number is the default\" — it")
    println(io, "may be a proven invariant (fine), a falsified derivation (fine, documented), or unconverted")
    println(io, "debt. The `# PDM:` marker is the human judgement and is the column that matters.")
    println(io)
    # summary
    tiers = Dict{String, Int}()
    for r in rows
        tiers[r.tier] = get(tiers, r.tier, 0) + 1
    end
    audited = count(r -> !isempty(r.pdm), rows)
    println(io, "| Syntactic tier | Count |")
    println(io, "|---|---|")
    for t in sort!(collect(keys(tiers)))
        println(io, "| $t | $(tiers[t]) |")
    end
    println(io)
    println(io, "**Audited: $audited / $(length(rows))** knobs carry a `# PDM:` marker. The rest are the")
    println(io, "worklist — an unaudited knob is one nobody has justified, which is exactly how redundant")
    println(io, "and duplicated knobs survive.")
    println(io)
    fam = ""
    for r in rows
        if r.family != fam
            fam = r.family
            println(io)
            println(io, "## $fam")
            println(io)
            println(io, "| Key | Const | Default | Tier | PDM justification |")
            println(io, "|---|---|---|---|---|")
        end
        j = isempty(r.pdm) ? "*unaudited*" : _kr_esc(r.pdm)
        println(io, "| `$(r.key)` | `$(r.const_name)` | `$(_kr_esc(r.default))` | $(r.tier) | $j |")
    end
    return String(take!(io))
end

if abspath(PROGRAM_FILE) == @__FILE__
    open(joinpath(_KR_ROOT, "docs", "src", "knobs.md"), "w") do io
        write(io, knob_markdown())
    end
    println("wrote docs/src/knobs.md ($(length(knob_rows())) knobs)")
end
