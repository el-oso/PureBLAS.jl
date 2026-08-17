# Recompute the GATED verdict and the geo/worst columns of docs/src/coverage.md's hand-maintained
# ROUTING tables, straight from the v3 caches, and render the verdict as a colour band instead of a
# 🐰/🐢 glyph — the same treatment bench/coverage_ops.jl already gives the BLAS tables.
#
# WHY. Those tables were the last hand-typed numbers on the page, and hand-typed numbers drift: on
# 2026-08-06 four rows carried a verdict their OWN figures contradicted, and `geqp3` still had a
# footnote calling its blocked port "reverted, not shipped" months after it landed and started gating.
# The glyph made it worse by flattening 0.999 and 0.61 into the same 🐢. Everything here is computed
# from the cache, so a stale row becomes impossible rather than merely unlikely.
#
# WHAT STAYS HAND-WRITTEN, deliberately: Op, Routines, Types and Routes. Those are documentation — what
# exists and what it dispatches to — not measurements, and no cache can produce them.
#
# THE MAP COMES FROM THE PAGE ITSELF. The `Routines` column already names the cache rows, so this
# parses it rather than carrying a second copy that could disagree. Names that do not match a cache row
# 1:1 get an alias below; anything still unmatched is REPORTED, never silently skipped — an unmatched
# row is exactly how a stale number would survive.
#
# ⚠ ONE CACHE PER MICROARCHITECTURE, ONE COLUMN EACH — never several caches pooled into one verdict.
# This tool used to REFUSE multiple caches. That guard identified a real hazard and drew the wrong
# conclusion from it: fed the fleet, the old code min-ed ACROSS boxes into a single number, and on
# 2026-08-07 that published `Banded Cholesky` as 0.891 MISSING when Zen4 alone read 1.03 GATING —
# four of five rows on that table moved. POOLING is what is unsafe. Giving each box its own column is
# not, and it is what these rows always needed: the gate is per box, so one verdict cannot express
# "gates on Zen4, misses on Zen3" — precisely the complex gemv-T/C split. Same reasoning
# bench/coverage_ops.jl already applies to the BLAS tables.
#
# Column order follows ARGUMENT order and must match the table headers in docs/src/coverage.md.
# The `vs OB / vs AOCL geo/worst` columns stay scoped to the FIRST cache given and the header says so
# ("Zen4 vs OB …"); they are a per-reference detail, not a per-box verdict.
#
# Usage:  julia --project=bench bench/coverage_routing.jl \
#             bench/plots_data_avx512_wintermute.txt \
#             bench/plots_data_avx2_galen.txt \
#             bench/plots_data_zen5_neuromancer.txt
using Printf

const QN = 48
med(v) = sort(v)[max(1, cld(length(v), 2))]

# routine token in the docs  =>  cache row name(s). Only the ones that are not identity.
const ALIAS = Dict(
    "potrf uplo='U'" => ["potrfU"], "potrf `uplo='U'`" => ["potrfU"], "potrf" => ["potrf"],
    "potrs" => ["potrsL", "potrsU"], "pstrf" => ["pstrf", "pstrfU"],
    "pbtrf" => ["pbtrfL", "pbtrfU"], "pptrf" => ["pptrfL", "pptrfU"],
    "getrf, gesv" => ["getrf"], "geqrf, orgqr, ormqr" => ["geqrf"],
    "sytrf, hetrf" => ["sytrf"], "sytrs, hetrs" => ["sytrs"],
    "gesvd, gesdd" => ["gesvd"], "syev, heev" => ["syev", "syevN"],
    "gttrf, gttrs, gtsv" => ["gttrf", "gttrs", "gtsv"],
    "pttrf, pttrs, ptsv" => ["pttrf", "pttrs", "ptsv"],
)

function cells(paths)
    out = Dict{String, Vector{Tuple{Int, Float64}}}()   # routine => [(size, gate ratio)]
    for p in paths, ln in eachline(p)
        (isempty(strip(ln)) || startswith(ln, "#")) && continue
        f = split(ln, "\t"); length(f) >= 4 || continue
        f[1] in ("LP", "CLP") || continue
        d = Dict{String, Vector{Float64}}()
        for g in f[4:end]
            _p = split(g, "|"); a, csv = _p[1], _p[end]
            d[String(a)] = parse.(Float64, split(csv, ","))
        end
        haskey(d, "pb") || continue
        rr(q) = (n = min(length(q), length(d["pb"])) ÷ QN;
            n == 0 ? NaN : med([med([q[(r - 1) * QN + i] / d["pb"][(r - 1) * QN + i] for i in 1:QN]) for r in 1:n]))
        rs = [rr(d[r]) for r in ("openblas", "aocl") if haskey(d, r)]
        rs = filter(isfinite, rs)
        isempty(rs) && continue
        push!(get!(out, String(f[2]), Tuple{Int, Float64}[]), (parse(Int, f[3]), minimum(rs)))
    end
    return out
end

# per-reference geo/worst, so the two numeric columns stay separable
function refstats(paths, ref)
    out = Dict{String, Vector{Float64}}()
    for p in paths, ln in eachline(p)
        (isempty(strip(ln)) || startswith(ln, "#")) && continue
        f = split(ln, "\t"); length(f) >= 4 || continue
        f[1] in ("LP", "CLP") || continue
        d = Dict{String, Vector{Float64}}()
        for g in f[4:end]
            _p = split(g, "|"); a, csv = _p[1], _p[end]
            d[String(a)] = parse.(Float64, split(csv, ","))
        end
        (haskey(d, "pb") && haskey(d, ref)) || continue
        n = min(length(d[ref]), length(d["pb"])) ÷ QN
        n == 0 && continue
        r = med([med([d[ref][(k - 1) * QN + i] / d["pb"][(k - 1) * QN + i] for i in 1:QN]) for k in 1:n])
        isfinite(r) && push!(get!(out, String(f[2]), Float64[]), r)
    end
    return out
end

geo(v) = isempty(v) ? NaN : exp(sum(log, v) / length(v))

function band(g)                      # same thresholds/classes as bench/coverage_ops.jl
    isnan(g) && return "b1"
    g >= 1.0 && return "ok"
    g >= 0.99 && return "b1"
    g >= 0.95 && return "b2"
    g >= 0.85 && return "b3"
    return "b4"
end

function routines_of(col)
    s = replace(col, "`" => "")
    s = replace(s, r"\[\^[a-z0-9]+\]" => "")          # drop footnote markers
    s = strip(s)
    haskey(ALIAS, s) && return ALIAS[s]
    # ...otherwise expand TOKEN BY TOKEN, because a column like "pbtrf, pbtrs" is two routines and only
    # the first has a cache alias (pbtrfL/pbtrfU). Matching whole columns only left those rows unmapped.
    out = String[]
    for t in split(s, ",")
        tok = strip(t)
        isempty(tok) && continue
        append!(out, get(ALIAS, tok, [tok]))
    end
    return out
end

"µarch label from a cache header, e.g. \"Zen4\" — the column key. Authoritative (the measuring box
stamps its own CpuId), never re-derived from the filename."
function uarch_of(path)
    for ln in eachline(path)
        startswith(ln, "#pbbench") || continue
        m = match(r"uarch=(\S+)", ln)
        return isnothing(m) ? basename(path) : String(m.captures[1])
    end
    return basename(path)
end

function main()
    paths = ARGS
    isempty(paths) && error("usage: coverage_routing.jl <cache> [more…]  (one per microarchitecture)")
    # ⚠ ONE COLUMN PER BOX — NOT one pooled verdict over several caches. The earlier version of this
    # tool REFUSED multiple caches, and that guard was right about the hazard and wrong about the
    # remedy: feeding it the fleet made it min ACROSS boxes into a single number, which published
    # `Banded Cholesky` as 0.891 FAILING when Zen4 alone read 1.03. Pooling is what is unsafe; showing
    # each box its own column is not, and it is what these rows needed all along — the gate is per box,
    # so a single verdict cannot express "gates on Zen4, misses on Zen3" (exactly the complex gemv-T/C
    # split). Same reasoning bench/coverage_ops.jl already applies to the BLAS tables.
    boxes = [(uarch_of(p), p) for p in paths]
    length(unique(first.(boxes))) == length(boxes) ||
        error("two caches report the same uarch — that would silently overwrite a column: " *
            join(first.(boxes), ", "))
    gates_by = Dict(u => cells([p]) for (u, p) in boxes)
    # geo/worst are a per-REFERENCE detail, not a per-box verdict, and the header names which box they
    # describe. They used to be scoped to `boxes[1]`, i.e. to POSITION — which silently re-pointed them
    # at a different µarch the moment the column order changed (columns are now sorted Zen3/Zen4/Zen5,
    # so boxes[1] became Zen3). Pin them by NAME instead, so ordering the columns and choosing the
    # reference box are independent decisions and the header cannot drift out of sync with the data.
    refu = get(ENV, "PUREBLAS_COVERAGE_REF", "Zen4")
    refi = findfirst(b -> occursin(refu, first(b)), boxes)
    isnothing(refi) && error("geo/worst reference \"$refu\" is not among the caches given: " *
        join(first.(boxes), ", ") * "  (set PUREBLAS_COVERAGE_REF)")
    ob = refstats([last(boxes[refi])], "openblas"); ao = refstats([last(boxes[refi])], "aocl")
    gates = gates_by[first(boxes[refi])]
    println("geo/worst reference: ", first(boxes[refi]), "   column order: ", join(first.(boxes), " | "))

    doc = collect(eachline("docs/src/coverage.md"))
    unmatched = String[]; changed = 0
    for (i, ln) in enumerate(doc)
        startswith(ln, "| ") || continue
        # Trigger on ANY verdict form the Gated column has ever held — the retired glyphs, the escaped
        # inline span from the bad first attempt, or the bolded number it holds now — so re-running is
        # idempotent. Matching only the glyph made the script a no-op the moment it had run once.
        occursin(r"🐰|🐢|<span class=\"pbg|\| *\*{0,2}[01]\.?[0-9]*\*{0,2} *\|", ln) || continue
        parts = split(ln, "|")
        # TWO TABLE SHAPES: 8 columns (with geo/worst) and 5 (verdict only, e.g. eigensolvers).
# The 5-column form was skipped by an earlier `>= 8` guard, which is why glyphs survived there.
# Shapes now scale with the number of boxes. With B per-box columns a row splits to
#   narrow: Op|Routines|Types|Routes|<B verdicts>            -> 4 + B + 2 parts
#   wide:   … plus vs-OB and vs-AOCL geo/worst               -> 4 + B + 2 + 2 parts
        nb = length(boxes)
        length(parts) >= 6 + nb || continue
        wide = length(parts) >= 8 + nb
        rts = routines_of(String(parts[3]))
        gs = Float64[]; obs = Float64[]; aos = Float64[]
        miss = String[]
        for r in rts
            if haskey(gates, r)
                append!(gs, last.(gates[r])); append!(obs, get(ob, r, Float64[])); append!(aos, get(ao, r, Float64[]))
            else
                push!(miss, r)
            end
        end
        if isempty(gs)
            push!(unmatched, "line $i: " * strip(String(parts[2])) * "  [" * join(rts, ", ") * "]")
            continue
        end
        isempty(miss) || push!(unmatched, "line $i PARTIAL: missing " * join(miss, ", "))
        w = minimum(gs)
        # ⚠ MARKDOWN-SAFE, NOT INLINE HTML. A `<span>` inside a markdown table cell is ESCAPED by the
        # Vitepress pipeline and publishes as literal `&lt;span class=&quot;pbg&quot;…` text — I shipped
        # exactly that, and it read worse than the glyphs it replaced. The BLAS tables can carry colour
        # because they are emitted as whole ```@raw html blocks; these routing tables cannot, because
        # their cells hold Documenter footnote refs ([^upper]) which raw HTML would not resolve.
        # So the verdict is the NUMBER, bolded when it misses. That still fixes the actual complaint
        # against 🐰/🐢 — it distinguishes 0.999 from 0.61, which one glyph could not.
        verdict(x) = isnan(x) ? " — " : x >= 1.0 ? @sprintf("%.3g", x) : @sprintf("**%.3g**", x)
        # One verdict cell per box, in the order the caches were given (= the header order).
        for (k, (u, _)) in enumerate(boxes)
            gu = Float64[]
            for r in rts
                haskey(gates_by[u], r) && append!(gu, last.(gates_by[u][r]))
            end
            parts[5 + k] = " " * verdict(isempty(gu) ? NaN : minimum(gu)) * " "
        end
        if wide                      # geo/worst columns follow the per-box block (first cache only)
            parts[6 + length(boxes)] = @sprintf(" %.3g / %.3g ", geo(obs), isempty(obs) ? NaN : minimum(obs))
            parts[7 + length(boxes)] = @sprintf(" %.3g / %.3g ", geo(aos), isempty(aos) ? NaN : minimum(aos))
        end
        doc[i] = join(parts, "|"); changed += 1
    end
    write("docs/src/coverage.md", join(doc, "\n") * "\n")
    @printf("rewrote %d routing rows from cache\n", changed)
    if !isempty(unmatched)
        println("\n⚠ ROWS NOT FULLY MAPPED — fix ALIAS or the Routines column, do not ignore:")
        foreach(x -> println("   ", x), unmatched)
    end
    return
end

main()
