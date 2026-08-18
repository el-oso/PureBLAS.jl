# Emit the per-FUNCTION coverage tables of docs/src/coverage.md straight from the v3 caches.
#
# WHY PER FUNCTION. The doc used to carry four lumped rows (BLAS-1 / BLAS-2 dense / BLAS-2 banded-packed
# / BLAS-3) with one verdict each. A single 🐢 over ~40 routines hides which ones actually miss: it made
# `gbmv` (1.9× and gating) and `ztrsm` (0.65) indistinguishable, and it made progress invisible — closing
# three of four cells in a group did not change the row. One line per routine, one verdict per routine.
#
# THE VERDICT IS COMPUTED, NEVER JUDGED. `gate` = min over that routine's cells of the ratio against the
# FASTER reference at each cell — i.e. exactly the project rule PB ≥ max(OpenBLAS, AOCL), evaluated per
# cell and then worst-cased. 🐰 iff gate ≥ 1.0. Both arms come from the same round of the same run (v3),
# so the max is over numbers that saw one machine state.
#
# Usage: julia --project=bench bench/coverage_ops.jl bench/plots_data_<uarch>_<host>.txt [more…] > out.md
const QN = 48
med(v) = (s = sort(v); s[max(1, cld(length(s), 2))])
include(joinpath(@__DIR__, "freqgate.jl"))     # _freq_ref / _freq_of / _freq_offlock — see that file

# OFF-LOCK CELLS ARE EXCLUDED AND REPORTED. A cell measured while the frequency lock was floating is not
# adjudicable, and this table's verdict is a MIN over cells — so one drifted cell can set a whole
# routine's published number. The report goes to STDERR because stdout is spliced verbatim into
# docs/src/coverage.md; a partially excluded run must be visible to whoever runs the splice.
const OFFLOCK = []                             # (cache, level, op, uarch, size, [arm => kHz])
const FREQMETA = []                            # (cache, reference kHz, was-backfilled)
const EXCLUDED = Dict{Tuple{String, String, String}, Int}()   # (level, op, uarch) => cells dropped

# op => (level, types) — types are what the bench row actually exercises, not what the routine supports.
const LEVEL = Dict{String, String}()
lvlof(l) = l in ("L1", "CL1") ? "BLAS-1" : l in ("L2", "CL2") ? "BLAS-2" : l in ("L3", "CL3") ? "BLAS-3" : "LAPACK"

# ONE COLUMN PER MICROARCHITECTURE. A single pooled verdict answers "does this routine gate SOMEWHERE",
# which is the wrong question — the gate is per box. Pooling also makes progress invisible in exactly the
# way the four lumped rows did: on 2026-08-05 `iamax` and `dzasum` went green on Zen4 and the pooled rows
# stayed 🐢, with nothing to say which box still missed or at what size. Per box, the next machine to fix
# is readable straight off the row.
cells = Dict{Tuple{String, String, String}, Vector{Tuple{Int, Float64}}}()  # (level, op, uarch) => [(size, gate)]
const UARCH = String[]                                    # column order = the order caches were given
for path in ARGS
    ua = "?"
    fref = 0
    for ln in eachline(path)
        if startswith(ln, "#pbbench")
            mu = match(r"uarch=(\S+)", ln); mi = match(r"isa=(\S+)", ln)
            ua = string(isnothing(mu) ? "?" : mu[1], isnothing(mi) ? "" : " · " * mi[1])
            ua in UARCH || push!(UARCH, ua)
            fref, bf = _freq_ref(ln)
            push!(FREQMETA, (basename(path), fref, bf))
            continue
        end
        (isempty(strip(ln)) || startswith(ln, "#")) && continue
        p = split(ln, "\t"); length(p) >= 4 || continue
        lvl, op, sz = String(p[1]), String(p[2]), parse(Int, p[3])
        d = Dict{String, Vector{Float64}}()
        drift = []
        for f in p[4:end]
            _p = split(f, "|"); a, csv = _p[1], _p[end]
            _freq_offlock(_freq_of(_p), fref) && push!(drift, String(a) => _freq_of(_p))
            d[String(a)] = parse.(Float64, split(csv, ","))
        end
        haskey(d, "pb") || continue
        if !isempty(drift)                     # any arm off-lock ⇒ the RATIO is not adjudicable
            push!(OFFLOCK, (basename(path), lvlof(lvl), op, ua, sz, drift))
            EXCLUDED[(lvlof(lvl), op, ua)] = get(EXCLUDED, (lvlof(lvl), op, ua), 0) + 1
            continue
        end
        # THE RATIO IS FORMED EXACTLY AS plots.jl DOES IT: elementwise over the sorted sample vectors
        # (`_ratio` = qref ./ qpb), then ONE median over all of them. This used to take the median of
        # per-round medians instead, which is a different statistic and DISAGREES AT MARGINAL CELLS —
        # measured 2026-08-05: galen iamax n=1e6 read 1.0077 vs the gate's 1.0085, and Zen5 iamax
        # n=1e3 flipped verdict outright (round-pooled <1.0, gate 1.007). A published coverage table
        # that can say 🐢 where the gate says PASS is worse than no table.
        rs = Float64[]
        for r in ("openblas", "aocl")
            haskey(d, r) || continue
            m = min(length(d[r]), length(d["pb"]))
            m == 0 && continue
            push!(rs, med([d[r][i] / d["pb"][i] for i in 1:m]))
        end
        isempty(rs) && continue
        push!(get!(cells, (lvlof(lvl), op, ua), Tuple{Int, Float64}[]), (sz, minimum(rs)))  # vs the FASTER ref
    end
end

# RENDERING: colour bands, not 🐰/🐢. A binary glyph makes `trsm` 0.999 and `zpotrfU` 0.61 look
# identical, which is exactly backwards — the whole point of a per-routine table is to show WHERE the
# work is. Banding the shortfall (≥1.0 / ≥0.99 / ≥0.95 / ≥0.85 / below) sorts the page by severity at
# a glance. The ratio stays in the cell and is the authoritative signal: colour is a redundant
# encoding, never the only one, so the table still reads correctly in monochrome or to a colourblind
# reader. Emitted as `@raw html` because Documenter markdown tables cannot carry per-cell classes.
band(g) = g >= 1.0 ? "ok" : g >= 0.99 ? "b1" : g >= 0.95 ? "b2" : g >= 0.85 ? "b3" : "b4"

# A FAILING gate is printed at 3 digits and FLOORED; a passing one at 2, rounded. Two reasons, both
# about not lying with a rounded number:
#   * flooring can never round a miss up to "1.0" (0.9996 prints 0.999, not 1.0), so a cell that reads
#     like parity always IS parity;
#   * 3 digits keeps the printed value inside its own colour band, so two cells showing the same
#     number always get the same colour — at 2 digits, 0.9887 and 0.9938 both printed "0.99" and were
#     banded differently, which looks like a rendering bug.
# The band is then computed from the DISPLAYED value, so colour and number cannot disagree.
function gatestr(g)
    s = g >= 1.0 ? string(round(g; digits = 2)) : string(floor(g * 1000) / 1000)
    return s, band(parse(Float64, s))
end

# One stylesheet for every table below. Vitepress toggles `.dark` on <html>, so that is the hook;
# `prefers-color-scheme` is kept as a fallback for a bare Documenter build.
println("""
```@raw html
<style>
.pbg{--l:#e3e6ee;--m:#5d6675;--ok:#1f8a5b;--b1:#7a8496;--b2:#c07d12;--b3:#cf5a35;--b4:#b3243a;
 --okbg:#e9f6ef;--b1bg:#f1f3f7;--b2bg:#fdf3e2;--b3bg:#fceee9;--b4bg:#fbe9ed;
 border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;display:table}
html.dark .pbg{--l:#242c3b;--m:#98a1b3;--ok:#4cc98d;--b1:#8891a3;--b2:#e0a63c;--b3:#f08055;--b4:#ff5f7a;
 --okbg:#12271d;--b1bg:#1a2130;--b2bg:#2a2113;--b3bg:#2c1a15;--b4bg:#2c1420}
@media (prefers-color-scheme:dark){html:not(.light) .pbg{--l:#242c3b;--m:#98a1b3;--ok:#4cc98d;--b1:#8891a3;
 --b2:#e0a63c;--b3:#f08055;--b4:#ff5f7a;--okbg:#12271d;--b1bg:#1a2130;--b2bg:#2a2113;--b3bg:#2c1a15;--b4bg:#2c1420}}
.pbg th,.pbg td{border-bottom:1px solid var(--l);padding:7px 12px;text-align:left}
.pbg thead th{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--m);white-space:nowrap}
.pbg tbody th{font-weight:400}
.pbg td{border-left:1px solid var(--l);white-space:nowrap}
.pbg .v{font-weight:600}
.pbg .n{font-size:.82em;color:var(--m);margin-left:7px}
.pbg td.ok{background:var(--okbg)} .pbg td.ok .v{color:var(--ok)}
.pbg td.b1{background:var(--b1bg)} .pbg td.b1 .v{color:var(--b1)}
.pbg td.b2{background:var(--b2bg)} .pbg td.b2 .v{color:var(--b2)}
.pbg td.b3{background:var(--b3bg)} .pbg td.b3 .v{color:var(--b3)}
.pbg td.b4{background:var(--b4bg)} .pbg td.b4 .v{color:var(--b4)}
.pbg-key{display:flex;flex-wrap:wrap;gap:14px;margin:10px 0 0;font-size:12px;color:#5d6675}
html.dark .pbg-key{color:#98a1b3}
.pbg-key i{font-style:normal;display:inline-block;width:11px;height:11px;border-radius:2px;
 margin-right:5px;vertical-align:-1px}
</style>
```
""")

for section in ("BLAS-1", "BLAS-2", "BLAS-3", "LAPACK")
    # Ops are drawn from EXCLUDED as well as `cells`: a routine whose every cell was off-lock must still
    # get a row, saying so. Dropping the row would render an unmeasurable routine as "not benchmarked".
    ops = sort(unique(k[2] for k in Iterators.flatten((keys(cells), keys(EXCLUDED))) if k[1] == section))
    isempty(ops) && continue
    println("\n#### $section\n")
    println("```@raw html")
    println("<table class=\"pbg\"><thead><tr><th>routine</th>",
            join(("<th>$ua</th>" for ua in UARCH)), "</tr></thead><tbody>")
    for op in ops
        print("<tr><th><code>$op</code></th>")
        for ua in UARCH
            cs = sort(get(cells, (section, op, ua), Tuple{Int, Float64}[]); by = first)
            nx = get(EXCLUDED, (section, op, ua), 0)    # cells dropped as off-lock
            if isempty(cs)
                # "—" means NOT MEASURED. If cells exist but every one was off-lock, say that instead:
                # the two are opposite situations and must not print the same glyph.
                print(nx == 0 ? "<td><span class=\"n\">—</span></td>" :
                    "<td><span class=\"n\">off-lock ($nx)</span></td>")
                continue
            end
            gs = [c[2] for c in cs]
            gate = minimum(gs)                          # the gate IS the worst cell, per box
            # A passing row needs no size — it gates everywhere. A failing one names the cell to fix,
            # which is the actionable unit; the ratio alone would say nothing about where to look.
            sz = gate >= 1.0 ? "" : "<span class=\"n\">n=$(cs[argmin(gs)][1])</span>"
            # A partially excluded routine is a PARTIAL verdict — the min is over fewer cells than the
            # sweep measured, and the missing ones could be the worst. Say so in the cell itself.
            nx > 0 && (sz *= "<span class=\"n\">−$nx off-lock</span>")
            gstr, cls = gatestr(gate)
            print("<td class=\"$cls\"><span class=\"v\">$gstr</span>$sz</td>")
        end
        println("</tr>")
    end
    println("</tbody></table>")
    println("```")
end

println("""
```@raw html
<p class="pbg-key">
 <span><i style="background:#1f8a5b"></i>gates (≥ 1.0)</span>
 <span><i style="background:#7a8496"></i>≥ 0.99</span>
 <span><i style="background:#c07d12"></i>≥ 0.95</span>
 <span><i style="background:#cf5a35"></i>≥ 0.85</span>
 <span><i style="background:#b3243a"></i>below 0.85</span>
</p>
```
""")

# ── FREQUENCY REPORT → STDERR (stdout is the doc page) ────────────────────────────────────────────
for (f, ref, bf) in FREQMETA
    println(stderr, "freq ref  ", rpad(replace(f, "plots_data_" => "", ".txt" => ""), 24),
        ref == 0 ? "(none in header — no cell can be judged off-lock)" :
        string(ref, "kHz", bf ? "  BACKFILLED from the run header — the per-cell clocks in this cache " *
            "are NOT measured samples and cannot flag anything" : "  (header achieved clock; cells >1% above it are excluded)"))
end
if !isempty(OFFLOCK)
    println(stderr, "\nOFF-LOCK: $(length(OFFLOCK)) cell(s) EXCLUDED from the table above — measured >1% \
    above the cache\nheader's achieved clock, i.e. while the frequency lock was floating. NOT ADJUDICABLE. \
    Any routine\nrow marked \"off-lock\" is a partial verdict; re-measure those cells (`op=`/`group=`) and \
    re-splice.")
    for (f, sect, op, ua, sz, drift) in OFFLOCK
        println(stderr, "  ", rpad("$sect $op@$sz", 30), rpad(ua, 18),
            rpad(replace(f, "plots_data_" => "", ".txt" => ""), 24),
            join(("$a=$(k)kHz" for (a, k) in drift), " "))
    end
end
