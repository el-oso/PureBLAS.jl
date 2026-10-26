# Route-token lint — a size-keyed predicate that reads its OWN extent breaks req#11 silently.
#
# WHY THIS FILE EXISTS. A threaded caller partitions one dimension and passes the UNDIVIDED extents in
# a route token, so every size-keyed branch picks the arithmetic the unsplit problem picks. A predicate
# that reads the chunk's own extent instead routes a band one way and the whole problem the other, and
# the two arms do not agree bit for bit — so the answer depends on `set_num_threads(n)`. Nothing in the
# suite catches it: the arms are each individually correct, the divergence is in the last bit, and it
# only appears at worker counts a test has to reach deliberately.
#
# THREE INSTANCES, each found by an adversarial review rather than by a test:
#   * `_gemm_unpacked!`'s `m >= _SMR * _vwidth(T)` floor read the band's own rows while the cap beside
#     it read the token — 18 divergences in the side-R trsm gate, fixed by `097e9882`.
#   * `_trsm_right!`'s `sme_recur` read `size(B, 1)` twice with `mrt` in scope and used one line below.
#   * `_gemm_core_body!`'s `_sme_eligible(T, m, nrt, …)` reads the local row extent, which made
#     `potrf!` lower non-reproducible above eight workers.
#
# THE RULE THAT MAKES THIS CHECKABLE, and it is sharper than "read the token":
#
#     ONE TOKEN IN THE n SLOT REPAIRS AN UNLIMITED NUMBER OF `max` PREDICATES AND CANNOT REPAIR ONE
#     THAT IS NOT A `max`.
#
# `_trsmr_run_chunk` feeds the whole problem's row count into the n slot precisely so that every
# `max(m, nrt, k) <= C` test reads the serial value even though `m` is the band height — `max` absorbs
# it. So those predicates are safe without being touched. A predicate built from a `min`, a modulus or
# a bare one-sided bound gets no such protection: `_sme_tile_ok` is `min(m,n) >= 2*_SME_MR` plus two
# moduli, and it is the only route predicate in the tree that the `max` trick does not cover.
#
# So the lint has two halves. Find the size-keyed predicates in a function that HAS a route token, then
# among them flag the ones that are not a `max` over the resolved values. A `max` form is reported only
# if it reads a raw extent where a resolved one exists.
#
# Mark a deliberate local read with `# route-ok: <reason>` on the same line or the line above. There is
# one legitimate reason and it should be stated: a FLOOR on the tile actually handed to a kernel must
# read the local extent, because routing it from the undivided shape would admit a chunk below the
# kernel's minimum and trade a reproducibility bug for a correctness one.
const _RT_SRC = normpath(joinpath(@__DIR__, "..", "src"))
const _RT_BASELINE = joinpath(@__DIR__, "routetoken_lint_baseline.txt")

# A function that has a route token available: it either takes the triple, takes the legacy scalar, or
# resolves one into `mrt`/`nrt`/`krt`.
const _RT_HAS_TOKEN = r"(rt::NTuple\{3,\s*Int\}|nroute::Int|\b(mrt|nrt|krt)\s*=)"
# The resolved names, plus the raw extents a predicate must not prefer over them.
#
# `k` IS DELIBERATELY NOT A RAW EXTENT HERE. req#11 forbids ever letting a worker count reach `k` — the
# reduction chain must stay whole on one worker — so a predicate on `k` alone is worker-invariant by
# construction and needs no token. Only the two dimensions a caller may divide can carry the defect:
# `m` (side R, `_trsm_rchunk`) and `n` (side L and the gemm column split).
const _RT_RESOLVED = r"\b(mrt|nrt|krt)\b"
const _RT_RAW = r"(\bsize\(\s*[A-Za-z_][A-Za-z0-9_]*\s*,\s*[12]\s*\)|\bm\b|\bn\b)"
# A size-keyed predicate: compared against a tuning const or handed to a tile predicate.
const _RT_PRED = r"(<=|>=|<|>)\s*_[A-Za-z_]|_sme_tile_ok\(|_sme_eligible\(|_sme_owns\("
# Forms the n-slot token cannot repair.
const _RT_UNPROTECTED = r"(\bmin\(|%\s*_|\brem\(|_sme_tile_ok\(|_sme_eligible\(|_sme_owns\()"
# A ONE-SIDED BOUND ON A RAW EXTENT, and this one must fire even when resolved names appear elsewhere
# on the same line. The most dangerous shape in this whole class is the HALF-PROTECTED predicate: the
# original defect read `m >= _SMR * _vwidth(T) && max(mrt, nrt, krt) <= _gemm_split_max()`, where the
# cap is routed and the floor is not. A check that suppressed itself whenever `mrt` appeared anywhere
# would have passed that line — it did, until a positive control reverted the fix and the lint stayed
# silent. `max(m, …)` and `min(m, …)` are excluded here because a raw extent INSIDE a reduction is
# handled by the two tests above.
const _RT_ONESIDED = r"(?<![a-z_(])\b([mn])\b\s*(>=|<=|>|<)\s*_[A-Za-z_]|_[A-Za-z_][A-Za-z0-9_()]*\s*(>=|<=|>|<)\s*\b([mn])\b(?![a-z_])"
const _RT_OK = r"#\s*route-ok:"

"""Scan `src/` for size-keyed predicates inside token-bearing functions. Returns `file:line  kind  text`."""
function routetoken_scan()
    hits = String[]
    for (root, _, files) in walkdir(_RT_SRC), f in files
        endswith(f, ".jl") || continue
        path = joinpath(root, f)
        rel = relpath(path, _RT_SRC)
        lines = readlines(path)
        intoken = false
        depth = 0
        for (i, ln) in enumerate(lines)
            # Crude function tracking: a `function` at column 0 opens, `end` at column 0 closes. Good
            # enough because every token-bearing routine here is a top-level function.
            if occursin(r"^(function|@inline function|@noinline function)\b", ln)
                depth = 1
                intoken = occursin(_RT_HAS_TOKEN, ln)
            elseif depth == 1 && occursin(r"^end\b", ln)
                depth = 0; intoken = false
            end
            intoken || (intoken = depth == 1 && occursin(_RT_HAS_TOKEN, ln))
            intoken && depth == 1 || continue
            # A comment line states a predicate, it does not evaluate one. Prose about `m >= _SMR·W`
            # is exactly the text this lint exists to explain, so matching it would make the lint
            # fire on its own documentation.
            startswith(lstrip(ln), "#") && continue
            occursin(_RT_PRED, ln) || continue
            # An explicit marker, on this line or the one above, settles it.
            (occursin(_RT_OK, ln) || (i > 1 && occursin(_RT_OK, lines[i - 1]))) && continue
            onesided = occursin(_RT_ONESIDED, ln)
            unprot = occursin(_RT_UNPROTECTED, ln)
            raw = occursin(_RT_RAW, ln) && !occursin(_RT_RESOLVED, ln)
            (onesided || unprot || raw) || continue
            kind = onesided ? "half-protected" : unprot ? "not-a-max" : "raw-extent"
            push!(hits, "$rel:$i  $kind  $(strip(ln))")
        end
    end
    return sort(hits)
end

_rt_baseline() = isfile(_RT_BASELINE) ?
    Set(filter(l -> !isempty(l) && !startswith(l, "#"), strip.(readlines(_RT_BASELINE)))) : Set{String}()

"""Keyed by `file:kind:text` so a line-number shift does not read as a new defect."""
_rt_key(h) = let p = split(h, "  "; limit = 3)
    string(split(p[1], ":")[1], ":", p[2], ":", length(p) >= 3 ? p[3] : "")
end

function routetoken_lint()
    hits = routetoken_scan()
    base = _rt_baseline()
    new = filter(h -> !(_rt_key(h) in base), hits)
    live = Set(_rt_key.(hits))
    stale = filter(b -> !(b in live), collect(base))
    return (; hits, new, stale)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--baseline" in ARGS
        foreach(println, routetoken_scan())
    else
        r = routetoken_lint()
        if isempty(r.new) && isempty(r.stale)
            println("route-token lint: PASS ($(length(r.hits)) size-keyed predicates reviewed)")
        else
            println("route-token lint: FAIL")
            foreach(x -> println("  NEW   ", x,
                "\n          → read the token (`_rt_m`/`_rt_n`/`_rt_k`), or baseline it with the reason, ",
                "or mark `# route-ok:` if it is a FLOOR on the tile actually handed to a kernel"), r.new)
            foreach(x -> println("  STALE ", x, "  — the site moved or was fixed; update the baseline"), r.stale)
            exit(1)
        end
    end
end
