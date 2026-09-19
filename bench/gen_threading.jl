# Generate docs/src/threading.md from the mt caches. Same contract as `test/knob_registry.jl`: the
# generator emits the WHOLE file, so the page cannot drift from the measurements by hand-editing.
#
#   julia --project=bench bench/gen_threading.jl            # print to stdout
#   julia --project=bench bench/gen_threading.jl --write     # write docs/src/threading.md
#
# WHY THIS IS NOT PART OF publish.sh. `publish.sh` rebuilds artifacts that are a pure function of the
# GATE caches (`plots_data_*`) and verifies them byte-for-byte against a fresh rebuild. The mt caches
# are deliberately outside that glob — see `CACHE` in plots.jl — so no gate generator can see them, and
# wiring them in would defeat the separation that keeps a 6-core-affinity run from ever touching a
# 1-core-pinned gate cell. This generator is run on its own, by hand, after an mt sweep.

include(joinpath(@__DIR__, "mt_summary.jl"))     # cells / roundratios / med / mt_rows / mt_header

const DOC = joinpath(@__DIR__, "..", "docs", "src", "threading.md")

# ── WHAT THIS PAGE SHOWS, AND WHY IT IS A SUBSET ────────────────────────────────────────────────────
# Only operations that threading can actually reach. Two filters, both derived rather than chosen:
#
# 1. TYPE. `gemm!` refuses to thread unless `T === Float64 || T === Float32`, so every complex and dual
#    cell is provably unthreadable. Listing them would be 1038 rows of noise presented as a result.
#    They are not silently dropped — they are the NOISE SAMPLE that sets the threshold below.
#
# 2. MOVEMENT beyond that measured noise. Across the three boxes those 1038 unthreadable cells scatter
#    with |ratio − 1| of 0.1% median, 1.5% at p95, **3.7% at p99** and 12.2% at worst. So 3.7% is not a
#    taste literal: it is this harness's own p99 under a condition where the true answer is known to be
#    exactly 1.00. An op is listed when its best cell moves further than that on at least one box.
#    13 of 60 real-typed ops qualify; the other 47 are flat because nothing threads them.
const NOISE_P99 = 0.037        # measured, see above — regenerate this page after any harness change
const FLAT_BAND = NOISE_P99
const THREADABLE = ("L1", "L2", "L3", "LP")     # real element types only; see filter 1
_threadable(lvl) = lvl in THREADABLE
# Ops whose cells are all flat are not listed one by one — the page would be 900 rows of 1.00. They are
# summarised as a count, and the ones that are flat BY DESIGN are named in prose.
const CACHES = sort(filter(f -> startswith(basename(f), "mt_data_") && endswith(f, ".txt"),
    readdir(@__DIR__; join = true)))

fmt(x) = @sprintf("%.2f", x)          # "4.00", not "4.0" — a column of ratios must line up
# "L3 gemm" -> ("L3", " ", "gemm"), so the level stays plain and only the op name is code-formatted.
partition_op(k) = (first(split(k, " ")), " ", last(split(k, " ")))
# "3701000kHz" is what the header stores; a reader wants MHz.
function mhz(s)
    m = match(r"^(\d+)kHz$", String(s))
    return isnothing(m) ? String(s) : string(parse(Int, m.captures[1]) ÷ 1000, " MHz")
end

# A box is identified by its MICROARCHITECTURE, never its hostname. Hostnames are private to the fleet
# and say nothing a reader needs: the thing that explains a number is the µarch, ISA and clock. The
# cache header carries `host=`; this page deliberately does not print it.
boxlabel(kv) = string(get(kv, "uarch", "?"), " · ", get(kv, "isa", "?"))

function box_section(io, path)
    kv = mt_header(path)
    rows, st = mt_rows(path)
    println(io, "### ", boxlabel(kv))
    println(io)
    println(io, "Measured ", get(kv, "time", "?"), " at commit `", get(kv, "commit", "?"),
        "`, ", get(kv, "cpu", "?"), ", pinned at ", mhz(get(kv, "freq", "?")), " with boost off.")
    println(io)
    @printf(io, "%d cells measured, %d off-lock. Listed below: the threadable ops whose best cell moves further than the %.1f%% noise floor.\n",
        st.cells, st.offlock, 100 * NOISE_P99)
    println(io)

    rows = [r for r in rows if _threadable(r[3])]        # drop what cannot thread — see NOISE_P99
    movers = [r for r in rows if r[1] > 1 + FLAT_BAND]
    losers = [r for r in rows if r[1] < 1 - FLAT_BAND]

    if !isempty(movers)
        println(io, "**Where threading pays.** Best cell per operation:")
        println(io)
        println(io, "| op | best speedup | at n | round spread |")
        println(io, "|---|---|---|---|")
        best = Dict{String, Tuple{Float64, Int, Float64}}()
        for (s, sp, lvl, op, sz) in movers
            k = string(lvl, " `", op, "`")
            (!haskey(best, k) || s > best[k][1]) && (best[k] = (s, sz, sp))
        end
        for (k, v) in sort(collect(best); by = x -> -x[2][1])
            @printf(io, "| %s | **%s×** | %d | %.0f%% |\n", k, fmt(v[1]), v[2], 100 * v[3])
        end
        println(io)
    end

    if !isempty(losers)
        println(io, "**Where threading COSTS.** Every cell that got slower with six threads:")
        println(io)
        println(io, "| op | n | speedup | round spread |")
        println(io, "|---|---|---|---|")
        for (s, sp, lvl, op, sz) in sort(losers; by = first)
            @printf(io, "| %s `%s` | %d | **%s×** | %.0f%% |\n", lvl, op, sz, fmt(s), 100 * sp)
        end
        println(io)
    end
    return nothing
end

"""
    best_per_op(path) -> Dict("L3 gemm" => (speedup, n))

The best cell of each operation. A ladder that CLIMBS with n is the healthy shape; one that falls means
the routine is not amortising, which is the diagnosis worth surfacing.
"""
function best_per_op(path)
    rows, _ = mt_rows(path)
    best = Dict{String, Tuple{Float64, Int}}()
    for (s, _, lvl, op, sz) in rows
        _threadable(lvl) || continue
        k = string(lvl, " ", op)
        (!haskey(best, k) || s > best[k][1]) && (best[k] = (s, sz))
    end
    return best
end

# One table across every box, because the per-box sections cannot be read side by side and the
# cross-box comparison is the thing a reader actually wants: does this scale everywhere, or on one box?
function cross_box(io)
    boxes = [(mt_header(p), best_per_op(p)) for p in CACHES]
    ops = sort(collect(union((Set(keys(b)) for (_, b) in boxes)...)))
    # Only ops that move somewhere — a table of 1.00s is noise.
    ops = [o for o in ops if any(haskey(b, o) && b[o][1] > 1 + FLAT_BAND for (_, b) in boxes)]
    isempty(ops) && return nothing
    println(io, "Best speedup per operation, six threads against one, on each box.")
    println(io)
    print(io, "| op |")
    for (kv, _) in boxes
        print(io, " ", boxlabel(kv), " |")
    end
    println(io)
    print(io, "|---|")
    for _ in boxes
        print(io, "---|")
    end
    println(io)
    # Rank by the best figure anywhere, so the strongest wins lead.
    sort!(ops; by = o -> -maximum(haskey(b, o) ? b[o][1] : 0.0 for (_, b) in boxes))
    for o in ops
        lvl, _, opn = partition_op(o)
        print(io, "| ", lvl, " `", opn, "` |")
        for (_, b) in boxes
            haskey(b, o) ? @printf(io, " %s× @%d |", fmt(b[o][1]), b[o][2]) : print(io, " — |")
        end
        println(io)
    end
    println(io)
    return nothing
end

function page(io)
    println(io, """
# Multi-threading

PureBLAS threads `gemm` — one split, over the columns of C, across a parked worker pool. Routines that
call `gemm!` inherit it for free. Threading is **off** until you ask for it:

```julia
PureBLAS.set_num_threads(6)     # opt in; same shape as openblas_set_num_threads
PureBLAS.get_num_threads()
```

## These numbers are NOT the gate

The gate is `PB ≥ max(OpenBLAS, AOCL)`, **single-threaded**, and nothing on this page changes it. Every
figure here is a **scaling** measurement: the same PureBLAS, on the same box, in the same process, with
six threads instead of one.

A threaded PureBLAS compared against a single-threaded OpenBLAS would be flattery, not parity, so that
comparison is not made anywhere here. A real threaded gate needs freshly measured **threaded reference
arms**, which is six to eight hours per box and has not been done.

## How it is measured

- **Six threads on every box**, pinned one per *physical* core. The Zen3 part has twelve cores and is
  capped to six so the three boxes stay comparable, and its six are taken from a single L3 so it matches
  the single-CCX shape of the other two. A second thread on a core shares the same FMA units, so pinning
  to logical cores would measure contention rather than parallelism — and the sibling numbering differs
  between parts, which is a trap: `0,2,4,6,8,10` is six distinct cores on one of these boxes and only
  three on the others.
- **`pb` and `pb_mt` are measured in one process, in rotated rounds**, so a speedup divides two windows
  that saw the same machine state. It is a paired A/B, not two runs compared afterwards.
- **A separate cache.** The gate sweep is pinned to ONE core on purpose; the mt arm needs six. Since the
  two cannot share an invocation's affinity, an mt run writes `bench/mt_data_<uarch>_<host>.txt` and
  never touches the gate cache. The name sits outside the `plots_data_*` glob that the artifact
  generators and audits use, so these numbers cannot leak into a gate verdict.
- **Median of per-round medians**, the same estimator the gate uses. The round spread is reported
  because it is the precision of the thing doing the measuring.
- Frequency-locked with boost off, verified by achieved clock **under load** — not by reading sysfs,
  which on one box reported a perfect lock while the core ran at 4772 MHz against a 2000 MHz pin.

Reproduce with:

```bash
taskset -c 0,1,2,3,4,5 julia --project=bench -t 6 bench/plots.jl bench arms=pb,pb_mt nodraw
julia --project=bench bench/mt_summary.jl bench/mt_data_*.txt
```

## What is threaded, and what is deliberately not

**There is exactly ONE splitter, in `gemm!`.** No other routine contains threading code. `_symm!` is
the only other place that calls the threaded driver, and it does so by routing its already-materialised
product through that same splitter. Everything else that speeds up — `getrf`, `geqrf`, `trmm` — does so
because it *calls* `gemm!`; not a line was written for them.

That is why this page lists 13 operations and not 60. Of the real-typed ops measured, 47 do not move
beyond the noise floor, because nothing threads them.

Flat at 1.00 **by design**, because no reference threads them either: `trsv`, `tbsv`, `tpsv`, `copy`,
`asum`, `iamax`, and gemm's k-loop. Their cells are a control — if one ever moves off 1.00, something
is wrong with the measurement rather than right with the library.

`syrk`, `syr2k`, `hemm` and the rest of Level-3 inherit nothing for a structural reason worth knowing:
they reach `_gemm_core!` **directly**, which sits *below* the split point. Threading them is Phase 4 of
the plan and is not started.

`trsm` is also flat, but for a different reason: it wraps its body in an arena scope, and a threaded
`gemm` refuses to run while any scope is live (see [the arena](arena.md)). That guard is what makes
per-thread workspaces safe, so the flatness is a deliberate trade, not an oversight.

### Why `Dual` gets nothing — and why that is wiring, not a law

The dual groups carry no `pb_mt` arm at all, because their reference is LinearAlgebra's generic
fallback, which replaces the arm list. But the interesting part is what would happen if they did.

**A dual gemm is already three REAL `Float64` gemms.** `_gemm_dual3!` splits the operands into value
and partial planes and computes `P1 = Av·Bv`, `P2 = Av·Bp`, `P2 += Ap·Bv`, then combines. Those three
products are ordinary real gemms of the same shape as the original — exactly the kind of call that
threads well. So there is no type-level reason a `Dual` gemm cannot be threaded.

Two concrete things block it today, and neither is fundamental:

1. **It calls `_gemm_core!` directly**, which sits *below* the split point — the same structural reason
   `syrk` and `syr2k` inherit nothing. The guard in `gemm!` never even runs for these.
2. **The plane scratch is per-THREAD and held across all three products.** The buffers come from
   `L3Workspace`, whose owner is safe today only because it is claimed *inside* a chunk body, which
   never yields. A driver holding it across a threaded join is precisely the shape that made concurrent
   `getrf!` return wrong answers 20 times in 96 — it would need the same per-task conversion `symm`
   received.

There is also a parallel opportunity the column split does not reach: `P1` and the `P2` pair are
**independent products**, so they could run concurrently as whole gemms rather than being split
internally. That is task-level parallelism on the same three-plane structure.

None of this is scheduled. It belongs with Phase 4, and it is recorded here so the absence reads as a
decision rather than an oversight.

## Plots

One panel per operation, one curve per microarchitecture, against problem size. The dashed line is
1.00× — **no gain** — so a curve above it means threading paid and a curve below it means threading
cost. The band is the q10–q90 spread of the pooled per-round ratios.

Read the SHAPE, not just the peak. A curve that climbs with `n` is a routine amortising the fork-join
correctly; one that falls is a routine that is not. The flat lines sitting exactly on 1.00 — `syrk`,
`syr2k`, `trsm`, `trmmR`, every Level-1 and Level-2 panel — are the controls described above, and they
are supposed to be flat.

![BLAS-3 — PureBLAS 6 threads / 1 thread](assets/perf_mt_l3.svg)
![LAPACK — PureBLAS 6 threads / 1 thread](assets/perf_mt_lapack.svg)

Only Level-3 and LAPACK are plotted, because they are the only groups anything threads. Level-1 and
Level-2 have no splitter, and complex and dual cannot reach one — so their panels would be flat lines
by construction rather than by measurement.

Within these two panels the flat curves ARE informative, and they are kept for exactly that reason:
`syrk`, `syr2k`, `trsm`, `trmmR` and `potrf` sit on 1.00 next to `gemm` climbing to ~5×. They reach
`_gemm_core!` *below* the split point, or refuse to thread inside an arena scope. Seeing them flat in
the same picture is what shows the measurement discriminates rather than flattering everything.

Regenerate them with `julia --project=bench bench/plots.jl mtdraw`. That mode renders only
`perf_mt_*.svg` and exits before the gate rendering, so it cannot touch a gate artifact.

## Results
""")
    isempty(CACHES) && (println(io, "_No mt caches on disk._"); return)
    cross_box(io)
    for p in CACHES
        box_section(io, p)
    end
    println(io, "## Open: `gesvd` gets slower with threads")
    println(io)
    println(io, """
`gesvd` is the one routine that is **worse** with threads, and it reproduces on all three
microarchitectures — though not equally, which is itself a clue:
""")
    # Printed from the caches rather than asserted in prose: an earlier draft of this page claimed
    # "slower at every size below 2048", which the Zen5 sweep then contradicted at n=1000.
    println(io, "| box | n=256 | n=512 | n=1000 | n=1024 | n=2048 |")
    println(io, "|---|---|---|---|---|---|")
    for p in CACHES
        kv = mt_header(p); rows, _ = mt_rows(p)
        g = Dict(sz => s for (s, _, lvl, op, sz) in rows if op == "gesvd")
        print(io, "| ", boxlabel(kv), " |")
        for n in (256, 512, 1000, 1024, 2048)
            haskey(g, n) ? @printf(io, " %s× |", fmt(g[n])) : print(io, " — |")
        end
        println(io)
    end
    println(io, """

The root cause is **not yet known**. Three plausible explanations have been measured and rejected:
""")
    println(io, """
- **Not the fork-join price.** A gemm is only allowed to thread when it is at least 32× the measured
  join cost, so aggregate join overhead cannot exceed about 3%. The measured cost is ~1.29 ms per
  dispatch against a 616 ns join — roughly 2000×, and still ~300× a full sleep-wake.
- **Not the operand shape.** Skinny gemms were suspected, since every worker packs the whole of A. But
  a shape grid shows thin-`n`, thin-`k` and thin-`m` gemms mostly speeding up 1.9–3.7×.
- **Not the spin budget.** Setting the worker spin window to zero moved the bad band rather than
  removing it, and left n≥512 unchanged.

What is established: the threaded path genuinely runs (72 dispatches witnessed at n=512 via the pool's
generation counter, zero at one thread), and `gemm` itself is unstable under threading at n=256 while
being stable and fast at n≥512. Until this is understood, **do not enable threading for a workload
dominated by SVD at these sizes**.
""")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--write" in ARGS
        open(io -> page(io), DOC, "w")
        println("wrote ", DOC)
    else
        page(stdout)
    end
end
