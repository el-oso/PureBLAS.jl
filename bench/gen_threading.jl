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

PureBLAS threads most of BLAS-1 and BLAS-2 and the bulk of BLAS-3: `axpy`, `scal`, `blascopy`,
`swap`, `dot`, `asum`, `nrm2`, `iamax`; `ger`, `gemv` both ways, `symv`, `gbmv` non-transposed;
`gemm`, `symm`, `syrk`, `syr2k`, both sides of `trsm`, `trmm` side R; and — through those — `getrf`
and `potrf`. Threading is **off** until you ask for it:


```julia
PureBLAS.set_num_threads(6)     # opt in; same shape as openblas_set_num_threads
PureBLAS.get_num_threads()
```

## What these numbers are

A **scaling** measurement: the same PureBLAS, on the same box, in the same process, with six threads
instead of one. The gate is a separate, single-threaded criterion and this page does not speak to it.

A threaded gate would need freshly measured threaded reference arms, six to eight hours per box.

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
# The mask is one CPU per physical core PLUS one spare for the runtime threads, and CPU numbering
# differs per box — `bench/plots.jl`'s `_ARM_PB_MT` comment carries the mask for each.
taskset -c 0,2,4,6,8,10,1 julia --project=bench -t 6 bench/plots.jl bench arms=pb,pb_mt nodraw
julia --project=bench bench/mt_summary.jl bench/mt_data_*.txt
```

## What is threaded

### BLAS-3 and LAPACK

`gemm` splits the columns of C. `syrk`, `syr2k` and `symm`
reach the kernel below that split point, so they carry their own kind: a triangular output needs a
flop-balanced column split, since equal widths hand the first worker roughly twice the work of the
last. `trsm` splits the columns of B on side L and its rows on side R — every column of one and
every row of the other is an independent solve, so the bands are write-disjoint and need no barrier.
`getrf` adds one more: it factors the next panel on one worker while the rest apply the current
panel's update.

`getrf` and `potrf` reach the pool through those, and `potrf` reaches it at every size — its leaf
routes its trailing update through the public `syrk!` rather than a private kernel.

**A band must route from the unsplit problem's dimensions.** Several kernel choices key on
`max(m, n, k)`, so a band whose width lands on one of those constants would take a different kernel
from the call it belongs to. That is not a slower answer, it is a different one, and it is why the
route token reaches the kernel switches themselves — `getrf` lost its thread-count invariance until
it did.

### Reductions: a fixed block grid

`dot`, `asum` and `nrm2` cut `n` into blocks whose size is a function of `(n, T)` alone, reduce each
with the unchanged serial kernel, and have the **driver** fold the partials in index order after the
join. A worker count therefore decides who computes a partial and never how the partials combine,
which is what makes the answer identical at every thread count. The blocked form is the only form,
used at one thread too.

`iamax` rides the same grid with a different fold — a strict `>` scan in block order, so ties go to
the lower index — and every block is seeded from the same `|x[1]|`, which is what preserves netlib's
NaN contract under a split.

### BLAS-2: the partition follows how far the scatter reaches

Three shapes, and which one applies is decided by the kernel, not by taste.

**No scatter.** `ger` writes each `A[i,j]` once, and `gemv` gives each output element its own
accumulation chain. Bands are write-disjoint already, so only the *route* needs guarding: `gemv-N`
splits rows and `gemv-T` columns, and both carry the undivided dimension because two of their kernel
choices read the A byte count — one of them selects a panel width, which re-chunks the chain and so
changes the arithmetic rather than just the speed.

**Bounded scatter.** `gbmv` non-transposed writes only within `kl + ku` of its own rows, so output
row bands are write-disjoint and bit-identical for free. Nothing is folded and no token is needed —
the arm choice reads `kl + ku + 1`, which a row split cannot move.

**Full scatter.** `symv` reads each stored element once and uses it twice, so a column block writes
`y` rows outside its own range. Workers need private `y` vectors folded in block order. That fold is
the interesting part: a private vector starts at zero, so the folded form rounds differently from an
unblocked sweep, and since a caller that loses the pool claim runs serially, the blocked form has to
be the **only** form — at one thread as well. The fold is therefore a cost on the *serial* path, and
`symv`'s serial margin at n=4096 is 0.995 on one box and 1.000 on another. So the grid is cut by
residency: past roughly twice L3 it collapses to one block, `y` is written directly, and not one
private vector is touched. Below the pool's own admission floor it collapses for the same reason —
there a grid would be pure fold with no worker to pay for it.

`trmv`, `trsv` and the packed and banded triangular ops are **not** threaded, and that is a decision
rather than an omission: a triangular sweep and a substitution are sequential. They also reach the
gemv panel drivers directly rather than through the public entry, which is what keeps their
per-thread scratch un-forked.

## Plots

One panel per operation, one curve per microarchitecture, against problem size. The dashed line is
1.00×, the band is the q10–q90 spread of the pooled per-round ratios.

### The gate, with threads on

This is the criterion of req#1 measured threaded: **PureBLAS at N threads over whichever THREADED
vendor is faster in that cell**, `max(OpenBLAS, AOCL)`, chosen per size — so one curve may switch
references along its own x-axis, exactly as the gate does. Above 1.00× is a pass.

Every group appears here, including the ones that do not thread at all. That is deliberate: a group
with no splitter still has to be measured against a vendor that has one, and that gap IS the finding
for BLAS-2 and for the complex groups.

A cell with no threaded reference arm is DROPPED rather than compared against a serial one — a
threaded PureBLAS arm over a single-thread vendor is not the gate and must not be drawn as if it were.

!!! warning "The Zen4 curve is not yet adjudicable"
    That box is a laptop mainboard with no battery, and it was running on a supply that could not hold
    its pin once every core was busy: one core held 2795 MHz indefinitely while six oscillated
    2332–2804 MHz, at 52–58 °C, far below any thermal limit. 1208 of its cached arms are stamped
    below the pin and 413 cells compare two power states rather than two libraries. The error runs in
    PureBLAS's favour, because a throttled reference is a slower reference. A larger supply has since
    moved its sustainable pin to 3501 MHz and `fleet_freqlock.sh verify-mt` confirms six cores hold
    it, so those cells are being re-measured with both arms in one machine state; until then read the
    Zen3 and Zen5 curves, whose cross-arm clocks agree on 929 of 937 and 951 of 951 cells.

![BLAS-1 — PureBLAS threaded / faster of threaded OpenBLAS and AOCL](assets/perf_mtgate_l1.svg)
![BLAS-2 — PureBLAS threaded / faster of threaded OpenBLAS and AOCL](assets/perf_mtgate_l2.svg)
![BLAS-3 — PureBLAS threaded / faster of threaded OpenBLAS and AOCL](assets/perf_mtgate_l3.svg)
![LAPACK — PureBLAS threaded / faster of threaded OpenBLAS and AOCL](assets/perf_mtgate_lapack.svg)
![complex BLAS-1 — PureBLAS threaded / faster of threaded OpenBLAS and AOCL](assets/perf_mtgate_cl1.svg)
![complex BLAS-2 — PureBLAS threaded / faster of threaded OpenBLAS and AOCL](assets/perf_mtgate_cl2.svg)
![complex BLAS-3 — PureBLAS threaded / faster of threaded OpenBLAS and AOCL](assets/perf_mtgate_cl3.svg)
![complex LAPACK — PureBLAS threaded / faster of threaded OpenBLAS and AOCL](assets/perf_mtgate_clapack.svg)

### What threading bought

The panels below ask a different question and are NOT the gate: both arms are PureBLAS, so they show
N threads over 1 thread — what the pool won, with no vendor in it. A panel appears when its routine
reaches at least 1.25× somewhere on the fleet, which is well clear of the harness's own 3.7% noise
floor; a routine with no splitter would draw a flat line at 1.00× and reports the harness rather than
the library.

Read the SHAPE, not just the peak: a curve that climbs with `n` is a routine amortising the
fork-join correctly.

![BLAS-1 — PureBLAS 6 threads / 1 thread](assets/perf_mt_l1.svg)
![BLAS-3 — PureBLAS 6 threads / 1 thread](assets/perf_mt_l3.svg)
![LAPACK — PureBLAS 6 threads / 1 thread](assets/perf_mt_lapack.svg)

Regenerate both sets with `julia --project=bench bench/plots.jl mtdraw`, which writes
`perf_mtgate_*.svg` and `perf_mt_*.svg` and exits before the gate rendering.

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
