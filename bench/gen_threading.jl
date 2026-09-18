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
const FLAT_BAND = 0.05
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

function box_section(io, path)
    kv = mt_header(path)
    rows, st = mt_rows(path)
    host = get(kv, "host", "?"); uarch = get(kv, "uarch", "?"); isa_ = get(kv, "isa", "?")
    println(io, "### ", host, " — ", uarch, ", ", isa_)
    println(io)
    println(io, "Measured ", get(kv, "time", "?"), " at commit `", get(kv, "commit", "?"),
        "`, ", get(kv, "cpu", "?"), ", pinned at ", mhz(get(kv, "freq", "?")), " with boost off.")
    println(io)
    @printf(io, "%d cells, %d off-lock, %d flat within %.0f%% of 1.00, %d with no `pb_mt` arm.\n",
        st.cells, st.offlock, st.flat, 100 * FLAT_BAND, st.no_mt)
    println(io)

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
        print(io, " ", get(kv, "host", "?"), " (", get(kv, "uarch", "?"), ") |")
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

- **Six threads on every box**, pinned one per *physical* core. Galen has twelve and is capped to six so
  the three boxes stay comparable. A second thread on a core shares the same FMA units, so pinning to
  logical cores would measure contention rather than parallelism — and the sibling numbering differs per
  box, which is a trap: `0,2,4,6,8,10` is six distinct cores on wintermute and only three on the others.
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

`gemm` has the only splitter. `symm` routes its materialised product through it. `getrf`, `geqrf` and
`trmm` inherit it through their trailing updates.

Flat at 1.00 **by design**, because no reference threads them either: `trsv`, `tbsv`, `tpsv`, `copy`,
`asum`, `iamax`, and gemm's k-loop. Their cells are a control — if one ever moves off 1.00, something
is wrong with the measurement rather than right with the library.

`trsm` is also flat, but for a different reason: it wraps its body in an arena scope, and a threaded
`gemm` refuses to run while any scope is live (see [the arena](arena.md)). That guard is what makes
per-thread workspaces safe, so the flatness is a deliberate trade, not an oversight.

The dual (`ForwardDiff.Dual`) groups carry no `pb_mt` arm at all: their reference is LinearAlgebra's
generic fallback, which replaces the arm list, and dual element types cannot reach the threaded path.

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
        print(io, "| ", get(kv, "host", "?"), " (", get(kv, "uarch", "?"), ") |")
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
