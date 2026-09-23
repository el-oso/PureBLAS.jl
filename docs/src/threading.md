# Multi-threading

PureBLAS threads `gemm`, `symm`, `syrk`, `syr2k`, both sides of `trsm`, and — through them —
`getrf` and `potrf`. Threading is **off** until you ask for it:


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
# The mask is one CPU per physical core PLUS one spare for the runtime threads, and CPU numbering
# differs per box — `bench/plots.jl`'s `_ARM_PB_MT` comment carries the mask for each.
taskset -c 0,2,4,6,8,10,1 julia --project=bench -t 6 bench/plots.jl bench arms=pb,pb_mt nodraw
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

`syrk`, `syr2k` and `symm` reach `_gemm_core!` **directly**, below the public split point, so they
carry their own job kinds in the pool rather than inheriting gemm's: a triangular output needs a
flop-balanced column split, since equal widths hand the first worker roughly twice the work of the
last.

`trsm` threads on both sides — columns of B for side L, rows for side R — and each band is a whole
problem, so the bands need no barrier between them. What a band DOES need is to route from the
unsplit problem's dimensions: several kernel choices key on `max(m, n, k)`, and a band that lands on
one of those constants takes a different kernel from the call it is part of. That is not a slower
answer, it is a different one, and it broke `getrf`'s thread-count invariance until the route token
reached the kernel switches themselves.

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
correctly; one that falls is a routine that is not. The flat lines sitting exactly on 1.00 — every
Level-1 and Level-2 panel, and `trmmR` — are the controls described above, and they
are supposed to be flat.

![BLAS-3 — PureBLAS 6 threads / 1 thread](assets/perf_mt_l3.svg)
![LAPACK — PureBLAS 6 threads / 1 thread](assets/perf_mt_lapack.svg)

Only Level-3 and LAPACK are plotted, because they are the only groups anything threads. Level-1 and
Level-2 have no splitter, and complex and dual cannot reach one — so their panels would be flat lines
by construction rather than by measurement.

Within these two panels the flat curves ARE informative, and they are kept for exactly that reason:
`trmmR` and the Level-1/Level-2 panels sit on 1.00 next to `gemm` and `trsm` climbing past 4.9×.
They have no splitter at all, or no reference threads them either. Seeing them flat in
the same picture is what shows the measurement discriminates rather than flattering everything.

Regenerate them with `julia --project=bench bench/plots.jl mtdraw`. That mode renders only
`perf_mt_*.svg` and exits before the gate rendering, so it cannot touch a gate artifact.

## Results

Best speedup per operation, six threads against one, on each box.

| op | Zen3 · AVX2 | Zen4 · AVX-512 | Zen5 · AVX-512 |
|---|---|---|---|
| L3 `trsm` | 5.64× @4096 | 4.90× @2100 | 5.88× @2100 |
| L3 `trsmR` | 5.18× @4096 | 4.61× @4096 | 5.57× @4096 |
| L3 `syrk` | 5.45× @4096 | 4.80× @4096 | 5.30× @2100 |
| L3 `syr2k` | 2.87× @4096 | 4.29× @4096 | 5.25× @4096 |
| LP `potrfU` | 5.17× @4096 | 4.30× @4096 | 5.24× @4096 |
| L3 `gemm` | 4.75× @2100 | 4.01× @2100 | 5.06× @2100 |
| LP `potrf` | 4.53× @4096 | 3.94× @4096 | 4.89× @4096 |
| L3 `symm` | 4.50× @2100 | 3.74× @2100 | 4.83× @2100 |
| LP `getrf` | 3.89× @2100 | 3.45× @4096 | 4.25× @4096 |
| LP `pptrfL` | 2.54× @2048 | 3.20× @2048 | 4.09× @2048 |
| LP `pptrfU` | 2.52× @2048 | 3.10× @2048 | 4.00× @2048 |
| LP `geqrf` | 3.52× @4096 | 1.57× @4096 | 1.73× @512 |
| LP `getri` | 2.44× @2048 | 1.98× @2048 | 2.60× @2048 |
| LP `pstrf` | 2.16× @2100 | 1.76× @2100 | 2.15× @2100 |
| LP `pstrfU` | 1.85× @2100 | 1.62× @2100 | 2.12× @2100 |
| LP `syev` | 1.67× @2048 | 1.74× @2048 | 2.07× @2048 |
| LP `pbtrfL` | 1.39× @384 | 1.55× @384 | 2.00× @384 |
| LP `gesvd` | 1.81× @2048 | 1.54× @2048 | 1.95× @2048 |
| LP `pbtrfU` | 1.00× @32 | 1.44× @384 | 1.82× @384 |
| LP `gels` | 1.39× @1024 | 1.39× @512 | 1.67× @512 |
| LP `getrs` | 1.01× @256 | 1.00× @50 | 1.58× @1024 |
| LP `potrsU` | 1.40× @1000 | 1.00× @50 | 1.00× @50 |
| LP `potrsL` | 1.40× @1000 | 1.00× @512 | 1.17× @1024 |
| LP `syevN` | 1.17× @1000 | 1.14× @2048 | 1.31× @2048 |
| LP `gelsd` | 1.25× @1000 | 1.05× @1000 | 1.17× @1000 |
| LP `potri` | 1.14× @1000 | 1.16× @2048 | 1.23× @2048 |
| LP `geev` | 1.13× @1000 | 1.09× @1000 | 1.21× @1000 |

### Zen3 · AVX2

Measured 2026-09-23T12:44 at commit `157895df`, AMD Ryzen 9 5900X 12-Core Processor, pinned at 3701 MHz with boost off.

1104 cells measured, 0 off-lock. Listed below: the threadable ops whose best cell moves further than the 3.7% noise floor.

**Where threading pays.** Best cell per operation:

| op | best speedup | at n | round spread |
|---|---|---|---|
| L3 `trsm` | **5.64×** | 4096 | 1% |
| L3 `syrk` | **5.45×** | 4096 | 1% |
| L3 `trsmR` | **5.18×** | 4096 | 0% |
| LP `potrfU` | **5.17×** | 4096 | 5% |
| L3 `gemm` | **4.75×** | 2100 | 0% |
| LP `potrf` | **4.53×** | 4096 | 7% |
| L3 `symm` | **4.50×** | 2100 | 0% |
| LP `getrf` | **3.89×** | 2100 | 1% |
| LP `geqrf` | **3.52×** | 4096 | 1% |
| L3 `syr2k` | **2.87×** | 4096 | 0% |
| LP `pptrfL` | **2.54×** | 2048 | 4% |
| LP `pptrfU` | **2.52×** | 2048 | 18% |
| LP `getri` | **2.44×** | 2048 | 0% |
| LP `pstrf` | **2.16×** | 2100 | 0% |
| LP `pstrfU` | **1.85×** | 2100 | 0% |
| LP `gesvd` | **1.81×** | 2048 | 0% |
| LP `syev` | **1.67×** | 2048 | 6% |
| LP `potrsU` | **1.40×** | 1000 | 11% |
| LP `potrsL` | **1.40×** | 1000 | 2% |
| LP `gels` | **1.39×** | 1024 | 1% |
| LP `pbtrfL` | **1.39×** | 384 | 0% |
| LP `gelsd` | **1.25×** | 1000 | 0% |
| LP `syevN` | **1.17×** | 1000 | 0% |
| LP `potri` | **1.14×** | 1000 | 3% |
| LP `geev` | **1.13×** | 1000 | 2% |

**Where threading COSTS.** Every cell that got slower with six threads:

| op | n | speedup | round spread |
|---|---|---|---|
| LP `getrs` | 1024 | **0.79×** | 38% |
| LP `getrs` | 2048 | **0.83×** | 10% |
| LP `pbtrfU` | 384 | **0.84×** | 2% |
| LP `getrs` | 1000 | **0.88×** | 17% |
| LP `trtri` | 32 | **0.94×** | 25% |
| LP `pbtrfU` | 256 | **0.95×** | 1% |
| LP `trtrs` | 50 | **0.95×** | 77% |

### Zen4 · AVX-512

Measured 2026-09-23T12:53 at commit `157895df`, AMD Ryzen 5 7640U w/ Radeon 760M Graphics, pinned at 2813 MHz with boost off.

1104 cells measured, 0 off-lock. Listed below: the threadable ops whose best cell moves further than the 3.7% noise floor.

**Where threading pays.** Best cell per operation:

| op | best speedup | at n | round spread |
|---|---|---|---|
| L3 `trsm` | **4.90×** | 2100 | 0% |
| L3 `syrk` | **4.80×** | 4096 | 5% |
| L3 `trsmR` | **4.61×** | 4096 | 1% |
| LP `potrfU` | **4.30×** | 4096 | 2% |
| L3 `syr2k` | **4.29×** | 4096 | 0% |
| L3 `gemm` | **4.01×** | 2100 | 1% |
| LP `potrf` | **3.94×** | 4096 | 0% |
| L3 `symm` | **3.74×** | 2100 | 2% |
| LP `getrf` | **3.45×** | 4096 | 1% |
| LP `pptrfL` | **3.20×** | 2048 | 2% |
| LP `pptrfU` | **3.10×** | 2048 | 4% |
| LP `getri` | **1.98×** | 2048 | 5% |
| LP `pstrf` | **1.76×** | 2100 | 1% |
| LP `syev` | **1.74×** | 2048 | 4% |
| LP `pstrfU` | **1.62×** | 2100 | 0% |
| LP `geqrf` | **1.57×** | 4096 | 1% |
| LP `pbtrfL` | **1.55×** | 384 | 2% |
| LP `gesvd` | **1.54×** | 2048 | 1% |
| LP `pbtrfU` | **1.44×** | 384 | 1% |
| LP `gels` | **1.39×** | 512 | 4% |
| LP `potri` | **1.16×** | 2048 | 0% |
| LP `syevN` | **1.14×** | 2048 | 2% |
| LP `geev` | **1.09×** | 1000 | 4% |
| LP `gelsd` | **1.05×** | 1000 | 1% |

**Where threading COSTS.** Every cell that got slower with six threads:

| op | n | speedup | round spread |
|---|---|---|---|
| L2 `trmv` | 2048 | **0.78×** | 73% |
| LP `potrsU` | 2048 | **0.89×** | 1% |
| LP `potrsL` | 1000 | **0.91×** | 12% |
| LP `potrsU` | 1024 | **0.92×** | 3% |
| LP `potrsL` | 1024 | **0.93×** | 2% |
| LP `potrsU` | 256 | **0.93×** | 13% |
| LP `potrsU` | 1000 | **0.93×** | 4% |
| LP `pptrfU` | 256 | **0.94×** | 2% |
| LP `potrsL` | 2048 | **0.95×** | 3% |
| LP `pstrf` | 256 | **0.96×** | 6% |

### Zen5 · AVX-512

Measured 2026-09-23T15:08 at commit `3b594f39`, AMD Ryzen AI 5 340 w/ Radeon 840M, pinned at 2000 MHz with boost off.

1104 cells measured, 0 off-lock. Listed below: the threadable ops whose best cell moves further than the 3.7% noise floor.

**Where threading pays.** Best cell per operation:

| op | best speedup | at n | round spread |
|---|---|---|---|
| L3 `trsm` | **5.88×** | 2100 | 0% |
| L3 `trsmR` | **5.57×** | 4096 | 1% |
| L3 `syrk` | **5.30×** | 2100 | 1% |
| L3 `syr2k` | **5.25×** | 4096 | 2% |
| LP `potrfU` | **5.24×** | 4096 | 0% |
| L3 `gemm` | **5.06×** | 2100 | 1% |
| LP `potrf` | **4.89×** | 4096 | 0% |
| L3 `symm` | **4.83×** | 2100 | 1% |
| LP `getrf` | **4.25×** | 4096 | 0% |
| LP `pptrfL` | **4.09×** | 2048 | 0% |
| LP `pptrfU` | **4.00×** | 2048 | 0% |
| LP `getri` | **2.60×** | 2048 | 0% |
| LP `pstrf` | **2.15×** | 2100 | 0% |
| LP `pstrfU` | **2.12×** | 2100 | 0% |
| LP `syev` | **2.07×** | 2048 | 3% |
| LP `pbtrfL` | **2.00×** | 384 | 1% |
| LP `gesvd` | **1.95×** | 2048 | 1% |
| LP `pbtrfU` | **1.82×** | 384 | 0% |
| LP `geqrf` | **1.73×** | 512 | 9% |
| LP `gels` | **1.67×** | 512 | 14% |
| LP `getrs` | **1.58×** | 1024 | 14% |
| LP `syevN` | **1.31×** | 2048 | 1% |
| LP `potri` | **1.23×** | 2048 | 0% |
| LP `geev` | **1.21×** | 1000 | 3% |
| LP `potrsL` | **1.17×** | 1024 | 3% |
| LP `gelsd` | **1.17×** | 1000 | 0% |

**Where threading COSTS.** Every cell that got slower with six threads:

| op | n | speedup | round spread |
|---|---|---|---|
| LP `potrsU` | 2048 | **0.87×** | 2% |
| LP `potrsU` | 1000 | **0.92×** | 9% |
| LP `potrsU` | 1024 | **0.93×** | 2% |
| LP `potrsU` | 256 | **0.93×** | 3% |
| LP `potrsL` | 1000 | **0.94×** | 8% |
| LP `potrsL` | 2048 | **0.95×** | 4% |

## Open: `gesvd` gets slower with threads

`gesvd` is the one routine that is **worse** with threads, and it reproduces on all three
microarchitectures — though not equally, which is itself a clue:

| box | n=256 | n=512 | n=1000 | n=1024 | n=2048 |
|---|---|---|---|---|---|
| Zen3 · AVX2 | 1.13× | 1.35× | 1.58× | 1.41× | 1.81× |
| Zen4 · AVX-512 | 1.08× | 1.26× | 1.50× | 1.45× | 1.54× |
| Zen5 · AVX-512 | 1.17× | 1.46× | 1.86× | 1.80× | 1.95× |

The root cause is **not yet known**. Three plausible explanations have been measured and rejected:

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

