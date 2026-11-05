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
![BLAS-2 — PureBLAS 6 threads / 1 thread](assets/perf_mt_l2.svg)
![BLAS-3 — PureBLAS 6 threads / 1 thread](assets/perf_mt_l3.svg)
![LAPACK — PureBLAS 6 threads / 1 thread](assets/perf_mt_lapack.svg)

Regenerate both sets with `julia --project=bench bench/plots.jl mtdraw`, which writes
`perf_mtgate_*.svg` and `perf_mt_*.svg` and exits before the gate rendering.

## Results

Best speedup per operation, six threads against one, on each box.

| op | Zen3 · AVX2 | Zen4 · AVX-512 | Zen5 · AVX-512 |
|---|---|---|---|
| L1 `blascopy` | 4.98× @100000 | 6.30× @1000000 | 7.58× @300000 |
| L2 `gemvTrect` | 5.49× @758 | 5.79× @2880 | 6.98× @758 |
| L1 `axpy` | 4.50× @100000 | 5.39× @1000000 | 6.79× @300000 |
| L1 `swap` | 4.26× @100000 | 6.40× @1000000 | 6.70× @300000 |
| L1 `asum` | 4.01× @300000 | 4.56× @1000000 | 6.30× @300000 |
| L1 `nrm2` | 3.90× @300000 | 4.65× @1000000 | 6.27× @300000 |
| L2 `gemvNrect` | 3.45× @256 | 4.78× @448 | 6.08× @448 |
| L1 `dot` | 3.86× @100000 | 5.58× @1000000 | 6.00× @300000 |
| L3 `trmm` | 5.77× @2100 | 5.73× @2100 | 5.88× @2100 |
| L1 `scal` | 4.60× @100000 | 4.31× @300000 | 5.86× @1000000 |
| L3 `trsm` | 5.76× @2100 | 5.72× @2100 | 5.86× @2100 |
| L3 `gemm` | 4.87× @2100 | 5.67× @1000 | 5.74× @1000 |
| L3 `trsmRrect` | 5.06× @758 | 5.72× @2880 | 5.73× @2880 |
| L1 `iamax` | 4.46× @300000 | 4.36× @1000000 | 5.67× @1000000 |
| L2 `gemvT` | 5.47× @512 | 5.00× @1024 | 5.64× @1000 |
| L3 `trmmR` | 4.76× @512 | 5.46× @1000 | 5.63× @1024 |
| L3 `trsmR` | 5.19× @2100 | 5.31× @4096 | 5.62× @4096 |
| L2 `ger` | 5.55× @512 | 4.88× @512 | 5.50× @1000 |
| L3 `syr2k` | 5.47× @2100 | 4.52× @2100 | 5.29× @4096 |
| L3 `syrk` | 5.43× @2100 | 5.12× @2100 | 5.36× @2100 |
| LP `potrfU` | 5.43× @4096 | 5.19× @4096 | 5.27× @4096 |
| L3 `symm` | 4.61× @2100 | 4.88× @1000 | 5.35× @1000 |
| LP `geqrf` | 4.52× @2100 | 4.32× @4096 | 5.31× @4096 |
| LP `getri` | 4.78× @2048 | 4.68× @2048 | 5.30× @2048 |
| LP `trtri` | 4.79× @2048 | 4.90× @2048 | 5.27× @2048 |
| LP `potrf` | 4.73× @4096 | 4.59× @2100 | 5.05× @2100 |
| LP `potri` | 4.13× @1000 | 4.70× @2048 | 4.92× @2048 |
| L3 `trsmRband` | 4.29× @512 | 4.28× @512 | 4.84× @512 |
| L2 `gbmvN` | 4.11× @4096 | 4.13× @4096 | 4.40× @4096 |
| LP `getrf` | 4.02× @2100 | 3.87× @4096 | 4.28× @4096 |
| L2 `gemvN` | 3.09× @1024 | 3.95× @1000 | 4.24× @1024 |
| LP `pptrfL` | 2.92× @2048 | 3.80× @2048 | 4.16× @2048 |
| LP `pptrfU` | 2.70× @2048 | 3.85× @2048 | 4.08× @2048 |
| L2 `symv` | 3.21× @2048 | 3.59× @2100 | 3.58× @1024 |
| LP `gels` | 2.19× @512 | 3.23× @1024 | 3.55× @1000 |
| LP `geqp3` | 3.07× @1000 | 3.15× @2048 | 3.38× @1000 |
| LP `sytri` | 3.11× @2048 | 3.24× @2048 | 3.06× @1000 |
| LP `gesvd` | 3.16× @2048 | 3.10× @2048 | 3.23× @2048 |
| LP `gelsy` | 2.73× @1000 | 2.86× @1000 | 3.19× @1000 |
| LP `pstrf` | 2.74× @4096 | 2.52× @4096 | 3.11× @2100 |
| LP `syev` | 2.86× @2048 | 2.91× @2048 | 3.08× @2048 |
| LP `pstrfU` | 2.44× @2100 | 1.95× @2100 | 2.78× @2100 |
| LP `sytrf` | 2.71× @4096 | 2.09× @4096 | 2.46× @4096 |
| LP `syevN` | 2.53× @2048 | 2.38× @2048 | 2.57× @2048 |
| LP `gelsd` | 2.35× @1000 | 2.21× @1000 | 2.43× @1000 |
| LP `pbtrfL` | 1.53× @256 | 1.93× @384 | 2.07× @384 |
| LP `pbtrfU` | 1.26× @192 | 1.70× @384 | 1.87× @384 |
| LP `getrs` | 1.01× @512 | 1.05× @2048 | 1.54× @1024 |
| LP `potrsL` | 1.46× @1000 | 1.00× @512 | 1.07× @1024 |
| LP `potrsU` | 1.46× @1000 | 1.01× @512 | 1.01× @32 |
| LP `geev` | 1.27× @1000 | 1.24× @1000 | 1.33× @1000 |
| LP `sytrs` | 1.01× @2048 | 1.00× @50 | 1.06× @2048 |

### Zen3 · AVX2

Measured 2026-11-01T15:11 at commit `51228327`, AMD Ryzen 9 5900X 12-Core Processor, pinned at 3701 MHz with boost off.

1142 cells measured, 0 off-lock. Listed below: the threadable ops whose best cell moves further than the 3.7% noise floor.

**Where threading pays.** Best cell per operation:

| op | best speedup | at n | round spread |
|---|---|---|---|
| L3 `trmm` | **5.77×** | 2100 | 0% |
| L3 `trsm` | **5.76×** | 2100 | 0% |
| L2 `ger` | **5.55×** | 512 | 3% |
| L2 `gemvTrect` | **5.49×** | 758 | 1% |
| L3 `syr2k` | **5.47×** | 2100 | 0% |
| L2 `gemvT` | **5.47×** | 512 | 4% |
| L3 `syrk` | **5.43×** | 2100 | 1% |
| LP `potrfU` | **5.43×** | 4096 | 0% |
| L3 `trsmR` | **5.19×** | 2100 | 0% |
| L3 `trsmRrect` | **5.06×** | 758 | 20% |
| L1 `blascopy` | **4.98×** | 100000 | 1% |
| L3 `gemm` | **4.87×** | 2100 | 0% |
| LP `trtri` | **4.79×** | 2048 | 2% |
| LP `getri` | **4.78×** | 2048 | 2% |
| L3 `trmmR` | **4.76×** | 512 | 9% |
| LP `potrf` | **4.73×** | 4096 | 0% |
| L3 `symm` | **4.61×** | 2100 | 0% |
| L1 `scal` | **4.60×** | 100000 | 1% |
| LP `geqrf` | **4.52×** | 2100 | 0% |
| L1 `axpy` | **4.50×** | 100000 | 3% |
| L1 `iamax` | **4.46×** | 300000 | 20% |
| L3 `trsmRband` | **4.29×** | 512 | 1% |
| L1 `swap` | **4.26×** | 100000 | 2% |
| LP `potri` | **4.13×** | 1000 | 1% |
| L2 `gbmvN` | **4.11×** | 4096 | 0% |
| LP `getrf` | **4.02×** | 2100 | 0% |
| L1 `asum` | **4.01×** | 300000 | 21% |
| L1 `nrm2` | **3.90×** | 300000 | 30% |
| L1 `dot` | **3.86×** | 100000 | 2% |
| L2 `gemvNrect` | **3.45×** | 256 | 3% |
| L2 `symv` | **3.21×** | 2048 | 19% |
| LP `gesvd` | **3.16×** | 2048 | 4% |
| LP `sytri` | **3.11×** | 2048 | 1% |
| L2 `gemvN` | **3.09×** | 1024 | 5% |
| LP `geqp3` | **3.07×** | 1000 | 1% |
| LP `pptrfL` | **2.92×** | 2048 | 13% |
| LP `syev` | **2.86×** | 2048 | 6% |
| LP `pstrf` | **2.74×** | 4096 | 2% |
| LP `gelsy` | **2.73×** | 1000 | 2% |
| LP `sytrf` | **2.71×** | 4096 | 1% |
| LP `pptrfU` | **2.70×** | 2048 | 21% |
| LP `syevN` | **2.53×** | 2048 | 3% |
| LP `pstrfU` | **2.44×** | 2100 | 0% |
| LP `gelsd` | **2.35×** | 1000 | 1% |
| LP `gels` | **2.19×** | 512 | 0% |
| LP `pbtrfL` | **1.53×** | 256 | 0% |
| LP `potrsL` | **1.46×** | 1000 | 0% |
| LP `potrsU` | **1.46×** | 1000 | 1% |
| LP `geev` | **1.27×** | 1000 | 4% |
| LP `pbtrfU` | **1.26×** | 192 | 12% |

**Where threading COSTS.** Every cell that got slower with six threads:

| op | n | speedup | round spread |
|---|---|---|---|
| LP `pbtrfU` | 256 | **0.58×** | 3% |
| L2 `gemvN` | 128 | **0.73×** | 6% |
| LP `getrs` | 1000 | **0.76×** | 40% |
| LP `pbtrfU` | 384 | **0.77×** | 1% |
| LP `getrs` | 1024 | **0.79×** | 14% |
| LP `sytrf` | 1000 | **0.85×** | 3% |
| LP `sytrs` | 1000 | **0.87×** | 12% |
| LP `getrs` | 2048 | **0.88×** | 5% |
| LP `sytrs` | 1024 | **0.91×** | 5% |
| LP `potrsL` | 1024 | **0.92×** | 22% |
| LP `pbtrfL` | 384 | **0.94×** | 4% |
| L2 `ger` | 4096 | **0.96×** | 1% |

### Zen4 · AVX-512

Measured 2026-11-03T00:12 at commit `26da5b14`, AMD Ryzen 5 7640U w/ Radeon 760M Graphics, pinned at 3501 MHz with boost off.

1142 cells measured, 0 off-lock. Listed below: the threadable ops whose best cell moves further than the 3.7% noise floor.

**Where threading pays.** Best cell per operation:

| op | best speedup | at n | round spread |
|---|---|---|---|
| L1 `swap` | **6.40×** | 1000000 | 3% |
| L1 `blascopy` | **6.30×** | 1000000 | 9% |
| L2 `gemvTrect` | **5.79×** | 2880 | 2% |
| L3 `trmm` | **5.73×** | 2100 | 0% |
| L3 `trsm` | **5.72×** | 2100 | 0% |
| L3 `trsmRrect` | **5.72×** | 2880 | 0% |
| L3 `gemm` | **5.67×** | 1000 | 0% |
| L1 `dot` | **5.58×** | 1000000 | 7% |
| L3 `trmmR` | **5.46×** | 1000 | 0% |
| L1 `axpy` | **5.39×** | 1000000 | 9% |
| L3 `trsmR` | **5.31×** | 4096 | 1% |
| LP `potrfU` | **5.19×** | 4096 | 1% |
| L3 `syrk` | **5.12×** | 2100 | 0% |
| L2 `gemvT` | **5.00×** | 1024 | 2% |
| LP `trtri` | **4.90×** | 2048 | 1% |
| L3 `symm` | **4.88×** | 1000 | 0% |
| L2 `ger` | **4.88×** | 512 | 2% |
| L2 `gemvNrect` | **4.78×** | 448 | 11% |
| LP `potri` | **4.70×** | 2048 | 1% |
| LP `getri` | **4.68×** | 2048 | 0% |
| L1 `nrm2` | **4.65×** | 1000000 | 2% |
| LP `potrf` | **4.59×** | 2100 | 1% |
| L1 `asum` | **4.56×** | 1000000 | 6% |
| L3 `syr2k` | **4.52×** | 2100 | 1% |
| L1 `iamax` | **4.36×** | 1000000 | 6% |
| LP `geqrf` | **4.32×** | 4096 | 1% |
| L1 `scal` | **4.31×** | 300000 | 12% |
| L3 `trsmRband` | **4.28×** | 512 | 6% |
| L2 `gbmvN` | **4.13×** | 4096 | 1% |
| L2 `gemvN` | **3.95×** | 1000 | 23% |
| LP `getrf` | **3.87×** | 4096 | 1% |
| LP `pptrfU` | **3.85×** | 2048 | 0% |
| LP `pptrfL` | **3.80×** | 2048 | 1% |
| L2 `symv` | **3.59×** | 2100 | 5% |
| LP `sytri` | **3.24×** | 2048 | 1% |
| LP `gels` | **3.23×** | 1024 | 1% |
| LP `geqp3` | **3.15×** | 2048 | 0% |
| LP `gesvd` | **3.10×** | 2048 | 3% |
| LP `syev` | **2.91×** | 2048 | 8% |
| LP `gelsy` | **2.86×** | 1000 | 1% |
| LP `pstrf` | **2.52×** | 4096 | 1% |
| LP `syevN` | **2.38×** | 2048 | 2% |
| LP `gelsd` | **2.21×** | 1000 | 0% |
| LP `sytrf` | **2.09×** | 4096 | 3% |
| LP `pstrfU` | **1.95×** | 2100 | 0% |
| LP `pbtrfL` | **1.93×** | 384 | 0% |
| LP `pbtrfU` | **1.70×** | 384 | 1% |
| LP `geev` | **1.24×** | 1000 | 3% |
| LP `getrs` | **1.05×** | 2048 | 4% |

**Where threading COSTS.** Every cell that got slower with six threads:

| op | n | speedup | round spread |
|---|---|---|---|
| LP `potrsL` | 1000 | **0.81×** | 8% |
| LP `potrsU` | 1000 | **0.83×** | 5% |
| LP `potrsU` | 1024 | **0.83×** | 4% |
| LP `geqp3` | 256 | **0.86×** | 2% |
| LP `gelsy` | 256 | **0.86×** | 2% |
| LP `potrsL` | 1024 | **0.86×** | 3% |
| LP `potrsU` | 2048 | **0.91×** | 11% |
| LP `sytrf` | 1000 | **0.94×** | 1% |
| LP `syevN` | 256 | **0.95×** | 1% |

### Zen5 · AVX-512

Measured 2026-11-01T01:20 at commit `51228327`, AMD Ryzen AI 5 340 w/ Radeon 840M, pinned at 2000 MHz with boost off.

1125 cells measured, 17 off-lock. Listed below: the threadable ops whose best cell moves further than the 3.7% noise floor.

**Where threading pays.** Best cell per operation:

| op | best speedup | at n | round spread |
|---|---|---|---|
| L1 `blascopy` | **7.58×** | 300000 | 4% |
| L2 `gemvTrect` | **6.98×** | 758 | 17% |
| L1 `axpy` | **6.79×** | 300000 | 5% |
| L1 `swap` | **6.70×** | 300000 | 4% |
| L1 `asum` | **6.30×** | 300000 | 2% |
| L1 `nrm2` | **6.27×** | 300000 | 3% |
| L2 `gemvNrect` | **6.08×** | 448 | 6% |
| L1 `dot` | **6.00×** | 300000 | 4% |
| L3 `trmm` | **5.88×** | 2100 | 0% |
| L1 `scal` | **5.86×** | 1000000 | 7% |
| L3 `trsm` | **5.86×** | 2100 | 0% |
| L3 `gemm` | **5.74×** | 1000 | 1% |
| L3 `trsmRrect` | **5.73×** | 2880 | 0% |
| L1 `iamax` | **5.67×** | 1000000 | 6% |
| L2 `gemvT` | **5.64×** | 1000 | 5% |
| L3 `trmmR` | **5.63×** | 1024 | 1% |
| L3 `trsmR` | **5.62×** | 4096 | 1% |
| L2 `ger` | **5.50×** | 1000 | 2% |
| L3 `syrk` | **5.36×** | 2100 | 0% |
| L3 `symm` | **5.35×** | 1000 | 2% |
| LP `geqrf` | **5.31×** | 4096 | 1% |
| LP `getri` | **5.30×** | 2048 | 0% |
| L3 `syr2k` | **5.29×** | 4096 | 2% |
| LP `trtri` | **5.27×** | 2048 | 0% |
| LP `potrfU` | **5.27×** | 4096 | 1% |
| LP `potrf` | **5.05×** | 2100 | 0% |
| LP `potri` | **4.92×** | 2048 | 0% |
| L3 `trsmRband` | **4.84×** | 512 | 1% |
| L2 `gbmvN` | **4.40×** | 4096 | 2% |
| LP `getrf` | **4.28×** | 4096 | 0% |
| L2 `gemvN` | **4.24×** | 1024 | 18% |
| LP `pptrfL` | **4.16×** | 2048 | 0% |
| LP `pptrfU` | **4.08×** | 2048 | 0% |
| L2 `symv` | **3.58×** | 1024 | 6% |
| LP `gels` | **3.55×** | 1000 | 1% |
| LP `geqp3` | **3.38×** | 1000 | 0% |
| LP `gesvd` | **3.23×** | 2048 | 1% |
| LP `gelsy` | **3.19×** | 1000 | 0% |
| LP `pstrf` | **3.11×** | 2100 | 1% |
| LP `syev` | **3.08×** | 2048 | 2% |
| LP `sytri` | **3.06×** | 1000 | 2% |
| LP `pstrfU` | **2.78×** | 2100 | 0% |
| LP `syevN` | **2.57×** | 2048 | 2% |
| LP `sytrf` | **2.46×** | 4096 | 3% |
| LP `gelsd` | **2.43×** | 1000 | 1% |
| LP `pbtrfL` | **2.07×** | 384 | 1% |
| LP `pbtrfU` | **1.87×** | 384 | 1% |
| LP `getrs` | **1.54×** | 1024 | 40% |
| LP `geev` | **1.33×** | 1000 | 3% |
| LP `potrsL` | **1.07×** | 1024 | 7% |
| LP `sytrs` | **1.06×** | 2048 | 2% |

**Where threading COSTS.** Every cell that got slower with six threads:

| op | n | speedup | round spread |
|---|---|---|---|
| LP `potrsU` | 1000 | **0.80×** | 5% |
| LP `potrsU` | 2048 | **0.80×** | 2% |
| LP `potrsU` | 1024 | **0.82×** | 5% |
| LP `potrsL` | 1000 | **0.84×** | 7% |

## Open: `gesvd` gets slower with threads

`gesvd` is the one routine that is **worse** with threads, and it reproduces on all three
microarchitectures — though not equally, which is itself a clue:

| box | n=256 | n=512 | n=1000 | n=1024 | n=2048 |
|---|---|---|---|---|---|
| Zen3 · AVX2 | 1.30× | 1.57× | 2.00× | 1.82× | 3.16× |
| Zen4 · AVX-512 | 1.21× | 1.84× | 2.65× | 2.56× | 3.10× |
| Zen5 · AVX-512 | 1.37× | 2.12× | 3.01× | 2.84× | 3.23× |

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

