# Methodology & provenance

How every published number on [Performance](performance.md) and [Coverage](coverage.md) is produced,
where it comes from, and the caveats that change how a cell should be read.

## The gate

**PB ≥ max(OpenBLAS, AOCL), per (op, size, machine), single-threaded.** A ratio > 1.0 means PureBLAS is
faster. Both references are always measured, because a routine can beat one by a wide margin and lose to
the other; the gate takes the faster of the two at each individual cell.

Verdicts are **per microarchitecture** — a single pooled verdict cannot express "gates on Zen4, misses on
Zen3", and pooling once misreported banded Cholesky as 0.891 failing when Zen4 alone read 1.03. Every
table therefore carries one column per box, each from that box's own `bench/plots.jl` sweep.

The dev fleet, gated independently (tuning for one µarch does not transfer):

| µarch | host | ISA | role |
|---|---|---|---|
| Zen4 | `wintermute` | double-pumped AVX-512 | primary tuning target |
| Zen3 | `galen` | AVX2, 16 ymm registers | hardest target |
| Zen5 | `neuromancer` | native 512-bit AVX-512 | disjoint residual profile |

## Measurement

`bench/plots.jl`, single-thread (`BLAS.set_num_threads(1)`), native PureBLAS API vs
`LinearAlgebra.BLAS`, Float64 plus the full ComplexF64 surface. Each (op, size) is measured over
repeated rounds of ABBA-alternated windows with every arm (PureBLAS, OpenBLAS, AOCL) interleaved inside
one run; per-round ratios are pooled and the **median** is the reported number (never a mean, never a
minimum). Runs are only valid at locked frequency — a floating boost clock drifts between the two
windows and fabricates ratios. Reproduce:

```
sudo bench/fleet_freqlock.sh lock       # passive governor, boost off, pin to base clock
taskset -c N julia --project=bench bench/plots.jl bench
```

The published artifacts are a pure function of the caches on disk and are rebuilt in one command
(`bench/publish.sh`); `bench/check_artifacts_current.sh` re-renders into a temp dir and byte-compares.

## Provenance

Per-box CPU, host, code commit and measurement timestamp for the caches behind every published number:
[`bench/provenance.md`](https://github.com/el-oso/PureBLAS.jl/blob/master/bench/provenance.md).
Cells are stamped **individually** — a routine untouched since an earlier run keeps that run's stamp,
which the cache records per arm — so the file's headline commit is the run, not a guarantee about each
cell. Nothing in the tables is hand-transcribed: `bench/coverage_ops.jl` and `bench/coverage_routing.jl`
compute each verdict from the stored quantile vectors.

Both reference views (OpenBLAS and AOCL) are rendered from that one cache set in a single invocation, so
their provenance is identical by construction and they cannot disagree about the same fleet. The µarch
labels are stamped authoritatively into each cache header (`uarch=`).

## How to read a cell

- **Colour bands** in the per-routine tables band the shortfall (**≥ 1.0 gates** · **≥ 0.99** ·
  **≥ 0.95** · **≥ 0.85** · **below 0.85**). The number is the verdict; colour only bands it, so the
  table reads correctly in monochrome and for a colourblind reader. A failing gate is printed at three
  digits and **floored**, so a miss can never round up to a reassuring `1.0`.
- **A full re-sweep moves every cell by ±1–2%, without any code change.** A sweep re-measures the
  references as well as PB: two independent all-arms runs of `trmm` on Zen3 hours apart gave 0.948 and
  0.929. Only differences larger than that, or ones corroborated by a controlled same-process A/B,
  should be read as real.
- **Cells within ~1% of parity are not adjudicable** when the run used `arms=pb` (fresh PB arm, cached
  reference arms). `bench/gate_verdict.jl` models this, flags them ⚠ and widens the interval by a drift
  term.
- **A small ratio is not automatically a defect.** `bench/gate_gaps.jl` prints each cell's
  round-to-round spread and marks any shortfall smaller than its own spread `within spread`. Those need
  more rounds, not engineering. The genuinely open cells are the ones outside their spread.
- **Off-lock cells are excluded, not published.** A cell measured while the frequency lock was floating
  is not adjudicable; `bench/coverage_ops.jl` drops it, reports it on stderr, and marks the routine's
  cell `off-lock (n)` so a partial verdict never reads as a complete one.
- **A large win can mean the reference did not implement the routine.** AOCL's packed Cholesky is ~6×
  slower than OpenBLAS's (near-reference code), so beating it there means little. This does not change
  the gate — `max(OpenBLAS, AOCL)` already handles it — but a cell far above one reference and below the
  other is the tell that the high ratio measures the reference's absence, not our speed.
- **Small-n LAPACK vs AOCL flatters PureBLAS.** AOCL-libFLAME has heavy per-call overhead at tiny n
  (e.g. `potrf`/`getrf` show 4–160× at n ≤ 32) — that is dispatch overhead, not an algorithmic gap. The
  mid/large-n figures are the honest signal.
- **Zen5 is a laptop, not a dedicated bench box.** Repeated sweeps there move worst cells by several
  points — `gemvT` n=2048 read 1.16 and 0.977 in two runs an hour apart at the same commit. Treat Zen5
  cells within a few points of 1.0 as unresolved rather than as measurements; Zen3 and Zen4 are stable.

## The AOCL baseline

OpenBLAS is the default oracle; AOCL (AOCL-BLIS for BLAS, AOCL-libFLAME for LAPACK) is the second,
AMD-tuned baseline, hand-tuned for these exact Zen chips. Everything is single-threaded (BLIS pinned to
one thread — verified: `dgemm` runs at one-core throughput), boost-locked, same methodology; the AOCL
artifacts carry an `_aocl` suffix and never mix with the OpenBLAS ones. The AOCL plots cover exactly the
same ground as the OpenBLAS plots — same ops, panels, size sweep and µarchs — because both views come
from the same cache.

The AOCL binary is a genuinely optimized build, not a reference fallback: measured directly against
OpenBLAS, AOCL-BLIS `dgemm` matches it and AOCL-libFLAME `potrf`/`geqrf` meet or beat it (e.g. geqrf 40
vs 30 GFlops). It is a *mixed* competitor rather than uniformly tougher — its `geqrf` beats OpenBLAS
while its `getrf` trails it — and it is tuned first for multi-threaded EPYC, so on these single-thread
Zen parts it is a fair-but-not-dominant baseline.

## Contamination class fixed on 2026-08-09

`bench/plots.jl` merges its cache per arm, so an A/B run with a `PUREBLAS_FORCE_<knob>` variable
exported persisted its **deliberately non-default** PureBLAS arm next to reference arms measured in a
different run, and every render afterwards republished it as the gate number. The Zen5 `gemvT` n=256
cell was published at 0.751 on that basis; measured with all three arms in one run it is 1.09, and
`PUREBLAS_FORCE_gemvt_deep=0` reproduces the bad record exactly. `gemvT` n=128 and `trmv` n=256 came
from the same contamination. `save_cache` now refuses to write while any `PUREBLAS_FORCE_*` is set, so
the class cannot recur silently. The corrections did not all favour PureBLAS — `ztrsv` on Zen4 went from
0.991 to 0.865 and `ger` on Zen5 from 0.996 to 0.896.
