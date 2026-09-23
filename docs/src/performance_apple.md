# Apple Silicon

First-pass results on Apple Silicon (arm64/NEON), comparing PureBLAS against **OpenBLAS** and
**Accelerate** (Apple's own vecLib BLAS/LAPACK, forwarded via `libblastrampoline`'s ILP64
`$NEWLAPACK$ILP64` interface — see [Methodology](methodology.md#accelerate-on-apple-silicon)).
This is the first time PureBLAS has been benchmarked on this architecture; it is not yet part of the
[main fleet](performance.md), which stays AMD-only.

**Scope, deliberately limited for this first pass:** BLAS-1, BLAS-2, BLAS-3, and the four core LAPACK
factorizations (`potrf`, `getrf`, `geqrf`, `gesvd`), Float64 only, sizes capped at 2048. Complex,
ForwardDiff-dual, and the rest of LAPACK are not yet covered here.

**Caveat that does not apply to the AMD fleet:** macOS has no equivalent of the Linux `cpufreq`/`taskset`
locking [Methodology](methodology.md) treats as mandatory for a gate-quality measurement. These numbers
are **not frequency-locked** — read them as directional, not gate verdicts.

**Machine:** Apple M6 (ARM · NEON, `_vwidth(Float64)=2`), unlocked clock, commit `76cac7b6`, measured
2026-09-22. Full provenance: [`provenance.md`](assets/apple/provenance.md).

**Two real measurement bugs were found and fixed while building this page** — both caught by pushing
back on results that didn't pass a physical-plausibility check, not by anything in the harness itself.
Both are written up below and in `ROADMAP.md`; the table and plots reflect the numbers *after* both
fixes.

## Headline

Of the 29 real (Float64) cells measured, **1 gates outright** (`nrm2`, where OpenBLAS's always-scaled
algorithm is slow on every platform PureBLAS has been measured on) — but the gate figure (worst-cell)
understates BLAS-1: on the **median**, PureBLAS *beats* Accelerate on `dot` (1.09×) and `axpy` (1.19×),
and sits near parity on `asum`/`scal`.

PureBLAS is roughly at parity with **OpenBLAS** overall (0.5-2.1× across BLAS-1/2, 0.7-1.6× on BLAS-3/
LAPACK), and trails **Accelerate** substantially on BLAS-2/3/LAPACK (0.08-0.3× on most of BLAS-3). That
part of the gap is real — it holds under genuinely cold, genuinely single-threaded measurement — and
most plausibly tracks Accelerate routing through Apple's AMX matrix coprocessor (~469 GFLOP/s single-core
`dgemm`, verified single-threaded — see below), which pure-Julia NEON code (128-bit, 2 Float64 lanes) has
no access to. Closing that is a tuning/architecture question for a future session — `tune!()`'s existing
Measure-tier calibrator does not cover it (see [Tuning](#tuning-tune-and-its-limits) below).

## Measurement bug #1: large-n BLAS-1 was reading a warm buffer, not DRAM streaming

The first pass through this page reported `dot`/`axpy`/`asum`/`scal` losing to Accelerate by 3-5× at
large n. That number was inflated by a benchmark-harness artifact, not a real hardware gap.

`bench/plots.jl`'s `_L1REP(s) = clamp(8_000_000 ÷ s, 30, 20000)` amortizes per-call timer overhead by
running `reps` calls back-to-back **on the same operand buffer** inside one timed window. At n=1e6 that
buffer is 16MB (`x`+`y`) — small enough to plausibly stay resident in L2/SLC across all `reps=30` calls,
so the measurement can end up reading repeated-access-to-a-warm-buffer throughput rather than genuine
cold/DRAM-streaming throughput.

Verified directly: with genuinely cold, single-shot, freshly-allocated arrays at sizes far too large for
any cache (10M-50M elements, 240MB-1.2GB), Accelerate's real advantage over OpenBLAS on `axpy` is only
**1.06-1.36×** — physically consistent with a modest per-core DRAM-bandwidth edge — not the ~4.3× the
reps-amortized measurement implied.

**Fix:** `bench/plots.jl` gained a `cold` flag (opt-in, off by default) forcing `reps=1` on the L1 sweep.
Not the default; changes nothing for the AMD fleet. **Open question:** whether the same artifact affects
the AMD fleet's own large-n L1 cells (Zen3's 32 MiB L3 is large enough for n=1e6's 16MB working set to
fit there too) — diagnosed on one Apple Silicon box, unconfirmed on x86.

## Measurement bug #2: `set_num_threads(1)` doesn't constrain Accelerate

More consequential: this one gave Accelerate a silent, uncontrolled thread-count advantage on every
compute-heavy op, for as long as the Accelerate backend existed, until fixed. Found after refusing to
believe a ~550 GFLOP/s single-core `dgemm` figure — correctly; it wasn't single-core.

`bench/plots.jl` relies on `LinearAlgebra.BLAS.set_num_threads(1)` (called after every backend forward)
to keep every reference single-threaded. That works for OpenBLAS/AOCL/MKL. It does **not** work for
Accelerate — vecLib reads `VECLIB_MAXIMUM_THREADS` at its own first-use initialization, independent of
LBT's thread knob, and that variable was never set. Confirmed empirically: a `dgemm` at n=4000 pinned
process CPU at a sustained **~190-200%** (`ps -o %cpu=`, sampled continuously across a 14 s / 60-call
window) despite `set_num_threads(1)` having been called.

**Fix:** `ENV["VECLIB_MAXIMUM_THREADS"] = "1"` set at `bench/plots.jl` load time, before Accelerate can
ever be forwarded — mirrors the existing BLIS/OMP env-var pattern already there for AOCL. Verified: the
same workload then pins at a clean, sustained ~99-100% CPU, both via a raw timing loop and via Chairmarks
(the real measurement path).

**How much this actually moved the numbers — smaller than the discovery made it sound.** The corrected
single-thread `dgemm` is ~469 GFLOP/s, not ~550 (the "second thread" bought only ~16%, not 2×, consistent
with AMX being a per-core resource a second software thread can't usefully double up on). Re-measuring
`gemm` with the fix moved its ratio against Accelerate by **+2% to +15%, growing with n** — real and
correctly signed, not a dramatic reversal. **LAPACK was unaffected** — `potrf` tested directly at
n=2048 (400 calls, continuous CPU sampling) stayed clean single-threaded *both* before and after the fix;
vecLib apparently only auto-parallelizes certain BLAS-3 routines, not its LAPACK factorizations, at least
not at this problem size. Every cell in the table below was re-measured from scratch after this fix.

## Tuning: `tune!()` and its limits

Running `PureBLAS.tune!(dryrun=true, unlocked=true)` (the project's own Measure-tier auto-calibrator —
see `src/tune.jl`) found two real, unpinned wins on this box: `gemvT` routing switches to per-column
mode at n≥2048 for **+11-13%**, and `potrf`'s upper-direct cutoff moves from 20 to 8 for **+10-11%**.
Everything else it checks (gemv tile shape, `ger` panel width, `sytrf`, banded LU) — the shipped default
already wins or it's a wash. (One portability bug fixed along the way: `bench/calibrate.jl` read a
Linux-only cpufreq sysfs path unconditionally even in `unlocked` mode, crashing on any box without it —
now guarded with `isfile()`, matching `FreqLock.lock_state`'s existing degradation pattern.)

**`tune!()` does not touch the thing actually driving the broad L3 gap.** GEMM's microkernel tile shape
(`_MR`×`_NR`) is governed by `_ILP_TARGET=16`, a *Derive-tier* formula — an explicit x86 FMA-latency/port
assumption, applied here completely unvalidated. There is no calibrator for it, on any architecture.
Closing it needs either a real ARM-specific derivation or a new Measure-tier calibrator for it — separate,
larger work than this data-generation pass.

## vs OpenBLAS and Accelerate, per op

Ratio is PB / reference, **median (worst cell)** across the measured size ladder (capped at n=2048 for
BLAS-2/3/LAPACK; full 1e3..1e6 for BLAS-1, L1 measured `cold` — see above). Gate is PB / max(OpenBLAS,
Accelerate) — the same two-significant-digit rounding rule as the
[main fleet](methodology.md#the-gate) (`bench/gatecrit.jl`). The gate is a **worst-cell** figure, so a
FAIL can still have a winning median — true for `dot`/`axpy`/`asum` below.

| level | op | vs OpenBLAS | vs Accelerate | gate | verdict |
|---|---|---|---|---|---|
| L1 | `dot` | 1.00 (0.99) | **1.09** (0.41) | 0.411 | FAIL |
| L1 | `axpy` | 1.02 (1.00) | **1.19** (0.54) | 0.541 | FAIL |
| L1 | `nrm2` | 8.37 (7.00) | 1.96 (1.67) | 1.672 | **PASS** |
| L1 | `asum` | 1.02 (1.00) | **1.00** (0.54) | 0.544 | FAIL |
| L1 | `scal` | 0.99 (0.99) | 0.83 (0.39) | 0.389 | FAIL |
| L1 | `iamax` | 0.50 (0.49) | 0.93 (0.90) | 0.494 | FAIL |
| L2 | `gemvN` | 1.52 (0.84) | 0.17 (0.10) | 0.103 | FAIL |
| L2 | `gemvT` | 1.51 (0.94) | 0.34 (0.16) | 0.162 | FAIL |
| L2 | `ger` | 1.00 (0.98) | 0.37 (0.25) | 0.247 | FAIL |
| L2 | `symv` | 0.74 (0.59) | 1.15 (0.33) | 0.329 | FAIL |
| L2 | `trmv` | 1.80 (1.36) | 0.67 (0.16) | 0.157 | FAIL |
| L2 | `trsv` | 1.65 (1.37) | 0.95 (0.68) | 0.679 | FAIL |
| L2 | `trsvLN` | 1.42 (1.32) | 0.85 (0.69) | 0.687 | FAIL |
| L2 | `trsvLT` | 2.14 (1.36) | 1.06 (0.79) | 0.795 | FAIL |
| L2 | `spmv` | 1.55 (1.05) | 1.05 (0.67) | 0.669 | FAIL |
| L2 | `gbmvN` | 0.49 (0.47) | 0.46 (0.40) | 0.402 | FAIL |
| L2 | `sbmv` | 3.08 (3.03) | 0.78 (0.69) | 0.689 | FAIL |
| L3 | `gemm` | 0.70 (0.66) | 0.10 (0.08) | 0.084 | FAIL |
| L3 | `symm` | 0.81 (0.67) | 0.12 (0.08) | 0.084 | FAIL |
| L3 | `syrk` | 0.94 (0.83) | 0.13 (0.11) | 0.112 | FAIL |
| L3 | `syr2k` | 0.92 (0.81) | 0.13 (0.12) | 0.120 | FAIL |
| L3 | `trmm` | 0.76 (0.61) | 0.13 (0.09) | 0.091 | FAIL |
| L3 | `trmmR` | 0.70 (0.51) | 0.14 (0.08) | 0.085 | FAIL |
| L3 | `trsm` | 0.93 (0.68) | 0.24 (0.13) | 0.127 | FAIL |
| L3 | `trsmR` | 1.15 (0.93) | 0.29 (0.18) | 0.176 | FAIL |
| LP | `potrf` | 1.58 (0.98) | 0.51 (0.17) | 0.173 | FAIL |
| LP | `getrf` | 0.93 (0.71) | 0.61 (0.19) | 0.194 | FAIL |
| LP | `geqrf` | 1.29 (0.78) | 0.98 (0.18) | 0.180 | FAIL |
| LP | `gesvd` | 1.04 (0.97) | 0.80 (0.23) | 0.225 | FAIL |

Full machine-readable tables: [`gen_table.md`](assets/apple/gen_table.md) (vs OpenBLAS),
[`gen_table_accelerate.md`](assets/apple/gen_table_accelerate.md) (vs Accelerate).

## Plots

Ratio-vs-size, same style as the [main fleet pages](performance.md) (dashed line = parity, band = q10-q90
spread). "gate" divides by whichever reference is faster at each point.

![BLAS-1 vs gate](assets/apple/perf_l1_gate.svg)
![BLAS-2 vs gate](assets/apple/perf_l2_gate.svg)
![BLAS-3 vs gate](assets/apple/perf_l3_gate.svg)
![LAPACK vs gate](assets/apple/perf_lapack_gate.svg)
![BLAS-1 vs OpenBLAS](assets/apple/perf_l1.svg)
![BLAS-2 vs OpenBLAS](assets/apple/perf_l2.svg)
![BLAS-3 vs OpenBLAS](assets/apple/perf_l3.svg)
![LAPACK vs OpenBLAS](assets/apple/perf_lapack.svg)
![BLAS-1 vs Accelerate](assets/apple/perf_l1_accelerate.svg)
![BLAS-2 vs Accelerate](assets/apple/perf_l2_accelerate.svg)
![BLAS-3 vs Accelerate](assets/apple/perf_l3_accelerate.svg)
![LAPACK vs Accelerate](assets/apple/perf_lapack_accelerate.svg)

## Reproduce

```
julia --project=bench/apple bench/plots.jl bench group=L1 arms=pb,openblas,accelerate cold nodraw
julia --project=bench/apple bench/plots.jl bench group=L2 arms=pb,openblas,accelerate maxsize=2048 nodraw
julia --project=bench/apple bench/plots.jl bench group=L3 arms=pb,openblas,accelerate maxsize=2048 nodraw
julia --project=bench/apple bench/plots.jl bench op=potrf arms=pb,openblas,accelerate maxsize=2048 nodraw
julia --project=bench/apple bench/plots.jl bench op=getrf arms=pb,openblas,accelerate maxsize=2048 nodraw
julia --project=bench/apple bench/plots.jl bench op=geqrf arms=pb,openblas,accelerate maxsize=2048 nodraw
julia --project=bench/apple bench/plots.jl bench op=gesvd arms=pb,openblas,accelerate maxsize=2048 nodraw
julia --project=bench/apple bench/plots.jl outdir=docs/src/assets/apple
```

`bench/apple/Project.toml` is a separate bench environment from `bench/Project.toml`: it drops
`AOCL`/`AOCL_jll` (AMD-only, no aarch64-apple-darwin build) since Accelerate needs no package at all — it
is an OS framework, forwarded by raw dylib path exactly like OpenBLAS.
