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

**Machine:** Apple M6 (ARM · NEON, `_vwidth(Float64)=2`), unlocked clock, commit `5c960578`, measured
2026-09-22. Full provenance: [`provenance.md`](assets/apple/provenance.md).

## Headline

Of the 29 real (Float64) cells measured, **1 gates** (`nrm2`, where OpenBLAS's always-scaled algorithm is
slow on every platform PureBLAS has been measured on). PureBLAS is competitive with **OpenBLAS** on this
first pass — roughly at parity on BLAS-1/2 and trailing modestly on BLAS-3/LAPACK (0.7-1.6×, median) — but
well behind **Accelerate** almost everywhere, especially on BLAS-3 (0.08-0.3×). That gap tracks Accelerate
routing through Apple's AMX matrix coprocessor, which pure-Julia NEON code (128-bit, 2 Float64 lanes) has
no access to; closing it is a tuning/architecture question for a future session, not something this
data-generation pass attempts.

**Widening L1 mattered.** At n=1000 alone, `dot`/`axpy`/`scal` looked competitive with Accelerate
(1.04×/0.41×/0.57×); across the full 1e3..1e6 ladder the picture is starker (0.18×/0.23×/0.29× median) —
Accelerate's advantage grows with problem size on bandwidth-bound ops too.

## vs OpenBLAS and Accelerate, per op

Ratio is PB / reference, **median (worst cell)** across the measured size ladder (capped at n=2048 for
BLAS-2/3/LAPACK; full 1e3..1e6 for BLAS-1). Gate is PB / max(OpenBLAS, Accelerate) — the same
two-significant-digit rounding rule as the [main fleet](methodology.md#the-gate) (`bench/gatecrit.jl`).

| level | op | vs OpenBLAS | vs Accelerate | gate | verdict |
|---|---|---|---|---|---|
| L1 | `dot` | 1.01 (0.98) | 0.18 (0.14) | 0.138 | FAIL |
| L1 | `axpy` | 1.00 (0.97) | 0.23 (0.22) | 0.225 | FAIL |
| L1 | `nrm2` | 8.26 (7.89) | 1.97 (1.88) | 1.882 | **PASS** |
| L1 | `asum` | 1.01 (0.99) | 0.56 (0.45) | 0.447 | FAIL |
| L1 | `scal` | 0.99 (0.98) | 0.29 (0.28) | 0.280 | FAIL |
| L1 | `iamax` | 0.45 (0.43) | 0.94 (0.94) | 0.427 | FAIL |
| L2 | `gemvN` | 1.52 (0.84) | 0.17 (0.10) | 0.102 | FAIL |
| L2 | `gemvT` | 1.51 (0.94) | 0.34 (0.16) | 0.162 | FAIL |
| L2 | `ger` | 1.00 (0.92) | 0.37 (0.24) | 0.241 | FAIL |
| L2 | `symv` | 0.74 (0.59) | 1.19 (0.33) | 0.331 | FAIL |
| L2 | `trmv` | 1.81 (1.35) | 0.67 (0.16) | 0.156 | FAIL |
| L2 | `trsv` | 1.65 (1.37) | 0.94 (0.68) | 0.679 | FAIL |
| L2 | `trsvLN` | 1.42 (1.29) | 0.85 (0.67) | 0.670 | FAIL |
| L2 | `trsvLT` | 2.13 (1.40) | 1.06 (0.84) | 0.838 | FAIL |
| L2 | `spmv` | 1.54 (1.04) | 1.05 (0.67) | 0.672 | FAIL |
| L2 | `gbmvN` | 0.49 (0.47) | 0.46 (0.40) | 0.402 | FAIL |
| L2 | `sbmv` | 3.03 (2.98) | 0.78 (0.69) | 0.689 | FAIL |
| L3 | `gemm` | 0.70 (0.66) | 0.10 (0.07) | 0.075 | FAIL |
| L3 | `symm` | 0.81 (0.67) | 0.12 (0.08) | 0.084 | FAIL |
| L3 | `syrk` | 0.94 (0.83) | 0.13 (0.10) | 0.101 | FAIL |
| L3 | `syr2k` | 0.92 (0.81) | 0.13 (0.11) | 0.109 | FAIL |
| L3 | `trmm` | 0.76 (0.61) | 0.14 (0.08) | 0.079 | FAIL |
| L3 | `trmmR` | 0.70 (0.51) | 0.13 (0.08) | 0.077 | FAIL |
| L3 | `trsm` | 0.93 (0.68) | 0.24 (0.11) | 0.107 | FAIL |
| L3 | `trsmR` | 1.15 (0.93) | 0.29 (0.15) | 0.149 | FAIL |
| LP | `potrf` | 1.58 (0.99) | 0.50 (0.17) | 0.173 | FAIL |
| LP | `getrf` | 0.93 (0.71) | 0.61 (0.19) | 0.195 | FAIL |
| LP | `geqrf` | 1.28 (0.78) | 0.98 (0.18) | 0.180 | FAIL |
| LP | `gesvd` | 1.04 (0.97) | 0.80 (0.23) | 0.228 | FAIL |

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
julia --project=bench/apple bench/plots.jl bench group=L1 arms=pb,openblas,accelerate maxsize=2048 nodraw
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
