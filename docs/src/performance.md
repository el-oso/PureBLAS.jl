# Performance

The gate: **≥ 1.0× OpenBLAS, per machine, single-threaded** (0.96× is the older floor and is
still drawn on the plots). All plots show the PureBLAS/OpenBLAS speed ratio — higher is better,
1.0 is parity.

The dev fleet is three machines, gated independently (tuning for one µarch does not transfer):

- **Zen4** (`wintermute`) — double-pumped AVX-512; the primary tuning target.
- **Zen3** (`galen`) — AVX2 (16 ymm registers); the hardest target and current campaign focus.
- **Zen5** (`neuromancer`) — native 512-bit AVX-512; clears the AVX2 pain points but has its own
  disjoint residuals.

**How the plots read:** one panel per op, the three µarchs overlaid as ratio-vs-size curves.
The grey line is 1.0 parity (the gate), the dashed line the 0.96 floor; the band on each curve
is the q10–q90 spread of the pooled per-round ratios.

**Methodology** (`bench/plots.jl`): single-thread (`BLAS.set_num_threads(1)`), native PureBLAS
API vs `LinearAlgebra.BLAS`, Float64 plus the full ComplexF64 surface. Each (op, size) is
measured over repeated rounds of ABBA-alternated windows; per-round ratios are pooled and the
median is the reported number. Runs are only valid at locked frequency — a floating boost clock
drifts between the two windows and fabricates ratios. Reproduce:

```
sudo bench/fleet_freqlock.sh lock       # passive governor, boost off, pin to base clock
taskset -c N julia --project=bench bench/plots.jl bench
```

## Real: BLAS-1 / BLAS-2 / BLAS-3 / LAPACK

![BLAS-1 — PureBLAS/OpenBLAS ratio per op, three µarchs](assets/perf_l1.svg)

Bandwidth-bound; at parity fleet-wide (worst sizes ≥ 0.98, except `iamax` on AVX2 at 0.95).
`nrm2` runs 7–10× because OpenBLAS uses the always-scaled LAPACK algorithm; PureBLAS scales
only on overflow/underflow.

![BLAS-2 — PureBLAS/OpenBLAS ratio per op, three µarchs](assets/perf_l2.svg)

Gates fleet-wide with two exceptions, both on Zen5: `gemvN` (mid-n native-512 residual, worst
0.90) and `trmv`/`trsv` at large n (DRAM regime, worst 0.98–0.99). `spmv` is flat ≈ 1.9–2.2
across the fleet (AP-residency packed panel); `ger` sits at gate on all three boxes (worst
0.97–1.06) with a per-µarch-calibrated write-stream count.

![BLAS-3 — PureBLAS/OpenBLAS ratio per op, three µarchs](assets/perf_l3.svg)

`gemm` gates every size on all three boxes (Strassen–Winograd at large n runs 1.2–1.4×). The
triangular/symmetric ops gate on AVX-512; on AVX2 the worst sizes of `trmm` (0.84) and `trsm`
(0.89) are still open.

![LAPACK — PureBLAS/OpenBLAS ratio per op, three µarchs](assets/perf_lapack.svg)

`potrf`/`geqrf`/`getrf`/`gesvd` gate on all three boxes (worst-size ≥ 0.99, geomeans 1.15–1.49),
driven by the gated BLAS-3. The current campaign pushes `potrf` beyond 1.2× at every size on
AVX2, via its `trsm`-R and `syrk` components.

## Complex (ComplexF64): CL1 / CL2 / CL3 / complex LAPACK

The complex surface is SIMD across all levels, in portable SIMD.jl kernels (no x86 intrinsics);
the generic scalar path remains for AD.

![Complex BLAS-1 — three µarchs](assets/perf_cl1.svg)

At parity or better fleet-wide, except worst-size dips in `zdotc`/`zdotu` on AVX2 (0.79) and
`zaxpy` (0.94–0.96). `dznrm2` is 7–9× (same scaled-accumulation story as real `nrm2`);
`izamax` 1.2–1.8×.

![Complex BLAS-2 — three µarchs](assets/perf_cl2.svg)

Gates broadly on all three boxes; the residuals are worst-size dips in the 0.87–0.95 range
(`zgemvT`/`zgemvC` on AVX-512, `ztrmv` Zen4, `zhpmv` Zen5 — `zhpmv` has no packed panel yet).

![Complex BLAS-3 — three µarchs](assets/perf_cl3.svg)

`zgemm` beats OpenBLAS fleet-wide (geomean 1.26–1.40; Karatsuba 3M at mid/large n). The
rank-k ops gate within a few percent; the open items are AVX2 worst sizes: `ztrmm` 0.78,
`ztrmmR` 0.75, `ztrsm` 0.89, and a `zhemm`/`zsymm` small-n dip (0.91–0.94).

![Complex LAPACK — three µarchs](assets/perf_clapack.svg)

The weak group. `zpotrf` gates fleet-wide and `zgetrf` is close (Zen3 worst 0.80), but
`zgeqrf` is well under gate (geomean 0.76–0.85), and `zgesvd` — singular values only, benched
against `zgesdd('N')` — collapses at large n because its bidiagonalization is still unblocked
BLAS-2 (panel capped at n=1024). Both are open work, not measurement artifacts.

## Numeric summary

The per-op numbers — **geomean (worst-size) ratio per op per µarch** — live in
[`bench/gen_table.md`](https://github.com/el-oso/PureBLAS.jl/blob/master/bench/gen_table.md).
That file is auto-generated from the fleet result caches by the same run that produces the
plots, with a provenance header (CPU model, code commit, timestamp per box), so it cannot
drift from the plots. It is the numeric source of truth; numbers are deliberately not
duplicated here.

## Where we are

**Zen4 (AVX-512, double-pumped).** The tuning target; gates essentially everywhere. Real
residuals are worst-size only (`syrk` 0.94, `syr2k` 0.97, `trmm` 0.97 — geomeans all ≥ 1.07).
Complex residuals are the shared LAPACK gaps plus small `zgemvT`/`zgemvC`/`ztrmv` worst-size
dips.

**Zen5 (AVX-512, native 512-bit).** Clears every AVX2 ceiling but shows a disjoint residual
profile — the reason the gate is per-machine. Open: `gemvN` mid-n (~0.90; the m-inner panel
that fixed Zen3/Zen4 regressed here and is gated off) and `trmv`/`trsv` in the DRAM regime at
n=4096.

**Zen3 (AVX2).** The hardest target: 16 ymm registers vs AVX-512's 32 zmm. Real surface gates
except `trmm`/`trsm` worst sizes; complex carries the widest residual set (`zdot`, `ztrmm`
both sides, `ztrsm`, `zgetrf`). The active campaign is `potrf` > 1.2× at every size, AVX2
first, through its `potf2`/`trsm`/`syrk`/`syr2k` components.

**Known open items** (tracked in [`ROADMAP.md`](https://github.com/el-oso/PureBLAS.jl/blob/master/ROADMAP.md)):

- `gemvN` Zen5 mid-n (~0.90): needs a native-512 lever; no config fix found.
- `trmv`/`trsv` Zen5 n=4096 just under parity, and the Zen3 L2→L3 blocking edge at n=512.
- `hpmv` still per-column — port the spmv AP-residency panel to complex.
- `trsm` side-R is not yet a gated op in `plots.jl` (its numbers were never gate-measured).
- Complex LAPACK: `zgeqrf` under gate fleet-wide; `zgesvd` blocked-bidiagonalization port
  pending (values-only, capped at n=1024).
- Tuning-constant debt: several block-size literals remain to be re-derived as formulas over
  detected cache/register parameters.

Both consumption modes share one kernel set: the **native API** (`PureBLAS.gemm!(…)`,
AD-traceable) and the **LBT drop-in** `.so` (`@ccallable` ILP64 symbols via `juliac --trim`).
