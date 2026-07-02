# Performance

PureBLAS targets a hard, non-negotiable gate: **≥ 0.96× OpenBLAS**, measured per machine, single
threaded. The plots below are the PureBLAS/OpenBLAS speed ratio (so **higher is better; 1.0× is
parity; the dashed line is the 0.96× gate**). Each **violin** is the distribution of the ratio across the
size sweep (and rounds); the horizontal line is the median and the dark dot is the *worst* size — a violin
is green only when **every** size clears the gate.

Methodology (see `bench/`): single-thread (`BLAS.set_num_threads(1)`), Float64 (plus ComplexF64 for
`zgemm`), native PureBLAS API vs `LinearAlgebra.BLAS`, **interleaved** timing (each round times OpenBLAS
then PureBLAS back-to-back so frequency drift cancels), **median** of many rounds, core pinned with
`taskset`, and — importantly — **CPU boost disabled** (a floating boost clock silently biases small-n
ratios; see below). Reproduce: `taskset -c 2 julia --project=bench bench/plots.jl`. Each section shows
**both** ISAs of the dev fleet — **AVX-512** (Zen4, the primary tuning target) and native **AVX2** (Zen3).

## Per-ISA gate (dev fleet)

The 0.96× gate is **per machine**. The fleet spans double-pumped **AVX-512** (Zen4, the tuning target)
and native **AVX2** (Zen3). Below is the full-stack `plots.jl bench` at pinned frequency, worst-size
ratio (the gate metric) with ✓ ≥ 0.96 / ✗ < 0.96; geomeans are given in text.

| Op | AVX-512 (Zen4) | AVX2 (Zen3) |
|----|:---:|:---:|
| L1 axpy · dot · asum · scal | ✓ 0.99–1.06 | ✓ 0.99–1.33 |
| L1 nrm2 (scaled-accum beats OpenBLAS) | ✓ 3.6× | ✓ 5.5× |
| L1 iamax | ✓ 1.15 | ✗ 0.90 |
| L2 gemvT · ger · symv · trmv · trsv · spmv · gbmv · sbmv | ✓ 1.0–1.6 | ✓ 0.97–1.64 |
| L2 gemvN | ✗ 0.96 | ✗ 0.87 |
| L3 gemm | ✗ 0.89 (n=8) | ✓ 1.01 |
| **L3 syrk** (branch-free scaleC + small-n unified pack) | ✓ 0.98 | ✓ 1.00 |
| **L3 zgemm (complex)** | **✓ 1.12** | ◐ 0.93 (n=32 cold; ~1.03 hot) |
| **L3 trsm** (vectorized trtri + 0-alloc) | ✓ 1.00 | ◐ 0.86 (geomean 0.95; was 0.65) |
| L3 syr2k · trmm · symm | ✓ 0.97 | ✗ 0.87–0.95 |
| LAPACK geqrf · gesvd | ✓ 1.03–1.21 | ✓ 1.07–1.08 |
| LAPACK potrf · getrf | ✓ 1.10 · ✓ 0.99 | ✗ 0.67 · 0.90 |

On **AVX-512** every op's **geomean** clears the gate (1.0–1.5×); the ✗ cells are small-n **worst-size**
dips only (n=8 dispatch / cold-cache — `gemm` geomean is still 1.03). On **AVX2**, BLAS-1/2, real `gemm`,
and now `syrk` gate. `syrk` was lifted from worst 0.79 to **1.00**: decomposition showed the *kernel*
already gated at every size (0.99–1.02), so the gap was overhead around it — a full-n² `_syrk_scaleC!`
pre-scale pass with a per-element branch (gemm folds β into its kernel), made branch-free and
stored-triangle-only (gates n≥128), plus routing small n through the unified single-pack whose halved
cold pack-traffic gates n=32 (0.90→1.19). `trsm` reached geomean **0.95** (worst 0.86) via the vectorized
`_trtri!` (solve `A·V=I` on the SIMD dense-L base) and removing a per-recursion-leaf boxing (a
non-concrete scratch return made the wide-B invL/invR `view` type-unstable). Complex `zgemm` sits at the
n=32 cold boundary (0.93 cold / **~1.03 hot** — cold small-complex is 2× the bytes and unrepresentative).
The remaining ✗ are the other **triangular/symmetric L3** ops
(`syr2k` 0.95 · `trmm` 0.87 · `symm` 0.90) and `potrf`/`getrf` built on them — the in-progress Zen3
small-n campaign (the same overhead-decomposition discipline that just gated `syrk` isn't fully ported to
those kernels yet; `syr2k` shares `syrk`'s scaleC fix and is closest at 0.95).
`zgemm` is a SIMD split-pack kernel that **beats OpenBLAS on AVX-512**.

> **Measurement note (learned the hard way):** with CPU boost enabled, allocating between timed regions
> drops the core off boost mid-measurement, biasing whichever side is timed first — this once fabricated
> a fake `zgemm` "n=32 hardware floor" that clean pinned-frequency measurement puts at 1.0–1.03. Always
> disable boost (`echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost`, `performance` governor)
> before trusting a gate number.

## BLAS-1

**AVX-512 (Zen4):**

![BLAS-1 ratio vs OpenBLAS, AVX-512](assets/perf_l1_avx512.svg)

**AVX2 (Zen3):**

![BLAS-1 ratio vs OpenBLAS, AVX2](assets/perf_l1_avx2.svg)

Bandwidth-bound; PureBLAS matches or beats OpenBLAS across the board. `nrm2` is markedly faster
because OpenBLAS's `dnrm2` uses the slow always-scaled LAPACK algorithm, while PureBLAS uses a SIMD
sum-of-squares with a scaled fallback only on overflow/underflow.

## BLAS-2

**AVX-512 (Zen4):**

![BLAS-2 ratio vs OpenBLAS, AVX-512](assets/perf_l2_avx512.svg)

**AVX2 (Zen3):**

![BLAS-2 ratio vs OpenBLAS, AVX2](assets/perf_l2_avx2.svg)

Matrix-vector and the packed/banded variants. The headline lessons (full detail in the project's
`kb/` findings):

- **gemv** is column-major, so `gemv-N` is transpose-like — a size-dispatched **column-panel** kernel
  cuts the y-restream; `gemv-T` is a column-block sharing x-chunks.
- **symv** reads only half of A, so the vector re-stream costs more — a **fused panel** does the
  symmetric `gemv-N + gemv-T` in one pass with the triangular diagonal folded into the same
  accumulators.
- **trmv/trsv** are blocked at large n (diagonal block + off-diagonal `gemv`, reading A once).
- **packed/banded** reuse the same per-column kernels (packed and band columns are contiguous);
  `gbmv` uses convolution-style kernels with BLASFEO-style register reuse for wide bands.

## BLAS-3

**AVX-512 (Zen4):**

![BLAS-3 ratio vs OpenBLAS, AVX-512](assets/perf_l3_avx512.svg)

**AVX2 (Zen3):**

![BLAS-3 ratio vs OpenBLAS, AVX2](assets/perf_l3_avx2.svg)

The compute-bound level — **median ratio vs size** (log-log). On **AVX-512**
every op's geomean clears the 0.96× gate and small sizes run 1–3× OpenBLAS (the former small-n dips were
hidden overheads — scratch-lookup costs, per-call workspaces, kwarg dispatch — all catalogued in the
project kb). A few small-n **worst-size** cells still dip below the strict gate (`gemm` n=8 ≈ 0.87 from
dispatch/cold-cache; `symm`, `gemvN` ≈ 0.96) — geomeans stay ≥1.0. On **AVX2** (see the fleet table
above) `gemm` and complex `zgemm` gate, but the triangular/symmetric ops still carry a Zen3 small-n
gap. `gemm` is the BLIS 5-loop with a SIMD micro-kernel (unpacked small-matrix path); `zgemm` is a
complex split-pack kernel (real+imag panels, 4-real-FMA MAC); the rest are built on `gemm`:

- **syrk/syr2k/symm** pack the stored/symmetric triangle into `gemm`'s format in a single pass and
  use a triangular-store micro-kernel at the diagonal (no materialize, no wasted flops).
- **trmm/trsm** trim the contracted K-range over the zero band per tile; `trsm` inverts small diagonal
  blocks and applies them as `gemm`s. Both factor/pack into a padded leading dim to dodge power-of-2
  cache aliasing.

## LAPACK

**AVX-512 (Zen4):**

![LAPACK ratio vs OpenBLAS, AVX-512](assets/perf_lapack_avx512.svg)

**AVX2 (Zen3):**

![LAPACK ratio vs OpenBLAS, AVX2](assets/perf_lapack_avx2.svg)

Factorizations driven by the gated BLAS, again **median ratio vs size** for **Zen4/AVX-512** — gating at
every size, with tiny-n factors 1.5–4× OpenBLAS after the workspace-caching fixes. (On AVX2, `geqrf`/
`gesvd` gate; `potrf`/`getrf` inherit the below-gate triangular L3 kernels — see the fleet table.)
`potrf`/`geqrf` port the irreducible faer SIMD kernels and drive the blocked level
with PureBLAS `gemm!`/`trsm!`; `getrf` is blocked right-looking with deferred pivoting; `gesvd` is
gebrd → divide-and-conquer bidiagonal SVD → blocked compact-WY back-transform.

## Milestones

| Milestone | Scope | Status |
|-----------|-------|--------|
| **M1** | BLAS-1 (axpy, dot, nrm2, asum, scal, copy, swap, iamax; s/d/c/z) | ✅ gate met; LBT `.so` + native API |
| **M2** | `dgemm` (BLIS 5-loop + SIMD microkernel; unpacked small-matrix path) | ✅ single-thread parity (geomean ≈ 1.0×) |
| **M3 (core L2)** | gemv, ger, symv, hemv, trmv, trsv + packed (spmv/hpmv/tpmv/tpsv) and banded (gbmv/sbmv/hbmv/tbmv/tbsv) | ✅ gate met across the surface |
| **L3** | gemm, symm, syrk, syr2k, trmm, trsm | ✅ AVX-512 gates all n; AVX2 gates gemm + syrk, other triangular ops WIP |
| **L3 complex** | zgemm (ComplexF64 split-pack SIMD) | ✅ beats OpenBLAS on AVX-512; gates on AVX2 |
| **LAPACK** | potrf (Cholesky), geqrf (QR), getrf (LU), gesvd (SVD) | ✅ AVX-512 gates all n; AVX2 geqrf/gesvd gate |
| **M4** | multithreading | deferred |

Both consumption modes share one kernel set: the **native API** (`PureBLAS.gemv!(…)`, AD-traceable)
and the **LBT drop-in** `.so` (`@ccallable` ILP64 symbols via `juliac --trim`).
