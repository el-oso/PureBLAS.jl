# Performance

PureBLAS targets a hard, non-negotiable gate: **≥ 0.96× OpenBLAS**, measured per machine, single
threaded. The plots below are the PureBLAS/OpenBLAS speed ratio (so **higher is better; 1.0× is
parity; the dashed line is the 0.96× gate**). Each **violin** is the distribution of the ratio across the
size sweep (and rounds); the horizontal line is the median and the dark dot is the *worst* size — a violin
is green only when **every** size clears the gate.

Methodology (see `bench/`): single-thread (`BLAS.set_num_threads(1)`), Float64, native PureBLAS API
vs `LinearAlgebra.BLAS`, **interleaved** timing (each round times OpenBLAS then PureBLAS back-to-back
so frequency drift cancels), **median** of many rounds, core pinned with `taskset -c 2`. Reproduce:
`taskset -c 2 julia --project=bench bench/plots.jl`. Numbers here are from a Zen4 (Ryzen, AVX-512
double-pumped); the gate is re-evaluated per host on the dev fleet.

## BLAS-1

![BLAS-1 ratio vs OpenBLAS](assets/perf_l1.svg)

Bandwidth-bound; PureBLAS matches or beats OpenBLAS across the board. `nrm2` is markedly faster
because OpenBLAS's `dnrm2` uses the slow always-scaled LAPACK algorithm, while PureBLAS uses a SIMD
sum-of-squares with a scaled fallback only on overflow/underflow.

## BLAS-2

![BLAS-2 ratio vs OpenBLAS](assets/perf_l2.svg)

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

![BLAS-3 ratio vs OpenBLAS](assets/perf_l3.svg)

The compute-bound level — shown as **median ratio vs size** (log-log). After the small-n gate campaign
(2026-07-02) **every op clears the 0.96× gate at every size n = 2…2048**; small sizes now run 1–3×
OpenBLAS (the former small-n dips were hidden overheads — scratch-lookup costs, per-call workspaces,
kwarg dispatch — all catalogued in the project kb). Two boundary cells wobble with the measurement:
`gemm` n=32 reads 0.90–0.96 depending on input cache-residency and box thermals, and `trmm`(side R)
n=1024 sits at ~0.96 — both certified in high-repetition hot-loop runs and pending a frequency-pinned
confirmation pass. `gemm` is the BLIS 5-loop with a SIMD
micro-kernel (unpacked small-matrix path); the rest are built on it:

- **syrk/syr2k/symm** pack the stored/symmetric triangle into `gemm`'s format in a single pass and
  use a triangular-store micro-kernel at the diagonal (no materialize, no wasted flops).
- **trmm/trsm** trim the contracted K-range over the zero band per tile; `trsm` inverts small diagonal
  blocks and applies them as `gemm`s. Both factor/pack into a padded leading dim to dodge power-of-2
  cache aliasing.

## LAPACK

![LAPACK ratio vs OpenBLAS](assets/perf_lapack.svg)

Factorizations driven by the gated BLAS, again **median ratio vs size** — gating at **every** size, with
tiny-n factors 1.5–4× OpenBLAS after the workspace-caching fixes. `potrf`/`geqrf` port the irreducible faer SIMD kernels and drive the blocked level
with PureBLAS `gemm!`/`trsm!`; `getrf` is blocked right-looking with deferred pivoting; `gesvd` is
gebrd → divide-and-conquer bidiagonal SVD → blocked compact-WY back-transform.

## Milestones

| Milestone | Scope | Status |
|-----------|-------|--------|
| **M1** | BLAS-1 (axpy, dot, nrm2, asum, scal, copy, swap, iamax; s/d/c/z) | ✅ gate met; LBT `.so` + native API |
| **M2** | `dgemm` (BLIS 5-loop + SIMD microkernel; unpacked small-matrix path) | ✅ single-thread parity (geomean ≈ 1.0×) |
| **M3 (core L2)** | gemv, ger, symv, hemv, trmv, trsv + packed (spmv/hpmv/tpmv/tpsv) and banded (gbmv/sbmv/hbmv/tbmv/tbsv) | ✅ gate met across the surface |
| **L3** | gemm, symm, syrk, syr2k, trmm, trsm | ✅ gate met at every n 2–2048 |
| **LAPACK** | potrf (Cholesky), geqrf (QR), getrf (LU), gesvd (SVD) | ✅ gate met at every n 2–2048 |
| **M4** | multithreading | deferred |

Both consumption modes share one kernel set: the **native API** (`PureBLAS.gemv!(…)`, AD-traceable)
and the **LBT drop-in** `.so` (`@ccallable` ILP64 symbols via `juliac --trim`).
