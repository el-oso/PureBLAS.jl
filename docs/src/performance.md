# Performance

Each plot shows how fast PureBLAS is relative to a reference, against problem size. Higher is better and
the dashed line is parity, so anything above it is a win. There is one curve per microarchitecture
(Zen3 · AVX2, Zen4 · AVX-512, Zen5 · AVX-512) and one panel per operation, and the band around a curve
is the q10–q90 spread of the pooled per-round ratios. Everything is single-threaded, Float64 and
ComplexF64.

Roughly 87% of the measured cells are at or above `max(OpenBLAS, AOCL)`. The ones that are not are
listed in [Notes](notes.md), and most of them sit within a few percent.

For the numbers behind the plots, see
[`bench/gen_table.md`](https://github.com/el-oso/PureBLAS.jl/blob/master/bench/gen_table.md) against
OpenBLAS and
[`bench/gen_table_aocl.md`](https://github.com/el-oso/PureBLAS.jl/blob/master/bench/gen_table_aocl.md)
against AOCL. [Methodology](methodology.md) covers how the measurements are taken and how to read a
cell; [Notes](notes.md) has the per-routine analysis and history.

## vs OpenBLAS

![BLAS-1 — PB / OpenBLAS ratio per op, three µarchs](assets/perf_l1.svg)
![BLAS-2 — PB / OpenBLAS ratio per op, three µarchs](assets/perf_l2.svg)
![BLAS-3 — PB / OpenBLAS ratio per op, three µarchs](assets/perf_l3.svg)
![LAPACK — PB / OpenBLAS ratio per op, three µarchs](assets/perf_lapack.svg)
![Complex BLAS-1 — PB / OpenBLAS ratio per op, three µarchs](assets/perf_cl1.svg)
![Complex BLAS-2 — PB / OpenBLAS ratio per op, three µarchs](assets/perf_cl2.svg)
![Complex BLAS-3 — PB / OpenBLAS ratio per op, three µarchs](assets/perf_cl3.svg)
![Complex LAPACK — PB / OpenBLAS ratio per op, three µarchs](assets/perf_clapack.svg)

## vs AOCL

![BLAS-1 — PB / AOCL ratio per op, three µarchs](assets/perf_l1_aocl.svg)
![BLAS-2 — PB / AOCL ratio per op, three µarchs](assets/perf_l2_aocl.svg)
![BLAS-3 — PB / AOCL ratio per op, three µarchs](assets/perf_l3_aocl.svg)
![LAPACK — PB / AOCL ratio per op, three µarchs](assets/perf_lapack_aocl.svg)
![Complex BLAS-1 — PB / AOCL ratio per op, three µarchs](assets/perf_cl1_aocl.svg)
![Complex BLAS-2 — PB / AOCL ratio per op, three µarchs](assets/perf_cl2_aocl.svg)
![Complex BLAS-3 — PB / AOCL ratio per op, three µarchs](assets/perf_cl3_aocl.svg)
![Complex LAPACK — PB / AOCL ratio per op, three µarchs](assets/perf_clapack_aocl.svg)

These numbers apply however you call PureBLAS. The native API (`PureBLAS.gemm!(…)`), the in-process
reroute through `activate()`, and the `libpureblas.so` built for C hosts all run the same kernels.
