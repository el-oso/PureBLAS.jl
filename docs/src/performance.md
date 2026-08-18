# Performance

Every plot is a **PB / reference speed ratio** against problem size — higher is better, the dashed line
is 1.0 parity — with one curve per microarchitecture (Zen3 · AVX2, Zen4 · AVX-512, Zen5 · AVX-512) and
one panel per op. The band around each curve is the q10–q90 spread of the pooled per-round ratios.
Single-threaded, Float64 and ComplexF64.

Per-op numbers: [`bench/gen_table.md`](https://github.com/el-oso/PureBLAS.jl/blob/master/bench/gen_table.md)
(vs OpenBLAS) and [`bench/gen_table_aocl.md`](https://github.com/el-oso/PureBLAS.jl/blob/master/bench/gen_table_aocl.md)
(vs AOCL). Measurement, provenance and how to read a cell: [Methodology](methodology.md).
Per-routine analysis and history: [Notes](notes.md).

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

Both consumption modes share one kernel set: the **native API** (`PureBLAS.gemm!(…)`, AD-traceable) and
the **LBT drop-in** `.so` (`@ccallable` ILP64 symbols via `juliac --trim`).
