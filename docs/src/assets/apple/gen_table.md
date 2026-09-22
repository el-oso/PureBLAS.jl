PB / OpenBLAS speed ratio, median (worst cell) per op per µarch. Provenance: [`bench/provenance.md`](provenance.md).

### Real

| op | ARM · NEON |
|---|---|
| `dot` | 1.00 (1.00) |
| `axpy` | 1.01 (1.00) |
| `nrm2` | 8.42 (7.00) |
| `asum` | 1.01 (1.00) |
| `scal` | 0.99 (0.99) |
| `iamax` | 0.50 (0.49) |
| `gemvN` | 1.52 (0.84) |
| `gemvT` | 1.51 (0.94) |
| `ger` | 1.00 (0.92) |
| `symv` | 0.74 (0.59) |
| `trmv` | 1.81 (1.35) |
| `trsv` | 1.65 (1.37) |
| `trsvLN` | 1.42 (1.29) |
| `trsvLT` | 2.13 (1.40) |
| `spmv` | 1.54 (1.04) |
| `gbmvN` | 0.49 (0.47) |
| `sbmv` | 3.03 (2.98) |
| `gemm` | 0.70 (0.66) |
| `symm` | 0.81 (0.67) |
| `syrk` | 0.94 (0.83) |
| `syr2k` | 0.92 (0.81) |
| `trmm` | 0.76 (0.61) |
| `trmmR` | 0.70 (0.51) |
| `trsm` | 0.93 (0.68) |
| `trsmR` | 1.15 (0.93) |
| `potrf` | 1.58 (0.99) |
| `getrf` | 0.93 (0.71) |
| `geqrf` | 1.28 (0.78) |
| `gesvd` | 1.04 (0.97) |


### Complex

| op | ARM · NEON |
|---|---|


### Dual (ForwardDiff, N=1) — reference is LinearAlgebra generic, NOT a vendor BLAS

| op | ARM · NEON |
|---|---|

