PB / Accelerate speed ratio, median (worst cell) per op per µarch. Provenance: [`bench/provenance.md`](provenance.md).

### Real

| op | ARM · NEON |
|---|---|
| `dot` | 1.12 (0.43) |
| `axpy` | 1.19 (0.56) |
| `nrm2` | 1.95 (1.67) |
| `asum` | 1.00 (0.53) |
| `scal` | 0.92 (0.35) |
| `iamax` | 0.93 (0.90) |
| `gemvN` | 0.17 (0.10) |
| `gemvT` | 0.34 (0.16) |
| `ger` | 0.37 (0.24) |
| `symv` | 1.19 (0.33) |
| `trmv` | 0.67 (0.16) |
| `trsv` | 0.94 (0.68) |
| `trsvLN` | 0.85 (0.67) |
| `trsvLT` | 1.06 (0.84) |
| `spmv` | 1.05 (0.67) |
| `gbmvN` | 0.46 (0.40) |
| `sbmv` | 0.78 (0.69) |
| `gemm` | 0.10 (0.07) |
| `symm` | 0.12 (0.08) |
| `syrk` | 0.13 (0.10) |
| `syr2k` | 0.13 (0.11) |
| `trmm` | 0.14 (0.08) |
| `trmmR` | 0.13 (0.08) |
| `trsm` | 0.24 (0.11) |
| `trsmR` | 0.29 (0.15) |
| `potrf` | 0.50 (0.17) |
| `getrf` | 0.61 (0.19) |
| `geqrf` | 0.98 (0.18) |
| `gesvd` | 0.80 (0.23) |


### Complex

| op | ARM · NEON |
|---|---|


### Dual (ForwardDiff, N=1) — reference is LinearAlgebra generic, NOT a vendor BLAS

| op | ARM · NEON |
|---|---|

