PB / OpenBLAS speed ratio, median (worst cell) per op per µarch. Provenance: [`bench/provenance.md`](provenance.md).

### Real

| op | ARM · NEON |
|---|---|
| `axpy` | 1.17 (1.00) |
| `dot` | 1.10 (1.00) |
| `nrm2` | 8.29 (7.00) |
| `asum` | 1.10 (1.00) |
| `scal` | 1.00 (0.99) |
| `iamax` | 0.99 (0.50) |
| `gemvN` | 7.47 (1.92) |
| `gemvT` | 2.14 (1.76) |
| `ger` | 2.54 (0.99) |
| `symv` | 1.16 (0.74) |
| `trmv` | 2.00 (1.73) |
| `trsv` | 1.63 (1.37) |
| `trsvLN` | 1.43 (1.30) |
| `trsvLT` | 2.13 (1.32) |
| `spmv` | 1.57 (1.04) |
| `gbmvN` | 0.84 (0.80) |
| `sbmv` | 3.05 (3.01) |
| `gemm` | 6.76 (0.98) |
| `symm` | 4.65 (0.93) |
| `syrk` | 4.12 (0.83) |
| `syr2k` | 4.95 (1.05) |
| `trmm` | 1.07 (0.55) |
| `trmmR` | 1.00 (0.45) |
| `trsm` | 1.87 (1.02) |
| `trsmR` | 1.82 (1.27) |
| `potrf` | 1.88 (1.42) |
| `getrf` | 2.12 (1.08) |
| `geqrf` | 1.25 (1.14) |
| `gesvd` | 1.76 (1.00) |
| `potrfU` | – |
| `getrs` | – |
| `potrsL` | – |
| `potrsU` | – |
| `trtrs` | – |
| `sytrf` | – |
| `sytrs` | – |
| `potri` | – |
| `trtri` | – |
| `getri` | – |
| `sytri` | – |
| `gelsy` | – |
| `gelsd` | – |
| `geev` | – |
| `gbtrf` | – |
| `geqp3` | – |
| `gels` | – |
| `pstrf` | – |
| `pstrfU` | – |
| `syev` | – |
| `syevN` | – |
| `gtsv` | – |
| `gttrf` | – |
| `gttrs` | – |
| `pttrf` | – |
| `pttrs` | – |
| `ptsv` | – |
| `pbtrfL` | – |
| `pbtrfU` | – |
| `pptrfL` | – |
| `pptrfU` | – |


### Complex

| op | ARM · NEON |
|---|---|


### Dual (ForwardDiff, N=1) — reference is LinearAlgebra generic, NOT a vendor BLAS

| op | ARM · NEON |
|---|---|

