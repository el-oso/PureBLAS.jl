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
| `gemvN` | 7.24 (1.85) |
| `gemvT` | 2.03 (1.67) |
| `ger` | 2.54 (0.99) |
| `symv` | 1.16 (0.74) |
| `trmv` | 2.51 (1.78) |
| `trsv` | 1.63 (1.37) |
| `trsvLN` | 1.43 (1.30) |
| `trsvLT` | 2.13 (1.32) |
| `spmv` | 1.57 (1.04) |
| `gbmvN` | 0.84 (0.80) |
| `sbmv` | 3.05 (3.01) |
| `gemm` | 6.93 (2.32) |
| `symm` | 4.55 (1.15) |
| `syrk` | 4.01 (0.84) |
| `syr2k` | 4.95 (1.31) |
| `trmm` | 1.63 (0.64) |
| `trmmR` | 2.74 (0.58) |
| `trsm` | 2.31 (1.09) |
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

