PB / OpenBLAS speed ratio, median (worst cell) per op per µarch. Provenance: [`bench/provenance.md`](provenance.md).

### Real

| op | ARM · NEON |
|---|---|
| `axpy` | 1.02 (1.00) |
| `dot` | 1.00 (0.99) |
| `nrm2` | 8.37 (7.00) |
| `asum` | 1.02 (1.00) |
| `scal` | 0.99 (0.99) |
| `iamax` | 0.50 (0.49) |
| `gemvN` | 1.52 (0.84) |
| `gemvT` | 1.51 (0.94) |
| `ger` | 1.00 (0.98) |
| `symv` | 0.74 (0.59) |
| `trmv` | 1.80 (1.36) |
| `trsv` | 1.65 (1.37) |
| `trsvLN` | 1.42 (1.32) |
| `trsvLT` | 2.14 (1.36) |
| `spmv` | 1.55 (1.05) |
| `gbmvN` | 0.49 (0.47) |
| `sbmv` | 3.08 (3.03) |
| `gemm` | 6.54 (0.70) |
| `symm` | 4.20 (0.71) |
| `syrk` | 0.99 (0.84) |
| `syr2k` | 0.96 (0.80) |
| `trmm` | 1.07 (0.56) |
| `trmmR` | 0.88 (0.45) |
| `trsm` | 1.53 (1.02) |
| `trsmR` | 1.82 (1.27) |
| `potrf` | 1.52 (1.11) |
| `getrf` | 2.14 (0.83) |
| `geqrf` | 1.29 (1.04) |
| `gesvd` | 1.56 (0.95) |
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

