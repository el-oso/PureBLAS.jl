PB / Accelerate speed ratio, median (worst cell) per op per µarch. Provenance: [`bench/provenance.md`](provenance.md).

### Real

| op | ARM · NEON |
|---|---|
| `axpy` | 1.22 (0.98) |
| `dot` | 1.19 (1.02) |
| `nrm2` | 1.92 (1.67) |
| `asum` | 1.03 (0.87) |
| `scal` | 0.98 (0.83) |
| `iamax` | 1.85 (1.00) |
| `gemvN` | 0.92 (0.48) |
| `gemvT` | 0.94 (0.34) |
| `ger` | 0.95 (0.63) |
| `symv` | 1.72 (0.94) |
| `trmv` | 0.89 (0.45) |
| `trsv` | 0.93 (0.68) |
| `trsvLN` | 0.85 (0.69) |
| `trsvLT` | 1.04 (0.79) |
| `spmv` | 1.05 (0.68) |
| `gbmvN` | 0.80 (0.69) |
| `sbmv` | 0.78 (0.69) |
| `gemm` | 0.94 (0.82) |
| `symm` | 0.76 (0.39) |
| `syrk` | 0.57 (0.15) |
| `syr2k` | 0.70 (0.34) |
| `trmm` | 0.32 (0.20) |
| `trmmR` | 0.44 (0.17) |
| `trsm` | 0.55 (0.36) |
| `trsmR` | 0.57 (0.32) |
| `potrf` | 0.59 (0.34) |
| `getrf` | 1.42 (0.69) |
| `geqrf` | 0.94 (0.28) |
| `gesvd` | 1.22 (0.78) |
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

