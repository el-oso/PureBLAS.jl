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
| `gemvN` | 0.86 (0.48) |
| `gemvT` | 0.74 (0.34) |
| `ger` | 0.37 (0.25) |
| `symv` | 1.59 (0.94) |
| `trmv` | 0.72 (0.24) |
| `trsv` | 0.92 (0.69) |
| `trsvLN` | 0.85 (0.69) |
| `trsvLT` | 1.04 (0.79) |
| `spmv` | 1.05 (0.67) |
| `gbmvN` | 0.46 (0.40) |
| `sbmv` | 0.77 (0.68) |
| `gemm` | 0.92 (0.23) |
| `symm` | 0.76 (0.25) |
| `syrk` | 0.57 (0.15) |
| `syr2k` | 0.70 (0.27) |
| `trmm` | 0.23 (0.12) |
| `trmmR` | 0.22 (0.10) |
| `trsm` | 0.52 (0.27) |
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

