PB / Accelerate speed ratio, median (worst cell) per op per µarch. Provenance: [`bench/provenance.md`](provenance.md).

### Real

| op | ARM · NEON |
|---|---|
| `axpy` | 1.04 (0.65) |
| `dot` | 1.00 (0.96) |
| `nrm2` | 2.09 (1.67) |
| `asum` | 1.00 (0.51) |
| `scal` | 0.85 (0.38) |
| `iamax` | 1.13 (0.99) |
| `blascopy` | – |
| `swap` | – |
| `gemvN` | 0.78 (0.49) |
| `gemvT` | 0.83 (0.32) |
| `ger` | 0.87 (0.58) |
| `symv` | 1.71 (0.83) |
| `trmv` | 0.62 (0.36) |
| `trsv` | 0.80 (0.65) |
| `trsvLN` | 0.79 (0.65) |
| `trsvLT` | 1.00 (0.79) |
| `spmv` | 1.02 (0.62) |
| `gbmvN` | 0.73 (0.65) |
| `sbmv` | 0.71 (0.62) |
| `gemvNrect` | – |
| `gemvTrect` | – |
| `gemm` | 0.90 (0.69) |
| `symm` | 0.56 (0.42) |
| `syrk` | 0.47 (0.14) |
| `syr2k` | 0.63 (0.33) |
| `trmm` | 0.32 (0.16) |
| `trmmR` | 0.39 (0.14) |
| `trsm` | 0.50 (0.32) |
| `trsmR` | 0.56 (0.43) |
| `trsmRrect` | – |
| `trsmRband` | – |
| `potrf` | 0.55 (0.36) |
| `getrf` | 1.11 (0.52) |
| `geqrf` | 0.98 (0.26) |
| `gesvd` | 1.24 (0.88) |
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
| `zaxpy` | – |
| `zdotc` | – |
| `zscal` | – |
| `zscalc` | – |
| `dznrm2` | – |
| `dzasum` | – |
| `zdotu` | – |
| `izamax` | – |
| `zgemvN` | – |
| `zgemvT` | – |
| `zgemvC` | – |
| `zgeru` | – |
| `zhemv` | – |
| `ztrmv` | – |
| `ztrsv` | – |
| `zhpmv` | – |
| `zgbmvN` | – |
| `zhbmv` | – |
| `zgemm` | – |
| `zhemm` | – |
| `zsymm` | – |
| `zsyrk` | – |
| `zherk` | – |
| `zher2k` | – |
| `zsyr2k` | – |
| `ztrmm` | – |
| `ztrsm` | – |
| `ztrmmR` | – |
| `ztrsmR` | – |
| `zpotrf` | – |
| `zpotrfU` | – |
| `zgeqrf` | – |
| `zgetrf` | – |
| `zgesvd` | – |
| `zheev` | – |
| `zheevN` | – |


### Dual (ForwardDiff, N=1) — reference is LinearAlgebra generic, NOT a vendor BLAS

| op | ARM · NEON |
|---|---|
| `daxpy1` | – |
| `dscal1` | – |
| `ddot1` | – |
| `dnrm21` | – |
| `dasum1` | – |
| `diamax1` | – |
| `dgemvN1` | – |
| `dgemvT1` | – |
| `dger1` | – |
| `dtrmv1` | – |
| `dtrsv1` | – |
| `dgemm1` | – |
| `dsyrk1` | – |
| `dtrmm1` | – |
| `dtrsm1` | – |
| `dpotrf1` | – |
| `dgetrf1` | – |
| `dgeqrf1` | – |
| `dtrtri1` | – |
| `dpotri1` | – |
| `dsyev1` | – |

