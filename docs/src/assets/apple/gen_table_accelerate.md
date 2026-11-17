PB / Accelerate speed ratio, median (worst cell) per op per µarch. Provenance: [`bench/provenance.md`](provenance.md).

### Real

| op | ARM · NEON |
|---|---|
| `axpy` | 1.20 (0.68) |
| `dot` | 1.21 (1.01) |
| `nrm2` | 2.20 (1.67) |
| `asum` | 1.00 (0.54) |
| `scal` | 0.98 (0.39) |
| `iamax` | 1.20 (1.00) |
| `blascopy` | – |
| `swap` | – |
| `gemvN` | 0.89 (0.50) |
| `gemvT` | 1.01 (0.34) |
| `ger` | 0.95 (0.66) |
| `symv` | 1.63 (0.94) |
| `trmv` | 0.88 (0.40) |
| `trsv` | 0.94 (0.68) |
| `trsvLN` | 0.85 (0.69) |
| `trsvLT` | 1.06 (0.82) |
| `spmv` | 1.05 (0.68) |
| `gbmvN` | 0.77 (0.66) |
| `sbmv` | 0.77 (0.68) |
| `gemvNrect` | – |
| `gemvTrect` | – |
| `gemm` | 1.04 (0.88) |
| `symm` | 0.79 (0.39) |
| `syrk` | 0.59 (0.18) |
| `syr2k` | 0.79 (0.36) |
| `trmm` | 0.42 (0.21) |
| `trmmR` | 0.47 (0.18) |
| `trsm` | 0.58 (0.35) |
| `trsmR` | 0.74 (0.49) |
| `trsmRrect` | – |
| `trsmRband` | – |
| `potrf` | 0.61 (0.42) |
| `getrf` | 1.43 (0.75) |
| `geqrf` | 1.08 (0.28) |
| `gesvd` | 1.35 (0.90) |
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

