PB / OpenBLAS speed ratio, median (worst cell) per op per µarch. Provenance: [`bench/provenance.md`](provenance.md).

### Real

| op | ARM · NEON |
|---|---|
| `axpy` | 1.00 (0.97) |
| `dot` | 1.07 (1.00) |
| `nrm2` | 8.83 (7.00) |
| `asum` | 1.00 (0.96) |
| `scal` | 1.00 (0.92) |
| `iamax` | 0.61 (0.50) |
| `blascopy` | – |
| `swap` | – |
| `gemvN` | 6.41 (1.73) |
| `gemvT` | 2.01 (1.47) |
| `ger` | 2.34 (0.91) |
| `symv` | 1.38 (0.66) |
| `trmv` | 1.84 (1.51) |
| `trsv` | 1.44 (1.23) |
| `trsvLN` | 1.35 (1.19) |
| `trsvLT` | 2.05 (1.39) |
| `spmv` | 1.44 (1.03) |
| `gbmvN` | 0.79 (0.74) |
| `sbmv` | 2.79 (2.73) |
| `gemvNrect` | – |
| `gemvTrect` | – |
| `gemm` | 6.28 (2.18) |
| `symm` | 3.15 (1.03) |
| `syrk` | 3.50 (0.77) |
| `syr2k` | 4.19 (1.27) |
| `trmm` | 1.40 (0.54) |
| `trmmR` | 2.27 (0.50) |
| `trsm` | 2.04 (1.03) |
| `trsmR` | 2.45 (1.22) |
| `trsmRrect` | – |
| `trsmRband` | – |
| `potrf` | 1.97 (1.53) |
| `getrf` | 1.90 (0.89) |
| `geqrf` | 1.21 (1.13) |
| `gesvd` | 1.89 (1.02) |
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

