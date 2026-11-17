PB / OpenBLAS speed ratio, median (worst cell) per op per µarch. Provenance: [`bench/provenance.md`](provenance.md).

### Real

| op | ARM · NEON |
|---|---|
| `axpy` | 1.07 (1.00) |
| `dot` | 1.14 (1.00) |
| `nrm2` | 9.52 (7.00) |
| `asum` | 1.06 (1.00) |
| `scal` | 1.00 (0.99) |
| `iamax` | 0.64 (0.51) |
| `blascopy` | – |
| `swap` | – |
| `gemvN` | 7.48 (1.90) |
| `gemvT` | 2.15 (1.81) |
| `ger` | 2.55 (1.03) |
| `symv` | 1.29 (0.74) |
| `trmv` | 2.49 (1.64) |
| `trsv` | 1.64 (1.36) |
| `trsvLN` | 1.43 (1.29) |
| `trsvLT` | 2.14 (1.41) |
| `spmv` | 1.60 (1.05) |
| `gbmvN` | 0.81 (0.77) |
| `sbmv` | 3.06 (3.02) |
| `gemvNrect` | – |
| `gemvTrect` | – |
| `gemm` | 7.36 (2.30) |
| `symm` | 4.91 (1.15) |
| `syrk` | 3.91 (0.97) |
| `syr2k` | 5.14 (1.41) |
| `trmm` | 1.92 (0.69) |
| `trmmR` | 3.01 (0.64) |
| `trsm` | 2.30 (1.22) |
| `trsmR` | 3.05 (1.44) |
| `trsmRrect` | – |
| `trsmRband` | – |
| `potrf` | 2.22 (1.71) |
| `getrf` | 2.17 (1.17) |
| `geqrf` | 1.34 (1.18) |
| `gesvd` | 1.95 (1.07) |
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

