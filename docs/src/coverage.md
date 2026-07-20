# LAPACK / BLAS coverage

This page tracks which `LinearAlgebra` operations route to PureBLAS after
`PureBLAS.activate()`, and their optimization status.

**Legend**

- **Routes** — the routine forwards to PureBLAS via LBT after `activate()` (or is composed from routines that do).
- **Optimized** — ✅ perf-gated `≥ OpenBLAS` on the dev fleet (Zen3/4/5); ⏳ correctness-first (numerically LAPACK-accurate, blocked/SIMD perf tuning is a documented follow-up); — n/a.
- Element types: **s** = Float32, **d** = Float64, **c** = ComplexF32, **z** = ComplexF64.

## BLAS

| Level | Routines | Types | Routes | Optimized |
|---|---|---|---|---|
| BLAS-1 | axpy, scal, copy, dot, nrm2, asum, iamax, rot, swap | s/d/c/z | ✅ | ✅ |
| BLAS-1 complex dot | dotu, dotc | c/z | ✅ | ✅ |
| BLAS-2 | gemv, ger, symv, hemv, trmv, trsv, banded, packed | s/d/c/z | ✅ | ✅ |
| BLAS-3 | gemm, symm, hemm, syrk, herk, syr2k, her2k, trmm, trsm | s/d/c/z | ✅ | ✅ |

GEMM additionally uses Strassen–Winograd (real) and Karatsuba 3M (complex) above a
size crossover — **beats** OpenBLAS at large `n`.

## LAPACK — factorizations & solves

| Op | Routines | Types | Routes | Optimized |
|---|---|---|---|---|
| Cholesky | potrf, potrs, potri | s/d/c/z | ✅ | ✅ |
| Pivoted Cholesky | pstrf | s/d/c/z | ✅ | ⏳ |
| LU | getrf, getrs, getri, gesv | s/d/c/z | ✅ | ✅ |
| QR | geqrf, geqrt, gemqrt, orgqr, ormqr | s/d/c/z | ✅ | ✅ |
| LQ | gelqf, orglq, ormlq | s/d/c/z | ✅ | ⏳ |
| QL / RQ | geqlf, gerqf, org/orm ql/rq | s/d/c/z | ✅ | ⏳ |
| Pivoted QR | geqp3 | s/d/c/z | ✅ | ⏳ |
| RZ (for gelsy) | tzrzf, ormrz | s/d/c/z | ✅ | ⏳ |
| Bunch–Kaufman | sytrf, hetrf, sytrs, hetrs | s/d/c/z | ✅ | ⏳ |
| Indefinite solve/inv | sysv, hesv, sytri, hetri | s/d/c/z | ✅ | ⏳ |
| Triangular | trtrs, trtri, trcon | s/d/c/z | ✅ | ✅ |
| Condition est. | gecon, pocon, trcon | s/d/c/z | ✅ | ⏳ |
| Least-squares | gels | s/d/c/z | ✅ | ⏳ |
| Rank-deficient LS | gelsd, gelsy | s/d/c/z | ✅ | ⏳ |

## LAPACK — SVD

| Op | Routines | Types | Routes | Optimized |
|---|---|---|---|---|
| SVD (values+vectors) | gesvd, gesdd, gebrd, bdsqr, bdsdc | s/d | ✅ | ✅ |
| SVD complex | gesvd, gesdd (z/c) | c/z | ✅ | ⏳ |
| Generalized SVD | ggsvd, ggsvd3 | s/d/c/z | ✅ (rank-deficient) | ⏳ |

## LAPACK — eigensolvers

| Op | Routines | Types | Routes | Optimized |
|---|---|---|---|---|
| Symmetric / Hermitian | syev, syevd, syevr, heev*, sytrd, hetrd, stedc, steqr, sterf, ormtr | s/d/c/z | ✅ | ⏳ |
| Sym-tridiagonal | stev, stegr, stebz, stein | s/d | ✅ | ⏳ |
| Generalized symmetric | sygvd, hegvd | s/d/c/z | ✅ | ⏳ |
| Nonsymmetric | geev, geevx, gebal, gehrd, hseqr, trevc, gebak | s/d/c/z | ✅ | ⏳ |
| Schur | gees | s/d/c/z | ✅ | ⏳ |
| Generalized nonsym (QZ) | ggev, gges, gghrd, hgeqz, tgevc | s/d/c/z | ✅ | ⏳ |
| Schur reordering | trexc, trsen | s/d/c/z | ✅ | ⏳ |
| Sylvester / Lyapunov | trsyl | s/d/c/z | ✅ | ⏳ |

## LAPACK — banded / tridiagonal / packed

| Op | Routines | Types | Routes | Optimized |
|---|---|---|---|---|
| General banded LU | gbtrf, gbtrs | s/d/c/z | ✅ | ⏳ |
| General tridiagonal | gtsv, gttrf, gttrs | s/d/c/z | ✅ | ⏳ |
| SPD tridiagonal | pttrf, pttrs, ptsv | s/d/c/z | ✅ | ⏳ |
| Banded Cholesky | pbtrf, pbtrs | s/d/c/z | ✅ | ⏳ |
| Packed Cholesky | pptrf, pptrs | s/d/c/z | ✅ | ⏳ |

## Free via composition

`exp`, `sqrt`, `log`, `^` of a matrix, `sylvester`/`lyap`, `pinv`, `nullspace`,
`rank`, `cond`, `factorize` — computed in Julia on top of the routed
`eigen`/`schur`/`svd`/`\` kernels; no separate LAPACK wrapper needed.

## OpenBLAS fallthrough: ZERO

**Every LAPACK symbol `LinearAlgebra` can `ccall` now forwards to PureBLAS after
`activate()`** — including the auxiliaries (`larf`/`larfg`/`lacpy`), the driver internals
(`gebrd`/`bdsqr`/`bdsdc`/`hseqr`/`trevc`/`gebak`/`sytrd`/`hetrd`/`orgtr`/`ormtr`), the
combined and expert drivers (`gesv`, `posv`, **`gesvx`** with equilibration + iterative
refinement + condition/error bounds), the reordering routines (`trexc`/`trsen`/**`tgsen`**,
real *and* complex — the real path does the 2×2 conjugate-pair swap), **`trrfs`**,
**`syconv`**, complex **`bdsqr`**, and the **rank-deficient generalized SVD** (`ggsvd`,
all s/d/c/z). This is enforced by a machine-checkable ratchet test (`test/lbt_forward_tests.jl`)
that enumerates every symbol the stdlib wraps and asserts the fallthrough count is **0**.

The only two names excluded from the count are `cstev_`/`zstev_`, which are **not real LAPACK
symbols** — they appear only in commented-out lines of the stdlib and have no OpenBLAS export.

## Summary

- **BLAS 1/2/3 and the core dense factorizations (Cholesky, LU, QR, SVD real+complex+F32)
  are routed AND perf-gated `≥ OpenBLAS`.**
- **Everything else routable through the `LinearAlgebra` API** — all eigensolvers (symmetric,
  Hermitian, nonsymmetric, generalized, Schur), Sylvester/Schur-reordering, the expert general
  solver (`gesvx`), generalized SVD, and the remaining factorizations (indefinite, QL/RQ, RZ,
  pivoted Cholesky, banded/tridiagonal/packed, rank-deficient LS) — **is routed and numerically
  LAPACK-accurate, correctness-first**. The `≥ OpenBLAS` performance gate for this second tier
  (blocked `dlaqr0` multishift+AED, blocked `dlahr2`, SIMD Bunch-Kaufman/QZ, blocked complex
  `zunmbr`, a convergent perf `dbdsqr`, …) is a scheduled follow-up campaign.
- The coverage audit (the ratchet gate) confirms **zero OpenBLAS fallthrough**: after
  `activate()`, the OpenBLAS fallback is fully removed.
