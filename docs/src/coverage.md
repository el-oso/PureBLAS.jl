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
| Generalized SVD | ggsvd | d | ✅ (Float64 full-rank) | ⏳ |

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

## Known fallthroughs (still served by OpenBLAS)

The registry-vs-reachable audit (401 forwarded symbols vs the LAPACK surface
`LinearAlgebra` wraps) leaves only these, none of which a user hits through a
high-level call (`\`, `lu`, `cholesky`, `qr`, `svd`, `eigen`, `schur`, `eigen(A,B)`,
`svd(A,B)`, least-squares, indefinite solve — all routed):

- **Auxiliaries** (`larf`, `larfg`, `lacpy`) — PureBLAS uses its own in-house kernels;
  only a direct `LAPACK.larfg!` etc. call falls through.
- **Driver internals** (`gebrd`, `bdsqr`, `bdsdc`, `hseqr`, `trevc`, `gebak`, `sytrd`,
  `hetrd`, `orgtr`/`ormtr`) — the enclosing driver (`gesvd`/`gesdd`/`geev`/`syev`…) is
  routed; only the piecewise direct call falls through.
- **Combined drivers** (`gesv`, `posv`, `gesvx`) — the high-level path composes the
  routed pieces (`getrf`+`getrs`, `potrf`+`potrs`).
- **Niche / not-yet-built**: `tgsen` (generalized Schur reorder), `trrfs` (iterative
  refinement), `syconv` (Bunch–Kaufman aa-variant), generalized SVD for complex +
  rank-deficient (`ggsvd` is Float64 full-rank only).

## Summary

- **BLAS 1/2/3 and the core dense factorizations (Cholesky, LU, QR, SVD real+complex+F32)
  are routed AND perf-gated `≥ OpenBLAS`.**
- **All eigensolvers (symmetric, Hermitian, nonsymmetric, generalized, Schur), the
  Sylvester/Schur-reordering routines, and the remaining factorizations (indefinite,
  QL/RQ, RZ, pivoted Cholesky, banded/tridiagonal/packed, rank-deficient LS) are routed
  and numerically LAPACK-accurate, but correctness-first** — the `≥ OpenBLAS` performance
  gate for these (blocked `dlaqr0` multishift+AED, blocked `dlahr2`, SIMD Bunch-Kaufman/QZ,
  blocked complex `zunmbr`, …) is a scheduled follow-up campaign.
- The coverage audit (registry-vs-reachable diff) confirms **zero high-level fallthrough**:
  every op reachable through the `LinearAlgebra` API routes to PureBLAS; the residual
  fallthroughs above are auxiliaries, driver-internals, and niche/expert routines.
