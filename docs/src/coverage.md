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
| Pivoted Cholesky | pstrf | s/d/c/z | ✅ | ⏳ [^ps] |
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
| Symmetric / Hermitian | syev, syevd, syevr, heev, sytrd, hetrd, stedc, steqr, sterf, ormtr | s/d/c/z | ✅ | ✅ |
| Sym-tridiagonal | stev, stegr, stebz, stein | s/d | ✅ | ⏳ |
| Generalized symmetric | sygvd, hegvd | s/d/c/z | ✅ | ⏳ |
| Nonsymmetric | geev, geevx, gebal, gehrd, hseqr, trevc, gebak | s/d/c/z | ✅ | ⏳ |
| Schur | gees | s/d/c/z | ✅ | ⏳ |
| Generalized nonsym (QZ) | ggev, gges, gghrd, hgeqz, tgevc | s/d/c/z | ✅ | ⏳ |
| Schur reordering | trexc, trsen | s/d/c/z | ✅ | ⏳ |
| Sylvester / Lyapunov | trsyl | s/d/c/z | ✅ | ⏳ |

**Symmetric / Hermitian eigensolver** gates `≥ max(OpenBLAS, AOCL)` at every size on the whole fleet
(Zen3/4/5), all four types and both `jobz`. Three Level-3 pieces get it there: the two-sided reduction
(`sytrd`/`hetrd`) is blocked `dlatrd`+`syr2k`/`her2k`; the eigenvector back-transform (`ormtr`/`unmtr`)
is compact-WY block reflectors; and the divide-and-conquer solver (`stedc`) assembles eigenvectors once
(no per-merge column copies) and combines survivors with the `dlaed3` COLTYP **two half-height gemms**
(exploiting the block-diagonal zero structure — ~half the combine flops, matching libFLAME).

## LAPACK — banded / tridiagonal / packed

| Op | Routines | Types | Routes | Optimized |
|---|---|---|---|---|
| General banded LU | gbtrf, gbtrs | s/d/c/z | ✅ | ⏳ |
| General tridiagonal | gtsv, gttrf, gttrs | s/d/c/z | ✅ | ✅ |
| SPD tridiagonal | pttrf, pttrs, ptsv | s/d/c/z | ✅ | ✅ [^tri] |
| Banded Cholesky | pbtrf, pbtrs | s/d/c/z | ✅ | ⏳ [^pb] |
| Packed Cholesky | pptrf, pptrs | s/d/c/z | ✅ | ⏳ [^pp] |

[^pb]: `pbtrf` gates fully on **Zen4** (Float64, `kd = 32…384`, n ∈ {1024, 4096}, both triangles:
    1.02–3.28× AOCL, 1.49–2.00× OpenBLAS) and beats **OpenBLAS on all three µarchs**, but it does
    **not** yet clear AOCL fleet-wide — which is why this row is ⏳. Residuals are all `uplo='U'`
    at mid bandwidth: Zen5 `kd=128/192/256/384` at 0.954–0.991, Zen3 `kd=128/160/192` at
    0.881–0.984 (plus `uplo='L'` `kd=160/192` at 0.959–0.995). Zen3 is the worst box and also the
    one whose `_pbtrf_ucross` bracket is derived from a narrower `_vwidth` (W=4 → the switch lands
    on 192, itself a 0.90 cell), so the leading hypothesis is a candidate bracket putting the
    optimum at an edge — the failure this file's own history records for the wide-band panel
    bracket. Not yet confirmed by measurement. Correctness is clean on all three boxes
    (11368/11369, identical).
    `uplo='U'` dispatches between two kernels at a measured bandwidth crossover: a conj-transpose
    re-pack onto the (much faster) lower kernel for narrow bands, and a native upper-storage port
    once the re-pack's diagonal walk starts to dominate. The panel width likewise has two measured
    regimes — clamping the wide-band width to `kd` collapses the in-band panel and was the
    routine's only Zen4 gate miss. See §5.2 and §5.3 of [Tuning](tuning.md).

[^ps]: `pstrf`'s blocked pivoted factorization gates broadly — Zen4 Float64 vs OpenBLAS, `uplo='L'`
    1.13–1.76 at every size and `uplo='U'` 1.01–1.99 — with **two remaining cells**: `uplo='U'` at
    n=48 (**0.943**) and n=64 (**0.921**). Those two were known but had never been *gated*: the
    benchmark harness measured only `uplo='L'`, and 'U' is a separate code path (both the pivoted
    panel and the trailing update mirror). A `pstrfU` row now tracks them. The large-n wins come from
    batching the pivot row swaps per panel — they are stride-`lda` and were ~47% of the runtime, i.e.
    the swap, not the BLAS-3 update, was the bottleneck.
    ⚠ Zen4 Float64 only; not yet fleet-validated.

[^pp]: `pptrf` does **not** yet gate, and the two triangles fail against *different* references —
    which is why it needs both. Zen4, Float64, n = 8…2048:
    `uplo='L'` clears OpenBLAS at every size (1.05–1.37) and clears AOCL except n=32 (**0.949**) and
    n=48 (**0.980**). `uplo='U'` beats AOCL by a wide margin (1.17–**6.27**, growing with n) but
    loses to OpenBLAS at n=48…2048 (**0.918–0.990**) — OpenBLAS's packed upper is far stronger than
    AOCL's, so measuring against AOCL alone would have shown a 6× win and hidden eight missing cells.
    The lower path improved this session (worst cell 0.738 → 0.949) once its gap was decomposed: the
    `spr!` *kernel* already ties or beats AOCL at every order, and the whole deficit was per-call
    overhead — the public entry plus a `SubArray` built per column — compounded by the lower path
    reusing the *upper* path's cutoff constant, which meant `spr!` was never called at all at n=32.
    See §5.3 of [Tuning](tuning.md). The remaining cells are gaps, not ceilings.
    ⚠ Zen4 only; not yet fleet-validated. The committed plot SVGs/tables predate the lower-path fix.

[^tri]: All six tridiagonal routines gate `≥ max(OpenBLAS, AOCL)` on Zen3/4/5 with one exception:
    `pttrs` measures 0.99 against AOCL (1.67–1.73 vs OpenBLAS). That is a *shared* dependency-chain
    bound, not a gap — both libraries run at ~12.2 cyc/elem, which is the analytic 2×(multiply+subtract)
    limit for the pair of triangular sweeps, with 0.0% run-to-run drift. See "Tridiagonal" in
    [Performance](performance.md).

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
