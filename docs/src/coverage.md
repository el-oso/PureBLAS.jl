# LAPACK / BLAS coverage

This page tracks which `LinearAlgebra` operations route to PureBLAS after
`PureBLAS.activate()`, and their optimization status.

**Legend**

- **Routes** — the routine forwards to PureBLAS via LBT after `activate()` (or is composed from routines that do).
- **Gated** — the performance verdict, and it is a plain yes/no:
  - ✅ **Yes** — clears `≥ max(OpenBLAS, AOCL)` at **every** size in the Zen4 gate sweep. No caveats,
    no footnote. If a row needs qualifying, it is not a ✅.
  - ❌ **No** — measured, and misses at one or more sizes. The footnote says where and by how much.
    ❌ is about *speed only* — every routine here is correct and numerically LAPACK-accurate.
  - ⏳ **In progress / not measured** — no gate row yet, or actively being worked.
  
  Both references are always measured, because a routine can beat one by a wide margin and lose to
  the other. A large win over a library that did not implement a routine is not evidence of speed
  (AOCL's packed Cholesky is ~6× slower than OpenBLAS's — near-reference code — so beating it there
  means little). Where a footnote gives numbers it names the reference.
  Verdicts are from the **Zen4** sweep (`bench/plots.jl`, freq-locked); footnotes note fleet results
  where Zen3/Zen5 have been measured.
- Element types: **s** = Float32, **d** = Float64, **c** = ComplexF32, **z** = ComplexF64.

## BLAS

| Level | Routines | Types | Routes | Gated | vs OB geo/worst | vs AOCL geo/worst |
|---|---|---|---|---|---|---|
| BLAS-1 | axpy, scal, dot, nrm2, asum, iamax | s/d/c/z | ✅ | ❌ | 1.0 / 0.97 | 0.95 / 0.84 |
| BLAS-2 dense | gemv, ger, symv, trmv, trsv | s/d/c/z | ✅ | ❌ | 1.08 / 0.98 | 0.97 / 0.87 |
| BLAS-2 banded/packed | gbmv, sbmv, spmv | s/d/c/z | ✅ | ✅ | 1.14 / 1.1 | 1.12 / 1.11 |
| BLAS-3 | gemm, symm, syrk, syr2k, trmm, trsm | s/d/c/z | ✅ | ❌ | 1.08 / 0.95 | 1.02 / 0.9 |

GEMM additionally uses Strassen–Winograd (real) and Karatsuba 3M (complex) above a
size crossover — **beats** OpenBLAS at large `n`.

## LAPACK — factorizations & solves

| Op | Routines | Types | Routes | Gated | vs OB geo/worst | vs AOCL geo/worst |
|---|---|---|---|---|---|---|
| Cholesky (lower) | potrf | s/d/c/z | ✅ | ✅ | 1.49 / 1.11 | 1.72 / 1.12 |
| Cholesky (upper) | potrf `uplo='U'` | s/d/c/z | ✅ | ❌ | 1.42 / 0.81 | 1.39 / 1.05 |
| Cholesky solve | potrs [^solv] | s/d/c/z | ✅ | ❌ | 2.2 / 0.93 | 0.94 / 0.78 |
| Pivoted Cholesky | pstrf [^ps] | s/d/c/z | ✅ | ❌ | 1.28 / 0.85 | 1.33 / 1.15 |
| LU | getrf, gesv | s/d/c/z | ✅ | ❌ | 1.37 / 1.03 | 1.76 / 0.98 |
| LU solve | getrs [^solv] | s/d/c/z | ✅ | ❌ | 1.05 / 0.99 | 0.93 / 0.85 |
| QR | geqrf, orgqr, ormqr | s/d/c/z | ✅ | ❌ | 1.94 / 1.65 | 1.54 / 0.93 |
| Pivoted QR | geqp3 [^unblk] | s/d/c/z | ✅ | ❌ | 0.34 / 0.18 | 0.31 / 0.17 |
| Bunch–Kaufman | sytrf, hetrf [^unblk] | s/d/c/z | ✅ | ❌ | 0.56 / 0.15 | 0.59 / 0.15 |
| Bunch–Kaufman solve | sytrs, hetrs [^unblk] | s/d/c/z | ✅ | ❌ | 0.86 / 0.52 | 1.1 / 0.47 |
| Triangular solve | trtrs [^solv] | s/d/c/z | ✅ | ❌ | 1.11 / 1.02 | 1.07 / 0.88 |
| Least-squares | gels | s/d/c/z | ✅ | ✅ | 2.42 / 1.93 | 1.88 / 1.28 |
| SVD | gesvd, gesdd | s/d/c/z | ✅ | ✅ | 1.23 / 1.12 | 1.17 / 1.05 |
| Symmetric eigen | syev, syevd, syevr | s/d/c/z | ✅ | ❌ | 1.13 / 0.98 | 1.31 / 1.14 |

## LAPACK — SVD

`gesvd`/`gesdd` appear in the factorizations table above (✅). The remaining SVD rows are not yet gated:

| Op | Routines | Types | Routes | Gated |
|---|---|---|---|---|
| SVD complex | gesvd, gesdd (z/c) | c/z | ✅ | ⏳ |
| Generalized SVD | ggsvd, ggsvd3 | s/d/c/z | ✅ (rank-deficient) | ⏳ |

## LAPACK — eigensolvers

| Op | Routines | Types | Routes | Gated |
|---|---|---|---|---|
| Symmetric / Hermitian (vectors) | syev, syevd, syevr, heev, sytrd, hetrd, stedc, steqr, ormtr | s/d/c/z | ✅ | ✅ |
| Symmetric / Hermitian (values only) | syev, sterf | s/d/c/z | ✅ | ❌ |
| Sym-tridiagonal | stev, stegr, stebz, stein | s/d | ✅ | ⏳ |
| Generalized symmetric | sygvd, hegvd | s/d/c/z | ✅ | ⏳ |
| Nonsymmetric | geev, geevx, gebal, gehrd, hseqr, trevc, gebak | s/d/c/z | ✅ | ⏳ |
| Schur | gees | s/d/c/z | ✅ | ⏳ |
| Generalized nonsym (QZ) | ggev, gges, gghrd, hgeqz, tgevc | s/d/c/z | ✅ | ⏳ |
| Schur reordering | trexc, trsen | s/d/c/z | ✅ | ⏳ |
| Sylvester / Lyapunov | trsyl | s/d/c/z | ✅ | ⏳ |

**Symmetric / Hermitian eigensolver.** With eigenvectors (`jobz='V'`) it clears both references on
the Zen4 sweep (1.32 / 1.25 vs OpenBLAS, 1.36 / 1.19 vs AOCL). **Values-only (`jobz='N'`) does not**:
1.13 / **0.98** vs OpenBLAS, though 1.31 / 1.14 vs AOCL — so the row above is ❌ on the strict
worst-case rule, driven by a single size. An earlier version of this page claimed it gated "at every
size on the whole fleet … both `jobz`"; that was written against the OpenBLAS-only criterion and the
`jobz='N'` cell does not survive the two-reference one.
Three Level-3 pieces get the vectors path there: the two-sided reduction
(`sytrd`/`hetrd`) is blocked `dlatrd`+`syr2k`/`her2k`; the eigenvector back-transform (`ormtr`/`unmtr`)
is compact-WY block reflectors; and the divide-and-conquer solver (`stedc`) assembles eigenvectors once
(no per-merge column copies) and combines survivors with the `dlaed3` COLTYP **two half-height gemms**
(exploiting the block-diagonal zero structure — ~half the combine flops, matching libFLAME).

## LAPACK — banded / tridiagonal / packed

| Op | Routines | Types | Routes | Gated | vs OB geo/worst | vs AOCL geo/worst |
|---|---|---|---|---|---|---|
| General banded LU | gbtrf, gbtrs [^gb] | s/d/c/z | ✅ | ❌ | 0.96 / 0.48 | 0.78 / 0.34 |
| General tridiagonal | gtsv, gttrf, gttrs | s/d/c/z | ✅ | ✅ | 1.0 / 1.0 | 1.03 / 1.03 |
| SPD tridiagonal | pttrf, pttrs, ptsv [^tri] | s/d/c/z | ✅ | ❌ | 1.13 / 1.13 | 1.0 / 0.99 |
| Banded Cholesky | pbtrf, pbtrs | s/d/c/z | ✅ | ✅ | 1.52 / 1.12 | 1.41 / 1.03 |
| Packed Cholesky | pptrf, pptrs [^pp] | s/d/c/z | ✅ | ❌ | 1.07 / 0.92 | 1.17 / 0.93 |

### Fleet status

The verdicts above are the **Zen4** sweep. Where Zen3/Zen5 have been measured they mostly agree; the
known exception is banded Cholesky.

**`pbtrf`** gates fully on **Zen4** (Float64, `kd = 32…384`, n ∈ {1024, 4096}, both triangles:
1.02–3.28× AOCL, 1.49–2.00× OpenBLAS) and beats **OpenBLAS on all three µarchs**, but it does
**not** yet clear AOCL fleet-wide. The table above marks it ✅ because that verdict is
scoped to Zen4; on Zen3/Zen5 it would be ❌. Residuals are all `uplo='U'`
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

[^unblk]: **Measured as unblocked.** `geqp3` and `sytrf`/`sytrs` are BLAS-2 implementations racing
    blocked BLAS-3 ones, and their ratios degrade monotonically with `n` — the signature. Zen4 gate
    sweep (geomean / worst, n = 8…4096): `geqp3` **0.34 / 0.18** vs OpenBLAS and **0.31 / 0.17** vs
    AOCL — the worst routine in the real LAPACK surface; `sytrf` **0.56 / 0.15** and 0.59 / 0.15;
    `sytrs` 0.86 / 0.52 and 1.10 / 0.47. Both are confirmed from source, not inferred:
    `geqp3.jl`'s header states it is the unblocked core, and `sytrf!` calls `_sytf2_*` directly with
    no blocked driver at all. These had **never been benchmarked** before 2026-07-29 — they routed
    and passed correctness, which reads as coverage while measuring nothing.
    A blocked `dlaqps` port for `geqp3` was written and **verified correct** (pivots and R identical
    to LAPACK, reconstruction to 1e-13, over full-rank/rank-deficient/graded input) but was **not
    faster**, and larger panels were *worse* — the diagnostic that the cost is the panel's own
    per-column BLAS-2 calls, not the trailing gemm. It is reverted, not shipped. The retry must reach
    the kernels pointer-direct; see §5 of [Tuning](tuning.md) and the note on `sytrs`, where two
    codegen hazards (an unforwarded store-to-load and a per-element `herm ?` branch) were worth
    +11–18%.


[^gb]: `gbtrf` has no clean shape — Zen4 vs OpenBLAS 1.44 / 1.30 / 0.79 / 1.11 / 1.08 / 0.66 at
    n = 32…1024, and vs AOCL 1.06 / 0.89 / 0.60 / 1.25 / 1.12 / 0.44 (gate sweep: 0.96 / 0.48 and
    0.78 / 0.34). The dips at n=128 and n=1024 are undiagnosed and should not be assumed to share a
    cause with the unblocked routines above.

[^solv]: The ✅ on these rows is for the **factorizations**; their **solves are not yet gated**, and
    the distinction is worth stating because the solves were invisible for a long time.
    `getrs`/`potrs`/`trtrs` had no native entry point until 2026-07-29, so they could not be measured
    at all — the harness compares `PureBLAS.foo!` against `LAPACK.foo!` and there was nothing to call.
    Adding one exposed `A \ b` with a single right-hand side running **14× slower than OpenBLAS**,
    traced to three defects in `trsm`: a blocked setup never repaid for narrow B, a power-of-two-`lda`
    A-pad that cost +3–357% across all sixteen operand combinations and paid nowhere, and a **ragged
    SIMD lane charged as a full lane** (`nrhs`=1…7 each cost what eight columns cost together).
    After those, Zen4 gate sweep: vs **OpenBLAS** `getrs` geomean 1.05 / worst 0.99, `potrsL`
    2.20 / 0.93, `potrsU` 2.74 / 1.07 ✓, `trtrs` 1.11 / 1.02 ✓ — `potrs` reaching **4.9×** at n=1024.
    vs **AOCL** they still miss: `getrs` 0.93 / 0.85, `potrsL` 0.94 / 0.78, `potrsU` 1.13 / 0.91,
    `trtrs` 1.07 / 0.88. So under `max(OB, AOCL)` the solves remain open, and `trsm` itself passes
    OpenBLAS (worst 1.18, up from 1.04) while still missing AOCL (worst 0.90).
    A residual sits at `nrhs = 8`, between the two mechanisms — too wide for the per-column sweep,
    and on Zen3 (`W=4`) already two full lanes, so the ragged-tail fix does not reach it.

[^ps]: `pstrf`'s blocked pivoted factorization clears **AOCL completely on both Zen4 and Zen5**, both
    triangles, every size (Zen5: `uplo='L'` 1.07–1.75, `uplo='U'` 1.06–1.75). Against **OpenBLAS** it
    clears everywhere except a narrow `uplo='U'` window that is consistent across µarchs and mild:
    Zen4 n=48 **0.943** and n=64 **0.921**; Zen5 n=64 **0.978** only. Everything else runs 1.01–1.99.
    Those cells were known but had never been *gated* — the harness passed `uplo='L'` for both sides,
    so the separate 'U' code path (the pivoted panel and the trailing update both mirror) was never
    measured. A `pstrfU` row now tracks it. The large-n wins come from batching the pivot row swaps
    per panel: they are stride-`lda` and were ~47% of the runtime, so the swap — not the BLAS-3
    update — was the bottleneck.
    ⚠ Float64; Zen3 not yet measured for this routine.
    One caveat on reading the published table: the Zen4 gate sweep reports `pstrf` worst = 0.85 vs
    OpenBLAS, which is **not** one of the cells above — it is n=8, where the per-round ratios come out
    bimodal (1.03 / 1.00 / 0.79 / 1.01 / 1.01 / 0.79 / 0.80 / 1.03). Rounds alternating like that at
    sub-microsecond scale is the ABBA ordering effect, not a stable regression; the sweep's size grid
    (8, 32, 128, …) skips 48 and 64 entirely, so it never samples the real residual.

[^pp]: **`uplo='L'` now clears OpenBLAS** on the Zen4 gate sweep (geomean 1.11, worst 1.04) after the
    per-call overhead fix described below — but it still misses **AOCL** (1.17 / 0.93), so the row
    stays ⏳. `uplo='U'` is the mirror image: 3.30 / 1.43 vs AOCL ✓ but 1.07 / **0.92** vs OpenBLAS.
    Neither triangle clears both references, and they fail against *different* ones.
    Historical detail, kept because the diagnosis generalises —
    which is why it needs both. Zen4, Float64, n = 8…2048:
    `uplo='L'` clears OpenBLAS at every size (1.05–1.37) and clears AOCL except n=32 (**0.949**) and
    n=48 (**0.980**). `uplo='U'` beats AOCL by a wide margin (1.17–**6.27**, growing with n) but
    loses to OpenBLAS at n=48…2048 (**0.918–0.990**). That 6.27× is **not** evidence of quality:
    timing the *same reference routine* under each library, AOCL's `dpptrf` `uplo='U'` is
    **6.0× slower than OpenBLAS's** at n=1024 (175 281 µs vs 29 297 µs) — near-stock netlib — while
    on `dpotrf` the two sit within 20% of each other. AOCL optimizes dense Cholesky and does not
    optimize the packed variant. The gate is unchanged by this — it is `max(OpenBLAS, AOCL)`, so a
    slow AOCL never *lowers* the bar, it merely means `max()` here **equals OpenBLAS**; both
    references are still measured, and both sets of misses above still count. What the asymmetry does
    change is how much a large ratio is worth as *evidence*: a cell far above one reference and below
    the other is the tell that the high ratio is measuring the reference's absence, not our speed.
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

- **Routing is complete** — see the fallthrough ratchet above. Every operation in the tables reaches
  PureBLAS after `activate()`, and every one is numerically LAPACK-accurate.
- **Performance is typically well ahead of both references but not yet uniformly so.** On the Zen4
  sweep the geomeans run ~1.0–2.4× across BLAS and the dense factorizations, and 7 of 33 measured
  rows clear `≥ max(OpenBLAS, AOCL)` at *every* size. The rest miss somewhere, usually narrowly
  (0.9–0.99 at one or two sizes — often the smallest, where per-call overhead dominates and the
  measurement is least stable). Four are genuinely behind and are the active work: pivoted QR
  (`geqp3`, 0.18), Bunch–Kaufman (`sytrf` 0.15, `sytrs` 0.52) and banded LU (`gbtrf`, 0.48) — the
  first two confirmed as unblocked BLAS-2 implementations racing blocked ones.
  An earlier version of this summary said BLAS 1/2/3 and the core factorizations were "perf-gated
  `≥ OpenBLAS`". That was written against an OpenBLAS-only, geomean-flavoured reading; under the
  project's actual rule — `max(OpenBLAS, AOCL)` at every size — it does not hold, and the tables
  above now show the per-row numbers instead of a blanket claim.
- **Everything else routable through the `LinearAlgebra` API** — all eigensolvers (symmetric,
  Hermitian, nonsymmetric, generalized, Schur), Sylvester/Schur-reordering, the expert general
  solver (`gesvx`), generalized SVD, and the remaining factorizations (indefinite, QL/RQ, RZ,
  pivoted Cholesky, banded/tridiagonal/packed, rank-deficient LS) — **is routed and numerically
  LAPACK-accurate, correctness-first**. The `≥ max(OpenBLAS, AOCL)` performance gate for this second tier
  (blocked `dlaqr0` multishift+AED, blocked `dlahr2`, SIMD Bunch-Kaufman/QZ, blocked complex
  `zunmbr`, a convergent perf `dbdsqr`, …) is a scheduled follow-up campaign.
- The coverage audit (the ratchet gate) confirms **zero OpenBLAS fallthrough**: after
  `activate()`, the OpenBLAS fallback is fully removed.
