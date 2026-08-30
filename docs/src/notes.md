# Performance notes

Per-routine analysis, diagnoses and history behind the numbers on [Performance](performance.md) and
[Coverage](coverage.md). The tables and plots on those pages are the source of truth for *what* the
current numbers are; this page records *why* they are what they are, including the levers that were
measured and rejected. Figures quoted here are as measured on the date stated. How the numbers are
produced and how to read them: [Methodology](methodology.md).

## Real BLAS

**BLAS-1** is bandwidth-bound and at parity fleet-wide (worst sizes ≥ 0.98, except `iamax` on AVX2 at
0.95). `nrm2` runs 7–10× because OpenBLAS uses the always-scaled LAPACK algorithm; PureBLAS scales only
on overflow/underflow.

**BLAS-2** gates fleet-wide with two exceptions, both on Zen5: `gemvN` (mid-n native-512 residual, worst
0.90) and `trmv`/`trsv` at large n (DRAM regime, worst 0.98–0.99). `spmv` is flat ≈ 1.9–2.2 across the
fleet (AP-residency packed panel); `ger` sits at gate on all three boxes (worst 0.97–1.06) with a
per-µarch-calibrated write-stream count.

**BLAS-3.** `gemm` gates every size on all three boxes (Strassen–Winograd at large n runs 1.2–1.4×). The
triangular/symmetric ops gate on AVX-512; on AVX2 the worst size of `trmm` (0.84) is still open, while
`trsm` now gates against OpenBLAS at every size on both available boxes (AVX2 worst 1.14).

### Complex `gemv`-T/C is an ISA split

Past L2 these route to a column-block whose width is chosen by a Measure-tier knob bounded by the
register file: `regs = (2·NC + 2)·(HALF ? 1 : 2)`. On **AVX-512** that admits `NC = 8` narrow
accumulators — eight concurrent column streams, matching what AOCL's fused `zdotxf` runs — and `zgemvT`
then gates at every size on Zen4 (n=512 0.96→1.00, n=1024 0.91→1.02 vs AOCL). On **AVX2** the same
formula needs 18 vector registers against 16 available, so Zen3 keeps `NC = 4` and its mid-band misses
stand: `zgemvT` 0.95–0.98 and `zgemvC` 0.94–0.96 vs AOCL at n=512…2048. That is a register-file limit,
not an unexplored lever — closing it needs a different accumulator shape, not a wider block.

The split below L2 is separate and derived, not tuned: while A is L2-resident the loop is
FMA-latency-bound and prefers fewer, wider accumulators (measured n=256: 1.15 wide vs 1.04 narrow), so
the wide arm is kept there on both ISAs.

### The BLAS-3 conversion gap

`gemm` is the fastest thing on all three boxes and pulls *further* ahead as `n` grows — but the routines
that build on that same engine fall *behind* AOCL at exactly the sizes where `gemm` wins most:

| vs max(OpenBLAS, AOCL) | n=2048 | n=4096 |
|---|---|---|
| `gemm` — Zen4 / Zen3 / Zen5 | 1.150 / 1.150 / 1.168 | **1.252 / 1.280 / 1.308** |
| `syrk` | 0.955 / 0.963 / 0.978 | **0.937 / 0.968 / 0.953** |
| `syr2k` | 0.976 / 0.974 / 0.977 | 0.966 / 0.983 / 0.957 |

`syrk`, `syr2k`, `symm`, `trmm` and `trsm` all route their trailing updates through the same
`_gemm_core!` / `_microkernel_db!` engine. That engine is demonstrably 25–31% faster than AOCL at n=4096
on this silicon, so the ~30-point spread between it and the routines built on it is a **conversion loss,
not a hardware limit** — and it reproduces on three microarchitectures, which makes it structural rather
than a per-box tuning artifact. It is the single largest shared lever currently on the board.

**`trsm` n=32 was one such invariant and is now CLOSED on both available boxes** (2026-08-10): Zen4
0.898 → 1.056 via interleaved back-substitution chains, and Zen3 0.927 → 1.102 by deleting a
`_GT_TRANSPOSE` conjunct that was using an ISA *capability* bit as an unmeasured *crossover* and so
pinned all of AVX2 to the scalar dense base — the fused leaf measured 15–52% faster there across the
whole tiny-`k` range. Zen5 has not been re-swept (box offline), so its 0.91 predates both fixes.

`trsm`'s residual has therefore MOVED from tiny-`n` to mid-`n` (Zen4 0.941 @ n=128), and it is now
decomposed rather than guessed: fitting the leaf's rate as `1/GF = α + β/KC` puts the asymptotic
microkernel rate at **44.4 GF, above PureBLAS's own dgemm (43.75)** — the inner loop is not the problem —
with an 18.6% per-slab *fixed* cost at KC=128. An opcode histogram of the emitted slab prices that cost:
scalar addressing ~22% and vector moves ~24% of it, against only ~3.2% of total leaf time for the two
8×8 transposes and ~2.8% for the back-substitution arithmetic. Removing the largest addressing
population (recomputed `(jc + c)·ldb` products) was worth 3.0% at n=128, bit-identically.

**`syrk`/`syr2k` n=4096** remains a genuine cross-µarch invariant (the conversion gap above). By contrast
`symm` ≥ 256 and `trsmR` ≥ 2048 miss on Zen4/Zen3 but *gate* on Zen5, so those are µarch-specific.

`trmm`'s crossover between recursion-over-`gemm!` and the packed routine is derived as
`5·_GEMM_UNPACK_MAX ÷ 2` rather than borrowed from gemm's own unpacked/blocked cut: in a controlled
same-process A/B the two sizes whose route changes gained 4.0% and 5.1% (n=512, n=1024), which moved the
worst cell to n=2048 — whose miss lies in the packed path itself and is a separate defect.

## Real LAPACK

`potrf`/`geqrf`/`getrf`/`gesvd` gate on all three boxes (geomeans 1.25–1.52). The small-n `potrf`
campaign — block-small Cholesky plus a **fused** 12-accumulator `trsm`-R (downdate + triangular solve in
one register pass, in both the small-n and NB=128 panel drivers) — brings AVX2 `potrf` to **BLASFEO
parity** (the MKL proxy): 0.91–1.04× its column-major `dpotrf` at n≤224 and 0.87–0.91× at n≥256, and
1.5–2.2× vs OpenBLAS fleet-wide.

### Cholesky, upper triangle

`potrf` is gated for **both triangles**. That is worth stating because it was not always true: the
benchmark harness passed `uplo='L'` to the `potrf`/`zpotrf` rows, so the upper path — a different route
(transpose into a padded scratch, factor with the gating lower kernels, transpose back) rather than a
mirror image — went unmeasured. Adding `potrfU`/`zpotrfU` rows surfaced a real miss at large `n`, whose
cause was the same power-of-two-`lda` cache-set aliasing that the factorizations already defend against,
this time inside the *copy*: the transpose read `A` strided across a row at its raw `lda`, and at
`lda·sizeof(T) % 4096 == 0` — precisely the benchmarked sizes — every column maps into one L1 set group.
Reordering the loops so the contiguous side is the read and the strided side is the (non-power-of-two)
scratch is bit-identical and worth 2.1–3.8× on the copy, taking real `potrfU` from 0.96 to 1.03 vs AOCL
on Zen4 and 0.98 → 1.04 on Zen3. Complex `zpotrfU` still sits at 0.95 for `n ≥ 1024` on Zen4 (it gates on
Zen3): the two transposes are the entire remaining gap — remove them and PureBLAS would be 3.4% *ahead*
of AOCL — but they cannot simply be deleted, since factoring the upper triangle natively measures slower
than the transpose route at every size and type.

The upper path clears both references at every size on Zen4 (worst 1.11 vs OpenBLAS, 1.03 vs AOCL) and
on Zen3 (worst 1.06). The single residual is **Zen5 n=2048 at 0.970** (re-measured 2026-08-16; it read
0.936 on the stale column); every other Zen5 size gates, which is why the Zen5 column reads below 1.0
while Zen3 and Zen4 gate. The gap is 3.0% against a within-run spread of 0.002 — small, but well clear of
the noise that produced it, so it is a real cell rather than a rounding artifact.

At `n = 32` the same route was losing for a *second*, independent aliasing reason, and the transposes
were the wrong suspect. `zpotrfU@32` read 0.695 on Zen4 and 0.765 on Zen3 while its lower sibling, the
same kernel at the same size, gated at 1.245/1.227. Decomposing before patching is what caught it: the
two transposes account for only 1.28 µs of the 3.23 µs that upper costs over lower, so a fix aimed at
them would have bought a fifth of the gap. The rest was a *copy that should never have happened*. The
base kernel stages its block through a contiguous scratch when `A isa SubArray && stride(A,2) > n`, but
that tests for the **existence** of a stride, whereas the staging is an **aliasing** remedy: it is worth
2× at `lda = 512` or `1024` and costs 1.6× at `lda = 520`. The upper route hands it a view of a padded
scratch whose leading dimension is deliberately kept *off* the way-stride, so the copy round-trip could
never pay — it was pure loss on every call. Replacing the predicate with the byte-scaled way-stride test
the padding helper already derives (no new tuning constant) took `zpotrfU@32` to **0.993 on Zen4** — a
1.0% residual inside its own 6.3% round-to-round spread, i.e. no longer adjudicable — and **1.145 on
Zen3**, with `n = 128` and `n = 256` picking up 7.5% and 5.0% from the same change.

### Triangular solves and the narrow-B `trsm`

The gating band on the `getrs`/`potrs`/`trtrs` rows is for the **factorizations**; their **solves are not
yet gated**, and the distinction is worth stating because the solves were invisible for a long time.

`A \ b` with a **single** right-hand side — the most common solve there is — ran **14× slower than
OpenBLAS** until 2026-07-29. It had never been measured: `getrs`/`potrs`/`trtrs` existed only as C-ABI
shims, and the benchmark harness compares `PureBLAS.foo!` against `LAPACK.foo!`, so with no native entry
point there was nothing to call. Adding one turned up three separate defects, all in `trsm` rather than
the solves:

1. **The blocked setup is never repaid for narrow B.** It costs `O(k·nb²)` — a triangular inverse of the
   diagonal blocks — amortised over B's columns. Measured at k=1024: 1123 µs at `nrhs=1` and 1197 µs at
   `nrhs=8`, i.e. eight solves for the price of one. Narrow B now sweeps `trsv` per column, which beat
   OpenBLAS's own `trsm` by 2.9× on that shape.
2. **A po2-`lda` A-pad that never paid.** When `stride(A,2)` was a power of two, all of A was copied into
   an odd-`ld` scratch to dodge cache-set aliasing — `O(k²)` work, charged even to a single-column B.
   Measured across **all sixteen** combinations of side × uplo × transA × B-width: it cost between **+3%
   and +357%** and won nowhere, including at square B, where removing it slightly *improved* the gate
   cell. Deleted.
3. **A ragged SIMD lane costs a full lane.** The kernel processes B's columns in `W`-wide lanes, so a
   partial lane is charged as a whole one. At k=512: `nrhs` = 1…7 cost 96/180/278/381/464/543/675 µs —
   about 95 µs *per column* — while `nrhs=8` cost 94 µs in total. Widening a ragged B into scratch before
   the solve collapses that (2.9–6.8× at k=512). This is the same defect the *upper* fused leaf had,
   fixed there in an earlier campaign; it survived in the lower/notrans path, which is exactly the one
   `getrs`/`potrs` use — hence `nrhs=1` being the worst case of all.

Result, Zen4 vs OpenBLAS, `nrhs=1`, n = 8…2048:

| | 8 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|---|
| `getrs` before | 0.817 | 0.479 | 0.389 | 0.276 | 0.268 | 0.092 | **0.072** | 0.113 |
| `getrs` after | 1.142 | 1.013 | 1.084 | 1.095 | 1.036 | 0.994 | 0.984 | 1.162 |
| `potrs` after | 1.157 | 0.980 | 1.196 | 1.667 | 2.470 | **4.550** | **4.913** | 4.120 |

A residual remains at `nrhs = 8`, which falls between the two mechanisms — too wide for the per-column
sweep, and on Zen3 (`W=4`) already two full lanes, so the ragged-tail fix does not reach it either.

The threshold guarding the sweep is worth recording as a tuning lesson: it was first written as
`nrhs ≤ k/(4·_TRSM_DBASE)`, on the reasoning that a larger `k` amortises the setup over more columns.
That was the wrong **shape** — blocked costs `setup + nrhs·b`, the sweep costs `nrhs·v`, and all three
terms scale as `k²`, so the crossover is **`k`-invariant**. The growing rule selected the sweep exactly
where it loses, and was itself the cause of the `nrhs=8` misses it was meant to fix.

After those fixes the Zen4 gate sweep reads, vs **OpenBLAS**: `getrs` geomean 1.05 / worst 0.99,
`potrsL` 2.20 / 0.93, `potrsU` 2.74 / 1.07 ✓, `trtrs` 1.11 / 1.02 ✓ — `potrs` reaching **4.9×** at
n=1024. Vs **AOCL** they still miss: `getrs` 0.93 / 0.85, `potrsL` 0.94 / 0.78, `potrsU` 1.13 / 0.91,
`trtrs` 1.07 / 0.88. So under `max(OB, AOCL)` the solves remain open, and `trsm` itself passes OpenBLAS
(worst 1.18, up from 1.04) while still missing AOCL (worst 0.90).

### Tridiagonal

`gtsv`, `gttrf`, `gttrs`, `pttrf`, `pttrs`, `ptsv` gate `≥ max(OpenBLAS, AOCL)` on **all three µarchs**,
every result bit-identical to the reference. Worst-case vs AOCL: `gtsv` 1.20–1.22, `gttrf` 1.51–1.54,
`gttrs` 1.03–1.06, `pttrf` 1.11–1.12, `ptsv` 1.04–1.05; vs OpenBLAS the same ops run 1.31–1.57 and
`pttrs` 1.67–1.73.

These are `O(n)` serial three-term recurrences, so the limit is a divide→multiply→subtract **latency
chain** (~19.5 cyc/elem on Zen4), not flops — and every lever was about how data moves *around* that
chain rather than about the arithmetic. Three cost roughly 10 cyc/elem each: a live `fadd x, 0.0` left on
the real path of a complex-shaped expression (LLVM may only fold it under `nsz`, since `x + 0.0` is not
an identity for `x = -0.0`); re-reading a recurrence variable the previous iteration just stored; and a
per-element inner loop with a *runtime* trip count. A fourth, an extra store stream, cost 10.1 and was
hoisted into a bulk pass priced at 0.3. Unusually for this project these fixes **transfer across
µarchs** — they are properties of the generated code, not cache- or ISA-dependent tuning.

`pttrs` is the one cell at parity rather than above it (0.99 vs AOCL). Both libraries run at
~12.2 cyc/elem, which is exactly the analytic bound: the forward sweep's recurrence is multiply+subtract
(~6 cyc) and the backward sweep's divide sits *off* the chain, since `B[i]/D[i]` does not depend on
`B[i+1]`. That is a *shared* dependency-chain bound, not a gap, with 0.0% run-to-run drift.
Register-carrying the recurrence and hoisting the `uplo` test out of both loops were each measured and
are neutral — LLVM already does both.

### Pivoted QR (`geqp3`)

**Blocked, and now gating.** `geqp3` used to be the worst routine in the real LAPACK surface: a BLAS-2
implementation racing a blocked BLAS-3 one, ratio degrading monotonically with `n` — **0.34 / 0.18** vs
OpenBLAS and **0.34 / 0.19** vs AOCL on the Zen4 sweep. It had **never been benchmarked** before
2026-07-29 — it routed and passed correctness, which reads as coverage while measuring nothing.
`sytrf`/`sytrs` were in exactly the same state.

The first blocked `dlaqps` port was verified correct but **not faster**, and larger panels were *worse* —
the diagnostic that the cost is the panel's own per-column BLAS-2 calls, not the trailing gemm. The
shipped retry (`src/lapack/laqps.jl`, wired from `geqp3.jl`) reaches the kernels pointer-direct, adds a
SIMD downdate in the unblocked remainder and routes the `laqps` gemv-T through the blocked kernel. The
row now gates on **all three** boxes (1.11 Zen3 / 1.02 Zen4 / 1.03 Zen5). See §5 of [Tuning](tuning.md).

### Bunch–Kaufman (`sytrf`/`hetrf`)

**Blocked since 2026-07-30.** They dispatched straight to the unblocked `_sytf2_*` (rank-1/rank-2 BLAS-2
column downdates) against OpenBLAS's blocked `dlasyf` — hence the old 0.56 / 0.15 and 0.59 / 0.15, and
`hetrf` measuring **0.116** vs OpenBLAS at n=1024. All four variants are now blocked: real-symmetric and
complex-symmetric through a `dlasyf` port, Hermitian through a separate `zlahef` port, both `uplo`.
`zlahef` is deliberately NOT `dlasyf` plus a few conjugations — it realifies the diagonal at points that
have no symmetric counterpart, and its 2×2 inverse is asymmetric in the conjugates with the asymmetry
FLIPPING between triangles. Speedup over the kernel replaced: 1.5–2.9× real, 9.8× Hermitian, 11.3×
complex-symmetric at n = 1024–2048. Dispatch-path PB/OB, freq-locked, `uplo='L'`, n = 64…2048:

| | Zen3 | Zen4 |
|---|---|---|
| `sytrf` F64 | 1.28 / 1.42 / 1.46 / 1.38 / 1.22 / 1.15 | 1.26 / 1.60 / 1.69 / 1.50 / 1.24 / 1.10 |
| `hetrf` C64 | 1.37 / 1.16 / 1.17 / 1.06 / 1.07 / 1.12 | 1.37 / 1.16 / 1.33 / 1.20 / 1.16 / 1.13 |
| `zsytrf` C64 | 1.33 / 1.03 / 1.05 / 1.00 / 1.01 / 1.08 | 1.27 / 1.10 / 1.16 / 1.08 / 1.08 / 1.08 |

The row misses on ONE cell: vs AOCL on Zen4 the gate sweep reads 1.40 / **0.99**, from n=2048 measuring
0.98–1.02 across rounds — parity within noise rather than a structural gap. Every other size clears, and
on **Zen3 both references clear outright** (1.34 / 1.04 vs OpenBLAS, 1.49 / 1.17 vs AOCL). The panel
width is `8·⌈√n/8⌉` clamped to [16,96] for real — a formula over the problem size, validated on two
microarchitectures — times a **measured** multiplier for complex, because the complex optimum is
microarchitecture-dependent (Zen4 selects 1, Zen3 selects 3) while the real one is not. That split was
made by measurement: the real formula gates on both boxes, but applied to complex on Zen3 it produced
three gate misses (0.95–0.97).

### Banded LU (`gbtrf`)

**Blocked since 2026-07-30**, and the one residual has a known cause. `gbtrf` was a faithful port of
reference `dgbtf2` — the UNBLOCKED banded LU — racing OpenBLAS's blocked `dgbtrf`, which is where
0.96 / 0.48 (OB) and 0.78 / 0.34 (AOCL) came from. It is now a port of blocked `dgbtrf`, with the two
corner blocks (A13/A31, whose parts fall outside the stored `ldab` window) staged dense as the reference
does. Zen4 vs OpenBLAS at `kl = ku = n/8`: 1.90 / 1.67 / **0.93** / 1.36 / 1.87 / 1.55 / 1.35 for
n = 8…2048 (**1.49 / 0.93**); vs AOCL **1.19 / 0.72**. **Zen3 clears OpenBLAS outright: 1.41 / 1.16.**

The single miss is n=128, i.e. `kl`=16, and it is **not** a blocking question: OpenBLAS is *also*
unblocked at that width (its ILAENV `nb`=64 exceeds `kl`, so `dgbtrf` takes the `dgbtf2` branch), so we
are losing to its **unblocked** band downdate. The remaining lever is the quality of `_gbtf2!`'s rank-1
band update against `dger`, not the panel width — no `nb` rescues that cell past ~0.93, measured across
`nb` = 2…16.

Two knobs, and the second exists because of a mistake worth recording. The panel width is a formula over
`kl` (`8·(1 + kl÷128)`), fitted to the measured optimum at fourteen band widths on Zen4. Shipping only
that **regressed Zen3** from PASS to FAIL (1.43 / 1.18 → 1.32 / 0.91): at `kl`=16 blocking wins on Zen4
(0.93 against 0.796 unblocked) but *loses* on Zen3 (0.91 against ~1.18), because at that width OpenBLAS
is unblocked too, so the cell is a scalar-downdate race rather than a blocking race. The
blocked-vs-unblocked floor is therefore a **measured** knob — Zen4 selects 16, Zen3 selects 24 — which
restored Zen3 to 1.41 / 1.16 with Zen4 unchanged. That is req#8(b)'s "derive → validate on the **fleet**
→ ship" catching a one-box derivation.

### Pivoted Cholesky (`pstrf`)

`pstrf`'s blocked pivoted factorization clears **AOCL completely on both Zen4 and Zen5**, both triangles,
every size (Zen5 re-measured 2026-08-16: worst 1.20 for `uplo='L'`, 1.19 for `uplo='U'`). Against
**OpenBLAS** it clears everywhere except a narrow window that is consistent across µarchs and mild: Zen4
n=48 **0.943** and n=64 **0.921**; on Zen5 the window sits at small n rather than n=64 — `pstrfU` n=32
**0.905** is the one cell clear of its own noise (spread 0.022), while `pstrf` n=8 **0.876** and `pstrfU`
n=8 **0.930** both fall *within* their round-to-round spread (0.289 and 0.142 — tiny-n cells are too
noisy to adjudicate and are not counted as misses).

Those cells were known but had never been *gated* — the harness passed `uplo='L'` for both sides, so the
separate 'U' code path (the pivoted panel and the trailing update both mirror) was never measured. A
`pstrfU` row now tracks it. The large-n wins come from batching the pivot row swaps per panel: they are
stride-`lda` and were ~47% of the runtime, so the swap — not the BLAS-3 update — was the bottleneck.
⚠ Float64; Zen3 not yet measured for this routine.

One caveat on reading the published table: the Zen4 gate sweep reports `pstrf` worst = 0.85 vs OpenBLAS,
which is **not** one of the cells above — it is n=8, where the per-round ratios come out bimodal
(1.03 / 1.00 / 0.79 / 1.01 / 1.01 / 0.79 / 0.80 / 1.03). Rounds alternating like that at sub-microsecond
scale is the ABBA ordering effect, not a stable regression; the sweep's size grid (8, 32, 128, …) skips
48 and 64 entirely, so it never samples the real residual.

### Packed Cholesky (`pptrf`)

**`uplo='L'` now clears OpenBLAS** on the Zen4 gate sweep (geomean 1.11, worst 1.04) after the per-call
overhead fix described below — but it still misses **AOCL** (1.17 / 0.93). `uplo='U'` is the mirror
image: 3.30 / 1.43 vs AOCL ✓ but 1.07 / **0.92** vs OpenBLAS. Neither triangle clears both references,
and they fail against *different* ones.

Historical detail, kept because the diagnosis generalises — which is why it needs both. Zen4, Float64,
n = 8…2048: `uplo='L'` clears OpenBLAS at every size (1.05–1.37) and clears AOCL except n=32 (**0.949**)
and n=48 (**0.980**). `uplo='U'` beats AOCL by a wide margin (1.17–**6.27**, growing with n) but loses to
OpenBLAS at n=48…2048 (**0.918–0.990**). That 6.27× is **not** evidence of quality: timing the *same
reference routine* under each library, AOCL's `dpptrf` `uplo='U'` is **6.0× slower than OpenBLAS's** at
n=1024 (175 281 µs vs 29 297 µs) — near-stock netlib — while on `dpotrf` the two sit within 20% of each
other. AOCL optimizes dense Cholesky and does not optimize the packed variant. The gate is unchanged by
this — it is `max(OpenBLAS, AOCL)`, so a slow AOCL never *lowers* the bar, it merely means `max()` here
**equals OpenBLAS**; both references are still measured, and both sets of misses above still count.

The lower path improved once its gap was decomposed: the `spr!` *kernel* already ties or beats AOCL at
every order, and the whole deficit was per-call overhead — the public entry plus a `SubArray` built per
column — compounded by the lower path reusing the *upper* path's cutoff constant, which meant `spr!` was
never called at all at n=32 (worst cell 0.738 → 0.949). See §5.3 of [Tuning](tuning.md). The remaining
cells are gaps, not ceilings.

### Banded Cholesky (`pbtrf`)

`pbtrf` gates fully on **Zen4** (Float64, `kd = 32…384`, n ∈ {1024, 4096}, both triangles: 1.02–3.28×
AOCL, 1.49–2.00× OpenBLAS) and beats **OpenBLAS on all three µarchs**, but it does **not** yet clear AOCL
fleet-wide. Residuals are all `uplo='U'` at mid bandwidth: Zen5 `kd=128/256` at 0.989/0.969 (re-measured
2026-08-16 — `kd=192` now gates and `kd=384` lands on 1.0 within its own spread, so that residual
narrowed from four cells to two), Zen3 `kd=128/160/192` at 0.881–0.984 (plus `uplo='L'` `kd=160/192` at
0.959–0.995). Zen3 is the worst box and also the one whose `_pbtrf_ucross` bracket is derived from a
narrower `_vwidth` (W=4 → the switch lands on 192, itself a 0.90 cell), so the leading hypothesis is a
candidate bracket putting the optimum at an edge — the failure this project's own history records for
the wide-band panel bracket. Not yet confirmed by measurement. Correctness is clean on all three boxes
(11368/11369, identical).

`uplo='U'` dispatches between two kernels at a measured bandwidth crossover: a conj-transpose re-pack
onto the (much faster) lower kernel for narrow bands, and a native upper-storage port once the re-pack's
diagonal walk starts to dominate. The panel width likewise has two measured regimes — clamping the
wide-band width to `kd` collapses the in-band panel and was the routine's only Zen4 gate miss. See §5.2
and §5.3 of [Tuning](tuning.md).

### Symmetric / Hermitian eigensolver

With eigenvectors (`jobz='V'`) it clears both references on the Zen4 sweep (1.32 / 1.25 vs OpenBLAS,
1.36 / 1.19 vs AOCL). **Values-only (`jobz='N'`) does not**: 1.13 / **0.98** vs OpenBLAS, though
1.31 / 1.14 vs AOCL — so the coverage row is BELOW 1.0 on the strict worst-case rule, driven by a single
size. An earlier version of the coverage page claimed it gated "at every size on the whole fleet … both
`jobz`"; that was written against the OpenBLAS-only criterion and the `jobz='N'` cell does not survive
the two-reference one.

Three Level-3 pieces get the vectors path there: the two-sided reduction (`sytrd`/`hetrd`) is blocked
`dlatrd`+`syr2k`/`her2k`; the eigenvector back-transform (`ormtr`/`unmtr`) is compact-WY block
reflectors; and the divide-and-conquer solver (`stedc`) assembles eigenvectors once (no per-merge column
copies) and combines survivors with the `dlaed3` COLTYP **two half-height gemms** (exploiting the
block-diagonal zero structure — ~half the combine flops, matching libFLAME).

## Complex

The complex surface is SIMD across all levels, in portable SIMD.jl kernels (no x86 intrinsics); the
generic scalar path remains for AD.

**CL1** is at parity or better fleet-wide. `dznrm2` is 7–9× (same scaled-accumulation story as real
`nrm2`); `izamax` 1.2–1.8×. The former AVX2 `zdotc`/`zdotu` small-n dip is closed by a parity-preserving
fold epilogue (shared with complex gemv-T/C). Residuals: `zaxpy` at the L3 edge (0.94–0.96) and
`zdotc`/`zdotu` on Zen5 at DRAM sizes (0.89–0.94, bandwidth).

**CL2** gates broadly on all three boxes. The complex gemv-T/C small-n dip on AVX-512 (was 0.68–0.74 at
n=32) is closed by a parity-preserving fold epilogue (Vec{2W} deinterleave + two horizontal sums → one
halving fold to [Σeven,Σodd]); `zgemvN` on Zen5 (was ~0.91 across small/mid n) by sign-folding the `ci`
broadcast so the per-row epilogue drops an FMA-port op. `ztrsv`/`ztrmv` large-n on AVX-512 were largely
closed (n=2048: `ztrmv` 1.13 Zen4 / 1.17 Zen5, `ztrsv` 1.12 Zen5 and 0.99 Zen4) by routing the triangular
off-diagonal scatter through the tuned `ri` gemv — a stale branch had used the older row-tile kernel.
Remaining residuals cluster at **n=1024** on the AVX-512 boxes: `ztrsv` (0.948 Zen4, 0.936 Zen5) and
`ztrmv` (0.948 Zen4) — a coupled upper/lower-triangular tradeoff — alongside the complex gemv trio
`zgemvN`/`zgemvT`/`zgemvC` (0.848–0.877 Zen4, 0.935–0.951 Zen5). `zgeru` dips mid/large-n (0.817 Zen4 at
n=1024, 0.857 Zen5 at n=4096), and `zhemv` misses only on Zen3 at large n (0.919 at n=2048, near the DRAM
roofline) while gating 1.9–2.5× on both AVX-512 boxes.

**CL3.** `zgemm` beats OpenBLAS fleet-wide (geomean 1.26–1.40; Karatsuba 3M at mid/large n). The rank-k
ops gate within a few percent. `@simd ivdep` on the complex microkernel's k-loop (4 FMA/cell) helped the
small-n complex trmm.

### `ztrsmR` — codegen, not blocking

**`ztrsmR` (complex side-R) improved at every size on 2026-08-11**, and the cause was codegen rather than
blocking or cache. Four cells crossed into PASS — Zen4 n=32 0.974→1.196 and n=128 0.933→1.022, Zen3 n=32
0.829→1.091 and n=256 0.975→1.111 — and the fleet's worst complex-BLAS-3 cell, Zen3 n=128, went
**0.768→0.926**. The cell was first *attributed*: it fits `1/GF = α + β/n` with an 83.5 GF asymptote
(predicting n=192 to 0.2%), so it was a 14% per-call fixed cost, not a size-specific conflict;
sum-of-parts came to 99.7% of the call, and PB's own `zgemm` is flat at ~42.6 GF from k=48 to k=512 (≈95%
of Zen4's double-pumped AVX-512 ceiling), which left the register-tile leaf as the whole target. Inside
it, the update was written `muladd(V2(-cr), xv, …)` — the negation sitting on the *scalar* coefficient
between its load and its splat, which blocks both the `vfnmadd` opcode and AVX-512's embedded broadcast
`(mem){1to8}`. LLVM therefore emitted `vmovsd` + `vxorpd` + `vbroadcastsd` per coefficient: 96
instructions of coefficient preparation against 64 useful FMAs. Negating the B vector once per step
instead is exact (`nswap(-v) == -nswap(v)`, so the swapped operand needs no separate handling) and takes
the loop from **2.95 to 1.96 instructions per useful FMA**, with residuals identical to the last digit.
Four cheaper hypotheses were falsified first — power-of-two `lda`/`ldb`, a wider column block, load-slot
pressure, and tile-loop interchange — and the same rewrite applied to the complex *side-L* slab compiled
byte-identical, because LLVM had already chosen `vfnmadd` there on its own. Remaining: Zen3 n=128 (0.926)
and Zen4 n≥512 (~0.99).

### Complex `trmm` small-n

The complex `trmm` small-n dips were partly closed by a dispatch fix. `_uker_cmplx!` takes a `FULL` flag
that selects unmasked A-loads and unmasked stores, and the two small-`trmm` drivers hardwired it off on
every branch — so every tile ran masked even when full, which at these sizes is *every* tile. Dispatching
it the way the gemm sweep always has is worth **+16.7%/+13.1%** at n=8/32 on Zen3 and
**+0.4%/+3.6%/+3.8%/+2.7%** at n=8/32/48/128 on Zen4 (measured in one process with the arms alternated
and verified bit-identical). `n=8` now gates on both microarchitectures for both ops, and Zen4 `ztrmmR`
n=128 crossed AOCL (0.976 → 1.003). The split follows the ISA — a masked op is `vmaskmovpd` on AVX2 and
k-register predication on AVX-512 — though the residual Zen4 gain is not fully explained by that alone.

What remains open on this path is smaller than the raw ratios suggest, and separating the two took a
second measurement. A targeted single-op run and a full sweep — same code, both with every arm inside one
run — disagreed by 3–4% at n=32 (`ztrmm` 0.933 vs 0.903, `ztrmmR` 0.871 vs 0.839), which is the regime
difference between measuring an operation alone and measuring it in sequence with the rest of the suite.
The full sweep is the canonical artifact, so those are the numbers quoted here. More importantly,
`bench/gate_gaps.jl` reports each cell's shortfall against its own round-to-round spread:

- `ztrmmR` n=32 on Zen3 — 0.839, a 16.3% gap against a 0.047 spread. **Real.**
- `ztrmm` n=128 on Zen3 vs AOCL — 0.917 at a spread of 0.003, the most tightly measured miss in the whole
  table. **Real**, and on the *packed* path, which the dispatch fix provably never reached (its control
  read exactly 1.000).
- `ztrmm` n=32 on Zen3 — a 9.5% shortfall against a **0.115** spread. That cell is *within spread*: it
  needs more rounds, not engineering, and optimising against it would be optimising against measurement
  error.

Also still open, and outside their spreads: `ztrsm` 0.89 (n=128) and a `zsymm`/`zhemm`/rank-2k small-n
dip — materialize+microkernel overhead on the small-n complex-L3 path.

That residual has since been decomposed rather than guessed at. Holding `k` at 32 and varying only the
number of columns of `B` separates per-call cost from per-column cost through the public entry point, by
fitting `1/GF = a·(1/c) + b`. At the gate shape `ztrmm` carries **15.9%** fixed per-call cost, while
`zgemm` at the same `k` — same kernel family, no triangle — carries **0.7%** and holds its rate flat to
within 0.8% across its whole sweep. Shared BLAS-3 entry cost is therefore negligible, and essentially all
of `trmm`'s fixed cost is the O(k²) triangle materialize. A direct-read fused-triangle base closed the
trmm dips in development but its runtime `Val` args broke trim-safety and it was reverted; the trim-safe
refix (compile-time `Val` dispatch) is the follow-up, and this puts its value at about 15% against the
2.9–14.8% the individual cells need.

A second and unrelated inefficiency shows in the same fit: with the triangle amortized to nothing, `trmm`
still costs 0.1535 µs per column against `gemm`'s 0.2037 for *half* the multiply-accumulates — 66%
efficiency. That is the granularity of the K-TRIM tiling in steady state, not a fixed cost, and the
triangle base would not address it. (Two scratch-layout attempts were measured neutral and reverted: a
non-po2 scratch, and sizing the shared `symm`/`hemm` scratch exactly — the latter is *slower* at
power-of-two n, where an exact leading dimension is itself an aliasing stride.)

### The 3M routing constant

Complex BLAS-3 **was** further out and lost to AOCL almost across the board — `zgemm` 0.922, `zsyrk`
0.941, `zher2k` 0.908, `zsyr2k` 0.894, `ztrmm` 0.943, with only `ztrsm` gating. The cause was one routing
constant, and it is worth recording because the evidence for it had been sitting in the repository for a
month.

`zsyrk`/`zherk`/`zsyr2k`/`zher2k` read 0.94–0.99 against AOCL on Zen4 while **the same source** read
1.19–1.31 on Zen3. That inversion is not a microarchitecture property: the source routed a different
*algorithm* per SIMD width. The Karatsuba-3M route — three **real** products on the split real/imaginary
parts, 25% fewer flops than the direct four-FMA complex kernel — was enabled only for `_vwidth == 4`.
Zen3 got the flop cut; Zen4 did not. The default's own comment said AVX-512 was *"untested"*, not
falsified, and the internal notes listed "3M on AVX512" under **unbuilt** levers with the prediction that
it would win there too "via the flop cut, since AVX512 complex is throughput- not latency-bound".

It does, and the width test was deleted rather than retuned. Nothing physical made it width-dependent:
the 25% reduction is algebraic, and 3M's split/combine overhead is O(n²) against O(n²·k) of product,
which the existing size window already bounds — so the test was an artefact of only ever having measured
AVX2, and removing it takes out a hardware-gated literal instead of adding a tuning knob. Measured
in-process (Zen4, 3M/direct, so lower is faster): `zgemm` 0.797/0.787/0.818/0.801/0.800 at n=128…2048,
rank-k 0.81–0.95. On the gate that is `zgemm` 0.952 → **1.213** at n=128 and 1.000 → **1.255** at 2048,
`zsyrk` 0.958 → **1.133** at 512, and every complex rank-k cell at n ≥ 256 converted.

The **secondary** effect is larger than the primary one: everything layered on complex gemm moved without
being touched — `zhemm` 1.004 → **1.136**, `zsymm` 0.994 → **1.146**, `zherk` 0.982 → **1.053**,
`ztrsmR` 0.987 → **1.084**, `ztrmmR` 0.969 → **1.005**, `ztrsm` 1.067 → **1.095**. Zen3, which already ran
3M, is flat across the same sweep — which is the control that says the change did one thing only.

The rank-k window's lower edge was the next thing, and it turned into a lesson about knobs rather than
about kernels. Below 256, complex rank-k could not reach 3M at all: for `trans='N'` the *unpacked* branch
is tested before the packed path, and on AVX-512 that branch owns everything up to n=192 — so lowering the
edge alone would have changed nothing. Fixing that exposed the real question: **where should the edge
be?** The honest answer was that it is not derivable. The crossover carries a ratio of *our own two
microkernels'* rates — the complex kernel is latency-bound on AVX2 and throughput-bound on AVX-512 — so no
cache or ISA constant predicts it, every throughput model gives the wrong *sign*, and the tidy `768/W`
that fits both machines has no physical criterion behind it. Worse, the measured crossovers were far
apart (≈80 on Zen4, ≈160 on Zen3), so no single value existed: 128 would have cost Zen3's `zherk@128`
8.2%.

So the overhead was removed instead of the threshold being located. 3M's cost over the direct complex
kernel was two extra passes — a strided split of the operands, and three full n×n product buffers read
back by the combine. Both are now fused away: the third Karatsuba panel is derived from the *packed*
panels (the complex pack already deinterleaves into two real panels, so the third is one add in a loop
that exists), and the combine happens per micro-tile into a small scratch instead of via n×n buffers,
with the triangle mask moving from the microkernel's store into the combine bounds.

That did not eliminate the threshold — and the reason is worth stating, because the original claim was
that it would. **The pack itself is O(n·k), and 3M packs three panels where the direct kernel packs
two.** That +50% is O(n·k) against O(n²·k) of product, so overhead-per-flop still decays as 1/n; three
real products structurally require three real panels. What fusion bought was moving the crossover
(≈160 → ≈112 on Zen3, below 64 on Zen4) until **one value finally served both machines**. Measured
fused-vs-packed at n=128: 0.886/0.887 on Zen4, 0.956/0.963 on Zen3 — both win, so 128 ships, labelled in
the source as the empirical bound it is rather than dressed as a derivation.

The result on the gate: `zsyrk@128` 0.894 → **1.002** and `zsyr2k@128` 0.899 → **1.011** on Zen4, with
`zherk` and `zher2k` gaining margin (1.063 → 1.135, 1.068 → 1.111); on Zen3 `zsyrk`/`zherk` at 128 rise to
1.037/1.036. Fusion is a strict improvement everywhere it applies — same kernels, strictly less traffic —
measured 0.83–1.00 against the unfused path on both boxes.

What remains is **n = 32 and below**, outside the window on every machine: `zsyr2k` 0.910 and `zher2k`
0.906 on Zen4, `zgemm` 0.902, and `ztrmm` 0.972. Complex rank-2k also has a second, independent fence on
AVX2, which is why Zen3's `zsyr2k@128` did not move; that path needs the rank-2k reformulation (one
product plus a symmetrized add) rather than a window change.

### Complex `symm`/`hemm`

Complex symm used to materialize a dense n×n copy of the symmetric operand and call gemm; complex hemm
avoided the copy but ran through a *fork* of the complex gemm driver that had drifted, hardwiring off the
two specializations (overwrite-C when β=0, and the α=1 store) that the driver itself dispatches —
precisely the case the benchmark measures. Wiring those back in and routing complex symm through the same
packing path removes the copy entirely. Measured wrapper cost against a bare gemm of the same shape fell
from 1.065/1.069/1.053/1.026 at n=128/256/512/1024 to 1.026/1.017/1.004/1.002. On Zen4 `zhemm` now gates
at every size from 32 up (1.004–1.012) and `zsymm`'s three remaining shortfalls are each smaller than that
cell's own round-to-round spread; on Zen3 both gate across 128–2048 at 1.18–1.28. What is left in this
family is tiny-n — n=8 and n=32, below the packing cut, still on the copy path — and real `symm` at
n≥256, whose packed path has not yet had the same β=0 fold applied.

This family used to carry the caveat that the engine itself (`zgemm`) missed, so nothing above it could
convert. That is no longer true above the 3M window's lower edge — `zhemm` and `zsymm` now gate at
1.136/1.146 on Zen4 purely because the engine improved. It remains true *below* it: `zgemm` at n=32 is
untouched, and the tiny-n `symm`/`hemm` cells sit on the copy path beneath the packing cut.

### Complex LAPACK

`zpotrf` gates fleet-wide. `zgetrf` **now gates fleet-wide** (was Zen3 worst 0.80): a rank-2
SIMD-`izamax` `getf2` panel, a derived complex block width, a base-32 crossover, and a po2-aliasing-dodge
scratch. The only residual is Zen3 n=256 (0.94), localized to PB's complex gemm at the trailing
208×208×48 shape (a gemm-kernel gap, not a getrf-structure one). `zgeqrf` **gates at every size
fleet-wide** (1.07–1.83; was geomean 0.76–0.85): the complex trailing update was rebuilt to mirror the
real path (`herk` for VᴴV at half the flops + `trmm` for TᴴW), the reflector norm moved from per-element
`hypot` to SIMD `_nrm2`, the unblocked rank-2 panel runs while the matrix is L3-resident, and the AVX2
trailing update is chunked to bound the Karatsuba-3M scratch below ½L3. `zgesvd` — singular values only,
vs `zgesdd('N')` — **gates at every size** (real + ComplexF64): the large-n collapse was fixed by a
blocked complex bidiagonalization (zlabrd/zgebrd port), and the small-n floor by switching the values
path to **dqds** (`dlasq1–6`) — OpenBLAS's `gesdd('N')` uses the sqrt-free dqds recurrence, not
Golub–Kahan QR; matching it lifted n=32 from ~0.86 to ~1.07 (real *and* complex, since the bidiagonal is
real). ComplexF32 small-n (n=32/48 ≈ 0.9) is the remaining sub-gate, a `cgebrd` limit. `zgeqrf` was
validated on square inputs; tall/skinny QR is a separate benchmark shape.

## Per microarchitecture

**Zen4 (AVX-512, double-pumped).** The tuning target. Real BLAS-1/2 gate essentially everywhere; the
residuals are the BLAS-3 set above (`trsm` 0.941 @ n=128 — n=32 now gates at 1.056 — `syrk` 0.937 @
n=4096, `trmm` 0.958 @ n=2048, `symm` 0.955). The complex residuals are the shared LAPACK gaps plus a
handful of BLAS-2 cells: `zgeru` 0.906 (n=1024), `zgemvN` 0.954 (n=512) and `ztrsv` 0.865 (n=1024).

**Zen5 (AVX-512, native 512-bit).** Clears every AVX2 ceiling but shows a disjoint residual profile — the
reason the gate is per-machine. Open: `gemvN` (0.942, n=2048), `ger` (0.896, n=2048) and the complex pair
`zgemvN`/`zgeru` (0.895/0.892, n=2048).

**Zen3 (AVX2).** The hardest target: 16 ymm registers vs AVX-512's 32 zmm. Real surface gates except
`trmm`/`trsm` worst sizes; complex carries the widest residual set (`zdot`, `ztrmm` both sides, `ztrsm`,
`zgetrf`). The `potrf` small-n campaign (block-small Cholesky + a fused 12-accumulator `trsm`-R) reaches
**BLASFEO column-major parity at n≤224** (0.91–1.04×), 1.5–2.2× vs OB.

### Zen5's complex-L3 transfer — a retracted explanation

An earlier version of the coverage page predicted that Zen5's complex-L3 cells would move like Zen4's
because both are AVX-512, and that none of that column should be read as a µarch difference. Every one of
the eleven cells did rise, and five crossed the gate — but they split cleanly in two. The
`symm`/`hemm`/`trsmR` family transferred in full (`zsymm` 0.954 → 1.170 against Zen4's 0.994 → 1.146;
`zhemm` 0.979 → 1.153; `ztrsmR` 0.957 → 1.047), while **`zgemm` and `zsyrk` transferred only partially** —
0.947 → 0.996 and 0.949 → 0.969, against Zen4's 0.952 → 1.213 and 0.958 → 1.133.

That split is a genuine microarchitectural signal rather than older code, but **its cause is currently
unknown**. An earlier revision attributed it to Zen5 having twice Zen4's FMA throughput, and that
explanation has since been measured and **retracted**. Taking each box's classical-gemm OpenBLAS arm as an
honest throughput probe (the PureBLAS arm cannot serve here — Strassen-Winograd makes `2n³/t` overstate
the hardware FMA rate), all three boxes sustain the *same* per-cycle width: **6.84, 7.48 and 6.89 F64 FMA
lanes per cycle** on Zen3, Zen4 and Zen5, i.e. an 8-lane ceiling everywhere. Zen5 is therefore **not** less
FMA-bound than Zen4 — it has the same lanes per cycle at a lower clock, giving it the fleet's *lowest* F64
peak (31.7 GF against Zen4's 44.9 and Zen3's 58.9; the narrowest-ISA box has the highest peak because it
clocks highest). Untested candidates for the real cause: Zen5's 48 KiB L1d, unique in the fleet, and the
lower clock shifting the memory-versus-compute balance.

## vs AOCL — the residuals

**PureBLAS matches-or-beats AMD's own library on gemm and every LAPACK factorization — real and complex —
by 5–80%.** Verified PB/AOCL at mid/large n (>1 = PureBLAS faster; Zen4, single-thread),
alongside PB/OpenBLAS for context:

| op | PB / OpenBLAS | PB / AOCL |
|---|---|---|
| gemm    | 1.05–1.11 | 1.08–1.24 |
| potrf   | 1.16–1.28 | 1.12–1.20 |
| geqrf   | 1.63–1.80 | 1.53–1.70 |
| getrf   | 1.14–2.24 | 1.59–1.78 |
| gtsv    | 1.41–1.43 | 1.20–1.22 |
| gttrf   | 1.53–1.57 | 1.51–1.54 |
| pttrf   | 1.11–1.13 | 1.11–1.12 |
| ptsv    | 1.31–1.33 | 1.04–1.05 |
| **pttrs** | 1.67–1.73 | **0.99** (shared chain bound) |
| **trsm** (side-L)  | 1.13–2.73 | **0.93–1.07** |
| **trsmR** (side-R) | 1.03–2.29 | 0.97–1.73     |

The one residual vs AOCL is real **side-L `trsm` at mid-n** (worst-size 0.92–0.97): a codegen-scheduling
gap on the fused-leaf kernel — the fused-back-substitution's latency chain runs at ~2 IPC where BLIS's
hand-scheduled assembly hides it — not a structural lever (building AOCL's own packed-triangle structure
in portable SIMD.jl lands at the same ~0.95).

Side-R `trsmR` (the potrf/getrf panel shape): both `transA` route through the fused leaf via the
reflection identity Ã = J·Aᵀ·J. Its long-standing AVX2 worst cell (n=128, 0.817 vs AOCL — the worst
BLAS-3 cell on the fleet, and recorded in the kb as unexplained after two falsified hypotheses) was closed
to **0.917 on 2026-08-10**, with **n=512 and n=1024 crossing into PASS** (0.941→1.105, 0.985→1.086) and
every size improving. Cause: at `transA='T'` the reflection branch does not fire, so A reaches the leaf
**verbatim at the caller's lda**; for a square operand that is a power of two, and at k=128 the 1024-byte
column stride is a quarter L1 way period. Padding A into an odd-`ld` scratch recovers 11.5–15.1% at the
leaf. The pad is gated on whether the leaf actually **re-reads** A — i.e. whether it spills, which is true
on AVX2's 16 ymm and false on AVX-512's 32 zmm — because on a non-spilling box the same pad measured
*null* at n≥512 and 4.9% *slower* at n=128. Zen4 therefore never pads and is byte-identical to before; its
worst size remains ~0.95 (n=2048).

The former **small-n `gemm` dip** (~0.92× AOCL at n≤256, where BLIS's lower packing overhead won) is now
**closed** by a *direct-B microkernel* that skips the B-pack entirely when B is contiguous in the k-index
(col-major, no transpose) — dgemm now gates **≥0.98× AOCL fleet-wide from n=128 up** (Zen3/Zen4/Zen5), with
no large-n or OpenBLAS regression.

## Where the misses are (Zen4 sweep)

36 of the 83 measured rows clear `≥ max(OpenBLAS, AOCL)` at *every* size. The rest miss somewhere,
usually narrowly (0.9–0.99 at one or two sizes — often the smallest, where per-call overhead dominates
and the measurement is least stable). The deepest remaining misses on Zen4 are `zpotrfU` (0.663 at n=32),
`gbtrf` (0.671 at n=128) and `ztrsv` (0.865 at n=1024).

`trtrs` **used to sit here** at 0.753 and now gates at every size (worst 1.005); so do `zgemvT` and
`zgemvC`, which were published at 0.855 and 0.877 as part of a "complex BLAS-2 n=1024 cluster" that the
2026-08-09 re-measure dissolved — what remains of it is `zgeru` (0.906) and `ztrsv`, at one size each
rather than four ops at one size. Pivoted QR (`geqp3`) **used to sit here** at 0.84 geo / 0.51 worst, an
unblocked BLAS-2 implementation racing a blocked one; the blocked `dlaqps` port has since landed and it
now gates on all three boxes (1.11 / 1.02 / 1.03). The other three that used to sit here — Bunch–Kaufman
(`sytrf` was 0.15, `sytrs` 0.52) and banded LU (`gbtrf`, 0.48) — shared that diagnosis and were blocked
on 2026-07-30: `sytrf` 1.41 / 1.07, `sytrs` 1.69 / 1.46, `gbtrf` 1.48 / 0.98.

An earlier version of the coverage summary said BLAS 1/2/3 and the core factorizations were "perf-gated
`≥ OpenBLAS`". That was written against an OpenBLAS-only, geomean-flavoured reading; under the project's
actual rule — `max(OpenBLAS, AOCL)` at every size — it does not hold, which is why the tables carry
per-row numbers instead of a blanket claim.

## Known open items

Tracked in [`ROADMAP.md`](https://github.com/el-oso/PureBLAS.jl/blob/master/ROADMAP.md):

- **BLAS-3 conversion gap** (above): `gemm` beats AOCL by 25–31% at n=4096 while `syrk`/`syr2k` on the
  same engine sit at 0.94–0.97. Largest shared lever; reproduces on all three µarchs.
- `trsm` n=32 **is closed** on both available boxes as of 2026-08-10 (Zen4 1.056, Zen3 1.102); the Zen5
  figure predates the fixes (box offline). The residual moved to mid-`n` (Zen4 0.941 @ n=128), and the
  earlier "~0.95 codegen floor, latency-bound back-substitution, register-walled" reading is
  **superseded**: fitting `1/GF = α + β/KC` on the leaf puts the asymptotic microkernel rate at 44.4 GF —
  *above* PureBLAS's own dgemm — so the inner loop is not the limiter. The whole deficit is an 18.6%
  per-slab fixed cost, of which an opcode histogram attributes ~22% to scalar addressing and ~24% to
  vector moves, versus ~3.2% of total leaf time for the transposes and ~2.8% for the back-substitution.
- `gemvT` n=2048 is the one BLAS-2 cell that misses on **all three** µarchs (0.96 / 0.94 / 0.97) — a
  size-specific behaviour rather than three per-box stories, and the better-posed problem for it.
- `gemvN` Zen5 n=2048 (0.942) and `ger` Zen5 n=2048 (0.896).
- `trmv`/`trsv` Zen5 in the DRAM regime **now gate** (1.06 / 1.03 at n=4096) after the fuse-factor routing
  landed; the residual `trmv` cell moved to n=256.
- `hpmv` still per-column — port the spmv AP-residency panel to complex.
- `trsm`/`ztrsm` side-L remain the flagship AOCL gaps (4K power-of-two aliasing in the column-lane
  back-substitution); side-R (`trsmR`/`ztrsmR`) is now gate-measured and mostly clears — `ztrsmR`'s worst
  cells improved on both measured boxes on 2026-08-11 (sign placement in the register-tile leaf), leaving
  Zen3 n=128 at 0.926 and Zen4 n≥512 at ~0.99.
- Real `geqrf` panel width is now hardware-derived (register-count floor `256/NVREG`, grown with the
  matrix vs L2) — it gates AOCL fleet-wide (worst ≥ 0.96), closing the former n=48 dip.
- Complex LAPACK: `zgeqrf` worst-size still just under gate on Zen5; `zgesvd` blocked-bidiagonalization
  port pending (values-only, capped at n=1024).
- Tuning-constant debt: several block-size literals remain to be re-derived as formulas over detected
  cache/register parameters.
