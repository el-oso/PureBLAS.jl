# Performance notes

Per-routine analysis behind the numbers on [Performance](performance.md) and
[Coverage](coverage.md). Those pages are the source of truth for what the numbers are; this
page is about why they are what they are, and what is still open. How the measurements are
produced and how to read a cell: [Methodology](methodology.md).

The ratios quoted below come from the specific experiment being described, on the date
stated, and they are hand-written rather than generated. Where one disagrees with the
generated tables, the tables are right — they are rebuilt from the caches on every publish,
and this page is not.

## Real BLAS

**BLAS-1** is bandwidth-bound and at parity fleet-wide (worst sizes ≥ 0.98, except `iamax`
on AVX2 at 0.95). `nrm2` runs 7–10× because OpenBLAS uses the always-scaled LAPACK
algorithm, while PureBLAS scales only on overflow or underflow.

**BLAS-2** gates fleet-wide with two exceptions, both on Zen5: `gemvN` (mid-n native-512
residual, worst 0.90) and `trmv`/`trsv` at large n (DRAM regime, worst 0.98–0.99). `spmv`
is flat at 1.9–2.2 across the fleet from an AP-residency packed panel, and `ger` sits at
gate on all three boxes (worst 0.97–1.06) with a per-µarch-calibrated write-stream count.

**BLAS-3.** `gemm` gates at every size on all three boxes, with Strassen–Winograd running
1.2–1.4× at large n. The triangular and symmetric ops gate on AVX-512; on AVX2 the worst
size of `trmm` (0.84) is still open, while `trsm` gates against OpenBLAS at every size on
both available boxes (AVX2 worst 1.14).

### Complex `gemv`-T/C is an ISA split

Past L2 these route to a column-block whose width comes from a Measure-tier knob bounded by
the register file: `regs = (2·NC + 2)·(HALF ? 1 : 2)`. On AVX-512 that admits `NC = 8`
narrow accumulators — eight concurrent column streams, matching what AOCL's fused `zdotxf`
runs — and `zgemvT` gates at every size on Zen4 (n=512 0.96→1.00, n=1024 0.91→1.02 vs
AOCL). On AVX2 the same formula wants 18 vector registers against the 16 available, so Zen3
keeps `NC = 4` and its mid-band misses stand: `zgemvT` 0.95–0.98 and `zgemvC` 0.94–0.96 vs
AOCL at n=512…2048. That is a register-file limit rather than an unexplored lever. Closing
it needs a different accumulator shape, not a wider block.

The split below L2 is separate, and derived rather than tuned: while A is L2-resident the
loop is FMA-latency-bound and prefers fewer, wider accumulators (measured at n=256, 1.15
wide against 1.04 narrow), so the wide arm stays there on both ISAs.

### The BLAS-3 conversion gap

`gemm` is the fastest thing on all three boxes and pulls further ahead as `n` grows, yet the
routines built on that same engine fall behind AOCL at exactly the sizes where `gemm` wins
most:

| vs max(OpenBLAS, AOCL) | n=2048 | n=4096 |
|---|---|---|
| `gemm` — Zen4 / Zen3 / Zen5 | 1.150 / 1.150 / 1.168 | **1.252 / 1.280 / 1.308** |
| `syrk` | 0.955 / 0.963 / 0.978 | **0.937 / 0.968 / 0.953** |
| `syr2k` | 0.976 / 0.974 / 0.977 | 0.966 / 0.983 / 0.957 |

`syrk`, `syr2k`, `symm`, `trmm` and `trsm` all route their trailing updates through the same
`_gemm_core!` / `_microkernel_db!` engine. That engine is 25–31% faster than AOCL at n=4096
on this silicon, so the roughly 30-point spread between it and the routines built on it is a
conversion loss rather than a hardware limit. It reproduces on three microarchitectures,
which makes it structural rather than a per-box tuning artifact, and it is the largest
shared lever currently on the board.

`trsm`'s residual sits at mid-`n` (Zen4 0.941 at n=128), and it is decomposed rather than
guessed. Fitting the leaf's rate as `1/GF = α + β/KC` puts the asymptotic microkernel rate
at 44.4 GF, above PureBLAS's own dgemm at 43.75, so the inner loop is not the problem. What
is left is an 18.6% per-slab fixed cost at KC=128, and an opcode histogram of the emitted
slab prices it: scalar addressing about 22% and vector moves about 24%, against only ~3.2%
of total leaf time for the two 8×8 transposes and ~2.8% for the back-substitution
arithmetic. Removing the largest addressing population — recomputed `(jc + c)·ldb` products
— was worth 3.0% at n=128, bit-identically. Zen5's `trsm` column still reads 0.91 because
that box has not been re-swept since the tiny-`n` fixes landed.

`syrk`/`syr2k` at n=4096 is the cross-µarch invariant above. By contrast `symm` ≥ 256 and
`trsmR` ≥ 2048 miss on Zen4 and Zen3 but gate on Zen5, so those are µarch-specific.

`trmm`'s crossover between recursion-over-`gemm!` and the packed routine is derived as
`5·_GEMM_UNPACK_MAX ÷ 2` rather than borrowed from gemm's own unpacked/blocked cut. In a
controlled same-process A/B the two sizes whose route changes gained 4.0% and 5.1% (n=512
and n=1024), which moved the worst cell to n=2048 — whose miss lies in the packed path
itself and is a separate defect.

## Real LAPACK

`potrf`/`geqrf`/`getrf`/`gesvd` gate on all three boxes, with geomeans of 1.25–1.52. The
small-n `potrf` work — block-small Cholesky plus a fused 12-accumulator `trsm`-R that does
the downdate and triangular solve in one register pass, in both the small-n and NB=128 panel
drivers — brings AVX2 `potrf` to BLASFEO parity, the MKL proxy: 0.91–1.04× its column-major
`dpotrf` at n≤224, 0.87–0.91× at n≥256, and 1.5–2.2× against OpenBLAS fleet-wide.

### Cholesky, upper triangle

`potrf` gates for both triangles. The upper path is a different route rather than a mirror
image — transpose into a padded scratch, factor with the gating lower kernels, transpose
back — and its large-n miss came from the same power-of-two-`lda` cache-set aliasing the
factorizations already defend against, this time inside the copy. The transpose read `A`
strided across a row at its raw `lda`, and at `lda·sizeof(T) % 4096 == 0`, precisely the
benchmarked sizes, every column maps into one L1 set group. Reordering the loops so the
contiguous side is the read and the strided side is the non-power-of-two scratch is
bit-identical and worth 2.1–3.8× on the copy, taking real `potrfU` from 0.96 to 1.03 vs AOCL
on Zen4 and 0.98 to 1.04 on Zen3.

The upper path now clears both references at every size on Zen4 (worst 1.11 vs OpenBLAS,
1.03 vs AOCL) and on Zen3 (worst 1.06). The single residual is Zen5 n=2048 at 0.970, a 3.0%
gap against a within-run spread of 0.002 — small, but well clear of its noise, so it is a
real cell rather than a rounding artifact.

Complex `zpotrfU` still sits at 0.95 for `n ≥ 1024` on Zen4, where it gates on Zen3. The two
transposes are the entire remaining gap — remove them and PureBLAS would be 3.4% ahead of
AOCL — but they cannot simply be deleted, because factoring the upper triangle natively
measures slower than the transpose route at every size and type.

At `n = 32` the same route was losing for a second, independent aliasing reason, and it is
worth keeping because the predicate involved is easy to write again. The base kernel stages
its block through a contiguous scratch when `A isa SubArray && stride(A,2) > n`, which tests
for the *existence* of a stride where the staging is an *aliasing* remedy — worth 2× at
`lda` = 512 or 1024, and a 1.6× cost at `lda` = 520. The upper route hands it a view of a
padded scratch whose leading dimension is deliberately kept off the way-stride, so the copy
round-trip could never pay there. Replacing the predicate with the byte-scaled way-stride
test the padding helper already derives, adding no new tuning constant, took `zpotrfU@32` to
0.993 on Zen4 and 1.145 on Zen3, with n=128 and n=256 picking up 7.5% and 5.0% from the same
change.

### Triangular solves and the narrow-B `trsm`

The gating band on the `getrs`/`potrs`/`trtrs` rows is for the factorizations. Their solves
are not yet gated: they pass OpenBLAS but miss AOCL — `getrs` 0.93 geomean / 0.85 worst,
`potrsL` 0.94 / 0.78, `potrsU` 1.13 / 0.91, `trtrs` 1.07 / 0.88. `trsm` itself passes
OpenBLAS (worst 1.18) while still missing AOCL (worst 0.90).

Getting there meant fixing three defects in `trsm`, all of which showed up only once
`A \ b` with a single right-hand side — the most common solve there is — was measured at
all. Each is worth knowing because the shapes recur:

1. The blocked setup is never repaid for narrow B. It costs `O(k·nb²)` for a triangular
   inverse of the diagonal blocks, amortised over B's columns; at k=1024 that was 1123 µs at
   `nrhs=1` against 1197 µs at `nrhs=8`, eight solves for the price of one. Narrow B now
   sweeps `trsv` per column, which beat OpenBLAS's own `trsm` by 2.9× on that shape.
2. A po2-`lda` A-pad that never paid. When `stride(A,2)` was a power of two, all of A was
   copied into an odd-`ld` scratch to dodge cache-set aliasing — `O(k²)` work, charged even
   to a single-column B. Measured across all sixteen combinations of side × uplo × transA ×
   B-width it cost between 3% and 357% and won nowhere. Deleted.
3. A ragged SIMD lane costs a full lane. The kernel processes B's columns in `W`-wide lanes,
   so a partial lane is charged as a whole one: at k=512, `nrhs` = 1…7 cost
   96/180/278/381/464/543/675 µs, about 95 µs per column, while `nrhs=8` cost 94 µs in
   total. Widening a ragged B into scratch before the solve collapses that, 2.9–6.8× at
   k=512.

A residual remains at `nrhs = 8`, which falls between the two mechanisms — too wide for the
per-column sweep, and on Zen3 (`W=4`) already two full lanes, so the ragged-tail fix does
not reach it either.

The threshold guarding the sweep is worth recording as a tuning lesson. It was first written
as `nrhs ≤ k/(4·_TRSM_DBASE)`, reasoning that a larger `k` amortises the setup over more
columns. That was the wrong shape: blocked costs `setup + nrhs·b`, the sweep costs `nrhs·v`,
and all three terms scale as `k²`, so the crossover is `k`-invariant. The growing rule
selected the sweep exactly where it loses, and was itself the cause of the `nrhs=8` misses it
was meant to fix.

### Tridiagonal

`gtsv`, `gttrf`, `gttrs`, `pttrf`, `pttrs` and `ptsv` gate against `max(OpenBLAS, AOCL)` on
all three µarchs, every result bit-identical to the reference. Worst case vs AOCL: `gtsv`
1.20–1.22, `gttrf` 1.51–1.54, `gttrs` 1.03–1.06, `pttrf` 1.11–1.12, `ptsv` 1.04–1.05. Against
OpenBLAS the same ops run 1.31–1.57, and `pttrs` 1.67–1.73.

These are `O(n)` serial three-term recurrences, so the limit is a divide→multiply→subtract
latency chain (about 19.5 cyc/elem on Zen4) rather than flops, and every lever was about how
data moves around that chain rather than about the arithmetic. Three cost roughly 10
cyc/elem each: a live `fadd x, 0.0` left on the real path of a complex-shaped expression
(LLVM may only fold it under `nsz`, since `x + 0.0` is not an identity for `x = -0.0`);
re-reading a recurrence variable the previous iteration had just stored; and a per-element
inner loop with a runtime trip count. A fourth, an extra store stream, cost 10.1 and was
hoisted into a bulk pass priced at 0.3. Unusually for this project these fixes transfer
across µarchs — they are properties of the generated code, not cache- or ISA-dependent
tuning.

`pttrs` is the one cell at parity rather than above it, 0.99 vs AOCL. Both libraries run at
about 12.2 cyc/elem, which is exactly the analytic bound: the forward sweep's recurrence is
multiply+subtract at roughly 6 cyc, and the backward sweep's divide sits off the chain, since
`B[i]/D[i]` does not depend on `B[i+1]`. That is a shared dependency-chain bound rather than
a gap, with 0.0% run-to-run drift. Register-carrying the recurrence and hoisting the `uplo`
test out of both loops were each measured and are neutral, because LLVM already does both.

### Pivoted QR (`geqp3`)

Blocked, and gating on all three boxes (1.11 Zen3 / 1.02 Zen4 / 1.03 Zen5). The first
blocked `dlaqps` port was verified correct but not faster, and larger panels were worse —
which was the diagnostic that the cost is the panel's own per-column BLAS-2 calls rather than
the trailing gemm. The shipped version (`src/lapack/laqps.jl`, wired from `geqp3.jl`) reaches
the kernels pointer-direct, adds a SIMD downdate in the unblocked remainder, and routes the
`laqps` gemv-T through the blocked kernel. See §5 of [Tuning](tuning.md).

### Bunch–Kaufman (`sytrf`/`hetrf`)

All four variants are blocked: real-symmetric and complex-symmetric through a `dlasyf` port,
Hermitian through a separate `zlahef` port, both `uplo`. `zlahef` is deliberately not
`dlasyf` plus a few conjugations — it realifies the diagonal at points that have no symmetric
counterpart, and its 2×2 inverse is asymmetric in the conjugates, with the asymmetry flipping
between triangles.

The row misses one cell: against AOCL on Zen4 the gate sweep reads 1.40 geomean / 0.99 worst,
from n=2048 measuring 0.98–1.02 across rounds, which is parity within noise rather than a
structural gap. Every other size clears, and on Zen3 both references clear outright (1.34 /
1.04 vs OpenBLAS, 1.49 / 1.17 vs AOCL).

The panel width is `8·⌈√n/8⌉` clamped to [16,96] for real — a formula over the problem size,
validated on two microarchitectures — times a measured multiplier for complex, because the
complex optimum is microarchitecture-dependent (Zen4 selects 1, Zen3 selects 3) where the
real one is not. That split was made by measurement: the real formula gates on both boxes,
but applied to complex on Zen3 it produced three gate misses at 0.95–0.97.

### Banded LU (`gbtrf`)

Blocked, as a port of `dgbtrf`, with the two corner blocks (A13/A31, whose parts fall outside
the stored `ldab` window) staged dense the way the reference does. Zen4 vs OpenBLAS at
`kl = ku = n/8` reads 1.49 geomean / 0.93 worst, and 1.19 / 0.72 vs AOCL; Zen3 clears
OpenBLAS outright at 1.41 / 1.16.

The single miss is n=128, i.e. `kl`=16, and it is not a blocking question. OpenBLAS is also
unblocked at that width — its ILAENV `nb`=64 exceeds `kl`, so `dgbtrf` takes the `dgbtf2`
branch — so we are losing to its *unblocked* band downdate. The remaining lever is the
quality of `_gbtf2!`'s rank-1 band update against `dger`, not the panel width; no `nb`
rescues that cell past about 0.93, measured across `nb` = 2…16.

There are two knobs here, and the second exists because of a mistake worth recording. The
panel width is a formula over `kl` (`8·(1 + kl÷128)`), fitted to the measured optimum at
fourteen band widths on Zen4. Shipping only that regressed Zen3 from pass to fail (1.43 /
1.18 → 1.32 / 0.91): at `kl`=16 blocking wins on Zen4 (0.93 against 0.796 unblocked) but
loses on Zen3 (0.91 against ~1.18), because at that width OpenBLAS is unblocked too, so the
cell is a scalar-downdate race rather than a blocking race. The blocked-vs-unblocked floor is
therefore a measured knob — Zen4 selects 16, Zen3 selects 24 — which restored Zen3 with Zen4
unchanged. That is req#8(b)'s "derive → validate on the fleet → ship" catching a one-box
derivation.

### Pivoted Cholesky (`pstrf`)

`pstrf` clears AOCL completely on both Zen4 and Zen5, both triangles, every size (Zen5 worst
1.20 for `uplo='L'`, 1.19 for `uplo='U'`). Against OpenBLAS it clears everywhere except a
narrow window that is consistent across µarchs and mild: Zen4 n=48 at 0.943 and n=64 at
0.921. On Zen5 the window sits at small n instead — `pstrfU` n=32 at 0.905 is the one cell
clear of its own noise (spread 0.022), while `pstrf` n=8 at 0.876 and `pstrfU` n=8 at 0.930
both fall within their round-to-round spread (0.289 and 0.142), too noisy to adjudicate and
not counted as misses.

The large-n wins come from batching the pivot row swaps per panel. They are stride-`lda` and
were about 47% of the runtime, so the swap rather than the BLAS-3 update was the bottleneck.
⚠ Float64 only; Zen3 not yet measured for this routine.

One caveat on reading the published table: the Zen4 gate sweep reports `pstrf` worst = 0.85
vs OpenBLAS, which is not one of the cells above. It is n=8, where the per-round ratios come
out bimodal (1.03 / 1.00 / 0.79 / 1.01 / 1.01 / 0.79 / 0.80 / 1.03). Rounds alternating like
that at sub-microsecond scale is the ABBA ordering effect, not a stable regression, and the
sweep's size grid (8, 32, 128, …) skips 48 and 64 entirely, so it never samples the real
residual.

### Packed Cholesky (`pptrf`)

Neither triangle clears both references, and they fail against different ones. `uplo='L'`
clears OpenBLAS on the Zen4 gate sweep (geomean 1.11, worst 1.04) but misses AOCL (1.17 /
0.93). `uplo='U'` is the mirror image: 3.30 / 1.43 vs AOCL, but 1.07 / 0.92 vs OpenBLAS.

The wide margin against AOCL on `uplo='U'` is not evidence of quality. Timing the same
reference routine under each library, AOCL's `dpptrf` `uplo='U'` is 6.0× slower than
OpenBLAS's at n=1024 (175 281 µs against 29 297 µs) — near-stock netlib — while on `dpotrf`
the two sit within 20% of each other. AOCL optimizes dense Cholesky and does not optimize the
packed variant. This does not move the gate, which is `max(OpenBLAS, AOCL)`: a slow AOCL
never lowers the bar, it just means `max()` here equals OpenBLAS.

The lower path improved once its gap was decomposed. The `spr!` kernel already ties or beats
AOCL at every order, and the whole deficit was per-call overhead — the public entry plus a
`SubArray` built per column — compounded by the lower path reusing the upper path's cutoff
constant, which meant `spr!` was never called at all at n=32 (worst cell 0.738 → 0.949). See
§5.3 of [Tuning](tuning.md). The remaining cells are gaps, not ceilings.

### Banded Cholesky (`pbtrf`)

`pbtrf` gates fully on Zen4 (Float64, `kd` = 32…384, n ∈ {1024, 4096}, both triangles:
1.02–3.28× AOCL, 1.49–2.00× OpenBLAS) and beats OpenBLAS on all three µarchs, but does not
yet clear AOCL fleet-wide. The residuals are all `uplo='U'` at mid bandwidth: Zen5 `kd`=128
and 256 at 0.989/0.969, and Zen3 `kd`=128/160/192 at 0.881–0.984, plus `uplo='L'` `kd`=160
and 192 at 0.959–0.995. Zen3 is the worst box and also the one whose `_pbtrf_ucross` bracket
is derived from a narrower `_vwidth` (W=4, so the switch lands on 192, itself a 0.90 cell),
so the leading hypothesis is a candidate bracket putting the optimum at an edge — the same
failure this project's history records for the wide-band panel bracket. Not yet confirmed by
measurement. Correctness is clean on all three boxes (11368/11369, identical).

`uplo='U'` dispatches between two kernels at a measured bandwidth crossover: a
conj-transpose re-pack onto the much faster lower kernel for narrow bands, and a native
upper-storage port once the re-pack's diagonal walk starts to dominate. The panel width
likewise has two measured regimes — clamping the wide-band width to `kd` collapses the
in-band panel, and was the routine's only Zen4 gate miss. See §5.2 and §5.3 of
[Tuning](tuning.md).

### Symmetric / Hermitian eigensolver

With eigenvectors (`jobz='V'`) it clears both references on the Zen4 sweep: 1.32 / 1.25 vs
OpenBLAS, 1.36 / 1.19 vs AOCL. Values-only (`jobz='N'`) does not — 1.13 / 0.98 vs OpenBLAS,
though 1.31 / 1.14 vs AOCL — so the coverage row is below 1.0 on the strict worst-case rule,
driven by a single size.

Three Level-3 pieces get the vectors path there: the two-sided reduction (`sytrd`/`hetrd`)
is blocked `dlatrd` + `syr2k`/`her2k`; the eigenvector back-transform (`ormtr`/`unmtr`) uses
compact-WY block reflectors; and the divide-and-conquer solver (`stedc`) assembles
eigenvectors once, with no per-merge column copies, and combines survivors with the `dlaed3`
COLTYP two half-height gemms — exploiting the block-diagonal zero structure for roughly half
the combine flops, matching libFLAME.

## Complex

The complex surface is SIMD across all levels, in portable SIMD.jl kernels with no x86
intrinsics; the generic scalar path remains for AD.

**CL1** is at parity or better fleet-wide. `dznrm2` is 7–9× (the same scaled-accumulation
story as real `nrm2`) and `izamax` 1.2–1.8×. Residuals: `zaxpy` at the L3 edge (0.94–0.96)
and `zdotc`/`zdotu` on Zen5 at DRAM sizes (0.89–0.94, bandwidth).

**CL2** gates broadly on all three boxes. Remaining residuals cluster at n=1024 on the
AVX-512 boxes: `ztrsv` (0.948 Zen4, 0.936 Zen5) and `ztrmv` (0.948 Zen4), a coupled
upper/lower-triangular tradeoff, alongside the complex gemv trio `zgemvN`/`zgemvT`/`zgemvC`
(0.848–0.877 Zen4, 0.935–0.951 Zen5). `zgeru` dips at mid/large n (0.817 Zen4 at n=1024,
0.857 Zen5 at n=4096), and `zhemv` misses only on Zen3 at large n (0.919 at n=2048, near the
DRAM roofline) while gating 1.9–2.5× on both AVX-512 boxes.

**CL3.** `zgemm` beats OpenBLAS fleet-wide (geomean 1.26–1.40, Karatsuba 3M at mid/large n),
and the rank-k ops gate within a few percent.

### `ztrsmR` — codegen, not blocking

Worth keeping because the mechanism is invisible at the source level and easy to reintroduce.
The register-tile leaf wrote its update as `muladd(V2(-cr), xv, …)`, putting the negation on
the *scalar* coefficient between its load and its splat. That blocks both the `vfnmadd`
opcode and AVX-512's embedded broadcast `(mem){1to8}`, so LLVM emitted `vmovsd` + `vxorpd` +
`vbroadcastsd` per coefficient — 96 instructions of coefficient preparation against 64 useful
FMAs. Negating the B vector once per step instead is exact, since `nswap(-v) == -nswap(v)`
means the swapped operand needs no separate handling, and it took the loop from 2.95 to 1.96
instructions per useful FMA with residuals identical to the last digit.

The cell was attributed before it was patched: it fits `1/GF = α + β/n` with an 83.5 GF
asymptote, predicting n=192 to 0.2%, so it was a 14% per-call fixed cost rather than a
size-specific conflict. The same rewrite applied to the complex side-L slab compiled
byte-identical, because LLVM had already chosen `vfnmadd` there on its own. Remaining: Zen3
n=128 at 0.926 and Zen4 n≥512 at about 0.99.

### Complex `trmm` small-n

Open cells here are smaller than the raw ratios suggest, and `bench/gate_gaps.jl` reports
each one's shortfall against its own round-to-round spread:

- `ztrmmR` n=32 on Zen3 — 0.839, a 16.3% gap against a 0.047 spread. Real.
- `ztrmm` n=128 on Zen3 vs AOCL — 0.917 at a spread of 0.003, the most tightly measured miss
  in the whole table. Real, and on the packed path.
- `ztrmm` n=32 on Zen3 — a 9.5% shortfall against a 0.115 spread. That cell is within spread:
  it needs more rounds, not engineering, and optimising against it would be optimising
  against measurement error.

Also open, and outside their spreads: `ztrsm` 0.89 at n=128, and a `zsymm`/`zhemm`/rank-2k
small-n dip from materialize-plus-microkernel overhead on the small-n complex-L3 path.

That residual is decomposed rather than guessed at. Holding `k` at 32 and varying only the
number of columns of `B` separates per-call from per-column cost through the public entry
point, by fitting `1/GF = a·(1/c) + b`. At the gate shape `ztrmm` carries 15.9% fixed
per-call cost, while `zgemm` at the same `k` — same kernel family, no triangle — carries 0.7%
and holds its rate flat to within 0.8% across its whole sweep. Shared BLAS-3 entry cost is
therefore negligible, and essentially all of `trmm`'s fixed cost is the O(k²) triangle
materialize. A direct-read fused-triangle base closed these dips in development but its
runtime `Val` args broke trim-safety and it was reverted; the trim-safe refix, using
compile-time `Val` dispatch, is the follow-up, and this puts its value at about 15% against
the 2.9–14.8% the individual cells need.

A second, unrelated inefficiency shows in the same fit: with the triangle amortized to
nothing, `trmm` still costs 0.1535 µs per column against `gemm`'s 0.2037 for half the
multiply-accumulates, so 66% efficiency. That is the granularity of the K-TRIM tiling in
steady state, not a fixed cost, and the triangle base would not address it.

### The 3M routing constant

Complex BLAS-3 lost to AOCL almost across the board until one routing constant was removed,
and it is worth recording because the mistake is a general one.

`zsyrk`/`zherk`/`zsyr2k`/`zher2k` read 0.94–0.99 against AOCL on Zen4 while the same source
read 1.19–1.31 on Zen3. That inversion was not a microarchitecture property: the source
routed a different *algorithm* per SIMD width. The Karatsuba-3M route — three real products
on the split real/imaginary parts, 25% fewer flops than the direct four-FMA complex kernel —
was enabled only for `_vwidth == 4`, so Zen3 got the flop cut and Zen4 did not. The default's
own comment called AVX-512 "untested" rather than falsified.

Nothing physical made it width-dependent. The 25% reduction is algebraic, and 3M's
split/combine overhead is O(n²) against O(n²·k) of product, which the existing size window
already bounds, so the test was an artefact of only ever having measured AVX2. Deleting it
took `zgemm` from 0.952 to 1.213 at n=128 and 1.000 to 1.255 at n=2048, and converted every
complex rank-k cell at n ≥ 256. The secondary effect was larger than the primary one:
everything layered on complex gemm moved without being touched — `zhemm` 1.004 → 1.136,
`zsymm` 0.994 → 1.146, `zherk` 0.982 → 1.053, `ztrsmR` 0.987 → 1.084. Zen3, which already ran
3M, was flat across the same sweep, which is the control saying the change did one thing
only.

The rank-k window's lower edge turned into a lesson about knobs rather than kernels. The
crossover carries a ratio of *our own two microkernels'* rates — the complex kernel is
latency-bound on AVX2 and throughput-bound on AVX-512 — so no cache or ISA constant predicts
it, every throughput model gives the wrong sign, and the measured crossovers were far apart
(about 80 on Zen4, about 160 on Zen3), so no single value existed. Rather than locate a
threshold that could not exist, the overhead was removed: the third Karatsuba panel is now
derived from the packed panels, since the complex pack already deinterleaves into two real
panels and the third is one add in a loop that exists, and the combine happens per micro-tile
into a small scratch instead of via n×n buffers.

That did not eliminate the threshold, and the reason matters. The pack itself is O(n·k), and
3M packs three panels where the direct kernel packs two; three real products structurally
require three real panels. What fusion bought was moving the crossover (about 160 → 112 on
Zen3, below 64 on Zen4) until one value finally served both machines. At n=128 fused-vs-packed
measured 0.886/0.887 on Zen4 and 0.956/0.963 on Zen3 — both win — so 128 ships, labelled in
the source as the empirical bound it is rather than dressed up as a derivation.

What remains is n = 32 and below, outside the window on every machine: `zsyr2k` 0.910 and
`zher2k` 0.906 on Zen4, `zgemm` 0.902, and `ztrmm` 0.972. Complex rank-2k also has a second,
independent fence on AVX2, which is why Zen3's `zsyr2k@128` did not move; that path needs the
rank-2k reformulation — one product plus a symmetrized add — rather than a window change.

### Complex `symm`/`hemm`

On Zen4 `zhemm` gates at every size from 32 up (1.004–1.012) and `zsymm`'s three remaining
shortfalls are each smaller than that cell's own round-to-round spread; on Zen3 both gate
across 128–2048 at 1.18–1.28. Complex symm used to materialize a dense n×n copy of the
symmetric operand and call gemm, while complex hemm avoided the copy but ran through a fork
of the complex gemm driver that had drifted, hardwiring off the two specializations —
overwrite-C when β=0, and the α=1 store — that the driver itself dispatches, which is
precisely the case the benchmark measures. Wiring those back in and routing complex symm
through the same packing path removed the copy entirely, taking wrapper cost against a bare
gemm of the same shape from 1.065/1.069/1.053/1.026 at n=128/256/512/1024 to
1.026/1.017/1.004/1.002.

What is left is tiny-n — n=8 and n=32, below the packing cut, still on the copy path — and
real `symm` at n≥256, whose packed path has not yet had the same β=0 fold applied.

### Complex LAPACK

`zpotrf` and `zgetrf` gate fleet-wide. `zgetrf`'s only residual is Zen3 n=256 at 0.94,
localized to PB's complex gemm at the trailing 208×208×48 shape — a gemm-kernel gap rather
than a getrf-structure one.

`zgeqrf` gates at every size fleet-wide (1.07–1.83). The complex trailing update mirrors the
real path (`herk` for VᴴV at half the flops, plus `trmm` for TᴴW), the reflector norm moved
from per-element `hypot` to SIMD `_nrm2`, the unblocked rank-2 panel runs while the matrix is
L3-resident, and the AVX2 trailing update is chunked to bound the Karatsuba-3M scratch below
½L3. It was validated on square inputs; tall/skinny QR is a separate benchmark shape.

`zgesvd` — singular values only, against `zgesdd('N')` — gates at every size for real and
ComplexF64. The large-n collapse was fixed by a blocked complex bidiagonalization
(`zlabrd`/`zgebrd` port), and the small-n floor by switching the values path to dqds
(`dlasq1–6`): OpenBLAS's `gesdd('N')` uses the sqrt-free dqds recurrence rather than
Golub–Kahan QR, and matching it lifted n=32 from about 0.86 to about 1.07 for real and
complex alike, since the bidiagonal is real. ComplexF32 small-n (n=32/48 ≈ 0.9) is the
remaining sub-gate, a `cgebrd` limit.

## Per microarchitecture

Which routines carry residuals differs per box — that is why the gate is per-machine rather
than pooled. Current ratios are in the generated tables; what follows is which families to
look at.

**Zen4 (AVX-512, double-pumped).** The tuning target. Real BLAS-1/2 gate essentially
everywhere, and the residuals are the BLAS-3 set above: `trsm` at mid-n, `syrk` at n=4096,
`trmm` and `symm`. The complex residuals are the shared LAPACK gaps plus a handful of BLAS-2
cells — `zgeru`, `zgemvN` and `ztrsv`, all against AOCL rather than OpenBLAS.

**Zen5 (AVX-512, native 512-bit).** Clears every AVX2 ceiling but shows a disjoint residual
profile. Open: `gemvN`, which is the deepest BLAS-2 miss anywhere on the fleet, `ger`, and
the complex pair `zgemvN`/`zgeru`, all at large n.

**Zen3 (AVX2).** The hardest target: 16 ymm registers against AVX-512's 32 zmm. The real
surface gates except the `trmm`/`trsm` worst sizes; complex carries the widest residual set
(`zdot`, `ztrmm` both sides, `ztrsm`, `zgetrf`). The `potrf` small-n work reaches BLASFEO
column-major parity at n≤224 (0.91–1.04×), and 1.5–2.2× vs OpenBLAS.

### Zen5's partial complex-L3 transfer

When the 3M change landed, all eleven of Zen5's complex-L3 cells rose and five crossed the
gate, but they split cleanly in two. The `symm`/`hemm`/`trsmR` family transferred in full
(`zsymm` 0.954 → 1.170 against Zen4's 0.994 → 1.146; `zhemm` 0.979 → 1.153; `ztrsmR` 0.957 →
1.047), while `zgemm` and `zsyrk` transferred only partially — 0.947 → 0.996 and 0.949 →
0.969, against Zen4's 0.952 → 1.213 and 0.958 → 1.133.

That split is a genuine microarchitectural signal, and its cause is currently unknown. It is
*not* an FMA-throughput difference, which was the obvious guess and has been measured and
ruled out. Taking each box's classical-gemm OpenBLAS arm as an honest throughput probe — the
PureBLAS arm cannot serve here, since Strassen–Winograd makes `2n³/t` overstate the hardware
FMA rate — all three boxes sustain the same per-cycle width: 6.84, 7.48 and 6.89 F64 FMA
lanes per cycle on Zen3, Zen4 and Zen5, an 8-lane ceiling everywhere. Zen5 is therefore not
less FMA-bound than Zen4; it has the same lanes per cycle at a lower clock, giving it the
fleet's lowest F64 peak (31.7 GF against Zen4's 44.9 and Zen3's 58.9 — the narrowest-ISA box
has the highest peak because it clocks highest). Untested candidates for the real cause:
Zen5's 48 KiB L1d, unique in the fleet, and the lower clock shifting the memory-versus-compute
balance.

## vs AOCL — the residuals

PureBLAS matches or beats AMD's own library on gemm and every LAPACK factorization, real and
complex, by 5–80%. Verified PB/AOCL at mid/large n (>1 means PureBLAS is faster; Zen4,
single-thread), with PB/OpenBLAS alongside for context:

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

The one residual against AOCL is side-L `trsm` at mid-n, worst size 0.92–0.97. It is a
codegen-scheduling gap on the fused-leaf kernel — the fused back-substitution's latency chain
runs at about 2 IPC where BLIS's hand-scheduled assembly hides it — rather than a structural
lever: building AOCL's own packed-triangle structure in portable SIMD.jl lands at the same
~0.95.

Side-R `trsmR`, the potrf/getrf panel shape, routes both `transA` through the fused leaf via
the reflection identity Ã = J·Aᵀ·J. Its AVX2 worst cell is now 0.917, and the mechanism is
worth keeping: at `transA='T'` the reflection branch does not fire, so A reaches the leaf
verbatim at the caller's `lda`; for a square operand that is a power of two, and at k=128 the
1024-byte column stride is a quarter L1 way period. Padding A into an odd-`ld` scratch
recovers 11.5–15.1% at the leaf. The pad is gated on whether the leaf actually re-reads A —
that is, whether it spills, true on AVX2's 16 ymm and false on AVX-512's 32 zmm — because on
a non-spilling box the same pad measured null at n≥512 and 4.9% slower at n=128. Zen4
therefore never pads; its worst size remains about 0.95 at n=2048.

## Where the misses are

36 of the 83 measured rows clear `≥ max(OpenBLAS, AOCL)` at every size. The rest miss
somewhere, usually narrowly — 0.9–0.99 at one or two sizes, often the smallest, where
per-call overhead dominates and the measurement is least stable.

Which cells those are is not listed here on purpose. The per-routine worst cells live in
[`bench/gen_table.md`](https://github.com/el-oso/PureBLAS.jl/blob/master/bench/gen_table.md)
and
[`bench/gen_table_aocl.md`](https://github.com/el-oso/PureBLAS.jl/blob/master/bench/gen_table_aocl.md),
both generated from the caches, and `bench/gate_gaps.jl` prints each shortfall against that
cell's own round-to-round spread — which is the number that says whether a cell is worth
engineering or just needs more rounds. A hand-copied list here would go stale against those
tables, and has.

## Known open items

Tracked in [`ROADMAP.md`](https://github.com/el-oso/PureBLAS.jl/blob/master/ROADMAP.md):

- **BLAS-3 conversion gap** (above): `gemm` beats AOCL by 25–31% at n=4096 while
  `syrk`/`syr2k` on the same engine sit at 0.94–0.97. Largest shared lever, and it reproduces
  on all three µarchs.
- `trsm` mid-`n` (Zen4 0.941 at n=128). The whole deficit is an 18.6% per-slab fixed cost, of
  which an opcode histogram attributes about 22% to scalar addressing and 24% to vector moves,
  against ~3.2% of total leaf time for the transposes and ~2.8% for the back-substitution.
  The Zen5 figure predates the tiny-`n` fixes, since that box has not been re-swept.
- `gemvT` misses against AOCL on all three µarchs at its worst size, which makes it a
  size-specific behaviour rather than three per-box stories, and the better-posed problem
  for it.
- `gemvN` on Zen5 is the deepest BLAS-2 miss on the fleet, against both references, with
  `ger` on the same box behind it.
- The `trmv` residual on Zen5 has moved to n=256; the DRAM-regime cells now gate after the
  fuse-factor routing landed.
- `hpmv` is still per-column — port the spmv AP-residency panel to complex.
- `trsm`/`ztrsm` side-L remain the flagship AOCL gaps (4K power-of-two aliasing in the
  column-lane back-substitution). Side-R mostly clears, leaving `ztrsmR` at Zen3 n=128 (0.926)
  and Zen4 n≥512 (about 0.99).
- Complex LAPACK: `zgeqrf` worst size is still just under gate on Zen5, and the `zgesvd`
  blocked-bidiagonalization port is pending for values-only, capped at n=1024.
- Tuning-constant debt: several block-size literals remain to be re-derived as formulas over
  detected cache and register parameters.
