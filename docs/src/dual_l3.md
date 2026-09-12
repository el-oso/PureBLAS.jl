# Dual numbers: BLAS-3 design (Phase 1 — for sparring, nothing implemented)

Status: **design only, 2026-09-12.** Companion to `dual.md` and `dual_l2.md`. Citations are into `a0f0a703`.

## 0. Triage: what the generic scalar path achieves today

galen (Zen3, core 6, boost off), plots.jl's CL3 regime (one call per sample on fresh operands), Chairmarks
median of 6 rounds. Cell = complex-twin time ÷ dual time on identical bytes; the complex twin's absolute time
is given so the route it took is visible. Probe: `bench/probes/dual_l23_triage.jl` (gitignored).

| op (dual / complex twin) | n=32 | n=128 | n=512 | n=1024 |
|---|---|---|---|---|
| gemm / zgemm (α=β default) | 0.17 (0.01 ms) | 0.13 (0.27 ms) | 0.12 (15.9 ms) | 0.12 (124 ms) |
| gemm / zgemm (dual α,β vs complex α,β) | 0.20 | 0.14 | 0.12 | 0.12 |
| syrk / zsyrk | 0.12 | 0.11 | 0.10 (9.0 ms) | 0.10 (69 ms) |
| trmm / ztrmm (side L) | 0.28 | 0.24 | 0.12 (9.5 ms) | 0.11 (69 ms) |
| trsm / ztrsm (side L) | 0.19 | 0.19 | 0.12 (9.4 ms) | 0.11 (70 ms) |

The generic loop (`_gemm_generic!`, `gemm.jl:1385`) is 8–10× slower than the complex twin at every size
≥ 128 and 4–6× at n=32. zgemm at n=1024 runs 124 ms = 69 GFlops on Zen3 (the 3M route, `gemm.jl:2947-2951`,
window `48 ≤ max(m,n,k) ≤ 2048`, `min ≥ 16`). **Every L3 op is worth a kernel**; gemm first.

## 1. The shape of the level: ride the REAL kernels through planar splits

For a dual product `(A_v + A_p ε)(B_v + B_p ε)`:

```
C_v = A_v·B_v                 C_p = A_v·B_p + A_p·B_v          (A_p·B_p is ε² and is never formed)
```

Three real products of the value/partial **planes**. This is exactly the structure PureBLAS already has for
complex "3M" (`_gemm_3m!` `gemm.jl:2649`): split each operand into planes with one O(n²) pass
(`_split3!` `:2549`), run three real gemms through `_gemm_real_dims!` (`:2629`, the real
unpacked/blocked dispatch on `PtrMatrix` views of a persistent pool, `_gemm_3m_scratch` `workspace.jl:164`),
and recombine (`_combine3!` `:2595`). The differences for dual are all simplifications:

| | complex Karatsuba-3M | dual |
|---|---|---|
| split outputs per operand | 3 planes (re, im, re+im) | **2** (value, partial) |
| real products | 3 (`Ar·Br`, `Ai·Bi`, `(Ar+Ai)(Br+Bi)`) | **3** (`Av·Bv`, `Av·Bp`, `Ap·Bv`) |
| combine | `Cr = P1−P2`, `Ci = P3−P1−P2` — **cancellation**, the reason 3M is windowed | `Cv = P1`, `Cp = P2 + P3` — a plain sum, no cancellation |
| scratch | 9 buffers | 6 (or 5, accumulating `P3` into `P2` with β=1) |

So dual gemm does the same three real products as the complex twin's 3M route, with less O(n²) traffic and
no numerical reason for a window. Op count: complex gemm is 4 real products (non-3M) or 3 (3M); dual is 3.
Where the twin runs 3M the honest bar is **equal time**; outside the 3M window (n < 48, n > 2048, thin k) the
dual has fewer products than complex and should be **faster** — that is not a claim, it is what §4 measures.

**No dual microkernel, no pair pack buffers, no register-blocking question.** The alternative — tagging the
complex microkernel — was examined and rejected on the ladder: `_cmplx_kernel_body` (`gemm.jl:1640`) already
packs *planar* (`ApR`/`ApI`, `_pack_A_cmplx!` `:1573`) and holds two accumulators per cell (`cr`, `ci`); a
dual instantiation would keep the same tile and accumulator count and drop one of the four FMAs (`:1660`,
the `−ai·bi` term). That is a real 25% flop cut on a kernel the real path already beats (the complex
non-3M kernel exists because Karatsuba costs complex precision; dual has no such constraint). It would
also put dual on the complex blocked driver and off Strassen. Rejected unless §5's measurement says the
planar route loses somewhere that matters.

## 2. Per-op decisions

### 2.1 gemm — rides the real gemm; three products, one split, one combine

```
split:   A → (A_v, A_p)    B → (B_v, B_p)                 raw pair reads, planar writes; `_split2!`, mirror of `_split3!` minus the sum plane
P1 = A_v·B_v                                               _gemm_real_dims!(tA, tB, …, 1, 0, A_v, B_v, P1)
P2 = A_v·B_p ;  P2 += A_p·B_v                              second call with β = 1 (accumulate in gemm's own epilogue)
combine: C_v = α_v P1 + β_v C_v
         C_p = α_v P2 + α_p P1 + β_v C_p + β_p C_v(old)    elementwise pass over interleaved C, β=0 ⇒ overwrite; mirror of `_combine3!`
```

Routing: a new branch in `_gemm_core!` (`gemm.jl:2841`) between the complex branch (`:2924`) and the generic
fallback (`:2971`): `_pairT(T) && _strided1(C) && _strided1(A) && _strided1(B)` → `_gemm_dual3!`. `tA/tB`
pass through to the real sub-gemms exactly as 3M does; `cA/cB` are dropped (conjugation is the identity on
`Dual <: Real`, `dual_l2.md` §4.2). The existing α==0 quick-return is deliberately **not** widened to dual
(`:2848-2852` explains why: `Dual(0, p)` has `iszero == true` and a live derivative) — the combine multiplies
by both parts, so `α = Dual(0, p)` is correct by construction. Scratch: the same `_gemm_3m_scratch` pool,
indices 1,2,4,5,7,8. Nothing complex is touched; **complex byte-identity is trivially satisfied**.

Window: none proposed for the upper edge (the complex `_CGEMM_3M_MAX = 2048`, `:1488`, exists because above
it "the extra passes cost more than the flop cut" *relative to the 4-product kernel*; dual has no 4-product
kernel to fall back to and the generic loop is 8× off). Lower edge: the split/combine is O(n²) against
O(n³), 6% of the FMA count at n=32 and 1.5% at n=128; the crossover against the generic loop is measured
(§4) and expected at or below the complex `_CGEMM_TINY = 6` (`:1443`). Thin-k: see §5.

### 2.2 syrk / herk — real syrk + real syr2k

`C = α·A·Aᵀ + β·C` with `A = A_v + A_p ε`: value `A_v A_vᵀ`, partial `A_v A_pᵀ + A_p A_vᵀ` — the second
is **exactly `syr2k(A_v, A_p)`**. So: `P1 = syrk(A_v)`, `S2 = syr2k(A_v, A_p)` (both real, both triangular,
`syrk!` `level3.jl:6283`, `syr2k!` `:7042`), then the triangular combine
`C_v = α_v P1 + β_v C_v`, `C_p = α_v S2 + α_p P1 + β_v C_p + β_p C_v(old)` over the stored triangle (mirror of
`_combine3_tri!` `:5512`). `trans='T'` passes through. `herk` on dual collapses to syrk (α, β real by its
contract; conjugation identity) — same reasoning as hemv → symv. One half-product more than a real syrk and
the same three half-products as the complex 3M syrk (`_ctrk_3m_ok` `:5505`).

### 2.3 trmm — three real trmms

`B := α·op(A)·B`, A triangular, `B = B_v + B_p ε`:

```
B_p ← trmm(A_v, B_p) + trmm(A_p°, B_v(old))      A_p° = A_p with its diagonal zeroed when diag='U'; non-unit uses A_p's diagonal as is
B_v ← trmm(A_v, B_v)                              computed AFTER B_p, which needs the old B_v
```

Three real trmms on planar scratch (the second is `diag='N'` on `A_p°`, since the partial of a unit diagonal
is 0 — the only subtlety on this op, and it is handled by zeroing a diagonal in a private plane). `side`,
`uplo`, `transA` pass through to the real `_trmm!` (`level3.jl:1020`). α applied in the combine, as gemm.
The complex twin's own tiny-k arms (`_trmm_cmplx_small_L!` `:395` etc.) are irrelevant here.

### 2.4 trsm — two real trsms and one real trmm; **no dual division**

Solve `op(A)·X = B` (side L; side R is the mirror). `(A_v + A_p ε)(X_v + X_p ε) = B_v + B_p ε` gives

```
A_v X_v = B_v                  →  X_v = trsm(A_v, B_v)
A_v X_p = B_p − A_p X_v        →  R   = B_p − trmm(A_p°, X_v);   X_p = trsm(A_v, R)
```

This is the "own derivation" the division rule demands, and it lands entirely on real kernels: the division
by the dual diagonal is never performed — it is replaced by a second real solve with the same `A_v`. Cost:
2 real trsms + 1 real trmm (+ α combine), against the complex twin's 4 real-equivalent products in its
recursive/gemm-blocked split (`level3.jl:4048`). `diag='U'` is not a special case here either. Numerically
the partial inherits the real trsm's conditioning twice (the second solve's right-hand side is a residual of
the first) — the same as ForwardDiff's own `\` on a dual matrix, which does exactly this.

### 2.5 symm / hemm / syr2k / her2k — same pattern, listed for completeness, after the four above

* symm: `_split3_sym!` (`gemm.jl:2569`, `herm=false`) already reflects a stored triangle into full planes; dual
  symm is three real gemms on (`A_v`, `A_p`) × (`B_v`, `B_p`), the mirror of `_hemm_3m_L!` (`:2681`).
  hemm collapses to symm.
* syr2k: `C = α(ABᵀ + BAᵀ)`: value `syr2k(A_v, B_v)`, partial `syr2k(A_v, B_p) + syr2k(A_p, B_v)`. her2k
  collapses.

### 2.6 What every L3 op shares

One `_split2!` (raw pair → two planes, r×c, column stride honoured; `_split3!` minus the sum stream, ~10
lines), one interleaved `_combine_dual!` per shape class (full, triangular; dual α, β applied through
`_parts`), scratch from the existing per-real-type pool. Zero new SIMD code. Zero complex code touched.

## 3. The design questions, explicitly

* **Do the pack buffers hold pairs? What happens to register blocking?** They do not: the split happens
  *before* the real gemm, whose pack buffers, `_MR×_NR` tile and accumulators are untouched and real. There
  is no pair in any register on the L3 path. (For the record, the rejected tagged-complex alternative would
  also be planar — `_pack_A_cmplx!` splits re/im — and would keep complex's 2 accumulators per cell, so it
  would not spill *more* than complex; it would just be on the slower driver.)
* **trsm division:** §2.4 — eliminated, not vectorised.
* **hemm / herk / her2k on dual:** collapse to the symmetric op (identity conjugation; `dual_l2.md` §4.2).

## 4. Measurement plan (galen, `taskset -c 6`, Chairmarks median)

1. Correctness before speed: dual gemm/syrk/trmm/trsm vs the generic loop on awkward `(m, n, k)`, all
   `trans`/`side`/`uplo`/`diag`, dual α with zero value, β = 0/1/dual; `ForwardDiff.jacobian` of `A*B`,
   `A\B` against the planar route; the ε²-leak test (`A_p`, `B_p ~ 1e155`: `A_p·B_p` never forms, so the
   value must be exactly the real product and the partial finite).
2. The triage table re-run: gemm at n ∈ {8, 16, 32, 48, 128, 512, 1024, 2048, 4096}; the bar is **equal
   time to zgemm inside its 3M window (48–2048)** and *less* outside it. Absolute: ≈ 3× dgemm's time
   + O(n²).
3. Thin-k sweep: `m = n = 1024`, `k ∈ {1, 2, 4, 8, 16, 32}` — where the O(n²) split stops being amortised
   (§5).
4. Scratch footprint at n = 4096 (§5), and whether the sub-gemms should go through `_gemm_core!` to pick up
   Strassen at large n (`_gemm_real_dims!` deliberately does not: `gemm.jl:2620-2628`).

## 5. Adversarial: the case against this design

* **Thin k is where the planar route stops fitting.** The split moves `2(mk + kn)` elements and the combine
  `2mn`; the products cost `3mnk` FMAs. At `k = 1` (a rank-1 update through gemm, common in AD code that
  builds Jacobians column by column) the O(n²) passes are of the same order as the work, and `P1`/`P2` are
  each a full `m×n` write + read. The complex 3M keeps `min(m,n,k) ≥ 16` for the same reason
  (`_CGEMM_3M_KMIN`, `:1491`). I propose **no k floor for dual** — below it the alternative is the generic
  loop at 0.12, and even at `k = 1` the planar route is 3 vectorised real gers plus two streaming passes —
  but the sweep in §4.3 decides, and if a floor is needed the honest fix for thin k is the tagged L2 ger
  applied k times, not a dual microkernel.
* **Scratch at n = 4096 is 6 planes × 128 MB (Float64).** The complex 3M stops at 2048 (9 × 32 MB); dual has
  no fallback, so an unpaneled implementation would hold ~800 MB for a 4096² dual gemm. Fix if the user
  cares: panel over `nc` columns of B/C (the three sub-gemms are column-separable), bounding scratch to A's
  two planes plus panels. It is a loop around the same three calls, not a new kernel — but it is also a
  knob (`nc`), and the PDM ladder says it must derive from `_L3_BYTES`. I would ship v1 unpaneled with the
  pool and add the panel loop only if the footprint is objected to.
* **The sub-gemms do not ride Strassen.** `_gemm_real_dims!` goes to unpacked/blocked only. At n ≥ the
  Strassen threshold real gemm beats OB by 1.2–1.3× through the 7-multiply recursion; the dual route would
  leave that on the table unless the sub-gemms go through `_gemm_core!` on the `PtrMatrix` planes. Cheap to
  try; measured in §4.4; it changes nothing in complex.
* **trmm/trsm on the planar route are 3× a real trmm/trsm, which is the same ratio as gemm — but the
  triangular ops have tiny-k arms in complex** (`_trmm_cmplx_small_L!` `:395`, the gemmtrsm leaf `:4048`)
  that the dual route does not reproduce at n ≤ 48. Expect the dual ratio to drop below 1 vs the complex
  twin at n = 32 on trmm/trsm; whether that matters is a question for the user (AD code calls `A\B` at
  n = 32 far less than it calls gemm).
* **The op forced into a shape that does not fit: none on L3** — planar is the natural shape for ε² = 0.
  The forced *decision* is dropping `cA/cB` (conjugation) silently; a user who calls `gemm!(…, transA='C')`
  on duals gets the transpose, which is the only consistent answer, but nothing tells them.
* **What measurement would detect a wrong design?** Ratio vs zgemm inside the 3M window ≠ 1.0 ± noise means
  the O(n²) passes are not free (memory-bound combine on a cold C — the same `_combine3!` cost complex
  pays); the thin-k sweep; the ε² test (a `reinterpret` shortcut anywhere would fail it instantly).
* **If only half ships: gemm.** It is the op that matters, it is the one AD frameworks actually hit
  (`A*B` on `Matrix{Dual}` falls to a scalar loop today, everywhere in the ecosystem), and syrk/trmm/trsm
  are 30-line drivers over the same split/combine once it exists. If only a quarter ships: gemm with a
  `k ≥ 16` floor and the generic loop below, i.e. the complex 3M's own window without the upper edge.

## Implementation notes (2026-09-13, after the sparring round)

### Sparring point 1 — the cost model, confirmed by measurement

Dual gemm is three real gemms with no cancellation and no window; complex 3M is three real gemms plus a sum
plane per operand and a cancelling combine. The sub-gemms go through `_gemm_core!` (not `_gemm_real_dims!`),
so they take Strassen where real gemm does. Measured on galen (median of 4 rounds, plots.jl's CL3 regime),
zgemm ÷ dual on identical bytes, with dgemm for scale:

| n | dual ms | zgemm ms | dgemm ms | zgemm/dual | dual/(3·dgemm) |
|---|---|---|---|---|---|
| 8 | 0.0004 | 0.0001 | 0.0001 | 0.35 | 1.39 |
| 16 | 0.001 | 0.001 | 0.000 | 0.70 | 1.31 |
| 32 | 0.006 | 0.006 | 0.002 | 0.96 | 1.15 |
| 48 | 0.015 | 0.016 | 0.004 | 1.05 | 1.19 |
| 128 | 0.265 | 0.268 | 0.080 | 1.01 | 1.11 |
| 512 | 16.6 | 16.0 | 5.10 | 0.96 | 1.09 |
| 1024 | 117.9 | 124.3 | 37.3 | 1.05 | 1.06 |
| 2048 | 834 | 977 | 272 | **1.17** | 1.02 |

So the bar — dual ≥ the complex twin — holds from n = 48 and is within noise at 32 and 512 (0.96); at 2048,
where the twin leaves its 3M window and the dual sub-gemms ride Strassen, dual is 17 % faster. The
`dual/(3·dgemm)` column is the O(n²) overhead share: 2–15 %, shrinking with n. Below 32 the three entry
overheads and the split dominate (0.35 at n = 8) — still 2× the generic loop's 0.17 — so the route is taken
above `_fh_cgemm_tiny()` (= 6), the complex floor, reused as is. The triage's 0.12 became 1.0–1.17.

Thin k, m = n = 1024 (the adversarial §5 case): dual beats the generic loop at every k (1.5× at k = 1, 7.7× at
k = 32) and beats zgemm from k = 16; below that the O(n²) planes are not amortised (0.44 at k = 1). No k floor
was added: below 16 the alternative is the generic loop, which is slower still. The honest fix for thin k, if
it ever matters, is k tagged L2 gers — not a floor.

### Sparring point 2 — scratch, with numbers

Peak scratch is 6 planes: `A_v, A_p` (m×k), `B_v, B_p` (k×n), `P1, P2` (m×n) — for square n exactly `6n²`
reals = **the byte size of the three dual operands** (each 2n² reals), i.e. a 2× footprint, from the same
grow-only pool as complex 3M (slots 1,2 / 4,5 / 7,8 of `_gemm_3m_scratch`). At n = 4096 Float64 that is
805 MB, on a 30 GB box; the real Strassen path at the same n already holds ~440 MB of Winograd level scratch
(10 buffers per level, depth 3), so the dual route is the same order as what the fastest real path does, and
nothing pages. Correction to the sparring premise: `_split3!` does NOT panel — complex 3M splits the whole
operand and bounds its scratch only by the `_CGEMM_3M_MAX = 2048` window (9 × 32 MB there). The two costs
that remain real: the pool never shrinks (a process that once ran a 4096² dual gemm keeps 805 MB resident —
complex 3M has the same property at 288 MB), and paneling would cap it at the price of losing Strassen on the
panels. Unpaneled is what shipped; the panel loop is the upgrade path if the footprint is objected to.

### The compositions (galen, complex twin ÷ dual, side L, median of 6)

| op | n=32 | n=128 | n=512 | n=1024 |
|---|---|---|---|---|
| syrk (real syrk + real syr2k) | 0.88 | 1.17 | 1.14 | 1.17 |
| trmm (three real trmms) | 0.81 | 0.98 | 1.04 | 1.00 |
| trsm (two real trsms + one trmm; no dual division) | 0.82 | 1.07 | 1.06 | 1.04 |

At n = 32 the complex twins' tiny-k arms (`_trmm_cmplx_small_*`, the gemmtrsm leaf) have no dual counterpart
and the three entries cost more than the work — as §5 predicted. From 128 the compositions sit at or above
the twin. `herk` routes to `syrk` on a Dual (identity conjugation; `trans='C'` → `'T'`), the same decision as
`hemv → symv` and `dotc == dotu`. Correctness: `bench/probes/dual_l3_check.jl` — every uplo/trans/side/diag,
n ∈ {1,3,8,17,64,129,257} × k ∈ {1,5,32,130}, both real types, dual α and a zero-value β, plus a
`ForwardDiff.derivative` through `trsm!` against `−L⁻¹ dL L⁻¹ B` — 0 failures; the same in
`test/dual_tests.jl` ("Dual L3: syrk/herk/trmm/trsm compositions …").

### What the L3 route never touches

No complex function changed on L3. The additions are `_split2!`, `_combine_dual!`, `_combine_dual_tri!`,
`_zero_diag!`, `_copy_plane!`, the four drivers (`_gemm_dual3!`, `_syrk_dual!`, `_trmm_dual!`, `_trsm_dual!`)
and one branch in each of `gemm!`, `syrk!`, `herk!`, `trmm!`, `trsm!` behind `_pairT`, which is `false` for
every type in the main env — so the branches fold away without ForwardDiff and the `--trim` build is unaffected.
