# Dual numbers: LAPACK (Phase 1 — triage and design; §3/§4's `geqrf` fix landed, see §7)

Companion to [`dual.md`](dual.md) (BLAS-1), [`dual_l2.md`](dual_l2.md) and [`dual_l3.md`](dual_l3.md).

**The headline is that most of this is already done**, and that is a consequence of CLAUDE.md
requirement #3 (one kernel generic over `T<:Number`) rather than luck. A LAPACK routine is mostly a
driver over BLAS-2/BLAS-3 calls; dual L2/L3 landed in `6109aeec`; so the factorizations inherited the
speedup without a line of new code. Triage exists to find the exceptions.

## 1. Triage: correctness

`bench/probes/dual_lapack_triage.jl`, n=32, `Dual{Nothing,Float64,1}`, compared against
LinearAlgebra's generic factorization on **both value and partial** — a routine that is right on the
value alone has lost the derivative, which is the entire point.

| routine | status | max |Δvalue| | max |Δpartial| |
|---|---|---|---|
| `potrf` | OK | 0.0 | 0.0 |
| `getrf` | OK | 0.0 | 2.2e-16 |
| `geqrf` | OK | 3.6e-15 | 2.8e-14 |
| `getrs` | OK | 6.9e-18 | 2.8e-17 |
| `potrs` | OK | 1.7e-17 | 1.9e-17 |
| `trtrs` | OK | 1.4e-17 | 5.6e-17 |
| `trtri` | OK | 1.7e-18 | 1.1e-18 |
| `potri` | OK | 1.4e-17 | 5.2e-18 |
| `sytrf` | runs | no generic oracle compared |
| `pstrf` | OK | — (see the correction below) |
| `gesvd` (values) | OK | returns `Vector{Dual}` |
| `_syev!` | OK **since `18fc0e20`+** | eigenvalues 1.7e-13, **derivatives 1.7e-12** vs `ForwardDiff.derivative` |
| `gesvd` (vectors) | **ArgumentError** | gated to Float64/complex by design (`svd.jl:1656`) |
| `geqp3` | **MethodError** | genuinely gated: both methods are `where {T <: BlasFloat}` |

### Correction (2026-09-13) — three of the four "gaps" were the PROBE, not the code

An earlier revision of this page listed `_syev!`, `pstrf` and `geqp3` as `MethodError`s. Two of those
were my triage probe calling the wrong signature, and I published them as missing capability:

* **`pstrf!`** is `(A, piv, tol::Real; uplo)` and is declared `where {T}` — **no type restriction at
  all**. The probe omitted `tol`. It has worked on duals the whole time.
* **`_syev!`** is `(jobz::Char, uplo::Char, A)`, not `(A; uplo)`. The probe passed a keyword and one
  positional. A `MethodError` was the correct answer to a wrong call.
* **`geqp3!`** is the only real gate, and it is a type gate: `where {T <: BlasFloat}` on both methods.

The lesson is that a `MethodError` says "no method for THESE arguments", not "this routine cannot do
this" — and the difference is one `methods()` call away. Three probe bugs in one triage is a bad rate;
read the signature before reporting a capability gap.

`_syev!` did turn out to have a real, shallow defect underneath the bad call — see §8.

`getrf`'s partial at 2.2e-16 is the one to notice: **pivoting already selects on |value|**, because
`_l1v` (`core.jl`) exists precisely so `iamax` does not break value ties on the derivative. Had it
compared `_l1` on duals, ForwardDiff's lexicographic `isless` would have made the pivot sequence
depend on the seed direction.

## 2. Triage: performance

`bench/probes/dual_lapack_perf.jl`. Reference = LinearAlgebra's generic factorization (what a
ForwardDiff user gets today without PureBLAS), Chairmarks median, fresh operand per sample.

| routine | n=64 | n=256 | reads as |
|---|---|---|---|
| `potri` | 8.42 | **21.03** | reaches the blocked path |
| `trtri` | 9.79 | **20.84** | reaches the blocked path |
| `potrf` | 2.32 | 7.24 | reaches the blocked path |
| `getrf` | 1.98 | 5.67 | reaches the blocked path |
| `geqrf` | 0.57 | **0.45** | does NOT |

The direction of the trend is the diagnosis. Four routines get *better* with n — the signature of a
blocked BLAS-3 trailing update amortising. `geqrf` gets *worse* (0.57 → 0.49 → 0.45 at 64/128/256).

**Control that isolates it:** the same PB `geqrf!` on `Float64` at n=256 measures **2.03×** generic.
So this is not a QR problem, it is a dispatch problem. (An earlier probe allocated `tau` inside the
timed core; re-measured with `tau` in setup, the numbers are unchanged — that was not the cause.)

## 3. Root cause for `geqrf`

`qr.jl:467`, the method duals dispatch to:

```julia
function geqrf!(A::AbstractMatrix{T}, tau::AbstractVector{T}; nb::Int = 0) where {T <: Real}
    qr_unblocked!(view(A, 1:m, 1:n), view(tau, 1:k))    # nb accepted and DISCARDED
end
```

It runs a fully unblocked BLAS-2 reduction for every size. `Float64` (`:480`) and `BlasComplex`
(`:340`) each get a blocked compact-WY driver whose trailing update is `gemm!`; `T<:Real` gets
nothing. The `nb` keyword is accepted and thrown away, which is how it reads as intentional and is
not.

Not a bug, for the record: `_qr_nb`'s `sizeof(Float64)` is correct — it is only called from the
Float64 path, and complex has its own `_zqr_nb(T, m, n)` over `sizeof(T)` (`:325`).

## 4. Two designs for dual `geqrf`, and they are not equivalent

**(a) Generalise the blocked driver over `T`.** Port `geqrf!(::AbstractMatrix{Float64}, …)` to
`T<:Real`: `1.0`/`0.0` → `one(T)`/`zero(T)`, a `T`-typed workspace pool beside `_qr_ws`, and a
decision about the `useskinny` µarch gate (which assumes 8-byte elements: `16 * pb * mp <= _L2_BYTES`
is an L2-residency criterion in *bytes of Float64*, so it needs `sizeof(T)`, req#8). Mechanical, and
it lifts every `T<:Real` — not just duals.

**(b) Planar, mirroring `dual_l3.md`.** For `A = A_v + ε A_p`, QR has a closed-form derivative: factor
the value plane with the **existing fast Float64 driver**, then recover `R_p` and the `Q_p` action
from real-only products. The differential relations are
`Qᵀ A_p = Ṙ + S R` with `S = Qᵀ Q_p` skew-symmetric, so `S` is determined by the strictly-lower part
of `Qᵀ A_p R⁻¹` and `Ṙ = Qᵀ A_p − S R` — i.e. one real `gemm`, one real `trsm`, and a skew
completion. That rides the 2.03× Float64 path instead of re-deriving it, and mirrors the L3 result
where planar dual **beat** the complex twin (`dsyrk1` 8.39 vs its complex sibling).

Recommendation: **(b) for the dual entry, (a) anyway for `T<:Real` generally** — (a) is the honest
fix for every other real element type (`Float32` already has its own path; `BigFloat`, `Measurement`,
etc. do not), and (b) is where the performance is. They are complementary, not alternatives.

Open question for sparring: (b) needs `Q` applied, not formed. `ormqr`/`larfb` already exist on the
Float64 path — can `S` be obtained without materialising `Q`, or does the skew completion force it?

## 5. The remaining gaps

| gap | kind | note |
|---|---|---|
| `geqrf` blocked path for `T<:Real` | perf | **DONE** (§7): 0.57/0.49/0.45 → 1.67/2.56/2.75 at n=64/128/256 |
| `_syev!` on `Dual` | dispatch | **DONE** (§8): the gap was `_trdws`, not the driver |
| `pstrf` on `Dual` | — | **NOT A GAP** — my probe omitted `tol`; it always worked (§1 correction) |
| `geqp3` on `Dual` | dispatch | **the one genuine gate**: both methods are `where {T <: BlasFloat}` |
| `gesvd` singular **vectors** | feature | needs generic `orgbr` + vector-carrying `bdsqr`; **values already work** |
| stopping-rule audit | **correctness** | clean for everything that dispatches, `pstrf` and `_syev!` included; see below |

## 6. The audit nothing else catches

`<` on a `Dual` compares lexicographically on (value, partial). Every routine that **branches on a
comparison** can therefore take a different branch depending on the seed direction, producing a
factorization that is correct-looking and whose derivative is wrong — and no ordinary correctness
test detects it, because any single seed is self-consistent.

**Built and run first, per that reasoning**: `bench/probes/dual_seed_independence.jl` holds the value
plane fixed, varies only the partial plane, and asserts the value part of the output is bit-identical.

Result — **clean for everything that currently dispatches**:

| routine | |
|---|---|
| `potrf` `getrf` `geqrf` `trtri` `potri` `sytrf` | seed-independent |
| `getrf` **pivot sequence** (`ipiv`) | bit-identical across seeds |
| **`pstrf`** — rank, pivots AND values | bit-identical across seeds (`dual_seed_pstrf.jl`) |
| **`gesvd`** singular values | bit-identical across seeds |
| **`_syev!`** eigenvalues | bit-identical across seeds (`syev_dual_verify.jl`) |
| `geqp3` | still untestable — the one genuine `MethodError` |

An earlier revision said `pstrf` "could not be tested" and framed this audit as a gate on future work.
That was wrong, and wrong in the risky direction: `pstrf` terminates on a **rank criterion** and had
been dispatching on duals all along, so a seed-dependent rank would have been a LIVE correctness bug
in shipped code, not a hypothetical one. Tested now — rank 48 vs 48, pivots and values identical
under two unrelated seed directions.

`_syev!` matters for a second reason: it is the first **iterative** routine to reach dual dispatch.
Everything else audited here is a direct factorization with at most a pivot comparison, whereas an
eigensolver runs to a convergence test — and `<` on a `Dual` is lexicographic on (value, partial), so
it could converge at a different iteration count per seed, giving eigenvalues correct to tolerance but
*different* between seeds. It does not; the value plane is bit-identical.

Still owed: `geqp3` when it accepts duals (it pivots on column norms, so it can change the column
ORDER), and an iteration-count witness rather than an output comparison — a convergence test that
exits one step early for one seed is invisible in the output when both results are within tolerance.

## 7. Result: (a) shipped, (b) measured and declined (2026-09-13, branch `geqrf-real-blocked`)

**(a)** The Float64 blocked driver is now the one `geqrf!` for every `T<:Real` (`qr.jl`): `one(T)`/`zero(T)`,
the panel width from `_qr_nb(T, m, n)` — the same L2-residency ramp with the byte term `sizeof(T)`, so a
16-byte `Dual` spills at half the m·n a Float64 does — the skinny unpacked `W = Vᵀ·C` gated on
`T <: BlasReal` with its criterion spelled in bytes (`2·pb·mp·sizeof(T) ≤ L2`, which is the old
`16·pb·mp ≤ L2` for Float64), and a per-call workspace for the non-Float64 reals beside the owned Float64
pool. On a `Dual` the trailing `syrk!`/`trmm!`/`gemm!` are the planar compositions of `dual_l3.md`, so
nothing new was written for the trailing update; the panel is the generic dgeqr2 (`_larfg!` +
`_house_left!`, scalar on a Dual). Float32 and BigFloat were on the unblocked path too and are lifted by
the same change.

galen (Zen3, core 6, freq-locked), `bench/probes/dual_geqrf_check.jl` — the probe §2's numbers came from,
Chairmarks median of 24, fresh operand per sample, PB / LinearAlgebra generic:

| n | before | after |
|---|---|---|
| 64 | 0.57 | **1.67** |
| 128 | 0.49 | **2.56** |
| 256 | 0.45 | **2.75** |

The trend now has the sign of the other four factorizations (§2's table re-run: potrf 2.00/5.46, getrf
1.93/5.15, geqrf 1.65/2.73, trtri 9.40/17.9, potri 7.44/17.0 at n=64/256). The Float64 control is
unchanged at 2.04×. Seed independence (`dual_seed_independence.jl`, N=48 — above `_QR_UNBLK_MAX`, so it
runs the new blocked path): `geqrf` still seed-independent, as are the other five and getrf's pivot
sequence. Float64 and complex byte-identity (`bench/probes/dual_qr_native.jl`, 32 instantiations: the
Float64 driver body on `Matrix` and on views, the SIMD panel and its five helpers, `_qr_nb`, `_qr_ws`,
`geqp3` as a `_qr_nb` consumer, the complex driver/panel/reflector/apply for ComplexF64 and ComplexF32,
orgqr/ungqr/ormqr controls): **19 identical, 12 same instruction count + opcode multiset, 0 different** —
exactly the classification the unedited tree gives against its own baseline in a second process (the 12
are the large drivers whose frame layout drifts per process), and the only DIFFERENT is Float32's
`geqrf` body (123 → 1469 instructions), which is the change.

**(b), answered.** Can `S = Qᵀ Q̇` be obtained without forming `Q`? Yes: `M = Qᵀ A_p` is one implicit
application (`ormqr`, reflectors only), `S` is the strictly-lower part of `M R⁻¹` (one `trsm`, skew
completion), `Ṙ = M − S R` (one `trmm`). No `Q` is ever materialised. **But `S` is not the deliverable.**
`geqrf!`'s output contract is the stored reflectors and `τ` — on a `Dual`, their partials `v̇_k`, `τ̇_k` —
because that is what `ormqr`/`orgqr`/`\`/LinearAlgebra's `QRCompactWY` read back. `v̇_k` depends on the
derivative of column `k` of the *partially reduced* matrix `H_{k-1}⋯H_1 A`, i.e. on `Ḣ_1 … Ḣ_{k-1}`;
recovering the `v̇_k, τ̇_k` from `S` (or from `Q̇ = QS`) means peeling the reflector sequence in order —
which is the forward recursion (a) already performs. So (b) computes `Ṙ` and `Q̇` and then still has to
run (a) for the reflector partials; it cannot replace it.

And it would not be faster even if that recovery were free. `bench/probes/dual_qr_b_accounting.jl`
(galen, same regime, median of 24) measures (b)'s optimistic bound — the value-plane Float64 QR plus one
`ormqr` plus one `trsm` plus one `trmm`, the recovery *excluded* — against (a):

| n | (a) dual geqrf | = panel + trailing | (b) bound: F64 qr + ormqr + trsm + trmm | (a)/(b) |
|---|---|---|---|---|
| 128 | 539 µs | 204 (38 %) + 335 (62 %) | 113 + 324 + 54 + 55 = 546 µs | 0.99 |
| 256 | 3626 µs | 1615 (45 %) + 2011 (55 %) | 689 + 2621 + 440 + 386 = 4136 µs | **0.88** |

The `ormqr` term dominates: PB's `ormqr!` is the unblocked dorm2r, and applying `n` reflectors to an
`n×n` `A_p` is ~2× the flops of the QR itself (4mn²−2n³ vs 2mn²−⅔n³ for square), so even a blocked
larfb-style `ormqr` at the QR's own efficiency (~1.4 ms at 256) would put the bound at ~2.9 ms — 1.24×
(a) at best, before paying the reflector-partial recovery, which is (a) again. Declined.

**What the accounting does say is where the next lever is:** the panel — the scalar generic dgeqr2 on a
Dual — is 45 % of (a) at n=256 and the trailing update is already planar. A pair-tagged panel (the dual
BLAS-1 dot/axpy kernels of `dual.md` behind `_house_left!`, which today gates its SIMD arm on
`T <: BlasReal`) is the honest follow-on, worth up to ~1.8× on top of the 2.75 if the panel went to
zero. Not done here.

## 8. `_syev!`: the gap was the WORKSPACE, not the driver (2026-09-13)

Once called correctly, `_syev!` still failed — but not on its own signature:

    MethodError: no method matching _trdws(::Type{ForwardDiff.Dual{Nothing, Float64, 1}})

`_sytrd_lower!` and `_syev!` are both `where {T <: Real}` and generic throughout. What was not generic
was the scratch: `_trdws` had exactly four methods, one per `BlasFloat`, each returning a const owned
pool (`_TRDWS_F64` …) — the GKH ownership pattern. An open type set cannot have a const pool, so a
`Dual` found no method at all.

**This is the same defect `_qr_ws` had**, fixed the same way: a generic
`_trdws(::Type{T}) = _TRDWork{T, real(T)}()` allocating per call. The four consts still win by
specificity, so every `BlasFloat` path is untouched; only types that have no pool pay an allocation,
against an O(n³) reduction.

That two independent routines had the identical hole suggests the class is worth sweeping rather than
discovering one `MethodError` at a time: **every const-per-type owned workspace is a candidate**.

Verified after the fix (n=32, `Dual{Nothing,Float64,1}`), and note the second line is the one that
matters — a factorization can be right on the value and have silently lost the derivative:

| | |
|---|---|
| eigenvalues vs `eigvals` | 1.7e-13 |
| **derivatives vs `ForwardDiff.derivative`** | **1.7e-12** (against a derivative magnitude of 2.14, so not vacuous) |
| seed-independent | yes — and `_syev!` is the first ITERATIVE routine to reach dual dispatch, so its convergence test was the real risk |

### `_sytrd_nb` was mis-sizing the panel, silently

`_sytrd_nb(n) = _qr_nb(n, n)` used the Float64 two-arg form, so a 16-byte `Dual` was blocked as if it
were 8 — a req#8 violation that produces no error, just a wrong panel width. Now
`_sytrd_nb(::Type{T}, n) = _qr_nb(T, n, n)`, with the 2-arg form kept for existing callers.

**The complex `_hetrd!` call site is deliberately NOT typed — and the cost of typing it was measured,
not guessed.** The width is `clamp(8 · cld(m·n·sizeof(T), L2), 8, 32)`, a coarse step function taking
only 8/16/24/32, so doubling `sizeof(T)` moves the bucket boundaries rather than every value. Swept
n=32:16:1200 on Zen4: the two sizings **agree at 51 of 74 sizes and differ at 23 (31%)**, the split
opening at **n=272**, where Float64-sized gives 8 and ComplexF64-sized gives **16 — a doubled panel**.
So typing it changes shipped complex tridiagonalization (`hetrd` → `zheev`/`zheevN`/`hegvd`) across
about a third of the size range, and `zheev`/`zheevN` publish at 1.176/1.072 against the current
blocking.

The residency criterion arguably *does* want `sizeof(T)` there too — this is a real req#8 debt, not a
false alarm. But it is a **measured** change needing a CLP sweep, not a free one taken as a side
effect of a dual-number fix, and the sweep now has its range: the cells that can move are the 23
sizes from n=272 up where the buckets disagree. Deferred with its cost known rather than silently
applied.

## 9. What is actually left

| gap | kind |
|---|---|
| `geqp3` on `Dual` | the one genuine type gate (`where {T <: BlasFloat}`) |
| `gesvd` singular **vectors** | deliberate; needs generic `orgbr` + vector-carrying `bdsqr`. Values work |
| `geqrf` scalar panel | **DONE** (§10.2): `_house_left!` has a pair arm on the tagged dot/axpy; the n=32 panel went 32.4 → 13.8 µs |
| `_syev!` on `Dual` degrading with n | **DONE** (§10.1): planar; 0.68× → 1.83× at n=256 |
| `_QR_UNBLK_MAX` for pair types | data, not a change: with the tagged panel, unblocked beats blocked to n≈48–56 on Zen3 (§10.2), so the shared 32 is early for a Dual — same PDM-literal debt the Float64 knob already carries, one more row for its table |

## 10. `_syev!` on a Dual: the reference was already planar (2026-09-14, branch `dual-syev`)

### 10.1 The degradation and its cause

DLP `dsyev1` fell monotonically with n on all three boxes (Zen4/Zen3/Zen5: 1.95/2.24/1.91 at n=32 → 0.66/0.67/0.69
at 256). Decomposed on galen (`bench/probes/dual_syev_decomp.jl`, Chairmarks median of 10, fresh SPD operand per
sample, one call per sample — the DLP regime):

| n | path | ref | `_syev!` | `_sytrd_lower!` | `_sytd2_lower!` | `_sterf!` | n × `symv!` |
|---|---|---|---|---|---|---|---|
| 32 | unblocked (nb=16, nx=32) | 102 | 48 | 21 | 21 | 25 | 10 |
| 128 | blocked | 1724 | 1713 | 1373 | 1069 | 326 | 512 |
| 256 | blocked | 8246 | 12077 | **10837** | 8556 | 1232 | 4105 |

(µs.) Three things the table settles. The iteration (`_sterf!`) is 10 % at n=256 — the guess that it was the
O(n³)-ish part was wrong. The blocked reduction IS taken and is **slower than the unblocked one** on a Dual
(10.8 vs 8.6 ms): its `syr2k!` lands on the generic `_syr2k_acc!`, and the `symv!` it shares with `_sytd2_lower!`
is the scalar generic `_symv!` (no SIMD arm on a pair — dual_l2.md §2.6 left symv at tier 2), which alone is
half of the unblocked reduction. So the dual-arithmetic route was scalar on ⅔ n³ flops.

And the reference is not a generic dual eigensolver at all. `eigvals(Symmetric{Dual})` dispatches to
**ForwardDiff's own `_eigvals`** (`ForwardDiff/src/dual.jl`): `eigen` of the VALUE plane through LAPACK, then
`diag(Q' * A_p * Q)` as two full gemms. It is the closed form of this task's planar suggestion, riding OpenBLAS.
That is why the ratio fell with n — PB was running scalar dual arithmetic against a vendor O(n³) path.

### 10.2 What shipped

**Planar `_syev!` for pair types** (`_syev_pair!`, eigen.jl; routed from `_syev!` on `_pairT(T) && _strided1(A)`):
split A into (A_v, A_p); `_syev!('V', uplo, A_v)` on the real driver; `B = A_p·Q` by one real `symm!` (reads
only the `uplo` triangle — the same contract A has); `λ_p[i] = q_iᵀ b_i` (n dots). For `'V'`, `C = Qᵀ B`,
`S_ij = C_ij/(λ_j − λ_i)`, `Q_p = Q·S` (two more real gemms). **Degenerate clusters** — eigenvalues closer than
`n·eps·max|λ|`, the value-plane solver's own resolution — take the projected block: `M = Q_cᵀ A_p Q_c`, its
ascending eigenvalues are the derivatives of the ascending branches (degenerate perturbation theory, verified
against a one-sided finite difference to 1e-6), and for `'V'` the cluster's columns of Q are rotated onto M's
eigenvectors with S zero inside the cluster. ForwardDiff's reference gets the cluster case wrong (it reads the
diagonal of `Q'A_pQ` in whatever basis LAPACK picked: `[-0.99, -0.38, 0.22]` where the block gives
`[-2.73, 0.21, 1.37]` = the finite difference), so the test covers it against the block formula, not the oracle.
Simple eigenvalues match ForwardDiff to 1e-13 (values and partials), eigenvector partials to 1e-12–1e-10.

Planar accounting that justified it before writing (galen, `dual_syev_planar_bound.jl`, Float64): `_syev!('V')`
+ `symm!` + n dots = 54 / 127 / 534 / 951 / **4783** µs at n=32/50/100/128/256, against the dual path's 48 / 149 /
883 / 1713 / 12077 — a win everywhere but n=32, where the fixed cost of the vector solve loses 6 µs (12 %). Not
worth a size knob; noted.

**Pair arm in `_house_left!`** (svd.jl): the same dot+axpy shape as the BlasReal arm on `_dot_pair_simd` /
`_axpy_pair_simd!`. This was the whole of the `dgeqrf1` n=32 cell: at n ≤ 32 `geqrf!` is ONE `qr_unblocked!`
panel (the crossover was never in play — both arms were unblocked), and that scalar panel ran 32.4 µs against
27.7 for LinearAlgebra's `qrfactUnblocked!`, the same algorithm. Tagged: 13.8 µs. `bench/probes/dual_qr32.jl`
after the change: unblocked 3.1 / 7.2 / 13.8 / 23.1 / 36.6 / 79.4 µs vs blocked 28.3 / 39.1 / 73.7 at n=40/48/64 —
so for a pair the unblocked panel now wins to n≈48–56, past the shared `_QR_UNBLK_MAX = 32` (§9 row).

### 10.3 Result (galen, Zen3, core 6, freq-locked; `bench/probes/dual_lp_ratios.jl`, the DLP probe shape, both arms same run)

| n | `dsyev1` before | after | `dgeqrf1` before | after |
|---|---|---|---|---|
| 32 | 2.24 | **1.89** | 0.87 | **2.10** |
| 50 | 1.62 | **1.84** | 1.30 | **1.90** |
| 100 | 1.15 | **1.81** | 2.08 | **2.96** |
| 128 | 1.00 | **1.77** | 2.36 | **3.47** |
| 256 | 0.67 | **1.96** | 2.74 | **4.14** |

("before" = the published Zen3 fleet cells; "after" = `3a6f315b`, the arena-workspace commit. The planar-only
commit `6147e924` read 1.88/1.82/1.80/1.64/1.83 and 2.10/1.88/2.92/3.25/4.16 — the same picture within run drift.) `dsyev1` is flat at ~1.8 instead of falling; the reference and PB
now run the same algorithm, and the ratio is PB's real `syev 'V'` + `symm` against OpenBLAS's `syevr` + two gemms.

`jobz='V'` changed too, so its ratio is stated (`dual_syev_v_ratio.jl`, vs ForwardDiff's `eigen(Symmetric{Dual})`,
same regime): **1.77 / 1.64 / 1.29 / 1.63 / 1.43** at n=32/50/100/128/256. There is no "before": the dual-arithmetic
`'V'` path went through `_stedc!` on a Dual and was never timed (it is not a DLP cell). The n=100 dip is a single
run's number and was not chased.

### 10.4 The `!` contract: dual `geqrf!` allocated 18 736 B per call; now 0

`bench/probes/lapack_entry_alloc.jl` (warm first, `@allocated` on the 2nd and 3rd call, n=64): every LAPACK bang
entry was 0 B on Float64 and on Dual except `geqrf!` on Dual — the generic `_qr_ws(::Type{T}, …)` built four fresh
`Matrix{T}` per call (64·8·16 + 2·8·8·16 + 8·64·16 = 18 432 B + headers), rationalised in its comment as small
"against the O(m·n·k) factorization" — the reasoning the rule forbids. Now: the blocked loop is `_geqrf_wy!` over a
supplied workspace, Float64 hands it the owned pool exactly as before, every other `T` borrows inside a `@scope`
(`Vt` 0×0 for a non-BlasReal — the skinny arm never reads it). All 14 entries read 0 B; the Float64 driver body is
still same-count + same-multiset against the baseline. Same treatment for `_sytrd_lower!`: the per-call
`_TRDWork{T}()` fallback from `c6a9ef19` is gone, non-BlasReal reals borrow `W`/`tmp` (`_sytrd_blocked!`); a Dual no
longer reaches it at all. **Still allocating, pre-existing and on every type including Float64:** `_sytd2_lower!`
builds its `v`/`w` scratch per call (192 B on Float16, 640 B on a Dual at n=96 via the blocked tail) — that is
shipped Float64 code and gets its own byte-identity pass, not a side effect here. **CLOSED in `9b8e8463`;
see §11.2** — both leaves now borrow from the arena, 0 B on Float64/ComplexF64/Dual, `syev` unchanged.

Byte identity (`dual_qr_native.jl`, `dual_syev_native.jl`; Float64, Float32, ComplexF64, ComplexF32): every
instantiation of the edited functions is identical or same-count+same-multiset. The probe flags `_latrd_lower!`,
`_sytrd_lower!` (F32), `_hetrd!`, `_heev!` (C32) as DIFFERENT by 1–31 instructions of `mov`/`lea`/`movabs` — none
of which were edited — and **the unedited master tree reproduces the same cells with the same deltas across two
fresh processes** (3051 vs 3020 for `_latrd_lower!` Float64), so for those large drivers instruction COUNT is
process drift too, not just order. Seed independence (`dual_seed_independence.jl`, `syev_dual_verify.jl`): clean;
the planar `_syev!` has no dual comparison left to leak through.

## 11. Fleet result and the two `!`-contract fixes (2026-09-14, `9b8e8463`)

### 11.1 The DLP/DL3 gate, all three boxes, after the planar `_syev!` and pair-tagged panel

Measured on the fleet under a verified lock (2794 / 3674 / 1972 MHz achieved under load), `arms=pb,generic`
so both arms are recorded in the SAME run — which is why these cells carry no anchor question at all,
unlike the `arms=pb` real groups. Published geomean (worst cell):

| row | Zen3 · galen | Zen4 · wintermute | Zen5 · neuromancer |
|---|---|---|---|
| `dsyev1` | 1.15 (0.67) → **1.85 (1.69)** | 1.12 (0.66) → **1.92 (1.75)** | 1.13 (0.69) → **1.97 (1.72)** |
| `dgeqrf1` | 2.08 (0.87) → **3.11 (2.09)** | 2.16 (0.82) → **2.69 (1.82)** | 2.24 (0.79) → **2.78 (1.77)** |
| `dgemm1` | 8.21 (1.85) → 8.29 (1.87) | 8.13 (2.33) → 8.33 (2.31) | 7.61 (2.32) → 7.70 (2.49) |

`dsyev1` per cell is now **flat in n** on every box — 1.88/1.86/1.85/1.80/1.69 (Zen3),
1.96/1.90/1.96/1.92/1.75 (Zen4), 2.00/1.97/2.03/1.88/1.72 (Zen5) at n = 32/50/100/128/256. The curve that
motivated req#9 — monotonically falling to 0.66–0.69 and therefore *slower than the generic fallback it
replaces* — is gone. No dual cell on any box is below gate.

### 11.2 `_sytd2_lower!` / `_hetd2_lower!`: the allocation §8 left open is closed

§8 recorded `_sytd2_lower!` building `v`/`w` per call as "shipped Float64 code, gets its own pass". That
pass is this one. Both the symmetric and the Hermitian leaf now open a `@scope` and `borrow!`, opened in
the LEAF rather than at the two call sites so one borrow covers both reach paths (the whole reduction
when `n ≤ 2·nb`, and the tail of every blocked `_sytrd_lower!`); nesting inside the non-BlasReal `@scope`
in `_sytrd_lower!` is fine, each scope rewinds its own bump.

`bench/probes/sytd2_alloc.jl`, leaf and driver, n = 48 and 256: **0 B on Float64, ComplexF64 and Dual**,
spectrum residual ≤ 3.3e-15 on all three. Per req#9 the speed is stated too, and it is a null result:
published `syev` is 1.41 → 1.40 / 1.46 → 1.46 / 1.34 → 1.34 across the fleet, i.e. unchanged.

### 11.3 Dual `gemm!` was leaking 1344 B at n≥512, and the cause was not the one in the code

Per call at steady state: 1344 B at n=512 **and n=600** (so not a power-of-two effect), 5376 B at n=1024,
Dual only. `Profile.Allocs` at `sample_rate=1` put it inside `_strassen_rec!` and named the boxed types —
`SubArray{Float64,2,Matrix{Float64}}` and, tellingly, `PtrMatrix{Float64}`, an ISBITS handle built
precisely to avoid a heap header.

It is a container-type MIX. Each world is closed under sub-viewing on its own: `Matrix` operands give
`SubArray{…,Matrix}` quadrants that collapse back to themselves, `PtrMatrix` operands give `PtrMatrix`
quadrants. But `_gemm_dual3!` passes `PtrMatrix` planes while `_strassen_lvl_scratch` hands back `Matrix`,
so every level pairs the two and the seven recursive call sites' signature combinations multiply level by
level until inference gives up. The A/B that settles it varies only the handle:

| n | `_strassen_depth` | Matrix handles | PtrMatrix handles |
|---|---|---|---|
| 512 | 2 | 0 B | 448 B |
| 1024 | 3 | 0 B | 1792 B |

448 × 3 plane products = 1344; 1792 × 3 = 5376. Exact, both sizes. Matching the scratch to the operand
container (`_str_like`) closes each world and one signature serves every level.

**The first diagnosis — "depth 3 makes SubArray headers" — predicted the Matrix arm would leak too, and it
measures 0 B.** That is why the A/B was run instead of the reading being shipped.

Speed, per req#9, by controlled same-process A/B on galen (PB arm only, since nothing else changes; a
cross-run comparison against a cached `generic` arm would not be adjudicable):

| n | before `779e4051` | after | ratio |
|---|---|---|---|
| 256 | 2.097 ms | 2.003 ms | 0.955 |
| 512 | 15.235 ms | 14.936 ms | 0.980 |
| 1024 | 117.10 ms | 114.997 ms | 0.982 |
| 2048 | 835.73 ms | 830.27 ms | 0.993 |

Faster at every size, 0.7–4.5%. The Float64 path is untouched and that is verified rather than argued:
`code_native` on `_strassen_rec!` specialized for `Matrix` arguments is 18478 instructions with an
identical opcode multiset before and after, only the per-process label symbol differing.

### 11.4 What is still open

`gemm!`, `syrk!` and `trmm!` still cannot carry a **static** `@test_noalloc` on any element type. That is
not this defect: AllocCheck proves ALL paths, and `_strassen_lvl_scratch`'s lazily sized `Vector{Matrix}`
pool is counted even where it is runtime-dead. The two escapes are deleting the allocation site or
registering a genuine high-water barrier, and the pool is neither today — it re-allocates on shape
mismatch by design. Putting that scratch on the arena would make `_arena_grow!` the single barrier, and
the broadcast cost of the `PtrMatrix` handles that route implies has been measured and is not a
blocker (equal on a plain add, 0.50–0.82× on the fused combine, i.e. faster). Until then the instrument
matching req#10 for BLAS-3 is a warmed runtime assertion, not the static proof.
