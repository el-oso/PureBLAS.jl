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
| `_syev!` | **MethodError** | does not accept `Dual` |
| `gesvd` (vectors) | **ArgumentError** | gated to Float64/complex by design (`svd.jl:1656`) |

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
| `_syev!` on `Dual` | dispatch | MethodError; symmetric eigen is `_syev!`, not `syev!` |
| `pstrf` on `Dual` | dispatch | MethodError — and it is rank-terminating, see §6 |
| `geqp3` on `Dual` | dispatch | MethodError — and it pivots on column norms, see §6 |
| `gesvd` singular **vectors** | feature | needs generic `orgbr` + vector-carrying `bdsqr`; **values already work** |
| stopping-rule audit | **correctness** | clean for everything that currently dispatches; see below |

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
| `pstrf`, `geqp3` | could not be tested — MethodError on `Dual` |

So the two routines whose branches would matter MOST are exactly the two that do not accept duals
yet. That makes this audit a **gate on their implementation, not a clearance of it**: `pstrf`
terminates on a rank criterion and `geqp3` pivots on column norms, and either can change the SHAPE of
the answer (chosen rank, column order) rather than merely its ordering. Whoever implements them runs
this probe as part of the work, and `gesvd`/`_syev!` iteration counts need the same treatment —
extend the probe with an iteration-count witness, since a convergence test that exits one step early
for one seed is invisible in the output alone.

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
