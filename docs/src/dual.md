# Dual numbers: BLAS-1 for forward-mode AD

Status: **all three steps implemented** (2026-09-12; see "Implementation notes" at the end). This page
records the design, the measurements it rests on, and the decisions taken — including the ones that
closed off tempting alternatives.

`ForwardDiff.Dual{Tag,V,1}` and `Complex{V}` have **byte-identical memory layout**, so PureBLAS can
serve forward-mode AD from the same SIMD machinery it uses for complex. Nothing else in the Julia
ecosystem ships a BLAS optimised for array-of-structs dual numbers; ForwardDiff users currently fall
through to scalar loops because `Dual` is not a `BlasFloat`.

## The layout fact this rests on

Verified 2026-09-12 on Julia 1.13.0 / ForwardDiff:

```julia
sizeof(Dual{Nothing,Float64,1}) == 16 == sizeof(ComplexF64)
reinterpret(Float64, [Dual{Nothing}(1.0,2.0), Dual{Nothing}(3.0,4.0)]) == [1.0, 2.0, 3.0, 4.0]
```

Same as `reinterpret(Float64, ComplexF64[1+2im, 3+4im])`. `Dual` is `value::V` +
`partials::Partials{N,V}` wrapping an `NTuple{N,V}`, so at `N = 1` it is exactly an interleaved pair.
Every load, store, stride, alignment and in-lane shuffle transfers unchanged.

The extension asserts this at load time (`sizeof == 16 && fieldoffset(..., 2) == 8`), so a future
ForwardDiff layout change becomes a load error rather than silent garbage.

## ⛔ The trap the layout creates

Because the layouts are bit-identical it is tempting to `reinterpret` a `Vector{Dual}` to
`Vector{ComplexF64}` and call the complex kernel. **That is wrong wherever two elements multiply.**

| | product | real part / value |
|---|---|---|
| complex | `(a+bi)(c+di)` | `ac − bd` |
| dual | `(a+bε)(c+dε)` | `ac`  (ε² = 0) |

Reinterpreting silently subtracts `bd` from the value. It is only valid where no element × element
product occurs: `copy`, `swap`, `axpy` with a **real** alpha, `scal` by a **real** alpha.

The ε²-leak test exists to catch exactly this: run `dot(x,x)` with partials ~`1e155` so `Σ x_p²`
overflows. The value must stay finite and the partial must equal `2 Σ x_v x_p`. A reinterpreted
complex kernel fails it.

## ⛔ "Dual needs 3 multiplies, complex needs 4" does not survive into SIMD

The scalar op counts are real — dual multiply is `ac`, `ad + bc`; complex is `ac − bd`, `ad + bc` —
but in the interleaved swap-adjacent form both algebras are the same two FMAs and one shuffle:

```julia
muladd(shuffle(x), sgn, muladd(x, arv, y))
```

with `sgn = [−b, +b, …]` for complex and `[0, +b, …]` for dual. The second FMA runs over the whole
vector either way. **The dual advantage is a scalar-op-count fact, not a kernel fact**, and the design
does not claim otherwise. (As shipped, the dual arm pairs `[0, +b, …]` with the **duplicate-even**
shuffle rather than the swap — same op count, and it keeps the `0·` off the partial lane; see the NaN
note under `dupEven`.)

## Measured: where dual actually loses today

Dual GB/s ÷ the complex twin's GB/s on identical bytes, through the real PureBLAS entries
(wintermute, Julia 1.13.0, scratch env — adequate for an order-of-magnitude decision, **not** a gate
number):

| op | n=1e3 | n=1e4 | n=1e5 | n=1e6 |
|---|---|---|---|---|
| axpy | 0.82 | 0.80 | 1.03 | 1.00 |
| scal | 0.69 | 0.65 | 0.77 | 0.99 |
| dot | 0.37 | 0.34 | 0.49 | 0.92 |
| asum | 0.21 | 0.18 | 0.23 | 0.61 |
| iamax | 0.37 | 0.29 | 0.29 | 0.36 |
| **nrm2** | **0.07** | **0.06** | **0.08** | **0.20** |

`nrm2` is flat at **4.8 GB/s at every n** — 13–16× slower than complex. Flat in `n` means it is not
memory-bound: neither fast branch applies (`_simd1` is `BlasReal`-only, `_cplx_re` is `Complex`-only),
so it falls to the `lassq` scaling loop, `_nrm2_acc` per element with a division and branches.

**This is what sets the scope.** The reductions are the prize; `axpy`/`scal` converge to parity past
L2 and are only 20–35% down in cache.

## The `dupEven` formulation

Rather than the swap-adjacent shuffle, dual reductions use an **in-lane duplicate-even** shuffle
`(0,0,2,2,…)`, so `dupEven(v) = [x_v, x_v]` per pair. Same cost class as the swap, and it lands the
useful quantity in both lanes:

| op | body | parity fold gives |
|---|---|---|
| nrm2 | `acc = muladd(v, dupEven(v), acc)` | `[Σ x_v², Σ x_v·x_p]` — one FMA, no wasted lanes |
| asum | `acc += flipsign(v, dupEven(v))` | `[Σ\|x_v\|, Σ sign(x_v)·x_p]` — no blend |
| iamax | magnitude `= abs(dupEven(v))` | value magnitude broadcast to both lanes |

`flipsign` uses the sign **bit**, matching ForwardDiff's `abs` derivative
(`signbit(x) ? -1 : 1`) bit-for-bit at ±0.0.

`dupEven` also removes a NaN hazard. With `sgn = [0, b, …]` the even lane computes `t + 0·x_p`, so an
infinite partial makes `0·Inf = NaN` poison the **value**. Putting the `0·` on the value lane instead
means it only misbehaves when the value is already `Inf`, where the scalar result is `Inf` too.

## Per-op decisions

| op | dual implementation | semantics |
|---|---|---|
| copy, swap | route to `_copy_simd!`/`_swap_simd!` over 2n reals | no arithmetic |
| axpy, scal | shared tagged body; real-alpha bypass to the real kernel over 2n | — |
| dotu | shared body, per-algebra epilogue | value `= Σ x_v y_v`; the `Σ x_p y_p` lane is the ε² term and is **discarded** |
| dotc | identical to dotu | `Dual <: Real` and ForwardDiff defines no `conj`, so `conj(::Real) = x`. **The ring involution `a − bε` is wrong for AD**: `dot(x,x)` would give `Σa²` with a zero partial, i.e. `d‖x‖²/dx = 0` |
| nrm2 | standalone; `dupEven` body, not the complex `_sumsq_simd` | `√(Σx_v²) + (Σ x_v x_p / ‖x_v‖) ε`. Fast path guards `isfinite(ss) && !iszero(ss) && isfinite(sp)`; falls back to the existing Dual lassq loop only on overflow |
| asum | standalone; `dupEven` + `flipsign` | value `Σ\|x_v\|`, partial `Σ flipsign(x_p, x_v)` |
| iamax | magnitude functor on the existing argmax scaffold | **argmax of `\|value\|`, first occurrence on ties** — see below |

`N ≠ 1`, nested duals, and non-`BlasReal` `V` miss the routing predicate and take the generic scalar
loop unchanged. Generalising to `N > 1` is a different layout (stride `N+1`, not a pair) and is out of
scope.

### iamax: a deliberate, user-visible semantics change

`abs(::Dual)` returns a **Dual**, and ForwardDiff's `isless` is lexicographic — value first, then
partials. So when two elements have equal magnitude, the current generic loop breaks the tie **on the
derivative**:

```julia
x = [Dual(3.0, -5.0), Dual(3.0, 1.0)]   # identical magnitude 3.0
argmax(abs.(x))              # 2  — decided by the partial
argmax(abs.(value.(x)))      # 1  — first occurrence, real-BLAS semantics
```

A partial is an infinitesimal and carries no magnitude information, so letting it decide means the
answer depends on the seed direction. It matters because **`iamax` is the pivot-selection primitive in
`getrf`**: a pivot chosen by perturbation direction can make the LU pivot sequence differ between the
primal and the AD run on identical input, and forward-mode derivatives of a pivoted factorisation
assume the sequence is locally constant. Nothing is lost by ignoring partials — the return type is an
`Int`.

**Decision (user, 2026-09-12): value-only, first occurrence, matching real BLAS.** `_l1v` is added so
the scalar and SIMD paths agree.

## Performance bar

Not "dual ≥ complex" uniformly — that is too weak in places and unachievable in others. Honest
per-vector op counts:

| op | dual | complex |
|---|---|---|
| nrm2 | shuffle + FMA | FMA |
| asum | shuffle + flipsign + add | abs + add |
| iamax | shuffle + abs | abs + shuffle + add |
| dot, axpy, scal | identical | identical |

Complex `asum` is `Σ\|Re\|+\|Im\|` (`core.jl:217-218`) — **not** the modulus — so it is already the
minimal real kernel over 2n reals, and dual cannot beat it: dual must move data across the lane pair
and complex need not. The bar is therefore:

* **equal** to the twin at every n for dot/axpy/scal/copy/swap;
* **strictly greater in L1/L2** for iamax (dual drops one op; `simd_kernels.jl:1043-1050` measured one
  deleted op as +21% at n=1e3);
* **equal past L2** for nrm2/asum, and in-cache within the cost of the extra op — priced with
  `mca_report` before anyone calls it a limit.

## Architecture

`PairAlgebra` as a type *hierarchy* was dropped: the largest win (nrm2) shares nothing with complex,
so an abstraction spanning all ops would not have earned its place. What survives is a codegen-time
**multiply-rule tag** on the three generated bodies where dual and complex differ only by constants or
epilogue — `_dot_cmplx_simd`, `_axpy_cmplx_{phase,wide}!`, `_scal_cmplx_simd!`.

It earns that on `dot` (3× in cache, byte-identical complex body) and because every future complex arm
— the `U = 8` band arm, a DRAM prefetch arm — then lands on dual for free. A separate copy would
silently miss both.

The tag and its singletons live in **PureBLAS, not the extension**: a `@generated` body runs at the
world age of its own definition, so a codegen hook defined later in an extension would be invisible
inside the generator. The tag needs no ForwardDiff knowledge — it is `Val(:cplx)` / `Val(:dual)`, the
same form the iamax scaffold uses, and it resolves to two codegen constants (`_pair_shuf`, the in-lane
shuffle: swap-adjacent vs duplicate-even; `_pair_sgn`, the lane multiplier: `[−b, +b, …]` vs
`[0, +b, …]`) plus, for dot, the epilogue. The complex entry names (`_dot_cmplx_simd`,
`_axpy_cmplx_{simd,phase,wide}!`, `_scal_cmplx_simd!`) survive as `@inline` forwarders to the `:cplx`
instantiation, so no complex caller changed.

`Expr(:meta, :inline)` stays inside returned generator bodies; `@inline` does not propagate into
`@generated` CodeInfo.

## The extension

```
[weakdeps]  ForwardDiff = "..."
[extensions] PureBLASForwardDiffExt = "ForwardDiff"
```

Five one-line methods and **no kernels**: `_pairalg`, `_pairreal`, `_parts`, `_mkpair`, `_l1v`, plus
the layout assert. ForwardDiff never enters `[deps]`, so the `--trim` build never loads it and
`libpureblas.so` is unaffected.

`Tag` is never inspected. The only Duals the fast path constructs carry `x`'s own tag, exactly as the
scalar path does today via `convert`; a mixed-tag alpha fails in `convert` identically on both paths.

## Implementation sequence

Deliberately ordered so a complex regression can never be confounded with the dual work.

1. **Complex only, no dual code.** Parameterise `U` in `_axpy_cmplx_phase!` (currently hardcoded 4 at
   `simd_kernels.jl:471`, while real axpy has a `U = 8` band arm) — Derive `min(8, (_NVREG−4)÷3)`,
   which reproduces today's AVX2 code exactly. A/B at n=2e4/3e4/6e4 with real axpy at 6e4 as the
   L2-edge control. Route complex copy/swap off the scalar loop. Real-alpha zaxpy bypass, measured at
   n=1e3 and dropped if it costs. Add a `zscalc` bench row. Gate, commit.
2. **Extension + standalone dual kernels**: routing, then nrm2 (the 13×), asum, iamax. No complex
   arithmetic kernel is touched.
3. **Tagged shared bodies**, dot first, then axpy/scal. Each op its own commit, guarded by complex
   `code_native` byte-identity against the post-step-1 baseline plus an opcode-histogram equality test
   between the complex and dual instantiations. If complex moves by one instruction, that op ships as
   a copy under the histogram test instead.

## What this does NOT fix

Stated plainly because the complex gate gaps motivated the work and mostly are not complex-kernel
gaps:

* **zscal is not a complex kernel problem.** The gate row scales by `1.0000001 + 0im`
  (`bench/plots.jl:1231`), and `_scal!` routes a real alpha to the real kernel over 2n reals
  (`level1.jl:33-37`). Every zscal cell is the real-scal L2/L3-edge notch. `_scal_cmplx_simd!` has **no
  gate coverage at all** — hence the new `zscalc` row in step 1.
* **zaxpy past L3 on Zen5 tracks real axpy** (0.921/0.919 vs 0.925/0.949 at n=3e5/1e6) — memory
  system, not algebra.
* **zdotu Zen5 at n=1e6** is a DRAM-regime question for a roofline decomposition, not a kernel.

The one genuinely complex-specific cell is **zaxpy@3e4** (Zen4 0.963, Zen5 0.834), which step 1
targets.

⚠ Zen5 (neuromancer) is not gate-authoritative: its lock drops on power change, and it reads **FP256**
(`test/autotune_tests.jl:21-23` — Strix/Krackan, `fpw = 32`, double-pumped, lands with Zen4). Do not
build a width argument on "Zen5 is native-512"; that is a retracted claim from an old family lookup.

## Implementation notes (steps 1 and 2, 2026-09-12)

Measured after step 2, same method as the baseline table above (wintermute, Chairmarks median of 6
rounds, plots.jl regime, `bench/probes/dual_twins.jl`) — dual GB/s ÷ complex twin:

⚠ **READ THE ALPHA COLUMN BEFORE THE RATIO.** axpy/scal have two regimes and only one of them is
implemented. Step 2 routes a **real** alpha to the real kernel over 2n reals; a **dual** alpha (nonzero
partial) still takes the generic scalar loop, because that is step 3. The first draft of this table
printed the real-alpha 1.00 against a dual-alpha baseline of 0.82, which reads as "axpy is done" and
would argue a reader out of step 3. Measured directly (axpy, n=1e3 / 1e4):

| alpha | dual GB/s | complex GB/s | D/C |
|---|---|---|---|
| real (`Dual(1.7, 0.0)`) — bypass applies | 117.1 / 129.8 | 117.1 / 130.2 | **1.00 / 1.00** |
| dual (`Dual(1.7, 0.3)`) — generic, step 3 | 95.8 / 103.5 | 116.8 / 129.8 | **0.82 / 0.80** |

| op | n=1e3 | n=1e4 | n=1e5 | n=1e6 | before |
|---|---|---|---|---|---|
| axpy (real α) | 1.00 | 1.01 | 1.01 | 1.02 | — (new path) |
| axpy (dual α) | 0.82 | 0.80 | 1.01 | 1.00 | 0.82 0.80 1.03 1.00 — **unchanged, step 3** |
| scal (real α) | 1.00 | 1.00 | 1.01 | 1.01 | — (new path) |
| scal (dual α) | 0.69 | 0.66 | 0.90 | 0.94 | 0.69 0.65 0.77 0.99 — **unchanged, step 3** |
| copy | 1.00 | 1.03 | 1.01 | 1.01 | — |
| swap | 1.00 | 1.05 | 0.99 | 1.02 | — |
| dot | 0.20 | 0.34 | 0.40 | 0.58 | 0.37 0.34 0.49 0.92 (generic loop; step 3) |
| asum | 0.60 | 0.99 | 0.98 | 1.01 | 0.21 0.18 0.23 0.61 |
| iamax | 1.17 | 1.20 | 1.20 | 1.18 | 0.37 0.29 0.29 0.36 |
| nrm2 | 0.73 | 1.00 | 1.00 | 0.97 | 0.07 0.06 0.08 0.20 |

What `@code_native` settled: `dupEven` emits `vmovddup` (in-lane, one per vector, no cross-lane shuffle);
nrm2 is `vmovddup` + FMA. SIMD.jl's `flipsign` lowered to **two** ops (`vpmovq2m` + masked `vxorpd`) and
put asum at 0.48× in L1; written as `v ⊻ (dupEven(v) & signmask)` it is one `vpternlogq`
(`_flipsign_bits`), 0.60× in L1 and parity from L2. In-L1 nrm2/asum sit at the extra op priced above;
not yet run through `mca_report`.

**Deviation from this design — nrm2's overflow fallback.** The page said "falls back to the existing Dual
lassq loop". The 1e200-scale test it prescribed showed that loop is numerically wrong there: `_lassq`
divides Duals and ForwardDiff's quotient rule squares the ~1e200 denominator, so the partial is garbage
(one input: −1.66 vs analytic −1.39; `LinearAlgebra.norm` over Duals — the intended oracle — gave −1.47,
broken the same way). The overflow path is now `_nrm2_dual_scaled`: real lassq over the values, partial
`Σ (x_v/scale)·x_p / √ssq`. The generic scalar loop for `N ≠ 1` / nested duals keeps the old behaviour.
The test oracle at extreme scale is the analytic partial, not `ForwardDiff.derivative` on `norm`.

Step 1 findings: the phase-kernel `U` parameterisation is byte-identical at `U=4` (code_native, 202 =
202 instructions); `U=8` on AVX-512 is +1.3% [1.004, 1.024] at n=3e4 and a tie at 2e4/6e4, so it does
not by itself close zaxpy@3e4. The real-alpha zaxpy bypass costs nothing measurable (compare-not-taken
0.990 [0.961, 1.025] at n=1e3) and wins 8.7% with a real alpha. LLVM does not vectorize the scalar
complex copy/swap loops (memmove / memcpy+memmove); routed onto the real kernels: copy 1.9× in L1, level
from L2; swap 2.4–3.1× everywhere. The complex iamax scaffold is byte-identical after taking the
magnitude as a codegen parameter (352 = 352 / 426 = 426 instructions, F64/F32).

## Implementation notes (step 3, 2026-09-12)

The three tagged bodies landed as three commits (dot, axpy, scal), each gated by
`bench/probes/dual_step3_native.jl`: the complex instantiation's normalised `code_native` diffed
against a pre-edit dump (20 cells — five bodies × F64/F32 × `Vector`/`Ptr` carrier), plus an opcode
histogram of the **main SIMD loop** of the complex vs dual instantiation. Both gates held for all
three ops, so **nothing shipped as a copy**:

| body | complex code_native, before = after (F64 vec / ptr; F32 vec / ptr) | main-loop histogram, complex vs dual |
|---|---|---|
| `_dot_pair_simd` dotu / dotc | 201 = 201 / 199 = 199 ; 219 = 219 / 217 = 217 (dotc 232 / 230 ; 219 / 217) | 46 = 46 (F64), 45 = 45 (F32), raw-equal |
| `_axpy_pair_phase!` | 213 = 213 / 209 = 209 ; 238 = 238 / 234 = 234 | 46 = 46 / 45 = 45, equal modulo 8 `vshufpd` ↔ 8 `vmovddup` |
| `_axpy_pair_wide!` | 226 = 226 / 222 = 222 ; 251 = 251 / 247 = 247 | 54 = 54 / 53 = 53, same |
| `_scal_pair_simd!` | 292 = 292 / 289 = 289 ; 257 = 257 / 254 = 254 | 46 = 46 / 45 = 45, same |

The histogram is taken over the loop, not the whole function, because the epilogue is per-algebra by
design (dot drops the ε² lane; the scalar tails differ) and because LLVM happens to auto-vectorise the
complex scalar tail of dotu/axpy/scal (`vaddsubpd`) and not the dual one — tail code, not the kernel.
The loop histogram is also a `@testitem` (`DualNative`), with a positive control (an FMA loop was
found) and a negative one (the histogram is not blind).

Measured with a **dual alpha** (`bench/probes/dual_twins3.jl`: axpy `Dual(1.7, 0.3)` vs
`1.7 + 0.3im`, scal `Dual(1.0000001, 1e‑7)` vs `1.0000001 + 1e‑7im`, so both arms run the tagged
complex-layout kernel and neither takes the real-alpha bypass; Chairmarks median of 6 rounds,
wintermute, dual GB/s ÷ complex twin, two samples):

| op | n=1e3 | n=1e4 | n=1e5 | n=1e6 | before (generic loop) |
|---|---|---|---|---|---|
| axpy (dual α) | 0.98–0.99 (188.6 vs 191.8 GB/s) | 1.00 | 1.00–1.01 | 1.00–1.02 | 0.61 0.80 0.93 1.03 |
| scal (dual α) | 0.99 (166.5 vs 168.2) | 1.00 | 1.01 | 0.99 | 0.62 0.64 0.72 0.83 |
| dot | 1.00 (152.5 vs 152.4) | 1.00 | 1.01 | 1.00 | 0.20 0.34 0.40 0.58 |

The "before" row here is the true dual-alpha baseline against the true complex kernel — the step-2
table's complex arm used a real alpha and so measured the real-axpy bypass (117 GB/s at n=1e3), not
`_axpy_cmplx_wide!` (192 GB/s). Two things follow. First, the only residual is the ~1% at n=1e3 on
axpy/scal, where the loop differs by one mnemonic (`vmovddup` for duplicate-even vs `vshufpd` for the
swap) — it moved 0.98→0.99 between two samples and has not been chased. Second, an observation outside
this step's scope: at n=1e3 the real-alpha bypass (real axpy over 2n reals, 117 GB/s) is markedly
slower than the complex wide arm on the same bytes (192 GB/s), so the bypass may be the wrong arm in
L1; not measured further here.

Design on contact: the doc's "differ only by constants" held literally for axpy/scal — the tag resolves
to a shuffle pattern and a sign vector and nothing else in the loop changes. For dot it held for the
loop and the epilogue is per-algebra (value = `pfld[1]` alone; the discarded `Σ x_p y_p` lane must not
even be multiplied by zero, since it may be `Inf`). The dual dot returns the bare `(value, partial)`
tuple and `_dotu`/`_dotc` wrap it with `_mkpair`, so the kernel never sees a Dual type; `_pairv(x)`
(the real type under a pair vector, static) was the one accessor the routing needed beyond the five.
The NaN-hygiene item now covers the dual-alpha axpy/scal paths, which is why duplicate-even rather
than the swap was the right shuffle for the dual arm: with the swap, an infinite partial would put
`0·Inf` on the value lane, which the generic loop keeps finite.
