# Dual numbers: BLAS-1 for forward-mode AD

Status: **design agreed, not yet implemented.** This page records the design, the measurements it
rests on, and the decisions taken — including the ones that closed off tempting alternatives.

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
does not claim otherwise.

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
inside the generator. The tag needs no ForwardDiff knowledge — it is "zero even lanes".

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
