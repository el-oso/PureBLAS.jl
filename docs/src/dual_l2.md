# Dual numbers: BLAS-2 design (Phase 1 — for sparring, nothing implemented)

Status: **design only, 2026-09-12.** Companion to `dual.md` (BLAS-1, shipped) and `dual_l3.md`. Every claim
below is either a citation into the current tree (`a0f0a703`) or a number from the triage probe; where it is a
prediction it says so.

## 0. Triage: what the generic scalar path achieves today

galen (Zen3, core 6, boost off, 3673 MHz), Julia 1.13.0, scratch env with ForwardDiff, through the public
entries, plots.jl's CL2 regime (fresh operands per sample, `_L2REP(n)` dependent reps), Chairmarks median of 6
rounds. Cell = complex-twin time ÷ dual time on identical bytes (1.0 = parity; 0.25 = dual is 4× slower).
Probe: `bench/probes/dual_l23_triage.jl` (gitignored).

| op (dual / complex twin) | n=32 | n=128 | n=512 | n=1024 | n=2048 | n=4096 |
|---|---|---|---|---|---|---|
| gemvN / zgemvN | 0.25 | 0.24 | 0.28 | 0.27 | 0.36 | 0.47 |
| gemvT / zgemvT | 0.31 | 0.21 | 0.25 | 0.25 | 0.38 | 0.47 |
| ger / zgeru | 0.43 | 0.37 | 0.36 | 0.36 | 0.70 | 0.83 |
| symv / zsymv (generic in PB too) | 0.88 | 0.86 | 0.83 | 0.83 | 0.80 | 0.84 |
| symv / zhemv (the SIMD twin) | 0.74 | 0.38 | 0.31 | 0.29 | 0.30 | 0.33 |
| trmv / ztrmv | 0.68 | 0.39 | 0.37 | 0.34 | 0.35 | 0.47 |
| trsv / ztrsv | 0.72 | 0.47 | 0.36 | 0.33 | 0.36 | 0.45 |

Absolute, for scale: zgemvN at n=4096 streams 268 MB in 7.05 ms (38 GB/s, DRAM-bound); the dual generic loop
takes 15.0 ms (18 GB/s). Unlike BLAS-1, **nothing here is at parity**: even past L3 the generic loop is 2×
off on gemv/trmv/trsv and 1.2–1.4× off on ger. Every op is worth a kernel except `symv` measured against
`zsymv` — and that comparison is against a complex twin PureBLAS itself runs on the generic loop
(`_symv!`, `level2.jl:2648`, has no complex SIMD branch), so it is not a parity, it is two slow paths.

## 1. The one decision that shapes the level: interleaved tags, not planar splits

BLAS-3 (see `dual_l3.md`) rides the real kernels by splitting each operand into value/partial planes once
(O(n²)) and running three real products (O(n³)). That does **not** transfer to BLAS-2: a gemv reads A once,
16 B per element, and is bandwidth-bound; a planar split reads 16 B and writes 16 B per element, and the three
real gemvs then read 8 B each — 56 B per element against 16. It would be slower than the generic loop at every
n in the table. So BLAS-2 keeps the operands interleaved and **tags the complex kernels**, exactly as
`dot`/`axpy`/`scal` were tagged in BLAS-1 (`_pair_shuf`/`_pair_sgn`, `simd_kernels.jl:65-67`).

The dual multiply in interleaved form, with `a = [a_v, a_p]` a vector of pairs and `c = c_v + c_p ε` a scalar:

```
(c·a)_v = c_v·a_v                 (c·a)_p = c_v·a_p + c_p·a_v
```

which is `muladd(a, [c_v,c_v,…], ·)` (both lanes) followed by `muladd(dupEven(a), [0, c_p, …], ·)`, the same two
FMAs and one shuffle as complex — differing only in the shuffle pattern and the sign/zero vector. Every L2 body
below is one of three shapes: an **axpy** (ger, gemvN's inner update, trmv/trsv columns), a **dot** (gemvT,
trmv/trsv transposed columns), or a **fused dot+axpy** (hemv). The first two are already-solved tags.

## 2. Per-op decisions

| op | decision | complex scaffolding it rides | what is algebra-specific |
|---|---|---|---|
| gemvN | **tagged** | `_gemv_n_ri_panel!` `level2.jl:1404`, drivers `:1532/:1604/:1619` | hoisted scalar `cr`; one in-loop shuffle (new pattern, §2.1) |
| gemvT | **tagged, epilogue only** | `_gemv_tc_block_cmplx!` `:1728`, drivers `:1812/:1837` | the fold (as `_dot_pair_simd`, `simd_kernels.jl:755-765`) |
| gemvC | **routes to gemvT** | — | conj is the identity on `Dual <: Real` (§4.2) |
| ger / geru / gerc | **tagged** | `_ger_panel_cmplx!` `:2211`, per-column arm `:2364` (already-tagged axpy bodies) | `_pair_shuf`/`_pair_sgn`, the scalar tail |
| trmv | **rides the tagged gemv/axpy/dot** | drivers `_trmv_cmplx!` `:3917`, `_trmv_cmplx_blk!` `:4040` | nothing in SIMD; diagonal product is scalar generic `*` |
| trsv | **rides the tagged gemv/axpy/dot + own scalar reciprocal** | `_trsv_cmplx!` `:3971`, `_trsv_cmplx_blk!` `:4080` | the diagonal reciprocal (§4.1); no SIMD division anywhere |
| symv | **tagged hemv kernels with a `CJ` flag** — tier 2 | `_hemv_col_cmplx!` `:2675`, `_hemv_rect_cmplx!` `:2724`, driver `:2806` | fold + the diagonal (`real(A[j,j])` → the full dual) |
| hemv | **routes to symv** | — | §4.2 |
| spmv/tpmv/gbmv/… | **nothing** (out of scope; generic) | — | — |

### 2.1 gemvN — the tag needs one new shuffle form

`_gemv_n_ri_panel!` (`:1404-1470`) is column-outer: per column `cr = V2(αr·xr − αi·xi)`,
`ci = V2(αr·xi + αi·xr)·altv` with `altv = [+1,−1,…]` (the sign pre-folded so the epilogue is a plain add,
`:1409-1413`), then per row-vector `Pv = Σ av·cr`, `Qv = Σ av·ci`, `y += Pv + swap(Qv)`.

For dual: `cr = V2(αv·xv)` (the `−αi·xi` term is the ε² one and is dropped — a hoisted scalar, outside the
loop), `ci = V2(αv·xp + αp·xv)` unchanged, and the epilogue must add `a_v·ci` to the **partial** lane and
nothing to the value lane. Working through the lane algebra:

| form | `altv` | shuffle | value lane receives | hazard |
|---|---|---|---|---|
| complex | `[+1,−1]` | swap | `−a_i·ci` ✓ | — |
| dual, swap | `[+1, 0]` | swap | `0·(a_p·ci)` | **`0·Inf = NaN` on the value** when a partial is infinite (the hazard `dual.md` §dupEven exists to avoid) |
| dual, dupEven | `[0,+1]` | dupEven | `0` | partial lane gets `0·…` — wrong lane, ✗ |
| **dual, zero-unpack** | none | `shufflevector(zero(V2), Qv, (0,2W+0, 2,2W+2, …))` | exact `0` | none: no lane is ever multiplied by zero |

The zero-unpack is one two-source shuffle (`vunpcklpd ymm, zero, Qv` on AVX2 — same port class as the swap's
`vpermilpd`), and `altv` disappears for dual. It is a third pattern beyond `_pair_shuf`/`_pair_sgn`, so it
becomes `_pair_oddsel(ALG, V2, q)` (name to be argued): `:cplx` → `shufflevector(q, swp)`; `:dual` →
`shufflevector(zero, q, zpat)`. The masked half-width tail (`:1440-1462`) takes the same substitution.

**Prediction, not measurement:** the dual loop histogram equals the complex one modulo the shuffle mnemonic
plus possibly one hoisted `vxorpd` for the zero register. If LLVM rematerialises the zero *inside* the loop
(register pressure at `NC = 3W/2 = 6` on AVX2: 6 loads + 6 `cr` + 6 `ci` + `Pv`/`Qv` already exceed 16 ymm),
the histogram test fails — see §5.

### 2.2 gemvT — epilogue-only, the dot precedent

`_gemv_tc_block_cmplx!` (`:1728-1810`) accumulates `p_c = Σ a·x`, `q_c = Σ a·swap(x)` for NC columns and
folds `sr = pf[1] ∓ pf[2]`, `si = qf[1] ± qf[2]`. The loop is algebra-agnostic — this is exactly
`_dot_pair_simd`'s shape, whose tag is confined to the fold (`simd_kernels.jl:755-765`). Dual fold:
`value = pf[1]` (the `pf[2] = Σ a_p x_p` lane is ε² and is **not touched**, not even multiplied by zero),
`partial = qf[1] + qf[2]`. `CJ` is ignored for dual. The masked tail (`:1770-1783`) is unchanged.

Signature: the block takes `Ptr{Complex{T}}` and `α::Complex{T}` (`:1729-1731`); the tagged form takes the
pair type `P` (`Ptr{P}`, `α::P`) with the real type read at generation time (`fieldtype(P, 1)`, compile-time
per the trim rules). The epilogue `s = Complex(sr, si); α*s; muladd(β, yj, α*s)` becomes
`s = _mkpair(P, sr, si)` and generic scalar `*`/`+` on `P` — ForwardDiff's own scalar rules, so the dual
alpha/beta multiply is correct by construction.

### 2.3 ger — the axpy tag, verbatim

`A[:,j] += (α·y[j])·x`. Surviving lane products of `α·x_i·y_j` under ε² = 0: value `α_v x_v y_v`; partial
`α_v(x_v y_p + x_p y_v) + α_p x_v y_v` — three of the four scalar products, and `x_p·y_p` never forms. In the
panel kernel this is: hoisted `ay = α·y[j]` (scalar pair multiply, generic on `P`), then per element the
axpy `A += ay·x`, which `_ger_panel_cmplx!` already writes as `t = muladd(xv, V2(re ay), A);
A = muladd(swap(xv), [−im,+im,…], t)` (`:2237-2240`). The dual arm is `_pair_shuf` + `_pair_sgn`
(`simd_kernels.jl:65-67`) — identical constants to `_axpy_pair_wide!`. The scalar tail (`:2242-2252`) is
per-algebra. The per-column arm (`_ger_cmplx_percol!` `:2364`, and the `< np` remainder in `_ger_pdc_cj!`
`:2300-2307`) calls `_axpy_cmplx_simd!`/`_axpy_cmplx_cold!`, which are already forwarders to the tagged
bodies (`simd_kernels.jl:642, 683`) — they only need the tag threaded through. The residency ladder
(`_cger_cold_den`, `_ger_paneldrv_np`) transfers unchanged: identical bytes, identical streams.

### 2.4 trmv — nothing special

`_trmv_cmplx!` (`:3917-3968`) is a driver: column axpys (`_axpy_cmplx_simd!`) for `N`, column dots
(`_dot_cmplx_disp` `:3914`) for `T/C`, and one scalar diagonal product `xj * A[j,j]` per column. The blocked
form (`:4040`) adds `_tri_scat_cmplx!` = the gemvN ri driver (`:4031`) and `_tri_scatT_cmplx!` = the gemvT
driver (`:4033`). With gemvN/gemvT/axpy/dot tagged, trmv is the same driver over the pair type: the only
algebra-specific step is the scalar diagonal product, which is ForwardDiff's `*`. A dual diagonal changes
nothing structurally — it is one pair multiply per column, off the vector path.

### 2.5 trsv — division, answered

See §4.1. Structurally identical to trmv's driver plus one scalar reciprocal per column.

### 2.6 symv / hemv — tier 2

`_hemv_col_cmplx!` (`:2675-2703`) fuses the axpy `y += tmp·a` (swap + `[−tmpi,+tmpi]`) and the conj-dot
`s += conj(a)·x` (p/q accumulators, fold `sr = Σp_r + Σp_i`, `si = Σq_r − Σq_i`) reading `a` once;
`_hemv_rect_cmplx!` (`:2724`) is the NB-column panel version. The dual symv is the same fused body with
(i) the axpy half on the `_pair_shuf`/`_pair_sgn` constants, (ii) the dot half on the dual fold
(`value = Σp_even`, `partial = Σq_even + Σq_odd` — no conjugation), and (iii) the driver's diagonal
`real(A[j,j])` (`:2839, :2857`) replaced by the whole pair. (i)–(ii) are the two tags already proven; (iii)
is a `_pdiag(ALG, a)` in the driver. It is the most driver-invasive op on the level, and the least used in AD
code, which is why it is tier 2. Side observation, out of scope: the same `CJ` generalisation would give
**complex** `zsymv` a SIMD path for the first time (it is generic today, hence the 0.85 "parity" above).

## 3. Routing and the extension surface

* A type-level predicate is needed: `_pairalg` (`simd_kernels.jl:49`) is value-based on `DenseArray`, so a
  `view(A, I, J)` of a `Matrix{Dual}` misses it; the complex L2 predicates accept any `_strided1` matrix
  (`_l2c_ok` `:1297`). Add `_pairT(::Type)` (ext: `true` for `Dual{Tag,V<:BlasReal,1}`) and build
  `_l2pair_ok(A, x, y, incx, incy) = incx == incy == 1 && _pairT(eltype(A)) && eltype(x) === eltype(y) ===
  eltype(A) && _strided1(A) && _dense1(x) && _dense1(y)` — the exact mirror of `_l2c_ok`. Two more one-liners
  in the extension: `_palg(::Type{P})` (`Val(:cplx)` / `Val(:dual)`) and `_parts`/`_mkpair` methods for
  `Complex` so the generalised drivers read `αr, αi = _parts(α)` for both algebras.
* Entries gain one branch each, after the complex one: `_gemv!` (`:1904-1912`, `:1935-1939`), `_ger!`
  (`:2389`), `_trmv!` (`:4124`), `_trsv!` (`:4172`), `_symv!`/`_hemv!` (tier 2). In the main env `_pairT`
  is `false` for every type, so the branch is dead code and `--trim` never sees it — the BLAS-1 arrangement.
* No `reinterpret` anywhere. Pointers are taken as `Ptr{V}(pointer(A))` (raw pair buffer, layout asserted by
  the extension at load), never through a `Complex` type.

## 4. The design questions, explicitly

### 4.1 trsv: division by a dual

`(a + bε)/(c + dε) = a/c + (b/c − ad/c²)ε`. Where it occurs in the complex driver: **only on the diagonal**,
one scalar per column — `x[j] /= A[j,j]` at `:3996/:4003` (`N` forms) and `s / djj(j)` at `:4012/:4019`
(`T` forms); every off-diagonal step is an axpy or a dot. The complex driver hoists those divisions off the
substitution's dependency chain by precomputing reciprocals into `_TRSV_RCP64` (`:3960-3966`, 512 entries,
`_crecip` `:3967`) and then multiplying. Dual does the same with the dual reciprocal
`1/(c + dε) = (1/c) − (d/c²)ε`: `r = inv(c); _mkpair(P, r, −d·r·r)` — one real division, two multiplies —
then `x[j] *= rcp[j]` is the ordinary pair multiply. So:

* **Is there a clean SIMD form?** Not needed and not wanted: it is n scalar reciprocals in an O(n²) op, each
  independent (they pipeline), and the only thing on the critical path is the multiply, which is already the
  tagged scalar pair product. A vectorised reciprocal would save nothing measurable.
* **Hoisted per diagonal element?** Yes, exactly as complex: precomputed once per solve into the reciprocal
  buffer. The buffer is typed `ComplexF64`; for dual it is used as raw pair storage through `_parts`/`_mkpair`
  (two reals in, two reals out — no complex arithmetic touches it), or a `Ptr{V}` view of it. The `n > 512`
  fallthrough divides in place with ForwardDiff's `/`, which implements the rule above.
* **Is unit-diagonal the only case worth doing?** No. Non-unit costs one hoisted scalar reciprocal per column
  on top of unit, off the chain; both ride the same driver. There is no reason to ship only `diag='U'`.
* **trsm** avoids dual division entirely — see `dual_l3.md` §2.4 (`X_p = A_v⁻¹(B_p − A_p X_v)`).

### 4.2 hemv vs symv (and gemvC, gerc)

`Dual <: Real`, ForwardDiff defines no `conj`, so `conj(::Dual)` is Base's `conj(x::Real) = x`, and
`real(d) = d`. The generic `_hemv!` loop on a `Matrix{Dual}` therefore **already computes symv** — the
`conj(aij)` at `:2878` and `real(A[j,j])` at `:2880` are identities. Routing dual `hemv` onto the dual symv
kernel is the same decision as `dotc == dotu` in BLAS-1: the involution `a − bε` is wrong for AD (it would
zero `d‖x‖²/dx`), the identity is the only conjugation consistent with `Dual <: Real`, and erroring would
break code that is correct today. Same for `gemv(trans='C')` → the `T` path with `CJ=false`, and `gerc` →
`geru`. Julia's own `BLAS.hemv!` is only defined for complex element types; a real `Hermitian` is handled as
`Symmetric`. Decision proposed: **collapse, don't error**.

### 4.3 ger: which lane products survive

Answered in §2.3: three of four (`x_p·y_p` is ε²). The complex scaffolding tags because its inner update is
the axpy form (`:2237-2240`), whose dual constants already exist.

### 4.4 trmv with a dual diagonal

Answered in §2.4: rides the tagged gemv/axpy/dot; the diagonal is one scalar pair product per column.

## 5. Adversarial: the case against this design

* **gemvN's zero-unpack may cost a register the AVX2 `NC=6` arm does not have.** `_CGEMVN_NC_BIG = 3W/2`
  (`:1600`) is chosen for memory-level parallelism past L3/2; at `W=4` the loop already holds 6 A-streams and
  12 broadcasts. A live zero register is one more. If LLVM keeps it hoisted the histogram is equal modulo the
  shuffle mnemonic; if it spills or rematerialises `vxorpd` inside the loop, the histogram test fails and
  `spill_report` on the `:dual` instantiation shows it. Fallback: the dual gemvN drops the `NC=6` arm and
  ships `NC=4` above L3/2 (a few % on Zen3 by the table at `:1580-1590`, nothing on AVX-512 where 32 zmm
  hold it). The swap-based alternative is **not** a fallback: it puts `0·Inf` on the value lane, which the
  generic loop keeps finite — a correctness regression relative to today.
* **Generalising the drivers over the pair type is where complex can move.** The bodies are gated by
  byte-identity; the drivers (`_gemv_n_ri_run!`, `_gemv_tc_run!`, `_ger_pdc_cj!`, `_trmv_cmplx!`,
  `_trsv_cmplx!` and the blocked forms) change signature from `Complex{T}` to `P`, and `real(α)` becomes
  `_parts(α)`. Semantically identical, but register allocation may differ. The gate probe
  (`bench/probes/dual_l23_native.jl`, written, baseline not yet captured) covers every driver and entry as
  well as every body. If a driver moves by one instruction, that driver is copied for dual and the complex one
  stays untouched — the cost is ~30 duplicated lines per driver, not a kernel.
* **trsv's reciprocal buffer is a global** (`_TRSV_RCP64`, `:3960`): fine single-threaded, the same
  assumption complex already makes; the dual path inherits it, not worsens it.
* **The op being forced into a shape that does not fit is symv.** The fused hemv kernel was designed around
  conjugation and a real diagonal; the dual version needs a `CJ` flag in two kernels and a diagonal hook in
  the driver, and its dual/complex fold differs in **both** halves. It is the most likely place for the
  histogram test to show a real difference (the dot half's fold is outside the loop, but the axpy half's
  constants are inside). Which is why it is tier 2 and could be dropped without touching the rest.
* **What would detect a wrong tag?** (1) the ε²-leak test extended to gemv/ger/trmv/trsv: partials ~1e155
  so any `x_p·y_p`/`a_p·x_p` product overflows — value must stay finite, partial must equal the generic loop;
  (2) infinite-partial hygiene (the `0·Inf` case above) on gemvN specifically; (3) `ForwardDiff.jacobian`
  of `A*x`, `x*y'`, `A\x` against the generic path at awkward `m,n` (the shapes in the probe's §3).
* **If only half ships:** gemvN + gemvT. They are the two bodies with a real design question, they are the
  bandwidth-bound ops where the 4× matters most, and they unlock trmv/trsv's blocked forms for free (the
  unblocked `n ≤ _TRI_C_BLK_MIN = 256` on AVX2, `:3034`, needs only the BLAS-1 tags). ger is second (it is
  one constant substitution). symv is last.

## 6. Measurement plan (galen, `taskset -c 6`, Chairmarks median, same probe as the triage)

1. Baseline capture on the pre-edit tree: `dual_l23_native.jl` writes one normalised `code_native` per complex
   body/driver/entry (both `Float64`/`Float32`, every `NC`/`NP`/`HALF`/`CJ` arm the ladders can select).
2. After each op: byte-identity (part 1), loop histogram complex vs dual (part 2), awkward-shape correctness
   (part 3). Gate as in BLAS-1: **one instruction moved in complex ⇒ that op ships as a copy.**
3. Re-run the triage table; the bar is the BLAS-1 bar — equal time to the complex twin on identical bytes,
   op counts being identical for every op on this level (gemvN drops one hoisted scalar multiply per column,
   nothing in the loop).
4. `spill_report` on the dual gemvN `NC=6` instantiation before the first probe (§5, item 1).

## Implementation notes (2026-09-13, after the sparring round)

Shipped on branch `dual-l2l3`: gemvN, gemvT (and 'C' → 'T'), ger (gerc → geru), trmv, trsv. **symv is dropped
from this round** (sparring point 4): PureBLAS has no SIMD complex symv to tag either — `zsymv` runs the generic
loop (`_symv!`), so the 0.83–0.88 "parity" in the triage is two scalar loops. Complex symv is an ungated gap of
the same class `_scal_cmplx_simd!` was before `zscalc`; a symv kernel is separate work for both algebras.

### Measured (galen, Zen3, same probe and regime as the triage; median of 6 rounds)

Complex-twin time ÷ dual time on identical bytes, after tagging:

| op | n=32 | n=128 | n=512 | n=1024 | n=2048 | n=4096 | before (generic loop) |
|---|---|---|---|---|---|---|---|
| gemvN | 1.05 | 1.01 | 1.00 | 1.01 | 1.01 | 1.00 | 0.25 … 0.47 |
| gemvT | 1.00 | 1.01 | 0.99 | 1.00 | 1.01 | 1.02 | 0.31 … 0.47 |
| ger | 1.24 | 1.09 | 1.03 | 1.00 | 1.01 | 1.02 | 0.43 … 0.83 |
| trmv | 0.98 | 0.98 | 1.01 | 1.00 | 1.02 | 1.02 | 0.69 … 0.47 |
| trsv | 0.90 | 0.93 | 1.00 | 1.00 | 1.01 | 1.01 | 0.71 … 0.46 |

The bar (§5: equal time to the twin, op counts being identical) holds for gemvN/gemvT/trmv at every n and for
ger everywhere (it is *faster* in cache: the hoisted `α·y[j]` is one scalar multiply cheaper per column). trsv
is 10 % / 7 % behind at n = 32 / 128 and at parity from 512: the residual is the scalar diagonal step — one
real division plus a pair multiply per column against the complex reciprocal multiply — which is a larger share
of a small solve. Before the reciprocal hoist it read 0.79 / 0.84, so the hoist was worth 11 points.

### The gate: 94 of 102 complex instantiations byte-identical; the other 8 are not adjudicable

`bench/probes/dual_l23_native.jl` baselines every complex body, driver and entry on the pre-edit tree (100
specs, 2 real types) and diffs after each op. Every **body** — gemvN panel (nc 1/4/6, pf on/off), gemvT block
(4 arms × CJ), ger panel (np 2/4/8 × CJ × HALF) — and every driver and entry (`_gemv_n_ri_run!`,
`_gemv_n_ri_cmplx!`, `_gemv_tc_run!`, `_gemv_tc_cmplx!`, `_ger_pdc_cj!`, `_ger_cmplx_percol!`, `_ger_cmplx!`,
`_trmv_cmplx!`, `_trsv_cmplx!`, `_gemv!`, `_ger!`) is byte-identical, both types. The drivers are generic over
the pair type `P` (they read `_parts(α)`, tag the kernels with `_palg(P)`, build results with `_mkpair`) and
that generalisation moved nothing — the copy fallback was not needed.

The 8 that differ are `_trmv_cmplx_blk!`, `_trsv_cmplx_blk!`, `_trmv!`, `_trsv!` (× 2 types): same
instruction count, same opcode multiset, ~100 of 688 lines differing only in stack-slot offsets and register
names. **That drift reproduces on the unedited `a0f0a703` tree in a fresh process**
(`bench/probes/dual_l23_basecheck.jl`: a second worktree at the base commit, cold julia, compared against the
hot session's baselines — the same four functions drift, `hemv` and `_gemv_n_ri_run!` do not). LLVM's frame
layout of these large inlined drivers is not stable across processes, so byte-identity cannot adjudicate them;
`bench/probes/dual_l23_regnorm.jl` records the residual as register/slot renaming only. A first attempt to
"fix" it by copying the ladders for dual (per the ship-as-a-copy rule) changed nothing, which is what prompted
the base-tree check; the copies were reverted.

Two real complex regressions were caught by the gate during the work and fixed before any commit landed on
its own: the ger scalar tail had its operand association changed by the tag splice (`load + (a·b − c·d)` vs
`load + a·b − c·d`; +25 instructions, and a different rounding), and an early `_pair_rcp` one-liner failed to
parse under Revise and took `_trsv_cmplx!` with it. The probe folds callee serials (`j__f_1234`) on both
sides; without that fold every driver reads as "different" after a Revise.

### Histograms, complex vs dual main loop (part 2 of the probe)

* gemvN nc1 / nc4: EQUAL modulo shuffle class (`vshufpd` → `vunpcklpd`; F32 `vshufps` → `vpshufb`) — the
  zero-unpack lowers to ONE instruction as predicted. nc6 (F64, the AVX2 past-L3/2 arm) and nc12 (F32): the
  dual loop carries one extra `vxorpd` / `vmovdqa` — the zero register rematerialised inside the loop, exactly
  the §5 spill concern. It is a zeroing idiom (no execution uop on Zen3) and the measured n = 2048/4096 cells,
  which run that arm, sit at 1.00–1.01, so it was left alone rather than dropping the arm.
* gemvT: the loop is algebra-agnostic in source; the per-algebra epilogue changes register allocation and the
  loop shows ±`vmovapd` moves (F64 nc4 full-width: one FEWER for dual; F32 nc4 full: +6, nc8 half: −16).
  The default arms on both ISAs (nc4 half, nc8 half F64, nc2) are EQUAL. Measured parity at every n.
* ger: EQUAL modulo shuffle class on all 12 arms (`vshufpd`/`vshufps` → `vmovddup`/`vmovsldup`).

### Sparring point 3, answered by the lane algebra (and confirmed by the histogram)

`Pv = Σ a·[c_v,c_v]` already delivers `a_v·c_v` (value lane) and `a_p·c_v` (partial lane) in one FMA; the only
cross-lane term is `a_v·c_p` → partial lane. Applying `dupEven` to A (the proposal) costs one shuffle per
A-vector, i.e. NC per row-iteration; accumulating `Qv = Σ a·[c_p,c_p]` and moving its even lanes once per
row-iteration costs ONE shuffle regardless of NC — the same count the complex kernel pays, and the complex
kernel already works this way. Splitting x into planes does not remove it: the term that must change lanes is
A's value lane, which is interleaved by definition. The two-source zero-unpack is that one shuffle with no
lane ever multiplied by zero (an Inf in the discarded `Σ a_p·c_p` lane cannot reach the value), and LLVM picks
the ISA form itself from the one `shufflevector(zero, q, pat)` source; on AVX2 it is `vunpcklpd`. The AVX-512
lowering (a zero-masked permute) is unverified: galen is the only box this work may touch.

### trsv: the reciprocal IS hoisted, through the complex buffer

`1/(c + dε) = 1/c − (d/c²)ε` is stored as the two reals `(r, −d·r²)` in a `Complex` slot of the existing
512-entry buffer (`_pair_rcp`) and read back with `_mkpair` (`_rcp_mul`); no complex arithmetic touches it,
so the trap does not apply. Complex `_trsv_cmplx!` stayed byte-identical through the change (the complex
methods of both helpers inline to the previous expressions).

### Tests

`test/dual_tests.jl`: "Dual L2: gemv N/T/C, ger, trmv, trsv match the plane formulas" (15 shapes × 2 types,
every uplo/trans/diag, dual α and a zero-value β), "ε²-leak and infinite-partial hygiene + ForwardDiff
derivatives through the entries", and the `:checks` item "BLAS-2 dual strict contract" (`@test_noalloc` /
`@test_typestable` on the five entries). The BLAS-1 iamax item indexed past `n` on AVX2 (`4W = 16`; the tie
positions were fixed at 10/20/30) — found running the suite on galen, fixed in its own commit.
