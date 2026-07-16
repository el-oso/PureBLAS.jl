# req#8 constant classification — methodical plan

**Branch:** `req8-classification`. Scope: **real G3–G7 + cold bundle** (complex G1c/G2 deferred; trsm
consts parked). The reframe (2026-07-16): this is a **classification** campaign, not a "derive
everything" campaign. Every flagged literal is *already fleet-correct* (measured on Zen3/Zen4/Zen5);
the job is to decide, per const, which of three classes it is — and only *some* need a formula.

## The three classes

| Class | Meaning | Action | Fleet A/B? |
|-------|---------|--------|-----------|
| **SCALES** | Optimum tracks hardware (cache size / register count). A fixed number is wrong on an unbenchmarked cache. | Derive formula over detected consts; **validate it reproduces the fleet optima**. | Yes — derive + validate |
| **INVARIANT** | Optimum is a fixed algorithmic crossover / recursion base, hardware-independent. A formula would *introduce* error. | **Keep the literal**, document the measured evidence. | Only to confirm flatness (cheap, or already known) |
| **MISTUNED** | µarch ternary where one side was never measured → possibly wrong on that box. | Fix the wrong side (real bug, trsmR-style). Highest payoff. | Yes — this is where gate wins hide |

Guardrail (non-negotiable, [[kink-fixes-potrf-gemm]]): **a mistuned derived const is worse than a
literal** — it's wrong on the fleet *and* on unknown machines. Never ship a formula that hasn't
reproduced the fleet optima. When in doubt, the fleet-correct literal stays.

## Two-tier method (cheap first)

**Tier 1 — static classify (no fleet time).** For each const: read its use-site + physical role.
- Used as a cache-blocking size (tiles a matrix to fit L1/L2/L3, or a pack/panel width)? → **SCALES** candidate.
- A recursion base / small-n cutoff / algorithm crossover independent of cache? → **INVARIANT** candidate.
- A µarch ternary with far-apart values? → **MISTUNED** candidate.
Many already carry the intended formula *in-comment* — those SCALES ones are a **refactor-to-reproduce**
(apply the written formula, validate it hits the literal on the fleet).

**Tier 2 — fleet A/B (only SCALES-derive + MISTUNED).** Boost-locked (`fleet_freqlock.sh lock`),
single-thread, **pooled fresh-input ABBA** (never warm-micro — [[measure-controlled-ab-not-crossrun]]),
affected op at the sizes where the const bites, candidate values vs literal, all 3 boxes. Formula ships
only if it reproduces every box's optimum within noise.

INVARIANT consts skip Tier 2 (or get one confirming A/B if flatness is unproven) → document + done.

## Worklist (Tier-1 hypotheses — confirm/revise as executed)

Sequenced **payoff-first**: MISTUNED (gate wins) → SCALES-refactor (correctness on unknown HW) → document INVARIANT.

### Phase A — MISTUNED candidates (far-apart µarch ternaries; highest payoff)
| const | file:line | value | hypothesis | Tier-2 plan |
|-------|-----------|-------|-----------|-------------|
| ~~`_GEMVN_RB`~~ | level2.jl:83 | `64 : 448` | **NOT a Zen4 gate bug** — see verdict below | Zen3/Zen5 confirm → Phase B |
| ~~`_TRI_C_BLK_MIN`~~ | level2.jl:1327 | `256 : 1024` | **OUT OF SCOPE — complex** (tri unblocked threshold) | defer to complex batch |
| ~~`_GBMV_CONV_MAX`~~ | level2_banded.jl:17 | `20 : 48` | **INVARIANT-per-ISA** — measured datapath crossover (conv re-reads AB ~band/W×; comment: stable n=256…4096). Formula `band*≈f(W)` under-determined (2 pts), low payoff. | keep+document; revisit if a W-formula batch covers it |

**Phase A conclusion (2026-07-16): NO gate wins.** The only suspicious real ternary, `_GEMVN_RB`, is
correctly tuned on Zen4; the near-miss was a wrong-shape (tall M) artifact I caught. `_GBMV_CONV_MAX` is a
characterized per-ISA crossover; `_TRI_C_BLK_MIN` is complex. → Confirms campaign value is **Phase B**
(correct block-sizing on unbenchmarked HW), not gate wins.

**`_GEMVN_RB` Zen4 verdict (2026-07-16):** row-block/panel A/B at the **gated square shape** (forced each
path via `gemvn_rb` pref): row-block wins n≤64 (+20-33%), **flat tie 96→448**, panel wins ≥512 (row-block
collapses at the po2-512 aliasing cliff, 0.41× at n=2048). Literal `448` sits at the knee → correct on
Zen4. Value is **insensitive in [64,511]** (flat band) → 64 and 448 tie on Zen4-square. NOT mistuned; no
Zen4 gate win. TRAP: a tall M=2048 test said "panel always wins" (row-block's y can't be register-resident
there) — the wrong, ungated shape. → SCALES-vs-INVARIANT deferred to Phase B fleet batch (needs Zen3/Zen5).

### Phase B — SCALES / refactor-to-reproduce (formula often already in-comment)
| const | file:line | value | hypothesis | Tier-2 plan |
|-------|-----------|-------|-----------|-------------|
| ⛔`_TRMM_RPACK` | level3.jl:14 | `448` (kept) | **DERIVATION FALSIFIED** — the "= gemm `_GEMM_UNPACK_MAX`" hypothesis regressed Zen3 (see below); reverted to literal 448 | galen A/B: 96 slower 3–18% |
| ✅`_CHOL_RL_MAX` | lapack.jl:524 | `128:224` | **PARTIAL DERIVE** — AVX2 `√(_L2_BYTES/8)·7/8` (=224 galen); W=8 `128` halving base kept flat (µarch-invariant) | no-op all 3 boxes |
| ~~`_TRMM_RPANEL`~~ | level3.jl:9 | `512` | **soft width** ("keep off-diag gemm fat") — no clean cache/reg formula; ungated | keep + document; revisit only on side-R gap |

**Phase B remainder = a GATED NB-sweep campaign (B2).** The clean no-op refactors are done. What's left are
**gated block sizes** that would change fleet gate numbers if re-derived → each needs a real multi-box NB
sweep (getrf/geqrf-style), NOT a guessed formula. These are the genuine "derive→fleet-validate" targets and
the measurement-heavy part:
| const | file:line | value | note |
|-------|-----------|-------|------|
| ✅`_CHOL_BLOCK` | lapack.jl:194 | `128` | **INVARIANT — measured NB-insensitive** (Zen4 sweep 96/128/192/256 → potrf n=1024…4096 within 0.5%). Large-n residual is structural panel-major, not NB. Kept flat + documented. |
| `_L3_NB` | workspace.jl:18 | `128` | NB×NB trmm/syrk scratch. Same 128 class as `_CHOL_BLOCK`; insensitivity inferred but `_TRMM_BASE` gates side-L → light confirm before closing. |
| `_TRMM_BASE` | level3.jl:8 | `128` | = `_L3_NB` cap; side-L trmm base (GATED). Confirm with a small NB check. |
| ✅`_LU_NB` | lu.jl:6 | `48`/÷8/`128` | **INVARIANT — fleet nb-sweep (2026-07-16), derivation FALSIFIED.** Full nb×n grid Zen4+Zen3: FLAT band (n≥384 nb∈[96,192] within ~1%); FLOOR 48 sharply validated (n=256 wants 48 both boxes, nb64 +2.4/+4.4%); CAP/slope in-band. Curve is parity-BUMPY (mult-64 win), so the derived `_l1_block(F64,_MR·W)` (128 Zen4/168 Zen3) is a Zen3 TROUGH — 168 WORSE than 128 (+0.4–2.6%). True large-n opt is 64-aligned + per-µarch (Zen4 256/Zen3 192), no clean formula, only ~0.5–1.5% over 128. Literals kept + documented. |
| `_GETF2_BASE` | lu.jl:25 | `16` | self-flagged "derive from store-BW/L1"; `_CGETF2_BASE` scales it by sizeof. |
| `_BRD_NB`/`_BT_NB` | svd.jl | (lit) | bidiag block; low priority (SVD). |

### Phase C — INVARIANT ✅ DONE (2026-07-16): all recursion/algorithm crossovers, validated-by-gate
**Classification principle:** every const below is a **recursion base / algorithm-switch crossover**, not a
cache-residency BLOCK size — its optimum is set by base-kernel-vs-recursion-overhead balance (a broad flat
region, µarch-weak), NOT by how much fits in L1/L2. Each also **feeds a GATED op that currently passes**
(syrk/potrf/gemm/trsv/trmv/getrf/gesvd all gate in plots.jl on the fleet) → a mistuned base would show as a
gate miss, so they are **validated-by-gate**. Per req#8(b), a literal is CORRECT for these (a residency
formula would add spurious variation — as the `_TRMM_RPACK` and `_LU_NB` derivations proved by getting
FALSIFIED). Literals retained + documented; no fleet re-measure needed.
| const | file:line | value | verdict |
|-------|-----------|-------|---------|
**⚠️ FABLE REVIEW (2026-07-16) corrected 4 over-claims below — "validated-by-gate" was stamped without
checking, per const, that the GATED path executes the const. Corrections applied; owed A/Bs pending (B3).**
| `_CHOL_STH`/`_CHOL_SB` | lapack.jl:201-2 | `16`/`32` | ✅ INVARIANT (f622318; Zen4 A/B 16 beats 32) |
| `_CHOL_FAER_BASE` | **lapack.jl:532** | `_CHOLW==8?1024:_CHOL_RL_MAX` | ✅ INVARIANT — real f64 potrf faer base (NOT level3/cpotrf; my earlier row was wrong on file+mechanism). W=8 side = bare `1024` literal, executed by `_potrf_f64_lower!` (gated) → validated-by-gate; AVX2 side rides `_CHOL_RL_MAX`. |
| `_SYRK_BASE` | level3.jl:2406 | `48` | ✅ INVARIANT (diagonal scalar-base vs gemm-recurse crossover; real gate coverage n=8/32 + every recursion diagonal) |
| `_GETF2_BASE` | lu.jl:33 | `16` | ✅ INVARIANT (store-traffic rank-1↔BLAS-3-split crossover; executed at all gated getrf sizes; 24→16 measured; `_CGETF2_BASE`=sizeof-derived) |
| ✅`_TRI_NB` | level2.jl:1319 | `min(64,√(L1/8))` | **B3 DONE — genuine SCALES, L1-residency clamp shipped.** trmv/trsv N+T NB∈{32,64,128} A/B'd Zen4+Zen3: NB=128 REGRESSES mid-n 3–8% on Zen4 (128²·8=128KB SPILLS the 32KB L1) while 32/64 tie ≤2% → the L1-residency bound is REAL (Fable's `64=√(L1/8)` was NOT coincidence). Derived `_TRI_NB = clamp(_round_dn(isqrt(_L1_BYTES÷8),16),16,64)` (fleet 32K→√4096=64 EXACT → no-op; ≤16K-L1 shrinks). Prevents the 128-spill on any box + protects small-L1. Suite 7931/7932. |
| ✅`_TRI_T_UNB` | level2.jl:1326 | `1024`→**`512`** | **B3 DONE — MISTUNED literal corrected (1024→512), + prior "x-restream" rationale FALSIFIED.** Same-process interleaved A/B (Zen4 W=8 + Zen3 W=4, boost-locked): blocked path (offloads O(n²) off-diag to gemv-T) crosses over at n≈512–768 on BOTH boxes — unblocked wins ≤3% at n≤512, blocked wins 6–40% at n≥768. Nearly FLAT across ISA width (512 vs ~640) ⇒ INVARIANT crossover, literal not formula. The old 1024 forced the SLOWER unblocked path at n=768/1024; at n=1024 unblocked is a gate **MISS on both** (Zen4 1.03×, Zen3 1.09× OB) that blocking FIXES (→0.95/0.87×). 512 keeps the small-n unblocked win + blocks the loss region. `tri_t_unb` pref. |
| ⚠️`_POTRF_BASE` | lapack.jl:11 | `512` | **FALSE PROVENANCE (corrected)** — the gated potrf (f64-lower, `_potrf_f64_lower!`) NEVER reads it; `_POTRF_BASE` governs only UNGATED F32/Dual/complex-upper/non-strided (lapack.jl:920). NOT validated by the BLASFEO campaign (that was f64-lower = `_CHOL_BLOCK`). base=512 ⇒ a 2MB unblocked potf2 block on untimed paths. → INVARIANT-for-correctness (AD/F32 fallback), but unvalidated for speed. |
| ⚠️`_GEMM_TINY` | gemm.jl:2034 | `6` | **UNGATED + immaterial** (fires only max-dim≤6; smallest gated size n=8). Honest label = "unvalidated + immaterial", not "validated-by-gate". Dispatch safe (`m≤_vwidth` guard). Mild ≈W dependence. |
| ✅`_L3_NB`/`_TRMM_BASE` | workspace.jl:18 / level3.jl:8 | `min(128,√(L2/32))` | **B3 DONE — measured flat + no-op L2 clamp shipped.** trmm side-L NB∈{96,128,192} A/B'd 2-box (Zen4+Zen3): FLAT (within noise every size, PB≥OB throughout) → base is an algorithm crossover, doesn't grow with L2. But the 128² scratch IS an L2 tile → derived `_L3_NB = clamp(_round_dn(isqrt(_L2_BYTES÷32),16),16,128)` (galen 512K→128 EXACT, Zen4/Zen5 1M→cap 128 → no-op fleet-wide; ≤256K-L2 shrinks to a fitting tile). `_TRMM_BASE=_L3_NB` coupled (scratch-overflow constraint). Suite 7931/7932. Zen5 backfill still owed. |
| `_BRD_NB`/`_BT_NB` | svd.jl:279/281 | `16`/`32` | ✅ INVARIANT (bidiag/back-transform panel; measured optimal n=256–2048) |
| `_SVD_DC_CROSS`/`_DC_THRESHOLD` | svd.jl:283 / svd_dc.jl:711 | `96`/`64` | ✅ INVARIANT (algorithm/recursion crossover; 96 measured boost-locked — [[gesvd-dc-crossover-and-lbt]]) |

**`_TRMM_RPACK` status (2026-07-16) — DERIVATION FALSIFIED, reverted to literal 448:** the hypothesis was
`_TRMM_RPACK = _GEMM_UNPACK_MAX` (=2·_acc_cap, "same register-capacity crossover as gemm — identical
microkernel"). Byte-identical on Zen4/Zen5 (both 448) but changed Zen3 448→96. Two-box direct-vs-packed
crossover sweep (boost-locked, pure rpack=100000 vs 1) REFUTED it and mapped the real behavior:
- `_TRMM_RPACK` is a pure path selector (n≤thr → direct/unpacked, n>thr → packed); nothing else reads it → clean A/B.
- The crossover is **µarch-dependent and SOFT**: packed decisively wins only at large n (Zen3 n≥512, up to
  17% @512; Zen4 n≥1536, ~3-6%), with a wide tied band below and DIRECT winning the small/mid band on BOTH
  boxes. On Zen4, direct GATES at n=256/384 (0.93/0.96 PB/OB) where packed MISSES (1.00/1.01).
- ∴ 96 regresses BOTH boxes (packs the direct-favoring small/mid band; Zen4 256/384 ~5-7% + below gate,
  Zen3 128-448 wash-to-worse) and gains nothing large-n (both pack there). 448 keeps small/mid on the
  winning direct path AND packs the large-n win region → **fleet-best of the tested values**.
Per req#8(b) (a formula must reproduce the fleet optimum before it ships) the register-cap derivation is
rejected; literal 448 kept + documented in-code. NOTE: the crossover appears to scale UP with cache (Zen3
L2=512K→~500, Zen4 L2=1M→~1200) — a candidate FUTURE SCALES derivation, but too soft/noisy to pin without a
dedicated 2-box crossover campaign; a guessed formula would be worse than 448. **The method worked: a
plausible mechanism ("identical microkernel") was a HYPOTHESIS, measurement killed it — and the measurement
also corrected an early single-shot artifact (a spurious "n=160 +18%" that a wider ABBA sweep showed was noise).**

## Ship discipline
Per-const small commits on this branch. Update this table's hypothesis→verdict as each clears. Do NOT
batch-formula-ize; each SCALES/MISTUNED const clears its own fleet A/B before shipping. Merge to master
only when the phase is done + suite green (`Pkg.test()`, [[pureblas-test-invocation]]).
