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
| `_GEMVN_RB` | level2.jl:83 | `64 : 448` | MISTUNED? (comment says L2-derived but AVX2 64 is a measured crossover) | gemv-N sweep, both RB values, 3 boxes |
| `_TRI_C_BLK_MIN` | level2.jl:1327 | `256 : 1024` | MISTUNED? | trtri/tri-solve sweep |
| `_GBMV_CONV_MAX` | level2_banded.jl:17 | `20 : 48` | MISTUNED? | gbmv sweep at the crossover |

### Phase B — SCALES / refactor-to-reproduce (formula often already in-comment)
| const | file:line | value | hypothesis | Tier-2 plan |
|-------|-----------|-------|-----------|-------------|
| `_TRMM_RPACK` | level3.jl:14 | `448` | SCALES — **= `_at_gemm_unpack_max` (register-capacity, G1 lesson)** | reuse derived `_acc_cap`; A/B trmm-R |
| `_TRMM_RPANEL` | level3.jl:9 | `512` | SCALES (pack panel) | pair with RPACK |
| `_CHOL_RL_MAX` | level3.jl:520 | (lit) | SCALES — formula in-comment `√(_L2/8)·7/8` | apply + validate potrf |
| `_CHOL_BLOCK` | lapack.jl:194 | `128` | SCALES (L2) — feeds already-derived `_CHOL_MC` | A/B potrf; may share NB with `_L3_NB` |
| `_L3_NB` | workspace.jl:18 | `128` | SCALES (L2/L3) — feeds `_TRMM_BASE` + scratch | shared-NB A/B with `_CHOL_BLOCK` |
| `_TRMM_BASE` | level3.jl:8 | `128` | SCALES (follows `_L3_NB`) | trmm A/B |
| `_LU_NB` | lu.jl:6 | `48` | SCALES **or** MISTUNED (self-flagged; getrf wanted nb(n) growth) | getrf nb sweep 3 boxes |
| `_BRD_NB` / `_BT_NB` | svd.jl | (lit) | SCALES (bidiag L2 block) | gebrd A/B (low priority — SVD) |

### Phase C — INVARIANT (document + done; confirming A/B only if flatness unproven)
| const | file:line | value | hypothesis | note |
|-------|-----------|-------|-----------|------|
| `_CHOL_STH`/`_CHOL_SB` | lapack.jl:201-2 | `16`/`32` | **INVARIANT ✅ DONE** (f622318) | Zen4 A/B: 16 beats 32 |
| `_CHOL_FAER_BASE` | level3.jl:521 | (lit) | INVARIANT (recursion base) | confirm flat |
| `_SYRK_BASE` | level3.jl:2395 | `48` | INVARIANT (small-n base) | confirm flat |
| `_GETF2_BASE` | lu.jl:25 | `16` | INVARIANT (base; self-flagged) | `_CGETF2_BASE` scales by sizeof — mirror if not flat |
| `_TRI_NB` | level2.jl:1319 | `64` | INVARIANT? | confirm |
| `_TRI_T_UNB` | level2.jl:1320 | `1024` | INVARIANT (unblocked-max) | confirm |
| `_POTRF_BASE` | (cold) | `512` | INVARIANT (recursion base) | document |
| `_GEMM_TINY` | (cold) | `6` | INVARIANT (flop crossover) | document |
| `_SVD_DC_CROSS`/`_DC_THRESHOLD` | svd.jl | (lit) | INVARIANT (algorithm crossover; 96 was measured — [[gesvd-dc-crossover-and-lbt]]) | document |

## Ship discipline
Per-const small commits on this branch. Update this table's hypothesis→verdict as each clears. Do NOT
batch-formula-ize; each SCALES/MISTUNED const clears its own fleet A/B before shipping. Merge to master
only when the phase is done + suite green (`Pkg.test()`, [[pureblas-test-invocation]]).
