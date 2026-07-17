# Gate-gap closing campaign — PB ≥ max(OpenBLAS, AOCL) on the whole fleet

**Goal:** drive every (op, size, µarch) to `median PB/max(OB,AOCL) ≥ 1.0`. **Priority: `potrf` and the
kernels supernodal sparse Cholesky calls** (dense diagonal factor + off-diagonal `trsm`/`trsmR` panel
solve + `syrk`/`gemm` Schur update) — fast on **both AVX2 (Zen3) and AVX-512 (Zen4/Zen5), all sizes**.

## Fleet & method (definition of done for every item)
- Boxes: **wintermute Zen4/AVX-512**, **galen Zen3/AVX2**, **neuromancer Zen5/AVX-512**.
- Per fix: **diagnose** (decompose + roofline, don't guess the bottleneck) → **fix** → **boost-locked A/B
  on all 3 boxes vs BOTH OB and AOCL** (`sudo bench/fleet_freqlock.sh lock`) → **suite green** → merge.
  One op at a time; re-measure the touched op into the caches after each fix.
- Any tuning const change obeys req#8 (derive-or-documented-INVARIANT; validate on the fleet).
- Baseline snapshot: `bench/plots_data_*.txt` (+`_aocl`), commit 592dca4. Re-extract misses with
  `scratchpad/gate_misses.jl` after each merge.

## Miss inventory (median ratio < 1.0; `ob`=loses to OpenBLAS [same silicon → hardest], `ao`=loses to AOCL)

Snapshot totals: **194 miss-cells (94 real / 100 complex).** Tiered by severity below. "worst" = worst
median ratio across the fleet for that op. Most 0.97–1.00 vs-AOCL cells are last-mile/noise; the campaign
targets the < 0.96 gaps first, and treats vs-OB losses as higher priority than vs-AOCL.

---

## P0 — POTRF for supernodes (THE priority)

The plotted **F64-lower** potrf gates (only miss: n=2048 Zen5 vs AOCL 0.98). But "potrf fast for all sizes"
has gaps the main plot doesn't show, plus its building blocks miss. Sub-items:

| id | item | evidence | status |
|----|------|----------|--------|
| P0.1 | **F64-UPPER po2 aliasing** | `_potrf_upper!` trailing syrk/trsm read power-of-2 lda → cache-set conflict misses. Zen4 n=256 **1.64×**, n=512 **1.50×** slower vs OB; padding lda +8 → 1.06 / **0.74** (PB beats OB unaliased). Fleet-confirmed pattern. | task #77, diagnosed |
| P0.2 | **F32 potrf Zen3** | generic recursion sits **1.24–1.78× slower** than OB at n=192–1024 on Zen3 even at optimal base=32 (AVX-512 gates 0.80–1.07). AVX2 recursion kernel gap. | found this session |
| P0.3 | **potrf small-n (n=8/32/128)** | lower gates; UPPER n=128 = 1.78× (overhead, not aliasing). Supernode diagonal blocks are often small → verify all uplo/precision small-n gate. | to measure |
| P0.4 | **potrf n=2048 Zen5** | 0.98 vs AOCL (last-mile). | low |
| P0.5 | **zpotrf n=2048** | 1.00 vs AOCL Zen4/Zen5 (borderline). | low |

Building blocks feeding supernodal potrf are P1 below (trsmR, syrk, trsm, gemm) — **P0 and P1 are one
campaign for the supernodal goal**; sequence P0.1→P1(trsmR,syrk)→P0.2→P0.3.

## P1 — Real L3 building blocks (supernodal + general), by severity

| op | miss-cells | worst | vs-OB cells | boxes | notes |
|----|-----------|-------|-------------|-------|-------|
| **trsmR** | 7 | **0.85** (n=128 Zen3) | 2 | all 3 | side-R panel solve; loses to AOCL n=256–4096 all boxes. The supernode off-diagonal solve. TOP L3 gap. [[trsm-rowlane-design]] |
| **trmm** | 6 | **0.82** (n=8) | 4 | all 3 | small-n overhead + n=1024–4096 Zen4 vs OB 0.96–0.97 |
| **trsm** | 6 | 0.92 (n=512 Zen4) | 0 | all 3 | side-L; vs AOCL. [[trsm-sideL-zen4-campaign]] |
| **syrk** | 5 | **0.88** (n=128 Zen5) | 2 | all 3 | Schur update; n=128 Zen4 0.91/Zen5 0.88, n=2048/4096 Zen4 vs OB 0.95–0.97 |
| **symm** | 5 | 0.92 | 4 | all 3 | vs OB at several sizes |
| **syr2k** | 5 | 0.96 | 5 | all 3 | mostly marginal vs OB/AOCL |
| **gemm** | 3 | 0.97 | 0 | all 3 | n=128–512 vs AOCL only, last-mile |

## P2 — Real L2

| op | cells | worst | vs-OB | notes |
|----|-------|-------|-------|-------|
| **ger** | 4 | **0.74** | 5 | worst real L2 gap; [[fleet-gate-snapshot-locked]] Zen5 regression |
| **gemvT** | 5 | **0.75** (n=256 Zen3) | 3 | |
| **trmv** | 6 | 0.87 | 0 | rides gemvN panel; vs AOCL |
| **trsv** | 5 | 0.87 | 0 | vs AOCL |
| **gemvN** | 5 | 0.95 | 4 | [[gemvn-trmm-residuals-falsified]] — several levers already killed; Zen5 DRAM-bound residual |
| symv | 2 | 0.96 | 0 | marginal |

## P3 — Real L1 (mostly bandwidth-bound, marginal vs OB)

| op | cells | worst | vs-OB | notes |
|----|-------|-------|-------|-------|
| **iamax** | 6 | **0.85** (n=1000) | 0 | vs AOCL; SIMD idamax small-n |
| scal | 5 | 0.91 | 8 | vs OB, small-n |
| axpy/dot/asum | 4–5 | 0.97–0.99 | 4–9 | near-parity, likely noise floor |

## P4 — Complex (lower priority for the supernodal goal, but has the single worst miss)

| op | cells | worst | notes |
|----|-------|-------|-------|
| **ztrsm** | 5 | **0.64** | worst single miss in the fleet; complex small-n trsm. [[pureblas-ztrsmr-direct-base]] |
| ztrmmR | 6 | 0.81 | |
| ztrmm / zgemm | 6 / 4 | 0.84 | |
| zaxpy / zscal | 5 / 5 | 0.87 / 0.96 | complex L1 [[complex-l1-consolidation]] |
| zsymm/zher2k/zsyr2k/zhemm/zsyrk/ztrsmR/zgeru/zgemvT/zgemvN/zgemvC/... | 3–6 | 0.91–0.98 | complex L2/L3 batch |

---

## Execution order (one op at a time, fleet-validated each)

1. **P0.1 F64-upper po2 aliasing** — the concrete supernodal blocker; fix = pad/pack trailing operands in
   `_potrf_upper!` (padded-lda scratch is lower-risk than touching gated syrk/trsm dispatch). Guard: must
   NOT regress the gated square-shape syrk/trsm.
2. **P1 trsmR** then **P1 syrk** — the two building blocks a supernodal factor spends the most time in;
   biggest real L3 gaps (0.85 / 0.88) and both feed potrf. Decompose vs AOCL's packed panel kernels.
3. **P0.2 F32 potrf Zen3** — AVX2 recursion kernel gap; likely shares the trsm/syrk fix above.
4. **P0.3 potrf small-n** across uplo/precision (supernode diagonal blocks).
5. **P1 trsm / trmm** — remaining L3.
6. **P2 ger / gemvT** — worst L2 gaps (feed nothing supernodal directly; do after L3).
7. **P3 L1 iamax/scal**; **P4 complex** (ztrsm 0.64 first) — last, unless a shared kernel fix from P1
   lands them for free.

**Track progress here** (this file), same discipline as `req8_classification.md`: per-op commit, verdict
updated as each clears, boost-locked fleet A/B evidence in the commit. Re-run `scratchpad/gate_misses.jl`
after each merge to confirm the cell closed on ALL THREE boxes and nothing regressed.
