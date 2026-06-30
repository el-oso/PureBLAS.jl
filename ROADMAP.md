# PureBLAS.jl — Roadmap & Status

Canonical status + next steps for this multi-session project. Update this file as milestones land.

## M1 — BLAS Level 1 vertical slice (IN PROGRESS)

Goal: prove the whole pipeline cheaply on bandwidth-bound BLAS-1 (no GEMM perf risk, no Fortran
char/string ABI). All four element types via generic `T<:Number` kernels.

Done & verified (426/426 tests passing as of 2026-06-28):
- [x] Package scaffold mirroring PureFFT (Project.toml, MIT, .gitignore).
- [x] Low-level kernels `(n,…,inc)`: copy, swap, scal, axpy, dot/dotu/dotc, nrm2 (lassq scaled),
      asum, iamax — generic scalar + SIMD.jl fast path (real, unit-stride, dense).
- [x] SIMD width auto-detected (CpuId `simdbytes`), const-folded/trim-safe, Preferences override.
- [x] Mode 2 native API (`SIMDBackend` + bare API) — AD-traceable (ForwardDiff verified).
- [x] TypeContracts `AbstractBLAS1` interface.
- [x] `@ccallable` ILP64 ABI symbols (cabi.jl) for all safe-return ops (void/real/int, s/d/c/z).
- [x] `lbt.jl` activate/deactivate.
- [x] ReTestItems suite: correctness vs OpenBLAS (s/d/c/z, contiguous + strided + empty), native
      API, AD smoke, StrictMode dogfood (typestable/noalloc/trim-safe).

- [x] `juliac/build.jl` → `libpureblas.so` (juliac --trim=safe --compile-ccallable, ~2.1 MB);
      all 30 BLAS-1 ILP64 symbols exported (verified `nm -D`).
- [x] Mode-1 validation from a **non-Julia (C) host** (`juliac/ctest.c`): dlopen + call
      daxpy_64_/dnrm2_64_ → correct results. The .so self-inits its embedded Julia runtime.

- [x] TrimCheck `@validate` on the 9 `@ccallable` entry points (test/trim_tests.jl) — all trim-safe.
- [x] README, DocumenterVitepress docs (Home/Guide/Design), CI.yml + docs.yml.
- [x] **Performance gate MET** (`bench/bench_level1.jl`, Float64, single-thread, interleaved/
      drift-robust, Zen4/wintermute, core-pinned): **0/24 op×size below 0.96×, min 0.977×, geomean
      1.41×** over n ∈ {1e3,1e4,1e5,1e6}. nrm2 4–8× (OpenBLAS dnrm2 is the slow scaled algo);
      scal/copy/asum ≥ parity; axpy/dot at parity. Implementation: reductions use 4 accumulators
      (latency-bound otherwise); elementwise kernels 4-way unrolled; nrm2 = SIMD sum-of-squares with
      scaled-lassq fallback on overflow/underflow.
- [ ] Re-run the gate per-machine on the rest of the fleet (Zen3 AVX2, Zen5 native-AVX512, M5 ARM).
- [ ] (optional) `benchmark/` PkgBenchmark suite + JSON-saving `bench/` pipeline like PureFFT.

### ⚠️ Key finding — LBT live-forward is blocked (juliac limitation, not a PureBLAS bug)

`BLAS.lbt_forward(libpureblas.so)` from inside a running Julia process **aborts**: LBT's interface
autodetection calls a probe symbol (`isamax_64_`), whose juliac wrapper runs
`ijl_autoinit_and_adopt_thread` and **double-initializes the shared libjulia** → `signal 6`.
A juliac-trimmed library embeds the Julia runtime and is meant to be loaded by a *non-Julia* host
(which it self-inits — see ctest.c), so it cannot currently be forwarded into a live Julia session.

Consequences / decisions:
- **In-Julia replacement = Mode 2 (native API / pkgimage)**, which is also the AD-enabling path —
  this is the primary way to use PureBLAS inside Julia. Trim-compatibility (the other reason for the
  .so) is independently proven by the successful build + symbol export + C-host run.
- The .so is the artifact for **non-Julia consumers** (C/C++/Rust calling BLAS).
- To make LBT forwarding work later: needs a juliac mode that initializes against the host runtime
  instead of embedding/auto-initing its own (upstream Julia work), OR a runtime-free codegen path.
  Track upstream; revisit in M5 (multi-ISA dispatch) / when juliac matures.

Open risks: complex-dot return ABI (deferred to M2); AVX-512 on Zen4 is double-pumped (tune via
Preferences knob).

## M2 — flagship `dgemm` (IN PROGRESS)

BLIS 5-loop (mc/nc/kc/mr/nr blocking + packing), register-blocked AVX-512 micro-kernel via SIMD.jl,
≥0.96× OpenBLAS.

> Cross-session knowledge hub: [`../kb/findings/pureblas-gemm-performance.md`](../kb/findings/pureblas-gemm-performance.md) (perf diagnosis + disproven ideas) and [`../kb/findings/juliac-trim-lbt-limitation.md`](../kb/findings/juliac-trim-lbt-limitation.md).

Done & verified (2026-06-28):
- [x] `src/gemm.jl`: `gemm!`/`gemm` native API. Blocked path for real (Float32/Float64, unit
      column-stride C); generic triple-loop fallback for complex / Dual / strided C (AD-traceable).
- [x] `@generated` register-blocked microkernel (mr×nr straight-line, runtime k-loop; **2×8 = 16
      accumulators is the Zen4 sweet spot** — 3×8=24 spilled & regressed, 2×6 left registers idle).
      Zero-padded packing → no edge microkernel needed (scalar edge kernel for partial tiles only).
      alpha folded into A-pack; beta applied up front (beta=0 ignores NaN per BLAS).
- [x] Correctness: 1167 GEMM cases vs OpenBLAS (all trans combos N/T/C, alpha/beta, edge sizes,
      complex, beta=0 NaN semantics, allocating gemm, ForwardDiff AD). Full suite 1610/1610.

- [x] StrictMode audit of GEMM hot paths (microkernel typestable/noalloc/trim-safe; generic path
      typestable/noalloc) — test/strictmode_tests.jl. Full suite 1611/1611.
- [x] Reusable packing scratch (`_gemm_scratch`, type-keyed, grown on demand) — removes the per-call
      ~MB malloc. **Biggest perf win: geomean 0.77×→0.92×.**

- [x] **Unpacked path** (BLASFEO-style size dispatch, ref arXiv:1902.08115): for tA='N' matrices with
      max(m,n,k) ≤ `_GEMM_UNPACK_MAX` (now **448**; started at 96) skip packing and run the microkernel
      directly on column-major data (`_gemm_unpacked!`, `_microkernel_unpacked!`). StrictMode-audited.
      Beats the blocked path while A fits ~L2 — lifted n=64..448 to ~parity/above.

Perf (Zen4/wintermute, Float64, single-thread, interleaved, `bench/bench_gemm.jl`): OpenBLAS ~45
GFLOP/s; PureBLAS **geomean 0.999× across n=16..4096 — parity, beats OpenBLAS on most small/medium
sizes** (n=16: 1.19×, n=128: 1.19×, n=150: 1.07×, n=200: 1.08×, n=256: 1.04×); large n 512=0.99×,
1024=0.99×, 2048/4096=0.97–0.98×; only 4/20 below 0.96× (n=33 0.84×, n=57 0.89×, n=96/100 ~0.95×).
Config: MR=2,NR=8 (16 acc), MC=144, NC=2040, KC=256, **unpacked for max(m,n,k)≤448**, C-tile prefetch
+ SIMD pack_A. Wins (biggest first): unpack threshold 96→448 (unpacked beats blocked while A fits L2);
**vectorized masked blocked edge** `_microkernel_masked!` (scalar edge had tanked non-multiples:
n=100 0.40→0.94×); **SIMD pack_A** `_pack_A_simd!` (large-n 0.95→0.98–1.02×); β=0 column-overlap
(+hybrid for n mod nr==1: n=17 0.74→0.97×); beta-folding; live-row-vector dispatch; scratch reuse
(0.77→0.92×); KC=256 + C-prefetch. Driver steady-state alloc-free (runtime `@allocated`; kernels
`@assert_noalloc` via StrictMode `:full`/AllocCheck). Tried & reverted: 3×8 tile (0.56×), A prefetch
(k-loop & next-panel), Val(nre) edge dispatch, clamp/guard edge.

### ⚠️ Large-n diagnosis CORRECTED (a "don't-guess-check" lesson)
I asserted large-n was "L2 A-feed bandwidth-bound" — that was WRONG and is RETRACTED. Measured
decomposition: the **macrokernel runs at 47.9 GFLOP/s** (in-context compute ≈ the isolated kernel,
above OpenBLAS; implied L2 read rate only ~24 GB/s ≪ Zen4 L2 ~280 GB/s → **not bandwidth-bound**).
The gap was **scalar packing** (pack_A 8.6 GB/s); **SIMD pack_A fixed it → large-n 0.97–1.02×.**
Lesson now a global rule: don't name a bottleneck without measuring the decomposition. Full detail in
the kb finding.

Remaining / next:
- [ ] Small **n mod nr == 1** (n=33 0.84×): tiny matrices on the per-column edge. Measured dead-ends
      (don't re-try): `Val(nre)` dispatch, clamp/guard.
- [ ] (optional) SIMD `pack_B` (transpose-y; smaller total cost than pack_A); reduce C re-streaming.
- [ ] C-ABI `dgemm_64_`/`sgemm_64_`: **char args (transA/transB) + hidden Fortran string-length
      args** at the @ccallable boundary (the L3 ABI complication M1 avoided).
- [ ] Optimize complex GEMM (currently generic) + the **complex-return ABI** (resolves deferred c/zdot).

## M3 — Level 2 (CORE COMPLETE ✅) + rest of Level 3 (IN PROGRESS)

**Milestone — core BLAS-2 complete & at the gate (2026-06-29):** gemv, ger, symv, hemv, trmv, trsv,
plus packed (spmv/hpmv/tpmv/tpsv) and banded (gbmv/sbmv/hbmv/tbmv/tbsv). Full suite 5854/5854; every
real-SIMD-path op ≥0.96× OpenBLAS on Zen4 (single-thread, per-machine gate). hpmv/hbmv complex/generic.
Perf plots (BLAS-1, BLAS-2) in `docs/src/performance.md` (regenerate with `bench/plots.jl`). Details
per-op below; the hard-won kernel lessons live in `kb/findings/pureblas-{gemv,symv,triangular,packed-banded}.md`.

**Docs/perf (2026-06-29):** `docs/src/performance.md` (Performance page) with BLAS-1/BLAS-2 gate plots;
`bench/plots.jl` regenerates them as hand-written SVG (no plotting dep). Generating the plots surfaced
that **`iamax` was a scalar loop (~0.3× OpenBLAS)** — now a **SIMD argmax** (`_iamax_simd!`, 4 independent
running-max chains + parallel index vector, first-occurrence tie rule): **median ≥1.06× at every size**
(n=64…1e6) — at gate. NB: OB's idamax is alignment-volatile (~60% time swing by array address) so
single-allocation ratios mislead; median over many fresh allocations is the fair measure. Two-pass
(max-only + SIMD locate) tried, slower (the extra array read > in-loop index-select savings).

### Rest of Level 3 (symm/syrk/herk/syr2k/her2k/trmm/trsm)

**Done (2026-06-29): trmm + trsm CORRECT** (`src/level3.jl`) — `PureBLAS.trmm!`/`trsm!(B, A; side, uplo,
transA, diag, alpha)`. Recursive 2×2 blocking: off-diagonal via `gemm!`, base (≤32) via trmv/trsv
per-column (side L) / column axpy-or-solve (side R). Correct vs OpenBLAS for f32/f64/c64 × side L/R ×
uplo × trans N/T/C × diag (224/224 each). Clean control = the `up != tr` grouping (which B-block the
off-diagonal feeds; trsm reverses the order + subtracts).

**trmm/trsm GATE — NOT met yet; needs a dedicated blocked kernel (measured analysis 2026-06-29):**
three reuse-based approaches all cap below 0.96×:
- recursion-over-`gemm!` (current, correct flops): 0.40 (m=64) → 0.79 (m=1024) — overhead-bound (slow
  triangular base + many off-peak gemm calls).
- triangularize A (zero non-stored half) + one full `gemm!`: ~0.5× — wastes **2× flops** (full product
  over the zeroed half). Hard cap.
- iterative block-row (one wide gemm + small diag trmm): ~0.68× — off-diagonal gemm is short-M (NB rows)
  + the triangular diagonal block is still slow.
Root cause: the triangular **diagonal** computation can't be a clean gemm, and the block-structured
gemms are off-peak shapes. **Gate needs the BLIS approach:** pack the triangular A into gemm's packed
format, **skip the all-zero (below/above-diagonal) blocks** (keeps correct flops), and use a
**triangular-aware microkernel** at the diagonal block; handle the in-place B aliasing (C=B) via loop
order or a small B-copy. trsm additionally needs the diagonal-block solve. Reuses `gemm.jl`'s
`_microkernel!`/`_pack_*`/blocking — a real sub-project (a fraction of gemm's effort). **Next:** that
kernel (or breadth-first syrk/herk/symm correctness, then a unified L3 gate pass).

**L3 BREADTH COMPLETE — all ops CORRECT (2026-06-29, Route 2 phase 1)** (`src/level3.jl`). Added
**syrk/herk, symm/hemm, syr2k/her2k** alongside trmm/trsm — recursion-over-`gemm!` (diagonal blocks
recurse to a scalar base; off-diagonal blocks = full `gemm!`s). Correct vs OpenBLAS for f32/f64/c64,
all uplo/trans/side combos (suite testitems: trmm/trsm + syrk/herk/symm/hemm/syr2k/her2k). Native API
public + AD-traceable. **Gate NOT yet met** for the triangular/diagonal-heavy ones (trmm/trsm; syrk
etc. TBD — their off-diagonal is full gemm so likely closer).

**Route 2 phase 2 IN PROGRESS — gemm-speed diagonals (2026-06-29).** Baseline measurement found the
recursion's **scalar diagonal base** was the killer (syrk/symm/syr2k were 0.15–0.19× at n=256!).
Replaced with gemm-speed diagonals:
- **syrk/herk**: recursive 2×2; off-diagonal = direct gemm into C's triangle; small diagonal base
  (≤32) = gemm→temp + triangle-add. 0.17 → **n≤256 ≈ 1.2×, large-n ≈ 0.82–0.87×**.
- **symm/hemm**: output is a full matrix ⇒ **materialize the symmetric/Hermitian A to dense + one
  gemm!** (no triangle-waste, β/α folded into gemm). 0.15 → **≈ 0.89–0.94×** (residual = the n²
  materialize traffic OB avoids by reading the triangle in-kernel).
- **syr2k/her2k**: recursion + gemm→temp base (two rank-k gemms). 0.19 → n≤256 ≈ 1.1×, large-n lower.
- gemm verified at gate for ALL transpose variants (NN/NT/TN 0.97–1.03× at n=512–2048) ⇒ the residual
  is purely L3 orchestration overhead, NOT gemm.

**Ceiling finding:** reuse-of-gemm L3 caps ~0.85–0.94× at large n (recursion call/packing overhead).
**Resolved for syrk** with the dedicated kernel below.

**SINGLE-PASS PACKED syrk — DONE, at gate (2026-06-29).** `_syrk_packed!` (+ `_microkernel_tri!`) in
`level3.jl`: syrk = gemm(A, Aᴴ) with a triangular C, reusing gemm's `_pack_A!`/`_pack_B!`/`_microkernel!`
(B-operand = A). Each micro-tile is classified vs the diagonal — **skip** below-diagonal, regular/masked
microkernel fully-stored, and a new **triangular-store microkernel** (`_microkernel_tri!`, masks the
store to the stored triangle so K-accumulation stays correct, no temp) for diagonal-straddling tiles.
Packs A once (reads A like a single gemm — no recursion re-reads). Real only (n > 448); complex/herk/
small-n stay on the recursion. Result: **n=512=1.11×, 896=1.01×, 1024=0.95× (power-of-2 ldc cache dip),
1536=0.99×, 2048=0.96×** (was 0.17 scalar / 0.85 recursion). Correct (suite 6934/6934; triangular store
verified not to touch C's other triangle). The `_microkernel_tri!` + tile-classification driver is the
**reusable template** for the rest.
**syr2k/her2k — DONE via the same packed kernel (2026-06-29).** `_trgemm_packed!` generalizes the
syrk core to `C[tri] += α·op(X)·op(Y)`; syr2k = two passes (`A·Bᴴ` + `B·Aᴴ`), same `_microkernel_tri!`.
Real n>448 → packed, else recursion. Result: **n=512=1.05×, 1024=0.93×, 2048=0.94×** (was 0.71–0.84×);
two passes (2× packing) keep it ~0.93–0.94 large-n vs syrk's 0.95–0.96. Large-n tests added (the
testitems previously only went to n≤130 — didn't exercise the packed path).

**trmm packed — built + correct, but capped at ~0.64–0.81× (2026-06-29).** `_trmm_packed!` +
`_pack_A_tri!` (level3.jl): trmm-L = gemm(op(A_triangle), B) with A's non-stored half packed as zero
(skip fully-zero A-panels, plain `_pack_A!` fully-stored, `_pack_A_tri!` diagonal-straddling), B copied
to a scratch to dodge the in-place C=B aliasing. Correct (32/32 incl. unit-diag/trans). **Not yet a win
vs recursion** because the diagonal-straddling A-panels use the SCALAR `_pack_A_tri!`, and with mc=144/
kc=256 that band is a big fraction. **Gate fix = vectorize `_pack_A_tri!`** (within a straddling mc×kc
block, most mr-sub-panels are fully-stored/fully-zero → SIMD copy; only the one crossing the diagonal
needs masking). Until then trmm dispatches to the recursion (no regression); `_trmm_packed!` is dormant.

**syrk / syr2k / symm — ALL GATED (2026-06-29, suite 6966/6966).** Three updates landed:
- **symm/hemm** — killed the n² materialize: pack the symmetric A panels directly inside a single-pass
  gemm (`_pack_A_sym!` side-L / `_pack_B_sym!` side-R, `_symm_packed_L!`/`_symm_packed_R!`). 0.95 →
  side-L 0.97–0.98×, side-R 0.98–1.00×.
- **syrk + syr2k — unified single-pack redesign.** OpenBLAS packs A once and reuses it for BOTH operand
  roles; we couldn't because mr=16≠nr=8 → packed A twice (syrk) / four panels (syr2k), amortizing
  packing over only the triangle. Fix: switch the triangular path to an **8×8 tile** (mr==nr==W, F64/
  AVX-512) so `_pack_A!`/`_pack_B!` layouts coincide → pack each operand ONCE and read it as both roles
  (`_trgemm_packed_u!` syrk, `_trgemm_packed2_u!` syr2k). α moved to the store (`_microkernel_u!` /
  `_microkernel2!`) since a shared buffer can't carry α (would give α²). Result: **syrk 1.05/1.03/1.02/
  0.97 (n=768/1000/1024/2048), syr2k 1.02/1.02/1.00/0.96.** Power-of-2 n=1024 dip eliminated
  (syr2k 0.92→1.00). Float32/AVX2 keep the 16×8 multi-pack fallback (`_trgemm_packed!`/`2!`).
- **DIAGNOSIS (by ablation; perf counters locked):** the dip was packing amortization. DISPROVEN:
  tri-kernel (tri→masked 0.949→0.956), A-aliasing (`gemm(A,Aᵀ)`=0.997, no dip), recursion (gemm! call
  overhead 0.76), 8×8-tile-too-small (8×8≈16×8). See kb finding `pureblas-l3-syrk-syr2k-symm`.

**trmm — K-range trimming (2026-06-29): 0.4–0.83 → 0.93–0.96, suite 6966/6966.** `_trmm_packed!` now
trims each straddling tile's contraction to its nonzero p-band (upper p≥row / lower p≤row) instead of
FMAing the full kc zero band — that band was the ~kc/k waste (25% at k=1024). 8×8 tile (finer staircase).
Side-L real large → packed; else recursion. **Residual ~3–5% (not yet uniform gate):** diagonal mc-band
(~14% of compute at k=2048) runs short-cnt latency-bound tiles (needs a dedicated diagonal kernel);
transpose cases pay scalar A-pack (SIMD pack is N-only). B-copy ruled out (<1%).

**trsm — GATED (2026-06-29): 0.4–0.9 → 1.02–1.12×, beats OpenBLAS.** Inverse base: a diagonal block
≤`_TRSM_BASE`=128 is solved by inverting its triangle (`_trtri!`, tiny) + applying op(inv) as a gemm —
diagonal solve at gemm speed; off-diagonal already gemm!. Real only (stable for trsm's well-conditioned
blocks); complex/conj keep scalar trsv. **KEY cross-cutting insight: packing a triangular matrix's
SUB-views at a pure-power-of-2 ld thrashes one cache set** (k=1024→0.78, 2048→0.94); copying A into a
padded-ld scratch (ld=k+8, `_badld`/`_l3_apad`) fixes it → 1.12/1.06. B-padding doesn't help (only A);
NOT β=1 C-RMW (gemm β=1 @1024=0.999). Bug fixed: `_trtri!` must zero the non-stored half (gemm reads
the full NB×NB block). Same A-padding applied to trmm (+1–2%; k=2048→0.962 gates).

**trmm — ~0.94–0.96, the one sub-gate L3 op; residual UNRESOLVED.** K-trim + 8×8 got it 0.4→0.94–0.96.
The last ~3–5% resisted every lever — ruled out by measurement: (a) B-copy (the in-place recursion
`_trmm_left!` has none yet is WORSE, 0.4–0.83); (b) diagonal band blocking (mc 48–144 AND kc 64–256
sweeps both fail; bigger kc is better, smaller worse → band-fraction theory wrong); (c) cache-ld (A- and
Bc-padding, no close); gemm profiled across K=16..2048 = 0.97–1.08 (no short-K weakness). The one
fixable sliver — scalar transpose A-pack for transA=T — is **FIXED: SIMD transpose pack**
(`_pack_A_simd_T!`/`_tblk!`, W×W shuffle-butterfly transpose), bit-identical, helps gemm-T/syrk-T too.
**Final trmm: IN-PLACE single-pass + OVERWRITE-ON-FIRST (Val(1) 8×8 + transpose pack, NO A-pad).**
Two structural wins (git branches, merged): (1) **in-place** — eliminate the Bc full copy by packing each
jc panel into Bpf before overwriting (trmm-L columns independent); big at small k (k=768 0.90→0.95–0.99,
the O(k²) copy is a large fraction of O(k³)). (2) **drop the zero pass** — each tile's first contributing
pc-block writes β=0 (overwrite, no C read; first block div(r0,kc) upper / 0 lower), later accumulate;
`Val{B0}` path added to the microkernels (default false → gemm/syrk unchanged). This closed the po2
holdouts (k=2048 UN/LN 0.957→0.963). **trmm now GATES k=1024/1536/2048 (ALL variants, 0.963–0.994) and
k=768 (3/4); only k=768 UT ~0.954 left** (small-k transpose, within noise). A-pad re-tested twice
post-Bc-removal: still net-negative → removed. Suite 6966/6966, relerr ~5e-16. Late
findings: cache-oblivious RECURSION (Elmroth–Gustavson/ReLAPACK) measured SLOWER than single-pass at
every size → DISPROVEN (anchor on the fastest path, extend it); trmm A-pad REMOVED (po2 conflict mild,
the k² copy is net-negative; kept for trsm where the conflict is catastrophic); 16×8 tile-by-trans
non-robust → reverted. Matches OB per-flop (1.175 vs 1.162); hand-unrolled diagonal kernel DISPROVEN for
MR=1 (triangle zeros are free vector lanes). **trmm ≈ column-major ceiling; non-po2 sizes gate.**
Net: all 8 L3 correct; **ALL routines effectively gate** — gemm/symm/syrk/syr2k/trsm fully (complex via
fallback), trmm gates k=1024/1536/2048 (all variants) + k=768 (3/4), only k=768 UT ~0.954 left. Packed infra:
`_microkernel_tri!`/`_microkernel_u!`/`_microkernel2!`, `_trgemm_packed{,2,_u,2_u}!`,
`_pack_A_sym!`/`_pack_B_sym!`, `_pack_A_tri!` (+SIMD), `_pack_A_simd_T!`/`_tblk!` (SIMD transpose pack).

### ⚠ KNOWN GATE GAP — REVISIT (L3 otherwise DONE, 2026-06-30)
The ONE place the ≥0.96× gate is not met on Zen4: **trmm small-k transpose (UT) — k=512 ~0.945,
k=640 ~0.942** (k=768 UT ~0.954–0.962 borderline). All other trmm cases (k≥768 most variants;
k=1024/1536/2048 all variants) and all other L3 routines gate. The benches there are noisy (±2–5%).
- **Cause:** the transpose A-pack (`_pack_A_simd_T!` shuffle butterfly) costs more than non-trans
  column-copy, and at small k it's a large fraction of the overhead-dominated work. Inherent to `transA=T`.
- **Disproven fixes (do NOT re-chase):** SIMD transpose TRI-pack for the straddle (mask overhead offsets
  the gather savings); A-pad (net-negative for trmm); cache-oblivious recursion (slower); 16×8 tile;
  hand-unrolled diagonal kernel (free vector lanes at MR=1); option-1 zero-pad.
- **To revisit:** a BLASFEO-style UNPACKED small-matrix trmm path (skip packing entirely for cache-
  resident k, like gemm's `_use_unpacked` ≤448 path) — the most promising untried angle for small-k; or a
  cheaper/faster transpose pack. Reference: kb finding `pureblas-l3-syrk-syr2k-symm`.

## LAPACK — Cholesky (potrf) — ✅ CORRECT + AD + GATED (2026-06-30)
**Float64 lower GATES: 0.985–1.12× LAPACK dpotrf across n=512–3072** (suite 7043/7043, relerr ~1e-15).
The unlock was porting **faer 0.24.1's Cholesky** (el-oso/BlazingPorts.jl `src/Factorizations.jl`) onto
PureBLAS's SIMD.jl layer: custom register-blocked kernels (left-looking base, fused trsm NB=4, fused syrk
3×4=12 accs) — no packing overhead → fast at the small Cholesky block sizes where the generic recursion
(below, maxed ~0.81) lost. Pure faer faded at large n (un-cache-blocked syrk re-streams), so a **hybrid**:
halve, big off-diagonal via PureBLAS's cache-blocked `trsm!`/`syrk!`, faer kernels as the base (≤1024).
Pad on `stride%512==0` (L1 set-aliasing; faer's `ispow2` missed 1536/2560). Float64 lower fast path;
Float32/complex/Dual/upper keep the generic AD-traceable recursion. kb: `pureblas-cholesky`. Lesson: a
faithful proven-fast port beat incremental tuning of the generic version. Below = the historical journey.

## LAPACK — QR (geqrf) — ✅ CORRECT + GATED (2026-06-30)
**Float64 GATES: 0.96–1.32× LAPACK dgeqrf across n=512–3072** (beats it at 5/6 sizes; suite 7060/7060,
|R| & Q·R ~1e-15). `src/qr.jl`. Same recipe as Cholesky: port only the **irreducible** faer kernel —
`qr_unblocked!`, the SIMD Householder panel reduction (el-oso/BlazingPorts.jl) — onto PureBLAS's SIMD.jl;
drive the blocked **compact-WY** dlarfb (`C −= V·(Tᵀ·(Vᵀ·C))`, nb=32) with **PureBLAS's gated `gemm!`** for
the two big gemms (Y=TᵀW tiny → scalar). **Skipped faer's bespoke packed BLIS gemm + `@generated`
microkernel entirely — PureBLAS has a gemm** → far less code, gates+beats dgeqrf. Float64 only (faer
kernels Float64-specific); generic/AD QR deferred. n=768 borderline (0.962, noise). kb: `pureblas-qr`.

## LAPACK — LU (getrf) — CORRECT; GATES 768–2048 (0.97–1.06×); 512/3072 residual (2026-06-30)
`src/lu.jl`. **BlazingPorts has no LU source** (only bench JSONs) → from scratch, but blocked
right-looking = LAPACK dgetrf's own algorithm + PureBLAS trsm!/gemm!. Correct: matches LAPACK exactly
(factor + ipiv), P·A=L·U ~1e-14, suite 7085/7085. **Ground to gate the mid-large range** (don't-guess-check:
our gemm at the trailing shape AND trsm at the panel shape both BEAT OpenBLAS 1.0–1.5×, so the bulk is
optimal — the gap was small components): (1) **laswp loop order** cols-outer/pivots-inner 108→18ms;
(2) **size-adaptive laswp** (small m column-outer / large m 32-col blocked) recovered 768 + gated 1024–2048;
(3) **po2/stride%512 padding** (+0.05 @2048/3072, cache aliasing). SIMD panel: no help (memory-bound, as is
dgetf2). nb=48. **DISPROVEN: faer-style recursive LU** (over-decomposes → many small gemm! calls, 0.83 <
blocked 0.89). **Residual: 512 (0.87, small-n O(n²)/O(n³) overhead) + 3072 (0.94, pad-copy + scaling)** —
diminishing returns vs a decade-tuned dgetrf. kb: `pureblas-lu`.
## LAPACK — SVD — NOT STARTED (no BlazingPorts source; large from-scratch effort)
Bidiagonalization (two-sided Householder) + iterative (implicit-QR / divide-and-conquer) + singular
vectors. Weeks of work, not a port. Needs a scope decision before starting.

### (historical, Cholesky) generic recursion tuning — CORRECT + AD, maxed ~0.81 before the faer port
First LAPACK routine, `src/lapack.jl`. Recursive (cache-oblivious) Cholesky on the gated L3: split 2×2 →
factor A11, trsm the off-diagonal panel, syrk-downdate the trailing, recurse; unblocked `potf2` base
(≤`_POTRF_BASE`=512, vectorized inner loop). Lower (L·Lᵀ) + upper (Uᵀ·U). **Generic over real T →
ForwardDiff-traceable** (the headline Mode-2 win: differentiable Cholesky, e.g. ∇logdet); BlasReal hits
the SIMD trsm/syrk. PosDefException on non-PD. Correctness vs LAPACK `cholesky` ~1e-16, suite 7031/7031.
**⚠ GATE NOT MET (revisit): n=1024 0.57, n=2048 0.81, n=4096 0.90 vs LAPACK dpotrf** (after the
contiguous-buffer panel below; efficiency grows with n as overhead amortizes). Decomposed (n=2048): trsm
42ms / syrk 33ms (we MATCH LAPACK) / panel 26ms; the ~20ms gap = memory-bound panel (~11) + k=512 trsm
(~9), not syrk. **Tuning done:** base=512 sweet spot (smaller→small-k trsm cost, larger→bigger panel);
**contiguous-buffer panel** (copy strided base block→contiguous, factor, copy back) lifted n=2048
0.70→0.81. **Remaining to gate:** (1) BLOCK the panel within the buffer (cache-REUSE, compute-bound — the
buffered potf2 is still unblocked/memory-bound by volume); (2) the k=512 side-R trsm. Both real multi-step
work; n≥4096 likely gates with just the panel fix. Like dgemm: correct first, dedicated tuning pass next.

**Bench harness `bench/l3bench.jl` (2026-06-30): staged screen→full, faster + more correct.** SCREEN =
one non-po2 size (k=1536) × all variants; full size sweep only on routines that fail the screen. Adaptive
rounds (grow until IQR/median<2%, cap 45 — keeps interleaved-median, no under-sampling). reps right-sized
so one timed call ≳80ms (L3 large-k ⇒ few reps). Line-buffered file output (no grep-in-a-pipe). **Fixed a
real methodology bug:** in-place ops (trmm/trsm) were benched as reps× in-place on ONE buffer → OVERFLOW
to Inf (trmm) / denormal underflow (trsm) corrupting old numbers, and per-call `copy` gave 14–25% IQR.
Fix: reps=1 for in-place with an UNTIMED `reset` (copyto!) per round → pure-kernel timing, IQR 1–2%.
Screen of all 6 L3 routines ≈ 2:10 wall. Usage: `taskset -c 2 julia --project=bench bench/l3bench.jl
[screen|full] [routines...]`. Numerics/StrictMode/suite gates unchanged.



Done (2026-06-29): **gemv + ger, performance gate MET** (`src/level2.jl`) — native API +
`AbstractBLAS2` contract + SIMDBackend; generic `T<:Number` path (AD-traceable) + SIMD fast paths;
StrictMode-audited; correct vs OpenBLAS (all trans N/T/C, alpha/beta, edges, complex geru/gerc).
Full suite 2519/2519. Perf (Zen4, F64, single-thread): **0/39 below 0.96×, min 1.007×, geomean
1.22× — beats OpenBLAS at every size** (16..4096, gemv-N/gemv-T/ger). Kernels (see kb finding
pureblas-gemv): gemv-N = 2 regimes — n ≤ 448 row-block (y in registers), else column-panel
(`_GEMV_NP=8` cols/pass → y re-streamed n/8 times, A in 8 sequential streams; **unmasked full-block**
kernel + masked remainder). gemv-T = column-block (4 dots share each x-chunk) for all n. ger =
per-column axpy. β folded into the SIMD kernels. Public API is `@inline` with explicit kwarg
forwarding (the `; kw...` splat otherwise cost ~200 ns/call — dominated tiny-matrix gemv).

Done (2026-06-29): **symv + hemv, performance gate MET** — symv ≥0.96× for f32/f64 ×
{U,L} across n=16..4096 (geomean 1.20–1.32×); hemv complex/generic (correct vs `Hermitian·x`). symv
reads only n²/2 of A, so the vector re-stream costs more than gemv (naive column kernel hit 0.63× at
n=4096). Kernel (see kb finding pureblas-symv): a **unified fused panel** — gemv-N (yL kept in MR=4
registers across NB=8 cols) + gemv-T (NB dot accumulators), A read ONCE, with the **triangular
diagonal block folded into the same `d_c` accumulators** (one reduction per column + vectorized
diagonal) and an nv-adaptive masked remainder; lower/upper are mirror kernels. Full suite 2734/2734.

Done (2026-06-29): **trmv + trsv, performance gate MET** — all 8 combos (trmv/trsv × N/T × U/L),
f32/f64, n=16..4096: **0/104 below 0.96×, geomean 1.118×**; complex/AD via the generic path. Full
suite 3603/3603. Kernels (see kb finding pureblas-triangular): per-column SIMD (N=axpy via
`_axpy_simd!`, T=dot via `_dot_simd`) + scalar diagonal; large-n **blocked** — diagonal block
(per-column) + off-diagonal **gemv** (reads A once). Lessons: the off-diagonal block must be **TALL**
(N forms organized by column-block) for locality; the tall scatter calls the gemv-N column-panel
directly (n=NB cols would hit the row-block = NB strided streams that thrash on sub-block column
spacing), the T off-diagonal calls the gemv-T kernel directly (skip the ~200 ns kwarg layer);
**per-OP unblock threshold** (measured): trmv-T blocks at NB=64, trsv-T unblocks ≤1024. `_l2_simd_ok`
relaxed to unit-stride `StridedVector` so contiguous sub-views take the SIMD gemv path (general win).

Done (2026-06-29): **packed + banded L2, GATE MET for all 9** (`src/level2_packed.jl`,
`src/level2_banded.jl`) — spmv, hpmv, tpmv, tpsv (packed); gbmv, sbmv, hbmv, tbmv, tbsv (banded).
Full suite 5854/5854. spmv/sbmv/tpmv/tpsv/tbmv/tbsv geomean 1.24–1.51×; **gbmv 0/36 below 0.96×,
min 0.989×** (band 1..256 × n 300..4096). hpmv/hbmv complex/generic. Reuse: packed & band columns are
contiguous ⇒ same per-column kernels with packed/band offsets. gbmv needed 3 kernels (kb finding
pureblas-packed-banded): gbmv-N conv-by-output-block (band≤48); gbmv-T scalar dot (band<W) +
BLASFEO-style x-register-reuse conv (band≥W, `shufflevector` register shift, no gather) + β fused.
Dead ends: dense-routing (0.11–0.28×), per-diagonal/transpose (gather). tpmv/tpsv/tbmv/tbsv have no
LinearAlgebra wrapper → ccall OpenBLAS `_64_` symbols for the gate.

**Core L2 is complete:** gemv, ger, symv, hemv, trmv, trsv + the 9 packed/banded routines.
Remaining L2 (optional): packed/full rank updates spr/spr2/hpr/hpr2.
Then the rest of L3 (symm/syrk/herk/syr2k/her2k/trmm/trsm).

## M4 — multithreading (DEFERRED by user — do not start until explicitly requested)

Parallelize the gemm jj-loop, threshold-gated (small sizes stay serial). Per-host tuning. This is
for **absolute throughput / scaling across cores** — NOT for closing an OpenBLAS gap: single-thread
`dgemm` is already at parity (geomean 0.999×). (Earlier note claimed large-n was "single-thread
L2-bandwidth-bound, needs threading" — that was wrong; it was scalar packing, fixed by SIMD pack_A.)
**Standing instruction (2026-06-28): defer ALL multithreading requests until later — keep everything
single-threaded for now.**

## M5 — complex SIMD + multi-ISA dispatch

Interleaved re/im SIMD for complex kernels. Runtime AVX-512/AVX2/NEON dispatch in one build
(so a single artifact runs optimally across the Zen3/Zen4/Zen5/ARM fleet).

## M6 — AD rules

`PureBLASChainRulesExt` (weakdep) + Enzyme rules so Mode 2 supports reverse-mode through the
in-place ops. (Native path is already ForwardDiff-traceable today.)

## Later

ARM/aarch64 trim build for the Mac M5 (cross-compiled .so/.dylib). LAPACK surface. SparseArrays
interop; CHOLMOD / sparse Cholesky.
