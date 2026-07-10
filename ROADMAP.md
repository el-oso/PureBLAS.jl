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
- [~] Re-run the gate per-machine on the rest of the fleet (Zen3 AVX2, Zen5 native-AVX512, M5 ARM).
  - **Zen3 (galen, Ryzen 9 5900X, native AVX2, W=4) — measured 2026-07-02:** CORRECTNESS ✅ full
    suite **7213/7213** (native AVX2 — the real-hardware confirmation of the fallback path, incl. the
    symv NB≤W kernel). PERFORMANCE ❌ below gate on L3/LAPACK: L1 mostly PASS (iamax 0.81 FAIL), L2
    PASS except gemvN 0.80 / gbmvN 0.62, **all L3 FAIL (gemm 0.63, symm 0.58, syrk 0.59, syr2k 0.65,
    trmm 0.64, trsm 0.64)**, all LAPACK FAIL (potrf 0.61, geqrf 0.82, getrf 0.68, gesvd 0.86).
    **ROOT CAUSE (grounded, not guessed):** the gemm microkernel tile is `_MR=2 × _NR=8 = 16 vector
    accumulators` — sized for AVX-512's 32 registers; on AVX2's **16 ymm** it spills accumulators to
    stack every FMA. All L3/LAPACK build on gemm ⇒ the whole stack inherits it. **Fix = an
    architecture-adaptive tile, then per-op threshold retune** — a Zen3 tuning campaign analogous to
    the wintermute small-n one. Baseline cache: `bench/plots_data_galen.txt` on galen. Per-machine gate
    (unpinned; cpufreq pin needs sudo on galen).
    - **PROGRESS (2026-07-02, commit 8706372) — gemm GATES on Zen3.** Made the gemm tile (`_MR/_NR`)
      and unpacked-crossover (`_GEMM_UNPACK_MAX`) width-adaptive + Preferences-overridable
      (`gemm_mr/gemm_nr/gemm_unpack_max` — the fleet calibration knob). Swept on galen:
      **W=4 -> MR=3,NR=4 (12 accs, fits 16 ymm), unpack<=128**; W=8 unchanged (MR=2,NR=8,unpack<=448,
      wintermute suite bit-identical). gemm worst **0.63->0.99**, geomean 1.18, gates every n=8..2048.
      Lifted **geqrf/gesvd/iamax -> PASS**, symm 0.58->0.90, getrf 0.68->0.87. Full suite 7213/7213 on
      BOTH galen (AVX2) and wintermute (AVX-512). Microkernels were already `Val{MR,NR}`-parameterized;
      only driver consts changed.
    - **PROGRESS (2026-07-02, commit d1a748a) — syrk/syr2k/trmm structural fix (large-n gates).**
      Root found: the unified single-pack (syrk/syr2k/trmm) needs mr=nr=W (MR=1) → exactly W
      accumulators; on AVX2 that's 4 = latency-STARVED (opposite of gemm's spilling). Fixed:
      `_unified_ok` now requires W>=8, so W=4 routes to the multi-pack `_trgemm_packed!` with the wider
      `_MR×_NR`=12-acc tile. **syrk 0.57→geomean 0.98, syr2k 0.55→0.82, trmm 0.59→1.10** — all large-n
      (n≥256) gate. W=8 unchanged (unified still on; wintermute 7213/7213). `_SYRK_DBASE` made
      Preferences-overridable.
    - **REMAINING Zen3 gaps:** **small-n (n≤128) syrk/syr2k** — recursion path: n=32 ~0.79, n=128 ~0.82
      (the gemm-into-temp diagonal base wastes 2× flops; `syrk_dbase` swept 8..48, does NOT gate → needs
      a dedicated small-n triangular kernel, like the wintermute small-n campaign built per op). trsm
      0.68 (small-n n≤256; large-n gates), symm 0.90, potrf 0.67, getrf 0.88 (potrf/getrf follow once syrk
      small-n + trsm small-n gate). L2 gemvN / gbmvN **now done** (phase 1, see below). This is the Zen3
      analogue of the wintermute small-n grind — a multi-op campaign.
    - **UPDATE 2026-07-04 (autonomous) — re-measured galen (AVX2, boost OFF) + syrk/syr2k FIXED.** The
      2026-07-02 numbers above were partly stale (many lifted by the width-adaptive gemm/L3 landings +
      the const-dispatch workspace). CURRENT measured small-n gate state (bench/smalln_probe.jl, taskset
      -c 8): trmm ALL gate (1.06–1.25); getrf ALL gate; syr2k n≤128 gate. **syrk n=64/128/256 (0.83/0.94/
      0.94) + syr2k n=256 (0.92) were still LOW → NOW FIXED** (commit 947336d): the single-product tri
      multi-pack `_trgemm_packed!` used the gemm row-tile MR=3 (mr=12); at n not divisible by 12 the last
      row-panel is zero-padded → wasted flops. Width-adaptive `_tri_mr(T)=W==4 ? 2 : _MR` (mr=8 divides
      64/96/128/192/256, 8 accs ample ILP) → syrk/syr2k gate the WHOLE AVX2 range (MR2 ≥ MR3 at every
      n=64..2048, exact correctness, no large-n regression; AVX-512 bit-identical). Knob "syrk_mr".
      **DISPROVEN this round (don't re-try): power-of-2 leading-dim padding of A/C for small-n syrk — made
      it WORSE** (the gap was mr-divisibility, not ld-thrash; verified by the n%12==0 sizes 48/96/192
      already gating). **STILL OPEN on AVX2 (the real remaining grind):** trsm n=32/64 (**0.787** — the
      `_trsm_dense_L!` per-column rank-1 base: serial dependency + shrinking vector length, hard to beat
      OB small-n); potrf n=128/256 (**~0.87** — its faer base kernel's fused syrk is a 3×4=12-acc tile
      sized for AVX-512; the Zen3 analogue of the syrk fix likely applies but the kernel is in lapack.jl
      and more involved). trsm large-n + potrf n≥512 gate. NEXT: trsm dense-base and/or faer-potrf base
      register-blocking for W=4. **DISPROVEN-by-measurement this session (don't re-try):** (a) routing
      small-n narrow-B trsm through the invL+gemm base instead of `_trsm_dense_L!` — invL is FAR worse on
      AVX2 (k=32 nrhs=32: dense 0.855 vs invL 0.495; dense is already the better base — the rank-1 dense
      solve IS the ceiling, the gap is the serial-dependency/shrinking-vector nature). (b) guarding the
      faer `_syrk_lower_f64!` MR=3 (12-acc) block to W≥8 so AVX2 starts at MR=2 — did NOT help potrf
      (n=128 0.873→0.853, n=256 ~same), so MR=3 register pressure is NOT the potrf limiter; the bottleneck
      is elsewhere in the faer base (`_chol_base_f64!` left-looking panel or `_trsm_right_lower_f64!`, both
      also 3·W-unrolled) — needs per-kernel decomposition, not a blanket tile change.
    - **UPDATE 2026-07-04 (session 2) — gemvN mid-band FIXED + residual triage (commit 676582e).**
      Decomposed every remaining AVX2 gate FAIL to per-size on galen (boost off):
      - **gemvN — FIXED** the mid-band (n=128..768 sat ~0.90-0.94). Root cause: `_GEMV_MR=4`
        accumulators half-fed Zen3's two FMA units (~5-cyc latency wants ~10 independent chains) while
        A is still cache/L3-resident → FMA-latency-bound, not bandwidth. Two width-conditional consts
        (AVX2 only; AVX-512 verified bit-identical, W=8→MR=4/RB=448): `_GEMV_MR 4→8` (8 accs feed both
        FMA ports, 10/16 ymm, no spill; MR=10/12 spill+regress), `_GEMVN_RB 192→64` (with MR=8 the
        sequential-streaming panel path beats strided row-block for all n≥96, so route mid-n to it;
        row-block only wins n≤64 where panel's m<mr all-masked tail dominates). galen real public gemvN
        median: n=64 1.09→1.44, **128 0.91→0.98, 256 0.91→1.04**, 512 0.92→0.94, 1024 0.98→1.03,
        2048 1.03→1.07. Correctness nfail=0 over m,n∈{1..513}×{1..512}, s/d, α≠1, β≠0. **ONLY n=512
        remains** (0.93 median, min 0.91/max 0.97 — L3-bandwidth ceiling, A=2MB; a single-size ceiling
        like zgemm's registers / potrf's serial dep).
      - **iamax — NOT a real fail (measurement noise).** Cached plots showed n=3000 0.912; re-measured
        median 0.994 with ±40% per-round variance (min 0.55/max 1.3) — alignment-sensitive reduction,
        at parity in expectation. Kernel is fine; the plots.jl median-of-20 fabricates the red.
      - **DISPROVEN this session (don't re-try):** (a) **symm n=256 (0.944) via smaller kc** — the
        n=256 all-straddle-pack (kc=256=n → every M-block crosses the diagonal → 100% `_pack_A_sym!`)
        is NOT the bottleneck; kc-parameterized reimpl showed KC 256/192/128 all ~0.93 at n=256, KC≤96
        WORSE. Gap is inherent packed-gemm efficiency at that shape, no clean lever. (b) **gemvN n=512
        via α==1 fast path** (`Val{A1}` skipping the per-column `av*` broadcast-multiply) — built +
        verified correct, but n=512 unmoved (0.936→0.93; the "0.975" leaner-probe reading was a lucky
        median). Only helped already-passing n=64. Reverted (dead-purpose complexity). (c) **gemvN n=512
        via software prefetch** (llvmcall `@llvm.prefetch`) — naive per-inner-iteration placement tanked
        it to ~0.58; prefetch is non-portable + fiddle-tuned, not worth it for one borderline size.
      - **trmm n=8 (0.842) — public-wrapper dispatch overhead, not the kernel.** `_trmm_small!` called
        directly at k=8 = 0.999; the `trmm!`→`_trmm!`→`_trmm_left!` chain adds ~16% on a ~50ns 8×8 op.
        Tiny corner; not chased (would need a tiny-k fast-path in the public entry).
    - **potrf AVX2 decomposition (2026-07-04, session 2) — already well-tuned; no fruit found; TARGET IS
      HASWELL not Zen3.** Full sub-kernel decomposition on galen (Zen3): the trailing update `syrk!`
      **beats** OpenBLAS (128=1.03, 256=1.04), panel `trsm!` fine (0.95), faer base-64 excellent (1.30).
      The gap is the **faer path at n=128 = 0.87** (potrf n=128 IS that path), which degrades from base-64's
      1.30 because `_chol_rl(128)` splits into bs=64 blocks adding the fused trsm/syrk. **DISPROVEN levers
      (don't re-try):** (a) `_CHOL_FAER_BASE` crossover sweep — 128 (current) is optimal; 64/96 make n=256
      WORSE (0.79). (b) raising `_CHOL_THRESHOLD` so n=128 uses the monolithic `_chol_base_f64!` directly —
      the base kernel is **L1-bound** (base@64=1.30 fits 32KB L1 exactly; base@128=0.53 thrashes L1), so
      thresh=64 is already optimal, not arbitrary. Residual ~0.83–0.91 is the intrinsic serial small-n
      Cholesky vs a strong OpenBLAS-on-Zen3 — no single sub-kernel to fix. **Key caveat:** this is all vs
      *OpenBLAS-on-Zen3*. The user's deployment target is **Haswell** (also AVX2), where OpenBLAS/MKL kernel
      quality differs; the faer approach hit ~1.8× on **Zen5-AVX2** (likely OpenBLAS under-tuned there), so
      the Haswell outcome is genuinely **unmeasurable without a Haswell box** and hinges on OpenBLAS/MKL-
      Haswell. Side experiment: BlazingPorts `cholesky_llt!` as-is is NOT faster (equal on AVX-512, slower
      on AVX2 — PureBLAS is its tuned descendant). Bench infra: `plots.jl bench mkl` added for the eventual
      real-Haswell vs-MKL run; docs `performance.md` has a footnoted **Haswell\*** column (AVX2 proxy).
    - **Haswell static tuning via `llvm-mca` (2026-07-04, session 2) — one lever landed, auto-detected.**
      Workflow (reusable): `julia -C haswell` → `code_native` the kernel → `llvm-mca -mcpu=haswell` (already
      on galen at `/usr/lib/llvm-20/bin`). Findings on the potrf kernels: **syrk 12-acc tile is throughput-
      optimal** on Haswell (6 cyc/iter = 12 FMA/2 units); **trsm diagonal** is inherently serial (unfixable);
      **chol_base k-reduction was latency-bound** (10 vs 4 cyc/iter — only 3 accumulators, each 2 serial FMAs
      after LLVM's ×2 unroll; Haswell's narrow OOO can't hide the 5-cyc FMA chain). **LANDED:** split each
      row-block's k-reduction into even/odd partials → 6 independent chains → llvm-mca 10→5 cyc/iter. It only
      helps Haswell: Zen3/Zen4's wider OOO already hid the chain (measured slight regression), so it's keyed
      on `_INTEL_AVX2` (cpuinfo.jl: Intel && AVX2 && !AVX512F, const-folded) via `_CHOL_BASE_SPLIT`
      (@load_preference override `chol_base_split`). Auto-on when built on Haswell, auto-off on Zen/AVX-512
      (bit-exact faer path preserved there). Reassociates the reduction (loses faer bit-exactness on the split
      path, stays OpenBLAS-correct ~1e-14). Payoff marginal (base runs only at n≤64) and **static-only —
      unvalidated on real Haswell**; flip `chol_base_split` off if a real run shows it doesn't help.
    - **potrf AVX2 po2-stride GATE CLOSED (2026-07-05) — fused `@inline` panel driver, BEATS OpenBLAS.**
      `_chol_panel_f64!` (lapack.jl) replaces the whole-pad on AVX2 for po2 strides n>128: per NB=128
      block, factor the diag in a conflict-free D scratch, solve the panel INTO a conflict-free T via a
      split-ld faer trsm whose FIRST TOUCH reads po2 A21 (copy-in fused away), trailing update reads T
      (@inline split syrk when T ≤ L2/2, packed syrk! reading T above), single streaming writeback T→A21.
      **galen (BenchmarkTools, boost off, taskset): PB/OB = 256 1.08 · 512 1.06–1.07 · 1024 1.04–1.05 ·
      2048 1.03–1.04** (was 0.92/0.95/0.96/0.97 whole-pad; nopad ceiling 0.99–1.01 — the driver beats it
      because the composition is better, not just the stride fix). Decisive lever (per-stage decomposition):
      the MR=1×4 trsm gemm-pass was load-port-bound (25% slower than packed trsm!); upgrading it to the
      **MR=3×NC=4 12-accumulator tile** (7 loads/12 FMAs) made the split trsm FASTER than packed trsm!
      (318 vs 411 µs @512). AVX-512 + non-po2 + n≤128 untouched. Bit-reproducible, relerr ≤4e-16 at
      n∈{129,200,256,384,512,1000,1024,2048} po2 parent strides, 0-alloc steady state, suite green both
      boxes. Fusion verified: `@code_llvm` shows the split kernels fully inlined (387 fmuladd.v4f64 in the
      driver body; native 558 vfnmadd). kb `pureblas-cholesky.md` has the full stage decomposition.
    - **DISPROVEN (2026-07-02, do NOT re-try): unpacked triangular small-n syrk.** Built
      `_microkernel_unpacked_u!` + `_syrk_unpacked!` (compute the triangle directly from column-major A,
      no packing, no 2× waste) and routed small-n W<8 real trans='N' to it. **Measured WORSE:** n=8
      1.26→0.58, n=32 0.79→0.59, n=128 0.82→0.65. Reason: at small n most tiles straddle the diagonal →
      *masked* triangular stores, which cost far more than the clean full-tile stores that make unpacked
      *gemm* fast. Correct (validated vs A·Aᵀ, all n/uplo/α) but slower than recursion → reverted. The
      recursion (which leans on our fast tiny gemm for the base) remains the best small-n path (~0.82).
    - **NEXT PHASES (planned 2026-07-02, in priority order):**
      1. **Independent Zen3 ops — DONE for gemvN + gbmvN (2026-07-02); trsm already optimal.** Reused the
         width-adaptive-default playbook (W=8 default = old value → bit-identical; W=4 default = measured).
         - **gbmvN GATES (0.62 → geomean ~1.08, every gate size ≥1.00).** Root cause (measured, not guessed):
           the masked convolution kernel wins big on W=8 (band 33 → 1.32) but LOSES to per-column axpy on
           AVX2 above band ~17 (band 33 conv 0.62–0.74 vs axpy 1.07–1.10). The conv↔axpy crossover is
           band-based and stable across n=256…4096 (NOT an AB-cache effect — axpy wins at n=64 too; and NOT
           latency — adding partial accumulators made it *worse*, both hypotheses tested & rejected). Fix:
           `_GBMV_CONV_MAX` width-adaptive (`W==4 ? 20 : 48`), one-line threshold, kernel body unchanged.
           `src/level2_banded.jl`. Correct on galen (264 cases, all bands/α/β).
         - **gemvN 0.80 → geomean 0.98 (n=256 rowblock cliff 0.74→0.90 fixed).** Root cause: the row-block
           path assumes A is cache-resident, but at n=256 A (512KB) exceeds Zen3's 512KB L2 → panel path
           (streams A once) wins. `_GEMVN_RB` width-adaptive (`W==4 ? 192 : 448`, the cut where n²·8B crosses
           L2). `_GEMV_NP=8` swept — optimal on both widths, left a plain const. `src/level2.jl`. Residual:
           n=256/512 panel ~0.90 (noisy, panel-kernel limit, not routing) — geomean gates, per-size doesn't.
         - **trsm — routing knobs already optimal, no change.** Swept `_TRSM_NCUT/_TRSM_BASE/_TRSM_DBASE`:
           current values (96/32/16) are the best; raising NCUT or BASE *collapses* n=128 (0.45–0.55, dense
           leaves lose to invL leaves). The trsm loss (n=32 0.66, n=128 0.76, n=256 0.79) is small-triangular
           *kernel* efficiency (invL base + dense back-sub), not routing → **moved to phase 2**. Knobs reverted
           (YAGNI; phase 2 re-adds the sweep surface). Large-n trsm (512+) already ~0.92–0.97.
         - **trsm/getrf UPDATE (2026-07-03, commits 63268ae + 6b0b2ab).** Decomposed the invL/invR base on
           an idle galen core: the n=256 gap is the leaf **GEMM** (18µs/leaf, dominant), NOT the copyback
           (~1.5% — the prior "copyback restructure needed" claim was WRONG; do not rebuild a packed
           `_trsm_kernel!`). The leaf shape is skewed (nb≤32 tiny, B wide) → routed its multiply through
           `_gemm_unpacked!` (no B-pack, `Val{B0}` overwrite; 0.72× packed time). **trsm n=256 0.85→~0.93,
           getrf n=256 0.91→0.98 (GATES).** Extending unpacked to the off-diagonal gemms REGRESSED (cache
           thrash in context — isolated micro-bench lied). Profile now: off-diagonal packed gemms = 68% of
           trsm n=256, unpacked leaf 22%, trtri minor. Remaining trsm n≤256 (~0.88–0.93) is the packed
           off-diagonal gemm at skewed shapes. Also unified all 7 L3 scratch globals into one owned
           `L3Workspace{T}` (`src/workspace.jl`, PureFFT plan-owned pattern) — kills the abstract-Matrix
           boxing class, thread-ready, perf-neutral single-thread.
         - **gemm clip kernel (2026-07-03, commit 1712e75) — closes trsm n=256 + general small-m win.**
           Traced the trsm off-diagonal gap to gemm **m-alignment**: PB packed gemm at h=32/64 (wide B,
           untested by the square gate) was 0.89–0.90× OB *purely* because m wasn't a multiple of mr=12
           (aligned m ran 1.10–1.16; k irrelevant). The packed path masked the W-aligned remainder tile
           (computing all mr rows to use mre). Added `_microkernel_clip!` (reads the mr-strided panel,
           computes only mre÷W live vectors, unmasked, literal Val(1)/Val(2); full-mr path untouched).
           Result: **m=32 0.97→1.17**, **trsm n=256 0.93→~0.95 (gate boundary), getrf n=256 →~1.02**,
           square-gemm gate holds + several non-aligned n improve, AVX-512 pure gain (n=24/32/40/100 all up,
           no regression). Suite 57/57; clip in the StrictMode GEMM dogfood. Residual trsm/getrf n≤128
           (~0.89) is small-n leaf overhead. **The clip is the general lever for any misaligned-m gemm.**
      2. **Small-n triangular campaign (DELIBERATE — study first).** STUDY DONE (2026-07-02): read
         OpenBLAS's syrk/trmm/trsm Level-3 drivers (memory `openblas-triangular-diagonal`). Key findings:
         OpenBLAS has **NO small-n special path** (size only gates threading; the blocked driver is just
         efficient at small n); its syrk diagonal tile is computed **dense into scratch then scalar
         triangular copy-back** (no masks); trmm/trsm bake the triangle into the PACK (zero strict-tri /
         set-or-invert diagonal) so the dense kernel runs unmasked.
         - **syrk PARTIAL WIN (2026-07-02):** the recursion base wastes 2× flops (`gemm→temp + _add_tri`);
           the packed path (`_trgemm_packed!`, per-microtile diagonal, no waste) beats it from n≈24.
           Added `_SYRK_PACK_CUT` (W==4 → 23, W==8 = `_GEMM_UNPACK_MAX` so unchanged) routing 24≤n≤128 off
           recursion onto packed: **n=32 0.75→0.80, n=128 0.75→0.88** (crossover measured n=12..32; recursion
           still wins n≤20, e.g. n=16 1.08). Suite 7213/7213 on galen. `src/level3.jl`.
         - **DISPROVEN (do NOT re-try): OpenBLAS scratch+copyback diagonal tile.** Implemented
           `_microkernel_tri_scratch!` (dense tile → stack scratch → bounded scalar triangular copy-back)
           and A/B-tested vs the existing masked `_microkernel_tri!` on galen: **EQUAL** (geomean 0.901 vs
           0.902; n=32 0.83 vs 0.79, n=256 0.92 vs 0.94 — all noise). SIMD.jl masked stores on AVX2 are
           already cheap; the masked-store cost was never the bottleneck. Reverted.
         - **REMAINING:** n=32/128 syrk still ~0.80/0.88 (< gate) — the gap is general small-n kernel
           efficiency (packing overhead + tile geometry), not diagonals. syr2k n≤128, symm/trsm small-n
           (trsm n≤256 invL base) still open. Matching OpenBLAS's broadly-efficient small blocked kernel is
           the real (hard) lever.
      3. **Empirically certify Zen4 — DONE (2026-07-02): no regression.** Re-ran full `plots.jl` gate on
         wintermute (Ryzen 5 7640U, Zen4, W=8) and diffed vs baseline: changed ops statistically identical
         (gbmvN 1.34, gemvN 1.20 geomean; W=8 codegen bit-identical). The few sub-gate worst sizes (gemm n=8
         0.82, gemvN n=512 0.95, getrf 0.93) PRE-EXIST in the baseline and are this mobile chip's noise
         floor — not introduced by the phase-1/2 changes.
- [x] Perf guardrails (2026-07-01) — TWO complementary tools + a CI gate:
  1. **Self-regression (`judge`)** — `benchmark/benchmarks.jl` PkgBenchmark suite (SVD + getrf/geqrf/potrf
     + gemm). **CI `perf` job** (`.github/workflows/CI.yml` → `benchmark/judge_ci.jl`) runs
     `judge(PureBLAS, "HEAD", <base branch>)` on every PR and **BLOCKS the merge on a >25% slowdown** vs
     base (coarse tol — GitHub runners are noisy/unpinned; catches gross regressions like the 2× back-
     transform we hit by hand). Graceful skip if the base ref predates the suite. Verified: self=0
     regressions, simulated 2×-slower ⇒ all 14 flagged.
  2. **Absolute OpenBLAS gate** — `bench/lapackbench.jl` [save] [potrf geqrf getrf gesvd] [--sizes …]:
     interleaved-median our/OpenBLAS ratio (≥0.96 gate) + correctness vs LAPACK + per-host ratio baseline
     (`bench/lapack_baseline_<host>.txt`) that flags a ratio DROP even above the gate. Run pinned:
     `taskset -c 2 julia --project=bench bench/lapackbench.jl`.
  `bench/`: bench_gemm/bench_level1/l3bench (manual L1/L3 gate sweeps). Correctness is guarded by the
  7130-item suite + StrictMode. `benchmark/base.json` is a machine-specific local baseline (wintermute).

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
- [x] **Complex GEMM SIMD path DONE (2026-07-02, commits 283af9b, 845486f, 813455f).** Was scalar
      (~0.2×); now split-pack blocked (4-real-FMA MAC, conj via Val signs, interleave-add store,
      vectorized packs, B0 overwrite for beta=0) + an unpacked tiny-n path. Tile W=8 2×4 / W=4 1×6
      (12 accs = exact 16-ymm AVX2 fit). **wintermute (Zen4/W=8) BEATS OpenBLAS at EVERY gate size**
      (zC 1.12–1.49×, cC 1.01–1.48×). galen (Zen3/AVX2/W=4): large-n (96–2048) GATES 0.95–1.05, n=8
      F64 gates (1.18), F32 mid-small mostly gates; F64 mid-small n=16–64 = 0.81–0.92 residual (AVX2
      alpha=1 store fast-path d041349). zgemm in the L3 gate. Full suite 7243/7243 both machines.
      **CORRECTION (freq-locked re-measure): the earlier "n=32 F64 0.90 hardware floor" was a BOOST
      artifact, NOT a hardware floor.** With boost disabled on both boxes: wintermute beats every size
      (F64 1.12–1.53, F32 1.06–1.48); galen n=32 F64 = 1.03 HOT / ~0.93 cold, everything else ≥0.965.
      The kernel is sound — hot (the stated in-place-reps methodology) it BEATS OpenBLAS; the residual
      cold gap is a modest cold-cache effect from complex being 2× the bytes. Rejected-with-measurement
      alternatives (packless 0.76, pack-A-stream-B 0.44, vgather, packB-transpose): memory
      complex-gemm-implemented. **Benchmarking now requires boost OFF ([[dev-fleet]]).** Complex GEMM DONE.
- [ ] **complex-return ABI** for the deferred c/zdotu,c/zdotc symbols (LBT NORMAL vs ARGUMENT retstyle).

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

## LAPACK — LU (getrf) — ✅ CORRECT + GATES THE FULL RANGE n=512–4096 (0.96–1.06×) (2026-06-30)
`src/lu.jl`. Blocked right-looking (= LAPACK dgetrf's algorithm) on PureBLAS trsm!/gemm!. Exact LAPACK
match (factor + ipiv), P·A=L·U ~1e-14, suite 7085/7085. **Gates everywhere — 512:1.00 768:1.06 1024:1.05
1536:1.02 2048:1.01 2560:0.98 3072:0.99 4096:0.98** (+ non-po2 2500–4000: 0.98–1.01). Two fixes (both
"the overhead, not the gemm" — our gemm beats OB at the LU shape even at 4096, 1.03–1.07×):
(1) **explicit-copy pad** (contiguous per-column `unsafe_copyto!` vs `copyto!` on views — the small-n
killer; 512 0.87→1.00); (2) **deferred pivoting** — the in-loop left-block laswp re-touched cold left
columns every panel (O(n²) cache-miss traffic, the large-n killer: decomp n=4000 laswp 99.5ms); fixed by
laswp-ing only the right block in-loop and applying each panel's later pivots to its own columns once at
the end (2560–4096: 0.93→0.97–0.99). nb=48, stride%512 padded, size-adaptive laswp. DISPROVEN: faer
recursive LU; panel-pad/column-temp anti-alias; "blocked on large-n gemm" (it was the laswp). kb:
`pureblas-lu`.
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
### LU residual (512/3072) — grinding tried, at the ceiling (2026-06-30)
Grinding 512 (0.87) and 3072 (0.94) further: tried **panel-pad** (copy mp×pb panel to non-aliasing buffer
per panel) and **column-temp** (copy pivot column to contiguous temp in `_getf2_simd!`) — BOTH backfired
(po2 sizes 1024/2048 crashed to ~0.61). Reason: the whole-matrix pad isn't only for the panel — the LU
**trsm and gemm also operate on po2-strided sub-blocks of A** and need the non-conflicting ld; a panel-only
fix leaves them thrashing. So the whole-matrix pad is required, and its O(n²) copy (~15ms @3072) is the
inherent residual at large n; 512 is small-n overhead (O(n²)/O(n³)). DON'T re-chase panel-only anti-alias.
To gate 3072: a cheaper whole-matrix anti-alias (or overlap the copy); 512: lower fixed overhead. Both deep
diminishing returns. kb: `pureblas-lu`.

## LAPACK — SVD (gesvd!) — ✅ CORRECT (all shapes ~1e-14) + GATES ALL n 96–2048 (valley eliminated 2026-07-01)
Fourth LAPACK routine. `src/svd.jl` (gebrd + bdsqr + driver + blocked back-transform) and `src/svd_dc.jl`
(divide-and-conquer bidiagonal solver, faithful faer port). Two paths in `gesvd!(A; want_vectors)`:
values-only → bdsqr (cheap); vectors → bdsdc D&C (per user's gate decision: oracle = `gesdd`, D&C).
**Correct:** A=U·Σ·Vᵀ ~1e-14, σ vs LAPACK ~1e-15, U/Vᵀ orthonormal, square/tall/wide. Suite covers it.

**Gate (vs `gesdd`, Zen4, interleaved-median): VECTORS gate ≥0.96× at EVERY n 96→1024 (worst 0.970 @168);
VALUES gate ALL n (128 … 2048).** The old small-n VALLEY (144–224 = 0.88–0.95×, worst 0.73× @192) is GONE
as of 2026-07-01 — three changes eliminated it (the whole gap was small-n `bdsdc`; `gebrd` already BEATS
LAPACK 1.2–1.4× there, back-transform beats OB's ormbr):
1. **`_compute_singular_vectors!` restructure** — compute each column's `o` nonzeros contiguously into `vbuf`
   (divisions vectorize), norm O(n)→O(o), single scatter via precomputed `rowidx`; `vm` reuses `dgp*zhp`.
   Fixed n=192 0.73→0.98.
2. **`_SEC_BISECT_CAP` 4→0** (secular finder) — faer's pre-secant bounded bisection (5 iters) is only a
   secant warm-start; secant + the `use_bisection` fallback already guarantee convergence, so CAP=0 saves
   ~4 secular-eq evals/root with ZERO correctness change (stress: clustered/graded/repeated/tiny-gap spectra
   n≤512 all ~1.7e-14). Cleared n=152–224. **Root-finding is ~45% of bdsdc (verified vs LAPACK `dbdsdc` +
   inclusive-count profile) — this is the decisive lever, NOT the ~10% an earlier note wrongly claimed.**
3. **`_SVD_DC_CROSS` 144→128** — with the merge cheap, bdsdc beats bdsqr at 136–144 (0.99× vs 0.92×) while
   n=120 still prefers bdsqr. Fixed n=144.
Also: `f_max` computed only when `last` (dead for non-last roots). See kb `pureblas-svd` for the sweep +
disproven levers (threshold-down, crossover-up were disproven BEFORE the CAP cut, then the crossover optimum
moved once the merge got cheaper — the coupled system is real).

Earlier large-n gating history (2026-06-30..07-01) — **larfg SIMD-norm:**
`_larfg!` used `hypot` in its norm loop → O(n²) Base.hypot in gebrd (the THIRD time hypot-in-a-loop bit,
after the SVD-normalization and givens fixes). SIMD sum-of-squares + sqrt (scaled-hypot fallback on
overflow only) → **gebrd 128 0.80→0.99, 256→1.36, 384 0.85→1.18**; lifted SVD VECTORS 384→gate and
VALUES→gate everywhere.
Two 2026-07-01 wins: **(1) gemm `transA='T'` unpacked path** — the back-transform's `W=VᵀC` (transA='T')
forced blocked+PACKED (packs the large C) → 0.58× @256; added a transA='T' unpacked route in `gemm.jl`
(SIMD-transpose A→Aᵀ via `_tblk!` into scratch, then the unpacked N·N kernel, no B-packing). Cross-cutting:
gemm-T ≤448 now 0.98–1.16×; lifted SVD 384→0.92, 768→gate. **(2) po2-pad the back-transform accumulator** —
`VᵀC`'s C is the SVD's own UA/Vmat; at n%256==0 its column stride thrashes ≤2 L1 sets (gcd(n/8,64)≥32).
Pad the leading dim +8 (view into a padded buffer, no per-gemm copy) → **256 0.85→0.98, 512→1.18, 768→1.00
GATE**. Grind (2026-06-30..07-01), driven by isolating `bdsdc!`
vs `LAPACK.bdsdc!`: the unlock was the **`hypot`-in-a-loop singular-vector normalization** → SIMD
sum-of-squares (**isolated bdsdc now 1.22× — BEATS LAPACK dbdsdc**); plus `_mkgivens`/`_givens` hypot→sqrt,
bdsqr scale-to-O(1), `@simd` `_secular_eq`, SIMD Givens, crossover→96, **gebrd `_BRD_NB`→16 + direct `_gemv!`
kernel in `_labrd` + decoupled back-transform `_BT_NB`=32**. (Correction: an earlier note here called the root
finder "a red herring, ~10% of bdsdc" — that was WRONG; direct measurement showed ~45%, and the CAP=0 cut
above is what closed the small-n valley.) See kb `pureblas-svd`.

Three layers, the proven faer recipe (port the irreducible kernel, drive the blocked level with PureBLAS):
1. **gebrd** (`gebrd!`/`_labrd!`) — blocked two-sided Householder bidiag (LAPACK dgebrd: dlabrd panel +
   2 trailing `gemm!`). **Matches LAPACK (1.01× @512)** after the strided-row fix (route every row-vector
   of A/X/Y through a contiguous buffer — the gemv kernels already match OpenBLAS; the 3× gap was the
   strided access, same disease as formP). m<n handled by transpose.
2. **bidiagonal SVD** — `bdsdc!`/`_dc!` (D&C, faer `bidiag_svd.rs`: secular-equation root finder, deflation
   43/44, rank-one merge `compute_svd_of_m`, augmented (n+1)×(n+1) U) for vectors; `bdsqr!` (Golub-Kahan
   implicit-QR) for values-only. D&C is compute-bound on the secular solver (the small/mid-n gate limiter —
   LAPACK dlasd4's constant factor). Serial post-order ⇒ one shared scratch-buffer set across all nodes.
3. **back-transform** — `_apply_reflectors_left!`: blocked compact-WY (dlarft + dlarfb via `gemm!`) applies
   the gebrd reflectors directly to the bidiagonal singular vectors, FUSING form-Q/P + combine into one
   BLAS-3 pass (replaced the old gemv-based `_form_Q!`/`_form_P!`). dlarft writes only T's upper triangle —
   zero-init T (the full `gemm!(Y=T·W)` reads the lower triangle).
**Remaining:** re-run the gate per-machine on the fleet (Zen3/Zen5/M5). Float64 vectors path only; generic
`T<:Number`/AD SVD deferred. kb: `pureblas-svd`.

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
**L2 rank updates DONE + GATED (2026-07-01):** spr/spr2 (symmetric packed rank-1/2, real) + hpr/hpr2
(Hermitian packed, complex). `src/level2_packed.jl` — per-column contiguous packed-column axpy reusing
`_axpy_simd!` (real SIMD path) + generic scalar (complex / AD). Correct vs OpenBLAS `spr` and a dense
oracle ~1e-16 (upper/lower, s/d/c/z), ForwardDiff-traceable. **spr GATES 1.02–1.09×, spr2 1.01–1.12×
(n=256–4096)**; hpr/hpr2 correct-but-generic (complex SIMD deferred to M5, like the other complex ops).
Native API mirrors `ger!`: `spr!(α,x,AP;uplo)`, `spr2!(α,x,y,AP;uplo)`, `hpr!`/`hpr2!`. **BLAS L1/L2/L3
now complete.** Next: LAPACK breadth (eigensolvers), or M4/M5/M6.

## Small-n gate campaign — ALL L3+LAPACK ops gate 0.96× at n=2–2048 (2026-07-02) ✅ CONTRACT FULFILLED

New standing requirement (user, 2026-07-01): every BLAS-3 and LAPACK routine must gate ≥0.96× OpenBLAS at
EVERY size n=2…2048 — "smaller sizes usually indicate hidden unresolved overheads." Executed overnight
(commits b75b3ff, e5375ff, 86b1db8, c6a5b28, 27e7ba6; suite 7213/7213 throughout).

Final grid (typed harness, interleaved reps+reset medians, Zen4 unpinned — ±0.02 wobble):
gemm/symm/syrk/syr2k/trsm(L,R)/trmm-L: gate at every size, most cells 1.1–3×.
trmm-R: NOW GATES EVERYWHERE — the final cell (1024, was 0.94) closed by PRE-PACKING all of B in
`_trmm_packedR!` (packing doubles as the in-place capture: the separate copy+repack was ~2–3% of runtime).
Final trmm-R row: 2.70 1.45 1.02 1.08 0.96 1.13 1.08 1.06 0.98 0.97 0.97. potrf/geqrf/getrf/gesvd: gate at every size (geqrf tiny-n
2–4×, gesvd n=4 0.40→1.35). **Certify the at-gate cells (trmm-R 512/1024/2048, getrf/gemm 2048, symm 512)
with `sudo bench/cpufreq_lock.sh pin 4500` — the overnight box was thermally wobbling.**

What fixed it (catalog in kb `pureblas-l3-syrk-syr2k-symm`): const-dispatch scratch lookups (IdDict get =
130 ns), cached per-call workspaces (geqrf 5-matrix, gesvd back-transform 32 KB), `_gemm_core!` (kwarg-free
dispatch core) for all internal gemm calls, `_trmm_small!` (materialized-M + K-trimmed unpacked microkernels,
in-place dependency-ordered), syr2k's transpose identity (one gemm instead of two), potrf pad guard %512→%256
+ per-column pad copies, `@simd ivdep` pack_B (a wide-vload transpose pack DISPROVEN: −25% geqrf via
store-forwarding stalls), packed single-pass trmm-R (`_trmm_packedR!`).

## Fleet-gate certification — 3 boxes, per-µarch (2026-07-04) ✅

The per-machine gate (Zen4 dev + Zen3 AVX2 + **Zen5 native AVX-512, added 2026-07-04**) is now certified at
locked freq (boost off). **Finding: the residual profiles are DISJOINT across µarch — tuning does not
transfer, which is exactly why the gate is per-machine.** Full `plots.jl bench` on each; caches
`bench/plots_data_<host>.txt`, per-ISA SVGs.

- **wintermute (Zen4 mobile, AVX-512 double-pumped 256-bit):** all-green bar small-n 7640U-noise dips.
- **galen (Zen3, AVX2):** all-green except documented ceilings — potrf po2-conflict (0.88–0.95; the copy is
  the gap, `d61b332` lower-triangle pad-copy cleared n=2048 & lifted the rest), trmm n=8 (materialize
  tiny-kernel wall), zgemm 0.95–0.958 (16-reg tile), gemvN-512 0.94 (L3 bandwidth), iamax addr-noise.
- **neuromancer (Zen5, native AVX-512):** clears EVERY AVX2 ceiling (potrf 1.40, zgemm 1.14, trmm 1.09,
  trsm 1.65) but surfaces NEW L2 gaps: **ger n=2048/4096 = 0.71/0.82** (write-bandwidth-bound — likely OB
  non-temporal stores vs our cached writes; reproduced tight on a quiet box, REAL), gemvN-256 0.87, gemm-32
  0.91, symm-256 0.94. Correctness bit-exact (gemm 0, potrf 2.8e-14, zgemm 7.6e-16). Detection auto-adapts
  (W=8, `_INTEL_AVX2=false`). **Next AVX-512-AMD residual pass = ger-first** (its own campaign; the AVX2
  campaign does not carry over). See memory `fleet-gate-snapshot-locked`.

This satisfies M7's stated prerequisite ("start after the Zen3/Zen5 fleet gate runs").

## M4 — multithreading (DEFERRED by user — do not start until explicitly requested)

Parallelize the gemm jj-loop, threshold-gated (small sizes stay serial). Per-host tuning. This is
for **absolute throughput / scaling across cores** — NOT for closing an OpenBLAS gap: single-thread
`dgemm` is already at parity (geomean 0.999×). (Earlier note claimed large-n was "single-thread
L2-bandwidth-bound, needs threading" — that was wrong; it was scalar packing, fixed by SIMD pack_A.)
**Standing instruction (2026-06-28): defer ALL multithreading requests until later — keep everything
single-threaded for now.**

## M5 — complex SIMD + multi-ISA dispatch (IN PROGRESS)

Complex-SIMD for complex kernels (was correct-but-scalar → the biggest beat-OpenBLAS opportunity).
Runtime AVX-512/AVX2/NEON dispatch in one build (so a single artifact runs optimally across the fleet).

### Progress (2026-07-04/05)

Complex GEMM family was done earlier (gemm/hemm/herk/syr2k/her2k gate). This session extended the SIMD
complex path across L1 + much of L2. Two portable interleaved-`Vec{2W}` idioms (NO x86 intrinsics — SIMD.jl
`fma`/`muladd` suffice): **swap-pairs** for scalar×vector (scal/axpy) and **interleaved-product** (`x·y`,
`x·swap(y)`, deinterleave only the accumulators) for dot. AVX2 lesson (twice): avoid per-iteration
deinterleave — it starves Zen3's shuffle ports.

- **L1 DONE — all gate both boxes** (commits 5c4e217, c148aa8, 586ced4): nrm2/asum (real-reinterpret,
  nrm2 4–6×), scal/axpy (swap-pairs), dotu/dotc (interleaved-product). Each in @verify_strict + dogfood.
- **L2 gemv N/T/C** (b345b5f): T/C = per-column complex dot (reuses L1 dot) → gate AVX-512 (1.11). N =
  column-panel driver → gates AVX-512 small-mid (0.99–1.02), large-n memory-ceiling (~0.93, mirrors real
  gemvN). **AVX2 gemvN 0.5–0.7 (below gate): shuffle/throughput-bound TUNING residual** — a split-`Vec{W}`
  variant measured WORSE; fma primitives suffice (not intrinsic-blocked). TODO: AVX2 gemvN tuning.
- **L2 geru/gerc DONE — gate both** (3b62394): per-column complex axpy — galen 0.97–1.31, wintermute 1.03–1.32.
- **L2 hemv DONE — BEATS OB both** (77d8873): fused axpy+conj-dot column kernel (one A-column read). galen
  1.28–1.87×, wintermute 1.21–2.01×.
- **L2 trmv/trsv DONE** (e676b3f): per-column axpy(N)/dot(T/C) reuse, all 48 combos correct. trmv gates
  both (galen 0.86–1.68), trsv gates AVX-512 (0.96–1.03), AVX2 0.84–0.94 (sequential solve + complex divide
  — AVX2 residual, but improves scalar 0.57). **⇒ complex L2 essentially complete** (symv skipped — no
  standard LinearAlgebra complex-symv oracle).
- **L3 ctrmm/ctrsm side-L DONE — materialized-gemm bases** (9a5d3b1, b5ad32c): the complex bases re-read A
  n times (trmv/trsv-per-column); replaced with the real bases' strategy — ctrmm: materialize op(A) triangle
  once (`_mat_tri!`+conj) then B:=M·B; ctrsm: invert once via the now-generic `_trtri!` then B:=op(M⁻¹)·B —
  both via the gating SIMD complex gemm (reads A once). Correct nfail=0 all combos both boxes, 0-alloc warm.
  ctrmm wintermute n=64 0.26→0.79 … 1024 0.97; ctrsm n=512 0.75→**0.97**, 1024→**1.00** (gate). galen
  improved but below gate (complex-gemm AVX2 ceiling + trtri/materialize+copyback overhead on small-n).
  NOT in strict dogfood (complex L3 scratch = keyed workspace fallback, 0-alloc but not const-dispatched).
- **L3 ctrmm/ctrsm side-R DONE** (c7ce41a): materialized bases mirroring side-L (B:=B·M / B:=B·op(M⁻¹)),
  nfail=0 all combos. **⇒ triangular complex L3 complete, both sides.**
- **csymm/csyrk/csyr2k ALREADY GATE** (measured csymm 1.08, csyrk 1.05) — they're built on the complex
  gemm/packed kernels. ⇒ **the entire complex L1/L2/L3 surface is now SIMD.**
- **REMAINING (residuals only):** ctrmm/ctrsm **small-n** (64–256, trtri/materialize+copyback overhead) and
  **AVX2** (below gate — complex-gemm ceiling; ctrmm n=512 in-place microkernel would avoid the copyback,
  a deeper rewrite); gemvN/trsv **AVX2** tuning. These mirror the real ops' small-n/AVX2 hard spots.
- **NOT STARTED:** runtime multi-ISA dispatch (AVX-512/AVX2/NEON in one artifact) — detection is
  compile-time per-build today. That's the other half of M5's title; a separate feature.

**M5 core goal ACHIEVED:** the complex surface (was the biggest beat-OpenBLAS opportunity) is SIMD across
L1/L2/L3 — gates/beats OpenBLAS broadly (esp. AVX-512: hemv 2.0×, nrm2 4–6×, most ops ≥1.0×). Remaining is
residual tuning + the multi-ISA-dispatch feature.
- **AVX2 TUNING RESIDUALS:** gemvN (0.5–0.7), trsv (0.84–0.94), ctrmm/ctrsm — shuffle/latency-bound on Zen3;
  fma primitives suffice (not intrinsic-blocked).

Runtime multi-ISA dispatch (below) still not started — detection is compile-time per-build today.

### Original design

**Design ready (OpenBLAS study 2026-07-02, memory `openblas-complex-simd-design`):** pack SPLIT
(de-interleave `[Ar…][Ai…]` once at pack time, NOT interleaved+shuffle — the x86 `vmovsldup` trick
isn't portable); one complex MAC = **4 real FMAs** on `Vec{N,T}` Cr/Ci accumulators; the 4 conj
variants (N/C×N/C) are ONE kernel with the two imag-term signs passed as `Val` params (fma/fnma);
**2 accumulators/tile** → halve the real tile dims for register pressure. Skip 3M/Karatsuba for v1.
Keep the generic scalar path as oracle + AD path.

## ⓘ OpenBLAS-source-informed opportunity scan (2026-07-02)

Studied OpenBLAS `develop` for beat-OpenBLAS wins. Prioritized:
1. **Complex SIMD GEMM (M5) — the standout.** Complex is scalar today; design above is ready. High
   impact (whole complex L3/L2/L1 surface), high effort.
2. **Tall-skinny GEMM — NOT a gap (verified, dismissed).** Hypothesis: our `_use_unpacked` gates on
   `max(m,n,k)` while OpenBLAS gates on volume `M·N·K≤1e6`, so we'd pack tall-skinny that OB runs
   direct. MEASURED on galen: PureBLAS already **beats** OB 1.14–2.64× on 4096×8×8-type shapes *despite*
   packing → no action (a switch to volume-gating might help further but the gate is already met).
3. **Small SQUARE gemm n=8 (~0.82 Zen4).** We already skip packing there (`max=8≤128`), so it's
   dispatch/call overhead or mobile-chip noise, not packing — low ROI; revisit if it blocks a gate.
4. **Operation fusion via Mode 2 (structural beat OB).** The native API (no ccall boundary) can fuse
   `α·A·B+β·C` with a following op, gemm+activation, etc. — OpenBLAS can't fuse across its opaque
   C-ABI. Real advantage, but API-design work; schedule when there's a consuming use case.

## M6 — AD rules

`PureBLASChainRulesExt` (weakdep) + Enzyme rules so Mode 2 supports reverse-mode through the
in-place ops. (Native path is already ForwardDiff-traceable today.)

## M7 — GPU backend (CUDA-first; gate vs cuBLAS, CUTLASS as structural reference)

Requested 2026-07-01 ("extend PureBLAS to run on GPU and match cuBLAS and CUTLASS"). Planned
2026-07-02; **start after the pinned certification + Zen3/Zen5 fleet gate runs** (user decision).

**Hardware:** GeForce RTX (consumer) — NOT wintermute (verified: no NVIDIA device/driver, only the
AMD Phoenix iGPU), so GPU dev/benching happens on the box that hosts the card. Consumer FP64 runs at
1/64 rate — the FP64 gate is still fair (cuBLAS pays the same rate), but tensor-core paths are where
the CUTLASS fight is.

**Stack (no-Python-compatible, all pure Julia):** CUDA.jl — kernels are Julia source compiled via
GPUCompiler→LLVM→PTX; cuBLAS ships in CUDA.jl's artifacts and its `CUBLAS` wrapper module is the
correctness oracle + gate denominator (exact analogue of OpenBLAS-via-LinearAlgebra on CPU).
Tensor cores via CUDA.jl's WMMA API. Prior art to MINE, not depend on: **GemmKernels.jl** (JuliaGPU;
typically 50–80% of cuBLAS, sometimes parity — below our gate, but its layout/config abstractions are
proven) and **cuTile.jl** (NVIDIA's 2025/26 tile-programming model for Julia, a CUTLASS-analogue —
evaluate at G2 kickoff; if it reaches the gate it may replace hand-rolled WMMA scheduling).

**Delivery: package extension `PureBLASCUDAExt`** (weakdep on CUDA, like the planned ChainRules ext).
Core package stays CPU-only with zero new hard deps. Dispatch on `CuArray` methods of the SAME native
API (`gemm!`, `axpy!`, …) + a `CUDABackend` for the contract layer. **Gate: ≥0.96× cuBLAS, per-GPU**
(extends the per-machine rule; per-GPU baseline files like per-host now). Timing = CUDA events /
`CUDA.@elapsed` with explicit sync + warmup — never CPU timers around async launches.

**Gate scope (user decision): all four tiers** — FP64 SIMT, FP32 SIMT, TF32 tensor-core
(FP32-in/out), FP16/BF16 tensor-core (mixed-precision accumulate). SIMT first (proves the
structure), tensor cores second (the CUTLASS numbers).

Phases (same de-risk logic as M1/M2 — plumbing on easy kernels first, then the flagship):
- **G0 toolchain:** CUDA.jl on the RTX box, pick the cc target, repo CI story (GPU tests can't run
  on GitHub runners — local/self-hosted or tagged-skip).
- **G1 vertical slice:** extension + `CUDABackend` + BLAS-1 (axpy/dot/nrm2/scal…) as simple CUDA
  kernels + a naive gemm. Correctness vs `CUBLAS.*`; event-based bench harness; **gate BLAS-1**
  (bandwidth-bound → parity ≈ free, proves extension load, dispatch, harness, per-GPU baselines).
- **G2 flagship gemm:** CUTLASS's hierarchy is the structure to match — threadblock tile in shared
  memory (double-buffered async-copy pipeline) → warp tile → MMA/FMA fragment. FP32/FP64 SIMT first,
  then WMMA TF32/FP16/BF16. Tile shapes autotuned per-GPU (Preferences knob, like CPU widths).
- **G3 L3 breadth:** syrk/symm/trmm/trsm over the device gemm. ⚠ The CPU L3/LAPACK **drivers do NOT
  port as code** — they lean on host scratch consts, `unsafe_copyto!` pad tricks, per-column scalar
  loops (poison on GPU). The *structure* (recursion shapes, triangle-aware tiling, K-trim) ports;
  the drivers are rewritten device-side with on-device workspaces.
- **G4 LAPACK:** MAGMA-style hybrid (panel factorization on CPU, trailing update on GPU), gate vs
  cuSOLVER — its own sub-project, after G3.

**Portability note (fleet: future Mac M5 → Metal.jl, AMD → AMDGPU.jl):** kernels stay CUDA-native —
a portability layer (KernelAbstractions.jl) typically taxes exactly the few % the gate lives in.
Acceptable for bandwidth-bound L1/L2 if measured free; the tiling structure + test suite are what
Metal/AMDGPU reuse later, not kernel code. GPU parallelism does not touch the CPU
no-multithreading standing rule (that covers CPU threads).

## Later

ARM/aarch64 trim build for the Mac M5 (cross-compiled .so/.dylib). LAPACK surface. SparseArrays
interop; CHOLMOD / sparse Cholesky.

### Wishlist

- **Pure-Julia reimplementation of BLASFEO kernel ideas.** BLASFEO (github.com/giaf/blasfeo, BSD-2)
  hand-tunes small/medium-n dense kernels in asm and reaches MKL-level GFlops — *but* its headline speed
  comes from **panel-major storage** (pre-packed, no runtime packing), which PureBLAS's column-major
  drop-in contract can't adopt. What transfers is the **kernel STRUCTURE** (register-block shapes, loop
  order, accumulator counts, prefetch placement, small-n no-pack microkernels, its strong `dtrsv`/`dgemv`
  level-2 kernels) reimplemented in SIMD.jl/LLVM — our small/mid-n overhead regime is exactly its domain.
  License is a non-issue: BSD-2 permits derivative work with attribution, and reimplementing the
  *algorithm* isn't even a copyright question (ideas aren't copyrightable) — **attribution to BLASFEO is
  the right thing to do regardless, and the user has OK'd giving it.** Pull relevant pieces into PureBLAS
  *now* wherever a Fable-mapped technique measurably closes a gate on locked HW; the broader systematic
  port (or a dedicated panel-major sibling package for the embedded-optimization use case) is a future
  project. Reference-only — never a C/asm dependency (pure-Julia). See the Fable BLASFEO technique-map.
