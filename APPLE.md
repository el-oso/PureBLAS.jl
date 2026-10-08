# Apple silicon — work plan for the dedicated agent

You own Apple silicon. The box is an **M6 mac-mini**. A second agent owns the AMD fleet
(Zen3 galen / Zen4 wintermute / Zen5 neuromancer) and the cross-architecture multithreading
campaign; the two meet only through `master`.

**The campaign content is already written: `ROADMAP.md` "SME across the full surface (Apple
Silicon)".** It holds the per-cell reference rule, the three SME mechanisms and their measured
rates, the phase order, the current medians against Accelerate, and three rules earned from shipped
defects. Read it first and treat it as authoritative. This file adds only what that section does
not cover: the **multithreading** dimension, the order the two passes run in, and the coordination
boundary with the MT campaign.

---

## Two passes, in this order

**Pass 1 — SME as the stand-in for multithreading.** One SME coprocessor beats twelve threaded cores
on this box (gemm n=4096 Float64: **503 GFLOP/s owned vs 301 split vs 177 threaded-NEON**), so on
Apple the right answer to "make it parallel" is usually "make it reach SME". Pass 1 is therefore the
whole of `ROADMAP.md`'s campaign, and it is where the value is.

**Pass 2 — NEON-MT, for generic aarch64.** Older Apple parts and non-Apple ARM have no SME, and
every cell SME declines today runs single-threaded NEON against a twelve-core reference. Pass 2 makes
the existing worker pool work there. It is not urgent for the M6 and it must not regress pass 1.

---

## Step 0 — three blockers, and none of them is a kernel

All three have landed; a threaded Apple cell is expressible and one exists. They are kept here
because each records a trap that is invisible until it bites.

- [x] **`accelerate_mt` exists.** `_REF_MT_ARMS` in `bench/plots.jl` carries it, and a threaded Apple
      gate is expressible.
- [x] **Accelerate ignores `BLAS.set_num_threads`, and `bench/plots.jl` handles it.** It reads
      `VECLIB_MAXIMUM_THREADS` **once, at first use**, so `_VECLIB_NT` is decided from the arm
      selection and exported before any forward into the library. Getting this wrong does not error;
      it silently measures the wrong thread count. The two Accelerate arms therefore cannot share a
      process, and a run requesting both is refused with the mechanism named.
      ⚠ Related trap already paid for once: the Accelerate dylib needs
      `suffix_hint = "\x1a$NEWLAPACK$ILP64"`. Without the leading `0x1A`, LBT falls back to LP64 and
      `dgemm_64_` **returns zeros rather than raising**.
- [x] **`openblas_mt` is measured across every group.** `bench/mt_data_neon_mac.home.txt` holds 937
      cells with all three threaded arms — `accelerate_mt`, `openblas_mt`, `pb_mt` — so the per-cell
      reference rule has both reference arms it needs.
      ⚠ The cache header records `freq=0kHz base=0kHz boost=-1`: this platform reports no pin, so
      every figure drawn from it is directional. See the caveats section below.

---

## Pass 1 — SME

Follow `ROADMAP.md` phases 1→3 (real → complex → dual). Additions from the MT analysis:

- [ ] **Keep `_sme_owns` declining the worker pool.** `src/blas3/sme_kernel.jl:775-785` and the test
      at `test/gemm_tests.jl:405-412` encode this: the matrix unit is shared by the cluster, so
      splitting queues workers behind one unit. The 503 / 301 / 177 measurement is the evidence.
      **A threaded gate on Apple gemm therefore compares serial SME against threaded Accelerate.**
      That is the correct physics, but it must be stated wherever the number is published.
- [ ] **SME eligibility is decided at the threaded entry, never inside a chunk.** `_sme_owns`
      (`sme_kernel.jl:784`) mirrors `_strassen_owns` for exactly this reason — by the time a chunk
      reaches `_gemm_core_body!` the split has already happened.
- [ ] **`_gemv!`'s first branch is `_sme_gemv_eligible`.** A threaded gemv-N must decline for the
      same reason gemm does. The MT campaign's Phase 2 threads gemv-N on AMD; that change must not
      capture SME-eligible calls here.
- [ ] **SME covers Float64 gemm, BOTH gemv arms and `ger`, plus four BLAS-1 kernels.** In
      `src/blas3/sme_kernel.jl`: gemm, gemv-N, gemv-T (whose ZA-resident and multi-by-multi inner arms
      split on `_SME_GEMVT_RESIDENT_MAX`), and `ger`. In `src/blas1/sme_l1.jl`: `dot`, `asum`, `axpy!`,
      `scal!`, with `src/blas1/sme_l1_cf.jl` holding their `@cfunction` trampolines because a constant
      `@cfunction` binds its callee at method-definition time. `symv` and `trmv` reach SME through
      their own cuts in `src/blas2/level2.jl`. Float32 is NOT covered anywhere; extending it is
      pass-1 work.
      SME is Float64-only (`FEAT_SME_F64F64`), but complex and dual decompose into *real* products,
      so they inherit SME iff those products reach an eligible call — routing, not kernels.

---

## Pass 2 — NEON-MT

- [ ] ⛔ **THREADING `symv`, `trmv` OR `trsv` NEEDS THE gemv-T SCRATCH PREFIT IN THE SAME CHANGE.**
      `_SME_GEMVT_SCR` (`src/blas3/sme_kernel.jl`) is a per-thread strip grown by `_ws_grow!` inside
      `_sme_gemvt_cabi`. It is safe today only because **SME eligibility is decided at the threaded
      entry and never inside a chunk, so every SME route declines the worker pool before a worker
      exists** — `_gemv!` tests `_sme_gemvt_eligible` ahead of its thread seam, `_symv_split!` is
      selected ahead of symv's, and `trmv`/`trsv` fork no task. Measured: symv at n=2428 with four
      workers available dispatches no pool. Thread any of those three and the growth happens inside a
      published job, where the driver spins rather than yields and so reaches no GC safepoint — a
      swallowed worker exception or a silent hang, by interleaving. The fix is a driver-sized,
      worker-indexed strip prefit before publishing, as `_trmmr_prefit!` does. A different owner does
      not fix it: the growth is the hazard, not the ownership.

- [ ] **The pool has no architecture guard anywhere.** No `isapple`/`aarch64`/`Sys.ARCH` in
      `src/blas3/gemm.jl`, `src/arena.jl` or `src/workspace.jl` — it is plain Julia tasks and should
      run as-is. Verify that before assuming it needs porting.
- [ ] **`_GEMM_MT_WORK` is x86-derived and cannot be trusted here.** It is built from
      `_MT_JOIN_NS_MEASURED = 616` ns at 2796 MHz — a single Zen4 recording — and every threaded
      admission predicate divides by it. The MT campaign's Phase 0 replaces it with an **on-host**
      fork-join measurement (`OncePerProcess`, `@static if isnothing(pref)`-gated so a trimmed build
      never benchmarks). Take that when it lands rather than deriving a second Apple-only constant.
- [ ] **Re-check every cutoff after any large speedup.** `ROADMAP.md` rule 3 — every cutoff derived
      against a 44 GFLOP/s gemm is suspect now that gemm reaches 500.

---

## Measurement caveats specific to this box

- **There is no frequency locking on Apple.** `bench/fleet_freqlock.sh` is `amd_pstate` sysfs only,
  and `FreqLock.lock_state` returns `(0, 0, -1)` where cpufreq does not exist. Apple numbers are
  **directional, not gate verdicts** — say so wherever they are published. Measured unlocked spread:
  0.04% on the anchor, 0.3–1.7% on gemm cells, so the noise is small but the guarantee is absent.
- **`bench/probe.sh` gives `MASK=""` for an unknown host** — pinning is `taskset`, which is Linux.
  A thread-count-sensitive measurement here has no affinity control.
- **`bench/apple/` holds only a `Project.toml`** (the AMD-free bench env). No sync script, no lock,
  no Apple-specific tooling. PR #4 (`bench-portable-references`) generalized `plots.jl` and
  `probe.sh` for this box — build on that, do not re-fork them.
- **Two harness bugs were found on this path and fixed**; both may still bite elsewhere. `_L1REP`'s
  reps loop measured a warm buffer (a `cold` flag was added, and the fix is **unconfirmed on x86** —
  Zen3's 32 MiB L3 is the candidate). And `BLAS.set_num_threads(1)` not constraining Accelerate, per
  Step 0. Write-ups: `docs/src/performance_apple.md:58-100`.

---

## Guarantees — identical on every architecture

These are not negotiable and they are not softened by SME being a coprocessor:

1. **Bitwise invariance across thread counts** (CLAUDE.md req #11). An SME-eligible call that
   declines the pool satisfies this trivially. One that *takes* it must split a dimension no
   accumulation chain crosses, route every size-keyed predicate from the **undivided** problem, and
   make the **lost-claim fallback compute the same bits as the chunk**. A threaded `syr2k` once
   returned a wrong answer because a chunk hardcoded a kernel instead of asking serial's route.
2. **Allocation-free** (req #10) outside the arena and workspaces.
3. **StrictMode/StrictModeTest gates the build.** Ship every new guarantee with a **positive
   control** — `@test_noalloc` records nothing on success, so a vacuous item and a passing one look
   identical.
4. **Measured, or it is not done** (req #9). A ratio, or the words "SPEED UNMEASURED".
5. **Dual must keep working.** `sme_tests.jl` currently *asserts* Dual takes the generic path, so
   admitting Dual to SME means rewriting that contract deliberately, not discovering it broke.

---

## Coordination with the MT campaign

The other agent is changing these; **pull before touching them**:

| area | what changes |
|---|---|
| `bench/gatecrit.jl` | gains a threaded criterion (`max` over the `*_mt` arms). Use it — do not open-code a second one |
| `bench/gate_gaps.jl`, `coverage_*.jl`, `cellratios.jl`, `failing.jl` | taught to read `mt_data_*` caches |
| `cache_staleness.sh`, `check_arm_anchors.sh`, `check_clock_outliers.sh` | extended to the mt caches |
| `check_arm_clocks.sh` | threaded-window throttle detection (Linux-only; Apple keeps its "unchecked" degradation) |
| `src/blas1/`, `src/blas2/` | threaded entries for axpy/scal/dot/asum, then ger/gemvN/gemvT/symv |
| `_GEMM_MT_WORK` | replaced by an on-host join calibrator |

Things the MT campaign will **not** touch, so they are yours alone: `src/blas3/sme_kernel.jl`,
`bench/apple/`, `docs/src/performance_apple.md`, and the Apple caches.

**The shared hazard is `_gemm_threaded!`.** The MT campaign widens its `T <: BlasReal` bound to admit
complex. `test/gemm_tests.jl:430-450` reads the admitted type list **out of the method signature by
reflection**, so that test widens in the same edit — by design. If you add an Apple-specific entry,
expect that test to pick it up and require a reproducibility proof for it.

---

## Working rules

- **Iterate through `bench/probe.sh`** — 6m33 cold vs 9s warm. A **new pool job kind cannot be tested
  in a live Revise session**; the parked workers run the old compiled chunk body. Restart it.
- **Sample every `n` before deriving a cut, not every other one.** `_SME_MIN = 3*MR` came from a
  2-step sweep and regressed gemm@50 by 38%.
- **Routing before kernels.** Every large miss found on this path so far has been a predicate, not
  arithmetic.
- **No commit during a sweep** — `plots.jl` stamps each group from `git rev-parse`, so a mid-sweep
  commit splits one sweep across two hashes. **No `src/` edit while a run is in flight.**
- **Sync the fleet with git, never rsync** (`bench/fleet_sync.sh`). rsync copies the code but not its
  identity, and the cache header then lies about which commit produced the numbers.
- **Fable reviews adversarially at the end of each phase**, before it is committed.

- **There is no core pinning either, and `bench/probe.sh` is right to leave this host unpinned.**
  macOS exposes no `sched_setaffinity` equivalent that binds a thread to a core, so `probe.sh`'s
  `*) MASK="" ;;` default is the correct behavior here rather than a missing entry. Worker placement
  across the three core tiers — 2 "Super" and 4 "Performance" cores sharing 20 MB of L2, 6
  "Efficiency" cores on 8 MB — is a request, not a guarantee, so a threaded Apple cell must carry its
  spread rather than a single ratio, and self-speedup is not a quantity this box can report honestly.

- **`bench/fleet_sync.sh` does not cover this host** (`BOXES_ALL=(galen neuromancer)`), so the Apple
  cache is carried by hand. The caches are untracked, so nothing here reaches the AMD fleet's agent
  except through a committed file.

- **An Apple instrument can be validated for DETECTION but never for linearity.**
  `/usr/sbin/taskpolicy -b` moves work to the Efficiency cluster and gives a repeatable 5.4x
  degradation (single-threaded `asum` on 8 MB: 204.0–204.2 GB/s normal, 36.8–38.4 background, the
  nominal arm reproducing to 0.1%), which is enough to confirm an instrument notices a slower machine
  — the half a broken probe fails while passing its one-sided test. But there is nothing between:
  `-c utility` and `-t 0|1|2` all read 204.0–204.2 unchanged. Two levels, 1x and 5.4x. So no
  "tracks an induced pin within two points" calibration is possible, and because the fault changes
  **which core runs** rather than the clock, it validates a throughput instrument and nothing that
  claims to read a frequency.

---

## How to measure on this box

Each rule below cost a withdrawn number. They are here because the knowledge base they were first
written into is not a git repository, so nothing in it reaches the AMD fleet's agent or survives a
fresh checkout.

- **A figure is a property of the HARNESS SHAPE as much as the kernel, and the gap is large.** Timing
  one operand reused across samples against building a fresh one in Chairmarks' setup position — what
  `sweep_heavy` does — gives, for the same gemv-T kernel on the same 4 MB operand: 550.1 against
  297.0 GB/s on the SME arm, 117.7 against 108.9 on the NEON arm. A pool of 8 pre-generated operands
  cycled round-robin reads 135.1, the SLOWEST of the three, because the pool itself does not fit in
  cache — the careful-looking option is the coldest one. **Size any operand pool against the cache.**
- ⛔ **A RATIO of two kernels is not shape-invariant and the error CHANGES SIGN.** SME against NEON,
  ratio taken inside each shape: the probe shape overstates the faster kernel's lead by up to 2.59x
  below 0.6x L2 and understates it by 0.75x above 0.8x, flipping near 0.7x L2. So no correction
  factor exists, and **a probe-measured ratio is not a statement about a gate cell** until it is
  re-measured in the shape the gate uses. State a band position in units of L2, not in MB.
- **The throughput mode belongs to the ALLOCATION, not the process or the thread.** Eight fresh
  operands in ONE process span 2.8-3.4x at 14 MB. Repeated timing of one operand gives a 1-3% spread
  that reads as precision and is not. A shared file mapping across processes removes the spread
  entirely (1.8% over 20 draws, zero degraded) but sits ~3% below the fresh fast mode, so it compares
  kernels and must not be mixed with fresh-operand numbers in one table.
- **Choose the draw count from the NOISIER arm, and print the raw draws before writing a conclusion.**
  A five-draw comparison of two lda values read 1.04x and would have closed a live item; at fourteen
  draws the distributions did not overlap and the answer was 1.16x. One tight arm against one noisy
  arm reads as a null result rather than as an undersampled one.
- **Measure ONE CONDITION PER PROCESS when conditions differ in allocation size or count.** Position
  within a process moves a reading in EITHER direction — the same lda read 1.26x slower measured
  fourth than first, and an in-process size ladder both depressed a 16 MB cell by 1.11x and INFLATED
  24 and 32 MB cells by 1.13x and 1.08x. Measured one per process, this box is essentially noiseless
  below capacity: four draws at 4, 8, 12 and 16 MB are identical to one decimal.
- **Check the instrument reaches the mechanism.** Failures of this class produce a readable, plausible
  number about something other than the question, and none of them errors: a replica passing
  triangular flags as literals where the real path passes runtime Bools (overstated a component gain
  threefold); `const UP = ARGS[2]=="1"`, which is still a compile-time constant and so does not defeat
  that folding; asking `_l1_workers` anything without `set_num_threads` first, where it short-circuits
  before it looks at bytes; a probe that cannot name which build produced it; and a predicate keyed on
  `max(m, nrt, k)` tested only on shapes where `max` cannot move.

## gemv-T: what is closed and what is open

- ⛔ **The gate-binding cell, square n=128, is CLOSED as not winnable by restructuring this kernel.**
  Seven mechanisms investigated, six falsified: the ZA fold (21% of the call at that size, but
  stripping it leaves 1.81x of the 2.33x shortfall), the zero-mask width (bitwise identical, no
  consistent gain), the zero itself (0 within noise), loop unrolling (1.002x, bitwise identical),
  explicit `prfm` prefetch (1.002x there and 0.197x at m=1024 — five times slower), and cross-group ZA
  serialization (evidence against, on a probe with an unexplained 36% penalty, so not a clean
  refutation). The seventh, **double-buffered ZA tiles**, is confirmed and too small: block 2b in tiles
  0-3 and block 2b+1 in the idle tiles 4-7 is bitwise identical and reads 1.11-1.15x warm and 1.03x
  cold at that cell, 1.00 everywhere larger. It moves the cell 0.335 to about 0.344 against a need of
  2-3x. Both arms TIE there (SME 828 ns against NEON 797), so lowering `_SME_GEMVT_MINM` is priced at
  nothing.
- **Leading-dimension aliasing is OPEN at 1.16x**, re-measured in the gate's own shape: lda=1024 spans
  467.1-495.9 GB/s over 14 allocations and lda=1040 spans 511.0-585.3, not overlapping. The probe
  shape claims 1.29-1.38x, which is the shape gap above. ⛔ Falsified: fewer concurrent streams (the
  8-stream arm wins at every lda), packing (gemv reads A once, so a copy is three passes to buy one),
  and 128-byte column alignment (1024 and 1040 are both `mod 16 == 0` and differ 1.20x). In-kernel
  staggering of the 8 streams' row offsets is the only unfalsified mitigation and is unmeasured.
- ⚠ The ragged-row window, the strided-register load and the ZA-internal drain are all SHIPPED and
  measured; `sme_kernel.jl` carries their numbers. The remaining gemv-T lever is the lda one above.

## Reading the Apple caches

Two conventions here will produce a wrong number quietly. Both were hit in one sitting.

- ⛔ **THE TWO APPLE CACHES SIT AT DIFFERENT COMMITS, AND A SELF-SPEEDUP ACROSS THEM IS A
  CROSS-COMMIT RATIO.** The serial arms live in `bench/plots_data_neon_*.txt` and the threaded arms
  in `bench/mt_data_neon_*.txt`, and the two are swept separately, so their `commit=` stamps drift
  apart. Measured 2026-10-08: `plots_data` at `cc154a7d` (2026-10-04) against `mt_data` at
  `ca717b40` (2026-10-07) — eight days and many merges. Dividing a `pb` arm from the first by a
  `pb_mt` arm from the second compares two different libraries. **Read both `commit=` stamps before
  computing any self-speedup, and discard the figure if they differ.**
  This is a property of the Apple cache LAYOUT rather than of self-speedup: the AMD `mt_data_*`
  caches carry their own `pb` arm in the same file, so the ratio is same-commit there by
  construction. Here it is cross-commit by default.
  ✅ AND IT IS AN `arms=` SELECTION, NOT A CODE CHANGE. `_ANY_MT` (`plots.jl:287`) sends a run to
  `mt_data_*` when ANY threaded arm is selected, while `_DO_PB` independently decides whether the
  serial `pb` arm is measured — so `arms=pb,pb_mt` writes BOTH arms into `mt_data_*` and a
  self-speedup from it is same-commit by construction. That is how the AMD caches came to carry their
  own serial arm: an arms selection, not a layout decision. The Apple mt sweep selected `pb_mt`
  without `pb`.
  ⛔ IF THE APPLE mt CACHE IS EVER REFRESHED, INCLUDE `pb` IN THE SAME RUN. Re-sweeping `pb_mt` alone
  is what produced 110 per-cell cross-commit cells in one AMD cache, and that is WORSE than drifting
  across two files: the file header then reads a single commit while individual cells disagree. The
  per-arm stamp is the evidence; the file header is not.
  ✅ AND IT IS AN `arms=` SELECTION, NOT A CODE CHANGE. `_ANY_MT` (plots.jl:287) sends a run to
  `mt_data_*` when ANY threaded arm is selected, while `_DO_PB` independently decides whether the
  serial `pb` arm is measured — so `arms=pb,pb_mt` writes BOTH arms into `mt_data_*` and the ratio is
  same-commit by construction. That is how the AMD caches came to carry their own serial arm; it was
  an arms selection, not a layout decision. The Apple mt sweep selected `pb_mt` without `pb`.
  ⛔ IF THE APPLE mt CACHE IS EVER REFRESHED, INCLUDE `pb` IN THE SAME RUN. Re-sweeping `pb_mt`
  alone is what produced 110 per-cell cross-commit cells in one AMD cache — and that is WORSE than
  drifting across two files, because the file header then reads one commit while individual cells
  disagree. The per-arm stamp is the evidence; the file header is not.
- ⚠ **A CACHED SAMPLE IS `reps` CALLS, NOT ONE.** `bench/plots.jl` builds each sample from
  `reps = repsof(s)` fresh contexts, default `_reps_cubic(s) = clamp(20_000_000 ÷ s^3, 1, 512)` —
  20 at n=100, 9 at n=128, 1 from n=256 up. So a cached figure compared against a per-call probe is
  wrong by that factor: cached `syrk` n=100 reads 304.61 us against a probe's 15.5, and 304.61/15.5
  is 19.65. ✅ It CANNOT affect a ratio: `reps` is a pure function of the size, computed once per
  cell and passed to every arm, so it cancels exactly in a self-speedup or a gate ratio. It bites
  only a cache-against-probe ABSOLUTE comparison.
  ⚠ The tell is cheap and worth looking for first: a cached `zgemm` at n=8 reads 5.6e-5 s, absurd
  for one 8x8 product and unremarkable for 500 of them.
- **Pair every threaded arm with the serial arm of the SAME cell and commit before believing
  either.** On the 2026-10-07 `mt_data` cache, 15 of 591 cells have a threaded arm whose own samples
  span more than 1.5x — worst `LP potrsU` 256 at 4.71x, `LP gttrf` 256 at 2.55x, `LP potrfU` 256 at
  2.19x, `LP trtrs` 1024 at 2.09x, `L3 syr2k` 128 at 2.03x, `L3 symm` 100 at 1.95x. Each published
  median there sits inside one mode of a bimodal arm. That spread is internal to one arm at one
  commit, so it survives both errors above.
