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

Nothing on Apple can be scored until these land. Do them before any kernel work.

- [ ] **`accelerate_mt` does not exist as an arm.** `_REF_MT` in `bench/plots.jl` is
      `{"openblas_mt", "aocl_mt"}` only. On Apple a threaded gate is not merely unmeasured, it is
      **unexpressible**. Add the arm.
- [ ] **Accelerate ignores `BLAS.set_num_threads`.** It reads `VECLIB_MAXIMUM_THREADS` **once, at
      first use**, so the variable must be set at file-load time before the library is touched — the
      same shape as the existing single-thread pin at `bench/plots.jl:155-175`. Getting this wrong
      does not error; it silently measures the wrong thread count.
      ⚠ Related trap already paid for once: the Accelerate dylib needs
      `suffix_hint = "\x1a$NEWLAPACK$ILP64"`. Without the leading `0x1A`, LBT falls back to LP64 and
      `dgemm_64_` **returns zeros rather than raising**.
- [ ] **Measure `openblas_mt` across every group.** `ROADMAP.md` Step 0 says this and it is still
      true: the per-cell reference rule needs *both* OpenBLAS arms present, because a single op draws
      on each depending on size. There is **no `mt_data_neon_*.txt` cache at all** today.

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
- [ ] **SME covers Float64 gemm plus one gemv kernel and nothing else.** Extending it is pass-1 work.
      SME is Float64-only (`FEAT_SME_F64F64`), but complex and dual decompose into *real* products,
      so they inherit SME iff those products reach an eligible call — routing, not kernels.

---

## Pass 2 — NEON-MT

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
