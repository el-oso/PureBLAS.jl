# PureBLAS.jl — agent guidelines

Project-specific REQUIREMENTS for anyone (human or agent) working on PureBLAS. This is a
**multi-session, long-horizon** project — preserve knowledge here and in `ROADMAP.md` (the
canonical status + next steps), not just in a chat transcript.

PureBLAS is the second package of the **Pure Julia Ecosystem** ("Pure"): pure-Julia replacements
for Julia's non-Julia default libraries. PureFFT.jl is the first (sibling repo). PureBLAS replaces
OpenBLAS/MKL. Mirror PureFFT's conventions (layout, ReTestItems, StrictMode, TypeContracts, trim,
DocumenterVitepress).

## What PureBLAS is (architecture)

Pure-Julia BLAS plugged into Julia **two ways**, both first-class:
1. **Native API (Mode 2)** — `PureBLAS.axpy!(y,a,x)`, `PureBLAS.dot(x,y)`, … Direct Julia calls,
   no `ccall` boundary, so they are **AD-traceable** (ForwardDiff/Enzyme/ChainRules). This is the
   higher-value mode — opaque OpenBLAS `ccall`s never allowed differentiation through BLAS.
2. **LBT drop-in (Mode 1)** — reroute LinearAlgebra's BLAS/LAPACK to PureBLAS with ONE call,
   MKL.jl-style. **`PureBLAS.activate()`** registers in-process `@cfunction` pointers to the native
   `@ccallable` kernels via `lbt_set_forward` (`cabi_forward.jl`, 123 symbols); after it, `A*B`,
   `mul!`, `cholesky`, `qr`, `svd`, `LinearAlgebra.BLAS.*` all dispatch to PureBLAS. `deactivate()`
   restores OpenBLAS. This runs in the LIVE process against its own runtime — no `.so`, no double-init.
   The SAME `@ccallable` symbols also `juliac --trim` → `libpureblas.so` for **non-Julia hosts**
   (C/C++/Rust; self-inits its embedded runtime — see `juliac/ctest.c`) and prove trim-compatibility.
   **DON'T re-chase:** `BLAS.lbt_forward(libpureblas.so)` from *inside* live Julia still aborts (signal
   6 — the juliac lib double-inits the embedded libjulia); that path is only for non-Julia hosts. In
   Julia, use `activate()` (@cfunction registration), NOT `.so` forwarding. See `ROADMAP.md` "In-process
   LBT forwarding".

Both modes share ONE set of low-level kernels. Source map:
`core.jl` (accessors `_ld`/`_st!` over Ptr AND AbstractVector, lassq, |·|) · `cpuinfo.jl`
(SIMD width, const-folded, trim-safe) · `simd_kernels.jl` (SIMD.jl fast paths) · `level1.jl`
(low-level `(n,…,inc)` kernels) · `level2.jl` (gemv/ger/symv/hemv/trmv/trsv) · `level2_packed.jl`
(spmv/hpmv/tpmv/tpsv) · `level2_banded.jl` (gbmv/sbmv/hbmv/tbmv/tbsv) · `gemm.jl` (L3) ·
`contracts.jl` (TypeContracts `AbstractBLAS1`/`AbstractBLAS2`) · `backend.jl`
(`SIMDBackend`, Mode 2) · `native.jl` (bare API) · `cabi.jl`/`cabi_l2.jl`/`cabi_l3.jl`/`cabi_lapack.jl`
(`@ccallable` ABI) · `cabi_forward.jl` (in-process LBT `@cfunction` forward registry) · `lbt.jl`
(`activate`/`deactivate`).

## Hard requirements (MUST follow)

1. **Performance gate: ≥ 1.00× `max(OpenBLAS, AOCL)`, non-negotiable — evaluated on the ratio ROUNDED
   TO TWO SIGNIFICANT DIGITS.** Per-machine (per µarch on the fleet). Beat it where possible. BLAS-1 is
   bandwidth-bound (easy parity); the real fight is M2 `dgemm`.
   The threshold is unchanged at 1.00; what is specified is the PRECISION of the comparison. `0.995`
   rounds to `1.0` and PASSES; `0.9949` rounds to `0.99` and FAILS. Rationale: per-cell machine-state
   drift on this fleet runs ~1–6% (every cached arm stores the anchor it was measured under), so the
   third digit is not adjudicable, and the rounded figure is what the published tables and plots show —
   the verdict must agree with the number a reader can see. **The criterion lives in ONE place,
   `bench/gatecrit.jl` (`gate_pass` / `GATE_MIN`); never re-spell the comparison inline.** Consumers:
   `plots.jl`, `gate_misses.jl`, `gate_gaps.jl`, `coverage_ops.jl`, `coverage_routing.jl`,
   `gate_verdict.jl` (compares in log space against `log(GATE_MIN)`), and `adjudicate.sh` (carries the
   literal, kept in sync by comment — inline julia cannot `include`).
2. **SIMD.jl for kernels** (`Vec`, `vload`/`vstore`, `muladd`). Real unit-stride dense → SIMD fast
   path; everything else (complex, strided, any other `T<:Number`) → generic scalar loop.
3. **Generic over `T<:Number`.** ONE kernel implementation covers s/d/c/z (and ForwardDiff.Dual,
   etc.). The generic scalar path is what makes Mode 2 differentiable — do not specialize it away.
4. **Trim-compatible** (juliac --trim builds the .so). No runtime `eval`/`invokelatest`, no
   `Vector{Any}` at runtime, no CpuId ccall at runtime (bake detection into consts — see cpuinfo.jl).
   Verify with TrimCheck `@validate`.
5. **TypeContracts for interfaces** (`AbstractBLAS1` in contracts.jl). Backends carry explicit
   return-type annotations so inference matches the contract; eliminated by the trimmer.
6. **`nrm2` uses LAPACK scaled accumulation (lassq)** — overflow/underflow safe. Correctness
   boundary; never simplify to `sqrt(sum(abs2))`.
7. **Adapt to the CPU via compile-time detection, NOT manual flags.** The fleet spans different
   ISAs, cache sizes, and microarchitectures. PureBLAS detects the build machine with
   **`CpuId` / `HostCPUFeatures` / `CPUSummary`** and bakes the result into **const-folded, trim-safe
   consts** (`cpuinfo.jl`: `_SIMD_BYTES`/`_vwidth`, `_L1_BYTES`, `_INTEL_AVX2`, …), each **overridable
   via `Preferences`** (cross-compile / pinning / correcting a heuristic). When a kernel choice or
   tuning parameter depends on the CPU — ISA width, cache size, **or microarchitecture/vendor** — you
   MUST key it on one of these detected consts. Do **not** reach for an opt-in flag, and do **not**
   claim "we can't detect it." Note: width and cache size do **not** distinguish microarchitectures with
   the same ISA (Haswell vs Zen3 are both AVX2/W=4) — for a µarch-dependent choice use the **vendor +
   feature bits** (`cpuvendor`/`cpufeature`), as `_INTEL_AVX2` does for the `_CHOL_BASE_SPLIT` latency
   split. Detection stays at build time (const-folds away → no runtime `CpuId` ccall, per req. 4).

8. **DERIVE tuning from detected hardware — do NOT hardcode per-µarch literals. (Julia's advantage; USE IT.)**
   Every machine-dependent tuning parameter — block sizes (`mc`/`nc`/`kc`), base-case cutoffs, panel
   widths, unroll factors (`MR`), packing/algorithm-switch thresholds — MUST have a **default that is a
   FORMULA over the detected consts** (`_L1_BYTES`/`_L2_BYTES`/`_L3_BYTES`, `_vwidth`/`_SIMD_BYTES`,
   `cpuvendor`/`cpufeature`/family + `sizeof(elt)`), keyed on a **physical criterion** — cache RESIDENCY
   for block sizes (e.g. `kc·nr·sizeof(elt) ≲ ½·L1`), datapath LATENCY for unroll, ISA for width
   granularity. **Cache size and ISA come hand-in-hand — take BOTH into account** (a block size depends on
   how much fits in cache AND the vector width). A bare literal like `_vwidth==4 ? 48 : 64`, `const _KC =
   256`, or any hand-fit magic block size is a **VIOLATION** — the tell is a number you can't trace to a
   detected const via a residency/latency formula. **Why this is mandatory, not optional:** Julia JITs to
   the host at load time, so PureBLAS can *compute* the right sizes for the ACTUAL machine — including CPUs
   never benchmarked (a new laptop, a cloud box). Static C/Rust BLAS (OpenBLAS/BLIS) can't — they ship
   hand-tuned per-µarch tables baked at their compile time; hardcoded literals here throw away Julia's one
   real structural advantage and silently mis-size on any box off the test fleet. Rules: (a) Preferences
   override stays (pinning/calibration/correcting a heuristic), but the **default must autotune**. (b) A
   derived formula must **reproduce the measured-optimal values on the known fleet** (Zen4/Zen3/Zen5) before
   it's trusted to extrapolate — derive → validate on the fleet → ship. (c) Every tuning const cites which
   detected consts it derives from and the residency/latency criterion. (d) When tempted to write a magic
   block-size number: STOP and derive it. Applies to **all of BLAS-1/2/3 + LAPACK**, not just new code —
   existing literals (`_KC`, `_MC`, `_NC`, `_CPOTRF_BASE`, `_CPOTRF_NBMAX`, the `_vwidth==4 ? …` cuts) are
   tech debt to migrate to derived formulas.

8b. **THE "PDM LADDER" (Pin → Derive → Measure) — the NAMED, MANDATORY mechanism for req#7/#8. Every
   machine-dependent constant is a "self-tuning constant"; a bare OR validated-literal default is a
   VIOLATION ("No Fixed Tuning").** This is the rule I keep forgetting — enforce it explicitly. Refer to
   the mechanism as **"the PDM ladder"**; a knob obeying it is a **"self-tuning constant."** EVERY tuning
   knob (block size, cutoff, panel width, unroll, fuse factor, stream count, algorithm-switch threshold)
   resolves in exactly this order:
   - **P — Pin:** `@load_preference("name", <default>)`. A set Preference always wins (calibration,
     cross-compile, and the trim/.so build — which MUST pin any Measure-tier knob, since a runtime
     benchmark is not trim-safe).
     **THE PIN TIER IS THE USER'S (or the build's) — NEVER THE AGENT'S.** An agent may only ever work in
     Derive or Measure. Concretely: authoring the `<default>` argument is Derive-tier work and IS
     allowed (a formula, or a falsified-derivation literal with its measured table, marked `req8-ok`);
     supplying a VALUE for the preference that overrides it is NOT — that means no writing/editing any
     `LocalPreferences.toml`, no adding pins to `juliac/build.jl`, and no *recommending* a pin as a
     knob's resting state. A pin is a deployment decision for one machine or one shipped artifact; it
     silently overrides every derivation, so an agent that pins can mask its own bad default and make
     the fleet unreproducible. If a conversion leaves something that looks like it wants a pin, say so
     and STOP — it is the user's call.
   - **D — Derive:** if the optimum is **physically predictable from a detected const** (cache RESIDENCY,
     SIMD width, register count), the default is a **FORMULA** over `_L1/L2/L3_BYTES`, `_vwidth`, `_NVREG`,
     … — zero runtime cost, const-folds (trim-safe), adapts to unseen machines. Live examples to copy:
     `_TRI_NB` (L1 residency of the diagonal block), `_qr_nb` (L2-residency ramp under a register-
     invariant cap), `_TRSV_REG_MAX` (`_SCALAR_FPREGS − temps`, validated against a measured knee).
     (`_GEMVN_RB_MAXA` was cited here for a while and does not exist — the real const is `_GEMVN_RB`.)
   - **M — Measure:** if the optimum is **NOT predictable from detected consts** — it depends on port
     balance / prefetcher / write-stream count and can INVERT sign across µarchs (the TELL: our own model
     mispredicts a box we HAVE) — the default is an **on-host auto-tune**: `Base.OncePerProcess` measuring
     a **formula-bounded candidate set**, `@static if isnothing(pref)`-gated so a pinned build never
     benchmarks. Mirror `_ger_np` / `_gemvt_nc`. The candidate set is itself **Derived** (e.g. NC bounded
     by `_NVREG`) — so BOTH the bounds and the selection adapt to unseen hardware.

   **The decision D-vs-M IS the rule:** for every machine-dependent number, ask *"is this physically
   predictable from a detected const?"* — **Yes ⇒ Derive, No ⇒ Measure. There is NO third option.** A
   fixed literal is NOT a valid answer — it is a Measure-tier knob that hasn't been converted yet
   (correct only for the µarchs we've benchmarked; wrong on unseen ones). When you write ANY tuning
   number, STATE its tier in the code comment; if you can neither give a Derive formula nor a Measure
   harness+candidate-set, STOP — it's a violation.

   **NARROWED 2026-08-19 — when a µarch predicate IS the derivation.** This clause used to forbid
   `_double_pumped(_HW) ? 8 : 4` outright. That over-reached: it is a violation when used as a lazy
   two-way lookup for a knob whose real criterion is something else, but it is the CORRECT derivation
   when the knob's physical criterion genuinely IS the datapath/vendor/family fact the predicate
   encodes. `cpuinfo.jl` calls `_double_pumped` "silicon FACTS, not tuned magic" for exactly this
   reason. Test to apply, and it must be argued in the comment: name the physical mechanism, show it
   maps 1:1 onto the predicate, and give the fleet table. Worked example — `_axpy_dram`: Zen4
   double-pumps 512-bit ops over a 256-bit path, so the narrow 256-bit phase kernel is the right arm
   there and not on a native-256 AVX2 part; measured DRAM-regime, arm208 beats arm4 by 17% on Zen4 and
   LOSES on Zen3. That is a mechanism, not a lookup table. A predicate used without that argument is
   still a violation.

## ABI conventions (Mode 1)

- Symbols are the **ILP64** reference-BLAS names Julia resolves: trailing `64_` (e.g. `daxpy_64_`).
  Args **by reference** (`Ptr`), **column-major**, `Int64` integers. BLAS-1 has **no character
  args** → no hidden Fortran string-length args (a reason it's the M1 slice).
- **Deferred:** the 4 complex-dot symbols (`c/zdotu`, `c/zdotc`) — their complex-return ABI (LBT
  NORMAL vs ARGUMENT retstyle) is unresolved; lands in M2 with GEMM's char/string ABI. Native API
  covers complex dot meanwhile.

## Testing (TestItemRunner — self-contained, individually triggerable)

- **NEVER `--project=test` — main env + `Pkg.test()` only.** `Pkg.test()` resolves test deps in a
  TEMPORARY env, so it needs no manifest; activating `test/` is the only thing that creates
  `test/Manifest.toml` (which must not exist) and it strips every comment from `test/Project.toml`.
  Fresh deps: `julia --project=. -e 'using Pkg; Pkg.Registry.update()'`. One item:
  `Pkg.test(test_args=["<name regex>"])`, ANDed with the group/shard filter.
- Shared oracle helpers go in `@testmodule Name begin … end` (TestItemRunner; ReTestItems'
  `@testsetup module` is not recognized). ReTestItems can't run on 1.13 and isn't coming back — see
  kb `julia-113-test-toolchain-and-env-discipline`.
- Correctness oracle = OpenBLAS via `LinearAlgebra.BLAS.*` over s/d/c/z, many `n`, strides, edges.
  Note: single-vector ops (nrm2/asum/iamax/scal) are spec'd `incx ≥ 1` (reference returns 0 for
  `incx<1`); only two-vector ops (axpy/dot) take negative/mismatched increments.
- StrictMode dogfood (`@assert_typestable/@assert_noalloc/@assert_trim_safe`) on hot paths, gated by
  `StrictMode.checks_enabled()`. AD smoke test via ForwardDiff (proves Mode 2).

## Benchmarking (reuse PureFFT methodology)

`BLAS.set_num_threads(1)` for fair single-thread comparison · `@noinline` concrete wrappers (not
closures) · repeated in-place reps · **median** times (not min) · `taskset -c N` + cpufreq pin for
low noise · results→JSON, plot from JSON · **per-host JSON filenames** (fleet: Zen4 dev / Zen3 AVX2 /
Zen5 native-AVX512 / future M5 ARM — the 1.0× gate is evaluated per machine).

- **CHAIRMARKS ONLY — DO NOT AUTHOR TIMING FUNCTIONS.** Benchmarks and timing use Chairmarks (`@be`),
  as `bench/plots.jl` does. Writing a timing/benchmark function, or any wrapper around Chairmarks,
  requires EXPLICIT APPROVAL FIRST — it is not a judgement call. If a measurement appears to need
  something Chairmarks does not provide, stop and ask. Enforced by `test/estimator_lint.jl`: only
  `plots.jl` and `measure.jl` may drive a benchmark, raw clocks (`@elapsed`/`time_ns`/`@btime`) are
  banned across `bench/`, and `test/harness_baseline.txt` carries the pre-existing debt so new
  violations fail immediately. Why it is a hard rule: a hand-rolled loop took ONE timing per window
  where `@be` takes hundreds, and at `axpy` n=1e6 that flipped the sign of the result — hand loop
  0.991 [0.941, 1.076] ("falsified") vs Chairmarks 1.022 [1.005, 1.030] (2.2% faster, decisive).
- **ESTIMATOR — MEDIAN, and it is ENFORCED, not remembered.** Every timing that informs a gate decision
  reduces through `Measure.tstat` (`bench/measure.jl`) = `median`. **Never `minimum`, never `mean`** —
  `min` is optimistic AND tail-blind, `mean` over-weights the tail; the median is chosen precisely to be
  insensitive to window tails. Report numbers WITH their estimator and sample count (`Measure.report` →
  "0.946 (median of 8 rounds)"); a bare figure hides which statistic produced it.
  **Throwaway probes go in `bench/probes/`, never a /tmp scratch dir** — contents are gitignored but the
  directory IS scanned by `test/estimator_lint.jl`, which fails the suite on an unapproved reduction
  (escape hatch `# estimator-ok: <reason>` for deliberate non-gate uses like `cpuvalidate.jl`'s
  cliff-finding). Why this is a rule and not advice: on 2026-08-03/04 the shipped kernels obeyed the
  median rule while the PROBES used `minimum(@elapsed …)`, which ranked an `iamax` unroll NB=2 *above*
  NB=4 at n=1e6 where the gate's median ranked it 15% *worse*. A day went into explaining that
  contradiction with tail hypotheses and a kernel port; the estimator swap was the whole of it. Sample
  count is not a fix — with `min`, more samples drifts further from the median.
- **FREQUENCY METHODOLOGY — one command, never re-decided: `sudo bench/fleet_freqlock.sh lock`** (that
  script is the single source of truth; read its header). It sets `amd_pstate=passive` + **boost OFF** +
  all cores pinned to **base clock** (min=max) + **verifies the achieved freq under load**. This is the
  ONLY valid state for a gate/plot measurement, on every box. Rules: (a) a run whose `verify` is not ✅
  (boost floating, `boost=1`) is **INVALID — discard it, don't rationalize it**; a floating boost clock
  drifts between the OB and PB windows → wide, meaningless ratios. (b) **There is no stable high pin** —
  a clock above base (e.g. 4000 on a ~2 GHz-base chip) lives in the boost range and floats above
  `scaling_max_freq`; `pin >base` is refused by design. Base clock is the ceiling for a LOCKED run.
  (c) Absolute clock is irrelevant (the gate is a PB/OB *ratio*, both at one clock) → higher clock buys
  nothing and costs drift. Do **not** reopen this per session — measure on `lock`, full stop.

## Standing rules

- **ITERATE PROBES IN THE HOT REVISE SESSION (`bench/hot.jl`) — never relaunch `julia` per probe, and
  never restart it just to pick up a `src/` edit.** A fresh launch pays a full pkgimage precompile
  (**200–311 s** measured); a correctly configured Revise applies a method-body edit in **~3 s**.
  ```bash
  mkfifo /tmp/pbhot.fifo
  julia --project=bench bench/hot.jl /tmp/pbhot.fifo > /tmp/pbhot.log 2>&1 &   # await <<<HOT-READY>>>
  echo bench/probes/some_probe.jl > /tmp/pbhot.fifo                            # await <<<HOT-DONE>>>
  ```
  **The trap that makes Revise look broken:** a driver loop that calls `open(readline, FIFO)` per command
  blocks inside libuv **without yielding**, so Julia's scheduler never runs Revise's async file-watcher
  task. The revision queue stays empty and a bare `Revise.revise()` returns *"success" having applied
  nothing* — silently, indistinguishable from "no changes". On 2026-08-10 a 30-minute A/B ran entirely on
  stale code and was caught only because that probe carried a witness counter. `Revise.retry()` does NOT
  help (it retries *errored* revisions; nothing was queued to fail), and a restart is NOT required.
  The fix, already in `hot.jl`: open the FIFO **once as a stream** (`open(FIFO, read=true, write=true)` —
  read+write so the process holds its own writer and the pipe never EOFs) and `readline` that handle.
  Async I/O yields, the watcher runs, revision is incremental (**3.3 s**), and a bare `touch` triggers it.
  `Revise.revise(PureBLAS)` (~62 s, whole module) stays only as an **announced fallback**. Diagnose any
  recurrence with `Revise.pkgdatas` / `revision_queue` / `queue_errors`: tracked + empty queue + no errors
  ⇒ scheduler starvation, not a Revise fault.
  **VERIFY, don't assume.** Three failure modes look identical to a real measurement — stale code (above),
  a **dead knob** (the flag's branch isn't in the call graph for that shape), and a **stale cache**
  (`plots.jl` *without* its `bench` arg loads cached data and prints a complete, plausible, pre-change gate
  table). So: read the `<<<HOT-REVISE …>>>` line, give every A/B knob an execution **witness** asserted
  before timing, and read the benchmark provenance header every time.
  Also: probes dispatch with `Base.include`, **not** `Revise.includet` (a probe is top-level side effects
  and `includet` won't re-execute them — it returns "ok" in 0.0 s having run nothing); `includet` is right
  only for a helper *module*. A newly added top-level `const` comes up **unassigned** (Revise applies
  method definitions, not top-level statements) — prefer the pre-declared `_EXPFLAG`/`_EXPINT` tables so a
  new knob is a new INDEX. Gate numbers still come from a standalone `plots.jl` run, which owns provenance.
  Full write-up: `../kb/findings/julia-revise-hot-session-workflow.md`.

- **SYNC THE FLEET WITH GIT, NEVER rsync — use `bench/fleet_sync.sh <box|all> [ref]`.**
  Commit and push first; the script fetches and hard-resets the box to a pushed ref, then re-verifies
  source parity by md5. Do not `rsync src/` to a fleet box, and do not hand-roll an `ssh … git reset`.
  **Why:** rsync copies the code but not its identity. `bench/plots.jl` stamps `commit=` into every
  cache header from `git rev-parse`, so an rsync'd box benchmarks new code while claiming its old
  HEAD. On 2026-07-31 galen emitted a full gate sweep stamped `commit=ac96c00` while actually running
  the tree from `78eafc7` — 13 commits and two perf fixes later. The numbers were fine, but the
  provenance in the published coverage table was false and **nothing in the artifact revealed it**;
  it was caught only by manually md5-ing both trees. A benchmark cache is evidence, and evidence needs
  a truthful provenance line. rsync of a subdirectory is also silently partial — new `src/` with stale
  `test/` or `juliac/` and no indication. Bench caches are gitignored, so the hard reset preserves them
  and `op=`/merge runs keep working. The script refuses if local HEAD is not an ancestor of the target
  ref (catches "I synced my uncommitted tree" before a 3-hour sweep, not after).

- **PUBLISH ARTIFACTS WITH `bench/publish.sh` — never by hand-running one generator.** Every published
  number (the `perf_*.svg` pairs, `bench/gen_table*.md`, `bench/provenance.md`, the generated tables in `docs/src/coverage.md`)
  is a pure function of the caches on disk, and `publish.sh` rebuilds the whole set in one fixed order:
  cell-staleness audit → both reference views → coverage tables → re-verify → print the `git add` line
  (it never commits or pushes). `bench/check_artifacts_current.sh` is the gate — it re-renders into a
  temp dir and byte-compares, so "cells current w.r.t. `src/`" and "artifacts current w.r.t. the cells"
  fail as one. **Why:** on 2026-08-17 four independent staleness bugs shipped in one night — a cell
  predating the code, the OpenBLAS SVGs stuck at `bdb9497` while the AOCL set was re-rendered three
  times, the two views therefore contradicting each other about the same fleet, and prose asserting
  something false about the plots. Running a subset of the generators is how each of them happened.
  The render now emits **both** reference views per invocation (`_VIEWS` in `plots.jl`), so they cannot
  diverge; `bench/check_view_pairing.sh` (CI job `artifact-pairing`) catches the residual case of a
  partial `git add`, and is the only artifact check that runs in GitHub CI — the caches are gitignored.

- **READ `../kb/findings/` BEFORE any perf diagnosis or gate campaign — before measuring, not after.**
  The sibling `kb/` is the cross-session knowledge hub: 25 digests of diagnostics, decisions, measured
  results, and **disproven hypotheses so nobody re-chases a dead end**. Start at
  `../kb/wiki/index.md`, then grep by routine (`grep -rli syrk ../kb/findings/`).
  This rule exists because it was violated: on 2026-07-30 a session re-measured all of BLAS 1–3 and
  reported the po2-ld L3 cells (syrk 0.95, syr2k 0.96, trmm 0.97) as new findings, then started
  diagnosing syrk from scratch — all of it already root-caused in
  `kb/findings/pureblas-l3-syrk-syr2k-symm.md`, **with the A-pad remedy already measured and
  deliberately rejected for trmm**. Two methodology rules were re-derived from scratch as well
  (`kb/findings/pureblas-avx2-l3-gate-campaign.md` items #1–#2). Hours lost.
  **Write back too** (`kb/CLAUDE.md` rule #2): a diagnosis, decision, or disproven idea belongs in a
  `findings/` file plus a `wiki/index.md` refresh. A commit message is not the kb — the kb sat dormant
  2026-07-12 → 2026-07-30 while five campaigns shipped.
- **A sub-1.0 PB/OB ratio is NEVER a "ceiling" — it is an implementation gap.** OpenBLAS runs on the
  *same silicon*; if it reaches ≥1.0, the hardware is demonstrably capable, so any `PB/OB < 1.0` is an
  algorithm/kernel-formulation problem in PureBLAS, full stop. Do **not** write "ceiling," "near-ceiling,"
  "hardware limit," or "irreducible" for a sub-gate number OB beats. The word "ceiling" is reserved for a
  limit that binds OB too (a real roofline: bandwidth / instruction throughput / unhideable latency) —
  and then you must show OB is also stuck there. When tempted to shelve a residual as a ceiling, instead
  ask **how OpenBLAS achieves it on this machine** and target that mechanism (e.g. OB packs every L3
  operand → po2-`lda`-immune; a PB kernel that reads the matrix directly is the gap). See memory
  `no-ceiling-if-openblas-does-it` + `gate-is-non-negotiable`. (Same root error as "assume Rust is
  faster without measuring.")

- **SIMD microkernel pipelining.** A register-blocked microkernel's k-reduction loop wants (a) a
  **prefetch of the output (C) tile at entry** (overlaps the cold RMW store epilogue), and (b)
  possibly **`@inbounds @simd ivdep`** on the k-loop (register accumulators, no cross-iteration
  memory dep → LLVM software-pipelines). The prefetch is safe everywhere; `@simd ivdep` is
  **FMA-density-dependent — measure per kernel/µarch**: it *helped* complex gemm (4 FMA/cell,
  0.93–0.95→gates on AVX2, commit that added it) but *regressed* real gemm (1 FMA/cell — LLVM already
  optimal, broke n=16 AVX-512). Diagnostic: a *flat* few-% under gate across all sizes ⇒ microkernel
  gap ⇒ diff vs a sibling that gates. Build the loop with the block `quote…end` form (inline
  `:(@simd for…;…;end)` is a ParseError). See kb `pureblas-gemm-microkernel-simd-prefetch`.
- No Python anywhere (global rule). Native lib via `ccall` or CLI subprocess if external is needed.
- `isnothing(x)` / `!isnothing(x)`, never `=== nothing`.
- Commit author email: `15278831+el-oso@users.noreply.github.com` (never a real address).
- End commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- The approved plan is a contract: do not skip/substitute a requirement without asking first.
