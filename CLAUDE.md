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

9. **"IT WORKS" IS NOT A RESULT — MEASURE IT, OR IT IS NOT DONE. (HARD RULE.)** This project's entire
   mandate is a PERFORMANCE gate (`PB ≥ max(OpenBLAS, AOCL)`, req#1). Correctness is table stakes, not
   the deliverable. So **any change that alters which code runs** — a new dispatch path, a missing
   method or workspace added, a type generalisation, a routing or predicate fix, a fallback — is
   UNFINISHED until its ratio is measured and stated. Not "it dispatches now", not "tests pass", not
   "derivatives match to 1e-12": **a number, against the reference that applies to it**, and ideally
   across the size ladder, because the SHAPE of the curve is the diagnosis (a ratio that FALLS with n
   means the routine is not amortising — see `geqrf` below).

   **The rule exists because it was violated, and the violation shipped.** 2026-09-13: `_syev!` was a
   `MethodError` on `ForwardDiff.Dual` because `_trdws` had no generic fallback. I added the fallback,
   verified eigenvalues to 1.7e-13 and derivatives to 1.7e-12 against `ForwardDiff.derivative`, checked
   seed-independence, and reported the gap CLOSED. I never timed it. It was measured only because a
   DLP bench group happened to land days later, and it reads

       dsyev1   n=32 1.95 → n=50 1.59 → n=100 1.12 → n=128 0.99 → n=256 0.66

   — degrading monotonically on all three boxes, i.e. slower than LinearAlgebra's GENERIC fallback at
   n≥128 and getting worse. The same session had already diagnosed exactly that curve shape for dual
   `geqrf` (0.57/0.49/0.45, "does NOT reach a blocked path") and fixed it. Making a routine RUN and
   calling that done is how a routine that runs 1.5× slower than the thing it replaces gets published
   as a closed gap.

   **The test, before you call anything done:** *can I state the ratio?* If not, it is not done — say
   so explicitly ("dispatches and is correct; SPEED UNMEASURED") rather than implying completion. A
   capability with no number attached is a liability, because it looks finished to the next reader.

10. **A `!` FUNCTION MUST NOT ALLOCATE. (HARD RULE — that is what the arena is FOR.)** Every
   bang-suffixed entry point is allocation-free at steady state, for EVERY element type it accepts —
   `Float64`, complex, `ForwardDiff.Dual`, and anything else that dispatches. Scratch comes from the
   arena (`@scope arn begin … borrow!(arn, T, …) end`, `src/arena.jl`; `src/lapack/sysv.jl:192` is the
   canonical pattern) or from caller-supplied workspace — never from a fresh `Matrix`/`Vector` in the
   call. "Steady state" means from the SECOND call: the arena grows once at a new high-water mark and
   `docs/src/arena.md` says so explicitly.

   This is not aspirational, it is the measured status quo. `bench/probes/lapack_entry_alloc.jl`,
   n=64, warm then `@allocated`: **13 of 14 LAPACK bang entries are 0 B** — `potrf!`, `getrf!`,
   `trtri!`, `potri!`, `sytrf!`, `sytri!`, `geqrf!` on Float64, and all but one on Dual.

   **Written down because I argued my way out of it.** Asked whether dual functions carried strict
   contracts, I answered that LAPACK entries "legitimately allocate — `sytri!` borrows from the arena,
   blocked drivers take workspace", and concluded `@test_noalloc` was the wrong instrument for LAPACK.
   Every clause of that was wrong: `sytri!` measures **0 B** precisely BECAUSE it borrows from the
   arena — borrowing is the mechanism that prevents allocation, not an admission of it. The one real
   violation, dual `geqrf!` at **18736 B/call**, came from a generic `_qr_ws` allocating four fresh
   `Matrix{T}` per call while the Float64 path forwarded to the owned pool. I had also written the
   same defect myself in `_trdws` and rationalised it in a code comment as acceptable "against an
   O(n³) reduction".

   The failure mode to recognise: an unwritten invariant is one you can talk yourself out of with a
   plausible-sounding cost argument. **Measure the entry (`@allocated`, warm, wrapped in a function
   over pre-built operands — at top level it boxes the return and reports phantom bytes) before
   claiming any allocation is justified.** A non-zero result on a `!` entry is a DEFECT TO FIX, never
   an exemption to document.

11. **BITWISE REPRODUCIBLE ACROSS THREAD COUNTS. (HARD RULE, with a test gate.)** One build, one
   machine, one input: `PureBLAS.set_num_threads(n)` must not change a single bit of the result, for
   any `n`. Scope is thread count only — NOT across microarchitectures, NOT across versions. It is
   always on; there is no opt-in mode, because a mode is a promise nobody can rely on by default.

   **The rule that makes it hold: the reduction TREE must be thread-count independent, not merely the
   schedule static.** Partition `m` and `n` freely; never let worker count reach `k`. The classical
   column split already satisfies this — it splits `n` only, and `kc = min(_KC, k)` is a function of
   the problem — so every element of C accumulates its whole `k` chain on one worker in one order.
   Two things break it and both have bitten:
   - **A size-keyed predicate read from a chunk's own slice.** `_use_unpacked` is keyed on
     `max(m, n, k)`, and the unpacked and blocked routes differ in α placement, β handling and `kc`
     chunking, so a narrow chunk silently computes by different arithmetic. Hence `_gemm_core!`'s
     `nroute`: a partitioned caller passes the WHOLE problem's column count and computes on its slice.
     Any new size-keyed branch reachable from a chunk body must take its size from `nroute`.
   - **An algorithm a worker cannot run.** Strassen chooses its depth from the width it is handed, so
     a worker holding a slice runs a different algorithm entirely. It is therefore never column-split:
     `gemm!`, `_symm!` and `_trmm_split_L!` ask `_strassen_owns` and run the recursion serially when it
     answers yes. `_gemm_core!`'s `strassen` defaults to `false` so the chunk body and the lost-claim
     fallback — the two paths a worker count can reach — cannot take the route by omission.

   **Every path a worker count can reach must agree, including the ones that are not the happy one.**
   A caller that loses the pool claim runs the serial fallback; if that fallback takes a different
   route than the winner's workers, the same call returns different bits depending on a race with an
   unrelated thread. That shipped once (`6940a2d3`).

   **Forward constraint:** if `dot`/`nrm2` are ever threaded they break this by construction unless
   their reduction tree is fixed independently of worker count. The standing decision not to thread
   them is load-bearing.

   **The gate** is `test/gemm_tests.jl` "gemm/symm: bit-identical at every thread count" and its
   syrk/syr2k sibling. Both compare `reinterpret(UInt64, …)` patterns, never a tolerance — a 1e-13
   divergence is a divergence. Both assert a **witness** first, because a shape Strassen claims runs
   serially at any thread count and would pass without a worker ever starting; and both include an
   α that is not a power of two, because scaling by `±2^j` is exact and hides an α-placement
   difference. CI sets `JULIA_NUM_THREADS`, without which these items skip and report green.

12. **A SWEEP IS SCOPED TO THE GROUPS THE CHANGE CAN REACH — PROVE THE SCOPE BEFORE LAUNCHING.
    (HARD RULE.)** A fleet sweep costs hours of box time on three machines and cannot be interrupted
    without discarding it, so the scope is a decision that has to be made and defended BEFORE the
    launch, never rationalised after. State, in the launch message: which groups the change can
    reach, by what mechanism, and why the rest cannot move. A group survives that argument only if a
    source file the change touched is on its call path.

    **The cheap proof already exists in the cache.** Every arm record carries its raw samples, so a
    damaged regime is visible as a sample SPREAD the comparison arm does not have. Query the cache
    before deciding, not after:

        awk -F'\t' 'NR>1{g=$1; for(i=4;i<=NF;i++){split($i,a,"|");
          m=split(a[8],s,","); if(m<3) continue; lo=hi=s[1];
          for(j=1;j<=m;j++){if(s[j]+0<lo+0)lo=s[j]; if(s[j]+0>hi+0)hi=s[j]}
          k=g"|"a[1]; if(hi/lo>mx[k])mx[k]=hi/lo}}
          END{for(k in mx) printf "%-12s worst spread %6.2fx\n", k, mx[k]}' bench/mt_data_*.txt | sort

    **Measured, and it is why this rule exists.** 2026-09-21: the `pb_mt` mask starved the gemm
    pool's join, and all eight groups were re-swept to repair it. The query above, run afterwards,
    showed the damage confined to **L3 and LP** — worst threaded spread 7.43x and 4.87x against
    serial arms at 1.59x and 2.25x — while the other six groups' threaded and serial arms agreed to
    within 0.05x. They agreed because they do not thread: threading exists only in `gemm`, `symm`,
    `syrk`, `syr2k` and the LAPACK routines that call them. 329 of 937 cells, about 50 minutes per
    box on three boxes, bought nothing.

    **The one standing exception is anchor coherence**, and it must be argued, not assumed: refreshing
    part of a cache leaves the rest at an older machine state, and a partial refresh has twice left
    most of a cache anchor-mismatched. When that is the reason for a wider scope, say so at launch
    and name the cells it protects — it does not license a full sweep by default.

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
- **NAME THE BLAST RADIUS — a bare `Pkg.test()` is a PR-time gate, not an iteration step.** Filter to
  the items the change can reach: measured **16m59 for the full suite against ~2m** for a filter
  covering the same changed files. `JULIA_NUM_THREADS` must be set either way, or every threading item
  skips and reports green (see the `checks` CI job and the item "threading guarantees actually ran").
  `julia-guard.sh` blocks the bare form; the PR-time run is explicit:
  `PB_FULL=1 julia --project=. -e 'using Pkg; Pkg.test()'`.
  **The tell that generalises past what a hook can see** (a sweep, fleet time): you are about to spend
  minutes of box time and cannot name the single question it answers. A targeted run answers a named
  question; a full run answers "did I break anything", which is a question for submission time.
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

- **ITERATE PROBES THROUGH `bench/probe.sh` — never relaunch `julia` per probe, and never restart the
  session just to pick up a `src/` edit.** A fresh launch pays a full pkgimage precompile (**200–311 s**
  measured, 6m33 end to end on this box); a correctly configured Revise applies a method-body edit in
  **~3 s**. Measured on the same probe: **6m33 cold vs 9s warm, 43×**.
  ```bash
  bench/probe.sh bench/probes/some_probe.jl   # starts the session if needed, then reuses it
  bench/probe.sh --status                     # is one up?
  bench/probe.sh --restart                    # only after an include-graph change
  ```
  The wrapper owns the fifo, the per-box CPU mask, the thread count and the completion marker, so
  "run a probe" is one command that is never cold. It reads only the log its own run appends — a
  `<<<HOT-DONE>>>` from an earlier probe would otherwise report a probe finished before it started.
  `julia-guard.sh` blocks a cold `bench/probes/` launch while a session is live (escape hatch
  `JULIA_GUARD=off`), because rule 3's allow-list exempts `bench/` and never caught this shape.
  The raw form below is what the wrapper does, kept because the failure mode under it is subtle:
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
  HEAD. On 2026-07-31 Zen3 emitted a full gate sweep stamped `commit=ac96c00` while actually running
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
- **EXPLOIT StrictMode TO ITS MAXIMUM when tuning kernels or testing. Failing to is UNACCEPTABLE, not
  merely suboptimal — its STATIC checks come BEFORE any hand-rolled analysis, probe, or guessed
  mechanism.** StrictMode is not a generic third-party utility that happens to be installed: it is
  **purpose-built for this project and the Pure ecosystem** — it exists precisely because these SIMD
  kernels needed properties like "did this vectorize", "did it spill", "is there a scalar loop" made
  checkable instead of discovered later with a profiler. It is a `[deps]` dependency of this project
  (main env, NOT `bench/`). Hand-rolling an analysis it already provides is a process failure, and the
  surface is far wider than the `kernel_report` that gets reached for by habit. Match the tool to the
  symptom, and run it *before* writing a probe:

  | symptom / question | tool |
  |---|---|
  | is there a scalar tail / glue loop? | `scalar_fp_loops`, `@assert_no_scalar_loops` |
  | did this vectorize at all? | `@assert_vectorized` |
  | did adding accumulators SPILL? (tell: the change hurts even ALIGNED sizes) | `spill_report`, `@assert_no_spill`, `register_report` |
  | where do the cycles go — port pressure, IPC, recurrence depth? | `mca_report`, `@assert_mca` |
  | arithmetic intensity / shuffle-port bound? | `kernel_report` |
  | is this call actually inlined? | `@assert_inlined`, `inline_suggestions` |
  | boxing / instability / allocation on a hot path | `@assert_noboxing`, `@assert_typestable`, `@assert_noalloc` |
  | trim/.so safety | `@assert_trim_safe`, `explain_trim` |

  **Why this is a rule:** on 2026-08-25/26 three separate investigations were run by edit-measure-revert
  that a static check answers in seconds — eleven "SIMD body + scalar tail" kernels found one at a time
  with 25-second sawtooth probes (`scalar_fp_loops` detects exactly that class, and its source comment
  names it); a gemv-T/C regression that hurt even ALIGNED sizes, diagnosed as a register spill only
  after writing the code (`spill_report`); and the `@generated` inline-meta hazard, tested by
  edit/measure/revert and falsified (`@assert_inlined`). Two independent subagents also asserted
  "PureBLAS has nothing equivalent to llvm-mca" — wrong; `mca_report` was always there.
  **Notes:** `mca_report` needs `LLVM_full_jll` (`using LLVM_full_jll`, ~680 MiB **weak** dep — Julia
  ships libLLVM but NOT the `llvm-mca` CLI; do **not** apt-install it). These checks are static and need
  no privileges, unlike perf counters (`perf_event_paranoid` gates those on this fleet) — so the static
  pass is always the cheaper first move. A check that reports "clean" for everything is worthless
  evidence: give it a POSITIVE CONTROL (a known-bad shape) before believing a negative sweep.

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
