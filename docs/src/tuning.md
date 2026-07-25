# The Tuning-Constant System

Every fast BLAS is full of magic numbers: block sizes, panel widths, unroll factors,
algorithm-switch thresholds. This page explains how PureBLAS handles those numbers — the
**PDM ladder** — and gives you the vocabulary and the workflow to read, audit, and add one.

It is written for a developer new to the codebase. After reading it you should be able to
look at a constant like `_CPOTRF_NBMAX = 64 + 16·W` and know *why* it has that shape, and
know what to do when you are tempted to write `const _MY_BLOCK = 256`.

## 1. Why tuning constants exist, and the one rule

Dense linear algebra performance is mostly about fitting the right amount of data in the
right level of the memory hierarchy and keeping enough independent FMA chains in flight.
Both depend on the machine: cache sizes, SIMD width, register count, microarchitecture.
So every real BLAS is parameterized by dozens of machine-dependent numbers.

Static C/Rust BLAS libraries (OpenBLAS, BLIS, AOCL) solve this with **hand-tuned per-µarch
tables** baked in at *their* compile time. That works for CPUs their maintainers
benchmarked and silently mis-sizes on everything else.

PureBLAS is Julia. Julia JITs to the host at load time, which means PureBLAS can
**compute** the right sizes for the *actual* machine it is running on — including CPUs
nobody ever benchmarked (a new laptop, a cloud box). This is the project's one structural
advantage over static BLAS, and hardcoded literals throw it away. Hence the rule
(`CLAUDE.md` requirements #7, #8, #8b):

> **The PDM ladder (Pin → Derive → Measure).** Every machine-dependent constant is a
> *self-tuning constant*. It resolves in exactly this order:
>
> - **P — Pin:** `@load_preference("name", <default>)`. A set Preference always wins
>   (calibration, cross-compilation, and the trim/`.so` build — which *must* pin any
>   Measure-tier knob, because a runtime benchmark is not trim-safe).
> - **D — Derive:** if the optimum is **physically predictable from a detected const**
>   (cache residency, SIMD width, register count), the default is a **formula** over
>   `_L1_BYTES`/`_L2_BYTES`/`_L3_BYTES`, `_vwidth`, `_NVREG`, … Zero runtime cost,
>   const-folds (trim-safe), adapts to unseen machines.
> - **M — Measure:** if the optimum is **not** predictable from detected consts — it
>   depends on port balance, the prefetcher, or write-stream behavior, and can *invert
>   sign* across microarchitectures — the default is an on-host auto-tune: a
>   `Base.OncePerProcess` that benchmarks a **formula-bounded candidate set** on first
>   use. See `_ger_np` below.

The decision rule between D and M **is** the rule:

> *"Is the optimum physically predictable from a detected const?"*
> **Yes ⇒ Derive. No ⇒ Measure. There is no third option.**

A bare literal (`const _KC = 256`) or a datapath-gated literal
(`_double_pumped(_HW) ? 8 : 4`) is not a valid answer — it is a Measure-tier knob that
has not been converted yet: correct only on the machines someone benchmarked, wrong on
the next one. A literal is permitted only as a documented **exception** — either a
proven machine-*independent* invariant, or a validated literal whose derivation was
tried and falsified — and it must say so in its comment (Section 4 catalogs these).
The test suite enforces this: `test/req8_lint_tests.jl` fails on any *new* un-derived
tuning literal that lacks a `# req8-ok:` justification.

### Where the detected consts come from

The raw material for every Derive formula lives in `src/cpuinfo.jl`. The build machine is
probed **once, at package load/precompile time**, via `CpuId.jl` / `HostCPUFeatures` /
`CPUSummary`, and the results are baked into plain `const`s. This satisfies the
trim-safety requirement (no `cpuid` ccall at runtime — the consts fold into the generated
code) and every one is overridable through `Preferences` (for cross-compiling, pinning a
`.so` build, or correcting a detection heuristic).

| Const | What it is | Source / fallback |
|---|---|---|
| `_SIMD_BYTES` | Widest SIMD register, in bytes (64 = AVX-512, 32 = AVX2, 16 = NEON/SSE2) | `CpuId.simdbytes()`; pref `"simd_bytes"`; fallback 16 |
| `_vwidth(T)` | SIMD lane count for element type `T`: `_SIMD_BYTES ÷ sizeof(T)` (the ubiquitous **W**) | derived from `_SIMD_BYTES` |
| `_L1_BYTES` | L1 data cache, bytes | `CPUSummary.cache_size(Val(1))`; fallback 32 KiB |
| `_L2_BYTES` | L2 cache, bytes | `CPUSummary.cache_size(Val(2))`; fallback 512 KiB |
| `_L3_BYTES` | L3 **total** size, bytes (`CPUSummary` reports a per-core *share*; L3 is shared, and nc-blocking wants the total, so this uses `CpuId.cachesize()[3]`) | pref `"l3_bytes"`; fallback 8 MiB |
| `_CACHELINE` | Cache-line size (prefetch density unit) | `CpuId.cachelinesize()`; fallback 64 |
| `_L1D_ASSOC`, `_L1_WAY_BYTES` | L1d associativity (raw `cpuid` leaf) and the derived way stride — the address stride at which parallel streams collide in one L1 set (caps gemv-N stream counts on power-of-2 `lda`) | pref `"l1d_assoc"`; fallback 8-way |
| `_NVREG` | Architectural vector registers: 32 (AVX-512, AArch64) vs 16 (AVX2/SSE) — the register-budget cap | derived from `_SIMD_BYTES` + arch |
| `_CPU_VENDOR`, `_CPU_FAMILY` | Vendor symbol and display family (e.g. AMD `0x19` = Zen3/Zen4, `0x1A` = Zen5) | `CpuId.cpuvendor()` / `cpumodel()`; pref `"cpu_family"` |
| `_INTEL_AVX2` | Intel-AVX2-without-AVX-512 (Haswell/Broadwell class): narrow OOO window makes serial FMA reductions latency-bound; kernels opt into extra-accumulator splits on this | vendor + feature bits (width and cache size *cannot* distinguish µarchs with the same ISA) |
| `_HW` | The whole machine as a `NamedTuple` const (`simd`, `l1`, `l2`, `l3`, `vendor`, `family`, `nvreg`). Derivation functions take `hw` as an argument, so they are **pure** — `test/autotune_tests.jl` feeds descriptors of every fleet box offline and asserts each formula reproduces the measured optima | assembled from the above |
| `_double_pumped(hw)` | Full-width registers but **half-width FP datapath** (a 512-bit op occupies the 256-bit pipes twice). Not CPUID-discoverable — the one legitimate family lookup, a silicon *fact*: AMD family `0x19` + AVX-512 = Zen4 | `hw.vendor`/`hw.family`/`hw.simd` |
| `_datapath_bytes(hw)` | Effective FP datapath width: `simd ÷ 2` if double-pumped, else `simd` | derived |

## 2. Glossary

The vocabulary you will hit in every kernel file. Skim it now, come back when a comment
confuses you.

| Term | What it physically is | Example |
|---|---|---|
| **W** | SIMD lanes per register for the element type (`_vwidth(T)`): 8 for `Float64` on AVX-512, 4 on AVX2 | everywhere |
| **NB** | Block / panel width of a blocked LAPACK factorization: columns factored by the slow BLAS-2 panel per iteration before handing the trailing update to BLAS-3 | `_qr_nb`, `_lu_nb`, `_chol_nb` |
| **MR / NR** | Register microkernel tile: MR row-blocks × NR columns of C held *in registers* through the k-loop. Bounded by `_NVREG` | `_at_gemm_mr`, `_at_gemm_nr`, `_CPOTF2_MR` |
| **KC / MC / NC** (cache blocks) | The BLIS 5-loop cache blocking: KC = k-depth so a B micropanel stays L1-resident; MC = A-block rows so `mc·kc` stays in L2; NC = B-block columns so `kc·nc` stays in L3 | `_at_gemm_kc/_mc/_nc` |
| **NC / NP** (streams) | In Level-2 kernels: columns processed per pass / concurrent memory streams (each stream = one column being read or RMW-updated) | `_ger_np`, `_CGEMVT_NC`, `_gemvn_minner_np` |
| **base / base-case cutoff** | The size at/below which a recursion or blocked driver stops and runs a simple unblocked kernel | `_POTRF_BASE`, `_GETF2_BASE`, `_CPOTRF_BASE` |
| **crossover / algorithm-switch threshold** | The size where one *algorithm* stops winning and another takes over (not a block size — a routing decision) | `_QR_UNBLK_MAX`, `_zqr_unblk_max`, `_SVD_DC_CROSS` |
| **unblocked vs blocked** | Unblocked = straight BLAS-2 loop over the whole problem (low overhead, poor cache reuse). Blocked = factor an NB-wide panel, then update the trailing matrix with BLAS-3 | `geqrf!` in `src/lapack/qr.jl` |
| **panel** | The tall-skinny `m×NB` submatrix a blocked factorization processes per step; also a group of columns a Level-2 kernel sweeps together | QR/LU panel, ger NP-column panel |
| **unroll / fuse factor (U)** | How many W-wide steps one loop iteration performs — independent accumulators to cover FMA latency (ILP) | `_GER_PANEL_U = 4`, `_GEMVN_MINNER_U = 4` |
| **compact-WY** | The LAPACK blocked-Householder representation `Q = I − V·T·Vᵀ`: lets a panel's NB reflectors be applied to the trailing matrix as two gemms instead of NB rank-1 updates | `src/lapack/wy.jl` |
| **residency** | "Working set X stays in cache level Y across the sweep" — the criterion behind almost every block size | `kc·nr·sizeof(T) ≤ ½·L1` |
| **double-pumped** | See `_double_pumped` above: 512-bit registers fed through 256-bit pipes (Zen4) | `_datapath_bytes` |

## 3. The three physical criteria

Every Derive formula keys on one of three physical mechanisms. When you read a derived
const, your job is to run this decoding: *which detected consts appear, what fraction of
which cache (or which budget) do they express, and what do the clamp bounds protect?*

### 3.1 Cache residency (block sizes)

The block size is chosen so a working set stays resident in a specific cache level while
the kernel sweeps over it. The canonical shape is
`size = capacity_share ÷ footprint_per_unit`, clamped by overhead floors and caps.

**Worked example — `_qr_nb` (`src/lapack/qr.jl`), the real blocked-QR panel width:**

```julia
@inline _qr_nb(m::Int, n::Int) = (
    fl = 256 ÷ _NVREG;
    clamp(fl * cld(m * n * sizeof(Float64), _L2_BYTES), fl, 32)
)
```

Decode it term by term:

- `cld(m·n·8, _L2_BYTES)` — *how many times does the matrix overflow L2?* While the
  matrix is L2-resident the trailing update re-streams from cache and a narrow panel is
  fine; once it spills, each panel re-streams the trailing matrix from DRAM (total sweep
  traffic ∝ k/nb), so nb must **grow** with the overflow factor to bound DRAM bytes.
- `fl = 256 ÷ _NVREG` — the floor keys on **register count**, not vector width, because
  the µarch relationship *inverts*: fewer vector registers (AVX2's 16 vs AVX-512's 32)
  ⇒ a smaller gemm microkernel tile ⇒ the trailing gemm needs a *deeper* panel to
  amortize. So `nb_floor ∝ 1/_NVREG`: Zen4/Zen5 (32 regs) → 8, Zen3 (16 regs) → 16.
  The 256 scale is fleet-calibrated (measured Zen4 floor 8 × `_NVREG` 32).
- `cap 32` — where the BLAS-2 panel share (≈ 0.75·nb/n) starts to bite; also equals the
  old flat value, so large sizes are regression-free by construction.

Measured-validated on Zen4 (8→16→32) and Zen3 (16→24→32); the formula reproduces both
within ~5% — that fleet validation is what lets it be *trusted to extrapolate*.

More residency formulas to pattern-match against:

- `_at_gemm_kc(hw, T)` (`cpuinfo.jl`): the B micropanel per k-step, `kc·nr·sizeof(T) ≤ ½·L1`.
- `_at_gemm_mc` / `_at_gemm_nc`: A-block ≤ ~30% of L2 (the rest streams B/C/prefetch);
  B-block ≤ ¼ of shared L3, with `_avoid_po2` dodging power-of-2 strides (set aliasing).
- `_zqr_nb(T, m, n)` (`qr.jl`): complex panel width, grows one base-width per ¼-L3
  overflow, `clamp(_QR_NB_C · cld(m·n·sizeof(T), _L3_BYTES ÷ 4), _QR_NB_C, 4·_QR_NB_C)`.
- `_zqr_unblk_max(T)` (`qr.jl`): the *crossover* below which unblocked complex QR wins —
  `_CGEMM_3M ? 2·_L2_BYTES : _L3_BYTES ÷ 2` (matrix bytes, not a dimension: the BLAS-2
  panel wins exactly while its O(n³) re-streams hit cache rather than DRAM, and the
  relevant cache level differs between the AVX2/3M and wide-SIMD complex-gemm regimes).
- `_GEMVN_MINNER_MAXA` (`src/blas2/level2.jl`): `4·_L3_BYTES` — the m-inner gemv-N path
  helps the mid-n/L3 regime but its y-restream loses in deep DRAM, so it is capped at
  "A no bigger than a few × L3".
- `_CGETF2_BASE` (`src/lapack/lu.jl`): `_GETF2_BASE · (sizeof(ComplexF64) ÷
  sizeof(Float64)) = 32` — a *sizeof-derived* sibling: the complex panel base is wider
  than the real one by exactly the element-size ratio, so `ComplexF32` tracks the same
  criterion automatically.
- `_lu_nb(n)` (`lu.jl`): `clamp((n ÷ 8) & ~15, _LU_NB, 128)` — shape-adaptive slope
  (grow nb so the rank-nb trailing gemm isn't skinny at large n), but note its floor and
  cap are *validated literals*, not residency formulas — see `_LU_NB` in Section 4.

### 3.2 Datapath and latency (unrolls, fuse factors, stream counts)

These knobs exist to keep the execution units fed: enough independent FMA chains to cover
latency, enough load streams to keep the datapath at line rate.

- **Derivable case:** `_ILP_TARGET = 16` in `cpuinfo.jl` is 2 × FMA_latency(4) ×
  FMA_ports(2) — the number of independent accumulator chains needed to saturate the FMA
  pipes; `_at_gemm_mr/_at_gemm_nr` split that budget into a tile. `_CPOTF2_MR`
  (`lapack.jl`) is a line-rate match: unroll rows until one step consumes a 64-byte cache
  line at datapath width — `max(1, 64 ÷ _datapath_bytes(_HW))`, i.e. MR=2 on
  double-pumped Zen4 and AVX2, MR=1 on native-512 Zen5/Intel. Both are Derive: the
  optimum follows from published, detected silicon facts.
- **Non-derivable case:** the DRAM-bound `ger!` stream count `_ger_np` (Section 3.4 shows
  the mechanism). The measured optima are Zen5 → 1, Zen3 → 4, Zen4 → 8 — with every
  external cause eliminated (memory subsystem scales fine on all three, DIMMs
  rank-matched, OS/clock cancel in the ratio, 4K aliasing padded out, identical LLVM
  codegen on Zen4/Zen5). The optimum is an intrinsic per-core property of the
  prefetcher/write-stream machinery with **no detected const that predicts it**, and it
  *inverts sign* across µarchs — that inversion is the tell that a knob is Measure-tier,
  not Derive-tier. Any formula we wrote would just be a disguised per-µarch table.

### 3.3 ISA / register count (width granularity, budgets, candidate bounds)

The third family of constraints comes from the ISA itself:

- **Width granularity:** blocks and unrolls are rounded to multiples of W
  (`_round_dn(x, W)`), because a partial vector means masked tails.
- **Register budget:** microkernel tiles must fit `_NVREG` — e.g. `_at_gemm_mr` caps MR
  at `(nvreg − 1) ÷ (nr + 1)` (accumulators + A-loads + one B-broadcast), and
  `_acc_cap(hw, T) = (nvreg − 4)·W` — the register file's accumulator capacity in
  elements — is the base quantity behind the gemm/rank-k pack crossovers.
- **Candidate-set bounds:** in Measure-tier knobs, even the *search space* should be
  Derived (a stream count bounded by `_NVREG`, an unroll bounded by the register budget)
  so that both the bounds and the selection adapt to unseen hardware.

## 4. Constants that are NOT pure formulas — and why

The PDM ladder allows literals only as documented exceptions. This table is the current
catalog. Two honest categories show up: **proven invariants** (the number is genuinely
machine-independent — an algorithm property or a bound that holds on any real machine)
and **validated literals** (a derivation was attempted and *falsified*, or the fleet data
to derive from doesn't exist yet — the literal preserves measured-good behavior until it
does). The second category is acknowledged debt, not a loophole.

| Const | Value | What it controls | Why it's not a formula | PDM classification |
|---|---|---|---|---|
| `_STEDC_NB` (`eigen_dc.jl`) | 25 | Symmetric-tridiagonal eigensolver: `steqr` ↔ divide-and-conquer base-case switch | LAPACK `dstedc`'s SMLSIZ — an **algorithm-intrinsic** crossover between two algorithms' O() constants, machine-independent. Not a hardware knob at all. | Proven-invariant — **Exempt** |
| `_SVD_DC_CROSS` (`svd.jl`) | 1 | SVD with vectors: `bdsqr` (QR sweeps) ↔ divide-and-conquer | Currently a **correctness override**, not a perf choice: the `bdsqr!` port fails on near-degenerate σ clusters (non-convergence, or silently dropped largest σ on graded bidiagonals), so *all* with-vectors SVD routes through D&C. If `bdsqr` convergence is fixed, the perf crossover (bdsqr won n≤96 pre-fix) returns — and it is algorithm-intrinsic like SMLSIZ, still not cache-derived. | Proven-invariant / correctness-pinned — **Exempt** (revisit only if `bdsqr!` is fixed) |
| `_QR_UNBLK_MAX` (`qr.jl`) | 32 | Real QR: unblocked ↔ blocked crossover (dimension-wise, so tall-skinny fronts stay unblocked) | The per-µarch unblocked crossover is **unmeasured off Zen4**. A vwidth-scaled formula would shrink Zen3's proven-good unblocked range (32 → 16) with no data behind it. Pinned at the old flat value so every µarch keeps master's validated n≤32 behavior. | Validated-literal, **pending fleet data** (derive when fleet unblocked-crossover measurements exist) |
| `_LU_NB` (`lu.jl`) | 48 (floor of `_lu_nb`; cap 128) | Real LU blocked panel width base | The residency derivation was **fleet-falsified**: a full nb×n sweep on Zen4+Zen3 shows a flat, roughly µarch-invariant optimum whose curve is *parity-bumpy* (multiples of 64 win — 128, 192 beat 168), not residency-shaped; the derived `_l1_block` value (168 on Zen3) lands in a measured *trough*, worse than the literal. The true large-n optimum is per-µarch by only ~0.5–1.5% — a formula would add spurious variation for no gain. | Validated-literal (**derivation falsified**; Measure-candidate only if that last ~1% ever matters) |
| `_GETF2_BASE` (`lu.jl`) | 16 | LU panel: flat rank-1 sweep ↔ recursive BLAS-3 split | A **store-traffic algorithm-switch crossover**, not a cache-residency block — the split halves O(pb²·mp) store traffic regardless of cache size. Validated by the gate (getrf gates vs OB+AOCL fleet-wide); a residency formula would add spurious variation (cf. the falsified `_LU_NB` derivation). Its complex sibling `_CGETF2_BASE` *is* sizeof-derived from this anchor. | Proven-invariant — **Exempt** (invariant crossover, validated-by-gate) |
| `_TR_TB` (`lapack.jl`) | 32 | Cache-blocked triangular-transpose tile (the upper→lower reuse levers) | Residency bound that holds **universally**: two 32² Float64 tiles = 16 KB ≤ any real L1 (complex: 32 KB, still fits typical L1). Deriving it from `_L1_BYTES` would change nothing on any machine that exists. | Proven-invariant — **Exempt** (residency-invariant) |
| `_POTRF_BASE` (`lapack.jl`) | 32 | Generic Cholesky recursion: unblocked-base cutoff (F64/Dual; F32 uses `_POTRF_BASE >> 1` — a sizeof-style derived sibling) | Measured to be a **recursion-overhead floor, not L1 residency**: the residency guess √(L1/8) = 64 is *worse* than 32; the optimum is small and µarch-flat (16–32 on both fleet AVX2/AVX-512 boxes). Fleet A/B validated (base=512 was up to 43.9× slower than OB). | Validated-literal (invariant crossover; knob `"potrf_base"`) |
| `_chol_sth` / `_chol_sb` (`lapack.jl`) | 16 / 32 | F64 fast-path Cholesky small-n blocking: diagonal base-case cap and block size (`sb = 2·sth`) | The width-scaled hypothesis was **disproven by controlled A/B** on Zen4: th=16 beats th=32(=4·W) at n=32–128, ties above — the slow left-looking base wants a *small fixed* diagonal block regardless of SIMD width, so 16 is a measured µarch-invariant crossover, correctly flat. Source comment still says "fleet-tuned later". | Validated-literal, **pending fleet confirmation** (measured invariant on 2 boxes; knob-able) |

Note what the table does *not* contain: `_CPOTRF_BASE`, `_CPOTF2_MR`, `_CPOTRF_NBMAX`,
and `_CGETF2_BASE` were all literals once and are now derived
(`_at_cpotrf_base(_HW) = 32 + 4·W`, `_at_cpotf2_mr = 64 ÷ datapath`,
`_at_cpotrf_nbmax = 64 + 16·W`, `sizeof`-ratio). The affine-in-W shape of the cpotrf pair
is worth internalizing: a width-independent overhead floor plus a per-lane slope is the
common form for implementation crossovers that are cache-independent but width-dominant.
The table above is the *residue* after that migration, and each row states which kind of
residue it is.

## 5. The Measure tier: `_ger_np` walkthrough

When the D-vs-M question answers "No", the pattern to copy is `_ger_np` in
`src/blas2/level2.jl` (`CLAUDE.md` 8b names `_ger_np`/`_gemvt_nc` as the pattern; as of
this writing `_ger_np` is the implemented instance — the real gemv-T column-block still
rides a fixed `Val(4)` and the complex one a pinned `_CGEMVT_NC`, i.e. Measure-tier
candidates not yet converted). Four pieces, in order:

```julia
# 1. Pin: a set Preference always wins.
const _GER_NP_PREF = @load_preference("ger_panel_np", nothing)

# 2. @static if: when pinned, the auto-tune branch is NEVER DEFINED.
@static if isnothing(_GER_NP_PREF)
    function _measure_ger_np()::Int
        Base.generating_output() && return 4      # no benchmarking during precompilation
        try
            # DRAM-bound problem (~2×L3 rows), pre-touched pages,
            # candidates (1, 2, 4, 8), warmup absorbs each Val's JIT,
            # min-of-4 timing, best wins.
            ...
        catch
            return 4                              # safe middle; see poisoning note below
        end
    end
    const _GER_NP_ONCE = Base.OncePerProcess{Int}(_measure_ger_np)
    @inline _ger_np() = _GER_NP_ONCE()
else
    @inline _ger_np() = _GER_NP_PREF::Int
end
```

Why each piece is the way it is:

- **`@load_preference` pin first.** `bench/calibrate.jl` writes the Preference after a
  proper locked-frequency calibration; a pinned box never benchmarks itself.
- **`@static if`, not dead-code-elimination-by-faith.** When the Preference *is* set, the
  measuring function and the `OncePerProcess` const are never even defined — the pinned
  build is trivially trim-clean rather than hopefully trim-clean.
- **`Base.OncePerProcess`, not `__init__`.** The benchmark runs at most once per process,
  lazily, on the first DRAM-bound `ger!` call — so a process that never hits that path
  never pays for it, and there is no load-time side effect. Two guards matter: the
  `Base.generating_output()` early-out (don't burn a benchmark during precompilation) and
  the total `try/catch` (an `OncePerProcess` whose initializer throws **poisons every
  subsequent call** in the process, so the initializer must be infallible, falling back
  to the safe middle value).
- **Formula-bounded candidate set.** The candidates are few and bounded by a Derived
  quantity (a stream count can't usefully exceed the register/associativity budget — cf.
  `_NVREG`, `_L1D_ASSOC`), so both the bounds *and* the selection adapt to unseen
  hardware instead of being a hidden per-µarch table.
- **The `.so` build MUST pin.** `juliac --trim` cannot contain a runtime benchmark.
  `juliac/build.jl` sets `ger_panel_np => 4` for the duration of the build (and restores
  the previous state after), which flips the `@static if` to the pinned branch. Any new
  Measure-tier knob needs the same treatment in `juliac/build.jl` — forgetting it breaks
  the trim build.

One more trap, learned the hard way (`_POTRF_BASE_F32`): `@load_preference` **must sit in
a top-level `const`**, never inside a function body — in a function it becomes a per-call
runtime Preferences lookup, which dominated microsecond-scale small-n factorizations.

## 6. How to add or change a tuning constant (checklist)

1. **Confirm it's actually a tuning constant.** Machine-dependent block size, cutoff,
   panel width, unroll, fuse factor, stream count, or algorithm-switch threshold ⇒ yes,
   the PDM ladder applies. A correctness bound (e.g. `lassq` scaling) ⇒ no — different
   rules, don't "tune" it.
2. **Ask the D-vs-M question:** *is the optimum physically predictable from a detected
   const?* Cache residency, SIMD width, register count, datapath width ⇒ **Derive**.
   Port balance / prefetcher / stream behavior, or your model mispredicts a box you
   *have* ⇒ **Measure**. There is no third option; if you can produce neither a formula
   nor a measure harness, stop and flag it.
3. **Derive tier:** write the default as a pure formula over `_HW` fields (an
   `_at_*(hw, T)` function in `cpuinfo.jl` if it's shared, or inline over the detected
   consts). Take cache size **and** ISA into account together — a block size depends on
   what fits in cache *and* the vector width. Round to W, clamp with a stated floor/cap.
   Add the fleet descriptors' expected values to `test/autotune_tests.jl`.
4. **Measure tier:** mirror `_ger_np` exactly — top-level `@load_preference` const,
   `@static if isnothing(pref)`, `Base.OncePerProcess` with `generating_output()` guard
   and total `try/catch`, formula-bounded candidates. Then **add the pin to
   `juliac/build.jl`** (the trim build must never contain the benchmark branch) and a
   calibration entry to `bench/calibrate.jl`.
5. **Keep the Preferences override** in both tiers (`@load_preference("name", <derived
   default>)`): pinning, cross-compilation, and calibration all ride it.
6. **Write the comment** — every tuning const cites (a) which detected consts it derives
   from, (b) the physical criterion (residency share / latency budget / register budget /
   store-traffic crossover), and (c) its PDM tier. If it's a deliberate literal, say
   *why* (invariant? falsified derivation? pending fleet data?) and add `# req8-ok:
   <reason>` — otherwise `test/req8_lint_tests.jl` will fail on it. Watch for the
   "derived selection, hardcoded materialization" trap: `Val(8)` at a call site instead
   of `Val(_THE_CONST)` is a violation the lint catches.
7. **Validate on the fleet before trusting it.** A derived formula must *reproduce the
   measured optima on the known fleet* before it's trusted to extrapolate to unseen
   machines. All gate measurements run under `sudo bench/fleet_freqlock.sh lock`
   (passive governor, boost off, cores pinned to base clock, frequency verified under
   load) — a run whose verify step fails is invalid, full stop. Use controlled
   same-machine A/B comparisons, never cross-run one-offs.
8. **Be ready for falsification.** `_LU_NB` and `_chol_sth` both killed plausible
   formulas with measurements. A validated literal with an honest comment beats an
   elegant formula that adds spurious variation — but only *after* the derivation was
   tried and measured, never instead of trying.
