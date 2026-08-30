# The Tuning-Constant System

Every fast BLAS is full of magic numbers: block sizes, panel widths, unroll factors,
algorithm-switch thresholds. This page covers how PureBLAS handles those numbers — the
PDM ladder — and gives you enough vocabulary to read one, audit it, or add your own.

It is written for someone new to the codebase. By the end you should be able to look at a
constant like `_CPOTRF_NBMAX = 64 + 16·W` and see why it has that shape, and know what to
do when you catch yourself about to write `const _MY_BLOCK = 256`.

## 1. Why tuning constants exist, and the one rule

Dense linear algebra performance is mostly about fitting the right amount of data in the
right level of the memory hierarchy and keeping enough independent FMA chains in flight.
Both depend on the machine: cache sizes, SIMD width, register count, microarchitecture.
So every real BLAS is parameterized by dozens of machine-dependent numbers.

Static C and Rust BLAS libraries (OpenBLAS, BLIS, AOCL) solve this with hand-tuned
per-µarch tables baked in at *their* compile time. That works on the CPUs their
maintainers benchmarked, and silently mis-sizes on everything else.

PureBLAS is Julia, and Julia compiles to the host at load time — so it can work out the
right sizes for the machine actually running it, including CPUs nobody has ever
benchmarked. That is the project's one structural advantage over a static BLAS, and a
hardcoded literal throws it away. Hence the rule (`CLAUDE.md` requirements #7, #8, #8b):

> **The PDM ladder (Pin → Derive → Measure).** Every machine-dependent constant is a
> *self-tuning constant*, resolved in exactly this order:
>
> - **Pin** — `@load_preference("name", <default>)`. A Preference that is set always wins.
>   This is how calibration, cross-compilation and the trim/`.so` build get their say, and
>   the `.so` build has to pin any Measure-tier knob, since a runtime benchmark is not
>   trim-safe.
> - **Derive** — when the optimum follows from a detected const (cache residency, SIMD
>   width, register count), the default is a formula over `_L1_BYTES`/`_L2_BYTES`/
>   `_L3_BYTES`, `_vwidth`, `_NVREG` and friends. It costs nothing at runtime, const-folds
>   away, and adapts to machines nobody has seen.
> - **Measure** — when it does not follow from anything detectable, because it turns on
>   port balance, the prefetcher or write-stream behaviour and can flip sign between
>   microarchitectures, the default is an on-host auto-tune: a `Base.OncePerProcess` that
>   benchmarks a formula-bounded candidate set on first use. `_ger_np` below is the model.

Everything turns on one question:

> *Is the optimum physically predictable from a detected const?*
> Yes ⇒ Derive. No ⇒ Measure. There is no third option.

A bare literal (`const _KC = 256`), or one gated on the datapath
(`_double_pumped(_HW) ? 8 : 4`), is not an answer to that question. It is a Measure-tier
knob nobody has converted yet — right on the machines someone benchmarked, wrong on the
next one. Literals are allowed only as documented exceptions: a genuinely
machine-independent invariant, or a value whose derivation was tried and falsified. The
comment has to say which, and Section 4 lists every one of them. `test/req8_lint_tests.jl`
fails on any new un-derived tuning literal that lacks a `# req8-ok:` justification.

### Where the detected consts come from

The raw material for every Derive formula lives in `src/cpuinfo.jl`. The build machine is
probed once, at load/precompile time, through `CpuId.jl`, `HostCPUFeatures` and
`CPUSummary`, and the answers are baked into plain `const`s. That keeps the trim build
safe — there is no `cpuid` ccall at runtime, because the consts fold into the generated
code — and every one can be overridden through `Preferences` when you are
cross-compiling, pinning a `.so` build, or correcting a bad detection.

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

Every Derive formula keys on one of three physical mechanisms. Reading one means asking
the same three things each time: which detected consts appear, what share of which cache
(or which budget) they express, and what the clamp bounds are protecting.

### 3.1 Cache residency (block sizes)

The block size is chosen so a working set stays resident in a specific cache level while
the kernel sweeps over it. The canonical shape is
`size = capacity_share ÷ footprint_per_unit`, clamped by overhead floors and caps.

Worked example: `_qr_nb` (`src/lapack/qr.jl`), the real blocked-QR panel width.

```julia
@inline _qr_nb(m::Int, n::Int) = (
    fl = 256 ÷ _NVREG;
    clamp(fl * cld(m * n * sizeof(Float64), _L2_BYTES), fl, 32)
)
```

Decode it term by term:

- `cld(m·n·8, _L2_BYTES)` asks how many times the matrix overflows L2. While it stays
  L2-resident the trailing update re-streams from cache and a narrow panel is fine; once
  it spills, every panel re-streams the trailing matrix from DRAM (sweep traffic ∝ k/nb),
  so nb has to grow with the overflow factor to keep those bytes bounded.
- `fl = 256 ÷ _NVREG` is the floor, and it keys on register count rather than vector
  width because the relationship inverts: fewer vector registers (AVX2's 16 against
  AVX-512's 32) means a smaller gemm microkernel tile, which means the trailing gemm needs
  a *deeper* panel to amortize. So `nb_floor ∝ 1/_NVREG` — 8 on Zen4/Zen5 with 32
  registers, 16 on Zen3 with 16. The 256 scale is fleet-calibrated, from the measured Zen4
  floor of 8 × `_NVREG` 32.
- `cap 32` is where the BLAS-2 panel's share (≈ 0.75·nb/n) starts to bite. It is also the
  old flat value, so large sizes cannot regress.

Validated on Zen4 (8→16→32) and Zen3 (16→24→32), where the formula reproduces both within
about 5%. That fleet check is what earns it the right to extrapolate.

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

The derivable case: `_ILP_TARGET = 16` in `cpuinfo.jl` is 2 × FMA_latency(4) ×
FMA_ports(2), the number of independent accumulator chains needed to saturate the FMA
pipes, and `_at_gemm_mr`/`_at_gemm_nr` split that budget into a tile. `_CPOTF2_MR`
(`lapack.jl`) matches line rate instead: unroll rows until one step consumes a 64-byte
cache line at datapath width, `max(1, 64 ÷ _datapath_bytes(_HW))` — MR=2 on double-pumped
Zen4 and on AVX2, MR=1 on native-512 Zen5 and Intel. Both are Derive, because the optimum
follows from published silicon facts we can detect.

The non-derivable case: the DRAM-bound `ger!` stream count `_ger_np`. The measured optima
are 1 on Zen5, 4 on Zen3, 8 on Zen4, with every external cause ruled out — the memory
subsystem scales fine on all three, the DIMMs are rank-matched, OS and clock effects
cancel in the ratio, 4K aliasing was padded out, and Zen4 and Zen5 get identical LLVM
codegen. What is left is an intrinsic property of each core's prefetcher and write-stream
machinery, and no detected const predicts it. It also flips sign between
microarchitectures, which is the giveaway that a knob belongs in Measure rather than
Derive: any formula would just be a per-µarch table in disguise.

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

## 4. Constants that are not pure formulas — and why

The PDM ladder allows literals only as documented exceptions, and this table is the
current catalog. They come in two kinds. A *proven invariant* is genuinely
machine-independent — an algorithm property, or a bound that holds on any real machine. A
*validated literal* is one whose derivation was tried and falsified, or where the fleet
data needed to derive it does not exist yet; it preserves the measured-good behaviour
until that changes. The second kind is debt, and it is tracked as debt.

| Const | Value | What it controls | Why it's not a formula | PDM classification |
|---|---|---|---|---|
| `_STEDC_NB` (`eigen_dc.jl`) | 25 | Symmetric-tridiagonal eigensolver: `steqr` ↔ divide-and-conquer base-case switch | LAPACK `dstedc`'s SMLSIZ — an **algorithm-intrinsic** crossover between two algorithms' O() constants, machine-independent. Not a hardware knob at all. | Proven-invariant — **Exempt** |
| `_SVD_DC_CROSS` (`svd.jl`) | 1 | SVD with vectors: `bdsqr` (QR sweeps) ↔ divide-and-conquer | Currently a **correctness override**, not a perf choice: the `bdsqr!` port fails on near-degenerate σ clusters (non-convergence, or silently dropped largest σ on graded bidiagonals), so *all* with-vectors SVD routes through D&C. If `bdsqr` convergence is fixed, the perf crossover (bdsqr won n≤96 pre-fix) returns — and it is algorithm-intrinsic like SMLSIZ, still not cache-derived. | Proven-invariant / correctness-pinned — **Exempt** (revisit only if `bdsqr!` is fixed) |
| `_QR_UNBLK_MAX` (`qr.jl`) | 32 | Real QR: unblocked ↔ blocked crossover (dimension-wise, so tall-skinny fronts stay unblocked) | The per-µarch unblocked crossover is **unmeasured off Zen4**. A vwidth-scaled formula would shrink Zen3's proven-good unblocked range (32 → 16) with no data behind it. Pinned at the old flat value so every µarch keeps master's validated n≤32 behavior. | Validated-literal, **pending fleet data** (derive when fleet unblocked-crossover measurements exist) |
| `_LU_NB` (`lu.jl`) | 48 (floor of `_lu_nb`; cap 128) | Real LU blocked panel width base | The residency derivation was **fleet-falsified**: a full nb×n sweep on Zen4+Zen3 shows a flat, roughly µarch-invariant optimum whose curve is *parity-bumpy* (multiples of 64 win — 128, 192 beat 168), not residency-shaped; the derived `_l1_block` value (168 on Zen3) lands in a measured *trough*, worse than the literal. The true large-n optimum is per-µarch by only ~0.5–1.5% — a formula would add spurious variation for no gain. | Validated-literal (**derivation falsified**; Measure-candidate only if that last ~1% ever matters) |
| `_GETF2_BASE` (`lu.jl`) | 16 | LU panel: flat rank-1 sweep ↔ recursive BLAS-3 split | A **store-traffic algorithm-switch crossover**, not a cache-residency block — the split halves O(pb²·mp) store traffic regardless of cache size. Validated by the gate (getrf gates vs OB+AOCL fleet-wide); a residency formula would add spurious variation (cf. the falsified `_LU_NB` derivation). Its complex sibling `_CGETF2_BASE` *is* sizeof-derived from this anchor. | Proven-invariant — **Exempt** (invariant crossover, validated-by-gate) |
| `_CIAMAX_SIMD_MIN` (`lu.jl`) | `max(4·_vwidth, 128)` | Complex LU panel: SIMD complex argmax ↔ scalar `cabs1` argmax | An **algorithm cost-ratio crossover** with no cache or ISA quantity behind it — and it is µarch-invariant, so no `_vwidth` formula can express it. `_iamax_cmplx_simd!` carries ~197 cycles of fixed setup, so at panel lengths the scalar scan wins outright. Fleet-measured SIMD/scalar time ratio (>1 = scalar wins): Zen3 `1.86 / 1.28 / 0.98 / 0.84`, Zen4 `3.17 / 1.96 / 1.02 / 0.66`, Zen5 `3.36 / 1.38 / 0.94 / 0.66` at nrows `32 / 64 / 128 / 256` — crossover ≈128 on all three. **This constant replaced a guard that was not a crossover at all**: `nrows ≥ 4·_vwidth` only ever encoded the SIMD kernel's OOB-lane *legality* bound, then got reused as a speed threshold. It was 4–8× too low, so every column of a 50-row panel took the slow path (izamax was 35% of the zgetrf(50) panel). Worse on the wider box — `4W` is 16 on AVX2 and 32 on AVX-512 — which presents as "AVX-512 buys nothing". The `max(4·_vwidth, …)` form keeps the legality bound explicit so it cannot be lost. **Per-kernel, not a rule:** the real sibling `_iamax_simd!` has no such fixed cost and wins at every length, so do not raise its guard. | Validated-literal (invariant crossover, fleet table above) |
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

!!! note "The full inventory lives in [the Knob Registry](knobs.md)"
    Every `@load_preference` key in `src/` — 118 of them — with its owning routine, default, syntactic
    tier and `# PDM:` justification. That page is **generated** from the source by
    `test/knob_registry.jl` and byte-verified by the suite, so it cannot drift. §4 and §4b below are the
    hand-written *analysis*: which exceptions exist and why they are defensible.

## 4b. Predicate-keyed knobs — the justification register

Section 4 catalogs knobs that ended up a literal where a formula was wanted. This section
covers the other non-formula shape: a knob keyed on a µarch predicate (`_double_pumped`,
`_wide_simd`, `_datapath_bytes`). `CLAUDE.md` 8b allows those only when the knob's physical
criterion really is the fact the predicate encodes, argued in the comment and backed by a
fleet table. A predicate used as a convenient two-way lookup is a violation.

The register exists because a binary predicate over a three-box fleet is barely
constrained at all. There are only three non-trivial ways to partition three points and we
use two of them, so "it reproduces the fleet" is worth about one bit of evidence. What
separates a derivation from a curve fit is not the fit — it is a named mechanism and a
stated falsifier. Every row below carries both, or says plainly that it does not.

`test/predicate_knob_lint.jl` enforces it: any `_at_*` whose body calls a µarch predicate
has to appear here, and a new one fails the suite until it is justified. Without that, a
wrong justification can sit in the source for weeks, because only a human reading it
closely would ever notice.

| Knob | Predicate | Mechanism argued? | Falsifier — what would kill it | Grade |
|---|---|---|---|---|
| `_at_zaxpy_narrow` | `_datapath_bytes(hw) <= 32` | **Yes, and the criterion IS the predicate.** The knob chooses a narrow vs wide complex-axpy kernel; the physical question is literally how many bytes the datapath retires per cycle. | A box whose datapath is ≤32 B and wants the wide arm (or vice versa). Zen4 double-pumps 512-bit over a 256-bit path and measured narrow 6/6 — that is the case that made this a datapath question and not a width question. | **Sound** — predicate = criterion |
| `_at_gemm_split_ok` | `nvreg >= 32 && simd >= 64` | **Yes** — an ISA *capability* test, not a tuning split: the split-reduction tile needs that many architectural registers to hold its accumulators. | A 32-register AVX-512 box where the split loses. Note this is a **feasibility** predicate; unlike the rest of the table it is not choosing between two viable arms. | **Sound** — capability, not tuning |
| `_at_gemvt_perscan` | `_double_pumped(hw) ? 1 : 0` | **No, and it says so.** Labelled a keyed literal. A `_wide_simd` mechanism (loads per cache line) was written and falsified: it gives Zen5 mode 1, which loses ~10% at n=1024. | Already falsified once. The live falsifier: Zen4 and Zen5 differ only in L1 (32 vs 48 KiB) yet cross over at x = 8 KiB and 4 KiB respectively — the **larger** L1 crosses **earlier**, anti-correlated with the only differing const, so no capacity rule fits. ⚠ Untested: Zen3 shares Zen4's 32 KiB L1; if it crosses where Zen4 crosses, an L1 rule is back in play. That measurement has not been made yet. | **Honest** — declared non-derived, one open test |
| `_at_axpy_dram`, `_at_axpy_band` | `_double_pumped(hw) ? 208 : 4` | **Yes** — arm 208 is the narrow 256-bit phase kernel, correct exactly where 512-bit ops are double-pumped over a 256-bit path. Measured +17% Zen4, loses on Zen3. | A native-512 box (not double-pumped) preferring arm 208, or a double-pumped box preferring 4. Zen5 is the live check and measures 4. | **Sound** — mechanism named |
| `_at_pbtrf_nb` (F32/CF32/CF64), `_at_pbtrf_nbs` (F64) — **panel widths** | `_wide_simd(hw) ? a : b` | **Yes.** A panel width is *counted in vector registers*, so SIMD width is the physically right unit. The evidence discriminates rather than merely fits: Zen4 and Zen5 agree on every row and share `simd=64`, while `_double_pumped` **separates** them — two independent AVX-512 boxes agreeing is what a 2-point fit cannot give. 6 fresh processes per box. | A wide-SIMD box wanting the narrow width. Note the F32 sibling `_at_pbtrf_nbs` is a *pure formula* (`_lanes(hw, Float32)`), reproduced on all three boxes — evidence the unit is right. | **Sound** — mechanism = the unit |
| `_at_pbtrf_cross` (×4), `_at_gbtrf_cross` (×3), `_at_pbtrf_ucross` (F64) — **crossovers** | `_wide_simd(hw) ? a : b` | **No — and the width mechanism does not extend here.** A crossover is a size threshold, not a register count, so "counted in vector registers" justifies the widths above and nothing in this row. Same 6×3-process evidence, but evidence of *what*, not *why*. | **Two derivations attempted and falsified (2026-08-20)** — do not re-try. Lanes-proportional: `cross/lanes` = 3, 4, 2, 2 on Zen4 vs 8, 16, 4, 8 on Zen3. Fixed byte budget: `cross × sizeof(T)` = 192, 256, 128, 128 vs 256, 512, 128, 256. Neither is constant within a box, let alone across. Also note `_at_pbtrf_ucross` F64 measures 192 on Zen3 where its own F32/C32 sibling formula (`l2 ÷ 4096`) says 128 — the family does not obey one rule. | **Honest** — declared non-derived, dead ends recorded |
| `_at_strassen_min`, `_at_trmm_rpack` | `_datapath_bytes(hw) >= 64 ? lo : hi` | **Yes** — Strassen buys a flop cut and pays in extra additions and memory passes; the trmm side-R cut decides whether to pack, which is pure traffic. Doubling the real datapath doubles per-cycle FMA throughput while neither the Strassen adds nor the pack traffic speed up, so the flops-vs-traffic crossover moves down and the threshold must follow. The predicate is `_datapath_bytes` and not `_vwidth` (cannot separate Zen4 from Zen5) nor `_double_pumped` (cannot separate Zen3 from Zen5): Zen4 double-pumps 512-bit ops over a 256-bit path, so its effective datapath is 32 B — the same as native-AVX2 Zen3 — which is exactly why those two measured flat together and Zen5 did not. | A native-64 B box preferring the shipped 1024/448, or a 32 B box preferring 256/1792. Zen5 measured 256 → 1.0155/1.0756/1.0625/1.0565/1.0000 at n=256..4096 (no losing cell) across four runs (2 unlocked + 2 locked, agreeing to ~0.5%); locked gate A/B geomean +6.6% vs OpenBLAS, +3.7% vs AOCL, gate min unchanged. Zen3+Zen4 flat on 74-cell sweeps. Note that `trmm_rpack` is probe-only: `plots.jl` has NO real trmmR row, so the gate is blind to it. | **Sound** — mechanism named, falsifier named |
| `_at_gemvn_minner` | `_datapath_bytes(hw) < 64` | **Yes** — the m-inner panel holds a row-block's `y` in registers and sweeps columns inner, so it runs ~NP concurrent strided A-streams. That trade pays while the FMA units are the bottleneck and stops paying once the datapath doubles and the loads are. Same predicate and same reason as `_at_strassen_min`/`_at_trmm_rpack`: it IS the datapath, not a lookup. | A 32 B-datapath box preferring the old path, or a native-64 B box preferring m-inner **at an aliasing lda**. Measured on all three boxes freq-locked and verified before/after: Zen3 m-inner wins (gate 0.925 vs 0.901), Zen4 wins (0.969 vs 0.953), Zen5 loses at po2 sizes (n=512 0.916, n=1024 0.964). ⚠ Caveat, and not a small one: the Zen5 negative holds only at power-of-two `lda`. At non-po2 sizes m-inner wins there by 8–21% (511/513/1023/1025/2047/2049 all win; only 512/1024/2048 lose). The gate's `L2SZ` is entirely powers of two, so this default is calibrated on the single pathological case — see the kb finding. The honest long-run form is an `lda`-aliasing test at the call site, not a compile-time boolean. | **Sound on the measured axis, incomplete on the other** — mechanism and falsifier named, but the po2-only evidence base is recorded above rather than hidden |
| `_at_potrf_udirect` | `hw.nvreg >= 32 ? 20 : 12` | **Yes** — the tiny-n uplo='U' direct path keeps the upper triangle's columns live in vector registers while factoring, so the register file sets how large `n` can get before it spills and the blocked path wins. A register-capacity crossover, hence `_NVREG` rather than width or cache. | A 32-register box preferring 12, or a 16-register box preferring 20. Calibrated per box (stabilise + anchor + 8 rounds + `decide` needing a CI excluding 1.0 and a 2% margin), freq-locked and verified: Zen3 nvreg=16 → 12 (20 costs **−5.4%**, 24 costs −14%); Zen4 nvreg=32 → **20 (+6.8%)**; Zen5 nvreg=32 → **20 (+8.2%)**. Not a two-point fit — the 16-register box actively loses at 20, so a flat 20 was unshippable. | **Sound** — mechanism named, falsifier named, three-box separation |

Reading the grades: *Sound* means the predicate encodes the physical criterion and a
falsifier is named. *Honest* means it is not a derivation, says so, and records the failed
attempts so nobody retries them. *Weak* would mean it fits the fleet with no mechanism and
no falsifier — debt in the same sense as §4's validated literals. No row is currently
graded Weak.

Note that `pbtrf`/`gbtrf` appears as two rows rather than one. The width mechanism —
a panel width is counted in vector registers, so SIMD width is the right unit — genuinely
covers the widths and does not extend to the crossovers, which are size thresholds. One
grade for the whole family would have hidden that.

Finding a knob genuinely un-derivable is not the end of the ladder; it is the entry
condition for Measure, which since the 2026-08 campaign means an offline `tune!()` pin
rather than a runtime duel. To declare it you have to name the physical quantity the
optimum depends on, show it is not among the detected consts, and show the optimum
inverting across boxes that agree on those consts. `_ger_np` in §5 is the worked example.

## 5. The Measure tier: `_ger_np` walkthrough

When the D-vs-M question comes back "No", the pattern to copy is `_ger_np` in
`src/blas2/level2.jl`. `CLAUDE.md` 8b names both `_ger_np` and `_gemvt_nc` as the pattern,
but `_ger_np` is the one fully implemented instance: the real gemv-T column-block still
rides a fixed `Val(4)` and the complex one a pinned `_CGEMVT_NC`, both Measure-tier
candidates nobody has converted yet. Four pieces, in order:

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

- The `@load_preference` pin comes first. `bench/calibrate.jl` writes that Preference after
  a proper locked-frequency calibration, and a pinned box never benchmarks itself.
- `@static if`, rather than trusting dead-code elimination. When the Preference is set, the
  measuring function and the `OncePerProcess` const are never defined at all, so the pinned
  build is trim-clean by construction instead of by hope.
- `Base.OncePerProcess`, not `__init__`. The benchmark runs at most once per process,
  lazily, on the first DRAM-bound `ger!` call, so a process that never takes that path
  never pays for it and nothing happens at load time. Two guards matter here: the
  `Base.generating_output()` early-out keeps it from burning a benchmark during
  precompilation, and the blanket `try`/`catch` is there because an `OncePerProcess` whose
  initializer throws poisons every later call in that process. The initializer has to be
  infallible, falling back to the safe middle value.
- The candidate set is formula-bounded. There are few candidates and a Derived quantity
  bounds them — a stream count cannot usefully exceed the register or associativity budget,
  so `_NVREG` and `_L1D_ASSOC` set the range. Both the bounds and the selection then adapt
  to unseen hardware, instead of hiding a per-µarch table.
- The `.so` build has to pin. `juliac --trim` cannot contain a runtime benchmark, so
  `juliac/build.jl` sets `ger_panel_np => 4` for the duration of the build and restores the
  previous state afterwards, which flips the `@static if` to the pinned branch. Every new
  Measure-tier knob needs the same treatment there; forgetting it breaks the trim build.

One more trap, learned the hard way (`_POTRF_BASE_F32`): `@load_preference` **must sit in
a top-level `const`**, never inside a function body — in a function it becomes a per-call
runtime Preferences lookup, which dominated microsecond-scale small-n factorizations.

### 5.1 Measure a *crossover*, not a flag: `_trsm_r_scratch_kmin`

`_ger_np` picks a count. The cautionary example is `_trsm_r_scratch_kmin`
(`src/blas3/level3.jl`, pref `"trsm_r_alias_scratch_kmin"`): when side-R `trsm`'s `ldb` is
an L1 way-stride multiple (`_alias_ld`), the solved-column re-reads all collide in one L1
set. Two cures — solve each row-chunk into the odd-`ld` `rpack` scratch and copy back, or
solve in place and eat the conflicts.

It was first written as a `Bool`, probed at `k = _TRSM_DBASE`. That was wrong in two ways,
and only the fleet run caught either of them:

1. The winner depends on `k`, and `k` was the dimension nobody was looking at. Fleet A/B on
   Zen3 at an aliasing `ldb` gives scratch/in-place = 1.18 at k=16 (in-place wins),
   1.00–1.02 at k=32, and 0.68–0.79 at k≥48 (scratch wins by 20–32%). That is physics
   rather than noise: the copy-back costs O(m·k) against an O(m·k²) solve, so its relative
   price falls as 1/k and scratch can only start paying above some k. A single Bool cannot
   express a crossover. It is worth knowing why this was missed — the original comment
   claimed a µarch inversion ("Zen3 scratch +41–49%, Zen4 a net loss"), so the knob got
   modelled as a per-box flag. Measured properly, both boxes cross over at the same k of
   64. The µarch framing was an artifact, and it hid the variable that actually mattered.
2. The probe sat exactly on the tie. `k = _TRSM_DBASE` = 32 is the crossover itself, so the
   measurement was a coin flip: two Zen3 processes in one run returned different answers,
   costing `'T'` 1024×16 = 0.763 (chose scratch) and `'N'` 1024×48 = 0.718 (chose
   in-place). Wrong in both directions, on the same box.

The fix is to measure the **crossover k** itself: walk a Derived candidate set (powers of
two from `_vwidth` up to `_TRSM_R_FUSE`) at an aliasing probe shape and return the first k
where scratch wins, or `typemax(Int)` for "never". Lessons that generalize:

- Probe where the decision is decisive, never where it is marginal. A probe shape that
  lands on the tie point returns noise and looks like a flaky benchmark.
- Before settling on a Bool, ask what the answer depends on. If it varies along a dimension
  the caller already knows — here `k` — measure the threshold along that dimension.
- The `catch` returns the conservative strategy, not the fast one. An infallible
  initializer is mandatory, since a throwing `OncePerProcess` poisons the process, and the
  safe fallback is whichever arm adds no buffer traffic.
- Don't reach for Measure when Derive fits. The sibling question in the same routine —
  should side-R `op='N'` pay an O(k²) reflect copy to reach the fused leaf? — is
  predictable. The copy's per-element cost turns on whether A's k×k footprint stays
  L1-resident, so the gate is the formula `k·k·sizeof(T) ≤ _L1_BYTES`, not a measured
  crossover. Measuring what a residency formula already predicts is its own kind of debt.

### 5.2 The overhead you set out to delete can be the fast path: `_pbtrf_ucross`

Banded Cholesky's `uplo='U'` had one kernel: conj-transpose the band into lower storage,
run the much faster lower kernel, transpose the factor back. Profiling at `kd=256` pointed
straight at the copy — 43% of the whole factorization, with the op sitting at 0.845 against
AOCL, a gate miss. Tuning the copy went nowhere; the best tiling recovered 1149 µs of
2749 µs for a still-missing 0.917, and it helped `kd=128/256` while hurting
`kd=160/192/384`. So the copy was replaced outright with a native port of the reference's
own `UPLO='U'` branch.

The native kernel fixed `kd ≥ 256` — and lost badly below it. Same-process ABBA, Zen4 F64,
ratio vs AOCL:

| kd | 48 | 64 | 96 | 128 | 160 | 192 | 256 | 384 |
|---|---|---|---|---|---|---|---|---|
| re-pack | 2.41 | 3.26 | 1.20 | 1.02 | 1.11 | 1.10 | **0.845** | 1.01 |
| native | 1.36 | 1.71 | 0.63 | 0.83 | 0.87 | 0.92 | **1.055** | 1.08 |

Both are needed; the switch point is `_pbtrf_ucross` (Measure, pref
`"pbtrf_u_native_kd"`, harness returns 256 for F64 on Zen4 — matching the table). Lessons
beyond 5.1's:

- A profiler calling something 43% of runtime does not make removing it a win. The copy
  really was 43% at that one bandwidth; at `kd=96` it was about 27%, and the kernel it
  bought access to was worth far more than it cost. "Dominant cost" and "wrong design" are
  different claims, and only the second justifies a rewrite.
- Measure the replacement against the incumbent across the whole parameter range, not at
  the size that motivated the work. This rewrite was validated at `kd=256`, where it wins.
  Shipped as a straight replacement it would have taken `kd=96` from 1.20 to 0.63.
- A superlinear overhead has a crossover in it. The re-pack walks the band down a diagonal,
  touching `kd` distinct columns per pass — one cache line, and eventually one page, per
  8-byte element. Its cost grows faster than `kd` while the factor itself grows as `kd²`,
  so it is cheap and then suddenly catastrophic rather than uniformly bad. The fix for that
  shape is a threshold, not a deletion.
- Keep the loser. The re-pack still ships for every `kd` below the crossover, where it
  gates at 1.02–3.26×. Deleting it to keep a single code path would have cost more than the
  rewrite gained.

### 5.3 A Measure knob must be *reproducible*, not just correct on average

`_pbtrf_cross` (blocked-vs-unblocked band Cholesky) was written exactly to the §5.1 recipe — measure
the crossover, walk a Derived candidate set, `typemax` for never — and still shipped a 2× regression,
because the harness answered **32 in some processes and 64 in others**. Two independent flaws, both
of which apply to any threshold-finding harness:

The comparison at the crossover is a coin flip by construction. At the crossover the two
kernels are equal — that is what makes it the crossover — so which one times faster is
decided by noise. §5.1's advice to probe where the decision is decisive does not help,
because this harness *searches* for the crossover and so has to evaluate the tie. What it
can control is what a wrong tie costs, and candidate spacing sets that. The old ladder
doubled (`{2,4,8,16}·W`), so one flipped comparison skipped a whole octave. Measured on
Zen4 (F64, `uplo='L'`, n=4096): at `kd=40` blocked runs 815 µs against unblocked's 1644 µs,
and at `kd=48` it is 1132 against 2138. A harness answering 64 instead of 32 left every
`kd` from 32 to 63 running about 2× slow — and both answers turned up across runs on one
box. Stepping by `W` bounds the damage of a flip to a single `W`-wide band.

When the two mistakes cost different amounts, the tie-break has to say so. Switching one
candidate too early costs 9% here (`kd=24`: blocked 227 µs against unblocked 208).
Switching one too late costs 103%, as `kd=40` above shows. At a 20:1 asymmetry a bare
`tb < tu` is the wrong test, since it treats both directions as equally bad. The harness
now accepts blocked when it is within 5% (`20·tb < 21·tu`). That band is the measured gap
at the true crossover, and at `kd=24` blocked is 9.3% behind — safely outside it, so the
bias does not drag the answer down an extra step.

The general rule: after writing a Measure harness, run it in several fresh processes and
check it returns the same value. A knob that varies run to run is worse than imprecise. It
makes every gate number irreproducible, and it will quietly pick the slow kernel on
someone's machine. Checked for these two — `_pbtrf_cross` and `_pbtrf_ucross` return 32 and
256 in 6 of 6 Float64 runs, while the complex knobs still move by exactly one ladder step
(20↔24, 192↔208), which is the bounded outcome the finer ladder exists to guarantee.

A related failure in the same routine: `_pbtrf_nb` was one measurement (`min(nb_tuned, kd)`)
serving two regimes. `nb_tuned` is probed at a mid band, where it is right, but for
`kd < nb_tuned` the clamp collapses `nb` onto `kd` itself. That kills the in-band panel
(`i2 = kd − ib = 0`) and lands `nb` on whatever `kd` happens to be — on Zen4, 32, which
this very file documents as a sharp local minimum for the band factor. It turned a 1.6×
win into the routine's only gate miss:

| kd=32, vs AOCL | nb=8 | nb=16 | nb=24 | nb=kd=32 |
|---|---|---|---|---|
| n=1024 | **1.63** | 1.36 | 1.05 | 0.99 |
| n=4096 | **1.64** | 1.33 | 1.04 | 0.96 |

The fix is a second Measure knob, `_pbtrf_nb_small`, with candidates `{1,2,3,4}·W` probed
at `kd = 4W`, used when the clamp binds. If one measured constant has to serve two regimes
with different optima — here 40 and 8, a 5× spread — it is two knobs, not one.

## 6. How to add or change a tuning constant (checklist)

1. **Confirm it really is a tuning constant.** A machine-dependent block size, cutoff,
   panel width, unroll, fuse factor, stream count or algorithm-switch threshold means the
   PDM ladder applies. A correctness bound such as `lassq` scaling does not — different
   rules, and you should not be "tuning" it.
2. **Ask the D-vs-M question.** Is the optimum physically predictable from a detected
   const? Cache residency, SIMD width, register count and datapath width all say Derive.
   Port balance, the prefetcher, stream behaviour, or your model mispredicting a box you
   already have, all say Measure. There is no third option, and if you can write neither a
   formula nor a measure harness, stop and flag it.
3. **Derive tier.** Write the default as a pure formula over `_HW` fields — an
   `_at_*(hw, T)` function in `cpuinfo.jl` if it is shared, otherwise inline over the
   detected consts. Take cache size and ISA together, since a block size depends both on
   what fits in cache and on the vector width. Round to W, clamp with a stated floor and
   cap, then add the fleet descriptors' expected values to `test/autotune_tests.jl`.
4. **Measure tier.** Mirror `_ger_np` exactly: a top-level `@load_preference` const,
   `@static if isnothing(pref)`, a `Base.OncePerProcess` with the `generating_output()`
   guard and a blanket `try`/`catch`, and formula-bounded candidates. Then add the pin to
   `juliac/build.jl`, so the trim build never contains the benchmark branch, and a
   calibration entry to `bench/calibrate.jl`.
5. **Keep the Preferences override** in both tiers, as `@load_preference("name", <derived
   default>)`. Pinning, cross-compilation and calibration all ride on it.
6. **Write the comment.** Every tuning const cites the detected consts it derives from, the
   physical criterion (residency share, latency budget, register budget, store-traffic
   crossover), and its PDM tier. For a deliberate literal, say why — invariant, falsified
   derivation, or pending fleet data — and add `# req8-ok: <reason>`, or
   `test/req8_lint_tests.jl` will fail on it. Watch for the derived-selection,
   hardcoded-materialization trap: `Val(8)` at a call site instead of `Val(_THE_CONST)` is
   a violation, and the lint catches it.
7. **Validate on the fleet before trusting it.** A derived formula has to reproduce the
   measured optima on the machines we have before it earns the right to extrapolate to ones
   we don't. Gate measurements run under `sudo bench/fleet_freqlock.sh lock` — passive
   governor, boost off, cores pinned to base clock, frequency verified under load — and a
   run whose verify step fails is invalid, full stop. Use controlled same-machine A/B
   comparisons, never cross-run one-offs.
8. **Expect falsification.** `_LU_NB` and `_chol_sth` both killed plausible formulas with
   measurements. A validated literal with a clear comment beats an elegant formula that
   adds spurious variation — but only once the derivation has been tried and measured,
   never instead of trying.
