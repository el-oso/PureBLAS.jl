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
2. **LBT drop-in (Mode 1)** — `@ccallable` Fortran-ABI symbols → `juliac --trim` → `libpureblas.so`.
   The .so works from **non-Julia hosts** (C/C++/Rust; it self-inits its embedded runtime — see
   `juliac/ctest.c`) and proves trim-compatibility. **KNOWN LIMITATION (don't re-chase):**
   `BLAS.lbt_forward` of this .so from inside a *live Julia* process aborts (signal 6) — juliac libs
   embed the Julia runtime and double-init the shared libjulia on the autodetect probe. So the
   in-Julia path is Mode 2; revisit LBT forwarding only when juliac gains host-runtime init. See
   `ROADMAP.md` "Key finding".

Both modes share ONE set of low-level kernels. Source map:
`core.jl` (accessors `_ld`/`_st!` over Ptr AND AbstractVector, lassq, |·|) · `cpuinfo.jl`
(SIMD width, const-folded, trim-safe) · `simd_kernels.jl` (SIMD.jl fast paths) · `level1.jl`
(low-level `(n,…,inc)` kernels) · `level2.jl` (gemv/ger/symv/hemv/trmv/trsv) · `level2_packed.jl`
(spmv/hpmv/tpmv/tpsv) · `level2_banded.jl` (gbmv/sbmv/hbmv/tbmv/tbsv) · `gemm.jl` (L3) ·
`contracts.jl` (TypeContracts `AbstractBLAS1`/`AbstractBLAS2`) · `backend.jl`
(`SIMDBackend`, Mode 2) · `native.jl` (bare API) · `cabi.jl` (`@ccallable` ABI) · `lbt.jl`
(`activate`/`deactivate`).

## Hard requirements (MUST follow)

1. **Performance gate: ≥ 0.96× OpenBLAS, non-negotiable.** Per-machine (the gate is measured on
   the dev box). Beat it where possible. BLAS-1 is bandwidth-bound (easy parity); the real fight is
   M2 `dgemm`.
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

## ABI conventions (Mode 1)

- Symbols are the **ILP64** reference-BLAS names Julia resolves: trailing `64_` (e.g. `daxpy_64_`).
  Args **by reference** (`Ptr`), **column-major**, `Int64` integers. BLAS-1 has **no character
  args** → no hidden Fortran string-length args (a reason it's the M1 slice).
- **Deferred:** the 4 complex-dot symbols (`c/zdotu`, `c/zdotc`) — their complex-return ABI (LBT
  NORMAL vs ARGUMENT retstyle) is unresolved; lands in M2 with GEMM's char/string ABI. Native API
  covers complex dot meanwhile.

## Testing (ReTestItems — self-contained, individually triggerable)

- `runtests(PureBLAS)`; trigger one item via `runtests(PureBLAS; name="...")`. Use `@testsetup`
  modules for shared oracle helpers so items are independent.
- Correctness oracle = OpenBLAS via `LinearAlgebra.BLAS.*` over s/d/c/z, many `n`, strides, edges.
  Note: single-vector ops (nrm2/asum/iamax/scal) are spec'd `incx ≥ 1` (reference returns 0 for
  `incx<1`); only two-vector ops (axpy/dot) take negative/mismatched increments.
- StrictMode dogfood (`@assert_typestable/@assert_noalloc/@assert_trim_safe`) on hot paths, gated by
  `StrictMode.checks_enabled()`. AD smoke test via ForwardDiff (proves Mode 2).

## Benchmarking (reuse PureFFT methodology)

`BLAS.set_num_threads(1)` for fair single-thread comparison · `@noinline` concrete wrappers (not
closures) · repeated in-place reps · **median** times (not min) · `taskset -c N` + cpufreq pin for
low noise · results→JSON, plot from JSON · **per-host JSON filenames** (fleet: Zen4 dev / Zen3 AVX2 /
Zen5 native-AVX512 / future M5 ARM — the 0.96× gate is evaluated per machine).

## Standing rules

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
