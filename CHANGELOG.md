# Changelog

All notable changes to PureBLAS.jl are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 0.1.0

First release: a pure-Julia BLAS/LAPACK implementation — a drop-in, AD-traceable replacement for the
OpenBLAS/MKL that Julia ships by default. Part of the **Pure Julia Ecosystem** (sibling of PureFFT.jl).

### Added

- **BLAS Levels 1–3, real and complex** (`s`/`d`/`c`/`z`), pure Julia. Real unit-stride dense kernels use
  `SIMD.jl` fast paths; complex, strided, and any other `T<:Number` (incl. `ForwardDiff.Dual`) run one
  generic scalar kernel. `nrm2` uses LAPACK scaled accumulation (`lassq`) for overflow/underflow safety.
- **Core LAPACK factorizations**, real and complex: Cholesky (`potrf`), LU (`getrf`), QR (`geqrf`), and
  SVD (`gesvd`, values + vectors).
- **Two first-class integration modes, sharing one kernel set:**
  - *Native API (Mode 2)* — `PureBLAS.axpy!`, `.dot`, `.gemm!`, … Direct Julia calls, no `ccall`
    boundary, so they are **AD-traceable** (ForwardDiff/Enzyme/ChainRules) — something opaque OpenBLAS
    `ccall`s never permitted.
  - *Whole-ecosystem reroute (Mode 1)* — `PureBLAS.activate()` reroutes all of LinearAlgebra's
    BLAS/LAPACK to PureBLAS in the running process (MKL.jl-style), by registering in-process `@cfunction`
    pointers to the native kernels via libblastrampoline (`lbt_set_forward`). After it, `A*B`, `mul!`,
    `cholesky`, `qr`, `svd`, and `LinearAlgebra.BLAS.*` use PureBLAS with no code change; `deactivate()`
    restores OpenBLAS. The same `@ccallable` reference-BLAS (ILP64) symbols also build to `libpureblas.so`
    via `juliac --trim` (self-inits its embedded runtime) for **non-Julia hosts** (C/C++/Rust).
- **Hardware-adaptive tuning.** Block sizes, base-case cutoffs, panel widths, and unroll/stream counts are
  **derived at load time** from the detected CPU (cache sizes, vector width, register count, µarch/vendor
  via `CpuId`/`HostCPUFeatures`/`CPUSummary`), const-folded and trim-safe — so PureBLAS sizes itself to the
  actual machine, including CPUs never benchmarked. A CI lint (`test/req8_lint.jl`) blocks new hardcoded
  tuning literals.
- **Performance:** meets or beats `max(OpenBLAS, AOCL-BLIS)` on the bulk of operations across the AMD fleet
  (Zen3/AVX2, Zen4/AVX-512, Zen5/native-AVX-512), single-threaded, boost-locked. `gemm` beats both baselines
  via Strassen–Winograd (real) and Karatsuba 3M (complex); `trmm` uses a Strassen split at large n.
- **Quality gates:** Aqua.jl (ambiguities, stale deps, compat, piracy), StrictMode dogfood
  (`@assert_typestable`/`@assert_noalloc`/`@assert_trim_safe`/`@assert_no_spill`/`@assert_memsafe`),
  TypeContracts interface contracts, juliac `--trim` build verification, and an OpenBLAS oracle test suite
  over s/d/c/z with many sizes, strides, and edge cases.

### Known limitations

- **Forwarding the `.so` into a live Julia process is blocked** (a juliac limitation, not a PureBLAS bug):
  `BLAS.lbt_forward(libpureblas.so)` from inside a *running* Julia process aborts (`signal 6`) — a juliac
  library embeds its own `libjulia` and double-inits it on LBT's autodetect probe. This is *not* a limit on
  using PureBLAS as the in-Julia backend: `PureBLAS.activate()` reroutes the whole ecosystem via in-process
  `@cfunction` forwarding (no `.so`). The `.so` is for **non-Julia hosts**. See `ROADMAP.md` → "In-process
  LBT forwarding".
- **Complex-dot ABI symbols deferred** — the four `c/zdotu`, `c/zdotc` `@ccallable` symbols have an
  unresolved complex-return ABI (LBT NORMAL vs ARGUMENT retstyle); the native API covers complex dot.
- **Single-threaded** — multithreading is deferred by design; all kernels are single-thread today.
- **Large-n `trmm`/`syrk` vs AOCL** sit at ~0.95–0.98 (n≥2048) — an LLVM-vs-hand-asm classical-microkernel
  gap; everything else meets the `max(OB, AOCL)` gate.
