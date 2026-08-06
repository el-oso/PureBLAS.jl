# Changelog

All notable changes to PureBLAS.jl are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Versioning

Pre-1.0, following Julia's convention that the **leading non-zero component** marks breaking changes:
for `0.x.y`, bumping `x` is breaking and bumping `y` is not.

**Performance work is a patch bump.** Optimising a kernel does not change the API, so it is not
breaking — even though reordered summation moves results bit-for-bit, which is ordinary for any BLAS
and is not an interface change. `0.2.0` is therefore reserved for a genuine break (for example
resolving the deferred complex-dot ABI, which would change exported signatures), not spent on tuning.

**`1.0.0` means the gate is met**: every routine ≥ `max(OpenBLAS, AOCL)` at every measured size on
every microarchitecture in the fleet. Not a date, not a feature count — the project's actual thesis,
demonstrated.

## [0.1.0] — 2026-08-06

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

### Fixed

- **`iamax`/`izamax` returned the WRONG INDEX when a NaN followed the maximum in the same SIMD lane.**
  A silent wrong answer from a shipped BLAS routine, on ordinary input, in three kernels from one
  root cause: every comparison against NaN is false, so a max-reduction written as *"take the new
  value if it compares greater, else keep the old"* silently keeps whatever occupies the fallback
  slot. `_iamax_tree!` (live for `_L1_BYTES < n·sizeof(T) ≤ _L2_BYTES`, ≈4097–131072 `Float64` on
  Zen4) let a NaN swallow a larger value at one fold node and then be discarded itself at the next,
  erasing the true maximum — `PureBLAS.iamax` returned its seed index 1 where netlib and OpenBLAS
  return 2. `_iamax_cmplx_simd!` (**live on every box**, so `izamax`/`icamax` were affected) and
  `_iamax_chain4!` (SSE/NEON only) seeded their running maximum from the first blocks' own values,
  so a NaN there poisoned that lane permanently. netlib's contract is that a NaN *mid-vector* is
  skipped and `dmax` can only be NaN if element 1 is; `_iamax_thresh!` was correct throughout
  precisely because it broadcasts its threshold from `|x[1]|`, and the fixes make the others match.
  The testitem named for this exact shape had never executed the kernel it was named after — its
  sizes are L1-resident, so dispatch routed every assertion to the one correct kernel. Kernels are
  now also tested **by name**, independently of dispatch.
- **`A \ b` with one right-hand side ran 14× slower than OpenBLAS, and nothing was measuring it.**
  `getrs`/`potrs`/`trtrs` existed only as C-ABI shims, so the benchmark harness — which compares
  `PureBLAS.foo!` against `LAPACK.foo!` — had nothing to call. Native entry points now exist (Mode-2,
  AD-traceable, with the shims calling them so there is one implementation), and gating them exposed
  three defects, all in `trsm`: the blocked path's `O(k·nb²)` setup is never repaid when B has few
  columns (at k=1024 it cost 1123 µs for one column and 1197 µs for eight); a power-of-two-`lda`
  A-pad copied all of A into scratch on every call, costing **+3% to +357%** across all sixteen
  side × uplo × transA × B-width combinations and paying nowhere, including at square B; and a
  **ragged SIMD lane was charged as a full lane**, so `nrhs`=1…7 each cost what eight columns cost
  together. Narrow B now sweeps `trsv` per column, the pad is deleted, and a ragged B is widened into
  GKH-owned scratch before the solve. `getrs` at `nrhs=1` went from 0.072–0.817 to **0.98–1.16×**
  OpenBLAS across n = 8…2048; `potrs` reaches **4.9×** at n=1024. `trsm` itself now gates at
  worst 1.18× (was 1.04). Residual at `nrhs = 8`, which falls between the two mechanisms.
- **`sytrs` had two codegen hazards** on the hot path of every LDLᵀ solve: an unforwarded
  store-to-load (`B[i,j] -= A[i,k]*B[k,j]` forces a reload of `B[k,j]` because alias analysis cannot
  see that `i > k`), and a per-element `herm ? conj(x) : x` runtime branch blocking vectorisation.
  Both fixed; +11–18% (n=1024, `nrhs=1`: 0.435 → 0.514 vs OpenBLAS). Still short — the rest is
  structural, `sytrf` being unblocked.
- **Five real factorizations had never been benchmarked once**: `sytrf`, `sytrs`, `gbtrf`, `geqp3`,
  `gels`. They routed and passed correctness, which reads as coverage while measuring nothing. Gate
  rows added. Only `gels` passes (1.87–2.82×); `geqp3` is the worst routine in the real surface at
  **0.181×** OpenBLAS, and both it and `sytrf` are confirmed-unblocked BLAS-2 implementations.
- **Packed Cholesky `pptrf` (lower) was paying per-call overhead, not doing slow arithmetic.**
  Its worst cell ran 0.738× AOCL at n=32. Decomposing it: PureBLAS's `spr!` *kernel* already ties or
  beats AOCL's `dspr` at every trailing order (70 vs 70 ns at m=8, 591 vs 641 at m=64), but the
  factorization reached it through the public entry (~11 ns of validation) and built two `SubArray`s
  per column (a further 9–50 ns) — at n=32 that per-call tax was the entire gap. Compounding it, the
  lower path gated on the *upper* path's cutoff constant, whose measurement table is upper-path only;
  at 32 that meant `m = n−j` was below the cutoff for **every** column at n=32, so `spr!` was never
  called and the whole factorization ran the scalar fallback. Split into its own Measure-tier knob
  (`_pptrf_spr_min`) and given a pointer-direct kernel entry (`_spr_simd_lower_ptr!`). Both were
  needed — lowering the cutoff alone only reached 0.846. `uplo='L'` vs AOCL: n=32 0.738 → **0.949**,
  n=48 0.880 → 0.980, n=64 0.922 → 1.009, n=96 0.980 → 1.050, n=128 1.002 → 1.069; vs OpenBLAS it now
  clears every size (1.05–1.37). Still open: `uplo='L'` n=32/48 vs AOCL, and `uplo='U'` n=48…2048 vs
  OpenBLAS (0.918–0.990) — the latter a systematic miss that measuring against AOCL alone concealed,
  since PureBLAS beats AOCL's packed upper by up to 6.3×.
- **Banded Cholesky `pbtrf` had no correctness coverage, and wide upper bands missed the gate.**
  The only test was a trim-compatibility check at `uplo='L', kd=2`, so the entire blocked kernel —
  both triangles, the panel and corner blocks, the work-array copy-back — was untested; a
  deliberately rewritten `uplo='U'` kernel passed the full suite and then failed on its first real
  matrix. There is now a reconstruction-oracle testitem over `s`/`d`/`c`/`z` × both triangles ×
  `kd ∈ {1,2,8,31,32,33,40,64}` × `n ∈ {65,300}`, which also exercises both upper kernels directly
  (which one `pbtrf!` selects is host-dependent). Separately, `uplo='U'` at `kd ≥ 256` ran at
  0.845× AOCL: it reached the upper triangle through a conj-transpose re-pack whose diagonal walk
  costs one cache line per element, and that copy grows superlinearly in `kd`. A native port of the
  reference's `UPLO='U'` branch now takes over above a measured bandwidth crossover
  (`_pbtrf_ucross`), while the re-pack — which is *faster* for narrow bands — keeps everything
  below it. Two further defects surfaced once the whole bandwidth range was swept: the
  blocked-vs-unblocked crossover harness returned **different answers in different processes**
  (32 or 64), and because its candidate ladder doubled, one flipped comparison left every
  `kd ∈ 32…63` on a kernel that is ~2× slower there; and the panel width `min(nb_tuned, kd)`
  collapsed onto `kd` whenever the clamp bound, killing the in-band panel and landing `nb` on a
  value this code already documents as a local minimum (0.99× at `kd=32`, where a narrower panel
  gives 1.63×). The crossover ladder now steps by `W` with a measured-asymmetry tie-break, and the
  narrow-band regime has its own measured width. `pbtrf` now gates at **1.02–3.28× AOCL** across
  `kd = 32…384` in both triangles.
- **`juliac/build.jl` restored a preference that was never captured.** The `finally` block tested
  an undefined `_prev_trs`, so every trim build would have thrown `UndefVarError` while unwinding —
  masking whatever the real build outcome was.
- **`potrf`/`cholesky` now report the true failing column.** `PosDefException` carried column `1`
  regardless of where the factorization actually failed, so `cholesky(A; check=true)` named the wrong
  column and the `dpotrf_64_`/`zpotrf_64_` C-ABI symbols returned the wrong `info` to LBT callers. The
  blocked and hybrid drivers factor sub-blocks of *views*, so the base kernel's column index is now
  lifted back through every recursion level, matching LAPACK's `info` exactly for `s`/`d`/`c`/`z` and
  both triangles (validated against the LAPACK oracle across poisoned columns at `n = 8…257`).
- **`gtsv`/`gttrf` pivot selection for complex input.** These compared pivots with `abs` where LAPACK
  uses `CABS1` (`|Re| + |Im|`), which selects a *different* pivot on near-ties. The factorization stayed
  valid, but the overwritten factor arrays and `ipiv` disagreed with `cgttrf`/`zgttrf`. Now matches
  LAPACK, and is also cheaper (no `hypot` on the complex path). This was invisible until row-interchange
  test coverage was added — the previous tests were diagonally dominant and never took a pivot.

### Falsified

Measured dead ends, recorded so they are not re-attempted. In a project whose rule is "measure, don't
guess", a disproven hypothesis is worth as much as a win.

- **`pbtrf` uplo='U' band re-pack tiling — REVERTED.** The diagnosis holds: the re-pack is 31.6% of
  the call at kd=128 and shows a **power-of-two stride aliasing spike**, 3× its immediate neighbours
  (pack/unpack µs at n=4096: kd=127 215/324, **128 616/640**, 129 204/340), because both passes walk
  diagonals stepping `ld-1` elements = exactly 1024 B. Tiling the walk fixed it on Zen3 (−14.6%) and
  **made it worse on Zen4** (+8.8% at kd=128, +18.6% at kd=192, taking a gating cell to 0.995). Both
  boxes have identical L1 geometry (32 KB, 8-way, 64 sets), so a set-capacity model computes the same
  value on each and cannot explain opposite signs — the mechanism is not set capacity. Any revival
  must be a Measure-tier auto-tune, not another predicate.
- **Narrow-B (`nrhs=1`) `trsm` is not a routing problem.** Routing single-column solves to `trsv`
  above `_TRSM_DBASE` was the obvious fix; measured, PureBLAS's `trsm!` already matches or beats its
  own `trsv!` on the configurations `potrs` issues. At k=2048 the solve moves 16.8 MB of triangle in
  449 µs ≈ 37 GB/s against AOCL's ≈40 — a streaming-efficiency question, not blocking or dispatch.
- **`getrs@8` is not wrapper overhead.** The entire wrapper (`getrs!` + `laswp` + two public `trsm!`
  entries) costs ~30 ns of 191 ns; AOCL completes the whole solve in ~154 ns, less than our two
  kernel calls alone. The cost is the tiny-`n` triangular kernel. Related coverage hole: `trsv` is
  benchmarked only from n=128 up, so the n ≤ 32 solve path has no gate cell of its own.

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
  gap.
- **Complex `zpotrf` with `uplo='U'`** sits at ~0.95 vs AOCL for `n ≥ 1024` on Zen4 (it gates on Zen3).
  The upper path transposes into a padded scratch and back; those two copies are the entire gap.
- **`pttrs` measures 0.99 vs AOCL** (1.67–1.73 vs OpenBLAS). This one is *not* an implementation gap:
  both libraries sit on the same ~12.2 cyc/elem dependency-chain bound for the pair of triangular
  sweeps, with 0.0% run-to-run drift.
