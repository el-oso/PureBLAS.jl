# SIMD & Hardware Adaptation

PureBLAS is a pure-Julia BLAS, but its kernels are not written against any one CPU. This page
describes how one generic kernel set adapts to the machine it runs on — the vector abstraction, the
compile-time hardware detection, and the rule that **every tuning parameter is a formula over
detected hardware, never a hardcoded per-microarchitecture literal**. It also contrasts this with the
ahead-of-time (AOT) model used by C/Rust BLAS such as OpenBLAS and faer.

## The vector abstraction: one kernel, every width

Hot kernels are written with **[SIMD.jl](https://github.com/eschnett/SIMD.jl)** — `Vec{N,T}` plus
`vload` / `vstore` / `muladd` / `vifelse` / `shufflevector` — not `_mm256_*`-style intrinsics. A
kernel is written once, parameterized on the vector width `N = _vwidth(T)`, and the same source
compiles to AVX-512, AVX2, or NEON code depending on the build host. For example a Level-1 axpy inner
step is just:

```julia
V = Vec{_vwidth(T), T}
vstore(muladd(va, vload(V, px + o), vload(V, py + o)), py + o)
```

There are no per-ISA kernel copies to maintain. Real, unit-stride, dense inputs take this SIMD path;
everything else (complex, strided, `ForwardDiff.Dual`, any other `T<:Number`) falls to the generic
scalar loop — which is what keeps Mode 2 differentiable (see [Design](design.md)).

## Compile-time hardware detection

The register width and cache geometry are detected **once, at build/load time**, and baked into
const-folded constants in `cpuinfo.jl`. Detection uses `CpuId` (`simdbytes`, `cpuvendor`,
`cpufeature`, `cachesize`, `cachelinesize`), `CPUSummary` (`cache_size`), and `HostCPUFeatures`:

| Constant | Meaning |
|---|---|
| `_SIMD_BYTES`, `_vwidth(T)` | vector width in bytes / lanes per `T` |
| `_L1_BYTES`, `_L2_BYTES`, `_L3_BYTES` | cache sizes (L3 is the **total**, not per-core share) |
| `_CACHELINE`, `_L1D_ASSOC`, `_L1_WAY_BYTES` | line size, L1 associativity, way-stride |
| `_NVREG` | vector register count (32 on AVX-512, 16 on AVX2) |
| `_CPU_VENDOR`, `_CPU_FAMILY`, `_INTEL_AVX2` | vendor / family / feature bits |
| `_HW` | a named tuple bundling the above for the `_at_*` derive helpers |

Because Julia JIT-compiles to the host, these are ordinary `const`s that **const-fold away**: the
generated kernel contains no runtime `CpuId`/`cpuid` `ccall`, which is what makes it
`juliac --trim`-compatible (Mode 1). Every one of these constants also accepts a **`Preferences`
override** (`@load_preference`) for cross-compilation, pinning, or correcting a heuristic — e.g.
`simd_bytes`, `l3_bytes`, `l1d_assoc`.

## Deriving tuning from hardware, not tabulating it

This is the core rule (and PureBLAS's main structural advantage over AOT BLAS): **block sizes,
base-case cutoffs, panel widths, and unroll factors are formulas over the detected constants, keyed
on a physical criterion** — cache *residency* for block sizes, datapath *latency* for unroll, ISA for
width granularity. The `_at_*` helpers in `cpuinfo.jl` are these formulas:

```julia
_at_gemm_nr(hw, T)  = max(_lanes(hw, T), _ILP_TARGET ÷ _lanes(hw, T))     # NR from the ILP target
_at_gemm_kc(hw, T)  = _l1_block(hw, T, _at_gemm_nr(hw, T))                # B micropanel ≤ ½·L1
_acc_cap(hw, T)     = (hw.nvreg - 4) * _lanes(hw, T)                      # accumulators the reg file holds
```

A bare literal like `_vwidth == 4 ? 48 : 64`, or `const _KC = 256`, is a **bug**: the tell is a
number you cannot trace to a detected constant through a residency/latency formula. Why this is
mandatory rather than stylistic: Julia compiles to the host at load time, so PureBLAS can *compute*
the right sizes for the **actual** machine — including CPUs never benchmarked (a new laptop, a cloud
box). A static C/Rust BLAS cannot; it ships hand-tuned per-µarch tables baked at *its* compile time.
Hardcoding literals here throws away Julia's one real structural advantage.

A derived formula must **reproduce the measured-optimal values on the known fleet (Zen3 / Zen4 /
Zen5) before it is trusted to extrapolate** — derive → validate on the fleet → ship. A recent
example: the triangular-solve fused-leaf cutoff is the size at which the hot `KC × NR` panel fills L1,

```julia
_TRSM_FUSED_BASE = _L1_BYTES ÷ (_GT_NR * sizeof(Float64))   # 170 on Zen4 (32 K L1), 256 on Zen5 (48 K)
```

— two different values, both computed from each machine's own L1, not a lookup table.

## Microarchitecture, not just width

Width and cache size do **not** distinguish microarchitectures with the same ISA (Haswell and Zen3
are both AVX2 / `_vwidth == 4`). For a µarch-dependent choice, key on the **vendor and feature bits**,
not the width:

- `_INTEL_AVX2` (`cpuvendor() === :Intel && cpufeature(:AVX2) && !cpufeature(:AVX512F)`) selects a
  latency split in the Cholesky base.
- `_double_pumped(hw)` (`hw.simd == 64 && hw.vendor === :AMD && hw.family == 0x19`) recognizes that
  early Zen4 executes 512-bit ops as two 256-bit halves, so `_datapath_bytes` — not `_SIMD_BYTES` —
  drives latency-bound unroll counts.

These are still build-time detections (`req#7`): they const-fold, so no runtime branch survives.

## Tail handling: pack over mask

Partial vectors at the edge of a tile can be handled by masked load/store (`vifelse`, masked stores)
*or* by packing the operand into a padded buffer so the dense kernel runs unmasked. PureBLAS uses
both but leans toward **packing/padding** where masking would cost throughput — for instance packing a
triangular diagonal block dense-with-zeros so the ordinary microkernel multiplies it (the OpenBLAS
scheme), or padding a ragged column stripe to the panel width. Masked *vector stores* on
diagonal-straddling tiles were measured slower than dense-compute-to-scratch plus a scalar
triangular copy-back, and direct (unpacked) reads must clamp to valid bounds to stay inside the
allocation.

## Complex numbers

Complex kernels stay in the **interleaved** `[re, im, re, im, …]` domain and use **swap-adjacent**
shuffles (`shufflevector`) for the `i·i = -1` cross terms, rather than deinterleaving real and
imaginary parts across lanes. This keeps the data layout contiguous and avoids cross-lane
permutations, and it is expressed entirely in portable SIMD.jl — no x86 shuffle intrinsics. The same
`Vec{N,Complex{T}}`-free formulation covers `c` and `z`.

## Contrast with ahead-of-time BLAS (OpenBLAS, faer)

The *kernel* philosophy is shared with modern portable-SIMD libraries — faer, via
[pulp](https://github.com/sarah-quinones/pulp), writes one generic kernel over a `Simd` trait with
register-blocked multiple accumulators, much as PureBLAS writes one kernel over `Vec{N,T}`. The deep
difference is **when specialization happens**:

- **AOT (Rust/C):** the library is compiled before it knows the target. To cover a range of CPUs it
  must carry several ISA variants and select one at runtime, and its block-size tuning is a static
  table baked at the library's compile time. It cannot re-derive tuning for a CPU it was never built
  or benchmarked for.
- **JIT-to-host (PureBLAS):** Julia specializes the kernel to the *one* machine at load time. There
  is no runtime ISA dispatch and no multi-versioning — and, crucially, tuning is *computed* from the
  detected cache/ISA of the actual host, so an unbenchmarked CPU still gets sized blocks.

The practical consequence is that any faer-versus-PureBLAS performance gap on a given machine is a
matter of library maturity and tuning, not of language: Julia's LLVM backend generates numerical code
competitive with Rust's, and JIT-to-host is the more flexible substrate for hardware adaptation, not
a handicap.

## Reproducibility and BLAS compatibility

Two numerical properties are worth separating, because the reference BLAS spec treats them very
differently:

- **Overflow/underflow safety is mandated.** The spec *defines* `nrm2` to return `√(Σxᵢ²)` without
  spurious overflow or underflow, so a faithful drop-in **must** use scaled accumulation (LAPACK
  `lassq`, which carries a running rescale). It is not optional, and it does cost some speed in that
  one kernel versus a naive square-and-sum.
- **Bit-reproducibility is not in the spec.** faer *elects* to guarantee it — identical results
  across alignment offsets on a machine — by rotating reduction accumulators (needed because faer
  aligns its loads, which would otherwise make the summation order alignment-dependent).

PureBLAS is not *required* to be reproducible, but it is — **by construction, and now locked by
regression tests** (`test/reproducibility_tests.jl`). Its kernels load from the base pointer with a
fixed lane grouping (no alignment peeling) and run single-threaded, so two properties hold and are
asserted: every operation is bit-identical **run-to-run** (same input, repeated call), and a reduction
is bit-identical across memory alignments **within a code path**.

What is *not* guaranteed — for PureBLAS, and typically not for any BLAS — is that the SIMD fast path
and the generic scalar fallback agree to the last bit: they sum in a different order, so `nrm2` of a
plain `Vector` (SIMD path) can differ by a ULP or two from `nrm2` of an offset view (scalar path).
That is the normal fast-path/fallback split, not a reproducibility bug. Cross-*machine* reproducibility
is likewise not guaranteed (a different vector width builds a different reduction tree), and adding
multithreading would require a fixed reduction tree to preserve run-to-run identity.
