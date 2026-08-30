# SIMD & Hardware Adaptation

PureBLAS is written in Julia, but its kernels are not written against any one CPU. This page covers how
a single generic kernel set adapts to the machine it runs on: the vector abstraction, the hardware
detection that happens at compile time, and the rule that every tuning parameter is a formula over
detected hardware rather than a hardcoded per-microarchitecture literal. It also sets that against the
ahead-of-time model used by C and Rust libraries like OpenBLAS and faer.

## Why not just let the compiler vectorise it?

This is the first question most people ask, and it deserves a straight answer, because the premise is
correct: Julia's compiler *is* good at vectorising loops. Write the textbook three-line matrix multiply,
put `@inbounds` and `@simd` on it, and LLVM really does turn the inner loop into AVX-512 code. You can
check — the generated IR for that loop contains `fmul <8 x double>`, `fadd <8 x double>` and
`load <8 x double>`, which is eight lanes at a time, exactly what you would hope for.

Then you measure it against PureBLAS on the same machine, one thread, Float64:

| n | naive loop | PureBLAS | |
|---|---|---|---|
| 256 | 0.70 GFlop/s | 43.6 GFlop/s | 62× |
| 512 | 0.21 GFlop/s | 42.5 GFlop/s | 199× |
| 1000 | 2.41 GFlop/s | 42.2 GFlop/s | 17× |

Both are running vectorised code. The difference is not the instructions — it is where the numbers are
coming from.

Think about what the naive loop does to memory. To compute one entry of the result it walks a whole row
of `A` and a whole column of `B`. Walking down a column means each step jumps a full row-length ahead in
memory, so every value it wants sits in a different cache line. The CPU fetches 64 bytes and uses 8 of
them. Do that a few billion times and the arithmetic units spend nearly all their time idle, waiting.
Vectorising that loop makes the CPU wait eight lanes at a time. It does not make it wait less.

What a real BLAS does is rearrange the *work* so that data, once fetched, gets used many times before it
is evicted. It chops the matrices into blocks sized to fit in L1 and L2, copies each block into a small
contiguous scratch buffer so the inner loop reads straight through memory, and holds a tile of the
result in registers across the whole inner loop. None of that changes the arithmetic — the same
multiplications happen in a different order — but it changes how often each number has to be fetched,
and that is what the 17–200× is.

A compiler will not do this for you, and not because it is not clever enough. Choosing block sizes needs
the cache sizes, which are a property of the machine, not of the code. Packing operands into scratch
buffers means allocating memory and copying data that the program you wrote never asked for. Reordering
the loop nest that far changes the order of floating-point additions, which a compiler is not allowed to
do on its own. These are algorithm-level decisions, and they belong to the library.

That is also why the naive numbers above bounce around — 0.70, then 0.21, then 2.41 — while PureBLAS
sits flat near 42 across all three sizes. The naive version is at the mercy of how each size happens to
land in the cache; `n=512` is the worst because a power-of-two row length makes columns collide in the
same cache sets. A blocked kernel is not at the mercy of anything: it decides the access pattern itself.

The rest of this page is about how those decisions get made without hardcoding them for one CPU.

## The vector abstraction: one kernel, every width

Hot kernels are written with [SIMD.jl](https://github.com/eschnett/SIMD.jl) — `Vec{N,T}` plus
`vload` / `vstore` / `muladd` / `vifelse` / `shufflevector` — not `_mm256_*`-style intrinsics. A
kernel is written once, parameterized on the vector width `N = _vwidth(T)`, and the same source
compiles to AVX-512 or AVX2 code depending on the host. Width detection also has a 16-byte fallback
that suits SSE2 and NEON, but the benchmark fleet is x86-64 and ARM is untested. For example a Level-1
axpy inner step is just:

```julia
V = Vec{_vwidth(T), T}
vstore(muladd(va, vload(V, px + o), vload(V, py + o)), py + o)
```

There are no per-ISA kernel copies to maintain. Unit-stride dense inputs take this SIMD path, real and
complex alike — the complex kernels have their own vectorized form, described below. Strided data,
`ForwardDiff.Dual` and other element types fall to the generic scalar loop, which is slower but is what
keeps the native API differentiable (see [Design](design.md)).

## Compile-time hardware detection

The register width and cache geometry are detected once, at build or load time, and baked into
const-folded constants in `cpuinfo.jl`. Detection uses `CpuId` (`simdbytes`, `cpuvendor`,
`cpufeature`, `cachesize`, `cachelinesize`), `CPUSummary` (`cache_size`), and `HostCPUFeatures`:

| Constant | Meaning |
|---|---|
| `_SIMD_BYTES`, `_vwidth(T)` | vector width in bytes / lanes per `T` |
| `_L1_BYTES`, `_L2_BYTES`, `_L3_BYTES` | cache sizes (L3 is the total, not per-core share) |
| `_CACHELINE`, `_L1D_ASSOC`, `_L1_WAY_BYTES` | line size, L1 associativity, way-stride |
| `_NVREG` | vector register count (32 on AVX-512, 16 on AVX2) |
| `_CPU_VENDOR`, `_CPU_FAMILY`, `_INTEL_AVX2` | vendor / family / feature bits |
| `_HW` | a named tuple bundling the above for the `_at_*` derive helpers |

Because Julia compiles to the host, these are ordinary `const`s that fold away entirely: the generated
kernel contains no runtime `CpuId` or `cpuid` call, which is what lets it survive `juliac --trim`. Each
constant also accepts a `Preferences` override (`@load_preference`) for cross-compilation, pinning, or
correcting a bad heuristic — `simd_bytes`, `l3_bytes`, `l1d_assoc` and so on.

## Deriving tuning from hardware, not tabulating it

This is the core rule, and the main structural advantage over an ahead-of-time BLAS. Block sizes,
base-case cutoffs, panel widths and unroll factors are all formulas over the detected constants, each
keyed on a physical criterion — cache *residency* for block sizes, datapath *latency* for unroll, ISA
for width granularity. The `_at_*` helpers in `cpuinfo.jl` are those formulas:

```julia
_at_gemm_nr(hw, T)  = max(_lanes(hw, T), _ILP_TARGET ÷ _lanes(hw, T))     # NR from the ILP target
_at_gemm_kc(hw, T)  = _l1_block(hw, T, _at_gemm_nr(hw, T))                # B micropanel ≤ ½·L1
_acc_cap(hw, T)     = (hw.nvreg - 4) * _lanes(hw, T)                      # accumulators the reg file holds
```

A bare literal like `_vwidth == 4 ? 48 : 64`, or `const _KC = 256`, is a bug: the tell is a
number you cannot trace to a detected constant through a residency/latency formula. Why this is
mandatory rather than stylistic: Julia compiles to the host at load time, so PureBLAS can *compute*
the right sizes for the actual machine — including CPUs never benchmarked (a new laptop, a cloud
box). A static C/Rust BLAS cannot; it ships hand-tuned per-µarch tables baked at *its* compile time.
Hardcoding literals here throws away Julia's one real structural advantage.

A derived formula has to reproduce the measured-optimal values on the known fleet (Zen3, Zen4, Zen5)
before it is trusted to extrapolate: derive, validate on the fleet, then ship. The triangular-solve
fused-leaf cutoff is a good example — the size at which the hot `KC × NR` panel fills L1:

```julia
_GT_TRANSPOSE ? max(_GT_MR, _L1_BYTES ÷ (_GT_NR * sizeof(Float64))) : 128
```

On AVX-512 that computes a different number from each machine's own L1 rather than reading a table.
Off AVX-512 it is a literal 128, because a larger base was measured to regress `n=256` — the rule
allows a literal when a derivation has been tried and falsified, provided the measurement is recorded
next to it. Most exceptions in the codebase look like this one: not an untested guess, but a formula
that lost to the numbers.

## Microarchitecture, not just width

Width and cache size do not distinguish microarchitectures with the same ISA (Haswell and Zen3
are both AVX2 / `_vwidth == 4`). For a µarch-dependent choice, key on the vendor and feature bits,
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
both but leans toward packing and padding where masking would cost throughput — for instance packing a
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
difference is when specialization happens:

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
  spurious overflow or underflow, so a faithful drop-in must use scaled accumulation (LAPACK
  `lassq`, which carries a running rescale). It is not optional, and it does cost some speed in that
  one kernel versus a naive square-and-sum.
- **Bit-reproducibility is not in the spec.** faer *elects* to guarantee it — identical results
  across alignment offsets on a machine — by rotating reduction accumulators (needed because faer
  aligns its loads, which would otherwise make the summation order alignment-dependent).

PureBLAS is not *required* to be reproducible, but it is — **by construction, and now locked by
regression tests** (`test/reproducibility_tests.jl`). Its kernels load from the base pointer with a
fixed lane grouping (no alignment peeling) and run single-threaded, so two properties hold and are
asserted: every operation is bit-identical run to run (same input, repeated call), and a reduction
is bit-identical across memory alignments within a code path.

What is *not* guaranteed — for PureBLAS, and typically not for any BLAS — is that the SIMD fast path
and the generic scalar fallback agree to the last bit: they sum in a different order, so `nrm2` of a
plain `Vector` (SIMD path) can differ by a ULP or two from `nrm2` of an offset view (scalar path).
That is the normal fast-path/fallback split, not a reproducibility bug. Cross-*machine* reproducibility
is likewise not guaranteed (a different vector width builds a different reduction tree), and adding
multithreading would require a fixed reduction tree to preserve run-to-run identity.
