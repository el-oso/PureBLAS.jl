# PureBLAS.jl

A pure-Julia BLAS and LAPACK, part of the **Pure Julia Ecosystem** — pure-Julia replacements for
Julia's non-Julia default libraries (sibling: [PureFFT.jl](https://github.com/el-oso/PureFFT.jl)).

Everything is plain Julia source built on [SIMD.jl](https://github.com/eschnett/SIMD.jl): no
`libopenblas`, no Fortran, no vendor blob.

## The performance rule

**PB ≥ max(OpenBLAS, AOCL), at every size, on every microarchitecture.** Not a geometric mean, not
"competitive with" — the gate is the *worst* cell of a routine against the *faster* of the two
references, and a routine passes only if that worst cell clears parity.

Measurements are single-threaded and frequency-locked (`amd_pstate=passive`, boost off, cores pinned to
base clock) on a fleet spanning Zen3/AVX2, Zen4/AVX-512 and Zen5/AVX-512. Both references are measured
in the same round as PureBLAS, so the `max` is taken over numbers that saw one machine state. The
per-routine results — **including every routine that currently misses** — are on the
[Coverage](coverage.md) page, generated from the benchmark caches rather than transcribed.

## Two ways to use it, both first-class

**1. Native Julia API.** Because the whole call tree is Julia source, it is **AD-traceable**
(ForwardDiff today; Enzyme/ChainRules planned) — something opaque `ccall`s into OpenBLAS never allowed.
The generic `T<:Number` scalar path is what makes that work and is deliberately not specialised away.

```julia
using PureBLAS
x = randn(1000); y = randn(1000)
PureBLAS.axpy!(y, 2.0, x)        # y .+= 2x
PureBLAS.dot(x, y)               # conjugated inner product
```

**2. LBT drop-in.** `PureBLAS.activate()` registers in-process `@cfunction` pointers to the native
`@ccallable` kernels (123 symbols), after which `A*B`, `mul!`, `cholesky`, `qr`, `svd`, `eigen` and
`LinearAlgebra.BLAS.*` all dispatch to PureBLAS. `deactivate()` restores OpenBLAS.

```julia
using PureBLAS, LinearAlgebra
PureBLAS.activate()
F = cholesky(A)                  # → PureBLAS potrf
U, S, V = svd(B)                 # → PureBLAS gesvd
PureBLAS.deactivate()
```

The same `@ccallable` symbols compile to `libpureblas.so` via `juliac --trim` for C/C++/Rust hosts,
which doubles as a standing proof that the kernels stay trim-compatible.

## Coverage

All four element types (`Float32`, `Float64`, `ComplexF32`, `ComplexF64`) from one generic kernel set.
Every run measures **13 BLAS-1**, **19 BLAS-2**, **18 BLAS-3** and **35 LAPACK** routines —
factorizations (Cholesky including banded, packed and pivoted; LU; QR including pivoted;
Bunch–Kaufman), triangular and general solves, least squares, SVD, and symmetric/Hermitian and general
eigensolvers.

Correctness is oracled against OpenBLAS/LAPACK across types, sizes, strides and edge cases, with
StrictMode contracts (`@assert_typestable`, `@assert_noalloc`, trim-safety) dogfooded on the hot paths.

## How it adapts to your machine

PureBLAS detects the host at build time (`CpuId`/`HostCPUFeatures`) and bakes cache sizes, SIMD width
and microarchitecture into const-folded, trim-safe constants. Every tuning parameter — block sizes,
base-case cutoffs, unroll factors, algorithm-switch thresholds — then resolves through the **PDM
ladder**: a `Preferences` **P**in wins if set; otherwise the default is either **D**erived as a formula
over those detected constants, or — where the optimum depends on a ratio of our own kernels' rates that
no formula predicts — **M**easured on the host by a one-shot auto-tune. A fixed per-microarchitecture
literal is treated as a defect, not a tuning. See [Tuning Constants](tuning.md).

That is the one structural advantage a JIT-compiled BLAS has over a statically shipped one, and it is
much of the point of the project: OpenBLAS and BLIS must choose their block sizes when *they* are
compiled; PureBLAS chooses yours when it loads.

MIT licensed. See the [Guide](guide.md), [Design](design.md) and
[SIMD & Hardware Adaptation](simd.md).
