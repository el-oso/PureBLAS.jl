# PureBLAS.jl

A BLAS and LAPACK written entirely in Julia, part of the Pure Julia Ecosystem — pure-Julia replacements
for Julia's non-Julia default libraries (sibling: [PureFFT.jl](https://github.com/el-oso/PureFFT.jl)).

Everything is plain Julia source built on [SIMD.jl](https://github.com/eschnett/SIMD.jl). There is no
`libopenblas`, no Fortran and no vendor blob anywhere in the stack.

## The performance rule

PureBLAS has to be at least as fast as `max(OpenBLAS, AOCL)` at every size, on every microarchitecture.
Not a geometric mean, not "competitive with" — the gate looks at the *worst* cell of a routine against
the *faster* of the two references, and the routine passes only if that worst cell reaches parity.

Measurements run single-threaded on frequency-locked machines (`amd_pstate=passive`, boost off, cores
pinned to base clock) across Zen3/AVX2, Zen4/AVX-512 and Zen5/AVX-512. Both references are measured in
the same round as PureBLAS, so the `max` compares numbers that saw one machine state rather than three.

The [Coverage](coverage.md) page has the per-routine results, generated from the benchmark caches rather
than typed in by hand — including every routine that currently misses.

## Two ways to use it

Call the native API directly, and you are calling ordinary Julia functions with no `ccall` boundary in
the way:

```julia
using PureBLAS
x = randn(1000); y = randn(1000)
PureBLAS.axpy!(y, 2.0, x)        # y .+= 2x
PureBLAS.dot(x, y)               # conjugated inner product
```

Or hand the whole ecosystem over. `PureBLAS.activate()` registers in-process `@cfunction` pointers to
the native `@ccallable` kernels — 492 symbols — after which `A*B`, `mul!`, `cholesky`, `qr`, `svd`,
`eigen` and `LinearAlgebra.BLAS.*` all land in PureBLAS. `deactivate()` puts OpenBLAS back.

```julia
using PureBLAS, LinearAlgebra
PureBLAS.activate()
F = cholesky(A)                  # → PureBLAS potrf
U, S, V = svd(B)                 # → PureBLAS gesdd
PureBLAS.deactivate()
```

Those same `@ccallable` symbols compile to `libpureblas.so` through `juliac --trim`, so a C, C++ or Rust
program can link PureBLAS as its BLAS backend. Keeping that build working also keeps the kernels
trim-compatible, which is a useful thing to have checked continuously.

## What is covered

One generic kernel set handles all four element types — `Float32`, `Float64`, `ComplexF32` and
`ComplexF64`. Every benchmark run measures the full BLAS levels 1 through 3 plus a large slice of
LAPACK: Cholesky in its dense, banded, packed and pivoted forms, LU, QR with and without pivoting,
Bunch–Kaufman, triangular and general solves, least squares, SVD, and the symmetric, Hermitian and
general eigensolvers. [Coverage](coverage.md) lists them with current numbers.

Results are checked against OpenBLAS and LAPACK across types, sizes, strides and edge cases, and
StrictMode contracts (`@assert_typestable`, `@assert_noalloc`, trim-safety) run against the hot paths as
part of the suite.

## Differentiation

Forward mode works today. The kernels are generic, so a `ForwardDiff.Dual` passes straight through them
and the suite differentiates a dozen routines against known derivatives. It is not fast: the SIMD
kernels only fire for `Float32` and `Float64`, so a `Dual` drops to plain scalar loops and you get the
right answer slowly.

Reverse mode is not implemented. The plan is ChainRules and Enzyme rules that call the ordinary
`Float64` kernel at full speed and express the adjoint as further BLAS calls — differentiation at BLAS
speed rather than element by element. That work has not started.

## How it adapts to your machine

PureBLAS detects the host when it precompiles (`CpuId`/`HostCPUFeatures`) and bakes cache sizes, SIMD
width and microarchitecture into const-folded, trim-safe constants. Every tuning parameter — block
sizes, base-case cutoffs, unroll factors, algorithm-switch thresholds — then resolves through what the
project calls the PDM ladder. A `Preferences` **P**in wins if one is set. Otherwise the default is
either **D**erived, as a formula over those detected constants, or **M**easured on the host by a
one-shot auto-tune, which is reserved for the cases where the optimum depends on a ratio of our own
kernels' rates that no formula predicts. A fixed per-microarchitecture literal counts as a defect rather
than a tuning, and there is a lint that fails the build over it. [Tuning Constants](tuning.md) goes into
detail.

This is the one structural advantage a JIT-compiled BLAS has over a statically shipped one, and much of
the point of the project. OpenBLAS and BLIS have to choose their block sizes when *they* are compiled;
PureBLAS chooses yours when it loads.

MIT licensed. See the [Guide](guide.md), [Design](design.md) and
[SIMD & Hardware Adaptation](simd.md).
