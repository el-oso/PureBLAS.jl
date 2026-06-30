# Design

## One kernel set, two consumption modes

PureBLAS has a single set of low-level Level-1 kernels in BLAS-native `(n, …, inc)` form, written
over a tiny accessor interface (`_ld`/`_st!`) that works uniformly over `Ptr{T}` (C-ABI) and
`AbstractVector{T}` (native). Two layers sit on top:

- **Mode 2 — native API** (`backend.jl`, `native.jl`): ergonomic `AbstractVector` methods on a
  `SIMDBackend <: AbstractBLAS1` (a TypeContracts interface). No `ccall` boundary, so the whole
  call tree is plain Julia and **differentiable**.
- **Mode 1 — C ABI** (`cabi.jl`): `@ccallable` ILP64 reference-BLAS symbols (`daxpy_64_`, by
  reference, column-major, `Int64`). BLAS-1 has no character arguments, so there are no hidden
  Fortran string-length args. Compiled to `libpureblas.so` by `juliac --trim`.

## Generic over `T<:Number`

One kernel implementation covers `s/d/c/z` and any other `T<:Number`. Real, unit-stride, dense
inputs dispatch to a SIMD.jl path (`Vec{N,T}` at the detected register width — AVX-512 / AVX2 /
NEON); everything else uses the generic scalar loop. The generic path is what lets `ForwardDiff.Dual`
flow through.

## Why LBT forwarding into live Julia does not work (yet)

A `juliac --trim` library **embeds the Julia runtime**. Its `@ccallable` entry points are wrapped
with `ijl_autoinit_and_adopt_thread`, which lazily initializes that runtime on first call — perfect
for a non-Julia host. But `BLAS.lbt_forward` from inside a running Julia process makes LBT call a
probe symbol (`isamax_64_`) during interface autodetection, which **double-initializes the shared
`libjulia`** and aborts (signal 6).

Therefore:

- Inside Julia, use **Mode 2** (native API / pkgimage) — also the AD-enabling path.
- The `.so` is for **non-Julia consumers** and for proving **trim-compatibility** (the build
  succeeds and exports all 30 BLAS-1 symbols; a C host runs it correctly — `juliac/ctest.c`).
- Re-enabling LBT forwarding needs upstream juliac support for initializing against the host
  runtime (or a runtime-free codegen path). Tracked in `ROADMAP.md`.
