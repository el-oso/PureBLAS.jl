# Design

## One kernel set, two ways in

The kernels are written once, in BLAS-native `(n, …, inc)` form, over a small accessor interface —
`_ld` and `_st!` in `core.jl` — that reads and writes uniformly through either a `Ptr{T}` or an
`AbstractVector{T}`:

```julia
@inline _ld(p::Ptr, i::Integer) = unsafe_load(p, i)
@inline _ld(a, i::Integer) = @inbounds a[i]
```

That is the whole trick behind having two front ends without two implementations. Two layers sit on
top of the same kernels:

The **native API** (`backend.jl`, `native.jl`) exposes ergonomic `AbstractArray` methods on a
`SIMDBackend`, which satisfies a TypeContracts interface. There is no `ccall` boundary anywhere in the
call tree, so it is all plain Julia the compiler can see into — which is what lets `ForwardDiff.Dual`
flow through.

The **C ABI** (`cabi/`) exposes 167 `@ccallable` ILP64 reference-BLAS and LAPACK symbols — `daxpy_64_`
and the rest, arguments by reference, column-major, `Int64`. These serve two consumers. `juliac --trim`
compiles them into `libpureblas.so` for C, C++ and Rust hosts, and `activate()` registers in-process
`@cfunction` pointers to the same entry points so LinearAlgebra dispatches to them without any shared
library involved.

The implementation spans BLAS levels 1 through 3 — dense, packed and banded — and about forty LAPACK
files covering the factorizations, solves, least squares, SVD and the eigensolvers.

## Generic over `T<:Number`

One implementation covers `s`, `d`, `c` and `z`, and anything else numeric. Unit-stride dense inputs
dispatch to a SIMD.jl path using `Vec{N,T}` at the detected register width; everything else — strided
data, unusual element types — runs the generic scalar loop. Keeping that scalar path around, rather
than specialising it away, is what makes the library differentiable.

Width detection covers AVX-512 and AVX2, and falls back to 16 bytes where it cannot identify the host,
which is the right answer for SSE2 and NEON. The benchmark fleet is x86-64 only, so treat ARM as
untested rather than supported.

## Why the shared library cannot be forwarded into live Julia

A `juliac --trim` library embeds the Julia runtime. Its `@ccallable` entry points are wrapped in
`ijl_autoinit_and_adopt_thread`, which initializes that runtime lazily on the first call — exactly what
a C host needs. But calling `BLAS.lbt_forward` on it from inside a running Julia process makes LBT call
a probe symbol (`isamax_64_`) while autodetecting the interface, and that second initialization of the
shared `libjulia` aborts the process with signal 6.

So the division is:

- Inside Julia, use `activate()`. It forwards `@cfunction` pointers to the in-process kernels and needs
  no `.so` at all.
- The `.so` is for non-Julia hosts, and keeping it building doubles as a standing check that the kernels
  stay trim-compatible. `juliac/ctest.c` exercises it from C on every push.
- Getting `lbt_forward` to work would need upstream juliac support for initializing against the host
  runtime, or a codegen path with no embedded runtime. Tracked in `ROADMAP.md`.
