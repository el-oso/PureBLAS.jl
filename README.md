# PureBLAS.jl

A BLAS and LAPACK written entirely in Julia. No Fortran, no C, no assembly.

It replaces OpenBLAS or MKL inside Julia, and it also compiles to a shared library that C, C++ or Rust
can link — so the same kernels serve both worlds. It is part of the Pure Julia Ecosystem, a set of
pure-Julia replacements for Julia's non-Julia default libraries (sibling:
[PureFFT.jl](https://github.com/el-oso/PureFFT.jl)).

Requires Julia 1.12. MIT licensed.

## Why bother

OpenBLAS and AOCL ship kernels that were hand-written for each microarchitecture and baked in when
those libraries were compiled. Run them on a CPU nobody benchmarked and they are guessing. PureBLAS
reads the cache sizes, vector width and register count when it loads, and works its block sizes out
from those. It adapts to the machine instead of recognising it.

That is also why it is roughly 36k lines of code, against a few hundred thousand lines of x86 kernels
in OpenBLAS alone — one generic kernel does the job of a table of hand-written ones.

And it is all ordinary Julia, so you can read it, step through it in the debugger, and change it. A
block size is a formula you can follow rather than a number someone measured once.

## Two ways to use it

### Call it directly

```julia
using PureBLAS

x = randn(1000); y = randn(1000)
PureBLAS.axpy!(y, 2.0, x)     # y .+= 2.0 .* x
PureBLAS.nrm2(x)              # Euclidean norm, overflow-safe

A = randn(256, 256); B = randn(256, 256); C = zeros(256, 256)
PureBLAS.gemm!(C, A, B)       # C = A·B
PureBLAS.potrf!(A*A' + 256I; uplo='L')   # Cholesky, in place
```

These are ordinary Julia functions, so there is no `ccall` boundary in the way.

### Reroute the whole ecosystem

One call, and everything LinearAlgebra sends to BLAS or LAPACK goes to PureBLAS instead. No code
changes.

```julia
using PureBLAS, LinearAlgebra
PureBLAS.activate()

C = randn(512, 512) |> A -> A * A'   # gemm  → PureBLAS
cholesky(C)                          # potrf → PureBLAS
svd(C)                               # gesdd → PureBLAS

PureBLAS.deactivate()                # back to OpenBLAS
```

`activate()` registers function pointers with libblastrampoline in the running process, so there is
nothing to build or load. This route goes through the C ABI, so `Dual` numbers cannot pass through it —
if you need to differentiate, call `PureBLAS.*` directly and read the section below first.

## What works

Every LAPACK symbol Julia can call goes to PureBLAS — not just the headline factorizations, but the
eigensolvers, Schur reordering, generalized SVD, expert drivers like `gesvx`, and the banded and packed
routines too. A test walks every symbol the standard library wraps and fails if any of them still lands
on OpenBLAS.

For speed the bar is `max(OpenBLAS, AOCL-BLIS)`, so beating one of them and losing to the other counts
as a loss. About 87% of benchmarked cells clear it, single-threaded, on frequency-locked Zen3, Zen4 and
Zen5 machines. The ones that do not are listed in [Performance Notes](docs/src/notes.md), and most of
them are within a few percent.

BLAS levels 1–3 work for any `T<:Number`, so `Float32`, `Float64` and complex are all covered. Of the
factorizations, Cholesky, LU, QR and SVD singular values take any real element type; singular *vectors*
are Float64 and complex only.

## Automatic differentiation

Forward mode works. The kernels are generic, so a `ForwardDiff.Dual` passes straight through them, and
the test suite differentiates `gemm`, `gemv`, `ger`, `trmm`, `trsm`, `trmv`, `trsv`, `symv`, `nrm2`,
`asum`, `dot` and `potrf` against known derivatives.

It is slow, though. The SIMD kernels only fire for `Float32` and `Float64`, and a `Dual` is neither, so
it falls back to plain scalar loops. You get the right answer at nothing like BLAS speed. Reverse mode
does not work at all yet — there are no ChainRules or Enzyme rules.

Where we want to end up is differentiation that runs at full BLAS speed, and the way to get there is
reverse-mode rules rather than pushing `Dual` numbers through the kernels. An `rrule` can call the
ordinary `Float64` kernel at full speed and write the adjoint as a couple more BLAS calls, which are
also full speed — no element-by-element differentiation anywhere. That is the payoff for the kernels
being Julia the compiler can see into instead of an opaque `ccall`, and it is why the project cares
about AD in the first place.

None of that is written yet. It is milestone M6 in [`ROADMAP.md`](ROADMAP.md). Making LU, QR and SVD
work for any real element type was the groundwork.

## A shared library other languages can link

Something we are deliberately aiming at, rather than a side effect: a BLAS written in Julia that any
language can use. A C program, a Rust crate or a Python extension links `libpureblas.so` exactly as it
would link OpenBLAS, and never has to care what it was written in.

```bash
julia juliac/build.jl        # builds libpureblas.so
```

It exports the usual ILP64 BLAS symbols and starts its own embedded runtime, so a non-Julia program can
use it as its BLAS backend. CI builds it on every push and then calls into it from C, because a library
that compiles but does not actually work would otherwise sail through. See
[`juliac/ctest.c`](juliac/ctest.c).

It builds and it works today, but `juliac --trim` is experimental and tied to Julia 1.12, so this is
promising rather than something to put in production. The same `@ccallable` symbols serve both the
shared library and the in-process reroute above, which keeps the C entry points exercised instead of
quietly rotting.

> Do not load this `.so` into a running Julia process. It embeds its own libjulia, which double-initializes
> and aborts. Inside Julia, use `PureBLAS.activate()` instead — it needs no shared library at all.

## Limitations

Everything is single-threaded for now; multithreading is deliberately on hold.

Singular vectors only work for Float64 and complex. Ask the generic path for vectors and it raises an
error rather than quietly handing back less than you asked for.

A handful of large-`n` triangular and rank-k cells trail AOCL by a couple of percent. That is the gap
between what LLVM emits and hand-written assembly, and closing it properly would mean writing assembly.

## Developing

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. -e 'using Pkg; Pkg.test(test_args=["Aqua"])'   # one item
```

The suite checks results against OpenBLAS across every type, size and stride, differentiates the native
kernels with ForwardDiff, and runs Aqua for package quality. StrictMode enforces a set of guarantees on
the hot paths — type stability, no allocation, no register spills, and that everything still compiles
under `juliac --trim`. A lint fails the build if someone hardcodes a hardware-tuning constant instead of
deriving it from detected CPU features.

[`ROADMAP.md`](ROADMAP.md) has the current status and what is planned. [`CLAUDE.md`](CLAUDE.md) has the
rules the project holds itself to.
