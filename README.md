# PureBLAS.jl

A BLAS and LAPACK written entirely in Julia. No Fortran, no C, no assembly.

It replaces OpenBLAS or MKL inside Julia, and it also compiles to a shared library that C, C++ or Rust
can link — so the same kernels serve both worlds. It is part of the Pure Julia Ecosystem, a set of
pure-Julia replacements for Julia's non-Julia default libraries (sibling:
[PureFFT.jl](https://github.com/el-oso/PureFFT.jl)).

Requires Julia 1.12. MIT licensed.

## Why bother

**It tunes itself to the machine it runs on.** OpenBLAS and AOCL ship kernels hand-written per
microarchitecture, baked in when *they* were compiled — on a CPU they never benchmarked, they guess.
PureBLAS reads the cache sizes, vector width and register count at load time and computes its block
sizes from them. That is also why it is about 36k lines of code where OpenBLAS is a few hundred thousand
lines of x86 kernels alone: one generic kernel replaces a table of hand-written ones.

**The kernels are ordinary Julia, so you can read and change them.** No assembly, no build system, no
Fortran. A block size is a formula you can follow, not a constant in a table.

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

**Every LAPACK symbol Julia can call now routes to PureBLAS.** That includes the eigensolvers,
Schur reordering, generalized SVD, the expert drivers like `gesvx`, and the banded and packed routines —
not just the headline factorizations. A test enumerates every symbol the standard library wraps and
fails if any still falls through to OpenBLAS.

**Performance.** The bar is `max(OpenBLAS, AOCL-BLIS)` — beating only one of them does not count. About
87% of benchmarked cells meet it, single-threaded, on frequency-locked Zen3, Zen4 and Zen5 machines.
The rest are tracked openly in [Performance Notes](docs/src/notes.md); most sit within a few percent.

**Element types.** BLAS levels 1–3 are generic over `T<:Number` — `Float32`, `Float64`, and complex.
Among the factorizations, Cholesky, LU, QR and SVD singular values accept any real element type;
singular *vectors* are Float64 and complex only.

## Automatic differentiation: partial

Be aware of what this does and does not do today.

Because the kernels are generic, a `ForwardDiff.Dual` flows through them and **forward-mode
differentiation works**. It is tested — eight test items check real derivatives through `gemm`, `gemv`,
`ger`, `trmm`, `trsm`, `trmv`, `trsv`, `symv`, `nrm2`, `asum`, `dot` and `potrf`.

Two honest caveats:

- **Reverse mode is not supported.** There are no ChainRules or Enzyme rules yet. That is planned work,
  not finished work.
- **Differentiating is correct but slow.** The SIMD kernels are selected on `T<:BlasReal`, and a `Dual`
  is not one, so it falls to the generic scalar path. You get the right derivative at scalar speed, not
  at BLAS speed.

So today: useful if you need to differentiate through a linear-algebra call and correctness matters more
than throughput. Not yet a replacement for a hand-written adjoint.

**Where this is going.** The aim is differentiation at full BLAS speed, and the route is reverse-mode
rules rather than pushing `Dual` numbers through the kernels. An `rrule` calls the ordinary `Float64`
primal — the SIMD kernel, at full speed — and expresses the adjoint as more BLAS calls, which run at
full speed too. Nothing has to be differentiated element by element.

That is achievable because these kernels are Julia the compiler can see through, rather than an opaque
`ccall`. It is the reason the project cares about AD at all. **None of it is written yet** — it is
milestone M6 in [`ROADMAP.md`](ROADMAP.md), and the recent work making LU, QR and SVD generic was
groundwork for it, not the thing itself.

## A shared library other languages can link

This is a goal of the project, not a side effect: **a BLAS written in Julia that any language can use.**
If it works, "written in Julia" stops being a constraint on who can benefit. A C program, a Rust crate
or a Python extension links `libpureblas.so` the same way it would link OpenBLAS, and never needs to
know what it was written in.

```bash
julia juliac/build.jl        # builds libpureblas.so
```

The library exports the standard ILP64 BLAS symbols and initializes its own embedded runtime, so a
non-Julia program can link it as its BLAS backend. CI builds it on every push and then calls into it
from C — a library that compiles but does not work cannot ship green. See
[`juliac/ctest.c`](juliac/ctest.c).

Status: it builds and it works, but `juliac --trim` is experimental and Julia 1.12 specific, so treat
this as promising rather than production-ready. The same `@ccallable` symbols serve both this and the
in-process reroute above, so the C entry points are exercised constantly rather than bit-rotting.

> Do not load this `.so` into a running Julia process. It embeds its own libjulia, which double-initializes
> and aborts. Inside Julia, use `PureBLAS.activate()` instead — it needs no shared library at all.

## Limitations

- **Single-threaded.** Multithreading is deliberately deferred.
- **Singular vectors are Float64 and complex only.** The generic path computes singular values and
  raises an error if you ask it for vectors, rather than quietly returning less than you asked for.
- **A few large-`n` triangular and rank-k cells trail AOCL** by a couple of percent. This is the gap
  between what LLVM emits and hand-written assembly, and closing it fully would mean writing assembly.

## Developing

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. -e 'using Pkg; Pkg.test(test_args=["Aqua"])'   # one item
```

The suite checks results against OpenBLAS across every type, size and stride; forward-mode derivatives
through the native kernels; package
quality with Aqua; and a set of guarantees enforced with StrictMode — type stability, no allocation, no
register spills, and compatibility with `juliac --trim`. There is also a lint that fails the build if a
hardware-tuning constant is hardcoded instead of derived from detected CPU features.

[`ROADMAP.md`](ROADMAP.md) has the current status and what is planned. [`CLAUDE.md`](CLAUDE.md) has the
rules the project holds itself to.
