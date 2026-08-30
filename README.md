# PureBLAS.jl

A BLAS and LAPACK written entirely in Julia. No Fortran, no C, no assembly.

It replaces OpenBLAS or MKL, and it is part of the Pure Julia Ecosystem — pure-Julia replacements for
Julia's non-Julia default libraries (sibling: [PureFFT.jl](https://github.com/el-oso/PureFFT.jl)).

Requires Julia 1.12. MIT licensed.

## Why bother

Two reasons a compiled BLAS cannot match.

**You can differentiate through it.** The kernels are plain Julia, generic over the element type, so a
`ForwardDiff.Dual` flows straight through them. An opaque `ccall` into OpenBLAS never allowed that.

**It tunes itself to the machine it runs on.** OpenBLAS and AOCL ship kernels hand-written per
microarchitecture, baked in when *they* were compiled — on a CPU they never benchmarked, they guess.
PureBLAS reads the cache sizes, vector width and register count at load time and computes its block
sizes from them. That is also why it is about 36k lines of code where OpenBLAS is a few hundred thousand
lines of x86 kernels alone: one generic kernel replaces a table of hand-written ones.

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

This is the path to use for automatic differentiation — there is no `ccall` boundary in the way.

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
nothing to build or load. This route goes through the C ABI, so it is *not* differentiable — for AD,
call `PureBLAS.*` directly.

## What works

**Every LAPACK symbol Julia can call now routes to PureBLAS.** That includes the eigensolvers,
Schur reordering, generalized SVD, the expert drivers like `gesvx`, and the banded and packed routines —
not just the headline factorizations. A test enumerates every symbol the standard library wraps and
fails if any still falls through to OpenBLAS.

**Performance.** The bar is `max(OpenBLAS, AOCL-BLIS)` — beating only one of them does not count. About
87% of benchmarked cells meet it, single-threaded, on frequency-locked Zen3, Zen4 and Zen5 machines.
The rest are tracked openly in [Performance Notes](docs/src/notes.md); most sit within a few percent.

**Element types.** BLAS levels 1–3 are generic over `T<:Number`, so `Float32`, `Float64`, complex, and
`ForwardDiff.Dual` all work. Among the factorizations, Cholesky, LU, QR and SVD singular values accept
any real type; singular *vectors* are Float64 and complex only.

## A shared library for C, C++ and Rust

```bash
julia juliac/build.jl        # builds libpureblas.so (Julia 1.12, experimental juliac --trim)
```

The result exports the standard ILP64 BLAS symbols and initializes its own embedded runtime, so a
non-Julia program can link it as its BLAS backend. See [`juliac/ctest.c`](juliac/ctest.c).

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

The suite checks results against OpenBLAS across every type, size and stride; differentiability; package
quality with Aqua; and a set of guarantees enforced with StrictMode — type stability, no allocation, no
register spills, and compatibility with `juliac --trim`. There is also a lint that fails the build if a
hardware-tuning constant is hardcoded instead of derived from detected CPU features.

[`ROADMAP.md`](ROADMAP.md) has the current status and what is planned. [`CLAUDE.md`](CLAUDE.md) has the
rules the project holds itself to.
