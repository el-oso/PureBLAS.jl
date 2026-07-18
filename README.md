# PureBLAS.jl

A pure-Julia BLAS/LAPACK, part of the **Pure Julia Ecosystem** — pure-Julia replacements for Julia's
non-Julia default libraries (sibling: [PureFFT.jl](https://github.com/el-oso/PureFFT.jl)). PureBLAS
replaces OpenBLAS/MKL, and it is usable two ways:

1. **Native Julia API** — call it directly. Because it's plain Julia source, it is **differentiable**
   (ForwardDiff/Enzyme/ChainRules), which opaque `ccall`s into OpenBLAS never allowed.
2. **Whole-ecosystem reroute** — `PureBLAS.activate()` reroutes all of LinearAlgebra's BLAS/LAPACK to
   PureBLAS in the running process (MKL.jl-style), so existing `A*B`/`cholesky`/`qr`/… code uses it with
   no changes. The same symbols also compile to `libpureblas.so` (`juliac --trim`) for **non-Julia hosts**.

**Status:** feature-complete BLAS Levels 1–3 (real + complex, `s`/`d`/`c`/`z`) plus core LAPACK
factorizations — Cholesky (`potrf`), LU (`getrf`), QR (`geqrf`), SVD (`gesvd`) — real and complex.
Performance meets or beats **`max(OpenBLAS, AOCL-BLIS)`** (gate ≥ 1.0×) on the bulk of operations across an
AMD fleet (Zen3/AVX2, Zen4/AVX-512, Zen5/native-AVX-512), single-threaded. See [`ROADMAP.md`](ROADMAP.md)
and [`CHANGELOG.md`](CHANGELOG.md). MIT licensed.

## Native API (Mode 2)

The recommended in-Julia path — direct calls, AD-traceable, no `ccall` boundary.

```julia
using PureBLAS

# BLAS-1
x = randn(1000); y = randn(1000)
PureBLAS.axpy!(y, 2.0, x)   # y .+= 2.0 .* x
PureBLAS.dot(x, y)          # conjugated inner product (matches LinearAlgebra.dot)
PureBLAS.nrm2(x)            # Euclidean norm (overflow-safe, via lassq)

# BLAS-2 / BLAS-3
A = randn(256, 256); B = randn(256, 256); C = zeros(256, 256)
PureBLAS.gemv!(y, A, x)                                  # y = A·x
PureBLAS.gemm!(C, A, B)                                  # C = A·B
PureBLAS.trmm!(B, A; side='L', uplo='U')                # B = op(A)·B, A triangular

# LAPACK factorizations (in place)
S = A*A' + size(A,1)*I |> Matrix
PureBLAS.potrf!(S; uplo='L')                            # Cholesky
PureBLAS.getrf!(copy(A))                                # LU (returns A, ipiv, info)
```

Works for `Float32/Float64/ComplexF32/ComplexF64` — and any other `T<:Number`, including
`ForwardDiff.Dual`, so you can differentiate through these calls.

## Whole-ecosystem reroute (Mode 1, in-process)

Reroute everything LinearAlgebra dispatches through BLAS/LAPACK to PureBLAS — no code changes, just one call:

```julia
using PureBLAS, LinearAlgebra
PureBLAS.activate()          # A*B, mul!, cholesky, qr, svd, LinearAlgebra.BLAS.* now use PureBLAS
A = randn(512, 512)
C = A * A'                   # gemm → PureBLAS
cholesky(C)                  # potrf → PureBLAS
PureBLAS.deactivate()        # restore OpenBLAS
```

This registers in-process `@cfunction` pointers to PureBLAS's native kernels via libblastrampoline
(`lbt_set_forward`) — it runs in the live process, so there is no separate library to build or load.
Note: this C-ABI path is **not** differentiable (no compiled BLAS is) — for AD, call the native `PureBLAS.*`
API directly.

## `libpureblas.so` — for non-Julia hosts

```bash
julia juliac/build.jl        # build libpureblas.so (Julia 1.12, experimental juliac --trim)
```

The resulting `libpureblas.so` exports the reference-BLAS ILP64 symbols and self-initializes its embedded
runtime, so a **C/C++/Rust** host can link it as its BLAS backend (see [`juliac/ctest.c`](juliac/ctest.c)).

> Note: the `.so` is for **non-Julia hosts**. Do *not* `BLAS.lbt_forward(libpureblas.so)` from inside a
> running Julia process — a juliac library embeds its own `libjulia` and double-inits it on LBT's probe
> (`signal 6`). Inside Julia use `PureBLAS.activate()` (above), which forwards `@cfunction` pointers to the
> in-process kernels and needs no `.so`. See [`ROADMAP.md`](ROADMAP.md) → "In-process LBT forwarding".

## Known limitations

- **Forwarding the `.so` into a live Julia process is blocked** (juliac double-init); this is *not* a
  limitation on rerouting to PureBLAS in Julia — use `PureBLAS.activate()` (in-process `@cfunction`
  forwarding, above), which needs no `.so`.
- **Complex-dot ABI symbols deferred** — the four `c/zdotu`, `c/zdotc` `@ccallable` symbols have an
  unresolved complex-return ABI; the native API covers complex dot.
- **Single-threaded** — multithreading is deferred by design.
- **Large-n `trmm`/`syrk` vs AOCL** sit ~0.95–0.98 at n≥2048 (an LLVM-vs-hand-asm microkernel gap); every
  other benched op meets the `max(OB, AOCL)` gate.

## Develop & test

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'                       # full suite (ReTestItems)
# trigger one item:
julia --project=. -e 'using Pkg; Pkg.test(test_args=["Aqua"])'
```

The suite checks correctness against OpenBLAS (`LinearAlgebra.BLAS`) across all types/sizes/strides, the
native + AD paths, Aqua package-quality, StrictMode guarantees (type-stable, allocation-free, trim-safe,
no-spill, memory-safe), and a req#8 lint against hardcoded hardware-tuning literals. See
[`CLAUDE.md`](CLAUDE.md) for the project's hard requirements.
