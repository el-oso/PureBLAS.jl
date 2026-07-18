# PureBLAS.jl

A pure-Julia BLAS/LAPACK, part of the **Pure Julia Ecosystem** — pure-Julia replacements for Julia's
non-Julia default libraries (sibling: [PureFFT.jl](https://github.com/el-oso/PureFFT.jl)). PureBLAS
replaces OpenBLAS/MKL, and it is usable two ways:

1. **Native Julia API** — call it directly. Because it's plain Julia source, it is **differentiable**
   (ForwardDiff/Enzyme/ChainRules), which opaque `ccall`s into OpenBLAS never allowed.
2. **libblastrampoline drop-in** — compiled to `libpureblas.so` via `juliac --trim`, providing the
   reference-BLAS (ILP64) symbols for **non-Julia hosts** (C/C++/Rust).

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

## LBT drop-in (Mode 1) — for non-Julia hosts

```bash
julia juliac/build.jl        # build libpureblas.so (Julia 1.12, experimental juliac --trim)
```

The resulting `libpureblas.so` exports the reference-BLAS ILP64 symbols and self-initializes its embedded
runtime, so a **C/C++/Rust** host can link it as its BLAS backend (see [`juliac/ctest.c`](juliac/ctest.c)).

> **Known limitation:** forwarding this `.so` back into a *running* Julia process
> (`BLAS.lbt_forward(libpureblas.so)`) currently **aborts** (`signal 6`) — juliac libraries embed the Julia
> runtime and double-initialize the shared `libjulia` on LBT's autodetect probe. So **inside Julia, use the
> native API (Mode 2)**; the `.so` is for non-Julia hosts. This is revisitable once juliac gains a
> host-runtime-init mode. See [`ROADMAP.md`](ROADMAP.md) → "Key finding".

## Known limitations

- **LBT live-forward from a running Julia process is blocked** (juliac limitation, above).
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
