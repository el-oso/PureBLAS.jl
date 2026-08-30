# Guide

## Calling PureBLAS directly

The BLAS-1 calls take any `AbstractVector{T}` where `T<:Number` — the four BLAS types
(`Float32`, `Float64`, `ComplexF32`, `ComplexF64`) and others besides, `ForwardDiff.Dual` among them.

| Call | Meaning |
|------|---------|
| `PureBLAS.axpy!(y, a, x)` | `y .+= a .* x` |
| `PureBLAS.scal!(a, x)` | `x .*= a` |
| `PureBLAS.blascopy!(y, x)` | `y .= x` |
| `PureBLAS.swap!(x, y)` | swap contents |
| `PureBLAS.dot(x, y)` | conjugated inner product `conj(x)·y` (matches `LinearAlgebra.dot`) |
| `PureBLAS.dotu(x, y)` | unconjugated inner product `x·y` |
| `PureBLAS.nrm2(x)` | Euclidean norm, safe against overflow and underflow (LAPACK lassq) |
| `PureBLAS.asum(x)` | `Σ|xᵢ|` (complex: `Σ|Re|+|Im|`) |
| `PureBLAS.iamax(x)` | 1-based index of `argmax|xᵢ|` |

Levels 2 and 3 follow the same pattern — `gemv!`, `ger!`, `trmm!`, `trsm!`, `gemm!` and the rest — as
do the LAPACK entry points, `potrf!`, `getrf!`, `geqrf!`, `gesvd!` and friends.

Unit-stride dense inputs take a SIMD.jl fast path, real and complex alike; the complex kernels use
portable interleaved `Vec` operations rather than anything x86-specific. Strided inputs and unusual
element types fall to a generic scalar loop, which is slower but is also what lets `Dual` numbers
through.

### Differentiating

```julia
using PureBLAS, ForwardDiff
x = randn(128); v = randn(128)
ForwardDiff.derivative(t -> PureBLAS.nrm2(x .+ t .* v), 0.0)
```

This works because the kernels are ordinary Julia. Bear in mind it runs on the scalar path — a `Dual`
is not one of the types the SIMD kernels handle — so it is correct but not fast. Reverse mode is not
supported yet.

## Handing the whole ecosystem over

`PureBLAS.activate()` reroutes LinearAlgebra's BLAS and LAPACK calls to PureBLAS inside the running
process, MKL.jl-style. After it, `A*B`, `mul!`, `\`, `cholesky`, `qr`, `svd`, `eigen` and
`LinearAlgebra.BLAS.*` all go to PureBLAS. `deactivate()` puts the original backend back.

Every LAPACK symbol `LinearAlgebra` can `ccall` is covered — see [Coverage](coverage.md), which a
ratchet test keeps honest by failing if anything falls through to OpenBLAS.

```julia
using LinearAlgebra, PureBLAS
PureBLAS.activate()
PureBLAS.is_active()          # true
PureBLAS.status()             # how many symbols route to PureBLAS, plus notes
A = randn(1000, 1000); A = A'A + 1000I
cholesky!(copy(A))            # → PureBLAS potrf
PureBLAS.deactivate()
```

One thing that trips people up: `BLAS.get_config()` still lists `libopenblas` afterwards, and that is
expected. `activate()` overlays per-symbol `@cfunction` forwards on top of the loaded OpenBLAS rather
than swapping the loaded library, and `get_config()` reports libraries, so it cannot show PureBLAS.
Use `PureBLAS.is_active()` or `PureBLAS.status()` to check, or compare `BLAS.lbt_get_forward(sym, …)`
before and after — a changed pointer means PureBLAS is handling it.

## Building the shared library

```bash
julia juliac/build.jl     # -> juliac/build/libpureblas.so, exporting daxpy_64_, ddot_64_, ...
```

This is for calling PureBLAS from C, C++ or Rust; `juliac/ctest.c` is a worked example. Do not load the
result into a live Julia process — see [Design](design.md) for why that aborts, and use `activate()`
instead.

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
# a single item:
julia --project=. -e 'using Pkg; Pkg.test(test_args=["Level-1 contiguous vs OpenBLAS"])'
```
