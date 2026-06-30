# Guide

## Native API (Mode 2)

All operations accept any `AbstractVector{T}` with `T<:Number` — the BLAS types
(`Float32/Float64/ComplexF32/ComplexF64`) and others such as `ForwardDiff.Dual`.

| Call | Meaning |
|------|---------|
| `PureBLAS.axpy!(y, a, x)` | `y .+= a .* x` |
| `PureBLAS.scal!(a, x)` | `x .*= a` |
| `PureBLAS.blascopy!(y, x)` | `y .= x` |
| `PureBLAS.swap!(x, y)` | swap contents |
| `PureBLAS.dot(x, y)` | conjugated inner product `conj(x)·y` (matches `LinearAlgebra.dot`) |
| `PureBLAS.dotu(x, y)` | unconjugated inner product `x·y` |
| `PureBLAS.nrm2(x)` | Euclidean norm (overflow/underflow-safe, LAPACK lassq) |
| `PureBLAS.asum(x)` | `Σ|xᵢ|` (complex: `Σ|Re|+|Im|`) |
| `PureBLAS.iamax(x)` | 1-based index of `argmax|xᵢ|` |

Real, unit-stride, dense vectors take the SIMD.jl fast path; complex, strided, and AD element
types take a generic scalar loop (which is what makes the calls differentiable).

### Automatic differentiation

```julia
using PureBLAS, ForwardDiff
x = randn(128); v = randn(128)
ForwardDiff.derivative(t -> PureBLAS.nrm2(x .+ t .* v), 0.0)   # works — pure-Julia kernels
```

## Shared library (Mode 1)

```bash
julia juliac/build.jl     # -> juliac/build/libpureblas.so, exports daxpy_64_, ddot_64_, ...
```

Use it from a non-Julia host (see `juliac/ctest.c`). Forwarding the library into a *live Julia*
process via `BLAS.lbt_forward` is not currently possible — see [Design](design.md).

## Testing

```bash
julia --project=test test/runtests.jl
# one item:
julia --project=test -e 'using ReTestItems, PureBLAS; runtests(PureBLAS; name="Level-1 contiguous vs OpenBLAS")'
```
