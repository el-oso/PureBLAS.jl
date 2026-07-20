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

Real **and complex** unit-stride dense inputs take a SIMD.jl fast path (the complex path uses portable
interleaved-`Vec` kernels — see the Performance page's Complex section); strided inputs and AD element
types (e.g. `ForwardDiff.Dual`) take the generic scalar loop, which is what keeps the calls differentiable.

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

## In-process drop-in (`activate`)

To reroute `LinearAlgebra`'s BLAS/LAPACK to PureBLAS **inside a live Julia process** (MKL.jl-style), call
`PureBLAS.activate()`. After it, `A*B`, `mul!`, `\`, `cholesky`, `qr`, `svd`, `eigen`, and
`LinearAlgebra.BLAS.*` dispatch to PureBLAS. `PureBLAS.deactivate()` restores the original backend.

Coverage is **complete**: every LAPACK symbol `LinearAlgebra` can `ccall` forwards to PureBLAS
after `activate()` — the OpenBLAS fallback is fully removed (see [Coverage](coverage.md), enforced
by a ratchet test that asserts zero fallthrough).

```julia
using LinearAlgebra, PureBLAS
PureBLAS.activate()
PureBLAS.is_active()          # true
PureBLAS.status()             # how many symbols route to PureBLAS + notes
A = randn(1000, 1000); A = A'A + 1000I
cholesky!(copy(A))            # → PureBLAS potrf
PureBLAS.deactivate()
```

**`get_config()` still lists `libopenblas` — that is expected.** `activate()` overlays per-symbol
`@cfunction` forwards on top of the loaded OpenBLAS (it does not swap the *loaded library*), so
`LinearAlgebra.BLAS.get_config()` — which reports loaded libraries — cannot show PureBLAS. Use
`PureBLAS.is_active()` / `PureBLAS.status()` to confirm routing, or compare
`BLAS.lbt_get_forward(sym, …)` before/after `activate()` (a changed pointer = handled by PureBLAS).

### Performance note: the Cholesky/`potrf` triangle

`potrf!` (Cholesky) has a specialized fast base for the **lower** triangle; the **upper** triangle
currently goes through the generic path and is meaningfully slower (roughly ~2× at moderate `n`).
`LinearAlgebra.cholesky!(A::Matrix)` defaults to the **upper** factorization, so it hits the slower
path. For the fast base today, factor the **lower** triangle:

```julia
cholesky!(Hermitian(A, :L))         # routes to potrf!('L') — the fast base
PureBLAS.potrf!(A; uplo = 'L')      # native (Mode 2), same fast base
```

A fast upper path (transpose → fast-lower base → transpose back) is a scheduled fix; once it lands the
default `cholesky!` gets the fast base with no change on your side.

## Testing

```bash
julia --project=test test/runtests.jl
# one item:
julia --project=test -e 'using ReTestItems, PureBLAS; runtests(PureBLAS; name="Level-1 contiguous vs OpenBLAS")'
```
