# PureBLAS.jl

A pure-Julia BLAS, part of the **Pure Julia Ecosystem** — pure-Julia replacements for Julia's
non-Julia default libraries (sibling: [PureFFT.jl](https://github.com/el-oso/PureFFT.jl)).
PureBLAS aims to match/beat OpenBLAS (gate: ≥ 0.96×) while being usable two ways:

1. **Native Julia API** — call it directly; because it's plain Julia source, it is
   **differentiable** (ForwardDiff/Enzyme), which opaque `ccall`s into OpenBLAS never allowed.
2. **libblastrampoline drop-in** — compiled to `libpureblas.so` via `juliac --trim` and forwarded
   with `BLAS.lbt_forward`, so all of `LinearAlgebra` transparently uses PureBLAS.

**Status:** Milestone 1 (BLAS Level 1, all four element types). See [`ROADMAP.md`](ROADMAP.md).
MIT licensed.

## Native API (Mode 2)

```julia
using PureBLAS
x = randn(1000); y = randn(1000)

PureBLAS.axpy!(y, 2.0, x)   # y .+= 2.0 .* x
PureBLAS.dot(x, y)          # conjugated inner product (matches LinearAlgebra.dot)
PureBLAS.dotu(x, y)         # unconjugated
PureBLAS.nrm2(x)            # Euclidean norm (overflow-safe)
PureBLAS.asum(x)            # Σ|xᵢ|
PureBLAS.iamax(x)           # argmax|xᵢ| (1-based)
PureBLAS.scal!(2.0, x)      # x .*= 2.0
```

Works for `Float32/Float64/ComplexF32/ComplexF64` — and any other `T<:Number`, including
`ForwardDiff.Dual`, so you can differentiate through these calls.

## LBT drop-in (Mode 1)

```julia
julia juliac/build.jl                 # build libpureblas.so (Julia 1.12, experimental juliac --trim)

using LinearAlgebra, PureBLAS
PureBLAS.activate()                   # forward libblastrampoline to PureBLAS
BLAS.get_config()                     # shows libpureblas
# ... LinearAlgebra BLAS-1 now runs through PureBLAS ...
PureBLAS.deactivate()                 # restore the original backend (e.g. OpenBLAS)
```

## Develop & test

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=test test/runtests.jl                      # full suite
# trigger one item:
julia --project=test -e 'using ReTestItems, PureBLAS; runtests(PureBLAS; name="Level-1 contiguous vs OpenBLAS")'
```

Tests check correctness against OpenBLAS (`LinearAlgebra.BLAS`) across all types/sizes/strides,
the native + AD paths, and StrictMode guarantees (type-stable, allocation-free, trim-safe). See
[`CLAUDE.md`](CLAUDE.md) for the project's hard requirements.
