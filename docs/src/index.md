# PureBLAS.jl

A pure-Julia BLAS, part of the **Pure Julia Ecosystem** — pure-Julia replacements for Julia's
non-Julia default libraries (sibling: [PureFFT.jl](https://github.com/el-oso/PureFFT.jl)).

PureBLAS targets ≥ 0.96× OpenBLAS performance while being usable two ways:

1. **Native Julia API** — call it directly. Because it is plain Julia source, it is
   **differentiable** (ForwardDiff today; Enzyme/ChainRules planned) — something opaque `ccall`s
   into OpenBLAS never allowed.
2. **Shared library** — compiled to `libpureblas.so` via `juliac --trim`, usable from non-Julia
   hosts (C/C++/Rust) and as a trim-compatibility proof.

**Status:** Milestone 1 — BLAS Level 1, all four element types (`Float32/Float64/ComplexF32/ComplexF64`)
via generic `T<:Number` kernels with SIMD.jl fast paths. See the [Guide](guide.md) and
[Design](design.md). MIT licensed.

```julia
using PureBLAS
x = randn(1000); y = randn(1000)
PureBLAS.axpy!(y, 2.0, x)   # y .+= 2x
PureBLAS.nrm2(x)            # Euclidean norm
PureBLAS.dot(x, y)          # conjugated inner product
```
