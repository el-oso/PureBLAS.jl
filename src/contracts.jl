# Compile-time interface contract for BLAS Level-1 backends (TypeContracts.jl).
#
# A *backend* is the swappable unit of the Pure ecosystem: the M1 `SIMDBackend` (backend.jl), a
# future reference/GPU backend, etc. Each must provide the same Level-1 operation set so callers
# (the native API in native.jl, and downstream packages) depend on the interface, not the struct.
# `@contract` checks method existence + inferred return types at PRECOMPILE time and is eliminated
# by the trimmer — zero runtime cost (mirrors PureFFT's `AbstractFFTPlan` contract). Implementing
# methods carry explicit concrete return-type annotations so inference matches the contract.

using TypeContracts
using StrictMode  # @strict_contract / @verify_strict — the performance layer over the method surface

"""
    AbstractBLAS1

Supertype of all PureBLAS Level-1 backends. Concrete backends must satisfy the [`@contract`](@ref)
below over high-level `AbstractVector` arguments.
"""
abstract type AbstractBLAS1 end

function axpy! end      # y .+= a .* x
function scal! end      # x .*= a
function blascopy! end  # y .= x
function swap! end      # x ⇄ y
function dot end        # conjugated inner product (conj(x)·y)
function dotu end       # unconjugated inner product (x·y)
function nrm2 end       # Euclidean norm
function asum end       # Σ|xᵢ| (complex: Σ|Re|+|Im|)
function iamax end      # argmax|xᵢ|

# Level-1 is a *strict* contract: implementations must satisfy not just the method surface
# (TypeContracts) but StrictMode's performance guarantees — type-stable and allocation-free. The
# bandwidth-bound L1 kernels are where a stray allocation or type instability is most costly, so
# they carry the hardest guarantee. Verified by `@verify_strict SIMDBackend` (backend.jl).
@strict_contract AbstractBLAS1 begin
    axpy!(::Self, ::AbstractVector, ::Number, ::AbstractVector)::AbstractVector
    scal!(::Self, ::Number, ::AbstractVector)::AbstractVector
    blascopy!(::Self, ::AbstractVector, ::AbstractVector)::AbstractVector
    swap!(::Self, ::AbstractVector, ::AbstractVector)::Nothing
    dot(::Self, ::AbstractVector, ::AbstractVector)::Number
    dotu(::Self, ::AbstractVector, ::AbstractVector)::Number
    nrm2(::Self, ::AbstractVector)::Real
    asum(::Self, ::AbstractVector)::Real
    iamax(::Self, ::AbstractVector)::Integer
end

"""
    AbstractBLAS2 <: AbstractBLAS1

Supertype of Level-2 backends (matrix-vector); an L2 backend is also an L1 backend. Concrete
backends provide `gemv!` and `ger!` in addition to the Level-1 ops.
"""
abstract type AbstractBLAS2 <: AbstractBLAS1 end

function gemv! end  # y := β·y + α·op(A)·x
function ger! end   # A := α·x·yᵀ + A  (geru / gerc)
function symv! end  # y := α·A·x + β·y, A symmetric
function hemv! end  # y := α·A·x + β·y, A Hermitian
function trmv! end  # x := op(A)·x, A triangular
function trsv! end  # x := op(A)⁻¹·x, A triangular (solve)
# Packed storage (AP::AbstractVector) and band storage (AB::AbstractMatrix) L2 variants.
function spmv! end  # y := α·A·x + β·y, A symmetric packed
function hpmv! end  # y := α·A·x + β·y, A Hermitian packed
function tpmv! end  # x := op(A)·x, A triangular packed
function tpsv! end  # x := op(A)⁻¹·x, A triangular packed (solve)
function gbmv! end  # y := α·op(A)·x + β·y, A general banded
function sbmv! end  # y := α·A·x + β·y, A symmetric banded
function hbmv! end  # y := α·A·x + β·y, A Hermitian banded
function tbmv! end  # x := op(A)·x, A triangular banded
function tbsv! end  # x := op(A)⁻¹·x, A triangular banded (solve)
function spr! end   # A := α·x·xᵀ + A, A symmetric packed (rank-1)
function spr2! end  # A := α·x·yᵀ + α·y·xᵀ + A, A symmetric packed (rank-2)
function hpr! end   # A := α·x·xᴴ + A, A Hermitian packed (rank-1, α real)
function hpr2! end  # A := α·x·yᴴ + ᾱ·y·xᴴ + A, A Hermitian packed (rank-2)

# Level-2 is also a *strict* contract: matrix-vector kernels must be type-stable and
# allocation-free. Verified on the dense hot paths by `@verify_strict SIMDBackend` (backend.jl).
@strict_contract AbstractBLAS2 begin
    gemv!(::Self, ::AbstractVector, ::AbstractMatrix, ::AbstractVector)::AbstractVector
    ger!(::Self, ::Number, ::AbstractVector, ::AbstractVector, ::AbstractMatrix)::AbstractMatrix
    symv!(::Self, ::AbstractVector, ::AbstractMatrix, ::AbstractVector)::AbstractVector
    hemv!(::Self, ::AbstractVector, ::AbstractMatrix, ::AbstractVector)::AbstractVector
    trmv!(::Self, ::AbstractMatrix, ::AbstractVector)::AbstractVector
    trsv!(::Self, ::AbstractMatrix, ::AbstractVector)::AbstractVector
    spmv!(::Self, ::AbstractVector, ::AbstractVector, ::AbstractVector)::AbstractVector
    hpmv!(::Self, ::AbstractVector, ::AbstractVector, ::AbstractVector)::AbstractVector
    tpmv!(::Self, ::AbstractVector, ::AbstractVector)::AbstractVector
    tpsv!(::Self, ::AbstractVector, ::AbstractVector)::AbstractVector
    gbmv!(::Self, ::AbstractVector, ::AbstractMatrix, ::AbstractVector, ::Integer, ::Integer, ::Integer)::AbstractVector
    sbmv!(::Self, ::AbstractVector, ::AbstractMatrix, ::AbstractVector)::AbstractVector
    hbmv!(::Self, ::AbstractVector, ::AbstractMatrix, ::AbstractVector)::AbstractVector
    tbmv!(::Self, ::AbstractMatrix, ::AbstractVector)::AbstractVector
    tbsv!(::Self, ::AbstractMatrix, ::AbstractVector)::AbstractVector
    spr!(::Self, ::Number, ::AbstractVector, ::AbstractVector)::AbstractVector
    spr2!(::Self, ::Number, ::AbstractVector, ::AbstractVector, ::AbstractVector)::AbstractVector
    hpr!(::Self, ::Number, ::AbstractVector, ::AbstractVector)::AbstractVector
    hpr2!(::Self, ::Number, ::AbstractVector, ::AbstractVector, ::AbstractVector)::AbstractVector
end

"""
    AbstractBLAS3 <: AbstractBLAS2

Supertype of Level-3 backends (matrix-matrix); an L3 backend is also an L2 (and L1) backend. Concrete
backends provide the matrix-matrix set below in addition to the Level-1/2 ops. The `@contract` is the
single discoverable spec of what a swappable L3 backend must implement — a second backend (reference,
GPU, …) that omits any op fails the precompile-time check. The high-level `op!(::Backend, …)` methods
dispatch on the backend; the bare `op!(C, A, B; …)` entry points (gemm.jl / level3.jl) are the
default-backend fast paths (kept backend-free so the hot L3 kernels take no extra dispatch).
"""
abstract type AbstractBLAS3 <: AbstractBLAS2 end

function gemm! end   # C := β·C + α·op(A)·op(B)
function symm! end   # C := β·C + α·A·B / α·B·A,  A symmetric
function hemm! end   # C := β·C + α·A·B / α·B·A,  A Hermitian
function syrk! end   # C := β·C + α·op(A)·op(A)ᵀ, C symmetric
function herk! end   # C := β·C + α·op(A)·op(A)ᴴ, C Hermitian (α,β real)
function syr2k! end  # C := β·C + α·(op(A)·op(B)ᵀ + op(B)·op(A)ᵀ), C symmetric
function her2k! end  # C := β·C + α·op(A)·op(B)ᴴ + ᾱ·op(B)·op(A)ᴴ, C Hermitian
function trmm! end   # B := α·op(A)·B / α·B·op(A),      A triangular
function trsm! end   # B := α·op(A)⁻¹·B / α·B·op(A)⁻¹,  A triangular (solve)

# Level-3 is a strict contract: every matrix-matrix op is type-stable and allocation-free, verified by
# `@verify_strict SIMDBackend` (verify.jl). The rank-k/hemm family's divide-and-conquer drivers were
# refactored to carry integer offsets into the original arrays (not fresh sub-block SubArrays, which
# are non-isbits and heap-box when passed to the non-inlined recursive call — the sub-block views are
# also built per concrete type, never as a Union, so they stay stack-allocated). All nine now gate 0-alloc.
@strict_contract AbstractBLAS3 begin
    gemm!(::Self, ::AbstractMatrix, ::AbstractMatrix, ::AbstractMatrix)::AbstractMatrix
    symm!(::Self, ::AbstractMatrix, ::AbstractMatrix, ::AbstractMatrix)::AbstractMatrix
    hemm!(::Self, ::AbstractMatrix, ::AbstractMatrix, ::AbstractMatrix)::AbstractMatrix
    syrk!(::Self, ::AbstractMatrix, ::AbstractMatrix)::AbstractMatrix
    herk!(::Self, ::AbstractMatrix, ::AbstractMatrix)::AbstractMatrix
    syr2k!(::Self, ::AbstractMatrix, ::AbstractMatrix, ::AbstractMatrix)::AbstractMatrix
    her2k!(::Self, ::AbstractMatrix, ::AbstractMatrix, ::AbstractMatrix)::AbstractMatrix
    trmm!(::Self, ::AbstractMatrix, ::AbstractMatrix)::AbstractMatrix
    trsm!(::Self, ::AbstractMatrix, ::AbstractMatrix)::AbstractMatrix
end

"""
    AbstractLAPACK <: AbstractBLAS3

Supertype of LAPACK backends (dense factorizations); a LAPACK backend is also an L3/L2/L1 backend
(it builds the factorizations on the gated Level-3 kernels). Concrete backends provide the
factorization set below. As with the BLAS levels this `@contract` is the single discoverable spec
of a swappable LAPACK backend — a second backend (reference, GPU, …) that omits any factorization
fails the precompile-time check. The high-level `fac!(::Backend, …)` methods dispatch on the
backend; the bare `fac!(A; …)` entry points (lapack.jl / lu.jl / qr.jl / svd.jl) are the
default-backend implementations. Factorizations that return multiple outputs (pivots, τ, U/S/Vᵀ)
are typed `::Tuple`; potrf! overwrites-and-returns its matrix.
"""
abstract type AbstractLAPACK <: AbstractBLAS3 end

function potrf! end  # Cholesky:  A = L·Lᴴ (or Uᴴ·U); overwrites the stored triangle
function getrf! end  # LU w/ partial pivoting: P·A = L·U → (A, ipiv, info)
function geqrf! end  # QR (Householder): A = Q·R → (A, tau)
function gesvd! end  # SVD: A = U·Σ·Vᵀ → (U, S, Vᵀ)

# LAPACK is a strict contract too. `@verify_strict SIMDBackend` (verify.jl) enforces the
# type-stable + allocation-free guarantee on potrf!, which is 0-alloc in steady state — it factors
# through its own pointer-based `_syrk_lower_f64!`/trsm kernels (lapack.jl), not the public syrk!
# with the boxing recursion. The other three are NOT yet in the strict list: getrf!/geqrf! allocate
# their pivot/τ workspace (~1 KB) and gesvd! allocates the U/S/Vᵀ outputs — inherent to their result
# shape, not a fixable boxing bug. They stay interface-verified; strict-verifying them would need an
# in-place / pre-allocated-workspace API variant first.
@strict_contract AbstractLAPACK begin
    potrf!(::Self, ::AbstractMatrix)::AbstractMatrix
    getrf!(::Self, ::AbstractMatrix)::Tuple
    geqrf!(::Self, ::AbstractMatrix)::Tuple
    gesvd!(::Self, ::AbstractMatrix)::Tuple
end
