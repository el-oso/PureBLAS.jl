# Compile-time interface contract for BLAS Level-1 backends (TypeContracts.jl).
#
# A *backend* is the swappable unit of the Pure ecosystem: the M1 `SIMDBackend` (backend.jl), a
# future reference/GPU backend, etc. Each must provide the same Level-1 operation set so callers
# (the native API in native.jl, and downstream packages) depend on the interface, not the struct.
# `@contract` checks method existence + inferred return types at PRECOMPILE time and is eliminated
# by the trimmer — zero runtime cost (mirrors PureFFT's `AbstractFFTPlan` contract). Implementing
# methods carry explicit concrete return-type annotations so inference matches the contract.

using TypeContracts

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

@contract AbstractBLAS1 begin
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

@contract AbstractBLAS2 begin
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
