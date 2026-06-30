# The M1 backend. High-level `AbstractVector` operations satisfying the `AbstractBLAS1` contract,
# delegating to the shared low-level kernels (level1.jl). Native arrays are indexed *logically*
# (increment 1 over 1:n); a dense real vector still reaches the SIMD fast path through `pointer`,
# while strided/complex/AD-element vectors take the generic scalar loop. Return-type annotations
# are explicit so inference matches the contract (zero-cost typeasserts).

struct SIMDBackend <: AbstractBLAS2 end

"""Default Level-1 backend used by the bare native API in native.jl."""
const DEFAULT_BACKEND = SIMDBackend()

@inline function _eqlen(x, y)
    length(x) == length(y) ||
        throw(DimensionMismatch("PureBLAS: length(x)=$(length(x)) ≠ length(y)=$(length(y))"))
    return length(x)
end

function axpy!(::SIMDBackend, y::AbstractVector, a::Number, x::AbstractVector)::AbstractVector
    _axpy!(_eqlen(x, y), a, x, 1, y, 1)
    return y
end

function scal!(::SIMDBackend, a::Number, x::AbstractVector)::AbstractVector
    _scal!(length(x), a, x, 1)
    return x
end

# Copy src `x` into dest `y` (`y .= x`).
function blascopy!(::SIMDBackend, y::AbstractVector, x::AbstractVector)::AbstractVector
    _copy!(_eqlen(x, y), x, 1, y, 1)
    return y
end

function swap!(::SIMDBackend, x::AbstractVector, y::AbstractVector)::Nothing
    _swap!(_eqlen(x, y), x, 1, y, 1)
    return nothing
end

# Conjugated inner product conj(x)·y (matches LinearAlgebra.dot for complex).
function dot(::SIMDBackend, x::AbstractVector, y::AbstractVector)::Number
    return _dotc(_eqlen(x, y), x, 1, y, 1)
end

# Unconjugated inner product x·y (BLAS ?dotu).
function dotu(::SIMDBackend, x::AbstractVector, y::AbstractVector)::Number
    return _dotu(_eqlen(x, y), x, 1, y, 1)
end

function nrm2(::SIMDBackend, x::AbstractVector)::Real
    return _nrm2(length(x), x, 1)
end

function asum(::SIMDBackend, x::AbstractVector)::Real
    return _asum(length(x), x, 1)
end

function iamax(::SIMDBackend, x::AbstractVector)::Integer
    return _iamax(length(x), x, 1)
end

# ── Level 2 ──────────────────────────────────────────────────────────────────────────────────
# y := β·op(A)·x·α + β·y form; trans ∈ {'N','T','C'}. Native arrays index logically (inc 1).
@inline function gemv!(::SIMDBackend, y::AbstractVector, A::AbstractMatrix, x::AbstractVector;
        alpha = one(eltype(A)), beta = zero(eltype(A)), trans::Char = 'N')::AbstractVector
    tA = trans != 'N'; cj = trans == 'C'
    m, n = size(A)
    if tA
        length(x) == m || throw(DimensionMismatch("gemv!('$trans'): length(x)=$(length(x)) ≠ size(A,1)=$m"))
        length(y) == n || throw(DimensionMismatch("gemv!('$trans'): length(y)=$(length(y)) ≠ size(A,2)=$n"))
    else
        length(x) == n || throw(DimensionMismatch("gemv!: length(x)=$(length(x)) ≠ size(A,2)=$n"))
        length(y) == m || throw(DimensionMismatch("gemv!: length(y)=$(length(y)) ≠ size(A,1)=$m"))
    end
    _gemv!(tA, cj, m, n, alpha, A, x, 1, beta, y, 1)
    return y
end

# A := α·x·yᵀ + A  (conj=true ⇒ α·x·yᴴ).
@inline function ger!(::SIMDBackend, alpha::Number, x::AbstractVector, y::AbstractVector,
        A::AbstractMatrix; conj::Bool = false)::AbstractMatrix
    m, n = size(A)
    length(x) == m || throw(DimensionMismatch("ger!: length(x)=$(length(x)) ≠ size(A,1)=$m"))
    length(y) == n || throw(DimensionMismatch("ger!: length(y)=$(length(y)) ≠ size(A,2)=$n"))
    _ger!(conj, m, n, alpha, x, 1, y, 1, A)
    return A
end

@inline function _symhemv_dims(A, x, y, op)
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("$op: A is $(size(A,1))×$(size(A,2)), not square"))
    length(x) == n || throw(DimensionMismatch("$op: length(x)=$(length(x)) ≠ size(A)=$n"))
    length(y) == n || throw(DimensionMismatch("$op: length(y)=$(length(y)) ≠ size(A)=$n"))
    return n
end

# y := α·A·x + β·y, A symmetric; uplo ∈ {'U','L'} selects the stored triangle.
@inline function symv!(::SIMDBackend, y::AbstractVector, A::AbstractMatrix, x::AbstractVector;
        uplo::Char = 'U', alpha = one(eltype(A)), beta = zero(eltype(A)))::AbstractVector
    n = _symhemv_dims(A, x, y, "symv!")
    _symv!(uplo == 'U', n, alpha, A, x, 1, beta, y, 1)
    return y
end

# y := α·A·x + β·y, A Hermitian.
@inline function hemv!(::SIMDBackend, y::AbstractVector, A::AbstractMatrix, x::AbstractVector;
        uplo::Char = 'U', alpha = one(eltype(A)), beta = zero(eltype(A)))::AbstractVector
    n = _symhemv_dims(A, x, y, "hemv!")
    _hemv!(uplo == 'U', n, alpha, A, x, 1, beta, y, 1)
    return y
end

@inline function _tri_dims(A, x, op)
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("$op: A is $(size(A,1))×$(size(A,2)), not square"))
    length(x) == n || throw(DimensionMismatch("$op: length(x)=$(length(x)) ≠ size(A)=$n"))
    return n
end

# x := op(A)·x, A triangular. uplo∈{'U','L'}, trans∈{'N','T','C'}, diag∈{'N','U'} (U⇒unit diagonal).
@inline function trmv!(::SIMDBackend, A::AbstractMatrix, x::AbstractVector;
        uplo::Char = 'U', trans::Char = 'N', diag::Char = 'N')::AbstractVector
    n = _tri_dims(A, x, "trmv!")
    _trmv!(uplo == 'U', trans != 'N', trans == 'C', diag == 'U', n, A, x, 1)
    return x
end

# x := op(A)⁻¹·x, A triangular (solve).
@inline function trsv!(::SIMDBackend, A::AbstractMatrix, x::AbstractVector;
        uplo::Char = 'U', trans::Char = 'N', diag::Char = 'N')::AbstractVector
    n = _tri_dims(A, x, "trsv!")
    _trsv!(uplo == 'U', trans != 'N', trans == 'C', diag == 'U', n, A, x, 1)
    return x
end

# ── Level 2 packed storage (AP::AbstractVector) ────────────────────────────────────────────────
@inline function _pkvec_dims(AP, x, n, op)
    length(x) == n || throw(DimensionMismatch("$op: length(x)=$(length(x)) ≠ n=$n"))
    length(AP) >= (n * (n + 1)) ÷ 2 || throw(DimensionMismatch("$op: length(AP)=$(length(AP)) < n(n+1)/2 for n=$n"))
    return n
end

@inline function spmv!(::SIMDBackend, y::AbstractVector, AP::AbstractVector, x::AbstractVector;
        uplo::Char = 'U', alpha = one(eltype(AP)), beta = zero(eltype(AP)))::AbstractVector
    n = length(y); _pkvec_dims(AP, x, n, "spmv!")
    _spmv!(uplo == 'U', n, alpha, AP, x, 1, beta, y, 1)
    return y
end
@inline function hpmv!(::SIMDBackend, y::AbstractVector, AP::AbstractVector, x::AbstractVector;
        uplo::Char = 'U', alpha = one(eltype(AP)), beta = zero(eltype(AP)))::AbstractVector
    n = length(y); _pkvec_dims(AP, x, n, "hpmv!")
    _hpmv!(uplo == 'U', n, alpha, AP, x, 1, beta, y, 1)
    return y
end
@inline function tpmv!(::SIMDBackend, AP::AbstractVector, x::AbstractVector;
        uplo::Char = 'U', trans::Char = 'N', diag::Char = 'N')::AbstractVector
    n = length(x); _pkvec_dims(AP, x, n, "tpmv!")
    _tpmv!(uplo == 'U', trans != 'N', trans == 'C', diag == 'U', n, AP, x, 1)
    return x
end
@inline function tpsv!(::SIMDBackend, AP::AbstractVector, x::AbstractVector;
        uplo::Char = 'U', trans::Char = 'N', diag::Char = 'N')::AbstractVector
    n = length(x); _pkvec_dims(AP, x, n, "tpsv!")
    _tpsv!(uplo == 'U', trans != 'N', trans == 'C', diag == 'U', n, AP, x, 1)
    return x
end

# ── Level 2 band storage (AB::AbstractMatrix, leading dim = #band rows) ─────────────────────────
@inline function gbmv!(::SIMDBackend, y::AbstractVector, AB::AbstractMatrix, x::AbstractVector,
        m::Integer, kl::Integer, ku::Integer; trans::Char = 'N',
        alpha = one(eltype(AB)), beta = zero(eltype(AB)))::AbstractVector
    n = size(AB, 2); tA = trans != 'N'
    size(AB, 1) >= kl + ku + 1 || throw(DimensionMismatch("gbmv!: size(AB,1)=$(size(AB,1)) < kl+ku+1=$(kl+ku+1)"))
    length(x) == (tA ? m : n) || throw(DimensionMismatch("gbmv!('$trans'): length(x)=$(length(x)) ≠ $(tA ? m : n)"))
    length(y) == (tA ? n : m) || throw(DimensionMismatch("gbmv!('$trans'): length(y)=$(length(y)) ≠ $(tA ? n : m)"))
    _gbmv!(tA, trans == 'C', m, n, kl, ku, alpha, AB, x, 1, beta, y, 1)
    return y
end
@inline function _sbvec_dims(AB, x, y, op)
    n = size(AB, 2)
    length(x) == n || throw(DimensionMismatch("$op: length(x)=$(length(x)) ≠ n=$n"))
    length(y) == n || throw(DimensionMismatch("$op: length(y)=$(length(y)) ≠ n=$n"))
    return n, size(AB, 1) - 1
end
@inline function sbmv!(::SIMDBackend, y::AbstractVector, AB::AbstractMatrix, x::AbstractVector;
        uplo::Char = 'U', alpha = one(eltype(AB)), beta = zero(eltype(AB)))::AbstractVector
    n, k = _sbvec_dims(AB, x, y, "sbmv!")
    _sbmv!(uplo == 'U', n, k, alpha, AB, x, 1, beta, y, 1)
    return y
end
@inline function hbmv!(::SIMDBackend, y::AbstractVector, AB::AbstractMatrix, x::AbstractVector;
        uplo::Char = 'U', alpha = one(eltype(AB)), beta = zero(eltype(AB)))::AbstractVector
    n, k = _sbvec_dims(AB, x, y, "hbmv!")
    _hbmv!(uplo == 'U', n, k, alpha, AB, x, 1, beta, y, 1)
    return y
end
@inline function tbmv!(::SIMDBackend, AB::AbstractMatrix, x::AbstractVector;
        uplo::Char = 'U', trans::Char = 'N', diag::Char = 'N')::AbstractVector
    n = size(AB, 2); length(x) == n || throw(DimensionMismatch("tbmv!: length(x)=$(length(x)) ≠ n=$n"))
    _tbmv!(uplo == 'U', trans != 'N', trans == 'C', diag == 'U', n, size(AB, 1) - 1, AB, x, 1)
    return x
end
@inline function tbsv!(::SIMDBackend, AB::AbstractMatrix, x::AbstractVector;
        uplo::Char = 'U', trans::Char = 'N', diag::Char = 'N')::AbstractVector
    n = size(AB, 2); length(x) == n || throw(DimensionMismatch("tbsv!: length(x)=$(length(x)) ≠ n=$n"))
    _tbsv!(uplo == 'U', trans != 'N', trans == 'C', diag == 'U', n, size(AB, 1) - 1, AB, x, 1)
    return x
end
