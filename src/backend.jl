# The M1 backend. High-level `AbstractVector` operations satisfying the `AbstractBLAS1` contract,
# delegating to the shared low-level kernels (level1.jl). Native arrays are indexed *logically*
# (increment 1 over 1:n); a dense real vector still reaches the SIMD fast path through `pointer`,
# while strided/complex/AD-element vectors take the generic scalar loop. Return-type annotations
# are explicit so inference matches the contract (zero-cost typeasserts).

struct SIMDBackend <: AbstractLAPACK end

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
# packed rank updates: A := α·x·xᵀ(+…) + A (spr/spr2 symmetric) / x·xᴴ (hpr/hpr2 Hermitian)
@inline function spr!(::SIMDBackend, alpha::Number, x::AbstractVector, AP::AbstractVector;
        uplo::Char = 'U')::AbstractVector
    n = length(x); _pkvec_dims(AP, x, n, "spr!"); _spr!(uplo == 'U', n, alpha, x, 1, AP); return AP
end
@inline function spr2!(::SIMDBackend, alpha::Number, x::AbstractVector, y::AbstractVector,
        AP::AbstractVector; uplo::Char = 'U')::AbstractVector
    n = length(x); _pkvec_dims(AP, x, n, "spr2!")
    length(y) == n || throw(DimensionMismatch("spr2!: length(y)=$(length(y)) ≠ n=$n"))
    _spr2!(uplo == 'U', n, alpha, x, 1, y, 1, AP); return AP
end
@inline function hpr!(::SIMDBackend, alpha::Number, x::AbstractVector, AP::AbstractVector;
        uplo::Char = 'U')::AbstractVector
    n = length(x); _pkvec_dims(AP, x, n, "hpr!"); _hpr!(uplo == 'U', n, alpha, x, 1, AP); return AP
end
@inline function hpr2!(::SIMDBackend, alpha::Number, x::AbstractVector, y::AbstractVector,
        AP::AbstractVector; uplo::Char = 'U')::AbstractVector
    n = length(x); _pkvec_dims(AP, x, n, "hpr2!")
    length(y) == n || throw(DimensionMismatch("hpr2!: length(y)=$(length(y)) ≠ n=$n"))
    _hpr2!(uplo == 'U', n, alpha, x, 1, y, 1, AP); return AP
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

# ── Level-3 backend surface (AbstractBLAS3 contract). The bare `op!(C, A, B; …)` entry points in
# gemm.jl / level3.jl ARE the SIMD implementations and the hot path (internal L3/LAPACK callers use
# them directly, taking no extra dispatch). These thin backend-dispatched wrappers exist so a second
# backend can provide its own L3 via `op!(::OtherBackend, …)` and the contract enforces completeness.
# @inline + a const singleton backend ⇒ compile-time resolved, zero-cost. (ponytail: the bare entry is
# not yet routed through DEFAULT_BACKEND — add that forwarding when a real default-swap need appears.)
@inline gemm!(::SIMDBackend, C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix; kw...)::AbstractMatrix = gemm!(C, A, B; kw...)
@inline symm!(::SIMDBackend, C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix; kw...)::AbstractMatrix = symm!(C, A, B; kw...)
@inline hemm!(::SIMDBackend, C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix; kw...)::AbstractMatrix = hemm!(C, A, B; kw...)
@inline syrk!(::SIMDBackend, C::AbstractMatrix, A::AbstractMatrix; kw...)::AbstractMatrix = syrk!(C, A; kw...)
@inline herk!(::SIMDBackend, C::AbstractMatrix, A::AbstractMatrix; kw...)::AbstractMatrix = herk!(C, A; kw...)
@inline syr2k!(::SIMDBackend, C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix; kw...)::AbstractMatrix = syr2k!(C, A, B; kw...)
@inline her2k!(::SIMDBackend, C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix; kw...)::AbstractMatrix = her2k!(C, A, B; kw...)
@inline trmm!(::SIMDBackend, B::AbstractMatrix, A::AbstractMatrix; kw...)::AbstractMatrix = trmm!(B, A; kw...)
@inline trsm!(::SIMDBackend, B::AbstractMatrix, A::AbstractMatrix; kw...)::AbstractMatrix = trsm!(B, A; kw...)

# ── LAPACK backend surface (AbstractLAPACK contract). Same thin-wrapper form as L3: the bare
# `fac!(A; …)` entry points (lapack.jl/lu.jl/qr.jl/svd.jl) are the implementations; these dispatch a
# second backend's factorizations and let the contract enforce completeness. Zero-cost (inlined).
@inline potrf!(::SIMDBackend, A::AbstractMatrix; kw...)::AbstractMatrix = potrf!(A; kw...)
@inline getrf!(::SIMDBackend, A::AbstractMatrix; kw...)::Tuple = getrf!(A; kw...)
@inline geqrf!(::SIMDBackend, A::AbstractMatrix; kw...)::Tuple = geqrf!(A; kw...)
@inline gesvd!(::SIMDBackend, A::AbstractMatrix; kw...)::Tuple = gesvd!(A; kw...)

# Assert at precompile time that SIMDBackend satisfies every contract in its supertype chain.
# @verify_strict is the single backend verifier: it runs TypeContracts.@verify SIMDBackend — method
# existence + declared return types over the whole chain (AbstractBLAS1 → BLAS2 → BLAS3 → LAPACK) —
# AND @strict on each representative L1 call (the SIMD real path and the generic complex path), since
# AbstractBLAS1 is a @strict_contract (contracts.jl) demanding type-stability + no allocation. The
# @strict calls self-gate on the `checks_enabled` preference; the main package ships fast mode
# (runtime @allocated / @inferred, no AllocCheck/JET dep) so they fire at PureBLAS's OWN precompile.
# The guard skips the whole thing under full-mode environments (e.g. the test project) where @strict
# demands the AllocCheck/JET backend not loaded during PureBLAS's own precompile — there the test
# suite's strictmode dogfood runs the interface + deep static proof at test runtime with the backend
# present. (Trim-safety of the C-ABI entries is covered exhaustively by TrimCheck.@validate.) Values
# in a `let` — only their types matter, so `ones` (no Random dep).
if StrictMode.analysis_mode() === :fast || StrictMode.backend_available()
    let bk = DEFAULT_BACKEND, n = 1000,
        xd = ones(n), yd = ones(n), xz = ones(ComplexF64, n), yz = ones(ComplexF64, n)
        @verify_strict SIMDBackend begin
            axpy!(bk, yd, 2.0, xd)
            scal!(bk, 2.0, xd)
            blascopy!(bk, yd, xd)
            swap!(bk, xd, yd)
            dot(bk, xd, yd)
            dotu(bk, xd, yd)
            nrm2(bk, xd)
            asum(bk, xd)
            iamax(bk, xd)
            axpy!(bk, yz, 2.0 + 1.0im, xz)   # generic complex path
            dot(bk, xz, yz)
            nrm2(bk, xz)
        end
    end
end
