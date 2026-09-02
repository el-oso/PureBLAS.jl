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
        throw(DimensionMismatch(lazy"PureBLAS: length(x)=$(length(x)) ≠ length(y)=$(length(y))"))
    return length(x)
end

@inline function axpy!(::SIMDBackend, y::AbstractVector, a::Number, x::AbstractVector)::AbstractVector
    _axpy!(_eqlen(x, y), a, x, 1, y, 1)
    return y
end

@inline function scal!(::SIMDBackend, a::Number, x::AbstractVector)::AbstractVector
    _scal!(length(x), a, x, 1)
    return x
end

# Copy src `x` into dest `y` (`y .= x`).
@inline function blascopy!(::SIMDBackend, y::AbstractVector, x::AbstractVector)::AbstractVector
    _copy!(_eqlen(x, y), x, 1, y, 1)
    return y
end

@inline function swap!(::SIMDBackend, x::AbstractVector, y::AbstractVector)::Nothing
    _swap!(_eqlen(x, y), x, 1, y, 1)
    return nothing
end

# Conjugated inner product conj(x)·y (matches LinearAlgebra.dot for complex).
@inline function dot(::SIMDBackend, x::AbstractVector, y::AbstractVector)::Number
    return _dotc(_eqlen(x, y), x, 1, y, 1)
end

# Unconjugated inner product x·y (BLAS ?dotu).
@inline function dotu(::SIMDBackend, x::AbstractVector, y::AbstractVector)::Number
    return _dotu(_eqlen(x, y), x, 1, y, 1)
end

@inline function nrm2(::SIMDBackend, x::AbstractVector)::Real
    return _nrm2(length(x), x, 1)
end

@inline function asum(::SIMDBackend, x::AbstractVector)::Real
    return _asum(length(x), x, 1)
end

@inline function iamax(::SIMDBackend, x::AbstractVector)::Integer
    return _iamax(length(x), x, 1)
end

# ── Level 2 ──────────────────────────────────────────────────────────────────────────────────
# y := β·op(A)·x·α + β·y form; trans ∈ {'N','T','C'}. Native arrays index logically (inc 1).
@inline function gemv!(
        ::SIMDBackend, y::AbstractVector, A::AbstractMatrix, x::AbstractVector;
        alpha = one(eltype(A)), beta = zero(eltype(A)), trans::Char = 'N'
    )::AbstractVector
    tA = trans != 'N'; cj = trans == 'C'
    m, n = size(A)
    if tA
        length(x) == m || throw(DimensionMismatch(lazy"gemv!('$trans'): length(x)=$(length(x)) ≠ size(A,1)=$m"))
        length(y) == n || throw(DimensionMismatch(lazy"gemv!('$trans'): length(y)=$(length(y)) ≠ size(A,2)=$n"))
    else
        length(x) == n || throw(DimensionMismatch(lazy"gemv!: length(x)=$(length(x)) ≠ size(A,2)=$n"))
        length(y) == m || throw(DimensionMismatch(lazy"gemv!: length(y)=$(length(y)) ≠ size(A,1)=$m"))
    end
    _gemv!(tA, cj, m, n, alpha, A, x, 1, beta, y, 1)
    return y
end

# A := α·x·yᵀ + A  (conj=true ⇒ α·x·yᴴ).
@inline function ger!(
        ::SIMDBackend, alpha::Number, x::AbstractVector, y::AbstractVector,
        A::AbstractMatrix; conj::Bool = false
    )::AbstractMatrix
    m, n = size(A)
    length(x) == m || throw(DimensionMismatch(lazy"ger!: length(x)=$(length(x)) ≠ size(A,1)=$m"))
    length(y) == n || throw(DimensionMismatch(lazy"ger!: length(y)=$(length(y)) ≠ size(A,2)=$n"))
    _ger!(conj, m, n, alpha, x, 1, y, 1, A)
    return A
end

@inline function _symhemv_dims(A, x, y, op)
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch(lazy"$op: A is $(size(A, 1))×$(size(A, 2)), not square"))
    length(x) == n || throw(DimensionMismatch(lazy"$op: length(x)=$(length(x)) ≠ size(A)=$n"))
    length(y) == n || throw(DimensionMismatch(lazy"$op: length(y)=$(length(y)) ≠ size(A)=$n"))
    return n
end

# y := α·A·x + β·y, A symmetric; uplo ∈ {'U','L'} selects the stored triangle.
@inline function symv!(
        ::SIMDBackend, y::AbstractVector, A::AbstractMatrix, x::AbstractVector;
        uplo::Char = 'U', alpha = one(eltype(A)), beta = zero(eltype(A))
    )::AbstractVector
    n = _symhemv_dims(A, x, y, "symv!")
    _symv!(uplo == 'U', n, alpha, A, x, 1, beta, y, 1)
    return y
end

# y := α·A·x + β·y, A Hermitian.
@inline function hemv!(
        ::SIMDBackend, y::AbstractVector, A::AbstractMatrix, x::AbstractVector;
        uplo::Char = 'U', alpha = one(eltype(A)), beta = zero(eltype(A))
    )::AbstractVector
    n = _symhemv_dims(A, x, y, "hemv!")
    _hemv!(uplo == 'U', n, alpha, A, x, 1, beta, y, 1)
    return y
end

# The two throws are `lazy"…"` behind `@noinline`, the same idiom as `_throw_square` (core.jl), and it
# is a TRIM requirement (req#4), not style. An eager `"$op: …$(size(A,2))…"` builds the String at the
# throw site via `Base.print_to_string`, whose heterogeneous argument loop lowers to
# `_str_sizehint(φ ()::Any)` / `print(::IOBuffer, φ ()::Any)` — dynamic calls the `juliac --trim=safe`
# verifier rejects. Because `trsv!` and `trmv!` both route through here, that one eager interpolation
# made every caller trim-incompatible: it is what failed `gelsy!` in the LAPACK trim dogfood, reported
# against `gelsy.jl:255` (a `trsm!` call) -> `level3.jl:4228` (a `trsv!` call) -> here.
# `LazyString` stores the parts and formats only if the message is actually rendered.
@noinline _throw_tri_square(op, m::Int, k::Int) =
    throw(DimensionMismatch(lazy"$op: A is $m×$k, not square"))
@noinline _throw_tri_len(op, lx::Int, n::Int) =
    throw(DimensionMismatch(lazy"$op: length(x)=$lx ≠ size(A)=$n"))

@inline function _tri_dims(A, x, op)
    n = size(A, 1)
    size(A, 2) == n || _throw_tri_square(op, size(A, 1), size(A, 2))
    length(x) == n || _throw_tri_len(op, length(x), n)
    return n
end

# x := op(A)·x, A triangular. uplo∈{'U','L'}, trans∈{'N','T','C'}, diag∈{'N','U'} (U⇒unit diagonal).
@inline function trmv!(
        ::SIMDBackend, A::AbstractMatrix, x::AbstractVector;
        uplo::Char = 'U', trans::Char = 'N', diag::Char = 'N'
    )::AbstractVector
    n = _tri_dims(A, x, "trmv!")
    _trmv!(uplo == 'U', trans != 'N', trans == 'C', diag == 'U', n, A, x, 1)
    return x
end

# x := op(A)⁻¹·x, A triangular (solve).
@inline function trsv!(
        ::SIMDBackend, A::AbstractMatrix, x::AbstractVector;
        uplo::Char = 'U', trans::Char = 'N', diag::Char = 'N'
    )::AbstractVector
    n = _tri_dims(A, x, "trsv!")
    _trsv!(uplo == 'U', trans != 'N', trans == 'C', diag == 'U', n, A, x, 1)
    return x
end

# ── Level 2 packed storage (AP::AbstractVector) ────────────────────────────────────────────────
@inline function _pkvec_dims(AP, x, n, op)
    length(x) == n || throw(DimensionMismatch(lazy"$op: length(x)=$(length(x)) ≠ n=$n"))
    length(AP) >= (n * (n + 1)) ÷ 2 || throw(DimensionMismatch(lazy"$op: length(AP)=$(length(AP)) < n(n+1)/2 for n=$n"))
    return n
end

@inline function spmv!(
        ::SIMDBackend, y::AbstractVector, AP::AbstractVector, x::AbstractVector;
        uplo::Char = 'U', alpha = one(eltype(AP)), beta = zero(eltype(AP))
    )::AbstractVector
    n = length(y); _pkvec_dims(AP, x, n, "spmv!")
    _spmv!(uplo == 'U', n, alpha, AP, x, 1, beta, y, 1)
    return y
end
@inline function hpmv!(
        ::SIMDBackend, y::AbstractVector, AP::AbstractVector, x::AbstractVector;
        uplo::Char = 'U', alpha = one(eltype(AP)), beta = zero(eltype(AP))
    )::AbstractVector
    n = length(y); _pkvec_dims(AP, x, n, "hpmv!")
    _hpmv!(uplo == 'U', n, alpha, AP, x, 1, beta, y, 1)
    return y
end
@inline function tpmv!(
        ::SIMDBackend, AP::AbstractVector, x::AbstractVector;
        uplo::Char = 'U', trans::Char = 'N', diag::Char = 'N'
    )::AbstractVector
    n = length(x); _pkvec_dims(AP, x, n, "tpmv!")
    _tpmv!(uplo == 'U', trans != 'N', trans == 'C', diag == 'U', n, AP, x, 1)
    return x
end
@inline function tpsv!(
        ::SIMDBackend, AP::AbstractVector, x::AbstractVector;
        uplo::Char = 'U', trans::Char = 'N', diag::Char = 'N'
    )::AbstractVector
    n = length(x); _pkvec_dims(AP, x, n, "tpsv!")
    _tpsv!(uplo == 'U', trans != 'N', trans == 'C', diag == 'U', n, AP, x, 1)
    return x
end
# packed rank updates: A := α·x·xᵀ(+…) + A (spr/spr2 symmetric) / x·xᴴ (hpr/hpr2 Hermitian)
@inline function spr!(
        ::SIMDBackend, alpha::Number, x::AbstractVector, AP::AbstractVector;
        uplo::Char = 'U'
    )::AbstractVector
    n = length(x); _pkvec_dims(AP, x, n, "spr!"); _spr!(uplo == 'U', n, alpha, x, 1, AP); return AP
end
@inline function spr2!(
        ::SIMDBackend, alpha::Number, x::AbstractVector, y::AbstractVector,
        AP::AbstractVector; uplo::Char = 'U'
    )::AbstractVector
    n = length(x); _pkvec_dims(AP, x, n, "spr2!")
    length(y) == n || throw(DimensionMismatch(lazy"spr2!: length(y)=$(length(y)) ≠ n=$n"))
    _spr2!(uplo == 'U', n, alpha, x, 1, y, 1, AP); return AP
end
@inline function hpr!(
        ::SIMDBackend, alpha::Number, x::AbstractVector, AP::AbstractVector;
        uplo::Char = 'U'
    )::AbstractVector
    n = length(x); _pkvec_dims(AP, x, n, "hpr!"); _hpr!(uplo == 'U', n, alpha, x, 1, AP); return AP
end
@inline function hpr2!(
        ::SIMDBackend, alpha::Number, x::AbstractVector, y::AbstractVector,
        AP::AbstractVector; uplo::Char = 'U'
    )::AbstractVector
    n = length(x); _pkvec_dims(AP, x, n, "hpr2!")
    length(y) == n || throw(DimensionMismatch(lazy"hpr2!: length(y)=$(length(y)) ≠ n=$n"))
    _hpr2!(uplo == 'U', n, alpha, x, 1, y, 1, AP); return AP
end

# ── Level 2 band storage (AB::AbstractMatrix, leading dim = #band rows) ─────────────────────────
@inline function gbmv!(
        ::SIMDBackend, y::AbstractVector, AB::AbstractMatrix, x::AbstractVector,
        m::Integer, kl::Integer, ku::Integer; trans::Char = 'N',
        alpha = one(eltype(AB)), beta = zero(eltype(AB))
    )::AbstractVector
    n = size(AB, 2); tA = trans != 'N'
    size(AB, 1) >= kl + ku + 1 || throw(DimensionMismatch(lazy"gbmv!: size(AB,1)=$(size(AB, 1)) < kl+ku+1=$(kl + ku + 1)"))
    length(x) == (tA ? m : n) || throw(DimensionMismatch(lazy"gbmv!('$trans'): length(x)=$(length(x)) ≠ $(tA ? m : n)"))
    length(y) == (tA ? n : m) || throw(DimensionMismatch(lazy"gbmv!('$trans'): length(y)=$(length(y)) ≠ $(tA ? n : m)"))
    _gbmv!(tA, trans == 'C', m, n, kl, ku, alpha, AB, x, 1, beta, y, 1)
    return y
end
@inline function _sbvec_dims(AB, x, y, op)
    n = size(AB, 2)
    length(x) == n || throw(DimensionMismatch(lazy"$op: length(x)=$(length(x)) ≠ n=$n"))
    length(y) == n || throw(DimensionMismatch(lazy"$op: length(y)=$(length(y)) ≠ n=$n"))
    return n, size(AB, 1) - 1
end
@inline function sbmv!(
        ::SIMDBackend, y::AbstractVector, AB::AbstractMatrix, x::AbstractVector;
        uplo::Char = 'U', alpha = one(eltype(AB)), beta = zero(eltype(AB))
    )::AbstractVector
    n, k = _sbvec_dims(AB, x, y, "sbmv!")
    _sbmv!(uplo == 'U', n, k, alpha, AB, x, 1, beta, y, 1)
    return y
end
@inline function hbmv!(
        ::SIMDBackend, y::AbstractVector, AB::AbstractMatrix, x::AbstractVector;
        uplo::Char = 'U', alpha = one(eltype(AB)), beta = zero(eltype(AB))
    )::AbstractVector
    n, k = _sbvec_dims(AB, x, y, "hbmv!")
    _hbmv!(uplo == 'U', n, k, alpha, AB, x, 1, beta, y, 1)
    return y
end
@inline function tbmv!(
        ::SIMDBackend, AB::AbstractMatrix, x::AbstractVector;
        uplo::Char = 'U', trans::Char = 'N', diag::Char = 'N'
    )::AbstractVector
    n = size(AB, 2); length(x) == n || throw(DimensionMismatch(lazy"tbmv!: length(x)=$(length(x)) ≠ n=$n"))
    _tbmv!(uplo == 'U', trans != 'N', trans == 'C', diag == 'U', n, size(AB, 1) - 1, AB, x, 1)
    return x
end
@inline function tbsv!(
        ::SIMDBackend, AB::AbstractMatrix, x::AbstractVector;
        uplo::Char = 'U', trans::Char = 'N', diag::Char = 'N'
    )::AbstractVector
    n = size(AB, 2); length(x) == n || throw(DimensionMismatch(lazy"tbsv!: length(x)=$(length(x)) ≠ n=$n"))
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

# ── LAPACK solve/inverse/condition surface (the rest of the AbstractLAPACK contract). Same form as
# above. Every one is IN-PLACE, so the contract is strict (type-stable + allocation-free) — see the
# interface docstring in contracts.jl for why that is load-bearing rather than decorative. Where a
# routine has both an in-place and an allocating convenience form (gbtrf!, gesvx!), the wrapper binds
# the IN-PLACE one: the contract must not be satisfiable by a method that allocates.
@inline potrs!(::SIMDBackend, A::AbstractMatrix, B::AbstractVecOrMat; kw...)::AbstractVecOrMat = potrs!(A, B; kw...)
@inline potri!(::SIMDBackend, A::AbstractMatrix; kw...)::AbstractMatrix = potri!(A; kw...)
@inline pptrf!(::SIMDBackend, AP::AbstractVector; kw...)::AbstractVector = pptrf!(AP; kw...)
@inline pptrs!(::SIMDBackend, AP::AbstractVector, B::AbstractVecOrMat; kw...)::AbstractVecOrMat = pptrs!(AP, B; kw...)
@inline pbtrf!(::SIMDBackend, AB::AbstractMatrix; kw...)::AbstractMatrix = pbtrf!(AB; kw...)
@inline pbtrs!(::SIMDBackend, AB::AbstractMatrix, B::AbstractVecOrMat; kw...)::AbstractVecOrMat = pbtrs!(AB, B; kw...)
@inline pstrf!(::SIMDBackend, A::AbstractMatrix, piv::AbstractVector, tol::Real; kw...)::Tuple = pstrf!(A, piv, tol; kw...)
@inline pocon!(::SIMDBackend, normA::Real, A::AbstractMatrix; kw...)::Real = pocon!(normA, A; kw...)
@inline getrs!(::SIMDBackend, A::AbstractMatrix, ipiv::AbstractVector, B::AbstractVecOrMat; kw...)::AbstractVecOrMat = getrs!(A, ipiv, B; kw...)
@inline getri!(::SIMDBackend, A::AbstractMatrix, ipiv::AbstractVector)::AbstractMatrix = getri!(A, ipiv)
@inline trtrs!(::SIMDBackend, A::AbstractMatrix, B::AbstractVecOrMat; kw...)::AbstractVecOrMat = trtrs!(A, B; kw...)
@inline trtri!(::SIMDBackend, A::AbstractMatrix; kw...)::AbstractMatrix = trtri!(A; kw...)
@inline gecon!(::SIMDBackend, normA::Real, A::AbstractMatrix, ipiv::AbstractVector; kw...)::Real = gecon!(normA, A, ipiv; kw...)
@inline trcon!(::SIMDBackend, A::AbstractMatrix; kw...)::Real = trcon!(A; kw...)
@inline trrfs!(::SIMDBackend, uplo::AbstractChar, trans::AbstractChar, diag::AbstractChar, A::AbstractMatrix, B::AbstractVecOrMat, X::AbstractVecOrMat, Ferr::AbstractVector, Berr::AbstractVector)::Tuple = trrfs!(uplo, trans, diag, A, B, X, Ferr, Berr)
@inline gbtrf!(::SIMDBackend, kl::Integer, ku::Integer, m::Integer, AB::AbstractMatrix, ipiv::AbstractVector)::Tuple = gbtrf!(kl, ku, m, AB, ipiv)
@inline gbtrs!(::SIMDBackend, trans::AbstractChar, kl::Integer, ku::Integer, m::Integer, AB::AbstractMatrix, ipiv::AbstractVector, B::AbstractVecOrMat)::AbstractVecOrMat = gbtrs!(trans, kl, ku, m, AB, ipiv, B)
@inline gtsv!(::SIMDBackend, dl::AbstractVector, d::AbstractVector, du::AbstractVector, B::AbstractVecOrMat)::AbstractVecOrMat = gtsv!(dl, d, du, B)
@inline gttrf!(::SIMDBackend, dl::AbstractVector, d::AbstractVector, du::AbstractVector, du2::AbstractVector, ipiv::AbstractVector)::Tuple = gttrf!(dl, d, du, du2, ipiv)
@inline gttrs!(::SIMDBackend, trans::AbstractChar, dl::AbstractVector, d::AbstractVector, du::AbstractVector, du2::AbstractVector, ipiv::AbstractVector, B::AbstractVecOrMat)::AbstractVecOrMat = gttrs!(trans, dl, d, du, du2, ipiv, B)
@inline pttrf!(::SIMDBackend, D::AbstractVector, E::AbstractVector)::Tuple = pttrf!(D, E)
@inline pttrs!(::SIMDBackend, D::AbstractVector, E::AbstractVector, B::AbstractVecOrMat; kw...)::AbstractVecOrMat = pttrs!(D, E, B; kw...)
@inline ptsv!(::SIMDBackend, D::AbstractVector, E::AbstractVector, B::AbstractVecOrMat; kw...)::Tuple = ptsv!(D, E, B; kw...)
@inline gesvx!(::SIMDBackend, fact::Char, trans::Char, A::AbstractMatrix, AF::AbstractMatrix, ipiv::AbstractVector, equed::Char, R::AbstractVector, C::AbstractVector, B::AbstractMatrix, X::AbstractMatrix, ferr::AbstractVector, berr::AbstractVector)::Tuple = gesvx!(fact, trans, A, AF, ipiv, equed, R, C, B, X, ferr, berr)

# ── QR / LQ / QL / RZ backend surface (the rest of the AbstractLAPACK contract). Same thin-wrapper
# form. Only the 14 DISTINCT implementations get a method: `ungqr!`/`unglq!`/`ungql!`/`ungrq!` and
# `unmqr!`/`unmlq!`/`unmql!`/`unmrq!`/`unmrz!` are `const un*! = or*!` aliases — the same function
# object — so each `or*!` method below IS the `un*!` method. orgtr!/ungtr! ARE here, but only through
# their IN-PLACE (caller-supplied Q) forms; their 3-arg convenience forms allocate Q.
@inline gelqf!(::SIMDBackend, A::AbstractMatrix, tau::AbstractVector)::AbstractMatrix = gelqf!(A, tau)
@inline geqlf!(::SIMDBackend, A::AbstractMatrix, tau::AbstractVector)::AbstractMatrix = geqlf!(A, tau)
@inline gerqf!(::SIMDBackend, A::AbstractMatrix, tau::AbstractVector)::AbstractMatrix = gerqf!(A, tau)
@inline geqp3!(::SIMDBackend, A::AbstractMatrix, jpvt::AbstractVector, tau::AbstractVector)::Tuple = geqp3!(A, jpvt, tau)
@inline tzrzf!(::SIMDBackend, A::AbstractMatrix, tau::AbstractVector)::Tuple = tzrzf!(A, tau)
@inline orgqr!(::SIMDBackend, A::AbstractMatrix, tau::AbstractVector)::AbstractMatrix = orgqr!(A, tau)
@inline orglq!(::SIMDBackend, A::AbstractMatrix, tau::AbstractVector)::AbstractMatrix = orglq!(A, tau)
@inline orgql!(::SIMDBackend, A::AbstractMatrix, tau::AbstractVector)::AbstractMatrix = orgql!(A, tau)
@inline orgrq!(::SIMDBackend, A::AbstractMatrix, tau::AbstractVector)::AbstractMatrix = orgrq!(A, tau)
@inline ormqr!(::SIMDBackend, side::Char, trans::Char, A::AbstractMatrix, tau::AbstractVector, C::AbstractMatrix)::AbstractMatrix = ormqr!(side, trans, A, tau, C)
@inline ormlq!(::SIMDBackend, side::Char, trans::Char, A::AbstractMatrix, tau::AbstractVector, C::AbstractMatrix)::AbstractMatrix = ormlq!(side, trans, A, tau, C)
@inline ormql!(::SIMDBackend, side::Char, trans::Char, A::AbstractMatrix, tau::AbstractVector, C::AbstractMatrix)::AbstractMatrix = ormql!(side, trans, A, tau, C)
@inline ormrq!(::SIMDBackend, side::Char, trans::Char, A::AbstractMatrix, tau::AbstractVector, C::AbstractMatrix)::AbstractMatrix = ormrq!(side, trans, A, tau, C)
@inline ormrz!(::SIMDBackend, side::Char, trans::Char, A::AbstractMatrix, tau::AbstractVector, C::AbstractMatrix)::AbstractMatrix = ormrz!(side, trans, A, tau, C)
# orgtr!/ungtr! bind the IN-PLACE (caller-supplied Q) forms — the 3-arg convenience forms allocate Q.
@inline orgtr!(::SIMDBackend, uplo::Char, A::AbstractMatrix, tau::AbstractVector, Q::AbstractMatrix)::AbstractMatrix = orgtr!(uplo, A, tau, Q)
@inline ungtr!(::SIMDBackend, uplo::Char, A::AbstractMatrix, tau::AbstractVector, Q::AbstractMatrix)::AbstractMatrix = ungtr!(uplo, A, tau, Q)

# ── Nonsymmetric-eigen backend surface (AbstractLAPACK contract). Same thin-wrapper form; every one
# binds the IN-PLACE method, so the contract cannot be satisfied by a form that allocates its own
# output. One method per DISTINCT implementation: `unmhr!` is a `const` alias of `ormhr!`
# (hessenberg.jl:379) and `unghr!` a one-line forwarding method onto `orghr!` (hessenberg.jl:344), so
# neither adds an implementation to wrap. `geev!` gets TWO methods because its real and complex
# in-place forms have different arities (wr+wi vs a single complex w) — the contract can only declare
# one of them, so the complex arm is held to the guarantee by verify.jl instead.
@inline gebal!(::SIMDBackend, A::AbstractMatrix, scale::AbstractVector; kw...)::Tuple = gebal!(A, scale; kw...)
@inline gebak!(::SIMDBackend, job::AbstractChar, side::AbstractChar, ilo::Integer, ihi::Integer, scale::AbstractVector, V::AbstractMatrix)::AbstractMatrix = gebak!(job, side, ilo, ihi, scale, V)
@inline gehrd!(::SIMDBackend, A::AbstractMatrix, ilo::Integer, ihi::Integer, tau::AbstractVector)::AbstractMatrix = gehrd!(A, ilo, ihi, tau)
@inline orghr!(::SIMDBackend, A::AbstractMatrix, ilo::Integer, ihi::Integer, tau::AbstractVector)::AbstractMatrix = orghr!(A, ilo, ihi, tau)
@inline ormhr!(::SIMDBackend, side::Char, trans::Char, ilo::Integer, ihi::Integer, A::AbstractMatrix, tau::AbstractVector, C::AbstractMatrix)::AbstractMatrix = ormhr!(side, trans, ilo, ihi, A, tau, C)
@inline hseqr!(::SIMDBackend, job::AbstractChar, compz::AbstractChar, H::AbstractMatrix, ilo::Integer, ihi::Integer, w::AbstractVector, Z::AbstractMatrix)::Integer = hseqr!(job, compz, H, ilo, ihi, w, Z)
@inline trevc!(::SIMDBackend, side::AbstractChar, howmny::AbstractChar, Ts::AbstractMatrix, VL::AbstractMatrix, VR::AbstractMatrix)::AbstractMatrix = trevc!(side, howmny, Ts, VL, VR)
@inline trexc!(::SIMDBackend, compq::AbstractChar, Ts::AbstractMatrix, Q::AbstractMatrix, ifst::Integer, ilst::Integer)::Tuple = trexc!(compq, Ts, Q, ifst, ilst)
@inline trsyl!(::SIMDBackend, transa::AbstractChar, transb::AbstractChar, isgn::Integer, A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)::Tuple = trsyl!(transa, transb, isgn, A, B, C)
@inline geev!(::SIMDBackend, jobvl::AbstractChar, jobvr::AbstractChar, A::AbstractMatrix, wr::AbstractVector, wi::AbstractVector, VL::AbstractMatrix, VR::AbstractMatrix, scale::AbstractVector)::Tuple = geev!(jobvl, jobvr, A, wr, wi, VL, VR, scale)
@inline geev!(::SIMDBackend, jobvl::AbstractChar, jobvr::AbstractChar, A::AbstractMatrix, w::AbstractVector, VL::AbstractMatrix, VR::AbstractMatrix, scale::AbstractVector)::Tuple = geev!(jobvl, jobvr, A, w, VL, VR, scale)
@inline gees!(::SIMDBackend, jobvs::AbstractChar, A::AbstractMatrix, w::AbstractVector, VS::AbstractMatrix, scale::AbstractVector)::Tuple = gees!(jobvs, A, w, VS, scale)

# ── Bunch-Kaufman backend surface. `sysv!`/`hesv!`/`syconv!` bind the 4-argument forms; the 3-argument
# siblings allocate `ipiv`/`work`.
@inline sytrf!(::SIMDBackend, A::AbstractMatrix, ipiv::AbstractVector; kw...)::Integer = sytrf!(A, ipiv; kw...)
@inline hetrf!(::SIMDBackend, A::AbstractMatrix, ipiv::AbstractVector; kw...)::Integer = hetrf!(A, ipiv; kw...)
@inline sytrs!(::SIMDBackend, A::AbstractMatrix, ipiv::AbstractVector, B::AbstractVecOrMat; kw...)::AbstractVecOrMat = sytrs!(A, ipiv, B; kw...)
@inline hetrs!(::SIMDBackend, A::AbstractMatrix, ipiv::AbstractVector, B::AbstractVecOrMat; kw...)::AbstractVecOrMat = hetrs!(A, ipiv, B; kw...)
@inline sytri!(::SIMDBackend, A::AbstractMatrix, ipiv::AbstractVector; kw...)::AbstractMatrix = sytri!(A, ipiv; kw...)
@inline hetri!(::SIMDBackend, A::AbstractMatrix, ipiv::AbstractVector; kw...)::AbstractMatrix = hetri!(A, ipiv; kw...)
@inline sysv!(::SIMDBackend, uplo::Char, A::AbstractMatrix, B::AbstractVecOrMat, ipiv::AbstractVector)::Tuple = sysv!(uplo, A, B, ipiv)
@inline hesv!(::SIMDBackend, uplo::Char, A::AbstractMatrix, B::AbstractVecOrMat, ipiv::AbstractVector)::Tuple = hesv!(uplo, A, B, ipiv)
@inline syconv!(::SIMDBackend, uplo::AbstractChar, A::AbstractMatrix, ipiv::AbstractVector, work::AbstractVector)::Tuple = syconv!(uplo, A, ipiv, work)

# ── QZ / generalized-eigen backend surface. `ggev!` gets two methods for the same reason `geev!` does.
@inline gghrd!(::SIMDBackend, compq::AbstractChar, compz::AbstractChar, A::AbstractMatrix, B::AbstractMatrix, Q::AbstractMatrix, Z::AbstractMatrix; kw...)::Tuple = gghrd!(compq, compz, A, B, Q, Z; kw...)
@inline hgeqz!(::SIMDBackend, job::AbstractChar, compq::AbstractChar, compz::AbstractChar, H::AbstractMatrix, Tm::AbstractMatrix, alpha::AbstractVector, beta::AbstractVector, Q::AbstractMatrix, Z::AbstractMatrix; kw...)::Integer = hgeqz!(job, compq, compz, H, Tm, alpha, beta, Q, Z; kw...)
@inline tgevc!(::SIMDBackend, side::AbstractChar, howmny::AbstractChar, S::AbstractMatrix, Pm::AbstractMatrix, VL::AbstractMatrix, VR::AbstractMatrix)::Integer = tgevc!(side, howmny, S, Pm, VL, VR)
@inline tgsen!(::SIMDBackend, select::AbstractVector, S::AbstractMatrix, Tm::AbstractMatrix, Q::AbstractMatrix, Z::AbstractMatrix, alpha::AbstractVector, beta::AbstractVector)::Tuple = tgsen!(select, S, Tm, Q, Z, alpha, beta)
@inline ggev!(::SIMDBackend, jobvl::AbstractChar, jobvr::AbstractChar, A::AbstractMatrix, B::AbstractMatrix, alphar::AbstractVector, alphai::AbstractVector, beta::AbstractVector, vl::AbstractMatrix, vr::AbstractMatrix)::Tuple = ggev!(jobvl, jobvr, A, B, alphar, alphai, beta, vl, vr)
@inline ggev!(::SIMDBackend, jobvl::AbstractChar, jobvr::AbstractChar, A::AbstractMatrix, B::AbstractMatrix, alpha::AbstractVector, beta::AbstractVector, vl::AbstractMatrix, vr::AbstractMatrix)::Tuple = ggev!(jobvl, jobvr, A, B, alpha, beta, vl, vr)
@inline gges!(::SIMDBackend, jobvsl::AbstractChar, jobvsr::AbstractChar, A::AbstractMatrix, B::AbstractMatrix, alpha::AbstractVector, beta::AbstractVector, vsl::AbstractMatrix, vsr::AbstractMatrix)::Tuple = gges!(jobvsl, jobvsr, A, B, alpha, beta, vsl, vsr)

# ── SVD front-half / generalized SVD. `bdsqr!` gets two methods: the real 4-argument form the contract
# declares, and the complex 6-argument form, which is a separate kernel at a different arity.
@inline gebd2!(::SIMDBackend, A::AbstractMatrix, d::AbstractVector, e::AbstractVector, tauq::AbstractVector, taup::AbstractVector)::AbstractMatrix = gebd2!(A, d, e, tauq, taup)
@inline bdsqr!(::SIMDBackend, d::AbstractVector, e::AbstractVector, U::Union{Nothing, AbstractMatrix}, V::Union{Nothing, AbstractMatrix})::AbstractVector = bdsqr!(d, e, U, V)
@inline bdsqr!(::SIMDBackend, uplo::AbstractChar, d::AbstractVector, e::AbstractVector, Vt::AbstractMatrix, U::AbstractMatrix, C::AbstractMatrix)::Tuple = bdsqr!(uplo, d, e, Vt, U, C)
@inline ggsvd!(::SIMDBackend, jobu::AbstractChar, jobv::AbstractChar, jobq::AbstractChar, A::AbstractMatrix, B::AbstractMatrix, U::AbstractMatrix, V::AbstractMatrix, Q::AbstractMatrix, alpha::AbstractVector, beta::AbstractVector, R::AbstractMatrix)::Tuple = ggsvd!(jobu, jobv, jobq, A, B, U, V, Q, alpha, beta, R)

# ── Least squares / constrained least squares / tridiagonal eigen. `gglse!`/`stebz!`/`stein!` bind the
# trailing-buffer forms; their convenience siblings allocate x / w+iblock+isplit / Z.
@inline gels!(::SIMDBackend, trans::Char, A::AbstractMatrix, B::AbstractMatrix)::Tuple = gels!(trans, A, B)
@inline gelsy!(::SIMDBackend, A::AbstractMatrix, B::AbstractMatrix, jpvt::AbstractVector, rcond::Real)::Tuple = gelsy!(A, B, jpvt, rcond)
@inline gglse!(::SIMDBackend, A::AbstractMatrix, c::AbstractVector, B::AbstractMatrix, d::AbstractVector, x::AbstractVector)::Tuple = gglse!(A, c, B, d, x)
@inline stebz!(::SIMDBackend, range::AbstractChar, order::AbstractChar, vl::Real, vu::Real, il::Integer, iu::Integer, abstol::Real, d::AbstractVector, e::AbstractVector, w::AbstractVector, iblock::AbstractVector, isplit::AbstractVector)::Tuple = stebz!(range, order, vl, vu, il, iu, abstol, d, e, w, iblock, isplit)
@inline stein!(::SIMDBackend, d::AbstractVector, e::AbstractVector, w::AbstractVector, iblock::AbstractVector, isplit::AbstractVector, Z::AbstractMatrix)::AbstractMatrix = stein!(d, e, w, iblock, isplit, Z)
