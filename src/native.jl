# Mode 2 bare native API: ergonomic, backend-free entry points that forward to the default
# backend. These are what direct Julia callers (and AD) use — e.g. `PureBLAS.dot(x, y)`,
# `PureBLAS.axpy!(y, a, x)`. AD-traceable because the whole call tree is plain Julia source.

axpy!(y::AbstractVector, a::Number, x::AbstractVector) = axpy!(DEFAULT_BACKEND, y, a, x)
scal!(a::Number, x::AbstractVector) = scal!(DEFAULT_BACKEND, a, x)
blascopy!(y::AbstractVector, x::AbstractVector) = blascopy!(DEFAULT_BACKEND, y, x)
swap!(x::AbstractVector, y::AbstractVector) = swap!(DEFAULT_BACKEND, x, y)
dot(x::AbstractVector, y::AbstractVector) = dot(DEFAULT_BACKEND, x, y)
dotu(x::AbstractVector, y::AbstractVector) = dotu(DEFAULT_BACKEND, x, y)
nrm2(x::AbstractVector) = nrm2(DEFAULT_BACKEND, x)
asum(x::AbstractVector) = asum(DEFAULT_BACKEND, x)
iamax(x::AbstractVector) = iamax(DEFAULT_BACKEND, x)

# Level 2. @inline + explicit kwarg forwarding (no `kw...` splat) so the keyword-argument overhead
# is elided at the call site — it otherwise dominates tiny-matrix gemv (~200 ns vs a ~33 ns kernel).
@inline gemv!(
    y::AbstractVector, A::AbstractMatrix, x::AbstractVector;
    alpha = one(eltype(A)), beta = zero(eltype(A)), trans::Char = 'N'
) =
    gemv!(DEFAULT_BACKEND, y, A, x; alpha, beta, trans)
@inline ger!(alpha::Number, x::AbstractVector, y::AbstractVector, A::AbstractMatrix; conj::Bool = false) =
    ger!(DEFAULT_BACKEND, alpha, x, y, A; conj)
@inline symv!(
    y::AbstractVector, A::AbstractMatrix, x::AbstractVector;
    uplo::Char = 'U', alpha = one(eltype(A)), beta = zero(eltype(A))
) =
    symv!(DEFAULT_BACKEND, y, A, x; uplo, alpha, beta)
@inline hemv!(
    y::AbstractVector, A::AbstractMatrix, x::AbstractVector;
    uplo::Char = 'U', alpha = one(eltype(A)), beta = zero(eltype(A))
) =
    hemv!(DEFAULT_BACKEND, y, A, x; uplo, alpha, beta)
@inline trmv!(A::AbstractMatrix, x::AbstractVector; uplo::Char = 'U', trans::Char = 'N', diag::Char = 'N') =
    trmv!(DEFAULT_BACKEND, A, x; uplo, trans, diag)
@inline trsv!(A::AbstractMatrix, x::AbstractVector; uplo::Char = 'U', trans::Char = 'N', diag::Char = 'N') =
    trsv!(DEFAULT_BACKEND, A, x; uplo, trans, diag)

"""
    gemv(A, x; trans='N') -> y

Allocating matrix-vector product `op(A)·x`.
"""
function gemv(A::AbstractMatrix, x::AbstractVector; trans::Char = 'N')
    T = promote_type(eltype(A), eltype(x))
    y = zeros(T, trans == 'N' ? size(A, 1) : size(A, 2))
    return gemv!(y, A, x; alpha = one(T), beta = zero(T), trans)
end
# Level 2 packed
@inline spmv!(y::AbstractVector, AP::AbstractVector, x::AbstractVector; uplo::Char = 'U', alpha = one(eltype(AP)), beta = zero(eltype(AP))) =
    spmv!(DEFAULT_BACKEND, y, AP, x; uplo, alpha, beta)
@inline hpmv!(y::AbstractVector, AP::AbstractVector, x::AbstractVector; uplo::Char = 'U', alpha = one(eltype(AP)), beta = zero(eltype(AP))) =
    hpmv!(DEFAULT_BACKEND, y, AP, x; uplo, alpha, beta)
@inline tpmv!(AP::AbstractVector, x::AbstractVector; uplo::Char = 'U', trans::Char = 'N', diag::Char = 'N') =
    tpmv!(DEFAULT_BACKEND, AP, x; uplo, trans, diag)
# packed rank-1/2 updates (mirror ger!'s arg order: scalar, vectors, matrix)
@inline spr!(alpha::Number, x::AbstractVector, AP::AbstractVector; uplo::Char = 'U') =
    spr!(DEFAULT_BACKEND, alpha, x, AP; uplo)
@inline spr2!(alpha::Number, x::AbstractVector, y::AbstractVector, AP::AbstractVector; uplo::Char = 'U') =
    spr2!(DEFAULT_BACKEND, alpha, x, y, AP; uplo)
@inline hpr!(alpha::Number, x::AbstractVector, AP::AbstractVector; uplo::Char = 'U') =
    hpr!(DEFAULT_BACKEND, alpha, x, AP; uplo)
@inline hpr2!(alpha::Number, x::AbstractVector, y::AbstractVector, AP::AbstractVector; uplo::Char = 'U') =
    hpr2!(DEFAULT_BACKEND, alpha, x, y, AP; uplo)
@inline tpsv!(AP::AbstractVector, x::AbstractVector; uplo::Char = 'U', trans::Char = 'N', diag::Char = 'N') =
    tpsv!(DEFAULT_BACKEND, AP, x; uplo, trans, diag)
# Level 2 banded
@inline gbmv!(
    y::AbstractVector, AB::AbstractMatrix, x::AbstractVector, m::Integer, kl::Integer, ku::Integer;
    trans::Char = 'N', alpha = one(eltype(AB)), beta = zero(eltype(AB))
) =
    gbmv!(DEFAULT_BACKEND, y, AB, x, m, kl, ku; trans, alpha, beta)
@inline sbmv!(y::AbstractVector, AB::AbstractMatrix, x::AbstractVector; uplo::Char = 'U', alpha = one(eltype(AB)), beta = zero(eltype(AB))) =
    sbmv!(DEFAULT_BACKEND, y, AB, x; uplo, alpha, beta)
@inline hbmv!(y::AbstractVector, AB::AbstractMatrix, x::AbstractVector; uplo::Char = 'U', alpha = one(eltype(AB)), beta = zero(eltype(AB))) =
    hbmv!(DEFAULT_BACKEND, y, AB, x; uplo, alpha, beta)
@inline tbmv!(AB::AbstractMatrix, x::AbstractVector; uplo::Char = 'U', trans::Char = 'N', diag::Char = 'N') =
    tbmv!(DEFAULT_BACKEND, AB, x; uplo, trans, diag)
@inline tbsv!(AB::AbstractMatrix, x::AbstractVector; uplo::Char = 'U', trans::Char = 'N', diag::Char = 'N') =
    tbsv!(DEFAULT_BACKEND, AB, x; uplo, trans, diag)
