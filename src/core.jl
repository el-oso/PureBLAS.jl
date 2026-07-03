# Shared kernel primitives.
#
# The low-level kernels are written ONCE over a tiny accessor interface (`_ld`/`_st!`) so the
# exact same loop serves both consumption modes:
#   * Mode 1 (LBT C-ABI): arguments arrive as `Ptr{T}` from libblastrampoline.
#   * Mode 2 (native/AD):  arguments are `AbstractVector{T}` (incl. ForwardDiff.Dual elements).
# No copying, no allocation — `_ld`/`_st!` inline to `unsafe_load`/`getindex` respectively.

# Element types BLAS names cover. `BlasReal` get the SIMD.jl fast path; everything else
# (complex, and any other `T<:Number` such as ForwardDiff.Dual) uses the generic scalar loop.
const BlasReal = Union{Float32, Float64}
const BlasComplex = Union{ComplexF32, ComplexF64}
const BlasFloat = Union{Float32, Float64, ComplexF32, ComplexF64}

@inline _ld(p::Ptr, i::Integer) = unsafe_load(p, i)
@inline _ld(a, i::Integer) = @inbounds a[i]
@inline _st!(p::Ptr{T}, i::Integer, v::T) where {T} = (unsafe_store!(p, v, i); v)
# Ptr store of a differently-typed value (e.g. a real diagonal into a complex buffer — hpr/hpr2):
# convert to the pointee type, matching AbstractVector `setindex!`'s implicit convert. The exact-type
# method above stays the (more specific) fast path.
@inline _st!(p::Ptr{T}, i::Integer, v) where {T} = (unsafe_store!(p, convert(T, v), i); v)
@inline _st!(a, i::Integer, v) = (@inbounds a[i] = v; v)

@inline _et(::Ptr{T}) where {T} = T
@inline _et(a) = eltype(a)

# Fortran BLAS start index for a (possibly negative) increment: walk backwards from the end.
@inline _start(n::Integer, inc::Integer) = inc > 0 ? 1 : 1 + (1 - n) * inc

# |·| used by asum/iamax: BLAS uses |Re|+|Im| for complex (NOT the modulus), abs for real.
@inline _l1(x::Real) = abs(x)
@inline _l1(z::Complex) = abs(real(z)) + abs(imag(z))

# One step of the LAPACK `lassq` scaled sum-of-squares (overflow/underflow safe) — the
# correctness boundary for nrm2. Returns the updated (scale, ssq).
@inline function _lassq(scale::R, ssq::R, a::R) where {R<:Real}
    if !iszero(a)
        absa = abs(a)
        if scale < absa
            ssq = one(R) + ssq * (scale / absa)^2
            scale = absa
        else
            ssq = ssq + (absa / scale)^2
        end
    end
    return scale, ssq
end
@inline _nrm2_acc(scale, ssq, x::Real) = _lassq(scale, ssq, x)
@inline function _nrm2_acc(scale, ssq, z::Complex)
    scale, ssq = _lassq(scale, ssq, real(z))
    return _lassq(scale, ssq, imag(z))
end
