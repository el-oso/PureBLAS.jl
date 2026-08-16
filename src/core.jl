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
@inline function _lassq(scale::R, ssq::R, a::R) where {R <: Real}
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

# ── TRIM-SAFE ERROR PATHS: USE `lazy"…"`, NOT `"…"` ──────────────────────────────────────────────────
# ROOT CAUSE (2026-08-16, Julia 1.13; three wrong hypotheses discarded before this one). An interpolated
# message with ≥4 pieces lowers to a ≥4-argument `string(...)` → `Base.print_to_string`. On 1.13, stock
# inference DESPECIALIZES that to `print_to_string(::Symbol, ::Vararg{Any})`, whose vararg-element value
# is genuinely `Any`, and `--trim=safe` rejects it:
#     Base._str_sizehint(φ ()::Any)::Int64 / Base.print(%new()::IOBuffer, φ ()::Any)::Any
# (The `φ ()` is a DISPLAY artifact — `verifytrim.jl` prints every PhiNode as empty. It is not dead code.)
#
# Despecialization is a CALLEE-SIDE `max_args` policy, which is why the three obvious rewrites all failed
# identically: `@noinline`, concretely-typed `Int` arguments, and `Symbol`-instead-of-`String` cannot
# influence it. Every variant still funnelled into the same ≥4-arg `string(...)`.
#
# `lazy"…"` fixes it because `LazyString` defers ALL printing past the verified call graph — it is also
# Base's own idiom for error paths (no eager allocation, less invalidation). Messages are unchanged; the
# string is built only if someone actually displays the exception.
#
# WHY THE REAL BUILD WAS ALWAYS FINE: `juliac --trim=safe` includes `juliac-trim-base.jl` /
# `juliac-trim-stdlib.jl`, which raise `print_to_string`'s `max_args` so no despecialization happens.
# StrictMode's `@assert_trim_compatible` verifies STOCK Base and therefore a DIFFERENT program — the
# `.so` builds clean on 1.13 (verified: exit 0, 48.5 MB) while the assertion fails. That is a StrictMode
# bug, tracked separately; `lazy"…"` makes these paths clean under BOTH configurations so PureBLAS does
# not depend on that fix landing. `@noinline` is kept purely for cold-path hygiene (Base does the same),
# NOT because the verifier needs it. See kb `julia-113-test-toolchain-and-env-discipline`.
@noinline _throw_square(op::Symbol, k::Int) = throw(DimensionMismatch(lazy"$op: A must be $k×$k"))
@noinline _throw_mge_n(op::Symbol, m::Int, n::Int) =
    throw(ArgumentError(lazy"$op: requires m ≥ n (got $m×$n)"))
@noinline _throw_mle_n(op::Symbol, m::Int, n::Int) =
    throw(ArgumentError(lazy"$op: requires m ≤ n (got $m×$n)"))
@noinline _throw_len_m(op::Symbol, m::Int) = throw(DimensionMismatch(lazy"$op: length(tau) < m=$m"))
@noinline _throw_len_jpvt(op::Symbol, n::Int) =
    throw(DimensionMismatch(lazy"$op: length(jpvt) < n=$n"))
@noinline _throw_len_tau_mn(op::Symbol, k::Int) =
    throw(DimensionMismatch(lazy"$op: length(tau) < min(size(A))=$k"))
@noinline _throw_brows_mn(op::Symbol, got::Int, need::Int) =
    throw(DimensionMismatch(lazy"$op: size(B,1)=$got must be ≥ max(m,n)=$need"))
@noinline _throw_brows_opa(op::Symbol, got::Int, need::Int) =
    throw(DimensionMismatch(lazy"$op: size(B,1)=$got must be ≥ max(rows,cols of op(A))=$need"))
@noinline _throw_trans_ntc(op::Symbol, c::Char) =
    throw(ArgumentError(lazy"$op: trans must be 'N', 'T' or 'C', got $(repr(c))"))
@noinline _throw_trans_nt(op::Symbol, c::Char) =
    throw(ArgumentError(lazy"$op: trans must be 'N' or 'T', got $(repr(c))"))
@noinline _throw_side_lr(op::Symbol, c::Char) =
    throw(ArgumentError(lazy"$op: side must be 'L' or 'R', got $(repr(c))"))
@noinline _throw_gglse_dims(m::Int, n::Int, p::Int) =
    throw(DimensionMismatch(lazy"gglse!: need p ≤ n ≤ m+p (got m=$m,n=$n,p=$p)"))
@noinline _throw_gbtrf_nb(nb::Int, kl::Int) =
    throw(ArgumentError(lazy"_gbtrf_blocked!: needs nb ≤ kl (got nb=$nb, kl=$kl)"))
@noinline _throw_packed_len(L::Int) =
    throw(DimensionMismatch(lazy"packed length $L is not n(n+1)/2"))
