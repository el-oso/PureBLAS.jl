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

# ── CHECKED-CARRIER ACCESSORS ───────────────────────────────────────────────────────────────────
#
# THE PROBLEM THEY SOLVE. The SIMD fast paths hoist a `pointer(x)` out of the loop and address it by
# byte offset — that is what makes them fast, and it is also why a wrong loop bound writes past the
# array in silence. `@inbounds` is irrelevant here: `unsafe_store!`/`vstore` on a raw `Ptr` have NO
# bounds check to elide, so `--check-bounds=yes` cannot see them under ANY setting. The gbmv OOB
# write (186f63f) wrote 25 elements past a 40-element `y` and every test stayed green.
#
# THE FIX. Keep the hoisted pointer — most kernels already take the carrier (`x`, `y`, `AB`) and
# derive the pointer internally, so the array is in scope for free. Route the store through a helper
# that also checks the CARRIER, and gate the check on the compiler flag:
#
#   release  (`_CHECKED == false`): `false && checkbounds(...)` folds away at compile time. The
#            emitted code is BYTE-IDENTICAL to the raw-Ptr form — measured on the `_axpy_phase!`
#            shape, 176 vs 176 normalized instructions, textually equal. Zero cost, and trim-safe:
#            the juliac build runs default flags so nothing survives into the .so.
#   checked  (`julia --check-bounds=yes`): a real `checkbounds`, throwing a BoundsError naming the
#            offending line. Costs ~1.03-1.5x (measured +54% in L1, +3% at n=3e4) — a debug/CI price.
#
# `Base.JLOptions().check_bounds == 1` is safe as a `const`: the pkgimage cache is KEYED on that flag
# (`Base.CacheFlags`), so a checked run cannot pick up a cache built unchecked, and vice versa.
# Preferred over `@boundscheck`/`@inbounds` propagation, which this codebase has been burned by
# (`@inline` not propagating into `@generated` bodies, kb generated-inline-meta-hazard).
#
# WHY NOT JUST USE ARRAYS. Measured, and rejected: SIMD.jl's array `vstore` calls `pointer(a, i)` per
# access and its store carries no TBAA/alias metadata, so LLVM cannot prove a wide store does not
# clobber the Vector's own data-pointer field — the loop reloads `mov rcx, [rax]` after EVERY vector
# store. Loads CSE fine; stores do not. Cost: +61% at axpy n=1000, +21% on the gbmv conv block.
# The carrier here is passed for its BOUNDS, never indexed through — so no reload.
#
# MASKED / PARTIAL TILES: pass the ACTIVE span, not the full vector width. A direct-read microkernel
# legitimately loads a clamped partial tile, and checking the full W lanes would false-positive on
# correct code (the `directb` rule — see test/memsafe_verify.jl).
const _CHECKED = Base.JLOptions().check_bounds == 1

# Scalar store through a hoisted pointer, bounds-checked against its carrier. `i` is 1-based, in
# ELEMENTS, matching `unsafe_store!`'s index convention.
@inline function _stc!(a, p::Ptr{T}, i::Integer, v::T) where {T}
    _CHECKED && checkbounds(a, i)
    unsafe_store!(p, v, i)
    return v
end
# Vector store of `nact` ACTIVE lanes at 0-based element offset `off`. `nact` is what the mask (or
# the clamp) actually writes — never the nominal width, or a legitimately clamped tail trips it.
@inline function _vstc!(a, p::Ptr{T}, off::Integer, nact::Integer, v) where {T}
    _CHECKED && nact > 0 && checkbounds(a, (off + 1):(off + nact))
    vstore(v, p + off * sizeof(T))
    return nothing
end
# Masked variant: SIMD.jl writes only the masked lanes, but the mask is a runtime value, so the
# caller states the active count it guarantees.
@inline function _vstc!(a, p::Ptr{T}, off::Integer, nact::Integer, v, msk) where {T}
    _CHECKED && nact > 0 && checkbounds(a, (off + 1):(off + nact))
    vstore(v, p + off * sizeof(T), msk)
    return nothing
end

# ── PRE-OFFSET POINTERS ─────────────────────────────────────────────────────────────────────────
# Panel/tile kernels are handed a pointer that ALREADY points into the middle of the carrier
# (`_spmv_lpanel!(…, yp + jbl*sz, …)`), and then index it from 0/1 locally. Checking the LOCAL index
# against the carrier would validate the wrong element — and validate it permissively, since the
# local index is always smaller. These variants take `base`, the 0-based element offset of `p` within
# `a`, and check `base + i`.
#
# This is the common case in BLAS-3 and in every panel driver, so it is the general form; the
# 4-argument versions above are the special case `base == 0`.
# ── THE CARRIER MUST VANISH IN RELEASE, NOT JUST THE CHECK ──────────────────────────────────────
# MEASURED (2026-08-17): threading `(y, ybase)` into the four @generated spmv panel kernels cost +4
# instructions INSIDE the loop bodies (456 -> 460; prologue identical at 113). Those are NOT the
# bounds check — `_CHECKED` is false there, so the check folds away completely. They are the cost of
# two extra LIVE VALUES through the generated bodies perturbing register allocation. Pure overhead in
# a build that cannot use them.
#
# So the carrier is passed through `_carrier`, which yields `nothing` in a release build. `_CHECKED`
# is a compile-time const, so the ternary folds and the kernel SPECIALIZES on `::Nothing` — a
# singleton, no register, no live range. The `::Nothing` methods below are the unchecked stores.
#
# This is what "free in release" has to mean for a signature change: not merely that the check
# disappears, but that nothing about the argument survives to compete for registers.
# ── SEGMENT VALIDATION: check where the OFFSET IS COMPUTED, not at every store ──────────────────
#
# THE OBSERVATION THAT DRIVES THIS. Every out-of-bounds write found in this library came from an
# offset or extent computed from the WRONG operand — ONCE per call or per column — after which every
# store rides it (kb pureblas-shape-blind-tests-oob-writes):
#   * gbmv: `chi` bounded by the x READ limit, ignoring that `y` has length n on trans='T'
#   * ger:  `np` striding 3 while the dispatched panel wrote 8 columns
#   * spr:  a packed `_pkL`/`_pkU` offset landing in the wrong column
# So per-store checking pays O(n) to catch an O(1) error. Validating the SEGMENT where it is built
# costs two checks per column and fails AT the arithmetic that was wrong — a far better diagnostic
# than a BoundsError deep inside a SIMD kernel.
#
# There are 126 segment constructions in this tree against 655 store sites: 5x smaller surface.
#
# MECHANISM is Julia's own (1.11+), not a bespoke one: `Base.memoryrefnew(ref, i, boundscheck::Bool)`
# is an offsettable reference carrying provenance, with a COMPILE-TIME-SELECTABLE check — the same
# `Expr(:boundscheck)` machinery `--check-bounds` already drives for `Array`.
#
# MEASURED FREE IN RELEASE, by construction rather than by luck (bench/probes/memoryref_segment.jl):
#     pointer(A) + off*sz                        lines=29  branches=0
#     Ptr{T}(pointer(memoryrefnew(ref,i,false))) lines=29  branches=0   <- IDENTICAL
#     Ptr{T}(pointer(memoryrefnew(ref,i,true)))  lines=87  branches=2   <- the check, when enabled
# Positive control verified: the checked form throws BoundsError on a bad offset, the unchecked form
# does not. Offsets are 1-BASED (ref@4 - ref@1 = 24 bytes = 3 elements), matching kernel indexing.
#
# ⚠ `pointer(::MemoryRef{T})` returns `Ptr{Nothing}`, NOT `Ptr{T}`. A missed cast does not fail —
# `unsafe_load` quietly returns `nothing` and a store would be the wrong width. That is the exact
# silent-corruption class this exists to remove, so the cast lives HERE, once, never at call sites.
"""
    _seg(a, off, len) -> Ptr{T}

Pointer to the `len`-element segment of `a` starting at 0-based element offset `off`. Under
`--check-bounds=yes` both ends are validated against `a`'s own memory; in a release build this is
exactly `pointer(a) + off*sizeof(T)`.

Pass the TRUE length the caller will touch: `len` is what makes a segment that starts in bounds and
runs off the end detectable.
"""
# ⚠ TAKE THE ALREADY-HOISTED POINTER. The first version of this derived the pointer itself
# (`Ptr{T}(pointer(memoryrefnew(a.ref, off+1, false)))`). That is free in a STANDALONE function —
# measured 29 lines vs 29 for raw arithmetic — but NOT inside a kernel loop, where the original code
# hoists `Ap = pointer(AB)` once and does pure arithmetic per iteration. Re-deriving `a.ref` and
# calling `pointer` every iteration cost **+63 instructions (+3.1%) on `_gbmv_n_simd!`** (2027 →
# 2090), measured. The synthetic probe could not see it because it had nothing to hoist.
#
# So: the pointer stays hoisted by the caller and is passed in; `a` is used ONLY to validate the
# extent. In release this compiles to exactly `p + off*sizeof(T)` — the arithmetic it replaces.
@inline function _seg(a, p::Ptr{T}, off::Integer, len::Integer) where {T}
    _CHECKED && _seg_check(a, off, len)
    return p + Int(off) * sizeof(T)
end
# A raw Ptr carrier has nothing to validate against (C-ABI entry, or a segment already derived from
# one) — passthrough, honestly unchecked rather than falsely reassuring.
@inline _seg(::Ptr, p::Ptr{T}, off::Integer, len::Integer) where {T} = p + Int(off) * sizeof(T)
@inline _seg(::Nothing, p::Ptr{T}, off::Integer, len::Integer) where {T} = p + Int(off) * sizeof(T)

@noinline function _seg_check(a::Array, off::Integer, len::Integer)
    len <= 0 && return nothing                                   # empty segment: nothing dereferenced
    Base.memoryrefnew(a.ref, Int(off) + 1, true)                 # first element in bounds
    Base.memoryrefnew(a.ref, Int(off) + Int(len), true)          # LAST element in bounds
    return nothing
end

@inline _carrier(a) = _CHECKED ? a : nothing
@inline _cbase(b::Integer) = _CHECKED ? b : 0

# Shared kernels (`_axpy_simd!`, `_axpy_unrolled!`, …) take `x`/`y` UNTYPED and normalize with
# `_ptr`, so an argument is either a raw `Ptr` (a segment the caller already offset) or an array. When
# it is an array, the carrier is ALREADY IN HAND — no signature change, no call-site churn, and every
# native-API entry point becomes checkable for free. When it is a `Ptr` there is nothing to check
# against, and this yields `nothing` (the unchecked methods).
#
# `y isa Ptr` is a TYPE test, so it resolves at specialization time and costs nothing either way.
#
# HONEST SCOPE: this covers callers that hand over a whole array. It does NOT cover the delegated
# segment writes — `gbmv` passing `yp + (ilo-1)*sz`, `spr` passing a packed column base — which are
# precisely where a wrong offset would land. Covering those needs the carrier threaded explicitly to
# each call site (28 for `_axpy_simd!` alone); tracked in #148.
@inline _carrier_arr(y) = (_CHECKED && !(y isa Ptr)) ? y : nothing

@inline function _stcb!(a, base::Integer, p::Ptr{T}, i::Integer, v::T) where {T}
    _CHECKED && checkbounds(a, base + i)
    unsafe_store!(p, v, i)
    return v
end
# Release-build methods: the carrier is `nothing`, so there is nothing to check and nothing live.
@inline _stc!(::Nothing, p::Ptr{T}, i::Integer, v::T) where {T} = (unsafe_store!(p, v, i); v)
@inline _stcb!(::Nothing, ::Integer, p::Ptr{T}, i::Integer, v::T) where {T} = (unsafe_store!(p, v, i); v)
@inline _vstc!(::Nothing, p::Ptr{T}, off::Integer, ::Integer, v) where {T} =
    (vstore(v, p + off * sizeof(T)); nothing)
@inline _vstc!(::Nothing, p::Ptr{T}, off::Integer, ::Integer, v, msk) where {T} =
    (vstore(v, p + off * sizeof(T), msk); nothing)
@inline _vstcb!(::Nothing, ::Integer, p::Ptr{T}, off::Integer, ::Integer, v) where {T} =
    (vstore(v, p + off * sizeof(T)); nothing)
@inline _vstcb!(::Nothing, ::Integer, p::Ptr{T}, off::Integer, ::Integer, v, msk) where {T} =
    (vstore(v, p + off * sizeof(T), msk); nothing)
@inline function _vstcb!(a, base::Integer, p::Ptr{T}, off::Integer, nact::Integer, v) where {T}
    _CHECKED && nact > 0 && checkbounds(a, (base + off + 1):(base + off + nact))
    vstore(v, p + off * sizeof(T))
    return nothing
end
@inline function _vstcb!(a, base::Integer, p::Ptr{T}, off::Integer, nact::Integer, v, msk) where {T}
    _CHECKED && nact > 0 && checkbounds(a, (base + off + 1):(base + off + nact))
    vstore(v, p + off * sizeof(T), msk)
    return nothing
end

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
