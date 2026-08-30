# Isbits pointer-backed matrix/vector for the Mode-1 C-ABI boundary.
#
# The old bridge `view(unsafe_wrap(Array, ptr, (ld, nc)), 1:r, 1:c)` builds a non-owning Array header
# PER operand (~80 B) plus a SubArray that heap-boxes when it escapes into a non-inlined pack/driver
# kernel (SubArray is non-isbits — it holds a mutable Array ref) — ~384 B/call across A/B/C.
#
# `PtrMatrix`/`PtrVector` carry only isbits fields, so the whole struct is isbits and passes BY VALUE
# into non-inlined kernels with NO heap box. The buffer is CALLER-owned (not Julia-GC): `parent(A)=A`,
# so the kernels' `GC.@preserve parent(A) …` pattern is a safe no-op (isbits ⇒ @preserve does nothing,
# and the C caller keeps the buffer alive across the call). Column-major, `ld` leading dimension.
# getindex/setindex are unsafe_load/store, so the whole call graph stays trim-clean.

struct PtrMatrix{T} <: AbstractMatrix{T}
    ptr::Ptr{T}
    m::Int
    n::Int
    ld::Int
end

@inline Base.size(A::PtrMatrix) = (A.m, A.n)
Base.IndexStyle(::Type{<:PtrMatrix}) = IndexCartesian()
@inline function Base.getindex(A::PtrMatrix, i::Integer, j::Integer)
    @boundscheck (1 <= i <= A.m && 1 <= j <= A.n) || throw(BoundsError(A, (i, j)))
    return unsafe_load(A.ptr, (j - 1) * A.ld + i)
end
@inline function Base.setindex!(A::PtrMatrix{T}, v, i::Integer, j::Integer) where {T}
    @boundscheck (1 <= i <= A.m && 1 <= j <= A.n) || throw(BoundsError(A, (i, j)))
    unsafe_store!(A.ptr, convert(T, v), (j - 1) * A.ld + i)
    return v
end
@inline Base.pointer(A::PtrMatrix) = A.ptr
@inline Base.pointer(A::PtrMatrix{T}, k::Integer) where {T} = A.ptr + (k - 1) * sizeof(T)
@inline Base.strides(A::PtrMatrix) = (1, A.ld)
@inline Base.stride(A::PtrMatrix, d::Integer) = d <= 1 ? 1 : A.ld
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::PtrMatrix{T}) where {T} = A.ptr
@inline Base.elsize(::Type{PtrMatrix{T}}) where {T} = sizeof(T)
@inline Base.parent(A::PtrMatrix) = A

struct PtrVector{T} <: AbstractVector{T}
    ptr::Ptr{T}
    n::Int
end

@inline Base.size(v::PtrVector) = (v.n,)
Base.IndexStyle(::Type{<:PtrVector}) = IndexLinear()
@inline function Base.getindex(v::PtrVector, i::Integer)
    @boundscheck (1 <= i <= v.n) || throw(BoundsError(v, i))
    return unsafe_load(v.ptr, i)
end
@inline function Base.setindex!(v::PtrVector{T}, x, i::Integer) where {T}
    @boundscheck (1 <= i <= v.n) || throw(BoundsError(v, i))
    unsafe_store!(v.ptr, convert(T, x), i)
    return x
end
@inline Base.pointer(v::PtrVector) = v.ptr
@inline Base.pointer(v::PtrVector{T}, k::Integer) where {T} = v.ptr + (k - 1) * sizeof(T)
@inline Base.strides(v::PtrVector) = (1,)
@inline Base.stride(v::PtrVector, d::Integer) = d <= 1 ? 1 : v.n
@inline Base.unsafe_convert(::Type{Ptr{T}}, v::PtrVector{T}) where {T} = v.ptr
@inline Base.elsize(::Type{PtrVector{T}}) where {T} = sizeof(T)
@inline Base.parent(v::PtrVector) = v

# Sub-views stay pointer-matrices (isbits, fast-path-eligible) instead of becoming SubArray-of-PtrMatrix
# (non-strided ⇒ falls to the generic kernel). Covers the range/colon combinations the L3 recursion and
# LAPACK panel loops use (view(A, r, r), view(A, :, r), view(A, r, :)); a single-column view(A, :, j) is
# a contiguous PtrVector. Column-major: sub-block (1,1) sits at ptr + (i0 + j0·ld); ld is unchanged.
@inline _vspan(::Colon, n::Int) = (0, n)
@inline _vspan(r::AbstractUnitRange{<:Integer}, ::Int) = (Int(first(r)) - 1, length(r))
@inline function Base.view(
        A::PtrMatrix{T}, I::Union{Colon, AbstractUnitRange{<:Integer}},
        J::Union{Colon, AbstractUnitRange{<:Integer}}
    ) where {T}
    i0, ni = _vspan(I, A.m)
    j0, nj = _vspan(J, A.n)
    return PtrMatrix(A.ptr + (i0 + j0 * A.ld) * sizeof(T), ni, nj, A.ld)
end
@inline function Base.view(A::PtrMatrix{T}, ::Colon, j::Integer) where {T}
    @boundscheck (1 <= j <= A.n) || throw(BoundsError(A, (:, j)))
    return PtrVector(A.ptr + (j - 1) * A.ld * sizeof(T), A.m)
end
@inline function Base.view(v::PtrVector{T}, I::AbstractUnitRange{<:Integer}) where {T}
    return PtrVector(v.ptr + (Int(first(I)) - 1) * sizeof(T), length(I))
end

# Fast-path predicate: unit-stride-1 dense matrix. For a StridedMatrix argument this const-folds to the
# identical `isa StridedMatrix && stride(A,1)==1` check the kernels used before (zero Mode-2 overhead);
# PtrMatrix is not in the closed StridedMatrix Union, so it gets an explicit `true` method.
#
# ⚠ `isbitstype(eltype(A))` IS LOAD-BEARING, not belt-and-braces. Every one of this predicate's ~79 call
# sites uses it to gate a `pointer(A)` + `unsafe_load`/`unsafe_store!` fast path, so what the callers
# actually need is "the elements are stored INLINE", which "unit stride" does not imply. A
# `Matrix{BigFloat}` satisfies both original clauses — it is a dense Array with `stride(A,1)==1` — while
# its buffer holds REFERENCES to heap-allocated mantissas. Taking `pointer` of it yields a Ptr into that
# reference array, and emitting a `pointerref` on a non-bits type SEGFAULTS JULIA'S CODE GENERATOR
# (`emit_pointerref`, intrinsics.cpp:804) — on 1.12.7 and 1.13.0-rc3 alike, with no Julia frames because
# it dies during compilation, not execution.
#
# It crashed even when the guarded call was never REACHED: `_trsm!` does `isone(α) || _scal_all!(B, α)`,
# and with α==1 that call never runs, but it is still compiled. So a runtime-dead branch was enough.
# Found via getrf!'s widening to `T<:Real` (BigFloat now reaches trsm); the hazard predates it.
#
# Zero cost: `eltype(A)` is statically known at every call site, so `isbitstype` const-folds. Nothing
# that works today changes — Float64/Float32/complex and ForwardDiff.Dual are all bits types.
@inline _strided1(A) = A isa StridedMatrix && isbitstype(eltype(A)) && stride(A, 1) == 1
# PtrMatrix wraps a raw pointer, so its elements are inline by construction.
@inline _strided1(::PtrMatrix) = true

# VECTOR analogue of `_strided1`, and it exists for the same reason — with a measured incident behind it.
#
# The BLAS-2 fast-path predicates (`_l2_simd_ok`, `_l2c_simd_ok`) ask `x isa StridedVector && stride(x,1)
# == 1`. `_strided1` was taught about `PtrMatrix` so the C-ABI's matrix argument passes; the VECTOR half
# was never taught about `Ptr`, and the C-ABI passes raw `Ptr{T}` vectors. A raw pointer IS unit-stride
# by construction — it is the densest possible representation — but `Ptr <: StridedVector` is false, so
# every C-ABI BLAS-2 call failed the guard and fell to the generic scalar loop.
#
# MEASURED (Zen5, dgemv trans='T', A = 400x200, the PureOSQP inner-loop shape):
#     OpenBLAS                                     10.40 us   1.00x
#     PureBLAS, native call (Matrix, Vector)        6.54 us   1.59x   <- kernel is FASTER than OB
#     PureBLAS, via activate()/LBT (PtrMatrix, Ptr) 104.08 us  0.10x   <- 16x slower, same kernel
# So an application using `activate()` got the scalar path for every gemv while the gate — which calls
# natively — saw the SIMD path and reported gemvT passing. No gate row could see it. Same class as the
# `izamax` C-ABI miss recorded in level1.jl: "a wire-the-fastest-path miss that no gate row could see".
@inline _dense1(x) = x isa StridedVector && stride(x, 1) == 1
@inline _dense1(::Ptr) = true

# ===== Container-type normalization for the gemm kernels =====
#
# The microkernels only ever consume `pointer`, `stride(·,2)` and the dims, but they were specialized on
# whatever container reached them. Twelve distinct types did — 6 shapes × 2 element types:
#   Matrix{T}; PtrMatrix{T}; SubArray{T,2,Matrix{T},(Slice,UnitRange),true};
#   SubArray{T,2,Matrix{T},(UnitRange,Slice),false}; SubArray{T,2,Matrix{T},(UnitRange,UnitRange),false};
#   ReshapedArray{T,2,SubArray{T,1,Vector{T},(UnitRange,),true},()}
# all of them strided column-major and semantically identical to the kernels. That fan-out — not the
# Val-parameterized tile geometry, whose every axis is only 2 wide — is what put 280 specializations on
# each of `_gemm_unpacked!`/`_split`/`_mr1` and 432 on `_gemm_cmplx_impl!`, and it drives BOTH `.text`
# and `.ldata` (the serialized MethodInstance/CodeInstance graph) in the pkgimage.
#
# `_pm` collapses them to one type per element type. Mode 1 already feeds PtrMatrix into these exact
# kernels, so after normalization Mode 1 and Mode 2 SHARE instances instead of duplicating the gemm
# universe per container.
@inline _pm(A::PtrMatrix) = A
@inline _pm(A::AbstractMatrix{T}) where {T} =
    PtrMatrix{T}(pointer(A), size(A, 1), size(A, 2), Int(stride(A, 2)))

# `_root` is the load-bearing half. The drivers root their operands with `parA = parent(A); GC.@preserve
# parA …`, but `parent(::PtrMatrix)` returns the struct ITSELF (isbits), so once a PtrMatrix is passed
# down those preserves become SILENT NO-OPS and nothing keeps the buffer alive. The preserve therefore
# has to move to the conversion site and root the real owner, which is what this peels to: an `Array`
# for Julia-owned memory, or the PtrMatrix itself when the buffer is caller/C-owned (preserving an
# isbits value is a legitimate no-op there — the C caller owns it across the call).
@inline _root(A::Array) = A
@inline _root(A::PtrMatrix) = A
@inline _root(A) = _root(parent(A))
