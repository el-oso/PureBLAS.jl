# ForwardDiff extension: tells PureBLAS that a dense `Dual{Tag,V,1}` vector is an interleaved pair buffer
# (docs/src/dual.md). Five one-line accessors and a layout assert — NO kernels live here. A `@generated`
# body runs at the world age of its own definition, so a codegen hook defined in an extension would be
# invisible inside PureBLAS's generators; every kernel is in src/, keyed on these accessors. ForwardDiff is
# a weak dependency: the --trim build never loads this file and `libpureblas.so` is unaffected.
#
# `Tag` is never inspected. The only Duals the fast paths construct carry the vector's own eltype (so its
# tag), exactly as the scalar path does via `convert`; a mixed-tag alpha fails in `convert` on both paths.
# `N ≠ 1`, nested duals (`V` itself a Dual) and non-`BlasReal` `V` miss every predicate here and take the
# generic scalar loop unchanged.
module PureBLASForwardDiffExt

using PureBLAS: PureBLAS, BlasReal
using ForwardDiff: Dual, value, partials

# THE LAYOUT FACT everything rests on: `Dual{T,V,1}` is `value::V` + `Partials{1,V}` wrapping an
# `NTuple{1,V}` — exactly the interleaved [v, p] pair `Complex{V}` is. Asserted at load so that a future
# ForwardDiff layout change is a load error, never silent garbage.
for V in (Float32, Float64)
    D = Dual{Nothing, V, 1}
    (isbitstype(D) && sizeof(D) == 2 * sizeof(V) && fieldoffset(D, 2) == sizeof(V)) ||
        error("PureBLASForwardDiffExt: Dual{Tag,$V,1} is no longer an interleaved (value, partial) pair; refusing to load the pair routing")
end

const _Dual1{V} = Dual{T, V, 1} where {T}
const _DualArg{V} = Union{Ptr{<:_Dual1{V}}, DenseArray{<:_Dual1{V}}}

PureBLAS._pairalg(::_DualArg{V}) where {V <: BlasReal} = true
PureBLAS._pairT(::Type{<:_Dual1{V}}) where {V <: BlasReal} = true
PureBLAS._pairvT(::Type{<:_Dual1{V}}) where {V} = V
PureBLAS._palg(::Type{<:_Dual1}) = Val(:dual)
PureBLAS._pairreal(x::Ptr{D}) where {V <: BlasReal, D <: _Dual1{V}} = Ptr{V}(x)
PureBLAS._pairreal(x::DenseArray{D}) where {V <: BlasReal, D <: _Dual1{V}} = Ptr{V}(pointer(x))
PureBLAS._parts(d::Dual{T, V, 1}) where {T, V} = (value(d), partials(d, 1))
PureBLAS._mkpair(::Type{Dual{T, V, 1}}, v, p) where {T, V} = Dual{T}(V(v), V(p))
# iamax compares |value| only (first occurrence on ties); see `_l1v` in src/core.jl for why.
PureBLAS._l1v(d::Dual) = abs(value(d))

end
