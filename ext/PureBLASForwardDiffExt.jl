# ForwardDiff extension: implements the public pair protocol (`pair_eltype`, `pair_nplanes`, `pair_parts`,
# `pair_make`; docs/src/dual.md) for `Dual{Tag,V,1}`. NO kernels live here: PureBLAS derives every routing
# predicate from these methods, and a third-party dual type can do the same without ForwardDiff. ForwardDiff
# is a weak dependency: the --trim build never loads this file and `libpureblas.so` is unaffected.
#
# `Tag` is never inspected. The only Duals the fast paths construct carry the vector's own eltype (so its
# tag), exactly as the scalar path does via `convert`; a mixed-tag alpha fails in `convert` on both paths.
# `N ≠ 1`, nested duals (`V` itself a Dual) and non-`BlasReal` `V` are not pair types and take the generic
# scalar loop unchanged.
module PureBLASForwardDiffExt

using PureBLAS: PureBLAS, BlasReal
using ForwardDiff: Dual, value, partials

PureBLAS.pair_eltype(::Type{Dual{T, V, 1}}) where {T, V <: BlasReal} = V
PureBLAS.pair_nplanes(::Type{Dual{T, V, 1}}) where {T, V} = 2
PureBLAS.pair_parts(d::Dual{T, V, 1}) where {T, V} = (value(d), partials(d, 1))
PureBLAS.pair_make(::Type{Dual{T, V, 1}}, parts::NTuple{2}) where {T, V} = Dual{T}(V(parts[1]), V(parts[2]))
# iamax compares |value| only (first occurrence on ties) for EVERY Dual, whatever N; see `_l1v` in src/core.jl.
PureBLAS._l1v(d::Dual) = abs(value(d))

end
