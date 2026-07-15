# Compact-WY block-reflector kernels (LAPACK dlarft/dlarfb roles) — P1a/P1b of PureSparse.jl's
# M5b multifrontal QR design (PureSparse repo, docs/design_qr_m5b.md §A7). Adapted from two
# already-proven-correct inline call sites: `qr.jl`'s `geqrf!` blocked trailing update (the 'T'
# direction, C := QᵀC) and `svd.jl`'s `_apply_reflectors_left!` (the 'N' direction, C := QC) —
# both compute the same triple-gemm compact-WY identity (T·W or Tᵀ·W, block order forward or
# reverse); extraction/generalization, not new derivation.
#
# CONVENTION: `tau` here is STANDARD LAPACK (H = I − tau·v·vᵀ, tau multiplies directly — T[c,c]
# = tau[c], no inversion), matching `svd.jl`'s native usage. `qr.jl`'s OWN stored `tau` array
# uses the inverted "faer" convention (H = I − v·vᵀ/tau, i.e. `tau_stored = 1/tau_LAPACK`, `Inf`
# for a trivial reflector — see `lapack_tests.jl`'s `geqrf` reconstruction test, which divides by
# `tau[kk]`, confirming this). A caller bridging from `qr_unblocked!`'s output must invert
# (`tau_lapack = isfinite(t) ? 1/t : 0.0`) before calling `wy_t!` — reconciliation is the
# CALLER'S job, kept out of these kernels so there is exactly one exposed convention (the P1
# requirement, design_qr_m5b.md §A7.1: "P1 must expose one documented convention").
#
# `Apanel` (both functions) must be an EXPLICIT unit-lower-trapezoidal copy of the reflector
# vectors — `Apanel[i,c] = i==c ? 1 : (i>c ? <essential value> : 0)` — not a raw view into a
# post-factorization matrix (whose diagonal/upper entries hold R values, not the implicit 1/0
# structure `VᵀV` needs). This matches what `geqrf!`/`_apply_reflectors_left!` already build
# internally (`Vv[i,c] = i==c ? 1.0 : (i>c ? A[...] : 0.0)`) — callers do the same 2-line copy
# using `WYApplyWorkspace.V`.

"""
    WYApplyWorkspace{T}(maxrows, maxbs, maxncols)

Caller-owned scratch for [`wy_t!`](@ref)/[`wy_apply!`](@ref), sized once per (max panel rows,
max block size, max trailing columns) and reused across calls — zero-allocation after
construction, matching PureSparse.jl's M5b numeric-loop contract (design_qr_m5b.md §A4.4/§A7.2).
"""
struct WYApplyWorkspace{T}
    V::Matrix{T}   # maxrows × maxbs: explicit-unit-diagonal reflector panel (the `Apanel` storage)
    G::Matrix{T}   # maxbs × maxbs: VᵀV scratch (wy_t!'s own use)
    W::Matrix{T}   # maxbs × maxncols: VᵀC / (T or Tᵀ)·W scratch
end

function WYApplyWorkspace{T}(maxrows::Int, maxbs::Int, maxncols::Int) where {T}
    return WYApplyWorkspace{T}(
        Matrix{T}(undef, max(maxrows, 1), max(maxbs, 1)),
        Matrix{T}(undef, max(maxbs, 1), max(maxbs, 1)),
        Matrix{T}(undef, max(maxbs, 1), max(maxncols, 1)),
    )
end

"""
    wy_t!(Tm, Apanel, tau, G) -> Tm

Compact-WY `T` factor (LAPACK dlarft role, forward/columnwise): given an explicit-unit-diagonal
reflector panel `Apanel` (m×bs — see module header) and its `tau` (length bs, standard LAPACK
convention), builds the bs×bs `T` such that the block reflector is `Q = I - V·T·Vᵀ`. `G` is
bs×bs scratch (`VᵀV`, upper triangle only is read back). `Tm` is written in FULL, including an
explicit-zero strict lower triangle, so `wy_apply!`/any `gemm!`-based consumer may read the
whole matrix safely.
"""
function wy_t!(Tm::AbstractMatrix{T}, Apanel::AbstractMatrix{T}, tau::AbstractVector{T},
        G::AbstractMatrix{T}) where {T}
    bs = length(tau)
    bs == 0 && return Tm
    m = size(Apanel, 1)
    Vv = view(Apanel, 1:m, 1:bs)
    Gv = view(G, 1:bs, 1:bs)
    syrk!(Gv, Vv; uplo = 'U', trans = 'T', alpha = true, beta = false)  # G = VᵀV, upper only
    @inbounds for c in 1:bs
        tc = tau[c]
        Tm[c, c] = tc
        for r in 1:(c - 1)
            s = zero(T)
            for kk in r:(c - 1)
                s = muladd(Tm[r, kk], Gv[kk, c], s)
            end
            Tm[r, c] = -tc * s
        end
        for r in (c + 1):bs
            Tm[r, c] = zero(T)
        end
    end
    return Tm
end

"""
    wy_apply!(trans, C, Apanel, Tm, ws) -> C

Block-reflector apply (LAPACK dlarfb role), ONE block: `C := Q·C` (`trans='N'`) or `C := Qᵀ·C`
(`trans='T'`), where `Q = I - V·Tm·Vᵀ` (`V` = `Apanel`, `Tm` from [`wy_t!`](@ref)). Multi-block
sweeps are the CALLER'S responsibility (loop panels forward for `'T'`, reverse for `'N'` — the
mirrors of `Qᵀ = H_k⋯H_1` / `Q = H_1⋯H_k`); this kernel is single-block by design (dissolves the
"reversed block order" concern into caller-side looping rather than an internal direction flag).
"""
function wy_apply!(trans::Char, C::AbstractMatrix{T}, Apanel::AbstractMatrix{T},
        Tm::AbstractMatrix{T}, ws::WYApplyWorkspace{T}) where {T}
    trans === 'N' || trans === 'T' || throw(ArgumentError("wy_apply!: trans must be 'N' or 'T', got $(repr(trans))"))
    m = size(Apanel, 1)
    bs = size(Apanel, 2)
    nc = size(C, 2)
    (bs == 0 || nc == 0 || m == 0) && return C
    Vv = view(Apanel, 1:m, 1:bs)
    Wv = view(ws.W, 1:bs, 1:nc)
    gemm!(Wv, Vv, C; transA = 'T', alpha = true, beta = false)                     # W = VᵀC
    trmm!(Wv, view(Tm, 1:bs, 1:bs); side = 'L', uplo = 'U', transA = trans)        # W := (T or Tᵀ)·W
    gemm!(C, Vv, Wv; alpha = -one(T), beta = one(T))                               # C -= V·W
    return C
end

"""
    qr_block_size(m::Int, n::Int) -> Int

Query PureBLAS's own derived QR panel width for an m×n problem (the `geqrf!`/`_QR_NB` sizing
logic, made queryable — design_qr_m5b.md §A7.1: PureSparse's frontal block width must be
PureBLAS-derived, never a PureSparse literal, mirroring `faer`'s own `recommended_block_size`
query as precedent, not spec). Float64 only (the tuned path); grows with the m×n cache footprint
via `_qr_nb` (`qr.jl`), base overridable via the `qr_nb` Preference.
"""
qr_block_size(m::Int, n::Int) = _qr_nb(Float64, m, n)
