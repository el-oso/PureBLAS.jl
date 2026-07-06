# ── Level-3 / LAPACK scratch: one owned workspace per element type ──────────────────────────────────
# Replaces the former per-role global caches (_L3_TMP, _TRSM_TMP, _L3_APAD, _POTF2_BUF as abstract-Matrix
# IdDicts; _GEMM_SCRATCH, _CGEMM_SCRATCH, _SYR2K_SCRATCH as tuple Dicts). Those were the "loose global
# buffer" antipattern: the IdDict-of-abstract-Matrix ones went type-unstable and boxed on the returned
# view (a recurring bug), and every cache was independent global mutable state. Here all L3 scratch is
# bundled into ONE concrete-typed struct owned per element type — the buffer travels with its owner, à la
# PureFFT's plan-owned scratch (the ownership form of the Linux-kernel "the lock guards its data" rule).
#
# Buffers are concrete fields, grown on demand, reused across calls. Distinct fields per role preserve the
# old non-aliasing (e.g. the trsm base holds `diag`+`trsm_tmp` while a nested gemm holds `gpackA/B` — all
# separate). Access is const-dispatched for Float64/Float32 (the gated hot types) so it stays a bare field
# load with NO dict lookup — the ~130 ns IdDict lookup that const-dispatch was added to dodge (it costs
# more than a whole tiny trmm). Rare types fall back to a keyed lookup, exactly as before.
#
# Single global instance per hot type ⇒ single-thread only (the project's current mode; multithreading is
# deferred). M4 threading swaps _l3ws for a per-task/per-thread owner — the ~4 lines below, nothing else.

const _L3_NB = 128   # NB×NB diagonal-block scratch side (was in level3.jl); caps trmm/syrk materialize.

mutable struct L3Workspace{T}
    diag::Matrix{T}       # _l3_tmp:      fixed _L3_NB×_L3_NB diagonal-block scratch
    trtri::Matrix{T}      # _trtri_tmp:   blocked-trtri off-block gemm scratch (≤ _TRSM_BASE/2 square)
    trsm_tmp::Matrix{T}   # _trsm_tmp:    trsm invL/invR copyback temp (grows m×n)
    apad::Matrix{T}       # _l3_apad:     trsm po2-ld A-pad, ld=k+8 (grows)
    potf2::Matrix{T}      # _potf2_buf:   potrf diagonal-base contiguous buffer (grows n×n)
    gpackA::Vector{T}     # _gemm_scratch:      packed A panel
    gpackB::Vector{T}     # _gemm_scratch:      packed B panel
    cg::NTuple{4, Vector{T}}   # _gemm_scratch_cmplx: complex split-pack (2×A, 2×B)
    s2::NTuple{4, Vector{T}}   # _syr2k_scratch:      fused two-product (2×A, 2×B)
    m3::Vector{Matrix{T}}      # _gemm_3m_scratch:    Karatsuba 3M matrices (Ar/Ai/As, Br/Bi/Bs, P1/P2/P3)
    str::Vector{Matrix{T}}     # _strassen scratch:   pad (1-3) + per-level Winograd buffers (10/level)
end
L3Workspace{T}() where {T} = L3Workspace{T}(
    Matrix{T}(undef, _L3_NB, _L3_NB), Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0),
    Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0),
    T[], T[], (T[], T[], T[], T[]), (T[], T[], T[], T[]),
    [Matrix{T}(undef, 0, 0) for _ in 1:9],
    Matrix{T}[],
)

# Owner accessors. Const-dispatch the gated real types (bare field load, no lookup); a keyed fallback for
# everything else (complex/Dual/…), same cost as the old per-role Dicts.
const _L3WS_F64 = L3Workspace{Float64}()
const _L3WS_F32 = L3Workspace{Float32}()
const _L3WS_OTHER = IdDict{DataType, L3Workspace}()
@inline _l3ws(::Type{Float64}) = _L3WS_F64
@inline _l3ws(::Type{Float32}) = _L3WS_F32
_l3ws(::Type{T}) where {T} = get!(() -> L3Workspace{T}(), _L3WS_OTHER, T)::L3Workspace{T}

# Per-role accessors (unchanged signatures — call sites are untouched). Each returns/grows one owned field.
_l3_tmp(::Type{T}) where {T} = _l3ws(T).diag

function _trtri_tmp(::Type{T}, m::Int, n::Int) where {T}
    ws = _l3ws(T); b = ws.trtri
    if size(b, 1) < m || size(b, 2) < n
        b = Matrix{T}(undef, m, n); ws.trtri = b
    end
    return view(b, 1:m, 1:n)
end

function _trsm_tmp(::Type{T}, m::Int, n::Int) where {T}
    ws = _l3ws(T); b = ws.trsm_tmp
    if size(b, 1) < m || size(b, 2) < n
        b = Matrix{T}(undef, m, n); ws.trsm_tmp = b
    end
    return b
end

function _l3_apad(::Type{T}, k::Int) where {T}   # ld = k+8 (non-po2) to dodge cache-set aliasing
    ws = _l3ws(T); b = ws.apad
    if size(b, 1) < k + 8 || size(b, 2) < k
        b = Matrix{T}(undef, k + 8, k); ws.apad = b
    end
    return view(b, 1:k, 1:k)
end

function _potf2_buf(::Type{T}, n::Int) where {T}
    ws = _l3ws(T); b = ws.potf2
    if size(b, 1) < n
        b = Matrix{T}(undef, n, n); ws.potf2 = b
    end
    return view(b, 1:n, 1:n)
end

function _gemm_scratch(::Type{T}, lenA::Int, lenB::Int) where {T}
    ws = _l3ws(T)
    length(ws.gpackA) < lenA && resize!(ws.gpackA, lenA)
    length(ws.gpackB) < lenB && resize!(ws.gpackB, lenB)
    return ws.gpackA, ws.gpackB
end

function _gemm_scratch_cmplx(::Type{T}, lenA::Int, lenB::Int) where {T}
    t = _l3ws(T).cg
    length(t[1]) < lenA && (resize!(t[1], lenA); resize!(t[2], lenA))
    length(t[3]) < lenB && (resize!(t[3], lenB); resize!(t[4], lenB))
    return t
end

function _syr2k_scratch(::Type{T}, lenA::Int, lenB::Int) where {T}
    t = _l3ws(T).s2
    length(t[1]) < lenA && (resize!(t[1], lenA); resize!(t[3], lenA))
    length(t[2]) < lenB && (resize!(t[2], lenB); resize!(t[4], lenB))
    return t
end

# Karatsuba-3M complex-gemm scratch (REAL buffers): Ar/Ai/As (t1-3, len lenA), Br/Bi/Bs (t4-6, len lenB),
# P1/P2/P3 (t7-9, len lenC). Keyed on the REAL element type so it lives in the real workspace, disjoint
# from the sub-gemms' own gpackA/B. Grown on demand.
# Persistent 3M matrices, GROWN PER-DIM to the max shape seen (never shrink). The caller operates on the
# top-left (r,c) block via explicit dims + the matrix's own leading dim — so ztrsm's VARYING recursion
# shapes reuse the same buffers with ZERO per-call allocation (grow only while enlarging to the max).
function _gemm_3m_scratch(::Type{Tr}, ra::Int, ca::Int, rb::Int, cb::Int, m::Int, n::Int) where {Tr}
    t = _l3ws(Tr).m3
    @inline function grow!(i, r, c)
        M = t[i]
        (size(M, 1) < r || size(M, 2) < c) && (t[i] = Matrix{Tr}(undef, max(size(M, 1), r), max(size(M, 2), c)))
    end
    grow!(1, ra, ca); grow!(2, ra, ca); grow!(3, ra, ca)
    grow!(4, rb, cb); grow!(5, rb, cb); grow!(6, rb, cb)
    grow!(7, m, n);   grow!(8, m, n);   grow!(9, m, n)
    return t
end

# Strassen scratch pool (real). Slots 1-3: odd-n pad buffers (Ap mp×kp, Bp kp×np, Cp mp×np). Slots
# 4+: per-recursion-level Winograd buffers, 10 per level (TA mh×kh, TB kh×nh, P1..P7 + U all mh×nh) at
# base 3+level*10. Exact-sized (realloc on shape mismatch — negligible at the large n Strassen runs at).
@inline function _str_fit!(pool, i::Int, r::Int, c::Int, ::Type{Tr}) where {Tr}
    while length(pool) < i; push!(pool, Matrix{Tr}(undef, 0, 0)); end
    M = pool[i]; (size(M, 1) != r || size(M, 2) != c) && (pool[i] = Matrix{Tr}(undef, r, c))
    return pool[i]
end
function _strassen_pad_scratch(::Type{Tr}, mp::Int, kp::Int, np::Int) where {Tr}
    p = _l3ws(Tr).str
    return _str_fit!(p, 1, mp, kp, Tr), _str_fit!(p, 2, kp, np, Tr), _str_fit!(p, 3, mp, np, Tr)
end
function _strassen_lvl_scratch(::Type{Tr}, level::Int, mh::Int, nh::Int, kh::Int) where {Tr}
    p = _l3ws(Tr).str; b = 3 + level * 10
    TA = _str_fit!(p, b + 1, mh, kh, Tr); TB = _str_fit!(p, b + 2, kh, nh, Tr)
    P1 = _str_fit!(p, b + 3, mh, nh, Tr); P2 = _str_fit!(p, b + 4, mh, nh, Tr)
    P3 = _str_fit!(p, b + 5, mh, nh, Tr); P4 = _str_fit!(p, b + 6, mh, nh, Tr)
    P5 = _str_fit!(p, b + 7, mh, nh, Tr); P6 = _str_fit!(p, b + 8, mh, nh, Tr)
    P7 = _str_fit!(p, b + 9, mh, nh, Tr); U = _str_fit!(p, b + 10, mh, nh, Tr)
    return TA, TB, P1, P2, P3, P4, P5, P6, P7, U
end
