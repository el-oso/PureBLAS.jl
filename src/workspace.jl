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

# NB×NB diagonal-block scratch side; caps trmm/syrk materialize (the _trmm_small! `_mat_tri!` M tile is
# re-read across all B columns → it must stay L2-resident: NB²·8 ≲ ¼·L2 ⇒ NB ≤ √(L2/32)). req#8: DERIVED as
# an L2-residency CLAMP — capped at 128 (the measured flat on-fleet optimum; trmm side-L NB∈{96,128,192} tie
# within noise on Zen4+Zen3, PB≥OB throughout — the base is an algorithm crossover that does NOT grow with L2),
# shrunk only when L2 can't hold the 128² F64 tile at ¼ occupancy. No-op on the fleet (galen 512K→√16384=128
# EXACT; Zen4/Zen5 1M→181→cap 128); a ≤256K-L2 box gets a smaller, still-fitting tile. `l3_nb` pref pins it.
const _L3_NB = @load_preference("l3_nb", clamp(_round_dn(isqrt(_L2_BYTES ÷ 32), 16), 16, 128))::Int

mutable struct L3Workspace{T}
    diag::Matrix{T}       # _l3_tmp:      fixed _L3_NB×_L3_NB diagonal-block scratch
    trtri::Matrix{T}      # _trtri_tmp:   blocked-trtri off-block gemm scratch (≤ _TRSM_BASE/2 square)
    trsm_tmp::Matrix{T}   # _trsm_tmp:    trsm invL/invR copyback temp (grows m×n)
    apad::Matrix{T}       # _l3_apad:     trsm po2-ld A-pad, ld=k+8 (grows)
    rpack::Matrix{T}      # _trsm_rpack:  side-R fused-leaf pT scratch, ODD ld (conflict-free re-reads; grows)
    ftrsm::Vector{T}      # _trsm_fused_buf: side-L gemmtrsm leaf packed row-major stripe P + recip (grows)
    potf2::Matrix{T}      # _potf2_buf:   potrf diagonal-base contiguous buffer (grows n×n)
    padf::Matrix{T}       # _potrf_pad:   potrf po2-ld whole-matrix pad, alias-free ld (grows (n+8|+16)×n)
    gpackA::Vector{T}     # _gemm_scratch:      packed A panel
    gpackB::Vector{T}     # _gemm_scratch:      packed B panel
    cg::NTuple{4, Vector{T}}   # _gemm_scratch_cmplx: complex split-pack (2×A, 2×B)
    s2::NTuple{4, Vector{T}}   # _syr2k_scratch:      fused two-product (2×A, 2×B)
    m3::NTuple{9, Vector{T}}   # _gemm_3m_scratch:    Karatsuba 3M buffers (Ar/Ai/As, Br/Bi/Bs, P1/P2/P3)
    str::Vector{Matrix{T}}     # _strassen scratch:   pad (1-3) + per-level Winograd buffers (10/level)
    cholpad::Matrix{T}    # _chol_pad:    faer potrf po2-ld whole-matrix pad, ld=n+8 (grows R×n)
    chold::Matrix{T}      # _chol_d:      faer potrf diag-block scratch, (_chol_block+8)×_chol_block
    cholt::Matrix{T}      # _chol_t:      faer potrf panel workspace, grows R×_chol_block
end
L3Workspace{T}() where {T} = L3Workspace{T}(
    Matrix{T}(undef, _L3_NB, _L3_NB), Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0),
    Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0), T[], Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0),
    T[], T[], (T[], T[], T[], T[]), (T[], T[], T[], T[]),
    (T[], T[], T[], T[], T[], T[], T[], T[], T[]),
    Matrix{T}[],
    Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0),
)

# Owner accessors. Const-dispatch (GKH ownership: bare field load, no lookup) EVERY gated hot type — the
# four BLAS element types s/d/c/z. The IdDict fallback is ONLY for the open-ended non-gated set
# (ForwardDiff.Dual & co. on the generic AD path), which can't have a compile-time owner and isn't a hot
# path. Complex was on the fallback before — that cost the complex L3 hot path a ~130 ns `get!` per call
# (and read as a runtime box to static alloc checks); const owners fix both.
const _L3WS_F64 = L3Workspace{Float64}()
const _L3WS_F32 = L3Workspace{Float32}()
const _L3WS_C64 = L3Workspace{ComplexF64}()
const _L3WS_C32 = L3Workspace{ComplexF32}()
const _L3WS_OTHER = IdDict{DataType, L3Workspace}()
@inline _l3ws(::Type{Float64}) = _L3WS_F64
@inline _l3ws(::Type{Float32}) = _L3WS_F32
@inline _l3ws(::Type{ComplexF64}) = _L3WS_C64
@inline _l3ws(::Type{ComplexF32}) = _L3WS_C32
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

# Side-R fused-leaf pT scratch with an ODD leading dim: an odd ld can never be a multiple of the (power-of-2)
# L1 way stride, so the leaf's solved-column re-reads are conflict-free — vs an in-place po2/way-stride ldb
# where they collide in one set. Grown odd-and-only (never shrinks) so the ld stays odd across reuse.
function _trsm_rpack(::Type{T}, rows::Int, cols::Int) where {T}
    ws = _l3ws(T); b = ws.rpack
    need = rows + 8; iseven(need) && (need += 1)      # odd ld ⇒ never a way-stride multiple
    if size(b, 1) < need || size(b, 2) < cols
        b = Matrix{T}(undef, need, cols); ws.rpack = b
    end
    return b
end

function _trsm_fused_buf(::Type{T}, len::Int) where {T}   # side-L gemmtrsm leaf: P stripe + recip (flat)
    ws = _l3ws(T); b = ws.ftrsm
    length(b) < len && (b = Vector{T}(undef, len); ws.ftrsm = b)
    return b
end

function _potf2_buf(::Type{T}, n::Int) where {T}
    ws = _l3ws(T); b = ws.potf2
    if size(b, 1) < n
        b = Matrix{T}(undef, n, n); ws.potf2 = b
    end
    return view(b, 1:n, 1:n)
end

# potrf whole-matrix pad: an alias-free leading dim (n+8, bumped +8 more if that ld would ITSELF land on the
# L1 quarter-way stride) so the generic potrf recursion's trailing trsm!/syrk! read the copy conflict-free.
function _potrf_pad(::Type{T}, n::Int) where {T}
    ws = _l3ws(T); b = ws.padf
    need = n + 8
    (need * sizeof(T)) % (_L1_WAY_BYTES >> 2) == 0 && (need += 8)   # keep the scratch's own ld off the way-stride
    if size(b, 1) < need || size(b, 2) < n
        b = Matrix{T}(undef, need, n); ws.padf = b
    end
    return b
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
# 3M scratch: GROW-ONLY flat buffers (Ar/Ai/As ≥ lenA, Br/Bi/Bs ≥ lenB, P1/P2/P3 ≥ lenC). The caller
# unsafe_wraps the first r·c elements as a CONTIGUOUS r×c matrix (ld=r) — NOT a max-ld top-left block:
# a persistent max-sized matrix would give small-n-after-large-n calls a huge leading dim ⇒ cache-hostile
# strided access (measured: zgemm/ztrsm at n=128 tank 1.16→0.57 once the buffer is grown to 2048). Grow-
# only avoids the MB-realloc churn of exact-sizing under ztrsm's varying recursion shapes.
function _gemm_3m_scratch(::Type{Tr}, lenA::Int, lenB::Int, lenC::Int) where {Tr}
    t = _l3ws(Tr).m3
    length(t[1]) < lenA && (resize!(t[1], lenA); resize!(t[2], lenA); resize!(t[3], lenA))
    length(t[4]) < lenB && (resize!(t[4], lenB); resize!(t[5], lenB); resize!(t[6], lenB))
    length(t[7]) < lenC && (resize!(t[7], lenC); resize!(t[8], lenC); resize!(t[9], lenC))
    return t
end

# Strassen scratch pool (real). Slots 1-3: odd-n pad buffers (Ap mp×kp, Bp kp×np, Cp mp×np). Slots
# 4+: per-recursion-level Winograd buffers, 10 per level (TA mh×kh, TB kh×nh, P1..P7 + U all mh×nh) at
# base 3+level*10. Exact-sized (realloc on shape mismatch — negligible at the large n Strassen runs at).
@inline function _str_fit!(pool, i::Int, r::Int, c::Int, ::Type{Tr}) where {Tr}
    while length(pool) < i
        push!(pool, Matrix{Tr}(undef, 0, 0))
    end
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
