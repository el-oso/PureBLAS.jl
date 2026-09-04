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
# shrunk only when L2 can't hold the 128² F64 tile at ¼ occupancy. No-op on the fleet (Zen3 512K→√16384=128
# EXACT; Zen4/Zen5 1M→181→cap 128); a ≤256K-L2 box gets a smaller, still-fitting tile. `l3_nb` pref pins it.
# PDM: Derived — formula over detected consts: `clamp(_round_dn(isqrt(_L2_BYTES ÷ 32), 16), 16, 128`
const _L3_NB = @load_preference("l3_nb", clamp(_round_dn(isqrt(_L2_BYTES ÷ 32), 16), 16, 128))::Int

# stein!'s deterministic inverse-iteration restart seed — stebz.jl:306's literal, lifted here so the
# constructor and `_stein_work`'s per-call reset cannot drift apart. NOT a tuning knob (no PDM tier): it
# is dstein's ISEED analogue, and its only property that matters is that it is the SAME every call.
# An RNG seed, not a machine-dependent tuning knob: nothing about the host can make one seed better than
# another, so there is no residency/latency criterion to derive and no candidate set to measure.
# req8-ok: RNG seed, host-invariant by construction — stebz.jl:306's value, kept identical.
const _STEIN_SEED0 = 0x2545f4914f6cdd1d

mutable struct L3Workspace{T}
    diag::Matrix{T}       # _l3_tmp:      fixed _L3_NB×_L3_NB diagonal-block scratch
    trtri::Matrix{T}      # _trtri_tmp:   blocked-trtri off-block gemm scratch (≤ _TRSM_BASE/2 square)
    trsm_tmp::Matrix{T}   # _trsm_tmp:    trsm invL/invR copyback temp (grows m×n)
    rpack::Matrix{T}      # _trsm_rpack:  side-R fused-leaf pT scratch, ODD ld (conflict-free re-reads; grows)
    rrefl::Matrix{T}      # _trsm_rrefl:  side-R NOTRANS reflected coefficients Ã=J·Aᵀ·J (lower), ODD ld (grows)
    ftrsm::Vector{T}      # _trsm_fused_buf: side-L gemmtrsm leaf packed row-major stripe P + recip (grows)
    potf2::Matrix{T}      # _potf2_buf:   potrf diagonal-base contiguous buffer (grows n×n)
    padf::Matrix{T}       # _potrf_pad:   potrf po2-ld whole-matrix pad, alias-free ld (grows (n+8|+16)×n)
    gpackA::Vector{T}     # _gemm_scratch:      packed A panel
    gpackB::Vector{T}     # _gemm_scratch:      packed B panel
    cg::NTuple{4, Vector{T}}   # _gemm_scratch_cmplx: complex split-pack (2×A, 2×B)
    s2::NTuple{4, Vector{T}}   # _syr2k_scratch:      fused two-product (2×A, 2×B)
    m3::NTuple{9, Vector{T}}   # _gemm_3m_scratch:    Karatsuba 3M buffers (Ar/Ai/As, Br/Bi/Bs, P1/P2/P3)
    str::Vector{Matrix{T}}     # _strassen scratch:   pad (1-3) + per-level Winograd buffers (10/level)
    strbt::Matrix{T}      # _strassen_bt: transB route Bᵀ, EXACT k×n. Deliberately NOT a `str` slot —
    # the nested Winograd recursion owns that whole pool, so sharing would alias.
    cholpad::Matrix{T}    # _chol_pad:    faer potrf po2-ld whole-matrix pad, ld=n+8 (grows R×n)
    chold::Matrix{T}      # _chol_d:      faer potrf diag-block scratch, (_chol_block+8)×_chol_block
    cholt::Matrix{T}      # _chol_t:      faer potrf panel workspace, grows R×_chol_block
    # ── The last two trsm roles, and the only reason they are not borrows yet: both are read by the
    # side-L / side-R fused leaves that the GATE scores, so converting them is a measurable change and
    # belongs with the rest of BLAS-3 (stage 4), behind a re-gate. Everything the arena has already taken
    # off this struct was gate-invisible; these are not.
    #
    # SEPARATE FROM `trsm_tmp` ON PURPOSE — do not "simplify" the two together. The side-L ragged
    # column-tail arm (level3.jl:4341) stages B into scratch, widens it to `_vwidth` columns, and holds
    # that view ACROSS `_trsm!`; the complex path inside then took `trsm_tmp` again for `_trtri!`'s
    # output (level3.jl:2017) and wrote A⁻¹ over the staged right-hand side. Measured wrong answers,
    # relerr ~1.0: ComplexF32, side='L', k=32/48, nrhs=5/6, transA ∈ {T,C}. Note nrhs=5/transA='T' was
    # CORRECT on the first call and wrong on the second — the corruption is history-dependent on the
    # buffer's grown size, which is why a single-call test never saw it.
    trsmw::Matrix{T}      # _trsm_widen: side-L ragged-tail staging, k × _vwidth
    # SEPARATE FROM `rpack` ON PURPOSE — same shape of bug as `trsmw` above, found by test/workspace_lint.jl.
    # The side-R pad arm (level3.jl:4045) copies A's lower triangle into scratch and holds that view across
    # `_trsm_rl_fused_drv!`, which takes `rpack` AGAIN as its own solve scratch (level3.jl:3929) — same
    # buffer, same base pointer, so the leaf writes the solution over the coefficients it is reading.
    # Aliases whenever the second claim does not realloc, i.e. `mc0 = (_L2_BYTES÷2)÷(k·8) ≤ k`: unconditional
    # on a 256 KiB-L2 AVX2 part at k=128; on Zen3 (512 KiB) only once an earlier call has grown the buffer
    # to ≥265 rows — history-dependent, right on call 1, wrong on call 2, exactly like `trsmw`.
    # The arm is AVX2-only (`_RL_MR_LIVE > _NVREG`), so no AVX-512 box can execute it; it was found by
    # READING, and the fix is the field split, not an audit.
    rpad::Matrix{T}       # _trsm_rpad: side-R pad-arm A copy, ODD ld (grows k×k)
end
# ONE ARGUMENT PER LINE, TAGGED WITH ITS FIELD. The list is positional, and it was ~180 entries long
# when the arena conversion started; the stage-1 pass mis-aligned it by two and shipped a
# `BoundsError: 0×0 Matrix at [1:6,1:6]`. The tags are the guard rail: they must read in the same
# order as the struct above.
L3Workspace{T}() where {T} = L3Workspace{T}(
    Matrix{T}(undef, _L3_NB, _L3_NB),            # diag
    Matrix{T}(undef, 0, 0),                      # trtri
    Matrix{T}(undef, 0, 0),                      # trsm_tmp
    Matrix{T}(undef, 0, 0),                      # rpack
    Matrix{T}(undef, 0, 0),                      # rrefl
    T[],                                         # ftrsm
    Matrix{T}(undef, 0, 0),                      # potf2
    Matrix{T}(undef, 0, 0),                      # padf
    T[],                                         # gpackA
    T[],                                         # gpackB
    (T[], T[], T[], T[]),                        # cg
    (T[], T[], T[], T[]),                        # s2
    (T[], T[], T[], T[], T[], T[], T[], T[], T[]), # m3
    Matrix{T}[],                                 # str
    Matrix{T}(undef, 0, 0),                      # strbt
    Matrix{T}(undef, 0, 0),                      # cholpad
    Matrix{T}(undef, 0, 0),                      # chold
    Matrix{T}(undef, 0, 0),                      # cholt
    Matrix{T}(undef, 0, 0),                      # trsmw
    Matrix{T}(undef, 0, 0),                      # rpad
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

# Side-R NOTRANS reflected-coefficient scratch. X·A=B (A lower, no transpose) is the column-reversal
# conjugate of the lower-TRANSPOSE recurrence the fused leaf already implements, so we hand the leaf
# Ã = J·Aᵀ·J (J = reversal; Ã is still lower) instead of a second hand-unrolled kernel. Cost is one
# O(k²/2) triangle copy, negligible against the O(m·k²) solve. ODD ld (as _trsm_rpack) so the leaf's
# coefficient-column walks stay conflict-free whatever A's own lda is. Dedicated field — must NOT share
# `apad` (trsm! can hold that across the recursion reaching this branch) or `rpack` (the dealias arm).
function _trsm_rrefl(::Type{T}, k::Int) where {T}
    ws = _l3ws(T); b = ws.rrefl
    need = k + 8; iseven(need) && (need += 1)
    if size(b, 1) < need || size(b, 2) < k
        b = Matrix{T}(undef, need, k); ws.rrefl = b
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


# Both blocked band-Cholesky kernels need two dense scratches per call: the corner work array W
# ((nb+1)×nb — LDWORK = nb+1 mirrors the reference's NBMAX+1, an odd leading dim so the column
# stride can't po2-alias) and the diagonal-block factor scratch S (nb×nb). GKH-owned rather than
# per-call `Matrix` allocations: at kd≈nb the whole factorization of a narrow band is microseconds,
# so two allocations per call are pure overhead, and both leading dimensions are load-bearing — the
# kernels build `PtrMatrix(…, nb+1)` / `PtrMatrix(…, nb)` views over them.
# The size tests are `!=`, NOT `>=`: a *larger* stale buffer has the WRONG ld, and handing the
# kernel a view whose ld disagrees with its PtrMatrix stride silently reads the wrong elements
# (this exact bug — a `<` row test leaking an old ld — cost pbtrf kd=64 a 2.10→1.81 regression).
# W is re-zeroed every call: the kernels write only the in-band triangle and rely on the rest being
# zero, and which part is "the rest" moves with ib/i3, so a reused buffer must not carry stale values.
# transB-Strassen Bᵀ buffer. EXACT k×n, not grow-only: `transpose!` requires an exact-shape dest, and
# keeping ld == k contiguous avoids the strided-top-left cache hazard the recursion's quadrant reads
# would otherwise hit. Realloc churn is negligible — this only fires at n ≥ _STRASSEN_MIN (1024), where
# a single allocation is a rounding error against 7 half-size matrix products.
function _strassen_bt(::Type{T}, k::Int, n::Int) where {T}
    ws = _l3ws(T)
    bt = ws.strbt
    if size(bt, 1) != k || size(bt, 2) != n
        bt = Matrix{T}(undef, k, n); ws.strbt = bt
    end
    return bt
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

# ── Solve / inverse / condition-estimate scratch accessors ─────────────────────────────────────────
# Same grow-on-demand, return-a-view shape as the L3 roles above. Each returns exactly the length the
# caller asked for, so the routine body is unchanged apart from where the buffer comes from.
#
# The REAL-typed buffers (`pstrfv`, `trrfsa`, `trrfst`) are reached through `_l3ws(real(T))`. That works
# because real(T) ∈ {Float64, Float32} is itself a const-dispatched owner, so it costs the same bare
# field load and needs no second type parameter on `L3Workspace`. For complex `T` it is a different
# owner object from the element-typed buffers, so a routine holding both can never alias them.










# Side-L ragged-column-tail staging for trsm (level3.jl:4341). Deliberately NOT `_trsm_tmp`: that arm
# holds its staged B across `_trsm!`, which reaches for `trsm_tmp` again on the complex path
# (level3.jl:2017) — see the `trsmw` field comment for the measured corruption.
function _trsm_widen(::Type{T}, m::Int, n::Int) where {T}
    ws = _l3ws(T); b = ws.trsmw
    if size(b, 1) < m || size(b, 2) < n
        b = Matrix{T}(undef, m, n); ws.trsmw = b
    end
    return b
end

# Side-R pad-arm A copy (level3.jl:4045). Deliberately NOT `_trsm_rpack`: that arm holds this copy across
# `_trsm_rl_fused_drv!`, which takes `rpack` again as its solve scratch — see the `rpad` field comment.
# Same odd-ld growth policy as `_trsm_rpack` (the leaf re-reads solved columns out of whichever buffer it
# is handed, so a way-stride multiple would put them all in one cache set).
function _trsm_rpad(::Type{T}, rows::Int, cols::Int) where {T}
    ws = _l3ws(T); b = ws.rpad
    need = rows + 8; iseven(need) && (need += 1)
    if size(b, 1) < need || size(b, 2) < cols
        b = Matrix{T}(undef, need, cols); ws.rpad = b
    end
    return b
end




# ── Grow-on-demand helper for the accessors below ───────────────────────────────────────────────────
# The 30-odd roles added for the eigen/QZ/LSQ families all want the same three lines the older accessors
# spell out by hand (`too small ⇒ fresh, else keep`). One helper, two methods, so a new role is a
# one-liner instead of a copy of the same `if size(b,1) < … ` block. Same policy as those accessors:
# GROW-ONLY, and a fresh buffer is `undef` — every zeroing that is load-bearing is done explicitly by
# the accessor that owns it (`fill!` calls below), never inherited from the allocator's spelling.
@inline _wsgrow(v::Vector{S}, n::Int) where {S} = length(v) < n ? Vector{S}(undef, n) : v
@inline _wsgrow(M::Matrix{S}, r::Int, c::Int) where {S} =
    (size(M, 1) < r || size(M, 2) < c) ? Matrix{S}(undef, r, c) : M

# ── Non-symmetric eigenproblem ──────────────────────────────────────────────────────────────────────









# STAGE 1: `_laexc_work` and `_dlasy2_work` are gone. Both handed out only FIXED-size buffers, so their
# bodies became `borrow!`s at their single call sites — `_dlaexc!` (trsen.jl) and `_syl_dlasy2`
# (trsyl.jl) — inside the `@scope` each of those now opens. The reasoning moved with them: D is borrowed
# EXACTLY nd×nd (the dnorm loop is `for x in D`, so an oversized 4×4 at nd==3 would fold stale elements
# into the `thresh` rejection test), and t16 keeps its load-bearing fill!.




# ── Bunch–Kaufman inverse ───────────────────────────────────────────────────────────────────────────


# ── Generalized eigenproblem / QZ ───────────────────────────────────────────────────────────────────

# STAGE 1: `_hgeqz_v` is gone — `_hgeqz!` borrows its length-3 shift vector from the arena (qz.jl). The
# separate-accessor reason it carried (bundling it with alphar/alphai would re-claim those from inside
# the entry's live claim) evaporates with the field: a borrow is per-scope, not per-role.




# STAGE 1: `_tgsy2_work` and the four `_tgex2_*` fixed-size accessors (blocks / qr / mul / copies / rot)
# are gone. Every buffer they handed out was FIXED (m ≤ 4, nz ≤ 8), so their bodies are now `borrow!`s at
# their single call sites in tgsen.jl — `_tgs_tgsy2!` for the Kronecker system, `_dtgex2_big!` for the
# rest — each inside the `@scope` that routine opens. The load-bearing zeroing travelled verbatim: Z, LI,
# IR, QL2 and IR2 are each `fill!`ed at their borrow, for the reasons still written at those sites.




# ── ggsvd! ──────────────────────────────────────────────────────────────────────────────────────────







# ── Least squares ───────────────────────────────────────────────────────────────────────────────────









# ── Symmetric-tridiagonal eigen (stebz.jl) ──────────────────────────────────────────────────────────


