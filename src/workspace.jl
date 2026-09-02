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

mutable struct L3Workspace{T}
    diag::Matrix{T}       # _l3_tmp:      fixed _L3_NB×_L3_NB diagonal-block scratch
    trtri::Matrix{T}      # _trtri_tmp:   blocked-trtri off-block gemm scratch (≤ _TRSM_BASE/2 square)
    trsm_tmp::Matrix{T}   # _trsm_tmp:    trsm invL/invR copyback temp (grows m×n)
    apad::Matrix{T}       # _l3_apad:     trsm po2-ld A-pad, ld=k+8 (grows)
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
    bandl::Matrix{T}      # _pbtrf_band:  pbtrf uplo='U' conj-transposed band re-pack, grows (kd+1)×n
    bandw::Matrix{T}      # _pbtrf_work:  pbtrf corner work array W, exactly (nb+1)×nb (ld is load-bearing)
    bands::Matrix{T}      # _pbtrf_work:  pbtrf dense diagonal-block scratch, exactly nb×nb (ld load-bearing)
    gbw13::Matrix{T}      # _gbtrf_work:  gbtrf upper-corner WORK13, exactly (nb+1)×nb (ld load-bearing)
    gbw31::Matrix{T}      # _gbtrf_work:  gbtrf lower-corner WORK31, exactly (nb+1)×nb (ld load-bearing)
    gbs::Matrix{T}        # _gbtrf_work:  gbtrf dense L11 staging scratch, exactly nb×nb (ld load-bearing)
    sylw::Matrix{T}       # _sytrf_work:  dlasyf panel W (n×nb) + gather col; exact rows, grow-only cols
    # ── Solve/inverse/condition scratch. These roles used to allocate per call inside their routines,
    # which broke the `!` in-place promise: gecon!/trcon!/pocon! 352 B, potri!/getri! 592 B, pstrf! 640 B,
    # trrfs! 1056 B (measured at n=8). Owned here like every other L3 buffer, they reach 0 alloc without
    # the per-call lookup a task-local cache would add (+7.5 ns, which is 7.4% of a trmm! at n=8).
    lacnx::Vector{T}      # _lacn_bufs:  Higham–Hager 1-norm estimator, candidate x (grows n)
    lacnv::Vector{T}      # _lacn_bufs:  Higham–Hager 1-norm estimator, previous iterate v (grows n)
    lacnsgn::Vector{Int}  # _lacn_bufs:  real-only sign-repeat test vector (grows n; empty for complex)
    lauumt::Matrix{T}     # _lauum_tmp:  lauum dense-base zeroed triangle copy (≤ _trtri_base square)
    getriw::Matrix{T}     # _getri_work: getri blocked inversion panel W (grows n×nb)
    pstrfv::Vector{T}     # _pstrf_work: running dot products + scratch, 2n. REAL-typed, so it is reached
                          # as `_l3ws(real(T)).pstrfv` — for complex A that is a DIFFERENT owner object
                          # than `pstrfs` below, so the two can never alias.
    pstrfs::Vector{T}     # _pstrf_work: blocked panel scratch (nb + 2n), element-typed
    pstrfsw::Vector{Int}  # _pstrf_swaps: blocked pivot bookkeeping swj/swp, 2·nb as two DISJOINT views
    trrfsr::Vector{T}     # _trrfs_work: residual r = op(A)·x − b (grows n), element-typed
    trrfsa::Vector{T}     # _trrfs_work: |op(A)|·|x| + |b|  — REAL-typed, via `_l3ws(real(T))`
    trrfst::Vector{T}     # _trrfs_work: LACN2 weight |r| + nz·eps·wabs — REAL-typed, via `_l3ws(real(T))`
    # The SECOND Higham–Hager estimator. `_lacn2_estimate` (trsen.jl) is a separate implementation of the
    # same algorithm as `_lacn2!` (gecon.jl), with different callers: trrfs! (trrfs.jl:141) and trsen!
    # (trsen.jl:555, :614) vs gecon!/trcon!/pocon!. (trsyl! does NOT call it, despite the neighbouring
    # file.) They get DISTINCT fields rather than sharing `lacn*` — proving the two can never be live at
    # once would mean auditing every path through trsen/trrfs, and the cost of being
    # wrong is silent numerical corruption. Three Vectors is cheap insurance. (Deduplicating the two
    # estimators is the real fix and is left as separate work.) Note `lacn2s` is length n for BOTH real
    # and complex here, unlike `lacnsgn` — `_lacn2_estimate` zeros a full-length isgn either way.
    lacn2x::Vector{T}     # _lacn2e_bufs: candidate x
    lacn2v::Vector{T}     # _lacn2e_bufs: previous iterate v
    lacn2s::Vector{Int}   # _lacn2e_bufs: sign vector, always length n
    # pptrf blocked packed Cholesky. These allocated per call and so broke `pptrf!`'s `!` promise for
    # every n ≥ _PPTRF_BLK_MIN (16) — 16,560 B at n=32, growing with n. It measured 0 B at n=8 only
    # because that is BELOW the blocking threshold and takes the unblocked arm, which is exactly how the
    # bug survived an allocation audit: pick a size on the other side of every threshold.
    pptrfw::Matrix{T}     # _pptrf_lower_work: lower panel W, n×nb
    pptrfv::Matrix{T}     # _pptrf_{lower,upper}_work: trailing-update V, n×nb (shared — one uplo per call)
    pptrfr::Matrix{T}     # _pptrf_upper_work: upper packed-row block R, nb×n (note the transposed shape)
    gerfsr::Vector{T}     # _gerfs_work: residual r = b − op(A)·x (grows n), element-typed
    gerfsw::Vector{T}     # _gerfs_work: |b| + |op(A)|·|x| — REAL-typed, via `_l3ws(real(T))`
    gerfsd::Vector{T}     # _gerfs_work: refinement correction dx (grows n), element-typed
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
    # geqp3 (column-pivoted QR). Six buffers, allocated per call before this: 384 B at n=8 rising to
    # 12,952 B at n=129. Worth noting HOW that was found — the audit measured five sizes straddling the
    # blocking threshold, because `vn1`/`vn2`/`hbuf` are allocated on every path while `F`/`auxv`/`wrow`
    # appear only once `nb > 1`. A single-size check sees at most half the leak, which is precisely the
    # trap that hid pptrf!'s blocked arm at n=8.
    qp3vn1::Vector{T}     # _geqp3_norms_work: downdated partial column 2-norms — REAL, via _l3ws(real(T))
    qp3vn2::Vector{T}     # _geqp3_norms_work: reference norm at last exact recompute — REAL
    qp3hbuf::Vector{T}    # _geqp3_norms_work: contiguous heads for the SIMD downdate — REAL
    qp3f::Matrix{T}       # _geqp3_panel_work: laqps block reflector F, n × nb (blocked arm only)
    qp3auxv::Vector{T}    # _geqp3_panel_work: laqps auxiliary vector, length nb (blocked arm only)
    qp3wrow::Vector{T}    # _geqp3_panel_work: pivot-row delta buffer, length n (blocked arm only)
    # ormtr/unmtr blocked compact-WY scratch (eigen.jl). `_ormtr!`/`_unmtr!` built a fresh
    # `WYApplyWorkspace{T}(n, nb, nc)` PLUS an nb×nb T factor (and, complex, a second nb×nb Gram) on every
    # call — the scratch half of the orgtr!/ungtr! leak (~9.6 KB real / ~20 KB complex at n=64, on top of
    # the returned Q). Four fields, one per role: `mtrv` is the only one that grows with n, the other three
    # are bounded by nb (= `_qr_nb`) and nc. Grow-only; every element read is written first on the same
    # call (the panel copy fills V[1:m,1:pb]; syrk!/herk! and gemm! all run with beta=false), so NO `fill!`
    # is needed — the old code used `Matrix{T}(undef, …)` throughout, never `zeros`.
    mtrv::Matrix{T}       # _ormtr_work: explicit unit-lower-trapezoid reflector panel V (grows n×nb)
    mtrg::Matrix{T}       # _ormtr_work: VᵀV / VᴴV Gram scratch for wy_t!/_wy_t_cplx! (grows nb×nb)
    mtrw::Matrix{T}       # _ormtr_work: VᵀC then (T or Tᵀ)·W apply scratch (grows nb×nc)
    # `mtrt` is a DISTINCT field from `mtrg`, not a share: both are live at once inside wy_t!/_wy_t_cplx!,
    # which builds T by reading G.
    mtrt::Matrix{T}       # _ormtr_work: compact-WY T factor (grows nb×nb)
end
L3Workspace{T}() where {T} = L3Workspace{T}(
    Matrix{T}(undef, _L3_NB, _L3_NB), Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0),
    Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0),
    T[], Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0),
    T[], T[], (T[], T[], T[], T[]), (T[], T[], T[], T[]),
    (T[], T[], T[], T[], T[], T[], T[], T[], T[]),
    Matrix{T}[], Matrix{T}(undef, 0, 0),
    Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0),
    Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0),
    Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0),
    Matrix{T}(undef, 0, 0),
    T[], T[], Int[],                                 # lacnx, lacnv, lacnsgn
    Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0),  # lauumt, getriw
    T[], T[], Int[],                                 # pstrfv, pstrfs, pstrfsw
    T[], T[], T[],                                   # trrfsr, trrfsa, trrfst
    T[], T[], Int[],                                 # lacn2x, lacn2v, lacn2s
    Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0),  # pptrfw, pptrfv, pptrfr
    T[], T[], T[],                                   # gerfsr, gerfsw, gerfsd
    Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0),  # trsmw, rpad
    T[], T[], T[],                                   # qp3vn1, qp3vn2, qp3hbuf
    Matrix{T}(undef, 0, 0), T[], T[],                # qp3f, qp3auxv, qp3wrow
    Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0),  # mtrv, mtrg
    Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0),  # mtrw, mtrt
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

# pbtrf uplo='U' band re-pack target. OWNED rather than freshly allocated: at kd=256, n=4096 this buffer
# is 8.4 MB, and allocating it per call made PureBLAS's own runtime vary 22.8% run-to-run for identical
# input inside one process (the reference varied 1.3%) — GC noise larger than the gap being tuned, which
# made the cell unmeasurable before it made it slow.
# NOTE the row test is `!=`, not `<`: the re-pack's leading dimension IS kd+1, so a buffer kept from a
# WIDER band must not be reused by viewing a sub-block of it — that would hand the blocked kernel an ld
# from the previous call. Measured: reusing a kd=256 buffer for kd=64 dropped that cell from 2.10 to 1.81.
function _pbtrf_band(::Type{T}, kd::Int, n::Int) where {T}
    ws = _l3ws(T); b = ws.bandl
    if size(b, 1) != kd + 1 || size(b, 2) < n
        b = Matrix{T}(undef, kd + 1, n); ws.bandl = b
    end
    return view(b, :, 1:n)                             # ld == kd+1 by construction
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

function _pbtrf_work(::Type{T}, nb::Int) where {T}
    ws = _l3ws(T)
    w = ws.bandw
    if size(w, 1) != nb + 1 || size(w, 2) != nb
        w = Matrix{T}(undef, nb + 1, nb); ws.bandw = w
    end
    fill!(w, zero(T))
    s = ws.bands
    if size(s, 1) != nb || size(s, 2) != nb
        s = Matrix{T}(undef, nb, nb); ws.bands = s
    end
    return w, s
end

# Blocked dgbtrf needs three dense scratches per call. Two are the reference's WORK13/WORK31: the
# band's two CORNER blocks A13 and A31 fall outside the stored ldab window — A31[ii,jj] sits at
# storage row kv+kl+1+ii-jj, which is ≤ ldab only for ii ≤ jj, and A13 symmetrically only for
# ii ≥ jj — yet both fill in completely during the panel (A31 through the pivot swaps, A13 through
# the trsm). So each is staged dense, exactly as dgbtrf does. The third, S, stages the unit-lower
# L11 block: the strictly-upper triangle of an ld-1 band view anchored at AB(kv+1,j) IS U11, live
# matrix data, and handing that to a triangular kernel is the aliasing bug banded_chol.jl:420-428
# records costing a 4e-4 factor error. jb²/2 copies per panel is O(n·nb/2) total — noise.
#
# Shapes mirror the reference (LDWORK = NBMAX+1): W13/W31 are (nb+1)×nb, S is nb×nb. Size tests are
# `!=`, NOT `>=`, for the same reason as `_pbtrf_work`: the kernels build `PtrMatrix(…, nb+1)` views
# over these, so a *larger* stale buffer carries the WRONG ld and silently reads wrong elements.
# W13/W31 are re-zeroed each call — their off-triangles must stay zero across panels, and which part
# is "the off-triangle" moves with i3/j3.
# Deliberately NOT sharing pbtrf's bandw/bands: the two routines tune to different nb, so alternating
# them would reallocate on every call under the exact-size test.
function _gbtrf_work(::Type{T}, nb::Int) where {T}
    ws = _l3ws(T)
    w13 = ws.gbw13
    if size(w13, 1) != nb + 1 || size(w13, 2) != nb
        w13 = Matrix{T}(undef, nb + 1, nb); ws.gbw13 = w13
    end
    fill!(w13, zero(T))
    w31 = ws.gbw31
    if size(w31, 1) != nb + 1 || size(w31, 2) != nb
        w31 = Matrix{T}(undef, nb + 1, nb); ws.gbw31 = w31
    end
    fill!(w31, zero(T))
    s = ws.gbs
    if size(s, 1) != nb || size(s, 2) != nb
        s = Matrix{T}(undef, nb, nb); ws.gbs = s
    end
    return w13, w31, s
end

# dlasyf's panel W (n×nb, accumulating L21·D) plus one extra column used as the contiguous gather
# buffer for the panel gemv's strided W row (`_l2_simd_ok` needs incx==1).
# EXACT on rows, grow-only on columns. The stated rule permits grow-only throughout here — every
# consumer is a `view` carrying its own stride and no correctness depends on ldw — but there is a
# second, PERF reason to pin the rows: `_gemm_3m_scratch` (:220-224) records a grow-only buffer that
# retained a large leading dimension tanking zgemm/ztrsm at n=128 from 1.16 to 0.57, because the
# small-n view then walks a huge stride. W's columns are the trailing gemm's B operand and the panel
# gemv's y, both stride-sensitive, so the rows get the exact test. nb is bounded and monotone in n,
# so the column test rarely fires. No `fill!`: every element read is written by the copy-then-gemv in
# the same step. The W view is handed out at exactly nb columns so a panel-boundary bug trips
# @boundscheck in a debug build rather than silently reading the gather column.
function _sytrf_work(::Type{T}, n::Int, nb::Int) where {T}
    ws = _l3ws(T); b = ws.sylw
    ld = n + 8                                          # off the way stride (same lore as _l3_apad)
    (ld * sizeof(T)) % (_L1_WAY_BYTES >> 2) == 0 && (ld += 8)
    if size(b, 1) != ld || size(b, 2) < nb + 1
        b = Matrix{T}(undef, ld, nb + 1); ws.sylw = b
    end
    return view(b, 1:n, 1:nb), view(b, 1:nb, nb + 1)
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

# Higham–Hager 1-norm estimator scratch: candidate x, previous iterate v, and (real only) sign vector.
function _lacn_bufs(::Type{T}, n::Int) where {T}
    ws = _l3ws(T)
    length(ws.lacnx) < n && (ws.lacnx = Vector{T}(undef, n))
    length(ws.lacnv) < n && (ws.lacnv = Vector{T}(undef, n))
    ns = T <: Real ? n : 0
    length(ws.lacnsgn) < ns && (ws.lacnsgn = Vector{Int}(undef, ns))
    return view(ws.lacnx, 1:n), view(ws.lacnv, 1:n), view(ws.lacnsgn, 1:ns)
end

# lauum dense-base scratch: an n×n zeroed copy of the stored triangle. n ≤ _trtri_base() by construction.
function _lauum_tmp(::Type{T}, n::Int) where {T}
    ws = _l3ws(T); b = ws.lauumt
    if size(b, 1) < n || size(b, 2) < n
        b = Matrix{T}(undef, n, n); ws.lauumt = b
    end
    return view(b, 1:n, 1:n)
end

# getri blocked-inversion panel W (n×nb).
function _getri_work(::Type{T}, n::Int, nb::Int) where {T}
    ws = _l3ws(T); b = ws.getriw
    if size(b, 1) < n || size(b, 2) < nb
        b = Matrix{T}(undef, n, nb); ws.getriw = b
    end
    return view(b, 1:n, 1:nb)
end

# pstrf scratch: `work` is REAL (running dot products, length 2n) and lives on the real owner; `scr` is
# element-typed (length nb + 2n). Distinct owners for complex T ⇒ no aliasing.
function _pstrf_work(::Type{T}, n::Int) where {T}
    R = real(T); ws = _l3ws(R)
    length(ws.pstrfv) < 2n && (ws.pstrfv = Vector{R}(undef, 2n))
    return view(ws.pstrfv, 1:(2n))
end
function _pstrf_scr(::Type{T}, len::Int) where {T}
    ws = _l3ws(T)
    length(ws.pstrfs) < len && (ws.pstrfs = Vector{T}(undef, len))
    return view(ws.pstrfs, 1:len)
end

# trrfs scratch: residual r (element-typed) plus two REAL accumulators on the real owner.
function _trrfs_work(::Type{T}, n::Int) where {T}
    R = real(T); ws = _l3ws(T); wr = _l3ws(R)
    length(ws.trrfsr) < n && (ws.trrfsr = Vector{T}(undef, n))
    length(wr.trrfsa) < n && (wr.trrfsa = Vector{R}(undef, n))
    length(wr.trrfst) < n && (wr.trrfst = Vector{R}(undef, n))
    return view(ws.trrfsr, 1:n), view(wr.trrfsa, 1:n), view(wr.trrfst, 1:n)
end

# pstrf blocked pivot bookkeeping: swj/swp, both length nb. One owned Int buffer, returned as two
# DISJOINT views (1:nb and nb+1:2nb) so they cannot alias each other.
function _pstrf_swaps(::Type{T}, nb::Int) where {T}
    ws = _l3ws(T)
    length(ws.pstrfsw) < 2nb && (ws.pstrfsw = Vector{Int}(undef, 2nb))
    return view(ws.pstrfsw, 1:nb), view(ws.pstrfsw, (nb + 1):(2nb))
end

# Scratch for `_lacn2_estimate` (trsen.jl) — the second Higham–Hager estimator. Distinct fields from
# `_lacn_bufs`; see the struct comment. The caller MUST initialise: unlike the old `fill`/`zeros` these
# buffers carry the previous call's values.
function _lacn2e_bufs(::Type{V}, n::Int) where {V}
    ws = _l3ws(V)
    length(ws.lacn2x) < n && (ws.lacn2x = Vector{V}(undef, n))
    length(ws.lacn2v) < n && (ws.lacn2v = Vector{V}(undef, n))
    length(ws.lacn2s) < n && (ws.lacn2s = Vector{Int}(undef, n))
    return view(ws.lacn2x, 1:n), view(ws.lacn2v, 1:n), view(ws.lacn2s, 1:n)
end

# gerfs iterative-refinement scratch: residual r and correction dx (element-typed), plus the real
# accumulator |b| + |op(A)|·|x| on the real owner. Three distinct fields ⇒ no aliasing.
function _gerfs_work(::Type{T}, n::Int) where {T}
    R = real(T); ws = _l3ws(T); wr = _l3ws(R)
    length(ws.gerfsr) < n && (ws.gerfsr = Vector{T}(undef, n))
    length(wr.gerfsw) < n && (wr.gerfsw = Vector{R}(undef, n))
    length(ws.gerfsd) < n && (ws.gerfsd = Vector{T}(undef, n))
    return view(ws.gerfsr, 1:n), view(wr.gerfsw, 1:n), view(ws.gerfsd, 1:n)
end

# pptrf blocked packed-Cholesky panels. Two shapes: the LOWER arm wants two n×nb blocks (W, V); the
# UPPER arm wants an nb×n packed-row block (R) plus the same n×nb V. Only one uplo runs per call and the
# blocked helpers do not nest, so V is shared between them; W and R are separate because their shapes are
# transposes of each other and sharing would force a reallocation on every uplo switch.
function _pptrf_lower_work(::Type{T}, n::Int, nb::Int) where {T}
    ws = _l3ws(T)
    (size(ws.pptrfw, 1) < n || size(ws.pptrfw, 2) < nb) && (ws.pptrfw = Matrix{T}(undef, n, nb))
    (size(ws.pptrfv, 1) < n || size(ws.pptrfv, 2) < nb) && (ws.pptrfv = Matrix{T}(undef, n, nb))
    return view(ws.pptrfw, 1:n, 1:nb), view(ws.pptrfv, 1:n, 1:nb)
end
function _pptrf_upper_work(::Type{T}, n::Int, nb::Int) where {T}
    ws = _l3ws(T)
    (size(ws.pptrfr, 1) < nb || size(ws.pptrfr, 2) < n) && (ws.pptrfr = Matrix{T}(undef, nb, n))
    (size(ws.pptrfv, 1) < n || size(ws.pptrfv, 2) < nb) && (ws.pptrfv = Matrix{T}(undef, n, nb))
    return view(ws.pptrfr, 1:nb, 1:n), view(ws.pptrfv, 1:n, 1:nb)
end

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

# geqp3 column-norm tracking. All three are REAL-typed, so they live on `_l3ws(real(T))` — the same
# const-dispatched owner, no second type parameter needed. Allocated on EVERY geqp3! path.
function _geqp3_norms_work(::Type{T}, n::Int) where {T}
    R = real(T); ws = _l3ws(R)
    length(ws.qp3vn1) < n && (ws.qp3vn1 = Vector{R}(undef, n))
    length(ws.qp3vn2) < n && (ws.qp3vn2 = Vector{R}(undef, n))
    length(ws.qp3hbuf) < n && (ws.qp3hbuf = Vector{R}(undef, n))
    return view(ws.qp3vn1, 1:n), view(ws.qp3vn2, 1:n), view(ws.qp3hbuf, 1:n)
end

# geqp3 blocked panel (laqps). Reached only when `nb > 1`, which is why the allocation was invisible
# below the blocking threshold. Element-typed, so a different owner from the REAL norms above for
# complex T — they cannot alias.
function _geqp3_panel_work(::Type{T}, n::Int, nb::Int) where {T}
    ws = _l3ws(T)
    (size(ws.qp3f, 1) < n || size(ws.qp3f, 2) < nb) && (ws.qp3f = Matrix{T}(undef, n, nb))
    length(ws.qp3auxv) < nb && (ws.qp3auxv = Vector{T}(undef, nb))
    length(ws.qp3wrow) < n && (ws.qp3wrow = Vector{T}(undef, n))
    return view(ws.qp3f, 1:n, 1:nb), view(ws.qp3auxv, 1:nb), view(ws.qp3wrow, 1:n)
end

# ormtr/unmtr blocked back-transform scratch: V (n×nb), G (nb×nb), W (nb×nc), T factor (nb×nb).
# Returns the WHOLE buffers, not views: `V`/`G`/`W` become the fields of a `WYApplyWorkspace{T}` (wy.jl,
# included AFTER this file, so it cannot be an `L3Workspace` field type) and `wy_t!`/`wy_apply!` already
# take their own `view(…, 1:bs, …)` sub-blocks. Grow-only, and the three nb-sized buffers only ever grow
# to `_qr_nb`'s bounded width, so the leading-dimension drift `_gemm_3m_scratch` warns about is capped.
# No `fill!`: the panel copy writes V[1:m,1:pb] entirely, wy_t! writes the full bs×bs T (zeroed strict
# lower included), and every G/W producer is a beta=false syrk!/herk!/gemm!.
function _ormtr_work(::Type{T}, n::Int, nb::Int, nc::Int) where {T}
    ws = _l3ws(T)
    (size(ws.mtrv, 1) < n || size(ws.mtrv, 2) < nb) && (ws.mtrv = Matrix{T}(undef, n, nb))
    (size(ws.mtrg, 1) < nb || size(ws.mtrg, 2) < nb) && (ws.mtrg = Matrix{T}(undef, nb, nb))
    (size(ws.mtrw, 1) < nb || size(ws.mtrw, 2) < nc) && (ws.mtrw = Matrix{T}(undef, nb, nc))
    (size(ws.mtrt, 1) < nb || size(ws.mtrt, 2) < nb) && (ws.mtrt = Matrix{T}(undef, nb, nb))
    return ws.mtrv, ws.mtrg, ws.mtrw, ws.mtrt
end
