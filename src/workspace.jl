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

# BLAS3Workspace — 19 roles. Split out of the former 178-field L3Workspace so
# each ALGORITHM FAMILY owns its own scratch and can carry its own M4 sharing decision
# (kb: workspace-mt-ownership-per-buffer-not-uniform). Field lines and their comments are
# unchanged from the original struct.
mutable struct BLAS3Workspace{T}
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

BLAS3Workspace{T}() where {T} = BLAS3Workspace{T}(
    Matrix{T}(undef, _L3_NB, _L3_NB),       # diag
    Matrix{T}(undef, 0, 0),                 # trtri
    Matrix{T}(undef, 0, 0),                 # trsm_tmp
    Matrix{T}(undef, 0, 0),                 # apad
    Matrix{T}(undef, 0, 0),                 # rpack
    Matrix{T}(undef, 0, 0),                 # rrefl
    T[],                                    # ftrsm
    Matrix{T}(undef, 0, 0),                 # potf2
    Matrix{T}(undef, 0, 0),                 # padf
    T[],                                    # gpackA
    T[],                                    # gpackB
    (T[], T[], T[], T[]),                   # cg
    (T[], T[], T[], T[]),                   # s2
    (T[], T[], T[], T[], T[], T[], T[], T[], T[]),# m3
    Matrix{T}[],                            # str
    Matrix{T}(undef, 0, 0),                 # strbt
    Matrix{T}(undef, 0, 0),                 # cholpad
    Matrix{T}(undef, 0, 0),                 # chold
    Matrix{T}(undef, 0, 0),                 # cholt
    Matrix{T}(undef, 0, 0),                 # trsmw
    Matrix{T}(undef, 0, 0),                 # rpad
)

const _BLAS3WS_F64 = BLAS3Workspace{Float64}()
const _BLAS3WS_F32 = BLAS3Workspace{Float32}()
const _BLAS3WS_C64 = BLAS3Workspace{ComplexF64}()
const _BLAS3WS_C32 = BLAS3Workspace{ComplexF32}()
const _BLAS3WS_OTHER = IdDict{DataType, BLAS3Workspace}()
@inline _blas3ws(::Type{Float64}) = _BLAS3WS_F64
@inline _blas3ws(::Type{Float32}) = _BLAS3WS_F32
@inline _blas3ws(::Type{ComplexF64}) = _BLAS3WS_C64
@inline _blas3ws(::Type{ComplexF32}) = _BLAS3WS_C32
_blas3ws(::Type{T}) where {T} = get!(() -> BLAS3Workspace{T}(), _BLAS3WS_OTHER, T)::BLAS3Workspace{T}

####################################################################################################

# EigWorkspace — 36 roles. Split out of the former 178-field L3Workspace so
# each ALGORITHM FAMILY owns its own scratch and can carry its own M4 sharing decision
# (kb: workspace-mt-ownership-per-buffer-not-uniform). Field lines and their comments are
# unchanged from the original struct.
mutable struct EigWorkspace{T}
    # ── Solve/inverse/condition scratch. These roles used to allocate per call inside their routines,
    # which broke the `!` in-place promise: gecon!/trcon!/pocon! 352 B, potri!/getri! 592 B, pstrf! 640 B,
    # trrfs! 1056 B (measured at n=8). Owned here like every other L3 buffer, they reach 0 alloc without
    # the per-call lookup a task-local cache would add (+7.5 ns, which is 7.4% of a trmm! at n=8).
    lacnx::Vector{T}      # _lacn_bufs:  Higham–Hager 1-norm estimator, candidate x (grows n)
    lacnv::Vector{T}      # _lacn_bufs:  Higham–Hager 1-norm estimator, previous iterate v (grows n)
    lacnsgn::Vector{Int}  # _lacn_bufs:  real-only sign-repeat test vector (grows n; empty for complex)
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
    # ── Non-symmetric eigenproblem (hessenberg.jl / hseqr.jl / trevc.jl / geev.jl / trsen.jl / trsyl.jl).
    # Every routine below allocated per call. Two of these buffers carry a LOAD-BEARING zero (`trevcn`,
    # `syl16`); the rest were `undef`/`Matrix(undef,…)` already, so nothing zeroed is being lost — each
    # claim is spelled out on its own field line and was checked against the producer/consumer pair, not
    # inherited from the old allocator's spelling.
    gehrdv::Vector{T}     # _gehrd_work:  gehrd! reflector staging v, length ihi-ilo+1 (grows). 1088 B F64
                          # / 2152 B C64 at n=129. DISTINCT from `orghrv`: ormhr! → orghr! is a real
                          # nesting chain, so one shared `v` across the family is the trsm_tmp mistake.
    orghrq::Matrix{T}     # _orghr_work:  orghr!/unghr! identity-seeded accumulation target Q (grows n×n).
                          # Arrives zeroed by the routine's own fill!+unit-diagonal, so no fill! here.
    orghrv::Vector{T}     # _orghr_work:  orghr! reflector staging v, length ihi-ilo+1 (grows). Q+v together
                          # measured 134,360 B F64 / 268,544 B C64 at n=129.
    mhrq::Matrix{T}       # _ormhr_work:  ormhr!/unmhr!'s OWN Q, filled by the non-destructive orghr form
                          # (grows n×n). Distinct field from `orghrq` because ormhr! nests orghr!. Owning it
                          # also deletes ormhr!'s `copy(A)`, which existed only to survive orghr!'s in-place
                          # contract: the three n×n matrices measured 400,904 B F64 / 801,328 B C64 at n=129.
    mhrc::Matrix{T}       # _ormhr_work:  ormhr! gemm! staging for BOTH side='L' and side='R' (grows
                          # size(C,1)×size(C,2)) — mutually exclusive branches of one call, one role. No
                          # fill!: both gemm! sites run beta=0, a pure overwrite of the whole tile.
    hqrwr::Vector{T}      # _hseqr_work:  real hseqr! eigenvalue real-part scratch (grows n). REAL-typed, so
                          # reached as `_eigws(real(T)).hqrwr`. 2240 B F64 at n=129 (wr+wi); the COMPLEX
                          # hseqr! path is already 0 B and claims neither field.
    hqrwi::Vector{T}      # _hseqr_work:  imaginary-part scratch (grows n). Own field — wr and wi are live
                          # simultaneously across the whole _dlahqr! call.
    trevcn::Vector{T}     # _trevc_rwork: trevc! column 1-norms of the strict upper triangle (grows n).
                          # REAL-typed → `_eigws(real(T))`. ZEROED BY THE ACCESSOR, load-bearing: the producer
                          # accumulates with `+=` and never touches cnorm[1], while the consumer reads
                          # cnorm[j] unconditionally for j down to 1.
    trevcxr::Vector{T}    # _trevc_rwork: real-path solution real part (grows n). REAL-typed.
    trevcxi::Vector{T}    # _trevc_rwork: real-path solution imaginary part (grows n). Own field — both are
                          # live across the whole complex-pair branch. Real trio measured 3328 B at n=129.
    trevcx::Vector{T}     # _trevc_work:  complex-path solution (grows n), ELEMENT-typed → `_eigws(T)`, a
                          # different owner object from the three real buffers above for complex T, so the
                          # two sets can never alias. 2152 B C64 at n=129.
    geevtau::Vector{T}    # _geev_work:   geev!/gees! gehrd reflector coefficients, length max(n-1,0) (grows).
                          # Own field, not `gehrdv`: it is live ACROSS the gehrd!→orghr! pair.
    geevw::Vector{T}      # _geev_cwork:  the REAL path's complex eigenvalue staging (grows n). The MIRROR of
                          # the pstrfv idiom — a COMPLEX-typed buffer needed by a REAL-typed T, so it is
                          # reached as `_eigws(Complex{T}).geevw`. Not used by the complex path (w is the
                          # output there).
    geevz0::Matrix{T}     # _geev_work:   the one shared 0×0 placeholder replacing geev!/gees!'s four
                          # `Zdummy` sites (~48 B of header each). Never grown, never written — hseqr! with
                          # compz='N' does not touch Z.
    laexcd::Matrix{T}     # _laexc_work:  _dlaexc! local diagonal-block copy D, FIXED 4×4 (nd = n1+n2 ≤ 4).
                          # REAL-typed → `_eigws(real(T))`. Handed out as an EXACT view(…,1:nd,1:nd): the
                          # dnorm loop is `for x in D`, so an oversized 4×4 at nd==3 would fold 7 stale
                          # elements into the `thresh` rejection test.
    laexcu1::Vector{T}    # _laexc_work:  _dlaexc! reflector u / u1, FIXED length 3. REAL-typed.
    laexcu2::Vector{T}    # _laexc_work:  _dlaexc! reflector u2, FIXED length 3. Own field — in the
                          # n1=2,n2=2 branch u1 is still read and applied while u2 is live.
    syl16::Matrix{T}      # _dlasy2_work: _syl_dlasy2 Kronecker system t16, FIXED 4×4. REAL-typed.
                          # ZEROED BY THE ACCESSOR, load-bearing: the k==4 arm writes only TWELVE of the
                          # sixteen entries, yet the complete-pivot search scans all sixteen and the
                          # elimination reads and updates them — stale [1,4]/[4,1]/[2,3]/[3,2] pick a wrong
                          # pivot, i.e. a wrong ANSWER, not noise.
    sylbt::Vector{T}      # _dlasy2_work: right-hand side btmp, FIXED length 4. All four written by its
                          # producer, no zeroing.
    sylpv::Vector{Int}    # _dlasy2_work: _tgs/_syl column-pivot record jpiv, FIXED length 4. Int-typed on
                          # the real owner (the struct already carries Vector{Int} fields).
    syltv::Vector{T}      # _dlasy2_work: back-substitution result tmpv, FIXED length 4.
    trsenr::Matrix{T}     # _trsen_work:  trsen! T₁₂ coupling block Rm (grows n1×n2), element-typed.
    trsen11::Matrix{T}    # _trsen_work:  T₁₁ block (grows n1×n1).
    trsen22::Matrix{T}    # _trsen_work:  T₂₂ block (grows n2×n2).
    trsenx::Matrix{T}     # _trsen_work:  Sylvester-apply staging Xm (grows n1×n2). Own field even though
                          # plain UInt64 field, so `_stein_randvec!(x, seed::Base.RefValue{UInt64})` needs
                          # no signature change. THE ACCESSOR RESETS IT TO `_STEIN_SEED0` ON EVERY CALL —
                          # see there; owning it removes the 16 B per-call RefValue without turning
                          # stein! into a stateful RNG.
    trsensc::Vector{T}    # _trsen_sc: 1-element carrier for the scale `_dtrsyl!`/`_ztrsyl!` returns through
end

EigWorkspace{T}() where {T} = EigWorkspace{T}(
    T[],                                    # lacnx
    T[],                                    # lacnv
    Int[],                                  # lacnsgn
    T[],                                    # lacn2x
    T[],                                    # lacn2v
    Int[],                                  # lacn2s
    Matrix{T}(undef, 0, 0),                 # mtrv
    Matrix{T}(undef, 0, 0),                 # mtrg
    Matrix{T}(undef, 0, 0),                 # mtrw
    Matrix{T}(undef, 0, 0),                 # mtrt
    T[],                                    # gehrdv
    Matrix{T}(undef, 0, 0),                 # orghrq
    T[],                                    # orghrv
    Matrix{T}(undef, 0, 0),                 # mhrq
    Matrix{T}(undef, 0, 0),                 # mhrc
    T[],                                    # hqrwr
    T[],                                    # hqrwi
    T[],                                    # trevcn
    T[],                                    # trevcxr
    T[],                                    # trevcxi
    T[],                                    # trevcx
    T[],                                    # geevtau
    T[],                                    # geevw
    Matrix{T}(undef, 0, 0),                 # geevz0
    Matrix{T}(undef, 4, 4),                 # laexcd
    Vector{T}(undef, 3),                    # laexcu1
    Vector{T}(undef, 3),                    # laexcu2
    Matrix{T}(undef, 4, 4),                 # syl16
    Vector{T}(undef, 4),                    # sylbt
    Vector{Int}(undef, 4),                  # sylpv
    Vector{T}(undef, 4),                    # syltv
    Matrix{T}(undef, 0, 0),                 # trsenr
    Matrix{T}(undef, 0, 0),                 # trsen11
    Matrix{T}(undef, 0, 0),                 # trsen22
    Matrix{T}(undef, 0, 0),                 # trsenx
    T[],                                    # trsensc
)

const _EIGWS_F64 = EigWorkspace{Float64}()
const _EIGWS_F32 = EigWorkspace{Float32}()
const _EIGWS_C64 = EigWorkspace{ComplexF64}()
const _EIGWS_C32 = EigWorkspace{ComplexF32}()
const _EIGWS_OTHER = IdDict{DataType, EigWorkspace}()
@inline _eigws(::Type{Float64}) = _EIGWS_F64
@inline _eigws(::Type{Float32}) = _EIGWS_F32
@inline _eigws(::Type{ComplexF64}) = _EIGWS_C64
@inline _eigws(::Type{ComplexF32}) = _EIGWS_C32
_eigws(::Type{T}) where {T} = get!(() -> EigWorkspace{T}(), _EIGWS_OTHER, T)::EigWorkspace{T}

####################################################################################################

# GenEigWorkspace — 44 roles. Split out of the former 178-field L3Workspace so
# each ALGORITHM FAMILY owns its own scratch and can carry its own M4 sharing decision
# (kb: workspace-mt-ownership-per-buffer-not-uniform). Field lines and their comments are
# unchanged from the original struct.
mutable struct GenEigWorkspace{T}
                          # columns of A. No fill!: every read is preceded by a copyto! of exactly the slice
                          # about to be read, on the same call.
    # ── Generalized eigenproblem / QZ (qz.jl, tgevc_gen.jl, tgsen.jl, ggev.jl).
    hgzar::Vector{T}      # _hgeqz_work: real hgeqz! entry alphar scratch (grows n). REAL-typed →
                          # `_geneigws(real(T))`. 2240 B at n=129; the COMPLEX hgeqz! path is already 0 B.
    hgzai::Vector{T}      # _hgeqz_work: real hgeqz! entry alphai scratch (grows n). Own field — both are
                          # live across the whole `_hgeqz!` call.
    hgzv::Vector{T}       # _hgeqz_work: `_hgeqz!` implicit-shift Householder vector, FIXED length 3.
    tgvw::Matrix{T}       # _tgevc_work:  tgevc! real-path W, grows n×6 (six packed roles: Snorm, Pnorm,
                          # x.re, x.im, bt.re, bt.im). Kept ONE matrix, not six carved views, because the
                          # engine indexes it as W[j, 3+jw] with a RUNTIME column offset. 6296 B at n=129.
    tgvx::Vector{T}       # _tgevc_work:  tgevc! complex-path `work`, grows 2n, element-typed.
    tgvr::Vector{T}       # _tgevc_rwork: tgevc! complex-path `rwork`, grows 2n. REAL-typed →
                          # `_geneigws(real(T))`, a different owner object from `tgvx` for complex T, so the
                          # two can never alias.
    tgsar::Vector{T}      # _tgsen_alpha_work: `_dtgsen!` alphar scratch (grows n). REAL-typed.
    tgsai::Vector{T}      # _tgsen_alpha_work: `_dtgsen!` alphai scratch (grows n). REAL-typed.
    # `_dtgex2_big!`'s scratch. n1,n2 ∈ {1,2} over every `_dtgex2!` call site ⇒ m = n1+n2 ≤ 4, nz = 2·n1·n2
    # ≤ 8. All FIXED-size, allocated at construction, never grown — 39 allocation sites per swap, and this
    # is the family's only O(n²) leak (3,326,600 B at n=129 with 53 conjugate-pair blocks). All REAL-typed:
    # the complex `_ztgex2!` is pure scalar and claims none of these.
    tgxs::Matrix{T}       # _tgex2_blocks: S block, FIXED 4×4
    tgxt::Matrix{T}       # _tgex2_blocks: Tm block, FIXED 4×4
    tgxc::Matrix{T}       # _tgex2_blocks: Cm, FIXED 2×2
    tgxf::Matrix{T}       # _tgex2_blocks: Fm, FIXED 2×2
    tgxz::Matrix{T}       # _tgsy2_work: `_tgs_tgsy2!` Kronecker system Z, FIXED 8×8. ZEROED BY THE ACCESSOR,
                          # load-bearing: it is built by `+=`/`-=` accumulation with no prior full write.
    tgxrhs::Vector{T}     # _tgsy2_work: rhs, FIXED length 8
    tgxip::Vector{Int}    # _tgsy2_work: `_tgs_getc2!` row pivots, FIXED length 8
    tgxjp::Vector{Int}    # _tgsy2_work: `_tgs_getc2!` column pivots, FIXED length 8
    tgxli::Matrix{T}      # _tgex2_qr: LI, FIXED 4×4. ZEROED BY THE ACCESSOR, load-bearing: only LI[1:n1,1:n2]
                          # and a unit sub-diagonal are written, yet `_tgs_geqr2!` reads all of LI[1:m,1:n2].
    tgxir::Matrix{T}      # _tgex2_qr: IR, FIXED 4×4. ZEROED, load-bearing for the mirror reason —
                          # `_tgs_gerq2!` reads rows n2+1:n2+n1 across all of cols 1:m.
    tgxtaul::Vector{T}    # _tgex2_qr: τl, FIXED length 4
    tgxtaur::Vector{T}    # _tgex2_qr: τr, FIXED length 4
    tgxtaul2::Vector{T}   # _tgex2_qr: τl2, FIXED length 4
    tgxtaur2::Vector{T}   # _tgex2_qr: τr2, FIXED length 4
    tgxvq::Vector{T}      # _tgex2_qr: `_tgs_geqr2!` reflector staging, FIXED length 4. Own field per role.
    tgxvr::Vector{T}      # _tgex2_qr: `_tgs_gerq2!` reflector staging, FIXED length 4. Own field per role.
    tgxm1::Matrix{T}      # _tgex2_mul: `_tgs_matmul` inner result (ping), FIXED 4×4
    tgxm2::Matrix{T}      # _tgex2_mul: `_tgs_matmul` outer result (pong), FIXED 4×4 — the call sites nest
                          # inner-inside-outer, so BOTH are live at once.
    tgxpa::Matrix{T}      # _tgex2_mul: PA, FIXED 4×4
    tgxpb::Matrix{T}      # _tgex2_mul: PB, FIXED 4×4
    tgxscpy::Matrix{T}    # _tgex2_copies: SCPY, FIXED 4×4
    tgxtcpy::Matrix{T}    # _tgex2_copies: TCPY, FIXED 4×4
    tgxircop::Matrix{T}   # _tgex2_copies: IRCOP, FIXED 4×4
    tgxlicop::Matrix{T}   # _tgex2_copies: LICOP, FIXED 4×4
    tgxql2::Matrix{T}     # _tgex2_rot: QL2, FIXED 4×4. ZEROED, load-bearing: only the (1,1)/(m,m) entries
                          # and two 2×2 corner blocks are set, and it is then read IN FULL.
    tgxir2::Matrix{T}     # _tgex2_rot: IR2, FIXED 4×4. ZEROED, same argument as QL2.
    tgxta::Matrix{T}      # _tgex2_rot: tmpA, FIXED 2×2
    tgxtb::Matrix{T}      # _tgex2_rot: tmpB, FIXED 2×2
    tgxw::Vector{T}       # _tgex2_rot: off-block row/col update w, FIXED length 4
    tgxbufq::Matrix{T}    # _tgex2_accum: wantq accumulate buffer, grows n×4 — one of the two sites that
                          # scale with n, i.e. the whole O(n²) term.
    tgxbufz::Matrix{T}    # _tgex2_accum: wantz accumulate buffer, grows n×4. Own field per role.
    ggqtau::Vector{T}     # _ggev_qrb_work: `_ggev_qrB!` geqrf! τ (grows n), element-typed. Shared by ggev!
                          # AND gges! — one role, identical call, and the two entries never nest.
    ggqtaul::Vector{T}    # _ggev_qrb_work: LAPACK-convention τ (grows n)
    ggqq::Matrix{T}       # _ggev_qrb_work: explicit Q_B (grows n×n). NOT shareable with ggqtmp — the gemm!
                          # reads Q while writing tmp. Arrives zeroed by `_ggev_formQ!`'s own fill!.
    ggqtmp::Matrix{T}     # _ggev_qrb_work: Qᴴ·A gemm destination (grows n×n). Written by a beta=0 gemm!,
                          # so no fill!. The four together measured 268,784 B F64 at n=129.
    ggevac::Vector{T}     # _ggev_cwork: real ggev!'s alphaC scratch (grows n). COMPLEX-typed for a REAL A,
end

GenEigWorkspace{T}() where {T} = GenEigWorkspace{T}(
    T[],                                    # hgzar
    T[],                                    # hgzai
    Vector{T}(undef, 3),                    # hgzv
    Matrix{T}(undef, 0, 0),                 # tgvw
    T[],                                    # tgvx
    T[],                                    # tgvr
    T[],                                    # tgsar
    T[],                                    # tgsai
    Matrix{T}(undef, 4, 4),                 # tgxs
    Matrix{T}(undef, 4, 4),                 # tgxt
    Matrix{T}(undef, 2, 2),                 # tgxc
    Matrix{T}(undef, 2, 2),                 # tgxf
    Matrix{T}(undef, 8, 8),                 # tgxz
    Vector{T}(undef, 8),                    # tgxrhs
    Vector{Int}(undef, 8),                  # tgxip
    Vector{Int}(undef, 8),                  # tgxjp
    Matrix{T}(undef, 4, 4),                 # tgxli
    Matrix{T}(undef, 4, 4),                 # tgxir
    Vector{T}(undef, 4),                    # tgxtaul
    Vector{T}(undef, 4),                    # tgxtaur
    Vector{T}(undef, 4),                    # tgxtaul2
    Vector{T}(undef, 4),                    # tgxtaur2
    Vector{T}(undef, 4),                    # tgxvq
    Vector{T}(undef, 4),                    # tgxvr
    Matrix{T}(undef, 4, 4),                 # tgxm1
    Matrix{T}(undef, 4, 4),                 # tgxm2
    Matrix{T}(undef, 4, 4),                 # tgxpa
    Matrix{T}(undef, 4, 4),                 # tgxpb
    Matrix{T}(undef, 4, 4),                 # tgxscpy
    Matrix{T}(undef, 4, 4),                 # tgxtcpy
    Matrix{T}(undef, 4, 4),                 # tgxircop
    Matrix{T}(undef, 4, 4),                 # tgxlicop
    Matrix{T}(undef, 4, 4),                 # tgxql2
    Matrix{T}(undef, 4, 4),                 # tgxir2
    Matrix{T}(undef, 2, 2),                 # tgxta
    Matrix{T}(undef, 2, 2),                 # tgxtb
    Vector{T}(undef, 4),                    # tgxw
    Matrix{T}(undef, 0, 0),                 # tgxbufq
    Matrix{T}(undef, 0, 0),                 # tgxbufz
    T[],                                    # ggqtau
    T[],                                    # ggqtaul
    Matrix{T}(undef, 0, 0),                 # ggqq
    Matrix{T}(undef, 0, 0),                 # ggqtmp
    T[],                                    # ggevac
)

const _GENEIGWS_F64 = GenEigWorkspace{Float64}()
const _GENEIGWS_F32 = GenEigWorkspace{Float32}()
const _GENEIGWS_C64 = GenEigWorkspace{ComplexF64}()
const _GENEIGWS_C32 = GenEigWorkspace{ComplexF32}()
const _GENEIGWS_OTHER = IdDict{DataType, GenEigWorkspace}()
@inline _geneigws(::Type{Float64}) = _GENEIGWS_F64
@inline _geneigws(::Type{Float32}) = _GENEIGWS_F32
@inline _geneigws(::Type{ComplexF64}) = _GENEIGWS_C64
@inline _geneigws(::Type{ComplexF32}) = _GENEIGWS_C32
_geneigws(::Type{T}) where {T} = get!(() -> GenEigWorkspace{T}(), _GENEIGWS_OTHER, T)::GenEigWorkspace{T}

####################################################################################################

# LstsqWorkspace — 28 roles. Split out of the former 178-field L3Workspace so
# each ALGORITHM FAMILY owns its own scratch and can carry its own M4 sharing decision
# (kb: workspace-mt-ownership-per-buffer-not-uniform). Field lines and their comments are
# unchanged from the original struct.
mutable struct LstsqWorkspace{T}
    # geqp3 (column-pivoted QR). Six buffers, allocated per call before this: 384 B at n=8 rising to
    # 12,952 B at n=129. Worth noting HOW that was found — the audit measured five sizes straddling the
    # blocking threshold, because `vn1`/`vn2`/`hbuf` are allocated on every path while `F`/`auxv`/`wrow`
    # appear only once `nb > 1`. A single-size check sees at most half the leak, which is precisely the
    # trap that hid pptrf!'s blocked arm at n=8.
    qp3vn1::Vector{T}     # _geqp3_norms_work: downdated partial column 2-norms — REAL, via _lstsqws(real(T))
    qp3vn2::Vector{T}     # _geqp3_norms_work: reference norm at last exact recompute — REAL
    qp3hbuf::Vector{T}    # _geqp3_norms_work: contiguous heads for the SIMD downdate — REAL
    qp3f::Matrix{T}       # _geqp3_panel_work: laqps block reflector F, n × nb (blocked arm only)
    qp3auxv::Vector{T}    # _geqp3_panel_work: laqps auxiliary vector, length nb (blocked arm only)
    qp3wrow::Vector{T}    # _geqp3_panel_work: pivot-row delta buffer, length n (blocked arm only)
    # ── Least squares (gels.jl / gelsd.jl / gelsy.jl / gglse.jl) and symmetric-tridiagonal eigen
    # (stebz.jl). All measured `undef`/`copy` today, so no zeroing is being lost except where stated.
    gelsm::Matrix{T}      # _gels_work: gels! op(A) working copy (grows p×q). gels! deliberately does NOT
                          # overwrite A, which is what forces this to exist even for trans='N'. The three
                          # gels! fields measured 134,392 B F64 square / 533,904 B F64 wide at n=129.
    gelsmh::Matrix{T}     # _gels_adj: gels! adjoint copy, UNDERDETERMINED ARM ONLY (grows q×p). Its own
                          # accessor so the common arm never grows it.
    gelstau::Vector{T}    # _gels_work: gels! Householder τ (grows min(p,q)).
    gelsdu::Matrix{T}     # _gelsd_work: gelsd! SVD left factor U (grows m×min(m,n)).
    gelsdvt::Matrix{T}    # _gelsd_work: gelsd! SVD right factor Vᴴ (grows min(m,n)×n).
    gelsdc::Matrix{T}     # _gelsd_work: gelsd! C = Uᴴ·b, then Σ⁺ applied in place (grows min(m,n)×nrhs).
                          # Written by a beta=0 gemm!, so no fill!.
    gelsda::Matrix{T}     # _gelsd_work: gelsd! destructible copy of A for gesvd! (grows m×n). Becomes dead
                          # if the team honours the docstring and lets gesvd! destroy A directly.
    gelsdad::Matrix{T}    # _gelsd_promote_work: the Float32 wrapper's Float64 A staging (grows m×n). Lives
                          # on `_lstsqws(Float64)` — the mirror of the real()-indirection, reached UP from a
                          # Float32 entry rather than down from a complex one.
    gelsdbd::Matrix{T}    # _gelsd_promote_work: the Float32 wrapper's Float64 B staging (grows m×nrhs).
    gelsdsd::Vector{T}    # _gelsd_promote_work: the Float32 wrapper's Float64 singular values (grows
                          # min(m,n)) — the buffer the nested Float64 call's in-place form writes.
    gelsytau::Vector{T}   # _gelsy_work: gelsy! geqp3 τ (grows min(m,n)).
    gelsyxmin::Vector{T}  # _gelsy_work: `_laic1` job=2 iterate (grows max(min(m,n),1)).
    gelsyxmax::Vector{T}  # _gelsy_work: `_laic1` job=1 iterate. Own field — both are live simultaneously
                          # across the whole rank loop; sharing is a wrong answer, not a saving.
    gelsytauz::Vector{T}  # _gelsy_work: RZ τ (grows min(m,n)). Own field — `gelsytau` is still read by
                          # `_apply_Qh!` AFTER tauz is filled.
    gelsyw::Vector{T}     # _gelsy_work: un-pivot gather buffer (grows n).
    tzrzfb::Vector{T}     # _tzrzf_buf: tzrzf!'s (z)larfg staging row (grows (n-m)+1). Owed regardless of
                          # gelsy! — tzrzf! is a public entry in its own right (also via SIMDBackend).
                          # Measured 576 B F64 / 1088 B C64 at 64×129.
    gglg::Matrix{T}       # _gglse_work: gglse! explicit RQ orthogonal factor G (grows n×n) — the single
                          # biggest site; the six fields measured 272,192 B F64 / 543,704 B C64 at n=129.
    gglat::Matrix{T}      # _gglse_work: Ã = A·G (grows m×n). Written by a beta=0 gemm!, so no fill!.
    gglt::Vector{T}       # _gglse_work: QR τ (grows min(m,n)).
    gglc::Matrix{T}       # _gglse_work: Zᴴ·c staging. A MATRIX (grows m×1), NOT a Vector, on purpose: the
                          # accessor hands `_ggl_apply_Zh!` a view(…,1:m,1:1) so gglse.jl:70's
                          # `reshape(c, m, 1)` is DELETED rather than relocated — a reshape of a SubArray
                          # routes to Base._throw_dmrs and fails `juliac --trim=safe`.
    gglyy::Vector{T}      # _gglse_work: transformed solution y = Q·x (grows n).
    gglr::Vector{T}       # _gglse_work: residual A·x for the returned norm (grows m).
end

LstsqWorkspace{T}() where {T} = LstsqWorkspace{T}(
    T[],                                    # qp3vn1
    T[],                                    # qp3vn2
    T[],                                    # qp3hbuf
    Matrix{T}(undef, 0, 0),                 # qp3f
    T[],                                    # qp3auxv
    T[],                                    # qp3wrow
    Matrix{T}(undef, 0, 0),                 # gelsm
    Matrix{T}(undef, 0, 0),                 # gelsmh
    T[],                                    # gelstau
    Matrix{T}(undef, 0, 0),                 # gelsdu
    Matrix{T}(undef, 0, 0),                 # gelsdvt
    Matrix{T}(undef, 0, 0),                 # gelsdc
    Matrix{T}(undef, 0, 0),                 # gelsda
    Matrix{T}(undef, 0, 0),                 # gelsdad
    Matrix{T}(undef, 0, 0),                 # gelsdbd
    T[],                                    # gelsdsd
    T[],                                    # gelsytau
    T[],                                    # gelsyxmin
    T[],                                    # gelsyxmax
    T[],                                    # gelsytauz
    T[],                                    # gelsyw
    T[],                                    # tzrzfb
    Matrix{T}(undef, 0, 0),                 # gglg
    Matrix{T}(undef, 0, 0),                 # gglat
    T[],                                    # gglt
    Matrix{T}(undef, 0, 0),                 # gglc
    T[],                                    # gglyy
    T[],                                    # gglr
)

const _LSTSQWS_F64 = LstsqWorkspace{Float64}()
const _LSTSQWS_F32 = LstsqWorkspace{Float32}()
const _LSTSQWS_C64 = LstsqWorkspace{ComplexF64}()
const _LSTSQWS_C32 = LstsqWorkspace{ComplexF32}()
const _LSTSQWS_OTHER = IdDict{DataType, LstsqWorkspace}()
@inline _lstsqws(::Type{Float64}) = _LSTSQWS_F64
@inline _lstsqws(::Type{Float32}) = _LSTSQWS_F32
@inline _lstsqws(::Type{ComplexF64}) = _LSTSQWS_C64
@inline _lstsqws(::Type{ComplexF32}) = _LSTSQWS_C32
_lstsqws(::Type{T}) where {T} = get!(() -> LstsqWorkspace{T}(), _LSTSQWS_OTHER, T)::LstsqWorkspace{T}

####################################################################################################

# TridiagWorkspace — 13 roles. Split out of the former 178-field L3Workspace so
# each ALGORITHM FAMILY owns its own scratch and can carry its own M4 sharing decision
# (kb: workspace-mt-ownership-per-buffer-not-uniform). Field lines and their comments are
# unchanged from the original struct.
mutable struct TridiagWorkspace{T}
    stbe2::Vector{T}      # _stebz_work: stebz! squared off-diagonals (grows max(n-1,1)). The old allocator
                          # was `zeros`, and the zeroing was checked and is NOT load-bearing: the split loop
                          # writes every index 1..n-1 unconditionally before either reader runs, and the one
                          # slot that stays unwritten (e2[1] at n==1) is never read.
    stbperm::Vector{Int}  # _stebz_work: sortperm destination for the range=='I' band slice (grows n).
    stbperm2::Vector{Int} # _stebz_work: sortperm destination for the order=='E' re-sort (grows n). A SECOND
                          # field, overriding the audit's "one is genuinely enough" — its own argument for
                          # sharing was "the first p is dead by then", which is precisely the never-overlap
                          # reasoning lesson 4 forbids. Two Int vectors is cheap insurance.
    stbidx::Vector{Int}   # _stebz_work: the sorted il:iu index band (grows n). Own field — the producer
                          # reads `stbperm` while writing this.
    stbgw::Vector{T}      # _stebz_work: permutation staging for w (grows n) — a permute needs a temp.
    stbgi::Vector{Int}    # _stebz_work: permutation staging for iblock (grows n). Own field: different
                          # element type, and both are live simultaneously.
    stnav::Vector{T}      # _stein_work: stein! dlagtf `a` (grows n).
    stnbv::Vector{T}      # _stein_work: stein! dlagtf `b` (grows n).
    stncv::Vector{T}      # _stein_work: stein! dlagtf `c` (grows n).
    stnd2::Vector{T}      # _stein_work: stein! dlagtf `d2` (grows n). The one slot `_dlagtf!` leaves
                          # unwritten at bz==2 is also never read, so no fill!.
    stnrhs::Vector{T}     # _stein_work: inverse-iteration iterate (grows n).
    stninn::Vector{Int}   # _stein_work: dlagtf interchange record (grows n).
    stnseed::Base.RefValue{UInt64}  # _stein_work: `_stein_randvec!` xorshift seed. Kept a RefValue, not a
end

TridiagWorkspace{T}() where {T} = TridiagWorkspace{T}(
    T[],                                    # stbe2
    Int[],                                  # stbperm
    Int[],                                  # stbperm2
    Int[],                                  # stbidx
    T[],                                    # stbgw
    Int[],                                  # stbgi
    T[],                                    # stnav
    T[],                                    # stnbv
    T[],                                    # stncv
    T[],                                    # stnd2
    T[],                                    # stnrhs
    Int[],                                  # stninn
    Ref(_STEIN_SEED0),                      # stnseed
)

const _TRIDIAGWS_F64 = TridiagWorkspace{Float64}()
const _TRIDIAGWS_F32 = TridiagWorkspace{Float32}()
const _TRIDIAGWS_C64 = TridiagWorkspace{ComplexF64}()
const _TRIDIAGWS_C32 = TridiagWorkspace{ComplexF32}()
const _TRIDIAGWS_OTHER = IdDict{DataType, TridiagWorkspace}()
@inline _tridiagws(::Type{Float64}) = _TRIDIAGWS_F64
@inline _tridiagws(::Type{Float32}) = _TRIDIAGWS_F32
@inline _tridiagws(::Type{ComplexF64}) = _TRIDIAGWS_C64
@inline _tridiagws(::Type{ComplexF32}) = _TRIDIAGWS_C32
_tridiagws(::Type{T}) where {T} = get!(() -> TridiagWorkspace{T}(), _TRIDIAGWS_OTHER, T)::TridiagWorkspace{T}

####################################################################################################

# GsvdWorkspace — 16 roles. Split out of the former 178-field L3Workspace so
# each ALGORITHM FAMILY owns its own scratch and can carry its own M4 sharing decision
# (kb: workspace-mt-ownership-per-buffer-not-uniform). Field lines and their comments are
# unchanged from the original struct.
mutable struct GsvdWorkspace{T}
                          # so reached as `_gsvdws(Complex{T}).ggevac` — same mirror as `geevw`, and a
                          # DISTINCT field from it (geev! and ggev! are unrelated routines).
    # ── ggsvd! (ggsvd.jl). Sixteen roles. Two of them carry the whole allocation curve — `ggs_w` is one
    # allocation PER REFLECTOR APPLICATION and `ggs_rqw` replaces one retained `wv` per reflector, both
    # O(n²) bytes per call; the rest are O(n) each. Scratch only — the outputs (U/V/Q/alpha/beta/R) need
    # the in-place form, not fields.
    ggs_w::Vector{T}      # _ggsvd_larf_w: `_ggs_larf_right!` accumulator (grows max(m,p,n)). ZEROED BY THE
                          # ACCESSOR, load-bearing: the accumulation loop is guarded by `if u[k] != 0`, so a
                          # column of zeros writes nothing at all, and the consumer reads w[i]
                          # UNCONDITIONALLY — a skipped fill is an active wrong answer, not a latent one.
    ggs_rqw::Matrix{T}    # _ggsvd_rq_work: column i holds reflector i's retained `wv` (grows n×min(m,p,n)).
    ggs_rqc::Vector{Int}  # _ggsvd_rq_work: per-reflector `c` (grows min(m,p,n)) — replaces the Int slot of
                          # the `Vector{Tuple{Int,T,Vector{T}}}` carrier.
    ggs_rqtau::Vector{T}  # _ggsvd_rq_work: per-reflector tau (grows min(m,p,n)).
    ggs_u::Vector{T}      # _ggsvd_u_qp:    `_ggs_geqpf!` reflector u (grows max(m,n)). Own field — it is
                          # live across a `_ggs_larf_left!` that a shared buffer would clobber.
    ggs_ul::Vector{T}     # _ggsvd_u_applyl: `_ggs_qr_apply_left!` u (grows m).
    ggs_uq::Vector{T}     # _ggsvd_u_formq:  `_ggs_formQ!` u (grows max(m,p) — called for both U and V).
    ggs_ur::Vector{T}     # _ggsvd_u_rq:     `_ggs_gerq2!` pivot-first conjugated row (grows n).
    ggs_u5::Vector{T}     # _ggsvd_u_step5:  `_ggs_ggsvp!` step-5 u (grows m).
    ggs_xc::Matrix{T}     # _ggsvd_permcols: `_ggs_permcols!` Xc (grows max(m,p,n)×n — it is called on A
                          # (m×n) and on view(Q,:,1:nl) (n×nl), so it must fit both).
    ggs_taub::Vector{T}   # _ggsvd_tau_work: B's QR τ (grows min(p,n)).
    ggs_taua::Vector{T}   # _ggsvd_tau_work: A11's QR τ (grows min(m,n)). Own field — taub still feeds
                          # `_ggs_formQ!(V, …)` and keeping them apart is exactly what lesson 4 names.
    ggs_jpvt::Vector{Int} # _ggsvd_pivots: B's column pivots (grows n).
    ggs_jp2::Vector{Int}  # _ggsvd_pivots: A11's column pivots (grows n). Own field.
    ggs_wx::Vector{T}     # _ggsvd_tgsja_work: `_ggs_tgsja!` wx (grows max(n,1)).
    ggs_wy::Vector{T}     # _ggsvd_tgsja_work: `_ggs_tgsja!` wy (grows max(n,1)).
end

GsvdWorkspace{T}() where {T} = GsvdWorkspace{T}(
    T[],                                    # ggs_w
    Matrix{T}(undef, 0, 0),                 # ggs_rqw
    Int[],                                  # ggs_rqc
    T[],                                    # ggs_rqtau
    T[],                                    # ggs_u
    T[],                                    # ggs_ul
    T[],                                    # ggs_uq
    T[],                                    # ggs_ur
    T[],                                    # ggs_u5
    Matrix{T}(undef, 0, 0),                 # ggs_xc
    T[],                                    # ggs_taub
    T[],                                    # ggs_taua
    Int[],                                  # ggs_jpvt
    Int[],                                  # ggs_jp2
    T[],                                    # ggs_wx
    T[],                                    # ggs_wy
)

const _GSVDWS_F64 = GsvdWorkspace{Float64}()
const _GSVDWS_F32 = GsvdWorkspace{Float32}()
const _GSVDWS_C64 = GsvdWorkspace{ComplexF64}()
const _GSVDWS_C32 = GsvdWorkspace{ComplexF32}()
const _GSVDWS_OTHER = IdDict{DataType, GsvdWorkspace}()
@inline _gsvdws(::Type{Float64}) = _GSVDWS_F64
@inline _gsvdws(::Type{Float32}) = _GSVDWS_F32
@inline _gsvdws(::Type{ComplexF64}) = _GSVDWS_C64
@inline _gsvdws(::Type{ComplexF32}) = _GSVDWS_C32
_gsvdws(::Type{T}) where {T} = get!(() -> GsvdWorkspace{T}(), _GSVDWS_OTHER, T)::GsvdWorkspace{T}

####################################################################################################

# FactWorkspace — 22 roles. Split out of the former 178-field L3Workspace so
# each ALGORITHM FAMILY owns its own scratch and can carry its own M4 sharing decision
# (kb: workspace-mt-ownership-per-buffer-not-uniform). Field lines and their comments are
# unchanged from the original struct.
mutable struct FactWorkspace{T}
    bandl::Matrix{T}      # _pbtrf_band:  pbtrf uplo='U' conj-transposed band re-pack, grows (kd+1)×n
    bandw::Matrix{T}      # _pbtrf_work:  pbtrf corner work array W, exactly (nb+1)×nb (ld is load-bearing)
    bands::Matrix{T}      # _pbtrf_work:  pbtrf dense diagonal-block scratch, exactly nb×nb (ld load-bearing)
    gbw13::Matrix{T}      # _gbtrf_work:  gbtrf upper-corner WORK13, exactly (nb+1)×nb (ld load-bearing)
    gbw31::Matrix{T}      # _gbtrf_work:  gbtrf lower-corner WORK31, exactly (nb+1)×nb (ld load-bearing)
    gbs::Matrix{T}        # _gbtrf_work:  gbtrf dense L11 staging scratch, exactly nb×nb (ld load-bearing)
    sylw::Matrix{T}       # _sytrf_work:  dlasyf panel W (n×nb) + gather col; exact rows, grow-only cols
    lauumt::Matrix{T}     # _lauum_tmp:  lauum dense-base zeroed triangle copy (≤ _trtri_base square)
    getriw::Matrix{T}     # _getri_work: getri blocked inversion panel W (grows n×nb)
    pstrfv::Vector{T}     # _pstrf_work: running dot products + scratch, 2n. REAL-typed, so it is reached
                          # as `_factws(real(T)).pstrfv` — for complex A that is a DIFFERENT owner object
                          # than `pstrfs` below, so the two can never alias.
    pstrfs::Vector{T}     # _pstrf_work: blocked panel scratch (nb + 2n), element-typed
    pstrfsw::Vector{Int}  # _pstrf_swaps: blocked pivot bookkeeping swj/swp, 2·nb as two DISJOINT views
    trrfsr::Vector{T}     # _trrfs_work: residual r = op(A)·x − b (grows n), element-typed
    trrfsa::Vector{T}     # _trrfs_work: |op(A)|·|x| + |b|  — REAL-typed, via `_factws(real(T))`
    trrfst::Vector{T}     # _trrfs_work: LACN2 weight |r| + nz·eps·wabs — REAL-typed, via `_factws(real(T))`
    # pptrf blocked packed Cholesky. These allocated per call and so broke `pptrf!`'s `!` promise for
    # every n ≥ _PPTRF_BLK_MIN (16) — 16,560 B at n=32, growing with n. It measured 0 B at n=8 only
    # because that is BELOW the blocking threshold and takes the unblocked arm, which is exactly how the
    # bug survived an allocation audit: pick a size on the other side of every threshold.
    pptrfw::Matrix{T}     # _pptrf_lower_work: lower panel W, n×nb
    pptrfv::Matrix{T}     # _pptrf_{lower,upper}_work: trailing-update V, n×nb (shared — one uplo per call)
    pptrfr::Matrix{T}     # _pptrf_upper_work: upper packed-row block R, nb×n (note the transposed shape)
    gerfsr::Vector{T}     # _gerfs_work: residual r = b − op(A)·x (grows n), element-typed
    gerfsw::Vector{T}     # _gerfs_work: |b| + |op(A)|·|x| — REAL-typed, via `_factws(real(T))`
    gerfsd::Vector{T}     # _gerfs_work: refinement correction dx (grows n), element-typed
                          # job='B' uses Rm then Xm sequentially. All four are fully overwritten by their
                          # copyto! before any read, so no fill!. CAVEAT recorded at trsen.jl:552-558: Xm is
                          # a dense Matrix today to keep _dtrsyl! off the SubArray path. The view handed out
                          # here IS a SubArray — trim-safe (no reshape) but a possible perf change; if it
                          # measures, switch this one to the `_ormtr_work` whole-buffer idiom.
    # ── Bunch–Kaufman (sysv.jl). ONE field for sytri! AND hetri!, not two: they are two entry points onto
    # the identical `_sytri_lower!`/`_sytri_upper!` engine differing only in a `herm` Bool, so it is ONE
    # role, and neither is re-entrant. Removes 1120 B F64 / 2184 B C64 at n=129.
    sytriw::Vector{T}     # _sytri_work: sytri!/hetri! column staging (grows n). Element-typed — it holds
end

FactWorkspace{T}() where {T} = FactWorkspace{T}(
    Matrix{T}(undef, 0, 0),                 # bandl
    Matrix{T}(undef, 0, 0),                 # bandw
    Matrix{T}(undef, 0, 0),                 # bands
    Matrix{T}(undef, 0, 0),                 # gbw13
    Matrix{T}(undef, 0, 0),                 # gbw31
    Matrix{T}(undef, 0, 0),                 # gbs
    Matrix{T}(undef, 0, 0),                 # sylw
    Matrix{T}(undef, 0, 0),                 # lauumt
    Matrix{T}(undef, 0, 0),                 # getriw
    T[],                                    # pstrfv
    T[],                                    # pstrfs
    Int[],                                  # pstrfsw
    T[],                                    # trrfsr
    T[],                                    # trrfsa
    T[],                                    # trrfst
    Matrix{T}(undef, 0, 0),                 # pptrfw
    Matrix{T}(undef, 0, 0),                 # pptrfv
    Matrix{T}(undef, 0, 0),                 # pptrfr
    T[],                                    # gerfsr
    T[],                                    # gerfsw
    T[],                                    # gerfsd
    T[],                                    # sytriw
)

const _FACTWS_F64 = FactWorkspace{Float64}()
const _FACTWS_F32 = FactWorkspace{Float32}()
const _FACTWS_C64 = FactWorkspace{ComplexF64}()
const _FACTWS_C32 = FactWorkspace{ComplexF32}()
const _FACTWS_OTHER = IdDict{DataType, FactWorkspace}()
@inline _factws(::Type{Float64}) = _FACTWS_F64
@inline _factws(::Type{Float32}) = _FACTWS_F32
@inline _factws(::Type{ComplexF64}) = _FACTWS_C64
@inline _factws(::Type{ComplexF32}) = _FACTWS_C32
_factws(::Type{T}) where {T} = get!(() -> FactWorkspace{T}(), _FACTWS_OTHER, T)::FactWorkspace{T}

####################################################################################################


# Per-role accessors (unchanged signatures — call sites are untouched). Each returns/grows one owned field.
_l3_tmp(::Type{T}) where {T} = _blas3ws(T).diag

function _trtri_tmp(::Type{T}, m::Int, n::Int) where {T}
    ws = _blas3ws(T); b = ws.trtri
    if size(b, 1) < m || size(b, 2) < n
        b = Matrix{T}(undef, m, n); ws.trtri = b
    end
    return view(b, 1:m, 1:n)
end

function _trsm_tmp(::Type{T}, m::Int, n::Int) where {T}
    ws = _blas3ws(T); b = ws.trsm_tmp
    if size(b, 1) < m || size(b, 2) < n
        b = Matrix{T}(undef, m, n); ws.trsm_tmp = b
    end
    return b
end

function _l3_apad(::Type{T}, k::Int) where {T}   # ld = k+8 (non-po2) to dodge cache-set aliasing
    ws = _blas3ws(T); b = ws.apad
    if size(b, 1) < k + 8 || size(b, 2) < k
        b = Matrix{T}(undef, k + 8, k); ws.apad = b
    end
    return view(b, 1:k, 1:k)
end

# Side-R fused-leaf pT scratch with an ODD leading dim: an odd ld can never be a multiple of the (power-of-2)
# L1 way stride, so the leaf's solved-column re-reads are conflict-free — vs an in-place po2/way-stride ldb
# where they collide in one set. Grown odd-and-only (never shrinks) so the ld stays odd across reuse.
function _trsm_rpack(::Type{T}, rows::Int, cols::Int) where {T}
    ws = _blas3ws(T); b = ws.rpack
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
    ws = _blas3ws(T); b = ws.rrefl
    need = k + 8; iseven(need) && (need += 1)
    if size(b, 1) < need || size(b, 2) < k
        b = Matrix{T}(undef, need, k); ws.rrefl = b
    end
    return b
end

function _trsm_fused_buf(::Type{T}, len::Int) where {T}   # side-L gemmtrsm leaf: P stripe + recip (flat)
    ws = _blas3ws(T); b = ws.ftrsm
    length(b) < len && (b = Vector{T}(undef, len); ws.ftrsm = b)
    return b
end

function _potf2_buf(::Type{T}, n::Int) where {T}
    ws = _blas3ws(T); b = ws.potf2
    if size(b, 1) < n
        b = Matrix{T}(undef, n, n); ws.potf2 = b
    end
    return view(b, 1:n, 1:n)
end

# potrf whole-matrix pad: an alias-free leading dim (n+8, bumped +8 more if that ld would ITSELF land on the
# L1 quarter-way stride) so the generic potrf recursion's trailing trsm!/syrk! read the copy conflict-free.
function _potrf_pad(::Type{T}, n::Int) where {T}
    ws = _blas3ws(T); b = ws.padf
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
    ws = _factws(T); b = ws.bandl
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
    ws = _blas3ws(T)
    bt = ws.strbt
    if size(bt, 1) != k || size(bt, 2) != n
        bt = Matrix{T}(undef, k, n); ws.strbt = bt
    end
    return bt
end

function _pbtrf_work(::Type{T}, nb::Int) where {T}
    ws = _factws(T)
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
    ws = _factws(T)
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
    ws = _factws(T); b = ws.sylw
    ld = n + 8                                          # off the way stride (same lore as _l3_apad)
    (ld * sizeof(T)) % (_L1_WAY_BYTES >> 2) == 0 && (ld += 8)
    if size(b, 1) != ld || size(b, 2) < nb + 1
        b = Matrix{T}(undef, ld, nb + 1); ws.sylw = b
    end
    return view(b, 1:n, 1:nb), view(b, 1:nb, nb + 1)
end

function _gemm_scratch(::Type{T}, lenA::Int, lenB::Int) where {T}
    ws = _blas3ws(T)
    length(ws.gpackA) < lenA && resize!(ws.gpackA, lenA)
    length(ws.gpackB) < lenB && resize!(ws.gpackB, lenB)
    return ws.gpackA, ws.gpackB
end

function _gemm_scratch_cmplx(::Type{T}, lenA::Int, lenB::Int) where {T}
    t = _blas3ws(T).cg
    length(t[1]) < lenA && (resize!(t[1], lenA); resize!(t[2], lenA))
    length(t[3]) < lenB && (resize!(t[3], lenB); resize!(t[4], lenB))
    return t
end

function _syr2k_scratch(::Type{T}, lenA::Int, lenB::Int) where {T}
    t = _blas3ws(T).s2
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
    t = _blas3ws(Tr).m3
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
    p = _blas3ws(Tr).str
    return _str_fit!(p, 1, mp, kp, Tr), _str_fit!(p, 2, kp, np, Tr), _str_fit!(p, 3, mp, np, Tr)
end
function _strassen_lvl_scratch(::Type{Tr}, level::Int, mh::Int, nh::Int, kh::Int) where {Tr}
    p = _blas3ws(Tr).str; b = 3 + level * 10
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
# The REAL-typed buffers (`pstrfv`, `trrfsa`, `trrfst`) are reached through `_blas3ws(real(T))`. That works
# because real(T) ∈ {Float64, Float32} is itself a const-dispatched owner, so it costs the same bare
# field load and needs no second type parameter on `L3Workspace`. For complex `T` it is a different
# owner object from the element-typed buffers, so a routine holding both can never alias them.

# Higham–Hager 1-norm estimator scratch: candidate x, previous iterate v, and (real only) sign vector.
function _lacn_bufs(::Type{T}, n::Int) where {T}
    ws = _eigws(T)
    length(ws.lacnx) < n && (ws.lacnx = Vector{T}(undef, n))
    length(ws.lacnv) < n && (ws.lacnv = Vector{T}(undef, n))
    ns = T <: Real ? n : 0
    length(ws.lacnsgn) < ns && (ws.lacnsgn = Vector{Int}(undef, ns))
    return view(ws.lacnx, 1:n), view(ws.lacnv, 1:n), view(ws.lacnsgn, 1:ns)
end

# lauum dense-base scratch: an n×n zeroed copy of the stored triangle. n ≤ _trtri_base() by construction.
function _lauum_tmp(::Type{T}, n::Int) where {T}
    ws = _factws(T); b = ws.lauumt
    if size(b, 1) < n || size(b, 2) < n
        b = Matrix{T}(undef, n, n); ws.lauumt = b
    end
    return view(b, 1:n, 1:n)
end

# getri blocked-inversion panel W (n×nb).
function _getri_work(::Type{T}, n::Int, nb::Int) where {T}
    ws = _factws(T); b = ws.getriw
    if size(b, 1) < n || size(b, 2) < nb
        b = Matrix{T}(undef, n, nb); ws.getriw = b
    end
    return view(b, 1:n, 1:nb)
end

# pstrf scratch: `work` is REAL (running dot products, length 2n) and lives on the real owner; `scr` is
# element-typed (length nb + 2n). Distinct owners for complex T ⇒ no aliasing.
function _pstrf_work(::Type{T}, n::Int) where {T}
    R = real(T); ws = _factws(R)
    length(ws.pstrfv) < 2n && (ws.pstrfv = Vector{R}(undef, 2n))
    return view(ws.pstrfv, 1:(2n))
end
function _pstrf_scr(::Type{T}, len::Int) where {T}
    ws = _factws(T)
    length(ws.pstrfs) < len && (ws.pstrfs = Vector{T}(undef, len))
    return view(ws.pstrfs, 1:len)
end

# trrfs scratch: residual r (element-typed) plus two REAL accumulators on the real owner.
function _trrfs_work(::Type{T}, n::Int) where {T}
    R = real(T); ws = _factws(T); wr = _factws(R)
    length(ws.trrfsr) < n && (ws.trrfsr = Vector{T}(undef, n))
    length(wr.trrfsa) < n && (wr.trrfsa = Vector{R}(undef, n))
    length(wr.trrfst) < n && (wr.trrfst = Vector{R}(undef, n))
    return view(ws.trrfsr, 1:n), view(wr.trrfsa, 1:n), view(wr.trrfst, 1:n)
end

# pstrf blocked pivot bookkeeping: swj/swp, both length nb. One owned Int buffer, returned as two
# DISJOINT views (1:nb and nb+1:2nb) so they cannot alias each other.
function _pstrf_swaps(::Type{T}, nb::Int) where {T}
    ws = _factws(T)
    length(ws.pstrfsw) < 2nb && (ws.pstrfsw = Vector{Int}(undef, 2nb))
    return view(ws.pstrfsw, 1:nb), view(ws.pstrfsw, (nb + 1):(2nb))
end

# Scratch for `_lacn2_estimate` (trsen.jl) — the second Higham–Hager estimator. Distinct fields from
# `_lacn_bufs`; see the struct comment. The caller MUST initialise: unlike the old `fill`/`zeros` these
# buffers carry the previous call's values.
function _lacn2e_bufs(::Type{V}, n::Int) where {V}
    ws = _eigws(V)
    length(ws.lacn2x) < n && (ws.lacn2x = Vector{V}(undef, n))
    length(ws.lacn2v) < n && (ws.lacn2v = Vector{V}(undef, n))
    length(ws.lacn2s) < n && (ws.lacn2s = Vector{Int}(undef, n))
    return view(ws.lacn2x, 1:n), view(ws.lacn2v, 1:n), view(ws.lacn2s, 1:n)
end

# gerfs iterative-refinement scratch: residual r and correction dx (element-typed), plus the real
# accumulator |b| + |op(A)|·|x| on the real owner. Three distinct fields ⇒ no aliasing.
function _gerfs_work(::Type{T}, n::Int) where {T}
    R = real(T); ws = _factws(T); wr = _factws(R)
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
    ws = _factws(T)
    (size(ws.pptrfw, 1) < n || size(ws.pptrfw, 2) < nb) && (ws.pptrfw = Matrix{T}(undef, n, nb))
    (size(ws.pptrfv, 1) < n || size(ws.pptrfv, 2) < nb) && (ws.pptrfv = Matrix{T}(undef, n, nb))
    return view(ws.pptrfw, 1:n, 1:nb), view(ws.pptrfv, 1:n, 1:nb)
end
function _pptrf_upper_work(::Type{T}, n::Int, nb::Int) where {T}
    ws = _factws(T)
    (size(ws.pptrfr, 1) < nb || size(ws.pptrfr, 2) < n) && (ws.pptrfr = Matrix{T}(undef, nb, n))
    (size(ws.pptrfv, 1) < n || size(ws.pptrfv, 2) < nb) && (ws.pptrfv = Matrix{T}(undef, n, nb))
    return view(ws.pptrfr, 1:nb, 1:n), view(ws.pptrfv, 1:n, 1:nb)
end

# Side-L ragged-column-tail staging for trsm (level3.jl:4341). Deliberately NOT `_trsm_tmp`: that arm
# holds its staged B across `_trsm!`, which reaches for `trsm_tmp` again on the complex path
# (level3.jl:2017) — see the `trsmw` field comment for the measured corruption.
function _trsm_widen(::Type{T}, m::Int, n::Int) where {T}
    ws = _blas3ws(T); b = ws.trsmw
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
    ws = _blas3ws(T); b = ws.rpad
    need = rows + 8; iseven(need) && (need += 1)
    if size(b, 1) < need || size(b, 2) < cols
        b = Matrix{T}(undef, need, cols); ws.rpad = b
    end
    return b
end

# geqp3 column-norm tracking. All three are REAL-typed, so they live on `_blas3ws(real(T))` — the same
# const-dispatched owner, no second type parameter needed. Allocated on EVERY geqp3! path.
function _geqp3_norms_work(::Type{T}, n::Int) where {T}
    R = real(T); ws = _lstsqws(R)
    length(ws.qp3vn1) < n && (ws.qp3vn1 = Vector{R}(undef, n))
    length(ws.qp3vn2) < n && (ws.qp3vn2 = Vector{R}(undef, n))
    length(ws.qp3hbuf) < n && (ws.qp3hbuf = Vector{R}(undef, n))
    return view(ws.qp3vn1, 1:n), view(ws.qp3vn2, 1:n), view(ws.qp3hbuf, 1:n)
end

# geqp3 blocked panel (laqps). Reached only when `nb > 1`, which is why the allocation was invisible
# below the blocking threshold. Element-typed, so a different owner from the REAL norms above for
# complex T — they cannot alias.
function _geqp3_panel_work(::Type{T}, n::Int, nb::Int) where {T}
    ws = _lstsqws(T)
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
    ws = _eigws(T)
    (size(ws.mtrv, 1) < n || size(ws.mtrv, 2) < nb) && (ws.mtrv = Matrix{T}(undef, n, nb))
    (size(ws.mtrg, 1) < nb || size(ws.mtrg, 2) < nb) && (ws.mtrg = Matrix{T}(undef, nb, nb))
    (size(ws.mtrw, 1) < nb || size(ws.mtrw, 2) < nc) && (ws.mtrw = Matrix{T}(undef, nb, nc))
    (size(ws.mtrt, 1) < nb || size(ws.mtrt, 2) < nb) && (ws.mtrt = Matrix{T}(undef, nb, nb))
    return ws.mtrv, ws.mtrg, ws.mtrw, ws.mtrt
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

# gehrd! reflector staging. Length ihi-ilo+1 ≤ n; only 1:m is ever touched and m shrinks monotonically,
# so a reused longer buffer is never read past m. No fill!.
function _gehrd_work(::Type{T}, len::Int) where {T}
    ws = _eigws(T); ws.gehrdv = _wsgrow(ws.gehrdv, len)
    return view(ws.gehrdv, 1:len)
end

# orghr!/unghr!'s n×n accumulation target Q, and — SEPARATELY — the reflector staging v.
# Two accessors, not one, and that split is load-bearing for the lint as much as for correctness: the
# public orghr! claims Q and then calls `_orghr_into!`, which is also what ormhr! calls with its OWN
# `mhrq`. If one accessor handed out both, `_orghr_into!` would re-claim `orghrv` from inside orghr!'s
# live claim of it — a self re-claim of exactly the trsm_tmp/rpack shape. So Q belongs to whoever is
# building it and `v` belongs to `_orghr_into!` alone.
# Q needs no fill! here: `_orghr_into!`'s own `fill!(Q, 0)` + unit-diagonal write is its first statement.
function _orghr_q(::Type{T}, n::Int) where {T}
    ws = _eigws(T); ws.orghrq = _wsgrow(ws.orghrq, n, n)
    return view(ws.orghrq, 1:n, 1:n)
end
# `_orghr_into!`'s reflector staging. A DISTINCT field from `gehrdv` — ormhr! → `_orghr_into!` is a real
# nesting chain. v[1]=1 and v[2:m] are written from A before `view(v,1:m)` is read, so no fill!.
function _orghr_v(::Type{T}, len::Int) where {T}
    ws = _eigws(T); ws.orghrv = _wsgrow(ws.orghrv, len)
    return view(ws.orghrv, 1:len)
end

# ormhr!/unmhr!: its OWN Q (filled by the non-destructive `_orghr_into!`, which is also what lets the
# `copy(A)` go) plus the gemm! staging tile for whichever of side='L'/'R' this call took. Neither needs
# a fill!: Q is fully written by _orghr_into!, and the gemm! runs beta=0 over the whole mc×nc tile.
function _ormhr_work(::Type{T}, n::Int, mc::Int, nc::Int) where {T}
    ws = _eigws(T)
    ws.mhrq = _wsgrow(ws.mhrq, n, n)
    ws.mhrc = _wsgrow(ws.mhrc, mc, nc)
    return view(ws.mhrq, 1:n, 1:n), view(ws.mhrc, 1:mc, 1:nc)
end

# Real hseqr!'s wr/wi split. REAL-typed ⇒ `_eigws(real(T))`; for the real method that is the same owner
# object, but writing it this way keeps the idiom uniform and makes the complex case (which claims
# neither — `_zlahqr!` carries the bulge in scalars and is already 0 B) unambiguous.
# NO fill! on entry. One caveat for the caller: on non-convergence (info = i > 0) `_dlahqr!` leaves
# ilo:i unwritten, and the entry unconditionally forms w[i] = Complex(wr[i], wi[i]) for ALL i — so the
# failure branch must zero (or otherwise mark) that range, or it hands back the PREVIOUS call's
# eigenvalues instead of the old undef garbage.
function _hseqr_work(::Type{T}, n::Int) where {T}
    R = real(T); ws = _eigws(R)
    ws.hqrwr = _wsgrow(ws.hqrwr, n)
    ws.hqrwi = _wsgrow(ws.hqrwi, n)
    return view(ws.hqrwr, 1:n), view(ws.hqrwi, 1:n)
end

# trevc! real-path scratch, all three REAL-typed on `_eigws(real(T))`.
# `cnorm` IS ZEROED HERE and that is load-bearing: its only producer accumulates with `+=` over
# 2:n and never writes cnorm[1] at all, while the consumer reads cnorm[j] and cnorm[j-1]
# unconditionally with j reaching 1. A reused buffer would give both a stale cnorm[1] and
# stale-plus-accumulated cnorm[2:n]. xr/xi need no fill!: each branch writes them down to 1:ki before
# any read, and every subsequent read and scaling loop is bounded by 1:ki.
function _trevc_rwork(::Type{T}, n::Int) where {T}
    R = real(T); ws = _eigws(R)
    ws.trevcn = _wsgrow(ws.trevcn, n)
    ws.trevcxr = _wsgrow(ws.trevcxr, n)
    ws.trevcxi = _wsgrow(ws.trevcxi, n)
    cn = view(ws.trevcn, 1:n); fill!(cn, zero(R))
    return cn, view(ws.trevcxr, 1:n), view(ws.trevcxi, 1:n)
end

# trevc! complex-path solution. ELEMENT-typed, so for complex T this is a different owner object from
# the three real buffers above and the two sets cannot alias. No fill!: x[1:ki-1] and x[ki] are written
# before any read and reads never exceed 1:ki.
function _trevc_work(::Type{T}, n::Int) where {T}
    ws = _eigws(T); ws.trevcx = _wsgrow(ws.trevcx, n)
    return view(ws.trevcx, 1:n)
end

# geev!/gees! shared scratch: the gehrd τ (live ACROSS the gehrd!→orghr! pair, hence its own field) and
# the single 0×0 placeholder that replaces four per-call `Zdummy` matrices. geev! and gees! never nest —
# neither calls the other — so one set of fields serves both. τ needs no fill!: gehrd! writes all of
# 1:n-1 on every path, including its `ihi-ilo < 1` early return.
function _geev_work(::Type{T}, n::Int) where {T}
    ws = _eigws(T); ws.geevtau = _wsgrow(ws.geevtau, max(n - 1, 0))
    return view(ws.geevtau, 1:max(n - 1, 0)), ws.geevz0
end

# The real path's COMPLEX eigenvalue staging — the mirror of the pstrfv idiom: a complex buffer needed by
# a real T, so it lives on `_eigws(Complex{R})`, itself a const-dispatched owner. Fully written by hseqr!
# before it is read back, so no fill!.
function _geev_cwork(::Type{R}, n::Int) where {R<:Real}
    ws = _eigws(Complex{R}); ws.geevw = _wsgrow(ws.geevw, n)
    return view(ws.geevw, 1:n)
end

# `_dlaexc!` scratch (real path only; `_ztrexc!` is a pure Givens sweep and claims nothing).
# D is handed out as an EXACT nd×nd view, never the raw 4×4 field: the dnorm loop is `for x in D`, so an
# oversized buffer at nd==3 would fold 7 stale elements into the `thresh` rejection test and silently
# change which swaps are accepted. u1/u2 are fully written by their `R[...]` producers, no fill!.
function _laexc_work(::Type{T}, nd::Int) where {T}
    ws = _eigws(real(T))
    return view(ws.laexcd, 1:nd, 1:nd), ws.laexcu1, ws.laexcu2
end

# `_syl_dlasy2` fixed 4×4/4-vector scratch — shared by `_dtrsyl!` and `_dlaexc!`, which never nest.
# t16 IS ZEROED HERE and it is load-bearing: the k==4 arm writes only twelve of sixteen entries, leaving
# [1,4],[4,1],[2,3],[3,2] untouched, and the complete-pivot search scans all sixteen while the
# elimination reads and updates them — stale values pick a wrong pivot, i.e. a wrong ANSWER.
# btmp/jpiv/tmpv need none (btmp fully written by its literal; jpiv[4] never written AND never read;
# tmpv written back-to-front so each read is of an already-written slot).
function _dlasy2_work(::Type{T}) where {T}
    ws = _eigws(real(T))
    fill!(ws.syl16, zero(real(T)))
    return ws.syl16, ws.sylbt, ws.sylpv, ws.syltv
end


# trsen! block scratch. All four are fully overwritten by the copyto!/broadcast that replaces their old
# `collect(view(…))`, and Xm by `copyto!(Xm, xv)` as the first statement of the apply closure, so none
# needs a fill!. See the `trsenx` field comment for the SubArray-vs-Matrix caveat on _dtrsyl!.
function _trsen_work(::Type{T}, n1::Int, n2::Int) where {T}
    ws = _eigws(T)
    ws.trsenr = _wsgrow(ws.trsenr, n1, n2)
    ws.trsen11 = _wsgrow(ws.trsen11, n1, n1)
    ws.trsen22 = _wsgrow(ws.trsen22, n2, n2)
    ws.trsenx = _wsgrow(ws.trsenx, n1, n2)
    return view(ws.trsenr, 1:n1, 1:n2), view(ws.trsen11, 1:n1, 1:n1),
           view(ws.trsen22, 1:n2, 1:n2), view(ws.trsenx, 1:n1, 1:n2)
end

# 1-element scale carrier for trsen!'s `_lacn2_estimate` closure — see the `trsensc` field comment. The
# caller seeds slot 1 before the estimate; the closure overwrites it on every kase.
function _trsen_sc(::Type{T}) where {T}
    ws = _eigws(T)
    ws.trsensc = _wsgrow(ws.trsensc, 1)
    return ws.trsensc
end

# ── Bunch–Kaufman inverse ───────────────────────────────────────────────────────────────────────────

# sytri! AND hetri! — one field, one role: two entries onto the same engine, differing only in a `herm`
# Bool, neither re-entrant. Element-typed (it holds columns of A). No fill!: every read is preceded by a
# copyto! of exactly the slice about to be read, and the tail past m = n-k (or k-1) is never touched.
function _sytri_work(::Type{T}, n::Int) where {T}
    ws = _factws(T); ws.sytriw = _wsgrow(ws.sytriw, n)
    return view(ws.sytriw, 1:n)
end

# ── Generalized eigenproblem / QZ ───────────────────────────────────────────────────────────────────

# Real hgeqz!'s alphar/alphai scratch plus `_hgeqz!`'s length-3 shift vector. REAL-typed. The complex
# entry and `_zhgeqz!` claim none of these — measured 0 B at every size. No fill!: `_hgeqz!` assigns
# both alpha halves at every index on every exit path, and all three of v's slots are written
# immediately before each `_qz_larfg!` call.
function _hgeqz_work(::Type{T}, n::Int) where {T}
    R = real(T); ws = _geneigws(R)
    ws.hgzar = _wsgrow(ws.hgzar, n)
    ws.hgzai = _wsgrow(ws.hgzai, n)
    return view(ws.hgzar, 1:n), view(ws.hgzai, 1:n)
end
# `_hgeqz!`'s length-3 shift vector, in its OWN accessor rather than bundled above: alphar/alphai are
# claimed by the hgeqz! ENTRY and `v` by the `_hgeqz!` engine the entry then calls, so one bundled
# accessor would re-claim alphar/alphai from inside the entry's live claim of them.
_hgeqz_v(::Type{T}) where {T} = _geneigws(real(T)).hgzv

# tgevc! element-typed scratch: the real path's packed n×6 W and the complex path's 2n `work`. One call
# takes one of the two. W stays a single matrix because the engine indexes it as W[j, 3+jw] with a
# runtime column offset. COLUMNS 3–4 ARE ZEROED here, once per call. Col 3 is belt-and-braces (the
# engine re-clears it every je-iteration anyway); col 4 is the one that matters — at nw==1 it is never
# written, yet W[j,4]/W[j+1,4] are still passed to `_laln2` in live argument slots. They are dead inside
# `_laln2` for na==1, so this is not a wrong answer, but an undef field would be feeding uninitialised
# memory into an argument. Cols 1–2 need none (written for every j in 2:n, and index 1 is never read);
# cols 5–6 and all of `work` are fully written before read on every je-iteration.
function _tgevc_work(::Type{T}, n::Int) where {T}
    ws = _geneigws(T)
    ws.tgvw = _wsgrow(ws.tgvw, n, 6)
    ws.tgvx = _wsgrow(ws.tgvx, 2n)
    W = view(ws.tgvw, 1:n, 1:6); fill!(view(W, :, 3:4), zero(T))
    return W, view(ws.tgvx, 1:(2n))
end

# tgevc! complex-path `rwork`. REAL-typed ⇒ a DIFFERENT owner object from `tgvx` for complex T, so the
# two cannot alias. rwork[j]/rwork[n+j] are written for every j in 2:n before any read, and rwork[1] /
# rwork[n+1] are never read, so no fill!.
function _tgevc_rwork(::Type{T}, n::Int) where {T}
    R = real(T); ws = _geneigws(R); ws.tgvr = _wsgrow(ws.tgvr, 2n)
    return view(ws.tgvr, 1:(2n))
end

# `_dtgsen!`'s alphar/alphai scratch (the entry combines them into the caller's complex alpha). REAL.
# No fill!: the k-loop assigns alphar[k]/alphai[k]/beta[k] at every k on both the 2×2 and 1×1 branch.
function _tgsen_alpha_work(::Type{R}, n::Int) where {R}
    ws = _geneigws(R)
    ws.tgsar = _wsgrow(ws.tgsar, n)
    ws.tgsai = _wsgrow(ws.tgsai, n)
    return view(ws.tgsar, 1:n), view(ws.tgsai, 1:n)
end

# `_tgs_tgsy2!` / `_tgs_getc2!`: the Kronecker system Z (nz = 2·n1·n2 ≤ 8) with its rhs and two pivot
# records. Z IS ZEROED and that is load-bearing — it is built entirely by `+=`/`-=` accumulation with no
# prior full write. rhs is fully written (all nz slots) and the pivot vectors are written by getc2
# before any unwind reads them, so neither needs one.
function _tgsy2_work(::Type{R}, nz::Int) where {R}
    ws = _geneigws(R)
    Z = view(ws.tgxz, 1:nz, 1:nz); fill!(Z, zero(R))
    return Z, view(ws.tgxrhs, 1:nz), view(ws.tgxip, 1:nz), view(ws.tgxjp, 1:nz)
end

# `_dtgex2_big!` diagonal blocks. m = n1+n2 ≤ 4, n1,n2 ∈ {1,2} — all four fully written by their copy
# loops, no fill!. Exact views so a shape bug trips @boundscheck rather than reading a stale corner.
function _tgex2_blocks(::Type{R}, m::Int, n1::Int, n2::Int) where {R}
    ws = _geneigws(R)
    return view(ws.tgxs, 1:m, 1:m), view(ws.tgxt, 1:m, 1:m),
           view(ws.tgxc, 1:n1, 1:n2), view(ws.tgxf, 1:n1, 1:n2)
end

# `_dtgex2_big!` QR/RQ stage: LI, IR and the four τ vectors plus the two reflector staging vectors
# (`tgxvq` for `_tgs_geqr2!`, `tgxvr` for `_tgs_gerq2!` — one field per role, they are separate helpers).
# LI AND IR ARE ZEROED and both are load-bearing: LI has only LI[1:n1,1:n2] and a unit sub-diagonal
# written yet `_tgs_geqr2!` reads all of LI[1:m,1:n2]; IR has only two sparse patterns written yet
# `_tgs_gerq2!` reads rows n2+1:n2+n1 across all of cols 1:m. (Zeroing the whole m×m covers the columns
# `_tgs_org2r!`/`_tgs_orgr2!` would have cleared themselves — 16 elements, not worth splitting.)
# The τ and v fields need none: v[1..lv] is filled before each `_qz_larfg!` and τ[i] assigned before use.
function _tgex2_qr(::Type{R}, m::Int) where {R}
    ws = _geneigws(R)
    LI = view(ws.tgxli, 1:m, 1:m); fill!(LI, zero(R))
    IR = view(ws.tgxir, 1:m, 1:m); fill!(IR, zero(R))
    return LI, IR, view(ws.tgxtaul, 1:m), view(ws.tgxtaur, 1:m),
           view(ws.tgxtaul2, 1:m), view(ws.tgxtaur2, 1:m), ws.tgxvq, ws.tgxvr
end

# `_tgs_matmul` destinations. `tgxm1`/`tgxm2` are a ping/pong pair because the call sites nest
# inner-inside-outer and BOTH results are live at once; PA/PB are the two independent outer products.
# Every element is assigned by `_tgs_matmul`'s own loop, so none needs a fill!.
function _tgex2_mul(::Type{R}, m::Int) where {R}
    ws = _geneigws(R)
    return view(ws.tgxm1, 1:m, 1:m), view(ws.tgxm2, 1:m, 1:m),
           view(ws.tgxpa, 1:m, 1:m), view(ws.tgxpb, 1:m, 1:m)
end

# The four route-selection copies (SCPY/TCPY/IRCOP/LICOP). Fully written by their copy, no fill!.
function _tgex2_copies(::Type{R}, m::Int) where {R}
    ws = _geneigws(R)
    return view(ws.tgxscpy, 1:m, 1:m), view(ws.tgxtcpy, 1:m, 1:m),
           view(ws.tgxircop, 1:m, 1:m), view(ws.tgxlicop, 1:m, 1:m)
end

# `_dtgex2_big!` rotation stage. QL2 AND IR2 ARE ZEROED, load-bearing: only their (1,1)/(m,m) entries
# and two 2×2 corner blocks are set, and both are then read IN FULL by `_tgs_matmul` and by the
# tmpA/tmpB loops. tmpA/tmpB/w are fully written before read.
function _tgex2_rot(::Type{R}, m::Int, n1::Int, n2::Int) where {R}
    ws = _geneigws(R)
    QL2 = view(ws.tgxql2, 1:m, 1:m); fill!(QL2, zero(R))
    IR2 = view(ws.tgxir2, 1:m, 1:m); fill!(IR2, zero(R))
    return QL2, IR2, view(ws.tgxta, 1:n2, 1:n1), view(ws.tgxtb, 1:n2, 1:n1), view(ws.tgxw, 1:m)
end

# The two Q/Z accumulate buffers — the ONLY part of `_dtgex2_big!` that scales with n, and therefore the
# whole O(n²) term. Separate fields per role even though the two blocks run sequentially. Fully written
# by their gather loops, no fill!.
function _tgex2_accum(::Type{R}, n::Int, m::Int) where {R}
    ws = _geneigws(R)
    ws.tgxbufq = _wsgrow(ws.tgxbufq, n, m)
    ws.tgxbufz = _wsgrow(ws.tgxbufz, n, m)
    return view(ws.tgxbufq, 1:n, 1:m), view(ws.tgxbufz, 1:n, 1:m)
end

# `_ggev_qrB!`'s four buffers — shared by ggev! and gges!, which reach them through the identical call
# and never nest. Q and tmp must stay separate fields: the gemm! reads Q while writing tmp. No fill!:
# `_ggev_formQ!` opens with its own `fill!(Q, 0)`, the gemm! runs beta=0, tauL is fully written by its
# loop, and tau is written by geqrf! over 1:min(m,n) = 1:n for the square B this path always passes.
function _ggev_qrb_work(::Type{T}, n::Int) where {T}
    ws = _geneigws(T)
    ws.ggqtau = _wsgrow(ws.ggqtau, n)
    ws.ggqtaul = _wsgrow(ws.ggqtaul, n)
    ws.ggqq = _wsgrow(ws.ggqq, n, n)
    ws.ggqtmp = _wsgrow(ws.ggqtmp, n, n)
    return view(ws.ggqtau, 1:n), view(ws.ggqtaul, 1:n),
           view(ws.ggqq, 1:n, 1:n), view(ws.ggqtmp, 1:n, 1:n)
end

# Real ggev!'s complex alphaC staging — same mirror as `_geev_cwork`, on `_geneigws(Complex{R})`, but a
# DISTINCT field (geev! and ggev! are unrelated routines). hgeqz! assigns alpha[i] for every i in 1:n
# before it is read back, so no fill!.
function _ggev_cwork(::Type{R}, n::Int) where {R<:Real}
    ws = _geneigws(Complex{R}); ws.ggevac = _wsgrow(ws.ggevac, n)
    return view(ws.ggevac, 1:n)
end

# ── ggsvd! ──────────────────────────────────────────────────────────────────────────────────────────

# `_ggs_larf_right!`'s accumulator — one allocation per reflector application before this, i.e. O(n²)
# bytes per ggsvd! call. THE FILL IS LOAD-BEARING and is done here, not left to the caller: the
# accumulation loop is guarded by `if u[k] != 0`, so a column of zeros writes nothing at all, while the
# consumer reads w[i] unconditionally. Do NOT add a capacity guard around it — a skipped fill here is an
# active wrong answer, not a latent one.
function _ggsvd_larf_w(::Type{T}, len::Int) where {T}
    ws = _gsvdws(T); ws.ggs_w = _wsgrow(ws.ggs_w, len)
    w = view(ws.ggs_w, 1:len); fill!(w, zero(T))
    return w
end

# `_ggs_gerq2!`'s reflector record, replacing the `Vector{Tuple{Int,T,Vector{T}}}` carrier: column i of
# W is reflector i's retained `wv`, with its `c` and τ in two parallel vectors. The second O(n²) site.
# Each column is written over 1:c before it is applied, so no fill!.
function _ggsvd_rq_work(::Type{T}, rows::Int, k::Int) where {T}
    ws = _gsvdws(T)
    ws.ggs_rqw = _wsgrow(ws.ggs_rqw, rows, k)
    ws.ggs_rqc = _wsgrow(ws.ggs_rqc, k)
    ws.ggs_rqtau = _wsgrow(ws.ggs_rqtau, k)
    return view(ws.ggs_rqw, 1:rows, 1:k), view(ws.ggs_rqc, 1:k), view(ws.ggs_rqtau, 1:k)
end

# The five `u` roles, one field each. They are NOT shared: `_ggs_geqpf!`'s u is live across a
# `_ggs_larf_left!` that any shared buffer would clobber, and the other four belong to different
# helpers. Each writes u[1] plus the tail covering exactly the view it then passes on, so none is zeroed.
_ggsvd_u_qp(::Type{T}, len::Int) where {T} =
    (ws = _gsvdws(T); ws.ggs_u = _wsgrow(ws.ggs_u, len); view(ws.ggs_u, 1:len))
_ggsvd_u_applyl(::Type{T}, len::Int) where {T} =
    (ws = _gsvdws(T); ws.ggs_ul = _wsgrow(ws.ggs_ul, len); view(ws.ggs_ul, 1:len))
_ggsvd_u_formq(::Type{T}, len::Int) where {T} =
    (ws = _gsvdws(T); ws.ggs_uq = _wsgrow(ws.ggs_uq, len); view(ws.ggs_uq, 1:len))
_ggsvd_u_rq(::Type{T}, len::Int) where {T} =
    (ws = _gsvdws(T); ws.ggs_ur = _wsgrow(ws.ggs_ur, len); view(ws.ggs_ur, 1:len))
_ggsvd_u_step5(::Type{T}, len::Int) where {T} =
    (ws = _gsvdws(T); ws.ggs_u5 = _wsgrow(ws.ggs_u5, len); view(ws.ggs_u5, 1:len))

# `_ggs_permcols!`'s Xc. Called on A (m×n) and on view(Q,:,1:nl) (n×nl), so it must fit both — the
# caller passes the shape it needs. Filled by a full row×column nest, no fill!.
function _ggsvd_permcols(::Type{T}, rows::Int, cols::Int) where {T}
    ws = _gsvdws(T); ws.ggs_xc = _wsgrow(ws.ggs_xc, rows, cols)
    return view(ws.ggs_xc, 1:rows, 1:cols)
end

# The two pivoted-QR τ vectors and the two column-pivot vectors — FOUR accessors, not two grouped pairs.
# taua/jp2 are claimed on A11 only after `l` is known, i.e. long after taub/jpvt were claimed on B, so a
# grouped accessor could not be called with both lengths in hand. Separate fields regardless: taub still
# feeds `_ggs_formQ!(V, …)` while taua is live, and jpvt is consumed after jp2 is created.
# No fill! on any: τ is written over 1:min(mm,nn) — the range later read — and `_ggs_geqpf!` writes
# jpvt[j] = j over the full length it is handed.
_ggsvd_taub(::Type{T}, kb::Int) where {T} =
    (ws = _gsvdws(T); ws.ggs_taub = _wsgrow(ws.ggs_taub, kb); view(ws.ggs_taub, 1:kb))
_ggsvd_taua(::Type{T}, ka::Int) where {T} =
    (ws = _gsvdws(T); ws.ggs_taua = _wsgrow(ws.ggs_taua, ka); view(ws.ggs_taua, 1:ka))
_ggsvd_jpvt(::Type{T}, n::Int) where {T} =
    (ws = _gsvdws(T); ws.ggs_jpvt = _wsgrow(ws.ggs_jpvt, n); view(ws.ggs_jpvt, 1:n))
_ggsvd_jp2(::Type{T}, nl::Int) where {T} =
    (ws = _gsvdws(T); ws.ggs_jp2 = _wsgrow(ws.ggs_jp2, nl); view(ws.ggs_jp2, 1:nl))

# `_ggs_tgsja!`'s wx/wy. Both write t in 1:len before the 1:len view is used, so no fill!.
function _ggsvd_tgsja_work(::Type{T}, len::Int) where {T}
    ws = _gsvdws(T)
    ws.ggs_wx = _wsgrow(ws.ggs_wx, len)
    ws.ggs_wy = _wsgrow(ws.ggs_wy, len)
    return view(ws.ggs_wx, 1:len), view(ws.ggs_wy, 1:len)
end

# ── Least squares ───────────────────────────────────────────────────────────────────────────────────

# gels!: the op(A) working copy (which exists because gels! deliberately does not overwrite A) and the
# Householder τ. Both fully written before read — M by the op(A) copy, τ[1:k] by geqrf!/_geqr2! — and
# entries past k are never read, so a longer τ buffer is safe uninitialised.
function _gels_work(::Type{T}, p::Int, q::Int, k::Int) where {T}
    ws = _lstsqws(T)
    ws.gelsm = _wsgrow(ws.gelsm, p, q)
    ws.gelstau = _wsgrow(ws.gelstau, k)
    return view(ws.gelsm, 1:p, 1:q), view(ws.gelstau, 1:k)
end

# gels!'s adjoint copy — UNDERDETERMINED ARM ONLY, hence its own accessor: the square/overdetermined
# path never touches it and so never grows it. Fully written by the adjoint copy.
function _gels_adj(::Type{T}, q::Int, p::Int) where {T}
    ws = _lstsqws(T); ws.gelsmh = _wsgrow(ws.gelsmh, q, p)
    return view(ws.gelsmh, 1:q, 1:p)
end

# gelsd!: the SVD factors, the Uᴴ·b staging, and the destructible A copy. None needs a fill! — U/Vt/A
# are fully written by gesvd! and the copy, and C is written by a beta=0 gemm! before the
# read-modify-write that applies Σ⁺. `gelsda` becomes dead if gelsd! is ever allowed to destroy A, as
# its own docstring already claims it does.
function _gelsd_work(::Type{T}, m::Int, n::Int, mn::Int, nrhs::Int) where {T}
    ws = _lstsqws(T)
    ws.gelsdu = _wsgrow(ws.gelsdu, m, mn)
    ws.gelsdvt = _wsgrow(ws.gelsdvt, mn, n)
    ws.gelsdc = _wsgrow(ws.gelsdc, mn, nrhs)
    ws.gelsda = _wsgrow(ws.gelsda, m, n)
    return view(ws.gelsdu, 1:m, 1:mn), view(ws.gelsdvt, 1:mn, 1:n),
           view(ws.gelsdc, 1:mn, 1:nrhs), view(ws.gelsda, 1:m, 1:n)
end

# The Float32-real gelsd! wrapper's Float64 staging. Lives on `_lstsqws(Float64)` — the mirror of the
# real()-indirection, reached UP from a Float32 entry rather than DOWN from a complex one, so it is a
# different owner object from the Float32 buffers above and cannot alias them. All three fully written
# (Bd by copyto!, Ad by the conversion, sd by the nested call) before any read.
function _gelsd_promote_work(m::Int, n::Int, mn::Int, nrhs::Int)
    ws = _lstsqws(Float64)
    ws.gelsdad = _wsgrow(ws.gelsdad, m, n)
    ws.gelsdbd = _wsgrow(ws.gelsdbd, m, nrhs)
    ws.gelsdsd = _wsgrow(ws.gelsdsd, mn)
    return view(ws.gelsdad, 1:m, 1:n), view(ws.gelsdbd, 1:m, 1:nrhs), view(ws.gelsdsd, 1:mn)
end

# gelsy!'s five vectors. xmin/xmax are separate fields (both live across the whole rank loop) and tauz
# is separate from tau (tau is still read by `_apply_Qh!` after tauz is filled). No fill! on any: tau by
# geqp3!, xmin/xmax written at index rank+1 before rank increments and never read past rank, tauz by
# tzrzf! over 1:m on both arms, w[jpvt[i]] over a guaranteed permutation of 1:n.
# CALLER CONTRACT: ormrz! derives k = length(tau) and L = size(A,2) - k, so it MUST be passed
# view(tauz, 1:rank) — handing it the whole grown field silently changes k and L and corrupts the result.
# That is why this returns exact-length views and not the raw buffers.
function _gelsy_work(::Type{T}, mn::Int, n::Int) where {T}
    ws = _lstsqws(T)
    ws.gelsytau = _wsgrow(ws.gelsytau, mn)
    ws.gelsyxmin = _wsgrow(ws.gelsyxmin, max(mn, 1))
    ws.gelsyxmax = _wsgrow(ws.gelsyxmax, max(mn, 1))
    ws.gelsyw = _wsgrow(ws.gelsyw, n)
    return view(ws.gelsytau, 1:mn), view(ws.gelsyxmin, 1:max(mn, 1)),
           view(ws.gelsyxmax, 1:max(mn, 1)), view(ws.gelsyw, 1:n)
end

# The RZ τ, in its own accessor because `rank` is not known until the `_laic1` loop above has finished —
# it cannot be claimed in the same call as `tau`/`xmin`/`xmax`. Returns an EXACT-length view: ormrz!
# derives k = length(tau) and L = size(A,2) - k, so handing it the whole grown field would silently
# change k and L and corrupt the result.
function _gelsy_tauz(::Type{T}, rank::Int) where {T}
    ws = _lstsqws(T); ws.gelsytauz = _wsgrow(ws.gelsytauz, max(rank, 1))
    return view(ws.gelsytauz, 1:rank)
end

# tzrzf!'s (z)larfg staging row. Its own accessor because tzrzf! is a public entry in its own right, not
# only a gelsy! internal. buf[1] and buf[2:L+1] are both written at the top of each iteration, no fill!.
function _tzrzf_buf(::Type{T}, len::Int) where {T}
    ws = _lstsqws(T); ws.tzrzfb = _wsgrow(ws.tzrzfb, len)
    return view(ws.tzrzfb, 1:len)
end

# gglse!: G, Ã = A·G, the QR τ, the Zᴴ·c staging, the transformed solution y and the residual r.
# `gglc` is handed out as an m×1 MATRIX view so `_ggl_apply_Zh!` can drop its `reshape(c, m, 1)` — a
# reshape of a SubArray routes to Base._throw_dmrs and fails `juliac --trim=safe`. Nothing needs a fill!:
# G is written by its explicit `G := I` nest, Ã and r by beta=0 gemm!/gemv!, τ by `_ggl_geqr2!` over
# 1:min(m,n), chat by its copy loop, and y in two halves that together cover all of 1:n.
function _gglse_work(::Type{T}, m::Int, n::Int, k::Int) where {T}
    ws = _lstsqws(T)
    ws.gglg = _wsgrow(ws.gglg, n, n)
    ws.gglat = _wsgrow(ws.gglat, m, n)
    ws.gglt = _wsgrow(ws.gglt, k)
    ws.gglc = _wsgrow(ws.gglc, m, 1)
    ws.gglyy = _wsgrow(ws.gglyy, n)
    ws.gglr = _wsgrow(ws.gglr, m)
    return view(ws.gglg, 1:n, 1:n), view(ws.gglat, 1:m, 1:n), view(ws.gglt, 1:k),
           view(ws.gglc, 1:m, 1:1), view(ws.gglyy, 1:n), view(ws.gglr, 1:m)
end

# ── Symmetric-tridiagonal eigen (stebz.jl) ──────────────────────────────────────────────────────────

# stebz! scratch. `e2`'s old allocator was `zeros` and the zeroing was CHECKED and is not load-bearing:
# the split loop writes every index 1..n-1 unconditionally on both branches before either reader runs,
# and the only allocated-but-unwritten slot (e2[1] at n==1) is never read. The two sortperm destinations
# get SEPARATE fields — the audit argued one suffices because the first is dead by the time the second
# is built, which is the never-overlap reasoning lesson 4 forbids. idx/gw/gi are fully written by the
# sort and the gather.
function _stebz_work(::Type{T}, n::Int) where {T}
    ws = _tridiagws(T)
    ws.stbe2 = _wsgrow(ws.stbe2, max(n - 1, 1))
    ws.stbperm = _wsgrow(ws.stbperm, n)
    ws.stbperm2 = _wsgrow(ws.stbperm2, n)
    ws.stbidx = _wsgrow(ws.stbidx, n)
    ws.stbgw = _wsgrow(ws.stbgw, n)
    ws.stbgi = _wsgrow(ws.stbgi, n)
    return view(ws.stbe2, 1:max(n - 1, 1)), view(ws.stbperm, 1:n), view(ws.stbperm2, 1:n),
           view(ws.stbidx, 1:n), view(ws.stbgw, 1:n), view(ws.stbgi, 1:n)
end

# stein! scratch: the four dlagtf vectors, the inverse-iteration iterate, the interchange record and the
# xorshift seed. None of the six arrays needs a fill! (av/bv/cv from copyto!,
# rhs from `_stein_randvec!`, inn fully written by `_dlagtf!`, and the one d2 slot left unwritten at
# bz==2 is never read).
# THE SEED IS RESET TO `_STEIN_SEED0` HERE, and that is a correction to the audit, not an oversight.
# stebz.jl:306 builds `Ref(0x2545F4914F6CDD1D)` INSIDE stein!, so today every call starts from the same
# seed and identical input gives bit-identical output. The seed "advances" only WITHIN one call (across
# the blocks) — which is what the comment at stebz.jl:268-273 means. Simply owning the RefValue would
# have made the stream continue across calls, so call 2 on the same matrix would take a different
# inverse-iteration restart and return a different (still valid, but not identical) eigenvector basis —
# a reproducibility regression, and one test/reproducibility_tests.jl exists to catch. Resetting here
# keeps today's semantics exactly while still removing the per-call 16 B RefValue.
# NOT COVERED HERE, deliberately: stein!'s Z output. It is dynamically sized (n × length(w)) and MUST be
# `fill!`ed by whoever supplies it — only rows b1:bn of each column are ever written, and a tridiagonal
# eigenvector has to be exactly zero outside its own split block, so an undef Z is a wrong answer.
function _stein_work(::Type{T}, n::Int) where {T}
    ws = _tridiagws(T)
    ws.stnav = _wsgrow(ws.stnav, n)
    ws.stnbv = _wsgrow(ws.stnbv, n)
    ws.stncv = _wsgrow(ws.stncv, n)
    ws.stnd2 = _wsgrow(ws.stnd2, n)
    ws.stnrhs = _wsgrow(ws.stnrhs, n)
    ws.stninn = _wsgrow(ws.stninn, n)
    ws.stnseed[] = _STEIN_SEED0
    return view(ws.stnav, 1:n), view(ws.stnbv, 1:n), view(ws.stncv, 1:n),
           view(ws.stnd2, 1:n), view(ws.stnrhs, 1:n), view(ws.stninn, 1:n), ws.stnseed
end
