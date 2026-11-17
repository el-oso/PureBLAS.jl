# Precompile-time guarantees for the SIMDBackend. Included LAST, after every backend op and C-ABI entry
# is defined, because the declarations below name them.
#
# `@verify_strict SIMDBackend` runs `TypeContracts.@verify SIMDBackend` — method existence + declared
# return types over the whole chain (AbstractBLAS1 → BLAS2 → BLAS3 → LAPACK → LAPACKSolve) — and checks
# each representative call listed in it at the argument types it names:
#   :typestable — the inferred return type must be concrete. This GATES: a failure stops PureBLAS from
#                 precompiling.
#   :noalloc    — StrictMode's value-free IR scan. This REPORTS: a finding is a `StrictViolation`
#                 warning, recorded in `_LOADTIME_WARNINGS` below and compared against
#                 test/loadtime_warnings_baseline.txt by test/loadtime_warnings_tests.jl.
# The listed calls are NOT performed: every check reads argument types only, so the operands in the
# `let` exist to name those types and their values are never used. The proofs of both guarantees —
# JET for stability, AllocCheck for allocation, TrimCheck for trim — run from test/ (StrictModeTest is a
# test-only dependency, so it cannot load at this package's own precompile).
#
# Do NOT use StrictMode's IR heuristic (`_alloc_signals`) as the eligibility test for a new
# member: the control run in bench/probes/strict_heuristic_control.jl flags EVERY already-contracted
# LAPACK member — gemm!, potrf!, getrf!, geqrf!, potrs!, getri!, pstrf!, the whole QR family — at a
# measured 0 B. The predicate that means something here is empirical @allocated after warm-up, plus a
# concrete inferred return type; that is what every member below was admitted on
# (bench/probes/strict_contract_eligibility.jl, strict_contract_dryrun.jl).
#
# Values live in a `let` — only their types matter, so `ones` (no Random dep).
# getrf!/geqrf!/gesvd! are listed at their IN-PLACE forms (caller-provided ipiv/τ, U/S/Vᵀ): the
# convenience forms allocate those outputs, which is inherent, not a bug.

# ── Growth functions excused from the static `:noalloc` guarantee. Each allocates at a new high-water
# mark, so none can be a barrier; the property they do have (finitely many growths, then none) is asserted
# by measurement: test/strictmode_tests.jl warms each entry point and checks that a second identical call
# allocates 0 bytes. These must be in effect before any check below runs.
@strict_exempt _ws_grow! guarantees = (:noalloc,) reason = "grows a pool to a new high-water mark; a warmed second call allocates 0 bytes (test/strictmode_tests.jl)"
@strict_exempt _ws_slot! guarantees = (:noalloc,) reason = "grows a pool slot to a new high-water mark; a warmed second call allocates 0 bytes (test/strictmode_tests.jl)"
@strict_exempt _arena_grow! guarantees = (:noalloc,) reason = "adds a slab at a new arena high-water mark; a warmed second call allocates 0 bytes (test/strictmode_tests.jl)"
@strict_exempt _arena_coalesce! guarantees = (:noalloc,) reason = "folds the slab chain after a call that grew; a warmed second call allocates 0 bytes (test/strictmode_tests.jl)"
@strict_exempt _lu_pad_grow! guarantees = (:noalloc,) reason = "replaces the LU pad scratch at a new size high-water mark; a warmed second call allocates 0 bytes (test/strictmode_tests.jl)"
@strict_exempt _clu_pad_grow! guarantees = (:noalloc,) reason = "replaces the complex LU pad scratch at a new size high-water mark; a warmed second call allocates 0 bytes (test/strictmode_tests.jl)"
@strict_exempt _gvd_grow! guarantees = (:noalloc,) reason = "grows the generalized-eigenproblem workspace to a new size high-water mark; a warmed second call allocates 0 bytes (test/strictmode_tests.jl)"
@strict_exempt _dcwork_grow! guarantees = (:noalloc,) reason = "replaces the divide-and-conquer workspace at a new size high-water mark; a warmed second call allocates 0 bytes (test/strictmode_tests.jl)"
@strict_exempt _gm guarantees = (:noalloc,) reason = "replaces a workspace matrix that is too small; a warmed second call allocates 0 bytes (test/strictmode_tests.jl)"
@strict_exempt _gv guarantees = (:noalloc,) reason = "replaces a workspace vector that is too small; a warmed second call allocates 0 bytes (test/strictmode_tests.jl)"
@strict_exempt _svd_grow! guarantees = (:noalloc,) reason = "grows the SVD workspace to a new size high-water mark; a warmed second call allocates 0 bytes (test/strictmode_tests.jl)"
@strict_exempt _svd_grow_bidiag! guarantees = (:noalloc,) reason = "grows the SVD bidiagonalization panels to a new size high-water mark; a warmed second call allocates 0 bytes (test/strictmode_tests.jl)"
@strict_exempt _svd_grow_dc! guarantees = (:noalloc,) reason = "grows the SVD divide-and-conquer staging to a new size high-water mark; a warmed second call allocates 0 bytes (test/strictmode_tests.jl)"
@strict_exempt _edc_grow! guarantees = (:noalloc,) reason = "grows the divide-and-conquer workspace to a new size high-water mark; a warmed second call allocates 0 bytes (test/strictmode_tests.jl)"
@strict_exempt _trd_grow! guarantees = (:noalloc,) reason = "grows the tridiagonalization panel scratch to a new size high-water mark; a warmed second call allocates 0 bytes (test/strictmode_tests.jl)"
@strict_exempt _qr_ws_c_grow! guarantees = (:noalloc,) reason = "replaces the owned complex QR workspace at a new size high-water mark; a warmed second call allocates 0 bytes (test/strictmode_tests.jl)"
@strict_exempt _qr_ws_grow! guarantees = (:noalloc,) reason = "replaces the owned QR workspace at a new size high-water mark; a warmed second call allocates 0 bytes (test/strictmode_tests.jl)"

# Functions whose allocations the static no-allocation proof treats as expected: each only performs the
# first touch of a task-local owner (the allocation is the one-time initialization of that task's slot,
# and every later call is a lookup). Each must be `@noinline` and be the `:invoke` callee itself to be
# recognized. Functions that grow a pool to a new high-water mark are not here — they allocate again at
# each new mark, so they are `@strict_exempt :noalloc` (above) and asserted by measurement.
const _BARRIER_FUNCTIONS = (_m3ws, _gpackws, _strws, _opt)
# The registry is process-global and not part of the saved image, so this registration covers only the
# checks in this file, which run while PureBLAS precompiles. test/strictmode_tests.jl registers (and
# removes) its own barriers around each proof it runs in the test process.
foreach(StrictMode.register_alloc_barrier!, _BARRIER_FUNCTIONS)

# A heuristic check below reports by logging a warning instead of throwing: a per-call check logs a
# message starting `StrictViolation`, a module policy one starting `StrictMode:`. From here to the end
# of this file the global logger is wrapped so each such message is also pushed onto
# `_LOADTIME_WARNINGS`, which is saved in the package image; test/loadtime_warnings_tests.jl compares it
# against a baseline. Messages still reach the original logger unchanged.
const _LOADTIME_WARNINGS = String[]
struct _WarnRecorder{L <: Base.CoreLogging.AbstractLogger} <: Base.CoreLogging.AbstractLogger
    inner::L
end
Base.CoreLogging.min_enabled_level(r::_WarnRecorder) = Base.CoreLogging.min_enabled_level(r.inner)
Base.CoreLogging.shouldlog(r::_WarnRecorder, args...) = Base.CoreLogging.shouldlog(r.inner, args...)
Base.CoreLogging.catch_exceptions(r::_WarnRecorder) = Base.CoreLogging.catch_exceptions(r.inner)
function Base.CoreLogging.handle_message(r::_WarnRecorder, level, message, args...; kwargs...)
    if level == Base.CoreLogging.Warn && message isa AbstractString &&
            (startswith(message, "StrictViolation") || startswith(message, "StrictMode:"))
        push!(_LOADTIME_WARNINGS, message)
    end
    return Base.CoreLogging.handle_message(r.inner, level, message, args...; kwargs...)
end
Base.CoreLogging.global_logger(_WarnRecorder(Base.CoreLogging.global_logger()))

let bk = DEFAULT_BACKEND, n = 1000, m = 64,
        xd = ones(n), yd = ones(n), xz = ones(ComplexF64, n), yz = ones(ComplexF64, n),
        Ad = ones(m, m), Az = ones(ComplexF64, m, m), um = ones(m), vm = ones(m),
        uz = ones(ComplexF64, m), wz = ones(ComplexF64, m),
        C3 = ones(m, m), A3 = ones(m, m), B3 = ones(m, m), Bt = ones(m, m), At = ones(m, m),
        # SubArray operands for the L3 view assertions below. The parents are LARGER than the view,
        # so `lda != nrows` — the shape the owned-workspace accessors (`_pptrf_lower_work`,
        # `_getri_work`, `_trsm_rrefl`, …) actually produce, and the one a same-size `view` would not
        # exercise. `_strided1` still holds, so these take the same pointer fast paths a Matrix does;
        # what differs is the argument TYPE, which is precisely what went unchecked.
        P3 = ones(m + 8, m + 8), Q3 = ones(m + 8, m + 8), R3 = ones(m + 8, m + 8),
        Pt = ones(m + 8, m + 8), Qt = ones(m + 8, m + 8),
        C3v = view(P3, 1:m, 1:m), A3v = view(Q3, 1:m, 1:m), B3v = view(R3, 1:m, 1:m),
        Btv = view(Pt, 1:m, 1:m), Atv = view(Qt, 1:m, 1:m),
        Cz3 = ones(ComplexF64, m, m), Az3 = ones(ComplexF64, m, m), Bz3 = ones(ComplexF64, m, m),
        # A working matrix the factorizations overwrite; pre-allocated pivot/τ for the in-place
        # LU/QR kernels.
        Aw = zeros(m, m),
        ipiv = Vector{Int}(undef, m), tau = Vector{Float64}(undef, m),
        # QR-family sources: a pristine A, and pre-factored copies for the Q-generators and
        # Q-appliers (which need real reflectors, not raw data).
        Aqr = ones(m, m), Aqw = zeros(m, m), tauq = Vector{Float64}(undef, m),
        jpq = zeros(Int, m), Cqw = zeros(m, m),
        # COMPLEX counterparts. The contract previously verified `symm!`/`syrk!`/`syr2k!`/`trmm!`/
        # `trsm!` and every factorization on REAL operands only, so any defect confined to a complex
        # kernel was invisible to it — which is exactly how ~1 KB/call of `unsafe_wrap` allocation in
        # the complex-only 3M path passed for a month. `Atz` is the triangular operand for
        # ztrmm/ztrsm, `Btz` the one they overwrite, mirroring the real `At`/`Bt` pair.
        Awz3 = zeros(ComplexF64, m, m), tauz = Vector{ComplexF64}(undef, m),
        Atz = ones(ComplexF64, m, m), Btz = ones(ComplexF64, m, m),
        # L2 packed storage (AP length m(m+1)/2) and band storage (kb sub/super-diagonals).
        kb = 8, APd = ones(m * (m + 1) ÷ 2), APz = ones(ComplexF64, m * (m + 1) ÷ 2),
        ABg = ones(2 * 8 + 1, m), ABs = ones(8 + 1, m), ABz = ones(ComplexF64, 8 + 1, m),
        # gesvd in-place output buffers (square m×m: U m×m, S m, Vᵀ m×m).
        Usv = zeros(m, m), Ssv = zeros(m), Vtsv = zeros(m, m),
        # ── AbstractLAPACKSolve operands. SMALL on purpose (ms=32, nrhs=2, band widths 1-2): this
        # block runs at every user's precompile, so each of the 24 members gets the smallest shape
        # that still reaches its real kernel rather than a degenerate one. The checks never run
        # these routines, so the operands carry types and dimensions, not values.
        ms = 32, nrhs = 2,
        Asw = zeros(ms, ms), Achol = zeros(ms, ms), Alu = zeros(ms, ms), AFx = zeros(ms, ms),
        ipivs = Vector{Int}(undef, ms), ipivx = Vector{Int}(undef, ms), pivs = Vector{Int}(undef, ms),
        Bs0 = ones(ms, nrhs), Bsw = zeros(ms, nrhs), Xs = zeros(ms, nrhs),
        ferrs = zeros(nrhs), berrs = zeros(nrhs), Rq = ones(ms), Cq = ones(ms),
        # Packed lower storage: column j is rows j:ms, so the length is ms(ms+1)/2. APc stands for
        # the factor pptrs! consumes.
        APc = zeros(ms * (ms + 1) ÷ 2),
        # pptrf!'s own operand is order 32, deliberately ABOVE `_PPTRF_BLK_MIN` (16) so the strict
        # window covers `_pptrf_lower_blocked!` — see the note at its call below.
        nps = 32,
        APuw = zeros(nps * (nps + 1) ÷ 2),
        # Band SPD (kdS sub-diagonals; row 1 of the lower band store is the diagonal).
        kdS = 2,
        ABpw = zeros(kdS + 1, ms), ABpc = zeros(kdS + 1, ms),
        # General band LU: ldab = 2kl+ku+1 with the diagonal on row kl+ku+1 and rows 1:kl reserved
        # for the factorization's fill, exactly dgbtrf's convention.
        klS = 1, kuS = 1,
        ABgw = zeros(2 * klS + kuS + 1, ms), ABgc = zeros(2 * klS + kuS + 1, ms),
        ipivg = Vector{Int}(undef, ms), ipivgw = Vector{Int}(undef, ms),
        # A SECOND, WIDE band for gbtrf! only. kl=ku=1 above reaches `_gbtf2!` and nothing else:
        # `gbtrf!` takes `_gbtrf_blocked!` (and the `_gbtrf_work` W13/W31/S scratch behind it) only
        # when `kl >= max(2·nb, _gbtrf_cross(T))`, and `_gbtrf_cross(Float64)` is 32 on a wide-SIMD
        # box and 64 otherwise (cpuinfo.jl:507). kl=64 clears the wider of the two, so this arm is
        # exercised on EVERY µarch rather than only where the crossover happens to be low. (A user
        # who pins `gbtrf_cross` above 64 falls back to the unblocked kernel here — still 0-alloc,
        # just less coverage.) gbtrs! is unaffected by blocking and stays on the narrow band above.
        klB = 64, kuB = 8, msB = 128,
        ABbw = zeros(2 * klB + kuB + 1, msB), ipivb = Vector{Int}(undef, msB),
        # Tridiagonal working triple the factor-in-place routines overwrite, and a separately
        # factored triple gttrs! solves against.
        dlw = zeros(ms - 1), dw = zeros(ms), duw = zeros(ms - 1), du2w = zeros(ms - 2),
        dlf = zeros(ms - 1), df = zeros(ms), duf = zeros(ms - 1), du2f = zeros(ms - 2),
        ipivt = Vector{Int}(undef, ms), ipivtw = Vector{Int}(undef, ms),
        # SPD tridiagonal (pttrf/pttrs/ptsv): D real, E the sub-diagonal.
        Dpw = zeros(ms), Epw = zeros(ms - 1), Dpf = zeros(ms), Epf = zeros(ms - 1)
    # No PureBLAS routine is called below this line: the checks read argument types only, so a
    # factored source has the type of the matrix it would be factored from, and calling a routine
    # here would only compile its call tree into this package's precompile.
    # QR-family sources, one per reflector layout (QR, LQ, QL, RQ, RZ).
    Afqr = copy(Aqr)
    Aflq = copy(Aqr); taul = Vector{Float64}(undef, m)
    Afql = copy(Aqr); tauql = Vector{Float64}(undef, m)
    Afrq = copy(Aqr); taurq = Vector{Float64}(undef, m)
    Afrz = copy(Aqr); taurz = Vector{Float64}(undef, m)
    Aqtr = ones(m, m); tautr = Vector{Float64}(undef, m); Qtr = zeros(m, m)
    Aztr = ones(ComplexF64, m, m); tauztr = Vector{ComplexF64}(undef, m); Qztr = zeros(ComplexF64, m, m)
    # ── Operands for the 34 members added with the nonsymmetric-eigen / Bunch-Kaufman / QZ /
    # SVD-front / least-squares owned-workspace conversion. Everything is order `ms` (=32) or
    # smaller: this block runs at every user's precompile, so each member gets the smallest shape
    # that still reaches its real kernel. `ev*`/`ez*` are the real/complex nonsymmetric-eigen sets,
    # `qz*`/`qzz*` the generalized pencils, `bkf*`/`bkz*` Bunch-Kaufman, `sv*`/`svz*` the SVD front
    # half and generalized SVD, `ls*`/`lsz*` least squares, `st*` the tridiagonal eigen pair.
    # Each `ev/ez` set carries four ms×ms scratch matrices because several members need two or three
    # distinct mutable targets live at once (trexc!'s T and Q, gghrd!'s A/B/Q/Z).
    evA = ones(ms, ms)
    evW1 = zeros(ms, ms); evW2 = zeros(ms, ms); evW3 = zeros(ms, ms); evW4 = zeros(ms, ms)
    evsc = ones(ms); evtau = Vector{Float64}(undef, ms)
    evilo, evihi = 1, ms              # the (ilo, ihi) range gebal! returns: two Ints
    evV = ones(ms, ms)
    evAf = copy(evA)
    evw = Vector{ComplexF64}(undef, ms); evwr = zeros(ms); evwi = zeros(ms)
    evT = ones(ms, ms)
    # trsen!: `select` marks the leading cluster; `w` receives the eigenvalues (must not alias T).
    tsnsel = zeros(Int, ms); tsnsel[1:(ms ÷ 2)] .= 1
    tsnw = Vector{ComplexF64}(undef, ms)
    # sygvd!/hegvd!: A symmetric/Hermitian, B symmetric/Hermitian POSITIVE DEFINITE (potrf! must
    # succeed). Diagonally dominant; the complex diagonals are real, as a Hermitian's must be.
    gvAw = zeros(ms, ms); gvBw = zeros(ms, ms); gvw = zeros(ms)
    gvzAw = zeros(ComplexF64, ms, ms); gvzBw = zeros(ComplexF64, ms, ms)
    evVL = zeros(ms, 0); evU = ones(ms, ms)
    ezA = ones(ComplexF64, ms, ms)
    ezW1 = zeros(ComplexF64, ms, ms); ezW2 = zeros(ComplexF64, ms, ms)
    ezW3 = zeros(ComplexF64, ms, ms); ezW4 = zeros(ComplexF64, ms, ms)
    ezsc = ones(ms); eztau = Vector{ComplexF64}(undef, ms)
    ezilo, ezihi = 1, ms
    ezV = ones(ComplexF64, ms, ms)
    ezAf = copy(ezA)
    ezw = Vector{ComplexF64}(undef, ms)
    ezVL = zeros(ComplexF64, ms, 0)
    ezU = ones(ComplexF64, ms, ms)
    # QZ: eigenvalue / selection operands for hgeqz!, tgevc!, tgsen!.
    qzal = Vector{ComplexF64}(undef, ms); qzbe = zeros(ms)
    qzsel = zeros(Int, ms); qzsel[1] = 1; qzsel[2] = 1; qzsel[5] = 1
    qzzal = Vector{ComplexF64}(undef, ms); qzzbe = Vector{ComplexF64}(undef, ms)
    # Bunch-Kaufman: one source per (sy/he) × (L/U), each with its own pivots, because `_sytri_lower!`
    # / `_sytri_upper!` and the symmetric / Hermitian arms are four separate kernels.
    bkfA = ones(ms, ms)
    bkzA = ones(ComplexF64, ms, ms)
    bkfip = Vector{Int}(undef, ms); bkfiph = Vector{Int}(undef, ms)
    bkzip = Vector{Int}(undef, ms); bkziph = Vector{Int}(undef, ms)
    bkfwk = Vector{Float64}(undef, ms); bkzwk = Vector{ComplexF64}(undef, ms)
    bkfb0 = ones(ms); bkfbw = zeros(ms); bkzb0 = ones(ComplexF64, ms); bkzbw = zeros(ComplexF64, ms)
    bkzB0 = ones(ComplexF64, ms, nrhs); bkzBw = zeros(ComplexF64, ms, nrhs)
    bkfLs = copy(bkfA); bkfLsp = Vector{Int}(undef, ms)
    bkfUs = copy(bkfA); bkfUsp = Vector{Int}(undef, ms)
    bkfLh = copy(bkfA); bkfLhp = Vector{Int}(undef, ms)
    bkzLs = copy(bkzA); bkzLsp = Vector{Int}(undef, ms)
    bkzUs = copy(bkzA); bkzUsp = Vector{Int}(undef, ms)
    bkzLh = copy(bkzA); bkzLhp = Vector{Int}(undef, ms)
    bkzUh = copy(bkzA); bkzUhp = Vector{Int}(undef, ms)
    # SVD front half + generalized SVD. `p < n` for ggsvd! so the RQ branches (`_ggs_gerq2!` /
    # `_ggs_apply_rq_right!`, the two O(n²) sites the conversion actually targeted) are on the path.
    svm, svn, svp = 24, 16, 8
    svAw = zeros(svm, svn); svd = zeros(svn); sve = zeros(svn)
    svtq = Vector{Float64}(undef, svn); svtp = Vector{Float64}(undef, svn)
    svzAw = zeros(ComplexF64, svm, svn)
    svztq = Vector{ComplexF64}(undef, svn); svztp = Vector{ComplexF64}(undef, svn)
    ggAw = zeros(svm, svn); ggBw = zeros(svp, svn)
    ggU = zeros(svm, svm); ggV = zeros(svp, svp); ggQ = zeros(svn, svn)
    ggal = zeros(svn); ggbe = zeros(svn); ggR = zeros(svn, svn)
    ggzAw = zeros(ComplexF64, svm, svn); ggzBw = zeros(ComplexF64, svp, svn)
    ggzU = zeros(ComplexF64, svm, svm); ggzV = zeros(ComplexF64, svp, svp)
    ggzQ = zeros(ComplexF64, svn, svn); ggzR = zeros(ComplexF64, svn, svn)
    # Bidiagonal SVD driver: d/e are the bidiagonal itself, so no factored source is needed.
    bqd = zeros(svn); bqe = zeros(svn)
    bqU = ones(svn, svn); bqV = ones(svn, svn)
    bqVt = ones(ComplexF64, svn, svn); bqUz = ones(ComplexF64, svn, svn)
    bqC = zeros(ComplexF64, svn, 0)
    # bdsdc! writes both singular-vector sets into caller buffers (n×n each).
    bdLv = zeros(svn, svn); bdRv = zeros(svn, svn)
    # Least squares. B is max(m,n)×nrhs (LAPACK ldb): rows 1:m hold b on entry, 1:n hold X on exit.
    lsB0 = ones(max(svm, svn), nrhs); lsBw = zeros(max(svm, svn), nrhs); lsjp = zeros(Int, svn)
    lszB0 = ones(ComplexF64, max(svm, svn), nrhs); lszBw = zeros(ComplexF64, max(svm, svn), nrhs)
    lsc0 = ones(svm); lscw = zeros(svm); lsd0 = ones(svp); lsdw = zeros(svp); lsx = zeros(svn)
    lszc0 = ones(ComplexF64, svm); lszcw = zeros(ComplexF64, svm)
    lszd0 = ones(ComplexF64, svp); lszdw = zeros(ComplexF64, svp); lszx = zeros(ComplexF64, svn)
    # Symmetric-tridiagonal eigen. stein! takes stebz!'s w/iblock/isplit trimmed to the m eigenvalues
    # and nsplit blocks found, so those three arrive as views.
    std = [2.0 + i / ms for i in 1:ms]; ste = fill(0.4, ms - 1)
    stw = zeros(ms); stib = zeros(Int, ms); stisp = zeros(Int, ms)
    stm, stns = ms, 1
    stwv = view(stw, 1:stm); stibv = view(stib, 1:stm); stispv = view(stisp, 1:stns)
    stZ = zeros(ms, stm)
    # SubArray operands over LARGER parents (lda != nrows), for the same reason the L3 block has
    # them: the owned-workspace accessors hand exactly this shape to exactly these routines, and a
    # union-typed slot that costs nothing on a `Matrix` boxes on a `SubArray`.
    vP1 = ones(ms + 8, ms + 8); vP2 = ones(ms + 8, ms + 8)
    vP3 = ones(ms + 8, ms + 8); vP4 = ones(ms + 8, ms + 8)
    vA = view(vP1, 1:ms, 1:ms); vB = view(vP2, 1:ms, 1:ms)
    vC = view(vP3, 1:ms, 1:ms); vD = view(vP4, 1:ms, 1:ms)
    vsc = view(ones(ms + 8), 1:ms); vtau = view(Vector{Float64}(undef, ms + 8), 1:ms)
    vwr = view(ones(ms + 8), 1:ms); vwi = view(ones(ms + 8), 1:ms)
    vwc = view(Vector{ComplexF64}(undef, ms + 8), 1:ms)
    vipP = Vector{Int}(undef, ms + 8); vip = view(vipP, 1:ms)
    vwk = view(ones(ms + 8), 1:ms)
    vAf = copy(evA); vtauP = Vector{Float64}(undef, ms + 8)
    vtauf = view(vtauP, 1:ms)
    vbkf = copy(bkfLs); copyto!(vip, bkfLsp)
    vBP = ones(ms + 8, nrhs + 4); vBv = view(vBP, 1:ms, 1:nrhs)
    vlsP = ones(svm + 8, svn + 8); vlsA = view(vlsP, 1:svm, 1:svn)
    vlsBP = ones(svm + 8, nrhs + 4); vlsB = view(vlsBP, 1:svm, 1:nrhs)
    vlsjp = view(vipP, 1:svn)
    vstP = ones(ms + 8, stm + 4); vstZ = view(vstP, 1:ms, 1:stm)
    # The guarantees are named so every listed signature is also registered, and
    # test_registered (StrictModeTest) re-proves them in the test environment.
    @verify_strict SIMDBackend guarantees = (:typestable, :noalloc, :owned) begin
        # ── Level 1 (bandwidth-bound; SIMD real path + generic complex path)
        axpy!(bk, yd, 2.0, xd)
        scal!(bk, 2.0, xd)
        blascopy!(bk, yd, xd)
        swap!(bk, xd, yd)
        dot(bk, xd, yd)
        dotu(bk, xd, yd)
        nrm2(bk, xd)
        asum(bk, xd)
        iamax(bk, xd)
        axpy!(bk, yz, 2.0 + 1.0im, xz)
        scal!(bk, 2.0 + 1.0im, xz)         # complex scal: interleaved-SIMD (swap-pairs) path
        dot(bk, xz, yz)                     # complex dot/dotu: split-deinterleave SIMD reduction
        dotu(bk, xz, yz)
        nrm2(bk, xz)                       # complex nrm2/asum now take the SIMD real-reinterpret path
        asum(bk, xz)
        blascopy!(bk, yz, xz)
        swap!(bk, xz, yz)
        iamax(bk, xz)                      # izamax: swap-adjacent magnitude reduction (its own kernel)
        # ── Level 2 (dense hot paths; real + complex)
        gemv!(bk, vm, Ad, um; alpha = 2.0, beta = 1.0, trans = 'N')
        gemv!(bk, vm, Ad, um; alpha = 2.0, beta = 1.0, trans = 'T')
        gemv!(bk, wz, Az, uz; alpha = 2.0 + 0im, beta = 1.0 + 0im, trans = 'N')   # complex gemv SIMD
        gemv!(bk, wz, Az, uz; alpha = 2.0 + 0im, beta = 1.0 + 0im, trans = 'C')
        ger!(bk, 1.5, um, vm, Ad)
        ger!(bk, 1.5 + 0.5im, uz, wz, Az)     # complex geru/gerc: per-column complex axpy
        symv!(bk, vm, Ad, um)
        hemv!(bk, wz, Az, uz)
        trmv!(bk, Ad, um)
        trsv!(bk, Ad, um)
        trmv!(bk, Az, uz)                     # complex trmv/trsv: per-column axpy(N)/dot(T/C) reuse
        trsv!(bk, Az, uz)
        # ── Level 2 packed storage (symmetric/Hermitian/triangular; rank-1/2 updates)
        spmv!(bk, vm, APd, um; uplo = 'U', alpha = 2.0, beta = 1.0)
        hpmv!(bk, wz, APz, uz; uplo = 'U', alpha = 2.0 + 0im, beta = 1.0 + 0im)
        tpmv!(bk, APd, um; uplo = 'U', trans = 'N', diag = 'N')
        tpsv!(bk, APd, um; uplo = 'U', trans = 'N', diag = 'N')
        spr!(bk, 1.5, um, APd; uplo = 'U')
        spr2!(bk, 1.5, um, vm, APd; uplo = 'U')
        hpr!(bk, 1.5, uz, APz; uplo = 'U')
        hpr2!(bk, 1.5 + 0im, uz, wz, APz; uplo = 'U')
        # ── Level 2 band storage (general/symmetric/Hermitian/triangular banded)
        gbmv!(bk, vm, ABg, um, m, kb, kb; trans = 'N', alpha = 2.0, beta = 1.0)
        sbmv!(bk, vm, ABs, um; uplo = 'U', alpha = 2.0, beta = 1.0)
        hbmv!(bk, wz, ABz, uz; uplo = 'U', alpha = 2.0 + 0im, beta = 1.0 + 0im)
        tbmv!(bk, ABs, um; uplo = 'U', trans = 'N', diag = 'N')
        tbsv!(bk, ABs, um; uplo = 'U', trans = 'N', diag = 'N')
        # ── Level 3 (all matrix-matrix ops; real + complex)
        gemm!(bk, C3, A3, B3)
        # COMPLEX gemm, and deliberately at m=64 — INSIDE the Karatsuba-3M window
        # (`_CGEMM_3M_MIN`=48 ≤ max(m,n,k) ≤ 2048, min ≥ `_CGEMM_3M_KMIN`=16). Until now every
        # `gemm!` here was REAL, and 3M is complex-only, so the strict contract verified the one arm
        # that cannot reach `_gemm_3m!`. That is how nine `unsafe_wrap(Array, …)` per call — ~1 KB of
        # steady-state allocation — passed this check on AVX2 from the day 3M shipped. The other
        # complex L3 calls below cannot substitute: they run at m=64, under `_CSYRK_3M_MIN`=256, so
        # `herk!`/`her2k!` never enter rank-k 3M. One line, and it closes the whole class.
        gemm!(bk, Cz3, Az3, Bz3)
        symm!(bk, C3, A3, B3)
        hemm!(bk, Cz3, Az3, Bz3; side = 'L', uplo = 'L', alpha = 1.0 + 0im, beta = 1.0 + 0im)
        syrk!(bk, C3, A3; uplo = 'L', trans = 'N', alpha = 1.0, beta = 1.0)
        herk!(bk, Cz3, Az3; uplo = 'L', trans = 'N', alpha = 1.0, beta = 1.0)
        syr2k!(bk, C3, A3, B3; uplo = 'L', trans = 'N', alpha = 1.0, beta = 1.0)
        her2k!(bk, Cz3, Az3, Bz3; uplo = 'L', trans = 'N', alpha = 1.0 + 0im, beta = 1.0)
        trmm!(bk, Bt, At; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0)
        trsm!(bk, Bt, At; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0)
        # COMPLEX arms of the five L3 ops that were verified on the real path only. `hemm!`/`herk!`/
        # `her2k!` above do NOT substitute: the symmetric (non-Hermitian) complex kernels are
        # separate specializations — `zsymm`/`zsyrk`/`zsyr2k` conjugate nothing — and `ztrmm`/`ztrsm`
        # share no code with any of them.
        symm!(bk, Cz3, Az3, Bz3)
        syrk!(bk, Cz3, Az3; uplo = 'L', trans = 'N', alpha = 1.0 + 0im, beta = 1.0 + 0im)
        syr2k!(bk, Cz3, Az3, Bz3; uplo = 'L', trans = 'N', alpha = 1.0 + 0im, beta = 1.0 + 0im)
        trmm!(bk, Btz, Atz; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0 + 0im)
        trsm!(bk, Btz, Atz; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0 + 0im)
        # ── SubArray arguments. Everything above passes `Matrix`, and that gap let a REAL contract
        # violation stand: `_trsm_right!` assigned `Ar` three different types (the caller's `A`, and
        # two workspace views), so the slot was union-typed. With `Matrix` the union split and cost
        # nothing; with a `SubArray` it BOXED — 64 B per call, allocation-free contract broken, and
        # every one of these Matrix-only assertions passed anyway. Views are not an exotic input
        # here: the owned-workspace panels (`_pptrf_lower_work`, `_getri_work`, …) hand views to
        # exactly these ops, so this is the shape the package uses internally.
        # side='R' with transA='T' is the arm that boxed; keep it explicitly.
        trsm!(bk, Btv, Atv; side = 'R', uplo = 'L', transA = 'T', diag = 'N', alpha = 1.0)
        trsm!(bk, Btv, Atv; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0)
        trmm!(bk, Btv, Atv; side = 'R', uplo = 'L', transA = 'T', diag = 'N', alpha = 1.0)
        gemm!(bk, C3v, A3v, B3v)
        syrk!(bk, C3v, A3v; uplo = 'L', trans = 'N', alpha = 1.0, beta = 1.0)
        # ── LAPACK: potrf!/getrf!/geqrf! are 0-alloc (potrf via its own pointer kernels; LU/QR via
        # their in-place pre-allocated-output forms). gesvd! reaches 0-alloc via the in-place
        # gesvd!(A,U,S,Vᵀ) form + a cached SVDWorkspace for the bidiagonalization scratch.
        potrf!(bk, Aw; uplo = 'L')
        getrf!(Aw, ipiv)
        geqrf!(Aw, tau)
        gesvd!(Aw, Usv, Ssv, Vtsv)
        # COMPLEX factorizations — a wholly separate kernel set (`_cpotrf_lower!`, `_cgetf2_simd!`,
        # the complex QR panel).
        # gesvd is deliberately NOT included for complex: its in-place form caches a Float64
        # SVDWorkspace and the complex path has not been shown 0-alloc — asserting it here without
        # that evidence would be guessing, so it stays an open item rather than a silent pass.
        potrf!(bk, Awz3; uplo = 'L')
        getrf!(Awz3, ipiv)
        geqrf!(Awz3, tauz)
        # ── AbstractLAPACKSolve (contracts.jl): the solve/inverse/condition surface built on those
        # four factorizations. Declaring the contract only fixes the method signatures; it is THIS
        # block that makes "type-stable AND allocation-free" true, and eight of these were breaking
        # it before the owned-workspace conversion. Grouped by the factorization they hang off.
        #
        # Cholesky family: solve, explicit inverse (potri! takes the FACTOR, not the SPD matrix,
        # and overwrites it with A⁻¹), packed and banded factor+solve, the rank-revealing pivoted
        # factor, and the condition estimate.
        potrs!(bk, Achol, Bsw; uplo = 'L')
        potri!(bk, Asw; uplo = 'L')
        # pptrf! is verified at order 32, i.e. ABOVE `_PPTRF_BLK_MIN` (16), so the arm under test is
        # `_pptrf_lower_blocked!`, the arm that holds W/V in `_pptrf_lower_work`'s owned scratch.
        # Measured 0 B at n ∈ {8, 16, 32, 64}, blocked and unblocked alike.
        pptrf!(bk, APuw; uplo = 'L')
        pptrs!(bk, APc, Bsw; uplo = 'L')
        pbtrf!(bk, ABpw; uplo = 'L', kd = kdS)
        pbtrs!(bk, ABpc, Bsw; uplo = 'L')
        pstrf!(bk, Asw, pivs, -1.0; uplo = 'L')
        pocon!(bk, float(2 * ms), Achol; uplo = 'L')
        # LU family: solve against the factors and the explicit inverse. getri! runs LAPACK's
        # block-column algorithm over an owned n×nb buffer, which is exactly what this line holds it to.
        getrs!(bk, Alu, ipivs, Bsw; trans = 'N')
        getri!(bk, Asw, ipivs)
        # Triangular: solve, in-place inverse, condition estimate, and the trrfs! error bounds.
        trtrs!(bk, Achol, Bsw; uplo = 'L', trans = 'N', diag = 'N')
        trtri!(bk, Asw; uplo = 'L', diag = 'N')
        gecon!(bk, float(2 * ms), Alu, ipivs; norm = '1')
        trcon!(bk, Achol; uplo = 'L', diag = 'N', norm = '1')
        trrfs!(bk, 'L', 'N', 'N', Achol, Bs0, Xs, ferrs, berrs)
        # Banded LU: the IN-PLACE gbtrf!(kl, ku, m, AB, ipiv) form, not the convenience form that
        # allocates ipiv — the contract binds the former precisely so it cannot be satisfied by a
        # method that allocates its own output. TWO shapes, for the same reason pptrf! moved above
        # its blocking threshold: kl=ku=1 reaches `_gbtf2!` and nothing else, so the blocked kernel
        # and the `_gbtrf_work` W13/W31/S scratch behind it went unverified. kl=64 clears
        # `_gbtrf_cross(Float64)` on every µarch (32 wide-SIMD / 64 otherwise). Both measure 0 B.
        gbtrf!(bk, klS, kuS, ms, ABgw, ipivgw)
        gbtrf!(bk, klB, kuB, msB, ABbw, ipivb)
        gbtrs!(bk, 'N', klS, kuS, ms, ABgc, ipivg, Bsw)
        # Tridiagonal: the fused driver, the split factor/solve pair, and their SPD counterparts.
        # These are pure recurrences with no BLAS underneath, so an allocation here would be a
        # bounds-check or boxing artefact rather than a workspace — worth pinning for that reason.
        gtsv!(bk, dlw, dw, duw, Bsw)
        gttrf!(bk, dlw, dw, duw, du2w, ipivtw)
        gttrs!(bk, 'N', dlf, df, duf, du2f, ipivt, Bsw)
        pttrf!(bk, Dpw, Epw)
        pttrs!(bk, Dpf, Epf, Bsw; uplo = 'L')
        ptsv!(bk, Dpw, Epw, Bsw; uplo = 'L')
        # Expert driver — the deepest composition in the contract (equilibrate → getrf → solve →
        # gecon ×2 → iterative refinement), so it is also the one most likely to leak an allocation
        # from a callee. fact='N'/equed='N' skips equilibration, keeping the arm under test the
        # factor-solve-refine chain rather than the scaling. Again the IN-PLACE form: X/ferr/berr
        # come from the caller, which is what netlib's OUT arguments always implied.
        gesvx!(bk, 'N', 'N', Asw, AFx, ipivx, 'N', Rq, Cq, Bsw, Xs, ferrs, berrs)
        # ── QR / LQ / QL / RZ. 14 distinct implementations; the nine `un*!` names are `const`
        # aliases of these same function objects, so verifying `or*!` verifies them too.
        gelqf!(Aqw, tauq)
        geqlf!(Aqw, tauq)
        gerqf!(Aqw, tauq)
        tzrzf!(Aqw, tauq)
        geqp3!(Aqw, jpq, tauq)
        orgqr!(Aqw, tauq)
        orglq!(Aqw, taul)
        orgql!(Aqw, tauql)
        orgrq!(Aqw, taurq)
        ormqr!('L', 'N', Afqr, tauq, Cqw)
        ormlq!('L', 'N', Aflq, taul, Cqw)
        ormql!('L', 'N', Afql, tauql, Cqw)
        ormrq!('L', 'N', Afrq, taurq, Cqw)
        ormrz!('L', 'N', Afrz, taurz, Cqw)
        orgtr!(bk, 'L', Aqtr, tautr, Qtr)
        ungtr!(bk, 'L', Aztr, tauztr, Qztr)
        # ── Nonsymmetric eigen: balance, Hessenberg reduction, Schur, eigenvectors, drivers.
        # REAL and COMPLEX arms both, because they are separate kernel sets throughout (dlahqr vs
        # zlahqr, the real quasi-triangular 2×2 machinery vs plain triangular back-substitution), and
        # a real-only assertion is exactly how ~1 KB/call in the complex 3M gemm passed for a month.
        # `trsen!` is deliberately ABSENT: measured 16 B/call at job∈{'V','B'} on BOTH types — the
        # `Base.RefValue` carrying `_dtrsyl!`'s scale out of the `_lacn2_estimate` callback. It goes
        # in when that Ref does; asserting it now would fail the build, and dropping the failing jobs
        # would make the member's guarantee a lie.
        gebal!(bk, evW1, evsc; job = 'B')
        gebak!(bk, 'B', 'R', evilo, evihi, evsc, evV)
        gehrd!(bk, evW1, evilo, evihi, evtau)
        orghr!(bk, evW1, evilo, evihi, evtau)
        # Both sides AND both trans arms: side='R' is a different gemm! shape, and it is the arm the
        # L3 block's own SubArray note records as the one that boxed when only side='L' was checked.
        ormhr!(bk, 'L', 'N', evilo, evihi, evAf, evtau, evW2)
        ormhr!(bk, 'R', 'T', evilo, evihi, evAf, evtau, evW2)
        # job='S'/compz='I' is the full Schur arm (T and Z formed); job='E'/compz='N' the
        # eigenvalues-only arm, which skips the Z accumulation entirely.
        hseqr!(bk, 'S', 'I', evW1, evilo, evihi, evw, evW3)
        hseqr!(bk, 'E', 'N', evW1, evilo, evihi, evw, evW3)
        trevc!(bk, 'R', 'B', evW1, evVL, evW2)
        trexc!(bk, 'V', evW1, evW2, 5, 1)
        # trsen! at all four jobs: 'N'/'E' skip the sep estimate entirely, while 'V'/'B' run the
        # `_lacn2_estimate` reverse-communication loop — the arm that carried the 16 B Ref.
        trsen!(bk, 'N', 'V', tsnsel, evW1, evW2, tsnw)
        trsen!(bk, 'E', 'V', tsnsel, evW1, evW2, tsnw)
        trsen!(bk, 'V', 'V', tsnsel, evW1, evW2, tsnw)
        trsen!(bk, 'B', 'V', tsnsel, evW1, evW2, tsnw)
        # sygvd!/hegvd!: itype 1 vs 2/3 are different REDUCTIONS (trsm vs trmm) and itype 3 a
        # different BACK-TRANSFORM (trmm vs trsm); jobz 'V' vs 'N' are different eigensolvers
        # (_stedc! divide-and-conquer vs _sterf!). Both uplos: 'U' takes the mirror-to-lower path.
        sygvd!(bk, 1, 'V', 'L', gvAw, gvBw, gvw)
        sygvd!(bk, 2, 'V', 'U', gvAw, gvBw, gvw)
        sygvd!(bk, 3, 'V', 'L', gvAw, gvBw, gvw)
        sygvd!(bk, 1, 'N', 'U', gvAw, gvBw, gvw)
        hegvd!(bk, 1, 'V', 'L', gvzAw, gvzBw, gvw)
        hegvd!(bk, 3, 'V', 'U', gvzAw, gvzBw, gvw)
        hegvd!(bk, 1, 'N', 'L', gvzAw, gvzBw, gvw)
        # transa='N' and 'T' are separate back-substitution orders; the third call feeds trsyl! a
        # quasi-triangular A so the `_syl_dlasy2` 2×2 arm (and its `t16` scratch) is on the path.
        trsyl!(bk, 'N', 'N', 1, evU, evU, evW2)
        trsyl!(bk, 'T', 'N', 1, evU, evU, evW2)
        trsyl!(bk, 'N', 'N', 1, evT, evU, evW2)
        geev!(bk, 'N', 'V', evW1, evwr, evwi, evVL, evW2, evsc)
        gees!(bk, 'V', evW1, evw, evW2, evsc)
        gebal!(bk, ezW1, ezsc; job = 'B')
        gebak!(bk, 'B', 'R', ezilo, ezihi, ezsc, ezV)
        gehrd!(bk, ezW1, ezilo, ezihi, eztau)
        orghr!(bk, ezW1, ezilo, ezihi, eztau)
        ormhr!(bk, 'L', 'N', ezilo, ezihi, ezAf, eztau, ezW2)
        ormhr!(bk, 'R', 'C', ezilo, ezihi, ezAf, eztau, ezW2)
        hseqr!(bk, 'S', 'I', ezW1, ezilo, ezihi, ezw, ezW3)
        hseqr!(bk, 'E', 'N', ezW1, ezilo, ezihi, ezw, ezW3)
        trevc!(bk, 'R', 'B', ezW1, ezVL, ezW2)
        trexc!(bk, 'V', ezW1, ezW2, 5, 1)
        trsen!(bk, 'N', 'V', tsnsel, ezW1, ezW2, tsnw)
        trsen!(bk, 'B', 'V', tsnsel, ezW1, ezW2, tsnw)
        trsyl!(bk, 'N', 'N', 1, ezU, ezU, ezW2)
        trsyl!(bk, 'C', 'N', 1, ezU, ezU, ezW2)
        geev!(bk, 'N', 'V', ezW1, ezw, ezVL, ezW2, ezsc)
        gees!(bk, 'V', ezW1, ezw, ezW2, ezsc)
        # ── Bunch-Kaufman. uplo='L' and 'U' are separate kernels throughout (`_sytri_lower!` /
        # `_sytri_upper!`, `_sytrs_lower!` / `_sytrs_upper!`), so both are asserted for COMPLEX,
        # where the Hermitian arm is genuinely distinct. For REAL eltype `hetrf!`/`hetrs!`/`hetri!`/
        # `hesv!` ARE `sytrf!`/… (herm=false), so their real arms are pinned once, at uplo='L', to
        # hold the dispatch rather than to re-verify the same kernel a second time.
        sytrf!(bk, evW1, bkfip; uplo = 'L')
        sytrf!(bk, evW1, bkfip; uplo = 'U')
        hetrf!(bk, evW1, bkfiph; uplo = 'L')
        sytrs!(bk, bkfLs, bkfLsp, Bsw; uplo = 'L')
        sytrs!(bk, bkfUs, bkfUsp, Bsw; uplo = 'U')
        # VECTOR right-hand side: `sytrs!` must not `reshape(B, n, 1)` it — that is 48 B/call of
        # Array header, and a `Base.ReshapedArray` (the trim-hostile `_throw_dmrs` path) when the RHS
        # is itself a view.
        sytrs!(bk, bkfLs, bkfLsp, bkfbw; uplo = 'L')
        hetrs!(bk, bkfLh, bkfLhp, Bsw; uplo = 'L')
        sytri!(bk, evW1, bkfLsp; uplo = 'L')
        sytri!(bk, evW1, bkfUsp; uplo = 'U')
        hetri!(bk, evW1, bkfLhp; uplo = 'L')
        sysv!(bk, 'L', evW1, Bsw, bkfip)
        sysv!(bk, 'U', evW1, Bsw, bkfip)
        hesv!(bk, 'L', evW1, Bsw, bkfiph)
        syconv!(bk, 'L', evW1, bkfLsp, bkfwk)
        syconv!(bk, 'U', evW1, bkfUsp, bkfwk)
        sytrf!(bk, ezW1, bkzip; uplo = 'L')
        sytrf!(bk, ezW1, bkzip; uplo = 'U')
        hetrf!(bk, ezW1, bkziph; uplo = 'L')
        hetrf!(bk, ezW1, bkziph; uplo = 'U')
        sytrs!(bk, bkzLs, bkzLsp, bkzBw; uplo = 'L')
        sytrs!(bk, bkzUs, bkzUsp, bkzBw; uplo = 'U')
        sytrs!(bk, bkzLs, bkzLsp, bkzbw; uplo = 'L')
        hetrs!(bk, bkzLh, bkzLhp, bkzBw; uplo = 'L')
        hetrs!(bk, bkzUh, bkzUhp, bkzBw; uplo = 'U')
        sytri!(bk, ezW1, bkzLsp; uplo = 'L')
        sytri!(bk, ezW1, bkzUsp; uplo = 'U')
        hetri!(bk, ezW1, bkzLhp; uplo = 'L')
        hetri!(bk, ezW1, bkzUhp; uplo = 'U')
        sysv!(bk, 'L', ezW1, bkzBw, bkzip)
        hesv!(bk, 'L', ezW1, bkzBw, bkziph)
        hesv!(bk, 'U', ezW1, bkzBw, bkziph)
        syconv!(bk, 'L', ezW1, bkzLsp, bkzwk)
        syconv!(bk, 'U', ezW1, bkzUsp, bkzwk)
        # ── QZ / generalized eigen. hgeqz!/tgevc!/tgsen! take a GENUINE Hessenberg-triangular or
        # generalized-Schur pair (built above): handing the QZ iteration a raw pencil would run a
        # different, shorter path and "succeed" on nonsense. `ggev!` appears at both arities — the
        # contract can only declare the real one (alphar, alphai, beta), so the complex form is held
        # to the guarantee here.
        gghrd!(bk, 'I', 'I', evW1, evW2, evW3, evW4)
        hgeqz!(bk, 'S', 'I', 'I', evW1, evW2, qzal, qzbe, evW3, evW4)
        tgevc!(bk, 'R', 'B', evW1, evW2, evVL, evW3)
        tgsen!(bk, qzsel, evW1, evW2, evW3, evW4, qzal, qzbe)
        ggev!(bk, 'N', 'V', evW1, evW2, evwr, evwi, qzbe, evVL, evW3)
        gges!(bk, 'V', 'V', evW1, evW2, qzal, qzbe, evW3, evW4)
        gghrd!(bk, 'I', 'I', ezW1, ezW2, ezW3, ezW4)
        hgeqz!(bk, 'S', 'I', 'I', ezW1, ezW2, qzzal, qzzbe, ezW3, ezW4)
        tgevc!(bk, 'R', 'B', ezW1, ezW2, ezVL, ezW3)
        tgsen!(bk, qzsel, ezW1, ezW2, ezW3, ezW4, qzzal, qzzbe)
        ggev!(bk, 'N', 'V', ezW1, ezW2, qzzal, qzzbe, ezVL, ezW3)
        gges!(bk, 'V', 'V', ezW1, ezW2, ezw, qzzbe, ezW3, ezW4)
        # ── SVD front half + generalized SVD. `gebrd!`/`bdsdc!` are now asserted at their
        # workspace-free arity (the `ws`-taking forms stay internal); see contracts.jl. `bdsqr!` is
        # asserted at all three shapes it actually gets: real with U/V, real with `nothing`/`nothing`
        # (the values-only path, a Union argument and so the one that could union-split), and the
        # complex 6-argument kernel, which shares no code with either.
        gebd2!(bk, svAw, svd, sve, svtq, svtp)
        gebd2!(bk, svzAw, svd, sve, svztq, svztp)
        gebrd!(bk, svAw, svd, sve, svtq, svtp)
        gebrd!(bk, svzAw, svd, sve, svztq, svztp)
        bdsdc!(bk, bqd, bqe, bdLv, bdRv)
        bdsqr!(bk, bqd, bqe, bqU, bqV)
        bdsqr!(bk, bqd, bqe, nothing, nothing)
        bdsqr!(bk, 'U', bqd, bqe, bqVt, bqUz, bqC)
        ggsvd!(bk, 'U', 'V', 'Q', ggAw, ggBw, ggU, ggV, ggQ, ggal, ggbe, ggR)
        ggsvd!(bk, 'U', 'V', 'Q', ggzAw, ggzBw, ggzU, ggzV, ggzQ, ggal, ggbe, ggzR)
        # ── Least squares. `gelsd!` is absent: MEASURED 100 096 B/call at ComplexF64 because its
        # complex arm goes through the complex `gesvd!`, the one factorization this file already
        # documents as not shown 0-alloc. Its Float64 arm is 0 B, so it becomes eligible the moment
        # complex gesvd! does — the same open item, not a second one.
        gels!(bk, 'N', svAw, lsBw)
        gels!(bk, 'N', svzAw, lszBw)
        gelsy!(bk, svAw, lsBw, lsjp, 1.0e-12)
        gelsy!(bk, svzAw, lszBw, lsjp, 1.0e-12)
        gglse!(bk, ggAw, lscw, ggBw, lsdw, lsx)
        gglse!(bk, ggzAw, lszcw, ggzBw, lszdw, lszx)
        # ── Symmetric-tridiagonal eigen. stein! is fed stebz!'s OWN w/iblock/isplit, trimmed to the
        # m eigenvalues and nsplit blocks found — views, therefore, which is also the SubArray shape
        # its five read-per-column inputs actually arrive in.
        stebz!(bk, 'A', 'B', 0.0, 0.0, 1, 1, -1.0, std, ste, stw, stib, stisp)
        stein!(bk, std, ste, stwv, stibv, stispv, stZ)
        # ── SubArray arguments for the new families, same rationale as the L3 SubArray block above:
        # every assertion so far in this section passes a `Matrix`/`Vector`, and that is precisely the
        # gap that let `_trsm_right!`'s union-typed slot box 64 B/call while every Matrix-only
        # assertion passed. Parents are LARGER than the views, so `lda != nrows`.
        gebal!(bk, vA, vsc; job = 'B')
        gehrd!(bk, vA, 1, ms, vtau)
        orghr!(bk, vA, 1, ms, vtauf)
        ormhr!(bk, 'L', 'N', 1, ms, vAf, vtauf, vC)
        hseqr!(bk, 'S', 'I', vA, 1, ms, vwc, vB)
        trevc!(bk, 'R', 'B', vA, evVL, vB)
        trexc!(bk, 'V', vA, vB, 5, 1)
        trsyl!(bk, 'N', 'N', 1, evU, evU, vC)
        geev!(bk, 'N', 'V', vA, vwr, vwi, evVL, vB, vsc)
        gees!(bk, 'V', vA, vwc, vB, vsc)
        gghrd!(bk, 'I', 'I', vA, vB, vC, vD)
        sytrf!(bk, vA, vip; uplo = 'L')
        sytrs!(bk, vbkf, vip, vBv; uplo = 'L')
        sytri!(bk, vA, vip; uplo = 'L')
        syconv!(bk, 'L', vA, vip, vwk)
        gels!(bk, 'N', vlsA, vlsB)
        gelsy!(bk, vlsA, vlsB, vlsjp, 1.0e-12)
        stein!(bk, std, ste, stwv, stibv, stispv, vstZ)
    end
end

# Register pressure and vectorization of the gemm microkernels. A tile sized
# past the register file spills vectors to the stack on every k step, which neither the type-stability
# nor the allocation checks see; these scan the compiled native code of each listed signature, and
# both `:vectorized` and `:no_spill` gate. The signatures are the FULL method signatures, trailing
# `Val{B0}` included: the 9-argument `_microkernel_masked!` form is a default-argument forwarder whose
# native code is a call and a return, which a scan passes without reaching the kernel. The α-scaled
# `_microkernel!` shares the tile with a different epilogue and has its own `B0 = false` and
# `B0 = true` instances.
@strict_function guarantees = (:vectorized, :no_spill) signatures = [
    (Ptr{Float64}, Int, Ptr{Float64}, Ptr{Float64}, Int, Val{_MR}, Val{_NR}, Val{false}),
    (Ptr{Float64}, Int, Ptr{Float64}, Ptr{Float64}, Int, Float64, Val{_MR}, Val{_NR}, Val{false}),
    (Ptr{Float64}, Int, Ptr{Float64}, Ptr{Float64}, Int, Float64, Val{_MR}, Val{_NR}, Val{true}),
] _microkernel!
@strict_function guarantees = (:vectorized, :no_spill) signatures = [
    (Ptr{Float64}, Int, Ptr{Float64}, Ptr{Float64}, Int, Int, Int, Val{_MR}, Val{_NR}, Val{false}),
] _microkernel_masked!

# Module policy: checks, here at load, every method above whose own signature is concrete and whose
# name it selects — a policy cannot reach a parametric method, whose signature names no concrete types.
# It selects the C-ABI entry points (`*_64_`, the roots `juliac --trim` builds from): concrete inferred
# return type (gates), and no trim-hostile construct StrictMode's static scan can see (reports).
# test/trim_tests.jl runs juliac's verifier over the same roots.
@strict_policy (:typestable, :trim_compatible) only = r"_64_$"
# The trsm side-R tile: the same register-pressure check as the microkernels above (`:no_spill` gates).
@strict_function guarantees = (:no_spill,) signatures = [
    (Bool, Bool, Bool, Ptr{Float64}, Int, Ptr{Float64}, Int, Int, Int),
] _trsm_tile_R_f64!
# Guarantees not checked at load, and why:
#   `:noalloc` on the C-ABI entries: the value-free scan reports about a third of them as allocating
#                       (151 of 492 on Zen4), each a warning at every precompile. The backend calls
#                       under the C ABI carry `:noalloc` in the `@verify_strict` block above.
#   `:inlined`, `:noboxing`, `:no_scalar_loops`, `:no_threadid_state`: each waits for a measurement of
#                       the precompile time and memory it adds.
#   `:owned`            inner functions take their per-task buffer as an argument.
#   `:concurrency_safe` job operands reach chunk bodies as arguments, not through the shared pool.
#   `:noescape`         StrictMode treats a borrowed handle passed to an out-of-line callee as escaping.
#   `:trusted`          StrictMode does not track raw `Ptr` reads.

# The checks are over: put back the logger `_WarnRecorder` wrapped.
let lg = Base.CoreLogging.global_logger()
    lg isa _WarnRecorder && Base.CoreLogging.global_logger(lg.inner)
end
