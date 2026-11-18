# Precompile-time guarantees for the SIMDBackend. Included LAST, after every backend op and C-ABI entry
# is defined, because the declarations below name them.
#
# `@verify_strict SIMDBackend` runs `TypeContracts.@verify SIMDBackend` — method existence + declared
# return types over the whole chain (AbstractBLAS1 → BLAS2 → BLAS3 → LAPACK → LAPACKSolve) — and checks
# each representative signature listed in it:
#   :typestable — the inferred return type must be concrete. This GATES: a failure stops PureBLAS from
#                 precompiling.
#   :noalloc    — StrictMode's value-free IR scan. This REPORTS: a finding is a `StrictViolation`
#                 warning, recorded in `_LOADTIME_WARNINGS` below and compared against
#                 test/loadtime_warnings_baseline.txt by test/loadtime_warnings_tests.jl.
# A listed line is either a signature (`f(::T1, ::T2; k::K)`), which names argument types, or an
# example call, whose arguments are evaluated only for their types; no listed call is performed. A
# keyword line whose method takes `kw...` must be an example call: a signature line can name only
# keywords the method declares. The proofs of both guarantees — JET for stability, AllocCheck for
# allocation, TrimCheck for trim — run from test/ (StrictModeTest is a test-only dependency, so it
# cannot load at this package's own precompile).
#
# Do NOT use StrictMode's IR heuristic (`_alloc_signals`) as the eligibility test for a new
# member: the control run in bench/probes/strict_heuristic_control.jl flags EVERY already-contracted
# LAPACK member — gemm!, potrf!, getrf!, geqrf!, potrs!, getri!, pstrf!, the whole QR family — at a
# measured 0 B. The predicate that means something here is empirical @allocated after warm-up, plus a
# concrete inferred return type; that is what every member below was admitted on
# (bench/probes/strict_contract_eligibility.jl, strict_contract_dryrun.jl).
#
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

# The `view(A, i:j, k:l)` and `view(v, i:j)` types: what the owned-workspace accessors
# (`_pptrf_lower_work`, `_getri_work`, `_trsm_rrefl`, …) hand to the routines below.
const _MatView{T} = SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int}, UnitRange{Int}}, false}
const _VecView{T} = SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}

# Operands of the example-call lines, one per argument type; the checks read only their types.
let bk = DEFAULT_BACKEND, Md = zeros(0, 0), Mz = zeros(ComplexF64, 0, 0), vd = zeros(0),
        vz = zeros(ComplexF64, 0), ip = Int[], Mv = view(zeros(0, 0), 1:0, 1:0),
        vv = view(zeros(0), 1:0), ipv = view(Int[], 1:0)
    # The guarantees are named so every listed signature is also registered, and
    # test_registered (StrictModeTest) re-proves them in the test environment.
    @verify_strict SIMDBackend guarantees = (:typestable, :noalloc, :owned) begin
        # ── Level 1 (bandwidth-bound; SIMD real path + generic complex path)
        axpy!(::SIMDBackend, ::Vector{Float64}, ::Float64, ::Vector{Float64})
        scal!(::SIMDBackend, ::Float64, ::Vector{Float64})
        blascopy!(::SIMDBackend, ::Vector{Float64}, ::Vector{Float64})
        swap!(::SIMDBackend, ::Vector{Float64}, ::Vector{Float64})
        dot(::SIMDBackend, ::Vector{Float64}, ::Vector{Float64})
        dotu(::SIMDBackend, ::Vector{Float64}, ::Vector{Float64})
        nrm2(::SIMDBackend, ::Vector{Float64})
        asum(::SIMDBackend, ::Vector{Float64})
        iamax(::SIMDBackend, ::Vector{Float64})
        axpy!(::SIMDBackend, ::Vector{ComplexF64}, ::ComplexF64, ::Vector{ComplexF64})
        scal!(::SIMDBackend, ::ComplexF64, ::Vector{ComplexF64})         # complex scal: interleaved-SIMD (swap-pairs) path
        dot(::SIMDBackend, ::Vector{ComplexF64}, ::Vector{ComplexF64})   # complex dot/dotu: split-deinterleave SIMD reduction
        dotu(::SIMDBackend, ::Vector{ComplexF64}, ::Vector{ComplexF64})
        nrm2(::SIMDBackend, ::Vector{ComplexF64})                        # complex nrm2/asum now take the SIMD real-reinterpret path
        asum(::SIMDBackend, ::Vector{ComplexF64})
        blascopy!(::SIMDBackend, ::Vector{ComplexF64}, ::Vector{ComplexF64})
        swap!(::SIMDBackend, ::Vector{ComplexF64}, ::Vector{ComplexF64})
        iamax(::SIMDBackend, ::Vector{ComplexF64})                       # izamax: swap-adjacent magnitude reduction (its own kernel)
        # ── Level 2 (dense hot paths; real + complex)
        gemv!(::SIMDBackend, ::Vector{Float64}, ::Matrix{Float64}, ::Vector{Float64}; alpha::Float64, beta::Float64, trans::Char)
        gemv!(::SIMDBackend, ::Vector{ComplexF64}, ::Matrix{ComplexF64}, ::Vector{ComplexF64}; alpha::ComplexF64, beta::ComplexF64, trans::Char)   # complex gemv SIMD
        ger!(::SIMDBackend, ::Float64, ::Vector{Float64}, ::Vector{Float64}, ::Matrix{Float64})
        ger!(::SIMDBackend, ::ComplexF64, ::Vector{ComplexF64}, ::Vector{ComplexF64}, ::Matrix{ComplexF64})   # complex geru/gerc: per-column complex axpy
        symv!(::SIMDBackend, ::Vector{Float64}, ::Matrix{Float64}, ::Vector{Float64})
        hemv!(::SIMDBackend, ::Vector{ComplexF64}, ::Matrix{ComplexF64}, ::Vector{ComplexF64})
        trmv!(::SIMDBackend, ::Matrix{Float64}, ::Vector{Float64})
        trsv!(::SIMDBackend, ::Matrix{Float64}, ::Vector{Float64})
        trmv!(::SIMDBackend, ::Matrix{ComplexF64}, ::Vector{ComplexF64})   # complex trmv/trsv: per-column axpy(N)/dot(T/C) reuse
        trsv!(::SIMDBackend, ::Matrix{ComplexF64}, ::Vector{ComplexF64})
        # ── Level 2 packed storage (symmetric/Hermitian/triangular; rank-1/2 updates)
        spmv!(::SIMDBackend, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Float64}; uplo::Char, alpha::Float64, beta::Float64)
        hpmv!(::SIMDBackend, ::Vector{ComplexF64}, ::Vector{ComplexF64}, ::Vector{ComplexF64}; uplo::Char, alpha::ComplexF64, beta::ComplexF64)
        tpmv!(::SIMDBackend, ::Vector{Float64}, ::Vector{Float64}; uplo::Char, trans::Char, diag::Char)
        tpsv!(::SIMDBackend, ::Vector{Float64}, ::Vector{Float64}; uplo::Char, trans::Char, diag::Char)
        spr!(::SIMDBackend, ::Float64, ::Vector{Float64}, ::Vector{Float64}; uplo::Char)
        spr2!(::SIMDBackend, ::Float64, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Float64}; uplo::Char)
        hpr!(::SIMDBackend, ::Float64, ::Vector{ComplexF64}, ::Vector{ComplexF64}; uplo::Char)
        hpr2!(::SIMDBackend, ::ComplexF64, ::Vector{ComplexF64}, ::Vector{ComplexF64}, ::Vector{ComplexF64}; uplo::Char)
        # ── Level 2 band storage (general/symmetric/Hermitian/triangular banded)
        gbmv!(::SIMDBackend, ::Vector{Float64}, ::Matrix{Float64}, ::Vector{Float64}, ::Int, ::Int, ::Int; trans::Char, alpha::Float64, beta::Float64)
        sbmv!(::SIMDBackend, ::Vector{Float64}, ::Matrix{Float64}, ::Vector{Float64}; uplo::Char, alpha::Float64, beta::Float64)
        hbmv!(::SIMDBackend, ::Vector{ComplexF64}, ::Matrix{ComplexF64}, ::Vector{ComplexF64}; uplo::Char, alpha::ComplexF64, beta::ComplexF64)
        tbmv!(::SIMDBackend, ::Matrix{Float64}, ::Vector{Float64}; uplo::Char, trans::Char, diag::Char)
        tbsv!(::SIMDBackend, ::Matrix{Float64}, ::Vector{Float64}; uplo::Char, trans::Char, diag::Char)
        # ── Level 3 (all matrix-matrix ops; real + complex)
        gemm!(::SIMDBackend, ::Matrix{Float64}, ::Matrix{Float64}, ::Matrix{Float64})
        # COMPLEX gemm: 3M (`_gemm_3m!`) is complex-only, so a real-only `gemm!` line cannot reach it.
        gemm!(::SIMDBackend, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Matrix{ComplexF64})
        symm!(::SIMDBackend, ::Matrix{Float64}, ::Matrix{Float64}, ::Matrix{Float64})
        hemm!(bk, Mz, Mz, Mz; side = 'L', uplo = 'L', alpha = 1.0 + 0im, beta = 1.0 + 0im)
        syrk!(bk, Md, Md; uplo = 'L', trans = 'N', alpha = 1.0, beta = 1.0)
        herk!(bk, Mz, Mz; uplo = 'L', trans = 'N', alpha = 1.0, beta = 1.0)
        syr2k!(bk, Md, Md, Md; uplo = 'L', trans = 'N', alpha = 1.0, beta = 1.0)
        her2k!(bk, Mz, Mz, Mz; uplo = 'L', trans = 'N', alpha = 1.0 + 0im, beta = 1.0)
        trmm!(bk, Md, Md; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0)
        trsm!(bk, Md, Md; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0)
        # COMPLEX arms of the five L3 ops that were verified on the real path only. `hemm!`/`herk!`/
        # `her2k!` above do NOT substitute: the symmetric (non-Hermitian) complex kernels are
        # separate specializations — `zsymm`/`zsyrk`/`zsyr2k` conjugate nothing — and `ztrmm`/`ztrsm`
        # share no code with any of them.
        symm!(::SIMDBackend, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Matrix{ComplexF64})
        syrk!(bk, Mz, Mz; uplo = 'L', trans = 'N', alpha = 1.0 + 0im, beta = 1.0 + 0im)
        syr2k!(bk, Mz, Mz, Mz; uplo = 'L', trans = 'N', alpha = 1.0 + 0im, beta = 1.0 + 0im)
        trmm!(bk, Mz, Mz; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0 + 0im)
        trsm!(bk, Mz, Mz; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0 + 0im)
        # ── SubArray arguments. A slot that holds both the caller's argument and a workspace view is
        # union-typed; that costs nothing when the argument is a `Matrix`, and boxes when it is a
        # `SubArray`. The owned-workspace panels hand views to exactly these ops.
        trsm!(bk, Mv, Mv; side = 'R', uplo = 'L', transA = 'T', diag = 'N', alpha = 1.0)
        trmm!(bk, Mv, Mv; side = 'R', uplo = 'L', transA = 'T', diag = 'N', alpha = 1.0)
        gemm!(::SIMDBackend, ::_MatView{Float64}, ::_MatView{Float64}, ::_MatView{Float64})
        syrk!(bk, Mv, Mv; uplo = 'L', trans = 'N', alpha = 1.0, beta = 1.0)
        # ── LAPACK: potrf!/getrf!/geqrf! are 0-alloc (potrf via its own pointer kernels; LU/QR via
        # their in-place pre-allocated-output forms). gesvd! reaches 0-alloc via the in-place
        # gesvd!(A,U,S,Vᵀ) form + a cached SVDWorkspace for the bidiagonalization scratch.
        potrf!(bk, Md; uplo = 'L')
        getrf!(::Matrix{Float64}, ::Vector{Int})
        geqrf!(::Matrix{Float64}, ::Vector{Float64})
        gesvd!(::Matrix{Float64}, ::Matrix{Float64}, ::Vector{Float64}, ::Matrix{Float64})
        # COMPLEX factorizations — a wholly separate kernel set (`_cpotrf_lower!`, `_cgetf2_simd!`,
        # the complex QR panel).
        # gesvd is deliberately NOT included for complex: its in-place form caches a Float64
        # SVDWorkspace and the complex path has not been shown 0-alloc — asserting it here without
        # that evidence would be guessing, so it stays an open item rather than a silent pass.
        potrf!(bk, Mz; uplo = 'L')
        getrf!(::Matrix{ComplexF64}, ::Vector{Int})
        geqrf!(::Matrix{ComplexF64}, ::Vector{ComplexF64})
        # ── AbstractLAPACKSolve (contracts.jl): the solve/inverse/condition surface built on those
        # four factorizations. Declaring the contract only fixes the method signatures; it is THIS
        # block that makes "type-stable AND allocation-free" true. Grouped by the factorization they
        # hang off.
        #
        # Cholesky family: solve, explicit inverse (potri! takes the FACTOR, not the SPD matrix,
        # and overwrites it with A⁻¹), packed and banded factor+solve, the rank-revealing pivoted
        # factor, and the condition estimate.
        potrs!(bk, Md, Md; uplo = 'L')
        potri!(bk, Md; uplo = 'L')
        pptrf!(bk, vd; uplo = 'L')
        pptrs!(bk, vd, Md; uplo = 'L')
        pbtrf!(bk, Md; uplo = 'L', kd = 1)
        pbtrs!(bk, Md, Md; uplo = 'L')
        pstrf!(bk, Md, ip, -1.0; uplo = 'L')
        pocon!(bk, 1.0, Md; uplo = 'L')
        # LU family: solve against the factors and the explicit inverse. getri! runs LAPACK's
        # block-column algorithm over an owned n×nb buffer, which is exactly what this line holds it to.
        getrs!(bk, Md, ip, Md; trans = 'N')
        getri!(::SIMDBackend, ::Matrix{Float64}, ::Vector{Int})
        # Triangular: solve, in-place inverse, condition estimate, and the trrfs! error bounds.
        trtrs!(bk, Md, Md; uplo = 'L', trans = 'N', diag = 'N')
        trtri!(bk, Md; uplo = 'L', diag = 'N')
        gecon!(bk, 1.0, Md, ip; norm = '1')
        trcon!(bk, Md; uplo = 'L', diag = 'N', norm = '1')
        trrfs!(::SIMDBackend, ::Char, ::Char, ::Char, ::Matrix{Float64}, ::Matrix{Float64}, ::Matrix{Float64}, ::Vector{Float64}, ::Vector{Float64})
        # Banded LU: the IN-PLACE gbtrf!(kl, ku, m, AB, ipiv) form, not the convenience form that
        # allocates ipiv — the contract binds the former precisely so it cannot be satisfied by a
        # method that allocates its own output.
        gbtrf!(::SIMDBackend, ::Int, ::Int, ::Int, ::Matrix{Float64}, ::Vector{Int})
        gbtrs!(::SIMDBackend, ::Char, ::Int, ::Int, ::Int, ::Matrix{Float64}, ::Vector{Int}, ::Matrix{Float64})
        # Tridiagonal: the fused driver, the split factor/solve pair, and their SPD counterparts.
        # These are pure recurrences with no BLAS underneath, so an allocation here would be a
        # bounds-check or boxing artefact rather than a workspace — worth pinning for that reason.
        gtsv!(::SIMDBackend, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Float64}, ::Matrix{Float64})
        gttrf!(::SIMDBackend, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Int})
        gttrs!(::SIMDBackend, ::Char, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Int}, ::Matrix{Float64})
        pttrf!(::SIMDBackend, ::Vector{Float64}, ::Vector{Float64})
        pttrs!(bk, vd, vd, Md; uplo = 'L')
        ptsv!(bk, vd, vd, Md; uplo = 'L')
        # Expert driver — the deepest composition in the contract (equilibrate → getrf → solve →
        # gecon ×2 → iterative refinement), so it is also the one most likely to leak an allocation
        # from a callee. Again the IN-PLACE form: X/ferr/berr come from the caller, which is what
        # netlib's OUT arguments always implied.
        gesvx!(::SIMDBackend, ::Char, ::Char, ::Matrix{Float64}, ::Matrix{Float64}, ::Vector{Int}, ::Char, ::Vector{Float64}, ::Vector{Float64}, ::Matrix{Float64}, ::Matrix{Float64}, ::Vector{Float64}, ::Vector{Float64})
        # ── QR / LQ / QL / RZ. 14 distinct implementations; the nine `un*!` names are `const`
        # aliases of these same function objects, so verifying `or*!` verifies them too.
        gelqf!(::Matrix{Float64}, ::Vector{Float64})
        geqlf!(::Matrix{Float64}, ::Vector{Float64})
        gerqf!(::Matrix{Float64}, ::Vector{Float64})
        tzrzf!(::Matrix{Float64}, ::Vector{Float64})
        geqp3!(::Matrix{Float64}, ::Vector{Int}, ::Vector{Float64})
        orgqr!(::Matrix{Float64}, ::Vector{Float64})
        orglq!(::Matrix{Float64}, ::Vector{Float64})
        orgql!(::Matrix{Float64}, ::Vector{Float64})
        orgrq!(::Matrix{Float64}, ::Vector{Float64})
        ormqr!(::Char, ::Char, ::Matrix{Float64}, ::Vector{Float64}, ::Matrix{Float64})
        ormlq!(::Char, ::Char, ::Matrix{Float64}, ::Vector{Float64}, ::Matrix{Float64})
        ormql!(::Char, ::Char, ::Matrix{Float64}, ::Vector{Float64}, ::Matrix{Float64})
        ormrq!(::Char, ::Char, ::Matrix{Float64}, ::Vector{Float64}, ::Matrix{Float64})
        ormrz!(::Char, ::Char, ::Matrix{Float64}, ::Vector{Float64}, ::Matrix{Float64})
        orgtr!(::SIMDBackend, ::Char, ::Matrix{Float64}, ::Vector{Float64}, ::Matrix{Float64})
        ungtr!(::SIMDBackend, ::Char, ::Matrix{ComplexF64}, ::Vector{ComplexF64}, ::Matrix{ComplexF64})
        # ── Nonsymmetric eigen: balance, Hessenberg reduction, Schur, eigenvectors, drivers.
        # REAL and COMPLEX arms both, because they are separate kernel sets throughout (dlahqr vs
        # zlahqr, the real quasi-triangular 2×2 machinery vs plain triangular back-substitution), and
        # a real-only assertion is exactly how ~1 KB/call in the complex 3M gemm passed for a month.
        gebal!(bk, Md, vd; job = 'B')
        gebak!(::SIMDBackend, ::Char, ::Char, ::Int, ::Int, ::Vector{Float64}, ::Matrix{Float64})
        gehrd!(::SIMDBackend, ::Matrix{Float64}, ::Int, ::Int, ::Vector{Float64})
        orghr!(::SIMDBackend, ::Matrix{Float64}, ::Int, ::Int, ::Vector{Float64})
        ormhr!(::SIMDBackend, ::Char, ::Char, ::Int, ::Int, ::Matrix{Float64}, ::Vector{Float64}, ::Matrix{Float64})
        hseqr!(::SIMDBackend, ::Char, ::Char, ::Matrix{Float64}, ::Int, ::Int, ::Vector{ComplexF64}, ::Matrix{Float64})
        trevc!(::SIMDBackend, ::Char, ::Char, ::Matrix{Float64}, ::Matrix{Float64}, ::Matrix{Float64})
        trexc!(::SIMDBackend, ::Char, ::Matrix{Float64}, ::Matrix{Float64}, ::Int, ::Int)
        trsen!(::SIMDBackend, ::Char, ::Char, ::Vector{Int}, ::Matrix{Float64}, ::Matrix{Float64}, ::Vector{ComplexF64})
        sygvd!(::SIMDBackend, ::Int, ::Char, ::Char, ::Matrix{Float64}, ::Matrix{Float64}, ::Vector{Float64})
        hegvd!(::SIMDBackend, ::Int, ::Char, ::Char, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Vector{Float64})
        trsyl!(::SIMDBackend, ::Char, ::Char, ::Int, ::Matrix{Float64}, ::Matrix{Float64}, ::Matrix{Float64})
        geev!(::SIMDBackend, ::Char, ::Char, ::Matrix{Float64}, ::Vector{Float64}, ::Vector{Float64}, ::Matrix{Float64}, ::Matrix{Float64}, ::Vector{Float64})
        gees!(::SIMDBackend, ::Char, ::Matrix{Float64}, ::Vector{ComplexF64}, ::Matrix{Float64}, ::Vector{Float64})
        gebal!(bk, Mz, vd; job = 'B')
        gebak!(::SIMDBackend, ::Char, ::Char, ::Int, ::Int, ::Vector{Float64}, ::Matrix{ComplexF64})
        gehrd!(::SIMDBackend, ::Matrix{ComplexF64}, ::Int, ::Int, ::Vector{ComplexF64})
        orghr!(::SIMDBackend, ::Matrix{ComplexF64}, ::Int, ::Int, ::Vector{ComplexF64})
        ormhr!(::SIMDBackend, ::Char, ::Char, ::Int, ::Int, ::Matrix{ComplexF64}, ::Vector{ComplexF64}, ::Matrix{ComplexF64})
        hseqr!(::SIMDBackend, ::Char, ::Char, ::Matrix{ComplexF64}, ::Int, ::Int, ::Vector{ComplexF64}, ::Matrix{ComplexF64})
        trevc!(::SIMDBackend, ::Char, ::Char, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Matrix{ComplexF64})
        trexc!(::SIMDBackend, ::Char, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Int, ::Int)
        trsen!(::SIMDBackend, ::Char, ::Char, ::Vector{Int}, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Vector{ComplexF64})
        trsyl!(::SIMDBackend, ::Char, ::Char, ::Int, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Matrix{ComplexF64})
        geev!(::SIMDBackend, ::Char, ::Char, ::Matrix{ComplexF64}, ::Vector{ComplexF64}, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Vector{Float64})
        gees!(::SIMDBackend, ::Char, ::Matrix{ComplexF64}, ::Vector{ComplexF64}, ::Matrix{ComplexF64}, ::Vector{Float64})
        # ── Bunch-Kaufman. For REAL eltype `hetrf!`/`hetrs!`/`hetri!`/`hesv!` ARE `sytrf!`/…
        # (herm=false).
        sytrf!(bk, Md, ip; uplo = 'L')
        hetrf!(bk, Md, ip; uplo = 'L')
        sytrs!(bk, Md, ip, Md; uplo = 'L')
        # VECTOR right-hand side: `sytrs!` must not `reshape(B, n, 1)` it — that is 48 B/call of
        # Array header, and a `Base.ReshapedArray` (the trim-hostile `_throw_dmrs` path) when the RHS
        # is itself a view.
        sytrs!(bk, Md, ip, vd; uplo = 'L')
        hetrs!(bk, Md, ip, Md; uplo = 'L')
        sytri!(bk, Md, ip; uplo = 'L')
        hetri!(bk, Md, ip; uplo = 'L')
        sysv!(::SIMDBackend, ::Char, ::Matrix{Float64}, ::Matrix{Float64}, ::Vector{Int})
        hesv!(::SIMDBackend, ::Char, ::Matrix{Float64}, ::Matrix{Float64}, ::Vector{Int})
        syconv!(::SIMDBackend, ::Char, ::Matrix{Float64}, ::Vector{Int}, ::Vector{Float64})
        sytrf!(bk, Mz, ip; uplo = 'L')
        hetrf!(bk, Mz, ip; uplo = 'L')
        sytrs!(bk, Mz, ip, Mz; uplo = 'L')
        sytrs!(bk, Mz, ip, vz; uplo = 'L')
        hetrs!(bk, Mz, ip, Mz; uplo = 'L')
        sytri!(bk, Mz, ip; uplo = 'L')
        hetri!(bk, Mz, ip; uplo = 'L')
        sysv!(::SIMDBackend, ::Char, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Vector{Int})
        hesv!(::SIMDBackend, ::Char, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Vector{Int})
        syconv!(::SIMDBackend, ::Char, ::Matrix{ComplexF64}, ::Vector{Int}, ::Vector{ComplexF64})
        # ── QZ / generalized eigen. `ggev!` appears at both arities — the contract can only declare
        # the real one (alphar, alphai, beta), so the complex form is held to the guarantee here.
        gghrd!(::SIMDBackend, ::Char, ::Char, ::Matrix{Float64}, ::Matrix{Float64}, ::Matrix{Float64}, ::Matrix{Float64})
        hgeqz!(::SIMDBackend, ::Char, ::Char, ::Char, ::Matrix{Float64}, ::Matrix{Float64}, ::Vector{ComplexF64}, ::Vector{Float64}, ::Matrix{Float64}, ::Matrix{Float64})
        tgevc!(::SIMDBackend, ::Char, ::Char, ::Matrix{Float64}, ::Matrix{Float64}, ::Matrix{Float64}, ::Matrix{Float64})
        tgsen!(::SIMDBackend, ::Vector{Int}, ::Matrix{Float64}, ::Matrix{Float64}, ::Matrix{Float64}, ::Matrix{Float64}, ::Vector{ComplexF64}, ::Vector{Float64})
        ggev!(::SIMDBackend, ::Char, ::Char, ::Matrix{Float64}, ::Matrix{Float64}, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Float64}, ::Matrix{Float64}, ::Matrix{Float64})
        gges!(::SIMDBackend, ::Char, ::Char, ::Matrix{Float64}, ::Matrix{Float64}, ::Vector{ComplexF64}, ::Vector{Float64}, ::Matrix{Float64}, ::Matrix{Float64})
        gghrd!(::SIMDBackend, ::Char, ::Char, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Matrix{ComplexF64})
        hgeqz!(::SIMDBackend, ::Char, ::Char, ::Char, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Vector{ComplexF64}, ::Vector{ComplexF64}, ::Matrix{ComplexF64}, ::Matrix{ComplexF64})
        tgevc!(::SIMDBackend, ::Char, ::Char, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Matrix{ComplexF64})
        tgsen!(::SIMDBackend, ::Vector{Int}, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Vector{ComplexF64}, ::Vector{ComplexF64})
        ggev!(::SIMDBackend, ::Char, ::Char, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Vector{ComplexF64}, ::Vector{ComplexF64}, ::Matrix{ComplexF64}, ::Matrix{ComplexF64})
        gges!(::SIMDBackend, ::Char, ::Char, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Vector{ComplexF64}, ::Vector{ComplexF64}, ::Matrix{ComplexF64}, ::Matrix{ComplexF64})
        # ── SVD front half + generalized SVD. `gebrd!`/`bdsdc!` are asserted at their workspace-free
        # arity (the `ws`-taking forms stay internal); see contracts.jl. `bdsqr!` is asserted at three
        # signatures: real with U/V, real with `nothing`/`nothing` (the values-only path, a Union
        # argument and so the one that could union-split), and the complex 6-argument kernel, which
        # shares no code with either.
        gebd2!(::SIMDBackend, ::Matrix{Float64}, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Float64})
        gebd2!(::SIMDBackend, ::Matrix{ComplexF64}, ::Vector{Float64}, ::Vector{Float64}, ::Vector{ComplexF64}, ::Vector{ComplexF64})
        gebrd!(::SIMDBackend, ::Matrix{Float64}, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Float64})
        gebrd!(::SIMDBackend, ::Matrix{ComplexF64}, ::Vector{Float64}, ::Vector{Float64}, ::Vector{ComplexF64}, ::Vector{ComplexF64})
        bdsdc!(::SIMDBackend, ::Vector{Float64}, ::Vector{Float64}, ::Matrix{Float64}, ::Matrix{Float64})
        bdsqr!(::SIMDBackend, ::Vector{Float64}, ::Vector{Float64}, ::Matrix{Float64}, ::Matrix{Float64})
        bdsqr!(::SIMDBackend, ::Vector{Float64}, ::Vector{Float64}, ::Nothing, ::Nothing)
        bdsqr!(::SIMDBackend, ::Char, ::Vector{Float64}, ::Vector{Float64}, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Matrix{ComplexF64})
        ggsvd!(::SIMDBackend, ::Char, ::Char, ::Char, ::Matrix{Float64}, ::Matrix{Float64}, ::Matrix{Float64}, ::Matrix{Float64}, ::Matrix{Float64}, ::Vector{Float64}, ::Vector{Float64}, ::Matrix{Float64})
        ggsvd!(::SIMDBackend, ::Char, ::Char, ::Char, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Vector{Float64}, ::Vector{Float64}, ::Matrix{ComplexF64})
        # ── Least squares. `gelsd!` is absent: MEASURED 100 096 B/call at ComplexF64 because its
        # complex arm goes through the complex `gesvd!`, the one factorization this file already
        # documents as not shown 0-alloc. Its Float64 arm is 0 B, so it becomes eligible the moment
        # complex gesvd! does — the same open item, not a second one.
        gels!(::SIMDBackend, ::Char, ::Matrix{Float64}, ::Matrix{Float64})
        gels!(::SIMDBackend, ::Char, ::Matrix{ComplexF64}, ::Matrix{ComplexF64})
        gelsy!(::SIMDBackend, ::Matrix{Float64}, ::Matrix{Float64}, ::Vector{Int}, ::Float64)
        gelsy!(::SIMDBackend, ::Matrix{ComplexF64}, ::Matrix{ComplexF64}, ::Vector{Int}, ::Float64)
        gglse!(::SIMDBackend, ::Matrix{Float64}, ::Vector{Float64}, ::Matrix{Float64}, ::Vector{Float64}, ::Vector{Float64})
        gglse!(::SIMDBackend, ::Matrix{ComplexF64}, ::Vector{ComplexF64}, ::Matrix{ComplexF64}, ::Vector{ComplexF64}, ::Vector{ComplexF64})
        # ── Symmetric-tridiagonal eigen. stein! takes stebz!'s w/iblock/isplit trimmed to the
        # eigenvalues and blocks found — views, therefore, which is the SubArray shape its
        # read-per-column inputs arrive in.
        stebz!(::SIMDBackend, ::Char, ::Char, ::Float64, ::Float64, ::Int, ::Int, ::Float64, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Int}, ::Vector{Int})
        stein!(::SIMDBackend, ::Vector{Float64}, ::Vector{Float64}, ::_VecView{Float64}, ::_VecView{Int}, ::_VecView{Int}, ::Matrix{Float64})
        # ── SubArray arguments for the families above, same rationale as the L3 SubArray lines.
        gebal!(bk, Mv, vv; job = 'B')
        gehrd!(::SIMDBackend, ::_MatView{Float64}, ::Int, ::Int, ::_VecView{Float64})
        orghr!(::SIMDBackend, ::_MatView{Float64}, ::Int, ::Int, ::_VecView{Float64})
        ormhr!(::SIMDBackend, ::Char, ::Char, ::Int, ::Int, ::Matrix{Float64}, ::_VecView{Float64}, ::_MatView{Float64})
        hseqr!(::SIMDBackend, ::Char, ::Char, ::_MatView{Float64}, ::Int, ::Int, ::_VecView{ComplexF64}, ::_MatView{Float64})
        trevc!(::SIMDBackend, ::Char, ::Char, ::_MatView{Float64}, ::Matrix{Float64}, ::_MatView{Float64})
        trexc!(::SIMDBackend, ::Char, ::_MatView{Float64}, ::_MatView{Float64}, ::Int, ::Int)
        trsyl!(::SIMDBackend, ::Char, ::Char, ::Int, ::Matrix{Float64}, ::Matrix{Float64}, ::_MatView{Float64})
        geev!(::SIMDBackend, ::Char, ::Char, ::_MatView{Float64}, ::_VecView{Float64}, ::_VecView{Float64}, ::Matrix{Float64}, ::_MatView{Float64}, ::_VecView{Float64})
        gees!(::SIMDBackend, ::Char, ::_MatView{Float64}, ::_VecView{ComplexF64}, ::_MatView{Float64}, ::_VecView{Float64})
        gghrd!(::SIMDBackend, ::Char, ::Char, ::_MatView{Float64}, ::_MatView{Float64}, ::_MatView{Float64}, ::_MatView{Float64})
        sytrf!(bk, Mv, ipv; uplo = 'L')
        sytrs!(bk, Md, ipv, Mv; uplo = 'L')
        sytri!(bk, Mv, ipv; uplo = 'L')
        syconv!(::SIMDBackend, ::Char, ::_MatView{Float64}, ::_VecView{Int}, ::_VecView{Float64})
        gels!(::SIMDBackend, ::Char, ::_MatView{Float64}, ::_MatView{Float64})
        gelsy!(::SIMDBackend, ::_MatView{Float64}, ::_MatView{Float64}, ::_VecView{Int}, ::Float64)
        stein!(::SIMDBackend, ::Vector{Float64}, ::Vector{Float64}, ::_VecView{Float64}, ::_VecView{Int}, ::_VecView{Int}, ::_MatView{Float64})
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
