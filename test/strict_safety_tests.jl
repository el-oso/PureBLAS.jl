# StrictMode guarantees for threading and memory safety. Every check that reads code or arguments comes
# with a control that must be flagged, so a pass is evidence the check looks at the thing it names.
#
# Items that need StrictMode APIs newer than the registered 0.5.10 (`spin_wait_report`, `reach_report`,
# `route_parity`, `threadcount_invariant_report`, `threaded_speedup_report`) test
# `isdefined(StrictMode, ...)` first and record a `@test_skip` where the API is absent.

@testmodule SafetyHelp begin
    using StrictMode, Test

    # True when the guarantee raised a violation for `f`, false when it passed; any other error is a
    # bug in the test and propagates.
    function flags(f)
        try
            f()
            return false
        catch e
            e isa StrictMode.StrictViolation || rethrow()
            return true
        end
    end

    const THREADED_ONLY = (:_gemm_threaded!, :_trmvt_threaded!, :_trmvn_threaded!, :_gbmvn_threaded!)
end

# ── @assert_no_threadid_state ───────────────────────────────────────────────────────────────────────
@testitem "strict safety: threadid state, owners and controls" tags = [:checks] setup = [SafetyHelp] begin
    using StrictMode, StrictModeTest
    import PureBLAS as P
    using .SafetyHelp: flags
    @test StrictMode.checks_enabled()

    # Controls: a store into a slot chosen by `threadid()`, directly and through an object loaded out of
    # it, and a `OncePerThread` owner written inline. All three must be flagged.
    bufs = [zeros(4) for _ in 1:Threads.nthreads()]
    global const _CTL_BUFS = bufs
    global const _CTL_OPT = Base.OncePerThread{Vector{Float64}}(() -> Float64[])
    global const _CTL_OPK = Base.OncePerTask{Vector{Float64}}(() -> Float64[])
    ctl_slot_elem(i) = (_CTL_BUFS[Threads.threadid()][1] = i; nothing)
    ctl_slot_replace(i) = (_CTL_BUFS[Threads.threadid()] = [1.0]; nothing)
    ctl_oncepert(n) = (v = _CTL_OPT(); resize!(v, n); v[1] = 1.0; nothing)
    ctl_oncepertask(n) = (v = _CTL_OPK(); resize!(v, n); v[1] = 1.0; nothing)
    ctl_local(i) = (b = zeros(4); b[1] = i; nothing)
    @test flags(() -> StrictMode.@assert_no_threadid_state ctl_slot_elem(1))
    @test flags(() -> StrictMode.@assert_no_threadid_state ctl_slot_replace(1))
    @test flags(() -> StrictMode.@assert_no_threadid_state ctl_oncepert(5))
    @test !flags(() -> StrictMode.@assert_no_threadid_state ctl_oncepertask(5))
    @test !flags(() -> StrictMode.@assert_no_threadid_state ctl_local(1))

    # The per-thread and per-task owners, each written through its accessor.
    w_gpk(n) = P._ws_grow!(P._gpackws(Float64)[1], n)
    w_m3(n) = P._ws_grow!(P._m3ws(Float64)[1], n)
    w_str(n) = P._ws_grow!(P._strws(Float64)[2], n)
    w_l3(n) = P._ws_grow!(P._l3ws(Float64).cg[1], n)
    w_acc(n) = P._ws_grow!(P._trmv_accbuf(Float64), n)
    w_slot(n) = P._ws_slot!(Vector{Float64}[], 2, n)
    for w in (w_gpk, w_m3, w_str, w_l3, w_acc, w_slot)
        @test !flags(() -> StrictMode.@assert_no_threadid_state w(16))
    end
    @test !flags(() -> StrictMode.@assert_no_threadid_state P._arena())
    @test !flags(() -> StrictMode.@assert_no_threadid_state P._symm_scr(Float64, 8))
    @test !flags(() -> StrictMode.@assert_no_threadid_state P._mt_poll_span())
    A = randn(64, 64); x = randn(64)
    trmv_split!(x, A) = P._trmv_split!(true, false, size(A, 1), A, x)
    @test !flags(() -> StrictMode.@assert_no_threadid_state trmv_split!(x, A))

    # The scan does not follow a call left out of line, and every PureBLAS owner is written through the
    # out-of-line `_ws_grow!`, so the per-thread owners above pass without being read. A hazardous toy
    # written through such a call passes the same way; this marks the gap until the scan follows calls.
    global const _CTL_OPT2 = Base.OncePerThread{Vector{Float64}}(() -> Float64[])
    ctl_oncepert_noinline(n) = (P._ws_grow!(_CTL_OPT2(), n)[1] = 1.0; nothing)
    @test_broken flags(() -> StrictMode.@assert_no_threadid_state ctl_oncepert_noinline(5))

    # What the lexical lint knows: every `OncePerThread` owner it reviews is invisible to the scan above.
    include(joinpath(pkgdir(P), "test", "perthread_lint.jl"))
    owners = perthread_scan()
    @test length(owners) >= 20
end

# ── @assert_concurrency_safe and pool_balance_report ─────────────────────────────────────────────────
@testitem "strict safety: concurrency_safe and pool balance" tags = [:checks] setup = [SafetyHelp] begin
    using StrictMode, StrictModeTest, LinearAlgebra
    import PureBLAS as P
    using .SafetyHelp: flags
    @test StrictMode.checks_enabled()

    n = 64
    A = randn(n, n); B = randn(n, n); C = zeros(n, n); x = randn(n); y = zeros(n)
    gemm3!(C, A, B) = P.gemm!(C, A, B)
    syrk3!(C, A) = P.syrk!(C, A)
    gemv3!(y, A, x) = P.gemv!(y, A, x)
    axpy3!(y, a, x) = P.axpy!(y, a, x)
    ger3!(A, x, y) = P.ger!(1.0, x, y, A)
    trmv3!(x, A) = P.trmv!(A, x)
    trmm3!(B, A) = P.trmm!(B, A)
    trsm3!(B, A) = P.trsm!(B, A)

    # Control: a write to the shared operand, through a field-level store, is flagged.
    badw!(out, plan) = (plan[1] = 0.0; out[1] = 1.0; nothing)
    @test flags(() -> StrictMode.@assert_concurrency_safe self = 2 badw!(C, A))

    # The shared operands are read-only: A and B of gemm, A of syrk/gemv/trmm/trsm, x of axpy/gemv/ger.
    @test !flags(() -> StrictMode.@assert_concurrency_safe self = 2 gemm3!(C, A, B))
    @test !flags(() -> StrictMode.@assert_concurrency_safe self = 3 gemm3!(C, A, B))
    @test !flags(() -> StrictMode.@assert_concurrency_safe self = 2 syrk3!(C, A))
    @test !flags(() -> StrictMode.@assert_concurrency_safe self = 2 gemv3!(y, A, x))
    @test !flags(() -> StrictMode.@assert_concurrency_safe self = 3 gemv3!(y, A, x))
    @test !flags(() -> StrictMode.@assert_concurrency_safe self = 3 axpy3!(y, 2.0, x))
    @test !flags(() -> StrictMode.@assert_concurrency_safe self = 2 trmm3!(copy(B), A))
    @test !flags(() -> StrictMode.@assert_concurrency_safe self = 2 trsm3!(copy(B), A + 64I))

    # The kernels write through raw pointers, and so does this control, which the scan does not follow.
    # The passes above therefore cover less than they name.
    badp!(out, plan) = (GC.@preserve plan unsafe_store!(pointer(plan), 0.0); out[1] = 1.0; nothing)
    @test_broken flags(() -> StrictMode.@assert_concurrency_safe self = 2 badp!(C, A))

    # pool_balance_report counts lock/take! against unlock/put! in the driver's own IR.
    unbal(l) = (lock(l); 1)
    bal(l) = (lock(l); unlock(l); 1)
    @test !StrictMode.pool_balance_report(unbal, (ReentrantLock,)).balanced
    @test StrictMode.pool_balance_report(bal, (ReentrantLock,)).balanced
    M = Matrix{Float64}
    r = StrictMode.pool_balance_report(gemm3!, (M, M, M))
    @test r.balanced
    # The pool is claimed with an atomic compare-and-swap, which the count does not see: the report on the
    # threaded gemm driver counts nothing in either direction.
    @test_broken r.takes + r.puts > 0
    # A worker that waits on a `Threads.Event` is counted as an unreleased acquire.
    rw = StrictMode.pool_balance_report(P._gemm_pool_worker, (P.GemmPool{Float64}, Int))
    @test_broken rw.balanced
end

# ── memsafe: guard pages on the direct-read kernels ──────────────────────────────────────────────────
# Each probe runs in a subprocess started with `--project=$(Base.active_project())` that loads
# Serialization, so the active project must list it as a direct dependency.
@testitem "strict safety: memsafe on direct-read kernels" tags = [:checks] begin
    using StrictMode, StrictModeTest
    import PureBLAS as P
    @test StrictMode.checks_enabled()

    clean(f, args...) = (r = memsafe_report(f, args...; using_module = :PureBLAS); isnothing(r.violation) && isempty(r.unguarded))
    viol(f, args...) = !isnothing(memsafe_report(f, args...; using_module = :PureBLAS).violation)

    # Controls: the same kernels with a dimension that exceeds the guarded operand read past its end.
    @test viol(P._gemv!, false, false, 4096, 40, 1.0, randn(8, 40), randn(40), 1, 0.0, zeros(4096), 1)
    @test viol(P._gbmv!, true, false, 4096, 96, 15, 16, 1.0, randn(32, 96), randn(40), 1, 0.0, zeros(96), 1)
    @test viol(P._gemm_core!, zeros(64, 453), randn(64, 64), randn(64, 8), 1.0, 0.0, false, false, false, false, (-1, -1, -1))

    # Controls in the level-1 axpy shape (raw pointers, one element past the end): the harness must name
    # a store past the end as a WRITE and a load past the end as a READ, so a clean verdict on the real
    # axpy kernel below is a verdict from a probe that can fail.
    include(joinpath(@__DIR__, "memsafe_controls.jl"))
    rs = memsafe_report(ctl_store_past!, 37, 1.5, randn(37), randn(37))
    @test !isnothing(rs.violation) && occursin("out-of-bounds WRITE", rs.violation)
    rl = memsafe_report(ctl_load_past!, 37, 1.5, randn(37), randn(37))
    @test !isnothing(rl.violation) && occursin("out-of-bounds READ", rl.violation)

    # axpy: n = 37 leaves a scalar tail; n = 64 is a multiple of every vector width and unroll step.
    for T in (Float64, Float32), n in (37, 64)
        @test clean(P._axpy_simd!, n, T(1.5), randn(T, n), randn(T, n))
    end

    # gemm: unpacked direct reads with partial tiles, and the blocked direct-B masked tail (n = 453).
    for (m, k, n) in ((12, 8, 5), (13, 7, 11), (64, 64, 453))
        @test clean(P.gemm!, zeros(m, n), randn(m, k), randn(k, n))
    end

    # gemv: rectangular and odd shapes, both transposes, real and complex.
    for T in (Float64, Float32, ComplexF64), (m, n) in ((97, 40), (40, 97), (41, 41))
        A = randn(T, m, n)
        @test clean(P._gemv!, false, false, m, n, one(T), A, randn(T, n), 1, zero(T), zeros(T, m), 1)
        @test clean(P._gemv!, true, false, m, n, one(T), A, randn(T, m), 1, zero(T), zeros(T, n), 1)
    end

    # gbmv: m > n overruns y, m < n gives the empty-column suffix, odd n hits the single-column tail.
    for T in (Float64, Float32, ComplexF64), (kl, ku) in ((15, 16), (31, 32)), (m, n) in ((40, 96), (96, 40), (40, 97), (97, 40), (41, 41))
        AB = randn(T, kl + ku + 1, n)
        @test clean(P._gbmv!, false, false, m, n, kl, ku, one(T), AB, randn(T, n), 1, zero(T), zeros(T, m), 1)
        @test clean(P._gbmv!, true, false, m, n, kl, ku, one(T), AB, randn(T, m), 1, zero(T), zeros(T, n), 1)
    end

    # ger: rectangular, n not a multiple of any panel width.
    for T in (Float64, ComplexF64), (m, n) in ((97, 40), (40, 97))
        @test clean(P._ger!, false, m, n, one(T), randn(T, m), 1, randn(T, n), 1, randn(T, m, n))
    end
end

# ── @assert_noalias and alias_guard_report ───────────────────────────────────────────────────────────
@testitem "strict safety: noalias on the mutating drivers" tags = [:checks] setup = [SafetyHelp] begin
    using StrictMode, StrictModeTest, LinearAlgebra
    import PureBLAS as P
    using .SafetyHelp: flags
    @test StrictMode.checks_enabled()

    n = 64
    A = randn(n, n); B = randn(n, n); C = zeros(n, n); x = randn(n); y = zeros(n)
    gemm3!(C, A, B) = P.gemm!(C, A, B)
    syrk3!(C, A) = P.syrk!(C, A)
    ger3!(A, x, y) = P.ger!(1.0, x, y, A)
    axpy3!(y, a, x) = P.axpy!(y, a, x)
    potrf3!(A) = P.potrf!(A)

    # Distinct operands are accepted.
    @test !flags(() -> StrictMode.@assert_noalias gemm3!(C, A, B))
    @test !flags(() -> StrictMode.@assert_noalias syrk3!(C, A))
    @test !flags(() -> StrictMode.@assert_noalias ger3!(C, x, y))
    @test !flags(() -> StrictMode.@assert_noalias axpy3!(y, 2.0, x))
    # Controls: the destination reappears as an input, directly or through a view.
    @test flags(() -> StrictMode.@assert_noalias gemm3!(C, C, B))
    @test flags(() -> StrictMode.@assert_noalias gemm3!(C, A, view(C, :, :)))
    @test flags(() -> StrictMode.@assert_noalias syrk3!(C, C))
    @test flags(() -> StrictMode.@assert_noalias axpy3!(y, 2.0, y))
    @test flags(() -> StrictMode.@assert_noalias ger3!(C, view(C, :, 1), y))
    # potrf! writes its only array operand: there is no second operand to overlap.
    S = let M = randn(n, n); M * M' + n * I; end
    @test !flags(() -> StrictMode.@assert_noalias potrf3!(S))

    # An aliased call is accepted by the drivers and returns wrong values. `mul!` throws on these.
    ref_a = let C1 = randn(n, n); B1 = randn(n, n); ref = copy(C1) * B1; P.gemm!(C1, C1, B1); isapprox(C1, ref) end
    ref_b = let C1 = randn(n, n); A1 = randn(n, n); ref = A1 * copy(C1); P.gemm!(C1, A1, C1); isapprox(C1, ref) end
    ref_g = let A1 = randn(n, n); x1 = view(A1, :, 1); y1 = randn(n); ref = A1 + collect(x1) * y1'; P.ger!(1.0, x1, y1, A1); isapprox(A1, ref) end
    @test_broken ref_a
    @test_broken ref_b
    @test_broken ref_g

    # alias_guard_report: a toy module with one unguarded and one guarded mutating method.
    @eval module AliasToy
    unguarded!(y::AbstractVector, x::AbstractVector) = (y .= x; y)
    guarded!(y::AbstractVector, x::AbstractVector) = (Base.mightalias(y, x) && throw(ArgumentError("alias")); y .= x; y)
    end
    toy = StrictMode.alias_guard_report(AliasToy)
    @test any(f -> Symbol(f.func) == :unguarded!, toy)
    @test !any(f -> Symbol(f.func) == :guarded!, toy)
    # PureBLAS: none of the driver entries asks whether its operands overlap.
    found = StrictMode.alias_guard_report(P; only = r"^(gemm|syrk|ger|axpy|trmm|trsm|gemv)!$")
    flagged = Set(Symbol(f.func) for f in found)
    @test flagged == Set([:gemm!, :syrk!, :ger!, :axpy!, :trmm!, :trsm!, :gemv!])
end

# ── Untrusted / trust_boundary! / @assert_trusted ────────────────────────────────────────────────────
@testitem "strict safety: trust boundary and the C-ABI entries" tags = [:checks] setup = [SafetyHelp] begin
    using StrictMode, StrictModeTest
    import PureBLAS as P
    using .SafetyHelp: flags
    @test StrictMode.checks_enabled()

    struct Hdr
        n::Int
    end
    parse_h(u::StrictMode.Untrusted{Vector{UInt8}}) = (b = StrictMode.unsafe_trust(u); length(b) >= 1 || throw(ArgumentError("short")); Hdr(Int(b[1])))
    peek(u::StrictMode.Untrusted{Vector{UInt8}}) = StrictMode.unsafe_trust(u)[1]
    StrictMode.trust_boundary!(parse_h)
    u = StrictMode.Untrusted(UInt8[3, 4])
    # Control: reading the payload outside a registered boundary is flagged; inside one it is accepted.
    @test flags(() -> StrictMode.@assert_trusted peek(u))
    @test !flags(() -> StrictMode.@assert_trusted parse_h(u))

    # The C-ABI entries take `Ptr` arguments and dereference them with `unsafe_load`; none takes an
    # `Untrusted`, so `@assert_trusted` accepts every one of them without reading a payload.
    entries = [getfield(P, n) for n in names(P; all = true) if endswith(string(n), "_64_") && isdefined(P, n) && getfield(P, n) isa Function]
    @test length(entries) > 150
    takes_untrusted(f) = any(m -> any(t -> t <: StrictMode.Untrusted, Base.unwrap_unionall(m.sig).parameters[2:end]), methods(f))
    @test count(takes_untrusted, entries) == 0
    x = randn(8); a = Ref(2.0); nn = Ref(8); inc = Ref(1)
    GC.@preserve x a nn inc begin
        @test !flags(
            () -> StrictMode.@assert_trusted P.dscal_64_(
                Base.unsafe_convert(Ptr{Int64}, nn), Base.unsafe_convert(Ptr{Float64}, a),
                pointer(x), Base.unsafe_convert(Ptr{Int64}, inc)
            )
        )
    end
end

# ── Unreleased StrictMode (0.5.11): spin waits, reach, route parity ──────────────────────────────────
@testitem "strict safety: pool wait loops, reach and route parity" tags = [:checks, :strictmode_unreleased] setup = [SafetyHelp] begin
    using StrictMode, StrictModeTest
    import PureBLAS as P
    using .SafetyHelp: THREADED_ONLY
    if !isdefined(StrictMode, :spin_wait_report)
        @test_skip isdefined(StrictMode, :spin_wait_report)
    else
        GP = P.GemmPool{Float64}
        # Controls: a migratable task that polls and yields without blocking is reported; one that
        # blocks is not; the pool barrier, which spins and yields, is reported when run in a migratable task.
        spinner() = Threads.@spawn(begin; a = Threads.Atomic{Int}(0); while a[] == 0; yield(); end; end)
        waiter() = Threads.@spawn(begin; c = Channel{Int}(1); take!(c); end)
        bad_barrier(p) = Threads.@spawn P._pool_barrier(p, 2)
        @test !isempty(StrictMode.spin_wait_report(spinner, ()).findings)
        @test isempty(StrictMode.spin_wait_report(waiter, ()).findings)
        @test !isempty(StrictMode.spin_wait_report(bad_barrier, (GP,)).findings)
        # The pool: its workers are bound to a thread (`jl_set_task_tid`), which the check counts and does
        # not judge. A worker spawned unbound parks on an event after its spin and is not reported.
        rs = StrictMode.spin_wait_report(P._gemm_pool_start!, (GP,))
        @test isempty(rs.findings) && isempty(rs.unread)
        @test rs.bound == 1
        bad_worker(p) = Threads.@spawn P._gemm_pool_worker(p, 1)
        @test isempty(StrictMode.spin_wait_report(bad_worker, (GP,)).findings)

        # Which drivers statically reach which threaded entry. `stop` keeps the pool's dispatcher from
        # pulling every job kind into every graph.
        V = Vector{Float64}; M = Matrix{Float64}
        gemm3!(C, A, B) = P.gemm!(C, A, B)
        syrk3!(C, A) = P.syrk!(C, A)
        trmm3!(B, A) = P.trmm!(B, A)
        trsm3!(B, A) = P.trsm!(B, A)
        trmv3!(A, x) = P.trmv!(A, x)
        gemv3!(y, A, x) = P.gemv!(y, A, x)
        axpy3!(y, a, x) = P.axpy!(y, a, x)
        dot3(x, y) = P.dot(x, y)
        asum3(x) = P.asum(x)
        ger3!(A, x, y) = P.ger!(1.0, x, y, A)
        gbmv3!(y, AB, x) = P.gbmv!(y, AB, x, 8, 2, 2)
        serial_core(C, A, B) = P._gemm_core!(C, A, B, 1.0, 0.0, false, false, false, false, (-1, -1, -1))
        reach(f, T) = StrictMode.reach_report(f, T; mod = P, stop = (P._gemm_threaded!,))
        reached(r) = Set(n for n in THREADED_ONLY if StrictMode.reaches(r, getfield(P, n)))
        table = (
            (gemm3!, (M, M, M), Set([:_gemm_threaded!])),
            (syrk3!, (M, M), Set([:_gemm_threaded!])),
            (trmm3!, (M, M), Set([:_gemm_threaded!, :_trmvt_threaded!, :_trmvn_threaded!])),
            (trsm3!, (M, M), Set([:_gemm_threaded!])),
            (trmv3!, (M, V), Set([:_gemm_threaded!, :_trmvt_threaded!, :_trmvn_threaded!])),
            (gemv3!, (V, M, V), Set([:_gemm_threaded!])),
            (axpy3!, (V, Float64, V), Set([:_gemm_threaded!])),
            (dot3, (V, V), Set([:_gemm_threaded!])),
            (asum3, (V,), Set([:_gemm_threaded!])),
            (ger3!, (M, V, V), Set([:_gemm_threaded!])),
            (gbmv3!, (V, M, V), Set([:_gemm_threaded!, :_gbmvn_threaded!])),
        )
        for (f, T, want) in table
            r = reach(f, T)
            @test r.unread == 0 && r.dynamic == 0
            @test reached(r) == want
        end
        # Control: the serial core reaches none of them.
        @test isempty(reached(reach(serial_core, (M, M, M))))
        C = zeros(8, 8); A = randn(8, 8); B = randn(8, 8)
        @eval StrictMode.@assert_reaches gemm3!($C, $A, $B) P._gemm_threaded!
        @test SafetyHelp.flags(() -> @eval StrictMode.@assert_reaches serial_core($C, $A, $B) P._gemm_threaded!)

        # The native entry and the C-ABI entry take one route; they differ in the native entry's argument
        # checks and the C-ABI's pointer bridge.
        CT = (Ptr{UInt8}, Ptr{UInt8}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int64}, Ptr{Float64}, Ptr{Int64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int64}, Clong, Clong)
        p = StrictMode.route_parity(gemm3! => (M, M, M), P.dgemm_64_ => CT; mod = P, stop = (P._gemm_threaded!,))
        @test StrictMode.status(p) != :incomplete
        # Control: the threaded public entry and the serial core do not agree.
        q = StrictMode.route_parity(gemm3! => (M, M, M), serial_core => (M, M, M); mod = P, stop = (P._gemm_threaded!,))
        @test StrictMode.status(q) == :differ
        @test !isempty(q.only_a)
    end
end

# ── Unreleased StrictMode: bit identity across thread counts ─────────────────────────────────────────
@testitem "strict safety: results are bit-identical at 1 to 4 threads" tags = [:checks, :strictmode_unreleased] begin
    using StrictMode, StrictModeTest
    if !isdefined(StrictMode, :threadcount_invariant_report)
        @test_skip isdefined(StrictMode, :threadcount_invariant_report)
    else
        # The pool counts its jobs in the high bits of its generation word. Starting the pool already runs
        # calibration jobs, so the witness compares the count with its value after setup.
        pre = "using LinearAlgebra; ENV[\"PUREBLAS_MT_ALL_THREADS\"]=\"1\"; PureBLAS.set_num_threads(Threads.nthreads()); using Random; Random.seed!(1); " *
            "__gen() = isempty(PureBLAS._GEMM_POOL_F64) ? 0 : (getfield(PureBLAS._GEMM_POOL_F64[1], :gen, :acquire) >> PureBLAS._MT_NW_SHIFT); __g0 = __gen(); "
        wit = "__gen() > __g0"
        ops = (
            ("gemm", "A=randn(1024,1024); B=randn(1024,1024)", "(C=zeros(1024,1024); PureBLAS.gemm!(C,A,B); C)"),
            ("syrk", "A=randn(1024,1024)", "(C=zeros(1024,1024); PureBLAS.syrk!(C,A); C)"),
            ("trmm", "A=randn(1024,1024)+1024I; B=randn(1024,1024)", "(X=copy(B); PureBLAS.trmm!(X,A); X)"),
            ("trsm", "A=randn(1024,1024)+1024I; B=randn(1024,1024)", "(X=copy(B); PureBLAS.trsm!(X,A); X)"),
            ("trmv", "A=randn(4096,4096)+4096I; x0=randn(4096)", "(x=copy(x0); PureBLAS.trmv!(A,x); x)"),
            ("gemv", "A=randn(8192,8192); x0=randn(8192)", "(y=zeros(8192); PureBLAS.gemv!(y,A,x0); y)"),
            ("axpy", "x0=randn(2^23); y0=randn(2^23)", "(y=copy(y0); PureBLAS.axpy!(y,2.0,x0); y)"),
            ("dot", "x0=randn(2^23); y0=randn(2^23)", "PureBLAS.dot(x0,y0)"),
            ("asum", "x0=randn(2^23)", "PureBLAS.asum(x0)"),
        )
        for (nm, setup, call) in ops
            r = StrictModeTest.test_threadcount_invariant(:PureBLAS, call; setup = pre * setup, witness = wit, threads = 1:4)
            @test r.nthreads == [1, 2, 3, 4]
            @test StrictMode.invariant(r)
            @test StrictMode.witnessed(r)
        end
        # Controls: a result that depends on the thread count is reported as differing, and a call that
        # stays serial is reported as not witnessed.
        rd = StrictMode.threadcount_invariant_report(
            :PureBLAS, "fill(Float64(Threads.nthreads()), 4)"; setup = pre, witness = wit, threads = 1:3
        )
        @test !StrictMode.invariant(rd)
        rs = StrictMode.threadcount_invariant_report(
            :PureBLAS, "(C=zeros(8,8); PureBLAS.gemm!(C,A,B); C)";
            setup = pre * "A=randn(8,8); B=randn(8,8)", witness = wit, threads = 1:3
        )
        @test StrictMode.invariant(rs)
        @test !StrictMode.witnessed(rs)
        # The gating form refuses the unwitnessed call.
        @test_throws StrictMode.StrictViolation StrictModeTest.test_threadcount_invariant(
            :PureBLAS, "(C=zeros(8,8); PureBLAS.gemm!(C,A,B); C)";
            setup = pre * "A=randn(8,8); B=randn(8,8)", witness = wit, threads = 1:3
        )
    end
end

# ── Unreleased StrictMode: threaded speedup on 4 CPUs ────────────────────────────────────────────────
@testitem "strict safety: threaded gemm speedup on four CPUs" tags = [:checks, :strictmode_unreleased] begin
    using StrictMode, StrictModeTest
    if !isdefined(StrictMode, :threaded_speedup_report)
        @test_skip isdefined(StrictMode, :threaded_speedup_report)
    else
        setup = "ENV[\"PUREBLAS_MT_ALL_THREADS\"]=\"1\"; using Random; Random.seed!(1); A = randn(1024,1024); B = randn(1024,1024); C = zeros(1024,1024); "
        r = StrictMode.threaded_speedup_report(
            :PureBLAS, "PureBLAS.gemm!(C, A, B)"; setup = setup * "PureBLAS.set_num_threads(Threads.nthreads())",
            cpus = 4, threads = 4, workers = "PureBLAS.get_num_threads()", repeats = 3
        )
        @test r.workers >= 1
        # A threaded call that is slower than its own serial path by half is a defect on any host.
        @test r.threaded_ms <= 1.5 * r.serial_ms
        if r.pinned && Sys.EFFECTIVE_CPU_THREADS >= 4
            @test r.threaded_ms <= r.serial_ms / (0.4 * r.workers)
        end
        # Control: the same call with threading off, credited with four workers, misses the efficiency bound.
        c = StrictMode.threaded_speedup_report(
            :PureBLAS, "PureBLAS.gemm!(C, A, B)"; setup = setup * "PureBLAS.set_num_threads(1)",
            cpus = 4, threads = 4, workers = 4, repeats = 3
        )
        if c.pinned
            @test c.threaded_ms > c.serial_ms / (0.4 * c.workers)
        end
    end
end
