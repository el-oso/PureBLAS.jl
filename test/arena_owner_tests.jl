# `_getri!` picks its arena owner once, at the root, and runs a body specialized on it: `ThreadArena`
# (the per-thread arena) only where every callee is a serial entry, `TaskArena` (the per-task arena)
# wherever a callee may thread. These items check the choice, the structure of the `ThreadArena`
# body, the run-time invariant behind it, bitwise invariance across thread counts and allocation.
#
# CI sets `JULIA_NUM_THREADS`; with one thread the threaded halves report a skip.

@testmodule SuspensionReach begin
    # Static reachability of task-switch points from one signature, over the OPTIMIZED call graph
    # (`:invoke` edges after inlining and constant folding).
    const SUSPEND = (:yield, :wait, :yieldto, :try_yieldto, :sleep, :_gemm_threaded!, :_pool_barrier, :wait_forever)

    _mi(x) = x isa Core.MethodInstance ? x :
        x isa Core.CodeInstance ? (d = x.def; d isa Core.MethodInstance ? d : d.def) : nothing

    # Every chain of method names from `tt` to a callee in `SUSPEND`.
    function reach(tt::Type{<:Tuple})
        seen = Set{Any}(); hits = Vector{Vector{Symbol}}()
        stack = Any[(tt, Symbol[])]
        while !isempty(stack)
            sig, path = pop!(stack)
            sig in seen && continue
            push!(seen, sig)
            cis = try
                Base.code_typed_by_type(sig; optimize = true)
            catch
                continue
            end
            for (ci, _) in cis
                ci isa Core.CodeInfo || continue
                for st in ci.code
                    (st isa Expr && st.head === :invoke) || continue
                    mi = _mi(st.args[1]); mi === nothing && continue
                    nm = mi.def isa Method ? mi.def.name : :unknown
                    p = vcat(path, nm)
                    nm in SUSPEND ? push!(hits, p) : push!(stack, (mi.specTypes, p))
                end
            end
        end
        return hits
    end
    export reach
end

@testitem "getri!: arena owner chosen at the root" tags = [:checks] begin
    using PureBLAS
    const P = PureBLAS
    P.set_num_threads(1)
    @test P._thread_arena_ok(Float64)
    @test P._thread_arena_ok(ComplexF32)
    # Any other element type gets the per-task arena: a dual reaches thread-capable plane products.
    @test !P._thread_arena_ok(Rational{Int})
    @test !P._thread_arena_ok(BigFloat)
    if Threads.nthreads() >= 2
        P.set_num_threads(2)
        @test !P._thread_arena_ok(Float64)
        P.set_num_threads(1)
    end
    # Both owners hand out the same arena type, so a borrow is the same handle either way.
    @test P.ThreadArena(P._arena()).a isa P.Arena
    @test P.TaskArena(P._tarena()).a isa P.Arena
    # The per-task arena belongs to the task: another task gets another one.
    a1 = P._tarena(); a2 = fetch(Threads.@spawn P._tarena())
    @test a1 === P._tarena() && a1 !== a2
end

# The `ThreadArena` body must not call a thread-capable entry. Its callees are the serial entries
# (`_gemm_entry!`/`_trsm_entry!` with `Val(false)`), so no public `gemm!`/`trsm!`/`trmm!` body and no
# direct `_gemm_threaded!` is reachable from its own frame. What remains reachable lies below
# `_gemm_core!`'s `nw`-gated cooperative and Strassen paths and the BLAS-1/2 kernels that size their
# own worker count from `_MT_NTHREADS`; that set is listed here exactly, so a new path into a
# task switch fails this item.
@testitem "getri!: ThreadArena body reaches no thread-capable entry" tags = [:checks] setup = [SuspensionReach] begin
    using PureBLAS
    const P = PureBLAS
    # The frame that calls each task-switch point, for every path not through a public entry.
    residual = Set(
        [
            :_blocked_pc_sweep!,          # cooperative A-pack barrier, taken only when nw > 1
            :_strassen_rec!, :_strassen_leaf!, :_comb!,   # Strassen leaf/combine split, only when nw > 1
            :_gemvn_threaded!, :_gemvt_threaded!,         # gemv, worker count from _MT_NTHREADS
            :_scal_threaded!,             # scal, worker count from _MT_NTHREADS
            :slowlock, :wait_no_relock,   # first-touch lock of a OncePerProcess owner
        ]
    )
    public_bodies = r"^#(gemm|trsm|trmm|syrk|herk|symm)!#"
    for T in (Float64, Float32, ComplexF64, ComplexF32)
        tt = Tuple{typeof(P._getri_blocked!), Matrix{T}, Vector{Int}, Int, P.ThreadArena}
        hits = reach(tt)
        through_public = filter(p -> any(s -> occursin(public_bodies, string(s)), p), hits)
        @test isempty(through_public)
        direct = filter(p -> length(p) <= 2, hits)
        @test isempty(direct)
        callers = Set(p[end - 1] for p in hits if length(p) >= 2)
        @test issubset(callers, residual)
        # POSITIVE CONTROL: the TaskArena body calls the pool from its own frame, and the walk sees it.
        tq = Tuple{typeof(P._getri_blocked!), Matrix{T}, Vector{Int}, Int, P.TaskArena}
        @test any(p -> p == [:_gemm_threaded!], reach(tq))
    end
end

# `_ARENA_CHECK` asserts, at a claim win in `_gemm_threaded!` and at every `_pool_barrier`, that the
# per-thread arena on that thread holds no live scope. Off in production (the branch compiles away);
# this item redefines it, which invalidates and recompiles the callers.
@testitem "getri!: no per-thread scope is live at claim-win or barrier (_ARENA_CHECK)" tags = [:checks] begin
    using PureBLAS, Random
    const P = PureBLAS
    if Threads.nthreads() < 2
        @test_skip "needs >= 2 julia threads"
    else
        @eval P const _ARENA_CHECK = true
        try
            P.set_num_threads(min(4, Threads.nthreads()))
            for T in (Float64, Float32, ComplexF64)
                n = 1024
                A = randn(MersenneTwister(1), T, n, n); ip = zeros(Int, n); P.getrf!(A, ip)
                g0 = (@atomic P._gemm_pool(T).gen)
                W = copy(A)
                @test P.getri!(W, ip) === W
                @test (@atomic P._gemm_pool(T).gen) != g0      # witness: the pool ran
            end
            # POSITIVE CONTROL: a threaded gemm inside a live per-thread scope trips the check.
            n = 1024; C = zeros(n, n); A = randn(n, n); B = randn(n, n)
            f() = (
                P.@scope s begin
                    w = P.borrow!(s, Float64, 1)
                    P.gemm!(C, A, B)
                    nothing
                end
            )
            @test_throws "per-thread arena scope is live" f()
        finally
            @eval P const _ARENA_CHECK = false
            P.set_num_threads(1)
        end
    end
end

@testitem "getri!: bit-identical at every thread count" tags = [:checks] begin
    using PureBLAS, Random
    const P = PureBLAS
    nt = min(4, Threads.nthreads())
    if nt < 2
        @test_skip "needs >= 2 julia threads"
    else
        bits(X) = reinterpret(UInt8, vec(X))
        for T in (Float64, Float32, ComplexF64, ComplexF32), n in (8, 33, 200, 513, 1024)
            A = randn(MersenneTwister(n), T, n, n); ip = zeros(Int, n); P.getrf!(A, ip)
            got = Dict{Int, Vector{UInt8}}()
            g0 = (@atomic P._gemm_pool(T).gen)
            for t in (1, 2, 4)
                t <= nt || continue
                P.set_num_threads(t)
                W = copy(A); P.getri!(W, ip); got[t] = copy(bits(W))
            end
            n == 1024 && @test (@atomic P._gemm_pool(T).gen) != g0   # witness: some thread count threaded
            @test all(v -> v == got[1], values(got))
        end
        P.set_num_threads(1)
    end
end

@testitem "getri!: warm calls allocate nothing, serial and threaded" tags = [:checks] begin
    using PureBLAS, Random
    const P = PureBLAS
    run!(W, A, ip) = (copyto!(W, A); P.getri!(W, ip); W)
    alloc(W, A, ip) = @allocated run!(W, A, ip)
    for t in unique((1, min(4, Threads.nthreads())))
        P.set_num_threads(t)
        for T in (Float64, Float32, ComplexF64), n in (8, 64, 1024)
            A = randn(MersenneTwister(1), T, n, n); ip = zeros(Int, n); P.getrf!(A, ip); W = similar(A)
            run!(W, A, ip); run!(W, A, ip)
            @test minimum(alloc(W, A, ip) for _ in 1:5) == 0
        end
    end
    P.set_num_threads(1)
end
