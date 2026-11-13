@testmodule GemmOracle begin
using LinearAlgebra
export gerr, gtol
gerr(a, b) = norm(a .- b) / max(norm(b), eps(Float64))
gtol(::Type{T}) where {T} = T <: Union{Float32, ComplexF32} ? 1.0e-3 : 1.0e-11
end

@testitem "GEMM real (blocked) vs OpenBLAS" setup = [GemmOracle] begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    @testset "$T $tA$tB m=$m n=$n k=$k" for T in (Float32, Float64),
            (tA, tB) in (('N', 'N'), ('T', 'N'), ('N', 'T'), ('T', 'T')),
            m in (1, 16, 17, 40), n in (1, 6, 7, 31), k in (1, 16, 33)

        A = tA == 'N' ? randn(T, m, k) : randn(T, k, m)
        Bm = tB == 'N' ? randn(T, k, n) : randn(T, n, k)
        C0 = randn(T, m, n)
        for (al, be) in ((one(T), zero(T)), (T(0.5), T(2)), (zero(T), T(1.5)))
            Cref = copy(C0); B.gemm!(tA, tB, al, A, Bm, be, Cref)
            Cp = copy(C0); PureBLAS.gemm!(Cp, A, Bm; alpha = al, beta = be, transA = tA, transB = tB)
            @test gerr(Cp, Cref) < gtol(T)
        end
    end
end

@testitem "GEMM blocked path (>unpack threshold) vs OpenBLAS" setup = [GemmOracle] begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    # Sizes above the unpacked threshold (96) exercise the blocked path incl. its masked edge.
    @testset "$T $tA$tB m=$m n=$n k=$k" for T in (Float32, Float64),
            (tA, tB) in (('N', 'N'), ('T', 'N'), ('N', 'T')),
            (m, n, k) in ((100, 100, 100), (130, 97, 113), (160, 200, 128), (97, 150, 99))

        A = tA == 'N' ? randn(T, m, k) : randn(T, k, m)
        Bm = tB == 'N' ? randn(T, k, n) : randn(T, n, k)
        C0 = randn(T, m, n)
        for (al, be) in ((one(T), zero(T)), (T(0.7), T(1.3)))
            Cref = copy(C0); B.gemm!(tA, tB, al, A, Bm, be, Cref)
            Cp = copy(C0); PureBLAS.gemm!(Cp, A, Bm; alpha = al, beta = be, transA = tA, transB = tB)
            @test gerr(Cp, Cref) < gtol(T)
        end
    end
end

# A one-row or one-column C is computed by `_gemv!`, not by packing the other operand. Bit equality
# with the gemv is the witness: the gemm kernels sum k in a different order. Checked through
# `activate()` as well, which reaches the same route from the `dgemm_64_` entry.
@testitem "GEMM with one row or column of C is a gemv" setup = [GemmOracle] begin
    using PureBLAS, LinearAlgebra
    const P = PureBLAS
    k, w = 988, 553
    for T in (Float32, Float64), tA in ('N', 'T'), tB in ('N', 'T')
        # m == 1: C (1×w) = op(A) (1×k) · op(B) (k×w); the gemv runs over B.
        A = tA == 'N' ? randn(T, 1, k) : randn(T, k, 1)
        Bm = tB == 'N' ? randn(T, k, w) : randn(T, w, k)
        C0 = randn(T, 1, w); α = T(0.7); β = T(1.3)
        y = vec(copy(C0)); P._gemv!(tB == 'N', false, size(Bm)..., α, Bm, vec(A), 1, β, y, 1)
        C = copy(C0); P.gemm!(C, A, Bm; alpha = α, beta = β, transA = tA, transB = tB)
        @test vec(C) == y
        # n == 1: C (w×1) = op(A) (w×k) · op(B) (k×1); the gemv runs over A.
        A = tA == 'N' ? randn(T, w, k) : randn(T, k, w)
        Bm = tB == 'N' ? randn(T, k, 1) : randn(T, 1, k)
        C0 = randn(T, w, 1)
        y = vec(copy(C0)); P._gemv!(tA != 'N', false, size(A)..., α, A, vec(Bm), 1, β, y, 1)
        C = copy(C0); P.gemm!(C, A, Bm; alpha = α, beta = β, transA = tA, transB = tB)
        @test vec(C) == y
    end
    x = randn(1, k); A = randn(k, w); C = zeros(1, w)
    y = zeros(w); P._gemv!(true, false, k, w, 1.0, A, vec(x), 1, 0.0, y, 1)
    P.activate()
    try
        mul!(C, x, A)
        @test vec(C) == y
    finally
        P.deactivate()
    end
end

@testitem "GEMM beta=0 ignores NaN in C (BLAS semantics)" setup = [GemmOracle] begin
    using PureBLAS
    A = randn(64, 48); B = randn(48, 32)
    C = fill(NaN, 64, 32)
    PureBLAS.gemm!(C, A, B; alpha = 1.0, beta = 0.0)
    @test all(isfinite, C) && gerr(C, A * B) < gtol(Float64)
end

@testitem "GEMM complex (SIMD split-pack + unpacked) vs OpenBLAS" setup = [GemmOracle] begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    # sizes span the routing: tiny→generic (5), unpacked small-n (23,40 — >nr cols & >mr rows to
    # exercise the jr/ir tiling), and >_CGEMM_UNPACK_MAX→blocked (100). n>nr with jr>0 was a real bug.
    @testset "$T $tA$tB $m×$n×$k" for T in (ComplexF32, ComplexF64),
            (tA, tB) in (('N', 'N'), ('C', 'N'), ('N', 'C'), ('T', 'T'), ('C', 'C')),
            (m, n, k) in ((23, 17, 19), (5, 5, 5), (40, 48, 33), (100, 96, 77))

        A = tA == 'N' ? randn(T, m, k) : randn(T, k, m)
        Bm = tB == 'N' ? randn(T, k, n) : randn(T, n, k)
        C0 = randn(T, m, n); al = T(0.7, -0.3); be = T(1.4, 0.2)
        Cref = copy(C0); B.gemm!(tA, tB, al, A, Bm, be, Cref)
        Cp = copy(C0); PureBLAS.gemm!(Cp, A, Bm; alpha = al, beta = be, transA = tA, transB = tB)
        @test gerr(Cp, Cref) < gtol(T)
    end
end

@testitem "GEMM allocating gemm(A,B) == A*B" setup = [GemmOracle] begin
    using PureBLAS
    A = randn(50, 30); B = randn(30, 40)
    @test gerr(PureBLAS.gemm(A, B), A * B) < gtol(Float64)
    @test gerr(PureBLAS.gemm(A, A; transA = 'T'), A' * A) < gtol(Float64)
end

@testitem "GEMM is AD-traceable (generic path, ForwardDiff)" begin
    using PureBLAS, ForwardDiff, LinearAlgebra
    A = randn(8, 5); dA = randn(8, 5); B = randn(5, 6)
    # d/dt sum((A + t·dA)·B) = sum(dA·B)
    f(t) = sum(PureBLAS.gemm(A .+ t .* dA, B))
    @test ForwardDiff.derivative(f, 0.0) ≈ sum(dA * B)
end

@testitem "GEMM steady-state is allocation-free (driver level)" begin
    using PureBLAS
    # Measured through a function barrier so the operands arrive with concrete types, as they do
    # from LinearAlgebra and LBT. At module scope the call is `Any`-typed, and the first compile of
    # each such call site caches a method instance — dispatch state that scales with the number of
    # call sites, not with the number of calls, and that no warmup removes. The SME path is the only
    # one that shows it, because its portability barrier is a function pointer (one dynamic dispatch);
    # measuring it as steady-state allocation reports a cost no caller pays.
    steady(f, args...; kw...) = (f(args...; kw...); @allocated f(args...; kw...))
    # unpacked (small, max dim ≤ 96) path — no buffers at all
    A = randn(48, 48); B = randn(48, 48); C = zeros(48, 48)
    @test steady(PureBLAS.gemm!, C, A, B; alpha = 1.0, beta = 0.0) == 0
    @test steady(PureBLAS.gemm!, C, A, B; alpha = 2.0, beta = 1.0) == 0  # beta≠0 branch
    # blocked (large) path — scratch allocated on first call, then reused → 0 thereafter
    Al = randn(300, 300); Bl = randn(300, 300); Cl = zeros(300, 300)
    @test steady(PureBLAS.gemm!, Cl, Al, Bl; beta = 0.0) == 0
    # COMPLEX, and specifically INSIDE the Karatsuba-3M window (_CGEMM_3M_MIN=48 ≤ max(m,n,k) ≤ 2048,
    # min ≥ _CGEMM_3M_KMIN=16). This case had NO allocation coverage at any size, which is how
    # `_gemm_3m!` shipped building nine `unsafe_wrap(Array, …)` headers per call — ~1 KB of steady-state
    # allocation on every 3M gemm, live on AVX2 from the day 3M landed and invisible because the only
    # allocation test here was real Float64. Fixed by taking `PtrMatrix` views over the pooled buffers
    # (isbits ⇒ no header). n=128 sits in the window on every µarch; n=32 is the below-window control,
    # so the pair also pins the routing, not just the total.
    for nc in (32, 128)
        Az = randn(ComplexF64, nc, nc); Bz = randn(ComplexF64, nc, nc); Cz = zeros(ComplexF64, nc, nc)
        @test steady(
            PureBLAS.gemm!, Cz, Az, Bz;
            alpha = one(ComplexF64), beta = zero(ComplexF64)
        ) == 0
    end
    # complex rank-k rides the same buffers through `_ctrgemm_3m!` (n ≥ _CSYRK_3M_MIN)
    As = randn(ComplexF64, 300, 300); Cs = zeros(ComplexF64, 300, 300)
    @test steady(PureBLAS.syrk!, Cs, As; uplo = 'U', trans = 'N', alpha = true, beta = false) == 0
    @test steady(PureBLAS.herk!, Cs, As; uplo = 'U', trans = 'N', alpha = 1.0, beta = 0.0) == 0
end

@testitem "GEMM dimension mismatch is caught" begin
    using PureBLAS
    @test_throws DimensionMismatch PureBLAS.gemm!(zeros(3, 3), zeros(3, 4), zeros(5, 3))
end

# LIVENESS GATE FOR EVERY THREADING ITEM BELOW.
#
# Each of them takes a skip branch under two threads, and a skipped guarantee is indistinguishable in
# the summary from a passing one — so a single-threaded run reports green having started no worker and
# verified no bit-identity. This item fails on exactly that process, which turns a silent gap into a
# named one.
#
# The fix when it fires is to give the process threads, never to weaken the gate:
#     JULIA_NUM_THREADS=4 julia --project=. -e 'using Pkg; Pkg.test()'
# CI does this per job; the threading items are `:checks`-tagged, so it is the `checks` job that needs
# it and not only `main`, whose filter excludes them.
@testitem "threading guarantees actually ran" tags = [:checks] begin
    using Base.Threads: nthreads
    threading_items_can_run = nthreads() >= 2
    @test threading_items_can_run
end

@testitem "fast-core count matches the sysctl command" begin
    using PureBLAS
    P = PureBLAS
    # Independent of the library call: the `sysctl` command, and on any other OS the documented 0.
    if Sys.isapple() && success(`sysctl -n hw.nperflevels`)
        q(k) = parse(Int, readchomp(`sysctl -n $k`))
        nl = q("hw.nperflevels")
        @test P._FAST_CORES == (nl > 1 ? sum(l -> q("hw.perflevel$l.physicalcpu"), 0:(nl - 2)) : 0)
    else
        @test P._FAST_CORES == 0
    end
end

@testitem "threaded gemm: same answer as the serial path, and still 0 B" tags = [:checks] begin
    using PureBLAS, Base.Threads
    P = PureBLAS
    # The split is by COLUMNS of C, so the shapes that matter are the ones where a chunk boundary can
    # land badly: n not a multiple of the register tile, n smaller than the worker count, and both
    # transpose flags — with `transB` a chunk of op(B) is a ROW slice of B, a different pointer walk
    # from the column slice every other case uses.
    # Threading is opt-in, so the default must be OFF even on a threaded worker — that is the check
    # that a host which never asks for threads never gets them.
    @test P.get_num_threads() == 1
    @test P._gemm_workers(512, 512, 512) == 1
    if Threads.nthreads() < 2
        @info "single-threaded worker — threading cannot engage; checking it stays off"
        @test P.set_num_threads(4) == 1
    else
        @test P.set_num_threads(Threads.nthreads()) == Threads.nthreads()
        # The core cap: without the suite's override, no more workers than physical cores.
        withenv("PUREBLAS_MT_ALL_THREADS" => nothing) do
            @test P.set_num_threads(Threads.nthreads()) == min(Threads.nthreads(), P._mt_core_cap())
        end
        @test 1 <= P._mt_physical_cores() <= Sys.EFFECTIVE_CPU_THREADS
        @test 1 <= P._mt_core_cap() <= P._mt_physical_cores()
        # Placement: every worker bound to its own default-pool thread.
        tw = P._gemm_pool(Float64).tids
        @test length(tw) == Threads.nthreads() - 1
        @test allunique(tw) && all(in(Threads.threadpooltids(:default)), tw)
        # A host that pins its threads one per core leaves THIS thread one core wide; the count is fixed
        # for the process, so it must not drop. Pin this thread to one core (Linux, where the test can),
        # then restore.
        if Sys.islinux()
            c0 = P._mt_physical_cores()
            if c0 > 1
                old = zeros(UInt64, 16)
                ccall(:sched_getaffinity, Cint, (Cint, Csize_t, Ptr{UInt64}), 0, sizeof(old), old)
                cpu = ccall(:sched_getcpu, Cint, ())
                one = zeros(UInt64, 16); one[cpu ÷ 64 + 1] = UInt64(1) << (cpu % 64)
                try
                    @test iszero(ccall(:sched_setaffinity, Cint, (Cint, Csize_t, Ptr{UInt64}), 0, sizeof(one), one))
                    @test P._mt_physical_cores() == c0
                finally
                    ccall(:sched_setaffinity, Cint, (Cint, Csize_t, Ptr{UInt64}), 0, sizeof(old), old)
                end
            end
        end
        P.set_num_threads(Threads.nthreads())
        for (m, n, k) in ((256, 256, 256), (512, 300, 128), (200, 512, 333), (1024, 129, 64), (192, 7, 192))
            for tA in (false, true), tB in (false, true)
                A = tA ? randn(k, m) : randn(m, k)
                B = tB ? randn(n, k) : randn(k, n)
                C0 = randn(m, n)
                al, be = 1.7, -0.3
                Cs = copy(C0)
                P._gemm_core!(Cs, A, B, al, be, tA, tB, false, false)
                Ct = copy(C0)
                P.gemm!(
                    Ct, A, B; alpha = al, beta = be,
                    transA = tA ? 'T' : 'N', transB = tB ? 'T' : 'N'
                )
                @test Ct ≈ Cs rtol = 1.0e-12
            end
        end
        # req#10 on the threaded path. THIS MEASURES A PROCESS, NOT A CALL, and the assertion is
        # shaped around that: `@allocated` reads `jl_gc_get_total_bytes`, which sums EVERY thread.
        # Measured on x86 with a 12.6 ms non-allocating function and one allocating task alongside
        # it: 1.49 MB charged to a function that allocates nothing. So a single window reading
        # non-zero says "something in this process allocated", not "gemm! allocated" — and in a test
        # process with a runner and other items' leftovers in it, that is not gemm!'s to answer for.
        #
        # THE MINIMUM OVER SEVERAL WINDOWS IS THE HONEST STATISTIC, and it is not a weakening: a
        # per-call leak allocates on EVERY call, so no window can come back clean, while a foreign
        # allocation lands in some windows and not others. (This is the one place a minimum is right;
        # timing still reduces through the median, because there the contamination is not one-sided.)
        #
        # `@allocated` also sits INSIDE a compiled barrier taking three named arguments: at module
        # scope the call is interpreted over `Any`-typed globals and the window then measures the
        # interpreter's own method lookup, and a varargs barrier allocates through the splat.
        #
        # The warm-up covers both paths the measured call can take. The route is one; the pool's
        # sleep/wake handoff is the other, and the shapes above miss it because they run back-to-back
        # and leave the workers inside their polling window rather than parked.
        gemm1(Cx, Ax, Bx) = @allocated P.gemm!(Cx, Ax, Bx)
        A = randn(512, 512); B = randn(512, 512); C = zeros(512, 512)
        @test P._gemm_workers(512, 512, 512) > 1
        P.gemm!(C, A, B)                 # the route
        sleep(0.05); P.gemm!(C, A, B)    # and a wake from a real park, not from the polling window
        @test minimum(gemm1(C, A, B) for _ in 1:8) == 0
        P.set_num_threads(1)                      # leave the process as we found it
        @test P.get_num_threads() == 1
    end
end

@testitem "threaded gemm: a concurrent caller falls back to serial, not to a shared pool" tags = [:checks] begin
    using PureBLAS, Base.Threads
    P = PureBLAS
    if Threads.nthreads() < 2
        @test_skip Threads.nthreads() >= 2
    else
        # Several callers hit `gemm!` at once. Exactly one may own the pool; the rest must run serially
        # and still get the right answer. A wrong claim here would show up as two drivers publishing
        # jobs into one pool, i.e. torn results — not as an error.
        nt = min(4, Threads.nthreads())
        P.set_num_threads(nt)
        # The smallest square `gemm!` lets the recursion claim (`_STR_FLOOR`: level 1 obeys the leaf
        # floor), so the combine passes below are exercised; two repetitions per caller because a
        # claimed product is 2048^3 on every box.
        ns = 2 * P._fh_strassen_base()
        A = randn(ns, ns); B = randn(ns, ns)
        want = similar(A); P._gemm_core!(want, A, B, 1.0, 0.0, false, false, false, false)
        outs = [zeros(ns, ns) for _ in 1:nt]
        Threads.@threads :static for t in 1:nt
            for _ in 1:2
                P.gemm!(outs[t], A, B)
            end
        end
        for t in 1:nt
            @test outs[t] ≈ want rtol = 1.0e-12
        end
        # AND WITH alpha AND beta BOTH NON-TRIVIAL, because the combine passes take a DIFFERENT arm
        # there. With alpha = 1, beta = 0 above, three of the four C-assembly passes take their
        # beta-free arm and the `beta * C` form is never exercised on the losing path at all. A
        # non-power-of-two alpha for the same reason it appears in the bit-identity item: scaling by
        # +-2^j is exact and would hide a misplaced alpha.
        #
        # Witness, and it is a PREMISE: `ns` is Strassen-claimed only where `_STRASSEN` holds, which is
        # `_W64 == 4 || _W64 == 8` -- false wherever `_vwidth(Float64)` is 2. There the call takes the
        # column split, so there is no claim to lose and the four C-assembly passes this paragraph is
        # about are not reached; the comparisons below still test the invariant, and only the witness
        # is skipped. Same shape as the big/thin witnesses further down.
        if P._STRASSEN
            @test P._strassen_owns(Float64, ns, ns, ns, false, A, B, P._STR_FLOOR)
        else
            @test_skip "Strassen is off at _W64=$(PureBLAS._W64)"
        end
        C0 = randn(ns, ns)
        want2 = copy(C0)
        P.set_num_threads(1)
        P.gemm!(want2, A, B; alpha = 3.7, beta = -1.25)
        P.set_num_threads(nt)
        outs2 = [copy(C0) for _ in 1:nt]
        Threads.@threads :static for t in 1:nt
            o = copy(C0)
            for _ in 1:2
                copyto!(o, C0)
                P.gemm!(o, A, B; alpha = 3.7, beta = -1.25)
            end
            copyto!(outs2[t], o)
        end
        for t in 1:nt
            @test outs2[t] ≈ want2 rtol = 1.0e-12
        end
        P.set_num_threads(1)
    end
end

# A DRIVER ON A WORKER'S THREAD. Any task but the pool's creator may run on a thread that hosts a
# participating worker — a `Threads.@spawn`ed caller, for one — and that worker could only run when the
# driver yields, after the driver's own chunk. `_gemm_threaded!` publishes such a job with that worker
# sitting out (`GemmPool.skip`). Both shapes that produces are exercised: at `nw == nthreads()` the pool
# has no spare worker and the job is one chunk narrower; at `nw == nthreads() - 1` the spare worker,
# asleep above the thread count, is woken to take the chunk and the count is unchanged.
#
# The witness comes first, as in every placement item: a driver that happened to land on a free thread
# would pass the bit comparison without the path ever running.
@testitem "threaded driver on a worker's thread: the worker sits out, bits unchanged, 0 B" tags = [:checks] begin
    using PureBLAS, LinearAlgebra, Random
    using Base.Threads: nthreads
    P = PureBLAS
    if nthreads() < 3
        @test_skip "needs ≥3 julia threads: a worker on a thread that is not the creator's, and a second worker"
    else
        # A sticky task bound to `tid` — the placement the pool uses for its own workers.
        function on_thread(f, tid)
            t = Task(f)
            t.sticky = true
            isone(ccall(:jl_set_task_tid, Cint, (Any, Cint), t, tid - 1)) || error("could not bind to thread $tid")
            schedule(t)
            return fetch(t)
        end
        bitsame(X, Y) = all(i -> reinterpret(UInt64, X[i]) === reinterpret(UInt64, Y[i]), eachindex(X))
        n, k = 1024, 256                        # B is packed at this width; k thin so Strassen declines
        al, be = 2.5, 0.5                        # α not a power of two, so its placement is visible
        A = randn(MersenneTwister(7), n, k); B = randn(MersenneTwister(8), n, k)
        C0 = randn(MersenneTwister(9), n, n)
        L0 = randn(MersenneTwister(10), n, n)
        nx = max(2 * nthreads() * P._red_block(Float64), cld(P._L1_MT_MIN, sizeof(Float64))) + 3
        x = randn(MersenneTwister(11), nx); y = randn(MersenneTwister(12), nx)
        # The serial answers.
        P.set_num_threads(1)
        Cs = copy(C0); P.gemm!(Cs, A, B; alpha = al, beta = be, transB = 'T')
        Ls = copy(L0); ps = Vector{Int}(undef, n); P.getrf!(Ls, ps)
        ds = P.dot(x, y)
        # `PureBLAS.gemm!`, not `P.gemm!`: a testitem's locals are module globals, so `P` is a non-const
        # global, and a keyword call through it is dispatched at run time and boxes its keywords (32 B
        # with α and β). The arrays and scalars are arguments for the same reason.
        gemm1(Cx, Ax, Bx, a, b) = @allocated PureBLAS.gemm!(Cx, Ax, Bx; alpha = a, beta = b, transB = 'T')
        for nw in (nthreads(), nthreads() - 1)
            @test P.set_num_threads(nw) == nw
            @test P._gemm_workers(n, n, k) == nw
            pool = P._gemm_pool(Float64)
            tid = pool.tids[1]                   # hosts worker 1, which participates at any nw ≥ 2
            @test tid != Threads.threadid()
            # Negative control: a driver on the creator's thread has no worker to skip.
            s0 = P._MT_SKIPPED[]
            P.gemm!(copy(C0), A, B; alpha = al, beta = be, transB = 'T')
            @test P._MT_SKIPPED[] == s0
            Ct = copy(C0)
            on_thread(() -> P.gemm!(Ct, A, B; alpha = al, beta = be, transB = 'T'), tid)
            @test P._MT_SKIPPED[] > s0           # the skip path ran
            @test P._mt_gen_nw(@atomic pool.gen) == nthreads()   # every worker in range, one sitting out
            @test bitsame(Ct, Cs)
            s1 = P._MT_SKIPPED[]
            Lt = copy(L0); pt = Vector{Int}(undef, n)
            on_thread(() -> P.getrf!(Lt, pt), tid)
            @test P._MT_SKIPPED[] > s1           # the look-ahead job, whose chunk 1 is the panel
            @test bitsame(Lt, Ls) && pt == ps
            s2 = P._MT_SKIPPED[]
            dt = on_thread(() -> P.dot(x, y), tid)
            @test P._MT_SKIPPED[] > s2           # the reduction, folded by block index by the driver
            @test reinterpret(UInt64, dt) === reinterpret(UInt64, ds)
            # req#10 on this path, with the same minimum-over-windows statistic the threaded gemm item
            # uses and for the same reason: `@allocated` sums every thread in the process.
            on_thread(() -> gemm1(Ct, A, B, al, be), tid)
            @test on_thread(() -> minimum(gemm1(Ct, A, B, al, be) for _ in 1:8), tid) == 0
        end
        P.set_num_threads(1)
    end
end

# The flop-balanced triangular chunker. Two properties matter and they are checked separately, because
# one is a CORRECTNESS invariant and the other is the point of the thing.
#
# PARTITION (correctness). The chunks must tile [0, n) exactly: no gap, and above all NO OVERLAP. Two
# workers sharing a column of a triangular C is a write race, not a slow chunk — it would produce a
# silently wrong answer of the same family as the ones M4 already had to fix. Checked exhaustively over
# a size/worker grid rather than sampled.
#
# BALANCE (the purpose). Equal-WIDTH chunks of a triangle are imbalanced ~2:1 across the pair of
# extreme workers, and a fork-join waits for the slowest. The flop-balanced split must beat that by a
# wide margin at gate sizes. The bound asserted here is deliberately loose (1.25) because rounding to
# `_NR` costs up to one block per worker and that share grows as n shrinks; the equal-width comparison
# in the same testset is what shows the improvement is real rather than the bound being generous.
@testitem "triangular chunker: exact partition, and flop-balanced" begin
    using PureBLAS
    const P = PureBLAS
    tri_work(j0, len, n, up) = sum(up ? (j + 1) : (n - j) for j in j0:(j0 + len - 1); init = 0)

    @testset "partition is exact (no gap, NO OVERLAP)" begin
        for n in (8, 17, 64, 100, 128, 512, 1000, 1024, 2048), nw in 1:8, up in (false, true)
            cov = zeros(Int, n)
            for i in 1:nw
                j0, len = P._tri_chunk(n, nw, i, up)
                @test len >= 0
                @test j0 >= 0 && j0 + len <= n
                for j in (j0 + 1):(j0 + len)
                    cov[j] += 1
                end
            end
            @test all(isone, cov)            # every column covered exactly once
        end
    end

    @testset "flops are balanced, and beat an equal-width split" begin
        for n in (512, 1024, 2048), nw in (2, 4, 6), up in (false, true)
            tot = tri_work(0, n, n, up)
            act = [tri_work(P._tri_chunk(n, nw, i, up)..., n, up) for i in 1:nw]
            @test sum(act) == tot                                   # nothing lost or double-counted
            @test maximum(act) / (tot / nw) <= 1.25                 # worst worker near the mean
            # the equal-width split this replaces, on the same cells
            eq = [tri_work(P._gemm_chunk(n, nw, i)..., n, up) for i in 1:nw]
            @test maximum(act) < maximum(eq)                        # strictly better than what it replaces
        end
    end
end

# Threaded syrk/syr2k. NONE of this phase's behaviour was reachable from `Pkg.test()` before: the
# concurrency runs, the lost-claim fallback and the syr2k `sym2` route all lived in probes, which are
# gitignored. Two bugs in this phase were found only by running those probes — a lost-claim caller
# computing a full rectangular product for a syrk (87 of 96 wrong), and a syr2k gating condition that
# was dead code so the threaded path never ran at all. Both would have shipped silently.
#
# ASSERTS THE WITNESS FIRST. `_syrk_workers(n, k) > 1` is checked before any comparison, because a
# "threaded == serial" result from a call that never threaded is the exact shape of the dead-code bug.
#
# BITWISE, not a tolerance. Thread count must not change a single bit of the result, so the
# comparison is on the bit patterns. A tolerance cannot enforce that: the α-placement divergence this
# guards against is ~1e-13 relative, which any sane tolerance admits.
#
# The α list must contain a value that is not ±2ʲ. Scaling by a power of two is exact, so α ∈ {1, 2,
# 0.5, -4} agree whichever kernel runs and whichever place α is applied — a test written only at
# those values passes without exercising anything.
@testitem "threaded syrk/syr2k: matches serial, and the pool's lost-claim path does too" tags = [:checks] begin
    using PureBLAS, LinearAlgebra
    using Base.Threads: @spawn, nthreads
    const P = PureBLAS
    n, k = 256, 96
    # RAISE THE POOL BEFORE READING THE WITNESS. `_syrk_workers` reports the CONFIGURED pool size,
    # which is 1 until `set_num_threads` raises it. Read first and it answers 1 whatever the shape,
    # the guard below takes the skip branch, and the item reports success having run nothing.
    nthreads() >= 2 && P.set_num_threads(nthreads())
    if nthreads() < 2 || !(P._syrk_workers(n, k) > 1)
        P.set_num_threads(1)
        @test_skip "needs ≥2 julia threads and a shape the amortisation floor admits"
    else
        @test P._syrk_workers(n, k) > 1            # the witness: this shape really does thread
        bitsame(X, Y) = all(i -> reinterpret(UInt64, X[i]) === reinterpret(UInt64, Y[i]), eachindex(X))
        for up in ('L', 'U'), tr in ('N', 'T'), a in (1.0, 0.75, 2.5)
            A = tr == 'T' ? randn(k, n) : randn(n, k)
            B = tr == 'T' ? randn(k, n) : randn(n, k)
            P.set_num_threads(1)
            r1 = zeros(n, n); P.syrk!(r1, A; uplo = up, trans = tr, alpha = a)
            r2 = zeros(n, n); P.syr2k!(r2, A, B; uplo = up, trans = tr, alpha = a)
            P.set_num_threads(nthreads())
            g1 = zeros(n, n); P.syrk!(g1, A; uplo = up, trans = tr, alpha = a)
            g2 = zeros(n, n); P.syr2k!(g2, A, B; uplo = up, trans = tr, alpha = a)
            @test bitsame(g1, r1)
            @test bitsame(g2, r2)
        end
        # CONCURRENT callers: most LOSE the pool claim and take the serial fallback, which is the path
        # that silently ran a rectangular gemm. With this many callers, losing is the common case.
        P.set_num_threads(1)
        As = [randn(n, k) for _ in 1:8]
        want = [(C = zeros(n, n); P.syrk!(C, As[i]; uplo = 'L'); C) for i in 1:8]
        P.set_num_threads(nthreads())
        got = [zeros(n, n) for _ in 1:8]
        foreach(wait, [@spawn P.syrk!(got[i], As[i]; uplo = 'L') for i in 1:8])
        for i in 1:8
            @test bitsame(got[i], want[i])
        end
        P.set_num_threads(1)
    end
end

# THE THREAD-COUNT REPRODUCIBILITY GATE. `gemm!` and `symm!` must return bit-identical results at
# every thread count on one build and one machine — a project requirement, not a nicety, and the one
# property that cannot be checked by reading the code because it turns on which algorithm each worker
# picks rather than on what any single routine computes.
#
# TWO SHAPES, AND THE SECOND IS WHY THIS ITEM IS NOT VACUOUS. `_strassen_owns` declines the column
# split whenever the recursion claims the call, so the large square shape runs serially at any thread
# count and would agree without a worker ever starting. The k-thin shape falls under Strassen's floor,
# takes the classical split, and really is computed by the pool. Both witnesses are asserted.
#
# α = ±2^j is exact, so an α-placement divergence is invisible there; 2.5 is what catches it. The
# comparison is on bit patterns, not a tolerance — a 1e-13 divergence is a divergence.
@testitem "gemm/symm: bit-identical at every thread count" tags = [:checks] begin
    using PureBLAS, LinearAlgebra, Random
    using Base.Threads: nthreads
    const P = PureBLAS
    n, kthin = 512, 96
    nb = 2 * P._fh_strassen_base()            # the smallest square `gemm!` lets the recursion claim
    nthreads() >= 2 && P.set_num_threads(nthreads())
    if nthreads() < 2 || !(P._gemm_workers(n, n, kthin) > 1)
        P.set_num_threads(1)
        @test_skip "needs ≥2 julia threads and a shape the amortisation floor admits"
    else
        Abig = randn(nb, nb); Bbig = randn(nb, nb)
        Athin = randn(n, kthin); Bthin = randn(kthin, n)
        # Witnesses: the first shape is the recursion's, the second the pool's. Strassen is unavailable
        # at vector widths whose packing path does not implement it (`_STRASSEN`), and there the big
        # shape takes the column split like the thin one -- the bit-identity below is the same
        # property either way, so the invariant is still tested and only the witness is skipped.
        if P._STRASSEN
            @test P._strassen_owns(Float64, nb, nb, nb, false, Abig, Bbig, P._STR_FLOOR)
        else
            @test_skip "Strassen is off at _W64=$(PureBLAS._W64)"
        end
        @test !P._strassen_owns(Float64, n, n, kthin, false, Athin, Bthin, P._STR_FLOOR)
        @test P._gemm_workers(n, n, kthin) > 1
        # AND THAT THE COMBINES THEMSELVES SPLIT, which is a second witness and not a duplicate of the
        # one above. The recursion's leaves threading says nothing about its combine passes: those are
        # published as their own jobs and decline below an amortisation floor, so a floor that moved
        # would leave them serial and every assertion below would still pass — the invariant holds
        # trivially when nothing is split. The half-block is what a level-1 combine covers.
        @test P._elem_workers(n ÷ 2, n ÷ 2, nthreads(), Float64) > 1
        bitsame(X, Y) = all(i -> reinterpret(UInt64, X[i]) === reinterpret(UInt64, Y[i]), eachindex(X))
        S = randn(n, n); Bs = randn(n, n)
        for α in (1.0, -1.0, 2.5), β in (0.0, 0.5)
            for (A, B) in ((Abig, Bbig), (Athin, Bthin))
                P.set_num_threads(1)
                r = randn(size(A, 1), size(B, 2)); g = copy(r)
                P.gemm!(r, A, B; alpha = α, beta = β)
                P.set_num_threads(nthreads())
                P.gemm!(g, A, B; alpha = α, beta = β)
                @test bitsame(g, r)
            end
            for side in ('L', 'R'), up in ('L', 'U')
                P.set_num_threads(1)
                r = randn(n, n); g = copy(r)
                P.symm!(r, S, Bs; side, uplo = up, alpha = α, beta = β)
                P.set_num_threads(nthreads())
                P.symm!(g, S, Bs; side, uplo = up, alpha = α, beta = β)
                @test bitsame(g, r)
            end
        end
        P.set_num_threads(1)
    end
end

# THE OTHER HALF OF THE THREADING CONTRACT: the pool must ACTUALLY BE USED.
#
# The reproducibility item above is satisfied perfectly by a library that ignores every thread it is
# given. A change that routed large real `gemm!` to a serial recursion passed it, passed the whole
# suite, and passed the gate — the gate is single-threaded — while leaving a 1024^3 product at 1.00x
# on six cores. An invariant gate needs a liveness gate beside it, one that the do-nothing
# implementation of that invariant fails, or the pair certifies a library that does nothing.
#
# WITNESS, NOT A CLOCK. `p.gen` is bumped once per dispatched job and by the driver only, so reading
# it either side of a call says whether the pool ran, with no timing and nothing to be flaky about on
# a loaded runner. It catches any veto between `_gemm_workers` saying yes and the pool being used —
# including one nobody has written yet, which a check against a named predicate would not.
@testitem "threaded gemm/symm: the pool is actually used" tags = [:checks] begin
    using PureBLAS, LinearAlgebra
    using Base.Threads: nthreads
    const P = PureBLAS
    nt = min(6, nthreads())
    if nt < 2
        @test_skip "needs >=2 julia threads"
    else
        P.set_num_threads(nt)
        p = P._gemm_pool(Float64)
        # LIVENESS IS "A FAST UNIT RAN", NOT "THE POOL RAN". Where a machine has a matrix
        # coprocessor, `_sme_owns` keeps an eligible call off the column split on purpose: the unit is
        # shared by the cluster, so splitting queues the workers behind it rather than dividing the
        # work (measured on an M6 at n=4096: 503 GFLOP/s owned, 301 split, 177 for the threaded NEON
        # path the guard declines). Demanding the pool there would demand the slower of the two.
        #
        # Both witnesses are counters a driver bumps once per dispatched job, so this keeps the
        # property the pool check was written for: an implementation that quietly does neither fails,
        # and so does one that routes to a serial NEON recursion — that bumps nothing.
        ran(f) = (
            g0 = @atomic p.gen; s0 = P._SME_CALLS[]; f();
            (@atomic p.gen) != g0 || P._SME_CALLS[] != s0
        )
        shapes = ((512, 512, 512), (1024, 1024, 1024), (1024, 1024, 128), (2048, 1024, 512))
        for (m, n, k) in shapes
            P._gemm_workers(m, n, k) > 1 || continue
            A = randn(m, k); B = randn(k, n); C = zeros(m, n)
            @test ran(() -> P.gemm!(C, A, B))
        end
        # The same shapes through the `dgemm_64_` entry `activate()` installs, which must take
        # `gemm!`'s route and not a serial core of its own.
        P.activate()
        try
            for (m, n, k) in shapes
                P._gemm_workers(m, n, k) > 1 || continue
                A = randn(m, k); B = randn(k, n); C = zeros(m, n)
                @test ran(() -> mul!(C, A, B))
            end
        finally
            P.deactivate()
        end
        n = 1024
        if P._gemm_workers(n, n, n) > 1
            S = randn(n, n); Bs = randn(n, n); C = zeros(n, n)
            @test ran(() -> P.symm!(C, S, Bs; side = 'L', uplo = 'L'))
        end
        P.set_num_threads(1)
    end
end

@testitem "threaded gemm: every element type the driver admits is reproducible" tags = [:checks] begin
    using PureBLAS, LinearAlgebra, Random
    using Base.Threads: nthreads
    const P = PureBLAS
    # THE TYPE LIST IS READ FROM THE DRIVER'S SIGNATURE, NEVER WRITTEN HERE. `_gemm_threaded!`'s bound
    # names exactly the types `_gemm_poolvec` holds a pool for, so widening that bound widens THIS
    # test in the same edit, and it fails until the new type is genuinely reproducible — which is the
    # point. A list of types spelled out here would have gone on passing while the driver silently
    # grew a type it cannot split correctly.
    bounds = Any[]
    for mm in methods(P._gemm_threaded!)
        s = mm.sig
        while s isa UnionAll
            push!(bounds, s.var.ub)
            s = s.body
        end
    end
    admitted = filter(T -> any(b -> T <: b, bounds), [Float64, Float32, ComplexF64, ComplexF32])
    # Liveness: if the signature walk ever stops finding types, this test would pass by testing
    # nothing. Two real types are admitted today, so anything less means the extraction broke.
    @test length(admitted) >= 2

    n, k = 512, 96
    if nthreads() < 2
        @test_skip "needs >= 2 julia threads"
    else
        bitsame(X, Y) = reinterpret(UInt8, vec(X)) == reinterpret(UInt8, vec(Y))
        for T in admitted
            Random.seed!(20261)
            A = randn(T, n, k); B = randn(T, k, n)
            P.set_num_threads(nthreads())
            # Witness per type: a shape that does NOT reach the pool would make bit-identity vacuous.
            @test P._gemm_workers(n, n, k) > 1
            for α in (one(T), -one(T), T(2.5)), β in (zero(T), T(0.5))
                C0 = randn(T, n, n)
                P.set_num_threads(1);          r = copy(C0)
                P.gemm!(r, A, B; alpha = α, beta = β)
                P.set_num_threads(nthreads()); g = copy(C0)
                P.gemm!(g, A, B; alpha = α, beta = β)
                @test bitsame(g, r)
            end
        end
        P.set_num_threads(1)
    end
end

@testitem "LAPACK drivers: bit-identical at every thread count" tags = [:checks] begin
    using PureBLAS, LinearAlgebra, Random
    using Base.Threads: nthreads
    const P = PureBLAS
    # THE SHAPES ARE THE DRIVERS' OWN, NOT SHAPES CHOSEN HERE. A threaded Level-3 routine is split
    # into column bands, and a band can land on a width that selects a different KERNEL from the one
    # the unsplit call takes — `_gemm_split_max()` is such a width. Whether a band lands there is a
    # function of the driver's blocking, so only the driver's real call sequence exercises it.
    #
    # This item exists because the gemm/symm item above passed while `getrf` was broken: its shapes
    # were square and wide, and every band came out wider than the kernel switch. The bands `getrf`
    # actually produces at n=1024 and n=2048 come out at exactly 64, and there the threaded result
    # differed from the serial one — correct to 1e-14, identical pivots, wrong bits.
    if nthreads() < 2
        @test_skip "needs >= 2 julia threads"
    else
        bitsame(X, Y) = reinterpret(UInt8, vec(X)) == reinterpret(UInt8, vec(Y))
        # Witness FIRST, and under the threaded setting: `_trsm_workers` reads the live thread count,
        # so asking it while threads are set to 1 always answers 1 and the bit-identity below would
        # pass by never threading at all.
        #
        # The witness is that SOME size in the ladder threads, not that every one does. The predicate
        # prices the fork-join against `_gemm_mt_work()`, which scales with the host's flop rate, and at
        # n=512 the work clears that floor by only ~4% — so a machine with a wider datapath answers 1
        # there and reddens the item without anything being wrong. n=2048 clears it by more than an
        # order of magnitude on any host, so the ladder as a whole cannot pass vacuously.
        P.set_num_threads(nthreads())
        @test count(n -> P._trsm_workers(P._lu_nb(n), n - P._lu_nb(n)) > 1, (512, 1024, 2048)) > 0
        # getrf's data movement is threaded too — the trailing row swaps (`_lu_nb(n)` pivot rows over
        # the columns to the right), the deferred left swaps, and the pad copy at a 512-aliased
        # leading dimension. Pure data movement cannot change a bit, so the witness carries this
        # part: the move predicate must admit the driver's own shapes at n=2048.
        @test P._lu_move_workers(P._lu_nb(2048), 2048 - P._lu_nb(2048), Float64) > 1
        @test P._lu_move_workers(2048, 2048, Float64) > 1
        P.set_num_threads(1)
        # n=256 is the small-panel shape (nb=48, no pad); 1024 and 2048 take the pad. The last case is
        # 1024 at a leading dimension that is not 512-aliased, so the no-pad path at a threaded size
        # is covered as well.
        p = P._gemm_pool(Float64)
        for (n, lda) in ((256, 256), (512, 512), (1024, 1024), (2048, 2048), (1024, 1032))
            Random.seed!(4242 + n + lda)
            B0 = randn(lda, n); A0 = view(B0, 1:n, 1:n)
            r = copy(A0); ipr = Vector{Int}(undef, n); P.getrf!(r, ipr)
            for nw in (2, nthreads())
                P.set_num_threads(nw)
                Bg = copy(B0); g = view(Bg, 1:n, 1:n); ipg = Vector{Int}(undef, n)
                g0 = @atomic p.gen
                P.getrf!(g, ipg)
                n >= 1024 && @test (@atomic p.gen) != g0
                @test bitsame(g, r)
                @test ipg == ipr
            end
            # A pivot vector whose storage is not a dense run of `Int` must stay off the pool jobs,
            # which carry `ipiv` as a raw `Ptr{Int}`; `pointer(::Vector{Int32})` converted silently
            # and the look-ahead job read past the vector (a segfault at n=1024). Same bits, no pool
            # dependence on the element type.
            P.set_num_threads(nthreads())
            Bg = copy(B0); g = view(Bg, 1:n, 1:n); ip32 = Vector{Int32}(undef, n)
            P.getrf!(g, ip32)
            @test bitsame(g, r)
            @test ip32 == ipr
        end
        # The threaded entry allocates nothing at steady state (req#10): warm call first, then measure
        # inside a function so the return is not boxed at testitem scope.
        P.set_num_threads(nthreads())
        let A0 = randn(MersenneTwister(7), 1024, 1024), ip = Vector{Int}(undef, 1024)
            f(A, ip) = (P.getrf!(A, ip); @allocated P.getrf!(A, ip))
            @test f(A0, ip) == 0
        end
        P.set_num_threads(1)
        # potrf needs n = 2048 to be worth testing at all: below `_chol_faer_base` it issues NO
        # Level-3 call and runs entirely inside a scalar kernel, so a smaller size would compare two
        # serial runs and pass no matter what. The witness is the pool's generation counter rather
        # than a worker-count predicate, because it catches any veto between "the predicate says
        # yes" and the pool actually running — including one nobody has written yet.
        #
        # EVERY COUNT, NOT TWO SAMPLES. The invariant is quantified over the thread count, so two
        # probe points do not cover it: the lower factorization can return the serial bits at two
        # workers and at the top of a small pool while differing in between, which is invisible on
        # any box with eight or fewer threads. A generation-counter witness does not close that gap
        # — it proves the pool RAN, not that the count was varied.
        p = P._gemm_pool(Float64)
        for n in (1024, 2048)
            Random.seed!(99 + n)
            S = randn(n, n); S = S * S' + n * I
            P.set_num_threads(1)
            r = copy(S); P.potrf!(r; uplo = 'L')
            for nw in 2:nthreads()
                P.set_num_threads(nw)
                g = copy(S)
                g0 = @atomic p.gen
                P.potrf!(g; uplo = 'L')
                n >= 2048 && @test (@atomic p.gen) != g0
                @test bitsame(g, r)
            end
        end
        P.set_num_threads(1)
    end
end

# getrf runs a panel step as ONE job when the pool's workers span cache groups (`_lustep_try!`), and
# the pool only reports that on a host that has more than one group and pinned threads — which is
# neither the CI runner nor most development machines. So the item above never reaches that path
# there. This one declares a split by hand (`_MT_G0`, the number of participants in the driver's
# group) so the grouped step runs on any host with three threads, and asserts that it did.
#
# Shapes: a 512-aliased leading dimension (the pad path) and one that is not, a size that is not a
# multiple of the panel width, and a tall and a wide matrix, because the step's column layout is
# computed from the trailing width and the next panel's width and those differ in each.
@testitem "getrf grouped panel step: bit-identical to serial, taken, 0 B" tags = [:checks] begin
    using PureBLAS, LinearAlgebra, Random
    using Base.Threads: nthreads
    const P = PureBLAS
    if nthreads() < 3
        @test_skip "needs >= 3 julia threads: two local participants and one remote"
    else
        bitsame(X, Y) = reinterpret(UInt8, vec(X)) == reinterpret(UInt8, vec(Y))
        P.set_num_threads(nthreads())
        saved = P._MT_G0[]
        try
            for (m, n, lda) in ((1024, 1024, 1024), (1024, 1024, 1032), (1000, 1000, 1000),
                    (2048, 2048, 2048), (1500, 1100, 1500), (1100, 1500, 1100))
                Random.seed!(777 + m + n + lda)
                B0 = randn(lda, n); k = min(m, n)
                P.set_num_threads(1)
                r = copy(B0); ipr = Vector{Int}(undef, k); P.getrf!(view(r, 1:m, 1:n), ipr)
                P.set_num_threads(nthreads())
                # Every split of the participants into a local and a remote set.
                for g0 in 1:(nthreads() - 1)
                    P._MT_G0[] = g0
                    g = copy(B0); ipg = Vector{Int}(undef, k)
                    s0 = P._MT_LUSTEPS[]
                    P.getrf!(view(g, 1:m, 1:n), ipg)
                    # All but one participant local is the split every size here admits.
                    g0 == nthreads() - 1 && @test P._MT_LUSTEPS[] > s0
                    @test bitsame(g, r)
                    @test ipg == ipr
                end
            end
            # A singular next panel reports the same `info` through the grouped step.
            P._MT_G0[] = nthreads() - 1
            Z = randn(MersenneTwister(5), 1024, 1024); Z[:, 200] .= 0
            P.set_num_threads(1)
            zr = copy(Z); _, _, ir = P.getrf!(zr, Vector{Int}(undef, 1024))
            P.set_num_threads(nthreads())
            s0 = P._MT_LUSTEPS[]
            zg = copy(Z); _, _, ig = P.getrf!(zg, Vector{Int}(undef, 1024))
            @test P._MT_LUSTEPS[] > s0
            @test ig == ir == 200
            @test bitsame(zg, zr)
            # req#10 on the grouped step: warm call, then measure inside a function.
            let A0 = randn(MersenneTwister(7), 1024, 1024), ip = Vector{Int}(undef, 1024)
                f(A, ip) = (PureBLAS.getrf!(A, ip); @allocated PureBLAS.getrf!(A, ip))
                s1 = P._MT_LUSTEPS[]
                @test f(A0, ip) == 0
                @test P._MT_LUSTEPS[] > s1
            end
        finally
            P._MT_G0[] = saved
            P.set_num_threads(1)
        end
    end
end

# potrf's own thread-invariance item. The Cholesky cases in the item above are lower-only at two
# power-of-two sizes; this one covers both triangles, sizes that take the pad round-trip (a leading
# dimension on the L1 way stride) and sizes that do not, and a view whose row count is not its
# leading dimension. The pad's two triangle copies run on the pool, so the copy job has its own
# witness, its own forced lost claim and a place in the allocation check.
@testitem "potrf: bit-identical at every thread count, both triangles, pad copy on the pool" tags = [:checks] begin
    using PureBLAS, LinearAlgebra, Random
    using Base.Threads: nthreads, @spawn
    const P = PureBLAS
    if nthreads() < 2
        @test_skip "needs >= 2 julia threads"
    else
        bitsame(X, Y) = reinterpret(UInt64, vec(X)) == reinterpret(UInt64, vec(Y))
        P.set_num_threads(nthreads())
        p = P._gemm_pool(Float64)
        jobs() = (@atomic p.gen) >> P._MT_NW_SHIFT
        # ── the triangle copy alone: it reaches the pool, moves exactly one triangle, and a caller
        # that loses the claim moves the same elements. `dst` starts as NaN, so `isequal` against a
        # reference that is NaN outside the triangle also proves nothing else was written.
        #
        # THE SIZES COME FROM THE ADMISSION FLOOR, not from this box. The copy moves n·(n+1) elements
        # (one triangle read, one written) and `_lu_move_workers` declines below `_L1_MT_MIN` bytes,
        # which follows the host's L1. `nlo` is the first n past that floor with a little margin, so
        # the smallest admitted shape is exercised wherever the floor sits; the other two clear it
        # by more than an order of magnitude.
        nlo = isqrt(P._L1_MT_MIN ÷ sizeof(Float64)) + 3
        nhi = 8 * nlo
        for up in (false, true), (n, lds, ldd) in ((nhi, nhi, nhi + 8), (nhi - 24, nhi, nhi - 16), (nlo, nlo + 5, nlo))
            @test P._lu_move_workers((n + 1) ÷ 2, n, Float64) > 1
            src = randn(MersenneTwister(n + up), lds, n)
            ref = fill(NaN, ldd, n)
            for j in 1:n, i in (up ? (1:j) : (j:n))
                ref[i, j] = src[i, j]
            end
            dst = fill(NaN, ldd, n)
            g0 = jobs()
            GC.@preserve src dst P._tricopy_mt!(pointer(dst), ldd, pointer(src), lds, n, up)
            @test jobs() == g0 + 1
            @test isequal(dst, ref)
            # FORCED LOSS: holding the claim makes the fallback certain, and `_MT_LOST` shows it ran.
            fill!(dst, NaN)
            lost0 = P._MT_LOST[]
            _, held = @atomicreplace p.busy false => true
            @test held
            GC.@preserve src dst P._tricopy_mt!(pointer(dst), ldd, pointer(src), lds, n, up)
            @atomic p.busy = false
            @test P._MT_LOST[] == lost0 + 1
            @test isequal(dst, ref)
        end
        # ── potrf, every worker count. (n, lda): 256/512/1024/2048 pad the lower factorization; 1000
        # and 1029 do not; (1000, 1024) pads a view; (1024, 1032) is a threaded size without the pad.
        cases = ((256, 256), (512, 512), (1000, 1000), (1000, 1024), (1024, 1024), (1024, 1032), (1029, 1029), (2048, 2048))
        for (n, lda) in cases, ul in ('L', 'U')
            Random.seed!(7 * n + lda)
            G = randn(n, n); S = G * G' + n * I
            B0 = zeros(lda, n); copyto!(view(B0, 1:n, :), S)
            P.set_num_threads(1)
            r = copy(B0); P.potrf!(view(r, 1:n, 1:n); uplo = ul)
            for nw in 2:nthreads()
                P.set_num_threads(nw)
                g = copy(B0)
                g0 = jobs()
                P.potrf!(view(g, 1:n, 1:n); uplo = ul)
                n >= 2048 && @test jobs() > g0                # the pool ran
                @test bitsame(g, r)
            end
        end
        # WITNESS THAT THE PAD COPIES ARE POOL JOBS INSIDE potrf: the same matrix factored at a
        # way-aligned leading dimension makes exactly two more jobs than at one that is not. The pad
        # predicate fires when the byte stride is a multiple of twice its period `q = way ÷ 4`, so `n`
        # is the smallest size >= 1024 whose byte stride is such a multiple: that holds for any way
        # size, power of two or not. It is far above the `_L1_MT_MIN` floor, so both copies are admitted.
        P.set_num_threads(nthreads())
        let q2 = (P._L1_WAY_BYTES >> 2) << 1, m = q2 ÷ gcd(q2, sizeof(Float64)), n = m * cld(1024, m)
            geom = "L1 = $(P._L1_BYTES) B, $(P._L1D_ASSOC)-way, way = $(P._L1_WAY_BYTES) B, " *
                "L2 = $(P._L2_BYTES) B, n = $n"
            # Logs the detected geometry when a check fails, so a failure on another host explains itself.
            chk(c) = (c || @error "pad witness check failed" geom; c)
            @test chk(n * n * sizeof(Float64) >= 8 * P._L1_MT_MIN)
            G = randn(MersenneTwister(5), n, n); S = G * G' + n * I
            @test chk(P._chol_needs_pad(S, n))
            Bq = zeros(n + 8, n); Xq = view(Bq, 1:n, 1:n)
            if P._chol_needs_pad(Xq, n)
                # The unpadded control needs a stride off the aliasing period, which no n + 8 gives
                # when the period divides 64 B.
                @test_skip P._chol_needs_pad(Xq, n)
                @info "pad witness skipped: control stride also aliases" geom
            else
                X = copy(S); j0 = jobs(); P.potrf!(X; uplo = 'L'); jp = jobs() - j0
                copyto!(Xq, S); j0 = jobs(); P.potrf!(Xq; uplo = 'L'); jq = jobs() - j0
                @test chk(jp == jq + 2)
                @test bitsame(tril(X), tril(Xq))
            end
        end
        # Float32 takes the same padded driver and the same copy job.
        let n = 1024
            G = randn(MersenneTwister(6), Float32, n, n); S = G * G' + n * I
            P.set_num_threads(1); r = copy(S); P.potrf!(r; uplo = 'L')
            p32 = P._gemm_pool(Float32)
            for nw in 2:nthreads()
                P.set_num_threads(nw)
                g = copy(S); g0 = @atomic p32.gen
                P.potrf!(g; uplo = 'L')
                @test (@atomic p32.gen) != g0
                @test reinterpret(UInt32, vec(g)) == reinterpret(UInt32, vec(r))
            end
        end
        # CONCURRENT callers: all but one lose the claim somewhere inside the call, the pad copies
        # included, and every result must still be the serial bits.
        P.set_num_threads(nthreads())
        let n = 1024
            G = randn(MersenneTwister(8), n, n); S = G * G' + n * I
            P.set_num_threads(1); r = copy(S); P.potrf!(r; uplo = 'L')
            P.set_num_threads(nthreads())
            lost0 = P._MT_LOST[]
            tasks = [@spawn (t = copy(S); P.potrf!(t; uplo = 'L'); t) for _ in 1:(nthreads() + 1)]
            for t in tasks
                @test bitsame(fetch(t), r)
            end
            @test P._MT_LOST[] > lost0
        end
        # req#10: 0 B at steady state with the pool on, both triangles, at a padded size. Measured in
        # a function over pre-built operands, after a warm call.
        P.set_num_threads(nthreads())
        let n = 1024
            G = randn(MersenneTwister(9), n, n); S = G * G' + n * I; X = similar(S)
            f(X, S, ul) = (copyto!(X, S); P.potrf!(X; uplo = ul); copyto!(X, S); @allocated P.potrf!(X; uplo = ul))
            @test f(X, S, 'L') == 0
            @test f(X, S, 'U') == 0
        end
        P.set_num_threads(1)
    end
end

# trsm had NO thread-invariance item, which is how a complex admission that violated requirement 11 in
# four places passed its own certification: that certification used square power-of-two B, where every
# band is a multiple of the row granule and no band ends mid-vector. This item exists so a future
# widening of either `trsm!` admission is caught rather than measured.
#
# IT IS A REGRESSION GUARD, AND FOR COMPLEX IT IS CURRENTLY VACUOUS BY CONSTRUCTION — both admissions
# are `BlasReal`, so the complex cases below cannot thread and verify trivially. That is stated rather
# than hidden: the real cases carry the liveness witness, and the complex cases fire the moment anyone
# admits complex. The three mechanisms they are aimed at, each measured on Zen4 before the revert:
#   * the complex-gemm 3M window reads `min(m, nrt, k)` with `m` the band's ROW count, so a band narrower
#     than kmin declines Karatsuba while the unsplit solve takes it. B 48x512 at alpha=1 diverged.
#   * `_trsm_cmplx_small_R!` routes on `size(B, 1)`, not `mrt`; the last band can be one row.
#   * `_scal_all!` applies a complex alpha flat or per column, and its vector body uses an fma where
#     its scalar tail uses the two-rounding form. ComplexF32 m=n=k=1001 at alpha=3.7-1.25i diverged.
# Hence the shapes: NON-square and non-power-of-two B, an odd dimension, and an alpha that is not a
# power of two. A square po2 ladder reproduces none of it.
@testitem "trsm: bit-identical at every thread count" tags = [:checks] begin
    using Random, LinearAlgebra
    import PureBLAS as P
    # BYTE-WISE, not `reinterpret(UInt64, …)`: a `Float32` is four bytes, so a fixed-width reinterpret
    # throws on half the element types this item covers. Bytes also distinguish -0.0 and NaN payloads,
    # which is the point of comparing patterns rather than values.
    bitsame(X, Y) = size(X) == size(Y) &&
        reinterpret(UInt8, vec(collect(X))) == reinterpret(UInt8, vec(collect(Y)))
    nt = Threads.nthreads()
    # m=41 AND m=48 SIDE R PIN `_gemm_unpacked_split!`'s ROW BLOCKS. The side-R band granule is
    # `_MR * _vwidth(Float64)` = 16 and the split driver tiles rows in `_SMR * _vwidth` = 32, so the two
    # carvings do not nest: m=41 is 32+9 unsplit and 16, 25 at two workers, which places the same rows
    # in an aligned block one way and a ragged block the other. Every block must reach the SAME
    # microkernel with the same `_GEMM_SPLIT_S` chain count, masked when ragged — a kernel chosen per
    # block from its own width sums k in an order that follows the worker count. m=48 is the control:
    # 32+16 and 16+16+16 are all-aligned, so divisibility by the band granule is NOT the rule.
    tri(T, s) = (
        A = randn(T, s, s) ./ (2s); for i in 1:s
            A[i, i] = one(T) + abs(A[i, i])
        end; A
    )

    # ROUTING WITNESS, derived from the consts rather than asserted about the literal 41. The three
    # shapes above are only meaningful while m=41 still reaches a RAGGED row block inside the split
    # driver; if `_gemm_split_max()` drops below the leaf extents, or `_SMR`/`_MR` move, the shape stops
    # entering that kernel and the comparisons below pass without testing anything. This fails loudly
    # instead. It cannot witness the leaf's own n and k — that needs a probe, not a predicate — so it
    # bounds the row axis only.
    let W = P._vwidth(Float64), m = 41
        @test !P._SPLIT_OK || (
            m >= P._SMR * W &&                               # clears the split driver's row floor
                m <= P._gemm_split_max() &&                  # and its cap
                rem(m, P._SMR * W) != 0 &&                   # leaves a ragged row block unsplit
                rem(P._trsm_rchunk(m, 2, 2, Float64)[2], W) != 0   # and a ragged one in band 2 of 2
        )
    end

    # LIVENESS, on the real path that does thread: without this the whole item can pass with no worker
    # ever started, which is the failure mode CLAUDE.md flags for every threading gate here.
    P.set_num_threads(nt)
    @test P._trsm_workers(1024, 1024) > 1 || P._trsm_workers_r(1024, 1024, Float64) > 1
    P.set_num_threads(1)

    for T in (Float64, Float32, ComplexF64, ComplexF32)
        for (side, m, n) in (('L', 1001, 257), ('L', 512, 41), ('R', 41, 1024), ('R', 48, 512))
            k = side == 'L' ? m : n
            for uplo in ('U', 'L'), tA in ('N', 'T', 'C'), al in (one(T), T(3.7) - T(1.25) * im)
                T <: Real && !isreal(al) && continue          # a complex alpha is not a real alpha
                Random.seed!(97 + m + n + Int(side) + Int(uplo) + Int(tA))
                A = tri(T, k); B0 = randn(T, m, n)
                r = copy(B0)
                P.set_num_threads(1)
                P.trsm!(r, A; side, uplo, transA = tA, alpha = al)
                for nw in 2:nt
                    P.set_num_threads(nw)
                    g = copy(B0)
                    P.trsm!(g, A; side, uplo, transA = tA, alpha = al)
                    @test bitsame(g, r)
                end
                P.set_num_threads(1)
            end
        end
    end
end

# THE ADMISSION BOUNDARY IS WHERE A FLOOR CHANGE LANDS, AND NOTHING WAS ANCHORED TO IT.
#
# The reproducibility items above each pick a shape and assert the pool runs there, so a floor that
# ROSE past their shape fails them loudly. The opposite direction had no guard: lowering the floor
# admits smaller shapes that no item exercised, and those are exactly the shapes whose chunk geometry
# degenerates. At `_NR`-rounded boundaries `_tri_chunk` emits ZERO-LENGTH chunks — at n=50 with six
# workers and `_NR`=8 its boundaries are 0,8,8,16,24,32,50, so one worker is handed nothing and skips.
# That branch becomes reachable the moment the floor drops below it.
#
# So this item does not name a size. It walks a ladder and finds, per op, the SMALLEST shape at which
# the pool actually runs, then checks bit-identity there and asserts the ladder entry below it stays
# serial. Move the floor and the test moves with it.
#
# LIVENESS BY WITNESS, NOT BY PREDICATE: `p.gen` is bumped once per dispatched job by the driver, the
# same mechanism the pool-usage items use, so this sees any veto between a predicate saying yes and
# the pool being used. VACUITY GUARD at the end: if no op found a boundary on the ladder, the item
# FAILS rather than passing with nothing checked.
@testitem "L3: bit-identical at the admission boundary, wherever the floor puts it" tags = [:checks] begin
    using PureBLAS, LinearAlgebra
    using Base.Threads: nthreads
    const P = PureBLAS
    nt = min(6, nthreads())
    if nt < 2
        @test_skip "needs >=2 julia threads"
    else
        P.set_num_threads(nt)
        p = P._gemm_pool(Float64)
        ran(f) = (g0 = @atomic p.gen; s0 = P._SME_CALLS[]; f();
            (@atomic p.gen) != g0 || P._SME_CALLS[] != s0)
        bitsame(X, Y) = all(i -> reinterpret(UInt64, X[i]) === reinterpret(UInt64, Y[i]), eachindex(X))
        LADDER = (16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128, 160, 192)
        tri(s) = tril(randn(s, s)) + s * I
        # Each entry: name, build the operands at size n, run the op on them in place.
        OPS = (
            ("gemm", n -> (randn(n, n), randn(n, n), zeros(n, n)),
                c -> P.gemm!(c[3], c[1], c[2]; alpha = 1.0, beta = 0.0), 3),
            ("symm", n -> (let S = randn(n, n); (S + S', randn(n, n), zeros(n, n)) end),
                c -> P.symm!(c[3], c[1], c[2]; side = 'L', uplo = 'L', alpha = 1.0, beta = 0.0), 3),
            ("syrk", n -> (randn(n, n), zeros(n, n)),
                c -> P.syrk!(c[2], c[1]; uplo = 'L', trans = 'N', alpha = 1.0, beta = 0.0), 2),
            ("syr2k", n -> (randn(n, n), randn(n, n), zeros(n, n)),
                c -> P.syr2k!(c[3], c[1], c[2]; uplo = 'L', trans = 'N', alpha = 1.0, beta = 0.0), 3),
            ("trsmL", n -> (tri(n), randn(n, n)),
                c -> P.trsm!(c[2], c[1]; side = 'L', uplo = 'L', transA = 'N', diag = 'N'), 2),
            ("trsmR", n -> (tri(n), randn(n, n)),
                c -> P.trsm!(c[2], c[1]; side = 'R', uplo = 'L', transA = 'N', diag = 'N'), 2),
            ("trmmL", n -> (tri(n), randn(n, n)),
                c -> P.trmm!(c[2], c[1]; side = 'L', uplo = 'L', transA = 'N', diag = 'N'), 2),
            ("trmmR", n -> (tri(n), randn(n, n)),
                c -> P.trmm!(c[2], c[1]; side = 'R', uplo = 'L', transA = 'N', diag = 'N'), 2),
        )
        # A `Ref`, not a plain binding: a testitem body is top-level scope, so `found += 1` inside the
        # loop below would create a new local and the vacuity guard would always read 0.
        found = Ref(0)
        for (nm, mk, op, out) in OPS
            prev = 0
            for n in LADDER
                master = mk(n)
                threaded = map(copy, master)
                P.set_num_threads(nt)
                if !ran(() -> op(threaded))
                    prev = n                      # still serial here; remember it for the boundary claim
                    continue
                end
                # BOUNDARY: this is the smallest ladder shape the pool takes for this op.
                found[] += 1
                if prev > 0
                    pm = mk(prev); pc = map(copy, pm)
                    P.set_num_threads(nt)
                    @test !ran(() -> op(pc))      # the entry below it must stay serial
                end
                serial = map(copy, master)
                P.set_num_threads(1); op(serial)
                th = map(copy, master)
                P.set_num_threads(nt); op(th)
                @test bitsame(th[out], serial[out])
                break
            end
        end
        # A ladder on which nothing threads would pass every assertion above by checking none of them.
        @test found[] >= 4
        P.set_num_threads(1)
    end
end

# What the pool measures when it is created. The cut that finds cache groups is checked on readings
# stated by hand, because a host with one group never produces a split; the rest is that a created
# pool has a polling span and a floor at all.
@testitem "threaded pool: host measurement" tags = [:checks] begin
    using PureBLAS
    using Base.Threads: nthreads
    const P = PureBLAS
    # Two sets further apart than either is wide (Zen3's readings: near 1.36-1.48, far 4.02-4.86).
    # Two sets with a clear gap (Zen3's readings: near 1.36-1.48, far 4.02-4.86). Each worker has one
    # reading taken across an interrupt, which its median ignores.
    @test P._mt_cut([[1.36, 1.41, 9.0], [4.02, 4.5, 0.9], [1.48, 1.4, 7.0], [4.86, 4.3, 30.0]]) == [false, true, false, true]
    # One set: workers of one group differ by a few percent.
    @test !any(P._mt_cut([[1.42, 1.5, 1.41], [1.44, 1.4, 1.47], [1.48, 1.43, 1.4], [1.52, 1.5, 1.55]]))
    # A gap that is real but below the margin.
    @test !any(P._mt_cut([[1.0, 1.05, 1.02], [1.5, 1.9, 1.6]]))
    @test P._mt_settled(() -> 3.0, typemax(UInt64)) == 3.0
    # A deadline already past: the rule stops at its first chance, on the three readings it has.
    let k = Ref(0)
        @test P._mt_settled(() -> Float64(k[] += 1), UInt64(0)) == 2.0
    end
    @test P._mt_clock_step() > 0
    @test P._mt_affinity_count() >= 0
    if nthreads() < 2
        @test_skip "needs >= 2 julia threads: there is no pool to measure without one"
    else
        P.set_num_threads(nthreads())
        try
            @test P._MT_POLL_NS[] >= 1 && P._MT_POLL_NS_HOME[] >= 1
            @test 0 < P._gemm_mt_work() < typemax(Int)
            if isnothing(P._GEMM_MT_WORK_PREF)
                h = P._MT_HOST
                @test h.join_ns > 0 && h.flops_per_ns > 0
                @test P._gemm_mt_work() == P._mt_floor()
            end
        finally
            P.set_num_threads(1)
        end
    end
end
