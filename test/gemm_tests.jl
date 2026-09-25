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
        @test steady(PureBLAS.gemm!, Cz, Az, Bz;
                     alpha = one(ComplexF64), beta = zero(ComplexF64)) == 0
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
        for (m, n, k) in ((256, 256, 256), (512, 300, 128), (200, 512, 333), (1024, 129, 64), (192, 7, 192))
            for tA in (false, true), tB in (false, true)
                A = tA ? randn(k, m) : randn(m, k)
                B = tB ? randn(n, k) : randn(k, n)
                C0 = randn(m, n)
                al, be = 1.7, -0.3
                Cs = copy(C0)
                P._gemm_core!(Cs, A, B, al, be, tA, tB, false, false)
                Ct = copy(C0)
                P.gemm!(Ct, A, B; alpha = al, beta = be,
                    transA = tA ? 'T' : 'N', transB = tB ? 'T' : 'N')
                @test Ct ≈ Cs rtol = 1e-12
            end
        end
        # req#10 holds on the threaded path too: the pool is built once, so a warm call allocates
        # nothing even when it wakes four workers.
        #
        # Measured through a function barrier so the operands arrive with concrete types, as they do
        # from LinearAlgebra and LBT. At module scope the call is `Any`-typed, and the first compile
        # of each such call site caches a method instance — state that scales with the number of call
        # sites, not with the number of calls, so no amount of warming removes it. Only the SME route
        # shows it, because its portability barrier is a function pointer (one dynamic dispatch).
        # A CONCRETE THREE-ARGUMENT WRAPPER, not a varargs one: splatting a tuple through `f(args...)`
        # allocates here even when the call it wraps does not, which turns the barrier into the thing
        # being measured. Measured on x86: the varargs form reported 224 B for a call that is
        # allocation-free.
        gemm3(Cx, Ax, Bx) = P.gemm!(Cx, Ax, Bx)
        A = randn(512, 512); B = randn(512, 512); C = zeros(512, 512)
        @test P._gemm_workers(512, 512, 512) > 1
        gemm3(C, A, B)
        @test @allocated(gemm3(C, A, B)) == 0
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
        A = randn(300, 300); B = randn(300, 300)
        want = similar(A); P._gemm_core!(want, A, B, 1.0, 0.0, false, false, false, false)
        outs = [zeros(300, 300) for _ in 1:nt]
        Threads.@threads :static for t in 1:nt
            for _ in 1:8
                P.gemm!(outs[t], A, B)
            end
        end
        for t in 1:nt
            @test outs[t] ≈ want rtol = 1e-12
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
    n, kbig, kthin = 512, 384, 96
    nthreads() >= 2 && P.set_num_threads(nthreads())
    if nthreads() < 2 || !(P._gemm_workers(n, n, kthin) > 1)
        P.set_num_threads(1)
        @test_skip "needs ≥2 julia threads and a shape the amortisation floor admits"
    else
        Abig = randn(n, kbig); Bbig = randn(kbig, n)
        Athin = randn(n, kthin); Bthin = randn(kthin, n)
        # Witnesses: the first shape is the recursion's, the second the pool's. Strassen is unavailable
        # at vector widths whose packing path does not implement it (`_STRASSEN`), and there the big
        # shape takes the column split like the thin one -- the bit-identity below is the same
        # property either way, so the invariant is still tested and only the witness is skipped.
        if P._STRASSEN
            @test P._strassen_owns(Float64, n, n, kbig, false, Abig, Bbig)
        else
            @test_skip "Strassen is off at _W64=$(PureBLAS._W64)"
        end
        @test !P._strassen_owns(Float64, n, n, kthin, false, Athin, Bthin)
        @test P._gemm_workers(n, n, kthin) > 1
        bitsame(X, Y) = all(i -> reinterpret(UInt64, X[i]) === reinterpret(UInt64, Y[i]), eachindex(X))
        S = randn(n, n); Bs = randn(n, n)
        for α in (1.0, -1.0, 2.5), β in (0.0, 0.5)
            for (A, B) in ((Abig, Bbig), (Athin, Bthin))
                P.set_num_threads(1)
                r = randn(n, n); g = copy(r)
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
        ran(f) = (g0 = @atomic p.gen; s0 = P._SME_CALLS[]; f();
                  (@atomic p.gen) != g0 || P._SME_CALLS[] != s0)
        for (m, n, k) in ((512, 512, 512), (1024, 1024, 1024), (1024, 1024, 128), (2048, 1024, 512))
            P._gemm_workers(m, n, k) > 1 || continue
            A = randn(m, k); B = randn(k, n); C = zeros(m, n)
            @test ran(() -> P.gemm!(C, A, B))
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
    # THE TYPE LIST IS READ FROM THE DRIVER'S SIGNATURE, NEVER WRITTEN HERE. `_gemm_threaded!` is
    # declared `where {T <: BlasReal}` because three things in its body exist only for the real types:
    # the pool registry, the shared-pack prefit, and the `nroute` discipline that keeps a column slice
    # on the whole problem's route. Widening that bound to admit complex therefore widens THIS test in
    # the same edit, and it fails until complex is genuinely reproducible — which is the point. A list
    # of types spelled out here would have gone on passing while the driver silently grew a type it
    # cannot split correctly.
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
        for n in (512, 1024, 2048)
            Random.seed!(4242 + n)
            A0 = randn(n, n)
            # Witness FIRST, and under the threaded setting: `_trsm_workers` reads the live thread
            # count, so asking it while threads are set to 1 always answers 1 and the bit-identity
            # below would pass by never threading at all.
            nb = P._lu_nb(n)
            P.set_num_threads(nthreads())
            @test P._trsm_workers(nb, n - nb) > 1
            P.set_num_threads(1)
            r = copy(A0); ipr = Vector{Int}(undef, n); P.getrf!(r, ipr)
            for nw in (2, nthreads())
                P.set_num_threads(nw)
                g = copy(A0); ipg = Vector{Int}(undef, n); P.getrf!(g, ipg)
                @test bitsame(g, r)
                @test ipg == ipr
            end
        end
        # potrf needs n = 2048 to be worth testing at all: below `_chol_faer_base` it issues NO
        # Level-3 call and runs entirely inside a scalar kernel, so a smaller size would compare two
        # serial runs and pass no matter what. The witness is the pool's generation counter rather
        # than a worker-count predicate, because it catches any veto between "the predicate says
        # yes" and the pool actually running — including one nobody has written yet.
        p = P._gemm_pool(Float64)
        for n in (1024, 2048)
            Random.seed!(99 + n)
            S = randn(n, n); S = S * S' + n * I
            P.set_num_threads(1)
            r = copy(S); P.potrf!(r; uplo = 'L')
            for nw in (2, nthreads())
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
