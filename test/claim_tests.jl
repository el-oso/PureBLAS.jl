# The threaded-job claim is ONE claim for the whole process (`_MT_CLAIM`), not one per element-type pool.
#
# Three things outside any single pool rely on "one threaded job at a time": the per-thread arena
# (`src/arena.jl`, part (c)), the shared reduction/argmax partials buffers (`_REDI`, `_red_partials`)
# and the shared pack buffers. A claim per pool lets a Float64 and a Float32 driver run together: both
# sit suspended inside an arena scope across their joins and the first to exit rewinds the arena under
# the other; two argmax reductions write one `_REDI`. These items test the invariant directly and then
# the failure it prevents.
#
# CI sets `JULIA_NUM_THREADS`; with one thread nothing threads and every item here reports a skip.

@testmodule ClaimHarness begin
using PureBLAS, Random, LinearAlgebra
const P = PureBLAS
export P, ELTYPES, sme, bitsame, njobs, alljobs, cases, getri_input, getri_run!, loop_getri, loop_iamax

const ELTYPES = (Float64, Float32, ComplexF64, ComplexF32)
bitsame(X, Y) = size(X) == size(Y) && reinterpret(UInt8, vec(collect(X))) == reinterpret(UInt8, vec(collect(Y)))
njobs(T) = (@atomic P._gemm_pool(T).gen) >> P._MT_NW_SHIFT
alljobs() = map(njobs, ELTYPES)

# Real gemm on an SME host is taken by the coprocessor route before the pool claim, so its products
# never publish a job or lose the claim; every pool witness below is asked only where the pool route
# exists.
sme(T) = (T === Float64 && P._SME_F64) || (T === Float32 && P._SME_F32)

# One closure per (kind, element type): it builds nothing but fresh outputs from fixed inputs and returns
# them as a tuple, so the same closure run at one thread and at many is a bitwise comparison.
function cases(T, n; vectors = true)
    rng = MersenneTwister(7)
    A = randn(rng, T, n, n); B = randn(rng, T, n, n)
    S = A + transpose(A)
    U = triu(A) ./ T(2n) + T(2) * I          # well conditioned triangular
    Pd = A * A' + T(n) * I                   # positive definite
    ipiv0 = zeros(Int, n); LU = copy(A); P.getrf!(LU, ipiv0)
    out = Pair{String, Function}[]
    push!(out, "gemm!" => () -> (C = zeros(T, n, n); P.gemm!(C, A, B); (C,)))
    push!(out, "trsm! L" => () -> (X = copy(B); P.trsm!(X, U; side = 'L', uplo = 'U'); (X,)))
    push!(out, "trsm! R" => () -> (X = copy(B); P.trsm!(X, U; side = 'R', uplo = 'U'); (X,)))
    push!(out, "trmm! L" => () -> (X = copy(B); P.trmm!(X, U; side = 'L', uplo = 'U'); (X,)))
    push!(out, "trmm! R" => () -> (X = copy(B); P.trmm!(X, U; side = 'R', uplo = 'U'); (X,)))
    push!(out, "syrk!" => () -> (C = zeros(T, n, n); P.syrk!(C, A; uplo = 'L'); (C,)))
    push!(out, "symm!" => () -> (C = zeros(T, n, n); P.symm!(C, S, B; side = 'L', uplo = 'U'); (C,)))
    push!(out, "getrf!" => () -> (W = copy(A); ip = zeros(Int, n); P.getrf!(W, ip); (W, ip)))
    push!(out, "potrf!" => () -> (W = copy(Pd); P.potrf!(W; uplo = 'L'); (W,)))
    push!(out, "getri!" => () -> (W = copy(LU); P.getri!(W, ipiv0); (W,)))
    if vectors
        m = 8200
        G = randn(rng, T, m, 8); gx = randn(rng, T, 8); gy = randn(rng, T, m)
        push!(out, "gemv!" => () -> (y = copy(gy); P.gemv!(y, G, gx; alpha = T(0.75), beta = T(2.5)); (y,)))
        ns = 2048
        Sy = randn(rng, T, ns, ns); sx = randn(rng, T, ns); sy = randn(rng, T, ns)
        push!(out, "symv!" => () -> (y = copy(sy); P.symv!(y, Sy, sx; uplo = 'L', alpha = T(0.75), beta = T(2.5)); (y,)))
        if !(T <: Complex)
            v = randn(rng, T, 1 << 21); w = randn(rng, T, 1 << 21); v[1_500_000] = T(10)
            push!(out, "dot" => () -> (T[P.dot(v, w)],))
            push!(out, "iamax" => () -> (Int[P.iamax(v)],))
        end
    end
    return out
end

getri_input(T, n) = (rng = MersenneTwister(1); A = randn(rng, T, n, n); ip = zeros(Int, n); P.getrf!(A, ip); (A, ip))
function getri_run!(W, LU, ip)
    copyto!(W, LU); P.getri!(W, ip); W
end

# `n` getri! calls in a task, each followed by a yield so two such tasks on one thread interleave at
# their joins. Returns how many results differed from the serial reference `ref`.
function loop_getri(LU, ip, ref, n)
    W = similar(LU); bad = 0
    for _ in 1:n
        getri_run!(W, LU, ip)
        bitsame(W, ref) || (bad += 1)
        yield()
    end
    return bad
end

function loop_iamax(x, ref, n)
    bad = 0
    for _ in 1:n
        P.iamax(x) == ref || (bad += 1)
        sleep(1.0e-4)      # leaves the claim free for the other thread's call
    end
    return bad
end
end

@testitem "claim: while held, every threaded kind of every element type runs serial and publishes nothing" tags = [:checks] setup = [ClaimHarness] begin
    using Base.Threads
    if nthreads() < 2
        @test_skip "needs >= 2 julia threads"
    else
        P.set_num_threads(nthreads())
        # Kinds that reach the pool for a type at these sizes on every host without an SME route for
        # that type. Complex trsm/trmm/syrk/symm/potrf do not reach it, and neither does Float32 trmm
        # side R. Where the type has an SME route nothing is required. The table is exercised wherever
        # the liveness check shows a kind threads, so a kind that does not thread for a type on this
        # host is not asserted about vacuously.
        allkinds = Set(["gemm!", "trsm! L", "trsm! R", "trmm! L", "trmm! R", "syrk!", "symm!", "getrf!", "potrf!", "getri!"])
        requiredfor(T) = sme(T) ? Set{String}() : T <: Complex ? Set(["gemm!", "getrf!", "getri!"]) :
            T === Float32 ? setdiff(allkinds, ["trmm! R"]) : allkinds
        nlive = Ref(0)
        for T in ELTYPES
            pending = Set{String}()
            for n in (512, 1024, 2048)
                n > 512 && issubset(requiredfor(T), pending) && break
                for (name, f) in cases(T, n; vectors = n == 512)
                    name in pending && continue
                    P.set_num_threads(1)
                    want = f()
                    P.set_num_threads(nthreads())
                    # LIVENESS FIRST: without the claim held, this call must publish on its own pool —
                    # otherwise "nothing was published while held" proves nothing about it.
                    g0 = njobs(T)
                    free = f()
                    live = njobs(T) > g0
                    live || continue
                    push!(pending, name)
                    @testset "$T $name n=$n" begin
                        nlive[] += 1
                        @test all(bitsame.(free, want))
                        # HOLD THE CLAIM: every pool's generation stays put, the loss is counted,
                        # bits are serial.
                        acquired = !Threads.atomic_cas!(P._MT_CLAIM, false, true)
                        @test acquired
                        acquired && try
                            j0 = alljobs(); l0 = P._MT_LOST[]
                            held = f()
                            @test alljobs() == j0
                            @test P._MT_LOST[] > l0
                            @test all(bitsame.(held, want))
                        finally
                            P._MT_CLAIM[] = false
                        end
                    end
                end
            end
            @info "claim liveness" T threads = sort!(collect(pending))
            @test isempty(setdiff(requiredfor(T), pending))
        end
        @test nlive[] >= sum(length ∘ requiredfor, ELTYPES)
        P.set_num_threads(1)
    end
end

@testitem "claim: two tasks on one thread, different element types, getri! stays bitwise serial" tags = [:checks] setup = [ClaimHarness] begin
    using Base.Threads
    if nthreads() < 2
        @test_skip "needs >= 2 julia threads"
    else
        n = 1024; iters = 40
        D = Dict{DataType, Any}()
        for T in ELTYPES
            LU, ip = getri_input(T, n)
            P.set_num_threads(1); ref = getri_run!(similar(LU), LU, ip)
            P.set_num_threads(nthreads())
            D[T] = (LU, ip, ref)
        end
        # Both tasks are sticky to this thread, so each is suspended inside its arena scope while the
        # other runs.
        pairs = ((Float64, Float32), (Float64, ComplexF64), (Float32, ComplexF32), (ComplexF64, ComplexF32))
        @testset "$A + $B" for (A, B) in pairs
            j0 = alljobs(); l0 = P._MT_LOST[]
            ta = @async loop_getri(D[A]..., iters)
            tb = @async loop_getri(D[B]..., iters)
            wa = fetch(ta); wb = fetch(tb)
            @test (wa, wb) == (0, 0)
            ia = findfirst(==(A), ELTYPES); ib = findfirst(==(B), ELTYPES)
            dj = alljobs() .- j0
            # WITNESSES that the loops overlapped and reached the pool: some job was published, and a
            # caller lost the claim to the other type's task. Which task wins is the scheduler's: the
            # task holding the claim across its join lets the other run a whole serial iteration, so
            # one loop can finish while the other still publishes.
            if !sme(A) && !sme(B)
                @test dj[ia] + dj[ib] > 0
                @test P._MT_LOST[] > l0
            end
        end
        # CONTROLS. Same type on both tasks: one pool, one claim.
        @testset "same type F64 + F64" begin
            l0 = P._MT_LOST[]
            ta = @async loop_getri(D[Float64]..., iters)
            tb = @async loop_getri(D[Float64]..., iters)
            @test (fetch(ta), fetch(tb)) == (0, 0)
            sme(Float64) || @test P._MT_LOST[] > l0
        end
        # Mixed types run one after the other: no overlap, so no loss and every call publishes.
        @testset "mixed, sequential" begin
            j0 = alljobs(); l0 = P._MT_LOST[]
            @test loop_getri(D[Float64]..., iters) == 0
            @test loop_getri(D[Float32]..., iters) == 0
            dj = alljobs() .- j0
            sme(Float64) || @test dj[1] > 0
            sme(Float32) || @test dj[2] > 0
            @test P._MT_LOST[] == l0
        end
        # The comparison can fail: a reference with one bit changed is flagged on every iteration.
        @testset "perturbed reference is flagged" begin
            LU, ip, ref = D[Float64]
            bad = copy(ref); bad[1] = nextfloat(bad[1])
            @test loop_getri(LU, ip, bad, 5) == 5
        end
        P.set_num_threads(1)
    end
end

@testitem "claim: two threads, different element types, threaded iamax shares _REDI without corruption" tags = [:checks] setup = [ClaimHarness] begin
    using Base.Threads, Random
    if nthreads() < 2
        @test_skip "needs >= 2 julia threads"
    else
        mk(T, len, pos) = (x = rand(MersenneTwister(len), T, len); x[pos] = T(10); x)
        x64 = mk(Float64, 1 << 21, 1_500_000)
        x32 = mk(Float32, 1 << 22, 3_100_000)
        P.set_num_threads(1); r64 = P.iamax(x64); r32 = P.iamax(x32)
        P.set_num_threads(nthreads())
        @test (r64, r32) == (1_500_000, 3_100_000)
        iters = 200
        j0 = alljobs(); l0 = P._MT_LOST[]
        t1 = Threads.@spawn loop_iamax(x64, r64, iters)
        t2 = Threads.@spawn loop_iamax(x32, r32, iters)
        @test (fetch(t1), fetch(t2)) == (0, 0)
        dj = alljobs() .- j0
        @test dj[1] > 0 && dj[2] > 0       # both pools published, so both loops used `_REDI`
        @test P._MT_LOST[] > l0            # and the two overlapped
        # CONTROLS: the comparison flags a wrong reference, and a lone caller is unaffected.
        @test loop_iamax(x64, r64 + 1, 3) == 3
        @test loop_iamax(x32, r32, 5) == 0
        P.set_num_threads(1)
    end
end
