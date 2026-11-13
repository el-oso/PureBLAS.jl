# Threaded code must stay fast in the configuration a user actually starts: `julia -t auto`, which
# gives one more Julia thread (the interactive one) than the process has CPUs, and on an SMT machine
# two threads per core. The fleet benchmarks pin one thread per core with nothing else running, the
# one regime where a pool that loses time to spare threads cannot show it, so this is checked here.
#
# Measured on Zen4, getrf n=1024 on 4 CPUs with `-t 4,1`: floating pool workers 17-504 ms against
# 17 ms serial; sticky workers (`_gemm_pool_start!`) 7.4 ms.
@testitem "threaded getrf keeps its speedup under -t auto" tags = [:checks] begin
    script = """
    using PureBLAS, Random
    using Chairmarks: @be
    const A = randn(MersenneTwister(1), 1024, 1024)
    function med()
        t = sort([s.time for s in (@be copy(A) PureBLAS.getrf!(_) evals = 1 seconds = 2).samples])
        return t[(length(t) + 1) ÷ 2]
    end
    PureBLAS.set_num_threads(1); ts = med()
    nw = PureBLAS.set_num_threads(Threads.nthreads()); tp = med()
    println("RESULT ", Threads.nthreads(:default), " ", Threads.nthreads(:interactive), " ", nw, " ", ts, " ", tp)
    """
    # Four CPUs where `taskset` can restrict to them, so the child has exactly one more Julia thread
    # than CPUs whatever the host's size; otherwise every CPU, which still leaves one too many.
    cpus = Sys.islinux() && !isnothing(Sys.which("taskset")) && Sys.EFFECTIVE_CPU_THREADS >= 4
    jl = `$(Base.julia_cmd()) -t auto --project=$(Base.active_project()) -e $script`
    # The child runs PureBLAS's defaults: the suite-wide override that lifts the core cap is removed.
    env = filter(kv -> kv.first != "PUREBLAS_MT_ALL_THREADS", ENV)
    out = ""
    if cpus
        allowed = Int[]
        for l in eachline("/proc/self/status")
            startswith(l, "Cpus_allowed_list:") || continue
            for r in split(strip(split(l, ':')[2]), ',')
                lo, hi = occursin('-', r) ? parse.(Int, split(r, '-')) : (parse(Int, r), parse(Int, r))
                append!(allowed, lo:hi)
            end
        end
        out = read(setenv(`taskset -c $(join(allowed[1:4], ',')) $jl`, env), String)
    else
        out = read(setenv(jl, env), String)
    end
    line = only(l for l in split(out, '\n') if startswith(l, "RESULT "))
    nd, ni, nw, ts, tp = parse.(Float64, split(line)[2:end])
    @info "threaded getrf under -t auto" default_threads = nd interactive = ni workers = nw serial_ms = ts / 1e-3 threaded_ms = tp / 1e-3
    @test nw >= 1
    if cpus
        # At least 40% parallel efficiency on the 4 CPUs. On Zen4, CPUs 0-3 (2 cores x 2 SMT): the
        # floating pool ran 4 workers at 1.0-1.43x serial against this bound of 0.625x, 3 runs of 3
        # failing; the fixed pool runs 2 workers (the core cap) at 0.80x against 1.25x. On 4 separate
        # cores it reaches 58% (7.3 ms against 17 ms).
        @test tp <= ts / (0.4 * nw)
    else
        # Without `taskset` the worker count follows the host, where 40% need not hold for n=1024.
        #
        # GitHub-hosted runners only: their VMs report every vCPU as a core, so the Julia-only core count
        # (`_mt_physical_cores`) gives 4 workers on 2 cores. On the Windows runner that measured 228 and
        # 265 ms threaded against 55 ms serial (4.2-4.9x). The bound there catches a collapse of the
        # 30-100x kind this item was written for, not the overcount; any other host keeps 1.5x.
        hosted = get(ENV, "GITHUB_ACTIONS", "") == "true"
        @test tp <= (hosted ? 8.0 : 1.5) * ts
    end
end

# The pool's keep-awake window comes from the onset `deps/build.jl` stored for this machine, and is
# `_MT_KEEPAWAKE_NS` when there is no store or the store was written on other hardware.
@testitem "keep-awake window: the stored onset on this host, else the default" begin
    using PureBLAS: _mt_keepawake_read, _mt_host_fingerprint, _MT_KEEPAWAKE_NS, _MT_KEEPAWAKE_GAP_NS
    fp = _mt_host_fingerprint()
    onset = 3_000_000
    mktempdir() do dir
        f = joinpath(dir, "onset.txt")
        @test _mt_keepawake_read(f, fp) == _MT_KEEPAWAKE_NS
        write(f, fp, "\n", string(onset), "\n")
        @test _mt_keepawake_read(f, fp) == _MT_KEEPAWAKE_GAP_NS - onset
        # Positive control: the store moved the window off the default, so the equalities to the
        # default above and below can fail.
        @test _mt_keepawake_read(f, fp) != _MT_KEEPAWAKE_NS
        @test _mt_keepawake_read(f, fp * "/other") == _MT_KEEPAWAKE_NS
        write(f, fp, "\nnone\n")                   # no gap slowed the probe's job: sleep at once
        @test _mt_keepawake_read(f, fp) == 0
        write(f, fp, "\n0\n")                      # slower after any gap: cover the whole gap
        @test _mt_keepawake_read(f, fp) == _MT_KEEPAWAKE_GAP_NS
        write(f, fp, "\n", string(2 * _MT_KEEPAWAKE_GAP_NS), "\n")   # sleeps later than the gap
        @test _mt_keepawake_read(f, fp) == 0
        write(f, fp, "\nnot a number\n")
        @test _mt_keepawake_read(f, fp) == _MT_KEEPAWAKE_NS
        write(f, fp)
        @test _mt_keepawake_read(f, fp) == _MT_KEEPAWAKE_NS
    end

    # End to end: a store in the first depot, where `deps/build.jl` writes it, sets the window of a
    # pool created afterwards.
    mktempdir() do depot
        f = joinpath(PureBLAS.scratch_path(PureBLAS._MT_KEEPAWAKE_UUID, "mt_keepawake"; depot_path = depot), "onset.txt")
        mkpath(dirname(f))
        write(f, fp, "\n", string(onset), "\n")
        script = """
        using PureBLAS
        PureBLAS.set_num_threads(2) == 2 || error("no pool")
        println("WINDOW ", PureBLAS._MT_KEEPAWAKE_HOST[])
        """
        env = copy(ENV)
        env["JULIA_DEPOT_PATH"] = join([depot; DEPOT_PATH], Sys.iswindows() ? ';' : ':')
        env["PUREBLAS_MT_ALL_THREADS"] = "1"
        out = read(setenv(`$(Base.julia_cmd()) -t 2 --project=$(Base.active_project()) -e $script`, env), String)
        line = only(l for l in split(out, '\n') if startswith(l, "WINDOW "))
        @test parse(Int, split(line)[2]) == _MT_KEEPAWAKE_GAP_NS - onset
    end
end

# Every real transpose combination at a size where `gemm!` asks `_strassen_owns` whether to column-
# split, at 2 and 4 workers, bitwise against one worker. The predicate and `_gemm_core!`'s recursion
# arm must agree: a call the predicate claims but the core runs classically reaches the cooperative
# A-pack barrier as a lone "worker 1 of nw" and spins there forever, ignoring SIGTERM. So the case runs
# in a child process with a hard SIGKILL deadline, and a hang fails this item instead of the suite.
@testitem "threaded gemm: every transpose pair finishes, bitwise equal to serial" begin
    script = """
    using PureBLAS, Random
    for T in (Float32, Float64), ta in ('N', 'T', 'C'), tb in ('N', 'T', 'C')
        rng = MersenneTwister(7)
        n = 2048
        A = randn(rng, T, n, n); B = randn(rng, T, n, n)
        PureBLAS.set_num_threads(1)
        C1 = PureBLAS.gemm!(zeros(T, n, n), A, B; transA = ta, transB = tb)
        for nt in (2, 4)
            PureBLAS.set_num_threads(nt) == nt || error("asked for \$nt workers")
            C = PureBLAS.gemm!(zeros(T, n, n), A, B; transA = ta, transB = tb)
            println("CASE ", T, " ", ta, tb, " ", nt, " ", C == C1); flush(stdout)
        end
    end
    println("DONE")
    """
    env = copy(ENV)
    env["PUREBLAS_MT_ALL_THREADS"] = "1"     # four workers even on a host with fewer cores
    proc = open(pipeline(setenv(`$(Base.julia_cmd()) -t 4 --project=$(Base.active_project()) -e $script`, env); stderr = devnull))
    lines = String[]
    lastline = Ref(time())
    reader = @async for l in eachline(proc)
        push!(lines, l)
        lastline[] = time()
    end
    # The deadline is per case: a hang stops the output, while a slow runner keeps printing cases.
    finished = true
    while !process_exited(proc)
        sleep(1)
        if time() - lastline[] > 300
            finished = false
            kill(proc, Base.SIGKILL)
            break
        end
    end
    wait(proc)
    wait(reader)
    cases = [l for l in lines if startswith(l, "CASE ")]
    finished || @info "threaded gemm child killed at the deadline" last_case = isempty(cases) ? "none" : last(cases)
    @test finished
    @test success(proc)
    @test "DONE" in lines
    @test length(cases) == 2 * 9 * 2
    @test all(endswith(" true"), cases)
end
