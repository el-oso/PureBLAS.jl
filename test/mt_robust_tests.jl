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
        @test tp <= 1.5 * ts
    end
end

# Worker pinning (`_mt_pin_threads!`): decided once, when the pool is created, so every case runs in its
# own child process. The child reports the decision and each default-pool thread's CPU mask, read on
# that thread from /proc/thread-self.
@testitem "worker threads are pinned on a restricted CPU set, and only there" tags = [:checks] begin
    using PureBLAS
    script = """
    using PureBLAS
    if get(ENV, "PREPIN", "") == "1"       # a host that pins its own main thread first
        c = ccall(:sched_getcpu, Cint, ()); m = zeros(UInt64, 16); m[c ÷ 64 + 1] = UInt64(1) << (c % 64)
        ccall(:sched_setaffinity, Cint, (Cint, Csize_t, Ptr{UInt64}), 0, sizeof(m), m)
    end
    PureBLAS.set_num_threads(Threads.nthreads())
    println("MSG ", PureBLAS._MT_PINNED[])
    for tid in Threads.threadpooltids(:default)
        t = Task(() -> strip(match(r"Cpus_allowed_list:\\s*(.*)", read("/proc/thread-self/status", String))[1]))
        t.sticky = true; ccall(:jl_set_task_tid, Cint, (Any, Cint), t, tid - 1); schedule(t)
        println("MASK ", tid, " ", fetch(t))
    end
    """
    linux = Sys.islinux() && !isnothing(Sys.which("taskset")) && isfile("/sys/devices/system/cpu/online")
    if !linux
        @test_skip "needs Linux and taskset"
    else
        P = PureBLAS
        allowed = reduce(union, P._mt_task_masks())
        online = P._mt_parse_cpulist(read("/sys/devices/system/cpu/online", String))
        sib(c) = P._mt_parse_cpulist(read("/sys/devices/system/cpu/cpu$c/topology/thread_siblings_list", String))
        cores = sort!(unique([sort!(collect(intersect(sib(c), allowed))) for c in allowed]); by = first)
        function child(cpus, env...)
            jl = `$(Base.julia_cmd()) -t 3,0 --project=$(Base.active_project()) -e $script`
            cmd = isnothing(cpus) ? jl : `taskset -c $(join(cpus, ',')) $jl`
            e = filter(kv -> kv.first ∉ ("PUREBLAS_MT_ALL_THREADS", "PUREBLAS_PIN_THREADS"), ENV)
            out = read(setenv(cmd, merge(e, Dict(env))), String)
            msg = only(l[5:end] for l in split(out, '\n') if startswith(l, "MSG "))
            masks = Dict(parse(Int, split(l)[2]) => P._mt_parse_cpulist(split(l)[3]) for l in split(out, '\n') if startswith(l, "MASK "))
            return msg, masks
        end
        if length(cores) < 3
            @test_skip "needs three cores in this process's CPU set"
        else
            three = reduce(vcat, cores[1:3])
            # Restricted set: the two worker threads (2 and 3) each on one whole core, never the same one;
            # thread 1 created the pool and keeps the whole set.
            msg, m = child(three)
            @test startswith(msg, "on")
            @test m[2] == Set(cores[2]) && m[3] == Set(cores[3])
            @test m[1] == Set(three)
            # Opt-out.
            msg, m = child(three, "PUREBLAS_PIN_THREADS" => "0")
            @test startswith(msg, "off (PUREBLAS_PIN_THREADS=0")
            @test all(==(Set(three)), values(m))
            # A host that already pinned a thread: left alone.
            msg, m = child(three, "PREPIN" => "1")
            @test startswith(msg, "off (a thread of this process already has its own CPU mask")
            if allowed == online
                # Whole machine: not pinned by default, pinned when asked.
                msg, m = child(nothing)
                @test startswith(msg, "off (the process may use every CPU")
                @test all(==(online), values(m))
                msg, m = child(nothing, "PUREBLAS_PIN_THREADS" => "1")
                @test startswith(msg, "on")
                @test m[2] == Set(cores[2]) && m[3] == Set(cores[3])
            end
        end
    end
end
