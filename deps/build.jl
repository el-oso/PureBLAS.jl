# Pkg runs this after installing, updating or building PureBLAS. It measures, once per machine, how
# long a core stays idle before it enters the idle state that slows the next threaded call (see the
# record above `_MT_KEEPAWAKE_HOST` in src/blas3/gemm.jl), and stores that onset in PureBLAS's
# scratch space, where `set_num_threads` reads it when it creates the worker pool. The measurement
# runs in a child process, because this one is single-threaded, and takes a few seconds.
#
# Set PUREBLAS_SKIP_BUILD_TUNE=1 to skip it (CI does). It never fails the build: on any error or
# timeout it stores nothing and PureBLAS keeps its default window.

using Scratch: get_scratch!, scratch_path

include(joinpath(@__DIR__, "..", "src", "blas3", "keepawake.jl"))

# A limit on a one-off setup cost, not a tuning value: the probe takes seconds, plus the first
# precompile of its few dependencies.
const PROBE_TIMEOUT_S = 300

function measure_keepawake_onset()
    # Half the logical CPUs, at least two: the job then has workers on cores other than the
    # driver's even where the OS places a worker on the driver's SMT sibling, which never idles.
    nt = max(2, Sys.EFFECTIVE_CPU_THREADS ÷ 2)
    probe = joinpath(@__DIR__, "keepawake_probe.jl")
    # Pkg starts this process with -O0, and `julia_cmd` carries that over; the probe times a kernel.
    cmd = `$(Base.julia_cmd()) -O2 --startup-file=no --threads=$nt,0 $probe`
    p = open(pipeline(cmd; stderr))
    if timedwait(() -> process_exited(p), PROBE_TIMEOUT_S) !== :ok
        kill(p)
        println("keep-awake probe: no result within $PROBE_TIMEOUT_S s; nothing stored")
        return nothing
    end
    out = read(p, String)
    if !success(p)
        println("keep-awake probe: exited with code $(p.exitcode); nothing stored")
        return nothing
    end
    # The last line is an onset in ns, or `_MT_KEEPAWAKE_NONE`; returned as the string to store.
    lines = split(out, '\n'; keepempty = false)
    onset = isempty(lines) ? "" : String(strip(last(lines)))
    if onset != _MT_KEEPAWAKE_NONE && isnothing(tryparse(Int, onset))
        println("keep-awake probe: no onset in its output; nothing stored")
        return nothing
    end
    return onset
end

function store_keepawake_onset(onset::String)
    # Attribute the space to this copy of PureBLAS, so `Pkg.gc` keeps it while this version is
    # installed: the active project of a build is a temporary sandbox that is deleted afterwards.
    prev = Base.active_project()
    Base.set_active_project(joinpath(dirname(@__DIR__), "Project.toml"))
    dir = try
        get_scratch!(_MT_KEEPAWAKE_UUID, "mt_keepawake", nothing)
    finally
        Base.set_active_project(prev)
    end
    file = _mt_keepawake_file()
    dirname(file) == dir || error("scratch path mismatch: $dir vs $(dirname(file))")
    tmp = tempname(dir)
    write(tmp, _mt_host_fingerprint(), "\n", onset, "\n")
    mv(tmp, file; force = true)
    w = _mt_keepawake_read(file, _mt_host_fingerprint())
    println("keep-awake onset $onset, window $w ns, stored in $file")
    return nothing
end

if get(ENV, "PUREBLAS_SKIP_BUILD_TUNE", "") == "1"
    println("PUREBLAS_SKIP_BUILD_TUNE=1: keep-awake probe skipped")
else
    try
        onset = measure_keepawake_onset()
        isnothing(onset) || store_keepawake_onset(onset)
    catch e
        println("keep-awake probe failed; nothing stored. PureBLAS keeps its default window.")
        showerror(stdout, e, catch_backtrace())
        println()
    end
end
