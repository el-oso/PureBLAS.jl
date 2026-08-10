# PERSISTENT REVISE SESSION — one Julia process reused across src/ edits.
#
# WHY. Editing `src/` invalidates the PureBLAS pkgimage, so every fresh probe launch pays a full
# precompile (measured 311 s on wintermute, 2026-08-09). Three probes = three precompiles, and that
# dominates an iteration far more than the benchmark does. Revise recompiles only the changed methods,
# so the same edit-measure loop costs well under a second.
#
# USAGE
#   mkfifo /tmp/pbhot.fifo
#   julia --project=bench bench/hot.jl /tmp/pbhot.fifo > /tmp/pbhot.log 2>&1 &
#   echo bench/probes/some_probe.jl > /tmp/pbhot.fifo     # runs it; output lands in the log
#   echo EXIT                       > /tmp/pbhot.fifo
# Each command is answered with a `<<<HOT-DONE ...>>>` sentinel so a watcher can tell when it finished.
#
# LIMITS, stated so they are not rediscovered the hard way:
#   * Revise cannot redefine a `const` (e.g. `_TRSM_TINY_PERV`, the `_GT_*` tuning consts). Editing one
#     of those needs a fresh session — the sentinel prints a warning if the include throws on it.
#   * A long-lived process accumulates heap and code; that is fine for A/B probes (both arms live in the
#     SAME process, which is what makes them controlled) but GATE numbers still come from a standalone
#     `bench/plots.jl` run, whose methodology owns provenance and cache writing.
#   * Reference arms are NEVER measured here. They live in the v3 cache; probes read them or do without.
using Revise
using PureBLAS, LinearAlgebra, Printf, Statistics

const FIFO = ARGS[1]
BLAS.set_num_threads(1)
# OPEN THE FIFO ONCE, AS A STREAM — this is what makes Revise work incrementally here.
# The loop used to call `open(readline, FIFO)` per command. That blocks inside libuv WITHOUT yielding, so
# Julia's scheduler never ran Revise's async file-watcher task: the revision queue stayed empty and a bare
# `Revise.revise()` returned "success" having applied nothing. Silently — which cost a 30-minute A/B run
# on stale code, caught only because that probe carried a witness counter. (`Revise.retry()` cannot help
# either; it retries FAILED revisions, and nothing was ever queued to fail.)
# Reading from a persistent stream goes through async I/O and DOES yield, so the watcher runs, only
# CHANGED files are queued, and `revise()` is incremental — sub-second instead of the ~62 s whole-module
# `revise(PureBLAS)` fallback, and no session restart (~200-300 s precompile) either.
# The handle is opened read+write so WE hold a writer: with readers only, every client disconnect EOFs the
# stream and spins the loop.
const FIFO_R = open(FIFO, read = true, write = true)
# Preload the shared measurement helper ONCE, tracked. Probes then guard with
# `isdefined(Main, :Measure) || include(...)`, so they never rebuild the module and never invalidate
# the `tstat` binding Main holds.
Revise.includet(joinpath(@__DIR__, "measure.jl"))
# mtime high-water mark for the src/ scan in the loop below (see the HOT-REVISE note there).
const SRC = joinpath(@__DIR__, "..", "src")
const LAST_SRC = Ref(maximum(mtime(f) for (r, _, fs) in walkdir(SRC) for f in joinpath.(r, fs)
                             if endswith(f, ".jl"); init = 0.0))
println("<<<HOT-READY>>> pid=", getpid())
flush(stdout)

while true
    cmd = try
        strip(readline(FIFO_R))                   # yields to the scheduler, so Revise's watcher runs
    catch e
        @warn "fifo read failed" exception = e
        break
    end
    isempty(cmd) && continue
    cmd == "EXIT" && break
    t0 = time()
    try
        # PICK UP src/ EDITS. With the stream-based FIFO read above, the watcher task actually runs, so
        # this is the normal INCREMENTAL path: only changed methods are re-evaluated, typically <1 s.
        # It is verified, not assumed — the mtime high-water mark says whether src/ changed, and if it did
        # but Revise applied nothing, that is the silent-staleness failure and we say so loudly and fall
        # back to the whole-module re-eval rather than running a probe against stale code.
        newest = maximum(mtime(f) for (r, _, fs) in walkdir(SRC) for f in joinpath.(r, fs)
                         if endswith(f, ".jl"); init = 0.0)
        if newest > LAST_SRC[]
            nq = length(Revise.revision_queue)
            Revise.revise()
            if nq == 0                                    # watcher never saw it ⇒ incremental path failed
                println("<<<HOT-REVISE watcher missed the edit, full module re-eval>>>")
                Revise.revise(PureBLAS)
            else
                println("<<<HOT-REVISE ", nq, " file(s)>>>")
            end
            # No `@elapsed` here, deliberately: raw clocks are banned everywhere under bench/ and that
            # lint has NO escape hatch by design, because every previous exemption was taken behind a
            # plausible-sounding reason exactly like "it is only a progress message". The duration is
            # already in the <<<HOT-DONE …>>> line below, so timing it here buys nothing anyway.
            LAST_SRC[] = newest
            flush(stdout)
        end
        # `Base.include` for the PROBE, deliberately. A probe is a script: its whole content is
        # top-level side effects, and `Revise.includet` does NOT re-execute top-level code on a
        # subsequent call — it only re-tracks method definitions. Dispatching an already-tracked probe
        # through includet returned "ok" in 0.0s having run NOTHING, silently.
        # `includet` is still right for a helper MODULE (see the measure.jl preload below), where the
        # problem it solves is that a plain re-`include` builds a NEW module object and invalidates the
        # binding Main imported from the previous one.
        Base.include(Main, abspath(cmd))
        @printf("<<<HOT-DONE ok %s %.1fs>>>\n", cmd, time() - t0)
    catch e
        showerror(stdout, e, catch_backtrace())
        println()
        @printf("<<<HOT-DONE FAILED %s %.1fs>>>\n", cmd, time() - t0)
    end
    flush(stdout)
end
println("<<<HOT-EXIT>>>")
