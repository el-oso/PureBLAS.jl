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
# Preload the shared measurement helper ONCE, tracked. Probes then guard with
# `isdefined(Main, :Measure) || include(...)`, so they never rebuild the module and never invalidate
# the `tstat` binding Main holds.
Revise.includet(joinpath(@__DIR__, "measure.jl"))
println("<<<HOT-READY>>> pid=", getpid())
flush(stdout)

while true
    cmd = try
        strip(open(readline, FIFO))               # blocks until a writer sends a line
    catch e
        @warn "fifo read failed" exception = e
        break
    end
    isempty(cmd) && continue
    cmd == "EXIT" && break
    t0 = time()
    try
        Revise.revise()                            # pick up src/ edits since the last command
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
