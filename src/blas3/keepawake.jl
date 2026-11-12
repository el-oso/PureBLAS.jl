# The pool's keep-awake window, and the on-host measurement it is derived from (see the record above
# `_MT_KEEPAWAKE_HOST` in gemm.jl for what the window does and why).
#
# PLAIN BASE, NO PACKAGE STATE. `deps/build.jl` and `deps/keepawake_probe.jl` include this file
# without loading PureBLAS, so that the writer and the reader share one fingerprint, one file format
# and one window formula. Nothing here may use a PureBLAS name or a macro from a dependency;
# `scratch_path` is resolved by the includer (PureBLAS and `deps/build.jl` both import it from
# Scratch.jl; the probe never calls a function that uses it).

# The window used when the host has not been measured, and the window the derivation below must
# reproduce on Zen3, where the sweep in gemm.jl measured it.
# PDM: Measured — when an idle core enters its deep state is host- and OS-dependent; deps/build.jl measures the onset at install and the window is derived from it.
const _MT_KEEPAWAKE_NS = 10_000_000   # req8-ok: politeness window measured in the sweep recorded in gemm.jl, not a perf tuning

# THE LONGEST IDLE GAP THE WINDOW MUST COVER. A core drops into the damaging idle state only after
# `onset` ns of continuous idleness, and a worker keeps its core busy for the window W after a job, so
# W covers gaps up to W + onset. The gaps that matter are the library's own: the serial panel phases a
# blocked factorization runs between threaded trailing updates. The Zen3 sweep found 10 ms covers
# them with an onset of ~2 ms, so the gap to cover is that sum, and on a host with onset `o` the
# window is that gap minus `o`. Zen3 reproduces 10 ms by construction; a host whose core sleeps later
# needs a shorter window and one whose core sleeps sooner a longer one.
# PDM: Literal — the gap the Zen3 sweep's 10 ms window covers at that box's ~2 ms onset; the panel gaps are not measured anywhere else.
const _MT_KEEPAWAKE_GAP_NS = _MT_KEEPAWAKE_NS + 2_000_000   # req8-ok: Zen3 onset measured in the sweep recorded in gemm.jl

# `onset` is what the probe stores: the shortest idle gap, in ns, after which a threaded job ran
# slower than its own noise (0: slower after any gap), or `nothing` when no gap on its ladder slowed
# the job — then there is nothing to cover and workers sleep at once.
_mt_keepawake_window(::Nothing) = 0
_mt_keepawake_window(onset::Integer) = max(0, _MT_KEEPAWAKE_GAP_NS - max(0, Int(onset)))

# How the store spells "no gap slowed the job".
const _MT_KEEPAWAKE_NONE = "none"

# The machine a stored onset is valid for. Only what Julia itself reports, because the writer cannot
# load PureBLAS's own detection (`_tuning_fingerprint`) without precompiling the whole package. A
# depot copied to other hardware, a different OS or a different core count reads as unmeasured.
function _mt_host_fingerprint()
    ci = Sys.cpu_info()
    model = isempty(ci) ? "" : strip(first(ci).model)
    return string(Sys.ARCH, "/", Sys.KERNEL, "/", Sys.CPU_NAME, "/", Sys.CPU_THREADS, "/", model)
end

# Where `deps/build.jl` stores the measurement, in PureBLAS's scratch space of the first depot.
const _MT_KEEPAWAKE_UUID = Base.UUID("cc9e14db-574f-4602-bf53-1167cc4b26d2")
_mt_keepawake_file() = joinpath(scratch_path(_MT_KEEPAWAKE_UUID, "mt_keepawake"), "onset.txt")

# The store is two lines: the fingerprint, then the onset in ns or `_MT_KEEPAWAKE_NONE`. The window for this host, or
# `_MT_KEEPAWAKE_NS` when there is no store, the store is unreadable or malformed, or it was written
# on another machine.
function _mt_keepawake_read(path::AbstractString, fingerprint::AbstractString)
    isfile(path) || return _MT_KEEPAWAKE_NS
    lines = try
        readlines(path)
    catch e
        e isa Union{SystemError, Base.IOError} || rethrow()
        return _MT_KEEPAWAKE_NS
    end
    (length(lines) == 2 && lines[1] == fingerprint) || return _MT_KEEPAWAKE_NS
    lines[2] == _MT_KEEPAWAKE_NONE && return _mt_keepawake_window(nothing)
    onset = tryparse(Int, lines[2])
    isnothing(onset) && return _MT_KEEPAWAKE_NS
    return _mt_keepawake_window(onset)
end
