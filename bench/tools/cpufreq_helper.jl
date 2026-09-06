# pureblas-cpufreq — a minimal setuid helper for the ONE privileged thing a gate run needs: writing the
# cpufreq knobs. Policy (which clock, does the box hold it, verify afterwards) stays in
# bench/fleet_freqlock.sh, which needs no privileges and which you can read.
#
# Build:  julia bench/tools/build_cpufreq_helper.jl      → bench/tools/build/pureblas-cpufreq
# Install: sudo install -o root -g root -m 4755 bench/tools/build/pureblas-cpufreq /usr/local/sbin/
#
# ── WHY THIS SHAPE ──────────────────────────────────────────────────────────────────────────────────
# NOT a sudoers rule on fleet_freqlock.sh: that script lives in a git repo the agent can write to, so
# whitelisting it by path would hand over root — edit the script, run it, done.
# NOT `chmod u+s` on a shell script: Linux ignores the setuid bit on `#!` files (a deliberate defence
# against a race between the kernel's check and the interpreter re-opening the file), so it would
# silently do nothing.
#
# ── WHAT IT DELIBERATELY DOES NOT DO ────────────────────────────────────────────────────────────────
# Every one of these is a root hole in a setuid program:
#   * it never runs another program — no shell, no PATH, no IFS, no LD_* surface;
#   * it never reads its environment;
#   * it takes ONE verb from a fixed table plus at most one integer, both validated before any write;
#   * every path is a literal with a bounded CPU index appended;
#   * `pin` is range-checked against the CPU's OWN cpuinfo_min/max, so it cannot be argued into a value
#     the hardware does not advertise.
# It grants exactly one capability — changing CPU frequency policy on this machine, which affects other
# workloads on it. It touches nothing else and nothing it does survives a reboot.
#
# ── TRIM NOTES (why the code looks like this) ───────────────────────────────────────────────────────
# Built with `juliac --trim=safe`, which is stricter than ordinary Julia:
#   * All I/O goes through `ccall` to libc (open/read/write/close) rather than Base's `open`/`read`.
#     Base's file layer drags in IOStream, buffering and error paths that --trim cannot resolve, and it
#     is also the wrong tool here: sysfs wants one exact write per file, not a buffered stream.
#   * NO multi-piece string interpolation anywhere. Eager interpolation lowers to
#     `print_to_string(::String, ::Vararg{Any})`, which `--trim=safe` REJECTS — this is the same
#     constraint src/arena.jl documents for its own error paths. Messages are emitted as literal pieces,
#     and integers via a hand-rolled decimal formatter into a fixed buffer.
#   * Concrete types throughout, no `Any` containers, no dynamic dispatch.

module CpuFreqHelper

const O_WRONLY = Cint(1)
const O_RDONLY = Cint(0)
const O_CLOEXEC = Cint(0o2000000)

# ── libc I/O. `Cstring` conversion of a Julia String is safe here: every path is built into a `String`
# that stays rooted for the duration of the ccall. ───────────────────────────────────────────────────
@inline function _open(path::String, flags::Cint)
    return @ccall open(path::Cstring, flags::Cint)::Cint
end
@inline _close(fd::Cint) = @ccall close(fd::Cint)::Cint

function write_str(path::String, val::String, quiet::Bool)::Bool
    fd = _open(path, O_WRONLY | O_CLOEXEC)
    if fd < 0
        quiet || (err_lit("cannot open "); err_lit(path); err_nl())
        return false
    end
    n = GC.@preserve val @ccall write(fd::Cint, pointer(val)::Ptr{UInt8}, sizeof(val)::Csize_t)::Cssize_t
    _close(fd)
    if n < 0
        quiet || (err_lit("cannot write "); err_lit(path); err_nl())
        return false
    end
    return true
end

# Read a small decimal integer from a sysfs file. Returns -1 on any failure — the callers all treat a
# non-positive result as "unreadable", so there is no error channel to thread.
function read_long(path::String)::Int
    fd = _open(path, O_RDONLY | O_CLOEXEC)
    fd < 0 && return -1
    buf = Vector{UInt8}(undef, 64)
    n = GC.@preserve buf @ccall read(fd::Cint, pointer(buf)::Ptr{UInt8}, 63::Csize_t)::Cssize_t
    _close(fd)
    n <= 0 && return -1
    v = 0
    seen = false
    @inbounds for i in 1:n
        c = buf[i]
        if c >= UInt8('0') && c <= UInt8('9')
            v = v * 10 + Int(c - UInt8('0'))
            seen = true
        else
            break
        end
    end
    return seen ? v : -1
end

# ── stderr, one literal at a time. No interpolation: see the trim notes above. ───────────────────────
err_lit(s::String) = (GC.@preserve s @ccall write(Cint(2)::Cint, pointer(s)::Ptr{UInt8}, sizeof(s)::Csize_t)::Cssize_t; nothing)
err_nl() = err_lit("\n")

# Decimal formatting without `string(::Int)` — hand-rolled so no formatting machinery is pulled in.
function err_int(v::Int)
    if v < 0
        err_lit("-")
        v = -v
    end
    buf = Vector{UInt8}(undef, 24)
    i = 24
    if v == 0
        buf[i] = UInt8('0'); i -= 1
    end
    while v > 0
        buf[i] = UInt8('0') + UInt8(v % 10)
        v = div(v, 10)
        i -= 1
    end
    s = String(buf[(i + 1):24])
    err_lit(s)
    return nothing
end

# ── CPU iteration. A CPU with no cpufreq directory is skipped silently: that covers offline cores and
# the non-CPU entries under /sys/devices/system/cpu. ─────────────────────────────────────────────────
const MAXCPU = 512

cpu_path(i::Int, leaf::String) = string("/sys/devices/system/cpu/cpu", i, "/cpufreq/", leaf)

function for_each_cpu(leaf::String, val::String)::Cint
    touched = 0
    failed = 0
    for i in 0:(MAXCPU - 1)
        p = cpu_path(i, leaf)
        # access(F_OK) — cheaper than open, and it is how "is there a cpufreq dir here" is asked.
        (@ccall access(p::Cstring, Cint(0)::Cint)::Cint) == 0 || continue
        if write_str(p, val, false)
            touched += 1
        else
            failed += 1
        end
    end
    if touched == 0 && failed == 0
        err_lit("pureblas-cpufreq: no cpufreq/")
        err_lit(leaf)
        err_lit(" on any CPU")
        err_nl()
        return Cint(1)
    end
    return failed > 0 ? Cint(2) : Cint(0)
end

const GOVERNORS = ("performance", "powersave", "schedutil", "ondemand", "conservative")
const PSTATE_MODES = ("passive", "active", "guided")

in_list(list::NTuple{N, String}, s::String) where {N} = any(==(s), list)

function usage()::Cint
    err_lit("usage: pureblas-cpufreq boost 0|1\n")
    err_lit("       pureblas-cpufreq governor <performance|powersave|schedutil|ondemand|conservative>\n")
    err_lit("       pureblas-cpufreq pstate   <passive|active|guided>\n")
    err_lit("       pureblas-cpufreq pin      <kHz>\n")
    err_lit("       pureblas-cpufreq unpin\n")
    return Cint(1)
end

# Parse a positive decimal integer, rejecting anything else. No `parse(Int, s)`: its error path is not
# trim-friendly and a silent 0 here would be worse than a rejection.
function parse_pos_int(s::String)::Int
    isempty(s) && return -1
    v = 0
    for c in codeunits(s)
        (c >= UInt8('0') && c <= UInt8('9')) || return -1
        v = v * 10 + Int(c - UInt8('0'))
        v > 100_000_000 && return -1        # far above any real kHz; also bounds the loop
    end
    return v > 0 ? v : -1
end

const CPUINFO_MIN = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq"
const CPUINFO_MAX = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"

function do_unpin()::Cint
    lo = read_long(CPUINFO_MIN)
    hi = read_long(CPUINFO_MAX)
    if lo <= 0 || hi <= 0 || lo > hi
        err_lit("pureblas-cpufreq: cannot read cpuinfo_min/max_freq\n")
        return Cint(1)
    end
    # Widen the ceiling BEFORE lowering the floor, or a kernel enforcing min <= max rejects one of the
    # two writes and leaves the box half-restored.
    a = for_each_cpu("scaling_max_freq", int_str(hi))
    b = for_each_cpu("scaling_min_freq", int_str(lo))
    return (a != 0 || b != 0) ? Cint(2) : Cint(0)
end

function int_str(v::Int)::String
    buf = Vector{UInt8}(undef, 24)
    i = 24
    x = v
    if x == 0
        buf[i] = UInt8('0'); i -= 1
    end
    while x > 0
        buf[i] = UInt8('0') + UInt8(x % 10)
        x = div(x, 10)
        i -= 1
    end
    return String(buf[(i + 1):24])
end

function do_pin(arg::String)::Cint
    khz = parse_pos_int(arg)
    if khz < 0
        err_lit("pureblas-cpufreq: pin takes a positive integer in kHz\n")
        return Cint(1)
    end
    # Range-check against what the HARDWARE advertises, not a constant. This is the one place a bad
    # argument could do something surprising, so it is bounded by the CPU itself.
    lo = read_long(CPUINFO_MIN)
    hi = read_long(CPUINFO_MAX)
    if lo <= 0 || hi <= 0
        err_lit("pureblas-cpufreq: cannot read cpuinfo_min/max_freq\n")
        return Cint(1)
    end
    if khz < lo || khz > hi
        err_lit("pureblas-cpufreq: ")
        err_int(khz)
        err_lit(" kHz outside the CPU's own range [")
        err_int(lo)
        err_lit(", ")
        err_int(hi)
        err_lit("]\n")
        return Cint(1)
    end
    s = int_str(khz)
    # Raise the ceiling first: if the new pin is ABOVE the current max, writing the floor first is
    # rejected for exceeding it.
    a = for_each_cpu("scaling_max_freq", s)
    b = for_each_cpu("scaling_min_freq", s)
    return (a != 0 || b != 0) ? Cint(2) : Cint(0)
end

function run(args::Vector{String})::Cint
    isempty(args) && return usage()
    verb = args[1]

    if verb == "unpin"
        length(args) == 1 || return usage()
        return do_unpin()
    end

    length(args) == 2 || return usage()
    arg = args[2]

    if verb == "boost"
        (arg == "0" || arg == "1") || return usage()
        # ABSENT and UNWRITABLE are different failures and must not print the same message. Not every
        # platform exposes this knob (intel_pstate has no_turbo instead) — but running unprivileged also
        # fails here, and calling that "no boost on this platform" sends the reader hunting for a
        # hardware difference that does not exist.
        p = "/sys/devices/system/cpu/cpufreq/boost"
        if (@ccall access(p::Cstring, Cint(0)::Cint)::Cint) != 0
            err_lit("pureblas-cpufreq: no cpufreq/boost on this platform\n")
            return Cint(1)
        end
        return write_str(p, arg, false) ? Cint(0) : Cint(2)
    end

    if verb == "governor"
        in_list(GOVERNORS, arg) || return usage()
        return for_each_cpu("scaling_governor", arg)
    end

    if verb == "pstate"
        in_list(PSTATE_MODES, arg) || return usage()
        # Same distinction as `boost`: find the knob first, then report a write failure as a write
        # failure rather than as a missing driver.
        for p in ("/sys/devices/system/cpu/amd_pstate/status",
                  "/sys/devices/system/cpu/intel_pstate/status")
            (@ccall access(p::Cstring, Cint(0)::Cint)::Cint) == 0 || continue
            return write_str(p, arg, false) ? Cint(0) : Cint(2)
        end
        err_lit("pureblas-cpufreq: no amd_pstate/intel_pstate status knob\n")
        return Cint(1)
    end

    verb == "pin" && return do_pin(arg)
    return usage()
end

end # module

# juliac's entry point must be `Main.main` — a `@main` nested inside a module is invisible to it
# ("To generate an executable a `@main` function must be defined"), so the binding lives out here and
# forwards into the module. `args` excludes the program name.
function (@main)(args::Vector{String})::Cint
    return CpuFreqHelper.run(args)
end
