# pureblas-cpufreq — a setuid helper for the one privileged step of a gate run

`bench/fleet_freqlock.sh` needs root for exactly one thing: writing the cpufreq knobs. Everything else
it does — choosing the clock, measuring whether the box sustains it, verifying afterwards — needs no
privileges. This directory holds that one privileged step, small enough to read in a sitting.

Three implementations of the same five verbs. Pick one; they are not meant to coexist.

| | build | setuid-safe? |
|---|---|---|
| `pureblas-cpufreq.go` | `CGO_ENABLED=0 go build -trimpath -ldflags='-s -w' -o build/pureblas-cpufreq bench/tools/pureblas-cpufreq.go` | **yes** — fully static, no dynamic loader, no RUNPATH |
| `pureblas-cpufreq.c` | `cc -O2 -Wall -Wextra -Werror -o build/pureblas-cpufreq bench/tools/pureblas-cpufreq.c` | **yes** — links only libc from a root-owned path |
| `cpufreq_helper.jl` | `julia bench/tools/build_cpufreq_helper.jl` (juliac `--trim=safe`) | **NOT AS BUILT** — see below |

    sudo install -o root -g root -m 4755 build/pureblas-cpufreq /usr/local/sbin/

## Why the Julia build cannot be installed setuid as it stands

It compiles and works — 1.8 MB, trim-safe — but the binary carries

    RUNPATH: /home/<user>/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/lib/julia:.../lib

and resolves **seven** shared objects from there: `libjulia`, `libjulia-internal`, `libunwind`,
`libstdc++`, `libgcc_s`, `libz`, `libatomic`. That directory is writable by the invoking user, and a
setuid program that loads shared objects from a caller-writable path is a direct root escalation —
replace the `.so`, run the binary, get root. The kernel strips `LD_LIBRARY_PATH` and `LD_PRELOAD` for
setuid binaries but cannot help with a RUNPATH baked in at link time.

To use it anyway: copy those seven libraries somewhere only root can write and repoint the binary
(`patchelf --set-rpath /usr/local/lib/pureblas-rt`), accepting that you now maintain a private copy of
the Julia runtime. Check with `readelf -d <binary> | grep RUNPATH` and `ldd <binary>` — nothing may
resolve under `$HOME`.

## Why not simpler things

**A sudoers rule on `fleet_freqlock.sh`** — that script lives in a git repo the agent can write to, so
whitelisting it by path hands over root: edit the script, run it, done.

**`chmod u+s` on the shell script** — Linux ignores the setuid bit on `#!` files, deliberately, because
of a race between the kernel's check and the interpreter re-opening the file. It silently does nothing.

## What the binary refuses to do

Each of these is a root hole in a setuid program, so none of them is here: it never execs anything (no
shell, no `PATH`, no `IFS`); it never reads its environment; it takes one verb from a fixed table plus
at most one integer, both validated before any write; every path is a literal with a bounded CPU index
appended; and `pin` is range-checked against the CPU's own `cpuinfo_min/max_freq`, so it cannot be
argued into a value the hardware does not advertise.

It grants exactly one capability — changing CPU frequency policy on this machine, which affects other
workloads on it. Nothing it does survives a reboot.

## Verbs

    pureblas-cpufreq boost 0|1          # /sys/devices/system/cpu/cpufreq/boost
    pureblas-cpufreq governor <name>    # performance | powersave | schedutil | ondemand | conservative
    pureblas-cpufreq pstate   <mode>    # passive | active | guided
    pureblas-cpufreq pin      <kHz>     # scaling_min_freq = scaling_max_freq = kHz, every CPU
    pureblas-cpufreq unpin              # back to cpuinfo_min_freq / cpuinfo_max_freq

Exit status: 0 success, 1 request rejected, 2 write failure (possibly partially applied — re-run, or
`unpin` to restore the hardware's own range). `absent` and `unwritable` are reported as different
failures: an early version said "no cpufreq/boost on this platform" when the real answer was permission
denied, which sends the reader hunting for a hardware difference that does not exist.
