// pureblas-cpufreq — a minimal setuid helper for the ONE privileged thing a gate run needs: writing the
// cpufreq knobs. Policy (which clock, does the box hold it, verify afterwards) stays in
// bench/fleet_freqlock.sh, which needs no privileges and which you can read.
//
// Build (STATIC — this is the whole point, see below):
//     CGO_ENABLED=0 go build -trimpath -ldflags='-s -w' -o build/pureblas-cpufreq bench/tools/pureblas-cpufreq.go
//     file build/pureblas-cpufreq        # must say "statically linked"
//     sudo install -o root -g root -m 4755 build/pureblas-cpufreq /usr/local/sbin/
//
// ── WHY GO IS THE RIGHT LANGUAGE FOR THIS PARTICULAR JOB ────────────────────────────────────────────
// The Julia build (bench/tools/cpufreq_helper.jl) works and is trim-safe, but its binary carries a
// RUNPATH into ~/.julia/juliaup/… and resolves SEVEN shared objects from there — libjulia,
// libjulia-internal, libunwind, libstdc++, libgcc_s, libz, libatomic. That directory is writable by the
// invoking user, and a setuid program loading shared objects from a caller-writable path is a direct
// root escalation: replace the .so, run the binary, get root. The kernel strips LD_LIBRARY_PATH and
// LD_PRELOAD for setuid binaries but cannot help with a RUNPATH baked in at link time.
//
// With CGO_ENABLED=0 Go emits a fully static executable: no dynamic loader, no RUNPATH, no libc, nothing
// to substitute. That removes the entire class of problem rather than mitigating it. The C version has
// the same property in practice (it links only libc from /lib, which is root-owned); Go gets there
// without depending on that being true.
//
// One Go-specific caveat, stated rather than assumed: the Go runtime consults a few environment
// variables (GODEBUG, GOMAXPROCS, GOGC). It checks the kernel's AT_SECURE flag and disables the
// environment-driven debug knobs when the binary is setuid, but this program does not depend on that —
// it reads no environment of its own, and every decision below comes from argv alone.
//
// ── WHAT IT DELIBERATELY DOES NOT DO ────────────────────────────────────────────────────────────────
// Each of these is a root hole in a setuid program:
//   - it never runs another program: no shell, no PATH, no exec at all;
//   - it never reads its own environment;
//   - it takes ONE verb from a fixed table plus at most one integer, both validated before any write;
//   - every path is a literal with a bounded CPU index appended;
//   - `pin` is range-checked against the CPU's OWN cpuinfo_min/max, so it cannot be argued into a value
//     the hardware does not advertise.
//
// It grants exactly one capability — changing CPU frequency policy on this machine, which affects other
// workloads on it. It touches nothing else, and nothing it does survives a reboot.
//
// Exit status: 0 success, 1 request rejected, 2 write failure (possibly partially applied — re-run, or
// use `unpin` to restore the hardware's own range).

package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

const maxCPU = 512

const (
	cpuinfoMin = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq"
	cpuinfoMax = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"
	boostPath  = "/sys/devices/system/cpu/cpufreq/boost"
)

var governors = []string{"performance", "powersave", "schedutil", "ondemand", "conservative"}
var pstateModes = []string{"passive", "active", "guided"}

// writeStr writes exactly one value to one sysfs file. No buffering, no append: sysfs wants a single
// write per open, and os.WriteFile does precisely that.
func writeStr(path, val string, quiet bool) bool {
	// 0 perms: the file already exists; sysfs ignores the mode. Truncation is what sysfs expects.
	if err := os.WriteFile(path, []byte(val), 0); err != nil {
		if !quiet {
			fmt.Fprintf(os.Stderr, "pureblas-cpufreq: %v\n", err)
		}
		return false
	}
	return true
}

// readLong reads a small decimal from sysfs. Returns -1 on any failure; every caller treats a
// non-positive result as "unreadable", so there is no error channel worth threading.
func readLong(path string) int64 {
	b, err := os.ReadFile(path)
	if err != nil {
		return -1
	}
	v, err := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
	if err != nil || v <= 0 {
		return -1
	}
	return v
}

// forEachCPU applies val to <cpu N>/cpufreq/<leaf> for every CPU that has one. A CPU without a cpufreq
// directory is skipped silently: that covers offline cores and the non-CPU entries under
// /sys/devices/system/cpu.
func forEachCPU(leaf, val string) int {
	touched, failed := 0, 0
	for i := 0; i < maxCPU; i++ {
		path := "/sys/devices/system/cpu/cpu" + strconv.Itoa(i) + "/cpufreq/" + leaf
		if _, err := os.Stat(path); err != nil {
			continue
		}
		if writeStr(path, val, false) {
			touched++
		} else {
			failed++
		}
	}
	if touched == 0 && failed == 0 {
		fmt.Fprintf(os.Stderr, "pureblas-cpufreq: no cpufreq/%s on any CPU\n", leaf)
		return 1
	}
	if failed > 0 {
		return 2
	}
	return 0
}

func inList(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}

func usage() int {
	fmt.Fprint(os.Stderr,
		"usage: pureblas-cpufreq boost 0|1\n"+
			"       pureblas-cpufreq governor <performance|powersave|schedutil|ondemand|conservative>\n"+
			"       pureblas-cpufreq pstate   <passive|active|guided>\n"+
			"       pureblas-cpufreq pin      <kHz>\n"+
			"       pureblas-cpufreq unpin\n")
	return 1
}

func doUnpin() int {
	lo, hi := readLong(cpuinfoMin), readLong(cpuinfoMax)
	if lo <= 0 || hi <= 0 || lo > hi {
		fmt.Fprintln(os.Stderr, "pureblas-cpufreq: cannot read cpuinfo_min/max_freq")
		return 1
	}
	// Widen the ceiling BEFORE lowering the floor, or a kernel enforcing min <= max rejects one of the
	// two writes and leaves the box half-restored.
	a := forEachCPU("scaling_max_freq", strconv.FormatInt(hi, 10))
	b := forEachCPU("scaling_min_freq", strconv.FormatInt(lo, 10))
	if a != 0 || b != 0 {
		return 2
	}
	return 0
}

func doPin(arg string) int {
	khz, err := strconv.ParseInt(arg, 10, 64)
	if err != nil || khz <= 0 {
		fmt.Fprintln(os.Stderr, "pureblas-cpufreq: pin takes a positive integer in kHz")
		return 1
	}
	// Range-check against what the HARDWARE advertises, not a constant. This is the one place a bad
	// argument could do something surprising, so it is bounded by the CPU itself.
	lo, hi := readLong(cpuinfoMin), readLong(cpuinfoMax)
	if lo <= 0 || hi <= 0 {
		fmt.Fprintln(os.Stderr, "pureblas-cpufreq: cannot read cpuinfo_min/max_freq")
		return 1
	}
	if khz < lo || khz > hi {
		fmt.Fprintf(os.Stderr, "pureblas-cpufreq: %d kHz outside the CPU's own range [%d, %d]\n", khz, lo, hi)
		return 1
	}
	s := strconv.FormatInt(khz, 10)
	// Raise the ceiling first: if the new pin is ABOVE the current max, writing the floor first is
	// rejected for exceeding it.
	a := forEachCPU("scaling_max_freq", s)
	b := forEachCPU("scaling_min_freq", s)
	if a != 0 || b != 0 {
		return 2
	}
	return 0
}

func run(args []string) int {
	if len(args) == 0 {
		return usage()
	}
	verb := args[0]

	if verb == "unpin" {
		if len(args) != 1 {
			return usage()
		}
		return doUnpin()
	}

	if len(args) != 2 {
		return usage()
	}
	arg := args[1]

	switch verb {
	case "boost":
		if arg != "0" && arg != "1" {
			return usage()
		}
		// ABSENT and UNWRITABLE are different failures and must not print the same message. Not every
		// platform exposes this knob (intel_pstate has no_turbo instead) — but running unprivileged also
		// fails here, and reporting that as "no boost on this platform" sends the reader hunting for a
		// hardware difference that does not exist. Found by testing this binary as a normal user.
		if _, err := os.Stat(boostPath); err != nil {
			fmt.Fprintln(os.Stderr, "pureblas-cpufreq: no cpufreq/boost on this platform")
			return 1
		}
		if writeStr(boostPath, arg, false) {
			return 0
		}
		return 2

	case "governor":
		if !inList(governors, arg) {
			return usage()
		}
		return forEachCPU("scaling_governor", arg)

	case "pstate":
		if !inList(pstateModes, arg) {
			return usage()
		}
		// Same distinction as `boost`: find the knob first, then report a write failure as a write
		// failure rather than as a missing driver.
		for _, p := range []string{
			"/sys/devices/system/cpu/amd_pstate/status",
			"/sys/devices/system/cpu/intel_pstate/status",
		} {
			if _, err := os.Stat(p); err != nil {
				continue
			}
			if writeStr(p, arg, false) {
				return 0
			}
			return 2
		}
		fmt.Fprintln(os.Stderr, "pureblas-cpufreq: no amd_pstate/intel_pstate status knob")
		return 1

	case "pin":
		return doPin(arg)
	}
	return usage()
}

func main() {
	os.Exit(run(os.Args[1:]))
}
