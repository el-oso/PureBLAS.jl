/* pureblas-cpufreq — a minimal setuid helper for the ONE privileged thing a gate run needs:
 * writing the cpufreq knobs. Everything else about the lock (deciding the clock, measuring whether the
 * box holds it, verifying afterwards) stays in bench/fleet_freqlock.sh, which needs no privileges and
 * which you can read.
 *
 * WHY A BINARY AND NOT A SUDO RULE ON THE SCRIPT. The script lives in a git repo the agent can write to,
 * so whitelisting it by path in sudoers would hand over root: edit the script, run it, done. A separate
 * root-owned binary that the agent cannot modify closes that.
 *
 * WHY NOT `chmod u+s` ON THE SCRIPT ITSELF. Linux ignores the setuid bit on `#!` files — deliberately,
 * because of a race between the kernel checking the file and the interpreter re-opening it. It would
 * silently do nothing.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO, because a setuid program that does any of it is a root hole:
 *   - it never execs anything, so no PATH, IFS or LD_* attack surface, and no shell is ever involved;
 *   - it ignores its environment completely;
 *   - it takes ONE verb from a fixed table and at most one integer, both validated before use;
 *   - every path is a compile-time constant with a bounded CPU index substituted;
 *   - `pin` is range-checked against the CPU's OWN cpuinfo_min/max, so it cannot be talked into a value
 *     the hardware does not advertise.
 *
 * It grants exactly one capability: changing CPU frequency policy on this machine. That affects other
 * workloads on the box. It touches nothing else, and nothing it does survives a reboot.
 *
 * BUILD AND INSTALL (do this yourself; do not accept a prebuilt binary):
 *     cc -O2 -Wall -Wextra -Werror -o pureblas-cpufreq bench/tools/pureblas-cpufreq.c
 *     sudo install -o root -g root -m 4755 pureblas-cpufreq /usr/local/sbin/pureblas-cpufreq
 *
 * USAGE (all verbs are idempotent):
 *     pureblas-cpufreq boost 0|1          # /sys/devices/system/cpu/cpufreq/boost
 *     pureblas-cpufreq governor <name>    # performance | powersave | schedutil | ondemand
 *     pureblas-cpufreq pstate <mode>      # passive | active | guided   (amd_pstate/intel_pstate)
 *     pureblas-cpufreq pin <kHz>          # scaling_min_freq = scaling_max_freq = kHz, every CPU
 *     pureblas-cpufreq unpin              # back to cpuinfo_min_freq / cpuinfo_max_freq
 *
 * Exit status: 0 on success, 1 on a rejected request, 2 on a write failure (partially applied — re-run
 * or use `unpin`). Anything it could not write is named on stderr rather than being swallowed.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <limits.h>

#define MAXCPU 512
#define BOOST_PATH "/sys/devices/system/cpu/cpufreq/boost"

static int write_str(const char *path, const char *val, int quiet)
{
    int fd = open(path, O_WRONLY | O_CLOEXEC);
    if (fd < 0) {
        if (!quiet) fprintf(stderr, "pureblas-cpufreq: open %s: %s\n", path, strerror(errno));
        return -1;
    }
    ssize_t n = write(fd, val, strlen(val));
    int e = errno;
    close(fd);
    if (n < 0) {
        if (!quiet) fprintf(stderr, "pureblas-cpufreq: write %s: %s\n", path, strerror(e));
        return -1;
    }
    return 0;
}

static long read_long(const char *path)
{
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    char buf[64];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return -1;
    buf[n] = '\0';
    errno = 0;
    char *end = NULL;
    long v = strtol(buf, &end, 10);
    if (errno != 0 || end == buf) return -1;
    return v;
}

/* Apply `val` to <cpufreq>/<leaf> for every present CPU. A CPU without a cpufreq directory is skipped
 * silently (offline cores, and the non-cpufreq entries under /sys/devices/system/cpu). */
static int for_each_cpu(const char *leaf, const char *val)
{
    int touched = 0, failed = 0;
    for (int i = 0; i < MAXCPU; i++) {
        char path[PATH_MAX];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/%s", i, leaf);
        if (access(path, F_OK) != 0) continue;
        if (write_str(path, val, 0) == 0) touched++; else failed++;
    }
    if (touched == 0 && failed == 0) {
        fprintf(stderr, "pureblas-cpufreq: no cpufreq/%s on any CPU\n", leaf);
        return 1;
    }
    return failed ? 2 : 0;
}

/* The governor names the kernel will accept. A free-form string here would be harmless (the kernel
 * rejects unknown governors) but an explicit list documents what this tool is for. */
static const char *GOVERNORS[] = { "performance", "powersave", "schedutil", "ondemand", "conservative", NULL };
static const char *PSTATE_MODES[] = { "passive", "active", "guided", NULL };

static int in_list(const char *const *list, const char *s)
{
    for (int i = 0; list[i]; i++) if (strcmp(list[i], s) == 0) return 1;
    return 0;
}

static int usage(void)
{
    fprintf(stderr,
        "usage: pureblas-cpufreq boost 0|1\n"
        "       pureblas-cpufreq governor <performance|powersave|schedutil|ondemand|conservative>\n"
        "       pureblas-cpufreq pstate   <passive|active|guided>\n"
        "       pureblas-cpufreq pin      <kHz>\n"
        "       pureblas-cpufreq unpin\n");
    return 1;
}

int main(int argc, char **argv)
{
    /* The environment is never read, but clear it anyway: it costs nothing and removes any doubt for
     * someone auditing this, since a setuid program inheriting a hostile environment is the classic
     * hole even when the code itself does not consult it. */
    if (clearenv() != 0) { fprintf(stderr, "pureblas-cpufreq: clearenv failed\n"); return 2; }

    if (argc < 2) return usage();
    const char *verb = argv[1];

    if (strcmp(verb, "unpin") == 0) {
        if (argc != 2) return usage();
        long lo = read_long("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq");
        long hi = read_long("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
        if (lo <= 0 || hi <= 0 || lo > hi) {
            fprintf(stderr, "pureblas-cpufreq: cannot read cpuinfo_min/max_freq\n");
            return 1;
        }
        char slo[32], shi[32];
        snprintf(slo, sizeof(slo), "%ld", lo);
        snprintf(shi, sizeof(shi), "%ld", hi);
        /* Widen the ceiling BEFORE lowering the floor, or a kernel that enforces min <= max rejects
         * one of the two writes and leaves the box half-restored. */
        int a = for_each_cpu("scaling_max_freq", shi);
        int b = for_each_cpu("scaling_min_freq", slo);
        return (a || b) ? 2 : 0;
    }

    if (argc != 3) return usage();
    const char *arg = argv[2];

    if (strcmp(verb, "boost") == 0) {
        if (strcmp(arg, "0") != 0 && strcmp(arg, "1") != 0) return usage();
        /* ABSENT and UNWRITABLE are different failures and must not print the same message. Not every
         * platform exposes this knob (intel_pstate has no_turbo instead) — but running unprivileged
         * also fails here, and reporting that as "no boost on this platform" sends the reader hunting
         * for a hardware difference that does not exist. */
        if (access(BOOST_PATH, F_OK) != 0) {
            fprintf(stderr, "pureblas-cpufreq: no cpufreq/boost on this platform\n");
            return 1;
        }
        return write_str(BOOST_PATH, arg, 0) == 0 ? 0 : 2;
    }

    if (strcmp(verb, "governor") == 0) {
        if (!in_list(GOVERNORS, arg)) return usage();
        return for_each_cpu("scaling_governor", arg);
    }

    if (strcmp(verb, "pstate") == 0) {
        if (!in_list(PSTATE_MODES, arg)) return usage();
        /* Same distinction as `boost`: find the knob first, then report a write failure as a write
         * failure rather than as a missing driver. */
        static const char *const PSTATE_PATHS[] = {
            "/sys/devices/system/cpu/amd_pstate/status",
            "/sys/devices/system/cpu/intel_pstate/status", NULL
        };
        for (int i = 0; PSTATE_PATHS[i]; i++) {
            if (access(PSTATE_PATHS[i], F_OK) != 0) continue;
            return write_str(PSTATE_PATHS[i], arg, 0) == 0 ? 0 : 2;
        }
        fprintf(stderr, "pureblas-cpufreq: no amd_pstate/intel_pstate status knob\n");
        return 1;
    }

    if (strcmp(verb, "pin") == 0) {
        errno = 0;
        char *end = NULL;
        long khz = strtol(arg, &end, 10);
        if (errno != 0 || end == arg || *end != '\0' || khz <= 0) {
            fprintf(stderr, "pureblas-cpufreq: pin takes a positive integer in kHz\n");
            return 1;
        }
        /* Range-check against what the HARDWARE advertises, not against a constant. This is the one
         * place a bad argument could do something surprising, so it is bounded by the CPU itself. */
        long lo = read_long("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq");
        long hi = read_long("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
        if (lo <= 0 || hi <= 0) {
            fprintf(stderr, "pureblas-cpufreq: cannot read cpuinfo_min/max_freq\n");
            return 1;
        }
        if (khz < lo || khz > hi) {
            fprintf(stderr, "pureblas-cpufreq: %ld kHz outside the CPU's own range [%ld, %ld]\n", khz, lo, hi);
            return 1;
        }
        char s[32];
        snprintf(s, sizeof(s), "%ld", khz);
        /* Raise the ceiling first, then the floor: if the new pin is ABOVE the current max, writing
         * scaling_min_freq first would be rejected for exceeding it. */
        int a = for_each_cpu("scaling_max_freq", s);
        int b = for_each_cpu("scaling_min_freq", s);
        return (a || b) ? 2 : 0;
    }

    return usage();
}
