#!/usr/bin/env bash
# Run a probe in the warm Revise session — starting that session if it is not up yet.
#
#   bench/probe.sh bench/probes/foo.jl
#
# WHY THIS EXISTS. A cold `julia --project=bench` pays the full load+compile tax on every call:
# measured 200-311 s on this package against ~3 s for a method-body edit applied by Revise in a live
# session. `bench/hot.jl` has provided that session for a long time, and the rule to use it has been in
# CLAUDE.md just as long — and it still gets forgotten, because using it means remembering a fifo, a
# mask, a thread count and a completion marker. This script is that knowledge, so "run a probe" is one
# command that is never cold.
#
# It is idempotent: the first call starts the session, every later call reuses it. Nothing here kills a
# running session — a restart costs the compile tax this exists to avoid, and is needed only when the
# include graph changes (`bench/probe.sh --restart`).
set -u
cd "$(dirname "$0")/.." || exit 2

FIFO=${PBHOT_FIFO:-/tmp/pbhot.fifo}
LOG=${PBHOT_LOG:-/tmp/pbhot.log}
NT=${PBHOT_THREADS:-6}
TIMEOUT=${PBHOT_TIMEOUT:-1800}

# One CPU per physical core plus one spare, matching `bench/fleet_refresh.sh` and the mask recorded on
# `_ARM_PB_MT`. An unknown host runs unpinned rather than guessing a topology — as does every macOS
# host, which has no CPU affinity API at all (`taskset` does not exist and `thread_policy_set` is
# inert on Apple silicon; QoS is the only scheduler steer there).
case "$(hostname)" in
    wintermute)  MASK=0,2,4,6,8,10,1 ;;
    galen)       MASK=6,7,8,9,10,11,18 ;;
    neuromancer) MASK=0,1,2,3,4,5,6 ;;
    *)           MASK="" ;;
esac
# `PBHOT_MASK` overrides it, and `PBHOT_MASK=none` runs unpinned. The mask is itself a subject of
# investigation — the per-box default puts SEVEN CPUs over SIX physical cores, so one core always holds
# two of the process's threads, and whether those two are both pool WORKERS is a question the default
# cannot be used to answer. Changing it requires a restart, which is why this exists rather than a flag.
if [ -n "${PBHOT_MASK:-}" ]; then
    [ "$PBHOT_MASK" = none ] && MASK="" || MASK="$PBHOT_MASK"
fi

# ── CHECK THE CLOCK BEFORE MEASURING, AND RELOCK IF THE BOX LETS US ─────────────────────────────────
# A probe that informs a decision is a measurement, and a measurement off the base-clock pin is INVALID
# rather than merely noisy. `fleet_refresh.sh` already refuses to sweep an unlocked box; a probe had no
# such check, so it was left to whoever remembered.
#
# It was not remembered. neuromancer — a LAPTOP, where things happen around it — dropped its pin three
# times in one session, reporting `boost=0` and `pin=2000-2000` while running at 4763 MHz, because the
# SETTINGS look locked and only the achieved-under-load figure sees it. One whole probe was measured at
# 2.4x the pinned clock and had to be discarded.
#
# `CORE` MATTERS: the default in fleet_freqlock.sh is 8, which is neuromancer's bench core, so verifying
# on another box without setting it measures a core the work does not run on. One per box, matching
# `MASK` above and `fleet_refresh.sh`.
#
# RELOCKING NEEDS NO SUDO ON NEUROMANCER ONLY, via the root-owned setuid helper at
# /usr/local/sbin/pureblas-cpufreq (built from bench/tools, Go, installed there and nowhere else). That is
# the difference between "ask a human and wait" and "relock the box and carry on" mid-session — so where
# the helper exists this relocks and continues, and where it does not it says what to run and refuses.
#
# `PBHOT_NOLOCK=1` skips it, for a probe that is not a measurement (a correctness check, a code dump).
case "$(hostname)" in
    wintermute)  VCORE=2 ;;
    galen)       VCORE=6 ;;
    neuromancer) VCORE=8 ;;
    *)           VCORE="" ;;
esac
_lockok() { CORE="${VCORE:-8}" bash "$(dirname "$0")/fleet_freqlock.sh" verify 2>&1 | grep -q '✅'; }
if [ -z "${PBHOT_NOLOCK:-}" ] && [ -n "$VCORE" ] && [ -r /sys/devices/system/cpu/amd_pstate/status ]; then
    if ! _lockok; then
        echo "[probe.sh] clock NOT locked on $(hostname) (bench core $VCORE) — relocking" >&2
        CORE="$VCORE" bash "$(dirname "$0")/fleet_freqlock.sh" lock >&2 2>&1 || true
        if ! _lockok; then
            echo "[probe.sh] STILL not locked. A probe measured off the base-clock pin is INVALID, not" >&2
            echo "           noisy — refusing. Run: sudo bench/fleet_freqlock.sh lock" >&2
            echo "           (no sudo needed only where the setuid helper is installed.)" >&2
            echo "           Set PBHOT_NOLOCK=1 if this probe is not a measurement." >&2
            exit 2
        fi
        echo "[probe.sh] relocked." >&2
    fi
fi

# `pgrep -x julia` matches the interpreter ONLY, never this wrapper — a `pgrep -f bench/hot.jl` also
# matches the shell running this script, which makes the session look alive when it is not.
#
# The command line comes from `ps`, not from `/proc/<pid>/cmdline`: macOS has no `/proc`, so the
# read fails for every candidate, `hotpid` reports no session, and the wrapper starts a second one
# and then declares it dead. `ps -ww -o command=` prints the full, untruncated command line on both
# BSD and GNU `ps`.
hotpid() {
    local p
    for p in $(pgrep -x julia 2>/dev/null); do
        if ps -ww -o command= -p "$p" 2>/dev/null | grep -q 'bench/hot\.jl'; then
            echo "$p"; return 0
        fi
    done
    return 1
}

start_session() {
    rm -f "$FIFO" "$LOG"
    mkfifo "$FIFO" || return 1
    # shellcheck disable=SC2086  # MASK is deliberately unquoted: empty must expand to NO taskset
    nohup bash -lc "cd '$PWD' && ${MASK:+taskset -c $MASK }julia -t $NT --project=bench bench/hot.jl '$FIFO'" \
        > "$LOG" 2>&1 < /dev/null &
    local waited=0
    until grep -q 'HOT-READY' "$LOG" 2>/dev/null; do
        sleep 5; waited=$((waited + 5))
        if [ "$waited" -ge "$TIMEOUT" ]; then
            echo "hot session did not become ready within ${TIMEOUT}s; last log:" >&2
            tail -20 "$LOG" >&2; return 1
        fi
    done
    echo "[probe.sh] hot session ready (pid $(hotpid), -t $NT${MASK:+, cpus $MASK})" >&2
}

if [ "${1:-}" = "--restart" ]; then
    pid=$(hotpid) && kill "$pid" 2>/dev/null && sleep 2
    start_session || exit 2
    exit 0
fi
if [ "${1:-}" = "--status" ]; then
    if pid=$(hotpid); then echo "hot session up, pid $pid, fifo $FIFO"; else echo "no hot session"; fi
    exit 0
fi

probe="${1:?usage: bench/probe.sh <probe.jl> | --restart | --status}"
[ -f "$probe" ] || { echo "no such probe: $probe" >&2; exit 2; }

hotpid >/dev/null || start_session || exit 2

# Read only what THIS run appends. The session log is append-only across probes, so a marker from an
# earlier probe would otherwise satisfy the wait immediately — which has already produced a "finished"
# report for a probe that had not started.
mark=$(wc -l < "$LOG")
echo "$probe" > "$FIFO"
waited=0
until tail -n "+$((mark + 1))" "$LOG" | grep -q '<<<HOT-DONE'; do
    if ! hotpid >/dev/null; then
        echo "[probe.sh] hot session died; output so far:" >&2
        tail -n "+$((mark + 1))" "$LOG"; exit 2
    fi
    sleep 3; waited=$((waited + 3))
    if [ "$waited" -ge "$TIMEOUT" ]; then
        echo "[probe.sh] probe still running after ${TIMEOUT}s — output so far:" >&2
        tail -n "+$((mark + 1))" "$LOG"; exit 3
    fi
done
tail -n "+$((mark + 1))" "$LOG"
