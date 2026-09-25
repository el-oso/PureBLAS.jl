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
# `_ARM_PB_MT`. An unknown host runs unpinned rather than guessing a topology.
case "$(hostname)" in
    wintermute)  MASK=0,2,4,6,8,10,1 ;;
    galen)       MASK=6,7,8,9,10,11,18 ;;
    neuromancer) MASK=0,1,2,3,4,5,6 ;;
    *)           MASK="" ;;
esac

# `pgrep -x julia` matches the interpreter ONLY, never this wrapper — a `pgrep -f bench/hot.jl` also
# matches the shell running this script, which makes the session look alive when it is not.
hotpid() {
    local p
    for p in $(pgrep -x julia 2>/dev/null); do
        if tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q 'bench/hot\.jl'; then
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
