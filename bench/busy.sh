#!/usr/bin/env bash
# Is ANY measurement or test in flight, on ANY fleet box? Exit 0 = everything idle, safe to edit src or
# start a run. Exit 1 = something is running; do not touch src, do not start a probe.
#
# WHY THIS EXISTS. Three separate process failures on 2026-09-19, all preventable by one honest check:
#
#   1. Edited `src/` twice while `Pkg.test()` was in flight. The rule is already in my notes — it cost
#      two full suite runs once before. A mid-run edit does not usually corrupt anything; what it
#      destroys is ATTRIBUTION. You can no longer say which tree produced the result, so the run has to
#      be thrown away either way. Cheaper to check first.
#   2. Ran probes and a hot session on wintermute WHILE its reference sweep was measuring. The sweep's
#      own contention guard only checks at STARTUP, so a job that appears later is invisible to it —
#      214 of 1874 reference arms came out 4-10% slow, and those cells are now caveated forever.
#   3. Watched the wrong PID. `pgrep -f "bench/plots.jl"` MATCHES ITS OWN WRAPPER and matched a shell
#      that had already exited, so a finished-looking check reported a run that was still going, and a
#      still-running check reported one that had died. Every process test here uses `ps`+`awk` on the
#      resolved juliaup path, never `pgrep -f` on a pattern that can match the asker.
#
# Usage:
#   bash bench/busy.sh          # print what is running; exit 1 if anything is
#   bash bench/busy.sh --quiet  # exit code only
# SCOPE MATTERS, and conflating the two scopes is its own bug. A LOCAL benchmark only needs this box
# idle — another machine cannot contend for these cores. A FLEET sweep needs all three. Checking all
# three for a local measurement blocks work for no reason and trains you to ignore the guard, which is
# worse than not having it.
#   --local   this box only          (a probe, a benchmark, editing src while a LOCAL run is going)
#   default   every box              (a fleet sweep, or before syncing the fleet)
set -u
QUIET=0; SCOPE=fleet
for a in "$@"; do
    case "$a" in
        --quiet) QUIET=1 ;;
        --local) SCOPE=local ;;
    esac
done
say() { [ $QUIET -eq 1 ] || printf '%s\n' "$*"; }

# One matcher, used identically everywhere: a real julia binary running a project, excluding this
# script's own pipeline. `-C native` children (precompile workers) count as busy too — they are the
# part that actually steals the cores.
JMATCH='/juliaup/julia-[^ ]*/bin/julia'
count_local() { ps -eo command | awk -v m="$JMATCH" '$0 ~ m && $0 !~ /JuliaMCP/ && $0 !~ /awk/ {n++} END{print n+0}'; }
count_remote() { ssh "$1" "ps -eo command | awk -v m='$JMATCH' '\$0 ~ m && \$0 !~ /JuliaMCP/ && \$0 !~ /awk/ {n++} END{print n+0}'" 2>/dev/null || echo "?"; }

busy=0
l=$(count_local)
if [ "$l" -gt 0 ]; then
    busy=1
    say "BUSY  local: $l julia process(es)"
    [ $QUIET -eq 1 ] || ps -eo etime,command | awk -v m="$JMATCH" '$0 ~ m && $0 !~ /JuliaMCP/ && $0 !~ /awk/ {printf "        %s  %.100s\n", $1, substr($0, index($0,$2))}'
else
    say "idle  local"
fi
for b in $([ "$SCOPE" = local ] && echo "" || echo "galen neuromancer"); do
    r=$(count_remote "$b")
    if [ "$r" = "?" ]; then
        say "UNREACHABLE  $b — treat as BUSY; a box you cannot see is not a box you can measure on"
        busy=1
    elif [ "$r" -gt 0 ]; then
        say "BUSY  $b: $r julia process(es)"
        busy=1
    else
        say "idle  $b"
    fi
done

if [ $busy -eq 0 ]; then
    say ""
    say "ALL IDLE — safe to edit src/ or start a measurement."
    exit 0
fi
say ""
say "SOMETHING IS RUNNING. Do NOT edit src/ (it destroys the run's attribution) and do NOT start a"
say "probe (it contends and silently biases the measurement). Wait, or kill only the PIDs you started."
exit 1
