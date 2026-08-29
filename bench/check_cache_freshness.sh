#!/bin/bash
# Is the LOCAL copy of each fleet cache the same as the one on the box that produced it?
#
#   bench/check_cache_freshness.sh          # every bench/plots_data_*.txt with a remote host
#
# WHY THIS EXISTS. `fleet_sync.sh` pushes SOURCE to the boxes. Nothing pulls CACHES back. So a box can
# re-measure, improve, and sit there while the copy on this machine — the one every analysis tool reads
# — stays at whatever it was when it was last copied. Every reader (`gate_gaps.jl`, `cellcycles.jl`,
# `coverage_*.jl`, and any agent) then reports confidently on data that is simply old.
#
# Measured, 2026-08-29, and it wasted real time: the local copy of neuromancer's cache was from 08-27
# (`commit=c5a0692`), while the box's own cache was from 08-28 (`commit=ffbb5e6`). A whole session's
# analysis said Zen5 had 143 red cells, 126 anchor-mismatched cells and 26 reps artifacts. The box
# actually had 115 red, 0 mismatched, 0 artifacts — it had already been repaired. A multi-agent
# classification pass ran on the stale copy too and produced a "fleet 378 -> 339" improvement that was
# entirely fictitious, and a multi-hour repair sweep was launched on a laptop to fix a cache that was
# already correct. None of the existing checks could see it: cache_staleness.sh compares cells to
# SOURCE, check_arm_clocks/anchors compare arms WITHIN a file. Nothing compared the file to its origin.
#
# NOTE ON SCOPE. This compares the header `time=` stamp, which `save_cache` writes at the end of every
# run. It answers "is my copy the newest one that box has produced", not "are the cells any good" —
# that is what the anchor/clock/staleness checks are for. All four are needed.
#
# UNREACHABLE IS NOT CLEAN. If a host cannot be reached the file is reported UNVERIFIED and the exit
# status is nonzero. A freshness check that passes silently when it cannot check anything is the exact
# failure it exists to prevent.
set -u
cd "$(dirname "$0")/.." || exit 2
SELF="$(hostname)"
mapfile -t files < <(ls bench/plots_data_*.txt 2>/dev/null | grep -v _lite)
[ ${#files[@]} -eq 0 ] && { echo "no cache files found"; exit 2; }

stale=0; unver=0; ok=0; local_=0
for f in "${files[@]}"; do
    host=$(head -1 "$f" | tr '\t' '\n' | sed -n 's/^host=//p')
    lt=$(head -1 "$f" | tr '\t' '\n' | sed -n 's/^time=//p')
    lc=$(head -1 "$f" | tr '\t' '\n' | sed -n 's/^commit=//p')
    base=$(basename "$f")
    if [ -z "$host" ]; then
        printf '  %-38s no host= in header — UNVERIFIED\n' "$base"; unver=$((unver + 1)); continue
    fi
    if [ "$host" = "$SELF" ]; then
        printf '  %-38s local box (this file IS the source)  %s\n' "$base" "$lt"; local_=$((local_ + 1)); continue
    fi
    remote=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$host" \
        "head -1 ~/Documents/claude/PureBLAS.jl/bench/$base 2>/dev/null | tr '\t' '\n' | sed -n 's/^time=//p'" 2>/dev/null)
    rc=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$host" \
        "head -1 ~/Documents/claude/PureBLAS.jl/bench/$base 2>/dev/null | tr '\t' '\n' | sed -n 's/^commit=//p'" 2>/dev/null)
    if [ -z "$remote" ]; then
        printf '  %-38s %-12s UNREACHABLE or no cache there — UNVERIFIED\n' "$base" "$host"
        unver=$((unver + 1)); continue
    fi
    # ISO-8601 `YYYY-MM-DDTHH:MM` sorts lexicographically, so a plain string compare is a date compare.
    if [[ "$remote" > "$lt" ]]; then
        printf '  %-38s STALE: local %s (%s) < %s has %s (%s)\n' "$base" "$lt" "$lc" "$host" "$remote" "$rc"
        stale=$((stale + 1))
    else
        printf '  %-38s current (%s)\n' "$base" "$lt"; ok=$((ok + 1))
    fi
done

echo
if [ "$stale" -gt 0 ]; then
    echo "=> $stale cache(s) STALE. Every analysis reading them is reporting on old data."
    echo "   Pull before analysing:  scp <host>:~/Documents/claude/PureBLAS.jl/bench/<file> bench/"
fi
[ "$unver" -gt 0 ] && echo "=> $unver cache(s) UNVERIFIED (host unreachable) — treat as unknown, not as current."
[ "$stale" -eq 0 ] && [ "$unver" -eq 0 ] && echo "=> all $((ok + local_)) cache(s) current w.r.t. their boxes."
exit $(( stale > 0 || unver > 0 ? 1 : 0 ))
