#!/usr/bin/env bash
# Re-measure the PB arm of every op the tiny-n trsv change can touch, on one fleet box.
# Reference arms come from the box's cache and are NEVER re-run (standing rule).
#
# TWO FAILURE MODES THIS GUARDS, both of which have produced a structurally-complete log that measured
# nothing:
#   1. bare `julia` is not on the non-login ssh PATH (juliaup) — asserted before the loop, not after.
#   2. `op=` takes a SINGLE op compared with `==`; a comma-separated list silently matches nothing and
#      plots.jl then prints a full, plausible, entirely-CACHED table. So the `merged N` line is echoed
#      for every op and N==0 is called out as a failure rather than passed over.
set -u
cd ~/Documents/claude/PureBLAS.jl || exit 1
echo "=== $(hostname) ==="
command -v julia >/dev/null 2>&1 || { echo "FATAL: julia not on PATH"; exit 1; }
julia --version || exit 1
git log --oneline -1
bash bench/fleet_freqlock.sh verify

for op in trsv trsm trsmR getrs potrsL potrsU trtrs; do
    echo "### $op ###"
    out=$(julia --project=bench bench/plots.jl bench arms=pb "op=$op" 2>&1)
    m=$(printf '%s\n' "$out" | grep -oE 'merged [0-9]+ re-measured' | head -1)
    printf '%s\n' "${m:-NO MERGE LINE}"
    case "$m" in
        "merged 0 re-measured") echo "  !! ZERO CELLS MEASURED for $op — op name did not match" ;;
    esac
    printf '%s\n' "$out" | grep -iE 'ERROR|StrictViolation|REFUSING|not adjudicable|DRIFT' | head -3
done
echo "=== DONE $(hostname) ==="
