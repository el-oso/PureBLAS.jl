#!/usr/bin/env bash
# FULL all-arms sweep. Needed because n=100 and n=1000 are NEW sizes with no cached reference arms —
# `arms=pb` cannot fill them. Measuring PB and both references in ONE window also makes every cell
# self-consistent, which is what settles the 33 ops whose gate verdict currently rests on a
# non-adjudicable binding cell (17 of them FAILs that may be phantom).
# Group-scoped so each group MERGES per arm; a single full run would replace the cache wholesale.
set -u
cd "$(dirname "$0")/.."
JL="${JULIA:-julia}"
echo "=== PRE-LOCK ==="; bash bench/fleet_freqlock.sh verify 2>&1 | tail -2
for g in L1 L2 L3 LP CL1 CL2 CL3 CLP; do
    echo "=== group $g ==="
    "$JL" --project=bench bench/plots.jl bench group=$g nodraw 2>&1 | tail -3
done
echo "=== POST-LOCK ==="; bash bench/fleet_freqlock.sh verify 2>&1 | tail -2
echo "=== RESWEEP DONE ==="
