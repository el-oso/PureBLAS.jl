#!/usr/bin/env bash
# Refresh every PB arm on ONE box, group by group, reusing the cached OpenBLAS/AOCL arms.
#
# WHY GROUP-BY-GROUP AND NOT ONE FULL RUN: a full `bench arms=pb` with no op=/group= is REFUSED by
# design — a full run REPLACES the cache and would drop the reference arms entirely. Only op=/group=
# runs merge per arm. (publish.sh says this in its refusal message; it is not a workaround.)
#
# The lock is verified BEFORE and AFTER. A gate measurement is only valid under a verified lock, and
# on at least one fleet box the lock has been observed to drop — so "it was locked when I started" is
# not evidence it was locked at the end.
set -u
cd "$(dirname "$0")/.."
JL="${JULIA:-julia}"
echo "=== PRE-LOCK ==="; bash bench/fleet_freqlock.sh verify 2>&1 | tail -2
for g in L1 L2 L3 LP CL1 CL2 CL3 CLP; do
    echo "=== group $g ==="
    "$JL" --project=bench bench/plots.jl bench group=$g arms=pb nodraw 2>&1 | tail -4
done
echo "=== POST-LOCK ==="; bash bench/fleet_freqlock.sh verify 2>&1 | tail -2
echo "=== REFRESH DONE ==="
