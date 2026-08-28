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

# PIN THE SWEEP TO ONE CORE. `adjudicate.sh` and `screen_cells.sh` have always done this; the group
# sweeps did not, so a two-hour refresh ran with `affinity 0-23` and the scheduler was free to migrate
# it mid-measurement — between cores with different cache state, and on a multi-CCD part between
# entirely different L3 instances.
#
# That is not hypothetical. Measured on galen (5900X, TWO 32 MiB L3s: cores 0-5,12-17 and 6-11,18-23)
# on 2026-08-28: a full-arms sweep and a resident GPU job were BOTH unpinned and had landed on cores 2
# and 3 — the same die, sharing one L3. Against an otherwise identical quiet sweep, 3.7% of cells
# flipped their pass/fail verdict, concentrated near 1.0 where the gate decision is made. Pinning the
# sweep does not remove a co-tenant, but it makes WHICH L3 it shares a decision rather than an
# accident, so a contending job can be `taskset` onto the other die.
#
# Default core per box, matching what fleet_freqlock.sh verifies under load (it must be the SAME core,
# or the lock is verified somewhere the work does not run): wintermute 2, galen 6, neuromancer 8.
# Override with BENCH_CORE=<n>.
case "${BENCH_CORE:-}" in
    "") case "$(hostname)" in
            wintermute)  CORE=2 ;;
            galen)       CORE=6 ;;
            neuromancer) CORE=8 ;;
            *)           CORE=0 ;;
        esac ;;
    *) CORE="$BENCH_CORE" ;;
esac
echo "=== pinning sweep to core $CORE ($(hostname)) ==="
echo "=== PRE-LOCK ==="; bash bench/fleet_freqlock.sh verify 2>&1 | tail -2
for g in L1 L2 L3 LP CL1 CL2 CL3 CLP; do
    echo "=== group $g ==="
    taskset -c "$CORE" "$JL" --project=bench bench/plots.jl bench group=$g arms=pb nodraw 2>&1 | tail -4
done
echo "=== POST-LOCK ==="; bash bench/fleet_freqlock.sh verify 2>&1 | tail -2
echo "=== REFRESH DONE ==="
