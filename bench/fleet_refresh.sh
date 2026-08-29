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
# ── MODE: `pb` (default) or `full` ───────────────────────────────────────────────────────────────────
#   bench/fleet_refresh.sh          # arms=pb — reuse cached reference arms (the normal, cheap refresh)
#   bench/fleet_refresh.sh full     # all three arms per cell, in ONE machine state (the repair mode)
#
# WHEN `pb` IS WRONG, AND IT IS NOT RARE. `arms=pb` rewrites only the PB field and keeps the cached
# OpenBLAS/AOCL arms — correct, and the standing rule, PROVIDED the box is in the same machine state it
# was in when those references were measured. The per-cell `anchor` is the field that records whether it
# was. When it is not, every ratio in the file divides two machine states instead of two kernels.
#
# Measured, twice, both self-inflicted:
#   galen  2026-08-27 — a whole-cache `arms=pb` refresh left 616 of 863 cells (71%) anchor-mismatched.
#   zen5   2026-08-27 — the same, 126 cells mismatched, pb arms all at one commit against references
#                       cached FOUR DAYS earlier from two different runs. On a laptop, whose thermal and
#                       load state moves between days, a 4-day-old reference is a bad bet.
# Both were repaired by a FULL-ARMS sweep, which is the documented exception to "never re-measure a
# reference": the rule protects reference arms from pointless churn, not from a state mismatch that has
# already invalidated them. Verify with `bench/check_arm_anchors.sh` after either mode.
MODE="${1:-pb}"
case "$MODE" in
    pb)   ARMSARG="arms=pb" ;;
    full) ARMSARG="" ;;
    *)    echo "usage: $0 [pb|full]   (pb = reuse cached reference arms; full = re-measure all arms)"; exit 2 ;;
esac

echo "=== pinning sweep to core $CORE ($(hostname)), mode=$MODE ==="
[ "$MODE" = full ] && echo "=== FULL ARMS: all three arms per cell in one machine state (anchors match by construction) ==="
echo "=== PRE-LOCK ==="; bash bench/fleet_freqlock.sh verify 2>&1 | tail -2

# A GROUP THAT DOES NOT LAND MUST BE LOUD. This loop used to pipe each run through `tail -4` and move
# on, so a group that died took its exit status with it (the pipeline reports tail's status, not
# julia's) and the refresh still printed "REFRESH DONE". Measured 2026-08-29: wintermute's L1 group
# crashed and neuromancer's CLP never ran, and BOTH boxes reported success — 112 cells silently stayed
# at the previous commit and were published. `bench/cache_staleness.sh` caught it only afterwards.
#
# Two failure modes, deliberately handled differently:
#   contention — plots.jl refuses when a foreign process is >= 25% CPU, and the agent driving this
#                trips that itself at startup. Transient, so RETRY rather than lose the group.
#   anything else — a crash or a real error. Report it and keep going so one bad group does not cost
#                the other seven, but remember it and exit non-zero at the end.
FAILED=""
for g in L1 L2 L3 LP CL1 CL2 CL3 CLP; do
    echo "=== group $g ==="
    ok=0
    for try in 1 2 3 4 5 6; do
        # Capture, THEN tail — piping julia straight into `tail` reports tail's exit status, which is
        # how the silent partial refresh happened in the first place.
        # shellcheck disable=SC2086  # ARMSARG is deliberately unquoted: empty must expand to NO argument
        out=$(taskset -c "$CORE" "$JL" --project=bench bench/plots.jl bench group=$g $ARMSARG nodraw 2>&1)
        st=$?
        printf '%s\n' "$out" | tail -4
        if [ $st -eq 0 ]; then ok=1; break; fi
        if printf '%s' "$out" | grep -q "REFUSING to benchmark"; then
            echo "    group $g: box contended, retry $try in 90s"; sleep 90; continue
        fi
        echo "    group $g: FAILED (exit $st, not contention) — see output above"; break
    done
    [ $ok -eq 1 ] || FAILED="$FAILED $g"
done
echo "=== POST-LOCK ==="; bash bench/fleet_freqlock.sh verify 2>&1 | tail -2
if [ -n "$FAILED" ]; then
    echo "=== REFRESH INCOMPLETE — these groups did NOT land:$FAILED"
    echo "    Their cells still carry the PREVIOUS commit. Re-run them before publishing:"
    for g in $FAILED; do echo "      taskset -c $CORE $JL --project=bench bench/plots.jl bench group=$g $ARMSARG nodraw"; done
    echo "    Then confirm with: bench/cache_staleness.sh"
    exit 1
fi
echo "=== REFRESH DONE ==="
