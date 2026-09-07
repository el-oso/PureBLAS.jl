#!/usr/bin/env bash
# TARGETED SWEEP — re-measure ONE op across the whole fleet at ONE revision, then publish.
#
#   bench/sweep_op.sh geev              # the fleet
#   bench/sweep_op.sh geev galen        # a subset, space-separated
#   bench/sweep_op.sh geev --no-publish # measure + pull, stop before publishing
#
# WHY THIS EXISTS. A full sweep is hours; iterating on one routine through it wastes most of that on
# rows that did not change. This measures one op everywhere and leaves every other row exactly as it
# was — which is only honest if the table SAYS which revision each row describes, so `coverage_ops.jl`
# now emits a "swept at" column per row (one short SHA when the whole row agrees, a loud `mixed` when
# it does not). A full sweep is then simply the case where every row carries the same hash.
#
# ALL ARMS, NOT `arms=pb`. The standing rule is to reuse cached reference arms, and it is right for a
# broad refresh — but a targeted sweep exists to adjudicate a MARGINAL cell, and a cached reference
# measured under a different machine state cannot do that. Measured 2026-09-07: an `arms=pb` run of
# `geev` on neuromancer came back with the harness's own verdict "MACHINE-STATE DRIFT 9.6% … gaps
# smaller than this are NOT adjudicable", which is useless for a cell sitting at 0.93. `op=` re-measures
# every arm inside one window, which is exactly the exception the rule carves out.
#
# REFUSES TO SWEEP AN UNLOCKED BOX. A ratio from a floating clock is not a gate number; the methodology
# says discard such a run rather than rationalise it, so this declines to produce one in the first place.
set -euo pipefail
cd "$(dirname "$0")/.."

OP="${1:-}"
[ -n "$OP" ] || { echo "usage: bench/sweep_op.sh <op> [box ...] [--no-publish]"; exit 2; }
shift

PUBLISH=1
BOXES=()
for a in "$@"; do
    case "$a" in
        --no-publish) PUBLISH=0 ;;
        -*) echo "unknown flag: $a"; exit 2 ;;
        *) BOXES+=("$a") ;;
    esac
done
[ ${#BOXES[@]} -gt 0 ] || BOXES=(galen wintermute neuromancer)

SHA="$(git rev-parse HEAD)"
SHORT="$(git rev-parse --short HEAD)"
SELF="$(hostname)"

# The dev box is a fleet member too, but it is not ssh-reachable from itself and it IS the working tree:
# it needs no sync, and its commands run locally. Everything else goes over ssh with `cd` baked in and
# `bash -lc`, because a non-login ssh shell does not have juliaup's `julia` on PATH.
run_on() {   # run_on <box> <command…>
    local b="$1"; shift
    if [ "$b" = "$SELF" ]; then
        bash -lc "cd '$PWD' && $*"
    else
        ssh "$b" "bash -lc 'cd ~/Documents/claude/PureBLAS.jl && $*'"
    fi
}

# The tree must BE the commit it will stamp. `fleet_sync.sh` refuses a non-ancestor, but uncommitted
# work never reaches the fleet at all, so a dirty tree means the boxes measure something else.
if ! git diff --quiet -- src bench test juliac; then
    echo "REFUSING: uncommitted changes under src/bench/test/juliac — the fleet would measure a"
    echo "different tree than the SHA the cells will be stamped with. Commit or stash first:"
    git --no-pager diff --stat -- src bench test juliac
    exit 1
fi
if ! git merge-base --is-ancestor "$SHA" "$(git rev-parse @{u} 2>/dev/null || echo "$SHA")" 2>/dev/null; then
    echo "note: HEAD may not be pushed; fleet_sync.sh will fail if the boxes cannot fetch it."
fi

echo "══ targeted sweep: op=$OP  at $SHORT  on ${BOXES[*]}"

# ── 1. every box must be frequency-locked, VERIFIED UNDER LOAD ───────────────────────────────────────
# Settings that merely read locked are not enough: neuromancer has been observed reading `boost=0` while
# its cores ran far above the pin after a power-source change (kb/memory `neuromancer-lock-drops`), and
# only the achieved-under-load figure sees it. `fleet_freqlock.sh verify` measures cycles with perf.
echo "══ 1  frequency lock"
FAILED=0
for b in "${BOXES[@]}"; do
    out="$(run_on "$b" "bash bench/fleet_freqlock.sh verify" 2>&1 || true)"
    ach="$(printf '%s' "$out" | sed -n 's/.*achieved under load = \([0-9]*\) MHz.*/\1/p' | head -1)"
    boost="$(printf '%s' "$out" | sed -n 's/.*boost=\([01]\).*/\1/p' | head -1)"
    pin="$(run_on "$b" "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq" 2>/dev/null || echo 0)"
    pin_mhz=$(( pin / 1000 ))
    if [ -z "$ach" ] || [ "$boost" != "0" ] || [ "$pin_mhz" -eq 0 ]; then
        echo "  $b  LOCK UNVERIFIABLE — $(printf '%s' "$out" | tr '\n' ' ')"; FAILED=1; continue
    fi
    # within 12%, the same tolerance fleet_freqlock.sh asserts after locking
    lo=$(( pin_mhz * 88 / 100 )); hi=$(( pin_mhz * 112 / 100 ))
    if [ "$ach" -lt "$lo" ] || [ "$ach" -gt "$hi" ]; then
        echo "  $b  LOCK FAIL — achieved ${ach} MHz vs pin ${pin_mhz} MHz"; FAILED=1
    else
        echo "  $b  lock OK  ${ach} MHz (pin ${pin_mhz})"
    fi
done
if [ "$FAILED" -ne 0 ]; then
    echo
    echo "ABORT: not sweeping an unlocked box — the ratios would not be gate numbers."
    echo "Lock it with:  ssh <box> 'cd ~/Documents/claude/PureBLAS.jl && sudo bench/fleet_freqlock.sh lock'"
    exit 1
fi

# ── 2. every box AT the revision the cells will claim ────────────────────────────────────────────────
echo "══ 2  sync to $SHORT"
for b in "${BOXES[@]}"; do
    if [ "$b" = "$SELF" ]; then
        echo "  $b  is the working tree (no sync)"
    else
        bash bench/fleet_sync.sh "$b" "$SHA" >/dev/null || { echo "  $b  SYNC FAILED"; exit 1; }
        echo "  $b  at $SHORT"
    fi
done

# ── 3. measure, ALL ARMS, one op, one window per box ─────────────────────────────────────────────────
echo "══ 3  measure op=$OP"
for b in "${BOXES[@]}"; do
    echo "  ── $b"
    run_on "$b" "JULIA_NUM_PRECOMPILE_TASKS=1 taskset -c 8 julia --project=bench bench/plots.jl bench op=$OP" \
        2>&1 | grep -iE "^(L1|L2|L3|LP|CL1|CL2|CL3|CLP) +$OP|DRIFT|NOT ADJUDICABLE|ABORT|ERROR" || true
done

# ── 4. pull the caches back ──────────────────────────────────────────────────────────────────────────
# The boxes own their caches; publishing reads the LOCAL copies, and publish.sh's step 1a2 refuses if a
# local copy is older than the box that produced it. Pull before publishing, not after.
echo "══ 4  pull caches"
for b in "${BOXES[@]}"; do
    [ "$b" = "$SELF" ] && { echo "  $b  local (no pull)"; continue; }
    for f in $(ssh "$b" 'ls ~/Documents/claude/PureBLAS.jl/bench/plots_data_*.txt 2>/dev/null | xargs -n1 basename' 2>/dev/null); do
        scp -q "$b:~/Documents/claude/PureBLAS.jl/bench/$f" "bench/$f" && echo "  $b  $f"
    done
done

# ── 5. publish (rebuilds BOTH reference views, tables, and re-verifies) ──────────────────────────────
if [ "$PUBLISH" -eq 1 ]; then
    echo "══ 5  publish"
    bash bench/publish.sh
else
    echo "══ 5  skipped (--no-publish); run bench/publish.sh when ready"
fi
