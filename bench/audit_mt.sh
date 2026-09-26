#!/usr/bin/env bash
# THE THREADED CACHES' AUDIT CHAIN — one command, fixed order, never a subset.
#
# The serial chain runs inside bench/publish.sh, which globs `plots_data_*` only. Every audit script
# did the same, so the threaded caches carried NO staleness, anchor, clock or freshness check at all —
# while holding 937 cells per box that the gate tools can now read.
#
# WHY A SEPARATE CHAIN RATHER THAN A WIDER GLOB. A threaded anchor problem must not block a serial
# publish: the two sets describe different measurements at different thread counts and fail
# independently. Widening the default glob would couple them, and the first coupled failure would be
# read as a serial regression. So the scripts take explicit files, and this is the file list.
#
# THROTTLE DETECTION DOES EXIST HERE, and step 3 is the whole reason this chain matters. Measured on
# `mt_data_avx512_wintermute`, L1/axpy@1000000:
#
#     aocl_mt / openblas_mt   2013 MHz   (in-window 1917 … 2675)
#     pb / pb_mt              2795 MHz   (in-window 2794 … 2795)
#
# The clock sample is shared by the arms timed in one window, so the THREADED reference window ran 28%
# below the lock while the PB window held it — PB does not thread `axpy`, so it ran one core. The ratio
# at that cell compares two machine states, which the frequency rule calls INVALID rather than noisy.
# Fleet-wide: 204/937 cells on wintermute (worst 45.9%), 7/937 on galen, 0 on neuromancer. The error
# FLATTERS PureBLAS — a throttled reference is a slower reference.
#
# THE MECHANISM IS CACHE-RESIDENT POWER, NOT MEMORY TRAFFIC. Reproduced with a plain 6-thread Julia
# axpy on this box, minimum over the working cores against a 2813000 kHz setpoint:
#
#     L1-resident   32 KB   2632366   -6.4%
#     L2-resident  512 KB   2726951   -3.1%
#     L3-resident    8 MB   2299687   -18.2%     <- axpy@1e6 is 8 MB: this cell
#     DRAM         640 MB   2793602   -0.7%
#
# A DRAM-bound loop is memory-STALLED and draws little power, so it barely moves the clock; a
# cache-resident one retires work continuously on six cores and does. Two alternatives were tested and
# FALSIFIED: an idle or blocked core reporting low (it reports the SETPOINT — measured 2813000 while
# the main thread slept and workers spun), and DRAM-bound traffic (0.7%). Do not re-chase either.
#
# FORWARD CONSEQUENCE for the campaign: once PureBLAS threads `axpy`/`dot`, ITS window will throttle
# the same way, and the comparison becomes fair. Today it is a throttled reference against an
# unthrottled serial PB.
#
# What is genuinely absent is the per-cell IN-WINDOW check for the pb_mt arm: its clock is sampled
# from /proc/self/stat field 39, the MAIN thread's CPU, which spins then yields while the workers
# work. `check_arm_clocks.sh` stands that one down for `mt_data_*` rather than pretending. The
# cross-arm comparison in step 3 does not stand down, and it is what caught the above.
#
# NO PUBLISH STEP. Nothing here renders or writes an artifact. The coverage generators REFUSE a
# threaded cache outright (they rewrite docs/src/coverage.md, which publishes the serial gate).
#
#   bench/audit_mt.sh              # every bench/mt_data_*.txt
#   bench/audit_mt.sh <tol_pct>    # anchor tolerance, default 5
set -u
cd "$(dirname "$0")/.." || exit 2
tol=${1:-5}

mapfile -t files < <(ls bench/mt_data_*.txt 2>/dev/null | grep -v _lite)
[ ${#files[@]} -eq 0 ] && { echo "no threaded cache files found (bench/mt_data_*.txt)"; exit 2; }

echo "threaded caches under audit:"
for f in "${files[@]}"; do printf '   %s\n' "$f"; done

rc=0
step() {
    printf '\n══ %s\n' "$1"; shift
    "$@" || rc=1
}

step "1/5  cell staleness — do these cells predate the code they describe?" \
    bash bench/cache_staleness.sh "${files[@]}"
step "2/5  arm anchors — pb_mt against the threaded vendors (ADVISORY, see that script's header)" \
    bash bench/check_arm_anchors.sh "$tol" "${files[@]}"
# THE STEP THAT MATTERS MOST for a threaded cache, and the reason this is a HARD failure: a cell whose
# PB window and reference window ran at different clocks is not adjudicable at all.
step "3/5  arm clocks — did the PB window and the reference window run at the same clock? (HARD)" \
    bash bench/check_arm_clocks.sh 3 "${files[@]}"
step "4/5  clock outliers — cells whose stamped clock sits apart from the rest of their own cache" \
    bash bench/check_clock_outliers.sh "${files[@]}"
step "5/5  cache freshness — is a remote box's cache newer than the copy here?" \
    bash bench/check_cache_freshness.sh "${files[@]}"

printf '\n'
if [ "$rc" -eq 0 ]; then
    echo "threaded audit: every step clean. The one check that does NOT run is the per-cell in-window"
    echo "clock for pb_mt (it samples the idle main thread); step 3's cross-arm comparison covers the"
    echo "case that actually bites, so a clean run here is a real result."
else
    echo "threaded audit: at least one step reported a problem — read it before scoring any cell."
    echo "A step-3 failure means those cells compare two machine states. Discard them; do not explain"
    echo "them, and do not average them into a group score."
fi
exit "$rc"
