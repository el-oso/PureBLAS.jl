#!/bin/bash
# Was every cell in this cache measured at the SAME clock the rest of the cache was?
#
# WHY THIS EXISTS, AND WHY THE SIBLING CHECKS CANNOT DO IT. `check_arm_clocks.sh` asks whether the arms
# WITHIN one cell agree, and `check_arm_anchors.sh` asks the same about the calibration workload. Both
# are within-cell questions. A frequency lock that drops PART WAY THROUGH a sweep defeats both: with
# full arms every arm of a cell is measured together, so each cell stays internally consistent and every
# within-cell check passes — while half the file was measured at 2.0 GHz and half at 4.8.
#
# Measured 2026-09-06 on neuromancer, whose cpufreq pin drops when something on its power cluster is
# disconnected (and whose sysfs settings still read LOCKED while the cores run at 4.8 GHz against a
# 2.0 GHz pin). `fleet_refresh.sh` verifies the lock before and after, so the drop was caught — but only
# at the END, which condemned the whole four-hour sweep because nothing said WHEN the clock let go.
#
# It did say, though. `bench/plots.jl` stamps every arm with the clock it actually ran at
# (`arm|date|commit|anchor|khz|flo|fhi|samples`), and on a locked box those figures are tight: measured
# spreads across a whole cache are 2790-2794 MHz on wintermute and 3671-3675 on galen. A cell measured
# after a drop stands out by more than a factor of two. So a mid-sweep drop does NOT require discarding
# the run — it requires FILTERING it, and this script is the filter.
#
#   bench/check_clock_outliers.sh                    # every bench/plots_data_*.txt
#   bench/check_clock_outliers.sh <cache> [more...]
#
# Exit status is the point: non-zero when any cell disagrees with its cache's own modal clock, so it can
# gate a publish rather than merely inform one.
set -u
cd "$(dirname "$0")/.." || exit 2
TOL=${CLOCK_TOL_PCT:-3}          # same tolerance check_arm_clocks.sh uses between arms of one cell
files=("$@")
if [ ${#files[@]} -eq 0 ]; then
    mapfile -t files < <(ls bench/plots_data_*.txt 2>/dev/null | grep -v _lite)
fi
rc=0
for f in "${files[@]}"; do
    [ -f "$f" ] || continue
    echo "── $(basename "$f" .txt | sed 's/^plots_data_//')"
    # Modal clock in MHz over every arm of every cell — the value the box was actually pinned at for
    # the bulk of the run. Mode, not mean: a mean is dragged by the very outliers being looked for.
    mode=$(awk -F'\t' '{for(i=4;i<=NF;i++){n=split($i,p,"|"); if(n>=5 && p[5]+0>0) print int(p[5]/1000)}}' "$f" \
           | sort -n | uniq -c | sort -rn | head -1 | awk '{print $2}')
    if [ -z "$mode" ]; then echo "   (no clock stamps — pre-v3 cache, skipped)"; continue; fi
    out=$(awk -F'\t' -v m="$mode" -v tol="$TOL" '
        {
            for (i = 4; i <= NF; i++) {
                n = split($i, p, "|")
                if (n < 5 || p[5] + 0 <= 0) continue
                mhz = int(p[5] / 1000)
                d = (mhz - m) * 100.0 / m; if (d < 0) d = -d
                if (d > tol) printf "   %s/%s@%s  arm=%s  %d MHz vs %d modal  (%.0f%%)\n", $1, $2, $3, p[1], mhz, m, d
            }
        }' "$f")
    n=$(printf '%s' "$out" | grep -c . || true)
    if [ "$n" -eq 0 ]; then
        echo "   => all cells at the modal clock ${mode} MHz (tolerance ${TOL}%)"
    else
        printf '%s\n' "$out" | head -12
        [ "$n" -gt 12 ] && echo "   … and $((n - 12)) more"
        echo "   => $n arm(s) measured at a DIFFERENT clock than the rest of this cache."
        echo "      Those cells are invalid; the rest of the file is fine. Re-measure just them:"
        printf '%s\n' "$out" | awk '{split($1,a,"/"); split(a[2],b,"@"); print b[1]}' | sort -u | tr '\n' ' ' \
            | sed 's/^/        ops: /; s/ $/\n/'
        rc=1
    fi
done
exit $rc
