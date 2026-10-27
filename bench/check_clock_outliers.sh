#!/bin/bash
# Was every cell in this cache measured at the SAME clock the rest of the cache was?
#
# WHY THIS EXISTS, AND WHY THE SIBLING CHECKS CANNOT DO IT. `check_arm_clocks.sh` asks whether the arms
# WITHIN one cell agree, and `check_arm_anchors.sh` asks the same about the calibration workload. Both
# are within-cell questions. A frequency lock that drops PART WAY THROUGH a sweep defeats both: with
# full arms every arm of a cell is measured together, so each cell stays internally consistent and every
# within-cell check passes — while half the file was measured at 2.0 GHz and half at 4.8.
#
# Measured 2026-09-06 on Zen5, whose cpufreq pin drops when something on its power cluster is
# disconnected (and whose sysfs settings still read LOCKED while the cores run at 4.8 GHz against a
# 2.0 GHz pin). `fleet_refresh.sh` verifies the lock before and after, so the drop was caught — but only
# at the END, which condemned the whole four-hour sweep because nothing said WHEN the clock let go.
#
# It did say, though. `bench/plots.jl` stamps every arm with the clock it actually ran at
# (`arm|date|commit|anchor|khz|flo|fhi|samples`), and on a locked box those figures are tight: measured
# spreads across a whole cache are 2790-2794 MHz on Zen4 and 3671-3675 on Zen3. A cell measured
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
    # while-read, not mapfile: macOS ships bash 3.2, where that builtin does not exist.
    files=()
    while IFS= read -r _ln; do [ -n "$_ln" ] && files+=("$_ln"); done < <(ls bench/plots_data_*.txt 2>/dev/null | grep -v _lite)
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

    # ── THE IN-WINDOW FLOOR, WHICH THE PASS ABOVE CANNOT SEE ──────────────────────────────────────
    # The check above reads field 5, the window's REPRESENTATIVE clock. Fields 6 and 7 carry the range
    # the window actually spanned, and a window whose representative sits at the pin can still have
    # dipped far below it: measured on the Zen4 AVX-512 threaded cache, a `pb_mt` arm with a floor of
    # 1543 MHz against a 3468 MHz modal passed the pass above without comment.
    #
    # COMPARED ACROSS THE ARMS OF ONE CELL, NOT AGAINST THE MODAL CLOCK, and that choice is the whole
    # check. An absolute floor fires on a transient every governor produces at window entry, and the
    # median over a window's samples barely moves for it. What biases a RATIO is the arms of one cell
    # dipping UNEQUALLY — then the two numbers describe two machine states, which the frequency rule
    # calls invalid rather than noisy. Same logic as the cross-arm check on field 5; different field.
    #
    # Why it matters per microarchitecture rather than in general, measured on the 2026-10-26 threaded
    # caches:
    #
    #     uarch              cells   arms disagree >3%     >20%
    #     Zen4 AVX-512         951        19  ( 2%)           4
    #     Zen3 AVX2            937       118  (13%)          72
    #     Zen5 AVX-512         951       486  (51%)         419
    #
    # Over half of the Zen5 part's gate cells compare arms measured at materially different in-window
    # clocks, while the pass above flagged 59 arms there — it is the mobile part of the three, and a
    # chip that holds its pin on the representative sample is not thereby holding it across a
    # six-thread window.
    #
    # Only the arms that enter a gate ratio are compared. `pb` is excluded deliberately: PureBLAS runs
    # it on one core, so its window legitimately holds a clock a six-thread window cannot, and including
    # it would flag every threaded cell of every op PureBLAS does not thread.
    fout=$(awk -F'\t' -v tol="$TOL" '
        {
            delete F; nar = 0
            for (i = 4; i <= NF; i++) {
                n = split($i, p, "|")
                if (n < 8) continue                      # pre-range record: floor unknown, not zero
                if (p[1] != "pb_mt" && p[1] != "openblas_mt" && p[1] != "aocl_mt" && p[1] != "accelerate_mt") continue
                flo = p[6] + 0; if (flo <= 0) continue
                F[p[1]] = flo; nar++
            }
            if (nar < 2) next
            lo = ""; hi = ""; loa = ""; hia = ""
            for (a in F) {
                if (lo == "" || F[a] < lo) { lo = F[a]; loa = a }
                if (hi == "" || F[a] > hi) { hi = F[a]; hia = a }
            }
            seen++
            d = (hi - lo) * 100.0 / hi
            if (d > tol)
                printf "   %s/%s@%s  %s floor %d MHz vs %s %d MHz  (%.0f%%)\n",
                       $1, $2, $3, loa, int(lo/1000), hia, int(hi/1000), d
        }
        # A cache with no threaded arms examines nothing here, and saying "held one clock" for that is
        # the vacuity this chain exists to prevent — a serial cache would report the strongest possible
        # result from zero evidence. Report the count instead, so clean and unexamined never read alike.
        END { if (seen + 0 == 0) print "   __NOCELLS__" }' "$f")
    if printf '%s' "$fout" | grep -q __NOCELLS__; then
        echo "   (no cell has two threaded arms with a window range — nothing to compare, NOT a pass)"
        continue
    fi
    fn=$(printf '%s' "$fout" | grep -c . || true)
    if [ "$fn" -eq 0 ]; then
        echo "   => every gate cell's arms held one clock ACROSS the window (tolerance ${TOL}%)"
    else
        printf '%s\n' "$fout" | head -8
        [ "$fn" -gt 8 ] && echo "   … and $((fn - 8)) more"
        echo "   => $fn cell(s) compare arms that spanned DIFFERENT in-window clocks."
        echo "      Those ratios compare two machine states. Re-measure them with the box quiet and"
        echo "      the lock verified UNDER LOAD (bench/fleet_freqlock.sh verify-mt), not at idle."
        rc=1
    fi
done
exit $rc
