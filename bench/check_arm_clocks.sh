#!/bin/bash
# Were the PB arm and the REFERENCE arms of a cell measured at the same clock?
#
# WHY THIS EXISTS, AND WHY freqgate.jl CANNOT DO IT. `bench/freqgate.jl` compares each cell's clock to
# the CACHE HEADER's own achieved clock, so it detects cells that left a lock the rest of the sweep
# held. It says so itself: "a run that floated for its WHOLE duration has a floated header too, and
# nothing here will flag it." That is exactly what happened on 2026-08-18 — neuromancer's full PB-only
# refresh verified `passive/boost=0/1978 MHz` at launch, then ran the entire sweep BOOSTING at
# 4688-4772 MHz against reference arms cached at 1982 MHz. Every ratio was inflated ~2.3x; Zen5 dznrm2
# "improved" 1.79 -> 3.62 and dzasum 0.985 -> 2.31 on BLAS-1 kernels that had not been touched. Nothing
# in the pipeline objected, because the header floated with the cells.
#
# THE SIGNAL IT USES INSTEAD is cross-ARM and immune to that: under `arms=pb` the reference arms are
# OLD records carrying the clock they were measured at, and the pb arm is fresh. If a cell's pb clock
# differs from its own reference clocks by more than the tolerance, the two arms describe different
# machine states and their RATIO is meaningless — regardless of what any header says.
#
# Not a replacement for `fleet_freqlock.sh verify`, which is still the only pre-run check. This is the
# post-hoc one that gates a publish.
#
#   bench/check_arm_clocks.sh [tol_pct]      # default 3; every bench/plots_data_*.txt
set -u
cd "$(dirname "$0")/.." || exit 2
tol=${1:-3}
mapfile -t files < <(ls bench/plots_data_*.txt 2>/dev/null | grep -v _lite)
[ ${#files[@]} -eq 0 ] && { echo "no cache files found"; exit 2; }

bad=0
for f in "${files[@]}"; do
    printf '── %s\n' "$(basename "$f" .txt | sed 's/^plots_data_//')"
    out=$(awk -F'\t' -v TOL="$tol" '
        /^#/ { next }
        NF >= 4 {
            pb = 0; ref = 0; refname = ""
            for (i = 4; i <= NF; i++) {
                n = split($i, a, "|")
                if (n < 6) continue
                fq = a[n-1] + 0
                if (fq <= 0) continue
                if (a[1] == "pb") pb = fq
                else if (ref == 0) { ref = fq; refname = a[1] }
            }
            if (pb > 0 && ref > 0) {
                d = (pb - ref) / ref * 100; if (d < 0) d = -d
                tot++
                if (d > TOL) {
                    off++
                    if (off <= 3) printf "   %s/%s@%s  pb=%.0fMHz vs %s=%.0fMHz  (%.1f%%)\n", $1, $2, $3, pb/1000, refname, ref/1000, d
                    badop[$1 "/" $2] = 1; badgrp[$1] = 1     # scope for a TARGETED re-measure
                } else ok++
                if (d > worst) { worst = d }
            }
        }
        END {
            if (tot == 0) { print "   no cells carry both a pb and a reference clock — cannot check"; exit 0 }
            if (off == 0) { printf "   => all %d cells within %s%% (worst %.1f%%)\n", tot, TOL, worst; exit 0 }
            printf "   => %d/%d cells clock-mismatched (%d ok), worst %.1f%% (tolerance %s%%)\n", off, tot, ok, worst, TOL
            # THE POINT OF THE PER-CELL CLOCK: re-measure ONLY what is broken. A lock that floats
            # part-way through a sweep leaves most cells VALID; condemning the whole cache and
            # re-sweeping it wastes hours and is what this field exists to prevent.
            no = 0; for (o in badop) no++
            ng = 0; gl = ""; for (g in badgrp) { ng++; gl = gl " " g }
            printf "   affected: %d op(s) in %d group(s):%s\n", no, ng, gl
            printf "   targeted re-measure (groups, fewest julia startups):\n     for g in%s; do julia --project=bench bench/plots.jl bench group=$g arms=pb nodraw; done\n", gl
            printf "   per-op instead (finest scope, one startup each):\n    "
            for (o in badop) { split(o, q, "/"); printf " op=%s", q[2] }
            printf "\n"
            exit 1
        }' "$f")
    st=$?
    printf '%s\n' "$out"
    [ $st -ne 0 ] && bad=1
done

if [ $bad -ne 0 ]; then
    cat <<'MSG'

CLOCK-MISMATCHED CELLS PRESENT — the PB arm and its reference arms were measured at different clocks,
so those ratios compare two machine states and are INVALID (freq rule: discard, do not explain).
Re-lock the box (`sudo bench/fleet_freqlock.sh lock`), confirm it HOLDS under load, and re-measure:
    for g in L1 L2 L3 LP CL1 CL2 CL3 CLP; do
        julia --project=bench bench/plots.jl bench group=$g arms=pb nodraw
    done
MSG
fi
exit $bad
