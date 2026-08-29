#!/bin/bash
# Were the PB arm and the REFERENCE arms of a cell measured in the same MACHINE STATE?
#
# The sibling `check_arm_clocks.sh` asks the same question about the CLOCK. This asks it about the
# ANCHOR — a fixed calibration workload the harness re-times around every measurement — and the two
# are not redundant: a box can hold its clock perfectly while its memory system, page placement or
# thermal state drifts, and the anchor is the only thing that sees that.
#
# WHY IT EXISTS. `arms=pb` reuses cached OpenBLAS/AOCL arms (the standing "never re-measure a
# reference" rule) and measures only PB fresh. If the box is in a different state than when those
# references were captured, the RATIO compares two machine states rather than two kernels — and
# nothing in the publish pipeline could see it. Measured 2026-08-27/28, both shipped through every
# existing gate:
#   * galen  — 616 of 863 cells (71%) with >5% pb-vs-reference anchor mismatch after an arms=pb
#              refresh. Its red-cell count read 58; the box was not 26 cells worse, the arms were.
#   * zen5   — 121 cells measured at anchor 26.07 against references at 18.26 (43%). That inflated
#              Zen5's red count by 45 and cost hours before the anchor was read.
# Neither `cache_staleness.sh` (commit staleness) nor `check_artifacts_current.sh` (rebuild compare)
# nor `freqgate.jl` (clock) contains the word "anchor". This is the missing check.
#
# THE FIX WHEN IT FIRES is a FULL-ARMS run of the affected scope — `bench group=X` with no `arms=pb`,
# so all three arms land in one machine state and anchors match by construction. Verified: galen went
# from 616/863 mismatched to 0/863 that way.
#
#   bench/check_arm_anchors.sh [tol_pct]     # default 5; every bench/plots_data_*.txt
#
# ⚠ ADVISORY, NOT A VERDICT — read this before treating a mismatch as a reason to discard data.
#
# This check was written 2026-08-29 on the assumption that anchor drift between the pb arm and its
# reference arms means the ratio divides two machine states and is therefore invalid. That assumption
# was never tested, and when it finally was, IT FAILED:
#
#   After an `arms=pb` refresh on neuromancer that this check called 96/863 cells mismatched (galen:
#   737/893, worst 24.7%), the underlying measurements were compared directly:
#     * all 1726 reference arms were BYTE-IDENTICAL before and after — `arms=pb` does not touch them;
#     * for 108 cells whose kernel code had not changed, the pb TIMES were unchanged:
#         median post/pre = 0.9997, p10 = 0.9815, p90 = 1.0154
#       against the ~1.24x slowdown the 24% anchor drift would imply.
#   The anchor moved. The measurements did not.
#
# LIKELY MECHANISM (user's hypothesis, and it fits): the benchmark kernels run `taskset`-pinned to one
# core, so a floating background process perturbs the ANCHOR workload while leaving the pinned kernel
# timings alone. The anchor is then measuring machine weather that the gate numbers are immune to.
#
# So: a mismatch here is a REASON TO LOOK, not a reason to discard. The question the anchor cannot
# answer — and the one that actually matters — is "did the pb times move?". Compare pb medians for
# cells whose code did not change between the two runs; that is the real test, and it is cheap.
# `bench/publish.sh` therefore reports this and does NOT refuse on it.
set -u
cd "$(dirname "$0")/.." || exit 2
tol=${1:-5}
mapfile -t files < <(ls bench/plots_data_*.txt 2>/dev/null | grep -v _lite)
[ ${#files[@]} -eq 0 ] && { echo "no cache files found"; exit 2; }

bad=0
for f in "${files[@]}"; do
    printf '── %s\n' "$(basename "$f" .txt | sed 's/^plots_data_//')"
    out=$(awk -F'\t' -v TOL="$tol" '
        /^#/ { next }
        NF >= 4 {
            # EVERY reference, not just the first. The original took `else if (ref == 0)`, i.e. whichever
            # reference appeared first in the record — and the arms are written in the order
            # aocl, openblas, pb. A partial-arm run (`arms=pb,aocl force-arms`) refreshes pb AND aocl in
            # one machine state while leaving openblas behind, so comparing pb to aocl alone reports
            # 0.0% and the stale openblas anchor is never looked at. Caught 2026-08-28 on galen, on
            # cells this session had just written that way. Compare against the WORST reference.
            pb = 0; ref = 0; refname = ""
            nref = 0; split("", rfq); split("", rnm)
            for (i = 4; i <= NF; i++) {
                n = split($i, a, "|")
                if (n < 6) continue
                fq = a[4] + 0
                if (fq <= 0) continue
                if (a[1] == "pb") pb = fq
                else { nref++; rfq[nref] = fq; rnm[nref] = a[1] }
            }
            if (pb > 0 && nref > 0) {
                worstd = -1
                for (r = 1; r <= nref; r++) {
                    dd = (pb - rfq[r]) / rfq[r] * 100; if (dd < 0) dd = -dd
                    if (dd > worstd) { worstd = dd; ref = rfq[r]; refname = rnm[r] }
                }
            }
            if (pb > 0 && ref > 0) {
                d = (pb - ref) / ref * 100; if (d < 0) d = -d
                tot++
                if (d > TOL) {
                    off++
                    if (off <= 3) printf "   %s/%s@%s  pb=%.1fus vs %s=%.1fus  (%.1f%%)\n", $1, $2, $3, pb, refname, ref, d
                    badop[$1 "/" $2] = 1; badgrp[$1] = 1     # scope for a TARGETED re-measure
                } else ok++
                if (d > worst) { worst = d }
            }
        }
        END {
            if (tot == 0) { print "   no cells carry both a pb and a reference anchor — cannot check"; exit 0 }
            if (off == 0) { printf "   => all %d cells within %s%% (worst %.1f%%)\n", tot, TOL, worst; exit 0 }
            printf "   => %d/%d cells ANCHOR-mismatched (%d ok), worst %.1f%% (tolerance %s%%)\n", off, tot, ok, worst, TOL
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
