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
# THE SECOND SIGNAL, and it is not cross-arm at all: each cell also records the MINIMUM clock seen
# inside its own timing windows (`flo`). A box can hold its lock at every arm boundary and still spend
# half a cell throttled — six cores under load reach a package power limit one core never does, and the
# pin is not what gives way. Cross-arm cannot see it, because `flo|fhi` is stamped once per cell and
# both arms of a same-run pair therefore carry the identical range and agree perfectly.
#
#   bench/check_arm_clocks.sh [tol_pct]      # default 3; every plots_data_* AND mt_data_* cache
set -u
cd "$(dirname "$0")/.." || exit 2
tol=${1:-3}
shift 2>/dev/null || true
# Explicit caches may follow the tolerance. With none, BOTH tiers are audited — unlike the other
# checks, this one has always scanned `mt_data_*` too, because the cross-arm comparison is the only
# throttle evidence a threaded cache carries and it must not be opt-in.
files=("$@")
if [ ${#files[@]} -eq 0 ]; then
    mapfile -t files < <(ls bench/plots_data_*.txt bench/mt_data_*.txt 2>/dev/null | grep -v _lite)
fi
[ ${#files[@]} -eq 0 ] && { echo "no cache files found"; exit 2; }

bad=0
for f in "${files[@]}"; do
    printf '── %s\n' "$(basename "$f" .txt | sed 's/^plots_data_//')"
    # A threaded cache: its PB arms ran on workers, so the per-cell clock samples a core that was not
    # doing the work. The in-window check below stands down for those; the cross-arm check does not.
    mt=0; case "$f" in *mt_data_*) mt=1 ;; esac
    out=$(awk -F'\t' -v TOL="$tol" -v MT="$mt" '
        /^#pbbench/ {
            if (match($0, /base=[0-9]+kHz/)) base = substr($0, RSTART + 5, RLENGTH - 8) + 0
            # `khzspan=threads` means flo|fhi span every core this process threads sat on, so the
            # MINIMUM is a working core and the in-window check below is meaningful even for pb_mt.
            # Absent or `core` means the range is the main thread only — an idling spectator during a
            # threaded window — and the check stands down. See `_cell_khz_span` in plots.jl.
            if ($0 ~ /khzspan=threads/) span_threads = 1
            next
        }
        /^#/ { next }
        NF >= 4 {
            pb = 0; ref = 0; refname = ""
            for (i = 4; i <= NF; i++) {
                n = split($i, a, "|")
                if (n < 6) continue
                # Fields are indexed FROM THE FRONT: arm|time|commit|anchor|freq|flo|fhi|samples.
                # Counting from the end lands on `fhi`, the in-window MAXIMUM — the one clock field a
                # throttle cannot move, so a cell that ran at half speed reads as perfectly locked.
                fq = a[5] + 0
                # In-window MINIMUM against this box own base clock.
                #
                # SERIAL ARMS ONLY, and that is a correctness limit rather than a scoping preference.
                # `_cell_khz`/`_khz_range!` read `/proc/self/stat` field 39 — the MAIN thread current
                # CPU. For a serial arm the main thread IS the work, so the reading is the work clock.
                # For a threaded arm the workers are on other cores and the driver spins then yields,
                # so the sampled core can report its IDLE frequency while every worker runs at base.
                # Measured: galen `trsmR@512` records flo = 1066 MHz against a 3701 MHz base while
                # every arm in that cell posts full throughput, and `trsm@1000` records 1714 MHz while
                # OpenBLAS and AOCL post the HIGHEST figures of their ladder. A real drop would slow
                # every arm in the window; these slow none of them.
                #
                # Restricted to `pb` for the same reason the rest of this script is: a cached vendor
                # arm carries the state of the epoch it was measured in and is never re-run.
                # The in-window check runs for the serial `pb` arm always, and for `pb_mt` only once the
                # range spans the working cores (`khzspan=threads`). A threaded cache written by the
                # old single-core sampler still stands down.
                if (base > 0 && n >= 8 && ((a[1] == "pb" && !MT) || (a[1] == "pb_mt" && MT && span_threads))) {
                    flo = a[6] + 0
                    if (flo > 0) {
                        lotot++
                        dl = (base - flo) / base * 100
                        if (dl > TOL) {
                            looff++
                            if (looff <= 3) printf "   %s/%s@%s  %s fell to %.0fMHz of %.0fMHz base (-%.0f%%)\n", $1, $2, $3, a[1], flo / 1000, base / 1000, dl
                            badop[$1 "/" $2] = 1; badgrp[$1] = 1
                            if (dl > loworst) loworst = dl
                        }
                    }
                }
                if (fq <= 0) continue
                # A VENDOR whitelist, not "the first arm that is not pb". This check exists to say
                # whether a PB window and a REFERENCE window ran at the same clock — a cross-epoch
                # claim. Two cached arms are neither: `generic` is the LinearAlgebra fallback, and
                # `pb_mt` is PureBLAS itself at N threads, measured in the SAME run as `pb`. Letting
                # either land in `ref` reports a same-run pair as a verified cross-epoch comparison
                # that never happened — and for `pb_mt` it would always read ~0% and look reassuring.
                if (a[1] == "pb") pb = fq
                else if (ref == 0 && (a[1] == "openblas" || a[1] == "aocl" || a[1] == "mkl" || a[1] == "openblas_mt" || a[1] == "aocl_mt" || a[1] == "mkl_mt")) { ref = fq; refname = a[1] }
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
            if (lotot > 0) {
                if (looff == 0) printf "   => in-window clock: all %d PB cells held base (tolerance %s%%)\n", lotot, TOL
                else            printf "   => in-window clock: %d/%d PB cells fell below base, worst -%.0f%% (tolerance %s%%)\n", looff, lotot, loworst, TOL
            }
            if (tot == 0 && looff == 0) { print "   no cells carry both a pb and a reference clock — cross-arm check skipped"; exit 0 }
            if (tot > 0 && off == 0) printf "   => cross-arm: all %d cells within %s%% (worst %.1f%%)\n", tot, TOL, worst
            if (off > 0) printf "   => cross-arm: %d/%d cells clock-mismatched (%d ok), worst %.1f%% (tolerance %s%%)\n", off, tot, ok, worst, TOL
            if (off == 0 && looff == 0) exit 0
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
    for g in L1 L2 L3 LP CL1 CL2 CL3 CLP DL1 DL2 DL3 DLP; do
        julia --project=bench bench/plots.jl bench group=$g arms=pb nodraw
    done
MSG
fi
exit $bad
