#!/bin/bash
# Regenerate EVERY published performance artifact from the caches on disk, in one fixed order.
#
# WHY ONE COMMAND. The pipeline is four generators over three caches, and running a subset of them is
# what produced every artifact bug of 2026-08-17: SVGs re-rendered for one reference view but not the
# other, tables regenerated but not the plots, plots regenerated but not the tables. Every partial state
# is reachable by hand and none of them announces itself. There is no partial state here — the whole set
# is rebuilt or nothing is.
#
# WHAT IT DOES NOT DO: it does not measure (no `bench` argument reaches plots.jl — see
# bench/artifact_build.sh), it does not commit, and it does not push. It prints the `git add` line and
# stops; pushing publishes the docs site, and that stays a human decision.
#
#   bench/publish.sh           # refuses if any published cell predates the shipping src/
#   bench/publish.sh --force   # publish anyway (cell staleness is reported, not enforced)
set -u
cd "$(dirname "$0")/.." || exit 2
source bench/artifact_build.sh
FORCE=""; [ "${1:-}" = "--force" ] && FORCE=1

echo "══ 1  cells vs src/"
if ! bench/cache_staleness.sh; then
    if [ -n "$FORCE" ]; then
        echo "(--force given: publishing over stale cells)"
    else
        echo
        echo "REFUSING TO PUBLISH — these caches carry cells measured against code that no longer ships."
        echo "Refresh them (PB-only group runs, per the message above) or re-run with --force."
        exit 1
    fi
fi

echo
echo "══ 1b  pb arm vs its reference arms: same clock?"
# `cache_staleness.sh` answers "is this cell measured against today's CODE"; this answers "is it
# measured in the same MACHINE STATE as the references it is divided by". Both are needed: on
# 2026-08-18 neuromancer's refresh was 100% current AND 100% invalid, because the box boosted to
# 4.7 GHz for the whole sweep against references cached at 2.0 GHz, inflating every Zen5 ratio ~2.3x.
# `freqgate.jl` structurally cannot see that (its reference is the header, which floated too).
if ! bench/check_arm_clocks.sh; then
    if [ -n "$FORCE" ]; then
        echo "(--force given: publishing over clock-mismatched cells)"
    else
        echo
        echo "REFUSING TO PUBLISH — these ratios divide arms measured at different clocks."
        echo "Re-lock the box and re-measure (per the message above), or re-run with --force."
        exit 1
    fi
fi

echo
echo "══ 1a2  is each local cache the newest its box has produced?"
# The other three checks all reason INSIDE a file (cells vs source, arms vs arms). None of them can see
# that the file itself is an old copy: fleet_sync.sh pushes SOURCE to the boxes, nothing pulls CACHES
# back, so a box can re-measure and improve while the copy published from here stays behind.
# Measured 2026-08-29: the local copy of neuromancer's cache was a day stale, and a whole analysis
# reported 143 red / 126 anchor-mismatched / 26 artifacts for a box that actually had 115 / 0 / 0.
if ! bench/check_cache_freshness.sh; then
    if [ -n "$FORCE" ]; then
        echo "(--force given: publishing from cache copies that may be behind their boxes)"
    else
        echo
        echo "REFUSING TO PUBLISH — a local cache is older than the box that produced it, or could not"
        echo "be verified. Pull it (see the scp line above) and re-run, or --force."
        exit 1
    fi
fi

echo
echo "══ 1c  pb arm vs its reference arms: same MACHINE STATE?"
# 1b asks whether the arms ran at the same CLOCK. This asks whether they ran in the same machine state
# at all, which the clock cannot see: a box can hold its clock perfectly while its memory system, page
# placement or thermal state drifts, and the per-cell ANCHOR (a fixed calibration workload re-timed
# around every measurement) is the only field that records it.
#
# Not hypothetical, and not covered by any other check here. Both incidents shipped through 1a and 1b:
#   galen  — 616 of 863 cells (71%) anchor-mismatched after an `arms=pb` refresh; its red count read 58,
#            and the box was not 26 cells worse, the ARMS were.
#   zen5   — 121 cells measured at anchor 26.07 against references at 18.26 (43%).
# Neither cache_staleness.sh (commit staleness) nor check_artifacts_current.sh (rebuild compare) nor
# freqgate.jl (clock) contains the word "anchor".
if ! bench/check_arm_anchors.sh; then
    if [ -n "$FORCE" ]; then
        echo "(--force given: publishing over anchor-mismatched cells)"
    else
        echo
        echo "REFUSING TO PUBLISH — these ratios divide arms measured in different machine states."
        echo "Re-measure the named scope FULL-ARMS (no arms=pb), so all arms share one state, or --force."
        exit 1
    fi
fi

echo
echo "══ 2  rebuild artifacts (both reference views + tables)"
build_artifacts || { echo "BUILD FAILED — nothing published"; exit 2; }

echo
echo "══ 3  verify the rebuild"
bench/check_artifacts_current.sh ${FORCE:+--force} > /dev/null 2>&1
v=$?
if [ $v -eq 2 ]; then
    echo "verifier could not run — inspect with: bench/check_artifacts_current.sh"; exit 2
elif [ $v -ne 0 ] && [ -z "$FORCE" ]; then
    # A rebuild that its own verifier still calls stale means the build is not a pure function of the
    # caches. That is a bug in the pipeline, not a reason to commit.
    echo "STILL STALE AFTER REBUILD — do not commit; run bench/check_artifacts_current.sh"; exit 1
fi
echo "   artifacts match a fresh rebuild"

echo
echo "══ 4  commit these (nothing was committed or pushed):"
git status --short -- docs/src/assets bench/gen_table.md bench/gen_table_aocl.md bench/provenance.md docs/src/coverage.md
echo
# Only the artifact paths — never bench/plots_data_*.txt (gitignored, ~11 MB each, and evidence, not output).
echo "   git add docs/src/assets/perf_*.svg bench/gen_table.md bench/gen_table_aocl.md bench/provenance.md docs/src/coverage.md"
