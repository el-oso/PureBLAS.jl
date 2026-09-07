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
# DISCLOSE, DO NOT REFUSE — changed 2026-09-07 when targeted sweeps landed.
#
# This gate used to abort. That was right while a published table said nothing about WHERE its numbers
# came from: a stale cell then silently described code that no longer shipped, and the reader had no way
# to tell. `bench/sweep_op.sh` makes staleness the NORMAL state by design — it re-measures one op across
# the fleet and deliberately leaves every other row at the revision it was last measured at — so an
# abort here would make targeted sweeps unpublishable, which defeats the point of having them.
#
# What replaced the prohibition is disclosure: `coverage_ops.jl` now emits a per-row "swept at" column
# (one short SHA when the whole row agrees, a loud `⚠ mixed` when its boxes disagree). A stale row is
# therefore visible AS stale in the artifact itself, which is the property the refusal was protecting.
# The summary below still prints in full, so a run that quietly went stale is still loud in the log.
#
# The other gates stay HARD: 1b (arms measured at different clocks) and 1a2 (a local cache behind the
# box that produced it) are not disclosable — they make a ratio wrong rather than merely old.
if ! bench/cache_staleness.sh; then
    echo
    echo "NOTE: stale cells present (above). Publishing anyway — each table row states the revision it"
    echo "was measured at in its \"swept at\" column, so a stale row is disclosed rather than hidden."
    echo "Refresh a single row with:  bench/sweep_op.sh <op> --parallel"
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
# ADVISORY ONLY — this does NOT refuse. It was written as a blocking gate on the assumption that anchor
# drift invalidates a ratio; that assumption was tested on 2026-08-29 and FAILED. After an `arms=pb`
# refresh this check flagged 96/863 cells on zen5 and 737/893 on galen, yet all 1726 reference arms were
# byte-identical (arms=pb never touches them) and the pb TIMES for unchanged code were flat — median
# post/pre 0.9997 — against the ~1.24x the drift implied. The anchor moved; the measurements did not.
# Likely because the kernels run taskset-pinned while the anchor workload does not, so the anchor picks
# up background load the gate numbers are immune to.
# Blocking on it would refuse every legitimate arms=pb refresh, which is the documented way to refresh.
# Keep reporting it — a mismatch is worth a look — but the test that decides is "did the pb times move?".
bench/check_arm_anchors.sh || echo "  (advisory: anchor drift reported above does NOT block — see the note in that script)"

echo
echo "══ 2  rebuild artifacts (both reference views + tables)"
build_artifacts || { echo "BUILD FAILED — nothing published"; exit 2; }

echo
echo "══ 3  verify the rebuild"
# `check_artifacts_current.sh` bundles TWO checks: 1/2 cache-vs-src staleness, 2/2 committed-artifacts-
# vs-a-fresh-rebuild. Only the SECOND belongs here. The first is step 1's business and is now disclosed
# rather than fatal (targeted sweeps make staleness permanent by design); gating on the composite exit
# code made every targeted publish fail with "STILL STALE AFTER REBUILD" while the artifacts were in
# fact byte-identical to a fresh rebuild. What this step must catch is a build that is NOT a pure
# function of the caches — so match on that line specifically.
_av="$(bench/check_artifacts_current.sh ${FORCE:+--force} 2>&1)"; v=$?
if [ $v -eq 2 ]; then
    echo "verifier could not run — inspect with: bench/check_artifacts_current.sh"; exit 2
fi
if ! printf '%s' "$_av" | grep -q "artifacts match a fresh rebuild"; then
    # A rebuild whose own verifier says the artifacts differ means the build is not a pure function of
    # the caches. That is a bug in the pipeline, not a reason to commit.
    echo "ARTIFACTS DIFFER FROM A FRESH REBUILD — do not commit; run bench/check_artifacts_current.sh"
    exit 1
fi
echo "   artifacts match a fresh rebuild"

echo
echo "══ 4  commit these (nothing was committed or pushed):"
git status --short -- docs/src/assets bench/gen_table.md bench/gen_table_aocl.md bench/provenance.md docs/src/coverage.md
echo
# Only the artifact paths — never bench/plots_data_*.txt (gitignored, ~11 MB each, and evidence, not output).
echo "   git add docs/src/assets/perf_*.svg bench/gen_table.md bench/gen_table_aocl.md bench/provenance.md docs/src/coverage.md"
