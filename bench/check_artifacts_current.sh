#!/bin/bash
# Does what is COMMITTED still match what the caches say today?
#
# WHY THIS EXISTS. bench/cache_staleness.sh answers "are the cells current w.r.t. src/". That is only
# half the chain. The other half — "are the rendered artifacts current w.r.t. the cells" — had nothing
# checking it, and on 2026-08-17 it failed four different ways in one night:
#   • docs/src/assets/perf_*.svg (OpenBLAS view) had not been re-rendered since bdb9497 while the
#     *_aocl.svg set had been re-rendered three times — the two published pages disagreed about the
#     same fleet;
#   • a pptrfL gate run merged 7 fresh cells into a cache AFTER the committed tables were generated,
#     so docs/src/coverage.md published a number the cache no longer contained.
# Neither is visible by reading the artifacts. Both are trivially visible by REBUILDING them.
#
# HOW: rebuild every artifact from the caches on disk into a temp dir and byte-compare. The render is
# deterministic — two rebuilds from an unchanged cache produce byte-identical files (verified
# 2026-08-18, 18/18 files) — so ANY difference means the committed artifact is stale, never noise.
# Nothing here measures anything: no OpenBLAS, no AOCL, no `bench` argument.
#
# EXIT: 0 clean · 1 something is stale · 2 could not run the check.
#
#   bench/check_artifacts_current.sh            # both halves gate
#   bench/check_artifacts_current.sh --force     # report cell staleness but gate only on artifacts
#                                                # (bench/publish.sh --force passes this through)
set -u
cd "$(dirname "$0")/.." || exit 2
source bench/artifact_build.sh
FORCE=""; [ "${1:-}" = "--force" ] && FORCE=1

rc=0
echo "══ 1/2  cells vs src/  (bench/cache_staleness.sh)"
if ! bench/cache_staleness.sh; then
    if [ -n "$FORCE" ]; then echo "(--force: cell staleness reported, not gated)"; else rc=1; fi
fi

echo
echo "══ 2/2  committed artifacts vs caches  (rebuild + byte-compare)"
T=$(mktemp -d) || exit 2
BK="$T/coverage.md.orig"
cp docs/src/coverage.md "$BK" || exit 2
# The coverage generators rewrite docs/src/coverage.md IN PLACE and offer no output redirect, so the
# check has to build over it and put the committed content back — including on Ctrl-C. Restoring from a
# copy (not `git checkout`) is deliberate: an uncommitted edit to that page is not this tool's to discard.
trap 'cp -f "$BK" docs/src/coverage.md 2>/dev/null; rm -rf "$T"' EXIT INT TERM

if ! build_artifacts "$T" > "$T/build.log" 2>&1; then
    echo "REBUILD FAILED — cannot judge the artifacts:"; tail -25 "$T/build.log"; exit 2
fi

stale=()
for f in "$T"/perf_*.svg; do
    b=$(basename "$f"); cmp -s "$f" "docs/src/assets/$b" || stale+=("docs/src/assets/$b")
done
for f in "$T"/gen_table*.md "$T"/provenance.md; do
    b=$(basename "$f"); cmp -s "$f" "bench/$b" || stale+=("bench/$b")
done
cmp -s docs/src/coverage.md "$BK" || stale+=("docs/src/coverage.md")

n=$(ls "$T"/perf_*.svg "$T"/gen_table*.md "$T"/provenance.md 2>/dev/null | wc -l)
if [ ${#stale[@]} -eq 0 ]; then
    echo "   => all $((n + 1)) artifacts match a fresh rebuild from the current caches"
else
    rc=1
    echo "   => ${#stale[@]} of $((n + 1)) artifacts are STALE:"
    printf '      %s\n' "${stale[@]}"
    echo "   Regenerate them all in the right order with:  bench/publish.sh"
fi
# NOT CHECKED, deliberately: docs/src/assets/*_lite.svg and bench/gen_table*_lite.md come from *_lite
# caches, which no box keeps (they are smoke artifacts, explicitly not gate numbers). Nothing on disk
# can rebuild them, so nothing here can judge them.
exit $rc
