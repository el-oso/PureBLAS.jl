#!/usr/bin/env bash
# Sync a fleet box to a pushed commit, by GIT — not rsync.
#
# WHY THIS EXISTS. The fleet used to be synced with `rsync src/ box:.../src/`. That copies the code
# but not its identity: the box's git HEAD stays wherever it was, and `bench/plots.jl` stamps
# `commit=` into every cache header from `git rev-parse`. On 2026-07-31 galen produced a full gate
# sweep stamped `commit=ac96c00` while actually running the tree from `78eafc7` — 13 commits and two
# perf fixes later. The numbers were fine (source parity was verified by md5), but the provenance in
# the published table was a lie, and nothing in the artifact revealed it.
#
# A benchmark cache is evidence. Evidence needs a truthful provenance line, so the box must be AT the
# commit it claims. That means fetch + hard reset, which also guarantees no half-synced tree: rsync of
# a subdirectory can leave a box with new src/ and stale test/ or juliac/, and nothing would say so.
#
# Bench caches (bench/plots_data_*.txt) are gitignored, so a hard reset preserves them — `op=`/merge
# runs keep working across a sync.
#
# Usage:  bench/fleet_sync.sh galen [ref]        # ref defaults to origin/master
#         bench/fleet_sync.sh all   [ref]
set -euo pipefail

BOXES_ALL=(galen neuromancer)
REMOTE_DIR='~/Documents/claude/PureBLAS.jl'

target="${1:?usage: fleet_sync.sh <box|all> [ref]}"
ref="${2:-origin/master}"

if [[ "$target" == "all" ]]; then boxes=("${BOXES_ALL[@]}"); else boxes=("$target"); fi

# The commit must be ON the remote, or the box cannot fetch it. Catch the classic "synced my
# uncommitted working tree" mistake up front rather than after a 3-hour sweep.
local_head=$(git rev-parse --short HEAD)
if ! git merge-base --is-ancestor "$local_head" "$ref" 2>/dev/null; then
    echo "REFUSING: local HEAD $local_head is not an ancestor of $ref." >&2
    echo "  Commit and push first — a box can only be synced to something it can fetch." >&2
    exit 1
fi
if [[ -n "$(git status --porcelain -- src test bench juliac 2>/dev/null)" ]]; then
    echo "WARNING: local src/test/bench/juliac has uncommitted changes; they will NOT reach the fleet." >&2
    git status --porcelain -- src test bench juliac | sed 's/^/    /' >&2
fi

for box in "${boxes[@]}"; do
    echo "=== $box -> $ref ==="
    if ! ssh -o ConnectTimeout=8 -o BatchMode=yes "$box" true 2>/dev/null; then
        echo "  UNREACHABLE — skipped" >&2; continue
    fi
    ssh "$box" "cd $REMOTE_DIR && \
        git fetch -q origin && \
        git checkout -q master 2>/dev/null || true; \
        git reset --hard -q $ref && \
        echo \"  HEAD  \$(git rev-parse --short HEAD)  \$(git log -1 --format=%s | cut -c1-60)\" && \
        echo \"  dirty \$(git status --porcelain | grep -vc '^??' || true) tracked file(s)\" && \
        echo \"  caches kept: \$(ls bench/plots_data_*.txt 2>/dev/null | wc -l)\""
    # Parity check: identical source content, not just an identical commit id. Cheap, and it is the
    # check that caught a locale sort-order false alarm when this was verified by hand.
    lm=$(find src -name '*.jl' -exec md5sum {} \; | LC_ALL=C sort | md5sum | cut -d' ' -f1)
    rm_=$(ssh "$box" "cd $REMOTE_DIR && find src -name '*.jl' -exec md5sum {} \; | LC_ALL=C sort | md5sum | cut -d' ' -f1")
    if [[ "$lm" == "$rm_" ]]; then echo "  src parity OK"; else echo "  SRC PARITY MISMATCH ($lm vs $rm_)" >&2; fi
done
