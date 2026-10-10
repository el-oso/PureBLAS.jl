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
    # DETACH, NEVER CHECK OUT `master`. Checking out `master` and hard-resetting it to `$ref` leaves
    # the box's `master` pointing at whatever was synced — a feature commit, or one that has since
    # been reverted — so `git branch`, `git worktree list` and the shell prompt all name it `master`
    # while it is nothing of the kind. That is the same provenance lie this script exists to prevent,
    # one level up: rsync made the cache's `commit=` stamp false, and hijacking the branch label makes
    # the BOX's own report of itself false. Measured consequence: galen sat on a reverted commit for
    # hours under the label `master`, and `git worktree list` showed `15bd758c [master]`.
    # Detached HEAD is the honest state for a box that tracks whatever it was last told to.
    # DETACH AT CURRENT HEAD, THEN RESET. `checkout --detach <ref>` REFUSES when a tracked file is
    # locally modified and differs between the two trees, and under `set -euo pipefail` that refusal
    # kills this script outright: the `echo`s in the `&&` chain never print, the parity check never
    # runs, and in `all` mode the remaining boxes are never synced. A dirty box is an EXPECTED state —
    # the line below reports the count — so the reset has to keep its force. Detaching at HEAD moves
    # no files and cannot refuse; `reset --hard $ref` then does the work it always did.
    ssh "$box" "cd $REMOTE_DIR && \
        git fetch -q origin && \
        git checkout -q --detach && \
        git reset --hard -q $ref && \
        echo \"  HEAD  \$(git rev-parse --short HEAD)  \$(git log -1 --format=%s | cut -c1-60)\" && \
        echo \"  dirty \$(git status --porcelain | grep -vc '^??' || true) tracked file(s)\" && \
        echo \"  caches kept: \$(ls bench/*_data_*.txt 2>/dev/null | wc -l)\""
    # PARITY AGAINST THE REF, NOT AGAINST THE WORKING TREE. Comparing an md5 walk of the local `src/`
    # answers "does the box match what is on my disk", which is the wrong question twice over: with
    # uncommitted edits it can never pass, and when syncing a box to a DIFFERENT ref on purpose it
    # reports a mismatch for a sync that did exactly what was asked. Git already has the exact answer
    # — the tree object id of `src/` — so compare that: identical ids mean byte-identical content, no
    # file walk, no locale ordering to get wrong.
    lt=$(git rev-parse "$ref:src")
    rt=$(ssh "$box" "cd $REMOTE_DIR && git rev-parse HEAD:src")
    if [[ "$lt" == "$rt" ]]; then echo "  src parity OK (tree $lt)"; else echo "  SRC PARITY MISMATCH ($lt vs $rt)" >&2; fi
done
