#!/bin/bash
# WHERE IS EACH FLEET BOX, AND IS THAT SOMEWHERE REAL? Read-only, seconds, no compute.
#
# `fleet_sync.sh` is fire-and-forget: it puts a box at a ref and nothing afterwards notices that local
# work has moved on. The box then measures code that is behind, or — the case this was written for —
# code that has since been REVERTED, which is worse than behind because the commit is no longer on any
# branch and no `git log` on master will ever mention it. galen sat at a reverted commit for hours and
# the only hint was `git worktree list` printing a feature sha next to the word `master`.
#
# So this answers three things per box, and the third is the one that catches a reverted sync:
#   HEAD        — the commit and its subject, plus whether the tree is dirty
#   vs local    — same / behind N / ahead N / DIVERGED, against this checkout's HEAD
#   on a branch — is HEAD contained in ANY remote branch? If not it is ORPHANED: reverted, rebased
#                 away, or force-pushed over, and anything measured there is unreproducible.
#
#   bench/fleet_status.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BOXES=(galen neuromancer)
REMOTE_DIR='~/Documents/claude/PureBLAS.jl'

git fetch -q origin 2>/dev/null || true
lh=$(git rev-parse HEAD)
printf "local  %s  %s\n" "$(git rev-parse --short HEAD)" "$(git log -1 --format=%s | cut -c1-58)"
printf "       branch %s, %s\n\n" "$(git rev-parse --abbrev-ref HEAD)" \
    "$([ -n "$(git status --porcelain -- src test bench juliac)" ] && echo 'DIRTY in src/test/bench/juliac' || echo 'clean')"

for box in "${BOXES[@]}"; do
    if ! ssh -n -o ConnectTimeout=8 -o BatchMode=yes "$box" true 2>/dev/null; then
        printf "%-13s UNREACHABLE\n" "$box"; continue
    fi
    # A REMOTE THAT PRINTS NOTHING MUST NOT KILL THE RUN. If `cd` fails, the directory is not a git
    # repo, or git errors, the chain emits nothing; `read` then returns 1 and `set -e` exits the
    # script — before this box has even been named, and without reaching the boxes after it. That is
    # strictly worse than the UNREACHABLE case handled above, which at least says which box.
    if ! read -r rh dirty subj < <(ssh -n "$box" "cd $REMOTE_DIR && printf '%s %s %s\n' \
        \"\$(git rev-parse HEAD)\" \
        \"\$(git status --porcelain | grep -vc '^??' || true)\" \
        \"\$(git log -1 --format=%s | tr ' ' '_' | cut -c1-52)\""); then
        printf "%-13s NO REPO or git error at %s\n\n" "$box" "$REMOTE_DIR"; continue
    fi
    printf "%-13s %s  %s\n" "$box" "${rh:0:8}" "$(printf '%s' "$subj" | tr '_' ' ')"
    # Relation to local HEAD. Both shas must exist HERE for this to mean anything; a box on a commit
    # this checkout has never fetched is itself a finding.
    if ! git cat-file -e "$rh^{commit}" 2>/dev/null; then
        rel="UNKNOWN COMMIT — not in this checkout, fetch or it was never pushed"
    elif [[ "$rh" == "$lh" ]]; then rel="same as local"
    elif git merge-base --is-ancestor "$rh" "$lh" 2>/dev/null; then
        rel="BEHIND local by $(git rev-list --count "$rh..$lh") commit(s)"
    elif git merge-base --is-ancestor "$lh" "$rh" 2>/dev/null; then
        rel="ahead of local by $(git rev-list --count "$lh..$rh") commit(s)"
    else rel="DIVERGED from local"; fi
    # The orphan test. `--contains` over remote branches only: a LOCAL branch on the box proves
    # nothing, because `reset --hard` can point a local label anywhere.
    br=$(git branch -r --contains "$rh" 2>/dev/null | sed 's/^[ *]*//' | grep -v HEAD | paste -sd, - || true)
    printf "              %s\n" "$rel"
    if [[ -z "$br" ]]; then
        printf "              ⛔ ORPHANED — on NO remote branch. Reverted or force-pushed over;\n"
        printf "                 anything measured here is unreproducible. Re-sync before use.\n"
    else
        printf "              on %s\n" "$br"
    fi
    printf "              %s tracked file(s) dirty\n\n" "${dirty:-?}"
done
