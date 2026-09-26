#!/usr/bin/env bash
# PUBLISH THE THREADED ARTIFACTS — one command, fixed order, never a subset. The sibling of
# `bench/publish.sh`, which owns the single-threaded gate and cannot see these caches.
#
# WHY A SIBLING RATHER THAN A WIDER publish.sh. `publish.sh` rebuilds artifacts that are a pure
# function of the GATE caches (`plots_data_*`) and byte-verifies them. The mt caches sit deliberately
# outside that glob, so no gate generator can see them, and wiring them in would defeat the separation
# that keeps a 6-core-affinity run from ever touching a 1-core-pinned gate cell. The coverage
# generators REFUSE an `mt_data_*` argument outright for the same reason.
#
# `gen_threading.jl` already existed and its own header says it "is run on its own, by hand". That is
# the gap this closes: run by hand, the audit chain gets skipped, and the page then publishes numbers
# whose clock provenance nobody looked at. On 2026-09-26 that was not hypothetical — 204 of 937
# wintermute cells compare a throttled reference window against an unthrottled PB one.
#
#   bench/publish_mt.sh            # audit, rebuild, verify, print the git add line
#   bench/publish_mt.sh --force    # rebuild and verify even if the audit reports problems
#
# Never commits, never pushes, never measures.
set -u
cd "$(dirname "$0")/.." || exit 2
FORCE=""
[ "${1:-}" = "--force" ] && FORCE=1

mapfile -t caches < <(ls bench/mt_data_*.txt 2>/dev/null | grep -v _lite)
[ ${#caches[@]} -eq 0 ] && { echo "no threaded caches (bench/mt_data_*.txt) — nothing to publish"; exit 2; }

DOC=docs/src/threading.md

echo "══ 1  audit the threaded caches"
# NOT fatal by default, and deliberately: staleness is permanent under targeted sweeps, and the
# anchor check is advisory by its own header. A clock mismatch IS a real blocker, so it is named
# separately below rather than buried in one exit code.
audit="$(bash bench/audit_mt.sh 2>&1)"; arc=$?
printf '%s\n' "$audit" | grep -E "^   =>|^══" | sed 's/^/   /'
clockbad=$(printf '%s' "$audit" | grep -cE "cells clock-mismatched" || true)
if [ "$clockbad" -gt 0 ] && [ -z "$FORCE" ]; then
    echo
    echo "   CLOCK-MISMATCHED CELLS PRESENT. Those cells compare two machine states — the frequency"
    echo "   rule calls them invalid, not noisy — so a page built on them publishes a number that"
    echo "   cannot be defended. Re-measure them with both windows at one clock, or pass --force if"
    echo "   you are deliberately publishing a page that says so."
    exit 1
fi
[ "$arc" -ne 0 ] && echo "   (audit reported non-clock problems; continuing — see above)"

echo
echo "══ 2  rebuild $DOC from the caches"
julia --project=bench bench/gen_threading.jl --write || { echo "BUILD FAILED — nothing published"; exit 2; }
echo "   rebuilt"

echo
echo "══ 3  verify the rebuild is a pure function of the caches"
# Same contract as publish.sh step 3: generate again into a temp and byte-compare. A generator that
# does not reproduce itself is a pipeline bug, not something to commit.
tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT
julia --project=bench bench/gen_threading.jl > "$tmp" || { echo "re-generate FAILED"; exit 2; }
if ! cmp -s "$tmp" "$DOC"; then
    echo "   $DOC DIFFERS from a fresh regenerate — the build is not a pure function of the caches."
    diff -u "$DOC" "$tmp" | head -20
    exit 1
fi
echo "   $DOC matches a fresh regenerate"

echo
echo "══ 4  commit this (nothing was committed or pushed):"
git status --short -- "$DOC"
echo
echo "   git add $DOC"
