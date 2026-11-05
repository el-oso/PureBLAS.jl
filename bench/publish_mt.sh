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
# Zen4 cells compare a throttled reference window against an unthrottled PB one.
#
#   bench/publish_mt.sh            # audit, rebuild, verify, print the git add line
#   bench/publish_mt.sh --force    # rebuild and verify even if the audit reports problems
#
# Never commits, never pushes, never measures.
set -u
cd "$(dirname "$0")/.." || exit 2
FORCE=""
[ "${1:-}" = "--force" ] && FORCE=1

# while-read, not mapfile: macOS ships bash 3.2, where that builtin does not exist.
caches=()
while IFS= read -r _ln; do [ -n "$_ln" ] && caches+=("$_ln"); done < <(ls bench/mt_data_*.txt 2>/dev/null | grep -v _lite)
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
echo "══ 2  rebuild $DOC AND the panels from the caches"
# THE PANELS BELONG HERE, NOT TO A SEPARATE HAND-RUN COMMAND. `gen_threading.jl` writes the page and
# `plots.jl mtdraw` writes the figures it embeds; run apart, the two describe different cache states
# and the page keeps rendering, so nothing announces the drift. Measured 2026-11-05: the panels had
# sat at their 2026-10-05 render while this chain refused for a month and the mt caches were rewritten
# three times. A reader saw potrf at n=4096 as ~0.6x of AOCL where the cache held 1.07x — and the
# stale panel's own y-axis topped out at 1.0x, so it could not have drawn a passing cell at all.
julia --project=bench bench/gen_threading.jl --write || { echo "BUILD FAILED — nothing published"; exit 2; }
julia --project=bench bench/plots.jl mtdraw > /dev/null || { echo "PANEL RENDER FAILED"; exit 2; }
echo "   rebuilt: $DOC + perf_mt_*.svg + perf_mtgate_*.svg"

echo
echo "══ 3  verify the rebuild is a pure function of the caches"
# Same contract as publish.sh step 3: generate again into a temp and byte-compare. A generator that
# does not reproduce itself is a pipeline bug, not something to commit. The PANELS get the same
# treatment as the page — `mtdraw` honours `outdir=`, so a second render into a temp dir costs one
# julia start and closes the hole that let a month-old figure publish beside a current page.
tmpd="$(mktemp -d)"; tmp="$tmpd/threading.md"; trap 'rm -rf "$tmpd"' EXIT
julia --project=bench bench/gen_threading.jl > "$tmp" || { echo "re-generate FAILED"; exit 2; }
if ! cmp -s "$tmp" "$DOC"; then
    echo "   $DOC DIFFERS from a fresh regenerate — the build is not a pure function of the caches."
    diff -u "$DOC" "$tmp" | head -20
    exit 1
fi
julia --project=bench bench/plots.jl mtdraw "outdir=$tmpd" > /dev/null || { echo "re-render FAILED"; exit 2; }
stale=0
for f in "$tmpd"/perf_mt_*.svg "$tmpd"/perf_mtgate_*.svg; do
    [ -e "$f" ] || continue
    b="$(basename "$f")"
    if ! cmp -s "$f" "docs/src/assets/$b"; then echo "   STALE PANEL: $b"; stale=1; fi
done
[ "$stale" -eq 0 ] || {
    echo "   A committed panel differs from a fresh render of the same caches — publish, do not commit."
    exit 1
}
echo "   $DOC and every panel match a fresh rebuild"

echo
echo "══ 4  commit these (nothing was committed or pushed):"
git status --short -- "$DOC" docs/src/assets/perf_mt_*.svg docs/src/assets/perf_mtgate_*.svg
echo
echo "   git add $DOC 'docs/src/assets/perf_mt_*.svg' 'docs/src/assets/perf_mtgate_*.svg'"
