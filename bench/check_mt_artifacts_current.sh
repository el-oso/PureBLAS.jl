#!/usr/bin/env bash
# ARE THE COMMITTED THREADED ARTIFACTS CURRENT? Renders the panels and the page from the caches into a
# temp directory and byte-compares against what is in the working tree. It rebuilds NOTHING in place,
# which is the whole point: a check that regenerates first can only ever compare two fresh renders and
# passes by construction.
#
# This is the threaded sibling of `check_artifacts_current.sh`, and it exists because the serial set had
# that guard and the threaded set did not. Measured 2026-11-05: `perf_mt_*.svg` and `perf_mtgate_*.svg`
# had sat at their 2026-10-05 render while `publish_mt.sh` refused for a month and the mt caches were
# rewritten three times. Nothing failed, because the page still built and the figures still drew — a
# stale figure renders perfectly. A reader comparing potrf at n=4096 saw roughly 0.6x of AOCL where the
# cache held 1.07x, and the stale panel's own y-axis topped out at 1.0x, so it could not have drawn a
# passing cell at all.
#
#   bench/check_mt_artifacts_current.sh        # exit 0 = current, 1 = stale, 2 = could not run
#
# Run it before opening a PR that changes a measured number, and after any mt sweep. Refresh with
# `bench/publish_mt.sh`.
set -u
cd "$(dirname "$0")/.." || exit 2

ls bench/mt_data_*.txt >/dev/null 2>&1 || { echo "no mt_data_* caches — nothing to check"; exit 2; }

T="$(mktemp -d)" || exit 2
trap 'rm -rf "$T"' EXIT INT TERM

julia --project=bench bench/plots.jl mtdraw "outdir=$T" >/dev/null 2>&1 || {
    echo "render FAILED — inspect with: julia --project=bench bench/plots.jl mtdraw"; exit 2; }
julia --project=bench bench/gen_threading.jl > "$T/threading.md" 2>/dev/null || {
    echo "page generate FAILED — inspect with: julia --project=bench bench/gen_threading.jl"; exit 2; }

stale=0; checked=0
for f in "$T"/perf_mt_*.svg "$T"/perf_mtgate_*.svg; do
    [ -e "$f" ] || continue
    b="$(basename "$f")"; c="docs/src/assets/$b"; checked=$((checked + 1))
    if [ ! -e "$c" ]; then
        echo "   MISSING: $b has never been committed"; stale=1
    elif ! cmp -s "$f" "$c"; then
        # The age is the actionable part: a panel older than the cache it claims to draw is the case
        # this script exists for, and it reads very differently from a same-day render that merely
        # differs.
        echo "   STALE: $b  (committed $(date -r "$c" '+%Y-%m-%d %H:%M'), caches up to $(date -r "$(ls -t bench/mt_data_*.txt | head -1)" '+%Y-%m-%d %H:%M'))"
        stale=1
    fi
done
checked=$((checked + 1))
if ! cmp -s "$T/threading.md" docs/src/threading.md; then
    echo "   STALE: docs/src/threading.md"; stale=1
fi

if [ "$stale" -ne 0 ]; then
    echo
    echo "THREADED ARTIFACTS ARE STALE — they describe a cache state that no longer exists."
    echo "Refresh and commit them with:  bench/publish_mt.sh"
    exit 1
fi
echo "   all $checked threaded artifacts match a fresh render of the current caches"
