#!/bin/bash
# Reads changed paths on stdin; fails if a commit touches one reference VIEW of a plot without the other.
#
# The render now emits both views in one invocation (bench/plots.jl `_VIEWS`), so a regenerated pair can
# only be split apart afterwards — by a partial `git add`, a hand-edited SVG, or a cherry-pick. That is
# the residual hole, and it is the only artifact check that needs NO cache, so it is the only one that
# can run in GitHub CI at all (bench/plots_data_*.txt are gitignored, ~11 MB each, and never pushed).
#
#   git diff --name-only base head | bench/check_view_pairing.sh
set -u
mapfile -t changed
declare -A seen
for f in "${changed[@]}"; do
    case "$f" in docs/src/assets/perf_*.svg) seen["$f"]=1 ;; esac
done
rc=0
for f in "${!seen[@]}"; do
    case "$f" in
        *_lite.svg) continue ;;                       # gitignored smoke artifacts; never committed
        *_aocl.svg) other="${f%_aocl.svg}.svg" ;;
        *)          other="${f%.svg}_aocl.svg" ;;
    esac
    [ -n "${seen[$other]:-}" ] && continue
    echo "UNPAIRED: $f changed but $other did not"
    rc=1
done
if [ $rc -ne 0 ]; then
    cat <<'MSG'

The OpenBLAS and AOCL views are rendered from the SAME cache by the SAME command, so one moving without
the other means the published pages now disagree about the same fleet — exactly the state that had the
OpenBLAS plots stuck at bdb9497 while the AOCL set was re-rendered three times.
Fix by regenerating the whole set:  bench/publish.sh   (then commit every path it lists)
MSG
else
    echo "reference views paired"
fi
exit $rc
