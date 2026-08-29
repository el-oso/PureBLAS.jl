#!/bin/bash
# Is every published cell measured against the code that ships TODAY?
#
# WHY THIS EXISTS. On 2026-08-17 the published Zen5 `dot` row read 0.91 because ONE cell (dot@10000)
# carried a pb arm measured at `edd2753`, before eight src/ files changed. Two fresh PB-only runs put
# that cell back above parity and moved the row to 0.98. The number had been wrong for days, and it
# surfaced only because someone asked an offhand question about Zen5 BLAS-1. An audit then found
# 485/485/478 of 604 cells PER BOX resting on pre-change arms — ~80% of every published column.
#
# Nothing in the pipeline checked this. `plots.jl` stamps each arm with its commit, `gate_gaps.jl`
# prints STALE for cells it happens to rank, but no tool answers "is the WHOLE table current?" — and a
# stale cell that currently PASSES is just as untrustworthy as one that fails, only quieter.
#
# STALE MEANS "src/ MOVED", NOT "the hash differs" — the definition bench/gate_gaps.jl already uses. A
# cell measured five doc commits ago is not stale. A hash comparison would flag every row the moment
# anything is committed, and a flag that is always on gets ignored. Since 2026-08-29 "moved" is judged
# on the PARSED source, so a comment-only commit to src/ does not stale a cell either — see the
# `semantic_only` helper below and bench/src_semantic_diff.jl.
#
# EXIT STATUS IS THE POINT: non-zero when any cache has cells older than the last src/ change, so this
# can gate a publish rather than merely inform one.
#
#   bench/cache_staleness.sh                    # every bench/plots_data_*.txt
#   bench/cache_staleness.sh <cache> [more...]  # specific caches
set -u
cd "$(dirname "$0")/.." || exit 2
files=("$@")
if [ ${#files[@]} -eq 0 ]; then
    mapfile -t files < <(ls bench/plots_data_*.txt 2>/dev/null | grep -v _lite)
fi
[ ${#files[@]} -eq 0 ] && { echo "no cache files found"; exit 2; }

HEAD_SHA=$(git rev-parse --short HEAD)
echo "HEAD=$HEAD_SHA   (a cell is STALE iff src/ changed between its pb arm's commit and HEAD)"

# "src/ MOVED" is judged on the PARSED source, not the file list. git's diff cannot tell a kernel edit
# from a comment fix, so a single comment-only commit staled all 2790 cells on 2026-08-29 and left only
# a full re-sweep or `--force` — and --force also waves through the real staleness this gate exists to
# catch. bench/src_semantic_diff.jl compares Expr trees; see its header for why it is conservative.
# Memoized: the same commit recurs across all three caches, and each check forks git per changed file.
declare -A _SEMCACHE
semantic_only() {                      # returns 0 when the src/ diff since $1 is comments/formatting
    local c=$1
    if [ -z "${_SEMCACHE[$c]:-}" ]; then
        julia --startup-file=no bench/src_semantic_diff.jl "$c" >/dev/null 2>&1
        # A missing/broken julia exits non-zero here, which reads as "real change" — the safe side.
        _SEMCACHE[$c]=$?
    fi
    return "${_SEMCACHE[$c]}"
}
rc=0
for f in "${files[@]}"; do
    box=$(basename "$f" .txt | sed 's/plots_data_//')
    total=0; stale=0
    echo "── $box"
    # one line per (level, op, size); fields 4.. are arm|timestamp|commit|...  — report the pb arm
    while read -r n c; do
        total=$((total + n))
        if git cat-file -e "${c}^{commit}" 2>/dev/null; then
            nsrc=$(git diff --name-only "$c"..HEAD -- src/ 2>/dev/null | wc -l)
            if [ "$nsrc" -eq 0 ]; then
                printf "   ok     %5d cells @ %-9s\n" "$n" "$c"
            elif semantic_only "$c"; then
                printf "   ok     %5d cells @ %-9s  %s src file(s) touched, comments/format only\n" "$n" "$c" "$nsrc"
            else
                stale=$((stale + n)); rc=1
                printf "   STALE  %5d cells @ %-9s  %s src file(s) changed since\n" "$n" "$c" "$nsrc"
            fi
        else
            # A commit absent from this clone cannot be judged — do not cry wolf, but do not call it ok.
            stale=$((stale + n)); rc=1
            printf "   UNKNOWN %4d cells @ %-9s  (commit not in this clone — fetch, then re-run)\n" "$n" "$c"
        fi
    done < <(awk -F'\t' 'NR>1 {for(i=4;i<=NF;i++){split($i,a,"|"); if(a[1]=="pb") print a[3]}}' "$f" |
             sort | uniq -c | sort -rn | awk '{print $1, $2}')
    if [ "$stale" -gt 0 ]; then
        printf "   => %d/%d cells stale (%d%%)\n" "$stale" "$total" $((100 * stale / total))
    else
        printf "   => all %d cells current\n" "$total"
    fi
done

if [ $rc -ne 0 ]; then
    cat <<'MSG'

STALE CELLS PRESENT — published numbers from these caches may describe code that no longer ships.
Refresh with PB-only runs, which reuse the cached reference arms (never re-measure OpenBLAS/AOCL):
    for g in L1 L2 L3 LP CL1 CL2 CL3 CLP; do
        julia --project=bench bench/plots.jl bench group=$g arms=pb nodraw
    done
NOTE: a single full `bench arms=pb` with no op=/group= is REFUSED by design — a full run REPLACES the
cache and would drop the openblas/aocl arms. Only op=/group= runs merge per arm.
MSG
fi
exit $rc
