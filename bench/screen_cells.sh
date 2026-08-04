#!/usr/bin/env bash
# Screen a list of gate cells for "is this really failing?" — K replications each via bench/cellrep.jl,
# then an exact sign test + bootstrap CI per cell (bench/probes/cell_hypothesis.jl).
#
# WHY A SCREEN. Every cell below is a sub-1% reading taken from a CACHED reference, i.e. a ratio whose
# two arms came from different processes. That comparison was measured wrong by up to 2.3pp at
# axpy n=1e6 (cached 0.960 vs same-run 0.983), and the per-process spread at L3-straddling sizes is
# ~3pp — bigger than every gap in this list. So none of these are known to be real yet, and `dot`
# n=1e6 already retired this way (5/10 below 1.0, sign p=1.0) after sitting on the 🐢 list.
#
# COST: ~8.4 s per replication, so K=10 is ~85 s per cell. A unanimous 10/10 reaches sign p=0.002 (P99);
# plots.jl needs 5.7 min per cell to reach only p=0.02.
#
# NOT the published number — cellrep reads ~2pp pessimistic vs plots.jl (see its header). This decides
# WHICH cells deserve kernel work; plots.jl still produces the ratio of record for anything that stays.
#
#   bench/screen_cells.sh [K]        # K defaults to 10
set -euo pipefail
cd "$(dirname "$0")/.."
K="${1:-10}"
CORE="${SCREEN_CORE:-4}"
OUT=bench/probes
mkdir -p "$OUT"

# Cells carrying a sub-1% cached reading on Zen4 (wintermute). Zen3's list is disjoint and gets screened
# on galen with the same script.
CELLS=(
    "asum 30000" "asum 100000" "asum 300000" "asum 1000000"
    "scal 10000" "scal 30000" "scal 100000"
    "zaxpy 100000" "zaxpy 300000"
    "zscal 300000"
    "dzasum 1000"
)

for cell in "${CELLS[@]}"; do
    set -- $cell
    op=$1; n=$2
    log="$OUT/${op}_${n}_runs.log"
    : > "$log"
    for _ in $(seq 1 "$K"); do
        taskset -c "$CORE" julia --startup-file=no --project=bench bench/cellrep.jl "$op" "$n" >> "$log" 2>/dev/null
    done
    printf '%-16s ' "$op n=$n"
    julia --startup-file=no -e '
        using Statistics
        r = [parse(Float64, split(l, "\t")[3]) for l in readlines(ARGS[1])]
        k = count(<(1.0), r)
        # exact two-sided sign test, same as bench/probes/cell_hypothesis.jl
        lg(a,b) = sum(log, (a-b+1):a; init=0.0) - sum(log, 1:b; init=0.0)
        tail(m,n) = sum(exp(lg(n,i)) for i in 0:m; init=0.0) / 2.0^n
        n = length(r); p = min(1.0, 2 * (k <= n/2 ? tail(k,n) : tail(n-k,n)))
        println("median=", round(median(r); digits=4), "  below=", k, "/", n,
                "  sign p=", round(p; digits=4), "   ",
                p < 0.05 && median(r) < 1 ? "FAIL (real)" : "PASS≈ (retire)")' "$log"
done
