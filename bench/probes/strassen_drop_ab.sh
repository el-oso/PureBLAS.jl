#!/usr/bin/env bash
# Does dropping Strassen cost anything on the gate for the ops that are NOT gemm?
#
# gemm was answered already: identical in all twelve cells. trmm routes its off-diagonal update
# through gemm! for exactly Strassen's flop cut, and symm picks its route from `_strassen_depth`, so
# both have to be asked separately before the recursion can be deleted.
#
# Both arms are FORCED — the shipped value 256 on one side, an unreachable threshold on the other —
# so neither run can write the cache, and the comparison is back-to-back in one methodology.
# Reference arms come from the cache (`arms=pb`); OpenBLAS/AOCL are never re-timed.
set -u
cd "$(dirname "$0")/../.."
JL="${JULIA:-julia}"
OUT="${1:-/tmp/strassen_drop_ab}"

for v in 256 1073741824; do
    for op in trmm trmmR symm; do
        echo "===== strassen_min=$v op=$op ====="
        PUREBLAS_FORCE_strassen_min=$v "$JL" --project=bench bench/plots.jl bench arms=pb "op=$op" \
            2>&1 | tee "$OUT.$v.$op.log" | grep -E "^  $op:" | tail -3
        echo
    done
done
echo "logs: $OUT.*.log"
