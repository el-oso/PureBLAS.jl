#!/usr/bin/env bash
# GATE validation of strassen_min = 512 against the shipped 1024, real dgemm, one box.
#
# WHY BOTH ARMS ARE FORCED. The probe said 512 wins by 2-4% PB-vs-PB; the gate asks a different
# question — the ratio against max(OpenBLAS, AOCL) — and a PB-vs-PB gain is not a gate win. The clean
# way to answer it is a CONTROLLED back-to-back A/B in one methodology, not a fresh run compared
# against a cache captured under different machine state (per the standing cross-run rule).
#
# So the shipped arm is forced to its OWN value (1024). That is deliberate and does two things:
#   1. it measures the shipped configuration, and
#   2. it trips plots.jl's PUREBLAS_FORCE_* guard, so NEITHER arm can write the cache.
# The second point matters: a forced run that reached the cache once cost three phantom gate misses
# and a day of kernel work (plots.jl:1454). Both arms print per-round output either way.
#
# Reference arms come from the cache (`arms=pb`) — never re-time OpenBLAS/AOCL.
set -u
cd "$(dirname "$0")/.."
JL="${JULIA:-julia}"
OUT="${1:-/tmp/strassen_gate}"

for v in 1024 512; do
    echo "===== strassen_min=$v ====="
    PUREBLAS_FORCE_strassen_min=$v "$JL" --project=bench bench/plots.jl bench arms=pb op=gemm \
        2>&1 | tee "$OUT.$v.log" | grep -E "gemm|ratio|PASS|FAIL|refus|cache" | tail -25
    echo
done
echo "logs: $OUT.1024.log $OUT.512.log"
