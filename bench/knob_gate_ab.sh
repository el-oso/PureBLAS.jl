#!/usr/bin/env bash
# GATE A/B for one knob: shipped value vs candidate, on one box, one op.
#   usage: knob_gate_ab.sh <knob> <shipped> <candidate> <op> <outprefix>
#
# BOTH ARMS ARE FORCED, including the shipped one. That is deliberate:
#   1. it makes this a CONTROLLED back-to-back A/B in one methodology, instead of a fresh run compared
#      against a cache captured under different machine state (the standing cross-run rule), and
#   2. forcing both trips plots.jl's PUREBLAS_FORCE_* guard, so NEITHER arm can write the cache —
#      a forced run that reached the cache once cost three phantom gate misses (plots.jl:1454).
# Reference arms come from cache via arms=pb; OpenBLAS/AOCL are never re-timed.
set -u
cd "$(dirname "$0")/.."
KNOB=$1; SHIPPED=$2; CAND=$3; OP=$4; OUT=$5
JL="${JULIA:-julia}"
for v in "$SHIPPED" "$CAND"; do
    echo "===== $KNOB=$v ====="
    env "PUREBLAS_FORCE_$KNOB=$v" "$JL" --project=bench bench/plots.jl bench arms=pb "op=$OP" \
        > "$OUT.$v.log" 2>&1
    grep -E "^L3 |^CL3 " "$OUT.$v.log" | grep -iE "$OP" | head -3
done
echo "logs: $OUT.$SHIPPED.log $OUT.$CAND.log"
