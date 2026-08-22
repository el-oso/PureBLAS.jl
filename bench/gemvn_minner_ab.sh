#!/usr/bin/env bash
# Redo the gemvn_minner A/B on a VERIFIED-LOCKED box, as level2.jl:57-66 asks for.
# The existing Zen5 negative (n=1024 0.91 -> 0.85) was measured while that box ran at 4841 MHz against
# a 2000 MHz base — the lock had silently dropped. Both arms forced, so plots.jl refuses the cache write
# and neither arm can pollute published data; the per-round output is the A/B result.
set -u
cd "$(dirname "$0")/.."
JL="${JULIA:-julia}"; OUT="${1:-/tmp/minner}"
echo "PRE-LOCK:"; bash bench/fleet_freqlock.sh verify 2>&1 | tail -2
for v in 1 0; do
    echo "===== gemvn_minner=$v ====="
    PUREBLAS_FORCE_gemvn_minner=$v "$JL" --project=bench bench/plots.jl bench arms=pb op=gemvN \
        > "$OUT.$v.log" 2>&1
    grep -E "^L2 +gemvN" "$OUT.$v.log" | head -1
done
echo "POST-LOCK:"; bash bench/fleet_freqlock.sh verify 2>&1 | tail -2
echo "MINNER-AB-DONE"
