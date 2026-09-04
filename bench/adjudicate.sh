#!/usr/bin/env bash
# Adjudicate a borderline cell on the AUTHORITATIVE scale: K plots.jl replications, capturing the cell
# row after each so the per-run ratios can be sign-tested.
#
# WHY NOT cellrep. cellrep screens cheaply (8.4 s vs 34 s) but reads ~2pp pessimistic — validated at
# axpy n=1e6, where cellrep says 0.9676 [0.9602, 0.9798] and plots.jl says 0.9868 [0.9816, 0.9925],
# non-overlapping. That asymmetry makes cellrep SAFE FOR RETIRING (a pessimistic instrument reading
# >= 1.0 is conservative) but NOT for confirming a miss: any cell it puts within ~2pp of 1.0 could be a
# pass on the real scale. Those cells come here.
#
#   bench/adjudicate.sh <op> <size> [K]
set -uo pipefail
cd "$(dirname "$0")/.."
op="${1:?usage: adjudicate.sh <op> <size> [K]}"; n="${2:?}"; K="${3:-10}"
CORE="${SCREEN_CORE:-4}"
cache=$(ls bench/plots_data_*_$(hostname).txt 2>/dev/null | grep -v _lite | head -1)
[ -n "$cache" ] || { echo "no v3 cache for $(hostname)"; exit 1; }
log="bench/probes/adj_${op}_${n}_runs.log"; : > "$log"
err="bench/probes/adj_${op}_${n}_err.log"; : > "$err"
# GATE ON THE EXIT CODE. This loop used to send plots.jl to /dev/null and then read the cell out of the
# cache regardless of whether the run had written it. When plots.jl REFUSES — which it does, loudly and
# by design, if the frequency lock is not in the one valid state — every replication re-read the SAME
# STALE CELL, and the sign test below duly reported `below=0/K, sign p=0.0078, PASS`. Eight measurements
# that never happened, presented as a confident verdict. Measured 2026-09-04, off-lock on wintermute.
# The refusal was the tool working correctly; discarding its output is what turned it into a fabrication.
for i in $(seq 1 "$K"); do
    if ! taskset -c "$CORE" julia --startup-file=no --project=bench bench/plots.jl bench "op=$op" "size=$n" nodraw >> "$err" 2>&1; then
        echo "adjudicate: replication $i FAILED — plots.jl exited non-zero. Last lines:" >&2
        tail -6 "$err" >&2
        echo "adjudicate: ABORTING; a partial or stale log cannot be adjudicated." >&2
        exit 2
    fi
    awk -F'\t' -v o="$op" -v s="$n" '$2==o && $3==s' "$cache" >> "$log"
done
printf '%-22s ' "$op n=$n"
julia --startup-file=no -e '
using Statistics
rat = Float64[]
for ln in readlines(ARGS[1])
    p = split(ln, "\t"); d = Dict{String,Vector{Float64}}()
    for f in p[4:end]
        isempty(f) && continue
        # arm|time|commit|anchor|freq|samples. The field count GROWS (4 → 5 anchor → 6 per-cell freq)
        # and a fixed `limit=` threw on every bump ("cannot parse \"20.894|0.0088…\" as Float64").
        # The csv is ALWAYS last — index from the end, per plots.jl'"'"'s extension invariant.
        _p = split(f, "|"); a, s = _p[1], _p[end]; d[a] = parse.(Float64, split(s, ","))
    end
    haskey(d, "pb") || continue
    push!(rat, minimum(median(d[r] ./ d["pb"]) for r in ("openblas","aocl") if haskey(d, r)))
end
GATE_MIN = 0.995   # keep in sync with bench/gatecrit.jl (inline julia here cannot include it)
k = count(<(GATE_MIN), rat); n = length(rat)
lg(a,b) = sum(log, (a-b+1):a; init=0.0) - sum(log, 1:b; init=0.0)
tail(m,nn) = sum(exp(lg(nn,i)) for i in 0:m; init=0.0) / 2.0^nn
p = n == 0 ? 1.0 : min(1.0, 2 * (k <= n/2 ? tail(k,n) : tail(n-k,n)))
println("median=", n==0 ? "NA" : round(median(rat); digits=4), "  below=", k, "/", n,
        "  sign p=", round(p; digits=4), "   ",
        n == 0 ? "NO DATA" : (p < 0.05 && median(rat) < GATE_MIN ? "FAIL (real)" : "PASS≈ (retire)"))' "$log"
