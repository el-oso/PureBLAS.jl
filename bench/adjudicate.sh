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
for _ in $(seq 1 "$K"); do
    taskset -c "$CORE" julia --startup-file=no --project=bench bench/plots.jl bench "op=$op" "size=$n" nodraw > /dev/null 2>&1
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
        a,_,_,s = split(f, "|"; limit=4); d[a] = parse.(Float64, split(s, ","))
    end
    haskey(d, "pb") || continue
    push!(rat, minimum(median(d[r] ./ d["pb"]) for r in ("openblas","aocl") if haskey(d, r)))
end
k = count(<(1.0), rat); n = length(rat)
lg(a,b) = sum(log, (a-b+1):a; init=0.0) - sum(log, 1:b; init=0.0)
tail(m,nn) = sum(exp(lg(nn,i)) for i in 0:m; init=0.0) / 2.0^nn
p = n == 0 ? 1.0 : min(1.0, 2 * (k <= n/2 ? tail(k,n) : tail(n-k,n)))
println("median=", n==0 ? "NA" : round(median(rat); digits=4), "  below=", k, "/", n,
        "  sign p=", round(p; digits=4), "   ",
        n == 0 ? "NO DATA" : (p < 0.05 && median(rat) < 1 ? "FAIL (real)" : "PASS≈ (retire)"))' "$log"
