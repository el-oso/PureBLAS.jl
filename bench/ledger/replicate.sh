#!/bin/bash
# K INDEPENDENT PROCESSES FOR ONE CELL — the only thing that can CERTIFY a threaded verdict.
#
# `bench/ledger/cells.tsv`'s `spread_*` columns are inter-round, so they live inside one process, and a
# round reuses one page-colouring draw for both arms. Only a new process re-rolls it. `cellrep.jl`'s
# header records what that costs: at axpy n=1e6, 10 in-process rounds read 1.0153 [1.0000, 1.0214]
# where 10 FRESH PROCESSES read 0.9868 with 9 of 10 below 1.0 — the opposite sign, not a wider
# interval. So a tight in-process spread disqualifies nothing and confirms nothing. This does.
#
#   BENCH_CORE=0,2,4,6,8,10,1 bench/ledger/replicate.sh trmm 32 6 10
#
# ARMS ARE BOTH PUREBLAS — the same closure on the same operands at 1 thread and at `nt` — so no
# vendor library is forwarded, nothing is measured against a reference, and this needs no
# authorisation. It answers "does threading help or hurt this cell", never "do we beat the vendor".
#
# THE VERDICT IS A SIGN TEST, not a confidence interval on a mean. K processes each give one ratio;
# what matters is the median and how many fall the same side of 1.0, because the across-process
# distribution is the thing that is bimodal. 9 of 10 below 1.0 is a result; a median of 0.98 with 5
# below is not.
set -euo pipefail
cd "$(dirname "$0")/../.."
op="${1:?usage: replicate.sh <op> <n> <threads> [K=10]}"; n="${2:?}"; nt="${3:?}"; K="${4:-10}"

# The Julia thread count must EXCEED the pool width: `plots.jl`'s `_pin_threads!` pins one thread per
# physical core and the runtime needs an unpinned slot, so `-t nt` leaves the join waiting on a pinned
# thread that cannot move. `cellrep.jl` prints what the pool actually accepted, which is the check.
jt=$((nt + 1))
pre=""
[ -n "${BENCH_CORE:-}" ] && pre="taskset -c ${BENCH_CORE}"

# PRE-WARM FIRST, OR THE GUARD BELOW CATCHES THIS SCRIPT'S OWN COMPILER. `--project=bench` keeps its
# own pkgimage, so the first julia launched here can spend a minute at 100% precompiling PureBLAS —
# and the contention check then correctly refuses, naming a process this script started. Observed
# twice: two orphaned precompiles at 99.9% on a box measured idle seconds earlier.
# `fleet_refresh.sh` carries the same step for the same reason, before its first timed group.
# Free when the cache is warm (a second or two of load).
julia --project=bench -e 'using PureBLAS, LinearAlgebra' >/dev/null 2>&1 || {
    echo "REFUSING: the bench environment will not load — fix that before measuring." >&2; exit 2; }

# THEN THE BOX, and only then the clock. The all-core clock check deliberately loads every core for
# ~20 s, so it has to come AFTER the quiet check or it fails it itself.
#
# SAMPLED, NOT A LIFETIME AVERAGE. `ps` averages over a process's whole life, which over-reports a
# finished burst and under-reports a fresh one; `top -bn2` discards its first sample and measures the
# interval, which is what the guard inside `plots.jl` does by reading /proc over a window.
#
# ANCHOR ON THE PROCESS-TABLE HEADER, not on a line number: `NR` is global across both of top's
# iterations, so a count-based skip lets the second iteration's summary lines ("%Cpu(s): … id",
# "MiB Mem : … used") through as if they were process rows — which they parse as, with a plausible
# number in the %CPU column. The first version here reported "used," and "28321.7" as busy processes.
_busy=$(top -bn2 -d 2 -o %CPU 2>/dev/null \
        | awk '/^top -/ { it++; rows = 0 } /^ *PID +USER/ { if (it == 2) rows = 1; next } rows && it == 2 { print $9, $12 }' \
        | awk '$1 + 0 >= 25 { printf "%6.1f%%  %s\n", $1, $2; n++ } END { exit !n }') && {
    echo "REFUSING: the box is not quiet — a contended run produces a spread, not a ratio." >&2
    printf '%s\n' "$_busy" | sed 's/^/    /' >&2
    echo "  Wait for it, or pass FORCE_BUSY=1 if you accept the measurement is a screen only." >&2
    [ -n "${FORCE_BUSY:-}" ] || exit 2
    echo "  FORCE_BUSY set — continuing, and the numbers below are NOT adjudicable." >&2; }

bash bench/fleet_freqlock.sh verify-mt 2>&1 | tail -1 | grep -q '✅' || {
    echo "REFUSING: the all-core clock check does not pass — a threaded ratio here is not adjudicable." >&2
    exit 2; }

log=$(mktemp); trap 'rm -f "$log"' EXIT
echo "=== $op n=$n, $nt threads, $K independent processes ==="
for i in $(seq 1 "$K"); do
    $pre julia -t "$jt" --project=bench bench/cellrep.jl "$op" "$n" "$nt" 2>/dev/null >> "$log" || {
        echo "  process $i FAILED — not counted" >&2; }
done

# ONE-BASED, AND `c` INITIALISED. An uninitialised awk variable used as a SUBSCRIPT is the empty
# string, not 0 — so `r[c] = ...` on the first record creates `r[""]` while `c++` then makes the next
# index 1, and a sort that reads `r[0]` finds nothing and treats it as zero. That printed a perfectly
# plausible `min 0.0000` beside a correct median, which is the dangerous shape: a wrong number that
# does not look wrong. Verified against a clean three-record log before and after.
awk -F'\t' -v K="$K" '
  BEGIN { c = 0 }
  { nt_asked = $3; nt_got = $4; r[++c] = $5 + 0
    if (nt_got != nt_asked) clamp++ }
  END {
    if (c == 0) { print "  no usable lines — every process failed"; exit 1 }
    if (clamp > 0) printf "  ⚠ %d of %d process(es) had the pool CLAMP below the requested width — discard and re-run with more -t\n", clamp, c
    for (i = 2; i <= c; i++) { t = r[i]; j = i - 1
      while (j >= 1 && r[j] > t) { r[j + 1] = r[j]; j-- }; r[j + 1] = t }
    med = (c % 2 == 1) ? r[int((c + 1) / 2)] : (r[c / 2] + r[c / 2 + 1]) / 2
    below = 0; for (i = 1; i <= c; i++) if (r[i] < 1.0) below++
    printf "  self-speedup over %d process(es): median %.4f   min %.4f   max %.4f   spread %.3f\n",
      c, med, r[1], r[c], (r[1] > 0 ? r[c] / r[1] : 0)
    printf "  %d of %d below 1.0\n", below, c
    # A verdict only where the sign is nearly unanimous AND the median is clear of 1.0. Anything else
    # is reported as not adjudicable rather than rounded into a pass or a fail.
    # THE SIGN AND THE MAGNITUDE ARE SEPARATE CLAIMS, and a wide across-process spread kills the
    # second without touching the first. Measured here: galen axpy@300000 reads median 2.43 with 9 of
    # 10 processes above 1.0 — the sign is beyond doubt — while min 0.85 and max 3.81 put the spread at
    # 4.49, so "threading helps" is established and "threading gives 2.43x" is not. Reporting only the
    # median invites the second reading, so a spread past 1.5 says so in the verdict line.
    spr = (r[1] > 0 ? r[c] / r[1] : 0)
    mag = (spr > 1.5) ? sprintf(" ⚠ SIGN only — spread %.2f across processes, so the magnitude is NOT established", spr) : ""
    if (med < 0.98 && below >= c - 1)
      printf "  VERDICT: threading LOSES here — %d/%d processes agree, median %.4f%s\n", below, c, med, mag
    else if (med > 1.02 && below <= 1)
      printf "  VERDICT: threading WINS here — %d/%d processes agree, median %.4f%s\n", c - below, c, med, mag
    else
      printf "  VERDICT: NOT ADJUDICABLE — median %.4f with %d/%d below 1.0; the across-process draw dominates the effect\n", med, below, c
  }' "$log"
