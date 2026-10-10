#!/bin/bash
# PER-CELL LEDGER — an append-only record of what every cell read, so "better or worse" is a lookup.
#
# A sweep OVERWRITES the arms it re-measures, so once it has run, the previous values for those cells
# are gone and a per-cell before/after cannot be reconstructed. Answering "did anything regress" then
# costs two indirect measurements instead of one subtraction. This records each cache's cells as it
# finds them; run it BEFORE a sweep (on the copy about to be replaced) and again after.
#
#   bench/ledger/ledger.sh record <cache.txt> [box-label]   # append; idempotent per (cell, arm date)
#   bench/ledger/ledger.sh diff <box> <earlier-date> <later-date>
#
# Columns of bench/ledger/cells.tsv, tab-separated:
#   box group op n arm_date commit anchor pb_ms pb_mt_ms selfsp gate spread_pb spread_mt
#
# `selfsp` is pb/pb_mt — scaling, needs no reference arm. `gate` is the threaded criterion,
# min over the threaded vendor arms of ref/pb_mt, and is EMPTY when the cache holds no vendor arm
# (an `arms=pb,pb_mt` run), which is honest rather than 1.0.
#
# ESTIMATOR: every figure is the MEDIAN of that arm's per-round medians. An arm's sample field holds
# 48 quantiles PER ROUND concatenated, ascending within a round and resetting at each boundary, so a
# max/min over the whole vector is meaningless (it reads below 1.0 on real cells). `spread_*` is the
# inter-round max/min of those per-round medians — a ratio is not a result until both arms' spreads
# are known, because a bimodal arm's median is whichever mode that run sat in.
set -euo pipefail
cd "$(dirname "$0")/../.."
LEDGER=bench/ledger/cells.tsv

_extract() {   # <cache> <box>
  awk -F'\t' -v BOX="$2" '
    BEGIN { Q = 48; OFS = "\t" }
    NR == 1 { for (i = 1; i <= NF; i++) if ($i ~ /^commit=/) commit = substr($i, 8); next }
    {
      delete md; delete sp; delete dt
      for (i = 4; i <= NF; i++) {
        k = split($i, a, "|"); if (k < 9) continue
        m = split(a[9], q, ","); if (m < Q || m % Q != 0) continue
        nr = m / Q; n = 0; lo = 1e30; hi = 0
        for (r = 0; r < nr; r++) { v = q[r * Q + 24] + 0; if (v <= 0) continue
          mv[n++] = v; if (v < lo) lo = v; if (v > hi) hi = v }
        if (n == 0) continue
        for (x = 1; x < n; x++) { y = mv[x]; p = x - 1
          while (p >= 0 && mv[p] > y) { mv[p + 1] = mv[p]; p-- }; mv[p + 1] = y }
        md[a[1]] = mv[int((n - 1) / 2)]; sp[a[1]] = (lo > 0 ? hi / lo : 0)
        dt[a[1]] = substr(a[2], 1, 10); anchor = a[4]
        delete mv
      }
      if (!("pb" in md) || !("pb_mt" in md)) next
      selfsp = md["pb"] / md["pb_mt"]
      gate = ""
      for (ref in md) if (ref == "openblas_mt" || ref == "aocl_mt") {
        g = md[ref] / md["pb_mt"]
        if (gate == "" || g < gate) gate = g
      }
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.6g\t%.6g\t%.4f\t%s\t%.4f\t%.4f\n",
        BOX, $1, $2, $3, dt["pb_mt"], commit, anchor,
        md["pb"] * 1000, md["pb_mt"] * 1000, selfsp,
        (gate == "" ? "" : sprintf("%.4f", gate)), sp["pb"], sp["pb_mt"]
    }' "$1"
}

case "${1:-}" in
  record)
    cache="${2:?usage: ledger.sh record <cache.txt> [box-label]}"
    [ -f "$cache" ] || { echo "no such cache: $cache" >&2; exit 2; }
    box="${3:-$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++) if($i ~ /^host=/) print substr($i,6)}' "$cache")}"
    [ -n "$box" ] || { echo "cannot read host= from $cache; pass a box label" >&2; exit 2; }
    if [ ! -f "$LEDGER" ]; then
      printf 'box\tgroup\top\tn\tarm_date\tcommit\tanchor\tpb_ms\tpb_mt_ms\tselfsp\tgate\tspread_pb\tspread_mt\n' > "$LEDGER"
    fi
    tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
    _extract "$cache" "$box" > "$tmp"
    # IDEMPOTENT on (box, group, op, n, arm_date): re-running after a partial sweep adds only the
    # groups that moved, so the ledger never gains a duplicate row for a cell it already holds.
    added=$(awk -F'\t' 'NR==FNR{seen[$1"|"$2"|"$3"|"$4"|"$5]=1; next}
                        !($1"|"$2"|"$3"|"$4"|"$5 in seen){print; c++} END{print c+0 > "/dev/stderr"}' \
              "$LEDGER" "$tmp" 2>&1 >>"$LEDGER" | tail -1)
    echo "recorded $box: ${added:-0} new row(s); ledger now $(($(wc -l < "$LEDGER") - 1)) rows"
    ;;
  diff)
    box="${2:?usage: ledger.sh diff <box> <earlier-date> <later-date>}"; d0="${3:?}"; d1="${4:?}"
    awk -F'\t' -v BOX="$box" -v D0="$d0" -v D1="$d1" '
      NR == 1 { next }
      $1 != BOX { next }
      { key = $2 "/" $3 "@" $4
        if ($5 == D0) { a[key] = $10; ag[key] = $11 }
        if ($5 == D1) { b[key] = $10; bg[key] = $11 } }
      END {
        printf "%-22s %9s %9s %8s   %9s %9s\n", "cell", "selfsp0", "selfsp1", "change", "gate0", "gate1"
        for (k in b) if (k in a) {
          ch = (a[k] > 0 ? b[k] / a[k] : 0); n++
          if (ch < 0.97 || ch > 1.03) {
            printf "%-22s %9.3f %9.3f %7.2f%%   %9s %9s\n", k, a[k], b[k], (ch - 1) * 100,
              (ag[k] == "" ? "-" : ag[k]), (bg[k] == "" ? "-" : bg[k])
            moved++
          }
        }
        printf "\n%d cell(s) in both dates; %d moved by more than 3%%\n", n, moved + 0
      }' "$LEDGER"
    ;;
  *) echo "usage: $0 {record <cache.txt> [box] | diff <box> <earlier-date> <later-date>}"; exit 2 ;;
esac
