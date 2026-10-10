#!/bin/bash
# PER-CELL LEDGER — an append-only record of what every cell read, so "better or worse" is a lookup.
#
# A sweep OVERWRITES the arms it re-measures, so once it has run, the previous values for those cells
# are gone and a per-cell before/after cannot be reconstructed. Answering "did anything regress" then
# costs two indirect measurements instead of one subtraction. This records each cache's cells as it
# finds them; run it BEFORE a sweep (on the copy about to be replaced) and again after.
#
#   bench/ledger/ledger.sh record <cache.txt> [box-label]   # append; idempotent per (cell, arm stamp)
#   bench/ledger/ledger.sh diff <box> <earlier-commit> <later-commit>
#   bench/ledger/ledger.sh stamps [box]                     # what commits/times are recorded
#
# Columns of bench/ledger/cells.tsv, tab-separated:
#   box group op n arm_time commit anchor pb_ms pb_mt_ms selfsp gate spread_pb spread_mt
#
# PROVENANCE IS PER ARM, NEVER FROM THE HEADER. `commit` and `anchor` come from the `pb_mt` arm's own
# record, because a targeted `op=`/`group=` merge rewrites the header while the file's other cells keep
# the commit they were measured at — `plots.jl` writes a per-arm commit for exactly this reason and its
# comment calls a header stamp "the same per-row provenance hazard `coverage_ops.jl` documents". A
# galen cache measured across two runs holds 1008 arms at one commit and 110 at another under a single
# header; stamping the header against all of them is a false record of 1008 cells.
#
# `arm_time` keeps the full `%Y-%m-%dT%H:%M` the writer stores, and the dedup key uses it. Truncating
# to a date makes a second sweep on the same day match every existing row, so `record` adds nothing
# and reports success — which is this script's primary workflow failing on its commonest case. A
# sweep also SPANS stamps: 12 groups with a 300 s gap runs for hours and crosses midnight, so one
# sweep's cells carry many times and several commits. `diff` therefore keys on COMMIT, not on time,
# and prints how many cells each side holds so partial coverage cannot read as full coverage.
#
# ESTIMATOR: a per-round figure is the MEDIAN of that round's 48 stored quantiles, and an arm's figure
# is the median over rounds — the true median, averaging the two middle values when the count is even,
# because the lower median equals the MINIMUM at the 2 rounds `_rounds_light` uses and `min` is the one
# estimator this project forbids outright. `spread_*` is the inter-round max/min and is EMPTY, not
# 1.0000, when an arm has a single round: a spread of 1 by construction must not read as a measured
# tight arm. The `gate` column divides the two arms' quantile vectors ELEMENT-WISE and takes the median
# of the ratios, which is the reduction `plots.jl` uses (`_qvec`'s comment: the pairing "is preserved,
# nothing is approximated"); dividing two separately reduced scalars disagrees with it on a bimodal
# arm. It is EMPTY when the cache holds no threaded vendor arm, so an `arms=pb,pb_mt` run cannot be
# mistaken for a scored one.
set -euo pipefail
cd "$(dirname "$0")/../.."
LEDGER=bench/ledger/cells.tsv

_extract() {   # <cache> <box>
  awk -F'\t' -v BOX="$2" '
    function med(a, n,   i, j, t) {            # true median of a[0..n-1], sorted in place
      for (i = 1; i < n; i++) { t = a[i]; j = i - 1
        while (j >= 0 && a[j] > t) { a[j + 1] = a[j]; j-- }; a[j + 1] = t }
      if (n % 2 == 1) return a[int((n - 1) / 2)]
      return (a[n / 2 - 1] + a[n / 2]) / 2
    }
    BEGIN { Q = 48; OFS = "\t" }
    NR == 1 { next }                            # the header carries no per-cell truth; see the note above
    {
      delete md; delete sp; delete tm; delete cm; delete an; delete nrd; delete vec
      for (i = 4; i <= NF; i++) {
        k = split($i, a, "|"); if (k < 9) continue
        arm = a[1]
        # THE CSV IS ALWAYS THE LAST FIELD. `plots.jl` states the extension rule outright -- "append
        # before the csv, never after it" -- and `load_cache` indexes from both ends. A hardcoded 9
        # would silently skip EVERY arm the day a tenth field lands, and report 0 new rows, exit 0.
        m = split(a[k], q, ","); if (m < Q || m % Q != 0) continue
        nr = m / Q; n = 0; lo = 1e30; hi = 0
        for (r = 0; r < nr; r++) {
          delete rq
          for (j = 1; j <= Q; j++) rq[j - 1] = q[r * Q + j] + 0
          v = med(rq, Q)
          if (v <= 0) continue
          mv[n++] = v; if (v < lo) lo = v; if (v > hi) hi = v
        }
        if (n == 0) continue
        md[arm] = med(mv, n); delete mv
        nrd[arm] = nr
        sp[arm] = (nr >= 2 && lo > 0) ? hi / lo : ""
        tm[arm] = a[2]; cm[arm] = a[3]; an[arm] = a[4]
        for (j = 1; j <= m; j++) vec[arm, j] = q[j] + 0
        vlen[arm] = m
      }
      if (!("pb" in md) || !("pb_mt" in md)) next
      selfsp = md["pb"] / md["pb_mt"]
      # The gate ratio, formed element-wise against the threaded vendor arms and reduced by median --
      # the same pairing `plots.jl` preserves. Taken over the arm whose median ratio is SMALLEST,
      # which is `min over refs`, the rule `_series` applies.
      gate = ""
      for (ref in md) {
        if (ref != "openblas_mt" && ref != "aocl_mt") continue
        if (vlen[ref] != vlen["pb_mt"]) continue        # different round counts cannot be paired
        nn = 0
        for (j = 1; j <= vlen[ref]; j++) if (vec["pb_mt", j] > 0) rr[nn++] = vec[ref, j] / vec["pb_mt", j]
        if (nn == 0) continue
        g = med(rr, nn); delete rr
        if (gate == "" || g < gate) gate = g
      }
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.6g\t%.6g\t%.4f\t%s\t%s\t%s\n",
        BOX, $1, $2, $3, tm["pb_mt"], cm["pb_mt"], an["pb_mt"],
        md["pb"] * 1000, md["pb_mt"] * 1000, selfsp,
        (gate == "" ? "" : sprintf("%.4f", gate)),
        (sp["pb"] == "" ? "" : sprintf("%.4f", sp["pb"])),
        (sp["pb_mt"] == "" ? "" : sprintf("%.4f", sp["pb_mt"]))
    }' "$1"
}

case "${1:-}" in
  record)
    cache="${2:?usage: ledger.sh record <cache.txt> [box-label]}"
    [ -f "$cache" ] || { echo "no such cache: $cache" >&2; exit 2; }
    box="${3:-$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++) if($i ~ /^host=/) print substr($i,6)}' "$cache")}"
    [ -n "$box" ] || { echo "cannot read host= from $cache; pass a box label" >&2; exit 2; }
    if [ ! -f "$LEDGER" ]; then
      printf 'box\tgroup\top\tn\tarm_time\tcommit\tanchor\tpb_ms\tpb_mt_ms\tselfsp\tgate\tspread_pb\tspread_mt\n' > "$LEDGER"
    fi
    tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
    _extract "$cache" "$box" > "$tmp"
    [ -s "$tmp" ] && : || { echo "extracted NOTHING from $cache — refusing to report success" >&2; exit 3; }
    added=$(awk -F'\t' 'NR==FNR{seen[$1"|"$2"|"$3"|"$4"|"$5]=1; next}
                        !($1"|"$2"|"$3"|"$4"|"$5 in seen){print; c++} END{print c+0 > "/dev/stderr"}' \
              "$LEDGER" "$tmp" 2>&1 >>"$LEDGER" | tail -1)
    echo "recorded $box: ${added:-0} new row(s) from $(wc -l < "$tmp") cell(s); ledger now $(($(wc -l < "$LEDGER") - 1)) rows"
    ;;
  stamps)
    box="${2:-}"
    awk -F'\t' -v BOX="$box" 'NR>1 && (BOX=="" || $1==BOX) {c[$1"\t"$6]++; lo[$1"\t"$6]=(lo[$1"\t"$6]==""||$5<lo[$1"\t"$6])?$5:lo[$1"\t"$6]; hi[$1"\t"$6]=($5>hi[$1"\t"$6])?$5:hi[$1"\t"$6]}
      END{printf "%-12s %-10s %6s  %s\n","box","commit","cells","arm times"; for(k in c){split(k,p,"\t"); printf "%-12s %-10s %6d  %s .. %s\n", p[1], p[2], c[k], lo[k], hi[k]}}' "$LEDGER" | sort
    ;;
  diff)
    box="${2:?usage: ledger.sh diff <box> <earlier-commit> <later-commit>}"; c0="${3:?}"; c1="${4:?}"
    awk -F'\t' -v BOX="$box" -v C0="$c0" -v C1="$c1" '
      NR == 1 { next }
      $1 != BOX { next }
      { key = $2 "/" $3 "@" $4
        if ($6 == C0) { n0++; a[key] = $10; as[key] = $9; aa[key] = $7 + 0 }
        if ($6 == C1) { n1++; b[key] = $10; bs[key] = $9; ba[key] = $7 + 0 } }
      END {
        if (n0 == 0 || n1 == 0) {
          printf "no cells for %s (%d rows) or %s (%d rows) on %s — run `stamps` to see what is recorded\n", C0, n0+0, C1, n1+0, BOX
          exit 1
        }
        printf "%-22s %8s %8s %8s   %9s %9s\n", "cell", "selfsp0", "selfsp1", "d.self", "thr_ms0", "thr_ms1"
        for (k in b) if (k in a) {
          both++
          cs = (a[k] > 0 ? b[k] / a[k] : 0)                      # self-speedup change
          ct = (as[k] > 0 ? bs[k] / as[k] : 0)                   # THREADED TIME change
          # TWO COMMITS ARE TWO RUNS, so the absolute times carry the machine state they were measured
          # under and the stored anchor is what reports it. Beyond 3% the times are not comparable and
          # the column is withheld rather than printed with a caveat nobody reads; `selfsp` is a
          # within-run ratio and stays valid either way. `check_arm_anchors.sh` uses the same 3%.
          ok = (aa[k] > 0 && ba[k] > 0 && aa[k] / ba[k] < 1.03 && ba[k] / aa[k] < 1.03)
          ok || noanch++
          # Report a cell when EITHER moved. `selfsp` is a ratio of the two arms, so it cancels any
          # change that slows both equally -- which is exactly the shape of a shared-kernel
          # regression. The absolute threaded time is what catches that, so both are tested.
          if (cs < 0.97 || cs > 1.03 || (ok && (ct < 0.97 || ct > 1.03))) {
            moved++
            printf "%-22s %8.3f %8.3f %7.1f%%   %9.4g %9.4g  %s\n", k, a[k], b[k], (cs - 1) * 100,
              as[k], bs[k],
              (!ok ? "anchor " sprintf("%.0f%%", (aa[k] / ba[k] - 1) * 100) " — times not comparable" :
               ct > 1.03 ? "SLOWER " sprintf("%.1f%%", (ct - 1) * 100) : ct < 0.97 ? "faster" : "")
          }
        }
        printf "\n%s: %d cells   %s: %d cells   in BOTH: %d   moved >3%%: %d\n", C0, n0, C1, n1, both+0, moved+0
        if (noanch > 0)
          printf "⚠ %d of %d shared cells have anchors more than 3%% apart — for those, only `selfsp` is read; the absolute times describe two different machine states.\n", noanch, both+0
        if (both < n0 || both < n1)
          printf "⚠ %d and %d cells are in only one side — a sweep spans several commits, so this is a SUBSET; check `stamps`.\n", n0-both, n1-both
      }' "$LEDGER"
    ;;
  *) echo "usage: $0 {record <cache.txt> [box] | diff <box> <commit0> <commit1> | stamps [box]}"; exit 2 ;;
esac
