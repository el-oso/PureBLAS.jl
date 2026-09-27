#!/usr/bin/env bash
# Did a cache's sweep run against a FOREIGN PROCESS? `plots.jl` stamps `busy=<pid>:<pct>` into the header
# when its contention guard saw one, and then measures anyway. Nothing read that field, so the guard
# recorded the contamination and no step ever looked.
#
# WHY IT HAS TO BE A SEPARATE, AFTER-THE-FACT CHECK: the guard is START-ONLY. It refuses when a foreign
# process is already at >= 25% CPU, so a contender that appears three cells in is recorded rather than
# refused, and `fleet_refresh.sh` sees exit 0 and prints REFRESH DONE. Measured 2026-09-26: galen's
# threaded L1+L3 groups ran alongside a process at 97.2% CPU. The numbers looked like a scaling defect —
# dot at n=3e5 read 1.13x self-speedup where a probe on the quiet box read 3.72x — and the whole
# diagnosis pointed at the kernel. The field was in the header the entire time.
#
# A co-tenant does not always contaminate: `SWEEP_EXTRA=force-busy` is the honest flag for a resident job
# pinned to the OTHER die on a multi-CCD part, where the two share no L3. That is why this reports the
# recorded percentage and the PID rather than just failing — the reader decides whether the sweep and the
# contender shared a cache. It exits non-zero either way, so a chain cannot pass over it silently.
#
#   bash bench/check_cache_busy.sh bench/mt_data_*.txt
set -u
cd "$(dirname "$0")/.." || exit 2

[ "$#" -gt 0 ] || { echo "usage: $0 <cache file>..."; exit 2; }

rc=0
for f in "$@"; do
    [ -f "$f" ] || { printf '  %-38s MISSING\n' "$(basename "$f")"; rc=1; continue; }
    b=$(head -1 "$f" | tr '\t' '\n' | grep -m1 '^busy=' | cut -d= -f2-)
    if [ -z "$b" ]; then
        printf '  %-38s quiet box (no busy field)\n' "$(basename "$f")"
    else
        pct=${b#*:}; pid=${b%%:*}
        printf '  %-38s CONTENDED  pid %s at %.1f%% CPU\n' "$(basename "$f")" "$pid" "${pct%%.*}"
        rc=1
    fi
done

if [ "$rc" -ne 0 ]; then
    printf '\n'
    echo "=> A sweep shared this box with another process. Those cells time PureBLAS against a reference"
    echo "   while a third party holds cores, so a threaded arm loses more than a serial one and the"
    echo "   ratio reads as a scaling defect. Re-sweep the affected groups on a quiet box, unless the"
    echo "   contender was pinned to a die the sweep does not share (then it was a force-busy run and the"
    echo "   header says so on purpose)."
fi
exit $rc
