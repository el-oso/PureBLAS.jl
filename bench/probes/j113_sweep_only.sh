#!/usr/bin/env bash
# Sweep-only re-run WITH A CLOCK WATCHDOG. For a box whose suite already passed at this exact commit
# (re-running it would cost ~25 min and add nothing) but whose previous sweep was invalidated.
#
# WHY THE WATCHDOG. neuromancer's 2026-08-16 sweep verified ✅ at launch (1981 MHz) and still stamped
# `freq=2172337kHz` — 2172 MHz against a 2000 MHz base — because the lock floated DURING the run. That
# was only discoverable from the final cache header, i.e. after 2.5 h of measuring. Sampling the achieved
# clock throughout turns a 2.5-hour loss into a ~1-minute one, and leaves a per-run record of clock
# stability rather than a single before/after reading.
set -u
cd "$HOME/Documents/claude/PureBLAS.jl" || exit 1

BASE_KHZ=$(cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq)
TOL_KHZ=$(( BASE_KHZ * 101 / 100 ))          # 1% over base ⇒ boost is floating
WLOG=/tmp/j113_clockwatch.log
: > "$WLOG"

echo "=== host: $(hostname)  HEAD: $(git rev-parse --short HEAD)"
julia --version
bash bench/fleet_freqlock.sh verify
echo "=== watchdog: base=${BASE_KHZ}kHz  flag above ${TOL_KHZ}kHz  -> $WLOG"

# Sample core 2 (the pinned sweep core) every 30 s for as long as the sweep runs.
(
    while :; do
        f=$(cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq 2>/dev/null || echo 0)
        if [ "$f" -gt "$TOL_KHZ" ]; then echo "$(date +%H:%M:%S) FLOATED ${f}kHz" >> "$WLOG"
        else echo "$(date +%H:%M:%S) ok ${f}kHz" >> "$WLOG"; fi
        sleep 30
    done
) &
WATCH=$!
trap 'kill $WATCH 2>/dev/null' EXIT

echo "=== sweep (both arms, pinned to core 2)"
taskset -c 2 julia --project=bench bench/plots.jl bench aocl nodraw > /tmp/j113_sweep.log 2>&1
echo "=== sweep exit=$?"

kill $WATCH 2>/dev/null
echo "=== clock watchdog summary:"
echo "  samples : $(wc -l < "$WLOG")"
echo "  floated : $(grep -c FLOATED "$WLOG")"
echo "  max seen: $(awk '{print $3}' "$WLOG" | tr -d 'kHz' | sort -n | tail -1) kHz  (base ${BASE_KHZ})"
echo "=== header provenance:"
head -1 bench/plots_data_*[!e].txt | tr '\t' '\n' | grep -E "^(commit|julia|llvm|ob|freq|hw)="
