#!/usr/bin/env bash
# Frequency-lock a fleet box to its BASE clock for drift-free OB-vs-PB benchmarking — robust against the
# amd_pstate ACTIVE (epp) trap.
#
# Why base clock (boost off): the plotted number is a RATIO to OpenBLAS, so absolute speed is irrelevant;
# what matters is that the clock does not DRIFT between the OB window and the PB window. The base clock is
# the only thermally-sustainable one, so it stays flat across a multi-minute sweep. Boost floats with
# thermals → wide, unreproducible ratios. (This box's base is ~2.0 GHz; everything above is CPB/boost.)
#
# The trap this fixes: with the amd-pstate-epp driver in `active` mode, the kernel manages frequency via EPP
# hints and SILENTLY REVERTS manual `scaling_max_freq` clamps and ignores the `boost` node — so the obvious
# `echo 0 > boost` / cpufreq_lock.sh "just don't work" (they report success while the cores keep boosting).
# The clamps only stick in `passive` mode. Some kernels refuse a *runtime* active→passive switch and require
# it on the boot cmdline — so if the runtime switch is rejected, this script persists `amd_pstate=passive`
# to grub and asks for a reboot, then works on the second run.
#
# NEVER trusts the sysfs node readings — every lock is VERIFIED by measuring the real achieved frequency of
# the benchmark core under actual load (perf `cycles`, or scaling_cur_freq sampled under load as fallback).
#
# Usage (run on the target box):
#   sudo bench/fleet_freqlock.sh lock      # passive + base clock, boost off  ← what the gate/plots want
#   sudo bench/fleet_freqlock.sh pin 2000  # passive + hard-pin all cores to 2000 MHz (min=max)
#   sudo bench/fleet_freqlock.sh restore   # back to active/epp, boost on, full range
#        bench/fleet_freqlock.sh verify     # (no sudo) measure achieved freq of the bench core under load
# Env: CORE=<n> selects the core to verify (default 8, neuromancer's `taskset -c 8` bench core).

set -euo pipefail
STATUS=/sys/devices/system/cpu/amd_pstate/status
BOOST=/sys/devices/system/cpu/cpufreq/boost
CORE="${CORE:-8}"
cpus() { for d in /sys/devices/system/cpu/cpu[0-9]*; do [ -d "$d/cpufreq" ] && echo "$d/cpufreq"; done; }
need_root() { [ "$(id -u)" -eq 0 ] || { echo "!! needs root: sudo $0 $*"; exit 1; }; }

# Real achieved MHz of $1 under load — perf counts actual CPU cycles over ~1 s (immune to the sysfs lies).
achieved_mhz() {
    local core="$1" c
    if command -v perf >/dev/null 2>&1; then
        c=$(perf stat -x, -e cycles -- taskset -c "$core" timeout 1 bash -c 'while :; do :; done' 2>&1 \
            | awk -F, 'tolower($0) ~ /cycles/ {gsub(/ /,"",$1); print $1; exit}')
        [[ "$c" =~ ^[0-9]+$ ]] && { echo $(( c / 1000000 )); return; }
    fi
    taskset -c "$core" timeout 2 bash -c 'while :; do :; done' & local pid=$! s=0 n=0
    sleep 0.4
    for _ in 1 2 3; do s=$(( s + $(cat "/sys/devices/system/cpu/cpu$core/cpufreq/scaling_cur_freq") )); n=$((n+1)); sleep 0.35; done
    wait "$pid" 2>/dev/null || true
    echo $(( s / n / 1000 ))
}

# Ensure passive mode. 0=passive now; 1=runtime switch rejected (needs reboot); 3=not an amd_pstate box.
ensure_passive() {
    [ -e "$STATUS" ] || { echo "  (no amd_pstate — plain cpufreq box; boost node should work directly)"; return 3; }
    local m; m=$(cat "$STATUS")
    [ "$m" = passive ] && { echo "  amd_pstate already passive"; return 0; }
    echo passive > "$STATUS" 2>/dev/null || true
    m=$(cat "$STATUS")
    [ "$m" = passive ] && { echo "  amd_pstate: active → passive (runtime)"; return 0; }
    return 1
}

persist_grub() {
    local g=/etc/default/grub p="amd_pstate=passive"
    [ -f "$g" ] || { echo "!! $g not found — add '$p' to your kernel cmdline manually, then reboot."; return 1; }
    if grep -q "$p" "$g"; then echo "  grub already carries $p"; else
        cp "$g" "$g.bak.$(date +%s)"; echo "  backed up $g"
        sed -i "s/\(GRUB_CMDLINE_LINUX_DEFAULT=\"[^\"]*\)\"/\1 $p\"/" "$g"
        grep -q "$p" "$g" || { echo "!! failed to edit $g — add '$p' manually."; return 1; }
        echo "  added $p to GRUB_CMDLINE_LINUX_DEFAULT"
    fi
    if   command -v update-grub    >/dev/null 2>&1; then update-grub
    elif command -v grub-mkconfig  >/dev/null 2>&1; then grub-mkconfig -o /boot/grub/grub.cfg
    elif command -v grub2-mkconfig >/dev/null 2>&1; then grub2-mkconfig -o /boot/grub2/grub.cfg
    else echo "!! no update-grub found — regenerate grub.cfg yourself."; return 1; fi
}

# Hard-pin every core to $1 kHz (min=max) under the performance governor.
pin_khz() {
    local khz="$1"
    for f in $(cpus); do
        echo performance > "$f/scaling_governor" 2>/dev/null || true
        echo "$khz" > "$f/scaling_max_freq"
        echo "$khz" > "$f/scaling_min_freq"
    done
}

# Assert the measured freq is within 12% of target $1 (MHz), else FAIL loudly (don't ship a boosting run).
verify_or_die() {
    local target="$1" got; got=$(achieved_mhz "$CORE")
    local lo=$(( target * 88 / 100 )) hi=$(( target * 112 / 100 ))
    if [ "$got" -ge "$lo" ] && [ "$got" -le "$hi" ]; then
        echo "✅ VERIFIED core$CORE under load = ${got} MHz (target ${target}) — locked, safe to benchmark."
    else
        echo "❌ core$CORE measured ${got} MHz, target ${target} — NOT locked (still floating/boosting)."
        echo "   Do NOT benchmark. If a reboot was just requested, reboot then re-run 'sudo $0 lock'."
        exit 2
    fi
}

case "${1:-verify}" in
  lock)
    need_root lock
    echo "Locking $(hostname) to base clock (boost off)…"
    if ensure_passive; then
        echo 0 > "$BOOST" 2>/dev/null || true                 # passive honors this → cpuinfo_max drops to base
        base=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq)
        pin_khz "$base"
        verify_or_die $(( base / 1000 ))
    else
        echo "  runtime active→passive REJECTED by kernel — persisting the boot param instead:"
        persist_grub
        echo ""
        echo ">>> REBOOT this box, reconnect the tunnel, then run:  sudo $0 lock"
        echo "    (after the reboot it comes up passive and the lock will stick + verify)"
        exit 3
    fi ;;
  pin)
    need_root pin
    mhz="${2:?usage: sudo $0 pin <MHz>}"
    ensure_passive || { echo "  passive needed for a hard pin; persisting boot param:"; persist_grub; echo ">>> reboot, reconnect, re-run."; exit 3; }
    pin_khz $(( mhz * 1000 ))
    verify_or_die "$mhz" ;;
  restore)
    need_root restore
    echo active > "$STATUS" 2>/dev/null || true
    echo 1 > "$BOOST" 2>/dev/null || true
    for f in $(cpus); do
        echo "$(cat "$f/cpuinfo_max_freq")" > "$f/scaling_max_freq" 2>/dev/null || true
        echo "$(cat "$f/cpuinfo_min_freq")" > "$f/scaling_min_freq" 2>/dev/null || true
    done
    echo "Restored: amd_pstate=$(cat "$STATUS" 2>/dev/null||echo n/a), boost on, full range." ;;
  verify)
    echo "amd_pstate=$(cat "$STATUS" 2>/dev/null || echo n/a)  boost=$(cat "$BOOST" 2>/dev/null || echo n/a)"
    echo "core$CORE achieved under load = $(achieved_mhz "$CORE") MHz" ;;
  *) echo "usage: sudo $0 {lock|pin <MHz>|restore|verify}"; exit 1 ;;
esac
