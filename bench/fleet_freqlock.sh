#!/usr/bin/env bash
# ============================================================================================
# THE FLEET FREQUENCY METHODOLOGY — SINGLE SOURCE OF TRUTH. DO NOT RE-DECIDE THIS PER SESSION.
# ============================================================================================
# THE canonical command for every gate / plot measurement, on every box:
#
#       sudo bench/fleet_freqlock.sh lock
#
# It puts the box in the ONE reproducible state: amd_pstate=passive + BOOST OFF + all cores pinned
# (min=max) to the HIGHEST CLOCK THE BOX SUSTAINS UNDER LOAD + the achieved frequency VERIFIED. That is
# base clock on a desktop (galen 5900X holds it), or the thermal/power ceiling on a power-limited laptop
# (the 7640U / AI-5-340 can't hold base — a 3501 pin measured only 2367 achieved — so `lock` auto-steps
# down to 90% of the sustained measurement, the highest FLAT clock). Run it once per box before a sweep;
# run `verify` (no sudo) any time to confirm the box is still locked. `restore` undoes it.
#
# STRICT RULES (agents + humans — non-negotiable, this ends the recurring flip-flop):
#   1. NEVER benchmark for the gate unless this script's `verify` reports ✅ (locked at base, boost off).
#      A boosting/floating clock (boost=1) gives wide, irreproducible OB/PB ratios — any number measured
#      there is INVALID and must be discarded, not "explained".
#   2. There is NO stable high pin. base < requested → impossible. Concretely: you CANNOT lock 4000 MHz on
#      a chip whose base is ~2 GHz — 4000 lives in the boost range, and boost frequencies float ABOVE
#      scaling_max_freq and cannot be held at a fixed value. `pin <MHz>` with MHz > base will FAIL verify
#      BY DESIGN (see below). If you want "a higher clock", the answer is: base clock is the ceiling for a
#      LOCKED run. Do not chase a fixed boost frequency; it does not exist as a stable state.
#   3. Absolute clock is IRRELEVANT to the gate — it is a PB/OB ratio, both sides run at the same clock.
#      So there is no benefit to a higher clock and a real cost (drift). Base-clock-locked is the answer,
#      permanently. If a future session is tempted to re-open this: don't. Measure on `lock`, full stop.
#
# Why base clock (boost off): the plotted number is a RATIO to OpenBLAS, so absolute speed is irrelevant;
# what matters is that the clock does not DRIFT between the OB window and the PB window. The base clock is
# the only thermally-sustainable one, so it stays flat across a multi-minute sweep. Boost floats with
# thermals → wide, unreproducible ratios. (Neuromancer/Zen5 base ~2.0 GHz, boost to 4.9; the 4000 "pin"
# tried earlier was in the boost range → it floated to ~4844 and was never actually locked.)
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
# NO SUDO NEEDED where the setuid helper is installed (bench/tools/README.md):
#       bench/fleet_freqlock.sh lock
# `lock`, `pin` and `restore` route every privileged write through /usr/local/sbin/pureblas-cpufreq when
# it is present and we are not root. Two things it cannot do, and the script says so when it hits them:
# stopping power-profiles-daemon/tuned (the helper never execs anything, by design) and the grub edit.
# Override the path with PUREBLAS_CPUFREQ_HELPER=<path>. Install it with:
#       sudo install -o root -g root -m 4755 bench/tools/build/pureblas-cpufreq /usr/local/sbin/
#
# Usage (run on the target box):
#   sudo bench/fleet_freqlock.sh lock      # ← THE canonical gate state: passive + boost OFF + base clock + verify
#   sudo bench/fleet_freqlock.sh pin 1800  # passive + hard-pin ≤ base (boost off, verified); >base is REFUSED
#   sudo bench/fleet_freqlock.sh restore   # back to active/epp, boost on, full range (daily-use state)
#        bench/fleet_freqlock.sh verify     # (no sudo) measure achieved freq of the bench core under load
# Env: CORE=<n> selects the core to verify (default 8, neuromancer's `taskset -c 8` bench core).
#      (wintermute bench core = 2, galen = 6, neuromancer = 8 — pass CORE=<n> to match the box.)

set -euo pipefail
STATUS=/sys/devices/system/cpu/amd_pstate/status
BOOST=/sys/devices/system/cpu/cpufreq/boost
CORE="${CORE:-8}"
cpus() { for d in /sys/devices/system/cpu/cpu[0-9]*; do [ -d "$d/cpufreq" ] && echo "$d/cpufreq"; done; }
# ── THE SETUID HELPER ───────────────────────────────────────────────────────────────────────────────
# `bench/tools/pureblas-cpufreq` (see bench/tools/README.md) is a small root-owned setuid binary that
# does exactly the privileged writes below and nothing else — no exec, no environment, no shell, one
# verb from a fixed table plus at most one validated integer. When it is installed, this script needs
# no sudo at all, which is the difference between "ask a human and wait" and "relock the box and carry
# on" for an agent mid-sweep.
#
# It does NOT cover `systemctl stop power-profiles-daemon` or the grub edit. Those still need root, and
# the lock path below says so out loud rather than silently producing a pin that PPD will revert.
HELPER=${PUREBLAS_CPUFREQ_HELPER:-/usr/local/sbin/pureblas-cpufreq}
have_helper() { [ -x "$HELPER" ]; }
is_root() { [ "$(id -u)" -eq 0 ]; }
# Privileged writes, each routed through the helper when we are not root. `|| true` throughout because
# `set -e` is on and a knob that does not exist on this platform is not a failure of the lock.
priv_pstate()   { if is_root; then echo "$1" > "$STATUS" 2>/dev/null || true; else "$HELPER" pstate "$1" >/dev/null 2>&1 || true; fi; }
priv_boost()    { if is_root; then echo "$1" > "$BOOST"  2>/dev/null || true; else "$HELPER" boost  "$1" >/dev/null 2>&1 || true; fi; }
priv_governor() { if is_root; then for f in $(cpus); do echo "$1" > "$f/scaling_governor" 2>/dev/null || true; done
                  else "$HELPER" governor "$1" >/dev/null 2>&1 || true; fi; }
priv_unpin()    { if is_root; then for f in $(cpus); do echo "$(cat "$f/cpuinfo_min_freq")" > "$f/scaling_min_freq" 2>/dev/null || true
                                                        echo "$(cat "$f/cpuinfo_max_freq")" > "$f/scaling_max_freq" 2>/dev/null || true; done
                  else "$HELPER" unpin >/dev/null 2>&1 || true; fi; }
priv_pin_raw()  { if is_root; then for f in $(cpus); do echo "$1" > "$f/scaling_max_freq"; echo "$1" > "$f/scaling_min_freq"; done
                  else "$HELPER" pin "$1" >/dev/null 2>&1 || true; fi; }

# Root OR the helper is enough. Only the two things the helper cannot do still demand real root.
need_priv() {
    is_root && return 0
    have_helper && return 0
    echo "!! needs root (or the setuid helper at $HELPER — see bench/tools/README.md): sudo $0 $*"
    exit 1
}
need_root() { is_root || { echo "!! needs root: sudo $0 $*"; exit 1; }; }

# Real achieved MHz of $1 under load — perf counts actual CPU cycles over ~1 s (immune to the sysfs lies).
achieved_mhz() {
    local core="$1" c
    if command -v perf >/dev/null 2>&1; then
        # DIVIDE BY THE WINDOW perf ACTUALLY MEASURED, not by an assumed 1 s. `timeout 1` does not
        # guarantee a 1-second counting window — perf reports the enabled time in field 4 (ns), and on
        # 2026-09-13 neuromancer returned 675463463 cycles over 341375879 ns. That is 1.98 GHz, exactly
        # its 2000 MHz pin; dividing by 1e6 reported it as "675 MHz" and the box looked catastrophically
        # throttled. The error is proportional (0.341 s window => 0.341x the true figure), so it always
        # UNDER-reports, i.e. it condemns a healthy box rather than passing a broken one — but it cost a
        # killed sweep and a long detour through platform_profile, RAPL and thermals before the CSV was
        # read carefully. galen and wintermute were unaffected only because their windows landed near 1 s.
        local out ns
        out=$(perf stat -x, -e cycles -- taskset -c "$core" timeout 1 bash -c 'while :; do :; done' 2>&1 \
            | awk -F, 'tolower($0) ~ /cycles/ {print; exit}')
        c=$(printf '%s' "$out" | awk -F, '{gsub(/ /,"",$1); print $1}')
        ns=$(printf '%s' "$out" | awk -F, '{gsub(/ /,"",$4); print $4}')
        if [[ "$c" =~ ^[0-9]+$ ]] && [[ "$ns" =~ ^[0-9]+$ ]] && [ "$ns" -gt 0 ]; then
            echo $(( c * 1000 / ns )); return           # cycles / ns * 1000 = MHz
        fi
        [[ "$c" =~ ^[0-9]+$ ]] && { echo $(( c / 1000000 )); return; }   # pre-5.x perf: no field 4
    fi
    taskset -c "$core" timeout 2 bash -c 'while :; do :; done' & local pid=$! s=0 n=0
    sleep 0.4
    for _ in 1 2 3; do s=$(( s + $(cat "/sys/devices/system/cpu/cpu$core/cpufreq/scaling_cur_freq") )); n=$((n+1)); sleep 0.35; done
    wait "$pid" 2>/dev/null || true
    echo $(( s / n / 1000 ))
}

# power-profiles-daemon (and tuned) actively manage scaling_max_freq. In `performance` mode PPD drives it
# back up to the boost range, and on an INTERACTIVE box (open desktop sessions) it re-asserts on every
# AC/session/activity event — silently REVERTING the pin mid-sweep (the recurring neuromancer flip-flop;
# headless galen/wintermute never trigger the re-apply so their pin held). Stop them for the locked run;
# `restore` brings them back. ponytail: `stop` not `mask` — if socket-activation resurrects PPD mid-sweep,
# escalate to `mask` here (verify_or_die will catch a reverted pin regardless).
quiesce_ppd() {
    command -v systemctl >/dev/null 2>&1 || return 0
    for svc in power-profiles-daemon tuned; do
        if systemctl is-active --quiet "$svc" 2>/dev/null; then
            if systemctl stop "$svc" 2>/dev/null; then
                echo "  stopped $svc (was re-asserting scaling_max_freq)"
            else
                # The setuid helper deliberately cannot do this — it never execs anything. So on the
                # helper-only path a live PPD/tuned can still revert the pin AFTER we verify. Say so:
                # a silent failure here is how a run that verified once drifts later.
                echo "  ⚠ could NOT stop $svc (needs real root; the setuid helper does not exec)."
                echo "    It may re-assert scaling_max_freq and undo this pin mid-sweep."
                echo "    Re-run 'sudo $0 lock' for a fully quiesced box, or re-verify between groups."
            fi
        fi
    done
}

# Ensure passive mode. 0=passive now; 1=runtime switch rejected (needs reboot); 3=not an amd_pstate box.
ensure_passive() {
    [ -e "$STATUS" ] || { echo "  (no amd_pstate — plain cpufreq box; boost node should work directly)"; return 3; }
    local m; m=$(cat "$STATUS")
    [ "$m" = passive ] && { echo "  amd_pstate already passive"; return 0; }
    priv_pstate passive
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
#
# UNPIN FIRST, and that is not tidiness — it is the fix for a box that reads locked and is not.
# MEASURED on neuromancer (Zen5 mobile, 2026-09-18), mid-sweep: every readable knob said locked —
# `amd_pstate=passive`, `boost=0`, `min=max=2000000`, and `cpuinfo_max=2000000` — while the core
# achieved 4772 MHz under load. Re-asserting `boost 0` in place: still 4795 MHz. Toggling
# `pstate active`→`passive` in place: no effect either. Only releasing the clamp and re-applying it
# worked: 1975 MHz against the 2000 MHz pin.
#
# The mechanism is that `cpuinfo_max` is DYNAMIC here: it advertises 2000000 kHz with boost off and
# 4900000 with boost on. So in the broken state every node is SELF-CONSISTENT and correct, there is
# nothing for cpufreq to fix, and an in-place re-assert is a no-op by construction. Releasing the pin
# expands the policy range back to 4.9 GHz and the fresh pin then actually takes.
#
# Consequence for anyone reading a lock state: on this fleet, `boost=0` plus `min=max` is NOT evidence
# of a lock. Only `achieved_mhz` under real load is. That is why `verify_or_die` exists and why nothing
# here trusts sysfs.
pin_khz() {
    local khz="$1"
    priv_unpin                                   # release first — an in-place re-pin does NOT stick
    priv_governor performance
    priv_pin_raw "$khz"
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
    need_priv lock
    have_helper && ! is_root && echo "  using the setuid helper at $HELPER (no sudo needed)"
    echo "Locking $(hostname) to its highest VERIFIED-SUSTAINABLE clock (boost off)…"
    quiesce_ppd                                               # stop PPD/tuned so they can't revert the pin
    if ensure_passive; then
        priv_boost 0                                          # passive honors this → cpuinfo_max drops to base
        base=$(( $(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq) / 1000 ))
        pin_khz $(( base * 1000 ))
        got=$(achieved_mhz "$CORE")                           # does the box actually HOLD base under load?
        if [ "$got" -ge $(( base * 90 / 100 )) ]; then
            verify_or_die "$base"                             # yes (desktop, e.g. galen) — locked at base
        else
            # Power/thermally limited (mobile, e.g. the 7640U/AI-5-340 laptops): base does NOT hold under
            # sustained load (measured 3501-pin → 2367 achieved). Re-pin to 90% of the sustained measurement
            # — the highest clock that stays FLAT across a multi-minute sweep (10% margin for thermal creep).
            stable=$(( got * 90 / 100 ))
            echo "  ⚠ base ${base} MHz not sustained (load=${got} MHz) — power-limited box; re-pinning ${stable} MHz…"
            pin_khz $(( stable * 1000 ))
            verify_or_die "$stable"
        fi
    else
        echo "  runtime active→passive REJECTED by kernel — persisting the boot param instead:"
        persist_grub
        echo ""
        echo ">>> REBOOT this box, reconnect the tunnel, then run:  sudo $0 lock"
        echo "    (after the reboot it comes up passive and the lock will stick + verify)"
        exit 3
    fi ;;
  pin)
    need_priv pin
    mhz="${2:?usage: sudo $0 pin <MHz>  (MHz must be ≤ base clock; use 'lock' for the canonical base-clock state)}"
    quiesce_ppd                                               # stop PPD/tuned so they can't revert the pin
    ensure_passive || { echo "  passive needed for a hard pin; persisting boot param:"; persist_grub; echo ">>> reboot, reconnect, re-run."; exit 3; }
    priv_boost 0                                              # a hard pin MUST kill boost, else it floats above the pin
    base=$(( $(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq) / 1000 ))   # boost-off cpuinfo_max = base
    if [ "$mhz" -gt "$base" ]; then
        echo "❌ requested ${mhz} MHz > base ${base} MHz. Boost frequencies do NOT lock (they float above"
        echo "   scaling_max_freq with thermals). There is no stable pin above base — use 'sudo $0 lock'"
        echo "   for the canonical base-clock state. Refusing to ship an unlockable target."
        exit 2
    fi
    pin_khz $(( mhz * 1000 ))
    verify_or_die "$mhz" ;;
  restore)
    need_priv restore
    priv_pstate active
    priv_boost 1
    if command -v systemctl >/dev/null 2>&1; then             # bring PPD/tuned back (undo quiesce_ppd)
        for svc in power-profiles-daemon tuned; do systemctl start "$svc" 2>/dev/null || true; done
    fi
    # `unpin` restores each core to its OWN cpuinfo_min/max, which is what the loop here used to do by
    # hand. Note the ORDER matters: boost is re-enabled first, because `cpuinfo_max` is dynamic (2.0 GHz
    # with boost off, 4.9 GHz with it on — measured on neuromancer), so unpinning while boost is still
    # off would restore the range to base rather than to the full range.
    priv_unpin
    priv_governor powersave
    echo "Restored: amd_pstate=$(cat "$STATUS" 2>/dev/null||echo n/a), boost on, full range." ;;
  verify)
    echo "amd_pstate=$(cat "$STATUS" 2>/dev/null || echo n/a)  boost=$(cat "$BOOST" 2>/dev/null || echo n/a)"
    echo "core$CORE achieved under load = $(achieved_mhz "$CORE") MHz" ;;
  *) echo "usage: sudo $0 {lock|pin <MHz>|restore|verify}"; exit 1 ;;
esac
