#!/usr/bin/env bash
# Refresh every PB arm on ONE box, group by group, reusing the cached OpenBLAS/AOCL arms.
#
# WHY GROUP-BY-GROUP AND NOT ONE FULL RUN: a full `bench arms=pb` with no op=/group= is REFUSED by
# design — a full run REPLACES the cache and would drop the reference arms entirely. Only op=/group=
# runs merge per arm. (publish.sh says this in its refusal message; it is not a workaround.)
#
# The lock is verified BEFORE and AFTER. A gate measurement is only valid under a verified lock, and
# on at least one fleet box the lock has been observed to drop — so "it was locked when I started" is
# not evidence it was locked at the end.
set -u
cd "$(dirname "$0")/.."
JL="${JULIA:-julia}"

# PIN THE SWEEP TO ONE CORE. `adjudicate.sh` and `screen_cells.sh` have always done this; the group
# sweeps did not, so a two-hour refresh ran with `affinity 0-23` and the scheduler was free to migrate
# it mid-measurement — between cores with different cache state, and on a multi-CCD part between
# entirely different L3 instances.
#
# That is not hypothetical. Measured on galen (5900X, TWO 32 MiB L3s: cores 0-5,12-17 and 6-11,18-23)
# on 2026-08-28: a full-arms sweep and a resident GPU job were BOTH unpinned and had landed on cores 2
# and 3 — the same die, sharing one L3. Against an otherwise identical quiet sweep, 3.7% of cells
# flipped their pass/fail verdict, concentrated near 1.0 where the gate decision is made. Pinning the
# sweep does not remove a co-tenant, but it makes WHICH L3 it shares a decision rather than an
# accident, so a contending job can be `taskset` onto the other die.
#
# Default core per box, matching what fleet_freqlock.sh verifies under load (it must be the SAME core,
# or the lock is verified somewhere the work does not run): wintermute 2, galen 6, neuromancer 8.
# Override with BENCH_CORE=<n>.
#
# SWEEP_EXTRA passes extra words to plots.jl. Its one intended use is `force-busy` on a box with a
# RESIDENT co-tenant pinned to the OTHER die, where the global contention guard cannot see that the
# two share no L3 and would refuse forever. galen is the case: llama-server is pinned to CPUs
# 0-5,12-17 (CCD0's L3 domain) while CORE=6 sits in 6-11,18-23 with its own L3, so
# `SWEEP_EXTRA=force-busy bench/fleet_refresh.sh` is honest there — and plots.jl stamps `busy=` into
# the cache header either way, so the run is self-identifying. Do NOT use it to paper over a
# co-tenant on the SAME die; that is the contention the guard exists to catch.
case "${BENCH_CORE:-}" in
    "") case "$(hostname)" in
            wintermute)  CORE=2 ;;
            galen)       CORE=6 ;;
            neuromancer) CORE=8 ;;
            *)           CORE=0 ;;
        esac ;;
    *) CORE="$BENCH_CORE" ;;
esac
# ── MODE: `pb` (default) or `full` ───────────────────────────────────────────────────────────────────
#   bench/fleet_refresh.sh          # arms=pb — reuse cached reference arms (the normal, cheap refresh)
#   bench/fleet_refresh.sh full     # all three arms per cell, in ONE machine state (the repair mode)
#
# WHEN `pb` IS WRONG, AND IT IS NOT RARE. `arms=pb` rewrites only the PB field and keeps the cached
# OpenBLAS/AOCL arms — correct, and the standing rule, PROVIDED the box is in the same machine state it
# was in when those references were measured. The per-cell `anchor` is the field that records whether it
# was. When it is not, every ratio in the file divides two machine states instead of two kernels.
#
# Measured, twice, both self-inflicted:
#   galen  2026-08-27 — a whole-cache `arms=pb` refresh left 616 of 863 cells (71%) anchor-mismatched.
#   zen5   2026-08-27 — the same, 126 cells mismatched, pb arms all at one commit against references
#                       cached FOUR DAYS earlier from two different runs. On a laptop, whose thermal and
#                       load state moves between days, a 4-day-old reference is a bad bet.
# Both were repaired by a FULL-ARMS sweep, which is the documented exception to "never re-measure a
# reference": the rule protects reference arms from pointless churn, not from a state mismatch that has
# already invalidated them. Verify with `bench/check_arm_anchors.sh` after either mode.
# THE MULTI-THREADED ARM. `pb_mt` needs julia started with threads and a CPU mask wide enough to hold
# the runtime threads as well as the workers, so it comes in through JL_FLAGS + BENCH_CORE. The mask is
# one CPU per physical core PLUS one spare, and CPU numbering differs per box — `bench/plots.jl`'s
# `_ARM_PB_MT` comment carries the mask and the measurement behind the spare slot.
#
# ⚠ `-t` IS THE JULIA THREAD COUNT AND IT IS NOT THE POOL SIZE. It must EXCEED the physical core
# count, because `plots.jl` pins one thread per core and the runtime needs an unpinned slot; the pool
# size goes in separately as `mt=`. The guard below refuses `-t <= ncores`. On a 6-core box:
#
#   BENCH_CORE=0,2,4,6,8,10,1 JL_FLAGS="-t 7" SWEEP_EXTRA="mt=6" SWEEP_ARMS="arms=pb,pb_mt" \
#       SWEEP_GROUPS="L1 L2 L3 LP CL1 CL2 CL3 CLP" bench/fleet_refresh.sh
#
# The dual groups have no `pb_mt` arm — their reference is LinearAlgebra's generic fallback, which is
# single-threaded — so they stay out of the group list.
MODE="${1:-pb}"
# SWEEP_ARMS overrides the arms string for this invocation. Its one intended use is the DUAL groups
# (DL1/DL2/DL3/DLP), whose reference is not a vendor BLAS but LinearAlgebra's generic fallback, recorded
# as the arm `generic`. Those ratios are only trustworthy when `generic` is measured in the SAME RUN as
# the pb arm — that is the whole reason the dual groups do not divide by a cached reference — so they
# need `arms=pb,generic` while the real and complex groups correctly stay `arms=pb` and reuse the cached
# OpenBLAS/AOCL arms. One refresh cannot express both, hence two passes:
#
#   SWEEP_GROUPS="L1 L2 L3 LP CL1 CL2 CL3 CLP"  bench/fleet_refresh.sh
#   SWEEP_GROUPS="DL1 DL2 DL3 DLP" SWEEP_ARMS="arms=pb,generic" bench/fleet_refresh.sh
#
# This does NOT open a hole for re-measuring vendor arms: `arms=pb` must still appear in the string (the
# PreToolUse guard keys on it), and dropping it to measure openblas/aocl needs the user's explicit
# per-run authorisation exactly as before.
case "$MODE" in
    pb)   ARMSARG="${SWEEP_ARMS:-arms=pb}" ;;
    # `full` MUST NAME THE ARMS. It used to pass an empty string, because omitting `arms=` once meant
    # "measure every arm" — plots.jl flipped that default to PB-ONLY on 2026-09-12 so that forgetting
    # the flag could not silently re-run the vendors, and this branch was not updated with it. The
    # result was a mode that printed "all three arms per cell" and measured one: on 2026-09-29 a
    # wintermute re-sweep ran four groups that way, writing FRESH PureBLAS arms at a new 3501 MHz pin
    # against vendor arms still cached from 2793 MHz — a 24% error in PureBLAS's own favour, which is
    # the exact defect the re-sweep existed to remove. The banner below now prints the real string, so
    # the claim and the argument cannot drift apart again.
    full) ARMSARG="arms=pb,${SWEEP_REFS:-openblas,aocl}" ;;
    *)    echo "usage: $0 [pb|full]   (pb = reuse cached reference arms; full = re-measure all arms)"; exit 2 ;;
esac

echo "=== pinning sweep to core $CORE ($(hostname)), mode=$MODE ==="
[ "$MODE" = full ] && echo "=== FULL ARMS: passing '$ARMSARG' — every named arm measured per cell in one machine state ==="
# PRE-LOCK MUST PASS, not merely be readable. The per-group check below compares each reading against
# the OPENING one, so it catches a lock that lets go mid-sweep but not a box that was never locked: an
# unlocked box reads a stable boost clock and drifts 0%. neuromancer opened a sweep at 4774 MHz against
# its 2000 MHz pin and the group measured to completion. Only the verify verdict distinguishes the two.
echo "=== PRE-LOCK ==="
_pre=$(bash bench/fleet_freqlock.sh verify 2>&1)
printf '%s\n' "$_pre" | tail -2
if ! printf '%s' "$_pre" | grep -q '✅'; then
    echo "=== ABORT: the box is not locked. Run 'bench/fleet_freqlock.sh lock' first — a gate"
    echo "    measurement taken off the base-clock pin is INVALID, not merely noisy. ==="
    exit 2
fi
# AND, FOR A THREADED SWEEP, THAT THE PIN HOLDS WITH EVERY CORE BUSY. `verify` above loads ONE core
# with an integer loop, which is not the workload a threaded sweep runs and cannot see a PACKAGE power
# limit. wintermute passed it at a 2813 MHz pin while six cores oscillated 2332-2804 MHz on a 60 W
# supply, and every threaded sweep since ran under that ✅ — 1208 of its cached arms are stamped below
# the pin, and 413 cells compare two power states rather than two libraries.
#
# ANY threaded arm makes this a threaded sweep, not `pb_mt` alone: `arms=openblas_mt,aocl_mt` loads
# every core just as hard, so matching `_mt` is what keeps the all-core checks on. This mirrors
# `_ANY_MT` in `bench/plots.jl`, whose comment records that keying on `pb_mt` alone sent a
# reference-only threaded run down the single-threaded path.
if printf '%s' "${JL_FLAGS:-}" | grep -q -- '-t' || printf '%s' "$ARMSARG" | grep -q '_mt'; then
    _MT_RUN=1
    # THE JULIA THREAD COUNT MUST EXCEED THE PHYSICAL CORE COUNT, and this refuses rather than
    # measuring a handicapped pool. `bench/plots.jl`'s `_pin_threads!` pins one thread per physical
    # core and leaves anything beyond that floating, because the runtime needs a slot it can schedule
    # on; with `-t <ncores>` there is nothing left over and the pool's join waits on a pinned thread
    # that cannot move. The pool size is a SEPARATE number — pass it as `mt=<ncores>` through
    # SWEEP_EXTRA. Measured on wintermute (6 cores), gemm n=1000, pb_mt over pb: `-t 6` gave 0.36x,
    # `-t 7` gave 1.38x under the old one-hyperthread pin, and 3.61x once the pin was corrected to
    # cover the whole core. The 2026-10-03 sweep ran `-t 6` and its entire pb_mt arm was unusable.
    # DISTINCT PHYSICAL CORES INSIDE THE SWEEP'S MASK, which is the number `_pin_threads!` will pin:
    # it reads `Cpus_allowed_list` and takes one CPU per `topology/core_id`, so the mask bounds it, not
    # the box. Counting the whole box instead refuses galen's documented recipe — 12 cores, swept at
    # `-t 7 mt=6` on CCD1's six so the three boxes stay comparable.
    #
    # Counting rows of `lscpu -p` is wrong for a second reason: one row per LOGICAL cpu gives the
    # SMT-inflated figure (12 where a box has 6). Both counts here are per physical core.
    _mask_cores() {
        local n=0 seen=" " c id lo hi
        for part in $(printf '%s' "$1" | tr ',' ' '); do
            case "$part" in
                *-*) lo=${part%-*}; hi=${part#*-} ;;
                *)   lo=$part; hi=$part ;;
            esac
            for c in $(seq "$lo" "$hi"); do
                id=$(cat "/sys/devices/system/cpu/cpu$c/topology/core_id" 2>/dev/null || echo "$c")
                case "$seen" in *" $id "*) continue ;; esac
                seen="$seen$id "
                n=$((n + 1))
            done
        done
        echo "$n"
    }
    if [ -n "$CORE" ]; then
        _ncore=$(_mask_cores "$CORE")
    else
        _ncore=$(lscpu -p=Socket,Core 2>/dev/null | grep -v '^#' | sort -u | grep -c . || echo 0)
    fi
    _jlnt=$(printf '%s' "${JL_FLAGS:-}" | sed -n 's/.*-t[ =]*\([0-9]\+\).*/\1/p')
    if [ -n "$_jlnt" ] && [ "$_ncore" -gt 0 ] && [ "$_jlnt" -le "$_ncore" ]; then
        echo "=== ABORT: JL_FLAGS has -t $_jlnt against a mask holding $_ncore physical core(s)."
        echo "    A threaded sweep needs at least -t $((_ncore + 1)) so one thread stays unpinned for"
        echo "    the runtime; set the POOL size separately with SWEEP_EXTRA=\"mt=$_ncore\"."
        echo "    Mask: BENCH_CORE=$CORE. See the note above. ==="
        exit 2
    fi
    # AND THE MASK MUST BE WIDE ENOUGH FOR THE POOL. The default `CORE` is a SINGLE core (the serial
    # bench core), so a threaded sweep launched without BENCH_CORE pins one thread and the other five
    # workers share it — a legal-looking run whose pb_mt arm measures nothing but contention.
    _pool=$(printf '%s' "${SWEEP_EXTRA:-}" | sed -n 's/.*mt=\([0-9]\+\).*/\1/p')
    _pool=${_pool:-6}
    if [ "$_ncore" -gt 0 ] && [ "$_ncore" -lt "$_pool" ]; then
        echo "=== ABORT: the pool is mt=$_pool but BENCH_CORE=$CORE holds only $_ncore physical"
        echo "    core(s). Widen the mask to one CPU per pooled core PLUS one spare for the runtime —"
        echo "    the per-box masks are in bench/plots.jl's _ARM_PB_MT comment. ==="
        exit 2
    fi
    echo "=== PRE-LOCK (all cores) ==="
    _premt=$(NT="${_MT_NT:-6}" bash bench/fleet_freqlock.sh verify-mt 2>&1)
    printf '%s\n' "$_premt" | tail -3
    if ! printf '%s' "$_premt" | grep -q '✅'; then
        echo "=== ABORT: the pin does not hold with every core loaded, so a THREADED sweep here"
        echo "    measures the power limit rather than the library. ==="
        exit 2
    fi
fi

# PRE-WARM THE PACKAGE BEFORE THE FIRST GROUP, ONCE, OUTSIDE THE RETRY LOOP.
#
# A changed `src/` (or anything that invalidates the bench env's cache) makes the first group's julia
# precompile PureBLAS — minutes of multi-core work, orphaned at PPID 1, pegging a core. The
# contention guard inside plots.jl then correctly REFUSES to benchmark against it, and the group burns
# retries waiting for the sweep's own precompile to get out of its way. Observed twice: six retries at
# 90 s is just enough headroom, which means it is one slow box away from aborting a sweep for no
# reason at all.
#
# Doing it here is free when the cache is warm (a few seconds of load) and turns the cold case into
# one silent wait instead of a retry storm. It is deliberately NOT inside the group loop: the cost
# must be paid once, and a second invocation proves nothing.
# `Pkg.precompile()` AND NOT `using PureBLAS`. The first version warmed PureBLAS only, and the first
# group still spawned its own precompile at 99.9% CPU: `bench/plots.jl` loads the whole bench env —
# AOCL_jll, Chairmarks, ForwardDiff and the rest — and any one of those being stale is enough. The
# env is what has to be warm, not one package in it.
echo "=== PRE-WARM (precompile the bench env before the first timed group) ==="
"$JL" --project=bench -e 'using Pkg; Pkg.precompile()' >/dev/null 2>&1 || {
    echo "=== ABORT: the bench env does not precompile — fix that before sweeping. ==="
    exit 2
}
echo "    bench env is warm"

# A GROUP THAT DOES NOT LAND MUST BE LOUD. This loop used to pipe each run through `tail -4` and move
# on, so a group that died took its exit status with it (the pipeline reports tail's status, not
# julia's) and the refresh still printed "REFRESH DONE". Measured 2026-08-29: wintermute's L1 group
# crashed and neuromancer's CLP never ran, and BOTH boxes reported success — 112 cells silently stayed
# at the previous commit and were published. `bench/cache_staleness.sh` caught it only afterwards.
#
# Two failure modes, deliberately handled differently:
#   contention — plots.jl refuses when a foreign process is >= 25% CPU, and the agent driving this
#                trips that itself at startup. Transient, so RETRY rather than lose the group.
#   anything else — a crash or a real error. Report it and keep going so one bad group does not cost
#                the other seven, but remember it and exit non-zero at the end.
FAILED=""
# VERIFY THE LOCK BETWEEN GROUPS, not only at the two ends. neuromancer drops its cpufreq pin when
# something on its power cluster is disconnected (user, 2026-09-06) and the SETTINGS still read locked
# while the cores run at 4.8 GHz against a 2.0 GHz pin — so nothing short of an achieved-under-load
# measurement sees it. With only PRE/POST checks a mid-run drop costs the WHOLE sweep: on 2026-09-06 it
# went PRE 1978 MHz / POST 4693 MHz, `plots.jl` refused the last two groups, and the two groups already
# measured had to be discarded as well because nothing says when in the run the clock let go. Checking
# per group bounds that loss to one group and names it. Costs a few seconds against a ~40-minute group.
# `check_arm_clocks.sh` cannot substitute: full-arms measures every arm of a cell together, so a drop
# BETWEEN cells leaves each cell internally consistent and the check passes.
_lock_mhz() { bash bench/fleet_freqlock.sh verify 2>&1 | grep -oE 'achieved under load = [0-9]+' | grep -oE '[0-9]+$'; }
LOCK0=$(_lock_mhz)
[ -n "$LOCK0" ] || { echo "=== ABORT: cannot read the achieved frequency — refusing to measure ==="; exit 2; }
for g in ${SWEEP_GROUPS:-L1 L2 L3 LP CL1 CL2 CL3 CLP DL1 DL2 DL3 DLP}; do
    now=$(_lock_mhz)
    # 3% of the opening figure, the same tolerance check_arm_clocks.sh uses between arms of one cell.
    if [ -z "$now" ] || [ "$(( (now - LOCK0) * 100 / LOCK0 ))" -gt 3 ] || [ "$(( (LOCK0 - now) * 100 / LOCK0 ))" -gt 3 ]; then
        echo "=== ABORT before group $g: lock moved ${LOCK0} -> ${now} MHz. Every group measured after"
        echo "    the drop is INVALID, and which ones those are is unknowable — re-lock and re-run."
        FAILED="$FAILED $g(lock)"
        break
    fi
    # AND THE ALL-CORE CHECK BETWEEN GROUPS, for a threaded sweep, because the check above cannot see
    # the other direction. A dropped pin makes the cores run FASTER than the setpoint, which one core
    # under load reveals; a package power limit makes them run SLOWER only when every core is busy, and
    # one core never reproduces it. Both end a sweep's validity, so both are checked per group, and a
    # failure stops the run here rather than at the end — which bounds the loss to the groups already
    # written instead of discarding the whole sweep.
    #
    # Runs BEFORE the cooldown below, so the heat this check puts into the package is what the cooldown
    # then removes. After it, every group would start hotter than the references were measured in.
    if [ "${_MT_RUN:-0}" = 1 ]; then
        _gmt=$(NT="${_MT_NT:-6}" bash bench/fleet_freqlock.sh verify-mt 2>&1)
        if ! printf '%s' "$_gmt" | grep -q '✅'; then
            echo "=== ABORT before group $g: the pin no longer holds with every core loaded."
            printf '%s\n' "$_gmt" | tail -2
            echo "    Groups already written are kept; this one and the rest are not measured."
            FAILED="$FAILED $g(all-core)"
            break
        fi
    fi
    # LET THE BOX COOL BETWEEN GROUPS, or the anchors will not match the cached references.
    #
    # An `arms=pb` group runs ~3x faster than the full-arms group that produced the cached OpenBLAS
    # and AOCL arms, so back-to-back pb groups leave the box in a HOTTER, denser-duty state than the
    # references were measured in. The per-cell anchor records exactly that, and `check_arm_anchors.sh`
    # then declares the cells not adjudicable — the sweep is wasted even though the clock never moved.
    #
    # Measured 2026-09-10 on neuromancer: a back-to-back pb refresh left 697/930 cells
    # anchor-mismatched (worst 15.7%) with the lock verified at 1990 MHz throughout. Re-running the
    # single group L3 after a 300 s idle gap, nothing else changed, dropped it to 601/930 and removed
    # L3 from the affected list entirely. So the mismatch is thermal carry-over between groups, NOT an
    # irreconcilable machine-state difference — which matters, because the documented repair for the
    # latter is a FULL-ARMS sweep that re-measures the reference arms this project deliberately caches.
    # A few minutes per group is far cheaper than that, and cheaper still than a discarded sweep.
    #
    # Skipped before the first group: `PRE-LOCK` has already just idled the box.
    if [ -n "${_PB_NOTFIRST:-}" ]; then
        echo "    (idle ${GROUP_GAP:-300}s so the box returns to the references' thermal state)"
        sleep "${GROUP_GAP:-300}"
    fi
    _PB_NOTFIRST=1
    echo "=== group $g ==="
    ok=0
    for try in 1 2 3 4 5 6; do
        # Capture, THEN tail — piping julia straight into `tail` reports tail's exit status, which is
        # how the silent partial refresh happened in the first place.
        # shellcheck disable=SC2086  # ARMSARG is deliberately unquoted: empty must expand to NO argument
        # shellcheck disable=SC2086  # SWEEP_EXTRA is deliberately unquoted: empty must expand to NO argument
        # shellcheck disable=SC2086  # JL_FLAGS is deliberately unquoted: empty must expand to NO argument
        out=$(taskset -c "$CORE" "$JL" ${JL_FLAGS:-} --project=bench bench/plots.jl bench group=$g $ARMSARG ${SWEEP_EXTRA:-} nodraw 2>&1)
        st=$?
        printf '%s\n' "$out" | tail -4
        if [ $st -eq 0 ]; then ok=1; break; fi
        if printf '%s' "$out" | grep -q "REFUSING to benchmark"; then
            echo "    group $g: box contended, retry $try in 90s"; sleep 90; continue
        fi
        echo "    group $g: FAILED (exit $st, not contention) — see output above"; break
    done
    [ $ok -eq 1 ] || FAILED="$FAILED $g"
done
echo "=== POST-LOCK ==="; bash bench/fleet_freqlock.sh verify 2>&1 | tail -2
# A THREADED SWEEP NEEDS THE ALL-CORE CHECK AT BOTH ENDS. `verify` loads one core, so it cannot see a
# package power limit, and a sweep that drifted into one for its last hours closes with a ✅ that
# certifies nothing. The in-cell `mtanchor` field does not cover this either: it scales the pin by the
# all-core anchor's slowdown against the FIRST threaded window's reference, so a box throttled for the
# whole run throttles the reference too and the ratio reads 1.0 — `bench/plots.jl` states that
# limitation at `_mt_effective_khz`. A ❌ here does not delete the cache, because which cells it
# damaged is unknown; it says the sweep is not publishable until the box is re-checked.
if [ "${_MT_RUN:-0}" = 1 ]; then
    echo "=== POST-LOCK (all cores) ==="
    _postmt=$(NT="${_MT_NT:-6}" bash bench/fleet_freqlock.sh verify-mt 2>&1)
    printf '%s\n' "$_postmt" | tail -3
    printf '%s' "$_postmt" | grep -q '✅' || {
        echo "=== WARNING: the pin held at the START of this sweep and does NOT hold now. Every"
        echo "    threaded arm written above is suspect — the box may have been measuring its power"
        echo "    limit for part of the run. Do NOT publish; re-check the box and re-sweep. ==="
        FAILED="$FAILED post-lock-all-cores"
    }
fi
if [ -n "$FAILED" ]; then
    echo "=== REFRESH INCOMPLETE — these groups did NOT land:$FAILED"
    echo "    Their cells still carry the PREVIOUS commit. Re-run them before publishing:"
    for g in $FAILED; do echo "      taskset -c $CORE $JL ${JL_FLAGS:-} --project=bench bench/plots.jl bench group=$g $ARMSARG nodraw"; done
    echo "    Then confirm with: bench/cache_staleness.sh"
    exit 1
fi
echo "=== REFRESH DONE ==="
