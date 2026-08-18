#!/usr/bin/env bash
# Full PB-only cache refresh for one box: every group, reusing the cached OpenBLAS/AOCL arms.
#
# WHY GROUP-BY-GROUP AND NOT ONE `bench arms=pb`. A full run with no op=/group= is REFUSED by design —
# it REPLACES the cache and would drop the reference arms. Only op=/group= runs merge per arm. This is
# the exact remedy `bench/cache_staleness.sh` prints.
#
# GUARDS (each has produced a structurally-complete log that measured nothing):
#   * bare `julia` is not on the non-login ssh PATH (juliaup) — asserted before the loop.
#   * a group name that matches nothing merges 0 cells and still prints a plausible table, so the
#     `merged N` line is echoed per group and N==0 is called out loudly.
#   * never run two benchmarks on one box at once — the caller checks; this script does not fork.
set -u
cd ~/Documents/claude/PureBLAS.jl || exit 1
echo "=== $(hostname) full PB refresh ==="
command -v julia >/dev/null 2>&1 || { echo "FATAL: julia not on PATH"; exit 1; }
julia --version || exit 1
git log --oneline -1
bash bench/fleet_freqlock.sh verify

# NEW OPS FIRST, WITH ALL ARMS. `trsvLN`/`trsvLT` are new rows: they have no cached OpenBLAS/AOCL arms,
# so an `arms=pb` run would merge a pb-only cell with no reference to divide by. Measuring a reference
# for a cell that has never had one is not "re-measuring OB/AOCL" — the standing rule is about not
# re-running arms the cache already holds.
for op in trsvLN trsvLT; do
    echo "### new op $op (all arms) ###"
    out=$(julia --project=bench bench/plots.jl bench "op=$op" nodraw 2>&1)
    printf '%s\n' "$(printf '%s\n' "$out" | grep -oE 'merged [0-9]+ re-measured' | head -1)"
    printf '%s\n' "$out" | grep -iE 'ERROR|REFUSING' | head -3
done

for g in L1 L2 L3 LP CL1 CL2 CL3 CLP; do
    echo "### group $g ###"
    out=$(julia --project=bench bench/plots.jl bench "group=$g" arms=pb nodraw 2>&1)
    m=$(printf '%s\n' "$out" | grep -oE 'merged [0-9]+ re-measured' | head -1)
    printf '%s\n' "${m:-NO MERGE LINE}"
    case "$m" in
        "merged 0 re-measured") echo "  !! ZERO CELLS for group $g — name did not match" ;;
    esac
    printf '%s\n' "$out" | grep -iE 'ERROR|StrictViolation|REFUSING' | head -3
done
echo "=== staleness after refresh ==="
bash bench/cache_staleness.sh 2>&1 | tail -6
# POST-RUN LOCK CHECK. The launch-time verify passing is NOT sufficient: on 2026-08-18 neuromancer
# verified at 1978 MHz and then ran the whole sweep boosting at 4.7 GHz, silently inflating every ratio
# ~2.3x. Verify again at the END so a mid-run float is visible here, not three steps later.
echo "=== freq lock AFTER refresh (must still match the references' clock) ==="
bash bench/fleet_freqlock.sh verify
echo "=== pb arm vs reference arm clocks ==="
bash bench/check_arm_clocks.sh 2>&1 | tail -8
echo "=== DONE $(hostname) ==="
