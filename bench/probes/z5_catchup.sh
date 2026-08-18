#!/usr/bin/env bash
# Zen5 catch-up: full correctness suite, then — ONLY if it passed — the full both-arms sweep.
# The gate on the suite is deliberate: every Zen5 number predates seven shipped changes, so a
# benchmark run before correctness is confirmed would be measuring code of unknown validity.
# Both arms (no `arms=pb`) because Zen5's cached reference arms are stale at 2113738 (2026-08-09)
# and its whole complex-L3 block predates the 3M width-gate deletion.
set -u
cd "$HOME/Documents/claude/PureBLAS.jl" || exit 1

echo "=== HEAD: $(git rev-parse --short HEAD)"
bash bench/fleet_freqlock.sh verify

julia --project=. -e 'using Pkg; Pkg.test()' > /tmp/z5_suite2.log 2>&1
if grep -q "Testing PureBLAS tests passed" /tmp/z5_suite2.log; then
    echo "=== suite PASSED — starting full both-arms sweep"
    julia --project=bench bench/plots.jl bench aocl nodraw > /tmp/z5_sweep.log 2>&1
    echo "=== sweep exit=$?"
else
    echo "=== suite FAILED — sweep skipped (benchmarking unvalidated code is worthless)"
    grep -E "Test Summary|^PureBLAS |Test Failed" /tmp/z5_suite2.log | tail -5
fi
