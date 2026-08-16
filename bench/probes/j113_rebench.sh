#!/usr/bin/env bash
# Julia 1.13.0-rc3 fleet re-bench. The correctness suite GATES the sweep: a toolchain bump changes
# codegen, so the suite is not a formality here — benchmarking unvalidated code is worthless.
#
# BOTH ARMS (no `arms=pb`), deliberately inverting the standing "never re-measure the references" rule.
# That rule assumes the references did not change. Julia BUNDLES OpenBLAS, so upgrading Julia swapped the
# reference arm (0.3.30+0 under rc3); every cached OB number was produced by a different library.
set -u
cd "$HOME/Documents/claude/PureBLAS.jl" || exit 1

echo "=== host: $(hostname)  HEAD: $(git rev-parse --short HEAD)"
julia --version
bash bench/fleet_freqlock.sh verify

# Registry refresh ONLY, from the MAIN project environment. This run needs StrictMode 0.3.10 (which
# dropped the weakdep compat caps pinning JET below 0.12), and a stale registry resolves the OLD
# StrictMode, whereupon the suite dies on ReTestItems — the very failure this run exists to get past.
#
# NEVER `--project=test`. `Pkg.test()` resolves the test deps in a TEMPORARY environment, so it picks
# up the refreshed registry on its own and writes no test/Manifest.toml. Activating test/ directly is
# what creates that file, and it must not exist.
echo "=== refreshing registry (main environment)"
julia --project=. -e 'using Pkg; Pkg.Registry.update()' > /tmp/j113_resolve.log 2>&1

echo "=== running correctness suite (fresh 1.13 precompile expected first)"
julia --project=. -e 'using Pkg; Pkg.test()' > /tmp/j113_suite.log 2>&1
if grep -q "Testing PureBLAS tests passed" /tmp/j113_suite.log; then
    echo "=== suite PASSED"
    grep -E "^PureBLAS +\|" /tmp/j113_suite.log | tail -1
    echo "=== starting full both-arms sweep"
    julia --project=bench bench/plots.jl bench aocl nodraw > /tmp/j113_sweep.log 2>&1
    echo "=== sweep exit=$?"
    echo "=== new header provenance:"
    head -1 bench/plots_data_*[!e].txt | tr '\t' '\n' | grep -E "^(julia|llvm|ob|hw|commit|freq)="
else
    echo "=== suite FAILED — sweep skipped"
    grep -E "Test Summary|Test Failed|ERROR|Error in testset|Evaluated" /tmp/j113_suite.log | tail -25
fi
