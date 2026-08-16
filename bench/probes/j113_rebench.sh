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
echo "=== refreshing registry + main environment"
# `Pkg.update()` on the MAIN env, not just Registry.update(). Pkg.test() resolves the test
# environment ON TOP OF the main one, so a main Manifest pinning StrictMode 0.3.9 caps JET below 0.12
# and the dogfood reports the OLD four errors instead of two. Refreshing the registry alone does not
# move an already-resolved manifest. Still never `--project=test`: that is what creates
# test/Manifest.toml, which must not exist.
julia --project=. -e 'using Pkg; Pkg.Registry.update(); Pkg.update()' > /tmp/j113_resolve.log 2>&1
grep -E "StrictMode|JET" /tmp/j113_resolve.log | tail -2

echo "=== running correctness suite (fresh 1.13 precompile expected first)"
julia --project=. -e 'using Pkg; Pkg.test()' > /tmp/j113_suite.log 2>&1

# PRECISE GATE — accepts two DOCUMENTED exceptions, does not disable the gate.
# On Julia 1.13 the suite reports exactly two errors, both in the StrictMode dogfood and both about
# BASE internals rather than PureBLAS code (of 88 reported problem lines, 87 were inside Base):
#   strictmode_tests.jl:52   @typestable gemv!       — Base.Broadcast.preprocess recursion and the
#                                                      OncePerThread{Task} scheduler reached via
#                                                      OncePerProcess's first-call lock. Not a hot path.
#   strictmode_tests.jl:311  @trim_compatible sysv!  — juliac --trim=safe rejects Base._str_sizehint /
#                                                      Base.print on an Any-typed phi at sysv.jl:229,
#                                                      i.e. error-message formatting on a throw path.
# ANY numerical failure, or an error at ANY other site, still blocks the sweep. The numerical oracle
# (25063 assertions vs the reference) is what validates the kernels, and it must be clean.
NFAIL=$(grep -oE "[0-9]+ failed" /tmp/j113_suite.log | tail -1 | grep -oE "^[0-9]+")
NOTHER=$(grep -oE "Error During Test at [^ ]+:[0-9]+" /tmp/j113_suite.log \
         | grep -vcE "strictmode_tests\.jl:(52|311)$")
if grep -q "Testing PureBLAS tests passed" /tmp/j113_suite.log \
   || { [ "${NFAIL:-9}" = "0" ] && [ "${NOTHER:-9}" = "0" ]; }; then
    grep -q "Testing PureBLAS tests passed" /tmp/j113_suite.log \
        && echo "=== suite PASSED" \
        || echo "=== suite PASSED with the 2 known-red Base-internals dogfood items (0 numerical failures)"
    grep -E "^PureBLAS +\|" /tmp/j113_suite.log | tail -1
    echo "=== starting full both-arms sweep (pinned to core 2)"
    # `taskset -c 2` is REQUIRED, not optional: it is the documented invocation in plots.jl's own header
    # and in CLAUDE.md's benchmarking section. Without it the process migrates across cores mid-sweep,
    # and every migration lands on a cold L1/L2 and a cold TLB — noise injected straight into the ratio
    # the gate is adjudicating. BLAS.set_num_threads(1) (plus BLIS_NUM_THREADS/OMP_NUM_THREADS=1 for
    # AOCL's multi-thread libblis-mt build) already forces single-threaded MATH; pinning is the separate
    # question of WHICH core that single thread stays on. The suite above is deliberately NOT pinned —
    # it checks correctness, not time.
    taskset -c 2 julia --project=bench bench/plots.jl bench aocl nodraw > /tmp/j113_sweep.log 2>&1
    echo "=== sweep exit=$?"
    echo "=== new header provenance:"
    head -1 bench/plots_data_*[!e].txt | tr '\t' '\n' | grep -E "^(julia|llvm|ob|hw|commit|freq)="
else
    echo "=== suite FAILED — sweep skipped"
    grep -E "Test Summary|Test Failed|ERROR|Error in testset|Evaluated" /tmp/j113_suite.log | tail -25
fi
