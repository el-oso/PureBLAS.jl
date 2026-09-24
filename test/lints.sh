#!/usr/bin/env bash
# Run every CHEAP static check in one go — seconds, not the suite's half hour.
#
# WHY THIS EXISTS. All eleven checkers below already run inside `Pkg.test()`, and every one of them is
# also runnable on its own. But the only way anyone actually reached them was the full suite, so a
# baseline or a generated table could sit stale for a whole session and surface at the end of a 35
# minute run. That happened twice on 2026-09-17: `test/yield_lint_baseline.txt` carried the old
# spelling of two lines the same commit had edited, and `docs/src/knobs.md` went stale the moment a new
# `@load_preference` knob was added. Neither is a hard bug; both are exactly the class a pre-commit
# check is for.
#
# None of these load PureBLAS — they parse `src/` as text — so the whole run is a few seconds.
#
#   bash test/lints.sh            # check everything, exit 1 if anything drifted
#   bash test/lints.sh --fix      # additionally REGENERATE docs/src/knobs.md
#
# `--fix` regenerates only the GENERATED table. It never edits a baseline: a new line in a baseline is
# a claim that a human reviewed it, and a script must not make that claim on anyone's behalf.
set -uo pipefail
cd "$(dirname "$0")/.."

JULIA=${JULIA:-julia}
FIX=0
[[ "${1:-}" == "--fix" ]] && FIX=1

fail=0
run() {                                   # run <name> <script...>
    local name=$1; shift
    local out
    if out=$("$JULIA" --startup-file=no "$@" 2>&1); then
        printf '  ok    %s\n' "$name"
    else
        fail=1
        printf '  FAIL  %s\n' "$name"
        printf '%s\n' "$out" | sed 's/^/          /'
    fi
}

echo "cheap static checks:"
for l in armstate estimator expint fastpath generated_meta perthread pin probe_refblas probe_regime req8 workspace yield; do
    f="test/${l}_lint.jl"
    [[ -f $f ]] && run "$l" "$f"
done

# The registry generator WRITES by design, so checking it means comparing rather than running it.
if [[ $FIX -eq 1 ]]; then
    run "knob registry (regenerated)" --project=. test/knob_registry.jl
else
    run "knob registry" --project=. -e '
        include("test/knob_registry.jl")
        got = knob_markdown()
        want = read(joinpath("docs", "src", "knobs.md"), String)
        got == want && exit(0)
        println("docs/src/knobs.md is STALE — regenerate with: bash test/lints.sh --fix")
        exit(1)'
fi

if [[ $fail -eq 0 ]]; then
    echo "all clean"
else
    echo "drift found — fix it before committing"
fi
exit $fail
