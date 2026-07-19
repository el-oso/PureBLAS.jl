# Entry point: ReTestItems discovers and runs every `@testitem` under test/. Items run in
# isolated modules, in parallel where possible, and can be triggered individually:
#   julia --project=test -e 'using ReTestItems, PureBLAS; runtests(PureBLAS; name="...")'
using ReTestItems
using PureBLAS

# CI parallelizes the suite across two jobs via PUREBLAS_TEST_GROUP (see .github/workflows/CI.yml):
#   "checks" → only the inference-heavy dogfood (StrictMode strict contracts, TrimCheck trim-safety,
#              Aqua quality) — the items tagged `:checks`, which dominate wall-clock via full-inference.
#   "main"   → everything else (the correctness suite).
# UNSET (local `Pkg.test()`) runs the FULL suite — nothing is skipped, so local coverage equals the union
# of the two CI jobs. Splitting is a scheduling optimization only, never a correctness cut.
const _GROUP = get(ENV, "PUREBLAS_TEST_GROUP", "all")
_group_filter =
    _GROUP == "checks" ? (ti -> :checks in ti.tags) :
    _GROUP == "main"   ? (ti -> !(:checks in ti.tags)) :
                         (ti -> true)

# Optional name filter via test args, e.g. `Pkg.test(PureBLAS; test_args=["Reproducibility"])` runs
# only matching `@testitem`s (ANDed with the group filter). No args → the whole selected group.
if isempty(ARGS)
    runtests(_group_filter, PureBLAS)
else
    runtests(_group_filter, PureBLAS; name = Regex(join(ARGS, "|")))
end
