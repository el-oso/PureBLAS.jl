# Entry point: ReTestItems discovers and runs every `@testitem` under test/. Items run in
# isolated modules, in parallel where possible, and can be triggered individually:
#   julia --project=test -e 'using ReTestItems, PureBLAS; runtests(PureBLAS; name="...")'
using ReTestItems
using PureBLAS

# Optional name filter via test args, e.g. `Pkg.test(PureBLAS; test_args=["Reproducibility"])` runs
# only matching `@testitem`s. No args → the full suite.
if isempty(ARGS)
    runtests(PureBLAS)
else
    runtests(PureBLAS; name = Regex(join(ARGS, "|")))
end
