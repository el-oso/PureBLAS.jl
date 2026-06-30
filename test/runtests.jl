# Entry point: ReTestItems discovers and runs every `@testitem` under test/. Items run in
# isolated modules, in parallel where possible, and can be triggered individually:
#   julia --project=test -e 'using ReTestItems, PureBLAS; runtests(PureBLAS; name="...")'
using ReTestItems
using PureBLAS

runtests(PureBLAS)
