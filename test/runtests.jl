# Entry point: TestItemRunner discovers and runs every `@testitem` under test/. Items run in isolated
# modules and can be triggered individually by name:
#   julia --project=test -e 'using Pkg; Pkg.test("PureBLAS"; test_args=["Level-1 contiguous"])'
using TestItemRunner
using PureBLAS

# WHY TestItemRunner AND NOT ReTestItems (migrated 2026-08-16, Julia 1.13.0-rc3):
# ReTestItems 1.35.2 cannot run on 1.13 and there is no fixed release. It needs `Test.push_testset` /
# `Test.pop_testset` (both REMOVED in 1.13) and writes `Test.TESTSET_PRINT_ENABLE[]` (now a ScopedValue,
# no setindex!) at six sites on master. That is not shimmable: 1.13 moved testset state from a mutable
# stack to dynamic scoping, keeping only the readers (`get_testset`, `get_testset_depth`), and ReTestItems
# pushes/pops ACROSS function boundaries — which a ScopedValue, needing `with(...) do ... end` lexical
# nesting, cannot express. Upstream's only fix attempt (PR #235) has been open since 2025-12-18 and fails
# its own 1.12 CI; ReTestItems' CI matrix does not include 1.13 at all.
#
# The migration was cheap because both packages consume the SAME `@testitem` macro from TestItems.jl: all
# ~135 items are unchanged. Only the shared-setup blocks differ — `@testsetup module X ... end` became
# `@testmodule X begin ... end` in six files. This is ONE-WAY: ReTestItems has no `@testmodule`.
#
# Known tradeoff, accepted deliberately: TestItemRunner is single-process and sequential — no `nworkers`,
# `worker_init_expr`, per-item timeouts or retries. PureBLAS used none of those (parallelism here is
# CI-job-level sharding, below), so nothing is lost today. Revisit if in-process parallelism is ever wanted.

# CI parallelizes the suite across jobs via PUREBLAS_TEST_GROUP (see .github/workflows/CI.yml):
#   "checks" → only the inference-heavy dogfood (StrictMode strict contracts, TrimCheck trim-safety,
#              Aqua quality) — the items tagged `:checks`, which dominate wall-clock via full-inference.
#   "main"   → the correctness suite, further SHARDED across PUREBLAS_NSHARDS parallel runners: shard
#              PUREBLAS_SHARD (1-based) runs the items whose name lands in its bucket (stable codeunit-sum
#              partition — deterministic and identical across runners on the same items).
# UNSET (local `Pkg.test()`) → the FULL suite (all groups, all shards; NSHARDS defaults to 1). Nothing is
# ever skipped locally, so local coverage equals the union of the CI jobs — splitting is scheduling only.
const _GROUP = get(ENV, "PUREBLAS_TEST_GROUP", "all")
const _NSHARDS = parse(Int, get(ENV, "PUREBLAS_NSHARDS", "1"))
const _SHARD = parse(Int, get(ENV, "PUREBLAS_SHARD", "1"))
# The few expensive items (LAPACK factorizations, SVD/eigen, the OpenBLAS L3 sweeps) dominate wall-clock —
# round-robin them across shards by list position so no single shard collects them all (a pure name hash
# clustered them). Light items fall to a stable codeunit-sum split; their individual cost is negligible.
const _HEAVY = (
    "gesvd", "syev (symmetric", "heev (Hermitian", "getrf (LU", "geqrf (QR", "potrf (Cholesky",
    "GEMM real (blocked", "GEMM blocked", "GEMM complex", "trsm vs OpenBLAS", "trmm vs OpenBLAS",
    "syrk/herk", "syrk/syr2k", "symm/hemm", "trmv/trsv blocked",
)
function _shard_of(nm::AbstractString)
    for (k, h) in enumerate(_HEAVY)
        occursin(h, nm) && return mod(k, _NSHARDS) + 1
    end
    return mod(sum(codeunits(nm)), _NSHARDS) + 1
end
_in_shard(ti) = _NSHARDS <= 1 || _shard_of(ti.name) == _SHARD
_group_filter =
    _GROUP == "checks" ? (ti -> :checks in ti.tags) :
    _GROUP == "main" ? (ti -> !(:checks in ti.tags) && _in_shard(ti)) :
    (ti -> true)

# Optional name filter via test args, e.g. `Pkg.test(PureBLAS; test_args=["Reproducibility"])` runs
# only matching `@testitem`s (ANDed with the group filter). No args → the whole selected group.
# TestItemRunner has no `name=` kwarg (ReTestItems did), so the regex folds into the same filter —
# which is strictly more capable anyway, since it composes with the group/shard predicate by `&&`.
const _NAME_RX = isempty(ARGS) ? nothing : Regex(join(ARGS, "|"))
_filter(ti) = _group_filter(ti) && (isnothing(_NAME_RX) || occursin(_NAME_RX, ti.name))

@run_package_tests filter = _filter
