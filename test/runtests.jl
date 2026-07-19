# Entry point: ReTestItems discovers and runs every `@testitem` under test/. Items run in
# isolated modules, in parallel where possible, and can be triggered individually:
#   julia --project=test -e 'using ReTestItems, PureBLAS; runtests(PureBLAS; name="...")'
using ReTestItems
using PureBLAS

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
const _HEAVY = ("gesvd", "syev (symmetric", "heev (Hermitian", "getrf (LU", "geqrf (QR", "potrf (Cholesky",
    "GEMM real (blocked", "GEMM blocked", "GEMM complex", "trsm vs OpenBLAS", "trmm vs OpenBLAS",
    "syrk/herk", "syrk/syr2k", "symm/hemm", "trmv/trsv blocked")
function _shard_of(nm::AbstractString)
    for (k, h) in enumerate(_HEAVY)
        occursin(h, nm) && return mod(k, _NSHARDS) + 1
    end
    return mod(sum(codeunits(nm)), _NSHARDS) + 1
end
_in_shard(ti) = _NSHARDS <= 1 || _shard_of(ti.name) == _SHARD
_group_filter =
    _GROUP == "checks" ? (ti -> :checks in ti.tags) :
    _GROUP == "main"   ? (ti -> !(:checks in ti.tags) && _in_shard(ti)) :
                         (ti -> true)

# Optional name filter via test args, e.g. `Pkg.test(PureBLAS; test_args=["Reproducibility"])` runs
# only matching `@testitem`s (ANDed with the group filter). No args → the whole selected group.
if isempty(ARGS)
    runtests(_group_filter, PureBLAS)
else
    runtests(_group_filter, PureBLAS; name = Regex(join(ARGS, "|")))
end
