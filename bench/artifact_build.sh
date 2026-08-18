# SOURCED, not executed. The ONE definition of how a published artifact is produced and from which
# caches, in which order — so bench/publish.sh and bench/check_artifacts_current.sh cannot disagree
# about it. A verifier that rebuilds differently from the publisher verifies nothing.
#
# CACHE ORDER IS LOAD-BEARING: bench/coverage_ops.jl and bench/coverage_routing.jl emit one column per
# cache IN ARGUMENT ORDER, and docs/src/coverage.md's headers read Zen3 · AVX2 | Zen4 · AVX-512 |
# Zen5 · AVX-512. Give them in another order and every number lands under the wrong microarchitecture —
# the tables still render, they are just silently wrong.
ARTIFACT_CACHES=(
    bench/plots_data_avx2_galen.txt        # Zen3 · AVX2
    bench/plots_data_avx512_wintermute.txt # Zen4 · AVX-512
    bench/plots_data_zen5_neuromancer.txt  # Zen5 · AVX-512
)

# build_artifacts [outdir]
#   outdir empty → write to the real locations (docs/src/assets/*.svg, bench/gen_table*.md)
#   outdir set   → write the SVGs and gen_tables there instead (the verifier's temp rebuild)
# docs/src/coverage.md is ALWAYS regenerated in place — the generators splice/rewrite that one file and
# have no output redirect. The verifier backs it up and restores it; see that script.
#
# NEVER pass `bench` to plots.jl here: that re-measures for hours and overwrites the caches. Publishing
# is a pure function of the caches on disk.
build_artifacts() {
    local out=${1:-} tbl rc=0
    for c in "${ARTIFACT_CACHES[@]}"; do
        [ -f "$c" ] || { echo "MISSING CACHE $c — the fleet is incomplete; a table built now would publish a partial fleet as if it were the whole one" >&2; return 2; }
    done
    julia --project=bench bench/plots.jl ${out:+outdir=$out} || return 2   # both reference views, one invocation
    tbl=$(mktemp) || return 2
    julia --project=bench bench/coverage_ops.jl "${ARTIFACT_CACHES[@]}" > "$tbl" || rc=2
    [ $rc -eq 0 ] && { julia bench/splice_coverage.jl "$tbl" || rc=2; }
    rm -f "$tbl"
    [ $rc -eq 0 ] || return $rc
    julia --project=bench bench/coverage_routing.jl "${ARTIFACT_CACHES[@]}" || return 2
}
