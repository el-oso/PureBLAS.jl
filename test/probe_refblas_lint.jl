# A probe must NEVER link a reference BLAS. References are CACHE-ONLY for benchmarks: the v3 cache holds
# every OpenBLAS/AOCL arm, `arms=pb` reuses them, and re-timing one both wastes the box and risks
# publishing a number whose two sides never saw the same machine state.
#
# Only bench/plots.jl may forward LBT to a reference — that is its job, and it stamps provenance.
# Diagnostic exceptions are possible but must be EXPLICIT and justified in the file, via a
# `# refblas-ok: <reason>` marker, so they are visible in review instead of accumulating silently.
# (Run standalone: `julia test/probe_refblas_lint.jl`)
const _PROBE_DIR = joinpath(@__DIR__, "..", "bench", "probes")
const _PAT = r"AOCL_jll|aocl_blas_ilp64|aocl_lapack_ilp64|OpenBLAS_jll|libopenblas"
const _BASELINE = Set(filter(l -> !isempty(l) && !startswith(l, "#"),
    strip.(readlines(joinpath(@__DIR__, "probe_refblas_baseline.txt")))))

function probe_refblas_scan()
    isdir(_PROBE_DIR) || return String[]
    bad = String[]
    for f in sort(readdir(_PROBE_DIR))
        endswith(f, ".jl") || continue
        s = read(joinpath(_PROBE_DIR, f), String)
        rel = "bench/probes/" * f
        rel in _BASELINE && continue                 # grandfathered legacy debt
        occursin(_PAT, s) && !occursin("# refblas-ok:", s) && push!(bad, rel)
    end
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    v = probe_refblas_scan()
    if isempty(v)
        println("ref-BLAS probe lint: PASS (no probe links OpenBLAS/AOCL)")
    else
        println("ref-BLAS probe lint: FAIL — probe(s) linking a reference BLAS:")
        foreach(p -> println("  ", p, "  (cache-only rule; add `# refblas-ok: <reason>` if genuinely diagnostic)"), v)
        exit(1)
    end
end
