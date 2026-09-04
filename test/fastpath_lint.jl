# Fast-path predicate lint — a bare `isa StridedMatrix`/`isa StridedVector` gate in src/ is a silent
# perf regression waiting for the next container type.
#
# WHY THIS FILE EXISTS. `StridedMatrix`/`StridedVector` are CLOSED unions (Array, SubArray,
# ReshapedArray, ReinterpretArray and combinations). `PtrMatrix`/`PtrVector` (src/ptrmat.jl) are not in
# them and never can be. So every kernel that gates its SIMD path on `x isa StridedVector` silently
# routes a pointer-backed operand to the generic scalar loop: same answer, an order of magnitude slower,
# and NO TEST FAILS. `_strided1` / `_dense1` exist precisely to be the predicate instead — they
# const-fold to the identical check for every Strided argument and add explicit `true` methods for the
# pointer types.
#
# THE COST WAS NOT HYPOTHETICAL, and it recurred four times in three different ways:
#   * the C-ABI `izamax` shim — recorded in level1.jl as "a wire-the-fastest-path miss that no gate row
#     could see", which is what motivated `_dense1(::Ptr)`;
#   * the arena conversion, stage 2 — every scratch vector a converted LAPACK routine hands to
#     `gemv!`/`trsv!`/`trmv!`, fixed by `_dense1(::PtrVector)`;
#   * `ormhr!`/`unmhr!`, stage 3 — an arena-borrowed `tmp` as `gemm!`'s C fell to `_gemm_generic!`,
#     an O(n³) scalar triple loop, on a LBT-forwarded path. gbtrf.jl and bunchkaufman.jl had each hit
#     this before and worked around it LOCALLY by calling `_gemm_core!` directly, so the trap was
#     documented twice and still caught a third caller. The gate is now `_strided1(C)`;
#   * sytrf!/hetrf!, stage 3 — `view(W, k:n, k)` on a `PtrMatrix` missed every `view` overload and
#     produced a `SubArray{…,PtrMatrix,…}`, which is not Strided either. Fixed with the partial-column
#     `view` method in ptrmat.jl.
#
# Four incidents, all found by reading, none by a test. Hence a lint.
#
# WHAT IS BASELINED. Not every occurrence is a defect: a `::StridedMatrix{T}` METHOD SIGNATURE is
# dispatch, not a fast-path gate, and some gates guard something a pointer type genuinely cannot do.
# Those live in `test/fastpath_lint_baseline.txt`, each with a written reason, exactly as
# `req8_lint_baseline.txt` carries reasoned literals. A line not in the baseline is a new finding; a
# baseline line that no longer occurs is stale and also fails, so a fixed site cannot leave a rubber
# stamp behind.

const _FP_SRC = normpath(joinpath(@__DIR__, "..", "src"))
const _FP_BASELINE = joinpath(@__DIR__, "fastpath_lint_baseline.txt")
# Only the VALUE-position test is a fast-path gate. `::StridedMatrix` in a signature is dispatch.
const _FP_PAT = r"isa\s+Strided(Matrix|Vector|Array)"
# ptrmat.jl is where the correct predicates are DEFINED, in terms of exactly this test.
const _FP_SKIP = Set(["ptrmat.jl"])

_fp_srcfiles() = sort!([joinpath(r, f) for (r, _, fs) in walkdir(_FP_SRC) for f in fs if endswith(f, ".jl")])

"""
    fastpath_scan() -> Vector{String}

Every `isa Strided*` value test in `src/`, as `relative/path.jl:<code>` keys. Line numbers are
deliberately NOT part of the key — an unrelated edit above a site must not invalidate its baseline entry.
"""
function fastpath_scan()
    hits = String[]
    for f in _fp_srcfiles()
        basename(f) in _FP_SKIP && continue
        rel = relpath(f, _FP_SRC)
        for ln in eachline(f)
            code = first(split(ln, '#'; limit = 2))     # a mention in a comment is documentation
            occursin(_FP_PAT, code) || continue
            push!(hits, string(rel, ":", strip(code)))
        end
    end
    return hits
end

_fp_baseline() = isfile(_FP_BASELINE) ?
    Set(filter(l -> !isempty(l) && !startswith(l, "#"), strip.(readlines(_FP_BASELINE)))) : Set{String}()

"""
    fastpath_violations() -> (new = …, stale = …)

`new`: a gate that is not in the reviewed baseline — replace it with `_strided1`/`_dense1`, or baseline
it with a reason. `stale`: a baselined line that no longer occurs — delete the entry.
"""
function fastpath_violations()
    got = Set(fastpath_scan())
    base = _fp_baseline()
    return (new = sort!(collect(setdiff(got, base))), stale = sort!(collect(setdiff(base, got))))
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--baseline" in ARGS
        foreach(println, fastpath_scan())
    else
        r = fastpath_violations()
        if isempty(r.new) && isempty(r.stale)
            println("fastpath lint: PASS (no unreviewed `isa Strided*` fast-path gates)")
        else
            println("fastpath lint: FAIL")
            foreach(x -> println("  NEW   ", x), r.new)
            foreach(x -> println("  STALE ", x), r.stale)
            exit(1)
        end
    end
end
