# Every `@generated` function whose SIGNATURE mentions `Vec` must emit `Expr(:meta, :inline)` in the
# body it returns.
#
# WHY THIS IS A LINT AND NOT A COMMENT. `@inline` does NOT propagate into @generated CodeInfo on Julia
# 1.12, so a helper annotated `@inline @generated` looks correct and silently passes its Vec arguments
# BY POINTER, stack-demoting the CALLER's accumulators. `_fold2_cmplx` (blas3/gemm.jl) documents the
# cost: zgemvC went 0.26-0.64 -> 1.06-1.37 vs OpenBLAS when the meta was added, and its comment already
# states the rule — "Any @generated helper taking/returning a Vec MUST carry it".
#
# The rule was stated in a comment and enforced by memory, so it drifted: an audit on 2026-08-22 found
# `_tblk!` (gemm.jl:280, the transpose-pack block feeding _pack_A_simd_T! and _transpose_dense!) had
# been missing it. Adding it measured NEUTRAL on trsmR — the point is that nothing would have caught a
# case where it was not neutral.
function generated_meta_scan(root = joinpath(@__DIR__, "..", "src"))
    bad = String[]
    for (r, _, fs) in walkdir(root), fn in fs
        endswith(fn, ".jl") || continue
        p = joinpath(r, fn); L = readlines(p)
        for (i, ln) in pairs(L)
            occursin("@generated function", ln) || continue
            sig = ln; j = i
            while j < length(L) && !occursin(r"\)\s*(where.*)?$", strip(sig)) && j - i <= 8
                j += 1; sig *= " " * strip(L[j])
            end
            occursin("Vec{", sig) || continue          # only Vec-taking/returning helpers are at risk
            k = j; found = false
            while k <= length(L) && k - j <= 200
                occursin("Expr(:meta, :inline)", L[k]) && (found = true; break)
                (startswith(L[k], "end") && k > j) && break
                k += 1
            end
            found && continue
            m = match(r"@generated function ([_A-Za-z0-9!]+)", ln)
            push!(bad, string(relpath(p, root), ":", i, "  ", isnothing(m) ? "?" : m.captures[1]))
        end
    end
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    v = generated_meta_scan()
    isempty(v) ? println("generated-meta lint: PASS") :
        (println("generated-meta lint: FAIL"); foreach(x -> println("  ", x), v))
end
