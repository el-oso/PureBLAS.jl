# @cfunction order lint — a constant `@cfunction` whose callee is defined later breaks the .so build
# ONLY, and only in a configuration no ordinary session compiles.
#
# WHY THIS FILE EXISTS. `@cfunction(f, ...)` in its CONSTANT form resolves `f` when the enclosing method
# is DEFINED, not when it runs, so the callee must already exist at that point in include order. Get it
# wrong and the module still loads, every test passes, and `juliac --trim` dies with `UndefVarError`.
# Three things stack to make that invisible:
#
#   * The static trampolines live behind `@static if ... elseif _SME_STATIC`, and `_SME_STATIC` is true
#     only when the CPU target names `+sme` — which only the juliac build passes. In a normal session
#     the branch is never compiled, so there is no code for StrictMode's `@assert_trim_safe` or
#     `explain_trim` to inspect. They answer "is this function's compiled output trim-safe"; they cannot
#     answer "does a differently-configured build of this module load at all".
#   * The authoritative check, `test/juliac_build_test.jl`, is gated on `PUREBLAS_JULIAC_BUILD=1` and
#     SKIPS otherwise, so a normal run reports green.
#   * Nothing else builds the .so, and trim-compatibility is requirement 4.
#
# It has already shipped once: `_sme_dot_cf`/`_sme_asum_cf`/`_sme_axpy_cf`/`_sme_scal_cf` sat in
# `blas3/sme_kernel.jl` naming `_sme_*_cabi` functions that `blas1/sme_l1.jl` defines one include LATER,
# and `juliac --trim` could not build. `sme_kernel.jl` even carries the rule in a comment — "the CONSTANT
# form resolves its callee when the enclosing method is DEFINED" — which is how the Level-2/3 trampolines
# came to be placed correctly while the BLAS-1 four were not.
#
# This check is lexical and runs in milliseconds, which is the point: the defect it catches otherwise
# needs a multi-minute `--trim` build that is switched off by default.
#
# WHAT IT DOES NOT COVER. The dynamic form `@cfunction($f, ...)` resolves at call time and is exempt by
# construction — it is the shape the non-static branch uses deliberately. A callee defined in another
# module, or reached through a `Ref`, is outside a lexical scan.

const _CO_SRC = normpath(joinpath(@__DIR__, "..", "src"))
const _CO_ROOT = joinpath(_CO_SRC, "PureBLAS.jl")
# Constant form only: `@cfunction(name, ...)` with a bare identifier. `@cfunction($f, ...)` is the
# dynamic form and resolves at call time.
const _CO_REF = r"@cfunction\(\s*([A-Za-z_][A-Za-z0-9_!]*)\s*,"
# A top-level definition of `name`: `function name(`, `name(args) = `, `const name =`, with any number
# of leading macro invocations — `@inline`, `@ccallable`, `Base.@ccallable`, `@noinline` — since the
# `@ccallable` entry points carry one and a missed prefix reads as "no definition found".
_co_defpat(name) = Regex("^\\s*(?:(?:[A-Za-z_][A-Za-z0-9_.]*\\.)?@[A-Za-z_][A-Za-z0-9_!]*\\s+)*" *
                         "(?:function\\s+|const\\s+)?" *
                         "\\Q" * name * "\\E\\s*(?:\\(|=[^=])")

"""
    cfunction_order_files() -> Vector{String}

`src/` files in INCLUDE ORDER, expanding `include("...")` depth-first from `src/PureBLAS.jl`. Order is
the whole point here: a callee's position is only meaningful relative to where it is referenced.
"""
function cfunction_order_files(path::AbstractString = _CO_ROOT, seen = Set{String}())
    out = String[]
    path in seen && return out
    push!(seen, path)
    push!(out, path)
    isfile(path) || return out
    dir = dirname(path)
    for ln in eachline(path)
        code = first(split(ln, '#'; limit = 2))
        m = match(r"^\s*include\(\"([^\"]+)\"\)", code)
        isnothing(m) && continue
        append!(out, cfunction_order_files(joinpath(dir, m[1]), seen))
    end
    return out
end

"""
    cfunction_order_violations() -> Vector{String}

Every constant `@cfunction` whose callee is defined at or after the reference in include order. Each
entry names the referencing site and where the callee is defined, because the fix is always to move one
of the two.
"""
function cfunction_order_violations()
    files = cfunction_order_files()
    # Flatten to (file index, line number) so "defined later" is one comparison.
    defs = Dict{String, Tuple{Int, Int}}()
    refs = Tuple{String, Int, Int, String}[]      # name, file index, line, display location
    for (fi, f) in enumerate(files)
        isfile(f) || continue
        rel = relpath(f, _CO_SRC)
        for (lno, ln) in enumerate(eachline(f))
            code = first(split(ln, '#'; limit = 2))
            m = match(_CO_REF, code)
            isnothing(m) && continue
            push!(refs, (String(m[1]), fi, lno, string(rel, ":", lno)))
        end
    end
    isempty(refs) && return String[]
    wanted = Set(r[1] for r in refs)
    for (fi, f) in enumerate(files)
        isfile(f) || continue
        rel = relpath(f, _CO_SRC)
        for (lno, ln) in enumerate(eachline(f))
            code = first(split(ln, '#'; limit = 2))
            for name in wanted
                haskey(defs, name) && continue
                occursin(_co_defpat(name), code) || continue
                defs[name] = (fi, lno)
            end
        end
    end
    bad = String[]
    for (name, fi, lno, where) in refs
        d = get(defs, name, nothing)
        if isnothing(d)
            push!(bad, "$where: @cfunction($name, …) — no definition of `$name` found in src/")
        elseif d[1] > fi || (d[1] == fi && d[2] > lno)
            push!(bad, "$where: @cfunction($name, …) resolves its callee at DEFINITION time, but " *
                       "`$name` is defined later, at $(relpath(files[d[1]], _CO_SRC)):$(d[2])")
        end
    end
    return sort!(bad)
end
