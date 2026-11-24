# One-based-indexing guard lint — every public `src/` method that takes an array argument must call
# `Base.require_one_based_indexing` on its array arguments before doing anything else, or be a
# one-statement forward to another public array-taking method (which carries the guard).
# (standalone: `julia test/onebased_lint.jl`)
#
# WHY THIS EXISTS. The kernels index from 1 under `@inbounds` and through raw `pointer(x)` offsets, so an
# array with offset axes (an `OffsetArray`) reads and writes outside its buffer: silent heap corruption,
# no error. The guard turns that into an `ArgumentError` at the entry point. A new public method added
# without the call reopens the hole, and nothing at runtime notices.
#
# "Public" here means a name that does not start with an underscore. An array argument is one annotated
# with an `Abstract*`/`Strided*`/`Dense*` array type, `Array`/`Vector`/`Matrix`, or a `where` type variable
# bounded by one. Arguments with no type annotation are not seen.

const _OBLINT_SRC = joinpath(@__DIR__, "..", "src")

const _OBLINT_ARRAY_TYPES = Set(
    [
        :AbstractArray, :AbstractVector, :AbstractMatrix, :AbstractVecOrMat,
        :StridedArray, :StridedVector, :StridedMatrix, :StridedVecOrMat,
        :DenseArray, :DenseVector, :DenseMatrix, :DenseVecOrMat,
        :Array, :Vector, :Matrix, :VecOrMat,
    ]
)

_oblint_files() = sort([joinpath(r, f) for (r, _, fs) in walkdir(_OBLINT_SRC) for f in fs if endswith(f, ".jl")])

# Head symbol of a type expression: `AbstractMatrix{T}` → :AbstractMatrix, `Base.Matrix` → :Matrix.
function _oblint_head(t)
    t isa Symbol && return t
    if t isa Expr
        t.head === :curly && return _oblint_head(t.args[1])
        t.head === :. && return _oblint_head(t.args[end] isa QuoteNode ? t.args[end].value : t.args[end])
        t.head === :where && return _oblint_head(t.args[1])
        t.head === :(<:) && return _oblint_head(t.args[end])
    end
    return nothing
end

# Type variables of a `where` clause that are bounded by an array type.
function _oblint_arraytvars(wheres)
    s = Set{Symbol}()
    for w in wheres
        if w isa Expr && w.head === :(<:) && w.args[1] isa Symbol && _oblint_head(w.args[2]) in _OBLINT_ARRAY_TYPES
            push!(s, w.args[1])
        end
    end
    return s
end

# Names of the arguments of call-expression `call` that are arrays.
function _oblint_arrayargs(call, tvars)
    names = Symbol[]
    for a in call.args[2:end]
        a isa Expr && a.head === :parameters && (a = nothing; continue)
        a isa Expr && a.head === :kw && (a = a.args[1])
        a isa Expr && a.head === :... && (a = a.args[1])
        (a isa Expr && a.head === :(::) && length(a.args) == 2) || continue
        h = _oblint_head(a.args[2])
        (h in _OBLINT_ARRAY_TYPES || h in tvars) || continue
        a.args[1] isa Symbol && push!(names, a.args[1])
    end
    return names
end

_oblint_strip(e) = e isa Expr && e.head === :block ? [x for x in e.args if !(x isa LineNumberNode)] : Any[e]

# Does `body` call `require_one_based_indexing` on `arrs` as a top-level statement, or forward in one
# statement to a public array-taking method (`forwards`)?
function _oblint_guarded(body, forwards::Set{Symbol}, arrs::Vector{Symbol})
    stmts = _oblint_strip(body)
    for s in stmts
        if s isa Expr && s.head === :call && _oblint_head(s.args[1]) === :require_one_based_indexing
            argn = Set(Symbol[a for a in s.args[2:end] if a isa Symbol])
            all(in(argn), arrs) && return true
        end
    end
    if length(stmts) == 1
        s = stmts[1]
        s isa Expr && s.head === :return && (s = s.args[1])
        s isa Expr && s.head === :(::) && (s = s.args[1])
        s isa Expr && s.head === :call && _oblint_head(s.args[1]) in forwards && return true
    end
    return false
end

# Peel `where` and return-type annotations off a signature; returns (call, tvars) or nothing.
function _oblint_sig(sig)
    wheres = Any[]
    while true
        if sig isa Expr && sig.head === :where
            append!(wheres, sig.args[2:end]); sig = sig.args[1]
        elseif sig isa Expr && sig.head === :(::) && length(sig.args) == 2 && sig.args[1] isa Expr
            sig = sig.args[1]
        else
            break
        end
    end
    (sig isa Expr && sig.head === :call) || return nothing
    return sig, _oblint_arraytvars(wheres)
end

_oblint_fname(call) = (n = _oblint_head(call.args[1]); n isa Symbol ? n : nothing)

# Collect (name, line, arrays, body) for every top-level-reachable method definition in `e`.
function _oblint_methods!(out, e, file, line = Ref(0))
    e isa LineNumberNode && (line[] = e.line; return)
    e isa Expr || return
    if e.head === :function || (e.head === :(=) && e.args[1] isa Expr && e.args[1].head in (:call, :where, :(::)))
        sg = _oblint_sig(e.args[1])
        if !isnothing(sg)
            call, tvars = sg
            name = _oblint_fname(call)
            if !isnothing(name)
                push!(out, (; file, line = line[], name, arrs = _oblint_arrayargs(call, tvars), body = e.args[2]))
            end
        end
        return
    end
    foreach(a -> _oblint_methods!(out, a, file, line), e.args)
    return
end

"Returns a vector of human-readable violations; empty means clean."
function onebased_scan(sources = _oblint_sources())
    ms = []
    for (f, text) in sources
        _oblint_methods!(ms, Meta.parseall(text; filename = f), f)
    end
    ispublic(m) = !startswith(String(m.name), "_") && !isempty(m.arrs)
    forwards = Set{Symbol}(m.name for m in ms if ispublic(m))
    v = String[]
    for m in ms
        ispublic(m) || continue
        _oblint_guarded(m.body, forwards, m.arrs) && continue
        push!(
            v, "$(m.file):$(m.line): public `$(m.name)` takes array argument(s) " *
                "$(join(m.arrs, ", ")) but does not call `Base.require_one_based_indexing` on them"
        )
    end
    return v
end

"`file => source text` for every file under src/."
_oblint_sources() = [relpath(f, joinpath(@__DIR__, "..")) => read(f, String) for f in _oblint_files()]

if abspath(PROGRAM_FILE) == @__FILE__
    viol = onebased_scan()
    if isempty(viol)
        println("onebased lint: clean")
    else
        foreach(x -> println("VIOLATION: ", x), viol)
        exit(1)
    end
end
