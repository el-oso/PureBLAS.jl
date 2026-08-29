# Did src/ change MEANINGFULLY between a commit and HEAD, or only in comments and formatting?
#
#   julia --startup-file=no bench/src_semantic_diff.jl <commit>
#       exit 0 — every src/ file git calls changed parses to the SAME tree (comments/format only)
#       exit 1 — at least one real code change (or anything we could not read or parse)
#
# WHY THIS EXISTS. `bench/cache_staleness.sh` gates the publish on "did src/ move since this cell was
# measured", using git's FILE-level diff. That cannot tell a kernel edit from a comment fix, so one
# comment-only commit stales every cell on every box and leaves two bad options: a multi-hour fleet
# re-sweep, or `--force` — which also waves through the real staleness the gate exists to catch, and
# that gate was written because ~80% of a published column once rested on pre-change arms.
# Measured 2026-08-29: commit df84ae2 changed two `# PDM:` marker lines and nothing else, and all
# three caches went from 930/930 current to 930/930 STALE — 2790 cells, zero of them affected.
#
# Comments and whitespace are TRIVIA to the parser: they never reach the Expr tree. So parse both
# versions of each changed file and compare. Line numbers DO shift when a comment line is added or
# removed, so they are stripped first.
#
# ⚠ `Base.remove_linenums!` IS NOT ENOUGH and was the first version of this file. It filters only
# inside `:block` and `:quote`, so the `LineNumberNode`s that `Meta.parseall` puts directly in the
# `:toplevel` args SURVIVE — and every top-level line number shifts the moment a comment LINE is added
# or removed, which is the exact edit this file exists to forgive. It passed the end-to-end check
# anyway, because the commit that motivated this (df84ae2) replaced one comment line with one comment
# line and shifted nothing. Caught only by the unit check in test/src_semantic_diff_tests.jl.
#
# ⚠ `Meta.parseall` NEVER THROWS — it EMBEDS the failure as an `:error` or `:incomplete` node in the
# tree it returns. A bare try/catch would therefore accept a truncated tree and could call two
# different files equal, so the nodes are scanned for explicitly. Verified 2026-08-29: "f(x) = )(" →
# `:error`, "f(x) = (1" → `:incomplete`, both returned rather than thrown.
# Out of scope, and harmless here: input that PARSES but fails to LOWER ("f(x) = (1,;)" parses clean)
# is not detected — but it is compared identically in both versions, so it cannot manufacture a false
# "current". That is the only property this needs.
#
# CONSERVATIVE BY CONSTRUCTION, and deliberately asymmetric: a file we cannot fetch, cannot parse, or
# that parses to an error node counts as a REAL change. A false "stale" costs a re-measure; a false
# "current" publishes a number describing code that no longer ships. Only the second one lies.
#
# SCOPE. This answers "is the shipped code identical", not "is the generated code identical". A
# docstring edit therefore still counts as a change — it is a string literal in the tree, and being
# wrong in the safe direction is the point of the paragraph above.

# Drop EVERY line marker, at every depth and in every head — see the warning above for why the stdlib
# helper cannot be used. Rebuilds rather than mutating, so a caller's tree is never edited underneath it.
strip_lines(x) = x
strip_lines(q::QuoteNode) = QuoteNode(strip_lines(q.value))
function strip_lines(e::Expr)
    args = Any[strip_lines(a) for a in e.args
               if !(a isa LineNumberNode) && !(a isa Expr && a.head === :line)]
    return Expr(e.head, args...)
end

# `nothing` means "unreadable or unparseable" — never "empty". Split from `tree` so the comparison is
# testable on strings without a git repo: see test/src_semantic_diff_tests.jl.
function parse_tree(src::AbstractString, path::AbstractString)
    ex = try
        Meta.parseall(src; filename = path)
    catch
        return nothing
    end
    ex isa Expr || return nothing
    for a in ex.args
        a isa Expr && (a.head === :error || a.head === :incomplete) && return nothing
    end
    return strip_lines(ex)
end

function tree(commit::AbstractString, path::AbstractString)
    src = try
        read(`git show $(commit):$(path)`, String)
    catch
        return nothing              # absent at that commit (added or deleted) ⇒ a real change
    end
    return parse_tree(src, path)
end

function semantic_change(commit::AbstractString)
    changed = try
        readlines(`git diff --name-only $(commit)..HEAD -- src/`)
    catch
        return true
    end
    for path in changed
        isempty(strip(path)) && continue
        a = tree(commit, path)
        b = tree("HEAD", path)
        (isnothing(a) || isnothing(b) || a != b) && return true
    end
    return false
end

# Guarded so the file can be `include`d by the test without exiting the process.
if abspath(PROGRAM_FILE) == @__FILE__
    if isempty(ARGS)
        println(stderr, "usage: julia bench/src_semantic_diff.jl <commit>")
        exit(2)
    end
    exit(semantic_change(ARGS[1]) ? 1 : 0)
end
