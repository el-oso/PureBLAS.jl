# Splice bench/coverage_ops.jl's generated BLAS tables into docs/src/coverage.md, in place.
#
# coverage_ops.jl writes the whole styled block (the `<style>` sheet, the per-level tables and the
# colour key) to stdout, and until now getting it into the page meant a hand copy-paste of ~150 lines
# of HTML between two fences. That is the same "hand-maintained numbers drift" hazard the generators
# were written to remove -- it just moved it from typing digits to pasting blocks, and a paste that
# lands one fence off corrupts the page silently.
#
# The region is delimited by the FIRST "```@raw html" and the LAST "```" of the generated span, which
# is everything from the style sheet to the colour key. Anchored on both ends and asserted, so a doc
# edit that moves either fence fails loudly instead of writing HTML into prose.
#
#   julia --project=bench bench/coverage_ops.jl <cache…> > /tmp/tbl.md
#   julia bench/splice_coverage.jl /tmp/tbl.md
const DOC = "docs/src/coverage.md"
gen = readlines(ARGS[1])
doc = readlines(DOC)
while !isempty(gen) && isempty(strip(gen[end]))   # the generator ends with a blank line after the fence
    pop!(gen)
end

first(gen) == "```@raw html" || error("generated block does not open with a raw-html fence")
last(gen) == "```" || error("generated block does not close with a fence")

lo = findfirst(==("```@raw html"), doc)
isnothing(lo) && error("$DOC has no raw-html fence to replace")
# The generated span ends at the LAST fence that closes a raw-html block before the prose resumes:
# take the closing fence of the final `@raw html` block in the file's table region.
opens = findall(==("```@raw html"), doc)
hi = findnext(==("```"), doc, last(opens) + 1)
isnothing(hi) && error("unterminated raw-html block in $DOC")

splice!(doc, lo:hi, gen)
write(DOC, join(doc, "\n") * "\n")
println("spliced ", length(gen), " generated lines over doc lines ", lo, ":", hi)
