# POOL-LIVENESS lint — a `p.gen` witness must name the constant that decides whether the pool runs.
#
# WHY A GATE AND NOT A HABIT. `g0 = @atomic p.gen; f(); @test (@atomic p.gen) != g0` is the natural
# way to assert "the split ran", and it is the right assertion. What is not natural is remembering
# that whether the split runs is a function of the MACHINE, so an operand sized by eye passes on the
# box it was written on and fails on another — taking every assertion beside it down from "verified"
# to "vacuous" without changing a line of library code. That has now happened four times, and three
# of them were caught only because one box in the fleet has a different answer.
#
# TWO DIFFERENT CONSTANTS DECIDE IT, and conflating them is the recurring error:
#
#   * `_L1_MT_MIN` = `4 * _L1_BYTES` — the BYTE FLOOR under which `_l1_workers` returns 1 outright.
#     128 KB where L1 is 32 KB, 512 KB where L1 is 128 KB. `_red_block` does NOT imply it: a size
#     derived from the block grid alone buys enough blocks to divide between workers and can still sit
#     a factor of eight under the floor.
#   * `_sme_*_ok` — a route predicate. Where an SME route exists it is consulted BEFORE the pool, so
#     an eligible call declines the pool deliberately and the witness must read `ran(f) == !sme`
#     rather than `ran(f)`.
#
# So this requires a liveness witness to mention one of them, which is the point at which the author
# has to decide WHICH applies. It cannot check that the size is correct — only that the binding
# constant was consulted rather than guessed.
#
# Escape hatch: `# liveness-ok: <reason>` inside the testitem, for a witness whose admission is
# genuinely size-independent. `liveness_baseline.txt` carries witnesses reviewed against a real run on
# the box that would expose them, so a NEW guessed size fails immediately.
#
# Run standalone:  julia test/liveness_lint.jl

const _TESTDIR = @__DIR__
const _LV_BASELINE = joinpath(_TESTDIR, "liveness_baseline.txt")
# The witness itself: an atomic read of the pool generation. Both the inline form and the `ran(f)`
# helper the larger items define reduce to this.
const _LV_WITNESS = r"@atomic\s+\w+\.gen"
# Either of the two constants that decide admission. `_SME_CALLS` counts the same route by its
# side effect and is how the gemm item distinguishes "owned by SME" from "never tried".
const _LV_GATE = r"_L1_MT_MIN|_sme_\w*_ok|_sme_\w*_eligible|_sme_owns|_SME_CALLS"
const _LV_OK = r"#\s*liveness-ok:"i

"""
    liveness_scan() -> Vector{String}

Testitems that assert pool liveness without naming the constant that gates admission.
"""
function liveness_scan()
    base = isfile(_LV_BASELINE) ?
        Set(filter(l -> !isempty(l) && !startswith(l, "#"), strip.(readlines(_LV_BASELINE)))) :
        Set{String}()
    bad = String[]
    for f in sort!(readdir(_TESTDIR))
        endswith(f, "_tests.jl") || continue
        src = read(joinpath(_TESTDIR, f), String)
        occursin(_LV_WITNESS, src) || continue
        # One block per `@testitem`, so a gate named in a NEIGHBOURING item does not vouch for this
        # one — that is exactly the mistake the lint exists to catch.
        for blk in split(src, "@testitem ")[2:end]
            occursin(_LV_WITNESS, blk) || continue
            occursin(_LV_OK, blk) && continue
            occursin(_LV_GATE, blk) && continue
            m = match(r"^\s*\"((?:[^\"\\]|\\.)*)\"", blk)
            name = isnothing(m) ? "<unnamed>" : m.captures[1]
            key = "$f::$name"
            key in base && continue
            push!(bad, "$key  asserts `p.gen` moved but names neither `_L1_MT_MIN` nor an " *
                       "`_sme_*` route predicate, so its size is guessed rather than derived")
        end
    end
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    v = liveness_scan()
    if isempty(v)
        println("pool-liveness lint: PASS (every witness names its admission gate)")
    else
        println("pool-liveness lint: FAIL — $(length(v)) witness(es) with a guessed size:")
        foreach(x -> println("  ", x), v)
        exit(1)
    end
end
