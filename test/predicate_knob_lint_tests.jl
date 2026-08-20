# Every µarch-PREDICATE-keyed tuning knob must carry a justification in docs/src/tuning.md §4b.
#
# WHY. `CLAUDE.md` 8b allows `_double_pumped(hw) ? a : b` ONLY when the knob's physical criterion IS
# the fact the predicate encodes, argued in the comment with a fleet table — otherwise it is a lazy
# two-way lookup and a violation. Nothing enforced that, and the failure is not hypothetical: the
# justification on `_at_gemvt_perscan` asserted "no mechanism could be found", was replaced on
# 2026-08-20 by a `_wide_simd` mechanism that measurement FALSIFIED before it shipped, and in both
# states only a human reading the comment would have noticed.
#
# The deeper reason a register is needed rather than a per-site comment: a binary predicate over a
# THREE-box fleet is nearly unconstrained — three points admit only three non-trivial partitions, and
# we use two of them. "It reproduces the fleet" is therefore about one bit of evidence. The register
# forces each knob to state a MECHANISM and a FALSIFIER, which is what actually distinguishes a
# derivation from a curve fit, and it makes the weak ones countable instead of scattered.
#
# Scan + testitem live in ONE file deliberately: this lint reads two static files and needs no
# standalone CLI entry point (unlike probe_regime_lint.jl, whose scanner is reused by tooling).
@testitem "predicate-knob lint: every µarch-keyed knob is justified in tuning.md" begin
    cpu = read(joinpath(@__DIR__, "..", "src", "cpuinfo.jl"), String)
    doc = read(joinpath(@__DIR__, "..", "docs", "src", "tuning.md"), String)

    # A knob is "predicate-keyed" if a BOOLEAN µarch test appears inside its definition.
    # `_datapath_bytes` counts only in a COMPARISON — used as a quantity it is a real formula
    # (`_at_cpotf2_mr = max(1, 64 ÷ _datapath_bytes(hw))` derives, it does not look up).
    ispred(ln) = occursin("_double_pumped(", ln) || occursin("_wide_simd(", ln) ||
                 !isnothing(match(r"_datapath_bytes\([^)]*\)\s*[<>=]", ln))

    # WRAPPED IN A FUNCTION DELIBERATELY: a bare `for` at testitem top level puts the loop body in
    # soft scope, so `cur = ...` inside it does not update the outer binding and the read throws
    # `UndefVarError: cur not defined in local scope`. Third recurrence of this in one session.
    function scan(src)
        keyed = String[]
        cur = ""
        for ln in split(src, '\n')
            m = match(r"^@inline (_at_\w+)\(", ln)
            !isnothing(m) && (cur = m.captures[1])
            # a non-`_at_` top-level definition ends the current knob's body
            (startswith(ln, "const ") || startswith(ln, "function ")) && (cur = "")
            isempty(cur) || !ispred(ln) || cur in keyed || push!(keyed, cur)
        end
        return keyed
    end
    keyed = scan(cpu)

    @test !isempty(keyed)   # the scanner must actually find something, or it proves nothing
    missing_rows = [k for k in keyed if !occursin(k, doc)]
    isempty(missing_rows) || @error "predicate-keyed knob(s) with no entry in docs/src/tuning.md §4b. \
        Add a row giving the MECHANISM (why the physical criterion IS that predicate) and the \
        FALSIFIER (what measurement would kill it), or grade it Weak and say so — an unjustified \
        two-way µarch lookup is a CLAUDE.md 8b violation." knobs = missing_rows
    @test isempty(missing_rows)
end
