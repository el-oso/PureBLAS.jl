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
            if !isnothing(m)
                cur = m.captures[1]
            elseif startswith(ln, "const ") || startswith(ln, "function ") || startswith(ln, "@inline ")
                # ANY other top-level definition ends the current knob's body.
                # ⚠ This used to reset only on `const`/`function`, so a later `@inline _wide_simd(hw) =
                # hw.simd >= 64` — a predicate DEFINITION, not a use — was credited to whichever `_at_`
                # was last seen, flagging an innocent knob (`_at_gemvt_percol_xmax`, 2026-08-21). Same
                # mis-attribution class as the knob-registry marker bug fixed the day before: a scanner
                # that keeps state across a definition boundary blames the wrong symbol.
                cur = ""
            end
            isempty(cur) || !ispred(ln) || cur in keyed || push!(keyed, cur)
        end
        return keyed
    end
    # ── FIXTURE: the scanner tested against ground truth, including the case that broke it ──────────
    # A bare `!isempty(result)` proves only that the scanner found SOMETHING; every failure of this
    # scanner class has been MIS-attribution, which sails past that check. So the binder is pinned to a
    # synthetic source whose correct answer is known by construction.
    fixture = """
    @inline _at_alpha(hw) = _double_pumped(hw) ? 1 : 0
    @inline _at_beta(hw) = hw.l1 ÷ 2
    @inline _wide_simd(hw) = hw.simd >= 64
    @inline _at_gamma(hw) = _wide_simd(hw) ? 8 : 4
    const _SOMETHING = 3
    @inline _at_delta(hw) = hw.l2
    """
    got = scan(fixture)
    # _at_alpha USES a predicate -> flagged. _at_beta does not -> not flagged.
    # _at_gamma USES one -> flagged. _at_delta does not -> not flagged.
    # ⚠ THE REAL BUG: `_wide_simd`'s own DEFINITION line contains "_wide_simd(", and the scanner used
    # to still be "inside" _at_beta there, so _at_beta was flagged for its neighbour's definition.
    # That is what this fixture exists to catch, and it is the same shape as three other bugs in this
    # tree (knob-registry marker binding, the marker rewriter, the audit skip-test).
    @test got == ["_at_alpha", "_at_gamma"]

    keyed = scan(cpu)

    @test !isempty(keyed)   # the scanner must actually find something, or it proves nothing
    missing_rows = [k for k in keyed if !occursin(k, doc)]
    isempty(missing_rows) || @error "predicate-keyed knob(s) with no entry in docs/src/tuning.md §4b. \
        Add a row giving the MECHANISM (why the physical criterion IS that predicate) and the \
        FALSIFIER (what measurement would kill it), or grade it Weak and say so — an unjustified \
        two-way µarch lookup is a CLAUDE.md 8b violation." knobs = missing_rows
    @test isempty(missing_rows)
end
