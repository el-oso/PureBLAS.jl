# THE GATE CRITERION — SINGLE SOURCE OF TRUTH. Every tool that prints PASS/FAIL, colours a coverage
# cell, or counts a miss must go through here. Do not re-spell the comparison inline: before this file
# existed the literal `1.0` appeared in plots.jl, gate_misses.jl, gate_gaps.jl, coverage_ops.jl,
# coverage_routing.jl and adjudicate.sh, and changing the rule meant finding all six.
#
# THE RULE (user decision, 2026-08-18): PB / max(OpenBLAS, AOCL) passes when the ratio **rounded to two
# significant digits** is >= 1.00. The gate itself is unchanged at >= 1.00; what changed is that the
# comparison is made on the ROUNDED figure, which is the figure the tables and plots actually publish.
#
# WHY THIS IS NOT A WEAKENING. Reporting a cell as a miss at a precision the measurement cannot support
# is a false negative. Per-cell machine-state drift on this fleet runs ~1-6% (each cached arm stores the
# anchor it was measured under), so the third digit of a ratio is not adjudicable — a cell reading 0.996
# and one reading 1.004 are the same measurement. Rounding to the published precision makes the verdict
# agree with what a reader can actually see in the table.
#
# Concretely: 0.995 -> 1.0 PASS · 0.9949 -> 0.99 FAIL · 1.004 -> 1.0 PASS · 0.98 -> 0.98 FAIL.
# The effective threshold is therefore 0.995, and `GATE_MIN` states it for tools that need a number
# (bands, sorts, shell one-liners) rather than a predicate.

"""
    gate_pass(r) -> Bool

True when ratio `r` meets the gate: `round(r; sigdigits=2) >= 1.0`. `NaN` (a missing cell) is never a
pass — an absent measurement is not a passing one.
"""
gate_pass(r) = !isnan(r) && round(r; sigdigits = 2) >= 1.0

"""
    GATE_MIN

The smallest ratio that passes, `0.995`. Exposed for band edges, sorting and shell tools; prefer
`gate_pass` wherever a predicate will do, so the rounding rule stays in one place.
"""
const GATE_MIN = 0.995
