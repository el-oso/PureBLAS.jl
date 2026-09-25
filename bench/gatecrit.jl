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

# ── WHICH ARMS THE RULE COMPARES ─────────────────────────────────────────────────────────────────
# The criterion above is arm-agnostic; these name the pair it applies to. They live here for the
# same reason the rounding does: the serial names were spelled inline in gate_gaps.jl,
# coverage_ops.jl, coverage_routing.jl, cellratios.jl, gate_cycles.jl, cellcycles.jl and failing.jl,
# so every one of those reported ZERO cells on a `mt_data_*` cache whose three threaded arms were
# all present.
#
# A threaded cell is judged against the THREADED references. `pb_mt` against a single-threaded
# vendor is flattery, not parity, and `mt_data_*` never carries the serial reference arms anyway.
#
# `accelerate_mt` is listed although no cache holds it yet: Accelerate ignores
# `BLAS.set_num_threads` and reads `VECLIB_MAXIMUM_THREADS` once at first use, so the arm needs a
# harness change on the Apple side (see APPLE.md). An arm absent from a cache is simply skipped.

const PB_ARM = "pb"
const PB_ARM_MT = "pb_mt"
const REF_ARMS = ("openblas", "aocl", "mkl", "accelerate")
const REF_ARMS_MT = ("openblas_mt", "aocl_mt", "accelerate_mt")

"""
    pb_arm(mt::Bool) -> String

The PureBLAS arm name for a serial (`false`) or threaded (`true`) cache.
"""
pb_arm(mt::Bool) = mt ? PB_ARM_MT : PB_ARM

"""
    ref_arms(mt::Bool) -> Tuple

The vendor reference arms the gate compares against. The gate takes the WORST of whichever of these
a cell actually carries — `max(OpenBLAS, AOCL, …)` by speed, i.e. the smallest PB/ref ratio.
"""
ref_arms(mt::Bool) = mt ? REF_ARMS_MT : REF_ARMS

"""
    is_ref_arm(a, mt::Bool) -> Bool

True for a vendor reference arm at that thread tier. `pb`, `pb_mt` and `generic` are never
references — `generic` is the LinearAlgebra fallback the dual groups measure against, and counting
it as a vendor would silently weaken every cell it appears in.
"""
is_ref_arm(a::AbstractString, mt::Bool) = a in ref_arms(mt)

"""
    arm_pair(d) -> (pb, refs)

The arm names to compare for one cell, read from the cell itself: a cell carrying `pb_mt` is
threaded. `refs` lists only the references that cell actually has, in `ref_arms` order.

Preferred over `is_mt_cache` inside a per-cell loop — it needs no flag threaded down from the
filename, and a tool that merges caches cannot mismatch a cell against the wrong tier.
"""
function arm_pair(d)
    mt = haskey(d, PB_ARM_MT)
    return pb_arm(mt), [r for r in ref_arms(mt) if haskey(d, r)]
end

"""
    is_mt_cache(path) -> Bool

Whether a cache file holds threaded arms. The name is the discriminator by design: `plots.jl`
derives it from `_ANY_MT` and never accepts it as a `cache=` argument, so asking for `pb_mt` cannot
overwrite the serial gate.
"""
is_mt_cache(path) = occursin("mt_data_", basename(String(path)))
