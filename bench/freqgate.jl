# FREQUENCY ADJUDICABILITY — shared by gate_gaps.jl and coverage_ops.jl. `include` it, no module.
#
# v3 arm records carry a PER-CELL achieved clock (`arm|time|commit|anchor|freq|csv`, kHz, sampled at
# measurement time). A cell measured while the lock was floating is NOT ADJUDICABLE: its ratio was taken
# at a different clock from the rest of the sweep, so averaging it into a verdict silently mixes two
# machine states. Such a cell is EXCLUDED and REPORTED — a partially drifted run must read as partial,
# not as quietly wrong. This exists because neuromancer's 2026-08-16 sweep verified ✅ at launch and
# still stamped freq=2172337kHz against a 2000 MHz base: the lock floated DURING the run, and a single
# run-level clock could not say which of the 85 ops were affected, so all of them were condemned.
#
# WHICH REFERENCE CLOCK, and why this one. The right reference would be the box's base clock
# (`cpuinfo_base_freq` / the `scaling_max_freq` a locked run pins to), but NOTHING in the cache header
# records it — `hw=` stamps cache sizes and ISA, not clocks. The only clock in the file is the header's
# own `freq=`, which is the ACHIEVED clock (max over cores) at save time. So that is the reference, and
# a cell is flagged only when it sits >1% ABOVE it. A per-box table of base clocks is NOT an option:
# that is a hardcoded per-machine literal, which project rule #8b forbids.
#
# Consequences of that choice, stated because they bound what a clean report means:
#   * The reference is self-referential — a run that floated for its WHOLE duration has a floated
#     header too, and nothing here will flag it. This detects DRIFT WITHIN a run (which cells left the
#     lock the rest of the sweep held), not a uniformly wrong lock. The launch-time `verify` in
#     bench/fleet_freqlock.sh remains the only check for the latter.
#   * The header is a max over ALL cores while a cell's clock is its own pinned core, so the reference
#     is an upper bound on the honest per-cell value. That makes the flag CONSERVATIVE: it under-reports
#     rather than condemning good cells, which is the right direction for a rule that triggers re-runs.
#
# freq == 0 means UNKNOWN (record written before the field existed, or no cpufreq). Unknown is NEVER
# off-lock — treating it as a float would invalidate every historical cache at a stroke.
_freq_offlock(khz::Integer, ref::Integer) = ref > 0 && khz > 0 && khz > 1.01 * ref

# Header reference clock. Returns (ref_khz, backfilled) — `backfilled` true when the per-cell values in
# this file were BACKFILLED from the header rather than sampled per cell (bench/ backfill marker
# `freq_backfilled=`). A backfilled file can never flag anything (every cell equals the reference by
# construction), and callers say so rather than reporting a clean bill of health it did not earn.
function _freq_ref(header::AbstractString)
    kv = Dict(String(p[1]) => String(p[2])
        for p in (split(x, "=", limit = 2) for x in split(header, "\t")[2:end]) if length(p) == 2)
    khz(k) = something(tryparse(Int, replace(get(kv, k, ""), "kHz" => "")), 0)
    return khz("freq"), haskey(kv, "freq_backfilled")
end

# Per-arm clock out of a split record. The csv is ALWAYS last and fields are appended before it, so
# index from the front only after checking the length (the same invariant plots.jl's reader relies on).
_freq_of(p) = length(p) >= 6 ? something(tryparse(Int, p[5]), 0) : 0
