# FREQUENCY ADJUDICABILITY — shared by gate_gaps.jl and coverage_ops.jl. `include` it, no module.
#
# v3 arm records carry a PER-CELL achieved clock (`arm|time|commit|anchor|freq|csv`, kHz, sampled at
# measurement time). A cell measured while the lock was floating is NOT ADJUDICABLE: its ratio was taken
# at a different clock from the rest of the sweep, so averaging it into a verdict silently mixes two
# machine states. Such a cell is EXCLUDED and REPORTED — a partially drifted run must read as partial,
# not as quietly wrong. This exists because Zen5's 2026-08-16 sweep verified ✅ at launch and
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

# CROSS-ARM IN-WINDOW CLOCK AGREEMENT — the check `_freq_offlock` above cannot make.
#
# `_freq_of` reads the PIN (field 5), one number per arm. A threaded reference arm can hold that pin on
# paper and still collapse DURING its own window: six cores of a vendor BLAS on a power-limited part
# pull the package clock down where one core never does. Measured on Zen5, L3/gemm@1024 — `openblas_mt`
# ran its window between 953764 and 2000000 kHz and `aocl_mt` between 1799900 and 2000000, while the
# `pb` and `pb_mt` windows of the same cell held 1993521-2000000. The ratio there compares two machine
# states, and the error runs in PureBLAS's FAVOUR, because a throttled reference is a slower reference.
#
# Arm records carry the window as `flo|fhi` (fields 6 and 7), so the disagreement is readable per cell.
# `_window_of` returns (lo, hi), zeros when the fields predate the format. A cell is flagged when the
# lowest floor across its arms sits more than `tol` below the highest ceiling: that is one arm having
# spent part of its window at a clock another arm never saw.
#
# Zero or missing bounds mean UNKNOWN and never flag, for the same reason `_freq_offlock` treats
# freq==0 as unknown — a stricter rule would condemn every cache written before the fields existed.
_window_of(p) = length(p) >= 8 ?
    (something(tryparse(Int, p[6]), 0), something(tryparse(Int, p[7]), 0)) : (0, 0)

# FLOOR AGAINST FLOOR, not floor against ceiling.
#
# The question is whether the arms of one cell ran at comparable clocks. Comparing the LOWEST floor to
# the HIGHEST ceiling answers a different one — it compounds jitter at both ends into a span no single
# arm ever had, and then flags arms that are fine. Measured: a freshly taken `openblas_mt` arm on a
# verified-locked box recorded flo = 1938223 against a sibling's fhi = 2000000, which that form called a
# 3.1% mismatch; the same arm against the other FLOORS is 2.6% and plainly healthy. Left as it was, the
# rule asked for a re-measure and would have flagged the re-measured arm too, without converging.
#
# The real failure it exists to catch is unaffected, because a throttled arm's floor collapses: the
# October case is `openblas_mt` at 953764 beside siblings at 1990236, a 52% spread between floors.
# TOLERANCE 10%, AND IT HAS TO ACCOMMODATE ALL-CORE DROOP. A single-threaded arm sees none and can
# record the pin exactly, while a six-threaded arm on the same cell dips a few percent below it — our
# own `fleet_freqlock.sh verify-mt` calls 1980 of 2000 MHz a healthy hold. A cut tight enough to catch
# that difference condemns arms that are fine, and the re-measured arm then fails it too, so the repair
# never converges. Measured across 1142 cells on each of three boxes, the two populations are plainly
# separated: healthy cells sit at or below 2% (671, 1000 and 1126 of them) and real throttles above 20%
# (405, 80 and 0), with only 69 cells fleet-wide anywhere between 2% and 10%. 10% also matches what the
# lock methodology already treats as close enough to the pin, since `lock` steps a power-limited box
# down to 90% of its sustained clock.
function _freq_window_mismatch(windows, tol::Float64 = 0.10)
    los = Int[]
    for (lo, hi) in windows
        (lo > 0 && hi > 0) || continue
        push!(los, lo)
    end
    length(los) >= 2 || return false
    hi = maximum(los)
    return hi > 0 && (hi - minimum(los)) / hi > tol
end

# WHICH ARMS OF A CELL TO DISCARD, which is what both consumers of the rule above actually need.
# `_freq_window_mismatch` answers "did these windows run at one clock"; it cannot answer "whose
# number is unusable", and two properties of the data decide that.
#
# ONE: A FLOOR IS ONLY COMPARABLE TO A FLOOR FROM THE SAME SAMPLER. Three samplers write `flo|fhi`,
# named per arm in `span`: `core` reads the main thread's own core, `threads` the running cores from
# sysfs, `mtanchor` a timed all-cores reduction scaled against the pin. They measure different
# quantities, so a spread between two of them is not a clock disagreement. Measured on Zen3,
# L1/axpy@300000: `pb_mt` at 2309308 (`mtanchor`) beside vendor arms at 3694847 and 3699470 (both
# `core`) on a box pinned 3701000 — a 38% spread that is entirely the sampler. Gating by `span` is
# also what `bench/check_arm_clocks.sh` already does, where the below-pin test applies only to
# `pb`+`core` and `pb_mt`+`mtanchor`.
#
# `mtanchor` additionally under-reads for an arm that does not load every core: it has to wake the
# idle ones to time them, so the reading carries their ramp. That is why the low arm in the example
# above is `pb_mt` on an op PureBLAS does not thread.
#
# TWO: ONLY A SLOW REFERENCE CORRUPTS A VERDICT. The filter exists because a throttled reference is a
# slower reference, so the ratio flatters PureBLAS. A throttled PB arm runs the other way: the gate
# view understates the ratio and the threading view understates the speedup. Both readings are
# conservative, so they are published rather than hidden, and no PB arm is ever dropped here.
#
# Measured across 1142 cells per box: dropping on cross-arm spread alone condemns 95 cells on Zen3 and
# 431 on Zen5; gating by `span` and requiring a REFERENCE floor below the best PB floor leaves 1 and
# 431. Zen5's are real reference throttles to a cell; Zen3's 94 were artifacts of this function.
#
# `arms` is any iterable of `(name, flo, fhi, span)`. Returns the names to discard, empty when the
# cell is sound or carries too little information to judge.
function _throttled_ref_arms(arms, tol::Float64 = 0.10)
    drop = Set{String}()
    spans = unique(a[4] for a in arms)
    for sp in spans
        g = [a for a in arms if a[4] == sp && a[2] > 0 && a[3] > 0]
        length(g) >= 2 || continue
        pb = [a[2] for a in g if !_is_ref_arm(a[1])]
        isempty(pb) && continue
        best = maximum(pb)
        for a in g
            _is_ref_arm(a[1]) && a[2] < (1 - tol) * best && push!(drop, String(a[1]))
        end
    end
    return drop
end

# A REFERENCE arm is a vendor library; everything else (`pb`, `pb_mt`, and the Dual/ForwardDiff arm) is
# ours. Matched by prefix because each vendor contributes both a serial and an `_mt` arm.
_is_ref_arm(name) = startswith(name, "openblas") || startswith(name, "aocl") ||
    startswith(name, "accelerate") || startswith(name, "mkl")
