# Unit tests for the PDM "Measure" tier's DECISION RULE (`_tune_better`, `_tune_pick`).
#
# WHY THESE EXIST. The Derive tier has had an offline unit test since day one (autotune_tests.jl feeds
# fleet hardware descriptors to the formulas and asserts the measured optima). The Measure tier had
# NONE — its decision logic was open-coded inside each `_measure_*` closure, so it could only be
# exercised by running a benchmark, which is slow, machine-dependent and untestable in CI. On
# 2026-08-06 that logic was changed with nothing to catch a regression, and two latent defects were
# found by reading rather than by testing:
#   * the sweeps re-based the margin against a RUNNING BEST, so what a later candidate had to beat
#     depended on which earlier one happened to win;
#   * `_tune_better`'s docstring promises "ties go to the incumbent, which is the derived default",
#     which is FALSE at two of three sweep sites — they start from `typemax`, making the effective
#     incumbent whichever candidate is first in sweep order.
# Both are properties of a PURE function of times. They belong in a unit test, not in a benchmark.
#
# These tests use SYNTHETIC times. They deliberately do not measure anything: the point is to pin the
# decision rule's semantics so a future change to it is a visible, reviewable diff rather than a silent
# behavioural drift discovered a month later on one microarchitecture.
@testitem "tuner: margin semantics and candidate selection" begin
    using PureBLAS: _tune_better, _tune_pick

    # ── _tune_better: `t` displaces `best` only by winning MORE than 5%.
    @test _tune_better(UInt64(90), UInt64(100))          # 10% faster → yes
    @test !_tune_better(UInt64(96), UInt64(100))         # 4% faster  → no
    @test !_tune_better(UInt64(100), UInt64(100))        # tie        → no, incumbent keeps it
    @test !_tune_better(UInt64(110), UInt64(100))        # slower     → no
    # exact boundary: the rule is `t*100 < best*95`, i.e. STRICTLY more than 5%.
    @test !_tune_better(UInt64(95), UInt64(100))         # exactly 5% → NOT better (strict)
    @test _tune_better(UInt64(94), UInt64(100))          # just over  → better
    # no overflow surprise at realistic ns magnitudes (a 4096³ gemm is ~1e10 ns)
    @test _tune_better(UInt64(9) * 10^10, UInt64(10) * 10^10)
    @test !_tune_better(UInt64(96) * 10^9, UInt64(10) * 10^10)

    # ── _tune_pick: 0 keeps the incumbent; otherwise the index of the displacing candidate.
    inc = UInt64(1000)

    # REPLICATES OLD BEHAVIOUR: nothing clears the margin ⇒ incumbent survives. This is the property
    # that makes a Measure knob safe to add — a knob that cannot prove itself changes nothing.
    @test _tune_pick(inc, (UInt64(1000), UInt64(970), UInt64(1200))) == 0   # tie, 3% better, worse
    @test _tune_pick(inc, (UInt64(951),)) == 0                             # 4.9% — just under
    @test _tune_pick(inc, ()) == 0                                         # no candidates at all

    # PROGRESSION: a candidate that clears the margin is taken.
    @test _tune_pick(inc, (UInt64(900),)) == 1                             # 10% better
    @test _tune_pick(inc, (UInt64(1200), UInt64(900))) == 2                # only the second qualifies
    @test _tune_pick(inc, (UInt64(949),)) == 1                             # just over 5%

    # AMONG QUALIFIERS the margin applies again, ties to the EARLIER (cheaper) candidate.
    @test _tune_pick(inc, (UInt64(900), UInt64(880))) == 1   # both qualify; 2 beats 1 by 2.2% → keep 1
    @test _tune_pick(inc, (UInt64(900), UInt64(800))) == 2   # 2 beats 1 by 11% → take 2

    # THE LADDER that motivated the 2026-08-06 rewrite, in the measured proportions: candidate 1 is 9%
    # better than the incumbent, candidate 2 is 13.2% better than the incumbent but only 3.9% better
    # than candidate 1. The rule takes candidate 1, and this test exists to state that outcome
    # explicitly rather than leave it as folklore: the chained and unchained forms AGREE here, so the
    # rewrite bought a stated invariant and testability, not a different answer.
    @test _tune_pick(inc, (UInt64(917), UInt64(883))) == 1

    # QUALIFICATION IS AGAINST THE INCUMBENT, NOT A RUNNING BEST — the invariant the old open-coded
    # sweeps did not hold. Order must not change the verdict: same times, permuted, same winner.
    @test _tune_pick(inc, (UInt64(800), UInt64(900))) == 1   # 800 wins from position 1
    @test _tune_pick(inc, (UInt64(900), UInt64(800))) == 2   # ... and from position 2

    # A candidate slower than the incumbent can never win, whatever its neighbours do.
    @test _tune_pick(inc, (UInt64(2000), UInt64(1500))) == 0

    # ── REGRESSION: the `bt = typemax(UInt64)` seed, which three sweeps used until 2026-08-06.
    # `typemax(UInt64) * 95` WRAPS to 2^64-95 — still astronomically above any real elapsed time — so
    # `_tune_better(t, typemax)` is TRUE for every candidate and the first one swept always displaces.
    # The declared default (`best = 4` / `_UNROLL`) was therefore unreachable, and the effective
    # incumbent was whichever arm happened to be written first. This asserts the arithmetic so the
    # seed can never be "simplified" back.
    @test typemax(UInt64) * UInt64(95) == typemax(UInt64) - UInt64(94)   # it wraps, it does not saturate
    @test _tune_better(UInt64(10)^9, typemax(UInt64))                    # ⇒ anything beats a typemax seed
    # ...whereas seeding with the incumbent's own measured time makes ties keep the default:
    @test !_tune_better(UInt64(10)^9, UInt64(10)^9)
end

# The DUEL rule that replaced the margin for knobs the margin could not resolve. `_tune_wins_it` is the
# pure half (the counting); `_tune_duel` does the timing and is exercised by the acceptance test in
# bench/probes/tune_freshops_validate.jl plus the cross-process determinism check recorded below.
@testitem "tuner: supermajority rule and its tie probability" begin
    using PureBLAS: _tune_wins_it, _tune_rounds, _TUNE_ROUNDS, _TUNE_ALPHA, _TUNE_NKNOBS

    # Default 5 rounds: 4 or 5 wins displaces, 3 does not. One spoiled round is survivable, two is not.
    @test _tune_wins_it(5, 5)
    @test _tune_wins_it(4, 5)
    @test !_tune_wins_it(3, 5)
    @test !_tune_wins_it(0, 5)
    @test _tune_wins_it(_TUNE_ROUNDS - 1)      # the default is DERIVED, so assert it relationally
    @test !_tune_wins_it(_TUNE_ROUNDS - 2)

    # More rounds tightens it, which is how determinism is DIALLED rather than assumed.
    @test !_tune_wins_it(7, 9)
    @test _tune_wins_it(8, 9)

    # The tie false-positive rate is (rounds+1)/2^rounds — asserted so the docstring cannot drift from
    # the arithmetic. This is the number that replaces "5% is above the probes' resolution", which was
    # an unverified fleet-empirical claim; this one is exact and a property of the rule alone.
    tie_fp(r) = (r + 1) / 2.0^r
    @test tie_fp(5) ≈ 0.1875
    @test tie_fp(9) ≈ 0.01953125
    @test tie_fp(13) < 0.002
    # ...and it must be monotone decreasing in rounds, or "dial it with `rounds`" would be false.
    @test all(tie_fp(r + 1) < tie_fp(r) for r in 3:20)

    # ── THE ROUND COUNT IS DERIVED, and this pins the derivation so no literal can creep back.
    # The rule: with `ncand` duels per sweep, the family-wise tie rate must fit the per-knob share of
    # the one stated risk budget. Everything below follows from _TUNE_ALPHA alone.
    budget = _TUNE_ALPHA / _TUNE_NKNOBS
    for c in (2, 3, 4, 6, 8)
        r = _tune_rounds(c)
        @test c * tie_fp(r) <= budget                    # it MEETS the budget...
        @test r == 5 || c * tie_fp(r - 1) > budget       # ...and is the SMALLEST r that does
    end
    # More candidates can never need fewer rounds — the multiple-comparisons correction, asserted.
    @test all(_tune_rounds(c + 1) >= _tune_rounds(c) for c in 1:12)
    # The default serves the widest sweep in the tree (6 candidates: _measure_axpy_unroll).
    @test _TUNE_ROUNDS == _tune_rounds(6)
    # And it is comfortably stricter than the 5 rounds that measurably failed: at 5 rounds a 6-candidate
    # sweep flips ~70% of the time at a tie, which is what `axpy_band` was observed doing.
    @test 6 * tie_fp(5) > 0.5
    @test _TUNE_ROUNDS > 5
end

# The knob resolvers must return something USABLE on any host, including one where the measurement
# throws or is skipped (precompilation). `OncePerProcess` poisons the whole process if its initializer
# raises, so every `_measure_*` is written to be total — this asserts the contract at the resolver.
@testitem "tuner: Measure-tier resolvers are total and in-range" begin
    using PureBLAS: _cgemvn_nc_big, _ger_np, _vwidth, _CGEMVN_NC

    W = _vwidth(Float64)
    nc = _cgemvn_nc_big()
    @test nc isa Int
    @test nc in (_CGEMVN_NC, W, 3W ÷ 2)          # exactly the derived candidate set, nothing else
    @test nc >= 1
    @test _cgemvn_nc_big() === nc                # OncePerProcess: stable within a process

    np = _ger_np()
    @test np isa Int
    @test np in (1, 2, 4, 8)
    @test _ger_np() === np
end
