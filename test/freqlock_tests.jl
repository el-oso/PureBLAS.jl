# The frequency-lock criterion (bench/freqlock.jl) tested against SYNTHETIC sysfs trees.
#
# WHY THIS FILE EXISTS. On 2026-08-23 the guard written to stop off-lock gate runs was itself wrong:
# it aggregated `lo = max over cores' scaling_min_freq`, which reports a LOCKED box whenever any single
# core happens to be pinned high — so a box with core 0 at 2.0 GHz and core 8 free to range 0.4-2.0 GHz
# PASSED. Every happy-path check agreed with it. Only a fixture that must say NO caught it.
# See kb/findings/checks-that-answer-the-wrong-question.md.
@testitem "freqlock: lock_state aggregates over EVERY core" begin
    include(joinpath(@__DIR__, "..", "bench", "freqlock.jl"))
    mk = (root, cores; boost = 0) -> begin
        mkpath(root)
        for (i, (a, b)) in enumerate(cores)
            d = joinpath(root, "cpu$i", "cpufreq"); mkpath(d)
            write(joinpath(d, "scaling_min_freq"), string(a))
            write(joinpath(d, "scaling_max_freq"), string(b))
        end
        if boost >= 0
            mkpath(joinpath(root, "cpufreq")); write(joinpath(root, "cpufreq", "boost"), string(boost))
        end
        root
    end
    base = mktempdir()

    all_pinned = mk(joinpath(base, "pinned"), [(2000000, 2000000), (2000000, 2000000), (2000000, 2000000)])
    @test FreqLock.lock_state(all_pinned) == (2000000, 2000000, 0)
    @test first(FreqLock.freq_locked(all_pinned))

    # THE REGRESSION CASE: one core left free. max-over-minimums reported this as locked.
    one_free = mk(joinpath(base, "one_free"), [(2000000, 2000000), (2000000, 2000000), (400000, 2000000)])
    @test FreqLock.lock_state(one_free) == (400000, 2000000, 0)
    @test !first(FreqLock.freq_locked(one_free))

    # Heterogeneous pins are not a valid gate state either.
    hetero = mk(joinpath(base, "hetero"), [(2000000, 2000000), (2500000, 2500000)])
    @test !first(FreqLock.freq_locked(hetero))

    # Boost on, everything else perfect.
    boosted = mk(joinpath(base, "boosted"), [(2000000, 2000000)]; boost = 1)
    @test !first(FreqLock.freq_locked(boosted))
    @test occursin("boost", last(FreqLock.freq_locked(boosted)))

    # No cpufreq at all (ARM): degrade to "unchecked", never to "every cell invalid".
    nofreq = mkpath(joinpath(base, "nofreq"))
    @test FreqLock.lock_state(nofreq) == (0, 0, -1)
    @test first(FreqLock.freq_locked(nofreq))
end

@testitem "freqlock: require_lock refuses off-lock and honours the override" begin
    include(joinpath(@__DIR__, "..", "bench", "freqlock.jl"))
    # require_lock() reads the real machine, so only the override path is asserted here; the
    # aggregation itself is covered by the fixture test above.
    withenv("PUREBLAS_BENCH_NOLOCK" => "1") do
        @test isnothing(FreqLock.require_lock())
    end
end

# THE PIN IS NOT THE CLOCK. On 2026-08-23 neuromancer read min==max==2000000 with boost=0 and ran
# every measured cell at ~4.8 GHz anyway (drifted pstate driver), inflating a whole op's ratios by
# 2.43x against references taken at 1.98 GHz. `require_lock` cannot see that — only a sample taken
# UNDER LOAD can — so `check_achieved` is a separate gate and needs its own must-say-NO fixture.
@testitem "freqlock: check_achieved refuses a core running above its pin" begin
    include(joinpath(@__DIR__, "..", "bench", "freqlock.jl"))
    (_, hi, _) = FreqLock.lock_state()
    if hi > 0                       # skip where there is no cpufreq to reason about
        @test isnothing(FreqLock.check_achieved(hi))                 # exactly at the ceiling: fine
        @test isnothing(FreqLock.check_achieved(round(Int, hi * 1.02)))  # within tolerance: fine
        @test_throws ErrorException FreqLock.check_achieved(round(Int, hi * 2.4))  # the neuromancer case
    end
    @test isnothing(FreqLock.check_achieved(0))                      # nothing sampled: do not assert
    withenv("PUREBLAS_BENCH_NOLOCK" => "1") do
        @test isnothing(FreqLock.check_achieved(99_000_000))         # override still works
    end
end
