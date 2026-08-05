# Direct correctness coverage for every real-axpy SHAPE ARM.
#
# Why this exists: the arm actually taken at run time is chosen by a Measure-tier knob, and the test
# environment PINS that knob (test/Project.toml) so the StrictMode all-paths noalloc proof sees the
# shipping path. A consequence, flagged in review, is that the pin also decides which arm the rest of
# the suite ever executes — so without this item the non-pinned arms (`_axpy_phase!` in particular)
# would ship with NO test-suite execution at all. Coverage must not depend on what a knob happens to
# resolve to; it is asserted here directly.
#
# Arms mirror the knob's codes (src/simd_kernels.jl): u -> interleaved(u); 100+u -> phase(u, full
# width); 208 -> phase(8, narrow/256-bit). The narrow arm is exercised at whatever `W ÷ 2` is on the
# host — on AVX2 that is 128-bit, which is slow but must still be CORRECT.
@testitem "axpy shape arms: every arm computes y += a*x" begin
    using Random
    W = PureBLAS._vwidth(Float64)
    Random.seed!(0xA491)
    for T in (Float64, Float32)
        Wt = PureBLAS._vwidth(T)
        for n in (0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 33, 64, 129, 1000, 4096)
            x = rand(T, n); y0 = rand(T, n); a = T(0.75)
            ref = y0 .+ a .* x
            arms = Any[
                ("i2", yy -> PureBLAS._axpy_unrolled!(Val(2), n, a, pointer(x), pointer(yy), 0)),
                ("i4", yy -> PureBLAS._axpy_unrolled!(Val(4), n, a, pointer(x), pointer(yy), 0)),
                ("i8", yy -> PureBLAS._axpy_unrolled!(Val(8), n, a, pointer(x), pointer(yy), 0)),
                ("p4", yy -> PureBLAS._axpy_phase!(Val(4), Val(Wt), n, a, pointer(x), pointer(yy))),
                ("p8", yy -> PureBLAS._axpy_phase!(Val(8), Val(Wt), n, a, pointer(x), pointer(yy))),
            ]
            Wt >= 2 && push!(
                arms, ("p8n", yy -> PureBLAS._axpy_phase!(Val(8), Val(Wt ÷ 2), n, a, pointer(x), pointer(yy)))
            )
            for (nm, f) in arms
                yy = copy(y0)
                GC.@preserve x yy f(yy)
                @test yy ≈ ref
            end
        end
    end
    # The prefetch-distance argument of the interleaved kernel must not change the RESULT — it is a
    # hint. A wrong distance is a perf bug; a wrong result would be a correctness bug hiding behind one.
    Random.seed!(7)
    n = 4096; x = rand(n); y0 = rand(n); ref = y0 .+ 0.5 .* x
    for pf in (0, 8, 64, 512)
        yy = copy(y0)
        GC.@preserve x yy PureBLAS._axpy_unrolled!(Val(4), n, 0.5, pointer(x), pointer(yy), pf)
        @test yy ≈ ref
    end
end
