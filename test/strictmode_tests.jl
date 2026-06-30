# Dogfooding StrictMode.jl — turns AllocCheck + JET + @inferred into declarable guarantees. We
# assert the hot-path kernels are type-stable, allocation-free, and trim-safe. Gated by a
# compile-time Preference (test/Project.toml ships them enabled); when disabled the macros are
# zero-cost no-ops, so we skip rather than pass vacuously. Mirrors PureFFT's strictmode dogfood.
#
# NOTE: @assert_noalloc is backed by AllocCheck (static, all-paths proof) ONLY in analysis="full"
# (our config); in :fast it degrades to a runtime check. The kernels here must be alloc-free on
# every path, so :full is the right mode. Driver steady-state (allocates scratch once) is guarded
# separately with runtime @allocated in gemm_tests.jl, where static AllocCheck would false-positive.

@testitem "StrictMode dogfood: PureBLAS Level-1 perf guarantees" begin
    using StrictMode, AllocCheck, JET
    if !StrictMode.checks_enabled()
        @info "StrictMode checks disabled — skipping dogfood (enable in test/Project.toml to run)"
        @test_skip StrictMode.checks_enabled()
    else
        P = PureBLAS
        n = 1000
        xd = randn(Float64, n); yd = randn(Float64, n)         # SIMD fast path
        xz = randn(ComplexF64, n); yz = randn(ComplexF64, n)   # generic scalar path
        # SIMD real path
        @assert_typestable P._axpy!(n, 2.0, xd, 1, yd, 1)
        @assert_noalloc P._axpy!(n, 2.0, xd, 1, yd, 1)
        @assert_trim_safe P._axpy!(n, 2.0, xd, 1, yd, 1)
        @assert_noalloc P._scal!(n, 2.0, xd, 1)
        @assert_noalloc P._copy!(n, xd, 1, yd, 1)
        @assert_typestable P._dotu(n, xd, 1, yd, 1)
        @assert_noalloc P._dotu(n, xd, 1, yd, 1)
        @assert_noalloc P._nrm2(n, xd, 1)
        @assert_noalloc P._asum(n, xd, 1)
        @assert_noalloc P._iamax(n, xd, 1)
        # generic complex path
        @assert_typestable P._axpy!(n, 2.0 + 1.0im, xz, 1, yz, 1)
        @assert_noalloc P._axpy!(n, 2.0 + 1.0im, xz, 1, yz, 1)
        @assert_typestable P._dotc(n, xz, 1, yz, 1)
        @assert_noalloc P._nrm2(n, xz, 1)
        @test true
    end
end

@testitem "StrictMode dogfood: Level-2 gemv/ger" begin
    using StrictMode, AllocCheck, JET
    if !StrictMode.checks_enabled()
        @info "StrictMode checks disabled — skipping L2 dogfood"
        @test_skip StrictMode.checks_enabled()
    else
        P = PureBLAS
        A = randn(64, 48); xN = randn(48); yN = randn(64); xT = randn(64); yT = randn(48)
        @assert_typestable P._gemv!(false, false, 64, 48, 2.0, A, xN, 1, 1.0, yN, 1)
        @assert_noalloc P._gemv!(false, false, 64, 48, 2.0, A, xN, 1, 1.0, yN, 1)
        @assert_trim_safe P._gemv!(false, false, 64, 48, 2.0, A, xN, 1, 1.0, yN, 1)
        @assert_typestable P._gemv!(true, false, 64, 48, 2.0, A, xT, 1, 1.0, yT, 1)
        @assert_noalloc P._gemv!(true, false, 64, 48, 2.0, A, xT, 1, 1.0, yT, 1)
        xg = randn(64); yg = randn(48); Ag = zeros(64, 48)
        @assert_typestable P._ger!(false, 64, 48, 1.5, xg, 1, yg, 1, Ag)
        @assert_noalloc P._ger!(false, 64, 48, 1.5, xg, 1, yg, 1, Ag)
        @assert_trim_safe P._ger!(false, 64, 48, 1.5, xg, 1, yg, 1, Ag)
        @test true
    end
end

@testitem "StrictMode dogfood: GEMM hot paths" begin
    using StrictMode, AllocCheck, JET
    if !StrictMode.checks_enabled()
        @info "StrictMode checks disabled — skipping GEMM dogfood"
        @test_skip StrictMode.checks_enabled()
    else
        P = PureBLAS
        W = P._vwidth(Float64); mr = P._MR * W; nr = P._NR; kc = 64
        Ap = randn(mr * kc); Bp = randn(nr * kc); C = zeros(mr, nr)
        GC.@preserve Ap Bp C begin
            ap = pointer(Ap); bp = pointer(Bp); cp = pointer(C); ldc = mr
            # register-blocked microkernel: the hot path — must be tight
            @assert_typestable P._microkernel!(cp, ldc, ap, bp, kc, Val(P._MR), Val(P._NR))
            @assert_noalloc P._microkernel!(cp, ldc, ap, bp, kc, Val(P._MR), Val(P._NR))
            @assert_trim_safe P._microkernel!(cp, ldc, ap, bp, kc, Val(P._MR), Val(P._NR))
            @assert_noalloc P._microkernel_masked!(cp, ldc, ap, bp, kc, 11, 5, Val(P._MR), Val(P._NR))
            @assert_typestable P._microkernel_masked!(cp, ldc, ap, bp, kc, 11, 5, Val(P._MR), Val(P._NR))
            @assert_trim_safe P._microkernel_masked!(cp, ldc, ap, bp, kc, 11, 5, Val(P._MR), Val(P._NR))
        end
        # unpacked microkernel (small-matrix path): A is mr×k, B is k×nr, column-major
        kk = 32; Au = randn(mr * kk); Bu = randn(kk * nr); Cu = zeros(mr, nr)
        GC.@preserve Au Bu Cu begin
            aup = pointer(Au); bup = pointer(Bu); cup = pointer(Cu)
            @assert_typestable P._microkernel_unpacked!(cup, mr, aup, mr, 0, bup, kk, 0, kk, 1.0, 0.0,
                Val(P._MR), Val(P._NR), Val(false), Val(true))
            @assert_noalloc P._microkernel_unpacked!(cup, mr, aup, mr, 0, bup, kk, 0, kk, 1.0, 2.0,
                Val(P._MR), Val(P._NR), Val(false), Val(false))
            @assert_trim_safe P._microkernel_unpacked!(cup, mr, aup, mr, 0, bup, kk, 0, kk, 1.0, 0.0,
                Val(P._MR), Val(P._NR), Val(false), Val(true))
            # masked-row kernel (partial rows): mre=12 → second row-vector partially masked
            @assert_typestable P._microkernel_unpacked_mrows!(cup, mr, aup, mr, 0, bup, kk, 0, kk,
                1.0, 0.0, 12, Val(P._MR), Val(P._NR), Val(false), Val(true))
            @assert_noalloc P._microkernel_unpacked_mrows!(cup, mr, aup, mr, 0, bup, kk, 0, kk,
                1.0, 2.0, 12, Val(P._MR), Val(P._NR), Val(false), Val(false))
            @assert_trim_safe P._microkernel_unpacked_mrows!(cup, mr, aup, mr, 0, bup, kk, 0, kk,
                1.0, 0.0, 12, Val(P._MR), Val(P._NR), Val(false), Val(true))
        end
        # packing + generic path allocate nothing
        A = randn(8, 5); Bm = randn(5, 6); Cg = zeros(8, 6)
        @assert_typestable P._gemm_generic!(false, false, false, false, 8, 6, 5, 1.0, A, Bm, 0.0, Cg)
        @assert_noalloc P._gemm_generic!(false, false, false, false, 8, 6, 5, 1.0, A, Bm, 0.0, Cg)
        @test true
    end
end
