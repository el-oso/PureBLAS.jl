# Dogfooding StrictMode.jl — turns AllocCheck + JET + @inferred into declarable guarantees. We
# assert the hot-path kernels are type-stable, allocation-free, and trim-safe. Gated by a
# compile-time Preference (test/Project.toml ships them enabled); when disabled the macros are
# zero-cost no-ops, so we skip rather than pass vacuously. Mirrors PureFFT's strictmode dogfood.
#
# NOTE: @assert_noalloc is backed by AllocCheck (static, all-paths proof) ONLY in analysis="full"
# (our config); in :fast it degrades to a runtime check. The kernels here must be alloc-free on
# every path, so :full is the right mode. Driver steady-state (allocates scratch once) is guarded
# separately with runtime @allocated in gemm_tests.jl, where static AllocCheck would false-positive.

@testitem "StrictMode dogfood: BLAS-1 strict contract" begin
    using StrictMode, AllocCheck, JET
    if !StrictMode.checks_enabled()
        @info "StrictMode checks disabled — skipping dogfood (enable in test/Project.toml to run)"
        @test_skip StrictMode.checks_enabled()
    else
        # `AbstractBLAS1` is a @strict_contract (src/contracts.jl). @verify_strict re-checks the
        # method surface (TypeContracts.@verify) AND that each L1 backend call is type-stable and
        # allocation-free — here in the test project's FULL mode, so @noalloc is a static AllocCheck
        # all-paths proof (the backend is loaded at test runtime). Mirrors the fast-mode in-src check.
        bk = PureBLAS.DEFAULT_BACKEND
        n = 1000
        xd = randn(n); yd = randn(n)                          # SIMD fast path
        xz = randn(ComplexF64, n); yz = randn(ComplexF64, n)  # complex: axpy/dot generic; nrm2/asum SIMD
        @verify_strict PureBLAS.SIMDBackend begin
            PureBLAS.axpy!(bk, yd, 2.0, xd)
            PureBLAS.scal!(bk, 2.0, xd)
            PureBLAS.blascopy!(bk, yd, xd)
            PureBLAS.swap!(bk, xd, yd)
            PureBLAS.dot(bk, xd, yd)
            PureBLAS.dotu(bk, xd, yd)
            PureBLAS.nrm2(bk, xd)
            PureBLAS.asum(bk, xd)
            PureBLAS.iamax(bk, xd)
            PureBLAS.axpy!(bk, yz, 2.0 + 1.0im, xz)   # generic complex path
            PureBLAS.dot(bk, xz, yz)
            PureBLAS.nrm2(bk, xz)                      # complex nrm2/asum → SIMD real-reinterpret path
            PureBLAS.asum(bk, xz)
        end
        @test true
    end
end

@testitem "StrictMode dogfood: BLAS-2 strict contract" begin
    using StrictMode, AllocCheck, JET
    if !StrictMode.checks_enabled()
        @info "StrictMode checks disabled — skipping L2 dogfood"
        @test_skip StrictMode.checks_enabled()
    else
        # `AbstractBLAS2` is a @strict_contract (src/contracts.jl). @verify_strict checks the method
        # surface AND that each dense L2 backend call is type-stable and allocation-free — full mode
        # here, so @noalloc is a static AllocCheck all-paths proof. Mirrors the fast-mode in-src check.
        bk = PureBLAS.DEFAULT_BACKEND
        Ad = randn(64, 64); Az = randn(ComplexF64, 64, 64)
        um = randn(64); vm = randn(64); uz = randn(ComplexF64, 64); wz = randn(ComplexF64, 64)
        @verify_strict PureBLAS.SIMDBackend begin
            PureBLAS.gemv!(bk, vm, Ad, um; alpha = 2.0, beta = 1.0, trans = 'N')
            PureBLAS.gemv!(bk, vm, Ad, um; alpha = 2.0, beta = 1.0, trans = 'T')
            PureBLAS.ger!(bk, 1.5, um, vm, Ad)
            PureBLAS.symv!(bk, vm, Ad, um)
            PureBLAS.hemv!(bk, wz, Az, uz)
            PureBLAS.trmv!(bk, Ad, um)
            PureBLAS.trsv!(bk, Ad, um)
        end
        @test true
    end
end

@testitem "StrictMode dogfood: L3 trsm/syrk/symm scratch + driver" begin
    using StrictMode, AllocCheck, JET, LinearAlgebra
    if !StrictMode.checks_enabled()
        @info "StrictMode checks disabled — skipping L3 dogfood"
        @test_skip StrictMode.checks_enabled()
    else
        P = PureBLAS
        tri(s) = (M = tril(randn(s, s)); for i in 1:s; M[i, i] += s; end; M)
        # ROOT-CAUSE guard, on the CONSUMER (not the accessor — passing `Float64` as a value infers over
        # DataType, where T is unresolvable, a test artifact). Here T = eltype(B) is a concrete compile-
        # time parameter, so :full-mode JET report_opt sees the real specialization: the invL/invR base
        # holds the trtri+gemm scratch, and a non-concrete scratch return (view of an abstract
        # IdDict{DataType,Matrix} value) shows up as internal runtime dispatch / boxing here. This is the
        # check that was missing when the per-leaf boxing shipped — it passes only with concrete returns.
        Atri = tri(32)
        @assert_typestable P._trsm_base_invL!(false, false, false, Atri, randn(32, 256))
        @assert_typestable P._trsm_base_invR!(false, false, false, Atri, randn(256, 32))
        # Driver steady-state is allocation-free. Cached scratch allocates once on first touch, so this
        # is the empirical warmed path (static=false); static AllocCheck would false-positive on the
        # get!/IdDict lazy alloc. StrictMode macros reject kwargs, so we assert on the POSITIONAL
        # internal drivers — which is exactly where the scratch/views live: the invL/invR bases hold
        # the trtri+gemm scratch (the boxing site), the packed syrk/syr2k/symm hold the pack buffers.
        @assert_noalloc P._trsm_base_invL!(false, false, false, Atri, randn(32, 256)) static = false
        @assert_noalloc P._trsm_base_invR!(false, false, false, Atri, randn(256, 32)) static = false
        As = randn(512, 512); Bs = randn(512, 512); Cs = zeros(512, 512)
        @assert_noalloc P._syrk_blocked!(false, false, false, 0.8, As, Cs, 512) static = false
        As32 = randn(32, 32); Cs32 = zeros(32, 32)   # small-n unified single-pack path (AVX2)
        @assert_noalloc P._syrk_blocked!(false, false, false, 0.8, As32, Cs32, 32) static = false
        @assert_noalloc P._syr2k_packed!(false, false, 0.8, 0.3, As, Bs, Cs, 512) static = false
        @assert_noalloc P._symm!(true, false, false, 0.8, 0.3, As, Bs, Cs) static = false
        # PUBLIC ENTRY POINTS — assertable directly now that StrictMode ≥0.3.4 supports kwarg calls
        # (issue el-oso/StrictMode.jl#4). This closes the mandate: StrictMode on every entry point, not
        # just the positional internal drivers. :full-mode JET sees the whole kwarg→dispatch→kernel tree.
        @assert_typestable P.trsm!(copy(Bs), tri(512); side = 'L', uplo = 'L', diag = 'N', alpha = 1.0)
        @assert_typestable P.syrk!(zeros(512, 512), As; uplo = 'L', trans = 'N', alpha = 0.8, beta = 0.3)
        @assert_typestable P.syr2k!(zeros(512, 512), As, Bs; uplo = 'L', trans = 'N', alpha = 0.8, beta = 0.3)
        @assert_typestable P.symm!(zeros(512, 512), As, Bs; side = 'L', uplo = 'L', alpha = 0.8, beta = 0.3)
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
            # clip kernel: W-aligned partial row-tile (reads _MR-strided panel, computes 1 live vector)
            @assert_typestable P._microkernel_clip!(cp, ldc, ap, bp, kc, Val(P._MR), Val(1), Val(P._NR))
            @assert_noalloc P._microkernel_clip!(cp, ldc, ap, bp, kc, Val(P._MR), Val(1), Val(P._NR))
            @assert_trim_safe P._microkernel_clip!(cp, ldc, ap, bp, kc, Val(P._MR), Val(1), Val(P._NR))
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
