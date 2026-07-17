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
    # StrictMode.TypeContracts: TypeContracts 0.14.0's @verify emits a `_seal_verified!(@__MODULE__,…)`
    # that resolves `TypeContracts` in THIS module (@verify_strict esc's the forwarded @verify call), so
    # the name must be in scope here. Reach it through StrictMode (already a dep) — no new test dep.
    using StrictMode, StrictMode.TypeContracts, AllocCheck, JET
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
            PureBLAS.axpy!(bk, yz, 2.0 + 1.0im, xz)   # axpy/dot: generic complex path
            PureBLAS.scal!(bk, 2.0 + 1.0im, xz)       # complex scal: interleaved-SIMD (swap-pairs)
            PureBLAS.dot(bk, xz, yz)                   # complex dot/dotu: split-deinterleave SIMD reduction
            PureBLAS.dotu(bk, xz, yz)
            PureBLAS.nrm2(bk, xz)                      # complex nrm2/asum → SIMD real-reinterpret path
            PureBLAS.asum(bk, xz)
        end
        @test true
    end
end

@testitem "StrictMode dogfood: BLAS-2 strict contract" begin
    # StrictMode.TypeContracts: see the BLAS-1 item — @verify_strict's forwarded @verify (TypeContracts
    # 0.14.0) seals into this module, so `TypeContracts` must resolve here.
    using StrictMode, StrictMode.TypeContracts, AllocCheck, JET
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
            PureBLAS.gemv!(bk, wz, Az, uz; alpha = 2.0 + 0im, beta = 1.0 + 0im, trans = 'N')  # complex gemv
            PureBLAS.gemv!(bk, wz, Az, uz; alpha = 2.0 + 0im, beta = 1.0 + 0im, trans = 'C')
            PureBLAS.ger!(bk, 1.5, um, vm, Ad)
            PureBLAS.ger!(bk, 1.5 + 0.5im, uz, wz, Az)   # complex geru/gerc: per-column complex axpy
            PureBLAS.symv!(bk, vm, Ad, um)
            PureBLAS.hemv!(bk, wz, Az, uz)
            PureBLAS.trmv!(bk, Ad, um)
            PureBLAS.trsv!(bk, Ad, um)
            PureBLAS.trmv!(bk, Az, uz)                    # complex trmv/trsv: per-column axpy(N)/dot(T/C)
            PureBLAS.trsv!(bk, Az, uz)
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
        # WARM the trsm_tmp scratch first: @assert_typestable above is JET-static (never executes), and
        # trsm_tmp is trsm-specific (unlike the syrk pack buffers, pre-grown by the earlier GEMM items in
        # this worker). Without this the noalloc target IS the first-touch grow → scheduling-flaky fail.
        P._trsm_base_invL!(false, false, false, Atri, randn(32, 256))
        P._trsm_base_invR!(false, false, false, Atri, randn(256, 32))
        @assert_noalloc P._trsm_base_invL!(false, false, false, Atri, randn(32, 256)) static = false
        @assert_noalloc P._trsm_base_invR!(false, false, false, Atri, randn(256, 32)) static = false
        # Fused gemmtrsm leaf (side-L upper, the wide-B gate shape f64) — typestable + alloc-free steady
        # state. Covers the transpose pack (shufflevector kernels) + the const-owned ftrsm buffer.
        Aup = (M = triu(randn(128, 128)); for i in 1:128; M[i, i] += 128.0; end; M)
        P._trsm_fused_L!(false, Aup, randn(128, 256))                     # warm the ftrsm buffer
        @assert_typestable P._trsm_fused_L!(false, Aup, randn(128, 256))
        @assert_noalloc P._trsm_fused_L!(false, Aup, randn(128, 256)) static = false
        # Whole-k packed sweep (shared-panel restructure; default-off toggle) — same typestable + alloc-free
        # contract. Covers _pack_U_micro! + the packed slab/tail kernels reading the ftrsm buffer. AVX-512-f64
        # ONLY: `_trsm_fused_full_L!` is dispatched (trsm.jl:_trsm_left!) solely under `_GT_TRANSPOSE` and has
        # no non-transpose fallback (its slab kernels require W==MR==8), so a DIRECT call on AVX2 throws by
        # design — gate the dogfood on the same predicate the dispatcher uses.
        if PureBLAS._GT_TRANSPOSE
            Auf = (M = triu(randn(512, 512)); for i in 1:512; M[i, i] += 512.0; end; M)
            P._trsm_fused_full_L!(false, Auf, randn(512, 256))               # warm the ftrsm buffer
            @assert_typestable P._trsm_fused_full_L!(false, Auf, randn(512, 256))
            @assert_noalloc P._trsm_fused_full_L!(false, Auf, randn(512, 256)) static = false
        end
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
        # OWNED-SCRATCH (GKH) guard — @assert_owned (StrictMode ≥0.3.5) fails on a runtime AbstractDict
        # lookup reached on the hot path. This is the check that WAS MISSING when the complex `_symm_scr`
        # scratch accessor shipped with const-dispatched owned Refs only for Float64/Float32: ComplexF64/F32
        # fell through to the generic `get(::IdDict, T, …)` (~130 ns/call, ~26% of a tiny-n op). That's
        # type-stable + alloc-free (warm hit) + trim-safe, so it passed all three OTHER asserts — only a
        # benchmark caught it. Now every eltype has an owned Ref, and this guard goes red if that regresses.
        Az = randn(ComplexF64, 64, 64); Hz = Az + Az'; Bz = randn(ComplexF64, 64, 64)
        @assert_owned P.hemm!(zeros(ComplexF64, 64, 64), Hz, Bz; side = 'L', uplo = 'U', alpha = 1.0 + 0im, beta = 0.0im)
        @assert_owned P.symm!(zeros(ComplexF64, 64, 64), Az, Bz; side = 'L', uplo = 'U', alpha = 1.0 + 0im, beta = 0.0im)
        # UNPACKED-TRI complex rank-k (small-n trans='N', `_ctri_unpacked!` → `_uker_cmplx!` TRI-store): the
        # path that fixed the zsyrk/zherk n≈24–48 valley. Direct-read A, no pack, masks the diagonal tile —
        # must stay typestable + alloc-free (the tri sweep resolves sb/a1/ar/nr to concrete Vals). trim-side
        # is covered ccallable-rooted in trim_tests.jl; here the type/alloc contract on the hot driver.
        Awz = randn(ComplexF64, 48, 40); Bwz = randn(ComplexF64, 48, 40)
        @assert_typestable P._ctri_unpacked!(true, true, 1.0, Awz, zeros(ComplexF64, 48, 48), 40)
        @assert_noalloc P._ctri_unpacked!(true, true, 1.0, Awz, zeros(ComplexF64, 48, 48), 40) static = false
        @assert_noalloc P._ctri_unpacked!(false, false, 1.2 + 0.3im, Awz, zeros(ComplexF64, 48, 48), 40) static = false
        @assert_typestable P.herk!(zeros(ComplexF64, 48, 48), Awz; uplo = 'U', trans = 'N', alpha = 1.0, beta = 0.0)
        # rank-2k (two products through the shared _ctri_core!)
        @assert_typestable P._ctri2_unpacked!(true, true, 1.0, Awz, Bwz, zeros(ComplexF64, 48, 48), 40)
        @assert_noalloc P._ctri2_unpacked!(true, true, 1.0, Awz, Bwz, zeros(ComplexF64, 48, 48), 40) static = false
        @assert_noalloc P._ctri2_unpacked!(false, false, 1.2 + 0.3im, Awz, Bwz, zeros(ComplexF64, 48, 48), 40) static = false
        # ztrsmR-C direct base (`_trsm_cmplx_dRC!`, the zpotrf-lower recursion path) — typestable + alloc-free.
        Atr = randn(ComplexF64, 48, 48) ./ 96; for d in 1:48; Atr[d, d] = 1 + abs(Atr[d, d]); end
        @assert_typestable P._trsm_cmplx_dRC!(true, false, 48, Atr, randn(ComplexF64, 64, 48))
        @assert_noalloc P._trsm_cmplx_dRC!(true, false, 48, Atr, randn(ComplexF64, 64, 48)) static = false
        @test true
    end
end

@testitem "StrictMode dogfood: GEMM hot paths" begin
    using StrictMode, AllocCheck, JET, TrimCheck  # TrimCheck → @assert_trim_compatible runs the
    # authoritative juliac verify_typeinf_trim here (test project is analysis="full"), not the heuristic.
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
            @assert_trim_compatible P._microkernel!(cp, ldc, ap, bp, kc, Val(P._MR), Val(P._NR))
            # StrictMode 0.3.9 @assert_no_spill: the µarch-derived _MR×_NR tile must fit the register file
            # with no vector spill/reload. Verified clean on both AVX-512 (Zen4/Zen5, 32 zmm) and AVX2 (Zen3,
            # 16 ymm) — the packed hot path. (NB the SMALL-matrix `_microkernel_unpacked!` spills 3 vectors on
            # AVX2 with the same tile — a real register-pressure finding, tracked separately; not asserted here.)
            @assert_no_spill P._microkernel!(cp, ldc, ap, bp, kc, Val(P._MR), Val(P._NR))
            @assert_noalloc P._microkernel_masked!(cp, ldc, ap, bp, kc, 11, 5, Val(P._MR), Val(P._NR))
            @assert_typestable P._microkernel_masked!(cp, ldc, ap, bp, kc, 11, 5, Val(P._MR), Val(P._NR))
            @assert_trim_compatible P._microkernel_masked!(cp, ldc, ap, bp, kc, 11, 5, Val(P._MR), Val(P._NR))
            # clip kernel: W-aligned partial row-tile (reads _MR-strided panel, computes 1 live vector)
            @assert_typestable P._microkernel_clip!(cp, ldc, ap, bp, kc, Val(P._MR), Val(1), Val(P._NR))
            @assert_noalloc P._microkernel_clip!(cp, ldc, ap, bp, kc, Val(P._MR), Val(1), Val(P._NR))
            @assert_trim_compatible P._microkernel_clip!(cp, ldc, ap, bp, kc, Val(P._MR), Val(1), Val(P._NR))
        end
        # unpacked microkernel (small-matrix path): A is mr×k, B is k×nr, column-major
        kk = 32; Au = randn(mr * kk); Bu = randn(kk * nr); Cu = zeros(mr, nr)
        GC.@preserve Au Bu Cu begin
            aup = pointer(Au); bup = pointer(Bu); cup = pointer(Cu)
            @assert_typestable P._microkernel_unpacked!(cup, mr, aup, mr, 0, bup, kk, 0, kk, 1.0, 0.0,
                Val(P._MR), Val(P._NR), Val(false), Val(true))
            @assert_noalloc P._microkernel_unpacked!(cup, mr, aup, mr, 0, bup, kk, 0, kk, 1.0, 2.0,
                Val(P._MR), Val(P._NR), Val(false), Val(false))
            @assert_trim_compatible P._microkernel_unpacked!(cup, mr, aup, mr, 0, bup, kk, 0, kk, 1.0, 0.0,
                Val(P._MR), Val(P._NR), Val(false), Val(true))
            # masked-row kernel (partial rows): mre=12 → second row-vector partially masked
            @assert_typestable P._microkernel_unpacked_mrows!(cup, mr, aup, mr, 0, bup, kk, 0, kk,
                1.0, 0.0, 12, Val(P._MR), Val(P._NR), Val(false), Val(true))
            @assert_noalloc P._microkernel_unpacked_mrows!(cup, mr, aup, mr, 0, bup, kk, 0, kk,
                1.0, 2.0, 12, Val(P._MR), Val(P._NR), Val(false), Val(false))
            @assert_trim_compatible P._microkernel_unpacked_mrows!(cup, mr, aup, mr, 0, bup, kk, 0, kk,
                1.0, 0.0, 12, Val(P._MR), Val(P._NR), Val(false), Val(true))
        end
        # COMPLEX unpacked path (`_gemm_cmplx_unpacked!` → `_uker_sweep!`): the exact class that regressed
        # zgemm_64_/cgemm_64_ trim-safety (four runtime `bool ? Val(true):Val(false)` flags → a Union{Val,Val}
        # split that exceeds juliac's reachability limit). `@assert_trim_compatible` in the test project's
        # :full mode (TrimCheck loaded) runs juliac's AUTHORITATIVE verify_typeinf_trim over this exact kernel
        # graph — VERIFIED to reproduce the pre-fix failure (4 verifier errors) when rooted here, so the class
        # is caught in the strict-verify pass at dev-time, not only at the ccallable in trim_tests.jl on CI.
        # NB the sibling `@assert_trim_safe` (heuristic TypeContracts scan) does NOT catch this reachability-
        # limit split — it's the known fast/full discrepancy: dev runs :fast (heuristic), tests run :full
        # (authoritative). trim_tests.jl stays as the ccallable-rooted belt (strict verify isn't perfect yet).
        for TC in (ComplexF64, ComplexF32)
            Az = randn(TC, 8, 8); Bz = randn(TC, 8, 8); Cz = zeros(TC, 8, 8)
            @assert_trim_compatible P._gemm_cmplx_unpacked!(Val(1), Val(1), false, 8, 8, 8, one(TC), Az, Bz, zero(TC), Cz)
            @assert_trim_compatible P._gemm_cmplx_unpacked!(Val(1), Val(-1), true, 8, 8, 8, TC(1.3, 0.7), Az, Bz, TC(0.9, -0.4), Cz)
        end
        # packing + generic path allocate nothing
        A = randn(8, 5); Bm = randn(5, 6); Cg = zeros(8, 6)
        @assert_typestable P._gemm_generic!(false, false, false, false, 8, 6, 5, 1.0, A, Bm, 0.0, Cg)
        @assert_noalloc P._gemm_generic!(false, false, false, false, 8, 6, 5, 1.0, A, Bm, 0.0, Cg)
        @test true
    end
end

@testitem "StrictMode dogfood: complex Cholesky base (zpotf2)" begin
    using StrictMode, AllocCheck, JET, LinearAlgebra
    if !StrictMode.checks_enabled()
        @test_skip StrictMode.checks_enabled()
    else
        P = PureBLAS
        # Vectorized Hermitian Cholesky base `_cpotf2_lower!` (the zpotrf n≤64 fix): the `cx` pointer-arith
        # closure + deinterleaved SIMD FMA must stay typestable + alloc-free on the Mode-2 native hot path.
        for TC in (ComplexF64, ComplexF32)
            A = randn(TC, 48, 48); A = A * A' + 48I + zeros(TC, 48, 48)
            @assert_typestable P._cpotf2_lower!(copy(A), 48)
            @assert_noalloc P._cpotf2_lower!(copy(A), 48)
            @assert_typestable P.potrf!(copy(A); uplo = 'L')          # n≤base → single base
            A2 = randn(TC, 128, 128); A2 = A2 * A2' + 128I + zeros(TC, 128, 128)
            @assert_typestable P.potrf!(copy(A2); uplo = 'L')         # n>base → recursive nb=n/4 blocked
            @assert_typestable P._cpotrf_lower!(copy(A2), 128)
            # complex getf2 panel (`_cgetf2_simd!`, zgetrf base) + QR panel (`qr_unblocked!`, zgeqrf) —
            # both vectorize via the L1 complex kernels; typestable + alloc-free on the native hot path.
            G = randn(TC, 48, 48) + 48I; ip = zeros(Int, 48); pG = pointer(G); ldG = stride(G, 2)
            GC.@preserve G begin
                @assert_typestable P._cgetf2_simd!(pG, ldG, 48, 48, 0, ip, 0)
                @assert_noalloc P._cgetf2_simd!(pG, ldG, 48, 48, 0, ip, 0)
            end
            Q = randn(TC, 48, 48); tau = similar(Q, 48)
            @assert_typestable P.qr_unblocked!(copy(Q), tau)
            @assert_noalloc P.qr_unblocked!(copy(Q), tau)
        end
        @test true
    end
end
