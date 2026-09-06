# Dogfooding StrictMode.jl — turns AllocCheck + JET + @inferred into declarable guarantees. We
# assert the hot-path kernels are type-stable, allocation-free, and trim-safe. Gated by a
# compile-time Preference (test/Project.toml ships them enabled); when disabled the macros are
# zero-cost no-ops, so we skip rather than pass vacuously. Mirrors PureFFT's strictmode dogfood.
#
# NOTE (StrictMode/StrictModeTest 0.4): `@test_noalloc` is ALWAYS AllocCheck's static all-paths
# proof now — 0.4 deleted the `analysis` preference and the `static = false` runtime form with it,
# and it deletes the kwarg SILENTLY (an unrecognized `k = v` is parsed as a positional argument and
# dropped), so a leftover `static = false` reads as an unchanged test while its meaning has flipped.
# Two properties, two tools, and they are not interchangeable:
#   * leaf kernels must be alloc-free on EVERY path       → `@test_noalloc` (the proof)
#   * drivers holding grow-once scratch are alloc-free in  → `@test (@allocated f(...)) == 0`, after
#     STEADY STATE, not on the first call                    a warm-up, as gemm_tests.jl does
# A driver cannot use the proof: it can statically reach `gemm!` → `_gemm_strassen!`, whose pad/level
# pool is a lazily sized `Vector{Matrix}` (workspace.jl `_str_fit!`), and an all-paths proof counts
# that branch even where it is runtime-dead. StrictMode 0.4's `register_alloc_barrier!` would exempt
# it, but it is a PROCESS-GLOBAL exemption and the pool re-allocates on shape change — so it would
# weaken every other `@test_noalloc` here to buy a claim that is not quite true.

@testitem "StrictMode dogfood: BLAS-1 strict contract" tags = [:checks] begin
    # StrictMode.TypeContracts: TypeContracts 0.14.0's @verify emits a `_seal_verified!(@__MODULE__,…)`
    # that resolves `TypeContracts` in THIS module (@verify_strict esc's the forwarded @verify call), so
    # the name must be in scope here. Reach it through StrictMode (already a dep) — no new test dep.
    using StrictModeTest, StrictMode, StrictMode.TypeContracts
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

@testitem "StrictMode dogfood: BLAS-2 strict contract" tags = [:checks] begin
    # StrictMode.TypeContracts: see the BLAS-1 item — @verify_strict's forwarded @verify (TypeContracts
    # 0.14.0) seals into this module, so `TypeContracts` must resolve here.
    using StrictModeTest, StrictMode, StrictMode.TypeContracts
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

@testitem "StrictMode dogfood: L3 trsm/syrk/symm scratch + driver" tags = [:checks] begin
    using StrictModeTest, StrictMode, LinearAlgebra
    if !StrictMode.checks_enabled()
        @info "StrictMode checks disabled — skipping L3 dogfood"
        @test_skip StrictMode.checks_enabled()
    else
        P = PureBLAS
        tri(s) = (
            M = tril(randn(s, s)); for i in 1:s
                M[i, i] += s
            end; M
        )
        # ROOT-CAUSE guard, on the CONSUMER (not the accessor — passing `Float64` as a value infers over
        # DataType, where T is unresolvable, a test artifact). Here T = eltype(B) is a concrete compile-
        # time parameter, so :full-mode JET report_opt sees the real specialization: the invL/invR base
        # holds the trtri+gemm scratch, and a non-concrete scratch return (view of an abstract
        # IdDict{DataType,Matrix} value) shows up as internal runtime dispatch / boxing here. This is the
        # check that was missing when the per-leaf boxing shipped — it passes only with concrete returns.
        Atri = tri(32)
        @assert_typestable P._trsm_base_invL!(false, false, false, Atri, randn(32, 256))
        @assert_typestable P._trsm_base_invR!(false, false, false, Atri, randn(256, 32))
        # Driver steady-state is allocation-free. Cached scratch allocates once on first touch, so what
        # is asserted is the empirical WARMED path — and from StrictMode 0.4 that has to be spelled with
        # runtime `@allocated`, not `@test_noalloc`. 0.4 dropped the `static = false` runtime form, so
        # `@test_noalloc` is now ONLY AllocCheck's static all-paths proof, and every driver below can
        # statically reach `gemm!` → `_gemm_strassen!`, whose pad/level pool is a lazily sized
        # `Vector{Matrix}` (workspace.jl `_str_fit!`). That branch is runtime-dead at these shapes but an
        # all-paths proof must still count it, so no warm-up can make the static form green. Same reason
        # gemm_tests.jl guards its drivers this way; the file header above says so.
        # StrictMode macros reject kwargs, so we assert on the POSITIONAL
        # internal drivers — which is exactly where the scratch/views live: the invL/invR bases hold
        # the trtri+gemm scratch (the boxing site), the packed syrk/syr2k/symm hold the pack buffers.
        # WARM the trsm_tmp scratch first: @assert_typestable above is JET-static (never executes), and
        # trsm_tmp is trsm-specific (unlike the syrk pack buffers, pre-grown by the earlier GEMM items in
        # this worker). Without this the noalloc target IS the first-touch grow → scheduling-flaky fail.
        # Operands are hoisted OUT of the measured expression: `@allocated` counts its whole argument
        # list, where the macro form bound each argument before measuring.
        BL = randn(32, 256); BR = randn(256, 32)
        P._trsm_base_invL!(false, false, false, Atri, BL)
        P._trsm_base_invR!(false, false, false, Atri, BR)
        @test (@allocated P._trsm_base_invL!(false, false, false, Atri, BL)) == 0
        @test (@allocated P._trsm_base_invR!(false, false, false, Atri, BR)) == 0
        # Fused gemmtrsm leaf (side-L upper, the wide-B gate shape f64) — typestable + alloc-free steady
        # state. Covers the transpose pack (shufflevector kernels) + the const-owned ftrsm buffer.
        Aup = (
            M = triu(randn(128, 128)); for i in 1:128
                M[i, i] += 128.0
            end; M
        )
        BF = randn(128, 256)
        P._trsm_fused_L!(false, Aup, BF)                                  # warm the ftrsm buffer
        @assert_typestable P._trsm_fused_L!(false, Aup, randn(128, 256))
        @test (@allocated P._trsm_fused_L!(false, Aup, BF)) == 0
        # Whole-k packed sweep (shared-panel restructure; default-off toggle) — same typestable + alloc-free
        # contract. Covers _pack_U_micro! + the packed slab/tail kernels reading the ftrsm buffer. AVX-512-f64
        # ONLY: `_trsm_fused_full_L!` is dispatched (trsm.jl:_trsm_left!) solely under `_GT_TRANSPOSE` and has
        # no non-transpose fallback (its slab kernels require W==MR==8), so a DIRECT call on AVX2 throws by
        # design — gate the dogfood on the same predicate the dispatcher uses.
        if PureBLAS._GT_TRANSPOSE
            Auf = (
                M = triu(randn(512, 512)); for i in 1:512
                    M[i, i] += 512.0
                end; M
            )
            BFF = randn(512, 256)
            P._trsm_fused_full_L!(false, Auf, BFF)                           # warm the ftrsm buffer
            @assert_typestable P._trsm_fused_full_L!(false, Auf, randn(512, 256))
            @test (@allocated P._trsm_fused_full_L!(false, Auf, BFF)) == 0
        end
        As = randn(512, 512); Bs = randn(512, 512); Cs = zeros(512, 512)
        P._syrk_blocked!(false, false, false, 0.8, As, Cs, 512)            # warm the pack buffers
        @test (@allocated P._syrk_blocked!(false, false, false, 0.8, As, Cs, 512)) == 0
        As32 = randn(32, 32); Cs32 = zeros(32, 32)   # small-n unified single-pack path (AVX2)
        P._syrk_blocked!(false, false, false, 0.8, As32, Cs32, 32)
        @test (@allocated P._syrk_blocked!(false, false, false, 0.8, As32, Cs32, 32)) == 0
        P._syr2k_packed!(false, false, 0.8, 0.3, As, Bs, Cs, 512)
        @test (@allocated P._syr2k_packed!(false, false, 0.8, 0.3, As, Bs, Cs, 512)) == 0
        P._symm!(true, false, false, 0.8, 0.3, As, Bs, Cs)
        @test (@allocated P._symm!(true, false, false, 0.8, 0.3, As, Bs, Cs)) == 0
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
        Cwz = zeros(ComplexF64, 48, 48)
        @assert_typestable P._ctri_unpacked!(true, true, 1.0, Awz, zeros(ComplexF64, 48, 48), 40)
        P._ctri_unpacked!(true, true, 1.0, Awz, Cwz, 40)
        @test (@allocated P._ctri_unpacked!(true, true, 1.0, Awz, Cwz, 40)) == 0
        P._ctri_unpacked!(false, false, 1.2 + 0.3im, Awz, Cwz, 40)
        @test (@allocated P._ctri_unpacked!(false, false, 1.2 + 0.3im, Awz, Cwz, 40)) == 0
        @assert_typestable P.herk!(zeros(ComplexF64, 48, 48), Awz; uplo = 'U', trans = 'N', alpha = 1.0, beta = 0.0)
        # rank-2k (two products through the shared _ctri_core!)
        @assert_typestable P._ctri2_unpacked!(true, true, 1.0, Awz, Bwz, zeros(ComplexF64, 48, 48), 40)
        P._ctri2_unpacked!(true, true, 1.0, Awz, Bwz, Cwz, 40)
        @test (@allocated P._ctri2_unpacked!(true, true, 1.0, Awz, Bwz, Cwz, 40)) == 0
        P._ctri2_unpacked!(false, false, 1.2 + 0.3im, Awz, Bwz, Cwz, 40)
        @test (@allocated P._ctri2_unpacked!(false, false, 1.2 + 0.3im, Awz, Bwz, Cwz, 40)) == 0
        # ztrsmR-C direct base (`_trsm_cmplx_dRC!`, the zpotrf-lower recursion path) — typestable + alloc-free.
        Atr = randn(ComplexF64, 48, 48) ./ 96; for d in 1:48
            Atr[d, d] = 1 + abs(Atr[d, d])
        end
        Btr = randn(ComplexF64, 64, 48)
        @assert_typestable P._trsm_cmplx_dRC!(true, false, 48, Atr, randn(ComplexF64, 64, 48))
        P._trsm_cmplx_dRC!(true, false, 48, Atr, Btr)
        @test (@allocated P._trsm_cmplx_dRC!(true, false, 48, Atr, Btr)) == 0
        @test true
    end
end

@testitem "StrictMode dogfood: GEMM hot paths" tags = [:checks] begin
    using StrictModeTest, StrictMode, TrimCheck  # @test_trim_compatible runs the authoritative
    # juliac verify_typeinf_trim (StrictModeTest depends on TrimCheck), not StrictMode's heuristic scan.
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
            @test_noalloc P._microkernel!(cp, ldc, ap, bp, kc, Val(P._MR), Val(P._NR))
            @test_trim_compatible P._microkernel!(cp, ldc, ap, bp, kc, Val(P._MR), Val(P._NR))
            # StrictMode 0.3.9 @assert_no_spill: the µarch-derived _MR×_NR tile must fit the register file
            # with no vector spill/reload. Verified clean on both AVX-512 (Zen4/Zen5, 32 zmm) and AVX2 (Zen3,
            # 16 ymm) — the packed hot path. (NB the SMALL-matrix `_microkernel_unpacked!` spills 3 vectors on
            # AVX2 with the same tile — a real register-pressure finding, tracked separately; not asserted here.)
            @assert_no_spill P._microkernel!(cp, ldc, ap, bp, kc, Val(P._MR), Val(P._NR))
            @test_noalloc P._microkernel_masked!(cp, ldc, ap, bp, kc, 11, 5, Val(P._MR), Val(P._NR))
            @assert_typestable P._microkernel_masked!(cp, ldc, ap, bp, kc, 11, 5, Val(P._MR), Val(P._NR))
            @test_trim_compatible P._microkernel_masked!(cp, ldc, ap, bp, kc, 11, 5, Val(P._MR), Val(P._NR))
            # clip kernel: W-aligned partial row-tile (reads _MR-strided panel, computes 1 live vector)
            @assert_typestable P._microkernel_clip!(cp, ldc, ap, bp, kc, Val(P._MR), Val(1), Val(P._NR))
            @test_noalloc P._microkernel_clip!(cp, ldc, ap, bp, kc, Val(P._MR), Val(1), Val(P._NR))
            @test_trim_compatible P._microkernel_clip!(cp, ldc, ap, bp, kc, Val(P._MR), Val(1), Val(P._NR))
        end
        # unpacked microkernel (small-matrix path): A is mr×k, B is k×nr, column-major
        kk = 32; Au = randn(mr * kk); Bu = randn(kk * nr); Cu = zeros(mr, nr)
        GC.@preserve Au Bu Cu begin
            aup = pointer(Au); bup = pointer(Bu); cup = pointer(Cu)
            @assert_typestable P._microkernel_unpacked!(
                cup, mr, aup, mr, 0, bup, kk, 0, kk, 1.0, 0.0,
                Val(P._MR), Val(P._NR), Val(false), Val(true)
            )
            @test_noalloc P._microkernel_unpacked!(
                cup, mr, aup, mr, 0, bup, kk, 0, kk, 1.0, 2.0,
                Val(P._MR), Val(P._NR), Val(false), Val(false)
            )
            @test_trim_compatible P._microkernel_unpacked!(
                cup, mr, aup, mr, 0, bup, kk, 0, kk, 1.0, 0.0,
                Val(P._MR), Val(P._NR), Val(false), Val(true)
            )
            # masked-row kernel (partial rows): mre=12 → second row-vector partially masked. VW is the
            # explicit vector-width Val (full register width here; the ≤W row tail passes the narrower
            # `_at_tail_vw` — that instance is trim-covered from the ccallable roots via `_mrows_tail!`).
            @assert_typestable P._microkernel_unpacked_mrows!(
                cup, mr, aup, mr, 0, bup, kk, 0, kk,
                1.0, 0.0, 12, Val(P._MR), Val(P._NR), Val(false), Val(true), Val(W)
            )
            @test_noalloc P._microkernel_unpacked_mrows!(
                cup, mr, aup, mr, 0, bup, kk, 0, kk,
                1.0, 2.0, 12, Val(P._MR), Val(P._NR), Val(false), Val(false), Val(W)
            )
            @test_trim_compatible P._microkernel_unpacked_mrows!(
                cup, mr, aup, mr, 0, bup, kk, 0, kk,
                1.0, 0.0, 12, Val(P._MR), Val(P._NR), Val(false), Val(true), Val(W)
            )
            # NARROW row tail. `_mrows_tail!` picks the width at COMPILE time from two detected consts,
            # so both arms must resolve statically; mre=2 is the width the shipped m=50 tail dispatches
            # on a double-pumped part. Asserting the dispatcher (not just the kernel) is the point — a
            # width choice that failed to const-fold would show up here as an instability, not as a
            # slightly slower benchmark.
            @assert_typestable P._mrows_tail!(
                cup, mr, aup, mr, 0, bup, kk, 0, kk,
                1.0, 0.0, 2, Val(P._NR), Val(false), Val(true)
            )
            @test_noalloc P._mrows_tail!(
                cup, mr, aup, mr, 0, bup, kk, 0, kk,
                1.0, 2.0, 2, Val(P._NR), Val(false), Val(false)
            )
            @test_trim_compatible P._mrows_tail!(
                cup, mr, aup, mr, 0, bup, kk, 0, kk,
                1.0, 0.0, 2, Val(P._NR), Val(false), Val(true)
            )
        end
        # COMPLEX unpacked path (`_gemm_cmplx_unpacked!` → `_uker_sweep!`): the exact class that regressed
        # zgemm_64_/cgemm_64_ trim-safety (four runtime `bool ? Val(true):Val(false)` flags → a Union{Val,Val}
        # split that exceeds juliac's reachability limit). `@test_trim_compatible` in the test project's
        # :full mode (TrimCheck loaded) runs juliac's AUTHORITATIVE verify_typeinf_trim over this exact kernel
        # graph — VERIFIED to reproduce the pre-fix failure (4 verifier errors) when rooted here, so the class
        # is caught in the strict-verify pass at dev-time, not only at the ccallable in trim_tests.jl on CI.
        # NB the sibling `@test_trim_compatible` (heuristic TypeContracts scan) does NOT catch this reachability-
        # limit split — it's the known fast/full discrepancy: dev runs :fast (heuristic), tests run :full
        # (authoritative). trim_tests.jl stays as the ccallable-rooted belt (strict verify isn't perfect yet).
        for TC in (ComplexF64, ComplexF32)
            Az = randn(TC, 8, 8); Bz = randn(TC, 8, 8); Cz = zeros(TC, 8, 8)
            @test_trim_compatible P._gemm_cmplx_unpacked!(Val(1), Val(1), false, 8, 8, 8, one(TC), Az, Bz, zero(TC), Cz)
            @test_trim_compatible P._gemm_cmplx_unpacked!(Val(1), Val(-1), true, 8, 8, 8, TC(1.3, 0.7), Az, Bz, TC(0.9, -0.4), Cz)
        end
        # packing + generic path allocate nothing
        A = randn(8, 5); Bm = randn(5, 6); Cg = zeros(8, 6)
        @assert_typestable P._gemm_generic!(false, false, false, false, 8, 6, 5, 1.0, A, Bm, 0.0, Cg)
        @test_noalloc P._gemm_generic!(false, false, false, false, 8, 6, 5, 1.0, A, Bm, 0.0, Cg)
        @test true
    end
end

@testitem "StrictMode dogfood: complex Cholesky base (zpotf2)" tags = [:checks] begin
    using StrictModeTest, StrictMode, LinearAlgebra
    if !StrictMode.checks_enabled()
        @test_skip StrictMode.checks_enabled()
    else
        P = PureBLAS
        # Vectorized Hermitian Cholesky base `_cpotf2_lower!` (the zpotrf n≤64 fix): the `cx` pointer-arith
        # closure + deinterleaved SIMD FMA must stay typestable + alloc-free on the Mode-2 native hot path.
        for TC in (ComplexF64, ComplexF32)
            A = randn(TC, 48, 48); A = A * A' + 48I + zeros(TC, 48, 48)
            @assert_typestable P._cpotf2_lower!(copy(A), 48)
            @test_noalloc P._cpotf2_lower!(copy(A), 48)
            @assert_typestable P.potrf!(copy(A); uplo = 'L')          # n≤base → single base
            A2 = randn(TC, 128, 128); A2 = A2 * A2' + 128I + zeros(TC, 128, 128)
            @assert_typestable P.potrf!(copy(A2); uplo = 'L')         # n>base → recursive nb=n/4 blocked
            @assert_typestable P._cpotrf_lower!(copy(A2), 128)
            # UPPER was covered by NEITHER of the above: it is Lever C (conj-transpose → `_potrf_pad`
            # scratch → `_cpotrf_lower!` → conj-transpose back), a different call graph that reaches
            # `_tri_upper_to_lowerT!`/`_tri_lowerT_to_upper!` and `_potf2_needs_buf` — none of which any
            # 'L' assertion roots. Both sizes: n≤base is the single-base window where `_potf2_needs_buf`
            # decides, n>base takes the recursion.
            @assert_typestable P.potrf!(copy(A); uplo = 'U')
            @assert_typestable P.potrf!(copy(A2); uplo = 'U')
            # complex getf2 panel (`_cgetf2_simd!`, zgetrf base) + QR panel (`qr_unblocked!`, zgeqrf) —
            # both vectorize via the L1 complex kernels; typestable + alloc-free on the native hot path.
            G = randn(TC, 48, 48) + 48I; ip = zeros(Int, 48); pG = pointer(G); ldG = stride(G, 2)
            GC.@preserve G begin
                @assert_typestable P._cgetf2_simd!(pG, ldG, 48, 48, 0, ip, 0)
                @test_noalloc P._cgetf2_simd!(pG, ldG, 48, 48, 0, ip, 0)
            end
            Q = randn(TC, 48, 48); tau = similar(Q, 48)
            @assert_typestable P.qr_unblocked!(copy(Q), tau)
            @test_noalloc P.qr_unblocked!(copy(Q), tau)
        end
        @test true
    end
end

# Eigen family — the contract members that were DECLARED but never proved anywhere.
#
# `gebal!`, `gehrd!`, `orghr!`, `ormhr!`, `hseqr!` and `geev!` are all @strict_contract members
# (src/contracts.jl:368-380) and src/verify.jl carries purpose-built probes for every one of them
# (`_strict_gehrd_probe` &c, ~20 call sites). None of it has ever run. Those call sites live inside the
# `if StrictMode.proofs_loaded()` block, and a package cannot have the proving tier loaded during its OWN
# precompile — that is the one moment src/verify.jl executes. The block was equally dead under 0.3.10,
# where the guard asked `backend_available()` about a weak-dep extension that is likewise never loaded
# there. So the tiers are: src/ REPORTS (the value-free IR scan, all StrictMode has available to it), and
# test/ PROVES — and the proving half simply had no home for this family until now.
#
# `test_signatures` (StrictModeTest 0.4) is the right tool rather than a per-call macro, for two reasons
# beyond not having to build valid eigen inputs. It takes SIGNATURES, so `Char` is a TYPE here: one entry
# analyses `geev!` for every jobvl/jobvr combination at once, where the src probes pinned 'N','V' and
# never covered the 'N','N' arm that bench/plots.jl:1003 actually benchmarks. And it costs nothing to
# list a large shape, so these are the n the gate cares about rather than an n=24 toy.
#
# GUARANTEE CHOICE, deliberate: `:typestable` only. NOT `:noalloc` — `gehrd!`'s blocked rewrite (the
# "flagged, not built" follow-up in hessenberg.jl:248, and the whole of geev's 0.039-0.076 gate gap) will
# route its trailing update through `gemm!`, and nothing reaching `gemm!` can pass a static all-paths
# noalloc proof: `_gemm_strassen!`'s pad pool is a lazily sized `Vector{Matrix}` that the proof counts on
# every path, runtime-dead or not. Gating :noalloc here would red the moment the gap is fixed and the
# pressure would be to weaken it. Steady-state allocation is asserted separately below, warmed, which is
# the property that actually holds and survives blocking.
@testitem "StrictMode dogfood: eigen family (gebal/gehrd/orghr/ormhr/hseqr/geev)" tags = [:checks] begin
    using StrictModeTest, StrictMode, LinearAlgebra
    if !StrictMode.checks_enabled()
        @info "StrictMode checks disabled — skipping eigen dogfood"
        @test_skip StrictMode.checks_enabled()
    else
        P = PureBLAS
        bk = P.DEFAULT_BACKEND
        SB = typeof(bk)
        MD = Matrix{Float64}; VD = Vector{Float64}
        MZ = Matrix{ComplexF64}; VZ = Vector{ComplexF64}
        test_signatures(
            [
                # balance / un-balance
                (P.gebal!, (SB, MD, VD)),
                (P.gebal!, (SB, MZ, VD)),
                (P.gebak!, (SB, Char, Char, Int, Int, VD, MD)),
                (P.gebak!, (SB, Char, Char, Int, Int, VD, MZ)),
                # Hessenberg reduction and its Q — the stage that owns 69% of the geev cell
                (P.gehrd!, (SB, MD, Int, Int, VD)),
                (P.gehrd!, (SB, MZ, Int, Int, VZ)),
                (P.orghr!, (SB, MD, Int, Int, VD)),
                (P.orghr!, (SB, MZ, Int, Int, VZ)),
                (P.ormhr!, (SB, Char, Char, Int, Int, MD, VD, MD)),
                (P.ormhr!, (SB, Char, Char, Int, Int, MZ, VZ, MZ)),
                # Francis QR — both job/compz arms, since Char is a type here
                (P.hseqr!, (SB, Char, Char, MD, Int, Int, VZ, MD)),
                (P.hseqr!, (SB, Char, Char, MZ, Int, Int, VZ, MZ)),
                # the drivers (real arity carries wr/wi separately, complex a single w)
                (P.geev!, (SB, Char, Char, MD, VD, VD, MD, MD, VD)),
                (P.geev!, (SB, Char, Char, MZ, VZ, MZ, MZ, VD)),
            ];
            guarantees = (:typestable,)
        )
        # Steady state, warmed — the counterpart the static proof cannot express (see the note above and
        # the file header). n is small because this asserts a property, not a rate.
        #
        # MEASURED INSIDE A FUNCTION, AND THROUGH THE CONST BINDING — neither is stylistic, and both were
        # established by measurement here rather than assumed. A @testitem body is module scope, so its
        # `bk`/`A`/`sc` are untyped globals and the call is dynamically dispatched, which BOXES a non-heap
        # return value: `gebal!` returns `Tuple{Int64,Int64}`, and measured at item scope that is a steady
        # 32 B (2×8 + header, padded) the kernel never allocates. A function wrapper alone does not fix it
        # either, because `P = PureBLAS` is a non-const global, so `P.gebal!` inside the function is still
        # a dynamic lookup — it has to be the const `PureBLAS` binding. Inside a function and through that
        # binding it is 0 B on every job arm at both n (bench/probes/gebal_alloc.jl).
        # None of this shows up in gemm_tests.jl only because those kernels return their `C` argument,
        # which is already on the heap; any routine returning a tuple or a scalar needs this shape or the
        # check measures Julia's dispatch rather than the kernel.
        function _eigen_steady(bk)
            n = 24
            A0 = randn(n, n); A = copy(A0); sc = ones(n); tau = Vector{Float64}(undef, n)
            PureBLAS.gebal!(bk, A, sc)                             # warm
            copyto!(A, A0); a_gebal = @allocated PureBLAS.gebal!(bk, A, sc)
            copyto!(A, A0); PureBLAS.gehrd!(bk, A, 1, n, tau)      # warm the arena high-water
            copyto!(A, A0); a_gehrd = @allocated PureBLAS.gehrd!(bk, A, 1, n, tau)
            return (a_gebal, a_gehrd)
        end
        @test _eigen_steady(bk) == (0, 0)
        @test true
    end
end

# LAPACK finish-all surface — trim-compatibility dogfood. In the :checks (analysis="full") project with
# TrimCheck loaded, @test_trim_compatible runs juliac's AUTHORITATIVE verify_typeinf_trim (the same
# verifier as juliac/build.jl). This is the dev-time net that was MISSING when the trsyl `scale` Core.Box
# and the qr.jl complex-muladd / svd.jl permutedims union-splits shipped a non-building .so with CI green.
# REQUIREMENT (memory lapack-strict-contract-required): a new LAPACK @ccallable entry point is not done
# until asserted here. @test_trim_compatible EXECUTES the call, so inputs must be runtime-valid (SPD /
# pre-factored where the kernel demands it). Covers the full finish-all surface, real + complex.
@testitem "StrictMode dogfood: LAPACK finish-all trim-compatibility" tags = [:checks] begin
    using StrictModeTest, StrictMode, TrimCheck, LinearAlgebra
    if !StrictMode.checks_enabled()
        @info "StrictMode checks disabled — skipping LAPACK finish-all trim dogfood"
        @test_skip StrictMode.checks_enabled()
    else
        P = PureBLAS
        n = 24; k = 12
        # symmetric / Hermitian sources; SPD variants for the Cholesky-family kernels
        Ad = (M = randn(n, n); M + transpose(M)); Az = (M = randn(ComplexF64, n, n); M + transpose(M))
        Ahe = (M = randn(ComplexF64, n, n); M + M')
        Sd = (M = randn(n, n); M * transpose(M) + n * I)            # real SPD
        She = (M = randn(ComplexF64, n, n); M * M' + n * I)         # Hermitian PD

        # ── symmetric-indefinite solve/inverse (sytri/hetri need a real factorization first) ──
        @test_trim_compatible P.sysv!('L', copy(Ad), randn(n, 2))
        @test_trim_compatible P.sysv!('L', copy(Az), randn(ComplexF64, n, 2))
        @test_trim_compatible P.hesv!('L', copy(Ahe), randn(ComplexF64, n, 2))
        LDd = copy(Ad); ipd = zeros(Int, n); P.sytrf!(LDd, ipd; uplo = 'L')
        LDz = copy(Az); ipz = zeros(Int, n); P.sytrf!(LDz, ipz; uplo = 'L')
        LDh = copy(Ahe); iph = zeros(Int, n); P.hetrf!(LDh, iph; uplo = 'L')
        # sytrf!/hetrf! were only ever covered TRANSITIVELY (through sysv!, and above only as input
# setup). Assert them directly, and at a size that reaches the BLOCKED panel — nb is
# 8*ceil(isqrt(n)/8) clamped [16,96], so any n > 16 does, but 96 exercises several panels
# plus the unblocked tail. Both uplo, and all three of symmetric / complex-symmetric /
# Hermitian, since each takes a different kernel (_lasyf_* vs _lahef_*).
nbig = 96
Abd = (M = randn(nbig, nbig); M + transpose(M))
Abz = (M = randn(ComplexF64, nbig, nbig); M + transpose(M))
Abh = (M = randn(ComplexF64, nbig, nbig); M + M')
ipb = zeros(Int, nbig)
P.sytrf!(copy(Abd), ipb; uplo = 'L')                    # warm the owned W workspace first
P.hetrf!(copy(Abh), ipb; uplo = 'L')
for ul in ('L', 'U')
    @test_trim_compatible P.sytrf!(copy(Abd), zeros(Int, nbig); uplo = ul)
    @test_trim_compatible P.sytrf!(copy(Abz), zeros(Int, nbig); uplo = ul)
    @test_trim_compatible P.hetrf!(copy(Abh), zeros(Int, nbig); uplo = ul)
end
@test_trim_compatible P.sytri!(copy(LDd), ipd; uplo = 'L')
        @test_trim_compatible P.sytri!(copy(LDz), ipz; uplo = 'L')
        @test_trim_compatible P.hetri!(copy(LDh), iph; uplo = 'L')
        # ── QL / RQ (geqlf/gerqf + org/orm), real + complex ──
        Aqd = randn(n, k); tqd = zeros(Float64, k); Aqz = randn(ComplexF64, n, k); tqz = zeros(ComplexF64, k)
        Fqd = copy(Aqd); P.geqlf!(Fqd, tqd); Fqz = copy(Aqz); P.geqlf!(Fqz, tqz)
        @test_trim_compatible P.geqlf!(copy(Aqd), zeros(Float64, k))
        @test_trim_compatible P.orgql!(copy(Fqd), copy(tqd))
        @test_trim_compatible P.ormql!('L', 'N', copy(Fqd), tqd, randn(n, 3))
        @test_trim_compatible P.geqlf!(copy(Aqz), zeros(ComplexF64, k))
        @test_trim_compatible P.orgql!(copy(Fqz), copy(tqz))
        @test_trim_compatible P.ormql!('L', 'N', copy(Fqz), tqz, randn(ComplexF64, n, 3))
        Ard = randn(k, n); Arz = randn(ComplexF64, k, n)
        Grd = copy(Ard); P.gerqf!(Grd, tqd); Grz = copy(Arz); P.gerqf!(Grz, tqz)
        @test_trim_compatible P.gerqf!(copy(Ard), zeros(Float64, k))
        @test_trim_compatible P.orgrq!(copy(Grd), copy(tqd))
        @test_trim_compatible P.ormrq!('R', 'N', copy(Grd), tqd, randn(3, n))
        @test_trim_compatible P.gerqf!(copy(Arz), zeros(ComplexF64, k))
        @test_trim_compatible P.orgrq!(copy(Grz), copy(tqz))
        # ── RZ (tzrzf/ormrz) ──
        Tzd = randn(k, n); ttd = zeros(Float64, k); P.tzrzf!(Tzd, ttd)
        @test_trim_compatible P.tzrzf!(randn(k, n), zeros(Float64, k))
        @test_trim_compatible P.tzrzf!(randn(ComplexF64, k, n), zeros(ComplexF64, k))
        @test_trim_compatible P.ormrz!('L', 'N', copy(Tzd), ttd, randn(n, 3))
        # ── pivoted Cholesky (pstrf) — SPD input ──
        @test_trim_compatible P.pstrf!(copy(Sd), zeros(Int, n), -1.0; uplo = 'L')
        @test_trim_compatible P.pstrf!(copy(She), zeros(Int, n), -1.0; uplo = 'L')
        # ── rank-deficient / constrained least squares ──
        @test_trim_compatible P.gelsd!(randn(n, k), randn(n, 2), -1.0)
        @test_trim_compatible P.gelsd!(randn(ComplexF64, n, k), randn(ComplexF64, n, 2), -1.0)
        @test_trim_compatible P.gelsy!(randn(n, k), randn(n, 2), zeros(Int, k), -1.0)
        @test_trim_compatible P.gelsy!(randn(ComplexF64, n, k), randn(ComplexF64, n, 2), zeros(Int, k), -1.0)
        @test_trim_compatible P.gglse!(randn(8, 6), randn(8), randn(4, 6), randn(4))
        @test_trim_compatible P.gglse!(randn(ComplexF64, 8, 6), randn(ComplexF64, 8), randn(ComplexF64, 4, 6), randn(ComplexF64, 4))
        # ── Sylvester (trsyl) + Schur reorder (trexc/trsen) ──
        @test_trim_compatible P.trsyl!('N', 'N', 1, triu(randn(n, n)) + n * I, triu(randn(n, n)) + n * I, randn(n, n))
        @test_trim_compatible P.trsyl!('N', 'N', 1, triu(randn(ComplexF64, n, n)) + n * I, triu(randn(ComplexF64, n, n)) + n * I, randn(ComplexF64, n, n))
        @test_trim_compatible P.trexc!('V', triu(randn(n, n)) + n * I, Matrix(1.0I, n, n), 2, n - 1)
        @test_trim_compatible P.trexc!('V', triu(randn(ComplexF64, n, n)) + n * I, Matrix(ComplexF64(1)I, n, n), 2, n - 1)
        @test_trim_compatible P.trsen!('N', 'V', rand(Bool, n), triu(randn(n, n)) + n * I, Matrix(1.0I, n, n))
        @test_trim_compatible P.trsen!('N', 'V', rand(Bool, n), triu(randn(ComplexF64, n, n)) + n * I, Matrix(ComplexF64(1)I, n, n))
        # ── generalized eigen (ggev/gges/sygvd/hegvd) — sygvd/hegvd need PD B ──
        @test_trim_compatible P.ggev!('N', 'V', randn(n, n), randn(n, n))
        @test_trim_compatible P.ggev!('N', 'V', randn(ComplexF64, n, n), randn(ComplexF64, n, n))
        @test_trim_compatible P.gges!('V', 'V', randn(n, n), randn(n, n))
        @test_trim_compatible P.gges!('V', 'V', randn(ComplexF64, n, n), randn(ComplexF64, n, n))
        @test_trim_compatible P.ggsvd!('U', 'V', 'Q', randn(8, 6), randn(6, 6))
        @test_trim_compatible P.ggsvd!('U', 'V', 'Q', randn(ComplexF64, 8, 6), randn(ComplexF64, 6, 6))
        @test_trim_compatible P.sygvd!(1, 'V', 'U', copy(Ad), copy(Sd))
        @test_trim_compatible P.hegvd!(1, 'V', 'U', copy(Ahe), copy(She))
        # ── symmetric-tridiagonal (stebz/stein), real ──
        dd = randn(n); ee = randn(n - 1)
        @test_trim_compatible P.stebz!('A', 'B', 0.0, 0.0, 1, n, 0.0, copy(dd), copy(ee))
        w, ib, isp, _ = P.stebz!('A', 'B', 0.0, 0.0, 1, n, 0.0, copy(dd), copy(ee))
        @test_trim_compatible P.stein!(copy(dd), copy(ee), w, ib, isp)
        # ── generalized SVD (ggsvd), Float64 full-rank ──
        # ── banded LU (gbtrf/gbtrs) — factor then solve ──
        ABd2 = randn(6, n); _, gip, _ = P.gbtrf!(2, 1, n, ABd2)
        @test_trim_compatible P.gbtrf!(2, 1, n, randn(6, n))
        @test_trim_compatible P.gbtrs!('N', 2, 1, n, copy(ABd2), gip, randn(n, 2))
        @test_trim_compatible P.gbtrf!(2, 1, n, randn(ComplexF64, 6, n))
        # BLOCKED path. The cells above are kl=2, so they can only ever reach `_gbtf2!` — a Base-only
        # scalar kernel — and the blocked port would ship trim-unvalidated, which is exactly the
        # failure the pbtrf item at test/lapack_tests.jl:24 is a post-mortem of. kl ≥ 2·nb is the
        # dispatch gate; kl=32 clears it for every measured nb. This puts trsm!, _gemm_core!,
        # PtrMatrix and the L3Workspace scratch inside gbtrf's trim call graph for the first time.
        ldgb = 2 * 32 + 8 + 1
        P.gbtrf!(32, 8, 64, randn(ldgb, 64))                 # warm the owned workspace first
        P.gbtrf!(32, 8, 64, randn(ComplexF64, ldgb, 64))
        @test_trim_compatible P.gbtrf!(32, 8, 64, randn(ldgb, 64))
        @test_trim_compatible P.gbtrf!(32, 8, 64, randn(ComplexF64, ldgb, 64))
        # ── SPD tridiagonal (pttrf/pttrs/ptsv) — factor then solve ──
        Dp = fill(4.0, n); Ep = fill(1.0, n - 1); Df = copy(Dp); Ef = copy(Ep); P.pttrf!(Df, Ef)
        @test_trim_compatible P.pttrf!(fill(4.0, n), fill(1.0, n - 1))
        @test_trim_compatible P.pttrs!(copy(Df), copy(Ef), randn(n, 2))
        @test_trim_compatible P.ptsv!(fill(4.0, n), fill(1.0, n - 1), randn(n, 2))
        Dpz = fill(4.0, n); Epz = fill(1.0 + 0im, n - 1)
        @test_trim_compatible P.ptsv!(copy(Dpz), copy(Epz), randn(ComplexF64, n, 2))
        # ── general tridiagonal (gtsv/gttrf/gttrs) — factor then solve ──
        dl = fill(1.0, n - 1); dm = fill(4.0, n); du = fill(1.0, n - 1)
        du2 = zeros(Float64, n - 2); gtip = zeros(Int, n); P.gttrf!(copy(dl), copy(dm), copy(du), du2, gtip)
        @test_trim_compatible P.gtsv!(copy(dl), copy(dm), copy(du), randn(n, 2))
        @test_trim_compatible P.gttrf!(copy(dl), copy(dm), copy(du), zeros(Float64, n - 2), zeros(Int, n))
        @test_trim_compatible P.gtsv!(fill(1.0 + 0im, n - 1), fill(4.0 + 0im, n), fill(1.0 + 0im, n - 1), randn(ComplexF64, n, 2))
        # ── banded / packed Cholesky (pbtrf/pbtrs, pptrf/pptrs) — SPD storage ──
        mkband(T) = (AB = zeros(T, 3, n); AB[1, :] .= T(10); AB[2, 1:(n - 1)] .= T(1); AB[3, 1:(n - 2)] .= T(0.5); AB)
        ABsd = mkband(Float64); P.pbtrf!(ABsd; uplo = 'L', kd = 2)
        @test_trim_compatible P.pbtrf!(mkband(Float64); uplo = 'L', kd = 2)
        @test_trim_compatible P.pbtrs!(copy(ABsd), randn(n, 2); uplo = 'L', kd = 2)
        @test_trim_compatible P.pbtrf!(mkband(ComplexF64); uplo = 'L', kd = 2)
        # kd=2 only reaches the UNBLOCKED kernel. The blocked kernels — the lower one, and the two
        # the 'U' path dispatches between — carry all the PtrMatrix/ld-1 band views and the tuned L3
        # calls, i.e. everything trim actually has to chew on, and none of it was covered here.
        # Both upper kernels are invoked DIRECTLY: which one pbtrf! selects depends on a per-host
        # measurement (_pbtrf_ucross), so going through the public entry would leave one of them
        # untested on any given box.
        wideband(T, kd) = (
            AB = zeros(T, kd + 1, n); AB[1, :] .= T(4 * kd);
            for d in 1:kd
                AB[1 + d, 1:(n - d)] .= T(0.5)
            end; AB
        )
        wideU(T, kd) = (
            AB = zeros(T, kd + 1, n); AB[kd + 1, :] .= T(4 * kd);
            for d in 1:kd
                AB[kd + 1 - d, (1 + d):n] .= T(0.5)
            end; AB
        )
        @test_trim_compatible P.pbtrf!(wideband(Float64, 64); uplo = 'L', kd = 64)
        @test_trim_compatible P.pbtrf!(wideU(Float64, 64); uplo = 'U', kd = 64)
        @test_trim_compatible P._pbtrf_blocked!(wideband(Float64, 64), n, 64)
        @test_trim_compatible P._pbtrf_repack_U!(wideU(Float64, 64), n, 64)
        @test_trim_compatible P._pbtrf_blocked_U!(wideU(Float64, 64), n, 64)
        @test_trim_compatible P._pbtrf_blocked!(wideband(ComplexF64, 32), n, 32)
        @test_trim_compatible P._pbtrf_repack_U!(wideU(ComplexF64, 32), n, 32)
        @test_trim_compatible P._pbtrf_blocked_U!(wideU(ComplexF64, 32), n, 32)
        # The corner work array W and the diagonal-block scratch S used to be allocated per CALL;
        # they are now GKH-owned (L3Workspace, via _pbtrf_work). Assert the steady state is actually
        # allocation-free — that IS the point of the ownership, and a per-call `Matrix` would sail
        # through every correctness test. Runtime `@allocated`, not `@test_noalloc`: the FIRST call
        # legitimately sizes the owned buffers, and StrictMode 0.4 dropped the `static = false` runtime
        # form, leaving `@test_noalloc` as a static all-paths proof that must count the (runtime-dead)
        # `_gemm_strassen!` pad pool — see the L3 item above for the full reasoning.
        P._pbtrf_blocked!(wideband(Float64, 64), n, 64)          # size the owned scratch for nb(64)
        Wpb = wideband(Float64, 64)
        @test (@allocated P._pbtrf_blocked!(Wpb, n, 64)) == 0
        P._pbtrf_blocked_U!(wideU(Float64, 64), n, 64)
        WpbU = wideU(Float64, 64)
        @test (@allocated P._pbtrf_blocked_U!(WpbU, n, 64)) == 0
        pack(M) = [M[i, j] for j in 1:n for i in j:n]              # lower column-packed
        APsd = pack(Sd); P.pptrf!(APsd; uplo = 'L')
        @test_trim_compatible P.pptrf!(pack(Sd); uplo = 'L')
        @test_trim_compatible P.pptrs!(copy(APsd), randn(n, 2); uplo = 'L')
        @test_trim_compatible P.pptrf!(pack(She); uplo = 'L')
        @test true
    end
end
