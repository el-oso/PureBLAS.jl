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
#   * drivers holding grow-once scratch are alloc-free in  → a warmed `@allocated` reading, inside a
#     STEADY STATE, not on the first call                    function over typed operands
# A pool that grows to a new high-water mark allocates again in the same process, thread and task, so
# it is not a memoization scope and `register_alloc_barrier!` is not legitimate for it: those growth
# functions (`_ws_grow!`, `_ws_slot!`, `_arena_grow!`, `_arena_coalesce!`, the LAPACK `_*_grow!`) are
# `@strict_exempt :noalloc`, and a driver that reaches one cannot take the all-paths proof. What it can
# take is the measurement: warm the call, then a second identical call must allocate 0 bytes. That is
# the "pool growth is bounded" item near the end of this file. The only registered barriers are the
# first-touch accessors of a per-task owner (`_m3ws`, `_gpackws`, `_strws`, `_opt`), whose allocation
# is the one-time initialization of the task's slot. Registration is session-wide, so each item
# unregisters afterwards; left in place it would weaken every other `@test_noalloc` in the same worker.

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
        # THE PROOF WHERE IT HOLDS, THE MEASUREMENT WHERE A DRIVER REACHES A GROWTH FUNCTION. A pool that
        # grows to a new high-water mark is not a one-time initialization, so its growth functions
        # (`workspace.jl`) are `@strict_exempt :noalloc` rather than registered barriers, and a driver
        # that can reach one is asserted by a warmed `@allocated` reading instead — see the "pool growth
        # is bounded" item at the end of this file.
        #
        # What is registered: the first-touch accessors of the per-task owners (`_m3ws`, `_gpackws`,
        # `_strws`, `_opt`). `Base.OncePerTask` is deliberately NOT auto-recognized as a barrier. The
        # registration is process-global, so it is torn down in `finally` and the verdict cache cleared.
        _POOLS = (P._m3ws, P._gpackws, P._strws, P._opt)
        for f in _POOLS
            StrictMode.register_alloc_barrier!(f)
        end
        try
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
            @test_typestable P._trsm_base_invL!(false, false, false, Atri, randn(32, 256))
            @test_typestable P._trsm_base_invR!(false, false, false, Atri, randn(256, 32))
            # Driver steady-state is allocation-free. Cached scratch allocates once on first touch, so what
            # is asserted is the empirical WARMED path — and from StrictMode 0.4 that has to be spelled with
            # `@test_noalloc` is ONLY AllocCheck's static all-paths proof since StrictModeTest 0.4, which
            # dropped the `static = false` runtime form. The drivers below reach `gemm!` →
            # `_gemm_strassen!`, whose pad/level pool is a lazily sized `Vector{Matrix}` (workspace.jl
            # `_str_fit!`), plus the arena and the pack pools — all grow-only, all behind the registered
            # barriers above, so the proof holds on them. The warm-up calls that remain are for the few
            # assertions still taking a runtime reading, not for the proofs, which never execute.
            # StrictMode macros reject kwargs, so we assert on the POSITIONAL
            # internal drivers — which is exactly where the scratch/views live: the invL/invR bases hold
            # the trtri+gemm scratch (the boxing site), the packed syrk/syr2k/symm hold the pack buffers.
            # WARM the trsm_tmp scratch on the measured operands first: trsm_tmp is trsm-specific (unlike the syrk pack buffers, pre-grown by the earlier GEMM items in
            # this worker). Without this the noalloc target IS the first-touch grow → scheduling-flaky fail.
            BL = randn(32, 256); BR = randn(256, 32)
            P._trsm_base_invL!(false, false, false, Atri, BL)
            P._trsm_base_invR!(false, false, false, Atri, BR)
            # These reach pool growth functions (`@strict_exempt :noalloc`), so they take the warmed
            # measurement; the typed wrappers keep `@allocated` inside a compiled function.
            a_invL(A, B) = @allocated PureBLAS._trsm_base_invL!(false, false, false, A, B)
            a_invR(A, B) = @allocated PureBLAS._trsm_base_invR!(false, false, false, A, B)
            @test minimum(a_invL(Atri, BL) for _ in 1:8) == 0
            @test minimum(a_invR(Atri, BR) for _ in 1:8) == 0
            # BROKEN: the all-paths proof on the same two drivers. Each assertion still runs, so it
            # reports an unexpected pass once it holds.
            @test_broken (
                try
                    @test_noalloc P._trsm_base_invL!(false, false, false, Atri, BL)
                    true
                catch e
                    e isa StrictMode.StrictViolation || rethrow()
                    false
                end
            )
            @test_broken (
                try
                    @test_noalloc P._trsm_base_invR!(false, false, false, Atri, BR)
                    true
                catch e
                    e isa StrictMode.StrictViolation || rethrow()
                    false
                end
            )
            # Fused gemmtrsm leaf (side-L upper, the wide-B gate shape f64) — typestable + alloc-free steady
            # state. Covers the transpose pack (shufflevector kernels) + the const-owned ftrsm buffer.
            Aup = (
                M = triu(randn(128, 128)); for i in 1:128
                    M[i, i] += 128.0
                end; M
            )
            BF = randn(128, 256)
            P._trsm_fused_L!(false, Aup, BF)                                  # warm the ftrsm buffer
            @test_typestable P._trsm_fused_L!(false, Aup, randn(128, 256))
            a_fusedL(A, B) = @allocated PureBLAS._trsm_fused_L!(false, A, B)
            @test minimum(a_fusedL(Aup, BF) for _ in 1:8) == 0
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
                @test_typestable P._trsm_fused_full_L!(false, Auf, randn(512, 256))
                a_fusedFull(A, B) = @allocated PureBLAS._trsm_fused_full_L!(false, A, B)
                @test minimum(a_fusedFull(Auf, BFF) for _ in 1:8) == 0
            end
            As = randn(512, 512); Bs = randn(512, 512); Cs = zeros(512, 512)
            # n=512 now takes the recursive route into gemm rather than the packed kernel, so it reaches
            # the SME path and its function pointer -- hence the barrier here too, same as below.
            # KNOWN GAP, ASSERTED AS A GAP so it cannot be forgotten. `@test_noalloc` is AllocCheck's
            # all-paths proof and these drivers do not pass it — measured on StrictMode/StrictModeTest
            # 0.4.11 with AllocCheck 0.2.7, 103 sites for `_syrk_blocked!`, 131 for complex
            # `qr_unblocked!`, 139 for `gehrd!`. Attributing every site to its PureBLAS frame shows ZERO
            # task or scheduler frames: they are one-time-init workspace acquisition — `OncePerThread`'s
            # lock and memo, the `Arena` constructor reached through it, `_gpackws`, `_ws_slot!`, `_l3ws`,
            # `_syr2k_scratch`, the `L3Workspace` constructor. Registering those as barriers does NOT
            # help, because `OncePerThread` already sets `barrier = true` intrinsically and the exemption
            # additionally requires `!alloc`, which is what fails.
            #
            # An all-paths prover states PATH properties and the guarantee here is TEMPORAL ("from the
            # SECOND call", req#10 / docs/src/arena.md), so the cold growth branch is reported however
            # unreachable it is at runtime.
            #
            # `@test_throws` rather than a deleted assertion: this FAILS LOUDLY the day the owners move
            # off `OncePerThread` and the pools stop acquiring inside the call graph, which is when these
            # three become plain `@test_noalloc` lines. A deleted assertion would go quiet instead.
            #
            # BOTH INSTRUMENTS, BECAUSE THEY FAIL ON OPPOSITE CHANGES AND NEITHER COVERS THE OTHER.
            # `@test_throws @test_noalloc` passes BECAUSE some path allocates, so a new per-call
            # `Matrix{T}` in the driver keeps it green — and a `!` entry allocating at steady state is
            # precisely the defect req#10 names. The warm `@allocated` below is what goes red on that,
            # and req#10's own wording prescribes it ("measure the entry, warm, wrapped in a function
            # over pre-built operands"). It proves one shape on one execution, which is why it
            # supplements the static proof instead of replacing it.
            steady(f, args...) = (f(args...); @allocated f(args...))   # barrier: top-level `Any` operands bill the call site
            @test_throws StrictMode.StrictViolation @test_noalloc P._syrk_blocked!(false, false, false, 0.8, As, Cs, 512)
            @test steady(P._syrk_blocked!, false, false, false, 0.8, As, Cs, 512) == 0
            As32 = randn(32, 32); Cs32 = zeros(32, 32)   # small-n unified single-pack path (AVX2)
            # Measured through a function barrier, per req#10: at top level the operands are `Any`-typed
            # and the first compile of each such call site caches a method instance, which reports as
            # allocation a caller passing concrete types never pays. `_symm!` reaches gemm, whose SME
            # route is behind a function pointer (one dynamic dispatch), so it is the one that shows it.
            @test_throws StrictMode.StrictViolation @test_noalloc P._syrk_blocked!(false, false, false, 0.8, As32, Cs32, 32)
            @test steady(P._syrk_blocked!, false, false, false, 0.8, As32, Cs32, 32) == 0
            a_syr2kp(A, B, C, nn) = @allocated PureBLAS._syr2k_packed!(false, false, 0.8, 0.3, A, B, C, nn)
            a_symm(A, B, C) = @allocated PureBLAS._symm!(true, false, false, 0.8, 0.3, A, B, C)
            P._syr2k_packed!(false, false, 0.8, 0.3, As, Bs, Cs, 512)
            P._symm!(true, false, false, 0.8, 0.3, As, Bs, Cs)
            @test minimum(a_syr2kp(As, Bs, Cs, 512) for _ in 1:8) == 0
            @test minimum(a_symm(As, Bs, Cs) for _ in 1:8) == 0
            # BROKEN: the all-paths proof on `_symm!`, as for `_trsm_base_invL!` above.
            @test_broken (
                try
                    @test_noalloc P._symm!(true, false, false, 0.8, 0.3, As, Bs, Cs)
                    true
                catch e
                    e isa StrictMode.StrictViolation || rethrow()
                    false
                end
            )
            # PUBLIC ENTRY POINTS — assertable directly now that StrictMode ≥0.3.4 supports kwarg calls
            # (issue el-oso/StrictMode.jl#4). This closes the mandate: StrictMode on every entry point, not
            # just the positional internal drivers. :full-mode JET sees the whole kwarg→dispatch→kernel tree.
            @test_typestable P.trsm!(copy(Bs), tri(512); side = 'L', uplo = 'L', diag = 'N', alpha = 1.0)
            @test_typestable P.syrk!(zeros(512, 512), As; uplo = 'L', trans = 'N', alpha = 0.8, beta = 0.3)
            @test_typestable P.syr2k!(zeros(512, 512), As, Bs; uplo = 'L', trans = 'N', alpha = 0.8, beta = 0.3)
            @test_typestable P.symm!(zeros(512, 512), As, Bs; side = 'L', uplo = 'L', alpha = 0.8, beta = 0.3)
            # OWNED-SCRATCH (GKH) guard — the `:owned` guarantee fails on a runtime AbstractDict lookup
            # reached on the hot path. This is the check that WAS MISSING when the complex `_symm_scr`
            # scratch accessor shipped with const-dispatched owned Refs only for Float64/Float32: ComplexF64/F32
            # fell through to the generic `get(::IdDict, T, …)` (~130 ns/call, ~26% of a tiny-n op). That's
            # type-stable + alloc-free (warm hit) + trim-safe, so it passed all three OTHER asserts — only a
            # benchmark caught it. Now every eltype has an owned Ref, and this guard goes red if that regresses.
            # A keyword call dispatches through `Core.kwcall`, so that is the signature proved.
            kwsig(f, args...; kw...) = (Core.kwcall, (typeof(values(kw)), typeof(f), map(typeof, args)...))
            Az = randn(ComplexF64, 64, 64); Hz = Az + Az'; Bz = randn(ComplexF64, 64, 64)
            test_signatures(
                [
                    kwsig(P.hemm!, zeros(ComplexF64, 64, 64), Hz, Bz; side = 'L', uplo = 'U', alpha = 1.0 + 0im, beta = 0.0im),
                    kwsig(P.symm!, zeros(ComplexF64, 64, 64), Az, Bz; side = 'L', uplo = 'U', alpha = 1.0 + 0im, beta = 0.0im),
                ];
                guarantees = (:owned,)
            )
            # UNPACKED-TRI complex rank-k (small-n trans='N', `_ctri_unpacked!` → `_uker_cmplx!` TRI-store): the
            # path that fixed the zsyrk/zherk n≈24–48 valley. Direct-read A, no pack, masks the diagonal tile —
            # must stay typestable + alloc-free (the tri sweep resolves sb/a1/ar/nr to concrete Vals). trim-side
            # is covered ccallable-rooted in trim_tests.jl; here the type/alloc contract on the hot driver.
            Awz = randn(ComplexF64, 48, 40); Bwz = randn(ComplexF64, 48, 40)
            Cwz = zeros(ComplexF64, 48, 48)
            MZ48 = Matrix{ComplexF64}
            @test_typestable P._ctri_unpacked!(true, true, 1.0, Awz, zeros(ComplexF64, 48, 48), 40)
            @test_typestable P.herk!(zeros(ComplexF64, 48, 48), Awz; uplo = 'U', trans = 'N', alpha = 1.0, beta = 0.0)
            # rank-2k (two products through the shared _ctri_core!)
            @test_typestable P._ctri2_unpacked!(true, true, 1.0, Awz, Bwz, zeros(ComplexF64, 48, 48), 40)
            # ztrsmR-C direct base (`_trsm_cmplx_dRC!`, the zpotrf-lower recursion path) — typestable + alloc-free.
            Atr = randn(ComplexF64, 48, 48) ./ 96; for d in 1:48
                Atr[d, d] = 1 + abs(Atr[d, d])
            end
            Btr = randn(ComplexF64, 64, 48)
            @test_typestable P._trsm_cmplx_dRC!(true, false, 48, Atr, randn(ComplexF64, 64, 48))
            # The complex rank-k and ztrsmR drivers reach no pool at all, so they prove with or without
            # the registration. `alpha`'s type selects the kernel arm, so both arms are asserted rather
            # than assumed to share one.
            #
            # NOT `test_registered()`: it re-proves the signatures `src/` declared with `@strict_function`
            # or `register_strict!`, and PureBLAS's `src/` declares through `@verify_strict` and
            # `@strict_policy`, which register none. `@test_noalloc` and `test_signatures` enumerate
            # directly.
            @test_noalloc P._ctri_unpacked!(true, true, 1.0, Awz, Cwz, 40)
            @test_noalloc P._ctri_unpacked!(false, false, 1.2 + 0.3im, Awz, Cwz, 40)
            @test_noalloc P._ctri2_unpacked!(true, true, 1.0, Awz, Bwz, Cwz, 40)
            @test_noalloc P._ctri2_unpacked!(false, false, 1.2 + 0.3im, Awz, Bwz, Cwz, 40)
            @test_noalloc P._trsm_cmplx_dRC!(true, false, 48, Atr, Btr)
            @test true
            # POSITIVE CONTROL. `@test_noalloc` records nothing when it holds, so an item whose macros
            # had gone inert would look identical to one whose proofs all passed. The all-paths proof
            # must still report the pool growth a warmed call never takes: if this stops throwing, the
            # growth moved out of the driver's call graph and the measured assertion above can become
            # a proof.
            @test_throws StrictMode.StrictViolation @test_noalloc P._trsm_base_invL!(false, false, false, Atri, BL)
        finally
            for f in _POOLS
                delete!(StrictMode._ALLOC_BARRIERS, f)   # internal: no public unregister exists
            end
            StrictMode.clear_cache!()
        end
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
            ap = pointer(Ap); bp = pointer(Bp); cptr = pointer(C); ldc = mr
            # register-blocked microkernel: the hot path — must be tight
            @test_typestable P._microkernel!(cptr, ldc, ap, bp, kc, Val(P._MR), Val(P._NR), Val(false))
            @test_noalloc P._microkernel!(cptr, ldc, ap, bp, kc, Val(P._MR), Val(P._NR), Val(false))
            @test_trim_compatible P._microkernel!(cptr, ldc, ap, bp, kc, Val(P._MR), Val(P._NR), Val(false))
            # `:no_spill`: the µarch-derived _MR×_NR tile must fit the register file
            # with no vector spill/reload. Verified clean on both AVX-512 (Zen4/Zen5, 32 zmm) and AVX2 (Zen3,
            # 16 ymm) — the packed hot path. (NB the SMALL-matrix `_microkernel_unpacked!` spills 3 vectors on
            # AVX2 with the same tile — a real register-pressure finding, tracked separately; not asserted here.)
            test_signatures(
                [(P._microkernel!, (Ptr{Float64}, Int, Ptr{Float64}, Ptr{Float64}, Int, Val{P._MR}, Val{P._NR}, Val{false}))];
                guarantees = (:no_spill,)
            )
            @test_noalloc P._microkernel_masked!(cptr, ldc, ap, bp, kc, 11, 5, Val(P._MR), Val(P._NR))
            @test_typestable P._microkernel_masked!(cptr, ldc, ap, bp, kc, 11, 5, Val(P._MR), Val(P._NR))
            @test_trim_compatible P._microkernel_masked!(cptr, ldc, ap, bp, kc, 11, 5, Val(P._MR), Val(P._NR))
            # clip kernel: W-aligned partial row-tile (reads _MR-strided panel, computes 1 live vector)
            @test_typestable P._microkernel_clip!(cptr, ldc, ap, bp, kc, Val(P._MR), Val(1), Val(P._NR))
            @test_noalloc P._microkernel_clip!(cptr, ldc, ap, bp, kc, Val(P._MR), Val(1), Val(P._NR))
            @test_trim_compatible P._microkernel_clip!(cptr, ldc, ap, bp, kc, Val(P._MR), Val(1), Val(P._NR))
        end
        # unpacked microkernel (small-matrix path): A is mr×k, B is k×nr, column-major
        kk = 32; Au = randn(mr * kk); Bu = randn(kk * nr); Cu = zeros(mr, nr)
        GC.@preserve Au Bu Cu begin
            aup = pointer(Au); bup = pointer(Bu); cup = pointer(Cu)
            @test_typestable P._microkernel_unpacked!(
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
            @test_typestable P._microkernel_unpacked_mrows!(
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
            @test_typestable P._mrows_tail!(
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
        # SPLIT-REDUCTION kernel, BOTH mask arms. `Val{MASK}` selects two different generated bodies —
        # the masked one emits lane arithmetic the unmasked one does not — so a guarantee proven on one
        # says nothing about the other, and the masked arm is the one every ragged row block in
        # `_gemm_unpacked_split!` takes. Trim-reachable from `dgemm_64_`.
        #
        # ITS OWN BUFFERS, `_SMR·W` rows deep: this kernel's tile is _SMR·W = 32 rows against the
        # unpacked kernel's `_MR·W` = 16, so reusing `Cu`/`Au` above would write past the end of a
        # 16-row column. Guarded on `_SPLIT_OK` — off wide-AVX-512 the driver const-folds away and
        # these instances are never compiled.
        #
        # THE ASSERTIONS BELOW RECORD NOTHING ON SUCCESS, so a running block and a skipped one are
        # indistinguishable from the test count. Non-vacuity here was established by a positive
        # control — a planted `@test false` inside the loop, which failed twice, once per mask arm.
        # Re-plant it rather than trusting this sentence if the guard or the buffers change.
        if P._SPLIT_OK
            smr = P._SMR * W
            As = randn(smr * kk); Cs = zeros(smr, P._SNR)
            GC.@preserve As Bu Cs begin
                asp = pointer(As); csp = pointer(Cs); bsp = pointer(Bu)
                for mask in (false, true)
                    mv = mask ? smr - 2 : smr        # ragged vs the full tile
                    @test_typestable P._microkernel_unpacked_split!(
                        csp, smr, asp, smr, 0, bsp, kk, 0, kk, 1.0, 0.0, mv,
                        Val(P._SMR), Val(P._SNR), Val(P._GEMM_SPLIT_S), Val(false), Val(true), Val(mask)
                    )
                    @test_noalloc P._microkernel_unpacked_split!(
                        csp, smr, asp, smr, 0, bsp, kk, 0, kk, 1.0, 2.0, mv,
                        Val(P._SMR), Val(P._SNR), Val(P._GEMM_SPLIT_S), Val(false), Val(false), Val(mask)
                    )
                    @test_trim_compatible P._microkernel_unpacked_split!(
                        csp, smr, asp, smr, 0, bsp, kk, 0, kk, 1.0, 0.0, mv,
                        Val(P._SMR), Val(P._SNR), Val(P._GEMM_SPLIT_S), Val(false), Val(true), Val(mask)
                    )
                end
            end
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
            # The trailing `-1` is `nroute`: route from this call's own width. A serial caller passes
            # -1; only a column-split caller passes the undivided width.
            @test_trim_compatible P._gemm_cmplx_unpacked!(Val(1), Val(1), false, 8, 8, 8, one(TC), Az, Bz, zero(TC), Cz, -1)
            @test_trim_compatible P._gemm_cmplx_unpacked!(Val(1), Val(-1), true, 8, 8, 8, TC(1.3, 0.7), Az, Bz, TC(0.9, -0.4), Cz, -1)
        end
        # packing + generic path allocate nothing
        A = randn(8, 5); Bm = randn(5, 6); Cg = zeros(8, 6)
        @test_typestable P._gemm_generic!(false, false, false, false, 8, 6, 5, 1.0, A, Bm, 0.0, Cg)
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
            @test_typestable P._cpotf2_lower!(copy(A), 48)
            @test_noalloc P._cpotf2_lower!(copy(A), 48)
            @test_typestable P.potrf!(copy(A); uplo = 'L')          # n≤base → single base
            A2 = randn(TC, 128, 128); A2 = A2 * A2' + 128I + zeros(TC, 128, 128)
            @test_typestable P.potrf!(copy(A2); uplo = 'L')         # n>base → recursive nb=n/4 blocked
            @test_typestable P._cpotrf_lower!(copy(A2), 128)
            # UPPER was covered by NEITHER of the above: it is Lever C (conj-transpose → `_potrf_pad`
            # scratch → `_cpotrf_lower!` → conj-transpose back), a different call graph that reaches
            # `_tri_upper_to_lowerT!`/`_tri_lowerT_to_upper!` and `_potf2_needs_buf` — none of which any
            # 'L' assertion roots. Both sizes: n≤base is the single-base window where `_potf2_needs_buf`
            # decides, n>base takes the recursion.
            @test_typestable P.potrf!(copy(A); uplo = 'U')
            @test_typestable P.potrf!(copy(A2); uplo = 'U')
            # complex getf2 panel (`_cgetf2_simd!`, zgetrf base) + QR panel (`qr_unblocked!`, zgeqrf) —
            # both vectorize via the L1 complex kernels; typestable + alloc-free on the native hot path.
            G = randn(TC, 48, 48) + 48I; ip = zeros(Int, 48); pG = pointer(G); ldG = stride(G, 2)
            GC.@preserve G begin
                @test_typestable P._cgetf2_simd!(pG, ldG, 48, 48, 0, ip, 0)
                @test_noalloc P._cgetf2_simd!(pG, ldG, 48, 48, 0, ip, 0)
            end
            Q = randn(TC, 48, 48); tau = similar(Q, 48)
            @test_typestable P.qr_unblocked!(copy(Q), tau)
            # RUNTIME form, not the static proof, and the reason is the one this file's header gives:
            # the panel's Householder norm goes through complex `nrm2`, which takes the BLOCKED
            # reduction — so an all-paths proof counts the L1 reduction pool's `wait` site even though
            # the split is runtime-dead at n=48. The real `nrm2`/`asum` have carried that same static
            # reachability from the day the blocked tree shipped and are under no static proof
            # anywhere, so this is parity with the real path rather than a weakening relative to it.
            # Measured warm, both types: 0 B here, and 0 B at n=1e6 where the split is genuinely LIVE
            # (bench/probes/zqr_alloc_check.jl) — so the guarantee holds, it is just not statically
            # provable.
            qw = copy(Q); P.qr_unblocked!(qw, tau)
            @test_throws StrictMode.StrictViolation @test_noalloc P.qr_unblocked!(qw, tau)
            # The static proof passes because a path allocates, so it cannot see a new per-call buffer;
            # this reads the steady state the guarantee is actually about. Both, not either.
            steady(f, args...) = (f(args...); @allocated f(args...))
            @test steady(P.qr_unblocked!, qw, tau) == 0
        end
        @test true
    end
end

# Eigen family — the contract members, proved.
#
# `gebal!`, `gehrd!`, `orghr!`, `ormhr!`, `hseqr!` and `geev!` are all @strict_contract members
# (src/contracts.jl:368-380) and src/verify.jl's `@verify_strict` block lists purpose-built probes for
# every one of them (`_strict_gehrd_probe` &c, ~20 call sites). That block runs at PureBLAS's own
# precompile, where the proving tier cannot load, so it checks with the value-free engine: inference
# for stability (gates), the IR scan for allocation (reports). test/ PROVES.
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
        let n = 24
            A0 = randn(n, n); A = copy(A0); sc = ones(n); tau = Vector{Float64}(undef, n)
            PureBLAS.gebal!(bk, A, sc)                             # warm
            copyto!(A, A0)
            @test_noalloc PureBLAS.gebal!(bk, A, sc)
            copyto!(A, A0); PureBLAS.gehrd!(bk, A, 1, n, tau)      # warm the arena high-water
            copyto!(A, A0)
            @test_throws StrictMode.StrictViolation @test_noalloc PureBLAS.gehrd!(bk, A, 1, n, tau)
            # The static proof passes because a path allocates, so it cannot see a new per-call buffer;
            # this reads the steady state. Operands pass as arguments rather than closing over the `let`,
            # so the measured call site sees concrete types, and `A` is restored before each call because
            # `gehrd!` overwrites it.
            steady_g(f, b, Aw, Aref, nn, t) =
                (copyto!(Aw, Aref); f(b, Aw, 1, nn, t); copyto!(Aw, Aref); @allocated f(b, Aw, 1, nn, t))
            @test steady_g(PureBLAS.gehrd!, bk, A, A0, n, tau) == 0
        end
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

        # EVERY SIGNATURE BELOW GOES THROUGH `trimok`, AND THE REASON IS THAT THE VERIFIER THROWS.
        # A bare rejection aborts the item, so the first one hides every signature after it — one
        # rejection here masked seventy-two unverified ones. Wrapping turns each into an independent
        # verdict.
        #
        # WHAT IT TOLERATES, AND NOTHING ELSE. Against stock Base the verifier rejects a string
        # interpolation of four or more arguments on a reachable throw path, reporting
        # `Base._str_sizehint` and `Base.print` as unresolved calls (StrictModeTest issue #19). Seven
        # of these drivers carry such a message — the symmetric-indefinite family: `sysv!` both real
        # and complex, `hesv!`, `sytrf!` both, `hetrf!`, and `sytri!` on Float64 only. juliac's
        # patched Base resolves those calls, and the CI job that builds the .so is the authority on
        # that: it compiles them clean. Any OTHER rejection fails the item and prints the finding.
        #
        # `set_juliac_patches!(true)` IS NOT THE WAY OUT ON THIS JULIA, stated so it is not retried:
        # StrictModeTest 0.5.0 looks for juliac's patch files under `share/julia/juliac/`, which does
        # not exist on 1.13.2 — they ship in `share/julia/test/trimming/`. The call warns, falls back
        # to stock Base in this process, and changes no verdict.
        #
        # NON-VACUITY ESTABLISHED, not assumed: narrowing `interp` to a string that never occurs makes
        # exactly the seven tolerated signatures FAIL here, so a rejection outside the class is
        # reported rather than swallowed. Re-run that control if the predicate is ever widened.
        function trimok(th)
            try
                th()                                  # a pass is recorded by the macro itself
            catch err
                e = sprint(showerror, err)
                interp = occursin("_str_sizehint", e) || occursin("Base.print", e)
                interp || @error "trim rejection outside the interpolated-throw class" finding = e
                @test interp
            end
        end
        # ── symmetric-indefinite solve/inverse (sytri/hetri need a real factorization first) ──
        trimok(() -> @test_trim_compatible P.sysv!('L', copy(Ad), randn(n, 2)))
        trimok(() -> @test_trim_compatible P.sysv!('L', copy(Az), randn(ComplexF64, n, 2)))
        trimok(() -> @test_trim_compatible P.hesv!('L', copy(Ahe), randn(ComplexF64, n, 2)))
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
            # Same interpolated-throw class as the solves above — the .so build compiles all three.
            trimok(() -> @test_trim_compatible P.sytrf!(copy(Abd), zeros(Int, nbig); uplo = ul))
            trimok(() -> @test_trim_compatible P.sytrf!(copy(Abz), zeros(Int, nbig); uplo = ul))
            trimok(() -> @test_trim_compatible P.hetrf!(copy(Abh), zeros(Int, nbig); uplo = ul))
        end
        # ONLY the Float64 inverse carries the interpolated throw; its complex siblings verify clean
        # against stock Base, so they are asserted directly. The asymmetry is real, not a slip — a
        # uniform wrapper here would turn two genuine proofs into assertions that something fails.
        trimok(() -> @test_trim_compatible P.sytri!(copy(LDd), ipd; uplo = 'L'))
        trimok(() -> @test_trim_compatible P.sytri!(copy(LDz), ipz; uplo = 'L'))
        trimok(() -> @test_trim_compatible P.hetri!(copy(LDh), iph; uplo = 'L'))
        # ── QL / RQ (geqlf/gerqf + org/orm), real + complex ──
        Aqd = randn(n, k); tqd = zeros(Float64, k); Aqz = randn(ComplexF64, n, k); tqz = zeros(ComplexF64, k)
        Fqd = copy(Aqd); P.geqlf!(Fqd, tqd); Fqz = copy(Aqz); P.geqlf!(Fqz, tqz)
        trimok(() -> @test_trim_compatible P.geqlf!(copy(Aqd), zeros(Float64, k)))
        trimok(() -> @test_trim_compatible P.orgql!(copy(Fqd), copy(tqd)))
        trimok(() -> @test_trim_compatible P.ormql!('L', 'N', copy(Fqd), tqd, randn(n, 3)))
        trimok(() -> @test_trim_compatible P.geqlf!(copy(Aqz), zeros(ComplexF64, k)))
        trimok(() -> @test_trim_compatible P.orgql!(copy(Fqz), copy(tqz)))
        trimok(() -> @test_trim_compatible P.ormql!('L', 'N', copy(Fqz), tqz, randn(ComplexF64, n, 3)))
        Ard = randn(k, n); Arz = randn(ComplexF64, k, n)
        Grd = copy(Ard); P.gerqf!(Grd, tqd); Grz = copy(Arz); P.gerqf!(Grz, tqz)
        trimok(() -> @test_trim_compatible P.gerqf!(copy(Ard), zeros(Float64, k)))
        trimok(() -> @test_trim_compatible P.orgrq!(copy(Grd), copy(tqd)))
        trimok(() -> @test_trim_compatible P.ormrq!('R', 'N', copy(Grd), tqd, randn(3, n)))
        trimok(() -> @test_trim_compatible P.gerqf!(copy(Arz), zeros(ComplexF64, k)))
        trimok(() -> @test_trim_compatible P.orgrq!(copy(Grz), copy(tqz)))
        # ── RZ (tzrzf/ormrz) ──
        Tzd = randn(k, n); ttd = zeros(Float64, k); P.tzrzf!(Tzd, ttd)
        trimok(() -> @test_trim_compatible P.tzrzf!(randn(k, n), zeros(Float64, k)))
        trimok(() -> @test_trim_compatible P.tzrzf!(randn(ComplexF64, k, n), zeros(ComplexF64, k)))
        trimok(() -> @test_trim_compatible P.ormrz!('L', 'N', copy(Tzd), ttd, randn(n, 3)))
        # ── pivoted Cholesky (pstrf) — SPD input ──
        trimok(() -> @test_trim_compatible P.pstrf!(copy(Sd), zeros(Int, n), -1.0; uplo = 'L'))
        trimok(() -> @test_trim_compatible P.pstrf!(copy(She), zeros(Int, n), -1.0; uplo = 'L'))
        # ── rank-deficient / constrained least squares ──
        trimok(() -> @test_trim_compatible P.gelsd!(randn(n, k), randn(n, 2), -1.0))
        trimok(() -> @test_trim_compatible P.gelsd!(randn(ComplexF64, n, k), randn(ComplexF64, n, 2), -1.0))
        trimok(() -> @test_trim_compatible P.gelsy!(randn(n, k), randn(n, 2), zeros(Int, k), -1.0))
        trimok(() -> @test_trim_compatible P.gelsy!(randn(ComplexF64, n, k), randn(ComplexF64, n, 2), zeros(Int, k), -1.0))
        trimok(() -> @test_trim_compatible P.gglse!(randn(8, 6), randn(8), randn(4, 6), randn(4)))
        trimok(() -> @test_trim_compatible P.gglse!(randn(ComplexF64, 8, 6), randn(ComplexF64, 8), randn(ComplexF64, 4, 6), randn(ComplexF64, 4)))
        # ── Sylvester (trsyl) + Schur reorder (trexc/trsen) ──
        trimok(() -> @test_trim_compatible P.trsyl!('N', 'N', 1, triu(randn(n, n)) + n * I, triu(randn(n, n)) + n * I, randn(n, n)))
        trimok(() -> @test_trim_compatible P.trsyl!('N', 'N', 1, triu(randn(ComplexF64, n, n)) + n * I, triu(randn(ComplexF64, n, n)) + n * I, randn(ComplexF64, n, n)))
        trimok(() -> @test_trim_compatible P.trexc!('V', triu(randn(n, n)) + n * I, Matrix(1.0I, n, n), 2, n - 1))
        trimok(() -> @test_trim_compatible P.trexc!('V', triu(randn(ComplexF64, n, n)) + n * I, Matrix(ComplexF64(1)I, n, n), 2, n - 1))
        trimok(() -> @test_trim_compatible P.trsen!('N', 'V', rand(Bool, n), triu(randn(n, n)) + n * I, Matrix(1.0I, n, n)))
        trimok(() -> @test_trim_compatible P.trsen!('N', 'V', rand(Bool, n), triu(randn(ComplexF64, n, n)) + n * I, Matrix(ComplexF64(1)I, n, n)))
        # ── generalized eigen (ggev/gges/sygvd/hegvd) — sygvd/hegvd need PD B ──
        trimok(() -> @test_trim_compatible P.ggev!('N', 'V', randn(n, n), randn(n, n)))
        trimok(() -> @test_trim_compatible P.ggev!('N', 'V', randn(ComplexF64, n, n), randn(ComplexF64, n, n)))
        trimok(() -> @test_trim_compatible P.gges!('V', 'V', randn(n, n), randn(n, n)))
        trimok(() -> @test_trim_compatible P.gges!('V', 'V', randn(ComplexF64, n, n), randn(ComplexF64, n, n)))
        trimok(() -> @test_trim_compatible P.ggsvd!('U', 'V', 'Q', randn(8, 6), randn(6, 6)))
        trimok(() -> @test_trim_compatible P.ggsvd!('U', 'V', 'Q', randn(ComplexF64, 8, 6), randn(ComplexF64, 6, 6)))
        trimok(() -> @test_trim_compatible P.sygvd!(1, 'V', 'U', copy(Ad), copy(Sd)))
        trimok(() -> @test_trim_compatible P.hegvd!(1, 'V', 'U', copy(Ahe), copy(She)))
        # ── symmetric-tridiagonal (stebz/stein), real ──
        dd = randn(n); ee = randn(n - 1)
        trimok(() -> @test_trim_compatible P.stebz!('A', 'B', 0.0, 0.0, 1, n, 0.0, copy(dd), copy(ee)))
        w, ib, isp, _ = P.stebz!('A', 'B', 0.0, 0.0, 1, n, 0.0, copy(dd), copy(ee))
        trimok(() -> @test_trim_compatible P.stein!(copy(dd), copy(ee), w, ib, isp))
        # ── generalized SVD (ggsvd), Float64 full-rank ──
        # ── banded LU (gbtrf/gbtrs) — factor then solve ──
        ABd2 = randn(6, n); _, gip, _ = P.gbtrf!(2, 1, n, ABd2)
        trimok(() -> @test_trim_compatible P.gbtrf!(2, 1, n, randn(6, n)))
        trimok(() -> @test_trim_compatible P.gbtrs!('N', 2, 1, n, copy(ABd2), gip, randn(n, 2)))
        trimok(() -> @test_trim_compatible P.gbtrf!(2, 1, n, randn(ComplexF64, 6, n)))
        # BLOCKED path. The cells above are kl=2, so they can only ever reach `_gbtf2!` — a Base-only
        # scalar kernel — and the blocked port would ship trim-unvalidated, which is exactly the
        # failure the pbtrf item at test/lapack_tests.jl:24 is a post-mortem of. kl ≥ 2·nb is the
        # dispatch gate; kl=32 clears it for every measured nb. This puts trsm!, _gemm_core!,
        # PtrMatrix and the L3Workspace scratch inside gbtrf's trim call graph for the first time.
        ldgb = 2 * 32 + 8 + 1
        P.gbtrf!(32, 8, 64, randn(ldgb, 64))                 # warm the owned workspace first
        P.gbtrf!(32, 8, 64, randn(ComplexF64, ldgb, 64))
        trimok(() -> @test_trim_compatible P.gbtrf!(32, 8, 64, randn(ldgb, 64)))
        trimok(() -> @test_trim_compatible P.gbtrf!(32, 8, 64, randn(ComplexF64, ldgb, 64)))
        # ── SPD tridiagonal (pttrf/pttrs/ptsv) — factor then solve ──
        Dp = fill(4.0, n); Ep = fill(1.0, n - 1); Df = copy(Dp); Ef = copy(Ep); P.pttrf!(Df, Ef)
        trimok(() -> @test_trim_compatible P.pttrf!(fill(4.0, n), fill(1.0, n - 1)))
        trimok(() -> @test_trim_compatible P.pttrs!(copy(Df), copy(Ef), randn(n, 2)))
        trimok(() -> @test_trim_compatible P.ptsv!(fill(4.0, n), fill(1.0, n - 1), randn(n, 2)))
        Dpz = fill(4.0, n); Epz = fill(1.0 + 0im, n - 1)
        trimok(() -> @test_trim_compatible P.ptsv!(copy(Dpz), copy(Epz), randn(ComplexF64, n, 2)))
        # ── general tridiagonal (gtsv/gttrf/gttrs) — factor then solve ──
        dl = fill(1.0, n - 1); dm = fill(4.0, n); du = fill(1.0, n - 1)
        du2 = zeros(Float64, n - 2); gtip = zeros(Int, n); P.gttrf!(copy(dl), copy(dm), copy(du), du2, gtip)
        trimok(() -> @test_trim_compatible P.gtsv!(copy(dl), copy(dm), copy(du), randn(n, 2)))
        trimok(() -> @test_trim_compatible P.gttrf!(copy(dl), copy(dm), copy(du), zeros(Float64, n - 2), zeros(Int, n)))
        trimok(() -> @test_trim_compatible P.gtsv!(fill(1.0 + 0im, n - 1), fill(4.0 + 0im, n), fill(1.0 + 0im, n - 1), randn(ComplexF64, n, 2)))
        # ── banded / packed Cholesky (pbtrf/pbtrs, pptrf/pptrs) — SPD storage ──
        mkband(T) = (AB = zeros(T, 3, n); AB[1, :] .= T(10); AB[2, 1:(n - 1)] .= T(1); AB[3, 1:(n - 2)] .= T(0.5); AB)
        ABsd = mkband(Float64); P.pbtrf!(ABsd; uplo = 'L', kd = 2)
        trimok(() -> @test_trim_compatible P.pbtrf!(mkband(Float64); uplo = 'L', kd = 2))
        trimok(() -> @test_trim_compatible P.pbtrs!(copy(ABsd), randn(n, 2); uplo = 'L', kd = 2))
        trimok(() -> @test_trim_compatible P.pbtrf!(mkband(ComplexF64); uplo = 'L', kd = 2))
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
        trimok(() -> @test_trim_compatible P.pbtrf!(wideband(Float64, 64); uplo = 'L', kd = 64))
        trimok(() -> @test_trim_compatible P.pbtrf!(wideU(Float64, 64); uplo = 'U', kd = 64))
        trimok(() -> @test_trim_compatible P._pbtrf_blocked!(wideband(Float64, 64), n, 64))
        trimok(() -> @test_trim_compatible P._pbtrf_repack_U!(wideU(Float64, 64), n, 64))
        trimok(() -> @test_trim_compatible P._pbtrf_blocked_U!(wideU(Float64, 64), n, 64))
        trimok(() -> @test_trim_compatible P._pbtrf_blocked!(wideband(ComplexF64, 32), n, 32))
        trimok(() -> @test_trim_compatible P._pbtrf_repack_U!(wideU(ComplexF64, 32), n, 32))
        trimok(() -> @test_trim_compatible P._pbtrf_blocked_U!(wideU(ComplexF64, 32), n, 32))
        # The corner work array W and the diagonal-block scratch S are GKH-owned (L3Workspace, via
        # _pbtrf_work), not allocated per CALL. Assert the steady state is actually allocation-free —
        # that IS the point of the ownership, and a per-call `Matrix` would sail through every
        # correctness test.
        # Measured, not proved: the owned W and S reach new high-water marks through `_ws_grow!`/
        # `_ws_slot!` and the arena folds through `_arena_coalesce!`, which are `@strict_exempt :noalloc`
        # growth functions, so the all-paths proof reports them. The warmed second call must allocate
        # nothing; `pb_blocked` and `pb_blocked_U` keep `@allocated` inside a function over typed operands.
        Wpb = wideband(Float64, 64)
        WpbU = wideU(Float64, 64)
        pb_blocked(W, nn) = @allocated P._pbtrf_blocked!(W, nn, 64)
        pb_blocked_U(W, nn) = @allocated P._pbtrf_blocked_U!(W, nn, 64)
        for (f, W0) in ((pb_blocked, Wpb), (pb_blocked_U, WpbU))
            W = copy(W0)
            f(W, n)                               # warm: grow the owned scratch once
            @test minimum(f(copyto!(W, W0), n) for _ in 1:8) == 0
        end
        # BROKEN: the all-paths proof on the same two drivers, with the per-task accessors registered as
        # barriers. Each assertion still runs, so it reports an unexpected pass once it holds.
        let pools = (P._m3ws, P._gpackws, P._strws, P._opt)
            foreach(StrictMode.register_alloc_barrier!, pools)
            try
                @test_broken (
                    try
                        @test_noalloc P._pbtrf_blocked!(Wpb, n, 64)
                        true
                    catch e
                        e isa StrictMode.StrictViolation || rethrow()
                        false
                    end
                )
                @test_broken (
                    try
                        @test_noalloc P._pbtrf_blocked_U!(WpbU, n, 64)
                        true
                    catch e
                        e isa StrictMode.StrictViolation || rethrow()
                        false
                    end
                )
            finally
                foreach(f -> delete!(StrictMode._ALLOC_BARRIERS, f), pools)
                StrictMode.clear_cache!()
            end
        end
        pack(M) = [M[i, j] for j in 1:n for i in j:n]              # lower column-packed
        APsd = pack(Sd); P.pptrf!(APsd; uplo = 'L')
        trimok(() -> @test_trim_compatible P.pptrf!(pack(Sd); uplo = 'L'))
        trimok(() -> @test_trim_compatible P.pptrs!(copy(APsd), randn(n, 2); uplo = 'L'))
        trimok(() -> @test_trim_compatible P.pptrf!(pack(She); uplo = 'L'))
        @test true
    end
end

# POOL GROWTH IS BOUNDED: a warmed second call of every entry that reaches a growth function (`_ws_grow!`,
# `_ws_slot!`, `_arena_grow!`, `_arena_coalesce!`, the LAPACK `_*_grow!`, the QR/LU pad growers)
# allocates 0 bytes. Those functions are `@strict_exempt :noalloc` because a grow-only pool allocates
# again at each new high-water mark, so "finitely many growths, then none" is a property of the call
# sequence that an all-paths proof cannot state. It is asserted here the way req#10 prescribes: warm the
# entry, then `@allocated` inside a function over typed operands (at testitem scope the globals are
# `Any`-typed and the window would measure dispatch), minimum over several windows because the counter
# is process-wide and a threaded call has other threads in it. A per-call leak allocates in EVERY
# window, so the minimum cannot hide one.
#
# Entries are the ones the strict-contract block in src/verify.jl reaches growth through: trsm!, trmm!,
# symv!, trmv!, syr2k!, getrf!, geqrf!, gesvd!, sygvd!, geev!, gees!, plus gemm! and the complex 3M path. The threaded
# arm re-runs the L3 entries and the vector kernels with the pool enabled, sleeping between the warm-up
# calls so a wake from a real park is warmed too.
@testitem "Pool growth is bounded: a warmed second call allocates 0 bytes" tags = [:checks] begin
    using LinearAlgebra, ForwardDiff
    const P = PureBLAS
    a_gemm(C, A, B) = @allocated P.gemm!(C, A, B)
    a_gemm_kw(C, A, B, al, be) = @allocated P.gemm!(C, A, B; alpha = al, beta = be, transB = 'T')
    a_trsm(B, A, side, tr, al) = @allocated P.trsm!(B, A; side = side, uplo = 'L', transA = tr, diag = 'N', alpha = al)
    a_trmm(B, A, side, tr, al) = @allocated P.trmm!(B, A; side = side, uplo = 'L', transA = tr, diag = 'N', alpha = al)
    a_syr2k(C, A, B, al, be) = @allocated P.syr2k!(C, A, B; uplo = 'L', trans = 'N', alpha = al, beta = be)
    a_symv(y, A, x) = @allocated P.symv!(y, A, x; uplo = 'L', alpha = 1.5, beta = 0.5)
    a_trmv(A, x, tr) = @allocated P.trmv!(A, x; uplo = 'L', trans = tr, diag = 'N')
    a_getrf(A, A0, ip) = (copyto!(A, A0); @allocated P.getrf!(A, ip))
    a_geqrf(A, A0, tau) = (copyto!(A, A0); @allocated P.geqrf!(A, tau))
    a_gesvd(A, A0, U, S, Vt) = (copyto!(A, A0); @allocated P.gesvd!(A, U, S, Vt))
    a_sygvd(A, A0, B, B0, w) = (copyto!(A, A0); copyto!(B, B0); @allocated P.sygvd!(1, 'V', 'L', A, B, w))
    a_hegvd(A, A0, B, B0, w) = (copyto!(A, A0); copyto!(B, B0); @allocated P.hegvd!(1, 'V', 'L', A, B, w))
    a_pocon(nrm, F) = @allocated P.pocon!(nrm, F; uplo = 'L')
    a_geev(A, A0, wr, wi, VL, VR, sc) = (copyto!(A, A0); @allocated P.geev!('N', 'V', A, wr, wi, VL, VR, sc))
    a_gees(A, A0, w, VS, sc) = (copyto!(A, A0); @allocated P.gees!('V', A, w, VS, sc))

    tri(T, s) = (M = tril(randn(T, s, s)); for i in 1:s
        M[i, i] += s
    end; M)
    spd(T, s) = (M = randn(T, s, s); M * M' + s * I)
    zero_after(a, args...) = minimum(a(args...) for _ in 1:8) == 0

    # The same entries serially and with the pool enabled. `n` is large enough that the pool admits the
    # L3 entries; whether a given op threads at it is the library's choice, and either way the
    # measurement applies.
    function run_entries(n)
        A = randn(n, n); B = randn(n, n); C = zeros(n, n)
        T1 = tri(Float64, n)
        # gemm!, keyword form, and a Dual element type (the open-type fallback owners)
        P.gemm!(C, A, B); sleep(0.02); P.gemm!(C, A, B)
        @test zero_after(a_gemm, C, A, B)
        P.gemm!(C, A, B; alpha = 2.5, beta = 0.5, transB = 'T')
        @test zero_after(a_gemm_kw, C, A, B, 2.5, 0.5)
        # trsm!/trmm!: both sides, a transposed operand, and alpha != 1 (a narrow B takes the prescale path)
        for side in ('L', 'R'), tr in ('N', 'T')
            Bw = copy(B)
            P.trsm!(Bw, T1; side = side, uplo = 'L', transA = tr, diag = 'N', alpha = 1.0)
            @test zero_after(a_trsm, Bw, T1, side, tr, 1.0)
            P.trmm!(Bw, T1; side = side, uplo = 'L', transA = tr, diag = 'N', alpha = 1.0)
            @test zero_after(a_trmm, Bw, T1, side, tr, 1.0)
        end
        Bn = randn(n, 2)
        P.trsm!(Bn, T1; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 2.0)
        @test zero_after(a_trsm, Bn, T1, 'L', 'N', 2.0)
        # syr2k!
        P.syr2k!(C, A, B; uplo = 'L', trans = 'N', alpha = 1.0, beta = 1.0)
        @test zero_after(a_syr2k, C, A, B, 1.0, 1.0)
        # level 2: symv!, trmv!
        x = randn(n); y = zeros(n)
        P.symv!(y, A, x; uplo = 'L', alpha = 1.5, beta = 0.5)
        @test zero_after(a_symv, y, A, x)
        for tr in ('N', 'T')
            xw = copy(x)
            P.trmv!(T1, xw; uplo = 'L', trans = tr, diag = 'N')
            @test zero_after(a_trmv, T1, xw, tr)
        end
        # complex 3M path
        Cz = zeros(ComplexF64, n, n); Az = randn(ComplexF64, n, n); Bz = randn(ComplexF64, n, n)
        P.gemm!(Cz, Az, Bz); sleep(0.02); P.gemm!(Cz, Az, Bz)
        @test zero_after(a_gemm, Cz, Az, Bz)
        Tz = tri(ComplexF64, n)
        Bzw = copy(Bz)
        P.trsm!(Bzw, Tz; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0 + 0im)
        @test zero_after(a_trsm, Bzw, Tz, 'L', 'N', 1.0 + 0im)
        P.trmm!(Bzw, Tz; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0 + 0im)
        @test zero_after(a_trmm, Bzw, Tz, 'L', 'N', 1.0 + 0im)
    end

    # LAPACK drivers that grow an owned workspace. Warmed once at the shape, then re-seeded per call.
    function run_lapack()
        for n in (64, 512)       # 512 takes the padded-LU scratch and the blocked QR path
            A0 = randn(n, n); A = similar(A0); ip = Vector{Int}(undef, n); tau = Vector{Float64}(undef, n)
            a_getrf(A, A0, ip)
            @test zero_after(a_getrf, A, A0, ip)
            a_geqrf(A, A0, tau)
            @test zero_after(a_geqrf, A, A0, tau)
        end
        Az0 = randn(ComplexF64, 256, 256); Az = similar(Az0); ipz = Vector{Int}(undef, 256); tauz = Vector{ComplexF64}(undef, 256)
        a_getrf(Az, Az0, ipz); @test zero_after(a_getrf, Az, Az0, ipz)
        a_geqrf(Az, Az0, tauz); @test zero_after(a_geqrf, Az, Az0, tauz)
        n = 64
        A0 = randn(n, n); A = similar(A0); U = zeros(n, n); S = zeros(n); Vt = zeros(n, n)
        a_gesvd(A, A0, U, S, Vt)
        @test zero_after(a_gesvd, A, A0, U, S, Vt)
        G0 = spd(Float64, n); B0 = spd(Float64, n); G = similar(G0); Bg = similar(B0); w = zeros(n)
        a_sygvd(G, G0, Bg, B0, w)
        @test zero_after(a_sygvd, G, G0, Bg, B0, w)
        Gz0 = spd(ComplexF64, n); Bz0 = spd(ComplexF64, n); Gz = similar(Gz0); Bgz = similar(Bz0)
        a_hegvd(Gz, Gz0, Bgz, Bz0, w)
        @test zero_after(a_hegvd, Gz, Gz0, Bgz, Bz0, w)
        F = copy(G0); P.potrf!(F; uplo = 'L')
        P.pocon!(opnorm(G0, 1), F; uplo = 'L')
        @test zero_after(a_pocon, opnorm(G0, 1), F)
        # The Francis bulge-chase scratch (`_dlahqr!`, `_dlaqr5!`) must stay off the heap under
        # `--check-bounds=yes`, which `Pkg.test` forces: n = 32 takes the unblocked sweep, n = 256 the multishift one.
        for ne in (32, 256)
            E0 = randn(ne, ne); E = similar(E0); ewr = zeros(ne); ewi = zeros(ne); EVL = zeros(ne, 0)
            EVR = zeros(ne, ne); esc = zeros(ne); ew = Vector{ComplexF64}(undef, ne)
            a_geev(E, E0, ewr, ewi, EVL, EVR, esc)
            @test zero_after(a_geev, E, E0, ewr, ewi, EVL, EVR, esc)
            a_gees(E, E0, ew, EVR, esc)
            @test zero_after(a_gees, E, E0, ew, EVR, esc)
        end
        # the open-type fallback owners: a Dual element type
        n = 32
        Ad = ForwardDiff.Dual{Nothing}.(randn(n, n), randn(n, n)); Bd = ForwardDiff.Dual{Nothing}.(randn(n, n), randn(n, n))
        Cd = zeros(eltype(Ad), n, n)
        P.gemm!(Cd, Ad, Bd)
        @test zero_after(a_gemm, Cd, Ad, Bd)
    end

    P.set_num_threads(1)
    run_entries(128)
    run_lapack()
    nt = Threads.nthreads()
    if nt >= 2
        P.set_num_threads(nt) > 1 || error("no pool")
        try
            run_entries(768)
            run_lapack()
        finally
            P.set_num_threads(1)
        end
    else
        @test_skip nt >= 2
    end
end

# The all-paths proof on `gemm!`, with the per-task accessors registered as barriers. Each assertion is
# marked broken and still runs, so it reports an unexpected pass once it holds. The registration is
# session-wide and keyed on function identity, so this item unregisters afterwards.
@testitem "StrictMode dogfood: gemm! is statically allocation-free (barrier-exempt pools)" tags = [:checks] begin
    using StrictModeTest, StrictMode, LinearAlgebra, ForwardDiff
    if !StrictMode.checks_enabled()
        @info "StrictMode checks disabled — skipping gemm! static noalloc proof"
        @test_skip StrictMode.checks_enabled()
    else
        P = PureBLAS
        pools = (P._m3ws, P._gpackws, P._strws, P._opt)
        foreach(StrictMode.register_alloc_barrier!, pools)
        try
            n = 64
            A = randn(n, n); B = randn(n, n); C = zeros(n, n)
            P.gemm!(C, A, B)                                   # warm: grow every pool once
            @test_broken (
                try
                    @test_noalloc P.gemm!(C, A, B)
                    true
                catch e
                    e isa StrictMode.StrictViolation || rethrow()
                    false
                end
            )
            # The keyword entry with every argument non-default, as `mul!(C, A, B, α, β)` and LAPACK
            # callers use it. Values come from variables, so nothing const-folds the keywords away.
            al = 2.5; be = 0.5
            P.gemm!(C, A, B; alpha = al, beta = be, transB = 'T')
            @test_broken (
                try
                    @test_noalloc P.gemm!(C, A, B; alpha = al, beta = be, transB = 'T')
                    true
                catch e
                    e isa StrictMode.StrictViolation || rethrow()
                    false
                end
            )
            D = ForwardDiff.Dual{Nothing, Float64, 1}
            Ad = ForwardDiff.Dual{Nothing}.(randn(n, n), randn(n, n))
            Bd = ForwardDiff.Dual{Nothing}.(randn(n, n), randn(n, n))
            Cd = zeros(D, n, n)
            P.gemm!(Cd, Ad, Bd)
            @test_broken (
                try
                    @test_noalloc P.gemm!(Cd, Ad, Bd)
                    true
                catch e
                    e isa StrictMode.StrictViolation || rethrow()
                    false
                end
            )
        finally
            foreach(f -> delete!(StrictMode._ALLOC_BARRIERS, f), pools)   # internal: no public unregister exists
            StrictMode.clear_cache!()                                      # the registration cached verdicts
        end
    end
end

# ── COMPILED-SPECIALIZATION BUDGET AND GROWTH ─────────────────────────────────────────────────────
# WHY THIS ITEM EXISTS, and it is a defect this suite could not catch. Two PRs widened `_trsv_reg!`'s
# `@generated` ladder from 4 shape combinations x 12 sizes to 8 x 16, which with the array-type pairs
# reaching it took the method from roughly 576 to 1536 compiled specializations. Lints were clean, the
# full suite was green, and both merged. The cost surfaced only when another session measured the
# package image on a different platform and found it 383 MiB larger — the `.ji` is indifferent (it
# varied by 42 bytes across an 8x range of the bound), because the expansion happens in codegen.
#
# THE GATE IS A BUDGET, NOT A GROWTH COMPARISON, and that is a constraint rather than a preference.
# `specialization_report` judges TWO limits, so `max_type_depth` is pinned at `typemax(Int)` — left at
# its default of 4 the item fails on depth and reads as a budget failure, which is a different bug
# than the one it is here to catch.
#
# ⚠ A GROWTH GATE CANNOT LIVE IN A TESTITEM. `StrictModeTest.test_specializations(mod; baseline,
# workload)` counts the specializations compiled in the CURRENT PROCESS, so its stored baseline is
# only valid against an identical process history. Testitems share a process, so by the time this one
# runs the items before it have already compiled gesvd, getrf, gebrd and the rest: a baseline recorded
# by running this item alone then reads `gemm!` at 325 against 67 and throws. Suite order and CI
# sharding both change that population, so no tolerance makes it reproducible — it is not noise, it is
# a different measurement. A budget is immune in the one direction that matters: extra process history
# can only raise a count, so it can report a false FAILURE but never a false pass.
#
# ⚠ `findings_diff`/`audit_changed` cannot substitute, confirmed in StrictMode's source by its author:
# `:specialization` findings are `:info`, `nfailures` counts only `status === :fail`, so an `:info`
# count change is invisible to the diff. Do not reach for it here either.
@testitem "StrictMode dogfood: compiled-specialization budget" tags = [:checks] begin
    using PureBLAS, StrictMode
    using LinearAlgebra
    P = PureBLAS

    # A workload, not a sweep: a method carries no specializations until something has called it, so
    # the budget below means nothing unless the kernels it judges have been exercised in this process.
    function spec_workload()
        for T in (Float64, Float32)
            A = Matrix{T}(I, 24, 24) .+ T(0.5); B = copy(A); C = zeros(T, 24, 24)
            P.gemm!(C, A, B)
            x = ones(T, 24)
            for ul in ('U', 'L'), tr in ('N', 'T'), dg in ('N', 'U')
                P.trsv!(P.SIMDBackend(), A, copy(x); uplo = ul, trans = tr, diag = dg)
            end
            P.symm!(P.SIMDBackend(), C, A, B; side = 'L', uplo = 'U', alpha = one(T), beta = zero(T))
        end
        return nothing
    end
    spec_workload()

    # BUDGET. The limit is REASONED FROM THE TWO METHODS THAT EXCEED IT, which are different problems:
    #
    #   * `_trsv_reg!` — 1792, and REDUCIBLE. Its 128 `Val` combinations (8 shapes x 16 sizes) are
    #     irreducible, but the ~14 `(A, x)` array-type pairs multiplying them are not: every one
    #     immediately takes `pointer(A)` and `stride(A, 2)`, so normalising at the ladder boundary
    #     collapses them to the 2 element types, 128 x 2 = 256.
    #   * `_uker_cmplx!` — 800, and NOT reducible the same way. It already takes `Ptr{T}`, so there is
    #     no array-type multiplier to remove; the 800 is ~400 `Val` combinations x 2 element types, and
    #     those combinations ARE the generated tile shape space.
    #
    # So the budget cannot be 256: that is below an irreducible count, and a gate below what the design
    # requires is a marker that fails forever rather than a target. 1024 sits above `_uker_cmplx!`'s
    # irreducible space and below `_trsv_reg!`'s reducible one, so it FAILS today at 1792 and PASSES
    # once the ladder is normalised — it can come out both ways, which is the whole requirement. A
    # `@test_broken` that starts passing errors as "Unexpectedly Passed", so the flip is not silent.
    #
    # ⚠ RAISING THIS TO SILENCE A FAILURE DEFEATS IT. If a third method appears above 1024, the
    # question is whether its `Val` space is irreducible like `_uker_cmplx!`'s or carries a removable
    # type multiplier like `_trsv_reg!`'s — not what number makes the line green.
    @test_broken isempty(specialization_report(PureBLAS;
        max_specializations = 1024, max_type_depth = typemax(Int)))
end

# RETURN ANNOTATIONS ON CONTRACT MEMBERS. `@verify` proves a member's declared return type from the
# method's own `::T` annotation when it can read one, and falls back to inference of the abstract
# signature when it cannot; for the 78 members of the SIMDBackend contracts that fallback costs
# minutes of precompile time. This item fails when any mandatory member loses its annotation, or has
# one that is not a subtype of the declared return type.
#
# It reads TypeContracts internals (`_contract_specs`, `_registry_key`, `_build_sig`,
# `_resolve_rt_spec`, `_method_annotation`) because the annotation reader is not public. The invariant
# those calls implement: every method applicable to a spec's signature has a return annotation `R`
# with `R <: declared`; a spec whose declared type is `Any` constrains nothing and is skipped.
@testitem "Contract members carry return annotations" tags = [:checks] begin
    using StrictMode, StrictMode.TypeContracts, InteractiveUtils
    const TC = StrictMode.TypeContracts

    # Descriptions of the mandatory specs of `T` (over its supertype chain) that are not proven by annotation.
    function unannotated_specs(T::Type)
        bad = String[]
        for S in supertypes(T)
            for spec in TC._contract_specs(TC._registry_key(S))
                spec.optional && continue
                expected = TC._resolve_rt_spec(T, spec)
                expected === Any && continue
                sig = TC._build_sig(spec.arg_types, T)
                ms = methods(spec.f, sig)
                if isempty(ms)
                    push!(bad, "$(spec.description): no method")
                    continue
                end
                for m in ms
                    ann = TC._method_annotation(m)
                    (!isnothing(ann) && ann <: expected) || push!(bad, "$(spec.description): $m has annotation $ann")
                end
            end
        end
        return bad
    end

    # Positive control: the same check must report a missing annotation and a wrong one.
    abstract type Toy end
    function toyf end
    @contract Toy begin
        toyf(::Self)::Int => "toy member"
    end
    struct ToyAnnotated <: Toy end
    struct ToyBare <: Toy end
    struct ToyWrong <: Toy end
    toyf(::ToyAnnotated)::Int = 1
    toyf(::ToyBare) = 1
    toyf(::ToyWrong)::Float64 = 1.0
    @test isempty(unannotated_specs(ToyAnnotated))
    @test length(unannotated_specs(ToyBare)) == 1
    @test length(unannotated_specs(ToyWrong)) == 1

    @test sum(S -> length(TC._contract_specs(TC._registry_key(S))), supertypes(PureBLAS.SIMDBackend)) > 70
    bad = unannotated_specs(PureBLAS.SIMDBackend)
    @test isempty(bad)
    isempty(bad) || foreach(b -> @info(b), bad)
end

# FAILING CONTROLS. A declaration that cannot fail checks nothing, so every mechanism a guarantee is
# declared through — in src/verify.jl and in this suite — is fed a known-bad target here, and the item
# asserts it fails. Each target is a toy whose defect is built in: a `Union` return, a `copy`, a
# dynamic `invokelatest`, a loop of accumulators past the register file, a pointer chase (each load's address is the
# previous load's value, so no vector form exists on any ISA), an `IdDict` lookup.
@testitem "StrictMode controls: every declaration fails on a known-bad target" tags = [:checks] setup = [KernelChecks] begin
    using StrictModeTest, StrictMode, StrictMode.TypeContracts, TrimCheck, Test
    using .KernelChecks: toy_spill_body, NVREG
    SV = StrictMode.StrictViolation
    abstract type CtlIface end
    function ctl_op end
    @strict_contract CtlIface begin
        ctl_op(::Self, x::Vector{Float64})::Vector{Float64}
    end
    struct CtlImpl <: CtlIface end
    ctl_op(::CtlImpl, x::Vector{Float64})::Vector{Float64} = x
    ctl_unstable(x::Vector{Float64}) = x[1] > 0 ? 1 : "negative"
    ctl_alloc(x::Vector{Float64}) = copy(x)
    ctl_dyn(x::Int) = Base.invokelatest(identity, x)
    chase_body = quote
        s = 0.0
        j = 1
        for _ in 1:n
            s += unsafe_load(p, j)
            j = unsafe_load(q, j)
        end
        return s
    end
    @eval ctl_chase(p::Ptr{Float64}, q::Ptr{Int}, n::Int) = $chase_body
    const CTL_SCRATCH = IdDict{DataType, Vector{Float64}}(Float64 => zeros(2))
    ctl_lookup(::Type{T}) where {T} = CTL_SCRATCH[T]
    t = CtlImpl(); x = [1.0, 2.0]

    # `@verify_strict` (src/verify.jl): stability gates, allocation reports, a clean call passes.
    @test_throws SV @verify_strict CtlImpl begin
        ctl_unstable(x)
    end
    @test_logs (:warn, r"StrictViolation") match_mode = :any @verify_strict CtlImpl begin
        ctl_alloc(x)
    end
    @verify_strict CtlImpl begin
        ctl_op(t, x)
    end

    # `@strict_policy` (src/verify.jl): `:typestable` and `:no_spill` gate at the module's load,
    # `:trim_compatible` reports. Each toy module holds one concrete method the policy selects.
    @test_throws SV @eval module CtlPolicyStable
        using StrictMode
        ctlp_unstable(x::Int) = x > 0 ? 1 : "negative"
        @strict_policy :typestable only = r"^ctlp_"
    end
    @test_throws SV @eval module CtlPolicySpill
        using StrictMode
        ctlp_spill!(p::Ptr{Float64}, n::Int) = $(toy_spill_body(3 * NVREG))
        @strict_policy :no_spill only = r"^ctlp_"
    end
    @test_throws SV @eval module CtlPolicyVectorized
        using StrictMode
        ctlp_chase(p::Ptr{Float64}, q::Ptr{Int}, n::Int) = $chase_body
        @strict_policy :vectorized only = r"^ctlp_"
    end
    # `@strict_function guarantees = … signatures = … name` (src/verify.jl, on the `@generated` gemm
    # microkernels): `:no_spill` and `:vectorized` gate at the module's load, on a `@generated` method.
    @test_throws SV @eval module CtlFnSpill
        using StrictMode
        @generated ctlg_spill!(p::Ptr{Float64}, n::Int, ::Val{N}) where {N} = $(toy_spill_body)(N)
        @strict_function guarantees = (:no_spill,) signatures = [(Ptr{Float64}, Int, Val{$(3 * NVREG)})] ctlg_spill!
    end
    @test_throws SV @eval module CtlFnVectorized
        using StrictMode
        @generated ctlg_chase(p::Ptr{Float64}, q::Ptr{Int}, n::Int, ::Val{U}) where {U} = $(QuoteNode(chase_body))
        @strict_function guarantees = (:vectorized,) signatures = [(Ptr{Float64}, Ptr{Int}, Int, Val{1})] ctlg_chase
    end
    # The same form on a plain concrete method (src/verify.jl, on `_trsm_tile_R_f64!`), in a module that
    # also holds a policy.
    @test_throws SV @eval module CtlFnPlainSpill
        using StrictMode
        ctlp_clean(x::Int) = x + 1
        ctlq_spill!(p::Ptr{Float64}, n::Int) = $(toy_spill_body(3 * NVREG))
        @strict_policy :typestable only = r"^ctlp_"
        @strict_function guarantees = (:no_spill,) signatures = [(Ptr{Float64}, Int)] ctlq_spill!
    end
    @test_logs (:warn, r"trim_compatible") match_mode = :any @eval module CtlPolicyTrim
        using StrictMode
        ctlp_dyn(x::Int) = Base.invokelatest(identity, x)
        @strict_policy :trim_compatible only = r"^ctlp_"
    end

    # The proving forms this suite uses.
    @test_throws SV @test_typestable ctl_unstable(x)
    @test_throws SV @test_noalloc ctl_alloc(x)
    @test_throws SV @test_trim_compatible ctl_dyn(1)
    @test_throws SV test_signatures([(ctl_unstable, (Vector{Float64},))]; guarantees = (:typestable,))
    @test_throws SV test_signatures([(ctl_alloc, (Vector{Float64},))]; guarantees = (:noalloc,))
    @test_throws SV test_signatures([(ctl_chase, (Ptr{Float64}, Ptr{Int}, Int))]; guarantees = (:vectorized,))
    @test_throws SV test_signatures([(ctl_lookup, (Type{Float64},))]; guarantees = (:owned,))
    # `:no_spill` and `:no_scalar_loops` under `test_signatures`: test/strict_kernel_tests.jl.
end
