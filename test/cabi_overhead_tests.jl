# C-ABI ROUTING PARITY — does an application reach the same kernel a probe does?
#
# WHY THIS FILE EXISTS. On 2026-08-28 a PureOSQP workload measured `A'*y` at **0.10x OpenBLAS** on
# Zen5 while the gate reported gemvT PASSING on all three boxes. Both were correct: the gate calls
# `PureBLAS.gemv!` directly, the application went through `activate()` → LBT → the `@ccallable` C-ABI,
# and the two took DIFFERENT PATHS. `_l2_simd_ok` gates the SIMD kernels on
# `x isa StridedVector`; the C-ABI passes raw `Ptr{T}` vectors, `Ptr <: StridedVector` is false, so
# every C-ABI BLAS-2 call fell to the generic scalar loop — a 16x penalty on the identical kernel:
#
#     OpenBLAS via BLAS.gemv!                     10.44 us   baseline
#     PureBLAS.gemv! (native call)                 6.57 us   1.59x
#     BLAS.gemv! after activate() [LBT]          104.02 us   0.10x   <- same kernel
#
# This is the SECOND instance of the class: `level1.jl` records an izamax C-ABI miss with the note
# "a wire-the-fastest-path miss that no gate row could see". Nothing in bench/ exercises the C-ABI
# entry, so the entire gate is blind to it by construction, and a 16x regression can ship green.
#
# WHAT THIS TESTS, and why it is shaped this way. It does NOT test absolute speed (that is the gate's
# job, on locked boxes) and it does NOT compare against OpenBLAS (that would make a correctness suite
# depend on a reference's performance). It tests ONE invariant that must hold on any machine, loaded
# or idle: **the same operation, same data, called natively vs through `activate()`, must reach the
# same code path** — so their times must be within a small factor. Route divergence produces
# order-of-magnitude gaps (16x here, 6x for the izamax case); ordinary noise produces well under 2x.
# The threshold is deliberately loose so this cannot flake on a busy CI runner while still catching
# the class. A ratio is used rather than a wall-clock budget so it is machine-independent.
#
# NOT Chairmarks: `test/estimator_lint.jl` forbids raw clocks under bench/, and rightly. This is
# test/, not bench/, and adding Chairmarks to the test env for a routing check would be the wrong
# trade — but the estimator rule's SPIRIT still applies, so this takes the MEDIAN of repeated timings
# and never a min/mean, and it warms both paths before measuring.

@testitem "C-ABI routing parity: activate() must not lose the fast path" tags = [:lbt] begin
    using PureBLAS, LinearAlgebra

    # Median of `reps` timed runs, after a warmup call. Median (never min/mean) per the project's
    # estimator rule; the absolute values are throwaway, only their RATIO is asserted on.
    # Sorted-middle inline rather than `using Statistics` — the test env does not carry it, and a
    # dependency is not worth one line for a routing check.
    med_time(f, reps = 9) = begin
        f()                                   # warm: compile + first-touch
        ts = Float64[]
        for _ in 1:reps
            t0 = time_ns(); f(); push!(ts, (time_ns() - t0) * 1e-9)
        end
        sort!(ts)[(length(ts) + 1) ÷ 2]
    end

    # A route divergence is 6-16x. Real noise on a loaded CI runner is well under 2x. 4x sits between
    # them with room on both sides: it cannot flake on a busy machine, and it cannot miss the class.
    MAXRATIO = 4.0

    m, n = 400, 200                            # tall-skinny: the PureOSQP shape that exposed this
    A  = randn(m, n)
    S  = (Q = randn(n, n); Q'Q + n * I)        # symmetric positive definite
    L  = tril(S)
    x  = randn(n); y = randn(m); v = randn(n)

    # (name, native call, the same operation through LinearAlgebra/BLAS — which activate() reroutes)
    cases = [
        ("gemv N", () -> PureBLAS.gemv!(y, A, x; alpha = 1.0, beta = 0.0, trans = 'N'),
                   () -> BLAS.gemv!('N', 1.0, A, x, 0.0, y)),
        ("gemv T", () -> PureBLAS.gemv!(x, A, y; alpha = 1.0, beta = 0.0, trans = 'T'),
                   () -> BLAS.gemv!('T', 1.0, A, y, 0.0, x)),
        ("symv",   () -> PureBLAS.symv!(x, S, v; uplo = 'L', alpha = 1.0, beta = 0.0),
                   () -> BLAS.symv!('L', 1.0, S, v, 0.0, x)),
        ("trsv",   () -> PureBLAS.trsv!(L, copy(v); uplo = 'L', trans = 'N', diag = 'N'),
                   () -> BLAS.trsv!('L', 'N', 'N', L, copy(v))),
        ("ger",    () -> PureBLAS.ger!(1.0, y, x, A),
                   () -> BLAS.ger!(1.0, y, x, A)),
    ]

    native = Dict(nm => med_time(f) for (nm, f, _) in cases)

    PureBLAS.activate()
    routed = try
        Dict(nm => med_time(g) for (nm, _, g) in cases)
    finally
        PureBLAS.deactivate()                  # restore OpenBLAS even if a case throws
    end

    for (nm, _, _) in cases
        r = routed[nm] / native[nm]
        # Only a LARGE ratio is a failure: routed being FASTER is fine (measurement order, cache
        # state), and this test makes no claim about which entry should win.
        r <= MAXRATIO || @error "C-ABI route divergence: `$nm` is $(round(r; digits = 1))x slower \
            through activate() than called natively. The C-ABI is almost certainly missing a \
            fast-path guard — check that the guard accepts raw `Ptr` arguments (`_dense1`/`_strided1`) \
            and that the kernel uses `_ptr(x)` rather than `pointer(x)`." native_s = native[nm] routed_s = routed[nm]
        @test r <= MAXRATIO
    end
end

# Correctness half: the C-ABI must compute the SAME ANSWER as the native call. A routing bug that
# silently reached a different (or no) kernel would show here even if it happened to be fast — the
# `ger` case in the 2026-08-28 investigation measured 39x "faster" and had to be checked for exactly
# this before the number could be believed.
@testitem "C-ABI routing parity: activate() must not change results" tags = [:lbt] begin
    using PureBLAS, LinearAlgebra

    m, n = 200, 100
    A0 = randn(m, n); x = randn(n); y = randn(m)
    S  = (Q = randn(n, n); Q'Q + n * I); L = tril(S); v = randn(n)

    # reference: native PureBLAS calls
    r_gemvN = (t = zeros(m); PureBLAS.gemv!(t, A0, x; alpha = 1.0, beta = 0.0, trans = 'N'); t)
    r_gemvT = (t = zeros(n); PureBLAS.gemv!(t, A0, y; alpha = 1.0, beta = 0.0, trans = 'T'); t)
    r_ger   = (t = copy(A0); PureBLAS.ger!(1.0, y, x, t); t)

    PureBLAS.activate()
    g_gemvN, g_gemvT, g_ger = try
        (BLAS.gemv!('N', 1.0, A0, x, 0.0, zeros(m)),
         BLAS.gemv!('T', 1.0, A0, y, 0.0, zeros(n)),
         (t = copy(A0); BLAS.ger!(1.0, y, x, t); t))
    finally
        PureBLAS.deactivate()
    end

    @test g_gemvN ≈ r_gemvN
    @test g_gemvT ≈ r_gemvT
    @test g_ger   ≈ r_ger
end
