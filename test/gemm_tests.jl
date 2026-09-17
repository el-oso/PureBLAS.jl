@testmodule GemmOracle begin
using LinearAlgebra
export gerr, gtol
gerr(a, b) = norm(a .- b) / max(norm(b), eps(Float64))
gtol(::Type{T}) where {T} = T <: Union{Float32, ComplexF32} ? 1.0e-3 : 1.0e-11
end

@testitem "GEMM real (blocked) vs OpenBLAS" setup = [GemmOracle] begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    @testset "$T $tA$tB m=$m n=$n k=$k" for T in (Float32, Float64),
            (tA, tB) in (('N', 'N'), ('T', 'N'), ('N', 'T'), ('T', 'T')),
            m in (1, 16, 17, 40), n in (1, 6, 7, 31), k in (1, 16, 33)

        A = tA == 'N' ? randn(T, m, k) : randn(T, k, m)
        Bm = tB == 'N' ? randn(T, k, n) : randn(T, n, k)
        C0 = randn(T, m, n)
        for (al, be) in ((one(T), zero(T)), (T(0.5), T(2)), (zero(T), T(1.5)))
            Cref = copy(C0); B.gemm!(tA, tB, al, A, Bm, be, Cref)
            Cp = copy(C0); PureBLAS.gemm!(Cp, A, Bm; alpha = al, beta = be, transA = tA, transB = tB)
            @test gerr(Cp, Cref) < gtol(T)
        end
    end
end

@testitem "GEMM blocked path (>unpack threshold) vs OpenBLAS" setup = [GemmOracle] begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    # Sizes above the unpacked threshold (96) exercise the blocked path incl. its masked edge.
    @testset "$T $tA$tB m=$m n=$n k=$k" for T in (Float32, Float64),
            (tA, tB) in (('N', 'N'), ('T', 'N'), ('N', 'T')),
            (m, n, k) in ((100, 100, 100), (130, 97, 113), (160, 200, 128), (97, 150, 99))

        A = tA == 'N' ? randn(T, m, k) : randn(T, k, m)
        Bm = tB == 'N' ? randn(T, k, n) : randn(T, n, k)
        C0 = randn(T, m, n)
        for (al, be) in ((one(T), zero(T)), (T(0.7), T(1.3)))
            Cref = copy(C0); B.gemm!(tA, tB, al, A, Bm, be, Cref)
            Cp = copy(C0); PureBLAS.gemm!(Cp, A, Bm; alpha = al, beta = be, transA = tA, transB = tB)
            @test gerr(Cp, Cref) < gtol(T)
        end
    end
end

@testitem "GEMM beta=0 ignores NaN in C (BLAS semantics)" setup = [GemmOracle] begin
    using PureBLAS
    A = randn(64, 48); B = randn(48, 32)
    C = fill(NaN, 64, 32)
    PureBLAS.gemm!(C, A, B; alpha = 1.0, beta = 0.0)
    @test all(isfinite, C) && gerr(C, A * B) < gtol(Float64)
end

@testitem "GEMM complex (SIMD split-pack + unpacked) vs OpenBLAS" setup = [GemmOracle] begin
    using PureBLAS, LinearAlgebra
    import LinearAlgebra.BLAS as B
    # sizes span the routing: tiny→generic (5), unpacked small-n (23,40 — >nr cols & >mr rows to
    # exercise the jr/ir tiling), and >_CGEMM_UNPACK_MAX→blocked (100). n>nr with jr>0 was a real bug.
    @testset "$T $tA$tB $m×$n×$k" for T in (ComplexF32, ComplexF64),
            (tA, tB) in (('N', 'N'), ('C', 'N'), ('N', 'C'), ('T', 'T'), ('C', 'C')),
            (m, n, k) in ((23, 17, 19), (5, 5, 5), (40, 48, 33), (100, 96, 77))

        A = tA == 'N' ? randn(T, m, k) : randn(T, k, m)
        Bm = tB == 'N' ? randn(T, k, n) : randn(T, n, k)
        C0 = randn(T, m, n); al = T(0.7, -0.3); be = T(1.4, 0.2)
        Cref = copy(C0); B.gemm!(tA, tB, al, A, Bm, be, Cref)
        Cp = copy(C0); PureBLAS.gemm!(Cp, A, Bm; alpha = al, beta = be, transA = tA, transB = tB)
        @test gerr(Cp, Cref) < gtol(T)
    end
end

@testitem "GEMM allocating gemm(A,B) == A*B" setup = [GemmOracle] begin
    using PureBLAS
    A = randn(50, 30); B = randn(30, 40)
    @test gerr(PureBLAS.gemm(A, B), A * B) < gtol(Float64)
    @test gerr(PureBLAS.gemm(A, A; transA = 'T'), A' * A) < gtol(Float64)
end

@testitem "GEMM is AD-traceable (generic path, ForwardDiff)" begin
    using PureBLAS, ForwardDiff, LinearAlgebra
    A = randn(8, 5); dA = randn(8, 5); B = randn(5, 6)
    # d/dt sum((A + t·dA)·B) = sum(dA·B)
    f(t) = sum(PureBLAS.gemm(A .+ t .* dA, B))
    @test ForwardDiff.derivative(f, 0.0) ≈ sum(dA * B)
end

@testitem "GEMM steady-state is allocation-free (driver level)" begin
    using PureBLAS
    # unpacked (small, max dim ≤ 96) path — no buffers at all
    A = randn(48, 48); B = randn(48, 48); C = zeros(48, 48)
    PureBLAS.gemm!(C, A, B; alpha = 1.0, beta = 0.0)            # warmup/compile
    @test (@allocated PureBLAS.gemm!(C, A, B; alpha = 1.0, beta = 0.0)) == 0
    @test (@allocated PureBLAS.gemm!(C, A, B; alpha = 2.0, beta = 1.0)) == 0  # beta≠0 branch
    # blocked (large) path — scratch allocated on first call, then reused → 0 thereafter
    Al = randn(300, 300); Bl = randn(300, 300); Cl = zeros(300, 300)
    PureBLAS.gemm!(Cl, Al, Bl; beta = 0.0)                       # warmup (allocates scratch)
    @test (@allocated PureBLAS.gemm!(Cl, Al, Bl; beta = 0.0)) == 0
    # COMPLEX, and specifically INSIDE the Karatsuba-3M window (_CGEMM_3M_MIN=48 ≤ max(m,n,k) ≤ 2048,
    # min ≥ _CGEMM_3M_KMIN=16). This case had NO allocation coverage at any size, which is how
    # `_gemm_3m!` shipped building nine `unsafe_wrap(Array, …)` headers per call — ~1 KB of steady-state
    # allocation on every 3M gemm, live on AVX2 from the day 3M landed and invisible because the only
    # allocation test here was real Float64. Fixed by taking `PtrMatrix` views over the pooled buffers
    # (isbits ⇒ no header). n=128 sits in the window on every µarch; n=32 is the below-window control,
    # so the pair also pins the routing, not just the total.
    for nc in (32, 128)
        Az = randn(ComplexF64, nc, nc); Bz = randn(ComplexF64, nc, nc); Cz = zeros(ComplexF64, nc, nc)
        PureBLAS.gemm!(Cz, Az, Bz; alpha = one(ComplexF64), beta = zero(ComplexF64))   # warmup + pool growth
        @test (@allocated PureBLAS.gemm!(Cz, Az, Bz; alpha = one(ComplexF64), beta = zero(ComplexF64))) == 0
    end
    # complex rank-k rides the same buffers through `_ctrgemm_3m!` (n ≥ _CSYRK_3M_MIN)
    As = randn(ComplexF64, 300, 300); Cs = zeros(ComplexF64, 300, 300)
    PureBLAS.syrk!(Cs, As; uplo = 'U', trans = 'N', alpha = true, beta = false)
    @test (@allocated PureBLAS.syrk!(Cs, As; uplo = 'U', trans = 'N', alpha = true, beta = false)) == 0
    PureBLAS.herk!(Cs, As; uplo = 'U', trans = 'N', alpha = 1.0, beta = 0.0)
    @test (@allocated PureBLAS.herk!(Cs, As; uplo = 'U', trans = 'N', alpha = 1.0, beta = 0.0)) == 0
end

@testitem "GEMM dimension mismatch is caught" begin
    using PureBLAS
    @test_throws DimensionMismatch PureBLAS.gemm!(zeros(3, 3), zeros(3, 4), zeros(5, 3))
end

@testitem "threaded gemm: same answer as the serial path, and still 0 B" tags = [:checks] begin
    using PureBLAS, Base.Threads
    P = PureBLAS
    # The split is by COLUMNS of C, so the shapes that matter are the ones where a chunk boundary can
    # land badly: n not a multiple of the register tile, n smaller than the worker count, and both
    # transpose flags — with `transB` a chunk of op(B) is a ROW slice of B, a different pointer walk
    # from the column slice every other case uses.
    # Threading is opt-in, so the default must be OFF even on a threaded worker — that is the check
    # that a host which never asks for threads never gets them.
    @test P.get_num_threads() == 1
    @test P._gemm_workers(512, 512, 512) == 1
    if Threads.nthreads() < 2
        @info "single-threaded worker — threading cannot engage; checking it stays off"
        @test P.set_num_threads(4) == 1
    else
        @test P.set_num_threads(Threads.nthreads()) == Threads.nthreads()
        for (m, n, k) in ((256, 256, 256), (512, 300, 128), (200, 512, 333), (1024, 129, 64), (192, 7, 192))
            for tA in (false, true), tB in (false, true)
                A = tA ? randn(k, m) : randn(m, k)
                B = tB ? randn(n, k) : randn(k, n)
                C0 = randn(m, n)
                al, be = 1.7, -0.3
                Cs = copy(C0)
                P._gemm_core!(Cs, A, B, al, be, tA, tB, false, false)
                Ct = copy(C0)
                P.gemm!(Ct, A, B; alpha = al, beta = be,
                    transA = tA ? 'T' : 'N', transB = tB ? 'T' : 'N')
                @test Ct ≈ Cs rtol = 1e-12
            end
        end
        # req#10 holds on the threaded path too: the pool is built once, so a warm call allocates
        # nothing even when it wakes four workers.
        A = randn(512, 512); B = randn(512, 512); C = zeros(512, 512)
        @test P._gemm_workers(512, 512, 512) > 1
        P.gemm!(C, A, B); P.gemm!(C, A, B)
        @test (@allocated P.gemm!(C, A, B)) == 0
        P.set_num_threads(1)                      # leave the process as we found it
        @test P.get_num_threads() == 1
    end
end

@testitem "threaded gemm: a concurrent caller falls back to serial, not to a shared pool" tags = [:checks] begin
    using PureBLAS, Base.Threads
    P = PureBLAS
    if Threads.nthreads() < 2
        @test_skip Threads.nthreads() >= 2
    else
        # Several callers hit `gemm!` at once. Exactly one may own the pool; the rest must run serially
        # and still get the right answer. A wrong claim here would show up as two drivers publishing
        # jobs into one pool, i.e. torn results — not as an error.
        nt = min(4, Threads.nthreads())
        P.set_num_threads(nt)
        A = randn(300, 300); B = randn(300, 300)
        want = similar(A); P._gemm_core!(want, A, B, 1.0, 0.0, false, false, false, false)
        outs = [zeros(300, 300) for _ in 1:nt]
        Threads.@threads :static for t in 1:nt
            for _ in 1:8
                P.gemm!(outs[t], A, B)
            end
        end
        for t in 1:nt
            @test outs[t] ≈ want rtol = 1e-12
        end
        P.set_num_threads(1)
    end
end
