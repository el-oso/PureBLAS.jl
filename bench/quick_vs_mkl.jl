# QUICK PureBLAS vs Intel MKL comparison, for a machine OUTSIDE the benchmark fleet (an Intel laptop).
#
#     julia -t auto bench/quick_vs_mkl.jl          # from the repository root; a few minutes
#
# It builds its own temporary environment (MKL + Chairmarks + this checkout of PureBLAS), so no project
# file in the repository changes. It times the SAME `LinearAlgebra` call with two backends behind it:
# MKL, forwarded by `using MKL`, and PureBLAS, forwarded by `PureBLAS.activate()`. MKL is loaded first,
# so `deactivate()` hands the calls back to MKL rather than to the OpenBLAS Julia ships with.
#
# THESE ARE INDICATIVE NUMBERS, NOT GATE NUMBERS. The fleet's results come from frequency-locked AMD
# boxes with every thread pinned to its own core; a laptop boosts, throttles, and on recent Intel parts
# mixes performance and efficiency cores. To reduce the noise: plug in, close other work, and run twice.
# The two backends are measured alternately (MKL, PureBLAS, MKL, PureBLAS) so drift hits both equally,
# with a pause before each PureBLAS window so MKL's threads have stopped spinning on the cores.
#
# Reading the ratio: MKL time / PureBLAS time, so ABOVE 1 MEANS PUREBLAS IS FASTER.
using Pkg
Pkg.activate(; temp = true)
Pkg.develop(path = joinpath(@__DIR__, ".."); io = devnull)
Pkg.add(["MKL", "Chairmarks"]; io = devnull)

using MKL                         # forward LinearAlgebra's BLAS/LAPACK to MKL — BEFORE PureBLAS loads
using PureBLAS, LinearAlgebra, Printf, Random
using Chairmarks: @be

const NT = Threads.nthreads()

# FAIL BEFORE TIMING ANYTHING if either backend is not what this script claims to be measuring.
# `deactivate()` re-forwards the libraries that were loaded when PureBLAS started — MKL's here — so check
# that the round trip really lands back on MKL and that both backends compute the right answer.
function check_backends()
    A = randn(300, 200); B = randn(200, 250); ref = [sum(A[i, l] * B[l, j] for l in 1:200) for i in 1:300, j in 1:250]
    PureBLAS.activate()
    PureBLAS.is_active() || error("PureBLAS.activate() did not take effect")
    pb_ok = isapprox(A * B, ref; rtol = 1e-12)
    PureBLAS.deactivate()
    libs = [basename(l.libname) for l in BLAS.get_config().loaded_libs]
    any(contains("mkl"), libs) || error("after deactivate() the loaded BLAS is $libs, not MKL")
    mkl_ok = isapprox(A * B, ref; rtol = 1e-12)
    (pb_ok && mkl_ok) || error("wrong product: PureBLAS ok=$pb_ok, MKL ok=$mkl_ok")
    println("backends checked: PureBLAS active/inactive round trip lands on ", join(libs, ", "), "; both products correct")
end

# One op at one thread count: alternate the two backends, pool each backend's samples, take medians.
function compare(f, setup, nt)
    BLAS.set_num_threads(nt); PureBLAS.set_num_threads(nt)
    PureBLAS.deactivate(); f(setup()); PureBLAS.activate(); f(setup())     # compile both paths
    mkl = Float64[]; pb = Float64[]
    for _ in 1:2
        PureBLAS.deactivate()
        append!(mkl, [s.time for s in (@be setup() f evals = 1 seconds = 0.5).samples])
        sleep(0.3)                                                       # let MKL's threads go idle
        PureBLAS.activate()
        append!(pb, [s.time for s in (@be setup() f evals = 1 seconds = 0.5).samples])
    end
    PureBLAS.deactivate()
    return median(mkl), median(pb)
end
median(v) = (s = sort(v); s[(length(s) + 1) ÷ 2])

function cases()
    rng = MersenneTwister(1)
    c = Tuple{String, Int, Float64, Function, Function}[]   # name, n, flops, setup, call
    for n in (256, 1024, 2048)
        A = randn(rng, n, n); B = randn(rng, n, n)
        S = A' * A + n * I; U = triu(A) + n * I
        push!(c, ("gemm", n, 2.0n^3, () -> zeros(n, n), C -> mul!(C, A, B)))
        push!(c, ("syrk", n, 1.0n^3, () -> zeros(n, n), C -> BLAS.syrk!('U', 'N', 1.0, A, 0.0, C)))
        push!(c, ("trsm", n, 1.0n^3, () -> copy(B), X -> BLAS.trsm!('L', 'U', 'N', 'N', 1.0, U, X)))
        push!(c, ("cholesky", n, n^3 / 3, () -> copy(S), X -> cholesky!(X)))
        push!(c, ("lu", n, 2n^3 / 3, () -> copy(A), X -> lu!(X)))
    end
    for n in (1024, 4096)
        A = randn(rng, n, n); x = randn(rng, n)
        push!(c, ("gemv", n, 2.0n^2, () -> zeros(n), y -> mul!(y, A, x)))
    end
    for n in (10_000, 1_000_000)
        x = randn(rng, n); y = randn(rng, n)
        push!(c, ("dot", n, 2.0n, () -> nothing, _ -> dot(x, y)))
    end
    return c
end

function main()
    println("PureBLAS vs MKL on ", Sys.CPU_NAME, ", ", Sys.CPU_THREADS, " hardware threads, Julia started with ",
            NT, " thread", NT == 1 ? "" : "s", "  (ratio = MKL time / PureBLAS time; >1 means PureBLAS is faster)\n")
    NT == 1 && println("  note: started with one thread, so only the single-threaded column runs. Use `julia -t auto`.\n")
    check_backends(); println()
    threads = NT == 1 ? (1,) : (1, NT)
    @printf("%-10s %8s", "op", "n")
    for nt in threads; @printf("   %18s", "$(nt) thread$(nt == 1 ? "" : "s")"); end
    println()
    for (name, n, fl, setup, f) in cases()
        @printf("%-10s %8d", name, n)
        for nt in threads
            tm, tp = compare(f, setup, nt)
            @printf("   %5.2fx  %6.1f GF/s", tm / tp, fl / tp / 1e9)
        end
        println(); flush(stdout)
    end
    println("\nGF/s is PureBLAS's throughput. Numbers are indicative — see the header of this script.")
end
main()
