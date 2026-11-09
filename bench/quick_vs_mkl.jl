# QUICK PureBLAS vs Intel MKL comparison, for a machine OUTSIDE the benchmark fleet (an Intel laptop).
#
#     julia -t auto bench/quick_vs_mkl.jl          # from the repository root; a few minutes
#
# It builds its own temporary environment (MKL + Chairmarks + this checkout of PureBLAS), so no project
# file in the repository changes. It times the SAME `LinearAlgebra` call with two backends behind it:
# MKL, forwarded by `using MKL`, and PureBLAS, forwarded by `PureBLAS.activate()`. Each switch back to
# MKL is `PureBLAS.deactivate()` followed by `MKL.__init__()`: MKL's ILP64 symbols carry a `64_` suffix,
# and only MKL's own forwarding (`suffix_hint = "64"`) finds them.
#
# THESE ARE INDICATIVE NUMBERS, NOT GATE NUMBERS. The fleet's results come from frequency-locked AMD
# boxes with every thread pinned to its own core; a laptop boosts, throttles, and on recent Intel parts
# mixes performance and efficiency cores. To reduce the noise: plug in, close other work, and run twice.
# The two backends are measured alternately (MKL, PureBLAS, MKL, PureBLAS) so drift hits both equally,
# with a pause before each PureBLAS window so MKL's threads have stopped spinning on the cores.
#
# EACH THREAD COUNT RUNS IN ITS OWN JULIA PROCESS, started with `-t N` and `MKL_NUM_THREADS=N`. MKL's
# LAPACK routines (Cholesky among them) keep the thread count MKL started with: `BLAS.set_num_threads(1)`
# reports 1 but `potrf` still runs on every core. Only the variable set at startup holds it to N.
#
# Reading the ratio: MKL time / PureBLAS time, so ABOVE 1 MEANS PUREBLAS IS FASTER.
const NT_CHILD = get(ENV, "QUICK_VS_MKL_THREADS", "")
if isempty(NT_CHILD)        # parent: build the environment once, then run one child per thread count
    using Pkg
    Pkg.activate(; temp = true)
    Pkg.develop(path = joinpath(@__DIR__, ".."); io = devnull)
    Pkg.add(["MKL", "Chairmarks"]; io = devnull)
    nmax = Threads.nthreads()
    nmax == 1 && println("note: started with one thread, so only the single-threaded table runs. Use `julia -t auto`.\n")
    for nt in unique((1, nmax))
        cmd = `$(Base.julia_cmd()) -t $nt --project=$(Base.active_project()) $(abspath(PROGRAM_FILE))`
        run(addenv(cmd, "QUICK_VS_MKL_THREADS" => string(nt), "MKL_NUM_THREADS" => string(nt)))
    end
    exit()
end

using MKL                         # forward LinearAlgebra's BLAS/LAPACK to MKL — BEFORE PureBLAS loads
using PureBLAS, LinearAlgebra, Printf, Random
using Chairmarks: @be

const NT = parse(Int, NT_CHILD)
NT == Threads.nthreads() || error("child started with $(Threads.nthreads()) Julia threads, expected $NT")

use_mkl() = (PureBLAS.deactivate(); MKL.__init__())

# FAIL BEFORE TIMING ANYTHING if either backend is not what this script claims to be measuring:
# check that the round trip lands back on MKL and that both backends compute the right answer.
function check_backends()
    A = randn(300, 200); B = randn(200, 250); ref = [sum(A[i, l] * B[l, j] for l in 1:200) for i in 1:300, j in 1:250]
    PureBLAS.activate()
    PureBLAS.is_active() || error("PureBLAS.activate() did not take effect")
    pb_ok = isapprox(A * B, ref; rtol = 1e-12)
    use_mkl()
    libs = [basename(l.libname) for l in BLAS.get_config().loaded_libs]
    any(contains("mkl"), libs) || error("after use_mkl() the loaded BLAS is $libs, not MKL")
    mkl_ok = isapprox(A * B, ref; rtol = 1e-12)
    (pb_ok && mkl_ok) || error("wrong product: PureBLAS ok=$pb_ok, MKL ok=$mkl_ok")
    println("backends checked: PureBLAS active/inactive round trip lands on ", join(libs, ", "), "; both products correct")
end

# One op at this process's thread count: alternate the two backends, pool each backend's samples, take medians.
function compare(f, setup)
    use_mkl(); BLAS.set_num_threads(NT); PureBLAS.set_num_threads(NT)
    f(setup()); PureBLAS.activate(); f(setup())                            # compile both paths
    mkl = Float64[]; pb = Float64[]
    for _ in 1:2
        use_mkl()
        append!(mkl, [s.time for s in (@be setup() f evals = 1 seconds = 0.5).samples])
        sleep(0.3)                                                       # let MKL's threads go idle
        PureBLAS.activate()
        append!(pb, [s.time for s in (@be setup() f evals = 1 seconds = 0.5).samples])
    end
    use_mkl()
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
    println("PureBLAS vs MKL on ", Sys.CPU_NAME, ", ", Sys.CPU_THREADS, " hardware threads — ", NT, " thread",
            NT == 1 ? "" : "s", "  (ratio = MKL time / PureBLAS time; >1 means PureBLAS is faster)")
    println("PureBLAS detected ", PureBLAS._HW)
    check_backends()
    @printf("%-10s %8s %8s %12s\n", "op", "n", "ratio", "PB GF/s")
    for (name, n, fl, setup, f) in cases()
        tm, tp = compare(f, setup)
        @printf("%-10s %8d %7.2fx %12.1f\n", name, n, tm / tp, fl / tp / 1e9); flush(stdout)
    end
    println()
end
abspath(PROGRAM_FILE) == @__FILE__() && main()      # `include`d by quick_chol_diag.jl without running
