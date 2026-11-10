# QUICK PureBLAS vs Intel MKL comparison, for a machine OUTSIDE the benchmark fleet (an Intel laptop).
#
#     julia -t auto bench/quick_vs_mkl.jl          # from the repository root; a few minutes
#     julia> include("bench/quick_vs_mkl.jl")      # or from a REPL started with `julia -t auto`
#
# It builds its own temporary environment (MKL + Chairmarks + this checkout of PureBLAS) and switches
# back to the caller's environment once the packages are loaded, so no project file changes. It times
# the SAME `LinearAlgebra` call with two backends behind it: MKL, forwarded by `using MKL`, and
# PureBLAS, forwarded by `PureBLAS.activate()`. Each switch back to MKL is `PureBLAS.deactivate()`
# followed by `MKL.__init__()`: MKL's ILP64 symbols carry a `64_` suffix, and only MKL's own forwarding
# (`suffix_hint = "64"`) finds them.
#
# MKL's thread count is set with MKL's own `MKL_Set_Num_Threads`, which covers all of MKL.
# `BLAS.set_num_threads` reaches MKL through `MKL_Domain_Set_Num_Threads`, and MKL's LAPACK routines
# (Cholesky among them) do not follow it: after `BLAS.set_num_threads(1)`, `potrf` still runs on every core.
#
# THESE ARE INDICATIVE NUMBERS, NOT GATE NUMBERS. The fleet's results come from frequency-locked AMD
# boxes with every thread pinned to its own core; a laptop boosts, throttles, and on recent Intel parts
# mixes performance and efficiency cores. To reduce the noise: plug in, close other work, and run twice.
# The two backends are measured alternately (MKL, PureBLAS, MKL, PureBLAS) so drift hits both equally,
# with a pause before each PureBLAS window so MKL's threads have stopped spinning on the cores.
#
# Reading the ratio: MKL time / PureBLAS time, so ABOVE 1 MEANS PUREBLAS IS FASTER.
using Pkg
let prev = Base.active_project()
    Pkg.activate(; temp = true, io = devnull)
    Pkg.develop(path = joinpath(@__DIR__, ".."); io = devnull)
    Pkg.add(["MKL", "Chairmarks"]; io = devnull)
    @eval Main begin
        using MKL                 # forward LinearAlgebra's BLAS/LAPACK to MKL — BEFORE PureBLAS loads
        using PureBLAS, LinearAlgebra, Printf, Random
        using Chairmarks: @be
    end
    Pkg.activate(prev; io = devnull)
end

const NT = Threads.nthreads()

mkl_set_threads(n) = ccall((:MKL_Set_Num_Threads, MKL.MKL_jll.libmkl_rt), Cvoid, (Cint,), n)

# Back to MKL at `nt` threads.
function use_mkl(nt = NT)
    PureBLAS.deactivate(); MKL.__init__()
    mkl_set_threads(nt); BLAS.set_num_threads(nt)
end

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

# One op at one thread count: alternate the two backends, pool each backend's samples, take medians.
# `mut`: the call overwrites its own input (trsm, potrf, getrf), so every evaluation needs the refill
# `setup` does and a sample is one evaluation. Otherwise Chairmarks picks the evaluation count, which
# matters below about 30 µs, where one evaluation per sample is dominated by timer overhead.
function compare(f, setup, nt; mut = true)
    PureBLAS.set_num_threads(nt)
    use_mkl(nt); f(setup()); PureBLAS.activate(); f(setup())               # compile both paths
    times() = [s.time for s in (mut ? (@be setup() f evals = 1 seconds = 0.5) : (@be setup() f seconds = 0.5)).samples]
    mkl = Float64[]; pb = Float64[]
    for _ in 1:2
        use_mkl(nt)
        append!(mkl, times())
        sleep(0.3)                                                       # let MKL's threads go idle
        PureBLAS.activate()
        append!(pb, times())
    end
    use_mkl()
    return median(mkl), median(pb)
end
median(v) = (s = sort(v); s[(length(s) + 1) ÷ 2])

# Every case owns ONE output buffer, allocated here; `setup` only refills it with `copyto!`, so no sample
# allocates and every sample works on the same memory. The factorizations are called in their
# non-allocating forms, `potrf!` and `getrf!(A, ipiv)` with a preallocated pivot vector: `cholesky!` and
# `lu!` allocate a result wrapper and a pivot vector, and `cholesky!` also runs an O(n^2) symmetry check.
function cases()
    rng = MersenneTwister(1)
    c = Tuple{String, Int, Float64, Function, Function}[]   # name, n, flops, setup, call
    for n in (256, 1024, 2048)
        A = randn(rng, n, n); B = randn(rng, n, n)
        S = A' * A + n * I; U = triu(A) + n * I
        C = zeros(n, n); X = similar(A); ipiv = Vector{Int}(undef, n)
        push!(c, ("gemm", n, 2.0n^3, () -> C, C -> mul!(C, A, B)))
        push!(c, ("syrk", n, 1.0n^3, () -> C, C -> BLAS.syrk!('U', 'N', 1.0, A, 0.0, C)))
        push!(c, ("trsm", n, 1.0n^3, () -> copyto!(X, B), X -> BLAS.trsm!('L', 'U', 'N', 'N', 1.0, U, X)))
        push!(c, ("potrf", n, n^3 / 3, () -> copyto!(X, S), X -> LAPACK.potrf!('U', X)))
        push!(c, ("getrf", n, 2n^3 / 3, () -> copyto!(X, A), X -> LAPACK.getrf!(X, ipiv)))
    end
    # The factorizations again at sizes that are not powers of two. At 1024/2048 PureBLAS's getrf and
    # potrf each copy the matrix into a padded buffer and back (power-of-two leading dimensions alias in
    # cache); at 1000/2000 neither does, so the two pairs of rows separate the copy's cost from the rest.
    for n in (1000, 2000)
        A = randn(rng, n, n); S = A' * A + n * I; X = similar(A); ipiv = Vector{Int}(undef, n)
        push!(c, ("potrf", n, n^3 / 3, () -> copyto!(X, S), X -> LAPACK.potrf!('U', X)))
        push!(c, ("getrf", n, 2n^3 / 3, () -> copyto!(X, A), X -> LAPACK.getrf!(X, ipiv)))
    end
    for n in (1024, 4096)
        A = randn(rng, n, n); x = randn(rng, n); y = zeros(n)
        push!(c, ("gemv", n, 2.0n^2, () -> y, y -> mul!(y, A, x)))
    end
    for n in (10_000, 1_000_000)
        x = randn(rng, n); y = randn(rng, n)
        push!(c, ("dot", n, 2.0n, () -> nothing, _ -> dot(x, y)))
    end
    return c
end

function main()
    println("PureBLAS vs MKL on ", Sys.CPU_NAME, ", ", Sys.CPU_THREADS, " hardware threads, Julia started with ",
            NT, " thread", NT == 1 ? "" : "s", "  (ratio = MKL time / PureBLAS time; >1 means PureBLAS is faster)")
    println("PureBLAS detected ", PureBLAS._HW)
    NT == 1 && println("  note: started with one thread, so only the single-threaded column runs. Use `julia -t auto`.")
    check_backends(); println()
    # PureBLAS uses at most one worker per physical core; MKL gets the same count, which is also its default.
    threads = unique((1, PureBLAS.set_num_threads(NT)))
    NT > last(threads) && println("  note: $NT Julia threads, $(last(threads)) physical cores; both libraries use $(last(threads)).")
    @printf("%-10s %8s", "op", "n")
    for nt in threads; @printf("   %18s", "$(nt) thread$(nt == 1 ? "" : "s")"); end
    println()
    for (name, n, fl, setup, f) in cases()
        @printf("%-10s %8d", name, n)
        for nt in threads
            tm, tp = compare(f, setup, nt; mut = name in ("trsm", "potrf", "getrf"))
            @printf("   %5.2fx  %6.1f GF/s", tm / tp, fl / tp / 1e9)
        end
        println(); flush(stdout)
    end
    println("\nGF/s is PureBLAS's throughput. Numbers are indicative — see the header of this script.")
end
@isdefined(QUICK_VS_MKL_NO_MAIN) || main()    # quick_chol_diag.jl defines it to reuse the functions above
