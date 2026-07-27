# zgeqrf gate sweep vs OpenBLAS across the complex-LAPACK sizes. Interleaved paired median (cancels
# freq drift), destructive → untimed reset. Prints our/OB ratio; flags < 1.0. Correctness vs LAPACK.
using PureBLAS, LinearAlgebra, Statistics, Printf, Random
BLAS.set_num_threads(1)
const S = Ref(0.0); @noinline _run(f) = f()
const T = ComplexF64

function _stable(ref, our, reset; rounds = 11, tol = 0.02, cap = 41)
    reset(); _run(ref); reset(); _run(our)
    rs = Float64[]
    while true
        for _ in 1:rounds
            reset(); t0 = time_ns(); _run(ref); t1 = time_ns()
            reset(); t2 = time_ns(); _run(our); t3 = time_ns()
            push!(rs, (t1 - t0) / (t3 - t2))
        end
        m = median(rs)
        ((quantile(rs, 0.75) - quantile(rs, 0.25)) / m < tol || length(rs) >= cap) && return m
    end
end

sizes = isempty(ARGS) ? [8, 32, 128, 256, 512, 1024, 2048] : parse.(Int, split(ARGS[1], ","))
@printf("%-6s %-9s %-8s %s\n", "n", "relerr", "our/OB", "flag")
for n in sizes
    Random.seed!(1234 + n)
    A0 = randn(T, n, n); Aw = similar(A0); reset = () -> copyto!(Aw, A0)
    # correctness
    F = copy(A0); tau = zeros(T, n); PureBLAS.geqrf!(F, tau); Fl = copy(A0); LAPACK.geqrf!(Fl)
    er = maximum(abs, triu(F) - triu(Fl)) / max(maximum(abs, Fl), eps())
    ref = () -> (LinearAlgebra.LAPACK.geqrf!(Aw); S[] += real(Aw[1,1]))
    our = () -> (PureBLAS.geqrf!(Aw); S[] += real(Aw[1,1]))
    r = _stable(ref, our, reset)
    @printf("%-6d %.2e  %.3f    %s\n", n, er, r, r < 1.0 ? "BELOW" : "gate")
end
S[] == -1.0 && println(S[])
