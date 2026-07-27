# Probe: blocked (nb=32) vs single-panel unblocked (nb=n) for zgeqrf across the transition sizes, to
# find the unblocked→blocked crossover NX. our/OB interleaved median.
using PureBLAS, LinearAlgebra, Statistics, Printf, Random
BLAS.set_num_threads(1)
@noinline _run(f) = f()
const T = ComplexF64
function _stable(ref, our, reset; rounds=11, tol=0.02, cap=41)
    reset(); _run(ref); reset(); _run(our); rs = Float64[]
    while true
        for _ in 1:rounds
            reset(); t0=time_ns(); _run(ref); t1=time_ns()
            reset(); t2=time_ns(); _run(our); t3=time_ns()
            push!(rs, (t1-t0)/(t3-t2))
        end
        m = median(rs)
        ((quantile(rs,0.75)-quantile(rs,0.25))/m < tol || length(rs)>=cap) && return m
    end
end
@printf("%-6s %-10s %-10s\n", "n", "blk(nb32)", "single")
for n in [40,48,56,64,80,96,112,128,160,192,256]
    Random.seed!(1234+n); A0 = randn(T,n,n); Aw = similar(A0); tau = zeros(T,n)
    reset = () -> copyto!(Aw, A0)
    ref = () -> LinearAlgebra.LAPACK.geqrf!(Aw)
    rb = _stable(ref, () -> PureBLAS.geqrf!(Aw, tau; nb=32), reset)
    rs = _stable(ref, () -> PureBLAS.geqrf!(Aw, tau; nb=n), reset)
    @printf("%-6d %-10.3f %-10.3f\n", n, rb, rs)
end
