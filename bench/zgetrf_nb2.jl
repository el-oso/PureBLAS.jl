# nb sweep WITH rank-2 panel: find complex getrf nb optima per n.
using PureBLAS, LinearAlgebra, Statistics, Printf
import LinearAlgebra.LAPACK as LA
BLAS.set_num_threads(1); const SINK = Ref(0.0)
const T = ComplexF64
rep3(n) = clamp(30_000_000 ÷ (n^3), 1, 300)
function gate_nb(n, nb, rounds)
    reps = rep3(n); rs = Float64[]
    for _ in 1:rounds
        base = randn(T, n, n)
        obb = [copy(base) for _ in 1:reps]; pbb = [copy(base) for _ in 1:reps]
        obp = [Vector{Int}(undef, n) for _ in 1:reps]; pbp = [Vector{Int}(undef, n) for _ in 1:reps]
        t0 = time_ns(); for r in 1:reps; LA.getrf!(obb[r], obp[r]); end; t1 = time_ns()
        for r in 1:reps; PureBLAS.getrf!(pbb[r], pbp[r]; nb=nb); end; t2 = time_ns()
        SINK[] += real(obb[1][1]) + real(pbb[1][1]); push!(rs, (t1-t0)/(t2-t1))
    end
    median(rs)
end
for (n, nbs) in ((128,(32,48,64)), (256,(32,48,64,96)), (512,(48,64,96,128,160)),
                 (1024,(96,128,160,192,256)), (2048,(128,160,192,256,320)))
    print(rpad("n=$n", 8))
    for nb in nbs
        r = gate_nb(n, nb, 13); @printf(" nb%d=%.3f%s", nb, r, r<1.0 ? "*" : " "); flush(stdout)
    end
    println()
end
