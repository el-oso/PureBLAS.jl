# A/B: zgetrf gate under different nb settings, plus rank-k zgemm ratio at trailing shapes.
using PureBLAS, LinearAlgebra, Statistics, Printf
import LinearAlgebra.BLAS as B
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

# rank-k zgemm ratio: C(m×n) -= A(m×k) * B(k×n), m=n given, sweep k
function gemmk(n, k, rounds)
    reps = clamp(15_000_000 ÷ (n*n*k), 1, 200); rs = Float64[]
    for _ in 1:rounds
        A = randn(T, n, k); Bm = randn(T, k, n); C = randn(T, n, n)
        t0 = time_ns(); for _ in 1:reps; B.gemm!('N','N', T(-1), A, Bm, T(1), C); end; t1 = time_ns()
        for _ in 1:reps; PureBLAS.gemm!(C, A, Bm; alpha=-1, beta=true); end; t2 = time_ns()
        SINK[] += real(C[1]); push!(rs, (t1-t0)/(t2-t1))
    end
    median(rs)
end

println("== rank-k zgemm OB/PB (m=n, trailing shape) ==")
for n in (512, 1024, 2048)
    print(rpad("n=$n", 8))
    for k in (48, 64, 96, 128, 192, 256)
        r = gemmk(n, k, 11); @printf(" k%d=%.3f%s", k, r, r<1.0 ? "*" : " "); flush(stdout)
    end
    println()
end

println("== zgetrf gate under nb ==")
for n in (128, 256, 512, 1024, 2048)
    print(rpad("n=$n", 8))
    for nb in (48, 64, 96, 128, 192)
        r = gate_nb(n, nb, 11); @printf(" nb%d=%.3f%s", nb, r, r<1.0 ? "*" : " "); flush(stdout)
    end
    println()
end
