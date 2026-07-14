using PureBLAS, LinearAlgebra, Statistics, Printf
import LinearAlgebra.BLAS as B
BLAS.set_num_threads(1); const T = ComplexF64; const P = PureBLAS; const S = Ref(ComplexF64(0)); @noinline r(f) = f()
function rat(n; rounds = 15)
    rs = Float64[]
    for _ in 1:rounds
        A = rand(T, n, n); Bb = rand(T, n, n); C = zeros(T, n, n); reps = clamp(20_000_000 ÷ n^3, 1, 200)
        r(() -> B.gemm!('N', 'N', one(T), A, Bb, zero(T), C)); r(() -> P.gemm!(C, A, Bb; alpha = one(T), beta = zero(T)))
        t0 = time_ns(); for _ in 1:reps; B.gemm!('N', 'N', one(T), A, Bb, zero(T), C); end; S[] += C[1]; t1 = time_ns()
        for _ in 1:reps; P.gemm!(C, A, Bb; alpha = one(T), beta = zero(T)); end; S[] += C[1]; t2 = time_ns()
        push!(rs, (t1 - t0) / (t2 - t1))
    end
    median(rs)
end
print("zgemm"); for n in (16, 32, 64, 128, 256); @printf(" %d=%.3f", n, rat(n)); end; println()
