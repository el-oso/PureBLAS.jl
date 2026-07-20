using PureBLAS, LinearAlgebra, BenchmarkTools
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)
const T = ComplexF64; const PB = PureBLAS; ratio(tp, to) = to / tp
tri(n) = (
    A = triu(rand(T, n, n)); for i in 1:n
        A[i, i] = 1 + abs(A[i, i])
    end; A
)
herm(n) = (
    A = rand(T, n, n); A = A + A'; for i in 1:n
        A[i, i] = real(A[i, i])
    end; A
)
ca = T(0.9, 0.4); cb = T(0.5, -0.2)
println("=== ztrmm side=L (plotted) ===")
for n in (8, 16, 32, 48, 64, 96, 128, 192, 256)
    A = tri(n); X = rand(T, n, n)
    tp = @belapsed PB.trmm!(B, $A; side = 'L', uplo = 'U', transA = 'N', alpha = $ca) setup = (B = copy($X)) evals = 1
    to = @belapsed BLAS.trmm!('L', 'U', 'N', 'N', $ca, $A, B) setup = (B = copy($X)) evals = 1
    println("  n=$n  PB/OB=", round(ratio(tp, to), digits = 3))
end
println("=== ztrsm side=L (plotted) ===")
for n in (8, 16, 32, 48, 64, 96, 128, 192, 256)
    A = tri(n); X = rand(T, n, n)
    tp = @belapsed PB.trsm!(B, $A; side = 'L', uplo = 'U', transA = 'N', alpha = $ca) setup = (B = copy($X)) evals = 1
    to = @belapsed BLAS.trsm!('L', 'U', 'N', 'N', $ca, $A, B) setup = (B = copy($X)) evals = 1
    println("  n=$n  PB/OB=", round(ratio(tp, to), digits = 3))
end
println("=== zhemm side=L (plotted) ===")
for n in (8, 16, 32, 48, 64, 96, 128, 192, 256)
    A = herm(n); X = rand(T, n, n); C0 = rand(T, n, n)
    tp = @belapsed PB.hemm!(C, $A, B; side = 'L', uplo = 'U', alpha = $ca, beta = $cb) setup = (C = copy($C0); B = copy($X)) evals = 1
    to = @belapsed BLAS.hemm!('L', 'U', $ca, $A, B, $cb, C) setup = (C = copy($C0); B = copy($X)) evals = 1
    println("  n=$n  PB/OB=", round(ratio(tp, to), digits = 3))
end
