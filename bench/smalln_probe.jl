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
println("=== ztrsv small-n ===")
for n in (8, 16, 32, 48, 64, 96, 128, 192, 256)
    A = tri(n); x0 = rand(T, n)
    tp = @belapsed PB.trsv!($A, X; uplo = 'U') setup = (X = copy($x0)) evals = 1
    to = @belapsed BLAS.trsv!('U', 'N', 'N', $A, X) setup = (X = copy($x0)) evals = 1
    println("  n=$n  PB/OB=", round(ratio(tp, to), digits = 3))
end
println("=== zher2k small-n (trans=N) ===")
for n in (8, 16, 32, 48, 64)
    A = rand(T, n, n); B = rand(T, n, n); C0 = rand(T, n, n); al = T(0.9, 0.4)
    tp = @belapsed PB.her2k!(C, $A, $B; uplo = 'U', trans = 'N', alpha = $al, beta = 0.3) setup = (C = copy($C0)) evals = 1
    to = @belapsed BLAS.her2k!('U', 'N', $al, $A, $B, 0.3, C) setup = (C = copy($C0)) evals = 1
    println("  n=$n  PB/OB=", round(ratio(tp, to), digits = 3))
end
println("=== zherk small-n (trans=N) ===")
for n in (8, 16, 32, 48, 64)
    A = rand(T, n, n); C0 = rand(T, n, n)
    tp = @belapsed PB.herk!(C, $A; uplo = 'U', trans = 'N', alpha = 1.0, beta = 0.0) setup = (C = copy($C0)) evals = 1
    to = @belapsed BLAS.herk!('U', 'N', 1.0, $A, 0.0, C) setup = (C = copy($C0)) evals = 1
    println("  n=$n  PB/OB=", round(ratio(tp, to), digits = 3))
end
