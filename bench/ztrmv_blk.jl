using PureBLAS, LinearAlgebra, BenchmarkTools
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)
const T = ComplexF64; const PB = PureBLAS
ratio(tp, to) = to / tp
tri(n, ul) = (
    A = ul == 'U' ? triu(rand(T, n, n)) : tril(rand(T, n, n)); for i in 1:n
        A[i, i] = 1 + abs(A[i, i])
    end; A
)

# correctness across uplo×trans×diag×n (blocked crossover at 512 on AVX2)
me = 0.0; nf = 0
for op in (:trmv, :trsv), ul in ('U', 'L'), ta in ('N', 'T', 'C'), dg in ('N', 'U'), n in (7, 63, 64, 65, 300, 512, 513, 1024, 2048)
    A = tri(n, ul); x0 = rand(T, n)
    x1 = copy(x0); op == :trmv ? PB.trmv!(A, x1; uplo = ul, trans = ta, diag = dg) : PB.trsv!(A, x1; uplo = ul, trans = ta, diag = dg)
    x2 = copy(x0); op == :trmv ? BLAS.trmv!(ul, ta, dg, A, x2) : BLAS.trsv!(ul, ta, dg, A, x2)
    e = norm(x1 - x2) / max(norm(x2), eps())
    (e < 1.0e-10 && !any(isnan, x1)) || (global nf += 1; println("  FAIL $op ul=$ul ta=$ta dg=$dg n=$n relerr=$e"))
    global me = max(me, e)
end
println("CORRECTNESS trmv/trsv: maxrelerr=$me  $(nf == 0 ? "PASS" : "*** $nf FAIL ***")")

for (nm, fp, fo) in (
        ("ztrmv", (A, x) -> PB.trmv!(A, x; uplo = 'U'), (A, x) -> BLAS.trmv!('U', 'N', 'N', A, x)),
        ("ztrsv", (A, x) -> PB.trsv!(A, x; uplo = 'U'), (A, x) -> BLAS.trsv!('U', 'N', 'N', A, x)),
    )
    println("=== $nm ===")
    for n in (256, 512, 768, 1024, 1536, 2048, 4096)
        A = tri(n, 'U'); x0 = rand(T, n)
        tp = @belapsed $fp($A, X) setup = (X = copy($x0)) evals = 1
        to = @belapsed $fo($A, X) setup = (X = copy($x0)) evals = 1
        println("  n=$n   PB/OB = ", round(ratio(tp, to), digits = 3))
    end
end
