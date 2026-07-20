using PureBLAS, LinearAlgebra, BenchmarkTools
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)
const T = ComplexF64; const PB = PureBLAS
ratio(t_pb, t_ob) = t_ob / t_pb

# correctness: herk/syrk, all uplo×trans, incl. non-mult-of-4 n (remainder tiles)
me = 0.0
for herm in (true, false), ul in ('U', 'L'), tr in (herm ? ('N', 'C') : ('N', 'T')), n in (1, 4, 7, 32, 100, 128, 130, 200)
    k = n
    A = rand(T, tr == 'N' ? n : k, tr == 'N' ? k : n)
    C0 = rand(T, n, n)
    C1 = copy(C0); C2 = copy(C0)
    if herm
        PB.herk!(C1, A; uplo = ul, trans = tr, alpha = 1.0, beta = 0.0)
        BLAS.herk!(ul, tr, 1.0, A, 0.0, C2)
    else
        PB.syrk!(C1, A; uplo = ul, trans = tr, alpha = one(T), beta = zero(T))
        BLAS.syrk!(ul, tr, one(T), A, zero(T), C2)
    end
    tri = ul == 'U' ? triu(C1 - C2) : tril(C1 - C2)
    global me = max(me, norm(tri) / max(norm(ul == 'U' ? triu(C2) : tril(C2)), eps()))
end
println("CORRECTNESS herk/syrk relerr=", me, me < 1.0e-11 ? "  PASS" : "  FAIL")

for (nm, herm) in (("zherk", true), ("zsyrk", false)), tr in (herm ? ('N', 'C') : ('N', 'T'))
    println("=== $nm trans=$tr ===")
    for n in (32, 64, 96, 128, 192, 256, 512, 1024)
        k = n
        A = rand(T, tr == 'N' ? n : k, tr == 'N' ? k : n); C0 = rand(T, n, n)
        if herm
            tp = @belapsed PB.herk!(C, $A; uplo = 'U', trans = $tr, alpha = 1.0, beta = 0.0) setup = (C = copy($C0)) evals = 1
            to = @belapsed BLAS.herk!('U', $tr, 1.0, $A, 0.0, C) setup = (C = copy($C0)) evals = 1
        else
            tp = @belapsed PB.syrk!(C, $A; uplo = 'U', trans = $tr, alpha = one($T), beta = zero($T)) setup = (C = copy($C0)) evals = 1
            to = @belapsed BLAS.syrk!('U', $tr, one($T), $A, zero($T), C) setup = (C = copy($C0)) evals = 1
        end
        println("  n=$n   PB/OB = ", round(ratio(tp, to), digits = 3))
    end
end
