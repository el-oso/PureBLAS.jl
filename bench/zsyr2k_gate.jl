using PureBLAS, LinearAlgebra, BenchmarkTools
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)
const T = ComplexF64; const PB = PureBLAS
ratio(t_pb, t_ob) = t_ob / t_pb

# correctness
me = 0.0
for herm in (false, true), ul in ('U', 'L'), tr in (herm ? ('N', 'C') : ('N', 'T')), n in (16, 32, 64, 128, 200)
    k = n
    A = rand(T, tr == 'N' ? n : k, tr == 'N' ? k : n); Bm = rand(T, size(A)...)
    C0 = rand(T, n, n); C0 = C0 + C0'  # hermitian-ish start
    al = T(0.9, 0.4); be = herm ? 0.3 : T(0.5, -0.2)
    C1 = copy(C0); C2 = copy(C0)
    if herm
        PB.her2k!(C1, A, Bm; uplo = ul, trans = tr, alpha = al, beta = real(be))
        BLAS.her2k!(ul, tr, al, A, Bm, real(be), C2)
    else
        PB.syr2k!(C1, A, Bm; uplo = ul, trans = tr, alpha = al, beta = be)
        BLAS.syr2k!(ul, tr, al, A, Bm, be, C2)
    end
    # compare only the stored triangle
    tri = ul == 'U' ? triu(C1 - C2) : tril(C1 - C2)
    global me = max(me, norm(tri) / max(norm(ul == 'U' ? triu(C2) : tril(C2)), eps()))
end
println("CORRECTNESS syr2k/her2k relerr=", me, me < 1.0e-11 ? "  PASS" : "  FAIL")

for (nm, herm) in (("zher2k", true), ("zsyr2k", false)), tr in (herm ? ('N', 'C') : ('N', 'T'))
    println("=== $nm trans=$tr ===")
    for n in (8, 16, 32, 64, 128, 256, 512, 1024)
        k = n
        A = rand(T, tr == 'N' ? n : k, tr == 'N' ? k : n); Bm = rand(T, size(A)...)
        C0 = rand(T, n, n)
        al = T(0.9, 0.4); be = herm ? 0.3 : T(0.5, -0.2)
        if herm
            tp = @belapsed PB.her2k!(C, $A, $Bm; uplo = 'U', trans = $tr, alpha = $al, beta = 0.3) setup = (C = copy($C0)) evals = 1
            to = @belapsed BLAS.her2k!('U', $tr, $al, $A, $Bm, 0.3, C) setup = (C = copy($C0)) evals = 1
        else
            tp = @belapsed PB.syr2k!(C, $A, $Bm; uplo = 'U', trans = $tr, alpha = $al, beta = $be) setup = (C = copy($C0)) evals = 1
            to = @belapsed BLAS.syr2k!('U', $tr, $al, $A, $Bm, $be, C) setup = (C = copy($C0)) evals = 1
        end
        println("  n=$n   PB/OB = ", round(ratio(tp, to), digits = 3))
    end
end
