using PureBLAS, LinearAlgebra
using LinearAlgebra: BLAS
const T = ComplexF64; const PB = PureBLAS
fails = 0; checks = 0
for herm in (false, true), ul in ('U', 'L'), tr in (herm ? ('N', 'C') : ('N', 'T')),
        n in (1, 3, 4, 7, 15, 32, 64, 100, 127, 128, 130, 200, 255, 256), al in (T(0.9, 0.4), one(T))
    k = n
    A = rand(T, tr == 'N' ? n : k, tr == 'N' ? k : n); Bm = rand(T, size(A)...)
    C0 = rand(T, n, n); C0 = (C0 + C0')
    be = herm ? 0.3 : T(0.5, -0.2)
    C1 = copy(C0); C2 = copy(C0)
    if herm
        PB.her2k!(C1, A, Bm; uplo = ul, trans = tr, alpha = al, beta = real(be))
        BLAS.her2k!(ul, tr, al, A, Bm, real(be), C2)
    else
        PB.syr2k!(C1, A, Bm; uplo = ul, trans = tr, alpha = al, beta = be)
        BLAS.syr2k!(ul, tr, al, A, Bm, be, C2)
    end
    global checks += 1
    dtri = ul == 'U' ? triu(C1 - C2) : tril(C1 - C2)
    e = norm(dtri) / max(norm(ul == 'U' ? triu(C2) : tril(C2)), eps())
    if !(e < 1.0e-11) || any(isnan, C1)
        global fails += 1
        println("  FAIL herm=$herm ul=$ul tr=$tr n=$n a=$al  relerr=$e")
    end
end
println("fused syr2k/her2k correctness: $checks checks, $fails failures  ", fails == 0 ? "ALL PASS" : "*** FAIL ***")
