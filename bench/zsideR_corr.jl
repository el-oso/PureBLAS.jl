using PureBLAS, LinearAlgebra
using LinearAlgebra: BLAS
const T = ComplexF64; const PB = PureBLAS
fails = 0; checks = 0
function chk(tag, B1, B2)
    global checks += 1
    e = norm(B1 - B2) / max(norm(B2), eps())
    return if !(e < 1.0e-10) || any(isnan, B1)
        global fails += 1
        println("  FAIL $tag  relerr=$e")
    end
end
for op in (:trmm, :trsm), ul in ('U', 'L'), ta in ('N', 'T', 'C'), dg in ('N', 'U'),
        mk in ((0, 0), (1, 1), (3, 3), (64, 64), (128, 128), (200, 96), (96, 200), (300, 128)), strided in (false, true), al in (one(T), T(0.7, -0.3))
    m, k = mk
    A = (ul == 'U' ? triu(rand(T, k, k)) : tril(rand(T, k, k))) + (k + 1) * I
    dg == 'U' && k > 0 && (A[diagind(A)] .= one(T))
    if strided
        # B is a view with row stride 2 (non-contiguous → routes to base_R path)
        Braw = rand(T, 2m + 1, k); Bv = @view Braw[1:2:2m, :]
        X = collect(Bv)
        B1 = copy(Bv); B2 = collect(X)
        op == :trmm ? PB.trmm!(B1, A; side = 'R', uplo = ul, transA = ta, diag = dg, alpha = al) :
            PB.trsm!(B1, A; side = 'R', uplo = ul, transA = ta, diag = dg, alpha = al)
        op == :trmm ? BLAS.trmm!('R', ul, ta, dg, al, A, B2) :
            BLAS.trsm!('R', ul, ta, dg, al, A, B2)
        chk("$op ul=$ul ta=$ta dg=$dg m=$m k=$k strided a=$al", collect(B1), B2)
    else
        X = rand(T, m, k)
        B1 = copy(X); B2 = copy(X)
        op == :trmm ? PB.trmm!(B1, A; side = 'R', uplo = ul, transA = ta, diag = dg, alpha = al) :
            PB.trsm!(B1, A; side = 'R', uplo = ul, transA = ta, diag = dg, alpha = al)
        op == :trmm ? BLAS.trmm!('R', ul, ta, dg, al, A, B2) :
            BLAS.trsm!('R', ul, ta, dg, al, A, B2)
        chk("$op ul=$ul ta=$ta dg=$dg m=$m k=$k a=$al", B1, B2)
    end
end
println("side-R correctness: $checks checks, $fails failures  ", fails == 0 ? "ALL PASS" : "*** FAILURES ***")
