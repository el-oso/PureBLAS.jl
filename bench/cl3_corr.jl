# Quick correctness-only check for the direct-read trmm bases (force ctrmm_direct via LocalPreferences).
using PureBLAS, LinearAlgebra, Printf
import LinearAlgebra.BLAS as B
function main()
    println("_CTRMM_DIRECT = ", PureBLAS._CTRMM_DIRECT, "  W=", PureBLAS._vwidth(Float64))
    me = 0.0; ok = true
    for Tc in (ComplexF64, ComplexF32), side in ('L','R'), uplo in ('U','L'), transA in ('N','T','C'), diag in ('N','U'),
            (k,n) in ((1,3),(4,4),(7,5),(8,8),(15,9),(16,20),(17,7),(32,13),(33,32),(64,17),(100,40),(128,20))
        tol = Tc==ComplexF64 ? 1e-11 : 1e-4
        A = rand(Tc, k, k) ./ k; for i in 1:k; A[i,i] = 1 + abs(A[i,i]); end
        Bm = side=='L' ? rand(Tc, k, n) : rand(Tc, n, k)
        reft = B.trmm(side, uplo, transA, diag, one(Tc), A, copy(Bm))
        p = copy(Bm); PureBLAS.trmm!(p, A; side=side, uplo=uplo, transA=transA, diag=diag, alpha=one(Tc))
        e = maximum(abs, p - reft) / max(1, maximum(abs, reft)); me = max(me, e)
        if e > tol
            ok = false; @printf("FAIL trmm %s side=%c uplo=%c tA=%c diag=%c k=%d n=%d err=%.2e\n", Tc, side, uplo, transA, diag, k, n, e)
        end
    end
    # zgemm regression check (shares the kernel)
    for Tc in (ComplexF64, ComplexF32), (m,n,k) in ((16,16,16),(31,17,23),(64,64,64),(100,50,40))
        A = rand(Tc,m,k); Bb = rand(Tc,n,k); C = rand(Tc,m,n)
        ref = A*Bb'; p = copy(C); PureBLAS.gemm!(p, A, Bb; transB='C', alpha=one(Tc), beta=zero(Tc))
        e = maximum(abs, p - ref)/max(1,maximum(abs,ref)); me = max(me, e)
        e > (Tc==ComplexF64 ? 1e-11 : 1e-4) && (ok=false; @printf("FAIL zgemm %s m=%d n=%d k=%d err=%.2e\n", Tc,m,n,k,e))
    end
    println(ok ? @sprintf("ALL CORRECT (maxerr %.2e)", me) : ">>> FAILURES <<<")
end
main()
