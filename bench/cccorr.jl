# Correctness: complex trsv/trmv (+ real for the scatter path) vs OpenBLAS oracle, all uplo/trans/diag.
using PureBLAS, LinearAlgebra, Printf
import LinearAlgebra.BLAS as B
maxerr=0.0
for TT in (ComplexF64, ComplexF32)
    tol = TT===ComplexF64 ? 1e-10 : 1e-3
    for n in (7, 33, 64, 65, 129, 256, 257, 512, 1024, 2048)
        # well-conditioned triangular for BOTH diag modes: scale off-diagonal small (unit diag forces
        # diagonal=1, so O(1) off-diag would be exponentially ill-conditioned → F32 overflow). Mirrors plots.jl.
        A = randn(TT,n,n) ./ (2n); for i in 1:n; A[i,i]=1+abs(A[i,i]); end
        for uplo in ('U','L'), trans in ('N','T','C'), diag in ('N','U')
            x0 = randn(TT,n)
            # trmv
            xp = copy(x0); xo = copy(x0)
            PureBLAS.trmv!(A, xp; uplo=uplo, trans=trans, diag=diag)
            B.trmv!(uplo, trans, diag, A, xo)
            e = norm(xp-xo)/(norm(xo)+eps()); global maxerr=max(maxerr,e)
            e>tol && @printf("TRMV FAIL %s n=%d %c%c%c err=%.2e\n", TT, n, uplo,trans,diag, e)
            # trsv
            xp = copy(x0); xo = copy(x0)
            PureBLAS.trsv!(A, xp; uplo=uplo, trans=trans, diag=diag)
            B.trsv!(uplo, trans, diag, A, xo)
            e = norm(xp-xo)/(norm(xo)+eps()); global maxerr=max(maxerr,e)
            e>tol && @printf("TRSV FAIL %s n=%d %c%c%c err=%.2e\n", TT, n, uplo,trans,diag, e)
        end
    end
end
@printf("max rel err = %.3e\n", maxerr)
println(maxerr < 1e-10 ? "F64 boundary OK (<1e-10)" : "CHECK: exceeds 1e-10 (F32 tol looser)")
