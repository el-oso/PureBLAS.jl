using LinearAlgebra, Printf; import PureBLAS
BLAS.set_num_threads(1)
function main()
worst = 0.0; nfail = 0
for T in (Float64, Float32)
    tol = T === Float64 ? 1e-12 : 1e-4
    for uplo in ('U','L'), transA in ('N','T'), diag in ('N','U'), side in ('L','R')
        for k in (1,2,7,8,15,16,31,32,33,40,63,64,65,100,127,128,129,200,256,300)
            for nc in (1,3,8,40,64,65,128,200)   # B width (side L: k×nc ; side R: nc×k)
                A = (uplo=='U' ? triu(randn(T,k,k)) : tril(randn(T,k,k)))
                @inbounds for i in 1:k; A[i,i] = (diag=='U' ? one(T) : A[i,i]+T(k)); end
                B = side=='L' ? randn(T,k,nc) : randn(T,nc,k)
                X = copy(B)
                PureBLAS.trsm!(X, A; side=side, uplo=uplo, transA=transA, diag=diag)
                Xr = copy(B); BLAS.trsm!(side, uplo, transA, diag, one(T), A, Xr)
                r = norm(X-Xr)/max(norm(Xr), eps(T))
                worst = max(worst, r)
                if r > tol || isnan(r)
                    nfail += 1
                    nfail <= 20 && @printf("FAIL %s side=%c uplo=%c tA=%c diag=%c k=%d nc=%d rel=%.2e\n", T, side, uplo, transA, diag, k, nc, r)
                end
            end
        end
    end
end
# strided-parent B (po2 lda stress): the gate shape on a padded parent
for T in (Float64,), k in (128, 256), nc in (256,)
    P = randn(T, k+7, nc); B = view(P, 1:k, :)
    A = triu(randn(T,k,k)); for i in 1:k; A[i,i]+=T(k); end
    X = copy(Matrix(B)); PureBLAS.trsm!(view(P,1:k,:), A; side='L', uplo='U', transA='N')
    Xr = copy(X); BLAS.trsm!('L','U','N','N', one(T), A, Xr)
    r = norm(Matrix(view(P,1:k,:))-Xr)/norm(Xr); worst=max(worst,r)
    r > 1e-12 && (nfail+=1; @printf("FAIL strided k=%d rel=%.2e\n", k, r))
end
@printf("\nCORRECTNESS worst_rel=%.3e nfail=%d\n", worst, nfail)
end
main()
