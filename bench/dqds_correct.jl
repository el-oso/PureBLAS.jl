# Correctness of the dqds values path vs LinearAlgebra.svdvals, real+complex, square+rectangular.
using PureBLAS, LinearAlgebra, Printf
const PB = PureBLAS

function checkone(A)
    m,n = size(A)
    S = Vector{real(eltype(A))}(undef, min(m,n))
    PB.gesvd_vals!(copy(A), S)
    sref = svdvals(copy(A))
    sp = sort(S, rev=true)
    maximum(abs.(sp .- sref) ./ max.(sref, eps(real(eltype(A)))))
end

maxerr = 0.0; nfail = 0
for T in (Float64, ComplexF64, ComplexF32)
    for (m,n) in ((3,3),(4,4),(5,5),(7,7),(8,8),(15,15),(16,16),(17,17),(31,31),(32,32),(33,33),
                  (48,48),(63,63),(64,64),(65,65),(100,100),(128,128),(200,200),(256,256),
                  (20,8),(8,20),(50,13),(13,50),(129,64),(64,129),(300,17),(17,300))
        for trial in 1:3
            A = T <: Complex ? randn(T,m,n) : randn(m,n)
            e = checkone(A)
            global maxerr = max(maxerr, e)
            tol = T == ComplexF32 ? 1e-4 : 1e-9
            if !(e < tol) || isnan(e)
                @printf("FAIL %s %dx%d trial%d err=%.3e\n", T, m, n, trial, e)
                global nfail += 1
            end
        end
    end
end
# edge cases: identity-ish, zeros, tiny
for T in (Float64, ComplexF64)
    A = Matrix{T}(I, 10, 10); global maxerr = max(maxerr, checkone(A))
    A = zeros(T, 6,6); A[1,1]=T(3); A[3,3]=T(2); global maxerr = max(maxerr, checkone(A))
    D = T <: Complex ? Diagonal(randn(T,12)) : Diagonal(randn(12)); global maxerr = max(maxerr, checkone(Matrix(D)))
end
@printf("\nmax rel err = %.3e   failures = %d\n", maxerr, nfail)
println(nfail == 0 ? "ALL PASS" : "HAS FAILURES")
