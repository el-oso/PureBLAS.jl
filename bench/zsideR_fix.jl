using PureBLAS, LinearAlgebra, BenchmarkTools
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)
const T = ComplexF64
const PB = PureBLAS
ratio(t_pb, t_ob) = t_ob / t_pb

# ---- correctness: side R, all uplo×trans×diag, non-square m ----
println("=== correctness (side R) ===")
maxerr = 0.0
for ul in ('U','L'), ta in ('N','T','C'), dg in ('N','U'), (m,k) in ((128,128),(200,96),(64,64))
    A = (ul=='U' ? triu(rand(T,k,k)) : tril(rand(T,k,k))) + k*I
    dg=='U' && (A[diagind(A)] .= one(T))
    X = rand(T,m,k)
    B1 = copy(X); PB.trmm!(B1, A; side='R', uplo=ul, transA=ta, diag=dg)
    B2 = copy(X); BLAS.trmm!('R', ul, ta, dg, one(T), A, B2)
    global maxerr = max(maxerr, norm(B1-B2)/max(norm(B2),eps()))
end
println("  max rel err = ", maxerr, maxerr < 1e-12 ? "  ✓" : "  ✗ FAIL")

# ---- full side-R sweep after fix ----
println("=== full trmm! side=R sweep (post-fix) ===")
for k in (8, 16, 32, 64, 128, 256, 512, 1024, 2048)
    A = triu(rand(T,k,k)) + k*I
    X = rand(T,k,k); al = one(T)
    t_pb = @belapsed PB.trmm!(B, $A; side='R', uplo='U', transA='N', alpha=$al) setup=(B=copy($X)) evals=1
    t_ob = @belapsed BLAS.trmm!('R','U','N','N', $al, $A, B) setup=(B=copy($X)) evals=1
    println("  k=$k   PB/OB = ", round(ratio(t_pb,t_ob),digits=3))
end
