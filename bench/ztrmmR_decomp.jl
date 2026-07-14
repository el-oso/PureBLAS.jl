# Decompose ztrmmR n=16/32 floor: allocations + absolute ns per call PB vs OB.
using PureBLAS, LinearAlgebra, Statistics, Printf
import LinearAlgebra.BLAS as B
BLAS.set_num_threads(1); const T = ComplexF64; const P = PureBLAS; const S = Ref(ComplexF64(0))
tri(n) = (A = rand(T,n,n)./(2n); for i in 1:n; A[i,i]=1+abs(A[i,i]); end; A)
function bestns(f, reps)   # min over rounds of (loop of reps)/reps
    best = Inf
    for _ in 1:200
        t0 = time_ns(); for _ in 1:reps; f(); end; t1 = time_ns()
        best = min(best, (t1-t0)/reps)
    end
    best
end
for n in (16, 32, 64)
    A = tri(n); C = rand(T,n,n); C2 = copy(C)
    al = @allocated P.trmm!(C2, A; side='R', uplo='U', transA='N', diag='N')
    reps = clamp(50_000_000 ÷ n^3, 1, 500)
    pb = bestns(() -> (P.trmm!(C, A; side='R', uplo='U', transA='N', diag='N'); S[]+=C[1]), reps)
    ob = bestns(() -> (B.trmm!('R','U','N','N',one(T),A,C); S[]+=C[1]), reps)
    @printf("n=%3d  PB=%7.1fns  OB=%7.1fns  OB/PB=%.3f  PB_alloc=%d bytes\n", n, pb, ob, ob/pb, al)
end
