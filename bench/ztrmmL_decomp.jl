# Decompose ztrmmL n=16/32 (tr=false UN vs tr=true UC): alloc + absolute ns PB vs OB.
using PureBLAS, LinearAlgebra, Statistics, Printf
import LinearAlgebra.BLAS as B
BLAS.set_num_threads(1); const T = ComplexF64; const P = PureBLAS; const S = Ref(ComplexF64(0))
tri(n) = (A = rand(T,n,n)./(2n); for i in 1:n; A[i,i]=1+abs(A[i,i]); end; A)
function bestns(f, reps)
    best = Inf
    for _ in 1:200
        t0 = time_ns(); for _ in 1:reps; f(); end; t1 = time_ns()
        best = min(best, (t1-t0)/reps)
    end
    best
end
for (tag, tA) in (("UN",'N'), ("UC",'C')), n in (16, 32)
    A = tri(n); C = rand(T,n,n)
    C2 = copy(C); P.trmm!(C2, A; side='L', uplo='U', transA=tA, diag='N')  # warm
    al = @allocated P.trmm!(C2, A; side='L', uplo='U', transA=tA, diag='N')
    reps = clamp(50_000_000 ÷ n^3, 1, 500)
    pb = bestns(() -> (P.trmm!(C, A; side='L', uplo='U', transA=tA, diag='N'); S[]+=C[1]), reps)
    ob = bestns(() -> (B.trmm!('L','U',tA,'N',one(T),A,C); S[]+=C[1]), reps)
    @printf("L-%s n=%3d  PB=%7.1fns  OB=%7.1fns  OB/PB=%.3f  alloc=%d\n", tag, n, pb, ob, ob/pb, al)
end
