# 3M vs blocked complex gemm at getrf's trailing/cross-update shapes. C := -A*B + C (LU downdate).
using PureBLAS, LinearAlgebra, Statistics, Printf
import LinearAlgebra.BLAS as B
using PureBLAS: _gemm_3m!, _gemm_cmplx_blocked!
BLAS.set_num_threads(1); const SINK = Ref(0.0); const T = ComplexF64
function ab(m, n, k, rounds)
    reps = clamp(15_000_000 ÷ (m*n*k), 1, 300)
    r3 = Float64[]; rb = Float64[]
    for _ in 1:rounds
        A = randn(T, m, k); Bm = randn(T, k, n)
        C0 = randn(T, m, n)
        Cob=[copy(C0) for _ in 1:reps]; C3=[copy(C0) for _ in 1:reps]; Cbl=[copy(C0) for _ in 1:reps]
        t0=time_ns(); for r in 1:reps; B.gemm!('N','N',T(-1),A,Bm,T(1),Cob[r]); end; t1=time_ns()
        for r in 1:reps; _gemm_3m!(false,false,false,false,m,n,k,T(-1),A,Bm,T(1),C3[r]); end; t2=time_ns()
        for r in 1:reps; _gemm_cmplx_blocked!(false,false,false,false,m,n,k,T(-1),A,Bm,T(1),Cbl[r]); end; t3=time_ns()
        SINK[]+=real(C3[1][1])+real(Cbl[1][1])+real(Cob[1][1])
        push!(r3,(t1-t0)/(t2-t1)); push!(rb,(t1-t0)/(t3-t2))
    end
    (median(r3), median(rb))
end
# correctness sanity
let m=64,n=48,k=40
  A=randn(T,m,k);Bm=randn(T,k,n);C0=randn(T,m,n)
  C1=copy(C0);C2=copy(C0)
  _gemm_3m!(false,false,false,false,m,n,k,T(-1),A,Bm,T(1),C1)
  _gemm_cmplx_blocked!(false,false,false,false,m,n,k,T(-1),A,Bm,T(1),C2)
  ref=copy(C0); B.gemm!('N','N',T(-1),A,Bm,T(1),ref)
  @printf("sanity 3M err=%.1e blocked err=%.1e\n", norm(C1-ref)/norm(ref), norm(C2-ref)/norm(ref))
end
println("shape             3M/OB  blk/OB   (>1 gates; pick higher)")
for (m,n,k) in ((208,208,48),(232,24,24),(1000,84,84),(1000,42,42),(1000,21,21),(856,856,168),(856,856,96))
    r3,rb = ab(m,n,k,13)
    @printf("%4d x%4d x%3d    %.3f  %.3f   %s\n", m,n,k, r3, rb, rb>r3 ? "BLOCKED" : "3M")
    flush(stdout)
end
