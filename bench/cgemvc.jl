# zgemvC + zgemvN gate check only (fast iteration).
using PureBLAS, LinearAlgebra, Statistics, Printf
import LinearAlgebra.BLAS as B
BLAS.set_num_threads(1); const SINK=Ref(0.0); @noinline _run(f)=f()
const T=ComplexF64; const TN=Char(78); const TC=Char(67)
const ca=one(T); const cb=zero(T)
L2SZ=(64,128,256,512,1024,2048,4096); rep(s)=clamp(400_000_000÷(s*s),30,20000)
function sweep(mk,ob,pb,s;rounds=25)
    reps=rep(s); rs=Float64[]
    for _ in 1:rounds
        c=mk(s); _run(()->ob(c,1)); _run(()->pb(c,1))
        t0=time_ns(); v1=_run(()->ob(c,reps)); t1=time_ns(); v2=_run(()->pb(c,reps)); t2=time_ns()
        SINK[]+=real(v1)+real(v2); push!(rs,(t1-t0)/(t2-t1))
    end; median(rs)
end
sq(s)=(randn(T,s,s),randn(T,s),randn(T,s))
ops=(
 ("zgemvC",(c,m)->(for _ in 1:m;B.gemv!(TC,ca,c[1],c[2],cb,c[3]);end;real(c[3][1])),(c,m)->(for _ in 1:m;PureBLAS.gemv!(c[3],c[1],c[2];alpha=ca,beta=cb,trans=TC);end;real(c[3][1]))),
 ("zgemvN",(c,m)->(for _ in 1:m;B.gemv!(TN,ca,c[1],c[2],cb,c[3]);end;real(c[3][1])),(c,m)->(for _ in 1:m;PureBLAS.gemv!(c[3],c[1],c[2];alpha=ca,beta=cb);end;real(c[3][1]))),
)
for (nm,ob,pb) in ops
    print(rpad(nm,8))
    for s in L2SZ
        r=sweep(sq,ob,pb,s); @printf("%d=%.3f%s ", s, r, r<0.96 ? "*" : " ")
    end
    println()
end
