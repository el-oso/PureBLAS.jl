# ClusterC: focused ztrsv/ztrmv/zhemv gate + GB/s decomposition, EXACT plots.jl methodology.
# Chairmarks @be evals=1 (cold mk per sample), pooled _qratios, median = gate number.
using PureBLAS, LinearAlgebra, Statistics, Printf, Chairmarks
import LinearAlgebra.BLAS as B
BLAS.set_num_threads(1)
const T=ComplexF64; const TN=Char(78); const TC=Char(67); const U=Char(85); const L=Char(76)
const ca=one(T); const cb=zero(T)
SZ=(256,512,1024,2048,4096); _L2REP(s)=clamp(400_000_000÷(s*s),30,20000)
_rounds(s)=8
_times(b)=Float64[smp.time for smp in b.samples]
_qratios(bo,bp)=(to=_times(bo);tp=_times(bp);qs=range(0.03,0.97;length=48);[quantile(to,q)/quantile(tp,q) for q in qs])
function gate(mk,ob,pb,s)
    reps=_L2REP(s); acc=Float64[]
    for r in 1:_rounds(s)
        if isodd(r)
            bo=@be mk(s) (c->ob(c,reps)) evals=1 samples=400 seconds=0.15
            bp=@be mk(s) (c->pb(c,reps)) evals=1 samples=400 seconds=0.15
        else
            bp=@be mk(s) (c->pb(c,reps)) evals=1 samples=400 seconds=0.15
            bo=@be mk(s) (c->ob(c,reps)) evals=1 samples=400 seconds=0.15
        end
        append!(acc,_qratios(bo,bp))
    end
    # min per-call times for GB/s
    bo=@be mk(s) (c->ob(c,reps)) evals=1 samples=200 seconds=0.15
    bp=@be mk(s) (c->pb(c,reps)) evals=1 samples=200 seconds=0.15
    (median(acc), minimum(_times(bo))/reps*1e9, minimum(_times(bp))/reps*1e9)  # ratio, ob ns, pb ns
end
herm(s)=(A=randn(T,s,s);A=A+A';for i in 1:s;A[i,i]=real(A[i,i]);end;(A,randn(T,s),randn(T,s)))
tri(s)=(A=randn(T,s,s);for i in 1:s;A[i,i]=1+abs(A[i,i]);end;(A,randn(T,s),randn(T,s)))
ops=(
 ("ztrmvU",tri,(c,m)->(for _ in 1:m;copyto!(c[3],c[2]);B.trmv!(U,TN,TN,c[1],c[3]);end;real(c[3][1])),(c,m)->(for _ in 1:m;copyto!(c[3],c[2]);PureBLAS.trmv!(c[1],c[3];uplo=U);end;real(c[3][1])), s->s*s/2*16),
 ("ztrsvU",tri,(c,m)->(for _ in 1:m;copyto!(c[3],c[2]);B.trsv!(U,TN,TN,c[1],c[3]);end;real(c[3][1])),(c,m)->(for _ in 1:m;copyto!(c[3],c[2]);PureBLAS.trsv!(c[1],c[3];uplo=U);end;real(c[3][1])), s->s*s/2*16),
 ("ztrsvL",tri,(c,m)->(for _ in 1:m;copyto!(c[3],c[2]);B.trsv!(L,TN,TN,c[1],c[3]);end;real(c[3][1])),(c,m)->(for _ in 1:m;copyto!(c[3],c[2]);PureBLAS.trsv!(c[1],c[3];uplo=L);end;real(c[3][1])), s->s*s/2*16),
 ("zhemvU",herm,(c,m)->(for _ in 1:m;B.hemv!(U,ca,c[1],c[2],cb,c[3]);end;real(c[3][1])),(c,m)->(for _ in 1:m;PureBLAS.hemv!(c[3],c[1],c[2];uplo=U,alpha=ca,beta=cb);end;real(c[3][1])), s->s*s*16),
)
for (nm,mk,ob,pb,byf) in ops
    println("── $nm ──")
    for s in SZ
        r,obns,pbns=gate(mk,ob,pb,s)
        gbob=byf(s)/obns; gbpb=byf(s)/pbns
        flag=r<1.0 ? (r<0.96 ? "**" : "*") : "  "
        @printf("  n=%-5d gate=%.3f%s  PB=%.0fGB/s OB=%.0fGB/s  (pb=%.1fµs ob=%.1fµs)\n", s, r, flag, gbpb, gbob, pbns/1e3, obns/1e3)
    end
end
