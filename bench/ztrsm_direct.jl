using PureBLAS, LinearAlgebra, BenchmarkTools
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)
const T=ComplexF64; const PB=PureBLAS; ratio(tp,to)=to/tp
tri(n,ul)=(A = ul=='U' ? triu(rand(T,n,n)) : tril(rand(T,n,n)); for i in 1:n; A[i,i]=1+abs(A[i,i]); end; A)
# correctness: side-L, all uplo×trans×diag, n incl base/recursion, non-square nrhs
me=0.0; nf=0
for ul in ('U','L'), ta in ('N','T','C'), dg in ('N','U'), (k,nr) in ((1,1),(7,3),(16,16),(64,64),(100,40),(128,128),(200,64))
    A=tri(k,ul); X=rand(T,k,nr); al=T(0.9,0.4)
    B1=copy(X); PB.trsm!(B1,A;side='L',uplo=ul,transA=ta,diag=dg,alpha=al)
    B2=copy(X); BLAS.trsm!('L',ul,ta,dg,al,A,B2)
    e=norm(B1-B2)/max(norm(B2),eps())
    (e<1e-10 && !any(isnan,B1)) || (global nf+=1; println("  FAIL ul=$ul ta=$ta dg=$dg k=$k nr=$nr relerr=$e"))
    global me=max(me,e)
end
println("CORRECTNESS ztrsm-L: maxrelerr=$me  $(nf==0 ? "PASS" : "*** $nf FAIL ***")")
println("=== ztrsm side=L (plotted) ===")
for n in (8,16,32,48,64,96,128,192,256,512)
  A=tri(n,'U'); X=rand(T,n,n); ca=T(0.9,0.4)
  tp=@belapsed PB.trsm!(B,$A;side='L',uplo='U',transA='N',alpha=$ca) setup=(B=copy($X)) evals=1
  to=@belapsed BLAS.trsm!('L','U','N','N',$ca,$A,B) setup=(B=copy($X)) evals=1
  println("  n=$n  PB/OB=",round(ratio(tp,to),digits=3))
end
