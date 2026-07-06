using PureBLAS, LinearAlgebra, BenchmarkTools
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)
const T=ComplexF64; const PB=PureBLAS; ratio(tp,to)=to/tp
tri(n,ul)=(A = ul=='U' ? triu(rand(T,n,n)) : tril(rand(T,n,n)); for i in 1:n; A[i,i]=1+abs(A[i,i]); end; A)
# correctness: all uplo×trans×diag, n incl blocks/crossover
me=0.0; nf=0
for ul in ('U','L'), ta in ('N','T','C'), dg in ('N','U'), n in (1,3,7,8,16,63,64,65,128,255,256,300,600)
    A=tri(n,ul); x0=rand(T,n)
    x1=copy(x0); PB.trsv!(A,x1;uplo=ul,trans=ta,diag=dg)
    x2=copy(x0); BLAS.trsv!(ul,ta,dg,A,x2)
    e=norm(x1-x2)/max(norm(x2),eps())
    (e<1e-11 && !any(isnan,x1)) || (global nf+=1; println("  FAIL ul=$ul ta=$ta dg=$dg n=$n relerr=$e"))
    global me=max(me,e)
end
println("CORRECTNESS ztrsv: maxrelerr=$me  $(nf==0 ? "PASS" : "*** $nf FAIL ***")")
println("=== ztrsv small-n (trans=N) ===")
for n in (8,16,32,48,64,96,128,192,256)
  A=tri(n,'U'); x0=rand(T,n)
  tp=@belapsed PB.trsv!($A,X;uplo='U') setup=(X=copy($x0)) evals=1
  to=@belapsed BLAS.trsv!('U','N','N',$A,X) setup=(X=copy($x0)) evals=1
  println("  n=$n  PB/OB=",round(ratio(tp,to),digits=3))
end
println("=== ztrsv trans=C (n=64,128) sanity ===")
for n in (64,128)
  A=tri(n,'U'); x0=rand(T,n)
  tp=@belapsed PB.trsv!($A,X;uplo='U',trans='C') setup=(X=copy($x0)) evals=1
  to=@belapsed BLAS.trsv!('U','C','N',$A,X) setup=(X=copy($x0)) evals=1
  println("  n=$n  PB/OB=",round(ratio(tp,to),digits=3))
end
