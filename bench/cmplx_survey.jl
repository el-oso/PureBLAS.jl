# Scope M5: measure the complex (ComplexF64) surface PureBLAS vs OpenBLAS. Ops far below 1.0 = still scalar.
using PureBLAS, LinearAlgebra, Statistics, Printf
import LinearAlgebra.BLAS as B
BLAS.set_num_threads(1); const SINK=Ref(ComplexF64(0)); @noinline _run(f)=f()
const T = ComplexF64
function ratio(mk, ob, pb; rounds=15)
    reps = mk === nothing ? 1 : 1
    rs=Float64[]
    for _ in 1:rounds
        c = mk()
        _run(()->ob(c)); _run(()->pb(c))
        t0=time_ns(); for _ in 1:reps; SINK[]+=ob(c); end; t1=time_ns()
        for _ in 1:reps; SINK[]+=pb(c); end; t2=time_ns()
        push!(rs,(t1-t0)/(t2-t1))
    end
    median(rs)
end
# reps folded into the op closures for cheap ops
rep1(n)=clamp(20_000_000÷n,50,20000)      # L1
rep2(n)=clamp(200_000_000÷(n*n),30,5000)  # L2
rep3(n)=clamp(20_000_000÷(n^3),1,200)     # L3
res = Tuple{String,Float64}[]
try r=ratio(()->(m=rep1(100000);(randn(T,100000),randn(T,100000),m)),
    c->(for _ in 1:c[3]; B.axpy!(1.7+0.3im,c[1],c[2]); end; c[2][1]),
    c->(for _ in 1:c[3]; PureBLAS.axpy!(c[2],1.7+0.3im,c[1]); end; c[2][1])); push!(res,("axpy",r)) catch e; push!(res,("axpy ERR",NaN)) end
try r=ratio(()->(m=rep1(100000);(randn(T,100000),randn(T,100000),m)),
    c->(s=zero(T); for _ in 1:c[3]; s+=B.dotc(c[1],c[2]); end; s),
    c->(s=zero(T); for _ in 1:c[3]; s+=PureBLAS.dot(c[1],c[2]); end; s)); push!(res,("dotc",r)) catch e; push!(res,("dotc ERR",NaN)) end
try r=ratio(()->(m=rep1(100000);(randn(T,100000),m)),
    c->(for _ in 1:c[2]; B.scal!(1.0001+0im,c[1]); end; c[1][1]),
    c->(for _ in 1:c[2]; PureBLAS.scal!(1.0001+0im,c[1]); end; c[1][1])); push!(res,("scal",r)) catch e; push!(res,("scal ERR",NaN)) end
try r=ratio(()->(m=rep1(100000);(randn(T,100000),m)),
    c->(s=0.0; for _ in 1:c[2]; s+=B.nrm2(c[1]); end; s),
    c->(s=0.0; for _ in 1:c[2]; s+=PureBLAS.nrm2(c[1]); end; s)); push!(res,("nrm2",r)) catch e; push!(res,("nrm2 ERR",NaN)) end
# L2 n=512
let n=512
try r=ratio(()->(randn(T,n,n),randn(T,n),randn(T,n),rep2(n)),
    c->(for _ in 1:c[4]; B.gemv!('N',one(T),c[1],c[2],zero(T),c[3]); end; c[3][1]),
    c->(for _ in 1:c[4]; PureBLAS.gemv!(c[3],c[1],c[2];alpha=one(T),beta=zero(T)); end; c[3][1])); push!(res,("gemvN",r)) catch e; push!(res,("gemvN ERR",NaN)) end
try r=ratio(()->(randn(T,n,n),randn(T,n),randn(T,n),rep2(n)),
    c->(for _ in 1:c[4]; B.geru!(one(T),c[2],c[3],c[1]); end; c[1][1]),
    c->(for _ in 1:c[4]; PureBLAS.ger!(one(T),c[2],c[3],c[1]); end; c[1][1])); push!(res,("geru",r)) catch e; push!(res,("geru ERR",NaN)) end
try r=ratio(()->(A=randn(T,n,n);A=A+A';(A,randn(T,n),randn(T,n),rep2(n))),
    c->(for _ in 1:c[4]; B.hemv!('U',one(T),c[1],c[2],zero(T),c[3]); end; c[3][1]),
    c->(for _ in 1:c[4]; PureBLAS.hemv!(c[3],c[1],c[2];uplo='U',alpha=one(T),beta=zero(T)); end; c[3][1])); push!(res,("hemv",r)) catch e; push!(res,("hemv ERR",NaN)) end
try r=ratio(()->(A=randn(T,n,n);for i in 1:n;A[i,i]=1+abs(A[i,i]);end;(A,randn(T,n),randn(T,n),rep2(n))),
    c->(for _ in 1:c[4]; copyto!(c[3],c[2]); B.trmv!('U','N','N',c[1],c[3]); end; c[3][1]),
    c->(for _ in 1:c[4]; copyto!(c[3],c[2]); PureBLAS.trmv!(c[1],c[3];uplo='U'); end; c[3][1])); push!(res,("trmv",r)) catch e; push!(res,("trmv ERR",NaN)) end
end
# L3 n=512
let n=512
try r=ratio(()->(randn(T,n,n),randn(T,n,n),zeros(T,n,n),rep3(n)),
    c->(for _ in 1:c[4]; B.gemm!('N','N',one(T),c[1],c[2],zero(T),c[3]); end; c[3][1]),
    c->(for _ in 1:c[4]; PureBLAS.gemm!(c[3],c[1],c[2]); end; c[3][1])); push!(res,("gemm",r)) catch e; push!(res,("gemm ERR",NaN)) end
try r=ratio(()->(A=randn(T,n,n);A=A+A';(A,randn(T,n,n),zeros(T,n,n),rep3(n))),
    c->(for _ in 1:c[4]; B.hemm!('L','U',one(T),c[1],c[2],zero(T),c[3]); end; c[3][1]),
    c->(for _ in 1:c[4]; PureBLAS.hemm!(c[3],c[1],c[2];side='L',uplo='U',alpha=one(T),beta=zero(T)); end; c[3][1])); push!(res,("hemm",r)) catch e; push!(res,("hemm ERR",NaN)) end
try r=ratio(()->(randn(T,n,n),zeros(T,n,n),rep3(n)),
    c->(for _ in 1:c[3]; B.herk!('U','N',1.0,c[1],0.0,c[2]); end; c[2][1]),
    c->(for _ in 1:c[3]; PureBLAS.herk!(c[2],c[1];uplo='U',trans='N',alpha=1.0,beta=0.0); end; c[2][1])); push!(res,("herk",r)) catch e; push!(res,("herk ERR",NaN)) end
try r=ratio(()->(A=randn(T,n,n);for i in 1:n;A[i,i]=1+abs(A[i,i]);end;(A,randn(T,n,n),rep3(n))),
    c->(for _ in 1:c[3]; D=copy(c[2]); B.trmm!('L','U','N','N',one(T),c[1],D); end; c[2][1]),
    c->(for _ in 1:c[3]; D=copy(c[2]); PureBLAS.trmm!(D,c[1];side='L',uplo='U'); end; c[2][1])); push!(res,("trmm",r)) catch e; push!(res,("trmm ERR",NaN)) end
try r=ratio(()->(A=randn(T,n,n);for i in 1:n;A[i,i]=1+abs(A[i,i]);end;(A,randn(T,n,n),rep3(n))),
    c->(for _ in 1:c[3]; D=copy(c[2]); B.trsm!('L','U','N','N',one(T),c[1],D); end; c[2][1]),
    c->(for _ in 1:c[3]; D=copy(c[2]); PureBLAS.trsm!(D,c[1];side='L',uplo='U'); end; c[2][1])); push!(res,("trsm",r)) catch e; push!(res,("trsm ERR",NaN)) end
end
println("=== ComplexF64 surface: PureBLAS/OpenBLAS (1.0 = parity; <0.9 = likely scalar) ===")
for (nm,r) in res; @printf("%-8s %s\n", nm, isnan(r) ? "ERR" : @sprintf("%.3f", r)); end
