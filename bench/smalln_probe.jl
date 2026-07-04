# Focused SMALL-N L3/LAPACK gate probe (the Zen3 grind gaps): syrk/syr2k/trsm/potrf/getrf at n≤256.
# Interleaved paired median (same methodology as l3bench/lapackbench). Usage: taskset -c 8 julia --project=bench smalln.jl
using PureBLAS, LinearAlgebra, Statistics, Random
import LinearAlgebra.BLAS as B
BLAS.set_num_threads(1)
const S = Ref(0.0); @noinline _run(f)=f()
const L='L'; const Nn='N'; const Nd='N'

function _stable(ob, our, reset; rounds=13, tol=0.02, cap=61)
    reset(); _run(ob); reset(); _run(our)
    rs=Float64[]
    while true
        for _ in 1:rounds
            reset(); t0=time_ns(); a=_run(ob); t1=time_ns()
            reset(); t2=time_ns(); b=_run(our); t3=time_ns()
            S[]+=(a isa Number ? a : 0.0)+(b isa Number ? b : 0.0); push!(rs,(t1-t0)/(t3-t2))
        end
        m=median(rs); ((quantile(rs,0.75)-quantile(rs,0.25))/m<tol||length(rs)>=cap) && return m,length(rs)
    end
end

# reps to lift a single timed block above timer noise for tiny n
_reps(n) = clamp(round(Int, 4e6 / (n^3 + 1)), 1, 20000)

function probe(name, n)
    Random.seed!(1000+n); reps=_reps(n)
    if name=="syrk"
        A=randn(n,n); C=randn(n,n)
        our=()->(for _ in 1:reps; PureBLAS.syrk!(C,A;uplo=L,trans=Nn,alpha=1.0,beta=0.0); end; C[1])
        ob =()->(for _ in 1:reps; B.syrk!(L,Nn,1.0,A,0.0,C); end; C[1])
        _stable(ob,our,()->nothing)
    elseif name=="syr2k"
        A=randn(n,n); Bm=randn(n,n); C=randn(n,n)
        our=()->(for _ in 1:reps; PureBLAS.syr2k!(C,A,Bm;uplo=L,trans=Nn,alpha=1.0,beta=0.0); end; C[1])
        ob =()->(for _ in 1:reps; B.syr2k!(L,Nn,1.0,A,Bm,0.0,C); end; C[1])
        _stable(ob,our,()->nothing)
    elseif name=="trsm"
        Ad=randn(n,n)+n*I; Bm=randn(n,n); Bw=copy(Bm); rst=()->copyto!(Bw,Bm)
        our=()->(PureBLAS.trsm!(Bw,Ad;side=L,uplo=L,transA=Nn,diag=Nd,alpha=1.0); Bw[1])
        ob =()->(B.trsm!(L,L,Nn,Nd,1.0,Ad,Bw); Bw[1])
        _stable(ob,our,rst)
    elseif name=="trmm"
        A=randn(n,n); Bm=randn(n,n); Bw=copy(Bm); rst=()->copyto!(Bw,Bm)
        our=()->(PureBLAS.trmm!(Bw,A;side=L,uplo=L,transA=Nn,diag=Nd,alpha=1.0); Bw[1])
        ob =()->(B.trmm!(L,L,Nn,Nd,1.0,A,Bw); Bw[1])
        _stable(ob,our,rst)
    elseif name=="potrf"
        M=randn(n,n); A0=M'M+n*I; Aw=similar(A0); rst=()->copyto!(Aw,A0)
        our=()->(PureBLAS.potrf!(Aw;uplo=L); Aw[1]); ob=()->(LAPACK.potrf!(L,Aw); Aw[1])
        _stable(ob,our,rst)
    elseif name=="getrf"
        A0=randn(n,n); Aw=similar(A0); ip=zeros(Int,n); rst=()->copyto!(Aw,A0)
        our=()->(PureBLAS.getrf!(Aw,ip); Aw[1]); ob=()->(LAPACK.getrf!(Aw); Aw[1])
        _stable(ob,our,rst)
    end
end

println("== small-n L3/LAPACK gate probe (host=$(strip(read(`hostname`,String))), 1-thread) ==")
for name in ("syrk","syr2k","trmm","trsm","potrf","getrf")
    for n in (32,64,128,256)
        m,nr=probe(name,n)
        flag = m>=0.96 ? "ok " : "LOW"
        println("  $name n=$n: $(round(m,digits=3)) [$flag] (rounds=$nr)"); flush(stdout)
    end
end
println("== done (S=$(S[])) ==")
