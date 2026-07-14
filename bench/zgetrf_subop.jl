# Sub-op PB/OB ratios at getrf's trailing shapes: trsm-L (unit lower) and the panel factorization.
using PureBLAS, LinearAlgebra, Statistics, Printf
import LinearAlgebra.BLAS as B
import LinearAlgebra.LAPACK as LA
BLAS.set_num_threads(1); const SINK = Ref(0.0)
const T = ComplexF64

# trsm side=L uplo=L transA=N diag=U alpha=1 : solve L(nb×nb) X = B(nb×ntrail)
function trsm_ratio(nb, ntrail, rounds)
    reps = clamp(20_000_000 ÷ (nb*nb*ntrail), 1, 300); rs = Float64[]
    for _ in 1:rounds
        L = randn(T, nb, nb); for i in 1:nb; L[i,i]=1; for j in i+1:nb; L[i,j]=0; end; end
        Bob = [randn(T, nb, ntrail) for _ in 1:reps]; Bpb = [copy(Bob[r]) for r in 1:reps]
        t0=time_ns(); for r in 1:reps; B.trsm!('L','L','N','U', one(T), L, Bob[r]); end; t1=time_ns()
        for r in 1:reps; PureBLAS.trsm!(Bpb[r], L; side='L', uplo='L', transA='N', diag='U', alpha=true); end; t2=time_ns()
        SINK[]+=real(Bob[1][1])+real(Bpb[1][1]); push!(rs,(t1-t0)/(t2-t1))
    end
    median(rs)
end

# panel factor: LAPACK getrf! on a tall m×nb  vs  PB _getf2! (the rank-2 panel base driver)
using PureBLAS: _getf2!
function panel_ratio(m, nb, rounds)
    reps = clamp(20_000_000 ÷ (m*nb*nb), 1, 300); rs = Float64[]
    for _ in 1:rounds
        base = randn(T, m, nb)
        ob = [copy(base) for _ in 1:reps]; pb = [copy(base) for _ in 1:reps]
        obp = [Vector{Int}(undef, nb) for _ in 1:reps]; pbp = [Vector{Int}(undef, nb) for _ in 1:reps]
        t0=time_ns(); for r in 1:reps; LA.getrf!(ob[r], obp[r]); end; t1=time_ns()
        for r in 1:reps; _getf2!(pb[r], m, nb, 0, pbp[r], 0); end; t2=time_ns()
        SINK[]+=real(ob[1][1])+real(pb[1][1]); push!(rs,(t1-t0)/(t2-t1))
    end
    median(rs)
end

println("== trsm-L (nb×ntrail) OB/PB ==")
for (nb,nt) in ((32,224),(48,208),(96,416),(168,856),(168,1880))
    @printf("nb=%-3d ntrail=%-4d  %.3f\n", nb, nt, trsm_ratio(nb,nt,13)); flush(stdout)
end
println("== panel getf2 (m×nb) OB/PB ==")
for (m,nb) in ((128,32),(256,48),(512,96),(1024,168),(2048,168))
    @printf("m=%-4d nb=%-3d  %.3f\n", m, nb, panel_ratio(m,nb,13)); flush(stdout)
end
