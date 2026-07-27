# F32 potrf decomposition: is the sub-parity the scalar base, or the trsm!/syrk! kernels the generic
# recursion calls at large n? Measure PB-vs-OB F32 for the potrf trailing shapes. Ratio=OB/PB (>1 gates).
# Run: julia --project=bench bench/f32_decomp.jl
using PureBLAS, Chairmarks, LinearAlgebra
import LinearAlgebra.LAPACK, LinearAlgebra.BLAS
import PureBLAS as P
BLAS.set_num_threads(1)
mt(b) = minimum(x.time for x in b.samples)

println("== F32 trsm!(side=R,uplo=L,transA=T) — potrf panel solve, m×bs, bs=n/2 ==")
for n in (128, 256, 512, 1024, 2048)
    bs = n ÷ 2; m = n - bs
    A = Matrix{Float32}(I(bs)) + 0.01f0*randn(Float32,bs,bs); B0 = randn(Float32, m, bs)
    ob = @be copy(B0) BLAS.trsm!('R','L','T','N',1.0f0,A,_) evals=1 seconds=0.4
    pb = @be copy(B0) P.trsm!(_,A; side='R',uplo='L',transA='T',diag='N',alpha=true) evals=1 seconds=0.4
    println("  n=$n (m=$m,bs=$bs)  trsmR OB/PB=", round(mt(ob)/mt(pb),digits=2))
end
println("== F32 syrk!(uplo=L,trans=N) — potrf trailing downdate, m×m rank-bs ==")
for n in (128, 256, 512, 1024, 2048)
    bs = n ÷ 2; m = n - bs
    A = randn(Float32, m, bs); C0 = Matrix{Float32}(randn(Float32,m,m)); C0 = C0*C0'
    ob = @be copy(C0) BLAS.syrk!('L','N',-1.0f0,A,1.0f0,_) evals=1 seconds=0.4
    pb = @be copy(C0) P.syrk!(_,A; uplo='L',trans='N',alpha=-1.0f0,beta=1.0f0) evals=1 seconds=0.4
    println("  n=$n (m=$m,rank=$bs)  syrk OB/PB=", round(mt(ob)/mt(pb),digits=2))
end
