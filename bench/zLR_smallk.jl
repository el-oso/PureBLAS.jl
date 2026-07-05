using PureBLAS, LinearAlgebra, BenchmarkTools
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)
const T = ComplexF64; const PB = PureBLAS
ratio(t_pb, t_ob) = t_ob / t_pb

println("=== trmm! side L vs R, small-k (square) ===")
println("   k     L        R")
for k in (8, 16, 32, 48, 64, 96, 128)
    A = triu(rand(T,k,k)) + k*I; X = rand(T,k,k); al = one(T)
    tL = @belapsed PB.trmm!(B, $A; side='L', uplo='U', transA='N', alpha=$al) setup=(B=copy($X)) evals=1
    oL = @belapsed BLAS.trmm!('L','U','N','N', $al, $A, B) setup=(B=copy($X)) evals=1
    tR = @belapsed PB.trmm!(B, $A; side='R', uplo='U', transA='N', alpha=$al) setup=(B=copy($X)) evals=1
    oR = @belapsed BLAS.trmm!('R','U','N','N', $al, $A, B) setup=(B=copy($X)) evals=1
    println("  ", lpad(k,3), "   ", rpad(round(ratio(tL,oL),digits=3),6), "   ", round(ratio(tR,oR),digits=3))
end
