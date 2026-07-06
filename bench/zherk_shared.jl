using PureBLAS, LinearAlgebra, BenchmarkTools
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)
const T = ComplexF64; const PB = PureBLAS
ratio(t_pb, t_ob) = t_ob / t_pb

# The SHARED tri-output complex kernel, exercised via zherk & zsyrk (single product) across n.
for (nm, herm) in (("zherk", true), ("zsyrk", false)), tr in (herm ? ('N','C') : ('N','T'))
    println("=== $nm trans=$tr ===")
    for n in (8,16,32,48,64,96,128,192,256,512,1024)
        k = n
        A = rand(T, tr=='N' ? n : k, tr=='N' ? k : n)
        C0 = rand(T, n, n)
        if herm
            tp = @belapsed PB.herk!(C, $A; uplo='U', trans=$tr, alpha=1.0, beta=0.0) setup=(C=copy($C0)) evals=1
            to = @belapsed BLAS.herk!('U', $tr, 1.0, $A, 0.0, C) setup=(C=copy($C0)) evals=1
        else
            tp = @belapsed PB.syrk!(C, $A; uplo='U', trans=$tr, alpha=one($T), beta=zero($T)) setup=(C=copy($C0)) evals=1
            to = @belapsed BLAS.syrk!('U', $tr, one($T), $A, zero($T), C) setup=(C=copy($C0)) evals=1
        end
        println("  n=$n   PB/OB = ", round(ratio(tp,to),digits=3))
    end
end
