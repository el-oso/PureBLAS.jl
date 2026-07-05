using PureBLAS, LinearAlgebra, BenchmarkTools
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)
const T = ComplexF64
const PB = PureBLAS

# ratio = OB_time / PB_time  (>1 means PB faster). Gate is PB/OB = OB... we report PB/OB = t_ob/t_pb.
ratio(t_pb, t_ob) = t_ob / t_pb

function tri(k)
    A = triu(rand(T, k, k)) + k*I           # upper, well-conditioned
    A
end

# ---- full public trmm!, side R vs L, at square shape ----
function bench_full(k)
    A = tri(k)
    for side in ('R', 'L')
        X = rand(T, k, k)
        al = one(T)
        t_pb = @belapsed PB.trmm!(B, $A; side=$side, uplo='U', transA='N', alpha=$al) setup=(B=copy($X)) evals=1
        t_ob = @belapsed BLAS.trmm!($side, 'U', 'N', 'N', $al, $A, B) setup=(B=copy($X)) evals=1
        println("  trmm! side=$side  k=$k   PB/OB = ", round(ratio(t_pb,t_ob), digits=3),
                "   (pb=$(round(t_pb*1e6,digits=1))us ob=$(round(t_ob*1e6,digits=1))us)")
    end
end

# ---- side-R small (unpacked K-TRIM) vs base (dense packed gemm) in isolation ----
function bench_R_paths(m, k)
    A = tri(k)
    Av = view(A, 1:k, 1:k)
    X = rand(T, m, k)
    al = one(T)
    # OB reference for this exact shape (B m×k := B·A)
    t_ob = @belapsed BLAS.trmm!('R', 'U', 'N', 'N', $al, $A, B) setup=(B=copy($X)) evals=1
    # small (unpacked K-TRIM, the contiguous path)
    t_small = @belapsed PB._trmm_cmplx_small_R!(true, false, false, false, $k, $Av, B) setup=(B=copy($X)) evals=1
    # base (materialize + ONE dense packed complex gemm B·M, 2x flops but peak kernel)
    t_base  = @belapsed PB._trmm_cmplx_base_R!(true, false, false, false, $k, $Av, B) setup=(B=copy($X)) evals=1
    println("  side-R m=$m k=$k:  small(unpacked) PB/OB=", round(ratio(t_small,t_ob),digits=3),
            "   base(dense-gemm) PB/OB=", round(ratio(t_base,t_ob),digits=3))
end

# ---- side-L small vs base, same, for the symmetric reference ----
function bench_L_paths(n, k)
    A = tri(k)
    Av = view(A, 1:k, 1:k)
    X = rand(T, k, n)
    al = one(T)
    t_ob = @belapsed BLAS.trmm!('L', 'U', 'N', 'N', $al, $A, B) setup=(B=copy($X)) evals=1
    t_small = @belapsed PB._trmm_cmplx_small_L!(true, false, false, false, $k, $Av, B) setup=(B=copy($X)) evals=1
    t_base  = @belapsed PB._trmm_cmplx_base_L!(true, false, false, false, $k, $Av, B) setup=(B=copy($X)) evals=1
    println("  side-L n=$n k=$k:  small(unpacked) PB/OB=", round(ratio(t_small,t_ob),digits=3),
            "   base(dense-gemm) PB/OB=", round(ratio(t_base,t_ob),digits=3))
end

# ---- peak packed complex gemm reference at the base shapes (C = A*B, beta=0) ----
function bench_gemm(m, n, k)
    A = rand(T, m, k); B = rand(T, k, n)
    t_pb = @belapsed PB.gemm!(C, $A, $B) setup=(C=zeros(T,$m,$n)) evals=1
    t_ob = @belapsed BLAS.gemm!('N','N', one($T), $A, $B, zero($T), C) setup=(C=zeros(T,$m,$n)) evals=1
    println("  gemm  m=$m n=$n k=$k   PB/OB=", round(ratio(t_pb,t_ob),digits=3))
end

println("=== full public trmm! (square) ===")
for k in (64, 128, 256); bench_full(k); end

println("\n=== side-R isolated: unpacked-small vs dense-base, vary m ===")
for (m,k) in ((128,128),(512,128),(2048,128),(128,64),(512,64))
    bench_R_paths(m,k)
end

println("\n=== side-L isolated (reference), vary n ===")
for (n,k) in ((128,128),(512,128),(2048,128),(128,64),(512,64))
    bench_L_paths(n,k)
end

println("\n=== peak packed complex gemm at base shapes ===")
for (m,n,k) in ((128,128,128),(512,128,128),(128,128,64),(2048,128,128))
    bench_gemm(m,n,k)
end
