using PureBLAS, LinearAlgebra, BenchmarkTools
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)
const T = ComplexF64; const PB = PureBLAS
ratio(t_pb, t_ob) = t_ob / t_pb

# correctness: trmm-R and trsm-R, all uplo×trans×diag, non-square
me_m = 0.0; me_s = 0.0
for ul in ('U', 'L'), ta in ('N', 'T', 'C'), dg in ('N', 'U'), mk in ((128, 128), (200, 96), (64, 64), (300, 128))
    m, k = mk
    A = (ul == 'U' ? triu(rand(T, k, k)) : tril(rand(T, k, k))) + k * I
    dg == 'U' && (A[diagind(A)] .= one(T))
    X = rand(T, m, k)
    B1 = copy(X); PB.trmm!(B1, A; side = 'R', uplo = ul, transA = ta, diag = dg)
    B2 = copy(X); BLAS.trmm!('R', ul, ta, dg, one(T), A, B2)
    global me_m = max(me_m, norm(B1 - B2) / max(norm(B2), eps()))
    C1 = copy(X); PB.trsm!(C1, A; side = 'R', uplo = ul, transA = ta, diag = dg)
    C2 = copy(X); BLAS.trsm!('R', ul, ta, dg, one(T), A, C2)
    global me_s = max(me_s, norm(C1 - C2) / max(norm(C2), eps()))
end
println(
    "CORRECTNESS  trmm-R relerr=", me_m, "  trsm-R relerr=", me_s,
    (me_m < 1.0e-11 && me_s < 1.0e-10) ? "  PASS" : "  FAIL"
)

println("=== trmm! side=R sweep (packed) ===")
for k in (8, 16, 32, 48, 64, 96, 128, 256, 512, 1024, 2048)
    A = triu(rand(T, k, k)) + k * I; X = rand(T, k, k); al = one(T)
    t_pb = @belapsed PB.trmm!(B, $A; side = 'R', uplo = 'U', transA = 'N', alpha = $al) setup = (B = copy($X)) evals = 1
    t_ob = @belapsed BLAS.trmm!('R', 'U', 'N', 'N', $al, $A, B) setup = (B = copy($X)) evals = 1
    println("  k=$k   PB/OB = ", round(ratio(t_pb, t_ob), digits = 3))
end
println("=== trsm! side=R sweep (packed delegation) ===")
for k in (64, 128, 256, 512, 1024, 2048)
    A = triu(rand(T, k, k)) + k * I; X = rand(T, k, k); al = one(T)
    t_pb = @belapsed PB.trsm!(B, $A; side = 'R', uplo = 'U', transA = 'N', alpha = $al) setup = (B = copy($X)) evals = 1
    t_ob = @belapsed BLAS.trsm!('R', 'U', 'N', 'N', $al, $A, B) setup = (B = copy($X)) evals = 1
    println("  k=$k   PB/OB = ", round(ratio(t_pb, t_ob), digits = 3))
end
