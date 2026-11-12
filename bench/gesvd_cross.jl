# gesvd (want_vectors) PB/OB across n around the n=128 kink. To compare the two bidiagonal solvers,
# run as is and again with `PureBLAS._svd_vectors_by_qr` edited in src. Boost-locked.
using PureBLAS, LinearAlgebra, Chairmarks, Statistics, Printf
BLAS.set_num_threads(1); mt(b) = median(x.time for x in b.samples)
@noinline pbsvd(A) = PureBLAS.gesvd!(A; want_vectors = true)
@noinline obsvd(A) = LinearAlgebra.LAPACK.gesdd!('A', A)
println("n      PB/OB")
for n in (48, 64, 96, 112, 128, 144, 160, 192, 256)
    A0 = randn(n, n)
    to = mt(@be copy(A0) obsvd seconds = 0.4)
    tp = mt(@be copy(A0) pbsvd seconds = 0.4)
    @printf("%-6d %.3f\n", n, to / tp)
end
