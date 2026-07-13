# gesvd (want_vectors) PB/OB across n around the n=128 kink. Run with default _SVD_DC_CROSS, then with a
# lowered value (edit src), to see whether D&C at n=128 beats bdsqr. Boost-locked.
using PureBLAS, LinearAlgebra, Chairmarks, Statistics, Printf
BLAS.set_num_threads(1); mt(b)=median(x.time for x in b.samples)
@noinline pbsvd(A)=PureBLAS.gesvd!(A; want_vectors=true)
@noinline obsvd(A)=LinearAlgebra.LAPACK.gesdd!('A', A)
@printf("_SVD_DC_CROSS = %d\n", PureBLAS._SVD_DC_CROSS)
println("n      PB/OB")
for n in (48,64,96,112,128,144,160,192,256)
    A0=randn(n,n)
    to=mt(@be copy(A0) obsvd seconds=0.4)
    tp=mt(@be copy(A0) pbsvd seconds=0.4)
    @printf("%-6d %.3f\n", n, to/tp)
end
