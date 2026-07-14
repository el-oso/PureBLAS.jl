# gesvd VALUES (want_vectors=false) PB/OB gate, real + complex, small n. Boost-locked, core-pinned.
using PureBLAS, LinearAlgebra, Chairmarks, Statistics, Printf
BLAS.set_num_threads(1)
mt(b) = minimum(x.time for x in b.samples)

@noinline pbval(A, S) = PureBLAS.gesvd_vals!(A, S)
@noinline obval(A) = LinearAlgebra.LAPACK.gesdd!('N', A)

function run(T, ns)
    @printf("%-8s  n     PB(us)   OB(us)   PB/OB   maxrelerr\n", string(T))
    for n in ns
        A0 = T <: Complex ? randn(T, n, n) : randn(n, n)
        S = Vector{real(T)}(undef, n)
        # correctness vs LinearAlgebra oracle
        sref = svdvals(copy(A0))
        PureBLAS.gesvd_vals!(copy(A0), S)
        err = maximum(abs.(sort(S, rev=true) .- sref) ./ max.(sref, eps()))
        tp = mt(@be (copy(A0), Vector{real(T)}(undef, n)) x -> pbval(x[1], x[2]) seconds=0.5)
        to = mt(@be copy(A0) obval seconds=0.5)
        @printf("          %-5d %7.2f %7.2f  %.3f   %.2e\n", n, tp*1e6, to*1e6, to/tp, err)
    end
end

ns = (8, 16, 32, 48, 64, 128)
run(Float64, ns)
run(ComplexF64, ns)
run(ComplexF32, ns)
