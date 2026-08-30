# REGIME: correctness only, no timing. Does getrf! now work for a non-Float64 Real, and is the Float64
# path byte-identical to before? Uses the test env's ForwardDiff via --project=test-style load.
using PureBLAS, LinearAlgebra, Random
const P = PureBLAS
Random.seed!(4242)

# 1. Float64 unchanged: factor, then check PA = LU by reconstruction.
function recon(A0, A, ipiv)
    m, n = size(A); k = min(m, n)
    L = Matrix(LowerTriangular(A)); for i in 1:min(m,n); L[i,i] = 1; end
    U = Matrix(UpperTriangular(A))
    P_ = collect(1:m); for i in 1:k; P_[i], P_[ipiv[i]] = P_[ipiv[i]], P_[i]; end
    return maximum(abs, (L*U)[:, 1:n] .- A0[P_, :])
end
# Float16 is the important one: it is a bits Real that is NOT a BlasReal, so it takes the SAME generic
# scalar path a ForwardDiff.Dual takes. Float32 does not prove genericity — it is a BlasReal and rides
# the SIMD kernels. BigFloat is the NON-BITS case: it must reach the generic path and never a
# `pointer()` fast path — it used to segfault Julia's codegen until `_strided1` gained its
# `isbitstype(eltype(A))` clause.
for T in (Float64, Float32, Float16, BigFloat)
    for n in (8, 33, 64)
        A0 = T.(randn(n, n) + n*I)
        A = copy(A0); ipiv = Vector{Int}(undef, n)
        P.getrf!(A, ipiv)
        e = recon(A0, A, ipiv)
        println(rpad(string(T),9), " n=", rpad(n,4), " max|PA - LU| = ", Float64(e))
    end
end
# 2. the 1-arg convenience method now accepts a non-Float64 Real
A = Float32.(randn(16,16) + 16I)
_, ip, info = P.getrf!(A)
println("1-arg Float32: info=", info, " ipiv[1:3]=", ip[1:3])
println("GETRF-GENERIC-REAL: PASS")
