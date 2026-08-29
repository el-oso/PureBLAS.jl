# REGIME: correctness + witness only, no timing. Flips the force Ref in-process, which is exactly what
# the env var does at __init__, so both arms are the shipped code paths.
using PureBLAS, LinearAlgebra, Random
const P = PureBLAS
Random.seed!(1234)
L3 = P._L3_BYTES
println("_L3_BYTES = ", L3, " (", round(L3/2^20, digits=1), " MiB)")
for den in (1, 2)
    P._FKR_cger_cold_den[] = den
    print("den=", P._cger_cold_den(), "  cold(n): ")
    for n in (512, 1000, 1024, 1448, 2048)
        print(n, "=", n*n*16 >= L3 ÷ P._cger_cold_den(), " ")
    end
    println()
end
# Both arms must compute the same thing — the divisor only selects which existing arm runs.
m = n = 1000
A0 = randn(ComplexF64, m, n); x = randn(ComplexF64, m); y = randn(ComplexF64, n); α = 0.7 - 0.3im
res = Dict{Int, Matrix{ComplexF64}}()
for den in (1, 2)
    P._FKR_cger_cold_den[] = den
    A = copy(A0)
    P._ger_cmplx!(m, n, α, x, y, A, false)
    res[den] = A
end
ref = A0 + α * x * transpose(y)
e1 = maximum(abs, res[1] .- ref); e2 = maximum(abs, res[2] .- ref)
d  = maximum(abs, res[1] .- res[2])
println("max|den1 - ref| = ", e1)
println("max|den2 - ref| = ", e2)
println("max|den1 - den2| = ", d)
println(d == 0 ? "IDENTICAL — arms agree bitwise" : "DIFFER (expected: same math, maybe different order)")
P._FKR_cger_cold_den[] = -1
