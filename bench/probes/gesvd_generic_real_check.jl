# REGIME: correctness only, no timing. Generic singular VALUES must match LinearAlgebra's svdvals
# (computed in Float64) to the precision of the element type. Float16 exercises the generic scalar path
# a ForwardDiff.Dual takes; BigFloat exercises the non-bits guards; Float64 must be unchanged.
using PureBLAS, LinearAlgebra, Random
const P = PureBLAS
Random.seed!(2718)
bad = 0
for T in (Float64, Float32, Float16, BigFloat)
    tol = T === Float64 ? 1e-12 : T === Float32 ? 1e-5 : T === Float16 ? 5e-2 : 1e-12
    for (m, n) in ((1,1), (6,6), (12,7), (7,12), (20,20), (33,17))
        A64 = randn(m, n)
        ref = svdvals(A64)                               # Float64 oracle
        A = T.(A64)
        S = T === Float64 ? P.gesvd!(copy(A); want_vectors=false)[1] :
                            P.gesvd!(copy(A); want_vectors=false)[1]
        e = maximum(abs, Float64.(S) .- ref) / max(ref[1], 1e-300)
        ok = e < tol; ok || (global bad += 1)
        println(rpad(string(T),9), " ", rpad("$(m)x$(n)",7), " rel σ err = ", rpad(e, 24), ok ? "" : "  <-- FAIL")
    end
end
# vectors must refuse loudly rather than silently return values
try
    P.gesvd!(Float16.(randn(4,4)))
    println("ERROR: want_vectors=true should have thrown")
    global bad += 1
catch err
    println("want_vectors=true on Float16 correctly throws: ", first(split(sprint(showerror, err), '\n')))
end
println(bad == 0 ? "GESVD-GENERIC-REAL: PASS" : "GESVD-GENERIC-REAL: $bad FAILURES")
