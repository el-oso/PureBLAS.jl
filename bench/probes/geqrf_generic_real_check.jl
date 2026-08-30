# REGIME: correctness only, no timing. geqrf! must produce the FAER tau convention for EVERY element
# type, not just Float64 — H_k = I − v·vᵀ/τ with τ=Inf meaning identity. The oracle below is the same
# `recon` test/lapack_tests.jl uses, so this probe checks the same contract the suite does.
# Float16 exercises the generic scalar path a ForwardDiff.Dual takes (bits, non-BlasReal); BigFloat
# exercises the non-bits pointer guards; Float64 must be unchanged.
using PureBLAS, LinearAlgebra, Random
const P = PureBLAS
Random.seed!(31415)

function recon(F, tau, ::Type{T}) where {T}
    m, n = size(F); k = min(m, n)
    R = T[i <= j ? F[i, j] : zero(T) for i in 1:m, j in 1:n]
    for kk in k:-1:1
        isfinite(tau[kk]) || continue                       # τ=Inf ⇒ H=I
        v = zeros(T, m); v[kk] = one(T); v[(kk + 1):m] = F[(kk + 1):m, kk]
        R .-= (v * (transpose(v) * R)) ./ tau[kk]           # faer: divide by τ
    end
    return R
end

bad = 0
for T in (Float64, Float32, Float16, BigFloat)
    tol = T === Float64 ? 1e-11 : T === Float32 ? 1e-4 : T === Float16 ? 5e-2 : 1e-30
    for (m, n) in ((1,1), (8,8), (16,8), (8,16), (33,20), (64,64))
        A0 = T.(randn(m, n)); F = copy(A0); tau = zeros(T, min(m, n))
        P.geqrf!(F, tau)
        e = Float64(maximum(abs, recon(F, tau, T) .- A0) / max(Float64(maximum(abs, A0)), 1e-300))
        ok = e < tol; ok || (global bad += 1)
        println(rpad(string(T),9), " ", rpad("$(m)x$(n)",7), " rel|A-QR| = ", rpad(e, 24), ok ? "" : "  <-- FAIL")
    end
end
println(bad == 0 ? "GEQRF-GENERIC-REAL: PASS" : "GEQRF-GENERIC-REAL: $bad FAILURES")
