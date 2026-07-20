# Calibrate _CSYRK_UNPACK_MAX on AVX2: compare the unpacked-tri kernel (_ctri_unpacked!) vs the packed
# path (_csyrk_packed!) vs OpenBLAS, in ONE process (no per-pref reload). herk trans='N' (the amplifying
# path). Correctness checked vs OB. Boost MUST be OFF. Find the largest n where unpacked-tri ≥ OB AND
# ≥ packed → that's the AVX2 cutoff.
using PureBLAS, LinearAlgebra, Chairmarks, Statistics, Printf
const B = LinearAlgebra.BLAS; BLAS.set_num_threads(1)
const T = ComplexF64
mt(b) = median(x.time for x in b.samples)
band = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256)

@printf "AVX2 calib: _vwidth=%d  current _CSYRK_UNPACK_MAX=%d  _CSYRK_PACK_CUT=%d\n\n" PureBLAS._vwidth(Float64) PureBLAS._CSYRK_UNPACK_MAX PureBLAS._CSYRK_PACK_CUT
println("=== herk (rank-1) ===")
println("n      unpk/OB  packed/OB  win     relerr")
for n in band
    k = n
    A = randn(T, n, k)
    Cob = zeros(T, n, n); B.herk!('U', 'N', 1.0, A, 0.0, Cob)
    Cu = zeros(T, n, n); PureBLAS._ctri_unpacked!(true, true, 1.0, A, Cu, k)
    Cp = zeros(T, n, n); PureBLAS._csyrk_packed!(true, false, true, 1.0, A, Cp, k)
    err = 0.0; for j in 1:n, i in 1:j
        err = max(err, abs(Cu[i, j] - Cob[i, j]))
    end
    relerr = err / (norm(Cob) + eps())
    to = mt(@be B.herk!('U', 'N', 1.0, A, 0.0, Cob) seconds = 0.4)
    tu = mt(@be (fill!(Cu, 0); PureBLAS._ctri_unpacked!(true, true, 1.0, A, Cu, k)) seconds = 0.4)
    tp = mt(@be (fill!(Cp, 0); PureBLAS._csyrk_packed!(true, false, true, 1.0, A, Cp, k)) seconds = 0.4)
    ru = to / tu; rp = to / tp
    win = ru >= 1.0 && ru >= rp ? "UNPK" : (rp >= 1.0 ? "packed" : "both<1")
    @printf "%-6d %.3f    %.3f     %-7s %.1e\n" n ru rp win relerr
end
println("\n=== her2k (rank-2) ===")
println("n      unpk/OB  packed/OB  win     relerr")
for n in band
    k = n
    A = randn(T, n, k); Bm = randn(T, n, k)
    Cob = zeros(T, n, n); B.her2k!('U', 'N', ComplexF64(1.0), A, Bm, 0.0, Cob)
    Cu = zeros(T, n, n); PureBLAS._ctri2_unpacked!(true, true, ComplexF64(1.0), A, Bm, Cu, k)
    Cp = zeros(T, n, n); PureBLAS._csyr2k_packed!(true, false, true, ComplexF64(1.0), A, Bm, Cp, k)
    err = 0.0; for j in 1:n, i in 1:j
        err = max(err, abs(Cu[i, j] - Cob[i, j]))
    end
    relerr = err / (norm(Cob) + eps())
    to = mt(@be B.her2k!('U', 'N', ComplexF64(1.0), A, Bm, 0.0, Cob) seconds = 0.4)
    tu = mt(@be (fill!(Cu, 0); PureBLAS._ctri2_unpacked!(true, true, ComplexF64(1.0), A, Bm, Cu, k)) seconds = 0.4)
    tp = mt(@be (fill!(Cp, 0); PureBLAS._csyr2k_packed!(true, false, true, ComplexF64(1.0), A, Bm, Cp, k)) seconds = 0.4)
    ru = to / tu; rp = to / tp
    win = ru >= 1.0 && ru >= rp ? "UNPK" : (rp >= 1.0 ? "packed" : "both<1")
    @printf "%-6d %.3f    %.3f     %-7s %.1e\n" n ru rp win relerr
end
