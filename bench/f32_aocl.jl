# Gate check vs BOTH baselines: PB must be >= max(OB, AOCL) in speed, i.e. PB_time <= min(OB,AOCL)_time.
# Reported ratio = competitor_time / PB_time (>1 = PB faster). The GATE column is min(OB,AOCL)/PB (the
# stricter — measured against whichever competitor is faster). PB is native Julia (LBT-independent), so we
# time PB+OB first, then lbt_forward to AOCL and time it in the same process.
#
# RUN WITH:  julia --threads=1 --project=bench bench/f32_aocl.jl
# `--threads 1` is REQUIRED: AOCL-BLIS(-mt)'s OpenMP pool tracks the Julia thread count, so a default
# multi-thread launch makes AOCL run parallel while PB/OB are single-thread → garbage (dpotrf n2048 "0.28",
# a 3.6× threading artifact; single-thread fair = 1.75). Verified galen: --threads=1 ALONE fixes it (matches
# committed plots_data_galen_aocl). BLAS.set_num_threads(1) below is belt-and-suspenders.
using PureBLAS, Chairmarks, LinearAlgebra
import LinearAlgebra.LAPACK, LinearAlgebra.BLAS
BLAS.set_num_threads(1)
mt(b) = minimum(x.time for x in b.samples)
hpd(T, s) = (A = randn(T, s, s); Matrix(A * A' + s * I))
const CASES = [
    (Float32, 'L', "spotrf-L"), (Float32, 'U', "spotrf-U"),
    (ComplexF64, 'U', "zpotrf-U"), (Float64, 'L', "dpotrf-L"),
]
const SZ = (32, 64, 128, 256, 512, 1024, 2048)

# phase 1: PB + OB (default OpenBLAS backend)
pb = Dict(); ob = Dict()
for (T, u, tag) in CASES, s in SZ
    A0 = hpd(T, s)
    pb[(tag, s)] = mt(@be copy(A0) PureBLAS.potrf!(_; uplo = u) evals = 1 seconds = 0.4)
    ob[(tag, s)] = mt(@be copy(A0) LAPACK.potrf!(u, _) evals = 1 seconds = 0.4)
end
# phase 2: forward LAPACK to AOCL, measure the same shapes
import AOCL_jll
BLAS.lbt_forward(AOCL_jll.aocl_blas_ilp64; clear = true)
BLAS.lbt_forward(AOCL_jll.aocl_lapack_ilp64)
BLAS.set_num_threads(1)   # re-enforce single-thread on the freshly-forwarded AOCL lib
ao = Dict()
for (T, u, tag) in CASES, s in SZ
    A0 = hpd(T, s)
    ao[(tag, s)] = mt(@be copy(A0) LAPACK.potrf!(u, _) evals = 1 seconds = 0.4)
end
# report: OB/PB, AOCL/PB, GATE=min(OB,AOCL)/PB
for (T, u, tag) in CASES
    println(rpad(tag, 10), "  [comp_t/PB_t; * = below gate min(OB,AOCL)/PB]")
    for s in SZ
        rob = ob[(tag, s)] / pb[(tag, s)]; rao = ao[(tag, s)] / pb[(tag, s)]
        gate = min(rob, rao)   # against the faster competitor
        flag = gate < 1.0 ? "*" : " "
        println("   n=$(rpad(s, 5)) OB=$(round(rob, digits = 2)) AOCL=$(round(rao, digits = 2))  GATE=$(round(gate, digits = 2))$flag")
    end
end
