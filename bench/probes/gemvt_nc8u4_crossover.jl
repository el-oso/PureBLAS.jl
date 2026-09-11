# REGIME: gate-exact. Fresh operands per SAMPLE (cold, like `sweep_heavy`), `reps` inside the timed
# core, ONE SIZE AND ONE ARM PER PROCESS (argv[1]=n, arm chosen by the PUREBLAS_FORCE_* env of the
# caller). reps mirrors the gate's `_L2REP = clamp(4e8 ÷ n², 30, 20000)` exactly so a probe number is
# directly comparable to a `plots.jl` cell.
#
# WHY. `_GEMVT_NC_DEEP = 8` / `_GEMVT_U_DEEP = 4` — AOCL's shape, 32 rows per iteration, one x-load per
# 8 FMAs — already SHIPS, but `_gemvt_deep` gates it on `A <= L2`. A gate-harness A/B on wintermute
# (2026-09-11, arms=pb, one size per process) measured that same shape forced through the blocked
# ladder at sizes the gate excludes:
#     n=2100  (A/L3 = 2.1)  36.7 -> 43.4 GB/s  (+18.1%)
#     n=4096  (A/L3 = 8.0)  27.3 -> 25.2 GB/s  (-7.4%)
# So the `A <= L2` cutoff is too early, and there is a crossover between 2.1x and 8.0x L3. Candidate
# mechanism: 8 concurrent column streams are affordable while part of A is still L3-served and stop
# being so once service is pure DRAM (`pureblas-dram-stream-count` measured Zen4 falling past 6
# streams). This probe locates the crossover so the gate can be a DERIVED capacity rule rather than two
# fitted points.
#
# TRUST: n=2100 and n=4096 are measured here ONLY as calibration against the gate numbers above. If the
# probe does not reproduce them the intermediate sizes mean nothing and must be discarded — a probe that
# disagrees with the gate is wrong about the regime, not a new result (probe-callsite-inlining).

using PureBLAS, Printf, Random, LinearAlgebra
include(joinpath(@__DIR__, "..", "measure.jl"))

const T = Float64
const M = parse(Int, get(ARGS, 1, "3072"))
const N = parse(Int, length(ARGS) >= 2 ? ARGS[2] : ARGS[1])
const REPS = clamp(400_000_000 ÷ (M * N), 30, 20000)   # _L2REP generalised to the m*n footprint

BLAS.set_num_threads(1)
Random.seed!(20260911)

blk = !PureBLAS._gemvt_perscan(M, N, T)
@printf("host=%s  m=%d n=%d  reps=%d  A=%.1f MiB  A/L3=%.2f  x=%.1f KiB  x/L1=%.2f  blk=%s  nc=%d u=%d pf=%d\n",
        gethostname(), M, N, REPS, M*N*sizeof(T)/2^20, M*N*sizeof(T)/PureBLAS._L3_BYTES,
        M*sizeof(T)/1024, M*sizeof(T)/PureBLAS._L1_BYTES, blk,
        PureBLAS._gemvt_nc(), PureBLAS._gemvt_u(), PureBLAS._gemvt_pf())
blk || error("m=$M n=$N routes PER-COLUMN (blk=false): NC does not apply, this arm would be INERT.")

# Witness: the kernel must compute gemv-T correctly under the forced shape, or the timing is of nothing.
let A = randn(T, M, N), x = randn(T, M), y = zeros(T, N)
    PureBLAS.gemv!(y, A, x; trans = 'T')
    ref = A' * x
    rel = maximum(abs.(y .- ref)) / max(1e-300, maximum(abs.(ref)))
    rel < 1e-10 || error("gemv-T wrong under this arm: rel=$rel")
    println("witness: gemv! trans='T' matches A'x (rel=", round(rel, sigdigits = 3), ")")
end

setup = () -> (randn(T, M, N), randn(T, M), zeros(T, N))
r = Measure.ab(["gemvT" => (c -> (for _ in 1:REPS
    PureBLAS.gemv!(c[3], c[1], c[2]; trans = 'T')
end; c[3][1]))]; rounds = 8, setup = setup)
@printf("  m=%d n=%d  %.4g s  %.2f GB/s   (estimator=%s)\n",
        M, N, r[1].secs, M * N * sizeof(T) * REPS / r[1].secs / 1e9, Measure.ESTIMATOR)
