# Is the tiny-n complex gap PER-TILE boundary, or IN-LOOP shuffles? They predict opposite k-scaling.
#
# TWO COMPETING ATTRIBUTIONS of the same ~124 cycles/tile:
#  (a) gemm.jl's own comment: per-tile BOUNDARY setup (clamps/imul). Constant per tile ⇒ its share
#      SHRINKS as k grows, because the k-loop's work grows and the setup does not.
#  (b) mine, from StrictMode kernel_report: the two cross-lane `shufflevector`s in `_deint_cmplx`,
#      issued once per A-load INSIDE the k-loop. Scales with k ⇒ its share is CONSTANT in k.
# They happen to predict the same number at k=32, which is why reading one static profile cannot
# separate them. Varying k does.
#
# PREDICTION, cost per useful flop (time / (m*n*k)):
#   boundary-bound -> falls steeply from k=16 to k=128 and flattens
#   shuffle-bound  -> roughly FLAT across k (both math and shuffles scale together)
#   Mixed -> falls, but to an asymptote clearly above the k=16 -> infinity extrapolation of (a).
#
# Calls `_gemm_cmplx_unpacked_go!` DIRECTLY: the public gemm! routes on max(m,n,k) <= _CGEMM_UNPACK_MAX
# (40 on AVX2), so k>40 through the API would silently switch to the PACKED path and compare two
# different kernels. This probe must hold the kernel fixed and vary only k.
#
# REGIME: L1-resident, freshly written. m=n=32 ComplexF64 with k<=256 means A,B,C are at most 32*256*16
# = 128 KiB total, so the whole working set is L1/L2 and never streams from DRAM — this measures TILE
# ECONOMICS, not bandwidth, and says nothing about large-n zgemm. Call structure: ONE kernel call per
# @be sample (evals=1), fresh operands allocated per sample in the untimed setup, so each sample pays a
# cold-C first touch like the gate's L3 sweep rather than a warm rep loop.
#
# DIRECTION ONLY. A @be closure inlines differently than the production call site; this ranks
# mechanisms, it never produces a gate number.
# Chairmarks only, median only. Run: julia --project=. bench/probes/zgemm32_k_scaling.jl

using PureBLAS, Chairmarks, Statistics, Printf, LinearAlgebra
BLAS.set_num_threads(1)
P = PureBLAS
_med(b) = median(Float64[s.time for s in b.samples])

m = n = 32
@printf("W=%d  _CMR=%d  _CNR_SMALL=%d  _CGEMM_UNPACK_MAX=%d   (m=n=%d fixed, k varies)\n",
        P._vwidth(Float64), P._CMR, P._CNR_SMALL, P._CGEMM_UNPACK_MAX, m)
println("k       us       ns/flop    rel_to_k16   (flat => in-loop/shuffle; falling => per-tile boundary)")
base = 0.0
for k in (16, 32, 64, 128, 256)
    A = randn(ComplexF64, m, k); B = randn(ComplexF64, k, n); C = zeros(ComplexF64, m, n)
    P._gemm_cmplx_unpacked_go!(false, false, m, n, k, 1.0 + 0im, A, B, 0.0 + 0im, C)   # warm this shape
    b = @be (randn(ComplexF64, m, k), randn(ComplexF64, k, n), zeros(ComplexF64, m, n)) (
        c -> P._gemm_cmplx_unpacked_go!(false, false, m, n, k, 1.0 + 0im, c[1], c[2], 0.0 + 0im, c[3])
    ) evals = 1
    t = _med(b)
    nspf = t * 1e9 / (m * n * k)
    global base; k == 16 && (base = nspf)
    @printf("%-7d %-8.3f %-10.5f %.3f\n", k, t * 1e6, nspf, nspf / base)
end
