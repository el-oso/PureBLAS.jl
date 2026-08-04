# ONE GATE CELL, ONE PROCESS, CHEAPLY — the replication runner for hypothesis-tested gating.
#
# WHY. Deciding whether a cell really misses needs K independent PROCESSES: the per-process page
# colouring is re-rolled each time (THP is `madvise`-only here and Julia never madvises, so an 8 MB
# array is always 4 KB-backed and its physical placement is a fresh draw), and at cells whose working
# set straddles L3 that placement is worth ~3pp — larger than the gaps we are chasing.
#
# IN-PROCESS ROUNDS DO NOT SUBSTITUTE. Measured 2026-08-04 at axpy n=1e6: 10 in-process rounds read
# 1.0153 [1.0000, 1.0214] where 10 fresh processes read 0.9868 with 9/10 below 1.0. Not tighter —
# OPPOSITE SIGN. A round reuses one placement draw for both arms; only a new process re-rolls it.
#
# WHY NOT plots.jl. A single-cell `bench op=... size=...` run costs ~34 s, of which the Chairmarks
# windows are ~1 s: `size=1000` (trivial work) still costs 33.3 s, and an IP profile of the run is
# dominated by @Compiler (abstractinterpretation/typeinfer/ssair) plus @Base/staticdata
# verify_method_graph — JIT of the script's machinery plus pkgimage validation, then an 11 MB cache
# parse and full rewrite to update ONE line. `using PureBLAS, LinearAlgebra` alone is 0.62 s, so
# essentially all of it is plots.jl's apparatus, not the measurement.
#
# So: same regime as plots.jl, same estimator, same fresh-input `@be evals=1`, no cache, no render.
# Emits ONE line per process; the driver loop appends them and bench/probes/cell_hypothesis.jl decides.
#
#   julia --project=bench bench/cellrep.jl axpy 1000000 >> log
#
# The ratio is (faster reference) / pb, matching plots.jl's gate definition.
#
# THIS IS A SCREEN, NOT THE NUMBER OF RECORD. Validated against 10 plots.jl replications of the same
# cell (axpy n=1e6, wintermute, freq-locked, 2026-08-04):
#     plots.jl  median 0.9868  IQR [0.9816, 0.9925]  spread 3.23pp   9/10 below 1.0  sign p=0.0215
#     cellrep   median 0.9676  IQR [0.9602, 0.9798]  spread 4.73pp  10/10 below 1.0  sign p=0.00195
# Same VERDICT, and cellrep is more decisive (P99 in ~85 s for K=10, where plots.jl needs ~5.7 min and
# still only reaches p=0.02). But the IQRs DO NOT OVERLAP: cellrep runs ~2pp pessimistic for PB, almost
# certainly because plots.jl pools 8 rounds of `samples=400, seconds=0.15` where this takes one median
# of `samples=96, seconds=1.0`. So:
#   * USE for "is this cell really failing?" — direction and significance, cheaply, before spending an
#     afternoon on a lever.
#   * DO NOT quote a cellrep ratio as a gate ratio, and do not put one in the cache, a table or the docs.
#     The published number stays plots.jl's.
using LinearAlgebra, PureBLAS, Statistics, Printf
using Chairmarks: @be                        # the ONLY thing allowed to touch a clock (CLAUDE.md)
import PureBLAS as P
include(joinpath(@__DIR__, "measure.jl"))
using .Measure
const B = LinearAlgebra.BLAS
B.set_num_threads(1)
# Capture the ORIGINAL BLAS before any forwarding — reading loaded_libs[1] later returns whatever was
# forwarded last, which would silently make the "openblas" arm measure AOCL twice.
const _OPENBLAS = B.get_config().loaded_libs[1].libname

const OP = ARGS[1]
const N = parse(Int, ARGS[2])
_l1rep(s) = clamp(8_000_000 ÷ s, 30, 20000)          # plots.jl's _L1REP — the regime must match

# op => (setup, pb work, reference work). Reference goes through LinearAlgebra.BLAS, which LBT points
# at whichever library is forwarded below.
const OPS = Dict(
    "axpy" => (() -> (randn(N), randn(N)),
               (c, m) -> (for _ in 1:m
                    P.axpy!(c[2], 1.7, c[1])
                end),
               (c, m) -> (for _ in 1:m
                    B.axpy!(1.7, c[1], c[2])
                end)),
    "dot" => (() -> (randn(N), randn(N)),
              (c, m) -> (s = 0.0; for _ in 1:m
                   s += P.dot(c[1], c[2])
               end; s),
              (c, m) -> (s = 0.0; for _ in 1:m
                   s += B.dot(c[1], c[2])
               end; s)),
    "scal" => (() -> (randn(N),),
               (c, m) -> (for _ in 1:m
                    P.scal!(1.0000001, c[1])
                end),
               (c, m) -> (for _ in 1:m
                    B.scal!(1.0000001, c[1])
                end)),
    "asum" => (() -> (randn(N),),
               (c, m) -> (s = 0.0; for _ in 1:m
                    s += P.asum(c[1])
                end; s),
               (c, m) -> (s = 0.0; for _ in 1:m
                    s += B.asum(c[1])
                end; s)),
    "iamax" => (() -> (randn(N),),
                (c, m) -> (s = 0; for _ in 1:m
                     s += P.iamax(c[1])
                 end; s),
                (c, m) -> (s = 0; for _ in 1:m
                     s += B.iamax(c[1])
                 end; s)),
    # Complex CL1 — call forms copied from plots.jl's cl1 block so the regime matches exactly
    # (PureBLAS.dot is the conjugating one, matching BLAS.dotc).
    "zaxpy" => (() -> (randn(ComplexF64, N), randn(ComplexF64, N)),
                (c, m) -> (for _ in 1:m
                     P.axpy!(c[2], 1.7 + 0.3im, c[1])
                 end),
                (c, m) -> (for _ in 1:m
                     B.axpy!(1.7 + 0.3im, c[1], c[2])
                 end)),
    "zscal" => (() -> (randn(ComplexF64, N),),
                (c, m) -> (for _ in 1:m
                     P.scal!(1.0000001 + 0im, c[1])
                 end),
                (c, m) -> (for _ in 1:m
                     B.scal!(1.0000001 + 0im, c[1])
                 end)),
    "zdotc" => (() -> (randn(ComplexF64, N), randn(ComplexF64, N)),
                (c, m) -> (s = zero(ComplexF64); for _ in 1:m
                     s += P.dot(c[1], c[2])
                 end; real(s)),
                (c, m) -> (s = zero(ComplexF64); for _ in 1:m
                     s += B.dotc(c[1], c[2])
                 end; real(s))),
    "dzasum" => (() -> (randn(ComplexF64, N),),
                 (c, m) -> (s = 0.0; for _ in 1:m
                      s += P.asum(c[1])
                  end; s),
                 (c, m) -> (s = 0.0; for _ in 1:m
                      s += B.asum(c[1])
                  end; s)),
)
haskey(OPS, OP) || error("cellrep: unknown op $OP (have: $(join(sort(collect(keys(OPS))), ", ")))")
mk, pbw, refw = OPS[OP]
reps = _l1rep(N)

"Median-of-samples time for one arm, Chairmarks, fresh input per sample (plots.jl's `evals=1` regime)."
function armtime(work)
    b = @be mk() (c -> work(c, reps)) evals = 1 samples = 96 seconds = 1.0
    return Measure.tstat(Float64[s.time for s in b.samples])
end

# Arms back-to-back in ONE process so the ratio cancels this process's common-mode state; the ACROSS
# process variation is what the K replications sample.
using AOCL_jll
out = Dict{String, Float64}()
for (nm, lib) in ("openblas" => nothing, "aocl" => AOCL_jll.aocl_blas_ilp64)
    isnothing(lib) ? B.lbt_forward(_OPENBLAS; clear = true) :
        B.lbt_forward(lib; clear = true)      # clear=true, as plots.jl:66 does — clear=false leaves routing ambiguous
    out[nm] = armtime(refw)
end
tpb = armtime(pbw)                            # PB is a direct Julia call; LBT state is irrelevant to it
ratio = minimum(values(out)) / tpb            # the FASTER reference sets the gate
@printf("%s\t%d\t%.6g\t%.6g\t%.6g\t%.6g\n", OP, N, ratio, tpb, out["openblas"], out["aocl"])
