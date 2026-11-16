# REGIME: L1/L2-resident operands reused across samples (warm), base-clock locked, ONE thread
# (no pool). Locates, per (T, r, k), the column count n at which the few-row split's dot-product tail
# starts to beat the unpacked kernel's vector tail, for C of m = W + r rows (W = `_vwidth(T)`).
# Both arms are forced, so the shipped predicate does not decide which runs:
#   dot    = `_rows_split_serial!` (pack + unpacked top W rows + dot-product tail), tk = 2
#   vector = `_gemm_core!` serial, which is what `gemm!` runs when the split declines.
# Each cell is timed in rotation A B B A (Chairmarks medians), and the ratio is vector/dot (> 1 means
# the dot tail is faster). Prints one TSV row per cell with the pick of the shipped rule
# (`_rows_dot_tail_pays`) and whether `gemm!` can reach the split for that T on this host, then a
# summary of how often the rule's pick is slower than the other arm. Run on each µarch with:
#   JULIA_NUM_THREADS=1 taskset -c <one core> julia --project=. bench/probes/rows_dot_crossover.jl
using PureBLAS, Chairmarks, Statistics, Random, LinearAlgebra
const P = PureBLAS
BLAS.set_num_threads(1)
med(b) = median(b).time * 1.0e9

function cell(::Type{T}, r, k, n) where {T}
    W = P._vwidth(T); m = W + r
    Random.seed!(1)
    A = randn(T, m, k); B = randn(T, k, n); C = zeros(T, m, n)
    α = T(1.3); β = zero(T)
    nb = P._rows_split_nb(T, k, r, 2)
    dot() = GC.@preserve A B C P._rows_split_serial!(P._pm(C), P._pm(A), P._pm(B), α, β, false, r, k, nb, 2)
    vtail() = P._gemm_core!(C, A, B, α, β, false, false, false, false, P._RT_NONE, P._STR_FLOOR, 1)
    dot(); vtail()
    s = 0.04
    a1 = med(@be dot() seconds = s); b1 = med(@be vtail() seconds = s)
    b2 = med(@be vtail() seconds = s); a2 = med(@be dot() seconds = s)
    return (a1 + a2) / 2, (b1 + b2) / 2
end

# Multiples of 8 plus, for Float64, every residue mod the register width above 32, 64 and 128: a C
# whose last column block is narrower than the unpacked tile is where the two arms handle the ragged
# edge differently.
ladder(::Type{T}) where {T} = T === Float64 ?
    sort!(unique!(vcat([8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512], [b + e for b in (32, 64, 128) for e in 1:(P._vwidth(T) - 1)]))) :
    [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]

function main()
    println("# host: W64=", P._vwidth(Float64), " lanes64=", P._at_tail_vw(P._HW, Float64), " NR=", P._NR, " MR=", P._MR,
        " NVREG=", P._NVREG, " L1=", P._L1_BYTES, " MR1_MAX=", P._GEMM_MR1_MAX)
    println("T\tW\tr\tk\tn\tdot_ns\tvec_ns\tvec/dot\trule\treachable"); flush(stdout)
    worst = 1.0; bad = 0; tot = 0
    for T in (Float64, Float32)
        W = P._vwidth(T)
        for r in 1:(W - 1), k in (33, 100, 300), n in ladder(T)
            d, v = cell(T, r, k, n)
            # The shipped rule's pick, and whether `gemm!` can reach the split for this T on this host.
            pick = P._rows_dot_tail_pays(T, W, r, k, n)
            reach = W + r <= P._GEMM_MR1_MAX
            println(T, '\t', W, '\t', r, '\t', k, '\t', n, '\t', round(d; digits = 1), '\t', round(v; digits = 1),
                '\t', round(v / d; digits = 3), '\t', pick ? "dot" : "vec", '\t', reach); flush(stdout)
            # The summary covers the ladder the rule was fitted on (n <= 384) and the reachable cells.
            if reach && n <= 384
                ratio = pick ? v / d : d / v     # chosen arm's speed against the other
                worst = min(worst, ratio); bad += ratio < 0.97; tot += 1
            end
        end
    end
    println("# reachable cells: $tot; rule's pick >3% slower than the other arm: $bad; worst chosen/other ",
        round(worst; digits = 3)); flush(stdout)
    return nothing
end
main()
