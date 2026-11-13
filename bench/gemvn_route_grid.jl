# gemv-N route grid: times OpenBLAS and the gemv-N routes over an (m, n, lda) grid.
# Usage: MS=... NS=... [LDAS=...|PAD=...] [ARMS=ob,rb,pan,alt,auto] julia --project=bench bench/gemvn_route_grid.jl > out.tsv
# REGIME: the gate's own — one operand, hot, `reps` back-to-back gemv-N calls per sample, serial.
# Arms: `ob` OpenBLAS; `rb` forced row-block (`_FKR_gemvn_rb = typemax`); `pan` forced panel with the
# shipped m-inner choice (`= 0`); `alt` forced panel with the m-inner choice inverted; `auto` the
# unforced dispatch. Arm order rotates per round; per-arm statistic = median over rounds of the
# per-round Chairmarks median. Output: one TSV row per cell, times in µs, then whether the forced
# routes agree bitwise and which route `_gemvn_rowblock_route` picks (`rb`/`pan`).
using LinearAlgebra, PureBLAS, Printf, Chairmarks
import PureBLAS as P
include(joinpath(@__DIR__, "measure.jl")); using .Measure: tstat
BLAS.set_num_threads(1); P.set_num_threads(1)

parse_list(s) = [parse(Int, t) for t in split(s, ',')]
const MS = parse_list(get(ENV, "MS", "64,100,131,200,256,384,505,553,768,1024,1420,2048,2880,4096"))
const NS = parse_list(get(ENV, "NS", "8,16,33,64,100,131,200,256,320,436,448,512,768,1024,2048,4096"))
const PAD = parse(Int, get(ENV, "PAD", "0"))          # lda = m + PAD (or ALIGN to a multiple when PAD<0)
const ROUNDS = parse(Int, get(ENV, "ROUNDS", "5"))
const ARMS = Symbol.(split(get(ENV, "ARMS", "ob,rb,pan,alt"), ','))
const MINNER0 = P._GEMVN_MINNER_REF[]
setrb(v) = (P._FKR_gemvn_rb[] = v)
arm!(a) = a === :rb ? (setrb(typemax(Int)); P._GEMVN_MINNER_REF[] = MINNER0) :
    a === :pan ? (setrb(0); P._GEMVN_MINNER_REF[] = MINNER0) :
    a === :alt ? (setrb(0); P._GEMVN_MINNER_REF[] = !MINNER0) :
    (setrb(-1); P._GEMVN_MINNER_REF[] = MINNER0)
@noinline wob(A, x, y, r) = (
    for _ in 1:r
        BLAS.gemv!('N', 1.0, A, x, 0.0, y)
    end; y[1]
)
@noinline wpb(A, x, y, r) = (
    for _ in 1:r
        P.gemv!(y, A, x; alpha = 1.0, beta = 0.0)
    end; y[1]
)
println("# minner_default=$MINNER0 rb_default=$(P._GEMVN_RB) L1=$(P._L1_BYTES) L2=$(P._L2_BYTES) L3=$(P._L3_BYTES) W=$(P._vwidth(Float64)) nvreg=$(P._NVREG)")
println(join(vcat(["m", "n", "lda"], string.(ARMS), ["bits", "route"]), '\t'))
# LDAS (optional, same length as MS) gives each m its own lda; otherwise PAD applies.
const LDAS = haskey(ENV, "LDAS") ? parse_list(ENV["LDAS"]) : [PAD >= 0 ? m + PAD : cld(m, -PAD) * (-PAD) for m in MS]
for (m, lda) in zip(MS, LDAS), n in NS
    Ab = randn(lda, n); A = view(Ab, 1:m, :); x = randn(n); y = randn(m)
    reps = clamp(cld(400_000, m * n), 1, 4000)
    # correctness + bit-equality witness across the three forced PB routes
    ys = map(a -> (arm!(a); yy = zeros(m); P.gemv!(yy, A, x; alpha = 1.3, beta = 0.0); yy), (:rb, :pan, :alt))
    arm!(:none)
    yref = A * x * 1.3
    maximum(abs, ys[1] .- yref) <= 1.0e-9 * (1 + maximum(abs, yref)) || error("wrong at $m $n")
    bits = all(==(reinterpret(UInt64, ys[1])), (reinterpret(UInt64, ys[2]), reinterpret(UInt64, ys[3])))
    route = isdefined(P, :_gemvn_rowblock_route) ? (P._gemvn_rowblock_route(m, n, lda, Float64) ? "rb" : "pan") :
        (n <= P._gemvn_rb() ? "rb" : "pan")
    acc = Dict(a => Float64[] for a in ARMS)
    for r in 1:ROUNDS, a in circshift(ARMS, r - 1)
        arm!(a)
        w = a === :ob ? wob : wpb
        b = @be w(A, x, y, reps) evals = 1 seconds = 0.06 samples = 300
        push!(acc[a], tstat([s.time for s in b.samples]) / reps * 1.0e6)
    end
    arm!(:none)
    @printf("%d\t%d\t%d", m, n, lda)
    for a in ARMS
        @printf("\t%.4f", tstat(acc[a]))
    end
    @printf("\t%s\t%s\n", bits ? "bits=" : "bits!", route)
    flush(stdout)
    Ab = A = nothing; GC.gc(false)
end
