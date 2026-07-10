# Per-box AUTOTUNE of PureBLAS parameters that a hardware feature-bit can't derive (memory-latency-class
# things like the ger DRAM prefetch distance). Measures on THIS machine and writes the winners into the
# active project's LocalPreferences.toml, which @load_preference then bakes into consts (and a per-µarch
# frozen .so carries them). This is Option "A": the trim-safe half of the A+C hybrid — the SAME winners a
# future load-time auto-calibration (Mode-2 only) would pick, just written once by hand here.
#
# RUN IT ON THE LOCKED BOX (after `sudo bench/fleet_freqlock.sh lock`) so the measurement is at the stable
# clock — a boosting/throttling clock gives a wrong winner. `verify` first that the box is ✅ locked.
#
#   taskset -c <core> julia --project=bench bench/calibrate.jl          # measure + WRITE the preferences
#   taskset -c <core> julia --project=bench bench/calibrate.jl dryrun   # measure + PRINT only (no write)
#
# The parameters are swept as RUNTIME kernel args (no recompile per candidate) — that's why this is fast.

using PureBLAS, LinearAlgebra, Printf, Chairmarks, TOML
import PureBLAS: _axpy_simd!, _vwidth, _L3_BYTES
BLAS.set_num_threads(1)
const DRYRUN = "dryrun" in ARGS

# ── ger with an EXPLICIT runtime prefetch distance `pf` (elements) — the thing we're tuning. Mirrors
# `_ger_simd!`'s per-column axpy but with pf passed in instead of read from the (const) preference.
@noinline function _ger_pf!(A::Matrix{T}, x, y, pf::Int) where {T}
    m, n = size(A); sz = sizeof(T)
    GC.@preserve A x y begin
        Ap = pointer(A); xp = pointer(x); yp = pointer(y); lda = stride(A, 2)
        @inbounds for j in 1:n
            ayj = unsafe_load(yp, j)
            _axpy_simd!(m, ayj, xp, Ap + (j - 1) * lda * sz, pf)
        end
    end
    return A
end

# Winner = the distance that MINIMIZES PB time (OB is fixed, so min PB-time ⇒ max PB/OB gate ratio). Only
# DRAM-bound sizes (A > L3) are swept — that's the only regime the prefetch fires in (`_ger_simd!` gates it
# on m·n·sizeof > L3), so cache-resident sizes are unaffected by the distance and must not vote.
function calibrate_ger_prefetch(::Type{T} = Float64) where {T}
    W = _vwidth(T)
    # candidate distances in BYTES (0 = prefetch OFF, e.g. what low-latency DDR5 wants). Spans DDR4/DDR5/
    # LPDDR5x optima (~1 page … ~2 pages) plus off.
    cand = [0, 512, 1024, 2048, 4096, 8192, 16384]
    # smallest square n with A = n²·sizeof > L3, then one larger — the two DRAM points prefetch acts on.
    n0 = ceil(Int, sqrt(_L3_BYTES / sizeof(T))); sizes = (max(2048, n0 + (W - n0 % W) % W), 4096)
    probs = [(randn(T, n, n), randn(T, n), randn(T, n), n) for n in sizes]
    @printf("calibrate ger prefetch (%s, sizes %s, DRAM > %.0f MB):\n", T, sizes, _L3_BYTES / 2^20)
    scores = zeros(length(cand))
    for (ci, cb) in pairs(cand)
        pf = cb ÷ sizeof(T)
        for (A, x, y, n) in probs
            scores[ci] += minimum(@be _ger_pf!(A, x, y, pf) seconds = 0.4).time / n^2   # per-element time
        end
        @printf("  pf = %6d B  →  %.3e\n", cb, scores[ci])
    end
    best = cand[argmin(scores)]
    @printf("  ⇒ BEST ger_prefetch_bytes = %d B\n", best)
    return best
end

# ── run every calibration, collect (key => value), write once ──────────────────────────────────────────
prefs = Pair{String,Any}[]
push!(prefs, "ger_prefetch_bytes" => calibrate_ger_prefetch())
# (add further runtime-swept params here as they arrive — same pattern)

const LP = joinpath(dirname(Base.active_project()), "LocalPreferences.toml")   # active project (bench/) is where @load_preference reads
if DRYRUN
    println("\n[dryrun] would write to $LP under [PureBLAS]:")
    foreach(p -> println("  $(p.first) = $(p.second)"), prefs)
else
    d = isfile(LP) ? TOML.parsefile(LP) : Dict{String,Any}()      # merge — preserve other packages' + other PureBLAS prefs
    sect = get!(d, "PureBLAS", Dict{String,Any}())
    for p in prefs; sect[p.first] = p.second; end
    open(io -> TOML.print(io, d), LP, "w")
    println("\nwrote $(length(prefs)) preference(s) → $LP  [PureBLAS]")
    println("(reload PureBLAS to bake them; a per-µarch frozen .so carries them forward)")
end
