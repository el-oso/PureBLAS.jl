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
import PureBLAS: _ger_panel!, _vwidth, _L3_BYTES
BLAS.set_num_threads(1)
const DRYRUN = "dryrun" in ARGS

# ── ger DRAM path with an EXPLICIT stream count NP (the thing we're tuning): the m-inner panel over NP
# concurrent wide-SIMD A-column RMW streams, prefetch off. The optimal NP is an intrinsic per-core property
# (measured opposite-sign across µarchs — Zen5→1, Zen3→4, Zen4→8), so it must be measured, not derived.
@noinline function _ger_np!(A::Matrix{T}, x, y, ::Val{NP}) where {T, NP}
    m, n = size(A)
    GC.@preserve A x y begin
        Ap = pointer(A); xp = pointer(x); yp = pointer(y); lda = stride(A, 2); jc = 0
        while jc + NP <= n; _ger_panel!(Ap, lda, xp, yp, jc, m, one(T), 0, Val(NP), Val(4)); jc += NP; end
        while jc < n; _ger_panel!(Ap, lda, xp, yp, jc, m, one(T), 0, Val(1), Val(4)); jc += 1; end
    end
    return A
end
runnp(A, x, y, NP::Int) = NP == 1 ? _ger_np!(A, x, y, Val(1)) : NP == 2 ? _ger_np!(A, x, y, Val(2)) :
    NP == 4 ? _ger_np!(A, x, y, Val(4)) : _ger_np!(A, x, y, Val(8))

# Winner = the NP that MINIMIZES PB time (OB fixed ⇒ min PB-time = max PB/OB gate ratio). Swept only at the
# DRAM sizes (A > L3) — the regime the panel serves; cache-resident ger stays on the per-column path.
function calibrate_ger_np(::Type{T} = Float64) where {T}
    W = _vwidth(T); cand = [1, 2, 4, 8]
    n0 = ceil(Int, sqrt(_L3_BYTES / sizeof(T))); sizes = (max(2048, n0 + (W - n0 % W) % W), 4096)
    probs = [(randn(T, n, n), randn(T, n), randn(T, n), n) for n in sizes]
    @printf("calibrate ger stream-count NP (%s, sizes %s, DRAM > %.0f MB):\n", T, sizes, _L3_BYTES / 2^20)
    scores = zeros(length(cand))
    for (ci, NP) in pairs(cand)
        for (A, x, y, n) in probs
            scores[ci] += minimum(@be runnp(A, x, y, NP) seconds = 0.4).time / n^2
        end
        @printf("  NP = %d  →  %.3e\n", NP, scores[ci])
    end
    best = cand[argmin(scores)]
    @printf("  ⇒ BEST ger_panel_np = %d\n", best)
    return best
end

# ── run every calibration, collect (key => value), write once ──────────────────────────────────────────
prefs = Pair{String,Any}[]
push!(prefs, "ger_panel_np" => calibrate_ger_np())
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
