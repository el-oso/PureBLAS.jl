# Per-machine AUTOTUNE. Measures the knobs that are NOT derivable from detected hardware and writes the
# winners as Preferences pins. Run it once per machine; `PureBLAS.tune!()` drives it.
#
#   julia --project=bench bench/calibrate.jl                 # measure + WRITE the preferences
#   julia --project=bench bench/calibrate.jl dryrun          # measure + PRINT only
#   julia --project=bench bench/calibrate.jl emit=/tmp/x.toml  # measure, write RESULT ONLY to that file
#
# `emit=` exists so tune!() can run this in several INDEPENDENT PROCESSES and pin only what they agree
# on. That is not belt-and-braces: the retired OncePerProcess tier failed precisely because a per-process
# draw (page placement, allocator state) can change the winner, and one knob picked a 17%-slower arm in
# 7 of 9 processes. More reps inside one process cannot see that; separate processes can.
#
# WHY THIS EXISTS AT ALL. `ger_panel_np` is measured 8 / 4 / 1 on Zen4 / Zen3 / Zen5 — and the first two
# of those boxes share L2, L3, SIMD width and register count. No formula reproduces it. Shipping a single
# default cost Zen5 29.5% against its own optimum and turned a gate PASS into a FAIL. This knob wants a
# pin, and the shipped default (minimax `np=1`) only has to be non-catastrophic until one exists.

using PureBLAS, LinearAlgebra, Printf, Statistics, TOML
import PureBLAS: _ger_panel!, _vwidth, _L3_BYTES
include(joinpath(@__DIR__, "measure.jl"))        # Measure.ab — the sanctioned Chairmarks harness
include(joinpath(@__DIR__, "tune_harness.jl"))   # stabilise! / with_anchor / contention / decide
BLAS.set_num_threads(1)

const DRYRUN = "dryrun" in ARGS
const EMIT = let i = findfirst(a -> startswith(a, "emit="), ARGS)
    isnothing(i) ? nothing : ARGS[i][6:end]
end

# ── ger DRAM path with an EXPLICIT stream count NP ──────────────────────────────────────────────────
@noinline function _ger_np!(A::Matrix{T}, x, y, ::Val{NP}) where {T, NP}
    m, n = size(A)
    GC.@preserve A x y begin
        Ap = pointer(A); xp = pointer(x); yp = pointer(y); lda = stride(A, 2); jc = 0
        while jc + NP <= n
            _ger_panel!(Ap, lda, xp, yp, jc, m, one(T), 0, Val(NP), Val(4)); jc += NP
        end
        while jc < n
            _ger_panel!(Ap, lda, xp, yp, jc, m, one(T), 0, Val(1), Val(4)); jc += 1
        end
    end
    return A
end
runnp(c, NP::Int) = NP == 1 ? _ger_np!(c..., Val(1)) : NP == 2 ? _ger_np!(c..., Val(2)) :
    NP == 4 ? _ger_np!(c..., Val(4)) : _ger_np!(c..., Val(8))

"""Winner for `ger_panel_np`, or `nothing` if the candidates are indistinguishable here."""
function calibrate_ger_np(::Type{T} = Float64) where {T}
    W = _vwidth(T)
    n0 = ceil(Int, sqrt(_L3_BYTES / sizeof(T)))
    n = max(2048, n0 + (W - n0 % W) % W)          # DRAM regime (A > L3) — the one the panel serves
    setup() = (randn(T, n, n), randn(T, n), randn(T, n))
    # Arm 1 is the INCUMBENT (the shipped default); `decide` only displaces it on a resolvable win.
    inc = PureBLAS._ger_np()
    cands = filter(!=(inc), [1, 2, 4, 8])
    arms = vcat([inc => (c -> runnp(c, inc))], [np => (c -> runnp(c, np)) for np in cands])
    @printf("  ger_panel_np @ n=%d (A = %.0f MB, DRAM): incumbent %d, candidates %s\n",
            n, n^2 * sizeof(T) / 2^20, inc, cands)
    res, ok, drift = with_anchor(() -> Measure.ab(arms; rounds = 8, setup = setup))
    for r in res
        @printf("    np=%-2d  %.4f  [%.4f, %.4f]\n", r.name, r.ratio, r.lo, r.hi)
    end
    if !ok
        @printf("    ⇒ REFUSED: machine state moved %.1f%% across the measurement — not adjudicable\n",
                100 * drift)
        return nothing
    end
    name, verdict = decide(res; delta = 0.02)
    if verdict === :tie
        println("    ⇒ tie — candidates within noise; leaving the in-code default in place")
        return nothing
    end
    @printf("    ⇒ WINNER np=%d\n", name)
    return name
end

# ── driver ──────────────────────────────────────────────────────────────────────────────────────────
la, busy = contention()
if busy
    @warn "machine is busy (1-min load $la) — a knob decided against someone else's workload is decided " *
          "wrong. Re-run on an idle machine."
    exit(3)
end
println("stabilising (burning the boost window so the ramp does not bias the first candidate)…")
anchor, khz, steady = stabilise!()
@printf("  anchor=%.3fus  clock=%.0fMHz  steady=%s\n", anchor * 1e6, khz / 1000, steady)
steady || @warn "did not reach steady state; results may be biased by a moving clock"

prefs = Pair{String, Any}[]
w = calibrate_ger_np()
isnothing(w) || push!(prefs, "ger_panel_np" => w)
# (further knobs append here — same shape: measure, decide, push only on a resolvable win)

if !isnothing(EMIT)
    # RESULT ONLY. tune!() aggregates several of these and pins what they agree on.
    # plain `key=value` — src/tune.jl parses this without a TOML dependency
open(io -> foreach(p -> println(io, p.first, "=", p.second), prefs), EMIT, "w")
    println("\n[emit] wrote $(length(prefs)) result(s) → $EMIT")
elseif DRYRUN
    println("\n[dryrun] would write under [PureBLAS]:")
    foreach(p -> println("  $(p.first) = $(p.second)"), prefs)
else
    LP = joinpath(dirname(Base.active_project()), "LocalPreferences.toml")
    d = isfile(LP) ? TOML.parsefile(LP) : Dict{String, Any}()
    sect = get!(d, "PureBLAS", Dict{String, Any}())
    for p in prefs
        sect[p.first] = p.second
    end
    sect["tuned_for"] = PureBLAS._tuning_fingerprint()
    open(io -> TOML.print(io, d), LP, "w")
    println("\nwrote $(length(prefs)) preference(s) + tuned_for → $LP")
    println("(reload Julia to pick them up — they are read at runtime, so no recompile)")
end
