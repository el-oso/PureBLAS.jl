# Per-machine AUTOTUNE. Measures the knobs that are NOT derivable from detected hardware and writes the
# winners as Preferences pins. Run it once per machine; `PureBLAS.tune!()` drives it.
#
#   julia --project=bench bench/calibrate.jl                 # measure + WRITE the preferences
#   julia --project=bench bench/calibrate.jl dryrun          # measure + PRINT only
#   julia --project=bench bench/calibrate.jl emit=/tmp/x.toml  # measure, write RESULT ONLY to that file
#   julia --project=bench bench/calibrate.jl only=ger_panel_np # one knob (debugging the harness)
#
# `emit=` exists so tune!() can run this in several INDEPENDENT PROCESSES and pin only what they agree
# on. That is not belt-and-braces: the retired OncePerProcess tier failed precisely because a per-process
# draw (page placement, allocator state) can change the winner, and one knob picked a 17%-slower arm in
# 7 of 9 processes. More reps inside one process cannot see that; separate processes can.
#
# ── THE KNOBS TABLE ────────────────────────────────────────────────────────────────────────────────
# Each entry is (name, fn) where `fn() -> Vector{Pair{String,Any}}` — EMPTY meaning "nothing to pin,
# the in-code default is adequate here". A knob owns its own arms because the arms differ in KIND:
# `ger_panel_np` picks the best scalar at one shape, while the gemv-T window is decided PER SIZE and its
# two edges are then derived from which sizes won. Forcing both into one "sweep a scalar" shape is what
# would make the second one measure the wrong thing — an op-level number cannot settle a per-size
# question, which this project learned the hard way (see cpuinfo.jl (c5)).
# What the DRIVER owns, uniformly, is the guarding: contention refusal, boost-window stabilisation, and
# the anchor check around every measurement.

using PureBLAS, LinearAlgebra, Printf, Statistics, TOML
import PureBLAS: _ger_panel!, _vwidth, _L1_BYTES, _L2_BYTES, _L3_BYTES
include(joinpath(@__DIR__, "measure.jl"))        # Measure.ab — the sanctioned Chairmarks harness
include(joinpath(@__DIR__, "tune_harness.jl"))   # stabilise! / with_anchor / contention / decide
BLAS.set_num_threads(1)

const DRYRUN = "dryrun" in ARGS
const EMIT = let i = findfirst(a -> startswith(a, "emit="), ARGS)
    isnothing(i) ? nothing : ARGS[i][6:end]
end
const ONLY = let i = findfirst(a -> startswith(a, "only="), ARGS)
    isnothing(i) ? nothing : ARGS[i][6:end]
end
# Rep count, verbatim from bench/plots.jl `_L2REP`. THE REGIME MUST MATCH THE GATE: at reps=1 a probe
# measures a COLD matrix while the gate runs many reps on a warm one, and that difference alone has
# INVERTED a routing verdict here (gemvT n=2048: 1.087 percol-wins -> 0.951 blocked-wins).
_l2rep(s) = clamp(400_000_000 ÷ (s * s), 30, 20000)

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

"""Winner for `ger_panel_np`, or no pin if the candidates are indistinguishable here."""
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
        return Pair{String, Any}[]
    end
    name, verdict = decide(res; delta = 0.02)
    if verdict === :tie
        println("    ⇒ tie — candidates within noise; leaving the in-code default in place")
        return Pair{String, Any}[]
    end
    @printf("    ⇒ WINNER np=%d\n", name)
    return Pair{String, Any}["ger_panel_np" => name]
end

# ── gemv-T per-column WINDOW — measured PER SIZE, edges DERIVED from the winners ────────────────────
# The knob is a window `A > AMIN && x <= XMAX`, and its optimum is per-(box, SIZE): on this fleet every
# pair of boxes disagrees at some size and no detected const partitions them. So the measurement is a
# paired per-size A/B of the two ROUTES, and the two edges are then fitted to the observed winner set.
# Sizes where the A/B is a TIE become DON'T-CARE in the fit rather than being forced to a side — n=2048
# is bimodal on Zen4 (A = 2x L3) and forcing it would fit noise.
const _GEMVT_SIZES = (512, 1024, 2048, 4096)

@noinline _route!(y, A, x, mode) = (PureBLAS._GEMVT_PERSCAN_REF[] = mode;
                                    PureBLAS.gemv!(y, A, x; alpha = 1.0, beta = 0.0, trans = 'T'))

function calibrate_gemvt_window()
    @static if !isnothing(PureBLAS._GEMVT_PERSCAN_PREF)
        println("  gemvt window: perscan is PINNED in this build — nothing to calibrate")
        return Pair{String, Any}[]
    end
    saved = PureBLAS._GEMVT_PERSCAN_REF[]
    obs = Dict{Int, Union{Bool, Nothing}}()       # n => percol wins? (nothing = tie, don't care)
    try
        for n in _GEMVT_SIZES
            setup() = (zeros(Float64, n), randn(Float64, n, n), randn(Float64, n))
            arms = ["blocked" => (c -> _route!(c[1], c[2], c[3], 0)),
                    "percol"  => (c -> _route!(c[1], c[2], c[3], 2))]
            res, ok, drift = with_anchor(() -> Measure.ab(arms; rounds = 8, reps = _l2rep(n),
                                                          setup = setup))
            r = res[2]
            if !ok
                @printf("    n=%-5d REFUSED (machine moved %.1f%%) — treated as don't-care\n", n, 100 * drift)
                obs[n] = nothing
            elseif r.lo > 1.0
                @printf("    n=%-5d percol  %.4f [%.4f, %.4f]  WINS\n", n, r.ratio, r.lo, r.hi)
                obs[n] = true
            elseif r.hi < 1.0
                @printf("    n=%-5d blocked %.4f [%.4f, %.4f]  wins\n", n, r.ratio, r.lo, r.hi)
                obs[n] = false
            else
                @printf("    n=%-5d tie     %.4f [%.4f, %.4f]  — don't-care in the fit\n", n, r.ratio, r.lo, r.hi)
                obs[n] = nothing
            end
        end
    finally
        PureBLAS._GEMVT_PERSCAN_REF[] = saved      # never leave the process on an instrument arm
    end

    # CANDIDATE SET IS DERIVED, not a range: the edges only ever sit at cache-geometry boundaries.
    amins = unique([_L2_BYTES, 2 * _L2_BYTES, 4 * _L2_BYTES])
    xmaxs = unique([_L1_BYTES ÷ 2, _L1_BYTES ÷ 4, _L1_BYTES ÷ 8, 8192, 4096])
    fits(a, x) = all(n -> isnothing(obs[n]) ||
                          obs[n] == ((n * n * 8 > a) && (n * 8 <= x)), _GEMVT_SIZES)
    hit = [(a, x) for a in amins, x in xmaxs if fits(a, x)]
    if isempty(hit)
        println("    ⇒ NO window reproduces the measured winners — the knob's SHAPE is wrong for this " *
                "box, not just its value. Reporting, not pinning.")
        return Pair{String, Any}[]
    end
    # NARROWEST fitting window, not the widest. Per-column is the NON-DEFAULT arm, so it is enabled only
    # where it demonstrably wins; a size with no verdict must not be swept in by widening.
    # ⚠ THIS WAS "widest" AND IT PICKED A KNOWN-BAD ARM. On galen the n=512 measurement was REFUSED
    # (anchor moved 9.5%) and so became don't-care; the widest fit then chose amin=L2, which routes
    # per-column at n=512 — where three earlier processes measured it LOSING ~4%. A refusal is MISSING
    # DATA, not a tie, and widening into missing data is how a tuner ships a regression it never measured.
    a, x = hit[argmin([count(n -> (n * n * 8 > aa) && (n * 8 <= xx), _GEMVT_SIZES) for (aa, xx) in hit])]
    if all(n -> isnothing(obs[n]) || obs[n] == false, _GEMVT_SIZES)
        println("    ⇒ per-column never wins here — mode 0 (the shipped default for non-Zen4) is right")
        return Pair{String, Any}[]
    end
    dflt_a, dflt_x = PureBLAS._GEMVT_PERCOL_AMIN, PureBLAS._GEMVT_PERCOL_XMAX
    if a == dflt_a && x == dflt_x && saved == 1
        println("    ⇒ the shipped window already IS the optimum here — nothing to pin")
        return Pair{String, Any}[]
    end
    @printf("    ⇒ WINDOW amin=%d (%.0f KiB), xmax=%d (%.0f KiB), mode=1\n", a, a / 1024, x, x / 1024)
    return Pair{String, Any}["gemvt_perscan" => 1, "gemvt_percol_amin" => a, "gemvt_percol_xmax" => x]
end

const KNOBS = (
    (name = "ger_panel_np", fn = calibrate_ger_np),
    (name = "gemvt_percol_window", fn = calibrate_gemvt_window),
)

# ── driver ──────────────────────────────────────────────────────────────────────────────────────────
# FREQUENCY LOCK FIRST — before contention, before anything. A value pinned from a boosting box is
# worse than no pin: it is wrong in a way nothing downstream can detect, because the clock drifts
# BETWEEN the arms of an A/B and paired measurement cannot cancel that.
locked, why = freq_locked()
if !locked
    @error "REFUSING TO TUNE — $why. The frequency lock is the ONLY state a tuning measurement is " *
           "valid in; a pin taken on a floating clock is a wrong answer with a confident face."
    exit(4)
end
println("freq: $why")
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
# ACHIEVED vs REQUESTED — the settings check above is NOT sufficient, which is the whole lesson here.
# neuromancer 2026-08-21 reported boost=0, scaling_min==scaling_max==2000 MHz, governor `performance`
# — every knob in the right position — and ran the calibration at an ACHIEVED 4843 MHz. The pstate
# driver was not honouring the cap (it needs amd_pstate=passive), so the requested state and the real
# clock disagreed by 2.4x. This is exactly why bench/fleet_freqlock.sh VERIFIES under load instead of
# trusting what it wrote, and the tuner must do the same.
let cap = tryparse(Int, strip(read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq", String)))
    if !isnothing(cap) && khz > 1.05 * cap
        @error "REFUSING TO TUNE — achieved $(round(Int, khz/1000)) MHz against a requested cap of " *
               "$(cap ÷ 1000) MHz. The cpufreq SETTINGS look locked but the clock is floating; the " *
               "driver is ignoring them. Run `sudo bench/fleet_freqlock.sh lock` and check its verify " *
               "step passes. A pin taken here would be measured on a clock that drifts BETWEEN the " *
               "arms of an A/B — the one error paired measurement cannot cancel."
        exit(5)
    end
end

prefs = Pair{String, Any}[]
for k in KNOBS
    isnothing(ONLY) || k.name == ONLY || continue
    println("\n── $(k.name) ──")
    append!(prefs, k.fn())
end

if !isnothing(EMIT)
    # RESULT ONLY. tune!() aggregates several of these and pins what they agree on.
    # plain `key=value` — src/tune.jl parses this without a TOML dependency
    open(io -> foreach(p -> println(io, p.first, "=", p.second), prefs), EMIT, "w")
    println("\n[emit] wrote $(length(prefs)) result(s) → $EMIT")
elseif DRYRUN
    println("\n[dryrun] would write under [PureBLAS]:")
    isempty(prefs) && println("  (nothing — every knob's in-code default is adequate on this machine)")
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
