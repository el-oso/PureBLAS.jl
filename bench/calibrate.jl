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

# ── potrf uplo='U' tiny-n direct cutoff ─────────────────────────────────────────────────────────────
# Measure tier because the optimum sits inside a NOISY BAND and no reasoning picks a point out of one.
# Its own history is the warning: the retired duel read "20 / 16 / 18 / 12 — four different cutoffs from
# one binary", and ComplexF64 gave 16/18/20 across ten fresh processes. `decide` is what makes running
# it safe anyway — a candidate must clear BOTH a CI excluding 1.0 and a 2% margin, so an unresolvable
# band returns :tie and the shipped 12 stands. Reporting "unresolvable on this host" is the useful
# output; silently pinning a draw from the band is the failure mode.
# Shapes are the SMALL uplo='U' factorizations the cutoff actually routes: the branch is n <= cutoff.
function calibrate_potrf_udirect(::Type{T} = Float64) where {T}
    inc = PureBLAS._potrf_udirect(T)
    cands = filter(!=(inc), [8, 12, 16, 20, 24])
    ns = (8, 12, 16, 20, 24, 32)
    setup() = [(A = randn(T, n, n); (Matrix(A * A' + n * I), Matrix{T}(undef, n, n))) for n in ns]
    run(v) = (cs -> begin
        PureBLAS._FKR_potrf_upper_direct_max[] = v
        for c in cs
            copyto!(c[2], c[1]); PureBLAS.potrf!(c[2]; uplo = 'U')
        end
        c1 = cs[1][2]; c1[1]
    end)
    arms = vcat([inc => run(inc)], [v => run(v) for v in cands])
    @printf("  potrf_upper_direct_max: incumbent %d, candidates %s (n = %s)\n", inc, cands, ns)
    res, ok, drift = with_anchor(() -> Measure.ab(arms; rounds = 8, setup = setup))
    PureBLAS._FKR_potrf_upper_direct_max[] = -1
    for r in res
        @printf("    %-6s %.4f [%.4f, %.4f]\n", r.name, r.ratio, r.lo, r.hi)
    end
    name, verdict = decide(res; delta = 0.02)
# Contract: a calibrator returns Pair{String,Any}[] — EMPTY on a tie, so tune!() pins nothing and
# the in-code default stands. That is the whole safety property for a knob like this one, whose own
# history is four different answers from one binary.
if verdict === :tie
    println("    ⇒ tie — candidates within noise; leaving the in-code default in place")
    return Pair{String, Any}[]
end
@printf("    ⇒ WINNER cutoff=%s\n", name)
return Pair{String, Any}["potrf_upper_direct_max" => parse(Int, string(name))]
end

# ── sytrf complex blocking multiplier ───────────────────────────────────────────────────────────────
# The Measure-tier quantity here is the MULTIPLIER on `_sytrf_nb_shape(n)`, not nb itself. The in-code
# fleet table says 3 wins on wintermute (+4.8%) and galen (+3.0%) and ties on neuromancer, while the
# retired duel resolved 2 in 16 of 18 samples — i.e. the duel and the gate DISAGREED. That is exactly
# the disagreement a calibrator should re-adjudicate on the host rather than inherit.
function calibrate_sytrf_cmult(::Type{T} = ComplexF64) where {T}
    inc = PureBLAS._SYTRF_CMULT
    cands = filter(!=(inc), [1, 2, 3, 4])
    n = 1024
    # sytrf! factors IN PLACE and takes an ipiv vector; both the working copy and ipiv are allocated in
    # setup (excluded from timing) so the measured region is the factorization alone. The matrix is
    # COMPLEX SYMMETRIC (A + transpose(A)), not Hermitian — sytrf is the symmetric factorization, and
    # `A + A'` would build a Hermitian matrix and exercise a different numerical path.
    setup() = (A = randn(T, n, n); H = A + transpose(A);
               (Matrix(H), Matrix{T}(undef, n, n), Vector{Int}(undef, n)))
    run(v) = (c -> begin
        PureBLAS._FKR_sytrf_cmult[] = v
        copyto!(c[2], c[1]); PureBLAS.sytrf!(c[2], c[3]; uplo = 'L')
        c[2][1]
    end)
    arms = vcat([inc => run(inc)], [v => run(v) for v in cands])
    @printf("  sytrf_cmult: incumbent %d, candidates %s (n = %d, %s)\n", inc, cands, n, T)
    res, ok, drift = with_anchor(() -> Measure.ab(arms; rounds = 8, setup = setup))
    PureBLAS._FKR_sytrf_cmult[] = -1
    for r in res
        @printf("    %-6s %.4f [%.4f, %.4f]\n", r.name, r.ratio, r.lo, r.hi)
    end
    name, verdict = decide(res; delta = 0.02)
# Contract: a calibrator returns Pair{String,Any}[] — EMPTY on a tie, so tune!() pins nothing and
# the in-code default stands. That is the whole safety property for a knob like this one, whose own
# history is four different answers from one binary.
if verdict === :tie
    println("    ⇒ tie — candidates within noise; leaving the in-code default in place")
    return Pair{String, Any}[]
end
@printf("    ⇒ WINNER cmult=%s\n", name)
return Pair{String, Any}["sytrf_cmult" => parse(Int, string(name))]
end

# ── gemv-T prefetch distance ────────────────────────────────────────────────────────────────────────
# Measure tier for a stated reason: prefetch distance depends on L2 hit latency and the hardware
# streamer, and PureBLAS detects neither. The candidate SET is derived (`_GEMVT_PF_CANDIDATES`, a
# multiple of the detected `_CACHELINE`), so the bounds adapt even though the choice cannot.
# Regime is the one the knob serves: the file's own measurement localises gemv-T's deficit to the
# L2->L1 stream supply, so A is sized to sit in L2, freshly written per sample, rep-looped like the gate.
function calibrate_gemvt_pf(::Type{T} = Float64) where {T}
    inc = PureBLAS._gemvt_pf()
    cands = filter(!=(inc), collect(PureBLAS._GEMVT_PF_CANDIDATES))
    n = max(256, isqrt(PureBLAS._L2_BYTES ÷ sizeof(T)))          # A ~ L2: the stream-supply regime
    reps = _l2rep(n)
    setup() = (randn(T, n, n), randn(T, n), zeros(T, n))
    run(v) = (c -> begin
        PureBLAS._GEMVT_PF_REF[] = v
        for _ in 1:reps; PureBLAS.gemv!(c[3], c[1], c[2]; trans = 'T'); end
        c[3][1]
    end)
    arms = vcat([inc => run(inc)], [v => run(v) for v in cands])
    @printf("  gemvt_pf: incumbent %d, candidates %s (n=%d, A=%.1f MiB ~ L2)\n",
            inc, cands, n, n^2 * sizeof(T) / 2^20)
    res, ok, drift = with_anchor(() -> Measure.ab(arms; rounds = 8, setup = setup))
    PureBLAS._GEMVT_PF_REF[] = inc
    for r in res
        @printf("    %-6s %.4f [%.4f, %.4f]\n", r.name, r.ratio, r.lo, r.hi)
    end
    name, verdict = decide(res; delta = 0.02)
    if verdict === :tie
        println("    ⇒ tie — candidates within noise; leaving the in-code default in place")
        return Pair{String, Any}[]
    end
    @printf("    ⇒ WINNER pf=%s\n", name)
    return Pair{String, Any}["gemvt_pf" => parse(Int, string(name))]
end

# ── trmv fused8 lower-n bound ───────────────────────────────────────────────────────────────────────
# The current default runs fused8 at EVERY n (the raw Ref reads -1 => no lower bound). That was decided
# on 2026-08-08 by forcing fused8 through the real entry path on all three boxes. This calibrator
# re-adjudicates it on the host: a positive value sends small n back to the unblocked `_trmv_simd!`.
# Candidates bracket the sizes where a lower bound could matter at all.
function calibrate_trmv_fused_min(::Type{T} = Float64) where {T}
    inc = PureBLAS._trmv_fused_min_raw()
    cands = filter(!=(inc), [-1, 64, 128, 256])
    ns = (64, 128, 256, 512)
    setup() = [(A = randn(T, n, n) ./ (2n); for i in 1:n; A[i, i] = 1 + abs(A[i, i]); end;
                (A, randn(T, n), Vector{T}(undef, n))) for n in ns]
    run(v) = (cs -> begin
        PureBLAS._TRMV_FUSED_MIN_REF[] = v
        for c in cs
            copyto!(c[3], c[2]); PureBLAS.trmv!(c[1], c[3]; uplo = 'U')
        end
        cs[1][3][1]
    end)
    arms = vcat([inc => run(inc)], [v => run(v) for v in cands])
    @printf("  trmv_fused_min: incumbent %d (-1 = fused8 everywhere), candidates %s, n = %s\n",
            inc, cands, ns)
    res, ok, drift = with_anchor(() -> Measure.ab(arms; rounds = 8, setup = setup))
    PureBLAS._TRMV_FUSED_MIN_REF[] = inc
    for r in res
        @printf("    %-6s %.4f [%.4f, %.4f]\n", r.name, r.ratio, r.lo, r.hi)
    end
    name, verdict = decide(res; delta = 0.02)
    if verdict === :tie
        println("    ⇒ tie — candidates within noise; leaving the in-code default in place")
        return Pair{String, Any}[]
    end
    @printf("    ⇒ WINNER trmv_fused_min=%s\n", name)
    return Pair{String, Any}["trmv_fused_min" => parse(Int, string(name))]
end

# ── gbtrf banded-LU panel multiplier ────────────────────────────────────────────────────────────────
# The tunable is the MULTIPLIER on the kl shape `8 * (1 + kl ÷ 128)`, never `nb` itself: pinning nb
# takes a branch that ignores kl entirely and would use one panel width at every band width. Same
# reasoning as `sytrf_cmult`, and the reason this knob had no calibrator until now.
# Swept across BAND WIDTHS, not matrix sizes — kl is what the shape reacts to. kl=16 is included
# deliberately: it is the cell where blocking wins on Zen4 and LOSES on Zen3 (bench/plots.jl's gbtrf row
# went PASS 1.43/1.18 -> FAIL 1.32/0.91 when a Zen4-only nb was shipped), so a multiplier that helps on
# one box must be checked there.
function calibrate_gbtrf_cmult(::Type{T} = Float64) where {T}
    inc = PureBLAS._gbtrf_cmult()
    cands = filter(!=(inc), [1, 2, 3, 4])
    n = 4096
    kls = (16, 64, 128, 256)
    # gbtrf!(kl, ku, m, AB) — band storage, factored IN PLACE. The pristine band and a working copy are
    # both built in setup (excluded from timing) so the measured region is the factorization alone.
    # Diagonally dominant so pivoting stays trivial and every arm does the same amount of work.
    setup() = [(ab = randn(T, 3kl + 1, n); ab[2kl + 1, :] .+= 4kl;
                (ab, copy(ab), kl)) for kl in kls]
    run(v) = (cs -> begin
        PureBLAS._FKR_gbtrf_cmult[] = v
        for c in cs
            copyto!(c[2], c[1]); PureBLAS.gbtrf!(c[3], c[3], n, c[2])
        end
        cs[1][2][1]
    end)
    arms = vcat([inc => run(inc)], [v => run(v) for v in cands])
    @printf("  gbtrf_cmult: incumbent %d, candidates %s (n=%d, kl = %s)\n", inc, cands, n, kls)
    res, ok, drift = with_anchor(() -> Measure.ab(arms; rounds = 8, setup = setup))
    PureBLAS._FKR_gbtrf_cmult[] = -1
    for r in res
        @printf("    cmult=%-3s %.4f [%.4f, %.4f]\n", r.name, r.ratio, r.lo, r.hi)
    end
    name, verdict = decide(res; delta = 0.02)
    if verdict === :tie
        println("    ⇒ tie — candidates within noise; leaving the in-code default in place")
        return Pair{String, Any}[]
    end
    @printf("    ⇒ WINNER cmult=%s\n", name)
    return Pair{String, Any}["gbtrf_cmult" => parse(Int, string(name))]
end

const KNOBS = (
    (name = "ger_panel_np", fn = calibrate_ger_np),
    (name = "gemvt_percol_window", fn = calibrate_gemvt_window),
    (name = "potrf_upper_direct_max", fn = calibrate_potrf_udirect),
    (name = "sytrf_cmult", fn = calibrate_sytrf_cmult),
    (name = "gemvt_pf", fn = calibrate_gemvt_pf),
    (name = "trmv_fused_min", fn = calibrate_trmv_fused_min),
    (name = "gbtrf_cmult", fn = calibrate_gbtrf_cmult),
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
