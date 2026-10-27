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
# ── UNLOCKED MODE ─────────────────────────────────────────────────────────────────────────────────
# `PureBLAS.tune!()` is a USER-FACING entry point — `_GER_NP`'s comment literally tells users to run
# it — but this script refused unless the machine was frequency-locked, and locking needs root. A
# normal user has no `sudo` on their laptop, so the tuner was unreachable for exactly the people it
# exists for. That is a design defect, not a safety property.
#
# What the lock actually protects is ABSOLUTE times. This script decides on a RATIO between arms run
# back-to-back within a round, and since 2026-08-28 `Measure.ab` reduces those rounds in CYCLES
# (time × the clock sampled around each window), which is invariant to boost: a faster clock shortens
# both arms' elapsed time and leaves the cycle counts where they were. So the decision does not need
# the lock; it needs the clock to be READABLE, which it is without privileges.
#
# `unlocked` therefore does not disable the safety checks — it TIGHTENS them, trading the lock for
# more evidence:
#   * `ANCHOR_TOL` 0.08 → 0.03, so the per-knob machine-state refusal does the work the lock did.
#   * `decide` margin 0.02 → 0.05, so a marginal win under a moving clock is not pinned.
#   * more rounds, so the bootstrap CI has to survive more of the drift.
#   * the emitted TOML is stamped `measured_unlocked = true`, so a pin's provenance is never ambiguous.
# The achieved-vs-requested check below is NOT relaxed: a clock that reports one thing and runs at
# another (Zen5 2026-08-21, 4843 MHz against a 2000 MHz cap) is a LYING clock, categorically
# different from an honestly-boosting one, and no amount of cycle conversion can rescue it.
const UNLOCKED = "unlocked" in ARGS
const TOL = UNLOCKED ? 0.03 : ANCHOR_TOL
const DELTA = UNLOCKED ? 0.05 : 0.02
const ROUNDS = UNLOCKED ? 12 : 8
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
    res, ok, drift = with_anchor(() -> Measure.ab(arms; rounds = ROUNDS, setup = setup); tol = TOL)
    for r in res
        @printf("    np=%-2d  %.4f  [%.4f, %.4f]\n", r.name, r.ratio, r.lo, r.hi)
    end
    if !ok
        @printf("    ⇒ REFUSED: machine state moved %.1f%% across the measurement — not adjudicable\n",
                100 * drift)
        return Pair{String, Any}[]
    end
    name, verdict = decide(res; delta = DELTA)
    if verdict !== :win
        println(noswitch_msg(verdict))
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
            res, ok, drift = with_anchor(() -> Measure.ab(arms; rounds = ROUNDS, reps = _l2rep(n),
                                                          setup = setup); tol = TOL)
            r = res[2]
            if !ok
                @printf("    n=%-5d REFUSED (machine moved %.1f%%) — treated as don't-care\n", n, 100 * drift)
                obs[n] = nothing
            # BOTH BARS, the same two `decide` applies: a CI excluding 1.0 says the difference is
            # resolvable, and the `DELTA` margin says it is worth acting on. The margin stops a
            # window being fitted to an effect too small to adjudicate — req#1 settles the gate at two
            # significant digits because per-cell drift on this fleet runs 1-6%, so a ratio inside the
            # margin rounds to 1.00 and is a tie by the project's own criterion however tight its
            # interval happens to come out in one process.
            elseif r.lo > 1.0 && r.ratio > 1 + DELTA
                @printf("    n=%-5d percol  %.4f [%.4f, %.4f]  WINS\n", n, r.ratio, r.lo, r.hi)
                obs[n] = true
            elseif r.hi < 1.0 && r.ratio < 1 - DELTA
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
    # ⚠ THIS WAS "widest" AND IT PICKED A KNOWN-BAD ARM. On Zen3 the n=512 measurement was REFUSED
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
    res, ok, drift = with_anchor(() -> Measure.ab(arms; rounds = ROUNDS, setup = setup); tol = TOL)
    PureBLAS._FKR_potrf_upper_direct_max[] = -1
    for r in res
        @printf("    %-6s %.4f [%.4f, %.4f]\n", r.name, r.ratio, r.lo, r.hi)
    end
    name, verdict = decide(res; delta = DELTA)
# Contract: a calibrator returns Pair{String,Any}[] — EMPTY on a tie, so tune!() pins nothing and
# the in-code default stands. That is the whole safety property for a knob like this one, whose own
# history is four different answers from one binary.
if verdict !== :win
    println(noswitch_msg(verdict))
    return Pair{String, Any}[]
end
@printf("    ⇒ WINNER cutoff=%s\n", name)
return Pair{String, Any}["potrf_upper_direct_max" => parse(Int, string(name))]
end

# ── sytrf complex blocking multiplier ───────────────────────────────────────────────────────────────
# The Measure-tier quantity here is the MULTIPLIER on `_sytrf_nb_shape(n)`, not nb itself. The in-code
# fleet table says 3 wins on Zen4 (+4.8%) and Zen3 (+3.0%) and ties on Zen5, while the
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
    res, ok, drift = with_anchor(() -> Measure.ab(arms; rounds = ROUNDS, setup = setup); tol = TOL)
    PureBLAS._FKR_sytrf_cmult[] = -1
    for r in res
        @printf("    %-6s %.4f [%.4f, %.4f]\n", r.name, r.ratio, r.lo, r.hi)
    end
    name, verdict = decide(res; delta = DELTA)
# Contract: a calibrator returns Pair{String,Any}[] — EMPTY on a tie, so tune!() pins nothing and
# the in-code default stands. That is the whole safety property for a knob like this one, whose own
# history is four different answers from one binary.
if verdict !== :win
    println(noswitch_msg(verdict))
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
    # Measure where `pf` APPLIES: the prefetch only reaches the non-deep kernel, so a size where the deep
    # shape runs times two identical arms. It used to be A ~ L2 (n = √(L2/8) ≈ 362), where deep IS the
    # incumbent — Zen5's pf=128 pin came from there. Smallest po2 n with A > L2: po2 ⇒ lda·8 is a
    # multiple of the L1 way, so deep is ineligible by `_gemvt_deep`'s own rule.
    n = nextpow(2, isqrt(PureBLAS._L2_BYTES ÷ sizeof(T)) + 1)
    PureBLAS._gemvt_deep(T, n, n, n) && (println("  gemvt_pf: deep shape eligible at n=$n — no regime to calibrate"); return Pair{String, Any}[])
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
    res, ok, drift = with_anchor(() -> Measure.ab(arms; rounds = ROUNDS, setup = setup); tol = TOL)
    PureBLAS._GEMVT_PF_REF[] = inc
    for r in res
        @printf("    %-6s %.4f [%.4f, %.4f]\n", r.name, r.ratio, r.lo, r.hi)
    end
    name, verdict = decide(res; delta = DELTA)
    if verdict !== :win
        println(noswitch_msg(verdict))
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
    res, ok, drift = with_anchor(() -> Measure.ab(arms; rounds = ROUNDS, setup = setup); tol = TOL)
    PureBLAS._TRMV_FUSED_MIN_REF[] = inc
    for r in res
        @printf("    %-6s %.4f [%.4f, %.4f]\n", r.name, r.ratio, r.lo, r.hi)
    end
    name, verdict = decide(res; delta = DELTA)
    if verdict !== :win
        println(noswitch_msg(verdict))
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
    res, ok, drift = with_anchor(() -> Measure.ab(arms; rounds = ROUNDS, setup = setup); tol = TOL)
    PureBLAS._FKR_gbtrf_cmult[] = -1
    for r in res
        @printf("    cmult=%-3s %.4f [%.4f, %.4f]\n", r.name, r.ratio, r.lo, r.hi)
    end
    name, verdict = decide(res; delta = DELTA)
    if verdict !== :win
        println(noswitch_msg(verdict))
        return Pair{String, Any}[]
    end
    @printf("    ⇒ WINNER cmult=%s\n", name)
    return Pair{String, Any}["gbtrf_cmult" => parse(Int, string(name))]
end

# ── gemv_mr: row-block height in vectors (mr = MR·W rows) ───────────────────────────────────────────
# MEASURE TIER BY NECESSITY. Zen4 wants 4 while Zen3 and Zen5-mobile want 8, and after the 2026-09-09
# datapath correction NO detected const separates them: Zen4 and Zen5-mobile are BOTH double-pumped,
# with the same L2, L3 and register count. Only L1 (32 vs 48 KiB) and the family differ, and a rule fit
# to that is a two-point curve fit — the failure that produced the bug this knob is being rescued from.
# Fleet evidence behind the shipped default of 8 (gate-exact regime, freq-locked, GB/s):
#   Zen4  MR=4 -> 8: n=50 92.6->72.4 (-21.8%), n=100 78.3->60.7 (-22.5%), n=256 83.6->85.8 (+2.6%)
#   Zen5-mobile: MR=4 measured 0.86 at gemvN@256, i.e. wants 8;  Zen3: wants 8
# So 8 is right on two boxes of three and this calibrator is how Zen4 recovers its 4.
#
# SIZES: MR trades accumulator count against register pressure, and the fleet split shows up at
# CACHE-RESIDENT mid-n (n=100 is a red cell on Zen4), not at DRAM scale where bandwidth dominates.
# Measure there. Arms run through the REAL driver via `Val(MR)` — the `_ger_np!(…, Val(NP))` shape —
# so this is not a prototype that can diverge from the shipped kernel.
function calibrate_gemv_mr(::Type{T} = Float64) where {T}
    inc = PureBLAS._GEMV_MR
    cands = filter(!=(inc), [4, 8])
    isempty(cands) && return Pair{String, Any}[]
    n = 100                                        # the cell the fleet split is largest at
    setup() = (randn(T, n, n), randn(T, n), randn(T, n))
    run(c, MR) = MR == 4 ?
        PureBLAS._gemv_n_paneldrv!(n, n, one(T), c[1], c[2], c[3], zero(T), Val(true), Val(4)) :
        PureBLAS._gemv_n_paneldrv!(n, n, one(T), c[1], c[2], c[3], zero(T), Val(true), Val(8))
    arms = vcat([inc => (c -> run(c, inc))], [mr => (c -> run(c, mr)) for mr in cands])
    @printf("  gemv_mr @ n=%d (cache-resident, where the fleet split is largest): incumbent %d, candidates %s\n",
        n, inc, cands)
    res, ok, drift = with_anchor(() -> Measure.ab(arms; rounds = ROUNDS, setup = setup); tol = TOL)
    for r in res
        @printf("    MR=%-2d  %.4f  [%.4f, %.4f]\n", r.name, r.ratio, r.lo, r.hi)
    end
    if !ok
        @printf("    ⇒ REFUSED: machine state moved %.1f%% across the measurement — not adjudicable\n", 100 * drift)
        return Pair{String, Any}[]
    end
    name, verdict = decide(res; delta = DELTA)
    if verdict !== :win
        println(noswitch_msg(verdict))
        return Pair{String, Any}[]
    end
    @printf("    ⇒ WINNER MR=%d\n", name)
    return Pair{String, Any}["gemv_mr" => name]
end

# ── gemm_mt_work: the amortisation floor below which threading cannot pay ───────────────────────────
# NOT AN A/B DUEL, and the difference is the point. Every other calibrator here races candidate arms and
# keeps the winner. This knob is a DERIVATION over one measured host fact — the pool fork-join round
# trip — fed through the formula `src/blas3/gemm.jl` already ships:
#
#     floor_flops = (join_ns × clock_MHz ÷ 1000) × _MT_AMORTISE × _mt_flops_per_cycle(Float64)
#                    └─ the join in core cycles ─┘   └ policy ┘    └─ detected FMA rate ─┘
#
# Only the join is a host measurement; the amortisation ratio is policy and the FMA rate is detected. So
# racing floors would be racing a number we can compute from something directly measurable, and would
# need a gemm size ladder straddling each candidate to tell them apart at all.
#
# WHY THIS CALIBRATOR HAS TO EXIST. The shipped default is ONE recording — Zen4, 616 ns at 2796 MHz —
# and `gemm.jl` says plainly that Zen3 and Zen5 have not confirmed it. Worse, the probe that produced it
# (`bench/probes/threading_forkjoin_cost.jl`) is NOT tracked: `bench/probes/*.jl` is gitignored, so the
# single measurement the floor rests on cannot be reproduced from a checkout. Hence a self-contained
# pool below rather than an include.
#
# WHAT IT MEASURES: the mechanism PureBLAS actually implements — a parked pool woken by a generation
# word, workers spinning then waiting on an Event, the driver spinning then yielding on an atomic
# arrival count. Zero work per worker, so the figure is the fixed price a threaded driver pays per call.
# It deliberately does NOT drive `gemm!`: a real gemm cannot be made to thread at a size where compute
# is negligible without pinning this very knob first.
mutable struct _CalPool
    @atomic gen::Int
    @atomic done::Int
    const evs::Vector{Threads.Event}
end
function _calpool_worker(p::_CalPool, i::Int, spins::Int)
    seen = 0
    while true
        g = @atomic p.gen
        t = 0
        while g == seen && t < spins
            t += 1
            g = @atomic p.gen
        end
        g == seen && (wait(p.evs[i]); g = @atomic p.gen)
        g < 0 && return nothing              # shutdown
        seen = g
        @atomic p.done += 1                  # the job is empty: this IS the round trip
    end
end
function _calpool_join!(p::_CalPool, nw::Int, spins::Int)
    @atomic p.done = 0
    @atomic p.gen = (@atomic p.gen) + 1
    for i in 1:nw
        notify(p.evs[i])
    end
    t = 0
    while (@atomic p.done) < nw
        t += 1
        t > spins && (yield(); t = 0)
    end
    return nothing
end

function calibrate_gemm_mt_work(::Type{T} = Float64) where {T}
    nt = Threads.nthreads()
    if nt < 2
        println("  gemm_mt_work: DECLINED — a fork-join cost cannot be measured in a single-threaded " *
                "process (Threads.nthreads()=$(nt)). Re-run `julia -t N`; this knob is the only one here " *
                "that needs threads, which is why it had no calibrator.")
        return Pair{String, Any}[]
    end
    nw = min(nt - 1, 5)                      # workers BESIDES the driver, capped as the pool caps itself
    spins = 2048                             # PureBLAS `_MT_SPINS`
    p = _CalPool(0, 0, [Threads.Event(true) for _ in 1:nw])
    ws = [Threads.@spawn _calpool_worker(p, i, spins) for i in 1:nw]
    _calpool_join!(p, nw, spins)             # warm: first wake pays task start-up
    # estimator-ok: `median` IS the sanctioned estimator; Measure.tstat is for ARM ratios, and this is a
    # single absolute latency with no arm to divide by.
    join_s = median([median(@be _calpool_join!(p, nw, spins) seconds = 0.2).time for _ in 1:5])
    khz = Measure._khz()
    @atomic p.gen = -1                       # shut the workers down
    foreach(i -> notify(p.evs[i]), 1:nw)
    foreach(wait, ws)

    if khz <= 0
        println("  gemm_mt_work: DECLINED — no cpufreq reading, so the join cannot be priced in cycles.")
        return Pair{String, Any}[]
    end
    join_ns = join_s * 1e9
    mhz = khz / 1000
    cycles = round(Int, join_ns * mhz / 1000)
    fpc = PureBLAS._mt_flops_per_cycle(Float64)
    floor_flops = cycles * PureBLAS._MT_AMORTISE * fpc
    shipped = PureBLAS._GEMM_MT_WORK_SHIPPED
    @printf("  gemm_mt_work: join %.0fns over %d worker(s) at %.0fMHz ⇒ %d cycles ⇒ floor %d flops\n",
            join_ns, nw, mhz, cycles, floor_flops)
    @printf("    shipped default %d (Zen4, 616ns @ 2796MHz) — this host is %.2fx that\n",
            shipped, floor_flops / shipped)
    # BOUNDED AGAINST THE SHIPPED LADDER, and a value outside it is REPORTED rather than pinned. The
    # ladder spans the two measured protocol regimes (a spin wake ~570ns, a sleep wake ~4000ns); a join
    # outside that bracket means the measurement caught something other than the mechanism — a busy box,
    # a migrating driver — and pinning it would bake that in.
    lo, hi = extrema(PureBLAS._GEMM_MT_WORK_CANDIDATES)
    if !(lo <= floor_flops <= hi)
        @printf("    ⇒ NOT PINNED: %d is outside the candidate bracket [%d, %d]. That bracket spans the \
                spin-wake and sleep-wake regimes, so a join beyond it means this measured something \
                other than the pool. Re-run on an idle, frequency-locked box.\n", floor_flops, lo, hi)
        return Pair{String, Any}[]
    end
    @printf("    ⇒ WINNER gemm_mt_work=%d\n", floor_flops)
    return Pair{String, Any}["gemm_mt_work" => floor_flops]
end

# ── SME in-place-A column-panel cut — DECOMPOSED, NOT SEARCHED ──────────────────────────────────────
# The macrokernel re-reads the A block once per column panel, so reading A in place costs `njp*S`
# while packing it costs `P + njp*C`, and the crossover is `njp* = P/(S-C)`. Three costs settle the
# knob in five timings. Searching candidate cuts end to end does NOT fit this harness: the answer is
# the largest cut that loses no gate shape, a constrained maximum, and `decide` adjudicates one arm
# against another.
#
# `C` and `S` are MARGINALS, differenced between two `njp` values, so the kernel's fixed
# streaming-region entry cost — identical in both arms — cancels and only the per-panel cost survives.
#
# ⚠ THE IN-PLACE ARM'S COST IS DRAWN PER PROCESS AND HELD FOR THAT PROCESS'S LIFE. Fourteen fresh
# operand sets inside one process span 1.10x and agree on the verdict, while cold processes span 2.6x.
# So one process measures precisely but CONDITIONALLY, and two processes on identical source disagree
# in sign. This function returns ONE process's verdict and claims nothing beyond it; `tune!()` runs
# `repeats` independent processes and pins only a unanimous answer, which IS the cross-process guard.
# The verdict is rounded onto a coarse grid for the same reason: three continuous estimates never
# agree to the integer, and a knob that can never pin is worse than no calibrator at all.
const SME_NJP_GRID = (8, 16, 24, 32)
# TWO BOUNDS LIMIT THE CUT, AND THIS FUNCTION MEASURES NEITHER. The crossover below says where
# packing starts to pay; it does not say which cuts leave every gate shape whole. Square gemm loses
# as soon as the cut admits its own panel count — 1.078 at a cut of 32 where njp is 32, and 1.079 at
# a cut of 64 at n = 1024 where njp is 64 — and `_sme_blocks` gives those panel counts, so that bound
# follows from the blocking. It is the weaker of the two: it bars only a cut EQUAL to a losing
# shape's panel count, and leaves the cuts between the shipped value and that count untested. The
# binding bound is the shipped value itself, which the rank-k sweep establishes as the largest cut
# measured safe. The grid is capped by both, so this function can only ever LOWER the cut — the one
# direction its own evidence supports.
const SME_NJP_CEIL_N = (512, 1024)

"""Crossover `njp` for reading A in place, or no pin when this machine cannot be adjudicated."""
function calibrate_sme_inplace_njp(::Type{T} = Float64) where {T}
    if !(PureBLAS._SME_F64 && PureBLAS._SME_ENTRY[] != C_NULL)
        println("  sme_inplace_njp: DECLINED — no live SME build. The knob gates the coprocessor path " *
                "only, and its kernels are not callable in this process.")
        return Pair{String, Any}[]
    end
    MR = PureBLAS._SME_MR; NR = PureBLAS._SME_NR; L = PureBLAS._SME_L
    # MEASURE AT THE GEOMETRY THE GATE'S SQUARE SHAPES HAVE: `lda = n = 512` puts the columns in
    # `_L1_WAY_BYTES / gcd(lda*sizeof(T), _L1_WAY_BYTES)` distinct L1 sets, and the cost of one
    # in-place column panel is a step function of that count rather than of the stride's magnitude.
    # This knob is the TWO-OR-MORE-SET value; the one-set case is derived from it as `>> 1` off the
    # same detected constant, so a one-set stride must not be measured here.
    n = 512
    lda = n
    sets = PureBLAS._L1_WAY_BYTES ÷ gcd(lda * sizeof(T), PureBLAS._L1_WAY_BYTES)
    if sets < 2
        println("  sme_inplace_njp: DECLINED — lda=$(lda) spans $(sets) L1 set on this part, which is " *
                "the derived half of this knob, not the knob.")
        return Pair{String, Any}[]
    end
    MC, _, KC = PureBLAS._sme_blocks(n, n, n)
    mce = min(MC, n)
    kce = min(KC, n)
    kpad = cld(kce, L) * L
    mpad = cld(mce, MR) * MR
    nip = mpad ÷ MR
    j1, j2 = 4, 16
    inc = PureBLAS._SME_INPLACE_NJP
    blockceil = minimum(nn -> cld(PureBLAS._sme_blocks(nn, nn, nn)[2], NR), SME_NJP_CEIL_N) - 1
    ceil_njp = min(inc, blockceil)
    grid = filter(<=(ceil_njp), SME_NJP_GRID)
    if isempty(grid)
        println("  sme_inplace_njp: DECLINED — no grid value sits at or below the ceiling " *
                "njp=$(ceil_njp), so every candidate would raise the cut past where it is safe.")
        return Pair{String, Any}[]
    end
    @printf("  sme_inplace_njp @ n=%d: block %dx%d, %d L1 set(s), ceiling njp %d, grid %s\n",
            n, mce, kce, sets, ceil_njp, grid)
    A = randn(T, lda, kpad)
    Ap = zeros(T, mpad * kpad)
    Bp = randn(T, kpad * NR * j2)
    Cp = zeros(T, mpad * NR * j2)
    # The macrokernel runs whole `mpad x (njp*NR)` tiles, so it is timed into a PRIVATE PADDED C with
    # `ldc = mpad`, never a real operand, which it would write past.
    costs, ok, drift = with_anchor(; tol = TOL) do
        GC.@preserve A Ap Bp Cp begin
            pA = pointer(A); pa = pointer(Ap); pb = pointer(Bp); pc = pointer(Cp)
            mac(p, aip, aks, njp) = PureBLAS._sme_macro!(
                pc, mpad, p, pb, nip, njp, kpad, aip, aks, kpad * NR, NR,
                mce - (nip - 1) * MR, NR, true)
            pk() = PureBLAS._sme_pack_A!(pa, pA, lda, 0, 0, mce, kce, kpad, 1.0)
            pk()                                          # warm, and fills Ap for the packed arm
            mac(pa, kpad * MR, MR, j1)
            mac(pA, MR, lda, j1)
            # estimator-ok: `median` IS the sanctioned estimator; `Measure.tstat` reduces ARM RATIOS,
            # and these are five absolute latencies with no arm to divide by.
            tP = median(@be pk() seconds = 0.3).time
            cp1 = median(@be mac(pa, kpad * MR, MR, j1) seconds = 0.3).time
            cp2 = median(@be mac(pa, kpad * MR, MR, j2) seconds = 0.3).time
            ip1 = median(@be mac(pA, MR, lda, j1) seconds = 0.3).time
            ip2 = median(@be mac(pA, MR, lda, j2) seconds = 0.3).time
            (P = tP, cp1 = cp1, cp2 = cp2, ip1 = ip1, ip2 = ip2)
        end
    end
    if !ok
        @printf("    ⇒ REFUSED: machine state moved %.1f%% across the measurement — not adjudicable\n",
                100 * drift)
        return Pair{String, Any}[]
    end
    Cm = (costs.cp2 - costs.cp1) / (j2 - j1)
    Sm = (costs.ip2 - costs.ip1) / (j2 - j1)
    @printf("    pack %.2fus  packed panel %.3fus  in-place panel %.3fus  sigma %.3f\n",
            costs.P * 1e6, Cm * 1e6, Sm * 1e6, Cm > 0 ? Sm / Cm : NaN)
    if !(Sm > Cm)
        println("    ⇒ no pin: a panel costs no more in place than packed here, so no crossover " *
                "exists and this evidence cannot move the cut.")
        return Pair{String, Any}[]
    end
    njpstar = costs.P / (Sm - Cm)
    pin = grid[argmin(abs.(grid .- njpstar))]
    @printf("    njp* %.1f  ⇒ grid %d (incumbent %d)\n", njpstar, pin, inc)
    if pin == inc
        println("    ⇒ no pin: the measured crossover rounds to the in-code default. That is a " *
                "result — the default is adequate here.")
        return Pair{String, Any}[]
    end
    return Pair{String, Any}["sme_inplace_njp" => pin]
end

# ── sme_{dot,asum,scal,axpy}_min: where a fixed ZA prologue disappears into the stream ──────────────
# DECOMPOSITION ROUTE, no runtime hook needed: each floor selects between two kernels that are both
# separately callable, so the crossover is measured by timing them against each other instead of by
# moving the floor and timing the public entry. That distinction is load-bearing. The entry takes the
# ZA route only at or above its floor, so a floor-gated sweep compares the SAME kernel against itself
# at every candidate below the floor and reports a clean row that means nothing — a knob cannot be
# calibrated by an instrument the knob steers.
#
# COLD REGIME, because that is what BLAS-1 is graded in: `evals = 1` re-runs setup per sample, so
# every call gets a freshly allocated operand touched once. Warm, a rep loop amortizes the prologue
# over many passes and the crossing sits 16-64x lower, which is a different question.
#
# THE FOUR DO NOT SHARE A CROSSING — it tracks the operation's stream count, the one-stream reduction
# turning earliest and the read-modify-write form last — so each is adjudicated separately and each
# may pin, decline or tie on its own evidence.
const SME_L1_MIN_GRID = (32768, 65536, 131072, 262144, 524288)

"""Crossover for one ZA level-1 floor: the smallest grid size where the ZA kernel wins by the margin."""
function _sme_l1_cross(key::String, ::Type{T} = Float64) where {T}
    if !(PureBLAS._SME_F64 && PureBLAS._SME_DOT_ENTRY[] != C_NULL)
        println("  $key: DECLINED — no live SME build, so the ZA kernels are not callable here.")
        return nothing
    end
    ops = (
        (key = "sme_dot_min", inc = PureBLAS._SME_DOT_MIN,
         za = n -> (x = randn(T, n); y = randn(T, n); () -> PureBLAS._sme_dot(n, x, y)),
         nb = n -> (x = randn(T, n); y = randn(T, n); () -> PureBLAS._dot_simd(n, x, y, T))),
        (key = "sme_asum_min", inc = PureBLAS._SME_ASUM_MIN,
         za = n -> (x = randn(T, n); () -> PureBLAS._sme_asum(n, x)),
         nb = n -> (x = randn(T, n); () -> PureBLAS._asum_simd(n, x, T))),
        (key = "sme_scal_min", inc = PureBLAS._SME_SCAL_MIN,
         za = n -> (x = randn(T, n); () -> PureBLAS._sme_scal!(n, T(1.5), x)),
         nb = n -> (x = randn(T, n); () -> PureBLAS._scal_simd!(n, T(1.5), x))),
        (key = "sme_axpy_min", inc = PureBLAS._SME_AXPY_MIN,
         za = n -> (x = randn(T, n); y = randn(T, n); () -> PureBLAS._sme_axpy!(n, T(1.5), x, y)),
         nb = n -> (x = randn(T, n); y = randn(T, n); () -> PureBLAS._axpy_simd!(n, T(1.5), x, y))),
    )
    op = ops[findfirst(o -> o.key == key, ops)::Int]
    gains, ok, drift = with_anchor(; tol = TOL) do
        # estimator-ok: `median` IS the sanctioned estimator; `Measure.tstat` reduces ARM RATIOS, and
        # each of these is one absolute latency with no arm to divide by until the gain below.
        map(SME_L1_MIN_GRID) do n
            tz = median(@be op.za(n) (f -> f()) evals = 1 seconds = 0.4).time
            tn = median(@be op.nb(n) (f -> f()) evals = 1 seconds = 0.4).time
            tn / tz
        end
    end
    if !ok
        @printf("  %s: REFUSED — machine state moved %.1f%% across the measurement\n", key, 100 * drift)
        return nothing
    end
    @printf("  %-13s gains %s\n", key,
            join((@sprintf("%d:%.2f", n, g) for (n, g) in zip(SME_L1_MIN_GRID, gains)), "  "))
    # The floor belongs at the SMALLEST candidate the ZA route wins by more than the margin. Below it
    # the route loses, and a floor sitting below its own crossing is the defect this measures away from.
    i = findfirst(>(1 + DELTA), collect(gains))
    if isnothing(i)
        println("    ⇒ no pin: the ZA route does not win by the margin at any candidate on this " *
                "grid, so this evidence cannot place a floor.")
        return nothing
    end
    star = SME_L1_MIN_GRID[i]
    if star == op.inc
        @printf("    ⇒ no pin: the crossover is the in-code default (%d). That is a result.\n", op.inc)
        return nothing
    end
    @printf("    ⇒ WINNER %s=%d (incumbent %d)\n", key, star, op.inc)
    return star
end

# One calibrator per floor rather than one that returns four pins: each knob then appears on its own
# line of the tuner's output and in `KNOBS`, and `test/knob_registry_tests.jl` reads the emitted keys
# out of these literals — a loop pushing computed keys would leave that lint seeing none of them and
# the registry still calling all four orphaned.
function calibrate_sme_dot_min(::Type{T} = Float64) where {T}
    v = _sme_l1_cross("sme_dot_min", T)
    return isnothing(v) ? Pair{String, Any}[] : Pair{String, Any}["sme_dot_min" => v]
end
function calibrate_sme_asum_min(::Type{T} = Float64) where {T}
    v = _sme_l1_cross("sme_asum_min", T)
    return isnothing(v) ? Pair{String, Any}[] : Pair{String, Any}["sme_asum_min" => v]
end
function calibrate_sme_scal_min(::Type{T} = Float64) where {T}
    v = _sme_l1_cross("sme_scal_min", T)
    return isnothing(v) ? Pair{String, Any}[] : Pair{String, Any}["sme_scal_min" => v]
end
function calibrate_sme_axpy_min(::Type{T} = Float64) where {T}
    v = _sme_l1_cross("sme_axpy_min", T)
    return isnothing(v) ? Pair{String, Any}[] : Pair{String, Any}["sme_axpy_min" => v]
end

const KNOBS = (
    (name = "gemm_mt_work", fn = calibrate_gemm_mt_work),
    (name = "gemv_mr", fn = calibrate_gemv_mr),
    (name = "ger_panel_np", fn = calibrate_ger_np),
    (name = "gemvt_percol_window", fn = calibrate_gemvt_window),
    (name = "potrf_upper_direct_max", fn = calibrate_potrf_udirect),
    (name = "sytrf_cmult", fn = calibrate_sytrf_cmult),
    (name = "gemvt_pf", fn = calibrate_gemvt_pf),
    (name = "trmv_fused_min", fn = calibrate_trmv_fused_min),
    (name = "gbtrf_cmult", fn = calibrate_gbtrf_cmult),
    (name = "sme_inplace_njp", fn = calibrate_sme_inplace_njp),
    (name = "sme_dot_min", fn = calibrate_sme_dot_min),
    (name = "sme_asum_min", fn = calibrate_sme_asum_min),
    (name = "sme_scal_min", fn = calibrate_sme_scal_min),
    (name = "sme_axpy_min", fn = calibrate_sme_axpy_min),
)

# ── driver ──────────────────────────────────────────────────────────────────────────────────────────
# FREQUENCY LOCK FIRST — before contention, before anything. A value pinned from a boosting box is
# worse than no pin: it is wrong in a way nothing downstream can detect, because the clock drifts
# BETWEEN the arms of an A/B and paired measurement cannot cancel that.
locked, why = freq_locked()
if !locked && !UNLOCKED
    @error "REFUSING TO TUNE — $why.\n" *
           "  WITHOUT root (the normal case), re-run in unlocked mode:\n" *
           "      PureBLAS.tune!(unlocked = true)          # the documented entry point\n" *
           "      julia --project=bench bench/calibrate.jl unlocked   # or the script directly\n" *
           "  It decides in CYCLES, which a boosting clock cannot bias, and tightens its drift " *
           "tolerance, win margin and round count to compensate for the missing lock.\n" *
           "  WITH root, you can instead pin the clock: `sudo bench/fleet_freqlock.sh lock`."
    exit(4)
end
if !locked
    @warn "UNLOCKED calibration — $why.\n" *
          "  Deciding on CYCLES, not wall time, so boost cancels rather than biasing an arm.\n" *
          "  Tightened: anchor tol $(TOL), win margin $(DELTA), $(ROUNDS) rounds.\n" *
          "  Emitted pins are stamped `measured_unlocked = true`. For the most reliable result, " *
          "close other applications and leave the machine idle while this runs (~1 min)."
else
    println("freq: $why")
end
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
# Zen5 2026-08-21 reported boost=0, scaling_min==scaling_max==2000 MHz, governor `performance`
# — every knob in the right position — and ran the calibration at an ACHIEVED 4843 MHz. The pstate
# driver was not honouring the cap (it needs amd_pstate=passive), so the requested state and the real
# clock disagreed by 2.4x. This is exactly why bench/fleet_freqlock.sh VERIFIES under load instead of
# trusting what it wrote, and the tuner must do the same.
let capfile = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"
    cap = isfile(capfile) ? tryparse(Int, strip(read(capfile, String))) : nothing   # no cpufreq (e.g. macOS/ARM) -- nothing to sanity-check against, same "unchecked" degradation as FreqLock.lock_state
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
# The MERGED preference view (LocalPreferences.toml + any Project.toml `[preferences]`), which is what
# `@load_preference` actually resolved at load time — not just the file this script writes.
const _ACTIVE_PREFS = try
    Base.get_preferences(Base.PkgId(PureBLAS).uuid)
catch
    Dict{String, Any}()
end
# AN ALREADY-PINNED KNOB CANNOT BE CALIBRATED — skip it, loudly, instead of crashing.
# The Measure-tier resolvers are `@static if isnothing(<KEY>_PREF)`-gated (see `_GEMVT_PF_REF` in
# src/blas2/level2.jl:837 and its siblings `_GEMVT_NC_REF`, `_TRMV_FUSED_MIN_REF`,
# `_GEMVN_MINNER_REF`): a pinned build must not COMPILE the resolver at all, because it allocates on
# first use and would redden the all-paths AllocCheck proof on `gemv!`/`trmv!`. So the Ref the
# calibrator writes does not exist on a pinned box, and `PureBLAS._GEMVT_PF_REF[] = v` throws
# `UndefVarError` — which is what killed all 3 of `tune!()`'s runs on Zen4 on 2026-09-10 and,
# because a failed run is dropped, produced "0 of 3 runs" and pinned nothing at all.
# Skipping is not a workaround, it is the correct answer twice over: the resolver is gone, and even
# if it were not, the pin outranks whatever the calibration decided. Re-tuning a pinned knob means
# clearing its pin first — a deployment decision, i.e. the user's.
for k in KNOBS
    isnothing(ONLY) || k.name == ONLY || continue
    if haskey(_ACTIVE_PREFS, k.name)
        println("\n── $(k.name) ── SKIPPED: already pinned in LocalPreferences.toml " *
                "(the pin compiles out the resolver this calibrator drives, and outranks it anyway). " *
                "Clear the pin to re-tune this knob.")
        continue
    end
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
    # PROVENANCE: never leave it ambiguous whether a pin came from a locked box. A value measured on a
    # boosting laptop is legitimate — the A/B reduces in cycles, so boost cancels — but it carries more
    # uncertainty than one taken under a pinned clock, and whoever reads this file later must be able to
    # tell which they are looking at without re-deriving it. Written only in unlocked mode so a locked
    # run's file is unchanged.
    UNLOCKED && (sect["measured_unlocked"] = true)
    open(io -> TOML.print(io, d), LP, "w")
    println("\nwrote $(length(prefs)) preference(s) + tuned_for → $LP")
    println("(reload Julia to pick them up — they are read at runtime, so no recompile)")
end
