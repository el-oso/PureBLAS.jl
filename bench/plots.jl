# Generates the performance plots embedded in docs/src/performance.md: per-op PureBLAS/OpenBLAS ratio
# (single-thread, Float64). BLAS-1/2 as VIOLINS (ratio distribution over the size sweep); BLAS-3/LAPACK as
# ratio-vs-size TREND lines on a log-y axis (their ratio has a strong size dependence — small n is
# overhead-bound, large n gates). Hand-written SVG — no plotting dependency (keeps the bench env light,
# matches the pure/minimal ethos). Each (op,size) is measured over repeated rounds; per round the OB and
# PB windows run consecutively (ABBA-alternated) and are reconciled by `_qratios`, then the per-round ratios
# are POOLED (median = gate). Repetition rejects the one-unlucky-window failure single windows are prone to.
#
# Measured ratio samples are CACHED per-size to bench/plots_data_<host>.txt so the (slow) benchmark runs
# once and re-plotting (styling tweaks) is instant. Usage (pinned):
#   taskset -c 2 julia --project=bench bench/plots.jl          # use cache if present, else measure + cache
#   taskset -c 2 julia --project=bench bench/plots.jl bench    # force re-measure (refresh the cache)
#   julia --project=bench bench/plots.jl plot                  # plot from cache only (never measure)
#   taskset -c 2 julia --project=bench bench/plots.jl bench mkl # reference = Intel MKL instead of OpenBLAS
#                                                              # (Haswell target: `]add MKL` first; MKL uses
#                                                              #  its native Haswell kernels. On AMD MKL
#                                                              #  throttles to a generic path — Intel only.)
using PureBLAS, LinearAlgebra, Statistics, Printf
using Chairmarks: @be   # robust per-side timing (auto sample-sizing + warmup); replaces hand-rolled time_ns
# Reference BLAS: OpenBLAS (default), Intel MKL (`mkl` arg), or AMD AOCL (`aocl` arg). Each package
# LBT-forwards LinearAlgebra's BLAS+LAPACK to itself on load, so `B`/`LAPACK` below transparently measure
# against whichever is active — one code path, three baselines. AOCL (AMD-tuned BLIS + libFLAME) is a
# SEPARATE baseline from OpenBLAS: its caches/SVGs carry an `_aocl` suffix and never mix with OpenBLAS's.
const REFBK = "aocl" in ARGS ? "aocl" : "mkl" in ARGS ? "mkl" : "openblas"
REFBK == "mkl" && @eval using MKL

# ══ v3 ARMS ═══════════════════════════════════════════════════════════════════════════════════════
# EVERY reference is measured in ONE run, interleaved with PureBLAS per round, and the cache stores
# TIMES — never ratios. Two reasons, and the first is correctness, not convenience:
#
#  1. THE GATE IS max(OpenBLAS, AOCL). Until v3 the two references lived in separate cache files from
#     separate runs, so that max combined numbers that never saw the same machine state — different
#     frequency history, different page/TLB state, different array addresses. Interleaving all arms
#     inside one round makes the gate exact instead of approximately-comparable.
#  2. Ratios are lossy. Stored times give absolute GB/s and GFlop/s for roofline work, let the ratio
#     definition change without re-measuring, and expose run-to-run drift (an `op=` re-measure and a
#     `group=` sweep read the SAME trmv@512 cell as 1.001 and 0.959 on 2026-08-01 — a 4% spread that
#     was invisible while only the quotient was kept).
#
# Switching backend is an LBT re-forward BETWEEN timed windows: the `ref` arm closure of every op calls
# `B.*`, which routes to whatever is currently forwarded, so one closure serves every backend and no
# per-backend benchmark code exists.
using OpenBLAS_jll
const _ARM_PB = "pb"
# `arms=pb` measures ONLY PureBLAS and reuses each reference arm already in the cache. That is the fast
# iteration path; it is also the one that can silently go stale, which is why every arm carries its own
# timestamp+commit and the table reports reference age rather than hiding it.
const _ARMS_SEL = (i = findfirst(a -> startswith(a, "arms="), ARGS);
                   isnothing(i) ? nothing : split(ARGS[i][6:end], ","))
# One cell, not a whole op: `op=trmv size=512`. Sizes are matched exactly against the op's own list.
const _SELSIZE = (i = findfirst(a -> startswith(a, "size="), ARGS);
                  isnothing(i) ? nothing : parse(Int, ARGS[i][6:end]))
# AOCL = AMD's Zen-tuned AOCL-BLIS + AOCL-libFLAME, shipped as the `AOCL_jll` artifact (AMD's own release,
# NOT generic blis_jll/libflame_jll). We LBT-forward its artifact .so paths directly (BLAS→libblis-mt,
# LAPACK→libflame), which is exactly what the AOCL.jl wrapper does — using the JLL keeps the dep to the
# reproducible binary artifact. `libblis-mt` is a multi-thread build; pin to 1 thread for a fair
# single-thread comparison (BLIS reads these at init; BLAS.set_num_threads(1) below re-enforces via LBT).
ENV["BLIS_NUM_THREADS"] = "1"; ENV["OMP_NUM_THREADS"] = "1"   # BLIS reads these at init, before any forward
@eval using AOCL_jll

# Forward LBT to one backend. Called between timed windows, never inside one. `clear=true` on the BLAS
# forward drops the previous backend's symbols so a partial forward can never leave a mixed BLAS/LAPACK
# state — the failure mode where you measure AOCL's BLAS against OpenBLAS's LAPACK and never notice.
function _use_ref!(name::AbstractString)
    if name == "aocl"
        LinearAlgebra.BLAS.lbt_forward(AOCL_jll.aocl_blas_ilp64; clear = true)   # BLAS   → libblis-mt.so
        LinearAlgebra.BLAS.lbt_forward(AOCL_jll.aocl_lapack_ilp64)               # LAPACK → libflame.so
    elseif name == "openblas"
        LinearAlgebra.BLAS.lbt_forward(OpenBLAS_jll.libopenblas_path; clear = true)
    else
        error("unknown reference backend $name")
    end
    BLAS.set_num_threads(1)      # re-assert through the NEW forward; a fresh backend does not inherit it
    return name
end

# Reference arms available this run. `_REF_ARMS` is what gets measured; PureBLAS is always measured
# unless the cache already holds it and only references were asked for.
const _REF_ALL = REFBK == "mkl" ? ["mkl"] : ["openblas", "aocl"]
const _REF_ARMS = isnothing(_ARMS_SEL) ? _REF_ALL : [a for a in _REF_ALL if a in _ARMS_SEL]
const _DO_PB = isnothing(_ARMS_SEL) || (_ARM_PB in _ARMS_SEL)
const _ACTIVE_ARMS = vcat(_DO_PB ? [_ARM_PB] : String[], _REF_ARMS)
isempty(_ACTIVE_ARMS) && error("arms=$(join(something(_ARMS_SEL, []), ",")) selected nothing; valid: $_ARM_PB,$(join(_REF_ALL, ","))")

# One measured arm of one cell, with ITS OWN provenance. Per-arm (not per-cell) timestamps are the whole
# point: `arms=pb` rewrites only the pb record, so the reference records keep the date and commit at
# which they were actually measured and the table can report their age instead of implying freshness.
struct ArmRec
    time::String
    commit::String
    # Machine-state anchor AT THE TIME THIS ARM WAS MEASURED, in seconds. NaN for records written before
    # this field existed. This is what makes a cached reference comparable at all: `arms=pb` measures PB
    # today against OpenBLAS/AOCL captured days ago, and a same-run ratio cancels machine state while a
    # CACHED ratio cancels nothing. The header already stamped an anchor, but only for the CURRENT run —
    # the reference epoch's value was overwritten every time, so the correction the anchor exists for
    # could never actually be computed. Measured 2026-08-07: references were 38 h older than the PB arm
    # on both wintermute and galen, and galen's anchor moved 13.97 → 16.46 µs (17.8%) between two
    # freq-locked runs, which is far larger than the gaps being adjudicated.
    anchor::Float64
    q::Vector{Float64}      # the 48 `_QS` quantiles of that arm's sample times, in seconds
end
const ArmData = Dict{String, Vector{Float64}}   # in-run:  arm => pooled quantile samples
const CellData = Dict{String, ArmRec}           # cached:  arm => record

# Rotate arm order by round. With 2 arms this is exactly the old ABBA alternation; with k arms it keeps
# each arm in the cold first slot equally often, which is the property ABBA was buying.
_round_arms(r::Int) = circshift(_ACTIVE_ARMS, r - 1)
# Stamp each freshly measured arm with the time and commit it was measured at. Done HERE, at the point of
# measurement, rather than at save time: a run that measures L1 at 14:00 and CL3 at 17:00 must not write
# 17:00 against the L1 cells, which is exactly the imprecision the v2 single header commit had.
_stamp(acc::ArmData) = CellData(
    a => ArmRec(Libc.strftime("%Y-%m-%dT%H:%M", time()), _COMMIT, _run_anchor(), q) for (a, q) in acc
)
# Measured ONCE per run, lazily, on the first cell stamped — not at save time. Save time is the END of a
# sweep that can run for hours, and the point of the field is to describe the machine while the arms were
# actually being timed. One anchor per run (not per cell) is deliberate: arms within a run are compared
# same-run and already cancel machine state; the field exists for the ACROSS-run comparison.
const _RUN_ANCHOR = Ref{Union{Nothing, Float64}}(nothing)
function _run_anchor()
    isnothing(_RUN_ANCHOR[]) && (_RUN_ANCHOR[] = try
            _anchor_secs()
        catch
            NaN
        end)
    return _RUN_ANCHOR[]::Float64
end
# Progress line only: median ratio of the first available reference against pb this round. Under
# `arms=pb` there IS no reference in this round (that is the point of the mode), so fall back to pb's
# median time in µs — a bare NaN told you nothing, and the per-round time is exactly what you want to
# eyeball for stability when iterating on the kernel. `_ROUNDLBL` says which you are looking at.
const _ROUNDLBL = isempty(_REF_ARMS) ? "rounds (pb µs)" : "rounds"
function _round_med(qs::ArmData)
    haskey(qs, _ARM_PB) || return NaN
    for a in _REF_ARMS
        haskey(qs, a) && return median(_ratio(qs[a], qs[_ARM_PB]))
    end
    return median(qs[_ARM_PB]) * 1e6
end
const REFNAME = REFBK == "mkl" ? "MKL" : REFBK == "aocl" ? "AOCL" : "OpenBLAS"
# cache/SVG filename suffix: "" for OpenBLAS (the default baseline — its artefacts are UNTOUCHED), "_mkl"/"_aocl" otherwise
const REFSUF = REFBK == "openblas" ? "" : "_$REFBK"
import LinearAlgebra.BLAS as B
BLAS.set_num_threads(1)

# ── ISA / µarch identity (derived once, up here so `save_cache` can STAMP it into the cache header). A
# later multi-host plot loads several `plots_data_<host>.txt` and must tell Zen4/Zen3/Zen5 apart — the
# filenames are bare hostnames and the SIMD width alone can't (Zen4 & Zen5 are both AVX-512). Same-ISA
# boxes disambiguate via `slug=`/`isa=` CLI overrides (e.g. neuromancer runs `slug=zen5 isa=Zen5`). ─────
const _BENCH_VERSION = 3   # v3 = per-arm TIMES + per-arm provenance; v2 = pooled ratios (unconvertible). Bump ⇒ old caches refused.
const _W64P = PureBLAS._vwidth(Float64)
# µarch slug DERIVED from CPU detection (CLAUDE.md req#7 — not a manual flag), so Zen4 vs Zen5 (both
# AVX-512) disambiguate on their own: Zen4 is double-pumped 512, Zen5 is native. Override stays as an
# escape hatch (`slug=`/`isa=`) for an unknown box. This fixes the "run Zen5 without slug=zen5 → mislabel".
const _ISAOVR = (i = findfirst(a -> startswith(a, "isa="), ARGS); isnothing(i) ? nothing : ARGS[i][5:end])
const _SLUGOVR = (i = findfirst(a -> startswith(a, "slug="), ARGS); isnothing(i) ? nothing : ARGS[i][6:end])
const _HWB = PureBLAS._HW
const _AUTOSLUG = _W64P == 8 ? (PureBLAS._double_pumped(_HWB) ? "avx512" : "zen5") :   # Zen4 dp-512 vs Zen5 native
    _W64P == 4 ? "avx2" : _W64P == 2 ? "neon" : "simd"
# ISA is the instruction set (AVX-512 for BOTH Zen4 double-pumped and Zen5 native — the native-vs-pumped
# distinction is a µarch trait, carried by `uarch=` now, not the ISA). Keeping them both AVX-512 avoids the
# redundant "Zen5 · Zen5" legend the old (µarch-in-ISA) value produced.
const _AUTOISA = _W64P == 8 ? "AVX-512" : _W64P == 4 ? "AVX2" : _W64P == 2 ? "NEON" : "SIMD"
const ISA = isnothing(_ISAOVR) ? _AUTOISA : _ISAOVR
const _SLUGB = isnothing(_SLUGOVR) ? _AUTOSLUG : _SLUGOVR
const SLUG = _SLUGB   # v3: identifies the MACHINE, not the reference — one cache serves all arms
# AUTHORITATIVE µarch name, resolved on the MEASURING machine (from its own CpuId-derived slug) and stamped
# into the cache header. The multi-host plot then READS this — it must NOT re-derive µarch at plot time from
# the plotting box's local vector width (that was the mislabel bug: three caches all relabelled as whatever
# CPU rendered them, so Zen3/Zen4/Zen5 lines got swapped). Self-documenting + can't be swapped downstream.
const _MYUARCH = get(
    Dict("avx512" => "Zen4", "zen5" => "Zen5", "avx2" => "Zen3", "neon" => "ARM"),
    _SLUGB, uppercasefirst(_SLUGB)
)
# Provenance stamped into every cache header (self-describing: which CPU, what code, when measured).
const _CPUNAME = replace(strip(Sys.cpu_info()[1].model), r"[\t\r\n]" => " ")   # e.g. "AMD Ryzen 9 7950X …"
const _COMMIT = try
    readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`)
catch
    "unknown"
end

# Resolved Measure-tier tuning state, stamped into the cache header (see the `tune=` note at the write
# site). These are the knobs whose value can differ per box AND be silently overridden by an untracked
# LocalPreferences.toml — so a cache file must carry them to be reproducible from its own header.
# Add a knob here whenever a new Measure-tier constant starts influencing a benched routine.
_tunestamp() = try
    join((
        "ger_np=$(PureBLAS._ger_np())",
        "gemvt_perscan=$(PureBLAS._gemvt_perscan_mode())",   # 0=blocked all n · 1=residency window · 2=per-column all n
        "gemvt_u=$(PureBLAS._gemvt_u())",
        "cgemvn_nc_big=$(PureBLAS._cgemvn_nc_big())",
        # The axpy shape knobs were NOT stamped until 2026-08-06, and their absence bit immediately:
        # the run that proved `axpy_dram`'s duel migration closed three gate cells could not show from
        # its own artifact WHICH kernel produced it — the value had to be inferred from a separate
        # acceptance test. A knob that selects a shipped kernel belongs in the provenance line.
        "axpy_band=$(PureBLAS._axpy_band())",
        "axpy_dram=$(PureBLAS._axpy_dram())",
        # trmv's unblocked→fused8 crossover (Derive-tier default, but forceable — the sub-threshold side
        # was validated against a structure `_trmv_fused8!` replaced, so a sweep is expected here).
        "trmv_fused_min=$(PureBLAS._trmv_fused_min(Float64))",
    ), ",")
catch e
    "unavailable($(typeof(e)))"
end

# Iteration / robustness modes (for a fast dev loop — full `bench` remains the trustworthy artifact):
#   bench lite       → few rounds + small sizes, ~1–2 min smoke (NOT gate numbers; cache is *_lite.txt)
#   bench op=gemm    → measure ONLY that op, full methodology, MERGE into the (v2) cache
#   bench group=L3   → measure ONLY that level, merge
const _LITE = "lite" in ARGS
const _NODRAW = "nodraw" in ARGS   # fleet boxes: measure + cache only, skip SVG/table render (so their
# working tree stays clean → `git pull` never blocks). Render centrally.
const _SELOP = (i = findfirst(a -> startswith(a, "op="), ARGS); isnothing(i) ? nothing : ARGS[i][4:end])
const _SELGRP = (i = findfirst(a -> startswith(a, "group="), ARGS); isnothing(i) ? nothing : ARGS[i][7:end])
_want(lvl, nm) = (isnothing(_SELOP) && isnothing(_SELGRP)) || _SELOP == nm || _SELGRP == lvl
_cap(szs, maxn) = Tuple(s for s in szs if s <= maxn)   # per-op size cap (e.g. skip 4096 for slow ops)
# lite caps sizes at 1024 (drops the expensive 2048/4096 tail) — keeps the meaningful mid-n range while
# skipping the O(n³) large-n sink that dominates wall time. Guarded so a cap never yields an empty tuple.
_sizes(szs) = _LITE ? (t = Tuple(s for s in szs if s <= 1024); isempty(t) ? szs[1:1] : t) : szs

# Repeated rounds reject the one-unlucky-window failure (gemm n=32 read 0.83 in a single window vs 1.01
# true). Keyed on SIZE, deterministic (never on measured duration → identical protocol on every host).
# CRUCIAL: mid-size heavy windows (n=512–1024) are SAMPLES-capped, not seconds-capped, so a single window
# there is exactly the unlucky-window regime — repeat 8×. Only n≥2048 fills a 2 s seconds-bound window;
# still keep 2 rounds there so ABBA order-balance applies (windows are hottest/most order-biased there).
_rounds_light(_sz) = _LITE ? 2 : 8
_rounds_heavy(sz) = _LITE ? 1 : (sz <= 1024 ? 8 : 4)   # n≥2048 was 2 → under-replicated (noise at 4096); 4

# Measure one op ROBUSTLY: skip if filtered out; a per-op try/catch means one op's failure logs and the
# sweep CONTINUES (never all-or-nothing); flush so a run is live-monitorable despite Julia's block-buffered
# file IO. `sweeper` is a thunk returning the per-size ratio vector list.
const _MISSING = String[]   # ops that threw during measurement (surfaced at the end, not just scrolled past)
function _meas!(vec, lvl, nm, sweeper)
    _want(lvl, nm) || return
    print(stderr, "  [$lvl $nm] "); flush(stderr)
    try
        push!(vec, nm => sweeper()); println(stderr, "done"); flush(stderr)
    catch e
        e isa InterruptException && rethrow()   # let Ctrl-C actually stop the run
        push!(_MISSING, "$lvl/$nm"); println(stderr, "FAILED: ", sprint(showerror, e)); flush(stderr)
    end
    return
end

# name => per-size samples: [(size, [ratio,ratio,…]), …]. The single data model the renderer consumes.
const OpData = Pair{String, Vector{Tuple{Int, CellData}}}
# Per-op summary = (MEDIAN across cells, worst cell). The across-cell reduction was a GEOMEAN
# (`exp(sum(log,m)/length(m))`) until 2026-08-04. A geomean is a mean; the project's estimator is the
# median at EVERY level, and the reason is the same here as inside a cell — one soft size drags a mean
# and hides the shape. It also read as authoritative in gen_table.md and the published docs, so the
# banned statistic was the headline number. The worst cell stays: the gate is "every cell >= 1.0", so
# the failing cell IS the decision, not an estimate of one.
gatestat(op) = (m = [median(v) for (s, v) in op]; (median(m), minimum(m)))
# Hermitian-positive-definite operand for (z)potrf, memoized per (T,size): the O(n³) `A*A'` is built ONCE;
# each sample gets a fresh O(n²) copy (potrf is destructive). Avoids an OpenBLAS gemm + 2 big allocs PER
# sample (seconds of wasted setup at n=4096). No `+zeros` — `A*A'+sI` is already dense HPD.
const _HPD = Dict{Tuple{DataType, Int}, Any}()
_hpd(T, s) = copy(get!(() -> (A = randn(T, s, s); A * A' + s * I), _HPD, (T, s)))::Matrix{T}

# Every Chairmarks sample time (seconds) — it reports min but stores all timings; we use the full set.
_times(b) = Float64[smp.time for smp in b.samples]
# v3: store the QUANTILE VECTOR of each arm, not the quotient. `_qvec` reduces a window's samples to the
# same 48 quantiles the ratio used to be formed from, so `ref_q ./ pb_q` reproduces the old `_qratios`
# EXACTLY — the pairing is preserved, nothing is approximated, and the absolute times survive.
const _QS = range(0.03, 0.97; length = 48)
_qvec(b) = (t = _times(b); [quantile(t, q) for q in _QS])
# Ratio of two stored quantile vectors. q=0.5 is the median ratio (the gate number); the spread across q
# is the violin body. Defined once here so tables and plots cannot drift apart in how they derive it.
_ratio(qref::Vector{Float64}, qpb::Vector{Float64}) = qref ./ qpb

# L1/L2 sweep: `_rounds_light(s)` rounds of consecutive ABBA-ordered OB/PB `@be` windows per size, pooling
# the per-round `_qratios`. `evals=1` reruns the setup `mk(s)` per sample so
# address/alignment varies (essential for iamax — OpenBLAS idamax swings ~60% by address) and the mk
# allocation is EXCLUDED from the timed core; `reps` amortizes the timer for tiny ops. All sample timings
# feed the ratio distribution.
function sweep(mk, sizes, work_ob, work_pb, repfn; samples = 400, seconds = 0.15)
    out = Tuple{Int, CellData}[]
    for s in sizes
        !isnothing(_SELSIZE) && s != _SELSIZE && continue     # `size=` selects ONE cell
        reps = repfn(s)
        rounds = _rounds_light(s); rmeds = Float64[]
        # Accumulate per arm. Every arm is measured inside the SAME round, so the quantile vectors that
        # later divide into a ratio saw one machine state — that is what makes max(OB,AOCL) legitimate.
        acc = Dict{String, Vector{Float64}}()
        for r in 1:rounds
            # Rotate arm order every round (the ABBA generalisation): with k arms, rotating by r keeps
            # every arm equally often in the cold first slot, which is what the old A/B alternation did.
            arms = _round_arms(r)
            qs = Dict{String, Vector{Float64}}()
            for a in arms
                w = a == _ARM_PB ? work_pb : (_use_ref!(a); work_ob)
                b = @be mk(s) (c -> w(c, reps)) evals = 1 samples = samples seconds = seconds
                qs[a] = _qvec(b)
            end
            for (a, q) in qs
                append!(get!(acc, a, Float64[]), q)
            end
            push!(rmeds, _round_med(qs))
        end
        rounds > 1 && (println(stderr, "    n=$s $_ROUNDLBL: ", join((@sprintf("%.3f", m) for m in rmeds), " ")); flush(stderr))
        push!(out, (s, _stamp(acc)))
    end
    return out
end
const _L1REP = s -> clamp(8_000_000 ÷ s, 30, 20000)           # O(s) work
const _L2REP = s -> clamp(400_000_000 ÷ (s * s), 30, 20000)   # O(s²) work

_reps_cubic(s) = clamp(20_000_000 ÷ (s * s * s), 1, 512)

# Heavy O(n³) sweep for L3 / LAPACK. `@be` with `evals=1` runs a FRESH `mk(s)` per sample (the destructive
# op mutates its input → one op per context) and EXCLUDES the mk allocation from the timed core — which
# removes the old hazard where the per-round alloc dropped the core off-clock and biased whichever side was
# timed first. Warmup + sample sizing are Chairmarks'. Small n is measured cleanly (no timer-quantization
# reps hack needed — Chairmarks amortizes internally).
# ⚠ STILL REQUIRES CPU BOOST DISABLED (`echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost`,
# performance governor) so the fixed clock keeps OB vs PB comparable. See memory dev-fleet.
function sweep_heavy(mk, ob1, pb1, sizes; samples = 64, seconds = 4.0, repsof = _reps_cubic)
    out = Tuple{Int, CellData}[]
    for s in sizes
        !isnothing(_SELSIZE) && s != _SELSIZE && continue     # `size=` selects ONE cell
        # reps fresh contexts per sample: setup (EXCLUDED from timing) pre-generates them, the core runs the
        # destructive op on each. reps→1 at large n; large at tiny n so a ~50 ns n=8 op isn't measured as a
        # single sub-timer-resolution call (which fabricated n=8 "fails" — evals=1 alone can't amortize it).
        # `repsof` is overridable because the default assumes the swept size IS the problem dimension and
        # the op is O(s³); rows that sweep some other axis (pbtrf sweeps the BANDWIDTH at fixed n, cost
        # O(n·kd²)) would otherwise be handed hundreds of full-size contexts to allocate.
        reps = repsof(s)
        rounds = _rounds_heavy(s)
        secs = s >= 1024 ? 2.0 : seconds   # large-n windows are seconds-bound; 2.0 is plenty and ~halves cost
        acc = ArmData(); rmeds = Float64[]
        for r in 1:rounds
            qs = ArmData()
            for a in _round_arms(r)      # rotated (generalised ABBA); reference switch is outside the window
                f = a == _ARM_PB ? pb1 : (_use_ref!(a); ob1)
                b = @be [mk(s) for _ in 1:reps] (
                    cs -> (
                        v = 0.0; for c in cs
                            v += f(c)
                        end; v
                    )
                ) evals = 1 samples = samples seconds = secs
                qs[a] = _qvec(b)
            end
            for (a, q) in qs
                append!(get!(acc, a, Float64[]), q)
            end
            push!(rmeds, _round_med(qs))
        end
        rounds > 1 && (println(stderr, "    n=$s $_ROUNDLBL: ", join((@sprintf("%.3f", m) for m in rmeds), " ")); flush(stderr))
        push!(out, (s, _stamp(acc)))
    end
    return out
end

const L1SZ = (1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000)
const L2SZ = (64, 128, 256, 512, 1024, 2048, 4096)
const L3SZ = (8, 32, 128, 256, 512, 1024, 2048, 4096)   # O(n³); 4096 shows large-n syrk/trmm behavior
const LPSZ = (8, 32, 128, 256, 512, 1024, 2048, 4096)   # LAPACK factorizations, to 4096
# Tridiagonal solvers are O(n), not O(n³) — at LPSZ's sizes they are microseconds of pure timer noise, and
# the interesting behaviour (L2-resident vs streaming) only appears well past 4096. Swept to 262144.
const TDSZ = (256, 1024, 4096, 16384, 65536, 262144)
# Banded Cholesky sweeps the BANDWIDTH kd (see the pbtrf rows); n is fixed at BANDN. The points
# straddle the two kd crossovers the kernel selection turns on: blocked-vs-unblocked (~4·W) and,
# for uplo='U', re-pack-vs-native (~256 on Zen4).
const BANDSZ = (16, 32, 64, 96, 128, 192, 256, 384)
const BANDN = 4096
const TN = Char(78); const TT = Char(84); const U = Char(85)

# Run the full benchmark sweep; returns (l1, l2, l3, lp) as vectors of OpData.
function run_benchmarks()
    # ── BLAS-1 ──────────────────────────────────────────────────────────────────────────────────────
    l1 = OpData[]
    let
        for (nm, ob, pb) in (
                (
                    "axpy", (c, m) -> (
                        for _ in 1:m
                            B.axpy!(1.7, c[1], c[2])
                        end; c[2][1]
                    ),
                    (c, m) -> (
                        for _ in 1:m
                            PureBLAS.axpy!(c[2], 1.7, c[1])
                        end; c[2][1]
                    ),
                ),
                (
                    "dot", (c, m) -> (
                        s = 0.0; for _ in 1:m
                            s += B.dot(c[1], c[2])
                        end; s
                    ),
                    (c, m) -> (
                        s = 0.0; for _ in 1:m
                            s += PureBLAS.dot(c[1], c[2])
                        end; s
                    ),
                ),
                (
                    "nrm2", (c, m) -> (
                        s = 0.0; for _ in 1:m
                            s += B.nrm2(c[1])
                        end; s
                    ),
                    (c, m) -> (
                        s = 0.0; for _ in 1:m
                            s += PureBLAS.nrm2(c[1])
                        end; s
                    ),
                ),
                (
                    "asum", (c, m) -> (
                        s = 0.0; for _ in 1:m
                            s += B.asum(c[1])
                        end; s
                    ),
                    (c, m) -> (
                        s = 0.0; for _ in 1:m
                            s += PureBLAS.asum(c[1])
                        end; s
                    ),
                ),
                (
                    "scal", (c, m) -> (
                        for _ in 1:m
                            B.scal!(1.0000001, c[1])
                        end; c[1][1]
                    ),
                    (c, m) -> (
                        for _ in 1:m
                            PureBLAS.scal!(1.0000001, c[1])
                        end; c[1][1]
                    ),
                ),
                (
                    "iamax", (c, m) -> (
                        s = 0; for _ in 1:m
                            s += B.iamax(c[1])
                        end; s
                    ),
                    (c, m) -> (
                        s = 0; for _ in 1:m
                            s += PureBLAS.iamax(c[1])
                        end; s
                    ),
                ),
            )
            _meas!(l1, "L1", nm, () -> sweep(s -> (randn(s), randn(s)), _sizes(L1SZ), ob, pb, _L1REP))
        end
    end

    # ── BLAS-2 ──────────────────────────────────────────────────────────────────────────────────────
    l2 = OpData[]
    let
        sq(s) = (randn(s, s), randn(s), randn(s))
        pk(s) = (randn((s * (s + 1)) ÷ 2), randn(s), randn(s))
        bd(s) = (k = 16; (randn(2k + 1, s), randn(s), randn(s), k))
        sbd(s) = (k = 16; (randn(k + 1, s), randn(s), randn(s), k))
        add(nm, mk, ob, pb) = _meas!(l2, "L2", nm, () -> sweep(mk, _sizes(L2SZ), ob, pb, _L2REP))
        add(
            "gemvN", sq, (c, m) -> (
                for _ in 1:m
                    B.gemv!(TN, 1.0, c[1], c[2], 0.0, c[3])
                end; c[3][1]
            ),
            (c, m) -> (
                for _ in 1:m
                    PureBLAS.gemv!(c[3], c[1], c[2]; alpha = 1.0, beta = 0.0)
                end; c[3][1]
            )
        )
        add(
            "gemvT", sq, (c, m) -> (
                for _ in 1:m
                    B.gemv!(TT, 1.0, c[1], c[2], 0.0, c[3])
                end; c[3][1]
            ),
            (c, m) -> (
                for _ in 1:m
                    PureBLAS.gemv!(c[3], c[1], c[2]; alpha = 1.0, beta = 0.0, trans = TT)
                end; c[3][1]
            )
        )
        add(
            "ger", sq, (c, m) -> (
                for _ in 1:m
                    B.ger!(1.0, c[2], c[3], c[1])
                end; c[1][1]
            ),
            (c, m) -> (
                for _ in 1:m
                    PureBLAS.ger!(1.0, c[2], c[3], c[1])
                end; c[1][1]
            )
        )
        add(
            "symv", sq, (c, m) -> (
                for _ in 1:m
                    B.symv!(U, 1.0, c[1], c[2], 0.0, c[3])
                end; c[3][1]
            ),
            (c, m) -> (
                for _ in 1:m
                    PureBLAS.symv!(c[3], c[1], c[2]; uplo = U, alpha = 1.0, beta = 0.0)
                end; c[3][1]
            )
        )
        add(
            "trmv", sq, (c, m) -> (
                for _ in 1:m
                    copyto!(c[3], c[2]); B.trmv!(U, TN, TN, c[1], c[3])
                end; c[3][1]
            ),
            (c, m) -> (
                for _ in 1:m
                    copyto!(c[3], c[2]); PureBLAS.trmv!(c[1], c[3]; uplo = U)
                end; c[3][1]
            )
        )
        add(
            "trsv", s -> (
                A = randn(s, s) ./ (2s); for i in 1:s
                    A[i, i] = 1 + abs(A[i, i])
                end; (A, randn(s), randn(s))
            ),
            (c, m) -> (
                for _ in 1:m
                    copyto!(c[3], c[2]); B.trsv!(U, TN, TN, c[1], c[3])
                end; c[3][1]
            ),
            (c, m) -> (
                for _ in 1:m
                    copyto!(c[3], c[2]); PureBLAS.trsv!(c[1], c[3]; uplo = U)
                end; c[3][1]
            )
        )
        add(
            "spmv", pk, (c, m) -> (
                for _ in 1:m
                    B.spmv!(U, 1.0, c[1], c[2], 0.0, c[3])
                end; c[3][1]
            ),
            (c, m) -> (
                for _ in 1:m
                    PureBLAS.spmv!(c[3], c[1], c[2]; uplo = U, alpha = 1.0, beta = 0.0)
                end; c[3][1]
            )
        )
        add(
            "gbmvN", bd, (c, m) -> (
                for _ in 1:m
                    B.gbmv!(TN, length(c[2]), c[4], c[4], 1.0, c[1], c[2], 0.0, c[3])
                end; c[3][1]
            ),
            (c, m) -> (
                n = length(c[2]); for _ in 1:m
                    PureBLAS.gbmv!(c[3], c[1], c[2], n, c[4], c[4]; trans = TN, alpha = 1.0, beta = 0.0)
                end; c[3][1]
            )
        )
        add(
            "sbmv", sbd, (c, m) -> (
                for _ in 1:m
                    B.sbmv!(U, c[4], 1.0, c[1], c[2], 0.0, c[3])
                end; c[3][1]
            ),
            (c, m) -> (
                for _ in 1:m
                    PureBLAS.sbmv!(c[3], c[1], c[2]; uplo = U, alpha = 1.0, beta = 0.0)
                end; c[3][1]
            )
        )
    end

    # ── BLAS-3 (O(n³), destructive trmm/trsm; fresh input per round) ──────────────────────────────────
    l3 = OpData[]
    let
        NN = Char(78); LT = Char(76); UP = Char(85)
        tri(s) = (
            A = randn(s, s) ./ (2s); for i in 1:s
                A[i, i] = 1 + abs(A[i, i])
            end; A
        )
        addh(nm, mk, ob, pb) = _meas!(l3, "L3", nm, () -> sweep_heavy(mk, ob, pb, _sizes(L3SZ)))
        addh(
            "gemm", s -> (randn(s, s), randn(s, s), zeros(s, s)),
            c -> (B.gemm!(NN, NN, 1.0, c[1], c[2], 0.0, c[3]); c[3][1]),
            c -> (PureBLAS.gemm!(c[3], c[1], c[2]); c[3][1])
        )
        # (zgemm is measured once, in the complex CL3 group — not duplicated here.)
        addh(
            "symm", s -> (randn(s, s), randn(s, s), zeros(s, s)),
            c -> (B.symm!(LT, UP, 1.0, c[1], c[2], 0.0, c[3]); c[3][1]),
            c -> (PureBLAS.symm!(c[3], c[1], c[2]; side = LT, uplo = UP); c[3][1])
        )
        addh(
            "syrk", s -> (randn(s, s), zeros(s, s)),
            c -> (B.syrk!(UP, NN, 1.0, c[1], 0.0, c[2]); c[2][1]),
            c -> (PureBLAS.syrk!(c[2], c[1]; uplo = UP, trans = NN); c[2][1])
        )
        addh(
            "syr2k", s -> (randn(s, s), randn(s, s), zeros(s, s)),
            c -> (B.syr2k!(UP, NN, 1.0, c[1], c[2], 0.0, c[3]); c[3][1]),
            c -> (PureBLAS.syr2k!(c[3], c[1], c[2]; uplo = UP, trans = NN); c[3][1])
        )
        addh(
            "trmm", s -> (tri(s), randn(s, s)),
            c -> (B.trmm!(LT, UP, NN, NN, 1.0, c[1], c[2]); c[2][1]),
            c -> (PureBLAS.trmm!(c[2], c[1]; side = LT, uplo = UP); c[2][1])
        )
        addh(
            "trsm", s -> (tri(s), randn(s, s)),
            c -> (B.trsm!(LT, UP, NN, NN, 1.0, c[1], c[2]); c[2][1]),
            c -> (PureBLAS.trsm!(c[2], c[1]; side = LT, uplo = UP); c[2][1])
        )
        addh(
            "trsmR", s -> (tri(s), randn(s, s)),   # side-R lower-T (the potrf/getrf panel-solve shape) — the lever
            c -> (B.trsm!('R', 'L', 'T', 'N', 1.0, c[1], c[2]); c[2][1]),
            c -> (PureBLAS.trsm!(c[2], c[1]; side = 'R', uplo = 'L', transA = 'T'); c[2][1])
        )
    end

    # ── LAPACK (O(n³) factorizations; all destructive → fresh input per round) ─────────────────────────
    lp = OpData[]
    let
        LP = Char(76); UP = Char(85)
        addh(nm, mk, ob, pb; sizes = LPSZ, repsof = _reps_cubic) =
            _meas!(lp, "LP", nm, () -> sweep_heavy(mk, ob, pb, _sizes(sizes); samples = 40, repsof = repsof))
        addh(
            "potrf", s -> _hpd(Float64, s),
            c -> (LinearAlgebra.LAPACK.potrf!(LP, c); c[1, 1]),
            c -> (PureBLAS.potrf!(c; uplo = LP); c[1, 1])
        )
        # Upper takes its own route (Lever A: transpose into scratch → faer lower kernels → transpose back),
        # so gating only 'L' leaves half of potrf unmeasured. See the zpotrfU note in the CLP group.
        addh(
            "potrfU", s -> _hpd(Float64, s),
            c -> (LinearAlgebra.LAPACK.potrf!(UP, c); c[1, 1]),
            c -> (PureBLAS.potrf!(c; uplo = UP); c[1, 1])
        )
        addh(
            "geqrf", s -> randn(s, s),
            c -> (LinearAlgebra.LAPACK.geqrf!(c); c[1, 1]),
            c -> (PureBLAS.geqrf!(c); c[1, 1])
        )
        addh(
            "getrf", s -> randn(s, s),
            c -> (LinearAlgebra.LAPACK.getrf!(c); c[1, 1]),
            c -> (PureBLAS.getrf!(c); c[1, 1])
        )
        # ── SOLVES on given factors: getrs / potrs / trtrs ────────────────────────────────────────────
        # These back `lu(A) \ b`, `cholesky(A) \ b` and triangular `\` — among the most-executed LAPACK
        # in practice — and had NEVER been gated: they existed only as C-ABI shims, and the harness
        # compares PureBLAS.foo! against LAPACK.foo!, so with no native entry point there was nothing to
        # measure. nrhs is fixed at 1 (the `\` case; a vector RHS is the common one and the one where
        # per-call overhead shows). `mk` returns the FACTORS, so the timed core is the solve alone.
        _lufac(s) = (F = _hpd(Float64, s); ip = Vector{Int}(undef, s); PureBLAS.getrf!(F, ip); (F, ip, randn(s)))
        addh(
            "getrs", _lufac,
            c -> (LinearAlgebra.LAPACK.getrs!(TN, c[1], c[2], c[3]); c[3][1]),
            c -> (PureBLAS.getrs!(c[1], c[2], c[3]; trans = TN); c[3][1]); sizes = _cap(LPSZ, 2048)
        )
        _chfac(s, uplo) = (C = _hpd(Float64, s); PureBLAS.potrf!(C; uplo = uplo); (C, randn(s)))
        for uplo in ('L', 'U')
            addh(
                "potrs$uplo", s -> _chfac(s, uplo),
                c -> (LinearAlgebra.LAPACK.potrs!(uplo, c[1], c[2]); c[2][1]),
                c -> (PureBLAS.potrs!(c[1], c[2]; uplo = uplo); c[2][1]); sizes = _cap(LPSZ, 2048)
            )
        end
        addh(
            "trtrs", s -> (triu(_hpd(Float64, s)), randn(s)),
            c -> (LinearAlgebra.LAPACK.trtrs!(UP, TN, Char(78), c[1], c[2]); c[2][1]),
            c -> (PureBLAS.trtrs!(c[1], c[2]; uplo = UP, trans = TN, diag = Char(78)); c[2][1]);
            sizes = _cap(LPSZ, 2048)
        )
        # ── Symmetric-indefinite (Bunch-Kaufman), banded LU, pivoted QR, least squares ────────────────
        # A whole factorization+solve pair (sytrf/sytrs) plus four more real factorizations that had
        # correctness tests but had NEVER been measured. Adding the rows is step 1 of the ALL-LAPACK
        # audit: a routine that is routed and tested still reads as covered while measuring nothing.
        # INDEFINITE, not positive definite. The original maker was (hpd + hpd')/2, which is PD, so
# every Bunch-Kaufman pivot was 1x1 and the 2x2 branches of BOTH sytrf and sytrs were never
# measured — the row reported a number for a code path real symmetric-indefinite input does
# not take. A plain M + M' is the actual workload.
_symm_hpd(s) = (M = randn(Float64, s, s); M .+ transpose(M))
        addh(
            "sytrf", _symm_hpd,
            c -> (LinearAlgebra.LAPACK.sytrf!(LP, c); c[1, 1]),
            c -> (PureBLAS.sytrf!(c, Vector{Int}(undef, size(c, 1)); uplo = LP); c[1, 1])
        )
        _sytrfac(s) = (A = _symm_hpd(s); ip = Vector{Int}(undef, s); PureBLAS.sytrf!(A, ip; uplo = LP); (A, ip, randn(s)))
        addh(
            "sytrs", _sytrfac,
            c -> (LinearAlgebra.LAPACK.sytrs!(LP, c[1], c[2], c[3]); c[3][1]),
            c -> (PureBLAS.sytrs!(c[1], c[2], c[3]; uplo = LP); c[3][1]); sizes = _cap(LPSZ, 2048)
        )
        # Banded LU: kd scales with n (a fixed narrow band makes this O(n) and hides the kernel).
        _gbd(s) = (kl = max(1, s ÷ 8); ku = kl; AB = zeros(Float64, 2kl + ku + 1, s);
            for j in 1:s, i in 1:(2kl + ku + 1); AB[i, j] = randn(); end;
            for j in 1:s; AB[kl + ku + 1, j] = 4 * (kl + ku); end; (kl, ku, AB))
        addh(
            "gbtrf", _gbd,
            c -> (LinearAlgebra.LAPACK.gbtrf!(c[1], c[2], size(c[3], 2), c[3]); c[3][1]),
            c -> (PureBLAS.gbtrf!(c[1], c[2], size(c[3], 2), c[3]); c[3][1]); sizes = _cap(LPSZ, 2048)
        )
        addh(
            "geqp3", s -> randn(s, s),
            c -> (LinearAlgebra.LAPACK.geqp3!(c); c[1, 1]),
            c -> (PureBLAS.geqp3!(c); c[1, 1]); sizes = _cap(LPSZ, 2048)
        )
        addh(
            "gels", s -> (randn(s, s), randn(s, 1)),
            c -> (LinearAlgebra.LAPACK.gels!(TN, copy(c[1]), copy(c[2])); c[2][1]),
            c -> (PureBLAS.gels!(TN, copy(c[1]), copy(c[2])); c[2][1]); sizes = _cap(LPSZ, 1024)
        )
        # Pivoted (semidefinite) Cholesky — blocked dpstrf: BLAS-2 pivoted panel + rank-jb syrk trailing,
        # with the leading row swaps batched per panel (they are stride-lda and were ~47% of the runtime).
        addh(
            "pstrf", s -> _hpd(Float64, s),
            c -> (LinearAlgebra.LAPACK.pstrf!(LP, c, -1.0); c[1, 1]),
            c -> (PureBLAS.pstrf!(c, -1.0; uplo = LP); c[1, 1])
        )
        # uplo='U' is a SEPARATE code path (pivoted panel and trailing update both mirror), and gating
        # only 'L' left the one cell with a known residual — n=48/64 upper vs OpenBLAS — unmeasured.
        # Same omission that hid potrf's, pbtrf's and pptrf's upper paths.
        addh(
            "pstrfU", s -> _hpd(Float64, s),
            c -> (LinearAlgebra.LAPACK.pstrf!(UP, c, -1.0); c[1, 1]),
            c -> (PureBLAS.pstrf!(c, -1.0; uplo = UP); c[1, 1])
        )
        # real gesvd capped at 2048: OB gesdd is divide-and-conquer, PB is QR-iteration — at 4096 with vectors
        # that algorithm mismatch dominates (no actionable signal) and a single sample isn't seconds-bounded.
        addh(
            "gesvd", s -> randn(s, s),
            c -> (LinearAlgebra.LAPACK.gesdd!(Char(65), c); c[1, 1]),
            c -> (PureBLAS.gesvd!(c; want_vectors = true); 0.0); sizes = _cap(LPSZ, 2048)
        )
        # Symmetric eigensolver, blocked sytrd + D&C stedc + blocked ormtr. Baseline = OB `dsyevd` (D&C — the
        # faster of Julia's two default drivers; syevr is ~15% slower), the conservative gate target. Two rows:
        # 'V' (eigenpairs, Julia's default `eigen(Symmetric)`) and 'N' (eigenvalues, `eigvals`). Capped at 2048
        # like gesvd (a single 4096 'V' solve isn't seconds-bounded at 40 samples). Fresh symmetric input/sample.
        _sym(s) = (A = randn(s, s); A .+ A')
        addh(
            "syev", _sym,
            c -> (LinearAlgebra.LAPACK.syevd!(Char(86), LP, c); c[1, 1]),   # 'V','L'
            c -> (PureBLAS._syev!('V', 'L', c); c[1, 1]); sizes = _cap(LPSZ, 2048)
        )
        addh(
            "syevN", _sym,
            c -> (LinearAlgebra.LAPACK.syevd!(Char(78), LP, c); c[1, 1]),   # 'N','L'
            c -> (PureBLAS._syev!('N', 'L', c); c[1, 1]); sizes = _cap(LPSZ, 2048)
        )
        # ── Tridiagonal (O(n), so swept over TDSZ's much larger n, not LPSZ) ───────────────────────────
        # These are serial 3-term recurrences: the gate here is a divide→multiply→subtract latency chain,
        # not flops, and the levers are store streams and register-carried recurrences (see tridiag.jl).
        # gttrs/pttrs consume a factorization, so `mk` builds and factors the context (mk is excluded from
        # the timed core). gttrf allocates du2/ipiv on BOTH sides — Julia's LAPACK.gttrf! wrapper allocates
        # them internally, so PureBLAS must pay the same to keep the comparison honest.
        _gtd(s) = (randn(s - 1), [4.0 + abs(randn()) for _ in 1:s], randn(s - 1), randn(s))
        _ptd(s) = ([2.0 + abs(randn()) for _ in 1:s], randn(s - 1) ./ 4, randn(s))
        addh(
            "gtsv", _gtd,
            c -> (LinearAlgebra.LAPACK.gtsv!(c[1], c[2], c[3], c[4]); c[4][1]),
            c -> (PureBLAS.gtsv!(c[1], c[2], c[3], c[4]); c[4][1]); sizes = TDSZ
        )
        addh(
            "gttrf", _gtd,
            c -> (LinearAlgebra.LAPACK.gttrf!(c[1], c[2], c[3]); c[2][1]),
            c -> (
                PureBLAS.gttrf!(
                    c[1], c[2], c[3], Vector{Float64}(undef, length(c[2]) - 2),
                    Vector{Int}(undef, length(c[2]))
                ); c[2][1]
            ); sizes = TDSZ
        )
        _gtf(s) = (c = _gtd(s); (LinearAlgebra.LAPACK.gttrf!(c[1], c[2], c[3])..., c[4]))
        addh(
            "gttrs", _gtf,
            c -> (LinearAlgebra.LAPACK.gttrs!(TN, c[1], c[2], c[3], c[4], c[5], c[6]); c[6][1]),
            c -> (PureBLAS.gttrs!(TN, c[1], c[2], c[3], c[4], c[5], c[6]); c[6][1]); sizes = TDSZ
        )
        addh(
            "pttrf", _ptd,
            c -> (LinearAlgebra.LAPACK.pttrf!(c[1], c[2]); c[1][1]),
            c -> (PureBLAS.pttrf!(c[1], c[2]); c[1][1]); sizes = TDSZ
        )
        _ptf(s) = (c = _ptd(s); (LinearAlgebra.LAPACK.pttrf!(c[1], c[2])..., c[3]))
        addh(
            "pttrs", _ptf,
            c -> (LinearAlgebra.LAPACK.pttrs!(c[1], c[2], c[3]); c[3][1]),
            c -> (PureBLAS.pttrs!(c[1], c[2], c[3]); c[3][1]); sizes = TDSZ
        )
        addh(
            "ptsv", _ptd,
            c -> (LinearAlgebra.LAPACK.ptsv!(c[1], c[2], c[3]); c[3][1]),
            c -> (PureBLAS.ptsv!(c[1], c[2], c[3]); c[3][1]); sizes = TDSZ
        )
        # ── Banded Cholesky (swept over BANDWIDTH kd at fixed n, not over n) ───────────────────────────
        # pbtrf's cost is O(n·kd²) and every interesting effect — the blocked/unblocked crossover, the
        # panel width, and (uplo='U') the re-pack-vs-native kernel crossover — is a function of kd, so kd
        # is the swept axis and n is held at BANDN. Both triangles are gated: they run entirely different
        # kernels, and 'U' is the one with two of them.
        # Julia's stdlib has NO pbtrf! wrapper, so the reference is a direct ILP64 ccall through LBT
        # (which is what `ref=aocl` re-points, so this row honours the AOCL comparison like every other).
        _pbref!(uplo::Char, n::Int, kd::Int, AB::Matrix{Float64}) =
            (
                i = Ref{Int64}(0); ccall(
                    (:dpbtrf_64_, LinearAlgebra.BLAS.libblastrampoline), Cvoid,
                    (Ref{UInt8}, Ref{Int64}, Ref{Int64}, Ptr{Float64}, Ref{Int64}, Ref{Int64}, Clong),
                    UInt8(uplo), Int64(n), Int64(kd), AB, Int64(size(AB, 1)), i, 1
                ); i[]
            )
        # Diagonally dominant ⇒ HPD for any kd, so no size in the sweep can fail to factor.
        function _pbd(kd, uplo)
            AB = zeros(Float64, kd + 1, BANDN)
            dr = uplo == 'L' ? 1 : kd + 1
            for j in 1:BANDN
                AB[dr, j] = 2kd + 4 + abs(randn())
                for i in 1:min(kd, uplo == 'L' ? BANDN - j : j - 1)
                    AB[uplo == 'L' ? 1 + i : kd + 1 - i, j] = randn() * 0.3
                end
            end
            AB
        end
        for uplo in ('L', 'U')
            addh(
                "pbtrf$uplo", s -> _pbd(s, uplo),
                c -> (_pbref!(uplo, BANDN, size(c, 1) - 1, c); c[1, 1]),
                c -> (PureBLAS.pbtrf!(c; uplo = uplo, kd = size(c, 1) - 1); c[1, 1]);
                sizes = BANDSZ, repsof = _ -> 1   # one (kd+1)×BANDN context per sample: ≥80 µs even at kd=16
            )
        end
        # ── Packed Cholesky (pptrf) — sweeps n like the dense factorizations ───────────────────────────
        # Also no stdlib wrapper, so the reference is a direct ILP64 ccall (honours ref=aocl). Both
        # triangles: packed 'U' and 'L' are different loops, and 'U' is the one that needed the tpsv
        # rewrite. Cost is O(n³/6) but on packed storage, so the cubic reps heuristic applies as-is.
        _ppref!(uplo::Char, n::Int, AP::Vector{Float64}) =
            (
                i = Ref{Int64}(0); ccall(
                    (:dpptrf_64_, LinearAlgebra.BLAS.libblastrampoline), Cvoid,
                    (Ref{UInt8}, Ref{Int64}, Ptr{Float64}, Ref{Int64}, Clong),
                    UInt8(uplo), Int64(n), AP, i, 1
                ); i[]
            )
        function _ppd(n, uplo)
            A = _hpd(Float64, n)
            uplo == 'L' ? [A[i, j] for j in 1:n for i in j:n] : [A[i, j] for j in 1:n for i in 1:j]
        end
        _ppn(c) = (isqrt(8 * length(c) + 1) - 1) ÷ 2      # recover n from the packed length n(n+1)/2
        for uplo in ('L', 'U')
            addh(
                "pptrf$uplo", s -> _ppd(s, uplo),
                c -> (_ppref!(uplo, _ppn(c), c); c[1]),
                c -> (PureBLAS.pptrf!(c; uplo = uplo); c[1]); sizes = _cap(LPSZ, 2048)
            )
        end
    end
    return l1, l2, l3, lp
end

# ── Complex (ComplexF64) surface: the M5 complex-SIMD work. Same methodology; separate plot family so the
# real (Float64) plots stay clean. L1/L2 violins, L3 trend. Oracle = OpenBLAS/MKL complex BLAS. ────────
function run_cmplx_benchmarks()
    T = ComplexF64; TC = Char(67)
    ca = one(T); cb = zero(T)
    cl1 = OpData[]
    let
        for (nm, ob, pb) in (
                (
                    "zaxpy", (c, m) -> (
                        for _ in 1:m
                            B.axpy!(1.7 + 0.3im, c[1], c[2])
                        end; real(c[2][1])
                    ),
                    (c, m) -> (
                        for _ in 1:m
                            PureBLAS.axpy!(c[2], 1.7 + 0.3im, c[1])
                        end; real(c[2][1])
                    ),
                ),
                (
                    "zdotc", (c, m) -> (
                        s = zero(T); for _ in 1:m
                            s += B.dotc(c[1], c[2])
                        end; real(s)
                    ),
                    (c, m) -> (
                        s = zero(T); for _ in 1:m
                            s += PureBLAS.dot(c[1], c[2])
                        end; real(s)
                    ),
                ),
                (
                    "zscal", (c, m) -> (
                        for _ in 1:m
                            B.scal!(1.0000001 + 0im, c[1])
                        end; real(c[1][1])
                    ),
                    (c, m) -> (
                        for _ in 1:m
                            PureBLAS.scal!(1.0000001 + 0im, c[1])
                        end; real(c[1][1])
                    ),
                ),
                (
                    "dznrm2", (c, m) -> (
                        s = 0.0; for _ in 1:m
                            s += B.nrm2(c[1])
                        end; s
                    ),
                    (c, m) -> (
                        s = 0.0; for _ in 1:m
                            s += PureBLAS.nrm2(c[1])
                        end; s
                    ),
                ),
                (
                    "dzasum", (c, m) -> (
                        s = 0.0; for _ in 1:m
                            s += B.asum(c[1])
                        end; s
                    ),
                    (c, m) -> (
                        s = 0.0; for _ in 1:m
                            s += PureBLAS.asum(c[1])
                        end; s
                    ),
                ),
                (
                    "zdotu", (c, m) -> (
                        s = zero(T); for _ in 1:m
                            s += B.dotu(c[1], c[2])
                        end; real(s)
                    ),
                    (c, m) -> (
                        s = zero(T); for _ in 1:m
                            s += PureBLAS.dotu(c[1], c[2])
                        end; real(s)
                    ),
                ),
                (
                    "izamax", (c, m) -> (
                        s = 0; for _ in 1:m
                            s += B.iamax(c[1])
                        end; s
                    ),
                    (c, m) -> (
                        s = 0; for _ in 1:m
                            s += PureBLAS.iamax(c[1])
                        end; s
                    ),
                ),
            )
            _meas!(cl1, "CL1", nm, () -> sweep(s -> (randn(T, s), randn(T, s)), _sizes(L1SZ), ob, pb, _L1REP))
        end
    end
    cl2 = OpData[]
    let
        sq(s) = (randn(T, s, s), randn(T, s), randn(T, s))
        herm(s) = (
            A = randn(T, s, s); A = A + A'; for i in 1:s
                A[i, i] = real(A[i, i])
            end; (A, randn(T, s), randn(T, s))
        )
        tri(s) = (
            A = randn(T, s, s); for i in 1:s
                A[i, i] = 1 + abs(A[i, i])
            end; (A, randn(T, s), randn(T, s))
        )
        cpk(s) = (randn(T, (s * (s + 1)) ÷ 2), randn(T, s), randn(T, s))          # Hermitian packed (hpmv)
        cbd(s) = (k = 16; (randn(T, 2k + 1, s), randn(T, s), randn(T, s), k))      # general banded (gbmv)
        csbd(s) = (k = 16; (randn(T, k + 1, s), randn(T, s), randn(T, s), k))      # Hermitian banded (hbmv)
        add(nm, mk, ob, pb) = _meas!(cl2, "CL2", nm, () -> sweep(mk, _sizes(L2SZ), ob, pb, _L2REP))
        add(
            "zgemvN", sq, (c, m) -> (
                for _ in 1:m
                    B.gemv!(TN, ca, c[1], c[2], cb, c[3])
                end; real(c[3][1])
            ),
            (c, m) -> (
                for _ in 1:m
                    PureBLAS.gemv!(c[3], c[1], c[2]; alpha = ca, beta = cb)
                end; real(c[3][1])
            )
        )
        add(
            "zgemvT", sq, (c, m) -> (
                for _ in 1:m
                    B.gemv!(TT, ca, c[1], c[2], cb, c[3])
                end; real(c[3][1])
            ),
            (c, m) -> (
                for _ in 1:m
                    PureBLAS.gemv!(c[3], c[1], c[2]; alpha = ca, beta = cb, trans = TT)
                end; real(c[3][1])
            )
        )
        add(
            "zgemvC", sq, (c, m) -> (
                for _ in 1:m
                    B.gemv!(TC, ca, c[1], c[2], cb, c[3])
                end; real(c[3][1])
            ),
            (c, m) -> (
                for _ in 1:m
                    PureBLAS.gemv!(c[3], c[1], c[2]; alpha = ca, beta = cb, trans = TC)
                end; real(c[3][1])
            )
        )
        add(
            "zgeru", sq, (c, m) -> (
                for _ in 1:m
                    B.geru!(ca, c[2], c[3], c[1])
                end; real(c[1][1])
            ),
            (c, m) -> (
                for _ in 1:m
                    PureBLAS.ger!(ca, c[2], c[3], c[1])
                end; real(c[1][1])
            )
        )
        add(
            "zhemv", herm, (c, m) -> (
                for _ in 1:m
                    B.hemv!(U, ca, c[1], c[2], cb, c[3])
                end; real(c[3][1])
            ),
            (c, m) -> (
                for _ in 1:m
                    PureBLAS.hemv!(c[3], c[1], c[2]; uplo = U, alpha = ca, beta = cb)
                end; real(c[3][1])
            )
        )
        add(
            "ztrmv", tri, (c, m) -> (
                for _ in 1:m
                    copyto!(c[3], c[2]); B.trmv!(U, TN, TN, c[1], c[3])
                end; real(c[3][1])
            ),
            (c, m) -> (
                for _ in 1:m
                    copyto!(c[3], c[2]); PureBLAS.trmv!(c[1], c[3]; uplo = U)
                end; real(c[3][1])
            )
        )
        add(
            "ztrsv", tri, (c, m) -> (
                for _ in 1:m
                    copyto!(c[3], c[2]); B.trsv!(U, TN, TN, c[1], c[3])
                end; real(c[3][1])
            ),
            (c, m) -> (
                for _ in 1:m
                    copyto!(c[3], c[2]); PureBLAS.trsv!(c[1], c[3]; uplo = U)
                end; real(c[3][1])
            )
        )
        add(
            "zhpmv", cpk, (c, m) -> (
                for _ in 1:m
                    B.hpmv!(U, ca, c[1], c[2], cb, c[3])
                end; real(c[3][1])
            ),
            (c, m) -> (
                for _ in 1:m
                    PureBLAS.hpmv!(c[3], c[1], c[2]; uplo = U, alpha = ca, beta = cb)
                end; real(c[3][1])
            )
        )
        add(
            "zgbmvN", cbd, (c, m) -> (
                for _ in 1:m
                    B.gbmv!(TN, length(c[2]), c[4], c[4], ca, c[1], c[2], cb, c[3])
                end; real(c[3][1])
            ),
            (c, m) -> (
                n = length(c[2]); for _ in 1:m
                    PureBLAS.gbmv!(c[3], c[1], c[2], n, c[4], c[4]; trans = TN, alpha = ca, beta = cb)
                end; real(c[3][1])
            )
        )
        add(
            "zhbmv", csbd, (c, m) -> (
                for _ in 1:m
                    B.hbmv!(U, c[4], ca, c[1], c[2], cb, c[3])
                end; real(c[3][1])
            ),
            (c, m) -> (
                for _ in 1:m
                    PureBLAS.hbmv!(c[3], c[1], c[2]; uplo = U, alpha = ca, beta = cb)
                end; real(c[3][1])
            )
        )
    end
    cl3 = OpData[]
    let
        NN = TN; LT = Char(76); RT = Char(82); UP = U; TC = Char(67)
        ctri(s) = (
            A = randn(T, s, s) ./ (2s); for i in 1:s
                A[i, i] = 1 + abs(A[i, i])
            end; A
        )
        cherm(s) = (
            A = randn(T, s, s); A = A + A'; for i in 1:s
                A[i, i] = real(A[i, i])
            end; A
        )
        addh(nm, mk, ob, pb) = _meas!(cl3, "CL3", nm, () -> sweep_heavy(mk, ob, pb, _sizes(_cap(L3SZ, 2048))))  # complex 4096 is a ~10min sink for little signal → cap at 2048 (real L3 keeps 4096)
        addh(
            "zgemm", s -> (randn(T, s, s), randn(T, s, s), zeros(T, s, s)),
            c -> (B.gemm!(NN, NN, ca, c[1], c[2], cb, c[3]); real(c[3][1])),
            c -> (PureBLAS.gemm!(c[3], c[1], c[2]); real(c[3][1]))
        )
        addh(
            "zhemm", s -> (cherm(s), randn(T, s, s), zeros(T, s, s)),
            c -> (B.hemm!(LT, UP, ca, c[1], c[2], cb, c[3]); real(c[3][1])),
            c -> (PureBLAS.hemm!(c[3], c[1], c[2]; side = LT, uplo = UP, alpha = ca, beta = cb); real(c[3][1]))
        )
        addh(
            "zsymm", s -> (randn(T, s, s), randn(T, s, s), zeros(T, s, s)),
            c -> (B.symm!(LT, UP, ca, c[1], c[2], cb, c[3]); real(c[3][1])),
            c -> (PureBLAS.symm!(c[3], c[1], c[2]; side = LT, uplo = UP, alpha = ca, beta = cb); real(c[3][1]))
        )
        addh(
            "zsyrk", s -> (randn(T, s, s), zeros(T, s, s)),
            c -> (B.syrk!(UP, NN, ca, c[1], cb, c[2]); real(c[2][1])),
            c -> (PureBLAS.syrk!(c[2], c[1]; uplo = UP, trans = NN, alpha = ca, beta = cb); real(c[2][1]))
        )
        addh(
            "zherk", s -> (randn(T, s, s), zeros(T, s, s)),
            c -> (B.herk!(UP, NN, 1.0, c[1], 0.0, c[2]); real(c[2][1])),
            c -> (PureBLAS.herk!(c[2], c[1]; uplo = UP, trans = NN, alpha = 1.0, beta = 0.0); real(c[2][1]))
        )
        addh(
            "zher2k", s -> (randn(T, s, s), randn(T, s, s), zeros(T, s, s)),   # were UNPLOTTED (like side-R)
            c -> (B.her2k!(UP, NN, ca, c[1], c[2], 0.0, c[3]); real(c[3][1])),
            c -> (PureBLAS.her2k!(c[3], c[1], c[2]; uplo = UP, trans = NN, alpha = ca, beta = 0.0); real(c[3][1]))
        )
        addh(
            "zsyr2k", s -> (randn(T, s, s), randn(T, s, s), zeros(T, s, s)),
            c -> (B.syr2k!(UP, NN, ca, c[1], c[2], cb, c[3]); real(c[3][1])),
            c -> (PureBLAS.syr2k!(c[3], c[1], c[2]; uplo = UP, trans = NN, alpha = ca, beta = cb); real(c[3][1]))
        )
        addh(
            "ztrmm", s -> (ctri(s), randn(T, s, s)),
            c -> (B.trmm!(LT, UP, NN, NN, ca, c[1], c[2]); real(c[2][1])),
            c -> (PureBLAS.trmm!(c[2], c[1]; side = LT, uplo = UP); real(c[2][1]))
        )
        addh(
            "ztrsm", s -> (ctri(s), randn(T, s, s)),
            c -> (B.trsm!(LT, UP, NN, NN, ca, c[1], c[2]); real(c[2][1])),
            c -> (PureBLAS.trsm!(c[2], c[1]; side = LT, uplo = UP); real(c[2][1]))
        )
        addh(
            "ztrmmR", s -> (ctri(s), randn(T, s, s)),     # side-R: plots measured only side-L → the 0.24
            c -> (B.trmm!(RT, UP, NN, NN, ca, c[1], c[2]); real(c[2][1])),   # side-R routing bug went unseen
            c -> (PureBLAS.trmm!(c[2], c[1]; side = RT, uplo = UP); real(c[2][1]))
        )
        addh(
            "ztrsmR", s -> (ctri(s), randn(T, s, s)),
            c -> (B.trsm!(RT, UP, NN, NN, ca, c[1], c[2]); real(c[2][1])),
            c -> (PureBLAS.trsm!(c[2], c[1]; side = RT, uplo = UP); real(c[2][1]))
        )
    end
    # ── Complex LAPACK (zpotrf/zgetrf/zgeqrf/zgesvd; destructive → fresh input per round). Mirrors the real
    # `lp` group. zgesvd compares VALUES-ONLY (gesdd 'N' vs PB want_vectors=false) — complex singular VECTORS
    # aren't implemented yet, so this is the honest fair fight for what ships. ─────────────────────────────
    clp = OpData[]
    let
        LP = Char(76); UP = Char(85)  # 'L' / 'U'
        addh(nm, mk, ob, pb; sizes = LPSZ) = _meas!(clp, "CLP", nm, () -> sweep_heavy(mk, ob, pb, _sizes(_cap(sizes, 2048)); samples = 40))  # cap complex LAPACK at 2048 (zgesvd's 1024 cap survives via nested _cap)
        addh(
            "zpotrf", s -> _hpd(T, s),
            c -> (LinearAlgebra.LAPACK.potrf!(LP, c); real(c[1, 1])),
            c -> (PureBLAS.potrf!(c; uplo = LP); real(c[1, 1]))
        )
        # UPPER is a genuinely different code path (Lever C: conj-transpose → _cpotrf_lower! →
        # conj-transpose back), not a mirror of lower — and it was UNGATED until now, which is exactly how
        # it came to sit at 0.92–0.94 vs AOCL at n=512/1024 unnoticed. "zpotrf PASSES" previously meant
        # only that the lower path passed.
        addh(
            "zpotrfU", s -> _hpd(T, s),
            c -> (LinearAlgebra.LAPACK.potrf!(UP, c); real(c[1, 1])),
            c -> (PureBLAS.potrf!(c; uplo = UP); real(c[1, 1]))
        )
        addh(
            "zgeqrf", s -> randn(T, s, s),
            c -> (LinearAlgebra.LAPACK.geqrf!(c); real(c[1, 1])),
            c -> (PureBLAS.geqrf!(c); real(c[1, 1]))
        )
        addh(
            "zgetrf", s -> randn(T, s, s),
            c -> (LinearAlgebra.LAPACK.getrf!(c); real(c[1, 1])),
            c -> (PureBLAS.getrf!(c); real(c[1, 1]))
        )
        # zgesvd now on the BLOCKED complex bidiag (zlabrd panels + gemm trailing) → gates; capped at 2048
        # (the group cap) like the other complex LAPACK ops.
        addh(
            "zgesvd", s -> randn(T, s, s),
            c -> (LinearAlgebra.LAPACK.gesdd!(Char(78), c); real(c[1, 1])),   # 'N' — singular values only
            c -> (PureBLAS.gesvd!(c; want_vectors = false); 0.0)
        )
        # Hermitian eigensolver (zheevd): blocked hetrd + D&C stedc + blocked unmtr. 'V' eigenpairs (Julia's
        # default eigen(Hermitian)) and 'N' eigenvalues. syevd! on a complex matrix dispatches to zheevd.
        _herm(s) = (A = randn(T, s, s); A .+ A')
        addh(
            "zheev", _herm,
            c -> (LinearAlgebra.LAPACK.syevd!(Char(86), LP, c); real(c[1, 1])),   # 'V','L'
            c -> (PureBLAS._heev!('V', 'L', c); real(c[1, 1]))
        )
        addh(
            "zheevN", _herm,
            c -> (LinearAlgebra.LAPACK.syevd!(Char(78), LP, c); real(c[1, 1])),   # 'N','L'
            c -> (PureBLAS._heev!('N', 'L', c); real(c[1, 1]))
        )
    end
    return cl1, cl2, cl3, clp
end

# ── cache: one line per op  «level⟶TAB⟶name⟶TAB⟶ s1=r,r,…;s2=r,r,… » ─────────────────────────────
const CACHE = joinpath(@__DIR__, "plots_data_$(SLUG)_$(gethostname())$(_LITE ? "_lite" : "").txt")   # v3: ONE cache per host holds EVERY arm (no _aocl/_mkl split); the reference suffix now applies only to the rendered views
# ── MACHINE-STATE PROVENANCE: `anchor=` and `freq=` ────────────────────────────────────────────────
# WHY. A cached reference is compared against a PB arm measured in a DIFFERENT process, possibly days
# later. Same-run ratios cancel machine state — thermal drift, clock, page placement — because both arms
# run seconds apart; a CACHED ratio cancels nothing. That is the mechanism behind cross-run drift, which
# read axpy n=1e6 at 0.960 cached against 0.983 same-run on identical code, and heat is one of its
# inputs: a hot afternoon makes PB look slower against a reference measured on a cool morning, and
# nothing in the comparison can distinguish that from a regression.
#
# So every run stamps two machine-state fields, and neither changes the record format:
#   * `anchor=` — the median time of a FIXED, CODE-INVARIANT workload (Base `sum(abs2, ·)` over a
#     constant L2-resident array). It does not touch PureBLAS, so it moves only with the machine, not
#     with the kernel under test. A later PB-only run can scale a cached reference by the ratio of
#     anchors to remove the machine-state difference — and can at minimum REFUSE to compare when the
#     anchors disagree by more than the effect being chased.
#   * `freq=` — the achieved clock under load (max over cores; the pinned core is the busy one). The
#     frequency methodology requires base-clock-locked runs, and until now a throttled run was
#     indistinguishable from a clean one after the fact.
# Chairmarks + median, like every other timing here (this file is scanned by test/estimator_lint.jl).
const _ANCHOR_N = 1 << 17                       # 128K Float64 = 1 MB: L2-resident on every fleet box
function _anchor_secs()
    a = fill(1.0000001, _ANCHOR_N)
    b = @be sum(abs2, a) evals = 1 samples = 64 seconds = 0.5
    return median(Float64[s.time for s in b.samples])
end
"Achieved kHz under load: max over cores, sampled right after the anchor while the core is still hot."
function _achieved_khz()
    best = 0
    for d in readdir("/sys/devices/system/cpu"; join = true)
        f = joinpath(d, "cpufreq", "scaling_cur_freq")
        isfile(f) || continue
        v = tryparse(Int, strip(read(f, String)))
        isnothing(v) || (best = max(best, v))
    end
    return best
end

# CONTENTION GUARD — refuse to start a gate sweep on a box that is already busy.
#
# The gate is a PB/OB ratio measured in two adjacent windows on one core. A foreign job saturating
# another core is not neutral: it contends for the shared L3 and the memory controller, which is
# exactly where the bandwidth-bound BLAS-1/2 cells live, and it does so unevenly across the two
# windows. On 2026-08-05 a stray `julia-1.13 --project=@plots` was found mid-sweep on neuromancer —
# the run had to be discarded after the fact. A three-hour sweep deserves a one-second check first.
#
# `ps -eo` and NOT `pgrep julia | wc -l`: the count form self-matches (this process, and any wrapper
# whose cmdline contains "julia"), which is the deadlock recorded in `bash-idle-loop-julia-selfmatch`.
# Matching on %CPU instead of on a name is both stricter and immune to that — an idle sibling shell is
# invisible, a busy foreign job of any name is not.
#
# Reports only, unless something is genuinely eating a core: then it stops, since the alternative is
# discovering it in the provenance afterwards. Override with `force-busy` when the contention is known
# and accepted (e.g. deliberately benching two µarchs at once on different sockets).
#
# CHECKED AT BOTH ENDS, and the second check is the one that earned its keep. A start-only guard sees
# a quiet box and then has nothing to say about the next three hours. On 2026-08-06 a `group=CL2`
# screen started clean and an advisory agent began running its own checks on the same box mid-sweep;
# the start guard passed, the run completed, and the numbers went into the cache with nothing marking
# them. `busy=` in the cache header is the fix: the exit check cannot un-contend a finished run, but
# it can stop the result from being read later as if the box had been quiet.
const _BUSY_PCPU = 25.0
"Foreign processes at or above _BUSY_PCPU, as (pid, %cpu, cmdline). Excludes us and our children."
function _busy_procs()
    out = try
        read(`ps -eo pid,ppid,pcpu,args --no-headers`, String)
    catch
        return nothing                               # ps unavailable — unknown, not "clear"
    end
    me = getpid()
    busy = Tuple{Int, Float64, String}[]
    for ln in eachsplit(out, '\n')
        f = split(strip(ln); limit = 4)
        length(f) == 4 || continue
        pid = tryparse(Int, f[1]); ppid = tryparse(Int, f[2]); pc = tryparse(Float64, f[3])
        (isnothing(pid) || isnothing(ppid) || isnothing(pc)) && continue
        (pid == me || ppid == me) && continue        # us, and anything we spawned
        pc >= _BUSY_PCPU && push!(busy, (pid, pc, String(f[4])))
    end
    sort!(busy; by = x -> -x[2])
    return busy
end
_busy_msg(busy) = join(("  pid=$(p)  $(c)% CPU  $(first(a, 100))" for (p, c, a) in busy), "\n")

function _contention_check()
    busy = _busy_procs()
    isnothing(busy) && return println("contention check: skipped (ps unavailable)")
    isempty(busy) && return println("contention check: clear (no foreign process ≥ $(_BUSY_PCPU)% CPU)")
    "force-busy" in ARGS && return println("contention check: BUSY but force-busy given:\n", _busy_msg(busy))
    return error("REFUSING to benchmark: $(length(busy)) foreign process(es) ≥ $(_BUSY_PCPU)% CPU on \
        $(gethostname()). A contended L3/memory controller skews the PB and reference windows \
        unequally and the run would have to be discarded.\n$(_busy_msg(busy))\nWait for the box, or pass \
        `force-busy` to measure anyway. Do NOT pattern-kill — kill only PIDs you launched.")
end

# Set by the exit check, stamped into the cache header so a contended run is self-identifying.
_BUSY_AT_EXIT = ""
function _contention_exit_check()
    busy = _busy_procs()
    (isnothing(busy) || isempty(busy)) && return nothing
    global _BUSY_AT_EXIT = join(("$(p):$(c)%" for (p, c, _) in busy), ",")
    @warn "BOX WAS CONTENDED AT END OF RUN — it was clear at the start, so this appeared during \
        measurement. Treat every cell this run touched as suspect and re-measure on a quiet box; \
        the cache header records it as `busy=`.\n$(_busy_msg(busy))"
    return nothing
end

function save_cache(path, groups)
    open(path, "w") do io
        # header stamps the methodology version (so old numbers can't silently coexist), the µarch identity
        # (slug/isa) for the multi-host plot, and full provenance: CPU model, code commit, measure time,
        # and the RESOLVED Measure-tier tuning state (`tune=`).
        # `tune=` exists because on 2026-07-30 an UNTRACKED `bench/LocalPreferences.toml` pinning
        # `ger_panel_np = 1` was found on BOTH fleet boxes. It overrode a correctly-working auto-measure
        # (wintermute wants 8, galen wants 4; 1 is the Zen5 value) and silently sandbagged every ger number
        # ever committed: removing it took wintermute n=2048 from 0.914 to 1.244 and n=4096 from 0.974 to
        # 1.408, and flipped galen's ger from FAIL to PASS vs OpenBLAS. A pin does not appear in
        # `git status`, so nothing in the cache file revealed that the run was measuring the pin rather
        # than the kernel. Stamping the resolved values makes a cache reproducible from its own header —
        # if two runs disagree, diff `tune=` first.
        ts = Libc.strftime("%Y-%m-%dT%H:%M", time())
        anc = try
            _anchor_secs()
        catch
            NaN
        end
        khz = try
            _achieved_khz()
        catch
            0
        end
        println(
            io, "#pbbench\tversion=$(_BENCH_VERSION)\tslug=$SLUG\tuarch=$(_MYUARCH)\tisa=$ISA",
            "\thost=$(gethostname())\tcpu=$(_CPUNAME)\tcommit=$(_COMMIT)\ttime=$ts\ttune=$(_tunestamp())",
            "\tanchor=$(round(anc * 1e6; digits = 3))us\tfreq=$(khz)kHz",
            isempty(_BUSY_AT_EXIT) ? "" : "\tbusy=$(_BUSY_AT_EXIT)"
        )
        # v3 record: ONE LINE PER CELL, one field per measured arm, each carrying its own timestamp and
        # commit. Per-arm provenance is what makes `arms=pb` safe to use: it rewrites only the pb field,
        # so the reference fields still say when they were actually measured and the table reports their
        # age rather than implying they are as fresh as the run that produced the page.
        #   <lvl> <op> <size> pb|<iso>|<commit>|t1,..,t48  openblas|<iso>|<commit>|t1,..  aocl|...
        # Times are SECONDS (the quantiles of that arm's sample times). Ratios are derived, never stored.
        for (lvl, d) in groups, (nm, op) in d, (s, cell) in op
            # FIVE fields now: the anchor sits between commit and the times. The cache VERSION is
            # deliberately NOT bumped — a bump refuses every existing cache, and re-measuring the
            # OpenBLAS/AOCL arms is exactly what the reference cache exists to prevent. Readers accept
            # 4-field (pre-anchor) and 5-field records side by side in one file; the csv is always the
            # LAST field, so every reader splits and takes `p[end]` rather than `limit = 4`.
            fields = [
                "$(a)|$(rec.time)|$(rec.commit)|$(isnan(rec.anchor) ? "" : round(rec.anchor * 1e6; digits = 3))|$(join(rec.q, ","))"
                    for (a, rec) in sort!(collect(cell); by = first)
            ]
            println(io, lvl, "\t", nm, "\t", s, "\t", join(fields, "\t"))
        end
    end
    return println("cached arm times → $path")
end
# Returns (groups, meta::NamedTuple). meta carries version/slug/isa/host from the header (µarch identity
# for the multi-host overlay). Refuses a cache from an older methodology version (forces re-measure).
function load_cache(path)
    g = Dict{String, Vector{OpData}}()
    meta = (version = 1, slug = "?", uarch = "?", isa = "?", host = "?", cpu = "?", commit = "?", time = "?")  # legacy ⇒ v1
    for ln in eachline(path)
        isempty(strip(ln)) && continue
        if startswith(ln, "#pbbench")
            kv = Dict(String(p[1]) => String(p[2]) for p in (split(x, "=") for x in split(ln, "\t")[2:end]) if length(p) == 2)
            meta = (
                version = parse(Int, get(kv, "version", "1")), slug = get(kv, "slug", "?"),
                uarch = get(kv, "uarch", "?"), isa = get(kv, "isa", "?"), host = get(kv, "host", "?"),
                cpu = get(kv, "cpu", "?"), commit = get(kv, "commit", "?"), time = get(kv, "time", "?"),
            )
            continue
        end
        parts = split(ln, "\t")
        length(parts) >= 4 || continue                       # v2 line shape ⇒ let the version check below speak
        lvl, nm, ssz = String(parts[1]), String(parts[2]), parse(Int, parts[3])
        cell = CellData()
        for f in parts[4:end]
            # 4-field (pre-anchor) and 5-field records coexist in one file — see the writer note. The
            # times are ALWAYS last, so index from both ends rather than assuming a field count.
            p = split(f, "|")
            a, tstamp, cmt, csv = p[1], p[2], p[3], p[end]
            anc = length(p) >= 5 ? something(tryparse(Float64, p[4]), NaN) * 1e-6 : NaN
            cell[String(a)] = ArmRec(String(tstamp), String(cmt), anc, parse.(Float64, split(csv, ",")))
        end
        ops = get!(g, lvl, OpData[])
        i = findfirst(p -> p.first == nm, ops)
        isnothing(i) ? push!(ops, nm => [(ssz, cell)]) : push!(ops[i].second, (ssz, cell))
    end
    meta.version == _BENCH_VERSION || error(
        "cache $path is methodology v$(meta.version); this is v$(_BENCH_VERSION) (per-arm TIMES with " *
            "per-arm provenance). v2 stored only the quotient, so it cannot be converted — the arm times " *
            "were never written. Re-measure with `bench`."
    )
    return g, meta
end


# ── Cross-µarch panel grid (the redesign): one SVG per group, one PANEL per op, each overlaying the
# fleet's µarchs as ratio-vs-size lines + q10–q90 bands. Size is always the x-axis (cache transitions show
# as steps); the 3 µarchs share a panel so cross-machine comparison is direct; no panel holds >3 lines so
# the old 11-line/8-colour collision is gone. Fixed colour per µarch, keyed on the cache's stamped slug. ──
const _UARCH = Dict(
    "avx512" => ("#1f77b4", "Zen4 · AVX-512"), "zen5" => ("#2ca02c", "Zen5 · AVX-512"),
    "avx2" => ("#d62728", "Zen3 · AVX2")
)
# AOCL/MKL caches stamp slug=<µarch>_<refbk> (e.g. avx512_aocl); _UARCH is keyed on the BARE µarch slug,
# so strip the refbk suffix before the color/label lookup — else every AOCL series fell to the grey fallback.
_baseslug(slug) = replace(slug, r"_(aocl|mkl)$" => "")
_ucolor(slug) = get(_UARCH, _baseslug(slug), ("#888888", slug))[1]
# Label from the AUTHORITATIVE stamped µarch (measuring machine's own CpuId), not re-derived here. Old caches
# (no uarch= field, meta.uarch=="?") fall back to the slug→label map so pre-fix caches still render.
_ulabel(meta) = meta.uarch != "?" ? "$(meta.uarch) · $(meta.isa)" :
    get(_UARCH, _baseslug(meta.slug), ("#888888", meta.isa))[2]

# Load every fleet cache (plots_data_<host>.txt) → [(meta, groups), …]. In lite mode loads only *_lite; in
# full mode only full caches. Skips MKL. Refuses stale-version caches via load_cache.
function load_fleet()
    fleet = Tuple{NamedTuple, Dict{String, Vector{OpData}}}[]
    for f in sort(readdir(@__DIR__))
        (startswith(f, "plots_data_") && endswith(f, ".txt")) || continue
        # v3: no reference filter. One cache per host carries every arm, and `_series` picks the arm for
        # the view being rendered — so the "never mix baselines" rule is now enforced by construction
        # (each ratio divides two arms measured in the SAME round) rather than by filename discipline.
        (occursin("_aocl", f) || occursin("_mkl", f)) && continue   # skip leftover v2 split caches
        occursin("_lite", f) == _LITE || continue
        try                                                    # a stale/foreign/half-written cache must NOT
            g, meta = load_cache(joinpath(@__DIR__, f))        # abort the whole fleet render — skip it loudly
            push!(fleet, (meta, g))
        catch e
            @warn "skipping cache $f (stale version or unreadable)" exception = (e, catch_backtrace())
        end
    end
    slugs = [m.slug for (m, _) in fleet]                        # duplicate µarch ⇒ lines overlap + mislabel
    for s in unique(slugs)
        count(==(s), slugs) > 1 && @warn "duplicate µarch slug '$s' across caches — pass slug=/isa= to disambiguate"
    end
    return fleet
end

_opsin(fleet, gk) = (
    ops = String[]; for (_, g) in fleet, (nm, _) in get(g, gk, OpData[])
        (nm in ops) || push!(ops, nm)
    end; ops
)
# THE one place a ratio is formed for output. Everything downstream (gen_table, svg_panels, gatestat)
# consumes `[(size, ratios)]` exactly as it did under v2, so deriving here — rather than at each call
# site — is what keeps tables and plots from drifting apart in how they define the number.
# Cells missing either arm are dropped: `arms=pb` on a fresh cache legitimately has no reference yet,
# and a half-populated series must not be silently plotted as if it were complete.
function _series(g, gk, op, ref::AbstractString = REFBK)
    ops = get(g, gk, OpData[])
    i = findfirst(p -> p.first == op, ops)
    isnothing(i) && return nothing
    out = Tuple{Int, Vector{Float64}}[]
    for (s, cell) in ops[i].second
        (haskey(cell, ref) && haskey(cell, _ARM_PB)) || continue
        push!(out, (s, _ratio(cell[ref].q, cell[_ARM_PB].q)))
    end
    return out
end

# Age of the reference arms behind a rendered view, so a page can say how old its baseline is instead of
# implying it matches the run that produced it. Returns (oldest, newest) ISO stamps over the cells used.
function _ref_age(g, ref::AbstractString = REFBK)
    stamps = String[]
    for (_, ops) in g, (_, sizes) in ops, (_, cell) in sizes
        haskey(cell, ref) && push!(stamps, cell[ref].time)
    end
    isempty(stamps) && return ("–", "–")
    # `stamps` holds ISO DATE STRINGS, not timings — oldest/newest reference-arm stamp for the
    # provenance line. Renamed from `ts` so it cannot read as a timing vector to either a human or the
    # lint; the lint has no exemption mechanism, so ambiguous names have to be fixed, not annotated.
    return (minimum(stamps), maximum(stamps))
end

function svg_panels(path, title, fleet, gk)
    ops = _opsin(fleet, gk); isempty(ops) && return
    ncol = min(4, length(ops)); nrow = cld(length(ops), ncol)
    pw = 210; ph = 138; ml = 46; mt = 60; gx = 20; gy = 34; pad = 16
    W = ml + ncol * pw + (ncol - 1) * gx + pad
    H = mt + nrow * (ph + gy) + pad
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="sans-serif">""")
    println(io, """<rect width="$W" height="$H" fill="white"/>""")
    println(io, """<text x="$(W / 2)" y="26" text-anchor="middle" font-size="17" font-weight="bold">$title</text>""")
    lx = ml
    for (meta, _) in fleet   # legend
        col = _ucolor(meta.slug); lab = _ulabel(meta)
        println(io, """<line x1="$lx" y1="42" x2="$(lx + 22)" y2="42" stroke="$col" stroke-width="3"/>""")
        println(io, """<text x="$(lx + 27)" y="46" font-size="12">$lab</text>""")
        lx += 27 + 7 * length(lab) + 26
    end
    for (k, op) in enumerate(ops)
        px = ml + ((k - 1) % ncol) * (pw + gx); py = mt + ((k - 1) ÷ ncol) * (ph + gy)
        series = Tuple{String, Vector{Tuple{Int, Vector{Float64}}}}[]; allsz = Int[]
        for (meta, g) in fleet
            ps = _series(g, gk, op); (isnothing(ps) || isempty(ps)) && continue
            push!(series, (meta.slug, ps)); for (s, _) in ps
                (s in allsz) || push!(allsz, s)
            end
        end
        (isempty(series) || isempty(allsz)) && continue
        sort!(allsz)
        xlo = log2(minimum(allsz)); xsp = max(log2(maximum(allsz)) - xlo, 1.0e-9)
        # y-range from the band extremes (q10/q90), not just medians, so noisy bands don't saturate flat
        ext = Float64[]; for (_, ps) in series, (_, v) in ps
            push!(ext, quantile(v, 0.1), median(v), quantile(v, 0.9))
        end
        yhi = max(1.6, 1.08 * maximum(ext)); ylo = min(0.5, 0.93 * minimum(ext)); L = log
        xof(s) = px + pw * (log2(s) - xlo) / xsp
        yof(r) = py + ph * (1 - (L(clamp(r, ylo, yhi)) - L(ylo)) / (L(yhi) - L(ylo)))
        println(io, """<rect x="$px" y="$py" width="$pw" height="$ph" fill="none" stroke="#e2e2e2"/>""")
        for (rr, cc, da) in ((1.0, "#d33", """ stroke-dasharray="4 3\""""),)   # gate = parity = 1.0×
            (rr < ylo || rr > yhi) && continue
            println(io, """<line x1="$px" y1="$(yof(rr))" x2="$(px + pw)" y2="$(yof(rr))" stroke="$cc"$da/>""")
        end
        for r in unique(round.([ylo, 1.0, yhi], digits = 2))
            (r < ylo || r > yhi) && continue
            println(io, """<text x="$(px - 4)" y="$(yof(r) + 3)" text-anchor="end" font-size="9" fill="#999">$(r)×</text>""")
        end
        for (slug, ps) in series
            col = _ucolor(slug)
            bhi = ["$(round(xof(s), digits = 1)),$(round(yof(quantile(v, 0.9)), digits = 1))" for (s, v) in ps]
            blo = ["$(round(xof(s), digits = 1)),$(round(yof(quantile(v, 0.1)), digits = 1))" for (s, v) in reverse(ps)]
            println(io, """<polygon points="$(join(vcat(bhi, blo), " "))" fill="$col" opacity="0.11"/>""")
            ln = ["$(round(xof(s), digits = 1)),$(round(yof(median(v)), digits = 1))" for (s, v) in ps]
            println(io, """<polyline points="$(join(ln, " "))" fill="none" stroke="$col" stroke-width="1.6"/>""")
            for (s, v) in ps
                println(io, """<circle cx="$(round(xof(s), digits = 1))" cy="$(round(yof(median(v)), digits = 1))" r="2" fill="$col"/>""")
            end
        end
        println(io, """<text x="$(px + pw / 2)" y="$(py - 5)" text-anchor="middle" font-size="12" font-weight="bold">$op</text>""")
        # x-axis: a tick + label at every measured size (≥1024 abbreviated as k so they fit the narrow panel)
        for s in allsz
            x = round(xof(s), digits = 1); lbl = s >= 1024 ? "$(s ÷ 1024)k" : "$s"
            println(io, """<line x1="$x" y1="$py" x2="$x" y2="$(py + ph)" stroke="#f4f4f4"/>""")
            println(io, """<text x="$x" y="$(py + ph + 11)" text-anchor="middle" font-size="8" fill="#999">$lbl</text>""")
        end
    end
    println(io, "</svg>"); write(path, String(take!(io)))
    return println("wrote $path")
end

# Drift-proof numeric companion to the hand-annotated narrative table: median (worst-cell) per op per µarch.
function gen_table(fleet, gkeys)
    io = IOBuffer()
    println(io, "| op | ", join((_ulabel(m) for (m, _) in fleet), " | "), " |")
    println(io, "|---|", repeat("---|", length(fleet)))
    for gk in gkeys, op in _opsin(fleet, gk)
        cells = String[]
        for (_, g) in fleet
            ps = _series(g, gk, op)
            if isnothing(ps) || isempty(ps)
                push!(cells, "–")
            else
                med, mn = gatestat(ps); push!(cells, @sprintf("%.2f (%.2f)", med, mn))
            end
        end
        println(io, "| `$op` | ", join(cells, " | "), " |")
    end
    return String(take!(io))
end

# ── measure (and cache) or load from cache, then draw ────────────────────────────────────────────
if "plot" in ARGS
    isfile(CACHE) || error("no cache at $CACHE — run without `plot` first to measure")
    g, _meta = load_cache(CACHE); println("loaded cached data ← $CACHE")
elseif !("bench" in ARGS) && isfile(CACHE)
    g, _meta = load_cache(CACHE); println("loaded cached data ← $CACHE  (pass `bench` to re-measure)")
else
    _contention_check()
    l1, l2, l3, lp = run_benchmarks()
    cl1, cl2, cl3, clp = run_cmplx_benchmarks()
    _contention_exit_check()        # before save_cache — it stamps `busy=` into the header
    measured = Dict("L1" => l1, "L2" => l2, "L3" => l3, "LP" => lp, "CL1" => cl1, "CL2" => cl2, "CL3" => cl3, "CLP" => clp)
    subset = !isnothing(_SELOP) || !isnothing(_SELGRP)
    if subset
        # subset re-measure: MERGE the measured op(s) into the existing (v2) cache, leaving the rest intact.
        isfile(CACHE) || error("subset re-measure (op=/group=) needs an existing full cache at $CACHE — run a full `bench` first")
        g, meta = load_cache(CACHE)   # load_cache refuses a non-v2 cache
        meta.slug == SLUG || error("subset slug ($SLUG) ≠ cache slug ($(meta.slug)) — merging would relabel the µarch; re-run full `bench`")
        # PER-ARM, PER-CELL merge. v2 replaced a whole op, which was fine when a run always measured both
        # sides. In v3 `arms=pb` measures only PureBLAS, so replacing the op would DELETE the reference
        # arms and silently turn the next table into "no reference data". Merge arm-by-arm instead: a cell
        # keeps every arm it had, each with the provenance of whenever that arm was last measured.
        # counted up front: `nc += 1` inside a top-level `for` would create a soft-scope local
        nc = sum((length(cells) for (_, ops) in measured for (_, cells) in ops); init = 0)
        for (lvl, ops) in measured, (nm, cells) in ops, (s, fresh) in cells
            gl = get!(g, lvl, OpData[])
            i = findfirst(p -> p.first == nm, gl)
            isnothing(i) && (push!(gl, nm => Tuple{Int, CellData}[]); i = length(gl))
            sizes = gl[i].second
            j = findfirst(t -> t[1] == s, sizes)
            if isnothing(j)
                push!(sizes, (s, copy(fresh)))
            else
                merge!(sizes[j][2], fresh)      # fresh arms win; untouched arms keep their own stamps
            end
        end
        for ops in values(g), (_, sizes) in ops
            sort!(sizes; by = first)          # a newly inserted `size=` cell must not land out of order
        end
        println("merged $nc re-measured cell(s) [arms: $(join(_ACTIVE_ARMS, ","))] into $CACHE")
    else
        # FULL run: the cache is REPLACED wholesale. That is correct only when this run measured every
        # arm — with a restricted `arms=`, saving here DELETES the reference arms for every cell in the
        # cache. The per-arm merge above protects the subset path; nothing protected this one, and on
        # 2026-08-06 a full `bench nodraw arms=pb` on neuromancer destroyed the Zen5 v3 reference arms
        # (10.9 MB -> 3.65 MB, ~2h45m of OpenBLAS+AOCL measurement) — the run looked like it succeeded
        # and the loss only surfaced when gate_gaps reported `cells=0`.
        # `arms=` is for SUBSET re-measures (op=/group=), where merging keeps the references. A full run
        # must either measure everything or be told explicitly that a pb-only cache is what you want.
        if !isnothing(_ARMS_SEL) && !issubset(_REF_ALL, _ACTIVE_ARMS) && !("force-arms" in ARGS)
            error("""
                REFUSING to overwrite $CACHE with a partial arm set.
                  full run + arms=$(join(_ACTIVE_ARMS, ",")) would DROP: $(join(setdiff(_REF_ALL, _ACTIVE_ARMS), ", "))
                A full `bench` REPLACES the cache; only op=/group= merges per arm. Either
                  • add op=<op> or group=<LVL>  (merges, keeps the reference arms), or
                  • drop `arms=` to measure every arm (~3x longer), or
                  • pass `force-arms` if a pb-only cache really is intended.""")
        end
        g = measured
    end
    save_cache(CACHE, [lvl => get(g, lvl, OpData[]) for lvl in ("L1", "L2", "L3", "LP", "CL1", "CL2", "CL3", "CLP")])
end

adir = joinpath(@__DIR__, "..", "docs", "src", "assets"); mkpath(adir)
# Draw the whole FLEET (every host cache on disk) as cross-µarch panel grids: 8 SVGs, NO per-host suffix
# (a 3-line panel IS the per-host view). One SVG per group. `nodraw` skips this (fleet boxes measure only).
fleet = _NODRAW ? [] : load_fleet()
if isempty(fleet)
    println("no fleet caches on disk to plot")
else
    L = _LITE ? "_lite" : ""; ref = REFNAME
    for (gk, base, ttl) in (
            ("L1", "l1", "BLAS-1"), ("L2", "l2", "BLAS-2"), ("L3", "l3", "BLAS-3"),
            ("LP", "lapack", "LAPACK"), ("CL1", "cl1", "Complex BLAS-1"),
            ("CL2", "cl2", "Complex BLAS-2"), ("CL3", "cl3", "Complex BLAS-3"),
            ("CLP", "clapack", "Complex LAPACK"),
        )
        svg_panels(joinpath(adir, "perf_$(base)$(REFSUF)$L.svg"), "$ttl — PureBLAS / $ref (PB/$ref ratio)", fleet, gk)
    end
    open(joinpath(@__DIR__, "gen_table$(REFSUF)$L.md"), "w") do io   # drift-proof numeric table: median (worst-cell) per op/µarch
        println(io, "_Measured (provenance):_\n")
        for (m, _) in fleet   # self-describing: CPU, code commit, measure time per µarch
            println(io, "- **$(_ulabel(m))** (`$(m.host)`) — $(m.cpu), commit `$(m.commit)`, $(m.time)")
        end
        println(io, "\n### Real\n\n", gen_table(fleet, ["L1", "L2", "L3", "LP"]))
        println(io, "\n### Complex\n\n", gen_table(fleet, ["CL1", "CL2", "CL3", "CLP"]))
    end
    println("wrote gen_table$L.md  (fleet: ", join((m.slug for (m, _) in fleet), ", "), ")")
end
# Gate summary for THIS host. v3 holds every arm in one cache, so this is the FIRST version that can
# state the project's actual rule — PB ≥ max(OpenBLAS, AOCL) — from a single run, per cell, instead of
# eyeballing two separately-measured tables. `gate` is the worst over cells of min over references,
# i.e. the margin against whichever reference is faster at each individual size.
for lvl in ("L1", "L2", "L3", "LP", "CL1", "CL2", "CL3", "CLP"), (nm, cells) in get(g, lvl, OpData[])
    per = Dict{String, Tuple{Float64, Float64}}()
    for r in _REF_ALL
        ps = _series(g, lvl, nm, r)
        (isnothing(ps) || isempty(ps)) && continue
        per[r] = gatestat(ps)
    end
    isempty(per) && continue
    # worst cell against the FASTER reference at that cell = the gate margin
    gate = Inf
    for (_, cell) in cells
        haskey(cell, _ARM_PB) || continue
        rs = [median(_ratio(cell[r].q, cell[_ARM_PB].q)) for r in _REF_ALL if haskey(cell, r)]
        isempty(rs) || (gate = min(gate, minimum(rs)))
    end
    txt = join((@sprintf("%s %.2f/%.2f", r, per[r][1], per[r][2]) for r in _REF_ALL if haskey(per, r)), "  ")
    @printf("%-3s %-8s %s   gate=%.3f %s\n", lvl, nm, txt, gate, gate >= 1.0 ? "PASS" : "FAIL")
end
isempty(_MISSING) || @warn "these ops FAILED during measurement (absent from the cache/plots): $(join(_MISSING, ", "))"

# ── CROSS-RUN MACHINE-STATE DRIFT ─────────────────────────────────────────────────────────────────
# A same-run ratio cancels machine state; a CACHED ratio cancels nothing. Under `arms=pb` the PB arm is
# fresh and the references can be days old, so a drifted machine is indistinguishable from a code
# change — and the drift here is NOT small: galen's anchor moved 13.97 → 16.46 µs (17.8%) between two
# freq-locked runs on 2026-08-07, against gate gaps of 0.3%. Now that every arm carries the anchor it
# was measured under, say so out loud. This REPORTS rather than rescales: a rescale would need the
# anchor to be a faithful proxy for whatever moved the cell, which is unproven — an honest "this
# comparison is not trustworthy to better than X%" is worth more than a wrong correction.
let worst = 0.0, worstlbl = "", nold = 0
    for lvl in keys(g), (nm, cells) in g[lvl], (s, cell) in cells
        haskey(cell, _ARM_PB) || continue
        ap = cell[_ARM_PB].anchor
        isnan(ap) && continue
        for r in _REF_ALL
            haskey(cell, r) || continue
            ar = cell[r].anchor
            isnan(ar) && (nold += 1; continue)
            d_ = abs(ap / ar - 1)
            d_ > worst && (worst = d_; worstlbl = "$lvl/$nm@$s vs $r")
        end
    end
    if nold > 0
        @warn "$nold cached reference arm(s) predate the per-arm anchor — their machine state at " *
            "capture is UNKNOWN, so no drift check is possible for them. Re-measure the references " *
            "(without `arms=pb`) to make those cells adjudicable."
    end
    if worst > 0.02
        pct = round(100 * worst; digits = 1)
        @warn "MACHINE-STATE DRIFT $pct% between this run and the cached reference (worst: " *
            "$worstlbl). Gaps smaller than this are NOT adjudicable from these numbers — re-measure " *
            "both arms in one run before believing any cell within $pct% of 1.0."
    elseif worst > 0
        @printf("machine-state drift vs cached references: %.2f%% (worst: %s)\n", 100 * worst, worstlbl)
    end
end
