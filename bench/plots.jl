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
using ForwardDiff              # DL1: the only group whose reference is a Julia implementation, not a BLAS
include(joinpath(@__DIR__, "gatecrit.jl"))   # gate_pass / GATE_MIN — THE gate criterion
include(joinpath(@__DIR__, "freqlock.jl")); using .FreqLock   # THE frequency-lock criterion
using TOML                    # _active_prefs: enumerate pins so a silent one cannot ride along
using Chairmarks: @be   # robust per-side timing (auto sample-sizing + warmup); replaces hand-rolled time_ns
# Reference BLAS: OpenBLAS (default), Intel MKL (`mkl` arg), or AMD AOCL (`aocl` arg). Each package
# LBT-forwards LinearAlgebra's BLAS+LAPACK to itself on load, so `B`/`LAPACK` below transparently measure
# against whichever is active — one code path, three baselines. AOCL (AMD-tuned BLIS + libFLAME) is a
# SEPARATE baseline from OpenBLAS: its SVGs/tables carry an `_aocl` suffix and never mix with OpenBLAS's.
# NOTE: the v3 cache holds BOTH reference arms and the render emits BOTH views every time (see `_VIEWS`),
# so `aocl` no longer selects which view is drawn — it is inert except under `mkl`.
const REFBK = "aocl" in ARGS ? "aocl" : "mkl" in ARGS ? "mkl" : "accelerate" in ARGS ? "accelerate" : "openblas"
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
# ── THE MULTI-THREADED PureBLAS ARM ────────────────────────────────────────────────────────────────
# `pb_mt` runs the SAME `work_pb` closure as `pb`, with `PureBLAS.set_num_threads(_MT_NT)` instead of 1.
# It is an ARM NAME, deliberately NOT a member of `_REF_ALL`/`_VIEWS`: adding it there would multiply
# the SVG set, break `check_view_pairing.sh`, and need new image references or the docs build hard-fails.
# As an arm it is purely additive — the per-arm merge writes one more record per cell and every existing
# artifact stays byte-identical, so NO cache version bump.
#
# ⚠ THIS IS A SCALING MEASUREMENT, NOT A GATE. A threaded PureBLAS against a single-threaded OpenBLAS is
# flattery, not parity. The gate stays `PB ≥ max(OB, AOCL)` single-threaded. A real threaded gate needs
# fresh THREADED reference arms — 6-8 h per box, three boxes — and the standing rule is that measuring a
# reference at all needs the user's explicit per-run authorisation. Everything downstream that adjudicates
# must therefore skip this arm; `_is_vendor_ref` below is the single place that decides.
const _ARM_PB_MT = "pb_mt"
# Threads for the `pb_mt` arm. 6 on every box BY DECISION, not by detection: wintermute and neuromancer
# have 6 physical cores and galen has 12, and capping galen to 6 is what keeps the three boxes
# comparable. Pin one CPU per PHYSICAL core when running it — a second thread on a core shares the same
# FMA units, so logical-core scaling measures contention, not parallelism.
#
# THE MASK NEEDS ONE CPU MORE THAN THE THREAD COUNT. `julia -t 6` runs 19 OS threads: 6 workers plus
# the GC, interactive and libuv threads. Masked to exactly 6 CPUs those runtime threads displace a
# worker mid-join, and the gemm pool's spin-then-yield join stalls behind them. Measured on Zen4,
# getrf@2048 threaded speedup: 6 CPUs gives 0.12-0.80 (bimodal against a flat serial arm), 6 physical
# cores + 1 spare CPU gives 2.32-2.35, no mask at all gives 2.08-2.29. The spare slot is where the
# runtime lands, so the workers keep their cores.
#
# CPU NUMBERING IS NOT THE SAME ON EVERY BOX — read `lscpu -p=CPU,CORE` before assuming a mask. On
# wintermute the two CPUs of a core are adjacent (0,1 = core 0), on galen and neuromancer the siblings
# are the whole second half (galen CPU 12 = core 0; neuromancer CPU 6 = core 0). So `0,2,4,6,8,10`
# selects six distinct cores on wintermute and only three, doubled up, on neuromancer.
#
#   wintermute   taskset -c 0,2,4,6,8,10,1   julia --project=bench -t 6 bench/plots.jl bench arms=pb,pb_mt
#   galen        taskset -c 6,7,8,9,10,11,18 julia --project=bench -t 6 …   # CCD1, its own 32 MiB L3
#   neuromancer  taskset -c 0,1,2,3,4,5,6    julia --project=bench -t 6 …
# `mt=` overrides for a scaling curve; it is a harness setting, not a shipped tuning knob, so no PDM tier.
const _MT_NT = let i = findfirst(a -> startswith(a, "mt="), ARGS)
    isnothing(i) ? 6 : parse(Int, ARGS[i][4:end])
end
# Is this arm a VENDOR reference — i.e. may it appear on the other side of a gate comparison? `pb` and
# `pb_mt` are both PureBLAS, and `generic` is LinearAlgebra's own fallback. Everything that adjudicates
# or audits a PB-vs-reference pair must ask THIS, never `a != "pb"`.
_is_vendor_ref(a::AbstractString) = a == "openblas" || a == "aocl" || a == "mkl" || a == "accelerate"
_is_pb_arm(a::AbstractString) = a == _ARM_PB || a == _ARM_PB_MT
# Set PureBLAS's thread count for the arm about to be timed. Called BETWEEN windows, never inside one.
#
# QUIET THE VENDOR BACKEND TOO, and it is not a formality. `_use_ref!` leaves OpenBLAS at `_MT_NT`
# after a threaded reference arm, and OpenBLAS's idle threads SPIN-WAIT — so without this line a PB arm
# measured next in the rotation competes for its own cores with N spinners.
#
# The damage is confined to a call SHAPE rather than to an op, which is why it hid: measured at 6
# PureBLAS threads with OpenBLAS at 1 against 6, a single large Level-3 call is unmoved (gemm, symm and
# syrk at n=512/1024/2048 all read 0.90-1.02x), while `getri` — a driver issuing many SMALL threaded
# calls, where worker wake latency dominates and contention multiplies it — reads 0.686 ms against
# 11.526 ms at n=256. That is 16x, and it inverts the verdict: `getri` reads 0.06x "slower threaded"
# contaminated, against a true 1.77x faster at n=1024.
_use_pb!(a::AbstractString) = (BLAS.set_num_threads(1); PureBLAS.set_num_threads(a == _ARM_PB_MT ? _MT_NT : 1); a)
# `arms=pb` measures ONLY PureBLAS and reuses each reference arm already in the cache. That is the fast
# iteration path; it is also the one that can silently go stale, which is why every arm carries its own
# timestamp+commit and the table reports reference age rather than hiding it.
const _ARMS_SEL = let i = findfirst(a -> startswith(a, "arms="), ARGS)   # `let` — see _SELOP
    isnothing(i) ? nothing : split(ARGS[i][6:end], ",")
end
# One cell, not a whole op: `op=trmv size=512`. Sizes are matched exactly against the op's own list.
const _SELSIZE = let i = findfirst(a -> startswith(a, "size="), ARGS)   # `let` — see _SELOP below
    isnothing(i) ? nothing : parse(Int, ARGS[i][6:end])
end
# AOCL = AMD's Zen-tuned AOCL-BLIS + AOCL-libFLAME, shipped as the `AOCL_jll` artifact (AMD's own release,
# NOT generic blis_jll/libflame_jll). We LBT-forward its artifact .so paths directly (BLAS→libblis-mt,
# LAPACK→libflame), which is exactly what the AOCL.jl wrapper does — using the JLL keeps the dep to the
# reproducible binary artifact. `libblis-mt` is a multi-thread build; pin to 1 thread for a fair
# single-thread comparison (BLIS reads these at init; BLAS.set_num_threads(1) below re-enforces via LBT).

# BLIS reads these at init, before any forward — so they are set HERE, above `using AOCL_jll`, and
# cannot be changed afterwards by `BLAS.set_num_threads`.
#
# RESPECT AN EXPLICIT LAUNCH SETTING. This used to assign "1" unconditionally, which silently undid the
# environment a threaded-reference run must be launched with: `aocl_mt` would then record a
# SINGLE-threaded AOCL under a threaded arm name, and every verdict built on it would be wrong in
# PureBLAS's favour. A caller who set it to >1 has done so deliberately (see `_REF_MT`), and the arm
# guard below refuses the run if it is too low. Default is still 1: a gate sweep is single-threaded,
# and forgetting the variable must not silently produce a threaded reference either.
for _v in ("BLIS_NUM_THREADS", "OMP_NUM_THREADS")
    ENV[_v] = string(max(1, tryparse(Int, get(ENV, _v, "1")) === nothing ? 1 : tryparse(Int, get(ENV, _v, "1"))))
end
# Remember what BLIS was actually initialised with; the arm guard reads this, not the live ENV.
const _BLIS_INIT_NT = parse(Int, ENV["BLIS_NUM_THREADS"])
# WHICH VENDOR BLAS A BOX CARRIES IS ASKED, NEVER INFERRED FROM THE OS. AOCL_jll loads on every
# platform but holds an artifact only where AMD builds one, and Accelerate is a macOS framework with
# no package at all. An aarch64 Linux box (Graviton, Grace) has NEITHER, so a test that reads "not
# Apple, therefore AOCL" would forward LBT at a library that is not on the machine.
using AOCL_jll
const _HAS_AOCL = AOCL_jll.is_available()

# Accelerate = Apple's system BLAS/LAPACK (vecLib, inside the Accelerate umbrella framework — no package,
# no artifact, it ships with the OS). Forwarded by raw dylib path exactly like OpenBLAS/AOCL above.
#
# FORWARD THE UMBRELLA FRAMEWORK WITH A SUFFIX, NOT vecLib's `libBLAS.dylib`. Two interfaces live in
# the same binary behind different symbol manglings: the legacy LP64 (32-bit int) path that
# `lbt_forward` autodetects by default, and the ILP64 path Apple added in macOS 13.3
# ("$NEWLAPACK$ILP64"-suffixed symbols, e.g. `dgemm$NEWLAPACK$ILP64` — note NO trailing underscore
# before the `$`, unlike the classic Fortran mangling). Julia's own entry points are ILP64
# (`dgemm_64_`, required since PureBLAS's ABI is ILP64 too — see CLAUDE.md), so forwarding
# `libBLAS.dylib` without a suffix hint leaves the ILP64 slot EMPTY: `A*B` then prints "no BLAS/LAPACK
# library loaded for dgemm_64_" and returns zeros — a wrong NUMBER, not an exception, which is exactly
# what would land in a cache and be published as a reference measurement.
#
# The working incantation (confirmed empirically here, and matching
# JuliaLinearAlgebra/AppleAccelerate.jl's `load_accelerate`) needs a LEADING \x1a (0x1A, ASCII SUB)
# byte before the suffix text; without it `lbt_forward`'s suffix autodetection tries the hint as a
# plain appended suffix (`dgemm_$NEWLAPACK$ILP64`, which does not exist) and falls back to LP64.
const _ACCELERATE_PATH = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
const _ACCELERATE_SUFFIX = "\x1a\$NEWLAPACK\$ILP64"
# Accelerate ships with every macOS, so the OS test IS the availability test here — unlike AOCL,
# whose presence depends on an artifact and must be asked of the JLL.
#
# DO NOT "verify" the path instead. `isfile` on it is FALSE: macOS keeps system frameworks in the
# dyld shared cache, and the path resolves only through `dlopen`. And `dlopen` is not usable here
# either — it would initialize vecLib at this line, before `VECLIB_MAXIMUM_THREADS` is set below,
# which is the one moment that setting can still take effect.
const _HAS_ACCELERATE = Sys.isapple()
# `LinearAlgebra.BLAS.set_num_threads(1)` (called after every forward, below) does NOT constrain
# Accelerate — confirmed empirically (2026-09-22): a `dgemm` at n=4000 pinned at ~190-200% CPU (ps,
# sampled continuously across a 14 s / 60-call window) despite `set_num_threads(1)` having been called.
# vecLib reads `VECLIB_MAXIMUM_THREADS` at its OWN first-use initialization, independent of LBT's thread
# knob, and (like BLIS/OMP_NUM_THREADS above) that read happens ONCE — setting the env var later in the
# same process, after Accelerate has already been forwarded/called once, has no effect (confirmed: doing
# so gave ~same 2-core timing). Setting it here, at file load, before `_use_ref!` can ever reach
# Accelerate for the first time, is required for a real single-thread measurement. Verified fix: with
# this set before first use, the SAME `dgemm` pins at a clean ~99-100% CPU for the whole window
# (272.7 ms/call vs the uncontrolled run's 234.6 ms/call — i.e. the "second core" bought only ~16%,
# consistent with the matrix unit being a shared resource a second software thread cannot usefully
# double up on, not with real 2x general-purpose parallelism).
#
# THE TWO ACCELERATE ARMS CANNOT SHARE A PROCESS, and that is this same read, not a policy: once
# vecLib has initialised at one thread count, nothing in the process can move it. So the value pinned
# here is decided by the arm selection — `accelerate_mt` asks for `_MT_NT`, everything else for 1 —
# and requesting both in one run is refused below rather than silently recording one of them wrong.
const _VECLIB_NT = (!isnothing(_ARMS_SEL) && "accelerate_mt" in _ARMS_SEL) ? _MT_NT : 1
ENV["VECLIB_MAXIMUM_THREADS"] = string(_VECLIB_NT)

# Forward LBT to one backend. Called between timed windows, never inside one. `clear=true` on the BLAS
# forward drops the previous backend's symbols so a partial forward can never leave a mixed BLAS/LAPACK
# state — the failure mode where you measure AOCL's BLAS against OpenBLAS's LAPACK and never notice.
# ── THREADED REFERENCE ARMS ─────────────────────────────────────────────────────────────────────────
# `openblas_mt` / `aocl_mt` are the SAME libraries at `_MT_NT` threads. They exist so a threaded gate is
# possible at all: `pb_mt` against a single-threaded OpenBLAS is flattery, and the only honest
# comparison is threaded-against-threaded.
#
# ⚠ AOCL'S THREAD COUNT IS FIXED BY THE PROCESS ENVIRONMENT, NOT BY US. BLIS reads BLIS_NUM_THREADS /
# OMP_NUM_THREADS at init — which is why the header of this file pins them to 1 BEFORE `using AOCL_jll`.
# A later `BLAS.set_num_threads` cannot undo that. So a threaded-reference run MUST be launched with
# those set, and the guard below refuses rather than silently measuring a single-threaded AOCL and
# publishing it as a threaded reference:
#
#   BLIS_NUM_THREADS=6 OMP_NUM_THREADS=6 taskset -c <this box's mask, see _ARM_PB_MT> \
#       julia --project=bench -t 6 bench/plots.jl bench arms=pb,pb_mt,openblas_mt,aocl_mt nodraw
#
# Verified before use by `bench/probes/mt_reference_witness.jl`, which times a 2048 dgemm at 1 and N
# threads and requires >1.5x. Measured on Zen4: OpenBLAS 4.72x, AOCL 3.32x.
#
# `accelerate_mt` is the Apple member of the same family, and it is the one whose thread count this
# file cannot change after the fact: vecLib takes `VECLIB_MAXIMUM_THREADS` at its first use, so the
# count is fixed by `_VECLIB_NT` above, before any forward. `BLAS.set_num_threads` below is a no-op
# for it — harmless, and left in place because the arm is otherwise identical to the others.
const _REF_MT = Dict("openblas_mt" => "openblas", "aocl_mt" => "aocl",
                     "accelerate_mt" => "accelerate")
_is_mt_ref(a::AbstractString) = haskey(_REF_MT, a)

function _use_ref!(name::AbstractString)
    if _is_mt_ref(name)
        _use_ref!(_REF_MT[name])          # forward the library, which also re-asserts 1 thread …
        BLAS.set_num_threads(_MT_NT)      # … then ask for N. OpenBLAS honours this; BLIS took it at init.
        return name
    end
    # DL1's reference is NOT a BLAS library. `Dual` is not a `BlasFloat`, so LinearAlgebra's generic
    # fallback never reaches BLAS at all — there is nothing to forward, and forwarding would be a lie
    # about what ran. This arm is deliberately kept OUT of `_REF_ALL` (hence out of `_VIEWS`) so the
    # two-reference-view invariant is untouched and no bogus third view is rendered for every group.
    if name == "generic"
        BLAS.set_num_threads(1)
        return name
    end
    if name == "aocl"
        LinearAlgebra.BLAS.lbt_forward(AOCL_jll.aocl_blas_ilp64; clear = true)   # BLAS   → libblis-mt.so
        LinearAlgebra.BLAS.lbt_forward(AOCL_jll.aocl_lapack_ilp64)               # LAPACK → libflame.so
    elseif name == "accelerate"
        # ONE forward covers both BLAS and LAPACK — Accelerate ships them in the same umbrella binary,
        # unlike AOCL's separate blis/flame .so files, and the suffix hint is what selects the ILP64
        # interface. Forwarding vecLib's `libBLAS.dylib`/`libLAPACK.dylib` instead registers LP64 and
        # leaves the `_64_` slot empty, which returns ZEROS rather than raising — see the constants
        # above for the full account and why the leading \x1a byte is required.
        LinearAlgebra.BLAS.lbt_forward(_ACCELERATE_PATH; clear = true, suffix_hint = _ACCELERATE_SUFFIX)
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
# The second reference is whichever vendor BLAS the box actually carries — AOCL on AMD, Accelerate on
# Apple silicon, each the platform's own. A box with neither publishes against OpenBLAS alone rather
# than naming an arm it cannot run; `_VIEWS` follows `_REF_ALL`, so such a box renders one view.
const _VENDOR_ARM = _HAS_AOCL ? "aocl" : _HAS_ACCELERATE ? "accelerate" : nothing
const _REF_ALL = REFBK == "mkl" ? ["mkl"] : REFBK == "accelerate" ? ["accelerate"] :
    isnothing(_VENDOR_ARM) ? ["openblas"] : ["openblas", _VENDOR_ARM]
# ⚠ REFERENCE ARMS ARE CACHE-ONLY BY DEFAULT. Omitting `arms=` used to mean "measure every arm", so
# forgetting the flag silently re-ran OpenBLAS and AOCL — which is the whole reason the v3 cache stores
# them. The default is now PB ONLY; re-measuring a reference is an explicit, typed-out request.
#
# WHY THE DEFAULT HAD TO FLIP RATHER THAN BE REMEMBERED. 2026-09-12: a `group=DL1` run without `arms=`
# measured both reference arms. The PreToolUse guard that exists to stop exactly that had gone silently
# dead — it jq-parsed a transcript that had grown to 1000 MB, taking 10.2 s against its 5 s timeout, so
# the harness killed it and treated the timeout as non-blocking. Two independent safety nets (a rule I
# must remember, and a hook) both failed in one afternoon; a default that cannot be forgotten does not.
#
# To measure a reference deliberately: `arms=pb,openblas,aocl` (or any subset). The user's standing
# instruction is that this needs their explicit authorisation, per-run — it is not an agent decision.
const _REF_ARMS = isnothing(_ARMS_SEL) ? String[] : [a for a in _REF_ALL if a in _ARMS_SEL]
const _DO_PB = isnothing(_ARMS_SEL) || (_ARM_PB in _ARMS_SEL)
# `pb_mt` is opt-in ONLY. It must never be implied by a bare run: it is a second full pass over every
# cell (~86 min/box), and a run that quietly measured it would double the cost of the fast iteration path.
const _DO_PB_MT = !isnothing(_ARMS_SEL) && (_ARM_PB_MT in _ARMS_SEL)
# Threaded reference arms, opt-in only and never implied. Measuring a reference at all needs the user's
# explicit per-run authorisation; measuring one THREADED is a second, larger ask (it doubles the sweep).
const _REF_MT_ARMS = isnothing(_ARMS_SEL) ? String[] :
    [a for a in ("openblas_mt", "aocl_mt", "accelerate_mt") if a in _ARMS_SEL]
const _ACTIVE_ARMS = vcat(_DO_PB ? [_ARM_PB] : String[], _DO_PB_MT ? [_ARM_PB_MT] : String[],
    _REF_ARMS, _REF_MT_ARMS)
# ANY threaded arm puts this run in the mt family, and that has to be true for the REFERENCE arms too,
# not just `pb_mt`. Keying the cache on `_DO_PB_MT` alone meant `arms=openblas_mt,aocl_mt` — a run with
# no `pb_mt` at all — selected the GATE cache and would have written threaded references into the
# single-threaded gate. Caught before it ran; the whole point of the separate file is that this cannot
# happen, so the predicate must cover every threaded arm.
const _ANY_MT = _DO_PB_MT || !isempty(_REF_MT_ARMS)
# REFUSE rather than measure a lie. BLIS fixes its thread count at init from the environment, so if the
# process was not launched with it, `aocl_mt` would be a single-threaded AOCL recorded under a threaded
# name — and every downstream verdict built on it would be wrong in PureBLAS's favour.
#
# The Accelerate pair is refused for the same reason, one step earlier: vecLib fixes its thread count
# at first use, so a run holding both arms would measure ONE of them at the other's count and record
# it under the wrong name. They are separate runs, and the cache merge is what joins them.
if "accelerate_mt" in _REF_MT_ARMS && "accelerate" in _REF_ARMS
    error("""
        accelerate and accelerate_mt cannot be measured in the same process. vecLib reads
        VECLIB_MAXIMUM_THREADS at its FIRST USE and ignores `BLAS.set_num_threads` thereafter, so one
        of the two arms would be recorded at the other's thread count. Run them separately — the
        cache merge joins them:
          julia --project=bench bench/plots.jl bench arms=pb,accelerate …
          julia --project=bench -t $_MT_NT bench/plots.jl bench mt=$_MT_NT arms=pb_mt,accelerate_mt …""")
end
if "aocl_mt" in _REF_MT_ARMS
    _BLIS_INIT_NT >= _MT_NT || error("""
        aocl_mt requested but BLIS initialised with BLIS_NUM_THREADS=$_BLIS_INIT_NT (need >= $_MT_NT).
        BLIS reads it at INIT — `BLAS.set_num_threads` cannot fix it later, so this run would record a
        SINGLE-THREADED AOCL under a threaded arm name. Relaunch as:
          BLIS_NUM_THREADS=$_MT_NT OMP_NUM_THREADS=$_MT_NT taskset -c … julia --project=bench -t $_MT_NT …
        Verify first with bench/probes/mt_reference_witness.jl.""")
end
isempty(_ACTIVE_ARMS) && error("arms=$(join(something(_ARMS_SEL, []), ",")) selected nothing; valid: " *
    "$_ARM_PB,$_ARM_PB_MT,$(join(_REF_ALL, ",")),$(join((r * "_mt" for r in _REF_ALL if r != "mkl"), ","))")
if _DO_PB_MT
    Threads.nthreads() >= _MT_NT || error(
        "arms=…,$_ARM_PB_MT needs at least $_MT_NT julia threads, got $(Threads.nthreads()). " *
        "Run with -t $_MT_NT under this box's mask — see the _ARM_PB_MT comment for the per-box masks.")
    println(stderr, "▶ $_ARM_PB_MT arm ON at $_MT_NT threads — SCALING measurement, not a gate comparison")
end

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
    # ACHIEVED CLOCK (kHz) on the core this arm was measured on, sampled AT MEASUREMENT TIME. 0 for
    # records written before this field existed.
    #
    # Why per-cell and not just the header (added 2026-08-16): neuromancer's sweep verified ✅ at launch
    # (1981 MHz) and still stamped `freq=2172337kHz` against a 2000 MHz base, because the lock floated
    # DURING the run. A single header clock cannot say WHICH cells were affected, so one float condemns
    # all 85 ops — a 2.5 h re-run. With a per-cell clock the drift localises: re-measure only the cells
    # actually taken off-lock and merge, using the selective-re-measure machinery `op=`/`group=` already
    # provide. This is the one field that must be per-cell rather than per-run; the anchor above is
    # deliberately per-run because it exists for ACROSS-run comparison, whereas this exists to segment a
    # SINGLE run into valid and invalid parts.
    freq::Int
    # IN-WINDOW clock RANGE (kHz) — the min and max observed by bracketing every timing window of this
    # cell, not a single sample. `freq` above is one reading taken when the cell was stamped, i.e. the
    # clock when the work FINISHED; under a variable clock that is not the clock the work ran at, and a
    # cycles view built on it would be silently wrong. This pair makes the assumption checkable instead
    # of assumed: `flo == fhi` is a genuinely steady window, and a wide spread says the cell's own
    # timings span more than one clock state, so a cycles conversion for it carries that uncertainty.
    # 0,0 for records written before this field existed (treated as "unknown", never as 0 Hz).
    flo::Int
    fhi::Int
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
function _stamp(acc::ArmData)
    lo, hi = _khz_range!()          # observed across THIS cell's windows; resets for the next cell
    return CellData(
        a => ArmRec(
            Libc.strftime("%Y-%m-%dT%H:%M", time()), _COMMIT, _run_anchor(), _cell_khz(), lo, hi, q
        ) for (a, q) in acc
    )
end

# Achieved clock of the core THIS PROCESS is currently on, in kHz. Reads `/proc/self/stat` field 39 for
# the CPU id, then that core's `scaling_cur_freq`.
#
# Deliberately NOT `_achieved_khz()` (max over all cores), which the header uses: max-over-cores reports
# any core boosting, including one running an unrelated process such as an ssh session, and a false
# "floated" flag would condemn a perfectly good cell and trigger a needless re-measure. The sweep is
# `taskset`-pinned, so the process's own core IS the core under test — the only clock that can affect
# this cell's timings. Returns 0 (= unknown, never treated as off-lock) if anything is unreadable, so a
# platform without cpufreq degrades to today's behaviour rather than marking every cell invalid.
# In-window clock RANGE for the cell currently being measured. `_khz_obs!` is called immediately before
# and after every arm's timing window (see the measurement loop); `_khz_range!` returns the observed
# (min, max) and resets for the next cell. A zero sample means "unreadable" and is ignored rather than
# treated as a 0 Hz clock.
const _KHZ_LO = Ref(typemax(Int))
const _KHZ_HI = Ref(0)
function _khz_obs!(v::Int)
    v > 0 || return nothing
    _KHZ_LO[] = min(_KHZ_LO[], v)
    _KHZ_HI[] = max(_KHZ_HI[], v)
    return nothing
end
function _khz_range!()
    lo, hi = _KHZ_LO[], _KHZ_HI[]
    _KHZ_LO[] = typemax(Int); _KHZ_HI[] = 0
    return hi == 0 ? (0, 0) : (lo, hi)
end

function _cell_khz()
    return try
        cpu = parse(Int, split(read("/proc/self/stat", String))[39])
        f = "/sys/devices/system/cpu/cpu$(cpu)/cpufreq/scaling_cur_freq"
        isfile(f) ? something(tryparse(Int, strip(read(f, String))), 0) : 0
    catch
        0
    end
end
# Measured ONCE per run, lazily, on the first cell stamped — not at save time. Save time is the END of a
# sweep that can run for hours, and the point of the field is to describe the machine while the arms were
# actually being timed. One anchor per run (not per cell) is deliberate: arms within a run are compared
# same-run and already cancel machine state; the field exists for the ACROSS-run comparison.
const _RUN_ANCHOR = Ref{Union{Nothing, Float64}}(nothing)
function _run_anchor()
    isnothing(_RUN_ANCHOR[]) && (
        _RUN_ANCHOR[] = try
            _anchor_secs()
        catch
            NaN
        end
    )
    return _RUN_ANCHOR[]::Float64
end
# Progress line only: median ratio of the first available reference against pb this round. Under
# `arms=pb` there IS no reference in this round (that is the point of the mode), so fall back to pb's
# median time in µs — a bare NaN told you nothing, and the per-round time is exactly what you want to
# eyeball for stability when iterating on the kernel. `_ROUNDLBL` says which you are looking at.
const _ROUNDLBL = !isempty(_REF_ARMS) ? "rounds" :
    _DO_PB_MT ? "rounds (pb/pb_mt speedup)" : "rounds (pb µs)"
function _round_med(qs::ArmData)
    haskey(qs, _ARM_PB) || return NaN
    for a in _REF_ARMS
        haskey(qs, a) && return median(_ratio(qs[a], qs[_ARM_PB]))
    end
    # With the mt arm there is no reference, so the fallback below would print pb's microseconds and the
    # pb_mt arm would be INVISIBLE for the whole sweep — ~86 min of flying blind, with no way to notice
    # that the arm had silently degenerated to 1.00. Report the SPEEDUP instead: pb / pb_mt, so >1 means
    # threading paid. Same orientation as the reference ratio above (bigger is better for PureBLAS).
    (_DO_PB_MT && haskey(qs, _ARM_PB_MT)) && return median(_ratio(qs[_ARM_PB], qs[_ARM_PB_MT]))
    return median(qs[_ARM_PB]) * 1.0e6
end
# A THIRD, DERIVED view — not a reference arm. It divides by whichever of `_REF_ALL` is faster at each
# cell (see `_series`), so it draws the gate itself. It is NOT in `_VIEWS`: nothing is measured against
# it and it has no gen_table of its own (the coverage table already reports this exact number).
const _GATE_VIEW = "gate"
_refname(r) = r == "mkl" ? "MKL" : r == "aocl" ? "AOCL" : r == "accelerate" ? "Accelerate" :
    r == "generic" ? "LinearAlgebra generic" :
    r == _GATE_VIEW ? "faster of " * join((_refname(x) for x in _REF_ALL), " and ") : "OpenBLAS"
# SVG/table filename suffix: "" for OpenBLAS (the default baseline), "_mkl"/"_aocl" otherwise
_refsuf(r) = r == "openblas" ? "" : "_$r"
const REFNAME = _refname(REFBK)
const REFSUF = _refsuf(REFBK)
# EVERY reference view is rendered in ONE invocation — the `aocl` argument no longer selects a view.
# WHY: the two views read the SAME v3 cache, so they can only disagree if one was rendered and the other
# was not. That is not hypothetical — on 2026-08-17 the OpenBLAS SVGs had sat at commit bdb9497 while the
# AOCL set had been re-rendered three times, and the two published pages contradicted each other about
# the same fleet. Rendering both together makes divergence unrepresentable rather than merely discouraged.
const _VIEWS = REFBK == "mkl" ? ["mkl"] : _REF_ALL
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
const _ISAOVR = let i = findfirst(a -> startswith(a, "isa="), ARGS)   # `let` — see _SELOP
    isnothing(i) ? nothing : ARGS[i][5:end]
end
const _SLUGOVR = let i = findfirst(a -> startswith(a, "slug="), ARGS)  # `let` — see _SELOP
    isnothing(i) ? nothing : ARGS[i][6:end]
end
const _HWB = PureBLAS._HW
# The slug is a BOX LABEL and must be STABLE: it names the cache file (`plots_data_$(SLUG)_$(host).txt`,
# :1531), keys the plot series (:2098) and is referenced by bench/artifact_build.sh and
# coverage_routing.jl. It used to key on `_double_pumped`, which was a proxy for "Zen4 vs Zen5" — that
# stopped being true on 2026-09-09 when the datapath became a detected CPUID fact and neuromancer (a
# Krackan mobile Zen5) correctly turned out double-pumped. Keying the LABEL on the µarch family keeps
# neuromancer on "zen5" and avoids colliding it with wintermute's "avx512" / orphaning its cached
# reference arms. The physical fact is stamped separately into every header as `datapath=`/`dp=` (:315).
const _AUTOSLUG = _W64P == 8 ? (PureBLAS._CPU_FAMILY == 0x1A ? "zen5" : "avx512") :
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
    join(
        (
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
        ), ","
    )
catch e
    "unavailable($(typeof(e)))"
end

# DETECTED HARDWARE, stamped into the cache header as `hw=`. `uarch=`/`isa=`/`cpu=` are LABELS — a µarch
# name, an ISA string and a marketing model string. None of them is the data that actually drives a single
# tuning decision: every derived block size, cutoff and unroll is a formula over `_HW`'s fields (req#8), so
# a cache that omits them cannot explain its own numbers.
#
# WHY THIS EXISTS (2026-08-16): building the fleet µarch table required ssh-ing three boxes and running
# `lscpu`, because the caches carried the CPU's *name* but not its cache sizes, lane count, vendor/family
# or datapath width. The same gap made a CI-only failure undiagnosable — CI records no hardware identity
# at all, so a red run cannot be attributed to a code change versus a runner CPU rotation.
#
# Stamp what PureBLAS DETECTED, not what the OS reports: a Preferences override or a mis-detection is
# exactly the thing worth catching, and it is invisible if this reads `lscpu` instead of `_HW`.
# `dp=` (double-pumped) is included because it is derived, not detected — Zen3 and Zen4 share family 0x19
# and are separated only by `simd`, so the resolved boolean is worth recording next to its inputs.
# Bundled OpenBLAS version — the reference arm ships INSIDE Julia, so it changes when Julia does.
_obversion() = try
    string(pkgversion(OpenBLAS_jll))
catch
    "?"
end

_hwstamp() = try
    hw = PureBLAS._HW
    join(
        (
            "simd=$(hw.simd)", "w64=$(PureBLAS._vwidth(Float64))",
            "l1=$(hw.l1)", "l2=$(hw.l2)", "l3=$(hw.l3)",
            "vendor=$(hw.vendor)", "family=$(hw.family)", "nvreg=$(hw.nvreg)",
            "datapath=$(PureBLAS._datapath_bytes(hw))", "dp=$(PureBLAS._double_pumped(hw))",
        ), ","
    )
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
# `outdir=<dir>` redirects BOTH render outputs (the SVGs and gen_table*.md) into one directory instead of
# docs/src/assets + bench/. Only bench/check_artifacts_current.sh uses it: it re-renders from the current
# caches into a temp dir and diffs against the committed artifacts, which it cannot do if rendering
# overwrites them first. Nothing about the measurement changes.
const _OUTDIR = let i = findfirst(a -> startswith(a, "outdir="), ARGS)   # `let` — see _SELOP
    isnothing(i) ? nothing : ARGS[i][8:end]
end
# `let`, not a bare `(i = …; …)`: the bare form leaks a GLOBAL `i`, which made every later top-level
# `for` that uses `i` emit a soft-scope warning — including the per-arm cache merge, the one loop where
# nobody wants to be wondering whether `i` is the local it looks like.
const _SELOP = let i = findfirst(a -> startswith(a, "op="), ARGS)
    isnothing(i) ? nothing : ARGS[i][4:end]
end
const _SELGRP = let i = findfirst(a -> startswith(a, "group="), ARGS)
    isnothing(i) ? nothing : ARGS[i][7:end]
end
# `maxsize=<n>` caps every op's size ladder at n, full methodology otherwise (unlike `lite`, which ALSO
# cuts rounds/samples). For scoping a first-pass run's wall-clock (e.g. a new, unlocked, un-fleeted box)
# without touching measurement quality on the sizes that ARE run. Never set by the AMD fleet scripts.
const _MAXSZ = let i = findfirst(a -> startswith(a, "maxsize="), ARGS)
    isnothing(i) ? nothing : parse(Int, ARGS[i][9:end])
end
_want(lvl, nm) = (isnothing(_SELOP) && isnothing(_SELGRP)) || _SELOP == nm || _SELGRP == lvl
_cap(szs, maxn) = Tuple(s for s in szs if s <= maxn)   # per-op size cap (e.g. skip 4096 for slow ops)
# lite caps sizes at 1024 (drops the expensive 2048/4096 tail) — keeps the meaningful mid-n range while
# skipping the O(n³) large-n sink that dominates wall time. Guarded so a cap never yields an empty tuple.
function _sizes(szs)
    t = szs
    _LITE && (t = Tuple(s for s in t if s <= 1024))
    isnothing(_MAXSZ) || (t = Tuple(s for s in t if s <= _MAXSZ))
    return isempty(t) ? szs[1:1] : t
end

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
# `refs` overrides the reference arm list for ONE group. Only DL1 uses it: its reference is
# LinearAlgebra's own generic fallback over `Vector{Dual}` (what a ForwardDiff user gets without
# PureBLAS), which is a different Julia IMPLEMENTATION rather than a different BLAS library — so it
# cannot be an entry in `_REF_ALL` without inventing a third reference view for every other group, and
# it must not be recorded under `openblas`/`aocl`, which would write false provenance into the cache.
function sweep(mk, sizes, work_ob, work_pb, repfn; samples = 400, seconds = 0.15, refs = nothing)
    out = Tuple{Int, CellData}[]
    armlist = isnothing(refs) ? nothing :
        vcat(_DO_PB ? [_ARM_PB] : String[], [a for a in refs if isnothing(_ARMS_SEL) || a in _ARMS_SEL])
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
            arms = isnothing(armlist) ? _round_arms(r) : circshift(armlist, r - 1)
            qs = Dict{String, Vector{Float64}}()
            for a in arms
                # `pb_mt` runs work_pb like `pb` does — only the thread count differs, which is what
                # makes this a PAIRED in-process A/B rather than two runs compared afterwards.
                w = _is_pb_arm(a) ? (_use_pb!(a); work_pb) : (_use_ref!(a); work_ob)
                # Bracket the window — see the note on the other measurement loop. THERE ARE TWO of
                # them (this one drives L1/L2, the other the rep-pool levels) and instrumenting only
                # one is why the first run with `flo|fhi` wrote EMPTY ranges for every L1/L2 cell:
                # the field was present, the samples were never taken, and `cellcycles.jl` correctly
                # reported `wobble ?` for cells that had in fact just been measured. If a third
                # measurement path is ever added it needs these two calls too.
                _khz_obs!(_cell_khz())
                b = @be mk(s) (c -> w(c, reps)) evals = 1 samples = samples seconds = seconds
                _khz_obs!(_cell_khz())
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
# `cold` forces reps=1 on the L1 sweep -- no repeated-call reuse of the SAME operand buffer within one
# timed window. WHY: `_L1REP` amortizes timer overhead for cheap ops by calling `reps` times on one
# `setup()`-allocated buffer; at large n that buffer can be small enough to stay resident in L2/SLC
# across the whole reps loop, so a nominally "bandwidth-bound" cell ends up measuring repeated-access-
# to-a-warm-buffer throughput rather than genuine cold/DRAM-streaming throughput -- and a library whose
# inner loop pipelines especially well against a HOT buffer can look disproportionately faster than it
# actually is on real, once-through data.
#
# CONFIRMED 2026-09-22 on Apple Silicon (M6): the reps-amortized cache read Accelerate's axpy as ~4.3x
# OpenBLAS at n=1e6 (PB/openblas=1.00, PB/accelerate=0.23); genuinely cold single-shot calls (fresh
# arrays, ONE call, no reuse) at unambiguously DRAM-scale sizes (10M-50M elements, 240MB-1.2GB, far past
# any on-chip cache) read only 1.06-1.36x -- the physically plausible number for a per-core DRAM-
# bandwidth edge. `evals=1` already re-runs `setup()` fresh per Chairmarks SAMPLE (see `sweep`'s own
# comment), so reps=1 removes exactly the one remaining reuse path (the inner `for _ in 1:reps` loop)
# without touching anything else about the methodology.
#
# NEVER the default -- an explicit, opt-in re-measure mode. Changes nothing for the AMD fleet's existing
# methodology or caches unless someone passes `cold` there too. Whether the SAME artifact inflates
# large-n L1 numbers on Zen3 (32 MiB L3 -- n=1e6's 16 MB working set fits) is unconfirmed; this was
# diagnosed on one Apple Silicon box, not validated on the fleet.
const _COLD = "cold" in ARGS
_l1_repfn(base) = _COLD ? (_ -> 1) : base

_reps_cubic(s) = clamp(20_000_000 ÷ (s * s * s), 1, 512)
# QUADRATIC sibling — for LP entries whose work is O(n²), not O(n³): a solve against an ALREADY
# factored matrix with a single RHS (trtrs/getrs/sytrs/pttrs). Using `_reps_cubic` for those scales the
# rep count by the wrong power, and the timed work per sample then swings 64x across the ladder:
#     n         8      32      256     1024    2048
#     reps    512     512        1        1       1     (cubic model)
#     reps·n² 32.8k  524k      65.5k    1.05M   4.19M
# It bottoms out at n=256, which is exactly where the measured cycle curves for all four ops bottom
# out — trtrs read 983 cycles at n=32, 15 at n=256 and 348 at n=1024, i.e. cost FALLING 65x while the
# problem grew 8x, which is impossible for an O(n²) kernel. At the bottom the reps clamp to 1, so a
# sample times a single ~5 µs solve and the ratio measures call overhead rather than the kernel.
# Same 20M budget, one power lower, so work per sample stays flat and every size is timed comparably.
_reps_quadratic(s) = clamp(20_000_000 ÷ (s * s), 1, 512)

# Heavy O(n³) sweep for L3 / LAPACK. `@be` with `evals=1` runs a FRESH `mk(s)` per sample (the destructive
# op mutates its input → one op per context) and EXCLUDES the mk allocation from the timed core — which
# removes the old hazard where the per-round alloc dropped the core off-clock and biased whichever side was
# timed first. Warmup + sample sizing are Chairmarks'. Small n is measured cleanly (no timer-quantization
# reps hack needed — Chairmarks amortizes internally).
# ⚠ STILL REQUIRES CPU BOOST DISABLED (`echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost`,
# performance governor) so the fixed clock keeps OB vs PB comparable. See memory dev-fleet.
                                       # `refs` mirrors `sweep`'s override, for DL3 — see the note there.
function sweep_heavy(mk, ob1, pb1, sizes; samples = 64, seconds = 4.0, repsof = _reps_cubic, refs = nothing)
    armlist = isnothing(refs) ? nothing :
        vcat(_DO_PB ? [_ARM_PB] : String[], [a for a in refs if isnothing(_ARMS_SEL) || a in _ARMS_SEL])
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
            for a in (isnothing(armlist) ? _round_arms(r) : circshift(armlist, r - 1))  # rotated (generalised ABBA); reference switch is outside the window
                f = _is_pb_arm(a) ? (_use_pb!(a); pb1) : (_use_ref!(a); ob1)   # see the note in `sweep`
                # Bracket each arm's window with a clock sample. One sample AFTER the fact (what
                # `_cell_khz()` alone gives) records the clock when we finished, not the clock the work
                # actually ran at — fine while the box is pinned, wrong the moment it is not. Keeping
                # the min/max ACROSS every window of this cell turns "assume pinned" into a measured
                # range, so a cell that drifted mid-measurement says so instead of silently reporting a
                # single number that was never true. Two /sys reads per arm-round against a ≥0.5 s
                # window is unmeasurable overhead.
                _khz_obs!(_cell_khz())
                b = @be [mk(s) for _ in 1:reps] (
                    cs -> (
                        v = 0.0; for c in cs
                            v += f(c)
                        end; v
                    )
                ) evals = 1 samples = samples seconds = secs
                _khz_obs!(_cell_khz())
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
# n=1000 IS NOT DECORATION — it is the only non-power-of-two size in L2/L3, and it exists because
# every other one aliases. A kernel running several concurrent column streams collides in one cache set
# when `lda * sizeof(T)` is a multiple of the L1 way stride (4096 B on this whole fleet), and EVERY
# power of two is. So a po2-only size list measures such kernels at their single pathological point, on
# every box, uniformly — which produces no differential signal and hides the effect completely.
# Concretely, measured 2026-08-22 on Zen5 gemv-N, m-inner arm vs the old path:
#     511 wins 8.4% | 512 LOSES 8.8% | 513 flat | 1023 wins 10% | 1024 LOSES 5% | 1025 wins 8.5%
#     2047 wins 21% | 2048 flat      | 2049 wins 20%
# `gemvn_minner` is DISABLED on that box on the strength of the po2 column alone, while real callers —
# who pass arbitrary n — would gain 8-21%. The same blind spot hid a hardcoded 8-way associativity in
# `_L1_WAY_D` that switched off trsm/trmm de-aliasing on Zen5 entirely, and it is the reason the
# potrf-upper and L3 syrk/trmm po2-lda findings were each stumbled into rather than measured.
# 1000 is chosen because 1000*8 = 8000 B is NOT a multiple of 4096 (1536 would have been: 12288 B), and
# it sits beside 1024 so the two are adjacent in every published table and plot.
# n=100 COVERS A SECOND BLIND SPOT, distinct from aliasing: every other L2/L3 size is a multiple of
# BOTH register-tile dimensions (mr = _MR*W = 16 or 8, nr = _NR = 8), so `_microkernel_masked!` — the
# partial-tile path that exists solely for n not divisible by the tile — is never entered by the gate.
# n=1000 does not fix that either (1000 % 8 == 0, so it masks m only). 100 % 16 = 4 and 100 % 8 = 4, so
# it exercises BOTH m- and n-masking on every box, and 100*8 = 800 B is not a way-stride multiple.
# Small AND non-po2 also means it lands in L1/L2, where per-call overhead and edge handling dominate —
# the regime the large sizes cannot speak to.
# ONE NON-POWER-OF-TWO SIZE PER RESIDENCY BAND. A single non-po2 size is not enough: the two blind
# spots it covers (cache-set aliasing and the masked partial-tile kernel) can both behave differently
# in L1, L2, L3 and DRAM, and every pre-existing size is a power of two — hence aliasing, and hence a
# multiple of both register-tile dims, so `_microkernel_masked!` was never entered at all.
# Residency is identical on ALL THREE boxes for each (checked against 32/32/48 KiB L1, 512/1024/1024
# KiB L2, 32/16/16 MiB L3), so a row means the same thing everywhere:
#     n=50    19.5 KiB   L1     %mr=2  %nr=2   masks both
#     n=100   78 KiB     L2     %mr=4  %nr=4   masks both
#     n=1000  7.6 MiB    L3     %mr=8  %nr=0   masks m
#     n=2100  33.6 MiB   DRAM   %mr=4  %nr=4   masks both
# None is a way-stride multiple (n*8 % 4096 != 0), so they are clean on the aliasing axis; the po2
# sizes beside them supply the aliasing case. n=300 was rejected: it is L3 on galen but L2 on the other
# two, so the row would not mean the same thing per box.
const L2SZ = (50, 64, 100, 128, 256, 512, 1000, 1024, 2048, 2100, 4096)
const L3SZ = (8, 32, 50, 100, 128, 256, 512, 1000, 1024, 2048, 2100, 4096)   # O(n³); 4096 shows large-n syrk/trmm behavior
# LAPACK factorizations, to 4096. NON-PO2 entries (50/100/1000/2100) are one per RESIDENCY BAND, the
# same set L2SZ/L3SZ use — and they matter MORE here than anywhere else: the panel kernels these ops
# route through (`_trsm_rl_split_f64!`, `_trsm_right_lower_f64!`) handle a sub-W row tail and a
# `nb < _fh_chol_nb()` column remainder in SEPARATE code paths, and a po2-only ladder leaves BOTH
# permanently empty. That is not hypothetical: those two tails were scalar loops walking the whole
# solved prefix one row and one column at a time, worth -59% at n=50 on the side-R gate shape, and the
# po2 ladder had reported the op as passing for months. See kb pureblas-nonpo2-sizes-were-invisible.
const LPSZ = (8, 32, 50, 100, 128, 256, 512, 1000, 1024, 2048, 2100, 4096)
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
            _meas!(l1, "L1", nm, () -> sweep(s -> (randn(s), randn(s)), _sizes(L1SZ), ob, pb, _l1_repfn(_L1REP)))
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
        # ── trsv LOWER, and the TRANSPOSED form — the shapes real callers actually issue ────────────
        # The `trsv` row above measures uplo='U', trans='N' ONLY. `potrs` issues L/N then L/T, and
        # `getrs` issues L/N-unit then U/N (and the transposed pair for trans='T'), so NEITHER of
        # potrs's shapes was gated. That hole hid a real miss: on 2026-08-18 `trsv` read 1.014/1.007/
        # 1.001 on Zen5 at n=256/512/1024 while `potrsL` read 0.891/0.836/0.905 at the SAME sizes on the
        # SAME box — the benchmark was measuring the one shape that was fine. The T form had no fused
        # panel kernel at all (the N form has had `_trsv_fused8!` for months); adding `_trsv_fused8_t!`
        # moved the T pass 1.4-1.5x. A kernel shape that no row measures can regress indefinitely, so
        # both shapes are gated from here on.
        let LO = Char(76)
            for (nm, ul, tr) in (("trsvLN", LO, TN), ("trsvLT", LO, TT))
                add(
                    nm, s -> (
                        A = randn(s, s) ./ (2s); for i in 1:s
                            A[i, i] = 1 + abs(A[i, i])
                        end; (A, randn(s), randn(s))
                    ),
                    (c, m) -> (
                        for _ in 1:m
                            copyto!(c[3], c[2]); B.trsv!(ul, tr, TN, c[1], c[3])
                        end; c[3][1]
                    ),
                    (c, m) -> (
                        for _ in 1:m
                            copyto!(c[3], c[2]); PureBLAS.trsv!(c[1], c[3]; uplo = ul, trans = tr)
                        end; c[3][1]
                    )
                )
            end
        end
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
        NN = Char(78); LT = Char(76); UP = Char(85); RT = Char(82)
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
            # REAL side-R trmm. Its absence was a COVERAGE GAP, not an oversight of no consequence: the
            # complex sibling `ztrmmR` exists precisely because "plots measured only side-L → the 0.24
            # side-R routing bug went unseen" (see its comment below), and the real side never got the
            # same treatment. `_trmm_right!` is a wholly separate driver from `_trmm_left!` — different
            # packing decision, different base — and `trmm_rpack` (its pack cut, worth +8.1% on Zen5 at
            # n=512) was tunable but NOT gate-verifiable while this row was missing.
            # Shape mirrors ztrmmR (side='R', uplo='U', trans='N') so the real and complex rows measure
            # the same routing.
            "trmmR", s -> (tri(s), randn(s, s)),
            c -> (B.trmm!(RT, UP, NN, NN, 1.0, c[1], c[2]); c[2][1]),
            c -> (PureBLAS.trmm!(c[2], c[1]; side = RT, uplo = UP); c[2][1])
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
            c -> (PureBLAS.getrs!(c[1], c[2], c[3]; trans = TN); c[3][1]); sizes = _cap(LPSZ, 2048), repsof = _reps_quadratic
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
            sizes = _cap(LPSZ, 2048), repsof = _reps_quadratic
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
            c -> (PureBLAS.sytrs!(c[1], c[2], c[3]; uplo = LP); c[3][1]); sizes = _cap(LPSZ, 2048), repsof = _reps_quadratic
        )
        # ── FORWARDED BUT NEVER BENCHMARKED: the inverse + least-squares/eigen family ─────────────────
        # `cabi_forward.jl` LBT-forwards potri, getri, trtri, sytri, gelsy, gelsd and geev into user
        # code. Not one of them had a gate cell. That is how a PureOSQP run found potri at 0.67x
        # (322 us vs OpenBLAS 216 us at n=200) while every red-cell report said the fleet was fine: the
        # gate cannot miss what it never measures. It is the same class as the C-ABI routing blind spot
        # — the thing that ships is not the thing that is measured — and it stayed open longer because
        # nothing even listed these as uncovered.
        #
        # potri and trtri were C-ABI-only, which is exactly why getrs/potrs/trtrs went ungated until
        # native entries were added for them (see the note above); they now have the same treatment in
        # `lapack/inverses.jl`, and the shims call those, so there is one implementation. `getri` is
        # still C-ABI-only and remains the last uncovered forwarded symbol.
        #
        # `mk` returns the FACTOR (potrf'd / already triangular) so the timed core is the inversion
        # alone, matching how the getrs/potrs rows time the solve alone.
        #
        # ⛔ THE TIMED CORE MUST NOT ALLOCATE. These rows used to call `…!(copy(c))`, defended as
        # "both arms copy, so the copy is common-mode". Common-mode it is; harmless it is not. The
        # setup already hands the core `reps` FRESH contexts, each used exactly once (`evals=1`), so
        # the copy was redundant — and at small n it dominated: 592 B per call at n=8 against a 140 ns
        # kernel, 512 calls per sample. That allocation drove GC INTO the timed window, and GC is what
        # produced the published bands: measured on wintermute 2026-09-11, trtri@8 samples split into
        # GC-free 106 us and GC-hit 764 us (7.2x), 2% of samples at n=8 and 12% at n=32, which is
        # exactly the round-level bimodality (round medians 103/247/119/241/113/248/251/253 us).
        # Dropping the copy takes GC to 0.0% of samples and the n=8 median down 20% — the cell stops
        # timing the allocator and starts timing the kernel.
        # The kernels themselves were never the problem: `trtri!` measures 0 B at n=8 and n=256, and
        # `src/verify.jl`'s entry-point contracts (`_strict_trtri_probe` et al.) already refresh their
        # operand the allocation-free way, with `copyto!` into a preallocated buffer.
        # Every other row here (sytrs, gbtrf, geqp3, …) already passed `c` straight through; these
        # eight were the outliers. KEEP IT THAT WAY — `bench/check_sweep_noalloc.jl` asserts it.
        # FACTOR WITH PureBLAS, NOT WITH LAPACK — see the note on `_lufac2` below. A setup that calls a
        # foreign BLAS immediately before the timed window leaves that library's threads spinning
        # through the measurement, and the cost scales with PureBLAS's worker count.
        _cholfac(s) = (A = _hpd(Float64, s); PureBLAS.potrf!(A; uplo = LP); A)
        addh(
            "potri", _cholfac,
            c -> (LinearAlgebra.LAPACK.potri!(LP, c); c[1, 1]),
            c -> (PureBLAS.potri!(c; uplo = LP); c[1, 1]); sizes = _cap(LPSZ, 2048)
        )
        _trifac(s) = tril(randn(Float64, s, s) + s * LinearAlgebra.I)
        addh(
            "trtri", _trifac,
            c -> (LinearAlgebra.LAPACK.trtri!(LP, TN, c); c[1, 1]),
            c -> (PureBLAS.trtri!(c; uplo = LP, diag = TN); c[1, 1]); sizes = _cap(LPSZ, 2048)
        )
        # `\` on a general matrix and `inv(A)` both land here through LBT. mk returns the LU factors so
        # the timed core is the inversion, not the factorization.
        # A SETUP MUST NOT CALL A FOREIGN BLAS, and this one did (`LinearAlgebra.LAPACK.getrf!`).
        #
        # OpenBLAS's worker threads spin for a timeout after each call rather than blocking, so a
        # foreign factorization run immediately before every sample leaves them hot through the timed
        # window. Setting `BLAS.set_num_threads(1)` does not help — the pool already exists and is
        # already spinning. Measured at n=256, Zen4, BLAS pinned to 1 throughout, on two inputs that
        # are numerically the same matrix (max|diff| 2.3e-13, identical ipiv, both the identity):
        #
        #     PureBLAS workers     1       2       3       4       6
        #     LAPACK-factored   0.898   4.037   6.959   7.723   8.606  ms
        #     PureBLAS-factored 0.694   0.641   0.678   0.677   0.677  ms
        #
        # The cost scales with PureBLAS's worker count, which is contention and not a property of the
        # data: copying the PureBLAS factors into brand-new arrays reproduces the fast column exactly.
        # Uncorrected it read `getri` as 0.06x "slower threaded" against a true 1.77x faster.
        #
        # `_lufac` and `_sytrfac` above already factor with PureBLAS; this is the same rule, and the
        # comparison stays fair either way because BOTH arms invert the same matrix.
        function _lufac2(s)
            F = Matrix{Float64}(randn(Float64, s, s) + s * LinearAlgebra.I)
            ip = Vector{Int}(undef, s)
            PureBLAS.getrf!(F, ip)
            return (F, ip)
        end
        addh(
            "getri", _lufac2,
            c -> (LinearAlgebra.LAPACK.getri!(c[1], c[2]); c[1][1, 1]),
            c -> (PureBLAS.getri!(c[1], c[2]); c[1][1, 1]); sizes = _cap(LPSZ, 2048)
        )
        addh(
            "sytri", _sytrfac,
            c -> (LinearAlgebra.LAPACK.sytri!(LP, c[1], c[2]); c[1][1, 1]),
            c -> (PureBLAS.sytri!(c[1], c[2]; uplo = LP); c[1][1, 1]); sizes = _cap(LPSZ, 2048)
        )
        # Least squares, square-ish and overdetermined-by-construction (m = n here, nrhs = 1) so the
        # timed core is the factor-and-solve, not the shape handling. Capped at 1024 like `gels`.
        _lsq(s) = (randn(Float64, s, s), randn(Float64, s, 1))
        # gelsy needs a pivot vector; it belongs in the CONTEXT, not in the timed core. `zeros(Int, n)`
        # inside the closure allocated 8n B on every call — the same defect as the `copy`s above.
        _lsqp(s) = (randn(Float64, s, s), randn(Float64, s, 1), zeros(Int, s))
        addh(
            "gelsy", _lsqp,
            c -> (LinearAlgebra.LAPACK.gelsy!(c[1], c[2], -1.0); c[2][1]),
            c -> (PureBLAS.gelsy!(c[1], c[2], c[3], -1.0); c[2][1]);
            sizes = _cap(LPSZ, 1024)
        )
        addh(
            "gelsd", _lsq,
            c -> (LinearAlgebra.LAPACK.gelsd!(c[1], c[2], -1.0); c[2][1]),
            c -> (PureBLAS.gelsd!(c[1], c[2], -1.0); c[2][1]); sizes = _cap(LPSZ, 1024)
        )
        # Nonsymmetric eigenvalues, VALUES ONLY ('N','N'): the vector paths differ enough between
        # implementations that timing them compares two algorithms rather than one kernel, the same
        # reason gesvd is gated values-only. Capped at 1024 — geev is O(n^3) with a large constant and
        # 2048+ would not be seconds-bounded at one sample.
        addh(
            "geev", s -> randn(Float64, s, s),
            c -> (LinearAlgebra.LAPACK.geev!(TN, TN, c); c[1, 1]),
            c -> (PureBLAS.geev!(TN, TN, c); c[1, 1]); sizes = _cap(LPSZ, 1024)
        )
        # Banded LU: kd scales with n (a fixed narrow band makes this O(n) and hides the kernel).
        _gbd(s) = (
            kl = max(1, s ÷ 8); ku = kl; AB = zeros(Float64, 2kl + ku + 1, s);
            for j in 1:s, i in 1:(2kl + ku + 1)
                AB[i, j] = randn()
            end;
            for j in 1:s
                AB[kl + ku + 1, j] = 4 * (kl + ku)
            end; (kl, ku, AB)
        )
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
            c -> (LinearAlgebra.LAPACK.gels!(TN, c[1], c[2]); c[2][1]),
            c -> (PureBLAS.gels!(TN, c[1], c[2]); c[2][1]); sizes = _cap(LPSZ, 1024)
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
            c -> (PureBLAS.pttrs!(c[1], c[2], c[3]); c[3][1]); sizes = TDSZ, repsof = _reps_quadratic
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
            return AB
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
            return uplo == 'L' ? [A[i, j] for j in 1:n for i in j:n] : [A[i, j] for j in 1:n for i in 1:j]
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
                # GENUINELY complex alpha, unit modulus (0.8+0.6i) so `m` in-place reps neither grow nor
                # decay. The `zscal` row above scales by `1.0000001+0im`, which `_scal!` routes to the REAL
                # kernel over 2n reals — so until this row existed `_scal_cmplx_simd!` had no gate coverage
                # at all. The `zscal` row is deliberately left as it is: its cache history is evidence.
                (
                    "zscalc", (c, m) -> (
                        for _ in 1:m
                            B.scal!(0.8 + 0.6im, c[1])
                        end; real(c[1][1])
                    ),
                    (c, m) -> (
                        for _ in 1:m
                            PureBLAS.scal!(0.8 + 0.6im, c[1])
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

    # ── DL1: BLAS-1 over ForwardDiff.Dual{Tag,Float64,1} (forward-mode AD) ───────────────────────────
    #
    # WHY THIS GROUP HAS NO VENDOR REFERENCE. OpenBLAS and AOCL have no dual-number support at all —
    # `Dual` is not a `BlasFloat`, so LinearAlgebra never forwards it to BLAS. The honest reference is
    # therefore **LinearAlgebra's own generic fallback**, which is exactly what a ForwardDiff user gets
    # today without PureBLAS. That is what the `generic` arm measures (see `_use_ref!`, which no-ops for
    # it because there is no library to forward to). The arm is NOT in `_REF_ALL`, so `_VIEWS` still
    # renders exactly the two reference views and no group gains a spurious third.
    #
    # The second, stricter bar is DL1 vs CL1 on IDENTICAL BYTES (`Dual{_,Float64,1}` and `ComplexF64`
    # are both 16 B interleaved pairs — docs/src/dual.md). That needs no third arm: it is computable
    # from the cached CL1 pb cells at the same n, and `coverage_ops.jl` derives it.
    #
    # `nrm2` is the cell that motivated the work: the generic path takes the Dual `lassq` loop with a
    # division and branches per element, flat at ~4.8 GB/s (13-16x off the complex twin) before the
    # dupEven kernel landed. `dot` here is `dotu` semantics; for Dual, `dotc == dotu` because
    # `Dual <: Real` and ForwardDiff defines no `conj`.
    dl1 = OpData[]
    let
        D = ForwardDiff.Dual{Nothing, Float64, 1}
        mkd(s) = (D[ForwardDiff.Dual{Nothing}(randn(), randn()) for _ in 1:s],
                  D[ForwardDiff.Dual{Nothing}(randn(), randn()) for _ in 1:s])
        ad = ForwardDiff.Dual{Nothing}(1.7, 0.3)      # a DUAL alpha: exercises the tagged kernel, not
        #                                               the real-alpha bypass (docs/src/dual.md)
        for (nm, ob, pb) in (
                ("daxpy1", (c, m) -> (for _ in 1:m; LinearAlgebra.axpy!(ad, c[1], c[2]); end; c[2][1].value),
                    (c, m) -> (for _ in 1:m; PureBLAS.axpy!(c[2], ad, c[1]); end; c[2][1].value)),
                ("dscal1", (c, m) -> (for _ in 1:m; LinearAlgebra.rmul!(c[1], ad); end; c[1][1].value),
                    (c, m) -> (for _ in 1:m; PureBLAS.scal!(ad, c[1]); end; c[1][1].value)),
                ("ddot1", (c, m) -> (s = zero(D); for _ in 1:m; s += LinearAlgebra.dot(c[1], c[2]); end; s.value),
                    (c, m) -> (s = zero(D); for _ in 1:m; s += PureBLAS.dot(c[1], c[2]); end; s.value)),
                ("dnrm21", (c, m) -> (s = zero(D); for _ in 1:m; s += LinearAlgebra.norm(c[1]); end; s.value),
                    (c, m) -> (s = zero(D); for _ in 1:m; s += PureBLAS.nrm2(c[1]); end; s.value)),
                ("dasum1", (c, m) -> (s = zero(D); for _ in 1:m; s += sum(abs, c[1]); end; s.value),
                    (c, m) -> (s = zero(D); for _ in 1:m; s += PureBLAS.asum(c[1]); end; s.value)),
                # `findmax(abs, x)[2]`, NOT `argmax(abs.(x))`. The dotted form builds a full n-element
                # temporary before it searches — measured 160072 B at n=10000, against 0 B for this one,
                # and both return the same index (6059, which is also what `PureBLAS.iamax` returns).
                # Timing the reference's ALLOCATION and calling the difference our speedup would inflate
                # this row and nothing else: its sibling `dasum1` already uses the 2-arg `sum(abs, x)`.
                # A reference arm has to be the fastest honest way to write the operation, or the ratio
                # measures our kernel against a strawman.
                ("diamax1", (c, m) -> (s = 0; for _ in 1:m; s += findmax(abs, c[1])[2]; end; s),
                    (c, m) -> (s = 0; for _ in 1:m; s += PureBLAS.iamax(c[1]); end; s)),
            )
            _meas!(dl1, "DL1", nm, () -> sweep(mkd, _sizes(L1SZ), ob, pb, _L1REP; refs = ["generic"]))
        end
    end

    # ── DL2 / DL3: BLAS-2 and BLAS-3 over Dual ───────────────────────────────────────────────────────
    # Same contract as DL1: the reference is LinearAlgebra's GENERIC fallback over `Dual`, because no
    # vendor BLAS has a dual path. See the DL1 note above for why `generic` is not in `_REF_ALL`.
    #
    # THE REFERENCE MUST BE THE FASTEST HONEST WAY TO WRITE THE OP, not the most obvious one — the
    # `diamax1` row above measured `argmax(abs.(x))`, whose dotted temporary cost 160 KB a call at
    # n=1e4, and the ratio flattered us by exactly that. So every reference below is the in-place
    # 5-argument `mul!` / `lmul!` / `ldiv!` form, which is what LinearAlgebra actually dispatches a
    # `Dual` matrix through and allocates nothing per call.
    #
    # L3 is capped at 2048 like CL3, and for a second reason beyond CL3's "10 min for little signal":
    # the planar route holds SIX real planes of scratch, so 4096² F64 is ~805 MB (docs/src/dual_l3.md).
    # That is the same order as real Strassen's level scratch at that size, but it is not a footprint
    # to put in a default sweep.
    dl2 = OpData[]
    dl3 = OpData[]
    let
        D = ForwardDiff.Dual{Nothing, Float64, 1}
        rd() = ForwardDiff.Dual{Nothing}(randn(), randn())
        dvec(n) = D[rd() for _ in 1:n]
        dmat(m, n) = D[rd() for _ in 1:m, _ in 1:n]
        # unit-ish diagonal so trsv/trsm are well conditioned, mirroring `tri`/`ctri`
        dtri(s) = (A = dmat(s, s) ./ (2s); for i in 1:s
                A[i, i] = ForwardDiff.Dual{Nothing}(1 + abs(ForwardDiff.value(A[i, i])), randn())
            end; A)
        dsq(s) = (dmat(s, s), dvec(s), dvec(s))
        dtriv(s) = (dtri(s), dvec(s), dvec(s))

        add2(nm, mk, ob, pb) = _meas!(dl2, "DL2", nm, () -> sweep(mk, _sizes(L2SZ), ob, pb, _L2REP; refs = ["generic"]))
        add2("dgemvN1", dsq,
            (c, m) -> (for _ in 1:m; LinearAlgebra.mul!(c[3], c[1], c[2]); end; c[3][1].value),
            (c, m) -> (for _ in 1:m; PureBLAS.gemv!(c[3], c[1], c[2]); end; c[3][1].value))
        add2("dgemvT1", dsq,
            (c, m) -> (for _ in 1:m; LinearAlgebra.mul!(c[3], transpose(c[1]), c[2]); end; c[3][1].value),
            (c, m) -> (for _ in 1:m; PureBLAS.gemv!(c[3], c[1], c[2]; trans = TT); end; c[3][1].value))
        add2("dger1", dsq,
            (c, m) -> (for _ in 1:m; LinearAlgebra.mul!(c[1], c[2], transpose(c[3]), true, true); end; c[1][1].value),
            (c, m) -> (for _ in 1:m; PureBLAS.ger!(one(D), c[2], c[3], c[1]); end; c[1][1].value))
        add2("dtrmv1", dtriv,
            (c, m) -> (for _ in 1:m; copyto!(c[3], c[2]); LinearAlgebra.lmul!(UpperTriangular(c[1]), c[3]); end; c[3][1].value),
            (c, m) -> (for _ in 1:m; copyto!(c[3], c[2]); PureBLAS.trmv!(c[1], c[3]; uplo = U); end; c[3][1].value))
        add2("dtrsv1", dtriv,
            (c, m) -> (for _ in 1:m; copyto!(c[3], c[2]); LinearAlgebra.ldiv!(UpperTriangular(c[1]), c[3]); end; c[3][1].value),
            (c, m) -> (for _ in 1:m; copyto!(c[3], c[2]); PureBLAS.trsv!(c[1], c[3]; uplo = U); end; c[3][1].value))

        add3(nm, mk, ob, pb) = _meas!(dl3, "DL3", nm,
            () -> sweep_heavy(mk, ob, pb, _sizes(_cap(L3SZ, 2048)); refs = ["generic"]))
        add3("dgemm1", s -> (dmat(s, s), dmat(s, s), zeros(D, s, s)),
            c -> (LinearAlgebra.mul!(c[3], c[1], c[2]); c[3][1].value),
            c -> (PureBLAS.gemm!(c[3], c[1], c[2]); c[3][1].value))
        add3("dsyrk1", s -> (dmat(s, s), zeros(D, s, s)),
            c -> (LinearAlgebra.mul!(c[2], c[1], transpose(c[1])); c[2][1].value),
            c -> (PureBLAS.syrk!(c[2], c[1]; uplo = U, trans = TN, alpha = one(D), beta = zero(D)); c[2][1].value))
        add3("dtrmm1", s -> (dtri(s), dmat(s, s)),
            c -> (LinearAlgebra.lmul!(UpperTriangular(c[1]), c[2]); c[2][1].value),
            c -> (PureBLAS.trmm!(c[2], c[1]; side = Char(76), uplo = U); c[2][1].value))
        add3("dtrsm1", s -> (dtri(s), dmat(s, s)),
            c -> (LinearAlgebra.ldiv!(UpperTriangular(c[1]), c[2]); c[2][1].value),
            c -> (PureBLAS.trsm!(c[2], c[1]; side = Char(76), uplo = U); c[2][1].value))
    end

    # ── DLP: LAPACK over Dual ────────────────────────────────────────────────────────────────────────
    # Same contract as DL1/DL2/DL3: the reference is LinearAlgebra's GENERIC factorization, measured in
    # the SAME run as the pb arm. See the DL1 note for why `generic` is not in `_REF_ALL`.
    #
    # WHY THIS GROUP HAS TO EXIST, stated plainly because it did not until now. The dual LAPACK
    # speedups were found by a throwaway probe on ONE box: potri 21.0x, trtri 20.8x, potrf 7.2x,
    # getrf 5.7x over the generic factorization. None of that was in the gate, so nothing was watching
    # it — and `geqrf` is the proof that matters: it sat at 0.45x (WORSE than generic, degrading with n)
    # for an unknown length of time, in a routine that had a green LP cell the whole while, because the
    # LP cell measures Float64 and no cell measured a dual. A group whose numbers live only in a probe
    # is a group that can regress silently.
    #
    # LPSZ is capped at 512: these are O(n^3) on a 16-byte type against a GENERIC reference that is
    # itself O(n^3) and slow, so n=1024+ is minutes per cell for no extra signal (the ratios are already
    # monotone by 256). `pstrf`/`gesvd`/`_syev!` are included deliberately — they branch on a
    # comparison, so a cell on them is also a regression test for the seed-independence property
    # (docs/src/dual_lp.md §6).
    dlp = OpData[]
    let
        Dd = ForwardDiff.Dual{Nothing, Float64, 1}
        rdd() = ForwardDiff.Dual{Nothing}(randn(), randn())
        dm(m, n) = Dd[rdd() for _ in 1:m, _ in 1:n]
        dv(n) = Dd[rdd() for _ in 1:n]
        dspd(s) = (A = dm(s, s); A = A * transpose(A); for i in 1:s
                A[i, i] += s
            end; A)
        dgen(s) = (A = dm(s, s); for i in 1:s
                A[i, i] += s
            end; A)
        # SIZES AND WINDOW BUDGET, both measured rather than copied from LP.
        #
        # LPSZ runs to 4096. DLP cannot: its REFERENCE is LinearAlgebra's generic factorization on a
        # 16-byte non-BlasFloat (`eigvals`/`qr`/`lu` with no BLAS underneath), so the reference arm is
        # itself O(n^3) with a large constant. Measured on wintermute: the whole group at n=128 took
        # **14.5 minutes**, which extrapolates to hours at 256 and many hours at 512 — DLP alone would
        # have dominated a full-fleet refresh that otherwise costs about an hour a box.
        #
        # Capped at 256, and the cap costs nothing worth having: the triage ratios are already monotone
        # well before it (potri 8.42 -> 21.03, trtri 9.79 -> 20.84 over n=64 -> 256), so a 512 cell would
        # buy a number nobody acts on at several hours a box.
        #
        # The window budget is cut because the measurement is stable, not to save precision: at n=128
        # `dsyev1` read 19934/20029/20038/20030/20044/20062/20045/20040 us across 8 rounds — a 0.6%
        # spread. Chairmarks was spending 4-second windows and 24 samples resolving something that is
        # already flat to well under 1%, i.e. the cost was WINDOW-bound, not precision-bound. 10 samples
        # in 1.5 s keeps far more resolution than the gate needs (`gate_pass` rounds to two significant
        # digits) at a fraction of the wall clock.
        DLPSZ = (32, 50, 100, 128, 256)
        addp(nm, mk, ob, pb) = _meas!(dlp, "DLP", nm,
            () -> sweep_heavy(mk, ob, pb, _sizes(DLPSZ); samples = 10, seconds = 1.5, refs = ["generic"]))

        addp("dpotrf1", dspd,
            c -> (cholesky(Symmetric(c, :L)); c[1, 1].value),
            c -> (PureBLAS.potrf!(c; uplo = 'L'); c[1, 1].value))
        addp("dgetrf1", dgen,
            c -> (lu(c); c[1, 1].value),
            c -> (PureBLAS.getrf!(c, zeros(Int, size(c, 1))); c[1, 1].value))
        addp("dgeqrf1", s -> dm(s, s),
            c -> (qr(c); c[1, 1].value),
            c -> (PureBLAS.geqrf!(c, dv(size(c, 1))); c[1, 1].value))
        addp("dtrtri1", dgen,
            c -> (inv(UpperTriangular(c)); c[1, 1].value),
            c -> (PureBLAS.trtri!(c; uplo = 'U', diag = 'N'); c[1, 1].value))
        addp("dpotri1", dspd,
            c -> (inv(Symmetric(c, :L)); c[1, 1].value),
            c -> (PureBLAS.potrf!(c; uplo = 'L'); PureBLAS.potri!(c; uplo = 'L'); c[1, 1].value))
        addp("dsyev1", dspd,
            c -> (eigvals(Symmetric(c, :L)); c[1, 1].value),
            c -> (PureBLAS._syev!('N', 'L', c); c[1, 1].value))
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
    return cl1, cl2, cl3, clp, dl1, dl2, dl3, dlp
end

# ── cache: one line per op  «level⟶TAB⟶name⟶TAB⟶ s1=r,r,…;s2=r,r,… » ─────────────────────────────
# v3: ONE cache per host holds EVERY arm (no _aocl/_mkl split); the reference suffix now applies only to
# the rendered views.
#
# ── EXCEPT the pb_mt arm, which gets ITS OWN FILE, derived — not a new argument ──────────────────────
# WHY THE GATE CACHE MUST NOT RECEIVE IT. `fleet_refresh.sh` pins the gate sweep to ONE core, on purpose:
# an unpinned sweep let the scheduler migrate mid-measurement, and on galen (two 32 MiB L3s) a sweep
# sharing a die with a GPU job moved 3.7% of cells. The `pb_mt` arm needs SIX cores, so `pb` and `pb_mt`
# cannot share one invocation's affinity. Merging them into the gate cache would therefore REWRITE every
# `pb` gate cell with a measurement taken at 6-core affinity instead of the sanctioned 1-core pinning —
# changing the published gate's methodology silently, which is exactly the class of drift this file's
# provenance machinery exists to make impossible.
#
# So the mt run writes `mt_data_…`, and that name is DELIBERATELY outside the `plots_data_*` glob that
# fifteen scripts and `load_fleet()` use. It cannot leak into a rendered view, an artifact check, a
# staleness audit or a gate verdict, because none of them can see it.
#
# Derived from `_DO_PB_MT` rather than taken as a `cache=` path: a path argument can be typo'd into the
# gate cache, and the one thing this must guarantee is that asking for pb_mt can never overwrite the gate.
const CACHE = joinpath(@__DIR__, _ANY_MT ?
    "mt_data_$(SLUG)_$(gethostname())$(_LITE ? "_lite" : "").txt" :
    "plots_data_$(SLUG)_$(gethostname())$(_LITE ? "_lite" : "").txt")
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

# LOCK STATE lives in bench/freqlock.jl — THE single source of truth, shared with the calibrator.
# `base=`/`boost=` in the header make `freq=` self-interpreting: freq/base ~ 1.0 is a locked run and a
# ratio well above 1 is a boosting one, WITHOUT going back to the machine to look up its base clock.
_lock_state() = FreqLock.lock_state()
_require_lock() = FreqLock.require_lock(what = "measure a gate sweep")

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
    # INSTANTANEOUS cpu, sampled over an interval — NOT `ps pcpu`, and NOT `ps cputimes`.
    #
    # `ps -eo pcpu` reports CPU averaged over the process's ENTIRE LIFETIME, so a browser tab that
    # burned a core for two minutes and then went idle keeps reporting ~35% for as long as it lives.
    # That false-positive refused a whole group run (CLP on neuromancer, 49 cells) on an idle box:
    # ps said 34.9%, a 5-second sample of the same pid said 0.4%.
    #
    # The obvious fix — diffing `ps -eo cputimes` — is ALSO wrong and worse: cputimes is INTEGER
    # SECONDS, so across a sub-second window a 100%-busy process accumulates 0.4 s and reports a
    # delta of 0. That silently disables the guard rather than merely over-firing it. Caught by a
    # null test that span a real CPU hog and watched the guard fail to see it.
    #
    # /proc/<pid>/stat gives utime+stime in CLOCK TICKS (~10 ms), which resolves a 25% threshold over
    # a 0.4 s window comfortably. Non-Linux falls back to returning `nothing` = "unknown, not clear",
    # which is the existing conservative behaviour when ps is unavailable.
    isdir("/proc") || return nothing
    hz = 100.0                                        # USER_HZ is 100 on every Linux target here
    snap() = begin
        d = Dict{Int, Tuple{Int, Float64}}()
        for e in readdir("/proc")
            pid = tryparse(Int, e); isnothing(pid) && continue
            st = try
                read(joinpath("/proc", e, "stat"), String)
            catch
                continue                              # process exited between readdir and read
            end
            k = findlast(')', st); isnothing(k) && continue     # comm can contain spaces/parens
            fs = split(SubString(st, k + 2))
            length(fs) < 22 && continue
            ppid = tryparse(Int, fs[2]); isnothing(ppid) && continue
            ut = tryparse(Float64, fs[12]); stt = tryparse(Float64, fs[13])
            (isnothing(ut) || isnothing(stt)) && continue
            d[pid] = (ppid, (ut + stt) / hz)
        end
        d
    end
    a = snap(); t0 = time(); sleep(0.4); b = snap(); dt = max(time() - t0, 1.0e-3)
    me = getpid()
    busy = Tuple{Int, Float64, String}[]
    for (pid, (ppid, c1)) in b
        (pid == me || ppid == me) && continue          # us, and anything we spawned
        haskey(a, pid) || continue                     # started mid-sample: no interval to measure
        pc = (c1 - a[pid][2]) / dt * 100
        pc >= _BUSY_PCPU || continue
        args = try
            replace(read(joinpath("/proc", string(pid), "cmdline"), String), '\0' => ' ')
        catch
            "?"
        end
        push!(busy, (pid, pc, strip(args)))
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

# The lock's EXIT check, mirroring the contention pair above. A start-only guard is not enough: on
# 2026-08-23 a Zen5 sweep began under a valid lock and the lock came off DURING the run, so every cell
# after that point was invalid while the run reported nothing. Re-reading the state at exit and
# comparing it to the state at entry turns "not my fault" into "detected and recorded".
# set on the measure path only: a plot-only render must never refuse on the live clock, since it is
# reporting cells measured earlier under whatever state THEY record.
const _MEASURED_ANYTHING = Ref(false)
_LOCK_AT_START = (0, 0, -1)
_LOCK_CHANGED = ""
function _lock_exit_check()
    now = _lock_state()
    now == _LOCK_AT_START && return nothing
    global _LOCK_CHANGED = "start=$(_LOCK_AT_START)->end=$(now)"
    @warn "FREQUENCY LOCK CHANGED DURING THE RUN — it was valid at the start, so this happened mid \
        measurement. Every cell this run touched is INVALID by the frequency methodology; discard them \
        and re-measure after `sudo bench/fleet_freqlock.sh lock`. The cache header records it as \
        `lockchg=`.\n  $(_LOCK_CHANGED)"
    return nothing
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

# A `PUREBLAS_FORCE_<knob>` run measures a DELIBERATELY NON-DEFAULT PureBLAS — it is an A/B probe, not
# a gate measurement — so its numbers must never reach the cache. They did, and it cost three phantom
# gate misses.
#
# On 2026-08-09 the Zen5 cache held pb=0.0498 s for L2/gemvT@256 against aocl=0.0367, published as a
# 0.747 miss and treated as the campaign's hardest open cell. A controlled re-measure read 0.0321 s
# (ratio 1.089, PASS), and `PUREBLAS_FORCE_gemvt_deep=0` reproduced 0.0500 s exactly: the record had
# been written by an A/B round with that variable exported. gemvT@128 and trmv@256 went the same way.
# A whole kb finding, a retraction, and a day of kernel work were built on top of it.
#
# `tune=` (below) was added for the untracked-LocalPreferences version of this bug and does NOT catch
# it — the env hooks are resolved per knob and `gemvt_deep` was not among the values stamped. Rather
# than chase the knob list, refuse the write: every forced knob is covered, including ones not yet
# written. The A/B loses nothing, because the per-round output it is read from is printed either way.
_forced_knobs() = sort([k for k in keys(ENV) if startswith(k, "PUREBLAS_FORCE_")])

# ── ACTIVE PREFERENCES: the same hazard as PUREBLAS_FORCE_*, but persistent and invisible ────────────
# A Preference pin overrides a Measure-tier auto-tune permanently, and `LocalPreferences.toml` is
# GITIGNORED — so it shows in no diff, no review, and survives every `fleet_sync.sh` hard reset.
#
# This has now bitten twice, the second time for ten days after the first was "fixed":
#   2026-07-30  `bench/LocalPreferences.toml` pinning `ger_panel_np = 1` found on wintermute AND galen.
#               Removing it took wintermute ger n=2048 from 0.914 to 1.244.
#   2026-08-09  the SAME pin was still live on neuromancer — never checked in that cleanup. Ten days of
#               Zen5 ger/zgeru numbers went out through it, it falsified two source comments and a whole
#               task premise ("Zen5 resolves _ger_np() = 1" — unpinned the box measures 8), and it
#               laundered itself into a kb finding that later work then cited as a documented constant.
#
# `tune=` stamps the RESOLVED values, which is necessary but not sufficient: it cannot distinguish "8
# because the tuner measured 8" from "8 because someone pinned it", and nobody diffs a header they have
# no reason to suspect. So: enumerate the pins themselves and say so, loudly, at the top of every run.
# Reported rather than refused — a pin is legitimate for calibration, and the trim build REQUIRES pins
# for every Measure-tier knob. What is never legitimate is not knowing.
# ⚠ DETECTION MUST USE THE RESOLVER, NOT A GUESS AT WHICH FILE. A first version of this walked the
# ACTIVE PROJECT's LocalPreferences.toml/Project.toml only. That is not where Julia looks: Base's
# `get_preferences` (loading.jl) merges the active project's workspace-to-root chain PLUS every entry of
# `load_path()` — so under `--project=bench` a pin in `~/.julia/environments/v1.12/` is live and would
# have been invisible to the check, i.e. the identical failure mode one directory over.
# `Base.get_preferences(uuid)` returns exactly the merged set that the package will see; defaults live
# in code and never in TOML, so anything it returns IS a pin. The file walk survives only to attribute
# WHICH file, and is allowed to come up empty without weakening the verdict.
const _PB_UUID = Base.UUID("cc9e14db-574f-4602-bf53-1167cc4b26d2")

function _active_prefs()
    merged = try
        Base.get_preferences(_PB_UUID)
    catch e
        @warn "could not resolve preferences — treating as UNKNOWN, not as clear" exception = e
        return ["<preference resolution failed>"]
    end
    isempty(merged) && return String[]
    where = Dict{String, String}()
    for dir in Base.load_path()
        d = dirname(dir)
        for (f, sect) in (
                (joinpath(d, "LocalPreferences.toml"), "PureBLAS"),
                (joinpath(d, "Project.toml"), "preferences"),
            )
            isfile(f) || continue
            try
                t = TOML.parsefile(f)
                p = sect == "preferences" ? get(get(t, "preferences", Dict()), "PureBLAS", Dict()) :
                    get(t, "PureBLAS", Dict())
                p isa AbstractDict || continue
                for k in keys(p)
                    get!(where, String(k), f)
                end
            catch e
                # NEVER silent: a parse failure here is the one thing this tool exists to not miss.
                @warn "could not parse $f while attributing pins" exception = e
            end
        end
    end
    return sort(["$k = $v   ($(get(where, String(k), "source not located")))" for (k, v) in merged])
end

function _pref_check()
    p = _active_prefs()
    isempty(p) && return println("preference check: clear (no PureBLAS pin in $(dirname(Base.active_project())))")
    @warn "PINNED PREFERENCES ARE ACTIVE — this run does NOT measure the shipped defaults for these \
        knobs. A pin overrides its Measure-tier auto-tune silently and LocalPreferences.toml is \
        gitignored, so nothing else will tell you.\n  " * join(p, "\n  ") *
        "\nIf this is not deliberate calibration, clear them and re-run."
    return nothing
end

# ── IS THIS BOX LEGITIMATELY TUNED? (user ruling 2026-08-29: the gate is the TUNED machine) ──────────
# Before this, ANY active pin refused the write, which meant a box that had run `PureBLAS.tune!()` could
# not publish a gate number at all — the knob defaults are a fleet-wide MINIMAX (e.g. `ger_panel_np = 1`
# is chosen so Zen5 passes; wintermute's own optimum is 8), so the shipped default is by construction not
# what any single machine should run.
#
# The carve-out is deliberately NARROW, because the blanket refusal exists for a real incident: an
# untracked pin of `ger_panel_np = 1` sandbagged every published ger number for ten days (see the `tune=`
# note below). Both conditions must hold:
#   1. EVERY active pin is a key `tune!()` itself owns (`_TUNABLE_KEYS`), plus its own `tuned_for` stamp.
#      A hand-written pin, a leftover, or a key from some other experiment still refuses.
#   2. `is_tuned()` — the stored `tuned_for` fingerprint matches the DETECTED hardware (src/tune.jl:43).
#      A depot copied between machines, or a BIOS cache change, makes the fingerprint stale and refuses.
# `PUREBLAS_FORCE_*` is NOT covered and still refuses unconditionally below: a forced arm is an A/B probe,
# never a gate number. And the header stamps `tuned=` so a published cell says out loud that it describes
# a tuned box rather than a fresh install.
function _tuned_pins_ok()
    merged = try
        Base.get_preferences(_PB_UUID)
    catch e
        return (false, "preference resolution failed ($e)")
    end
    isempty(merged) && return (false, "no pins")
    own = Set(String.(PureBLAS._TUNABLE_KEYS))
    stray = sort([String(k) for k in keys(merged) if String(k) != "tuned_for" && !(String(k) in own)])
    isempty(stray) || return (false, "pin(s) not owned by tune!(): " * join(stray, ", "))
    haskey(merged, "tuned_for") ||
        return (false, "pins present but no `tuned_for` stamp — not written by tune!()")
    PureBLAS.is_tuned() ||
        return (false, "`tuned_for` is STALE: it does not match this machine's detected hardware")
    return (true, "")
end
_tuned_stamp() = (
    t = try
        Base.get_preferences(_PB_UUID)
    catch
        Dict()
    end; get(t, "tuned_for", "")
)

function save_cache(path, groups)
    # A PIN IS THE SAME CONDITION AS A FORCE VAR, ONLY PERSISTENT — and it is the one that actually did
    # the damage, because the ten days of wrong Zen5 ger/zgeru numbers propagated through PUBLISHED
    # CACHES, which a start-of-run @warn scrolled away by a 3-hour sweep does not touch. The two
    # defences once offered for warn-not-refuse do not survive contact: calibration writes preferences,
    # it does not need to write the GATE cache; and the trim build does not run plots.jl at all.
    # `force-pins` is the deliberate escape, mirroring `force-busy`.
    pinned = _active_prefs()
    tuned_ok, tuned_why = _tuned_pins_ok()
    if !isempty(pinned) && !tuned_ok && !("force-pins" in ARGS)
        @warn "CACHE NOT WRITTEN — $(length(pinned)) PureBLAS preference pin(s) are active and this is \
            NOT a recognised tuned state ($tuned_why), so the PB arm is not a configuration this project \
            will stand behind:\n  $(join(pinned, "\n  "))\nEither clear the pins, or run \
            `PureBLAS.tune!()` so the pins are tune!()-owned and fingerprint-matched, or pass \
            `force-pins` if a pinned cache is genuinely intended (calibration)."
        return nothing
    end
    tuned_ok && @info "TUNED-STATE CACHE — every active pin is tune!()-owned and the `tuned_for` \
        fingerprint matches this machine, so this run is a valid gate measurement of the TUNED box \
        (user ruling 2026-08-29). The header records `tuned=` so readers know it is not a fresh install."
    forced = _forced_knobs()
    if !isempty(forced)
        @warn "CACHE NOT WRITTEN — $(length(forced)) PUREBLAS_FORCE_* variable(s) are set, so the PB \
            arm is not the shipped configuration:\n  $(join(("$k=$(ENV[k])" for k in forced), "\n  "))\n\
            The per-round numbers above are the A/B result; read them there. Unset the variable(s) to \
            produce a cacheable gate measurement."
        return nothing
    end
    # ── VALIDATE BEFORE TRUNCATING ────────────────────────────────────────────────────────────────
    # `open(path, "w")` truncates IMMEDIATELY, so any guard that throws from inside the do-block
    # destroys the cache it was protecting. That happened on 2026-08-27: neuromancer drifted above its
    # pin, `check_achieved` correctly refused mid-write, and the 863-cell Zen5 cache — including the
    # openblas/aocl reference arms that the "never re-measure a reference" rule exists to protect —
    # was left at 0 bytes. Recovered only because an unrelated copy happened to be in /tmp.
    # So the freq check runs HERE, before the file is opened, and the write itself goes to a temp file
    # that is renamed over the target only after it closes cleanly. A crash mid-write now costs
    # nothing; the previous cache survives untouched.
    khz_pre = try
        _achieved_khz()
    catch
        0
    end
    # THE PIN IS NOT THE CLOCK — refuse a cache whose cells were measured above the pinned ceiling
    # even though the pin reads correct. `_require_lock` at entry cannot see this; only a sample taken
    # under load can. See FreqLock.check_achieved for the incident.
    _MEASURED_ANYTHING[] && FreqLock.check_achieved(khz_pre; what = "write a gate cache")
    tmppath = path * ".tmp$(getpid())"
    open(tmppath, "w") do io
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
        # (the freq guard now runs BEFORE the file is opened — see the note above `open(tmppath)`)
        println(
            io, "#pbbench\tversion=$(_BENCH_VERSION)\tslug=$SLUG\tuarch=$(_MYUARCH)\tisa=$ISA",
            "\thost=$(gethostname())\tcpu=$(_CPUNAME)\tcommit=$(_COMMIT)\ttime=$ts",
            # `julia=`/`llvm=`/`ob=` — the TOOLCHAIN. PureBLAS kernels are Julia source compiled by LLVM at
            # load time, so the LLVM version is as much an input to a measured number as the source commit
            # is; a toolchain bump moves every cell with no diff to explain it. `ob=` because Julia BUNDLES
            # OpenBLAS: upgrading Julia silently swaps the reference arm, which is why a toolchain bump is
            # the one case that REQUIRES re-measuring both arms instead of `arms=pb`.
            "\tjulia=$(VERSION)\tllvm=$(Base.libllvm_version)\tob=$(_obversion())",
            "\thw=$(_hwstamp())\ttune=$(_tunestamp())",
            # `base=`/`boost=` make `freq=` self-interpreting: freq/base ~ 1.0 is a locked run, and a
            # ratio well above 1 is a boosting one, WITHOUT going back to the machine to look up its
            # base clock. See `_lock_state`.
            "\tanchor=$(round(anc * 1.0e6; digits = 3))us\tfreq=$(khz)kHz",
            (ls = _lock_state(); "\tbase=$(ls[2])kHz\tboost=$(ls[3])"),
            isempty(_LOCK_CHANGED) ? "" : "\tlockchg=$(_LOCK_CHANGED)",
            isempty(_BUSY_AT_EXIT) ? "" : "\tbusy=$(_BUSY_AT_EXIT)",
            # `tuned=` — present ONLY when this cache describes a legitimately tuned box (see
            # `_tuned_pins_ok`). Its absence means the shipped defaults were measured. A reader comparing
            # two caches must diff this before diffing any ratio: a tuned and an untuned cache of the same
            # commit are measuring different configurations, which is exactly the confusion the blanket
            # pin-refusal used to prevent by making the tuned case impossible.
            (ts_ = _tuned_stamp(); isempty(ts_) ? "" : "\ttuned=$(ts_)")
        )
        # v3 record: ONE LINE PER CELL, one field per measured arm, each carrying its own timestamp and
        # commit. Per-arm provenance is what makes `arms=pb` safe to use: it rewrites only the pb field,
        # so the reference fields still say when they were actually measured and the table reports their
        # age rather than implying they are as fresh as the run that produced the page.
        #   <lvl> <op> <size> pb|<iso>|<commit>|t1,..,t48  openblas|<iso>|<commit>|t1,..  aocl|...
        # Times are SECONDS (the quantiles of that arm's sample times). Ratios are derived, never stored.
        for (lvl, d) in groups, (nm, op) in d, (s, cell) in op
            # SIX fields now: anchor, then the achieved clock, then the times. The cache VERSION is
            # deliberately NOT bumped — a bump refuses every existing cache, and re-measuring the
            # OpenBLAS/AOCL arms is exactly what the reference cache exists to prevent. Readers accept
            # 4-field (pre-anchor), 5-field (anchor) and 6-field (anchor+freq) records side by side in
            # one file; the csv is always the LAST field, so every reader splits and takes `p[end]`
            # rather than `limit = 4`. That invariant is the extension mechanism — append before the
            # csv, never after it.
            fields = [
                "$(a)|$(rec.time)|$(rec.commit)|$(isnan(rec.anchor) ? "" : round(rec.anchor * 1.0e6; digits = 3))|$(rec.freq == 0 ? "" : rec.freq)|$(rec.flo == 0 ? "" : rec.flo)|$(rec.fhi == 0 ? "" : rec.fhi)|$(join(rec.q, ","))"
                    for (a, rec) in sort!(collect(cell); by = first)
            ]
            println(io, lvl, "\t", nm, "\t", s, "\t", join(fields, "\t"))
        end
    end
    # Atomic publish: the previous cache is only replaced once the new one is complete on disk.
    # `mv` within the same directory is a rename(2), so there is no window where the target is
    # partial. If anything above threw, `tmppath` is left behind and the real cache is untouched.
    mv(tmppath, path; force = true)
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
            # `limit = 2`: `hw=` and `tune=` are themselves k=v lists, so their VALUES contain `=`. An
            # unlimited split yields >2 parts for those fields, the `length(p) == 2` filter drops them, and
            # they are written but unreadable — which is how `tune=` sat in every cache since 2026-08-06
            # without any tool able to read it back. Split on the FIRST `=` only.
            kv = Dict(String(p[1]) => String(p[2]) for p in (split(x, "=", limit = 2) for x in split(ln, "\t")[2:end]) if length(p) == 2)
            meta = (
                version = parse(Int, get(kv, "version", "1")), slug = get(kv, "slug", "?"),
                uarch = get(kv, "uarch", "?"), isa = get(kv, "isa", "?"), host = get(kv, "host", "?"),
                cpu = get(kv, "cpu", "?"), commit = get(kv, "commit", "?"), time = get(kv, "time", "?"),
                hw = get(kv, "hw", "?"), tune = get(kv, "tune", "?"), freq = get(kv, "freq", "?"),
            )
            continue
        end
        parts = split(ln, "\t")
        length(parts) >= 4 || continue                       # v2 line shape ⇒ let the version check below speak
        lvl, nm, ssz = String(parts[1]), String(parts[2]), parse(Int, parts[3])
        cell = CellData()
        for f in parts[4:end]
            # An EMPTY record field is a cell that was swept with no arm that applies to it, and the
            # row was still written with its separator. The dual groups do it by construction: DL*'s
            # reference is LinearAlgebra's generic fallback, not a BLAS library, so a run selecting
            # only reference arms records nothing for them. Skipping is what makes such a cache
            # readable at all — without it the split below indexes a one-element vector and every
            # threaded run on the host dies in `load_cache` rather than reporting a missing arm.
            isempty(f) && continue
            # 4-field (pre-anchor), 5-field (anchor) and 6-field (anchor+freq) records coexist in one
            # file — see the writer note. The times are ALWAYS last, so index from both ends rather than
            # assuming a field count; that invariant is what lets a field be appended without a version
            # bump, and it is why `freq` slots in BEFORE the csv rather than after it.
            p = split(f, "|")
            a, tstamp, cmt, csv = p[1], p[2], p[3], p[end]
            anc = length(p) >= 5 ? something(tryparse(Float64, p[4]), NaN) * 1.0e-6 : NaN
            khz = length(p) >= 6 ? something(tryparse(Int, p[5]), 0) : 0
            # 8-field records add the in-window clock range (flo,fhi). Older records have no range and
            # read as 0,0 = unknown — NOT as a 0 Hz clock, and not as "steady": a consumer must treat
            # unknown as "cannot verify the window was pinned", which is exactly what those cells are.
            flo = length(p) >= 8 ? something(tryparse(Int, p[6]), 0) : 0
            fhi = length(p) >= 8 ? something(tryparse(Int, p[7]), 0) : 0
            cell[String(a)] =
                ArmRec(String(tstamp), String(cmt), anc, khz, flo, fhi, parse.(Float64, split(csv, ",")))
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
# `prefix` selects WHICH family of caches to draw. It defaults to the gate caches, so every existing
# call is unchanged; `mt_data_` draws the multi-threaded sweep instead. The two families are separate
# files precisely so an mt run can never touch a gate cell (see `CACHE`), and this is the one place
# that needs to know both names.
function load_fleet(prefix::AbstractString = "plots_data_")
    fleet = Tuple{NamedTuple, Dict{String, Vector{OpData}}}[]
    for f in sort(readdir(@__DIR__))
        (startswith(f, prefix) && endswith(f, ".txt")) || continue
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
        haskey(cell, _ARM_PB) || continue
        if ref == _GATE_VIEW
            # THE GATE, drawn: per cell, divide by whichever reference is FASTER — i.e. the one with the
            # smaller PB/ref median — which is `max(OpenBLAS, AOCL)` from req#1 and exactly the reference
            # `bench/cellratios.jl` and the coverage table pick. Chosen PER SIZE, so one panel can switch
            # references along its x-axis; that is correct, because the gate does too. Without this view a
            # red coverage cell can be invisible on the OpenBLAS plot and only show on the AOCL one.
            best = nothing
            for r in _REF_ALL
                haskey(cell, r) || continue
                v = _ratio(cell[r].q, cell[_ARM_PB].q)
                (isnothing(best) || median(v) < median(best)) && (best = v)
            end
            isnothing(best) || push!(out, (s, best))
        elseif ref == _ARM_PB_MT
            # THE MT VIEW IS INVERTED RELATIVE TO EVERY OTHER ONE, on purpose. Elsewhere the numerator
            # is the REFERENCE and the denominator PureBLAS, so "higher is better" means PB is faster.
            # Here both arms are PureBLAS and the question is what threads BOUGHT, so the single-thread
            # arm is the numerator: pb / pb_mt. Same reading — above the line is a win — which is the
            # point; a plot that silently flipped its sense against its neighbours would be a trap.
            haskey(cell, _ARM_PB_MT) || continue
            push!(out, (s, _ratio(cell[_ARM_PB].q, cell[_ARM_PB_MT].q)))
        else
            haskey(cell, ref) || continue
            push!(out, (s, _ratio(cell[ref].q, cell[_ARM_PB].q)))
        end
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

function svg_panels(path, title, fleet, gk, ref::AbstractString = REFBK; only = nothing)
    ops = _opsin(fleet, gk)
    # `only` keeps a panel to the operations the reader came for. The mt panels use it: a curve
    # that is flat because nothing splits that routine is not a result, and a page of them buries
    # the ones that are.
    isnothing(only) || (ops = [o for o in ops if o in only])
    # A GROUP WITH NO CELLS STILL WRITES A FILE. This used to `return` silently, which breaks the docs
    # build rather than the plot: `docs/src/performance.md` references each panel by name, Documenter
    # downgrades a missing image to a WARNING, and then vitepress hard-fails on the unresolved import
    # ("Rollup failed to resolve import assets/perf_dl3.svg"). That took the Documentation workflow red
    # three times in one day — once per newly wired group — because a group is referenced from the
    # moment it is wired and measured only later. A placeholder makes the reference valid immediately
    # and says, on the published page, that the group is pending rather than absent.
    if isempty(ops)
        W, H = 640, 96
        open(path, "w") do io
            println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="sans-serif">""")
            println(io, """<rect width="$W" height="$H" fill="white"/>""")
            println(io, """<text x="$(W ÷ 2)" y="38" text-anchor="middle" font-size="16" font-weight="bold">$title</text>""")
            println(io, """<text x="$(W ÷ 2)" y="66" text-anchor="middle" font-size="13" fill="#666">not yet measured on any box in this fleet</text>""")
            println(io, "</svg>")
        end
        return
    end
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
            ps = _series(g, gk, op, ref); (isnothing(ps) || isempty(ps)) && continue
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
function gen_table(fleet, gkeys, ref::AbstractString = REFBK)
    io = IOBuffer()
    println(io, "| op | ", join((_ulabel(m) for (m, _) in fleet), " | "), " |")
    println(io, "|---|", repeat("---|", length(fleet)))
    for gk in gkeys, op in _opsin(fleet, gk)
        cells = String[]
        for (_, g) in fleet
            ps = _series(g, gk, op, ref)
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
    _require_lock()        # an off-lock run is INVALID, not merely noisy — refuse at second zero
    _MEASURED_ANYTHING[] = true
    global _LOCK_AT_START = _lock_state()
    _contention_check()
    _pref_check()          # pins are legitimate; not KNOWING about them is not
    l1, l2, l3, lp = run_benchmarks()
    cl1, cl2, cl3, clp, dl1, dl2, dl3, dlp = run_cmplx_benchmarks()
    _lock_exit_check()              # catches a lock that came off DURING the run
    _contention_exit_check()        # before save_cache — it stamps `busy=` into the header
    measured = Dict("L1" => l1, "L2" => l2, "L3" => l3, "LP" => lp, "CL1" => cl1, "CL2" => cl2, "CL3" => cl3, "CLP" => clp, "DL1" => dl1, "DL2" => dl2, "DL3" => dl3, "DLP" => dlp)
    # AN MT RUN MERGES WHENEVER ITS CACHE EXISTS, even as a FULL run, because the mt cache ACCUMULATES
    # ARMS ACROSS RUNS by design: `pb`/`pb_mt` come from one sweep and the threaded references from
    # another (they cannot share a launch — BLIS fixes its thread count at init, so the reference run
    # must be started with BLIS_NUM_THREADS set, which would also change what `pb` sees).
    #
    # Without this, `arms=openblas_mt,aocl_mt` as a full run would REPLACE the mt cache and delete the
    # pb/pb_mt arms already in it — the exact 2026-08-06 shape the refusal below was written for, just
    # aimed at the other cache. The comment there said "the mt cache has none by design"; that was true
    # when only pb/pb_mt existed and is not true any more.
    subset = !isnothing(_SELOP) || !isnothing(_SELGRP) || (_ANY_MT && isfile(CACHE))
    # A scoped run (op=/group=) against a SLUG that has never been cached at all is not a merge — there is
    # nothing to merge into and nothing previously-measured to lose. Fall through to the full-run branch
    # below, which just writes `measured` as-is; `_want` already restricted it to the requested op/group,
    # so the resulting cache is a legitimately PARTIAL first cache for a new box, expandable later by the
    # normal merge path once it exists. This is how a new box's first pass can be scoped in wall-clock
    # (fewer groups/ops, via `group=`/`op=`, or fewer sizes via `maxsize=`) without a full-coverage run.
    if subset && !isfile(CACHE)
        println(
            "no existing cache at $CACHE — this scoped run becomes the FIRST (partial) cache for slug " *
                "\"$SLUG\": other groups/ops read as unmeasured (\"–\") until a later op=/group= run adds them."
        )
        subset = false
    end
    if subset
        # subset re-measure: MERGE the measured op(s) into the existing (v2) cache, leaving the rest intact.
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
        # ⚠ NOT gated on `!isnothing(_ARMS_SEL)` any more. That gate was only safe while omitting
        # `arms=` MEANT "every arm"; now that the default is PB-only (see `_REF_ARMS`), a bare full
        # `bench` measures no reference and would sail past this check straight into the 2026-08-06
        # failure it was written for. Condition on what was ACTUALLY measured, never on how it was
        # asked for.
        # `_ANY_MT` is exempt, and ONLY reaches here when the mt cache does NOT yet exist — once it
        # does, the merge branch above takes the run instead, so nothing can be destroyed. A first mt
        # sweep writing a fresh file has nothing to lose, which is the case this exemption covers.
        if !_ANY_MT && !issubset(_REF_ALL, _ACTIVE_ARMS) && !("force-arms" in ARGS)
            error(
                """
                REFUSING to overwrite $CACHE with a partial arm set.
                  full run + arms=$(join(_ACTIVE_ARMS, ",")) would DROP: $(join(setdiff(_REF_ALL, _ACTIVE_ARMS), ", "))
                A full `bench` REPLACES the cache; only op=/group= merges per arm. Either
                  • add op=<op> or group=<LVL>  (merges, keeps the reference arms), or
                  • drop `arms=` to measure every arm (~3x longer), or
                  • pass `force-arms` if a pb-only cache really is intended.

                MEASURING THE $_ARM_PB_MT ARM: use group=<LVL> runs, one per group. `$_ARM_PB_MT` is
                additive — the per-arm merge writes it alongside the cached openblas/aocl records — but
                a FULL run cannot express that, because a full run replaces rather than merges. Do NOT
                reach for `force-arms` here: that is exactly the path that destroyed 2h45m of Zen5
                reference arms on 2026-08-06."""
            )
        end
        g = measured
    end
    save_cache(CACHE, [lvl => get(g, lvl, OpData[]) for lvl in ("L1", "L2", "L3", "LP", "CL1", "CL2", "CL3", "CLP", "DL1", "DL2", "DL3", "DLP")])
end

# `export=<path>` dumps per-size ratios (the exact numbers `svg_panels`/`gen_table` draw from -- calls
# the SAME `_series`/`gatestat`, no re-derivation) as JSON, for building a chart outside this file's own
# hand-rolled SVG renderer. One record per (level, op, size): {level, op, size, ob, acc} where ob/acc are
# the per-size median ratio (PB/reference) against whichever refs are in `_REF_ALL`.
let i = findfirst(a -> startswith(a, "export="), ARGS)
    if !isnothing(i)
        path = ARGS[i][8:end]
        io = IOBuffer()
        print(io, "[")
        first_rec = true
        for lvl in ("L1", "L2", "L3", "LP", "CL1", "CL2", "CL3", "CLP")
            for op in _opsin([(nothing, g)], lvl)
                sizes = Set{Int}()
                for ref in _REF_ALL
                    ps = _series(g, lvl, op, ref)
                    isnothing(ps) || for (s, _) in ps
                        push!(sizes, s)
                    end
                end
                for s in sort(collect(sizes))
                    vals = Dict{String, Union{Float64, Nothing}}()
                    for ref in _REF_ALL
                        ps = _series(g, lvl, op, ref)
                        cell = isnothing(ps) ? nothing : findfirst(p -> p[1] == s, ps)
                        vals[ref] = isnothing(cell) ? nothing : median(ps[cell][2])
                    end
                    first_rec || print(io, ",")
                    first_rec = false
                    print(
                        io, "{\"level\":\"", lvl, "\",\"op\":\"", op, "\",\"size\":", s,
                        (isnothing(get(vals, "openblas", nothing)) ? "" : ",\"openblas\":$(vals["openblas"])"),
                        (isnothing(get(vals, "accelerate", nothing)) ? "" : ",\"accelerate\":$(vals["accelerate"])"),
                        (isnothing(get(vals, "aocl", nothing)) ? "" : ",\"aocl\":$(vals["aocl"])"), "}"
                    )
                end
            end
        end
        print(io, "]")
        write(path, String(take!(io)))
        println("wrote ", path)
    end
end

adir = isnothing(_OUTDIR) ? joinpath(@__DIR__, "..", "docs", "src", "assets") : _OUTDIR; mkpath(adir)
tdir = isnothing(_OUTDIR) ? (@__DIR__) : _OUTDIR
# Draw the whole FLEET (every host cache on disk) as cross-µarch panel grids: 8 SVGs, NO per-host suffix
# (a 3-line panel IS the per-host view). One SVG per group. `nodraw` skips this (fleet boxes measure only).
# ── DRAW THE MULTI-THREADED SWEEP AND STOP ──────────────────────────────────────────────────────────
# `mtdraw` is a RENDER-ONLY mode over the `mt_data_*` caches: it measures nothing, touches no gate
# artifact, and writes its own `perf_mt_*.svg` set. It exits before the gate rendering below, so an mt
# render can never overwrite a gate SVG even by accident — which matters because the gate artifacts are
# byte-compared against a fresh rebuild by `check_artifacts_current.sh`.
#
#   julia --project=bench bench/plots.jl mtdraw
if "mtdraw" in ARGS
    mtfleet = load_fleet("mt_data_")
    if isempty(mtfleet)
        println("no mt_data_* caches on disk — run: … bench/plots.jl bench arms=pb,pb_mt")
    else
        adir0 = isnothing(_OUTDIR) ? joinpath(@__DIR__, "..", "docs", "src", "assets") : _OUTDIR
        mkpath(adir0)
        # ONLY THE GROUPS AND OPERATIONS THAT THREAD. A routine with no splitter draws a flat line
        # at 1.00, which reports the harness rather than the library; eight panels of them hid the
        # six curves worth reading. `_MT_PLOT_MIN` is the harness noise floor measured on the cells
        # whose true answer is known to be 1.00 (`NOISE_P99`, bench/gen_threading.jl), so a curve that
        # appears here is threaded rather than merely noisy.
        _MT_PLOT_MIN = 1.25
        function _movers(fleet, gk)
            keep = String[]
            for op in _opsin(fleet, gk)
                best = 0.0
                for (_, g) in fleet
                    s = _series(g, gk, op, _ARM_PB_MT)
                    isnothing(s) && continue
                    for (_, rs) in s
                        isempty(rs) || (best = max(best, median(rs)))
                    end
                end
                best >= _MT_PLOT_MIN && push!(keep, op)
            end
            return keep
        end
        for (gk, base, ttl) in (("L3", "l3", "BLAS-3"), ("LP", "lapack", "LAPACK"))
            p = joinpath(adir0, "perf_mt_$(base).svg")
            svg_panels(p, "$ttl — PureBLAS 6 threads / 1 thread", mtfleet, gk, _ARM_PB_MT;
                only = _movers(mtfleet, gk))
            println("  ", relpath(p))
        end
        println("mt panels written — scaling curves for the routines that thread.")
    end
    exit(0)
end

fleet = _NODRAW ? [] : load_fleet()
if isempty(fleet)
    println("no fleet caches on disk to plot")
else
    L = _LITE ? "_lite" : ""
    for rb in _VIEWS                          # BOTH views, one invocation — see `_VIEWS`
        ref = _refname(rb); suf = _refsuf(rb)
        for (gk, base, ttl) in (
                ("L1", "l1", "BLAS-1"), ("L2", "l2", "BLAS-2"), ("L3", "l3", "BLAS-3"),
                ("LP", "lapack", "LAPACK"), ("CL1", "cl1", "Complex BLAS-1"),
                ("CL2", "cl2", "Complex BLAS-2"), ("CL3", "cl3", "Complex BLAS-3"),
                ("CLP", "clapack", "Complex LAPACK"),
            )
            svg_panels(joinpath(adir, "perf_$(base)$(suf)$L.svg"), "$ttl — PB / $ref ratio", fleet, gk, rb)
        end
        # THE TABLE IS THE DATA — nothing else. One pointer line, then the numbers. Provenance used to
        # sit inline as a per-box bullet list at the top of both tables; it now lives in ONE generated
        # file (below) that both views share, so it cannot be half-refreshed either.
        open(joinpath(tdir, "gen_table$(suf)$L.md"), "w") do io   # drift-proof numeric table: median (worst-cell) per op/µarch
            println(
                io, "PB / $ref speed ratio, median (worst cell) per op per µarch. ",
                "Provenance: [`bench/provenance.md`](provenance.md)."
            )
            println(io, "\n### Real\n\n", gen_table(fleet, ["L1", "L2", "L3", "LP"], rb))
            println(io, "\n### Complex\n\n", gen_table(fleet, ["CL1", "CL2", "CL3", "CLP"], rb))
            # `"generic"`, NOT `rb`. DL1 has no openblas or aocl arm to divide by — `Dual` is not a
            # `BlasFloat`, so LinearAlgebra never reaches a vendor BLAS (see `_use_ref!`). Passing `rb`
            # here asks for an arm that does not exist and renders an EMPTY Dual section in both views.
            # The section is therefore identical in the OpenBLAS and AOCL files, which is correct: there
            # is only one Dual denominator, so there is only one Dual table.
            println(io, "\n### Dual (ForwardDiff, N=1) — reference is LinearAlgebra generic, NOT a vendor BLAS\n\n", gen_table(fleet, ["DL1", "DL2", "DL3", "DLP"], "generic"))
        end
        println("wrote gen_table$(suf)$L.md  (fleet: ", join((m.slug for (m, _) in fleet), ", "), ")")
    end
    # THE GATE VIEW — rendered once per real/complex group, outside the reference loop, for the same reason
    # DL1 is below: it is not a reference arm, so rendering it inside `_VIEWS` would draw it twice. It exists
    # because the coverage table reports `max(OpenBLAS, AOCL)` while each reference plot shows one library,
    # so a red table cell could be invisible on the plot a reader looks at first (zgetrf n=50 on Zen4:
    # 1.39 on the OpenBLAS plot, 0.83 on the AOCL plot, 0.83 in the table). MKL runs have one reference,
    # where this view would only duplicate the MKL plot, so it is skipped there.
    if REFBK != "mkl"
        for (gk, base, ttl) in (
                ("L1", "l1", "BLAS-1"), ("L2", "l2", "BLAS-2"), ("L3", "l3", "BLAS-3"),
                ("LP", "lapack", "LAPACK"), ("CL1", "cl1", "Complex BLAS-1"),
                ("CL2", "cl2", "Complex BLAS-2"), ("CL3", "cl3", "Complex BLAS-3"),
                ("CLP", "clapack", "Complex LAPACK"),
            )
            svg_panels(joinpath(adir, "perf_$(base)$(_refsuf(_GATE_VIEW))$L.svg"),
                "$ttl — PB / $(_refname(_GATE_VIEW)) (the gate)", fleet, gk, _GATE_VIEW)
        end
    end
    # DL1 IS RENDERED ONCE, OUTSIDE THE VIEW LOOP, AND THAT IS THE WHOLE POINT. `_VIEWS` is the two
    # vendor references, and every other group legitimately has one panel per view. DL1 has neither
    # arm: its reference is LinearAlgebra's generic fallback over `Dual`, recorded as `generic`. Left
    # inside the loop it drew `perf_dl1.svg` against `openblas` and `perf_dl1_aocl.svg` against `aocl`
    # — two files, both empty, both looking like a measured group with no gap. One file, one honest
    # denominator. `_refsuf("generic")` is deliberately NOT used in the name: there is only ever one
    # Dual panel, so it needs no view suffix to disambiguate it from a sibling that does not exist.
    for (gk, base, ttl) in (
            ("DL1", "dl1", "Dual BLAS-1"), ("DL2", "dl2", "Dual BLAS-2"), ("DL3", "dl3", "Dual BLAS-3"), ("DLP", "dlp", "Dual LAPACK"),
        )
        svg_panels(
            joinpath(adir, "perf_$(base)$L.svg"),
            "$ttl (forward-mode AD) — PB / $(_refname("generic")) ratio", fleet, gk, "generic"
        )
    end
    # Provenance for EVERY artifact this invocation writes (both views come from the same caches, so it
    # is written once, outside the view loop — two copies could disagree, one cannot).
    open(joinpath(tdir, "provenance$L.md"), "w") do io
        println(io, "# Benchmark provenance\n")
        println(
            io, "The caches behind `bench/gen_table*.md`, `docs/src/assets/perf_*.svg` and the ",
            "generated tables in `docs/src/coverage.md`. Both reference views (OpenBLAS, AOCL) are ",
            "rendered from this one cache set. Methodology: `docs/src/methodology.md`.\n"
        )
        # Host name deliberately not published — µarch + CPU model identify the box for a reader, and
        # the cache header keeps `host=` for local fleet tooling.
        println(io, "| µarch | CPU | commit | measured |")
        println(io, "|---|---|---|---|")
        for (m, _) in fleet
            println(io, "| $(_ulabel(m)) | $(m.cpu) | `$(m.commit)` | $(m.time) |")
        end
    end
    println("wrote provenance$L.md")
end
# Per-cell anchor agreement required before a cross-run ratio is believed. 2% is the band the fleet's
# own cached arms sit in (60-67% of cells on 2026-08-22); above it the pb and reference arms saw
# different machine state and the ratio between them is not a measurement of the code.
const _ADJ_TOL = 0.02

# Gate summary for THIS host. v3 holds every arm in one cache, so this is the FIRST version that can
# state the project's actual rule — PB ≥ max(OpenBLAS, AOCL) — from a single run, per cell, instead of
# eyeballing two separately-measured tables. `gate` is the worst over cells of min over references,
# i.e. the margin against whichever reference is faster at each individual size.
for lvl in ("L1", "L2", "L3", "LP", "CL1", "CL2", "CL3", "CLP", "DL1", "DL2", "DL3", "DLP"), (nm, cells) in get(g, lvl, OpData[])
    per = Dict{String, Tuple{Float64, Float64}}()
    for r in _REF_ALL
        ps = _series(g, lvl, nm, r)
        (isnothing(ps) || isempty(ps)) && continue
        per[r] = gatestat(ps)
    end
    isempty(per) && continue
    # worst cell against the FASTER reference at that cell = the gate margin
    gate = Inf; gate_sz = 0; gate_adj = true; nnadj = 0
    for (sz, cell) in cells
        haskey(cell, _ARM_PB) || continue
        rs = Tuple{Float64, Bool}[]
        for r in _REF_ALL
            haskey(cell, r) || continue
            # PER-CELL adjudicability: compare THIS cell's pb anchor against THIS cell's reference
            # anchor. The run-level drift warning further down reports the WORST cell in the whole
            # cache and tars every other cell with it — on 2026-08-22 Zen5 reported 25.2% drift driven
            # solely by axpy@1e6, while two thirds of the cache sat under 2%. Reading that global
            # figure instead of the per-cell one turned two ordinary cells into phantom regressions
            # (trtrs 0.94->0.87, zgemvC 0.94->0.87); re-measured same-run they were 0.95 and 0.91, and
            # an afternoon went into "diagnosing" them. Hence the standing rule "per-cell anchor, not
            # run drift" — enforced in the output here rather than left to be remembered.
            ap = cell[_ARM_PB].anchor; ar = cell[r].anchor
            ok = !(isnan(ap) || isnan(ar)) && abs(ap / ar - 1) <= _ADJ_TOL
            push!(rs, (median(_ratio(cell[r].q, cell[_ARM_PB].q)), ok))
        end
        isempty(rs) && continue
        any(x -> !x[2], rs) && (nnadj += 1)
        i = argmin(first.(rs))
        if first(rs[i]) < gate
            gate = first(rs[i]); gate_sz = sz; gate_adj = last(rs[i])
        end
    end
    txt = join((@sprintf("%s %.2f/%.2f", r, per[r][1], per[r][2]) for r in _REF_ALL if haskey(per, r)), "  ")
    # A verdict resting on a cell whose two arms saw different machine state is not a verdict. Say so
    # ON the line that carries it, not in a footnote about some other cell.
    flag = gate_adj ? "" :
        @sprintf(
            "  [BINDING CELL n=%d NOT ADJUDICABLE: anchors differ >%.0f%% — re-measure `op=%s` in ONE run]",
            gate_sz, 100 * _ADJ_TOL, nm
        )
    note = (nnadj > 0 && gate_adj) ? @sprintf("  (%d non-adjudicable cell(s), not binding)", nnadj) : ""
    @printf(
        "%-3s %-8s %s   gate=%.3f %s%s%s\n", lvl, nm, txt, gate,
        gate_pass(gate) ? "PASS" : "FAIL", flag, note
    )
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
