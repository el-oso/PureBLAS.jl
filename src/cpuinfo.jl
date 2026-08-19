# CPU-generic SIMD parameters.
#
# The SIMD register width is detected once at precompile time and folded to a `const`, so the
# kernels stay branch-free and juliac-trimmable (no CpuId/cpuid ccall at runtime — mirrors how
# PureFFT bakes cache sizes from CPUSummary). The width can be pinned via a Preferences key for a
# reproducible cross-machine trim build (the fleet spans AVX-512 / AVX2 / NEON — see ROADMAP).
#
#   Zen4/Zen5 (AVX-512) -> 64 bytes -> Vec{8,Float64}, Vec{16,Float32}
#   Zen3     (AVX2)     -> 32 bytes -> Vec{4,Float64}, Vec{8,Float32}
#   Apple M*  (NEON)    -> 16 bytes -> Vec{2,Float64}, Vec{4,Float32}

using CpuId: simdbytes, cpuvendor, cpufeature, cachesize, cpumodel, cachelinesize, cpuid
using CPUSummary: cache_size
using Preferences: @load_preference

# Widest SIMD register in bytes. Preference override "simd_bytes" wins (cross-compile / pinning);
# otherwise detect on the build machine. CpuId is x86-only, so guard for portability.
const _SIMD_BYTES = let p = @load_preference("simd_bytes", nothing)
    if p !== nothing
        Int(p)::Int
    else
        b = try
            Int(simdbytes())
        catch
            16  # conservative SSE2/NEON fallback when detection is unavailable (e.g. aarch64)
        end
        b > 0 ? b : 16
    end
end

# SIMD lane count for element type `T` on this build (>=1). `# ponytail: width auto-detected,
# override via Preferences "simd_bytes"`.
@inline _vwidth(::Type{T}) where {T} = max(1, _SIMD_BYTES ÷ sizeof(T))

# Intel AVX2-without-AVX-512 (Haswell/Broadwell class). These cores have a narrower out-of-order window
# than Zen, so a serial FMA-reduction chain that Zen's OOO hides across iterations becomes latency-bound
# here (confirmed via `llvm-mca -mcpu=haswell`: the Cholesky base k-reduction runs 10 cyc/iter vs a 4-cyc
# resource bound). Kernels can opt into extra-accumulator splits keyed on this (see `_CHOL_BASE_SPLIT`).
# Detected at build (const-folds, trim-safe); AMD / AVX-512 / non-x86 → false. Width and cache size can't
# distinguish this — it's a microarchitecture trait, so it needs the vendor + feature bits.
const _INTEL_AVX2 = try
    cpuvendor() === :Intel && cpufeature(:AVX2) && !cpufeature(:AVX512F)
catch
    false
end

# L1 data-cache size in bytes (folded to a const; fallback if a level reports 0). Unused by the
# bandwidth-bound Level-1 kernels, but L2/L3 blocking for the M2 dgemm will read these.
const _L1_BYTES = let s = Int(cache_size(Val(1)))
    s > 0 ? s : 32 * 1024
end

# Cache-line size in bytes (folded to a const; 64 on all current x86/ARM). Governs software-prefetch
# density — one prefetch per line (the ger column-RMW prefetch) — and the prefetch-distance unit.
const _CACHELINE = let s = try
        Int(cachelinesize())
    catch
        0
    end
    s > 0 ? s : 64
end

# L1d associativity (ways), CPUID leaf 4 (Intel) / 0x8000001D (AMD) subleaf 0, ebx[22:31]+1 — the same
# leaf CpuId's `cachesize` parses for sizes. The derived way size (_L1_WAY_BYTES) is the address stride
# at which parallel streams collide in ONE L1 set: gemvN column streams at lda·sizeof ≡ 0 (mod way) all
# index the same set, so at most `ways` of them can coexist — the panel width selector keys on this.
# Fallback 8 (Zen2–5 and most Intel L1d are 8-way; Ice-Lake-class 48K/12-way also yields 4K ways).
const _L1D_ASSOC = @load_preference(
    "l1d_assoc",
    let w = try
            leaf = cpuvendor() === :AMD ? 0x8000_001d : 0x0000_0004
            eax, ebx, _, _ = cpuid(leaf, 0x0000_0000)
            # subleaf 0 must be a level-1 data/unified cache (eax[0:4] type≠0, eax[5:7] level==1)
            (eax & 0x1f) != 0 && ((eax >> 5) & 0x07) == 1 ? Int(1 + (ebx >> 22) & 0x03ff) : 0
        catch
            0
        end
        w > 0 ? w : 8
    end
)::Int
const _L1_WAY_BYTES = max(_CACHELINE, _L1_BYTES ÷ _L1D_ASSOC)

# L2 data-cache size in bytes (folded to a const; fallback 512 KiB if unreported). Governs the
# "operand fits L2 → one resident panel vs stream" thresholds (e.g. complex gemv _CGEMV_RB).
const _L2_BYTES = let s = Int(cache_size(Val(2)))
    s > 0 ? s : 512 * 1024
end

# L3 TOTAL size in bytes. NOTE: CPUSummary.cache_size(Val(3)) returns a PER-CORE SHARE (wintermute: 2.67M),
# but L3 is shared — nc blocking wants the TOTAL. CpuId.cachesize()[3] gives the total (16M). Fallback 8M.
const _L3_BYTES = @load_preference(
    "l3_bytes",
    let s = try
            Int(cachesize()[3])
        catch
            0
        end
        s > 0 ? s : 8 * 1024 * 1024
    end
)::Int

# ── AutoTune (req#8): derive machine-dependent tuning from detected cache + ISA + µarch, NOT hardcoded
# per-µarch literals. Julia JITs to the host, so we COMPUTE block sizes for the ACTUAL machine (incl. CPUs
# never benchmarked) — the structural advantage over static C/Rust BLAS. Every formula is a pure function
# of these load-time consts, so it const-folds (no runtime CpuId); every consumer keeps its Preferences
# override. Validated to reproduce the fleet's measured optima (see test/autotune_tests.jl). ─────────────
const _CPU_VENDOR = try
    cpuvendor()
catch
    :Unknown
end
# CpuId.cpumodel()[:Family] is raw-packed (ext<<4 | base); display family adds ext only when base==0xF
# (wintermute: raw 0xaf → 0xF + 0xA = 0x19 = Zen4; Zen5 = 0x1A). Baked to a const, Preferences-overridable.
_display_family(raw::Integer) = (raw & 0x0F) == 0x0F ? Int(raw & 0x0F) + Int(raw >> 4) : Int(raw & 0x0F)
const _CPU_FAMILY = @load_preference(
    "cpu_family",
    try
        _display_family(cpumodel()[:Family])
    catch
        0
    end
)::Int
# Architectural vector registers: 32 (AVX-512, AArch64 NEON) vs 16 (AVX2/SSE) — the register budget cap.
const _NVREG = _SIMD_BYTES >= 64 ? 32 : (Sys.ARCH === :x86_64 ? 16 : 32)

# Hardware descriptor: the detected machine as a plain NamedTuple const (every field a load-time const →
# the derivation functions below const-fold when called with `_HW`). The functions take a `hw` arg (not the
# globals) so they are PURE and the fleet table is an offline unit test (test/autotune_tests.jl feeds
# galen/Zen4/Zen5/TigerLake descriptors and asserts the measured optima). req#8.
const _HW = (
    simd = _SIMD_BYTES, l1 = _L1_BYTES, l2 = _L2_BYTES, l3 = _L3_BYTES,
    vendor = _CPU_VENDOR, family = _CPU_FAMILY, nvreg = _NVREG,
)

@inline _lanes(hw, ::Type{T}) where {T} = max(1, hw.simd ÷ sizeof(T))
# Double-pumped SIMD: full-width registers but HALF-width FP datapath (a 512-bit op occupies the 256-bit
# pipes twice). NOT CPUID-discoverable — the one legitimate family lookup (silicon FACTS, not tuned magic):
# AMD family 0x19 + AVX-512 = Zen4 (Zen3 shares 0x19 but simd==32 excludes it). Zen5 (0x1A) + all Intel
# AVX-512 = native. Extend as µarchs appear; unknown families default to native (the conservative side).
@inline _double_pumped(hw) = hw.simd == 64 && hw.vendor === :AMD && hw.family == 0x19
@inline _datapath_bytes(hw) = _double_pumped(hw) ? hw.simd ÷ 2 : hw.simd

@inline _round_dn(x::Int, m::Int) = max(m, x - rem(x, m))        # largest multiple of m ≤ x (≥ m)
@inline _avoid_po2(x::Int, m::Int) = ispow2(x) ? x - m : x       # dodge power-of-2 strides (set aliasing)

# (c) Datapath unrolls — criterion: latency×throughput. Keep ILP_TARGET independent FMA accumulator chains
# in flight (2 × FMA_latency(4) × FMA_ports(2) = 16) to cover latency on all pipes + hide panel loads.
const _ILP_TARGET = 16
@inline _at_gemm_nr(hw, ::Type{T} = Float64) where {T} = max(_lanes(hw, T), _ILP_TARGET ÷ _lanes(hw, T))
@inline function _at_gemm_mr(hw, ::Type{T} = Float64) where {T}                               # MR (row W-blocks)
    nr = _at_gemm_nr(hw, T)
    return min(cld(_ILP_TARGET, nr), (hw.nvreg - 1) ÷ (nr + 1))              # capped by reg budget
end
# Below max(m,n,k) ≤ this, the single-row unpacked tile beats the full _MR·W-row tile (can't fill one full
# tile of rows, so its 16-acc setup doesn't amortize). Criterion: the full tile's row height _MR·W. req#8
# (was a bare 40). W=8 → 2·8=16 (routes n=32 to the full tile). NOTE: W=4 → 3·4=12, but galen measured
# gating at 40 — sweep/pin before trusting W=4 (see gemm.jl _GEMM_MR1_MAX).
@inline _at_gemm_mr1_max(hw, ::Type{T} = Float64) where {T} = _at_gemm_mr(hw, T) * _lanes(hw, T)
# (a) L1-resident block: a `unit`-element micro-operand per k-step stays in (num/den)·L1 across the sweep.
@inline _l1_block(hw, ::Type{T}, unit::Int; num::Int = 1, den::Int = 2, mult::Int = 8) where {T} =
    _round_dn(((hw.l1 * num) ÷ den) ÷ (unit * sizeof(T)), mult)
@inline _at_gemm_kc(hw, ::Type{T} = Float64) where {T} = _l1_block(hw, T, _at_gemm_nr(hw, T))  # B micropanel ≤ ½·L1
# (b) L2/L3 blocks — A block ≤ ~30% L2 (rest streams B/C/prefetch); B block ≤ ¼ shared L3, po2-dodged.
@inline function _at_gemm_mc(hw, ::Type{T} = Float64) where {T}
    kc = _at_gemm_kc(hw, T)
    return _round_dn(((hw.l2 * 3) ÷ 10) ÷ (kc * sizeof(T)), _at_gemm_mr(hw, T) * _lanes(hw, T))
end
@inline function _at_gemm_nc(hw, ::Type{T} = Float64) where {T}
    nr = _at_gemm_nr(hw, T)
    return _avoid_po2(_round_dn((hw.l3 ÷ 4) ÷ (_at_gemm_kc(hw, T) * sizeof(T)), nr), nr)
end
# (e) Short-k split-reduction tile (small-n, wide-AVX-512 only). The wide steady-state tile (_MR×_NR)
# UNDER-FILLS at short k: each C-cell's reduction chain is only k deep, so during the fill few independent
# FMAs are in flight and the pipeline stalls (measured n=32: PB 0.85× OB — chains ready, but not enough of
# them early). A TALL tile with an S-way k-split keeps _ILP_TARGET partial chains live from iter 0 (every
# cell splits into S independent accumulators summed at store), covering latency during the fill. SAME
# _ILP_TARGET register budget as the wide tile, just redistributed. Gate to wide-reg AVX-512 (nvreg≥32):
# on AVX2 (16 regs) the _ILP_TARGET accumulators + load temps spill, AND small-n gemm there is already ≥1.0
# via _GEMM_MR1_MAX — so it const-folds off on Zen3. Derived: S=2 (double ILP for the fill), cells=_ILP_TARGET÷S
# arranged TALL (NR=2 → a B-reuse column pair; MR=cells÷NR), giving 4×2 on W=8. Validated n=32 0.85→1.07× OB
# (wintermute Zen4); crossover ~n=56 (the wide tile self-fills beyond). req#8.
const _GEMM_SPLIT_S = 2
@inline _at_gemm_split_ok(hw) = hw.nvreg >= 32 && hw.simd >= 64
@inline _at_gemm_split_nr(hw) = 2
@inline _at_gemm_split_mr(hw) = (_ILP_TARGET ÷ _GEMM_SPLIT_S) ÷ _at_gemm_split_nr(hw)
# Upper size: split helps until the wide tile self-fills. One tall split tile (_split_mr·W rows) plus one
# wide row-block (_MR·W) ≈ where the crossover sits (W=8: 32+16=48 — covers the measured wins 32/48, drops
# 64 to the wide path). Preferences-overridable ("gemm_split_max"); fleet-validate before trusting.
@inline _at_gemm_split_max(hw, ::Type{T} = Float64) where {T} =
    (_at_gemm_split_mr(hw) + _at_gemm_mr(hw, T)) * _lanes(hw, T)
# (c2) Real-axpy DRAM-regime arm. Past L3 the kernel is bandwidth-bound, so the FULL-width vector buys no
# bandwidth on a datapath that must split it anyway — the narrow 256-bit phase arm (208) wins there and
# only there. That criterion IS the datapath, so `_double_pumped` is the honest predictor rather than a
# µarch lookup. Native-256 (Zen3) and native-512 (Zen5) parts both take the interleaved arm (4).
# Measured, arms called directly at n=3e6 (bench/probes/axpy_arm_regime_ab.jl, µs, lower better):
#   wintermute Zen4 double-pumped: arm4 3412  arm208 2821  → 208 by 17%
#   galen Zen3 native 256:         arm4 3028  arm208 3146  → 4 (and at 6e6/12e6 too)
# Zen5 predicts 4; unconfirmed (that box's frequency lock was not holding). Replaced a OncePerProcess duel
# that resolved the 17%-slower arm on wintermute in 7 of 9 processes and flipped in the other 2.
@inline _at_axpy_dram(hw) = _double_pumped(hw) ? 208 : 4
# (c3) Real-axpy CACHE-BAND arm (L1 < ws < L3). SAME predicate, SEPARATE knob — do not merge them; the
# two regimes are independently measured and simd_kernels.jl records what happened the last time one
# value was applied across both. This one is pinned down by the GATE rather than by a probe, because on
# every fleet box the whole BLAS-1 size ladder (n ≤ 1e6, ws = 2·n·8 ≤ 16 MB) sits BELOW L3 and is
# therefore governed here — `_axpy_dram` is not exercised by the gate at all.
# The value reproduces each box's incumbent exactly, so the conversion is behaviour-neutral and buys only
# determinism: Zen4 208 (the narrow arm that closed the long-standing n=1e6 miss, +5.6-10% at that cell),
# Zen3 4. Zen5 also wants 4 — the narrow arm LOSES 0.6-1.4% there — and `_double_pumped` is what
# separates Zen4 from Zen5, the two boxes with identical L3 and vector width. That inversion was read as
# "not predictable from a detected const"; it is, just not from L3 or W.
# NOTE (2026-08-19): a direct-call A/B probe measured the interleaved arm FASTER on wintermute at n=1e5
# and n=1e6 (49.7 vs 52.8 µs, 815 vs 928 µs), contradicting the gate-regime numbers above. Not acted on
# — the probe-regime rule says the in-situ gate measurement wins, and this knob's history is of probes
# picking arms the gate then rejects. Unresolved; re-measure IN plots.jl before touching the value.
@inline _at_axpy_band(hw) = _double_pumped(hw) ? 208 : 4
# (d) Complex-Cholesky tuning. cpotf2 base row-unroll: line-rate match — unroll until one step consumes a
# 64B cache line at datapath width (native-512 → 1 op, MR=1; double-pump/AVX2 32B → MR=2). base/nbmax:
# implementation crossovers (fleet-measured cache-independent, width-dominant) → affine in W (width-
# independent overhead floor + per-lane slope): base 32+4W (48/64), nbmax 64+16W (128/192).
# A-row block for the ACTUAL local kc + element type (joint residency mc·kc·sizeof ≤ 30%·L2). The single
# authority for mc across ALL gemm/L3 sites (real & complex) — a standalone `_MC` const would bake the
# canonical kc + Float64, so it under-blocks small-kc callers (potrf's trailing gemm at kc=nb) AND mis-sizes
# complex (16 B/elt vs 8). Recomputes from the local kc/T; capped at `cap` (=cld(dim,mr)*mr), rounded to the
# local mr — so real (mr=_MR·W) and complex (mr=_CMR·W) sites each round correctly. req#8.
@inline _at_mc_kc(hw, ::Type{T}, kc::Int, mr::Int, cap::Int) where {T} =
    min(max(mr, (((((hw.l2 * 3) ÷ 10) ÷ (kc * sizeof(T))) ÷ mr) * mr)), cap)
@inline _at_cpotf2_mr(hw) = max(1, 64 ÷ _datapath_bytes(hw))
@inline _at_cpotrf_base(hw) = 32 + 4 * _lanes(hw, Float64)
@inline _at_cpotrf_nbmax(hw) = 64 + 16 * _lanes(hw, Float64)

# (f) Pack crossovers for gemm/syrk/syr2k (req#8). Above `n`, a single-pass PACKED kernel is used; at/below,
# the cheaper "re-stream the operand directly from cache" path (unpacked microkernel for gemm, recursion for
# rank-k). Criterion: REGISTER CAPACITY, not cache residency. The fleet's measured Zen4 crossover (448) puts
# one square operand at 448²·8 = 1.6 MB ≫ 1 MB L2 — the re-stream path demonstrably wins while only L3-
# resident, which FALSIFIES an L2-residency law (a `sqrt(L2)·W²` fit only appeared to work because on this
# fleet the small-L2 box is also the narrow-W box: L2 and W are collinear). The re-stream path wins while the
# microkernel's accumulator working set stays register-bound; it loses once the packed kernel's contiguous
# reuse amortizes. Base quantity = the FP register file's accumulator capacity in elements:
#   _acc_cap = (nvreg − 4)·W,  where 4 = the k-step's operand registers (MR A-loads + 1 B-broadcast; the same
#   budget `_at_gemm_mr` reserves — galen's 3×4 tile uses all 16 regs: 12 acc + 3 A + 1 B). Op multiples are
#   machine-INDEPENDENT op-shape calibration (like `_l1_block`'s ½, `_at_gemm_mc`'s 3/10): gemm pays 2 passes
#   (read+write) to pack → ×2; rank-k's re-stream alternative is recursion (2×-flop diagonal + split overhead)
#   so it packs slightly earlier → ×7/4. Reproduces galen (nvreg16,W4): gemm 96 (measured tie-band 80–112),
#   rank-k 84 (routes n=80→recursion, 96→packed). req#8.
# PATH-DEPENDENT crossover (the ×7/4·acc_cap form was calibrated for the AVX2 MULTI-pack `_trgemm_packed!`,
# whose double-A-pack keeps the recursion competitive to ~84). AVX-512 uses the UNIFIED single-pack
# `_trgemm_packed_u!` (half the pack traffic, 8 well-fed accumulators, mirror `_unified_ok`: W≥8 && W==NR),
# whose real crossover is ≈W — the recursion's 2×-flop diagonal waste + per-call overhead loses from ≈W up.
# The 392 the old formula gave on AVX-512 (mis)routed all n≤256 to the recursion → the syrk n=128 gate miss
# (Zen5 0.88 / Zen4 0.91); packed is +14% there (→~1.04). Fleet-validated: AVX-512 → W, galen AVX2 → 84.
@inline _acc_cap(hw, ::Type{T} = Float64) where {T} = (hw.nvreg - 4) * _lanes(hw, T)
@inline _at_gemm_unpack_max(hw, ::Type{T} = Float64) where {T} = 2 * _acc_cap(hw, T)
@inline function _at_rank_k_pack_cut(hw, ::Type{T} = Float64) where {T}
    W = _lanes(hw, T)
    return (W >= 8 && W == _at_gemm_nr(hw, T)) ? W : (7 * _acc_cap(hw, T)) >> 2   # unified single-pack vs AVX2 multi-pack
end
# symm is a DIFFERENT criterion: its re-stream alternative is materialize-then-gemm (a one-shot O(n²) dense
# copy of the symmetric triangle + the flagship gemm), not a strided microkernel. Materialize+gemm beats the
# packed symmetric kernel at EVERY measured galen n (the packed path is dead weight on AVX2), and the only
# thing that can kill it is the O(n²) copy evicting the gemm's resident A-block from L2 — i.e. when the
# materialized n×n copy no longer fits L2. Threshold = side of a square that fills L2: n = √(L2/sizeof). Galen
# measured a mat≈pack TIE at exactly n=256 = √(512K/8), pinning the fraction at 1. This lifts the cut off the
# mistuned 96 (which routed n=112–192 to the slower packed path, the AOCL misses) up to 256, routing the whole
# gate mid-range to materialize. Predicts Zen4/Zen5 362 (DOWN from the 448 placeholder — validate on wintermute). req#8.
@inline _at_symm_mat_max(hw, ::Type{T} = Float64) where {T} = Base.isqrt(hw.l2 ÷ sizeof(T))

# ── PDM "Measure" tier: the estimator ───────────────────────────────────────────────────────────────
# Measure-tier knobs run a real timing loop ON THE HOST, once per process, to pick a kernel variant or a
# block size (zaxpy narrow-vs-wide, ger stream count, pstrf batched swap, gebrd nb, …). Those harnesses
# cannot use Chairmarks — PureBLAS must not depend on it and must stay trim-safe — so `time_ns` is
# unavoidable here, and this is the ONLY place in the package permitted to touch a clock.
#
# THE ESTIMATOR IS STILL THE MEDIAN. Until 2026-08-05 every one of these harnesses reduced with
# `min(t, time_ns() - s)` over 3-5 samples. `min` is optimistic AND tail-blind: it reports the luckiest
# window and is blind to exactly the tail behaviour that separates two kernels on real data. That is the
# same estimator that ranked an iamax unroll backwards and cost a day — except a probe only misleads the
# person reading it, whereas THIS selects the code that ships, on every machine PureBLAS runs on, and
# `test/estimator_lint.jl` could not see it (it scanned bench/ only; it now scans src/ too).
#
# A/B INTERLEAVED, so drift between the two candidates is common-mode rather than assigned to whichever
# ran second, and the first (cold) round is discarded.
"""
    _tune_ab(fa, fb; reps=5) -> (ta, tb)

Median elapsed ns for two candidate implementations, measured alternately. Returns medians, never minima.
`reps` is the number of TIMED rounds; one untimed warmup round runs first.
"""
function _tune_ab(fa::FA, fb::FB; reps::Int = 5) where {FA, FB}
    ta = Vector{UInt64}(undef, reps)
    tb = Vector{UInt64}(undef, reps)
    fa(); fb()                                     # warmup, untimed (first touch + any JIT)
    for r in 1:reps
        s = time_ns(); fa(); ta[r] = time_ns() - s
        s = time_ns(); fb(); tb[r] = time_ns() - s
    end
    sort!(ta); sort!(tb)
    m = (reps + 1) ÷ 2
    return (ta[m], tb[m])
end

"""
    _tune_one(f; reps=9) -> t

Median elapsed ns for a single candidate (for sweeps over a candidate SET, where each candidate is timed
in its own call). Same estimator rule as `_tune_ab`.
"""
function _tune_one(f::F; reps::Int = 9) where {F}
    ts = Vector{UInt64}(undef, reps)
    f()                                            # warmup, untimed
    for r in 1:reps
        s = time_ns(); f(); ts[r] = time_ns() - s
    end
    sort!(ts)
    return ts[(reps + 1) ÷ 2]
end

"""
    _tune_better(t, best) -> Bool

Should a candidate at time `t` displace the incumbent at `best`? Only if it wins by a MARGIN.

WHY A MARGIN AND NOT `<`. A plain `t < best` switches on noise: when two candidates are within the
probe's resolution the winner is whichever got the luckier window, so the pick changes from process to
process — and since these knobs select a SHIPPED kernel, the emitted code then varies run to run.
MEASURED on neuromancer (Zen5), 2026-08-05: two fresh processes of the SAME binary resolved the axpy
DRAM knob to 208 and to 4. That is not a tuning result, it is a coin flip that changes what executes, and
it feeds straight back into the run-to-run variance that makes that box hard to measure at all.
5% is above the probes' resolution on every fleet box and far below the gaps worth switching for (the
phase body won Zen4 by ~11% at n=1e6). Ties therefore go to the INCUMBENT, which is the derived default.
"""
@inline _tune_better(t::UInt64, best::UInt64) = t * 100 < best * 95

# ROUND COUNT IS DERIVED, NOT PICKED. There is exactly ONE free choice in this whole rule and it is
# `_TUNE_ALPHA` below; everything else follows by arithmetic from it.
#
# `_tune_wins_it` demands `rounds-1` wins, so at a GENUINE TIE one duel displaces with probability
# `(rounds+1)/2^rounds` — that is exact, a property of the rule alone, and needs no assumption about
# any machine. A sweep runs one duel PER CANDIDATE, so what governs whether a knob is stable is the
# FAMILY-WISE rate `ncand·(rounds+1)/2^rounds`. Solve that for `rounds` and the constant disappears.
#
# Measured consequence of getting this wrong (2026-08-06, wintermute, one knob per fresh process, 5
# rounds throughout): `cgemvn_nc_big` (2 candidates) and `ger_np` (3) came out STABLE while `axpy_band`
# (6) and `axpy_dram` (4) still flipped. The instability tracked CANDIDATE COUNT — the signature of a
# multiple-comparisons error, not of a noisy machine, and the reason a single global round count was
# the wrong shape for this knob in the first place.
#
# WHY THIS IS NOT THE 5% MARGIN IN NEW CLOTHES. That margin was an empirical CLAIM about hardware
# ("5% is above the probes' resolution on every fleet box") — unverified, and measurement later put the
# real figure anywhere from 0.02% to 6.5% on ONE box. `_TUNE_ALPHA` claims nothing about hardware. It
# is a stated risk budget: how often we are willing to let a library load ship a different kernel than
# the last one WHEN THE CANDIDATES ARE GENUINELY TIED — in which case the two kernels are equivalent by
# construction and the cost of the flip is bounded by δ. Spread over the ~13 Measure knobs in the tree,
# 5% library-wide is ~0.4% per knob. Raise or lower it as a product decision; the rounds follow.
"""
    _force_knob(name) -> Int      # -1 when unset

`PUREBLAS_FORCE_<name>` overrides what a Measure-tier resolver returns, for ONE process, without
editing source or setting a Preference. Purely a MEASUREMENT INSTRUMENT — nothing ships reading it.

WHY IT EXISTS. Settling "is the tuner picking the wrong arm?" requires running the arm through the
REAL entry path and reading `bench/plots.jl`, because a standalone probe can rank the arms backwards
(see the falsification block in `simd_kernels.jl`: a probe put phase-narrow 8-13% ahead at axpy n=1e6
while forcing it moved the gate 0.962 -> 0.921). Before this, forcing an arm meant one of:
  - a Preference — but `bench/plots.jl` runs under `--project=bench`, which does NOT see the root
    `LocalPreferences.toml`. Verified: the same key read 108 under `--project=.` and 4 under
    `--project=bench`. Two full gate runs were spent producing the same configuration twice before the
    `tune=` stamp exposed it.
  - editing source and reverting — which works, and is one forgotten `git checkout` from a fake result.
An env var is per-process, cannot outlive the run, and is invisible to a run that does not set it.

SELF-DOCUMENTING BY CONSTRUCTION: `_tunestamp()` reads the RESOLVED values, so a forced run stamps its
forcing into the cache header. A forced cache can never be mistaken for a measured one.

Reading `ENV` here is trim-safe by placement, not by luck: every call site sits inside the
`@static if isnothing(<pref>)` branch that a pinned/trim build does not compile at all.
"""
@inline function _force_knob(name::String)::Int
    v = get(ENV, "PUREBLAS_FORCE_" * name, "")
    isempty(v) && return -1
    x = tryparse(Int, v)
    return isnothing(x) ? -1 : x
end

const _TUNE_ALPHA = 0.05                    # library-wide P(ANY knob flips at a tie), the one dial
# req8-ok: NOT a machine-dependent tuning value — it is a COUNT of the Measure-tier knobs in this tree,
# used as the Bonferroni denominator that splits `_TUNE_ALPHA` across them. It depends on the source,
# not on the host, so there is nothing to derive from a detected const. Being stale-high is the safe
# direction (a larger denominator buys MORE rounds), so drift costs conservatism, never correctness.
# Keep in step with bench/probes/knob_stability_audit.jl, which enumerates them.
# DELIBERATELY NOT LOWERED as knobs are converted (2026-08-19: cgemvt_cfg, gemvt_nc, pptrf_spr_min,
# pstrf_rowcache, axpy_band, axpy_dram all demoted to Derive). Lowering it shrinks `_TUNE_ROUNDS`, i.e.
# buys FEWER rounds for the duels that remain — the opposite of the retirement's intent. Per the note
# above, drift high costs conservatism only. Reset it if the Measure tier is ever repopulated.
# req8-ok: a COUNT of Measure knobs in this source tree (Bonferroni denominator), not a machine value
const _TUNE_NKNOBS = 12                     # 13 until cgemvn_nc_big was demoted to Derive (2026-08-07)

"""
    _tune_rounds(ncand) -> Int

Rounds needed so that `ncand` duels together risk at most `_TUNE_ALPHA/_TUNE_NKNOBS` of a spurious
displacement at a tie. Pure, integer, evaluated at the call site — so a sweep with 2 candidates does
not pay for a sweep with 6, and adding a candidate automatically buys the rounds it costs.
"""
@inline function _tune_rounds(ncand::Int)
    budget = _TUNE_ALPHA / _TUNE_NKNOBS
    r = 5
    while r < 25 && ncand * (r + 1) / 2.0^r > budget
        r += 1
    end
    return r
end
const _TUNE_ROUNDS = _tune_rounds(6)        # widest sweep in the tree; the default for callers

"""
    _tune_duel(fa, fb; rounds=_TUNE_ROUNDS, reps=5, δ=2) -> Int

How many of `rounds` the candidate `fb` beats the incumbent `fa`. Feed to [`_tune_wins_it`](@ref).
Each round reduces BOTH arms with the MEDIAN of `reps` timings, then records ONE bit: who won. Arm
order alternates per round (ABBA). One untimed warmup of each runs first.

WHY THIS REPLACES A MARGIN ON TWO MEDIANS. The old rule compared one median against one median and
demanded a fixed 5% win. Measured 2026-08-06 (wintermute, freq-locked, quiet): the complex gemvN NC
pair differs by a stable ~3-4%, which sits just under that threshold, so NOISE decided whether it
cleared — four fresh processes of the same binary resolved the knob to 8, 12, 8, 8. The threshold did
not prevent the coin flip, it *caused* it, by putting the decision boundary inside the noise band.

Counting round wins instead is distribution-free and self-calibrating: the evidence required scales
with the machine's own noise without ever estimating a noise floor. On a quiet shape a reproducible
3% win takes every round; on a noisy one it takes about half, and the supermajority refuses. Measured
separation for that same pair, in the tuner's own regime — null (an arm against ITSELF) 2/5, power
5/5. The margin could not resolve what this resolves cleanly.

⚠ THE ORDER OF OPERATIONS IS THE WHOLE TRICK, and getting it backwards is why an earlier attempt
failed. A sign test whose per-round statistic is a SINGLE timing makes every round a coin flip, and
null then overlaps power at every number of rounds (measured: null up to 7/15 against power 4/15).
AGGREGATE FIRST — median within a round — THEN count signs across rounds. Reducing twice is the point,
not redundancy: the median kills within-round outliers, the sign count kills between-round drift and
mode-hops.

`δ` is a REGRET bound in percent — "do not churn the shipped kernel for less than this" — machine
independent POLICY, not a noise estimate. That separation is the correction: the old 5% did both jobs
and only the policy half was ever legitimately a constant.

Cost is `rounds*reps*2` timings (~50 at the defaults) — tens of ms once per process, so this stays a
load-time self-tune and needs no pin and no persisted calibration.
"""
function _tune_duel(
        fa::FA, fb::FB; rounds::Int = _TUNE_ROUNDS, reps::Int = 5, δ::Int = 2,
        refresh::RF = nothing
    ) where {FA, FB, RF}
    fa(); fb()                                     # warmup, untimed (first touch + any JIT)
    wins = 0
    need = rounds - 1                              # the supermajority `_tune_wins_it` will demand
    # ⚠ TWO DIFFERENT QUESTIONS, AND δ ANSWERS ONLY THE SECOND. "Is the candidate faster?" is EVIDENCE
    # and is settled by the sign count. "Is it faster by enough to be worth changing the shipped
    # kernel?" is POLICY and is settled by δ. Testing δ inside each round conflates them, and that
    # conflation rejected a real win: measured 2026-08-07 on wintermute, freq-locked and quiet, the
    # axpy DRAM candidate 208 beats the incumbent by 5.6% pooled, but its PER-ROUND ratios ran
    # 1.015 1.004 1.116 1.100 — two rounds under 2%, so a per-round δ=2 scored them as losses and the
    # supermajority then refused a candidate that is 5.6% faster. Six fresh processes all resolved to
    # the incumbent while a controlled gate-regime A/B put the candidate 3.9-10.2% ahead.
    #
    # It also disagreed with the α this test claims. `_tune_rounds` derives `rounds` from
    # `(r+1)/2^r` = P(≥ r-1 wins of r | fair coin) — the null of a PURE sign test. δ inside the round
    # makes the realized test stricter than the α it advertises, so the false-NEGATIVE rate was
    # uncontrolled and unmeasured while the false-positive rate was over-bought.
    #
    # So: sign count at δ=0 (matches the derivation exactly), and the regret bound applied ONCE to the
    # POOLED effect. Both protections the docstring argues for survive; each is now applied at the
    # level it belongs to.
    ratios = Vector{Float64}(undef, rounds)        # per-round ta/tb, pooled for the regret check
    nr = 0
    for r in 1:rounds
        # `refresh` RESAMPLES THE OPERANDS between rounds, and for some knobs it is the difference
        # between a decidable and an undecidable question. Duelling resamples TIME; if the candidates'
        # relative speed depends on state fixed ONCE PER PROCESS — page placement, THP promotion, the
        # addresses the allocator happened to hand out — then every round re-measures the same draw and
        # no number of rounds converges. `axpy_band` demonstrated it: with 15 rounds a tie-driven false
        # positive has probability 0.05%, yet it still resolved 208 in some processes and 4 in others,
        # which can only mean the winner genuinely differs per process. Rotating operands makes each
        # round a fresh draw, so the sign count aggregates over placements — which is what production
        # sees anyway, since callers do not all share one lucky allocation.
        refresh === nothing || refresh(r)
        local ta::UInt64, tb::UInt64
        if isodd(r)                                # ABBA: neither arm always runs second
            ta = _tune_one(fa; reps = reps); tb = _tune_one(fb; reps = reps)
        else
            tb = _tune_one(fb; reps = reps); ta = _tune_one(fa; reps = reps)
        end
        nr += 1; ratios[nr] = ta / tb               # >1 ⇒ candidate faster this round
        tb < ta && (wins += 1)                      # EVIDENCE: pure sign test, δ handled below
        # EARLY EXIT once the verdict is settled — this is what pays for the round count. A candidate
        # that has already lost twice cannot reach `rounds-1`, and most candidates in a sweep are
        # hopeless, so the common case costs 2 rounds rather than `rounds`. Only a genuinely close
        # contest runs to the end, which is exactly where the extra rounds are needed.
        wins + (rounds - r) < need && return wins   # cannot still reach the threshold
        wins >= need && return _tune_regret(wins, ratios, nr, δ)
    end
    return _tune_regret(wins, ratios, nr, δ)
end

# POLICY gate, applied once to the POOLED effect (see the note in `_tune_duel`). The sign count has
# already established the candidate is faster; this asks whether it is faster by enough to justify
# changing which kernel ships. Median over rounds for the same reason `_tune_one` medians over reps —
# it is the estimator this project gates on, and it is insensitive to the window tails that a single
# unlucky round contributes. Returning 0 makes a candidate that wins but does not clear the regret
# bound indistinguishable from one that lost, which is exactly the intent: the incumbent holds.
@inline function _tune_regret(wins::Int, ratios::Vector{Float64}, nr::Int, δ::Int)
    nr == 0 && return wins
    r = sort!(view(ratios, 1:nr))[(nr + 1) ÷ 2]
    return r * 100 >= (100 + δ) ? wins : 0
end

"""
    _tune_wins_it(wins, rounds=_TUNE_ROUNDS) -> Bool

Does a candidate with `wins` of `rounds` displace the incumbent? Requires a SUPERMAJORITY (`rounds-1`),
so one spoiled round — a mode hop, a stray interrupt — cannot decide what ships.

DETERMINISM, the property the old fixed margin was defending: at an exact performance tie the chance of
displacing is `(rounds+1)/2^rounds` — 18.75% at 5 rounds, 2% at 9, 0.17% at 13 — and it is DIALLED by
`rounds` rather than assumed. The margin, by contrast, was deterministic only if the host's noise
happened to fall below it, which was asserted fleet-wide and never verified. Note also that a tie here
means the candidates are genuinely equivalent, so a flip costs at most `δ`; the old failure flipped
between candidates differing by a real 3-4%.
"""
@inline _tune_wins_it(wins::Int, rounds::Int = _TUNE_ROUNDS) = wins >= rounds - 1

"""
    _tune_duel_pick(inc, cands) -> value or `nothing`

Duel-based sibling of [`_tune_pick`](@ref): the SAME two rules, for sweeps whose arms are closures
rather than pre-measured times. `cands` is an iterable of `(value, thunk)`.

**THE BUG THIS EXISTS TO KILL.** Every duel-based sweep open-coded

    for c in (8, 2, 1)                                    # some fixed order
        _tune_wins_it(_tune_duel(inc, arm(c))) && return c
    end

which returns **the first candidate that beats the incumbent, in list order** — not the best one. It
never compares the candidates against each other, so a non-monotonic optimum is unreachable and the
shipped value depends on how someone wrote a tuple literal. Measured consequence (2026-08-09, Zen5):
`_ger_np` shipped 8 while the gate showed 1 faster by ~12% at n=2048; 8 came first, beat the incumbent
4, and 1 was never evaluated. The same loop also explains that knob's `1 8 8 8 8` instability across
fresh processes — not a tie, a FALL-THROUGH: when 8's duel misses its supermajority, control reaches
2 (fails) and then 1 (wins), so the answer flips on which duels happened to land.

Both of `_tune_pick`'s rules are preserved:

  * **Qualification is against the FIXED incumbent**, never a running best — so what a candidate must
    beat does not depend on what won earlier (that is the documented hazard, see `_tune_pick`).
  * **Selection is an argmin AMONG QUALIFIERS.** Comparing two qualifiers to each other is not the
    forbidden re-basing: they have each already cleared the same fixed bar.

Ties go to the incumbent (`nothing`), matching `_tune_pick`'s `0`. Costs at most `2N-1` duels against
the old `N`; this runs once per process at load time, never in a kernel.
"""
function _tune_duel_pick(inc::FA, cands; kwargs...) where {FA}
    champ = nothing
    champ_f = inc
    for (v, f) in cands
        _tune_wins_it(_tune_duel(inc, f; kwargs...)) || continue   # bar = the INCUMBENT, fixed
        if isnothing(champ)
            champ = v; champ_f = f
        elseif _tune_wins_it(_tune_duel(champ_f, f; kwargs...))     # argmin among qualifiers only
            champ = v; champ_f = f
        end
    end
    return champ
end

"""
    _tune_pick(t_inc, ts::NTuple{N,UInt64}) -> Int

Which candidate displaces the incumbent? `0` keeps it; otherwise the index into `ts`. Pure — no clock,
no allocation — so the SELECTION RULE is unit-testable independently of any measurement
(`test/tuner_tests.jl`). Every Measure-tier sweep should route its decision through this rather than
open-coding a loop, which is how the two properties below came to differ per site.

TWO RULES, and the first is the one that is easy to get wrong:

  * QUALIFY AGAINST THE INCUMBENT, never against a running best. The open-coded form
    `_tune_better(t, bt) && (bt = t; best = c)` re-bases the comparison after each win, so what a
    later candidate must beat depends on what happened to win earlier. Here `t_inc` is fixed.
  * Among QUALIFIERS the margin applies again, ties going to the EARLIER candidate. Callers order
    candidates smallest-first, so a tie keeps the cheaper one (fewer live registers, less scratch).

⚠ HONEST NOTE ON WHAT THIS CHANGED: for a MONOTONE ladder (each candidate faster than the last) this
is behaviourally IDENTICAL to the chained form — if `b` cannot beat `a` by the margin, both rules
return `a`. The difference appears only when a later candidate qualifies against the incumbent while an
earlier one that beat it did not, which cannot happen when times are ordered. So the rewrite buys
testability and a stated invariant, NOT a different answer on the case that motivated it; the measured
outcome was unchanged, and the unit tests below pin that down rather than letting the claim drift.
"""
@inline function _tune_pick(t_inc::UInt64, ts::NTuple{N, UInt64}) where {N}
    best = 0
    bt = t_inc
    for i in 1:N
        _tune_better(ts[i], t_inc) || continue        # qualify vs the INCUMBENT, fixed
        (best == 0 || _tune_better(ts[i], bt)) && (best = i; bt = ts[i])
    end
    return best
end

# A PAIRED ABBA SIGN TEST WAS PROPOSED TO REPLACE THE MARGIN, BUILT, AND MEASURED. It does not work
# here, and the reason is worth more than the rule was.
#
# The proposal (Fable, 2026-08-06): instead of comparing two block-timed medians against a fixed 5%,
# run k adjacent pairs with the order alternated (ABBA), record only WHICH arm won each pair, and
# displace on a supermajority. Attractive because it is distribution-free, needs no noise-floor
# estimate, and self-scales — a noisy shape demands a bigger true gap, a quiet one accepts a small
# reproducible win. The motivating case was the complex gemvN NC knob, where the 5% margin discards a
# measured 3.9% (W vs 3W/2 at A≈L3 on Zen4).
#
# MEASURED on wintermute, freq-locked, quiet box, at the knob's own probe shape (1024², A≈L3), sweeping
# k ∈ {9,15,21} and legs of r ∈ {1,5,15} calls — null (an arm against ITSELF), power (3W/2 over W) and
# reverse (W over 3W/2), all in one run:
#     r=1    null up to 6/15, power 4/15   → null and power OVERLAP at every k
#     r=5    null up to 7/15, power 6/15
#     r=15   null up to 7/9,  power 0/9
# No (k, threshold) separates them. And the diagnosis is NOT that the test is weak: as the legs get
# heavier the effect DISAPPEARS — at r=15 the plain median ratio is W/3W2 = 0.997, i.e. the two panel
# widths are EQUAL in this regime. The 3.9% is real, but it lives in the GATE regime (fresh arrays per
# sample, `_L1REP` dependent passes) and does not exist in the TUNER's regime (one pre-touched buffer,
# back-to-back calls on hot data).
#
# SO THE PREMISE WAS WRONG: the margin is not discarding a real win. There is no win here to discard,
# and no decision rule can select for an effect its own measurement regime does not reproduce. Which
# makes this a probe-regime finding about the TUNERS themselves — they measure in a regime that differs
# from the one the gate scores, so they can pick the wrong candidate however good the statistics are.
# That is the open problem (see the task tracker); the threshold is not.
# Do not re-propose the sign test without first showing the effect survives in the tuner's own regime.
# Harness kept: bench/probes/tune_signtest_validate.jl (null + power + reverse + cross-process modes).
