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

using CpuId: simdbytes, cpuvendor, cpufeature, cachesize, cpumodel
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

# L2 data-cache size in bytes (folded to a const; fallback 512 KiB if unreported). Governs the
# "operand fits L2 → one resident panel vs stream" thresholds (e.g. complex gemv _CGEMV_RB).
const _L2_BYTES = let s = Int(cache_size(Val(2)))
    s > 0 ? s : 512 * 1024
end

# L3 TOTAL size in bytes. NOTE: CPUSummary.cache_size(Val(3)) returns a PER-CORE SHARE (wintermute: 2.67M),
# but L3 is shared — nc blocking wants the TOTAL. CpuId.cachesize()[3] gives the total (16M). Fallback 8M.
const _L3_BYTES = @load_preference("l3_bytes",
    let s = try Int(cachesize()[3]) catch; 0 end; s > 0 ? s : 8 * 1024 * 1024 end)::Int

# ── AutoTune (req#8): derive machine-dependent tuning from detected cache + ISA + µarch, NOT hardcoded
# per-µarch literals. Julia JITs to the host, so we COMPUTE block sizes for the ACTUAL machine (incl. CPUs
# never benchmarked) — the structural advantage over static C/Rust BLAS. Every formula is a pure function
# of these load-time consts, so it const-folds (no runtime CpuId); every consumer keeps its Preferences
# override. Validated to reproduce the fleet's measured optima (see test/autotune_tests.jl). ─────────────
const _CPU_VENDOR = try cpuvendor() catch; :Unknown end
# CpuId.cpumodel()[:Family] is raw-packed (ext<<4 | base); display family adds ext only when base==0xF
# (wintermute: raw 0xaf → 0xF + 0xA = 0x19 = Zen4; Zen5 = 0x1A). Baked to a const, Preferences-overridable.
_display_family(raw::Integer) = (raw & 0xF) == 0xF ? Int(raw & 0xF) + Int(raw >> 4) : Int(raw & 0xF)
const _CPU_FAMILY = @load_preference("cpu_family",
    try _display_family(cpumodel()[:Family]) catch; 0 end)::Int
# Architectural vector registers: 32 (AVX-512, AArch64 NEON) vs 16 (AVX2/SSE) — the register budget cap.
const _NVREG = _SIMD_BYTES >= 64 ? 32 : (Sys.ARCH === :x86_64 ? 16 : 32)

# Hardware descriptor: the detected machine as a plain NamedTuple const (every field a load-time const →
# the derivation functions below const-fold when called with `_HW`). The functions take a `hw` arg (not the
# globals) so they are PURE and the fleet table is an offline unit test (test/autotune_tests.jl feeds
# galen/Zen4/Zen5/TigerLake descriptors and asserts the measured optima). req#8.
const _HW = (simd = _SIMD_BYTES, l1 = _L1_BYTES, l2 = _L2_BYTES, l3 = _L3_BYTES,
             vendor = _CPU_VENDOR, family = _CPU_FAMILY, nvreg = _NVREG)

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
    nr = _at_gemm_nr(hw, T); min(cld(_ILP_TARGET, nr), (hw.nvreg - 1) ÷ (nr + 1))              # capped by reg budget
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
    kc = _at_gemm_kc(hw, T); _round_dn(((hw.l2 * 3) ÷ 10) ÷ (kc * sizeof(T)), _at_gemm_mr(hw, T) * _lanes(hw, T))
end
@inline function _at_gemm_nc(hw, ::Type{T} = Float64) where {T}
    nr = _at_gemm_nr(hw, T); _avoid_po2(_round_dn((hw.l3 ÷ 4) ÷ (_at_gemm_kc(hw, T) * sizeof(T)), nr), nr)
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
