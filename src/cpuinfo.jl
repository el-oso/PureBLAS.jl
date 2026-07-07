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

using CpuId: simdbytes, cpuvendor, cpufeature, cpumodel
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

# Zen4's AVX-512 is DOUBLE-PUMPED (256-bit physical datapath), so the packed complex trmm-on-inverse base
# out-throughputs the direct-recurse side-R solve for WIDE B at n≥256 (measured 1.7× vs 1.3×) — unlike
# Zen5's native-512 datapath or AVX2, which prefer the recurse. `_trsm_right!` keeps the trtri base for
# wide B only on Zen4. Width can't distinguish it (Zen4/Zen5 both W=8), so key on vendor + AVX-512 +
# effective family: AMD family 0x19 (Zen3/Zen4) WITH AVX-512 ⇒ Zen4 (Zen3 shares 0x19 but lacks AVX-512;
# Zen5 is 0x1A). CpuId's :Family byte is the raw base|extended nibbles → effective = base + extended.
# Detected at build (const-folds, trim-safe), overridable via Preferences. AMD-only; else false.
const _ZEN4_AVX512 = let p = @load_preference("zen4_avx512", nothing)
    p isa Bool ? p : try
        cpuvendor() === :AMD && cpufeature(:AVX512F) &&
            (r = Int(cpumodel()[:Family]); (r & 0x0f) + (r >> 4) == 0x19)
    catch
        false
    end
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
