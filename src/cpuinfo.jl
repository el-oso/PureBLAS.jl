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

using CpuId: simdbytes
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

# L1 data-cache size in bytes (folded to a const; fallback if a level reports 0). Unused by the
# bandwidth-bound Level-1 kernels, but L2/L3 blocking for the M2 dgemm will read these.
const _L1_BYTES = let s = Int(cache_size(Val(1)))
    s > 0 ? s : 32 * 1024
end
