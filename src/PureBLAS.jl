module PureBLAS

# Pure-Julia BLAS for the Pure ecosystem. Milestone 1: BLAS Level-1, all four element types via
# generic `T<:Number` kernels, plugged into Julia two ways — directly (AD-friendly native API)
# and as a libblastrampoline drop-in (juliac --trim → libpureblas.so). See ROADMAP.md for L2/L3.

include("core.jl")          # type aliases, _ld/_st! accessors, lassq, |·|
include("cpuinfo.jl")       # SIMD width detection (const-folded, trim-safe)
include("simd_kernels.jl")  # SIMD.jl fast paths (real, unit-stride, dense)
include("level1.jl")        # low-level (n,…,inc) kernels — shared by both modes
include("level2.jl")        # Level-2 gemv/ger/symv/hemv/trmv/trsv kernels (on the L1 column kernels)
include("level2_packed.jl") # Level-2 packed storage: spmv/hpmv/tpmv/tpsv
include("level2_banded.jl") # Level-2 band storage: gbmv/sbmv/hbmv/tbmv/tbsv
include("contracts.jl")     # TypeContracts AbstractBLAS1 / AbstractBLAS2 interfaces
include("backend.jl")       # SIMDBackend: high-level AbstractVector ops (Mode 2)
include("native.jl")        # bare native API → default backend
include("gemm.jl")          # Level-3 GEMM (BLIS 5-loop + SIMD microkernel; generic fallback)
include("level3.jl")        # Level-3 trmm/trsm (recursive blocking, reuses gemm!)
include("lapack.jl")        # LAPACK: Cholesky (potrf) on the gated L3
include("cabi.jl")          # @ccallable Fortran-ABI symbols (Mode 1)
include("lbt.jl")           # activate/deactivate via BLAS.lbt_forward

end # module
