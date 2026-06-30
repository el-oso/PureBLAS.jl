# Mode 1 runtime plug-in: forward libblastrampoline to libpureblas.so so all of LinearAlgebra
# routes through PureBLAS. The .so is produced separately by juliac/build.jl; we don't auto-forward
# at load (tests need OpenBLAS as the correctness oracle), so swapping is explicit via activate().

using LinearAlgebra: BLAS

const _DLEXT = Sys.iswindows() ? "dll" : (Sys.isapple() ? "dylib" : "so")

"""Path to the locally-built PureBLAS shared library (juliac/build.jl output)."""
libpureblas_path() = joinpath(@__DIR__, "..", "juliac", "build", "libpureblas." * _DLEXT)

# OpenBLAS (or whatever was loaded at startup), captured so deactivate() can restore it.
const _ORIG_LIBS = Ref{Vector{String}}()

function __init__()
    _ORIG_LIBS[] = try
        String[l.libname for l in BLAS.get_config().loaded_libs]
    catch
        String[]
    end
    return
end

"""
    activate(lib = libpureblas_path())

Forward libblastrampoline to PureBLAS. After this, `LinearAlgebra` BLAS-1 calls dispatch to
PureBLAS. Returns the new `BLAS.get_config()`.
"""
function activate(lib::AbstractString = libpureblas_path())
    isfile(lib) ||
        error("PureBLAS.activate: $lib not found — run `julia juliac/build.jl` first")
    r = BLAS.lbt_forward(lib; clear = false, verbose = false)
    r == 0 || error("PureBLAS.activate: lbt_forward failed (code $r) for $lib")
    return BLAS.get_config()
end

"""
    deactivate()

Restore the BLAS backend that was loaded at startup (typically OpenBLAS).
"""
function deactivate()
    (isassigned(_ORIG_LIBS) && !isempty(_ORIG_LIBS[])) ||
        error("PureBLAS.deactivate: no original BLAS recorded at load time")
    libs = _ORIG_LIBS[]
    BLAS.lbt_forward(first(libs); clear = true)
    for l in Iterators.drop(libs, 1)
        BLAS.lbt_forward(l; clear = false)
    end
    return BLAS.get_config()
end
