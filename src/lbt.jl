# Mode 1 runtime plug-in: reroute LinearAlgebra's BLAS/LAPACK to PureBLAS INSIDE a live Julia
# process, the way MKL.jl reroutes to libmkl_rt in its __init__ — same call sites (`A*B`, `mul!`,
# `cholesky`, `qr`, `svd`, `LinearAlgebra.BLAS.*`), different backend. We register in-process
# `@cfunction` pointers to our native `@ccallable` kernels via `lbt_set_forward` (see cabi_forward.jl).
#
# Why NOT `lbt_forward(libpureblas.so)`: the juliac-trimmed .so embeds its OWN libjulia; dlopen-
# forwarding it double-inits the shared runtime → signal 6. The .so is for NON-Julia hosts. The
# @cfunction path runs against THIS process's runtime, so there is no double-init — it just works.
#
# We do NOT auto-forward at load: the test suite needs OpenBLAS as the correctness oracle, so
# swapping is explicit via activate()/deactivate().

using LinearAlgebra: BLAS
using LinearAlgebra.BLAS: lbt_set_forward, LBT_INTERFACE_ILP64,
    LBT_COMPLEX_RETSTYLE_NORMAL, LBT_F2C_PLAIN

const _DLEXT = Sys.iswindows() ? "dll" : (Sys.isapple() ? "dylib" : "so")

"""Path to the locally-built PureBLAS shared library (juliac/build.jl output; for NON-Julia hosts)."""
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
    activate() -> BLAS.LBTConfig

Reroute LinearAlgebra's BLAS/LAPACK to PureBLAS in the current Julia process by registering
`@cfunction` forwards for every symbol PureBLAS implements (BLAS 1–3 + potrf/getrf/geqrf/gesvd).
After this, `A*B`, `mul!`, `cholesky`, `qr`, `svd`, and `LinearAlgebra.BLAS.*` dispatch to PureBLAS.
Reverse with [`deactivate`](@ref). This is the in-process path — no `libpureblas.so` needed (that is
the drop-in for non-Julia hosts). AD users should call the native `PureBLAS.*` API directly; this
C-ABI forward is not differentiable (no compiled BLAS is).
"""
function activate()
    failed = String[]
    for (name, thunk) in _LBT_REGISTRARS
        r = lbt_set_forward(name, thunk(), LBT_INTERFACE_ILP64,
            LBT_COMPLEX_RETSTYLE_NORMAL, LBT_F2C_PLAIN)
        r == 0 || push!(failed, name)
    end
    isempty(failed) ||
        error("PureBLAS.activate: lbt_set_forward rejected $(length(failed)) symbol(s): $failed")
    return BLAS.get_config()
end

"""
    deactivate() -> BLAS.LBTConfig

Restore the BLAS backend that was loaded at startup (typically OpenBLAS), clearing PureBLAS's
per-symbol forwards.
"""
function deactivate()
    (isassigned(_ORIG_LIBS) && !isempty(_ORIG_LIBS[])) ||
        error("PureBLAS.deactivate: no original BLAS recorded at load time")
    libs = _ORIG_LIBS[]
    BLAS.lbt_forward(first(libs); clear = true)     # clear=true wipes our @cfunction forwards too
    for l in Iterators.drop(libs, 1)
        BLAS.lbt_forward(l; clear = false)
    end
    return BLAS.get_config()
end
