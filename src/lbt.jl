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
using LinearAlgebra.BLAS: lbt_set_forward, lbt_get_forward, LBT_INTERFACE_ILP64,
    LBT_COMPLEX_RETSTYLE_NORMAL, LBT_F2C_PLAIN

const _DLEXT = Sys.iswindows() ? "dll" : (Sys.isapple() ? "dylib" : "so")

# activate() state. `_ACTIVE` tracks whether we forwarded; `_OUR_PTRS` maps each forwarded reference
# symbol → the @cfunction pointer we installed, so status()/is_active() can VERIFY (live) that the
# symbol still routes to PureBLAS (and wasn't cleared by another lbt_forward). Julia-side only — lbt.jl
# is not reachable from the @ccallable roots, so it is never in the trimmed .so.
const _ACTIVE = Ref(false)
const _OUR_PTRS = Dict{String, Ptr{Cvoid}}()

_live_forward(name::AbstractString) =
    lbt_get_forward(name, LBT_INTERFACE_ILP64, LBT_F2C_PLAIN)

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
    empty!(_OUR_PTRS)
    for (name, thunk) in _LBT_REGISTRARS
        p = thunk()
        r = lbt_set_forward(
            name, p, LBT_INTERFACE_ILP64,
            LBT_COMPLEX_RETSTYLE_NORMAL, LBT_F2C_PLAIN
        )
        r == 0 ? (_OUR_PTRS[name] = Base.unsafe_convert(Ptr{Cvoid}, p)) : push!(failed, name)
    end
    isempty(failed) ||
        error("PureBLAS.activate: lbt_set_forward rejected $(length(failed)) symbol(s): $failed")
    _ACTIVE[] = true
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
    _ACTIVE[] = false
    empty!(_OUR_PTRS)
    return BLAS.get_config()
end

"""
    is_active() -> Bool

`true` iff [`activate`](@ref) is in effect AND still live — verified by checking that a canonical
reference symbol (`dgemm_`) currently forwards to the PureBLAS kernel we installed (so it also returns
`false` if another `lbt_forward`/`deactivate` cleared our forwards since `activate`).

Note: `LinearAlgebra.BLAS.get_config()` is NOT a reliable check — it lists the *loaded libraries*
(OpenBLAS stays loaded), while `activate()` overlays per-symbol `@cfunction` forwards on top of it. Use
`is_active()` / [`status`](@ref) instead.
"""
function is_active()
    _ACTIVE[] || return false
    p = get(_OUR_PTRS, "dgemm_", C_NULL)
    p == C_NULL && return false
    return _live_forward("dgemm_") == p
end

"""
    status([io::IO = stdout])

Report whether PureBLAS is handling BLAS/LAPACK in this process, how many symbols are live-routed to
PureBLAS kernels, and why `LinearAlgebra.BLAS.get_config()` still lists `libopenblas` (it is the loaded
library; `activate()` forwards individual symbols on top of it — it does not swap the library).
"""
function status(io::IO = stdout)
    reg = length(_LBT_REGISTRARS)
    if !_ACTIVE[]
        printstyled(io, "PureBLAS: INACTIVE\n"; bold = true, color = :yellow)
        println(io, "  Call PureBLAS.activate() to forward $reg BLAS/LAPACK symbols to PureBLAS.")
        return nothing
    end
    live = 0
    for (name, p) in _OUR_PTRS
        _live_forward(name) == p && (live += 1)
    end
    printstyled(io, "PureBLAS: ACTIVE"; bold = true, color = :green)
    println(io, " — $live/$reg forwarded symbols currently routed to PureBLAS kernels.")
    println(io, "  Every high-level LinearAlgebra op (*, \\, mul!, lu, cholesky, qr, svd, eigen, schur,")
    println(io, "  eigen(A,B), svd(A,B), least-squares, …) dispatches to PureBLAS. Low-level auxiliary")
    println(io, "  symbols no high-level op reaches (larf, gebrd, hseqr, tgsen, …) remain OpenBLAS-backed;")
    println(io, "  see the LAPACK/BLAS coverage docs for the exact list.")
    println(io, "  Note: BLAS.get_config() still lists libopenblas — that is the LOADED library. activate()")
    println(io, "  overlays per-symbol @cfunction forwards (lbt_set_forward); it does not swap the loaded")
    println(io, "  library, so get_config cannot report PureBLAS. This status / is_active() is the check.")
    return nothing
end
