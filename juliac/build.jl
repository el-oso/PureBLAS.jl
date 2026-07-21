#!/usr/bin/env julia
# Build libpureblas.<dlext> via juliac --trim (Julia ≥ 1.12, experimental). The resulting shared
# library exports the BLAS-1 ILP64 symbols (daxpy_64_, …) that libblastrampoline forwards to —
# see PureBLAS.activate(). Run: `julia juliac/build.jl`.

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUTDIR = joinpath(@__DIR__, "build")
mkpath(OUTDIR)

const JULIAC = normpath(joinpath(Sys.BINDIR, "..", "share", "julia", "juliac", "juliac.jl"))
isfile(JULIAC) || error("juliac.jl not found at $JULIAC — needs Julia ≥ 1.12")

const DLEXT = Sys.iswindows() ? "dll" : (Sys.isapple() ? "dylib" : "so")
const OUT = joinpath(OUTDIR, "libpureblas." * DLEXT)
const ENTRY = joinpath(@__DIR__, "entry.jl")

cmd = `$(Base.julia_cmd()) --startup-file=no --project=$ROOT $JULIAC
       --output-lib $OUT --experimental --trim=safe --compile-ccallable --verbose $ENTRY`

# Pin `ger_panel_np` for the trim build ONLY. ger!'s OncePerProcess auto-calibration branch (a runtime
# benchmark) is not trim-safe; setting the preference makes the `@static if` compile that branch out. The
# shipped Project.toml deliberately omits this pref so Mode-2 (in-Julia) users auto-calibrate per µarch — so
# we set it here via Preferences (writing ROOT's gitignored LocalPreferences.toml), scoped to the build, then
# restore the prior state. Value is arbitrary among 1/2/4/8; the .so uses a fixed stream count regardless.
Base.set_active_project(joinpath(ROOT, "Project.toml"))   # so set_preferences! targets ROOT's LocalPreferences
using Preferences
const PUREBLAS_UUID = Base.UUID("cc9e14db-574f-4602-bf53-1167cc4b26d2")
const _prev_ger = load_preference(PUREBLAS_UUID, "ger_panel_np")
set_preferences!(PUREBLAS_UUID, "ger_panel_np" => 4; force = true)

@info "PureBLAS: building trimmed library" OUT
try
    run(cmd)
finally
    if _prev_ger === nothing
        delete_preferences!(PUREBLAS_UUID, "ger_panel_np"; force = true)
    else
        set_preferences!(PUREBLAS_UUID, "ger_panel_np" => _prev_ger; force = true)
    end
end
# Strip DWARF debug info: juliac emits it (`-g1` default) and it dominates the file — ~110 MB of ~154 MB
# is `.debug_*` sections, dead weight in a distributed drop-in. `--strip-debug` keeps the full symbol
# table AND all 492 dynamic `_64_` exports (verified present after strip), so the library is functionally
# identical; only the debug sections go (154 MB → ~48 MB). Best-effort: skip silently if `strip` is absent.
if isfile(OUT) && Sys.which("strip") !== nothing
    presz = filesize(OUT)
    try
        run(`strip --strip-debug $OUT`)
        @info "PureBLAS: stripped debug info" before_MB = round(presz / 2^20; digits = 1) after_MB = round(filesize(OUT) / 2^20; digits = 1)
    catch e
        @warn "PureBLAS: strip failed (keeping debug info)" exception = e
    end
end
@info "PureBLAS: built" OUT filesize_bytes = (isfile(OUT) ? filesize(OUT) : 0)
