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

@info "PureBLAS: building trimmed library" OUT
run(cmd)
@info "PureBLAS: built" OUT filesize_bytes = (isfile(OUT) ? filesize(OUT) : 0)
