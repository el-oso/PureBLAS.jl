#!/usr/bin/env julia
# Build bench/tools/build/pureblas-cpufreq via juliac --trim=safe.
#
#   julia bench/tools/build_cpufreq_helper.jl
#   sudo install -o root -g root -m 4755 bench/tools/build/pureblas-cpufreq /usr/local/sbin/
#
# Mirrors juliac/build.jl's invocation, with `--output-exe` instead of `--output-lib`: this is a program
# with a `@main`, not a library of `@ccallable`s. `--trim=safe` is the same setting the .so uses.
#
# ⚠ READ THIS BEFORE INSTALLING IT SETUID. A juliac binary links the Julia runtime shared libraries, and
# a setuid program that loads shared objects from a directory its CALLER can write to is a root hole:
# replace the .so, run the binary, get root. The kernel drops LD_LIBRARY_PATH and LD_PRELOAD for setuid
# binaries, so the danger is not the environment — it is the RPATH baked into the executable, which will
# point at the juliaup tree under $HOME.
#
# So after building, CHECK IT and fix it if needed:
#     readelf -d bench/tools/build/pureblas-cpufreq | grep -E 'RPATH|RUNPATH|NEEDED'
#     ldd bench/tools/build/pureblas-cpufreq
# Every library it resolves must live somewhere only root can write. If any resolve under $HOME, either
#   (a) copy the runtime libs to a root-owned dir and patch RUNPATH to it (`patchelf --set-rpath`), or
#   (b) do not install this setuid — use the C version (bench/tools/pureblas-cpufreq.c), which links only
#       libc from /lib and has no such exposure.
# The C build is 190 lines and needs no runtime; this Julia one exists because the trim toolchain is
# already part of the project and the same source can be read by anyone who reads the rest of it.

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUTDIR = joinpath(@__DIR__, "build")
mkpath(OUTDIR)

const JULIAC = normpath(joinpath(Sys.BINDIR, "..", "share", "julia", "juliac", "juliac.jl"))
isfile(JULIAC) || error("juliac.jl not found at $JULIAC — needs Julia >= 1.12")

const OUT = joinpath(OUTDIR, "pureblas-cpufreq")
const ENTRY = joinpath(@__DIR__, "cpufreq_helper.jl")

cmd = `$(Base.julia_cmd()) --startup-file=no --project=$ROOT $JULIAC
       --output-exe $OUT --experimental --trim=safe --verbose $ENTRY`

println("── building ", OUT)
run(cmd)

println()
println("built: ", OUT, "  (", round(filesize(OUT) / 1024 / 1024; digits = 1), " MB)")
println()
println("BEFORE installing setuid, verify no library resolves under \$HOME:")
println("    readelf -d ", OUT, " | grep -E 'RPATH|RUNPATH'")
println("    ldd ", OUT)
println()
println("then:")
println("    sudo install -o root -g root -m 4755 ", OUT, " /usr/local/sbin/pureblas-cpufreq")
