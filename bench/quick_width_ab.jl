# Does PureBLAS pick the right vector width on this CPU? Times the same calls against MKL three times,
# each in its own Julia process with PureBLAS compiled for a different width assumption:
#
#   detected  what cpuinfo.jl derives on this machine
#   fpw32     `fp_datapath_bytes = 32`: 512-bit registers, but the FMA datapath treated as 256 bits wide,
#             the double-pumped case (the code paths PureBLAS takes on Zen4)
#   simd32    `simd_bytes = 32`: 256-bit vectors throughout, the AVX2 case
#
#     julia -t auto bench/quick_width_ab.jl            # from the repository root; 15-25 minutes
#     julia> include("bench/quick_width_ab.jl")        # or from a REPL started with `julia -t auto`
#
# The two preferences are written only into a temporary environment, which is deleted at the end, so
# nothing in the checkout or the user's Julia setup changes. Each arm compiles PureBLAS once for its
# setting. Ratios are MKL time / PureBLAS time, so ABOVE 1 MEANS PUREBLAS IS FASTER; compare the same row
# across the three blocks. Indicative numbers, as in quick_vs_mkl.jl: plug in, close other work.
using Pkg
let dir = mktempdir(), prev = Base.active_project()
    try
        Pkg.activate(dir; io = devnull)
        Pkg.develop(path = joinpath(@__DIR__, ".."); io = devnull)
        Pkg.add(["MKL", "Chairmarks"]; io = devnull)
        for (arm, pref) in ("detected" => "", "fpw32" => "fp_datapath_bytes = 32", "simd32" => "simd_bytes = 32")
            write(joinpath(dir, "LocalPreferences.toml"), isempty(pref) ? "" : "[PureBLAS]\n$pref\n")
            child = joinpath(@__DIR__, "quick_width_child.jl")
            run(addenv(`$(Base.julia_cmd()) -t $(Threads.nthreads()) --project=$dir $child`, "QUICK_WIDTH_ARM" => arm))
        end
    finally
        Pkg.activate(prev; io = devnull)
        rm(dir; recursive = true, force = true)
    end
end
