# ARM SWITCHING MUST QUIET THE OTHER BACKEND.
#
# `bench/plots.jl` measures several arms in one process, in a rotating order. Each arm sets global
# state, and whatever it sets is inherited by the arm measured next unless that arm overrides it.
# Thread count is the dangerous one, because OpenBLAS's idle threads SPIN-WAIT: a PureBLAS arm measured
# after `openblas_mt` competes for its own cores with N spinners.
#
# Measured on Zen4, PureBLAS held at 6 threads and only OpenBLAS's count varied (1 vs 6):
#   a single large Level-3 call is unmoved — gemm/symm/syrk at n=512/1024/2048 read 0.90-1.02x
#   `getri` reads 0.686 ms against 11.526 ms at n=256 — 16x, and it INVERTS the verdict
# The damage follows the call SHAPE, not the op: a driver issuing many small threaded calls pays worker
# wake latency per call, and a contended core multiplies it, while one large call amortises it away.
# That is why it hid for three separate investigations before being found.
#
# The reverse direction was measured too and is clean (OpenBLAS at 6, PureBLAS's pool 1 vs 6: 0.96-1.02x
# on both shapes), because PureBLAS's workers park and sleep rather than spin indefinitely. So the
# requirement is one-directional, and this lint asserts exactly it: the function that selects a PB arm
# must quiet the vendor backend first.
const _SRC = read(joinpath(@__DIR__, "..", "bench", "plots.jl"), String)

# The definition line of `_use_pb!`, whatever its body has grown into.
const _LINE = let
    i = findfirst(r"^_use_pb!\(a::AbstractString\).*$"m, _SRC)
    isnothing(i) ? nothing : _SRC[i]
end

if isnothing(_LINE)
    println("armstate: could not find `_use_pb!` in bench/plots.jl — the arm switch was renamed or")
    println("          restructured. Re-read this lint's header and re-point it; do not delete it.")
    exit(1)
end

if !occursin("BLAS.set_num_threads(1)", _LINE)
    println("armstate: `_use_pb!` does not quiet the vendor backend before timing a PureBLAS arm.")
    println()
    println("  found: ", strip(_LINE))
    println()
    println("It must call `BLAS.set_num_threads(1)`. `_use_ref!` leaves OpenBLAS at `_MT_NT` after a")
    println("threaded reference arm, and OpenBLAS's idle threads spin-wait — so the next PB arm in the")
    println("rotation competes for its own cores. Measured cost: 16x on `getri` at n=256, enough to")
    println("invert the verdict. Single large Level-3 calls are unaffected, which is why this hides.")
    exit(1)
end

println("armstate: the PB arm switch quiets the vendor backend")
exit(0)
