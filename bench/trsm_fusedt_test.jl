# Correctness + GF for the fused first/last-touch-transpose fullpack slab (Lever 1).
# Usage: taskset -c 8 julia --project=bench bench/trsm_fusedt_test.jl
using LinearAlgebra, Chairmarks, Statistics, Printf; import PureBLAS as P
BLAS.set_num_threads(1)
tri(s) = (
    A = triu(randn(Float64, s, s)); @inbounds for i in 1:s
        A[i, i] += Float64(s)
    end; A
)

function check(k, n; unit = false, pad = 0)
    A = tri(k)
    par = pad == 0 ? nothing : randn(Float64, k + pad, n)
    B0 = pad == 0 ? randn(Float64, k, n) : (view(par, 1:k, :) .= randn(Float64, k, n); view(par, 1:k, :))
    Bref = Matrix(B0)
    ref = (unit ? UnitUpperTriangular(A) : UpperTriangular(A)) \ Bref
    B = pad == 0 ? Matrix(B0) : (par2 = randn(Float64, k + pad, n); v = view(par2, 1:k, :); v .= B0; v)
    Boff = Matrix(B0)
    P._TRSM_FUSEDT_ON[] = false; P._trsm_fused_full_L!(unit, A, Boff)   # validated packed path
    P._TRSM_FUSEDT_ON[] = true;  P._trsm_fused_full_L!(unit, A, B)      # fused-transpose path
    P._TRSM_FUSEDT_ON[] = false
    err = maximum(abs.(Matrix(B) .- ref)) / max(1.0, maximum(abs.(ref)))
    errpb = maximum(abs.(Matrix(B) .- Matrix(Boff))) / max(1.0, maximum(abs.(ref)))  # ON vs OFF (PB-vs-PB)
    return (err, errpb)
end

function run_checks()
    println("== correctness: ON vs oracle (err) AND ON vs OFF/PB-vs-PB (errpb) ==")
    maxerr = 0.0; maxpb = 0.0
    for unit in (false, true), k in (512, 513, 520, 640, 641, 768, 897, 1024, 1025), n in (24, 48, 72, 96, 100, 200, 512)
        e, ep = check(k, n; unit = unit); maxerr = max(maxerr, e); maxpb = max(maxpb, ep)
    end
    for k in (512, 1024), n in (512,)  # strided-parent B
        e, ep = check(k, n; pad = 8); maxerr = max(maxerr, e); maxpb = max(maxpb, ep)
    end
    return @printf(
        "max err vs oracle = %.2e | max ON-vs-OFF (PB-vs-PB) = %.2e => %s\n",
        maxerr, maxpb, maxpb <= 1.0e-13 ? "FUSED==PACKED (equivalent)" : "DIVERGENT"
    )
end
run_checks()

println("\n== fullpack GF: fusedT ON vs OFF (vs dgemm) ==")
function fp_gf(k, n, fusedt)
    A = tri(k); Bs = [randn(Float64, k, n) for _ in 1:8]
    P._TRSM_FUSEDT_ON[] = fusedt
    f(B) = (P._trsm_fused_full_L!(false, A, B); B[1])
    b = @be Bs (
        x -> (
            v = 0.0; for B in x
                v += f(B)
            end; v
        )
    ) evals = 1 samples = 200 seconds = 2.0
    P._TRSM_FUSEDT_ON[] = false
    return median(getproperty.(b.samples, :time)) / length(Bs)
end
function gemm_gf(k, n)
    A = randn(Float64, k, k); Bs = [randn(Float64, k, n) for _ in 1:8]; C = randn(Float64, k, n)
    g(B) = (P.gemm!(C, A, B; alpha = true, beta = false); C[1])
    b = @be Bs (
        x -> (
            v = 0.0; for B in x
                v += g(B)
            end; v
        )
    ) evals = 1 samples = 200 seconds = 2.0
    return 2 * k^2 * n / (median(getproperty.(b.samples, :time)) / length(Bs)) / 1.0e9
end
@printf("%-6s %-6s | OFF GF | ON GF | dgemm | ON/OFF | ON/dgemm\n", "k", "n")
for (k, n) in ((512, 512), (512, 256), (513, 513), (640, 640), (768, 768), (896, 896), (1024, 1024))
    toff = fp_gf(k, n, false); ton = fp_gf(k, n, true); gg = gemm_gf(k, n)
    goff = n * k^2 / toff / 1.0e9; gon = n * k^2 / ton / 1.0e9
    @printf("%-6d %-6d | %6.1f | %5.1f | %5.1f | %.3f  | %.3f\n", k, n, goff, gon, gg, gon / goff, gon / gg)
end
