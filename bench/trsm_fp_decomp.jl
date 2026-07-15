# Decompose the fullpack sweep: pack+unpack share vs full solve, per k. Plus po2-aliasing test (padded ldb).
# Usage: taskset -c 8 julia --project=bench bench/trsm_fp_decomp.jl
using LinearAlgebra, Chairmarks, Statistics, Printf; import PureBLAS as P
BLAS.set_num_threads(1)
tri(s) = (A = triu(randn(Float64, s, s)); @inbounds for i in 1:s; A[i,i] += Float64(s); end; A)

# pack+unpack-only over all stripes (no solve), mirroring _trsm_fused_full_L!'s stripe loop.
function pack_unpack_only!(buf, B, k, n, NR, sz)
    ldb = stride(B, 2)
    GC.@preserve B buf begin
        pB = pointer(B); Pp = pointer(buf); jc = 0
        while jc < n
            wid = min(NR, n - jc)
            P._fused_packP_tr!(Pp, pB, ldb, jc, wid, k, NR, sz)
            P._fused_unpackP_tr!(Pp, pB, ldb, jc, wid, k, NR, sz)
            jc += NR
        end
    end
    return B[1]
end

med(b, m) = median(getproperty.(b.samples, :time)) / m
function decomp(k, n; pad = 0)
    NR = P._GT_NR; sz = 8
    A = tri(k)
    parent0 = pad == 0
    mkB() = parent0 ? randn(Float64, k, n) : (par = randn(Float64, k + pad, n); view(par, 1:k, :))
    Bs = [mkB() for _ in 1:8]
    buf = Vector{Float64}(undef, k * NR)
    f(B) = (P._trsm_fused_full_L!(false, A, B); B[1])
    bf = @be Bs (x -> (v = 0.0; for B in x; v += f(B); end; v)) evals = 1 samples = 200 seconds = 2.0
    bp = @be Bs (x -> (v = 0.0; for B in x; v += pack_unpack_only!(buf, B, k, n, NR, sz); end; v)) evals = 1 samples = 200 seconds = 2.0
    tf = med(bf, length(Bs)); tp = med(bp, length(Bs))
    gf = n * k^2 / tf / 1e9
    (gf, 100 * tp / tf)
end

@printf("%-6s %-6s | fullpack GF | pack+unpack %% of full\n", "k", "n")
for (k, n) in ((128,128),(256,256),(384,384),(512,512),(768,768),(1024,1024))
    gf, share = decomp(k, n)
    @printf("%-6d %-6d | %9.1f   | %6.1f%%\n", k, n, gf, share)
end
println("\n-- po2 aliasing test: fullpack GF unpadded vs ldb+8 padded --")
@printf("%-6s | unpadded GF | padded(ldb+8) GF | Δ%%\n", "k")
for k in (512, 1024)
    g0, _ = decomp(k, k; pad = 0)
    g1, _ = decomp(k, k; pad = 8)
    @printf("%-6d | %9.1f   | %13.1f    | %+.1f%%\n", k, g0, g1, 100*(g1/g0 - 1))
end
