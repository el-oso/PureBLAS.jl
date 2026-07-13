# Tier-1 CPU-VALIDATION SMOKE — a portable cliff-detector for the derive-from-hardware tuning (req#8).
#
# Purpose: drop this on ANY CPU (esp. ones not in the fleet) and answer, in ~2-4 min, "did PureBLAS's
# AUTOTUNED block sizes stay ≥ OpenBLAS on this µarch, or did they fall off a cliff?" It is NOT a tuner
# and NOT the full plots.jl sweep — it measures only the ~5 tuning-SENSITIVE ops (gemm/syrk/trsm/trmm/
# potrf) at a few cache-transition sizes, vs OpenBLAS (always present via LBT), single-threaded.
#
# It also PRINTS what cpuinfo.jl detected + derived for this CPU, so a bad probe (e.g. CpuId failing to
# read L3 on a VM/new part → wrong nc) is visible immediately, and you can correct it via Preferences.
#
# Run (no bench env needed — only PureBLAS + stdlib):
#   julia --project=/path/to/PureBLAS.jl bench/cpuvalidate.jl
#   # optional: sizes=256,512,1024   (override the default 256,512,1024,2048)
# Output: printed table + a self-describing  bench/cpuvalidate_<host>.txt  to send back for analysis.

using PureBLAS, LinearAlgebra, Printf, Dates
const B = LinearAlgebra.BLAS
B.set_num_threads(1)                                   # PureBLAS is single-threaded — fair baseline
const P = PureBLAS

# ---- detected + derived tuning consts (what req#8 computed for THIS box) ----------------------------
_c(s) = try string(getfield(P, s)) catch; "?" end
const DETECT = [(:_CPU_VENDOR,"vendor"), (:_SIMD_BYTES,"simd_B"), (:_L1_BYTES,"L1_B"), (:_L2_BYTES,"L2_B"),
                (:_L3_BYTES,"L3_B"), (:_CACHELINE,"line_B"), (:_INTEL_AVX2,"intelAVX2")]
const DERIVED = [(:_MR,"gemm_MR"), (:_NR,"gemm_NR"), (:_NC,"gemm_NC"), (:_KC,"gemm_KC"),
                 (:_GEMM_UNPACK_MAX,"unpack_max"), (:_POTRF_BASE,"potrf_base"), (:_TRSM_BASE,"trsm_base"),
                 (:_CHOL_BASE_SPLIT,"chol_split"), (:_SYRK_PACK_CUT,"syrk_cut"), (:_CSYRK_UNPACK_MAX,"csyrk_unpk")]
wF64 = P._vwidth(Float64)
isa_lbl = _c(:_SIMD_BYTES) == "64" ? "512-bit" : _c(:_SIMD_BYTES) == "32" ? "256-bit/AVX2" : "128-bit"

# ---- measurement: min-of-reps @elapsed ratio OB/PB (coarse is fine for a cliff) ---------------------
reps(n) = n <= 256 ? 30 : n <= 512 ? 12 : n <= 1024 ? 5 : 2
function ratio(pb, ob, n)
    pb(); ob(); pb(); ob()                             # warm (JIT + first-touch workspace grow)
    r = reps(n)
    tp = minimum(@elapsed(pb()) for _ in 1:r)
    to = minimum(@elapsed(ob()) for _ in 1:r)
    (to / tp, tp)
end
mkspd(n) = (M = randn(n, n); M = M * M' + n * I; Matrix(M))
mktri(n) = (T = tril(randn(n, n)); @inbounds for i in 1:n; T[i, i] += n; end; T)

# op → (build inputs for size n) → (pb closure, ob closure, flop-count). Destructive ops re-seed the
# working buffer via copyto! (identical overhead both sides → cancels in the ratio).
ops = [
 ("gemm", n -> begin A=randn(n,n); Bm=randn(n,n); C=zeros(n,n)
        (() -> P.gemm!(C, A, Bm),
         () -> B.gemm!('N','N',1.0,A,Bm,0.0,C), 2.0*n^3) end),
 ("syrk", n -> begin A=randn(n,n); C=zeros(n,n)
        (() -> P.syrk!(C, A; uplo='L', trans='N', alpha=1.0, beta=0.0),
         () -> B.syrk!('L','N',1.0,A,0.0,C), 1.0*n^3) end),
 ("trsm", n -> begin A=mktri(n); B0=randn(n,n); W=similar(B0)
        (() -> (copyto!(W,B0); P.trsm!(W, A; side='L', uplo='L', transA='N', diag='N', alpha=1.0)),
         () -> (copyto!(W,B0); B.trsm!('L','L','N','N',1.0,A,W)), 1.0*n^3) end),
 ("trmm", n -> begin A=mktri(n); B0=randn(n,n); W=similar(B0)
        (() -> (copyto!(W,B0); P.trmm!(W, A; side='L', uplo='L', transA='N', diag='N', alpha=1.0)),
         () -> (copyto!(W,B0); B.trmm!('L','L','N','N',1.0,A,W)), 1.0*n^3) end),
 ("potrf", n -> begin S=mkspd(n); W=similar(S)
        (() -> (copyto!(W,S); P.potrf!(W; uplo='L')),
         () -> (copyto!(W,S); LinearAlgebra.LAPACK.potrf!('L', W)), n^3/3.0) end),
]

sizes = let a = filter(x->startswith(x,"sizes="), ARGS)
    isempty(a) ? [256,512,1024,2048] : parse.(Int, split(split(a[1],"=")[2], ",")) end

host = gethostname()
cpu = try strip(read(pipeline(`lscpu`, `grep -m1 "Model name"`), String)) catch; "?" end
cpu = replace(cpu, r"Model name:\s*"=>"")

io = IOBuffer()
println(io, "#cpuvalidate\tversion=1\thost=$host\tcpu=$cpu\tvendor=$(_c(:_CPU_VENDOR))\tisa=$isa_lbl\twF64=$wF64\tdate=$(Dates.format(now(),"yyyy-mm-ddTHH:MM"))")
print(io, "#detect"); for (s,l) in DETECT; print(io, "\t$l=$(_c(s))"); end; println(io)
print(io, "#derived"); for (s,l) in DERIVED; print(io, "\t$l=$(_c(s))"); end; println(io)
hdr = "op     " * join((@sprintf("n=%-6d", n) for n in sizes)) * "worst  PB_GFlops(maxn)"
println(io, hdr)

allr = Float64[]
for (nm, mk) in ops
    ratios = Float64[]; gflast = 0.0
    for n in sizes
        pb, ob, fl = mk(n)
        r, tp = ratio(pb, ob, n)
        push!(ratios, r); gflast = fl/tp/1e9
    end
    append!(allr, ratios)
    line = @sprintf("%-6s ", nm) * join((@sprintf("%-8.2f", r) for r in ratios)) *
           @sprintf("%-6.2f %.1f", minimum(ratios), gflast)
    println(io, line)
end
gm = exp(sum(log, allr)/length(allr)); worst = minimum(allr)
@printf(io, "SUMMARY  worst=%.2f  geomean=%.2f  GATE(>=0.96)=%s\n", worst, gm, worst>=0.96 ? "PASS" : "FAIL")

out = String(take!(io))
print(out)
path = joinpath(@__DIR__, "cpuvalidate_$(host).txt")
write(path, out)
println("\nwrote $path  — send this back for the extrapolation analysis")
