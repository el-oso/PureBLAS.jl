# Real-path geqrf! nb-sweep vs reference (OpenBLAS default, or `aocl`). Locked-freq, single-thread.
# Finds the GFlops-optimal nb per size and compares to what _qr_nb derives. Reference measured once per
# size (nb-independent); PB measured per nb. Copy excluded from timing; median of reps; 3 pooled rounds.
using PureBLAS, LinearAlgebra, Statistics, Printf
const REFBK = "aocl" in ARGS ? "aocl" : "openblas"
if REFBK == "aocl"
    ENV["BLIS_NUM_THREADS"] = "1"; ENV["OMP_NUM_THREADS"] = "1"
    @eval using AOCL_jll
    LinearAlgebra.BLAS.lbt_forward(AOCL_jll.aocl_blas_ilp64; clear = true)
    LinearAlgebra.BLAS.lbt_forward(AOCL_jll.aocl_lapack_ilp64)
end
BLAS.set_num_threads(1)
const REFNAME = REFBK == "aocl" ? "AOCL" : "OpenBLAS"

qrflops(m, n) = (k = min(m, n); 2.0 * m * n * k - (2.0 / 3.0) * k^3)  # 2mn·k − 2/3 k³ (m≥n ⇒ 2mn²−2/3n³)

function gflops(f!, A0, flops; reps, warm = 1)
    for _ in 1:warm; B = copy(A0); f!(B); end
    ts = Float64[]
    for _ in 1:reps
        B = copy(A0); t = @elapsed f!(B); push!(ts, t)
    end
    flops / median(ts) / 1e9
end

const SIZES = [(256,256),(512,512),(768,768),(1024,1024),(1536,1536),(2048,2048),
               (3072,3072),(4096,4096),(1024,128),(2048,256),(4096,512)]
const NBS = [32,48,64,96,128,160,192]
repsfor(m,n) = (c = m*n*min(m,n); c > 4e9 ? 5 : c > 8e8 ? 9 : c > 1e8 ? 20 : 40)

@printf("# REF=%s  _L3_BYTES=%d  _L2_BYTES=%d  vw(F64)=%d\n", REFNAME,
        PureBLAS._L3_BYTES, PureBLAS._L2_BYTES, PureBLAS._vwidth(Float64))
println("# size        derived_nb  refGF   ", join([@sprintf("nb%-4d",nb) for nb in NBS]), "  bestNB  bestGF  PB/ref@best  PB/ref@derived")

for (m,n) in SIZES
    flops = qrflops(m,n); reps = repsfor(m,n)
    A0 = randn(m,n)
    dnb = PureBLAS._qr_nb(Float64, m, n)
    # pooled rounds: measure ref + each nb, 3 rounds, take median GFlops per method
    refs = Float64[]; pbs = [Float64[] for _ in NBS]
    for r in 1:3
        tau = zeros(min(m,n))
        push!(refs, gflops(X -> LinearAlgebra.LAPACK.geqrf!(X), A0, flops; reps))
        for (i,nb) in enumerate(NBS)
            push!(pbs[i], gflops(X -> PureBLAS.geqrf!(X, tau; nb=nb), A0, flops; reps))
        end
    end
    refGF = median(refs)
    pbGF = [median(p) for p in pbs]
    bi = argmax(pbGF); bestnb = NBS[bi]; bestGF = pbGF[bi]
    di = findfirst(==(dnb), NBS)
    dGF = isnothing(di) ? gflops(X -> PureBLAS.geqrf!(X, zeros(min(m,n)); nb=dnb), A0, flops; reps) : pbGF[di]
    @printf("%4dx%-5d   %4d      %6.1f  ", m, n, dnb, refGF)
    for g in pbGF; @printf("%6.1f", g); end
    @printf("  %4d   %6.1f   %5.3f       %5.3f\n", bestnb, bestGF, bestGF/refGF, dGF/refGF)
    flush(stdout)
end
