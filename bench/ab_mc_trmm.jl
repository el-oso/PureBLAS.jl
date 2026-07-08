# Controlled A/B for the two req#8 migrations: complex mc (144→64 derived) + trmm_rkc (384→256 derived).
# Run on working-tree (NEW/derived), then `git stash` and run again (OLD literals), same box back-to-back.
# Reuses plots.jl's sweep_heavy methodology (median PB/OB, fresh destructive input per sample). PB/OB ≥ 0.96 gate.
using PureBLAS, LinearAlgebra, Statistics, Chairmarks, Printf
const B = LinearAlgebra.BLAS; B.set_num_threads(1)
const Z = ComplexF64; const ca = one(Z); const cb = zero(Z)
const LT = 'L'; const RT = 'R'; const UP = 'U'; const NN = 'N'
tms(b) = Float64[s.time for s in b.samples]

function sweep_heavy(mk, ob1, pb1, sizes; samples = 64, seconds = 4.0)
    out = Pair{Int,Float64}[]
    for s in sizes
        reps = clamp(20_000_000 ÷ (s * s * s), 1, 512)
        bo = @be [mk(s) for _ in 1:reps] (cs -> (v = 0.0; for c in cs; v += ob1(c); end; v)) evals=1 samples=samples seconds=seconds
        bp = @be [mk(s) for _ in 1:reps] (cs -> (v = 0.0; for c in cs; v += pb1(c); end; v)) evals=1 samples=samples seconds=seconds
        push!(out, s => median(tms(bo)) / median(tms(bp)))
    end
    out
end
cherm(s) = (A = randn(Z, s, s); A + A')
ctri(s) = (A = randn(Z, s, s); triu(A) + I)
dtri(s) = (A = randn(s, s); triu(A) + I)

SZ = (64, 128, 256, 512, 1024)
ops = Pair{String,Any}[
  "zgemm"  => sweep_heavy(s -> (randn(Z,s,s), randn(Z,s,s), zeros(Z,s,s)),
      c -> (B.gemm!(NN,NN,ca,c[1],c[2],cb,c[3]); real(c[3][1])),
      c -> (PureBLAS.gemm!(c[3],c[1],c[2]; alpha=ca, beta=cb); real(c[3][1])), SZ),
  "zhemm"  => sweep_heavy(s -> (cherm(s), randn(Z,s,s), zeros(Z,s,s)),
      c -> (B.hemm!(LT,UP,ca,c[1],c[2],cb,c[3]); real(c[3][1])),
      c -> (PureBLAS.hemm!(c[3],c[1],c[2]; side=LT, uplo=UP, alpha=ca, beta=cb); real(c[3][1])), SZ),
  "zherk"  => sweep_heavy(s -> (randn(Z,s,s), zeros(Z,s,s)),
      c -> (B.herk!(UP,NN,1.0,c[1],0.0,c[2]); real(c[2][1])),
      c -> (PureBLAS.herk!(c[2],c[1]; uplo=UP, trans=NN, alpha=1.0, beta=0.0); real(c[2][1])), SZ),
  "ztrmmR" => sweep_heavy(s -> (ctri(s), randn(Z,s,s)),
      c -> (B.trmm!(RT,UP,NN,NN,ca,c[1],c[2]); real(c[2][1])),
      c -> (PureBLAS.trmm!(c[2],c[1]; side=RT, uplo=UP); real(c[2][1])), SZ),
  "dtrmmR" => sweep_heavy(s -> (dtri(s), randn(s,s)),
      c -> (B.trmm!(RT,UP,NN,NN,1.0,c[1],c[2]); c[2][1]),
      c -> (PureBLAS.trmm!(c[2],c[1]; side=RT, uplo=UP); c[2][1]), SZ),
]
tag = get(ENV, "AB_TAG", "?")
for (nm, r) in ops
    @printf("%-8s %s | %s\n", nm, tag, join(["n=$(p.first):$(round(p.second,digits=3))" for p in r], "  "))
end
