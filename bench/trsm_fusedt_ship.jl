# Correctness + A/B of the SHIPPED-path fusedT (Lever 1 on the recursion leaf), via full trsm! routing.
# Usage: taskset -c 8 julia --project=bench bench/trsm_fusedt_ship.jl [aocl]
using LinearAlgebra, Chairmarks, Statistics, Printf
const REF = "aocl" in ARGS ? "aocl" : "openblas"
if REF == "aocl"
    @eval using AOCL_jll
    BLAS.lbt_forward(AOCL_jll.aocl_blas_ilp64; clear = true); BLAS.lbt_forward(AOCL_jll.aocl_lapack_ilp64)
end
BLAS.set_num_threads(1); import PureBLAS as P
tri(s) = (A = triu(randn(Float64, s, s)); @inbounds for i in 1:s; A[i,i] += Float64(s); end; A)

function run_checks()
    println("== correctness: trsm! fusedT-ON vs oracle, and ON vs OFF (PB-vs-PB) ==")
    me = 0.0; mpb = 0.0
    for unit in (false, true), k in (65, 96, 128, 129, 200, 256, 300, 512, 513, 640, 1024, 1025), n in (16, 24, 48, 65, 100, 128, 300)
        A = tri(k); B0 = randn(Float64, k, n)
        ref = (unit ? UnitUpperTriangular(A) : UpperTriangular(A)) \ B0
        dg = unit ? 'U' : 'N'
        Bon = Matrix(B0); P._TRSM_FUSEDT_ON[] = true;  P.trsm!(Bon, A; side='L', uplo='U', transA='N', diag=dg)
        Boff = Matrix(B0); P._TRSM_FUSEDT_ON[] = false; P.trsm!(Boff, A; side='L', uplo='U', transA='N', diag=dg)
        me = max(me, maximum(abs.(Bon .- ref)) / max(1.0, maximum(abs.(ref))))
        mpb = max(mpb, maximum(abs.(Bon .- Boff)) / max(1.0, maximum(abs.(ref))))
    end
    # strided-parent B
    for k in (128, 512, 1024)
        A = tri(k); par = randn(Float64, k + 8, 200); B0 = view(par, 1:k, :); ref = UpperTriangular(A) \ Matrix(B0)
        par2 = randn(Float64, k + 8, 200); Bv = view(par2, 1:k, :); Bv .= B0
        P._TRSM_FUSEDT_ON[] = true; P.trsm!(Bv, A; side='L', uplo='U', transA='N')
        me = max(me, maximum(abs.(Matrix(Bv) .- ref)) / max(1.0, maximum(abs.(ref))))
    end
    P._TRSM_FUSEDT_ON[] = false
    @printf("max err vs oracle = %.2e | max ON-vs-OFF = %.2e => %s\n", me, mpb, mpb <= 1e-13 ? "EQUIV" : "DIVERGENT")
end
run_checks()

const SIZES = (128, 224, 256, 257, 384, 512, 513, 768, 1024)
mk(s) = (tri(s), randn(Float64, s, s))
@noinline ob1(c) = (BLAS.trsm!('L','U','N','N', 1.0, c[1], c[2]); c[2][1])
@noinline pb1(c) = (P.trsm!(c[2], c[1]; side='L', uplo='U', transA='N'); c[2][1])
med(bo, bp) = (to = sort(getproperty.(bo.samples,:time)); tp = sort(getproperty.(bp.samples,:time));
               n = min(length(to),length(tp)); median(to[i]/tp[i] for i in 1:n))
bench(s, reps, f, secs) = @be [mk(s) for _ in 1:reps] (x -> (v=0.0; for c in x; v+=f(c); end; v)) evals=1 samples=40 seconds=secs
println("\n== trsm! vs $REF: fusedT-ON vs shipped-OFF per size ==")
for s in SIZES
    reps = clamp(20_000_000 ÷ (s*s*s), 1, 512); secs = s >= 512 ? 2.0 : 2.5
    r_on = Float64[]; r_off = Float64[]
    for round in 1:3
        bo = bench(s, reps, ob1, secs)
        P._TRSM_FUSEDT_ON[] = true;  bon = bench(s, reps, pb1, secs)
        P._TRSM_FUSEDT_ON[] = false; bof = bench(s, reps, pb1, secs)
        push!(r_on, med(bo, bon)); push!(r_off, med(bo, bof))
    end
    @printf("n=%-5d  fusedT %.3f   shipped %.3f   (Δ %+.1f%%)\n", s, median(r_on), median(r_off),
            100*(median(r_on)/median(r_off)-1))
end
