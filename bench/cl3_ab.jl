# CL3 cluster A/B harness: warm-micro, boost-locked, back-to-back OB vs PB.
# Covers ztrmm/ztrmmR/ztrsm/ztrsmR (all uplo×transA×diag), zsymm/zhemm/zsyr2k/zher2k.
# Usage: julia --project=bench bench/cl3_ab.jl [ops] [sizes]   (defaults below)
using PureBLAS, LinearAlgebra, Statistics, Printf
import LinearAlgebra.BLAS as B
BLAS.set_num_threads(1); const SINK = Ref(ComplexF64(0)); @noinline _run(f) = f()
const T = ComplexF64; const P = PureBLAS

rep3(n) = clamp(20_000_000 ÷ (n^3), 1, 200)
function ratio(mk, ob, pb; rounds = 15)
    rs = Float64[]
    for _ in 1:rounds
        c = mk(); _run(() -> ob(c)); _run(() -> pb(c))
        t0 = time_ns(); SINK[] += ob(c); t1 = time_ns(); SINK[] += pb(c); t2 = time_ns()
        push!(rs, (t1 - t0) / (t2 - t1))
    end
    median(rs)
end
tri(n) = (A = rand(T, n, n) ./ (2n); for i in 1:n; A[i, i] = 1 + abs(A[i, i]); end; A)
herm(n) = (A = rand(T, n, n) ./ (2n); A = A + A'; for i in 1:n; A[i,i] = real(A[i,i]) + n; end; A)

# --- correctness: trmm/trsm over full config space, symm/hemm/syr2k/her2k ---
function corr()
    ok = true; me = 0.0
    for op in (:trmm, :trsm), side in ('L','R'), uplo in ('U','L'), transA in ('N','T','C'), diag in ('N','U'), (k,n) in ((7,5),(16,20),(32,13),(33,64),(64,17),(128,40))
        A = rand(T, k, k) ./ k; for i in 1:k; A[i,i] = 1 + abs(A[i,i]); end
        Bm = side=='L' ? rand(T, k, n) : rand(T, n, k)
        ref = op==:trmm ? B.trmm(side, uplo, transA, diag, one(T), A, copy(Bm)) :
                          B.trsm(side, uplo, transA, diag, one(T), A, copy(Bm))
        p = copy(Bm)
        op==:trmm ? P.trmm!(p, A; side=side, uplo=uplo, transA=transA, diag=diag, alpha=one(T)) :
                    P.trsm!(p, A; side=side, uplo=uplo, transA=transA, diag=diag, alpha=one(T))
        e = maximum(abs, p - ref) / max(1, maximum(abs, ref)); me = max(me, e)
        if e > 1e-10
            ok = false; @printf("FAIL %s side=%c uplo=%c tA=%c diag=%c k=%d n=%d err=%.2e\n", op, side, uplo, transA, diag, k, n, e)
        end
    end
    # F32 spot check
    for k in (16, 33, 64)
        A = rand(ComplexF32, k, k) ./ k; for i in 1:k; A[i,i] = 1 + abs(A[i,i]); end
        Bm = rand(ComplexF32, k, 20)
        ref = B.trmm('L','U','C','N', one(ComplexF32), A, copy(Bm))
        p = copy(Bm); P.trmm!(p, A; side='L', uplo='U', transA='C', diag='N', alpha=one(ComplexF32))
        e = maximum(abs, p - ref) / max(1, maximum(abs, ref)); me = max(me, e)
        e > 1e-4 && (ok = false; @printf("FAIL F32 trmm k=%d err=%.2e\n", k, e))
    end
    println(ok ? @sprintf("correctness OK (maxerr %.2e)", me) : ">>> CORRECTNESS FAILED <<<")
    ok
end

const OPS = Dict(
    "ztrmmL_UN"  => ((A,C)->B.trmm!('L','U','N','N',one(T),A,C), (A,C)->P.trmm!(C,A;side='L',uplo='U',transA='N',diag='N'), :triL),
    "ztrmmL_LN"  => ((A,C)->B.trmm!('L','L','N','N',one(T),A,C), (A,C)->P.trmm!(C,A;side='L',uplo='L',transA='N',diag='N'), :triL),
    "ztrmmL_UC"  => ((A,C)->B.trmm!('L','U','C','N',one(T),A,C), (A,C)->P.trmm!(C,A;side='L',uplo='U',transA='C',diag='N'), :triL),
    "ztrmmL_UT"  => ((A,C)->B.trmm!('L','U','T','N',one(T),A,C), (A,C)->P.trmm!(C,A;side='L',uplo='U',transA='T',diag='N'), :triL),
    "ztrmmR_UN"  => ((A,C)->B.trmm!('R','U','N','N',one(T),A,C), (A,C)->P.trmm!(C,A;side='R',uplo='U',transA='N',diag='N'), :triR),
    "ztrmmR_LN"  => ((A,C)->B.trmm!('R','L','N','N',one(T),A,C), (A,C)->P.trmm!(C,A;side='R',uplo='L',transA='N',diag='N'), :triR),
    "ztrmmR_UC"  => ((A,C)->B.trmm!('R','U','C','N',one(T),A,C), (A,C)->P.trmm!(C,A;side='R',uplo='U',transA='C',diag='N'), :triR),
    "ztrmmR_LC"  => ((A,C)->B.trmm!('R','L','C','N',one(T),A,C), (A,C)->P.trmm!(C,A;side='R',uplo='L',transA='C',diag='N'), :triR),
    "ztrsmL_UN"  => ((A,C)->B.trsm!('L','U','N','N',one(T),A,C), (A,C)->P.trsm!(C,A;side='L',uplo='U',transA='N',diag='N'), :triL),
    "ztrsmR_UN"  => ((A,C)->B.trsm!('R','U','N','N',one(T),A,C), (A,C)->P.trsm!(C,A;side='R',uplo='U',transA='N',diag='N'), :triR),
)

# symm/hemm/syr2k/her2k need their own arg shapes; handle separately.
sizes = length(ARGS) >= 2 ? parse.(Int, split(ARGS[2], ",")) : [8,32,64,128,256,512,1024]
opsel = length(ARGS) >= 1 && ARGS[1] != "all" ? split(ARGS[1], ",") : nothing

corr() || exit(1)

function bench_tri(nm, obf, pbf)
    print(rpad(nm, 11))
    for n in sizes
        r = ratio(() -> (tri(n), rand(T, n, n), rep3(n)),
            c -> (for _ in 1:c[3]; obf(c[1], c[2]); end; c[2][1]),
            c -> (for _ in 1:c[3]; pbf(c[1], c[2]); end; c[2][1]))
        @printf(" %d=%.3f%s", n, r, r < 0.995 ? "*" : " ")
    end
    println()
end

triops = ["ztrmmR_UN","ztrmmR_LN","ztrmmR_UC","ztrmmR_LC","ztrmmL_UN","ztrmmL_LN","ztrmmL_UC","ztrmmL_UT","ztrsmL_UN","ztrsmR_UN"]
for nm in triops
    (opsel === nothing || nm in opsel) || continue
    obf, pbf, _ = OPS[nm]
    bench_tri(nm, obf, pbf)
end

# symm/hemm: C := A*B (A n×n sym/herm, B n×n). syr2k/her2k: C := A*B' + B*A'.
function bench_symm()
    for (nm, obf, pbf, genA) in (
        ("zsymm_L", (A,Bm,C)->B.symm!('L','U',one(T),A,Bm,zero(T),C), (A,Bm,C)->P.symm!(C,A,Bm), tri),
        ("zhemm_L", (A,Bm,C)->B.hemm!('L','U',one(T),A,Bm,zero(T),C), (A,Bm,C)->P.hemm!(C,A,Bm), herm))
        (opsel === nothing || nm in opsel) || continue
        print(rpad(nm, 11))
        for n in sizes
            r = ratio(() -> (genA(n), rand(T,n,n), rand(T,n,n), rep3(n)),
                c -> (for _ in 1:c[4]; obf(c[1], c[2], c[3]); end; c[3][1]),
                c -> (for _ in 1:c[4]; pbf(c[1], c[2], c[3]); end; c[3][1]))
            @printf(" %d=%.3f%s", n, r, r < 0.995 ? "*" : " ")
        end
        println()
    end
end
bench_symm()

function bench_rank2k()
    for (nm, obf, pbf) in (
        ("zsyr2k_U", (A,Bm,C)->B.syr2k!('U','N',one(T),A,Bm,zero(T),C), (A,Bm,C)->P.syr2k!(C,A,Bm;uplo='U',trans='N',alpha=one(T),beta=zero(T))),
        ("zher2k_U", (A,Bm,C)->B.her2k!('U','N',one(T),A,Bm,0.0,C), (A,Bm,C)->P.her2k!(C,A,Bm;uplo='U',trans='N',alpha=one(T),beta=0.0)))
        (opsel === nothing || nm in opsel) || continue
        print(rpad(nm, 11))
        for n in sizes
            r = ratio(() -> (rand(T,n,n), rand(T,n,n), zeros(T,n,n), rep3(n)),
                c -> (for _ in 1:c[4]; obf(c[1], c[2], c[3]); end; c[3][1]),
                c -> (for _ in 1:c[4]; pbf(c[1], c[2], c[3]); end; c[3][1]))
            @printf(" %d=%.3f%s", n, r, r < 0.995 ? "*" : " ")
        end
        println()
    end
end
bench_rank2k()
