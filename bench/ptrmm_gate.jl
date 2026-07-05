# ztrmm/ztrsm gate sweep (AVX2 packed K-TRIM base) + routed correctness. Run on galen.
using PureBLAS, LinearAlgebra, Statistics, Printf
import LinearAlgebra.BLAS as B
BLAS.set_num_threads(1); const SINK = Ref(ComplexF64(0)); @noinline _run(f) = f()
const T = ComplexF64; const P = PureBLAS
# routed correctness (trmm! → packed on AVX2)
let ok = true, me = 0.0
    for uplo in ('U', 'L'), transA in ('N', 'T', 'C'), diag in ('N', 'U'), k in (8, 17, 64, 128), n in (16, 100)
        A = rand(T, k, k) ./ k; for i in 1:k; A[i, i] = 1 + abs(A[i, i]); end
        B0 = rand(T, k, n)
        ref = LinearAlgebra.BLAS.trmm('L', uplo, transA, diag, one(T), A, copy(B0))
        p = copy(B0); P.trmm!(p, A; side = 'L', uplo = uplo, transA = transA, diag = diag, alpha = one(T))
        e = maximum(abs, p - ref) / max(1, maximum(abs, ref)); me = max(me, e)
        e > 1e-12 * k && (ok = false; @printf("CORRECTNESS FAIL uplo=%s tA=%s diag=%s k=%d n=%d err=%.2e\n", uplo, transA, diag, k, n, e))
    end
    println(ok ? @sprintf("routed trmm correctness OK (maxerr %.2e)", me) : "CORRECTNESS FAILED")
end
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
sizes = (8, 32, 64, 128, 256, 512, 1024, 2048)
for (nm, obf, pbf) in (
    ("ztrmmL", (A, C) -> B.trmm!('L', 'U', 'N', 'N', one(T), A, C),
             (A, C) -> P.trmm!(C, A; side = 'L', uplo = 'U', transA = 'N', diag = 'N')),
    ("ztrmmR", (A, C) -> B.trmm!('R', 'U', 'N', 'N', one(T), A, C),
             (A, C) -> P.trmm!(C, A; side = 'R', uplo = 'U', transA = 'N', diag = 'N')),
    ("ztrsmL", (A, C) -> B.trsm!('L', 'U', 'N', 'N', one(T), A, C),
             (A, C) -> P.trsm!(C, A; side = 'L', uplo = 'U', transA = 'N', diag = 'N')),
    ("ztrsmR", (A, C) -> B.trsm!('R', 'U', 'N', 'N', one(T), A, C),
             (A, C) -> P.trsm!(C, A; side = 'R', uplo = 'U', transA = 'N', diag = 'N')))
    print(rpad(nm, 7))
    for n in sizes
        r = ratio(() -> (tri(n), rand(T, n, n), rep3(n)),   # fresh B per round; in-place reps (timing data-independent)
            c -> (for _ in 1:c[3]; obf(c[1], c[2]); end; c[2][1]),
            c -> (for _ in 1:c[3]; pbf(c[1], c[2]); end; c[2][1]))
        @printf(" %d=%.3f%s", n, r, r < 0.96 ? "*" : " ")
    end
    println()
end
