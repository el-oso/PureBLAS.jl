# zherk / zsyrk per-size gate sweep vs OpenBLAS. Prints ratio (OB/PB); `*` marks below 0.96 gate.
using PureBLAS, LinearAlgebra, Statistics, Printf
import LinearAlgebra.BLAS as B
BLAS.set_num_threads(1); const SINK = Ref(ComplexF64(0)); @noinline _run(f) = f()
const T = ComplexF64
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
sizes = (8, 32, 128, 256, 512, 1024, 2048)
mark(r) = @sprintf("%d=%.3f%s", 0, r, r < 0.96 ? "*" : " ")  # placeholder, replaced below
function sweep(name, obf, pbf)
    print(rpad(name, 7))
    for n in sizes
        r = ratio(() -> (randn(T, n, n), zeros(T, n, n), rep3(n)),
            c -> (for _ in 1:c[3]; obf(c[1], c[2]); end; c[2][1]),
            c -> (for _ in 1:c[3]; pbf(c[1], c[2]); end; c[2][1]))
        @printf(" %d=%.3f%s", n, r, r < 0.96 ? "*" : " ")
    end
    println()
end
# zgemm sanity (already gates), then zherk N/C, zsyrk N/T
sweep("zgemm",
    (A, C) -> B.gemm!('N', 'N', one(T), A, A, zero(T), C),
    (A, C) -> PureBLAS.gemm!(C, A, A))
sweep("zherkN",
    (A, C) -> B.herk!('U', 'N', 1.0, A, 0.0, C),
    (A, C) -> PureBLAS.herk!(C, A; uplo='U', trans='N', alpha=1.0, beta=0.0))
sweep("zherkC",
    (A, C) -> B.herk!('U', 'C', 1.0, A, 0.0, C),
    (A, C) -> PureBLAS.herk!(C, A; uplo='U', trans='C', alpha=1.0, beta=0.0))
sweep("zsyrkN",
    (A, C) -> B.syrk!('U', 'N', one(T), A, zero(T), C),
    (A, C) -> PureBLAS.syrk!(C, A; uplo='U', trans='N', alpha=one(T), beta=zero(T)))
