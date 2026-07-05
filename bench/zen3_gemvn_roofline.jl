# zgemvN roofline (Fable decomp item C): is the mid-n loss kernel/ports or bandwidth?
using PureBLAS, LinearAlgebra, Statistics, Printf
import LinearAlgebra.BLAS as B
BLAS.set_num_threads(1)
const T = ComplexF64; const P = PureBLAS; const SINK = Ref(0.0)
@noinline _run(f) = f()
function med(mk; rounds = 25)
    rs = Float64[]
    for _ in 1:rounds
        c = mk(); _run(c); t = time_ns(); v = c(); SINK[] += v; push!(rs, Float64(time_ns() - t))
    end
    median(rs)
end
println("== zgemvN roofline — GB/s (A read once = 16·m·n bytes) vs pure A-stream ==")
for n in (256, 512, 1024, 2048, 4096)
    m = n; reps = clamp(200_000_000 ÷ (m * n), 20, 3000)
    A = randn(T, m, n); x = randn(T, n); y = randn(T, m)
    Ar = reinterpret(Float64, vec(A)); bytes = 16.0 * m * n
    tob = med(() -> (yy = copy(y); () -> (for _ in 1:reps; B.gemv!('N', one(T), A, x, zero(T), yy); end; real(yy[1]))))
    tpb = med(() -> (yy = copy(y); () -> (for _ in 1:reps; P.gemv!(yy, A, x; alpha = one(T), beta = zero(T)); end; real(yy[1]))))
    tst = med(() -> (() -> sum(Ar)))   # pure A-stream reference (SIMD reduction, ~memory-bound)
    gob = bytes * reps / tob; gpb = bytes * reps / tpb; gst = bytes / tst
    @printf("n=%-5d OB=%.0f  PB=%.0f  stream=%.0f GB/s   PB/OB=%.2f  PB/stream=%.2f  OB/stream=%.2f\n",
        n, gob, gpb, gst, tob / tpb, gpb / gst, gob / gst)
end
println("SINK=", SINK[])
