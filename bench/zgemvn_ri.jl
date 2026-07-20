using PureBLAS, LinearAlgebra, BenchmarkTools
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)
const T = ComplexF64; const PB = PureBLAS
ratio(tp, to) = to / tp

# correctness: α∈{1,cplx}, β∈{0,cplx}, various shapes incl odd/tall/wide
me = 0.0; nf = 0
for (m, n) in ((1, 1), (3, 5), (7, 7), (16, 16), (15, 17), (64, 64), (100, 37), (37, 100), (128, 128), (130, 127), (256, 256), (511, 3)),
        al in (one(T), T(0.9, -0.4)), be in (zero(T), T(0.5, 0.3))
    A = rand(T, m, n); x = rand(T, n); y0 = rand(T, m)
    y1 = copy(y0); PB.gemv!(y1, A, x; alpha = al, beta = be, trans = 'N')
    y2 = copy(y0); BLAS.gemv!('N', al, A, x, be, y2)
    e = norm(y1 - y2) / max(norm(y2), eps())
    (e < 1.0e-11 && !any(isnan, y1)) || (global nf += 1; println("  FAIL m=$m n=$n a=$al b=$be relerr=$e"))
    global me = max(me, e)
end
println("CORRECTNESS zgemvN: maxrelerr=$me  $(nf == 0 ? "PASS" : "*** $nf FAIL ***")")

for (lbl, al, be) in (("a=1,b=0", one(T), zero(T)), ("a=cplx,b=cplx", T(0.9, 0.4), T(0.5, -0.2)))
    println("=== zgemvN $lbl ===")
    for n in (64, 128, 256, 512, 1024, 2048)
        A = rand(T, n, n); x = rand(T, n); y = rand(T, n)
        tp = @belapsed PB.gemv!(Y, $A, $x; alpha = $al, beta = $be, trans = 'N') setup = (Y = copy($y)) evals = 1
        to = @belapsed BLAS.gemv!('N', $al, $A, $x, $be, Y) setup = (Y = copy($y)) evals = 1
        println("  n=$n   PB/OB = ", round(ratio(tp, to), digits = 3))
    end
end
