# Measure PB-vs-OB potrf on paths NOT in the fleet plots: F32 lower/upper, complex upper (Lever B/C
# targets), with F64 upper as a Lever-A control. Ratio = OB_time / PB_time (>1 = PB faster = gates).
# Run: julia --project=. bench/lever_probe.jl
using PureBLAS, Chairmarks, LinearAlgebra
import LinearAlgebra.LAPACK
BLAS.set_num_threads(1)

hpd(T, s) = (A = randn(T, s, s); Matrix(A * A' + s * I))   # Hermitian PD
const SZ = (32, 64, 128, 256, 512, 1024, 2048)

function ratio(T, uplo, s)
    A0 = hpd(T, s)
    ob = @be copy(A0) LAPACK.potrf!(uplo, _) evals=1 seconds=0.4
    pb = @be copy(A0) PureBLAS.potrf!(_; uplo=uplo) evals=1 seconds=0.4
    tob = minimum(x.time for x in ob.samples); tpb = minimum(x.time for x in pb.samples)
    tob / tpb
end

for (T, uplo, tag) in ((ComplexF64,'L',"zpotrf-L [ctl]"), (ComplexF64,'U',"zpotrf-U [Lever C]"),
                       (Float64,'L',"dpotrf-L [ctl]"),   (Float64,'U',"dpotrf-U [Lever A]"),
                       (Float32,'L',"spotrf-L"),          (Float32,'U',"spotrf-U"))
    print(rpad(tag, 24))
    for s in SZ
        r = ratio(T, uplo, s)
        printf_r = r < 1.0 ? "*" : " "   # * = below gate
        print(" n$s=", round(r, digits=2), printf_r)
    end
    println()
end
