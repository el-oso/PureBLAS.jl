# NB sweep for the AVX2 threshold-scan iamax. Explicitly-unrolled thresh kernels (NB=2,4,6,8) generated
# via @eval so each vectorizes (a runtime ntuple/OR loop does NOT vectorize — measured 200x slower).
# Times each vs the active reference BLAS (OpenBLAS default / `aocl` arg). Fresh input per sample.
using LinearAlgebra, Statistics, Printf
using Chairmarks: @be
import PureBLAS
using PureBLAS: Vec, vload, _vwidth
const B = LinearAlgebra.BLAS
B.set_num_threads(1)

const REF = "aocl" in ARGS ? "aocl" : "openblas"
if REF == "aocl"
    using AOCL_jll
    B.lbt_forward(AOCL_jll.aocl_blas_ilp64; clear = true)
end
const REFNAME = REF == "aocl" ? "AOCL" : "OpenBLAS"

# Generate an explicitly-unrolled threshold-scan kernel for a given NB (blocks per hot iter).
for NB in (2, 4, 6, 8)
    fname = Symbol("thresh_", NB, "!")
    loads = [:($(Symbol(:v, j)) = abs(vload(V, xp + (o + $j * W) * sz))) for j in 0:NB-1]
    orexpr = foldl((a, b) -> :($a | $b), [:($(Symbol(:v, j)) > thr) for j in 0:NB-1])
    cold = [quote
        let v = $(Symbol(:v, j))
            bm = v[1]; bl = 1
            for l in 2:W; v[l] > bm && (bm = v[l]; bl = l); end
            bm > gmax && (gmax = bm; bi = o + $j * W + bl; thr = V(gmax))
        end
    end for j in 0:NB-1]
    @eval @inline function $fname(n::Int, xp::Ptr{T}) where {T}
        W = _vwidth(T); V = Vec{W,T}; sz = sizeof(T); step = $NB * W
        gmax = typemin(T); bi = 1; thr = V(gmax); o = 0
        @inbounds while o + step <= n
            $(loads...)
            if any($orexpr)
                $(cold...)
            end
            o += step
        end
        @inbounds while o + W <= n
            v0 = abs(vload(V, xp + o * sz))
            if any(v0 > thr)
                bm = v0[1]; bl = 1
                for l in 2:W; v0[l] > bm && (bm = v0[l]; bl = l); end
                bm > gmax && (gmax = bm; bi = o + bl; thr = V(gmax))
            end
            o += W
        end
        @inbounds for k in (o + 1):n
            a = abs(unsafe_load(xp, k)); a > gmax && (gmax = a; bi = k)
        end
        return bi
    end
end

@noinline pb2(x) = GC.@preserve x thresh_2!(length(x), pointer(x))
@noinline pb4(x) = GC.@preserve x thresh_4!(length(x), pointer(x))
@noinline pb6(x) = GC.@preserve x thresh_6!(length(x), pointer(x))
@noinline pb8(x) = GC.@preserve x thresh_8!(length(x), pointer(x))
@noinline pbc(x) = GC.@preserve x PureBLAS._iamax_chain4!(length(x), pointer(x))
const VARS = (:NB2 => pb2, :NB4 => pb4, :NB6 => pb6, :NB8 => pb8, :chain => pbc)

function check(T)
    for n in (7, 33, 64, 100, 1000, 4099, 100003)
        x = randn(T, n); x[rand(1:n)] *= 50
        ref = argmax(abs.(x))
        pb2(x) == ref || error("NB2 T=$T n=$n"); pb4(x) == ref || error("NB4 T=$T n=$n")
        pb6(x) == ref || error("NB6 T=$T n=$n"); pb8(x) == ref || error("NB8 T=$T n=$n")
        n >= 4 * _vwidth(T) && (pbc(x) == ref || error("chain T=$T n=$n"))
    end
end

med(f, mk; s = 0.15) = median(Float64[smp.time for smp in (@be mk() f evals=1 samples=400 seconds=s).samples])

function sweep(T)
    W = _vwidth(T)
    println("\n== $REFNAME  T=$T  W=$W ==   (ratio = ref_time / pb_time, >1 = PB faster)")
    @printf("%-9s %8s %8s %8s %8s %8s | %10s\n", "n", "NB2", "NB4", "NB6", "NB8", "chain", "ref s/call")
    for n in (1000, 3000, 10000, 30000, 100000, 300000, 1000000)
        reps = clamp(8_000_000 ÷ n, 30, 20000)
        mk = () -> randn(T, n)
        tref = med(x -> (s = 0; for _ in 1:reps; s += B.iamax(x); end; s), mk)
        rs = Float64[]
        for (_, f) in VARS
            t = med(x -> (s = 0; for _ in 1:reps; s += f(x); end; s), mk)
            push!(rs, tref / t)
        end
        @printf("%-9d %8.3f %8.3f %8.3f %8.3f %8.3f | %10.2e\n", n, rs..., tref / reps)
    end
end

for T in (Float64, Float32); check(T); end
println("correctness OK")
for T in (Float64, Float32); sweep(T); end
