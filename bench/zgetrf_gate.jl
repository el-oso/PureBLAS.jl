# zgetrf per-size gate sweep vs OpenBLAS (LAPACK.getrf!). Prints ratio (OB/PB); `*` marks below 1.0 gate.
# getrf is destructive → fresh copy per rep. ABBA-ish: pre-copy N buffers, time OB over all, PB over all.
using PureBLAS, LinearAlgebra, Statistics, Printf
import LinearAlgebra.LAPACK as LA
BLAS.set_num_threads(1); const SINK = Ref(0.0); @noinline _run(f) = f()
const T = length(ARGS) >= 1 && ARGS[1] == "c" ? ComplexF32 : ComplexF64
rep3(n) = clamp(30_000_000 ÷ (n^3), 1, 300)

# correctness: PA = LU reconstruct, max rel err
function checkcorrect()
    for n in (8, 32, 63, 128, 200, 256)
        A = randn(T, n, n)
        A2 = copy(A)
        F, ipiv, info = PureBLAS.getrf!(copy(A))
        L = tril(F, -1) + I; U = triu(F)
        R = L * U
        # apply pivots to A: PA
        PA = copy(A)
        for i in 1:n
            if ipiv[i] != i
                PA[i, :], PA[ipiv[i], :] = PA[ipiv[i], :], copy(PA[i, :])
            end
        end
        err = norm(R - PA) / norm(A)
        tol = T <: Complex{Float32} ? 1e-3 : 1e-10
        @printf("  n=%-4d relerr=%.2e %s\n", n, err, err < tol ? "OK" : "FAIL <<<")
    end
end

function ratio(n, rounds)
    reps = rep3(n)
    rs = Float64[]
    for _ in 1:rounds
        # fresh buffers
        base = randn(T, n, n)
        obb = [copy(base) for _ in 1:reps]
        pbb = [copy(base) for _ in 1:reps]
        obp = [Vector{Int}(undef, n) for _ in 1:reps]
        pbp = [Vector{Int}(undef, n) for _ in 1:reps]
        t0 = time_ns()
        for r in 1:reps; LA.getrf!(obb[r], obp[r]); end
        t1 = time_ns()
        for r in 1:reps; PureBLAS.getrf!(pbb[r], pbp[r]); end  # (A, ipiv) 2-arg reuses ipiv
        t2 = time_ns()
        SINK[] += real(obb[1][1]) + real(pbb[1][1])
        push!(rs, (t1 - t0) / (t2 - t1))
    end
    median(rs)
end

println("== correctness ($T) ==")
checkcorrect()
println("== gate ($T)  OB/PB, * = below 1.0 ==")
sizes = (8, 32, 128, 256, 512, 1024, 2048)
print(rpad("zgetrf", 8))
for n in sizes
    r = ratio(n, 15)
    @printf(" %d=%.3f%s", n, r, r < 1.0 ? "*" : " ")
    flush(stdout)
end
println()
