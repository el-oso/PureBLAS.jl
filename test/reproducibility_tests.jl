# Locks the per-machine reproducibility invariants documented in docs/src/simd.md. PureBLAS kernels
# load from the base pointer with a fixed lane grouping (no alignment peeling) and run single-threaded,
# so a result cannot depend on the data's memory alignment or on scheduling.
#
# Two distinct, both-true properties are asserted separately:
#   1. Run-to-run bit-identity — the same input object, called twice, yields identical bits.
#   2. Alignment invariance WITHIN a code path — the same values at different memory offsets yield
#      identical bits. (A plain `Vector` takes the SIMD path and an offset view takes the generic
#      scalar path; those two paths sum in different orders and are NOT bit-equal to each other — that
#      is normal for any BLAS with a fast path plus a fallback, so it is deliberately NOT asserted.)
#
# Scope is per-machine: a different vector width builds a different reduction tree, so cross-ISA
# equality is not asserted either.

@testsetup module ReproHelp
export shifted, biteq, biteqv
# same values placed contiguously at byte-offset `off` in a fresh buffer → different alignment,
# returned as a unit-stride view (so every offset, incl. 0, takes the SAME code path).
shifted(v, off) = (
    buf = Vector{eltype(v)}(undef, length(v) + off);
    buf[(off + 1):(off + length(v))] .= v; view(buf, (off + 1):(off + length(v)))
)
biteq(a, b) = reinterpret(UInt64, Float64(a)) == reinterpret(UInt64, Float64(b))
biteqv(A, B) = length(A) == length(B) &&
    all(reinterpret(UInt64, vec(collect(A))) .== reinterpret(UInt64, vec(collect(B))))
end

@testitem "Reproducibility: BLAS-1 run-to-run bit-identity" setup = [ReproHelp] begin
    using PureBLAS
    for n in (1000, 4096, 10007)
        x = randn(n); y = randn(n)
        @test biteq(PureBLAS.nrm2(x), PureBLAS.nrm2(x))
        @test biteq(PureBLAS.dot(x, y), PureBLAS.dot(x, y))
        @test biteq(PureBLAS.asum(x), PureBLAS.asum(x))
        @test PureBLAS.iamax(x) == PureBLAS.iamax(x)
    end
end

@testitem "Reproducibility: BLAS-1 alignment invariant within a code path" setup = [ReproHelp] begin
    using PureBLAS
    for n in (1000, 4096, 10007)
        x = randn(n); y = randn(n)
        # off=0 is itself a view → same path as the other offsets; compare offsets to it.
        rN = PureBLAS.nrm2(shifted(x, 0)); rD = PureBLAS.dot(shifted(x, 0), shifted(y, 0))
        rA = PureBLAS.asum(shifted(x, 0)); rI = PureBLAS.iamax(shifted(x, 0))
        for off in 1:7
            @test biteq(PureBLAS.nrm2(shifted(x, off)), rN)
            @test biteq(PureBLAS.dot(shifted(x, off), shifted(y, off)), rD)
            @test biteq(PureBLAS.asum(shifted(x, off)), rA)
            @test PureBLAS.iamax(shifted(x, off)) == rI
        end
    end
end

@testitem "Reproducibility: gemv/gemm run-to-run + gemv alignment invariant" setup = [ReproHelp] begin
    using PureBLAS
    m, k, nrhs = 257, 384, 200
    A = randn(m, k); x = randn(k)
    y1 = zeros(m); y2 = zeros(m)
    PureBLAS.gemv!(y1, A, x); PureBLAS.gemv!(y2, A, x)
    @test biteqv(y1, y2)                                   # gemv run-to-run
    # gemv's x access is path-consistent (x is broadcast; view and Vector take the same kernel), so
    # here the offset result DOES match the Vector reference.
    for off in 1:4
        yo = zeros(m); PureBLAS.gemv!(yo, A, shifted(x, off))
        @test biteqv(yo, y1)                               # gemv invariant to x alignment
    end
    B = randn(k, nrhs); C1 = zeros(m, nrhs); C2 = zeros(m, nrhs)
    PureBLAS.gemm!(C1, A, B); PureBLAS.gemm!(C2, A, B)
    @test biteqv(C1, C2)                                   # gemm run-to-run
end
