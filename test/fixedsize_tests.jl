# A dense array that is NOT `Array` must behave exactly like `Array`: same fast path, same results.
#
# `FixedSizeArray` is the concrete case in the ecosystem — `<: DenseArray`, `Memory`-backed, mutable —
# and it is the one this suite pins. The second testitem covers the underlying property without any
# dependency, because the bug these guard was never FixedSizeArrays-specific: `_root` peeled operands
# with `_root(A) = _root(parent(A))`, and `parent` is the identity for anything owning its own buffer,
# so gemm! recursed to a StackOverflowError on ANY dense type that is not `Array`. A `MethodError`
# would have been findable; a stack overflow inside a GC-rooting helper was not.

@testitem "FixedSizeArray rides the strided fast path" begin
    using PureBLAS, LinearAlgebra, FixedSizeArrays
    const FSA = FixedSizeArrays.FixedSizeArray

    n = 16
    A = randn(n, n); B = randn(n, n); x = randn(n)
    spd = A'A + n * I

    Af, Bf = FSA(copy(A)), FSA(copy(B))

    # THE fast-path guarantee, and the reason this type is worth a test at all. `_strided1` gates
    # every pointer/SIMD call site; if it goes false here, FixedSizeArray silently drops to the
    # generic scalar loop and stays correct, so only this assertion would notice.
    @test PureBLAS._strided1(Af)
    @test Af isa DenseArray && Af isa StridedMatrix

    # gemm! is the one that used to StackOverflow.
    Cf = FSA(zeros(n, n))
    PureBLAS.gemm!(Cf, Af, Bf)
    @test collect(Cf) ≈ A * B

    yf = FSA(zeros(n))
    PureBLAS.gemv!(yf, Af, FSA(copy(x)))
    @test collect(yf) ≈ A * x

    yv = FSA(copy(x))
    PureBLAS.axpy!(yv, 2.0, FSA(copy(x)))
    @test collect(yv) ≈ x .+ 2.0 .* x

    @test PureBLAS.dot(FSA(copy(x)), FSA(copy(x))) ≈ dot(x, x)
    @test PureBLAS.nrm2(FSA(copy(x))) ≈ norm(x)

    Sf = FSA(copy(spd))
    PureBLAS.potrf!(Sf; uplo = 'L')
    @test tril(collect(Sf)) ≈ tril(cholesky(Matrix(spd)).L)
end

@testitem "_root terminates on any buffer owner, not just Array" tags = [:unit] begin
    using PureBLAS
    # `parent` is the identity at the bottom, so peeling MUST stop at the fixed point rather than
    # recurse. `Memory` is the exact type FixedSizeArray peels to and the one that used to loop.
    m = Memory{Float64}(undef, 4)
    @test PureBLAS._root(m) === m

    A = randn(4, 4)
    @test PureBLAS._root(A) === A
    @test PureBLAS._root(view(A, 1:2, 1:2)) === A   # a real wrapper still peels to its owner

    # Whatever it returns must be something GC.@preserve can root, which is all the callers do
    # with it — never an isbits value that would make the preserve a silent no-op.
    @test !isbits(PureBLAS._root(m))
end
