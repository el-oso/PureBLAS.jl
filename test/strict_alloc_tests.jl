# StrictMode coverage of the allocation, ownership and escape guarantees.
#
# The BLAS-1/2/3 and LAPACK `!` entries listed in `AllocT` are measured at steady state: the call runs
# three times to warm every grow-once pool, then the allocation counter is read over five windows (and
# once more over five if the first reading is nonzero), inside a function over concrete operands. `AllocT.measure` is the one instrument; each table row builds its operands and
# returns the byte count of one warmed call. The same instrument is checked against a call that does
# allocate, so a zero cannot come from a dead harness.
#
# Forms whose signature has no output buffer (`getrf!(A)`, `gees!(job, A)`, ...) return freshly
# allocated results by construction; they are swept as "(allocating form)" rows and marked broken.

@testmodule AllocT begin
    using PureBLAS, LinearAlgebra, Random, ForwardDiff
    const P = PureBLAS
    const D64 = ForwardDiff.Dual{Nothing, Float64, 1}
    export P, measure, ENTRIES, FLOATS, COMPLEXES, ALLT

    const FLOATS = (Float32, Float64)
    const COMPLEXES = (ComplexF32, ComplexF64)
    const ALLT = (Float32, Float64, ComplexF32, ComplexF64)

    # Worst of five windows after three warmups. A nonzero reading is taken again once before it is
    # believed: a one-off figure (a first call down a rarely taken branch compiles it, and the compiler
    # allocates) does not repeat, a per-call allocation does. `run` is a concrete closure, so the call
    # inside the window is statically dispatched and its result (a tuple, a scalar) is not boxed by
    # the harness.
    function window(run::F) where {F}
        b = 0
        for _ in 1:5
            b = max(b, @allocated run())
        end
        return b
    end
    function measure(run::F) where {F}
        run(); run(); run()
        b = window(run)
        return iszero(b) ? b : window(run)
    end

    rnd(T, m, n) = randn(T, m, n)
    rnd(::Type{D64}, m, n) = ForwardDiff.Dual{Nothing}.(randn(m, n), randn(m, n))
    rnd(::Type{D64}, n) = ForwardDiff.Dual{Nothing}.(randn(n), randn(n))
    rnd(T, n) = randn(T, n)
    function spd(T, n)
        A = rnd(T, n, n)
        S = A * A' + n * I
        return Matrix{T}(S)
    end
    function sym(T, n)
        A = rnd(T, n, n)
        return Matrix{T}(A + transpose(A))
    end
    herm(T, n) = (A = rnd(T, n, n); Matrix{T}(A + A'))
    function tri(T, n, uplo)
        A = rnd(T, n, n) ./ T(2n)
        for i in 1:n
            A[i, i] = one(T) + abs(A[i, i])
        end
        return Matrix{T}(uplo == 'U' ? triu(A) : tril(A))
    end
    # Band storage of an `n`-column SPD/Hermitian band with `k` off-diagonals (diagonally dominant).
    function bandspd(T, n, k, uplo)
        AB = zeros(T, k + 1, n)
        for j in 1:n
            if uplo == 'U'
                AB[k + 1, j] = T(4k + 2)
                for d in 1:min(k, j - 1)
                    AB[k + 1 - d, j] = T(0.5)
                end
            else
                AB[1, j] = T(4k + 2)
                for d in 1:min(k, n - j)
                    AB[1 + d, j] = T(0.5)
                end
            end
        end
        return AB
    end
    packed(T, n, uplo, M) = T[uplo == 'U' ? M[i, j] : M[j, i] for j in 1:n for i in (uplo == 'U' ? (1:j) : (j:n))]

    isc(T) = T <: Complex
    realt(T) = real(T)

    const ENTRIES = Tuple{String, Tuple, Int, Function}[]
    # name, element types, size, body(T, n) -> bytes
    # `$x` in an entry's arguments stands for the loop variable `x` of the enclosing `for`.
    unq(ex) = ex isa Expr ? (ex.head === :$ ? ex.args[1] : Expr(ex.head, map(unq, ex.args)...)) : ex
    macro entry(name, types, n, body)
        return esc(:(push!(ENTRIES, ($(unq(name)), $(unq(types)), $(unq(n)), (T, n) -> $(unq(body))))))
    end
    export @entry

    # Entries measured to allocate (or throw) at steady state. Each is `@test_broken` in the sweep,
    # so the day one is fixed the sweep reports an unexpected pass and the row must be deleted.
    const KNOWN = Dict{Tuple{String, DataType}, String}(
        ("gesvd_vals!", Float32) => "generic real path allocates (src/lapack/svd.jl:1655)",
        ("gesvd_vals!", ComplexF32) => "complex path allocates (src/lapack/svd.jl:1719)",
        ("gesvd_vals!", ComplexF64) => "complex path allocates (src/lapack/svd.jl:1719)",
        ("gebrd!", Float32) => "MethodError: no method _svdws(Float32) (src/lapack/svd.jl:705)",
        ("_syev! (internal driver)", Float32) => "allocates d, e, tau, Z per call (src/lapack/eigen.jl:1206)",
        ("_syev! (internal driver)", Float64) => "allocates d, e, tau, Z per call (src/lapack/eigen.jl:1206)",
        ("_heev! (internal driver)", ComplexF32) => "allocates per call (src/lapack/eigen.jl:1361)",
        ("_heev! (internal driver)", ComplexF64) => "allocates per call (src/lapack/eigen.jl:1361)",
    )
    const ONLY = get(ENV, "ALLOC_ONLY", "")


    # ── BLAS-1 ──────────────────────────────────────────────────────────────────────────────
    @entry "axpy!" ALLT 1000 (x = rnd(T, n); y = rnd(T, n); a = T(1.5); measure(() -> P.axpy!(y, a, x)))
    @entry "scal!" ALLT 1000 (x = rnd(T, n); a = T(0.999); measure(() -> P.scal!(a, x)))
    @entry "blascopy!" ALLT 1000 (x = rnd(T, n); y = rnd(T, n); measure(() -> P.blascopy!(y, x)))
    @entry "swap!" ALLT 1000 (x = rnd(T, n); y = rnd(T, n); measure(() -> P.swap!(x, y)))
    @entry "dot" ALLT 1000 (x = rnd(T, n); y = rnd(T, n); measure(() -> P.dot(x, y)))
    @entry "dotu" ALLT 1000 (x = rnd(T, n); y = rnd(T, n); measure(() -> P.dotu(x, y)))
    @entry "nrm2" ALLT 1000 (x = rnd(T, n); measure(() -> P.nrm2(x)))
    @entry "asum" ALLT 1000 (x = rnd(T, n); measure(() -> P.asum(x)))
    @entry "iamax" ALLT 1000 (x = rnd(T, n); measure(() -> P.iamax(x)))

    # ── BLAS-2 ──────────────────────────────────────────────────────────────────────────────
    for tr in ('N', 'T', 'C')
        @entry "gemv! trans=$tr" ALLT 64 begin
            A = rnd(T, n, n); x = rnd(T, n); y = rnd(T, n); a = T(1.5); b = T(0.5)
            measure(() -> P.gemv!(y, A, x; alpha = a, beta = b, trans = $tr))
        end
    end
    @entry "gemv! 200x37 N (tall)" ALLT 200 begin
        A = rnd(T, n, 37); x = rnd(T, 37); y = rnd(T, n)
        measure(() -> P.gemv!(y, A, x; alpha = one(T), beta = zero(T), trans = 'N'))
    end
    @entry "gemv! 200x37 T (tall)" ALLT 200 begin
        A = rnd(T, n, 37); x = rnd(T, n); y = rnd(T, 37)
        measure(() -> P.gemv!(y, A, x; alpha = one(T), beta = zero(T), trans = 'T'))
    end
    @entry "ger!" ALLT 64 (A = rnd(T, n, n); x = rnd(T, n); y = rnd(T, n); measure(() -> P.ger!(T(0.1), x, y, A)))
    @entry "ger! conj" COMPLEXES 64 (A = rnd(T, n, n); x = rnd(T, n); y = rnd(T, n); measure(() -> P.ger!(T(0.1), x, y, A; conj = true)))
    for ul in ('U', 'L')
        @entry "symv! $ul" ALLT 64 (A = sym(T, n); x = rnd(T, n); y = rnd(T, n); measure(() -> P.symv!(y, A, x; uplo = $ul)))
        @entry "hemv! $ul" COMPLEXES 64 (A = herm(T, n); x = rnd(T, n); y = rnd(T, n); measure(() -> P.hemv!(y, A, x; uplo = $ul)))
    end
    for (ul, tr, dg) in (('U', 'N', 'N'), ('L', 'N', 'U'), ('U', 'T', 'N'), ('L', 'C', 'N'))
        @entry "trmv! $ul$tr$dg" ALLT 64 (A = tri(T, n, $ul); x = rnd(T, n); x0 = copy(x); measure(() -> (copyto!(x, x0); P.trmv!(A, x; uplo = $ul, trans = $tr, diag = $dg))))
        @entry "trsv! $ul$tr$dg" ALLT 64 (A = tri(T, n, $ul); x = rnd(T, n); x0 = copy(x); measure(() -> (copyto!(x, x0); P.trsv!(A, x; uplo = $ul, trans = $tr, diag = $dg))))
    end
    for ul in ('U', 'L')
        @entry "spmv! $ul" ALLT 64 (M = sym(T, n); AP = packed(T, n, $ul, M); x = rnd(T, n); y = rnd(T, n); measure(() -> P.spmv!(y, AP, x; uplo = $ul)))
        @entry "hpmv! $ul" COMPLEXES 64 (M = herm(T, n); AP = packed(T, n, $ul, M); x = rnd(T, n); y = rnd(T, n); measure(() -> P.hpmv!(y, AP, x; uplo = $ul)))
        @entry "spr! $ul" ALLT 64 (AP = packed(T, n, $ul, sym(T, n)); x = rnd(T, n); measure(() -> P.spr!(T(0.1), x, AP; uplo = $ul)))
        @entry "spr2! $ul" ALLT 64 (AP = packed(T, n, $ul, sym(T, n)); x = rnd(T, n); y = rnd(T, n); measure(() -> P.spr2!(T(0.1), x, y, AP; uplo = $ul)))
        @entry "hpr! $ul" COMPLEXES 64 (AP = packed(T, n, $ul, herm(T, n)); x = rnd(T, n); measure(() -> P.hpr!(real(T)(0.1), x, AP; uplo = $ul)))
        @entry "hpr2! $ul" COMPLEXES 64 (AP = packed(T, n, $ul, herm(T, n)); x = rnd(T, n); y = rnd(T, n); measure(() -> P.hpr2!(T(0.1), x, y, AP; uplo = $ul)))
        @entry "tpmv! $ul" ALLT 64 (AP = packed(T, n, $ul, tri(T, n, $ul)); x = rnd(T, n); x0 = copy(x); measure(() -> (copyto!(x, x0); P.tpmv!(AP, x; uplo = $ul))))
        @entry "tpsv! $ul" ALLT 64 (AP = packed(T, n, $ul, tri(T, n, $ul)); x = rnd(T, n); x0 = copy(x); measure(() -> (copyto!(x, x0); P.tpsv!(AP, x; uplo = $ul))))
    end
    @entry "gbmv!" ALLT 64 (AB = rnd(T, 5, n); x = rnd(T, n); y = rnd(T, n); measure(() -> P.gbmv!(y, AB, x, n, 2, 2)))
    @entry "gbmv! T" ALLT 64 (AB = rnd(T, 5, n); x = rnd(T, n); y = rnd(T, n); measure(() -> P.gbmv!(y, AB, x, n, 2, 2; trans = 'T')))
    for ul in ('U', 'L')
        @entry "sbmv! $ul" ALLT 64 (AB = rnd(T, 4, n); x = rnd(T, n); y = rnd(T, n); measure(() -> P.sbmv!(y, AB, x; uplo = $ul)))
        @entry "hbmv! $ul" COMPLEXES 64 (AB = rnd(T, 4, n); x = rnd(T, n); y = rnd(T, n); measure(() -> P.hbmv!(y, AB, x; uplo = $ul)))
        @entry "tbmv! $ul" ALLT 64 (AB = rnd(T, 4, n); x = rnd(T, n); x0 = copy(x); measure(() -> (copyto!(x, x0); P.tbmv!(AB, x; uplo = $ul))))
        @entry "tbsv! $ul" ALLT 64 (AB = bandspd(T, n, 3, $ul); x = rnd(T, n); x0 = copy(x); measure(() -> (copyto!(x, x0); P.tbsv!(AB, x; uplo = $ul))))
    end

    # ── BLAS-3 (small, blocked, and a size past the Strassen/pool thresholds) ────────────────
    for n3 in (24, 160, 320)
        @entry "gemm! NN n=$n3" ALLT $n3 (A = rnd(T, n, n); B = rnd(T, n, n); C = rnd(T, n, n); measure(() -> P.gemm!(C, A, B; alpha = T(1.5), beta = T(0.5))))
        @entry "gemm! TN n=$n3" ALLT $n3 (A = rnd(T, n, n); B = rnd(T, n, n); C = rnd(T, n, n); measure(() -> P.gemm!(C, A, B; alpha = one(T), beta = zero(T), transA = 'T', transB = 'N')))
        @entry "gemm! CT n=$n3" COMPLEXES $n3 (A = rnd(T, n, n); B = rnd(T, n, n); C = rnd(T, n, n); measure(() -> P.gemm!(C, A, B; alpha = one(T), beta = zero(T), transA = 'C', transB = 'T')))
        for (sd, ul) in (('L', 'U'), ('R', 'L'))
            @entry "symm! $sd$ul n=$n3" ALLT $n3 (A = sym(T, n); B = rnd(T, n, n); C = rnd(T, n, n); measure(() -> P.symm!(C, A, B; side = $sd, uplo = $ul, alpha = one(T), beta = T(0.5))))
            @entry "hemm! $sd$ul n=$n3" COMPLEXES $n3 (A = herm(T, n); B = rnd(T, n, n); C = rnd(T, n, n); measure(() -> P.hemm!(C, A, B; side = $sd, uplo = $ul, alpha = one(T), beta = T(0.5))))
        end
        for (ul, tr) in (('U', 'N'), ('L', 'T'))
            @entry "syrk! $ul$tr n=$n3" ALLT $n3 (A = rnd(T, n, n); C = sym(T, n); measure(() -> P.syrk!(C, A; uplo = $ul, trans = $tr, alpha = one(T), beta = T(0.5))))
            @entry "syr2k! $ul$tr n=$n3" ALLT $n3 (A = rnd(T, n, n); B = rnd(T, n, n); C = sym(T, n); measure(() -> P.syr2k!(C, A, B; uplo = $ul, trans = $tr, alpha = one(T), beta = T(0.5))))
        end
        for (ul, tr) in (('U', 'N'), ('L', 'C'))
            @entry "herk! $ul$tr n=$n3" COMPLEXES $n3 (A = rnd(T, n, n); C = herm(T, n); R = real(T); measure(() -> P.herk!(C, A; uplo = $ul, trans = $tr, alpha = R(1), beta = R(0.5))))
            @entry "her2k! $ul$tr n=$n3" COMPLEXES $n3 (A = rnd(T, n, n); B = rnd(T, n, n); C = herm(T, n); R = real(T); measure(() -> P.her2k!(C, A, B; uplo = $ul, trans = $tr, alpha = one(T), beta = R(0.5))))
        end
        for (sd, ul, ta) in (('L', 'L', 'N'), ('L', 'U', 'T'), ('R', 'U', 'N'), ('R', 'L', 'C'))
            ta2 = ta
            @entry "trsm! $sd$ul$ta n=$n3" ALLT $n3 (A = tri(T, n, $ul); B = rnd(T, n, n); B0 = copy(B); measure(() -> (copyto!(B, B0); P.trsm!(A, B; side = $sd, uplo = $ul, transA = $ta, diag = 'N', alpha = one(T)))))
            @entry "trmm! $sd$ul$ta n=$n3" ALLT $n3 (A = tri(T, n, $ul); B = rnd(T, n, n); B0 = copy(B); measure(() -> (copyto!(B, B0); P.trmm!(A, B; side = $sd, uplo = $ul, transA = $ta, diag = 'N', alpha = one(T)))))
        end
    end


    const LAP0 = length(ENTRIES)
    # ── LAPACK, caller-supplied outputs (blocked sizes included) ─────────────────────────────
    trc(T) = isc(T) ? 'C' : 'T'
    for nl in (24, 96, 200)
        for ul in ('U', 'L')
            @entry "potrf! $ul n=$nl" ALLT $nl begin
                A0 = spd(T, n); A = similar(A0)
                measure(() -> (copyto!(A, A0); P.potrf!(A; uplo = $ul)))
            end
        end
        @entry "getrf! n=$nl" ALLT $nl begin
            A0 = rnd(T, n, n); A = similar(A0); ipiv = zeros(Int, n)
            measure(() -> (copyto!(A, A0); P.getrf!(A, ipiv)))
        end
        @entry "getrf! wide n=$nl" ALLT $nl begin
            A0 = rnd(T, n, n + 17); A = similar(A0); ipiv = zeros(Int, n)
            measure(() -> (copyto!(A, A0); P.getrf!(A, ipiv)))
        end
        @entry "geqrf! n=$nl" ALLT $nl begin
            A0 = rnd(T, n + 8, n); A = similar(A0); tau = zeros(T, n)
            measure(() -> (copyto!(A, A0); P.geqrf!(A, tau)))
        end
        @entry "gelqf! n=$nl" ALLT $nl begin
            A0 = rnd(T, n, n + 8); A = similar(A0); tau = zeros(T, n)
            measure(() -> (copyto!(A, A0); P.gelqf!(A, tau)))
        end
    end
    @entry "potrs!" ALLT 48 begin
        F = spd(T, n); P.potrf!(F; uplo = 'L'); B0 = rnd(T, n, 5); B = similar(B0)
        measure(() -> (copyto!(B, B0); P.potrs!(F, B; uplo = 'L')))
    end
    @entry "potri!" ALLT 48 begin
        F = spd(T, n); P.potrf!(F; uplo = 'L'); A = similar(F)
        measure(() -> (copyto!(A, F); P.potri!(A; uplo = 'L')))
    end
    @entry "getrs!" ALLT 48 begin
        F = rnd(T, n, n); ipiv = zeros(Int, n); P.getrf!(F, ipiv); B0 = rnd(T, n, 5); B = similar(B0)
        measure(() -> (copyto!(B, B0); P.getrs!(F, ipiv, B; trans = 'N')))
    end
    @entry "getri!" ALLT 48 begin
        F = rnd(T, n, n); ipiv = zeros(Int, n); P.getrf!(F, ipiv); A = similar(F)
        measure(() -> (copyto!(A, F); P.getri!(A, ipiv)))
    end
    @entry "trtri!" ALLT 48 begin
        A0 = tri(T, n, 'U'); A = similar(A0)
        measure(() -> (copyto!(A, A0); P.trtri!(A; uplo = 'U', diag = 'N')))
    end
    @entry "trtrs!" ALLT 48 begin
        A = tri(T, n, 'L'); B0 = rnd(T, n, 5); B = similar(B0)
        measure(() -> (copyto!(B, B0); P.trtrs!(A, B; uplo = 'L', trans = 'N', diag = 'N')))
    end
    @entry "ormqr!" ALLT 48 begin
        A = rnd(T, n + 8, n); tau = zeros(T, n); P.geqrf!(A, tau); C0 = rnd(T, n + 8, 6); C = similar(C0)
        measure(() -> (copyto!(C, C0); P.ormqr!('L', trc(T), A, tau, C)))
    end
    @entry "orgqr!" ALLT 48 begin
        A0 = rnd(T, n + 8, n); tau = zeros(T, n); P.geqrf!(A0, tau); A = similar(A0)
        measure(() -> (copyto!(A, A0); P.orgqr!(A, tau)))
    end
    @entry "ormlq!" ALLT 48 begin
        A = rnd(T, n, n + 8); tau = zeros(T, n); P.gelqf!(A, tau); C0 = rnd(T, n + 8, 6); C = similar(C0)
        measure(() -> (copyto!(C, C0); P.ormlq!('L', 'N', A, tau, C)))
    end
    @entry "orglq!" ALLT 48 begin
        A0 = rnd(T, n, n + 8); tau = zeros(T, n); P.gelqf!(A0, tau); A = similar(A0)
        measure(() -> (copyto!(A, A0); P.orglq!(A, tau)))
    end
    @entry "geqlf!/orgql!" ALLT 48 begin
        A0 = rnd(T, n + 8, n); tau = zeros(T, n); A = similar(A0)
        measure(() -> (copyto!(A, A0); P.geqlf!(A, tau); P.orgql!(A, tau)))
    end
    @entry "gerqf!/orgrq!" ALLT 48 begin
        A0 = rnd(T, n, n + 8); tau = zeros(T, n); A = similar(A0)
        measure(() -> (copyto!(A, A0); P.gerqf!(A, tau); P.orgrq!(A, tau)))
    end
    for ul in ('U', 'L')
        @entry "sytrf!/sytrs!/sytri! $ul" ALLT 48 begin
            A0 = sym(T, n); A = similar(A0); ipiv = zeros(Int, n); B0 = rnd(T, n, 4); B = similar(B0)
            measure(() -> (copyto!(A, A0); P.sytrf!(A, ipiv; uplo = $ul); copyto!(B, B0); P.sytrs!(A, ipiv, B; uplo = $ul); P.sytri!(A, ipiv; uplo = $ul)))
        end
        @entry "hetrf!/hetrs!/hetri! $ul" COMPLEXES 48 begin
            A0 = herm(T, n); A = similar(A0); ipiv = zeros(Int, n); B0 = rnd(T, n, 4); B = similar(B0)
            measure(() -> (copyto!(A, A0); P.hetrf!(A, ipiv; uplo = $ul); copyto!(B, B0); P.hetrs!(A, ipiv, B; uplo = $ul); P.hetri!(A, ipiv; uplo = $ul)))
        end
    end
    @entry "sysv!" ALLT 48 begin
        A0 = sym(T, n); A = similar(A0); ipiv = zeros(Int, n); B0 = rnd(T, n, 4); B = similar(B0)
        measure(() -> (copyto!(A, A0); copyto!(B, B0); P.sysv!('L', A, B, ipiv)))
    end
    @entry "hesv!" COMPLEXES 48 begin
        A0 = herm(T, n); A = similar(A0); ipiv = zeros(Int, n); B0 = rnd(T, n, 4); B = similar(B0)
        measure(() -> (copyto!(A, A0); copyto!(B, B0); P.hesv!('L', A, B, ipiv)))
    end
    @entry "gtsv!" ALLT 64 begin
        dl0 = rnd(T, n - 1) ./ 4; d0 = fill(T(3), n); du0 = rnd(T, n - 1) ./ 4; B0 = rnd(T, n, 3)
        dl = similar(dl0); d = similar(d0); du = similar(du0); B = similar(B0)
        measure(() -> (copyto!(dl, dl0); copyto!(d, d0); copyto!(du, du0); copyto!(B, B0); P.gtsv!(dl, d, du, B)))
    end
    @entry "gttrf!/gttrs!" ALLT 64 begin
        dl0 = rnd(T, n - 1) ./ 4; d0 = fill(T(3), n); du0 = rnd(T, n - 1) ./ 4; B0 = rnd(T, n, 3)
        dl = similar(dl0); d = similar(d0); du = similar(du0); du2 = zeros(T, n - 2); ipiv = zeros(Int, n); B = similar(B0)
        measure(() -> (copyto!(dl, dl0); copyto!(d, d0); copyto!(du, du0); P.gttrf!(dl, d, du, du2, ipiv); copyto!(B, B0); P.gttrs!('N', dl, d, du, du2, ipiv, B)))
    end
    @entry "pttrf!/pttrs!/ptsv!" ALLT 64 begin
        d0 = fill(real(T)(3), n); e0 = rnd(T, n - 1) ./ 4; B0 = rnd(T, n, 3)
        d = similar(d0); e = similar(e0); B = similar(B0)
        measure(() -> (copyto!(d, d0); copyto!(e, e0); P.pttrf!(d, e); copyto!(B, B0); P.pttrs!(d, e, B); copyto!(d, d0); copyto!(e, e0); copyto!(B, B0); P.ptsv!(d, e, B)))
    end
    for ul in ('U', 'L')
        @entry "pbtrf!/pbtrs! $ul" ALLT 64 begin
            AB0 = bandspd(T, n, 3, $ul); AB = similar(AB0); B0 = rnd(T, n, 3); B = similar(B0)
            measure(() -> (copyto!(AB, AB0); P.pbtrf!(AB; uplo = $ul, kd = 3); copyto!(B, B0); P.pbtrs!(AB, B; uplo = $ul, kd = 3)))
        end
        @entry "pptrf!/pptrs! $ul" ALLT 64 begin
            AP0 = packed(T, n, $ul, spd(T, n)); AP = similar(AP0); B0 = rnd(T, n, 3); B = similar(B0)
            measure(() -> (copyto!(AP, AP0); P.pptrf!(AP; uplo = $ul); copyto!(B, B0); P.pptrs!(AP, B; uplo = $ul)))
        end
    end
    @entry "gesvd! (U,S,Vt) n=48" (Float64, ComplexF32, ComplexF64) 48 begin
        A0 = rnd(T, n + 8, n); A = similar(A0); U = zeros(T, n + 8, n); S = zeros(real(T), n); Vt = zeros(T, n, n)
        measure(() -> (copyto!(A, A0); P.gesvd!(A, U, S, Vt)))
    end
    @entry "gesvd_vals!" (Float32, Float64, ComplexF32, ComplexF64) 48 begin
        A0 = rnd(T, n + 8, n); A = similar(A0); S = zeros(real(T), n)
        measure(() -> (copyto!(A, A0); P.gesvd_vals!(A, S)))
    end
    @entry "geqp3!" ALLT 48 begin
        A0 = rnd(T, n + 8, n); A = similar(A0); jpvt = zeros(Int, n); tau = zeros(T, n)
        measure(() -> (copyto!(A, A0); fill!(jpvt, 0); P.geqp3!(A, jpvt, tau)))
    end
    @entry "gels!" ALLT 48 begin
        A0 = rnd(T, n + 8, n); A = similar(A0); B0 = rnd(T, n + 8, 2); B = similar(B0)
        measure(() -> (copyto!(A, A0); copyto!(B, B0); P.gels!('N', A, B)))
    end
    @entry "gecon!/pocon!/trcon!" ALLT 48 begin
        A0 = rnd(T, n, n); F = copy(A0); ipiv = zeros(Int, n); P.getrf!(F, ipiv)
        S = spd(T, n); P.potrf!(S; uplo = 'L'); Tr = tri(T, n, 'U')
        measure(() -> (P.gecon!(real(T)(n), F, ipiv); P.pocon!(real(T)(n), S; uplo = 'L'); P.trcon!(Tr; uplo = 'U')))
    end


    # ── eigen / SVD building blocks with caller-supplied outputs ─────────────────────────────
    @entry "gebal!" ALLT 32 begin
        A0 = rnd(T, n, n); A = similar(A0); sc = zeros(real(T), n)
        measure(() -> (copyto!(A, A0); P.gebal!(A, sc)))
    end
    @entry "gehrd!" ALLT 32 begin
        A0 = rnd(T, n, n); A = similar(A0); tau = zeros(T, n - 1)
        measure(() -> (copyto!(A, A0); P.gehrd!(A, 1, n, tau)))
    end
    @entry "orghr!" ALLT 32 begin
        F = rnd(T, n, n); tau = zeros(T, n - 1); P.gehrd!(F, 1, n, tau); A = similar(F)
        measure(() -> (copyto!(A, F); P.orghr!(A, 1, n, tau)))
    end
    @entry "ormhr!" ALLT 32 begin
        F = rnd(T, n, n); tau = zeros(T, n - 1); P.gehrd!(F, 1, n, tau); C0 = rnd(T, n, 4); C = similar(C0)
        measure(() -> (copyto!(C, C0); P.ormhr!('L', 'N', 1, n, F, tau, C)))
    end
    @entry "hseqr!" ALLT 32 begin
        H0 = Matrix{T}(triu(rnd(T, n, n), -1)); H = similar(H0); Z = zeros(T, n, n)
        w = zeros(isc(T) ? T : Complex{T}, n)
        measure(() -> (copyto!(H, H0); fill!(Z, zero(T)); for i in 1:n; Z[i, i] = one(T); end; P.hseqr!('S', 'V', H, 1, n, w, Z)))
    end
    @entry "gebrd!" ALLT 48 begin
        A0 = rnd(T, n + 8, n); A = similar(A0); d = zeros(real(T), n); e = zeros(real(T), n); tq = zeros(T, n); tp = zeros(T, n)
        measure(() -> (copyto!(A, A0); P.gebrd!(A, d, e, tq, tp)))
    end
    @entry "bdsqr!" FLOATS 48 begin
        d0 = rand(T, n) .+ 1; e0 = rand(T, n - 1) ./ 2; d = similar(d0); e = similar(e0)
        measure(() -> (copyto!(d, d0); copyto!(e, e0); P.bdsqr!(d, e, nothing, nothing)))
    end
    @entry "trsyl!" ALLT 24 begin
        A = Matrix{T}(triu(rnd(T, n, n) ./ 4)) + 4I; B = Matrix{T}(triu(rnd(T, n, n) ./ 4)) - 4I
        C0 = rnd(T, n, n); C = similar(C0)
        measure(() -> (copyto!(C, C0); P.trsyl!('N', 'N', 1, A, B, C)))
    end
    @entry "geev! (real)" FLOATS 32 begin
        A0 = rnd(T, n, n); A = similar(A0); wr = zeros(T, n); wi = zeros(T, n); VL = zeros(T, n, n); VR = zeros(T, n, n); sc = zeros(T, n)
        measure(() -> (copyto!(A, A0); P.geev!('N', 'V', A, wr, wi, VL, VR, sc)))
    end
    @entry "geev! (complex)" COMPLEXES 32 begin
        A0 = rnd(T, n, n); A = similar(A0); w = zeros(T, n); VL = zeros(T, n, n); VR = zeros(T, n, n); sc = zeros(real(T), n)
        measure(() -> (copyto!(A, A0); P.geev!('N', 'V', A, w, VL, VR, sc)))
    end
    @entry "gees!" ALLT 32 begin
        A0 = rnd(T, n, n); A = similar(A0); w = zeros(Complex{real(T)}, n); VS = zeros(T, n, n); sc = zeros(real(T), n)
        measure(() -> (copyto!(A, A0); P.gees!('V', A, w, VS, sc)))
    end
    @entry "sygvd!" FLOATS 32 begin
        A0 = sym(T, n); B0 = spd(T, n); A = similar(A0); B = similar(B0); w = zeros(T, n)
        measure(() -> (copyto!(A, A0); copyto!(B, B0); P.sygvd!(1, 'V', 'L', A, B, w)))
    end
    @entry "hegvd!" COMPLEXES 32 begin
        A0 = herm(T, n); B0 = spd(T, n); A = similar(A0); B = similar(B0); w = zeros(real(T), n)
        measure(() -> (copyto!(A, A0); copyto!(B, B0); P.hegvd!(1, 'V', 'L', A, B, w)))
    end
    @entry "_syev! (internal driver)" FLOATS 32 begin
        A0 = sym(T, n); A = similar(A0)
        measure(() -> (copyto!(A, A0); P._syev!('V', 'L', A)))
    end
    @entry "_heev! (internal driver)" COMPLEXES 32 begin
        A0 = herm(T, n); A = similar(A0)
        measure(() -> (copyto!(A, A0); P._heev!('V', 'L', A)))
    end


    # ── remaining caller-buffer LAPACK entries ───────────────────────────────────────────────
    @entry "gbtrf!/gbtrs!" ALLT 48 begin
        AB0 = rnd(T, 7, n); AB = similar(AB0); ipiv = zeros(Int, n); B0 = rnd(T, n, 3); B = similar(B0)
        measure(() -> (copyto!(AB, AB0); P.gbtrf!(2, 2, n, AB, ipiv); copyto!(B, B0); P.gbtrs!('N', 2, 2, n, AB, ipiv, B)))
    end
    @entry "pstrf!" ALLT 48 begin
        A0 = spd(T, n); A = similar(A0); piv = zeros(Int, n)
        measure(() -> (copyto!(A, A0); P.pstrf!(A, piv, real(T)(-1); uplo = 'L')))
    end
    @entry "geqlf!/ormql!" ALLT 48 begin
        A0 = rnd(T, n + 8, n); A = similar(A0); tau = zeros(T, n); C0 = rnd(T, n + 8, 4); C = similar(C0)
        measure(() -> (copyto!(A, A0); P.geqlf!(A, tau); copyto!(C, C0); P.ormql!('L', 'N', A, tau, C)))
    end
    @entry "gerqf!/ormrq!" ALLT 48 begin
        A0 = rnd(T, n, n + 8); A = similar(A0); tau = zeros(T, n); C0 = rnd(T, n + 8, 4); C = similar(C0)
        measure(() -> (copyto!(A, A0); P.gerqf!(A, tau); copyto!(C, C0); P.ormrq!('L', 'N', A, tau, C)))
    end
    @entry "tzrzf!/ormrz!" ALLT 48 begin
        A0 = Matrix{T}(triu(rnd(T, n, n + 8))); A = similar(A0); tau = zeros(T, n); C0 = rnd(T, n + 8, 4); C = similar(C0)
        measure(() -> (copyto!(A, A0); P.tzrzf!(A, tau); copyto!(C, C0); P.ormrz!('L', 'N', A, tau, C)))
    end
    @entry "gelsd!" ALLT 48 begin
        A0 = rnd(T, n + 8, n); A = similar(A0); B0 = rnd(T, n + 8, 2); B = similar(B0); s = zeros(real(T), n)
        measure(() -> (copyto!(A, A0); copyto!(B, B0); P.gelsd!(A, B, real(T)(-1), s)))
    end
    @entry "gelsy!" ALLT 48 begin
        A0 = rnd(T, n + 8, n); A = similar(A0); B0 = rnd(T, n + 8, 2); B = similar(B0); jp = zeros(Int, n)
        measure(() -> (copyto!(A, A0); copyto!(B, B0); fill!(jp, 0); P.gelsy!(A, B, jp, real(T)(1.0e-8))))
    end
    @entry "gglse!" ALLT 48 begin
        p = 12
        A0 = rnd(T, n + 4, n); c0 = rnd(T, n + 4); B0 = rnd(T, p, n); d0 = rnd(T, p)
        A = similar(A0); c = similar(c0); B = similar(B0); d = similar(d0); x = zeros(T, n)
        measure(() -> (copyto!(A, A0); copyto!(c, c0); copyto!(B, B0); copyto!(d, d0); P.gglse!(A, c, B, d, x)))
    end
    @entry "qr_unblocked!" ALLT 32 begin
        A0 = rnd(T, n + 8, n); A = similar(A0); tau = zeros(T, n)
        measure(() -> (copyto!(A, A0); P.qr_unblocked!(A, tau)))
    end
    @entry "gebak!" ALLT 32 begin
        V0 = rnd(T, n, 4); V = similar(V0); sc = ones(real(T), n)
        measure(() -> (copyto!(V, V0); P.gebak!('B', 'R', 1, n, sc, V)))
    end

    # ── Forms without an output buffer ───────────────────────────────────────────────────────
    # These return freshly allocated arrays by construction. They are not zero-byte entries; each is
    # `@test_broken` in the LAPACK sweep so that a form that becomes allocation-free is noticed.
    @entry "getrf!(A) (allocating form)" ALLT 48 (A0 = rnd(T, n, n); A = similar(A0); measure(() -> (copyto!(A, A0); P.getrf!(A))))
    @entry "geqrf!(A) (allocating form)" ALLT 48 (A0 = rnd(T, n, n); A = similar(A0); measure(() -> (copyto!(A, A0); P.geqrf!(A))))
    @entry "gelqf!(A) (allocating form)" ALLT 48 (A0 = rnd(T, n, n); A = similar(A0); measure(() -> (copyto!(A, A0); P.gelqf!(A))))
    @entry "geqlf!(A) (allocating form)" ALLT 48 (A0 = rnd(T, n, n); A = similar(A0); measure(() -> (copyto!(A, A0); P.geqlf!(A))))
    @entry "gerqf!(A) (allocating form)" ALLT 48 (A0 = rnd(T, n, n); A = similar(A0); measure(() -> (copyto!(A, A0); P.gerqf!(A))))
    @entry "gehrd!(A) (allocating form)" ALLT 32 (A0 = rnd(T, n, n); A = similar(A0); measure(() -> (copyto!(A, A0); P.gehrd!(A))))
    @entry "gebal!(A) (allocating form)" ALLT 32 (A0 = rnd(T, n, n); A = similar(A0); measure(() -> (copyto!(A, A0); P.gebal!(A))))
    @entry "geqp3!(A) (allocating form)" ALLT 48 (A0 = rnd(T, n, n); A = similar(A0); measure(() -> (copyto!(A, A0); P.geqp3!(A))))
    @entry "gesvd!(A) (allocating form)" ALLT 48 (A0 = rnd(T, n, n); A = similar(A0); measure(() -> (copyto!(A, A0); P.gesvd!(A))))
    @entry "gees!(job,A) (allocating form)" ALLT 32 (A0 = rnd(T, n, n); A = similar(A0); measure(() -> (copyto!(A, A0); P.gees!('V', A))))
    @entry "geev!(jobvl,jobvr,A) (allocating form)" ALLT 32 (A0 = rnd(T, n, n); A = similar(A0); measure(() -> (copyto!(A, A0); P.geev!('N', 'V', A))))
    @entry "sysv!(uplo,A,B) (allocating form)" ALLT 48 begin
        A0 = sym(T, n); A = similar(A0); B0 = rnd(T, n, 4); B = similar(B0)
        measure(() -> (copyto!(A, A0); copyto!(B, B0); P.sysv!('L', A, B)))
    end

    # ── ForwardDiff.Dual ─────────────────────────────────────────────────────────────────────
    const DUALT = (D64,)
    @entry "axpy! (Dual)" DUALT 500 (x = rnd(T, n); y = rnd(T, n); a = T(1.5); measure(() -> P.axpy!(y, a, x)))
    @entry "scal! (Dual)" DUALT 500 (x = rnd(T, n); a = T(0.999); measure(() -> P.scal!(a, x)))
    @entry "dot (Dual)" DUALT 500 (x = rnd(T, n); y = rnd(T, n); measure(() -> P.dot(x, y)))
    @entry "gemv! (Dual)" DUALT 48 begin
        A = rnd(T, n, n); x = rnd(T, n); y = rnd(T, n)
        measure(() -> P.gemv!(y, A, x; alpha = one(T), beta = zero(T), trans = 'N'))
    end
    for nd in (24, 96)
        @entry "gemm! (Dual) n=$nd" DUALT $nd begin
            A = rnd(T, n, n); B = rnd(T, n, n); C = rnd(T, n, n)
            measure(() -> P.gemm!(C, A, B; alpha = one(T), beta = zero(T)))
        end
        @entry "trsm! (Dual) n=$nd" DUALT $nd begin
            A = tri(T, n, 'U'); B0 = rnd(T, n, n); B = copy(B0)
            measure(() -> (copyto!(B, B0); P.trsm!(A, B; side = 'L', uplo = 'U', transA = 'N', diag = 'N', alpha = one(T))))
        end
        @entry "syrk! (Dual) n=$nd" DUALT $nd begin
            A = rnd(T, n, n); C = sym(T, n)
            measure(() -> P.syrk!(C, A; uplo = 'U', trans = 'N', alpha = one(T), beta = zero(T)))
        end
        @entry "potrf! (Dual) n=$nd" DUALT $nd begin
            A0 = spd(T, n); A = similar(A0)
            measure(() -> (copyto!(A, A0); P.potrf!(A; uplo = 'L')))
        end
        @entry "getrf! (Dual) n=$nd" DUALT $nd begin
            A0 = rnd(T, n, n); A = similar(A0); ipiv = zeros(Int, n)
            measure(() -> (copyto!(A, A0); P.getrf!(A, ipiv)))
        end
        @entry "geqrf! (Dual) n=$nd" DUALT $nd begin
            A0 = rnd(T, n + 8, n); A = similar(A0); tau = zeros(T, n)
            measure(() -> (copyto!(A, A0); P.geqrf!(A, tau)))
        end
    end

    # ── drivers and toys shared by the static, ownership and steady-state items ──────────────
    run_gemm!(C, A, B) = P.gemm!(C, A, B; alpha = 1.0, beta = 0.0)
    run_trsm!(A, B, B0) = (copyto!(B, B0); P.trsm!(A, B; side = 'L', uplo = 'U', transA = 'N', diag = 'N', alpha = 1.0))
    run_syrk!(C, A) = P.syrk!(C, A; uplo = 'U', trans = 'N', alpha = 1.0, beta = 0.0)
    run_potrf!(A, A0) = (copyto!(A, A0); P.potrf!(A; uplo = 'L'))
    run_getrf!(A, A0, ip) = (copyto!(A, A0); P.getrf!(A, ip))
    # Bytes `f` allocates on newly scheduled tasks, after `f` is compiled: `(first, later)`. `first` is the
    # largest first-call count over `tasks` tasks (the call that pays for any per-task workspace);
    # `later` is the largest count over every call after a task's first.
    function fresh_task_bytes(f; tasks = 3, calls = 5)
        f()
        per = [fetch(Threads.@spawn [@allocated(f()) for _ in 1:calls]) for _ in 1:tasks]
        return (maximum(first, per), maximum(b -> maximum(b[2:end]), per))
    end
    # Runs for milliseconds and allocates nothing: a control that `@allocated` sees only other threads' allocations.
    const SPIN_SINK = Ref(0)
    spin!() = (x = 0; for i in 1:30_000_000; x += i & 3; end; SPIN_SINK[] = x; nothing)
    toy_alloc(x) = zeros(length(x))
    toy_ok(x) = sum(x)
    toy_c_alloc(x::Vector{Float64}) = zeros(length(x))
    toy_c_ok(x::Vector{Float64}) = sum(x)
    toy_c_kw(x::Vector{Float64}; a = 1.0) = (x .*= a; x)
    const POOL = Vector{Float64}[]
    toy_grows!(x) = (push!(POOL, zeros(4)); x)                 # allocates on every call
    toy_grows_big!(x) = (push!(POOL, zeros(100_000)); length(POOL) > 8 && empty!(POOL); x)   # 800 KB per call
    const PERTASK = Base.OncePerTask{Vector{Float64}}(() -> zeros(1000))
    toy_pertask!(x) = (PERTASK(); x)                           # allocates once per task
    const SHARED = zeros(1000)
    toy_shared!(x) = (SHARED[1] = x[1]; x)                     # a pool: allocates for no task
    const OWN_DICT = IdDict{DataType, Vector{Float64}}(Float64 => zeros(3))
    toy_lookup(::Type{T}) where {T} = OWN_DICT[T]
    toy_plain(x) = sum(x)
    const NE_BUF = zeros(16)
    ne_leak() = P.PtrMatrix{Float64}(pointer(NE_BUF), 4, 4, 4)
    ne_confined(n) = GC.@preserve NE_BUF (M = P.PtrMatrix{Float64}(pointer(NE_BUF), 4, 4, 4); M[2, 2] + n)
    const NE_SINK = Ref{Any}(nothing)
    ne_store(n) = (NE_SINK[] = P.PtrMatrix{Float64}(pointer(NE_BUF), 4, 4, 4); n)

    # Controls live in the module so the measured closures are concrete (a closure over a testitem
    # global reads its captured value through `Any` and boxes the result).
    ctl_zero() = (x = randn(100); measure(() -> sum(x)))
    ctl_alloc() = (x = randn(100); measure(() -> zeros(100)))
    function sweep(lapack::Bool)
        out = Tuple{String, DataType, Int, Int}[]
        for (name, Ts, n, f) in (lapack ? ENTRIES[(LAP0 + 1):end] : ENTRIES[1:LAP0]), T in Ts
            isempty(ONLY) || any(p -> occursin(p, name), split(ONLY, ',')) || continue
            b = try
                f(T, n)
            catch err
                @info "entry threw: $name $T n=$n $(sprint(showerror, err))"
                -1
            end
            push!(out, (name, T, n, b))
        end
        return out
    end
    ctl_grow() = (pool = Vector{Float64}[]; measure(() -> push!(pool, zeros(4))))
end

@testitem "alloc: measurement harness can fail" setup = [AllocT] begin
    using Test
    # Positive controls: the instrument reads zero for a call that allocates nothing and a positive
    # figure for each way a call can allocate, including a pool that grows on every call (a warm-up
    # cannot hide a per-call allocation).
    @test AllocT.ctl_zero() == 0
    @test AllocT.ctl_alloc() > 0
    @test AllocT.ctl_grow() > 0
end

@testitem "alloc: every BLAS-1/2/3 entry is allocation-free at steady state" setup = [AllocT] tags = [:checks] begin
    using Test
    res = AllocT.sweep(false)
    for (name, T, n, b) in res
        b == 0 || @info "allocates: $name $T n=$n bytes=$b"
        if haskey(AllocT.KNOWN, (name, T)) || endswith(name, "(allocating form)")
            @test_broken b == 0
        else
            @test b == 0
        end
    end
    @test length(res) > 100
end

@testitem "alloc: every LAPACK entry is allocation-free at steady state" setup = [AllocT] tags = [:checks] begin
    using Test
    res = AllocT.sweep(true)
    for (name, T, n, b) in res
        b == 0 || @info "allocates: $name $T n=$n bytes=$b"
        if haskey(AllocT.KNOWN, (name, T)) || endswith(name, "(allocating form)")
            @test_broken b == 0
        else
            @test b == 0
        end
    end
    @test length(res) > 100
end

# ── Static all-paths proof (AllocCheck through StrictModeTest) ───────────────────────────────────
# `test_signatures(...; guarantees = (:noalloc,))` is a proof over every path, including a workspace's
# first-call growth. Entries that statically reach a pool, a per-task workspace accessor or the worker
# scheduler therefore fail it even though they measure 0 B warm (see the sweeps above). The verdicts
# below are pinned; a verdict that flips in either direction fails and the row has to be updated. Each
# failing entry is also attributed: every allocation site's innermost PureBLAS frame must be a workspace
# or pool grower, so an allocation in a kernel is not hidden behind the known growers.
@testitem "alloc: static all-paths noalloc verdicts and grower attribution" setup = [AllocT] tags = [:checks] begin
    using Test, StrictMode, StrictModeTest
    P = PureBLAS
    verdict(f, types) = try
        test_signatures([(f, types)]; guarantees = (:noalloc,))
        :pass
    catch e
        e isa StrictMode.StrictViolation ? :fail : rethrow()
    end
    # Controls: the same driver flags an allocating signature and accepts a clean one.
    @test verdict(AllocT.toy_alloc, (Vector{Float64},)) === :fail
    @test verdict(AllocT.toy_ok, (Vector{Float64},)) === :pass

    # (label, function, argument types as a function of T, element types, expected verdict)
    rows = Any[
        ("axpy!", P.axpy!, T -> (Vector{T}, T, Vector{T}), (Float32, Float64, ComplexF32, ComplexF64), :fail),
        ("scal!", P.scal!, T -> (T, Vector{T}), (Float32, Float64, ComplexF32, ComplexF64), :fail),
        ("nrm2", P.nrm2, T -> (Vector{T},), (Float32, Float64, ComplexF32, ComplexF64), :fail),
        ("asum", P.asum, T -> (Vector{T},), (Float64, ComplexF64), :fail),
        ("dot real", P.dot, T -> (Vector{T}, Vector{T}), (Float32, Float64), :fail),
        ("dot complex", P.dot, T -> (Vector{T}, Vector{T}), (ComplexF32, ComplexF64), :pass),
        ("iamax real", P.iamax, T -> (Vector{T},), (Float32, Float64), :fail),
        ("iamax complex", P.iamax, T -> (Vector{T},), (ComplexF32, ComplexF64), :pass),
        ("blascopy! real", P.blascopy!, T -> (Vector{T}, Vector{T}), (Float32, Float64), :fail),
        ("blascopy! complex", P.blascopy!, T -> (Vector{T}, Vector{T}), (ComplexF32, ComplexF64), :pass),
        ("swap! real", P.swap!, T -> (Vector{T}, Vector{T}), (Float32, Float64), :fail),
        ("swap! complex", P.swap!, T -> (Vector{T}, Vector{T}), (ComplexF32, ComplexF64), :pass),
        ("getrf!", P.getrf!, T -> (Matrix{T}, Vector{Int}), (Float64, ComplexF64), :fail),
        ("geqrf!", P.geqrf!, T -> (Matrix{T}, Vector{T}), (Float64, ComplexF64), :fail),
        ("getrs!", P.getrs!, T -> (Matrix{T}, Vector{Int}, Matrix{T}), (Float64, ComplexF64), :fail),
        ("gelqf!", P.gelqf!, T -> (Matrix{T}, Vector{T}), (Float64, ComplexF64), :pass),
        ("gtsv!", P.gtsv!, T -> (Vector{T}, Vector{T}, Vector{T}, Matrix{T}), (Float64, ComplexF64), :pass),
        ("gttrf!", P.gttrf!, T -> (Vector{T}, Vector{T}, Vector{T}, Vector{T}, Vector{Int}), (Float64, ComplexF64), :pass),
    ]
    for (label, f, tys, Ts, expect) in rows, T in Ts
        @test verdict(f, tys(T)) === expect
    end

end

# Every allocation site of a failing proof is attributed: its innermost PureBLAS frame must be a workspace,
# arena, pool or scheduler grower (the files that own those, or a named grower function), or the site
# must lie wholly inside Base. A site in a kernel is reported as stray.
@testitem "alloc: static noalloc failures are attributed to growers" setup = [AllocT] tags = [:checks] begin
    using Test, StrictModeTest
    P = PureBLAS
    # Attribution. The innermost PureBLAS frame of every allocation site is a workspace, arena, pool or
    # scheduler grower, in the files that own those, or a named grower function.
    AC = StrictModeTest.AllocCheck
    GROWER_FILES = ("workspace.jl", "arena.jl", "lu.jl", "qr.jl")
    GROWER_FUNCS = (
        "_pool_barrier", "_gemm_threaded!", "_trmm_bpf", "_trmv_accbuf", "_symm_scr_mt", "_l3ws",
        "_gpackws", "_strws", "_symm_scr", "_ws_grow!", "_ws_slot!", "_arena_grow!", "_arena_coalesce!", "_qr_ws",
        "Arena", "L3Workspace",
    )
    pkgsrc = joinpath(pkgdir(PureBLAS), "src") * "/"      # a checkout need not be named "PureBLAS"
    function stray_sites(f, types)
        r = AC.check_allocs(f, types; ignore_throw = true)
        stray = String[]
        for a in r
            fr = findfirst(fr -> startswith(string(fr.file), pkgsrc), a.backtrace)
            if isnothing(fr)   # an allocation reached only through Base (lock, task and scheduler internals)
                baseonly = all(x -> startswith(string(x.file), "./") || x.file === :unknown || occursin("/share/julia/", string(x.file)), a.backtrace)
                baseonly || push!(stray, "non-Base frame outside PureBLAS/src")
                continue
            end
            fr = a.backtrace[fr]
            ok = basename(string(fr.file)) in GROWER_FILES || string(fr.func) in GROWER_FUNCS ||
                (startswith(string(fr.func), "#") && basename(string(fr.file)) in ("level2.jl", "level3.jl", "gemm.jl"))
            ok || push!(stray, string(fr.func, " @ ", basename(string(fr.file)), ":", fr.line))
        end
        return stray, length(r)
    end
    # Positive control: a kernel that allocates outside any grower is reported as stray.
    st0, ns0 = stray_sites(AllocT.toy_alloc, (Vector{Float64},))
    @test ns0 > 0 && !isempty(st0)
    for (f, types) in (
            (P.scal!, (Float64, Vector{Float64})),
            (P.getrf!, (Matrix{Float64}, Vector{Int})),
            (P.geqrf!, (Matrix{ComplexF64}, Vector{ComplexF64})),
            (P.getrs!, (Matrix{Float64}, Vector{Int}, Matrix{Float64})),
        )
        st, ns = stray_sites(f, types)
        @test ns > 0
        @test isempty(st)
    end
end

# ── Ownership (`@assert_owned`) ──────────────────────────────────────────────────────────────────
# The check gates. It reads the optimized IR for a runtime AbstractDict lookup reached from the entry.
# `Base.OncePerTask` resolves through `task_local_storage()`, an IdDict, so every entry whose scratch is a
# per-task accessor fails it; the rows below pin which entries do. Those are `@test_broken`, not accepted.
@testitem "alloc: @assert_owned on L3, L2 and LAPACK entries" setup = [AllocT] tags = [:checks] begin
    using Test, StrictMode, StrictModeTest, LinearAlgebra, ForwardDiff
    P = PureBLAS
    D64 = AllocT.D64
    status(f) = try
        f()
        :pass
    catch e
        e isa StrictMode.StrictViolation ? :fail : rethrow()
    end
    # Controls: a keyed lookup flags; a plain reduction does not.
    @test status(() -> (@assert_owned AllocT.toy_lookup(Float64))) === :fail
    @test status(() -> (@assert_owned AllocT.toy_plain([1.0]))) === :pass

    function suite(::Type{T}, n = 48) where {T}
        A = AllocT.rnd(T, n, n); B = AllocT.rnd(T, n, n); C = AllocT.rnd(T, n, n)
        S = AllocT.spd(T, n); U = AllocT.tri(T, n, 'U'); x = AllocT.rnd(T, n); y = AllocT.rnd(T, n)
        tau = zeros(T, n); ipiv = zeros(Int, n)
        r = Dict{String, Symbol}()
        r["gemm!"] = status(() -> (@assert_owned P.gemm!(C, A, B; alpha = one(T), beta = zero(T))))
        r["trsm!"] = status(() -> (@assert_owned P.trsm!(U, B; side = 'L', uplo = 'U', transA = 'N', diag = 'N', alpha = one(T))))
        r["trmm!"] = status(() -> (@assert_owned P.trmm!(U, B; side = 'L', uplo = 'U', transA = 'N', diag = 'N', alpha = one(T))))
        r["syrk!"] = status(() -> (@assert_owned P.syrk!(C, A; uplo = 'U', trans = 'N', alpha = one(T), beta = zero(T))))
        r["syr2k!"] = status(() -> (@assert_owned P.syr2k!(C, A, B; uplo = 'U', trans = 'N', alpha = one(T), beta = zero(T))))
        r["symm!"] = status(() -> (@assert_owned P.symm!(C, S, B; side = 'L', uplo = 'U', alpha = one(T), beta = zero(T))))
        r["gemv!"] = status(() -> (@assert_owned P.gemv!(y, A, x; alpha = one(T), beta = zero(T), trans = 'N')))
        r["trmv!"] = status(() -> (@assert_owned P.trmv!(U, x; uplo = 'U')))
        r["potrf!"] = status(() -> (@assert_owned P.potrf!(S; uplo = 'L')))
        r["getrf!"] = status(() -> (@assert_owned P.getrf!(A, ipiv)))
        r["geqrf!"] = status(() -> (@assert_owned P.geqrf!(A, tau)))
        if T <: Complex
            r["hemm!"] = status(() -> (@assert_owned P.hemm!(C, AllocT.herm(T, n), B; side = 'L', uplo = 'U', alpha = one(T), beta = zero(T))))
            r["herk!"] = status(() -> (@assert_owned P.herk!(C, A; uplo = 'U', trans = 'N', alpha = 1.0, beta = 0.0)))
        end
        return r
    end
    # Entries that fail: the per-task scratch accessor is a task-local IdDict probe.
    known_fail = Dict(
        Float64 => ("getrf!", "geqrf!"), ComplexF64 => ("getrf!", "geqrf!"), ComplexF32 => ("getrf!", "geqrf!"),
        D64 => ("symm!",), Float32 => (),
    )
    for T in (Float64, Float32, ComplexF64, ComplexF32, D64)
        r = suite(T)
        for (k, v) in r
            if k in known_fail[T]
                @test_broken v === :pass
            else
                @test v === :pass
            end
        end
    end
end

# ── Escape (`@assert_noescape`, `register_scoped!`) ───────────────────────────────────────────────
# `@assert_noescape` reports and never throws, so the gating form is the analysis it calls,
# `StrictMode._escaping_scoped` (internal). Borrowed arena handles are `PtrMatrix`/`PtrVector`.
@testitem "alloc: noescape on arena handles" setup = [AllocT] tags = [:checks] begin
    using Test, StrictMode
    P = PureBLAS
    begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            StrictMode.register_scoped!(P.PtrMatrix{T}); StrictMode.register_scoped!(P.PtrVector{T})
        end
        esc(f, tys...) = StrictMode._escaping_scoped(Tuple{typeof(f), tys...})
        # Controls. A handle created and read in the frame is confined; returned or stored globally it is not.
        @test isempty(esc(AllocT.ne_confined, Int).escaped)
        @test !isempty(esc(AllocT.ne_leak).escaped)
        @test !isempty(esc(AllocT.ne_store, Int).escaped)
        # `@assert_noescape` itself does not gate, so it cannot fail a test.
        @test StrictMode._guarantee_gates(:noescape) === false
        # Real arena users. The handle is created in the entry's own body and handed to a kernel that does
        # not inline, which the analysis cannot see through, so it reads as escaping. The kw-less
        # `P.sytri!(A, ipiv)` is a forwarder whose body holds no handle, so it is analysed through its
        # body function. These are `@test_broken`: the check cannot certify them today.
        body(f, tys...) = esc(Base.bodyfunction(which(f, tys)), Char, typeof(f), tys...)
        for T in (Float64, ComplexF64)
            @test_broken isempty(body(P.sytri!, Matrix{T}, Vector{Int}).escaped)
            @test_broken isempty(body(P.hetri!, Matrix{T}, Vector{Int}).escaped)
        end
        @test_broken isempty(esc(P.geqp3!, Matrix{Float64}, Vector{Int}, Vector{Float64}).escaped)
        @test_broken isempty(esc(P.gglse!, Matrix{Float64}, Vector{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Float64}).escaped)
    end
end

# ── Steady state and GC pauses (released StrictModeTest macros) ──────────────────────────────────
# `@test_steady_state` warms then measures; `fresh_task = true` measures on a newly scheduled task, which
# separates a pool (zero for any task) from a per-task initializer (paid again by every new task).
@testitem "alloc: @test_steady_state and @test_no_gc_pause on L3 and LAPACK entries" setup = [AllocT] tags = [:checks] begin
    using Test, StrictMode, StrictModeTest, LinearAlgebra
    VIOL = StrictMode.StrictViolation
    P = PureBLAS
    # JET (a StrictModeTest dependency) loads Revise, whose REPL-backend poll allocates on another thread
    # for about a second after load, and `@allocated` counts every thread. Wait it out, then require a
    # call that allocates nothing to measure 0.
    sleep(1.5)
    @test AllocT.fresh_task_bytes(AllocT.spin!) == (0, 0)
    # Controls, on the same task and on a fresh task.
    x = zeros(8)
    @test_throws VIOL @test_steady_state AllocT.toy_grows!(x)
    @test_steady_state AllocT.toy_pertask!(x)
    @test_throws VIOL @test_steady_state AllocT.toy_pertask!(x) fresh_task = true
    @test_steady_state AllocT.toy_shared!(x) fresh_task = true
    for n in (64, 200, 512)
        A = randn(n, n); B = randn(n, n); B0 = copy(B); C = zeros(n, n); S0 = Matrix(A * A' + n * I); S = copy(S0)
        U = Matrix(triu(A) + n * I); ip = zeros(Int, n); LU = copy(A)
        @test_steady_state AllocT.run_gemm!(C, A, B)
        @test_steady_state AllocT.run_trsm!(U, B, B0)
        @test_steady_state AllocT.run_syrk!(C, A)
        @test_steady_state AllocT.run_potrf!(S, S0)
        @test_steady_state AllocT.run_getrf!(LU, A, ip)
        @test_no_gc_pause AllocT.run_gemm!(C, A, B)
        @test_no_gc_pause AllocT.run_potrf!(S, S0)
        @test_no_gc_pause AllocT.run_getrf!(LU, A, ip)
        # A newly scheduled task pays for any per-task workspace the route touches. Only gemm's route is a
        # single predicate: the blocked path packs into `_gpackws`, the unpacked path touches no
        # workspace. The other entries reach per-task state through recursion over sub-gemms, so what
        # holds on every CPU is that the calls after the first on the same task allocate nothing; the
        # first call's bytes are recorded.
        gemm_ws = !P._use_unpacked(n, n, n)
        if gemm_ws
            @test_throws VIOL @test_steady_state AllocT.run_gemm!(C, A, B) fresh_task = true
        else
            @test_steady_state AllocT.run_gemm!(C, A, B) fresh_task = true
        end
        for (name, f) in (
                ("gemm", () -> AllocT.run_gemm!(C, A, B)), ("trsm", () -> AllocT.run_trsm!(U, B, B0)),
                ("syrk", () -> AllocT.run_syrk!(C, A)), ("potrf", () -> AllocT.run_potrf!(S, S0)),
                ("getrf", () -> AllocT.run_getrf!(LU, A, ip)),
            )
            first_bytes, later_bytes = AllocT.fresh_task_bytes(f)
            @info "fresh-task bytes" cpu = Sys.CPU_NAME op = name n first_call = first_bytes later_calls = later_bytes
            @test later_bytes == 0
            name == "gemm" && @test (first_bytes > 0) == gemm_ws
        end
    end
    # Controls for the later-call property: a call that allocates every time fails it, a per-task
    # initializer does not.
    @test AllocT.fresh_task_bytes(() -> AllocT.toy_grows!(x))[2] > 0
    let (b1, b2) = AllocT.fresh_task_bytes(() -> AllocT.toy_pertask!(x))
        @test b1 > 0 && b2 == 0
    end
    # A collection pause is detected when the call allocates.
    @test_throws VIOL @test_no_gc_pause AllocT.toy_grows_big!(x) samples = 400
end

# ── Unreleased StrictMode (master, 0.5.11): measured-zero macro and entry report ─────────────────
# These use names that StrictMode/StrictModeTest 0.5.10 does not define, so each item is reported as
# skipped when the API is absent. `@test_allocated_zero` is expanded at run time for the same reason.
@testitem "alloc: @test_allocated_zero (unreleased StrictMode)" setup = [AllocT] tags = [:checks] begin
    using Test, StrictMode, StrictModeTest, LinearAlgebra
    P = PureBLAS
    if !isdefined(StrictModeTest, :alloc_entry_report)
        @test_skip isdefined(StrictModeTest, :alloc_entry_report)
    else
        VIOL = StrictMode.StrictViolation
        az(call) = (Core.eval(@__MODULE__, :(StrictModeTest.@test_allocated_zero $call)); true)
        az_fails(call) = try
            az(call)
            false
        catch e
            e isa VIOL
        end
        x = zeros(8)
        # Controls.
        @test az(:(sum($x)))
        @test az_fails(:(collect(1:$(length(x)))))
        @test az_fails(:($(AllocT.toy_grows!)($x)))
        for T in (Float32, Float64, ComplexF32, ComplexF64), n in (48, 200)
            A = AllocT.rnd(T, n, n); B = AllocT.rnd(T, n, n); C = AllocT.rnd(T, n, n); y = AllocT.rnd(T, n); v = AllocT.rnd(T, n)
            S0 = AllocT.spd(T, n); S = copy(S0); U = AllocT.tri(T, n, 'U'); B0 = copy(B); ip = zeros(Int, n); LU = copy(A)
            tau = zeros(T, n); one_T = one(T); zero_T = zero(T)
            @test az(:(PureBLAS.gemm!($C, $A, $B; alpha = $one_T, beta = $zero_T)))
            @test az(:(PureBLAS.syrk!($C, $A; uplo = 'U', trans = 'N', alpha = $one_T, beta = $zero_T)))
            @test az(:(PureBLAS.gemv!($y, $A, $v; alpha = $one_T, beta = $zero_T, trans = 'N')))
            @test az(:(PureBLAS.trsm!($U, $B; side = 'L', uplo = 'U', transA = 'N', diag = 'N', alpha = $one_T)))
            @test az(:($(AllocT.run_potrf!)($S, $S0)))
            @test az(:(PureBLAS.getrf!($LU, $ip)))
            @test az(:(PureBLAS.geqrf!($LU, $tau)))
        end
    end
end

# `alloc_entry_report` lists every entry point of a function and what the all-paths proof says about each.
# A PureBLAS entry is declared on abstract argument types, which the proof cannot take, so every
# positional entry reads `:unproven` and every keyword entry (`Core.kwcall`) is always `:unproven`. The
# test pins what the report enumerates and that none fails, and records "some entry proven" as broken.
@testitem "alloc: alloc_entry_report per family (unreleased StrictMode)" setup = [AllocT] tags = [:checks] begin
    using Test, StrictMode, StrictModeTest
    P = PureBLAS
    if !isdefined(StrictModeTest, :alloc_entry_report)
        @test_skip isdefined(StrictModeTest, :alloc_entry_report)
    else
        # Controls: concrete signatures are decided; a keyword method is listed and unproven.
        r = StrictModeTest.alloc_entry_report(AllocT.toy_c_alloc)
        @test [e.status for e in r.entries] == [:fail]
        r = StrictModeTest.alloc_entry_report(AllocT.toy_c_ok)
        @test [e.status for e in r.entries] == [:pass]
        r = StrictModeTest.alloc_entry_report(AllocT.toy_c_kw)
        @test sort([(e.kind, e.status) for e in r.entries]) == [(:keyword, :unproven), (:positional, :pass)]
        families = Dict(
            "BLAS-1" => [P.axpy!, P.scal!, P.blascopy!, P.swap!, P.dot, P.dotu, P.nrm2, P.asum, P.iamax],
            "BLAS-2" => [
                P.gemv!, P.ger!, P.symv!, P.hemv!, P.trmv!, P.trsv!, P.spmv!, P.hpmv!, P.gbmv!, P.sbmv!, P.hbmv!,
                P.tbmv!, P.tbsv!, P.tpmv!, P.tpsv!, P.spr!, P.spr2!, P.hpr!, P.hpr2!,
            ],
            "BLAS-3" => [P.gemm!, P.symm!, P.hemm!, P.syrk!, P.herk!, P.syr2k!, P.her2k!, P.trmm!, P.trsm!],
            "LAPACK" => [
                P.potrf!, P.potrs!, P.potri!, P.getrf!, P.getrs!, P.getri!, P.geqrf!, P.gelqf!, P.geqlf!, P.gerqf!,
                P.orgqr!, P.orglq!, P.ormqr!, P.ormlq!, P.trtri!, P.trtrs!, P.sytrf!, P.sytrs!, P.sytri!, P.hetrf!,
                P.hetrs!, P.hetri!, P.sysv!, P.hesv!, P.gtsv!, P.gttrf!, P.gttrs!, P.pttrf!, P.pttrs!, P.ptsv!,
                P.pbtrf!, P.pbtrs!, P.pptrf!, P.pptrs!, P.gesvd!, P.gesvd_vals!, P.geqp3!, P.gels!, P.gelsd!,
                P.gelsy!, P.gecon!, P.pocon!, P.trcon!, P.gebal!, P.gehrd!, P.orghr!, P.ormhr!, P.hseqr!, P.gebrd!,
                P.bdsqr!, P.trsyl!, P.geev!, P.gees!, P.sygvd!, P.hegvd!, P.gbtrf!, P.gbtrs!, P.pstrf!,
            ],
        )
        for (fam, fs) in families
            nentries = 0; nkw = 0
            for f in fs
                r = StrictModeTest.alloc_entry_report(f)
                nentries += length(r.entries)
                nkw += count(e -> e.kind === :keyword, r.entries)
                @test !isempty(r.entries)
                @test all(e -> e.status !== :fail, r.entries)
                @test_broken any(e -> e.status === :pass, r.entries)   # no PureBLAS entry is decidable by this report
            end
            @info "alloc_entry_report" fam nfunctions = length(fs) nentries nkeyword = nkw
            # BLAS-1 entries take no keywords, so nothing routes through `Core.kwcall`.
            fam == "BLAS-1" ? (@test nkw == 0) : (@test nkw > 0)
        end
    end
end

# `steady_state_report` and `gc_pause_report` are the function forms of the two macros above; they return
# a report instead of throwing, which is what a probe or a dashboard reads.
@testitem "alloc: steady_state_report and gc_pause_report" setup = [AllocT] tags = [:checks] begin
    using Test, StrictMode
    x = zeros(8)
    n = 200; A = randn(n, n); B = randn(n, n); C = zeros(n, n)
    r = StrictMode.steady_state_report(AllocT.run_gemm!, C, A, B; repeats = 3)
    @test StrictMode.steady(r)
    @test !StrictMode.steady(StrictMode.steady_state_report(AllocT.toy_grows!, x; repeats = 3))
    @test !StrictMode.steady(StrictMode.steady_state_report(AllocT.toy_pertask!, x; fresh_task = true, repeats = 3))
    # A fresh task pays for gemm's per-task pack buffers only on the blocked route.
    @test StrictMode.steady(StrictMode.steady_state_report(AllocT.run_gemm!, C, A, B; fresh_task = true, repeats = 3)) == PureBLAS._use_unpacked(n, n, n)
    @test StrictMode.gc_free(StrictMode.gc_pause_report(AllocT.run_gemm!, C, A, B; samples = 300))
    @test !StrictMode.gc_free(StrictMode.gc_pause_report(AllocT.toy_grows_big!, x; samples = 400))
end
