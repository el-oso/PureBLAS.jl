# LAPACK C-ABI entries must touch only the memory the LAPACK interface gives them, and read each
# argument where LAPACK says it is. Every buffer the routine may not write is filled with a sentinel
# and padded far enough that an out-of-region access stays inside the Julia allocation, where the
# test sees it as a changed sentinel or a wrong result instead of corrupting the process.
#
# Where the question is which elements LAPACK reads (`ormrz`, `larf`), the reference is the
# default LAPACK library itself, called through its ILP64 symbol on identical buffers.

@testmodule CabiRegion begin
    using LinearAlgebra
    const Libdl = Base.Libc.Libdl

    # The default LAPACK, independent of whether a test elsewhere has called `PureBLAS.activate()`.
    const REFLIB = Libdl.dlopen(first(LinearAlgebra.BLAS.lbt_get_config().loaded_libs).libname)
    refsym(name) = Libdl.dlsym(REFLIB, name * "_64_")

    # `ccall` of a Fortran symbol whose arguments are all pointers or `Clong` string lengths.
    @generated function fcall(p::Ptr{Cvoid}, args...)
        types = Expr(:tuple, (a <: Ptr ? :(Ptr{Cvoid}) : a for a in args)...)
        return :(ccall(p, Cvoid, $types, $((:(args[$i]) for i in eachindex(args))...)))
    end

    # Scalar arguments as 1-element vectors, kept alive by `keep` (callers `GC.@preserve keep`).
    function boxer()
        keep = Any[]
        box(x) = (a = [x]; push!(keep, a); pointer(a))
        return keep, box
    end

    pre(T) = T === Float32 ? "s" : T === Float64 ? "d" : T === ComplexF32 ? "c" : "z"
    sentinel(::Type{T}) where {T} = T(-7.25)
    tol(T) = 64 * eps(real(T))
end

@testitem "C-ABI ormrz/unmrz reads A's last L columns" setup = [CabiRegion] begin
    using .CabiRegion: refsym, fcall, boxer, pre, sentinel, tol
    import PureBLAS as P
    for T in (Float32, Float64, ComplexF32, ComplexF64), side in ('L', 'R'),
            (K, L, nq) in ((2, 2, 4), (2, 1, 5), (2, 3, 4))   # K+L == nq, < nq, > nq
        name = pre(T) * (T <: Complex ? "unmrz" : "ormrz")
        pb = getfield(P, Symbol(name, "_64_"))
        for tr in ('N', T <: Complex ? 'C' : 'T')
            M, N = side == 'L' ? (nq, 3) : (3, nq)
            lda = K + 1
            A = randn(T, lda, nq)
            padA = vcat(vec(A), fill(sentinel(T), lda * (K + L)))   # room for reads past column nq
            tau = randn(T, K); C0 = randn(T, M, N)
            outs = map((:ref, :pb)) do which
                Ab = copy(padA); Cb = copy(C0); w = zeros(T, 8192); tb = copy(tau)
                keep, box = boxer()
                info = Ref(Int64(-99))
                GC.@preserve keep Ab Cb w tb info begin
                    args = (
                        box(UInt8(side)), box(UInt8(tr)), box(Int64(M)), box(Int64(N)), box(Int64(K)),
                        box(Int64(L)), pointer(Ab), box(Int64(lda)), pointer(tb), pointer(Cb), box(Int64(M)),
                        pointer(w), box(Int64(length(w))),
                        Base.unsafe_convert(Ptr{Int64}, info), Clong(1), Clong(1),
                    )
                    which === :ref ? fcall(refsym(name), args...) : pb(args...)
                end
                (Cb, Ab, info[])
            end
            (Cr, Ar, ir), (Cp, Ap, ip) = outs
            @test ir == 0 && ip == 0
            @test isapprox(Cp, Cr; rtol = tol(T) * nq)
            @test Ap == padA
        end
    end
end

@testitem "C-ABI larf honors a negative incv" setup = [CabiRegion] begin
    using .CabiRegion: refsym, fcall, boxer, pre, sentinel, tol
    import PureBLAS as P
    for T in (Float32, Float64, ComplexF32, ComplexF64), side in ('L', 'R'), iv in (2, -2)
        name = pre(T) * "larf"
        pb = getfield(P, Symbol(name, "_64_"))
        M, N = 5, 4
        l = side == 'L' ? M : N
        a = abs(iv)
        v = randn(T, l); v[1] = one(T)                    # LAPACK callers store the implicit 1
        span = 1 + (l - 1) * a
        pad = span                                        # room for reads before or after v
        vbuf = fill(sentinel(T), pad + span + pad)
        for j in 1:l                                      # Fortran layout of a vector at stride iv
            vbuf[pad + (iv > 0 ? 1 + (j - 1) * a : 1 + (l - j) * a)] = v[j]
        end
        τ = randn(T); C0 = randn(T, M, N)
        outs = map((:ref, :pb)) do which
            Cb = copy(C0); w = zeros(T, max(M, N)); vb = copy(vbuf); tb = [τ]
            keep, box = boxer()
            GC.@preserve keep Cb w vb tb begin
                args = (
                    box(UInt8(side)), box(Int64(M)), box(Int64(N)), pointer(vb, pad + 1), box(Int64(iv)),
                    pointer(tb), pointer(Cb), box(Int64(M)), pointer(w), Clong(1),
                )
                which === :ref ? fcall(refsym(name), args...) : pb(args...)
            end
            Cb
        end
        @test isapprox(outs[2], outs[1]; rtol = tol(T) * l)
    end
end

@testitem "C-ABI sgebrd writes min(m,n)-1 elements of e" setup = [CabiRegion] begin
    using .CabiRegion: boxer, sentinel
    import PureBLAS as P
    for (M, N) in ((6, 4), (5, 5), (3, 1))
        k = min(M, N)
        A = randn(Float32, M, N)
        d = zeros(Float32, k); e = fill(sentinel(Float32), k + 4); tq = zeros(Float32, k); tp = zeros(Float32, k)
        w = zeros(Float32, 1); info = Int64[-99]
        keep, box = boxer()
        GC.@preserve keep A d e tq tp w info begin
            P.sgebrd_64_(
                box(Int64(M)), box(Int64(N)), pointer(A), box(Int64(M)), pointer(d), pointer(e),
                pointer(tq), pointer(tp), pointer(w), box(Int64(1)), pointer(info)
            )
        end
        @test info[1] == 0
        @test all(==(sentinel(Float32)), e[k:end])
        @test all(isfinite, d) && all(isfinite, e[1:(k - 1)])
    end
end

@testitem "C-ABI sgesvd job 'O' writes A, not U or VT" setup = [CabiRegion] begin
    using .CabiRegion: boxer, sentinel
    import PureBLAS as P
    M, N = 7, 5; mn = min(M, N)
    A0 = randn(Float32, M, N)
    function svd_cabi(ju, jv)
        A = copy(A0); S = zeros(Float32, mn)
        U = fill(sentinel(Float32), 2 * M * M); VT = fill(sentinel(Float32), 2 * N * N)
        ldu = ju in ('S', 'A') ? M : 1; ldvt = jv in ('S', 'A') ? mn : 1
        w = zeros(Float32, 1); info = Int64[-99]
        keep, box = boxer()
        GC.@preserve keep A S U VT w info begin
            P.sgesvd_64_(
                box(UInt8(ju)), box(UInt8(jv)), box(Int64(M)), box(Int64(N)), pointer(A), box(Int64(M)),
                pointer(S), pointer(U), box(Int64(ldu)), pointer(VT), box(Int64(ldvt)), pointer(w),
                box(Int64(1)), pointer(info), Clong(1), Clong(1)
            )
        end
        return A, S, U, VT, info[1]
    end
    _, Sr, Ur, VTr, _ = svd_cabi('S', 'S')
    Ur = reshape(Ur[1:(M * mn)], M, mn); VTr = reshape(VTr[1:(mn * N)], mn, N)
    for (ju, jv) in (('O', 'S'), ('O', 'N'), ('S', 'O'), ('N', 'O'))
        A, S, U, VT, info = svd_cabi(ju, jv)
        @test info == 0
        @test S ≈ Sr
        ju in ('O', 'N') && @test all(==(sentinel(Float32)), U)
        jv in ('O', 'N') && @test all(==(sentinel(Float32)), VT)
        ju == 'O' && @test A[:, 1:mn] ≈ Ur
        jv == 'O' && @test A[1:mn, :] ≈ VTr
    end
end

@testitem "C-ABI tgsen leaves Q and Z alone when not wanted" setup = [CabiRegion] begin
    using .CabiRegion: boxer, pre, sentinel
    using LinearAlgebra
    import PureBLAS as P
    N = 5
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        pb = getfield(P, Symbol(pre(T), "tgsen_64_"))
        S0 = triu(randn(T, N, N)); T0 = triu(randn(T, N, N)) + N * I
        sel = Int64[0, 1, 0, 1, 0]
        function run(want)
            S = copy(S0); Tm = copy(T0)
            Q = want ? Matrix{T}(I, N, N) : fill(sentinel(T), N * N)
            Z = want ? Matrix{T}(I, N, N) : fill(sentinel(T), N * N)
            ld = want ? N : 1
            al = zeros(T, N); ai = zeros(real(T), N); be = zeros(T, N)
            m = Int64[0]; w = zeros(T, 1); iw = Int64[0]; info = Int64[-99]; sl = copy(sel)
            keep, box = boxer()
            GC.@preserve keep S Tm Q Z al ai be m w iw info sl begin
                head = (box(Int64(0)), box(Int64(want)), box(Int64(want)), pointer(sl), box(Int64(N)),
                    pointer(S), box(Int64(N)), pointer(Tm), box(Int64(N)))
                tail = (pointer(Q), box(Int64(ld)), pointer(Z), box(Int64(ld)), pointer(m),
                    Ptr{Cvoid}(0), Ptr{Cvoid}(0), Ptr{Cvoid}(0), pointer(w), box(Int64(1)), pointer(iw),
                    box(Int64(1)), pointer(info))
                if T <: Complex
                    pb(head..., pointer(al), pointer(be), tail...)
                else
                    pb(head..., pointer(al), pointer(ai), pointer(be), tail...)
                end
            end
            return S, Tm, Q, Z, info[1]
        end
        Sw, Tw, _, _, info_w = run(true)
        Sn, Tn, Qn, Zn, info_n = run(false)
        @test info_w == 0 && info_n == 0
        @test all(==(sentinel(T)), Qn) && all(==(sentinel(T)), Zn)
        @test Sn ≈ Sw && Tn ≈ Tw
    end
end

@testitem "C-ABI sgesvx fact='F' uses the caller's factorization" setup = [CabiRegion] begin
    using .CabiRegion: boxer
    using LinearAlgebra
    import PureBLAS as P
    N, RH = 6, 2
    A0 = randn(Float32, N, N) + N * I; B0 = randn(Float32, N, RH)
    function gesvx(fact, AF, ipiv)
        A = copy(A0); B = copy(B0); X = zeros(Float32, N, RH)
        R = ones(Float32, N); C = ones(Float32, N); equed = [UInt8('N')]
        rc = Float32[0]; fe = zeros(Float32, RH); be = zeros(Float32, RH)
        w = zeros(Float32, 4N); iw = zeros(Int64, N); info = Int64[-99]
        keep, box = boxer()
        GC.@preserve keep A B X R C equed rc fe be w iw info AF ipiv begin
            P.sgesvx_64_(
                box(UInt8(fact)), box(UInt8('N')), box(Int64(N)), box(Int64(RH)), pointer(A), box(Int64(N)),
                pointer(AF), box(Int64(N)), pointer(ipiv), pointer(equed), pointer(R), pointer(C),
                pointer(B), box(Int64(N)), pointer(X), box(Int64(N)), pointer(rc), pointer(fe), pointer(be),
                pointer(w), pointer(iw), pointer(info), Clong(1), Clong(1), Clong(1)
            )
        end
        return X, info[1]
    end
    Xref = Float64.(A0) \ Float64.(B0)
    # fact='N' returns the factorization in AF/ipiv; fact='F' must then reuse it.
    AF = zeros(Float32, N, N); ipiv = zeros(Int64, N)
    Xn, in_ = gesvx('N', AF, ipiv)
    @test in_ == 0 && isapprox(Xn, Xref; rtol = 1e-4)
    @test all(i -> 1 <= ipiv[i] <= N, 1:N)
    @test any(!iszero, AF)
    Xf, if_ = gesvx('F', AF, ipiv)
    @test if_ == 0 && isapprox(Xf, Xref; rtol = 1e-4)
end

@testitem "C-ABI gels rejects lda/ldb below LAPACK's minimum instead of writing past B" setup = [CabiRegion] begin
    using .CabiRegion: refsym, fcall, boxer, pre, sentinel
    import PureBLAS as P
    # Underdetermined A (2×5): B is read and written as 5 rows, so ldb = 2 must be info = -8, as in dgels.
    for T in (Float32, Float64, ComplexF32, ComplexF64), (lda, ldb, want) in ((2, 2, -8), (1, 5, -6), (2, 5, 0))
        name = pre(T) * "gels"; pb = getfield(P, Symbol(name, "_64_")); M, N, R = 2, 5, 1
        A0 = randn(T, lda * N + 8); B0 = fill(sentinel(T), 16); B0[1:2] = randn(T, 2)
        function run(f)
            A = copy(A0); B = copy(B0); work = zeros(T, 256); keep, box = boxer(); info = [Int64(99)]
            GC.@preserve keep A B work info f(box(UInt8('N')), box(Int64(M)), box(Int64(N)), box(Int64(R)),
                pointer(A), box(Int64(lda)), pointer(B), box(Int64(ldb)), pointer(work), box(Int64(256)),
                pointer(info), Clong(1))
            return info[1], B
        end
        ipb, Bpb = run(pb)
        iref, _ = run((args...) -> fcall(refsym(name), args...))
        @test iref == want
        @test ipb == want
        want < 0 && @test Bpb == B0                      # nothing written, inside or past B
        want < 0 || @test all(==(sentinel(T)), Bpb[6:end])
    end
end
