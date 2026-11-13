# C-ABI conformance on the `activate()` path: every forwarded BLAS symbol, called through
# libblastrampoline exactly as LinearAlgebra calls it, must compute what the default BLAS computes.
#
# Each symbol is called with a direct `ccall` of LBT's own `<name>64_` trampoline, so the reference
# (before `activate()`) and PureBLAS (after) run the identical call with the identical buffers. A
# case covers one element type, one flag combination, one stride/leading-dimension layout and one
# size. Sizes are read from the module's routing constants at test time, so each case lands on a
# known side of a blocking or routing cut and keeps doing so if the constant moves.
#
# Output buffers are compared in full. A position the reference leaves at its initial value (a
# stride gap, rows past `m` in a leading dimension, the unreferenced triangle) must keep that value
# in PureBLAS's output too, and every other position must agree with the reference, both to a
# tolerance. Inputs fill the unreferenced triangle and the stride gaps with random values, so a
# kernel that reads the wrong half or the wrong element disagrees.

@testmodule CabiConf begin
    using PureBLAS, LinearAlgebra, Random, Test
    const Libdl = Base.Libc.Libdl

    const LBT = Libdl.dlopen(LinearAlgebra.libblastrampoline)
    fptr(name) = Libdl.dlsym(LBT, name * "64_")
    lbt_fwd(name) = LinearAlgebra.BLAS.lbt_get_forward(
        name, LinearAlgebra.BLAS.LBT_INTERFACE_ILP64, LinearAlgebra.BLAS.LBT_F2C_PLAIN
    )

    const TYPES = (Float32, Float64, ComplexF32, ComplexF64)
    pre(T) = T === Float32 ? "s" : T === Float64 ? "d" : T === ComplexF32 ? "c" : "z"

    # One call of one symbol. `run()` builds fresh outputs from `inits`, calls the symbol and returns
    # the outputs; `inits[i]` is the initial content of output `i`, or `nothing` for a returned value.
    struct Case
        label::String
        run::Function
        inits::Tuple
        rtol::Float64
    end

    rtol_for(T, dim) = 256 * sqrt(max(dim, 1)) * eps(real(T))

    # Vector of logical length `n` stored at stride `inc` (gaps hold random values too).
    vbuf(rng, T, n, inc) = randn(rng, T, max(1 + (n - 1) * abs(inc), 1))

    # Elementwise agreement with an RMS floor, so a small entry of a large result is judged against
    # the result's scale rather than its own magnitude. A position the reference left at its initial
    # value must hold that value in PureBLAS's output too, to the same tolerance: a stray write puts
    # an unrelated value there, while a computed entry can round back to its initial value.
    near(g, r, rtol, rms) = g === r || abs(g - r) <= rtol * (abs(r) + rms)
    function moved_rms(ref, init)
        moved = [!(r === i) for (r, i) in zip(ref, init)]
        rms = any(moved) ? sqrt(sum(abs2, ref[moved]) / count(moved)) : 0.0
        return moved, rms
    end
    function agree(got, ref, init, rtol)
        if isnothing(init)
            ref isa Integer && return got == ref
            return abs(got - ref) <= rtol * max(abs(ref), floatmin(Float32))
        end
        size(got) == size(ref) || return false
        moved, rms = moved_rms(ref, init)
        return all(near(g, m ? r : i, rtol, rms) for (g, r, i, m) in zip(got, ref, init, moved))
    end

    # What `agree` saw, for the failure report: the worst scaled error and the number of positions
    # written that the reference left alone.
    function describe(got, ref, init)
        isnothing(init) && return "got $got, reference $ref"
        size(got) == size(ref) || return "size $(size(got)) vs $(size(ref))"
        moved, rms = moved_rms(ref, init)
        stray = count(!near(g, i, 0.0, rms) for (g, i, m) in zip(got, init, moved) if !m)
        worst = maximum((abs(g - r) / (abs(r) + rms) for (g, r, m) in zip(got, ref, moved) if m); init = 0.0)
        return "worst scaled error $(Float32(worst)), $stray positions the reference left unchanged differ"
    end

    # Run every case through the default BLAS, then through PureBLAS after `activate()`, and return
    # the labels that disagree (a thrown exception counts as a disagreement).
    function conform(name::String, cases::Vector{Case})
        refs = map(c -> c.run(), cases)
        PureBLAS.activate()
        gots = try
            haskey(PureBLAS._OUR_PTRS, name) || error("$name is not forwarded by activate()")
            lbt_fwd(name) == PureBLAS._OUR_PTRS[name] || error("$name does not route to PureBLAS")
            map(cases) do c
                try
                    c.run()
                catch err
                    err
                end
            end
        finally
            PureBLAS.deactivate()
        end
        bad = String[]
        for (c, r, g) in zip(cases, refs, gots)
            ok = !(g isa Exception) && length(g) == length(r) &&
                all(agree(g[i], r[i], c.inits[i], c.rtol) for i in eachindex(r))
            ok && continue
            push!(
                bad, g isa Exception ? "$(c.label): $(sprint(showerror, g))" :
                    length(g) != length(r) ? "$(c.label): $(length(g)) outputs" :
                    "$(c.label): " * join((describe(g[i], r[i], c.inits[i]) for i in eachindex(r)), "; ")
            )
        end
        isempty(bad) || @error "C-ABI conformance: $name disagrees with the default BLAS in \
            $(length(bad)) of $(length(cases)) cases; the first ones:\n" * join(first(bad, 8), "\n")
        return bad
    end

    # ── Fortran-ABI calls, one per argument shape. Characters and scalars by reference, arrays by
    # pointer, one trailing hidden string length per character argument. `BI` is the BLAS integer.
    const BI = Int64
    c(ch::Char) = UInt8(ch)

    l1_axpy(fp, n, a::T, x::Vector{T}, ix, y::Vector{T}, iy) where {T} =
        ccall(fp, Cvoid, (Ref{BI}, Ref{T}, Ptr{T}, Ref{BI}, Ptr{T}, Ref{BI}), n, a, x, ix, y, iy)
    l1_scal(fp, n, a::T, x::Vector{T}, ix) where {T} =
        ccall(fp, Cvoid, (Ref{BI}, Ref{T}, Ptr{T}, Ref{BI}), n, a, x, ix)
    l1_xy(fp, n, x::Vector{T}, ix, y::Vector{T}, iy) where {T} =
        ccall(fp, Cvoid, (Ref{BI}, Ptr{T}, Ref{BI}, Ptr{T}, Ref{BI}), n, x, ix, y, iy)
    # A ccall's return type cannot be a static parameter, so the value-returning calls are
    # generated per element type.
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        @eval l1_dot(fp, n, x::Vector{$T}, ix, y::Vector{$T}, iy) =
            ccall(fp, $T, (Ref{BI}, Ptr{$T}, Ref{BI}, Ptr{$T}, Ref{BI}), n, x, ix, y, iy)
        @eval l1_real(fp, n, x::Vector{$T}, ix) =
            ccall(fp, $(real(T)), (Ref{BI}, Ptr{$T}, Ref{BI}), n, x, ix)
    end
    l1_iamax(fp, n, x::Vector{T}, ix) where {T} = ccall(fp, BI, (Ref{BI}, Ptr{T}, Ref{BI}), n, x, ix)

    l2_gemv(fp, t, m, n, a::T, A::Matrix{T}, lda, x, ix, b, y, iy) where {T} = ccall(
        fp, Cvoid, (Ref{UInt8}, Ref{BI}, Ref{BI}, Ref{T}, Ptr{T}, Ref{BI}, Ptr{T}, Ref{BI}, Ref{T}, Ptr{T}, Ref{BI}, Clong),
        c(t), m, n, a, A, lda, x, ix, b, y, iy, 1
    )
    l2_gbmv(fp, t, m, n, kl, ku, a::T, A::Matrix{T}, lda, x, ix, b, y, iy) where {T} = ccall(
        fp, Cvoid,
        (Ref{UInt8}, Ref{BI}, Ref{BI}, Ref{BI}, Ref{BI}, Ref{T}, Ptr{T}, Ref{BI}, Ptr{T}, Ref{BI}, Ref{T}, Ptr{T}, Ref{BI}, Clong),
        c(t), m, n, kl, ku, a, A, lda, x, ix, b, y, iy, 1
    )
    l2_ger(fp, m, n, a::T, x::Vector{T}, ix, y, iy, A, lda) where {T} = ccall(
        fp, Cvoid, (Ref{BI}, Ref{BI}, Ref{T}, Ptr{T}, Ref{BI}, Ptr{T}, Ref{BI}, Ptr{T}, Ref{BI}),
        m, n, a, x, ix, y, iy, A, lda
    )
    l2_symv(fp, u, n, a::T, A::Matrix{T}, lda, x, ix, b, y, iy) where {T} = ccall(
        fp, Cvoid, (Ref{UInt8}, Ref{BI}, Ref{T}, Ptr{T}, Ref{BI}, Ptr{T}, Ref{BI}, Ref{T}, Ptr{T}, Ref{BI}, Clong),
        c(u), n, a, A, lda, x, ix, b, y, iy, 1
    )
    l2_sbmv(fp, u, n, k, a::T, A::Matrix{T}, lda, x, ix, b, y, iy) where {T} = ccall(
        fp, Cvoid, (Ref{UInt8}, Ref{BI}, Ref{BI}, Ref{T}, Ptr{T}, Ref{BI}, Ptr{T}, Ref{BI}, Ref{T}, Ptr{T}, Ref{BI}, Clong),
        c(u), n, k, a, A, lda, x, ix, b, y, iy, 1
    )
    l2_spmv(fp, u, n, a::T, AP::Vector{T}, x, ix, b, y, iy) where {T} = ccall(
        fp, Cvoid, (Ref{UInt8}, Ref{BI}, Ref{T}, Ptr{T}, Ptr{T}, Ref{BI}, Ref{T}, Ptr{T}, Ref{BI}, Clong),
        c(u), n, a, AP, x, ix, b, y, iy, 1
    )
    l2_tr(fp, u, t, d, n, A::Matrix{T}, lda, x, ix) where {T} = ccall(
        fp, Cvoid, (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BI}, Ptr{T}, Ref{BI}, Ptr{T}, Ref{BI}, Clong, Clong, Clong),
        c(u), c(t), c(d), n, A, lda, x, ix, 1, 1, 1
    )
    l2_tb(fp, u, t, d, n, k, A::Matrix{T}, lda, x, ix) where {T} = ccall(
        fp, Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BI}, Ref{BI}, Ptr{T}, Ref{BI}, Ptr{T}, Ref{BI}, Clong, Clong, Clong),
        c(u), c(t), c(d), n, k, A, lda, x, ix, 1, 1, 1
    )
    l2_tp(fp, u, t, d, n, AP::Vector{T}, x, ix) where {T} = ccall(
        fp, Cvoid, (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BI}, Ptr{T}, Ptr{T}, Ref{BI}, Clong, Clong, Clong),
        c(u), c(t), c(d), n, AP, x, ix, 1, 1, 1
    )
    l2_spr(fp, u, n, a::S, x::Vector{T}, ix, AP) where {S, T} = ccall(
        fp, Cvoid, (Ref{UInt8}, Ref{BI}, Ref{S}, Ptr{T}, Ref{BI}, Ptr{T}, Clong),
        c(u), n, a, x, ix, AP, 1
    )
    l2_spr2(fp, u, n, a::T, x::Vector{T}, ix, y, iy, AP) where {T} = ccall(
        fp, Cvoid, (Ref{UInt8}, Ref{BI}, Ref{T}, Ptr{T}, Ref{BI}, Ptr{T}, Ref{BI}, Ptr{T}, Clong),
        c(u), n, a, x, ix, y, iy, AP, 1
    )

    l3_gemm(fp, ta, tb, m, n, k, a::T, A::Matrix{T}, lda, B, ldb, b, C, ldc) where {T} = ccall(
        fp, Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{BI}, Ref{BI}, Ref{BI}, Ref{T}, Ptr{T}, Ref{BI}, Ptr{T}, Ref{BI}, Ref{T}, Ptr{T}, Ref{BI}, Clong, Clong),
        c(ta), c(tb), m, n, k, a, A, lda, B, ldb, b, C, ldc, 1, 1
    )
    l3_symm(fp, s, u, m, n, a::T, A::Matrix{T}, lda, B, ldb, b, C, ldc) where {T} = ccall(
        fp, Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{BI}, Ref{BI}, Ref{T}, Ptr{T}, Ref{BI}, Ptr{T}, Ref{BI}, Ref{T}, Ptr{T}, Ref{BI}, Clong, Clong),
        c(s), c(u), m, n, a, A, lda, B, ldb, b, C, ldc, 1, 1
    )
    l3_syrk(fp, u, t, n, k, a::S, A::Matrix{T}, lda, b::S, C, ldc) where {S, T} = ccall(
        fp, Cvoid, (Ref{UInt8}, Ref{UInt8}, Ref{BI}, Ref{BI}, Ref{S}, Ptr{T}, Ref{BI}, Ref{S}, Ptr{T}, Ref{BI}, Clong, Clong),
        c(u), c(t), n, k, a, A, lda, b, C, ldc, 1, 1
    )
    l3_syr2k(fp, u, t, n, k, a::T, A::Matrix{T}, lda, B, ldb, b::S, C, ldc) where {S, T} = ccall(
        fp, Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{BI}, Ref{BI}, Ref{T}, Ptr{T}, Ref{BI}, Ptr{T}, Ref{BI}, Ref{S}, Ptr{T}, Ref{BI}, Clong, Clong),
        c(u), c(t), n, k, a, A, lda, B, ldb, b, C, ldc, 1, 1
    )
    l3_trxm(fp, s, u, t, d, m, n, a::T, A::Matrix{T}, lda, B, ldb) where {T} = ccall(
        fp, Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BI}, Ref{BI}, Ref{T}, Ptr{T}, Ref{BI}, Ptr{T}, Ref{BI}, Clong, Clong, Clong, Clong),
        c(s), c(u), c(t), c(d), m, n, a, A, lda, B, ldb, 1, 1, 1, 1
    )

    # Scalars that are not powers of two, so a misplaced α or β cannot be hidden by exact scaling.
    alpha(T) = T <: Complex ? T(0.7, -0.4) : T(0.7)
    beta(T) = T <: Complex ? T(-0.3, 0.2) : T(-0.3)

    # name => (level, builder). A builder takes an RNG and returns that symbol's cases; the level
    # items run the builders of their level and the completeness item checks the key set against
    # every forwarded symbol.
    const BUILDERS = Dict{String, Tuple{Int, Function}}()

    # Forwarded LAPACK symbols that the items in lbt_forward_tests.jl call after `activate()`,
    # found by counting calls into every forwarded symbol while those items ran, and the forwarded
    # LAPACK symbols none of them calls.
    const LAPACK_TESTED = """
    cbdsqr_ cgesdd_ cgesvx_ cggsvd_ cpotrf_ csyconv_ ctgsen_ ctrrfs_ dbdsdc_ dbdsqr_ dgbtrf_
    dgbtrs_ dgebak_ dgebrd_ dgees_ dgeevx_ dgehrd_ dgelqf_ dgelsd_ dgelsy_ dgemqrt_ dgeqlf_
    dgeqp3_ dgeqrf_ dgeqrt_ dgerqf_ dgesdd_ dgesv_ dgesvd_ dgesvx_ dgetrf_ dgetri_ dgetrs_
    dgges3_ dggev3_ dgglse_ dggsvd3_ dggsvd_ dgtsv_ dhseqr_ dlacpy_ dlarf_ dlarfg_ dorghr_
    dorglq_ dorgql_ dorgqr_ dorgrq_ dorgtr_ dormhr_ dormql_ dormqr_ dormtr_ dposv_ dpotrf_
    dpotri_ dpotrs_ dpstrf_ dptsv_ dpttrf_ dpttrs_ dstebz_ dstegr_ dstein_ dstev_ dsyconv_
    dsyevr_ dsygvd_ dsysv_ dsytrd_ dsytrf_ dsytri_ dsytrs_ dtgsen_ dtrcon_ dtrevc_ dtrexc_
    dtrrfs_ dtrsen_ dtrsyl_ dtrtrs_ sgemqrt_ sgeqrt_ sgesdd_ sgesv_ sgesvx_ sgetrf_ sggsvd_
    spotrf_ ssyconv_ stgsen_ strrfs_ zbdsqr_ zgeevx_ zgemqrt_ zgeqrf_ zgeqrt_ zgesdd_ zgesv_
    zgesvx_ zgetrf_ zgetrs_ zggev3_ zggsvd_ zheevr_ zhesv_ zhetrd_ zhetrf_ zhetrs_ zlacpy_
    zlarf_ zlarfg_ zposv_ zpotrf_ zsyconv_ ztgsen_ ztrrfs_ ztrtrs_ zungqr_ zungtr_ zunmtr_
    """
    const LAPACK_UNTESTED = """
    cgbtrf_ cgbtrs_ cgebak_ cgebal_ cgebrd_ cgecon_ cgees_ cgeev_ cgeevx_ cgehrd_ cgelqf_
    cgels_ cgelsd_ cgelsy_ cgemqrt_ cgeqlf_ cgeqp3_ cgeqrf_ cgeqrt_ cgerqf_ cgesv_ cgesvd_
    cgetrf_ cgetri_ cgetrs_ cgges3_ cgges_ cggev3_ cggev_ cgglse_ cggsvd3_ cgtsv_ cgttrf_
    cgttrs_ cheev_ cheevd_ cheevr_ chegvd_ chesv_ chetrd_ chetrf_ chetri_ chetrs_ chseqr_
    clacpy_ clarf_ clarfg_ cpbtrf_ cpbtrs_ cpocon_ cposv_ cpotri_ cpotrs_ cpptrf_ cpptrs_
    cpstrf_ cptsv_ cpttrf_ cpttrs_ csysv_ csytrf_ csytri_ csytrs_ ctrcon_ ctrevc_ ctrexc_
    ctrsen_ ctrsyl_ ctrtri_ ctrtrs_ ctzrzf_ cunghr_ cunglq_ cungql_ cungqr_ cungrq_ cungtr_
    cunmhr_ cunmlq_ cunmql_ cunmqr_ cunmrq_ cunmrz_ cunmtr_ dgebal_ dgecon_ dgeev_ dgels_
    dgges_ dggev_ dgttrf_ dgttrs_ dormlq_ dormrq_ dormrz_ dpbtrf_ dpbtrs_ dpocon_ dpptrf_
    dpptrs_ dsyev_ dsyevd_ dtrtri_ dtzrzf_ sbdsdc_ sbdsqr_ sgbtrf_ sgbtrs_ sgebak_ sgebal_
    sgebrd_ sgecon_ sgees_ sgeev_ sgeevx_ sgehrd_ sgelqf_ sgels_ sgelsd_ sgelsy_ sgeqlf_
    sgeqp3_ sgeqrf_ sgerqf_ sgesvd_ sgetri_ sgetrs_ sgges3_ sgges_ sggev3_ sggev_ sgglse_
    sggsvd3_ sgtsv_ sgttrf_ sgttrs_ shseqr_ slacpy_ slarf_ slarfg_ sorghr_ sorglq_ sorgql_
    sorgqr_ sorgrq_ sorgtr_ sormhr_ sormlq_ sormql_ sormqr_ sormrq_ sormrz_ sormtr_ spbtrf_
    spbtrs_ spocon_ sposv_ spotri_ spotrs_ spptrf_ spptrs_ spstrf_ sptsv_ spttrf_ spttrs_
    sstebz_ sstegr_ sstein_ sstev_ ssyev_ ssyevd_ ssyevr_ ssygvd_ ssysv_ ssytrd_ ssytrf_
    ssytri_ ssytrs_ strcon_ strevc_ strexc_ strsen_ strsyl_ strtri_ strtrs_ stzrzf_ zgbtrf_
    zgbtrs_ zgebak_ zgebal_ zgebrd_ zgecon_ zgees_ zgeev_ zgehrd_ zgelqf_ zgels_ zgelsd_
    zgelsy_ zgeqlf_ zgeqp3_ zgerqf_ zgesvd_ zgetri_ zgges3_ zgges_ zggev_ zgglse_ zggsvd3_
    zgtsv_ zgttrf_ zgttrs_ zheev_ zheevd_ zhegvd_ zhetri_ zhseqr_ zpbtrf_ zpbtrs_ zpocon_
    zpotri_ zpotrs_ zpptrf_ zpptrs_ zpstrf_ zptsv_ zpttrf_ zpttrs_ zsysv_ zsytrf_ zsytri_
    zsytrs_ ztrcon_ ztrevc_ ztrexc_ ztrsen_ ztrsyl_ ztrtri_ ztzrzf_ zunghr_ zunglq_ zungql_
    zungrq_ zunmhr_ zunmlq_ zunmql_ zunmqr_ zunmrq_ zunmrz_
    """
    reg!(level, name, f) = (BUILDERS[name] = (level, f); nothing)

    # ── BLAS-1 ──────────────────────────────────────────────────────────────────────────────────
    # The unit-stride kernels route on the SIMD step (`_UNROLL * _vwidth`), on the reduction block
    # grid (`_red_block`, for the real reductions and for complex ones reinterpreted as 2n reals),
    # on the pool admission (`_l1_workers`, which needs several blocks) and on working-set bytes
    # against L1, L2 and, for axpy, L3 — counting one, two or four streams. Strided calls take the
    # scalar loop, so they need only the small sizes.
    function l1_sizes(T; l3 = false)
        R = real(T); W = PureBLAS._vwidth(R); U = PureBLAS._UNROLL; B = PureBLAS._red_block(R)
        ns = Int[0, 1, 3, U * W - 1, U * W, 4W - 1, 4W, 4W + 1, 4U * W - 1, 4U * W, 4U * W + 1]
        for b in (B, B ÷ 2)
            append!(ns, (b - 1, b, b + 1, 2b + 3, 5b + 7))
        end
        cuts = [(cb, f) for cb in (PureBLAS._L1_BYTES, PureBLAS._L2_BYTES) for f in (1, 2, 4)]
        l3 && push!(cuts, (PureBLAS._L3_BYTES, 2))        # axpy's two-stream DRAM arm
        for (cb, f) in cuts
            n0 = cb ÷ (f * sizeof(T))
            append!(ns, (n0 - 1, n0, n0 + 1))
        end
        return sort!(unique!(filter!(>=(0), ns)))
    end
    l1_small(T) = filter(n -> n <= 2 * PureBLAS._red_block(real(T)) + 3, l1_sizes(T))

    # (n, incx, incy) layouts: unit stride at every size, strides 2 and -2 at the small ones.
    l1_layouts2(T; l3 = false) = vcat(
        [(n, 1, 1) for n in l1_sizes(T; l3)],
        [(n, ix, iy) for n in l1_small(T) for (ix, iy) in ((2, 2), (2, 1), (1, -2), (-2, 2))]
    )
    l1_layouts1(T) = vcat([(n, 1) for n in l1_sizes(T)], [(n, 2) for n in l1_small(T)])

    # Scalars for axpy/scal: complex types also take a real-valued α, which has its own route.
    l1_alphas(T) = T <: Complex ? (alpha(T), T(0.7)) : (alpha(T),)

    for T in TYPES
        p = pre(T)
        reg!(
            1, p * "axpy_", rng -> begin
                fp = fptr(p * "axpy_"); cs = Case[]
                for (n, ix, iy) in l1_layouts2(T; l3 = true), a in l1_alphas(T)
                    x = vbuf(rng, T, n, ix); y0 = vbuf(rng, T, n, iy)
                    push!(
                        cs, Case(
                            "$T n=$n incx=$ix incy=$iy α=$a",
                            () -> (y = copy(y0); l1_axpy(fp, n, a, x, ix, y, iy); (y,)), (y0,), rtol_for(T, 1)
                        )
                    )
                end
                cs
            end
        )
        reg!(
            1, p * "scal_", rng -> begin
                fp = fptr(p * "scal_"); cs = Case[]
                for (n, ix) in l1_layouts1(T), a in l1_alphas(T)
                    x0 = vbuf(rng, T, n, ix)
                    push!(
                        cs, Case(
                            "$T n=$n incx=$ix α=$a",
                            () -> (x = copy(x0); l1_scal(fp, n, a, x, ix); (x,)), (x0,), rtol_for(T, 1)
                        )
                    )
                end
                cs
            end
        )
        for op in ("copy_", "swap_")
            reg!(
                1, p * op, rng -> begin
                    fp = fptr(p * op); cs = Case[]
                    for (n, ix, iy) in l1_layouts2(T)
                        x0 = vbuf(rng, T, n, ix); y0 = vbuf(rng, T, n, iy)
                        push!(
                            cs, Case(
                                "$T n=$n incx=$ix incy=$iy",
                                () -> (x = copy(x0); y = copy(y0); l1_xy(fp, n, x, ix, y, iy); (x, y)),
                                (x0, y0), 0.0
                            )
                        )
                    end
                    cs
                end
            )
        end
        for op in (T <: Complex ? ("dotu_", "dotc_") : ("dot_",))
            reg!(
                1, p * op, rng -> begin
                    fp = fptr(p * op); cs = Case[]
                    for (n, ix, iy) in l1_layouts2(T)
                        x = vbuf(rng, T, n, ix); y = vbuf(rng, T, n, iy)
                        push!(
                            cs, Case(
                                "$T n=$n incx=$ix incy=$iy",
                                () -> (l1_dot(fp, n, x, ix, y, iy),), (nothing,), rtol_for(T, n)
                            )
                        )
                    end
                    cs
                end
            )
        end
        rp = T === ComplexF32 ? "sc" : T === ComplexF64 ? "dz" : p
        ip = T === ComplexF32 ? "ic" : T === ComplexF64 ? "iz" : "i" * p
        # nrm2 falls back to scaled accumulation when the plain sum of squares overflows or
        # underflows, so it also takes huge, tiny and all-zero vectors.
        for (op, f) in ((rp * "nrm2_", l1_real), (rp * "asum_", l1_real), (ip * "amax_", l1_iamax))
            reg!(
                1, op, rng -> begin
                    fp = fptr(op); cs = Case[]
                    big = sqrt(floatmax(real(T))); tiny = floatmin(real(T))
                    scales = startswith(op, rp * "nrm2") ? (1, big, tiny, 0) : (1,)
                    for (n, ix) in l1_layouts1(T), s in scales
                        x = vbuf(rng, T, n, ix) .* real(T)(s)
                        push!(
                            cs, Case(
                                "$T n=$n incx=$ix scale=$s",
                                () -> (f(fp, n, x, ix),), (nothing,), rtol_for(T, n)
                            )
                        )
                    end
                    # Ties resolve to the lowest index.
                    startswith(op, ip) && push!(
                        cs, Case(
                            "$T ties",
                            () -> (f(fp, 1000, ones(T, 1000), 1),), (nothing,), 0.0
                        )
                    )
                    cs
                end
            )
        end
    end

    # ── BLAS-2 operands ─────────────────────────────────────────────────────────────────────────
    # A triangular operand is well conditioned with either diagonal convention: off-diagonal
    # entries of the referenced triangle are O(1/n), the diagonal is in [1, 2). The other triangle
    # and the rows below `n` in the leading dimension hold O(1) random values, which a correct
    # kernel never reads.
    function trimat(rng, T, n, lda, uplo)
        A = randn(rng, T, lda, n)
        for j in 1:n, i in 1:n
            inref = uplo == 'U' ? i < j : i > j
            inref && (A[i, j] /= n)
            i == j && (A[i, j] = 1 + rand(rng, real(T)))
        end
        return A
    end
    # Packed triangle with the same conditioning; `pdiag` lists the diagonal's positions.
    pdiag(n, uplo) = uplo == 'U' ? [j * (j + 1) ÷ 2 for j in 1:n] :
        [(j - 1) * n - (j - 1) * (j - 2) ÷ 2 + 1 for j in 1:n]
    function tripacked(rng, T, n, uplo)
        AP = randn(rng, T, n * (n + 1) ÷ 2) ./ max(n, 1)
        for d in pdiag(n, uplo)
            AP[d] = 1 + rand(rng, real(T))
        end
        return AP
    end
    # Triangular band in (k+1)×n band storage: diagonal in row k+1 (upper) or row 1 (lower).
    function triband(rng, T, n, k, lda, uplo)
        AB = randn(rng, T, lda, n)
        AB[1:(k + 1), :] ./= (k + 1)
        drow = uplo == 'U' ? k + 1 : 1
        for j in 1:n
            AB[drow, j] = 1 + rand(rng, real(T))
        end
        return AB
    end

    # (incx, incy, extra rows in the leading dimension). Strided vectors take the generic loop,
    # which has no size routes, so the large sizes run unit stride only.
    l2_layouts(small::Bool) = small ? ((1, 1, 0), (1, 1, 3), (2, 1, 3), (1, 2, 3), (2, 2, 3)) :
        ((1, 1, 3),)

    # A y buffer of NaN at the referenced positions: with β = 0 the reference never reads y.
    function nan_at(y0, n, inc)
        y = copy(y0)
        for i in 0:(n - 1)
            y[1 + i * abs(inc)] = oftype(y[1], NaN)
        end
        return y
    end

    # ── BLAS-2 builders ─────────────────────────────────────────────────────────────────────────
    # `shapes` yields (m, n, small) — `small` selects the full layout set. Every symbol also takes
    # α = 0 and β = 0 against a NaN-filled y at one small shape. Those extra cases sit in a `let`:
    # a closure captures a variable, not its value, so a name the function assigns again after the
    # loop would hand every case the last operands.
    function gemv_cases(rng, T, fp, shapes)
        cs = Case[]
        for (m, n, small) in shapes, t in "NTC", (ix, iy, pad) in l2_layouts(small)
            lda = m + pad; lx, ly = t == 'N' ? (n, m) : (m, n)
            A = randn(rng, T, lda, n); x = vbuf(rng, T, lx, ix); y0 = vbuf(rng, T, ly, iy)
            push!(
                cs, Case(
                    "$T $t m=$m n=$n incx=$ix incy=$iy lda=$lda",
                    () -> (y = copy(y0); l2_gemv(fp, t, m, n, alpha(T), A, lda, x, ix, beta(T), y, iy); (y,)),
                    (y0,), rtol_for(T, max(m, n))
                )
            )
        end
        let m = 13, n = 9
            for t in "NT"
                lx, ly = t == 'N' ? (n, m) : (m, n)
                A = randn(rng, T, m, n); x = vbuf(rng, T, lx, 1); y0 = vbuf(rng, T, ly, 1); yn = nan_at(y0, ly, 1)
                push!(
                    cs, Case(
                        "$T $t α=0", () -> (y = copy(y0); l2_gemv(fp, t, m, n, zero(T), A, m, x, 1, beta(T), y, 1); (y,)),
                        (y0,), rtol_for(T, 1)
                    )
                )
                push!(
                    cs, Case(
                        "$T $t β=0 NaN y", () -> (y = copy(yn); l2_gemv(fp, t, m, n, alpha(T), A, m, x, 1, zero(T), y, 1); (y,)),
                        (yn,), rtol_for(T, n)
                    )
                )
            end
        end
        return cs
    end

    function ger_cases(rng, T, fp, shapes)
        cs = Case[]
        for (m, n, small) in shapes, (ix, iy, pad) in l2_layouts(small)
            lda = m + pad
            A0 = randn(rng, T, lda, n); x = vbuf(rng, T, m, ix); y = vbuf(rng, T, n, iy)
            push!(
                cs, Case(
                    "$T m=$m n=$n incx=$ix incy=$iy lda=$lda",
                    () -> (A = copy(A0); l2_ger(fp, m, n, alpha(T), x, ix, y, iy, A, lda); (A,)),
                    (A0,), rtol_for(T, 1)
                )
            )
        end
        return cs
    end

    # symv/hemv. The full matrix is random, so the unreferenced triangle and (for hemv) the
    # imaginary part of the diagonal differ from what the referenced half implies.
    function symv_cases(rng, T, fp, ns)
        cs = Case[]
        for (n, small) in ns, u in "UL", (ix, iy, pad) in l2_layouts(small)
            lda = n + pad
            A = randn(rng, T, lda, n); x = vbuf(rng, T, n, ix); y0 = vbuf(rng, T, n, iy)
            push!(
                cs, Case(
                    "$T $u n=$n incx=$ix incy=$iy lda=$lda",
                    () -> (y = copy(y0); l2_symv(fp, u, n, alpha(T), A, lda, x, ix, beta(T), y, iy); (y,)),
                    (y0,), rtol_for(T, n)
                )
            )
        end
        let n = 11, A = randn(rng, T, n, n), x = vbuf(rng, T, n, 1), y0 = vbuf(rng, T, n, 1), yn = nan_at(y0, n, 1)
            push!(
                cs, Case(
                    "$T α=0", () -> (y = copy(y0); l2_symv(fp, 'U', n, zero(T), A, n, x, 1, beta(T), y, 1); (y,)),
                    (y0,), rtol_for(T, 1)
                )
            )
            push!(
                cs, Case(
                    "$T β=0 NaN y", () -> (y = copy(yn); l2_symv(fp, 'L', n, alpha(T), A, n, x, 1, zero(T), y, 1); (y,)),
                    (yn,), rtol_for(T, n)
                )
            )
        end
        return cs
    end

    # trmv/trsv; trans 'C' on a real type means 'T'.
    function tr_cases(rng, T, fp, ns)
        cs = Case[]
        for (n, small) in ns, u in "UL"
            for (ix, _, pad) in l2_layouts(small)
                lda = n + pad; A = trimat(rng, T, n, lda, u); x0 = vbuf(rng, T, n, ix)
                for t in "NTC", d in "NU"
                    push!(
                        cs, Case(
                            "$T $u$t$d n=$n incx=$ix lda=$lda",
                            () -> (x = copy(x0); l2_tr(fp, u, t, d, n, A, lda, x, ix); (x,)),
                            (x0,), rtol_for(T, n)
                        )
                    )
                end
            end
        end
        return cs
    end

    function tp_cases(rng, T, fp, ns)
        cs = Case[]
        for (n, small) in ns, u in "UL", ix in (small ? (1, 2) : (1,))
            AP = tripacked(rng, T, n, u); x0 = vbuf(rng, T, n, ix)
            for t in "NTC", d in "NU"
                push!(
                    cs, Case(
                        "$T $u$t$d n=$n incx=$ix",
                        () -> (x = copy(x0); l2_tp(fp, u, t, d, n, AP, x, ix); (x,)), (x0,), rtol_for(T, n)
                    )
                )
            end
        end
        return cs
    end

    function spmv_cases(rng, T, fp, ns)
        cs = Case[]
        for (n, small) in ns, u in "UL", (ix, iy, _) in l2_layouts(small)
            AP = randn(rng, T, n * (n + 1) ÷ 2); x = vbuf(rng, T, n, ix); y0 = vbuf(rng, T, n, iy)
            push!(
                cs, Case(
                    "$T $u n=$n incx=$ix incy=$iy",
                    () -> (y = copy(y0); l2_spmv(fp, u, n, alpha(T), AP, x, ix, beta(T), y, iy); (y,)),
                    (y0,), rtol_for(T, n)
                )
            )
        end
        let n = 11, AP = randn(rng, T, n * (n + 1) ÷ 2), x = vbuf(rng, T, n, 1), y0 = vbuf(rng, T, n, 1), yn = nan_at(y0, n, 1)
            push!(
                cs, Case(
                    "$T α=0", () -> (y = copy(y0); l2_spmv(fp, 'U', n, zero(T), AP, x, 1, beta(T), y, 1); (y,)),
                    (y0,), rtol_for(T, 1)
                )
            )
            push!(
                cs, Case(
                    "$T β=0 NaN y", () -> (y = copy(yn); l2_spmv(fp, 'L', n, alpha(T), AP, x, 1, zero(T), y, 1); (y,)),
                    (yn,), rtol_for(T, n)
                )
            )
        end
        return cs
    end

    # spr/hpr (α real for hpr) and spr2/hpr2.
    function spr_cases(rng, T, fp, ns; two::Bool)
        cs = Case[]
        a = two ? alpha(T) : real(alpha(T))
        for (n, small) in ns, u in "UL", (ix, iy, _) in l2_layouts(small)
            AP0 = randn(rng, T, n * (n + 1) ÷ 2); x = vbuf(rng, T, n, ix); y = vbuf(rng, T, n, iy)
            run = two ? () -> (AP = copy(AP0); l2_spr2(fp, u, n, a, x, ix, y, iy, AP); (AP,)) :
                () -> (AP = copy(AP0); l2_spr(fp, u, n, a, x, ix, AP); (AP,))
            push!(cs, Case("$T $u n=$n incx=$ix" * (two ? " incy=$iy" : ""), run, (AP0,), rtol_for(T, 1)))
        end
        return cs
    end

    # gbmv over (m, n, kl, ku); the band storage has kl+ku+1 referenced rows.
    function gbmv_cases(rng, T, fp, shapes)
        cs = Case[]
        for (m, n, kl, ku, small) in shapes, t in "NTC", (ix, iy, pad) in l2_layouts(small)
            lda = kl + ku + 1 + pad; lx, ly = t == 'N' ? (n, m) : (m, n)
            AB = randn(rng, T, lda, n); x = vbuf(rng, T, lx, ix); y0 = vbuf(rng, T, ly, iy)
            push!(
                cs, Case(
                    "$T $t m=$m n=$n kl=$kl ku=$ku incx=$ix incy=$iy lda=$lda",
                    () -> (y = copy(y0); l2_gbmv(fp, t, m, n, kl, ku, alpha(T), AB, lda, x, ix, beta(T), y, iy); (y,)),
                    (y0,), rtol_for(T, kl + ku + 1)
                )
            )
        end
        let m = 13, n = 9, kl = 2, ku = 3, AB = randn(rng, T, kl + ku + 1, n), x = vbuf(rng, T, n, 1),
                y0 = vbuf(rng, T, m, 1), yn = nan_at(y0, m, 1)
            push!(
                cs, Case(
                    "$T α=0", () -> (y = copy(y0); l2_gbmv(fp, 'N', m, n, kl, ku, zero(T), AB, kl + ku + 1, x, 1, beta(T), y, 1); (y,)),
                    (y0,), rtol_for(T, 1)
                )
            )
            push!(
                cs, Case(
                    "$T β=0 NaN y", () -> (y = copy(yn); l2_gbmv(fp, 'N', m, n, kl, ku, alpha(T), AB, kl + ku + 1, x, 1, zero(T), y, 1); (y,)),
                    (yn,), rtol_for(T, kl + ku + 1)
                )
            )
        end
        return cs
    end

    function sbmv_cases(rng, T, fp, shapes)
        cs = Case[]
        for (n, k, small) in shapes, u in "UL", (ix, iy, pad) in l2_layouts(small)
            lda = k + 1 + pad
            AB = randn(rng, T, lda, n); x = vbuf(rng, T, n, ix); y0 = vbuf(rng, T, n, iy)
            push!(
                cs, Case(
                    "$T $u n=$n k=$k incx=$ix incy=$iy lda=$lda",
                    () -> (y = copy(y0); l2_sbmv(fp, u, n, k, alpha(T), AB, lda, x, ix, beta(T), y, iy); (y,)),
                    (y0,), rtol_for(T, 2k + 1)
                )
            )
        end
        let n = 11, k = 3, AB = randn(rng, T, k + 1, n), x = vbuf(rng, T, n, 1), y0 = vbuf(rng, T, n, 1),
                yn = nan_at(y0, n, 1)
            push!(
                cs, Case(
                    "$T α=0", () -> (y = copy(y0); l2_sbmv(fp, 'U', n, k, zero(T), AB, k + 1, x, 1, beta(T), y, 1); (y,)),
                    (y0,), rtol_for(T, 1)
                )
            )
            push!(
                cs, Case(
                    "$T β=0 NaN y", () -> (y = copy(yn); l2_sbmv(fp, 'L', n, k, alpha(T), AB, k + 1, x, 1, zero(T), y, 1); (y,)),
                    (yn,), rtol_for(T, 2k + 1)
                )
            )
        end
        return cs
    end

    function tb_cases(rng, T, fp, shapes)
        cs = Case[]
        for (n, k, small) in shapes, u in "UL"
            for (ix, _, pad) in l2_layouts(small)
                lda = k + 1 + pad; AB = triband(rng, T, n, k, lda, u); x0 = vbuf(rng, T, n, ix)
                for t in "NTC", d in "NU"
                    push!(
                        cs, Case(
                            "$T $u$t$d n=$n k=$k incx=$ix lda=$lda",
                            () -> (x = copy(x0); l2_tb(fp, u, t, d, n, k, AB, lda, x, ix); (x,)),
                            (x0,), rtol_for(T, n)
                        )
                    )
                end
            end
        end
        return cs
    end

    # ── BLAS-2 sizes ────────────────────────────────────────────────────────────────────────────
    # Shapes on both sides of the cuts the BLAS-2 kernels route on, from the module's constants:
    #   gemv  row-block vs panel (`_gemvn_rb()`), panel widths (`_GEMV_NP`, `_GEMVT_NC`,
    #         `_GEMVT_NC_DEEP`, `_CGEMVT_NC`, `_CGEMVN_NC`), row blocks (`_GEMV_MR * W`), and the
    #         working-set cuts at L1, L2, L3/2 and the pool floor `_L1_MT_MIN`;
    #   ger   the panel route at m·n·sizeof ≥ L3 (real) or L3/`_CGER_COLD_DEN` (complex), and the
    #         per-column axpy arms at L1 and two streams against L3;
    #   symv  the column panel `min(_SYMV_NB, W)` and the block grid `_symv_nblk`, which changes
    #         at n²·sizeof/2 = `_L1_MT_MIN` and 2·L3; hemv's panel `_ZHEMV_NB`;
    #   trmv/trsv  `_TRSV_REG_MAX`, 2·`_TRSV_T_F`, `_TRI_NB`, `_TRI_T_UNB`, `_TRI_C_BLK_MIN`,
    #         `_TRI_C_T_UNB`, the 512-entry complex reciprocal buffer, and trmv's F switch at
    #         n(n+1)/2·sizeof = `_TRMV_F_SWITCH`·L3;
    #   packed  spmv's panel route at n(n+1)/2·sizeof = `_SPMV_PANEL_MINAP`;
    #   banded  band width against W, `_GBMV_CONV_MAX` and `_GBMV_MT_BAND_MAX`.
    # `W` is the SIMD width of the real element type. A cut on bytes is straddled by the largest
    # size that stays under it and the next one.
    const P = PureBLAS
    wid(T) = P._vwidth(real(T))
    under(bytes, T, per = 1) = max(1, bytes ÷ (per * sizeof(T)))           # largest m with m·per·sizeof ≤ bytes
    sqrt_under(bytes, T, f) = max(1, isqrt(f * bytes ÷ sizeof(T)))          # n with n²·sizeof/f ≈ bytes
    straddle(v) = (max(v - 1, 1), v, v + 1)

    function gemv_shapes(T)
        W = wid(T)
        ms = unique!([1, 3, W - 1, W, W + 1, P._GEMV_MR * W - 1, P._GEMV_MR * W + 1, 2 * P._GEMV_MR * W + 3])
        ns = unique!([1, 3, 4, 5, 7, 8, 9, 16, 17, 2 * P._GEMVT_NC_DEEP + 3])
        sh = Any[(m, n, true) for m in ms for n in ns]
        for n in straddle(P._gemvn_rb())
            push!(sh, (P._GEMV_MR * W + 5, n, false))
        end
        for cb in (P._L1_BYTES, P._L2_BYTES, P._L3_BYTES ÷ 2, P._L1_MT_MIN), n in (9, 17)
            m0 = under(cb, T, n)
            push!(sh, (m0, n, false), (m0 + 1, n, false))
        end
        m0 = under(P._L1_BYTES, T)                     # gemv-T: x resident in L1
        push!(sh, (m0, 9, false), (m0 + 1, 9, false))
        return sh
    end

    function ger_shapes(T)
        W = wid(T)
        sh = Any[(m, n, true) for m in (1, 3, W + 1, 4 * P._UNROLL * W + 1) for n in (1, 2, 5, 9)]
        for cb in (P._L3_BYTES, P._L3_BYTES ÷ P._CGER_COLD_DEN), n in (5, 9)
            m0 = under(cb, T, n)
            push!(sh, (m0, n, false), (m0 + 1, n, false))
        end
        for (cb, per) in ((P._L1_BYTES, 1), (P._L1_BYTES, 4), (P._L2_BYTES, 2), (P._L3_BYTES, 2))
            m0 = under(cb, T, per)
            push!(sh, (m0, 2, false), (m0 + 1, 2, false))
        end
        return sh
    end

    function symv_ns(T)
        W = wid(T); nb = min(P._SYMV_NB, W)
        ns = Any[(n, true) for n in unique!([1, 2, 3, nb - 1, nb, nb + 1, 2nb + 1, P._ZHEMV_NB + 1, 2W + 3, 33])]
        for v in (sqrt_under(P._L1_MT_MIN, T, 2), sqrt_under(2 * P._L3_BYTES, T, 2))
            append!(ns, [(n, false) for n in straddle(v)])
        end
        return ns
    end

    function tr_ns(T)
        nb = P._TRI_NB
        small = unique!([1, 2, 3, 7, 8, 9, 15, 16, 17, nb - 1, nb + 1, 2nb + 3])
        cuts = T <: Real ?
            [P._TRSV_REG_MAX, 2 * P._TRSV_T_F, nb, P._TRI_T_UNB, max(nb, P._TRI_T_UNB) + nb] :
            [P._TRI_C_BLK_MIN, P._TRI_C_T_UNB, 512, nb]
        ns = Any[(n, true) for n in small]
        for v in cuts, n in straddle(v)
            push!(ns, (n, false))
        end
        if T <: Real                               # trmv's DRAM panel width: n(n+1)/2·sizeof vs F·L3
            v = sqrt_under(P._TRMV_F_SWITCH * P._L3_BYTES, T, 2)
            push!(ns, (v - 1, false), (v + 1, false))
        end
        return unique!(ns)
    end

    function packed_ns(T)
        W = wid(T)
        ns = Any[(n, true) for n in unique!([1, 2, 3, 7, 8, 9, W - 1, W, W + 1, 2W + 3, 33])]
        v = sqrt_under(P._SPMV_PANEL_MINAP, T, 2)
        append!(ns, [(n, false) for n in straddle(v)])
        return ns
    end

    function gbmv_shapes(T)
        W = wid(T)
        sh = Any[(m, n, kl, ku, true) for (m, n) in ((7, 7), (13, 9), (9, 13)) for (kl, ku) in ((0, 0), (1, 2), (3, 1), (8, 8))]
        for b in unique!([W - 1, W, W + 1, P._GBMV_CONV_MAX, P._GBMV_CONV_MAX + 1, P._GBMV_MT_BAND_MAX, P._GBMV_MT_BAND_MAX + 1])
            kl = (b - 1) ÷ 2; ku = b - 1 - kl
            for (dm, dn) in ((4W + 9, 4W + 5), (4W + 5, 4W + 9))
                push!(sh, (b + dm, b + dn, kl, ku, false))
            end
        end
        return sh
    end

    function band_shapes(T)
        W = wid(T)
        sh = Any[(n, k, true) for n in (1, 5, 2W + 3) for k in (0, 1, 3, 7)]
        for k in (W - 1, W, W + 1), n in (k + 2, 4W + 5)
            push!(sh, (n, k, false))
        end
        return unique!(sh)
    end

    for T in TYPES
        p = pre(T); cx = T <: Complex
        reg!(2, p * "gemv_", rng -> gemv_cases(rng, T, fptr(p * "gemv_"), gemv_shapes(T)))
        for nm in (cx ? ("geru_", "gerc_") : ("ger_",))
            reg!(2, p * nm, rng -> ger_cases(rng, T, fptr(p * nm), ger_shapes(T)))
        end
        let nm = p * (cx ? "hemv_" : "symv_")
            reg!(2, nm, rng -> symv_cases(rng, T, fptr(nm), symv_ns(T)))
        end
        for nm in ("trmv_", "trsv_")
            reg!(2, p * nm, rng -> tr_cases(rng, T, fptr(p * nm), tr_ns(T)))
        end
        let nm = p * (cx ? "hpmv_" : "spmv_")
            reg!(2, nm, rng -> spmv_cases(rng, T, fptr(nm), packed_ns(T)))
        end
        for nm in ("tpmv_", "tpsv_")
            reg!(2, p * nm, rng -> tp_cases(rng, T, fptr(p * nm), packed_ns(T)))
        end
        for (nm, two) in (cx ? (("hpr_", false), ("hpr2_", true)) : (("spr_", false), ("spr2_", true)))
            reg!(2, p * nm, rng -> spr_cases(rng, T, fptr(p * nm), packed_ns(T); two))
        end
        reg!(2, p * "gbmv_", rng -> gbmv_cases(rng, T, fptr(p * "gbmv_"), gbmv_shapes(T)))
        let nm = p * (cx ? "hbmv_" : "sbmv_")
            reg!(2, nm, rng -> sbmv_cases(rng, T, fptr(nm), band_shapes(T)))
        end
        for nm in ("tbmv_", "tbsv_")
            reg!(2, p * nm, rng -> tb_cases(rng, T, fptr(p * nm), band_shapes(T)))
        end
    end

    # ── BLAS-3 builders ─────────────────────────────────────────────────────────────────────────
    # A shape is (dims, trans flags, extra leading-dimension rows, βs, αs). `pads` containing 0
    # reaches the routes that require a leading dimension equal to the row count; `PAD` rows in an
    # output's leading dimension must stay intact.
    const PAD = 3
    const T9 = [(a, b) for a in "NTC" for b in "NTC"]
    const T3 = [('N', 'N'), ('T', 'N'), ('N', 'T')]

    function gemm_cases(rng, T, fp, shapes)
        cs = Case[]
        for ((m, n, k), fl, pads, bs, as) in shapes, (ta, tb) in fl, pad in pads, b in bs, a in as
            ar, ac = ta == 'N' ? (m, k) : (k, m); br, bc = tb == 'N' ? (k, n) : (n, k)
            lda, ldb, ldc = max(ar + pad, 1), max(br + pad, 1), max(m + pad, 1)
            A = randn(rng, T, lda, ac); B = randn(rng, T, ldb, bc)
            C0 = randn(rng, T, ldc, n)
            iszero(b) && (C0[1:m, :] .= T(NaN))
            push!(
                cs, Case(
                    "$T $ta$tb m=$m n=$n k=$k pad=$pad β=$b α=$a",
                    () -> (C = copy(C0); l3_gemm(fp, ta, tb, m, n, k, a, A, lda, B, ldb, b, C, ldc); (C,)),
                    (C0,), rtol_for(T, k)
                )
            )
        end
        return cs
    end

    # The pool's work floor (flops) once every thread is in it, or `nothing` with one thread.
    function mt_floor()
        Threads.nthreads() > 1 || return nothing
        old = PureBLAS.get_num_threads(); oldfloor = PureBLAS._MT_FLOOR[]
        try
            PureBLAS.set_num_threads(Threads.nthreads())
            f = PureBLAS._MT_FLOOR[]
            return f == typemax(Int) ? nothing : f
        finally
            PureBLAS.set_num_threads(old)
            PureBLAS._MT_FLOOR[] = oldfloor     # raising the thread count sets the floor; leave it as found
        end
    end
    mt_side(flops, deg) = ceil(Int, (flops / deg)^(1 / 3)) + 1        # n with deg·n³ ≥ flops

    # gemm routes, in the order `_gemm_strided!` tries them: m or n = 1 (gemv), small m against
    # long k (`_rows_dot_max`), k = 1 up to `_GEMM_MR1_MAX`, m below the tail width, the tiny
    # kernel (`_GEMM_TINY`), Strassen (real, min(m,n,k) ≥ 2·`_STRASSEN_BASE`), the unpacked kernel
    # (`_GEMM_UNPACK_MAX`, with its split arm), and the blocked driver past `_KC` and `_NC`.
    # Complex: `_CGEMM_TINY`, `_CGEMM_UNPACK_MAX`, the 3M window `_CGEMM_3M_MIN`..`_CGEMM_3M_MAX`
    # with `_CGEMM_3M_KMIN`, the NR cuts and `_CKC`. Real 'C' disables the tiny route, so every
    # small shape runs all nine trans pairs.
    function gemm_shapes(T)
        W = wid(T); bs = (beta(T), zero(T))
        as = T <: Complex ? (alpha(T), one(T), T(0.7)) : (alpha(T),)
        sh = Any[]
        tiny(d...) = push!(sh, (d, T9, (0, PAD), bs, as))
        mid(d...) = push!(sh, (d, T9, (PAD,), bs, (alpha(T),)))
        big(d...; fl = T3) = push!(sh, (d, fl, (PAD,), (beta(T),), (alpha(T),)))
        for d in (
                (0, 5, 3), (5, 0, 3), (5, 3, 0), (1, 7, 5), (7, 1, 5), (1, 1, 9), (5, 7, 1), (6, 6, 6),
                (7, 6, 6), (6, 7, 6), (6, 6, 7), (2, 9, 3W), (W - 1, 9, 2), (W + 1, 9, 2), (3, 2 * P._NR + 1, 5),
            )
            tiny(d...)
        end
        if T <: Real
            rd = P._rows_dot_max(T); tv = P._at_tail_vw(P._HW, T)
            tiny(rd, 9, (rd + 1) * W); tiny(rd + 1, 9, (rd + 2) * W); tiny(tv, 9, 5); tiny(tv + 1, 9, 5)
        end
        r1 = P._GEMM_MR1_MAX
        tiny(r1, 7, 1); tiny(r1 + 1, 7, 1); mid(r1, r1, r1); mid(r1 + 1, r1, r1)
        um = P._GEMM_UNPACK_MAX
        mid(um, 9, 9); mid(um + 1, 9, 9); mid(9, um + 1, 9); mid(r1 + 1, 9, um + 1)
        mid(r1 + 1, P._NR + 1, 33); mid(r1 + 1, 2 * P._NR - 1, 33)
        if P._SPLIT_OK
            sm = P._gemm_split_max()
            mid(P._SMR * W, 9, 9); mid(min(sm, um), 17, 9); mid(sm + 1, 17, 9)
        end
        big(um + 1, 2 * P._NR + 3, P._KC + 5); big(r1 + 1, P._NC + 3, 9)
        if T <: Complex
            mid(P._CGEMM_TINY, 5, 5); mid(P._CGEMM_TINY + 1, 5, 5)
            cu = P._CGEMM_UNPACK_MAX; mid(cu, 9, 9); mid(cu + 1, 9, 9)
            lo, hi, kmin = P._CGEMM_3M_MIN, P._CGEMM_3M_MAX, P._CGEMM_3M_KMIN
            mid(lo - 1, 20, kmin); mid(lo, 20, kmin); mid(lo, 20, kmin - 1)
            big(hi, kmin, kmin); big(hi + 1, kmin, kmin)
            P._CUKER_NR6_MIN < typemax(Int) && (mid(P._CUKER_NR6_MIN - 1, 9, 9); mid(P._CUKER_NR6_MIN, 9, 9))
            P._CGEMM_NRSMALL_MAX > 0 && (mid(P._CGEMM_NRSMALL_MAX, 9, 9); mid(P._CGEMM_NRSMALL_MAX + 1, 9, 9))
            big(cu + 1, 17, P._CKC + 5)
        else
            s = 2 * P._fh_strassen_base()
            big(s, s, s; fl = [('N', 'N'), ('N', 'T')])
        end
        f = mt_floor()
        isnothing(f) || (s = max(mt_side(f, 2), P._NR + 1); big(s, s, s))
        return sh
    end

    # (m, n, α): α = 1 reaches trsm's direct small-k base, α = 0 the quick return.
    function trxm_cases(rng, T, fp, shapes)
        cs = Case[]
        for (m, n, as, ldp) in shapes, s in "LR", u in "UL"
            ka = s == 'L' ? m : n
            lda = ka + PAD; ldb = isnothing(ldp) ? m + PAD : ldp
            A = trimat(rng, T, ka, lda, u); B0 = randn(rng, T, ldb, n)
            for t in "NTC", d in "NU", a in as
                push!(
                    cs, Case(
                        "$T $s$u$t$d m=$m n=$n ldb=$ldb α=$a",
                        () -> (B = copy(B0); l3_trxm(fp, s, u, t, d, m, n, a, A, lda, B, ldb); (B,)),
                        (B0,), rtol_for(T, ka)
                    )
                )
            end
        end
        return cs
    end

    # trmm/trsm: the triangle's order k against `_TRMM_DDIRECT`, `_TRSM_DBASE`, `_TRSM_BASE`,
    # `_TRMM_BASE`/`_TRMM_BASE_R`, `_CTRMM_PACK_MIN`, `_TRMM_PACK_MIN`, `_TRSM_R_FUSE`,
    # `_TRMM_RPANEL` and `_TRMM_RPACK`; the other dimension against `_TRSM_NARROW_MAX`, W,
    # `_TRSM_NCUT` (side L) and `_TRSM_NCUT_R` (side R); a leading dimension that is a multiple of
    # the L1 way (`_L1_WAY_D`), which side-R trsm routes on; and the pool floor.
    function trxm_shapes(T)
        W = wid(T); as = (alpha(T), one(T), zero(T)); a1 = (alpha(T), one(T))
        sh = Any[]
        for (m, n) in ((1, 1), (3, 2), (5, 7), (W - 1, W + 1), (W + 1, 3))
            push!(sh, (m, n, as, nothing))
        end
        ks = unique!(
            [
                P._TRMM_DDIRECT, P._TRSM_DBASE, P._TRSM_BASE, P._TRMM_BASE, P._TRMM_BASE_R,
                P._CTRMM_PACK_MIN, P._TRSM_R_FUSE, P._ZGT_BASE, P._CTRSM_REC_L,
            ]
        )
        for k in ks, kk in (k, k + 1), o in (P._TRSM_NARROW_MAX, W - 1, P._TRSM_NCUT + 1)
            push!(sh, (kk, o, a1, nothing), (o, kk, a1, nothing))
        end
        for k in (P._TRSM_NCUT, P._TRSM_NCUT + 1, P._TRSM_NCUT_R, P._TRSM_NCUT_R + 1)
            push!(sh, (k, 40, a1, nothing), (40, k, a1, nothing))
        end
        for k in (P._TRMM_PACK_MIN + 1, P._TRMM_RPANEL + 1)
            push!(sh, (k, 9, (alpha(T),), nothing), (9, k, (alpha(T),), nothing))
        end
        push!(sh, (17, P._TRMM_RPACK + 1, (alpha(T),), nothing))
        wd = P._L1_WAY_D; m = P._TRSM_NCUT_R + 1
        push!(sh, (m, 40, a1, cld(m, wd) * wd))
        f = mt_floor()
        isnothing(f) || (s = max(mt_side(f, 1), P._NR + 1, P._MR * W + 1); push!(sh, (s, s, (alpha(T),), nothing)))
        return unique!(sh)
    end

    function symm_cases(rng, T, fp, shapes)
        cs = Case[]
        for (m, n, sides) in shapes, s in sides, u in "UL", b in (beta(T), zero(T))
            ka = s == 'L' ? m : n
            A = randn(rng, T, ka + PAD, ka); B = randn(rng, T, m + PAD, n)
            C0 = randn(rng, T, m + PAD, n); iszero(b) && (C0[1:m, :] .= T(NaN))
            push!(
                cs, Case(
                    "$T $s$u m=$m n=$n β=$b",
                    () -> (C = copy(C0); l3_symm(fp, s, u, m, n, alpha(T), A, ka + PAD, B, m + PAD, b, C, m + PAD); (C,)),
                    (C0,), rtol_for(T, ka)
                )
            )
        end
        return cs
    end

    # symm/hemm: the pack cuts `_CSYMM_PACK_CUT`/`_CHEMM_PACK_CUT` and the 3M window on the
    # triangle's order (side L only), and the pool floor.
    function symm_shapes(T)
        sh = Any[(m, n, "LR") for (m, n) in ((1, 1), (3, 5), (8, 9), (17, 33), (33, 17))]
        lo, kmin = P._CGEMM_3M_MIN, P._CGEMM_3M_KMIN
        for (m, n) in ((lo - 1, kmin), (lo, kmin), (lo, kmin - 1), (P._CGEMM_3M_MAX + 1, kmin))
            push!(sh, (m, n, "L"))
        end
        for c in unique!([P._CSYMM_PACK_CUT, P._CHEMM_PACK_CUT]), m in (c, c + 1)
            push!(sh, (m, 5, "L"))
        end
        f = mt_floor()
        isnothing(f) || (s = max(mt_side(f, 2), P._NR + 1); push!(sh, (s, s, "LR")))
        return sh
    end

    # syrk/herk and syr2k/her2k. Legal `trans`: N/T (complex syrk/syr2k), N/C (herk/her2k), and all
    # three for real types, where C means T. herk/her2k take a real β (and herk a real α). β = 0
    # with α ≠ 0 is a separate real syrk route.
    function rk_cases(rng, T, fp, shapes; herm::Bool, two::Bool)
        cs = Case[]
        flags = T <: Real ? "NTC" : herm ? "NC" : "NT"
        a = herm && !two ? real(alpha(T)) : alpha(T)
        b0 = herm ? real(beta(T)) : beta(T)
        for (n, k) in shapes, u in "UL", t in flags, b in (b0, zero(b0))
            ar, ac = t == 'N' ? (n, k) : (k, n)
            A = randn(rng, T, ar + PAD, ac); B = randn(rng, T, ar + PAD, ac)
            C0 = randn(rng, T, n + PAD, n)
            if iszero(b)
                for j in 1:n, i in (u == 'U' ? (1:j) : (j:n))
                    C0[i, j] = T(NaN)
                end
            end
            run = two ?
                () -> (C = copy(C0); l3_syr2k(fp, u, t, n, k, a, A, ar + PAD, B, ar + PAD, b, C, n + PAD); (C,)) :
                () -> (C = copy(C0); l3_syrk(fp, u, t, n, k, a, A, ar + PAD, b, C, n + PAD); (C,))
            push!(cs, Case("$T $u$t n=$n k=$k β=$b", run, (C0,), rtol_for(T, k)))
        end
        return cs
    end

    # The order n against the pack cuts (`_SYRK_PACK_CUT`, `_SYR2K_PACK_CUT`, `_CSYRK_PACK_CUT`,
    # `_CSYRK_PACK_CUT_T`, `_CSYR2K_PACK_CUT`), the unpacked and fused limits
    # (`_CSYRK_UNPACK_MAX`, `_SYRK_UNIFIED_MAX`, `_CSYR2K_FUSED_MAX`), the recursion base
    # `_SYRK_DBASE`, the two-pass cut `_SYR2K_2PASS`, the complex 3M window from `_CSYRK_3M_MIN`
    # with `_CGEMM_3M_KMIN`, k past `_KC`, and the pool floor.
    function rk_shapes(T)
        sh = Any[(n, k) for (n, k) in ((1, 1), (2, 3), (5, 1), (7, 9))]
        cuts = [
            P._SYRK_PACK_CUT, P._SYR2K_PACK_CUT, P._CSYRK_PACK_CUT, P._CSYRK_PACK_CUT_T,
            P._CSYR2K_PACK_CUT, P._CSYRK_UNPACK_MAX, P._SYRK_UNIFIED_MAX, P._CSYR2K_FUSED_MAX, P._SYRK_DBASE,
        ]
        P._SYR2K_2PASS < typemax(Int) && push!(cuts, P._SYR2K_2PASS)
        for c in unique!(filter(>(0), cuts)), n in (c, c + 1)
            push!(sh, (n, 9))
        end
        kmin = P._CGEMM_3M_KMIN
        push!(sh, (P._CSYRK_3M_MIN, kmin), (P._CSYRK_3M_MIN + 1, kmin), (P._CSYRK_3M_MIN + 1, kmin - 1))
        push!(sh, (P._SYRK_DBASE + 3, P._KC + 5))
        f = mt_floor()
        isnothing(f) || (s = max(mt_side(f, 2), P._NR + 1); push!(sh, (s, s)))
        return unique!(sh)
    end

    for T in TYPES
        p = pre(T); cx = T <: Complex
        reg!(3, p * "gemm_", rng -> gemm_cases(rng, T, fptr(p * "gemm_"), gemm_shapes(T)))
        for nm in (cx ? ("symm_", "hemm_") : ("symm_",))
            reg!(3, p * nm, rng -> symm_cases(rng, T, fptr(p * nm), symm_shapes(T)))
        end
        for (nm, herm, two) in (("syrk_", false, false), ("syr2k_", false, true))
            reg!(3, p * nm, rng -> rk_cases(rng, T, fptr(p * nm), rk_shapes(T); herm, two))
        end
        if cx
            for (nm, two) in (("herk_", false), ("her2k_", true))
                reg!(3, p * nm, rng -> rk_cases(rng, T, fptr(p * nm), rk_shapes(T); herm = true, two))
            end
        end
        for nm in ("trmm_", "trsm_")
            reg!(3, p * nm, rng -> trxm_cases(rng, T, fptr(p * nm), trxm_shapes(T)))
        end
    end

    function run_level(level)
        nts = unique!([1, Threads.nthreads()])
        for name in sort!([k for (k, (l, _)) in BUILDERS if l == level])
            cases = BUILDERS[name][2](Xoshiro(hash(name)))
            for nt in nts
                old = PureBLAS.get_num_threads(); oldfloor = PureBLAS._MT_FLOOR[]
                PureBLAS.set_num_threads(nt)
                try
                    @testset "$name, $(length(cases)) cases, $nt thread(s)" begin
                        @test isempty(conform(name, cases))
                    end
                finally
                    PureBLAS.set_num_threads(old)
                    PureBLAS._MT_FLOOR[] = oldfloor     # items that run later see the floor they would without this one
                end
            end
        end
    end
end

# Each level runs once with PureBLAS serial and, when Julia has more than one thread, once with
# every thread in the pool, so the threaded entries see C-ABI operands as well.
@testitem "C-ABI conformance: BLAS-1 through activate() matches the default BLAS" setup = [CabiConf] tags = [:lbt] begin
    CabiConf.run_level(1)
end

@testitem "C-ABI conformance: BLAS-2 through activate() matches the default BLAS" setup = [CabiConf] tags = [:lbt] begin
    CabiConf.run_level(2)
end

@testitem "C-ABI conformance: BLAS-3 through activate() matches the default BLAS" setup = [CabiConf] tags = [:lbt] begin
    CabiConf.run_level(3)
end

# Every symbol `activate()` forwards must be covered: a BLAS symbol by a builder above, a LAPACK
# symbol by an item in lbt_forward_tests.jl (`LAPACK_TESTED`) or by an entry in `LAPACK_UNTESTED`,
# which this item prints. A newly forwarded symbol in none of the three fails here.
@testitem "C-ABI conformance: every forwarded symbol has a case" setup = [CabiConf] tags = [:lbt] begin
    using PureBLAS
    fwd = Set(first.(PureBLAS._LBT_REGISTRARS))
    blas = Set(keys(CabiConf.BUILDERS))
    tested = Set(split(CabiConf.LAPACK_TESTED))
    untested = Set(split(CabiConf.LAPACK_UNTESTED))
    @test isempty(intersect(tested, untested))
    @test isempty(intersect(blas, union(tested, untested)))
    stale = setdiff(union(blas, tested, untested), fwd)
    isempty(stale) || @error "listed but not forwarded" sort!(collect(stale))
    @test isempty(stale)
    missing_ = setdiff(fwd, blas, tested, untested)
    isempty(missing_) || @error "forwarded with no case" sort!(collect(missing_))
    @test isempty(missing_)
    @info "C-ABI coverage: $(length(blas)) BLAS symbols with conformance cases, \
        $(length(tested)) LAPACK symbols exercised by lbt_forward_tests.jl, \
        $(length(untested)) LAPACK symbols forwarded with no test" untested = join(sort!(collect(untested)), " ")
end
