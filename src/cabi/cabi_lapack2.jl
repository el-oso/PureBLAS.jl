# Mode 1 C/Fortran-ABI boundary, batch 2 — OpenBLAS-removal ratchet follow-up. Same ABI conventions as
# cabi_lapack.jl (char args `Ptr{UInt8}` first + `_cabi_char`, then by-ref `Ptr` scalars/arrays, then
# TRAILING hidden Fortran string-length `Clong`s — one per char arg; `info::Ptr{Int64}` OUTPUT before the
# hidden lengths; `lwork==-1` workspace queries honored by reporting size 1 and returning). All new shims
# here are careful ABI wiring over kernels that ALREADY EXIST elsewhere in the package — no new numerics.
#
# Scope notes (each documented again at its shim): trevc/ormtr/unmtr/hseqr/gebrd/bdsqr etc. forward only
# the parameter subset their underlying PureBLAS kernel supports (mirrors the existing codebase's own
# precedent — e.g. cabi_lapack.jl's geev jobvl='V' rejection, trevc side='R'-only) — an unsupported
# combination returns a LAPACK-style negative `info` (illegal argument) rather than a silently-wrong
# result. `uplo='U'` for sytrd/hetrd/orgtr/ungtr/ormtr/unmtr is a genuinely different reflector layout
# (not a transpose of 'L') and is NOT implemented — only 'L' (matches `_sytd2_lower!`/`_hetrd!`).

# ── gesv: A·X=B via getrf!+getrs (composes exactly as getrf_64_/getrs_64_ do in cabi_lapack.jl) ────────
# `{d,c,z}gesv_64_(n, nrhs, A, lda, ipiv, B, ldb, info)` — 0 chars. `s` is mixed-precision (promote→F64
# getrf!+solve→demote), matching sgetrf_64_/sgeqrt_64_'s established pattern (no native F32 getrf!).
for (p, T) in (("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "gesv_64_"))(
            n::Ptr{Int64}, nrhs::Ptr{Int64},
            A::Ptr{$T}, lda::Ptr{Int64}, ipiv::Ptr{Int64}, B::Ptr{$T}, ldb::Ptr{Int64},
            info::Ptr{Int64}
        )::Cvoid
        N = Int(unsafe_load(n)); R = Int(unsafe_load(nrhs))
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        Bm = PtrMatrix(B, N, R, Int(unsafe_load(ldb)))
        ip = PtrVector(ipiv, N)
        _, _, inf = getrf!(Am, ip)
        if inf == 0
            _laswp!(Bm, ip, 1, N, 1, R)
            trsm!(Bm, Am; side = 'L', uplo = 'L', transA = 'N', diag = 'U', alpha = one($T))
            trsm!(Bm, Am; side = 'L', uplo = 'U', transA = 'N', diag = 'N', alpha = one($T))
        end
        unsafe_store!(info, Int64(inf)); return
    end
end
Base.@ccallable function sgesv_64_(
        n::Ptr{Int64}, nrhs::Ptr{Int64}, A::Ptr{Float32}, lda::Ptr{Int64},
        ipiv::Ptr{Int64}, B::Ptr{Float32}, ldb::Ptr{Int64}, info::Ptr{Int64}
    )::Cvoid
    N = Int(unsafe_load(n)); R = Int(unsafe_load(nrhs))
    Am = PtrMatrix(A, N, N, Int(unsafe_load(lda))); Bm = PtrMatrix(B, N, R, Int(unsafe_load(ldb)))
    Af = Matrix{Float64}(undef, N, N); _f32_to_f64!(Af, Am, N, N)
    Bf = Matrix{Float64}(undef, N, R)
    @inbounds for j in 1:R, i in 1:N
        Bf[i, j] = Float64(Bm[i, j])
    end
    ip = Vector{Int64}(undef, N)
    _, _, inf = getrf!(Af, ip)
    if inf == 0
        _laswp!(Bf, ip, 1, N, 1, R)
        trsm!(Bf, Af; side = 'L', uplo = 'L', transA = 'N', diag = 'U', alpha = 1.0)
        trsm!(Bf, Af; side = 'L', uplo = 'U', transA = 'N', diag = 'N', alpha = 1.0)
    end
    _f64_to_f32!(Am, Af, N, N)
    @inbounds for j in 1:R, i in 1:N
        Bm[i, j] = Float32(Bf[i, j])
    end
    ipv = PtrVector(ipiv, N); @inbounds for i in 1:N
        ipv[i] = ip[i]
    end
    unsafe_store!(info, Int64(inf)); return
end

# ── posv: A·X=B via potrf!+potrs (composes exactly as potrf_64_/potrs_64_ do) — potrf! is natively ─────
# generic over s/d/c/z (Float32-capable), so all 4 types are native, no mixed precision needed.
# `{s,d,c,z}posv_64_(uplo, n, nrhs, A, lda, B, ldb, info, len_uplo)` — 1 char.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "posv_64_"))(
            uplo::Ptr{UInt8}, n::Ptr{Int64},
            nrhs::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, B::Ptr{$T}, ldb::Ptr{Int64},
            info::Ptr{Int64}, lu::Clong
        )::Cvoid
        ul = _cabi_char(uplo)
        N = Int(unsafe_load(n)); R = Int(unsafe_load(nrhs))
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        Bm = PtrMatrix(B, N, R, Int(unsafe_load(ldb)))
        try
            potrf!(Am; uplo = ul)
            ct = $(T <: Complex ? 'C' : 'T')
            if ul == 'L'
                trsm!(Bm, Am; side = 'L', uplo = 'L', transA = 'N', alpha = one($T))
                trsm!(Bm, Am; side = 'L', uplo = 'L', transA = ct, alpha = one($T))
            else
                trsm!(Bm, Am; side = 'L', uplo = 'U', transA = ct, alpha = one($T))
                trsm!(Bm, Am; side = 'L', uplo = 'U', transA = 'N', alpha = one($T))
            end
            unsafe_store!(info, Int64(0))
        catch e
            e isa PosDefException || rethrow()
            unsafe_store!(info, Int64(e.info))
        end
        return
    end
end

# ── lacpy: copy the uplo triangle (or full) of A into B — no `info` arg (matches the real LAPACK ABI) ──
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "lacpy_64_"))(
            uplo::Ptr{UInt8}, m::Ptr{Int64},
            n::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, B::Ptr{$T}, ldb::Ptr{Int64}, lu::Clong
        )::Cvoid
        ul = _cabi_char(uplo)
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n))
        Am = PtrMatrix(A, M, N, Int(unsafe_load(lda)))
        Bm = PtrMatrix(B, M, N, Int(unsafe_load(ldb)))
        if ul == 'U'
            @inbounds for j in 1:N, i in 1:min(j, M)
                Bm[i, j] = Am[i, j]
            end
        elseif ul == 'L'
            @inbounds for j in 1:N, i in j:M
                Bm[i, j] = Am[i, j]
            end
        else
            @inbounds for j in 1:N, i in 1:M
                Bm[i, j] = Am[i, j]
            end
        end
        return
    end
end

# ── larfg: generate a Householder reflector (LAPACK dlarfg/zlarfg) via `_larfg!` (svd.jl) ──────────────
# `{s,d,c,z}larfg_64_(n, alpha, x, incx, tau)` — 0 chars. `alpha` is a SEPARATE by-ref scalar (distinct
# memory from `x`, per the real Fortran ABI — Base's own `larfg!` wrapper loses β because it never copies
# the mutated Ref back into `x[1]`; here we honor the real contract and write β into `alpha`). Round-trips
# through a small owned buffer since `_larfg!` wants one contiguous AbstractVector spanning α+tail, while
# the ABI hands them as two separate (possibly strided) pointers.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "larfg_64_"))(
            n::Ptr{Int64}, alpha::Ptr{$T}, x::Ptr{$T},
            incx::Ptr{Int64}, tau::Ptr{$T}
        )::Cvoid
        N = Int(unsafe_load(n))
        if N <= 1
            unsafe_store!(tau, zero($T)); return
        end
        ix = Int(unsafe_load(incx))
        v = Vector{$T}(undef, N)
        v[1] = unsafe_load(alpha)
        @inbounds for j in 2:N
            v[j] = unsafe_load(x, 1 + (j - 2) * ix)
        end
        β, τ = _larfg!(v)
        unsafe_store!(alpha, $T(β))
        @inbounds for j in 2:N
            unsafe_store!(x, v[j], 1 + (j - 2) * ix)
        end
        unsafe_store!(tau, τ)
        return
    end
end

# ── larf: apply a Householder reflector H=I−τ·v·vᴴ to C (LAPACK dlarf/zlarf) ────────────────────────────
# `{s,d,c,z}larf_64_(side, m, n, v, incv, tau, C, ldc, work, len_side)` — 1 char. side='L': `_house_left!`
# (svd.jl); side='R': `_larf_right!` (hessenberg.jl, generic T<:Number). Both ignore v[1] (implicit 1),
# matching how they're already used throughout the codebase (geqp3.jl, eigen.jl, hessenberg.jl).
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "larf_64_"))(
            side::Ptr{UInt8}, m::Ptr{Int64}, n::Ptr{Int64},
            v::Ptr{$T}, incv::Ptr{Int64}, tau::Ptr{$T}, C::Ptr{$T}, ldc::Ptr{Int64}, work::Ptr{$T},
            ls::Clong
        )::Cvoid
        sd = _cabi_char(side)
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); iv = Int(unsafe_load(incv))
        Cm = PtrMatrix(C, M, N, Int(unsafe_load(ldc)))
        τ = unsafe_load(tau)
        l = sd == 'L' ? M : N
        vv = Vector{$T}(undef, max(l, 1))
        @inbounds for j in 1:l
            vv[j] = unsafe_load(v, 1 + (j - 1) * iv)
        end
        if sd == 'L'
            _house_left!(Cm, vv, τ)
        else
            _larf_right!(Cm, vv, τ)
        end
        return
    end
end

# ── gebak: undo gebal's balancing on eigen/Schur vectors (LAPACK dgebak/zgebak) — direct wrap of ───────
# `gebak!` (geev.jl). `{s,d,c,z}gebak_64_(job, side, n, ilo, ihi, scale, m, V, ldv, info, len_job, len_side)`
# — 2 chars. `n`/`m` are V's row/column counts (the real Fortran ABI supports non-square V; PureBLAS's
# `gebak!` already handles arbitrary column count via `size(V,2)`).
for (p, T, Tr) in (
        ("s", Float32, Float32), ("d", Float64, Float64),
        ("c", ComplexF32, Float32), ("z", ComplexF64, Float64),
    )
    @eval Base.@ccallable function $(Symbol(p, "gebak_64_"))(
            job::Ptr{UInt8}, side::Ptr{UInt8},
            n::Ptr{Int64}, ilo::Ptr{Int64}, ihi::Ptr{Int64}, scale::Ptr{$Tr}, m::Ptr{Int64}, V::Ptr{$T},
            ldv::Ptr{Int64}, info::Ptr{Int64}, lj::Clong, ls::Clong
        )::Cvoid
        N = Int(unsafe_load(n)); M = Int(unsafe_load(m))
        Vm = PtrMatrix(V, N, M, Int(unsafe_load(ldv)))
        sc = PtrVector(scale, N)
        gebak!(_cabi_char(job), _cabi_char(side), Int(unsafe_load(ilo)), Int(unsafe_load(ihi)), sc, Vm)
        unsafe_store!(info, Int64(0)); return
    end
end

# ── hseqr: Schur factorization of an upper-Hessenberg matrix (LAPACK dhseqr/zhseqr) — direct wrap of ───
# `hseqr!` (hseqr.jl), already the engine behind geev/gees in cabi_lapack.jl. REAL ABI has separate wr/wi
# outputs (PureBLAS's real `hseqr!` wants one Complex-valued `w`, so we round-trip through a small owned
# buffer); COMPLEX ABI has one `w` output directly (`hseqr!`'s complex method writes into it in place).
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "hseqr_64_"))(
            job::Ptr{UInt8}, compz::Ptr{UInt8},
            n::Ptr{Int64}, ilo::Ptr{Int64}, ihi::Ptr{Int64}, H::Ptr{$T}, ldh::Ptr{Int64}, wr::Ptr{$T},
            wi::Ptr{$T}, Z::Ptr{$T}, ldz::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64},
            info::Ptr{Int64}, lj::Clong, lc::Clong
        )::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        N = Int(unsafe_load(n))
        Hm = PtrMatrix(H, N, N, Int(unsafe_load(ldh)))
        Zm = PtrMatrix(Z, N, N, Int(unsafe_load(ldz)))
        wtmp = Vector{Complex{$T}}(undef, N)
        inf = hseqr!(
            _cabi_char(job), _cabi_char(compz), Hm, Int(unsafe_load(ilo)), Int(unsafe_load(ihi)),
            wtmp, Zm
        )
        wrp = PtrVector(wr, N); wip = PtrVector(wi, N)
        @inbounds for i in 1:N
            wrp[i] = real(wtmp[i]); wip[i] = imag(wtmp[i])
        end
        unsafe_store!(info, Int64(inf)); return
    end
end
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "hseqr_64_"))(
            job::Ptr{UInt8}, compz::Ptr{UInt8},
            n::Ptr{Int64}, ilo::Ptr{Int64}, ihi::Ptr{Int64}, H::Ptr{$T}, ldh::Ptr{Int64}, w::Ptr{$T},
            Z::Ptr{$T}, ldz::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64},
            lj::Clong, lc::Clong
        )::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        N = Int(unsafe_load(n))
        Hm = PtrMatrix(H, N, N, Int(unsafe_load(ldh)))
        Zm = PtrMatrix(Z, N, N, Int(unsafe_load(ldz)))
        wm = PtrVector(w, N)
        inf = hseqr!(
            _cabi_char(job), _cabi_char(compz), Hm, Int(unsafe_load(ilo)), Int(unsafe_load(ihi)),
            wm, Zm
        )
        unsafe_store!(info, Int64(inf)); return
    end
end

# ── trevc: right eigenvectors of a Schur form by back-substitution (LAPACK dtrevc/ztrevc) — direct wrap ─
# of `trevc!` (trevc.jl). Only side='R' (right eigenvectors) and howmny∈{'A','B'} are implemented (matches
# `trevc!`'s own documented scope — left eigenvectors and the 'S' selected-subset mode are a follow-up);
# an unsupported side/howmny returns info=-1 (illegal argument) rather than a wrong result. `select`/`mm`
# are accepted (ABI positional) but unused (PureBLAS's `trevc!` always computes all n vectors); `m` is
# always set to n. `VL` is passed through (ignored) for signature compatibility, as `trevc!` documents.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "trevc_64_"))(
            side::Ptr{UInt8}, howmny::Ptr{UInt8},
            select::Ptr{Int64}, n::Ptr{Int64}, Tm_::Ptr{$T}, ldt::Ptr{Int64}, VL::Ptr{$T}, ldvl::Ptr{Int64},
            VR::Ptr{$T}, ldvr::Ptr{Int64}, mm::Ptr{Int64}, m::Ptr{Int64}, work::Ptr{$T}, info::Ptr{Int64},
            ls::Clong, lh::Clong
        )::Cvoid
        sd = _cabi_char(side); hn = _cabi_char(howmny)
        if sd != 'R' || (hn != 'A' && hn != 'B')
            unsafe_store!(info, Int64(-1)); return
        end
        N = Int(unsafe_load(n))
        Tmat = PtrMatrix(Tm_, N, N, Int(unsafe_load(ldt)))
        VLm = PtrMatrix(VL, N, N, max(Int(unsafe_load(ldvl)), 1))
        VRm = PtrMatrix(VR, N, N, Int(unsafe_load(ldvr)))
        trevc!('R', hn, Tmat, VLm, VRm)
        unsafe_store!(m, Int64(N)); unsafe_store!(info, Int64(0)); return
    end
end
for (p, T, R) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval Base.@ccallable function $(Symbol(p, "trevc_64_"))(
            side::Ptr{UInt8}, howmny::Ptr{UInt8},
            select::Ptr{Int64}, n::Ptr{Int64}, Tm_::Ptr{$T}, ldt::Ptr{Int64}, VL::Ptr{$T}, ldvl::Ptr{Int64},
            VR::Ptr{$T}, ldvr::Ptr{Int64}, mm::Ptr{Int64}, m::Ptr{Int64}, work::Ptr{$T}, rwork::Ptr{$R},
            info::Ptr{Int64}, ls::Clong, lh::Clong
        )::Cvoid
        sd = _cabi_char(side); hn = _cabi_char(howmny)
        if sd != 'R' || (hn != 'A' && hn != 'B')
            unsafe_store!(info, Int64(-1)); return
        end
        N = Int(unsafe_load(n))
        Tmat = PtrMatrix(Tm_, N, N, Int(unsafe_load(ldt)))
        VLm = PtrMatrix(VL, N, N, max(Int(unsafe_load(ldvl)), 1))
        VRm = PtrMatrix(VR, N, N, Int(unsafe_load(ldvr)))
        trevc!('R', hn, Tmat, VLm, VRm)
        unsafe_store!(m, Int64(N)); unsafe_store!(info, Int64(0)); return
    end
end

# ── sytrd / hetrd: reduce symmetric/Hermitian A to real tridiagonal form (LAPACK dsytrd/zhetrd) — direct
# wrap of `_sytd2_lower!`/`_hetrd!` (eigen.jl), the SAME kernels the dsyevr_/zheevr_ shims already route
# through internally. Only uplo='L' is implemented (see module header); uplo='U' → info=-1.
# `{s,d}sytrd_64_(uplo, n, A, lda, d, e, tau, work, lwork, info, len_uplo)` — 1 char.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "sytrd_64_"))(
            uplo::Ptr{UInt8}, n::Ptr{Int64}, A::Ptr{$T},
            lda::Ptr{Int64}, d::Ptr{$T}, e::Ptr{$T}, tau::Ptr{$T}, work::Ptr{$T}, lwork::Ptr{Int64},
            info::Ptr{Int64}, lu::Clong
        )::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        if _cabi_char(uplo) != 'L'
            unsafe_store!(info, Int64(-1)); return
        end
        N = Int(unsafe_load(n))
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        dm = PtrVector(d, N); em = PtrVector(e, max(N - 1, 0)); tm = PtrVector(tau, max(N - 1, 0))
        _sytrd_lower!(Am, dm, em, tm)   # blocked (dlatrd + syr2k)
        unsafe_store!(info, Int64(0)); return
    end
end
# `{c,z}hetrd_64_(uplo, n, A, lda, d, e, tau, work, lwork, info, len_uplo)` — d/e are REAL ($R).
for (p, T, R) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval Base.@ccallable function $(Symbol(p, "hetrd_64_"))(
            uplo::Ptr{UInt8}, n::Ptr{Int64}, A::Ptr{$T},
            lda::Ptr{Int64}, d::Ptr{$R}, e::Ptr{$R}, tau::Ptr{$T}, work::Ptr{$T}, lwork::Ptr{Int64},
            info::Ptr{Int64}, lu::Clong
        )::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        if _cabi_char(uplo) != 'L'
            unsafe_store!(info, Int64(-1)); return
        end
        N = Int(unsafe_load(n))
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        dm = PtrVector(d, N); em = PtrVector(e, max(N - 1, 0)); tm = PtrVector(tau, max(N - 1, 0))
        _hetrd!(Am, dm, em, tm)
        unsafe_store!(info, Int64(0)); return
    end
end

# ── orgtr / ungtr: form Q from sytrd/hetrd reflectors (LAPACK dorgtr/zungtr) — direct wrap of `orgtr!`/
# `ungtr!` (eigen.jl). Only uplo='L'; uplo='U' → info=-1. Overwrites A with Q (LAPACK contract; the
# kernel returns a fresh Q, copied back here). `{s,d}orgtr_64_(uplo, n, A, lda, tau, work, lwork, info,
# len_uplo)` — 1 char.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "orgtr_64_"))(
            uplo::Ptr{UInt8}, n::Ptr{Int64}, A::Ptr{$T},
            lda::Ptr{Int64}, tau::Ptr{$T}, work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64},
            lu::Clong
        )::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        if _cabi_char(uplo) != 'L'
            unsafe_store!(info, Int64(-1)); return
        end
        N = Int(unsafe_load(n))
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        tm = PtrVector(tau, max(N - 1, 0))
        Q = orgtr!('L', Am, tm)
        @inbounds for j in 1:N, i in 1:N
            Am[i, j] = Q[i, j]
        end
        unsafe_store!(info, Int64(0)); return
    end
end
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "ungtr_64_"))(
            uplo::Ptr{UInt8}, n::Ptr{Int64}, A::Ptr{$T},
            lda::Ptr{Int64}, tau::Ptr{$T}, work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64},
            lu::Clong
        )::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        if _cabi_char(uplo) != 'L'
            unsafe_store!(info, Int64(-1)); return
        end
        N = Int(unsafe_load(n))
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        tm = PtrVector(tau, max(N - 1, 0))
        Q = ungtr!('L', Am, tm)
        @inbounds for j in 1:N, i in 1:N
            Am[i, j] = Q[i, j]
        end
        unsafe_store!(info, Int64(0)); return
    end
end

# ── ormtr / unmtr: apply Q from sytrd/hetrd reflectors (LAPACK dormtr/zunmtr) — direct wrap of `_ormtr!`/
# `_unmtr!` (eigen.jl). Only side='L' (matches those kernels' own restriction) and uplo='L'; anything else
# → info=-1. `{s,d}ormtr_64_(side, uplo, trans, mC, nC, A, lda, tau, C, ldc, work, lwork, info, +3 lens)`.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "ormtr_64_"))(
            side::Ptr{UInt8}, uplo::Ptr{UInt8},
            trans::Ptr{UInt8}, mC::Ptr{Int64}, nC::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, tau::Ptr{$T},
            C::Ptr{$T}, ldc::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64},
            ls::Clong, lu::Clong, lt::Clong
        )::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        sd = _cabi_char(side); ul = _cabi_char(uplo); tr = _cabi_char(trans)
        if sd != 'L' || ul != 'L'
            unsafe_store!(info, Int64(-1)); return
        end
        M = Int(unsafe_load(mC)); Ncol = Int(unsafe_load(nC))
        Am = PtrMatrix(A, M, M, Int(unsafe_load(lda)))
        tm = PtrVector(tau, max(M - 1, 0))
        Cm = PtrMatrix(C, M, Ncol, Int(unsafe_load(ldc)))
        _ormtr!(Am, tm, Cm; side = 'L', trans = tr)
        unsafe_store!(info, Int64(0)); return
    end
end
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "unmtr_64_"))(
            side::Ptr{UInt8}, uplo::Ptr{UInt8},
            trans::Ptr{UInt8}, mC::Ptr{Int64}, nC::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, tau::Ptr{$T},
            C::Ptr{$T}, ldc::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64},
            ls::Clong, lu::Clong, lt::Clong
        )::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        sd = _cabi_char(side); ul = _cabi_char(uplo); tr = _cabi_char(trans)
        if sd != 'L' || ul != 'L'
            unsafe_store!(info, Int64(-1)); return
        end
        M = Int(unsafe_load(mC)); Ncol = Int(unsafe_load(nC))
        Am = PtrMatrix(A, M, M, Int(unsafe_load(lda)))
        tm = PtrVector(tau, max(M - 1, 0))
        Cm = PtrMatrix(C, M, Ncol, Int(unsafe_load(ldc)))
        _unmtr!(Am, tm, Cm; side = 'L', trans = tr)
        unsafe_store!(info, Int64(0)); return
    end
end

# ── orgqr / ungqr: form Q from geqrf reflectors (LAPACK dorgqr/zungqr) — direct wrap of `orgqr!`/`ungqr!`
# (qr.jl). `{s,d,c,z}orgqr_64_(m, n, k, A, lda, tau, work, lwork, info)` — 0 chars.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "orgqr_64_"))(
            m::Ptr{Int64}, n::Ptr{Int64}, k::Ptr{Int64},
            A::Ptr{$T}, lda::Ptr{Int64}, tau::Ptr{$T}, work::Ptr{$T}, lwork::Ptr{Int64},
            info::Ptr{Int64}
        )::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); K = Int(unsafe_load(k))
        Am = PtrMatrix(A, M, N, Int(unsafe_load(lda)))
        tm = PtrVector(tau, K)
        orgqr!(Am, tm, K)
        unsafe_store!(info, Int64(0)); return
    end
end
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "ungqr_64_"))(
            m::Ptr{Int64}, n::Ptr{Int64}, k::Ptr{Int64},
            A::Ptr{$T}, lda::Ptr{Int64}, tau::Ptr{$T}, work::Ptr{$T}, lwork::Ptr{Int64},
            info::Ptr{Int64}
        )::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); K = Int(unsafe_load(k))
        Am = PtrMatrix(A, M, N, Int(unsafe_load(lda)))
        tm = PtrVector(tau, K)
        ungqr!(Am, tm, K)
        unsafe_store!(info, Int64(0)); return
    end
end

# ── ormqr / unmqr: apply Q from geqrf reflectors (LAPACK dormqr/zunmqr) — direct wrap of `ormqr!`/
# `unmqr!` (qr.jl). `{s,d,c,z}ormqr_64_(side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info,
# +2 lens)` — 2 chars. Q's order (nq) = A's row count = M (side='L') or N (side='R').
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "ormqr_64_"))(
            side::Ptr{UInt8}, trans::Ptr{UInt8},
            m::Ptr{Int64}, n::Ptr{Int64}, k::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, tau::Ptr{$T},
            C::Ptr{$T}, ldc::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64},
            ls::Clong, lt::Clong
        )::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        sd = _cabi_char(side); tr = _cabi_char(trans)
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); K = Int(unsafe_load(k))
        mA = sd == 'L' ? M : N
        Am = PtrMatrix(A, mA, K, Int(unsafe_load(lda)))
        tm = PtrVector(tau, K)
        Cm = PtrMatrix(C, M, N, Int(unsafe_load(ldc)))
        ormqr!(sd, tr, Am, tm, Cm)
        unsafe_store!(info, Int64(0)); return
    end
end
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "unmqr_64_"))(
            side::Ptr{UInt8}, trans::Ptr{UInt8},
            m::Ptr{Int64}, n::Ptr{Int64}, k::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, tau::Ptr{$T},
            C::Ptr{$T}, ldc::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64}, info::Ptr{Int64},
            ls::Clong, lt::Clong
        )::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        sd = _cabi_char(side); tr = _cabi_char(trans)
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n)); K = Int(unsafe_load(k))
        mA = sd == 'L' ? M : N
        Am = PtrMatrix(A, mA, K, Int(unsafe_load(lda)))
        tm = PtrVector(tau, K)
        Cm = PtrMatrix(C, M, N, Int(unsafe_load(ldc)))
        unmqr!(sd, tr, Am, tm, Cm)
        unsafe_store!(info, Int64(0)); return
    end
end

# ── ormhr / unmhr: apply Q from gehrd reflectors (LAPACK dormhr/zunmhr) — direct wrap of `ormhr!`/
# `unmhr!` (hessenberg.jl). `{s,d,c,z}ormhr_64_(side, trans, mC, nC, ilo, ihi, A, lda, tau, C, ldc, work,
# lwork, info, +2 lens)` — 2 chars.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "ormhr_64_"))(
            side::Ptr{UInt8}, trans::Ptr{UInt8},
            mC::Ptr{Int64}, nC::Ptr{Int64}, ilo::Ptr{Int64}, ihi::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64},
            tau::Ptr{$T}, C::Ptr{$T}, ldc::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64},
            info::Ptr{Int64}, ls::Clong, lt::Clong
        )::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        sd = _cabi_char(side); tr = _cabi_char(trans)
        M = Int(unsafe_load(mC)); Ncol = Int(unsafe_load(nC))
        n = sd == 'L' ? M : Ncol
        Am = PtrMatrix(A, n, n, Int(unsafe_load(lda)))
        tm = PtrVector(tau, max(n - 1, 0))
        Cm = PtrMatrix(C, M, Ncol, Int(unsafe_load(ldc)))
        ormhr!(sd, tr, Int(unsafe_load(ilo)), Int(unsafe_load(ihi)), Am, tm, Cm)
        unsafe_store!(info, Int64(0)); return
    end
end
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "unmhr_64_"))(
            side::Ptr{UInt8}, trans::Ptr{UInt8},
            mC::Ptr{Int64}, nC::Ptr{Int64}, ilo::Ptr{Int64}, ihi::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64},
            tau::Ptr{$T}, C::Ptr{$T}, ldc::Ptr{Int64}, work::Ptr{$T}, lwork::Ptr{Int64},
            info::Ptr{Int64}, ls::Clong, lt::Clong
        )::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        sd = _cabi_char(side); tr = _cabi_char(trans)
        M = Int(unsafe_load(mC)); Ncol = Int(unsafe_load(nC))
        n = sd == 'L' ? M : Ncol
        Am = PtrMatrix(A, n, n, Int(unsafe_load(lda)))
        tm = PtrVector(tau, max(n - 1, 0))
        Cm = PtrMatrix(C, M, Ncol, Int(unsafe_load(ldc)))
        unmhr!(sd, tr, Int(unsafe_load(ilo)), Int(unsafe_load(ihi)), Am, tm, Cm)
        unsafe_store!(info, Int64(0)); return
    end
end

# ── gebrd: bidiagonalize A (LAPACK dgebrd/zgebrd) — direct wrap of `gebrd!` (svd.jl), the SAME kernel
# gesvd/gesdd already route through internally. Only m≥n is implemented (matches `gebrd!`'s own
# restriction — the m<n case is a genuinely different LOWER-bidiagonal reduction, not a transpose of
# this one); m<n → info=-1. `s` is mixed-precision (promote→F64→demote); `c`/`z` are native (`gebrd!` has
# a T<:Complex,R<:Real method already). `{s,d}gebrd_64_(m, n, A, lda, d, e, tauq, taup, work, lwork,
# info)` — 0 chars.
Base.@ccallable function dgebrd_64_(
        m::Ptr{Int64}, n::Ptr{Int64}, A::Ptr{Float64}, lda::Ptr{Int64},
        d::Ptr{Float64}, e::Ptr{Float64}, tauq::Ptr{Float64}, taup::Ptr{Float64}, work::Ptr{Float64},
        lwork::Ptr{Int64}, info::Ptr{Int64}
    )::Cvoid
    if unsafe_load(lwork) == Int64(-1)
        unsafe_store!(work, 1.0); unsafe_store!(info, Int64(0)); return
    end
    M = Int(unsafe_load(m)); N = Int(unsafe_load(n))
    if M < N
        unsafe_store!(info, Int64(-1)); return
    end
    k = min(M, N)
    # d,e,tauq,taup are ALL length k (the caller — LAPACK's gebrd! wrapper — allocates them so, and the
    # gebd2! tail writes taup[k]/d[k]; sizing e/taup at k-1 overruns).
    Am = PtrMatrix(A, M, N, Int(unsafe_load(lda)))
    dm = PtrVector(d, k); em = PtrVector(e, k)
    tqm = PtrVector(tauq, k); tpm = PtrVector(taup, k)
    ws = _svdws()
    _svd_grow_bidiag!(ws, M, N)
    gebrd!(Am, dm, em, tqm, tpm, ws)
    unsafe_store!(info, Int64(0)); return
end
Base.@ccallable function sgebrd_64_(
        m::Ptr{Int64}, n::Ptr{Int64}, A::Ptr{Float32}, lda::Ptr{Int64},
        d::Ptr{Float32}, e::Ptr{Float32}, tauq::Ptr{Float32}, taup::Ptr{Float32}, work::Ptr{Float32},
        lwork::Ptr{Int64}, info::Ptr{Int64}
    )::Cvoid
    if unsafe_load(lwork) == Int64(-1)
        unsafe_store!(work, 1.0f0); unsafe_store!(info, Int64(0)); return
    end
    M = Int(unsafe_load(m)); N = Int(unsafe_load(n))
    if M < N
        unsafe_store!(info, Int64(-1)); return
    end
    k = min(M, N)
    Am = PtrMatrix(A, M, N, Int(unsafe_load(lda)))
    Af = Matrix{Float64}(undef, M, N); _f32_to_f64!(Af, Am, M, N)
    df = Vector{Float64}(undef, k); ef = Vector{Float64}(undef, k)
    tqf = Vector{Float64}(undef, k); tpf = Vector{Float64}(undef, k)
    ws = _svdws()
    _svd_grow_bidiag!(ws, M, N)
    gebrd!(Af, df, ef, tqf, tpf, ws)
    _f64_to_f32!(Am, Af, M, N)
    dm = PtrVector(d, k); em = PtrVector(e, k)
    tqm = PtrVector(tauq, k); tpm = PtrVector(taup, k)
    @inbounds for i in 1:k
        dm[i] = Float32(df[i]); em[i] = Float32(ef[i]); tqm[i] = Float32(tqf[i]); tpm[i] = Float32(tpf[i])
    end
    unsafe_store!(info, Int64(0)); return
end
for (p, T, R) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval Base.@ccallable function $(Symbol(p, "gebrd_64_"))(
            m::Ptr{Int64}, n::Ptr{Int64}, A::Ptr{$T},
            lda::Ptr{Int64}, d::Ptr{$R}, e::Ptr{$R}, tauq::Ptr{$T}, taup::Ptr{$T}, work::Ptr{$T},
            lwork::Ptr{Int64}, info::Ptr{Int64}
        )::Cvoid
        if unsafe_load(lwork) == Int64(-1)
            unsafe_store!(work, one($T)); unsafe_store!(info, Int64(0)); return
        end
        M = Int(unsafe_load(m)); N = Int(unsafe_load(n))
        if M < N
            unsafe_store!(info, Int64(-1)); return
        end
        k = min(M, N)
        Am = PtrMatrix(A, M, N, Int(unsafe_load(lda)))
        dm = PtrVector(d, k); em = PtrVector(e, k)
        tqm = PtrVector(tauq, k); tpm = PtrVector(taup, k)
        ws = _svdws($T)
        _svd_grow_bidiag!(ws, M, N)
        gebrd!(Am, dm, em, tqm, tpm, ws)
        unsafe_store!(info, Int64(0)); return
    end
end

# ── bdsqr: implicit-shift QR SVD of a bidiagonal matrix (LAPACK dbdsqr/sbdsqr) — REAL ONLY (complex
# cbdsqr_/zbdsqr_ need a complex-accumulator Givens apply that doesn't exist yet; left as a documented
# gap, see lbt_forward_tests.jl / task notes). Built on `bdsqr!` (svd.jl, Float64, ALWAYS treats `e` as
# the SUPERdiagonal). `ncc` (the extra `C` operand, `C:=Qᴴ·C`) is NOT supported — ncc≠0 → info=-1
# (callers needing it should use the direct bdsqr!(d,e,U,Vt) 3-matrix form with ncc=0, which is what
# Julia's own SVD path and this shim both exercise).
#
# `uplo='L'` (e = SUBdiagonal): B's SVD comes from Bᵀ's (same d,e, e now Bᵀ's superdiagonal) —
# B=U·Σ·Vᵀ ⟺ Bᵀ=V·Σ·Uᵀ, so B's LEFT vectors accumulate where Bᵀ's RIGHT-vector slot would and vice
# versa — SWAP which scratch (direct-U vs transposed-Vt) feeds which argument position of the
# (always-"upper") `bdsqr!` core call. `Vt` (ABI, n×ncvt, rows=n) holds VT := Pᵀ·VT (a ROW-rotation);
# `bdsqr!`'s own accumulator is COLUMN-rotate, so `Vt` is transposed in/out of a scratch. `U` (ABI,
# nru×n) needs no transpose (`U:=U·Q` is already a natural column-rotation, matching `bdsqr!` directly).
@inline function _bdsqr_wrap!(uplo::Char, d::AbstractVector{Float64}, e::AbstractVector{Float64}, Uacc, Vtacc)
    n = length(d)
    Vscr = isnothing(Vtacc) ? nothing : Matrix{Float64}(undef, size(Vtacc, 2), n)
    if !isnothing(Vscr)
        @inbounds for i in 1:n, j in 1:size(Vtacc, 2)
            Vscr[j, i] = Vtacc[i, j]
        end
    end
    if uplo == 'U'
        bdsqr!(d, e, Uacc, Vscr)
    else
        bdsqr!(d, e, Vscr, Uacc)
    end
    if !isnothing(Vscr)
        @inbounds for i in 1:n, j in 1:size(Vtacc, 2)
            Vtacc[i, j] = Vscr[j, i]
        end
    end
    return
end
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "bdsqr_64_"))(
            uplo::Ptr{UInt8}, n::Ptr{Int64},
            ncvt::Ptr{Int64}, nru::Ptr{Int64}, ncc::Ptr{Int64}, d::Ptr{$T}, e::Ptr{$T}, Vt::Ptr{$T},
            ldvt::Ptr{Int64}, U::Ptr{$T}, ldu::Ptr{Int64}, C::Ptr{$T}, ldc::Ptr{Int64}, work::Ptr{$T},
            info::Ptr{Int64}, lu::Clong
        )::Cvoid
        if Int(unsafe_load(ncc)) != 0
            unsafe_store!(info, Int64(-5)); return
        end
        ul = _cabi_char(uplo)
        N = Int(unsafe_load(n)); NCVT = Int(unsafe_load(ncvt)); NRU = Int(unsafe_load(nru))
        dm = PtrVector(d, N); em = PtrVector(e, max(N - 1, 0))
        df = Vector{Float64}(undef, N); ef = Vector{Float64}(undef, max(N - 1, 0))
        @inbounds for i in 1:N
            df[i] = Float64(dm[i])
        end
        @inbounds for i in 1:max(N - 1, 0)
            ef[i] = Float64(em[i])
        end
        Uacc = NRU > 0 ? Matrix{Float64}(undef, NRU, N) : nothing
        Vtf = NCVT > 0 ? Matrix{Float64}(undef, N, NCVT) : nothing
        Um0 = NRU > 0 ? PtrMatrix(U, NRU, N, Int(unsafe_load(ldu))) : nothing
        Vtm0 = NCVT > 0 ? PtrMatrix(Vt, N, NCVT, Int(unsafe_load(ldvt))) : nothing
        NRU > 0 && (
            @inbounds for j in 1:N, i in 1:NRU
                Uacc[i, j] = Float64(Um0[i, j])
            end
        )
        NCVT > 0 && (
            @inbounds for j in 1:NCVT, i in 1:N
                Vtf[i, j] = Float64(Vtm0[i, j])
            end
        )
        _bdsqr_wrap!(ul, df, ef, Uacc, Vtf)
        NRU > 0 && (
            @inbounds for j in 1:N, i in 1:NRU
                Um0[i, j] = $T(Uacc[i, j])
            end
        )
        NCVT > 0 && (
            @inbounds for j in 1:NCVT, i in 1:N
                Vtm0[i, j] = $T(Vtf[i, j])
            end
        )
        @inbounds for i in 1:N
            dm[i] = $T(df[i])
        end
        unsafe_store!(info, Int64(0)); return
    end
end

# ── bdsdc: divide-and-conquer bidiagonal SVD (LAPACK dbdsdc/sbdsdc, REAL only — no complex variant in
# reference LAPACK). Only compq∈{'N' values-only,'I' full vectors} — compq='P' (compact packed form) is
# barely-supported even in Base's OWN wrapper (`@warn "COMPQ='P' is not tested"`); unsupported → info=-2.
# Built on the SAME `_bdsqr_wrap!` core as bdsqr_64_ (a different — but equally valid — SVD algorithm
# than LAPACK's true D&C; singular values/vectors match to numerical tolerance, which is all any SVD
# consumer can rely on). `q`/`iq` (compq='P' packed outputs) are unused.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "bdsdc_64_"))(
            uplo::Ptr{UInt8}, compq::Ptr{UInt8},
            n::Ptr{Int64}, d::Ptr{$T}, e::Ptr{$T}, u::Ptr{$T}, ldu::Ptr{Int64}, vt::Ptr{$T},
            ldvt::Ptr{Int64}, q::Ptr{$T}, iq::Ptr{Int64}, work::Ptr{$T}, iwork::Ptr{Int64},
            info::Ptr{Int64}, lu::Clong, lc::Clong
        )::Cvoid
        ul = _cabi_char(uplo); cq = _cabi_char(compq)
        if cq != 'N' && cq != 'I'
            unsafe_store!(info, Int64(-2)); return
        end
        N = Int(unsafe_load(n))
        dm = PtrVector(d, N); em = PtrVector(e, max(N - 1, 0))
        df = Vector{Float64}(undef, N); ef = Vector{Float64}(undef, max(N - 1, 0))
        @inbounds for i in 1:N
            df[i] = Float64(dm[i])
        end
        @inbounds for i in 1:max(N - 1, 0)
            ef[i] = Float64(em[i])
        end
        if cq == 'N'
            _bdsqr_wrap!(ul, df, ef, nothing, nothing)
        else
            Uacc = zeros(Float64, N, N); Vtacc = zeros(Float64, N, N)
            @inbounds for i in 1:N
                Uacc[i, i] = 1.0; Vtacc[i, i] = 1.0
            end
            _bdsqr_wrap!(ul, df, ef, Uacc, Vtacc)
            um = PtrMatrix(u, N, N, Int(unsafe_load(ldu)))
            vtm = PtrMatrix(vt, N, N, Int(unsafe_load(ldvt)))
            @inbounds for j in 1:N, i in 1:N
                um[i, j] = $T(Uacc[i, j]); vtm[i, j] = $T(Vtacc[i, j])
            end
        end
        @inbounds for i in 1:N
            dm[i] = $T(df[i])
        end
        unsafe_store!(info, Int64(0)); return
    end
end
