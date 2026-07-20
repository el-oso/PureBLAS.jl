# Mode 1 C-ABI, batch 3: build-required LAPACK routines PureBLAS implements natively — syconv (syconv.jl)
# + trrfs (trrfs.jl). Shims forward the @ccallable _64_ symbols to those kernels; registered in
# cabi_forward.jl. Sigs cross-checked vs LinearAlgebra/src/lapack.jl. (tgsen + complex/F32 ggsvd follow in
# a later batch once their ABIs are wired — the kernels exist but are not yet forwarded.)

# ── syconv: convert a Bunch-Kaufman factorization's 2×2 D off-diagonals into `work` (LAPACK d/z/csyconv,
# WAY='C' — the only mode Julia's wrapper drives). `syconv!(uplo, A, ipiv) -> (A, work)`. All 4 types
# (the factorization is symmetric — sytrf — for real AND complex; no Hermitian syconv exists).
# `{s,d,c,z}syconv_64_(uplo, way, n, A, lda, ipiv, work, info, len_uplo, len_way)` — 2 chars.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "syconv_64_"))(uplo::Ptr{UInt8}, way::Ptr{UInt8},
            n::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, ipiv::Ptr{Int64}, work::Ptr{$T},
            info::Ptr{Int64}, lu::Clong, lw::Clong)::Cvoid
        N = Int(unsafe_load(n))
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        ip = PtrVector(ipiv, N)
        _, wk = syconv!(_cabi_char(uplo), Am, ip)          # kernel does WAY='C'; `way` arg is always 'C' here
        wv = PtrVector(work, N)
        @inbounds for i in 1:N; wv[i] = wk[i]; end          # copy the extracted off-diagonals to the C buffer
        unsafe_store!(info, Int64(0)); return
    end
end

# ── trrfs: triangular-solve error bounds (LAPACK d/z/c/strrfs). `trrfs!(uplo,trans,diag,A,B,X,Ferr,Berr)`.
# REAL: `{s,d}trrfs_64_(uplo,trans,diag,n,nrhs,A,lda,B,ldb,X,ldx,Ferr,Berr,work,iwork,info, 3 lens)`.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "trrfs_64_"))(uplo::Ptr{UInt8}, trans::Ptr{UInt8},
            diag::Ptr{UInt8}, n::Ptr{Int64}, nrhs::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64},
            B::Ptr{$T}, ldb::Ptr{Int64}, X::Ptr{$T}, ldx::Ptr{Int64}, Ferr::Ptr{$T}, Berr::Ptr{$T},
            work::Ptr{$T}, iwork::Ptr{Int64}, info::Ptr{Int64}, lu::Clong, lt::Clong, ld::Clong)::Cvoid
        N = Int(unsafe_load(n)); R = Int(unsafe_load(nrhs))
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        Bm = PtrMatrix(B, N, R, Int(unsafe_load(ldb)))
        Xm = PtrMatrix(X, N, R, Int(unsafe_load(ldx)))
        trrfs!(_cabi_char(uplo), _cabi_char(trans), _cabi_char(diag), Am, Bm, Xm,
            PtrVector(Ferr, R), PtrVector(Berr, R))
        unsafe_store!(info, Int64(0)); return
    end
end
# COMPLEX: Ferr/Berr are REAL; `rwork` (real) replaces `iwork`.
for (p, T, Tr) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval Base.@ccallable function $(Symbol(p, "trrfs_64_"))(uplo::Ptr{UInt8}, trans::Ptr{UInt8},
            diag::Ptr{UInt8}, n::Ptr{Int64}, nrhs::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64},
            B::Ptr{$T}, ldb::Ptr{Int64}, X::Ptr{$T}, ldx::Ptr{Int64}, Ferr::Ptr{$Tr}, Berr::Ptr{$Tr},
            work::Ptr{$T}, rwork::Ptr{$Tr}, info::Ptr{Int64}, lu::Clong, lt::Clong, ld::Clong)::Cvoid
        N = Int(unsafe_load(n)); R = Int(unsafe_load(nrhs))
        Am = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        Bm = PtrMatrix(B, N, R, Int(unsafe_load(ldb)))
        Xm = PtrMatrix(X, N, R, Int(unsafe_load(ldx)))
        trrfs!(_cabi_char(uplo), _cabi_char(trans), _cabi_char(diag), Am, Bm, Xm,
            PtrVector(Ferr, R), PtrVector(Berr, R))
        unsafe_store!(info, Int64(0)); return
    end
end

# ── tgsen (COMPLEX ONLY): reorder a generalized Schur pair so selected eigenvalues lead. ijob/wantq/wantz
# are INTEGER args (no chars → no Clong). Julia calls twice (lwork=-1 query then real); our kernel self-
# allocates so the query reports minimal sizes. Complex ztgsen/ctgsen is COMPLETE (no 2×2 blocks) — the
# real dtgsen/stgsen path is NOT forwarded (2×2 conjugate-pair swap unbuilt; would throw). pl/pr/dif are
# ijob=0-unused (Ptr{Cvoid}). `_ztgsen!` mutates A/B/Q/Z in place → PtrMatrix writes reach the caller.
# `ztgsen_64_(ijob,wantq,wantz,select,n,A,lda,B,ldb,alpha,beta,Q,ldq,Z,ldz,m,pl,pr,dif,work,lwork,iwork,liwork,info)`.
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "tgsen_64_"))(ijob::Ptr{Int64}, wantq::Ptr{Int64},
            wantz::Ptr{Int64}, select::Ptr{Int64}, n::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64},
            B::Ptr{$T}, ldb::Ptr{Int64}, alpha::Ptr{$T}, beta::Ptr{$T}, Q::Ptr{$T}, ldq::Ptr{Int64},
            Z::Ptr{$T}, ldz::Ptr{Int64}, m::Ptr{Int64}, pl::Ptr{Cvoid}, pr::Ptr{Cvoid}, dif::Ptr{Cvoid},
            work::Ptr{$T}, lwork::Ptr{Int64}, iwork::Ptr{Int64}, liwork::Ptr{Int64},
            info::Ptr{Int64})::Cvoid
        N = Int(unsafe_load(n))
        if Int(unsafe_load(lwork)) < 0                       # workspace query — kernel self-allocates
            unsafe_store!(work, $T(1)); unsafe_store!(iwork, Int64(1))
            unsafe_store!(info, Int64(0)); return
        end
        S = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        Tm = PtrMatrix(B, N, N, Int(unsafe_load(ldb)))
        Qm = PtrMatrix(Q, N, N, Int(unsafe_load(ldq)))
        Zm = PtrMatrix(Z, N, N, Int(unsafe_load(ldz)))
        sel = PtrVector(select, N)
        _, _, al, be, _, _ = tgsen!(sel, S, Tm, Qm, Zm)
        av = PtrVector(alpha, N); bv = PtrVector(beta, N)
        cnt = 0
        @inbounds for i in 1:N
            av[i] = al[i]; bv[i] = be[i]
            (sel[i] != 0) && (cnt += 1)
        end
        unsafe_store!(m, Int64(cnt)); unsafe_store!(info, Int64(0)); return
    end
end
# REAL tgsen (dtgsen/stgsen) — now complete via _dtgex2_big! (2×2 conjugate-pair swaps). Separate
# alphar/alphai/beta (real) vs complex's alpha/beta. `tgsen!` real returns alpha::Complex, beta::real.
# `{d,s}tgsen_64_(ijob,wantq,wantz,select,n,A,lda,B,ldb,alphar,alphai,beta,Q,ldq,Z,ldz,m,pl,pr,dif,work,lwork,iwork,liwork,info)`.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "tgsen_64_"))(ijob::Ptr{Int64}, wantq::Ptr{Int64},
            wantz::Ptr{Int64}, select::Ptr{Int64}, n::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64},
            B::Ptr{$T}, ldb::Ptr{Int64}, alphar::Ptr{$T}, alphai::Ptr{$T}, beta::Ptr{$T},
            Q::Ptr{$T}, ldq::Ptr{Int64}, Z::Ptr{$T}, ldz::Ptr{Int64}, m::Ptr{Int64}, pl::Ptr{Cvoid},
            pr::Ptr{Cvoid}, dif::Ptr{Cvoid}, work::Ptr{$T}, lwork::Ptr{Int64}, iwork::Ptr{Int64},
            liwork::Ptr{Int64}, info::Ptr{Int64})::Cvoid
        N = Int(unsafe_load(n))
        if Int(unsafe_load(lwork)) < 0
            unsafe_store!(work, $T(1)); unsafe_store!(iwork, Int64(1))
            unsafe_store!(info, Int64(0)); return
        end
        S = PtrMatrix(A, N, N, Int(unsafe_load(lda)))
        Tm = PtrMatrix(B, N, N, Int(unsafe_load(ldb)))
        Qm = PtrMatrix(Q, N, N, Int(unsafe_load(ldq)))
        Zm = PtrMatrix(Z, N, N, Int(unsafe_load(ldz)))
        sel = PtrVector(select, N)
        _, _, al, be, _, _ = tgsen!(sel, S, Tm, Qm, Zm)   # al::Complex, be::real
        ar = PtrVector(alphar, N); ai = PtrVector(alphai, N); bv = PtrVector(beta, N)
        cnt = 0
        @inbounds for i in 1:N
            ar[i] = real(al[i]); ai[i] = imag(al[i]); bv[i] = real(be[i])
            (sel[i] != 0) && (cnt += 1)
        end
        unsafe_store!(m, Int64(cnt)); unsafe_store!(info, Int64(0)); return
    end
end

# ── gesvx: expert general solve (gesvx.jl driver). fact/trans/equed are chars (3 Clong lengths). equed
# is IN/OUT (written back for fact='E'). rpgf → work[1] (REAL gesvx) / rwork[1] (COMPLEX gesvx). REAL R/C/
# rcond/ferr/berr for complex. d/c/z are native (getrf! exists); s is mixed-precision (no native F32 getrf!).
# REAL: `{d}gesvx_64_(fact,trans,n,nrhs,A,lda,AF,ldaf,ipiv,equed,R,C,B,ldb,X,ldx,rcond,ferr,berr,work,iwork,info, 3 len)`.
let  # dgesvx (native Float64) + sgesvx (mixed-precision) share the REAL ABI (rpgf→work[1], trailing iwork).
    @eval Base.@ccallable function dgesvx_64_(fact::Ptr{UInt8}, trans::Ptr{UInt8}, n::Ptr{Int64},
            nrhs::Ptr{Int64}, A::Ptr{Float64}, lda::Ptr{Int64}, AF::Ptr{Float64}, ldaf::Ptr{Int64},
            ipiv::Ptr{Int64}, equed::Ptr{UInt8}, R::Ptr{Float64}, C::Ptr{Float64}, B::Ptr{Float64},
            ldb::Ptr{Int64}, X::Ptr{Float64}, ldx::Ptr{Int64}, rcond::Ptr{Float64}, ferr::Ptr{Float64},
            berr::Ptr{Float64}, work::Ptr{Float64}, iwork::Ptr{Int64}, info::Ptr{Int64},
            lf::Clong, lt::Clong, le::Clong)::Cvoid
        _gesvx_real!(Float64, fact, trans, n, nrhs, A, lda, AF, ldaf, ipiv, equed, R, C, B, ldb, X, ldx,
                     rcond, ferr, berr, work, info)
        return
    end
    @eval Base.@ccallable function sgesvx_64_(fact::Ptr{UInt8}, trans::Ptr{UInt8}, n::Ptr{Int64},
            nrhs::Ptr{Int64}, A::Ptr{Float32}, lda::Ptr{Int64}, AF::Ptr{Float32}, ldaf::Ptr{Int64},
            ipiv::Ptr{Int64}, equed::Ptr{UInt8}, R::Ptr{Float32}, C::Ptr{Float32}, B::Ptr{Float32},
            ldb::Ptr{Int64}, X::Ptr{Float32}, ldx::Ptr{Int64}, rcond::Ptr{Float32}, ferr::Ptr{Float32},
            berr::Ptr{Float32}, work::Ptr{Float32}, iwork::Ptr{Int64}, info::Ptr{Int64},
            lf::Clong, lt::Clong, le::Clong)::Cvoid
        _gesvx_f32_mixed!(fact, trans, n, nrhs, A, lda, AF, ldaf, ipiv, equed, R, C, B, ldb, X, ldx,
                          rcond, ferr, berr, work, info)
        return
    end
end
# COMPLEX: R/C/rcond/ferr/berr REAL; rpgf → rwork[1]; trailing rwork (not iwork).
for (p, T, Tr) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval Base.@ccallable function $(Symbol(p, "gesvx_64_"))(fact::Ptr{UInt8}, trans::Ptr{UInt8},
            n::Ptr{Int64}, nrhs::Ptr{Int64}, A::Ptr{$T}, lda::Ptr{Int64}, AF::Ptr{$T}, ldaf::Ptr{Int64},
            ipiv::Ptr{Int64}, equed::Ptr{UInt8}, R::Ptr{$Tr}, C::Ptr{$Tr}, B::Ptr{$T}, ldb::Ptr{Int64},
            X::Ptr{$T}, ldx::Ptr{Int64}, rcond::Ptr{$Tr}, ferr::Ptr{$Tr}, berr::Ptr{$Tr},
            work::Ptr{$T}, rwork::Ptr{$Tr}, info::Ptr{Int64}, lf::Clong, lt::Clong, le::Clong)::Cvoid
        _gesvx_cplx!($T, $Tr, fact, trans, n, nrhs, A, lda, AF, ldaf, ipiv, equed, R, C, B, ldb, X, ldx,
                     rcond, ferr, berr, rwork, info)
        return
    end
end

# Shared shim bodies (kept out of @ccallable so the trimmer sees one concrete method each).
@inline function _gesvx_common!(::Type{T}, ::Type{Tr}, fact, trans, n, nrhs, A, lda, AF, ldaf, ipiv,
        equed, R, C, B, ldb, X, ldx, rcond, ferr, berr) where {T,Tr}
    N = Int(unsafe_load(n)); RH = Int(unsafe_load(nrhs))
    Am = PtrMatrix(A, N, N, Int(unsafe_load(lda))); AFm = PtrMatrix(AF, N, N, Int(unsafe_load(ldaf)))
    Bm = PtrMatrix(B, N, RH, Int(unsafe_load(ldb))); Xm = PtrMatrix(X, N, RH, Int(unsafe_load(ldx)))
    ip = PtrVector(ipiv, N); Rv = PtrVector(R, N); Cv = PtrVector(C, N)
    Xo, eq, rc, fe, be, rpgf = gesvx!(_cabi_char(fact), _cabi_char(trans), Am, AFm, ip,
                                      _cabi_char(equed), Rv, Cv, Bm)
    @inbounds for j in 1:RH, i in 1:N; Xm[i, j] = Xo[i, j]; end
    @inbounds for j in 1:RH; PtrVector(ferr, RH)[j] = fe[j]; PtrVector(berr, RH)[j] = be[j]; end
    unsafe_store!(equed, UInt8(eq)); unsafe_store!(rcond, Tr(rc))
    return rpgf
end
function _gesvx_real!(::Type{T}, fact, trans, n, nrhs, A, lda, AF, ldaf, ipiv, equed, R, C, B, ldb,
        X, ldx, rcond, ferr, berr, work, info) where {T}
    rpgf = _gesvx_common!(T, T, fact, trans, n, nrhs, A, lda, AF, ldaf, ipiv, equed, R, C, B, ldb,
                          X, ldx, rcond, ferr, berr)
    unsafe_store!(work, T(rpgf)); unsafe_store!(info, Int64(0)); return
end
function _gesvx_cplx!(::Type{T}, ::Type{Tr}, fact, trans, n, nrhs, A, lda, AF, ldaf, ipiv, equed, R, C,
        B, ldb, X, ldx, rcond, ferr, berr, rwork, info) where {T,Tr}
    rpgf = _gesvx_common!(T, Tr, fact, trans, n, nrhs, A, lda, AF, ldaf, ipiv, equed, R, C, B, ldb,
                          X, ldx, rcond, ferr, berr)
    unsafe_store!(rwork, Tr(rpgf)); unsafe_store!(info, Int64(0)); return
end
# ── bdsqr (COMPLEX): SVD of a REAL bidiagonal (d,e real) accumulating real Givens into COMPLEX Vt/U/C.
# Native complex sweep (svd.jl `bdsqr!(uplo,d,e,Vt,U,C)`, robust Demmel–Kahan). `rwork` is the LAPACK
# real workspace (unused — the kernel needs none). `{c,z}bdsqr_64_(uplo,n,ncvt,nru,ncc,d,e,Vt,ldvt,U,ldu,
# C,ldc,rwork,info, len_uplo)`. Empty Vt/U/C (0 cols/rows) are handled by the kernel (isempty guards).
for (p, T, Tr) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval Base.@ccallable function $(Symbol(p, "bdsqr_64_"))(uplo::Ptr{UInt8}, n::Ptr{Int64},
            ncvt::Ptr{Int64}, nru::Ptr{Int64}, ncc::Ptr{Int64}, d::Ptr{$Tr}, e::Ptr{$Tr}, Vt::Ptr{$T},
            ldvt::Ptr{Int64}, U::Ptr{$T}, ldu::Ptr{Int64}, C::Ptr{$T}, ldc::Ptr{Int64},
            rwork::Ptr{$Tr}, info::Ptr{Int64}, lu::Clong)::Cvoid
        N = Int(unsafe_load(n))
        NCVT = Int(unsafe_load(ncvt)); NRU = Int(unsafe_load(nru)); NCC = Int(unsafe_load(ncc))
        dv = PtrVector(d, N); ev = PtrVector(e, max(N - 1, 0))
        Vtm = PtrMatrix(Vt, N, NCVT, Int(unsafe_load(ldvt)))
        Um = PtrMatrix(U, NRU, N, Int(unsafe_load(ldu)))
        Cm = PtrMatrix(C, N, NCC, Int(unsafe_load(ldc)))
        bdsqr!(_cabi_char(uplo), dv, ev, Vtm, Um, Cm)
        unsafe_store!(info, Int64(0)); return
    end
end

# sgesvx: promote to Float64, run the native driver, demote (no native real-F32 getrf!).
function _gesvx_f32_mixed!(fact, trans, n, nrhs, A, lda, AF, ldaf, ipiv, equed, R, C, B, ldb, X, ldx,
        rcond, ferr, berr, work, info)
    N = Int(unsafe_load(n)); RH = Int(unsafe_load(nrhs))
    Am = PtrMatrix(A, N, N, Int(unsafe_load(lda))); Bm = PtrMatrix(B, N, RH, Int(unsafe_load(ldb)))
    Ad = Matrix{Float64}(undef, N, N); Bd = Matrix{Float64}(undef, N, RH)
    @inbounds for j in 1:N, i in 1:N; Ad[i, j] = Float64(Am[i, j]); end
    @inbounds for j in 1:RH, i in 1:N; Bd[i, j] = Float64(Bm[i, j]); end
    AFd = Matrix{Float64}(undef, N, N); ipd = Vector{Int}(undef, N)
    Rd = Vector{Float64}(undef, N); Cd = Vector{Float64}(undef, N)
    Xo, eq, rc, fe, be, rpgf = gesvx!(_cabi_char(fact), _cabi_char(trans), Ad, AFd, ipd,
                                      _cabi_char(equed), Rd, Cd, Bd)
    Xm = PtrMatrix(X, N, RH, Int(unsafe_load(ldx)))
    @inbounds for j in 1:RH, i in 1:N; Xm[i, j] = Float32(Xo[i, j]); end
    @inbounds for j in 1:RH; PtrVector(ferr, RH)[j] = Float32(fe[j]); PtrVector(berr, RH)[j] = Float32(be[j]); end
    ipv = PtrVector(ipiv, N); @inbounds for i in 1:N; ipv[i] = Int64(ipd[i]); end
    unsafe_store!(equed, UInt8(eq)); unsafe_store!(rcond, Float32(rc))
    unsafe_store!(work, Float32(rpgf)); unsafe_store!(info, Int64(0)); return
end
