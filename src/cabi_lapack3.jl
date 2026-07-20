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
