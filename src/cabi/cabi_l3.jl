# Mode 1 C/Fortran-ABI, Level-3 beyond GEMM: symm/hemm, syrk/herk, syr2k/her2k, trmm/trsm.
# Same ABI shape as gemm (see cabi.jl): character args are `Ptr{UInt8}` FIRST (deref +
# byte-uppercase via `_cabi_char`), then by-ref scalars/arrays, then one trailing hidden Fortran
# string-length `Clong` per character arg (value 1 each — ignored). Column-major, ILP64.
#
# These forward to the PUBLIC kwarg ops (symm!/…/trsm!), not the positional `_symm!`/`_trmm!`
# cores: the gate-path routing (real single-pass packed kernels, po2-ld A-pad) lives in the public
# function bodies — e.g. side-L packed trmm is chosen only in `trmm!`, and `_trmm!` alone would fall
# to the measured-slower recursion. The kwarg box (~64 B) is dwarfed by the matrix-wrap bridge
# (~384 B/call) that dominates the Mode-1 boundary anyway (per task note). The redundant dimension
# checks the public ops run are cheap and 0-alloc.

# Pointer→matrix bridge (as in _gemm_cabi!): isbits `PtrMatrix` (ptr, nr, nc, ld) — a stride(1)==1,
# stride(2)==ld dense operand. Trim-safe, 0-alloc (isbits ⇒ passes by value with no heap box). Its
# sub-views (the recursive L3 blocking uses view(A, :, r) etc.) stay PtrMatrix, so the recursion leaves
# still hit the vectorized gemm/trsm fast paths.
@inline _l3wrap(p::Ptr{T}, ld, nr, nc) where {T} = PtrMatrix(p, Int(nr), Int(nc), Int(ld))

# ── symm / hemm: (side,uplo, m,n, α, A,lda, B,ldb, β, C,ldc) + 2 trailing Clongs ────────────────
# C(m×n) := α·A·B + β·C (side L, A m×m) or α·B·A + β·C (side R, A n×n); only `uplo` triangle of A.
for (op, name, list) in (
        (:symm!, "symm", (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))),
        (:hemm!, "hemm", (("c", ComplexF32), ("z", ComplexF64))),
    )
    for (p, T) in list
        @eval Base.@ccallable function $(Symbol(p, name, "_64_"))(
                side::Ptr{UInt8}, uplo::Ptr{UInt8}, m::Ptr{Int64}, n::Ptr{Int64},
                alpha::Ptr{$T}, A::Ptr{$T}, lda::Ptr{Int64}, B::Ptr{$T}, ldb::Ptr{Int64},
                beta::Ptr{$T}, C::Ptr{$T}, ldc::Ptr{Int64}, len_s::Clong, len_u::Clong
            )::Cvoid
            s = _cabi_char(side); mm = unsafe_load(m); nn = unsafe_load(n)
            ka = s == 'L' ? mm : nn
            Am = _l3wrap(A, unsafe_load(lda), ka, ka)
            Bm = _l3wrap(B, unsafe_load(ldb), mm, nn)
            Cm = _l3wrap(C, unsafe_load(ldc), mm, nn)
            $op(
                Cm, Am, Bm; side = s, uplo = _cabi_char(uplo),
                alpha = unsafe_load(alpha), beta = unsafe_load(beta)
            )
            return
        end
    end
end

# ── syrk / herk: (uplo,trans, n,k, α, A,lda, β, C,ldc) + 2 trailing Clongs ──────────────────────
# C(n×n) := α·op(A)·op(A)ᴴ + β·C, only `uplo` triangle. op(A)=A (n×k, trans 'N') or Aᴴ (k×n).
# herk: α,β REAL (Ptr{$ST}). A-wrap rows/cols swap with trans.
for (p, op, name, T, ST) in (
        ("s", :syrk!, "syrk", Float32, Float32),
        ("d", :syrk!, "syrk", Float64, Float64),
        ("c", :syrk!, "syrk", ComplexF32, ComplexF32),
        ("z", :syrk!, "syrk", ComplexF64, ComplexF64),
        ("c", :herk!, "herk", ComplexF32, Float32),
        ("z", :herk!, "herk", ComplexF64, Float64),
    )
    @eval Base.@ccallable function $(Symbol(p, name, "_64_"))(
            uplo::Ptr{UInt8}, trans::Ptr{UInt8}, n::Ptr{Int64}, k::Ptr{Int64},
            alpha::Ptr{$ST}, A::Ptr{$T}, lda::Ptr{Int64},
            beta::Ptr{$ST}, C::Ptr{$T}, ldc::Ptr{Int64}, len_u::Clong, len_t::Clong
        )::Cvoid
        t = _cabi_char(trans); nn = unsafe_load(n); kk = unsafe_load(k)
        ar = t == 'N' ? nn : kk; ac = t == 'N' ? kk : nn
        Am = _l3wrap(A, unsafe_load(lda), ar, ac)
        Cm = _l3wrap(C, unsafe_load(ldc), nn, nn)
        $op(
            Cm, Am; uplo = _cabi_char(uplo), trans = t,
            alpha = unsafe_load(alpha), beta = unsafe_load(beta)
        )
        return
    end
end

# ── syr2k / her2k: (uplo,trans, n,k, α, A,lda, B,ldb, β, C,ldc) + 2 trailing Clongs ─────────────
# C(n×n) := α·op(A)·op(B)ᴴ + (α or ᾱ)·op(B)·op(A)ᴴ + β·C, only `uplo` triangle. A,B same shape:
# n×k (trans 'N') or k×n. her2k: α complex (Ptr{$T}), β REAL (Ptr{$ST}).
for (p, op, name, T, ST) in (
        ("s", :syr2k!, "syr2k", Float32, Float32),
        ("d", :syr2k!, "syr2k", Float64, Float64),
        ("c", :syr2k!, "syr2k", ComplexF32, ComplexF32),
        ("z", :syr2k!, "syr2k", ComplexF64, ComplexF64),
        ("c", :her2k!, "her2k", ComplexF32, Float32),
        ("z", :her2k!, "her2k", ComplexF64, Float64),
    )
    @eval Base.@ccallable function $(Symbol(p, name, "_64_"))(
            uplo::Ptr{UInt8}, trans::Ptr{UInt8}, n::Ptr{Int64}, k::Ptr{Int64},
            alpha::Ptr{$T}, A::Ptr{$T}, lda::Ptr{Int64}, B::Ptr{$T}, ldb::Ptr{Int64},
            beta::Ptr{$ST}, C::Ptr{$T}, ldc::Ptr{Int64}, len_u::Clong, len_t::Clong
        )::Cvoid
        t = _cabi_char(trans); nn = unsafe_load(n); kk = unsafe_load(k)
        ar = t == 'N' ? nn : kk; ac = t == 'N' ? kk : nn
        Am = _l3wrap(A, unsafe_load(lda), ar, ac)
        Bm = _l3wrap(B, unsafe_load(ldb), ar, ac)
        Cm = _l3wrap(C, unsafe_load(ldc), nn, nn)
        $op(
            Cm, Am, Bm; uplo = _cabi_char(uplo), trans = t,
            alpha = unsafe_load(alpha), beta = unsafe_load(beta)
        )
        return
    end
end

# ── trmm / trsm: (side,uplo,transa,diag, m,n, α, A,lda, B,ldb) + 4 trailing Clongs ──────────────
# B(m×n) overwritten := α·op(A)·B / α·B·op(A) (trmm) or the op(A)⁻¹ solves (trsm). A triangular
# m×m (side L) or n×n (side R); `uplo`/`transa`/`diag` select the triangle/op/unit-diagonal.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64)),
        (op, name) in ((:trmm!, "trmm"), (:trsm!, "trsm"))
    @eval Base.@ccallable function $(Symbol(p, name, "_64_"))(
            side::Ptr{UInt8}, uplo::Ptr{UInt8}, transa::Ptr{UInt8}, diag::Ptr{UInt8},
            m::Ptr{Int64}, n::Ptr{Int64}, alpha::Ptr{$T}, A::Ptr{$T}, lda::Ptr{Int64},
            B::Ptr{$T}, ldb::Ptr{Int64},
            len_s::Clong, len_u::Clong, len_t::Clong, len_d::Clong
        )::Cvoid
        s = _cabi_char(side); mm = unsafe_load(m); nn = unsafe_load(n)
        ka = s == 'L' ? mm : nn
        Am = _l3wrap(A, unsafe_load(lda), ka, ka)
        Bm = _l3wrap(B, unsafe_load(ldb), mm, nn)
        $op(
            Bm, Am; side = s, uplo = _cabi_char(uplo), transA = _cabi_char(transa),
            diag = _cabi_char(diag), alpha = unsafe_load(alpha)
        )
        return
    end
end
