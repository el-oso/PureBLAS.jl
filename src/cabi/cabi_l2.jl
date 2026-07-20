# Mode 1 C/Fortran-ABI boundary for BLAS Level-2 (extends cabi.jl's GEMM char-arg pattern).
#
# Same ABI as GEMM: char args are `Ptr{UInt8}` FIRST (deref+ASCII-uppercase via `_cabi_char`), then
# the by-reference scalar/array `Ptr`s, then ONE trailing hidden Fortran string-length `Clong` per
# char arg (values ignored). Column-major, ILP64 (Int64). Char→flag conventions match the kernels:
#   trans: op≠N ⇒ transpose,  'C' ⇒ conjugate ;  uplo: 'U' ⇒ upper ;  diag: 'U' ⇒ unit.
#
# Matrix args are bridged to a StridedMatrix view (like GEMM). VECTOR args (x/y) and the PACKED
# vector (AP) are passed as the raw `Ptr` + inc straight into the low-level kernels — the kernels
# take `(n, ptr, inc)` directly (see cabi.jl BLAS-1). Raw pointers are not `StridedVector`, so the
# real-SIMD fast paths (which gate on `x isa StridedVector`) fall through to the generic scalar loop;
# that is correct for all s/d/c/z (only the SIMD micro-opt is skipped at this boundary).

# Ptr→matrix bridge: isbits `PtrMatrix` (ptr, nr, nc, ld) — stride(1)==1, stride(2)==ld dense operand.
# Trim-safe and 0-alloc (isbits ⇒ passes by value into the kernels with no heap box). Vector args
# (x/y) stay raw Ptr, so the real-SIMD L2 paths gate off (StridedVector) and take the generic loop —
# which indexes PtrMatrix via unsafe_load, still 0-alloc.
@inline function _cabi_mat(p::Ptr{T}, ld::Int64, nr::Int64, nc::Int64) where {T}
    return PtrMatrix(p, Int(nr), Int(nc), Int(ld))
end

# ── gemv: y := α·op(A)·x + β·y  (1 char: trans) ──────────────────────────────────────────────────
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "gemv_64_"))(
            trans::Ptr{UInt8}, m::Ptr{Int64}, n::Ptr{Int64}, alpha::Ptr{$T},
            A::Ptr{$T}, lda::Ptr{Int64}, x::Ptr{$T}, incx::Ptr{Int64},
            beta::Ptr{$T}, y::Ptr{$T}, incy::Ptr{Int64}, len_trans::Clong
        )::Cvoid
        tc = _cabi_char(trans); mm = unsafe_load(m); nn = unsafe_load(n)
        Am = _cabi_mat(A, unsafe_load(lda), mm, nn)          # A is always stored m×n
        _gemv!(
            tc != 'N', tc == 'C', mm, nn, unsafe_load(alpha), Am,
            x, unsafe_load(incx), unsafe_load(beta), y, unsafe_load(incy)
        )
        return
    end
end

# ── ger/geru/gerc: A := α·x·(y or ȳ)ᵀ + A  (0 chars) ─────────────────────────────────────────────
for (sym, T, cj) in (
        (:sger_64_, Float32, false), (:dger_64_, Float64, false),
        (:cgeru_64_, ComplexF32, false), (:zgeru_64_, ComplexF64, false),
        (:cgerc_64_, ComplexF32, true), (:zgerc_64_, ComplexF64, true),
    )
    @eval Base.@ccallable function $sym(
            m::Ptr{Int64}, n::Ptr{Int64}, alpha::Ptr{$T},
            x::Ptr{$T}, incx::Ptr{Int64}, y::Ptr{$T}, incy::Ptr{Int64},
            A::Ptr{$T}, lda::Ptr{Int64}
        )::Cvoid
        mm = unsafe_load(m); nn = unsafe_load(n)
        Am = _cabi_mat(A, unsafe_load(lda), mm, nn)
        _ger!(
            $cj, mm, nn, unsafe_load(alpha), x, unsafe_load(incx),
            y, unsafe_load(incy), Am
        )
        return
    end
end

# ── symv (s,d) / hemv (c,z): y := α·A·x + β·y  (1 char: uplo) ─────────────────────────────────────
for (sym, T, kern) in (
        (:ssymv_64_, Float32, :_symv!), (:dsymv_64_, Float64, :_symv!),
        (:chemv_64_, ComplexF32, :_hemv!), (:zhemv_64_, ComplexF64, :_hemv!),
    )
    @eval Base.@ccallable function $sym(
            uplo::Ptr{UInt8}, n::Ptr{Int64}, alpha::Ptr{$T},
            A::Ptr{$T}, lda::Ptr{Int64}, x::Ptr{$T}, incx::Ptr{Int64},
            beta::Ptr{$T}, y::Ptr{$T}, incy::Ptr{Int64}, len_uplo::Clong
        )::Cvoid
        nn = unsafe_load(n)
        Am = _cabi_mat(A, unsafe_load(lda), nn, nn)
        $kern(
            _cabi_char(uplo) == 'U', nn, unsafe_load(alpha), Am,
            x, unsafe_load(incx), unsafe_load(beta), y, unsafe_load(incy)
        )
        return
    end
end

# ── trmv / trsv: x := op(A)·x  /  op(A)⁻¹·x  (3 chars: uplo, trans, diag) ─────────────────────────
for (op, kern) in (("trmv", :_trmv!), ("trsv", :_trsv!))
    for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
        @eval Base.@ccallable function $(Symbol(p, op, "_64_"))(
                uplo::Ptr{UInt8}, trans::Ptr{UInt8}, diag::Ptr{UInt8}, n::Ptr{Int64},
                A::Ptr{$T}, lda::Ptr{Int64}, x::Ptr{$T}, incx::Ptr{Int64},
                len_u::Clong, len_t::Clong, len_d::Clong
            )::Cvoid
            nn = unsafe_load(n)
            uc = _cabi_char(uplo); tc = _cabi_char(trans); dc = _cabi_char(diag)
            Am = _cabi_mat(A, unsafe_load(lda), nn, nn)
            $kern(uc == 'U', tc != 'N', tc == 'C', dc == 'U', nn, Am, x, unsafe_load(incx))
            return
        end
    end
end

# ── spmv (s,d) / hpmv (c,z), packed: y := α·A·x + β·y  (1 char: uplo) ─────────────────────────────
for (sym, T, kern) in (
        (:sspmv_64_, Float32, :_spmv!), (:dspmv_64_, Float64, :_spmv!),
        (:chpmv_64_, ComplexF32, :_hpmv!), (:zhpmv_64_, ComplexF64, :_hpmv!),
    )
    @eval Base.@ccallable function $sym(
            uplo::Ptr{UInt8}, n::Ptr{Int64}, alpha::Ptr{$T},
            AP::Ptr{$T}, x::Ptr{$T}, incx::Ptr{Int64},
            beta::Ptr{$T}, y::Ptr{$T}, incy::Ptr{Int64}, len_uplo::Clong
        )::Cvoid
        $kern(
            _cabi_char(uplo) == 'U', unsafe_load(n), unsafe_load(alpha), AP,
            x, unsafe_load(incx), unsafe_load(beta), y, unsafe_load(incy)
        )
        return
    end
end

# ── tpmv / tpsv, packed: x := op(A)·x / op(A)⁻¹·x  (3 chars: uplo, trans, diag) ───────────────────
for (op, kern) in (("tpmv", :_tpmv!), ("tpsv", :_tpsv!))
    for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
        @eval Base.@ccallable function $(Symbol(p, op, "_64_"))(
                uplo::Ptr{UInt8}, trans::Ptr{UInt8}, diag::Ptr{UInt8}, n::Ptr{Int64},
                AP::Ptr{$T}, x::Ptr{$T}, incx::Ptr{Int64},
                len_u::Clong, len_t::Clong, len_d::Clong
            )::Cvoid
            uc = _cabi_char(uplo); tc = _cabi_char(trans); dc = _cabi_char(diag)
            $kern(uc == 'U', tc != 'N', tc == 'C', dc == 'U', unsafe_load(n), AP, x, unsafe_load(incx))
            return
        end
    end
end

# ── spr (s,d) / hpr (c,z), packed rank-1: A := α·x·(x or x̄)ᵀ + A  (1 char: uplo) ──────────────────
# hpr's α is REAL (chpr→Float32, zhpr→Float64); spr's α matches the element type.
for (sym, T, AT, kern) in (
        (:sspr_64_, Float32, Float32, :_spr!), (:dspr_64_, Float64, Float64, :_spr!),
        (:chpr_64_, ComplexF32, Float32, :_hpr!), (:zhpr_64_, ComplexF64, Float64, :_hpr!),
    )
    @eval Base.@ccallable function $sym(
            uplo::Ptr{UInt8}, n::Ptr{Int64}, alpha::Ptr{$AT},
            x::Ptr{$T}, incx::Ptr{Int64}, AP::Ptr{$T}, len_uplo::Clong
        )::Cvoid
        $kern(_cabi_char(uplo) == 'U', unsafe_load(n), unsafe_load(alpha), x, unsafe_load(incx), AP)
        return
    end
end

# ── spr2 (s,d) / hpr2 (c,z), packed rank-2  (1 char: uplo) ───────────────────────────────────────
# spr2's α is real, hpr2's α is complex — both match the element type $T, so alpha is Ptr{$T}.
for (sym, T, kern) in (
        (:sspr2_64_, Float32, :_spr2!), (:dspr2_64_, Float64, :_spr2!),
        (:chpr2_64_, ComplexF32, :_hpr2!), (:zhpr2_64_, ComplexF64, :_hpr2!),
    )
    @eval Base.@ccallable function $sym(
            uplo::Ptr{UInt8}, n::Ptr{Int64}, alpha::Ptr{$T},
            x::Ptr{$T}, incx::Ptr{Int64}, y::Ptr{$T}, incy::Ptr{Int64},
            AP::Ptr{$T}, len_uplo::Clong
        )::Cvoid
        $kern(
            _cabi_char(uplo) == 'U', unsafe_load(n), unsafe_load(alpha),
            x, unsafe_load(incx), y, unsafe_load(incy), AP
        )
        return
    end
end

# ── gbmv: y := α·op(A)·x + β·y, general banded  (1 char: trans) ──────────────────────────────────
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "gbmv_64_"))(
            trans::Ptr{UInt8}, m::Ptr{Int64}, n::Ptr{Int64}, kl::Ptr{Int64}, ku::Ptr{Int64},
            alpha::Ptr{$T}, AB::Ptr{$T}, ldab::Ptr{Int64}, x::Ptr{$T}, incx::Ptr{Int64},
            beta::Ptr{$T}, y::Ptr{$T}, incy::Ptr{Int64}, len_trans::Clong
        )::Cvoid
        tc = _cabi_char(trans)
        mm = unsafe_load(m); nn = unsafe_load(n); klv = unsafe_load(kl); kuv = unsafe_load(ku)
        ABm = _cabi_mat(AB, unsafe_load(ldab), klv + kuv + 1, nn)   # band packed (kl+ku+1)×n
        _gbmv!(
            tc != 'N', tc == 'C', mm, nn, klv, kuv, unsafe_load(alpha), ABm,
            x, unsafe_load(incx), unsafe_load(beta), y, unsafe_load(incy)
        )
        return
    end
end

# ── sbmv (s,d) / hbmv (c,z): y := α·A·x + β·y, symmetric/Hermitian banded  (1 char: uplo) ─────────
for (sym, T, kern) in (
        (:ssbmv_64_, Float32, :_sbmv!), (:dsbmv_64_, Float64, :_sbmv!),
        (:chbmv_64_, ComplexF32, :_hbmv!), (:zhbmv_64_, ComplexF64, :_hbmv!),
    )
    @eval Base.@ccallable function $sym(
            uplo::Ptr{UInt8}, n::Ptr{Int64}, k::Ptr{Int64},
            alpha::Ptr{$T}, AB::Ptr{$T}, ldab::Ptr{Int64}, x::Ptr{$T}, incx::Ptr{Int64},
            beta::Ptr{$T}, y::Ptr{$T}, incy::Ptr{Int64}, len_uplo::Clong
        )::Cvoid
        nn = unsafe_load(n); kv = unsafe_load(k)
        ABm = _cabi_mat(AB, unsafe_load(ldab), kv + 1, nn)         # band packed (k+1)×n
        $kern(
            _cabi_char(uplo) == 'U', nn, kv, unsafe_load(alpha), ABm,
            x, unsafe_load(incx), unsafe_load(beta), y, unsafe_load(incy)
        )
        return
    end
end

# ── tbmv / tbsv: x := op(A)·x / op(A)⁻¹·x, triangular banded  (3 chars: uplo, trans, diag) ────────
for (op, kern) in (("tbmv", :_tbmv!), ("tbsv", :_tbsv!))
    for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
        @eval Base.@ccallable function $(Symbol(p, op, "_64_"))(
                uplo::Ptr{UInt8}, trans::Ptr{UInt8}, diag::Ptr{UInt8}, n::Ptr{Int64}, k::Ptr{Int64},
                AB::Ptr{$T}, ldab::Ptr{Int64}, x::Ptr{$T}, incx::Ptr{Int64},
                len_u::Clong, len_t::Clong, len_d::Clong
            )::Cvoid
            nn = unsafe_load(n); kv = unsafe_load(k)
            uc = _cabi_char(uplo); tc = _cabi_char(trans); dc = _cabi_char(diag)
            ABm = _cabi_mat(AB, unsafe_load(ldab), kv + 1, nn)     # band packed (k+1)×n
            $kern(uc == 'U', tc != 'N', tc == 'C', dc == 'U', nn, kv, ABm, x, unsafe_load(incx))
            return
        end
    end
end
