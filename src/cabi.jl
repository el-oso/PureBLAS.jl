# Mode 1 C/Fortran-ABI boundary: `@ccallable` entry points that libblastrampoline forwards to.
#
# Names are the ILP64 reference-BLAS symbols Julia's LinearAlgebra resolves (`@blasfunc` →
# trailing `64_`). All arguments are by reference (`Ptr`), column-major, 64-bit integers — the
# Fortran ABI. BLAS-1 takes no character arguments, so there are no hidden string-length args.
# These wrappers only unpack scalars and call the shared kernels (level1.jl) — zero allocation,
# trim-safe, and the juliac --trim entry points for libpureblas.so.
#
# ponytail: the 4 complex-dot symbols (c/zdotu, c/zdotc) are intentionally NOT exported here —
# their complex-return ABI (LBT NORMAL vs ARGUMENT retstyle) is unresolved and lands with the
# GEMM character/string ABI work in M2. The native API (backend.jl) covers complex dot meanwhile.

# void-returning, all 4 element types: axpy, scal, copy, swap.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval begin
        Base.@ccallable function $(Symbol(p, "axpy_64_"))(n::Ptr{Int64}, a::Ptr{$T},
                x::Ptr{$T}, incx::Ptr{Int64}, y::Ptr{$T}, incy::Ptr{Int64})::Cvoid
            _axpy!(unsafe_load(n), unsafe_load(a), x, unsafe_load(incx), y, unsafe_load(incy))
            return
        end
        Base.@ccallable function $(Symbol(p, "scal_64_"))(n::Ptr{Int64}, a::Ptr{$T},
                x::Ptr{$T}, incx::Ptr{Int64})::Cvoid
            _scal!(unsafe_load(n), unsafe_load(a), x, unsafe_load(incx))
            return
        end
        Base.@ccallable function $(Symbol(p, "copy_64_"))(n::Ptr{Int64}, x::Ptr{$T},
                incx::Ptr{Int64}, y::Ptr{$T}, incy::Ptr{Int64})::Cvoid
            _copy!(unsafe_load(n), x, unsafe_load(incx), y, unsafe_load(incy))
            return
        end
        Base.@ccallable function $(Symbol(p, "swap_64_"))(n::Ptr{Int64}, x::Ptr{$T},
                incx::Ptr{Int64}, y::Ptr{$T}, incy::Ptr{Int64})::Cvoid
            _swap!(unsafe_load(n), x, unsafe_load(incx), y, unsafe_load(incy))
            return
        end
    end
end

# real dot (sdot, ddot) → real scalar.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval Base.@ccallable function $(Symbol(p, "dot_64_"))(n::Ptr{Int64}, x::Ptr{$T},
            incx::Ptr{Int64}, y::Ptr{$T}, incy::Ptr{Int64})::$T
        return _dotu(unsafe_load(n), x, unsafe_load(incx), y, unsafe_load(incy))
    end
end

# nrm2 → real scalar (real and complex inputs).
for (sym, T, R) in ((:snrm2_64_, Float32, Float32), (:dnrm2_64_, Float64, Float64),
                    (:scnrm2_64_, ComplexF32, Float32), (:dznrm2_64_, ComplexF64, Float64))
    @eval Base.@ccallable function $sym(n::Ptr{Int64}, x::Ptr{$T}, incx::Ptr{Int64})::$R
        return _nrm2(unsafe_load(n), x, unsafe_load(incx))
    end
end

# asum → real scalar (real and complex inputs).
for (sym, T, R) in ((:sasum_64_, Float32, Float32), (:dasum_64_, Float64, Float64),
                    (:scasum_64_, ComplexF32, Float32), (:dzasum_64_, ComplexF64, Float64))
    @eval Base.@ccallable function $sym(n::Ptr{Int64}, x::Ptr{$T}, incx::Ptr{Int64})::$R
        return _asum(unsafe_load(n), x, unsafe_load(incx))
    end
end

# iamax → Int64 index.
for (sym, T) in ((:isamax_64_, Float32), (:idamax_64_, Float64),
                 (:icamax_64_, ComplexF32), (:izamax_64_, ComplexF64))
    @eval Base.@ccallable function $sym(n::Ptr{Int64}, x::Ptr{$T}, incx::Ptr{Int64})::Int64
        return Int64(_iamax(unsafe_load(n), x, unsafe_load(incx)))
    end
end

# ── Level-3 GEMM: the character-argument ABI ─────────────────────────────────────────────────
# First BLAS op with char args (transA/transB). Julia's LinearAlgebra ccalls dgemm_64_ as
#   (Ref{UInt8}, Ref{UInt8}, Ref{Int64}×3, Ref{T}, Ptr{T}, Ref{Int64}, …, Clong, Clong)
# i.e. two char args by reference FIRST, all by-ref scalars/arrays, then TWO trailing hidden Fortran
# string-length Clongs (value 1 each — ignored). Deref chars with unsafe_load → UInt8 → Char.
# ASCII-uppercase the char at the boundary (reference BLAS is case-insensitive via lsame; gemm!
# checks `!= 'N'` and `== 'C'`, so a lowercase 'c' would transpose-without-conjugate — a bug). Manual
# byte uppercase, not `uppercase(::Char)` (Unicode tables aren't trim-safe / 0-alloc).
@inline function _cabi_char(p::Ptr{UInt8})
    b = unsafe_load(p)
    return b >= 0x61 ? Char(b - 0x20) : Char(b)
end

# Pointer→matrix bridge: wrap each operand as an isbits `PtrMatrix` (ptr, rows, cols, ld) — a
# stride(1)==1, stride(2)==ld dense-column operand, exactly what gemm!'s `_strided1` fast path wants.
# Trim-safe and 0-alloc: PtrMatrix is isbits, so it passes BY VALUE into the non-inlined pack/driver
# kernels with no heap box (the SubArray-of-unsafe_wrap bridge it replaces cost ~384 B/call).
@inline function _gemm_cabi!(transa::Ptr{UInt8}, transb::Ptr{UInt8}, m::Int64, n::Int64, k::Int64,
        alpha::T, A::Ptr{T}, lda::Int64, B::Ptr{T}, ldb::Int64,
        beta::T, C::Ptr{T}, ldc::Int64) where {T}
    ta = _cabi_char(transa); tb = _cabi_char(transb)
    trA = ta != 'N'; trB = tb != 'N'
    Arows = trA ? k : m; Acols = trA ? m : k    # op(A)=Aᵀ ⇒ A stored k×m; op(A)=A ⇒ m×k
    Brows = trB ? n : k; Bcols = trB ? k : n
    Am = PtrMatrix(A, Int(Arows), Int(Acols), Int(lda))
    Bm = PtrMatrix(B, Int(Brows), Int(Bcols), Int(ldb))
    Cm = PtrMatrix(C, Int(m), Int(n), Int(ldc))
    # Call the @inline dispatch core directly (not the public kwarg `gemm!`): the kwarg entry boxes
    # its keyword args (measured +64 B), and the hot L3 D&C path already routes through _gemm_core!.
    _gemm_core!(Cm, Am, Bm, alpha, beta, trA, trB, ta == 'C', tb == 'C')
    return
end

# s/d/c/z GEMM @ccallable entries — two Ptr{UInt8} chars, by-ref scalars/arrays, two trailing Clongs.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval Base.@ccallable function $(Symbol(p, "gemm_64_"))(
            transa::Ptr{UInt8}, transb::Ptr{UInt8}, m::Ptr{Int64}, n::Ptr{Int64}, k::Ptr{Int64},
            alpha::Ptr{$T}, A::Ptr{$T}, lda::Ptr{Int64}, B::Ptr{$T}, ldb::Ptr{Int64},
            beta::Ptr{$T}, C::Ptr{$T}, ldc::Ptr{Int64}, len_ta::Clong, len_tb::Clong)::Cvoid
        _gemm_cabi!(transa, transb, unsafe_load(m), unsafe_load(n), unsafe_load(k),
            unsafe_load(alpha), A, unsafe_load(lda), B, unsafe_load(ldb),
            unsafe_load(beta), C, unsafe_load(ldc))
        return
    end
end
