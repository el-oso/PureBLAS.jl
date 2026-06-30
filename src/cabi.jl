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
