# Mode 1 C/Fortran-ABI boundary: the 4 complex-dot symbols (c/zdotu, c/zdotc) — the ONE BLAS-1 gap
# that cabi.jl deferred. Their only wrinkle vs the real ddot/sdot is the complex-return ABI.
#
# RETSTYLE (verified, not guessed — see the scratchpad note complex-dot-abi.md): Julia's OpenBLAS64_
# (`complex_retstyle = normal` in `BLAS.lbt_get_config()`) and PureBLAS's own `activate()` both use the
# NORMAL convention — the complex result is returned BY VALUE in registers (System V xmm0:xmm1 for
# `_Complex double`), NOT via an f2c/GNU hidden first pointer argument (that would be RETSTYLE ARGUMENT).
# Empirical proof: ccalling the loaded `zdotc_64_` with a by-value `ComplexF64` return type yields the
# correct dot product, while the hidden-first-pointer form yields 0. So the @ccallable signature is
# IDENTICAL to the real `ddot_64_` — result via the Julia/C return value, all args by reference — just
# with a complex return type. Julia lowers a `::ComplexF64`/`::ComplexF32` @ccallable return to the C
# `_Complex double`/`_Complex float` ABI, which is exactly what LBT's NORMAL retstyle expects.
#
# `dotu` = Σ xᵢ·yᵢ (unconjugated, reference ?dotu);  `dotc` = Σ conj(xᵢ)·yᵢ (conjugated, reference ?dotc).
# Both reuse the shared level1 kernels (zero-alloc, trim-safe). Two-vector op ⇒ negative/mismatched
# increments are legal (matches the axpy/dot spec), so incx/incy are passed straight through.
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))
    @eval begin
        Base.@ccallable function $(Symbol(p, "dotu_64_"))(
                n::Ptr{Int64}, x::Ptr{$T},
                incx::Ptr{Int64}, y::Ptr{$T}, incy::Ptr{Int64}
            )::$T
            return _dotu(unsafe_load(n), x, unsafe_load(incx), y, unsafe_load(incy))
        end
        Base.@ccallable function $(Symbol(p, "dotc_64_"))(
                n::Ptr{Int64}, x::Ptr{$T},
                incx::Ptr{Int64}, y::Ptr{$T}, incy::Ptr{Int64}
            )::$T
            return _dotc(unsafe_load(n), x, unsafe_load(incx), y, unsafe_load(incy))
        end
    end
end
