# In-process LBT forwarding: register `@cfunction` pointers to the native `@ccallable` kernels so
# `PureBLAS.activate()` reroutes LinearAlgebra's BLAS/LAPACK calls to PureBLAS inside a LIVE Julia
# process. This is the analogue of MKL.jl's `__init__` (which `lbt_forward`s libmkl_rt) — but we
# register IN-PROCESS function pointers via `lbt_set_forward` instead of dlopening a library: the
# juliac `.so` embeds its own libjulia, so dlopen-forwarding it double-inits the runtime (signal 6).
# `@cfunction` pointers run in THIS process against THIS runtime, so there is no double-init.
#
# Each entry is `(lbt_name, thunk)` where `lbt_name` is the reference BLAS/LAPACK symbol WITHOUT the
# ILP64 `64_` suffix (LBT re-applies it given `interface=ILP64`) and `thunk()` builds the `@cfunction`
# pointer at activate() time (never at precompile — no raw Ptr is serialized). The `@cfunction`
# signatures MUST match the `@ccallable` defs in cabi*.jl exactly; a mismatch surfaces as a wrong
# result in the forward-correctness dogfood (test/lbt_forward_tests.jl).

const _LBT_REGISTRARS = Vector{Tuple{String, Function}}()
@inline _reg!(name::String, thunk) = push!(_LBT_REGISTRARS, (name, thunk))

const _CI = Ptr{Int64}     # by-ref Int64 (dims/incs)
const _CU = Ptr{UInt8}     # by-ref char

# ─────────────────────────────── BLAS-1 (cabi.jl) ────────────────────────────────────────────────
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval begin
        _reg!($(p * "axpy_"), () -> @cfunction($(Symbol(p, "axpy_64_")), Cvoid,
            (_CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI)))
        _reg!($(p * "scal_"), () -> @cfunction($(Symbol(p, "scal_64_")), Cvoid,
            (_CI, Ptr{$T}, Ptr{$T}, _CI)))
        _reg!($(p * "copy_"), () -> @cfunction($(Symbol(p, "copy_64_")), Cvoid,
            (_CI, Ptr{$T}, _CI, Ptr{$T}, _CI)))
        _reg!($(p * "swap_"), () -> @cfunction($(Symbol(p, "swap_64_")), Cvoid,
            (_CI, Ptr{$T}, _CI, Ptr{$T}, _CI)))
    end
end
for (p, T) in (("s", Float32), ("d", Float64))     # real dot only (complex dot ABI deferred)
    @eval _reg!($(p * "dot_"), () -> @cfunction($(Symbol(p, "dot_64_")), $T,
        (_CI, Ptr{$T}, _CI, Ptr{$T}, _CI)))
end
for (nm, T, R) in (("snrm2", Float32, Float32), ("dnrm2", Float64, Float64),
                   ("scnrm2", ComplexF32, Float32), ("dznrm2", ComplexF64, Float64))
    @eval _reg!($(nm * "_"), () -> @cfunction($(Symbol(nm, "_64_")), $R, (_CI, Ptr{$T}, _CI)))
end
for (nm, T, R) in (("sasum", Float32, Float32), ("dasum", Float64, Float64),
                   ("scasum", ComplexF32, Float32), ("dzasum", ComplexF64, Float64))
    @eval _reg!($(nm * "_"), () -> @cfunction($(Symbol(nm, "_64_")), $R, (_CI, Ptr{$T}, _CI)))
end
for (nm, T) in (("isamax", Float32), ("idamax", Float64), ("icamax", ComplexF32), ("izamax", ComplexF64))
    @eval _reg!($(nm * "_"), () -> @cfunction($(Symbol(nm, "_64_")), Int64, (_CI, Ptr{$T}, _CI)))
end
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * "gemm_"), () -> @cfunction($(Symbol(p, "gemm_64_")), Cvoid,
        (_CU, _CU, _CI, _CI, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, Clong, Clong)))
end

# ─────────────────────────────── BLAS-2 (cabi_l2.jl) ─────────────────────────────────────────────
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * "gemv_"), () -> @cfunction($(Symbol(p, "gemv_64_")), Cvoid,
        (_CU, _CI, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, Clong)))
end
for (nm, T) in (("sger", Float32), ("dger", Float64), ("cgeru", ComplexF32), ("zgeru", ComplexF64),
                ("cgerc", ComplexF32), ("zgerc", ComplexF64))
    @eval _reg!($(nm * "_"), () -> @cfunction($(Symbol(nm, "_64_")), Cvoid,
        (_CI, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI)))
end
for (nm, T) in (("ssymv", Float32), ("dsymv", Float64), ("chemv", ComplexF32), ("zhemv", ComplexF64))
    @eval _reg!($(nm * "_"), () -> @cfunction($(Symbol(nm, "_64_")), Cvoid,
        (_CU, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, Clong)))
end
for op in ("trmv", "trsv"), (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * op * "_"), () -> @cfunction($(Symbol(p, op, "_64_")), Cvoid,
        (_CU, _CU, _CU, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Clong, Clong, Clong)))
end
for (nm, T) in (("sspmv", Float32), ("dspmv", Float64), ("chpmv", ComplexF32), ("zhpmv", ComplexF64))
    @eval _reg!($(nm * "_"), () -> @cfunction($(Symbol(nm, "_64_")), Cvoid,
        (_CU, _CI, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, Clong)))
end
for op in ("tpmv", "tpsv"), (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * op * "_"), () -> @cfunction($(Symbol(p, op, "_64_")), Cvoid,
        (_CU, _CU, _CU, _CI, Ptr{$T}, Ptr{$T}, _CI, Clong, Clong, Clong)))
end
for (nm, T, AT) in (("sspr", Float32, Float32), ("dspr", Float64, Float64),
                    ("chpr", ComplexF32, Float32), ("zhpr", ComplexF64, Float64))
    @eval _reg!($(nm * "_"), () -> @cfunction($(Symbol(nm, "_64_")), Cvoid,
        (_CU, _CI, Ptr{$AT}, Ptr{$T}, _CI, Ptr{$T}, Clong)))
end
for (nm, T) in (("sspr2", Float32), ("dspr2", Float64), ("chpr2", ComplexF32), ("zhpr2", ComplexF64))
    @eval _reg!($(nm * "_"), () -> @cfunction($(Symbol(nm, "_64_")), Cvoid,
        (_CU, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, Clong)))
end
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * "gbmv_"), () -> @cfunction($(Symbol(p, "gbmv_64_")), Cvoid,
        (_CU, _CI, _CI, _CI, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, Clong)))
end
for (nm, T) in (("ssbmv", Float32), ("dsbmv", Float64), ("chbmv", ComplexF32), ("zhbmv", ComplexF64))
    @eval _reg!($(nm * "_"), () -> @cfunction($(Symbol(nm, "_64_")), Cvoid,
        (_CU, _CI, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, Clong)))
end
for op in ("tbmv", "tbsv"), (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * op * "_"), () -> @cfunction($(Symbol(p, op, "_64_")), Cvoid,
        (_CU, _CU, _CU, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Clong, Clong, Clong)))
end

# ─────────────────────────────── BLAS-3 (cabi_l3.jl) ─────────────────────────────────────────────
for (nm, T) in (("ssymm", Float32), ("dsymm", Float64), ("csymm", ComplexF32), ("zsymm", ComplexF64),
                ("chemm", ComplexF32), ("zhemm", ComplexF64))
    @eval _reg!($(nm * "_"), () -> @cfunction($(Symbol(nm, "_64_")), Cvoid,
        (_CU, _CU, _CI, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, Clong, Clong)))
end
for (nm, T, ST) in (("ssyrk", Float32, Float32), ("dsyrk", Float64, Float64),
                    ("csyrk", ComplexF32, ComplexF32), ("zsyrk", ComplexF64, ComplexF64),
                    ("cherk", ComplexF32, Float32), ("zherk", ComplexF64, Float64))
    @eval _reg!($(nm * "_"), () -> @cfunction($(Symbol(nm, "_64_")), Cvoid,
        (_CU, _CU, _CI, _CI, Ptr{$ST}, Ptr{$T}, _CI, Ptr{$ST}, Ptr{$T}, _CI, Clong, Clong)))
end
for (nm, T, ST) in (("ssyr2k", Float32, Float32), ("dsyr2k", Float64, Float64),
                    ("csyr2k", ComplexF32, ComplexF32), ("zsyr2k", ComplexF64, ComplexF64),
                    ("cher2k", ComplexF32, Float32), ("zher2k", ComplexF64, Float64))
    @eval _reg!($(nm * "_"), () -> @cfunction($(Symbol(nm, "_64_")), Cvoid,
        (_CU, _CU, _CI, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$ST}, Ptr{$T}, _CI, Clong, Clong)))
end
for op in ("trmm", "trsm"), (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * op * "_"), () -> @cfunction($(Symbol(p, op, "_64_")), Cvoid,
        (_CU, _CU, _CU, _CU, _CI, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, Clong, Clong, Clong, Clong)))
end

# ─────────────────────────────── LAPACK (cabi_lapack.jl) ─────────────────────────────────────────
# ONLY forward LAPACK symbols that are SELF-CONSISTENT under a mixed backend — i.e. correct even when
# their solve/inverse companions stay on OpenBLAS. potrf/getrf qualify: they produce standard-convention
# factors (Cholesky factor; LU L/U + LAPACK-1-based ipiv — verified identical to OpenBLAS), so OpenBLAS's
# potrs/getrs/getri/potri operate on them correctly. gesvd is self-contained (returns U/S/Vᵀ, no companion).
#
# geqrf is DELIBERATELY NOT forwarded: PureBLAS returns the faer reflector convention (τ = 1/τ_LAPACK,
# with differently-scaled v), so a forwarded geqrf feeding OpenBLAS's orgqr/ormqr produces a BROKEN Q
# (‖QᵀQ−I‖ = NaN). Julia's high-level `qr()` uses geqrt (not geqrf) anyway, so this costs no routing. To
# forward QR properly we'd need τ-compatible geqrf/geqrt wrappers + orgqr/ormqr — a follow-up.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))   # potrf: real + complex
    @eval _reg!($(p * "potrf_"), () -> @cfunction($(Symbol(p, "potrf_64_")), Cvoid,
        (_CU, _CI, Ptr{$T}, _CI, _CI, Clong)))
end
for (p, T) in (("d", Float64), ("c", ComplexF32), ("z", ComplexF64))                    # getrf: real + complex
    @eval _reg!($(p * "getrf_"), () -> @cfunction($(Symbol(p, "getrf_64_")), Cvoid,
        (_CI, _CI, Ptr{$T}, _CI, _CI, _CI)))
end
_reg!("dgesvd_", () -> @cfunction(dgesvd_64_, Cvoid,
    (_CU, _CU, _CI, _CI, Ptr{Float64}, _CI, Ptr{Float64}, Ptr{Float64}, _CI,
     Ptr{Float64}, _CI, Ptr{Float64}, _CI, _CI, Clong, Clong)))
# gesdd: Julia's svd()/svdvals route here (D&C), not gesvd — this is what makes them use PureBLAS.
_reg!("dgesdd_", () -> @cfunction(dgesdd_64_, Cvoid,
    (_CU, _CI, _CI, Ptr{Float64}, _CI, Ptr{Float64}, Ptr{Float64}, _CI,
     Ptr{Float64}, _CI, Ptr{Float64}, _CI, _CI, _CI, Clong)))
# Solves on caller-provided factors — trtrs/potrs/getrs (real + complex). getrs is the solve step of `\`.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval begin
        _reg!($(p * "trtrs_"), () -> @cfunction($(Symbol(p, "trtrs_64_")), Cvoid,
            (_CU, _CU, _CU, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI, Clong, Clong, Clong)))
        _reg!($(p * "potrs_"), () -> @cfunction($(Symbol(p, "potrs_64_")), Cvoid,
            (_CU, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI, Clong)))
        _reg!($(p * "getrs_"), () -> @cfunction($(Symbol(p, "getrs_64_")), Cvoid,
            (_CU, _CI, _CI, Ptr{$T}, _CI, _CI, Ptr{$T}, _CI, _CI, Clong)))
        _reg!($(p * "potri_"), () -> @cfunction($(Symbol(p, "potri_64_")), Cvoid,
            (_CU, _CI, Ptr{$T}, _CI, _CI, Clong)))
        _reg!($(p * "trtri_"), () -> @cfunction($(Symbol(p, "trtri_64_")), Cvoid,
            (_CU, _CU, _CI, Ptr{$T}, _CI, _CI, Clong, Clong)))
    end
end
for (p, T) in (("d", Float64), ("c", ComplexF32), ("z", ComplexF64))   # getri: matches getrf coverage (no s)
    @eval _reg!($(p * "getri_"), () -> @cfunction($(Symbol(p, "getri_64_")), Cvoid,
        (_CI, Ptr{$T}, _CI, _CI, Ptr{$T}, _CI, _CI)))
end
