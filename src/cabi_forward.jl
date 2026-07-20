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
for (p, T) in (("s", Float32), ("d", Float64))     # real dot
    @eval _reg!($(p * "dot_"), () -> @cfunction($(Symbol(p, "dot_64_")), $T,
        (_CI, Ptr{$T}, _CI, Ptr{$T}, _CI)))
end
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))   # complex dot (cabi_cdot.jl) — by-value complex return
    @eval begin
        _reg!($(p * "dotu_"), () -> @cfunction($(Symbol(p, "dotu_64_")), $T, (_CI, Ptr{$T}, _CI, Ptr{$T}, _CI)))
        _reg!($(p * "dotc_"), () -> @cfunction($(Symbol(p, "dotc_64_")), $T, (_CI, Ptr{$T}, _CI, Ptr{$T}, _CI)))
    end
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
# geqrf is now SAFE to forward: the dgeqrf_64_ wrapper converts PureBLAS's faer τ (=1/τ_LAPACK) back to
# LAPACK convention (v is already standard), so geqrf+OpenBLAS-orgqr gives a correct Q. QR routing for
# Julia's `qr()` (QRCompactWY) goes through geqrt+gemqrt — forwarded below (Float64; complex is TODO — its
# T-build needs VᴴV not VᵀV; no Float32 QR kernel exists).
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
# Complex SVD (ComplexF64 native + ComplexF32 mixed). The complex ABI inserts a REAL rwork block that the
# real gesvd/gesdd lack: gesvd → (…lwork, rwork, info); gesdd → (…lwork, rwork, iwork, info). S/rwork are
# real (Ptr{$Tr}); A/U/VT/work complex. Routes svd()/svdvals(::Matrix{Complex}) (gesdd) + QRIteration (gesvd).
for (p, T, R) in (("z", ComplexF64, Float64), ("c", ComplexF32, Float32))
    @eval _reg!($(p * "gesvd_"), () -> @cfunction($(Symbol(p, "gesvd_64_")), Cvoid,
        (_CU, _CU, _CI, _CI, Ptr{$T}, _CI, Ptr{$R}, Ptr{$T}, _CI, Ptr{$T}, _CI,
         Ptr{$T}, _CI, Ptr{$R}, _CI, Clong, Clong)))
    @eval _reg!($(p * "gesdd_"), () -> @cfunction($(Symbol(p, "gesdd_64_")), Cvoid,
        (_CU, _CI, _CI, Ptr{$T}, _CI, Ptr{$R}, Ptr{$T}, _CI, Ptr{$T}, _CI,
         Ptr{$T}, _CI, Ptr{$R}, _CI, _CI, Clong)))
end
# QR: geqrf (τ now LAPACK-converted) + geqrt/gemqrt (Float64) — routes LinearAlgebra.qr() to PureBLAS.
_reg!("dgeqrf_", () -> @cfunction(dgeqrf_64_, Cvoid,
    (_CI, _CI, Ptr{Float64}, _CI, Ptr{Float64}, Ptr{Float64}, _CI, _CI)))
for (p, T) in (("c", ComplexF32), ("s", Float32), ("z", ComplexF64))
    @eval _reg!($(p * "geqrf_"), () -> @cfunction($(Symbol(p, "geqrf_64_")), Cvoid,
        (_CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, _CI)))
end
_reg!("dgeqrt_", () -> @cfunction(dgeqrt_64_, Cvoid,
    (_CI, _CI, _CI, Ptr{Float64}, _CI, Ptr{Float64}, _CI, Ptr{Float64}, _CI)))
_reg!("dgemqrt_", () -> @cfunction(dgemqrt_64_, Cvoid,
    (_CU, _CU, _CI, _CI, _CI, _CI, Ptr{Float64}, _CI, Ptr{Float64}, _CI, Ptr{Float64}, _CI,
     Ptr{Float64}, _CI, Clong, Clong)))
# Complex QR: zgeqrt/zgemqrt (ComplexF64) + cgeqrt/cgemqrt (ComplexF32, native) route qr(::Matrix{Complex*}).
for (p, T) in (("z", ComplexF64), ("c", ComplexF32))
    @eval _reg!($(p * "geqrt_"), () -> @cfunction($(Symbol(p, "geqrt_64_")), Cvoid,
        (_CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI)))
    @eval _reg!($(p * "gemqrt_"), () -> @cfunction($(Symbol(p, "gemqrt_64_")), Cvoid,
        (_CU, _CU, _CI, _CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Clong, Clong)))
end
# Float32 LAPACK via mixed precision (compute F64, store F32): routes lu/qr/svd for Matrix{Float32}.
_reg!("sgetrf_", () -> @cfunction(sgetrf_64_, Cvoid, (_CI, _CI, Ptr{Float32}, _CI, _CI, _CI)))
_reg!("sgeqrt_", () -> @cfunction(sgeqrt_64_, Cvoid,
    (_CI, _CI, _CI, Ptr{Float32}, _CI, Ptr{Float32}, _CI, Ptr{Float32}, _CI)))
_reg!("sgemqrt_", () -> @cfunction(sgemqrt_64_, Cvoid,
    (_CU, _CU, _CI, _CI, _CI, _CI, Ptr{Float32}, _CI, Ptr{Float32}, _CI, Ptr{Float32}, _CI,
     Ptr{Float32}, _CI, Clong, Clong)))
_reg!("sgesvd_", () -> @cfunction(sgesvd_64_, Cvoid,
    (_CU, _CU, _CI, _CI, Ptr{Float32}, _CI, Ptr{Float32}, Ptr{Float32}, _CI,
     Ptr{Float32}, _CI, Ptr{Float32}, _CI, _CI, Clong, Clong)))
_reg!("sgesdd_", () -> @cfunction(sgesdd_64_, Cvoid,
    (_CU, _CI, _CI, Ptr{Float32}, _CI, Ptr{Float32}, Ptr{Float32}, _CI,
     Ptr{Float32}, _CI, Ptr{Float32}, _CI, _CI, _CI, Clong)))
# Symmetric/Hermitian eigensolver — routes eigen(Symmetric/Hermitian)/eigvals(...). The DEFAULT path is
# syevr_/heevr_ (RobustRepresentations); DivideAndConquer→syevd_/heevd_, QRIteration→syev_/heev_. Sigs
# must match the @ccallable defs in cabi_lapack.jl exactly (real: 2→2 lens syev/syevd, 3→3 syevr; complex
# adds the rwork block — heev/heevd 2 lens, heevr 3 lens). Float32 real is NATIVE; ComplexF32 (c) is mixed.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval begin
        _reg!($(p * "syev_"), () -> @cfunction($(Symbol(p, "syev_64_")), Cvoid,
            (_CU, _CU, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, _CI, Clong, Clong)))
        _reg!($(p * "syevd_"), () -> @cfunction($(Symbol(p, "syevd_64_")), Cvoid,
            (_CU, _CU, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, _CI, _CI, _CI, Clong, Clong)))
        _reg!($(p * "syevr_"), () -> @cfunction($(Symbol(p, "syevr_64_")), Cvoid,
            (_CU, _CU, _CU, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, _CI, Ptr{$T}, _CI,
             Ptr{$T}, Ptr{$T}, _CI, _CI, Ptr{$T}, _CI, _CI, _CI, _CI, Clong, Clong, Clong)))
    end
end
for (p, Tc, Tr) in (("z", ComplexF64, Float64), ("c", ComplexF32, Float32))
    @eval begin
        _reg!($(p * "heev_"), () -> @cfunction($(Symbol(p, "heev_64_")), Cvoid,
            (_CU, _CU, _CI, Ptr{$Tc}, _CI, Ptr{$Tr}, Ptr{$Tc}, _CI, Ptr{$Tr}, _CI, Clong, Clong)))
        _reg!($(p * "heevd_"), () -> @cfunction($(Symbol(p, "heevd_64_")), Cvoid,
            (_CU, _CU, _CI, Ptr{$Tc}, _CI, Ptr{$Tr}, Ptr{$Tc}, _CI, Ptr{$Tr}, _CI, _CI, _CI, _CI, Clong, Clong)))
        _reg!($(p * "heevr_"), () -> @cfunction($(Symbol(p, "heevr_64_")), Cvoid,
            (_CU, _CU, _CU, _CI, Ptr{$Tc}, _CI, Ptr{$Tr}, Ptr{$Tr}, _CI, _CI, Ptr{$Tr}, _CI,
             Ptr{$Tr}, Ptr{$Tc}, _CI, _CI, Ptr{$Tc}, _CI, Ptr{$Tr}, _CI, _CI, _CI, _CI, Clong, Clong, Clong)))
    end
end
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
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))   # getri (s native via generic trsm!)
    @eval _reg!($(p * "getri_"), () -> @cfunction($(Symbol(p, "getri_64_")), Cvoid,
        (_CI, Ptr{$T}, _CI, _CI, Ptr{$T}, _CI, _CI)))
end

# ─────────── Batch 6: LQ / Bunch-Kaufman / geqp3 / gels / gecon·trcon·pocon / hessenberg ───────────
# All self-consistent under a mixed backend (standard-LAPACK τ / ipiv encoding / factor storage), so an
# OpenBLAS companion (orgqr/ormhr/sytrs) still works on the forwarded factors. @cfunction sigs match the
# @ccallable defs in cabi_lapack.jl byte-for-byte (arg tuple + trailing Clong count per char arg).
# LQ — routes lq(A) (gelqf), Matrix(lq(A).Q) (orglq/unglq), lq(A).Q ops (ormlq/unmlq).
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * "gelqf_"), () -> @cfunction($(Symbol(p, "gelqf_64_")), Cvoid,
        (_CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, _CI)))
end
for (nm, T) in (("sorglq", Float32), ("dorglq", Float64), ("cunglq", ComplexF32), ("zunglq", ComplexF64))
    @eval _reg!($(nm * "_"), () -> @cfunction($(Symbol(nm, "_64_")), Cvoid,
        (_CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, _CI)))
end
for (nm, T) in (("sormlq", Float32), ("dormlq", Float64), ("cunmlq", ComplexF32), ("zunmlq", ComplexF64))
    @eval _reg!($(nm * "_"), () -> @cfunction($(Symbol(nm, "_64_")), Cvoid,
        (_CU, _CU, _CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI, Clong, Clong)))
end
# Bunch-Kaufman — routes bunchkaufman(Symmetric/Hermitian) (sytrf/hetrf) + its `\` solve (sytrs/hetrs).
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval begin
        _reg!($(p * "sytrf_"), () -> @cfunction($(Symbol(p, "sytrf_64_")), Cvoid,
            (_CU, _CI, Ptr{$T}, _CI, _CI, Ptr{$T}, _CI, _CI, Clong)))
        _reg!($(p * "sytrs_"), () -> @cfunction($(Symbol(p, "sytrs_64_")), Cvoid,
            (_CU, _CI, _CI, Ptr{$T}, _CI, _CI, Ptr{$T}, _CI, _CI, Clong)))
    end
end
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))     # hetrf/hetrs complex only
    @eval begin
        _reg!($(p * "hetrf_"), () -> @cfunction($(Symbol(p, "hetrf_64_")), Cvoid,
            (_CU, _CI, Ptr{$T}, _CI, _CI, Ptr{$T}, _CI, _CI, Clong)))
        _reg!($(p * "hetrs_"), () -> @cfunction($(Symbol(p, "hetrs_64_")), Cvoid,
            (_CU, _CI, _CI, Ptr{$T}, _CI, _CI, Ptr{$T}, _CI, _CI, Clong)))
    end
end
# geqp3 — routes qr(A, ColumnNorm()) and non-square A\b. Complex inserts a REAL rwork block.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval _reg!($(p * "geqp3_"), () -> @cfunction($(Symbol(p, "geqp3_64_")), Cvoid,
        (_CI, _CI, Ptr{$T}, _CI, _CI, Ptr{$T}, Ptr{$T}, _CI, _CI)))
end
for (p, T, R) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval _reg!($(p * "geqp3_"), () -> @cfunction($(Symbol(p, "geqp3_64_")), Cvoid,
        (_CI, _CI, Ptr{$T}, _CI, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$R}, _CI)))
end
# gels — direct LAPACK.gels! least-squares (solution in B); non-square `\` uses geqp3 above.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * "gels_"), () -> @cfunction($(Symbol(p, "gels_64_")), Cvoid,
        (_CU, _CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI, Clong)))
end
# gecon/trcon/pocon — condition estimates. Complex carries a REAL rwork where real carries integer iwork.
# trcon routes cond(::Triangular, 1/Inf); gecon/pocon are direct-call/non-Julia-host symbols.
for (p, T, R, IW) in (("s", Float32, Float32, Int64), ("d", Float64, Float64, Int64),
                      ("c", ComplexF32, Float32, Float32), ("z", ComplexF64, Float64, Float64))
    @eval begin
        _reg!($(p * "gecon_"), () -> @cfunction($(Symbol(p, "gecon_64_")), Cvoid,
            (_CU, _CI, Ptr{$T}, _CI, Ptr{$R}, Ptr{$R}, Ptr{$T}, Ptr{$IW}, _CI, Clong)))
        _reg!($(p * "trcon_"), () -> @cfunction($(Symbol(p, "trcon_64_")), Cvoid,
            (_CU, _CU, _CU, _CI, Ptr{$T}, _CI, Ptr{$R}, Ptr{$T}, Ptr{$IW}, _CI, Clong, Clong, Clong)))
        _reg!($(p * "pocon_"), () -> @cfunction($(Symbol(p, "pocon_64_")), Cvoid,
            (_CU, _CI, Ptr{$T}, _CI, Ptr{$R}, Ptr{$R}, Ptr{$T}, Ptr{$IW}, _CI, Clong)))
    end
end
# Hessenberg — routes hessenberg(A) (gehrd) + gebal/orghr·unghr reductions.
for (p, T, R) in (("s", Float32, Float32), ("d", Float64, Float64),
                  ("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval _reg!($(p * "gebal_"), () -> @cfunction($(Symbol(p, "gebal_64_")), Cvoid,
        (_CU, _CI, Ptr{$T}, _CI, _CI, _CI, Ptr{$R}, _CI, Clong)))
end
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * "gehrd_"), () -> @cfunction($(Symbol(p, "gehrd_64_")), Cvoid,
        (_CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, _CI)))
end
for (nm, T) in (("sorghr", Float32), ("dorghr", Float64), ("cunghr", ComplexF32), ("zunghr", ComplexF64))
    @eval _reg!($(nm * "_"), () -> @cfunction($(Symbol(nm, "_64_")), Cvoid,
        (_CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, _CI)))
end

# ─────────── General nonsymmetric eigensolver: geev / geevx / gees — routes eigen/eigvals/schur ───────
# eigen(A)/eigvals(A) call LAPACK.geevx! (sense='N'); schur(A) calls gees!('V',A). geev is registered too
# (plain driver + non-Julia hosts). @cfunction sigs match the @ccallable defs in cabi_lapack.jl exactly:
# real geev has separate wr/wi outputs; complex geev has one w + a REAL rwork block. geevx adds balanc/
# sense chars (4 char args → 4 lens) + ilo/ihi/scale/abnrm/rconde/rcondv. gees carries select/bwork
# function-pointer args (Ptr{Cvoid}, ignored) + sdim.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval begin
        _reg!($(p * "geev_"), () -> @cfunction($(Symbol(p, "geev_64_")), Cvoid,
            (_CU, _CU, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI,
             Ptr{$T}, _CI, _CI, Clong, Clong)))
        _reg!($(p * "geevx_"), () -> @cfunction($(Symbol(p, "geevx_64_")), Cvoid,
            (_CU, _CU, _CU, _CU, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI,
             _CI, _CI, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI, _CI, _CI,
             Clong, Clong, Clong, Clong)))
        _reg!($(p * "gees_"), () -> @cfunction($(Symbol(p, "gees_64_")), Cvoid,
            (_CU, _CU, Ptr{Cvoid}, _CI, Ptr{$T}, _CI, _CI, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI,
             Ptr{$T}, _CI, Ptr{Cvoid}, _CI, Clong, Clong)))
    end
end
for (p, T, R) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval begin
        _reg!($(p * "geev_"), () -> @cfunction($(Symbol(p, "geev_64_")), Cvoid,
            (_CU, _CU, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI,
             Ptr{$T}, _CI, Ptr{$R}, _CI, Clong, Clong)))
        _reg!($(p * "geevx_"), () -> @cfunction($(Symbol(p, "geevx_64_")), Cvoid,
            (_CU, _CU, _CU, _CU, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI,
             _CI, _CI, Ptr{$R}, Ptr{$R}, Ptr{$R}, Ptr{$R}, Ptr{$T}, _CI, Ptr{$R}, _CI,
             Clong, Clong, Clong, Clong)))
        _reg!($(p * "gees_"), () -> @cfunction($(Symbol(p, "gees_64_")), Cvoid,
            (_CU, _CU, Ptr{Cvoid}, _CI, Ptr{$T}, _CI, _CI, Ptr{$T}, Ptr{$T}, _CI,
             Ptr{$T}, _CI, Ptr{$R}, Ptr{Cvoid}, _CI, Clong, Clong)))
    end
end

# ─────────── Assembly batch: generalized eigen + tridiagonal + band/packed Cholesky ───────────────
# Routes: eigen(Sym,Sym)→sygvd/hegvd; eigen(A,B)/eigvals(A,B)→ggev3; schur(A,B)→gges3; eigen/eigvals(
# SymTridiagonal)→stegr/stev. gtsv/gttrf/gttrs + pbtrf/pbtrs + pptrf/pptrs serve DIRECT LAPACK callers
# and external packages (base LA's Tridiagonal\b is native-Julia, not gtsv). @cfunction sigs match the
# @ccallable defs in cabi_lapack.jl byte-for-byte (arg tuple + trailing Clong per char arg).
# sygvd/hegvd — eigen(Symmetric,Symmetric)/eigvals(Symmetric,Symmetric). Complex ref name is hegvd.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval _reg!($(p * "sygvd_"), () -> @cfunction($(Symbol(p, "sygvd_64_")), Cvoid,
        (_CI, _CU, _CU, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, _CI, _CI, _CI, Clong, Clong)))
end
for (p, Tc, Tr) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval _reg!($(p * "hegvd_"), () -> @cfunction($(Symbol(p, "hegvd_64_")), Cvoid,
        (_CI, _CU, _CU, _CI, Ptr{$Tc}, _CI, Ptr{$Tc}, _CI, Ptr{$Tr}, Ptr{$Tc}, _CI, Ptr{$Tr}, _CI, _CI, _CI, _CI, Clong, Clong)))
end
# gtsv / gttrf / gttrs — tridiagonal solve/factor (direct LAPACK callers).
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval begin
        _reg!($(p * "gtsv_"), () -> @cfunction($(Symbol(p, "gtsv_64_")), Cvoid,
            (_CI, _CI, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI, _CI)))
        _reg!($(p * "gttrf_"), () -> @cfunction($(Symbol(p, "gttrf_64_")), Cvoid,
            (_CI, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI, _CI)))
        _reg!($(p * "gttrs_"), () -> @cfunction($(Symbol(p, "gttrs_64_")), Cvoid,
            (_CU, _CI, _CI, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI, Clong)))
    end
end
# stev / stegr — SymTridiagonal eigensolver (real only). eigvals→stev('N'); eigen→stegr('V','A').
for (p, T) in (("s", Float32), ("d", Float64))
    @eval begin
        _reg!($(p * "stev_"), () -> @cfunction($(Symbol(p, "stev_64_")), Cvoid,
            (_CU, _CI, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, Clong)))
        _reg!($(p * "stegr_"), () -> @cfunction($(Symbol(p, "stegr_64_")), Cvoid,
            (_CU, _CU, _CI, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T},
             _CI, _CI, Ptr{$T}, _CI, _CI, _CI, _CI, Clong, Clong)))
    end
end
# pbtrf / pbtrs (band) + pptrf / pptrs (packed) — direct/external callers (BandedMatrices, packed storage).
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval begin
        _reg!($(p * "pbtrf_"), () -> @cfunction($(Symbol(p, "pbtrf_64_")), Cvoid,
            (_CU, _CI, _CI, Ptr{$T}, _CI, _CI, Clong)))
        _reg!($(p * "pbtrs_"), () -> @cfunction($(Symbol(p, "pbtrs_64_")), Cvoid,
            (_CU, _CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI, Clong)))
        _reg!($(p * "pptrf_"), () -> @cfunction($(Symbol(p, "pptrf_64_")), Cvoid,
            (_CU, _CI, Ptr{$T}, _CI, Clong)))
        _reg!($(p * "pptrs_"), () -> @cfunction($(Symbol(p, "pptrs_64_")), Cvoid,
            (_CU, _CI, _CI, Ptr{$T}, Ptr{$T}, _CI, _CI, Clong)))
    end
end
# ggev / ggev3 — eigen(A,B)/eigvals(A,B) (Julia≥3.6 calls ggev3; register both). REAL alphar/alphai/beta;
# COMPLEX one alpha + a REAL rwork block.
for (p, T) in (("s", Float32), ("d", Float64)), nm in ("ggev", "ggev3")
    @eval _reg!($(p * nm * "_"), () -> @cfunction($(Symbol(p, nm, "_64_")), Cvoid,
        (_CU, _CU, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI,
         Ptr{$T}, _CI, _CI, Clong, Clong)))
end
for (p, T, R) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64)), nm in ("ggev", "ggev3")
    @eval _reg!($(p * nm * "_"), () -> @cfunction($(Symbol(p, nm, "_64_")), Cvoid,
        (_CU, _CU, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T},
         _CI, Ptr{$R}, _CI, Clong, Clong)))
end
# gges / gges3 — schur(A,B) (Julia≥3.6 calls gges3; register both). select/bwork are Ptr{Cvoid}. REAL
# alphar/alphai; COMPLEX one alpha + a REAL rwork block. 3 char args → 3 trailing Clongs.
for (p, T) in (("s", Float32), ("d", Float64)), nm in ("gges", "gges3")
    @eval _reg!($(p * nm * "_"), () -> @cfunction($(Symbol(p, nm, "_64_")), Cvoid,
        (_CU, _CU, _CU, Ptr{Cvoid}, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI, Ptr{$T}, Ptr{$T}, Ptr{$T},
         Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{Cvoid}, _CI, Clong, Clong, Clong)))
end
for (p, T, R) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64)), nm in ("gges", "gges3")
    @eval _reg!($(p * nm * "_"), () -> @cfunction($(Symbol(p, nm, "_64_")), Cvoid,
        (_CU, _CU, _CU, Ptr{Cvoid}, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI, Ptr{$T}, Ptr{$T}, Ptr{$T},
         _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$R}, Ptr{Cvoid}, _CI, Clong, Clong, Clong)))
end

# ─────────── Assembly batch 2: banded LU, SPD tridiagonal, sym-tridiag bisection/inverse-iteration,
# pivoted Cholesky, QL/RQ, RZ least-squares, SVD least-squares, symmetric-indefinite solve/inverse,
# Sylvester, Schur reorder, equality-constrained LS, generalized SVD (Float64). Sigs match the
# @ccallable defs in cabi_lapack.jl byte-for-byte (arg tuple + trailing Clong count per char arg).
# gbtrf / gbtrs — general banded LU (direct/external callers; not reached via a high-level Julia call).
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval begin
        _reg!($(p * "gbtrf_"), () -> @cfunction($(Symbol(p, "gbtrf_64_")), Cvoid,
            (_CI, _CI, _CI, _CI, Ptr{$T}, _CI, Ptr{Int64}, _CI)))
        _reg!($(p * "gbtrs_"), () -> @cfunction($(Symbol(p, "gbtrs_64_")), Cvoid,
            (_CU, _CI, _CI, _CI, _CI, Ptr{$T}, _CI, Ptr{Int64}, Ptr{$T}, _CI, _CI, Clong)))
    end
end
# ptsv / pttrf / pttrs — SPD/Hermitian-PD tridiagonal (direct/external callers).
for (p, T, Tr) in (("s", Float32, Float32), ("d", Float64, Float64),
                   ("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval begin
        _reg!($(p * "ptsv_"), () -> @cfunction($(Symbol(p, "ptsv_64_")), Cvoid,
            (_CI, _CI, Ptr{$Tr}, Ptr{$T}, Ptr{$T}, _CI, _CI)))
        _reg!($(p * "pttrf_"), () -> @cfunction($(Symbol(p, "pttrf_64_")), Cvoid,
            (_CI, Ptr{$Tr}, Ptr{$T}, _CI)))
    end
end
for (p, T) in (("s", Float32), ("d", Float64))
    @eval _reg!($(p * "pttrs_"), () -> @cfunction($(Symbol(p, "pttrs_64_")), Cvoid,
        (_CI, _CI, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI, _CI)))
end
for (p, T, Tr) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval _reg!($(p * "pttrs_"), () -> @cfunction($(Symbol(p, "pttrs_64_")), Cvoid,
        (_CU, _CI, _CI, Ptr{$Tr}, Ptr{$T}, Ptr{$T}, _CI, _CI, Clong)))
end
# stebz / stein — real-symmetric-tridiag eigenvalues/eigenvectors (direct/external callers).
for (p, T) in (("s", Float32), ("d", Float64))
    @eval begin
        _reg!($(p * "stebz_"), () -> @cfunction($(Symbol(p, "stebz_64_")), Cvoid,
            (_CU, _CU, _CI, Ptr{$T}, Ptr{$T}, _CI, _CI, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI, _CI,
             Ptr{$T}, Ptr{Int64}, Ptr{Int64}, Ptr{$T}, Ptr{Int64}, _CI, Clong, Clong)))
        _reg!($(p * "stein_"), () -> @cfunction($(Symbol(p, "stein_64_")), Cvoid,
            (_CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, Ptr{Int64}, Ptr{Int64}, Ptr{$T}, _CI,
             Ptr{$T}, Ptr{Int64}, Ptr{Int64}, _CI)))
    end
end
# pstrf — pivoted (rank-revealing) Cholesky (direct/external callers; backs cholesky(A, RowMaximum())).
for (p, T, Tr) in (("s", Float32, Float32), ("d", Float64, Float64),
                   ("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval _reg!($(p * "pstrf_"), () -> @cfunction($(Symbol(p, "pstrf_64_")), Cvoid,
        (_CU, _CI, Ptr{$T}, _CI, Ptr{Int64}, Ptr{Int64}, Ptr{$Tr}, Ptr{$Tr}, _CI, Clong)))
end
# QL / RQ — routes ql(A)/rq(A)-style direct callers (no high-level LinearAlgebra wrapper for QL/RQ).
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval begin
        _reg!($(p * "geqlf_"), () -> @cfunction($(Symbol(p, "geqlf_64_")), Cvoid,
            (_CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, _CI)))
        _reg!($(p * "gerqf_"), () -> @cfunction($(Symbol(p, "gerqf_64_")), Cvoid,
            (_CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, _CI)))
    end
end
for (nm, T) in (("sorgql", Float32), ("dorgql", Float64), ("cungql", ComplexF32), ("zungql", ComplexF64))
    @eval _reg!($(nm * "_"), () -> @cfunction($(Symbol(nm, "_64_")), Cvoid,
        (_CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, _CI)))
end
for (nm, T) in (("sorgrq", Float32), ("dorgrq", Float64), ("cungrq", ComplexF32), ("zungrq", ComplexF64))
    @eval _reg!($(nm * "_"), () -> @cfunction($(Symbol(nm, "_64_")), Cvoid,
        (_CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, _CI)))
end
for (nm, T) in (("sormql", Float32), ("dormql", Float64), ("cunmql", ComplexF32), ("zunmql", ComplexF64))
    @eval _reg!($(nm * "_"), () -> @cfunction($(Symbol(nm, "_64_")), Cvoid,
        (_CU, _CU, _CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI, Clong, Clong)))
end
for (nm, T) in (("sormrq", Float32), ("dormrq", Float64), ("cunmrq", ComplexF32), ("zunmrq", ComplexF64))
    @eval _reg!($(nm * "_"), () -> @cfunction($(Symbol(nm, "_64_")), Cvoid,
        (_CU, _CU, _CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI, Clong, Clong)))
end
# tzrzf / ormrz(unmrz) — the "complete orthogonal" half of gelsy (direct/external callers).
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * "tzrzf_"), () -> @cfunction($(Symbol(p, "tzrzf_64_")), Cvoid,
        (_CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, _CI)))
end
for (nm, T) in (("sormrz", Float32), ("dormrz", Float64), ("cunmrz", ComplexF32), ("zunmrz", ComplexF64))
    @eval _reg!($(nm * "_"), () -> @cfunction($(Symbol(nm, "_64_")), Cvoid,
        (_CU, _CU, _CI, _CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI,
         Clong, Clong)))
end
# gelsy — routes non-square `\` least-squares (rank-deficient path, RZ factorization).
for (p, T) in (("s", Float32), ("d", Float64))
    @eval _reg!($(p * "gelsy_"), () -> @cfunction($(Symbol(p, "gelsy_64_")), Cvoid,
        (_CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{Int64}, Ptr{$T}, Ptr{Int64}, Ptr{$T}, _CI, _CI)))
end
for (p, T, R) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval _reg!($(p * "gelsy_"), () -> @cfunction($(Symbol(p, "gelsy_64_")), Cvoid,
        (_CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{Int64}, Ptr{$R}, Ptr{Int64}, Ptr{$T}, _CI,
         Ptr{$R}, _CI)))
end
# gelsd — rank-deficient least squares via SVD (direct LAPACK.gelsd! callers).
for (p, T) in (("s", Float32), ("d", Float64))
    @eval _reg!($(p * "gelsd_"), () -> @cfunction($(Symbol(p, "gelsd_64_")), Cvoid,
        (_CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, Ptr{Int64}, Ptr{$T}, _CI,
         Ptr{Int64}, _CI)))
end
for (p, T, R) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval _reg!($(p * "gelsd_"), () -> @cfunction($(Symbol(p, "gelsd_64_")), Cvoid,
        (_CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$R}, Ptr{$R}, Ptr{Int64}, Ptr{$T}, _CI,
         Ptr{$R}, Ptr{Int64}, _CI)))
end
# sysv / hesv — one-shot symmetric-indefinite / Hermitian solve (backs `\` on Symmetric/Hermitian, and
# direct callers). sytri / hetri — matrix inverse from the Bunch-Kaufman factors (direct callers).
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval begin
        _reg!($(p * "sysv_"), () -> @cfunction($(Symbol(p, "sysv_64_")), Cvoid,
            (_CU, _CI, _CI, Ptr{$T}, _CI, Ptr{Int64}, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI, Clong)))
        _reg!($(p * "sytri_"), () -> @cfunction($(Symbol(p, "sytri_64_")), Cvoid,
            (_CU, _CI, Ptr{$T}, _CI, Ptr{Int64}, Ptr{$T}, _CI, Clong)))
    end
end
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))
    @eval begin
        _reg!($(p * "hesv_"), () -> @cfunction($(Symbol(p, "hesv_64_")), Cvoid,
            (_CU, _CI, _CI, Ptr{$T}, _CI, Ptr{Int64}, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI, Clong)))
        _reg!($(p * "hetri_"), () -> @cfunction($(Symbol(p, "hetri_64_")), Cvoid,
            (_CU, _CI, Ptr{$T}, _CI, Ptr{Int64}, Ptr{$T}, _CI, Clong)))
    end
end
# trsyl — triangular Sylvester solve (backs sylvester/lyap; direct callers).
for (p, T, Tr) in (("s", Float32, Float32), ("d", Float64, Float64),
                   ("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval _reg!($(p * "trsyl_"), () -> @cfunction($(Symbol(p, "trsyl_64_")), Cvoid,
        (_CU, _CU, _CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$Tr}, _CI, Clong, Clong)))
end
# trexc / trsen — Schur reordering (backs ordschur; direct callers).
for (p, T) in (("s", Float32), ("d", Float64))
    @eval begin
        _reg!($(p * "trexc_"), () -> @cfunction($(Symbol(p, "trexc_64_")), Cvoid,
            (_CU, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI, _CI, Ptr{$T}, _CI, Clong)))
        _reg!($(p * "trsen_"), () -> @cfunction($(Symbol(p, "trsen_64_")), Cvoid,
            (_CU, _CU, Ptr{Int64}, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T},
             Ptr{$T}, Ptr{$T}, _CI, Ptr{Int64}, _CI, _CI, Clong, Clong)))
    end
end
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * "trexc_"), () -> @cfunction($(Symbol(p, "trexc_64_")), Cvoid,
        (_CU, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI, _CI, _CI, Clong)))
end
for (p, T, R) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval _reg!($(p * "trsen_"), () -> @cfunction($(Symbol(p, "trsen_64_")), Cvoid,
        (_CU, _CU, Ptr{Int64}, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$R}, Ptr{$R},
         Ptr{$T}, _CI, _CI, Clong, Clong)))
end
# gglse — equality-constrained least squares (direct LAPACK.gglse! callers).
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * "gglse_"), () -> @cfunction($(Symbol(p, "gglse_64_")), Cvoid,
        (_CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI, _CI)))
end
# ggsvd / ggsvd3 — generalized SVD, Float64 full-rank ONLY (direct LAPACK.ggsvd!/ggsvd3! callers).
_reg!("dggsvd_", () -> @cfunction(dggsvd_64_, Cvoid,
    (_CU, _CU, _CU, _CI, _CI, _CI, Ptr{Int64}, Ptr{Int64}, Ptr{Float64}, _CI, Ptr{Float64}, _CI,
     Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, _CI, Ptr{Float64}, _CI, Ptr{Float64}, _CI,
     Ptr{Float64}, Ptr{Int64}, _CI, Clong, Clong, Clong)))
_reg!("dggsvd3_", () -> @cfunction(dggsvd3_64_, Cvoid,
    (_CU, _CU, _CU, _CI, _CI, _CI, Ptr{Int64}, Ptr{Int64}, Ptr{Float64}, _CI, Ptr{Float64}, _CI,
     Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, _CI, Ptr{Float64}, _CI, Ptr{Float64}, _CI,
     Ptr{Float64}, _CI, Ptr{Int64}, _CI, Clong, Clong, Clong)))

# ═══════════ Batch 2 (cabi_lapack2.jl): gesv/posv/lacpy/larfg/larf, gebak/hseqr/trevc, sytrd·hetrd/
# orgtr·ungtr/ormtr·unmtr, orgqr·ungqr/ormqr·unmqr, ormhr·unmhr, gebrd/bdsqr/bdsdc — the OpenBLAS-removal
# ratchet follow-up. @cfunction sigs match the @ccallable defs in cabi_lapack2.jl byte-for-byte.
# gesv — composes getrf!+getrs (routes any DIRECT LAPACK.gesv! caller; s is mixed-precision).
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * "gesv_"), () -> @cfunction($(Symbol(p, "gesv_64_")), Cvoid,
        (_CI, _CI, Ptr{$T}, _CI, _CI, Ptr{$T}, _CI, _CI)))
end
# posv — composes potrf!+potrs (native for all 4 types).
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * "posv_"), () -> @cfunction($(Symbol(p, "posv_64_")), Cvoid,
        (_CU, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI, Clong)))
end
# lacpy — triangle/full copy. No `info` arg (matches the real Fortran ABI).
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * "lacpy_"), () -> @cfunction($(Symbol(p, "lacpy_64_")), Cvoid,
        (_CU, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Clong)))
end
# larfg — Householder reflector generator. 0 chars; alpha is a SEPARATE by-ref scalar from x.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * "larfg_"), () -> @cfunction($(Symbol(p, "larfg_64_")), Cvoid,
        (_CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T})))
end
# larf — apply a Householder reflector.
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * "larf_"), () -> @cfunction($(Symbol(p, "larf_64_")), Cvoid,
        (_CU, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, Clong)))
end
# syconv — Bunch-Kaufman factorization convert (cabi_lapack3.jl). 2 chars (uplo, way).
for (p, T) in (("s", Float32), ("d", Float64), ("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * "syconv_"), () -> @cfunction($(Symbol(p, "syconv_64_")), Cvoid,
        (_CU, _CU, _CI, Ptr{$T}, _CI, _CI, Ptr{$T}, _CI, Clong, Clong)))
end
# trrfs — triangular-solve error bounds (cabi_lapack3.jl). 3 chars (uplo, trans, diag).
for (p, T) in (("s", Float32), ("d", Float64))     # real: Ferr/Berr/work real, iwork Int
    @eval _reg!($(p * "trrfs_"), () -> @cfunction($(Symbol(p, "trrfs_64_")), Cvoid,
        (_CU, _CU, _CU, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI,
         Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI, _CI, Clong, Clong, Clong)))
end
for (p, T, Tr) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))  # complex: Ferr/Berr/rwork real
    @eval _reg!($(p * "trrfs_"), () -> @cfunction($(Symbol(p, "trrfs_64_")), Cvoid,
        (_CU, _CU, _CU, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI,
         Ptr{$Tr}, Ptr{$Tr}, Ptr{$T}, Ptr{$Tr}, _CI, Clong, Clong, Clong)))
end
# tgsen (cabi_lapack3.jl). Integer ijob/wantq/wantz (no chars → no Clong); pl/pr/dif Ptr{Cvoid}. COMPLEX
# has alpha/beta (2 vecs); REAL has alphar/alphai/beta (3 vecs).
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * "tgsen_"), () -> @cfunction($(Symbol(p, "tgsen_64_")), Cvoid,
        (_CI, _CI, _CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI,
         Ptr{$T}, _CI, _CI, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{$T}, _CI, _CI, _CI, _CI)))
end
for (p, T) in (("s", Float32), ("d", Float64))   # real: alphar, alphai, beta (3× Ptr{T})
    @eval _reg!($(p * "tgsen_"), () -> @cfunction($(Symbol(p, "tgsen_64_")), Cvoid,
        (_CI, _CI, _CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI,
         Ptr{$T}, _CI, _CI, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{$T}, _CI, _CI, _CI, _CI)))
end
# gesvx (cabi_lapack3.jl). 3 chars (fact/trans/equed → 3 Clong). REAL: R/C/rcond/ferr/berr/work all real,
# trailing iwork. COMPLEX: those five REAL (Ptr{Tr}), trailing rwork (Ptr{Tr}); work is complex (Ptr{T}).
for (p, T) in (("s", Float32), ("d", Float64))   # real: Tr = T; slot before info is iwork (_CI)
    @eval _reg!($(p * "gesvx_"), () -> @cfunction($(Symbol(p, "gesvx_64_")), Cvoid,
        (_CU, _CU, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI, _CU, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI,
         Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI, _CI, Clong, Clong, Clong)))
end
for (p, T, Tr) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))  # complex: real R/C/rcond/ferr/berr + rwork
    @eval _reg!($(p * "gesvx_"), () -> @cfunction($(Symbol(p, "gesvx_64_")), Cvoid,
        (_CU, _CU, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI, _CU, Ptr{$Tr}, Ptr{$Tr}, Ptr{$T}, _CI,
         Ptr{$T}, _CI, Ptr{$Tr}, Ptr{$Tr}, Ptr{$Tr}, Ptr{$T}, Ptr{$Tr}, _CI, Clong, Clong, Clong)))
end
# bdsqr (COMPLEX — cabi_lapack3.jl). d/e/rwork REAL; Vt/U/C complex. 1 char (uplo → 1 Clong).
for (p, T, Tr) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval _reg!($(p * "bdsqr_"), () -> @cfunction($(Symbol(p, "bdsqr_64_")), Cvoid,
        (_CU, _CI, _CI, _CI, _CI, Ptr{$Tr}, Ptr{$Tr}, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI,
         Ptr{$Tr}, _CI, Clong)))
end
# gebak — undo gebal's balancing on eigen/Schur vectors.
for (p, T, Tr) in (("s", Float32, Float32), ("d", Float64, Float64),
                   ("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval _reg!($(p * "gebak_"), () -> @cfunction($(Symbol(p, "gebak_64_")), Cvoid,
        (_CU, _CU, _CI, _CI, _CI, Ptr{$Tr}, _CI, Ptr{$T}, _CI, _CI, Clong, Clong)))
end
# hseqr — Schur factorization of upper-Hessenberg H. REAL has separate wr/wi; COMPLEX one w + no rwork.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval _reg!($(p * "hseqr_"), () -> @cfunction($(Symbol(p, "hseqr_64_")), Cvoid,
        (_CU, _CU, _CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI,
         Clong, Clong)))
end
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * "hseqr_"), () -> @cfunction($(Symbol(p, "hseqr_64_")), Cvoid,
        (_CU, _CU, _CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI, Clong, Clong)))
end
# trevc — right eigenvectors of a Schur form by back-substitution (side='R' only; select is Ptr{Int64}).
for (p, T) in (("s", Float32), ("d", Float64))
    @eval _reg!($(p * "trevc_"), () -> @cfunction($(Symbol(p, "trevc_64_")), Cvoid,
        (_CU, _CU, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI, _CI, Ptr{$T}, _CI,
         Clong, Clong)))
end
for (p, T, R) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval _reg!($(p * "trevc_"), () -> @cfunction($(Symbol(p, "trevc_64_")), Cvoid,
        (_CU, _CU, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI, _CI, Ptr{$T}, Ptr{$R}, _CI,
         Clong, Clong)))
end
# sytrd (real) / hetrd (complex) — tridiagonalize symmetric/Hermitian A. uplo='L' only (kernel scope).
for (p, T) in (("s", Float32), ("d", Float64))
    @eval _reg!($(p * "sytrd_"), () -> @cfunction($(Symbol(p, "sytrd_64_")), Cvoid,
        (_CU, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI, _CI, Clong)))
end
for (p, T, R) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval _reg!($(p * "hetrd_"), () -> @cfunction($(Symbol(p, "hetrd_64_")), Cvoid,
        (_CU, _CI, Ptr{$T}, _CI, Ptr{$R}, Ptr{$R}, Ptr{$T}, Ptr{$T}, _CI, _CI, Clong)))
end
# orgtr (real) / ungtr (complex) — form Q from sytrd/hetrd reflectors. uplo='L' only.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval _reg!($(p * "orgtr_"), () -> @cfunction($(Symbol(p, "orgtr_64_")), Cvoid,
        (_CU, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, _CI, Clong)))
end
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * "ungtr_"), () -> @cfunction($(Symbol(p, "ungtr_64_")), Cvoid,
        (_CU, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, _CI, Clong)))
end
# ormtr (real) / unmtr (complex) — apply Q from sytrd/hetrd reflectors. side='L' + uplo='L' only.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval _reg!($(p * "ormtr_"), () -> @cfunction($(Symbol(p, "ormtr_64_")), Cvoid,
        (_CU, _CU, _CU, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI,
         Clong, Clong, Clong)))
end
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * "unmtr_"), () -> @cfunction($(Symbol(p, "unmtr_64_")), Cvoid,
        (_CU, _CU, _CU, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI,
         Clong, Clong, Clong)))
end
# orgqr (real) / ungqr (complex) — form Q from geqrf reflectors. 0 chars.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval _reg!($(p * "orgqr_"), () -> @cfunction($(Symbol(p, "orgqr_64_")), Cvoid,
        (_CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, _CI)))
end
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * "ungqr_"), () -> @cfunction($(Symbol(p, "ungqr_64_")), Cvoid,
        (_CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, _CI)))
end
# ormqr (real) / unmqr (complex) — apply Q from geqrf reflectors.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval _reg!($(p * "ormqr_"), () -> @cfunction($(Symbol(p, "ormqr_64_")), Cvoid,
        (_CU, _CU, _CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI, Clong, Clong)))
end
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * "unmqr_"), () -> @cfunction($(Symbol(p, "unmqr_64_")), Cvoid,
        (_CU, _CU, _CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI, Clong, Clong)))
end
# ormhr (real) / unmhr (complex) — apply Q from gehrd reflectors.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval _reg!($(p * "ormhr_"), () -> @cfunction($(Symbol(p, "ormhr_64_")), Cvoid,
        (_CU, _CU, _CI, _CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI,
         Clong, Clong)))
end
for (p, T) in (("c", ComplexF32), ("z", ComplexF64))
    @eval _reg!($(p * "unmhr_"), () -> @cfunction($(Symbol(p, "unmhr_64_")), Cvoid,
        (_CU, _CU, _CI, _CI, _CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI,
         Clong, Clong)))
end
# gebrd — bidiagonalize A (Golub-Kahan reduction). m≥n only (kernel scope). 0 chars.
for (p, T) in (("s", Float32), ("d", Float64))
    @eval _reg!($(p * "gebrd_"), () -> @cfunction($(Symbol(p, "gebrd_64_")), Cvoid,
        (_CI, _CI, Ptr{$T}, _CI, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI, _CI)))
end
for (p, T, R) in (("c", ComplexF32, Float32), ("z", ComplexF64, Float64))
    @eval _reg!($(p * "gebrd_"), () -> @cfunction($(Symbol(p, "gebrd_64_")), Cvoid,
        (_CI, _CI, Ptr{$T}, _CI, Ptr{$R}, Ptr{$R}, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI, _CI)))
end
# bdsqr — bidiagonal SVD (implicit-shift QR). REAL only (s,d); ncc≠0 unsupported (checked at the ABI).
for (p, T) in (("s", Float32), ("d", Float64))
    @eval _reg!($(p * "bdsqr_"), () -> @cfunction($(Symbol(p, "bdsqr_64_")), Cvoid,
        (_CU, _CI, _CI, _CI, _CI, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T},
         _CI, Clong)))
end
# bdsdc — bidiagonal SVD (divide-and-conquer). REAL only (no complex variant in reference LAPACK);
# compq∈{'N','I'} only (checked at the ABI).
for (p, T) in (("s", Float32), ("d", Float64))
    @eval _reg!($(p * "bdsdc_"), () -> @cfunction($(Symbol(p, "bdsdc_64_")), Cvoid,
        (_CU, _CU, _CI, Ptr{$T}, Ptr{$T}, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, Ptr{$T}, _CI, _CI,
         Clong, Clong)))
end
