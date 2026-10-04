# In-process `@cfunction` trampolines for the BLAS-1 SME kernels.
#
# ⚠ THESE LIVE IN THEIR OWN FILE BECAUSE OF INCLUDE ORDER, AND IT IS A CORRECTNESS CONSTRAINT, NOT A
# TIDINESS ONE. `@cfunction` in its CONSTANT form — a bare callee name, which is what the `_SME_STATIC`
# branch below uses — resolves that callee when the enclosing method is DEFINED, not when it runs. So
# every one of these must come after `sme_l1.jl` defines the `_sme_*_cabi` it names. Their Level-2/3
# siblings can sit in `blas3/sme_kernel.jl` next to their own cabi entries; these cannot, because
# `src/PureBLAS.jl` includes `blas3/sme_kernel.jl` one line BEFORE `blas1/sme_l1.jl`, and the dependency
# runs both ways: `sme_l1.jl` needs `_SME_ATTRS`, `_SME_L`, `_sme_p` and `_sme_scal` from the kernel
# file, so the includes cannot simply be swapped.
#
# The failure mode if they move back is silent in every ordinary run: the module loads, the whole test
# suite passes, and only `juliac --trim` fails with `UndefVarError`, because `_SME_STATIC` is true only
# when the CPU target names `+sme` — which only the juliac build passes. `test/cfunction_order_lint.jl`
# is the guard; it is lexical and runs in milliseconds, unlike the `--trim` build it stands in for.
#
# The three branches mirror `blas3/sme_kernel.jl`'s, and the reasons are given there: off SME hardware
# the trampolines must still exist as top-level definitions but must name nothing (inferring an
# `inferencebarrier` + `@cfunction` pair drags the cabi entry and `_gemm_core!` in behind it); the
# constant form is required for the trim/.so build; the `inferencebarrier` form is what stops inference
# folding the callee back to a concrete function in an ordinary session.
@static if !_SME_F64
    _sme_dot_cf() = throw(AssertionError("SME trampoline requested without SME"))
    _sme_asum_cf() = throw(AssertionError("SME trampoline requested without SME"))
    _sme_axpy_cf() = throw(AssertionError("SME trampoline requested without SME"))
    _sme_scal_cf() = throw(AssertionError("SME trampoline requested without SME"))
elseif _SME_STATIC
    _sme_dot_cf() = @cfunction(_sme_dot_cabi, Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Int))
    _sme_asum_cf() = @cfunction(_sme_asum_cabi, Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Int))
    _sme_axpy_cf() = @cfunction(_sme_axpy_cabi, Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Float64, Int))
    _sme_scal_cf() = @cfunction(_sme_scal_cabi, Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Float64, Int))
else
    function _sme_dot_cf()
        d = Base.inferencebarrier(_sme_dot_cabi)
        return @cfunction($d, Cvoid, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Int))
    end
    function _sme_asum_cf()
        e = Base.inferencebarrier(_sme_asum_cabi)
        return @cfunction($e, Cvoid, (Ptr{Float64}, Ptr{Float64}, Int))
    end
    function _sme_axpy_cf()
        p = Base.inferencebarrier(_sme_axpy_cabi)
        return @cfunction($p, Cvoid, (Ptr{Float64}, Ptr{Float64}, Float64, Int))
    end
    function _sme_scal_cf()
        r = Base.inferencebarrier(_sme_scal_cabi)
        return @cfunction($r, Cvoid, (Ptr{Float64}, Ptr{Float64}, Float64, Int))
    end
end
