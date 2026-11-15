# Type-domain StrictMode guarantees: `@test_typestable`, `@test_noboxing`, `@assert_inlined`,
# `@assert_effects`, `@test_trim_compatible`, `@strict_policy`.
#
# Every verdict is taken through `StrictTypesT.@verdict`, which turns the macro's `StrictViolation`
# into a value, so one failing call neither hides the calls after it nor passes silently. A verdict
# can only be trusted if the same check can fail: the "controls" item runs each guarantee on a toy it
# must reject.

@testmodule StrictTypesT begin
    using StrictMode, StrictModeTest, Test

    # Names of `OptimizationFailureReport` frames, keyed by call label. JET reports a frame it could not
    # optimize (a recursive cycle) as information: the types are known, so this is not instability.
    const OPTFAIL = Dict{String, Vector{String}}()

    # Runs `f`, which expands one StrictMode/StrictModeTest macro. A `StrictViolation` becomes
    # `ok = false`; any other exception also becomes `ok = false` but is prefixed so a controls check
    # can tell a rejected call from a broken harness call.
    function run_verdict(f, label)
        t0 = time()
        logs, v = Test.collect_test_logs() do
            try
                f()
                (ok = true, msg = "")
            catch e
                e isa InterruptException && rethrow()
                msg = sprint(showerror, e)
                (ok = false, msg = e isa StrictMode.StrictViolation ? msg : "NON-VIOLATION EXCEPTION: " * msg)
            end
        end
        opt = String[string(l.message) for l in logs if occursin("OptimizationFailureReport", string(l.message))]
        isempty(opt) || (OPTFAIL[label] = opt)
        return (; label, v.ok, v.msg, secs = time() - t0, optfail = opt)
    end

    # `@verdict test_typestable f(x)` expands StrictModeTest's `@test_typestable f(x)`.
    macro verdict(kind, call)
        m = Expr(:macrocall, GlobalRef(StrictModeTest, Symbol("@", kind)), __source__, call)
        return esc(:($(run_verdict)(() -> $m, $(replace(string(call), r"\s+" => " ")))))
    end

    # `@verdictS assert_inlined f(x)` expands StrictMode's `@assert_inlined f(x)`; trailing arguments
    # (the effect list of `@assert_effects`) are forwarded.
    macro verdictS(kind, call, extra...)
        m = Expr(:macrocall, GlobalRef(StrictMode, Symbol("@", kind)), __source__, call, extra...)
        return esc(:($(run_verdict)(() -> $m, $(replace(string(call), r"\s+" => " ")))))
    end

    # `@gate test_typestable f(x)` takes the verdict and records it; `@gate_broken` records a known
    # violation. `@gateS` / `@gateS_broken` are the same for StrictMode's own macros.
    macro gate(kind, call)
        v = Expr(:macrocall, GlobalRef(@__MODULE__, Symbol("@verdict")), __source__, kind, call)
        return esc(:($(chk)($v)))
    end
    macro gate_broken(kind, call)
        v = Expr(:macrocall, GlobalRef(@__MODULE__, Symbol("@verdict")), __source__, kind, call)
        return esc(:($(chk)($v; broken = true)))
    end
    # Sweep forms: a verdict is expected to fail when its label is in `KNOWN_LAP`, or its (label, `T`)
    # pair is in `KNOWN_L123`, both bound in the calling scope.
    macro gate_known(kind, call)
        v = Expr(:macrocall, GlobalRef(@__MODULE__, Symbol("@verdict")), __source__, kind, call)
        lbl = replace(string(call), r"\s+" => " ")
        return esc(:($(chk)($v; broken = $lbl in KNOWN_LAP)))
    end
    macro gate_known_t(kind, call)
        v = Expr(:macrocall, GlobalRef(@__MODULE__, Symbol("@verdict")), __source__, kind, call)
        lbl = replace(string(call), r"\s+" => " ")
        return esc(:($(chk)($v; broken = ($lbl, T) in KNOWN_L123)))
    end
    macro gateS(kind, call, extra...)
        v = Expr(:macrocall, GlobalRef(@__MODULE__, Symbol("@verdictS")), __source__, kind, call, extra...)
        return esc(:($(chk)($v)))
    end
    macro gateS_broken(kind, call, extra...)
        v = Expr(:macrocall, GlobalRef(@__MODULE__, Symbol("@verdictS")), __source__, kind, call, extra...)
        return esc(:($(chk)($v; broken = true)))
    end

    # A trim verdict judged against an exact allowlist of call labels. An allowed call must be rejected,
    # for exactly the interpolation helpers the in-process verifier leaves unresolved on stock Base;
    # every other call must be accepted.
    const INTERP_CALLS = Set(["Base._str_sizehint", "Base.print"])
    function trim_check(r, allow)
        unresolved = Set(String(m.captures[1]) for m in eachmatch(r"unresolved call: :\(([\w.!]+)\(", r.msg))
        if r.label in allow
            @test !r.ok
            @test unresolved == INTERP_CALLS
        else
            r.ok || @error "trim rejection outside the allowlist" call = r.label finding = r.msg
            @test r.ok
        end
        return r
    end

    # Record a verdict. `broken = true` marks a known violation: it reports as `@test_broken`, so the
    # day the code is fixed the item errors with an unexpected pass instead of staying quiet.
    function chk(r; broken::Bool = false)
        if broken
            r.ok && @warn "a call recorded as a known violation now passes" call = r.label
            @test_broken r.ok
        else
            r.ok || @error "StrictMode violation" call = r.label finding = r.msg
            @test r.ok
        end
        return r
    end
end

# ── POSITIVE CONTROLS ────────────────────────────────────────────────────────────────────────────
# Each guarantee is run on a toy that violates it. A verdict from the items below means nothing unless
# these reject. The typestable toys include one whose RETURN type is concrete but whose body dispatches
# at run time: only JET's optimization analysis sees that, and the control proves `@test_typestable`
# does.
@testitem "StrictTypes controls: each guarantee rejects a violating toy" tags = [:checks] setup = [StrictTypesT] begin
    using StrictModeTest, StrictMode
    V = StrictTypesT
    ret_any(v::Vector{Any}) = v[1]
    interior(v::Vector{Any}) = (x = v[1]; sin(x)::Float64)        # concrete return, run-time dispatch inside
    boxer(n::Int) = (c = Ref{Any}(0); for i in 1:n; c[] = c[] + i; end; c[])
    @noinline outline(x) = x * x + 1
    @inline inl(x) = x * x + 1
    divs(x::Int) = div(1, x)                                       # may throw DivideError
    sumv(v) = sum(v)
    # rejected, and rejected as a violation rather than as a broken call
    for r in (
            V.@verdict(test_typestable, ret_any(Any[1.0])),
            V.@verdict(test_typestable, interior(Any[1.0])),
            V.@verdict(test_noboxing, interior(Any[1.0])),
            V.@verdict(test_noboxing, boxer(3)),
            V.@verdictS(assert_inlined, outline(3.0)),
            V.@verdictS(assert_effects, divs(2), (:nothrow,)),
        )
        @test !r.ok
        @test startswith(r.msg, "StrictViolation")
    end
    # accepted: the verdicts are not constant `false`
    for r in (
            V.@verdict(test_typestable, sumv([1.0, 2.0])),
            V.@verdict(test_noboxing, sumv([1.0, 2.0])),
            V.@verdictS(assert_inlined, inl(3.0)),
            V.@verdictS(assert_effects, inl(3.0), (:nothrow, :effect_free, :terminates, :consistent)),
        )
        @test r.ok
    end
end

# ── THE `@assert_typestable` CALLS OF test/strictmode_tests.jl, RE-CHECKED WITH `@test_typestable` ───
# `@assert_typestable` throws only on a non-concrete RETURN type. `@test_typestable` also runs JET's
# optimization analysis, so interior run-time dispatch fails it too. Frames JET cannot optimize are
# reported by StrictModeTest as information (`StrictTypesT.OPTFAIL`), not as failures.
@testitem "StrictTypes: strictmode_tests @assert_typestable calls under @test_typestable" tags = [:checks] setup = [StrictTypesT] begin
    using StrictModeTest, StrictMode, LinearAlgebra
    P = PureBLAS; V = StrictTypesT
    tri(s) = (M = tril(randn(s, s)); for i in 1:s; M[i, i] += s; end; M)
    Atri = tri(32)
    V.@gate test_typestable P._trsm_base_invL!(false, false, false, Atri, randn(32, 256))
    V.@gate test_typestable P._trsm_base_invR!(false, false, false, Atri, randn(256, 32))
    Aup = (M = triu(randn(128, 128)); for i in 1:128; M[i, i] += 128.0; end; M)
    P._trsm_fused_L!(false, Aup, randn(128, 256))
    V.@gate test_typestable P._trsm_fused_L!(false, Aup, randn(128, 256))
    if P._GT_TRANSPOSE
        Auf = (M = triu(randn(512, 512)); for i in 1:512; M[i, i] += 512.0; end; M)
        P._trsm_fused_full_L!(false, Auf, randn(512, 256))
        V.@gate test_typestable P._trsm_fused_full_L!(false, Auf, randn(512, 256))
    end
    As = randn(512, 512); Bs = randn(512, 512)
    V.@gate test_typestable P.trsm!(copy(Bs), tri(512); side = 'L', uplo = 'L', diag = 'N', alpha = 1.0)
    V.@gate test_typestable P.syrk!(zeros(512, 512), As; uplo = 'L', trans = 'N', alpha = 0.8, beta = 0.3)
    V.@gate test_typestable P.syr2k!(zeros(512, 512), As, Bs; uplo = 'L', trans = 'N', alpha = 0.8, beta = 0.3)
    V.@gate test_typestable P.symm!(zeros(512, 512), As, Bs; side = 'L', uplo = 'L', alpha = 0.8, beta = 0.3)
    Awz = randn(ComplexF64, 48, 40); Bwz = randn(ComplexF64, 48, 40)
    V.@gate test_typestable P._ctri_unpacked!(true, true, 1.0, Awz, zeros(ComplexF64, 48, 48), 40)
    V.@gate test_typestable P.herk!(zeros(ComplexF64, 48, 48), Awz; uplo = 'U', trans = 'N', alpha = 1.0, beta = 0.0)
    V.@gate test_typestable P._ctri2_unpacked!(true, true, 1.0, Awz, Bwz, zeros(ComplexF64, 48, 48), 40)
    Atr = randn(ComplexF64, 48, 48) ./ 96
    for d in 1:48
        Atr[d, d] = 1 + abs(Atr[d, d])
    end
    V.@gate test_typestable P._trsm_cmplx_dRC!(true, false, 48, Atr, randn(ComplexF64, 64, 48))
    # GEMM microkernels, full argtypes through the defaulted trailing `Val{B0}`
    W = P._vwidth(Float64); mr = P._MR * W; nr = P._NR; kc = 64
    Ap = randn(mr * kc); Bp = randn(nr * kc); C = zeros(mr, nr)
    GC.@preserve Ap Bp C begin
        ap = pointer(Ap); bp = pointer(Bp); cptr = pointer(C); ldc = mr
        V.@gate test_typestable P._microkernel!(cptr, ldc, ap, bp, kc, Val(P._MR), Val(P._NR))
        V.@gate test_typestable P._microkernel_masked!(cptr, ldc, ap, bp, kc, 11, 5, Val(P._MR), Val(P._NR))
        V.@gate test_typestable P._microkernel_clip!(cptr, ldc, ap, bp, kc, Val(P._MR), Val(1), Val(P._NR))
    end
    kk = 32; Au = randn(mr * kk); Bu = randn(kk * nr); Cu = zeros(mr, nr)
    GC.@preserve Au Bu Cu begin
        aup = pointer(Au); bup = pointer(Bu); cup = pointer(Cu)
        V.@gate test_typestable P._microkernel_unpacked!(cup, mr, aup, mr, 0, bup, kk, 0, kk, 1.0, 0.0, Val(P._MR), Val(P._NR), Val(false), Val(true))
        V.@gate test_typestable P._microkernel_unpacked_mrows!(cup, mr, aup, mr, 0, bup, kk, 0, kk, 1.0, 0.0, 12, Val(P._MR), Val(P._NR), Val(false), Val(true), Val(W))
        V.@gate test_typestable P._mrows_tail!(cup, mr, aup, mr, 0, bup, kk, 0, kk, 1.0, 0.0, 2, Val(P._NR), Val(false), Val(true))
    end
    if P._SPLIT_OK
        smr = P._SMR * W
        As2 = randn(smr * kk); Cs2 = zeros(smr, P._SNR); Bu2 = randn(kk * nr)
        GC.@preserve As2 Bu2 Cs2 begin
            for mask in (false, true)
                mv = mask ? smr - 2 : smr
                V.@gate test_typestable P._microkernel_unpacked_split!(pointer(Cs2), smr, pointer(As2), smr, 0, pointer(Bu2), kk, 0, kk, 1.0, 0.0, mv, Val(P._SMR), Val(P._SNR), Val(P._GEMM_SPLIT_S), Val(false), Val(true), Val(mask))
            end
        end
    end
    Ag = randn(8, 5); Bm = randn(5, 6); Cg = zeros(8, 6)
    V.@gate test_typestable P._gemm_generic!(false, false, false, false, 8, 6, 5, 1.0, Ag, Bm, 0.0, Cg)
    for TC in (ComplexF64, ComplexF32)
        A = randn(TC, 48, 48); A = A * A' + 48I + zeros(TC, 48, 48)
        V.@gate test_typestable P._cpotf2_lower!(copy(A), 48)
        V.@gate test_typestable P.potrf!(copy(A); uplo = 'L')
        A2 = randn(TC, 128, 128); A2 = A2 * A2' + 128I + zeros(TC, 128, 128)
        V.@gate test_typestable P.potrf!(copy(A2); uplo = 'L')
        V.@gate test_typestable P._cpotrf_lower!(copy(A2), 128)
        V.@gate test_typestable P.potrf!(copy(A); uplo = 'U')
        V.@gate test_typestable P.potrf!(copy(A2); uplo = 'U')
        G = randn(TC, 48, 48) + 48I; ip = zeros(Int, 48); pG = pointer(G); ldG = stride(G, 2)
        GC.@preserve G begin
            V.@gate test_typestable P._cgetf2_simd!(pG, ldG, 48, 48, 0, ip, 0)
        end
        Q = randn(TC, 48, 48); tau = similar(Q, 48)
        V.@gate test_typestable P.qr_unblocked!(copy(Q), tau)
    end
    @info "OptimizationFailureReport frames (information, not instability)" calls = length(V.OPTFAIL)
end

# ── ENTRY SWEEP: BLAS-1/2/3 entries over the four element types ────────────────────────────────────
@testitem "StrictTypes: BLAS entry sweep is type stable (JET)" tags = [:checks] setup = [StrictTypesT] begin
    using StrictModeTest, StrictMode, LinearAlgebra
    P = PureBLAS; V = StrictTypesT
    for T in (Float64, Float32, ComplexF64, ComplexF32)
        n = 64
        x = randn(T, n); y = randn(T, n); a = T(1.5)
        V.@gate test_typestable P.axpy!(copy(y), a, x)
        V.@gate test_typestable P.scal!(a, copy(x))
        V.@gate test_typestable P.blascopy!(copy(y), x)
        V.@gate test_typestable P.swap!(copy(x), copy(y))
        V.@gate test_typestable P.dot(x, y)
        V.@gate test_typestable P.dotu(x, y)
        V.@gate test_typestable P.nrm2(x)
        V.@gate test_typestable P.asum(x)
        V.@gate test_typestable P.iamax(x)
        A = randn(T, n, n); S = A + transpose(A); H = A + A'; U = triu(A) + n * I
        V.@gate test_typestable P.gemv!(copy(y), A, x)
        V.@gate test_typestable P.gemv!(copy(y), A, x; trans = 'T', alpha = a, beta = a)
        V.@gate test_typestable P.ger!(a, x, y, copy(A))
        V.@gate test_typestable P.symv!(copy(y), S, x)
        T <: Complex && V.@gate test_typestable P.hemv!(copy(y), H, x)
        V.@gate test_typestable P.trmv!(U, copy(x))
        V.@gate test_typestable P.trsv!(U, copy(x))
        ap = randn(T, n * (n + 1) ÷ 2)
        V.@gate test_typestable P.spmv!(copy(y), ap, x)
        V.@gate test_typestable P.tpmv!(ap, copy(x))
        V.@gate test_typestable P.tpsv!(ap .+ T(n), copy(x))
        V.@gate test_typestable P.spr!(a, x, copy(ap))
        T <: Complex && V.@gate test_typestable P.hpmv!(copy(y), ap, x)
        T <: Complex && V.@gate test_typestable P.hpr!(real(a), x, copy(ap))
        AB = randn(T, 5, n)
        V.@gate test_typestable P.gbmv!(copy(y), AB, x, n, 1, 3)
        V.@gate test_typestable P.sbmv!(copy(y), AB, x)
        V.@gate test_typestable P.tbmv!(AB, copy(x))
        V.@gate test_typestable P.tbsv!(AB .+ T(n), copy(x))
        C = zeros(T, n, n); B = randn(T, n, n)
        V.@gate test_typestable P.gemm!(copy(C), A, B)
        V.@gate test_typestable P.gemm!(copy(C), A, B; alpha = a, beta = a, transA = 'T', transB = 'N')
        V.@gate test_typestable P.trsm!(copy(B), U; side = 'L', uplo = 'U', diag = 'N', alpha = a)
        V.@gate test_typestable P.trsm!(copy(B), U; side = 'R', uplo = 'U', transA = 'N', diag = 'N', alpha = a)
        V.@gate test_typestable P.trmm!(copy(B), U; side = 'L', uplo = 'U', diag = 'N', alpha = a)
        V.@gate test_typestable P.syrk!(copy(C), A; uplo = 'L', trans = 'N', alpha = a, beta = a)
        V.@gate test_typestable P.syr2k!(copy(C), A, B; uplo = 'L', trans = 'N', alpha = a, beta = a)
        V.@gate test_typestable P.symm!(copy(C), S, B; side = 'L', uplo = 'L', alpha = a, beta = a)
        if T <: Complex
            V.@gate test_typestable P.herk!(copy(C), A; uplo = 'L', trans = 'N', alpha = real(a), beta = real(a))
            V.@gate test_typestable P.her2k!(copy(C), A, B; uplo = 'L', trans = 'N', alpha = a, beta = real(a))
            V.@gate test_typestable P.hemm!(copy(C), H, B; side = 'L', uplo = 'L', alpha = a, beta = a)
        end
        V.@gate test_typestable P.potrf!(S * S' + n * I |> Matrix; uplo = 'L')
        V.@gate test_typestable P.getrf!(copy(A))
        V.@gate test_typestable P.geqrf!(copy(A))
    end
    @info "OptimizationFailureReport frames (information, not instability)" calls = length(V.OPTFAIL)
end

@testitem "StrictTypes: LAPACK entry sweep is type stable (JET)" tags = [:checks] setup = [StrictTypesT] begin
    using StrictModeTest, StrictMode, LinearAlgebra
    P = PureBLAS; V = StrictTypesT
    n = 24; k = 12
    Ad = (M = randn(n, n); M + transpose(M)); Az = (M = randn(ComplexF64, n, n); M + transpose(M))
    Ahe = (M = randn(ComplexF64, n, n); M + M')
    Sd = (M = randn(n, n); M * transpose(M) + n * I)            # real SPD
    She = (M = randn(ComplexF64, n, n); M * M' + n * I)         # Hermitian PD

    V.@gate test_typestable P.sysv!('L', copy(Ad), randn(n, 2))
    V.@gate test_typestable P.sysv!('L', copy(Az), randn(ComplexF64, n, 2))
    V.@gate test_typestable P.hesv!('L', copy(Ahe), randn(ComplexF64, n, 2))
    LDd = copy(Ad); ipd = zeros(Int, n); P.sytrf!(LDd, ipd; uplo = 'L')
    LDz = copy(Az); ipz = zeros(Int, n); P.sytrf!(LDz, ipz; uplo = 'L')
    LDh = copy(Ahe); iph = zeros(Int, n); P.hetrf!(LDh, iph; uplo = 'L')
    nbig = 96
    Abd = (M = randn(nbig, nbig); M + transpose(M))
    Abz = (M = randn(ComplexF64, nbig, nbig); M + transpose(M))
    Abh = (M = randn(ComplexF64, nbig, nbig); M + M')
    ipb = zeros(Int, nbig)
    P.sytrf!(copy(Abd), ipb; uplo = 'L')                    # warm the owned W workspace first
    P.hetrf!(copy(Abh), ipb; uplo = 'L')
    for ul in ('L', 'U')
        V.@gate test_typestable P.sytrf!(copy(Abd), zeros(Int, nbig); uplo = ul)
        V.@gate test_typestable P.sytrf!(copy(Abz), zeros(Int, nbig); uplo = ul)
        V.@gate test_typestable P.hetrf!(copy(Abh), zeros(Int, nbig); uplo = ul)
    end
    V.@gate test_typestable P.sytri!(copy(LDd), ipd; uplo = 'L')
    V.@gate test_typestable P.sytri!(copy(LDz), ipz; uplo = 'L')
    V.@gate test_typestable P.hetri!(copy(LDh), iph; uplo = 'L')
    Aqd = randn(n, k); tqd = zeros(Float64, k); Aqz = randn(ComplexF64, n, k); tqz = zeros(ComplexF64, k)
    Fqd = copy(Aqd); P.geqlf!(Fqd, tqd); Fqz = copy(Aqz); P.geqlf!(Fqz, tqz)
    V.@gate test_typestable P.geqlf!(copy(Aqd), zeros(Float64, k))
    V.@gate test_typestable P.orgql!(copy(Fqd), copy(tqd))
    V.@gate test_typestable P.ormql!('L', 'N', copy(Fqd), tqd, randn(n, 3))
    V.@gate test_typestable P.geqlf!(copy(Aqz), zeros(ComplexF64, k))
    V.@gate test_typestable P.orgql!(copy(Fqz), copy(tqz))
    V.@gate test_typestable P.ormql!('L', 'N', copy(Fqz), tqz, randn(ComplexF64, n, 3))
    Ard = randn(k, n); Arz = randn(ComplexF64, k, n)
    Grd = copy(Ard); P.gerqf!(Grd, tqd); Grz = copy(Arz); P.gerqf!(Grz, tqz)
    V.@gate test_typestable P.gerqf!(copy(Ard), zeros(Float64, k))
    V.@gate test_typestable P.orgrq!(copy(Grd), copy(tqd))
    V.@gate test_typestable P.ormrq!('R', 'N', copy(Grd), tqd, randn(3, n))
    V.@gate test_typestable P.gerqf!(copy(Arz), zeros(ComplexF64, k))
    V.@gate test_typestable P.orgrq!(copy(Grz), copy(tqz))
    Tzd = randn(k, n); ttd = zeros(Float64, k); P.tzrzf!(Tzd, ttd)
    V.@gate test_typestable P.tzrzf!(randn(k, n), zeros(Float64, k))
    V.@gate test_typestable P.tzrzf!(randn(ComplexF64, k, n), zeros(ComplexF64, k))
    V.@gate test_typestable P.ormrz!('L', 'N', copy(Tzd), ttd, randn(n, 3))
    V.@gate test_typestable P.pstrf!(copy(Sd), zeros(Int, n), -1.0; uplo = 'L')
    V.@gate test_typestable P.pstrf!(copy(She), zeros(Int, n), -1.0; uplo = 'L')
    V.@gate test_typestable P.gelsd!(randn(n, k), randn(n, 2), -1.0)
    V.@gate test_typestable P.gelsd!(randn(ComplexF64, n, k), randn(ComplexF64, n, 2), -1.0)
    V.@gate test_typestable P.gelsy!(randn(n, k), randn(n, 2), zeros(Int, k), -1.0)
    V.@gate test_typestable P.gelsy!(randn(ComplexF64, n, k), randn(ComplexF64, n, 2), zeros(Int, k), -1.0)
    V.@gate test_typestable P.gglse!(randn(8, 6), randn(8), randn(4, 6), randn(4))
    V.@gate test_typestable P.gglse!(randn(ComplexF64, 8, 6), randn(ComplexF64, 8), randn(ComplexF64, 4, 6), randn(ComplexF64, 4))
    V.@gate test_typestable P.trsyl!('N', 'N', 1, triu(randn(n, n)) + n * I, triu(randn(n, n)) + n * I, randn(n, n))
    V.@gate test_typestable P.trsyl!('N', 'N', 1, triu(randn(ComplexF64, n, n)) + n * I, triu(randn(ComplexF64, n, n)) + n * I, randn(ComplexF64, n, n))
    V.@gate test_typestable P.trexc!('V', triu(randn(n, n)) + n * I, Matrix(1.0I, n, n), 2, n - 1)
    V.@gate test_typestable P.trexc!('V', triu(randn(ComplexF64, n, n)) + n * I, Matrix(ComplexF64(1)I, n, n), 2, n - 1)
    V.@gate test_typestable P.trsen!('N', 'V', rand(Bool, n), triu(randn(n, n)) + n * I, Matrix(1.0I, n, n))
    V.@gate test_typestable P.trsen!('N', 'V', rand(Bool, n), triu(randn(ComplexF64, n, n)) + n * I, Matrix(ComplexF64(1)I, n, n))
    V.@gate test_typestable P.ggev!('N', 'V', randn(n, n), randn(n, n))
    V.@gate test_typestable P.ggev!('N', 'V', randn(ComplexF64, n, n), randn(ComplexF64, n, n))
    V.@gate test_typestable P.gges!('V', 'V', randn(n, n), randn(n, n))
    V.@gate test_typestable P.gges!('V', 'V', randn(ComplexF64, n, n), randn(ComplexF64, n, n))
    V.@gate test_typestable P.ggsvd!('U', 'V', 'Q', randn(8, 6), randn(6, 6))
    V.@gate test_typestable P.ggsvd!('U', 'V', 'Q', randn(ComplexF64, 8, 6), randn(ComplexF64, 6, 6))
    V.@gate_broken test_typestable P.sygvd!(1, 'V', 'U', copy(Ad), copy(Sd))
    V.@gate_broken test_typestable P.hegvd!(1, 'V', 'U', copy(Ahe), copy(She))
    dd = randn(n); ee = randn(n - 1)
    V.@gate test_typestable P.stebz!('A', 'B', 0.0, 0.0, 1, n, 0.0, copy(dd), copy(ee))
    w, ib, isp, _ = P.stebz!('A', 'B', 0.0, 0.0, 1, n, 0.0, copy(dd), copy(ee))
    V.@gate test_typestable P.stein!(copy(dd), copy(ee), w, ib, isp)
    ABd2 = randn(6, n); _, gip, _ = P.gbtrf!(2, 1, n, ABd2)
    V.@gate test_typestable P.gbtrf!(2, 1, n, randn(6, n))
    V.@gate test_typestable P.gbtrs!('N', 2, 1, n, copy(ABd2), gip, randn(n, 2))
    V.@gate test_typestable P.gbtrf!(2, 1, n, randn(ComplexF64, 6, n))
    ldgb = 2 * 32 + 8 + 1
    P.gbtrf!(32, 8, 64, randn(ldgb, 64))                 # warm the owned workspace first
    P.gbtrf!(32, 8, 64, randn(ComplexF64, ldgb, 64))
    V.@gate test_typestable P.gbtrf!(32, 8, 64, randn(ldgb, 64))
    V.@gate test_typestable P.gbtrf!(32, 8, 64, randn(ComplexF64, ldgb, 64))
    Dp = fill(4.0, n); Ep = fill(1.0, n - 1); Df = copy(Dp); Ef = copy(Ep); P.pttrf!(Df, Ef)
    V.@gate test_typestable P.pttrf!(fill(4.0, n), fill(1.0, n - 1))
    V.@gate test_typestable P.pttrs!(copy(Df), copy(Ef), randn(n, 2))
    V.@gate test_typestable P.ptsv!(fill(4.0, n), fill(1.0, n - 1), randn(n, 2))
    Dpz = fill(4.0, n); Epz = fill(1.0 + 0im, n - 1)
    V.@gate test_typestable P.ptsv!(copy(Dpz), copy(Epz), randn(ComplexF64, n, 2))
    dl = fill(1.0, n - 1); dm = fill(4.0, n); du = fill(1.0, n - 1)
    du2 = zeros(Float64, n - 2); gtip = zeros(Int, n); P.gttrf!(copy(dl), copy(dm), copy(du), du2, gtip)
    V.@gate test_typestable P.gtsv!(copy(dl), copy(dm), copy(du), randn(n, 2))
    V.@gate test_typestable P.gttrf!(copy(dl), copy(dm), copy(du), zeros(Float64, n - 2), zeros(Int, n))
    V.@gate test_typestable P.gtsv!(fill(1.0 + 0im, n - 1), fill(4.0 + 0im, n), fill(1.0 + 0im, n - 1), randn(ComplexF64, n, 2))
    mkband(T) = (AB = zeros(T, 3, n); AB[1, :] .= T(10); AB[2, 1:(n - 1)] .= T(1); AB[3, 1:(n - 2)] .= T(0.5); AB)
    ABsd = mkband(Float64); P.pbtrf!(ABsd; uplo = 'L', kd = 2)
    V.@gate test_typestable P.pbtrf!(mkband(Float64); uplo = 'L', kd = 2)
    V.@gate test_typestable P.pbtrs!(copy(ABsd), randn(n, 2); uplo = 'L', kd = 2)
    V.@gate test_typestable P.pbtrf!(mkband(ComplexF64); uplo = 'L', kd = 2)
    wideband(T, kd) = (
        AB = zeros(T, kd + 1, n); AB[1, :] .= T(4 * kd);
        for d in 1:kd
            AB[1 + d, 1:(n - d)] .= T(0.5)
        end; AB
    )
    wideU(T, kd) = (
        AB = zeros(T, kd + 1, n); AB[kd + 1, :] .= T(4 * kd);
        for d in 1:kd
            AB[kd + 1 - d, (1 + d):n] .= T(0.5)
        end; AB
    )
    V.@gate test_typestable P.pbtrf!(wideband(Float64, 64); uplo = 'L', kd = 64)
    V.@gate test_typestable P.pbtrf!(wideU(Float64, 64); uplo = 'U', kd = 64)
    V.@gate test_typestable P._pbtrf_blocked!(wideband(Float64, 64), n, 64)
    V.@gate test_typestable P._pbtrf_repack_U!(wideU(Float64, 64), n, 64)
    V.@gate test_typestable P._pbtrf_blocked_U!(wideU(Float64, 64), n, 64)
    V.@gate test_typestable P._pbtrf_blocked!(wideband(ComplexF64, 32), n, 32)
    V.@gate test_typestable P._pbtrf_repack_U!(wideU(ComplexF64, 32), n, 32)
    V.@gate test_typestable P._pbtrf_blocked_U!(wideU(ComplexF64, 32), n, 32)
    pack(M) = [M[i, j] for j in 1:n for i in j:n]              # lower column-packed
    APsd = pack(Sd); P.pptrf!(APsd; uplo = 'L')
    V.@gate test_typestable P.pptrf!(pack(Sd); uplo = 'L')
    V.@gate test_typestable P.pptrs!(copy(APsd), randn(n, 2); uplo = 'L')
    V.@gate test_typestable P.pptrf!(pack(She); uplo = 'L')
    @test true
    # The 5-argument forms return `(w,)` or `(w, A)` by the runtime `jobz`; the 6-argument in-place forms
    # always return `A`.
    V.@gate test_typestable P.sygvd!(1, 'V', 'U', copy(Ad), copy(Sd), zeros(n))
    V.@gate test_typestable P.hegvd!(1, 'V', 'U', copy(Ahe), copy(She), zeros(n))
    @info "OptimizationFailureReport frames (information, not instability)" calls = length(V.OPTFAIL)
end

# ── NOBOXING ON THE BOXING-PRONE PATHS ───────────────────────────────────────────────────────────────
# `@test_noboxing` is AllocCheck's proof that nothing boxes or dispatches dynamically; typed heap
# allocations are allowed. The targets are the kernels where a closure or a tuple has boxed before
# (`_trsm_base_inv*!`, the gemv-T block callers, `_uker_cmplx!`, the microkernels), each with its
# FULL argument list: the defaulted trailing `::Val{B0}` is passed explicitly so the generated body
# is the one inspected, not the forwarder.
#
# The pool-growth functions are registered as allocation barriers, as in strictmode_tests.jl, for the
# trsm bases: they reach the threaded gemm pool, whose `yield()` statically reaches the scheduler
# (`trypoptask`, dynamic dispatch at task.jl). The first check below shows the verdict WITHOUT the
# registration, so the registration is known to be what clears it.
@testitem "StrictTypes: noboxing on boxing-prone kernels (full argtypes)" tags = [:checks] setup = [StrictTypesT] begin
    using StrictModeTest, StrictMode, InteractiveUtils, LinearAlgebra
    P = PureBLAS; V = StrictTypesT
    tri(s) = (M = tril(randn(s, s)); for i in 1:s; M[i, i] += s; end; M)
    Atri = tri(32); BL = randn(32, 256); BR = randn(256, 32)

    # Non-vacuity of the kernel checks: the inspected code is the generated body, not a forwarder.
    # The 8-argument `_microkernel!` has a vector-FMA loop; a forwarder would have neither.
    W0 = P._vwidth(Float64)
    mk_ty = Tuple{Ptr{Float64}, Int, Ptr{Float64}, Ptr{Float64}, Int, Val{P._MR}, Val{P._NR}, Val{false}}
    ci = only(code_typed(P._microkernel!, mk_ty))[1]
    @test length(ci.code) > 100
    ll = sprint(io -> code_llvm(io, P._microkernel!, mk_ty; optimize = true, debuginfo = :none))
    @test count(occursin("<$W0 x double>", l) && occursin("fmuladd", l) for l in split(ll, '\n')) >= P._MR * P._NR

    # Without the registration the scheduler reach is reported (the check can fail on these calls).
    V.@gate_broken test_noboxing P._trsm_base_invL!(false, false, false, Atri, BL)
    V.@gate_broken test_noboxing P._trsm_base_invR!(false, false, false, Atri, BR)

    pools = (P._ws_grow!, P._ws_slot!, P._arena_grow!, P._arena_coalesce!, P._m3ws, P._gpackws, P._strws, P._symm_scr_mt)
    foreach(StrictMode.register_alloc_barrier!, pools)
    StrictMode.clear_cache!()
    try
        V.@gate test_noboxing P._trsm_base_invL!(false, false, false, Atri, BL)
        V.@gate test_noboxing P._trsm_base_invR!(false, false, false, Atri, BR)
        for tr in (true,), up in (false, true)
            V.@gate test_noboxing P._trsm_base_invL!(up, tr, false, Atri, BL)
        end
        W = P._vwidth(Float64); mr = P._MR * W; nr = P._NR; kc = 64
        Ap = randn(mr * kc); Bp = randn(nr * kc); C = zeros(mr, nr)
        GC.@preserve Ap Bp C begin
            ap = pointer(Ap); bp = pointer(Bp); cptr = pointer(C); ldc = mr
            for b0 in (false, true)
                V.@gate test_noboxing P._microkernel!(cptr, ldc, ap, bp, kc, Val(P._MR), Val(P._NR), Val(b0))
                V.@gate test_noboxing P._microkernel!(cptr, ldc, ap, bp, kc, 1.5, Val(P._MR), Val(P._NR), Val(b0))
                V.@gate test_noboxing P._microkernel_masked!(cptr, ldc, ap, bp, kc, 11, 5, Val(P._MR), Val(P._NR), Val(b0))
                V.@gate test_noboxing P._microkernel_clip!(cptr, ldc, ap, bp, kc, Val(P._MR), Val(1), Val(P._NR), Val(b0))
            end
            V.@gate test_noboxing P._microkernel!(cptr, ldc, ap, bp, kc, Val(P._MR), Val(P._NR))
        end
        kk = 32; Au = randn(mr * kk); Bu = randn(kk * nr); Cu = zeros(mr, nr)
        GC.@preserve Au Bu Cu begin
            aup = pointer(Au); bup = pointer(Bu); cup = pointer(Cu)
            for tb in (false, true), b0 in (false, true)
                V.@gate test_noboxing P._microkernel_unpacked!(cup, mr, aup, mr, 0, bup, kk, 0, kk, 1.0, b0 ? 0.0 : 2.0, Val(P._MR), Val(P._NR), Val(tb), Val(b0))
            end
            V.@gate test_noboxing P._microkernel_unpacked_mrows!(cup, mr, aup, mr, 0, bup, kk, 0, kk, 1.0, 0.0, 12, Val(P._MR), Val(P._NR), Val(false), Val(true), Val(W))
            V.@gate test_noboxing P._mrows_tail!(cup, mr, aup, mr, 0, bup, kk, 0, kk, 1.0, 0.0, 2, Val(P._NR), Val(false), Val(true))
        end
        let m = P._CMR * W, n = 4, k = 16
            A = randn(8 * m * k); B = randn(8 * k * n); Cc = zeros(8 * m * n)
            GC.@preserve A B Cc for (tb, sa, sb, b0, a1, ar, full, tri) in (
                    (false, 1, 1, true, false, false, true, false), (false, -1, 1, false, true, false, true, false),
                    (true, 1, -1, true, false, true, false, false), (true, 1, 1, false, false, false, false, true),
                )
                F = (P._CMR, n, tb, sa, sb, b0, a1, ar, full, tri)
                V.@gate test_noboxing P._uker_cmplx!(pointer(Cc), m, pointer(A), m, 0, pointer(B), tb ? n : k, 0, k, 1.0, 0.5, m, n, Val(F), 0, true)
            end
        end
        let T = Float64, m = 256, nc = 4
            A = randn(m, 8); x = randn(m); y = zeros(8)
            GC.@preserve A x y begin
                ps = ntuple(c -> pointer(A) + (c - 1) * m * 8, nc)
                for b0 in (false, true), (u, pf) in ((1, 0), (4, 0), (1, 8 * P._CACHELINE))
                    V.@gate test_noboxing P._gemv_t_block!(pointer(y), ps, pointer(x), m, 1.0, 0.0, Val(nc), Val(b0), Val(u), Val(pf))
                end
                V.@gate test_noboxing P._gemvt_cols!(Val(nc), Val(false), 1, 0, pointer(y), pointer(A), pointer(x), m, m, 8, 1.0, 0.0, 8)
            end
            V.@gate test_noboxing P._gemv_t_sweep_nc!(m, 8, 1.0, A, x, 0.0, y, 4)
        end
        let m = 128, n = 8, T = ComplexF64
            A = randn(T, m, n); x = randn(T, m); y = zeros(T, n)
            for cj in (false, true), half in (false, true)
                V.@gate test_noboxing P._gemv_tc_run!(m, n, T(1.5), A, x, T(0.0), y, Val(cj), Val(4), Val(half))
            end
        end
    finally
        for f in pools
            delete!(StrictMode._ALLOC_BARRIERS, f)     # no public unregister exists
        end
        StrictMode.clear_cache!()
    end
end

# Entry sweep: no BLAS or LAPACK entry boxes or dispatches dynamically.
#
# Pool-growth functions and every `*_threaded!` driver are registered as barriers: the threaded drivers
# synchronize through `wait()`/`yield()`, and the static proof otherwise reports the scheduler's own
# dynamic dispatch (`trypoptask`, task.jl) for the serial entry too. What is left after the registration
# is listed in `KNOWN_*`: each is the same scheduler reach (no PureBLAS frame, no boxing in PureBLAS
# code), through a threaded driver that has no `_threaded!` name. They report as broken, so removing
# one of those reaches turns the entry into an unexpected pass.
@testitem "StrictTypes: BLAS and LAPACK entries do not box (noboxing sweep)" tags = [:checks] setup = [StrictTypesT] begin
    using StrictModeTest, StrictMode, LinearAlgebra
    P = PureBLAS; V = StrictTypesT
    pools = (P._ws_grow!, P._ws_slot!, P._arena_grow!, P._arena_coalesce!, P._m3ws, P._gpackws, P._strws, P._symm_scr_mt)
    threaded = Function[getfield(P, n) for n in names(P; all = true) if endswith(string(n), "_threaded!") && isdefined(P, n) && getfield(P, n) isa Function]
    @test length(threaded) >= 12
    barriers = (pools..., threaded...)
    foreach(StrictMode.register_alloc_barrier!, barriers)
    StrictMode.clear_cache!()
    KNOWN_L123 = Set(
        [
            ("P.trsm!(copy(B), U; side = 'L', uplo = 'U', diag = 'N', alpha = a)", Float64),
            ("P.trsm!(copy(B), U; side = 'R', uplo = 'U', transA = 'N', diag = 'N', alpha = a)", Float64),
            ("P.getrf!(copy(A))", Float64),
            ("P.geqrf!(copy(A))", Float64),
            ("P.trsm!(copy(B), U; side = 'L', uplo = 'U', diag = 'N', alpha = a)", Float32),
            ("P.trsm!(copy(B), U; side = 'R', uplo = 'U', transA = 'N', diag = 'N', alpha = a)", Float32),
            ("P.getrf!(copy(A))", Float32),
            ("P.geqrf!(copy(A))", Float32),
            ("P.trsm!(copy(B), U; side = 'L', uplo = 'U', diag = 'N', alpha = a)", ComplexF64),
            ("P.trsm!(copy(B), U; side = 'R', uplo = 'U', transA = 'N', diag = 'N', alpha = a)", ComplexF64),
            ("P.getrf!(copy(A))", ComplexF64),
            ("P.geqrf!(copy(A))", ComplexF64),
            ("P.trsm!(copy(B), U; side = 'L', uplo = 'U', diag = 'N', alpha = a)", ComplexF32),
            ("P.trsm!(copy(B), U; side = 'R', uplo = 'U', transA = 'N', diag = 'N', alpha = a)", ComplexF32),
            ("P.getrf!(copy(A))", ComplexF32),
            ("P.geqrf!(copy(A))", ComplexF32),
        ]
    )
    KNOWN_LAP = Set(
        [
            "P.sysv!('L', copy(Ad), randn(n, 2))",
            "P.sysv!('L', copy(Az), randn(ComplexF64, n, 2))",
            "P.hesv!('L', copy(Ahe), randn(ComplexF64, n, 2))",
            "P.gelsd!(randn(n, k), randn(n, 2), -1.0)",
            "P.gelsd!(randn(ComplexF64, n, k), randn(ComplexF64, n, 2), -1.0)",
            "P.gelsy!(randn(n, k), randn(n, 2), zeros(Int, k), -1.0)",
            "P.gelsy!(randn(ComplexF64, n, k), randn(ComplexF64, n, 2), zeros(Int, k), -1.0)",
            "P.gglse!(randn(8, 6), randn(8), randn(4, 6), randn(4))",
            "P.gglse!(randn(ComplexF64, 8, 6), randn(ComplexF64, 8), randn(ComplexF64, 4, 6), randn(ComplexF64, 4))",
            "P.ggev!('N', 'V', randn(n, n), randn(n, n))",
            "P.ggev!('N', 'V', randn(ComplexF64, n, n), randn(ComplexF64, n, n))",
            "P.gges!('V', 'V', randn(n, n), randn(n, n))",
            "P.gges!('V', 'V', randn(ComplexF64, n, n), randn(ComplexF64, n, n))",
            "P.sygvd!(1, 'V', 'U', copy(Ad), copy(Sd))",
            "P.hegvd!(1, 'V', 'U', copy(Ahe), copy(She))",
            "P.gbtrf!(2, 1, n, randn(6, n))",
            "P.gbtrf!(2, 1, n, randn(ComplexF64, 6, n))",
            "P.gbtrf!(32, 8, 64, randn(ldgb, 64))",
            "P.gbtrf!(32, 8, 64, randn(ComplexF64, ldgb, 64))",
        ]
    )
    try
        for T in (Float64, Float32, ComplexF64, ComplexF32)
            n = 64
            x = randn(T, n); y = randn(T, n); a = T(1.5)
            V.@gate_known_t test_noboxing P.axpy!(copy(y), a, x)
            V.@gate_known_t test_noboxing P.scal!(a, copy(x))
            V.@gate_known_t test_noboxing P.blascopy!(copy(y), x)
            V.@gate_known_t test_noboxing P.swap!(copy(x), copy(y))
            V.@gate_known_t test_noboxing P.dot(x, y)
            V.@gate_known_t test_noboxing P.dotu(x, y)
            V.@gate_known_t test_noboxing P.nrm2(x)
            V.@gate_known_t test_noboxing P.asum(x)
            V.@gate_known_t test_noboxing P.iamax(x)
            A = randn(T, n, n); S = A + transpose(A); H = A + A'; U = triu(A) + n * I
            V.@gate_known_t test_noboxing P.gemv!(copy(y), A, x)
            V.@gate_known_t test_noboxing P.gemv!(copy(y), A, x; trans = 'T', alpha = a, beta = a)
            V.@gate_known_t test_noboxing P.ger!(a, x, y, copy(A))
            V.@gate_known_t test_noboxing P.symv!(copy(y), S, x)
            T <: Complex && V.@gate_known_t test_noboxing P.hemv!(copy(y), H, x)
            V.@gate_known_t test_noboxing P.trmv!(U, copy(x))
            V.@gate_known_t test_noboxing P.trsv!(U, copy(x))
            ap = randn(T, n * (n + 1) ÷ 2)
            V.@gate_known_t test_noboxing P.spmv!(copy(y), ap, x)
            V.@gate_known_t test_noboxing P.tpmv!(ap, copy(x))
            V.@gate_known_t test_noboxing P.tpsv!(ap .+ T(n), copy(x))
            V.@gate_known_t test_noboxing P.spr!(a, x, copy(ap))
            T <: Complex && V.@gate_known_t test_noboxing P.hpmv!(copy(y), ap, x)
            T <: Complex && V.@gate_known_t test_noboxing P.hpr!(real(a), x, copy(ap))
            AB = randn(T, 5, n)
            V.@gate_known_t test_noboxing P.gbmv!(copy(y), AB, x, n, 1, 3)
            V.@gate_known_t test_noboxing P.sbmv!(copy(y), AB, x)
            V.@gate_known_t test_noboxing P.tbmv!(AB, copy(x))
            V.@gate_known_t test_noboxing P.tbsv!(AB .+ T(n), copy(x))
            C = zeros(T, n, n); B = randn(T, n, n)
            V.@gate_known_t test_noboxing P.gemm!(copy(C), A, B)
            V.@gate_known_t test_noboxing P.gemm!(copy(C), A, B; alpha = a, beta = a, transA = 'T', transB = 'N')
            V.@gate_known_t test_noboxing P.trsm!(copy(B), U; side = 'L', uplo = 'U', diag = 'N', alpha = a)
            V.@gate_known_t test_noboxing P.trsm!(copy(B), U; side = 'R', uplo = 'U', transA = 'N', diag = 'N', alpha = a)
            V.@gate_known_t test_noboxing P.trmm!(copy(B), U; side = 'L', uplo = 'U', diag = 'N', alpha = a)
            V.@gate_known_t test_noboxing P.syrk!(copy(C), A; uplo = 'L', trans = 'N', alpha = a, beta = a)
            V.@gate_known_t test_noboxing P.syr2k!(copy(C), A, B; uplo = 'L', trans = 'N', alpha = a, beta = a)
            V.@gate_known_t test_noboxing P.symm!(copy(C), S, B; side = 'L', uplo = 'L', alpha = a, beta = a)
            if T <: Complex
                V.@gate_known_t test_noboxing P.herk!(copy(C), A; uplo = 'L', trans = 'N', alpha = real(a), beta = real(a))
                V.@gate_known_t test_noboxing P.her2k!(copy(C), A, B; uplo = 'L', trans = 'N', alpha = a, beta = real(a))
                V.@gate_known_t test_noboxing P.hemm!(copy(C), H, B; side = 'L', uplo = 'L', alpha = a, beta = a)
            end
            V.@gate_known_t test_noboxing P.potrf!(S * S' + n * I |> Matrix; uplo = 'L')
            V.@gate_known_t test_noboxing P.getrf!(copy(A))
            V.@gate_known_t test_noboxing P.geqrf!(copy(A))
        end
        n = 24; k = 12
        Ad = (M = randn(n, n); M + transpose(M)); Az = (M = randn(ComplexF64, n, n); M + transpose(M))
        Ahe = (M = randn(ComplexF64, n, n); M + M')
        Sd = (M = randn(n, n); M * transpose(M) + n * I)            # real SPD
        She = (M = randn(ComplexF64, n, n); M * M' + n * I)         # Hermitian PD

        V.@gate_known test_noboxing P.sysv!('L', copy(Ad), randn(n, 2))
        V.@gate_known test_noboxing P.sysv!('L', copy(Az), randn(ComplexF64, n, 2))
        V.@gate_known test_noboxing P.hesv!('L', copy(Ahe), randn(ComplexF64, n, 2))
        LDd = copy(Ad); ipd = zeros(Int, n); P.sytrf!(LDd, ipd; uplo = 'L')
        LDz = copy(Az); ipz = zeros(Int, n); P.sytrf!(LDz, ipz; uplo = 'L')
        LDh = copy(Ahe); iph = zeros(Int, n); P.hetrf!(LDh, iph; uplo = 'L')
        nbig = 96
        Abd = (M = randn(nbig, nbig); M + transpose(M))
        Abz = (M = randn(ComplexF64, nbig, nbig); M + transpose(M))
        Abh = (M = randn(ComplexF64, nbig, nbig); M + M')
        ipb = zeros(Int, nbig)
        P.sytrf!(copy(Abd), ipb; uplo = 'L')                    # warm the owned W workspace first
        P.hetrf!(copy(Abh), ipb; uplo = 'L')
        for ul in ('L', 'U')
            V.@gate_known test_noboxing P.sytrf!(copy(Abd), zeros(Int, nbig); uplo = ul)
            V.@gate_known test_noboxing P.sytrf!(copy(Abz), zeros(Int, nbig); uplo = ul)
            V.@gate_known test_noboxing P.hetrf!(copy(Abh), zeros(Int, nbig); uplo = ul)
        end
        V.@gate_known test_noboxing P.sytri!(copy(LDd), ipd; uplo = 'L')
        V.@gate_known test_noboxing P.sytri!(copy(LDz), ipz; uplo = 'L')
        V.@gate_known test_noboxing P.hetri!(copy(LDh), iph; uplo = 'L')
        Aqd = randn(n, k); tqd = zeros(Float64, k); Aqz = randn(ComplexF64, n, k); tqz = zeros(ComplexF64, k)
        Fqd = copy(Aqd); P.geqlf!(Fqd, tqd); Fqz = copy(Aqz); P.geqlf!(Fqz, tqz)
        V.@gate_known test_noboxing P.geqlf!(copy(Aqd), zeros(Float64, k))
        V.@gate_known test_noboxing P.orgql!(copy(Fqd), copy(tqd))
        V.@gate_known test_noboxing P.ormql!('L', 'N', copy(Fqd), tqd, randn(n, 3))
        V.@gate_known test_noboxing P.geqlf!(copy(Aqz), zeros(ComplexF64, k))
        V.@gate_known test_noboxing P.orgql!(copy(Fqz), copy(tqz))
        V.@gate_known test_noboxing P.ormql!('L', 'N', copy(Fqz), tqz, randn(ComplexF64, n, 3))
        Ard = randn(k, n); Arz = randn(ComplexF64, k, n)
        Grd = copy(Ard); P.gerqf!(Grd, tqd); Grz = copy(Arz); P.gerqf!(Grz, tqz)
        V.@gate_known test_noboxing P.gerqf!(copy(Ard), zeros(Float64, k))
        V.@gate_known test_noboxing P.orgrq!(copy(Grd), copy(tqd))
        V.@gate_known test_noboxing P.ormrq!('R', 'N', copy(Grd), tqd, randn(3, n))
        V.@gate_known test_noboxing P.gerqf!(copy(Arz), zeros(ComplexF64, k))
        V.@gate_known test_noboxing P.orgrq!(copy(Grz), copy(tqz))
        Tzd = randn(k, n); ttd = zeros(Float64, k); P.tzrzf!(Tzd, ttd)
        V.@gate_known test_noboxing P.tzrzf!(randn(k, n), zeros(Float64, k))
        V.@gate_known test_noboxing P.tzrzf!(randn(ComplexF64, k, n), zeros(ComplexF64, k))
        V.@gate_known test_noboxing P.ormrz!('L', 'N', copy(Tzd), ttd, randn(n, 3))
        V.@gate_known test_noboxing P.pstrf!(copy(Sd), zeros(Int, n), -1.0; uplo = 'L')
        V.@gate_known test_noboxing P.pstrf!(copy(She), zeros(Int, n), -1.0; uplo = 'L')
        V.@gate_known test_noboxing P.gelsd!(randn(n, k), randn(n, 2), -1.0)
        V.@gate_known test_noboxing P.gelsd!(randn(ComplexF64, n, k), randn(ComplexF64, n, 2), -1.0)
        V.@gate_known test_noboxing P.gelsy!(randn(n, k), randn(n, 2), zeros(Int, k), -1.0)
        V.@gate_known test_noboxing P.gelsy!(randn(ComplexF64, n, k), randn(ComplexF64, n, 2), zeros(Int, k), -1.0)
        V.@gate_known test_noboxing P.gglse!(randn(8, 6), randn(8), randn(4, 6), randn(4))
        V.@gate_known test_noboxing P.gglse!(randn(ComplexF64, 8, 6), randn(ComplexF64, 8), randn(ComplexF64, 4, 6), randn(ComplexF64, 4))
        V.@gate_known test_noboxing P.trsyl!('N', 'N', 1, triu(randn(n, n)) + n * I, triu(randn(n, n)) + n * I, randn(n, n))
        V.@gate_known test_noboxing P.trsyl!('N', 'N', 1, triu(randn(ComplexF64, n, n)) + n * I, triu(randn(ComplexF64, n, n)) + n * I, randn(ComplexF64, n, n))
        V.@gate_known test_noboxing P.trexc!('V', triu(randn(n, n)) + n * I, Matrix(1.0I, n, n), 2, n - 1)
        V.@gate_known test_noboxing P.trexc!('V', triu(randn(ComplexF64, n, n)) + n * I, Matrix(ComplexF64(1)I, n, n), 2, n - 1)
        V.@gate_known test_noboxing P.trsen!('N', 'V', rand(Bool, n), triu(randn(n, n)) + n * I, Matrix(1.0I, n, n))
        V.@gate_known test_noboxing P.trsen!('N', 'V', rand(Bool, n), triu(randn(ComplexF64, n, n)) + n * I, Matrix(ComplexF64(1)I, n, n))
        V.@gate_known test_noboxing P.ggev!('N', 'V', randn(n, n), randn(n, n))
        V.@gate_known test_noboxing P.ggev!('N', 'V', randn(ComplexF64, n, n), randn(ComplexF64, n, n))
        V.@gate_known test_noboxing P.gges!('V', 'V', randn(n, n), randn(n, n))
        V.@gate_known test_noboxing P.gges!('V', 'V', randn(ComplexF64, n, n), randn(ComplexF64, n, n))
        V.@gate_known test_noboxing P.ggsvd!('U', 'V', 'Q', randn(8, 6), randn(6, 6))
        V.@gate_known test_noboxing P.ggsvd!('U', 'V', 'Q', randn(ComplexF64, 8, 6), randn(ComplexF64, 6, 6))
        V.@gate_known test_noboxing P.sygvd!(1, 'V', 'U', copy(Ad), copy(Sd))
        V.@gate_known test_noboxing P.hegvd!(1, 'V', 'U', copy(Ahe), copy(She))
        dd = randn(n); ee = randn(n - 1)
        V.@gate_known test_noboxing P.stebz!('A', 'B', 0.0, 0.0, 1, n, 0.0, copy(dd), copy(ee))
        w, ib, isp, _ = P.stebz!('A', 'B', 0.0, 0.0, 1, n, 0.0, copy(dd), copy(ee))
        V.@gate_known test_noboxing P.stein!(copy(dd), copy(ee), w, ib, isp)
        ABd2 = randn(6, n); _, gip, _ = P.gbtrf!(2, 1, n, ABd2)
        V.@gate_known test_noboxing P.gbtrf!(2, 1, n, randn(6, n))
        V.@gate_known test_noboxing P.gbtrs!('N', 2, 1, n, copy(ABd2), gip, randn(n, 2))
        V.@gate_known test_noboxing P.gbtrf!(2, 1, n, randn(ComplexF64, 6, n))
        ldgb = 2 * 32 + 8 + 1
        P.gbtrf!(32, 8, 64, randn(ldgb, 64))                 # warm the owned workspace first
        P.gbtrf!(32, 8, 64, randn(ComplexF64, ldgb, 64))
        V.@gate_known test_noboxing P.gbtrf!(32, 8, 64, randn(ldgb, 64))
        V.@gate_known test_noboxing P.gbtrf!(32, 8, 64, randn(ComplexF64, ldgb, 64))
        Dp = fill(4.0, n); Ep = fill(1.0, n - 1); Df = copy(Dp); Ef = copy(Ep); P.pttrf!(Df, Ef)
        V.@gate_known test_noboxing P.pttrf!(fill(4.0, n), fill(1.0, n - 1))
        V.@gate_known test_noboxing P.pttrs!(copy(Df), copy(Ef), randn(n, 2))
        V.@gate_known test_noboxing P.ptsv!(fill(4.0, n), fill(1.0, n - 1), randn(n, 2))
        Dpz = fill(4.0, n); Epz = fill(1.0 + 0im, n - 1)
        V.@gate_known test_noboxing P.ptsv!(copy(Dpz), copy(Epz), randn(ComplexF64, n, 2))
        dl = fill(1.0, n - 1); dm = fill(4.0, n); du = fill(1.0, n - 1)
        du2 = zeros(Float64, n - 2); gtip = zeros(Int, n); P.gttrf!(copy(dl), copy(dm), copy(du), du2, gtip)
        V.@gate_known test_noboxing P.gtsv!(copy(dl), copy(dm), copy(du), randn(n, 2))
        V.@gate_known test_noboxing P.gttrf!(copy(dl), copy(dm), copy(du), zeros(Float64, n - 2), zeros(Int, n))
        V.@gate_known test_noboxing P.gtsv!(fill(1.0 + 0im, n - 1), fill(4.0 + 0im, n), fill(1.0 + 0im, n - 1), randn(ComplexF64, n, 2))
        mkband(T) = (AB = zeros(T, 3, n); AB[1, :] .= T(10); AB[2, 1:(n - 1)] .= T(1); AB[3, 1:(n - 2)] .= T(0.5); AB)
        ABsd = mkband(Float64); P.pbtrf!(ABsd; uplo = 'L', kd = 2)
        V.@gate_known test_noboxing P.pbtrf!(mkband(Float64); uplo = 'L', kd = 2)
        V.@gate_known test_noboxing P.pbtrs!(copy(ABsd), randn(n, 2); uplo = 'L', kd = 2)
        V.@gate_known test_noboxing P.pbtrf!(mkband(ComplexF64); uplo = 'L', kd = 2)
        wideband(T, kd) = (
            AB = zeros(T, kd + 1, n); AB[1, :] .= T(4 * kd);
            for d in 1:kd
                AB[1 + d, 1:(n - d)] .= T(0.5)
            end; AB
        )
        wideU(T, kd) = (
            AB = zeros(T, kd + 1, n); AB[kd + 1, :] .= T(4 * kd);
            for d in 1:kd
                AB[kd + 1 - d, (1 + d):n] .= T(0.5)
            end; AB
        )
        V.@gate_known test_noboxing P.pbtrf!(wideband(Float64, 64); uplo = 'L', kd = 64)
        V.@gate_known test_noboxing P.pbtrf!(wideU(Float64, 64); uplo = 'U', kd = 64)
        V.@gate_known test_noboxing P._pbtrf_blocked!(wideband(Float64, 64), n, 64)
        V.@gate_known test_noboxing P._pbtrf_repack_U!(wideU(Float64, 64), n, 64)
        V.@gate_known test_noboxing P._pbtrf_blocked_U!(wideU(Float64, 64), n, 64)
        V.@gate_known test_noboxing P._pbtrf_blocked!(wideband(ComplexF64, 32), n, 32)
        V.@gate_known test_noboxing P._pbtrf_repack_U!(wideU(ComplexF64, 32), n, 32)
        V.@gate_known test_noboxing P._pbtrf_blocked_U!(wideU(ComplexF64, 32), n, 32)
        pack(M) = [M[i, j] for j in 1:n for i in j:n]              # lower column-packed
        APsd = pack(Sd); P.pptrf!(APsd; uplo = 'L')
        V.@gate_known test_noboxing P.pptrf!(pack(Sd); uplo = 'L')
        V.@gate_known test_noboxing P.pptrs!(copy(APsd), randn(n, 2); uplo = 'L')
        V.@gate_known test_noboxing P.pptrf!(pack(She); uplo = 'L')
        @test true
    finally
        for f in barriers
            delete!(StrictMode._ALLOC_BARRIERS, f)
        end
        StrictMode.clear_cache!()
    end
end

# ── INLINED: small helpers and default-argument forwarders ─────────────────────────────────────────
# `@assert_inlined` fails when the call survives as an `:invoke` in the caller's optimized IR. The
# 7-argument `_microkernel!` is the defaulted-`Val{B0}` forwarder: it inlines, leaving the call to
# the generated 8-argument body, which is deliberately out of line (a large `@generated` kernel) and
# is recorded as such.
@testitem "StrictTypes: small helpers and default-argument forwarders inline" tags = [:checks] setup = [StrictTypesT] begin
    using StrictModeTest, StrictMode
    P = PureBLAS; V = StrictTypesT
    V.@gateS assert_inlined P._vwidth(Float64)
    V.@gateS assert_inlined P._vwidth(Float32)
    V.@gateS assert_inlined P._lanes(P._HW, Float64)
    V.@gateS assert_inlined P._at_gemm_mr(P._HW)
    V.@gateS assert_inlined P._at_gemm_mr(P._HW, Float32)
    W = P._vwidth(Float64); mr = P._MR * W; nr = P._NR; kc = 64
    Ap = randn(mr * kc); Bp = randn(nr * kc); C = zeros(mr, nr)
    GC.@preserve Ap Bp C begin
        ap = pointer(Ap); bp = pointer(Bp); cptr = pointer(C); ldc = mr
        V.@gateS assert_inlined P._microkernel!(cptr, ldc, ap, bp, kc, Val(P._MR), Val(P._NR))
        V.@gateS_broken assert_inlined P._microkernel!(cptr, ldc, ap, bp, kc, Val(P._MR), Val(P._NR), Val(false))
        V.@gateS assert_inlined P._microkernel_masked!(cptr, ldc, ap, bp, kc, 11, 5, Val(P._MR), Val(P._NR))
        V.@gateS assert_inlined P._microkernel_clip!(cptr, ldc, ap, bp, kc, Val(P._MR), Val(1), Val(P._NR))
    end
    for T in (Float64, Float32)
        A = randn(T, 64, 64); Apk = zeros(T, 64 * 64); mrr = P._MR * P._vwidth(T)
        V.@gateS assert_inlined P._pack_A_simd!(Apk, A, 0, 0, 32, 16, T(1), mrr)
        V.@gateS assert_inlined P._pack_A_simd_T!(Apk, A, 0, 0, 32, 16, T(1), mrr)
        V.@gateS assert_inlined P._pack_B!(Apk, A, 0, 0, 16, 8, false, P._NR)
        V.@gateS assert_inlined P._pack_B!(Apk, A, 0, 0, 16, 8, false, P._NR, 0)
    end
    let T = ComplexF64
        A = randn(T, 64, 64); mrr = P._CMR * P._vwidth(Float64)
        V.@gateS assert_inlined P._pack_A_cmplx_simd_T!(zeros(Float64, 4096), zeros(Float64, 4096), A, 0, 0, 32, 16, mrr)
    end
    x = randn(64); y = randn(64); A = randn(64, 64)
    V.@gateS assert_inlined P.axpy!(y, 2.0, x)
    V.@gateS assert_inlined P.dot(x, y)
    V.@gateS assert_inlined P.scal!(2.0, x)
    V.@gateS assert_inlined P.gemv!(y, A, x)
    V.@gateS assert_inlined P.trsv!(A, x)
end

# ── EFFECTS: the const-folded tuning helpers ───────────────────────────────────────────────────────
# The helpers that derive tile sizes and cut-offs from the detected hardware (`_at_*(hw)`, `_vwidth`,
# `_lanes`, ...) are `@inline` pure functions of constants. `@assert_effects` reads the effects Julia
# infers for them. `:nothrow` is absent from the helpers that divide by a variable, so each helper
# declares the set that holds; a helper that loses `:consistent`, `:effect_free` or `:terminates`
# stops constant-folding and fails here.
@testitem "StrictTypes: tuning helpers infer foldable effects" tags = [:checks] setup = [StrictTypesT] begin
    using StrictModeTest, StrictMode
    P = PureBLAS; V = StrictTypesT
    hw = P._HW
    full = (:nothrow, :effect_free, :terminates, :consistent)
    fold = (:effect_free, :terminates, :consistent)
    # Positive control for the effect reader: a function that can throw lacks `:nothrow`.
    divs(x::Int) = div(1, x)
    @test !V.@verdictS(assert_effects, divs(2), (:nothrow,)).ok
    @test V.@verdictS(assert_effects, divs(2), fold).ok
    V.@gateS assert_effects P._acc_cap(hw) full
    V.@gateS assert_effects P._acc_cap(hw, Float64) full types = (typeof(hw), Type{Float64})
    V.@gateS assert_effects P._acc_cap(hw, Float32) full types = (typeof(hw), Type{Float32})
    V.@gateS assert_effects P._acc_cap(hw, ComplexF64) full types = (typeof(hw), Type{ComplexF64})
    V.@gateS assert_effects P._acc_cap(hw, ComplexF32) full types = (typeof(hw), Type{ComplexF32})
    V.@gateS assert_effects P._at_axpy_band(hw) full
    V.@gateS assert_effects P._at_axpy_dram(hw) full
    V.@gateS assert_effects P._at_cpotf2_mr(hw) fold
    V.@gateS_broken assert_effects P._at_cpotf2_mr(hw) full
    V.@gateS assert_effects P._at_cpotrf_base(hw) full
    V.@gateS assert_effects P._at_cpotrf_nbmax(hw) full
    V.@gateS assert_effects P._at_gbtrf_cross(hw, Float64) full types = (typeof(hw), Type{Float64})
    V.@gateS assert_effects P._at_gbtrf_cross(hw, Float32) full types = (typeof(hw), Type{Float32})
    V.@gateS assert_effects P._at_gbtrf_cross(hw, ComplexF64) full types = (typeof(hw), Type{ComplexF64})
    V.@gateS assert_effects P._at_gbtrf_cross(hw, ComplexF32) full types = (typeof(hw), Type{ComplexF32})
    V.@gateS assert_effects P._at_gemm_kc(hw) fold
    V.@gateS_broken assert_effects P._at_gemm_kc(hw) full
    V.@gateS assert_effects P._at_gemm_kc(hw, Float64) fold types = (typeof(hw), Type{Float64})
    V.@gateS_broken assert_effects P._at_gemm_kc(hw, Float64) full types = (typeof(hw), Type{Float64})
    V.@gateS assert_effects P._at_gemm_kc(hw, Float32) fold types = (typeof(hw), Type{Float32})
    V.@gateS_broken assert_effects P._at_gemm_kc(hw, Float32) full types = (typeof(hw), Type{Float32})
    V.@gateS assert_effects P._at_gemm_kc(hw, ComplexF64) fold types = (typeof(hw), Type{ComplexF64})
    V.@gateS_broken assert_effects P._at_gemm_kc(hw, ComplexF64) full types = (typeof(hw), Type{ComplexF64})
    V.@gateS assert_effects P._at_gemm_kc(hw, ComplexF32) fold types = (typeof(hw), Type{ComplexF32})
    V.@gateS_broken assert_effects P._at_gemm_kc(hw, ComplexF32) full types = (typeof(hw), Type{ComplexF32})
    V.@gateS assert_effects P._at_gemm_mc(hw) fold
    V.@gateS_broken assert_effects P._at_gemm_mc(hw) full
    V.@gateS assert_effects P._at_gemm_mc(hw, Float64) fold types = (typeof(hw), Type{Float64})
    V.@gateS_broken assert_effects P._at_gemm_mc(hw, Float64) full types = (typeof(hw), Type{Float64})
    V.@gateS assert_effects P._at_gemm_mc(hw, Float32) fold types = (typeof(hw), Type{Float32})
    V.@gateS_broken assert_effects P._at_gemm_mc(hw, Float32) full types = (typeof(hw), Type{Float32})
    V.@gateS assert_effects P._at_gemm_mc(hw, ComplexF64) fold types = (typeof(hw), Type{ComplexF64})
    V.@gateS_broken assert_effects P._at_gemm_mc(hw, ComplexF64) full types = (typeof(hw), Type{ComplexF64})
    V.@gateS assert_effects P._at_gemm_mc(hw, ComplexF32) fold types = (typeof(hw), Type{ComplexF32})
    V.@gateS_broken assert_effects P._at_gemm_mc(hw, ComplexF32) full types = (typeof(hw), Type{ComplexF32})
    V.@gateS assert_effects P._at_gemm_mr(hw) fold
    V.@gateS_broken assert_effects P._at_gemm_mr(hw) full
    V.@gateS assert_effects P._at_gemm_mr(hw, Float64) fold types = (typeof(hw), Type{Float64})
    V.@gateS_broken assert_effects P._at_gemm_mr(hw, Float64) full types = (typeof(hw), Type{Float64})
    V.@gateS assert_effects P._at_gemm_mr(hw, Float32) fold types = (typeof(hw), Type{Float32})
    V.@gateS_broken assert_effects P._at_gemm_mr(hw, Float32) full types = (typeof(hw), Type{Float32})
    V.@gateS assert_effects P._at_gemm_mr(hw, ComplexF64) fold types = (typeof(hw), Type{ComplexF64})
    V.@gateS_broken assert_effects P._at_gemm_mr(hw, ComplexF64) full types = (typeof(hw), Type{ComplexF64})
    V.@gateS assert_effects P._at_gemm_mr(hw, ComplexF32) fold types = (typeof(hw), Type{ComplexF32})
    V.@gateS_broken assert_effects P._at_gemm_mr(hw, ComplexF32) full types = (typeof(hw), Type{ComplexF32})
    V.@gateS assert_effects P._at_gemm_mr1_max(hw) fold
    V.@gateS_broken assert_effects P._at_gemm_mr1_max(hw) full
    V.@gateS assert_effects P._at_gemm_mr1_max(hw, Float64) fold types = (typeof(hw), Type{Float64})
    V.@gateS_broken assert_effects P._at_gemm_mr1_max(hw, Float64) full types = (typeof(hw), Type{Float64})
    V.@gateS assert_effects P._at_gemm_mr1_max(hw, Float32) fold types = (typeof(hw), Type{Float32})
    V.@gateS_broken assert_effects P._at_gemm_mr1_max(hw, Float32) full types = (typeof(hw), Type{Float32})
    V.@gateS assert_effects P._at_gemm_mr1_max(hw, ComplexF64) fold types = (typeof(hw), Type{ComplexF64})
    V.@gateS_broken assert_effects P._at_gemm_mr1_max(hw, ComplexF64) full types = (typeof(hw), Type{ComplexF64})
    V.@gateS assert_effects P._at_gemm_mr1_max(hw, ComplexF32) fold types = (typeof(hw), Type{ComplexF32})
    V.@gateS_broken assert_effects P._at_gemm_mr1_max(hw, ComplexF32) full types = (typeof(hw), Type{ComplexF32})
    V.@gateS assert_effects P._at_gemm_nc(hw) fold
    V.@gateS_broken assert_effects P._at_gemm_nc(hw) full
    V.@gateS assert_effects P._at_gemm_nc(hw, Float64) fold types = (typeof(hw), Type{Float64})
    V.@gateS_broken assert_effects P._at_gemm_nc(hw, Float64) full types = (typeof(hw), Type{Float64})
    V.@gateS assert_effects P._at_gemm_nc(hw, Float32) fold types = (typeof(hw), Type{Float32})
    V.@gateS_broken assert_effects P._at_gemm_nc(hw, Float32) full types = (typeof(hw), Type{Float32})
    V.@gateS assert_effects P._at_gemm_nc(hw, ComplexF64) fold types = (typeof(hw), Type{ComplexF64})
    V.@gateS_broken assert_effects P._at_gemm_nc(hw, ComplexF64) full types = (typeof(hw), Type{ComplexF64})
    V.@gateS assert_effects P._at_gemm_nc(hw, ComplexF32) fold types = (typeof(hw), Type{ComplexF32})
    V.@gateS_broken assert_effects P._at_gemm_nc(hw, ComplexF32) full types = (typeof(hw), Type{ComplexF32})
    V.@gateS assert_effects P._at_gemm_nr(hw) fold
    V.@gateS_broken assert_effects P._at_gemm_nr(hw) full
    V.@gateS assert_effects P._at_gemm_nr(hw, Float64) fold types = (typeof(hw), Type{Float64})
    V.@gateS_broken assert_effects P._at_gemm_nr(hw, Float64) full types = (typeof(hw), Type{Float64})
    V.@gateS assert_effects P._at_gemm_nr(hw, Float32) fold types = (typeof(hw), Type{Float32})
    V.@gateS_broken assert_effects P._at_gemm_nr(hw, Float32) full types = (typeof(hw), Type{Float32})
    V.@gateS assert_effects P._at_gemm_nr(hw, ComplexF64) fold types = (typeof(hw), Type{ComplexF64})
    V.@gateS_broken assert_effects P._at_gemm_nr(hw, ComplexF64) full types = (typeof(hw), Type{ComplexF64})
    V.@gateS assert_effects P._at_gemm_nr(hw, ComplexF32) fold types = (typeof(hw), Type{ComplexF32})
    V.@gateS_broken assert_effects P._at_gemm_nr(hw, ComplexF32) full types = (typeof(hw), Type{ComplexF32})
    V.@gateS assert_effects P._at_gemm_split_max(hw) full
    V.@gateS assert_effects P._at_gemm_split_max(hw, Float64) full types = (typeof(hw), Type{Float64})
    V.@gateS assert_effects P._at_gemm_split_max(hw, Float32) full types = (typeof(hw), Type{Float32})
    V.@gateS assert_effects P._at_gemm_split_max(hw, ComplexF64) full types = (typeof(hw), Type{ComplexF64})
    V.@gateS assert_effects P._at_gemm_split_max(hw, ComplexF32) full types = (typeof(hw), Type{ComplexF32})
    V.@gateS assert_effects P._at_gemm_split_mr(hw) full
    V.@gateS assert_effects P._at_gemm_split_nr(hw) full
    V.@gateS assert_effects P._at_gemm_split_ok(hw) full
    V.@gateS assert_effects P._at_gemm_unpack_max(hw) full
    V.@gateS assert_effects P._at_gemm_unpack_max(hw, Float64) full types = (typeof(hw), Type{Float64})
    V.@gateS assert_effects P._at_gemm_unpack_max(hw, Float32) full types = (typeof(hw), Type{Float32})
    V.@gateS assert_effects P._at_gemm_unpack_max(hw, ComplexF64) full types = (typeof(hw), Type{ComplexF64})
    V.@gateS assert_effects P._at_gemm_unpack_max(hw, ComplexF32) full types = (typeof(hw), Type{ComplexF32})
    V.@gateS assert_effects P._at_gemvn_minner(hw) full
    V.@gateS assert_effects P._at_gemvt_percol_amin(hw) full
    V.@gateS assert_effects P._at_gemvt_percol_xmax(hw) full
    V.@gateS assert_effects P._at_gemvt_perscan(hw) full
    V.@gateS assert_effects P._at_pbtrf_cross(hw, Float64) full types = (typeof(hw), Type{Float64})
    V.@gateS assert_effects P._at_pbtrf_cross(hw, Float32) full types = (typeof(hw), Type{Float32})
    V.@gateS assert_effects P._at_pbtrf_cross(hw, ComplexF32) full types = (typeof(hw), Type{ComplexF32})
    V.@gateS assert_effects P._at_pbtrf_nb(hw, Float32) full types = (typeof(hw), Type{Float32})
    V.@gateS assert_effects P._at_pbtrf_nb(hw, ComplexF64) full types = (typeof(hw), Type{ComplexF64})
    V.@gateS assert_effects P._at_pbtrf_nb(hw, ComplexF32) full types = (typeof(hw), Type{ComplexF32})
    V.@gateS assert_effects P._at_pbtrf_nbs(hw, Float64) full types = (typeof(hw), Type{Float64})
    V.@gateS assert_effects P._at_pbtrf_nbs(hw, Float32) full types = (typeof(hw), Type{Float32})
    V.@gateS assert_effects P._at_pbtrf_ucross(hw) full
    V.@gateS assert_effects P._at_pbtrf_ucross(hw, Float64) full types = (typeof(hw), Type{Float64})
    V.@gateS assert_effects P._at_potrf_udirect(hw) full
    V.@gateS assert_effects P._at_rank_k_pack_cut(hw) fold
    V.@gateS_broken assert_effects P._at_rank_k_pack_cut(hw) full
    V.@gateS assert_effects P._at_rank_k_pack_cut(hw, Float64) fold types = (typeof(hw), Type{Float64})
    V.@gateS_broken assert_effects P._at_rank_k_pack_cut(hw, Float64) full types = (typeof(hw), Type{Float64})
    V.@gateS assert_effects P._at_rank_k_pack_cut(hw, Float32) fold types = (typeof(hw), Type{Float32})
    V.@gateS_broken assert_effects P._at_rank_k_pack_cut(hw, Float32) full types = (typeof(hw), Type{Float32})
    V.@gateS assert_effects P._at_rank_k_pack_cut(hw, ComplexF64) fold types = (typeof(hw), Type{ComplexF64})
    V.@gateS_broken assert_effects P._at_rank_k_pack_cut(hw, ComplexF64) full types = (typeof(hw), Type{ComplexF64})
    V.@gateS assert_effects P._at_rank_k_pack_cut(hw, ComplexF32) fold types = (typeof(hw), Type{ComplexF32})
    V.@gateS_broken assert_effects P._at_rank_k_pack_cut(hw, ComplexF32) full types = (typeof(hw), Type{ComplexF32})
    V.@gateS assert_effects P._at_strassen_min(hw) full
    V.@gateS assert_effects P._at_syrk_pack_cut(hw) fold
    V.@gateS_broken assert_effects P._at_syrk_pack_cut(hw) full
    V.@gateS assert_effects P._at_syrk_pack_cut(hw, Float64) fold types = (typeof(hw), Type{Float64})
    V.@gateS_broken assert_effects P._at_syrk_pack_cut(hw, Float64) full types = (typeof(hw), Type{Float64})
    V.@gateS assert_effects P._at_syrk_pack_cut(hw, Float32) fold types = (typeof(hw), Type{Float32})
    V.@gateS_broken assert_effects P._at_syrk_pack_cut(hw, Float32) full types = (typeof(hw), Type{Float32})
    V.@gateS assert_effects P._at_syrk_pack_cut(hw, ComplexF64) fold types = (typeof(hw), Type{ComplexF64})
    V.@gateS_broken assert_effects P._at_syrk_pack_cut(hw, ComplexF64) full types = (typeof(hw), Type{ComplexF64})
    V.@gateS assert_effects P._at_syrk_pack_cut(hw, ComplexF32) fold types = (typeof(hw), Type{ComplexF32})
    V.@gateS_broken assert_effects P._at_syrk_pack_cut(hw, ComplexF32) full types = (typeof(hw), Type{ComplexF32})
    V.@gateS assert_effects P._at_tail_vw(hw) full
    V.@gateS assert_effects P._at_tail_vw(hw, Float64) full types = (typeof(hw), Type{Float64})
    V.@gateS assert_effects P._at_tail_vw(hw, Float32) full types = (typeof(hw), Type{Float32})
    V.@gateS assert_effects P._at_tail_vw(hw, ComplexF64) full types = (typeof(hw), Type{ComplexF64})
    V.@gateS assert_effects P._at_tail_vw(hw, ComplexF32) full types = (typeof(hw), Type{ComplexF32})
    V.@gateS assert_effects P._at_trmm_rpack(hw) full
    V.@gateS assert_effects P._at_zaxpy_narrow(hw) full
    V.@gateS assert_effects P._avoid_po2(1024, 8) full
    V.@gateS assert_effects P._datapath_bytes(hw) full
    V.@gateS assert_effects P._double_pumped(hw) full
    V.@gateS assert_effects P._lanes(hw, Float64) full types = (typeof(hw), Type{Float64})
    V.@gateS assert_effects P._lanes(hw, Float32) full types = (typeof(hw), Type{Float32})
    V.@gateS assert_effects P._lanes(hw, ComplexF64) full types = (typeof(hw), Type{ComplexF64})
    V.@gateS assert_effects P._lanes(hw, ComplexF32) full types = (typeof(hw), Type{ComplexF32})
    V.@gateS assert_effects P._round_dn(1024, 8) fold
    V.@gateS_broken assert_effects P._round_dn(1024, 8) full
    V.@gateS assert_effects P._vwidth(Float64) full types = (Type{Float64},)
    V.@gateS assert_effects P._vwidth(Float32) full types = (Type{Float32},)
    V.@gateS assert_effects P._vwidth(ComplexF64) full types = (Type{ComplexF64},)
    V.@gateS assert_effects P._vwidth(ComplexF32) full types = (Type{ComplexF32},)
    V.@gateS assert_effects P._way_doubles(1024, 8) fold
    V.@gateS_broken assert_effects P._way_doubles(1024, 8) full
    V.@gateS assert_effects P._wide_simd(hw) full
end

# ── TRIM COMPATIBILITY: the exact tolerated set ────────────────────────────────────────────────────
# `@test_trim_compatible` runs juliac's verifier in-process against stock Base. Every signature below
# is run WITHOUT a tolerance and must be accepted, except the ones in `ALLOW`: those are rejected only
# for `Base._str_sizehint` / `Base.print`, which stock Base leaves unresolved for a string interpolation
# on a throw path (any interpolation, not just long ones; see StrictModeTest issue 19) and juliac's
# patched Base resolves. An allowed signature that is accepted, or rejected for any other call, fails.
@testitem "StrictTypes: trim verifier accepts every LAPACK signature except the interpolated-throw set" tags = [:checks] setup = [StrictTypesT] begin
    using StrictModeTest, StrictMode, TrimCheck, LinearAlgebra
    P = PureBLAS; V = StrictTypesT
    ALLOW = Set(
        [
        "P.sysv!('L', copy(Ad), randn(n, 2))",
        "P.sysv!('L', copy(Az), randn(ComplexF64, n, 2))",
        "P.hesv!('L', copy(Ahe), randn(ComplexF64, n, 2))",
        "P.sytrf!(copy(Abd), zeros(Int, nbig); uplo = ul)",
        "P.sytrf!(copy(Abz), zeros(Int, nbig); uplo = ul)",
        "P.hetrf!(copy(Abh), zeros(Int, nbig); uplo = ul)",
        "P.sytri!(copy(LDd), ipd; uplo = 'L')",
        "P.pstrf!(copy(Sd), zeros(Int, n), -1.0; uplo = 'L')",
        "P.pstrf!(copy(She), zeros(Int, n), -1.0; uplo = 'L')",
        "P.gelsd!(randn(n, k), randn(n, 2), -1.0)",
        "P.gelsd!(randn(ComplexF64, n, k), randn(ComplexF64, n, 2), -1.0)",
        "P.gelsy!(randn(n, k), randn(n, 2), zeros(Int, k), -1.0)",
        "P.gelsy!(randn(ComplexF64, n, k), randn(ComplexF64, n, 2), zeros(Int, k), -1.0)",
        "P.gglse!(randn(8, 6), randn(8), randn(4, 6), randn(4))",
        "P.gglse!(randn(ComplexF64, 8, 6), randn(ComplexF64, 8), randn(ComplexF64, 4, 6), randn(ComplexF64, 4))",
        "P.ggev!('N', 'V', randn(n, n), randn(n, n))",
        "P.ggev!('N', 'V', randn(ComplexF64, n, n), randn(ComplexF64, n, n))",
        "P.gges!('V', 'V', randn(n, n), randn(n, n))",
        "P.gges!('V', 'V', randn(ComplexF64, n, n), randn(ComplexF64, n, n))",
        "P.sygvd!(1, 'V', 'U', copy(Ad), copy(Sd))",
        "P.hegvd!(1, 'V', 'U', copy(Ahe), copy(She))",
        "P.gbtrf!(2, 1, n, randn(6, n))",
        "P.gbtrf!(2, 1, n, randn(ComplexF64, 6, n))",
        "P.gbtrf!(32, 8, 64, randn(ldgb, 64))",
        "P.gbtrf!(32, 8, 64, randn(ComplexF64, ldgb, 64))",
        "P.pbtrf!(mkband(Float64); uplo = 'L', kd = 2)",
        "P.pbtrf!(mkband(ComplexF64); uplo = 'L', kd = 2)",
        "P.pbtrf!(wideband(Float64, 64); uplo = 'L', kd = 64)",
        "P.pbtrf!(wideU(Float64, 64); uplo = 'U', kd = 64)",
        "P._pbtrf_blocked!(wideband(Float64, 64), n, 64)",
        "P._pbtrf_repack_U!(wideU(Float64, 64), n, 64)",
        "P._pbtrf_blocked_U!(wideU(Float64, 64), n, 64)",
        "P._pbtrf_blocked!(wideband(ComplexF64, 32), n, 32)",
        "P._pbtrf_repack_U!(wideU(ComplexF64, 32), n, 32)",
        "P._pbtrf_blocked_U!(wideU(ComplexF64, 32), n, 32)",
        "P.pptrf!(pack(Sd); uplo = 'L')",
        "P.pptrf!(pack(She); uplo = 'L')",
        ]
    )
    # Controls: the check fails on a signature that is rejected for another reason, and accepts a clean one.
    clean(x::Int) = x + 1
    unresolved(f::Function) = Base.inferencebarrier(f)(1)         # call through an untyped value
    @test V.@verdict(test_trim_compatible, clean(1)).ok
    @test !V.@verdict(test_trim_compatible, unresolved(identity)).ok
    n = 24; k = 12
    Ad = (M = randn(n, n); M + transpose(M)); Az = (M = randn(ComplexF64, n, n); M + transpose(M))
    Ahe = (M = randn(ComplexF64, n, n); M + M')
    Sd = (M = randn(n, n); M * transpose(M) + n * I)            # real SPD
    She = (M = randn(ComplexF64, n, n); M * M' + n * I)         # Hermitian PD

    V.trim_check(V.@verdict(test_trim_compatible, P.sysv!('L', copy(Ad), randn(n, 2))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.sysv!('L', copy(Az), randn(ComplexF64, n, 2))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.hesv!('L', copy(Ahe), randn(ComplexF64, n, 2))), ALLOW)
    LDd = copy(Ad); ipd = zeros(Int, n); P.sytrf!(LDd, ipd; uplo = 'L')
    LDz = copy(Az); ipz = zeros(Int, n); P.sytrf!(LDz, ipz; uplo = 'L')
    LDh = copy(Ahe); iph = zeros(Int, n); P.hetrf!(LDh, iph; uplo = 'L')
    nbig = 96
    Abd = (M = randn(nbig, nbig); M + transpose(M))
    Abz = (M = randn(ComplexF64, nbig, nbig); M + transpose(M))
    Abh = (M = randn(ComplexF64, nbig, nbig); M + M')
    ipb = zeros(Int, nbig)
    P.sytrf!(copy(Abd), ipb; uplo = 'L')                    # warm the owned W workspace first
    P.hetrf!(copy(Abh), ipb; uplo = 'L')
    for ul in ('L', 'U')
        V.trim_check(V.@verdict(test_trim_compatible, P.sytrf!(copy(Abd), zeros(Int, nbig); uplo = ul)), ALLOW)
        V.trim_check(V.@verdict(test_trim_compatible, P.sytrf!(copy(Abz), zeros(Int, nbig); uplo = ul)), ALLOW)
        V.trim_check(V.@verdict(test_trim_compatible, P.hetrf!(copy(Abh), zeros(Int, nbig); uplo = ul)), ALLOW)
    end
    V.trim_check(V.@verdict(test_trim_compatible, P.sytri!(copy(LDd), ipd; uplo = 'L')), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.sytri!(copy(LDz), ipz; uplo = 'L')), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.hetri!(copy(LDh), iph; uplo = 'L')), ALLOW)
    Aqd = randn(n, k); tqd = zeros(Float64, k); Aqz = randn(ComplexF64, n, k); tqz = zeros(ComplexF64, k)
    Fqd = copy(Aqd); P.geqlf!(Fqd, tqd); Fqz = copy(Aqz); P.geqlf!(Fqz, tqz)
    V.trim_check(V.@verdict(test_trim_compatible, P.geqlf!(copy(Aqd), zeros(Float64, k))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.orgql!(copy(Fqd), copy(tqd))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.ormql!('L', 'N', copy(Fqd), tqd, randn(n, 3))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.geqlf!(copy(Aqz), zeros(ComplexF64, k))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.orgql!(copy(Fqz), copy(tqz))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.ormql!('L', 'N', copy(Fqz), tqz, randn(ComplexF64, n, 3))), ALLOW)
    Ard = randn(k, n); Arz = randn(ComplexF64, k, n)
    Grd = copy(Ard); P.gerqf!(Grd, tqd); Grz = copy(Arz); P.gerqf!(Grz, tqz)
    V.trim_check(V.@verdict(test_trim_compatible, P.gerqf!(copy(Ard), zeros(Float64, k))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.orgrq!(copy(Grd), copy(tqd))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.ormrq!('R', 'N', copy(Grd), tqd, randn(3, n))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.gerqf!(copy(Arz), zeros(ComplexF64, k))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.orgrq!(copy(Grz), copy(tqz))), ALLOW)
    Tzd = randn(k, n); ttd = zeros(Float64, k); P.tzrzf!(Tzd, ttd)
    V.trim_check(V.@verdict(test_trim_compatible, P.tzrzf!(randn(k, n), zeros(Float64, k))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.tzrzf!(randn(ComplexF64, k, n), zeros(ComplexF64, k))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.ormrz!('L', 'N', copy(Tzd), ttd, randn(n, 3))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.pstrf!(copy(Sd), zeros(Int, n), -1.0; uplo = 'L')), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.pstrf!(copy(She), zeros(Int, n), -1.0; uplo = 'L')), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.gelsd!(randn(n, k), randn(n, 2), -1.0)), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.gelsd!(randn(ComplexF64, n, k), randn(ComplexF64, n, 2), -1.0)), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.gelsy!(randn(n, k), randn(n, 2), zeros(Int, k), -1.0)), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.gelsy!(randn(ComplexF64, n, k), randn(ComplexF64, n, 2), zeros(Int, k), -1.0)), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.gglse!(randn(8, 6), randn(8), randn(4, 6), randn(4))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.gglse!(randn(ComplexF64, 8, 6), randn(ComplexF64, 8), randn(ComplexF64, 4, 6), randn(ComplexF64, 4))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.trsyl!('N', 'N', 1, triu(randn(n, n)) + n * I, triu(randn(n, n)) + n * I, randn(n, n))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.trsyl!('N', 'N', 1, triu(randn(ComplexF64, n, n)) + n * I, triu(randn(ComplexF64, n, n)) + n * I, randn(ComplexF64, n, n))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.trexc!('V', triu(randn(n, n)) + n * I, Matrix(1.0I, n, n), 2, n - 1)), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.trexc!('V', triu(randn(ComplexF64, n, n)) + n * I, Matrix(ComplexF64(1)I, n, n), 2, n - 1)), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.trsen!('N', 'V', rand(Bool, n), triu(randn(n, n)) + n * I, Matrix(1.0I, n, n))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.trsen!('N', 'V', rand(Bool, n), triu(randn(ComplexF64, n, n)) + n * I, Matrix(ComplexF64(1)I, n, n))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.ggev!('N', 'V', randn(n, n), randn(n, n))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.ggev!('N', 'V', randn(ComplexF64, n, n), randn(ComplexF64, n, n))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.gges!('V', 'V', randn(n, n), randn(n, n))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.gges!('V', 'V', randn(ComplexF64, n, n), randn(ComplexF64, n, n))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.ggsvd!('U', 'V', 'Q', randn(8, 6), randn(6, 6))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.ggsvd!('U', 'V', 'Q', randn(ComplexF64, 8, 6), randn(ComplexF64, 6, 6))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.sygvd!(1, 'V', 'U', copy(Ad), copy(Sd))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.hegvd!(1, 'V', 'U', copy(Ahe), copy(She))), ALLOW)
    dd = randn(n); ee = randn(n - 1)
    V.trim_check(V.@verdict(test_trim_compatible, P.stebz!('A', 'B', 0.0, 0.0, 1, n, 0.0, copy(dd), copy(ee))), ALLOW)
    w, ib, isp, _ = P.stebz!('A', 'B', 0.0, 0.0, 1, n, 0.0, copy(dd), copy(ee))
    V.trim_check(V.@verdict(test_trim_compatible, P.stein!(copy(dd), copy(ee), w, ib, isp)), ALLOW)
    ABd2 = randn(6, n); _, gip, _ = P.gbtrf!(2, 1, n, ABd2)
    V.trim_check(V.@verdict(test_trim_compatible, P.gbtrf!(2, 1, n, randn(6, n))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.gbtrs!('N', 2, 1, n, copy(ABd2), gip, randn(n, 2))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.gbtrf!(2, 1, n, randn(ComplexF64, 6, n))), ALLOW)
    ldgb = 2 * 32 + 8 + 1
    P.gbtrf!(32, 8, 64, randn(ldgb, 64))                 # warm the owned workspace first
    P.gbtrf!(32, 8, 64, randn(ComplexF64, ldgb, 64))
    V.trim_check(V.@verdict(test_trim_compatible, P.gbtrf!(32, 8, 64, randn(ldgb, 64))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.gbtrf!(32, 8, 64, randn(ComplexF64, ldgb, 64))), ALLOW)
    Dp = fill(4.0, n); Ep = fill(1.0, n - 1); Df = copy(Dp); Ef = copy(Ep); P.pttrf!(Df, Ef)
    V.trim_check(V.@verdict(test_trim_compatible, P.pttrf!(fill(4.0, n), fill(1.0, n - 1))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.pttrs!(copy(Df), copy(Ef), randn(n, 2))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.ptsv!(fill(4.0, n), fill(1.0, n - 1), randn(n, 2))), ALLOW)
    Dpz = fill(4.0, n); Epz = fill(1.0 + 0im, n - 1)
    V.trim_check(V.@verdict(test_trim_compatible, P.ptsv!(copy(Dpz), copy(Epz), randn(ComplexF64, n, 2))), ALLOW)
    dl = fill(1.0, n - 1); dm = fill(4.0, n); du = fill(1.0, n - 1)
    du2 = zeros(Float64, n - 2); gtip = zeros(Int, n); P.gttrf!(copy(dl), copy(dm), copy(du), du2, gtip)
    V.trim_check(V.@verdict(test_trim_compatible, P.gtsv!(copy(dl), copy(dm), copy(du), randn(n, 2))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.gttrf!(copy(dl), copy(dm), copy(du), zeros(Float64, n - 2), zeros(Int, n))), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.gtsv!(fill(1.0 + 0im, n - 1), fill(4.0 + 0im, n), fill(1.0 + 0im, n - 1), randn(ComplexF64, n, 2))), ALLOW)
    mkband(T) = (AB = zeros(T, 3, n); AB[1, :] .= T(10); AB[2, 1:(n - 1)] .= T(1); AB[3, 1:(n - 2)] .= T(0.5); AB)
    ABsd = mkband(Float64); P.pbtrf!(ABsd; uplo = 'L', kd = 2)
    V.trim_check(V.@verdict(test_trim_compatible, P.pbtrf!(mkband(Float64); uplo = 'L', kd = 2)), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.pbtrs!(copy(ABsd), randn(n, 2); uplo = 'L', kd = 2)), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.pbtrf!(mkband(ComplexF64); uplo = 'L', kd = 2)), ALLOW)
    wideband(T, kd) = (
        AB = zeros(T, kd + 1, n); AB[1, :] .= T(4 * kd);
        for d in 1:kd
            AB[1 + d, 1:(n - d)] .= T(0.5)
        end; AB
    )
    wideU(T, kd) = (
        AB = zeros(T, kd + 1, n); AB[kd + 1, :] .= T(4 * kd);
        for d in 1:kd
            AB[kd + 1 - d, (1 + d):n] .= T(0.5)
        end; AB
    )
    V.trim_check(V.@verdict(test_trim_compatible, P.pbtrf!(wideband(Float64, 64); uplo = 'L', kd = 64)), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.pbtrf!(wideU(Float64, 64); uplo = 'U', kd = 64)), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P._pbtrf_blocked!(wideband(Float64, 64), n, 64)), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P._pbtrf_repack_U!(wideU(Float64, 64), n, 64)), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P._pbtrf_blocked_U!(wideU(Float64, 64), n, 64)), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P._pbtrf_blocked!(wideband(ComplexF64, 32), n, 32)), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P._pbtrf_repack_U!(wideU(ComplexF64, 32), n, 32)), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P._pbtrf_blocked_U!(wideU(ComplexF64, 32), n, 32)), ALLOW)
    pack(M) = [M[i, j] for j in 1:n for i in j:n]              # lower column-packed
    APsd = pack(Sd); P.pptrf!(APsd; uplo = 'L')
    V.trim_check(V.@verdict(test_trim_compatible, P.pptrf!(pack(Sd); uplo = 'L')), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.pptrs!(copy(APsd), randn(n, 2); uplo = 'L')), ALLOW)
    V.trim_check(V.@verdict(test_trim_compatible, P.pptrf!(pack(She); uplo = 'L')), ALLOW)
    @test true
end

# ── @strict_policy ───────────────────────────────────────────────────────────────────────────────────
# `@strict_policy (:typestable,)` placed last in a module checks every definition the module made.
# It cannot be applied to `PureBLAS` from a test without registering a policy on the live package
# module, so the controls run on throwaway modules; each one must reject a violating definition and
# accept a clean one.
@testitem "StrictTypes: @strict_policy rejects violations and unknown guarantees" tags = [:checks] begin
    using StrictMode, StrictModeTest
    function policy(exprs...)
        m = Module(:StrictPolicyToy)
        Core.eval(m, :(using StrictMode))
        for ex in exprs
            Core.eval(m, ex)
        end
        return nothing
    end
    clean = :(good(x::Float64) = 2x)
    @test policy(clean, :(@strict_policy (:typestable,))) === nothing
    # return type `Any`
    @test_throws StrictMode.StrictViolation policy(clean, :(bad(x::Vector{Any}) = x[1]), :(@strict_policy (:typestable,)))
    # concrete return, dispatch inside: the policy reads the IR, so it sees this too
    @test_throws StrictMode.StrictViolation policy(:(f(v::Vector{Any}) = (x = v[1]; sin(x)::Float64)), :(@strict_policy (:typestable,)))
    # a misspelled guarantee is an error, not a silent no-op
    @test_throws "unknown guarantee" policy(clean, :(@strict_policy (:typestabel,)))
    # `exempt` scopes by function name
    @test policy(clean, :(bad(x::Vector{Any}) = x[1]), :(@strict_policy (:typestable,) exempt = [:bad])) === nothing
end
