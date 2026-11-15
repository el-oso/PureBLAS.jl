# Package-level StrictMode drivers and reports. Every check on PureBLAS comes with a control on a toy
# module (or a deliberately incomplete backend) that must flag, so a green result cannot be vacuous.

@testitem "StrictMode package: backend satisfies all four strict contracts" tags = [:checks] begin
    using StrictMode, StrictModeTest, TypeContracts
    P = PureBLAS
    contracts = (P.AbstractBLAS1, P.AbstractBLAS2, P.AbstractBLAS3, P.AbstractLAPACK)

    # `@verify` walks the supertype chain of SIMDBackend, checks every declared member's method
    # existence and inferred return type, and seals `verified_trait` for each contract in the chain.
    TypeContracts.@verify P.SIMDBackend
    for I in contracts
        @test TypeContracts.verified_trait(I, P.SIMDBackend) === TypeContracts.Implemented{I}()
        @test length(TypeContracts.list_contract(I)) > 0
        @test I in keys(TypeContracts.registered_contracts())
    end

    # Control: a backend one member short of AbstractBLAS1 is rejected, and accepted once the member
    # exists.
    struct ToyBackend <: P.AbstractBLAS1 end
    P.axpy!(::ToyBackend, y::AbstractVector, a::Number, x::AbstractVector)::AbstractVector = y
    P.scal!(::ToyBackend, a::Number, x::AbstractVector)::AbstractVector = x
    P.blascopy!(::ToyBackend, y::AbstractVector, x::AbstractVector)::AbstractVector = y
    P.swap!(::ToyBackend, x::AbstractVector, y::AbstractVector)::Nothing = nothing
    P.dot(::ToyBackend, x::AbstractVector, y::AbstractVector)::Number = 0.0
    P.dotu(::ToyBackend, x::AbstractVector, y::AbstractVector)::Number = 0.0
    P.nrm2(::ToyBackend, x::AbstractVector)::Real = 0.0
    P.asum(::ToyBackend, x::AbstractVector)::Real = 0.0
    @test_throws "iamax" (TypeContracts.@verify ToyBackend)
    @test TypeContracts.verified_trait(P.AbstractBLAS1, ToyBackend) === TypeContracts.NotImplemented{P.AbstractBLAS1}()
    P.iamax(::ToyBackend, x::AbstractVector)::Integer = 1
    TypeContracts.@verify ToyBackend
    @test TypeContracts.verified_trait(P.AbstractBLAS1, ToyBackend) === TypeContracts.Implemented{P.AbstractBLAS1}()
end

@testitem "StrictMode package: _trsv_reg! specialization count decomposes into Val combos x array-type pairs" tags = [:checks] begin
    using StrictMode, LinearAlgebra
    P = PureBLAS
    for T in (Float64, Float32)
        A = Matrix{T}(I, 24, 24) .+ T(0.5)
        x = ones(T, 24)
        for ul in ('U', 'L'), tr in ('N', 'T'), dg in ('N', 'U')
            P.trsv!(P.SIMDBackend(), A, copy(x); uplo = ul, trans = tr, diag = dg)
        end
    end
    m = only(methods(P._trsv_reg!))
    sigs = [s.specTypes for s in Base.specializations(m) if isdefined(s, :specTypes)]
    vals = Set(Tuple(s.parameters[2:5]) for s in sigs)
    pairs = Set(Tuple(s.parameters[6:7]) for s in sigs)
    @test length(sigs) == length(vals) * length(pairs)
    @test length(vals) == 128        # 8 shapes x 16 sizes: the generated ladder
    # The two trailing `(A, x)` array types are used only for `pointer`/`stride`; normalizing them at the
    # ladder boundary leaves one pair per element type.
    @test_broken length(pairs) <= 2

    # specialization_report counts what this decomposition counts.
    f = specialization_report(P; only = [:_trsv_reg!], max_specializations = 1024, max_type_depth = typemax(Int))
    @test length(f) == 1
    @test occursin("$(length(sigs)) compiled specializations", string(only(f)))

    # Control: a toy method specialized 40 times is flagged at limit 32 and not at limit 64.
    @eval module ToySpec
    f(::Val{N}) where {N} = N
    end
    for i in 1:40
        ToySpec.f(Val(i))
    end
    @test length(specialization_report(ToySpec; max_specializations = 32)) == 1
    @test isempty(specialization_report(ToySpec; max_specializations = 64))
end

@testitem "StrictMode package: test_no_unsafe on the level3.jl functions, allow-set by construct" tags = [:checks] begin
    using StrictMode, StrictModeTest
    P = PureBLAS
    f3 = joinpath(pkgdir(P), "src", "blas3", "level3.jl")
    scope = Symbol[]
    for n in names(P; all = true)
        isdefined(P, n) || continue
        v = getfield(P, n)
        v isa Function || continue
        any(m -> String(m.file) == f3, methods(v)) && push!(scope, n)
    end
    @test length(scope) > 100        # the file's functions, not an empty selector

    allow = (:inbounds, :unsafe_load, :unsafe_store!, :pointer, :unsafe_trunc)
    @test test_no_unsafe(P; only = scope, allow) isa Vector     # throws on any construct outside `allow`

    # Controls: with no allowance the scope fails; dropping `:pointer` alone also fails.
    @test_throws "safety guarantees" test_no_unsafe(P; only = scope)
    @test_throws "pointer" test_no_unsafe(P; only = scope, allow = filter(!=(:pointer), allow))

    @eval module ToyRaw
    raw(p::Ptr{Float64}) = unsafe_load(p)
    end
    r = Ref(1.0)
    GC.@preserve r ToyRaw.raw(Base.unsafe_convert(Ptr{Float64}, r))
    @test_throws "unsafe_load" test_no_unsafe(ToyRaw)
    @test test_no_unsafe(ToyRaw; allow = (:unsafe_load,)) isa Vector
end

@testitem "StrictMode package: reports on the entry points and C-ABI shims" tags = [:checks] begin
    using StrictMode, StrictModeTest, LinearAlgebra
    P = PureBLAS
    n = 48
    A = randn(n, n); B = randn(n, n); C = zeros(n, n); x = randn(n); y = randn(n)
    for _ in 1:2
        P.axpy!(y, 2.0, x); P.dot(x, y); P.nrm2(x); P.gemv!(y, A, x); P.gemm!(C, A, B)
        P.potrf!(A * A' + n * I); P.getrf!(copy(A)); P.syrk!(C, A); P.trsm!(copy(B), tril(A) + n * I)
        nm = Ref(Int64(n)); one_ = Ref(1.0); zero_ = Ref(0.0); ch = Ref(UInt8('N')); inc = Ref(Int64(1))
        GC.@preserve A B C x y nm one_ zero_ ch inc begin
            p(r) = Base.unsafe_convert(Ptr{eltype(r)}, r)
            P.daxpy_64_(p(nm), p(one_), pointer(x), p(inc), pointer(y), p(inc))
            P.ddot_64_(p(nm), pointer(x), p(inc), pointer(y), p(inc))
            P.dnrm2_64_(p(nm), pointer(x), p(inc))
            P.dgemv_64_(p(ch), p(nm), p(nm), p(one_), pointer(A), p(nm), pointer(x), p(inc), p(zero_), pointer(y), p(inc), 1)
            P.dgemm_64_(p(ch), p(ch), p(nm), p(nm), p(nm), p(one_), pointer(A), p(nm), pointer(B), p(nm), p(zero_), pointer(C), p(nm), 1, 1)
        end
    end
    scope = [:axpy!, :dot, :nrm2, :gemv!, :gemm!, :trsm!, :syrk!, :potrf!, :getrf!,
        :daxpy_64_, :ddot_64_, :dgemv_64_, :dgemm_64_, :dtrsm_64_, :dpotrf_64_, :dgetrf_64_, :zgemm_64_, :dnrm2_64_]

    @eval module ToyRep
    struct Box
        v::Any
    end
    dyn(b::Box) = b.v + 1
    stat(b::Box) = 1
    ptrset!(x::Vector{Float64}) = (GC.@preserve x unsafe_store!(pointer(x), 1.0); x)
    struct WS
        buf::Vector{Float64}
    end
    borrow(w::WS) = w.buf
    end
    ToyRep.dyn(ToyRep.Box(1)); ToyRep.stat(ToyRep.Box(1)); ToyRep.ptrset!([0.0])
    StrictMode.register_scoped!(ToyRep.WS)
    ToyRep.borrow(ToyRep.WS([1.0]))

    # dispatch_report: no dynamic or at-the-limit call on the entry points; the toy has one.
    d = dispatch_report(P; only = scope)
    @test d.nclear > 0
    @test isempty(d.dynamic)
    @test length(dispatch_report(ToyRep; only = [:dyn]).dynamic) == 1
    @test isempty(dispatch_report(ToyRep; only = [:stat]).dynamic)

    # bang_report: `axpy!` writes `y` through pointer stores, which the report does not model, so the
    # entry is flagged although it mutates. The toy shows the same miss on a minimal pointer writer.
    @test_broken isempty(bang_report(P; only = [:axpy!]))
    @test length(bang_report(ToyRep; only = [:ptrset!])) == 1

    # resource_leak_report is empty unless an owning type is registered; the control registers one.
    @test isempty(resource_leak_report(P; only = scope))
    @test length(resource_leak_report(ToyRep; only = [:borrow])) == 1
end

@testitem "StrictMode package: registering functions makes test_registered and test_coverage bind" tags = [:checks] begin
    using StrictMode, StrictModeTest
    # PureBLAS registers nothing, so `test_registered` has no signature to prove and `test_coverage`
    # fails on every compiled signature. A package adopting the drivers declares its entries like this:
    @eval module ToyPkg
    using StrictMode
    @strict_function scale3(a::NTuple{3,Float64}, s::Float64) = (a[1] * s, a[2] * s, a[3] * s)
    undeclared(x::Float64) = x + 1.0
    end
    ToyPkg.scale3((1.0, 2.0, 3.0), 2.0); ToyPkg.undeclared(1.0)

    # The declaration is proved ...
    @test test_registered(; modules = [ToyPkg]) isa Vector
    # ... and the undeclared method is reported until it is declared or visibly exempt.
    @test_throws "undeclared" test_coverage(ToyPkg)
    StrictMode.register_strict!(ToyPkg.undeclared, (Float64,); guarantees = (:typestable, :noalloc))
    @test test_coverage(ToyPkg) isa Vector

    # Control: a registered signature that violates its guarantee is rejected by test_registered.
    @eval module ToyBad
    using StrictMode
    unstable(x::Float64) = Ref{Any}(x)[]
    end
    ToyBad.unstable(1.0)
    StrictMode.register_strict!(ToyBad.unstable, (Float64,); guarantees = (:typestable,))
    @test_throws "unstable" test_registered(; modules = [ToyBad])
end

@testitem "StrictMode package: test_layers and test_exemptions controls" tags = [:checks] begin
    using StrictMode, StrictModeTest
    @eval module KernelL
    area(x) = x * x
    end
    @eval module AppL
    import ..KernelL
    KernelL.area(x::String) = 0
    spin(x) = x
    end
    @test test_layers(AppL => (KernelL,), KernelL => ()) isa Vector

    @eval module AppL2
    spin(x) = x
    end
    @eval module BadL2
    import ..AppL2
    AppL2.spin(x::String) = 0
    end
    @test_throws "does not declare" test_layers(AppL2 => (), BadL2 => ())

    # An exemption that the proofs find unnecessary is reported (info, never a failure).
    @eval module ToyE
    using StrictMode
    @strict_exempt cheap(x::Float64) = x + 1.0
    end
    ToyE.cheap(1.0)
    @test any(f -> String(f.func) == "cheap", test_exemptions(ToyE))
end

@testitem "StrictMode package: unsafe_report is scoped by name and flags the unsafe constructs" tags = [:checks] begin
    using StrictMode
    P = PureBLAS

    @eval module ToyUnsafe
    raw(p::Ptr{Float64}) = unsafe_load(p)
    inb(x::Vector{Float64}) = @inbounds x[1]
    safe(x::Vector{Float64}) = x[1]
    end
    ToyUnsafe.safe([1.0])
    # Control: each unsafe construct is reported on its own method and the safe method is clean.
    @test length(unsafe_report(ToyUnsafe; only = [:raw])) == 1
    @test length(unsafe_report(ToyUnsafe; only = [:inb])) == 1
    @test isempty(unsafe_report(ToyUnsafe; only = [:safe]))
    @test isempty(unsafe_report(ToyUnsafe; only = [:raw], allow = (:unsafe_load,)))

    # PureBLAS: `_iamax` writes `@inbounds` in three methods and nothing else, so the three findings
    # are all `:inbounds`.
    own = filter(f -> f.func == "_iamax", unsafe_report(P; only = [:_iamax]))
    @test length(own) == 3
    @test all(f -> occursin("@inbounds", f.reason) && !occursin("unsafe_", f.reason), own)
    @test isempty(filter(f -> f.func == "_iamax", unsafe_report(P; only = [:_iamax], allow = (:inbounds,))))

    # `only` does not scope methods without a singleton instance: closure bodies of unrelated
    # functions are reported too, so a scoped report is not a report of the named functions alone.
    @test_broken all(f -> f.func == "_iamax", unsafe_report(P; only = [:_iamax]))
end

@testitem "StrictMode package: conversion_report flags narrowing and widening on a toy and on kernels" tags = [:checks] begin
    using StrictMode
    P = PureBLAS
    @eval module ToyConv
    narrow(x::Float64) = Float32(x) * 2.0f0
    keep(x::Float64) = x * 2.0
    end
    ToyConv.narrow(1.0); ToyConv.keep(1.0)
    r = conversion_report(ToyConv; only = [:narrow])
    @test length(r) == 1 && occursin("Float64 to Float32 (narrowed)", r[1].reason)
    @test isempty(conversion_report(ToyConv; only = [:keep]))
    @test isempty(conversion_report(ToyConv; only = [:narrow], directions = "widened"))

    # PureBLAS, scoped: `_scal_col!` runs Float32 data through Float64 and narrows back. The finding
    # is advisory (status :info), so the assertion pins the finding, not a verdict.
    for T in (Float32, Float64)
        x = randn(T, 64); y = randn(T, 64); A = randn(T, 32, 32); C = zeros(T, 32, 32)
        P.axpy!(y, T(2), x); P.gemm!(C, A, A)
    end
    r = conversion_report(P; only = [:_scal_col!])
    @test length(r) == 1 && r[1].status === :info
    @test occursin("Float32 to Float64 (widened)", r[1].reason)
    @test occursin("Float64 to Float32 (narrowed)", r[1].reason)
end

@testitem "StrictMode package: throw_report lists what a body throws" tags = [:checks] begin
    using StrictMode, LinearAlgebra
    P = PureBLAS
    @eval module ToyThrow
    pos(x::Float64) = x > 0 ? x : throw(DomainError(x, "x must be positive"))
    quiet(x::Float64) = x + 1.0
    end
    @test throw_report(ToyThrow.pos, (Float64,)) == [DomainError]
    @test isempty(throw_report(ToyThrow.quiet, (Float64,)))
    ToyThrow.pos(1.0); ToyThrow.quiet(1.0)
    @test length(throw_report(ToyThrow; only = [:pos])) == 1
    @test isempty(throw_report(ToyThrow; only = [:quiet]))

    # PureBLAS: the Cholesky driver declares its two failure types in its own body; the sweep sees
    # the compiled specialization after a warm call. The report is a catalog, so the count is pinned.
    A = randn(8, 8); S = A * A' + 8I
    P.potrf!(copy(S))
    r = throw_report(P; only = [:potrf!])
    @test length(r) == 1
    @test occursin("DimensionMismatch", r[1].reason) && occursin("PosDefException", r[1].reason)
end

@testitem "StrictMode package: inline_suggestions, static_ownership_suggestions and dispatch_suggestions" tags = [:checks] begin
    using StrictMode, LinearAlgebra
    P = PureBLAS

    @eval module ToyAdv
    const REG = Dict{DataType, Vector{Float64}}(Float64 => [1.0])
    look(::Type{T}) where {T} = REG[T]
    plain(x::Float64) = x
    @noinline cg(x::Float64) = x + 1.0
    @noinline loopcall(v::Vector{Float64}) = (s = 0.0; for i in eachindex(v); s += cg(v[i]); end; s)
    dyn(a::Vector{Any}) = a[1] + 1
    stat(x::Float64) = x + 1.0
    end
    ToyAdv.look(Float64); ToyAdv.plain(1.0); ToyAdv.loopcall([1.0, 2.0]); ToyAdv.dyn(Any[1]); ToyAdv.stat(1.0)

    # Controls: each advisory flags its toy, stays empty on the clean one, and never counts as a failure.
    s = static_ownership_suggestions(ToyAdv; only = [:look])
    @test length(s) == 1 && s[1].status === :info
    @test isempty(static_ownership_suggestions(ToyAdv; only = [:plain]))
    @test nfailures(s) == 0
    i = inline_suggestions(ToyAdv.loopcall, (Vector{Float64},); only_flagged = true)
    @test length(i) >= 1 && all(f -> f.status === :info, i)
    @test length(dispatch_suggestions(ToyAdv.dyn, (Vector{Any},))) == 1
    @test isempty(dispatch_suggestions(ToyAdv.stat, (Float64,)))

    # PureBLAS, scoped to one entry each. `getrf!` reaches 12 type-keyed registry lookups through
    # non-inlined callees (compile-time ownership would remove them); `gemm!`'s hot loop calls
    # @generated microkernels that stay as :invoke; the BLAS-1 entries have no dispatch sites.
    n = 48
    A = randn(n, n); B = randn(n, n); C = zeros(n, n)
    for _ in 1:2
        P.getrf!(copy(A)); P.gemm!(C, A, B); P.dot(A[:, 1], B[:, 1])
    end
    so = static_ownership_suggestions(P; only = [:getrf!])
    @test length(so) == 12
    @test all(f -> f.guarantee === :static_ownership && f.status === :info, so)
    inl = inline_suggestions(P; only = [:gemm!])
    @test !isempty(inl) && all(f -> f.status === :info, inl)
    @test any(f -> occursin("@generated", f.reason), inl)
    @test isempty(dispatch_suggestions(P; only = [:nrm2, :dot]))
end

@testitem "StrictMode package: scan and proof drivers on nrm2 and axpy! (findings, audit, proof_audit, divergence_report)" tags = [:checks] begin
    using StrictMode, StrictModeTest
    P = PureBLAS
    x = randn(48); y = randn(48)
    for _ in 1:2
        P.nrm2(x); P.axpy!(y, 2.0, x)
    end

    @eval module ToyDiv
    boxes(x::Float64) = Ref{Any}(x)[]
    stable(x::Float64) = x + 1.0
    end
    ToyDiv.boxes(1.0); ToyDiv.stable(1.0)

    # Controls. `findings` flags the unstable toy; `divergence_report` records exactly the guarantee
    # on which the scan and the proof disagree; the diff and the audit see a change between runs.
    ft = StrictMode.findings(ToyDiv.boxes, (Float64,))
    @test any(f -> f.guarantee === :typestable && f.status === :fail, ft)
    d = divergence_report(ToyDiv.boxes, (Float64,))
    @test !isempty(d) && d.diverged == [(:noalloc, false, true)]
    @test isempty(divergence_report(ToyDiv.stable, (Float64,)))
    old = StrictMode.findings(ToyDiv.boxes, (Float64,)); new = StrictMode.findings(ToyDiv.stable, (Float64,))
    df = StrictMode.findings_diff(old, new)
    @test length(df.fixed) == 1 && isempty(df.introduced) && isempty(df.changed)
    @test nfailures(proof_audit(ToyDiv; sweep = true, only = [:boxes])) >= 1
    @test nfailures(proof_audit(ToyDiv; sweep = true, only = [:stable])) == 0

    # PureBLAS: the value-free scan passes `nrm2` on all six compiled signatures while AllocCheck
    # fails every one on `noalloc` (196 sites); a call from a function returns 16 bytes per call.
    # The divergence is therefore on `noalloc`, and `proof_audit` carries the proved verdict as data.
    scan = audit(P; sweep = true, only = [:nrm2])
    proof = proof_audit(P; sweep = true, only = [:nrm2])
    @test length(scan) == length(proof) == 12
    @test nfailures(scan) == 0
    @test nfailures(proof) == 6
    @test all(f -> f.guarantee === :noalloc, filter(StrictMode._failed, proof))
    @test all(f -> f.guarantee === :typestable && f.status === :pass, filter(f -> f.guarantee === :typestable, proof))
    dv = divergence_report(P.axpy!, (Vector{Float64}, Float64, Vector{Float64}))
    @test dv.diverged == [(:noalloc, false, true), (:noboxing, false, true)]

    # The same audit run twice against a baseline file sees no movement.
    path = joinpath(mktempdir(), "baseline.bin")
    first_run = audit_changed(P; sweep = true, only = [:nrm2], baseline = path, update = true)
    second_run = audit_changed(P; sweep = true, only = [:nrm2], baseline = path)
    @test isempty(first_run.introduced) && isempty(second_run.introduced) && isempty(second_run.fixed)
end

@testitem "StrictMode package: audit_changed reports what a changed scope fixed or introduced" tags = [:checks] begin
    using StrictMode
    # `only` is the scope a diff range names: the functions whose bodies changed. The baseline holds
    # the earlier run; the second run is the later revision of the same scope.
    @eval module ToyRev
    unst(x::Float64) = Ref{Any}(x)[]
    fine(x::Float64) = x + 1.0
    end
    ToyRev.unst(1.0); ToyRev.fine(1.0)
    path = joinpath(mktempdir(), "rev.bin")
    r0 = audit_changed(ToyRev; sweep = true, baseline = path, update = true)
    @test length(r0.introduced) >= 1                      # an absent baseline reports everything as new
    r1 = audit_changed(ToyRev; sweep = true, baseline = path)
    @test isempty(r1.introduced) && isempty(r1.fixed) && length(r1.unchanged) >= 1
    # The next revision drops the failing function from scope: its finding is reported fixed.
    r2 = audit_changed(ToyRev; sweep = true, baseline = path, exempt = [:unst])
    @test length(r2.fixed) >= 1 && isempty(r2.introduced)
    # A corrupt baseline is reported, never read as empty.
    write(path, "not a baseline")
    @test_throws "not a baseline of findings" audit_changed(ToyRev; sweep = true, baseline = path)
end

@testitem "StrictMode package: test_compiled, test_calls and proof_findings on nrm2" tags = [:checks] begin
    using StrictMode, StrictModeTest
    P = PureBLAS
    x = randn(48)
    for _ in 1:2
        P.nrm2(x)
    end

    @eval module ToyGate
    allocs(n::Int) = sum(zeros(n))
    callbad(x::Float64) = nomethod(x)
    nomethod(::Int) = 1
    fine(x::Float64) = x + 1.0
    end
    ToyGate.allocs(3); ToyGate.fine(1.0)
    try ToyGate.callbad(1.0) catch end

    # Controls: each driver throws on a toy that violates exactly its guarantee and passes a clean one.
    @test_throws "allocates" test_compiled(ToyGate; only = [:allocs])
    @test test_compiled(ToyGate; only = [:fine]) isa Vector
    @test_throws "MethodErrorReport" test_calls(ToyGate; only = [:callbad])
    @test test_calls(ToyGate; only = [:fine]) isa Vector

    # PureBLAS, scoped to `nrm2`: it is type stable on all six compiled signatures and AllocCheck
    # fails `noalloc` on every one, so the gate raises. `proof_findings` returns the same verdicts as
    # data; `only` leaves closure methods outside its filter, so they are excluded by name here.
    @test test_compiled(P; only = [:nrm2], guarantees = (:typestable,)) isa Vector
    @test_throws "allocates" test_compiled(P; only = [:nrm2])
    pf = filter(f -> f.func == "nrm2", StrictModeTest.proof_findings(P; only = [:nrm2]))
    @test length(pf) == 12 && count(StrictMode._failed, pf) == 6

    # `test_calls` finds a threaded copy/scale dispatch with no method for `PtrVector` operands: the
    # `_simd2` guard in `_copy!` admits `Ptr`/`PtrVector` mixes that `_copy_threaded!` does not define.
    @test_throws "_copy_threaded!" test_calls(P; only = [:nrm2])
end

@testitem "StrictMode package: test_latency and latency_report measure a cold child process" tags = [:checks] begin
    using StrictMode, StrictModeTest
    # The child loads PureBLAS from the active project; the precompile cache is built first, so the
    # numbers are warm-cache loads. Wall-clock budgets belong to a pinned host, so only the control
    # budgets (1 ms load, 0.001 ms first call) are asserted, and they must fail on any host.
    r = test_latency(:PureBLAS; calls = ["PureBLAS.nrm2(randn(8))"])
    @test r.load_ms > 0 && length(r.first_call_ms) == 1 && r.precompiled
    @test_throws "load took" test_latency(:PureBLAS; max_load_ms = 1, precompile = false)
    @test_throws "first `PureBLAS.nrm2(randn(8))` took" test_latency(
        :PureBLAS; calls = ["PureBLAS.nrm2(randn(8))"], max_first_call_ms = 0.001, precompile = false
    )
    @test StrictMode.latency_report(:PureBLAS; precompile = false) isa StrictMode.LatencyReport
end

@testitem "StrictMode package: kernel macros on the unrolled axpy kernel (@kernel, @test_kernel, @test_strict, @assert_noboxing, @explain)" tags = [:checks] begin
    using StrictMode, StrictModeTest, InteractiveUtils
    P = PureBLAS
    K = P._axpy_unrolled!
    AT = (Val{4}, Int, Float64, Ptr{Float64}, Ptr{Float64}, Int)
    n = 256; x = randn(n); y = randn(n)

    # The inspected method is the vector body, not a forwarder: hundreds of IR statements and
    # 8-wide double vector operations with fused multiply-adds in the optimized LLVM.
    @test length(Base.code_typed(K, AT)[1].first.code) > 50
    llvm = sprint(io -> code_llvm(io, K, AT; optimize = true))
    @test occursin("<8 x double>", llvm) && occursin(r"fmuladd|fma", llvm)

    @eval module ToyKern
    scal(x::Vector{Float64}) = (s = 0.0; for i in 1:length(x); s += x[i]; end; s)
    unst(x::Float64) = Ref{Any}(x)[]
    alloc(n::Int) = sum(zeros(n))
    boxy(t::Tuple, i::Int) = t[i]
    end
    v = randn(100)

    # Controls: every macro fails or warns on a toy that breaks its guarantee.
    @test_throws "vector ops" StrictMode.@kernel ToyKern.scal(v)
    @test_throws "return type is not concrete" StrictMode.@kernel ToyKern.unst(1.0)
    @test_throws "typestable" StrictModeTest.@test_kernel ToyKern.unst(1.0)
    @test_throws "noalloc" StrictModeTest.@test_strict ToyKern.alloc(10)
    @test_throws "typestable" StrictModeTest.@test_strict ToyKern.unst(1.0)
    @test_logs (:warn,) match_mode = :any StrictMode.@assert_noboxing ToyKern.boxy((1, 2.0, "x"), 2)
    @test_logs StrictMode.@assert_noboxing ToyKern.alloc(10)          # a typed buffer is not boxing
    rep = StrictMode.@explain ToyKern.unst(1.0)
    @test !rep.return_concrete && rep.return_type === Any

    # The real kernel passes all of them, with the full argument list.
    GC.@preserve x y begin
        px = pointer(x); py = pointer(y)
        StrictMode.@kernel K(Val(4), n, 0.5, px, py, 0)
        StrictModeTest.@test_kernel K(Val(4), n, 0.5, px, py, 0)
        StrictModeTest.@test_strict K(Val(4), n, 0.5, px, py, 0)
        @test_logs StrictMode.@assert_noboxing K(Val(4), n, 0.5, px, py, 0)
        rk = StrictMode.@explain K(Val(4), n, 0.5, px, py, 0)
        @test rk.return_concrete && rk.return_type === Ptr{Float64}
    end
end

@testitem "StrictMode package: @strict_stable and @golden controls, PureBLAS goldens" tags = [:checks] begin
    using StrictMode
    P = PureBLAS

    # @strict_stable: the stable specialization compiles and runs; the heterogeneous one throws when
    # it is compiled. PureBLAS defines no function with it, so this is a control only.
    @eval module ToyStable
    using StrictMode
    @strict_stable pick(t::Tuple, i::Int) = t[i]
    end
    @test ToyStable.pick((1, 2, 3), 2) == 2
    @test_throws "typestable" ToyStable.pick((1, 2.0, "x"), 2)

    # @golden: records when absent, compares after, and rejects a changed result.
    x = [sin(i) for i in 1:64]; y = [cos(2i) for i in 1:64]
    tmp = mktempdir()
    @test isempty(readdir(tmp))
    r = StrictMode.@golden "dot" P.dot(x, y) ulps = 16 dir = tmp
    @test readdir(tmp) == ["dot.golden"]
    @test StrictMode.@golden("dot", P.dot(x, y), ulps = 16, dir = tmp) == r
    @test_throws "ULP mismatch" StrictMode.@golden("dot", P.dot(x, 1.0001 .* y), ulps = 16, dir = tmp)

    # The committed goldens pin the dot and gemm results to 16 ULPs. Both files must exist: an absent
    # golden would be recorded instead of compared and the check would pass vacuously.
    gd = joinpath(pkgdir(PureBLAS), "test", "golden")
    @test isfile(joinpath(gd, "dot_f64_n64.golden")) && isfile(joinpath(gd, "gemm_f64_16.golden"))
    @test StrictMode.@golden("dot_f64_n64", P.dot(x, y), ulps = 16, dir = gd) isa Float64
    A = [sin(i + 3j) for i in 1:16, j in 1:16]; B = [cos(i - 2j) for i in 1:16, j in 1:16]
    @test StrictMode.@golden("gemm_f64_16", (C = zeros(16, 16); P.gemm!(C, A, B); C), ulps = 16, dir = gd) isa Matrix
    @test_throws "ULP mismatch" StrictMode.@golden("gemm_f64_16", (C = zeros(16, 16); P.gemm!(C, A, 1.001 .* B); C), ulps = 16, dir = gd)
end

@testitem "StrictMode package: migration_report, explain_trim, juliac patches and the allocation barrier switch" tags = [:checks] begin
    using StrictMode, StrictModeTest, TrimCheck
    P = PureBLAS

    # migration_report: control files with a deleted API call and a non-gating `@assert_noalloc`.
    d = mktempdir(); mkpath(joinpath(d, "test"))
    write(joinpath(d, "test", "t.jl"), "using StrictMode\nStrictMode.@assert_noalloc f()\n")
    @test StrictMode.migration_report(d; io = devnull) == 1
    write(joinpath(d, "test", "t.jl"), "using StrictModeTest\nStrictModeTest.set_ignore_barrier!(true)\n")
    @test StrictMode.migration_report(d; io = devnull) == 0
    # PureBLAS: the six hits are bare `set_ignore_barrier!` calls in files that `using StrictModeTest`,
    # which exports it, so they resolve and are not deleted API.
    sites = split(sprint(io -> StrictMode.migration_report(pkgdir(PureBLAS); io)), '\n')
    hits = filter(l -> occursin("strictmode_tests.jl:", l), sites)
    @test length(hits) == 6
    @test all(l -> occursin("set_ignore_barrier!", l), hits)
    @test :set_ignore_barrier! in names(StrictModeTest)

    # explain_trim translates verifier output into a diagnosis.
    @test StrictMode.explain_trim("") isa StrictMode.TypeContracts.TrimDiagnostics.TrimFailure

    # The allocation-barrier switch defaults on and is restored by the tests that flip it.
    @test StrictModeTest.ignore_barrier() === true

    # Patched trim verification looks for the patch files under share/julia/juliac. A Julia that ships
    # them elsewhere (1.13 puts them under share/julia/test/trimming) degrades the request to stock
    # Base with a warning, and the verdict is then the stock one: the interpolated-throw rejection of
    # `sysv!`. Where the directory exists the warning must be absent.
    n = 24
    Ad = let M = randn(n, n); M + transpose(M) end
    B = randn(n, 2)
    @test !StrictModeTest.juliac_patches()
    patches_present = isdir(joinpath(Sys.BINDIR, "..", "share", "julia", "juliac"))
    StrictModeTest.set_juliac_patches!(true)
    try
        @test StrictModeTest.juliac_patches()
        msg = ""
        logs, _ = Test.collect_test_logs() do
            try
                @test_trim_compatible P.sysv!('L', copy(Ad), copy(B))
            catch err
                msg = sprint(showerror, err)
            end
        end
        warned = any(l -> l.level == Base.CoreLogging.Warn && occursin("could not run the patched", l.message), logs)
        @test warned == !patches_present
        patches_present || @test occursin("_str_sizehint", msg) || occursin("Base.print", msg)
    finally
        StrictModeTest.set_juliac_patches!(false)
    end
    @test !StrictModeTest.juliac_patches()
end
