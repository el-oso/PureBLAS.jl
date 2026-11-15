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
