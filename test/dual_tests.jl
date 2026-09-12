# Dual-number BLAS-1 (docs/src/dual.md). A dense `Dual{Tag,V,1}` vector is byte-identical to a
# `Complex{V}` one, so it rides the real SIMD kernels where no element×element product occurs and its own
# `dupEven` kernels for the reductions, through the ForwardDiff extension. Every item here is a correctness
# boundary the design names — the ε²-leak item in particular is what catches a reinterpret-as-complex
# mistake (complex computes `ac − bd`, dual must compute `ac`).

@testmodule DualT begin
    using ForwardDiff
    using ForwardDiff: Dual, value, partials
    mkd(v, p) = Dual{Nothing}.(v, p)
    vals(x) = value.(x)
    pars(x) = partials.(x, 1)
    # netlib idamax over the VALUES, verbatim: init |x[1]|, sequential strict `>` (NaN never wins,
    # ties keep the first). This is the contract for a Dual vector — partials never enter.
    function ref_iamax(x)
        n = length(x)
        n < 1 && return 0
        dmax = abs(value(x[1])); ix = 1
        for k in 2:n
            if abs(value(x[k])) > dmax
                dmax = abs(value(x[k])); ix = k
            end
        end
        return ix
    end
end

@testitem "Dual: layout tripwire + extension routing predicate" setup = [DualT] begin
    using PureBLAS, ForwardDiff
    using ForwardDiff: Dual
    @test !isnothing(Base.get_extension(PureBLAS, :PureBLASForwardDiffExt))
    x = [Dual{Nothing}(1.0, 2.0), Dual{Nothing}(3.0, 4.0)]
    @test reinterpret(Float64, x) == [1.0, 2.0, 3.0, 4.0]                  # [v, p, v, p, …]
    @test sizeof(Dual{Nothing, Float64, 1}) == 16 == sizeof(ComplexF64)
    @test sizeof(Dual{Nothing, Float32, 1}) == 8 == sizeof(ComplexF32)
    @test PureBLAS._pairalg(x)
    @test PureBLAS._pairalg(DualT.mkd(randn(Float32, 4), randn(Float32, 4)))
    @test !PureBLAS._pairalg(randn(2)) && !PureBLAS._pairalg(randn(ComplexF64, 2))
    @test !PureBLAS._pairalg([Dual{Nothing}(1.0, 2.0, 3.0)])             # N = 2: not a pair
    @test !PureBLAS._pairalg(view(x, 1:2))                                # not dense: scalar path
    nested = Dual{Nothing}.(x, x)                                         # V is itself a Dual
    @test !PureBLAS._pairalg(nested)
    @test PureBLAS._pairreal(x) == Ptr{Float64}(pointer(x))
    @test PureBLAS._parts(x[1]) == (1.0, 2.0)
    @test PureBLAS._mkpair(eltype(x), 5.0, 6.0) === Dual{Nothing}(5.0, 6.0)
    @test PureBLAS._l1v(Dual{Nothing}(-3.0, 100.0)) === 3.0
end

@testitem "Dual: pair routing matches the generic loop (all ops, both real types, tails)" setup = [DualT] begin
    using PureBLAS, ForwardDiff, LinearAlgebra
    using ForwardDiff: Dual, value, partials
    for V in (Float64, Float32), n in (0, 1, 3, 16, 31, 64, 257, 1000, 1003)
        tol = 8 * sqrt(eps(V)) * max(n, 1)
        ≃(a, b) = isapprox(value(a), value(b); rtol = tol, atol = tol) &&
            isapprox(partials(a, 1), partials(b, 1); rtol = tol, atol = tol)
        x = DualT.mkd(randn(V, n), randn(V, n)); y = DualT.mkd(randn(V, n), randn(V, n))
        a = V(1.7)
        @test all(PureBLAS.axpy!(copy(y), a, x) .≃ (y .+ a .* x))
        @test all(PureBLAS.scal!(a, copy(x)) .≃ (a .* x))
        ad = Dual{Nothing}(V(1.7), V(0.3))                                # dual alpha: generic loop
        @test all(PureBLAS.axpy!(copy(y), ad, x) .≃ (y .+ ad .* x))
        @test all(PureBLAS.scal!(ad, copy(x)) .≃ (ad .* x))
        @test PureBLAS.blascopy!(similar(x), x) == x
        xs = copy(x); ys = copy(y); PureBLAS.swap!(xs, ys)
        @test xs == y && ys == x
        # n = 0: the oracle's sqrt(Dual(0,0)) has a NaN partial (d√ at 0); BLAS returns 0, and so do we.
        @test n == 0 ? iszero(PureBLAS.nrm2(x)) : PureBLAS.nrm2(x) ≃ sqrt(sum(abs2, x))
        @test PureBLAS.asum(x) ≃ sum(abs, x; init = zero(eltype(x)))
        @test PureBLAS.iamax(x) == DualT.ref_iamax(x)
        @test PureBLAS.dot(x, y) ≃ sum(x .* y; init = zero(eltype(x)))
        @test PureBLAS.dotu(x, y) ≃ sum(x .* y; init = zero(eltype(x)))
        if n > 0
            @test PureBLAS.nrm2(x) isa eltype(x) && PureBLAS.asum(x) isa eltype(x)
        end
    end
end

@testitem "Dual: directional derivatives via ForwardDiff.derivative for every op" begin
    using PureBLAS, ForwardDiff, LinearAlgebra
    n = 128
    x = randn(n); v = randn(n); y = randn(n); w = randn(n)
    D(f) = ForwardDiff.derivative(f, 0.0)
    @test D(t -> PureBLAS.nrm2(x .+ t .* v)) ≈ dot(x, v) / norm(x)
    @test D(t -> PureBLAS.asum(x .+ t .* v)) ≈ dot(sign.(x), v)
    @test D(t -> PureBLAS.dot(x .+ t .* v, y .+ t .* w)) ≈ dot(v, y) + dot(x, w)
    @test D(t -> PureBLAS.dotu(x .+ t .* v, y .+ t .* w)) ≈ dot(v, y) + dot(x, w)
    @test D(t -> sum(PureBLAS.axpy!(y .+ t .* w, 1.7, x .+ t .* v))) ≈ sum(w) + 1.7 * sum(v)
    @test D(t -> sum(PureBLAS.axpy!(y .+ t .* w, 1.7 + 0.3t, x .+ t .* v))) ≈ sum(w) + 1.7 * sum(v) + 0.3 * sum(x)
    @test D(t -> sum(PureBLAS.scal!(1.7, x .+ t .* v))) ≈ 1.7 * sum(v)
    @test D(t -> sum(PureBLAS.scal!(1.7 + 0.3t, x .+ t .* v))) ≈ 1.7 * sum(v) + 0.3 * sum(x)
    @test D(t -> (xt = x .+ t .* v; sum(PureBLAS.blascopy!(similar(xt), xt)))) ≈ sum(v)
    @test D(t -> (xt = x .+ t .* v; yt = y .+ t .* w; PureBLAS.swap!(xt, yt); sum(xt) - 2sum(yt))) ≈ sum(w) - 2sum(v)
    @test D(t -> (xt = x .+ t .* v; xt[PureBLAS.iamax(xt)])) ≈ v[argmax(abs.(x))]
    # gradients use chunked N > 1 duals: the generic path, same answers
    @test ForwardDiff.gradient(PureBLAS.nrm2, x) ≈ x ./ norm(x)
    @test ForwardDiff.gradient(PureBLAS.asum, x) ≈ sign.(x)
    @test ForwardDiff.gradient(z -> PureBLAS.dot(z, y), x) ≈ y
end

@testitem "Dual: ε²-leak — Σx_p² overflows, value stays finite, partial exact" setup = [DualT] begin
    using PureBLAS, ForwardDiff, LinearAlgebra
    using ForwardDiff: value, partials
    n = 1000
    xv = randn(n); xp = 1.0e155 .* randn(n)
    @test !isfinite(sum(abs2, xp))                                        # the ε² term would overflow
    x = DualT.mkd(xv, xp)
    for f in (PureBLAS.dot, PureBLAS.dotu)
        d = f(x, x)
        @test isfinite(value(d)) && value(d) ≈ sum(abs2, xv)
        @test partials(d, 1) ≈ 2 * dot(xv, xp)
    end
    r = PureBLAS.nrm2(x)                                                  # the SIMD fast path: Σx_v·x_p finite
    @test isfinite(value(r)) && value(r) ≈ norm(xv)
    @test partials(r, 1) ≈ dot(xv, xp) / norm(xv)
    s = PureBLAS.asum(x)
    @test isfinite(value(s)) && value(s) ≈ sum(abs, xv)
    @test partials(s, 1) ≈ dot(sign.(xv), xp)
end

@testitem "Dual: NaN hygiene — infinite partials never poison value lanes" setup = [DualT] begin
    using PureBLAS, ForwardDiff, LinearAlgebra
    using ForwardDiff: value, partials
    n = 1000
    xv = randn(n); xp = randn(n); xp[[1, 7, 500, n]] .= Inf; xp[300] = -Inf
    yv = randn(n); yp = randn(n)
    x = DualT.mkd(xv, xp); y = DualT.mkd(yv, yp)
    r = PureBLAS.axpy!(copy(y), 1.7, x)
    @test all(isfinite, value.(r)) && value.(r) ≈ yv .+ 1.7 .* xv
    @test isinf(partials(r[7], 1))                                        # the partial lane carries the Inf
    # DUAL alpha: the tagged SIMD body — the zeroed product must sit on the VALUE lane (duplicate-even), or
    # `0·Inf` from an infinite partial would poison the value, which the generic loop keeps finite.
    ad = ForwardDiff.Dual{Nothing}(1.7, 0.3)
    r = PureBLAS.axpy!(copy(y), ad, x)
    @test all(isfinite, value.(r)) && value.(r) ≈ yv .+ 1.7 .* xv
    @test isinf(partials(r[7], 1)) && partials(r[2], 1) ≈ yp[2] + 1.7 * xp[2] + 0.3 * xv[2]
    r = PureBLAS.scal!(1.7, copy(x))
    @test all(isfinite, value.(r)) && value.(r) ≈ 1.7 .* xv
    r = PureBLAS.scal!(ad, copy(x))                                      # dual alpha: the tagged SIMD body
    @test all(isfinite, value.(r)) && value.(r) ≈ 1.7 .* xv
    @test isinf(partials(r[7], 1)) && partials(r[2], 1) ≈ 1.7 * xp[2] + 0.3 * xv[2]
    r = PureBLAS.nrm2(x); @test isfinite(value(r)) && value(r) ≈ norm(xv)
    r = PureBLAS.asum(x); @test isfinite(value(r)) && value(r) ≈ sum(abs, xv)
    r = PureBLAS.dot(x, y); @test isfinite(value(r)) && value(r) ≈ dot(xv, yv)
    @test PureBLAS.iamax(x) == argmax(abs.(xv))
end

@testitem "Dual: dotc == dotu (no conjugation on a Real)" setup = [DualT] begin
    using PureBLAS, ForwardDiff
    for n in (1, 16, 1003)
        x = DualT.mkd(randn(n), randn(n)); y = DualT.mkd(randn(n), randn(n))
        @test PureBLAS.dot(x, y) === PureBLAS.dotu(x, y)
        @test PureBLAS.dot(x, x) === PureBLAS.dotu(x, x)
    end
end

@testitem "Dual: iamax is argmax of |value|, first occurrence; netlib NaN/Inf/tie oracle over Dual" setup = [DualT] begin
    using PureBLAS, ForwardDiff
    using ForwardDiff: Dual
    W = PureBLAS._vwidth(Float64)
    # the design's own example: equal magnitude, opposite partials — the partial must NOT decide
    @test PureBLAS.iamax([Dual{Nothing}(3.0, -5.0), Dual{Nothing}(3.0, 1.0)]) == 1
    @test PureBLAS.iamax([Dual{Nothing}(3.0, 1.0), Dual{Nothing}(3.0, -5.0)]) == 1
    for n in (31, 4W, 4W + 3, 200, 1003)                                 # scalar path (n < 4W) and every SIMD tier
        x = DualT.mkd(ones(n), randn(n))
        x[10] = Dual{Nothing}(3.0, -5.0); x[20] = Dual{Nothing}(-3.0, 100.0)   # |value| tie, partials differ
        @test PureBLAS.iamax(x) == 10 == DualT.ref_iamax(x)
        x[30] = Dual{Nothing}(1.0, 1.0e6)                                 # a huge PARTIAL is not a magnitude
        @test PureBLAS.iamax(x) == 10
        x[30] = Dual{Nothing}(1.0, Inf)
        @test PureBLAS.iamax(x) == 10
    end
    # netlib oracle: NaN/Inf in the VALUE lane, every position of a vector spanning the kernel tiers
    for n in (3, 4W + 1, 100), pos in unique((1, 2, n ÷ 2, n)), v in (NaN, Inf, -Inf)
        x = DualT.mkd(randn(n), randn(n)); x[pos] = Dual{Nothing}(v, 1.0)
        @test PureBLAS.iamax(x) == DualT.ref_iamax(x)
    end
    x = DualT.mkd(fill(NaN, 40), randn(40)); @test PureBLAS.iamax(x) == 1
    x = DualT.mkd(randn(40), fill(NaN, 40)); @test PureBLAS.iamax(x) == DualT.ref_iamax(x)   # NaN partials ignored
end

@testitem "Dual: nrm2 lassq fallback at 1e200 scale matches ForwardDiff on LinearAlgebra.norm" setup = [DualT] begin
    using PureBLAS, ForwardDiff, LinearAlgebra
    using ForwardDiff: value, partials
    for (V, big) in ((Float64, 1.0e200), (Float64, 1.0e-200), (Float32, 1.0f30), (Float32, 1.0f-30))
        n = 300
        xv = V(big) .* randn(V, n); xp = randn(V, n)
        x = DualT.mkd(xv, xp)
        r = PureBLAS.nrm2(x)
        @test isfinite(value(r)) && !iszero(value(r))
        @test value(r) ≈ norm(xv)
        # ORACLE = the analytic partial Σ x_v x_p / ‖x‖, computed in V without ever dividing a Dual.
        # `ForwardDiff.derivative(t -> norm(xv .+ t .* xp), 0)` is NOT usable here: at 1e200 both
        # LinearAlgebra's scaled `norm` over Duals and PureBLAS's old generic lassq loop go through
        # ForwardDiff's quotient rule, which squares a ~1e200 denominator; measured on this input class the
        # three disagreed (−1.47 / −1.66 / analytic −1.39). That is why `_nrm2_dual_scaled` exists.
        @test partials(r, 1) ≈ dot(xv, xp) / norm(xv) rtol = 100 * sqrt(eps(V))
    end
end

@testitem "Dual: N = 2 and nested duals take the generic path and still match" setup = [DualT] begin
    using PureBLAS, ForwardDiff, LinearAlgebra
    using ForwardDiff: Dual, value, partials
    n = 257
    xv = randn(n); xp = randn(n); xq = randn(n); yv = randn(n)
    x2 = Dual{Nothing}.(xv, xp, xq); y2 = Dual{Nothing}.(yv, xq, xp)       # N = 2
    @test !PureBLAS._pairalg(x2)
    r = PureBLAS.nrm2(x2)
    @test value(r) ≈ norm(xv) && partials(r, 1) ≈ dot(xv, xp) / norm(xv) && partials(r, 2) ≈ dot(xv, xq) / norm(xv)
    r = PureBLAS.asum(x2)
    @test value(r) ≈ sum(abs, xv) && partials(r, 1) ≈ dot(sign.(xv), xp)
    # `≈` on value AND both partials (the generic loop's `muladd` rounds differently from broadcast)
    ≃(a, b) = value(a) ≈ value(b) && partials(a, 1) ≈ partials(b, 1) && partials(a, 2) ≈ partials(b, 2)
    @test PureBLAS.dot(x2, y2) ≃ sum(x2 .* y2)
    @test PureBLAS.iamax(x2) == argmax(abs.(xv))
    @test all(PureBLAS.axpy!(copy(y2), 1.7, x2) .≃ (y2 .+ 1.7 .* x2))
    @test all(PureBLAS.scal!(1.7, copy(x2)) .≃ (1.7 .* x2))
    @test PureBLAS.blascopy!(similar(x2), x2) == x2
    a = copy(x2); b = copy(y2); PureBLAS.swap!(a, b); @test a == y2 && b == x2
    # nested: second directional derivative through ForwardDiff's own nesting, vs LinearAlgebra.norm
    v = randn(n); w = randn(n)
    d2(f) = ForwardDiff.derivative(s -> ForwardDiff.derivative(t -> f(xv .+ t .* v .+ s .* w), 0.0), 0.0)
    @test d2(PureBLAS.nrm2) ≈ d2(norm)
    @test d2(PureBLAS.asum) ≈ d2(z -> sum(abs, z))
    @test d2(z -> PureBLAS.dot(z, z)) ≈ d2(z -> dot(z, z))
end

@testitem "StrictMode dogfood: BLAS-1 dual strict contract" tags = [:checks] begin
    using StrictModeTest, StrictMode, PureBLAS, ForwardDiff
    using ForwardDiff: Dual
    if !StrictMode.checks_enabled()
        @info "StrictMode checks disabled — skipping dual dogfood"
        @test_skip StrictMode.checks_enabled()
    else
        # The main env has no ForwardDiff, so src/verify.jl cannot prove these; they live here. Every entry
        # over a dense Dual vector must stay allocation-free and type-stable on the pair fast paths AND on
        # the generic fallbacks (dual alpha, the lassq loop).
        bk = PureBLAS.DEFAULT_BACKEND
        n = 1000
        xd = Dual{Nothing}.(randn(n), randn(n)); yd = Dual{Nothing}.(randn(n), randn(n))
        ad = Dual{Nothing}(1.7, 0.3)
        @test_noalloc PureBLAS.axpy!(bk, yd, 2.0, xd)
        @test_noalloc PureBLAS.axpy!(bk, yd, ad, xd)
        @test_noalloc PureBLAS.scal!(bk, 2.0, xd)
        @test_noalloc PureBLAS.scal!(bk, ad, xd)
        @test_noalloc PureBLAS.blascopy!(bk, yd, xd)
        @test_noalloc PureBLAS.swap!(bk, xd, yd)
        @test_noalloc PureBLAS.dot(bk, xd, yd)
        @test_noalloc PureBLAS.dotu(bk, xd, yd)
        @test_noalloc PureBLAS.nrm2(bk, xd)
        @test_noalloc PureBLAS.asum(bk, xd)
        @test_noalloc PureBLAS.iamax(bk, xd)
        @test_typestable PureBLAS.axpy!(bk, yd, 2.0, xd)
        @test_typestable PureBLAS.scal!(bk, ad, xd)
        @test_typestable PureBLAS.nrm2(bk, xd)
        @test_typestable PureBLAS.asum(bk, xd)
        @test_typestable PureBLAS.iamax(bk, xd)
        @test_typestable PureBLAS.dot(bk, xd, yd)
        # The `@test_*` proofs record nothing on success and THROW a StrictViolation on failure (they are
        # deliberately not gated), so the item's own count is 0 without this. Non-vacuity was checked with a
        # positive control (`@test_noalloc` on a `push!` loop throws in this env).
        @test true
    end
end

@testmodule DualNative begin
    # Opcode histogram of the MAIN SIMD loop of a kernel (kb: loop-body-opcode-histogram): the epilogue of a
    # tagged body is per-algebra by design (dot drops the ε² lane, the scalar tails differ), so what must
    # match between the complex and dual instantiations is the unrolled vector loop — the backward-jump span
    # holding the most FMAs. Shuffle-class mnemonics are folded into one bucket: swap-adjacent lowers to
    # `vpermilpd`/`vshufpd`, duplicate-even to `vmovddup`/`vunpcklpd` — same port class, different immediate.
    using InteractiveUtils
    const SHUF = r"^v?(permil|shuf|movddup|movsldup|movshdup|unpck|pshuf|pshufd)"
    function native_labelled(f, types)
        io = IOBuffer(); code_native(io, f, types; debuginfo = :none, syntax = :intel)
        lines = String[]
        for l in eachline(IOBuffer(String(take!(io))))
            s = strip(replace(l, r"[#;].*$" => ""))
            isempty(s) && continue
            if endswith(s, ":")
                push!(lines, "LABEL " * chop(s))
            elseif !startswith(s, ".")
                push!(lines, replace(s, r"\s+" => " "))
            end
        end
        return lines
    end
    isfma(x) = startswith(x, "vfmadd") || startswith(x, "vfmsub")
    function mainloop(lines)
        best = String[]; bestfma = -1
        for (ji, l) in enumerate(lines)
            m = match(r"^j[a-z]+ (\S+)$", l); isnothing(m) && continue
            li = findfirst(==("LABEL " * m.captures[1]), lines)
            (isnothing(li) || li > ji) && continue
            span = filter(x -> !startswith(x, "LABEL"), lines[li:ji])
            nf = count(isfma, span)
            (nf > bestfma || (nf == bestfma && length(span) > length(best))) && (best = span; bestfma = nf)
        end
        return best
    end
    function hist(lines)
        h = Dict{String, Int}()
        for l in lines
            op = first(split(l)); occursin(SHUF, op) && (op = "SHUF")
            h[op] = get(h, op, 0) + 1
        end
        return h
    end
    loop(f, types) = mainloop(native_labelled(f, types))
end

@testitem "Dual: tagged bodies — complex and dual instantiations share the SIMD loop (opcode histogram)" setup = [DualT, DualNative] begin
    using PureBLAS, ForwardDiff
    using ForwardDiff: Dual
    C = Val{:cplx}; D = Val{:dual}
    for T in (Float64, Float32)
        CV = Vector{Complex{T}}; DV = Vector{Dual{Nothing, T, 1}}
        L = Val{PureBLAS._zaxpy_narrow_lanes(T)}; U = typeof(PureBLAS._ZAXPY_PHASE_UV)
        specs = [
            "scal" => (PureBLAS._scal_pair_simd!, (Int, T, T, CV), (Int, T, T, DV)),
            "axpy_phase" => (PureBLAS._axpy_pair_phase!, (L, U, Int, T, T, CV, CV), (L, U, Int, T, T, DV, DV)),
            "axpy_wide" => (PureBLAS._axpy_pair_wide!, (Int, T, T, CV, CV), (Int, T, T, DV, DV)),
            "dotu" => (PureBLAS._dot_pair_simd, (Int, CV, CV, Type{T}, Val{false}), (Int, DV, DV, Type{T}, Val{false})),
            "dotc" => (PureBLAS._dot_pair_simd, (Int, CV, CV, Type{T}, Val{true}), (Int, DV, DV, Type{T}, Val{true})),
        ]
        for (nm, (f, tc, td)) in specs
            lc = DualNative.loop(f, (C, tc...)); ld = DualNative.loop(f, (D, td...))
            @test count(DualNative.isfma, lc) >= 2                       # positive control: a real unrolled FMA loop was found
            @test length(lc) == length(ld)
            @test DualNative.hist(lc) == DualNative.hist(ld)
        end
        # negative control: the histogram is not blind — dotu's loop is not dotc's whole-function code
        @test DualNative.hist(DualNative.loop(PureBLAS._dot_pair_simd, (C, Int, CV, CV, Type{T}, Val{false}))) !=
            DualNative.hist(DualNative.native_labelled(PureBLAS._dot_pair_simd, (C, Int, CV, CV, Type{T}, Val{false})))
    end
end
