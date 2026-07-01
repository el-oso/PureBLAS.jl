# LAPACK gate + REGRESSION guardrail — the absolute gate (our/OpenBLAS ≥ 0.96) for potrf/geqrf/getrf/
# gesvd, with correctness (vs the LAPACK reference) and a per-host ratio baseline (flags a ratio DROP
# > REGR_TOL even when still above the gate). Complements the PkgBenchmark suite (benchmark/benchmarks.jl,
# `judge` = SELF-regression vs my own history); this is the vs-OpenBLAS gate `judge` can't express.
#
# Methodology = the gate: interleaved paired timing (cancels freq drift), MEDIAN, set_num_threads(1),
# adaptive rounds (IQR/median < tol). Destructive ops ⇒ untimed `reset` refills the work buffer.
#
# Usage:  taskset -c 2 julia --project=bench bench/lapackbench.jl [save] [routines...] [--sizes n1,n2]
#   routines ∈ {potrf geqrf getrf gesvd} (default all). exit 1 on regression or correctness fail.

using PureBLAS, LinearAlgebra, Statistics, Printf, Random
BLAS.set_num_threads(1)
const S = Ref(0.0); @noinline _run(f) = f()
const REGR_TOL = 0.03; const GATE = 0.96
_sink(x::Tuple) = @inbounds (S[] += Float64(x[1] isa AbstractArray ? length(x[1]) : 1); 0.0)
_sink(x::AbstractArray) = (S[] += Float64(length(x)); 0.0)
_sink(x) = 0.0

function _stable(ref, our, reset; rounds = 11, tol = 0.02, cap = 41)
    reset(); _run(ref); reset(); _run(our)
    rs = Float64[]
    while true
        for _ in 1:rounds
            reset(); t0 = time_ns(); a = _run(ref); t1 = time_ns()
            reset(); t2 = time_ns(); b = _run(our); t3 = time_ns()
            _sink(a); _sink(b); push!(rs, (t1 - t0) / (t3 - t2))
        end
        m = median(rs)
        ((quantile(rs, 0.75) - quantile(rs, 0.25)) / m < tol || length(rs) >= cap) && return m
    end
end

# Each routine: n → (ref, our, reset) sharing a work buffer, plus a correctness (relerr) probe.
function _case(name, n)
    Random.seed!(1234 + n)
    if name == "potrf"
        M = randn(n, n); A0 = M'M + n * I; Aw = Matrix{Float64}(undef, n, n); reset = () -> copyto!(Aw, A0)
        (() -> LAPACK.potrf!('L', Aw), () -> PureBLAS.potrf!(Aw; uplo = 'L'), reset)
    elseif name == "geqrf"
        A0 = randn(n, n); Aw = similar(A0); reset = () -> copyto!(Aw, A0)
        (() -> LAPACK.geqrf!(Aw), () -> PureBLAS.geqrf!(Aw), reset)
    elseif name == "getrf"
        A0 = randn(n, n); Aw = similar(A0); reset = () -> copyto!(Aw, A0)
        (() -> LAPACK.getrf!(Aw), () -> PureBLAS.getrf!(Aw), reset)
    elseif name == "gesvd"
        A0 = randn(n, n); Aw = similar(A0); reset = () -> copyto!(Aw, A0)
        (() -> LAPACK.gesdd!('A', Aw), () -> PureBLAS.gesvd!(Aw; want_vectors = true), reset)
    else
        error("unknown routine $name")
    end
end
function _relerr(name, n)
    Random.seed!(77 + n); A0 = randn(n, n)
    if name == "potrf"
        M = randn(n, n); SPD = M'M + n * I; L = PureBLAS.potrf!(copy(SPD); uplo = 'L')
        return maximum(abs, tril(L) * tril(L)' - SPD) / maximum(abs, SPD)
    elseif name == "geqrf"
        F = copy(A0); tau = zeros(n); PureBLAS.geqrf!(F, tau); Fl = copy(A0); LAPACK.geqrf!(Fl)
        return maximum(abs, triu(F) - triu(Fl)) / max(maximum(abs, Fl), eps())
    elseif name == "getrf"
        F = copy(A0); ip = zeros(Int, n); PureBLAS.getrf!(F, ip); Fl = copy(A0); _, ipl, _ = LAPACK.getrf!(Fl)
        return max(maximum(abs, F - Fl) / max(maximum(abs, Fl), eps()), ip == ipl ? 0.0 : 1.0)
    else # gesvd
        U, s, Vt = PureBLAS.gesvd!(copy(A0))
        return max(maximum(abs, U * Diagonal(s) * Vt - A0), maximum(abs, s - svdvals(A0))) / maximum(svdvals(A0))
    end
end

function _parseargs(args)
    routines = String[]; sizes = Int[]; save = false; i = 1
    while i <= length(args)
        a = args[i]
        if a == "save"; save = true
        elseif a == "--sizes"; sizes = parse.(Int, split(args[i+1], ",")); i += 1
        else push!(routines, a); end
        i += 1
    end
    isempty(routines) && (routines = ["potrf", "geqrf", "getrf", "gesvd"])
    isempty(sizes) && (sizes = [256, 512, 1024, 2048])
    return routines, sizes, save
end
routines, sizes, save = _parseargs(ARGS)

_host() = strip(read(`hostname`, String))
_bpath() = joinpath(@__DIR__, "lapack_baseline_$(_host()).txt")
base = Dict{String, Float64}()
isfile(_bpath()) && for ln in eachline(_bpath())
    (isempty(ln) || startswith(ln, '#')) && continue
    k, v = split(ln, '='); base[strip(k)] = parse(Float64, v)
end

results = Dict{String, Float64}(); NF = Ref(0)
@printf("%-7s %-6s %-9s %-8s %s\n", "routine", "n", "relerr", "our/OB", "flags")
for name in routines, n in sizes
    er = _relerr(name, n)
    er > 1e-9 && (NF[] += 1)
    ref, our, reset = _case(name, n)
    r = _stable(ref, our, reset); key = "$(name)_$n"; results[key] = r
    b = get(base, key, NaN)
    flags = r >= GATE ? "GATE" : ""
    (!isnan(b) && r < b * (1 - REGR_TOL)) && (NF[] += 1; flags *= @sprintf(" REGRESSION vs %.3f", b))
    er > 1e-9 && (flags *= @sprintf(" ✗CORRECTNESS(%.1e)", er))
    @printf("%-7s %-6d %.2e  %.3f    %s\n", name, n, er, r, flags)
end
if save
    open(_bpath(), "w") do io
        println(io, "# PureBLAS LAPACK gate baseline — $(_host()) — our/OpenBLAS interleaved-median ratios")
        for k in sort(collect(keys(results))); println(io, "$k = $(round(results[k], digits = 4))"); end
    end
    println("\nbaseline written: $(_bpath())")
end
S[] == -1.0 && println(S[])
exit(NF[] > 0 ? 1 : 0)
