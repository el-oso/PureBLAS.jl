# A GATE CELL MUST NOT ALLOCATE IN ITS TIMED CORE.
#
# WHY THIS EXISTS. Eight LP rows (potri, trtri, getri, sytri, gelsy, gelsd, geev, gels) called
# `…!(copy(c))` inside the timed closure, defended in a comment as "both arms copy, so the copy is
# common-mode". It is common-mode, and it still wrecked the measurement: the setup already hands the
# core `reps` FRESH contexts (one op per context, `evals=1`), so the copy was pure waste — 592 B per
# call at n=8 against a 140 ns kernel, 512 calls per sample. That allocation pulled GC INTO the timed
# window, and GC is what produced the wide bands in the published plots.
#
# Measured on wintermute 2026-09-11, trtri@8 under the gate's own harness shape:
#     core COPIES    median 100.5 us   band 11.1%   GC in 2.1% of samples   GC-hit/GC-free 7.38x
#     core NO-alloc  median  80.1 us   band  9.2%   GC in 0.0% of samples
# and at n=32, GC 12.0% -> 0.0% and band 64.6% -> 26.1%. The kernels were never at fault: `trtri!`
# measures 0 B at n=8 and n=256, and `src/verify.jl`'s entry-point contracts already refresh their
# operand allocation-free with `copyto!` into a preallocated buffer.
#
# ⚠ A RESIDUAL BAND SURVIVES (26.1% at n=32 with zero GC and zero allocation). That is a SEPARATE,
# still-unidentified effect — do not read a pass here as "the bands are gone".
#
# This script re-runs each cell's PB closure against a context built exactly as the sweep builds it,
# and fails if the closure allocates. Run it after touching any cell definition in bench/plots.jl.
using PureBLAS, LinearAlgebra, Random, Printf

const LP = 'L'
const TN = 'N'
Random.seed!(20260911)

_hpd(T, n) = (A = randn(T, n, n); A * A' + n * Matrix{T}(LinearAlgebra.I, n, n))
_symm_hpd(n) = _hpd(Float64, n)

# (name, context builder, PB closure) — mirrors bench/plots.jl's LP rows one-for-one.
const CASES = Any[
    ("potri", s -> (A = _hpd(Float64, s); LinearAlgebra.LAPACK.potrf!(LP, A); A),
        c -> (PureBLAS.potri!(c; uplo = LP); c[1, 1])),
    ("trtri", s -> tril(randn(Float64, s, s) + s * LinearAlgebra.I),
        c -> (PureBLAS.trtri!(c; uplo = LP, diag = TN); c[1, 1])),
    ("getri", function (s)
            F = Matrix{Float64}(randn(s, s) + s * LinearAlgebra.I)
            Fa, ip, _ = LinearAlgebra.LAPACK.getrf!(F)
            return (Fa, ip)
        end,
        c -> (PureBLAS.getri!(c[1], c[2]); c[1][1, 1])),
    ("sytri", function (s)
            A = _symm_hpd(s); ip = Vector{Int}(undef, s)
            PureBLAS.sytrf!(A, ip; uplo = LP)
            return (A, ip, randn(s))
        end,
        c -> (PureBLAS.sytri!(c[1], c[2]; uplo = LP); c[1][1, 1])),
    ("gelsy", s -> (randn(Float64, s, s), randn(Float64, s, 1), zeros(Int, s)),
        c -> (PureBLAS.gelsy!(c[1], c[2], c[3], -1.0); c[2][1])),
    ("gelsd", s -> (randn(Float64, s, s), randn(Float64, s, 1)),
        c -> (PureBLAS.gelsd!(c[1], c[2], -1.0); c[2][1])),
    ("geev", s -> randn(Float64, s, s),
        c -> (PureBLAS.geev!(TN, TN, c); c[1, 1])),
    ("gels", s -> (randn(s, s), randn(s, 1)),
        c -> (PureBLAS.gels!(TN, c[1], c[2]); c[2][1])),
]

const N = 32          # small enough that any per-call allocation is unmissable, big enough to be real

# ALLOWED, WITH A REASON — not a blanket exemption. These two call the CONVENIENCE entry, which must
# allocate the arrays it returns (geev: wr/wi; gelsd: the singular values / solution workspace). That
# is O(n) output, semantically required, and it is NOT the redundant per-call scratch that the copies
# were: measured at n=32 and n=256, GC fires in 0.0% of samples for both and the bands stay modest
# (geev 11.9%/5.6%, gelsd 6.3%/1.3%). The workspace-passing entries ARE 0-alloc and that is what the
# strict contracts cover (`_strict_geevr_probe`/`_strict_geevc_probe` in src/verify.jl, which pass
# every output in; verify.jl:923 records gelsd's Float64 arm as 0 B).
# Switching the PB arm to the workspace form while the reference arm keeps allocating its outputs
# would break arm symmetry, so the rows stay on the convenience entry deliberately.
# The BUDGET is what makes this a check and not a rubber stamp: exceed it and the run fails.
const ALLOWED = Dict("geev" => 2_000, "gelsd" => 1_000)

bad = String[]
println("timed-core allocation check (n=$N), PB arm:")
for (name, mk, f) in CASES
    f(mk(N))                                   # warm: compile this exact specialization
    a = 0
    for _ in 1:3                               # steady state; take the max of three fresh contexts
        c = mk(N)
        a = max(a, @allocated f(c))
    end
    budget = get(ALLOWED, name, 0)
    ok = a <= budget
    note = a == 0 ? "ok" : ok ? "ok (allowed output alloc, budget $budget B)" : "*** ALLOCATES ***"
    @printf("  %-7s %8d B   %s\n", name, a, note)
    ok || push!(bad, name)
end
if isempty(bad)
    println("\nall LP timed cores are allocation-free")
else
    println("\nALLOCATING CELLS: ", join(bad, ", "))
    println("A gate cell that allocates measures the allocator and drags GC into the timed window.")
    println("Move the allocation into the context builder (`mk`), as every other row already does.")
    exit(1)
end
