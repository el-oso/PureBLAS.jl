# Kernel-quality checks on the SIMD kernels: vector-register spills, vectorization, scalar loops,
# kernel_report, register_report and llvm-mca. Every check runs on the FULL argument types of the
# body that executes (never a defaulted-argument forwarder), after a test that the inspected code is
# the real kernel: not a forwarder, not a MethodError stub, with the expected vector FMAs. Every check
# has a positive control, a toy that the check must flag.
#
# Per-kernel verdicts depend on the vector width and register-file size, so each assertion is made for
# the box's own width (`W` = doubles per vector, `NVREG` = vector registers). A verdict that is known to
# differ on a width is recorded with `@test_broken`, which turns into an error when the kernel is fixed.
# An assertion that cannot run on a box is a visible `@test_skip` carrying its reason.

@testmodule KernelChecks begin
using PureBLAS, StrictMode, InteractiveUtils, Test
const P = PureBLAS

const W = P._vwidth(Float64)
const NVREG = P._NVREG
box() = "$(Sys.CPU_NAME) (W=$W, $NVREG vector registers, _MR=$(P._MR), _NR=$(P._NR))"
const StrictViolation = StrictMode.StrictViolation

# (name, f, argtypes): every entry names the FULL argument tuple of the method that holds the body.
# Kernel table: (name, f, argtypes). Every entry passes the FULL argtypes (no defaulted trailing Val).
function kernel_table(P)
    Pd = Ptr{Float64}; Ps = Ptr{Float32}
    MR = P._MR; NR = P._NR
    V(x) = Val{x}
    K = Tuple{String, Any, Any}[]
    add!(n, f, T) = push!(K, (n, f, T))
    for B0 in (false, true)
        add!("_microkernel! 8-arg B0=$B0", P._microkernel!, Tuple{Pd, Int, Pd, Pd, Int, V(MR), V(NR), V(B0)})
        add!("_microkernel! alpha B0=$B0", P._microkernel!, Tuple{Pd, Int, Pd, Pd, Int, Float64, V(MR), V(NR), V(B0)})
        add!("_microkernel_clip! B0=$B0", P._microkernel_clip!, Tuple{Pd, Int, Pd, Pd, Int, V(MR), V(1), V(NR), V(B0)})
        add!("_microkernel_clip! alpha B0=$B0", P._microkernel_clip!, Tuple{Pd, Int, Pd, Pd, Int, Float64, V(MR), V(1), V(NR), V(B0)})
        add!("_microkernel_masked! B0=$B0", P._microkernel_masked!, Tuple{Pd, Int, Pd, Pd, Int, Int, Int, V(MR), V(NR), V(B0)})
        add!("_microkernel_masked! alpha B0=$B0", P._microkernel_masked!, Tuple{Pd, Int, Pd, Pd, Int, Float64, Int, Int, V(MR), V(NR), V(B0)})
        add!("_microkernel_db! B0=$B0", P._microkernel_db!, Tuple{Pd, Int, Pd, Pd, Int, Int, V(MR), V(MR), V(NR), V(B0)})
        add!("_microkernel_db_masked! B0=$B0", P._microkernel_db_masked!, Tuple{Pd, Int, Pd, Pd, Int, Int, Int, Int, V(MR), V(NR), V(B0)})
        for TB in (false, true)
            add!("_microkernel_unpacked! TB=$TB B0=$B0", P._microkernel_unpacked!, Tuple{Pd, Int, Pd, Int, Int, Pd, Int, Int, Int, Float64, Float64, V(MR), V(NR), V(TB), V(B0)})
        end
    end
    add!("_microkernel_unpacked! F32 TB=false B0=false", P._microkernel_unpacked!, Tuple{Ps, Int, Ps, Int, Int, Ps, Int, Int, Int, Float32, Float32, V(MR), V(NR), V(false), V(false)})
    add!("_microkernel! F32 8-arg", P._microkernel!, Tuple{Ps, Int, Ps, Ps, Int, V(MR), V(NR), V(false)})
    add!("_microkernel_unpacked_mrows!", P._microkernel_unpacked_mrows!, Tuple{Pd, Int, Pd, Int, Int, Pd, Int, Int, Int, Float64, Float64, Int, V(MR), V(NR), V(false), V(false), V(P._vwidth(Float64))})
    add!("_microkernel_unpacked_split!", P._microkernel_unpacked_split!, Tuple{Pd, Int, Pd, Int, Int, Pd, Int, Int, Int, Float64, Float64, Int, V(MR), V(NR), V(2), V(false), V(false), V(false)})
    # complex
    for (nm, full, tri) in (("full", true, false), ("masked", false, false), ("tri", false, true))
        add!("_uker_cmplx! $nm", P._uker_cmplx!, Tuple{Pd, Int, Pd, Int, Int, Pd, Int, Int, Int, Float64, Float64, Int, Int, V((P._CMR, NR, false, false, false, false, false, false, full, tri)), Int, Bool})
    end
    add!("_uker_cmplx! MR=1", P._uker_cmplx!, Tuple{Pd, Int, Pd, Int, Int, Pd, Int, Int, Int, Float64, Float64, Int, Int, V((1, NR, false, false, false, false, false, false, false, false)), Int, Bool})
    add!("_microkernel_cmplx!", P._microkernel_cmplx!, Tuple{Pd, Int, Pd, Pd, Pd, Pd, Int, Float64, Float64, V(P._CMR), V(NR), V(false), V(false), V(false), V(false), V(false)})
    add!("_microkernel_cmplx_masked!", P._microkernel_cmplx_masked!, Tuple{Pd, Int, Pd, Pd, Pd, Pd, Int, Float64, Float64, Int, Int, V(P._CMR), V(NR), V(false), V(false), V(false), V(false), V(false)})
    add!("_microkernel_cmplx_tri!", P._microkernel_cmplx_tri!, Tuple{Pd, Int, Pd, Pd, Pd, Pd, Int, Float64, Float64, Int, Int, Int, Bool, V(P._CMR), V(NR), V(false), V(false), V(false), V(false), V(false)})
    add!("_microkernel2_cmplx!", P._microkernel2_cmplx!, Tuple{Pd, Int, Pd, Pd, Pd, Pd, Pd, Pd, Pd, Pd, Int, V(P._CMR), V(NR), V(false), V(false), V(false)})
    # trsm / trmm tiles
    add!("_microkernel_tri!", P._microkernel_tri!, Tuple{Pd, Int, Pd, Pd, Int, Float64, Int, Int, Int, Bool, V(MR), V(NR), V(false)})
    add!("_microkernel2!", P._microkernel2!, Tuple{Pd, Int, Pd, Pd, Pd, Pd, Int, Float64, Int, Int, Int, Bool, V(MR), V(NR), V(:full), V(false)})
    add!("_microkernel_u!", P._microkernel_u!, Tuple{Pd, Int, Pd, Pd, Int, Float64, Int, Int, Int, Bool, V(MR), V(NR), V(:full), V(false)})
    add!("_trsm_tile_L_f64!", P._trsm_tile_L_f64!, Tuple{Bool, Bool, Pd, Int, Pd, Int, Int, Int})
    add!("_trsm_tile_R_f64!", P._trsm_tile_R_f64!, Tuple{Bool, Bool, Bool, Pd, Int, Pd, Int, Int, Int})
    add!("_gemmtrsm_u_slab!", P._gemmtrsm_u_slab!, Tuple{Pd, Int, Pd, Int, Pd, Int, Int, V(P._GT_MR), V(P._GT_NRV)})
    add!("_zrt_tile!", P._zrt_tile!, Tuple{V(4), V(false), V(false), Pd, Int, Pd, Int, Pd, Int, Int, Int})
    # L1
    add!("_axpy_unrolled! U=4", P._axpy_unrolled!, Tuple{V(4), Int, Float64, Pd, Pd, Int})
    add!("_axpy_phase!", P._axpy_phase!, Tuple{V(4), V(P._vwidth(Float64)), Int, Float64, Pd, Pd})
    add!("_axpy_simd!", P._axpy_simd!, Tuple{Int, Float64, Pd, Pd, Int, Int})
    add!("_scal_simd!", P._scal_simd!, Tuple{Int, Float64, Pd, Int})
    add!("_dot_simd", P._dot_simd, Tuple{Int, Pd, Pd, Type{Float64}})
    add!("_dot_blocked", P._dot_blocked, Tuple{Int, Pd, Pd, Type{Float64}})
    add!("_dot_blocked_serial", P._dot_blocked_serial, Tuple{Int, Pd, Pd, Type{Float64}})
    add!("_asum_simd", P._asum_simd, Tuple{Int, Pd, Type{Float64}})
    add!("_sumsq_simd (nrm2)", P._sumsq_simd, Tuple{Int, Pd, Type{Float64}})
    add!("_iamax_tree!", P._iamax_tree!, Tuple{V(4), Int, Pd, Float64})
    add!("_iamax_thresh!", P._iamax_thresh!, Tuple{V(4), Int, Pd, Float64})
    add!("_iamax_simd!", P._iamax_simd!, Tuple{Int, Pd})
    Vc = Vector{ComplexF64}; Pc = Ptr{ComplexF64}
    add!("_dot_pair_simd cplx", P._dot_pair_simd, Tuple{V(:cplx), Int, Vc, Vc, Type{Float64}, V(false)})
    add!("_scal_pair_simd! cplx", P._scal_pair_simd!, Tuple{V(:cplx), Int, Float64, Float64, Vc})
    # L2
    for B0 in (false, true)
        add!("_gemv_n_block! B0=$B0", P._gemv_n_block!, Tuple{Pd, Pd, Int, Pd, Int, Float64, Float64, V(4), V(B0)})
    end
    add!("_gemv_n_block_masked!", P._gemv_n_block_masked!, Tuple{Pd, Pd, Int, Pd, Int, Float64, Float64, Int, V(2), V(false)})
    add!("_gemv_n_panel!", P._gemv_n_panel!, Tuple{Pd, Pd, Int, Pd, Int, Int, Int, Float64, V(2)})
    add!("_gemv_n_panel_full!", P._gemv_n_panel_full!, Tuple{Pd, Pd, Int, Pd, Int, Int, Float64, V(4)})
    add!("_gemv_n_panel_minner!", P._gemv_n_panel_minner!, Tuple{Pd, Pd, Int, Pd, Int, Int, Float64, V(4), V(2)})
    for B0 in (false, true)
        add!("_gemv_t_block! NC=4 B0=$B0", P._gemv_t_block!, Tuple{Pd, NTuple{4, Pd}, Pd, Int, Float64, Float64, V(4), V(B0), V(2), V(0)})
    end
    add!("_ger_panel!", P._ger_panel!, Tuple{Pd, Int, Pd, Pd, Int, Int, Float64, Int, V(4), V(2)})
    add!("_symv_lowerpanel!", P._symv_lowerpanel!, Tuple{Int, Float64, Pd, Int, Pd, Pd, V(2), V(4)})
    add!("_symv_upperpanel!", P._symv_upperpanel!, Tuple{Int, Float64, Pd, Int, Pd, Pd, V(2), V(4)})
    add!("_gemv_n_block_cmplx!", P._gemv_n_block_cmplx!, Tuple{Pd, Pd, Int, Pd, Int, Int, Float64, Float64, V(2)})
    add!("_gemv_n_pair_panel! cplx", P._gemv_n_pair_panel!, Tuple{V(:cplx), Pd, Pd, Int, Pd, Int, Int, Float64, Float64, V(2), V(false)})
    add!("_gemv_tc_pair_block! cplx", P._gemv_tc_pair_block!, Tuple{V(:cplx), Pc, Pc, Int, Pc, Int, ComplexF64, ComplexF64, Bool, V(2), V(false), V(false)})
    add!("_ger_pair_panel! cplx", P._ger_pair_panel!, Tuple{V(:cplx), Pc, Int, Pc, Pc, Int, Int, ComplexF64, V(2), V(false), V(2), V(false)})
    # pack
    Vd = Vector{Float64}; Md = Matrix{Float64}
    add!("_pack_A_simd!", P._pack_A_simd!, Tuple{Vd, Md, Int, Int, Int, Int, Float64, Int})
    add!("_pack_A_simd_T!", P._pack_A_simd_T!, Tuple{Vd, Md, Int, Int, Int, Int, Float64, Int})
    add!("_pack_B_val!", P._pack_B_val!, Tuple{Vd, Md, Int, Int, Int, Int, Bool, V(NR), Int})
    add!("_pack_B_dyn!", P._pack_B_dyn!, Tuple{Vd, Md, Int, Int, Int, Int, Bool, Int, Int})
    add!("_pack_A_tri_simd!", P._pack_A_tri_simd!, Tuple{Vd, Md, Int, Int, Int, Int, Bool, Bool, Float64, Int})
    add!("_pack_A_cmplx_simd!", P._pack_A_cmplx_simd!, Tuple{Vd, Vd, Matrix{ComplexF64}, Int, Int, Int, Int, Int})
    add!("_pack_B_cmplx!", P._pack_B_cmplx!, Tuple{Vd, Vd, Matrix{ComplexF64}, Int, Int, Int, Int, Bool, Int})
    return K
end

const KERNELS = kernel_table(P)

# Kernels that spill a full vector register (32-byte ymm, 64-byte zmm) in a loop or anywhere, per vector
# width. Measured on Zen3 (W=4, 16 ymm), Zen4 and Zen5 (W=8, 32 zmm; the two agree).
const KNOWN_WIDE_LOOP_SPILL = Set(
    [
        (4, "_microkernel_unpacked_mrows!"), (4, "_microkernel_unpacked_split!"),
        (8, "_microkernel_unpacked_split!"), (8, "_uker_cmplx! full"), (8, "_uker_cmplx! masked"),
        (8, "_uker_cmplx! tri"), (8, "_microkernel_cmplx!"), (8, "_microkernel_cmplx_masked!"),
        (8, "_microkernel_cmplx_tri!"), (8, "_microkernel2_cmplx!"),
    ]
)
const KNOWN_WIDE_SPILL = union(
    KNOWN_WIDE_LOOP_SPILL,
    Set(
        [
            (4, "_microkernel! alpha B0=false"), (4, "_microkernel! alpha B0=true"), (4, "_uker_cmplx! masked"),
            (4, "_uker_cmplx! tri"), (4, "_uker_cmplx! MR=1"), (4, "_microkernel_tri!"), (4, "_microkernel_u!"),
            (4, "_gemmtrsm_u_slab!"), (8, "_gemmtrsm_u_slab!"),
        ]
    )
)

native_lines(f, T) = split(sprint(io -> code_native(io, f, T; debuginfo = :none)), '\n')

# Spans (first line, last line) of every loop, found as a label targeted by a later backward jump.
function loopspans(ls)
    lab = Dict{String, Int}()
    for (i, l) in enumerate(ls)
        m = match(r"^\s*([A-Za-z_.$][A-Za-z0-9_.$]*):", l)
        m === nothing || (lab[m[1]] = i)
    end
    sp = Tuple{Int, Int}[]
    for (i, l) in enumerate(ls)
        m = match(r"^\s*j[a-z]*\s+([A-Za-z_.$][A-Za-z0-9_.$]*)\b", l)
        m === nothing && continue
        j = get(lab, m[1], 0)
        0 < j <= i && push!(sp, (j, i))
    end
    return sp
end

const SPILL_RE = r"#.*\b(?:Spill|Reload)\b"
const VREG_RE = r"\b(?:xmm|ymm|zmm)\d+\b"
# Size of the spilled value in bytes, from LLVM's "N-byte Spill/Reload" comment: 32 = ymm, 64 = zmm.
spill_bytes(l) = (m = match(r"# (\d+)-byte", l); m === nothing ? 0 : parse(Int, m[1]))

# Native-code facts for one kernel. `wide` counts spill/reload lines of a full ymm/zmm value (32 or 64 bytes);
# `spills` is `spill_report(f, T).vec_spills` (it also counts xmm lines, which carry spilled scalars).
function analyze(f, T)
    ls = native_lines(f, T)
    sidx = [i for (i, l) in enumerate(ls) if occursin(SPILL_RE, l) && occursin(VREG_RE, l)]
    widx = [i for i in sidx if spill_bytes(ls[i]) >= 32]
    sp = loopspans(ls)
    kr = StrictMode.kernel_report(f, T)
    rr = StrictMode.register_report(f, T)
    return (;
        lines = length(ls),
        vfma = count(l -> occursin(r"vfn?m(?:add|sub)\d+p[sd]", l), ls),
        stub = any(l -> occursin("throw_methoderror", l), ls),
        calls = count(l -> occursin(r"\bcall\b", l), ls),
        spills = StrictMode.spill_report(f, T).vec_spills,
        counted = length(sidx),
        wide = length(widx),
        wide_in_loop = count(i -> any(((a, b),) -> a <= i <= b, sp), widx),
        vectorized = kr.vectorized, fp_ops = kr.fp_ops, mem_ops = kr.mem_ops, intensity = kr.intensity,
        scalar = StrictMode.scalar_fp_loops(f, T),
        zmm_used = rr.vec_regs_used, zmm_spills = rr.vec_spills,
    )
end

const CACHE = Dict{String, Any}()
info(name) = get!(CACHE, name) do
    i = findfirst(k -> k[1] == name, KERNELS)
    i === nothing && error("no kernel named $name")
    analyze(KERNELS[i][2], Tuple(KERNELS[i][3].parameters))
end
forwarder_info(f, T) = analyze(f, T)

# Kernels with no FMA by construction: scale, abs-sum, max-search and data movement.
const FMA_FREE = Set(
    [
        "_scal_simd!", "_asum_simd", "_iamax_tree!", "_iamax_thresh!", "_iamax_simd!",
        "_pack_A_simd!", "_pack_A_simd_T!", "_pack_B_val!", "_pack_B_dyn!", "_pack_A_tri_simd!",
        "_pack_A_cmplx_simd!", "_pack_B_cmplx!",
    ]
)

# --- positive controls ----------------------------------------------------------------------------
# N vector accumulators that all stay live across the loop: more than the register file holds, so the
# allocator must spill them inside the loop. The body is also spliced into a concrete method by the
# `@strict_policy :no_spill` control in test/strictmode_tests.jl.
function toy_spill_body(N)
    V = P.Vec{W, Float64}
    accs = [Symbol(:a, i) for i in 1:N]
    init = [:($(a) = zero($V)) for a in accs]
    upd = [:($(accs[i]) = muladd($(P.vload)($V, p + (k + $(i - 1)) * $(8W)), $(accs[i]), $(accs[i]) + $(P.vload)($V, p + (k + $(N + i - 1)) * $(8W)))) for i in 1:N]
    st = [:($(P.vstore)($(accs[i]), p + $((i - 1) * 8W))) for i in 1:N]
    return quote
        $(init...)
        k = 0
        while k < n
            $(upd...)
            k += 1
        end
        $(st...)
        return nothing
    end
end
# Defined after its generator: a `@generated` body runs in the world where the method was defined.
@generated toy_spill!(p::Ptr{Float64}, n::Int, ::Val{N}) where {N} = toy_spill_body(N)

# A loop-carried recurrence: no vector form exists, so this is a scalar FP loop and nothing vectorizes.
function toy_recurrence!(p::Ptr{Float64}, n::Int, a::Float64)
    for i in 1:(n - 1)
        unsafe_store!(p, muladd(a, unsafe_load(p, i), unsafe_load(p, i + 1)), i + 1)
    end
    return nothing
end

# A pointer chase: each load's address is the previous load's value, so no vector form exists on any
# ISA. The `:vectorized` control.
function toy_chase(p::Ptr{Float64}, q::Ptr{Int}, n::Int)
    s = 0.0
    j = 1
    for _ in 1:n
        s += unsafe_load(p, j)
        j = unsafe_load(q, j)
    end
    return s
end

# A strict-order reduction: no reassociation is allowed, so its adds stay scalar. The scalar-loop
# control. Not a `:vectorized` control: on aarch64 its loads compile to 128-bit vector loads.
function toy_strict_sum(p::Ptr{Float64}, n::Int)
    s = 0.0
    for i in 1:n
        s += unsafe_load(p, i)
    end
    return s
end

# A correct SIMD dot product with the usual scalar remainder after the vector loops.
function toy_dot_tail(n::Int, x::Ptr{Float64}, y::Ptr{Float64})
    V = P.Vec{W, Float64}
    acc = zero(V)
    i = 0
    while i + W <= n
        acc = muladd(P.vload(V, x + 8i), P.vload(V, y + 8i), acc)
        i += W
    end
    s = sum(acc)
    while i < n
        s += unsafe_load(x, i + 1) * unsafe_load(y, i + 1)
        i += 1
    end
    return s
end

# A serial dependency chain of vector FMAs: its throughput is bound by FMA latency, not by registers.
function toy_chain!(p::Ptr{Float64}, n::Int)
    V = P.Vec{W, Float64}
    acc = P.vload(V, p)
    b = P.vload(V, p + 8W)
    for _ in 1:n
        acc = muladd(acc, b, acc)
    end
    P.vstore(acc, p)
    return nothing
end

# A defaulted-argument forwarder whose body is an out-of-line call: its own code is a call and a return.
toy_fwd!(p::Ptr{Float64}, n::Int, ::Val{B} = Val(false)) where {B} = (@noinline toy_chain!(p, n); nothing)
end

@testitem "kernel: inspected bodies are real" tags = [:checks] setup = [KernelChecks] begin
    using StrictMode, Test
    using .KernelChecks: P, W, NVREG, KERNELS, info, FMA_FREE, toy_spill!, toy_recurrence!, toy_strict_sum,
        toy_dot_tail, toy_chain!, forwarder_info, StrictViolation, box, analyze, KNOWN_WIDE_LOOP_SPILL,
        KNOWN_WIDE_SPILL, toy_fwd!
    @info "kernel checks run on $(box())"
    # Control: a defaulted-argument forwarder is a thin trampoline. Its own code holds a call and no
    # vector arithmetic, so a check pointed at it reads nothing and the real-body test must reject it.
    Pd = Ptr{Float64}
    fw = forwarder_info(toy_fwd!, (Pd, Int))
    @test fw.lines < 40 && fw.vfma == 0 && fw.calls >= 1 && !fw.vectorized
    # Control: a body that is a MethodError stub (here `_scal_pair_simd!` handed a raw Ptr instead of the
    # array it expects) has a plausible line count and no vector work.
    stub = forwarder_info(P._scal_pair_simd!, (Val{:cplx}, Int, Float64, Float64, Pd))
    @test stub.stub && stub.vfma == 0 && !stub.vectorized
    for (name, f, T) in KERNELS
        r = info(name)
        @testset "$name" begin
            @test !r.stub
            @test r.lines >= 60
            @test r.vectorized
            startswith(name, "_pack_") || @test r.fp_ops > 0 || r.mem_ops > 0
            if !any(s -> startswith(name, s), FMA_FREE)
                @test r.vfma >= 1
            end
        end
    end
end

@testitem "kernel: :no_spill / spill_report" tags = [:checks] setup = [KernelChecks] begin
    using StrictMode, StrictModeTest, Test
    using .KernelChecks: P, W, NVREG, KERNELS, info, FMA_FREE, toy_spill!, toy_recurrence!, toy_strict_sum,
        toy_dot_tail, toy_chain!, forwarder_info, StrictViolation, box, analyze, KNOWN_WIDE_LOOP_SPILL,
        KNOWN_WIDE_SPILL
    N = 3 * NVREG
    Pd = Ptr{Float64}
    # Control: a toy with 3x the register file of live accumulators spills inside its loop, and the
    # `:no_spill` gate fails on it.
    toy = StrictMode.spill_report(toy_spill!, (Pd, Int, Val{N}))
    @test toy.vec_spills > 0
    @test_throws StrictViolation test_signatures([(toy_spill!, (Pd, Int, Val{N}))]; guarantees = (:no_spill,))
    # Real kernels, at the signatures the packed gemm path compiles.
    MR = Val{P._MR}; NR = Val{P._NR}
    test_signatures(
        [
            (P._microkernel!, (Pd, Int, Pd, Pd, Int, MR, NR, Val{false})),
            (P._microkernel!, (Pd, Int, Pd, Pd, Int, MR, NR, Val{true})),
            (P._microkernel_masked!, (Pd, Int, Pd, Pd, Int, Int, Int, MR, NR, Val{false})),
            (P._microkernel_clip!, (Pd, Int, Pd, Pd, Int, MR, Val{1}, NR, Val{false})),
        ];
        guarantees = (:no_spill,)
    )
    # The signatures src/verify.jl declares with `@strict_function guarantees = …` (the microkernels'
    # `:vectorized`/`:no_spill`, the trsm side-R tile's `:no_spill`), re-proved here with each one's
    # own guarantees.
    # The per-task first-touch accessors are allocation barriers for the proofs, as in
    # strictmode_tests.jl; the registration is process-global, so it is removed again afterwards.
    pools = (P._m3ws, P._gpackws, P._strws, P._opt)
    foreach(StrictMode.register_alloc_barrier!, pools)
    fs = try
        test_registered(; modules = [P])
    finally
        foreach(f -> delete!(StrictMode._ALLOC_BARRIERS, f), pools)   # internal: no public unregister exists
        StrictMode.clear_cache!()
    end
    @test count(f -> f.status === :pass, fs) >= 9
    # Every kernel in the table. A wide (ymm/zmm) spill anywhere, or inside a loop, is the real defect.
    # `spill_report` also counts xmm lines: on a kernel that takes scalar alpha/beta arguments those
    # carry the spilled scalars, which is not register pressure on the vector tile.
    for (name, f, T) in KERNELS
        r = info(name)
        @testset "$name" begin
            if (W, name) in KNOWN_WIDE_LOOP_SPILL
                @test_broken r.wide_in_loop == 0
            else
                @test r.wide_in_loop == 0
            end
            if (W, name) in KNOWN_WIDE_SPILL
                @test_broken r.wide == 0
            else
                @test r.wide == 0
            end
            if r.wide == 0 && r.spills > 0
                @test_broken r.spills == 0    # xmm-only spill counted by spill_report
            elseif r.wide == 0
                @test r.spills == 0
            end
        end
    end
end

@testitem "kernel: :vectorized / kernel_report" tags = [:checks] setup = [KernelChecks] begin
    using StrictMode, StrictModeTest, Test
    using .KernelChecks: P, W, NVREG, KERNELS, info, FMA_FREE, toy_spill!, toy_recurrence!, toy_strict_sum,
        toy_dot_tail, toy_chain!, forwarder_info, StrictViolation, box, analyze, KNOWN_WIDE_LOOP_SPILL,
        KNOWN_WIDE_SPILL, toy_fwd!, toy_chase
    Pd = Ptr{Float64}
    # Controls: a pointer chase and a recurrence emit no vector code, and the `:vectorized` gate fails
    # on each.
    @test !StrictMode.kernel_report(toy_chase, (Pd, Ptr{Int}, Int)).vectorized
    @test !StrictMode.kernel_report(toy_recurrence!, (Pd, Int, Float64)).vectorized
    @test_throws StrictViolation test_signatures([(toy_chase, (Pd, Ptr{Int}, Int))]; guarantees = (:vectorized,))
    # The strict-order sum is a `:vectorized` control on x86 only. StrictMode gap: on aarch64 its scalar
    # FP adds sit behind 128-bit vector loads, and `:vectorized` passes them.
    sum_throws = try
        test_signatures([(toy_strict_sum, (Pd, Int))]; guarantees = (:vectorized,))
        false
    catch e
        e isa StrictViolation || rethrow()
        true
    end
    if Sys.ARCH === :aarch64
        @test_broken sum_throws
    else
        @test sum_throws
    end
    @test_throws StrictViolation test_signatures([(toy_recurrence!, (Pd, Int, Float64))]; guarantees = (:vectorized,))
    # The forwarder is judged unvectorized; a check that follows the forwarder to its body would pass.
    fwd_ok = try
        test_signatures([(toy_fwd!, (Pd, Int))]; guarantees = (:vectorized,))
        true
    catch e
        e isa StrictViolation || rethrow()
        false
    end
    @test_broken fwd_ok
    # `Type{Float64}`, not `DataType`: the latter names a different method instance of `_dot_simd`.
    test_signatures(
        [
            (P._microkernel!, (Pd, Int, Pd, Pd, Int, Val{P._MR}, Val{P._NR}, Val{false})),
            (P._dot_simd, (Int, Pd, Pd, Type{Float64})),
        ];
        guarantees = (:vectorized,)
    )
    for (name, f, T) in KERNELS
        r = info(name)
        @test r.vectorized
    end
    # kernel_report on the packed microkernel: register-blocked, so FP work per vector memory op is high.
    mk = info("_microkernel! 8-arg B0=false")
    @test mk.intensity >= 1.0
    # Control: a streaming kernel does about one FMA per loaded vector, a lower ratio than the tile.
    ax = info("_axpy_unrolled! U=4")
    @test ax.intensity < mk.intensity
end

@testitem "kernel: :no_scalar_loops / scalar_fp_loops" tags = [:checks] setup = [KernelChecks] begin
    using StrictMode, StrictModeTest, Test
    using .KernelChecks: P, W, NVREG, KERNELS, info, FMA_FREE, toy_spill!, toy_recurrence!, toy_strict_sum,
        toy_dot_tail, toy_chain!, forwarder_info, StrictViolation, box, analyze, KNOWN_WIDE_LOOP_SPILL,
        KNOWN_WIDE_SPILL
    Pd = Ptr{Float64}
    # Controls: scalar loops are flagged, and `test_signatures` fails on them under `:no_scalar_loops`
    # (the same `scalar_fp_loops` scan; this driver gates on it).
    @test StrictMode.scalar_fp_loops(toy_strict_sum, (Pd, Int))
    @test StrictMode.scalar_fp_loops(toy_recurrence!, (Pd, Int, Float64))
    @test_throws StrictViolation test_signatures([(toy_recurrence!, (Pd, Int, Float64))]; guarantees = (:no_scalar_loops,))
    @test_throws StrictViolation test_signatures([(toy_strict_sum, (Pd, Int))]; guarantees = (:no_scalar_loops,))
    # A vectorized SIMD dot product whose only scalar loop is the remainder (fewer than W iterations) is
    # flagged too: `scalar_fp_loops` cannot tell a bounded remainder from a hot loop once the function
    # holds a hand-vectorized loop.
    @test_broken !StrictMode.scalar_fp_loops(toy_dot_tail, (Int, Pd, Pd))
    for (name, f, T) in KERNELS
        r = info(name)
        @testset "$name" begin
            @test !r.scalar
        end
    end
end

@testitem "kernel: register_report" tags = [:checks] setup = [KernelChecks] begin
    using StrictMode, Test
    using .KernelChecks: P, W, NVREG, KERNELS, info, FMA_FREE, toy_spill!, toy_recurrence!, toy_strict_sum,
        toy_dot_tail, toy_chain!, forwarder_info, StrictViolation, box, analyze, KNOWN_WIDE_LOOP_SPILL,
        KNOWN_WIDE_SPILL
    N = 3 * NVREG
    if W >= 8
        toy = StrictMode.register_report(toy_spill!, (Ptr{Float64}, Int, Val{N}))
        @test toy.vec_regs_used == 32 && toy.vec_spills > 0     # control: saturated and spilling
        for (name, f, T) in KERNELS
            r = info(name)
            @test 0 < r.zmm_used <= 32
        end
        # The packed real microkernel keeps its tile in registers with room to spare.
        mk = info("_microkernel! 8-arg B0=false")
        @test mk.zmm_used < 32 && mk.zmm_spills == 0
    else
        @test_skip "register_report counts zmm registers only; this box has $(NVREG) ymm registers (W=$W), so it reports 0 for every kernel" == ""
        @test StrictMode.register_report(toy_spill!, (Ptr{Float64}, Int, Val{N})).vec_regs_used == 0
    end
end

@testitem "kernel: llvm-mca" tags = [:checks] setup = [KernelChecks] begin
    using StrictMode, Test
    using .KernelChecks: P, W, NVREG, KERNELS, info, FMA_FREE, toy_spill!, toy_recurrence!, toy_strict_sum,
        toy_dot_tail, toy_chain!, forwarder_info, StrictViolation, box, analyze, KNOWN_WIDE_LOOP_SPILL,
        KNOWN_WIDE_SPILL
    if Base.find_package("LLVM_full_jll") === nothing
        @test_skip "llvm-mca needs LLVM_full_jll, which is not in this test environment" == ""
    else
        @eval using LLVM_full_jll
        @test StrictMode.mca_available()
        Pd = Ptr{Float64}
        T8 = (Pd, Int, Pd, Pd, Int, Val{P._MR}, Val{P._NR}, Val{false})
        r = StrictMode.mca_report(P._microkernel!, T8)
        @test !r.whole_function                       # the hot loop was found
        @test r.ipc > 0 && r.block_rthroughput > 0
        # Control: the kernel's block throughput is above an unmeetable bound, and a serial FMA chain
        # is latency-bound.
        @test r.block_rthroughput > 0.01
        chain = StrictMode.mca_report(toy_chain!, (Pd, Int))
        @test chain.ipc < r.ipc
    end
end
