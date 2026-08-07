# SIMD.jl fast paths for the bandwidth-bound Level-1 kernels, real (`Float32`/`Float64`),
# unit-stride, dense memory. Everything else falls back to the generic scalar loops in
# level1.jl. Pointer-based `vload`/`vstore` over `GC.@preserve`d buffers; masked scalar tail.

using SIMD: Vec, vload, vstore, vifelse, shufflevector

# Vec type for element `T` at the detected register width (folds to a concrete type — see cpuinfo.jl).
@inline _vec(::Type{T}) where {T} = Vec{_vwidth(T), T}

# Eligibility: unit-stride + dense + real. `Ptr` inputs come from the C ABI (already raw); dense
# arrays expose a pointer. Complex deliberately excluded (interleaved re/im SIMD is M2 work).
@inline _simd1(::Ptr{T}) where {T <: BlasReal} = true
@inline _simd1(::DenseArray{T}) where {T <: BlasReal} = true
@inline _simd1(@nospecialize(_)) = false
@inline _simd2(::Ptr{T}, ::Ptr{T}) where {T <: BlasReal} = true
@inline _simd2(::DenseArray{T}, ::DenseArray{T}) where {T <: BlasReal} = true
@inline _simd2(@nospecialize(_), @nospecialize(_)) = false

# Complex unit-stride dense/Ptr → the underlying interleaved [re im re im …] buffer IS a contiguous
# 2n-real array. For the two reductions that are grouping-invariant — nrm2 (Σ|xᵢ|² = Σ over 2n reals r² )
# and asum (dzasum = Σ|Re|+|Im| = Σ over 2n reals |r|) — a complex op reduces EXACTLY to the real SIMD
# kernel over that reinterpreted buffer. `_reptr` gives the real Ptr (caller GC.@preserves the array).
@inline _cplx_re(::Ptr{Complex{T}}) where {T <: BlasReal} = true
@inline _cplx_re(::DenseArray{Complex{T}}) where {T <: BlasReal} = true
@inline _cplx_re(@nospecialize(_)) = false
@inline _reptr(x::Ptr{Complex{T}}) where {T <: BlasReal} = Ptr{T}(x)
@inline _reptr(x::DenseArray{Complex{T}}) where {T <: BlasReal} = Ptr{T}(pointer(x))
const _CplxArg{T} = Union{Ptr{Complex{T}}, DenseArray{Complex{T}}}
@inline _cplx2(::_CplxArg{T}, ::_CplxArg{T}) where {T <: BlasReal} = true      # both complex, same real T
@inline _cplx2(@nospecialize(_), @nospecialize(_)) = false

@inline _ptr(p::Ptr) = p
@inline _ptr(a) = pointer(a)

# Elementwise kernels are 4-way unrolled (4 vectors / iteration) to keep load/store ports busy and
# give the prefetcher a longer stride — a single vector/iteration is throughput-starved in the
# L2-resident regime. `_UNROLL` is defined with the reductions below. Pattern: unrolled body, then
# a W-at-a-time pass, then a scalar tail.
# `pf` = software-prefetch distance (elements ahead) for the OUTPUT stream `y`. Default 0 → the whole
# prefetch block const-folds away, so the L1 `axpy` path (and every other 4-arg caller) is byte-identical.
# `ger` passes `pf>0`: its `y` is a full A column, so at large m the sequential read-modify-write is
# memory-latency-bound on high-latency memory (e.g. LPDDR5x) — one prefetch PER CACHE LINE across the
# unrolled step (the HW prefetcher can't be relied on there) hides it (measured: neuromancer ger n=4096
# 0.88→~1.0). The prefetch may reach up to `pf` elements past the column end; `llvm.prefetch` lowers to a
# non-faulting `prefetcht0`, so that's safe. Distance `pf` is a derived const (see `_GER_PF_BYTES`).
# The unrolled body, U vectors per iteration. `U` is a `Val` so each arm is its own straight-line code.
@generated function _axpy_unrolled!(::Val{U}, n::Int, a::T, x, y, pf::Int) where {U, T <: BlasReal}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T)
    body = [:(vstore(muladd(va, vload($V, px + o + $(j * W * sz)), vload($V, py + o + $(j * W * sz))),
                     py + o + $(j * W * sz))) for j in 0:(U - 1)]
    return quote
        $(Expr(:meta, :inline))
        px = _ptr(x); py = _ptr(y); sz = $sz; step = $(U * W)
        GC.@preserve x y begin
            va = $V(a)
            i = 0
            while i + step <= n
                o = i * sz
                if pf > 0                             # const-folds OFF when pf==0 (default / axpy)
                    pb = py + (i + pf) * sz
                    for c in 0:_CACHELINE:(step * sz - 1)
                        _prefetch(pb + c)
                    end
                end
                $(body...)
                i += step
            end
            while i + $W <= n
                o = i * sz
                vstore(muladd(va, vload($V, px + o), vload($V, py + o)), py + o)
                i += $W
            end
            while i < n
                j = i + 1
                unsafe_store!(py, muladd(a, unsafe_load(px, j), unsafe_load(py, j)), j)
                i += 1
            end
        end
        return y
    end
end

# PHASE-SEPARATED body: all U loads+FMAs issued, THEN all U stores. `L` is the lane count, so `L = W`
# is full width and `L = W÷2` is the narrow (256-bit) variant.
#
# THIS IS THE SHAPE OF THE BINDING KERNEL. AOCL's `bli_daxpyv_zen_int_avx512` (objdump, libblis-mt.so
# +0xac030) runs 8 zmm per iteration as a rolling two-half-block pipeline: x-loads issued ahead, y folded
# into the FMA memory operands, and the STORES BATCHED AT THE BLOCK TAIL. Our shipped body interleaves
# load/FMA/store per vector, which caps how far ahead the load stream can run. Batching the stores raises
# outstanding-load MLP WITHOUT adding write streams — which is exactly the trade Zen5 wants (it is the
# one box whose measured `ger_np` is 1, i.e. fewest write streams) and it explains why every WIDTH change
# there did nothing while this structural one does. Same lesson as the iamax max-tree: on this fleet Zen5
# responds to scheduling structure, not to how many vectors are in flight.
@generated function _axpy_phase!(::Val{U}, ::Val{L}, n::Int, a::T, x, y) where {U, L, T <: BlasReal}
    W = _vwidth(T); V = Vec{W, T}; VL = Vec{L, T}; sz = sizeof(T)
    xs = [Symbol(:xv, j) for j in 0:(U - 1)]
    ys = [Symbol(:yv, j) for j in 0:(U - 1)]
    lds = [:($(xs[j + 1]) = vload($VL, px + o + $(j * L * sz))) for j in 0:(U - 1)]
    fms = [:($(ys[j + 1]) = muladd(va, $(xs[j + 1]), vload($VL, py + o + $(j * L * sz)))) for j in 0:(U - 1)]
    sts = [:(vstore($(ys[j + 1]), py + o + $(j * L * sz))) for j in 0:(U - 1)]
    return quote
        $(Expr(:meta, :inline))
        px = _ptr(x); py = _ptr(y)
        GC.@preserve x y begin
            va = $VL(a); vaw = $V(a)
            i = 0; step = $(U * L)
            while i + step <= n
                o = i * $sz
                $(lds...)                             # all loads + FMAs first
                $(fms...)
                $(sts...)                             # then all stores, batched at the block tail
                i += step
            end
            while i + $W <= n
                o = i * $sz
                vstore(muladd(vaw, vload($V, px + o), vload($V, py + o)), py + o)
                i += $W
            end
            while i < n
                j = i + 1
                unsafe_store!(py, muladd(a, unsafe_load(px, j), unsafe_load(py, j)), j)
                i += 1
            end
        end
        return y
    end
end

# AXPY UNROLL — PDM **Measure** tier, and the fleet is what proves it cannot be Derived.
#
# The optimum number of vectors in flight for a streaming read-modify-write is a property of the memory
# path, and on THIS fleet the analogous knob inverts end to end: `ger_np`, the concurrent write-stream
# count measured per host, resolves to 8 on Zen4, 4 on Zen3 and 1 on Zen5. Same physical question, three
# different answers — the req#8b tell that no formula over the detected consts predicts it.
#
# The fleet also says the shipped literal 4 is wrong somewhere. Per-size gate ratios (2026-08-05):
#     Zen3  1.049 1.013 1.002 1.000 0.995 0.988 1.005   isolated dip at 1e5-3e5
#     Zen4  1.095 1.025 1.007 1.002 1.009 1.007 0.986   fine until 1e6
#     Zen5  1.040 1.057 0.997 0.992 0.991 0.979 0.979   MONOTONE decline from the first size past L1
# Zen5 degrades from n=1e4 onward — its L1d is 48 KiB/core, so 1e4 (160 KB working set) is the first
# size that leaves L1 — and it is also the box that wants ONE write stream. A fixed 4-wide unroll is the
# widest streaming shape we ship, on the box that measurably prefers the narrowest.
#
# CANDIDATE SET is Derived: {2, 4, 8} vectors, bounded by the register file (`_NVREG`), spanning the
# range the fleet demonstrates is needed — `asum` wanted WIDER (4->8 closed dzasum on Zen4), Zen5's axpy
# slope points NARROWER. PROBE REGIME is Derived from where the misses actually start: just past L1,
# which is the first failing size on the worst box (the sytrf lesson — probe where the gate is decided,
# not at a convenient size).
# PRE-BUILT `Val` SINGLETONS for the measure closures. `Val(W)` / `Val(W ÷ 2)` with `W` a LOCAL does not
# const-fold once it is captured by a `_tune_one` closure: trim reported
#   unresolved call ... _axpy_phase!(Val{8}(), %new()::Val{_A} where _A, ...)  (simd_kernels.jl:194)
# i.e. the second Val was still being CONSTRUCTED at run time, which is an unresolved call under
# `juliac --trim` and took 10 C-ABI symbols with it (daxpy/dgesvd/dgbtrf/dgeqp3/dgels/dgehrd/dgecon/
# dorghr/dpocon/dtrcon — everything reaching `_axpy_simd!`). A module-level const IS its concrete type,
# so passing it introduces no construction at all. This is the boring fix; it needs no new dependency
# and no type-level machinery beyond what is already here.
const _AXPY_VW_F64 = Val(_vwidth(Float64))          # full width, F64 probe
const _AXPY_VHW_F64 = Val(_vwidth(Float64) ÷ 2)     # narrow (half) width, F64 probe

const _AXPY_UNROLL_PREF = @load_preference("axpy_unroll", nothing)
@static if isnothing(_AXPY_UNROLL_PREF)
    function _measure_axpy_unroll()::Int
        Base.generating_output() && return _UNROLL          # never burn a measure during precompile
        try
            # PROBE IN THE MIDDLE OF THE BAND IT GOVERNS. This knob covers L1 < bytes <= L3, and probing
# at 4xL1 picked a shape that wins at 128 KB and LOSES at 800 KB / 2.4 MB — the sizes that
# actually fail on Zen4 (1e5, 3e5). 2xL2 sits above L2 and inside L3 on every fleet box.
# ⚠ NOT A POWER OF TWO. `2*L2/8` and `2*L3/8` are both exact po2 (262144 and 4194304 on Zen4), and
# po2 lengths are a cache-set-aliasing pathology — the codebase has `_avoid_po2` for exactly
# this and this probe was not using it. Measured 2026-08-07 in the GATE regime: at n=262144
# the incumbent `u4` takes 2317 us, SLOWER than at n=300000 (1948 us) despite fewer elements,
# and `p208` beats it by 37% there versus 1.6% at the neighbouring non-po2 size. So the knob
# was being decided at a point where one arm is anomalously crippled — a pathological probe,
# not a representative one.
            n = _avoid_po2(max(4096, 2 * _L2_BYTES ÷ sizeof(Float64)), 8 * _vwidth(Float64))
            W = _vwidth(Float64)
            # NB SETS OF OPERANDS, ROTATED PER ROUND. One buffer pair is one draw of page placement /
            # THP state / allocator addresses, and for THIS knob the winner depends on that draw: with
            # 15 rounds a tie-driven false positive is 0.05%, yet fresh processes still resolved 208 in
            # some and 4 in others, so the candidates' relative speed genuinely differs per placement.
            # Rotating means the sign count aggregates over placements instead of re-measuring one.
            # The pointers are read through `Ref`s so the candidate closures (which must stay
            # allocation-free and literal-`Val`) see the current pair without being rebuilt per round.
            nb = 4
            xs = [fill(1.0, n) for _ in 1:nb]
            ys = [fill(0.5, n) for _ in 1:nb]
            # ⚠ `bt` IS SEEDED WITH THE DEFAULT'S OWN MEASURED TIME, not `typemax`. `typemax*95` WRAPS to
            # 2^64-95 — still far above any real time — so with a typemax seed the FIRST candidate always
            # displaces and `best = _UNROLL` can never win a tie; the effective incumbent was whichever
            # arm happened to be written first (Val(2)). That silently voided "ties go to the incumbent,
            # which is the derived default" (cpuinfo.jl) on exactly the boxes the margin exists for.
            best = _UNROLL
            GC.@preserve xs ys begin
                pxr = Ref(pointer(xs[1])); pyr = Ref(pointer(ys[1]))
                rot(r) = (i = mod1(r, nb); pxr[] = pointer(xs[i]); pyr[] = pointer(ys[i]); nothing)
                px() = pxr[]; py() = pyr[]
                inc() = _axpy_unrolled!(Val(4), n, 1.0e-9, px(), py(), 0)   # req8-ok: the incumbent _UNROLL=4
                # SHAPE x DEPTH, not depth alone. Codes: u -> interleaved(u); 100+u -> phase(u, full
                # width); 200+u -> phase(u, narrow/256-bit). Measured in situ (Fable's A/B, three fresh
                # runs per box): Zen5 n=3e4 phase4 reads 1.063/1.099/1.111 vs shipped and BEATS OpenBLAS
                # by ~2.6pp, while the interleaved-8 control reproduces the earlier width falsification
                # (0.99). Zen4/Zen3 keep an interleaved arm. Depth alone could never express this.
                # CANDIDATES ARE WRITTEN OUT, one literal `Val` each — NOT a loop over `Val(u)`. A loop
                # variable makes `Val(u)` non-concrete, and the resulting `runtime dispatch detected:
                # _axpy_unrolled!(%3::Val, …)` propagates out of the measure, through `_axpy_dram()`, and
                # breaks the @typestable contract on the PUBLIC `axpy!` path (strictmode_tests.jl:11).
                # The guards const-fold: `W` and `_NVREG` are compile-time consts.
                # ⚠ DUELS. This knob flipped between 2 and 4 across fresh processes of one binary
                # (measured 2026-08-06, wintermute, freq-locked, quiet: 2 2 4 4 4 4 2 2), so the margin
                # was choosing the shipped kernel by coin toss here too. NOTE how that was nearly missed:
                # an earlier 5-process sample drew 2 2 2 2 2 and was recorded as "stable". Five draws
                # cannot see a ~50/50 flip — the acceptance test needs enough processes to resolve one,
                # and the `tune=` stamp in the cache header is what exposed the disagreement.
                # Each candidate duels the INCUMBENT over rotated rounds with a per-round median and a
                # supermajority; first to earn it wins, so a later arm cannot displace on a difference
                # the rule has already judged unresolvable. Literal `Val`s throughout: a loop variable
                # makes `Val(u)` non-concrete and the runtime dispatch propagates out through
                # `_axpy_band()` to break the @typestable contract on the PUBLIC axpy! path.
                if W >= 8                                    # a narrow arm only exists at 512-bit
                    _tune_wins_it(_tune_duel(inc, () -> _axpy_phase!(Val(8), _AXPY_VHW_F64, n, 1.0e-9, px(), py()); refresh = rot)) && return 208  # req8-ok: candidate arm
                end
                if 8 * W <= _NVREG
                    _tune_wins_it(_tune_duel(inc, () -> _axpy_phase!(Val(8), _AXPY_VW_F64, n, 1.0e-9, px(), py()); refresh = rot)) && return 108  # req8-ok: candidate arm
                end
                if 4 * W <= _NVREG
                    _tune_wins_it(_tune_duel(inc, () -> _axpy_phase!(Val(4), _AXPY_VW_F64, n, 1.0e-9, px(), py()); refresh = rot)) && return 104  # req8-ok: candidate arm
                end
                if 8 * W <= _NVREG
                    _tune_wins_it(_tune_duel(inc, () -> _axpy_unrolled!(Val(8), n, 1.0e-9, px(), py(), 0); refresh = rot)) && return 8  # req8-ok: candidate arm
                end
                if 2 * W <= _NVREG
                    _tune_wins_it(_tune_duel(inc, () -> _axpy_unrolled!(Val(2), n, 1.0e-9, px(), py(), 0); refresh = rot)) && return 2  # req8-ok: candidate arm
                end
            end
            return best
        catch
            return _UNROLL
        end
    end
    const _AXPY_UNROLL_ONCE = Base.OncePerProcess{Int}(_measure_axpy_unroll)
    @inline _axpy_band() = _AXPY_UNROLL_ONCE()
else
    @inline _axpy_band() = _AXPY_UNROLL_PREF::Int
end

# SECOND KNOB, DRAM REGIME — ITS OWN PREFERENCE AND ITS OWN GATE. It previously lived inside the
# `axpy_unroll` block and fell back to `_AXPY_UNROLL_PREF`, which (a) silently collapsed two
# independently-measured knobs onto one value as soon as either was pinned, and (b) left NO
# `axpy_dram` preference to set, so the trim/.so build could not pin it — and req#8b requires every
# Measure-tier knob to be pinned there, since a runtime benchmark is not trim-safe. That is exactly
# what made `daxpy_64_` fail trim checking.
const _AXPY_DRAM_PREF = @load_preference("axpy_dram", nothing)
# One knob cannot serve both bands: probed at 4xL1 the winner was the
# narrow phase body, and shipping it everywhere past L1 REGRESSED the L2/L3 cells on Zen4
# (n=1e5 1.018 -> 0.986, n=3e5 1.020 -> 0.981) while helping only past L3. Same lesson as the first
# split — a Measure knob governs the regime it was probed in, and nothing else. Probe at 2xL3, which
# is unambiguously DRAM-resident on every fleet box.
@static if isnothing(_AXPY_DRAM_PREF)
    function _measure_axpy_dram()::Int
        Base.generating_output() && return _UNROLL
        try
            # po2 probe length — see the note on the band knob above.
            n = _avoid_po2(max(4096, 2 * _L3_BYTES ÷ sizeof(Float64)), 8 * _vwidth(Float64))
            W = _vwidth(Float64)
            # NB OPERAND SETS, ROTATED PER ROUND — the same fix the band knob needed. Duels ALONE left
            # this one resolving 4/208 across fresh processes on Zen4 while its rotated sibling was
            # stable (measured 2026-08-07, one knob per process). Duelling resamples TIME; when the
            # winner depends on state fixed once per process — page placement, THP, allocator addresses
            # — every round re-measures the same draw and more rounds converge harder onto it.
            # ⚠ The symptom is BOX-DEPENDENT: this knob is stable on Zen5 and unstable on Zen4, and
            # `zaxpy_narrow` is the other way round. So rotation is applied uniformly to every Measure
            # knob rather than chased per box, where it would look fixed on whichever box was checked.
            nb = 4
            xs = [fill(1.0, n) for _ in 1:nb]
            ys = [fill(0.5, n) for _ in 1:nb]
            # ⚠ DUELS, not a margin — this knob is THE reason the rule changed. Measured 2026-08-06,
            # wintermute, freq-locked and quiet, five fresh processes of the same binary under the old
            # margin rule: 208, 4, 2, 208, 4. THREE different kernels shipping from one binary, which is
            # the Zen5 failure this file's own comments already record, reproduced on Zen4. The margin
            # cannot separate these candidates because their differences sit inside its threshold, so
            # noise picked the winner every time.
            # `_tune_duel` runs each candidate against the INCUMBENT (_UNROLL) over rotated rounds whose
            # per-round statistic is a median, and `_tune_wins_it` needs a supermajority — no noise floor
            # is estimated, and a candidate that is merely lucky once cannot win. Candidates are still
            # written out with literal `Val`s (a loop variable makes `Val(u)` non-concrete and produced
            # the runtime dispatch that failed the @typestable contract on the public axpy! path).
            best = _UNROLL
            GC.@preserve xs ys begin
                pxr = Ref(pointer(xs[1])); pyr = Ref(pointer(ys[1]))
                rot(r) = (i = mod1(r, nb); pxr[] = pointer(xs[i]); pyr[] = pointer(ys[i]); nothing)
                px() = pxr[]; py() = pyr[]
                inc() = _axpy_unrolled!(Val(4), n, 1.0e-9, px(), py(), 0)   # req8-ok: the incumbent _UNROLL=4
                # Ordered widest-effect first; the first candidate to earn a supermajority wins, so a
                # later one cannot displace on a difference the rule has already judged unresolvable.
                if W >= 8
                    _tune_wins_it(_tune_duel(inc, () -> _axpy_phase!(Val(8), _AXPY_VHW_F64, n, 1.0e-9, px(), py()); refresh = rot)) && return 208  # req8-ok: candidate arm
                end
                _tune_wins_it(_tune_duel(inc, () -> _axpy_phase!(Val(8), _AXPY_VW_F64, n, 1.0e-9, px(), py()); refresh = rot)) && return 108  # req8-ok: candidate arm
                _tune_wins_it(_tune_duel(inc, () -> _axpy_unrolled!(Val(2), n, 1.0e-9, px(), py(), 0); refresh = rot)) && return 2  # req8-ok: candidate arm
            end
            return best
        catch
            return _UNROLL
        end
    end
    const _AXPY_DRAM_ONCE = Base.OncePerProcess{Int}(_measure_axpy_dram)
    @inline _axpy_dram() = _AXPY_DRAM_ONCE()
else
    @inline _axpy_dram() = _AXPY_DRAM_PREF::Int
end

# Static ladder: runtime knob -> compile-time `Val`, one branch, each arm statically dispatched (no
# dynamic `Val(u)` in the hot path, so this stays allocation-free and StrictMode-clean).
@inline function _axpy_simd!(n::Int, a::T, x, y, pf::Int = 0) where {T <: BlasReal}
    # THE KNOB ONLY GOVERNS WHERE IT WAS MEASURED. `_measure_axpy_unroll` probes past L1, so its answer
    # applies past L1 and nowhere else — the probe-regime rule, and here it is not academic. Measured on
    # Zen4 (freq-locked, plots.jl) with the tuned value applied at EVERY size, against the fixed 4:
    #     1k 1.095->1.046 | 3k 1.025->1.012 | 10k 1.007->0.997 | 30k 1.002->1.000
    #     100k 1.009->1.015 | 300k 1.007->1.018 | 1e6 0.986->0.993
    # Narrow wins past L2 and LOSES inside it: applying one value everywhere traded the n=1e6 miss for a
    # new n=1e4 miss. The optimum is size-dependent as well as machine-dependent, so the residency split
    # is part of the knob, not a detail.
    # Short calls (complex `ger` per column, tails) also skip the OncePerProcess lookup entirely.
    (n < 4 * _UNROLL * _vwidth(T) || n * sizeof(T) <= _L1_BYTES) &&
        return _axpy_unrolled!(Val(_UNROLL), n, a, x, y, pf)
    # `pf > 0` (ger's prefetching caller) stays on the interleaved body: its prefetch distance is tuned
    # against that step, and the phase bodies do not carry the prefetch block.
    pf > 0 && return _axpy_unrolled!(Val(_UNROLL), n, a, x, y, pf)
    # THREE REGIMES, each governed by a probe measured IN that regime. Applying one knob across the whole
    # range above L1 regressed Zen4 twice: first the tuned DEPTH broke n=1e4, then the tuned SHAPE broke
    # n=1e5/3e5 (1.018 -> 0.986 and 1.020 -> 0.981) while helping only past L3. Cache-resident streaming
    # and DRAM streaming are different problems and get different knobs.
    u = n * sizeof(T) > _L3_BYTES ? _axpy_dram() : _axpy_band()
    W = _vwidth(T)
    return u == 2 ? _axpy_unrolled!(Val(2), n, a, x, y, 0) :  # req8-ok: candidate arm, literal required for specialization
        u == 8 ? _axpy_unrolled!(Val(8), n, a, x, y, 0) :  # req8-ok: candidate arm, literal required for specialization
        u == 104 ? _axpy_phase!(Val(4), Val(W), n, a, x, y) :  # req8-ok: candidate arm, literal required for specialization
        u == 108 ? _axpy_phase!(Val(8), Val(W), n, a, x, y) :  # req8-ok: candidate arm, literal required for specialization
        u == 208 ? _axpy_phase!(Val(8), Val(W ÷ 2), n, a, x, y) :  # req8-ok: candidate arm, literal required for specialization
        _axpy_unrolled!(Val(4), n, a, x, y, 0)  # req8-ok: candidate arm, literal required for specialization
end

@inline function _scal_simd!(n::Int, a::T, x) where {T <: BlasReal}
    px = _ptr(x); V = _vec(T); W = _vwidth(T); sz = sizeof(T); step = _UNROLL * W
    GC.@preserve x begin
        va = V(a)
        i = 0
        # Beyond L1, HAND-UNROLLING LOSES TO `@simd ivdep`. Measured 2026-07-30, wintermute freq-locked,
        # one process, back-to-back, plots.jl's own L1 regime (`_L1REP` reps on the same vector) — GB/s of
        # this kernel vs the identical pointer loop written as `@simd ivdep for i in 1:n`:
        #     n=1e3  0.99   n=3e3  0.98   n=1e4  1.01   n=3e4  1.00
        #     n=1e5  1.04   n=3e5  1.03   n=1e6  1.08
        # The crossover is L1 residency: 3e3·8 = 24 KB fits a 32 KB L1 (manual unroll wins), 1e4·8 = 80 KB
        # does not (ivdep wins). So keep the explicit 4× unroll while resident — it pipelines a short loop
        # better — and hand the rest to LLVM, which picks its own unroll/addressing and does better.
        # This is also the ACTUAL scal gate gap, and it is NOT a memory-parallelism story: at n=1e6 (8 MB
        # in a 16 MB L3) a plain Julia `x[i] *= a` loop reaches 152.8 GB/s where this kernel reached 141.6
        # — PB/raw = 0.927 against the gate's PB/AOCL = 0.923, i.e. AOCL is simply achieving what the naive
        # loop achieves. The genuine DRAM regime (n=4e6, 32 MB) sits at 76-79 GB/s with PB/raw = 0.973.
        # PDM: DERIVE tier (L1 residency over a detected const), no new knob.
        # OPEN CELL (2026-08-06, wintermute/Zen4 freq-locked): this branch reads ~0.993 vs OpenBLAS in
        # the L2 band — gate scal n=30000 0.992-0.993 over two independent runs. It is a FLAT ~0.6%
        # bandwidth difference, not a per-call cost: PB 174.2-174.9 GB/s vs OB 175.3-176.1 GB/s across
        # n=2e4/3e4/4e4, where every cell moves the same total bytes (plots.jl's reps ∝ 1/n). Zen3 does
        # NOT show it (1.006 at the same sizes), and n≤1e4 and n≥1e5 gate on both boxes.
        # FALSIFIED for this cell — do not re-chase (all measured, bench/probes/scal_live.jl in the live
        # rep-loop regime unless noted):
        #   · the public entry frame — fixed in 989ade4, and the BARE ivdep loop still reads 0.9951 here;
        #   · loop shape — ivdep 0.9956, 2× 0.9959, 4× 0.9919, and a flat-1× shape is the WORST of the
        #     four at 0.9814. ⚠ That 1× arm is INDEX-SCALED (`px + i*sz`), so it is not OpenBLAS's loop:
        #     `dscal_k_COOPERLAKE` is pointer-bumped against a precomputed end. Nobody disassembled what
        #     LLVM emitted for the arm either, and LLVM rewrites addressing and may re-unroll. Read this
        #     line as "a 1× source shape loses", NOT as "OB's loop transcribed loses";
        #   · the `_axpy_band`/`_axpy_dram` knob lookup (+0.6 ns) and the `::AbstractVector` return
        #     annotation on the backend entry (0.27%, identical with/without/`::typeof(x)`);
        #   · the harness's unused SECOND array (plots.jl's L1 maker is `(randn(s), randn(s))` but scal
        #     touches only c[1]) — dropping it moves neither arm (ob1 1.002, pb1 0.9946);
        #   · unroll DEPTH and NON-TEMPORAL stores — falsified earlier, see the note below;
        #   · INLINING into the caller's loop (the last structural difference from OpenBLAS, which
        #     enters via an opaque ccall per call): forcing it back out of line reads 0.9912 against
        #     0.9924 inlined — no change.
        #   · the LOAD/STORE GROUPING. Disassembled 2026-08-06: LLVM's 4× unroll of the ivdep branch
        #     below issues ALL FOUR loads and only then all four stores (`vector.body40`: 4×vmulpd into
        #     zmm2-5, then 4×vmovupd), where OB alternates 1:1 and bumps a pointer against a precomputed
        #     end. Grouping four loads ahead of four stores is the textbook anti-pattern for a
        #     store-bound stream, so an arm was written with our depth, OB's interleave AND OB's
        #     pointer-bump addressing (the one combination `v1`/`v4` did not cover). It is not faster:
        #     0.996 at n=3e4, 0.990 at n=1e5, 0.954 at n=3e5. The grouping is not the mechanism.
        # THE SHAPE OF THE CELL, which is the best clue it has: the miss is a NOTCH. n=1e4 (80 KB) and
        # n≥1e5 (800 KB) both GATE, and both run the IDENTICAL ivdep branch below — only 160-320 KB
        # misses. Identical code either side means the mechanism is working-set-dependent µarch behaviour
        # (L2 prefetch pacing / replacement / store-drain), not instruction selection. That n=1e4 gates
        # is also the real argument that the deficit is per-BYTE not per-CALL, and it is much stronger
        # than the "flat across 2e4/3e4/4e4" line above: n=1e4 runs 3× the reps, so any per-rep cost
        # would miss 3× harder there. It does not.
        # Note the opaque-ccall asymmetry runs the OTHER way — OB pays ~266 LBT trampoline entries per
        # sample INSIDE its timed window and PB pays none, so OB's raw kernel beats our best loop by
        # somewhat MORE than the ratio says.
        #
        # ALIGNMENT IS DEAD FOR THE GATE REGIME — do not write the offset-pointer probe this comment
        # used to recommend. Measured 2026-08-06: the harness's arrays are unconditionally 64-byte
        # aligned, 1600/1600 across n=1e4..3e5 (allocator property of Julia's large-Memory path, not a
        # call-context one). An offset sweep would probe a regime the gate never generates. This is
        # scoped to the harness: a user-supplied misaligned view in production is a separate question.
        #
        # THE NEXT INSTRUMENT IS `perf stat`, and the plan is written out here because building it is
        # ~40 min and picking it up cold is where this cell keeps stalling. Native CLI, so rule-safe.
        #   · ISOLATE BY SUBTRACTION, not by delay flags. The maker (`randn(n)`×2 per sample) writes the
        #     whole working set and is 60-85% of wall time — its L2 traffic would swamp a 0.7% signal.
        #     Add a `count` mode to the probe: `scal_live.jl count <pb|ob|setup> <n> <S>` doing S
        #     iterations of "fresh pair + the live rep loop", where `setup` omits only the rep loop (but
        #     still calls its arm ONCE, so compilation is equalized). Subtract the setup run.
        #     The `ob` arm is a JUSTIFIED exception to never-re-measure-references: the v3 cache holds
        #     OpenBLAS's TIMES, not its COUNTERS. Keep its numbers out of every timing table.
        #   · S=20000 at n=3e4, S=40000 at n=1e4 (~30 s each); 3 repeats. Counters are exact, so the
        #     3× spread of the differences IS the error bar.
        #   · Two groups of 6 (Zen4 has 6 general counters — more would multiplex):
        #     traffic:  ls_dmnd_fills_from_sys.local_l2, ls_hw_pf_dc_fills.local_l2,
        #               l2_request_g1.rd_blk_l, l2_request_g1.l2_hw_pf, l2_pf_hit_l2.all,
        #               l2_pf_miss_l2_hit_l3.all
        #     occupancy: instructions, cycles, ls_dispatch.store_dispatch, ls_dispatch.ld_dispatch,
        #               de_no_dispatch_per_slot.backend_stalls, ls_stlf
        #   · NORMALIZE PER LINE PER PASS (lines L=n/8, passes P=S·_L1REP(n)); per-byte and per-rep both
        #     hide the ~1-fill-per-line-per-pass structure. Signal is the difference of differences:
        #         G = [r_pb(3e4) − r_ob(3e4)] − [r_pb(1e4) − r_ob(1e4)]
        #     i.e. the miss minus the cell that gates on the SAME code path.
        #   · SANITY-CHECK BEFORE READING ANYTHING: (1) demand+prefetch fills ≈ 1.0 line/line/pass for
        #     both arms at both n (both working sets exceed L1, so every line refetches each pass) — if
        #     not, the subtraction is broken; (2) cycles_pb/cycles_ob at 3e4 must reproduce ~1.007 — if
        #     the counted workload does not reproduce the gate gap it is not in the gate regime and
        #     nothing else in the run is admissible.
        #   · DECIDE: traffic G ≳ 0.007 lines/line/pass (the size of the bandwidth gap), spread under
        #     half that ⇒ PB induces more L2 traffic; then re-run with the `l2_pf_hit_l2.*` sub-masks to
        #     name the prefetcher. Traffic G ≲ 0.002 while cycles G ≈ +0.7% ⇒ same bytes, worse
        #     occupancy ⇒ read `backend_stalls`/`store_dispatch`. G ≈ 0 at BOTH n contradicts the notch
        #     and reopens the "n=1e4 gates" premise — stop rather than force an interpretation.
        #   · ONLY IF the verdict is "same traffic, worse occupancy" is an exact-asm arm worth writing,
        #     and it then needs `llvmcall`/inline asm — you cannot make LLVM emit OB's five-instruction
        #     loop from Julia source, and trying reproduces the source-vs-emitted mismatch struck above.
        #
        # ⚠ RESOLUTION IS CONFIGURATION-BOUND, and the earlier note here overstated it. Duplicating an
        # arm in one run gave 0.02% / 0.7% / 6.5% at n=3e4/1e5/3e5 — but `samples=400 seconds=0.15`
        # does NOT deliver 400 samples: the `seconds` budget includes the maker, and two `randn(n)`
        # calls dominate it at large n. Measured actual counts: 80 / 80 / 41. So the 6.5% is mostly
        # low-N sampling error in a starved window, NOT a property of the machine — raising `seconds`
        # shrinks it roughly as 1/√N. The n=3e4 floor (0.02%, and there the window is not starved) is
        # what makes the falsifications above trustworthy; the n≥1e5 entries are the weak ones.
        # And the 6.5% is NOT ordinary sampling error either: within-window spread at n=3e5 is ~5%
        # p10-p90, so a median over 41 samples has SE ≈ 0.4% and two duplicates should differ by ~0.55%
        # RMS. Seeing 6.5% — an order of magnitude more — means the per-sample distribution is SHIFTING
        # or BIMODAL between windows (page-placement/allocator modes on fresh 2.4 MB mmaps) and the
        # median hops modes. So raising `seconds` alone will NOT reliably get under 1%; the arms have to
        # be POOLED OVER ROTATED ROUNDS so each samples the same mode mixture. Do not extrapolate a
        # floor from √N — measure it with the duplicate, every time.
        # Prescription for a probe at these sizes: `seconds` 0.7 / 1.0 / 2.0 at n=3e4 / 1e5 / 3e5, with
        # 2 / 2 / 4 rounds; keep `evals=1` (the gate uses it, and >1 would average away exactly the
        # per-context variance being characterised); keep `reps` at `_L1REP` — raising it amortises
        # setup but changes the warmup-vs-steady-state split INSIDE the timed region, which is a
        # probe-regime violation with extra steps. Print the per-round medians (as plots.jl:267 does):
        # monotone across rounds = drift, jumping = the mode-hopping above.
        # Two rules for any future scal probe: duplicate the ANCHOR arm (`pb`, the one every ratio is
        # formed against), and ROTATE arm order per round like the gate does (plots.jl:252-255) rather
        # than merely putting the duplicate last. First-in-last-slot bounds only maximal-separation
        # drift — conservative, and its failure mode is calling a resolvable cell unresolvable.
        if n * sizeof(T) > _L1_BYTES
            @inbounds @simd ivdep for j in 1:n
                unsafe_store!(px, a * unsafe_load(px, j), j)
            end
            return x
        end
        # FALSIFIED 2026-07-30 (wintermute, freq-locked, plots.jl op=scal): scal misses AOCL ONLY at
        # n=1e6 (8 MB against a 16 MB L3, doubled by RMW read + dirty-writeback) and gates everywhere
        # smaller. Deepening the unroll to 8 for that regime — i.e. 8 cache lines in flight instead of
        # 4 — moved it 0.91 → 0.92, noise. So lines-in-flight is NOT the mechanism; do not re-try depth.
        # Also do not re-try NON-TEMPORAL stores: this is a load+store to the SAME address (an RMW), the
        # line is already resident from the load so there is no RFO to skip, and NT on the identical
        # structure was measured at ger n=1024 0.99 → 0.45 (see kb pureblas-zen5-ger).
        # Note `_UNROLL = 4`'s upstream justification ("reductions use 4 independent accumulators so the
        # FMA/add latency is hidden") does NOT apply here — `_scal_simd!` has no accumulator — but the
        # measurement above says the value is not what is costing us, so it stays.
        while i + step <= n
            o = i * sz
            vstore(va * vload(V, px + o), px + o)
            vstore(va * vload(V, px + o + W * sz), px + o + W * sz)
            vstore(va * vload(V, px + o + 2W * sz), px + o + 2W * sz)
            vstore(va * vload(V, px + o + 3W * sz), px + o + 3W * sz)
            i += step
        end
        while i + W <= n
            o = i * sz
            vstore(va * vload(V, px + o), px + o)
            i += W
        end
        while i < n
            j = i + 1
            unsafe_store!(px, a * unsafe_load(px, j), j)
            i += 1
        end
    end
    return x
end

# Complex scal: x .*= (alr + i·ali). Bandwidth-bound (read-modify-write), so minimise the shuffle chain:
# for a SCALAR multiplier, one swap-adjacent-pairs shuffle suffices — result = v·alr + swap(v)·[−ali,+ali…]
# (= [r·alr − i·ali, i·alr + r·ali, …] on the interleaved [r i r i…] buffer). One shuffle/vector (vs 3 for
# deinterleave+interleave), 4× unrolled to saturate memory bandwidth. `n` counts COMPLEX elements.
@generated function _scal_cmplx_simd!(n::Int, alr::T, ali::T, x) where {T <: BlasReal}
    W = _vwidth(T); V2 = Vec{2W, T}; sz = sizeof(T); Wc = 2 * W          # reals per Vec = 2W; W complex
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(2W - 1))...)     # swap adjacent (re,im)
    sgn = :($V2($(Expr(:tuple, (iseven(l) ? :(-ali) : :ali for l in 0:(2W - 1))...))))  # [−ali,ali,…]
    return quote
        px = _reptr(x); arv = $V2(alr); sv = $sgn; step = 4 * $W          # 4 vectors = 4W complex/step
        GC.@preserve x begin
            i = 0
            while i + step <= n
                @inbounds for u in 0:3
                    o = (i + u * $W) * 2 * $sz; v = vload($V2, px + o)
                    vstore(muladd(shufflevector(v, Val($swp)), sv, v * arv), px + o)
                end
                i += step
            end
            while i + $W <= n
                o = i * 2 * $sz; v = vload($V2, px + o)
                vstore(muladd(shufflevector(v, Val($swp)), sv, v * arv), px + o)
                i += $W
            end
            while i < n
                j = i + 1; re = unsafe_load(px, 2j - 1); im = unsafe_load(px, 2j)
                unsafe_store!(px, alr * re - ali * im, 2j - 1); unsafe_store!(px, alr * im + ali * re, 2j)
                i += 1
            end
        end
        return x
    end
end

# Complex axpy: y .+= (alr + i·ali)·x. Swap-pairs complex-multiply of x, fused straight into y.
# DECOMPOSITION (galen Zen3/AVX2, n=1M ≈ 32MB footprint = L3 edge, the worst point): the complex kernel
# under-extracted bandwidth with NO register spills and no prefetch/unroll sensitivity (all falsified) — so
# the residual was the compute critical path at the L3→DRAM transition, not memory. The old body was 4 vector
# ops/lane (mul xv·arv, shuffle, FMA, then a standalone `y + ax` add). Folding the y-load into the FIRST FMA
# as its addend drops it to 1 shuffle + 2 FMAs = 3 ops/lane (−25%), identical numerics, shorter dep chain
# (yv feeds the FMA directly): t = yv + xv·arv ; result = t + swap(xv)·sv. ISA-neutral. Controlled same-process
# A/B @1M: 96.5→99.1 GB/s, PB/OB 0.937→0.958 (5/5 trials). Residual vs OB is complex-specific L3-edge
# scheduling (shuffle+2FMA density limits bandwidth for OB's complex too: real 111 vs complex-OB 104 GB/s at
# matched 32MB/bytes). 4× unrolled (bandwidth-bound: reads x+y, writes y). `n` counts COMPLEX elements.
# PAST-L2 VARIANT: 256-bit vectors, and the iteration PHASE-SEPARATED (all x loads, all y loads, all
# compute, all stores) — the shape OpenBLAS's zaxpy_kernel_4 uses (verified by disassembly: it is a ymm
# kernel even under the Cooperlake dispatch, with all loads issued up front and the stores batched).
#
# BOTH halves are required, and only past L2. Measured Zen4, freq-locked (GB/s, shipped / narrow+phase /
# native+phase / OpenBLAS):
#   n=1e3   129.7 /  78.6 /  77.3 / 119.7      n=1e5  108.0 / 111.9 / 106.7 / 112.5
#   n=4096  130.0 / 113.4 / 112.8 / 129.2      n=3e5  105.0 / 110.5 / 103.2 / 109.9
#   n=1e4   128.8 / 117.4 / 118.6 / 126.4      n=1e6   69.5 /  69.9 /  69.2 /  70.7
#   n=3e4   114.2 / 106.9 / 107.8 / 111.8
# The phase shape is CATASTROPHIC while resident (78.6 vs 129.7 at n=1e3) and the narrow width only pays
# once a stream leaves L2 — applying it globally cost zaxpy 0.952→0.877 and, because complex `ger` calls
# this per COLUMN at exactly those short lengths, zgeru 0.909→0.754. Do not hoist the switch.
@generated function _axpy_cmplx_phase!(::Val{LANES}, n::Int, alr::T, ali::T, x, y) where {LANES, T <: BlasReal}
    V = Vec{LANES, T}; sz = sizeof(T); cpx = LANES ÷ 2; U = 4
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(LANES - 1))...)
    sgn = :($V($(Expr(:tuple, (iseven(l) ? :(-ali) : :ali for l in 0:(LANES - 1))...))))
    lx = [:($(Symbol(:xv, u)) = vload($V, px + (i + $u * $cpx) * 2 * $sz)) for u in 0:(U - 1)]
    ly = [:($(Symbol(:yv, u)) = vload($V, py + (i + $u * $cpx) * 2 * $sz)) for u in 0:(U - 1)]
    cp = [
        :(
            $(Symbol(:rv, u)) = muladd(
                shufflevector($(Symbol(:xv, u)), Val($swp)), sv,
                muladd($(Symbol(:xv, u)), arv, $(Symbol(:yv, u)))
            )
        ) for u in 0:(U - 1)
    ]
    st = [:(vstore($(Symbol(:rv, u)), py + (i + $u * $cpx) * 2 * $sz)) for u in 0:(U - 1)]
    return quote
        $(Expr(:meta, :inline))
        px = _reptr(x); py = _reptr(y); arv = $V(alr); sv = $sgn; step = $U * $cpx
        GC.@preserve x y begin
            i = 0
            @inbounds while i + step <= n
                $(lx...)                       # phase 1: x stream
                $(ly...)                       # phase 2: y stream
                $(cp...)                       # phase 3: compute
                $(st...)                       # phase 4: stores batched
                i += step
            end
            @inbounds while i + $cpx <= n
                o = i * 2 * $sz; xv = vload($V, px + o)
                t = muladd(xv, arv, vload($V, py + o))
                vstore(muladd(shufflevector(xv, Val($swp)), sv, t), py + o); i += $cpx
            end
            @inbounds while i < n
                j = i + 1; xr = unsafe_load(px, 2j - 1); xi = unsafe_load(px, 2j)
                unsafe_store!(py, unsafe_load(py, 2j - 1) + alr * xr - ali * xi, 2j - 1)
                unsafe_store!(py, unsafe_load(py, 2j) + alr * xi + ali * xr, 2j)
                i += 1
            end
        end
        return y
    end
end

# Is the past-L2 variant actually faster HERE? PDM **Measure** tier: it is a datapath property (Zen4
# double-pumps 512-bit ops over a 256-bit path, so the wide vector buys no bandwidth on a memory-bound
# kernel; a true 512-bit datapath can flip the sign), which no formula over the detected consts predicts
# — the req#8b tell. The WINDOW, by contrast, is Derived: one stream leaving L2.
@inline _zaxpy_narrow_lanes(::Type{T}) where {T} = 32 ÷ sizeof(T)      # 256 bits
const _ZAXPY_NARROW_PREF = @load_preference("zaxpy_narrow", nothing)
@static if isnothing(_ZAXPY_NARROW_PREF)
    function _measure_zaxpy_narrow()::Bool                 # straight-line, no closures, TOTAL
        Base.generating_output() && return false           # never burn a measure during precompile
        try
            nn = _zaxpy_narrow_lanes(Float64); nw = 2 * _vwidth(Float64)
            nn == nw && return false
            n = max(4096, 3 * _L2_BYTES ÷ sizeof(ComplexF64))   # inside the window: stream > L2
            x = fill(ComplexF64(1.0, 0.5), n); y = fill(ComplexF64(0.25, -0.75), n)
            # MEDIAN of interleaved rounds (`_tune_ab`, cpuinfo.jl) — was min-of-3, which is optimistic
            # and tail-blind and is the estimator that ranked an iamax unroll backwards.
            tn, tw = _tune_ab(
                () -> _axpy_cmplx_phase!(Val(_zaxpy_narrow_lanes(Float64)), n, 1.0e-9, 0.0, x, y),
                () -> _axpy_cmplx_wide!(n, 1.0e-9, 0.0, x, y)
            )
            return tn < tw
        catch
            return false
        end
    end
    const _ZAXPY_NARROW_ONCE = Base.OncePerProcess{Bool}(_measure_zaxpy_narrow)
    @inline _zaxpy_narrow() = _ZAXPY_NARROW_ONCE()
else
    @inline _zaxpy_narrow() = _ZAXPY_NARROW_PREF::Bool
end

# The size test comes FIRST so short calls (complex `ger`, per column) never even reach the
# OncePerProcess lookup and run byte-identical code to before.
# THE L1..L2 BAND — PDM **Derive** tier, criterion = WORKING-SET RESIDENCY IN L1.
#
# `bytes` counts ONE complex array, but axpy touches TWO streams (x read, y read-modify-write), so the
# footprint is `2*bytes`. The measured crossover is exactly where that footprint stops fitting in L1:
# while both streams are L1-resident the interleaved body's shorter dependency chain wins; once they are
# not, the phase-separated body's batched loads (more outstanding misses) win. Hence `2*bytes <= _L1`.
#
# Measured, all three µarchs, freq-locked, Chairmarks median, vs the live OpenBLAS (bench/probes/
# zaxpy_arms.jl). "wide" = interleaved incumbent, "ph" = 256-bit phase; ratios are PB/OB:
#            Zen5 (L1 48K)          Zen4 (L1 32K)          Zen3 (L1 32K)
#   n=1000   1.386 / 1.222  wide    1.094 / 1.092  tie     (L1-resident, wide kept)
#   n=2000   0.905 / 1.024  PH      1.014 / 1.011  tie
#   n=3000   0.919 / 1.037  PH      1.002 / 1.009  PH
#   n=6000   0.891 / 1.025  PH      0.875 / 0.998  PH
#   n=1e4    0.871 / 1.011  PH      0.999 / 1.003  PH      0.999 / 1.011  PH
#   n=2e4    1.005 / 1.004  tie     0.988 / 1.004  PH
#   n=6e4    0.996 / 0.988  tie     0.955 / 1.014  PH
# The rule predicts every point: Zen5 n=1000 is 32K ≤ 48K L1 ⇒ wide (1.386, and phase would cost 12pp);
# Zen5 n=2000 is 64K > 48K ⇒ phase (0.905 → 1.024). It also repairs Zen4 cells this campaign was not
# even chasing (n=6000 0.875 → 0.998, n=6e4 0.955 → 1.014).
#
# WHY DERIVE AND NOT MEASURE: an on-host `OncePerProcess` A/B here read 2,2,0,0,2,2,1,2 across eight
# fresh processes on ONE box — the arm times differ by less than the run-to-run noise of a ~5 µs call, so
# the knob selected on noise and landed on the WRONG arm (code 2, ~0.89) while the true winner was +15%.
# A residency criterion is const-folded, trim-safe, and identical every load. This is the req#8b "Yes ⇒
# Derive" branch: the optimum IS predictable from a detected const.
#
# The old "phase is CATASTROPHIC while resident (78.6 vs 129.7 at n=1e3)" note is NOT reproducible and is
# retired: that measurement predates the `@generated`/`@inline` meta fix, which is exactly the bug that
# made `Vec` args pass BY POINTER and demoted the caller's accumulators. At n=1000 the two now tie on
# Zen4 (184.6 vs 184.9). Zen5 still prefers wide deep in L1, which the residency cut already honours.
@inline function _axpy_cmplx_simd!(n::Int, alr::T, ali::T, x, y) where {T <: BlasReal}
    bytes = n * 2 * sizeof(T)                       # ONE array; the footprint is 2*bytes
    # Past L2 keeps its own **Measure** knob untouched — it is a datapath question (see `_zaxpy_narrow`),
    # it was measured in its own regime, and those sizes gate today.
    bytes > _L2_BYTES && return _zaxpy_narrow() ?
        _axpy_cmplx_phase!(Val(_zaxpy_narrow_lanes(T)), n, alr, ali, x, y) :
        _axpy_cmplx_wide!(n, alr, ali, x, y)
    # Derived band rule. Narrow lanes (256-bit) — NOT full width: at n=1e4 on Zen5 the full-width phase
    # arm measured 0.890 against the narrow arm's 1.011, so the win is the SCHEDULING STRUCTURE, not the
    # vector width. Same conclusion as real axpy, reached independently here.
    return 2 * bytes > _L1_BYTES ?
        _axpy_cmplx_phase!(Val(_zaxpy_narrow_lanes(T)), n, alr, ali, x, y) :
        _axpy_cmplx_wide!(n, alr, ali, x, y)
end

@generated function _axpy_cmplx_wide!(n::Int, alr::T, ali::T, x, y) where {T <: BlasReal}
    W = _vwidth(T); V2 = Vec{2W, T}; sz = sizeof(T)
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(2W - 1))...)
    sgn = :($V2($(Expr(:tuple, (iseven(l) ? :(-ali) : :ali for l in 0:(2W - 1))...))))
    return quote
        px = _reptr(x); py = _reptr(y); arv = $V2(alr); sv = $sgn; step = 4 * $W
        GC.@preserve x y begin
            i = 0
            while i + step <= n
                @inbounds for u in 0:3
                    o = (i + u * $W) * 2 * $sz; xv = vload($V2, px + o)
                    t = muladd(xv, arv, vload($V2, py + o))                      # y + arv·x
                    vstore(muladd(shufflevector(xv, Val($swp)), sv, t), py + o)  # + swap(x)·sv
                end
                i += step
            end
            while i + $W <= n
                o = i * 2 * $sz; xv = vload($V2, px + o)
                t = muladd(xv, arv, vload($V2, py + o))
                vstore(muladd(shufflevector(xv, Val($swp)), sv, t), py + o); i += $W
            end
            while i < n
                j = i + 1; xr = unsafe_load(px, 2j - 1); xi = unsafe_load(px, 2j)
                unsafe_store!(py, unsafe_load(py, 2j - 1) + alr * xr - ali * xi, 2j - 1)
                unsafe_store!(py, unsafe_load(py, 2j) + alr * xi + ali * xr, 2j)
                i += 1
            end
        end
        return y
    end
end

# Complex dot: Σ (CJ ? conj(xᵢ) : xᵢ)·yᵢ. NO per-iteration deinterleave (too many shuffles on AVX2 — cost
# dotu 0.70 there). Instead accumulate two INTERLEAVED products: p = Σ x·y = [Σxr·yr, Σxi·yi, …] and
# q = Σ x·swap(y) = [Σxr·yi, Σxi·yr, …] — ONE shuffle (swap y) + 2 FMAs/iter, identical for dotu/dotc.
# Deinterleave only the 2 accumulators ONCE at the end; CJ flips two combine signs. 4× unrolled for the
# FMA-reduction latency. Returns Complex{T}. `n` counts COMPLEX elements.
@generated function _dot_cmplx_simd(n::Int, x, y, ::Type{T}, ::Val{CJ}) where {T <: BlasReal, CJ}
    W = _vwidth(T); V2 = Vec{2W, T}; sz = sizeof(T)
    # Unroll from the REGISTER BUDGET (req#8), not a literal: the 2·UNR `Vec{2W}` accumulators each take 2
    # physical vector registers, so keep 4·UNR ≤ Nregs−RESERVE, where RESERVE covers the live x/y/swap
    # values. AVX2 has 16 vector regs, AVX-512 has 32 — a hardcoded 4× put all 16 YMM into accumulators on
    # AVX2 and spilled (dotc/dotu small-n 0.75×).
    #
    # RESERVE IS 4, MEASURED — it was 6, which is one register-pair too conservative on AVX2. galen (Zen3,
    # AVX2), plots.jl's L1-sweep regime, 40 samples, GB/s median, standalone kernels differing only in UNR:
    #     n=1000   UNR 1/2/3/4 = 117.7 / 158.5 / 169.9 / 129.4     (shipped UNR=2 measured 155.5)
    #     n=3000               = 116.4 / 116.7 / 116.7 / 116.3
    #     n=10000              = 114.3 / 114.4 / 114.9 / 113.6
    # UNR=3 is +9.3% at n=1000 — two 16 KB vectors exactly fill the 32 KB L1, the one size where the
    # accumulator chain is not hidden behind memory — and is neutral above it. UNR=4 collapsing to 129.4 is
    # the spill the old comment describes, so the ceiling is real; the reserve was simply set one pair high.
    # AVX-512 clamps to 4 under BOTH forms ((32−6)÷4 = 6 and (32−4)÷4 = 7 both clamp), so Zen4/Zen5 are
    # unchanged by construction and this is an AVX2-only correction.
    UNR = clamp(((_SIMD_BYTES == 64 ? 32 : 16) - 4) ÷ 4, 1, 4)
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(2W - 1))...)
    ps = [Symbol(:p, u) for u in 0:(UNR - 1)]; qs = [Symbol(:q, u) for u in 0:(UNR - 1)]
    init = Expr(:block, (:($(ps[u + 1]) = zero($V2); $(qs[u + 1]) = zero($V2)) for u in 0:(UNR - 1))...)
    psum = foldl((a, b) -> :($a + $b), ps); qsum = foldl((a, b) -> :($a + $b), qs)
    body = Expr(:block)
    for u in 0:(UNR - 1)
        o = :((i + $u * $W) * 2 * $sz)
        push!(
            body.args, quote
                xv = vload($V2, px + $o); yv = vload($V2, py + $o)
                $(ps[u + 1]) = muladd(xv, yv, $(ps[u + 1])); $(qs[u + 1]) = muladd(xv, shufflevector(yv, Val($swp)), $(qs[u + 1]))
            end
        )
    end
    return quote
        px = _reptr(x); py = _reptr(y); step = $UNR * $W
        $init
        GC.@preserve x y begin
            i = 0
            while i + step <= n
                @inbounds begin
                    $body
                end
                i += step
            end
            while i + $W <= n
                @inbounds begin
                    xv = vload($V2, px + i * 2 * $sz); yv = vload($V2, py + i * 2 * $sz)
                    p0 = muladd(xv, yv, p0); q0 = muladd(xv, shufflevector(yv, Val($swp)), q0)
                end
                i += $W
            end
            # dotu: real=Σxr·yr−Σxi·yi, imag=Σxr·yi+Σxi·yr ;  dotc (conj x): signs of the xi terms flip.
            # Parity-preserving fold → [Σeven, Σodd] instead of deint + two full horizontal sums (see gemm.jl).
            pfld = _fold2_cmplx($psum)         # [Σxr·yr, Σxi·yi]  (unique names: $psum reads p0..p_{UNR-1})
            qfld = _fold2_cmplx($qsum)         # [Σxr·yi, Σxi·yr]
            sr = pfld[1] + $(CJ ? :(pfld[2]) : :(-pfld[2]))
            si = qfld[1] + $(CJ ? :(-qfld[2]) : :(qfld[2]))
            @inbounds while i < n
                j = i + 1; xr = unsafe_load(px, 2j - 1); xi = unsafe_load(px, 2j); yr = unsafe_load(py, 2j - 1); yi = unsafe_load(py, 2j)
                sr += xr * yr + $(CJ ? :(xi * yi) : :(-xi * yi)); si += xr * yi + $(CJ ? :(-xi * yr) : :(xi * yr))
                i += 1
            end
            return Complex{$T}(sr, si)
        end
    end
end

# Fused rank-2 complex conj-dot for the QR panel: returns (Σ conj(v0)·c, Σ conj(v1)·c) in ONE pass over
# c — the two reflectors share the c-load (halves the panel's level-2 read traffic vs two _dot_cmplx_simd
# calls). Same interleaved swap-adjacent idiom as _dot_cmplx_simd (CJ=true). `n` counts COMPLEX elements.
@generated function _qr_dot2c_cmplx(n::Int, v0, v1, c, ::Type{T}) where {T <: BlasReal}
    W = _vwidth(T); V2 = Vec{2W, T}; sz = sizeof(T)
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(2W - 1))...)
    return quote
        pv0 = _reptr(v0); pv1 = _reptr(v1); pc = _reptr(c)
        p0 = zero($V2); q0 = zero($V2); p1 = zero($V2); q1 = zero($V2); i = 0
        GC.@preserve v0 v1 c begin
            while i + $W <= n
                @inbounds begin
                    o = i * 2 * $sz
                    cv = vload($V2, pc + o); cs = shufflevector(cv, Val($swp))
                    a0 = vload($V2, pv0 + o); a1 = vload($V2, pv1 + o)
                    p0 = muladd(a0, cv, p0); q0 = muladd(a0, cs, q0)
                    p1 = muladd(a1, cv, p1); q1 = muladd(a1, cs, q1)
                end
                i += $W
            end
            pf0 = _fold2_cmplx(p0); qf0 = _fold2_cmplx(q0); pf1 = _fold2_cmplx(p1); qf1 = _fold2_cmplx(q1)
            d0r = pf0[1] + pf0[2]; d0i = qf0[1] - qf0[2]; d1r = pf1[1] + pf1[2]; d1i = qf1[1] - qf1[2]
            @inbounds while i < n
                j = i + 1
                v0r = unsafe_load(pv0, 2j - 1); v0i = unsafe_load(pv0, 2j)
                v1r = unsafe_load(pv1, 2j - 1); v1i = unsafe_load(pv1, 2j)
                cr = unsafe_load(pc, 2j - 1); ci = unsafe_load(pc, 2j)
                d0r += v0r * cr + v0i * ci; d0i += v0r * ci - v0i * cr
                d1r += v1r * cr + v1i * ci; d1i += v1r * ci - v1i * cr
                i += 1
            end
            return (Complex{$T}(d0r, d0i), Complex{$T}(d1r, d1i))
        end
    end
end

# Fused rank-2 complex axpy for the QR panel: c .+= k0·v0 + k1·v1 in ONE pass over c (shares the c
# read/write across both reflectors). Swap-pairs complex-multiply, same idiom as _axpy_cmplx_simd.
@generated function _qr_axpy2_cmplx!(n::Int, k0r::T, k0i::T, k1r::T, k1i::T, v0, v1, c) where {T <: BlasReal}
    W = _vwidth(T); V2 = Vec{2W, T}; sz = sizeof(T)
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(2W - 1))...)
    sgn0 = :($V2($(Expr(:tuple, (iseven(l) ? :(-k0i) : :k0i for l in 0:(2W - 1))...))))
    sgn1 = :($V2($(Expr(:tuple, (iseven(l) ? :(-k1i) : :k1i for l in 0:(2W - 1))...))))
    return quote
        pv0 = _reptr(v0); pv1 = _reptr(v1); pc = _reptr(c)
        ar0 = $V2(k0r); ar1 = $V2(k1r); s0 = $sgn0; s1 = $sgn1; i = 0
        GC.@preserve v0 v1 c begin
            while i + $W <= n
                @inbounds begin
                    o = i * 2 * $sz
                    a0 = vload($V2, pv0 + o); a1 = vload($V2, pv1 + o); t = vload($V2, pc + o)
                    t = muladd(a0, ar0, t); t = muladd(shufflevector(a0, Val($swp)), s0, t)
                    t = muladd(a1, ar1, t); t = muladd(shufflevector(a1, Val($swp)), s1, t)
                    vstore(t, pc + o)
                end
                i += $W
            end
            @inbounds while i < n
                j = i + 1
                v0r = unsafe_load(pv0, 2j - 1); v0i = unsafe_load(pv0, 2j)
                v1r = unsafe_load(pv1, 2j - 1); v1i = unsafe_load(pv1, 2j)
                cr = unsafe_load(pc, 2j - 1); ci = unsafe_load(pc, 2j)
                unsafe_store!(pc, cr + k0r * v0r - k0i * v0i + k1r * v1r - k1i * v1i, 2j - 1)
                unsafe_store!(pc, ci + k0r * v0i + k0i * v0r + k1r * v1i + k1i * v1r, 2j)
                i += 1
            end
        end
        return c
    end
end

@inline function _copy_simd!(n::Int, x, y) # T inferred from the pointer/array element type
    T = _et(x); px = _ptr(x); py = _ptr(y); V = _vec(T); W = _vwidth(T); sz = sizeof(T); step = _UNROLL * W
    GC.@preserve x y begin
        i = 0
        while i + step <= n
            o = i * sz
            vstore(vload(V, px + o), py + o)
            vstore(vload(V, px + o + W * sz), py + o + W * sz)
            vstore(vload(V, px + o + 2W * sz), py + o + 2W * sz)
            vstore(vload(V, px + o + 3W * sz), py + o + 3W * sz)
            i += step
        end
        while i + W <= n
            o = i * sz
            vstore(vload(V, px + o), py + o)
            i += W
        end
        while i < n
            j = i + 1
            unsafe_store!(py, unsafe_load(px, j), j)
            i += 1
        end
    end
    return y
end

@inline function _swap_simd!(n::Int, x, y)
    T = _et(x); px = _ptr(x); py = _ptr(y); V = _vec(T); W = _vwidth(T); sz = sizeof(T); step = _UNROLL * W
    GC.@preserve x y begin
        i = 0
        while i + step <= n
            o = i * sz
            for u in 0:(_UNROLL - 1)
                oo = o + u * W * sz
                vx = vload(V, px + oo); vy = vload(V, py + oo)
                vstore(vy, px + oo); vstore(vx, py + oo)
            end
            i += step
        end
        while i + W <= n
            o = i * sz
            vx = vload(V, px + o); vy = vload(V, py + o)
            vstore(vy, px + o); vstore(vx, py + o)
            i += W
        end
        while i < n
            j = i + 1
            t = unsafe_load(px, j)
            unsafe_store!(px, unsafe_load(py, j), j)
            unsafe_store!(py, t, j)
            i += 1
        end
    end
    return nothing
end

# Reductions use 4 independent accumulators so the FMA/add latency is hidden — a single
# accumulator is latency-bound (serial dependency) and leaves the pipeline idle at L1-resident
# sizes. 4 chains × W lanes per iteration; then a W-at-a-time pass, then a scalar tail.
const _UNROLL = 4

@inline function _dot_simd(n::Int, x, y, ::Type{T}) where {T <: BlasReal}
    px = _ptr(x); py = _ptr(y); V = _vec(T); W = _vwidth(T); sz = sizeof(T); step = _UNROLL * W
    GC.@preserve x y begin
        a0 = zero(V); a1 = zero(V); a2 = zero(V); a3 = zero(V)
        i = 0
        while i + step <= n
            o = i * sz
            a0 = muladd(vload(V, px + o), vload(V, py + o), a0)
            a1 = muladd(vload(V, px + o + W * sz), vload(V, py + o + W * sz), a1)
            a2 = muladd(vload(V, px + o + 2W * sz), vload(V, py + o + 2W * sz), a2)
            a3 = muladd(vload(V, px + o + 3W * sz), vload(V, py + o + 3W * sz), a3)
            i += step
        end
        acc = (a0 + a1) + (a2 + a3)
        while i + W <= n
            o = i * sz
            acc = muladd(vload(V, px + o), vload(V, py + o), acc)
            i += W
        end
        s = sum(acc)
        while i < n
            j = i + 1
            s += unsafe_load(px, j) * unsafe_load(py, j)
            i += 1
        end
    end
    return s
end

# EIGHT accumulators, not four — **Derive**: cover the FP-add LATENCY across the add pipes. Each vector
# costs one load, one `abs` (a `vandpd`) and one `vaddpd` into an accumulator, so the loop's floor is set
# by how many INDEPENDENT add chains are in flight: chains >= add_latency x add_pipes. On Zen4 that is
# ~4 cycles x 2 pipes = 8; four chains leaves the pipes idle half the time.
# MEASURED, and this is why it matters (wintermute, freq-locked, `bench/cellrep.jl`): at 16 KB of
# L1-resident data our 4-chain kernel ran 2000 doubles in ~316 cycles = 1.26 cycles/vector where L1 can
# sustain ~1.0. OpenBLAS ships TWO kernels for this and we tie the slower one:
#     dasum_k_COOPERLAKE   4x zmm, 256 B/iter, serial fold   -> we WIN 1.227 (asum n=2000)
#     zasum_k_COOPERLAKE   8x zmm, 512 B/iter, pipelined     -> we LOSE 0.960 (dzasum n=1000)
# Same 16 KB, same PB time either way (112.4 vs 112.6 ns) — the complex reinterpret costs nothing, the
# gap is purely that our shared kernel is 4-wide against their 8-wide. `dzasum`/`asum` bind against
# OpenBLAS at EVERY size on this fleet (AOCL runs 7.6-19.5x slower — it never implemented these), so
# the 8-wide kernel is the target to match.
# Registers are not a constraint: 8 of 32 zmm on AVX-512, 8 of 16 ymm on AVX2.
@inline function _asum_simd(n::Int, x, ::Type{T}) where {T <: BlasReal}
    px = _ptr(x); V = _vec(T); W = _vwidth(T); sz = sizeof(T); step = 8 * W
    GC.@preserve x begin
        a0 = zero(V); a1 = zero(V); a2 = zero(V); a3 = zero(V)
        a4 = zero(V); a5 = zero(V); a6 = zero(V); a7 = zero(V)
        i = 0
        while i + step <= n
            o = i * sz
            a0 += abs(vload(V, px + o))
            a1 += abs(vload(V, px + o + W * sz))
            a2 += abs(vload(V, px + o + 2W * sz))
            a3 += abs(vload(V, px + o + 3W * sz))
            a4 += abs(vload(V, px + o + 4W * sz))
            a5 += abs(vload(V, px + o + 5W * sz))
            a6 += abs(vload(V, px + o + 6W * sz))
            a7 += abs(vload(V, px + o + 7W * sz))
            i += step
        end
        acc = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))   # balanced fold, log depth
        while i + W <= n
            acc += abs(vload(V, px + i * sz))
            i += W
        end
        s = sum(acc)
        while i < n
            j = i + 1
            s += abs(unsafe_load(px, j))
            i += 1
        end
    end
    return s
end

# Sum of squares, SIMD, 4 accumulators. Fast path for nrm2 — may overflow to Inf or underflow to 0
# on extreme inputs; the caller (_nrm2) detects that and falls back to scaled lassq.
@inline function _sumsq_simd(n::Int, x, ::Type{T}) where {T <: BlasReal}
    px = _ptr(x); V = _vec(T); W = _vwidth(T); sz = sizeof(T); step = _UNROLL * W
    GC.@preserve x begin
        a0 = zero(V); a1 = zero(V); a2 = zero(V); a3 = zero(V)
        i = 0
        while i + step <= n
            o = i * sz
            v0 = vload(V, px + o); v1 = vload(V, px + o + W * sz)
            v2 = vload(V, px + o + 2W * sz); v3 = vload(V, px + o + 3W * sz)
            a0 = muladd(v0, v0, a0); a1 = muladd(v1, v1, a1)
            a2 = muladd(v2, v2, a2); a3 = muladd(v3, v3, a3)
            i += step
        end
        acc = (a0 + a1) + (a2 + a3)
        while i + W <= n
            v = vload(V, px + i * sz); acc = muladd(v, v, acc); i += W
        end
        s = sum(acc)
        while i < n
            j = i + 1; v = unsafe_load(px, j); s = muladd(v, v, s); i += 1
        end
    end
    return s
end

# SIMD argmax for BLAS iamax: 1-based index of the first element with maximal |x|. Real unit-stride;
# assumes n ≥ 4W (caller routes shorter / strided / complex to the scalar loop). Two implementations,
# selected by ISA at build time (`_SIMD_BYTES` const-folds → trim-safe, no runtime branch):
#
#  • AVX2 + AVX-512 (`_iamax_thresh!`): DEPENDENCY-FREE THRESHOLD SCAN. The hot loop keeps a broadcast
#    running max `thr` (invariant except on the rare advance) and does FOUR INDEPENDENT `|xⱼ| > thr`
#    compares per iter, OR-ing the masks into ONE `any` branch — no loop-carried accumulator, so it
#    out-throughputs even a bare vmaxpd reduction. The predicted-not-taken cold path (max advances
#    ~O(log n) for random data) scalar-scans the blocks for the new max + its first lane. Tracks NO index
#    in the hot path, unlike the chain kernel below. WIDTH-GENERAL (W=4 f64 / W=8 f32). Beats OpenBLAS
#    1.6–2.1× on Zen4. (Old AVX2 chain was the worst AOCL miss: f64 0.55×, f32 0.32× — git log.)
#
#    It does NOT gate AOCL-BLIS. Measured 2026-08-03, wintermute, freq-locked, bench/plots.jl arms=pb,
#    pb/AOCL by size: n=1e3 0.967 | 3e3 1.039 | 1e4 0.957 | 3e4 0.974 | 1e5 0.937 | 3e5 1.03 | 1e6 1.016
#    ⇒ gate 0.937. Two claims that used to stand here are FALSE and were removed: "gates AOCL at EVERY
#    size" (unreproducible — it came from bench/iamax_nb.jl, which the kb already flags for stale seeding),
#    and "at n≥1e6 both sit on the shared DRAM roofline" (the harness runs `_L1REP(1e6)=30` reps over one
#    8 MB buffer, so it is L3-resident; 73–77 GB/s is far above Zen4 single-thread DRAM. n=1e6 is an
#    L3-bandwidth cell at the half-L3 edge, not a roofline).
#
#    THE GAP vs AOCL, read from `bli_damaxv_zen_int_avx512`'s disassembly: it issues TWO ops per vector —
#    `vandnpd (mem),zmm_sign,zmmX` (abs FUSED into the load) then `vmaxpd` into 4 accumulators — and finds
#    only the max VALUE, taking a SECOND PASS for the index. This scan issues three (vload, vandpd,
#    vcmppd). A ceiling probe deleting the abs measured +21% at n=1e3, +2% at n=1e5, ~0 at n=1e6, so the
#    third op costs real time only where the loop is issue-bound. AOCL can afford its second pass because
#    inside the gate's size range the re-read is L3-resident; it would lose in a true DRAM regime, which
#    the gate does not currently measure at all.
#    FALSIFIED 2026-08-03 (do not rebuild): porting that as a CHUNKED single-pass max-accumulate —
#    accumulate 2 ops/vector over an L1-resident chunk, re-walk only a chunk whose max beats `gmax` —
#    was correct (netlib NaN/tie intact: `max(::Vec,::Vec)` is llvm.maxnum and drops NaN, while
#    `maximum(::Vec)` and scalar `max` propagate, so the fold and tail used the strict `>` walk) and MUCH
#    slower: pb/AOCL 0.435/0.486/0.599/0.697/0.827/0.897/0.921 across n=1e3…1e6. Reason is RE-WALK
#    GRANULARITY: this scan detects per BLOCK and rescans ~NB·W elements, a chunk is 64× coarser, and at
#    small n the chunk IS the array so it degenerates to a literal two-pass. Any revival must detect at
#    block granularity — which is what this kernel already does.
#  • SSE / NEON (narrower than 32 B, unvalidated) (`_iamax_chain4!`): the original 4-chain lane-parallel
#    running max + parallel index vector (loop-carried, latency-bound → 4 chains). Kept as the fallback.
#
# Tie rule everywhere: equal value → keep the SMALLER index ⇒ BLAS first-occurrence semantics (strict `>`).
# NB: OB/AOCL idamax are alignment-volatile (time swings ~60% with the array's address) while these are
# stable — sample many fresh allocations and take the median (see bench/plots.jl iamax entry).
@inline _amax_up(m, i, nv, ni) = (t = nv > m; (vifelse(t, nv, m), vifelse(t, ni, i)))
# Fold across chains: indices interleave, so the tie must pick the smaller index explicitly.
@inline _amax_merge(m0, i0, m1, i1) = begin
    take = (m1 > m0) | ((m1 == m0) & (i1 < i0))
    (vifelse(take, m1, m0), vifelse(take, i1, i0))
end

# NB = independent |xⱼ|>thr compare-chains per hot-loop iteration. It is a REAL parameter (`Val{NB}`,
# body generated from it). It used to be `const _IAMAX_NB = min(4, _NVREG - 1)`, advertised as the
# tunable unroll but INERT and a trap: it fed only `step` while the body loaded four blocks
# unconditionally, so setting it to 2 made the loop re-read every element AND read up to 2W past the
# guard, and 8 made it SKIP half the data and return wrong answers. (The NB numbers in the old comment
# did not come from turning it — bench/iamax_nb.jl @evals its own unrolled kernels; those are suspect
# for a different reason, the kb records that file as carrying stale seeding.)
#
# NB IS CHOSEN BY L2 RESIDENCY — PDM **Derive** tier, no knob, no preference, no runtime measurement.
# Criterion: while the stream fits L2, load latency is low and two lines in flight cover it, so the
# shorter loop (less per-iteration overhead, fewer live registers) wins; once the stream leaves L2 the
# misses must be hidden by more outstanding lines, and 4 beats 2. Measured on wintermute (Zen4, L2 = 1 MB,
# freq-locked, plots.jl's own fresh-allocation regime, 90 samples, GB/s median) — NB=2 over NB=4:
#     7 KB +12.7% | 23 KB +8.8% | 78 KB +5.1% | 234 KB +4.2% | 781 KB +4.2% | 1024 KB (=L2) +1.5%
#     1562 KB −5.5% | 2343 KB −0.4% | 4.6 MB −7.6% | 7.6 MB −11.1% | 15 MB −9.5% | 30 MB −9.9%
# The sign flips exactly at L2, which is why the criterion is `n*sizeof(T) <= _L2_BYTES` and not a fitted
# constant. NOTE THE STATISTIC: at n≥600 KB the NB=2 loss is a TAIL, not a slower best case (n=1e6:
# identical min, ~15% worse median, p90/min 1.20 vs 1.06) — consistent with fewer outstanding lines
# absorbing an L3 miss worse. A 24-sample A/B could not see it and ranked NB=2 ahead at every size; it
# took ~90 samples. Do not re-rank this knob from a small sample.
# RESIDENT unroll — **Derive**: hold TWO CACHE LINES in flight per iteration. While the stream fits L2 the
# load latency is low, so extra chains buy nothing and the shorter loop's lower per-iteration overhead
# wins; two lines is what keeps the load stream fed without lengthening the body. The ISA sets how many
# blocks that is, which is the whole point of deriving it — AVX-512 (_SIMD_BYTES=64) → 2, AVX2 (32) → 4.
# BOTH are the measured optimum on their box, and they are DIFFERENT NUMBERS: shipping the Zen4 value (2)
# as a literal regressed galen's iamax from gate 1.022 (PASS) to 0.934 (FAIL), −18/−19% at n=1e3/3e3.
# That is the req#8 lesson in one line: the constant that looked µarch-specific was a fixed byte budget.
const _IAMAX_NB_RESIDENT = max(1, 2 * _CACHELINE ÷ _SIMD_BYTES)
# STREAMING unroll — past L2 the criterion changes from bytes to ILP: independent compare chains to cover
# the miss latency. Four is ISA-invariant here and is also the incumbent value, now validated BOTH ways —
# Zen4 4 blocks = 4 lines (n=1e6: 72.3 vs 64.3 GB/s for 2), Zen3 4 blocks = 2 lines (78.5, beating both 2
# at 75.0 and 8 at 77.4). Deriving this as a line budget like the resident arm gives 8 on AVX2, which
# MEASURED WORSE on galen at every size — so it is a chain count, not a byte budget.
# req8-ok: ILP chain count, ISA-invariant, incumbent value measured optimal on Zen3 and Zen4 (see above)
const _IAMAX_NB_STREAM = 4
# WIDTH IS NOT THE REMAINING GAP — measured 2026-08-04, wintermute, freq-locked, plots.jl op=iamax.
# AOCL's `bli_damaxv_zen_int_avx512` (the reference that binds here; OpenBLAS is at 2.04, we beat it 2x)
# streams 512 B/iteration = 8 zmm, against our 128 B at NB=2, so NB=8 looked like the obvious lever.
# It is decisively WORSE at every resident size — gate ratios NB=2 -> NB=8:
#     n=1e3 1.058 -> 0.822 | 3e3 1.112 -> 0.944 | 1e4 0.964 -> 0.882 | 3e4 0.976 -> 0.899 | 1e5 0.942 -> 0.890
# with NB=4 already recorded 4-5% worse in the same band. Widening OUR structure costs more than it buys:
# each extra block adds a live vector AND a live mask plus its index-walk arm, so register pressure and
# loop size grow faster than the memory-level parallelism helps.
# What is NOT yet falsified is AOCL's STRUCTURE at width: it has no per-block threshold compare and no
# mask OR-tree — it computes |x| with `vandnpd` straight off memory (we already fuse that, via `vandpd`),
# then a log-depth `vmaxpd` REDUCTION TREE, reduces to one scalar per iteration, and branches into the
# index search only when that scalar beats the running max. That keeps 8 blocks in flight with 8 vector
# registers and ONE mask-free compare, which is exactly what our width-8 attempt could not do. If iamax
# n=1e4..1e5 is attacked again, that is the design to port — not another NB value.

# MAX-REDUCTION-TREE scan — the structural port of AOCL's `bli_damaxv_zen_int_avx512`, which is the
# reference that binds `iamax` in the L2-resident band on Zen4 (OpenBLAS is at ~2x, we beat it).
#
# THE DIFFERENCE FROM `_iamax_thresh!` IS NOT WIDTH, IT IS WHAT THE HOT LOOP HOLDS. The threshold form
# compares every block against `thr` and ORs the masks, so each extra block costs a live VECTOR *and* a
# live MASK plus its own index-walk arm — which is why widening it collapsed (NB=8 measured 0.822 at
# n=1e3). This form folds the blocks into one vector with a log-depth `vifelse`-max tree and issues ONE
# compare for the whole iteration, so NB blocks cost NB vectors and a single mask. That is how AOCL keeps
# 8 blocks in flight, and it is the only structurally distinct design left after width was falsified at
# NB=4 and NB=8 under the mask form.
#
# LOWERINGS (verified on this box, Julia 1.12.6 / SIMD.jl, strict mode, no fast-math):
#   * `vifelse(a > b, a, b)` -> ONE `vmaxpd` (memory operand folds). Plain `max(::Vec,::Vec)` is
#     llvm.maxnum and lowers to `vmaxpd + vcmpunordpd + masked mov` = 3 ops, which would lose on paper.
#   * `any(!(m <= thr))`     -> ONE `vcmpnlepd` + `kortestb`, i.e. the unordered predicate in one insn.
#
# SEMANTICS — the netlib contract is preserved exactly, and every hazard is already test-pinned by the
# "iamax NaN/Inf semantics", "first-occurrence on ties" and "max-tree detect" testitems:
#   * seed `gmax = |x[1]|` (NOT typemin) — same as the threshold form; the reason is recorded there.
#   * DETECT is conservative under NaN: `vmaxpd` propagates its second operand, so a NaN anywhere in the
#     iteration reaches the tree root, and the unordered `!(m <= thr)` fires on it. It can fire when no
#     real max advanced (extra cold-path entries), never miss one.
#   * WALK is unchanged: per block, guarded by its own recomputed unordered test, then a strict `>` scan
#     lane 1..W in block order 0..NB-1, so first-occurrence on ties holds and NaN is skipped by `>`.
#     Recomputing the guard against an ALREADY-UPDATED `thr` mid-walk can only skip more blocks.
@generated function _iamax_tree!(::Val{NB}, n::Int, xp::Ptr{T}) where {NB, T <: BlasReal}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T)
    vs = [Symbol(:v, j) for j in 0:(NB - 1)]
    loads = [:($(vs[j + 1]) = abs(vload($V, xp + (o + $(j * W)) * $sz))) for j in 0:(NB - 1)]
    # BALANCED pairwise fold => depth log2(NB), not a left-fold chain of length NB.
    #
    # THE PREDICATE IS `!(b > a)`, NOT `a > b`, AND THAT IS A CORRECTNESS REQUIREMENT — not a style
    # choice. `vifelse(a > b, a, b)` takes `b` whenever the compare is false, and a compare against NaN
    # is ALWAYS false, so that form drops whichever operand sits in `a`. A NaN could therefore swallow a
    # larger value at one node and then be dropped itself one node up:
    #     node(v0,v1): 99 > NaN -> false -> takes NaN   (the 99 is gone)
    #     node(·,v2v3): NaN > 1 -> false -> takes 1     (the NaN is gone too)
    # leaving a root of all-1.0, so `any(!(m <= thr))` never fired and iamax returned its seed index.
    # That shipped: PureBLAS.iamax was WRONG (returned 1, netlib/OpenBLAS return 2) for any Float64
    # vector in this kernel's band (_L1_BYTES < n*sizeof(T) <= _L2_BYTES) holding a max followed by a
    # NaN in the SAME LANE of a later block — e.g. n=4161, x[2]=99, x[10]=NaN. The header note claiming
    # "vmaxpd propagates its second operand, so a NaN anywhere reaches the tree root" was false: a NaN
    # only survives while it stays in the `b` slot, which is why NaN-then-max passed and max-then-NaN
    # did not.
    # `!(b > a)` keeps `a` unless `b` is STRICTLY greater, so neither operand is ever discarded in
    # favour of a smaller one: for NaN-free data it is still the exact max, and when a NaN is present
    # the root is either the true max or NaN — both of which fire the unordered detect. The walk is
    # unchanged (per-block unordered guard + strict `>` scan), so first-occurrence on ties still holds.
    bal(items) = length(items) == 1 ? items[1] :
        (m = length(items) ÷ 2; a = bal(items[1:m]); b = bal(items[(m + 1):end]);
         :(vifelse(!($b > $a), $a, $b)))
    tree = bal(Any[vs...])
    walks = [quote
            if any(!($(vs[j + 1]) <= thr))            # per-block unordered guard, cold path only
                bm = gmax; bl = 0
                for l in 1:$W
                    $(vs[j + 1])[l] > bm && (bm = $(vs[j + 1])[l]; bl = l)   # strict > ⇒ first on ties
                end
                bl != 0 && (gmax = bm; bi = o + $(j * W) + bl; thr = $V(gmax))
            end
        end for j in 0:(NB - 1)]
    return quote
        $(Expr(:meta, :inline))
        step = $(NB * W)
        gmax = abs(unsafe_load(xp, 1)); bi = 1; thr = $V(gmax); o = 0
        @inbounds while o + step <= n
            $(loads...)
            m = $tree                                  # log-depth vmaxpd fold, NO masks
            if any(!(m <= thr))                        # ONE vcmpnlepd + kortestb for the whole iteration
                $(walks...)
            end
            o += step
        end
        @inbounds while o + $W <= n                    # leftover full blocks — identical to _iamax_thresh!
            u0 = abs(vload($V, xp + o * $sz))
            if any(u0 > thr)
                bm = gmax; bl = 0
                for l in 1:$W
                    u0[l] > bm && (bm = u0[l]; bl = l)
                end
                bl != 0 && (gmax = bm; bi = o + bl; thr = $V(gmax))
            end
            o += $W
        end
        @inbounds for k in (o + 1):n                   # scalar remainder
            a = abs(unsafe_load(xp, k)); a > gmax && (gmax = a; bi = k)
        end
        return bi
    end
end

@generated function _iamax_thresh!(::Val{NB}, n::Int, xp::Ptr{T}) where {NB, T <: BlasReal}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T)
    vs = [Symbol(:v, j) for j in 0:(NB - 1)]
    cs = [Symbol(:c, j) for j in 0:(NB - 1)]
    loads = [:($(vs[j + 1]) = abs(vload($V, xp + (o + $(j * W)) * $sz))) for j in 0:(NB - 1)]
    cmps = [:($(cs[j + 1]) = $(vs[j + 1]) > thr) for j in 0:(NB - 1)]
    ortree = reduce((a, b) -> :($a | $b), cs)
    walks = [quote
            if any($(cs[j + 1]))
                bm = gmax; bl = 0
                for l in 1:$W
                    $(vs[j + 1])[l] > bm && (bm = $(vs[j + 1])[l]; bl = l)   # strict > ⇒ first lane on ties
                end
                bl != 0 && (gmax = bm; bi = o + $(j * W) + bl; thr = $V(gmax))
            end
        end for j in 0:(NB - 1)]
    # Straight-line only: no ntuple, no closure, no tuple return. A closure in this loop body measured
    # 160× slower on 2026-07-31 and 300× on 2026-08-03. `Expr(:meta, :inline)` is emitted because
    # `@inline` does NOT propagate into @generated CodeInfo on 1.12.
    #
    # FALSIFIED 2026-08-03, do NOT rebuild — CHUNKED MAX-ACCUMULATE. AOCL's `bli_damaxv_zen_int_avx512`
    # issues 2 ops/vector (`vandnpd (mem),zmm_sign,zmmX` — abs fused into the load — then `vmaxpd` into 4
    # accumulators, no index in the hot loop, index on a SECOND pass) where this scan issues 3 (vload,
    # vandpd, vcmppd). Porting it single-pass — accumulate over an L1-resident chunk, re-walk only a chunk
    # whose max beats `gmax` — was CORRECT (netlib NaN/tie intact: `max(::Vec,::Vec)` is llvm.maxnum and
    # drops NaN, while `maximum(::Vec)` and scalar `max` propagate, so the fold and tail used the strict
    # `>` walk) and MUCH slower: pb/AOCL 0.435/0.486/0.599/0.697/0.827/0.897/0.921 at n=1e3…1e6. Reason is
    # RE-WALK GRANULARITY — this scan detects per BLOCK and rescans ~NB·W elements, a chunk is 64× coarser,
    # and at small n the chunk IS the array so it degenerates into a literal two-pass. Any revival must
    # detect at block granularity, which is what this already does. AOCL can afford its second pass only
    # because inside the gate's size range the re-read is L3-resident.
    return quote
        $(Expr(:meta, :inline))
        step = $(NB * W)
    # Seed from |x[1]|, NOT typemin — reference netlib `idamax` starts `DMAX = DABS(DX(1))`, and seeding
    # lower silently changed the NaN contract WITH VECTOR LENGTH: `[NaN, 1.0]` returned 1 on the scalar
    # path (n < 4W) but the index of the true max on this one, because a NaN never beats typemin and so
    # got skipped, whereas seeding from x[1] makes every later compare-with-NaN false and pins the answer
    # at 1. Same routine, two different semantics either side of an internal threshold (found 2026-07-31
    # by the new netlib-oracle NaN testitem). For finite data this is identical — element 1 is simply
    # accounted for up front instead of via the first compare — and it can only REDUCE cold-path entries,
    # since the starting threshold is now a real element rather than typemin.
    # `_iamax_simd_try` gates on n >= 4W, so x[1] always exists here.
        gmax = abs(unsafe_load(xp, 1)); bi = 1; thr = $V(gmax); o = 0
        @inbounds while o + step <= n                 # dependency-free: NB independent compares vs `thr`
            $(loads...)
            # FALSIFIED 2026-07-31: rescanning ONLY the blocks whose mask is non-empty via a small closure
            # returning the updated `(gmax, bi, thr)` is semantically exact and looks like an NB× cut of
            # cold-path work — but it measured 0.006 vs AOCL, ~160× SLOWER, at every size. The closure
            # defeats whatever keeps this loop in registers. That is why the walks below are generated
            # straight-line instead.
            $(cmps...)
            if any($ortree)                                              # rare (running max advances)
                $(walks...)
            end
            o += step
        end
        @inbounds while o + $W <= n                    # leftover full blocks
            u0 = abs(vload($V, xp + o * $sz))
            if any(u0 > thr)
                bm = gmax; bl = 0
                for l in 1:$W
                    u0[l] > bm && (bm = u0[l]; bl = l)
                end
                bl != 0 && (gmax = bm; bi = o + bl; thr = $V(gmax))
            end
            o += $W
        end
        @inbounds for k in (o + 1):n                   # scalar remainder (no OOB / masked read needed)
            a = abs(unsafe_load(xp, k)); a > gmax && (gmax = a; bi = k)
        end
        return bi
    end
end

@inline function _iamax_chain4!(n::Int, xp::Ptr{T}) where {T <: BlasReal}
    W = _vwidth(T); V = Vec{W, T}; sz = sizeof(T); step = 4W
    lane = Vec(ntuple(i -> i, Val(W)))               # 1,2,…,W
    ld(o) = abs(vload(V, xp + o * sz))
    # SEED = broadcast |x[1]|, NOT the first blocks' own values — a CORRECTNESS requirement.
    # `_amax_up` keeps its running max when `nv > m` is false, and every compare against NaN is false,
    # so a NaN that lands IN THE SEED poisons that lane forever: no later element, however large, can
    # ever displace it. Seeding each lane from ld(0..3W) made the first 4W elements act like netlib's
    # x[1] — but netlib's `dmax` can only become NaN if x[1] itself is NaN (`abs(x[k]) > dmax` never
    # succeeds for a NaN, so a NaN mid-vector is simply skipped). Broadcasting |x[1]| reproduces that
    # exactly: NaN elsewhere never enters the running max, and a NaN at position 1 correctly makes every
    # compare fail so index 1 is returned. This is the same defect that made `_iamax_tree!`'s fold drop
    # the true max (see the note there); `_iamax_thresh!`/`_iamax_tree!` were already correct here
    # because they broadcast `thr` from |x[1]|.
    a1 = abs(unsafe_load(xp, 1))
    seed_i = Vec(ntuple(_ -> 1, Val(W)))             # every lane starts at index 1, like netlib's ix
    m0 = V(a1); m1 = V(a1); m2 = V(a1); m3 = V(a1)
    i0 = seed_i; i1 = seed_i; i2 = seed_i; i3 = seed_i
    o = 0                                            # nothing preloaded now, so start at the top
    @inbounds while o + step <= n                    # 4 independent chains (cheap inner update)
        (m0, i0) = _amax_up(m0, i0, ld(o), lane + o)
        (m1, i1) = _amax_up(m1, i1, ld(o + W), lane + (o + W))
        (m2, i2) = _amax_up(m2, i2, ld(o + 2W), lane + (o + 2W))
        (m3, i3) = _amax_up(m3, i3, ld(o + 3W), lane + (o + 3W))
        o += step
    end
    (m0, i0) = _amax_merge(m0, i0, m1, i1); (m2, i2) = _amax_merge(m2, i2, m3, i3)
    (m0, i0) = _amax_merge(m0, i0, m2, i2)           # fold 4 chains → 1
    @inbounds while o + W <= n                        # leftover full blocks
        (m0, i0) = _amax_up(m0, i0, ld(o), lane + o); o += W
    end
    if o < n                                          # masked remainder (no OOB read)
        msk = lane <= (n - o)
        v = abs(vload(V, xp + o * sz, msk)); take = (v > m0) & msk
        m0 = vifelse(take, v, m0); i0 = vifelse(take, lane + o, i0)
    end
    mx = m0[1]; bi = i0[1]
    @inbounds for l in 2:W
        ml = m0[l]
        (ml > mx || (ml == mx && i0[l] < bi)) && (mx = ml; bi = i0[l])
    end
    return bi
end

# ISA-selected at build time: the `_SIMD_BYTES` compare const-folds, so the dead arm is eliminated
# (trim-safe, no runtime dispatch). AVX2 + AVX-512 (≥32 B) → threshold scan; narrower unvalidated ISAs
# (SSE/NEON) keep the 4-chain.
#
# The unroll is picked by L2 RESIDENCY — PDM Derive tier, see `_iamax_thresh!`. Both `Val`s are LITERAL,
# so this stays trim-safe (no runtime→Val) and both specializations are statically reachable; the test is
# a single compare against a const-folded `_L2_BYTES`.
# TREE width — **Derive**: eight cache lines in flight, bounded by the register file. The tree holds NB
# live vectors plus a root and `thr`, so NB must leave headroom in `_NVREG` (32 on AVX-512, 16 on AVX2);
# the line budget gives 8 on AVX-512 and 16 on AVX2, and the clamp keeps the latter buildable.
# Unlike the threshold form, an extra block here costs ONE vector and no mask — which is the whole
# reason widening is available to this structure and was not available to that one.
const _IAMAX_NB_TREE = clamp(8 * _CACHELINE ÷ _SIMD_BYTES, 4, _NVREG ÷ 4)

# THREE-WAY ROUTING, all Derive-tier over detected cache sizes:
#   L1-resident   -> threshold form, NB=2. It GATES here (n=1e3 1.058, n=3e3 1.112) and the tree's
#                    all-blocks-eligible cold path is worse at tiny n; do not disturb what passes.
#   L2-resident   -> TREE. This is the failing band (n=1e4/3e4/1e5 = 0.964/0.976/0.942 vs AOCL) and the
#                    only band where AOCL's structure beats ours.
#   beyond L2     -> threshold form, NB=4 (the stream arm). It GATES (n=3e5 1.029, n=1e6 1.022).
# The L1-resident band uses the TREE form at the RESIDENT width — form changes, width does not.
#
# Derived, not fitted. Per iteration of NB blocks the threshold form issues NB compares + (NB−1) ORs +
# 1 test ≈ 2·NB mask ops, while the tree folds log-depth and issues ONE compare: (NB−1) folds + 1
# compare + 1 test = NB+1. The mask work the tree removes therefore grows with NB — at NB=4 it is 8 ops
# against 5 — so the tree wins wherever NB is more than about 2, on any ISA. (The abs is NOT part of
# this: LLVM already folds it into the load as `vandpd (mem)`, which is AOCL's own `vandnpd` trick, so
# both forms pay 1 op/vector there. An earlier note in this file claiming we issue a separate vload was
# wrong — read off the AVX2 disassembly on galen.)
#
# Measured on galen (Zen3/AVX2, freq-locked, bench/probes/iamax_live.jl in plots.jl's rep-loop regime),
# tree vs thresh at the RESIDENT width, PB/OpenBLAS:
#     n=1000  1.937 -> 2.303 (+19%)   n=3000  1.940 -> 2.393 (+23%)
#     n=10000 2.000 -> 2.068 (+3%)    n=30000 2.129 -> 2.188 (+3%)
# n=3000 is the cell that missed: 0.966 vs AOCL, the largest real BLAS-1 gap on the fleet.
#
# NB stays `_IAMAX_NB_RESIDENT` rather than `_IAMAX_NB_TREE` on purpose. The two agree at
# _SIMD_BYTES=32 (both 4) but are 2 vs 8 at 64, so passing NB_TREE here would change form AND width on
# AVX-512 — and widening the resident band was already measured much worse (NB=8 threshold: 0.822 at
# n=1e3). Keeping the width fixed makes this one variable on every ISA.
@inline _iamax_simd!(n::Int, xp::Ptr{T}) where {T <: BlasReal} =
    _SIMD_BYTES >= 32 ?
    (n * sizeof(T) <= _L1_BYTES ? _iamax_tree!(Val(_IAMAX_NB_RESIDENT), n, xp) :
     n * sizeof(T) <= _L2_BYTES ? _iamax_tree!(Val(_IAMAX_NB_TREE), n, xp) :
     _iamax_thresh!(Val(_IAMAX_NB_STREAM), n, xp)) :
    _iamax_chain4!(n, xp)

# Complex iamax (icamax/izamax): 1-based index of the first element with maximal |re|+|im|. Same 4-chain
# argmax machinery as the real kernel, but each Vec{2W} load is W complex elements — deinterleave → re/im,
# magnitude = |re|+|im| → Vec{W}. `n` counts COMPLEX elements; xp points at the interleaved [r i r i…]
# buffer (Ptr{T}, T the real type). Was a scalar loop (~0.6× OB); this vectorizes the magnitude + argmax.
# |re|+|im| in the INTERLEAVED domain: abs, then add each real to its swapped partner (within-128-bit-lane
# shuffle — far cheaper than the cross-lane deinterleave). Result Vec{2W} has magnitude mₖ=|rₖ|+|iₖ|
# DUPLICATED in each (re,im) lane-pair: [m0,m0,m1,m1,…]. Argmax then runs over 2W lanes with the complex
# index duplicated per pair, so no deinterleave/extract shuffle at all — memory-bandwidth-bound like OB.
@inline @generated function _cmag2(v::Vec{N, T}) where {N, T}
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(N - 1))...)   # swap adjacent re↔im
    return :($(Expr(:meta, :inline)); (av = abs(v); av + shufflevector(av, Val($swp))))
end
@inline function _iamax_cmplx_simd!(n::Int, xp::Ptr{T}) where {T <: BlasReal}
    W = _vwidth(T); V = Vec{2W, T}; sz = sizeof(T); step = 4W
    clane = Vec(ntuple(i -> (i + 1) ÷ 2, Val(2W)))        # 1,1,2,2,…,W,W (complex index per real lane)
    magc(c) = _cmag2(vload(V, xp + 2c * sz))              # Vec{2W}, mₖ duplicated per pair
    # SEED = broadcast (|re|+|im|) of element 1, NOT the first blocks' own magnitudes — a CORRECTNESS
    # requirement, identical to the real `_iamax_chain4!` case (see the note there). `_amax_up` keeps
    # its running max whenever `nv > m` is false, and every compare against NaN is false, so a NaN in
    # the seed poisons that lane permanently. Measured before this fix: PureBLAS.iamax on a
    # ComplexF64 vector with z[2]=NaN+0im and z[10]=99+0im returned 1 where netlib returns 10.
    a1 = abs(unsafe_load(xp, 1)) + abs(unsafe_load(xp, 2))   # |re|+|im| of the first complex element
    seed_i = Vec(ntuple(_ -> 1, Val(2W)))                 # complex index 1 in every lane
    m0 = V(a1); m1 = V(a1); m2 = V(a1); m3 = V(a1)
    i0 = seed_i; i1 = seed_i; i2 = seed_i; i3 = seed_i
    c = 0                                                 # nothing preloaded now, so start at the top
    @inbounds while c + step <= n                         # 4 independent chains (loop-carried max latency)
        (m0, i0) = _amax_up(m0, i0, magc(c), clane + c)
        (m1, i1) = _amax_up(m1, i1, magc(c + W), clane + (c + W))
        (m2, i2) = _amax_up(m2, i2, magc(c + 2W), clane + (c + 2W))
        (m3, i3) = _amax_up(m3, i3, magc(c + 3W), clane + (c + 3W))
        c += step
    end
    (m0, i0) = _amax_merge(m0, i0, m1, i1); (m2, i2) = _amax_merge(m2, i2, m3, i3)
    (m0, i0) = _amax_merge(m0, i0, m2, i2)                # fold 4 chains → 1
    @inbounds while c + W <= n                             # leftover full blocks (W complex each)
        (m0, i0) = _amax_up(m0, i0, magc(c), clane + c); c += W
    end
    if c < n                                               # masked remainder (no OOB read)
        rem = n - c
        rmsk = Vec(ntuple(i -> i, Val(2W))) <= 2 * rem     # real-lane mask (2 reals / complex)
        av = abs(vload(V, xp + 2c * sz, rmsk)); mag = _cmag2_masked(av)
        cmsk = clane <= rem; take = (mag > m0) & cmsk
        m0 = vifelse(take, mag, m0); i0 = vifelse(take, clane + c, i0)
    end
    mx = m0[1]; bi = i0[1]
    @inbounds for l in 2:2W                                # reduce 2W lanes; strict > keeps first occurrence
        ml = m0[l]
        (ml > mx || (ml == mx && i0[l] < bi)) && (mx = ml; bi = i0[l])
    end
    return bi
end
# masked remainder magnitude: masked-out reals load as 0, so |·| pairs sum correctly (0 lanes stay 0).
@inline @generated function _cmag2_masked(av::Vec{N, T}) where {N, T}
    swp = Expr(:tuple, (isodd(l) ? l - 1 : l + 1 for l in 0:(N - 1))...)
    return :($(Expr(:meta, :inline)); av + shufflevector(av, Val($swp)))
end
