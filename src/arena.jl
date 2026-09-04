# Scratch arena — one byte-addressed bump allocator, replacing the named-field-per-routine pattern.
#
# WHY. `L3Workspace` (workspace.jl) was one mutable struct with 180 named fields, one dedicated field per
# routine's scratch buffer. That shape has three costs beyond the obvious one: every call site drags in
# scratch for routines it can never touch; the peak footprint is Σ(per-role maxima) rather than
# max(concurrently live); and it forces ONE sharing policy on every role, which is the wrong shape for
# multithreading (see kb `workspace-mt-ownership-per-buffer-not-uniform` — Octavian locks its shared
# B-cache rather than duplicating it, because a BLIS 5-loop shares the B panel BY DESIGN).
#
# Here a role does not own storage: it BORROWS a shape+stride for the duration of a lexical scope and the
# bump pointer rewinds on exit. Handles are `PtrMatrix`/`PtrVector` (ptrmat.jl) — isbits, so they pass by
# value into non-inlined kernels with no heap box, and their getindex/setindex are unsafe_load/store, so
# the whole call graph stays trim-clean.
#
# STAGE 1 (2026-09-04): the 37 FIXED-SIZE small roles are converted — `_dlaexc!`, `_syl_dlasy2`,
# `_hgeqz!`, `_tgs_tgsy2!` and `_dtgex2_big!` now open a `@scope` and borrow; their nine accessors and
# the 37 fields are deleted (L3Workspace: 180 → 143 fields). Everything that GROWS with n is still
# field-owned. Conversion continues staged, riskiest last.
#
# THE TWO HAZARDS, AND EXACTLY HOW FAR THE EXPANSION-TIME CHECKS GO.
#   * A borrow ESCAPING its scope would read memory the next borrow overwrites. `@scope` rejects the
#     TOKEN's escape at MACRO-EXPANSION time: `s` may appear only as `borrow!`'s first argument, so no
#     new borrow can be taken after the block. A compile error, not a lint finding.
#     ⚠ THE HANDLE IS NOT CHECKED, and cannot be by this mechanism — it is an ordinary local. Both of
#     these compile, and both are use-after-free:
#         @scope s begin; A = borrow!(s, Float64, 4, 4); A; end     # returned as the block's value
#         @scope s begin; f(borrow!(s, Float64, 4, 4)); end          # if `f` retains it
#     (test/arena_tests.jl's fence testitem does the first ON PURPOSE, to have something to fault on.)
#     So the escape rule buys "the token cannot leak", not "no borrow can outlive the scope"; the
#     latter is what `@fenced_scope` is for. Audit new call sites for a returned/stored handle.
#   * A borrow inside a LOOP would consume Σ(iterations) of arena rather than one buffer — at
#     n=k=4096 the per-tile `_trsm_tmp` sites would be ~16k borrows ≈ 128 MB. `@scope` rejects a
#     `borrow!` lexically inside any loop/comprehension/closure in its block: hoist it (both real sites
#     are loop-invariant) or let the loop body open its own scope, which costs nothing (try/finally is
#     free; measured).
#     ⚠ TWO WAYS PAST THIS RULE, both measured on 2026-09-04, neither closable lexically:
#       (a) A loop EMITTED BY ANOTHER MACRO. `_arena_check` walks the UNexpanded body, where
#           `@nloops`/`@turbo`/any user macro is a `:macrocall` node, not a `:for` — so a borrow under
#           one is accepted and does repeat per iteration. (Expanding first is not a fix: it would also
#           expand nested `@scope`s, whose borrows are legal and whose tokens are their own.)
#       (b) RECURSION. It is not lexical, so no macro can see it. A self-recursive routine that opens a
#           `@scope` per level holds every level's borrows at once — measured: 16 levels × 128 KiB
#           grew the arena 98 KB → 3.05 MB, and it never shrinks (the coalesced slab keeps the peak for
#           the life of the process). `_trsm!`, the named next conversion target, IS recursive: size its
#           borrows by the DEPTH-summed footprint, not the per-call one.
# Contrast today's invariant — "no accessor is reachable from inside a live claim of the same role" —
# which is whole-program and needs test/workspace_lint.jl's 300-line call-graph scanner. The arena's
# checks are cheaper and catch the common shapes; they are not a proof, and the three holes above are
# what a stage-N reviewer should be looking for.

# Borrow alignment. DERIVED, not a literal (req#8): the criterion is that a borrow must be alignable for
# BOTH of the things that read it — the widest vector load the ISA has (so a kernel can vload it aligned)
# and the cache line (so a borrow never begins mid-line and splits every access to it). That is
# max(_SIMD_BYTES, _CACHELINE), both detected in cpuinfo.jl. On the current fleet both are 64, so this
# evaluates to 64 — but it follows the hardware on a box we have never benchmarked, which is the point.
# The adjustment is computed from the ACTUAL pointer and CONSUMED from the slab, never assumed from
# `sizeof(T)`: assuming 8-byte elements overruns by up to 7 elements at Float32 and overlaps the next
# borrow, which is the bug review found in the prototype this file came from.
const _ARENA_ALIGN = max(_SIMD_BYTES, _CACHELINE)

# ── THE ELECTRIC FENCE'S GATE ───────────────────────────────────────────────────────────────────────
# The fence (further down) gives every borrow its own mmap behind a PROT_NONE guard page, so a use of a
# RELEASED handle faults at the offending line. Arming it is a per-scope decision taken by whoever OPENS
# the scope — write `@fenced_scope` instead of `@scope` — and that dispatch IS the gate. There is no
# global mode, so nothing can leave the fence armed for a whole `Pkg.test()`.
#
# Three gates were evaluated on 2026-09-03. The other two were rejected on MEASUREMENT, not taste:
#   * `Base.JLOptions().check_bounds == 1` — what stage 0 shipped with, now gone. `Pkg.test()` hardcodes
#     that flag on 1.12, so the fence armed for all 26k tests: 13.4 GB peak RSS and a hard SIGSEGV.
#   * a Preference (an `arena_fence` key defaulting to `false`) — right about one thing the others are not:
#     it is the only candidate whose value the compiler knows about, because Preferences ARE
#     part of the precompile cache key, so flipping the value recompiles correctly. But it is STICKY rather
#     than per-run, and a `LocalPreferences.toml` in the package root DOES reach `Pkg.test()`'s sandbox
#     (measured: a sandboxed test read `flag = true` written only in the package root). So once armed, the
#     next plain `Pkg.test()` reproduces the SIGSEGV above — the exact failure being fixed. It also costs a
#     full recompile per flip.
#   * `get(ENV, "PUREBLAS_ARENA_FENCE", "0") == "1"` folded into a const — per-run, but ENV is NOT in the
#     precompile cache key. Measured, not assumed: a module precompiled with the variable unset and then
#     loaded WITH it set still reported `false`. A sanitizer that silently fails to arm is worse than none.
# Dispatch has none of those failure modes — nothing to leave set, nothing to recompile, nothing to go
# stale — and the default path keeps no branch at all, not even a folded one, because `ArenaScope` and
# `FencedScope` are different types. What it cannot do is arm an already-written `@scope` site. When
# stage-N conversion makes a whole-library sanitizer run worth having, that is one line (`_arena_enter!`
# returning a `FencedScope` under a preference) — and THEN Preferences is the right gate, because by then
# the point is precisely to arm code you are not editing.
#
# It is NOT gated on StrictMode either: StrictMode verifies properties of GENERATED CODE (vectorized?
# spilled? allocating? trim-safe?), whereas fencing released memory is SANITIZER behaviour, and overloading
# `checks_enabled()` with that meaning would muddy both.

mutable struct Arena
    slabs::Vector{Vector{UInt8}}   # slab k+1 is used only once slab k overflowed; kept live until depth 0
    cur::Int                       # index of the slab currently being bumped
    off::Int                       # bytes consumed in slabs[cur]
    depth::Int                     # scope nesting; coalesce to one slab when it returns to 0
    carried::Int                   # bytes already consumed in slabs 1..cur-1 of THIS scope chain
    high::Int                      # peak bytes ever actually demanded, in one contiguous run
end
Arena() = Arena([UInt8[]], 1, 0, 0, 0, 0)

# The scope token. Only `@scope` constructs one, and the macro forbids it from being stored or passed
# anywhere except as `borrow!`'s first argument — that is what makes the escape hazard a compile error.
struct ArenaScope
    a::Arena
    cur::Int
    off::Int
end

# GKH ownership: one const owner, resolved at compile time, no runtime lookup.
# Multithreading is deferred by standing rule, and the per-task swap is deliberately NOT taken here: it is
# one line later (`_arena() = _ARENA_TASK()` over a `Base.OncePerTask`), but it measures 9-12 ns per entry,
# and workspace.jl:58-61 records +7.5 ns being REJECTED as 7.4% of trmm! at n=8. Buying thread safety we
# cannot yet use, at a price the gate already refused, is the wrong trade today.
# BUT STATE THE COST HONESTLY: `_l3ws` is process-global too, so this is not a regression in KIND — it is
# one in DEGREE. Interleave two tasks and the arena's failure is unbounded where the struct's was bounded:
# B enters (off=0), A enters (off=0), A borrows 0→128, B borrows 128→256, B EXITS and rewinds off to 0,
# A borrows again and is handed bytes 0..128 — aliasing its own live handle. With a named field the same
# race corrupts that one role; here it is arbitrary cross-role aliasing. The per-task owner is therefore a
# PRECONDITION of enabling threads, not an optimisation to weigh against 9-12 ns.
const _ARENA = Arena()
@inline _arena() = _ARENA

@inline function _arena_enter!(a::Arena)
    a.depth += 1
    return ArenaScope(a, a.cur, a.off)
end

@inline function _arena_exit!(s::ArenaScope)
    a = s.a
    a.cur = s.cur
    a.off = s.off
    a.depth -= 1
    # Cold, and only after a call that grew: fold the slab chain into ONE slab of the peak size, so the
    # fresh-slab page offset is TRANSIENT — it exists on the growing call only, and from the next call the
    # layout is contiguous and deterministic per call path. That matters because this library is
    # address-sensitive (odd `ld` for conflict-free re-reads, the +8 pads, the po2-lda finding), so a
    # layout that shifted per call would make gate cells unreproducible.
    # The `Vector{UInt8}` here is why a call path allocates ONCE at each new arena high-water (measured:
    # 1456 B on the first `trexc!` n=16, 0 B on every call after). `L3Workspace` paid that at module load
    # instead, so a converted routine is no longer 0-allocation on its very first call at a new peak — it
    # is 0-allocation from the second. Nothing in the suite asserts on the first call; if a routine ever
    # needs first-call-zero-alloc, pre-touch the peak rather than removing the coalesce.
    #
    # COALESCE TO THE HIGH-WATER *DEMAND*, NOT TO THE SUM OF SLAB CAPACITIES. Summing capacities compounds
    # `_arena_grow!`'s doubling into the permanent size, so the arena balloons geometrically on a sequence
    # of small overflows instead of converging on the peak: with a slab of S, a call needing S+1 grows to
    # 2S, capacities sum to 3S, and the NEXT one-byte overflow of 3S grows to 6S and coalesces to 9S — 9x
    # the memory ever asked for, and it keeps multiplying. `a.high` records what was actually demanded
    # (`carried + off + need` at each grow, which is exactly the contiguous run the doubling failed to
    # supply), so the fold lands on that. Never shrink below the slab already held: this is a high-water
    # allocator, and re-growing a slab we just released would put an allocation back on a hot path.
    if a.depth == 0
        a.carried = 0
        if length(a.slabs) > 1
            want = max(a.high, length(a.slabs[1]))
            resize!(a.slabs, 1)
            a.slabs[1] = Vector{UInt8}(undef, want)
            a.cur = 1
            a.off = 0
        end
    end
    return nothing
end

# Cold path: move to the next slab. The current slab is KEPT, so every handle already handed out in this
# scope chain stays valid — growth never invalidates a live borrow.
@noinline function _arena_grow!(a::Arena, need::Int)
    # Record the true demand BEFORE switching slabs. `carried + off` is what this scope chain has already
    # consumed, `need` is what would not fit; their sum is the contiguous run the next coalesce must
    # provide. This is the only place the high-water is updated, and it is the cold path — the hot
    # `borrow!` bump stays a pointer add with no max() in it.
    a.carried += a.off
    a.high = max(a.high, a.carried + need)
    a.cur += 1
    if a.cur > length(a.slabs) || length(a.slabs[a.cur]) < need
        a.cur > length(a.slabs) && push!(a.slabs, UInt8[])
        prev = length(a.slabs[a.cur - 1])
        a.slabs[a.cur] = Vector{UInt8}(undef, max(need, 2 * prev))
    end
    a.off = 0
    return nothing
end

# ── Leading-dimension helpers ───────────────────────────────────────────────────────────────────────
# Lifted VERBATIM from the code they replace, with the source cited, so the stride reasoning is not
# reinvented. Under the arena these become properties of the CALL rather than of call history — today
# `_trsm_tmp`/`rpack`/`rrefl`/`padf` return the whole GROWN matrix, so their `ld` depends on what an
# earlier call asked for. That is the hazard `_gemm_3m_scratch` documents as a 1.16→0.57 regression
# (workspace.jl:654-662): the gate sweep hides it because sizes ascend, but a caller going large-then-small
# gets a different library. An exact borrow cannot have that bug.

# An odd ld can never be a multiple of the (power-of-2) way stride, so successive columns cannot collide
# in one cache set. Source: workspace.jl:507 (`_trsm_rpack`), same rule at `_trsm_rrefl`/`_trsm_rpad`.
@inline _odd_ld(m::Int) = (ld = m + 8; iseven(ld) ? ld + 1 : ld)

# n+8, bumped a further +8 if that ld would ITSELF land on the L1 quarter-way stride. Source:
# workspace.jl:551 (`_potrf_pad`), same rule at `_sytrf_work`. `_L1_WAY_BYTES` is the detected
# way size from cpuinfo.jl:113 (`max(_CACHELINE, _L1_BYTES ÷ _L1D_ASSOC)`) — a derived const, not a literal.
const _ARENA_WAY_QUARTER = _L1_WAY_BYTES >> 2
@inline function _offway_ld(m::Int, ::Type{T}) where {T}
    ld = m + 8
    return (ld * sizeof(T)) % _ARENA_WAY_QUARTER == 0 ? ld + 8 : ld
end

# Error paths are @noinline with NON-interpolated messages: eager multi-piece interpolation lowers to
# `print_to_string(::String, ::Vararg{Any})`, which `juliac --trim=safe` rejects (req#4).
@noinline _throw_arena_ld() = throw(ArgumentError("borrow!: leading dimension is smaller than the row count"))
@noinline _throw_arena_dims() = throw(ArgumentError("borrow!: dimensions must be non-negative"))

"""
Borrow an `m`×`n` column-major block with leading dimension `ld` (default `m`, i.e. exact). Valid until
the enclosing `@scope` block exits. Bytes consumed are `ld*n*sizeof(T)`, aligned to `_ARENA_ALIGN` for
every `T` — that is `max(_SIMD_BYTES, _CACHELINE)`, DERIVED from the detected hardware (req#8), not the
64 an earlier draft of this docstring named. On a 32-byte-vector AVX2 part with a 64-byte line it is 64;
on a part with a wider line or a wider vector it is wider, and the code must not assume either number.
"""
@inline function borrow!(s::ArenaScope, ::Type{T}, m::Int, n::Int, ld::Int = m) where {T}
    # Non-isbits element types (BigFloat is the live case — it is a handle onto mpfr limbs) cannot be
    # pointer-backed, so they fall back to the heap. Correct, just not zero-alloc; no gated path uses them.
    isbitstype(T) || return Matrix{T}(undef, m, n)
    (m >= 0 && n >= 0) || _throw_arena_dims()
    ld >= m || _throw_arena_ld()
    bytes = ld * n * sizeof(T)
    a = s.a
    slab = a.slabs[a.cur]
    base = UInt(pointer(slab)) + a.off
    adj = Int((-base) & (_ARENA_ALIGN - 1))
    if a.off + adj + bytes > length(slab)
        _arena_grow!(a, adj + bytes + _ARENA_ALIGN)
        slab = a.slabs[a.cur]
        base = UInt(pointer(slab))
        adj = Int((-base) & (_ARENA_ALIGN - 1))
    end
    a.off += adj + bytes
    return PtrMatrix{T}(Ptr{T}(base + adj), m, n, ld)
end

"""
Borrow a length-`n` vector. Covers the `Vector{Int}` roles (pivots, `iblock`, `isplit`, `jpvt`) as well as
the element-typed ones.
"""
@inline function borrow!(s::ArenaScope, ::Type{T}, n::Int) where {T}
    isbitstype(T) || return Vector{T}(undef, n)
    M = borrow!(s, T, n, 1, n)
    return PtrVector{T}(pointer(M), n)
end

# ── Electric fence ──────────────────────────────────────────────────────────────────────────────────
# Inside a `@fenced_scope` every borrow is its own mmap, right-aligned against a PROT_NONE guard page.
# Scope exit mprotects the whole mapping and NEVER reuses the address, so any later access through a stale
# handle — read OR write, getindex OR raw pointer, driver OR kernel — faults at the offending line and
# Julia surfaces it as `ReadOnlyMemoryError` with a backtrace. An overrun past the borrow hits the guard.
#
# This replaces value-poisoning, which was the first design and is strictly weaker: poison catches READS
# only, and the two shipped self-alias bugs (`trsm_tmp`, `rpack`) were WRITES. It also disposes of the
# per-element-type poison question — `T(NaN)` is not constructible for `Int` or `Rational`, and for a
# `ForwardDiff.Dual` it would set the value to NaN while leaving the partials at zero, i.e. a derivative
# that looks computed.
#
# WHAT IT COSTS. RSS is fine — MADV_DONTNEED returns the pages, so a fenced sweep holds flat (measured
# 2026-09-04: 580k borrows, RSS unchanged at 580 MB). The MAPPINGS used not to be: each released borrow
# left VMAs behind (~0.069 permanently retained per borrow at the 29-borrows-per-scope shape
# `_dtgex2_big!` has — 580k borrows took the map count 334 → 40389), against a `vm.max_map_count` of
# 1048576 on wintermute but 65530 on a stock Linux box, past which `mmap` fails and `_throw_arena_mmap`
# ends the run INSIDE THE SANITIZER rather than in the code under test. `_syl_dlasy2` alone is ~4k calls ×
# 4 borrows at n=64, so a fenced full-suite run was not obviously under the stock limit and the
# "fence everything" hatch was hollow. `_ARENA_FENCE_QUARANTINE` (below) fixes that: releases are held
# poisoned in a bounded FIFO and munmapped once they age out, so the footprint is capped. The fence stays
# per-scope opt-in all the same — it is ~1000× the cost of a bump, so it is a debugging tool, not a mode.
const _ARENA_PROT_NONE = Cint(0)
const _ARENA_PROT_RW = Cint(3)                # PROT_READ | PROT_WRITE
const _ARENA_MAP_PRIVATE_ANON = Cint(0x22)    # MAP_PRIVATE | MAP_ANONYMOUS (Linux)
const _ARENA_PAGE = Sys.PAGESIZE   # an OS fact, QUERIED — mprotect works in pages, so this must be the real one
const _ARENA_FENCE_PTR = Ptr{UInt8}[]
const _ARENA_FENCE_LEN = Int[]

@noinline _throw_arena_mmap() = error("arena electric fence: mmap failed")
# mprotect's return is CHECKED, in both directions. A failed mprotect leaves the guard page readable (no
# overrun trap) or the released mapping writable (a use-after-release reports "no fault") — i.e. the
# sanitizer silently stops sanitizing and its green result becomes a lie, which is worse than not running
# it. Measured 2026-09-04: mprotect on a misaligned address returns -1 and the page stays readable.
@noinline _throw_arena_mprotect() = error("arena electric fence: mprotect failed — the fence is NOT armed")

function _arena_efence_alloc(::Type{T}, bytes::Int) where {T}
    total = cld(max(bytes, 1), _ARENA_PAGE) * _ARENA_PAGE + _ARENA_PAGE   # data pages + one guard page
    p = ccall(
        :mmap, Ptr{UInt8}, (Ptr{Cvoid}, Csize_t, Cint, Cint, Cint, Int),
        C_NULL, total, _ARENA_PROT_RW, _ARENA_MAP_PRIVATE_ANON, -1, 0
    )
    p == Ptr{UInt8}(-1) && _throw_arena_mmap()
    rc = ccall(:mprotect, Cint, (Ptr{UInt8}, Csize_t, Cint), p + total - _ARENA_PAGE, _ARENA_PAGE, _ARENA_PROT_NONE)
    rc == Cint(0) || _throw_arena_mprotect()
    push!(_ARENA_FENCE_PTR, p)
    push!(_ARENA_FENCE_LEN, total)
    # Right-align the block against the guard page so an overrun faults immediately, and keep the returned
    # pointer 64-byte aligned (the page base is, and the rounded size preserves it).
    off = total - _ARENA_PAGE - cld(bytes, _ARENA_ALIGN) * _ARENA_ALIGN
    return Ptr{T}(p + off)
end

const _ARENA_MADV_DONTNEED = Cint(4)   # Linux MADV_DONTNEED

# QUARANTINE, rather than poison-forever. `mprotect`-and-keep is the strongest possible check — the
# address can never be handed out again, so a stale handle faults no matter how much later it is used —
# but it retains one mapping per released borrow (~0.069 VMAs permanently per borrow at the shape
# `_dtgex2_big!` has; 580k borrows took the map count 334 → 40389). The ceiling is `vm.max_map_count`,
# and past it `mmap` fails, so the run dies inside the sanitizer instead of inside the code under test.
# That made the "fence the whole suite" hatch above hollow. So: keep the most RECENT releases poisoned
# and `munmap` the ones that age out. A use-after-release is still caught for that many subsequent
# borrows, which covers the hazard this exists to find — a handle used just after its scope exits — and
# the fence's address-space footprint becomes bounded instead of unbounded.
# TIER: a sanitizer budget, not a machine-dependent tuning constant; the PDM ladder does not apply (no
# generated-code property depends on it). The value is set by the SMALLEST default `vm.max_map_count`
# in the wild, 65530, leaving an order of magnitude for the rest of the process's mappings.
# A sanitizer's retention depth, and no property of the host makes one depth better than another — there
# is no residency or latency criterion to derive it from and no candidate set to measure. It is bounded
# from above by the smallest stock `vm.max_map_count` (65530), and this path never runs in a gated
# measurement or in the trimmed `.so`.
# req8-ok: sanitizer retention depth, not a machine-dependent tuning knob — bounded by the OS map limit.
# PDM: Exempt — a debug sanitizer's retention depth, not hardware tuning; no host property favours one value.
const _ARENA_FENCE_QUARANTINE = 4096
const _ARENA_QUAR_PTR = Ptr{UInt8}[]
const _ARENA_QUAR_LEN = Int[]

function _arena_efence_release!(nlive::Int)
    while length(_ARENA_FENCE_PTR) > nlive
        p = pop!(_ARENA_FENCE_PTR)
        n = pop!(_ARENA_FENCE_LEN)
        # mprotect, never munmap: the address must never be handed out again, or a stale handle could
        # silently land on a LIVE mapping and read plausible data instead of faulting.
        # Checked, and it THROWS — out of a `finally`, deliberately. If this fails the released block is
        # still writable, so the fence would go on to report "no fault" for a real use-after-release; a
        # sanitizer that lies is worse than one that stops. It cannot fire on a well-formed mapping (`p`
        # is mmap's page-aligned return and `n` a page multiple), so firing means the fence itself is broken.
        ccall(:mprotect, Cint, (Ptr{UInt8}, Csize_t, Cint), p, n, _ARENA_PROT_NONE) == Cint(0) ||
            _throw_arena_mprotect()
        # ...but mprotect does NOT return the physical pages, so without this every borrow would retain
        # its RSS for the life of the process. A loop of 20k scoped borrows (test/arena_tests.jl has
        # exactly that) would hold ~240 MB it can never use again. MADV_DONTNEED drops the pages while
        # LEAVING the mapping reserved, so the address stays poisoned and the memory comes back.
        ccall(:madvise, Cint, (Ptr{UInt8}, Csize_t, Cint), p, n, _ARENA_MADV_DONTNEED)
        # …and hold the poisoned mapping in a bounded FIFO. Only once it ages out of the quarantine is the
        # address space actually returned; until then the guarantee above is unchanged.
        push!(_ARENA_QUAR_PTR, p); push!(_ARENA_QUAR_LEN, n)
        while length(_ARENA_QUAR_PTR) > _ARENA_FENCE_QUARANTINE
            ccall(:munmap, Cint, (Ptr{UInt8}, Csize_t), popfirst!(_ARENA_QUAR_PTR), popfirst!(_ARENA_QUAR_LEN))
        end
    end
    return nothing
end

# The fenced counterpart of `ArenaScope`, and the whole of the gate: a scope opened as this type borrows
# from mmap, one mapping per borrow. It holds no `Arena` because it never bumps one — which is also why the
# fence can be opened anywhere without perturbing the slab layout the gate cells depend on.
struct FencedScope
    nlive::Int
end

@inline _arena_enter_fenced!() = FencedScope(length(_ARENA_FENCE_PTR))
@inline _arena_exit!(s::FencedScope) = _arena_efence_release!(s.nlive)

# Deliberately duplicated from the `ArenaScope` methods rather than shared through a Union or an abstract
# supertype: the default path must stay exactly the code it is today, and two three-line bodies are a
# smaller price than a signature change on the hot one.
function borrow!(s::FencedScope, ::Type{T}, m::Int, n::Int, ld::Int = m) where {T}
    isbitstype(T) || return Matrix{T}(undef, m, n)
    (m >= 0 && n >= 0) || _throw_arena_dims()
    ld >= m || _throw_arena_ld()
    return PtrMatrix{T}(_arena_efence_alloc(T, ld * n * sizeof(T)), m, n, ld)
end

function borrow!(s::FencedScope, ::Type{T}, n::Int) where {T}
    isbitstype(T) || return Vector{T}(undef, n)
    M = borrow!(s, T, n, 1, n)
    return PtrVector{T}(pointer(M), n)
end

# ── @scope ──────────────────────────────────────────────────────────────────────────────────────────
# Does the token appear anywhere OTHER than as borrow!'s first argument? If so it can be stored, returned
# or captured, and a borrow could outlive the scope.
# `borrow!` may be written unqualified inside PureBLAS or QUALIFIED from outside (`PureBLAS.borrow!`,
# `P.borrow!`), in which case the callee is a `.`-expression rather than a bare Symbol. Recognising only
# the bare form made the macro reject every qualified call site as an escaping token — caught by
# test/arena_tests.jl on its first run.
function _arena_is_borrow(f)
    f === :borrow! && return true
    f isa GlobalRef && return f.name === :borrow!
    f isa Expr || return false
    f.head === :. && length(f.args) == 2 && return f.args[2] === QuoteNode(:borrow!)
    return false
end

# Is this expression a legal borrow site for token `s`, i.e. `<borrow!>(s, …)`?
function _arena_borrow_call(ex, s::Symbol)
    return ex isa Expr && ex.head === :call && !isempty(ex.args) &&
        _arena_is_borrow(ex.args[1]) && length(ex.args) >= 2 && ex.args[2] === s
end

# A nested `@scope` carries its own token and gets its own checks when IT expands. Match on the last
# component so `PureBLAS.@scope` is recognised as well as a bare `@scope`.
function _arena_is_scope_macro(f)
    f === Symbol("@scope") && return true
    f isa GlobalRef && return f.name === Symbol("@scope")
    f isa Expr && f.head === :. && length(f.args) == 2 && return f.args[2] === QuoteNode(Symbol("@scope"))
    return false
end

function _arena_uses_token(ex, s::Symbol)
    ex === s && return true
    ex isa Expr || return false
    if _arena_borrow_call(ex, s)
        return any(a -> _arena_uses_token(a, s), ex.args[3:end])   # the token in slot 1 is the legal use
    end
    return any(a -> _arena_uses_token(a, s), ex.args)
end

function _arena_borrows(ex, s::Symbol)
    ex isa Expr || return false
    _arena_borrow_call(ex, s) && return true
    ex.head === :macrocall && !isempty(ex.args) && _arena_is_scope_macro(ex.args[1]) && return false
    return any(a -> _arena_borrows(a, s), ex.args)
end

# Every construct that can execute its body more than once, or later. `:for`/`:while` are the loops;
# comprehensions and generators are loops in disguise; `:->`, `:function` and `:do` defer execution and
# can also escape the borrow by capture.
const _ARENA_REPEATING = (
    :for, :while, :comprehension, :typed_comprehension, :generator,
    :flatten, :(->), :function, :do,
)

@noinline function _throw_arena_loop(s::Symbol)
    error(
        "@scope: borrow!(" * String(s) * ", …) inside a loop, comprehension or closure. Each execution " *
            "would consume more arena. Hoist the borrow above it (the shapes are usually loop-invariant), or " *
            "let the loop body call a helper that opens its own @scope — entry is free."
    )
end
@noinline function _throw_arena_escape(s::Symbol)
    error(
        "@scope: the token `" * String(s) * "` escapes. It may appear only as borrow!'s FIRST argument, " *
            "so that no borrow can outlive the scope."
    )
end

# ── Handle escape, the hazard the token check does NOT cover ────────────────────────────────────────
# The two checks above track the TOKEN, which is what makes a borrow impossible to take outside the block.
# They say nothing about the HANDLE a borrow returns, and a handle that outlives the scope is a pointer
# into rewound (and, at the next call, overwritten) arena bytes — a use-after-free that reads plausible
# numbers rather than faulting. The lexical cases of that are cheap to catch here, so catch them rather
# than leaving the whole hazard to `@fenced_scope` at runtime and to reading:
#   * `return A` (or `return (A, k)`, `return [A]`) where `A` was bound by a `borrow!` in this scope;
#   * the block's own TAIL VALUE being such a name — `@scope` expands to a `try`, whose value is the
#     body's, so a trailing bare `A` hands the handle straight to the caller.
# Deliberately NARROW: only a handle in VALUE POSITION is rejected. `return sum(A)` is fine and common,
# and so is passing `A` down into a callee — handles are meant to be passed, only not to outlive. What is
# still not caught, and still needs the fence and a reading audit: storage into a field or global, and
# capture by a closure that outlives the block.
function _arena_borrowed_names(ex, s::Symbol, acc::Vector{Symbol} = Symbol[])
    if ex isa Expr
        if ex.head === :(=) && ex.args[1] isa Symbol && _arena_borrows(ex.args[2], s)
            push!(acc, ex.args[1]::Symbol)
        end
        # do not descend into a nested `@scope`: its borrows are its own
        if !(ex.head === :macrocall && !isempty(ex.args) && _arena_is_scope_macro(ex.args[1]))
            for a in ex.args
                _arena_borrowed_names(a, s, acc)
            end
        end
    end
    return acc
end

_arena_in_value_position(ex, names) =
    ex isa Symbol ? (ex in names) :
    (ex isa Expr && ex.head in (:tuple, :vect) && any(a -> _arena_in_value_position(a, names), ex.args))

@noinline function _throw_arena_handle(n::Symbol)
    error(
        "@scope: the borrowed handle `" * String(n) * "` is returned from the scope. It points into arena " *
            "bytes that the scope rewinds on exit, so the caller would read scratch that the next call " *
            "overwrites. Copy what you need into a caller-owned array before the block ends, or move the " *
            "@scope up to the caller and pass the handle DOWN."
    )
end

function _arena_check_handles(ex, names::Vector{Symbol})
    isempty(names) && return nothing
    ex isa Expr || return nothing
    if ex.head === :return && length(ex.args) == 1 && _arena_in_value_position(ex.args[1], names)
        for n in names
            _arena_in_value_position(ex.args[1], [n]) && _throw_arena_handle(n)
        end
    end
    for a in ex.args
        _arena_check_handles(a, names)
    end
    return nothing
end

# The tail value of the block, skipping line-number nodes and trailing comments.
function _arena_tail(ex)
    (ex isa Expr && ex.head === :block) || return ex
    for i in length(ex.args):-1:1
        ex.args[i] isa LineNumberNode || return _arena_tail(ex.args[i])
    end
    return nothing
end

function _arena_check(ex, s::Symbol)
    ex isa Expr || return nothing
    if ex.head in _ARENA_REPEATING
        for a in ex.args
            _arena_borrows(a, s) && _throw_arena_loop(s)
        end
    end
    for a in ex.args
        _arena_check(a, s)
    end
    return nothing
end

"""
    @scope s begin … end

Open an arena scope bound to token `s`. Inside, `borrow!(s, T, m, n[, ld])` hands out scratch valid until
the block exits, at which point the bump pointer rewinds. Three rules are enforced when this macro EXPANDS:
the TOKEN may appear only as `borrow!`'s first argument; no `borrow!` may sit LEXICALLY inside a loop,
comprehension or closure in this block; and no borrowed HANDLE may be `return`ed from the block or be its
tail value.

Those are lexical checks on the unexpanded body, not a proof. What they still do not catch — see the
header comment for the measurements: a handle STORED into a field or global or captured by a closure that
outlives the block, a loop emitted by another macro, and recursion. `@fenced_scope` is the runtime check
for the handle cases.
"""
macro scope(s::Symbol, body)
    _arena_uses_token(body, s) && _throw_arena_escape(s)
    _arena_check(body, s)
    let names = _arena_borrowed_names(body, s), t = _arena_tail(body)
        _arena_check_handles(body, names)
        for n in names
            _arena_in_value_position(t, [n]) && _throw_arena_handle(n)
        end
    end
    return quote
        $(esc(s)) = _arena_enter!(_arena())
        try
            $(esc(body))
        finally
            _arena_exit!($(esc(s)))
        end
    end
end

"""
    @fenced_scope s begin … end

`@scope`, but every borrow is its own mmap behind a PROT_NONE guard page and scope exit poisons the
address for the life of the process — so a use of a released handle FAULTS instead of silently reading
whatever the next borrow wrote. Same two expansion-time rules as `@scope`.

This is the opt-in: there is no global fence mode, so a routine is fenced exactly when its own scope is
written with this macro, and a plain `Pkg.test()` cannot arm anything. It is a debugging tool — one mmap
plus two syscalls per borrow, and the address space is never reclaimed — so it belongs in a dedicated run,
and the fault it raises is not reliably catchable in-process (see `test/arena_tests.jl`, which drives it
from a subprocess for exactly that reason).
"""
macro fenced_scope(s::Symbol, body)
    _arena_uses_token(body, s) && _throw_arena_escape(s)
    _arena_check(body, s)
    return quote
        $(esc(s)) = _arena_enter_fenced!()
        try
            $(esc(body))
        finally
            _arena_exit!($(esc(s)))
        end
    end
end
