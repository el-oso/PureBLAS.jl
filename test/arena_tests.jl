# Scratch arena (src/arena.jl). STAGE 0: nothing in src/ borrows yet, so these tests ARE the only thing
# holding the arena's invariants — there is no numerical test that would notice a regression here.
#
# What each case exists to catch, since "it passes" is not evidence on its own:
#   * alignment/overlap — the prototype assumed 8-byte elements (`want = r*c + 8`), which at Float32 runs
#     up to 7 elements past the reservation and OVERLAPS the next borrow. Found by review, not by a test.
#   * growth — a slab overflow must not invalidate a handle already handed out. If it did, the failure is
#     a wrong number in a routine that borrowed before a big one, i.e. silent and shape-dependent.
#   * the macro's two rejections — these are the whole reason the escape and loop hazards are compile
#     errors rather than lint findings. If they can be evaded, the design loses its advantage over the
#     180-named-field struct it replaces, so they are tested by asserting the expansion THROWS.
#   * write-after-release — covered by the last testitem, which is OPT-IN (`PUREBLAS_ARENA_FENCE=1`) and
#     runs in a SUBPROCESS. Both of those are load-bearing; see the comment there.

@testitem "arena: alignment, packing and ld across every element size" begin
    using PureBLAS
    const P = PureBLAS

    # WARM FIRST. The arena starts with an empty slab, so the first borrows grow it — and a growth between
    # two borrows moves the second to a different slab, which mmap may place at a LOWER address. The
    # tightness assertion below is only meaningful within one slab, so take the growth here, before
    # measuring. (Coalescing at depth 0 leaves one slab of the peak size, so nothing after this grows.)
    warm(n) = (
        P.@scope s begin
            P.borrow!(s, Float64, n, n); nothing
        end
    )
    warm(1024)

    # 64-byte alignment for EVERY T, not just 8-byte ones, and no overlap with the next borrow.
    for T in (Float16, Float32, Float64, ComplexF32, ComplexF64, Int)
        P.@scope s begin
            A = P.borrow!(s, T, 33, 3)
            B = P.borrow!(s, T, 5, 5)
            @test UInt(pointer(A)) % 64 == 0
            @test UInt(pointer(B)) % 64 == 0
            # B must start at or after A's end — the overlap bug the prototype had at Float32.
            @test UInt(pointer(B)) >= UInt(pointer(A)) + 33 * 3 * sizeof(T)
            # ...and no more than one alignment gap past it, or the arena is wasting space.
            @test UInt(pointer(B)) < UInt(pointer(A)) + 33 * 3 * sizeof(T) + 64
        end
    end

    # ANTI-ALIASING BETWEEN SUCCESSIVE BORROWS. A borrow whose byte size is a multiple of the way stride
    # would otherwise put the NEXT borrow in the same scope on identical cache sets — the arena's own
    # version of the po2-lda hazard this library fights with `_odd_ld`/`_offway_ld`/the +8 pads, except
    # ACROSS buffers rather than within one. The live case is `diag`, a FIXED `_L3_NB × _L3_NB` block:
    # at `_L3_NB == 128` that is exactly 2^17 bytes for Float64 and 2^18 for ComplexF64, so every gated
    # trsm/trmm base that borrows `diag` and then a gemm operand would have aliased them.
    let q = P._ARENA_WAY_QUARTER
        P.@scope s begin
            # a borrow sized to an exact multiple of the quarter-way stride…
            A = P.borrow!(s, UInt8, 4 * q, 1)
            B = P.borrow!(s, UInt8, 64, 1)
            d = UInt(pointer(B)) - UInt(pointer(A))
            @test d > 4 * q                          # …must NOT leave B exactly 4q after A
            @test d % q != 0                         # …nor on the same set by any multiple
            @test d <= 4 * q + 2 * P._ARENA_ALIGN    # …and the nudge is one alignment unit, not a gap
        end
        # the nudge must NOT fire on a size that is already off the stride — no wasted bytes
        P.@scope s begin
            A = P.borrow!(s, UInt8, 4 * q + 8, 1)
            B = P.borrow!(s, UInt8, 64, 1)
            @test UInt(pointer(B)) - UInt(pointer(A)) <= 4 * q + 8 + P._ARENA_ALIGN
        end
    end

    # Non-overlap over many shapes and strides. Stated as "the byte RANGES do not intersect" rather than
    # "B starts after A ends": if a growth intervenes the two live in different slabs and their relative
    # order is whatever mmap chose, but they must still never share a byte.
    function pair_ok(::Type{T}, m, n, ld) where {T}
        P.@scope s begin
            A = P.borrow!(s, T, m, n, ld)
            B = P.borrow!(s, T, m, n, ld)
            sz = ld * n * sizeof(T)
            a0 = UInt(pointer(A)); a1 = a0 + sz
            b0 = UInt(pointer(B)); b1 = b0 + sz
            disjoint = (b0 >= a1) || (a0 >= b1)
            return disjoint && a0 % 64 == 0 && b0 % 64 == 0
        end
    end
    for T in (Float32, Float64, ComplexF64), m in (1, 3, 8, 33), n in (1, 2, 7)
        @test pair_ok(T, m, n, m)
        @test pair_ok(T, m, n, P._odd_ld(m))
    end

    # ld is honoured exactly, including an odd one, and indexing lands where the stride says.
    P.@scope s begin
        R = P.borrow!(s, Float64, 100, 7, 109)
        @test R.ld == 109
        @test UInt(pointer(R)) % 64 == 0
        R[100, 7] = 3.0
        @test unsafe_load(pointer(R), 6 * 109 + 100) == 3.0
    end

    # The two ld helpers keep the properties their source comments claim (workspace.jl:507 / :551).
    @test isodd(P._odd_ld(64)) && isodd(P._odd_ld(65))
    @test P._odd_ld(64) >= 64 && P._offway_ld(64, Float64) >= 64
    @test (P._offway_ld(64, Float64) * sizeof(Float64)) % P._ARENA_WAY_QUARTER != 0
end

@testitem "arena: growth keeps live borrows valid, and coalesces at depth 0" begin
    using PureBLAS
    const P = PureBLAS

    # A borrow, then a borrow big enough to overflow the slab, then a write through the FIRST handle.
    # If growth invalidated it, this reads garbage rather than 1.0.
    P.@scope s begin
        A = P.borrow!(s, Float64, 64, 64)
        fill!(A, 1.0)
        B = P.borrow!(s, Float64, 2048, 2048)
        @test all(==(1.0), A)
        B[1, 1] = 2.0
        @test A[1, 1] == 1.0          # B must not have landed on top of A
    end

    # Back at depth 0 the slab chain is folded into one, so the layout is deterministic per call path from
    # the next call onward (this library is address-sensitive; a per-call-varying layout would make gate
    # cells unreproducible). Only meaningful when the real bump path ran, not under the fence.
    begin   # `@scope` is always the bump path — the fence is a different scope type, not a mode
        a = P._arena()
        @test length(a.slabs) == 1
        @test a.off == 0
        @test a.depth == 0
    end

    # The fold lands on the HIGH-WATER DEMAND, not on the sum of slab capacities. Summing capacities
    # compounds `_arena_grow!`'s doubling into the permanent size: slab S, a call needing S+1 grows to 2S
    # and coalesces to 3S; the next one-byte overflow of 3S grows to 6S and coalesces to 9S, and so on
    # without bound. Drive exactly that pattern — a series of borrows each just over the slab held — and
    # assert the resulting slab stays within a small factor of what was actually asked for.
    begin
        a = P._arena()
        for _ in 1:6
            want = length(a.slabs[1]) + 8
            P.@scope s begin
                v = P.borrow!(s, UInt8, want)
                v[1] = 0x01
                v[end] = 0x02
            end
        end
        cap = length(a.slabs[1])
        @test length(a.slabs) == 1
        @test cap >= a.high                       # it must still cover the peak demand
        @test cap <= 4 * a.high                   # …without the geometric blow-up (was 3^k of it)
    end

    # Nested scopes: the inner rewind must not release the outer's borrows.
    P.@scope outer begin
        A = P.borrow!(outer, Float64, 32, 32)
        fill!(A, 7.0)
        P.@scope inner begin
            B = P.borrow!(inner, Float64, 512, 512)
            fill!(B, 9.0)
        end
        @test all(==(7.0), A)
    end

    # depth accounting must survive a throw, or one error leaks the whole arena for the process.
    a = P._arena()
    d0 = a.depth
    @test_throws ErrorException P.@scope s begin
        P.borrow!(s, Float64, 4, 4)
        error("boom")
    end
    @test a.depth == d0
end

@testitem "arena: recursion holds borrows across child calls; scoped helpers stay flat in a loop" begin
    using PureBLAS
    const P = PureBLAS

    # Each level's borrow must remain intact across the recursive call — this is the Strassen/D&C shape,
    # where P1..P7 are live across seven children. Today that is policed by slot-index arithmetic
    # (`3 + level*10` in _str_fit!) plus a comment; here it is structural.
    function rec(depth)
        P.@scope s begin
            Pm = P.borrow!(s, Float64, 8, 8)
            fill!(Pm, Float64(depth))
            depth > 0 && rec(depth - 1)
            @test all(==(Float64(depth)), Pm)
        end
    end
    rec(5)

    # A loop that borrows via a SCOPED HELPER is the sanctioned pattern, and must not accumulate: the
    # helper's scope rewinds every iteration. 20k iterations of an 8 KB borrow.
    function tile!(v)
        P.@scope s begin
            E = P.borrow!(s, Float64, 128, 8)
            E[1, 1] = v
            return E[1, 1]
        end
    end
    # In a function, not at testitem top level: a bare `acc = 0.0` followed by a `for` that reassigns it
    # is soft-scope ambiguous and fails with UndefVarError.
    function drive(n)
        acc = 0.0
        for i in 1:n
            acc += tile!(Float64(i))
        end
        return acc
    end
    @test drive(20_000) > 0
    begin   # `@scope` is always the bump path — the fence is a different scope type, not a mode
        @test P._arena().off < 1 << 16      # flat: no accumulation across iterations
    end
end

@testitem "arena: @scope rejects an escaping token and a borrow inside a loop AT EXPANSION" begin
    using PureBLAS
    const P = PureBLAS

    # These two rejections are the design's load-bearing claim: the hazards are compile errors, not lint
    # findings. Assert the MACRO throws, and that the message names the right problem.
    function expansion_error(ex)
        try
            @eval $ex
            return ""
        catch e
            return sprint(showerror, e)
        end
    end

    # (1) a borrow inside a loop would consume Σ(iterations) of arena
    for loopy in (
            :(
                PureBLAS.@scope s begin
                    for i in 1:2
                        X = PureBLAS.borrow!(s, Float64, 2, 2)
                    end
                end
            ),
            :(
                PureBLAS.@scope s begin
                    while true
                        X = PureBLAS.borrow!(s, Float64, 2, 2)
                    end
                end
            ),
            :(
                PureBLAS.@scope s begin
                    v = [PureBLAS.borrow!(s, Float64, 2, 2) for _ in 1:3]
                end
            ),
            :(
                PureBLAS.@scope s begin
                    f = () -> PureBLAS.borrow!(s, Float64, 2, 2)
                end
            ),
        )
        msg = expansion_error(loopy)
        @test occursin("inside a loop", msg)
    end

    # (2) the token escaping in any position other than borrow!'s first argument
    for leaky in (
            :(
                PureBLAS.@scope s begin
                    g(s)
                end
            ),
            :(
                PureBLAS.@scope s begin
                    t = (s, 1)
                end
            ),
            :(
                PureBLAS.@scope s begin
                    global _leaked_token = s
                end
            ),
            :(
                PureBLAS.@scope s begin
                    return s
                end
            ),
        )
        msg = expansion_error(leaky)
        @test occursin("escapes", msg)
    end

    # (3) a borrowed HANDLE returned from the block. The token checks above say nothing about this — the
    # handle points into bytes the scope rewinds on exit, so the caller reads scratch the next call
    # overwrites. Rejected in VALUE POSITION only: `return A`, a tuple containing it, or the block's tail.
    for escaped in (
            :(
                PureBLAS.@scope s begin
                    A = PureBLAS.borrow!(s, Float64, 4, 4)
                    return A
                end
            ),
            :(
                PureBLAS.@scope s begin
                    A = PureBLAS.borrow!(s, Float64, 4, 4)
                    return (A, 1)
                end
            ),
            :(
                PureBLAS.@scope s begin
                    A = PureBLAS.borrow!(s, Float64, 4, 4)
                    A
                end
            ),
        )
        msg = expansion_error(escaped)
        @test occursin("borrowed handle", msg)
    end

    # …and the NARROWNESS is the point: reading THROUGH a handle and returning the value is the normal
    # way to use one, and must keep expanding.
    function reads_through()
        P.@scope s begin
            A = P.borrow!(s, Float64, 4, 4)
            fill!(A, 3.0)
            return sum(A)
        end
    end
    @test reads_through() == 48.0

    # A legal block must still expand and run.
    P.@scope s begin
        A = P.borrow!(s, Float64, 4, 4)
        A[1, 1] = 1.0
        @test A[1, 1] == 1.0
    end
end

@testitem "arena: vector borrows cover the Int roles, and non-isbits falls back to the heap" begin
    using PureBLAS
    const P = PureBLAS

    # The Vector{Int} roles (pivots, iblock, isplit, jpvt) go through the vector borrow.
    P.@scope s begin
        piv = P.borrow!(s, Int, 16)
        @test length(piv) == 16
        @test UInt(pointer(piv)) % 64 == 0
        for i in 1:16
            piv[i] = i
        end
        @test piv[16] == 16
    end

    # BigFloat is a handle onto mpfr limbs, not isbits, so it cannot be pointer-backed. It must fall back
    # to a real heap array rather than returning a null handle.
    P.@scope s begin
        M = P.borrow!(s, BigFloat, 4, 4)
        @test M isa Matrix{BigFloat}
        @test size(M) == (4, 4)
        M[1, 1] = big"1.5"
        @test M[1, 1] == big"1.5"
        v = P.borrow!(s, BigFloat, 3)
        @test v isa Vector{BigFloat}
        @test length(v) == 3
    end
end

@testitem "arena: ForwardDiff.Dual borrows like any isbits element type" begin
    using PureBLAS, ForwardDiff
    const P = PureBLAS
    D = ForwardDiff.Dual{Nothing, Float64, 2}

    # Mode 2's AD path must be able to use arena scratch. Dual is isbits, so it is pointer-backed like any
    # other element type — and both the value and the partials round-trip.
    @test isbitstype(D)
    P.@scope s begin
        A = P.borrow!(s, D, 8, 8)
        @test UInt(pointer(A)) % 64 == 0
        x = D(1.5, ForwardDiff.Partials((2.0, 3.0)))
        A[3, 4] = x
        y = A[3, 4]
        @test ForwardDiff.value(y) == 1.5
        @test ForwardDiff.partials(y) == ForwardDiff.Partials((2.0, 3.0))
    end
end

@testitem "arena: electric fence faults on a released borrow (opt-in, subprocess)" begin
    # TWO things here are deliberate and neither is squeamishness.
    #
    # OPT-IN. Set `PUREBLAS_ARENA_FENCE=1` to run this. The fence costs an mmap plus two syscalls per
    # borrow and never reclaims address space; the version of this test that ran unconditionally measured
    # 13.4 GB peak RSS and killed the box. The ENV read is HERE, in test code evaluated at run time, and
    # not folded into a const in `src/` — a const would be baked into the .ji, which does not key on ENV,
    # so the flag would silently do nothing (measured 2026-09-03).
    #
    # SUBPROCESS. A store to a PROT_NONE page is a SIGSEGV that Julia converts to a catchable
    # ReadOnlyMemoryError only sometimes; in-process it took the whole test worker down. So the fault is
    # observed as "the child died and never printed NOFAULT", which is true under either outcome.
    if get(ENV, "PUREBLAS_ARENA_FENCE", "0") != "1"
        @test_skip "PUREBLAS_ARENA_FENCE=1 not set — electric fence check not run"
    else
        script = """
        using PureBLAS
        const P = PureBLAS
        function released()
            P.@fenced_scope s begin
                A = P.borrow!(s, Float64, 8, 8)
                A[1, 1] = 1.0                       # a LIVE fenced borrow must be perfectly usable...
                A[1, 1] == 1.0 || exit(3)
                A
            end
        end
        h = released()
        println("LIVE-OK"); flush(stdout)
        h[1, 1] = 2.0                               # ...and a released one must fault right here.
        println("NOFAULT"); flush(stdout)
        """
        # PRECOMPILE CACHE: the child must run in the PARENT'S configuration, and for one reason only —
        # the parent's pkgimage is warm BY CONSTRUCTION, since it is the image this very testitem is
        # running out of. Any other configuration is a coin flip on what happens to be on disk. The cache
        # slug keys on BOTH halves, so both have to be mirrored:
        #   * the ENVIRONMENT — `Base.active_project()` is the Pkg.test sandbox under `Pkg.test()` and the
        #     package itself when this file is run directly. `JULIA_LOAD_PATH`/`JULIA_PROJECT` are dropped
        #     so `--project` is what actually decides; `JULIA_DEPOT_PATH` is left alone so the depot, and
        #     hence the cache, is shared.
        #   * `--check-bounds`, which `Pkg.test()` hardcodes to `yes` on 1.12 while a bare `julia` defaults
        #     to `auto`; the two key SEPARATE caches.
        # History, so this is not "improved" back: stage 0 forced the child to the package env AND the
        # default flag, i.e. neither half mirrored, on the theory that the default image was the warm one.
        # It was, on that box, by hand. Review measured a cold one — >150 s and still precompiling when
        # killed — which is the same 6m39 failure one cache key over. Bounds checking does not affect what
        # is under test: the fault comes from a PROT_NONE page, and A[1,1] on an 8x8 is in bounds either way.
        exe = joinpath(Sys.BINDIR, "julia")
        proj = something(Base.active_project(), normpath(joinpath(@__DIR__, "..")))
        cb = Base.JLOptions().check_bounds
        cbflag = cb == 1 ? "--check-bounds=yes" : cb == 2 ? "--check-bounds=no" : "--check-bounds=auto"
        cmd = addenv(
            Cmd([exe, "--startup-file=no", cbflag, "--project=" * proj, "-e", script]),
            "JULIA_LOAD_PATH" => nothing, "JULIA_PROJECT" => nothing,
        )
        out, err = IOBuffer(), IOBuffer()
        p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = err); wait = true)
        so, se = String(take!(out)), String(take!(err))
        @test occursin("LIVE-OK", so)      # the mapping is real and writable while the scope is open
        @test !occursin("NOFAULT", so)     # the store through the released handle did not go through
        @test !success(p)                  # ...because the child died on it
        success(p) && @error "electric fence did not fault" stdout = so stderr = se
    end
end
