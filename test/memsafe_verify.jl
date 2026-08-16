# Guard-page memory-safety regression check for the DIRECT-READ gemm microkernels — StrictMode 0.3.9's
# @assert_memsafe / memsafe_report (issue #15), a PROT_NONE electric-fence harness built for exactly this
# bug class. Run STANDALONE under the package's own project (NOT the ReTestItems suite):
#
#     julia --project=. test/memsafe_verify.jl
#
# It is kept OUT of `Pkg.test()` on purpose: @assert_memsafe's `isolate=true` spawns a fresh julia subprocess
# with `--project=$(Base.active_project())` that must `using Serialization` (StrictMode's arg-marshaling dep) —
# resolvable here (a direct/transitive dep of THIS project) but NOT under Pkg.test's merged temp env, where
# Serialization is only transitive. Running under `--project=.` sidesteps that (and keeps the ~subprocess cost
# out of every dev/CI suite run).
#
# WHY THIS OP: the blocked direct-B masked kernel (`_microkernel_db_masked!`, src/gemm.jl) reads B's columns
# DIRECTLY (no packed padding), so a partial last col-tile (nre < NR) must clamp its read to the last valid
# column (`min(j-1, nre-1)`) or it walks off B's end → OOB (the historical directb-masked-oob flaky segfault).
# n=453 (mod 8 = 5) forces that partial tile in the BLOCKED path (max > _GEMM_UNPACK_MAX). memsafe_report runs
# gemm! in the isolated subprocess on guard-page-backed COPIES of A/B/C: the clamp keeps every read in-bounds
# ⇒ clean. POSITIVE-CONTROL verified (2026-07-17): deleting the clamp makes THIS EXACT shape SIGSEGV at the
# guard page (caught as a violation) — the guard genuinely bites, it is not a vacuous pass.
using PureBLAS, StrictMode
import PureBLAS as P

# Named positional wrappers. The native API takes `trans`/`conj` as KEYWORDS, but memsafe_report
# forwards only positional args to the probed call — so a kwarg variant has to be baked into a
# function. Named (not closures) so the same wrapper also works under `isolate=true`, which needs a
# function reachable by name in a fresh process.
gbmvN!(y, AB, x, m, kl, ku) = P.gbmv!(y, AB, x, m, kl, ku; trans = 'N')
gbmvT!(y, AB, x, m, kl, ku) = P.gbmv!(y, AB, x, m, kl, ku; trans = 'T')
gemvN!(y, A, x) = P.gemv!(y, A, x; trans = 'N')
gemvT!(y, A, x) = P.gemv!(y, A, x; trans = 'T')
geru!(x, y, A) = P.ger!(one(eltype(A)), x, y, A; conj = false)
gerc!(x, y, A) = P.ger!(one(eltype(A)), x, y, A; conj = true)

function main()
    if !(Sys.islinux() || Sys.isapple())
        @info "memsafe_verify skipped — Linux/macOS only (needs mmap/mprotect)"
        return
    end
    fails = 0
    # unpacked direct-read (partial rows/cols) + BLOCKED direct-B masked partial cols (n=453 → nre=5).
    for (m, k, n) in ((12, 8, 5), (13, 7, 11), (64, 64, 453))
        C = zeros(m, n); A = randn(m, k); B = randn(k, n)
        r = memsafe_report(P.gemm!, C, A, B; isolate = true, using_module = :PureBLAS)
        if r.violation === nothing
            println("  clean  gemm! m=$m k=$k n=$n")
        else
            fails += 1; println("  VIOLATION  gemm! m=$m k=$k n=$n : ", r.violation)
        end
    end
    fails == 0 || error("memsafe_verify: $fails direct-read gemm OOB violation(s) — a partial-tile read walked off bounds")
    println("memsafe_verify: all direct-read gemm shapes clean.")

    # ── PHASE 2 (added 2026-08-16): broad op × SHAPE sweep for OOB WRITES ────────────────────────
    #
    # WHY THIS EXISTS. Two OOB writes shipped and were found on the same day (186f63f gbmv, 76ffd06
    # complex ger). Both were the same class — a loop bound or dispatched tile width taken from the
    # WRONG operand — and neither was reachable from the suite's shapes. gbmv's is the sharp case:
    # its two bounds coincide algebraically when m == n, so the buggy code is genuinely in-bounds on
    # EVERY square input. More sizes would never have found it. Only a different SHAPE would.
    #
    # So this phase varies SHAPE, not size: m > n and m < n (they fail differently — m > n overruns
    # the destination, m < n produces the empty-column suffix), odd extents, and every trans/uplo.
    #
    # isolate=false: in-process, catches STORES only, no subprocess. That is the right trade here —
    # the write class is exactly what this phase hunts, and cheapness is what lets it cover many
    # shapes. Read-side OOB keeps the isolate=true treatment in phase 1, where the direct-read
    # kernels live.
    #
    # POSITIVE CONTROL, verified 2026-08-16 — this phase is NOT a vacuous pass. Restoring the buggy
    # bound (`chi = (m-kl)-2W`) makes this sweep fault immediately, inside `_gbmv_t_conv_block!`:
    #
    #     signal 11 (2): Segmentation fault
    #       muladd at SIMD/simdvec.jl:444 [inlined]
    #       _gbmv_t_conv_block! at src/blas2/level2_banded.jl:101
    #
    # READ THE FAILURE MODE CAREFULLY. That is a hard SIGSEGV that kills the process — NOT the tidy
    # "VIOLATION …" line this script prints, and NOT its final `error()`. Two reasons: `isolate=false`
    # intercepts STORES only, and on this shape the x super-window READ runs off the guard page first
    # (the fault is at the `muladd`'s load, not at the y store). So a regression here surfaces as a
    # CRASH with a PureBLAS frame on top — that IS the harness working, not the harness broken. The
    # stack names the offending kernel, which is the diagnostic that matters.
    #
    # HONEST LIMIT, stated because a passing harness invites over-reading: this inherits the very
    # blindness it was written to fix — it can only catch a bug some shape HERE reaches. It is a
    # complement to the correctness suite's shape matrix, not a substitute for widening it.
    println("\nphase 2: OOB-write sweep over shapes (isolate=false, stores only)")
    shapes = 0
    chk(f, args...; nm) = begin
        shapes += 1
        r = memsafe_report(f, args...; isolate = false)
        isnothing(r.violation) || (fails += 1; println("  VIOLATION  ", nm, " : ", r.violation))
    end

    for T in (Float64, Float32, ComplexF64)
        # gbmv — the 186f63f shape class. m>n is what overran `y` (length n on trans='T'); m<n adds
        # the empty-column suffix. Odd n hits the single-column remainder of any paired column loop.
        for (kl, ku) in ((15, 16), (31, 32)), (m, n) in ((40, 96), (96, 40), (40, 97), (97, 40), (41, 41))
            AB = randn(T, kl + ku + 1, n)
            chk(gbmvN!, randn(T, m), AB, randn(T, n), m, kl, ku; nm = "gbmv!N $T $(m)x$n kl=$kl ku=$ku")
            chk(gbmvT!, randn(T, n), AB, randn(T, m), m, kl, ku; nm = "gbmv!T $T $(m)x$n kl=$kl ku=$ku")
        end
        # ger — the 76ffd06 shape class: rectangular, and n not a multiple of any panel width. The
        # 1024x1027 cell is DRAM-resident on purpose: the panel driver only runs past L3.
        for (m, n) in ((1024, 1027), (97, 40), (40, 97))
            chk(geru!, randn(T, m), randn(T, n), randn(T, m, n); nm = "geru! $T $(m)x$n")
            T <: Complex && chk(gerc!, randn(T, m), randn(T, n), randn(T, m, n); nm = "gerc! $T $(m)x$n")
        end
        # gemv — rectangular + odd, both trans: masked-tail territory.
        for (m, n) in ((97, 40), (40, 97), (41, 41))
            A = randn(T, m, n)
            chk(gemvN!, randn(T, m), A, randn(T, n); nm = "gemv!N $T $(m)x$n")
            chk(gemvT!, randn(T, n), A, randn(T, m); nm = "gemv!T $T $(m)x$n")
        end
    end
    println("  swept $shapes shape(s)")
    fails == 0 || error("memsafe_verify: $fails OOB violation(s)")
    return println("memsafe_verify: all clean.")
end
main()
