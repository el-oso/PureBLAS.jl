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
    return println("memsafe_verify: all direct-read gemm shapes clean.")
end
main()
