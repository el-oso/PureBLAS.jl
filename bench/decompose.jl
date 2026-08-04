# PHASE DECOMPOSITION of blocked factorizations — "getrf is at 0.90" names the routine, not the phase.
# This answers WHICH of panel/laswp/trsm/gemm (or swap/gemv/syrk) the time actually goes to, because the
# naive guess is always "the BLAS-3 dominates" and the recorded pstrf campaign found the opposite: the
# pivot SWAP was 47% of the factorization at n=2048, and fixing it — not the syrk — gave 3.3×.
#
# HOW, given that Chairmarks is the only thing allowed to touch a clock:
#
#   * ABSOLUTE time  — `@be` on the whole routine, reduced by `tstat` (median). Chairmarks, as always.
#   * ATTRIBUTION    — stdlib `Profile` instruction-pointer sampling of the routine RUNNING IN SITU.
#     The profiler counts where the instruction pointer was; it takes no timestamps in our code, and
#     there is no estimator to choose (the whole class of min/mean/hand-loop errors this repo's lint
#     exists to prevent cannot occur — there are no timings to reduce, only sample counts).
#     Per-phase time = (phase's share of in-routine samples) × (the `@be` median).
#
# WHY NOT the "reconstruct each phase as a callable and `@be` it" design (the deleted
# bench/zgetrf_decomp.jl, minus its raw clocks): measured prior art IN THIS REPO says phase isolation
# cannot be trusted for exactly this question.
#   * src/lapack/pstrf.jl (rowcache knob, "TIER HONESTY" comment): an isolated-phase harness disagreed
#     with the gate TWICE, in the same direction, because the harness's buffer stayed L1-warm across
#     reps — "a harness whose warm-up regime differs from the live one on the very property being
#     measured cannot be trusted." A reconstructed phase inherits a setup()-warmed cache instead of
#     whatever the previous phase actually left behind; that is the same trap, per phase, by design.
#   * The sum-of-phases-vs-whole check bounds only the TOTAL distortion. Per-phase errors of opposite
#     sign cancel in the sum (an over-timed cold gemm against an under-timed warm trsm reads as a clean
#     1.00), so the check can pass while the attribution — the only thing the tool exists to produce —
#     is wrong. A validation that passes on a wrong answer is not a validation.
#   * Reconstruction must duplicate the blocked loop (step schedule, shapes, pivot replay — pivots are
#     data-dependent, so "the state at step k" requires factoring up to step k). That copy silently
#     diverges the next time the routine changes; today's pstrf (fused scalar loops, deferred swap
#     batches) no longer even decomposes into four clean calls.
# IP sampling has none of these: the real routine runs on the real data with the real cache state, and
# the phase structure is read off the call stack instead of being re-implemented.
#
# SELF-VALIDATION (so a distorted result is visible rather than silent), reported with every run:
#   * `consistency` = (in-routine samples × delay ÷ reps) / (`@be` median). The profiler's implied
#     per-call time against Chairmarks' — an independent cross-check of the two instruments. Expected
#     ≈1.0 (profiling overhead and regime differences push it a few % high or low); outside
#     [0.80, 1.25] the attribution is suspect and `phaseshow` says so instead of printing quietly.
#   * per-phase ±se — binomial sqrt(p(1-p)/N), so a share smaller than its own resolution is visibly so.
#   * `truncated` — the sample buffer filled before the planned reps ran (partial evidence, flagged).
#
# DELAY — the sampler tops out near 400 wakeups/s, so ASK FOR LESS THAN THAT. The profiler is a thread
# that sleeps `delay` and then signals the workload; the achieved rate saturates at a few hundred Hz
# (CFS wakeup granularity), so any `delay` far below ~1e-2 silently delivers a small fraction of the
# requested samples, the implied time collapses, and the consistency gate fires.
# MEASURED (wintermute, decomp_delaycal.jl, getrf n=1024, 2026-08-04) — consistency vs requested delay:
#     taskset -c 4,5  delay 1e-4 → 0.043   1e-3 → 0.369   3e-3 → 0.535   1e-2 → 0.909
#     taskset -c 4,6  delay 1e-4 → 0.039   1e-3 → 0.321   3e-3 → 0.476   1e-2 → 0.881
# The achieved rate is ~350–480 samples/s in EVERY cell: consistency is just (achieved ÷ requested), so
# it approaches 1 only once the request drops under the cap. Hence the 1e-2 default.
# GIVING THE SAMPLER ITS OWN CORE DOES NOT HELP — measured, not assumed. cpu4/cpu5 are SMT siblings
# here (`thread_siblings_list` = 4-5) and cpu6 is a distinct physical core; both columns above are the
# same within noise, and a single core (`taskset -c 4`) reads the same. An earlier version of this
# header claimed `-c 4,5` fixed it at 1e-4 (1.044); that does not reproduce — the cap is the scheduler,
# not core contention. Pin however the gate pins.
#
# LIMITS (known, accepted):
#   * Granularity is the call stack: a phase must be visible as a FUNCTION FRAME under the root.
#     @inline helpers still resolve by name (verified on this Julia 1.12.6 — inlined frames carry their
#     original name via debug info); code truly fused into the root's own body lands in "(self)",
#     which for pstrf is exactly the Schur-diagonal/pivot-search scalar loops — a meaningful bucket,
#     labeled as such by the caller.
#   * Attribution is by the OUTERMOST phase frame below the root, so BLAS-3 called from inside a panel
#     (e.g. _getf2_blocked!'s cross-half gemm) counts as PANEL — the phase you would have to fix.
#   * Fractions are of samples on the thread running `root`; sleeping scheduler/GC threads are sampled
#     too but never contain the root frame, so they fall out of the root gate automatically.

using Profile

"""
    phasefrac(f, setup; root, phases, want=12_000, delay=1e-4, ...) -> NamedTuple

Attribute the runtime of `f(setup())` across `phases`, in situ, via Profile IP sampling; anchor the
absolute scale with a Chairmarks `@be` median. `root` is the function symbol that bounds the routine
(samples outside it — `setup`, driver, other threads — are excluded). `phases` is
`name => [function symbols]`; a sample is attributed to the first phase frame found walking DOWN the
stack from `root` (outermost wins — see header). Samples inside `root` matching no phase go to `self`.

`want` is the target number of in-root samples: se of a share p is √(p(1-p)/want), so the default
4k resolves a 47%-sized share to ±0.8pp. Reps are derived as `want·delay ÷ median` — no clock read.
`delay` defaults to 1e-2 because the sampler saturates near 400 Hz (see the header's measured sweep);
asking for more only lowers `consistency`. At 4k samples / 100 Hz a case costs ~40 s of profiled run.
"""
function phasefrac(
        f, setup; root::Symbol, phases::Vector{<:Pair{String, Vector{Symbol}}},
        want::Int = 4_000, delay::Float64 = 1.0e-2, bufwords::Int = 5 * 10^7,
        samples::Int = 32, seconds::Float64 = 1.0
    )
    f(setup())                                            # compile before either instrument looks
    # ABSOLUTE anchor: Chairmarks, evals=1 (fresh ctx per sample, the gate's regime), median via tstat.
    b = @be setup() f evals = 1 samples = samples seconds = seconds
    med = tstat(Float64[s.time for s in b.samples])
    # Rep count DERIVED from the median (want·delay seconds of in-root work), not from watching a clock.
    reps = clamp(ceil(Int, want * delay / med), 8, 200_000)
    Profile.init(n = bufwords, delay = delay)
    Profile.clear()
    done = 0
    Profile.@profile for _ in 1:reps
        f(setup())
        done += 1
        # Buffer nearly full → stop with what we have; `truncated` is reported, never hidden.
        Profile.len_data() > (9 * bufwords) ÷ 10 && break
    end
    data = Profile.fetch()
    Profile.has_meta(data) && (data = Profile.strip_meta(data))
    lidict = Profile.getdict(data)
    counts, selfc, outside, inroot = _attribute(data, lidict, root, phases)
    # Cross-check of the two instruments: profiler-implied per-call in-root time vs the @be median.
    implied = inroot * delay / done
    out = [
        (name = phases[i][1], frac = counts[i] / max(inroot, 1),
         se = sqrt(counts[i] / max(inroot, 1) * (1 - counts[i] / max(inroot, 1)) / max(inroot, 1)),
         secs = counts[i] / max(inroot, 1) * med, nsamp = counts[i])
        for i in eachindex(phases)
    ]
    return (
        phases = out, med = med, nbe = length(b.samples), inroot = inroot,
        selffrac = selfc / max(inroot, 1), outside = outside, reps = done,
        consistency = implied / med, truncated = done < reps, root = root,
    )
end

# Walk the sample buffer: blocks of IPs (innermost-first) separated by zeros. For each block, find the
# root frame, then descend from just inside it toward the leaf; the first frame whose name belongs to a
# phase claims the sample. Inlined frames appear with their original names (lidict maps one IP to the
# whole inline chain), so @inline phase helpers still attribute correctly.
function _attribute(data, lidict, root::Symbol, phases)
    nameof = Dict{Symbol, Int}()
    for (i, (_, syms)) in enumerate(phases), s in syms
        nameof[s] = i
    end
    counts = zeros(Int, length(phases)); selfc = 0; outside = 0; inroot = 0
    names = Symbol[]                                       # innermost-first, reused per block
    for ip in data
        if ip == 0
            if !isempty(names)
                ri = findlast(==(root), names)
                if isnothing(ri)
                    outside += 1
                else
                    inroot += 1
                    hit = 0
                    for k in (ri - 1):-1:1                 # outermost-below-root first (see header)
                        pi = get(nameof, names[k], 0)
                        pi != 0 && (hit = pi; break)
                    end
                    hit == 0 ? (selfc += 1) : (counts[hit] += 1)
                end
            end
            empty!(names)
        else
            for fr in get(lidict, ip, Base.StackTraces.StackFrame[])
                push!(names, fr.func)
            end
        end
    end
    return counts, selfc, outside, inroot
end

"""Print a `phasefrac` result with its provenance — every number carries estimator, N, and the
consistency cross-check, so a distorted run announces itself."""
function phaseshow(res; label = "", selflabel = "(self)")
    isempty(label) || println(label)
    println("  envelope ", round(res.med * 1e3; digits = 3), " ms ($(ESTIMATOR) of ", res.nbe,
            " @be samples) — attribution over ", res.inroot, " in-root IP samples, ", res.reps, " reps")
    for p in sort(res.phases; by = x -> x.frac, rev = true)
        println("  ", rpad(p.name, 26), lpad(round(100p.frac; digits = 1), 5), "%  ±",
                round(100p.se; digits = 1), "pp   ≈", round(p.secs * 1e3; digits = 2), " ms")
    end
    println("  ", rpad(selflabel, 26), lpad(round(100res.selffrac; digits = 1), 5), "%")
    ok = 0.80 <= res.consistency <= 1.25 && !res.truncated
    println("  consistency (Profile-implied / @be median) = ", round(res.consistency; digits = 3),
            ok ? "  [OK]" : res.truncated ? "  [SUSPECT: sample buffer truncated]" :
            "  [SUSPECT: instruments disagree >25% — a LOW ratio usually means the sampler thread is " *
            "starved (needs its own core: taskset -c 4,5, see decompose.jl header)]")
    return nothing
end
