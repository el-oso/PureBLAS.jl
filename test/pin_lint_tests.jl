# Enforces req#8b's pinning obligation in the suite: every Measure-tier knob (a preference whose default
# is an on-host `OncePerProcess` benchmark) MUST be pinned in juliac/build.jl, so the trim/.so build
# compiles the benchmark out. The scanner lives in pin_lint.jl (run standalone: `julia test/pin_lint.jl`).
#
# Why a gate: a missing pin is otherwise caught only by ACCIDENT. The axpy knobs were caught because
# their measure body contained a non-concrete `Val(u)` that trim rejected; `sytrf_cmult`'s candidate loop
# is over plain integers, so trim had no complaint and the unpinned benchmark would have shipped into
# libpureblas.so via the @ccallable `sytrf_64_`. Nothing announced it — this does.
@testitem "pin lint: every Measure-tier knob is pinned for trim" begin
    include(joinpath(@__DIR__, "pin_lint.jl"))    # defines pin_scan; the CLI guard skips execution
    v = pin_scan()
    isempty(v) || @error "pin lint: Measure-tier knob(s) with no juliac/build.jl pin. Add a \
        `set_preferences!(PUREBLAS_UUID, \"<name>\" => <value>; force = true)` (with save/restore in the \
        `finally` block), or annotate the pref `# pin-ok: <reason>` if it is a Pin/Derive-tier user \
        override with no benchmark behind it." violations = v
    @test isempty(v)
end
