# AUTHORITATIVE trim-safety check: actually run `juliac --trim` to build libpureblas.so, plus a C-host
# LBT smoke test (ctest.c). This is the ONLY check that catches whole-program trim regressions that
# TrimCheck `@validate` (trim_tests.jl) cannot — @validate infers each ccallable signature in isolation
# (kernels inline, default-arg trampolines resolve), so it is more optimistic than juliac's whole-program
# compile. A @generated kernel with default args (`_uker_cmplx!`, 2026-07-13) once passed @validate while
# the real build failed with 36 `unresolved invoke ::Any` errors and shipped a stale .so for days.
#
# GATED by PUREBLAS_JULIAC_BUILD=1 because a cold `--trim` build is slow (~13 min: it precompiles PureBLAS
# against a trimmed base image). Run it in a dedicated CI job / before a release, not on every Pkg.test().
#   PUREBLAS_JULIAC_BUILD=1 julia --project=test -e 'using ReTestItems, PureBLAS; runtests(PureBLAS; name="juliac")'

@testitem "juliac --trim build + C-host LBT (authoritative; gated PUREBLAS_JULIAC_BUILD=1)" tags = [:juliac] begin
    using PureBLAS
    root = pkgdir(PureBLAS)
    juliac = normpath(joinpath(Sys.BINDIR, "..", "share", "julia", "juliac", "juliac.jl"))
    dlext = Sys.iswindows() ? "dll" : (Sys.isapple() ? "dylib" : "so")
    so = joinpath(root, "juliac", "build", "libpureblas." * dlext)

    if get(ENV, "PUREBLAS_JULIAC_BUILD", "0") != "1"
        @info "juliac build check SKIPPED — set PUREBLAS_JULIAC_BUILD=1 to run (slow full --trim build)"
        @test_skip true
    elseif !isfile(juliac)
        @info "juliac.jl not found (needs Julia ≥ 1.12) — skipping build check" juliac
        @test_skip true
    else
        # Build the .so. A nonzero exit means a trim-verifier error (the regression class @validate misses)
        # or a link failure — either way the LBT drop-in is broken. Clear JULIA_LOAD_PATH so the juliac
        # subprocess sees the DEFAULT path (incl. @stdlib → LazyArtifacts): a ReTestItems worker restricts
        # it, which would otherwise fail the build with "LazyArtifacts not found" before it even verifies.
        buildjl = joinpath(root, "juliac", "build.jl")
        cmd = addenv(
            `$(Base.julia_cmd()) --startup-file=no --project=$root $buildjl`,
            "JULIA_LOAD_PATH" => nothing, "JULIA_PROJECT" => nothing
        )
        proc = run(pipeline(cmd; stdout = stderr, stderr = stderr); wait = false)
        wait(proc)
        @test success(proc)
        @test isfile(so)

        # C-host functional smoke: compile + run ctest.c against the fresh .so (needs a C compiler). This
        # confirms the trimmed lib self-inits and computes correctly across the Fortran C-ABI (incl. dgesvd).
        cc = Sys.which("cc")
        if isnothing(cc)
            @info "no C compiler (cc) on PATH — skipping ctest.c host smoke"
        else
            exe = joinpath(mktempdir(), "ctest")
            ctest = joinpath(root, "juliac", "ctest.c")
            run(`$cc -O2 $ctest -o $exe -ldl -lm`)
            out = cd(() -> read(`$exe`, String), root)   # ctest.c dlopens juliac/build/... (relative)
            @info "ctest.c output" out
            @test occursin("daxpy: 12.0 24.0 36.0 48.0", out)
            @test occursin("dgesvd: info=0", out)
            @test occursin("dgesvd n=160: info=0", out)
            # both SVD reconstructions must hit machine precision through the C-ABI
            m = match(r"dgesvd n=160:.*recon\|err\|=([0-9.eE+-]+)\s+ortho\|err\|=([0-9.eE+-]+)", out)
            @test !isnothing(m)
            @test parse(Float64, m.captures[1]) < 1.0e-9
            @test parse(Float64, m.captures[2]) < 1.0e-9
        end
    end
end
