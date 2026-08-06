#!/usr/bin/env julia
# Build libpureblas.<dlext> via juliac --trim (Julia ≥ 1.12, experimental). The resulting shared
# library exports the BLAS-1 ILP64 symbols (daxpy_64_, …) that libblastrampoline forwards to —
# see PureBLAS.activate(). Run: `julia juliac/build.jl`.

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUTDIR = joinpath(@__DIR__, "build")
mkpath(OUTDIR)

const JULIAC = normpath(joinpath(Sys.BINDIR, "..", "share", "julia", "juliac", "juliac.jl"))
isfile(JULIAC) || error("juliac.jl not found at $JULIAC — needs Julia ≥ 1.12")

const DLEXT = Sys.iswindows() ? "dll" : (Sys.isapple() ? "dylib" : "so")
const OUT = joinpath(OUTDIR, "libpureblas." * DLEXT)
const ENTRY = joinpath(@__DIR__, "entry.jl")

cmd = `$(Base.julia_cmd()) --startup-file=no --project=$ROOT $JULIAC
       --output-lib $OUT --experimental --trim=safe --compile-ccallable --verbose $ENTRY`

# Pin `ger_panel_np` for the trim build ONLY. ger!'s OncePerProcess auto-calibration branch (a runtime
# benchmark) is not trim-safe; setting the preference makes the `@static if` compile that branch out. The
# shipped Project.toml deliberately omits this pref so Mode-2 (in-Julia) users auto-calibrate per µarch — so
# we set it here via Preferences (writing ROOT's gitignored LocalPreferences.toml), scoped to the build, then
# restore the prior state. Value is arbitrary among 1/2/4/8; the .so uses a fixed stream count regardless.
Base.set_active_project(joinpath(ROOT, "Project.toml"))   # so set_preferences! targets ROOT's LocalPreferences
using Preferences
const PUREBLAS_UUID = Base.UUID("cc9e14db-574f-4602-bf53-1167cc4b26d2")
const _prev_ger = load_preference(PUREBLAS_UUID, "ger_panel_np")
# Pin `brd_nb` too: gebrd's OncePerProcess panel-width auto-tune is a runtime benchmark (not trim-safe);
# the pref compiles the `@static if` measure branch out. The .so can't auto-tune per host, so it takes the
# measured Zen4 optimum; Mode-2 (in-Julia) users still measure per box because Project.toml omits it.
const _prev_brd = load_preference(PUREBLAS_UUID, "brd_nb")
const _prev_pbt = load_preference(PUREBLAS_UUID, "pbtrf_cross_kd")
# Pin `pbtrf_nb` for the same reason: the band factor's panel width is a Measure-tier OncePerProcess
# auto-tune, and a runtime benchmark is not trim-safe. 40 is the Zen4 F64 optimum (= 5·W, the derived
# bracket centre); Mode-2 users still measure per box because Project.toml omits the pref.
const _prev_pbnb = load_preference(PUREBLAS_UUID, "pbtrf_nb")
# And `pbtrf_u_native_kd`: uplo='U' has two blocked kernels (re-pack-onto-L vs native upper) whose
# winner inverts with the bandwidth, and the switch point is a Measure-tier OncePerProcess race —
# again not trim-safe. 256 is the Zen4 F64 measurement (repack 1.10 at kd=192, 0.845 at 256; native
# 1.055 there); Mode-2 users still measure per box because Project.toml omits the pref.
const _prev_pbu = load_preference(PUREBLAS_UUID, "pbtrf_u_native_kd")
# And `pbtrf_nb_small`: the band panel width has two measured regimes (wide-band `pbtrf_nb`, and this
# one for kd below it, where clamping to kd would collapse the in-band panel). Both are runtime
# benchmarks. 8 is the Zen4 F64 optimum (= W).
const _prev_pbns = load_preference(PUREBLAS_UUID, "pbtrf_nb_small")
# And `pptrf_spr_min`: the packed lower path's spr!-vs-inline cutoff is a Measure-tier
# OncePerProcess race. 8 is the Zen4 F64 optimum (= W).
const _prev_ppsm = load_preference(PUREBLAS_UUID, "pptrf_spr_min")
# And `gbtrf_cross`: the blocked-vs-unblocked band crossover is a Measure-tier OncePerProcess race and
# a runtime benchmark is not trim-safe. 16 is the Zen4 value; Zen3 measures higher (blocking loses at
# kl=16 there), so a trim build for AVX2 should re-pin this.
const _prev_gbcr = load_preference(PUREBLAS_UUID, "gbtrf_cross")
# And `gbtrf_nb`: the banded-LU panel width is a Measure-tier OncePerProcess race, and a runtime
# benchmark is not trim-safe. 16 is the Zen4/Zen3 F64 optimum (both boxes agree in absolute terms).
const _prev_gbnb = load_preference(PUREBLAS_UUID, "gbtrf_nb")
# And `gemvt_perscan`: gemv-T routes its columns per-column instead of NC-blocked inside a derived
# cache window, but ONLY on boxes where that wins — the sign is µarch-dependent (Zen4 up to 1.35x,
# Zen3 never; the Derive-only version regressed Zen3 gemvT to 0.69 vs AOCL). The decision is a
# OncePerProcess runtime benchmark, so it is not trim-safe; pin it to compile the measure branch out.
# `false` = always blocked = the conservative arm that is correct on every box measured so far.
# And `zaxpy_narrow`: the complex-axpy width decision is a Measure-tier OncePerProcess benchmark, so it
# allocates a probe buffer and is not trim-safe. `true` = the narrow arm, measured best on Zen4 and the
# arm that closes the zaxpy gate cells; a trim build for a true-512-bit datapath should re-pin it.
# And the two REAL-axpy shape knobs, `axpy_unroll` (L1..L3 band) and `axpy_dram` (past L3). Both are
# Measure-tier OncePerProcess benchmarks that allocate a probe buffer, so both must be pinned or
# `daxpy_64_` fails trim checking — which is exactly what it did. They are SEPARATE knobs measured in
# separate regimes and must be pinned separately; collapsing them onto one value is what the
# `_axpy_dram` fallback used to do. Codes: u = interleaved(u); 100+u = phase(u, full width);
# 200+u = phase(u, narrow/256-bit). 4 = interleaved-4, the arm that is correct on every box measured.
# And `sytrf_cmult`: the complex Bunch-Kaufman block multiplier is a Measure-tier OncePerProcess
# benchmark that allocates a 1024x1024 probe matrix. `sytrf_64_` is @ccallable, so leaving it unpinned
# shipped that benchmark into the .so — and unlike the axpy knobs it was never caught by trim, because
# its candidate loop is over plain integers (1,2,3,4) with no `Val`, so it is type-stable and trim has
# no complaint. Nothing announced the gap; test/pin_lint.jl now does.
# Value 2, not the 1 measured on Zen4: the measure's OWN failure path returns 2 with the comment
# "the safer default: 1 MISSES on Zen3", and a trimmed .so is one binary that must be safe on any host.
# NOTE this pins the MULTIPLIER, not `sytrf_nb` — pinning the latter takes a branch whose value is flat
# and would discard the derived `_sytrf_nb_shape(n)` scaling entirely.
# And `potrf_upper_direct_max`: the tiny-UPPER cutoff (factor in place vs transpose onto the
# vectorised lower kernel) is a Measure-tier OncePerProcess per eltype. `potrf_64_` is @ccallable, so
# unpinned it would ship an on-host benchmark into the .so — the first call would time two Cholesky
# variants and pick from momentary machine state. Value 12: measured crossover is ≈14-16 real and ≈22
# complex (bench/probes/potrf_upper_cross.jl), and a trimmed .so is ONE binary that must be safe on any
# host, so pin below the smallest measured crossover — a box whose vector kernel is relatively faster
# moves the crossover DOWN, and being early costs only the tiny-n win, while being late regresses n=16+.
const _prev_pud = load_preference(PUREBLAS_UUID, "potrf_upper_direct_max")
const _prev_sycm = load_preference(PUREBLAS_UUID, "sytrf_cmult")
const _prev_axu = load_preference(PUREBLAS_UUID, "axpy_unroll")
const _prev_axd = load_preference(PUREBLAS_UUID, "axpy_dram")
const _prev_zaxn = load_preference(PUREBLAS_UUID, "zaxpy_narrow")
const _prev_gvtp = load_preference(PUREBLAS_UUID, "gemvt_perscan")
set_preferences!(PUREBLAS_UUID, "ger_panel_np" => 4; force = true)
set_preferences!(PUREBLAS_UUID, "gemvt_perscan" => false; force = true)  # gemv-T column route (Measure tier)
set_preferences!(PUREBLAS_UUID, "zaxpy_narrow" => true; force = true)    # complex-axpy width (Measure tier)
set_preferences!(PUREBLAS_UUID, "sytrf_cmult" => 2; force = true)       # complex BK block multiplier (Measure tier)
set_preferences!(PUREBLAS_UUID, "potrf_upper_direct_max" => 12; force = true)  # tiny-upper potrf cutoff (Measure tier)
set_preferences!(PUREBLAS_UUID, "axpy_unroll" => 4; force = true)        # real-axpy band shape (Measure tier)
set_preferences!(PUREBLAS_UUID, "axpy_dram" => 4; force = true)          # real-axpy DRAM shape (Measure tier)
set_preferences!(PUREBLAS_UUID, "brd_nb" => 8; force = true)
set_preferences!(PUREBLAS_UUID, "pbtrf_cross_kd" => 32; force = true)   # blocked-vs-unblocked band crossover (Measure tier)
set_preferences!(PUREBLAS_UUID, "pbtrf_nb" => 40; force = true)         # band panel width (Measure tier)
set_preferences!(PUREBLAS_UUID, "pbtrf_u_native_kd" => 256; force = true)  # upper repack-vs-native (Measure tier)
set_preferences!(PUREBLAS_UUID, "pbtrf_nb_small" => 8; force = true)      # narrow-band panel width (Measure tier)
set_preferences!(PUREBLAS_UUID, "pptrf_spr_min" => 8; force = true)       # packed lower spr-vs-inline cutoff (Measure tier)
set_preferences!(PUREBLAS_UUID, "gbtrf_cross" => 16; force = true)         # banded-LU blocking floor (Measure tier)

@info "PureBLAS: building trimmed library" OUT
try
    run(cmd)
finally
    if _prev_ger === nothing
        delete_preferences!(PUREBLAS_UUID, "ger_panel_np"; force = true)
    else
        set_preferences!(PUREBLAS_UUID, "ger_panel_np" => _prev_ger; force = true)
    end
    if _prev_gvtp === nothing
        delete_preferences!(PUREBLAS_UUID, "gemvt_perscan"; force = true)
    else
        set_preferences!(PUREBLAS_UUID, "gemvt_perscan" => _prev_gvtp; force = true)
    end
    if _prev_zaxn === nothing
        delete_preferences!(PUREBLAS_UUID, "zaxpy_narrow"; force = true)
    else
        set_preferences!(PUREBLAS_UUID, "zaxpy_narrow" => _prev_zaxn; force = true)
    end
    if _prev_sycm === nothing
        delete_preferences!(PUREBLAS_UUID, "sytrf_cmult"; force = true)
    else
        set_preferences!(PUREBLAS_UUID, "sytrf_cmult" => _prev_sycm; force = true)
    end
    if _prev_pud === nothing
        delete_preferences!(PUREBLAS_UUID, "potrf_upper_direct_max"; force = true)
    else
        set_preferences!(PUREBLAS_UUID, "potrf_upper_direct_max" => _prev_pud; force = true)
    end
    if _prev_axu === nothing
        delete_preferences!(PUREBLAS_UUID, "axpy_unroll"; force = true)
    else
        set_preferences!(PUREBLAS_UUID, "axpy_unroll" => _prev_axu; force = true)
    end
    if _prev_axd === nothing
        delete_preferences!(PUREBLAS_UUID, "axpy_dram"; force = true)
    else
        set_preferences!(PUREBLAS_UUID, "axpy_dram" => _prev_axd; force = true)
    end
    if _prev_brd === nothing
        delete_preferences!(PUREBLAS_UUID, "brd_nb"; force = true)
    else
        set_preferences!(PUREBLAS_UUID, "brd_nb" => _prev_brd; force = true)
    end
    if _prev_pbt === nothing
        delete_preferences!(PUREBLAS_UUID, "pbtrf_cross_kd"; force = true)
    else
        set_preferences!(PUREBLAS_UUID, "pbtrf_cross_kd" => _prev_pbt; force = true)
    end
    if _prev_pbnb === nothing
        delete_preferences!(PUREBLAS_UUID, "pbtrf_nb"; force = true)
    else
        set_preferences!(PUREBLAS_UUID, "pbtrf_nb" => _prev_pbnb; force = true)
    end
    if _prev_pbu === nothing
        delete_preferences!(PUREBLAS_UUID, "pbtrf_u_native_kd"; force = true)
    else
        set_preferences!(PUREBLAS_UUID, "pbtrf_u_native_kd" => _prev_pbu; force = true)
    end
    if _prev_pbns === nothing
        delete_preferences!(PUREBLAS_UUID, "pbtrf_nb_small"; force = true)
    else
        set_preferences!(PUREBLAS_UUID, "pbtrf_nb_small" => _prev_pbns; force = true)
    end
    if _prev_gbcr === nothing
    delete_preferences!(PUREBLAS_UUID, "gbtrf_cross"; force = true)
else
    set_preferences!(PUREBLAS_UUID, "gbtrf_cross" => _prev_gbcr; force = true)
end
if _prev_gbnb === nothing
    delete_preferences!(PUREBLAS_UUID, "gbtrf_nb"; force = true)
else
    set_preferences!(PUREBLAS_UUID, "gbtrf_nb" => _prev_gbnb; force = true)
end
if _prev_ppsm === nothing
        delete_preferences!(PUREBLAS_UUID, "pptrf_spr_min"; force = true)
    else
        set_preferences!(PUREBLAS_UUID, "pptrf_spr_min" => _prev_ppsm; force = true)
    end
end
# Strip DWARF debug info: juliac emits it (`-g1` default) and it dominates the file — ~110 MB of ~154 MB
# is `.debug_*` sections, dead weight in a distributed drop-in. `--strip-debug` keeps the full symbol
# table AND all 492 dynamic `_64_` exports (verified present after strip), so the library is functionally
# identical; only the debug sections go (154 MB → ~48 MB). Best-effort: skip silently if `strip` is absent.
if isfile(OUT) && Sys.which("strip") !== nothing
    presz = filesize(OUT)
    try
        run(`strip --strip-debug $OUT`)
        @info "PureBLAS: stripped debug info" before_MB = round(presz / 2^20; digits = 1) after_MB = round(filesize(OUT) / 2^20; digits = 1)
    catch e
        @warn "PureBLAS: strip failed (keeping debug info)" exception = e
    end
end
@info "PureBLAS: built" OUT filesize_bytes = (isfile(OUT) ? filesize(OUT) : 0)
