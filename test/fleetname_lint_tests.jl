# Keeps fleet machine names out of a PUBLIC repository. They have been removed by hand twice — the
# second time 162 occurrences across 49 files — so this is enforced rather than remembered. The scanner
# is fleetname_lint.jl (run standalone: `julia test/fleetname_lint.jl`).
#
# IT DOES NOT STORE THE NAMES, which would be the leak itself: it reads them from `hostname` and from
# `PB_BOXES` in the gitignored `bench/fleet.local.sh`. A box that knows its own name can check for it.
#
# SO THE NAME HALF IS INERT IN CI, and that is stated rather than hidden. CI has no fleet and cannot
# know the names; a check reporting success from zero knowledge is worse than no check. The SHAPE half
# needs no names and runs everywhere — it catches the `case "$(hostname)"` construct that put the names
# in both times, which is the mechanism rather than the symptom.
@testitem "fleet-name lint: no machine names, no hostname-keyed dispatch" begin
    include(joinpath(@__DIR__, "fleetname_lint.jl"))
    r = fleetname_lint()
    isempty(r.names) || @error "Fleet machine name(s) in tracked files. This repository is PUBLIC. \
        Name the MICROARCHITECTURE the measurement came from (Zen3/Zen4/Zen5/arm64) — that is what the \
        sentence is using the name to mean, and it is more useful to a reader on an unknown part." names = r.names
    isempty(r.shapes) || @error "Hostname-keyed per-box dispatch in tracked files. A box states its own \
        values in the gitignored bench/fleet.local.sh; see bench/fleet.local.example.sh. Renaming the \
        case labels is NOT a fix — it leaves a table that matches no hostname and silently falls through \
        to the default, which is how fleet_refresh_g.sh broke." shapes = r.shapes
    @test isempty(r.names)
    @test isempty(r.shapes)
    # Not an assertion: the name half cannot run without a local name list, and a reader of a green CI
    # log should know which half actually executed.
    iszero(r.names_checked) && @info "fleet-name lint: name scan did not run (no local name list); \
        shape scan did"
end
