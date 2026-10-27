# Per-machine harness facts, copied to `bench/fleet.local.sh` on each box and NOT committed.
#
# WHY THESE LEFT THE SCRIPTS. They used to be `case "$(hostname)" in` tables, which put every machine's
# name in a public repository and keyed the values on a string only this fleet has. A box stating its
# OWN facts needs no table and no names: each file below describes one machine, and the scripts read
# whichever one they find.
#
# The values are hardware topology, not tuning. Nothing here is a `Preferences` pin and nothing here
# selects a kernel — `bench/probe.sh` and `bench/fleet_refresh.sh` use them to place threads and to pick
# the core whose achieved clock is verified. Tuning stays in `LocalPreferences.toml`, which this file
# must not duplicate.
#
# A SCRIPT THAT NEEDS A VALUE AND DOES NOT FIND IT REFUSES, rather than running unpinned. An unpinned
# threaded measurement on this fleet is not noisier, it is bimodal: measured on Zen4, a gemm
# self-speedup reads 2.6x unpinned against 4.7x pinned, and the median lands in the slow mode and looks
# stable. Silence would turn a missing config into a quiet factor-of-two error.

# CPU list for `taskset -c`, ONE CPU PER PHYSICAL CORE. Read `lscpu -p=CPU,CORE` and take one CPU per
# distinct core — the numbering differs per machine, and a sibling pair shares its FMA units, so a mask
# that doubles up measures contention rather than parallelism. Six cores on a six-core part:
#   siblings adjacent  (CPU 0,1 = core 0)   ->  0,2,4,6,8,10
#   siblings in the second half (CPU 6 = core 0 on a 6-core part)  ->  0,1,2,3,4,5
PB_MASK=0,2,4,6,8,10

# The core whose achieved-under-load clock `bench/fleet_freqlock.sh verify` samples. It must be a core
# the benchmark actually runs on — i.e. one of `PB_MASK` — or the check reports a clock from an idle
# core while the working cores throttle.
PB_VCORE=2

# SSH aliases of the OTHER machines, for `bench/fleet_sync.sh all`. Config aliases only, never
# `host.local` and never a raw address. Leave empty on a single-machine checkout.
PB_BOXES=""
