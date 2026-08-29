# Benchmark provenance

The caches behind `bench/gen_table*.md`, `docs/src/assets/perf_*.svg` and the generated tables in `docs/src/coverage.md`. Both reference views (OpenBLAS, AOCL) are rendered from this one cache set. Methodology: `docs/src/methodology.md`.

| µarch | host | CPU | commit | measured |
|---|---|---|---|---|
| Zen3 · AVX2 | `galen` | AMD Ryzen 9 5900X 12-Core Processor | `9212c41` | 2026-08-28T22:47 |
| Zen4 · AVX-512 | `wintermute` | AMD Ryzen 5 7640U w/ Radeon 760M Graphics | `9a5bfcd` | 2026-08-29T11:01 |
| Zen5 · AVX-512 | `neuromancer` | AMD Ryzen AI 5 340 w/ Radeon 840M | `89550f9` | 2026-08-29T10:08 |
