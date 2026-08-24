# Benchmark provenance

The caches behind `bench/gen_table*.md`, `docs/src/assets/perf_*.svg` and the generated tables in `docs/src/coverage.md`. Both reference views (OpenBLAS, AOCL) are rendered from this one cache set. Methodology: `docs/src/methodology.md`.

| µarch | host | CPU | commit | measured |
|---|---|---|---|---|
| Zen3 · AVX2 | `galen` | AMD Ryzen 9 5900X 12-Core Processor | `a538d84` | 2026-08-24T17:40 |
| Zen4 · AVX-512 | `wintermute` | AMD Ryzen 5 7640U w/ Radeon 760M Graphics | `a538d84` | 2026-08-24T17:33 |
| Zen5 · AVX-512 | `neuromancer` | AMD Ryzen AI 5 340 w/ Radeon 840M | `a538d84` | 2026-08-24T17:48 |
